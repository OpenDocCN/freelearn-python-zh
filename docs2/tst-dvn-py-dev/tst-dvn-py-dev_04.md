# 第四章：使用模拟对象测试交互

在查看`Rule`和`Stock`类之后，现在让我们将注意力转向`Event`类。`Event`类非常简单：接收者可以注册事件，以便在事件发生时收到通知。当事件触发时，所有接收者都会收到事件的通知。

更详细的描述如下：

+   事件类有一个`connect`方法，该方法在事件触发时调用一个方法或函数

+   当调用`fire`方法时，所有注册的回调函数都会用传递给`fire`方法的相同参数被调用

为`connect`方法编写测试相当直接——我们只需检查接收者是否被正确存储。但是，我们如何编写`fire`方法的测试？此方法不会更改任何状态或存储任何我们可以断言的值。此方法的主要责任是调用其他方法。我们如何测试这是否被正确执行？

这就是模拟对象出现的地方。与断言对象*状态*的普通单元测试不同，模拟对象用于测试多个对象之间的*交互*是否按预期发生。

# 手写一个简单的模拟

首先，让我们看看`Event`类的代码，以便我们了解测试需要做什么。以下代码位于源目录中的`event.py`文件中：

```py
class Event:
    """A generic class that provides signal/slot functionality"""

    def __init__(self):
        self.listeners = []

    def connect(self, listener):
        self.listeners.append(listener)

    def fire(self, *args, **kwargs):
        for listener in self.listeners:
            listener(*args, **kwargs)
```

这段代码的工作方式相当简单。希望收到事件通知的类应该调用`connect`方法并传递一个函数。这将注册该函数用于事件。然后，当使用`fire`方法触发事件时，所有注册的函数都会收到事件的通知。以下是如何使用此类的一个概述：

```py
>>> def handle_event(num):
...   print("I got number {0}".format(num))
...
>>> event = Event()
>>> event.connect(handle_event)
>>> event.fire(3)
I got number 3
>>> event.fire(10)
I got number 10

```

如您所见，每次调用`fire`方法时，所有使用`connect`方法注册的函数都会用给定的参数被调用。

那么，我们如何测试`fire`方法？上面的概述提供了一个提示。我们需要做的是创建一个函数，使用`connect`方法注册它，然后验证当调用`fire`方法时该方法是否被通知。以下是一种编写此类测试的方法：

```py
import unittest
from ..event import Event

class EventTest(unittest.TestCase):
    def test_a_listener_is_notified_when_an_event_is_raised(self):
        called = False
        def listener():
            nonlocal called
            called = True

        event = Event()
        event.connect(listener)
        event.fire()
        self.assertTrue(called)
```

将此代码放入测试文件夹中的`test_event.py`文件，并运行测试。测试应该通过。以下是我们所做的工作：

1.  首先，我们创建一个名为`called`的变量并将其设置为`False`。

1.  接下来，我们创建一个虚拟函数。当函数被调用时，它将`called`设置为`True`。

1.  最后，我们将虚拟函数连接到事件并触发事件。

1.  如果在事件触发时成功调用了虚拟函数，则`called`变量将被更改为`True`，我们断言该变量确实是我们预期的。

我们上面创建的虚拟函数是一个模拟的例子。**模拟**简单地说是一个在测试用例中替代真实对象的对象。模拟会记录一些信息，例如它是否被调用，传入了哪些参数等，然后我们可以断言模拟按预期被调用。

谈到参数，我们应该编写一个测试来检查参数是否被正确传入。以下是一个这样的测试：

```py
    def test_a_listener_is_passed_right_parameters(self):
        params = ()
        def listener(*args, **kwargs):
            nonlocal params
            params = (args, kwargs)
        event = Event()
        event.connect(listener)
        event.fire(5, shape="square")
        self.assertEquals(((5, ), {"shape":"square"}), params)
```

这个测试与上一个测试相同，只是它保存了参数，然后在断言中用来验证它们是否正确传入。

到目前为止，我们可以看到我们在设置模拟函数和保存调用信息的方式上出现了一些重复。我们可以将这段代码提取到一个单独的类中，如下所示：

```py
class Mock:
    def __init__(self):
        self.called = False
        self.params = ()

    def __call__(self, *args, **kwargs):
        self.called = True
        self.params = (args, kwargs)
```

一旦我们这样做，我们就可以在我们的测试中使用`Mock`类，如下所示：

```py
class EventTest(unittest.TestCase):
    def test_a_listener_is_notified_when_an_event_is_raised(self):
        listener = Mock()
        event = Event()
        event.connect(listener)
        event.fire()
        self.assertTrue(listener.called)

    def test_a_listener_is_passed_right_parameters(self):
        listener = Mock()
        event = Event()
        event.connect(listener)
        event.fire(5, shape="square")
        self.assertEquals(((5, ), {"shape": "square"}), listener.params)
```

我们刚刚做的是创建了一个简单的模拟类，它相当轻量级，适用于简单用途。然而，我们经常需要更高级的功能，比如模拟一系列调用或检查特定调用的顺序。幸运的是，Python 通过标准库提供的`unittest.mock`模块为我们提供了支持。

# 使用 Python 模拟框架

Python 提供的`unittest.mock`模块是一个非常强大的模拟框架，同时它也非常容易使用。

让我们使用这个库重做我们的测试。首先，我们需要在文件顶部导入`mock`模块，如下所示：

```py
from unittest import mock
```

接下来，我们将我们的第一个测试重写如下：

```py
class EventTest(unittest.TestCase):
    def test_a_listener_is_notified_when_an_event_is_raised(self):
        listener = mock.Mock()
        event = Event()
        event.connect(listener)
        event.fire()
        self.assertTrue(listener.called)
```

我们所做的唯一改变是将我们自己的自定义`Mock`类替换为 Python 提供的`mock.Mock`类。就是这样。通过这一行更改，我们的测试现在正在使用内置的模拟类。

`unittest.mock.Mock`类是 Python 模拟框架的核心。我们只需要实例化这个类，并将其传递到需要的地方。模拟将在`called`实例变量中记录是否被调用。

我们如何检查是否传入了正确的参数？让我们看看第二个测试的重写如下：

```py
    def test_a_listener_is_passed_right_parameters(self):
        listener = mock.Mock()
        event = Event()
        event.connect(listener)
        event.fire(5, shape="square")
        listener.assert_called_with(5, shape="square")
```

模拟对象会自动记录传入的参数。我们可以通过在`mock`对象上使用`assert_called_with`方法来断言参数。如果参数与预期不符，该方法将引发断言错误。如果我们对测试参数不感兴趣（可能我们只想检查方法是否被调用），则可以传递`mock.ANY`值。这个值将匹配任何传入的参数。

### 注意

与模拟上的断言相比，调用正常断言的方式有细微的差别。正常断言定义为`unittest.Testcase`类的一部分。由于我们的测试继承自该类，我们通过 self 调用断言，例如`self.assertEquals`。另一方面，模拟断言方法属于`mock`对象的一部分，所以你通过模拟对象调用它们，例如`listener.assert_called_with`。

模拟对象默认有四个断言可用：

+   `assert_called_with`：此方法断言最后一次调用是用给定的参数进行的

+   `assert_called_once_with`：这个断言检查方法是否恰好一次被调用，并且带有给定的参数

+   `assert_any_call`：这个方法检查在执行过程中是否在某个时刻调用了给定的调用

+   `assert_has_calls`：这个断言检查是否发生了一系列调用

四个断言非常微妙地不同，当模拟被多次调用时就会显现出来。`assert_called_with`方法只检查最后一次调用，所以如果有多次调用，则之前的调用将不会被断言。`assert_any_call`方法将检查在执行过程中是否发生了具有给定参数的调用。`assert_called_once_with`断言断言单个调用，所以如果模拟在执行过程中被多次调用，则此断言将失败。`assert_has_calls`断言可以用来断言具有给定参数的一组调用发生了。请注意，断言中检查的调用可能比我们检查的更多，但只要给定的调用存在，断言仍然会通过。

让我们更仔细地看看`assert_has_calls`断言。以下是我们可以使用此断言编写的相同测试：

```py
    def test_a_listener_is_passed_right_parameters(self):
        listener = mock.Mock()
        event = Event()
        event.connect(listener)
        event.fire(5, shape="square")
        listener.assert_has_calls([mock.call(5, shape="square")])
```

模拟框架内部使用`_Call`对象来记录调用。`mock.call`函数是一个创建这些对象的辅助工具。我们只需用预期的参数调用它来创建所需的调用对象。然后我们可以使用这些对象在`assert_has_calls`断言中，以断言预期的调用发生了。

当模拟被多次调用时，此方法很有用，我们只想断言一些调用。

## 模拟对象

在测试`Event`类时，我们只需要模拟单个函数。模拟的更常见用途是模拟一个类。

### 注意

本章的其余部分基于代码包中的`test_driven_python-CHAPTER4_PART2`。您可以从[`github.com/siddhi/test_driven_python/archive/CHAPTER4_PART2.zip`](https://github.com/siddhi/test_driven_python/archive/CHAPTER4_PART2.zip)下载。

以下是对`Alert`类实现的查看：

```py
class Alert:
    """Maps a Rule to an Action, and triggers the action if the rule
    matches on any stock update"""

    def __init__(self, description, rule, action):
        self.description = description
        self.rule = rule
        self.action = action

    def connect(self, exchange):
        self.exchange = exchange
        dependent_stocks = self.rule.depends_on()
        for stock in dependent_stocks:
            exchange[stock].updated.connect(self.check_rule)

    def check_rule(self, stock):
        if self.rule.matches(self.exchange):
            self.action.execute(self.description)
```

让我们按以下方式分解这个类的工作原理：

+   `Alert`类在初始化器中接受一个`Rule`和一个`Action`。

+   当调用`connect`方法时，它获取所有依赖的股票并将它们连接到它们的`updated`事件。

+   `updated`事件是我们之前看到的`Event`类的一个实例。每个`Stock`类都有一个此事件的实例，并且每当对那个股票进行新更新时，它就会被触发。

+   此事件的监听器是`Alert`类的`self.check_rule`方法。

+   在这种方法中，警报检查新更新是否导致匹配了规则。

+   如果规则匹配，它会在`Action`上调用执行方法。否则，不会发生任何操作。

该类有一些要求，如下所示，需要满足。这些要求中的每一个都需要被制作成一个单元测试。

+   如果股票被更新，类应该检查规则是否匹配

+   如果规则匹配，则应该执行相应的操作

+   如果规则不匹配，则不会发生任何操作

我们可以以多种不同的方式测试这一点；让我们来看看一些选项。

第一个选项是完全不使用模拟。我们可以创建一个规则，将其连接到测试操作，然后更新库存并验证操作是否已执行。以下是一个这样的测试示例：

```py
import unittest
from datetime import datetime
from unittest import mock

from ..alert import Alert
from ..rule import PriceRule
from ..stock import Stock

class TestAction:
    executed = False

    def execute(self, description):
        self.executed = True

class AlertTest(unittest.TestCase):
    def test_action_is_executed_when_rule_matches(self):
        exchange = {"GOOG": Stock("GOOG")}
        rule = PriceRule("GOOG", lambda stock: stock.price > 10)
        action = TestAction()
        alert = Alert("sample alert", rule, action)
        alert.connect(exchange)
        exchange["GOOG"].update(datetime(2014, 2, 10), 11)
        self.assertTrue(action.executed)
```

这是最直接的选择，但需要一些代码来设置，并且我们需要为测试用例创建一个`TestAction`。

我们可以不用创建测试动作，而是用模拟动作来替换它。然后我们可以简单地断言模拟动作已被执行。以下代码展示了这种测试用例的变体：

```py
    def test_action_is_executed_when_rule_matches(self):
        exchange = {"GOOG": Stock("GOOG")}
        rule = PriceRule("GOOG", lambda stock: stock.price > 10)
        action = mock.MagicMock()
        alert = Alert("sample alert", rule, action)
        alert.connect(exchange)
        exchange["GOOG"].update(datetime(2014, 2, 10), 11)
        action.execute.assert_called_with("sample alert")
```

关于这个测试的一些观察：

如果你注意到，警报不是我们迄今为止常用的普通`Mock`对象，而是一个`MagicMock`对象。`MagicMock`对象就像一个`Mock`对象，但它对 Python 上所有类都存在的特殊方法提供了支持，例如`__str__`、`hasattr`。如果我们不使用`MagicMock`，当代码使用这些方法中的任何一种时，我们有时可能会遇到错误或奇怪的行为。以下示例说明了这种差异：

```py
>>> from unittest import mock
>>> mock_1 = mock.Mock()
>>> mock_2 = mock.MagicMock()
>>> len(mock_1)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
TypeError: object of type 'Mock' has no len()
>>> len(mock_2)
0
>>>

```

通常，我们将在需要模拟类的大多数地方使用`MagicMock`。当我们需要模拟独立函数，或者在罕见情况下我们明确不想为魔法方法提供默认实现时，使用`Mock`是一个好选择。

关于这个测试的另一个观察点是处理方法的方式。在上面的测试中，我们创建了一个模拟动作对象，但我们在任何地方都没有指定这个模拟类应该包含一个`execute`方法以及它的行为方式。实际上，我们不需要这样做。当在模拟对象上访问方法或属性时，Python 会方便地创建一个模拟方法并将其添加到模拟类中。因此，当`Alert`类在我们的模拟动作对象上调用`execute`方法时，该方法就被添加到我们的模拟动作中。然后我们可以通过断言`action.execute.called`来检查该方法是否被调用。

Python 在访问时自动创建模拟方法的特性有一个缺点，那就是打字错误或接口更改可能不会被察觉。

例如，假设我们将所有 `Action` 类中的 `execute` 方法重命名为 `run`。但如果我们运行测试用例，它仍然通过。为什么它会通过？因为 `Alert` 类调用了 `execute` 方法，而测试只检查 `execute` 方法是否被调用，它确实被调用了。测试不知道在所有真实的 `Action` 实现中方法名称已被更改，并且当与实际操作集成时，`Alert` 类将无法工作。

为了避免这个问题，Python 支持使用另一个类或对象作为规范。当提供规范时，模拟对象只创建规范中存在的那些方法。所有其他方法或属性访问将引发错误。

规范是通过初始化时的 `spec` 参数传递给模拟的。`Mock` 和 `MagicMock` 类都支持设置规范。以下代码示例显示了设置 `spec` 参数与默认 `Mock` 对象之间的差异：

```py
>>> from unittest import mock
>>> class PrintAction:
...     def run(self, description):
...         print("{0} was executed".format(description))
...

>>> mock_1 = mock.Mock()
>>> mock_1.execute("sample alert") # Does not give an error
<Mock name='mock.execute()' id='54481752'>

>>> mock_2 = mock.Mock(spec=PrintAction)
>>> mock_2.execute("sample alert") # Gives an error
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "C:\Python34\lib\unittest\mock.py", line 557, in __getattr__
 raise AttributeError("Mock object has no attribute %r" % name)
AttributeError: Mock object has no attribute 'execute'

```

注意在上述示例中，`mock_1` 在没有错误的情况下执行了 `execute` 方法，尽管在 `PrintAction` 中该方法已被重命名。另一方面，通过提供一个规范，对不存在的 `execute` 方法的调用将引发异常。

## 模拟返回值

上面的第二个变体展示了我们如何在测试中使用模拟的 `Action` 类而不是真实的一个。同样，我们也可以在测试中使用模拟规则而不是创建 `PriceRule`。警报调用规则以查看新的库存更新是否导致规则匹配。警报的行为取决于规则返回 `True` 还是 `False`。

我们迄今为止创建的所有模拟都不需要返回值。我们只对是否调用了正确的调用感兴趣。如果我们模拟规则，那么我们必须配置它以返回测试的正确值。幸运的是，Python 使这变得非常简单。

我们所需要做的就是将返回值作为参数设置在模拟对象的构造函数中，如下所示：

```py
>>> matches = mock.Mock(return_value=True)
>>> matches()
True
>>> matches(4)
True
>>> matches(4, "abcd")
True

```

如上所示，模拟只是盲目地返回设置的值，不考虑参数。甚至不考虑参数的类型或数量。我们可以使用相同的程序来设置模拟对象中方法的返回值，如下所示：

```py
>>> rule = mock.MagicMock()
>>> rule.matches = mock.Mock(return_value=True)
>>> rule.matches()
True
>>>

```

设置返回值还有另一种方法，当处理模拟对象中的方法时非常方便。每个模拟对象都有一个 `return_value` 属性。我们只需将此属性设置为返回值，每次调用模拟对象都会返回该值，如下所示：

```py
>>> from unittest import mock
>>> rule = mock.MagicMock()
>>> rule.matches.return_value = True
>>> rule.matches()
True
>>>

```

在上述示例中，当我们访问 `rule.matches` 时，Python 会自动创建一个模拟的 `matches` 对象并将其放入 `rule` 对象中。这允许我们直接在一条语句中设置返回值，而无需为 `matches` 方法创建模拟。

现在我们已经看到了如何设置返回值，我们可以继续更改我们的测试，使用模拟规则对象，如下所示：

```py
    def test_action_is_executed_when_rule_matches(self):
        exchange = {"GOOG": Stock("GOOG")}
        rule = mock.MagicMock(spec=PriceRule)
        rule.matches.return_value = True
        rule.depends_on.return_value = {"GOOG"}
        action = mock.MagicMock()
        alert = Alert("sample alert", rule, action)
        alert.connect(exchange)
        exchange["GOOG"].update(datetime(2014, 2, 10), 11)
        action.execute.assert_called_with("sample alert")
```

`Alert` 对规则进行了两次调用：一次是 `depends_on` 方法，另一次是 `matches` 方法。我们为这两个方法都设置了返回值，并且测试通过了。

### 注意

如果没有为调用显式设置返回值，则默认返回值是返回一个新的模拟对象。对于每个被调用的方法，模拟对象都不同，但对于特定方法是一致的。这意味着如果多次调用相同的方法，每次都会返回相同的模拟对象。

## 模拟副作用

最后，我们来到 `Stock` 类。这是 `Alert` 类的最终依赖项。我们目前在测试中创建 `Stock` 对象，但我们可以像对 `Action` 和 `PriceRule` 类所做的那样，用模拟对象替换它。

`Stock` 类在行为上与其他两个模拟对象略有不同。`update` 方法不仅仅返回一个值——在这个测试中，其主要行为是触发更新事件。只有当这个事件被触发时，规则检查才会发生。

为了做到这一点，我们必须告诉我们的模拟股票类在调用 `update` 事件时触发事件。模拟对象有一个 `side_effect` 属性，使我们能够做到这一点。

我们可能有多种原因想要设置副作用。以下是一些原因：

+   我们可能想要调用另一个方法，比如在 `Stock` 类的例子中，当调用 `update` 方法时需要触发事件。

+   为了引发异常：这在测试错误情况时特别有用。一些错误，如网络超时，可能很难模拟，使用仅引发适当异常的模拟进行测试会更好。

+   为了返回多个值：这些值可能每次调用模拟时都不同，或者根据传递的参数返回特定的值。

设置副作用就像设置返回值一样。唯一的区别是副作用是一个 lambda 函数。当模拟执行时，参数会被传递给 lambda 函数，然后执行 lambda。以下是我们如何使用模拟的 `Stock` 类来做到这一点：

```py
    def test_action_is_executed_when_rule_matches(self):
        goog = mock.MagicMock(spec=Stock)
        goog.updated = Event()
        goog.update.side_effect = lambda date, value:
                goog.updated.fire(self)
        exchange = {"GOOG": goog}
        rule = mock.MagicMock(spec=PriceRule)
        rule.matches.return_value = True
        rule.depends_on.return_value = {"GOOG"}
        action = mock.MagicMock()
        alert = Alert("sample alert", rule, action)
        alert.connect(exchange)
        exchange["GOOG"].update(datetime(2014, 2, 10), 11)
        action.execute.assert_called_with("sample alert")
```

那么，那个测试中发生了什么？

1.  首先，我们创建 `Stock` 类的模拟而不是使用真实的一个。

1.  接下来，我们添加了 `updated` 事件。我们需要这样做，因为 `Stock` 类在 `__init__` 范围内运行时创建属性。因为属性是动态设置的，`MagicMock` 不会从 `spec` 参数中获取属性。我们在这里设置了一个实际的 `Event` 对象。我们也可以将其设置为模拟，但这可能有些过度。

1.  最后，我们在模拟股票对象中设置了`update`方法的副作用。lambda 函数接受方法所需的两个参数。在这个特定的例子中，我们只想触发事件，所以参数在 lambda 函数中没有使用。在其他情况下，我们可能希望根据参数的值执行不同的操作。设置`side_effect`属性允许我们做到这一点。

就像`return_value`属性一样，`side_effect`属性也可以在构造函数中设置。

运行测试，它应该通过。

`side_effect`属性也可以设置为异常或列表。如果设置为异常，那么在调用模拟时将抛出给定的异常，如下所示：

```py
>>> m = mock.Mock()
>>> m.side_effect = Exception()
>>> m()
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "C:\Python34\lib\unittest\mock.py", line 885, in __call__
 return _mock_self._mock_call(*args, **kwargs)
 File "C:\Python34\lib\unittest\mock.py", line 941, in _mock_call
 raise effect
Exception

```

如果将其设置为列表，那么每次调用模拟时，模拟将返回列表的下一个元素。这是一种模拟每次调用都要返回不同值的函数的好方法，如下所示：

```py
>>> m = mock.Mock()
>>> m.side_effect = [1, 2, 3]
>>> m()
1
>>> m()
2
>>> m()
3
>>> m()
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "C:\Python34\lib\unittest\mock.py", line 885, in __call__
 return _mock_self._mock_call(*args, **kwargs)
 File "C:\Python34\lib\unittest\mock.py", line 944, in _mock_call
 result = next(effect)
StopIteration

```

正如我们所见，模拟框架通过使用`side_effect`属性来处理副作用的方法非常简单，但非常强大。

## 模拟过多怎么办？

在前面的几个部分中，我们看到了使用不同级别的模拟编写的相同测试。我们从一个完全不使用任何模拟的测试开始，然后逐个模拟每个依赖项。这些解决方案中哪一个最好？

就像许多事情一样，这是一个个人偏好的问题。一个纯粹主义者可能会选择模拟所有依赖项。我个人的偏好是在对象小且自包含时使用真实对象。我不会模拟`Stock`类。这是因为模拟通常需要与返回值或副作用进行一些配置，这种配置可能会使测试变得杂乱无章，并使其可读性降低。对于小而自包含的类，直接使用真实对象会更简单。

在另一端，可能与外部系统交互的类、占用大量内存或运行缓慢的类是模拟的好候选。此外，需要大量其他对象来初始化的对象也是模拟的候选。使用模拟，你只需创建一个对象，传递它，并断言你感兴趣检查的部分。你不需要创建一个完全有效的对象。

即使在这里，也有模拟的替代方案。例如，当处理数据库时，通常会将数据库调用模拟出来，并将返回值硬编码到模拟中。这是因为数据库可能位于另一台服务器上，访问它会使测试变慢且不可靠。然而，而不是使用模拟，另一个选择可能是为测试使用快速内存数据库。这允许我们使用实时数据库而不是模拟数据库。哪种方法更好取决于具体情况。

## 模拟与存根与伪造与间谍之间的比较

到目前为止，我们一直在谈论模拟，但在术语上我们有点宽松。从技术上讲，我们谈论的所有内容都属于**测试双倍**的范畴。测试双倍是我们用来在测试用例中代替真实对象的某种假对象。

模拟是一种特定的测试双倍，它会记录对它的调用信息，这样我们就可以稍后对它们进行断言。

**存根**只是一个空的不做任何事情的对象或方法。当我们对测试中的某些功能不关心时，我们会使用它们。例如，假设我们有一个执行计算然后发送电子邮件的方法。如果我们正在测试计算逻辑，我们可能会在测试用例中将电子邮件发送方法替换为一个空的不做任何事情的方法，这样在测试运行时就不会发送任何电子邮件。

**伪造**是用一个更简单的对象或系统替换一个对象或系统，以简化测试。使用内存数据库而不是真实数据库，或者我们在本章前面创建的`TestAction`的示例，都是伪造的例子。

最后，**间谍**是类似于中间人的对象。像模拟一样，它们记录调用以便我们稍后可以对它们进行断言，但在记录之后，它们继续执行原始代码。与另外三个不同，间谍不会替换任何功能。在记录调用后，真实代码仍然被执行。间谍位于中间，不会导致执行模式发生变化。

## 方法修补

到目前为止，我们已经探讨了简单的模拟模式。这些是在大多数情况下你会使用的的方法。Python 的模拟框架并没有止步于此，它对更复杂的事情提供了巨大的支持。

让我们看看`PrintAction`类（将此代码放入`stock_alerter`目录下的`action.py`文件中）如下：

```py
class PrintAction:
    def execute(self, content):
        print(content)
```

这是一个简单的动作，当调用`execute`方法时，它只会将警报描述打印到屏幕上。

现在，我们如何进行测试呢？我们想要测试的是动作实际上以正确的参数调用打印方法。在先前的例子中，我们可以创建一个模拟对象并将其传递到类中，而不是传递一个真实对象。在这里，没有参数或属性我们可以简单地用模拟对象替换。

解决这个问题的方法是使用**修补**。修补是一种用模拟版本替换全局命名空间中的类或函数的方法。因为 Python 允许动态访问全局以及所有导入的模块，我们可以直接进入并更改标识符指向的对象。

在下面的序列中，你可以看到我们如何用另一个接受一个参数并返回双倍的函数替换`print`函数：

```py
>>> # the builtin print function prints a string
>>> print("hello") 
hello

>>> # the builtin print function handles multiple parameters
>>> print(1, 2) 
1 2

>>> # this is where the print function is mapped
>>> __builtins__.print 
<built-in function print>

>>> # make the builtin print point to our own lambda
>>> __builtins__.print = lambda x: x*2 

>>> # calling print now executes our substituted function
>>> print("hello") 
'hellohello'

>>> # our lambda does not support two parameters
>>> print(1, 2) Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
TypeError: <lambda>() takes 1 positional argument but 2 were given

```

如上图所示，现在所有的`print`调用都调用我们自己的函数，而不是默认的打印实现。

这给我们提供了进行模拟所需的提示。如果我们只是在运行测试之前将`print`替换为模拟，会怎样？这样代码最终会执行我们的模拟而不是默认的打印实现，然后我们可以在模拟上断言它是否以正确的参数被调用。

以下是这个技术的示例：

```py
import unittest
from unittest import mock
from ..action import PrintAction

class PrintActionTest(unittest.TestCase):
    def test_executing_action_prints_message(self):
        mock_print = mock.Mock()
        old_print = __builtins__["print"]
        __builtins__["print"] = mock_print
        try:
            action = PrintAction()
            action.execute("GOOG > $10")
            mock_print.assert_called_with("GOOG > $10")
        finally:
            __builtins__["print"] = old_print
```

这里发生了什么？

1.  首先，我们创建模拟函数。

1.  接下来，我们保存默认的打印实现。我们需要这样做，以便在测试结束时能够正确地恢复它。

1.  最后，我们用我们的模拟函数替换默认的打印。现在，每次调用打印函数时，它都会调用我们的模拟。

1.  我们在一个`try` - `finally`块中运行测试。

1.  在`finally`块中，我们恢复默认的打印实现。

非常、非常重要的是要恢复默认实现。记住，我们在这里更改的是全局信息，如果我们不恢复它，打印将指向我们的模拟函数，在所有后续的测试中也是如此。这可能会导致一些非常奇怪的行为，例如，在其他地方我们可能期望在屏幕上有输出，但没有打印出来，我们最终花费数小时试图弄清楚原因。这就是为什么测试被包裹在`try` - `finally`块中的原因。这样，即使在测试中抛出异常，模拟也会被重置回默认值。

我们刚刚看到了如何使用模拟修补函数和类，由于这是一个相当常见的任务，Python 通过`mock.patch`函数给了我们一个非常好的进行修补的方法。

`mock.patch`函数去除了修补函数所需的大量工作。让我们看看使用它的几种方法。

第一种方法复制了我们手动修补的方式。我们创建一个修补器，然后使用`start`方法来执行修补，使用`stop`方法来重置回原始实现，如下所示：

```py
    def test_executing_action_prints_message(self):
        patcher = mock.patch('builtins.print')
        mock_print = patcher.start()
        try:
            action = PrintAction()
            action.execute("GOOG > $10")
            mock_print.assert_called_with("GOOG > $10")
        finally:
            patcher.stop()
```

就像我们的手动修补一样，我们必须小心，即使在抛出异常的情况下也要调用停止方法。

修补也可以与`with`关键字一起用作上下文管理器。这种语法要干净得多，通常比我们自己调用开始和停止更可取：

```py
    def test_executing_action_prints_message(self):
        with mock.patch('builtins.print') as mock_print:
            action = PrintAction()
            action.execute("GOOG > $10")
            mock_print.assert_called_with("GOOG > $10")
```

让我们来看看这里发生了什么：

+   我们希望修补激活的代码被包裹在`with`块中。

+   我们调用`patch`函数，它返回要作为上下文管理器使用的修补器。模拟对象被设置在`as`部分指定的变量中。在这种情况下，修补后的模拟对象被设置为`mock_print`。

+   在这个块中，我们像往常一样执行测试和断言。

+   修补一旦执行离开上下文块就会被移除。这可能是因为所有语句都已执行，或者由于异常。

使用这种语法，我们不需要担心未处理的异常会导致修补出现问题。

`patch`函数也可以用作方法装饰器，如下面的示例所示：

```py
    @mock.patch("builtins.print")
    def test_executing_action_prints_message(self, mock_print):
        action = PrintAction()
        action.execute("GOOG > $10")
        mock_print.assert_called_with("GOOG > $10")
```

使用此语法，修补器修补所需的函数，并将替换模拟对象作为测试方法的第一个参数传入。然后我们可以像正常使用一样使用模拟对象。测试完成后，修补将被重置。

如果我们需要为多个测试修补相同的对象，则可以使用类装饰器语法，如下所示：

```py
@mock.patch("builtins.print")
class PrintActionTest(unittest.TestCase):
    def test_executing_action_prints_message(self, mock_print):
        action = PrintAction()
        action.execute("GOOG > $10")
        mock_print.assert_called_with("GOOG > $10")
```

此语法用修补装饰了类中的所有测试。默认情况下，装饰器会搜索以`test`开头的方法。但是，可以通过设置`patch.TEST_PREFIX`属性来更改此行为，类装饰器将修补所有以该前缀开头的方法。

### 修补时的一个重要注意事项

在修补时，我们应该记住修补类所使用的确切对象。Python 允许多个对象引用，很容易修补错误的对象。然后我们可能会花费数小时 wondering 为什么模拟对象没有被执行。

例如，支持文件`alert.py`使用如下导入：

```py
from rule import PriceRule
```

现在在警报测试中，如果我们想要修补`PriceRule`，那么我们需要这样做：

```py
import alert

@mock.patch("alert.PriceRule")
def test_patch_rule(self, mock_rule):
    ....
```

只有这样做，我们才能修补在`alert.py`文件中使用的`PriceRule`对象。以下方式将不起作用：

```py
import rule

@mock.patch("rule.PriceRule")
def test_patch_rule(self, mock_rule):
    ....
```

此代码将修补`rule.PriceRule`，这与我们想要修补的实际对象不同。当我们运行此测试时，我们会看到警报执行的是真实的`PriceRule`对象，而不是我们修补出的那个。

由于这是一个常见的错误，所以当我们遇到测试无法正确执行修补对象的问题时，我们应该首先检查这一点。

# 将所有内容串联起来

让我们用一个更复杂的例子来总结这一章。以下是`EmailAction`类的代码。此操作在规则匹配时向用户发送电子邮件。

```py
import smtplib
from email.mime.text import MIMEText

class EmailAction:
    """Send an email when a rule is matched"""
    from_email = "alerts@stocks.com"

    def __init__(self, to):
        self.to_email = to

    def execute(self, content):
        message = MIMEText(content)
        message["Subject"] = "New Stock Alert"
        message["From"] = "alerts@stocks.com"
        message["To"] = self.to_email
        smtp = smtplib.SMTP("email.stocks.com")
        try:
            smtp.send_message(message)
        finally:
            smtp.quit()
```

以下是如何使用该库：

1.  我们在`smtplib`库中实例化`SMTP`类，并传入我们想要连接的服务器。这将返回`SMTP`对象。

1.  我们在`SMTP`对象上调用`send_message`方法，传入电子邮件消息详情，以`MIMEText`对象的形式。

1.  最后，我们调用`quit`方法。此方法始终需要调用，即使在发送消息时发生异常。

基于此，我们需要测试以下内容：

1.  调用`smtplib`库时，使用正确的参数。

1.  消息内容（发件人、收件人、主题、正文）是正确的。

1.  即使在发送消息时抛出异常，也会调用`quit`方法。

让我们从简单的测试开始。此测试是为了验证`SMTP`类是否使用正确的参数初始化：

```py
class EmailActionTest(unittest.TestCase):
    def setUp(self):
        self.action = EmailAction(to="siddharta@silverstripesoftware.com")

    def test_email_is_sent_to_the_right_server(self, mock_smtp_class):
        self.action.execute("MSFT has crossed $10 price level")
        mock_smtp_class.assert_called_with("email.stocks.com")
```

首先，我们在每个测试中修补`smtplib`模块中的`SMTP`类。由于我们将为此做每个测试，我们将此设置为一个类装饰器。然后在`setUp`中实例化我们想要测试的`EmailAction`。

测试本身相当简单。我们调用动作的`execute`方法，并断言模拟类是用正确的参数实例化的。

以下测试验证了对`SMTP`对象是否执行了正确的调用：

```py
    def test_connection_closed_after_sending_mail(self, mock_smtp_class):
        mock_smtp = mock_smtp_class.return_value
        self.action.execute("MSFT has crossed $10 price level")
        mock_smtp.send_message.assert_called_with(mock.ANY)
        self.assertTrue(mock_smtp.quit.called)
        mock_smtp.assert_has_calls([
            mock.call.send_message(mock.ANY),
            mock.call.quit()])
```

在这个测试中，有几个新的方法值得讨论。

首先是这个测试系列中的一个微妙差异，其中我们模拟的是`SMTP`类而不是一个*对象*。在第一个测试中，我们检查了传递给构造函数的参数。由于我们模拟了类，我们可以在我们的模拟对象上直接断言。

在这个测试中，我们需要检查是否在`SMTP` *对象*上执行了正确的调用。由于对象是初始化类的返回值，我们可以从模拟`smtp`类的返回值中访问模拟的`smtp`对象。这正是测试的第一行所做的事情。

接下来，我们像往常一样执行动作。

最后，我们使用`assert_has_calls`方法断言执行了正确的调用。我们可以像以下这样断言调用：

```py
        mock_smtp.send_message.assert_called_with(mock.ANY)
        self.assertTrue(mock_smtp.quit.called)
```

主要区别在于上述断言没有断言序列。假设动作首先调用`quit`方法，然后调用`send_message`，它仍然会通过这两个断言。然而，`assert_has_calls`断言不仅检查方法是否被调用，还检查`quit`方法是在`send_message`之后被调用的。

以下第三个测试检查即使在发送消息时抛出异常，连接也会关闭：

```py
    def test_connection_closed_if_send_gives_error(self, mock_smtp_class):
        mock_smtp = mock_smtp_class.return_value
        mock_smtp.send_message.side_effect =
        smtplib.SMTPServerDisconnected()
        try:
            self.action.execute("MSFT has crossed $10 price level")
        except Exception:
            pass
        self.assertTrue(mock_smtp.quit.called)
```

在这个测试中，我们使用`side_effect`属性将发送消息的模拟设置为抛出异常。然后我们检查即使在抛出异常的情况下，`quit`方法也被调用了。

在最后一个测试中，我们需要检查正确的消息内容是否传递给了`send_message`。该函数接受一个`MIMEText`对象作为参数。我们如何检查传递了正确的对象？

以下是一种*不工作*的方法：

```py
    def test_email_is_sent_with_the_right_subject(self, mock_smtp_class):
        mock_smtp = mock_smtp_class.return_value
        self.action.execute("MSFT has crossed $10 price level")
        message = MIMEText("MSFT has crossed $10 price level")
        message["Subject"] = "New Stock Alert"
        message["From"] = "alerts@stocks.com"
        message["To"] = "siddharta@silverstripesoftware.com"
        mock_smtp.send_message.assert_called_with(message)
```

如果我们运行上述测试，我们会得到如下失败：

```py
AssertionError: Expected call: send_message(<email.mime.text.MIMEText object at 0x0000000003641F98>)
Actual call: send_message(<email.mime.text.MIMEText object at 0x000000000363A0F0>)

```

问题在于，尽管预期的`MIMEText`对象和传递给`send_message`的实际对象的内容相同，但测试仍然失败，因为它们是两个不同的对象。模拟框架通过相等性比较这两个参数，由于它们都是两个不同的对象，所以相等性测试失败。

解决这个问题的方法之一是进入模拟，提取调用中传递的参数，并检查它们是否包含正确的数据。以下是一个使用此方法的测试：

```py
    def test_email_is_sent_with_the_right_subject(self, mock_smtp_class):
        mock_smtp = mock_smtp_class.return_value
        self.action.execute("MSFT has crossed $10 price level")
        call_args, _ = mock_smtp.send_message.call_args
        sent_message = call_args[0]
        self.assertEqual("New Stock Alert", sent_message["Subject"])
```

一旦调用`execute`方法，我们就访问`mock`对象的`call_args`属性，以获取传递给`send_message`的参数。我们取第一个参数，即我们感兴趣的`MIMEText`对象。然后我们断言主题符合预期。

有一种更优雅的方法。记住我们说过模拟框架是通过相等性来比较参数的吗？这意味着我们可以传递一个实现了`__eq__`特殊方法的对象，并使用它来执行我们想要的任何比较。以下是一个用于检查两个`MIMEText`消息之间相等性的此类类：

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

这个类基本上接受一个值字典，然后可以用来比较`MIMEText`对象是否包含这些值（至少是我们感兴趣的值）。由于它实现了`__eq__`方法，可以直接使用相等性进行检查，如下所示：

```py
>>> message = MIMEText("d")
>>> message["Subject"] = "a"
>>> message["From"] = "b"
>>> message["To"] = "c"
>>> expected = MessageMatcher({"Subject":"a", "From":"b", "To":"c", "Message":"d"})
>>> message == expected
True

```

我们可以使用这种技术将这样的对象作为测试的预期参数传递，如下所示：

```py
    def test_email_is_sent_when_action_is_executed(self, mock_smtp_class):
        expected_message = {
            "Subject": "New Stock Alert",
            "Message": "MSFT has crossed $10 price level",
            "To": "siddharta@silverstripesoftware.com",
            "From": "alerts@stocks.com"
        }
        mock_smtp = mock_smtp_class.return_value
        self.action.execute("MSFT has crossed $10 price level")
        mock_smtp.send_message.assert_called_with(
            MessageMatcher(expected_message))
```

编写像这样的自定义参数匹配器是一种简单的方法，可以断言我们可能没有直接对象访问的参数，或者当我们只想为了测试目的比较对象的一些属性时。

# 摘要

在本章中，你学习了如何使用模拟来测试对象之间的交互。你看到了如何手动编写我们的模拟，然后是使用 Python 标准库中提供的模拟框架。接下来，你看到了如何使用补丁进行更高级的模拟。我们通过查看一个稍微复杂一些的模拟示例来结束，这个示例让我们将所有模拟技术付诸实践。

到目前为止，你一直在查看编写新代码的测试。在下一章中，你将了解如何处理没有测试的现有代码。
