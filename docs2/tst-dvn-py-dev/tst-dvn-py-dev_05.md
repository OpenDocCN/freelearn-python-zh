# 第五章 工作于遗留代码

拥有一套稳固的单元测试对于成功的项目至关重要。如您所见，单元测试不仅有助于防止错误进入代码，而且在许多其他方面也有帮助，如指导设计、使我们能够重构代码并保持其可维护性，以及作为参考，让您可以看到预期的行为应该是怎样的。

TDD（测试驱动开发）是确保我们的代码具有前一段所述所有特性的最佳方式。但是，任何参与过更大、更复杂项目的人都知道，总有一些代码片段没有经过测试。通常，这些是多年前编写的代码，在我们开始实践 TDD 之前就已经存在了。或者，它可能是为了赶在紧急截止日期前匆忙编写的代码。

无论哪种方式，这都是没有关联测试的代码。代码通常很混乱。它对其他类有大量的依赖。现在，我们需要向这段代码添加一个新功能。我们该如何着手？我们只是直接进去修改新功能吗？还是有一种更好的方法？

# 什么是遗留代码？

在本章中，我们将使用术语**遗留代码**来指代任何没有单元测试的代码。这是一个相当宽泛的定义，因为它包括很久以前编写的代码以及某些原因下没有编写测试的最近代码。虽然这并不是严格意义上的旧代码，但在 TDD 社区中，这是一个流行的定义，由迈克尔·费瑟斯的优秀著作《与遗留代码有效工作》（Prentice Hall，2004）使之流行起来，我们也将在这个书中采用这个含义。

与遗留代码工作的五个步骤：

1.  **理解代码**：如果我们很幸运，我们会有一些优秀的文档来帮助我们理解即将接触的代码。更有可能的是，文档会很少，或者根本不存在。由于没有测试，我们无法阅读测试来尝试理解代码应该做什么。而对于非常旧的代码，编写代码的人可能已经不再为您的组织工作了。这听起来就像是一场完美的风暴，会给我们带来麻烦，但正如任何参与过大型生产项目的人都可以证明的那样，这是大多数代码库的常态。因此，我们的第一步就是理解代码，弄清楚发生了什么。

1.  **打破依赖关系**：一旦我们开始理解代码，我们的下一步就是为代码编写一些测试。对于遗留代码来说，这并不简单，因为设计通常是与其他文件和类相互依赖的意大利面式的混乱。在我们编写单元测试之前，我们需要某种方法来打破这些依赖关系。

1.  **编写测试**：我们现在终于可以为我们即将修改的代码编写一些单元测试了。

1.  **重构**：现在我们已经有了测试，我们可以开始应用我们在本书前面看到的一些重构技术。

1.  **实现新功能**：在清理代码后，我们现在可以实施新功能，当然，包括测试。

虽然前面的步骤被显示为线性序列，但重要的是要理解，步骤往往以非线性的方式进行。例如，当我们试图理解一个大方法时，我们可能会取一小段代码，将其提取为方法，更详细地查看它，然后为它编写几个测试，最后回到原始方法并查看方法的另一部分。然后我们可能回到我们提取的方法并将它们提取到一个新类中。步骤来回进行，直到我们处于一个可以安全实施新功能且没有破坏东西的风险的位置。

# 理解代码

以下是我们将在本章中查看的代码：

```py
from datetime import datetime

from .stock import Stock
from .rule import PriceRule

class AlertProcessor:
    def __init__(self):
        self.exchange = {"GOOG": Stock("GOOG"), "AAPL": Stock("AAPL")}
        rule_1 = PriceRule("GOOG", lambda stock: stock.price > 10)
        rule_2 = PriceRule("AAPL", lambda stock: stock.price > 5)
        self.exchange["GOOG"].updated.connect(
            lambda stock: print(stock.symbol, stock.price) \
                if rule_1.matches(self.exchange) else None)
        self.exchange["AAPL"].updated.connect(
            lambda stock: print(stock.symbol, stock.price) \
                if rule_2.matches(self.exchange) else None)

        updates = []
        with open("updates.csv", "r") as fp:
            for line in fp.readlines():
                symbol, timestamp, price = line.split(",")
                updates.append((symbol, datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f"), int(price)))

        for symbol, timestamp, price in updates:
            stock = self.exchange[symbol]
            stock.update(timestamp, price)
```

这是一段执行某些操作的代码。我们所知道的是，它从文件中获取一些更新并通过一些警报运行它。以下是如何看起来`updates.csv`文件：

```py
GOOG,2014-02-11T14:10:22.13,5
AAPL,2014-02-11T00:00:00.0,8
GOOG,2014-02-11T14:11:22.13,3
GOOG,2014-02-11T14:12:22.13,15
AAPL,2014-02-11T00:00:00.0,10
GOOG,2014-02-11T14:15:22.13,21
```

我们现在需要向这段代码添加一些功能：

+   我们需要能够从网络服务器获取更新

+   我们需要能够在匹配到警报时发送电子邮件

在我们开始之前，我们需要能够理解当前的代码。我们通过特征测试来实现这一点。

## 什么是特征测试？

**特征测试**是描述代码当前行为的测试。我们不是针对预定义的期望来编写测试，而是针对实际行为来编写测试。你可能会问这能完成什么，因为如果我们打算查看当前行为并编写一个寻找相同内容的测试，测试是不可能失败的。然而，需要记住的是，我们并不是试图找到错误。相反，通过针对当前行为编写测试，我们正在构建一个测试的安全网。如果我们重构过程中破坏了某些东西，测试将失败，我们将知道我们必须撤销我们的更改。

## 使用 Python 交互式外壳来理解代码

那么，这段代码做什么呢？让我们打开 Python 交互式外壳并看看。交互式外壳是一个很好的帮助工具，因为它允许我们与代码互动，尝试不同的输入值并查看我们得到什么样的输出。现在让我们按照以下方式打开类：

```py
>>> from stock_alerter.legacy import AlertProcessor
>>> processor = AlertProcessor()
AAPL 8
GOOG 15
AAPL 10
GOOG 21
>>>

```

如我们所见，仅仅实例化类就使代码运行，我们在终端上打印了一些输出。考虑到所有代码都在`__init__`方法中，这并不令人惊讶。

## 编写特征测试

好的，我们现在有了一些可以编写测试的内容。我们理解，当`updates.csv`文件中的输入如下时：

```py
GOOG,2014-02-11T14:10:22.13,5
AAPL,2014-02-11T00:00:00.0,8
GOOG,2014-02-11T14:11:22.13,3
GOOG,2014-02-11T14:12:22.13,15
AAPL,2014-02-11T00:00:00.0,10
GOOG,2014-02-11T14:15:22.13,21
```

然后，当我们实例化类时的输出如下：

```py
AAPL 8
GOOG 15
AAPL 10
GOOG 21
```

我们可能还不知道为什么这是输出或它是如何计算的，但这足以开始测试。以下就是测试的样子：

```py
import unittest
from unittest import mock

from ..legacy import AlertProcessor

class AlertProcessorTest(unittest.TestCase):
    @mock.patch("builtins.print")
    def test_processor_characterization_1(self, mock_print):
        AlertProcessor()
        mock_print.assert_has_calls([mock.call("AAPL", 8),
                                     mock.call("GOOG", 15),
                                     mock.call("AAPL", 10),
                                     mock.call("GOOG", 21)])
```

所有这些测试所做的只是模拟`print`函数并实例化类。我们断言所需的数据被打印出来。

这是一个伟大的单元测试吗？可能不是。首先，它仍然从`updates.csv`文件中获取输入。理想情况下，我们会模拟文件访问。但此刻这并不重要。这个测试通过了，并且当开始修改代码时，它将是一个安全网。这就是我们现在需要的测试所做的一切。

## 使用 pdb 理解代码

Python 交互式外壳是理解方法调用边界代码的好方法。它允许我们传入各种输入组合并查看我们得到什么样的输出。但如果我们想看到函数或方法内部发生的事情呢？这就是`pdb`能极其有用的时候。

**pdb**是 Python 调试器，它是 Python 标准库的一部分。它有许多功能，例如能够逐行执行代码，查看变量如何变化，以及设置和删除断点。pdb 非常强大，有许多好书详细介绍了它。我们不会在这本书中介绍所有功能，但只是给出一个简短的例子，说明如何使用它来理解代码。

要在`pdb`中执行代码，请在交互式外壳中运行以下行：

```py
>>> import pdb
>>> from stock_alerter.legacy import AlertProcessor
>>> pdb.run("AlertProcessor()")
> <string>(1)<module>()
(Pdb)

```

`pdb.run`方法允许我们指定任何字符串作为参数。这个字符串将在调试器中执行。在这种情况下，我们正在实例化开始执行所有代码的类。

在这一点上，我们得到了`(Pdb)`提示符，从这里我们可以逐行执行代码。你可以通过输入`help`来获取各种命令的帮助，如下所示：

```py
(Pdb) help

Documented commands (type help <topic>):
EOF    c          d        h         list      q        rv       undisplay
a      cl         debug    help      ll        quit     s        unt
alias  clear      disable  ignore    longlist  r        source   until
args   commands   display  interact  n         restart  step     up
b      condition  down     j         next      return   tbreak   w
break  cont       enable   jump      p         retval   u        whatis
bt     continue   exit     l         pp        run      unalias  where

Miscellaneous help topics:
pdb  exec

(Pdb)
```

或者，你也可以通过输入`help <command>`来获取特定命令的帮助：

```py
(Pdb) help s
s(tep)
        Execute the current line, stop at the first possible occasion
        (either in a function that is called or in the current
        function).
(Pdb)
```

## 一些常见的 pdb 命令

大多数时候，我们会使用以下命令：

+   `s`: 这将执行一行代码（如果需要，将进入函数调用）

+   `n`: 这将执行代码直到你达到当前函数的下一行

+   `r`: 这将执行代码直到当前函数返回

+   `q`: 这将退出调试器

+   `b`: 这将在文件的特定行上设置断点

+   `cl`: 这将清除断点

+   `c`: 这将继续执行直到遇到断点或执行结束

这些命令应该足以在代码中移动并尝试检查正在发生的事情。pdb 还有许多其他命令，我们在这里不会介绍。

## 漫步 pdb 会话

现在我们来实际操作。以下是我们使用`pdb`遍历代码的过程。

首先，我们在`pdb`中运行我们的命令，如下所示：

```py
>>> import pdb
>>> from stock_alerter.legacy import AlertProcessor
>>> pdb.run("AlertProcessor()")
> <string>(1)<module>()
(Pdb)

```

让我们按照以下方式进入第一行：

```py
(Pdb) s
--Call—
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(8)__init__()
-> def __init__(self):
(Pdb)
```

pdb 告诉我们我们现在在`__init__`方法中。`n`命令将带我们通过这个方法的前几行，如下所示：

```py
(Pdb) n
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(9)__init__()
-> self.exchange = {"GOOG": Stock("GOOG"), "AAPL": Stock("AAPL")}
(Pdb) n
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(10)__init__()
-> rule_1 = PriceRule("GOOG", lambda stock: stock.price > 10)
(Pdb) n
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(11)__init__()
-> rule_2 = PriceRule("AAPL", lambda stock: stock.price > 5)
(Pdb) n
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(12)__init__()
-> self.exchange["GOOG"].updated.connect(
(Pdb) n
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(13)__init__()
-> lambda stock: print(stock.symbol, stock.price) \
(Pdb) n
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(15)__init__()
-> self.exchange["AAPL"].updated.connect(
(Pdb) n
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(16)__init__()
-> lambda stock: print(stock.symbol, stock.price) \
(Pdb) n
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(18)__init__()
-> updates = []
(Pdb)
```

我们可以通过查看一些变量来检查似乎已经完成的初始化，如下所示：

```py
(Pdb) self.exchange
{'GOOG': <stock_alerter.stock.Stock object at 0x0000000002E59400>, 'AAPL': <stock_alerter.stock.Stock object at 0x0000000002E593C8>}
(Pdb) rule_1
<stock_alerter.rule.PriceRule object at 0x0000000002E205F8>
```

我们甚至可以尝试用各种输入执行一些局部变量，如下所示：

```py
(Pdb) test_stock = Stock("GOOG")
(Pdb) test_stock.update(datetime.now(), 100)
(Pdb) rule_1.matches({"GOOG": test_stock})
True
(Pdb)
```

这有助于我们了解执行不同部分时各种对象的状态。

下一节代码是打开文件并读取的部分。让我们通过在 25 行设置断点并使用`c`命令直接执行来跳过这部分，如下所示：

```py
(Pdb) b stock_alerter\legacy.py:25
Breakpoint 1 at c:\projects\tdd_with_python\src\stock_alerter\legacy.py:25
(Pdb) c
> c:\projects\tdd_with_python\src\stock_alerter\legacy.py(25)__init__()
-> for symbol, timestamp, price in updates:
(Pdb)
```

现在文件读取部分已经完成，我们可以通过检查更新的局部变量来检查读取的数据格式。`pp`命令进行美化打印，以便输出更容易阅读，如下所示：

```py
(Pdb) pp updates
[('GOOG', datetime.datetime(2014, 2, 11, 14, 10, 22, 130000), 5),
 ('AAPL', datetime.datetime(2014, 2, 11, 0, 0), 8),
 ('GOOG', datetime.datetime(2014, 2, 11, 14, 11, 22, 130000), 3),
 ('GOOG', datetime.datetime(2014, 2, 11, 14, 12, 22, 130000), 15),
 ('AAPL', datetime.datetime(2014, 2, 11, 0, 0), 10),
 ('GOOG', datetime.datetime(2014, 2, 11, 14, 15, 22, 130000), 21)]
```

看起来文件被解析为包含`(股票代码，时间戳，价格)`的元组列表。让我们看看如果我们只有 GOOG 更新会发生什么，如下所示：

```py
(Pdb) updates = [update for update in updates if update[0] == "GOOG"]
(Pdb) pp updates
[('GOOG', datetime.datetime(2014, 2, 11, 14, 10, 22, 130000), 5),
 ('GOOG', datetime.datetime(2014, 2, 11, 14, 11, 22, 130000), 3),
 ('GOOG', datetime.datetime(2014, 2, 11, 14, 12, 22, 130000), 15),
 ('GOOG', datetime.datetime(2014, 2, 11, 14, 15, 22, 130000), 21)]
```

我们继续。正如我们所见，甚至在执行过程中中途改变局部变量的值也是可能的。以下是我们运行代码剩余部分时的输出：

```py
(Pdb) cl
Clear all breaks? y
Deleted breakpoint 4 at c:\projects\tdd_with_python\src\stock_alerter\legacy.py:25
(Pdb) c
GOOG 15
GOOG 21
>>>
```

`cl`命令清除断点，我们使用`c`命令运行到执行的末尾。修改后的更新变量输出被打印出来。由于此时执行已经完成，我们被返回到交互式 shell。

我们现在的探索已经完成。在任何时候，我们都可以像下面这样退出调试器：

```py
(Pdb) q
>>>
```

退出调试器将我们带回到交互式 shell。在这个时候，我们可能会根据我们刚刚进行的探索添加一些更多的特征测试。

# 打破依赖关系的技巧

现在我们已经看到了一些帮助我们理解代码的技巧，我们的下一步是打破依赖关系。这将帮助我们编写进一步的特性测试。为此，我们将非常小心地开始修改代码。在此期间，我们将尽量坚持以下目标：

+   进行非常不可能破坏的小改动

+   尽量少改变公共接口

为什么有这些目标？因为我们缺乏测试，我们必须对所做的更改保持谨慎。因此，小改动更好。我们还需要注意不要改变公共接口，因为我们必须去修复所有使用这个类的其他文件和模块。

## 线索重构库

**线索重构库**是一个用于执行代码自动重构的库。例如，你可以选择几行代码，然后输入命令将其提取为方法。库将自动创建这个方法，包括适当的代码、参数和返回值，并将自动在原始代码的位置放置对新提取方法的调用。在 Python 中自动重构有点棘手，因为语言的动态特性使得正确识别所有更改变得困难。然而，它非常适合进行小改动，就像我们将在本章中做的那样。

由于它是一个库，Rope 没有用于执行重构的 UI。相反，它集成到开发环境中，作为 IDE 或文本编辑器。大多数流行的 IDE 和文本编辑器都支持与 Rope 集成。Rope 可在[`github.com/python-rope/rope`](https://github.com/python-rope/rope)找到。

如果你的 IDE 或你选择的文本编辑器支持与 Rope 集成，或者有内置的重构功能，那么尽可能使用它。

## 将初始化与执行分离

我们正在工作的班级遇到的一个问题是整个执行过程都发生在`__init__`方法中。这意味着一旦类被构造，所有操作都会在我们有机会设置模拟或进行其他有助于编写特征测试的更改之前执行。幸运的是，这个问题有一个简单的解决方案。我们将简单地将执行部分移动到一个单独的方法中，如下所示：

```py
class AlertProcessor:
    def __init__(self):
        self.exchange = {"GOOG": Stock("GOOG"), "AAPL": Stock("AAPL")}
        rule_1 = PriceRule("GOOG", lambda stock: stock.price > 10)
        rule_2 = PriceRule("AAPL", lambda stock: stock.price > 5)
        self.exchange["GOOG"].updated.connect(
            lambda stock: print(stock.symbol, stock.price) \
                          if rule_1.matches(self.exchange) else None)
        self.exchange["AAPL"].updated.connect(
            lambda stock: print(stock.symbol, stock.price) \
                          if rule_2.matches(self.exchange) else None)

    def run(self):
        updates = []
        with open("updates.csv", "r") as fp:
            for line in fp.readlines():
                symbol, timestamp, price = line.split(",")
                updates.append(
                       (symbol,
                        datetime.strptime(timestamp,
                                          "%Y-%m-%dT%H:%M:%S.%f"),
                        int(price)))

        for symbol, timestamp, price in updates:
            stock = self.exchange[symbol]
            stock.update(timestamp, price)
```

聪明的读者可能会注意到，我们刚刚打破了我们的第二个目标——最小化对公共接口的更改。我们所做的更改已经改变了接口。如果有其他模块使用这个类，它们只会构造这个类，假设所有处理都已经完成。我们现在必须找到所有创建这个类的位置，并添加对`run`方法的调用。否则，这个类将无法按预期工作。

为了避免需要修复所有调用者，我们可以在初始化器内部自己调用`run`方法，如下所示：

```py
def __init__(self):
    self.exchange = {"GOOG": Stock("GOOG"), "AAPL": Stock("AAPL")}
    rule_1 = PriceRule("GOOG", lambda stock: stock.price > 10)
    rule_2 = PriceRule("AAPL", lambda stock: stock.price > 5)
    self.exchange["GOOG"].updated.connect(
        lambda stock: print(stock.symbol, stock.price) \
                      if rule_1.matches(self.exchange) else None)
    self.exchange["AAPL"].updated.connect(
        lambda stock: print(stock.symbol, stock.price) \
                      if rule_2.matches(self.exchange) else None)
    self.run()
```

所有测试都通过了，但又一次，所有代码在实例化类的那一刻就会执行。我们是不是回到了起点？让我们在下一节中看看。

## 为参数使用默认值

Python 最有用的特性之一是能够为参数设置默认值的概念。这允许我们更改接口，同时让现有的调用者看起来没有变化。

在上一节中，我们将一段代码移动到了`run`方法中，并从`__init__`方法中调用了这个方法。这似乎我们没有真正改变什么，但这是一种误导。

下面是`__init__`方法的下一个更改：

```py
def __init__(self, autorun=True):
    self.exchange = {"GOOG": Stock("GOOG"), "AAPL": Stock("AAPL")}
    rule_1 = PriceRule("GOOG", lambda stock: stock.price > 10)
    rule_2 = PriceRule("AAPL", lambda stock: stock.price > 5)
    self.exchange["GOOG"].updated.connect(
        lambda stock: print(stock.symbol, stock.price) \
                      if rule_1.matches(self.exchange) else None)
    self.exchange["AAPL"].updated.connect(
        lambda stock: print(stock.symbol, stock.price) \
                      if rule_2.matches(self.exchange) else None)
    if autorun:
        self.run()
```

我们所做的是引入了一个名为`autorun`的新参数，并将其默认值设置为`True`。然后我们用条件语句包装对`run`方法的调用。只有当`autorun`为`True`时，才会调用`run`方法。

所有现有的调用者使用这个类将保持不变——当不带参数调用构造函数时，`autorun`参数将被设置为`True`，并将调用`run`方法。一切都将如预期。

但添加参数给了我们一个选项，可以在测试中显式地将`autorun`参数设置为`False`，从而避免调用`run`方法。我们现在可以实例化这个类，然后设置我们想要的任何模拟或其他测试初始化，然后手动在测试中调用`run`方法。

下面的特征测试与之前我们编写的相同，但重写以利用这个新功能：

```py
def test_processor_characterization_2(self):
    processor = AlertProcessor(autorun=False)
    with mock.patch("builtins.print") as mock_print:
        processor.run()
    mock_print.assert_has_calls([mock.call("AAPL", 8),
                                 mock.call("GOOG", 15),
                                 mock.call("AAPL", 10),
                                 mock.call("GOOG", 21)])
```

哈哈！现在这个变化看起来很小，但它正是使我们能够编写所有后续特征测试的变化。

## 提取方法并测试

测试大的方法非常困难。这是因为测试只能检查输入、输出和交互。如果整个方法中只有几行是我们想要测试的，那么这就会成为一个问题。

让我们再次看看`run`方法，如下所示：

```py
def run(self):
    updates = []
    with open("updates.csv", "r") as fp:
        for line in fp.readlines():
            symbol, timestamp, price = line.split(",")
            updates.append((symbol, datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f"), int(price)))

    for symbol, timestamp, price in updates:
        stock = self.exchange[symbol]
        stock.update(timestamp, price)
```

假设我们只想为第二循环中的代码编写特征测试。如何做到这一点？一个简单的方法是将这些行提取到一个单独的方法中，如下所示：

```py
def do_updates(self, updates):
    for symbol, timestamp, price in updates:
        stock = self.exchange[symbol]
        stock.update(timestamp, price)
```

我们需要在原始位置调用这个新方法，如下所示：

```py
def run(self):
    updates = []
    with open("updates.csv", "r") as fp:
        for line in fp.readlines():
            symbol, timestamp, price = line.split(",")
            updates.append((symbol, datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f"), int(price)))
    self.do_updates(updates)
```

我们现在可以按照以下方式为这个新方法编写特征测试：

```py
def test_processor_characterization_3(self):
    processor = AlertProcessor(autorun=False)
    mock_goog = mock.Mock()
    processor.exchange = {"GOOG": mock_goog}
    updates = [("GOOG", datetime(2014, 12, 8), 5)]
    processor.do_updates(updates)
    mock_goog.update.assert_called_with(datetime(2014, 12, 8), 5)
```

理想情况下，我们尝试提取小的代码组，以便在不犯错误的情况下轻松执行提取方法重构。记住，我们在这里没有现有单元测试的安全网。

## 注入依赖

在先前的特征测试中，我们实例化了类，然后继续用另一个实例变量替换交换实例变量，其中`Stock`类被模拟。实现这一目标的另一种方法是使用以下早期技巧引入默认变量：

```py
def __init__(self, autorun=True, exchange=None):
    if exchange is None:
        self.exchange = {"GOOG": Stock("GOOG"), "AAPL": Stock("AAPL")}
    else: 
        self.exchange = exchange
    rule_1 = PriceRule("GOOG", lambda stock: stock.price > 10)
    rule_2 = PriceRule("AAPL", lambda stock: stock.price > 5)
    self.exchange["GOOG"].updated.connect(
        lambda stock: print(stock.symbol, stock.price) \
            if rule_1.matches(self.exchange) else None)
    self.exchange["AAPL"].updated.connect(
        lambda stock: print(stock.symbol, stock.price) \
            if rule_2.matches(self.exchange) else None)
    if autorun:
        self.run()
```

这允许我们在编写特征测试时注入模拟，如下所示：

```py
def test_processor_characterization_4(self):
    mock_goog = mock.Mock()
    mock_aapl = mock.Mock()
    exchange = {"GOOG": mock_goog, "AAPL": mock_aapl}
    processor = AlertProcessor(autorun=False, exchange=exchange)
    updates = [("GOOG", datetime(2014, 12, 8), 5)]
    processor.do_updates(updates)
    mock_goog.update.assert_called_with(datetime(2014, 12, 8), 5)
```

## 继承和测试

实现这一目标的另一种方法是编写一个从`AlertProcessor`继承的类，但包含依赖参数。

例如，我们可以在测试文件中创建一个如下所示的类：

```py
class TestAlertProcessor(AlertProcessor):
    def __init__(self, exchange):
        AlertProcessor.__init__(self, autorun=False)
        self.exchange = exchange
```

这个类从`AlertProcessor`继承，并接受我们在特征测试中想要模拟的参数。`__init__`方法调用原始类的初始化器，然后覆盖`exchange`参数为其传递的值。

在单元测试中，我们可以实例化测试类而不是真实类。我们传递一个包含模拟股票对象的交换。模拟被设置，我们可以测试是否调用了正确的调用，如下所示：

```py
def test_processor_characterization_5(self):
    mock_goog = mock.Mock()
    mock_aapl = mock.Mock()
    exchange = {"GOOG": mock_goog, "AAPL": mock_aapl}
    processor = TestAlertProcessor(exchange)
    updates = [("GOOG", datetime(2014, 12, 8), 5)]
    processor.do_updates(updates)
    mock_goog.update.assert_called_with(datetime(2014, 12, 8), 5)
```

与通过默认参数注入依赖相比，这种方法的优势在于它不需要在原始类中更改任何代码。

## 模拟局部方法

大多数时候，我们使用模拟来代替被测试类之外的其它类或函数。例如，我们在上一节的例子中模拟了`Stock`对象，并在本章早期模拟了内置的`print`函数。

然而，Python 并不允许我们模拟测试中相同类的成员方法。这是一种测试复杂类的强大方式。

假设我们想要测试`run`方法中解析文件的代码，而不执行更新股票价值的部分。以下是一个仅执行此操作的测试：

```py
def test_processor_characterization_6(self):
    processor = AlertProcessor(autorun=False)
    processor.do_updates = mock.Mock()
    processor.run()
    processor.do_updates.assert_called_with([
        ('GOOG', datetime(2014, 2, 11, 14, 10, 22, 130000), 5),
        ('AAPL', datetime(2014, 2, 11, 0, 0), 8),
        ('GOOG', datetime(2014, 2, 11, 14, 11, 22, 130000), 3),
        ('GOOG', datetime(2014, 2, 11, 14, 12, 22, 130000), 15),
        ('AAPL', datetime(2014, 2, 11, 0, 0), 10),
        ('GOOG', datetime(2014, 2, 11, 14, 15, 22, 130000), 21)])
```

在上面的测试中，我们在执行`run`方法之前模拟了类的`do_updates`方法。当我们执行`run`时，它解析文件，然后不是运行`do_updates`局部方法，而是执行我们的模拟方法。由于实际方法被模拟，代码不会更新`Stock`或打印任何内容到屏幕上。所有这些功能都已模拟。然后我们通过检查是否将正确的参数传递给`do_updates`方法来测试解析是否正确。

模拟局部方法是一个理解更复杂类的好方法，因为它允许我们单独为类的小部分编写特征测试。

## 提取方法和存根

有时，一个方法可能相当长，我们希望模拟方法的一部分。我们可以通过将我们想要模拟的部分提取到一个局部方法中，然后在测试中模拟该方法来组合上述技术。

以下是我们当前的`run`方法的样子：

```py
def run(self):
    updates = []
    with open("updates.csv", "r") as fp:
        for line in fp.readlines():
            symbol, timestamp, price = line.split(",")
            updates.append((symbol, datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f"), int(price)))
    self.do_updates(updates)
```

假设我们希望测试跳过方法中读取和解析文件的部分。

我们首先将行提取为一个方法，如下所示：

```py
def parse_file(self):
    updates = []
    with open("updates.csv", "r") as fp:
        for line in fp.readlines():
            symbol, timestamp, price = line.split(",")
            updates.append((symbol, datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f"), int(price)))
    return updates
```

然后，我们将`run`方法中的行替换为新提取的方法调用，如下所示：

```py
def run(self):
    updates = self.parse_file()
    self.do_updates(updates)
```

最后，特征测试模拟了新提取的方法，给它一个返回值，并调用`run`方法，如下所示：

```py
def test_processor_characterization_7(self):
    processor = AlertProcessor(autorun=False)
    processor.parse_file = mock.Mock()
    processor.parse_file.return_value = [
        ('GOOG', datetime(2014, 2, 11, 14, 12, 22, 130000), 15)]
    with mock.patch("builtins.print") as mock_print:
        processor.run()
    mock_print.assert_called_with("GOOG", 15)
```

将文件解析行提取到单独的方法后，我们可以轻松地为不同的输入组合编写多个不同的特征测试。例如，以下是一个检查对于不同输入没有打印到屏幕上的另一个特征测试：

```py
def test_processor_characterization_8(self):
    processor = AlertProcessor(autorun=False)
    processor.parse_file = mock.Mock()
    processor.parse_file.return_value = [
        ('GOOG', datetime(2014, 2, 11, 14, 10, 22, 130000), 5)]
    with mock.patch("builtins.print") as mock_print:
        processor.run()
    self.assertFalse(mock_print.called)
```

我们还可以使用这项技术来测试难以访问的代码，例如`lambda`函数。我们将`lambda`函数提取到一个单独的函数中，这使得我们可以单独为它编写特征测试，或者在编写其他部分代码的测试时模拟它。

让我们对我们的代码做以下操作。首先，将`lambda`函数提取到一个局部方法中。

```py
def print_action(self, stock, rule):
    print(stock.symbol, stock.price) \
        if rule.matches(self.exchange) else None
```

然后，将`print`行替换为对方法的调用，如下所示：

```py
self.exchange["GOOG"].updated.connect(
    lambda stock: self.print_action(stock, rule_1))
self.exchange["AAPL"].updated.connect(
    lambda stock: self.print_action(stock, rule_2))
```

### 注意

注意我们的调用方式。我们仍然使用`lambda`函数，但使用适当的参数委托到局部方法。

现在，我们可以在为`do_updates`方法编写特征测试时模拟此方法：

```py
def test_processor_characterization_9(self):
    processor = AlertProcessor(autorun=False)
    processor.print_action = mock.Mock()
    processor.do_updates([
        ('GOOG', datetime(2014, 2, 11, 14, 12, 22, 130000), 15)])
    self.assertTrue(processor.print_action.called)
```

# 循环继续

上节中提到的所有技术都有助于我们隔离代码片段，并与其他类断开依赖关系。这使我们能够引入存根和模拟，从而更容易编写更详细的特征测试。提取方法重构被大量使用，是一种隔离代码小部分的好技术。

整个过程是迭代的。在典型会话中，我们可能会通过`pdb`查看一段代码，然后决定将其提取为方法。我们可能会在交互式外壳中尝试向提取的方法传递不同的输入，之后我们可能会编写一些特征测试。然后我们会回到类的另一部分，在模拟或存根新方法后编写更多测试。之后我们可能会回到`pdb`或交互式外壳，查看另一段代码。

在整个过程中，我们不断进行小的更改，这些更改不太可能破坏现有系统，并持续运行所有现有的特征测试，以确保我们没有破坏任何东西。

# 重构时间

过了一段时间，我们可能会为遗留代码获得一套相当不错的特征测试。现在我们可以像对待任何经过良好测试的代码一样处理这段代码，并开始应用更大的重构，目的是在添加新功能之前改进设计。

例如，我们可能会决定将`print_action`方法提取到一个单独的`Action`类中，或者将`parse_file`方法提取到一个`Reader`类中。

以下是一个`FileReader`类，我们将内容从`parse_file`局部方法移动到这里：

```py
class FileReader:
    def __init__(self, filename):
        self.filename = filename

    def get_updates(self):
        updates = []
        with open("updates.csv", "r") as fp:
            for line in fp.readlines():
                symbol, timestamp, price = line.split(",")
                updates.append((symbol, datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f"), int(price)))
        return updates
```

然后我们使用注入依赖模式将`reader`作为参数传递给构造函数：

```py
def __init__(self, autorun=True, reader=None, exchange=None):
    self.reader = reader if reader else FileReader("updates.csv")
    if exchange is None:
        self.exchange = {"GOOG": Stock("GOOG"), "AAPL": Stock("AAPL")}
    else:
        self.exchange = exchange
    rule_1 = PriceRule("GOOG", lambda stock: stock.price > 10)
    rule_2 = PriceRule("AAPL", lambda stock: stock.price > 5)
    self.exchange["GOOG"].updated.connect(
        lambda stock: self.print_action(stock, rule_1))
    self.exchange["AAPL"].updated.connect(
        lambda stock: self.print_action(stock, rule_2))
    if autorun:
        self.run()
```

然后将`run`方法更改为调用读取器：

```py
def run(self):
    updates = self.reader.get_updates()
    self.do_updates(updates)
```

注意我们是如何设置默认值的，这样使用这个类的其他类就不需要更改。这允许我们在测试以及新代码中覆盖`reader`参数，而现有代码在无需更改的情况下也能正常工作。

我们现在可以通过向构造函数传递一个模拟对象来编写这个测试：

```py
def test_processor_gets_values_from_reader(self):
    mock_reader = mock.MagicMock()
    mock_reader.get_updates.return_value = \
        [('GOOG', datetime(2014, 2, 11, 14, 12, 22, 130000), 15)]
    processor = AlertProcessor(autorun=False, reader=mock_reader)
    processor.print_action = mock.Mock()
    processor.run()
    self.assertTrue(processor.print_action.called)
```

我们可以通过将`print_action`方法提取到`Action`类中并将其作为参数传递来实现同样的操作。

记得我们的原始目标吗？

在本章一开始，我们就说过我们想要实现以下两个功能：

+   我们需要能够从网络服务器获取更新

+   当警报匹配时，我们需要能够发送电子邮件

原始设计没有使添加此功能变得容易，我们可能需要对该代码进行一些修改——这是一个危险且容易出错的方案。

我们新重构的设计现在使得添加这些功能变得容易。我们只需要创建新的类，比如创建一个名为`NetworkReader`的类，它从服务器读取输入。我们通过读取器参数将这个对象的实例传递给初始化器。`AlertProcessor`将随后从服务器获取更新。

我们可以通过实现一个`EmailAction`类并将该对象传递给这个类来完成同样的操作。

## 长期重构

我们已经成功地将新功能安全地添加到我们的遗留代码中。但我们的工作还没有结束。我们在`__init__`方法中添加了一些默认参数，以便不破坏使用此类的现有代码。随着时间的推移，我们希望逐一访问这些地方，并将它们更改为使用新接口。一旦我们更改了所有地方，我们就可以从接口中删除默认参数，整个代码库就会迁移到新接口。

关于这一点很酷的是，我们不必一次性完成所有更改。代码库永远不会长时间处于破损状态。我们可以逐步进行这些更改，一次一个，应用程序在每一个点上始终是正确运行的。

我们还需要做的一件事是回到我们的特征测试，并对它们进行清理。还记得我们写的第一个特征测试吗？它如下所示：

```py
import unittest
from unittest import mock

from ..legacy import AlertProcessor

class AlertProcessorTest(unittest.TestCase):
    @mock.patch("builtins.print")
    def test_processor_characterization_1(self, mock_print):
        AlertProcessor()
        mock_print.assert_has_calls([mock.call("AAPL", 8),
                                     mock.call("GOOG", 15),
                                     mock.call("AAPL", 10),
                                     mock.call("GOOG", 21)])
```

在本章的开头，我们提到这并不是一个很好的单元测试，但作为一个特征测试来说已经足够好了。现在，是时候重新审视这个测试，并使其变得更好。经过重构设计后，我们现在可以向读者传递一个模拟对象。测试将不再依赖于`updates.csv`文件的存在。

我们还进行了一些测试，其中我们修补了打印函数。一旦我们将设计重构为接受`Action`类作为输入，我们就不再需要修补这个函数，因为我们可以直接传递一个模拟动作对象到初始化器。

# 摘要

在本章中，你看到了如何处理与遗留代码一起工作的棘手问题。我们将遗留代码定义为任何不包含测试的代码。我们必须处理这种代码是不幸的事实。幸运的是，有一些技术可以让我们安全地处理这种代码。交互式外壳以及极其强大的调试器在理解典型的乱麻代码方面提供了巨大的帮助。

Python 的动态特性也使得打破依赖变得容易。我们可以在重构到更好的设计的同时，使用默认值参数来保持与现有代码的兼容性。强大的修补功能和动态更改现有实例变量和局部方法的能力，使我们能够编写出通常会更困难的特征测试。

现在你已经看到了许多编写测试的方法，让我们来看看如何保持一切可维护。我们将在下一章中这样做。
