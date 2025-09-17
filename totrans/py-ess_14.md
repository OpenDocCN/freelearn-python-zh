# 第十四章。完善与测试 - 单元测试、打包和文档

除了 Python 语言及其库之外，Python 编程还有其他几个方面。我们将首先仔细研究文档字符串，它们应被视为每个包、模块、类和函数定义的基本组成部分。它们有几个目的，其中之一是阐明对象的功能。

在本章中，我们还将探讨单元测试的不同方法。`doctest`和`unittest`模块提供了一套全面的工具。外部工具如 Nose 也被广泛使用。

我们还将探讨如何利用`logging`模块作为完整应用程序的一部分。Python 的记录器也非常复杂，因此我们将关注一些基本功能。

我们将检查一些用于从嵌入的文档字符串注释构建 Python 文档的工具。使用工具提取文档使我们能够专注于编写正确的代码，并从代码中派生出参考文档。为了创建完整的文档——而不仅仅是 API 参考——许多开发者使用 Sphinx 工具。

我们还将讨论在大型 Python 项目中文件的组织结构。由于 Python 被用于许多不同的环境和框架，使用 Flask 构建的 Web 应用程序的布局将与使用 Django 构建的 Web 应用程序大相径庭。然而，我们可以遵循一些基本原则，以保持 Python 程序整洁和井然有序。

# 编写文档字符串

在第七章，*基本函数定义*中，我们指出所有函数都应该有一个描述函数的文档字符串。在第十一章，*类定义*和第十二章，*脚本、模块、包、库和应用程序*中，我们提供了类似的建议，但没有提供很多细节。

`def`语句和`class`语句应该普遍地后面跟着一个三引号字符串，描述函数、方法或类。这不是语言的要求——这是所有试图阅读、理解、扩展、改进或修复我们代码的人的要求。

我们将回顾第十一章中的一个示例，*类定义*，以展示省略的文档字符串类型。以下是我们可能创建的更完整的类定义示例：

```py
class Point:
    """
    Point on a plane.

    Distances are calculated using hypotenuse.
    This is the "as a crow flies" straight line distance.

    Point on a plane.

    Distances are calculated using hypotenuse.
    This is the "as a crow flies" straight line distance.

    >>> p_1 = Point(22, 7)
    >>> p_1.x
    22
    >>> p_1.y
    7
    >>> p_1
    Point(22, 7)
    """
    def __init__(self, x, y):
        """Create a new point

        :param x: X coördinate
        :param y: Y coördinate
        """
        self.x= x
        self.y= y
    def __repr__(self):
        """Returns string representation of this Point."""
        return "{cls}({x:.0f}, {y:.0f})".format(
            cls=self.__class__.__name__, x=self.x, y=self.y)
    def dist(self, point):
        """Distance to another point measured on a plane.

        >>> p_1 = Point(22, 7)
        >>> p_2 = Point(20, 5)
        >>> round(p_1.dist(p_2),4)
        2.8284

        :param point: Another instance of Point.
        :returns: float distance.
        """
        return math.hypot(self.x-point.x, self.y-point.y)
```

在这个类定义中，我们提供了四个独立的文档字符串。对于整个类，我们提供了一个概述，说明了类的作用，以及一个展示类行为的示例。这显示了从 Python 交互式解释器（REPL）复制粘贴的内容，输入前带有`>>>`提示符。

对于每个方法函数，我们提供了一个文档字符串，显示了方法函数的作用。在 `dist()` 方法的例子中，我们在文档字符串中包含了另一个交互示例，以展示该方法预期行为的示例。

参数和返回值的文档使用 **ReStructuredText** （**RST**） 标记语言。这因其工具如 `docutils` 和 Sphinx 而被广泛使用，这些工具可以将 RST 格式化为漂亮的 HTML 或 LaTeX。我们将在本章后面的 *使用 RST 标记编写文档* 部分查看 RST。

现在，我们可以关注 `:param name:` 和 `:returns:` 作为帮助工具理解这些构造语义的标记语法。然后，工具可以给予它们特殊的格式化以反映其含义。

# 使用 doctest 编写单元测试

在文档字符串中提供类和函数的具体示例是一种广泛采用的实践。正如前面示例所示，我们可以在文档字符串中提供以下类型的示例文本：

```py
>>> p_1 = Point(22, 7)
>>> p_2 = Point(20, 5)
>>> round(p_1.dist(p_2),4)
2.8284
```

具体示例有许多好处。Python 代码的目标是美观和可读。如果代码示例晦涩或令人困惑，这是一个真正应该解决的问题的设计问题。在注释中写更多文字来尝试解释糟糕的代码是更深层次问题的症状。具体示例应该像代码本身一样清晰和富有表现力。

具体示例的另一个好处是它们是测试用例。`doctest` 模块可以扫描每个文档字符串以定位这些示例，构建并执行测试用例。这将确认示例中的输出与实际输出相匹配。

使用 `doctest` 的一个常见方法是在 `library` 模块中包含以下内容：

```py
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=1)
```

如果模块作为主脚本执行而不是被导入，那么它将导入 `doctest`，扫描模块以查找文档字符串，并执行这些文档字符串中的所有测试。我们已将详细级别设置为一级，这将产生显示测试的详细输出。如果我们保留详细级别为其默认值零，则成功将是静默的；甚至不会显示 `Ok`。

我们也可以将 `doctest` 作为命令行应用程序运行。以下是一个示例：

```py
MacBookPro-SLott:Code slott$ python3 -m doctest Chapter_1/ch01_ex1.py -v
Trying:
 355/113
Expecting:
 3.1415929203539825
ok
...
1 items had no tests:
 ch01_ex1
9 items passed all tests:
 2 tests in __main__.__test__.assignment
 4 tests in __main__.__test__.cmath
 2 tests in __main__.__test__.division
 1 tests in __main__.__test__.expression
 3 tests in __main__.__test__.import 1
 1 tests in __main__.__test__.import 2
 2 tests in __main__.__test__.import 3
 2 tests in __main__.__test__.mixed_math
 2 tests in __main__.__test__.print
19 tests in 10 items.
19 passed and 0 failed.
Test passed.

```

我们已将 `doctest` 模块作为应用程序运行，向其提供要检查以定位文档字符串中测试示例的文件名。输出从找到的第一个示例开始。该示例是：

```py
>>> 355/113
3.1415929203539825
```

详细输出显示了表达式和预期结果。`ok` 的输出表示测试通过。

那么那个没有测试的项目呢？这是模块本身的文档字符串。这表明我们的测试用例覆盖率不完整。我们应该考虑在模块文档字符串中添加一个测试。

总结显示有 9 个项目有 19 个测试。这些项目用字符串如 `ch01_ex1.__test__.assignment` 来标识。特殊名称 `__test__` 既不是函数也不是类；它是一个全局变量。如果存在名为 `__test__` 的变量，它必须是一个字典。`__test__` 字典中的键是文档，值是必须包含 `doctest` 示例的字符串。

`__test__` 变量可能看起来像这样：

```py
__test__ = {
    'expression': """
        >>> 355/113
        3.1415929203539825
    """,
*etc.*
}
```

每个键标识一个测试。每个值是一个包含预期结果的 REPL 交互片段的三引号字符串。

作为一项实际措施，这个特定的测试受到了 `doctest` 示例潜在局限性之一的影响。

正如我们在 第五章 中提到的，*逻辑、比较和条件*，我们不应该在浮点值之间使用精确相等测试。编写这种测试的正确方式是使用 `round(355/113, 6)` 来截断尾数；最终数字可能会因硬件或底层浮点库而略有不同。编写与实现细节无关的测试会更好。

`doctest` 示例存在一些潜在的局限性。字典键没有定义的顺序。因此，当键以与测试中预期输出不同的顺序显示时，`doctest` 可能会失败。同样，集合项也没有定义的顺序。此外，错误跟踪消息可能不会精确匹配，因为它将包含类似 `File "<stdin>", line 1, in <module>` 的行，这可能会根据测试运行的上下文而变化。

对于这些潜在的局限性，`doctest` 提供了可以用来注释测试的指令。这些指令以特殊的注释形式出现，例如：`# doctest: +ELLIPSIS`。这将启用灵活的模式匹配，以应对显示输出的变化。对于其他局限性，我们需要正确构建我们的测试用例。我们可以使用 `sorted(some_dict.values())` 将字典结果转换为有序的元组列表，其中顺序是有保证的。

Docstrings 是良好 Python 编程的一个基本特性。示例是良好文档的一个基本特性。给定一个可以验证示例正确性的工具，这种测试应该被认为是强制性的。

# 使用 unittest 库进行测试

对于更复杂的测试，`doctest` 示例可能不足以提供深度或灵活性。包含大量案例的文档字符串可能太长，无法作为有效的文档。包含复杂测试设置、拆卸或模拟对象的文档字符串也可能不适用于文档。

对于这些情况，我们将使用 `unittest` 模块来定义测试用例和测试套件。当使用 `unittest` 时，我们通常会创建单独的模块。这些测试模块将包含包含测试方法的 `TestCase` 类。

下面是一个典型的测试用例类定义的快速概述：

```py
import unittest

from Chapter_7.ch07_ex1 import FtoC

class Test_FtoC(unittest.TestCase):
    def setUp(self):
        self.temps= [50, 60, 72]
    def test_single(self):
        self.assertAlmostEqual(0.0, FtoC(32))
        self.assertAlmostEqual(100.0, FtoC(212))
    def test_map(self):
        temps_c = list(map(FtoC, self.temps))
        self.assertEqual(3, len(temps_c))
        rounded = [round(t,3) for t in temps_c]
        self.assertEqual([10.0, 15.556, 22.222], rounded)
```

我们已经展示了 `setUp()` 方法以及两个测试方法。默认的 `runTest()` 方法将搜索所有以 `test` 开头名称的方法；然后运行在各个 `test...` 方法之前执行的 `setUp()` 方法。

我们可以使用 Python 的 `assert` 语句来比较实际和预期结果。因为有很多常见的比较，`TestCase` 类提供了方便的方法来比较不同类型的预期结果与实际结果。我们已经展示了 `assertEqual()` 和 `assertAlmostEqual()`。这些方法中的每一个都与 `assert` 语句平行——它们无声地成功。如果有问题，它们将引发 `AssertionError` 异常。

使用 `unittest` 模块允许我们编写大量的测试用例。`doctest` 字符串在表达几个有用的具体示例时最有用。单元测试是包含许多边缘和角落情况的更好方式。

`unittest` 模块也便于测试涉及与文件系统交互的示例。我们可能有一个包含多个示例的 `.csv` 格式文件。我们可以编写一个 `runTest()` 方法来读取这个文件，并将每一行作为测试用例处理。

在追求**验收测试驱动开发**（**ATDD**）时，测试用例本身可能变得相当复杂。测试用例设置可能涉及在执行大型应用程序功能之前用样本数据填充数据库，然后检查结果数据库的内容。ATDD 测试的基本结构符合 `unittest` 模块提供的单元测试设计模式。被测试的“单元”不是一个孤立的类；相反，我们正在测试一个完整的 Web API 或命令行应用程序。

## 结合 doctest 和 unittest

我们可以将 `doctest` 测试用例纳入 `unittest` 测试套件中。这确保了在使用 `unittest` 测试用例时不会遗漏 `doctest` 示例。我们将通过使用可以包含其他 `TestCase` 类以及 `TestSuite` 类的 `TestSuite` 类来实现这一点。

一个 `doctest.DocTestSuite` 对象将从给定模块中嵌入的 `doctest` 字符串创建一个合适的 `unittest.TestSuite` 方法。我们可以使用以下类似的功能来定位大量包和模块中的所有测试用例：

```py
def doctest_suite():
    files = glob.glob("Chapter*/ch*_ex*.py")
    by_chxx= lambda name: name.partition(".")[2].partition("_")[0]
    modules = sorted(
        (".".join(f.replace(".py","").split(os.sep)) for f in files),
        key=by_chxx)
    suites= [doctest.DocTestSuite(m) for m in modules]
    return unittest.TestSuite(suites)
```

这个函数将返回一个由其他 `TestSuite` 对象构建的 `TestSuite` 对象。这个函数有五个步骤：

1.  它使用 `glob.glob()` 来获取包中所有匹配的模块名称列表。这个特定的模式将定位到本书的所有示例代码。我们可能需要更改这个模式以通过或拒绝其他可能存在的名称。

1.  它定义了一个 lambda 对象，该对象从模块中提取章节号，忽略包。该表达式使用 `name.partition(".")` 将完整的模块名称拆分为包、点字符和模块名称。序列中的第 2 项是模块名称。这是在 `"_"` 上分割的，包括章节前缀、下划线和示例后缀。我们使用序列中的第 0 项，即章节前缀，作为模块的排序顺序。

1.  `sorted()` 函数的输入是一个重新结构化为模块名称的文件名序列。这种转换涉及替换 `".py"` 文件名后缀，然后在操作系统路径分隔符（在大多数操作系统中是 "/"，但在 Windows 中是 "\") 上分割文件名，得到单独的子字符串。当我们使用 "." 连接这些子字符串时，我们得到一个模块名称，我们可以用它来进行排序和测试用例发现。

1.  我们构建了一个列表推导式，该推导式可以构建每个模块中的 doctest 示例可以构建的测试套件。这包括从本书的示例中提取的超过 100 个单独的测试。

1.  我们从测试套件列表中组装一个单独的测试套件。然后可以执行它以确认所有示例都产生预期的结果。

我们可以将这个 doctest `TestSuite` 对象与从基于 `unittest.TestCase` 定义的测试构建的 `TestSuite` 对象合并。然后，这个完整的测试套件可以执行以证明代码按预期工作。

我们经常使用以下类似的方法：

```py
if __name__ == "__main__":
    runner= unittest.TextTestRunner( verbosity=1 )
    all_tests = unittest.TestSuite( suite() )
    runner.run( all_tests )
```

这将创建一个测试运行器，它可以生成测试和测试失败的摘要。`suite()` 函数（未显示）返回一个由 `doctest_suite()` 函数和一个扫描文件以查找 `unittest.TestCase` 类的函数构建的 `TestSuite()` 方法。

输出总结了运行的测试和失败情况。当我们构建这样的综合测试套件时，我们包括 `unittest` 和 `doctest` 测试用例。这允许我们自由地混合复杂的测试套件和简单的文档字符串示例。

# 使用其他附加测试库

`doctest` 和 `unittest` 模块允许我们方便地编写单元测试。在许多情况下，我们希望有更多的复杂性。更受欢迎的附加功能之一是测试发现。`nose` 包为我们提供了一种轻松检查模块和包以查找测试的方法。有关更多信息，请参阅 [`nose.readthedocs.org/en/latest/`](http://nose.readthedocs.org/en/latest/)。

使用 `nose` 作为 `unittest` 的扩展有几个好处。`nose` 模块可以从 `unittest.TestCase` 子类、简单的测试函数以及不是 `unittest.TestCase` 子类的测试类中收集测试。我们还可以使用 `nose` 来编写计时测试——这在 `unittest` 中可能有点尴尬。

由于`nose`特别擅长自动收集测试，因此无需手动将测试用例收集到测试套件中；我们不需要之前展示的一些示例。此外，`nose`支持在包、模块和类级别上使用测试固定装置，因此可以尽可能少地执行昂贵的初始化。这使得我们可以为多个相关测试模块填充测试数据库——这是`unittest`难以轻松做到的。

# 记录事件和条件

一个表现良好的应用程序可以生成各种处理摘要。对于命令行应用程序，摘要可能是一条简单的“一切正常”的消息。对于图形用户界面应用程序，这种摘要正好相反——沉默意味着一切正常，而带有错误消息的对话框则表明事情没有按预期进行。

在某些命令行处理上下文中，摘要可能包括一些关于处理的对象数量的额外细节。在金融应用程序中，一些计数和不同对象的总额必须正确平衡，以显示所有接收到的输入对象都变成了适当的输出。

当我们需要比简单的“工作或失败”摘要更多的详细信息时，我们可以利用`print()`函数。输出可以重定向到`sys.stderr`文件以生成一个方便的日志。虽然这在小型程序中很有效，但`logging`模块提供了许多期望的特性。

使用`logging`模块的第一步是创建记录器对象并使用记录器生成有用的输出。每个记录器都有一个名称，该名称使用`.`字符分隔，并适合到一个树结构中。记录器名称与模块名称的标准平行；我们可以使用以下方法：

```py
import logging
logger = logging.getLogger(__name__)
```

这将创建一个与模块名称匹配的模块级`logger`对象。根记录器具有名称`""`；即一个空字符串。

我们还可以创建类级别的记录器以及对象特定的记录器。例如，我们可以在对象创建的`__init__()`方法部分创建一个记录器。我们可能会使用对象类的`__qualname__`属性为记录器提供有资格的类名。要为类的特定实例创建记录器，我们可以在类名后缀一个`.`字符和一些唯一的实例标识符。

我们使用记录器创建具有从`DEBUGGING`（最不严重）到`FATAL`或`CRITICAL`（最严重级别的同义词）严重级别的消息。我们通过反映严重级别的名称来执行此操作。使用如下方法创建消息：

```py
logger.debug("Finished with {0} using {2}".format(message, details))
logger.error("Error due to {0}".format(data))
```

`logging`模块有一个默认配置，不执行任何操作。这意味着我们可以在应用程序中包含日志请求而不需要进一步考虑。只要我们正确创建`Logger`实例并使用记录器实例的方法，我们就不需要做任何事情。

要查看输出，我们需要创建一个处理器，将消息写入特定的流或文件。这通常作为日志系统整体配置的一部分来完成。

## 配置日志系统

我们有几种配置日志系统的方式。对于小型应用程序，我们可能会使用`logging.basicConfig()`函数来提供日志设置。这在第十三章 *元编程和装饰器* 中已经展示过。简单的初始化会将输出发送到标准错误流，并显式设置一个级别来过滤显示的消息。这使用了`stream`和`level`关键字参数。

一个稍微复杂一些的配置可能看起来像这样：

```py
logging.basicConfig(filename='app.log', filemode='a', level=logging.INFO)
```

我们已经打开了一个命名的文件，将其模式设置为`a`以追加，并将级别设置为显示严重性等于或大于`INFO`的消息。

由于每个单独的日志记录器都有名称，我们可以调整特定日志记录器的详细程度。我们可以包含以下类似行来在特定日志记录器上启用调试：

```py
logging.getLogger('Demonstration').setLevel(logging.DEBUG)
```

这允许我们查看特定类或模块的详细信息。这在调试时通常非常有帮助。

`logging.handlers`模块提供了大量用于路由、打印或保存日志消息序列的处理程序。前面的示例显示了文件处理器。流处理器用于写入标准错误流。在某些情况下，我们需要有多个处理器。我们可以对每个处理器应用过滤器，这样处理器将反映不同类型的细节。

日志配置通常过于复杂，无法使用`basicConfig()`函数。`logging.config`模块提供了几个可用于配置应用程序日志的函数。一种通用方法是用`logging.config.dictConfig()`函数。我们可以在 Python 中直接创建 Python `dict`对象，或者读取`dict`对象的某些序列化版本。标准库文档使用 YAML 标记语言编写的示例，因为它简单且灵活。

我们可能像这样创建一个配置对象：

```py
config = {
    'version': 1,
    'handlers': {
        'console': {
            'class' : 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
        }
    },
    'root': {
        'level': 'DEBUG',
        'handler': ['console'],
    },
}
```

此对象具有所需的`version`属性，用于指定配置的结构。定义了一个单个处理器；它被命名为`console`，并使用`logging.StreamHandler`写入标准错误流。根日志记录器配置为使用`console`处理器。严重级别被定义为包括任何`DEBUG`级别或以上的消息。

只有在配置文件中，根日志记录器才被命名为`'root'`。在应用程序代码中，根日志记录器使用空字符串命名。

较大且更复杂的应用程序将依赖于外部配置文件中的日志配置。这允许灵活且复杂的日志配置。

# 使用 RST 标记编写文档

虽然 Python 代码应该是美观且富有信息的，但它并不容易提供背景或上下文来展示为什么选择特定的算法或数据结构。我们经常需要提供这些额外的细节来帮助人们维护、扩展并有效地使用我们的软件。虽然我们可以在模块文档字符串中包含大量信息，但似乎最好将文档字符串集中在实现细节上，并单独提供额外的材料。

我们可以用各种格式编写额外的文档。我们可以使用具有复杂文件格式的复杂编辑器，或者我们可以使用简单的文本编辑器和纯文本格式。我们甚至可以完全用 HTML 编写我们的文档。Python 还提供了一种混合方法——我们可以使用带有简化**ReStructuredText**（**RST**）标记的文本编辑器来编写，并使用`docutils`工具从该标记创建漂亮的 HTML 页面或适合出版的 LaTeX 文件。

RST 标记语言被广泛用于创建 Python 文档。这种标记允许我们编写纯文本，同时遵守一些格式规则。在下一节中，我们将探讨使用`docutils`工具解析 RST 并创建输出文档。

RST 标记的规则很简单。存在段落级别的标记，适用于大块文本。段落必须由空白行分隔。当一行被字符序列“下划”时，它被视为标题。当一个段落以一个独立的标点符号开头时，它是一个项目符号。当一个段落以字母或数字开头，后面跟着一个标点符号时，这表示数字而不是项目符号。`docutils`的`rst2html.py`工具将输入的每个段落转换为适当的 HTML 结构。

有许多段落级别的“指令”可以用来插入图像、表格、方程式或大块代码。这些指令以前缀`..`开头，以`::`结尾。我们可能使用指令`.. contents::`来将目录添加到我们的文档中。

我们可以在段落的主体内编写内联标记。内联标记包括一些简单的结构。如果我们用`*`字符包围一个单词，如`*this*`，我们将在最终文档中看到以*斜体*风格的字体显示的单词；我们可以使用`**bold**`来表示**粗体**字符。如果我们想在不混淆工具的情况下写入`*`字符，我们可以用`\`字符来转义它。然而，在许多情况下，我们需要使用更复杂的语义标记，如下所示：`` :code:`code sample` ``。这包括文本角色`:code:`作为前缀，显示如何分类标记的字符；内容被`` ` ``字符包围。`:code:``和`:math:`的文本角色被广泛使用。

当我们编写文档字符串时，我们通常会使用额外的 RST 标记。当我们定义函数或类方法的参数时，我们会使用`:param name:`。我们使用`:returns:`来注释函数的返回值。当我们提供这些额外的标记时，我们可以确保各种格式化工具能够从我们的文档字符串中生成优雅的文档。

下面是一个 RST 文件可能包含的示例：

```py
Writing RST Documentation
==========================

For more information, see http://docutils.sourceforge.net/docs/user/rst/quickref.html

1\.  Separate paragraphs with blank lines.

2\.  Underline headings.

#.  Prefix with one character for an unordered list. Otherwise it may be
    interpreted as an ordered list.

#.  Indent freely to show structure.

#.  Inline markup.

    -   Use ``*word*`` for *italics*, and ``**word**`` for **bold**.

    -   Use ``:code:\`word\```以获取更复杂的语义标记。

```py

We've shown a heading, underlined with a sequence of `=` characters. We've provided a URL; in the final HTML output, this will become a proper link using the `<a>` tag. We've shown numbered paragraphs. When we omit the leading number and use `#`, the `docutils` tools will assign increasing numbers. We've also shown indented bullet point within the last numbered paragraph.

While this example shows numbering and simple hyphen bullets, we can use lettering or Roman numerals as well. The `docutils` tools are generally able to parse a wide variety of formatting conventions.

## Creating HTML documentation from an RST source

To create HTML or LaTeX (or any of the other supported formats), we'll use one of the `docutils` frontend tools. There are many individual conversion tools that are part of the `docutils` package.

The `docutils` tools are not part of Python. See [`docutils.sourceforge.net`](http://docutils.sourceforge.net) for the download.

All of the tools have a similar command-line interface. We might use the following command to create an HTML page from some RST input:

```

MacBookPro-SLott:Chapter_14 slott$ rst2html.py ch14_doc.rst ch14_doc.rst.html

```

我们提供了`rst2html.py`命令。我们已命名了输入文件和输出文件。这将使用默认的样式表值，并为生成的文档提供其他可选功能。我们可以通过命令行或提供配置文件来配置输出，以确保所有生成的 HTML 文件具有统一的样式。

要创建 LaTeX，我们可以使用`rst2latex.py`或`rst2xetex.py`工具，然后使用 LaTeX 格式化器。TeX Live 发行版非常适合从 LaTeX 创建 PDF 文件。请参阅[`www.tug.org/texlive/`](https://www.tug.org/texlive/)。

对于大型且复杂的文档，创建单个 RST 文件并不是最佳选择。虽然我们可以使用`.. include::`指令从单独的文件中插入内容，但文档必须作为一个整体来构建，这需要大量的内存；在文档进行小幅度修改后重新构建可能需要不成比例的处理量。

对于多页网站，我们必须使用像 Make、Ant 或 SCons 这样的工具来在源 RST 文件更新后重新构建相关的 HTML 页面。这种开销呼唤一个工具来自动化和简化大型或复杂文档的生产。

## 使用 Sphinx 工具

Sphinx 工具使我们能够轻松构建多页网站或复杂文档。有关更多信息，请参阅[`sphinx-doc.org`](http://sphinx-doc.org)。当我们使用`pip`或`easy_install`安装 Sphinx 时，安装程序也会为我们包括`docutils`。

要创建复杂的文档，我们将从`sphinx-quickstart`脚本开始。此应用程序将构建模板文件结构、配置文件以及一个 Makefile，我们可以使用它来高效地重新构建我们的文档。

Sphinx 在 RST 的基本指令和文本角色中添加了大量指令。这些额外的角色和指令使得编写关于代码的内容变得更加容易，可以正确地引用模块、类和函数。Sphinx 简化了文档间的引用——我们可以拥有多个文档，它们对目标位置的引用保持一致；我们可以移动目标，所有引用都将自动更新。

使用`sphinx-build`命令从 RST 源文件构建目标文件。Sphinx 可以构建十几种不同类型的目标文档，使其成为一个多功能的工具。

Python 文档是用 Sphinx 构建的。这意味着我们的项目可以包含看起来像 Python 文档一样光鲜和优雅的文档。

# 组织 Python 代码

Python 程序应该是美丽的。为此，语言有很少的语法开销；我们应该能够编写简短的脚本而不需要不愉快的模板代码。这个原则有时被阐述为“简单的事情应该简单”。“Hello World”脚本实际上是一行代码，它使用了`print()`函数。

一个更复杂的文件通常有几个主要部分：

+   一行`!#`，通常是`#!/usr/bin/env python3`。

+   一个注释文档字符串，解释模块的功能。

+   函数或类的定义。我们通常将多个函数和类组合成一个模块。在 Python 中，模块是重用的适当单元。

+   如果模块可以作为主脚本运行，我们将包括一个`if __name__ == "__main__":`部分，该部分定义了文件作为主脚本运行时的行为。

许多应用程序对于单个文件来说过于复杂。在设计较大的应用程序时，Python 的理想是尽可能保持结果结构尽可能扁平。虽然语言支持嵌套包，但深度嵌套并不被视为理想。在第十二章中，我们探讨了定义模块和包的细节。

# 摘要

在本章中，我们探讨了几个光鲜和完整的 Python 项目的特性。工作代码最重要的特性是一套单元测试，它证明了代码是有效的。没有测试用例的代码根本无法信任。为了使用任何软件，我们必须有显示软件是可信的测试。

我们已经探讨了在文档字符串中包含测试。`doctest`工具可以定位这些测试并执行它们。我们已经探讨了创建`unittest.TestCase`类。我们可以将两者结合到一个脚本中，该脚本将定位所有`doctest`和`unittest`测试用例到一个单独的主测试套件中。

软件的一个其他特点是关于如何安装和使用软件的解释。这可能只是一个提供基本信息的`README`文件。然而，通常我们需要一个更复杂的文档，它提供了各种附加信息。我们可能希望提供背景、设计背景或太大而无法打包到模块或类文档字符串中的示例。我们通常会使用超出 Python 基本组件的工具来编写文档。

在第十五章，“下一步”，我们将探讨 Python 探索的下一步。一旦我们掌握了基础知识，我们需要加深与我们需要解决的问题相关的领域的深度。我们可能想要研究大数据应用、Web 应用或游戏开发。这些更专业化的领域将涉及额外的 Python 概念、工具和框架。
