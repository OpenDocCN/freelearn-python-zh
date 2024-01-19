# 将 unittest 套件转换为 pytest

在上一章中，我们已经看到了灵活的 pytest 架构如何创建了丰富的插件生态系统，拥有数百个可用的插件。我们学习了如何轻松找到和安装插件，并概述了一些有趣的插件。

现在您已经熟练掌握 pytest，您可能会遇到这样的情况，即您有一个或多个基于`unittest`的测试套件，并且希望开始使用 pytest 进行测试。在本章中，我们将讨论从简单的测试套件开始做到这一点的最佳方法，这可能需要很少或根本不需要修改，到包含多年来有机地增长的各种自定义的大型内部测试套件。本章中的大多数提示和建议都来自于我在 ESSS（[`wwww.esss.co`](https://www.esss.co)）工作时迁移我们庞大的`unittest`风格测试套件的经验。

以下是本章将涵盖的内容：

+   使用 pytest 作为测试运行器

+   使用`unittest2pytest`转换断言

+   处理设置和拆卸

+   管理测试层次结构

+   重构测试工具

+   迁移策略

# 使用 pytest 作为测试运行器

令人惊讶的是，许多人不知道的一件事是，pytest 可以直接运行`unittest`套件，无需任何修改。

例如：

```py
class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.filepath = cls.temp_dir / "data.csv"
        cls.filepath.write_text(DATA.strip())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        self.grids = list(iter_grids_from_csv(self.filepath))

    def test_read_properties(self):
        self.assertEqual(self.grids[0], GridData("Main Grid", 48, 44))
        self.assertEqual(self.grids[1], GridData("2nd Grid", 24, 21))
        self.assertEqual(self.grids[2], GridData("3rd Grid", 24, 48))

    def test_invalid_path(self):
        with self.assertRaises(IOError):
            list(iter_grids_from_csv(Path("invalid file")))

    @unittest.expectedFailure
    def test_write_properties(self):
        self.fail("not implemented yet")
```

我们可以使用`unittest`运行器来运行这个：

```py
..x
----------------------------------------------------------------------
Ran 3 tests in 0.005s

OK (expected failures=1)
```

但很酷的是，pytest 也可以在不进行任何修改的情况下运行此测试：

```py
λ pytest test_simple.py
======================== test session starts ========================
...
collected 3 items

test_simple.py ..x                                             [100%]

================ 2 passed, 1 xfailed in 0.11 seconds ================
```

这使得使用 pytest 作为测试运行器变得非常容易，带来了几个好处：

+   您可以使用插件，例如`pytest-xdist`，来加速测试套件。

+   您可以使用几个命令行选项：`-k`选择测试，`--pdb`在错误时跳转到调试器，`--lf`仅运行上次失败的测试，等等。

+   您可以停止编写`self.assert*`方法，改用普通的`assert`。 pytest 将愉快地提供丰富的失败信息，即使对于基于`unittest`的子类也是如此。

为了完整起见，以下是直接支持的`unittest`习语和功能：

+   `setUp`和`tearDown`用于函数级`setup`/`teardown`

+   `setUpClass`和`tearDownClass`用于类级`setup`/`teardown`

+   `setUpModule`和`tearDownModule`用于模块级`setup`/`teardown`

+   `skip`，`skipIf`，`skipUnless`和`expectedFailure`装饰器，用于函数和类

+   `TestCase.skipTest`用于在测试内部进行命令式跳过

目前不支持以下习语：

+   `load_tests protocol`：此协议允许用户完全自定义从模块加载哪些测试（[`docs.python.org/3/library/unittest.html#load-tests-protocol`](https://docs.python.org/3/library/unittest.html#load-tests-protocol)）。 pytest 使用的集合概念与`load_tests`协议的工作方式不兼容，因此 pytest 核心团队没有计划支持此功能（如果您对细节感兴趣，请参见`#992`（[`github.com/pytest-dev/pytest/issues/992`](https://github.com/pytest-dev/pytest/issues/992)）问题）。

+   `subtests`：使用此功能的测试可以在同一测试方法内报告多个失败（[`docs.python.org/3/library/unittest.html#distinguishing-test-iterations-using-subtests`](https://docs.python.org/3/library/unittest.html#distinguishing-test-iterations-using-subtests)）。此功能类似于 pytest 自己的参数化支持，不同之处在于测试结果可以在运行时而不是在收集时确定。理论上，这可以由 pytest 支持，该功能目前正在通过问题`#1367`（[`github.com/pytest-dev/pytest/issues/1367`](https://github.com/pytest-dev/pytest/issues/1367)）进行跟踪。

**`pytest-xdist`的惊喜**

如果您决定在测试套件中使用`pytest-xdist`，请注意它会以任意顺序运行测试：每个工作进程将在完成其他测试后运行测试，因此测试执行的顺序是不可预测的。因为默认的`unittest`运行程序会按顺序顺序运行测试，并且通常以相同的顺序运行，这将经常暴露出测试套件中的并发问题，例如，试图使用相同名称创建临时目录的测试。您应该将这视为修复潜在并发问题的机会，因为它们本来就不应该是测试套件的一部分。

# unittest 子类中的 pytest 特性

尽管不是设计为在运行基于`unittest`的测试时支持所有其特性，但是支持一些 pytest 习语：

+   **普通断言**：当子类化`unittest.TestCase`时，pytest 断言内省的工作方式与之前一样

+   **标记**：标记可以正常应用于`unittest`测试方法和类。处理标记的插件在大多数情况下应该正常工作（例如`pytest-timeout`标记）

+   **自动使用**固定装置：在模块或`conftest.py`文件中定义的自动使用固定装置将在正常执行`unittest`测试方法时创建/销毁，包括在类范围的自动使用固定装置的情况下

+   **测试选择**：命令行中的`-k`和`-m`应该像正常一样工作

其他 pytest 特性与`unittest`不兼容，特别是：

+   **固定装置**：`unittest`测试方法无法请求固定装置。Pytest 使用`unittest`自己的结果收集器来执行测试，该收集器不支持向测试函数传递参数

+   **参数化**：由于与固定装置的原因相似，这也不受支持：我们需要传递参数化值，目前这是不可能的。

不依赖于固定装置的插件可能会正常工作，例如`pytest-timeout`或`pytest-randomly`。

# 使用 unitest2pytest 转换断言

一旦您将测试运行程序更改为 pytest，您就可以利用编写普通的断言语句来代替`self.assert*`方法。

转换所有的方法调用是无聊且容易出错的，这就是[`unittest2pytest`](https://github.com/pytest-dev/unittest2pytest)工具存在的原因。它将所有的`self.assert*`方法调用转换为普通的断言，并将`self.assertRaises`调用转换为适当的 pytest 习语。

使用`pip`安装它：

```py
λ pip install unittest2pytest
```

安装完成后，您现在可以在想要的文件上执行它：

```py
λ unittest2pytest test_simple2.py
RefactoringTool: Refactored test_simple2.py
--- test_simple2.py (original)
+++ test_simple2.py (refactored)
@@ -5,6 +5,7 @@
 import unittest
 from collections import namedtuple
 from pathlib import Path
+import pytest

 DATA = """
 Main Grid,48,44
@@ -49,12 +50,12 @@
 self.grids = list(iter_grids_from_csv(self.filepath))

 def test_read_properties(self):
-        self.assertEqual(self.grids[0], GridData("Main Grid", 48, 44))
-        self.assertEqual(self.grids[1], GridData("2nd Grid", 24, 21))
-        self.assertEqual(self.grids[2], GridData("3rd Grid", 24, 48))
+        assert self.grids[0] == GridData("Main Grid", 48, 44)
+        assert self.grids[1] == GridData("2nd Grid", 24, 21)
+        assert self.grids[2] == GridData("3rd Grid", 24, 48)

 def test_invalid_path(self):
-        with self.assertRaises(IOError):
+        with pytest.raises(IOError):
 list(iter_grids_from_csv(Path("invalid file")))

 @unittest.expectedFailure
RefactoringTool: Files that need to be modified:
RefactoringTool: test_simple2.py
```

默认情况下，它不会触及文件，只会显示它可以应用的更改的差异。要实际应用更改，请传递`-wn`（`--write`和`--nobackups`）。

请注意，在上一个示例中，它正确地替换了`self.assert*`调用，`self.assertRaises`，并添加了`pytest`导入。它没有更改我们测试类的子类，因为这可能会有其他后果，具体取决于您正在使用的实际子类，因此`unittest2pytest`会保持不变。

更新后的文件运行方式与以前一样：

```py
λ pytest test_simple2.py
======================== test session starts ========================
...
collected 3 items

test_simple2.py ..x                                            [100%]

================ 2 passed, 1 xfailed in 0.10 seconds ================
```

采用 pytest 作为运行程序，并能够使用普通的断言语句是一个经常被低估的巨大收获：不再需要一直输入`self.assert...`是一种解放。

在撰写本文时，`unittest2pytest`尚未处理最后一个测试中的`self.fail("not implemented yet")`语句。因此，我们需要手动用`assert 0, "not implemented yet"`替换它。也许您想提交一个 PR 来改进这个项目？([`github.com/pytest-dev/unittest2pytest`](https://github.com/pytest-dev/unittest2pytest))。

# 处理设置/拆卸

要完全将`TestCase`子类转换为 pytest 风格，我们需要用 pytest 的习语替换`unittest`。我们已经在上一节中看到了如何使用`unittest2pytest`来做到这一点。但是我们能对`setUp`和`tearDown`方法做些什么呢？

正如我们之前学到的，`TestCase`子类中的`autouse` fixtures 工作得很好，所以它们是替换`setUp`和`tearDown`方法的一种自然方式。让我们使用上一节的例子。

在转换`assert`语句之后，首先要做的是删除`unittest.TestCase`的子类化：

```py
class Test(unittest.TestCase):
    ...
```

这变成了以下内容：

```py
class Test:
    ...
```

接下来，我们需要将`setup`/`teardown`方法转换为 fixture 等效方法：

```py
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.filepath = cls.temp_dir / "data.csv"
        cls.filepath.write_text(DATA.strip())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
```

因此，类作用域的`setUpClass`和`tearDownClass`方法将成为一个单一的类作用域 fixture：

```py
    @classmethod
    @pytest.fixture(scope='class', autouse=True)
    def _setup_class(cls):
        temp_dir = Path(tempfile.mkdtemp())
        cls.filepath = temp_dir / "data.csv"
        cls.filepath.write_text(DATA.strip())
        yield
        shutil.rmtree(temp_dir)
```

由于`yield`语句，我们可以很容易地在 fixture 本身中编写拆卸代码，就像我们已经学到的那样。

以下是一些观察：

+   Pytest 不在乎我们如何称呼我们的 fixture，所以我们可以继续使用旧的`setUpClass`名称。我们选择将其更改为`setup_class`，有两个目标：避免混淆这段代码的读者，因为它可能看起来仍然是一个`TestCase`子类，并且使用`_`前缀表示这个 fixture 不应该像普通的 pytest fixture 一样使用。

+   我们将`temp_dir`更改为局部变量，因为我们不再需要在`cls`中保留它。以前，我们不得不这样做，因为我们需要在`tearDownClass`期间访问`cls.temp_dir`，但现在我们可以将其保留为一个局部变量，并在`yield`语句之后访问它。这是使用`yield`将设置和拆卸代码分开的美妙之一：你不需要保留上下文变量；它们自然地作为函数的局部变量保留。

我们使用相同的方法来处理`setUp`方法：

```py
    def setUp(self):
        self.grids = list(iter_grids_from_csv(self.filepath))
```

这变成了以下内容：

```py
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.grids = list(iter_grids_from_csv(self.filepath))
```

这种技术非常有用，因为你可以通过一组最小的更改得到一个纯粹的 pytest 类。此外，像我们之前做的那样为 fixtures 使用命名约定，有助于向读者传达 fixtures 正在转换旧的`setup`/`teardown`习惯。

现在这个类是一个合适的 pytest 类，你可以自由地使用 fixtures 和参数化。

# 管理测试层次结构

正如我们所看到的，在大型测试套件中需要共享功能是很常见的。由于`unittest`是基于子类化`TestCase`，所以在`TestCase`子类本身中放置额外的功能是很常见的。例如，如果我们需要测试需要数据库的应用逻辑，我们可能最初会直接在我们的`TestCase`子类中添加启动和连接到数据库的功能：

```py
class Test(unittest.TestCase):

    def setUp(self):
        self.db_file = self.create_temporary_db()
        self.session = self.connect_db(self.db_file)

    def tearDown(self):
        self.session.close()
        os.remove(self.db_file)

    def create_temporary_db(self):
        ...

    def connect_db(self, db_file):
        ...

    def create_table(self, table_name, **fields):
        ...

    def check_row(self, table_name, **query):
        ...

    def test1(self):
        self.create_table("weapons", name=str, type=str, dmg=int)
        ...
```

这对于单个测试模块效果很好，但通常情况下，我们需要在以后的某个时候在另一个测试模块中使用这个功能。`unittest`模块没有内置的功能来共享常见的`setup`/`teardown`代码，所以大多数人自然而然地会将所需的功能提取到一个超类中，然后在需要的地方从中创建一个子类：

```py
# content of testing.py
class DataBaseTesting(unittest.TestCase):

    def setUp(self):
        self.db_file = self.create_temporary_db()
        self.session = self.connect_db(self.db_file)

    def tearDown(self):
        self.session.close()
        os.remove(self.db_file)

    def create_temporary_db(self):
        ...

    def connect_db(self, db_file):
        ...

    def create_table(self, table_name, **fields):
        ...

    def check_row(self, table_name, **query):
        ...

# content of test_database2.py
from . import testing

class Test(testing.DataBaseTesting):

    def test1(self):
        self.create_table("weapons", name=str, type=str, dmg=int)
        ...

```

超类通常不仅包含`setup`/`teardown`代码，而且通常还包括调用`self.assert*`执行常见检查的实用函数（例如在上一个例子中的`check_row`）。

继续我们的例子：一段时间后，我们需要在另一个测试模块中完全不同的功能，例如，测试一个 GUI 应用程序。我们现在更加明智，怀疑我们将需要在几个其他测试模块中使用 GUI 相关的功能，所以我们首先创建一个具有我们直接需要的功能的超类：

```py
class GUITesting(unittest.TestCase):

    def setUp(self):
        self.app = self.create_app()

    def tearDown(self):
        self.app.close_all_windows()

    def mouse_click(self, window, button):
        ...

    def enter_text(self, window, text):
        ...
```

将`setup`/`teardown`和测试功能移动到超类的方法是可以的，并且易于理解。

当我们需要在同一个测试模块中使用两个不相关的功能时，问题就出现了。在这种情况下，我们别无选择，只能求助于多重继承。假设我们需要测试连接到数据库的对话框；我们将需要编写这样的代码：

```py
from . import testing

class Test(testing.DataBaseTesting, testing.GUITesting):

    def setUp(self):
 testing.DataBaseTesting.setUp(self)
 testing.GUITesting.setUp(self)

    def tearDown(self):
 testing.GUITesting.setUp(self)
 testing.DataBaseTesting.setUp(self)
```

一般来说，多重继承会使代码变得不太可读，更难以理解。在这里，它还有一个额外的恼人之处，就是我们需要显式地按正确的顺序调用`setUp`和`tearDown`。

还要注意的一点是，在 `unittest` 框架中，`setUp` 和 `tearDown` 是可选的，因此如果某个类不需要任何拆卸代码，通常不会声明 `tearDown` 方法。如果此类包含的功能后来移动到超类中，许多子类可能也不会声明 `tearDown` 方法。问题出现在后来的多重继承场景中，当您改进超类并需要添加 `tearDown` 方法时，因为现在您必须检查所有子类，并确保它们调用超类的 `tearDown` 方法。

因此，假设我们发现自己处于前述情况，并且希望开始使用与 `TestCase` 测试不兼容的 pytest 功能。我们如何重构我们的实用类，以便我们可以自然地从 pytest 中使用它们，并且保持现有的基于 `unittest` 的测试正常工作？

# 使用 fixtures 重用测试代码

我们应该做的第一件事是将所需的功能提取到定义良好的 fixtures 中，并将它们放入 `conftest.py` 文件中。继续我们的例子，我们可以创建 `db_testing` 和 `gui_testing` fixtures：

```py
class DataBaseFixture:

    def __init__(self):
        self.db_file = self.create_temporary_db()
        self.session = self.connect_db(self.db_file)

    def teardown(self):
        self.session.close()
        os.remove(self.db_file)

    def create_temporary_db(self):
        ...

    def connect_db(self, db_file):
        ...

    ...

@pytest.fixture
def db_testing():
    fixture = DataBaseFixture()
    yield fixture
    fixture.teardown()

class GUIFixture:

    def __init__(self):
        self.app = self.create_app()

    def teardown(self):
        self.app.close_all_windows()

    def mouse_click(self, window, button):
        ...

    def enter_text(self, window, text):
        ...

@pytest.fixture
def gui_testing():
    fixture = GUIFixture()
    yield fixture
    fixture.teardown()
```

现在，您可以开始使用纯 pytest 风格编写新的测试，并使用 `db_testing` 和 `gui_testing` fixtures，这很棒，因为它为在新测试中使用 pytest 功能打开了大门。但这里很酷的一点是，我们现在可以更改 `DataBaseTesting` 和 `GUITesting` 来重用 fixtures 提供的功能，而不会破坏现有代码：

```py
class DataBaseTesting(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _setup(self, db_testing):
 self._db_testing = db_testing

    def create_temporary_db(self):
        return self._db_testing.create_temporary_db()

    def connect_db(self, db_file):
        return self._db_testing.connect_db(db_file)

    ...

class GUITesting(unittest.TestCase):

    @pytest.fixture(autouse=True)
 def _setup(self, gui_testing):
 self._gui_testing = gui_testing

    def mouse_click(self, window, button):
        return self._gui_testing.mouse_click(window, button)

    ...
```

我们的 `DatabaseTesting` 和 `GUITesting` 类通过声明一个自动使用的 `_setup` fixture 来获取 fixture 值，这是我们在本章早期学到的一个技巧。我们可以摆脱 `tearDown` 方法，因为 fixture 将在每次测试后自行清理，而实用方法变成了在 fixture 中实现的方法的简单代理。

作为奖励分，`GUIFixture` 和 `DataBaseFixture` 也可以使用其他 pytest fixtures。例如，我们可能可以移除 `DataBaseTesting.create_temporary_db()`，并使用内置的 `tmpdir` fixture 为我们创建临时数据库文件：

```py
class DataBaseFixture:

    def __init__(self, tmpdir):
        self.db_file = str(tmpdir / "file.db")
        self.session = self.connect_db(self.db_file)

    def teardown(self):
        self.session.close()

    ...

@pytest.fixture
def db_testing(tmpdir):
    fixture = DataBaseFixture(tmpdir)
    yield fixture
    fixture.teardown()
```

然后使用其他 fixtures 可以极大地简化现有的测试实用程序代码。

值得强调的是，这种重构不需要对现有测试进行任何更改。这里，fixtures 的一个好处再次显而易见：fixture 的要求变化不会影响使用 fixture 的测试。

# 重构测试实用程序

在前一节中，我们看到测试套件可能使用子类来共享测试功能，并且如何将它们重构为 fixtures，同时保持现有的测试正常工作。

在 `unittest` 套件中通过超类共享测试功能的另一种选择是编写单独的实用类，并在测试中使用它们。回到我们的例子，我们需要具有与数据库相关的设施，这是一种在 `unittest` 友好的方式实现的方法，而不使用超类：

```py
# content of testing.py
class DataBaseTesting:

    def __init__(self, test_case):        
        self.db_file = self.create_temporary_db()
        self.session = self.connect_db(self.db_file)
        self.test_case = test_case
        test_case.addCleanup(self.teardown)

    def teardown(self):
        self.session.close()
        os.remove(self.db_file)

    ...

    def check_row(self, table_name, **query):
        row = self.session.find(table_name, **query)
        self.test_case.assertIsNotNone(row)
        ...

# content of test_1.py
from testing import DataBaseTesting

class Test(unittest.TestCase):

    def test_1(self):
        db_testing = DataBaseTesting(self)
        db_testing.create_table("weapons", name=str, type=str, dmg=int)
        db_testing.check_row("weapons", name="zweihander")
        ...

```

在这种方法中，我们将测试功能分离到一个类中，该类将当前的 `TestCase` 实例作为第一个参数，然后是任何其他所需的参数。

`TestCase`实例有两个目的：为类提供对各种`self.assert*`函数的访问，并作为一种方式向`TestCase.addCleanup`注册清理函数（[`docs.python.org/3/library/unittest.html#unittest.TestCase.addCleanup`](https://docs.python.org/3/library/unittest.html#unittest.TestCase.addCleanup)）。`TestCase.addCleanup`注册的函数将在每个测试完成后调用，无论它们是否成功。我认为它们是`setUp`/`tearDown`函数的一个更好的替代方案，因为它们允许资源被创建并立即注册进行清理。在`setUp`期间创建所有资源并在`tearDown`期间释放它们的缺点是，如果在`setUp`方法中引发任何异常，那么`tearDown`将根本不会被调用，从而泄漏资源和状态，这可能会影响后续的测试。

如果您的`unittest`套件使用这种方法进行测试设施，那么好消息是，您可以轻松地转换/重用这些功能以供 pytest 使用。

因为这种方法与 fixtures 的工作方式非常相似，所以很容易稍微改变类以使其作为 fixtures 工作：

```py
# content of testing.py
class DataBaseFixture:

    def __init__(self):
        self.db_file = self.create_temporary_db()
        self.session = self.connect_db(self.db_file)

    ...

    def check_row(self, table_name, **query):
        row = self.session.find(table_name, **query)
        assert row is not None

# content of conftest.py
@pytest.fixture
def db_testing():
    from .testing import DataBaseFixture
    result = DataBaseFixture()
    yield result
    result.teardown()
```

我们摆脱了对`TestCase`实例的依赖，因为我们的 fixture 现在负责调用`teardown()`，并且我们可以自由地使用普通的 asserts 而不是`Test.assert*`方法。

为了保持现有的套件正常工作，我们只需要创建一个薄的子类来处理在与`TestCase`子类一起使用时的清理：

```py
# content of testing.py
class DataBaseTesting(DataBaseFixture):

    def __init__(self, test_case):
        super().__init__()
        test_case.addCleanup(self.teardown) 
```

通过这种小的重构，我们现在可以在新测试中使用原生的 pytest fixtures，同时保持现有的测试与以前完全相同的工作方式。

虽然这种方法效果很好，但一个问题是，不幸的是，我们无法在`DataBaseFixture`类中使用其他 pytest fixtures（例如`tmpdir`），而不破坏在`TestCase`子类中使用`DataBaseTesting`的兼容性。

# 迁移策略

能够立即使用 pytest 作为运行器开始使用`unittest`-based 测试绝对是一个非常强大的功能。

最终，您需要决定如何处理现有的基于`unittest`的测试。您可以选择几种方法：

+   **转换所有内容**：如果您的测试套件相对较小，您可能决定一次性转换所有测试。这样做的好处是，您不必妥协以保持现有的`unittest`套件正常工作，并且更容易被他人审查，因为您的拉取请求将具有单一主题。

+   **边转换边进行**：您可能决定根据需要转换测试和功能。当您需要添加新测试或更改现有测试时，您可以利用这个机会转换测试和/或重构功能，使用前几节中的技术来创建 fixtures。如果您不想花时间一次性转换所有内容，而是慢慢地铺平道路，使 pytest 成为唯一的测试套件，那么这是一个很好的方法。

+   **仅新测试**：您可能决定永远不触及现有的`unittest`套件，只在 pytest 风格中编写新测试。如果您有成千上万的测试，可能永远不需要进行维护，那么这种方法是合理的，但您将不得不保持前几节中展示的混合方法永远正常工作。

根据您的时间预算和测试套件的大小选择要使用的迁移策略。

# 总结

我们已经讨论了一些关于如何在各种规模的基于`unittest`的测试套件中使用 pytest 的策略和技巧。我们从讨论如何使用 pytest 作为测试运行器开始，以及哪些功能适用于`TestCase`测试。我们看了看如何使用`unittest2pytest`工具将`self.assert*`方法转换为普通的 assert 语句，并充分利用 pytest 的内省功能。然后，我们学习了一些关于如何将基于`unittest`的`setUp`/`tearDown`代码迁移到 pytest 风格的测试类中的技巧，管理在测试层次结构中分散的功能，以及一般的实用工具。最后，我们总结了可能的迁移策略概述，适用于各种规模的测试套件。

在下一章中，我们将简要总结本书学到的内容，并讨论接下来可能会有什么。
