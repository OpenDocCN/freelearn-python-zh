# 第二章：标记和参数化

在学习了编写和运行测试的基础知识之后，我们将深入了解两个重要的 pytest 功能：**标记**和**参数化**。

首先，我们将学习标记，它允许我们根据应用的标记选择性地运行测试，并将一般数据附加到测试函数，这些数据可以被夹具和插件使用。在同一主题中，我们将看看内置标记及其提供的内容。

其次，我们将学习**测试参数化**，它允许我们轻松地将相同的测试函数应用于一组输入值。这极大地避免了重复的测试代码，并使得很容易添加随着软件发展可能出现的新测试用例。

总之，在本章中我们将涵盖以下内容：

+   标记基础知识

+   内置标记

+   参数化

# 标记基础知识

Pytest 允许您使用元数据对函数和类进行标记。此元数据可用于选择性运行测试，并且也可用于夹具和插件，以执行不同的任务。让我们看看如何创建和应用标记到测试函数，然后再进入内置的 pytest 标记。

# 创建标记

使用`@pytest.mark`装饰器创建标记。它作为工厂工作，因此对它的任何访问都将自动创建一个新标记并将其应用于函数。通过示例更容易理解：

```py
@pytest.mark.slow
def test_long_computation():
    ...
```

通过使用`@pytest.mark.slow`装饰器，您将标记命名为`slow`应用于`test_long_computation`。

标记也可以接收**参数**：

```py
@pytest.mark.timeout(10, method="thread")
def test_topology_sort():
    ...
```

在上一个示例中使用的`@pytest.mark.timeout`来自 pytest-timeout 插件；有关更多详细信息，请访问[`pypi.org/project/pytest-timeout/`](https://pypi.org/project/pytest-timeout/)。通过这样做，我们定义了`test_topology_sort`不应超过 10 秒，在这种情况下，应使用`thread`方法终止。为标记分配参数是一个非常强大的功能，为插件和夹具提供了很大的灵活性。我们将在下一章中探讨这些功能和`pytest-timeout`插件。

您可以通过多次应用`@pytest.mark`装饰器向测试添加多个标记，例如：

```py
@pytest.mark.slow
@pytest.mark.timeout(10, method="thread")
def test_topology_sort():
    ...
```

如果您一遍又一遍地应用相同的标记，可以通过将其分配给一个变量一次并根据需要在测试中应用它来避免重复自己：

```py
timeout10 = pytest.mark.timeout(10, method="thread")

@timeout10
def test_topology_sort():
    ...

@timeout10
def test_remove_duplicate_points():
    ...
```

如果此标记在多个测试中使用，可以将其移动到测试实用程序模块并根据需要导入：

```py
from mylib.testing import timeout10

@timeout10
def test_topology_sort():
    ...

@timeout10
def test_remove_duplicate_points():
    ...
```

# 基于标记运行测试

您可以使用`-m`标志将标记作为选择因素运行测试。例如，要运行所有带有`slow`标记的测试：

```py
λ pytest -m slow
```

`-m`标志还接受表达式，因此您可以进行更高级的选择。要运行所有带有`slow`标记的测试，但不运行带有`serial`标记的测试，您可以使用：

```py
λ pytest -m "slow and not serial"
```

表达式限制为`and`，`not`和`or`运算符。

自定义标记对于优化 CI 系统上的测试运行非常有用。通常，环境问题，缺少依赖项，甚至一些错误提交的代码可能会导致整个测试套件失败。通过使用标记，您可以选择一些快速和/或足够广泛以检测代码中大部分问题的测试，然后首先运行这些测试，然后再运行所有其他测试。如果其中任何一个测试失败，我们将中止作业，并避免通过运行注定会失败的所有测试来浪费大量时间。

我们首先将自定义标记应用于这些测试。任何名称都可以，但常用的名称是`smoke`，如*烟雾探测器*，以便在一切都燃烧之前检测问题。

然后首先运行烟雾测试，只有在它们通过后才运行完整的测试套件：

```py
λ pytest -m "smoke"
...
λ pytest -m "not smoke"
```

如果任何烟雾测试失败，您不必等待整个套件完成以获得此反馈。

您可以通过创建测试的层次结构，从最简单到最慢，增加此技术。例如：

+   `smoke`

+   `unittest`

+   `integration`

+   `<其余所有>`

然后执行如下：

```py
λ pytest -m "smoke"
...
λ pytest -m "unittest"
...
λ pytest -m "integration"
...
λ pytest -m "not smoke and not unittest and not integration"
```

确保包含第四步；否则，没有标记的测试将永远不会运行。

使用标记来区分不同 pytest 运行中的测试也可以用于其他场景。例如，当使用`pytest-xdist`插件并行运行测试时，我们有一个并行会话，可以并行执行大多数测试套件，但可能决定在单独的 pytest 会话中串行运行一些测试，因为它们在一起执行时很敏感或有问题。

# 将标记应用于类

您可以将`@pytest.mark`装饰器应用于一个类。这将使该标记应用于该类中的所有测试方法，避免了将标记代码复制粘贴到所有测试方法中：

```py
@pytest.mark.timeout(10)
class TestCore:

    def test_simple_simulation(self):
        ...

    def test_compute_tracers(self):
        ...
```

前面的代码本质上与以下代码相同：

```py
class TestCore:

 @pytest.mark.timeout(10)
    def test_simple_simulation(self):
        ...

    @pytest.mark.timeout(10)
    def test_compute_tracers(self):
        ...
```

然而，有一个区别：将`@pytest.mark`装饰器应用于一个类意味着所有它的子类都会继承该标记。子类测试类的继承并不常见，但有时是一种有用的技术，可以避免重复测试代码，或者确保实现符合某个特定接口。我们将在本章后面和第四章 *Fixture*中看到更多这方面的例子。

与测试函数一样，装饰器可以应用多次：

```py
@pytest.mark.slow
@pytest.mark.timeout(10)
class TestCore:

    def test_simple_simulation(self):
        ...

    def test_compute_tracers(self):
        ...
```

# 将标记应用于模块

我们还可以将一个标记应用于模块中的所有测试函数和测试类。只需声明一个名为`pytestmark`的**全局变量**：

```py
import pytest

pytestmark = pytest.mark.timeout(10)

class TestCore:

    def test_simple_simulation(self):
        ...

def test_compute_tracers():
    ...
```

以下是等效于这个的：

```py
import pytest

@pytest.mark.timeout(10)
class TestCore:

    def test_simple_simulation(self):
        ...

@pytest.mark.timeout(10)
def test_compute_tracers():
    ...
```

您也可以使用`tuple`或`list`的标记来应用多个标记：

```py
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10)]
```

# 自定义标记和 pytest.ini

通过应用`@pytest.mark`装饰器来动态声明新的标记是很方便的。这使得快速开始享受使用标记的好处变得轻而易举。

这种便利性是有代价的：用户可能会在标记名称中犯拼写错误，例如`@pytest.mark.solw`，而不是`@pytest.mark.slow`。根据被测试的项目，这种拼写错误可能只是一个小烦恼，也可能是一个更严重的问题。

因此，让我们回到我们之前的例子，其中一个测试套件根据标记的测试在 CI 上以层次结构执行：

+   `smoke`

+   `unittest`

+   `integration`

+   `<所有其他>`

```py
λ pytest -m "smoke"
...
λ pytest -m "unittest"
...
λ pytest -m "integration"
...
λ pytest -m "not smoke and not unittest and not integration"
```

开发人员在为其中一个测试应用标记时可能会犯拼写错误：

```py
@pytest.mark.smoek
def test_simulation_setup():
    ...
```

这意味着该测试将在最后一步执行，而不是与其他`smoke`测试一起在第一步执行。同样，这可能从一个小麻烦变成一个严重的问题，这取决于测试套件。

具有固定标记集的成熟测试套件可能会在`pytest.ini`文件中声明它们：

```py
[pytest]
markers =
    slow
    serial
    smoke: quick tests that cover a good portion of the code
    unittest: unit tests for basic functionality
    integration: cover to cover functionality testing    
```

`markers`选项接受一个标记列表，格式为`<name>: description`，其中描述部分是可选的（最后一个示例中的`slow`和`serial`没有描述）。

可以使用`--markers`标志显示所有标记的完整列表：

```py
λ pytest --markers
@pytest.mark.slow:

@pytest.mark.serial:

@pytest.mark.smoke: quick tests that cover a good portion of the code

@pytest.mark.unittest: unit tests for basic functionality

@pytest.mark.integration: cover to cover functionality testing

...
```

`--strict`标志使得在`pytest.ini`文件中未声明的标记使用成为错误。使用我们之前的带有拼写错误的示例，现在会得到一个错误，而不是在使用`--strict`运行时 pytest 悄悄地创建标记：

```py
λ pytest --strict tests\test_wrong_mark.py
...
collected 0 items / 1 errors

============================== ERRORS ===============================
_____________ ERROR collecting tests/test_wrong_mark.py _____________
tests\test_wrong_mark.py:4: in <module>
 @pytest.mark.smoek
..\..\.env36\lib\site-packages\_pytest\mark\structures.py:311: in __getattr__
 self._check(name)
..\..\.env36\lib\site-packages\_pytest\mark\structures.py:327: in _check
 raise AttributeError("%r not a registered marker" % (name,))
E AttributeError: 'smoek' not a registered marker
!!!!!!!!!!!!!! Interrupted: 1 errors during collection !!!!!!!!!!!!!!
====================== 1 error in 0.09 seconds ======================
```

希望确保所有标记都在`pytest.ini`中注册的测试套件也应该使用`addopts`：

```py
[pytest]
addopts = --strict
markers =
    slow
    serial
    smoke: quick tests that cover a good portion of the code
    unittest: unit tests for basic functionality
    integration: cover to cover functionality testing
```

# 内置标记

通过学习标记的基础知识以及如何使用它们，现在让我们来看一些内置的 pytest 标记。这不是所有内置标记的详尽列表，但是常用的标记。另外，请记住，许多插件也引入了其他标记。

# @pytest.mark.skipif

您可能有一些测试在满足某些条件之前不应该被执行。例如，一些测试可能依赖于并非总是安装的某些库，或者一个可能不在线的本地数据库，或者仅在某些平台上执行。

Pytest 提供了一个内置标记`skipif`，可以根据特定条件*跳过*测试。如果条件为真，则跳过测试不会被执行，并且不会计入测试套件的失败。

例如，您可以使用`skipif`标记来在 Windows 上执行时始终跳过测试：

```py
import sys
import pytest

@pytest.mark.skipif(
 sys.platform.startswith("win"),
 reason="fork not available on Windows",
)
def test_spawn_server_using_fork():
    ...
```

`@pytest.mark.skipif`的第一个参数是条件：在这个例子中，我们告诉 pytest 在 Windows 中跳过这个测试。`reason=`关键字参数是强制的，并且用于在使用`-ra`标志时显示为什么跳过测试：

```py
 tests\test_skipif.py s                                        [100%]
====================== short test summary info ======================
SKIP [1] tests\test_skipif.py:6: fork not available on Windows
===================== 1 skipped in 0.02 seconds =====================
```

始终写入描述性消息是一个好的风格，包括适用时的票号。

另外，我们可以将相同的条件写成如下形式：

```py
import os
import pytest

@pytest.mark.skipif(
 not hasattr(os, 'fork'), reason="os.fork not available"
)
def test_spawn_server_using_fork2():
    ...
```

后一种版本检查实际功能是否可用，而不是基于平台做出假设（Windows 目前没有`os.fork`函数，但也许将来 Windows 可能会支持该函数）。在测试库的功能时，通常也会出现相同的情况，而不是检查某些功能是否存在。我建议在可能的情况下，最好检查函数是否实际存在，而不是检查库的特定版本。

通常，检查功能和特性通常是更好的方法，而不是检查平台和库版本号。以下是完整的`@pytest.mark.skipif`签名：

```py
@pytest.mark.skipif(condition, *, reason=None)
```

# pytest.skip

`@pytest.mark.skipif`装饰器非常方便，但是标记必须在`import`/`collection`时间评估条件，以确定是否应该跳过测试。我们希望最小化测试收集时间，因为毕竟，如果使用`-k`或`--lf`标志，我们最终可能甚至不会执行所有测试

有时，要在导入时检查测试是否应该跳过几乎是不可能的（除非进行一些令人讨厌的黑客）。例如，您可以根据图形驱动程序的功能来决定是否跳过测试，但只能在初始化底层图形库后才能做出这个决定，而初始化图形子系统绝对不是您希望在导入时执行的事情。

对于这些情况，pytest 允许您在测试主体中使用`pytest.skip`函数来强制跳过测试：

```py
def test_shaders():
    initialize_graphics()
    if not supports_shaders():
 pytest.skip("shades not supported in this driver") # rest of the test code ... 
```

`pytest.skip`通过引发内部异常来工作，因此它遵循正常的 Python 异常语义，而且不需要为了正确跳过测试而做其他事情。

# pytest.importorskip

通常，库的测试会依赖于某个特定库是否已安装。例如，pytest 自己的测试套件中有一些针对`numpy`数组的测试，如果没有安装`numpy`，则应该跳过这些测试。

处理这个问题的一种方法是手动尝试导入库，并在库不存在时跳过测试：

```py
def test_tracers_as_arrays_manual():
    try:
        import numpy
    except ImportError:
        pytest.skip("requires numpy")
    ...
```

这可能很快就会过时，因此 pytest 提供了方便的`pytest.importorskip`函数：

```py
def test_tracers_as_arrays():
    numpy = pytest.importorskip("numpy")
    ...
```

`pytest.importorskip`将导入模块并返回模块对象，或者如果无法导入模块，则完全跳过测试。

如果您的测试需要库的最低版本，`pytest.importorskip`也支持`minversion`参数：

```py
def test_tracers_as_arrays_114():
    numpy = pytest.importorskip("numpy", minversion="1.14")
    ...
```

# @pytest.mark.xfail

您可以使用`@pytest.mark.xfail`装饰器来指示测试*`预计会失败`*。像往常一样，我们将标记装饰器应用到测试函数或方法上：

```py
@pytest.mark.xfail
def test_simulation_34():
    ...
```

这个标记支持一些参数，我们将在本节后面看到所有这些参数；但其中一个特别值得讨论：`strict`参数。此参数为标记定义了两种不同的行为：

+   使用`strict=False`（默认值），如果测试通过，测试将被单独计数为**XPASS**（如果测试通过），或者**XFAIL**（如果测试失败），并且**不会使测试套件失败**

+   使用`strict=True`，如果测试失败，测试将被标记为**XFAIL**，但如果测试意外地通过，它将**使测试套件失败**，就像普通的失败测试一样

但是为什么你想要编写一个你预计会失败的测试，这在哪些情况下有用呢？这一开始可能看起来很奇怪，但有一些情况下这是很方便的。

第一种情况是测试总是失败，并且您希望（大声地）得知它突然开始通过。这可能发生在：

+   你发现你的代码中的一个 bug 的原因是第三方库中的问题。在这种情况下，你可以编写一个演示问题的失败测试，并用`@pytest.mark.xfail(strict=True)`标记它。如果测试失败，测试将在测试会话摘要中标记为**XFAIL**，但如果测试**通过**，它将**失败测试套件**。当你升级导致问题的库时，这个测试可能会开始通过，这将提醒你问题已经解决，并需要你的注意。

+   你想到了一个新功能，并设计了一个或多个在你开始实施之前就对其进行测试的测试用例。你可以使用`@pytest.mark.xfail(strict=True)`标记提交测试，并在编写新功能时从测试中删除该标记。这在协作环境中非常有用，其中一个人提供了关于他们如何设想新功能/API 的测试，另一个人根据测试用例实现它。

+   你发现应用程序中的一个 bug，并编写一个演示问题的测试用例。你可能现在没有时间解决它，或者另一个人更适合在代码的那部分工作。在这种情况下，将测试标记为`@pytest.mark.xfail(strict=True)`是一个很好的方法。

上述所有情况都有一个共同点：你有一个失败的测试，并想知道它是否突然开始通过。在这种情况下，测试通过的事实警告你需要注意的事实：一个带有 bug 修复的库的新版本已发布，部分功能现在按预期工作，或者已修复了一个已知的 bug。

`xfail`标记有用的另一种情况是当你有*有时*失败的测试，也称为**不稳定**的**测试**。不稳定的测试是指有时会失败的测试，即使基础代码没有更改。测试失败看起来是随机的原因有很多；以下是其中一些：

+   多线程代码中的时间问题

+   间歇性的网络连接问题

+   没有正确处理异步事件的测试

+   依赖于不确定的行为

这只是列举了一些可能的原因。这种不确定性通常发生在范围更广的测试中，比如集成或 UI。事实上，你几乎总是需要处理大型测试套件中的不稳定测试。

不稳定的测试是一个严重的问题，因为测试套件应该是代码按预期工作并在发生真正问题时能够检测到的指标。不稳定的测试破坏了这一形象，因为开发人员经常会看到与最近的代码更改无关的不稳定的测试失败。当这种情况变得司空见惯时，人们开始再次运行测试套件，希望这次不稳定的测试通过（它经常会通过），但这会侵蚀对整个测试套件的信任，并给开发团队带来挫折。你应该把不稳定的测试视为一个应该被遏制和处理的威胁。

以下是关于如何处理开发团队中的不稳定测试的一些建议：

1.  首先，你需要能够正确识别不稳定的测试。如果一个测试失败，显然与最近的更改无关，再次运行测试。如果之前失败的测试现在**通过**，这意味着测试是不稳定的。

1.  在你的工单系统中创建一个处理特定不稳定测试的问题。使用命名约定或其他方式标记该问题与不稳定测试相关（例如 GitHub 或 JIRA 标签）。

1.  应用`@pytest.mark.xfail(reason="flaky test #123", strict=False)`标记，确保包括问题票号或标识。如果愿意，可以在描述中添加更多信息。

1.  确保定期将关于不稳定测试的问题分配给自己或其他团队成员（例如，在冲刺计划期间）。这样做的目的是以舒适的步伐处理不稳定的测试，最终减少或消除它们。

这些做法解决了两个主要问题：它们允许您避免破坏测试套件的信任，让不稳定的测试不会妨碍开发团队，并且它们制定了一项政策，以便及时处理不稳定的测试。

在涵盖了`xfail`标记有用的情况后，让我们来看一下完整的签名：

```py
@pytest.mark.xfail(condition=None, *, reason=None, raises=None, run=True, strict=False)
```

+   `condition`：如果给定，第一个参数是一个`True`/`False`条件，类似于`@pytest.mark.skipif`中使用的条件：如果为`False`，则忽略`xfail`标记。它可用于根据外部条件（例如平台、Python 版本、库版本等）标记测试为`xfail`。

```py
@pytest.mark.xfail(
 sys.platform.startswith("win"), 
    reason="flaky on Windows #42", strict=False
)
def test_login_dialog():
    ...
```

+   `reason`：一个字符串，在使用`-ra`标志时将显示在短测试摘要中。强烈建议始终使用此参数来解释为什么将测试标记为`xfail`和/或包括一个票号。

```py
@pytest.mark.xfail(
    sys.platform.startswith("win"), 
    reason="flaky on Windows #42", strict=False
)
def test_login_dialog():
    ...
```

+   `raises`：给定一个异常类型，它声明我们期望测试引发该异常的实例。如果测试引发了另一种类型的异常（甚至是`AssertionError`），测试将正常“失败”。这对于缺少功能或测试已知错误特别有用。

```py
@pytest.mark.xfail(raises=NotImplementedError,
                   reason='will be implemented in #987')
def test_credential_check():
    check_credentials('Hawkwood') # not implemented yet
```

+   `run`：如果为`False`，则测试甚至不会被执行，并且将作为 XFAIL 失败。这对于运行可能导致测试套件进程崩溃的代码的测试特别有用（例如，由于已知问题，C/C++扩展导致分段错误）。

```py
@pytest.mark.xfail(
    run=False, reason="undefined particles cause a crash #625"
)
def test_undefined_particle_collision_crash():
    collide(Particle(), Particle())
```

+   `strict`：如果为`True`，则通过的测试将使测试套件失败。如果为`False`，则无论结果如何，测试都不会使测试套件失败（默认为`False`）。这在本节开始时已经详细讨论过。

配置变量`xfail_strict`控制`xfail`标记的`strict`参数的默认值：

```py
[pytest]
xfail_strict = True
```

将其设置为`True`意味着所有标记为 xfail 的测试，没有显式的`strict`参数，都被视为实际的失败期望，而不是不稳定的测试。任何显式传递`strict`参数的`xfail`标记都会覆盖配置值。

# pytest.xfail

最后，您可以通过调用`pytest.xfail`函数在测试中强制触发 XFAIL 结果：

```py
def test_particle_splitting():
    initialize_physics()
    import numpy
    if numpy.__version__ < "1.13":
        pytest.xfail("split computation fails with numpy < 1.13")
    ...
```

与`pytest.skip`类似，当您只能在运行时确定是否需要将测试标记为`xfail`时，这是非常有用的。

# 参数化

一个常见的测试活动是将多个值传递给同一个测试函数，并断言结果。

假设我们有一个应用程序，允许用户定义自定义数学公式，这些公式将在运行时解析和评估。这些公式以字符串形式给出，并且可以使用诸如`sin`、`cos`、`log`等数学函数。在 Python 中实现这个非常简单的方法是使用内置的`eval`（[`docs.python.org/3/library/functions.html#eval`](https://docs.python.org/3/library/functions.html#eval)），但由于它可以执行任意代码，我们选择使用自定义的标记器和评估器来确保安全。

让我们不要深入实现细节，而是专注于一个测试：

```py
def test_formula_parsing():
    tokenizer = FormulaTokenizer()
    formula = Formula.from_string("C0 * x + 10", tokenizer)
    assert formula.eval(x=1.0, C0=2.0) == pytest.approx(12.0)
```

在这里，我们创建了一个`Tokenizer`类，我们的实现使用它来将公式字符串分解为内部标记，以供以后处理。然后，我们将公式字符串和标记器传递给`Formula.from_string`，以获得一个公式对象。有了公式对象，我们将输入值传递给`formula.eval`，并检查返回的值是否符合我们的期望。

但我们也想测试其他数学运算，以确保我们覆盖了`Formula`类的所有功能。

一种方法是通过使用多个断言来扩展我们的测试，以检查其他公式和输入值：

```py
def test_formula_parsing():
    tokenizer = FormulaTokenizer()
    formula = Formula.from_string("C0 * x + 10", tokenizer)
    assert formula.eval(x=1.0, C0=2.0) == pytest.approx(12.0)

    formula = Formula.from_string("sin(x) + 2 * cos(x)", tokenizer)
 assert formula.eval(x=0.7) == pytest.approx(2.1739021)

    formula = Formula.from_string("log(x) + 3", tokenizer)
    assert formula.eval(x=2.71828182846) == pytest.approx(4.0)
```

这样做是有效的，但如果其中一个断言失败，测试函数内部的后续断言将不会被执行。如果有多个失败，我们将不得不多次运行测试来查看所有失败，并最终修复所有问题。

为了在测试运行中看到多个失败，我们可能决定明确为每个断言编写单独的测试：

```py
def test_formula_linear():
    tokenizer = FormulaTokenizer()
    formula = Formula.from_string("C0 * x + 10", tokenizer)
    assert formula.eval(x=1.0, C0=2.0) == pytest.approx(12.0)

def test_formula_sin_cos():
    tokenizer = FormulaTokenizer()
    formula = Formula.from_string("sin(x) + 2 * cos(x)", tokenizer)
    assert formula.eval(x=0.7) == pytest.approx(2.1739021)

def test_formula_log():
    tokenizer = FormulaTokenizer()
    formula = Formula.from_string("log(x) + 3", tokenizer)
    assert formula.eval(x=2.71828182846) == pytest.approx(4.0)
```

但现在我们到处都在重复代码，这将使维护更加困难。假设将来`FormulaTokenizer`被更新为明确接收可以在公式中使用的函数列表。这意味着我们将不得不在多个地方更新`FormulaTokenzier`的创建。

为了避免重复，我们可能决定改为这样写：

```py
def test_formula_parsing2():
    values = [
 ("C0 * x + 10", dict(x=1.0, C0=2.0), 12.0),
 ("sin(x) + 2 * cos(x)", dict(x=0.7), 2.1739021),
 ("log(x) + 3", dict(x=2.71828182846), 4.0),
 ]
    tokenizer = FormulaTokenizer()
    for formula, inputs, result in values:
        formula = Formula.from_string(formula, tokenizer)
        assert formula.eval(**inputs) == pytest.approx(result)
```

这解决了重复代码的问题，但现在我们又回到了一次只看到一个失败的初始问题。

# 输入 `@pytest.mark.parametrize`

为了解决上述所有问题，pytest 提供了备受喜爱的`@pytest.mark.parametrize`标记。使用这个标记，您可以为测试提供一系列输入值，并且 pytest 会自动生成多个测试函数，每个输入值一个。

以下显示了这一点：

```py
@pytest.mark.parametrize(
 "formula, inputs, result",
 [
 ("C0 * x + 10", dict(x=1.0, C0=2.0), 12.0),
 ("sin(x) + 2 * cos(x)", dict(x=0.7), 2.1739021),
 ("log(x) + 3", dict(x=2.71828182846), 4.0),
 ],
)
def test_formula_parsing(formula, inputs, result):
    tokenizer = FormulaTokenizer()
    formula = Formula.from_string(formula, tokenizer)
    assert formula.eval(**inputs) == pytest.approx(result)
```

`@pytest.mark.parametrize` 标记会自动生成多个测试函数，并使用标记给出的参数对它们进行参数化。调用接收两个参数：

+   `argnames`: 逗号分隔的参数名称字符串，将传递给测试函数。

+   `argvalues`: 一系列元组，每个元组生成一个新的测试调用。元组中的每个项目对应一个参数名称，因此第一个元组 `("C0 * x + 10", dict(x=1.0, C0=2.0), 12.0)` 将生成一个对测试函数的调用，参数为：

+   `formula` = `"C0 * x + 10"`

+   `inputs` = `dict(x=1.0, C0=2.0)`

+   `expected` = `12.0`

使用这个标记，pytest 将运行 `test_formula_parsing` 三次，每次传递由`argvalues`参数给出的一组参数。它还会自动生成不同的节点 ID 用于每个测试，使得很容易区分它们：

```py
======================== test session starts ========================
...
collected 8 items / 5 deselected

test_formula.py::test_formula[C0 * x + 10-inputs0-12.0]
test_formula.py::test_formula[sin(x) + 2 * cos(x)-inputs1-2.1739021]
test_formula.py::test_formula[log(x) + 3-inputs2-4.0] 
============== 3 passed, 5 deselected in 0.05 seconds ===============
```

还要注意的是，函数的主体与本节开头的起始测试一样紧凑，但现在我们有多个测试，这使我们能够在发生多个失败时看到多个失败。

参数化测试不仅避免了重复的测试代码，使维护更容易，还邀请您和随后的开发人员随着代码的成熟添加更多的输入值。它鼓励开发人员覆盖更多的情况，因为人们更愿意向参数化测试的`argvalues`添加一行代码，而不是复制和粘贴整个新测试来覆盖另一个输入值。

总之，`@pytest.mark.parametrize` 将使您覆盖更多的输入情况，开销很小。这绝对是一个非常有用的功能，应该在需要以相同方式测试多个输入值时使用。

# 将标记应用于值集

通常，在参数化测试中，您会发现需要像对普通测试函数一样对一组参数应用一个或多个标记。例如，您想对一组参数应用`timeout`标记，因为运行时间太长，或者对一组参数应用`xfail`标记，因为它尚未实现。

在这些情况下，使用`pytest.param`来包装值集并应用您想要的标记：

```py
@pytest.mark.parametrize(
    "formula, inputs, result",
    [
        ...
        ("log(x) + 3", dict(x=2.71828182846), 4.0),
        pytest.param(
 "hypot(x, y)", dict(x=3, y=4), 5.0,
 marks=pytest.mark.xfail(reason="not implemented: #102"),
 ),
    ],
)
```

`pytest.param` 的签名是这样的：

```py
pytest.param(*values, **kw)
```

其中：

+   `*values` 是参数集：`"hypot(x, y)", dict(x=3, y=4), 5.0`。

+   `**kw` 是选项作为关键字参数：`marks=pytest.mark.xfail(reason="not implemented: #102")`。它接受单个标记或一系列标记。还有另一个选项`ids`，将在下一节中显示。

在幕后，传递给`@pytest.mark.parametrize`的每个参数元组都会转换为一个`pytest.param`，没有额外的选项，因此，例如，在以下第一个代码片段等同于第二个代码片段：

```py
@pytest.mark.parametrize(
    "formula, inputs, result",
    [
        ("C0 * x + 10", dict(x=1.0, C0=2.0), 12.0),
        ("sin(x) + 2 * cos(x)", dict(x=0.7), 2.1739021),
    ]
)
```

```py
@pytest.mark.parametrize(
    "formula, inputs, result",
    [
        pytest.param("C0 * x + 10", dict(x=1.0, C0=2.0), 12.0),
        pytest.param("sin(x) + 2 * cos(x)", dict(x=0.7), 2.1739021),
    ]
)
```

# 自定义测试 ID

考虑以下示例：

```py
@pytest.mark.parametrize(
    "formula, inputs, result",
    [
        ("x + 3", dict(x=1.0), 4.0,),
        ("x - 1", dict(x=6.0), 5.0,),
    ],
)
def test_formula_simple(formula, inputs, result):
    ...
```

正如我们所见，pytest 会根据参数在参数化调用中使用的参数自动生成自定义测试 ID。运行`pytest -v`将生成这些测试 ID：

```py
======================== test session starts ========================
...
tests/test_formula.py::test_formula_simple[x + 3-inputs0-4.0]
tests/test_formula.py::test_formula_simple[x - 1-inputs1-5.0]
```

如果你不喜欢自动生成的 ID，你可以使用`pytest.param`和`id`选项来自定义它：

```py
@pytest.mark.parametrize(
    "formula, inputs, result",
    [
        pytest.param("x + 3", dict(x=1.0), 4.0, id='add'),
        pytest.param("x - 1", dict(x=6.0), 5.0, id='sub'),
    ],
)
def test_formula_simple(formula, inputs, result):
    ...
```

这产生了以下结果：

```py
======================== test session starts ========================
...
tests/test_formula.py::test_formula_simple[add]
tests/test_formula.py::test_formula_simple[sub]
```

这也很有用，因为在使用`-k`标志时，选择测试变得更容易：

```py
λ pytest -k "x + 3-inputs0-4.0"
```

对比：

```py
λ pytest -k "add"
```

# 测试多个实现

设计良好的系统通常利用接口提供的抽象，而不是与特定实现绑定。这使得系统更能够适应未来的变化，因为要扩展它，你需要实现一个符合预期接口的新扩展，并将其集成到系统中。

经常出现的一个挑战是如何确保现有的实现符合特定接口的所有细节。

例如，假设我们的系统需要能够将一些内部类序列化为文本格式以保存和加载到磁盘。以下是我们系统中的一些内部类：

+   `Quantity`：表示一个值和一个计量单位。例如，`Quantity(10, "m")`表示*10 米*。`Quantity`对象具有加法、减法和乘法——基本上，所有你从本机`float`期望的运算符，但考虑到计量单位。

+   `Pipe`：表示液体可以流过的管道。它有一个`length`和`diameter`，都是`Quantity`实例。

最初，在我们的开发中，我们只需要以**JSON**格式保存这些对象，所以我们继续实现一个直接的序列化器类，能够序列化和反序列化我们的类：

```py
class JSONSerializer:

    def serialize_quantity(self, quantity: Quantity) -> str:
        ...

    def deserialize_quantity(self, data: str) -> Quantity:
        ...

    def serialize_pipe(self, pipe: Pipe) -> str:
        ...

    def deserialize_pipe(self, data: str) -> Pipe:
        ...
```

现在我们应该写一些测试来确保一切正常运行。

```py
class Test:

    def test_quantity(self):
        serializer = JSONSerializer()
        quantity = Quantity(10, "m")
        data = serializer.serialize(quantity)
        new_quantity = serializer.deserialize(data)
        assert new_quantity == quantity

    def test_pipe(self):
        serializer = JSONSerializer()
        pipe = Pipe(
            length=Quantity(1000, "m"), diameter=Quantity(35, "cm")
        )
        data = serializer.serialize(pipe)
        new_pipe = serializer.deserialize(data)
        assert new_pipe == pipe
```

这样做效果很好，也是一个完全有效的方法，考虑到我们的需求。

一段时间过去了，新的需求出现了：现在我们需要将我们的对象序列化为其他格式，即`XML`和[`YAML`](http://yaml.org/)。为了保持简单，我们创建了两个新类，`XMLSerializer`和`YAMLSerializer`，它们实现了相同的`serialize`/`deserialize`方法。因为它们符合与`JSONSerializer`相同的接口，我们可以在系统中互换使用新类，这很棒。

但是我们如何测试不同的实现？

一个天真的方法是在每个测试中循环遍历不同的实现：

```py
class Test:

    def test_quantity(self):
        for serializer in [
 JSONSerializer(), XMLSerializer(), YAMLSerializer()
 ]:
            quantity = Quantity(10, "m")
            data = serializer.serialize(quantity)
            new_quantity = serializer.deserialize(data)
            assert new_quantity == quantity

    def test_pipe(self):
        for serializer in [
 JSONSerializer(), XMLSerializer(), YAMLSerializer()
 ]:
            pipe = Pipe(
                length=Quantity(1000, "m"),
                diameter=Quantity(35, "cm"),
            )
            data = serializer.serialize(pipe)
            new_pipe = serializer.deserialize(data)
            assert new_pipe == pipe
```

这样做虽然有效，但并不理想，因为我们必须在每个测试中复制和粘贴循环定义，这样更难以维护。而且，如果其中一个序列化器失败，列表中的下一个序列化器将永远不会被执行。

另一种可怕的方法是每次复制和粘贴整个测试函数，替换序列化器类，但我们不会在这里展示。

一个更好的解决方案是在类级别使用`@pytest.mark.parametrize`。观察：

```py
@pytest.mark.parametrize(
 "serializer_class",
 [JSONSerializer, XMLSerializer, YAMLSerializer],
)
class Test:

    def test_quantity(self, serializer_class):
        serializer = serializer_class()
        quantity = Quantity(10, "m")
        data = serializer.serialize(quantity)
        new_quantity = serializer.deserialize(data)
        assert new_quantity == quantity

    def test_pipe(self, serializer_class):
        serializer = serializer_class()
        pipe = Pipe(
            length=Quantity(1000, "m"), diameter=Quantity(35, "cm")
        )
        data = serializer.serialize(pipe)
        new_pipe = serializer.deserialize(data)
        assert new_pipe == pipe
```

通过一个小改变，我们已经扩展了我们现有的测试，以覆盖所有新的实现：

```py
test_parametrization.py::Test::test_quantity[JSONSerializer] PASSED
test_parametrization.py::Test::test_quantity[XMLSerializer] PASSED
test_parametrization.py::Test::test_quantity[YAMLSerializer] PASSED
test_parametrization.py::Test::test_pipe[JSONSerializer] PASSED
test_parametrization.py::Test::test_pipe[XMLSerializer] PASSED
test_parametrization.py::Test::test_pipe[YAMLSerializer] PASSED
```

`@pytest.mark.parametrize`装饰器还清楚地表明，新的实现应该添加到列表中，并且所有现有的测试必须通过。也需要为类添加的新测试对所有实现都通过。

总之，`@pytest.mark.parametrize`可以是一个非常强大的工具，以确保不同的实现符合接口的规范。

# 总结

在本章中，我们学习了如何使用标记来组织我们的代码，并帮助我们以灵活的方式运行测试套件。然后我们看了如何使用`@pytest.mark.skipif`来有条件地跳过测试，以及如何使用`@pytest.mark.xfail`标记来处理预期的失败和不稳定的测试。然后我们讨论了在协作环境中处理不稳定测试的方法。最后，我们讨论了使用`@pytest.mark.parametrize`的好处，以避免重复我们的测试代码，并使自己和其他人能够轻松地向现有测试添加新的输入案例。

在下一章中，我们将终于介绍 pytest 最受喜爱和强大的功能之一：**fixtures**。
