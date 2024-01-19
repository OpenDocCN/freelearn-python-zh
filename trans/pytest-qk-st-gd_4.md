# 插件

在前一章中，我们探讨了pytest最重要的特性之一：fixture。我们学会了如何使用fixture来管理资源，并在编写测试时让我们的生活更轻松。

pytest是以定制和灵活性为目标构建的，并允许开发人员编写称为**插件**的强大扩展。pytest中的插件可以做各种事情，从简单地提供新的fixture，到添加命令行选项，改变测试的执行方式，甚至运行用其他语言编写的测试。

在本章中，我们将做以下事情：

+   学习如何查找和安装插件

+   品尝生态系统提供的插件

# 查找和安装插件

正如本章开头提到的，pytest是从头开始以定制和灵活性为目标编写的。插件机制是pytest架构的核心，以至于pytest的许多内置功能都是以内部插件的形式实现的，比如标记、参数化、fixture——几乎所有东西，甚至命令行选项。

这种灵活性导致了一个庞大而丰富的插件生态系统。在撰写本文时，可用的插件数量已经超过500个，而且这个数字以惊人的速度不断增加。

# 查找插件

考虑到插件的数量众多，如果有一个网站能够展示所有pytest插件以及它们的描述，那将是很好的。如果这个地方还能显示关于不同Python和pytest版本的兼容性信息，那就更好了。

好消息是，这样的网站已经存在了，并且由核心开发团队维护：pytest插件兼容性（[http://plugincompat.herokuapp.com/](http://plugincompat.herokuapp.com/)）。在这个网站上，你将找到PyPI中所有可用的pytest插件的列表，以及Python和pytest版本的兼容性信息。该网站每天都会从PyPI直接获取新的插件和更新，是一个浏览新插件的好地方。

# 安装插件

插件通常使用`pip`安装：

```py
λ pip install <PLUGIN_NAME>
```

例如，要安装`pytest-mock`，我们执行以下操作：

```py
λ pip install pytest-mock
```

不需要任何注册；pytest会自动检测你的虚拟环境或Python安装中安装的插件。

这种简单性使得尝试新插件变得非常容易。

# 各种插件概述

现在，我们将看一些有用和/或有趣的插件。当然，不可能在这里覆盖所有的插件，所以我们将尝试覆盖那些涵盖流行框架和一般功能的插件，还有一些晦涩的插件。当然，这只是皮毛，但让我们开始吧。

# pytest-xdist

这是一个非常受欢迎的插件，由核心开发人员维护；它允许你在多个CPU下运行测试，以加快测试运行速度。

安装后，只需使用`-n`命令行标志来使用给定数量的CPU来运行测试：

```py
λ pytest -n 4
```

就是这样！现在，你的测试将在四个核心上运行，希望能够加快测试套件的速度，如果测试是CPU密集型的话，尽管I/O绑定的测试不会看到太多改进。你也可以使用`-n auto`来让`pytest-xdist`自动计算出你可用的CPU数量。

请记住，当你的测试并行运行，并且以随机顺序运行时，它们必须小心避免相互干扰，例如，读/写到同一个目录。虽然它们应该是幂等的，但以随机顺序运行测试通常会引起之前潜伏的问题。

# pytest-cov

`pytest-cov`插件与流行的coverage模块集成，当运行测试时提供详细的覆盖报告。这让你可以检测到没有被任何测试代码覆盖的代码部分，这是一个机会，可以编写更多的测试来覆盖这些情况。

安装后，您可以使用`--cov`选项在测试运行结束时提供覆盖报告：

```py
λ pytest --cov=src
...
----------- coverage: platform win32, python 3.6.3-final-0 -----------
Name                  Stmts   Miss  Cover
----------------------------------------
src/series.py           108      5   96%
src/tests/test_series    22      0  100%
----------------------------------------
TOTAL                   130      5   97%
```

`--cov`选项接受应生成报告的源文件路径，因此根据项目的布局，您应传递您的`src`或包目录。

您还可以使用`--cov-report`选项以生成各种格式的报告：XML，annotate和HTML。后者特别适用于本地使用，因为它生成HTML文件，显示您的代码，未覆盖的行以红色突出显示，非常容易找到这些未覆盖的地方。

此插件还可以与`pytest-xdist`直接使用。

最后，此插件生成的`.coverage`文件与许多提供覆盖跟踪和报告的在线服务兼容，例如`coveralls.io`（[https://coveralls.io/](https://coveralls.io/)）和`codecov.io`（[https://codecov.io/](https://codecov.io/)）。

# pytest-faulthandler

此插件在运行测试时自动启用内置的`faulthandler`（[https://docs.python.org/3/library/faulthandler.html](https://docs.python.org/3/library/faulthandler.html)）模块，该模块在灾难性情况下（如分段错误）输出Python回溯。安装后，无需其他设置或标志；`faulthandler`模块将自动启用。

如果您经常使用用C/C++编写的扩展模块，则强烈建议使用此插件，因为这些模块更容易崩溃。

# pytest-mock

`pytest-mock`插件提供了一个fixture，允许pytest和标准库的`unittest.mock`（[https://docs.python.org/3/library/unittest.mock.html](https://docs.python.org/3/library/unittest.mock.html)）模块之间更顺畅地集成。它提供了类似于内置的`monkeypatch` fixture的功能，但是`unittest.mock`产生的模拟对象还记录有关它们如何被访问的信息。这使得许多常见的测试任务更容易，例如验证已调用模拟函数以及使用哪些参数。

该插件提供了一个`mocker` fixture，可用于修补类和方法。使用上一章中的`getpass`示例，以下是您可以使用此插件编写它的方式：

```py
import getpass

def test_login_success(mocker):
    mocked = mocker.patch.object(getpass, "getpass", 
                                 return_value="valid-pass")
    assert user_login("test-user")
    mocked.assert_called_with("enter password: ")
```

请注意，除了替换`getpass.getpass()`并始终返回相同的值之外，我们还可以确保`getpass`函数已使用正确的参数调用。

在使用此插件时，与上一章中如何以及在哪里修补`monkeypatch` fixture的建议也适用。

# pytest-django

顾名思义，此插件允许您使用pytest测试您的`Django`（[https://www.djangoproject.com/](https://www.djangoproject.com/)）应用程序。`Django`是当今最著名的Web框架之一。

该插件提供了大量功能：

+   一个非常好的快速入门教程

+   命令行和`pytest.ini`选项来配置Django

+   与`pytest-xdist`兼容

+   使用`django_db`标记访问数据库，在测试之间自动回滚事务，以及一堆fixture，让您控制数据库的管理方式

+   用于向应用程序发出请求的fixture：`client`，`admin_client`和`admin_user`

+   在后台线程中运行`Django`服务器的`live_server` fixture

总的来说，这是生态系统中最完整的插件之一，具有太多功能无法在此处覆盖。对于`Django`应用程序来说，这是必不可少的，因此请务必查看其广泛的文档。

# pytest-flakes

此插件允许您使用`pyflakes`（[https://pypi.org/project/pyflakes/](https://pypi.org/project/pyflakes/)）检查您的代码，这是一个用于常见错误的源文件的静态检查器，例如丢失的导入和未知变量。

安装后，使用`--flakes`选项来激活它：

```py
λ pytest pytest-flakes.py --flake
...
============================= FAILURES ==============================
__________________________ pyflakes-check ___________________________
CH5\pytest-flakes.py:1: UnusedImport
'os' imported but unused
CH5\pytest-flakes.py:6: UndefinedName
undefined name 'unknown'
```

这将在你的正常测试中运行flake检查，使其成为保持代码整洁和防止一些错误的简单而廉价的方法。该插件还保留了自上次检查以来未更改的文件的本地缓存，因此在本地使用起来快速和方便。

# pytest-asyncio

`asyncio` ([https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html))模块是Python 3的热门新功能之一，提供了一个新的用于异步应用程序的框架。`pytest-asyncio`插件让你编写异步测试函数，轻松测试你的异步代码。

你只需要将你的测试函数标记为`async def`并使用`asyncio`标记：

```py
@pytest.mark.asyncio
async def test_fetch_requests():
    requests = await fetch_requests("example.com/api")
    assert len(requests) == 2
```

该插件还在后台管理事件循环，提供了一些选项，以便在需要使用自定义事件循环时进行更改。

当然，你可以在异步函数之外拥有正常的同步测试函数。

# pytest-trio

Trio的座右铭是“Pythonic async I/O for humans” ([https://trio.readthedocs.io/en/latest/](https://trio.readthedocs.io/en/latest/))。它使用与`asyncio`标准模块相同的`async def`/`await`关键字，但被认为更简单和更友好，包含一些关于如何处理超时和一组并行任务的新颖想法，以避免并行编程中的常见错误。如果你对异步开发感兴趣，它绝对值得一试。

`pytest-trio`的工作方式类似于`pytest-asyncio`：你编写异步测试函数，并使用`trio`标记它们。它还提供了其他功能，使测试更容易和更可靠，例如可控的时钟用于测试超时，处理任务的特殊函数，模拟网络套接字和流，以及更多。

# pytest-tornado

Tornado ([http://www.tornadoweb.org/en/stable/](http://www.tornadoweb.org/en/stable/))是一个Web框架和异步网络库。它非常成熟，在Python 2和3中工作，标准的`asyncio`模块从中借鉴了许多想法和概念。

`pytest-asyncio`受`pytest-tornado`的启发，因此它使用相同的想法，使用`gen_test`来标记你的测试为协程。它使用`yield`关键字而不是`await`，因为它支持Python 2，但除此之外它看起来非常相似：

```py
@pytest.mark.gen_test
def test_tornado(http_client):
    url = "https://docs.pytest.org/en/latest"
    response = yield http_client.fetch(url)
    assert response.code == 200
```

# pytest-postgresql

该插件允许你测试需要运行的PostgreSQL数据库的代码。

以下是它的一个快速示例：

```py
def test_fetch_series(postgresql):
    cur = postgresql.cursor()
    cur.execute('SELECT * FROM comedy_series;')
    assert len(cur.fetchall()) == 5
    cur.close()
```

它提供了两个fixtures：

+   `postgresql`：一个客户端fixture，启动并关闭到正在运行的测试数据库的连接。在测试结束时，它会删除测试数据库，以确保测试不会相互干扰。

+   `postgresql_proc`：一个会话范围的fixture，每个会话启动一次PostgreSQL进程，并确保在结束时停止。

它还提供了几个配置选项，用于连接和配置测试数据库。

# docker-services

该插件启动和管理你需要的Docker服务，以便测试你的代码。这使得运行测试变得简单，因为你不需要手动启动服务；插件将在测试会话期间根据需要启动和停止它们。

你可以使用`.services.yaml`文件来配置服务；这里是一个简单的例子：

```py
database:
    image: postgres
    environment:
        POSTGRES_USERNAME: pytest-user
        POSTGRES_PASSWORD: pytest-pass
        POSTGRES_DB: test
    image: regis:10 
```

这将启动两个服务：`postgres`和`redis`。

有了这个，剩下的就是用以下命令运行你的套件：

```py
pytest --docker-services
```

插件会处理剩下的事情。

# pytest-selenium

Selenium是一个针对自动化浏览器的框架，用于测试Web应用程序 ([https://www.seleniumhq.org/](https://www.seleniumhq.org/))。它可以做诸如打开网页、点击按钮，然后确保某个页面加载等事情。它支持所有主流浏览器，并拥有一个蓬勃发展的社区。

`pytest-selenium`提供了一个fixture，让你编写测试来完成所有这些事情，它会为你设置`Selenium`。

以下是如何访问页面，点击链接并检查加载页面的标题的基本示例：

```py
def test_visit_pytest(selenium):
    selenium.get("https://docs.pytest.org/en/latest/")
    assert "helps you write better programs" in selenium.title
    elem = selenium.find_element_by_link_text("Contents")
    elem.click()
    assert "Full pytest documentation" in selenium.title
```

`Selenium`和`pytest-selenium`足够复杂，可以测试从静态页面到完整的单页前端应用程序的各种应用。

# pytest-html

`pytest-html` 生成美丽的HTML测试结果报告。安装插件后，只需运行以下命令：

```py
λ pytest --html=report.html
```

这将在测试会话结束时生成一个`report.html`文件。

因为图片胜过千言万语，这里有一个例子：

![](assets/f71502c4-fb19-427a-8299-7d04fbb01c59.png)

报告可以在Web服务器上进行服务以便更轻松地查看，而且它们包含了一些很好的功能，比如复选框来显示/隐藏不同类型的测试结果，还有其他插件如`pytest-selenium`甚至能够在失败的测试中附加截图，就像前面的图片一样。

它绝对值得一试。

# pytest-cpp

为了证明pytest框架非常灵活，`pytest-cpp`插件允许你运行用Google Test ([https://github.com/google/googletest](https://github.com/google/googletest)) 或Boost.Test ([https://www.boost.org](https://www.boost.org))编写的测试，这些是用C++语言编写和运行测试的框架。

安装后，你只需要像平常一样运行pytest：

```py
λ pytest bin/tests
```

Pytest将找到包含测试用例的可执行文件，并自动检测它们是用`Google Test`还是`Boost.Python`编写的。它将正常运行测试并报告结果，格式整齐，熟悉pytest用户。

使用pytest运行这些测试意味着它们现在可以利用一些功能，比如使用`pytest-xdist`进行并行运行，使用`-k`进行测试选择，生成JUnitXML报告等等。这个插件对于使用Python和C++的代码库特别有用，因为它允许你用一个命令运行所有测试，并且你可以得到一个独特的报告。

# pytest-timeout

`pytest-timeout`插件在测试达到一定超时后会自动终止测试。

你可以通过在命令行中设置全局超时来使用它：

```py
λ pytest --timeout=60
```

或者你可以使用`@pytest.mark.timeout`标记单独的测试：

```py
@pytest.mark.timeout(600)
def test_long_simulation():
   ...
```

它通过以下两种方法之一来实现超时机制：

+   `thread`：在测试设置期间，插件启动一个线程，该线程休眠指定的超时时间。如果线程醒来，它将将所有线程的回溯信息转储到`stderr`并杀死当前进程。如果测试在线程醒来之前完成，那么线程将被取消，测试继续运行。这是在所有平台上都有效的方法。

+   `signal`：在测试设置期间安排了一个`SIGALRM`，并在测试完成时取消。如果警报被触发，它将将所有线程的回溯信息转储到`stderr`并失败测试，但它将允许测试继续运行。与线程方法相比的优势是当超时发生时它不会取消整个运行，但它不支持所有平台。

该方法会根据平台自动选择，但可以在命令行或通过`@pytest.mark.timeout`的`method=`参数来进行更改。

这个插件在大型测试套件中是不可或缺的，以避免测试挂起CI。

# pytest-annotate

Pyannotate ([https://github.com/dropbox/pyannotate](https://github.com/dropbox/pyannotate)) 是一个观察运行时类型信息并将该信息插入到源代码中的项目，而`pytest-annotate`使得在pytest中使用它变得很容易。

让我们回到这个简单的测试用例：

```py
def highest_rated(series):
    return sorted(series, key=itemgetter(2))[-1][0]

def test_highest_rated():
    series = [
        ("The Office", 2005, 8.8),
        ("Scrubs", 2001, 8.4),
        ("IT Crowd", 2006, 8.5),
        ("Parks and Recreation", 2009, 8.6),
        ("Seinfeld", 1989, 8.9),
    ]
    assert highest_rated(series) == "Seinfeld"
```

安装了`pytest-annotate`后，我们可以通过传递`--annotations-output`标志来生成一个注释文件：

```py
λ pytest --annotate-output=annotations.json
```

这将像往常一样运行测试套件，但它将收集类型信息以供以后使用。

之后，你可以调用`PyAnnotate`将类型信息直接应用到源代码中：

```py
λ pyannotate --type-info annotations.json -w
Refactored test_series.py
--- test_series.py (original)
+++ test_series.py (refactored)
@@ -1,11 +1,15 @@
 from operator import itemgetter
+from typing import List
+from typing import Tuple

 def highest_rated(series):
+    # type: (List[Tuple[str, int, float]]) -> str
 return sorted(series, key=itemgetter(2))[-1][0]

 def test_highest_rated():
+    # type: () -> None
 series = [
 ("The Office", 2005, 8.8),
 ("Scrubs", 2001, 8.4),
Files that were modified:
pytest-annotate.py
```

快速高效地注释大型代码库是非常整洁的，特别是如果该代码库已经有了完善的测试覆盖。

# pytest-qt

`pytest-qt`插件允许您为使用`Qt`框架（[https://www.qt.io/](https://www.qt.io/)）编写的GUI应用程序编写测试，支持更受欢迎的Python绑定集：`PyQt4`/`PyQt5`和`PySide`/`PySide2`。

它提供了一个`qtbot`装置，其中包含与GUI应用程序交互的方法，例如单击按钮、在字段中输入文本、等待窗口弹出等。以下是一个快速示例，展示了它的工作原理：

```py
def test_main_window(qtbot):
    widget = MainWindow()
    qtbot.addWidget(widget)

    qtbot.mouseClick(widget.about_button, QtCore.Qt.LeftButton)
    qtbot.waitUntil(widget.about_box.isVisible)
    assert widget.about_box.text() == 'This is a GUI App'
```

在这里，我们创建一个窗口，单击“关于”按钮，等待“关于”框弹出，然后确保它显示我们期望的文本。

它还包含其他好东西：

+   等待特定`Qt`信号的实用程序

+   自动捕获虚拟方法中的错误

+   自动捕获`Qt`日志消息

# pytest-randomly

测试理想情况下应该是相互独立的，确保在测试完成后进行清理，这样它们可以以任何顺序运行，而且不会以任何方式相互影响。

`pytest-randomly`通过随机排序测试，每次运行测试套件时更改它们的顺序，帮助您保持测试套件的真实性。这有助于检测测试是否具有隐藏的相互依赖性，否则您将无法发现。

它会在模块级别、类级别和函数顺序上对测试项进行洗牌。它还会在每个测试之前将`random.seed()`重置为一个固定的数字，该数字显示在测试部分的开头。可以在以后使用随机种子通过`--randomly-seed`命令行来重现失败。

作为额外的奖励，它还特别支持`factory boy`（[https://factoryboy.readthedocs.io/en/latest/reference.html](https://factoryboy.readthedocs.io/en/latest/reference.html)）、`faker`（[https://pypi.python.org/pypi/faker](https://pypi.python.org/pypi/faker)）和`numpy`（[http://www.numpy.org/](http://www.numpy.org/)）库，在每个测试之前重置它们的随机状态。

# pytest-datadir

通常，测试需要一个支持文件，例如一个包含有关喜剧系列数据的CSV文件，就像我们在上一章中看到的那样。`pytest-datadir`允许您将文件保存在测试旁边，并以安全的方式从测试中轻松访问它们。

假设您有这样的文件结构：

```py
tests/
    test_series.py
```

除此之外，您还有一个`series.csv`文件，需要从`test_series.py`中定义的测试中访问。

安装了`pytest-datadir`后，您只需要在相同目录中创建一个与测试文件同名的目录，并将文件放在其中：

```py
tests/
 test_series/
 series.csv
    test_series.py
```

`test_series`目录和`series.csv`应该保存到您的版本控制系统中。

现在，`test_series.py`中的测试可以使用`datadir`装置来访问文件：

```py
def test_ratings(datadir):
    with open(datadir / "series.csv", "r", newline="") as f:
        data = list(csv.reader(f))
    ...
```

`datadir`是一个指向数据目录的Path实例（[https://docs.python.org/3/library/pathlib.html](https://docs.python.org/3/library/pathlib.html)）。

需要注意的一点是，当我们在测试中使用`datadir`装置时，我们并不是访问原始文件的路径，而是临时副本。这确保了测试可以修改数据目录中的文件，而不会影响其他测试，因为每个测试都有自己的副本。

# pytest-regressions

通常情况下，您的应用程序或库包含产生数据集作为结果的功能。

经常测试这些结果是很繁琐且容易出错的，产生了这样的测试：

```py
def test_obtain_series_asserts():
    data = obtain_series()
    assert data[0]["name"] == "The Office"
    assert data[0]["year"] == 2005
    assert data[0]["rating"] == 8.8
    assert data[1]["name"] == "Scrubs"
    assert data[1]["year"] == 2001
    ...
```

这很快就会变得老套。此外，如果任何断言失败，那么测试就会在那一点停止，您将不知道在那一点之后是否还有其他断言失败。换句话说，您无法清楚地了解整体失败的情况。最重要的是，这也是非常难以维护的，因为如果`obtain_series()`返回的数据发生变化，您将不得不进行繁琐且容易出错的代码更新任务。

`pytest-regressions`提供了解决这类问题的装置。像前面的例子一样，一般的数据是`data_regression`装置的工作：

```py
def test_obtain_series(data_regression):
    data = obtain_series()
    data_regression.check(data)
```

第一次执行此测试时，它将失败，并显示如下消息：

```py
...
E Failed: File not found in data directory, created:
E - CH5\test_series\test_obtain_series.yml
```

它将以一个格式良好的YAML文件的形式将传递给`data_regression.check()`的数据转储到`test_series.py`文件的数据目录中（这要归功于我们之前看到的`pytest-datadir`装置）：

```py
- name: The Office
  rating: 8.8
  year: 2005
- name: Scrubs
  rating: 8.4
  year: 2001
- name: IT Crowd
  rating: 8.5
  year: 2006
- name: Parks and Recreation
  rating: 8.6
  year: 2009
- name: Seinfeld
  rating: 8.9
  year: 1989
```

下次运行此测试时，`data_regression`现在将传递给`data_regressions.check()`的数据与数据目录中的`test_obtain_series.yml`中找到的数据进行比较。如果它们匹配，测试通过。

然而，如果数据发生了变化，测试将失败，并显示新数据与记录数据之间的差异：

```py
E AssertionError: FILES DIFFER:
E ---
E
E +++
E
E @@ -13,3 +13,6 @@
E
E  - name: Seinfeld
E    rating: 8.9
E    year: 1989
E +- name: Rock and Morty
E +  rating: 9.3
E +  year: 2013
```

在某些情况下，这可能是一个回归，这种情况下你可以在代码中找到错误。

但在这种情况下，新数据是*正确的；*你只需要用`--force-regen`标志运行pytest，`pytest-regressions`将为你更新数据文件的新内容：

```py
E Failed: Files differ and --force-regen set, regenerating file at:
E - CH5\test_series\test_obtain_series.yml
```

现在，如果我们再次运行测试，测试将通过，因为文件包含了新数据。

当你有数十个测试突然产生不同但正确的结果时，这将极大地节省时间。你可以通过单次pytest执行将它们全部更新。

我自己使用这个插件，我数不清它为我节省了多少时间。

# 值得一提的是

有太多好的插件无法放入本章。前面的示例只是一个小小的尝试，我试图在有用、有趣和展示插件架构的灵活性之间取得平衡。

以下是一些值得一提的其他插件：

+   `pytest-bdd`：pytest的行为驱动开发

+   `pytest-benchmark`：用于对代码进行基准测试的装置。它以彩色输出输出基准测试结果

+   `pytest-csv`：将测试状态输出为CSV文件

+   `pytest-docker-compose`：在测试运行期间使用Docker compose管理Docker容器

+   `pytest-excel`：以Excel格式输出测试状态报告

+   `pytest-git`：为需要处理git仓库的测试提供git装置

+   `pytest-json`：将测试状态输出为json文件

+   `pytest-leaks`：通过重复运行测试并比较引用计数来检测内存泄漏

+   `pytest-menu`：允许用户从控制台菜单中选择要运行的测试

+   `pytest-mongo`：MongoDB的进程和客户端装置

+   `pytest-mpl`：测试Matplotlib输出的图形的插件

+   `pytest-mysql`：MySQL的进程和客户端装置

+   `pytest-poo`：用"pile of poo"表情符号替换失败测试的`F`字符

+   `pytest-rabbitmq`：RabbitMQ的进程和客户端装置

+   `pytest-redis`：Redis的进程和客户端装置

+   `pytest-repeat`：重复所有测试或特定测试多次以查找间歇性故障

+   `pytest-replay`：保存测试运行并允许用户以后执行它们，以便重现崩溃和不稳定的测试

+   `pytest-rerunfailures`：标记可以运行多次以消除不稳定测试的测试

+   `pytest-sugar`：通过添加进度条、表情符号、即时失败等来改变pytest控制台的外观和感觉

+   `pytest-tap`：以TAP格式输出测试报告

+   `pytest-travis-fold`：在Travis CI构建日志中折叠捕获的输出和覆盖报告

+   `pytest-vagrant`：与vagrant boxes一起使用的pytest装置

+   `pytest-vcr`：使用简单的标记自动管理`VCR.py`磁带

+   `pytest-virtualenv`：提供一个虚拟环境装置来管理测试中的虚拟环境

+   `pytest-watch`：持续监视源代码的更改并重新运行pytest

+   `pytest-xvfb`：为UI测试运行`Xvfb`（虚拟帧缓冲区）

+   `tavern`：使用基于YAML的语法对API进行自动化测试

+   `xdoctest`：重写内置的doctests模块，使得编写和配置doctests更加容易

请记住，在撰写本文时，pytest插件的数量已经超过500个，所以一定要浏览插件列表，以便找到自己喜欢的东西。

# 总结

在本章中，我们看到了查找和安装插件是多么容易。我们还展示了一些我每天使用并且觉得有趣的插件。我希望这让你对pytest的可能性有所了解，但请探索大量的插件，看看是否有任何有用的。

创建自己的插件不是本书涵盖的主题，但如果你感兴趣，这里有一些资源可以帮助你入门：

+   pytest文档：编写插件（[https://docs.pytest.org/en/latest/writing_plugins.html](https://docs.pytest.org/en/latest/writing_plugins.html)）。

+   Brian Okken的关于pytest的精彩书籍《Python测试与pytest》，比本书更深入地探讨了如何编写自己的插件。

在下一章中，我们将学习如何将pytest与现有的基于`unittest`的测试套件一起使用，包括有关如何迁移它们并逐步使用更多pytest功能的提示和建议。
