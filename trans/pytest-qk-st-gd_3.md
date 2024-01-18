# fixtures

在上一章中，我们学习了如何有效地使用标记和参数化来跳过测试，将其标记为预期失败，并对其进行参数化，以避免重复。

现实世界中的测试通常需要创建资源或数据来进行操作：一个临时目录来输出一些文件，一个数据库连接来测试应用程序的I/O层，一个用于集成测试的Web服务器。这些都是更复杂的测试场景中所需的资源的例子。更复杂的资源通常需要在测试会话结束时进行清理：删除临时目录，清理并断开与数据库的连接，关闭Web服务器。此外，这些资源应该很容易地在测试之间共享，因为在测试过程中我们经常需要为不同的测试场景重用资源。一些资源创建成本很高，但因为它们是不可变的或者可以恢复到原始状态，所以应该只创建一次，并与需要它的所有测试共享，在最后一个需要它们的测试完成时销毁。

pytest最重要的功能之一是覆盖所有先前的要求和更多内容。

本章我们将涵盖以下内容：

+   引入fixtures

+   使用`conftest.py`文件共享fixtures

+   作用域

+   自动使用

+   参数化

+   使用fixtures中的标记

+   内置fixtures概述

+   提示/讨论

# 引入fixtures

大多数测试需要某种数据或资源来操作：

```py
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

这里，我们有一个(`series name`, `year`, `rating`)元组的列表，我们用它来测试`highest_rated`函数。在这里将数据内联到测试代码中对于孤立的测试效果很好，但通常你会有一个可以被多个测试使用的数据集。一种解决方法是将数据集复制到每个测试中：

```py
def test_highest_rated():
    series = [
        ("The Office", 2005, 8.8),
        ...,
    ]
    assert highest_rated(series) == "Seinfeld"

def test_oldest():
    series = [
        ("The Office", 2005, 8.8),
        ...,
    ]
    assert oldest(series) == "Seinfeld"
```

但这很快就会变得老套—此外，复制和粘贴东西会在长期内影响可维护性，例如，如果数据布局发生变化（例如，添加一个新项目到元组或演员阵容大小）。

# 进入fixtures

pytest对这个问题的解决方案是fixtures。fixtures用于提供测试所需的函数和方法。

它们是使用普通的Python函数和`@pytest.fixture`装饰器创建的：

```py
@pytest.fixture
def comedy_series():
    return [
        ("The Office", 2005, 8.8),
        ("Scrubs", 2001, 8.4),
        ("IT Crowd", 2006, 8.5),
        ("Parks and Recreation", 2009, 8.6),
        ("Seinfeld", 1989, 8.9),
    ]
```

在这里，我们创建了一个名为`comedy_series`的fixture，它返回我们在上一节中使用的相同列表。

测试可以通过在其参数列表中声明fixture名称来访问fixtures。然后测试函数会接收fixture函数的返回值作为参数。这里是`comedy_series` fixture的使用：

```py
def test_highest_rated(comedy_series):
    assert highest_rated(comedy_series) == "Seinfeld"

def test_oldest(comedy_series):
    assert oldest(comedy_series) == "Seinfeld"
```

事情是这样的：

+   pytest在调用测试函数之前查看测试函数的参数。这里，我们有一个参数：`comedy_series`。

+   对于每个参数，pytest获取相同名称的fixture函数并执行它。

+   每个fixture函数的返回值成为一个命名参数，并调用测试函数。

请注意，`test_highest_rated`和`test_oldest`各自获得喜剧系列列表的副本，因此如果它们在测试中更改列表，它们不会相互干扰。

还可以使用方法在类中创建fixtures：

```py
class Test:

    @pytest.fixture
    def drama_series(self):
        return [
            ("The Mentalist", 2008, 8.1),
            ("Game of Thrones", 2011, 9.5),
            ("The Newsroom", 2012, 8.6),
            ("Cosmos", 1980, 9.3),
        ]
```

在测试类中定义的fixtures只能被类或子类的测试方法访问：

```py
class Test:
    ...

    def test_highest_rated(self, drama_series):
        assert highest_rated(drama_series) == "Game of Thrones"

    def test_oldest(self, drama_series):
        assert oldest(drama_series) == "Cosmos"
```

请注意，测试类可能有其他非测试方法，就像任何其他类一样。

# 设置/拆卸

正如我们在介绍中看到的，测试中使用的资源通常需要在测试完成后进行某种清理。

在我们之前的例子中，我们有一个非常小的数据集，所以在fixture中内联它是可以的。然而，假设我们有一个更大的数据集（比如，1000个条目），那么在代码中写入它会影响可读性。通常，数据集在外部文件中，例如CSV格式，因此将其移植到Python代码中是一件痛苦的事情。

解决方法是将包含系列数据集的CSV文件提交到存储库中，并在测试中使用内置的`csv`模块进行读取；有关更多详细信息，请访问[https://docs.python.org/3/library/csv.html](https://docs.python.org/3/library/csv.html)。

我们可以更改`comedy_series` fixture来实现这一点：

```py
@pytest.fixture
def comedy_series():
    file = open("series.csv", "r", newline="")
    return list(csv.reader(file))
```

这样做是有效的，但是我们作为认真的开发人员，希望能够正确关闭该文件。我们如何使用fixtures做到这一点呢？

Fixture清理通常被称为**teardown**，并且可以使用`yield`语句轻松支持：

```py
@pytest.fixture
def some_fixture():
    value = setup_value()
    yield value
    teardown_value(value)
```

通过使用`yield`而不是`return`，会发生以下情况：

+   fixture函数被调用

+   它执行直到yield语句，其中暂停并产生fixture值

+   测试执行，接收fixture值作为参数

+   无论测试是否通过，函数都会恢复执行，以执行其清理操作

对于熟悉它的人来说，这与**上下文管理器**（[https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager](https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager)）非常相似，只是您不需要用try/except子句将yield语句包围起来，以确保在发生异常时仍执行yield后的代码块。

让我们回到我们的例子；现在我们可以使用`yield`而不是`return`并关闭文件：

```py
@pytest.fixture
def comedy_series():
    file = open("series.csv", "r", newline="")
    yield list(csv.reader(file))
    file.close()
```

这很好，但请注意，因为`yield`与文件对象的`with`语句配合得很好，我们可以这样写：

```py
@pytest.fixture
def comedy_series():
    with open("series.csv", "r", newline="") as file:
        return list(csv.reader(file))
```

测试完成后，`with`语句会自动关闭文件，这更短，被认为更符合Python风格。

太棒了。

# 可组合性

假设我们收到一个新的series.csv文件，其中包含更多的电视系列，包括以前的喜剧系列和许多其他类型。我们希望为一些其他测试使用这些新数据，但我们希望保持现有的测试与以前一样工作。

在pytest中，fixture可以通过声明它们为参数轻松依赖于其他fixtures。利用这一特性，我们能够创建一个新的series fixture，从`series.csv`中读取所有数据（现在包含更多类型），并将我们的`comedy_series` fixture更改为仅过滤出喜剧系列：

```py
@pytest.fixture
def series():
    with open("series.csv", "r", newline="") as file:
        return list(csv.reader(file))

@pytest.fixture
def comedy_series(series):
    return [x for x in series if x[GENRE] == "comedy"]
```

使用`comedy_series`的测试保持不变：

```py
def test_highest_rated(comedy_series):
    assert highest_rated(comedy_series) == "Seinfeld"

def test_oldest(comedy_series):
    assert oldest(comedy_series) == "Seinfeld"
```

请注意，由于这些特性，fixtures是依赖注入的一个典型例子，这是一种技术，其中函数或对象声明其依赖关系，但否则不知道或不关心这些依赖关系将如何创建，或者由谁创建。这使它们非常模块化和可重用。

# 使用conftest.py文件共享fixtures

假设我们需要在其他测试模块中使用前一节中的`comedy_series` fixture。在pytest中，通过将fixture代码移动到`conftest.py`文件中，可以轻松共享fixtures。

`conftest.py`文件是一个普通的Python模块，只是它会被pytest自动加载，并且其中定义的任何fixtures都会自动对同一目录及以下的测试模块可用。考虑一下这个测试模块的层次结构：

```py
tests/
    ratings/
        series.csv
        test_ranking.py
    io/
        conftest.py
        test_formats.py 
    conftest.py

```

`tests/conftest.py`文件位于层次结构的根目录，因此在该项目中，任何在其中定义的fixtures都会自动对所有其他测试模块可用。在`tests/io/conftest.py`中定义的fixtures将仅对`tests/io`及以下模块可用，因此目前仅对`test_formats.py`可用。

这可能看起来不像什么大不了的事，但它使共享fixtures变得轻而易举：当编写测试模块时，能够从小处开始使用一些fixtures，知道如果将来这些fixtures对其他测试有用，只需将fixtures移动到`conftest.py`中即可。这避免了复制和粘贴测试数据的诱惑，或者花费太多时间考虑如何从一开始组织测试支持代码，以避免以后进行大量重构。

# 作用域

夹具总是在测试函数请求它们时创建的，通过在参数列表上声明它们，就像我们已经看到的那样。默认情况下，每个夹具在每个测试完成时都会被销毁。

正如本章开头提到的，一些夹具可能很昂贵，需要创建或设置，因此尽可能少地创建实例将非常有帮助，以节省时间。以下是一些示例：

+   初始化数据库表

+   例如，从磁盘读取缓存数据，大型CSV数据

+   启动外部服务

为了解决这个问题，pytest中的夹具可以具有不同的**范围**。夹具的范围定义了夹具应该在何时清理。在夹具没有清理的情况下，请求夹具的测试将收到相同的夹具值。

`@pytest.fixture`装饰器的范围参数用于设置夹具的范围：

```py
@pytest.fixture(scope="session")
def db_connection():
    ...
```

以下范围可用：

+   `scope="session"`：当所有测试完成时，夹具被拆除。

+   `scope="module"`：当模块的最后一个测试函数完成时，夹具被拆除。

+   `scope="class"`：当类的最后一个测试方法完成时，夹具被拆除。

+   `scope="function"`：当请求它的测试函数完成时，夹具被拆除。这是默认值。

重要的是要强调，无论范围如何，每个夹具都只会在测试函数需要它时才会被创建。例如，会话范围的夹具不一定会在会话开始时创建，而是只有在第一个请求它的测试即将被调用时才会创建。当考虑到并非所有测试都可能需要会话范围的夹具，并且有各种形式只运行一部分测试时，这是有意义的，正如我们在前几章中所看到的。

# 范围的作用

为了展示作用域，让我们看一下在测试涉及某种数据库时使用的常见模式。在即将到来的示例中，不要关注数据库API（无论如何都是虚构的），而是关注涉及的夹具的概念和设计。

通常，连接到数据库和表的创建都很慢。如果数据库支持事务，即执行可以原子地应用或丢弃的一组更改的能力，那么可以使用以下模式。

首先，我们可以使用会话范围的夹具连接和初始化我们需要的表的数据库：

```py
@pytest.fixture(scope="session")
def db():
    db = connect_to_db("localhost", "test") 
    db.create_table(Series)
    db.create_table(Actors)
    yield db
    db.prune()
    db.disconnect()
```

请注意，我们会在夹具结束时修剪测试数据库并断开与其的连接，这将在会话结束时发生。

通过`db`夹具，我们可以在所有测试中共享相同的数据库。这很棒，因为它节省了时间。但它也有一个缺点，现在测试可以更改数据库并影响其他测试。为了解决这个问题，我们创建了一个事务夹具，在测试开始之前启动一个新的事务，并在测试完成时回滚事务，确保数据库返回到其先前的状态：

```py
@pytest.fixture(scope="function")
def transaction(db):
    transaction = db.start_transaction()
    yield transaction
    transaction.rollback()
```

请注意，我们的事务夹具依赖于`db`。现在测试可以使用事务夹具随意读写数据库，而不必担心为其他测试清理它：

```py
def test_insert(transaction):
    transaction.add(Series("The Office", 2005, 8.8))
    assert transaction.find(name="The Office") is not None
```

有了这两个夹具，我们就有了一个非常坚实的基础来编写我们的数据库测试：需要事务夹具的第一个测试将通过`db`夹具自动初始化数据库，并且从现在开始，每个需要执行事务的测试都将从一个原始的数据库中执行。

不同范围夹具之间的可组合性非常强大，并且使得在现实世界的测试套件中可以实现各种巧妙的设计。

# 自动使用

可以通过将`autouse=True`传递给`@pytest.fixture`装饰器，将夹具应用于层次结构中的所有测试，即使测试没有明确请求夹具。当我们需要在每个测试之前和/或之后无条件地应用副作用时，这是有用的。

```py
@pytest.fixture(autouse=True)
def setup_dev_environment():
    previous = os.environ.get('APP_ENV', '')
    os.environ['APP_ENV'] = 'TESTING'
    yield
    os.environ['APP_ENV'] = previous
```

自动使用的夹具适用于夹具可供使用的所有测试：

+   与夹具相同的模块

+   在方法定义的情况下，与装置相同的类。

+   如果装置在`conftest.py`文件中定义，那么在相同目录或以下目录中的测试

换句话说，如果一个测试可以通过在参数列表中声明它来访问一个`autouse`装置，那么该测试将自动使用`autouse`装置。请注意，如果测试函数对装置的返回值感兴趣，它可能会将`autouse`装置添加到其参数列表中，就像正常情况一样。

# @pytest.mark.usefixtures

`@pytest.mark.usefixtures`标记可用于将一个或多个装置应用于测试，就好像它们在参数列表中声明了装置名称一样。在您希望所有组中的测试始终使用不是`autouse`的装置的情况下，这可能是一种替代方法。

例如，下面的代码将确保`TestVirtualEnv`类中的所有测试方法在一个全新的虚拟环境中执行：

```py
@pytest.fixture
def venv_dir():
    import venv

    with tempfile.TemporaryDirectory() as d:
        venv.create(d)
        pwd = os.getcwd()
        os.chdir(d)
        yield d
        os.chdir(pwd)

@pytest.mark.usefixtures('venv_dir')
class TestVirtualEnv:
    ...
```

正如名称所示，您可以将多个装置名称传递给装饰器：

```py
@pytest.mark.usefixtures("venv_dir", "config_python_debug")
class Test:
    ...
```

# 参数化装置

装置也可以直接进行参数化。当一个装置被参数化时，所有使用该装置的测试现在将多次运行，每个参数运行一次。当我们有装置的变体，并且每个使用该装置的测试也应该与所有变体一起运行时，这是一个很好的工具。

在上一章中，我们看到了使用序列化器的多个实现进行参数化的示例：

```py
@pytest.mark.parametrize(
    "serializer_class",
    [JSONSerializer, XMLSerializer, YAMLSerializer],
)
class Test:

    def test_quantity(self, serializer_class):
        serializer = serializer_class()
        quantity = Quantity(10, "m")
        data = serializer.serialize_quantity(quantity)
        new_quantity = serializer.deserialize_quantity(data)
        assert new_quantity == quantity

    def test_pipe(self, serializer_class):
        serializer = serializer_class()
        pipe = Pipe(
            length=Quantity(1000, "m"), diameter=Quantity(35, "cm")
        )
       data = serializer.serialize_pipe(pipe)
       new_pipe = serializer.deserialize_pipe(data)
       assert new_pipe == pipe
```

我们可以更新示例以在装置上进行参数化：

```py
class Test:

 @pytest.fixture(params=[JSONSerializer, XMLSerializer,
 YAMLSerializer])
 def serializer(self, request):
 return request.param()

    def test_quantity(self, serializer):
        quantity = Quantity(10, "m")
        data = serializer.serialize_quantity(quantity)
        new_quantity = serializer.deserialize_quantity(data)
        assert new_quantity == quantity

    def test_pipe(self, serializer):
        pipe = Pipe(
            length=Quantity(1000, "m"), diameter=Quantity(35, "cm")
        )
        data = serializer.serialize_pipe(pipe)
        new_pipe = serializer.deserialize_pipe(data)
        assert new_pipe == pipe
```

请注意以下内容：

+   我们向装置定义传递了一个`params`参数。

+   我们使用`request`对象的特殊`param`属性在装置内部访问参数。当装置被参数化时，这个内置装置提供了对请求测试函数和参数的访问。我们将在本章后面更多地了解`request`装置。

+   在这种情况下，我们在装置内部实例化序列化器，而不是在每个测试中显式实例化。

可以看到，参数化装置与参数化测试非常相似，但有一个关键的区别：通过参数化装置，我们使所有使用该装置的测试针对所有参数化的实例运行，使它们成为`conftest.py`文件中共享的装置的绝佳解决方案。

当您向现有装置添加新参数时，看到自动执行了许多新测试是非常有益的。

# 使用装置标记

我们可以使用`request`装置来访问应用于测试函数的标记。

假设我们有一个`autouse`装置，它总是将当前区域初始化为英语：

```py
@pytest.fixture(autouse=True)
def setup_locale():
    locale.setlocale(locale.LC_ALL, "en_US")
    yield
    locale.setlocale(locale.LC_ALL, None)

def test_currency_us():
    assert locale.currency(10.5) == "$10.50"
```

但是，如果我们只想为一些测试使用不同的区域设置呢？

一种方法是使用自定义标记，并在我们的装置内部访问`mark`对象：

```py
@pytest.fixture(autouse=True)
def setup_locale(request):
    mark = request.node.get_closest_marker("change_locale")
    loc = mark.args[0] if mark is not None else "en_US"
    locale.setlocale(locale.LC_ALL, loc)
    yield
    locale.setlocale(locale.LC_ALL, None)

@pytest.mark.change_locale("pt_BR")
def test_currency_br():
    assert locale.currency(10.5) == "R$ 10,50"
```

标记可以用来将信息传递给装置。因为它有点隐式，所以我建议节俭使用，因为它可能导致难以理解的代码。

# 内置装置概述

让我们来看一些内置的pytest装置。

# tmpdir

`tmpdir`装置提供了一个在每次测试结束时自动删除的空目录：

```py
def test_empty(tmpdir):
    assert os.path.isdir(tmpdir)
    assert os.listdir(tmpdir) == []
```

作为`function`-scoped装置，每个测试都有自己的目录，因此它们不必担心清理或生成唯一的目录。

装置提供了一个`py.local`对象（[http://py.readthedocs.io/en/latest/path.html](http://py.readthedocs.io/en/latest/path.html)），来自`py`库（[http://py.readthedocs.io](http://py.readthedocs.io)），它提供了方便的方法来处理文件路径，比如连接，读取，写入，获取扩展名等等；它在哲学上类似于标准库中的`pathlib.Path`对象（[https://docs.python.org/3/library/pathlib.html](https://docs.python.org/3/library/pathlib.html)）：

```py
def test_save_curves(tmpdir):
    data = dict(status_code=200, values=[225, 300])
    fn = tmpdir.join('somefile.json')
    write_json(fn, data)
    assert fn.read() == '{"status_code": 200, "values": [225, 300]}'
```

为什么pytest使用`py.local`而不是`pathlib.Path`？

在`pathlib.Path`出现并被合并到标准库之前，Pytest已经存在多年了，而`py`库是当时路径类对象的最佳解决方案之一。核心pytest开发人员正在研究如何使pytest适应现在标准的`pathlib.Path`API。

# tmpdir_factory

`tmpdir`装置非常方便，但它只有`function`*-*scoped：这样做的缺点是它只能被其他`function`-scoped装置使用。

`tmpdir_factory`装置是一个*session-scoped*装置，允许在任何范围内创建空的唯一目录。当我们需要在其他范围的装置中存储数据时，例如`session`-scoped缓存或数据库文件时，这可能很有用。

为了展示它的作用，接下来显示的`images_dir`装置使用`tmpdir_factory`创建一个唯一的目录，整个测试会话中包含一系列示例图像文件：

```py
@pytest.fixture(scope='session')
def images_dir(tmpdir_factory):
    directory = tmpdir_factory.mktemp('images')
    download_images('https://example.com/samples.zip', directory)
    extract_images(directory / 'samples.zip')
    return directory
```

因为这将每个会话只执行一次，所以在运行测试时会节省我们相当多的时间。

然后测试可以使用`images_dir`装置轻松访问示例图像文件：

```py
def test_blur_filter(images_dir):
    output_image = apply_blur_filter(images_dir / 'rock1.png')
    ...
```

但请记住，此装置创建的目录是共享的，并且只会在测试会话结束时被删除。这意味着测试不应修改目录的内容；否则，它们可能会影响其他测试。

# 猴子补丁

在某些情况下，测试需要复杂或难以在测试环境中设置的功能，例如：

+   对外部资源的客户端（例如GitHub的API）需要在测试期间访问可能不切实际或成本太高

+   强制代码表现得好像在另一个平台上，比如错误处理

+   复杂的条件或难以在本地或CI中重现的环境

`monkeypatch`装置允许您使用其他对象和函数干净地覆盖正在测试的系统的函数、对象和字典条目，并在测试拆卸期间撤消所有更改。例如：

```py
import getpass

def user_login(name):
    password = getpass.getpass()
    check_credentials(name, password)
    ...
```

在这段代码中，`user_login`使用标准库中的`getpass.getpass()`函数（[https://docs.python.org/3/library/getpass.html](https://docs.python.org/3/library/getpass.html)）以系统中最安全的方式提示用户输入密码。在测试期间很难模拟实际输入密码，因为`getpass`尝试直接从终端读取（而不是从`sys.stdin`）。

我们可以使用`monkeypatch`装置来在测试中绕过对`getpass`的调用，透明地而不改变应用程序代码：

```py
def test_login_success(monkeypatch):
    monkeypatch.setattr(getpass, "getpass", lambda: "valid-pass")
    assert user_login("test-user")

def test_login_wrong_password(monkeypatch):
    monkeypatch.setattr(getpass, "getpass", lambda: "wrong-pass")
    with pytest.raises(AuthenticationError, match="wrong password"):
        user_login("test-user")
```

在测试中，我们使用`monkeypatch.setattr`来用一个虚拟的`lambda`替换`getpass`模块的真实`getpass()`函数，它返回一个硬编码的密码。在`test_login_success`中，我们返回一个已知的好密码，以确保用户可以成功进行身份验证，而在`test_login_wrong_password`中，我们使用一个错误的密码来确保正确处理身份验证错误。如前所述，原始的`getpass()`函数会在测试结束时自动恢复，确保我们不会将该更改泄漏到系统中的其他测试中。

# 如何和在哪里修补

`monkeypatch`装置通过用另一个对象（通常称为*模拟*）替换对象的属性来工作，在测试结束时恢复原始对象。使用此装置的常见问题是修补错误的对象，这会导致调用原始函数/对象而不是模拟函数/对象。

要理解问题，我们需要了解Python中`import`和`import from`的工作原理。

考虑一个名为`services.py`的模块：

```py
import subprocess

def start_service(service_name):
    subprocess.run(f"docker run {service_name}")
```

在这段代码中，我们导入`subprocess`模块并将`subprocess`模块对象引入`services.py`命名空间。这就是为什么我们调用`subprocess.run`：我们正在访问`services.py`命名空间中`subprocess`对象的`run`函数。

现在考虑稍微不同的以前的代码写法：

```py
from subprocess import run

def start_service(service_name):
    run(f"docker run {service_name}")
```

在这里，我们导入了`subprocess`模块，但将`run`函数对象带入了`service.py`命名空间。这就是为什么`run`可以直接在`start_service`中调用，而`subprocess`名称甚至不可用（如果尝试调用`subprocess.run`，将会得到`NameError`异常）。

我们需要意识到这种差异，以便正确地`monkeypatch`在`services.py`中使用`subprocess.run`。

在第一种情况下，我们需要替换`subprocess`模块的`run`函数，因为`start_service`就是这样使用它的：

```py
import subprocess
import services

def test_start_service(monkeypatch):
    commands = []
    monkeypatch.setattr(subprocess, "run", commands.append)
    services.start_service("web")
    assert commands == ["docker run web"]
```

在这段代码中，`services.py`和`test_services.py`都引用了相同的`subprocess`模块对象。

然而，在第二种情况下，`services.py`在自己的命名空间中引用了原始的`run`函数。因此，第二种情况的正确方法是替换`services.py`命名空间中的`run`函数：

```py
import services

def test_start_service(monkeypatch):
    commands = []
    monkeypatch.setattr(services, "run", commands.append)
    services.start_service("web")
    assert commands == ["docker run web"]
```

被测试代码导入需要进行monkeypatch的代码是人们经常被绊倒的原因，所以确保您首先查看代码。

# capsys/capfd

`capsys` fixture捕获了写入`sys.stdout`和`sys.stderr`的所有文本，并在测试期间使其可用。

假设我们有一个小的命令行脚本，并且希望在调用脚本时没有参数时检查使用说明是否正确：

```py
from textwrap import dedent

def script_main(args):
    if not args:
        show_usage()
        return 0
    ...

def show_usage():
    print("Create/update webhooks.")
    print(" Usage: hooks REPO URL")
```

在测试期间，我们可以使用`capsys` fixture访问捕获的输出。这个fixture有一个`capsys.readouterr()`方法，返回一个`namedtuple`([https://docs.python.org/3/library/collections.html#collections.namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple))，其中包含从`sys.stdout`和`sys.stderr`捕获的文本。

```py
def test_usage(capsys):
    script_main([])
    captured = capsys.readouterr()
    assert captured.out == dedent("""\
        Create/update webhooks.
          Usage: hooks REPO URL
    """)
```

还有`capfd` fixture，它的工作方式类似于`capsys`，只是它还捕获文件描述符`1`和`2`的输出。这使得可以捕获标准输出和标准错误，即使是对于扩展模块。

# 二进制模式

`capsysbinary`和`capfdbinary`是与`capsys`和`capfd`相同的fixtures，不同之处在于它们以二进制模式捕获输出，并且它们的`readouterr()`方法返回原始字节而不是文本。在特殊情况下可能会有用，例如运行生成二进制输出的外部进程时，如`tar`。

# request

`request` fixture是一个内部pytest fixture，提供有关请求测试的有用信息。它可以在测试函数和fixtures中声明，并提供以下属性：

+   `function`：Python `test`函数对象，可用于`function`-scoped fixtures。

+   `cls`/`instance`：Python类/实例的`test`方法对象，可用于`function`和`class`-scoped fixtures。如果fixture是从`test`函数请求的，而不是测试方法，则可以为`None`。

+   `module`：请求测试方法的Python模块对象，可用于`module`，`function`和`class`-scoped fixtures。

+   `session`：pytest的内部`Session`对象，它是测试会话的单例，代表集合树的根。它可用于所有范围的fixtures。

+   `node`：pytest集合节点，它包装了与fixture范围匹配的Python对象之一。

+   `addfinalizer(func)`: 添加一个将在测试结束时调用的`new finalizer`函数。finalizer函数将在不带参数的情况下调用。`addfinalizer`是在fixtures中执行拆卸的原始方法，但后来已被`yield`语句取代，主要用于向后兼容。

fixtures可以使用这些属性根据正在执行的测试自定义自己的行为。例如，我们可以创建一个fixture，使用当前测试名称作为临时目录的前缀，类似于内置的`tmpdir` fixture：

```py
@pytest.fixture
def tmp_path(request) -> Path:
    with TemporaryDirectory(prefix=request.node.name) as d:
        yield Path(d)

def test_tmp_path(tmp_path):
    assert list(tmp_path.iterdir()) == []
```

在我的系统上执行此代码时创建了以下目录：

```py
C:\Users\Bruno\AppData\Local\Temp\test_tmp_patht5w0cvd0
```

`request` fixture 可以在您想要根据正在执行的测试的属性自定义 fixture，或者访问应用于测试函数的标记时使用，正如我们在前面的部分中所看到的。

# 提示/讨论

以下是一些未适应前面部分的短话题和提示，但我认为值得一提。

# 何时使用 fixture，而不是简单函数

有时，您只需要为测试构造一个简单的对象，可以说这可以通过一个普通函数来完成，不一定需要实现为 fixture。假设我们有一个不接收任何参数的 `WindowManager` 类：

```py
class WindowManager:
    ...
```

在我们的测试中使用它的一种方法是编写一个 fixture：

```py
@pytest.fixture
def manager():
 return WindowManager()

def test_windows_creation(manager):
    window = manager.new_help_window("pipes_help.rst")
    assert window.title() == "Pipe Setup Help"
```

或者，您可以主张为这样简单的用法编写一个 fixture 是过度的，并且使用一个普通函数代替：

```py
def create_window_manager():
    return WindowManager()

def test_windows_creation():
    manager = create_window_manager()
    window = manager.new_help_window("pipes_help.rst")
    assert window.title() == "Pipe Setup Help"
```

或者您甚至可以在每个测试中显式创建管理器：

```py
def test_windows_creation():
    manager = WindowManager()
    window = manager.new_help_window("pipes_help.rst")
    assert window.title() == "Pipe Setup Help"
```

这是完全可以的，特别是如果在单个模块中的少数测试中使用。

然而，请记住，fixture **抽象了对象的构建和拆卸过程的细节**。在决定放弃 fixture 而选择普通函数时，这一点至关重要。

假设我们的 `WindowManager` 现在需要显式关闭，或者它需要一个本地目录用于记录目的：

```py
class WindowManager:

    def __init__(self, logging_directory):
        ...

    def close(self):
        """
        Close the WindowManager and all associated resources. 
        """
        ...
```

如果我们一直在使用像第一个例子中给出的 fixture，我们只需更新 fixture 函数，**测试根本不需要改变**：

```py
@pytest.fixture
def manager(tmpdir):
    wm = WindowManager(str(tmpdir))
    yield wm
 wm.close()
```

但是，如果我们选择使用一个普通函数，现在我们**必须更新调用我们函数的所有地方**：我们需要传递一个记录目录，并确保在测试结束时调用 `.close()`：

```py
def create_window_manager(tmpdir, request):
    wm = WindowManager(str(tmpdir))
    request.addfinalizer(wm.close)
    return wm

def test_windows_creation(tmpdir, request):
    manager = create_window_manager(tmpdir, request)
    window = manager.new_help_window("pipes_help.rst")
    assert window.title() == "Pipe Setup Help"
```

根据这个函数在我们的测试中被使用的次数，这可能是一个相当大的重构。

这个信息是：当底层对象简单且不太可能改变时，使用普通函数是可以的，但请记住，fixture 抽象了对象的创建/销毁的细节，它们可能在将来需要更改。另一方面，使用 fixture 创建了另一个间接层，稍微增加了代码复杂性。最终，这是一个需要您权衡的平衡。

# 重命名 fixture

`@pytest.fixture` 装饰器接受一个 `name` 参数，该参数可用于指定 fixture 的名称，与 fixture 函数不同：

```py
@pytest.fixture(name="venv_dir")
def _venv_dir():
    ...
```

这是有用的，因为有一些烦恼可能会影响用户在使用在相同模块中声明的 fixture 时：

+   如果用户忘记在测试函数的参数列表中声明 fixture，他们将得到一个 `NameError`，而不是 fixture 函数对象（因为它们在同一个模块中）。

+   一些 linters 抱怨测试函数参数遮蔽了 fixture 函数。

如果之前的烦恼经常发生，您可能会将这视为团队中的一个良好实践。请记住，这些问题只会发生在测试模块中定义的 fixture 中，而不会发生在 `conftest.py` 文件中。

# 在 conftest 文件中优先使用本地导入

`conftest.py` 文件在收集期间被导入，因此它们直接影响您从命令行运行测试时的体验。因此，我建议在 `conftest.py` 文件中尽可能使用本地导入，以保持导入时间较短。

因此，不要使用这个：

```py
import pytest
import tempfile
from myapp import setup

@pytest.fixture
def setup_app():
    ...
```

优先使用本地导入：

```py
import pytest

@pytest.fixture
def setup_app():
 import tempfile
 from myapp import setup
    ...
```

这种做法对大型测试套件的启动有明显影响。

# fixture 作为测试支持代码

您应该将 fixture 视为不仅提供资源的手段，还提供测试的支持代码。通过支持代码，我指的是为测试提供高级功能的类。

例如，一个机器人框架可能会提供一个 fixture，用于测试您的机器人作为黑盒：

```py
def test_hello(bot):
    reply = bot.say("hello")
    assert reply.text == "Hey, how can I help you?"

def test_store_deploy_token(bot):
    assert bot.store["TEST"]["token"] is None
    reply = bot.say("my token is ASLKM8KJAN")
    assert reply.text == "OK, your token was saved"
    assert bot.store["TEST"]["token"] == "ASLKM8KJAN"
```

`bot` fixture允许开发人员与机器人交谈，验证响应，并检查框架处理的内部存储的内容，等等。它提供了一个高级接口，使得测试更容易编写和理解，即使对于那些不了解框架内部的人也是如此。

这种技术对应用程序很有用，因为它将使开发人员轻松愉快地添加新的测试。对于库来说也很有用，因为它们将为库的用户提供高级测试支持。

# 总结

在本章中，我们深入了解了pytest最著名的功能之一：fixtures。我们看到了它们如何被用来提供资源和测试功能，以及如何简洁地表达设置/拆卸代码。我们学会了如何共享fixtures，使用`conftest.py`文件；如何使用fixture scopes，避免为每个测试创建昂贵的资源；以及如何自动使用fixtures，这些fixtures会在同一模块或层次结构中的所有测试中执行。然后，我们学会了如何对fixtures进行参数化，并从中使用标记。我们对各种内置fixtures进行了概述，并在最后对fixtures进行了一些简短的讨论。希望您喜欢这一过程！

在下一章中，我们将探索一下广阔的pytest插件生态系统，这些插件都可以供您使用。
