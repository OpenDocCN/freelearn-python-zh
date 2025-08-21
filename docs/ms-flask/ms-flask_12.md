# 第十二章：测试 Flask 应用程序

在本书中，每当我们对应用程序的代码进行修改时，我们都必须手动将受影响的网页加载到浏览器中，以测试代码是否正确工作。随着应用程序的增长，这个过程变得越来越繁琐，特别是如果您更改了低级别且在各处都使用的东西，比如 SQLAlchemy 模型代码。

为了自动验证我们的代码是否按预期工作，我们将使用 Python 的内置功能，通常称为单元测试，对我们应用程序的代码进行检查。

# 什么是单元测试？

测试程序非常简单。它只涉及运行程序的特定部分，并说明您期望的结果，并将其与程序片段实际的结果进行比较。如果结果相同，则测试通过。如果结果不同，则测试失败。通常，在将代码提交到 Git 存储库之前以及在将代码部署到实时服务器之前运行这些测试，以确保破损的代码不会进入这两个系统。

在程序测试中，有三种主要类型的测试。单元测试是验证单个代码片段（如函数）正确性的测试。第二种是集成测试，它测试程序中各个单元一起工作的正确性。最后一种测试类型是系统测试，它测试整个系统的正确性，而不是单独的部分。

在本章中，我们将使用单元测试和系统测试来验证我们的代码是否按计划工作。在本章中，我们不会进行集成测试，因为代码中各部分的协同工作方式不是由我们编写的代码处理的。例如，SQLAlchemy 与 Flask 的工作方式不是由我们的代码处理的，而是由 Flask SQLAlchemy 处理的。

这带我们来到代码测试的第一个规则之一。为自己的代码编写测试。这样做的第一个原因是很可能已经为此编写了测试。第二个原因是，您使用的库中的任何错误都将在您想要使用该库的功能时在您的测试中显现出来。

# 测试是如何工作的？

让我们从一个非常简单的 Python 函数开始进行测试。

```py
def square(x):
    return x * x
```

为了验证此代码的正确性，我们传递一个值，并测试函数的结果是否符合我们的期望。例如，我们会给它一个输入为 5，并期望结果为 25。

为了说明这个概念，我们可以在命令行中使用`assert`语句手动测试这个函数。Python 中的`assert`语句简单地表示，如果`assert`关键字后的条件语句返回`False`，则抛出异常如下：

```py
$ python
>>> def square(x): 
...     return x * x
>>> assert square(5) == 25
>>> assert square(7) == 49
>>> assert square(10) == 100
>>> assert square(10) == 0
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
AssertionError

```

使用这些`assert`语句，我们验证了平方函数是否按预期工作。

# 单元测试应用程序

Python 中的单元测试通过将`assert`语句组合到它们自己的函数中的类中来工作。这个类中的测试函数集合被称为测试用例。测试用例中的每个函数应该只测试一件事，这是单元测试的主要思想。在单元测试中只测试一件事会迫使您逐个验证每个代码片段，而不会忽略代码的任何功能。如果编写单元测试正确，您最终会得到大量的单元测试。虽然这可能看起来过于冗长，但它将为您节省后续的麻烦。

在构建测试用例之前，我们需要另一个配置对象，专门用于设置应用程序进行测试。在这个配置中，我们将使用 Python 标准库中的`tempfile`模块，以便在文件中创建一个测试 SQLite 数据库，当测试结束时会自动删除。这样可以确保测试不会干扰我们的实际数据库。此外，该配置禁用了 WTForms CSRF 检查，以允许我们在测试中提交表单而无需 CSRF 令牌。

```py
import tempfile

class TestConfig(Config):
    db_file = tempfile.NamedTemporaryFile()

    DEBUG = True
    DEBUG_TB_ENABLED = False

    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + db_file.name

    CACHE_TYPE = 'null'
    WTF_CSRF_ENABLED = False

    CELERY_BROKER_URL = "amqp://guest:guest@localhost:5672//"
    CELERY_BACKEND_URL = "amqp://guest:guest@localhost:5672//"

    MAIL_SERVER = 'localhost'
    MAIL_PORT = 25
    MAIL_USERNAME = 'username'
    MAIL_PASSWORD = 'password'
```

## 测试路由功能

让我们构建我们的第一个测试用例。在这个测试用例中，我们将测试如果我们访问它们的 URL，路由函数是否成功返回响应。在项目目录的根目录中创建一个名为`tests`的新目录，然后创建一个名为`test_urls.py`的新文件，该文件将保存所有路由的单元测试。每个测试用例都应该有自己的文件，并且每个测试用例都应该专注于你正在测试的代码的一个区域。

在`test_urls.py`中，让我们开始创建内置的 Python`unittest`库所需的内容。该代码将使用 Python 中的`unittest`库来运行我们在测试用例中创建的所有测试。

```py
import unittest

class TestURLs(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
```

让我们看看当运行这段代码时会发生什么。我们将使用`unittest`库的自动查找测试用例的能力来运行测试。`unittest`库查找的模式是`test*.py`：

```py
$ python -m unittest discover

---------------------------------------------------------------------
Ran 0 tests in 0.000s

OK

```

因为测试用例中没有测试，所以测试用例成功通过。

### 注意

测试脚本是从脚本的父目录而不是测试文件夹本身运行的。这是为了允许在测试脚本中导入应用程序代码。

为了测试 URL，我们需要一种在不实际运行服务器的情况下查询应用程序路由的方法，以便返回我们的请求。Flask 提供了一种在测试中访问路由的方法，称为测试客户端。测试客户端提供了在我们的路由上创建 HTTP 请求的方法，而无需实际运行应用程序的`app.run()`。

在这个测试用例中，我们将需要测试客户端对象，但是在每个`unittest`中添加代码来创建测试客户端并没有太多意义，因为我们有`setUp`方法。`setUp`方法在每个单元测试之前运行，并且可以将变量附加到 self 上，以便测试方法可以访问它们。在我们的`setUp`方法中，我们需要使用我们的`TestConfig`对象创建应用程序对象，并创建测试客户端。

此外，我们需要解决三个问题。前两个在 Flask Admin 和 Flask Restful 扩展中，当应用程序对象被销毁时，它们内部存储的 Blueprint 对象不会被移除。第三，Flask SQLAlchemy 的初始化程序在`webapp`目录之外时无法正确添加应用程序对象：

```py
class TestURLs(unittest.TestCase):
    def setUp(self):
        # Bug workarounds
        admin._views = []
        rest_api.resources = []

        app = create_app('webapp.config.TestConfig')
        self.client = app.test_client()

        # Bug workaround
        db.app = app

        db.create_all()
```

### 注意

在撰写本文时，之前列出的所有错误都存在，但在阅读本章时可能已经不存在。

除了`setUp`方法之外，还有`tearDown`方法，它在每次单元测试结束时运行。`tearDown`方法用于销毁`setUp`方法中创建的任何无法自动垃圾回收的对象。在我们的情况下，我们将使用`tearDown`方法来删除测试数据库中的表，以便每个测试都有一个干净的起点。

```py
class TestURLs(unittest.TestCase):
    def setUp(self):
        …

    def tearDown(self):
        db.session.remove()
        db.drop_all()
```

现在我们可以创建我们的第一个单元测试。第一个测试将测试访问我们应用程序的根目录是否会返回`302 重定向`到博客主页，如下所示：

```py
class TestURLs(unittest.TestCase):
    def setUp(self):
        …

    def tearDown(self):
        …

    def test_root_redirect(self):
        """ Tests if the root URL gives a 302 """

        result = self.client.get('/')
        assert result.status_code == 302
        assert "/blog/" in result.headers['Location']
```

每个单元测试必须以单词`test`开头，以告诉`unittest`库该函数是一个单元测试，而不仅仅是测试用例类中的某个实用函数。

现在，如果我们再次运行测试，我们会看到我们的测试被运行并通过检查：

```py
$ python -m unittest discover
.
---------------------------------------------------------------------
Ran 1 tests in 0.128s

OK

```

编写测试的最佳方法是事先询问自己要寻找什么，编写`assert`语句，并编写执行这些断言所需的代码。这迫使您在开始编写测试之前询问自己真正要测试什么。为每个单元测试编写 Python 文档字符串也是最佳实践，因为每当测试失败时，它将与测试名称一起打印，并且在编写 50 多个测试后，了解测试的确切目的可能会有所帮助。

与使用 Python 的内置`assert`关键字不同，我们可以使用`unittest`库提供的一些方法。当这些函数内部的`assert`语句失败时，这些方法提供了专门的错误消息和调试信息。

以下是`unittest`库提供的所有特殊`assert`语句及其功能列表：

+   `assertEqual(x, y)`: 断言 `x == y`

+   `assertNotEqual(x, y)`: 断言 `x != y`

+   `assertTrue(x)`: 断言 `x` 是 `True`

+   `assertFalse(x)`: 断言 `x` 是 `False`

+   `assertIs(x, y)`: 断言 `x` 是 `y`

+   `assertIsNot(x, y)`: 断言 `x` 不是 `y`

+   `assertIsNone(x)`: 断言 `x` 是 `None`

+   `assertIsNotNone(x)`: 断言 `x` 不是 `None`

+   `assertIn(x, y)`: 断言 `x` 在 `y` 中

+   `assertNotIn(x, y)`: 断言 `x` 不在 `y` 中

+   `assertIsInstance(x, y)`: 断言 `isinstance(x, y)`

+   `assertNotIsInstance(x, y)`: 断言不是 `isinstance(x, y)`

如果我们想测试普通页面的返回值，单元测试将如下所示：

```py
class TestURLs(unittest.TestCase):
    def setUp(self):
        …

    def tearDown(self):
        …

    def test_root_redirect(self):
        …
```

请记住，此代码仅测试 URL 是否成功返回。返回数据的内容不是这些测试的一部分。

如果我们想测试提交登录表单之类的表单，可以使用测试客户端的 post 方法。让我们创建一个`test_login`方法来查看登录表单是否正常工作：

```py
class TestURLs(unittest.TestCase):
    …
    def test_login(self):
        """ Tests if the login form works correctly """

        test_role = Role("default")
        db.session.add(test_role)
        db.session.commit()

        test_user = User("test")
        test_user.set_password("test")
        db.session.add(test_user)
        db.session.commit()

        result = self.client.post('/login', data=dict(
            username='test',
            password="test"
        ), follow_redirects=True)

        self.assertEqual(result.status_code, 200)
        self.assertIn('You have been logged in', result.data)
```

对返回数据中字符串的额外检查是因为返回代码不受输入数据有效性的影响。post 方法将适用于测试本书中创建的任何表单对象。

现在您了解了单元测试的机制，可以使用单元测试来测试应用程序的所有部分。例如，测试应用程序中的所有路由，测试我们制作的任何实用函数，如`sidebar_data`，测试具有特定权限的用户是否可以访问页面等。

如果您的应用程序代码具有任何功能，无论多么小，都应该为其编写测试。为什么？因为任何可能出错的事情都会出错。如果您的应用程序代码的有效性完全依赖于手动测试，那么随着应用程序的增长，某些事情将被忽视。一旦有事情被忽视，就会将错误的代码部署到生产服务器上，这会让您的用户感到恼火。

# 用户界面测试

为了测试应用程序代码的高级别，并创建系统测试，我们将编写与浏览器一起工作的测试，并验证 UI 代码是否正常工作。使用一个名为 Selenium 的工具，我们将创建 Python 代码，纯粹通过代码来控制浏览器。您可以在屏幕上找到元素，然后通过 Selenium 对这些元素执行操作。单击它或输入按键。此外，Selenium 允许您通过访问元素的内容，例如其属性和内部文本，对页面内容执行检查。对于更高级的检查，Selenium 甚至提供了一个接口来在页面上运行任意 JavaScript。如果 JavaScript 返回一个值，它将自动转换为 Python 类型。

在触及代码之前，需要安装 Selenium：

```py
$ pip install selenium

```

要开始编写代码，我们的 UI 测试需要在名为`test_ui.py`的测试目录中拥有自己的文件。因为系统测试不测试特定的事物，编写用户界面测试的最佳方法是将测试视为模拟典型用户流程。在编写测试之前，写下我们的虚拟用户将模拟的具体步骤：

```py
import unittest

class TestURLs(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_add_new_post(self):
        """ Tests if the new post page saves a Post object to the
            database

            1\. Log the user in
            2\. Go to the new_post page
            3\. Fill out the fields and submit the form
            4\. Go to the blog home page and verify that the post 
               is on the page
        """
        pass
```

现在我们知道了我们的测试要做什么，让我们开始添加 Selenium 代码。在`setUp`和`tearDown`方法中，我们需要代码来启动 Selenium 控制的 Web 浏览器，然后在测试结束时关闭它。

```py
import unittest
from selenium import webdriver
class TestURLs(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Firefox()
    def tearDown(self):
        self.driver.close()
```

这段代码生成一个由 Selenium 控制的新的 Firefox 窗口。当然，为了使其工作，你需要在计算机上安装 Firefox。还有其他浏览器的支持，但它们都需要额外的程序才能正常工作。Firefox 在所有浏览器中具有最好的支持。

在编写测试代码之前，让我们按照以下方式探索 Selenium API：

```py
$ python
>>> from selenium import webdriver
>>> driver = webdriver.Firefox()
# load the Google homepage
>>> driver.get("http://www.google.com")
# find a element by its class
>>> search_field = driver.find_element_by_class_name("gsfi")
# find a element by its name
>>> search_field = driver.find_element_by_name("q")
# find an element by its id
>>> search_field = driver.find_element_by_id("lst-ib")
# find an element with JavaScript
>>> search_field = driver.execute_script(
 "return document.querySelector('#lst-ib')"
)
# search for flask
>>> search_field.send_keys("flask")
>>> search_button = driver.find_element_by_name("btnK")
>>> search_button.click()

```

这些是我们将要使用的 Selenium 的主要功能，但还有许多其他方法可以查找和与网页上的元素进行交互。有关可用功能的完整列表，请参阅 Selenium-Python 文档[`selenium-python.readthedocs.org`](http://selenium-python.readthedocs.org)。

在编写测试时，Selenium 中有两个需要牢记的要点，否则你将遇到几乎无法从错误消息中调试的非常奇怪的错误：

1.  Selenium 的设计就像有一个实际的人控制浏览器一样。这意味着如果页面上看不到一个元素，Selenium 就无法与其交互。例如，如果一个元素覆盖了你想点击的另一个元素，比如一个模态窗口在按钮前面，那么按钮就无法被点击。如果元素的 CSS 将其显示设置为`none`或可见性设置为`hidden`，结果将是一样的。

1.  屏幕上指向元素的所有变量都存储为指向浏览器中这些元素的指针，这意味着它们不存储在 Python 的内存中。如果页面在不使用`get`方法的情况下发生更改，比如点击链接并创建新的元素指针时，测试将崩溃。这是因为驱动程序将不断寻找先前页面上的元素，而在新页面上找不到它们。驱动程序的`get`方法清除所有这些引用。

在以前的测试中，我们使用测试客户端来模拟对应用程序对象的请求。然而，因为我们现在使用的是需要直接通过 Web 浏览器与应用程序进行交互的东西，我们需要一个实际运行的服务器。这个服务器需要在用户界面测试运行之前在一个单独的终端窗口中运行，以便它们有东西可以请求。为了做到这一点，我们需要一个单独的 Python 文件来使用我们的测试配置运行服务器，并设置一些模型供我们的 UI 测试使用。在项目目录的根目录下新建一个名为`run_test_server.py`的新文件，添加以下内容：

```py
from webapp import create_app
from webapp.models import db, User, Role

app = create_app('webapp.config.TestConfig')

db.app = app
db.create_all()

default = Role("default")
poster = Role("poster")
db.session.add(default)
db.session.add(poster)
db.session.commit()

test_user = User("test")
test_user.set_password("test")
test_user.roles.append(poster)
db.session.add(test_user)
db.session.commit()

app.run()
```

现在我们既有了测试服务器脚本，又了解了 Selenium 的 API，我们终于可以为我们的测试编写代码了：

```py
class TestURLs(unittest.TestCase):
    def setUp(self):
        …

    def tearDown(self):
        …

    def test_add_new_post(self):
        """ Tests if the new post page saves a Post object to the
            database

            1\. Log the user in
            2\. Go to the new_post page
            3\. Fill out the fields and submit the form
            4\. Go to the blog home page and verify that
               the post is on the page
        """
        # login
        self.driver.get("http://localhost:5000/login")

        username_field = self.driver.find_element_by_name(
            "username"
        )
        username_field.send_keys("test")

        password_field = self.driver.find_element_by_name(
            "password"
        )
        password_field.send_keys("test")

        login_button = self.driver.find_element_by_id(
            "login_button"
        )
        login_button.click()

        # fill out the form
        self.driver.get("http://localhost:5000/blog/new")

        title_field = self.driver.find_element_by_name("title")
        title_field.send_keys("Test Title")

        # find the editor in the iframe
        self.driver.switch_to.frame(
            self.driver.find_element_by_tag_name("iframe")
        )
        post_field = self.driver.find_element_by_class_name(
            "cke_editable"
        )
        post_field.send_keys("Test content")
        self.driver.switch_to.parent_frame()

        post_button = self.driver.find_element_by_class_name(
            "btn-primary"
        )
        post_button.click()

        # verify the post was created
        self.driver.get("http://localhost:5000/blog")
        self.assertIn("Test Title", self.driver.page_source)
        self.assertIn("Test content", self.driver.page_source)
```

这个测试中使用了我们之前介绍的大部分方法。然而，在这个测试中有一个名为`switch_to`的新方法。`switch_to`方法是驱动程序的上下文，允许选择`iframe`元素内的元素。通常情况下，父窗口无法使用 JavaScript 选择`iframe`内的任何元素，但因为我们直接与浏览器进行交互，我们可以访问`iframe`元素的内容。我们需要像这样切换上下文，因为在创建页面内的 WYSIWYG 编辑器中使用`iframe`。在`iframe`内选择元素完成后，我们需要使用`parent_frame`方法切换回父上下文。

现在你已经拥有了测试服务器代码和用户界面代码的测试工具。在本章的其余部分，我们将专注于工具和方法，以使您的测试更加有效，以确保应用程序的正确性。

# 测试覆盖率

现在我们已经编写了测试，我们必须知道我们的代码是否经过了充分的测试。测试覆盖率的概念，也称为代码覆盖率，是为了解决这个问题而发明的。在任何项目中，测试覆盖率表示在运行测试时执行了项目中多少百分比的代码，以及哪些代码行从未运行过。这给出了项目中哪些部分在我们的单元测试中没有被测试的想法。要将覆盖报告添加到我们的项目中，请使用以下命令使用 pip 安装覆盖库：

```py
$ pip install coverage

```

覆盖库可以作为一个命令行程序运行，它将在测试运行时运行您的测试套件并进行测量。

```py
$ coverage run --source webapp --branch -m unittest discover

```

`--source`标志告诉覆盖仅报告`webapp`目录中文件的覆盖率。如果不包括这个标志，那么应用程序中使用的所有库的百分比也将被包括在内。默认情况下，如果执行了`if`语句中的任何代码，就会说整个`if`语句已经执行。`--branch`标志告诉`coverage`禁用这一点，并测量所有内容。

在`coverage`运行我们的测试并进行测量后，我们可以以两种方式查看其发现的报告。第一种是在命令行上查看每个文件的覆盖百分比：

```py
$ coverage report
Name                               Stmts   Miss Branch BrMiss  Cover
--------------------------------------------------------------------
webapp/__init__                       51      0      6      0   100%
webapp/config                         37      0      0      0   100%
webapp/controllers/__init__            0      0      0      0   100%
webapp/controllers/admin              27      4      0      0    85%
webapp/controllers/blog               77     45      8      8    38%
webapp/controllers/main               78     42     20     16    41%
webapp/controllers/rest/__init__       0      0      0      0   100%
webapp/controllers/rest/auth          13      6      2      2    47%
webapp/controllers/rest/fields        17      8      0      0    53%
webapp/controllers/rest/parsers       19      0      0      0   100%
webapp/controllers/rest/post          85     71     44     43    12%
webapp/extensions                     56     14      4      4    70%
webapp/forms                          48     15     10      7    62%
webapp/models                         89     21      4      3    74%
webapp/tasks                          41     29      4      4    27%
--------------------------------------------------------------------
TOTAL                                638    255    102     87    54%

```

第二种是使用覆盖的 HTML 生成功能在浏览器中查看每个文件的详细信息。

```py
$ coverage html

```

上述命令创建了一个名为`htmlcov`的目录。当在浏览器中打开`index.html`文件时，可以单击每个文件名以显示测试期间运行和未运行的代码行的详细情况。

![测试覆盖率](img/B03929_12_01.jpg)

在上面的截图中，打开了`blog.py`文件，覆盖报告清楚地显示了帖子路由从未执行过。然而，这也会产生一些错误的负面影响。由于用户界面测试未测试覆盖程序运行的代码，因此它不计入我们的覆盖报告。为了解决这个问题，只需确保测试用例中有测试，测试每个单独的函数，这些函数在用户界面测试中应该被测试。

在大多数项目中，目标百分比约为 90%的代码覆盖率。很少有项目的 100%代码是可测试的，随着项目规模的增加，这种可能性会减少。

# 测试驱动开发

现在我们已经编写了测试，它们如何融入开发过程？目前，我们正在使用测试来确保在创建某些功能后代码的正确性。但是，如果我们改变顺序，使用测试来从一开始就创建正确的代码呢？这就是**测试驱动开发**（**TDD**）的主张。

TDD 遵循一个简单的循环来编写应用程序中新功能的代码：

![测试驱动开发](img/B03929_12_02.jpg)

此图像的来源是维基百科上的用户 Excirial

在使用 TDD 的项目中，你在实际构建任何控制你实际构建的代码之前，编写的第一件事是测试。这迫使项目中的程序员在编写任何代码之前规划项目的范围、设计和要求。在设计 API 时，它还迫使程序员从消费者的角度设计 API 的接口，而不是在编写所有后端代码之后设计接口。

在 TDD 中，测试旨在第一次运行时失败。TDD 中有一句话，如果你的测试第一次运行时没有失败，那么你实际上并没有测试任何东西。这意味着你很可能在编写测试之后测试被测试单元给出的结果，而不是应该给出的结果。

在第一次测试失败后，您不断编写代码，直到所有测试通过。对于每个新功能，这个过程都会重复。

一旦所有原始测试通过并且代码被清理干净，TDD 告诉你停止编写代码。通过仅在测试通过时编写代码，TDD 还强制执行“你不会需要它”（YAGNI）哲学，该哲学规定程序员只应实现他们实际需要的功能，而不是他们认为他们将需要的功能。在开发过程中，当程序员试图在没有人需要的情况下预先添加功能时，会浪费大量的精力。

例如，在我参与的一个 PHP 项目中，我发现了以下代码，用于在目录中查找图像：

```py
$images = glob(
    $img_directory . "{*.jpg, *.jpeg, *.gif, *.png, *.PNG, *.Png, *.PnG, *.pNG, *.pnG, *.pNg, *.PNg}",
    GLOB_BRACE
);
```

在 PHP 中，glob 是一个函数，它查找目录中的内容，以找到与模式匹配的文件。我质问了编写它的程序员。他对`.png`扩展名的不同版本的解释是，某个用户上传了一个带有`.PNG`扩展名的文件，而函数没有找到它，因为它只寻找扩展名的小写版本。他试图解决一个不存在的问题，以确保他不必再次触及这段代码，我们可能会觉得浪费了一点时间，但这段代码是整个代码库的缩影。如果这个项目遵循 TDD，就会为大写文件扩展名添加一个测试用例，添加代码以通过测试，然后问题就解决了。

TDD 还提倡“保持简单，愚蠢”（KISS）的理念，这个理念规定从一开始就应该把简单作为设计目标。TDD 提倡 KISS，因为它需要小的、可测试的代码单元，这些单元可以相互分离，不依赖于共享的全局状态。

此外，在遵循 TDD 的项目中，测试始终保持最新的文档。编程的一个公理是，对于任何足够大的程序，文档总是过时的。这是因为当程序员在更改代码时，文档是最后考虑的事情之一。然而，通过测试，项目中的每个功能都有清晰的示例（如果项目的代码覆盖率很高）。测试一直在更新，因此展示了程序的功能和 API 应该如何工作的良好示例。

现在你已经了解了 Flask 的功能以及如何为 Flask 编写测试，你在 Flask 中创建的下一个项目可以完全使用 TDD。

# 总结

现在你已经了解了测试以及它对你的应用程序能做什么，你可以创建保证不会有太多错误的应用程序。你将花更少的时间修复错误，更多的时间添加用户请求的功能。

在下一章中，我们将通过讨论在服务器上将应用程序部署到生产环境的方式来完成这本书。

作为对读者的最后挑战，在进入下一章之前，尝试将你的代码覆盖率提高到 95%以上。
