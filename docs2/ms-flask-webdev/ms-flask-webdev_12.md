# 测试 Flask 应用程序

在整本书中，每当我们对我们的应用程序代码进行修改时，我们都必须手动将受影响的网页加载到浏览器中，以测试代码是否正确工作。随着应用程序的增长，这个过程变得越来越繁琐，尤其是如果你更改了低级且到处使用的代码，比如 SQLAlchemy 模型代码。

为了自动化验证我们的代码按预期工作的过程，我们将使用 Python 的一个内置功能来编写测试，通常称为单元测试或集成测试，这些测试将与我们的应用程序代码进行比对。

在本章中，你将学习以下内容：

+   使用 Python 的 unittest 库编写简单测试

+   测试安全性，并验证登录和基于角色的访问

+   编写 REST API 的测试

+   测试用户界面

+   测量测试覆盖率

# 什么是单元测试？

测试一个程序非常简单。它只涉及开发代码来运行程序中的特定部分，并指定你期望的结果，然后将结果与程序实际运行的结果进行比较。如果结果相同，测试通过。如果结果不同，测试失败。通常，这些测试是在 CI 服务器上创建 Pull Request 时运行的，这样所有 PR 的审查者都可以立即检查请求的更改是否破坏了某些内容。

在程序测试中，主要有三种类型的测试。**单元测试**是验证单个代码片段（如函数）正确性的测试。其次是**集成测试**，它测试程序中各种单元协同工作的正确性。最后一种测试是**端到端测试**，它一次性测试整个系统的正确性，而不是单个部分。还存在许多其他类型的测试，其中一些包括负载测试、安全测试和恢复测试。

在本章中，我们将使用单元测试和端到端测试来验证我们的代码按计划工作。

这使我们来到了代码测试的一些基本规则：确保你的测试可以真正失败，编写只测试一个东西的简单测试函数，并使测试代码易于阅读和编写。

# 测试是如何工作的？

让我们从一个非常简单的 Python 函数开始测试：

```py
def square(x): 
    return x * x 
```

为了验证代码的正确性，我们传递一个值，并测试函数的结果是否符合预期。例如，我们可以给它一个输入为 5，并期望结果为 25。

为了说明这个概念，我们可以使用命令行中的`assert`语句手动测试这个函数。Python 中的`assert`语句简单地说，如果`assert`关键字后面的条件语句返回`False`，则将抛出异常，如下所示：

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

使用这些`assert`语句，我们验证了平方函数按预期工作。

# 单元测试应用程序

**单元测试**在 Python 中是通过将`assert`语句组合成类内部的独立函数来工作的。这个类内部的**测试函数集合**被称为**测试用例**。测试用例内部的每个函数应该只测试一件事情，这是单元测试背后的主要思想。在单元测试中只测试一件事情迫使你必须单独验证每一块代码，而不会忽略你代码的任何功能。如果你正确编写单元测试，你最终会得到很多很多测试。虽然这看起来可能过于冗长，但它会为你节省未来的麻烦。

在此配置中，我们将使用内存引擎数据库的 SQLite，这使我们能够保证测试不会干扰我们的实际数据库。此外，该配置禁用了 WTForms 的 CSRF 检查，以便我们可以在测试中提交表单而不需要 CSRF 令牌：

```py
class TestConfig(Config):

    DEBUG = True
    DEBUG_TB_ENABLED = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CACHE_TYPE = 'null'
    WTF_CSRF_ENABLED = False

    CELERY_BROKER_URL = "amqp://guest:guest@localhost:5672//"
    CELERY_BACKEND_URL = "amqp://guest:guest@localhost:5672//"

    MAIL_SERVER = 'localhost'
    MAIL_PORT = 25
    MAIL_USERNAME = 'username'
    MAIL_PASSWORD = 'password'

```

# 测试路由函数

让我们构建第一个测试用例。在这个测试用例中，我们将测试当访问其 URL 时，路由函数是否成功返回响应。在项目目录根目录下创建一个名为`tests`的新目录，在该目录中创建一个名为`test_urls.py`的新文件，该文件将包含所有路由的单元测试。每个测试用例应该有自己的文件，每个测试用例应该专注于你正在测试的代码的一个区域。

在`test_urls.py`中，让我们开始创建内置 Python `unittest`库所需的内容。代码将使用 Python 中的`unittest`库来运行我们在测试用例中创建的所有测试：

```py
import unittest 

class TestURLs(unittest.TestCase): 
    pass 

if __name__ == '__main__': 
    unittest.main() 
```

让我们看看运行此代码会发生什么。我们将使用`unittest`库自动查找我们的测试用例的能力来运行测试。`unittest`库寻找的模式是`test*.py`：

```py
$ python -m unittest discover

---------------------------------------------------------------------
Ran 0 tests in 0.000s

OK
```

因为测试用例中没有测试，所以测试用例成功通过。

测试脚本是从脚本的父目录而不是测试文件夹本身运行的。这是为了允许在测试脚本中导入应用程序代码。

为了测试 URL，我们需要一种方法来查询应用程序的路由，而无需实际运行服务器，以便我们的请求得到响应。Flask 提供了一个在测试中访问路由的方法，称为**测试客户端**。测试客户端提供了在路由上创建 HTTP 请求的方法，而无需实际使用`app.run()`运行应用程序。

我们将需要为这个测试用例中的每个测试使用测试客户端对象，但在`unittest`中添加创建测试客户端的代码在`setUp`方法中并不合理。`setUp`方法在每次单元测试之前运行，并且可以将其变量附加到自身，以便测试方法可以访问它们。在我们的`setUp`方法中，我们需要使用`TestConfig`对象创建应用程序对象并创建测试客户端。

此外，还有三个我们需要解决的错误。前两个在 Flask Admin 和 Flask Restful 扩展中，它们在应用到的应用程序对象被销毁时没有删除内部存储的 Blueprint 对象。第三，Flask SQLAlchemy 的初始化器在`webapp`目录外没有正确添加应用程序对象。

```py
class TestURLs(unittest.TestCase):

    def setUp(self):
        admin._views = []
        rest_api.resources = []

        app = create_app('config.TestConfig')
        self.client = app.test_client()
        db.app = app
        db.create_all()
```

这里列出的所有错误在编写时都存在，但当你阅读这一章时可能已经不存在了。

除了`setUp`方法外，还有一个`tearDown`方法，它在每次单元测试结束时都会运行。`tearDown`方法的目标是销毁在`setUp`方法中创建的任何无法自动删除或关闭的对象。在我们的例子中，我们将使用`tearDown`方法来关闭和删除我们的数据库会话，如下所示：

```py
class TestURLs(unittest.TestCase): 
    def setUp(self): 
        ... 

    def tearDown(self): 
        db.session.remove()
```

现在我们可以创建我们的第一个单元测试。第一个测试将检查访问我们应用程序的根是否返回一个`302`重定向代码到博客主页，如下所示：

```py
class TestURLs(unittest.TestCase): 
    def setUp(self): 
        ... 

    def tearDown(self): 
        ... 

    def test_root_redirect(self): 
        """ Tests if the root URL gives a 302 """ 

        result = self.client.get('/') 
        assert result.status_code == 302 
        assert "/blog/" in result.headers['Location'] 
```

每个单元测试都必须以单词`test`开头，以便告诉`unittest`库该函数是一个单元测试，而不是测试用例类中的某个实用函数。

现在，如果我们再次运行测试，我们可以看到它的进度以及它是如何通过检查的：

```py
    $ python -m unittest discover
    .
    ---------------------------------------------------------------------
    Ran 1 tests in 0.128s

    OK

```

编写测试的最佳方式是在事先弄清楚你想要寻找什么，编写`assert`语句，然后编写执行这些断言所需的代码。这迫使你在实际编写测试之前先弄清楚你真正要测试什么。此外，为每个单元测试编写 Python 文档字符串也是一个惯例，因为当测试失败时，它将与测试名称一起打印出来。在你编写了 50 个或更多的测试之后，这将非常有帮助，可以确切地知道测试的目的。

我们可以不用 Python 内置的`assert`关键字，而是使用`unittest`库提供的一些方法。当这些函数中的`assert`语句失败时，这些方法提供了专门的错误信息和调试信息。

以下是由`unittest`库提供的所有特殊`assert`语句及其功能的列表：

+   `assertEqual(x, y)`: 断言`x == y`

+   `assertNotEqual(x, y)`: 断言`x != y`

+   `assertTrue(x)`: 断言`x`是`True`

+   `assertFalse(x)`: 断言`x`是`False`

+   `assertIs(x, y)`: 断言`x`是`y`

+   `assertIsNot(x, y)`: 断言`x`不是`y`

+   `assertIsNone(x)`: 断言`x`是`None`

+   `assertIsNotNone(x)`: 断言`x`不是`None`

+   `assertIn(x, y)`: 断言`y`包含`x`

+   `assertNotIn(x, y)`: 断言`x`不在`y`中

+   `assertIsInstance(x, y)`: 断言`isinstance(x, y)`

+   `assertNotIsInstance(x, y)`: 断言不是`isinstance(x, y)`

如果我们要测试普通页面的返回值，单元测试将看起来像这样：

```py
class TestURLs(unittest.TestCase): 
    def setUp(self): 
        ... 

    def tearDown(self): 
        ... 

    def test_root_redirect(self): 
        ... 

    def test_blog_home(self): 
        """ Tests if the blog home page returns successfully """ 
        result = self.client.get('/blog/')
        self.assertEqual(result.status_code, 200)
```

记住，前面的代码只测试了 URL 是否成功返回。返回数据的内容不是这些测试的一部分。

# 测试安全性

测试安全性显然非常重要——如果你将你的应用程序暴露在网络上，你可以确信你的安全性将会受到严重的测试，而且不是出于正确的理由。如果你的安全措施没有得到正确保护，所有的受保护端点都将被测试和利用。首先，我们应该测试我们的登录和注销过程。

如果我们想要测试提交表单，例如登录表单，我们可以使用测试客户端的 post 方法。让我们创建一个`test_login`方法来查看登录表单是否工作正常：

```py
class TestURLs(unittest.TestCase):
....

    def _insert_user(self, username, password, role_name):
        test_role = Role(role_name)
        db.session.add(test_role)
        db.session.commit()

        test_user = User(username)
        test_user.set_password(password)
        db.session.add(test_user)
        db.session.commit()

    def test_login(self):
        """ Tests if the login form works correctly """

        result = self.client.post('/auth/login', data=dict(
            username='test',
            password="test"
        ), follow_redirects=True)

        self.assertEqual(result.status_code, 200)
        self.assertIn('You have been logged in', result.data)
...
```

返回数据中的字符串额外检查是因为返回代码不受输入数据有效性的影响。`post`方法将适用于测试我们在整本书中创建的任何表单对象。

现在，让我们创建一个失败的登录尝试：

```py
def test_failed_login(self):
    self._insert_user('test', 'test', 'default')
    result = self.client.post('/auth/login', data=dict(
        username='test',
        password="badpassword"
    ), follow_redirects=True)

    self.assertEqual(result.status_code, 200)
    self.assertIn('Invalid username or password', result.data)
    result = self.client.get('/blog/new')
    self.assertEqual(result.status_code, 302)
```

在前面的代码片段中，我们确保使用失败的凭证进行的登录尝试不会给用户带来成功的登录，并且在同一测试中，我们还确保失败的登录不会给用户足够的权限来添加新的博客文章。这看起来可能很微不足道，而且很容易实现，但如前所述，你应该使每个测试尽可能简单，并且每个测试只测试一件事情，但目标是覆盖所有功能和可能性。

另一个重要测试的例子是覆盖已登录用户的未授权访问：

```py
def test_unauthorized_access_to_admin(self):
    self._insert_user('test', 'test', 'default')
    result = self.client.post('/auth/login', data=dict(
        username='test',
        password="test"
    ), follow_redirects=True)
    result = self.client.get('/admin/customview/')
    self.assertEqual(result.status_code, 403)
```

在这里，我们确保低权限用户无法访问我们应用程序中的高权限区域：管理员界面。

# 测试 REST API

仍然在安全性的背景下，我们现在将学习如何测试我们的 REST API。记住，我们已经实现了 JWT 安全性，因此对于每个请求，我们需要使用之前获取的访问令牌。

JWT 身份验证测试应该看起来像这样：

```py
def test_api_jwt_login(self):
    self._insert_user('test', 'test', 'default')
    headers = {'content-type': 'application/json'}
    result = self.client.post('/auth/api', headers=headers, data='{"username":"test","password":"test"}')
    self.assertEqual(result.status_code, 200)

def test_api_jwt_failed_login(self):
    self._insert_user('test', 'test', 'default')
    headers = {'content-type': 'application/json'}
    result = self.client.post('/auth/api', headers=headers, data='{"username":"test","password":"test123"}')
    self.assertEqual(result.status_code, 401)
```

在这里需要注意的一些重要细节包括我们如何设置 HTTP 头为 JSON 格式，以及我们如何在 HTTP POST 方法中传递 JSON 有效载荷——这将在我们所有的 REST API 测试中发生。

接下来，让我们看看如何开发一个针对新帖子 REST API 的测试。`/api/post`是博客文章的端点，POST HTTP 方法是向博客应用程序添加新帖子的方法。如果这还不清楚，请回顾第八章，*构建 RESTful API*。

```py
def test_api_new_post(self):
    self._insert_user('test', 'test', 'default')
    headers = {'content-type': 'application/json'}
    result = self.client.post('/auth/api', headers=headers, data='{"username":"test","password":"test"}')
    access_token = json.loads(result.data)['access_token']
 headers['Authorization'] = "Bearer %s" % access_token
    result = self.client.post('api/post', headers=headers, data='{"title":"Text Title","text":"Changed"}')
    self.assertEqual(result.status_code, 201)
```

这同样是一个简单的测试来开发——注意我们如何使用`/auth/api`端点从我们的认证 JWT API 请求访问令牌，以及我们如何使用它来调用`/api/post`。预期的访问令牌被用来构造 HTTP 授权头，形式为`Authorization: Bearer <ACCESS_TOKEN>`。这可能在每个 API 测试中重复可能有点繁琐，所以确保编写一个辅助函数来保持你的代码“DRY”——即“不要重复自己”。

现在您已经了解了单元测试的机制，您可以使用单元测试来测试您应用程序的所有部分。这可以包括测试应用程序中的所有路由；测试我们制作的任何实用函数，例如`sidebar_data`；以及测试所有可能的角色和受保护页面的组合。

如果您的应用程序代码有一个功能，无论多小，您都应该为它编写一个测试。为什么？因为任何可能出错的事情都会出错。如果您的应用程序代码的有效性完全依赖于手动测试，那么随着您的应用程序的增长，某些东西可能会被忽略。当某些东西被忽略时，损坏的代码就会被部署到生产服务器上，这会令您的用户感到烦恼。

# 用户界面测试

为了测试我们应用程序代码的高级功能并创建系统测试，我们将编写与浏览器一起工作的测试，并验证 UI 代码是否正常工作。使用一个名为 Selenium 的工具，我们将创建 Python 代码，该代码可以连接到浏览器并完全通过代码控制它。这是通过在屏幕上查找元素，然后通过 Selenium 对这些元素执行操作来实现的。点击它或输入按键。此外，Selenium 还允许您通过提供对元素内容的访问来对页面内容进行检查，例如它们的属性和内部文本。对于更高级的检查，Selenium 甚至有一个可以运行页面上的任意 JavaScript 的接口。如果 JavaScript 返回一个值，它将被自动转换为 Python 类型。

在我们接触代码之前，需要安装 Selenium。确保您的虚拟环境已激活，并且 Selenium 已包含在`requirements.txt`文件中：

```py
...    
selenium
...
```

要开始编写代码，我们的 UI 测试需要在`tests`目录中有一个自己的文件，命名为`test_ui.py`。因为系统测试不测试特定的事情，所以编写用户界面测试的最佳方式是将测试视为通过一个典型用户的流程。在您编写测试本身之前，写下我们的假用户将要模拟的具体步骤：

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

现在我们确切地知道我们的测试将要做什么，让我们开始添加 Selenium 代码。在`setUp`和`tearDown`方法中，我们需要代码来启动一个 Selenium 将控制的网络浏览器，然后在测试结束时关闭它：

```py
import unittest 
from selenium import webdriver

class TestURLs(unittest.TestCase): 
    def setUp(self): 
        self.driver = webdriver.Chrome() 

    def tearDown(self): 
        self.driver.close()
```

此代码通过 Selenium 控制创建一个新的 Firefox 窗口。当然，为了使其工作，您需要在您的计算机上安装 Firefox。Selenium 确实支持其他浏览器，但使用其他浏览器需要额外的程序才能正确工作。因此，Firefox 在所有浏览器中具有最好的支持。

在我们编写测试代码之前，让我们按照以下方式探索 Selenium API：

```py
    $ python
    >>> from selenium import webdriver
    >>> driver = webdriver.Chrome()
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

这些是我们将使用的主要 Selenium 函数，但还有许多其他方法可以找到和与网页上的元素交互。

要查看所有可用功能的完整列表，请参阅 Selenium-Python 文档，网址为 [`selenium-python.readthedocs.org`](http://selenium-python.readthedocs.org)。

在编写测试时，有两个需要注意的 Selenium 陷阱，否则你可能会遇到非常奇怪的错误，这些错误几乎不可能从错误消息中调试：

+   Selenium 被设计成好像有一个真实的人正在控制浏览器。这意味着，如果页面上的某个元素不可见，Selenium 就无法与之交互。例如，如果某个元素覆盖了你希望点击的另一个元素——比如说，一个模态窗口在按钮前面——那么按钮就无法被按下。如果元素的 CSS 将其 `display` 设置为 `none`，或者将其 `visibility` 设置为 `hidden`，结果将相同。

+   所有指向屏幕上元素的变量都存储为浏览器中这些元素的指针，这意味着它们不是存储在 Python 的内存中。如果页面在没有使用 `get` 方法的情况下发生变化，例如当点击链接并创建新的元素指针时，测试将崩溃。这是因为驱动程序将不断寻找前一个页面上的元素，但在新页面上找不到它们。驱动程序的 `get` 方法清除所有这些引用。

在前面的测试中，我们使用了测试客户端来模拟对应用程序对象的请求。然而，因为我们现在使用的是需要通过浏览器直接与应用程序交互的东西，我们需要一个实际的服务器正在运行。这个服务器需要在运行用户界面测试之前在单独的终端窗口中运行，以便后者有东西可以请求。为此，我们需要一个单独的 Python 文件来运行服务器，并使用我们的测试配置，还需要为我们的 UI 测试设置一些模型。在项目目录的根目录中，创建一个名为 `run_test_server.py` 的新文件，并添加以下内容：

```py
from webapp import create_app 
from webapp.models import db, User, Role 

app = create_app('config.TestConfig') 

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

现在我们已经有了测试服务器脚本和对 Selenium API 的了解，我们最终可以编写我们的测试代码了：

```py
import time
import unittest
from selenium import webdriver

class TestURLs(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()

    def tearDown(self):
        self.driver.close()

    def test_add_new_post(self):
        """ Tests if the new post page saves a Post object to the
            database

            1\. Log the user in
            2\. Go to the new_post page
            3\. Fill out the fields and submit the form
            4\. Go to the blog home page and verify that the post is
               on the page
        """
        # login
        self.driver.get("http://localhost:5000/auth/login")

        username_field = self.driver.find_element_by_name("username")
        username_field.send_keys("test")

        password_field = self.driver.find_element_by_name("password")
        password_field.send_keys("test")

        login_button = self.driver.find_element_by_id("login_button")
        login_button.click()

        # fill out the form
        self.driver.get("http://localhost:5000/blog/new")

        title_field = self.driver.find_element_by_name("title")
        title_field.send_keys("Test Title")

        #Locate the CKEditor iframe
        time.sleep(3)
        basic_page_body_xpath = "//div[contains(@id, 'cke_1_contents')]/iframe"
        ckeditor_frame = self.driver.find_element_by_xpath(basic_page_body_xpath)

        #Switch to iframe
        self.driver.switch_to.frame(ckeditor_frame)
        editor_body = self.driver.find_element_by_xpath("//body")
        editor_body.send_keys("Test content")
        self.driver.switch_to.default_content()

        post_button = self.driver.find_element_by_class_name("btn-primary")
        post_button.click()

        # verify the post was created
        self.driver.get("http://localhost:5000/blog")
        self.assertIn("Test Title", self.driver.page_source)
        self.assertIn("Test content", self.driver.page_source)

if __name__ == "__main__":
    unittest.main()
```

大部分这个测试使用了我们之前介绍的方法。然而，在这个测试中有一个新的方法，名为 `switch_to`。`switch_to` 方法是驱动程序上下文的一部分，允许选择 `iframe` 元素内的元素。通常情况下，使用 JavaScript，父窗口无法选择 `iframe` 元素内的任何元素，但因为我们直接与浏览器本身进行交互，我们可以访问 `iframe` 元素的内部内容。我们需要切换这样的联系人，因为帖子创建页面内的 WYSIWYG 编辑器使用 `iframe` 来创建自身。在完成选择 `iframe` 内的元素后，我们需要使用 `parent_frame` 方法切换回父上下文。

你现在拥有了完全测试你的服务器代码和用户界面代码所需的工具。在接下来的章节中，我们将重点关注工具和方法，以便使你的测试在确保应用程序正确性方面更加有效。

# 测试覆盖率

现在我们已经编写了测试，我们必须知道我们的代码是否得到了充分的测试。**测试覆盖率**（也称为**代码覆盖率**）的概念是为了解决这个问题而发明的。在任何项目中，测试覆盖率表示在运行测试时项目中的代码执行百分比，以及哪些行从未运行。这给出了一个关于哪些项目部分没有被我们的单元测试测试的想法。为了将覆盖率报告添加到我们的项目中，使用 `pip` 安装覆盖率库，并确保它包含在 `requirements.txt` 中：

```py
 (venv)$ pip install coverage
```

覆盖率库可以作为命令行程序运行，在测试运行时执行你的测试套件并获取其测量结果：

```py
    $ coverage run --source webapp --branch -m unittest discover
```

`--source` 标志告诉 `coverage` 只报告 `webapp` 目录中文件的测试覆盖率。如果不包括它，所有在应用程序中使用的库的百分比也会包括在内。默认情况下，如果 `if` 语句中的任何代码被执行，整个 `if` 语句就被说成是执行了。`--branch` 标志告诉 `coverage` 禁用此功能，并测量一切。

在 `coverage` 运行我们的测试并获取其测量结果后，我们可以通过两种方式查看其发现报告。第一种是在命令行上查看每个文件的覆盖率百分比：

```py
$ coverage report
...
# You will get a full detailed report of your test coverage, breakdown by python file name coverage, and with the line numbers missed by your test
...

TOTAL 729 312 118 10 56%
```

查看报告的第二种方式是使用 `coverage` 的 HTML 生成能力，在浏览器中查看每个文件的详细分解，使用以下命令：

```py
    $ coverage html
```

前一个命令创建了一个名为 `htmlcov` 的目录。当在浏览器中打开 `index.html` 文件时，可以点击每个文件名来查看测试期间哪些行被执行，哪些行没有执行：

![图片](img/6e4f6711-c002-4e0a-a7b0-5fbff0adb5a4.png)

在前面的屏幕截图中，打开了 `blog/controllers.py` 文件，覆盖率报告清楚地显示帖子路由从未被执行。然而，这也带来了一些错误的否定。由于用户界面测试没有测试由覆盖率程序运行的代码，它不计入我们的覆盖率报告。为了解决这个问题，确保你在测试用例中对每个将被用户界面测试测试的单独函数都有测试。

在大多数项目中，目标覆盖率百分比约为 90%。很少有项目会有 100% 的代码可测试，并且随着项目规模的增加，这种可能性会降低。

# 测试驱动开发

现在我们已经编写了测试，如何将它们集成到开发过程中呢？目前，我们使用测试来确保在创建功能后代码的正确性。但是，如果我们颠倒顺序，从一开始就使用测试来创建正确的代码，会怎么样呢？这正是**测试驱动开发**（**TDD**）所倡导的。

TDD 遵循一个简单的循环来编写你应用程序中新的功能代码：

![图片](img/55a76ee5-c1da-4c76-adc3-4a1af6308ab1.png)

在使用 TDD 的项目中，你首先编写的是测试，而不是控制你实际构建的任何代码。这迫使项目中的程序员在编写任何代码之前规划项目的范围、设计和需求。在设计 API 时，这也迫使程序员从消费者的角度设计 API 的接口（或合同），而不是在所有后端代码编写完毕后再设计接口。

在 TDD 中，测试被设计成第一次运行时就会失败。在 TDD 中有一句话，如果你的测试第一次运行时没有失败，那么你实际上并没有真正进行测试。这意味着你很可能是测试了测试单元的功能，而不是在编写测试之后如何应该工作。

在你的测试第一次失败后，你然后会持续编写代码，直到所有测试通过。这个过程会为每个新功能重复进行。

一旦所有原始测试通过并且代码重构完成，TDD 会告诉你停止编写代码。通过仅在测试通过之前编写代码，TDD 还强制执行了**YAGNI（你不会需要它）**哲学，该哲学指出程序员应该只实现他们实际需要的功能，而不是他们认为将来需要的功能。当程序员试图预先添加不需要的功能时，在开发过程中会浪费大量的精力。

TDD 还提倡**KISS（保持简单，傻瓜）**的理念，这规定从开始设计时，简单性应该是一个设计目标。TDD 提倡 KISS，因为它要求编写小的、可测试的代码单元，这些单元可以彼此分离，并且不依赖于共享的全局状态。

此外，在遵循 TDD（测试驱动开发）的项目中，测试过程中始终存在最新的文档。编程的一个公理是，对于任何足够大的程序，文档总是会过时。这是因为当程序员更改代码时，文档往往是他们最后考虑的事情。然而，通过测试，项目中每个功能模块都有明确的示例（如果项目有较高的代码覆盖率）。测试会不断更新，因此，展示了程序功能和 API 应该如何工作的良好示例。

现在你已经了解了 Flask 的功能以及如何为 Flask 编写测试，你接下来在 Flask 中创建的项目可以完全使用 TDD（测试驱动开发）方法来完成。

# 摘要

现在你已经了解了测试以及它能为你的应用程序带来什么，你可以创建出几乎零缺陷的应用程序。你将花费更少的时间来修复缺陷，更多的时间来添加用户请求的功能。

作为对读者的最终挑战，在进入下一章之前，尝试让你的代码覆盖率超过 95%。

在下一章中，我们将通过介绍你将应用程序部署到服务器上的生产环境的方法来结束本书。
