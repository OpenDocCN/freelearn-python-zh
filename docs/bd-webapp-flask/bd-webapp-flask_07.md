# 第七章：如果没有经过测试，那就不是游戏，兄弟！

您编写的软件是否具有质量？您如何证明？

通常根据特定的需求编写软件，无论是错误报告、功能和增强票据，还是其他。为了具有质量，软件必须完全和准确地满足这些需求；也就是说，它应该做到符合预期。

就像您会按下按钮来了解它的功能一样（假设您没有手册），您必须测试您的代码以了解它的功能或证明它应该做什么。这就是您确保**软件质量**的方式。

在软件开发过程中，通常会有许多共享某些代码库或库的功能。例如，您可以更改一段代码以修复错误，并在代码的另一个点上创建另一个错误。软件测试也有助于解决这个问题，因为它们确保您的代码执行了应该执行的操作；如果您更改了一段错误的代码并且破坏了另一段代码，您也将破坏一个测试。在这种情况下，如果您使用**持续集成**，则破损的代码将永远不会到达生产环境。

### 提示

不知道什么是持续集成？请参考[`www.martinfowler.com/articles/continuousIntegration.html`](http://www.martinfowler.com/articles/continuousIntegration.html)和[`jenkins-ci.org/`](https://jenkins-ci.org/)。

测试是如此重要，以至于有一个称为**测试驱动开发**（**TDD**）的软件开发过程，它规定测试应该在实际代码之前编写，并且只有当测试本身得到满足时，实际代码才是*准备就绪*。TDD 在高级开发人员及以上中非常常见。就为了好玩，我们将在本章中从头到尾使用 TDD。

# 有哪些测试类型？

我们想要测试，我们现在就想要；但是我们想要什么样的测试呢？

测试有两种主要分类，根据你对内部代码的访问程度：**黑盒**和**白盒**测试。

黑盒测试是指测试人员对其正在测试的实际代码没有知识和/或访问权限。在这些情况下，测试包括检查代码执行前后的系统状态是否符合预期，或者给定的输出是否对应于给定的输入。

白盒测试有所不同，因为您将可以访问您正在测试的实际代码内部，以及代码执行前后的系统预期状态和给定输入的预期输出。这种测试具有更强烈的主观目标，通常与性能和软件质量有关。

在本章中，我们将介绍如何实施黑盒测试，因为它们更容易让其他人接触并且更容易实施。另一方面，我们将概述执行白盒测试的工具。

代码库可能经过多种方式测试。我们将专注于两种类型的自动化测试（我们不会涵盖手动测试技术），每种测试都有不同的目标：**单元测试**和**行为测试**。这些测试各自有不同的目的，并相互补充。让我们看看这些测试是什么，何时使用它们以及如何在 Flask 中运行它们。

## 单元测试

单元测试是一种技术，您可以针对具有有意义功能的最小代码片段（称为**单元**）对输入和预期输出进行测试。通常，您会对代码库中不依赖于您编写的其他函数和方法的函数和方法运行单元测试。

在某种意义上，测试实际上是将单元测试堆叠在一起的艺术（首先测试一个函数，然后相互交互的函数，然后与其他系统交互的函数），以便整个系统最终得到充分测试。

对于 Python 的单元测试，我们可以使用内置模块`doctest`或`unittest`。`doctest`模块用于作为测试用例运行来自对象文档的嵌入式交互式代码示例。Doctests 是 Unittest 的一个很好的补充，Unittest 是一个更健壮的模块，专注于帮助您编写单元测试（正如其名称所暗示的那样），最好不要单独使用。让我们看一个例子：

```py
# coding:utf-8

"""Doctest example"""

import doctest
import unittest

def sum_fnc(a, b):
    """
    Returns a + b

    >>> sum_fnc(10, 20)
    30
    >>> sum_fnc(-10, -20)
    -30
    >>> sum_fnc(10, -20)
    -10
    """
    return a + b

class TestSumFnc(unittest.TestCase):
    def test_sum_with_positive_numbers(self):
        result = sum_fnc(10, 20)
        self.assertEqual(result, 30)

    def test_sum_with_negative_numbers(self):
        result = sum_fnc(-10, -20)
        self.assertEqual(result, -30)

    def test_sum_with_mixed_signal_numbers(self):
        result = sum_fnc(10, -20)
        self.assertEqual(result, -10)

if __name__ == '__main__':
    doctest.testmod(verbose=1)
    unittest.main()
```

在前面的例子中，我们定义了一个简单的`sum_fnc`函数，它接收两个参数并返回它们的和。`sum_fnc`函数有一个解释自身的文档字符串。在这个文档字符串中，我们有一个函数调用和输出的交互式代码示例。这个代码示例是由`doctest.testmod()`调用的，它检查给定的输出是否对于调用的函数是正确的。

接下来，我们有一个名为`TestSumFnc`的`TestCase`，它定义了三个测试方法（`test_<test_name>`），并且几乎完全与我们的文档字符串测试相同。这种方法的不同之处在于，我们能够在没有测试结果的情况下发现问题，*如果*有问题。如果我们希望对我们的文档字符串和测试用例做完全相同的事情，我们将在测试方法中使用`assert` Python 关键字来将结果与预期结果进行比较。相反，我们使用了`assertEqual`方法，它不仅告诉我们如果结果有问题，还告诉我们问题是结果和预期值都不相等。

如果我们希望检查我们的结果是否大于某个值，我们将使用`assertGreater`或`assertGreaterEqual`方法，这样断言错误也会告诉我们我们有什么样的错误。

### 提示

良好的测试彼此独立，以便一个失败的测试永远不会阻止另一个测试的运行。从测试中导入测试依赖项并清理数据库是常见的做法。

在编写脚本或桌面应用程序时，前面的情况很常见。Web 应用程序对测试有不同的需求。Web 应用程序代码通常是响应通过浏览器请求的用户交互而运行，并返回响应作为输出。要在这种环境中进行测试，我们必须模拟请求并正确测试响应内容，这通常不像我们的`sum_fnc`的输出那样直截了当。响应可以是任何类型的文档，它可能具有不同的大小和内容，甚至您还必须担心响应的 HTTP 代码，这包含了很多上下文含义。

为了帮助您测试视图并模拟用户与您的 Web 应用程序的交互，Flask 为您提供了一个测试客户端工具，通过它您可以向您的应用程序发送任何有效的 HTTP 方法的请求。例如，您可以通过`PUT`请求查询服务，或者通过`GET`请求查看常规视图。这是一个例子：

```py
# coding:utf-8

from flask import Flask, url_for, request
import unittest

def setup_database(app):
    # setup database ...
    pass

def setup(app):
    from flask import request, render_template

    # this is not a good production setup
    # you should register blueprints here
    @app.route("/")
    def index_view():
        return render_template('index.html', name=request.args.get('name'))

def app_factory(name=__name__, debug=True):
    app = Flask(name)
    app.debug = debug
    setup_database(app)
    setup(app)
    return app

class TestWebApp(unittest.TestCase):
    def setUp(self):
        # setUp is called before each test method
        # we create a clean app for each test
        self.app = app_factory()
        # we create a clean client for each test
        self.client = self.app.test_client()

    def tearDown(self):
        # release resources here
        # usually, you clean or destroy the test database
        pass

    def test_index_no_arguments(self):
        with self.app.test_request_context():
            path = url_for('index_view')
            resp = self.client.get(path)
            # check response content
            self.assertIn('Hello World', resp.data)

    def test_index_with_name(self):
        with self.app.test_request_context():
            name = 'Amazing You'
            path = url_for('index_view', name=name)
            resp = self.client.get(path)
            # check response content
            self.assertIn(name, resp.data)

if __name__ == '__main__':
    unittest.main()
```

前面的例子是一个完整的例子。我们使用`app_factory`模式来创建我们的应用程序，然后我们在`setUp`中创建一个应用程序和客户端，这在每个测试方法运行之前运行，我们创建了两个测试，一个是当请求接收到一个名称参数时，另一个是当请求没有接收到名称参数时。由于我们没有创建任何持久资源，我们的`tearDown`方法是空的。如果我们有任何类型的数据库连接和固定装置，我们将不得不在`tearDown`中重置数据库状态，甚至删除数据库。

此外，要注意`test_request_context`，它用于在我们的测试中创建一个请求上下文。我们创建这个上下文，以便`url_for`能够返回我们的视图路径，如果没有设置`SERVER_NAME`配置，它需要一个请求上下文。

### 提示

如果您的网站使用子域，设置`SERVER_NAME`配置。

## 行为测试

在单元测试中，我们测试函数的输出与预期结果。如果结果不是我们等待的结果，将引发断言异常以通知问题。这是一个简单的黑盒测试。现在，一些奇怪的问题：您是否注意到您的测试是以与错误报告或功能请求不同的方式编写的？您是否注意到您的测试不能被非技术人员阅读，因为它实际上是代码？

我想向您介绍 lettuce（[`lettuce.it/`](http://lettuce.it/)），这是一个能够将**Gherkin**语言测试转换为实际测试的工具。

### 提示

有关 Gherkin 语言的概述，请访问[`github.com/cucumber/cucumber/wiki/Gherkin`](https://github.com/cucumber/cucumber/wiki/Gherkin)。

Lettuce 可以帮助您将实际用户编写的功能转换为测试方法调用。这样，一个功能请求就像：

功能：计算总和

为了计算总和

作为学生

实现`sum_fnc`

+   **场景**：正数之和

+   **假设**我有数字 10 和 20

+   **当**我把它们加起来

+   **然后**我看到结果 30

+   **场景**：负数之和

+   **假设**我有数字-10 和-20

+   **当**我把它们加起来

+   **然后**我看到结果-30

+   **场景**：混合信号之和

+   **假设**我有数字 10 和-20

+   **当**我把它们加起来

+   **然后**我看到结果-10

该功能可以转换为将测试软件的实际代码。确保 lettuce 已正确安装：

```py
pip install lettuce python-Levenshtein

```

创建一个`features`目录，并在其中放置一个`steps.py`（或者您喜欢的任何其他 Python 文件名），其中包含以下代码：

```py
# coding:utf-8
from lettuce import *
from lib import sum_fnc

@step('Given I have the numbers (\-?\d+) and (\-?\d+)')
def have_the_numbers(step, *numbers):
    numbers = map(lambda n: int(n), numbers)
    world.numbers = numbers

@step('When I sum them')
def compute_sum(step):
    world.result = sum_fnc(*world.numbers)

@step('Then I see the result (\-?\d+)')
def check_number(step, expected):
    expected = int(expected)
    assert world.result == expected, "Got %d; expected %d" % (world.result, expected)
```

我们刚刚做了什么？我们定义了三个测试函数，have_the_numbers，compute_sum 和 check_number，其中每个函数都接收一个`step`实例作为第一个参数，以及用于实际测试的其他参数。用于装饰我们的函数的 step 装饰器用于将从我们的 Gherkin 文本解析的字符串模式映射到函数本身。我们的装饰器的另一个职责是将从步骤参数映射到函数的参数的参数解析为参数。

例如，`have_the_numbers`的步骤具有正则表达式模式（`\-?\d+`）和（`\-?\d+`），它将两个数字映射到我们函数的`numbers`参数。这些值是从我们的 Gherkin 输入文本中获取的。对于给定的场景，这些数字分别是[10, 20]，[-10, -20]和[10, -20]。最后，`world`是一个全局变量，您可以在步骤之间共享值。

使用功能描述行为对开发过程非常有益，因为它使业务人员更接近正在创建的内容，尽管它相当冗长。另外，由于它冗长，不建议在测试孤立的函数时使用，就像我们在前面的例子中所做的那样。行为应该由业务人员编写，也应该测试编写人员可以直观证明的行为。例如，“如果我点击一个按钮，我会得到某物的最低价格”或“假设我访问某个页面，我会看到一些消息或一些链接”。

“点击这里，然后那里发生了什么”。检查渲染的请求响应有点棘手，如果您问我的话。为什么？在我们的第二个例子中，我们验证了给定的字符串值是否在我们的`resp.data`中，这是可以的，因为我们的响应返回`complete`。我们不使用 JavaScript 在页面加载后渲染任何内容或显示消息。如果是这种情况，我们的验证可能会返回错误的结果，因为 JavaScript 代码不会被执行。

为了正确呈现和验证`view`响应，我们可以使用无头浏览器，如**Selenium**或**PhantomJS**（参见[`pythonhosted.org/Flask-Testing/#testing-with-liveserver`](https://pythonhosted.org/Flask-Testing/#testing-with-liveserver)）。**Flask-testing**扩展也会有所帮助。

## Flask-testing

与大多数 Flask 扩展一样，Flask-testing 并没有做太多事情，但它所做的事情都做得很好！我们将讨论 Flask-testing 提供的一些非常有用的功能：LiveServer 设置，额外的断言和 JSON 响应处理。在继续之前，请确保已安装：

```py
pip install flask-testing blinker

```

### LiveServer

LiveServer 是一个 Flask-testing 工具，允许您连接到无头浏览器，即不会将内容可视化呈现的浏览器（如 Firefox 或 Chrome），但会执行所有脚本和样式，并模拟用户交互。每当您需要在 JavaScript 交互后评估页面内容时，请使用 LiveServer。我们将使用 PhantomJS 作为我们的无头浏览器。我给您的建议是，您像我们的祖先一样安装旧浏览器，从源代码编译它。请按照[`phantomjs.org/build.html`](http://phantomjs.org/build.html)上的说明进行操作（您可能需要安装一些额外的库以获得 phantom 的全部功能）。`build.sh`文件将在必要时建议您安装它。

### 提示

编译**PhantomJS**后，确保它在您的 PATH 中被找到，将二进制文件`bin/phantomjs`移动到`/usr/local/bin`。

确保安装了 Selenium：

```py
pip install selenium

```

我们的代码将如下所示：

```py
# coding:utf-8

"""
Example adapted from https://pythonhosted.org/Flask-Testing/#testing-with-liveserver
"""

import urllib2
from urlparse import urljoin
from selenium import webdriver
from flask import Flask, render_template, jsonify, url_for
from flask.ext.testing import LiveServerTestCase
from random import choice

my_lines = ['Hello there!', 'How do you do?', 'Flask is great, ain't it?']

def setup(app):
    @app.route("/")
    def index_view():
        return render_template('js_index.html')

    @app.route("/text")
    def text_view():
        return jsonify({'text': choice(my_lines)})

def app_factory(name=None):
    name = name or __name__
    app = Flask(name)
    setup(app)
    return app

class IndexTest(LiveServerTestCase):
    def setUp(self):
        self.driver = webdriver.PhantomJS()

    def tearDown(self):
        self.driver.close()

    def create_app(self):
        app = app_factory()
        app.config['TESTING'] = True
        # default port is 5000
        app.config['LIVESERVER_PORT'] = 8943
        return app

    def test_server_is_up_and_running(self):
        resp = urllib2.urlopen(self.get_server_url())
        self.assertEqual(resp.code, 200)

    def test_random_text_was_loaded(self):
        with self.app.test_request_context():
            domain = self.get_server_url()
            path = url_for('.index_view')
            url = urljoin(domain, path)

            self.driver.get(url)
            fillme_element = self.driver.find_element_by_id('fillme')
            fillme_text = fillme_element.text
            self.assertIn(fillme_text, my_lines)

if __name__ == '__main__':
    import unittest
    unittest.main()
```

`templates/js_index.html`文件应如下所示：

```py
<html>
<head><title>Hello You</title></head>
<body>
<span id="fillme"></span>

<!-- Loading JQuery from CDN -->
<!-- what's a CDN? http://www.rackspace.com/knowledge_center/article/what-is-a-cdn -->
<script type="text/javascript" src="img/jquery-2.1.3.min.js"></script>
<script type="text/javascript">
  $(document).ready(function(){
    $.getJSON("{{ url_for('.text_view') }}",
    function(data){
       $('#fillme').text(data['text']);
    });
  });
</script>
</body></html>
```

前面的例子非常简单。我们定义了我们的工厂，它创建了我们的应用程序并附加了两个视图。一个返回一个带有脚本的`js_index.html`，该脚本查询我们的第二个视图以获取短语，并填充`fillme` HTML 元素，第二个视图以 JSON 格式返回一个从预定义列表中随机选择的短语。

然后我们定义`IndexTest`，它扩展了`LiveServerTestCase`，这是一个特殊的类，我们用它来运行我们的实时服务器测试。我们将我们的实时服务器设置为在不同的端口上运行，但这并不是必需的。

在`setUp`中，我们使用 selenium WebDriver 创建一个`driver`。该驱动程序类似于浏览器。我们将使用它通过 LiveServer 访问和检查我们的应用程序。`tearDown`确保每次测试后关闭我们的驱动程序并释放资源。

`test_server_is_up_and_running`是不言自明的，在现实世界的测试中实际上是不必要的。

然后我们有`test_random_text_was_loaded`，这是一个非常繁忙的测试。我们使用`test_request_context`来创建一个请求上下文，以便使用`url_open.get_server_url`生成我们的 URL 路径，这将返回我们的实时服务器 URL；我们将这个 URL 与我们的视图路径连接起来并加载到我们的驱动程序中。

使用加载的 URL（请注意，URL 不仅加载了，而且脚本也执行了），我们使用`find_element_by_id`来查找元素`fillme`并断言其文本内容具有预期值之一。这是一个简单的例子。例如，您可以测试按钮是否在预期位置；提交表单；并触发 JavaScript 函数。Selenium 加上 PhantomJS 是一个强大的组合。

### 提示

当您的开发是由功能测试驱动时，您实际上并没有使用**TDD**，而是**行为驱动开发**（**BDD**）。通常，两种技术的混合是您想要的。

### 额外的断言

在测试代码时，您会注意到一些测试有点重复。为了处理这种情况，可以创建一个具有特定例程的自定义 TestCases，并相应地扩展测试。使用 Flask-testing，您仍然需要这样做，但是要编写更少的代码来测试您的 Flask 视图，因为`flask.ext.testing.TestCase`捆绑了常见的断言，许多在 Django 等框架中找到。让我们看看最重要的（在我看来，当然）断言：

+   `assert_context(name, value)`: 这断言一个变量是否在模板上下文中。用它来验证给定的响应上下文对于一个变量具有正确的值。

+   `assert_redirects(response, location)`: 这断言了响应是一个重定向，并给出了它的位置。在写入存储后进行重定向是一个很好的做法，比如在成功的 POST 后，这是这个断言的一个很好的使用案例。

+   `assert_template_used(name, tmpl_name_attribute='name')`：这断言了请求中使用了给定的模板（如果您没有使用 Jinja2，则需要 `tmpl_name_attribute`；在我们的情况下不需要）；无论何时您渲染一个 HTML 模板，都可以使用它！

+   `assert404(response, message=None)`: 这断言了响应具有 404 HTTP 状态码；它对于“雨天”场景非常有用；也就是说，当有人试图访问不存在的内容时。它非常有用。

### JSON 处理

Flask-testing 为您提供了一个可爱的技巧。每当您从视图返回一个 JSON 响应时，您的响应将有一个额外的属性叫做 `json`。那就是您的 JSON 转换后的响应！以下是一个例子：

```py
# example from https://pythonhosted.org/Flask-Testing/#testing-json-responses
@app.route("/ajax/")
def some_json():
    return jsonify(success=True)

class TestViews(TestCase):
    def test_some_json(self):
        response = self.client.get("/ajax/")
        self.assertEquals(response.json, dict(success=True))
```

# 固定装置

良好的测试总是在考虑预定义的、可重现的应用程序状态下执行；也就是说，无论何时您在选择的状态下运行测试，结果都将是等价的。通常，这是通过自己设置数据库数据并清除缓存和任何临时文件（如果您使用外部服务，您应该模拟它们）来实现的。清除缓存和临时文件并不难，而设置数据库数据则不然。

如果您使用 **Flask-SQLAlchemy** 来保存您的数据，您需要在您的测试中某个地方硬编码如下：

```py
attributes = { … }
model = MyModel(**attributes)
db.session.add(model)
db.session.commit()
```

这种方法不易扩展，因为它不容易重复使用（当您将其定义为一个函数和一个方法时，为每个测试定义它）。有两种方法可以为测试填充您的数据库：**固定装置** 和 **伪随机数据**。

使用伪随机数据通常是特定于库的，并且生成的数据是上下文特定的，而不是静态的，但有时可能需要特定的编码，就像当您定义自己的字段或需要字段的不同值范围时一样。

固定装置是最直接的方法，因为您只需在文件中定义您的数据，并在每个测试中加载它。您可以通过导出数据库数据，根据您的方便进行编辑，或者自己编写。JSON 格式在这方面非常受欢迎。让我们看看如何实现这两种方法：

```py
# coding:utf-8
# == USING FIXTURES ===
import tempfile, os
import json

from flask import Flask
from flask.ext.testing import TestCase
from flask.ext.sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    gender = db.Column(db.String(1), default='U')

    def __unicode__(self):
        return self.name

def app_factory(name=None):
    name = name or __name__
    app = Flask(name)
    return app

class MyTestCase(TestCase):
    def create_app(self):
        app = app_factory()
        app.config['TESTING'] = True
        # db_fd: database file descriptor
        # we create a temporary file to hold our data
        self.db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        db.init_app(app)
        return app

    def load_fixture(self, path, model_cls):
        """
        Loads a json fixture into the database
        """
        fixture = json.load(open(path))

        for data in fixture:
            # Model accepts dict like parameter
            instance = model_cls(**data)
            # makes sure our session knows about our new instance
            db.session.add(instance)

        db.session.commit()

    def setUp(self):
        db.create_all()
        # you could load more fixtures if needed
        self.load_fixture('fixtures/users.json', User)

    def tearDown(self):
        # makes sure the session is removed
        db.session.remove()

        # close file descriptor
        os.close(self.db_fd)

        # delete temporary database file
        # as SQLite database is a single file, this is equivalent to a drop_all
        os.unlink(self.app.config['DATABASE'])

    def test_fixture(self):
        marie = User.query.filter(User.name.ilike('Marie%')).first()
        self.assertEqual(marie.gender, "F")

if __name__ == '__main__':
    import unittest
    unittest.main()
```

上述代码很简单。我们创建一个 SQLAlchemy 模型，将其链接到我们的应用程序，并在设置期间加载我们的固定装置。在 `tearDow`n 中，我们确保我们的数据库和 SQLAlchemy 会话对于下一个测试来说是全新的。我们的固定装置是使用 JSON 格式编写的，因为它足够快速且可读。

如果我们使用伪随机生成器来创建我们的用户（查找 Google 上关于这个主题的 **模糊测试**），我们可以这样做：

```py
def new_user(**kw):
    # this way we only know the user data in execution time
    # tests should consider it
    kw['name'] = kw.get('name', "%s %s" % (choice(names), choice(surnames)) )
    kw['gender'] = kw.get('gender', choice(['M', 'F', 'U']))
    return kw
user = User(**new_user())
db.session.add(user)
db.session.commit()
```

请注意，由于我们不是针对静态场景进行测试，我们的测试也会发生变化。通常情况下，固定装置在大多数情况下就足够了，但伪随机测试数据在大多数情况下更好，因为它迫使您的应用处理真实场景，而这些通常被忽略。

## 额外 - 集成测试

集成测试是一个非常常用的术语/概念，但其含义非常狭窄。它用于指代测试多个模块一起测试它们的集成。由于使用 Python 从同一代码库中测试多个模块通常是微不足道且透明的（这里导入，那里调用，以及一些输出检查），您通常会听到人们在指代测试他们的代码与不同代码库进行集成测试时使用术语 **集成测试**，或者当系统添加了新的关键功能时。

# 总结

哇！我们刚刚度过了一章关于软件测试的内容！这是令人自豪的成就。我们学到了一些概念，比如 TDD、白盒测试和黑盒测试。我们还学会了如何创建单元测试；测试我们的视图；使用 Gherkin 语言编写功能并使用 lettuce 进行测试；使用 Flask-testing、Selenium 和 PhantomJS 来测试用户角度的 HTML 响应；还学会了如何使用固定装置来控制我们应用程序的状态，以进行正确可重复的测试。现在，您可以使用不同的技术以正确的方式测试 Flask 应用程序，以满足不同的场景和需求。

在下一章中，事情会变得非常疯狂，因为我们的研究对象将是使用 Flask 的技巧。下一章将涵盖蓝图、会话、日志记录、调试等内容，让您能够创建更加健壮的软件。到时见！
