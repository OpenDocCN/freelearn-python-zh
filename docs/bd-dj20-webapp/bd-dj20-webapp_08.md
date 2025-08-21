# 第八章：测试 Answerly

在上一章中，我们为我们的问题和答案网站 Answerly 添加了搜索功能。然而，随着我们网站功能的增长，我们需要避免破坏现有的功能。为了确保我们的代码保持正常运行，我们将更仔细地测试我们的 Django 项目。

在本章中，我们将做以下事情：

+   安装 Coverage.py 以测量代码覆盖率

+   测量我们的 Django 项目的代码覆盖率

+   为我们的模型编写单元测试

+   为视图编写单元测试

+   为视图编写 Django 集成测试

+   为视图编写 Selenium 集成测试

让我们从安装 Coverage.py 开始。

# 安装 Coverage.py

**Coverage.py**是目前最流行的 Python 代码覆盖工具。它非常容易安装，因为可以从 PyPI 获取。让我们将其添加到我们的`requirements.txt`文件中：

```py
$ echo "coverage==4.4.2" >> requirements.txt
```

然后我们可以使用 pip 安装 Coverage.py：

```py
$ pip install -r requirements.txt
```

现在我们已经安装了 Coverage.py，我们可以开始测量我们的代码覆盖率。

# 为 Question.save()创建一个单元测试

Django 帮助您编写单元测试来测试代码的各个单元。如果我们的代码依赖于外部服务，那么我们可以使用标准的`unittest.mock`库来模拟该 API，防止对外部系统的请求。

让我们为`Question.save()`方法编写一个测试，以验证当我们保存一个`Question`时，它将被插入到 Elasticsearch 中。我们将在`django/qanda/tests.py`中编写这个测试：

```py
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase
from elasticsearch import Elasticsearch

from qanda.models import Question

class QuestionSaveTestCase(TestCase):
    """
    Tests Question.save()
    """

    @patch('qanda.service.elasticsearch.Elasticsearch')
    def test_elasticsearch_upsert_on_save(self, ElasticsearchMock):
        user = get_user_model().objects.create_user(
            username='unittest',
            password='unittest',
        )
        question_title = 'Unit test'
        question_body = 'some long text'
        q = Question(
            title=question_title,
            question=question_body,
            user=user,
        )
        q.save()

        self.assertIsNotNone(q.id)
        self.assertTrue(ElasticsearchMock.called)
        mock_client = ElasticsearchMock.return_value
        mock_client.update.assert_called_once_with(
            settings.ES_INDEX,
            id=q.id,
            body={
                'doc': {
                    '_type': 'doc',
                    'text': '{}\n{}'.format(question_title, question_body),
                    'question_body': question_body,
                    'title': question_title,
                    'id': q.id,
                    'created': q.created,
                },
                'doc_as_upsert': True,
            }
        )
```

在上面的代码示例中，我们创建了一个带有单个测试方法的`TestCase`。该方法创建一个用户，保存一个新的`Question`，然后断言模拟行为是否正确。

像大多数`TestCase`一样，`QuestionSaveTestCase`既使用了 Django 的测试 API，也使用了 Python 的`unittest`库中的代码（例如，`unittest.mock.patch()`）。让我们更仔细地看看 Django 的测试 API 如何使测试更容易。

`QuestionSaveTestCase`扩展了`django.test.TestCase`而不是`unittest.TestCase`，因为 Django 的`TestCase`提供了许多有用的功能，如下所示：

+   整个测试用例和每个测试都是原子数据库操作

+   Django 在每次测试前后都会清除数据库

+   `TestCase`提供了方便的`assert*()`方法，比如`self.assertInHTML()`（在*为视图创建单元测试*部分中更多讨论）

+   一个虚假的 HTTP 客户端来创建集成测试（在*为视图创建集成测试*部分中更多讨论）

由于 Django 的`TestCase`扩展了`unittest.TestCase`，因此当它遇到常规的`AssertionError`时，它仍然能够理解并正确执行。因此，如果`mock_client.update.assert_called_once_with()`引发`AssertionError`异常，Django 的测试运行器知道如何处理它。

让我们用`manage.py`运行我们的测试：

```py
$ cd django
$ python manage.py test
Creating test database for alias 'default'...
System check identified no issues (0 silenced).
.
----------------------------------------------------------------------
Ran 1 test in 0.094s

OK
Destroying test database for alias 'default'...
```

现在我们知道如何测试模型，我们可以继续测试视图。然而，在测试视图时，我们需要创建模型实例。使用模型的默认管理器来创建模型实例会变得太啰嗦。接下来，让我们使用 Factory Boy 更容易地创建测试所需的模型。

# 使用 Factory Boy 创建测试模型

在我们之前的测试中，我们使用`User.models.create_user`创建了一个`User`模型。然而，这要求我们提供用户名和密码，而我们并不真正关心。我们只需要一个用户，而不是特定的用户。对于我们的许多测试来说，`Question`和`Answer`也是如此。Factory Boy 库将帮助我们在测试中简洁地创建模型。

Factory Boy 对 Django 开发人员特别有用，因为它知道如何基于 Django 的`Model`类创建模型。

让我们安装 Factory Boy：

```py
$ pip install factory-boy==2.9.2
```

在这一部分，我们将使用 Factory Boy 创建一个`UserFactory`类和一个`QuestionFactory`类。由于`Question`模型必须在其`user`字段中有一个用户，`QuestionFactory`将向我们展示`Factory`类如何相互引用。

让我们从`UserFactory`开始。

# 创建一个 UserFactory

`Question`和`Answer`都与用户相关联。这意味着我们几乎在所有测试中都需要创建用户。使用模型管理器为每个测试生成所有相关模型非常冗长，并且分散了我们测试的重点。Django 为我们的测试提供了开箱即用的支持。但是，Django 的 fixtures 是单独的 JSON/YAML 文件，需要手动维护，否则它们将变得不同步并引起问题。Factory Boy 将通过让我们使用代码来帮助我们，即`UserFactory`，可以根据当前用户模型的状态在运行时简洁地创建用户模型实例。

我们的`UserFactory`将派生自 Factory Boy 的`DjangoModelFactory`类，该类知道如何处理 Django 模型。我们将使用内部`Meta`类告诉`UserFactory`它正在创建哪个模型（请注意，这与`Form`API 类似）。我们还将添加类属性以告诉 Factory Boy 如何设置模型字段的值。最后，我们将重写`_create`方法，使`UserFactory`使用管理器的`create_user()`方法而不是默认的`create()`方法。

让我们在`django/users/factories.py`中创建我们的`UserFactory`：

```py
from django.conf import settings

import factory

class UserFactory(factory.DjangoModelFactory):
    username = factory.Sequence(lambda n: 'user %d' % n)
    password = 'unittest'

    class Meta:
        model = settings.AUTH_USER_MODEL

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        manager = cls._get_manager(model_class)
        return manager.create_user(*args, **kwargs)
```

`UserFactory`是`DjangoModelFactory`的子类。`DjangoModelFactory`将查看我们类的`Meta`内部类（遵循与`Form`类相同的模式）。

让我们更仔细地看一下`UserFactory`的属性：

+   `password = 'unittest'`：这将为每个用户设置相同的密码。

+   `username = factory.Sequence(lambda n: 'user %d' % n)`: `Sequence`为每次工厂创建模型时的字段设置不同的值。`Sequence()`接受可调用对象，将其传递给工厂使用的次数，并使用可调用对象的返回值作为新实例的字段值。在我们的情况下，我们的用户将具有用户名，例如`user 0`和`user 1`。

最后，我们重写了`_create()`方法，因为`django.contrib.auth.models.User`模型具有异常的管理器。`DjangoModelFactory`的默认`_create`方法将使用模型的管理器的`create()`方法。对于大多数模型来说，这很好，但对于`User`模型来说效果不佳。要创建用户，我们应该真正使用`create_user`方法，以便我们可以传递明文密码并对其进行哈希处理以进行存储。这将让我们作为该用户进行身份验证。

让我们在 Django shell 中尝试一下我们的工厂：

```py
$ cd django
$ python manage.py shell
Python 3.6.3 (default, Oct 31 2017, 11:15:24) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
In [1]: from user.factories import UserFactory
In [2]:  user = UserFactory()
In [3]: user.username
Out[3]: 'user 0'
In [4]:  user2 = UserFactory()
In [5]:  assert user.username != user2.username
In [6]: user3 = UserFactory(username='custom')
In [7]: user3.username
Out[7]: 'custom'
```

在这个 Django shell 会话中，我们将注意到如何使用`UserFactory`：

+   我们可以使用单个无参数调用创建新模型，`UserFactory()`

+   每次调用都会导致唯一的用户名，`assert user.username != user2.username`

+   我们可以通过提供参数来更改工厂使用的值，`UserFactory(username='custom')`

接下来，让我们创建一个`QuestionFactory`。

# 创建 QuestionFactory

我们的许多测试将需要多个`Question`实例。但是，每个`Question`必须有一个用户。这可能会导致大量脆弱和冗长的代码。创建`QuestionFactory`将解决这个问题。

在前面的示例中，我们看到了如何使用`factory.Sequence`为每个新模型的属性赋予不同的值。Factory Boy 还提供了`factory.SubFactory`，其中我们可以指示字段的值是另一个工厂的结果。

让我们将`QuestionFactory`添加到`django/qanda/factories.py`中：

```py
from unittest.mock import patch

import factory

from qanda.models import Question
from user.factories import UserFactory

class QuestionFactory(factory.DjangoModelFactory):
    title = factory.Sequence(lambda n: 'Question #%d' % n)
    question = 'what is a question?'
    user = factory.SubFactory(UserFactory)

    class Meta:
        model = Question

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        with patch('qanda.service.elasticsearch.Elasticsearch'):
            return super()._create(model_class, *args, **kwargs)
```

我们的`QuestionFactory`与`UserFactory`非常相似。它们有以下共同点：

+   派生自`factory.DjangoModelFactory`

+   有一个`Meta`类

+   使用`factory.Sequence`为字段提供自定义值

+   有一个硬编码的值

有两个重要的区别：

+   `QuestionFactory`的`user`字段使用`SubFactory`，为每个`Question`创建一个新的用户，该用户是使用`UserFactory`创建的。

+   `QuestionFactory`的`_create`方法模拟了 Elasticsearch 服务，以便在创建模型时不会尝试连接到该服务。否则，它调用默认的`_create()`方法。

为了看到我们的`QuestionFactory`的实际应用，让我们为我们的`DailyQuestionList`视图编写一个单元测试。

# 创建一个视图的单元测试

在这一部分，我们将为我们的`DailyQuestionList`视图编写一个视图单元测试。

对视图进行单元测试意味着直接向视图传递一个请求，并断言响应是否符合我们的期望。由于我们直接将请求传递给视图，我们还需要直接传递视图通常会接收的任何参数，这些参数从请求的 URL 中解析出来。从 URL 路径中解析值是请求路由的责任，在视图单元测试中我们不使用它。

让我们来看看`django/qanda/tests.py`中的`DailyQuestionListTestCase`类：

```py
from datetime import date

from django.test import TestCase, RequestFactory

from qanda.factories import QuestionFactory
from qanda.views import DailyQuestionList

QUESTION_CREATED_STRFTIME = '%Y-%m-%d %H:%M'

class DailyQuestionListTestCase(TestCase):
"""
Tests the DailyQuestionList view
"""
QUESTION_LIST_NEEDLE_TEMPLATE = '''
<li >
    <a href="/q/{id}" >{title}</a >
    by {username} on {date}
</li >
'''

REQUEST = RequestFactory().get(path='/q/2030-12-31')
TODAY = date.today()

def test_GET_on_day_with_many_questions(self):
    todays_questions = [QuestionFactory() for _ in range(10)]

    response = DailyQuestionList.as_view()(
        self.REQUEST,
        year=self.TODAY.year,
        month=self.TODAY.month,
        day=self.TODAY.day
    )

    self.assertEqual(200, response.status_code)
    self.assertEqual(10, response.context_data['object_list'].count())
    rendered_content = response.rendered_content
    for question in todays_questions:
        needle = self.QUESTION_LIST_NEEDLE_TEMPLATE.format(
            id=question.id,
            title=question.title,
            username=question.user.username,
            date=question.created.strftime(QUESTION_CREATED_STRFTIME)
        )
        self.assertInHTML(needle, rendered_content)
```

让我们更仔细地看一下我们见过的新 API：

+   `RequestFactory().get(path=...)`: `RequestFactory`是一个用于创建测试视图的 HTTP 请求的实用工具。注意这里我们请求的`path`是任意的，因为它不会被用于路由。

+   `DailyQuestionList.as_view()(...)`: 我们已经讨论过每个基于类的视图都有一个`as_view()`方法，它返回一个可调用对象，但我们以前没有使用过。在这里，我们传递请求、年、月和日来执行视图。

+   `response.context_data['object_list'].count()`:我们的视图返回的响应仍然保留了它的上下文。我们可以使用这个上下文来断言视图是否工作正确，比起评估 HTML 更容易。

+   `response.rendered_content`: `rendered_content`属性让我们可以访问响应的渲染模板。

+   `self.assertInHTML(needle, rendered_content)`: `TestCase.assertInHTML()`让我们可以断言一个 HTML 片段是否在另一个 HTML 片段中。`assertInHTML()`知道如何解析 HTML，不关心属性顺序或空白。在测试视图时，我们经常需要检查响应中是否存在特定的 HTML 片段。

现在我们已经为一个视图创建了一个单元测试，让我们看看通过为`QuestionDetailView`创建一个集成测试来创建一个视图的集成测试。

# 创建一个视图集成测试

视图集成测试使用与单元测试相同的`django.test.TestCase`类。集成测试将告诉我们我们的项目是否能够将请求路由到视图并返回正确的响应。集成测试请求将不得不通过项目配置的所有中间件和 URL 路由。为了帮助我们编写集成测试，Django 提供了`TestCase.client`。

`TestCase.client`是`TestCase`提供的一个实用工具，让我们可以向我们的项目发送 HTTP 请求（它不能发送外部 HTTP 请求）。Django 会正常处理这些请求。`client`还为我们提供了方便的方法，比如`client.login()`，一种开始认证会话的方法。一个`TestCase`类也会在每个测试之间重置它的`client`。

让我们在`django/qanda/tests.py`中为`QuestionDetailView`编写一个集成测试：

```py
from django.test import TestCase

from qanda.factories import QuestionFactory
from user.factories import UserFactory

QUESTION_CREATED_STRFTIME = '%Y-%m-%d %H:%M'

class QuestionDetailViewTestCase(TestCase):
    QUESTION_DISPLAY_SNIPPET = '''
    <div class="question" >
      <div class="meta col-sm-12" >
        <h1 >{title}</h1 >
        Asked by {user} on {date}
      </div >
      <div class="body col-sm-12" >
        {body}
      </div >
    </div >'''
    LOGIN_TO_POST_ANSWERS = 'Login to post answers.'

    def test_logged_in_user_can_post_answers(self):
        question = QuestionFactory()

        self.assertTrue(self.client.login(
            username=question.user.username,
            password=UserFactory.password)
        )
        response = self.client.get('/q/{}'.format(question.id))
        rendered_content = response.rendered_content

        self.assertEqual(200, response.status_code)

         self.assertInHTML(self.NO_ANSWERS_SNIPPET, rendered_content)

        template_names = [t.name for t in response.templates]
        self.assertIn('qanda/common/post_answer.html', template_names)

        question_needle = self.QUESTION_DISPLAY_SNIPPET.format(
            title=question.title,
            user=question.user.username,
            date=question.created.strftime(QUESTION_CREATED_STRFTIME),
            body=QuestionFactory.question,
        )
        self.assertInHTML(question_needle, rendered_content)
```

在这个示例中，我们登录然后请求`Question`的详细视图。我们对结果进行多次断言以确认它是正确的（包括检查使用的模板的名称）。

让我们更详细地检查一些代码：

+   `self.client.login(...)`: 这开始了一个认证会话。所有未来的请求都将作为该用户进行认证，直到我们调用`client.logout()`。

+   `self.client.get('/q/{}'.format(question.id))`: 这使用我们的客户端发出一个 HTTP `GET`请求。不同于我们使用`RequestFactory`时，我们提供的路径是为了将我们的请求路由到一个视图（注意我们在测试中从未直接引用视图）。这返回了我们的视图创建的响应。

+   `[t.name for t in response.templates]`: 当客户端的响应渲染时，客户端会更新响应的使用的模板列表。在详细视图的情况下，我们使用了多个模板。为了检查我们是否显示了发布答案的 UI，我们将检查`qanda/common/post_answer.html`文件是否是使用的模板之一。

通过这种类型的测试，我们可以非常有信心地确认我们的视图在用户发出请求时是否有效。然而，这确实将测试与项目的配置耦合在一起。即使是来自第三方应用的视图，集成测试也是有意义的，以确认它们是否被正确使用。如果你正在开发一个库应用，你可能会发现最好使用单元测试。

接下来，让我们通过使用 Selenium 来测试我们的 Django 和前端代码是否都正确工作，创建一个实时服务器测试用例。

# 创建一个实时服务器集成测试

我们将编写的最后一种类型的测试是实时服务器集成测试。在这个测试中，我们将启动一个测试 Django 服务器，并使用 Selenium 控制 Google Chrome 向其发出请求。

Selenium 是一个工具，它具有许多语言的绑定（包括 Python），可以让你控制一个网页浏览器。这样你就可以测试真实浏览器在使用你的项目时的行为，因为你是用真实浏览器测试你的项目。

这种类型的测试有一些限制：

+   实时测试通常需要按顺序运行

+   很容易在测试之间泄漏状态。

+   使用浏览器比`TestCase.client()`慢得多（浏览器会发出真正的 HTTP 请求）

尽管存在所有这些缺点，实时服务器测试在当前客户端网页应用如此强大的时代是一个非常宝贵的工具。

让我们首先设置 Selenium。

# 设置 Selenium

让我们通过使用`pip`来将 Selenium 添加到我们的项目中进行安装：

```py
$pip install selenium==3.8.0
```

接下来，我们需要特定的 webdriver，告诉 Selenium 如何与 Chrome 通信。Google 在[`sites.google.com/a/chromium.org/chromedriver/`](https://sites.google.com/a/chromium.org/chromedriver/)提供了一个**chromedriver**。在我们的情况下，让我们把它保存在项目目录的根目录下。然后，让我们在`django/conf/settings.py`中添加该驱动程序的路径：

```py
CHROMEDRIVER = os.path.join(BASE_DIR, '../chromedriver')
```

最后，请确保你的计算机上安装了 Google Chrome。如果没有，你可以在[`www.google.com/chrome/index.html`](https://www.google.com/chrome/index.html)下载它。

所有主要的浏览器都声称对 Selenium 有一定程度的支持。如果你不喜欢 Google Chrome，你可以尝试其他浏览器。有关详细信息，请参阅 Selenium 的文档（[`www.seleniumhq.org/about/platforms.jsp`](http://www.seleniumhq.org/about/platforms.jsp)）。

# 使用 Django 服务器和 Selenium 进行测试

现在我们已经设置好了 Selenium，我们可以创建我们的实时服务器测试。当我们的项目有很多 JavaScript 时，实时服务器测试特别有用。然而，Answerly 并没有任何 JavaScript。然而，Django 的表单确实利用了大多数浏览器（包括 Google Chrome）支持的 HTML5 表单属性。我们仍然可以测试我们的代码是否正确地使用了这些功能。

在这个测试中，我们将检查用户是否可以提交一个空的问题。`title`和`question`字段应该被标记为`required`，这样如果这些字段为空，浏览器就不会提交表单。

让我们在`django/qanda/tests.py`中添加一个新的测试：

```py
from django.contrib.staticfiles.testing import StaticLiveServerTestCase

from selenium.webdriver.chrome.webdriver import WebDriver

from user.factories import UserFactory

class AskQuestionTestCase(StaticLiveServerTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.selenium = WebDriver(executable_path=settings.CHROMEDRIVER)
        cls.selenium.implicitly_wait(10)

    @classmethod
    def tearDownClass(cls):
        cls.selenium.quit()
        super().tearDownClass()

    def setUp(self):
        self.user = UserFactory()

    def test_cant_ask_blank_question(self):
        initial_question_count = Question.objects.count()

        self.selenium.get('%s%s' % (self.live_server_url, '/user/login'))

        username_input = self.selenium.find_element_by_name("username")
        username_input.send_keys(self.user.username)
        password_input = self.selenium.find_element_by_name("password")
        password_input.send_keys(UserFactory.password)
        self.selenium.find_element_by_id('log_in').click()

        self.selenium.find_element_by_link_text("Ask").click()
        ask_question_url = self.selenium.current_url
        submit_btn = self.selenium.find_element_by_id('ask')
        submit_btn.click()
        after_empty_submit_click = self.selenium.current_url

        self.assertEqual(ask_question_url, after_empty_submit_click)
        self.assertEqual(initial_question_count, Question.objects.count())
```

让我们来看看这个测试中引入的一些新的 Django 特性。然后，我们将审查我们的 Selenium 代码：

+   `class AskQuestionTestCase(StaticLiveServerTestCase)`: `StaticLiveServerTestCase`启动了一个 Django 服务器，并确保静态文件被正确地提供。你不必运行`python manage.py collectstatic`。文件将被正确地路由，就像你运行`python manage.py runserver`一样。

+   `def setUpClass(cls)`: 所有的 Django 测试用例都支持`setUpClass()`、`setup()`、`teardown()`和`teardownClass()`方法，就像往常一样。`setUpClass`和`tearDownClass()`每个`TestCase`只运行一次（分别在之前和之后）。这使它们非常适合昂贵的操作，比如用 Selenium 连接到 Google Chrome。

+   `self.live_server_url`：这是实时服务器的 URL。

Selenium 允许我们使用 API 与浏览器进行交互。本书不侧重于 Selenium，但让我们来介绍一些`WebDriver`类的关键方法：

+   `cls.selenium = WebDriver(executable_path=settings.CHROMEDRIVER)`: 这实例化了一个 WebDriver 实例，其中包含到`ChromeDriver`可执行文件的路径（我们在前面的*设置 Selenium*部分中下载了）。我们将`ChromeDriver`可执行文件的路径存储在设置中，以便在这里轻松引用它。

+   `selenium.find_element_by_name(...)`: 这返回一个其`name`属性与提供的参数匹配的 HTML 元素。`name`属性被所有值由表单处理的`<input>`元素使用，因此对于数据输入特别有用。

+   `self.selenium.find_element_by_id(...)`: 这与前面的步骤类似，只是通过其`id`属性查找匹配的元素。

+   `self.selenium.current_url`: 这是浏览器的当前 URL。这对于确认我们是否在预期的页面上很有用。

+   `username_input.send_keys(...)`: `send_keys()`方法允许我们将传递的字符串输入到 HTML 元素中。这对于`<input type='text'>`和`<input type='password'>`元素特别有用。

+   `submit_btn.click()`: 这会触发对元素的点击。

这个测试以用户身份登录，尝试提交表单，并断言仍然在同一个页面上。不幸的是，虽然带有空的必填`input`元素的表单不会自行提交，但没有 API 直接确认这一点。相反，我们确认我们没有提交，因为浏览器仍然在与之前点击提交之前相同的 URL 上（根据`self.selenium.current_url`）。

# 总结

在本章中，我们学习了如何在 Django 项目中测量代码覆盖率，以及如何编写四种不同类型的测试——用于测试任何函数或类的单元测试，包括模型和表单；以及用于使用`RequestFactory`测试视图的视图单元测试。我们介绍了如何查看集成测试，用于测试请求路由到视图并返回正确响应，以及用于测试客户端和服务器端代码是否正确配合工作的实时服务器集成测试。

现在我们有了一些测试，让我们将 Answerly 部署到生产环境中。
