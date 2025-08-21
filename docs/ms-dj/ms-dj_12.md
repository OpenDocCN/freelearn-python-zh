# 第十二章：Django 中的测试

# 测试简介

像所有成熟的编程语言一样，Django 提供了内置的*单元测试*功能。单元测试是一种软件测试过程，其中测试软件应用程序的各个单元，以确保它们执行预期的操作。

单元测试可以在多个级别进行-从测试单个方法以查看它是否返回正确的值以及如何处理无效数据，到测试整套方法以确保一系列用户输入导致期望的结果。

单元测试基于四个基本概念：

1.  **测试装置**是执行测试所需的设置。这可能包括数据库、样本数据集和服务器设置。测试装置还可能包括在测试执行后需要进行的任何清理操作。

1.  **测试用例**是测试的基本单元。测试用例检查给定的输入是否导致预期的结果。

1.  **测试套件**是一些测试用例或其他测试套件，作为一个组执行。

1.  **测试运行器**是控制测试执行并将测试结果反馈给用户的软件程序。

软件测试是一个深入而详细的主题，本章应被视为对单元测试的简要介绍。互联网上有大量关于软件测试理论和方法的资源，我鼓励你就这个重要主题进行自己的研究。有关 Django 对单元测试方法的更详细讨论，请参阅 Django 项目网站。

# 引入自动化测试

## 什么是自动化测试？

在本书中，你一直在测试代码；也许甚至没有意识到。每当你使用 Django shell 来查看一个函数是否有效，或者查看给定输入的输出时，你都在测试你的代码。例如，在第二章中，*视图和 URLconfs*，我们向一个期望整数的视图传递了一个字符串，以生成`TypeError`异常。

测试是应用程序开发的正常部分，但自动化测试的不同之处在于系统为你完成了测试工作。你只需创建一组测试，然后在对应用程序进行更改时，可以检查你的代码是否仍然按照最初的意图工作，而无需进行耗时的手动测试。

## 那么为什么要创建测试？

如果创建像本书中那样简单的应用程序是你在 Django 编程中的最后一步，那么确实，你不需要知道如何创建自动化测试。但是，如果你希望成为一名专业程序员和/或在更复杂的项目上工作，你需要知道如何创建自动化测试。

创建自动化测试将会：

+   **节省时间**：手动测试大型应用程序组件之间的复杂交互是耗时且容易出错的。自动化测试可以节省时间，让你专注于编程。

+   **预防问题**：测试突出显示了代码的内部工作原理，因此你可以看到哪里出了问题。

+   **看起来专业**：专业人士编写测试。Django 的原始开发人员之一 Jacob Kaplan-Moss 说：“没有测试的代码从设计上就是有问题的。”

+   **改善团队合作**：测试可以确保同事们不会无意中破坏你的代码（而你也不会在不知情的情况下破坏他们的代码）。

# 基本测试策略

有许多方法可以用来编写测试。一些程序员遵循一种称为**测试驱动开发**的纪律；他们实际上是在编写代码之前编写他们的测试。这可能看起来有些反直觉，但事实上，这与大多数人通常会做的事情相似：他们描述一个问题，然后创建一些代码来解决它。

测试驱动开发只是在 Python 测试用例中正式化了问题。更常见的是，测试的新手会创建一些代码，然后决定它应该有一些测试。也许更好的做法是早些时候编写一些测试，但现在开始也不算太晚。

# 编写一个测试

要创建您的第一个测试，让我们在您的 Book 模型中引入一个错误。

假设您已经决定在您的 Book 模型上创建一个自定义方法，以指示书籍是否最近出版。您的 Book 模型可能如下所示：

```py
import datetime 
from django.utils import timezone 

from django.db import models 

# ... # 

class Book(models.Model): 
    title = models.CharField(max_length=100) 
    authors = models.ManyToManyField(Author) 
    publisher = models.ForeignKey(Publisher) 
    publication_date = models.DateField() 

    def recent_publication(self): 
        return self.publication_date >= timezone.now().date() 
datetime.timedelta(weeks=8) 

    # ... # 

```

首先，我们导入了两个新模块：Python 的`datetime`和`django.utils`中的`timezone`。我们需要这些模块来进行日期计算。然后，我们在`Book`模型中添加了一个名为`recent_publication`的自定义方法，该方法计算出八周前的日期，并在书籍的出版日期更近时返回 true。

所以让我们跳到交互式 shell 并测试我们的新方法：

```py
python manage.py shell 

>>> from books.models import Book 
>>> import datetime 
>>> from django.utils import timezone 
>>> book = Book.objects.get(id=1) 
>>> book.title 
'Mastering Django: Core' 
>>> book.publication_date 
datetime.date(2016, 5, 1) 
>>>book.publication_date >= timezone.now().date()-datetime.timedelta(weeks=8) 
True 

```

到目前为止，一切都很顺利，我们已经导入了我们的书籍模型并检索到了一本书。今天是 2016 年 6 月 11 日，我已经在数据库中输入了我的书的出版日期为 5 月 1 日，这比八周前还要早，所以函数正确地返回了`True`。

显然，您将不得不修改数据中的出版日期，以便在您完成这个练习时，这个练习仍然对您有效。

现在让我们看看如果我们将出版日期设置为未来的某个时间，比如说 9 月 1 日会发生什么：

```py
>>> book.publication_date 
datetime.date(2016, 9, 1) 
>>>book.publication_date >= timezone.now().date()-datetime.timedelta(weeks=8) 
True 

```

哎呀！这里显然有些问题。您应该能够很快地看到逻辑上的错误-八周前之后的任何日期都将返回 true，包括未来的日期。

所以，暂且不管这是一个相当牵强的例子，现在让我们创建一个暴露我们错误逻辑的测试。

# 创建一个测试

当您使用 Django 的`startapp`命令创建了您的 books 应用程序时，它在您的应用程序目录中创建了一个名为`tests.py`的文件。这就是 books 应用程序的任何测试应该放置的地方。所以让我们马上开始编写一个测试：

```py
import datetime 
from django.utils import timezone 
from django.test import TestCase 
from .models import Book 

class BookMethodTests(TestCase): 

    def test_recent_pub(self): 
""" 
        recent_publication() should return False for future publication  
        dates. 
        """ 

        futuredate = timezone.now().date() + datetime.timedelta(days=5) 
        future_pub = Book(publication_date=futuredate) 
        self.assertEqual(future_pub.recent_publication(), False) 

```

这应该非常简单明了，因为它几乎与我们在 Django shell 中所做的一样，唯一的真正区别是我们现在将我们的测试代码封装在一个类中，并创建了一个断言，用于测试我们的`recent_publication()`方法是否与未来日期相匹配。

我们将在本章后面更详细地介绍测试类和`assertEqual`方法-现在，我们只想在进入更复杂的主题之前，看一下测试是如何在非常基本的水平上工作的。

# 运行测试

现在我们已经创建了我们的测试，我们需要运行它。幸运的是，这非常容易做到，只需跳转到您的终端并键入：

```py
python manage.py test books 

```

片刻之后，Django 应该打印出类似于这样的内容：

```py
Creating test database for alias 'default'... 
F 
====================================================================== 
FAIL: test_recent_pub (books.tests.BookMethodTests) 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
  File "C:\Users\Nigel\ ... mysite\books\tests.py", line 25, in test_recent_pub 
    self.assertEqual(future_pub.recent_publication(), False) 
AssertionError: True != False 

---------------------------------------------------------------------- 
Ran 1 test in 0.000s 

FAILED (failures=1) 
Destroying test database for alias 'default'... 

```

发生的事情是这样的：

+   Python `manage.py test books`在 books 应用程序中查找测试。

+   它找到了`django.test.TestCase`类的一个子类

+   它为测试目的创建了一个特殊的数据库

+   它寻找以“test”开头的方法

+   在`test_recent_pub`中，它创建了一个`Book`实例，其`publication_date`字段是未来的 5 天；而

+   使用`assertEqual()`方法，它发现它的`recent_publication()`返回`True`，而应该返回`False`。

测试告诉我们哪个测试失败了，甚至还告诉了失败发生的行。还要注意，如果您使用的是*nix 系统或 Mac，文件路径将会有所不同。

这就是 Django 中测试的非常基本的介绍。正如我在本章开头所说的，测试是一个深入而详细的主题，对于您作为程序员的职业非常重要。我不可能在一个章节中涵盖所有测试的方面，所以我鼓励您深入研究本章中提到的一些资源以及 Django 文档。

在本章的其余部分，我将介绍 Django 为您提供的各种测试工具。

# 测试工具

Django 提供了一套在编写测试时非常方便的工具。

## 测试客户端

测试客户端是一个 Python 类，充当虚拟网络浏览器，允许您以编程方式测试视图并与 Django 应用程序进行交互。测试客户端可以做的一些事情包括：

+   模拟 URL 上的`GET`和`POST`请求，并观察响应-从低级 HTTP（结果标头和状态代码）到页面内容的一切。

+   查看重定向链（如果有）并检查每一步的 URL 和状态代码。

+   测试给定请求是否由给定的 Django 模板呈现，并且模板上下文包含某些值。

请注意，测试客户端并不打算替代 Selenium（有关更多信息，请访问[`seleniumhq.org/`](http://seleniumhq.org/)）或其他浏览器框架。Django 的测试客户端有不同的重点。简而言之：

+   使用 Django 的测试客户端来确保正确的模板被渲染，并且模板传递了正确的上下文数据。

+   使用浏览器框架（如 Selenium）测试呈现的 HTML 和网页的行为，即 JavaScript 功能。Django 还为这些框架提供了特殊的支持；有关更多详细信息，请参阅`LiveServerTestCase`部分。

全面的测试套件应该结合使用这两种测试类型。

有关 Django 测试客户端的更详细信息和示例，请参阅 Django 项目网站。

## 提供的 TestCase 类

普通的 Python 单元测试类扩展了`unittest.TestCase`的基类。Django 提供了一些这个基类的扩展：

### 简单的 TestCase

扩展`unittest.TestCase`，具有一些基本功能，如：

+   保存和恢复 Python 警告机制的状态。

+   添加了一些有用的断言，包括：

+   检查可调用对象是否引发了特定异常。

+   测试表单字段的呈现和错误处理。

+   测试 HTML 响应中是否存在/缺少给定的片段。

+   验证模板是否已/未用于生成给定的响应内容。

+   验证应用程序执行了 HTTP 重定向。

+   强大地测试两个 HTML 片段的相等性/不相等性或包含关系。

+   强大地测试两个 XML 片段的相等性/不相等性。

+   强大地测试两个 JSON 片段的相等性。

+   使用修改后的设置运行测试的能力。

+   使用测试`Client`。

+   自定义测试时间 URL 映射。

### Transaction TestCase

Django 的`TestCase`类（在下一段中描述）利用数据库事务设施来加快在每个测试开始时将数据库重置为已知状态的过程。然而，这样做的一个后果是，一些数据库行为无法在 Django 的`TestCase`类中进行测试。

在这些情况下，您应该使用`TransactionTestCase`。`TransactionTestCase`和`TestCase`除了数据库重置到已知状态的方式和测试代码测试提交和回滚的效果外，两者是相同的：

+   `TransactionTestCase`通过截断所有表在测试运行后重置数据库。`TransactionTestCase`可以调用提交和回滚，并观察这些调用对数据库的影响。

+   另一方面，`TestCase`在测试后不会截断表。相反，它将测试代码封装在数据库事务中，在测试结束时回滚。这保证了测试结束时的回滚将数据库恢复到其初始状态。

`TransactionTestCase`继承自`SimpleTestCase`。

### TestCase

这个类提供了一些对于测试网站有用的额外功能。将普通的`unittest.TestCase`转换为 Django 的`TestCase`很容易：只需将测试的基类从`unittest.TestCase`更改为`django.test.TestCase`。所有标准的 Python 单元测试功能仍然可用，但它将增加一些有用的附加功能，包括：

+   自动加载 fixture。

+   将测试包装在两个嵌套的`atomic`块中：一个用于整个类，一个用于每个测试。

+   创建一个`TestClient`实例。

+   用于测试重定向和表单错误等内容的 Django 特定断言。

`TestCase`继承自`TransactionTestCase`。

### LiveServerTestCase

`LiveServerTestCase`基本上与`TransactionTestCase`相同，只是多了一个功能：它在设置时在后台启动一个实时的 Django 服务器，并在拆卸时关闭它。这允许使用除 Django 虚拟客户端之外的自动化测试客户端，例如 Selenium 客户端，来在浏览器中执行一系列功能测试并模拟真实用户的操作。

## 测试用例特性

### 默认测试客户端

`*TestCase`实例中的每个测试用例都可以访问 Django 测试客户端的一个实例。可以将此客户端访问为`self.client`。每个测试都会重新创建此客户端，因此您不必担心状态（例如 cookies）从一个测试传递到另一个测试。这意味着，而不是在每个测试中实例化`Client`：

```py
import unittest 
from django.test import Client 

class SimpleTest(unittest.TestCase): 
    def test_details(self): 
        client = Client() 
        response = client.get('/customer/details/') 
        self.assertEqual(response.status_code, 200) 

    def test_index(self): 
        client = Client() 
        response = client.get('/customer/index/') 
        self.assertEqual(response.status_code, 200) 

```

...您可以像这样引用`self.client`：

```py
from django.test import TestCase 

class SimpleTest(TestCase): 
    def test_details(self): 
        response = self.client.get('/customer/details/') 
        self.assertEqual(response.status_code, 200) 

    def test_index(self): 
        response = self.client.get('/customer/index/') 
        self.assertEqual(response.status_code, 200) 

```

### fixture 加载

如果数据库支持的网站的测试用例没有任何数据，则没有多大用处。为了方便地将测试数据放入数据库，Django 的自定义`TransactionTestCase`类提供了一种加载 fixtures 的方法。fixture 是 Django 知道如何导入到数据库中的数据集合。例如，如果您的网站有用户帐户，您可能会设置一个虚假用户帐户的 fixture，以便在测试期间填充数据库。

创建 fixture 的最直接方法是使用`manage.pydumpdata`命令。这假设您的数据库中已经有一些数据。有关更多详细信息，请参阅`dumpdata`文档。创建 fixture 并将其放置在`INSTALLED_APPS`中的`fixtures`目录中后，您可以通过在`django.test.TestCase`子类的`fixtures`类属性上指定它来在单元测试中使用它：

```py
from django.test import TestCase 
from myapp.models import Animal 

class AnimalTestCase(TestCase): 
    fixtures = ['mammals.json', 'birds'] 

    def setUp(self): 
        # Test definitions as before. 
        call_setup_methods() 

    def testFluffyAnimals(self): 
        # A test that uses the fixtures. 
        call_some_test_code() 

```

具体来说，将发生以下情况：

+   在每个测试用例开始之前，在运行`setUp()`之前，Django 将刷新数据库，将数据库返回到直接在调用`migrate`之后的状态。

+   然后，所有命名的 fixtures 都将被安装。在此示例中，Django 将安装名为`mammals`的任何 JSON fixture，然后是名为`birds`的任何 fixture。有关定义和安装 fixtures 的更多详细信息，请参阅`loaddata`文档。

这个刷新/加载过程对测试用例中的每个测试都会重复进行，因此您可以确保一个测试的结果不会受到另一个测试或测试执行顺序的影响。默认情况下，fixture 只加载到`default`数据库中。如果您使用多个数据库并设置`multi_db=True`，fixture 将加载到所有数据库中。

### 覆盖设置

### 注意

使用函数在测试中临时更改设置的值。不要直接操作`django.conf.settings`，因为 Django 不会在此类操作后恢复原始值。

#### settings()

为了测试目的，通常在运行测试代码后临时更改设置并恢复到原始值是很有用的。对于这种用例，Django 提供了一个标准的 Python 上下文管理器（参见 PEP 343at [`www.python.org/dev/peps/pep-0343`](https://www.python.org/dev/peps/pep-0343)）称为`settings()`，可以像这样使用：

```py
from django.test import TestCase 

class LoginTestCase(TestCase): 

    def test_login(self): 

        # First check for the default behavior 
        response = self.client.get('/sekrit/') 
        self.assertRedirects(response, '/accounts/login/?next=/sekrit/') 

        # Then override the LOGIN_URL setting 
        with self.settings(LOGIN_URL='/other/login/'): 
            response = self.client.get('/sekrit/') 
            self.assertRedirects(response, '/other/login/?next=/sekrit/') 

```

此示例将在`with`块中覆盖`LOGIN_URL`设置，并在之后将其值重置为先前的状态。

#### modify_settings()

重新定义包含值列表的设置可能会变得难以处理。实际上，添加或删除值通常就足够了。`modify_settings()`上下文管理器使这变得很容易：

```py
from django.test import TestCase 

class MiddlewareTestCase(TestCase): 

    def test_cache_middleware(self): 
        with self.modify_settings(MIDDLEWARE_CLASSES={ 
'append': 'django.middleware.cache.FetchFromCacheMiddleware', 
'prepend': 'django.middleware.cache.UpdateCacheMiddleware', 
'remove': [ 
 'django.contrib.sessions.middleware.SessionMiddleware', 
 'django.contrib.auth.middleware.AuthenticationMiddleware',  
 'django.contrib.messages.middleware.MessageMiddleware', 
            ], 
        }): 
            response = self.client.get('/') 
            # ... 

```

对于每个操作，您可以提供一个值列表或一个字符串。当值已经存在于列表中时，`append`和`prepend`没有效果；当值不存在时，`remove`也没有效果。

#### override_settings()

如果要为测试方法覆盖设置，Django 提供了`override_settings()`装饰器（请参阅[`www.python.org/dev/peps/pep-0318`](https://www.python.org/dev/peps/pep-0318)的 PEP 318）。用法如下：

```py
from django.test import TestCase, override_settings 

class LoginTestCase(TestCase): 

    @override_settings(LOGIN_URL='/other/login/') 
    def test_login(self): 
        response = self.client.get('/sekrit/') 
        self.assertRedirects(response, '/other/login/?next=/sekrit/') 

```

装饰器也可以应用于`TestCase`类：

```py
from django.test import TestCase, override_settings 

@override_settings(LOGIN_URL='/other/login/') 
class LoginTestCase(TestCase): 

    def test_login(self): 
        response = self.client.get('/sekrit/') 
        self.assertRedirects(response, '/other/login/?next=/sekrit/') 

```

#### modify_settings()

同样，Django 还提供了`modify_settings()`装饰器：

```py
from django.test import TestCase, modify_settings 

class MiddlewareTestCase(TestCase): 

    @modify_settings(MIDDLEWARE_CLASSES={ 
'append': 'django.middleware.cache.FetchFromCacheMiddleware', 
'prepend': 'django.middleware.cache.UpdateCacheMiddleware', 
    }) 
    def test_cache_middleware(self): 
        response = self.client.get('/') 
        # ... 

```

装饰器也可以应用于测试用例类：

```py
from django.test import TestCase, modify_settings 

@modify_settings(MIDDLEWARE_CLASSES={ 
'append': 'django.middleware.cache.FetchFromCacheMiddleware', 
'prepend': 'django.middleware.cache.UpdateCacheMiddleware', 
}) 
class MiddlewareTestCase(TestCase): 

    def test_cache_middleware(self): 
        response = self.client.get('/') 
        # ... 

```

在覆盖设置时，请确保处理应用程序代码使用缓存或类似功能保留状态的情况，即使更改了设置。Django 提供了`django.test.signals.setting_changed`信号，让您注册回调以在更改设置时清理和重置状态。

### 断言

由于 Python 的普通`unittest.TestCase`类实现了`assertTrue()`和`assertEqual()`等断言方法，Django 的自定义`TestCase`类提供了许多对测试 Web 应用程序有用的自定义断言方法：

+   `assertRaisesMessage`：断言可调用对象的执行引发了带有`expected_message`表示的异常。

+   `assertFieldOutput`：断言表单字段对各种输入的行为是否正确。

+   `assertFormError`：断言表单上的字段在表单上呈现时引发提供的错误列表。

+   `assertFormsetError`：断言`formset`在呈现时引发提供的错误列表。

+   `assertContains`：断言`Response`实例产生了给定的`status_code`，并且`text`出现在响应内容中。

+   `assertNotContains`：断言`Response`实例产生了给定的`status_code`，并且`text`不出现在响应内容中。

+   `assertTemplateUsed`：断言在呈现响应时使用了给定名称的模板。名称是一个字符串，例如`'admin/index.html'`。

+   `assertTemplateNotUsed`：断言在呈现响应时未使用给定名称的模板。

+   `assertRedirects`：断言响应返回了`status_code`重定向状态，重定向到`expected_url`（包括任何`GET`数据），并且最终页面以`target_status_code`接收到。

+   `assertHTMLEqual`：断言字符串`html1`和`html2`相等。比较基于 HTML 语义。比较考虑以下内容：

+   HTML 标签前后的空白会被忽略。

+   所有类型的空白都被视为等效。

+   所有未关闭的标签都会被隐式关闭，例如，当周围的标签关闭或 HTML 文档结束时。

+   空标签等同于它们的自关闭版本。

+   HTML 元素的属性排序不重要。

+   没有参数的属性等同于名称和值相等的属性（请参阅示例）。

+   `assertHTMLNotEqual`：断言字符串`html1`和`html2`*不*相等。比较基于 HTML 语义。详情请参阅`assertHTMLEqual()`。

+   `assertXMLEqual`：断言字符串`xml1`和`xml2`相等。比较基于 XML 语义。与`assertHTMLEqual()`类似，比较是基于解析内容的，因此只考虑语义差异，而不考虑语法差异。

+   `assertXMLNotEqual`：断言字符串`xml1`和`xml2`*不*相等。比较基于 XML 语义。详情请参阅`assertXMLEqual()`。

+   `assertInHTML`：断言 HTML 片段`needle`包含在`haystack`中。

+   `assertJSONEqual`：断言 JSON 片段`raw`和`expected_data`相等。

+   `assertJSONNotEqual`：断言 JSON 片段`raw`和`expected_data`不相等。

+   `assertQuerysetEqual`：断言查询集`qs`返回特定的值列表`values`。使用`transform`函数执行`qs`和`values`的内容比较；默认情况下，这意味着比较每个值的`repr()`。

+   `assertNumQueries`：断言当使用`*args`和`**kwargs`调用`func`时，将执行`num`个数据库查询。

## 电子邮件服务

如果您的 Django 视图使用 Django 的电子邮件功能发送电子邮件，您可能不希望每次使用该视图运行测试时都发送电子邮件。因此，Django 的测试运行器会自动将所有 Django 发送的电子邮件重定向到一个虚拟的 outbox。这样，您可以测试发送电子邮件的每个方面，从发送的消息数量到每个消息的内容，而无需实际发送消息。测试运行器通过透明地将正常的电子邮件后端替换为测试后端来实现这一点。（不用担心-这不会对 Django 之外的任何其他电子邮件发送者产生影响，比如您的机器邮件服务器，如果您正在运行的话。）

在测试运行期间，每封发送的电子邮件都会保存在`django.core.mail.outbox`中。这是所有已发送的`EmailMessage`实例的简单列表。`outbox`属性是仅在使用`locmem`电子邮件后端时才会创建的特殊属性。它通常不作为`django.core.mail`模块的一部分存在，也不能直接导入。以下代码显示了如何正确访问此属性。以下是一个检查`django.core.mail.outbox`长度和内容的示例测试：

```py
from django.core import mail 
from django.test import TestCase 

class EmailTest(TestCase): 
    def test_send_email(self): 
        # Send message. 
        mail.send_mail('Subject here', 'Here is the message.', 
'from@example.com', ['to@example.com'], 
            fail_silently=False) 

        # Test that one message has been sent. 
        self.assertEqual(len(mail.outbox), 1) 

        # Verify that the subject of the first message is correct. 
        self.assertEqual(mail.outbox[0].subject, 'Subject here') 

```

如前所述，在 Django 的`*TestCase`中，测试 outbox 在每个测试开始时都会被清空。要手动清空 outbox，请将空列表分配给`mail.outbox`：

```py
from django.core import mail 

# Empty the test outbox 
mail.outbox = [] 

```

## 管理命令

可以使用`call_command()`函数测试管理命令。输出可以重定向到`StringIO`实例中：

```py
from django.core.management import call_command 
from django.test import TestCase 
from django.utils.six import StringIO 

class ClosepollTest(TestCase): 
    def test_command_output(self): 
        out = StringIO() 
        call_command('closepoll', stdout=out) 
        self.assertIn('Expected output', out.getvalue()) 

```

## 跳过测试

`unittest`库提供了`@skipIf`和`@skipUnless`装饰器，允许您在预先知道这些测试在特定条件下会失败时跳过测试。例如，如果您的测试需要特定的可选库才能成功，您可以使用`@skipIf`装饰测试用例。然后，测试运行器将报告该测试未被执行以及原因，而不是失败测试或完全省略测试。

# 测试数据库

需要数据库的测试（即模型测试）不会使用生产数据库；测试时会为其创建单独的空白数据库。无论测试是否通过，测试数据库在所有测试执行完毕时都会被销毁。您可以通过在测试命令中添加`-keepdb`标志来阻止测试数据库被销毁。这将在运行之间保留测试数据库。

如果数据库不存在，将首先创建它。任何迁移也将被应用以保持数据库的最新状态。默认情况下，测试数据库的名称是在`DATABASES`中定义的数据库的`NAME`设置值前加上`test_`。在使用 SQLite 数据库引擎时，默认情况下测试将使用内存数据库（即，数据库将在内存中创建，完全绕过文件系统！）。

如果要使用不同的数据库名称，请在`DATABASES`中为任何给定数据库的`TEST`字典中指定`NAME`。在 PostgreSQL 上，`USER`还需要对内置的`postgres`数据库具有读取权限。除了使用单独的数据库外，测试运行器将使用与设置文件中相同的数据库设置：`ENGINE`、`USER`、`HOST`等。测试数据库由`USER`指定的用户创建，因此您需要确保给定的用户帐户具有在系统上创建新数据库的足够权限。

# 使用不同的测试框架

显然，`unittest`并不是唯一的 Python 测试框架。虽然 Django 不提供对替代框架的显式支持，但它提供了一种调用为替代框架构建的测试的方式，就像它们是普通的 Django 测试一样。

当您运行`./manage.py test`时，Django 会查看`TEST_RUNNER`设置以确定要执行的操作。默认情况下，`TEST_RUNNER`指向`django.test.runner.DiscoverRunner`。这个类定义了默认的 Django 测试行为。这种行为包括：

1.  执行全局的测试前设置。

1.  在当前目录中查找任何以下文件中的测试，其名称与模式`test*.py`匹配。

1.  创建测试数据库。

1.  运行迁移以将模型和初始数据安装到测试数据库中。

1.  运行找到的测试。

1.  销毁测试数据库。

1.  执行全局的测试后拆卸。

如果您定义自己的测试运行器类并将`TEST_RUNNER`指向该类，Django 将在运行`./manage.py test`时执行您的测试运行器。

通过这种方式，可以使用任何可以从 Python 代码执行的测试框架，或者修改 Django 测试执行过程以满足您可能有的任何测试要求。

请查看 Django 项目网站，了解更多关于使用不同测试框架的信息。

# 接下来呢？

现在您已经知道如何为您的 Django 项目编写测试，一旦您准备将项目变成一个真正的网站，我们将继续讨论一个非常重要的话题-将 Django 部署到 Web 服务器。
