# 第一章：Django 测试概述

您如何知道您编写的代码是否按预期工作？好吧，您测试它。但是如何测试？对于 Web 应用程序，您可以通过手动在 Web 浏览器中打开应用程序的页面并验证它们是否正确来测试代码。这不仅涉及快速浏览以查看它们是否具有正确的内容，还必须确保例如所有链接都有效，任何表单都能正常工作等。正如您可以想象的那样，这种手动测试很快就会在应用程序增长到几个简单页面以上时变得不可靠。对于任何非平凡的应用程序，自动化测试是必不可少的。

Django 应用程序的自动化测试利用了 Python 语言内置的基本测试支持：doctests 和单元测试。当您使用`manage.py startapp`创建一个新的 Django 应用程序时，生成的文件之一包含一个样本 doctest 和单元测试，旨在加速您自己的测试编写。在本章中，我们将开始学习测试 Django 应用程序。具体来说，我们将：

+   详细检查样本`tests.py`文件的内容，同时回顾 Python 测试支持的基本知识

+   查看如何使用 Django 实用程序来运行`tests.py`中包含的测试

+   学习如何解释测试的输出，无论测试成功还是失败

+   审查可以在测试时使用的各种命令行选项的影响

# 入门：创建一个新应用程序

让我们开始创建一个新的 Django 项目和应用程序。为了在整本书中有一致的工作内容，让我们假设我们打算创建一个新的市场调研类型的网站。在这一点上，我们不需要对这个网站做出太多决定，只需要为 Django 项目和至少一个将包含的应用程序取一些名称。由于`market_research`有点长，让我们将其缩短为`marketr`作为项目名称。我们可以使用`django-admin.py`来创建一个新的 Django 项目：

```py
kmt@lbox:/dj_projects$ django-admin.py startproject marketr

```

然后，从新的`marketr`目录中，我们可以使用`manage.py`实用程序创建一个新的 Django 应用程序。我们市场调研项目的核心应用程序之一将是一个调查应用程序，因此我们将从创建它开始：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py startapp survey

```

现在我们有了 Django 项目和应用程序的基本框架：`settings.py`文件，`urls.py`文件，`manage.py`实用程序，以及一个包含模型、视图和测试的`survey`目录。自动生成的模型和视图文件中没有实质性内容，但在`tests.py`文件中有两个样本测试：一个单元测试和一个 doctest。接下来我们将详细检查每个测试。

# 理解样本单元测试

单元测试是`tests.py`中包含的第一个测试，它开始于：

```py
""" 
This file demonstrates two different styles of tests (one doctest and one unittest). These will both pass when you run "manage.py test". 

Replace these with more appropriate tests for your application. 
"""

from django.test import TestCase 

class SimpleTest(TestCase): 
    def test_basic_addition(self): 
        """ 
        Tests that 1 + 1 always equals 2\. 
        """ 
        self.failUnlessEqual(1 + 1, 2) 
```

单元测试从`django.test`中导入`TestCase`开始。`django.test.TestCase`类基于 Python 的`unittest.TestCase`，因此它提供了来自基础 Python`unittest.TestCase`的一切，以及对测试 Django 应用程序有用的功能。这些对`unittest.TestCase`的 Django 扩展将在第三章和第四章中详细介绍。这里的样本单元测试实际上并不需要任何支持，但是将样本测试用例基于 Django 类也没有坏处。

然后，样本单元测试声明了一个基于 Django 的`TestCase`的`SimpleTest`类，并在该类中定义了一个名为`test_basic_addition`的测试方法。该方法包含一条语句：

```py
self.failUnlessEqual(1 + 1, 2)
```

正如你所期望的那样，该语句将导致测试用例报告失败，除非两个提供的参数相等。按照编码的方式，我们期望该测试会成功。我们将在本章稍后验证这一点，当我们实际运行测试时。但首先，让我们更仔细地看一下示例 doctest。

# 理解示例 doctest

示例`tests.py`的 doctest 部分是：

```py
__test__ = {"doctest": """
Another way to test that 1 + 1 is equal to 2.

>>> 1 + 1 == 2
True
"""}
```

这看起来比单元测试部分更神秘。对于示例 doctest，声明了一个特殊变量`__test__`。这个变量被设置为包含一个键`doctest`的字典。这个键被设置为一个类似于包含注释后面的字符串值的 docstring，后面跟着一个看起来像是交互式 Python shell 会话的片段。

看起来像交互式 Python shell 会话的部分就是 doctest 的组成部分。也就是说，以`>>>`开头的行将在测试期间执行（减去`>>>`前缀），并且实际产生的输出将与 doctest 中以`>>>`开头的行下面找到的预期输出进行比较。如果任何实际输出与预期输出不匹配，则测试失败。对于这个示例测试，我们期望在交互式 Python shell 会话中输入`1 + 1 == 2`会导致解释器产生输出`True`，所以看起来这个示例测试应该通过。

请注意，doctests 不必通过使用特殊的`__test__`字典来定义。实际上，Python 的 doctest 测试运行器会查找文件中所有文档字符串中的 doctests。在 Python 中，文档字符串是模块、函数、类或方法定义中的第一条语句。鉴于此，你会期望在`tests.py`文件顶部的注释中找到的交互式 Python shell 会话片段也会作为 doctest 运行。这是我们开始运行这些测试后可以尝试的另一件事情。

# 运行示例测试

示例`tests.py`文件顶部的注释说明了两个测试：`当你运行"manage.py test"时都会通过`。所以让我们看看如果我们尝试那样会发生什么：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test 
Creating test database... 
Traceback (most recent call last): 
 File "manage.py", line 11, in <module> 
 execute_manager(settings) 
 File "/usr/lib/python2.5/site-packages/django/core/management/__init__.py", line 362, in execute_manager 
 utility.execute() 
 File "/usr/lib/python2.5/site-packages/django/core/management/__init__.py", line 303, in execute 
 self.fetch_command(subcommand).run_from_argv(self.argv) 
 File "/usr/lib/python2.5/site-packages/django/core/management/base.py", line 195, in run_from_argv 
 self.execute(*args, **options.__dict__) 
 File "/usr/lib/python2.5/site-packages/django/core/management/base.py", line 222, in execute 
 output = self.handle(*args, **options) 
 File "/usr/lib/python2.5/site-packages/django/core/management/commands/test.py", line 23, in handle 
 failures = test_runner(test_labels, verbosity=verbosity, interactive=interactive) 
 File "/usr/lib/python2.5/site-packages/django/test/simple.py", line 191, in run_tests 
 connection.creation.create_test_db(verbosity, autoclobber=not interactive) 
 File "/usr/lib/python2.5/site-packages/django/db/backends/creation.py", line 327, in create_test_db 
 test_database_name = self._create_test_db(verbosity, autoclobber) 
 File "/usr/lib/python2.5/site-packages/django/db/backends/creation.py", line 363, in _create_test_db 
 cursor = self.connection.cursor() 
 File "/usr/lib/python2.5/site-packages/django/db/backends/dummy/base.py", line 15, in complain 
 raise ImproperlyConfigured, "You haven't set the DATABASE_ENGINE setting yet." 
django.core.exceptions.ImproperlyConfigured: You haven't set the DATABASE_ENGINE setting yet.

```

哎呀，我们似乎有点超前了。我们创建了新的 Django 项目和应用程序，但从未编辑设置文件以指定任何数据库信息。显然，我们需要这样做才能运行测试。

但测试是否会使用我们在`settings.py`中指定的生产数据库？这可能令人担忧，因为我们可能在某个时候在我们的测试中编写了一些我们不希望对我们的生产数据执行的操作。幸运的是，这不是问题。Django 测试运行器为运行测试创建了一个全新的数据库，使用它来运行测试，并在测试运行结束时删除它。这个数据库的名称是`test_`后跟`settings.py`中指定的`DATABASE_NAME`。因此，运行测试不会干扰生产数据。

为了运行示例`tests.py`文件，我们需要首先为`DATABASE_ENGINE`、`DATABASE_NAME`和`settings.py`中使用的数据库所需的其他任何内容设置适当的值。现在也是一个好时机将我们的`survey`应用程序和`django.contrib.admin`添加到`INSTALLED_APPS`中，因为我们在继续进行时会需要这两个。一旦这些更改已经在`settings.py`中进行了，`manage.py test`就能更好地工作：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test 
Creating test database... 
Creating table auth_permission 
Creating table auth_group 
Creating table auth_user 
Creating table auth_message 
Creating table django_content_type 
Creating table django_session 
Creating table django_site 
Creating table django_admin_log 
Installing index for auth.Permission model 
Installing index for auth.Message model 
Installing index for admin.LogEntry model 
................................... 
---------------------------------------------------------------------- 
Ran 35 tests in 2.012s 

OK 
Destroying test database...

```

看起来不错。但到底测试了什么？在最后，它说`Ran 35 tests`，所以肯定运行了比我们简单的`tests.py`文件中的两个测试更多的测试。其他 33 个测试来自`settings.py`中默认列出的其他应用程序：auth、content types、sessions 和 sites。这些 Django“contrib”应用程序附带了它们自己的测试，并且默认情况下，`manage.py test`会运行`INSTALLED_APPS`中列出的所有应用程序的测试。

### 注意

请注意，如果您没有将`django.contrib.admin`添加到`settings.py`中的`INSTALLED_APPS`列表中，则`manage.py test`可能会报告一些测试失败。对于 Django 1.1，`django.contrib.auth`的一些测试依赖于`django.contrib.admin`也包含在`INSTALLED_APPS`中，以便测试通过。这种相互依赖关系可能会在将来得到修复，但是现在最简单的方法是从一开始就将`django.contrib.admin`包含在`INTALLED_APPS`中，以避免可能的错误。无论如何，我们很快就会想要使用它。

可以仅运行特定应用程序的测试。要做到这一点，在命令行上指定应用程序名称。例如，仅运行`survey`应用程序的测试：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test survey 
Creating test database... 
Creating table auth_permission 
Creating table auth_group 
Creating table auth_user 
Creating table auth_message 
Creating table django_content_type 
Creating table django_session 
Creating table django_site 
Creating table django_admin_log 
Installing index for auth.Permission model 
Installing index for auth.Message model 
Installing index for admin.LogEntry model 
.. 
---------------------------------------------------------------------- 
Ran 2 tests in 0.039s 

OK 
Destroying test database... 

```

在这里——`Ran 2 tests`看起来适合我们的样本`tests.py`文件。但是关于创建表和安装索引的所有这些消息呢？为什么这些应用程序的表在不进行测试时被创建？这是因为测试运行程序不知道将要测试的应用程序与`INSTALLED_APPS`中列出的其他不打算进行测试的应用程序之间可能存在的依赖关系。

例如，我们的调查应用程序可能具有一个模型，其中包含对`django.contrib.auth User`模型的`ForeignKey`，并且调查应用程序的测试可能依赖于能够添加和查询`User`条目。如果测试运行程序忽略了对不进行测试的应用程序创建表，这将无法工作。因此，测试运行程序为`INSTALLED_APPS`中列出的所有应用程序创建表，即使不打算运行测试的应用程序也是如此。

我们现在知道如何运行测试，如何将测试限制在我们感兴趣的应用程序上，以及成功的测试运行是什么样子。但是，测试失败呢？在实际工作中，我们可能会遇到相当多的失败，因此确保我们了解测试输出在发生时的情况是很重要的。因此，在下一节中，我们将引入一些故意的破坏，以便我们可以探索失败的样子，并确保当我们遇到真正的失败时，我们将知道如何正确解释测试运行的报告。

# 故意破坏事物

让我们首先引入一个单一的简单失败。更改单元测试，期望将`1 + 1`加上`3`而不是`2`。也就是说，更改单元测试中的单个语句为：`self.failUnlessEqual(1 + 1, 3)`。

现在当我们运行测试时，我们会得到一个失败：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test
Creating test database... 
Creating table auth_permission 
Creating table auth_group 
Creating table auth_user 
Creating table auth_message 
Creating table django_content_type 
Creating table django_session 
Creating table django_site 
Creating table django_admin_log 
Installing index for auth.Permission model
Installing index for auth.Message model 
Installing index for admin.LogEntry model 
...........................F.......
====================================================================== 
FAIL: test_basic_addition (survey.tests.SimpleTest) 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/dj_projects/marketr/survey/tests.py", line 15, in test_basic_addition 
 self.failUnlessEqual(1 + 1, 3) 
AssertionError: 2 != 3 

---------------------------------------------------------------------- 
Ran 35 tests in 2.759s 

FAILED (failures=1) 
Destroying test database...

```

看起来相当简单。失败产生了一块以等号开头的输出，然后是失败的测试的具体内容。失败的方法被识别出来，以及包含它的类。有一个`Traceback`显示了生成失败的确切代码行，`AssertionError`显示了失败原因的细节。

注意等号上面的那一行——它包含一堆点和一个`F`。这是什么意思？这是我们在早期测试输出列表中忽略的一行。如果你现在回去看一下，你会发现在最后一个`Installing index`消息之后一直有一行点的数量。这行是在运行测试时生成的，打印的内容取决于测试结果。`F`表示测试失败，点表示测试通过。当有足够多的测试需要一段时间来运行时，这种实时进度更新可以帮助我们在运行过程中了解运行的情况。

最后，在测试输出的末尾，我们看到`FAILED (failures=1)`而不是之前看到的`OK`。任何测试失败都会使整体测试运行的结果变成失败，而不是成功。

接下来，让我们看看一个失败的 doctest 是什么样子。如果我们将单元测试恢复到其原始形式，并将 doctest 更改为期望 Python 解释器对`1 + 1 == 3`作出`True`的回应，那么运行测试（这次只限制在`survey`应用程序中进行测试）将产生以下输出：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test survey 
Creating test database... 
Creating table auth_permission 
Creating table auth_group 
Creating table auth_user 
Creating table auth_message 
Creating table django_content_type 
Creating table django_session 
Creating table django_site 
Creating table django_admin_log 
Installing index for auth.Permission model 
Installing index for auth.Message model 
Installing index for admin.LogEntry model 
.F 
====================================================================== 
FAIL: Doctest: survey.tests.__test__.doctest 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 2180, in runTest 
 raise self.failureException(self.format_failure(new.getvalue())) 
AssertionError: Failed doctest test for survey.tests.__test__.doctest 
 File "/dj_projects/marketr/survey/tests.py", line unknown line number, in doctest 

---------------------------------------------------------------------- 
File "/dj_projects/marketr/survey/tests.py", line ?, in survey.tests.__test__.doctest 
Failed example: 
 1 + 1 == 3 
Expected: 
 True 
Got: 
 False 

---------------------------------------------------------------------- 
Ran 2 tests in 0.054s 

FAILED (failures=1) 
Destroying test database... 

```

失败的 doctest 的输出比单元测试失败的输出稍微冗长，解释起来也没有那么直接。失败的 doctest 被标识为`survey.tests.__test__.doctest`——这意味着在`survey/tests.py`文件中定义的`__test__`字典中的`doctest`键。输出的`Traceback`部分不像在单元测试案例中那样有用，因为`AssertionError`只是指出 doctest 失败了。幸运的是，随后提供了导致失败的原因的详细信息，您可以看到导致失败的行的内容，期望的输出以及执行失败行产生的实际输出。

请注意，测试运行器没有准确定位`tests.py`中发生失败的行号。它报告了不同部分的`未知行号`和`第?行`。这是 doctest 的一般问题还是这个特定 doctest 的定义方式的结果，作为`__test__`字典的一部分？我们可以通过在`tests.py`顶部的文档字符串中放置一个测试来回答这个问题。让我们将示例 doctest 恢复到其原始状态，并将文件顶部更改为如下所示：

```py
""" 
This file demonstrates two different styles of tests (one doctest and one unittest). These will both pass when you run "manage.py test". 

Replace these with more appropriate tests for your application. 

>>> 1 + 1 == 3 
True
""" 
```

然后当我们运行测试时，我们得到：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test survey 
Creating test database... 
Creating table auth_permission 
Creating table auth_group 
Creating table auth_user 
Creating table auth_message 
Creating table django_content_type 
Creating table django_session 
Creating table django_site 
Creating table django_admin_log 
Installing index for auth.Permission model 
Installing index for auth.Message model 
Installing index for admin.LogEntry model 
.F. 
====================================================================== 
FAIL: Doctest: survey.tests 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 2180, in runTest 
 raise self.failureException(self.format_failure(new.getvalue())) 
AssertionError: Failed doctest test for survey.tests 
 File "/dj_projects/marketr/survey/tests.py", line 0, in tests 

---------------------------------------------------------------------- 
File "/dj_projects/marketr/survey/tests.py", line 7, in survey.tests 
Failed example: 
 1 + 1 == 3 
Expected: 
 True 
Got: 
 False 

---------------------------------------------------------------------- 
Ran 3 tests in 0.052s 

FAILED (failures=1) 
Destroying test database... 

```

这里提供了行号。`Traceback`部分显然标识了包含失败测试行的文档字符串开始的行的上面一行（文档字符串从`第 1 行`开始，而回溯报告`第 0 行`）。详细的失败输出标识了导致失败的文件中的实际行，本例中为`第 7 行`。

无法准确定位行号因此是在`__test__`字典中定义 doctest 的副作用。虽然在我们简单的测试中很容易看出哪一行导致了问题，但在编写更实质性的 doctest 放置在`__test__`字典中时，这是需要牢记的事情。如果测试中的多行是相同的，并且其中一行导致失败，可能很难确定导致问题的确切行号，因为失败输出不会标识发生失败的具体行号。

到目前为止，我们在样本测试中引入的所有错误都涉及预期输出与实际结果不匹配。这些被报告为测试失败。除了测试失败，有时我们可能会遇到测试错误。接下来描述这些。

# 测试错误与测试失败

看看测试错误是什么样子，让我们删除上一节介绍的失败的 doctest，并在我们的样本单元测试中引入一种不同类型的错误。假设我们想要测试`1 + 1`是否等于文字`2`，而是想要测试它是否等于一个函数`sum_args`的结果，该函数应该返回其参数的总和。但我们会犯一个错误，忘记导入该函数。所以将`self.failUnlessEqual`改为：

```py
self.failUnlessEqual(1 + 1, sum_args(1, 1))
```

现在当运行测试时，我们看到：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test survey 
Creating test database... 
Creating table auth_permission 
Creating table auth_group 
Creating table auth_user 
Creating table auth_message 
Creating table django_content_type 
Creating table django_session 
Creating table django_site 
Creating table django_admin_log 
Installing index for auth.Permission model 
Installing index for auth.Message model 
Installing index for admin.LogEntry model 
E. 
====================================================================== 
ERROR: test_basic_addition (survey.tests.SimpleTest) 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/dj_projects/marketr/survey/tests.py", line 15, in test_basic_addition 
 self.failUnlessEqual(1 + 1, sum_args(1, 1)) 
NameError: global name 'sum_args' is not defined 

---------------------------------------------------------------------- 
Ran 2 tests in 0.041s 

FAILED (errors=1) 
Destroying test database... 

```

测试运行器在甚至比较`1 + 1`和`sum_args`的返回值之前就遇到了异常，因为`sum_args`没有被导入。在这种情况下，错误在于测试本身，但如果`sum_args`中的代码引起问题，它仍然会被报告为错误，而不是失败。失败意味着实际结果与预期结果不匹配，而错误意味着在测试运行期间遇到了一些其他问题（异常）。错误可能暗示测试本身存在错误，但不一定必须意味着如此。

请注意，在 doctest 中发生的类似错误会报告为失败，而不是错误。例如，我们可以将 doctest 的`1 + 1`行更改为：

```py
>>> 1 + 1 == sum_args(1, 1) 
```

然后运行测试，输出将是：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test survey 
Creating test database... 
Creating table auth_permission 
Creating table auth_group 
Creating table auth_user 
Creating table auth_message 
Creating table django_content_type 
Creating table django_session 
Creating table django_site 
Creating table django_admin_log 
Installing index for auth.Permission model 
Installing index for auth.Message model 
Installing index for admin.LogEntry model 
EF 
====================================================================== 
ERROR: test_basic_addition (survey.tests.SimpleTest) 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/dj_projects/marketr/survey/tests.py", line 15, in test_basic_addition 
 self.failUnlessEqual(1 + 1, sum_args(1, 1)) 
NameError: global name 'sum_args' is not defined 

====================================================================== 
FAIL: Doctest: survey.tests.__test__.doctest 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 2180, in runTest 
 raise self.failureException(self.format_failure(new.getvalue())) 
AssertionError: Failed doctest test for survey.tests.__test__.doctest 
 File "/dj_projects/marketr/survey/tests.py", line unknown line number, in doctest 

---------------------------------------------------------------------- 
File "/dj_projects/marketr/survey/tests.py", line ?, in survey.tests.__test__.doctest 
Failed example: 
 1 + 1 == sum_args(1, 1) 
Exception raised: 
 Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 1267, in __run 
 compileflags, 1) in test.globs 
 File "<doctest survey.tests.__test__.doctest[0]>", line 1, in <module> 
 1 + 1 == sum_args(1, 1) 
 NameError: name 'sum_args' is not defined 

---------------------------------------------------------------------- 
Ran 2 tests in 0.044s 

FAILED (failures=1, errors=1) 
Destroying test database... 

```

因此，对于单元测试所做的错误与失败的区分并不一定适用于 doctests。因此，如果您的测试包括 doctests，则在最后打印的失败和错误计数摘要并不一定反映出产生意外结果的测试数量（单元测试失败计数）或出现其他错误的测试数量（单元测试错误计数）。但是，在任何情况下，都不希望出现失败或错误。最终目标是两者都为零，因此如果它们之间的差异有时有点模糊，那也没什么大不了的。不过，了解在什么情况下报告一个而不是另一个可能是有用的。

我们现在已经了解了如何运行测试，以及整体成功和一些失败和错误的结果是什么样子。接下来，我们将研究`manage.py test`命令支持的各种命令行选项。

# 运行测试的命令行选项

除了在命令行上指定要测试的确切应用程序之外，还有哪些控制`manage.py` test 行为的选项？找出的最简单方法是尝试使用`--help`选项运行命令：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test --help
Usage: manage.py test [options] [appname ...]

Runs the test suite for the specified applications, or the entire site if no apps are specified.

Options:
 -v VERBOSITY, --verbosity=VERBOSITY
 Verbosity level; 0=minimal output, 1=normal output,
 2=all output
 --settings=SETTINGS   The Python path to a settings module, e.g.
 "myproject.settings.main". If this isn't provided, the
 DJANGO_SETTINGS_MODULE environment variable will 
 be used.
 --pythonpath=PYTHONPATH
 A directory to add to the Python path, e.g.
 "/home/djangoprojects/myproject".
 --traceback           Print traceback on exception
 --noinput             Tells Django to NOT prompt the user for input of 
 any kind.
 --version             show program's version number and exit
 -h, --help            show this help message and exit

```

让我们依次考虑每个（除了`help`，因为我们已经看到它的作用）：

## 冗长度

冗长度是一个介于`0`和`2`之间的数字值。它控制测试产生多少输出。默认值为`1`，因此到目前为止我们看到的输出对应于指定`-v 1`或`--verbosity=1`。将冗长度设置为`0`会抑制有关创建测试数据库和表的所有消息，但不包括摘要、失败或错误信息。如果我们纠正上一节引入的最后一个 doctest 失败，并重新运行指定`-v0`的测试，我们将看到：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test survey -v0 
====================================================================== 
ERROR: test_basic_addition (survey.tests.SimpleTest) 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/dj_projects/marketr/survey/tests.py", line 15, in test_basic_addition 
 self.failUnlessEqual(1 + 1, sum_args(1, 1)) 
NameError: global name 'sum_args' is not defined 

---------------------------------------------------------------------- 
Ran 2 tests in 0.008s 

FAILED (errors=1) 

```

将冗长度设置为`2`会产生更多的输出。如果我们修复这个剩下的错误，并将冗长度设置为最高级别运行测试，我们将看到：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py test survey --verbosity=2 
Creating test database... 
Processing auth.Permission model 
Creating table auth_permission 
Processing auth.Group model 
Creating table auth_group 
 **[...more snipped...]**

**Creating many-to-many tables for auth.Group model** 
**Creating many-to-many tables for auth.User model** 
**Running post-sync handlers for application auth** 
**Adding permission 'auth | permission | Can add permission'** 
**Adding permission 'auth | permission | Can change permission'** 
 ****[...more snipped...]**

**No custom SQL for auth.Permission model** 
**No custom SQL for auth.Group model** 

**[...more snipped...]**
 ****Installing index for auth.Permission model** 
**Installing index for auth.Message model** 
**Installing index for admin.LogEntry model** 
**Loading 'initial_data' fixtures...** 
**Checking '/usr/lib/python2.5/site-packages/django/contrib/auth/fixtures' for fixtures...** 
**Trying '/usr/lib/python2.5/site-packages/django/contrib/auth/fixtures' for initial_data.xml fixture 'initial_data'...** 
**No xml fixture 'initial_data' in '/usr/lib/python2.5/site-packages/django/contrib/auth/fixtures'.** 

**[....much more snipped...]**
**No fixtures found.** 
**test_basic_addition (survey.tests.SimpleTest) ... ok** 
**Doctest: survey.tests.__test__.doctest ... ok** 

**----------------------------------------------------------------------** 
**Ran 2 tests in 0.004s** 

**OK** 
**Destroying test database...****** 
```

正如您所看到的，以这种详细程度，该命令报告了设置测试数据库所做的一切细节。除了我们之前看到的创建数据库表和索引之外，我们现在看到数据库设置阶段包括：

1.  运行`post-syncdb`信号处理程序。例如，`django.contrib.auth`应用程序使用此信号在安装每个应用程序时自动添加模型的权限。因此，您会看到有关在为`INSTALLED_APPS`中列出的每个应用程序发送`post-syncdb`信号时创建权限的消息。

1.  为数据库中已创建的每个模型运行自定义 SQL。根据输出，似乎`INSTALLED_APPS`中的任何应用程序都没有使用自定义 SQL。

1.  加载`initial_data` fixtures。初始数据 fixtures 是一种自动预先填充数据库的常量数据的方法。我们在`INSTALLED_APPS`中列出的任何应用程序都没有使用此功能，但是测试运行程序会产生大量输出，因为它寻找初始数据 fixtures，这些 fixtures 可以在几种不同的名称下找到。对于每个被检查的可能文件以及是否找到任何内容，都会有消息。如果测试运行程序找到初始数据 fixtures 时遇到问题，这些输出可能会在某个时候派上用场（我们将在第三章中详细介绍 fixtures），但是目前这些输出并不是很有趣。

****一旦测试运行程序完成初始化数据库，它就会开始运行测试。在`2`的冗长级别下，我们之前看到的点、Fs 和 Es 的行会被每个测试的更详细的报告所取代。测试的名称被打印出来，然后是三个点，然后是测试结果，可能是`ok`、`ERROR`或`FAIL`。如果有任何错误或失败，它们发生的详细信息将在测试运行结束时打印出来。因此，当您观看冗长的测试运行时，设置冗长级别为`2`，您将能够看到哪些测试遇到了问题，但直到运行完成，您才能得到它们发生原因的详细信息。

## ****设置****

****您可以将设置选项传递给`test`命令，以指定要使用的设置文件，而不是项目默认的设置文件。例如，如果要使用与通常使用的数据库不同的数据库运行测试（无论是为了加快测试速度还是验证代码在不同数据库上是否正确运行），则可以派上用场。

****请注意，此选项的帮助文本说明`DJANGO_SETTINGS_MODULE`环境变量将用于定位设置文件，如果未在命令行上指定设置选项。当使用`django-admin.py`实用程序运行`test`命令时，这才是准确的。当使用`manage.py test`时，`manage.py`实用程序负责设置此环境变量以指定当前目录中的`settings.py`文件。

## ****Pythonpath****

****此选项允许您在测试运行期间将附加目录追加到 Python 路径中。当使用`django-admin.py`时，通常需要将项目路径添加到标准 Python 路径中。`manage.py`实用程序负责将项目路径添加到 Python 路径中，因此在使用`manage.py test`时通常不需要此选项。

## ****Traceback****

****实际上，`test`命令并不使用此选项。它作为所有`django-admin.py`（和`manage.py`）命令支持的默认选项之一而被继承，但`test`命令从不检查它。因此，您可以指定它，但它不会产生任何效果。

## ****Noinput****

****此选项导致测试运行程序不会提示用户输入，这引发了一个问题：测试运行程序何时需要用户输入？到目前为止，我们还没有遇到过。测试运行程序在测试数据库创建期间会提示用户输入，如果测试数据库名称已经存在。例如，如果在测试运行期间按下*Ctrl* + *C*，则测试数据库可能不会被销毁，下次尝试运行测试时可能会遇到类似以下消息：

```py
****kmt@lbox:/dj_projects/marketr$ python manage.py test** 
**Creating test database...** 
**Got an error creating the test database: (1007, "Can't create database 'test_marketr'; database exists")** 
**Type 'yes' if you would like to try deleting the test database 'test_marketr', or 'no' to cancel:**** 
```

****如果在命令行上传递了`--noinput`，则不会打印提示，并且测试运行程序将继续进行，就好像用户已经输入了'yes'一样。如果要从无人值守脚本运行测试，并确保脚本不会在等待永远不会输入的用户输入时挂起，这将非常有用。

## ****版本****

此选项报告正在使用的 Django 版本，然后退出。因此，当使用`--version`与`manage.py`或`django-admin.py`一起使用时，实际上不需要指定`test`等子命令。实际上，由于 Django 处理命令选项的方式存在错误，在撰写本书时，如果同时指定`--version`和子命令，版本将被打印两次。这可能会在某个时候得到修复。

****# 摘要

Django 测试的概述现在已经完成。在本章中，我们：

+   详细查看了在创建新的 Django 应用程序时生成的样本`tests.py`文件

+   学习如何运行提供的样本测试

+   尝试在测试中引入故意的错误，以查看和理解测试失败或遇到错误时提供的信息

+   最后，我们检查了所有可能与`manage.py test`一起使用的命令行选项。

我们将在下一章继续建立这些知识，重点关注深入的 doctests。
