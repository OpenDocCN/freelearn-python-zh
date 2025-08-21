# 第二章：这段代码有效吗？深入了解文档测试

在第一章中，我们学习了如何运行`manage.py startapp`创建的示例测试。虽然我们使用了 Django 实用程序来运行测试，但是示例测试本身与 Django 无关。在本章中，我们将开始详细介绍如何为 Django 应用程序编写测试。我们将：

+   通过开发一些基本模型来开始编写第一章创建的市场调研项目

+   尝试向其中一个模型添加文档测试

+   开始学习哪些测试是有用的，哪些只会给代码增加混乱

+   发现文档测试的一些优缺点

上一章提到了文档测试和单元测试，而本章的重点将专门放在文档测试上。开发 Django 应用程序的单元测试将是第三章和第四章的重点。

# 调查应用程序模型

开始开发新的 Django 应用程序的常见地方是从模型开始：这些数据的基本构建块将由应用程序进行操作和存储。我们示例市场调研`survey`应用程序的基石模型将是`Survey`模型。

`Survey`将类似于 Django 教程`Poll`模型，只是：

+   教程`Poll`只包含一个问题，而`Survey`将有多个问题。

+   `Survey`将有一个标题用于参考目的。对于教程`Poll`，可以使用一个单一的问题。

+   `Survey`只会在有限的时间内（取决于`Survey`实例）开放回应。虽然`Poll`模型有一个`pub_date`字段，但它除了在索引页面上对`Polls`进行排序之外没有用。因此，`Survey`将需要两个日期字段，而`Poll`只有一个，`Survey`的日期字段将比`Poll pub_date`字段更常用。

只需这些简单的要求，我们就可以开始为`Survey`开发 Django 模型。具体来说，我们可以通过将以下内容添加到我们`survey`应用程序的自动生成的`models.py`文件中的代码来捕捉这些要求：

```py
class Survey(models.Model): 
    title = models.CharField(max_length=60) 
    opens = models.DateField() 
    closes = models.DateField() 
```

请注意，由于`Survey`可能有多个问题，它没有一个问题字段。相反，有一个单独的模型`Question`，用于保存与其相关的调查实例的问题：

```py
class Question(models.Model): 
    question = models.CharField(max_length=200) 
    survey = models.ForeignKey(Survey) 
```

我们需要的最终模型（至少是开始时）是一个用于保存每个问题的可能答案，并跟踪调查受访者选择每个答案的次数。这个模型`Answer`与教程`Choice`模型非常相似，只是它与`Question`相关联，而不是与`Poll`相关联：

```py
class Answer(models.Model): 
    answer = models.CharField(max_length=200) 
    question = models.ForeignKey(Question) 
    votes = models.IntegerField(default=0) 
```

# 测试调查模型

如果你和我一样，在这一点上你可能想要开始验证到目前为止是否正确。的确，现在还没有太多的代码，但特别是在项目刚开始的时候，我喜欢确保我到目前为止的东西是有效的。那么，我们如何开始测试？首先，我们可以通过运行`manage.py syncdb`来验证我们没有语法错误，这也会让我们在 Python shell 中开始尝试这些模型。让我们来做吧。由于这是我们为这个项目第一次运行`syncdb`，我们将收到关于为`INSTALLED_APPS`中列出的其他应用程序创建表的消息，并且我们将被问及是否要创建超级用户，我们也可以继续做。

## 测试调查模型创建

现在，我们可以用这些模型做些什么来在 Python shell 中测试它们？实际上，除了创建每个模型之外，我们并没有太多可做的事情，也许可以验证一下，如果我们没有指定其中一个字段，我们会得到一个错误，或者正确的默认值被分配，并验证我们是否可以遍历模型之间的关系。如果我们首先关注`Survey`模型以及为了测试其创建而可能做的事情，那么 Python shell 会话可能看起来像这样：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py shell 
Python 2.5.2 (r252:60911, Oct  5 2008, 19:24:49) 
[GCC 4.3.2] on linux2 
Type "help", "copyright", "credits" or "license" for more information. 
(InteractiveConsole) 
>>> from survey.models import Survey 
>>> import datetime 
>>> t = 'First!'
>>> d = datetime.date.today()
>>> s = Survey.objects.create(title=t, opens=d, closes=d) 
>>>

```

在这里，我们首先导入了我们的`Survey`模型和 Python 的`datetime`模块，然后创建了一个变量`t`来保存一个标题字符串和一个变量`d`来保存一个日期值，并使用这些值创建了一个`Survey`实例。没有报告错误，所以看起来很好。

如果我们想验证一下，如果我们尝试创建一个没有关闭日期的`Survey`，我们会得到一个错误吗，我们将继续进行：

```py
>>> s = Survey.objects.create(title=t, opens=d, closes=None) 
 File "<console>", line 1, in <module> 
 File "/usr/lib/python2.5/site-packages/django/db/models/manager.py", line 126, in create 
 return self.get_query_set().create(**kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 315, in create 
 obj.save(force_insert=True) 
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 410, in save 
 self.save_base(force_insert=force_insert, force_update=force_update) 
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 495, in save_base 
 result = manager._insert(values, return_id=update_pk) 
 File "/usr/lib/python2.5/site-packages/django/db/models/manager.py", line 177, in _insert 
 return insert_query(self.model, values, **kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 1087, in insert_query 
 return query.execute_sql(return_id) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/subqueries.py", line 320, in execute_sql 
 cursor = super(InsertQuery, self).execute_sql(None) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/query.py", line 2369, in execute_sql 
 cursor.execute(sql, params) 
 File "/usr/lib/python2.5/site-packages/django/db/backends/util.py", line 19, in execute 
 return self.cursor.execute(sql, params) 
 File "/usr/lib/python2.5/site-packages/django/db/backends/sqlite3/base.py", line 193, in execute 
 return Database.Cursor.execute(self, query, params) 
IntegrityError: survey_survey.closes may not be NULL 

```

在这里，我们尝试创建`Survey`实例的唯一不同之处是为`closes`值指定了`None`，而不是传入我们的日期变量`d`。结果是一个以`IntegrityError`结尾的错误消息，因为调查表的关闭列不能为 null。这证实了我们对应该发生的预期，所以到目前为止一切都很好。然后我们可以对其他字段执行类似的测试，并看到相同的回溯报告了其他列的`IntegrityError`。

如果我们想的话，我们可以通过直接从 shell 会话中剪切和粘贴它们到我们的`survey/models.py`文件中，将这些测试变成我们模型定义的永久部分，就像这样：

```py
import datetime
from django.db import models 

class Survey(models.Model): 
    """ 
    >>> t = 'First!' 
    >>> d = datetime.date.today() 
    >>> s = Survey.objects.create(title=t, opens=d, closes=d) 
    >>> s = Survey.objects.create(title=t, opens=d, closes=None) 
    Traceback (most recent call last): 
    ... 
    IntegrityError: survey_survey.closes may not be NULL 
    >>> s = Survey.objects.create(title=t, opens=None, closes=d) 
    Traceback (most recent call last): 
    ... 
    IntegrityError: survey_survey.opens may not be NULL 
    >>> s = Survey.objects.create(title=None, opens=d, closes=d) 
    Traceback (most recent call last): 
    ... 
    IntegrityError: survey_survey.title may not be NULL 
    """ 
    title = models.CharField(max_length=60) 
    opens = models.DateField() 
    closes = models.DateField()
```

您可能已经注意到，所显示的结果并不是直接从 shell 会话中剪切和粘贴的。差异包括：

+   `import datetime`被移出了 doctest，并成为`models.py`文件中的代码的一部分。这并不是严格必要的——如果作为 doctest 的一部分，它也可以正常工作，但是如果导入在主代码中，那么在 doctest 中就不是必要的。由于`models.py`中的代码可能需要稍后使用`datetime`函数，因此现在将导入放在主代码中可以减少稍后的重复和混乱，当主代码需要导入时。

+   回溯的调用堆栈部分，也就是除了第一行和最后一行之外的所有内容，都被删除并替换为包含三个点的行。这也并不是严格必要的，只是为了去除杂乱，并突出结果的重要部分。doctest 运行器在决定测试成功或失败时会忽略调用堆栈的内容（如果预期输出中存在）。因此，如果调用堆栈具有一些解释价值，可以将其保留在测试中。然而，大部分情况下，最好删除调用堆栈，因为它们会产生大量杂乱，而提供的有用信息并不多。

如果我们现在运行`manage.py test survey -v2`，输出的最后部分将是：

```py
No fixtures found. 
test_basic_addition (survey.tests.SimpleTest) ... ok 
Doctest: survey.models.Survey ... ok 
Doctest: survey.tests.__test__.doctest ... ok 

---------------------------------------------------------------------- 
Ran 3 tests in 0.030s 

OK 
Destroying test database... 

```

我们仍然在`tests.py`中运行我们的样本测试，现在我们还可以看到我们的`survey.models.Survey` doctest 被列为正在运行并通过。

## 那个测试有用吗？

但等等；我们刚刚添加的测试有用吗？它实际上在测试什么？实际上并没有什么，除了验证基本的 Django 函数是否按照广告那样工作。它测试我们是否可以创建我们定义的模型的实例，并且我们在模型定义中指定为必需的字段实际上在关联的数据库表中是必需的。看起来这个测试更像是在测试 Django 的底层代码，而不是我们的应用程序。在我们的应用程序中测试 Django 本身并不是必要的：Django 有自己的测试套件，我们可以运行它进行测试（尽管可以相当安全地假设基本功能在任何发布版本的 Django 中都能正确工作）。

可以说，这个测试验证了模型中每个字段是否已经指定了正确和预期的选项，因此这是对应用程序而不仅仅是底层 Django 函数的测试。然而，测试那些通过检查就很明显的事情（对于任何具有基本 Django 知识的人来说）让我觉得有点过分。这不是我通常会在自己写的项目中包含的测试。

这并不是说我在开发过程中不会在 Python shell 中尝试类似的事情：我会的，而且我也会。但是在开发过程中在 shell 中尝试的并不是所有东西都需要成为应用程序中的永久测试。您想要包含在应用程序中的测试类型是那些对应用程序独特行为进行测试的测试。因此，让我们开始开发一些调查应用程序代码，并在 Python shell 中进行测试。当我们的代码工作正常时，我们可以评估哪些来自 shell 会话的测试是有用的。

## 开发自定义调查保存方法

要开始编写一些特定于应用程序的代码，请考虑对于调查模型，如果在创建模型实例时没有指定`closes`，我们可能希望允许`closes`字段假定默认值为`opens`后的一周。我们不能使用 Django 模型字段默认选项，因为我们想要分配的值取决于模型中的另一个字段。因此，我们通常会通过覆盖模型的保存方法来实现这一点。首次尝试实现这一点可能是：

```py
import datetime
from django.db import models  

class Survey(models.Model): 
    title = models.CharField(max_length=60) 
    opens = models.DateField() 
    closes = models.DateField() 

    def save(self, **kwargs): 
        if not self.pk and not self.closes: 
            self.closes = self.opens + datetime.timedelta(7) 
        super(Survey, self).save(**kwargs) 
```

也就是说，在调用`save`并且模型实例尚未分配主键（因此这是对数据库的第一次保存），并且没有指定`closes`的情况下，我们在调用超类`save`方法之前将`closes`赋予一个比`opens`晚一周的值。然后我们可以通过在 Python shell 中进行实验来测试这是否正常工作：

```py
kmt@lbox:/dj_projects/marketr$ python manage.py shell 
Python 2.5.2 (r252:60911, Oct  5 2008, 19:24:49) 
[GCC 4.3.2] on linux2 
Type "help", "copyright", "credits" or "license" for more information. 
(InteractiveConsole) 
>>> from survey.models import Survey 
>>> import datetime 
>>> t = "New Year's Resolutions" 
>>> sd = datetime.date(2009, 12, 28) 
>>> s = Survey.objects.create(title=t, opens=sd) 
>>> s.closes 
datetime.date(2010, 1, 4) 
>>> 

```

这与我们之前的测试非常相似，只是我们选择了一个特定的日期来分配给`opens`，而不是使用今天的日期，并且在创建`Survey`实例时没有指定`closes`的值，我们检查了分配给它的值。显示的值比`opens`晚一周，所以看起来很好。

请注意，故意选择`opens`日期，其中一周后的值将在下个月和年份是一个明智的选择。测试边界值总是一个好主意，也是一个好习惯，即使（就像这里一样）我们正在编写的代码中没有任何东西负责为边界情况得到正确的答案。

接下来，我们可能希望确保如果我们指定了`closes`的值，它会被尊重，而不会被默认的一周后的日期覆盖：

```py
>>> s = Survey.objects.create(title=t, opens=sd, closes=sd)
>>> s.opens 
datetime.date(2009, 12, 28) 
>>> s.closes 
datetime.date(2009, 12, 28) 
>>> 

```

所有看起来都很好，`opens`和`closes`显示为具有相同的值，就像我们在`create`调用中指定的那样。我们还可以验证，如果我们在模型已经保存后将`closes`重置为`None`，然后尝试再次保存，我们会得到一个错误。在现有模型实例上将`closes`重置为`None`将是代码中的错误。因此，我们在这里测试的是我们的`save`方法重写不会通过悄悄地重新分配一个值给`closes`来隐藏该错误。在我们的 shell 会话中，我们可以这样继续并查看：

```py
>>> s.closes = None 
>>> s.save() 
Traceback (most recent call last): 
 File "<console>", line 1, in <module> 
 File "/dj_projects/marketr/survey/models.py", line 12, in save 
 super(Survey, self).save(**kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 410, in save 
 self.save_base(force_insert=force_insert, force_update=force_update) 
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 474, in save_base 
 rows = manager.filter(pk=pk_val)._update(values) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 444, in _update 
 return query.execute_sql(None) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/subqueries.py", line 120, in execute_sql 
 cursor = super(UpdateQuery, self).execute_sql(result_type) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/query.py", line 2369, in execute_sql 
 cursor.execute(sql, params) 
 File "/usr/lib/python2.5/site-packages/django/db/backends/util.py", line 19, in execute 
 return self.cursor.execute(sql, params) 
 File "/usr/lib/python2.5/site-packages/django/db/backends/sqlite3/base.py", line 193, in execute 
 return Database.Cursor.execute(self, query, params) 
IntegrityError: survey_survey.closes may not be NULL 
>>> 

```

同样，这看起来很好，因为这是我们期望的结果。最后，由于我们已经将一些自己的代码插入到基本模型保存处理中，我们应该验证我们没有在`create`上没有指定`title`或`opens`字段的其他预期失败情况中出现问题。如果我们这样做，我们会发现没有指定`title`的情况下工作正常（我们在数据库标题列上得到了预期的`IntegrityError`），但如果`opens`和`closes`都没有指定，我们会得到一个意外的错误：

```py
>>> s = Survey.objects.create(title=t) 
Traceback (most recent call last): 
 File "<console>", line 1, in <module> 
 File "/usr/lib/python2.5/site-packages/django/db/models/manager.py", line 126, in create 
 return self.get_query_set().create(**kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 315, in create 
 obj.save(force_insert=True) 
 File "/dj_projects/marketr/survey/models.py", line 11, in save 
 self.closes = self.opens + datetime.timedelta(7) 
TypeError: unsupported operand type(s) for +: 'NoneType' and 'datetime.timedelta' 
>>> 

```

在这里，我们用一个相当晦涩的消息来报告我们留下了一个必需的值未指定的错误，而不是一个相当清晰的错误消息。问题是我们在尝试在`save`方法重写中使用`opens`之前没有检查它是否有值。为了获得这种情况下的正确（更清晰）错误，我们的`save`方法应该修改为如下所示：

```py
    def save(self, **kwargs): 
        if not self.pk and self.opens and not self.closes: 
            self.closes = self.opens + datetime.timedelta(7) 
        super(Survey, self).save(**kwargs) 
```

也就是说，如果`opens`没有被指定，我们不应该尝试设置`closes`。在这种情况下，我们直接将`save`调用转发到超类，并让正常的错误路径报告问题。然后，当我们尝试创建一个没有指定`opens`或`closes`值的`Survey`时，我们会看到：

```py
>>> s = Survey.objects.create(title=t) 
Traceback (most recent call last): 
 File "<console>", line 1, in <module> 
 File "/usr/lib/python2.5/site-packages/django/db/models/manager.py", line 126, in create 
 return self.get_query_set().create(**kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 315, in create 
 obj.save(force_insert=True) 
 File "/dj_projects/marketr/survey/models.py", line 12, in save 
 super(Survey, self).save(**kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 410, in save 
 self.save_base(force_insert=force_insert, force_update=force_update) 
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 495, in save_base 
 result = manager._insert(values, return_id=update_pk) 
 File "/usr/lib/python2.5/site-packages/django/db/models/manager.py", line 177, in _insert 
 return insert_query(self.model, values, **kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 1087, in insert_query 
 return query.execute_sql(return_id) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/subqueries.py", line 320, in execute_sql 
 cursor = super(InsertQuery, self).execute_sql(None) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/query.py", line 2369, in execute_sql 
 cursor.execute(sql, params) 
 File "/usr/lib/python2.5/site-packages/django/db/backends/util.py", line 19, in execute 
 return self.cursor.execute(sql, params) 
 File "/usr/lib/python2.5/site-packages/django/db/backends/sqlite3/base.py", line 193, in execute 
 return Database.Cursor.execute(self, query, params) 
IntegrityError: survey_survey.opens may not be NULL 
>>> 

```

这样会好得多，因为报告的错误直接指出了问题所在。

## 决定测试什么

在这一点上，我们相当确定我们的`save`重写正在按我们的意图工作。在我们为验证目的在 Python shell 中运行的所有测试中，哪些测试有意义地包含在代码中？这个问题的答案涉及判断，并且不同的人可能会有不同的答案。就我个人而言，我倾向于包括：

+   受代码直接影响的参数的所有测试

+   在对代码进行初始测试时遇到的任何测试，这些测试在我编写的原始代码版本中没有起作用

因此，我的`save`重写函数，包括带有注释的 doctests，可能看起来像这样：

```py
    def save(self, **kwargs): 
        """ 
        save override to allow for Survey instances to be created without explicitly specifying a closes date. If not specified, closes will be set to 7 days after opens. 
        >>> t = "New Year's Resolutions" 
        >>> sd = datetime.date(2009, 12, 28) 
        >>> s = Survey.objects.create(title=t, opens=sd) 
        >>> s.closes 
        datetime.date(2010, 1, 4) 

        If closes is specified, it will be honored and not auto-set. 

        >>> s = Survey.objects.create(title=t, opens=sd, closes=sd) 
        >>> s.closes 
        datetime.date(2009, 12, 28) 

        Any changes to closes after initial creation need to be explicit. Changing closes to None on an existing instance will not result in closes being reset to 7 days after opens. 

        >>> s.closes = None 
        >>> s.save() 
        Traceback (most recent call last): 
          ... 
        IntegrityError: survey_survey.closes may not be NULL 

        Making the mistake of specifying neither opens nor closes results in the expected IntegrityError for opens, not any exception in the code here. 

        >>> s = Survey.objects.create(title=t) 
        Traceback (most recent call last): 
          ... 
        IntegrityError: survey_survey.opens may not be NULL 
        """ 
        if not self.pk and self.opens and not self.closes: 
            self.closes = self.opens + datetime.timedelta(7) 
        super(Survey, self).save(**kwargs) 
```

## 到目前为止，doctests 的一些优缺点

即使只是通过研究这一个例子方法的经验，我们也可以开始看到 doctests 的一些优缺点。显然，可以很容易地重用在 Python shell 会话中完成的工作（这些工作很可能已经作为编码的一部分而被完成）用于永久测试目的。这使得更有可能为代码编写测试，并且测试本身不需要被调试。这是 doctests 的两个很好的优点。

第三个是 doctests 提供了代码预期行为的明确文档。散文描述可能模糊不清，而以测试形式的代码示例是不可能被误解的。此外，测试作为文档字符串的一部分，使它们可以被所有使用文档字符串自动生成帮助和文档的 Python 工具访问。

在这里包括测试有助于使文档完整。例如，将`closes`重置为`None`后的行为可能不明显，一个同样有效的设计是在`save`期间将`closes`重置为一周后的日期。在编写文档时很容易忽略这种细节。因此，在 doctest 中详细说明预期的行为是有帮助的，因为它会自动记录下来。

然而，这种测试兼作文档的特性也有一个缺点：您可能希望包括的一些测试实际上可能并不适合作为文档，并且您可能会得到一个对相当简单的代码而言文档过多的情况。考虑我们开发的`save`重写案例。它有四行代码和超过 30 行的文档字符串。这种比例对于一些具有许多参数或参数以非明显方式相互作用的复杂函数可能是合适的，但是对于这种简单的方法来说，文档比代码多近十倍似乎过多了。

让我们考虑`save`中的各个测试，重点是它们作为文档的有用性：

+   第一个测试显示了使用`title`和`opens`创建`Survey`，但没有`closes`，并验证了在创建后将正确值分配给`closes`，这是`save`重写允许调用者执行的示例。这是通过添加的代码启用的特定调用模式，并且因此作为文档是有用的，即使它在很大程度上重复了散文描述。

+   第二个测试显示了如果指定了`closes`，它将被遵守，这并不特别适合作为文档。任何程序员都会期望，如果指定了`closes`，它应该被遵守。这种行为可能适合测试，但不需要记录。

+   第三个测试展示了在现有的`Survey`实例上将`closes`重置为`None`后`save`的预期行为，出于前面提到的原因，这对于文档来说是有用的。

+   第四个和最后一个测试说明了添加的代码不会在未指定`opens`或`closes`的错误情况下引发意外异常。这是另一个需要测试但不需要记录的例子，因为正确的行为是显而易见的。

将我们的文档字符串的一半分类为不适合文档目的是不好的。当人们遇到明显的、冗余的或无用的信息时，他们往往会停止阅读。我们可以通过将这些测试从文档字符串方法移到我们的`tests.py`文件中来解决这个问题，而不放弃 doctests 的一些优势。如果我们采取这种方法，我们可能会改变`tests.py`中的`__test__`字典，使其看起来像这样：

```py
__test__ = {"survey_save": """ 

Tests for the Survey save override method. 

>>> import datetime 
>>> from survey.models import Survey 
>>> t = "New Year's Resolutions" 
>>> sd = datetime.date(2009, 12, 28) 

If closes is specified, it will be honored and not auto-set. 

>>> s = Survey.objects.create(title=t, opens=sd, closes=sd) 
>>> s.closes 
datetime.date(2009, 12, 28) 

Making the mistake of specifying neither opens nor closes results 
in the expected IntegrityError for opens, not any exception in the 
save override code itself. 

>>> s = Survey.objects.create(title=t) 
Traceback (most recent call last): 
  ... 
IntegrityError: survey_survey.opens may not be NULL 
"""} 
```

在这里，我们将测试的关键字从通用的`doctest`改为`survey_save`，这样任何测试输出中报告的测试名称都会给出被测试的提示。然后我们将“非文档”测试（以及现在需要在两个地方都设置的一些变量设置代码）从我们的`save`覆盖文档字符串中移到这里的键值中，并在顶部添加一般注释，说明测试的目的。

`save`方法本身的文档字符串中剩下的测试确实具有一定的文档价值：

```py
    def save(self, **kwargs): 
        """ 
        save override to allow for Survey instances to be created without explicitly specifying a closes date. If not specified, closes will be set to 7 days after opens. 
        >>> t = "New Year's Resolutions" 
        >>> sd = datetime.date(2009, 12, 28) 
        >>> s = Survey.objects.create(title=t, opens=sd) 
        >>> s.closes 
        datetime.date(2010, 1, 4) 

        Any changes to closes after initial creation need to be explicit. Changing closes to None on an existing instance will not result in closes being reset to 7 days after opens. 

        >>> s.closes = None 
        >>> s.save() 
        Traceback (most recent call last): 
          ... 
        IntegrityError: survey_survey.closes may not be NULL 

        """ 
        if not self.pk and self.opens and not self.closes: 
            self.closes = self.opens + datetime.timedelta(7) 
        super(Survey, self).save(**kwargs) 
```

这对于函数的文档字符串来说肯定更容易管理，不太可能会让在 Python shell 中键入`help(Survey.save)`的人感到不知所措。

这种方法也有其不利之处。代码的测试不再集中在一个地方，很难知道或轻松确定代码被完全测试了多少。如果有人在`tests.py`中遇到测试，却不知道方法的文档字符串中还有额外的测试，很可能会想知道为什么只测试了这两个边缘情况，为什么忽略了基本功能的直接测试。

此外，当添加测试时，可能不清楚（特别是对于新加入项目的程序员）新测试应该放在哪里。因此，即使项目一开始在文档字符串测试中有一个很好的清晰分割，“适合文档的测试”和“必要但不适合文档的测试”在`tests.py`文件中，随着时间的推移，这种区别可能很容易变得模糊。

因此，测试选择和放置涉及权衡。并不是每个项目都有“正确”的答案。然而，采用一致的方法是最好的。在选择这种方法时，每个项目团队都应该考虑诸如以下问题的答案：

+   **自动生成的基于文档字符串的文档的预期受众是谁？**

如果存在其他文档（或正在编写），预期它们将成为代码“使用者”的主要来源，那么具有不太好的文档功能的 doctests 可能并不是问题。

+   **可能会有多少人在代码上工作？**

如果人数相对较少且稳定，让每个人记住测试分散在两个地方可能不是什么大问题。对于一个较大的项目或者如果开发人员流动性较高，教育开发人员关于这种分割可能会成为更大的问题，而且可能更难维护一致的代码。

# 附加的 doctest 注意事项

Doctests 还有一些我们可能还没有遇到或注意到的额外缺点。其中一些只是我们需要注意的事项，如果我们想确保我们的 doctests 在各种环境中能正常工作，并且在我们的代码周围的代码发生变化时。其他更严重的问题最容易通过切换到单元测试而不是 doctests 来解决，至少对受影响的测试来说是这样。在本节中，我们将列出许多需要注意的额外 doctest 问题，并提供关于如何避免或克服这些问题的指导。

## 注意环境依赖

doctests 很容易无意中依赖于实际被测试的代码以外的代码的实现细节。我们在`save`覆盖测试中已经有了一些这样的情况，尽管我们还没有被这个问题绊倒。我们现在所面临的依赖实际上是一种非常特定的环境依赖——数据库依赖。由于数据库依赖本身就是一个相当大的问题，它将在下一节中详细讨论。然而，我们首先将介绍一些其他可能会遇到的次要环境依赖，并看看如何避免将它们包含在我们的测试中。

一种极其常见的环境依赖形式是依赖于对象的打印表示。例如，`__unicode__`方法是首先在模型类中实现的常见方法。它在之前的`Survey`模型讨论中被省略，因为那时并不需要，但实际上我们可能会在`save`覆盖之前实现`__unicode__`。对于`Survey`的第一次尝试`__unicode__`方法可能看起来像这样：

```py
    def __unicode__(self): 
        return u'%s (Opens %s, closes %s)' % (self.title, self.opens, self.closes) 
```

在这里，我们决定`Survey`实例的打印表示将由标题值后跟括号中的有关此调查何时开放和关闭的注释组成。鉴于该方法的定义，我们在测试创建实例时正确设置`closes`时的 shell 会话可能看起来像这样：

```py
>>> from survey.models import Survey 
>>> import datetime 
>>> sd = datetime.date(2009, 12, 28) 
>>> t = "New Year's Resolutions" 
>>> s = Survey.objects.create(title=t, opens=sd) 
>>> s 
<Survey: New Year's Resolutions (Opens 2009-12-28, closes 2010-01-04)> 
>>> 

```

也就是说，我们可能不是专门检查`closes`分配的值，而是显示已创建实例的打印表示，因为它包括`closes`的值。在 shell 会话中进行实验时，自然而然地会以这种方式进行检查，而不是直接询问相关属性。首先，这样做更短（`s`比`s.closes`更容易输入）。此外，它通常显示的信息比我们可能正在测试的特定部分更多，这在我们进行实验时是有帮助的。

然而，如果我们直接从 shell 会话中复制并粘贴到我们的`save`覆盖 doctest 中，我们就会使该 doctest 依赖于`__unicode__`的实现细节。随后，我们可能会决定不想在`Survey`的可打印表示中包含所有这些信息，甚至只是认为如果“Opens”中的“o”不大写会看起来更好。因此，我们对`__unicode__`方法的实现进行了微小的更改，突然间一个与其他方法无关的 doctest 开始失败了。

```py
====================================================================== 
FAIL: Doctest: survey.models.Survey.save 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 2189, in runTest 
 raise self.failureException(self.format_failure(new.getvalue())) 
AssertionError: Failed doctest test for survey.models.Survey.save 
 File "/dj_projects/marketr/survey/models.py", line 9, in save 

---------------------------------------------------------------------- 
File "/dj_projects/marketr/survey/models.py", line 32, in survey.models.Survey.save 
Failed example: 
 s 
Expected: 
 <Survey: New Year's Resolutions (Opens 2009-12-28, closes 2010-01-04)> 
Got: 
 <Survey: New Year's Resolutions (opens 2009-12-28, closes 2010-01-04)> 

---------------------------------------------------------------------- 
Ran 3 tests in 0.076s 

FAILED (failures=1) 
Destroying test database... 

```

因此，在从 shell 会话创建 doctests 时，需要仔细考虑会话是否依赖于被测试的代码以外的任何代码的实现细节，并相应地进行调整以消除这种依赖。在这种情况下，使用`s.closes`来测试`closes`被赋予了什么值，消除了对`Survey`模型`__unicode__`方法实现方式的依赖。

在 doctests 中可能会出现许多其他环境依赖的情况，包括：

+   任何依赖于文件路径打印表示的测试都可能会遇到问题，因为在基于 Unix 的操作系统上，路径组件由正斜杠分隔，而 Windows 使用反斜杠。如果需要包含依赖于文件路径值的 doctests，可能需要使用实用函数来规范不同操作系统上的文件路径表示。

+   任何依赖于字典键以特定顺序打印的测试都可能会遇到一个问题，即这个顺序在不同操作系统或 Python 实现中可能是不同的。因此，为了使这些测试在不同平台上更加健壮，可能需要专门查询字典键值，而不仅仅是打印整个字典内容，或者使用一个实用函数，为打印表示应用一致的顺序到键上。

关于这些在 doctests 中经常出现的环境依赖问题，没有什么特别与 Django 相关的内容。然而，在 Django 应用程序中特别容易出现一种环境依赖：数据库依赖。接下来将讨论这个问题。

## 警惕数据库依赖

Django 的**对象关系管理器**（**ORM**）非常费力地屏蔽应用程序代码与底层数据库的差异。但是，让所有不同的支持的数据库在所有情况下看起来完全相同对 Django 来说是不可行的。因此，在应用程序级别可能观察到特定于数据库的差异。这些差异可能很容易进入 doctests，使得测试依赖于特定的数据库后端才能通过。

这种依赖已经存在于本章早期开发的`save`覆盖测试中。因为 SQLite 是最容易使用的数据库（因为它不需要安装或配置），所以到目前为止，示例代码和测试都是使用`settings.py`中的`DATABASE_ENGINE = 'sqlite3'`设置开发的。如果我们切换到使用 MySQL（`DATABASE_ENGINE = 'mysql'`）作为数据库，并尝试运行我们的`survey`应用程序测试，我们将看到失败。有两个失败，但我们首先只关注测试输出中的最后一个：

```py
====================================================================== 
FAIL: Doctest: survey.tests.__test__.survey_save 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 2189, in runTest 
 raise self.failureException(self.format_failure(new.getvalue())) 
AssertionError: Failed doctest test for survey.tests.__test__.survey_save 
 File "/dj_projects/marketr/survey/tests.py", line unknown line number, in survey_save 

---------------------------------------------------------------------- 
File "/dj_projects/marketr/survey/tests.py", line ?, in survey.tests.__test__.survey_save 
Failed example: 
 s = Survey.objects.create(title=t) 
Expected: 
 Traceback (most recent call last): 
 ... 
 IntegrityError: survey_survey.opens may not be NULL 
Got: 
 Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 1274, in __run 
 compileflags, 1) in test.globs 
 File "<doctest survey.tests.__test__.survey_save[6]>", line 1, in <module> 
 s = Survey.objects.create(title=t) 
 File "/usr/lib/python2.5/site-packages/django/db/models/manager.py", line 126, in create 
 return self.get_query_set().create(**kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 315, in create 
 obj.save(force_insert=True) 
 File "/dj_projects/marketr/survey/models.py", line 34, in save 
 super(Survey, self).save(**kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 410, in save 
 self.save_base(force_insert=force_insert, force_update=force_update) 
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 495, in save_base 
 result = manager._insert(values, return_id=update_pk) 
 File "/usr/lib/python2.5/site-packages/django/db/models/manager.py", line 177, in _insert 
 return insert_query(self.model, values, **kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 1087, in insert_query 
 return query.execute_sql(return_id) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/subqueries.py", line 320, in execute_sql 
 cursor = super(InsertQuery, self).execute_sql(None) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/query.py", line 2369, in execute_sql 
 cursor.execute(sql, params) 
 File "/usr/lib/python2.5/site-packages/django/db/backends/mysql/base.py", line 89, in execute 
 raise Database.IntegrityError(tuple(e)) 
 IntegrityError: (1048, "Column 'opens' cannot be null") 

---------------------------------------------------------------------- 
Ran 3 tests in 0.434s 

FAILED (failures=2) 
Destroying test database... 

```

这里的问题是什么？在`tests.py`中的 doctest 中的`save`调用中没有为`opens`指定值，预期会出现`IntegrityError`，而确实出现了`IntegrityError`，但`IntegrityError`消息的细节是不同的。SQLite 数据库返回：

```py
 IntegrityError: survey_survey.opens may not be NULL 

```

MySQL 以稍微不同的方式表达了同样的观点：

```py
 IntegrityError: (1048, "Column 'opens' cannot be null") 

```

有两种简单的方法可以解决这个问题。一种是在失败的测试上使用 doctest 指令`IGNORE_EXCEPTION_DETAIL`。使用此选项，doctest 运行程序在确定预期结果是否与实际结果匹配时，只会考虑异常的类型（在本例中为`IntegrityError`）。因此，不同数据库产生的确切异常消息的差异不会导致测试失败。

通过在包含测试的行上将 doctest 指令指定为单个测试来指定。注释以`doctest：`开头，后面跟着一个或多个指令名称，前面是`+`表示打开选项，`-`表示关闭选项。因此，在这种情况下，我们将更改`tests.py`中失败的测试行为（请注意，尽管此行在此页面上换行到第二行，但在测试中需要保持在一行上）：

```py
>>> s = Survey.objects.create(title=t) # doctest: +IGNORE_EXCEPTION_DETAIL 
```

另一种修复方法是用省略号替换测试中预期输出的详细消息部分，省略号是一个省略标记。也就是说，将测试更改为：

```py
>>> s = Survey.objects.create(title=t) 
Traceback (most recent call last): 
  ... 
IntegrityError: ... 
```

这是告诉 doctest 运行器忽略异常消息的具体方法。它依赖于 doctest 选项`ELLIPSIS`在 doctest 运行时被启用。虽然这个选项在 Python 中默认情况下是不启用的，但是 Django 使用的 doctest 运行器启用了它，所以你不需要在你的测试代码中做任何事情来启用期望输出中的省略号标记。还要注意，`ELLIPSIS`不仅仅适用于异常消息的细节；它是一种更一般的方法，让你指示 doctest 输出的部分可能因运行而异，而不会导致测试失败。

### 注意

如果你阅读了`ELLIPSIS`的 Python 文档，你可能会注意到它是在 Python 2.4 中引入的。因此，如果你正在运行 Python 2.3（这仍然是 Django 1.1 支持的），你可能会期望在你的 Django 应用程序的 doctests 中无法使用省略号标记技术。然而，Django 1.0 和 1.1 附带了一个定制的 doctest 运行器，当你运行你的应用程序的 doctests 时会使用它。这个定制的运行器是基于 Python 2.4 附带的 doctest 模块的。因此，即使你运行的是早期的 Python 版本，你也可以使用 Python 2.4 中的 doctest 选项，比如`ELLIPSIS`。

注意，尽管 Django 使用自己定制的 doctest 运行器的另一面是：如果你运行的 Python 版本比 2.4 更新，你不能在应用程序的 doctests 中使用比 2.4 更晚添加的 doctest 选项。例如，Python 在 Python 2.5 中添加了`SKIP`选项。在 Django 更新其定制的 doctest 模块的版本之前，你将无法在 Django 应用程序的 doctests 中使用这个新选项。

回想一下，有两次测试失败，我们只看了其中一个的输出（另一个很可能滚动得太快，无法阅读）。然而，考虑到我们检查过的一个失败，我们可能期望另一个也是一样的，因为在`models.py`的 doctest 中，我们对`IntegrityError`有一个非常相似的测试：

```py
        >>> s.closes = None 
        >>> s.save() 
        Traceback (most recent call last): 
          ... 
        IntegrityError: survey_survey.closes may not be NULL 
```

这肯定也需要被修复以忽略异常细节，所以我们可能会同时做这两件事，并且可能会纠正两个测试失败。事实上，当我们在将两个预期的`IntegrityErrors`都更改为包含省略号标记而不是具体错误消息后再次运行测试时，所有的测试都通过了。

### 注意

请注意，对于某些 MySQL 的配置，忽略异常细节将无法纠正第二个测试失败。具体来说，如果 MySQL 服务器配置为以“非严格”模式运行，尝试将行更新为包含`NULL`值的列声明为`NOT NULL`不会引发错误。相反，该值将设置为列类型的隐式默认值，并发出警告。

很可能，如果你正在使用 MySQL，你会想要配置它以在“严格模式”下运行。然而，如果由于某种原因你不能这样做，并且你需要在你的应用程序中有这样一个测试，并且你需要测试在多个数据库上通过，你将不得不考虑在你的测试中考虑数据库行为的差异。这是可以做到的，但在单元测试中更容易完成，而不是在 doctest 中，所以我们不会讨论如何修复这种情况的 doctest。

现在我们已经让我们的测试在两个不同的数据库后端上通过了，我们可能会认为我们已经准备好了，并且可能会在 Django 支持的所有数据库上获得一个干净的测试运行。我们错了，当我们尝试使用 PostgreSQL 作为数据库运行相同的测试时，我们会发现数据库的差异，这突出了在编写 doctests 时需要注意的下一项内容，并在下一节中进行了介绍。

## 注意测试之间的相互依赖

如果我们现在尝试使用 PostgreSQL 作为数据库运行我们的测试（在`settings.py`中指定`DATABASE_ENGINE = 'postgresql_psycopg2'`），我们会得到一个非常奇怪的结果。从`manage.py test survey -v2`的输出的末尾，我们看到：

```py
No fixtures found. 
test_basic_addition (survey.tests.SimpleTest) ... ok 
Doctest: survey.models.Survey.save ... ok 
Doctest: survey.tests.__test__.survey_save ... FAIL 

```

我们仍然在`tests.py`中有一个样本单元测试运行并通过，然后`models.py`中的 doctest 也通过了，但我们添加到`tests.py`中的 doctest 失败了。失败的细节是：

```py
====================================================================== 
FAIL: Doctest: survey.tests.__test__.survey_save 
---------------------------------------------------------------------- 
Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 2189, in runTest 
 raise self.failureException(self.format_failure(new.getvalue())) 
AssertionError: Failed doctest test for survey.tests.__test__.survey_save 
 File "/dj_projects/marketr/survey/tests.py", line unknown line number, in survey_save 

---------------------------------------------------------------------- 
File "/dj_projects/marketr/survey/tests.py", line ?, in survey.tests.__test__.survey_save 
Failed example: 
 s = Survey.objects.create(title=t, opens=sd, closes=sd) 
Exception raised: 
 Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 1274, in __run 
 compileflags, 1) in test.globs 
 File "<doctest survey.tests.__test__.survey_save[4]>", line 1, in <module> 
 s = Survey.objects.create(title=t, opens=sd, closes=sd) 
 File "/usr/lib/python2.5/site-packages/django/db/models/manager.py", line 126, in create 
 return self.get_query_set().create(**kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 315, in create 
 obj.save(force_insert=True) 
 File "/dj_projects/marketr/survey/models.py", line 34, in save 
 super(Survey, self).save(**kwargs)
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 410, in save 
 self.save_base(force_insert=force_insert, force_update=force_update) 
 File "/usr/lib/python2.5/site-packages/django/db/models/base.py", line 495, in save_base 
 result = manager._insert(values, return_id=update_pk) 
 File "/usr/lib/python2.5/site-packages/django/db/models/manager.py", line 177, in _insert 
 return insert_query(self.model, values, **kwargs) 
 File "/usr/lib/python2.5/site-packages/django/db/models/query.py", line 1087, in insert_query 
 return query.execute_sql(return_id) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/subqueries.py", line 320, in execute_sql 
 cursor = super(InsertQuery, self).execute_sql(None) 
 File "/usr/lib/python2.5/site-packages/django/db/models/sql/query.py", line 2369, in execute_sql 
 cursor.execute(sql, params) 
 InternalError: current transaction is aborted, commands ignored until end of transaction block 

---------------------------------------------------------------------- 
File "/dj_projects/marketr/survey/tests.py", line ?, in survey.tests.__test__.survey_save 
Failed example: 
 s.closes 
Exception raised: 
 Traceback (most recent call last): 
 File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 1274, in __run 
 compileflags, 1) in test.globs 
 File "<doctest survey.tests.__test__.survey_save[5]>", line 1, in <module> 
 s.closes 
 NameError: name 's' is not defined 
 ****----------------------------------------------------------------------** 
**Ran 3 tests in 0.807s** 
 ****FAILED (failures=1)** 
**Destroying test database...****** 
```

这次我们需要按顺序检查报告的错误，因为第二个错误是由第一个错误导致的。这种错误的链接是常见的，因此要记住，虽然从测试运行结束时最容易看到的最后一个失败开始可能很诱人，但这可能不是最有效的方法。如果不立即明显导致最后一个失败的原因，通常最好从头开始，找出导致第一个失败的原因。随后的失败原因可能会变得明显。供参考，正在失败的测试的开头是：

```py
**>>> import datetime 
>>> from survey.models import Survey 
>>> t = "New Year's Resolutions" 
>>> sd = datetime.date(2009, 12, 28) 

If closes is specified, it will be honored and not auto-set. 

>>> s = Survey.objects.create(title=t, opens=sd, closes=sd) 
>>> s.closes 
datetime.date(2009, 12, 28)** 
```

因此，根据测试输出，这个测试中对数据库的第一次访问——也就是尝试创建`Survey`实例——导致了错误。

```py
****InternalError: current transaction is aborted, commands ignored until end of transaction block****
```

然后，测试的下一行也会导致错误，因为它使用了应该在上一行中分配的变量`s`。然而，那一行没有完成执行，所以当测试尝试使用它时，变量`s`没有被定义。因此，第二个错误是有道理的，考虑到第一个错误，但为什么这个测试中的第一个数据库访问会导致错误呢？

为了理解这一点的解释，我们必须回顾一下紧接在这个测试之前运行的测试。从测试输出中我们可以看到，紧接在这个测试之前的测试是`models.py`中的 doctest。该测试的结尾是：

```py
 **>>> s.closes = None 
        >>> s.save() 
        Traceback (most recent call last): 
          ... 
        IntegrityError: ... 
        """** 
```

测试的最后一件事是预期引发数据库错误的事情。在 PostgreSQL 上的一个副作用是，数据库连接进入了一个状态，只允许结束事务块的命令。因此，这个测试结束时，数据库连接处于一个破碎的状态，当下一个 doctest 开始运行时，它仍然处于破碎状态，导致下一个 doctest 在尝试任何数据库访问时立即失败。

这个问题说明了 doctests 之间没有数据库隔离。一个 doctest 对数据库的操作可以被后续运行的 doctest 观察到。这包括在数据库表中创建、更新或删除行的问题，以及在这里看到的问题。这个特定的问题可以通过在故意引起数据库错误的代码后添加一个回滚当前事务的调用来解决。

```py
 **>>> s.closes = None 
        >>> s.save() 
        Traceback (most recent call last): 
          ... 
        IntegrityError: ... 
        >>> from django.db import transaction 
        >>> transaction.rollback() 
        """** 
```

这将允许测试在 PostgreSQL 上通过，并且在其他数据库后端上是无害的。因此，处理 doctests 中没有数据库隔离的一种方法是编写代码，使它们在自己之后进行清理。这可能是一个可以接受的方法，但如果测试已经在数据库中添加、修改或删除了对象，可能很难将一切恢复到最初的状态。

第二种方法是在每个 doctest 进入时将数据库重置为已知状态。Django 不会为您执行此操作，但您可以通过调用管理命令来手动执行。我通常不建议这种方法，因为随着应用程序的增长，它变得非常耗时。

第三种方法是使 doctests 在数据库状态上相对宽容，这样它们可能会在其他测试是否运行过的情况下正常运行。在这里使用的技术包括：

+   在测试本身创建测试所需的所有对象。也就是说，不要依赖于任何先前运行的测试创建的对象的存在，因为该测试可能会更改，或被删除，或测试运行的顺序可能会在某个时候更改。

+   在创建对象时，要防止与其他测试可能创建的相似对象发生冲突。例如，如果一个测试需要创建一个`is_superuser`字段设置为`True`的`User`实例，以便测试具有该属性的用户的某些行为，那么给`User`实例一个`username`为"superuser"可能是很自然的。然而，如果两个 doctest 都这样做了，那么不幸的是第二个运行的测试会遇到错误，因为`User`模型的`username`字段被声明为唯一，所以第二次尝试使用这个`username`创建`User`会失败。因此，最好使用在共享模型中不太可能被其他测试使用的唯一字段的值。

所有这些方法和技术都有其缺点。对于这个特定问题，单元测试是一个更好的解决方案，因为它们可以自动提供数据库隔离，而不会产生重置数据库的性能成本（只要在支持事务的数据库上运行）。因此，如果你开始遇到很多 doctest 的测试相互依赖的问题，我强烈建议考虑单元测试作为解决方案，而不是依赖于这里列出的任何方法。

## 谨防 Unicode

我们将在 doctest 注意事项中涵盖的最后一个问题是 Unicode。如果你在 Django（甚至只是 Python）中使用了比英语更广泛的字符集的数据，你可能已经遇到过`UnicodeDecodeError`或`UnicodeEncodeError`一两次。因此，你可能已经养成了在测试中包含一些非 ASCII 字符的习惯，以确保一切都能正常工作，不仅仅是英语。这是一个好习惯，但不幸的是，在 doctest 中使用 Unicode 值进行测试会出现一些意想不到的故障，需要克服。

先前提到的`Survey`的`__unicode__`方法可能是我们希望在面对非 ASCII 字符时测试其行为是否正确的一个地方。对此进行测试的第一步可能是：

```py
 **def __unicode__(self): 
        """ 
        >>> t = u'¿Como está usted?' 
        >>> sd = datetime.date(2009, 12, 28) 
        >>> s = Survey.objects.create(title=t, opens=sd) 
        >>> print s 
        ¿Como está usted? (opens 2009-12-28, closes 2010-01-04) 
        """ 
        return u'%s (opens %s, closes %s)' % (self.title, self.opens, self.closes)** 
```

这个测试与许多保存覆盖测试类似，因为它首先创建了一个`Survey`实例。在这种情况下，重要的参数是标题，它被指定为 Unicode 文字字符串，并包含非 ASCII 字符。创建了`Survey`实例后，调用打印它以验证非 ASCII 字符在实例的打印表示中是否正确显示，并且没有引发 Unicode 异常。

这个测试效果如何？不太好。在添加了那段代码后，尝试运行调查测试会导致错误：

```py
****kmt@lbox:/dj_projects/marketr$ python manage.py test survey** 
**Traceback (most recent call last):** 
 **File "manage.py", line 11, in <module>** 
 **execute_manager(settings)** 
 **File "/usr/lib/python2.5/site-packages/django/core/management/__init__.py", line 362, in execute_manager** 
 **utility.execute()** 
 **File "/usr/lib/python2.5/site-packages/django/core/management/__init__.py", line 303, in execute** 
 **self.fetch_command(subcommand).run_from_argv(self.argv)** 
 **File "/usr/lib/python2.5/site-packages/django/core/management/base.py", line 195, in run_from_argv** 
 **self.execute(*args, **options.__dict__)** 
 **File "/usr/lib/python2.5/site-packages/django/core/management/base.py", line 222, in execute** 
 **output = self.handle(*args, **options)** 
 **File "/usr/lib/python2.5/site-packages/django/core/management/commands/test.py", line 23, in handle** 
 **failures = test_runner(test_labels, verbosity=verbosity, interactive=interactive)** 
 **File "/usr/lib/python2.5/site-packages/django/test/simple.py", line 178, in run_tests** 
 **app = get_app(label)** 
 **File "/usr/lib/python2.5/site-packages/django/db/models/loading.py", line 114, in get_app** 
 **self._populate()** 
 **File "/usr/lib/python2.5/site-packages/django/db/models/loading.py", line 58, in _populate** 
 **self.load_app(app_name, True)** 
 **File "/usr/lib/python2.5/site-packages/django/db/models/loading.py", line 74, in load_app** 
 **models = import_module('.models', app_name)** 
 **File "/usr/lib/python2.5/site-packages/django/utils/importlib.py", line 35, in import_module** 
 **__import__(name)** 
 **File "/dj_projects/marketr/survey/models.py", line 40** 
**SyntaxError: Non-ASCII character '\xc2' in file /dj_projects/marketr/survey/models.py on line 41, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details**** 
```

这个很容易解决；我们只是忘记了声明 Python 源文件的编码。为了做到这一点，我们需要在文件顶部添加一个注释行，指定文件使用的编码。假设我们使用 UTF-8 编码，所以我们应该将以下内容添加为我们的`models.py`文件的第一行：

```py
**# -*- encoding: utf-8 -*-** 
```

现在新的测试会起作用吗？还没有，我们仍然失败了：

```py
****======================================================================** 
**FAIL: Doctest: survey.models.Survey.__unicode__** 
**----------------------------------------------------------------------** 
**Traceback (most recent call last):** 
 **File "/usr/lib/python2.5/site-packages/django/test/_doctest.py", line 2180, in runTest** 
 **raise self.failureException(self.format_failure(new.getvalue()))** 
**AssertionError: Failed doctest test for survey.models.Survey.__unicode__** 
 **File "/dj_projects/marketr/survey/models.py", line 39, in __unicode__** 

**----------------------------------------------------------------------** 
**File "/dj_projects/marketr/survey/models.py", line 44, in survey.models.Survey.__unicode__** 
**Failed example:** 
 **print s** 
**Expected:** 
 **¿Como está usted? (opens 2009-12-28, closes 2010-01-04)** 
**Got:** 
 **Â¿Como estÃ¡ usted? (opens 2009-12-28, closes 2010-01-04)** 

**----------------------------------------------------------------------** 
**Ran 4 tests in 0.084s** 

**FAILED (failures=1)** 
**Destroying test database...**** 
```

这个有点令人费解。虽然我们在测试中将标题指定为 Unicode 文字字符串`u'¿Como está usted?'`，但打印出来时显然是**Â¿Como estÃ¡ usted?**。这种数据损坏是错误地使用了错误的编码将字节字符串转换为 Unicode 字符串的明显迹象。事实上，这里的损坏特性，即原始字符串中的每个非 ASCII 字符在损坏版本中被两个（或更多）字符替换，是实际上以 UTF-8 编码的字符串被解释为如果它是以 ISO-8859-1（也称为 Latin-1）编码的特征。但是这里怎么会发生这种情况，因为我们指定了 UTF-8 作为我们的 Python 文件编码声明？为什么这个字符串会使用其他编码来解释？

此时，我们可能会去仔细阅读我们收到的第一个错误消息中引用的网页，并了解到我们添加的编码声明只影响 Python 解释器从源文件构造 Unicode 文字字符串的方式。然后我们可能会注意到，尽管我们的标题是一个 Unicode 文字字符串，但包含 doctest 的文档字符串却不是。因此，也许这个奇怪的结果是因为我们忽略了将包含 doctest 的文档字符串作为 Unicode 文字字符串。因此，我们下一个版本的测试可能是将整个文档字符串指定为 Unicode 文字字符串。

不幸的是，这也将是不成功的，因为存在 Unicode 文字文档字符串的问题。首先，doctest 运行器无法正确比较预期输出（现在是 Unicode，因为文档字符串本身是 Unicode 文字）和包含非 ASCII 字符的字节串的实际输出。这样的字节串必须转换为 Unicode 以进行比较。当必要时，Python 将自动执行此转换，但问题在于它不知道正在转换的字节串的实际编码。因此，它假定为 ASCII，并且如果字节串包含任何非 ASCII 字符，则无法执行转换。

这种转换失败将导致涉及字节串的比较被假定为失败，进而导致测试被报告为失败。即使预期和接收到的输出是相同的，如果只假定了字节串的正确编码，也没有办法使正确的编码被使用，因此测试将失败。对于`Survey`模型`__unicode__` doctest，这个问题将导致在尝试比较`print s`的实际输出（这将是一个 UTF-8 编码的字节串）和预期输出时测试失败。

Unicode 文字文档字符串的第二个问题涉及包含非 ASCII 字符的输出的报告，例如在`Survey`模型`__unicode__` doctest 中将发生的失败。doctest 运行器将尝试显示一个消息，显示预期和接收到的输出。然而，当它尝试将预期和接收到的输出合并成一个用于显示的单个消息时，它将遇到与比较期间遇到的相同问题。因此，与其生成一个至少能够显示测试遇到问题的消息，doctest 运行器本身会生成`UnicodeDecodeError`。

Python 的 bug 跟踪器中有一个未解决的 Python 问题报告了这些问题：[`bugs.python.org/issue1293741`](http://bugs.python.org/issue1293741)。在它被修复之前，最好避免在 doctests 中使用 Unicode 文字文档字符串。

那么，有没有办法在 doctests 中包含一些非 ASCII 数据的测试？是的，这是可能的。使这样的测试起作用的关键是避免在文档字符串中使用 Unicode 文字。而是显式将字符串解码为 Unicode 对象。例如：

```py
 **def __unicode__(self): 
        """ 
        >>> t = '¿Como está usted?'.decode('utf-8') 
        >>> sd = datetime.date(2009, 12, 28) 
        >>> s = Survey.objects.create(title=t, opens=sd) 
        >>> print s 
        ¿Como está usted? (opens 2009-12-28, closes 2010-01-04) 
        """ 
        return u'%s (opens %s, closes %s)' % (self.title, self.opens, self.closes)** 
```

也就是说，用一个明确使用 UTF-8 解码的字节串替换 Unicode 文字标题字符串，以创建一个 Unicode 字符串。

这样做有用吗？现在运行`manage.py test survey -v2`，我们在输出的最后看到以下内容：

```py
****No fixtures found.** 
**test_basic_addition (survey.tests.SimpleTest) ... ok** 
**Doctest: survey.models.Survey.__unicode__ ... ok** 
**Doctest: survey.models.Survey.save ... ok** 
**Doctest: survey.tests.__test__.survey_save ... ok** 

**----------------------------------------------------------------------** 
**Ran 4 tests in 0.046s** 

**OK** 
**Destroying test database...**** 
```

成功！因此，在 doctests 中正确测试非 ASCII 数据是可能的。只需注意避免遇到使用 Unicode 文字文档字符串或在 doctest 中嵌入 Unicode 文字字符串相关的现有问题。

# 总结

我们对 Django 应用程序的 doctests 的探索现在已经完成。在本章中，我们：

+   开始为我们的 Django`survey`应用程序开发一些模型

+   尝试向其中一个模型添加 doctests——`Survey`模型

+   了解了哪些类型的 doctests 是有用的，哪些只是为代码添加了混乱

+   体验了 doctests 的一些优势，即轻松重用 Python shell 会话工作和方便地将 doctests 用作文档

+   遇到了许多 doctests 的缺点，并学会了如何避免或克服它们

在下一章中，我们将开始探索单元测试。虽然单元测试可能不提供一些 doctests 的轻松重用功能，但它们也不会受到许多 doctests 的缺点的影响。此外，整体的单元测试框架允许 Django 提供特别适用于 Web 应用程序的便利支持，这将在第四章中详细介绍。
