# 第九章：高级模型

在第四章: 
    name = models.CharField(max_length=30) 
    address = models.CharField(max_length=50) 
    city = models.CharField(max_length=60) 
    state_province = models.CharField(max_length=30) 
    country = models.CharField(max_length=50) 
    website = models.URLField() 

    def __str__(self): 
        return self.name 

class Author(models.Model): 
    first_name = models.CharField(max_length=30) 
    last_name = models.CharField(max_length=40) 
    email = models.EmailField() 

    def __str__(self): 
        return '%s %s' % (self.first_name, self.last_name) 

class Book(models.Model): 
    title = models.CharField(max_length=100) 
    authors = models.ManyToManyField(Author) 
    publisher = models.ForeignKey(Publisher) 
    publication_date = models.DateField() 

    def __str__(self): 
        return self.title 

```

正如我们在第四章 
>>> b.title 
'The Django Book' 

```

但我们之前没有提到的一件事是，相关对象-表达为`ForeignKey`或`ManyToManyField`的字段-的行为略有不同。

## 访问 ForeignKey 值

当您访问一个`ForeignKey`字段时，您将获得相关的模型对象。例如：

```py
>>> b = Book.objects.get(id=50) 
>>> b.publisher 
<Publisher: Apress Publishing> 
>>> b.publisher.website 
'http://www.apress.com/' 

```

对于`ForeignKey`字段，它也可以反向工作，但由于关系的非对称性，它略有不同。要获取给定出版商的书籍列表，请使用“publisher.book_set.all（）”，如下所示：

```py
>>> p = Publisher.objects.get(name='Apress Publishing') 
>>> p.book_set.all() 
[<Book: The Django Book>, <Book: Dive Into Python>, ...] 

```

在幕后，`book_set`只是一个`QuerySet`（如第四章 
>>> p.book_set.filter(title__icontains='django') 
[<Book: The Django Book>, <Book: Pro Django>] 

```

属性名称`book_set`是通过将小写模型名称附加到`_set`而生成的。

## 访问多对多值

多对多值的工作方式与外键值相似，只是我们处理的是`QuerySet`值而不是模型实例。例如，以下是如何查看书籍的作者：

```py
>>> b = Book.objects.get(id=50) 
>>> b.authors.all() 
[<Author: Adrian Holovaty>, <Author: Jacob Kaplan-Moss>] 
>>> b.authors.filter(first_name='Adrian') 
[<Author: Adrian Holovaty>] 
>>> b.authors.filter(first_name='Adam') 
[] 

```

它也可以反向工作。要查看作者的所有书籍，请使用`author.book_set`，如下所示：

```py
>>> a = Author.objects.get(first_name='Adrian', last_name='Holovaty') 
>>> a.book_set.all() 
[<Book: The Django Book>, <Book: Adrian's Other Book>] 

```

在这里，与`ForeignKey`字段一样，`book_set`属性名称是通过将小写模型名称附加到`_set`而生成的。

# 管理器

在语句“Book.objects.all（）”中，`objects`是一个特殊的属性，通过它您可以查询您的数据库。在第四章: 
    def title_count(self, keyword): 
        return self.filter(title__icontains=keyword).count() 

class Book(models.Model): 
    title = models.CharField(max_length=100) 
    authors = models.ManyToManyField(Author) 
    publisher = models.ForeignKey(Publisher) 
    publication_date = models.DateField() 
    num_pages = models.IntegerField(blank=True, null=True) 
    objects = BookManager() 

    def __str__(self): 
        return self.title 

```

以下是有关代码的一些说明：

+   我们创建了一个扩展了`django.db.models.Manager`的`BookManager`类。这有一个名为“title_count（）”的方法，用于进行计算。请注意，该方法使用“self.filter（）”，其中`self`是指管理器本身。

+   我们将“BookManager（）”分配给模型上的`objects`属性。这会替换模型的默认管理器，称为`objects`，如果您没有指定自定义管理器，则会自动创建。我们将其称为`objects`而不是其他名称，以便与自动创建的管理器保持一致。

有了这个管理器，我们现在可以这样做：

```py
>>> Book.objects.title_count('django') 
4 
>>> Book.objects.title_count('python') 
18 

```

显然，这只是一个例子-如果您在交互式提示符中键入此内容，您可能会得到不同的返回值。

为什么我们想要添加一个像 `title_count()` 这样的方法？为了封装常用的执行查询，这样我们就不必重复代码。

## 修改初始管理器查询集

管理器的基本 `QuerySet` 返回系统中的所有对象。例如，`Book.objects.all()` 返回书数据库中的所有书籍。你可以通过覆盖 `Manager.get_queryset()` 方法来覆盖管理器的基本 `QuerySet`。`get_queryset()` 应该返回一个具有你需要的属性的 `QuerySet`。

例如，以下模型有两个管理器-一个返回所有对象，一个只返回罗尔德·达尔的书。

```py
from django.db import models 

# First, define the Manager subclass. 
class DahlBookManager(models.Manager): 
    def get_queryset(self): 
        return super(DahlBookManager, self).get_queryset().filter(author='Roald Dahl') 

# Then hook it into the Book model explicitly. 
class Book(models.Model): 
    title = models.CharField(max_length=100) 
    author = models.CharField(max_length=50) 
    # ... 

    objects = models.Manager() # The default manager. 
    dahl_objects = DahlBookManager() # The Dahl-specific manager. 

```

使用这个示例模型，`Book.objects.all()` 将返回数据库中的所有书籍，但 `Book.dahl_objects.all()` 只会返回罗尔德·达尔写的书。请注意，我们明确将 `objects` 设置为一个普通的 `Manager` 实例，因为如果我们没有这样做，唯一可用的管理器将是 `dahl_objects`。当然，因为 `get_queryset()` 返回一个 `QuerySet` 对象，你可以在其上使用 `filter()`、`exclude()` 和所有其他 `QuerySet` 方法。因此，这些语句都是合法的：

```py
Book.dahl_objects.all() 
Book.dahl_objects.filter(title='Matilda') 
Book.dahl_objects.count() 

```

这个例子还指出了另一个有趣的技术：在同一个模型上使用多个管理器。你可以将多个 `Manager()` 实例附加到一个模型上。这是定义模型的常见过滤器的简单方法。例如：

```py
class MaleManager(models.Manager): 
    def get_queryset(self): 
        return super(MaleManager, self).get_queryset().filter(sex='M') 

class FemaleManager(models.Manager): 
    def get_queryset(self): 
        return super(FemaleManager, self).get_queryset().filter(sex='F') 

class Person(models.Model): 
    first_name = models.CharField(max_length=50) 
    last_name = models.CharField(max_length=50) 
    sex = models.CharField(max_length=1,  
                           choices=( 
                                    ('M', 'Male'),   
                                    ('F', 'Female') 
                           ) 
                           ) 
    people = models.Manager() 
    men = MaleManager() 
    women = FemaleManager() 

```

这个例子允许你请求 `Person.men.all()`, `Person.women.all()`, 和 `Person.people.all()`, 产生可预测的结果。如果你使用自定义的 `Manager` 对象，请注意 Django 遇到的第一个 `Manager`（按照模型中定义的顺序）具有特殊状态。Django 将在类中定义的第一个 `Manager` 解释为默认的 `Manager`，并且 Django 的几个部分（尽管不包括管理应用程序）将专门使用该 `Manager` 来管理该模型。

因此，在选择默认管理器时要小心，以避免覆盖 `get_queryset()` 导致无法检索到你想要处理的对象的情况。

# 模型方法

在模型上定义自定义方法，为对象添加自定义的行级功能。而管理器旨在对整个表执行操作，模型方法应该作用于特定的模型实例。这是将业务逻辑集中在一个地方-模型中的一个有价值的技术。

举例是最容易解释这个问题的方法。下面是一个带有一些自定义方法的模型：

```py
from django.db import models 

class Person(models.Model): 
    first_name = models.CharField(max_length=50) 
    last_name = models.CharField(max_length=50) 
    birth_date = models.DateField() 

    def baby_boomer_status(self): 
        # Returns the person's baby-boomer status. 
        import datetime 
        if self.birth_date < datetime.date(1945, 8, 1): 
            return "Pre-boomer" 
        elif self.birth_date < datetime.date(1965, 1, 1): 
            return "Baby boomer" 
        else: 
            return "Post-boomer" 

    def _get_full_name(self): 
        # Returns the person's full name." 
        return '%s %s' % (self.first_name, self.last_name) 
    full_name = property(_get_full_name) 

```

附录 A 中的模型实例引用，*模型定义参考*，列出了自动赋予每个模型的完整方法列表。你可以覆盖大部分方法（见下文），但有一些你几乎总是想要定义的：

+   `__str__()`: 一个 Python *魔术方法*，返回任何对象的 Unicode 表示。这是 Python 和 Django 在需要将模型实例强制转换并显示为普通字符串时使用的方法。特别是，当你在交互式控制台或管理界面中显示对象时，就会发生这种情况。

+   你总是希望定义这个方法；默认情况下并不是很有用。

+   `get_absolute_url()`: 这告诉 Django 如何计算对象的 URL。Django 在其管理界面中使用这个方法，以及任何时候它需要为对象计算 URL。

任何具有唯一标识 URL 的对象都应该定义这个方法。

## 覆盖预定义的模型方法

还有一组模型方法，封装了一堆你想要自定义的数据库行为。特别是，你经常会想要改变 `save()` 和 `delete()` 的工作方式。你可以自由地覆盖这些方法（以及任何其他模型方法）来改变行为。覆盖内置方法的一个经典用例是，如果你想要在保存对象时发生某些事情。例如，（参见 `save()` 以获取它接受的参数的文档）：

```py
from django.db import models 

class Blog(models.Model): 
    name = models.CharField(max_length=100) 
    tagline = models.TextField() 

    def save(self, *args, **kwargs): 
        do_something() 
        super(Blog, self).save(*args, **kwargs) # Call the "real" save() method. 
        do_something_else() 

```

你也可以阻止保存：

```py
from django.db import models 

class Blog(models.Model): 
    name = models.CharField(max_length=100) 
    tagline = models.TextField() 

    def save(self, *args, **kwargs): 
        if self.name == "Yoko Ono's blog": 
            return # Yoko shall never have her own blog! 
        else: 
            super(Blog, self).save(*args, **kwargs) # Call the "real" save() method. 

```

重要的是要记住调用超类方法-也就是`super(Blog, self).save(*args, **kwargs)`，以确保对象仍然被保存到数据库中。如果忘记调用超类方法，就不会发生默认行为，数据库也不会被触及。

还要确保通过可以传递给模型方法的参数-这就是`*args, **kwargs`的作用。Django 会不时地扩展内置模型方法的功能，添加新的参数。如果在方法定义中使用`*args, **kwargs`，则可以确保在添加这些参数时，您的代码将自动支持这些参数。

# 执行原始 SQL 查询

当模型查询 API 不够用时，可以退而使用原始 SQL。Django 提供了两种执行原始 SQL 查询的方法：您可以使用`Manager.raw()`执行原始查询并返回模型实例，或者完全避开模型层并直接执行自定义 SQL。

### 注意

每次使用原始 SQL 时，都应该非常小心。您应该使用`params`正确转义用户可以控制的任何参数，以防止 SQL 注入攻击。

# 执行原始 SQL 查询

`raw()`管理器方法可用于执行返回模型实例的原始 SQL 查询：

```py
Manager.raw(raw_query, params=None, translations=None)
```

此方法接受原始 SQL 查询，执行它，并返回一个`django.db.models.query.RawQuerySet`实例。这个`RawQuerySet`实例可以像普通的`QuerySet`一样进行迭代，以提供对象实例。这最好用一个例子来说明。假设您有以下模型：

```py
class Person(models.Model): 
    first_name = models.CharField(...) 
    last_name = models.CharField(...) 
    birth_date = models.DateField(...) 

```

然后，您可以执行自定义的 SQL，就像这样：

```py
>>> for p in Person.objects.raw('SELECT * FROM myapp_person'): 
...     print(p) 
John Smith 
Jane Jones 

```

当然，这个例子并不是很令人兴奋-它与运行`Person.objects.all()`完全相同。但是，`raw()`有很多其他选项，使其非常强大。

## 模型表名称

在前面的例子中，`Person`表的名称是从哪里来的？默认情况下，Django 通过将模型的应用程序标签（您在`manage.py startapp`中使用的名称）与模型的类名结合起来，它们之间用下划线连接来确定数据库表名称。在我们的例子中，假设`Person`模型位于名为`myapp`的应用程序中，因此其表将是`myapp_person`。

有关`db_table`选项的更多详细信息，请查看文档，该选项还允许您手动设置数据库表名称。

### 注意

对传递给`raw()`的 SQL 语句不进行检查。Django 期望该语句将从数据库返回一组行，但不执行任何强制性操作。如果查询不返回行，将导致（可能是晦涩的）错误。

## 将查询字段映射到模型字段

`raw()`会自动将查询中的字段映射到模型中的字段。查询中字段的顺序并不重要。换句话说，以下两个查询的工作方式是相同的：

```py
>>> Person.objects.raw('SELECT id, first_name, last_name, birth_date FROM myapp_person') 
... 
>>> Person.objects.raw('SELECT last_name, birth_date, first_name, id FROM myapp_person') 
... 

```

匹配是通过名称完成的。这意味着您可以使用 SQL 的`AS`子句将查询中的字段映射到模型字段。因此，如果您有其他表中有`Person`数据，您可以轻松地将其映射到`Person`实例中：

```py
>>> Person.objects.raw('''SELECT first AS first_name, 
...                              last AS last_name, 
...                              bd AS birth_date, 
...                              pk AS id, 
...                       FROM some_other_table''') 

```

只要名称匹配，模型实例就会被正确创建。或者，您可以使用`raw()`的`translations`参数将查询中的字段映射到模型字段。这是一个将查询中的字段名称映射到模型字段名称的字典。例如，前面的查询也可以这样写：

```py
>>> name_map = {'first': 'first_name', 'last': 'last_name', 'bd': 'birth_date', 'pk': 'id'} 
>>> Person.objects.raw('SELECT * FROM some_other_table', translations=name_map) 

```

## 索引查找

`raw()`支持索引，因此如果只需要第一个结果，可以这样写：

```py
>>> first_person = Person.objects.raw('SELECT * FROM myapp_person')[0] 

```

但是，索引和切片不是在数据库级别执行的。如果数据库中有大量的`Person`对象，限制 SQL 级别的查询效率更高：

```py
>>> first_person = Person.objects.raw('SELECT * FROM myapp_person LIMIT 1')[0] 

```

## 延迟加载模型字段

字段也可以被省略：

```py
>>> people = Person.objects.raw('SELECT id, first_name FROM myapp_person') 

```

此查询返回的`Person`对象将是延迟加载的模型实例（参见`defer()`）。这意味着从查询中省略的字段将按需加载。例如：

```py
>>> for p in Person.objects.raw('SELECT id, first_name FROM myapp_person'): 
...     print(p.first_name, # This will be retrieved by the original query 
...           p.last_name) # This will be retrieved on demand 
... 
John Smith 
Jane Jones 

```

从外观上看，这似乎是查询已检索了名字和姓氏。但是，这个例子实际上发出了 3 个查询。只有第一个名字是由`raw()`查询检索到的-当打印它们时，姓氏是按需检索的。

只有一个字段是不能省略的-主键字段。Django 使用主键来标识模型实例，因此它必须始终包含在原始查询中。如果您忘记包括主键，将会引发`InvalidQuery`异常。

## 添加注释

您还可以执行包含模型上未定义的字段的查询。例如，我们可以使用 PostgreSQL 的`age()`函数来获取一个人的年龄列表，其年龄由数据库计算得出：

```py
>>> people = Person.objects.raw('SELECT *, age(birth_date) AS age FROM myapp_person') 
>>> for p in people: 
...     print("%s is %s." % (p.first_name, p.age)) 
John is 37\. 
Jane is 42\. 
... 

```

## 将参数传递给原始查询

如果您需要执行参数化查询，可以将`params`参数传递给`raw()`：

```py
>>> lname = 'Doe' 
>>> Person.objects.raw('SELECT * FROM myapp_person WHERE last_name = %s', [lname]) 

```

`params`是参数的列表或字典。您将在查询字符串中使用`%s`占位符来表示列表，或者使用`%(key)s`占位符来表示字典（其中`key`当然会被字典键替换），而不管您的数据库引擎如何。这些占位符将被`params`参数中的参数替换。

### 注意

**不要在原始查询上使用字符串格式化！**

很容易将前面的查询写成：

`>>> query = 'SELECT * FROM myapp_person WHERE last_name = %s' % lname` `Person.objects.raw(query)`

**不要这样做。**

使用`params`参数完全保护您免受 SQL 注入攻击，这是一种常见的攻击方式，攻击者会将任意 SQL 注入到您的数据库中。如果您使用字符串插值，迟早会成为 SQL 注入的受害者。只要记住始终使用`params`参数，您就会得到保护。

# 直接执行自定义 SQL

有时甚至`Manager.raw()`还不够：您可能需要执行与模型不太匹配的查询，或者直接执行`UPDATE`、`INSERT`或`DELETE`查询。在这些情况下，您可以始终直接访问数据库，完全绕过模型层。对象`django.db.connection`表示默认数据库连接。要使用数据库连接，调用`connection.cursor()`以获取游标对象。然后，调用`cursor.execute(sql, [params])`来执行 SQL，`cursor.fetchone()`或`cursor.fetchall()`来返回结果行。例如：

```py
from django.db import connection 

def my_custom_sql(self): 
    cursor = connection.cursor() 
    cursor.execute("UPDATE bar SET foo = 1 WHERE baz = %s", [self.baz]) 
    cursor.execute("SELECT foo FROM bar WHERE baz = %s", [self.baz]) 
    row = cursor.fetchone() 

    return row 

```

请注意，如果您想在查询中包含百分号，您必须在传递参数的情况下将其加倍：

```py
cursor.execute("SELECT foo FROM bar WHERE baz = '30%'") 
cursor.execute("SELECT foo FROM bar WHERE baz = '30%%' AND  
  id = %s", [self.id]) 

```

如果您使用多个数据库，可以使用`django.db.connections`来获取特定数据库的连接（和游标）。`django.db.connections`是一个类似字典的对象，允许您使用其别名检索特定连接：

```py
from django.db import connections 
cursor = connections['my_db_alias'].cursor() 
# Your code here... 

```

默认情况下，Python DB API 将返回结果而不带有它们的字段名称，这意味着您最终会得到一个值的`list`，而不是一个`dict`。以较小的性能成本，您可以通过类似以下的方式返回结果作为`dict`：

```py
def dictfetchall(cursor): 
    # Returns all rows from a cursor as a dict 
    desc = cursor.description 
    return [ 
        dict(zip([col[0] for col in desc], row)) 
        for row in cursor.fetchall() 
    ] 

```

以下是两者之间差异的示例：

```py
>>> cursor.execute("SELECT id, parent_id FROM test LIMIT 2"); 
>>> cursor.fetchall() 
((54360982L, None), (54360880L, None)) 

>>> cursor.execute("SELECT id, parent_id FROM test LIMIT 2"); 
>>> dictfetchall(cursor) 
[{'parent_id': None, 'id': 54360982L}, {'parent_id': None, 'id': 54360880L}] 

```

## 连接和游标

`connection`和`cursor`大多实现了 PEP 249 中描述的标准 Python DB-API（有关更多信息，请访问[`www.python.org/dev/peps/pep-0249`](https://www.python.org/dev/peps/pep-0249)），除了在处理事务时。如果您不熟悉 Python DB-API，请注意`cursor.execute()`中的 SQL 语句使用占位符"`%s`"，而不是直接在 SQL 中添加参数。

如果您使用这种技术，底层数据库库将根据需要自动转义参数。还要注意，Django 期望"`%s`"占位符，而不是 SQLite Python 绑定使用的`?`占位符。这是为了一致性和健全性。使用游标作为上下文管理器：

```py
with connection.cursor() as c: 
    c.execute(...) 

```

等同于：

```py
c = connection.cursor() 
try: 
    c.execute(...) 
finally: 
    c.close() 

```

## 添加额外的 Manager 方法

添加额外的`Manager`方法是向模型添加表级功能的首选方式。（对于行级功能，即对模型对象的单个实例进行操作的函数，请使用模型方法，而不是自定义的`Manager`方法。）自定义的`Manager`方法可以返回任何你想要的东西。它不一定要返回一个`QuerySet`。

例如，这个自定义的`Manager`提供了一个名为`with_counts()`的方法，它返回所有`OpinionPoll`对象的列表，每个对象都有一个额外的`num_responses`属性，这是聚合查询的结果。

```py
from django.db import models 

class PollManager(models.Manager): 
    def with_counts(self): 
        from django.db import connection 
        cursor = connection.cursor() 
        cursor.execute(""" 
            SELECT p.id, p.question, p.poll_date, COUNT(*) 
            FROM polls_opinionpoll p, polls_response r 
            WHERE p.id = r.poll_id 
            GROUP BY p.id, p.question, p.poll_date 
            ORDER BY p.poll_date DESC""") 
        result_list = [] 
        for row in cursor.fetchall(): 
            p = self.model(id=row[0], question=row[1], poll_date=row[2]) 
            p.num_responses = row[3] 
            result_list.append(p) 
        return result_list 

class OpinionPoll(models.Model): 
    question = models.CharField(max_length=200) 
    poll_date = models.DateField() 
    objects = PollManager() 

class Response(models.Model): 
    poll = models.ForeignKey(OpinionPoll) 
    person_name = models.CharField(max_length=50) 
    response = models.TextField() 

```

使用这个例子，您可以使用`OpinionPoll.objects.with_counts()`来返回带有`num_responses`属性的`OpinionPoll`对象列表。关于这个例子的另一点要注意的是，`Manager`方法可以访问`self.model`来获取它们所附加的模型类。

# 接下来呢？

在下一章中，我们将向您展示 Django 的通用视图框架，它可以帮助您节省时间，构建遵循常见模式的网站。
