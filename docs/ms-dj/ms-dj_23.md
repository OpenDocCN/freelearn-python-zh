# 附录 B.数据库 API 参考

Django 的数据库 API 是附录 A 中讨论的模型 API 的另一半。一旦定义了模型，您将在需要访问数据库时使用此 API。您已经在整本书中看到了此 API 的使用示例；本附录详细解释了各种选项。

在本附录中，我将引用以下模型，这些模型组成了一个 Weblog 应用程序：

```py
from django.db import models 

class Blog(models.Model): 
    name = models.CharField(max_length=100) 
    tagline = models.TextField() 

    def __str__(self): 
        return self.name 

class Author(models.Model): 
    name = models.CharField(max_length=50) 
    email = models.EmailField() 

    def __str__(self): 
        return self.name 

class Entry(models.Model): 
    blog = models.ForeignKey(Blog) 
    headline = models.CharField(max_length=255) 
    body_text = models.TextField() 
    pub_date = models.DateField() 
    mod_date = models.DateField() 
    authors = models.ManyToManyField(Author) 
    n_comments = models.IntegerField() 
    n_pingbacks = models.IntegerField() 
    rating = models.IntegerField() 

    def __str__(self):        
        return self.headline 

```

# 创建对象

为了在 Python 对象中表示数据库表数据，Django 使用了一个直观的系统：模型类表示数据库表，该类的实例表示数据库表中的特定记录。

要创建对象，请使用模型类的关键字参数进行实例化，然后调用`save()`将其保存到数据库中。

假设模型位于文件`mysite/blog/models.py`中，这是一个示例：

```py
>>> from blog.models import Blog
>>> b = Blog(name='Beatles Blog', tagline='All the latest Beatles news.')
>>> b.save()

```

这在幕后执行`INSERT` SQL 语句。直到您明确调用`save()`之前，Django 不会访问数据库。

`save()`方法没有返回值。

要在单个步骤中创建和保存对象，请使用`create()`方法。

# 保存对象的更改

要保存已经在数据库中的对象的更改，请使用`save()`。

假设已经将`Blog`实例`b5`保存到数据库中，此示例更改其名称并更新数据库中的记录：

```py
>>> b5.name = 'New name'
>>> b5.save()

```

这在幕后执行`UPDATE` SQL 语句。Django 直到您明确调用`save()`之前才会访问数据库。

## 保存 ForeignKey 和 ManyToManyField 字段

更新`ForeignKey`字段的方式与保存普通字段的方式完全相同-只需将正确类型的对象分配给相关字段。此示例更新了`Entry`实例`entry`的`blog`属性，假设已经适当保存了`Entry`和`Blog`的实例到数据库中（因此我们可以在下面检索它们）：

```py
>>> from blog.models import Entry
>>> entry = Entry.objects.get(pk=1)
>>> cheese_blog = Blog.objects.get(name="Cheddar Talk")
>>> entry.blog = cheese_blog
>>> entry.save()

```

更新`ManyToManyField`的方式略有不同-使用字段上的`add()`方法将记录添加到关系中。此示例将`Author`实例`joe`添加到`entry`对象中：

```py
>>> from blog.models import Author
>>> joe = Author.objects.create(name="Joe")
>>> entry.authors.add(joe)

```

要一次向`ManyToManyField`添加多条记录，请在调用`add()`时包含多个参数，如下所示：

```py
>>> john = Author.objects.create(name="John")
>>> paul = Author.objects.create(name="Paul")
>>> george = Author.objects.create(name="George")
>>> ringo = Author.objects.create(name="Ringo")
>>> entry.authors.add(john, paul, george, ringo)

```

如果尝试分配或添加错误类型的对象，Django 会发出警告。

# 检索对象

要从数据库中检索对象，请通过模型类上的`Manager`构建`QuerySet`。

`QuerySet`表示来自数据库的对象集合。它可以有零个、一个或多个过滤器。过滤器根据给定的参数缩小查询结果。在 SQL 术语中，`QuerySet`等同于`SELECT`语句，而过滤器是诸如`WHERE`或`LIMIT`的限制子句。

通过使用模型的`Manager`来获取`QuerySet`。每个模型至少有一个`Manager`，默认情况下称为`objects`。直接通过模型类访问它，就像这样：

```py
>>> Blog.objects
<django.db.models.manager.Manager object at ...>
>>> b = Blog(name='Foo', tagline='Bar')
>>> b.objects
Traceback:
 ...
AttributeError: "Manager isn't accessible via Blog instances."

```

## 检索所有对象

从表中检索对象的最简单方法是获取所有对象。要做到这一点，使用`Manager`上的`all()`方法：

```py
>>> all_entries = Entry.objects.all()

```

`all()`方法返回数据库中所有对象的`QuerySet`。

## 使用过滤器检索特定对象

`all()`返回的`QuerySet`描述了数据库表中的所有对象。通常，您需要选择完整对象集的子集。

要创建这样的子集，您需要细化初始的`QuerySet`，添加过滤条件。细化`QuerySet`的两种最常见的方法是：

+   `filter(**kwargs)`。返回一个包含匹配给定查找参数的对象的新`QuerySet`。

+   `exclude(**kwargs)`。返回一个包含不匹配给定查找参数的对象的新`QuerySet`。

查找参数（上述函数定义中的`**kwargs`）应该以本章后面描述的*字段查找*格式。

### 链接过滤器

细化`QuerySet`的结果本身是一个`QuerySet`，因此可以将细化链接在一起。例如：

```py
>>> Entry.objects.filter(
...     headline__startswith='What'
... ).exclude(
...     pub_date__gte=datetime.date.today()
... ).filter(pub_date__gte=datetime(2005, 1, 30)
... )

```

这需要数据库中所有条目的初始`QuerySet`，添加一个过滤器，然后一个排除，然后另一个过滤器。最终结果是一个包含所有以`What`开头的标题的条目，发布日期在 2005 年 1 月 30 日和当天之间的`QuerySet`。

## 过滤的查询集是唯一的

每次细化`QuerySet`，您都会得到一个全新的`QuerySet`，它与以前的`QuerySet`没有任何关联。每次细化都会创建一个单独且独特的`QuerySet`，可以存储、使用和重复使用。

例子：

```py
>>> q1 = Entry.objects.filter(headline__startswith="What")
>>> q2 = q1.exclude(pub_date__gte=datetime.date.today())
>>> q3 = q1.filter(pub_date__gte=datetime.date.today())

```

这三个`QuerySets`是独立的。第一个是一个基本的`QuerySet`，包含所有以 What 开头的标题的条目。第二个是第一个的子集，增加了一个额外的条件，排除了`pub_date`是今天或将来的记录。第三个是第一个的子集，增加了一个额外的条件，只选择`pub_date`是今天或将来的记录。初始的`QuerySet`（`q1`）不受细化过程的影响。

### QuerySets 是惰性的

`QuerySets`是惰性的-创建`QuerySet`的行为不涉及任何数据库活动。您可以整天堆叠过滤器，Django 实际上不会运行查询，直到`QuerySet`被*评估*。看看这个例子：

```py
>>> q = Entry.objects.filter(headline__startswith="What")
>>> q = q.filter(pub_date__lte=datetime.date.today())
>>> q = q.exclude(body_text__icontains="food")
>>> print(q)

```

尽管看起来像是三次数据库访问，实际上只有一次，在最后一行（`print(q)`）访问数据库。通常情况下，只有在要求时，`QuerySet`的结果才会从数据库中获取。当您这样做时，通过访问数据库来*评估*`QuerySet`。

## 使用 get 检索单个对象

`filter()`总是会给你一个`QuerySet`，即使只有一个对象匹配查询-在这种情况下，它将是包含单个元素的`QuerySet`。

如果您知道只有一个对象与您的查询匹配，您可以在`Manager`上使用`get()`方法直接返回对象：

```py
>>> one_entry = Entry.objects.get(pk=1)

```

您可以像使用`filter()`一样使用`get()`的任何查询表达式-再次参见本章的下一节中的*字段查找*。

请注意，使用`get()`和使用`filter()`与`[0]`的切片之间存在差异。如果没有结果与查询匹配，`get()`将引发`DoesNotExist`异常。此异常是正在执行查询的模型类的属性-因此在上面的代码中，如果没有主键为 1 的`Entry`对象，Django 将引发`Entry.DoesNotExist`。

类似地，如果`get()`查询匹配多个项目，Django 将抱怨。在这种情况下，它将引发`MultipleObjectsReturned`，这也是模型类本身的属性。

## 其他查询集方法

大多数情况下，当您需要从数据库中查找对象时，您将使用`all()`、`get()`、`filter()`和`exclude()`。但这远非全部；请参阅[`docs.djangoproject.com/en/1.8/ref/models/querysets/`](https://docs.djangoproject.com/en/1.8/ref/models/querysets/)上的 QuerySet API 参考，了解所有各种`QuerySet`方法的完整列表。

## 限制查询集

使用 Python 的数组切片语法的子集来限制您的`QuerySet`到一定数量的结果。这相当于 SQL 的`LIMIT`和`OFFSET`子句。

例如，这将返回前 5 个对象（`LIMIT 5`）：

```py
>>> Entry.objects.all()[:5]

```

这将返回第六到第十个对象（`OFFSET 5 LIMIT 5`）：

```py
>>> Entry.objects.all()[5:10]

```

不支持负索引（即`Entry.objects.all()[-1]`）。

通常，对`QuerySet`进行切片会返回一个新的`QuerySet`-它不会评估查询。一个例外是如果您使用 Python 切片语法的步长参数。例如，这实际上会执行查询，以返回前 10 个对象中每*第二个*对象的列表：

```py
>>> Entry.objects.all()[:10:2]

```

要检索*单个*对象而不是列表（例如，`SELECT foo FROM bar LIMIT 1`），请使用简单的索引而不是切片。

例如，这将按标题字母顺序返回数据库中的第一个`Entry`：

```py
>>> Entry.objects.order_by('headline')[0]

```

这大致相当于：

```py
>>> Entry.objects.order_by('headline')[0:1].get()

```

但是请注意，如果没有对象符合给定的条件，第一个将引发`IndexError`，而第二个将引发`DoesNotExist`。有关更多详细信息，请参见`get()`。

## 字段查找

字段查找是指定 SQL `WHERE`子句的主要方式。它们被指定为`QuerySet`方法`filter()`、`exclude()`和`get()`的关键字参数。（这是一个双下划线）。例如：

```py
>>> Entry.objects.filter(pub_date__lte='2006-01-01')

```

翻译（大致）成以下 SQL：

```py
SELECT * FROM blog_entry WHERE pub_date <= '2006-01-01';

```

查找中指定的字段必须是模型字段的名称。不过有一个例外，在`ForeignKey`的情况下，可以指定带有`_id`后缀的字段名。在这种情况下，值参数预期包含外键模型主键的原始值。例如：

```py
>>> Entry.objects.filter(blog_id=4)

```

如果传递了无效的关键字参数，查找函数将引发`TypeError`。

字段查找的完整列表如下：

+   `精确的`

+   `忽略大小写的精确的`

+   `包含`

+   `包含`

+   `在…中`

+   `大于`

+   `大于或等于`

+   `小于`

+   `小于或等于`

+   `以…开头`

+   `以…开头`

+   `以…结尾`

+   `以…结尾`

+   `范围`

+   `年`

+   `月`

+   `天`

+   `星期几`

+   `小时`

+   `分钟`

+   `秒`

+   `为空`

+   `搜索`

+   `正则表达式`

+   `iregex`

可以在字段查找参考中找到每个字段查找的完整参考和示例[`docs.djangoproject.com/en/1.8/ref/models/querysets/#field-lookups`](https://docs.djangoproject.com/en/1.8/ref/models/querysets/#field-lookups)。

## 跨关系的查找

Django 提供了一种强大且直观的方式来在查找中跟踪关系，自动在幕后为您处理 SQL `JOIN`。要跨越关系，只需使用跨模型的相关字段的字段名称，用双下划线分隔，直到您找到想要的字段。

这个例子检索所有`name`为`'Beatles Blog'`的`Blog`对象的`Entry`对象：

```py
>>> Entry.objects.filter(blog__name='Beatles Blog')

```

这种跨度可以深入到您想要的程度。

它也可以反向操作。要引用反向关系，只需使用模型的小写名称。

这个例子检索所有至少有一个`Entry`的`headline`包含`'Lennon'`的`Blog`对象：

```py
>>> Blog.objects.filter(entry__headline__contains='Lennon')

```

如果您在多个关系中进行过滤，并且中间模型之一没有满足过滤条件的值，Django 将把它视为一个空（所有值都为`NULL`），但有效的对象。这只意味着不会引发错误。例如，在这个过滤器中：

```py
Blog.objects.filter(entry__authors__name='Lennon') 

```

（如果有一个相关的`Author`模型），如果一个条目没有与作者相关联，它将被视为没有附加名称，而不是因为缺少作者而引发错误。通常这正是您希望发生的。唯一可能令人困惑的情况是如果您使用`isnull`。因此：

```py
Blog.objects.filter(entry__authors__name__isnull=True) 

```

将返回在`author`上有一个空的`name`的`Blog`对象，以及在`entry`上有一个空的`author`的`Blog`对象。如果您不想要后者的对象，您可以这样写：

```py
Blog.objects.filter(entry__authors__isnull=False, 
        entry__authors__name__isnull=True) 

```

### 跨多值关系

当您基于`ManyToManyField`或反向`ForeignKey`对对象进行过滤时，可能会对两种不同类型的过滤感兴趣。考虑`Blog`/`Entry`关系（`Blog`到`Entry`是一对多关系）。我们可能对找到有一个条目既在标题中有`Lennon`又在 2008 年发布的博客感兴趣。

或者我们可能想要找到博客中有一个标题中带有`Lennon`的条目以及一个在 2008 年发布的条目。由于一个`Blog`关联多个条目，这两个查询都是可能的，并且在某些情况下是有意义的。

与`ManyToManyField`相同类型的情况也会出现。例如，如果`Entry`有一个名为`tags`的`ManyToManyField`，我们可能想要找到链接到名称为`music`和`bands`的标签的条目，或者我们可能想要一个包含名称为`music`和状态为`public`的标签的条目。

为了处理这两种情况，Django 有一种一致的处理`filter()`和`exclude()`调用的方式。单个`filter()`调用中的所有内容同时应用于过滤掉符合所有这些要求的项目。

连续的`filter()`调用进一步限制对象集，但对于多值关系，它们适用于与主要模型链接的任何对象，不一定是由先前的`filter()`调用选择的对象。

这可能听起来有点混乱，所以希望通过一个例子来澄清。要选择包含标题中都有`Lennon`并且在 2008 年发布的条目的所有博客（同时满足这两个条件的相同条目），我们将写：

```py
Blog.objects.filter(entry__headline__contains='Lennon',
        entry__pub_date__year=2008) 

```

要选择包含标题中有`Lennon`的条目以及 2008 年发布的条目的所有博客，我们将写：

```py
Blog.objects.filter(entry__headline__contains='Lennon').filter(
        entry__pub_date__year=2008) 

```

假设只有一个博客既包含`Lennon`的条目，又包含 2008 年的条目，但 2008 年的条目中没有包含`Lennon`。第一个查询将不会返回任何博客，但第二个查询将返回那一个博客。

在第二个例子中，第一个过滤器将查询集限制为所有链接到标题中有`Lennon`的条目的博客。第二个过滤器将进一步将博客集限制为那些还链接到 2008 年发布的条目的博客。

第二个过滤器选择的条目可能与第一个过滤器中的条目相同，也可能不同。我们正在使用每个过滤器语句过滤`Blog`项，而不是`Entry`项。

所有这些行为也适用于`exclude()`：单个`exclude()`语句中的所有条件都适用于单个实例（如果这些条件涉及相同的多值关系）。后续`filter()`或`exclude()`调用中涉及相同关系的条件可能最终会过滤不同的链接对象。

## 过滤器可以引用模型上的字段

到目前为止给出的例子中，我们已经构建了比较模型字段值与常量的过滤器。但是，如果您想要比较模型字段的值与同一模型上的另一个字段呢？

Django 提供了`F 表达式`来允许这样的比较。`F()`的实例充当查询中模型字段的引用。然后可以在查询过滤器中使用这些引用来比较同一模型实例上两个不同字段的值。

例如，要查找所有博客条目中评论比 pingbacks 多的条目列表，我们构建一个`F()`对象来引用 pingback 计数，并在查询中使用该`F()`对象：

```py
>>> from django.db.models import F
>>> Entry.objects.filter(n_comments__gt=F('n_pingbacks'))

```

Django 支持使用`F()`对象进行加法、减法、乘法、除法、取模和幂运算，既可以与常量一起使用，也可以与其他`F()`对象一起使用。要查找所有评论比 pingbacks 多*两倍*的博客条目，我们修改查询：

```py
>>> Entry.objects.filter(n_comments__gt=F('n_pingbacks') * 2)

```

要查找所有评分小于 pingback 计数和评论计数之和的条目，我们将发出查询：

```py
>>> Entry.objects.filter(rating__lt=F('n_comments') + F('n_pingbacks'))

```

您还可以使用双下划线符号来跨越`F()`对象中的关系。带有双下划线的`F()`对象将引入访问相关对象所需的任何连接。

例如，要检索所有作者名称与博客名称相同的条目，我们可以发出查询：

```py
>>> Entry.objects.filter(authors__name=F('blog__name'))

```

对于日期和日期/时间字段，您可以添加或减去一个`timedelta`对象。以下将返回所有在发布后 3 天以上修改的条目：

```py
>>> from datetime import timedelta
>>> Entry.objects.filter(mod_date__gt=F('pub_date') + timedelta(days=3))

```

`F()`对象支持按位操作，通过`.bitand()`和`.bitor()`，例如：

```py
>>> F('somefield').bitand(16)

```

## pk 查找快捷方式

为了方便起见，Django 提供了一个`pk`查找快捷方式，代表主键。

在示例`Blog`模型中，主键是`id`字段，因此这三个语句是等价的：

```py
>>> Blog.objects.get(id__exact=14) # Explicit form
>>> Blog.objects.get(id=14) # __exact is implied
>>> Blog.objects.get(pk=14) # pk implies id__exact

```

`pk`的使用不限于`__exact`查询-任何查询条件都可以与`pk`组合，以对模型的主键执行查询：

```py
# Get blogs entries with id 1, 4 and 7
>>> Blog.objects.filter(pk__in=[1,4,7])
# Get all blog entries with id > 14
>>> Blog.objects.filter(pk__gt=14)

```

`pk`查找也适用于连接。例如，这三个语句是等价的：

```py
>>> Entry.objects.filter(blog__id__exact=3) # Explicit form
>>> Entry.objects.filter(blog__id=3)        # __exact is implied
>>> Entry.objects.filter(blog__pk=3)        # __pk implies __id__exact

```

## 在 LIKE 语句中转义百分号和下划线

等同于`LIKE` SQL 语句的字段查找（`iexact`，`contains`，`icontains`，`startswith`，`istartswith`，`endswith`和`iendswith`）将自动转义`LIKE`语句中使用的两个特殊字符-百分号和下划线。（在`LIKE`语句中，百分号表示多字符通配符，下划线表示单字符通配符。）

这意味着事情应该直观地工作，所以抽象不会泄漏。例如，要检索包含百分号的所有条目，只需像对待其他字符一样使用百分号：

```py
>>> Entry.objects.filter(headline__contains='%')

```

Django 会为您处理引用；生成的 SQL 将类似于这样：

```py
SELECT ... WHERE headline LIKE '%\%%';

```

下划线也是一样。百分号和下划线都会被透明地处理。

## 缓存和查询集

每个`QuerySet`都包含一个缓存，以最小化数据库访问。了解它的工作原理将使您能够编写最有效的代码。

在新创建的`QuerySet`中，缓存是空的。第一次评估`QuerySet`时-因此，数据库查询发生时-Django 会将查询结果保存在`QuerySet`类的缓存中，并返回已经明确请求的结果（例如，如果正在迭代`QuerySet`，则返回下一个元素）。后续的`QuerySet`评估将重用缓存的结果。

请记住这种缓存行为，因为如果您没有正确使用您的`QuerySet`，它可能会给您带来麻烦。例如，以下操作将创建两个`QuerySet`，对它们进行评估，然后丢弃它们：

```py
>>> print([e.headline for e in Entry.objects.all()])
>>> print([e.pub_date for e in Entry.objects.all()])

```

这意味着相同的数据库查询将被执行两次，有效地增加了数据库负载。此外，两个列表可能不包括相同的数据库记录，因为在两个请求之间的瞬间，可能已经添加或删除了`Entry`。

为了避免这个问题，只需保存`QuerySet`并重复使用它：

```py
>>> queryset = Entry.objects.all()
>>> print([p.headline for p in queryset]) # Evaluate the query set.
>>> print([p.pub_date for p in queryset]) # Re-use the cache from the evaluation.

```

### 当查询集没有被缓存时

查询集并不总是缓存它们的结果。当仅评估查询集的*部分*时，会检查缓存，但如果它没有被填充，那么后续查询返回的项目将不会被缓存。具体来说，这意味着使用数组切片或索引限制查询集将不会填充缓存。

例如，重复获取查询集对象中的某个索引将每次查询数据库：

```py
>>> queryset = Entry.objects.all()
>>> print queryset[5] # Queries the database
>>> print queryset[5] # Queries the database again

```

然而，如果整个查询集已经被评估，那么将检查缓存：

```py
>>> queryset = Entry.objects.all()
>>> [entry for entry in queryset] # Queries the database
>>> print queryset[5] # Uses cache
>>> print queryset[5] # Uses cache

```

以下是一些其他操作的例子，这些操作将导致整个查询集被评估，因此填充缓存：

```py
>>> [entry for entry in queryset]
>>> bool(queryset)
>>> entry in queryset
>>> list(queryset)

```

# 使用 Q 对象进行复杂的查找

关键字参数查询-在`filter()`和其他地方-会被 AND 在一起。如果您需要执行更复杂的查询（例如带有`OR`语句的查询），您可以使用`Q 对象`。

`Q 对象`（`django.db.models.Q`）是一个用于封装一组关键字参数的对象。这些关键字参数如上面的字段查找中所指定的那样。

例如，这个`Q`对象封装了一个单一的`LIKE`查询：

```py
from django.db.models import Q 
Q(question__startswith='What') 

```

`Q`对象可以使用`&`和`|`运算符进行组合。当两个`Q`对象上使用运算符时，它会产生一个新的`Q`对象。

例如，这个语句产生一个代表两个`"question__startswith"`查询的 OR 的单个`Q`对象：

```py
Q(question__startswith='Who') | Q(question__startswith='What') 

```

这等同于以下 SQL `WHERE`子句：

```py
WHERE question LIKE 'Who%' OR question LIKE 'What%'

```

您可以通过使用`&`和`|`运算符组合`Q`对象并使用括号分组来组成任意复杂的语句。此外，`Q`对象可以使用`~`运算符进行否定，从而允许组合查找结合了正常查询和否定（`NOT`）查询：

```py
Q(question__startswith='Who') | ~Q(pub_date__year=2005) 

```

每个接受关键字参数的查找函数（例如`filter()`、`exclude()`、`get()`）也可以作为位置（非命名）参数传递一个或多个`Q`对象。如果向查找函数提供多个`Q`对象参数，则这些参数将被 AND 在一起。例如：

```py
Poll.objects.get( 
    Q(question__startswith='Who'), 
    Q(pub_date=date(2005, 5, 2)) | Q(pub_date=date(2005, 5, 6)) 
) 

```

...大致翻译成 SQL：

```py
SELECT * from polls WHERE question LIKE 'Who%'
 AND (pub_date = '2005-05-02' OR pub_date = '2005-05-06')

```

查找函数可以混合使用`Q`对象和关键字参数。提供给查找函数的所有参数（无论是关键字参数还是`Q`对象）都会被 AND 在一起。但是，如果提供了`Q`对象，它必须在任何关键字参数的定义之前。例如：

```py
Poll.objects.get( 
    Q(pub_date=date(2005, 5, 2)) | Q(pub_date=date(2005, 5, 6)), 
    question__startswith='Who') 

```

...将是一个有效的查询，等同于前面的示例；但是：

```py
# INVALID QUERY 
Poll.objects.get( 
    question__startswith='Who', 
    Q(pub_date=date(2005, 5, 2)) | Q(pub_date=date(2005, 5, 6))) 

```

...将无效。

# 比较对象

要比较两个模型实例，只需使用标准的 Python 比较运算符，双等号：`==`。在幕后，这比较了两个模型的主键值。

使用上面的`Entry`示例，以下两个语句是等价的：

```py
>>> some_entry == other_entry
>>> some_entry.id == other_entry.id

```

如果模型的主键不叫`id`，没问题。比较将始终使用主键，无论它叫什么。例如，如果模型的主键字段叫`name`，这两个语句是等价的：

```py
>>> some_obj == other_obj
>>> some_obj.name == other_obj.name

```

# 删除对象

方便地，删除方法被命名为`delete()`。此方法立即删除对象，并且没有返回值。例如：

```py
e.delete() 

```

您还可以批量删除对象。每个`QuerySet`都有一个`delete()`方法，用于删除该`QuerySet`的所有成员。

例如，这将删除所有`pub_date`年份为 2005 的`Entry`对象：

```py
Entry.objects.filter(pub_date__year=2005).delete() 

```

请记住，这将在可能的情况下纯粹在 SQL 中执行，因此在过程中不一定会调用单个对象实例的`delete()`方法。如果您在模型类上提供了自定义的`delete()`方法，并希望确保它被调用，您将需要手动删除该模型的实例（例如，通过迭代`QuerySet`并在每个对象上调用`delete()`）而不是使用`QuerySet`的批量`delete()`方法。

当 Django 删除一个对象时，默认情况下会模拟 SQL 约束`ON DELETE CASCADE`的行为-换句话说，任何具有指向要删除的对象的外键的对象都将与其一起被删除。例如：

```py
b = Blog.objects.get(pk=1) 
# This will delete the Blog and all of its Entry objects. 
b.delete() 

```

此级联行为可以通过`ForeignKey`的`on_delete`参数进行自定义。

请注意，`delete()`是唯一不在`Manager`本身上公开的`QuerySet`方法。这是一个安全机制，可以防止您意外请求`Entry.objects.delete()`，并删除*所有*条目。如果*确实*要删除所有对象，则必须显式请求完整的查询集：

```py
Entry.objects.all().delete() 

```

# 复制模型实例

虽然没有内置的方法来复制模型实例，但可以轻松地创建具有所有字段值的新实例。在最简单的情况下，您可以将`pk`设置为`None`。使用我们的博客示例：

```py
blog = Blog(name='My blog', tagline='Blogging is easy') 
blog.save() # blog.pk == 1 

blog.pk = None 
blog.save() # blog.pk == 2 

```

如果使用继承，情况会变得更加复杂。考虑`Blog`的子类：

```py
class ThemeBlog(Blog): 
    theme = models.CharField(max_length=200) 

django_blog = ThemeBlog(name='Django', tagline='Django is easy',
  theme='python') 
django_blog.save() # django_blog.pk == 3 

```

由于继承的工作原理，您必须将`pk`和`id`都设置为 None：

```py
django_blog.pk = None 
django_blog.id = None 
django_blog.save() # django_blog.pk == 4 

```

此过程不会复制相关对象。如果要复制关系，您需要编写更多的代码。在我们的示例中，`Entry`有一个到`Author`的多对多字段：

```py
entry = Entry.objects.all()[0] # some previous entry 
old_authors = entry.authors.all() 
entry.pk = None 
entry.save() 
entry.authors = old_authors # saves new many2many relations 

```

# 一次更新多个对象

有时，您希望为`QuerySet`中的所有对象设置一个特定的值。您可以使用`update()`方法来实现这一点。例如：

```py
# Update all the headlines with pub_date in 2007.
Entry.objects.filter(pub_date__year=2007).update(headline='Everything is the same')

```

您只能使用此方法设置非关系字段和`ForeignKey`字段。要更新非关系字段，请将新值提供为常量。要更新`ForeignKey`字段，请将新值设置为要指向的新模型实例。例如：

```py
>>> b = Blog.objects.get(pk=1)
# Change every Entry so that it belongs to this Blog.
>>> Entry.objects.all().update(blog=b)

```

`update()`方法会立即应用，并返回查询匹配的行数（如果某些行已经具有新值，则可能不等于更新的行数）。

更新的`QuerySet`的唯一限制是它只能访问一个数据库表，即模型的主表。您可以基于相关字段进行过滤，但只能更新模型主表中的列。例如：

```py
>>> b = Blog.objects.get(pk=1)
# Update all the headlines belonging to this Blog.
>>> Entry.objects.select_related().filter(blog=b).update
(headline='Everything is the same')

```

请注意，`update()`方法会直接转换为 SQL 语句。这是一个用于直接更新的批量操作。它不会运行任何模型的`save()`方法，也不会发出`pre_save`或`post_save`信号（这是调用`save()`的结果），也不会遵守`auto_now`字段选项。如果您想保存`QuerySet`中的每个项目，并确保在每个实例上调用`save()`方法，您不需要任何特殊的函数来处理。只需循环遍历它们并调用`save()`：

```py
for item in my_queryset: 
    item.save() 

```

对更新的调用也可以使用`F 表达式`来根据模型中另一个字段的值更新一个字段。这对于根据其当前值递增计数器特别有用。例如，要为博客中的每个条目递增 pingback 计数：

```py
>>> Entry.objects.all().update(n_pingbacks=F('n_pingbacks') + 1)

```

但是，与在过滤和排除子句中使用`F()`对象不同，当您在更新中使用`F()`对象时，您不能引入连接-您只能引用要更新的模型本地字段。如果尝试使用`F()`对象引入连接，将引发`FieldError`：

```py
# THIS WILL RAISE A FieldError
>>> Entry.objects.update(headline=F('blog__name'))

```

# 相关对象

当您在模型中定义关系（即`ForeignKey`、`OneToOneField`或`ManyToManyField`）时，该模型的实例将具有便捷的 API 来访问相关对象。

使用本页顶部的模型，例如，`Entry`对象`e`可以通过访问`blog`属性获取其关联的`Blog`对象：`e.blog`。

（在幕后，这个功能是由 Python 描述符实现的。这对您来说并不重要，但我在这里指出它是为了满足好奇心。）

Django 还为关系的另一侧创建了 API 访问器-从相关模型到定义关系的模型的链接。例如，`Blog`对象`b`通过`entry_set`属性可以访问所有相关的`Entry`对象的列表：`b.entry_set.all()`。

本节中的所有示例都使用本页顶部定义的示例`Blog`、`Author`和`Entry`模型。

## 一对多关系

### 前向

如果模型具有`ForeignKey`，则该模型的实例将可以通过模型的简单属性访问相关（外键）对象。例如：

```py
>>> e = Entry.objects.get(id=2)
>>> e.blog # Returns the related Blog object.

```

您可以通过外键属性进行获取和设置。正如您可能期望的那样，对外键的更改直到调用`save()`之前都不会保存到数据库。例如：

```py
>>> e = Entry.objects.get(id=2)
>>> e.blog = some_blog
>>> e.save()

```

如果`ForeignKey`字段设置了`null=True`（即允许`NULL`值），则可以分配`None`来删除关系。例如：

```py
>>> e = Entry.objects.get(id=2)
>>> e.blog = None
>>> e.save() # "UPDATE blog_entry SET blog_id = NULL ...;"

```

第一次访问相关对象时，可以缓存对一对多关系的前向访问。对同一对象实例上的外键的后续访问将被缓存。例如：

```py
>>> e = Entry.objects.get(id=2)
>>> print(e.blog)  # Hits the database to retrieve the associated Blog.
>>> print(e.blog)  # Doesn't hit the database; uses cached version.

```

请注意，`select_related()` `QuerySet` 方法会预先递归填充所有一对多关系的缓存。例如：

```py
>>> e = Entry.objects.select_related().get(id=2)
>>> print(e.blog)  # Doesn't hit the database; uses cached version.
>>> print(e.blog)  # Doesn't hit the database; uses cached version.

```

### 向后跟踪关系

如果模型具有`ForeignKey`，则外键模型的实例将可以访问返回第一个模型的所有实例的`Manager`。默认情况下，此`Manager`命名为`foo_set`，其中`foo`是源模型名称的小写形式。此`Manager`返回`QuerySets`，可以像上面的检索对象部分中描述的那样进行过滤和操作。

例如：

```py
>>> b = Blog.objects.get(id=1)
>>> b.entry_set.all() # Returns all Entry objects related to Blog.
# b.entry_set is a Manager that returns QuerySets.
>>> b.entry_set.filter(headline__contains='Lennon')
>>> b.entry_set.count()

```

您可以通过在`ForeignKey`定义中设置`related_name`参数来覆盖`foo_set`名称。例如，如果`Entry`模型被修改为`blog = ForeignKey(Blog, related_name='entries')`，上面的示例代码将如下所示：

```py
>>> b = Blog.objects.get(id=1)
>>> b.entries.all() # Returns all Entry objects related to Blog.
# b.entries is a Manager that returns QuerySets.
>>> b.entries.filter(headline__contains='Lennon')
>>> b.entries.count()

```

### 使用自定义反向管理器

默认情况下，用于反向关系的`RelatedManager`是该模型的默认管理器的子类。如果您想为给定查询指定不同的管理器，可以使用以下语法：

```py
from django.db import models 

class Entry(models.Model): 
    #... 
    objects = models.Manager()  # Default Manager 
    entries = EntryManager()    # Custom Manager 

b = Blog.objects.get(id=1) 
b.entry_set(manager='entries').all() 

```

如果`EntryManager`在其`get_queryset()`方法中执行默认过滤，则该过滤将应用于`all()`调用。

当然，指定自定义的反向管理器也使您能够调用其自定义方法：

```py
b.entry_set(manager='entries').is_published() 

```

### 处理相关对象的附加方法

除了之前*检索对象*中定义的`QuerySet`方法之外，`ForeignKey` `Manager`还有其他用于处理相关对象集合的方法。每个方法的概要如下（完整详情可以在相关对象参考中找到[`docs.djangoproject.com/en/1.8/ref/models/relations/#related-objects-reference`](https://docs.djangoproject.com/en/1.8/ref/models/relations/#related-objects-reference)）：

+   `add(obj1, obj2, ...)` 将指定的模型对象添加到相关对象集合

+   `create(**kwargs)` 创建一个新对象，保存它并将其放入相关对象集合中。返回新创建的对象

+   `remove(obj1, obj2, ...)` 从相关对象集合中删除指定的模型对象

+   `clear()` 从相关对象集合中删除所有对象

+   `set(objs)` 替换相关对象的集合

要一次性分配相关集合的成员，只需从任何可迭代对象中分配给它。可迭代对象可以包含对象实例，也可以只是主键值的列表。例如：

```py
b = Blog.objects.get(id=1) 
b.entry_set = [e1, e2] 

```

在这个例子中，`e1`和`e2`可以是完整的 Entry 实例，也可以是整数主键值。

如果`clear()`方法可用，那么在将可迭代对象（在本例中是一个列表）中的所有对象添加到集合之前，`entry_set`中的任何现有对象都将被移除。如果`clear()`方法*不*可用，则将添加可迭代对象中的所有对象，而不会移除任何现有元素。

本节中描述的每个反向操作都会立即对数据库产生影响。每次添加、创建和删除都会立即自动保存到数据库中。

## 多对多关系

多对多关系的两端都自动获得对另一端的 API 访问权限。API 的工作方式与上面的反向一对多关系完全相同。

唯一的区别在于属性命名：定义`ManyToManyField`的模型使用该字段本身的属性名称，而反向模型使用原始模型的小写模型名称，再加上`'_set'`（就像反向一对多关系一样）。

一个例子可以更容易理解：

```py
e = Entry.objects.get(id=3) 
e.authors.all() # Returns all Author objects for this Entry. 
e.authors.count() 
e.authors.filter(name__contains='John') 

a = Author.objects.get(id=5) 
a.entry_set.all() # Returns all Entry objects for this Author. 

```

与`ForeignKey`一样，`ManyToManyField`可以指定`related_name`。在上面的例子中，如果`Entry`中的`ManyToManyField`指定了`related_name='entries'`，那么每个`Author`实例将具有一个`entries`属性，而不是`entry_set`。

## 一对一关系

一对一关系与多对一关系非常相似。如果在模型上定义了`OneToOneField`，那么该模型的实例将通过模型的简单属性访问相关对象。

例如：

```py
class EntryDetail(models.Model): 
    entry = models.OneToOneField(Entry) 
    details = models.TextField() 

ed = EntryDetail.objects.get(id=2) 
ed.entry # Returns the related Entry object. 

```

不同之处在于反向查询。一对一关系中的相关模型也可以访问`Manager`对象，但该`Manager`代表单个对象，而不是一组对象：

```py
e = Entry.objects.get(id=2) 
e.entrydetail # returns the related EntryDetail object 

```

如果没有对象分配给这个关系，Django 将引发`DoesNotExist`异常。

实例可以被分配到反向关系，就像你分配正向关系一样：

```py
e.entrydetail = ed 

```

## 涉及相关对象的查询

涉及相关对象的查询遵循涉及正常值字段的查询相同的规则。在指定要匹配的查询值时，您可以使用对象实例本身，也可以使用对象的主键值。

例如，如果您有一个`id=5`的 Blog 对象`b`，那么以下三个查询将是相同的：

```py
Entry.objects.filter(blog=b) # Query using object instance 
Entry.objects.filter(blog=b.id) # Query using id from instance 
Entry.objects.filter(blog=5) # Query using id directly 

```

# 回退到原始 SQL

如果你发现自己需要编写一个对 Django 的数据库映射器处理过于复杂的 SQL 查询，你可以回退到手动编写 SQL。

最后，重要的是要注意，Django 数据库层只是与您的数据库交互的接口。您可以通过其他工具、编程语言或数据库框架访问您的数据库；您的数据库与 Django 无关。
