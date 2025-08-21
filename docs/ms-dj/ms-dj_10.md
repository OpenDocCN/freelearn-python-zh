# 第十章：通用视图

这里再次出现了本书的一个重要主题：在最糟糕的情况下，Web 开发是乏味和单调的。到目前为止，我们已经介绍了 Django 如何在模型和模板层减轻了一些单调，但 Web 开发人员在视图层也会经历这种乏味。

Django 的*通用视图*是为了减轻这种痛苦而开发的。

它们采用了在视图开发中发现的某些常见习语和模式，并对它们进行抽象，以便您可以快速编写常见的数据视图，而无需编写太多代码。我们可以识别出某些常见任务，比如显示对象列表，并编写显示任何对象列表的代码。

然后，可以将相关模型作为 URLconf 的额外参数传递。Django 附带了用于执行以下操作的通用显示视图：

+   显示单个对象的列表和详细页面。如果我们正在创建一个管理会议的应用程序，那么`TalkListView`和`RegisteredUserListView`将是列表视图的示例。单个讲话页面是我们称之为详细视图的示例。

+   在年/月/日归档页面、相关详细信息和最新页面中呈现基于日期的对象。

+   允许用户创建、更新和删除对象-无论是否授权。

这些视图一起提供了执行开发人员在视图中显示数据库数据时遇到的最常见任务的简单界面。最后，显示视图只是 Django 全面基于类的视图系统的一部分。有关 Django 提供的其他基于类的视图的完整介绍和详细描述，请参阅附录 C，*通用视图参考*。

# 对象的通用视图

当涉及呈现数据库内容的视图时，Django 的通用视图确实表现出色。因为这是一个常见的任务，Django 附带了一些内置的通用视图，使生成对象的列表和详细视图变得非常容易。

让我们从一些显示对象列表或单个对象的示例开始。我们将使用这些模型：

```py
# models.py 
from django.db import models 

class Publisher(models.Model): 
    name = models.CharField(max_length=30) 
    address = models.CharField(max_length=50) 
    city = models.CharField(max_length=60) 
    state_province = models.CharField(max_length=30) 
    country = models.CharField(max_length=50) 
    website = models.URLField() 

    class Meta: 
        ordering = ["-name"] 

    def __str__(self): 
        return self.name 

class Author(models.Model): 
    salutation = models.CharField(max_length=10) 
    name = models.CharField(max_length=200) 
    email = models.EmailField() 
    headshot = models.ImageField(upload_to='author_headshots') 

    def __str__(self): 
        return self.name 

class Book(models.Model): 
    title = models.CharField(max_length=100) 
    authors = models.ManyToManyField('Author') 
    publisher = models.ForeignKey(Publisher) 
    publication_date = models.DateField() 

```

现在我们需要定义一个视图：

```py
# views.py 
from django.views.generic import ListView 
from books.models import Publisher 

class PublisherList(ListView): 
    model = Publisher 

```

最后将该视图挂接到您的 URL 中：

```py
# urls.py 
from django.conf.urls import url 
from books.views import PublisherList 

urlpatterns = [ 
    url(r'^publishers/$', PublisherList.as_view()), 
] 

```

这是我们需要编写的所有 Python 代码。但是，我们仍然需要编写一个模板。但是，我们可以通过向视图添加`template_name`属性来明确告诉视图使用哪个模板，但在没有显式模板的情况下，Django 将从对象的名称中推断一个模板。在这种情况下，推断的模板将是`books/publisher_list.html`-books 部分来自定义模型的定义应用程序的名称，而“publisher”部分只是模型名称的小写版本。

因此，当（例如）在`TEMPLATES`中将`DjangoTemplates`后端的`APP_DIRS`选项设置为 True 时，模板位置可以是：`/path/to/project/books/templates/books/publisher_list.html`

这个模板将根据包含名为`object_list`的变量的上下文进行渲染，该变量包含所有发布者对象。一个非常简单的模板可能如下所示：

```py
{% extends "base.html" %} 

{% block content %} 
    <h2>Publishers</h2> 
    <ul> 
        {% for publisher in object_list %} 
            <li>{{ publisher.name }}</li> 
        {% endfor %} 
    </ul> 
{% endblock %} 

```

这就是全部。通用视图的所有很酷的功能都来自于更改通用视图上设置的属性。附录 C，*通用视图参考*，详细记录了所有通用视图及其选项；本文档的其余部分将考虑您可能定制和扩展通用视图的一些常见方法。

# 创建“友好”的模板上下文

您可能已经注意到我们的示例发布者列表模板将所有发布者存储在名为`object_list`的变量中。虽然这样做完全没问题，但对于模板作者来说并不是很“友好”：他们必须“知道”他们在这里处理的是发布者。

在 Django 中，如果您正在处理模型对象，则已为您完成此操作。 当您处理对象或查询集时，Django 使用模型类名称的小写版本填充上下文。 除了默认的`object_list`条目之外，这是额外提供的，但包含完全相同的数据，即`publisher_list`。

如果这仍然不是一个很好的匹配，您可以手动设置上下文变量的名称。 通用视图上的`context_object_name`属性指定要使用的上下文变量：

```py
# views.py 
from django.views.generic import ListView 
from books.models import Publisher 

class PublisherList(ListView): 
    model = Publisher 
 context_object_name = 'my_favorite_publishers'

```

提供有用的`context_object_name`始终是一个好主意。 设计模板的同事会感谢您。

# 添加额外的上下文

通常，您只需要提供一些通用视图提供的信息之外的额外信息。 例如，考虑在每个出版商详细页面上显示所有书籍的列表。 `DetailView`通用视图提供了出版商的上下文，但是我们如何在模板中获取额外的信息呢？

答案是子类化`DetailView`并提供您自己的`get_context_data`方法的实现。 默认实现只是将要显示的对象添加到模板中，但您可以重写它以发送更多内容：

```py
from django.views.generic import DetailView 
from books.models import Publisher, Book 

class PublisherDetail(DetailView): 

    model = Publisher 

    def get_context_data(self, **kwargs): 
        # Call the base implementation first to get a context 
        context = super(PublisherDetail, self).get_context_data(**kwargs) 
        # Add in a QuerySet of all the books 
        context['book_list'] = Book.objects.all() 
        return context 

```

### 注意

通常，`get_context_data`将合并当前类的所有父类的上下文数据。 要在您自己的类中保留此行为，其中您想要更改上下文，您应该确保在超类上调用`get_context_data`。 当没有两个类尝试定义相同的键时，这将产生预期的结果。

但是，如果任何类尝试在父类设置它之后覆盖键（在调用 super 之后），那么该类的任何子类在 super 之后也需要显式设置它，如果他们想确保覆盖所有父类。 如果您遇到问题，请查看视图的方法解析顺序。

# 查看对象的子集

现在让我们更仔细地看看我们一直在使用的`model`参数。 `model`参数指定视图将操作的数据库模型，在操作单个对象或一组对象的所有通用视图上都可用。 但是，`model`参数不是指定视图将操作的对象的唯一方法-您还可以使用`queryset`参数指定对象的列表：

```py
from django.views.generic import DetailView 
from books.models import Publisher 

class PublisherDetail(DetailView): 

    context_object_name = 'publisher' 
    queryset = Publisher.objects.all() 

```

指定`model = Publisher`实际上只是简写为`queryset = Publisher.objects.all()`。 但是，通过使用`queryset`来定义对象的过滤列表，您可以更具体地了解视图中将可见的对象。 举个简单的例子，我们可能想要按出版日期对书籍列表进行排序，最新的排在前面：

```py
from django.views.generic import ListView 
from books.models import Book 

class BookList(ListView): 
    queryset = Book.objects.order_by('-publication_date') 
    context_object_name = 'book_list' 

```

这是一个非常简单的例子，但它很好地说明了这个想法。 当然，您通常希望做的不仅仅是重新排序对象。 如果要显示特定出版商的书籍列表，可以使用相同的技术：

```py
from django.views.generic import ListView 
from books.models import Book 

class AcmeBookList(ListView): 

    context_object_name = 'book_list' 
    queryset = Book.objects.filter(publisher__name='Acme Publishing') 
    template_name = 'books/acme_list.html' 

```

请注意，除了过滤的`queryset`之外，我们还使用了自定义模板名称。 如果没有，通用视图将使用与“普通”对象列表相同的模板，这可能不是我们想要的。

还要注意，这不是一个非常优雅的处理特定出版商书籍的方法。 如果我们想要添加另一个出版商页面，我们需要在 URLconf 中添加另外几行，而且超过几个出版商将变得不合理。 我们将在下一节中解决这个问题。

### 注意

如果在请求`/books/acme/`时收到 404 错误，请检查确保您实际上有一个名称为'ACME Publishing'的出版商。 通用视图具有`allow_empty`参数用于此情况。

# 动态过滤

另一个常见的需求是通过 URL 中的某个键来过滤列表页面中给定的对象。 早些时候，我们在 URLconf 中硬编码了出版商的名称，但是如果我们想编写一个视图，显示某个任意出版商的所有书籍怎么办？

方便的是，`ListView` 有一个我们可以重写的 `get_queryset()` 方法。以前，它只是返回 `queryset` 属性的值，但现在我们可以添加更多逻辑。使这项工作的关键部分是，当调用基于类的视图时，各种有用的东西都存储在 `self` 上；除了请求（`self.request`）之外，还包括根据 URLconf 捕获的位置参数（`self.args`）和基于名称的参数（`self.kwargs`）。

在这里，我们有一个带有单个捕获组的 URLconf：

```py
# urls.py 
from django.conf.urls import url 
from books.views import PublisherBookList 

urlpatterns = [ 
    url(r'^books/([\w-]+)/$', PublisherBookList.as_view()), 
] 

```

接下来，我们将编写 `PublisherBookList` 视图本身：

```py
# views.py 
from django.shortcuts import get_object_or_404 
from django.views.generic import ListView 
from books.models import Book, Publisher 

class PublisherBookList(ListView): 

    template_name = 'books/books_by_publisher.html' 

    def get_queryset(self): 
        self.publisher = get_object_or_404(Publisher name=self.args[0]) 
        return Book.objects.filter(publisher=self.publisher) 

```

正如你所看到的，向查询集选择添加更多逻辑非常容易；如果我们想的话，我们可以使用 `self.request.user` 来使用当前用户进行过滤，或者其他更复杂的逻辑。我们还可以同时将发布者添加到上下文中，这样我们可以在模板中使用它：

```py
# ... 

def get_context_data(self, **kwargs): 
    # Call the base implementation first to get a context 
    context = super(PublisherBookList, self).get_context_data(**kwargs) 

    # Add in the publisher 
    context['publisher'] = self.publisher 
    return context 

```

# 执行额外的工作

我们将看一下最后一个常见模式，它涉及在调用通用视图之前或之后做一些额外的工作。想象一下，我们在我们的 `Author` 模型上有一个 `last_accessed` 字段，我们正在使用它来跟踪任何人最后一次查看该作者的时间：

```py
# models.py 
from django.db import models 

class Author(models.Model): 
    salutation = models.CharField(max_length=10) 
    name = models.CharField(max_length=200) 
    email = models.EmailField() 
    headshot = models.ImageField(upload_to='author_headshots') 
    last_accessed = models.DateTimeField() 

```

当然，通用的 `DetailView` 类不会知道这个字段，但我们可以再次轻松地编写一个自定义视图来保持该字段更新。首先，我们需要在 URLconf 中添加一个作者详细信息，指向一个自定义视图：

```py
from django.conf.urls import url 
from books.views import AuthorDetailView 

urlpatterns = [ 
    #... 
    url(r'^authors/(?P<pk>[0-9]+)/$', AuthorDetailView.as_view(), name='author-detail'), 
] 

```

然后我们会编写我们的新视图 - `get_object` 是检索对象的方法 - 所以我们只需重写它并包装调用：

```py
from django.views.generic import DetailView 
from django.utils import timezone 
from books.models import Author 

class AuthorDetailView(DetailView): 

    queryset = Author.objects.all() 

    def get_object(self): 
        # Call the superclass 
        object = super(AuthorDetailView, self).get_object() 

        # Record the last accessed date 
        object.last_accessed = timezone.now() 
        object.save() 
        # Return the object 
        return object 

```

这里的 URLconf 使用了命名组 `pk` - 这个名称是 `DetailView` 用来查找用于过滤查询集的主键值的默认名称。

如果你想给组起一个别的名字，你可以在视图上设置 `pk_url_kwarg`。更多细节可以在 `DetailView` 的参考中找到。

# 接下来呢？

在这一章中，我们只看了 Django 预装的一些通用视图，但这里提出的一般思想几乎适用于任何通用视图。附录 C，通用视图参考，详细介绍了所有可用的视图，如果你想充分利用这一强大功能，建议阅读。

这结束了本书专门讨论模型、模板和视图的高级用法的部分。接下来的章节涵盖了现代商业网站中非常常见的一系列功能。我们将从构建交互式网站至关重要的主题开始 - 用户管理。
