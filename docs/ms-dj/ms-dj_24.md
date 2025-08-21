# 附录 C. 通用视图参考

第十章 *通用视图*介绍了通用视图，但略去了一些细节。本附录描述了每个通用视图以及每个视图可以采用的选项摘要。在尝试理解接下来的参考资料之前，请务必阅读第十章 *通用视图*。您可能希望参考该章中定义的`Book`、`Publisher`和`Author`对象；后面的示例使用这些模型。如果您想深入了解更高级的通用视图主题（例如在基于类的视图中使用混合），请参阅 Django 项目网站[`docs.djangoproject.com/en/1.8/topics/class-based-views/`](https://docs.djangoproject.com/en/1.8/topics/class-based-views/)。

# 通用视图的常见参数

这些视图大多需要大量的参数，可以改变通用视图的行为。这些参数中的许多在多个视图中起着相同的作用。*表 C.1*描述了每个这些常见参数；每当您在通用视图的参数列表中看到这些参数时，它将按照表中描述的方式工作。 

| **参数** | **描述** |
| --- | --- |
| `allow_empty` | 一个布尔值，指定是否在没有可用对象时显示页面。如果这是`False`并且没有可用对象，则视图将引发 404 错误，而不是显示空页面。默认情况下，这是`True`。 |
| `context_processors` | 要应用于视图模板的附加模板上下文处理器（除了默认值）的列表。有关模板上下文处理器的信息，请参见第九章 *高级模型*。 |
| `extra_context` | 要添加到模板上下文中的值的字典。默认情况下，这是一个空字典。如果字典中的值是可调用的，则通用视图将在呈现模板之前调用它。 |
| `mimetype` | 用于生成文档的 MIME 类型。如果您没有更改它，默认为`DEFAULT_MIME_TYPE`设置的值，即`text/html`。 |
| `queryset` | 从中读取对象的`QuerySet`（例如`Author.objects.all()`）。有关`QuerySet`对象的更多信息，请参见附录 B。大多数通用视图都需要此参数。 |
| `template_loader` | 加载模板时要使用的模板加载程序。默认情况下是`django.template.loader`。有关模板加载程序的信息，请参见第九章 *高级模型*。 |
| `template_name` | 用于呈现页面的模板的完整名称。这使您可以覆盖从`QuerySet`派生的默认模板名称。 |
| `template_object_name` | 模板上下文中要使用的模板变量的名称。默认情况下，这是`'object'`。列出多个对象的视图（即`object_list`视图和各种日期对象视图）将在此参数的值后附加`'_list'`。 |

表 C.1：常见的通用视图参数

# 简单的通用视图

模块`django.views.generic.base`包含处理一些常见情况的简单视图：在不需要视图逻辑时呈现模板和发出重定向。

## 呈现模板-TemplateView

此视图呈现给定模板，传递一个包含在 URL 中捕获的关键字参数的上下文。

**示例：**

给定以下 URLconf：

```py
from django.conf.urls import url 

    from myapp.views import HomePageView 

    urlpatterns = [ 
        url(r'^$', HomePageView.as_view(), name='home'), 
    ] 

```

和一个示例`views.py`：

```py
from django.views.generic.base import TemplateView 
from articles.models import Article 

class HomePageView(TemplateView): 

    template_name = "home.html" 

    def get_context_data(self, **kwargs): 
        context = super(HomePageView, self).get_context_data(**kwargs) 
        context['latest_articles'] = Article.objects.all()[:5] 
        return context 

```

对`/`的请求将呈现模板`home.html`，返回一个包含前 5 篇文章列表的上下文。

## 重定向到另一个 URL

`django.views.generic.base.RedirectView()`将重定向到给定的 URL。

给定的 URL 可能包含类似字典的字符串格式，它将根据在 URL 中捕获的参数进行插值。因为关键字插值*总是*会执行（即使没有传入参数），所以 URL 中的任何"`%`"字符必须写为"`%%`"，以便 Python 将它们转换为输出的单个百分号。

如果给定的 URL 为`None`，Django 将返回一个`HttpResponseGone`（410）。

**示例** **views.py**：

```py
from django.shortcuts import get_object_or_404 

from django.views.generic.base import RedirectView 

from articles.models import Article 

class ArticleCounterRedirectView(RedirectView): 

    permanent = False 
    query_string = True 
    pattern_name = 'article-detail' 

    def get_redirect_url(self, *args, **kwargs): 
        article = get_object_or_404(Article, pk=kwargs['pk']) 
        article.update_counter() 
        return super(ArticleCounterRedirectView,  
                     self).get_redirect_url(*args, **kwargs) 

```

**示例 urls.py**：

```py
from django.conf.urls import url 
from django.views.generic.base import RedirectView 

from article.views import ArticleCounterRedirectView, ArticleDetail 

urlpatterns = [ 
    url(r'^counter/(?P<pk>[0-9]+)/$',  
        ArticleCounterRedirectView.as_view(),  
        name='article-counter'), 
    url(r'^details/(?P<pk>[0-9]+)/$',  
        ArticleDetail.as_view(), 
        name='article-detail'), 
    url(r'^go-to-django/$',  
        RedirectView.as_view(url='http://djangoproject.com'),  
        name='go-to-django'), 
] 

```

### 属性

#### url

要重定向的 URL，作为字符串。或者`None`以引发 410（已消失）HTTP 错误。

#### pattern_name

要重定向到的 URL 模式的名称。将使用与此视图传递的相同的`*args`和`**kwargs`进行反转。

#### 永久

重定向是否应该是永久的。这里唯一的区别是返回的 HTTP 状态代码。如果为`True`，则重定向将使用状态码 301。如果为`False`，则重定向将使用状态码 302。默认情况下，`permanent`为`True`。

#### query_string

是否将 GET 查询字符串传递到新位置。如果为`True`，则查询字符串将附加到 URL。如果为`False`，则查询字符串将被丢弃。默认情况下，`query_string`为`False`。

### 方法

`get_redirect_url(*args, **kwargs)`构造重定向的目标 URL。

默认实现使用`url`作为起始字符串，并使用在 URL 中捕获的命名组执行`%`命名参数的扩展。

如果未设置`url`，`get_redirect_url()`将尝试使用在 URL 中捕获的内容（命名和未命名组都将被使用）来反转`pattern_name`。

如果由`query_string`请求，则还将查询字符串附加到生成的 URL。子类可以实现任何他们希望的行为，只要该方法返回一个准备好的重定向 URL 字符串。

# 列表/详细通用视图

列表/详细通用视图处理在一个视图中显示项目列表的常见情况，并在另一个视图中显示这些项目的单独详细视图。

## 对象列表

```py
django.views.generic.list.ListView 

```

使用此视图显示代表对象列表的页面。

**示例 views.py**：

```py
from django.views.generic.list import ListView 
from django.utils import timezone 

from articles.models import Article 

class ArticleListView(ListView): 

    model = Article 

    def get_context_data(self, **kwargs): 
        context = super(ArticleListView, self).get_context_data(**kwargs) 
        context['now'] = timezone.now() 
        return context 

```

**示例 myapp/urls.py**：

```py
from django.conf.urls import url 

from article.views import ArticleListView 

urlpatterns = [ 
    url(r'^$', ArticleListView.as_view(), name='article-list'), 
] 

```

**示例 myapp/article_list.html**：

```py
<h1>Articles</h1> 
<ul> 
{% for article in object_list %} 
    <li>{{ article.pub_date|date }}-{{ article.headline }}</li> 
{% empty %} 
    <li>No articles yet.</li> 
{% endfor %} 
</ul> 

```

## 详细视图

django.views.generic.detail.DetailView

此视图提供单个对象的详细视图。

**示例 myapp/views.py**：

```py
from django.views.generic.detail import DetailView 
from django.utils import timezone 

from articles.models import Article 

class ArticleDetailView(DetailView): 

    model = Article 

    def get_context_data(self, **kwargs): 
        context = super(ArticleDetailView,  
                        self).get_context_data(**kwargs) 
        context['now'] = timezone.now() 
        return context 

```

**示例 myapp/urls.py**：

```py
from django.conf.urls import url 

from article.views import ArticleDetailView 

urlpatterns = [ 
    url(r'^(?P<slug>[-_\w]+)/$',  
        ArticleDetailView.as_view(),  
        name='article-detail'), 
] 

```

**示例 myapp/article_detail.html**：

```py
<h1>{{ object.headline }}</h1> 
<p>{{ object.content }}</p> 
<p>Reporter: {{ object.reporter }}</p> 
<p>Published: {{ object.pub_date|date }}</p> 
<p>Date: {{ now|date }}</p> 

```

# 基于日期的通用视图

提供在`django.views.generic.dates`中的基于日期的通用视图，用于显示基于日期的数据的钻取页面。

## 存档索引视图

顶级索引页面显示最新的对象，按日期。除非将`allow_future`设置为`True`，否则不包括*未来*日期的对象。

**上下文**

除了`django.views.generic.list.MultipleObjectMixin`提供的上下文（通过`django.views.generic.dates.BaseDateListView`），模板的上下文将是：

+   `date_list`：包含根据`queryset`可用的所有年份的`DateQuerySet`对象，以降序表示为`datetime.datetime`对象

**注意**

+   使用默认的`context_object_name`为`latest`。

+   使用默认的`template_name_suffix`为`_archive`。

+   默认提供`date_list`按年份，但可以使用属性`date_list_period`更改为按月或日。这也适用于所有子类视图：

```py
Example myapp/urls.py: 
from django.conf.urls import url 
from django.views.generic.dates import ArchiveIndexView 

from myapp.models import Article 

urlpatterns = [ 
    url(r'^archive/$', 
        ArchiveIndexView.as_view(model=Article, date_field="pub_date"), 
        name="article_archive"), 
] 

```

**示例 myapp/article_archive.html**：

```py
<ul> 
    {% for article in latest %} 
        <li>{{ article.pub_date }}: {{ article.title }}</li> 
    {% endfor %} 
</ul> 

```

这将输出所有文章。

## YearArchiveView

年度存档页面显示给定年份中所有可用月份。除非将`allow_future`设置为`True`，否则不显示*未来*日期的对象。

**上下文**

除了`django.views.generic.list.MultipleObjectMixin`提供的上下文（通过`django.views.generic.dates.BaseDateListView`），模板的上下文将是：

+   `date_list`：包含根据`queryset`可用的所有月份的`DateQuerySet`对象，以升序表示为`datetime.datetime`对象

+   `year`：表示给定年份的`date`对象

+   `next_year`：表示下一年第一天的`date`对象，根据`allow_empty`和`allow_future`

+   `previous_year`：表示上一年第一天的`date`对象，根据`allow_empty`和`allow_future`

**注释**

+   使用默认的`template_name_suffix`为`_archive_year`

**示例 myapp/views.py**：

```py
from django.views.generic.dates import YearArchiveView 

from myapp.models import Article 

class ArticleYearArchiveView(YearArchiveView): 
    queryset = Article.objects.all() 
    date_field = "pub_date" 
    make_object_list = True 
    allow_future = True 

```

**示例 myapp/urls.py**：

```py
from django.conf.urls import url 

from myapp.views import ArticleYearArchiveView 

urlpatterns = [ 
    url(r'^(?P<year>[0-9]{4})/$', 
        ArticleYearArchiveView.as_view(), 
        name="article_year_archive"), 
] 

```

**示例 myapp/article_archive_year.html**：

```py
<ul> 
    {% for date in date_list %} 
        <li>{{ date|date }}</li> 
    {% endfor %} 
</ul> 
<div> 
    <h1>All Articles for {{ year|date:"Y" }}</h1> 
    {% for obj in object_list %} 
        <p> 
            {{ obj.title }}-{{ obj.pub_date|date:"F j, Y" }} 
        </p> 
    {% endfor %} 
</div> 

```

## 月存档视图

显示给定月份内所有对象的月度存档页面。具有*未来*日期的对象不会显示，除非您将`allow_future`设置为`True`。

**上下文**

除了`MultipleObjectMixin`（通过`BaseDateListView`）提供的上下文之外，模板的上下文将是：

+   `date_list`：包含给定月份中具有可用对象的所有日期的`DateQuerySet`对象，根据`queryset`表示为`datetime.datetime`对象，按升序排列

+   `month`：表示给定月份的`date`对象

+   `next_month`：表示下个月第一天的`date`对象，根据`allow_empty`和`allow_future`

+   `previous_month`：表示上个月第一天的`date`对象，根据`allow_empty`和`allow_future`

**注释**

+   使用默认的`template_name_suffix`为`_archive_month`

**示例 myapp/views.py**：

```py
from django.views.generic.dates import MonthArchiveView 

from myapp.models import Article 

class ArticleMonthArchiveView(MonthArchiveView): 
    queryset = Article.objects.all() 
    date_field = "pub_date" 
    make_object_list = True 
    allow_future = True 

```

**示例 myapp/urls.py**：

```py
from django.conf.urls import url 

from myapp.views import ArticleMonthArchiveView 

urlpatterns = [ 
    # Example: /2012/aug/ 
    url(r'^(?P<year>[0-9]{4})/(?P<month>[-\w]+)/$', 
        ArticleMonthArchiveView.as_view(), 
        name="archive_month"), 
    # Example: /2012/08/ 
    url(r'^(?P<year>[0-9]{4})/(?P<month>[0-9]+)/$', 
        ArticleMonthArchiveView.as_view(month_format='%m'), 
        name="archive_month_numeric"), 
] 

```

**示例 myapp/article_archive_month.html**：

```py
<ul> 
    {% for article in object_list %} 
        <li>{{ article.pub_date|date:"F j, Y" }}:  
            {{ article.title }} 
        </li> 
    {% endfor %} 
</ul> 

<p> 
    {% if previous_month %} 
        Previous Month: {{ previous_month|date:"F Y" }} 
    {% endif %} 
    {% if next_month %} 
        Next Month: {{ next_month|date:"F Y" }} 
    {% endif %} 
</p> 

```

## 周存档视图

显示给定周内所有对象的周存档页面。具有*未来*日期的对象不会显示，除非您将`allow_future`设置为`True`。

**上下文**

除了`MultipleObjectMixin`（通过`BaseDateListView`）提供的上下文之外，模板的上下文将是：

+   `week`：表示给定周的第一天的`date`对象

+   `next_week`：表示下周第一天的`date`对象，根据`allow_empty`和`allow_future`

+   `previous_week`：表示上周第一天的`date`对象，根据`allow_empty`和`allow_future`

**注释**

+   使用默认的`template_name_suffix`为`_archive_week`

**示例 myapp/views.py**：

```py
from django.views.generic.dates import WeekArchiveView 

from myapp.models import Article 

class ArticleWeekArchiveView(WeekArchiveView): 
    queryset = Article.objects.all() 
    date_field = "pub_date" 
    make_object_list = True 
    week_format = "%W" 
    allow_future = True 

```

**示例 myapp/urls.py**：

```py
from django.conf.urls import url 

from myapp.views import ArticleWeekArchiveView 

urlpatterns = [ 
    # Example: /2012/week/23/ 
    url(r'^(?P<year>[0-9]{4})/week/(?P<week>[0-9]+)/$', 
        ArticleWeekArchiveView.as_view(), 
        name="archive_week"), 
] 

```

**示例 myapp/article_archive_week.html**：

```py
<h1>Week {{ week|date:'W' }}</h1> 

<ul> 
    {% for article in object_list %} 
        <li>{{ article.pub_date|date:"F j, Y" }}: {{ article.title }}</li> 
    {% endfor %} 
</ul> 

<p> 
    {% if previous_week %} 
        Previous Week: {{ previous_week|date:"F Y" }} 
    {% endif %} 
    {% if previous_week and next_week %}--{% endif %} 
    {% if next_week %} 
        Next week: {{ next_week|date:"F Y" }} 
    {% endif %} 
</p> 

```

在这个例子中，您正在输出周数。`WeekArchiveView`中的默认`week_format`使用基于美国周系统的周格式"`%U`"，其中周从星期日开始。"`%W`"格式使用 ISO 周格式，其周从星期一开始。"`%W`"格式在`strftime()`和`date`中是相同的。

但是，`date`模板过滤器没有支持基于美国周系统的等效输出格式。`date`过滤器"`%U`"输出自 Unix 纪元以来的秒数。

## 日存档视图

显示给定日期内所有对象的日存档页面。未来的日期会抛出 404 错误，无论未来日期是否存在任何对象，除非您将`allow_future`设置为`True`。

**上下文**

除了`MultipleObjectMixin`（通过`BaseDateListView`）提供的上下文之外，模板的上下文将是：

+   `day`：表示给定日期的`date`对象

+   `next_day`：表示下一天的`date`对象，根据`allow_empty`和`allow_future`

+   `previous_day`：表示前一天的`date`对象，根据`allow_empty`和`allow_future`

+   `next_month`：表示下个月第一天的`date`对象，根据`allow_empty`和`allow_future`

+   `previous_month`：表示上个月第一天的`date`对象，根据`allow_empty`和`allow_future`

**注释**

+   使用默认的`template_name_suffix`为`_archive_day`

**示例 myapp/views.py**：

```py
from django.views.generic.dates import DayArchiveView 

from myapp.models import Article 

class ArticleDayArchiveView(DayArchiveView): 
    queryset = Article.objects.all() 
    date_field = "pub_date" 
    make_object_list = True 
    allow_future = True 

```

**示例 myapp/urls.py**：

```py
from django.conf.urls import url 

from myapp.views import ArticleDayArchiveView 

urlpatterns = [ 
    # Example: /2012/nov/10/ 
    url(r'^(?P<year>[0-9]{4})/(?P<month>[-\w]+)/(?P<day>[0-9]+)/$', 
        ArticleDayArchiveView.as_view(), 
        name="archive_day"), 
] 

```

**示例 myapp/article_archive_day.html**：

```py
<h1>{{ day }}</h1> 

<ul> 
    {% for article in object_list %} 
        <li> 
        {{ article.pub_date|date:"F j, Y" }}: {{ article.title }} 
        </li> 
    {% endfor %} 
</ul> 

<p> 
    {% if previous_day %} 
        Previous Day: {{ previous_day }} 
    {% endif %} 
    {% if previous_day and next_day %}--{% endif %} 
    {% if next_day %} 
        Next Day: {{ next_day }} 
    {% endif %} 
</p> 

```

## 今天存档视图

显示*今天*的所有对象的日存档页面。这与`django.views.generic.dates.DayArchiveView`完全相同，只是使用今天的日期而不是`year`/`month`/`day`参数。

**注释**

+   使用默认的`template_name_suffix`为`_archive_today`

**示例 myapp/views.py**：

```py
from django.views.generic.dates import TodayArchiveView 

from myapp.models import Article 

class ArticleTodayArchiveView(TodayArchiveView): 
    queryset = Article.objects.all() 
    date_field = "pub_date" 
    make_object_list = True 
    allow_future = True 

```

**示例 myapp/urls.py**：

```py
from django.conf.urls import url 

from myapp.views import ArticleTodayArchiveView 

urlpatterns = [ 
    url(r'^today/$', 
        ArticleTodayArchiveView.as_view(), 
        name="archive_today"), 
] 

```

`TodayArchiveView`的示例模板在哪里？

此视图默认使用与上一个示例中的`DayArchiveView`相同的模板。如果需要不同的模板，请将`template_name`属性设置为新模板的名称。

## DateDetailView

表示单个对象的页面。如果对象具有未来的日期值，默认情况下视图将抛出 404 错误，除非您将`allow_future`设置为`True`。

**上下文**

+   包括与`DateDetailView`中指定的`model`相关联的单个对象

**注**

+   使用默认的`template_name_suffix`为`_detail`

```py
Example myapp/urls.py: 
from django.conf.urls import url 
from django.views.generic.dates import DateDetailView 

urlpatterns = [ 
    url(r'^(?P<year>[0-9]+)/(?P<month>[-\w]+)/(?P<day>[0-9]+)/ 
      (?P<pk>[0-9]+)/$', 
        DateDetailView.as_view(model=Article, date_field="pub_date"), 
        name="archive_date_detail"), 
] 

```

**示例 myapp/article_detail.html**：

```py
<h1>{{ object.title }}</h1> 

```

# 使用基于类的视图处理表单

表单处理通常有 3 条路径：

+   初始`GET`（空白或预填充表单）

+   `POST`无效数据（通常重新显示带有错误的表单）

+   `POST`有效数据（处理数据并通常重定向）

自己实现这个通常会导致大量重复的样板代码（请参见在视图中使用表单）。为了避免这种情况，Django 提供了一组用于表单处理的通用基于类的视图。

## 基本表单

给定一个简单的联系表单：

```py
# forms.py 

from django import forms 

class ContactForm(forms.Form): 
   name = forms.CharField() 
   message = forms.CharField(widget=forms.Textarea) 

   def send_email(self): 
       # send email using the self.cleaned_data dictionary 
       pass 

```

可以使用`FormView`构建视图：

```py
# views.py 

from myapp.forms import ContactForm 
from django.views.generic.edit import FormView 

class ContactView(FormView): 
   template_name = 'contact.html' 
   form_class = ContactForm 
   success_url = '/thanks/' 

   def form_valid(self, form): 
       # This method is called when valid form data has been POSTed. 
       # It should return an HttpResponse. 
       form.send_email() 
       return super(ContactView, self).form_valid(form) 

```

注：

+   `FormView`继承了`TemplateResponseMixin`，因此`template_name`可以在这里使用

+   `form_valid()`的默认实现只是重定向到`success_url`

## 模型表单

与模型一起工作时，通用视图真正发挥作用。这些通用视图将自动创建`ModelForm`，只要它们可以确定要使用哪个模型类：

+   如果给定了`model`属性，将使用该模型类

+   如果`get_object()`返回一个对象，将使用该对象的类

+   如果给定了`queryset`，将使用该查询集的模型

模型表单视图提供了一个`form_valid()`实现，可以自动保存模型。如果有特殊要求，可以覆盖此功能；请参阅下面的示例。

对于`CreateView`或`UpdateView`，甚至不需要提供`success_url`-如果可用，它们将使用模型对象上的`get_absolute_url()`。

如果要使用自定义的`ModelForm`（例如添加额外的验证），只需在视图上设置`form_class`。

### 注意

在指定自定义表单类时，仍然必须指定模型，即使`form_class`可能是一个 ModelForm。

首先，我们需要在我们的`Author`类中添加`get_absolute_url()`：

```py
# models.py 

from django.core.urlresolvers import reverse 
from django.db import models 

class Author(models.Model): 
    name = models.CharField(max_length=200) 

    def get_absolute_url(self): 
        return reverse('author-detail', kwargs={'pk': self.pk}) 

```

然后我们可以使用`CreateView`和其他视图来执行实际工作。请注意，我们只是在这里配置通用基于类的视图；我们不必自己编写任何逻辑：

```py
# views.py 

from django.views.generic.edit import CreateView, UpdateView, DeleteView 
from django.core.urlresolvers import reverse_lazy 
from myapp.models import Author 

class AuthorCreate(CreateView): 
    model = Author 
    fields = ['name'] 

class AuthorUpdate(UpdateView): 
    model = Author 
    fields = ['name'] 

class AuthorDelete(DeleteView): 
    model = Author 
    success_url = reverse_lazy('author-list') 

```

我们必须在这里使用`reverse_lazy()`，而不仅仅是`reverse`，因为在导入文件时未加载 URL。

`fields`属性的工作方式与`ModelForm`上内部`Meta`类的`fields`属性相同。除非以其他方式定义表单类，否则该属性是必需的，如果没有，视图将引发`ImproperlyConfigured`异常。

如果同时指定了`fields`和`form_class`属性，将引发`ImproperlyConfigured`异常。

最后，我们将这些新视图挂接到 URLconf 中：

```py
# urls.py 

from django.conf.urls import url 
from myapp.views import AuthorCreate, AuthorUpdate, AuthorDelete 

urlpatterns = [ 
    # ... 
    url(r'author/add/$', AuthorCreate.as_view(), name='author_add'), 
    url(r'author/(?P<pk>[0-9]+)/$', AuthorUpdate.as_view(),   
        name='author_update'), 
    url(r'author/(?P<pk>[0-9]+)/delete/$', AuthorDelete.as_view(),  
        name='author_delete'), 
] 

```

在这个例子中：

+   `CreateView`和`UpdateView`使用`myapp/author_form.html`

+   `DeleteView`使用`myapp/author_confirm_delete.html`

如果您希望为`CreateView`和`UpdateView`设置单独的模板，可以在视图类上设置`template_name`或`template_name_suffix`。

## 模型和 request.user

要跟踪使用`CreateView`创建对象的用户，可以使用自定义的`ModelForm`来实现。首先，将外键关系添加到模型中：

```py
# models.py 

from django.contrib.auth.models import User 
from django.db import models 

class Author(models.Model): 
    name = models.CharField(max_length=200) 
    created_by = models.ForeignKey(User) 

    # ... 

```

在视图中，确保不要在要编辑的字段列表中包含`created_by`，并覆盖`form_valid()`以添加用户：

```py
# views.py 

from django.views.generic.edit import CreateView 
from myapp.models import Author 

class AuthorCreate(CreateView): 
    model = Author 
    fields = ['name'] 

    def form_valid(self, form): 
        form.instance.created_by = self.request.user 
        return super(AuthorCreate, self).form_valid(form) 

```

请注意，您需要使用`login_required()`装饰此视图，或者在`form_valid()`中处理未经授权的用户。

## AJAX 示例

这里是一个简单的示例，展示了如何实现一个既适用于 AJAX 请求又适用于*普通*表单`POST`的表单。

```py
from django.http import JsonResponse 
from django.views.generic.edit import CreateView 
from myapp.models import Author 

class AjaxableResponseMixin(object): 
    def form_invalid(self, form): 
        response = super(AjaxableResponseMixin, self).form_invalid(form) 
        if self.request.is_ajax(): 
            return JsonResponse(form.errors, status=400) 
        else: 
            return response 

    def form_valid(self, form): 
        # We make sure to call the parent's form_valid() method because 
        # it might do some processing (in the case of CreateView, it will 
        # call form.save() for example). 
        response = super(AjaxableResponseMixin, self).form_valid(form) 
        if self.request.is_ajax(): 
            data = { 
                'pk': self.object.pk, 
            } 
            return JsonResponse(data) 
        else: 
            return response 

class AuthorCreate(AjaxableResponseMixin, CreateView): 
    model = Author 
    fields = ['name'] 

```
