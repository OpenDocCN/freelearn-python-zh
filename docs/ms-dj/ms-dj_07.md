# 第七章：高级视图和 URLconfs

在第二章*视图和 URLconfs*中，我们解释了 Django 的视图函数和 URLconfs 的基础知识。本章将更详细地介绍框架中这两个部分的高级功能。

# URLconf 提示和技巧

URLconfs 没有什么特别的-就像 Django 中的其他任何东西一样，它们只是 Python 代码。您可以以几种方式利用这一点，如下面的部分所述。

## 简化函数导入

考虑这个 URLconf，它基于第二章*视图和 URLconfs*中的示例构建：

```py
from django.conf.urls import include, url 
from django.contrib import admin 
from mysite.views import hello, current_datetime, hours_ahead 

urlpatterns = [ 
      url(r'^admin/', include(admin.site.urls)), 
      url(r'^hello/$', hello), 
      url(r'^time/$', current_datetime), 
      url(r'^time/plus/(\d{1,2})/$', hours_ahead), 
      ] 

```

如第二章*视图和 URLconfs*中所述，URLconf 中的每个条目都包括其关联的视图函数，直接作为函数对象传递。这意味着需要在模块顶部导入视图函数。

但是随着 Django 应用程序的复杂性增加，其 URLconf 也会增加，并且保持这些导入可能很繁琐。 （对于每个新的视图函数，您必须记住导入它，并且如果使用这种方法，导入语句往往会变得过长。）

可以通过导入`views`模块本身来避免这种单调。这个示例 URLconf 等同于前一个：

```py
from django.conf.urls import include, url 
from . import views 

urlpatterns = [ 
         url(r'^hello/$', views.hello), 
         url(r'^time/$', views.current_datetime), 
         url(r'^time/plus/(d{1,2})/$', views.hours_ahead), 
] 

```

## 在调试模式下特殊处理 URL

说到动态构建`urlpatterns`，您可能希望利用这种技术来在 Django 的调试模式下更改 URLconf 的行为。为此，只需在运行时检查`DEBUG`设置的值，如下所示：

```py
from django.conf import settings 
from django.conf.urls import url 
from . import views 

urlpatterns = [ 
    url(r'^$', views.homepage), 
    url(r'^(\d{4})/([a-z]{3})/$', views.archive_month), 
] 

if settings.DEBUG: 
 urlpatterns += [url(r'^debuginfo/$', views.debug),]

```

在这个例子中，只有当您的`DEBUG`设置为`True`时，URL`/debuginfo/`才可用。

## 命名组预览

上面的示例使用简单的非命名正则表达式组（通过括号）来捕获 URL 的部分并将它们作为位置参数传递给视图。

在更高级的用法中，可以使用命名的正则表达式组来捕获 URL 部分并将它们作为关键字参数传递给视图。

在 Python 正则表达式中，命名正则表达式组的语法是`(?P<name>pattern)`，其中`name`是组的名称，`pattern`是要匹配的某个模式。

例如，假设我们在我们的书籍网站上有一系列书评，并且我们想要检索特定日期或日期范围的书评。

这是一个示例 URLconf：

```py
from django.conf.urls import url 

from . import views 

urlpatterns = [ 
    url(r'^reviews/2003/$', views.special_case_2003), 
    url(r'^reviews/([0-9]{4})/$', views.year_archive), 
    url(r'^reviews/([0-9]{4})/([0-9]{2})/$', views.month_archive), 
    url(r'^reviews/([0-9]{4})/([0-9]{2})/([0-9]+)/$', views.review_detail), 
] 

```

### 提示

**注意：**

要从 URL 中捕获一个值，只需在其周围加括号。不需要添加一个前导斜杠，因为每个 URL 都有。例如，它是`^reviews`，而不是`^/reviews`。

每个正则表达式字符串前面的`'r'`是可选的，但建议使用。它告诉 Python 字符串是原始的，字符串中的任何内容都不应该被转义。

**示例请求：**

+   对`/reviews/2005/03/`的请求将匹配列表中的第三个条目。Django 将调用函数`views.month_archive(request,``'2005',``'03')`。

+   `/reviews/2005/3/`不会匹配任何 URL 模式，因为列表中的第三个条目要求月份需要两位数字。

+   `/reviews/2003/`将匹配列表中的第一个模式，而不是第二个模式，因为模式是按顺序测试的，第一个模式是第一个通过的测试。可以随意利用排序来插入这样的特殊情况。

+   `/reviews/2003`不会匹配这些模式中的任何一个，因为每个模式都要求 URL 以斜杠结尾。

+   `/reviews/2003/03/03/`将匹配最终模式。Django 将调用函数`views.review_detail(request,``'2003',``'03',``'03')`。

以下是上面的示例 URLconf，重写以使用命名组：

```py
from django.conf.urls import url 

from . import views 

urlpatterns = [ 
    url(r'^reviews/2003/$', views.special_case_2003), 
    url(r'^reviews/(?P<year>[0-9]{4})/$', views.year_archive), 
    url(r'^reviews/(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/$', views.month_archive), 
    url(r'^reviews/(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/(?P<day>[0-9]{2})/$', views.review_detail), 
] 

```

这与前面的示例完全相同，只有一个细微的区别：捕获的值作为关键字参数传递给视图函数，而不是作为位置参数。例如：

+   对`/reviews/2005/03/`的请求将调用函数`views.month_archive(request,``year='2005',``month='03')`，而不是`views.month_archive(request,``'2005',``'03')`。

+   对`/reviews/2003/03/03/`的请求将调用函数`views.review_detail(request,``year='2003',``month='03',``day='03')`。

实际上，这意味着您的 URLconf 更加明确，不太容易出现参数顺序错误-您可以重新排列视图函数定义中的参数。当然，这些好处是以简洁为代价的；一些开发人员认为命名组语法难看且过于冗长。

### 匹配/分组算法

以下是 URLconf 解析器遵循的算法，关于正则表达式中的命名组与非命名组：

1.  如果有任何命名参数，它将使用这些参数，忽略非命名参数。

1.  否则，它将把所有非命名参数作为位置参数传递。

在这两种情况下，任何给定的额外关键字参数也将传递给视图。

## URLconf 搜索的内容

URLconf 会针对请求的 URL 进行搜索，作为普通的 Python 字符串。这不包括`GET`或`POST`参数，也不包括域名。例如，在对`http://www.example.com/myapp/`的请求中，URLconf 将查找`myapp/`。在对`http://www.example.com/myapp/?page=3`的请求中，URLconf 将查找`myapp/`。URLconf 不会查看请求方法。换句话说，所有请求方法-`POST`、`GET`、`HEAD`等等-都将被路由到相同的函数以处理相同的 URL。

## 捕获的参数始终是字符串

每个捕获的参数都作为普通的 Python 字符串发送到视图中，无论正则表达式的匹配类型如何。例如，在这个 URLconf 行中：

```py
url(r'^reviews/(?P<year>[0-9]{4})/$', views.year_archive), 

```

...`views.year_archive()`的`year`参数将是一个字符串，而不是一个整数，即使`[0-9]{4}`只匹配整数字符串。

## 指定视图参数的默认值

一个方便的技巧是为视图的参数指定默认参数。以下是一个示例 URLconf：

```py
# URLconf 
from django.conf.urls import url 

from . import views 

urlpatterns = [ 
    url(r'^reviews/$', views.page), 
    url(r'^reviews/page(?P<num>[0-9]+)/$', views.page), 
] 

# View (in reviews/views.py) 
def page(request, num="1"): 
    # Output the appropriate page of review entries, according to num. 
    ... 

```

在上面的示例中，两个 URL 模式都指向相同的视图-`views.page`-但第一个模式不会从 URL 中捕获任何内容。如果第一个模式匹配，`page()`函数将使用其默认参数`num`，即`"1"`。如果第二个模式匹配，`page()`将使用正则表达式捕获的`num`值。

### 注意

**关键字参数 vs. 位置参数**

Python 函数可以使用关键字参数或位置参数调用-在某些情况下，两者同时使用。在关键字参数调用中，您指定要传递的参数的名称以及值。在位置参数调用中，您只需传递参数，而不明确指定哪个参数匹配哪个值；关联是在参数的顺序中隐含的。例如，考虑这个简单的函数：

`def sell(item, price, quantity): print "以%s 的价格出售%s 个单位的%s" % (quantity, item, price)`

要使用位置参数调用它，您需要按照函数定义中列出的顺序指定参数：`sell('Socks', '$2.50', 6)`

要使用关键字参数调用它，您需要指定参数的名称以及值。以下语句是等效的：`sell(item='Socks', price='$2.50', quantity=6)` `sell(item='Socks', quantity=6, price='$2.50')` `sell(price='$2.50', item='Socks', quantity=6)` `sell(price='$2.50', quantity=6, item='Socks')` `sell(quantity=6, item='Socks', price='$2.50')` `sell(quantity=6, price='$2.50', item='Socks')`

最后，您可以混合使用关键字和位置参数，只要所有位置参数在关键字参数之前列出。以下语句与前面的示例等效：`sell('Socks', '$2.50', quantity=6)` `sell('Socks', price='$2.50', quantity=6)` `sell('Socks', quantity=6, price='$2.50')`

# 性能

`urlpatterns`中的每个正则表达式在第一次访问时都会被编译。这使得系统运行非常快。

# 错误处理

当 Django 找不到与请求的 URL 匹配的正则表达式，或者当引发异常时，Django 将调用一个错误处理视图。用于这些情况的视图由四个变量指定。这些变量是：

+   `handler404`

+   `handler500`

+   `handler403`

+   `handler400`

它们的默认值对于大多数项目应该足够了，但可以通过为它们分配值来进一步定制。这些值可以在您的根 URLconf 中设置。在任何其他 URLconf 中设置这些变量都不会产生效果。值必须是可调用的，或者是表示应该被调用以处理当前错误条件的视图的完整 Python 导入路径的字符串。

# 包含其他 URLconfs

在任何时候，您的 `urlpatterns` 可以包括其他 URLconf 模块。这实质上将一组 URL 根据其他 URL 的下方。例如，这是 Django 网站本身的 URLconf 的摘录。它包括许多其他 URLconfs：

```py
from django.conf.urls import include, url 

urlpatterns = [ 
    # ... 
    url(r'^community/', include('django_website.aggregator.urls')), 
    url(r'^contact/', include('django_website.contact.urls')), 
    # ... 
] 

```

请注意，此示例中的正则表达式没有 `$`（字符串结束匹配字符），但包括一个尾随斜杠。每当 Django 遇到 `include()` 时，它会截掉到目前为止匹配的 URL 的任何部分，并将剩余的字符串发送到包含的 URLconf 进行进一步处理。另一个可能性是通过使用 `url()` 实例的列表来包含其他 URL 模式。例如，考虑这个 URLconf：

```py
from django.conf.urls import include, url 
from apps.main import views as main_views 
from credit import views as credit_views 

extra_patterns = [ 
    url(r'^reports/(?P<id>[0-9]+)/$', credit_views.report), 
    url(r'^charge/$', credit_views.charge), 
] 

urlpatterns = [ 
    url(r'^$', main_views.homepage), 
    url(r'^help/', include('apps.help.urls')), 
    url(r'^credit/', include(extra_patterns)), 
] 

```

在这个例子中，`/credit/reports/` URL 将由 `credit.views.report()` Django 视图处理。这可以用来消除 URLconfs 中重复使用单个模式前缀的冗余。例如，考虑这个 URLconf：

```py
from django.conf.urls import url 
from . import views 

urlpatterns = [ 
    url(r'^(?P<page_slug>\w+)-(?P<page_id>\w+)/history/$',   
        views.history), 
    url(r'^(?P<page_slug>\w+)-(?P<page_id>\w+)/edit/$', views.edit), 
    url(r'^(?P<page_slug>\w+)-(?P<page_id>\w+)/discuss/$',   
        views.discuss), 
    url(r'^(?P<page_slug>\w+)-(?P<page_id>\w+)/permissions/$',  
        views.permissions), 
] 

```

我们可以通过仅声明共同的路径前缀一次并分组不同的后缀来改进这一点：

```py
from django.conf.urls import include, url 
from . import views 

urlpatterns = [ 
    url(r'^(?P<page_slug>\w+)-(?P<page_id>\w+)/',  
        include([ 
        url(r'^history/$', views.history), 
        url(r'^edit/$', views.edit), 
        url(r'^discuss/$', views.discuss), 
        url(r'^permissions/$', views.permissions), 
        ])), 
] 

```

## 捕获的参数

包含的 URLconf 会接收来自父 URLconfs 的任何捕获的参数，因此以下示例是有效的：

```py
# In settings/urls/main.py 
from django.conf.urls import include, url 

urlpatterns = [ 
    url(r'^(?P<username>\w+)/reviews/', include('foo.urls.reviews')), 
] 

# In foo/urls/reviews.py 
from django.conf.urls import url 
from . import views 

urlpatterns = [ 
    url(r'^$', views.reviews.index), 
    url(r'^archive/$', views.reviews.archive), 
] 

```

在上面的示例中，捕获的 `"username"` 变量如预期地传递给了包含的 URLconf。

# 向视图函数传递额外选项

URLconfs 具有一个钩子，可以让您将额外的参数作为 Python 字典传递给视图函数。`django.conf.urls.url()` 函数可以接受一个可选的第三个参数，应该是一个额外关键字参数的字典，用于传递给视图函数。例如：

```py
from django.conf.urls import url 
from . import views 

urlpatterns = [ 
    url(r'^reviews/(?P<year>[0-9]{4})/$',  
        views.year_archive,  
        {'foo': 'bar'}), 
] 

```

在这个例子中，对于对 `/reviews/2005/` 的请求，Django 将调用 `views.year_archive(request,` `year='2005',` `foo='bar')`。这种技术在辅助框架中用于向视图传递元数据和选项（参见第十四章，“生成非 HTML 内容”）。

### 注意

**处理冲突**

可能会有一个 URL 模式，它捕获了命名的关键字参数，并且还在其额外参数的字典中传递了相同名称的参数。当这种情况发生时，字典中的参数将被用于替代 URL 中捕获的参数。

## 向 include() 传递额外的选项

同样，您可以向 `include()` 传递额外的选项。当您向 `include()` 传递额外的选项时，包含的 URLconf 中的每一行都将传递额外的选项。例如，这两个 URLconf 集是功能上相同的：集合一：

```py
# main.py 
from django.conf.urls import include, url 

urlpatterns = [ 
    url(r'^reviews/', include('inner'), {'reviewid': 3}), 
] 

# inner.py 
from django.conf.urls import url 
from mysite import views 

urlpatterns = [ 
    url(r'^archive/$', views.archive), 
    url(r'^about/$', views.about), 
] 

```

集合二：

```py
# main.py 
from django.conf.urls import include, url 
from mysite import views 

urlpatterns = [ 
    url(r'^reviews/', include('inner')), 
] 

# inner.py 
from django.conf.urls import url 

urlpatterns = [ 
    url(r'^archive/$', views.archive, {'reviewid': 3}), 
    url(r'^about/$', views.about, {'reviewid': 3}), 
] 

```

请注意，无论包含的 URLconf 中的视图是否实际接受这些选项作为有效选项，额外的选项都将始终传递给包含的 URLconf 中的每一行。因此，只有在您确定包含的 URLconf 中的每个视图都接受您传递的额外选项时，这种技术才有用。

# URL 的反向解析

在开发 Django 项目时通常需要的是获取 URL 的最终形式，无论是用于嵌入生成的内容（视图和资源 URL，向用户显示的 URL 等）还是用于服务器端的导航流程处理（重定向等）

强烈建议避免硬编码这些 URL（一种费力、不可扩展和容易出错的策略）或者不得不设计专门的机制来生成与 URLconf 描述的设计并行的 URL，因此有可能在某个时刻产生过时的 URL。换句话说，需要的是一种 DRY 机制。

除了其他优点，它还允许 URL 设计的演变，而无需在整个项目源代码中搜索和替换过时的 URL。我们可以作为获取 URL 的起点的信息是处理它的视图的标识（例如名称），必须参与查找正确 URL 的其他信息是视图参数的类型（位置，关键字）和值。

Django 提供了一种解决方案，即 URL 映射器是 URL 设计的唯一存储库。您可以用 URLconf 提供给它，然后可以在两个方向上使用它： 

+   从用户/浏览器请求的 URL 开始，它调用正确的 Django 视图，并提供可能需要的任何参数及其值，这些值是从 URL 中提取的。

+   从对应的 Django 视图的标识开始，以及将传递给它的参数的值，获取相关联的 URL。

第一个是我们在前几节中讨论的用法。第二个是所谓的**URL 的反向解析**，**反向 URL 匹配**，**反向 URL 查找**或简称**URL 反转**。

Django 提供了执行 URL 反转的工具，这些工具与需要 URL 的不同层次匹配：

+   在模板中：使用`url`模板标签。

+   在 Python 代码中：使用`django.core.urlresolvers.reverse()`函数。

+   与 Django 模型实例的 URL 处理相关的高级代码：`get_absolute_url()`方法。

## 示例

再次考虑这个 URLconf 条目：

```py
from django.conf.urls import url 
from . import views 

urlpatterns = [ 
    #... 
    url(r'^reviews/([0-9]{4})/$', views.year_archive,  
        name='reviews-year-archive'), 
    #... 
] 

```

根据这个设计，对应于年份**nnnn**的存档的 URL 是`/reviews/nnnn/`。您可以通过在模板代码中使用以下方式来获取这些：

```py
<a href="{% url 'reviews-year-archive' 2012 %}">2012 Archive</a> 
{# Or with the year in a template context variable: #} 

<ul> 
{% for yearvar in year_list %} 
<li><a href="{% url 'reviews-year-archive' yearvar %}">{{ yearvar }} Archive</a></li> 
{% endfor %} 
</ul> 

```

或者在 Python 代码中：

```py
from django.core.urlresolvers import reverse 
from django.http import HttpResponseRedirect 

def redirect_to_year(request): 
    # ... 
    year = 2012 
    # ... 
    return HttpResponseRedirect(reverse('reviews-year-archive', args=(year,))) 

```

如果出于某种原因，决定更改发布年度审查存档内容的 URL，则只需要更改 URLconf 中的条目。在某些情况下，如果视图具有通用性质，则 URL 和视图之间可能存在多对一的关系。对于这些情况，当需要反转 URL 时，视图名称并不是足够好的标识符。阅读下一节以了解 Django 为此提供的解决方案。

# 命名 URL 模式

为了执行 URL 反转，您需要使用上面示例中所做的命名 URL 模式。用于 URL 名称的字符串可以包含任何您喜欢的字符。您不受限于有效的 Python 名称。当您命名您的 URL 模式时，请确保使用不太可能与任何其他应用程序选择的名称冲突的名称。如果您称呼您的 URL 模式为`comment`，另一个应用程序也这样做，那么当您使用这个名称时，无法保证将插入哪个 URL 到您的模板中。在您的 URL 名称上加上前缀，可能来自应用程序名称，将减少冲突的机会。我们建议使用`myapp-comment`而不是`comment`之类的东西。

# URL 命名空间

URL 命名空间允许您唯一地反转命名的 URL 模式，即使不同的应用程序使用相同的 URL 名称。对于第三方应用程序来说，始终使用命名空间 URL 是一个好习惯。同样，它还允许您在部署多个应用程序实例时反转 URL。换句话说，由于单个应用程序的多个实例将共享命名的 URL，命名空间提供了一种区分这些命名的 URL 的方法。

正确使用 URL 命名空间的 Django 应用程序可以针对特定站点部署多次。例如，`django.contrib.admin`有一个`AdminSite`类，允许您轻松部署多个管理员实例。URL 命名空间由两部分组成，两者都是字符串：

1.  **应用程序命名空间**：描述正在部署的应用程序的名称。单个应用程序的每个实例都将具有相同的应用程序命名空间。例如，Django 的管理员应用程序具有相对可预测的应用程序命名空间`admin`。

1.  **实例命名空间**：标识应用程序的特定实例。实例命名空间应该在整个项目中是唯一的。但是，实例命名空间可以与应用程序命名空间相同。这用于指定应用程序的默认实例。例如，默认的 Django 管理员实例具有`admin`的实例命名空间。

使用`:`运算符指定命名空间 URL。例如，管理员应用程序的主索引页面使用"`admin:index`"引用。这表示命名空间为"`admin`"，命名为"`index`"。

命名空间也可以是嵌套的。命名为`members:reviews:index`的 URL 将在顶级命名空间`members`中查找名为"`index`"的模式。

## 反转命名空间 URL

在给定要解析的命名空间 URL（例如"`reviews:index`"）时，Django 将完全限定的名称分成部分，然后尝试以下查找：

1.  首先，Django 会查找匹配的应用程序命名空间（在本例中为`reviews`）。这将产生该应用程序的实例列表。

1.  如果定义了当前应用程序，Django 会查找并返回该实例的 URL 解析器。当前应用程序可以作为请求的属性指定。期望有多个部署的应用程序应该在正在处理的请求上设置`current_app`属性。

1.  当前应用程序也可以作为`reverse()`函数的参数手动指定。

1.  如果没有当前应用程序。 Django 将寻找默认的应用程序实例。默认的应用程序实例是具有与应用程序命名空间匹配的实例命名空间的实例（在本例中，称为"`reviews`"的 reviews 的实例）。

1.  如果没有默认的应用程序实例，Django 将选择应用程序的最后部署实例，无论其实例名称是什么。

1.  如果提供的命名空间与第 1 步中的应用程序命名空间不匹配，Django 将尝试直接查找该命名空间作为实例命名空间。

如果有嵌套的命名空间，这些步骤将针对命名空间的每个部分重复，直到只剩下视图名称未解析。然后，视图名称将被解析为在找到的命名空间中的 URL。

## URL 命名空间和包含的 URLconfs

包含的 URLconfs 的 URL 命名空间可以通过两种方式指定。首先，当构建 URL 模式时，您可以将应用程序和实例命名空间作为参数提供给`include()`。例如：

```py
url(r'^reviews/', include('reviews.urls', namespace='author-reviews', 
    app_name='reviews')), 

```

这将包括在应用程序命名空间'reviews'中定义的 URL，实例命名空间为'author-reviews'。其次，您可以包含包含嵌入式命名空间数据的对象。如果您包含一个`url()`实例列表，那么该对象中包含的 URL 将被添加到全局命名空间中。但是，您也可以包含一个包含 3 个元素的元组：

```py
(<list of url() instances>, <application namespace>, <instance namespace>) 

```

例如：

```py
from django.conf.urls import include, url 

from . import views 

reviews_patterns = [ 
    url(r'^$', views.IndexView.as_view(), name='index'), 
    url(r'^(?P<pk>\d+)/$', views.DetailView.as_view(), name='detail'),  
] 

url(r'^reviews/', include((reviews_patterns, 'reviews', 
    'author-reviews'))), 

```

这将把提名的 URL 模式包含到给定的应用程序和实例命名空间中。例如，Django 管理界面被部署为`AdminSite`的实例。`AdminSite`对象有一个`urls`属性：一个包含相应管理站点中所有模式的 3 元组，加上应用程序命名空间"`admin`"和管理实例的名称。当你部署一个管理实例时，就是这个`urls`属性被`include()`到你的项目`urlpatterns`中。

一定要向`include()`传递一个元组。如果你只是简单地传递三个参数：`include(reviews_patterns`,`'reviews'`,`'author-reviews')`，Django 不会报错，但由于`include()`的签名，`'reviews'`将成为实例命名空间，`'author-reviews'`将成为应用程序命名空间，而不是相反。

# 接下来呢？

本章提供了许多关于视图和 URLconfs 的高级技巧。接下来，在第八章*高级模板*中，我们将对 Django 的模板系统进行高级处理。
