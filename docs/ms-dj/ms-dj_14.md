# 第十四章：生成非 HTML 内容

通常，当我们谈论开发网站时，我们谈论的是生成 HTML。当然，网页不仅仅是 HTML；我们使用网页以各种格式分发数据：RSS、PDF、图像等等。

到目前为止，我们专注于 HTML 生成的常见情况，但在本章中，我们将走一条弯路，看看如何使用 Django 生成其他类型的内容。Django 有方便的内置工具，可以用来生成一些常见的非 HTML 内容：

+   逗号分隔（CSV）文件，用于导入到电子表格应用程序中。

+   PDF 文件。

+   RSS/Atom 订阅源。

+   站点地图（最初由谷歌开发的 XML 格式，为搜索引擎提供提示）。

我们稍后会详细讨论这些工具，但首先我们将介绍基本原则。

# 基础知识：视图和 MIME 类型

从第二章中回忆，*视图和 URLconfs*，视图函数只是一个接受 Web 请求并返回 Web 响应的 Python 函数。这个响应可以是网页的 HTML 内容，或者重定向，或者 404 错误，或者 XML 文档，或者图像...或者任何东西。更正式地说，Django 视图函数必须*：*

1.  接受一个`HttpRequest`实例作为其第一个参数；和

1.  返回一个`HttpResponse`实例。

从视图返回非 HTML 内容的关键在于`HttpResponse`类，特别是`content_type`参数。默认情况下，Django 将`content_type`设置为 text/html。但是，您可以将`content_type`设置为 IANA 管理的任何官方互联网媒体类型（MIME 类型）（有关更多信息，请访问[`www.iana.org/assignments/media-types/media-types.xhtml`](http://www.iana.org/assignments/media-types/media-types.xhtml)）。

通过调整 MIME 类型，我们可以告诉浏览器我们返回了不同格式的响应。例如，让我们看一个返回 PNG 图像的视图。为了保持简单，我们只需从磁盘上读取文件：

```py
from django.http import HttpResponse 

def my_image(request): 
    image_data = open("/path/to/my/image.png", "rb").read() 
    return HttpResponse(image_data, content_type="image/png") 

```

就是这样！如果您用`open()`调用中的图像路径替换为真实图像的路径，您可以使用这个非常简单的视图来提供图像，浏览器将正确显示它。

另一个重要的事情是`HttpResponse`对象实现了 Python 的标准文件类对象 API。这意味着您可以在任何需要文件的地方使用`HttpResponse`实例，包括 Python（或第三方库）。让我们看一下如何使用 Django 生成 CSV 的示例。

# 生成 CSV

Python 自带一个 CSV 库，`csv`。使用它与 Django 的关键在于`csv`模块的 CSV 创建功能作用于类似文件的对象，而 Django 的`HttpResponse`对象是类似文件的对象。下面是一个例子：

```py
import csv 
from django.http import HttpResponse 

def some_view(request): 
    # Create the HttpResponse object with the appropriate CSV header. 
    response = HttpResponse(content_type='text/csv') 
    response['Content-Disposition'] = 'attachment; 
      filename="somefilename.csv"' 

    writer = csv.writer(response) 
    writer.writerow(['First row', 'Foo', 'Bar', 'Baz']) 
    writer.writerow(['Second row', 'A', 'B', 'C', '"Testing"']) 

    return response 

```

代码和注释应该是不言自明的，但有几件事值得一提：

+   响应获得了特殊的 MIME 类型`text/csv`。这告诉浏览器该文档是 CSV 文件，而不是 HTML 文件。如果不这样做，浏览器可能会将输出解释为 HTML，这将导致浏览器窗口中出现丑陋、可怕的胡言乱语。

+   响应获得了额外的`Content-Disposition`头，其中包含 CSV 文件的名称。这个文件名是任意的；随便取什么名字。它将被浏览器用于“另存为...”对话框等。

+   连接到 CSV 生成 API 很容易：只需将`response`作为`csv.writer`的第一个参数。`csv.writer`函数期望一个类似文件的对象，而`HttpResponse`对象符合要求。

+   对于 CSV 文件中的每一行，调用`writer.writerow`，将其传递给一个可迭代对象，如列表或元组。

+   CSV 模块会为您处理引用，因此您不必担心用引号或逗号转义字符串。只需将`writerow()`传递给您的原始字符串，它就会做正确的事情。

## 流式传输大型 CSV 文件

处理生成非常大响应的视图时，您可能希望考虑改用 Django 的`StreamingHttpResponse`。例如，通过流式传输需要很长时间生成的文件，您可以避免负载均衡器在服务器生成响应时可能会超时而断开连接。在这个例子中，我们充分利用 Python 生成器来高效地处理大型 CSV 文件的组装和传输：

```py
import csv 

from django.utils.six.moves import range 
from django.http import StreamingHttpResponse 

class Echo(object): 
    """An object that implements just the write method of the file-like 
    interface. 
    """ 
    def write(self, value): 
        """Write the value by returning it, instead of storing in a buffer.""" 
        return value 

def some_streaming_csv_view(request): 
    """A view that streams a large CSV file.""" 
    # Generate a sequence of rows. The range is based on the maximum number of 
    # rows that can be handled by a single sheet in most spreadsheet 
    # applications. 
    rows = (["Row {}".format(idx), str(idx)] for idx in range(65536)) 
    pseudo_buffer = Echo() 
    writer = csv.writer(pseudo_buffer) 
    response = StreamingHttpResponse((writer.writerow(row)  
      for row in rows), content_type="text/csv") 
    response['Content-Disposition'] = 'attachment;    
      filename="somefilename.csv"' 
    return response 

```

# 使用模板系统

或者，您可以使用 Django 模板系统来生成 CSV。这比使用方便的 Python `csv`模块更低级，但是这里提供了一个完整的解决方案。这里的想法是将一个项目列表传递给您的模板，并让模板在`for`循环中输出逗号。以下是一个示例，它生成与上面相同的 CSV 文件：

```py
from django.http import HttpResponse 
from django.template import loader, Context 

def some_view(request): 
    # Create the HttpResponse object with the appropriate CSV header. 
    response = HttpResponse(content_type='text/csv') 
    response['Content-Disposition'] = 'attachment;    
      filename="somefilename.csv"' 

    # The data is hard-coded here, but you could load it  
    # from a database or some other source. 
    csv_data = ( 
        ('First row', 'Foo', 'Bar', 'Baz'), 
        ('Second row', 'A', 'B', 'C', '"Testing"', "Here's a quote"), 
    ) 

    t = loader.get_template('my_template_name.txt') 
    c = Context({'data': csv_data,}) 
    response.write(t.render(c)) 
    return response 

```

这个例子和之前的例子唯一的区别是这个例子使用模板加载而不是 CSV 模块。其余的代码，比如`content_type='text/csv'`，都是一样的。然后，创建模板`my_template_name.txt`，其中包含以下模板代码：

```py
{% for row in data %} 
            "{{ row.0|addslashes }}", 
            "{{ row.1|addslashes }}", 
            "{{ row.2|addslashes }}", 
            "{{ row.3|addslashes }}", 
            "{{ row.4|addslashes }}" 
{% endfor %} 

```

这个模板非常基础。它只是遍历给定的数据，并为每一行显示一个 CSV 行。它使用`addslashes`模板过滤器来确保引号没有问题。

# 其他基于文本的格式

请注意，这里与 CSV 相关的内容并不多，只是特定的输出格式。您可以使用这些技术中的任何一种来输出您梦想中的任何基于文本的格式。您还可以使用类似的技术来生成任意二进制数据；例如，生成 PDF 文件。

# 生成 PDF

Django 能够使用视图动态输出 PDF 文件。这得益于出色的开源 ReportLab（有关更多信息，请访问[`www.reportlab.com/opensource/`](http://www.reportlab.com/opensource/)）Python PDF 库。动态生成 PDF 文件的优势在于，您可以为不同目的创建定制的 PDF 文件，比如为不同用户或不同内容创建。

# 安装 ReportLab

**ReportLab**库可在 PyPI 上获得。还可以下载用户指南（不巧的是，是一个 PDF 文件）。您可以使用`pip`安装 ReportLab：

```py
$ pip install reportlab 

```

通过在 Python 交互解释器中导入它来测试您的安装：

```py
>>> import reportlab 

```

如果该命令没有引发任何错误，则安装成功。

# 编写您的视图

使用 Django 动态生成 PDF 的关键是 ReportLab API，就像`csv`库一样，它作用于文件样对象，比如 Django 的`HttpResponse`。以下是一个 Hello World 示例：

```py
from reportlab.pdfgen import canvas 
from django.http import HttpResponse 

def some_view(request): 
    # Create the HttpResponse object with the appropriate PDF headers. 
    response = HttpResponse(content_type='application/pdf') 
    response['Content-Disposition'] = 'attachment;    
      filename="somefilename.pdf"' 

    # Create the PDF object, using the response object as its "file." 
    p = canvas.Canvas(response) 

    # Draw things on the PDF. Here's where the PDF generation happens. 
    # See the ReportLab documentation for the full list of functionality. 
    p.drawString(100, 100, "Hello world.") 

    # Close the PDF object cleanly, and we're done. 
    p.showPage() 
    p.save() 
    return response 

```

代码和注释应该是不言自明的，但有几点值得一提：

+   响应获得了特殊的 MIME 类型，`application/pdf`。这告诉浏览器该文档是一个 PDF 文件，而不是 HTML 文件。

+   响应获得了额外的`Content-Disposition`头部，其中包含 PDF 文件的名称。这个文件名是任意的：随便取什么名字都可以。浏览器将在“另存为...”对话框中使用它，等等。

+   在这个例子中，`Content-Disposition`头部以`'attachment; '`开头。这会强制 Web 浏览器弹出一个对话框，提示/确认如何处理文档，即使在计算机上设置了默认值。如果省略`'attachment;'`，浏览器将使用为 PDF 配置的任何程序/插件来处理 PDF。以下是该代码的样子：

```py
        response['Content-Disposition'] = 'filename="somefilename.pdf"'
```

+   连接到 ReportLab API 很容易：只需将`response`作为`canvas.Canvas`的第一个参数传递。`Canvas`类需要一个文件样对象，而`HttpResponse`对象正合适。

+   请注意，所有后续的 PDF 生成方法都是在 PDF 对象（在本例中是`p`）上调用的，而不是在`response`上调用的。

+   最后，重要的是在 PDF 文件上调用`showPage()`和`save()`。

# 复杂的 PDF

如果你正在使用 ReportLab 创建复杂的 PDF 文档，考虑使用`io`库作为 PDF 文件的临时存储位置。这个库提供了一个特别高效的类文件对象接口。以下是上面的 Hello World 示例重写，使用`io`：

```py
from io import BytesIO 
from reportlab.pdfgen import canvas 
from django.http import HttpResponse 

def some_view(request): 
    # Create the HttpResponse object with the appropriate PDF headers. 
    response = HttpResponse(content_type='application/pdf') 
    response['Content-Disposition'] = 'attachment;   
      filename="somefilename.pdf"' 

    buffer = BytesIO() 

    # Create the PDF object, using the BytesIO object as its "file." 
    p = canvas.Canvas(buffer) 

    # Draw things on the PDF. Here's where the PDF generation happens. 
    # See the ReportLab documentation for the full list of functionality. 
    p.drawString(100, 100, "Hello world.") 

    # Close the PDF object cleanly. 
    p.showPage() 
    p.save() 

    # Get the value of the BytesIO buffer and write it to the response. 
    pdf = buffer.getvalue() 
    buffer.close() 
    response.write(pdf) 
    return response 

```

# 更多资源

+   PDFlib ([`www.pdflib.org/`](http://www.pdflib.org/))是另一个具有 Python 绑定的 PDF 生成库。要在 Django 中使用它，只需使用本文中解释的相同概念。

+   Pisa XHTML2PDF ([`www.xhtml2pdf.com/`](http://www.xhtml2pdf.com/)) 是另一个 PDF 生成库。Pisa 附带了如何将 Pisa 与 Django 集成的示例。

+   HTMLdoc ([`www.htmldoc.org/`](http://www.htmldoc.org/))是一个可以将 HTML 转换为 PDF 的命令行脚本。它没有 Python 接口，但你可以使用`system`或`popen`跳出到 shell，并在 Python 中检索输出。

# 其他可能性

在 Python 中，你可以生成许多其他类型的内容。以下是一些更多的想法和一些指向你可以用来实现它们的库的指针：

+   **ZIP 文件**：Python 的标准库配备了`zipfile`模块，可以读取和写入压缩的 ZIP 文件。你可以使用它提供一堆文件的按需存档，或者在请求时压缩大型文档。你也可以使用标准库的`tarfile`模块类似地生成 TAR 文件。

+   **动态图片**：**Python Imaging Library**（**PIL**）([`www.pythonware.com/products/pil/`](http://www.pythonware.com/products/pil/))是一个用于生成图片（PNG、JPEG、GIF 等）的绝妙工具包。你可以使用它自动缩小图片为缩略图，将多个图片合成单个框架，甚至进行基于网络的图像处理。

+   **图表和图表**：有许多强大的 Python 绘图和图表库，你可以使用它们生成按需地图、图表、绘图和图表。我们不可能列出它们所有，所以这里是一些亮点：

+   `matplotlib` ([`matplotlib.sourceforge.net/`](http://matplotlib.sourceforge.net/))可用于生成通常使用 MatLab 或 Mathematica 生成的高质量图表。

+   `pygraphviz` ([`networkx.lanl.gov/pygraphviz/`](http://networkx.lanl.gov/pygraphviz/))，一个与 Graphviz 图形布局工具包的接口，可用于生成图和网络的结构化图表。

一般来说，任何能够写入文件的 Python 库都可以连接到 Django。可能性是巨大的。现在我们已经了解了生成非 HTML 内容的基础知识，让我们提高一个抽象级别。Django 配备了一些非常巧妙的内置工具，用于生成一些常见类型的非 HTML 内容。

# 联合供稿框架

Django 配备了一个高级别的联合供稿生成框架，可以轻松创建 RSS 和 Atom 供稿。RSS 和 Atom 都是基于 XML 的格式，你可以用它们提供站点内容的自动更新供稿。在这里阅读更多关于 RSS 的信息([`www.whatisrss.com/`](http://www.whatisrss.com/))，并在这里获取有关 Atom 的信息([`www.atomenabled.org/`](http://www.atomenabled.org/))。

创建任何联合供稿，你所要做的就是编写一个简短的 Python 类。你可以创建任意数量的供稿。Django 还配备了一个低级别的供稿生成 API。如果你想在网页上下文之外或以其他低级别方式生成供稿，可以使用这个 API。

# 高级别框架

## 概述

高级别的供稿生成框架由`Feed`类提供。要创建一个供稿，编写一个`Feed`类，并在你的 URLconf 中指向它的一个实例。

## 供稿类

`Feed`类是表示订阅源的 Python 类。订阅源可以是简单的（例如，站点新闻订阅，或者显示博客最新条目的基本订阅源）或更复杂的（例如，显示特定类别中的所有博客条目的订阅源，其中类别是可变的）。Feed 类是`django.contrib.syndication.views.Feed`的子类。它们可以存在于代码库的任何位置。`Feed`类的实例是视图，可以在您的 URLconf 中使用。

## 一个简单的例子

这个简单的例子，取自一个假设的警察打击新闻网站，描述了最新的五条新闻项目的订阅：

```py
from django.contrib.syndication.views import Feed 
from django.core.urlresolvers import reverse 
from policebeat.models import NewsItem 

class LatestEntriesFeed(Feed): 
    title = "Police beat site news" 
    link = "/sitenews/" 
    description = "Updates on changes and additions to police beat central." 

    def items(self): 
        return NewsItem.objects.order_by('-pub_date')[:5] 

    def item_title(self, item): 
        return item.title 

    def item_description(self, item): 
        return item.description 

    # item_link is only needed if NewsItem has no get_absolute_url method. 
    def item_link(self, item): 
        return reverse('news-item', args=[item.pk]) 

```

要将 URL 连接到此订阅源，请在您的 URLconf 中放置`Feed`对象的实例。例如：

```py
from django.conf.urls import url 
from myproject.feeds import LatestEntriesFeed 

urlpatterns = [ 
    # ... 
    url(r'^latest/feed/$', LatestEntriesFeed()), 
    # ... 
] 

```

**注意：**

+   Feed 类是`django.contrib.syndication.views.Feed`的子类。

+   `title`，`link`和`description`分别对应于标准的 RSS`<title>`，`<link>`和`<description>`元素。

+   `items()`只是一个返回应包含在订阅源中的对象列表的方法。尽管此示例使用 Django 的对象关系映射器返回`NewsItem`对象，但不必返回模型实例。尽管使用 Django 模型可以免费获得一些功能，但`items()`可以返回任何类型的对象。

+   如果您要创建 Atom 订阅源，而不是 RSS 订阅源，请设置`subtitle`属性，而不是`description`属性。有关示例，请参见本章后面的同时发布 Atom 和 RSS 订阅源。

还有一件事要做。在 RSS 订阅源中，每个`<item>`都有一个`<title>`，`<link>`和`<description>`。我们需要告诉框架将哪些数据放入这些元素中。

对于`<title>`和`<description>`的内容，Django 尝试在`Feed`类上调用`item_title()`和`item_description()`方法。它们传递了一个参数`item`，即对象本身。这些是可选的；默认情况下，对象的 unicode 表示用于两者。

如果您想对标题或描述进行任何特殊格式化，可以使用 Django 模板。它们的路径可以在`Feed`类的`title_template`和`description_template`属性中指定。模板为每个项目呈现，并传递了两个模板上下文变量：

+   `{{ obj }}`-：当前对象（您在`items()`中返回的任何对象之一）。

+   `{{ site }}`-：表示当前站点的 Django`site`对象。这对于`{{ site.domain }}`或`{{ site.name }}`非常有用。

请参阅下面使用描述模板的*一个复杂的例子*。

如果您需要提供比之前提到的两个变量更多的信息，还有一种方法可以将标题和描述模板传递给您。您可以在`Feed`子类中提供`get_context_data`方法的实现。例如：

```py
from mysite.models import Article 
from django.contrib.syndication.views import Feed 

class ArticlesFeed(Feed): 
    title = "My articles" 
    description_template = "feeds/articles.html" 

    def items(self): 
        return Article.objects.order_by('-pub_date')[:5] 

    def get_context_data(self, **kwargs): 
        context = super(ArticlesFeed, self).get_context_data(**kwargs) 
        context['foo'] = 'bar' 
        return context 

```

和模板：

```py
Something about {{ foo }}: {{ obj.description }} 

```

此方法将针对`items()`返回的列表中的每个项目调用一次，并带有以下关键字参数：

+   `item`：当前项目。出于向后兼容的原因，此上下文变量的名称为`{{ obj }}`。

+   `obj`：由`get_object()`返回的对象。默认情况下，这不会暴露给模板，以避免与`{{ obj }}`（见上文）混淆，但您可以在`get_context_data()`的实现中使用它。

+   `site`：如上所述的当前站点。

+   `request`：当前请求。

`get_context_data()`的行为模仿了通用视图的行为-您应该调用`super()`来从父类检索上下文数据，添加您的数据并返回修改后的字典。

要指定`<link>`的内容，您有两个选项。对于`items()`中的每个项目，Django 首先尝试在`Feed`类上调用`item_link()`方法。类似于标题和描述，它传递了一个参数-`item`。如果该方法不存在，Django 尝试在该对象上执行`get_absolute_url()`方法。

`get_absolute_url()`和`item_link()`都应返回项目的 URL 作为普通的 Python 字符串。与`get_absolute_url()`一样，`item_link()`的结果将直接包含在 URL 中，因此您负责在方法本身内部执行所有必要的 URL 引用和转换为 ASCII。

## 一个复杂的例子

该框架还通过参数支持更复杂的源。例如，网站可以为城市中每个警察拍摄提供最新犯罪的 RSS 源。为每个警察拍摄创建单独的`Feed`类是愚蠢的；这将违反 DRY 原则，并将数据耦合到编程逻辑中。

相反，辛迪加框架允许您访问从 URLconf 传递的参数，因此源可以根据源 URL 中的信息输出项目。警察拍摄源可以通过以下 URL 访问：

+   `/beats/613/rss/`-：返回 613 拍摄的最新犯罪。

+   `/beats/1424/rss/`-：返回 1424 拍摄的最新犯罪。

这些可以与 URLconf 行匹配，例如：

```py
url(r'^beats/(?P[0-9]+)/rss/$', BeatFeed()), 

```

与视图一样，URL 中的参数将与请求对象一起传递到`get_object()`方法。以下是这些特定于拍摄的源的代码：

```py
from django.contrib.syndication.views import FeedDoesNotExist 
from django.shortcuts import get_object_or_404 

class BeatFeed(Feed): 
    description_template = 'feeds/beat_description.html' 

    def get_object(self, request, beat_id): 
        return get_object_or_404(Beat, pk=beat_id) 

    def title(self, obj): 
        return "Police beat central: Crimes for beat %s" % obj.beat 

    def link(self, obj): 
        return obj.get_absolute_url() 

    def description(self, obj): 
        return "Crimes recently reported in police beat %s" % obj.beat 

    def items(self, obj): 
        return Crime.objects.filter(beat=obj).order_by(  
          '-crime_date')[:30] 

```

为了生成源的`<title>`，`<link>`和`<description>`，Django 使用`title()`，`link()`和`description()`方法。

在上一个示例中，它们是简单的字符串类属性，但是此示例说明它们可以是字符串*或*方法。对于`title`，`link`和`description`，Django 遵循此算法：

+   首先，它尝试调用一个方法，传递`obj`参数，其中`obj`是`get_object()`返回的对象。

+   如果失败，它将尝试调用一个没有参数的方法。

+   如果失败，它将使用 class 属性。

还要注意，`items()`也遵循相同的算法-首先尝试`items(obj)`，然后尝试`items()`，最后尝试`items`类属性（应该是一个列表）。我们正在使用模板来描述项目。它可以非常简单：

```py
{{ obj.description }} 

```

但是，您可以根据需要自由添加格式。下面的`ExampleFeed`类完整记录了`Feed`类的方法和属性。

## 指定源的类型

默认情况下，此框架生成的源使用 RSS 2.0。要更改此设置，请向您的`Feed`类添加`feed_type`属性，如下所示：

```py
from django.utils.feedgenerator import Atom1Feed 

class MyFeed(Feed): 
    feed_type = Atom1Feed 

```

请注意，将`feed_type`设置为类对象，而不是实例。当前可用的源类型有：

+   `django.utils.feedgenerator.Rss201rev2Feed`（RSS 2.01。默认）。

+   `django.utils.feedgenerator.RssUserland091Feed`（RSS 0.91）。

+   `django.utils.feedgenerator.Atom1Feed`（Atom 1.0）。

## 附件

要指定附件，例如在创建播客源时使用的附件，请使用`item_enclosure_url`，`item_enclosure_length`和`item_enclosure_mime_type`挂钩。有关用法示例，请参阅下面的`ExampleFeed`类。

## 语言

使用辛迪加框架创建的源自动包括适当的`<language>`标签（RSS 2.0）或`xml:lang`属性（Atom）。这直接来自您的`LANGUAGE_CODE`设置。

## URL

`link`方法/属性可以返回绝对路径（例如，`/blog/`）或具有完全合格的域和协议的 URL（例如，`http://www.example.com/blog/`）。如果`link`不返回域，辛迪加框架将根据您的`SITE_ID`设置插入当前站点的域。Atom 源需要定义源的当前位置的`<link rel="self">`。辛迪加框架会自动填充这一点，使用当前站点的域，根据`SITE_ID`设置。

## 同时发布 Atom 和 RSS 源

一些开发人员喜欢提供其源的 Atom 和 RSS 版本。在 Django 中很容易做到：只需创建`Feed`类的子类，并将`feed_type`设置为不同的内容。然后更新您的 URLconf 以添加额外的版本。以下是一个完整的示例：

```py
from django.contrib.syndication.views import Feed 
from policebeat.models import NewsItem 
from django.utils.feedgenerator import Atom1Feed 

class RssSiteNewsFeed(Feed): 
    title = "Police beat site news" 
    link = "/sitenews/" 
    description = "Updates on changes and additions to police beat central." 

    def items(self): 
        return NewsItem.objects.order_by('-pub_date')[:5] 

class AtomSiteNewsFeed(RssSiteNewsFeed): 
    feed_type = Atom1Feed 
    subtitle = RssSiteNewsFeed.description 

```

### 注意

在这个例子中，RSS feed 使用 `description`，而 Atom feed 使用 `subtitle`。这是因为 Atom feed 不提供 feed 级别的描述，但它们提供了一个副标题。如果您在 `Feed` 类中提供了 `description`，Django 将不会自动将其放入 `subtitle` 元素中，因为副标题和描述不一定是相同的。相反，您应该定义一个 `subtitle` 属性。

在上面的示例中，我们将 Atom feed 的 `subtitle` 设置为 RSS feed 的 `description`，因为它已经相当短了。并且相应的 URLconf：

```py
from django.conf.urls import url 
from myproject.feeds import RssSiteNewsFeed, AtomSiteNewsFeed 

urlpatterns = [ 
    # ... 
    url(r'^sitenews/rss/$', RssSiteNewsFeed()), 
    url(r'^sitenews/atom/$', AtomSiteNewsFeed()), 
    # ... 
] 

```

### 注意

有关 `Feed` 类的所有可能属性和方法的示例，请参见：`https://docs.djangoproject.com/en/1.8/ref/contrib/syndication/#feed-class-reference`

# 低级别框架

在幕后，高级 RSS 框架使用较低级别的框架来生成 feed 的 XML。这个框架存在于一个单独的模块中：`django/utils/feedgenerator.py`。您可以自己使用这个框架进行较低级别的 feed 生成。您还可以创建自定义 feed 生成器子类，以便与 `feed_type` `Feed` 选项一起使用。

## SyndicationFeed 类

`feedgenerator` 模块包含一个基类：

+   `django.utils.feedgenerator.SyndicationFeed`

和几个子类：

+   `django.utils.feedgenerator.RssUserland091Feed`

+   `django.utils.feedgenerator.Rss201rev2Feed`

+   `django.utils.feedgenerator.Atom1Feed`

这三个类都知道如何将某种类型的 feed 渲染为 XML。它们共享这个接口：

### SyndicationFeed.__init__()

使用给定的元数据字典初始化 feed，该元数据适用于整个 feed。必需的关键字参数是：

+   `标题`

+   `链接`

+   `描述`

还有一堆其他可选关键字：

+   `语言`

+   `作者电子邮件`

+   `作者名称`

+   `作者链接`

+   `副标题`

+   `类别`

+   `feed_url`

+   `feed_copyright`

+   `feed_guid`

+   `ttl`

您传递给 `__init__` 的任何额外关键字参数都将存储在 `self.feed` 中，以便与自定义 feed 生成器一起使用。

所有参数都应该是 Unicode 对象，除了 `categories`，它应该是 Unicode 对象的序列。

### SyndicationFeed.add_item()

使用给定参数向 feed 添加一个项目。

必需的关键字参数是：

+   `标题`

+   `链接`

+   `描述`

可选关键字参数是：

+   `作者电子邮件`

+   `作者名称`

+   `作者链接`

+   `pubdate`

+   `评论`

+   `unique_id`

+   `enclosure`

+   `类别`

+   `item_copyright`

+   `ttl`

+   `updateddate`

额外的关键字参数将被存储以供自定义 feed 生成器使用。所有参数，如果给定，都应该是 Unicode 对象，除了：

+   `pubdate` 应该是 Python `datetime` 对象。

+   `updateddate` 应该是 Python `datetime` 对象。

+   `enclosure` 应该是 `django.utils.feedgenerator.Enclosure` 的一个实例。

+   `categories` 应该是 Unicode 对象的序列。

### SyndicationFeed.write()

将 feed 以给定编码输出到 outfile，这是一个类似文件的对象。

### SyndicationFeed.writeString()

以给定编码的字符串形式返回 feed。例如，要创建 Atom 1.0 feed 并将其打印到标准输出：

```py
>>> from django.utils import feedgenerator 
>>> from datetime import datetime 
>>> f = feedgenerator.Atom1Feed( 
...     , 
...     link="http://www.example.com/", 
...     description="In which I write about what I ate today.", 
...     language="en", 
...     author_name="Myself", 
...     feed_url="http://example.com/atom.xml") 
>>> f.add_item(, 
...     link="http://www.example.com/entries/1/", 
...     pubdate=datetime.now(), 
...     description="<p>Today I had a Vienna Beef hot dog. It was pink, plump and perfect.</p>") 
>>> print(f.writeString('UTF-8')) 
<?xml version="1.0" encoding="UTF-8"?> 
<feed  xml:lang="en"> 
... 
</feed> 

```

## 自定义 feed 生成器

如果您需要生成自定义 feed 格式，您有几个选择。如果 feed 格式完全自定义，您将需要对 `SyndicationFeed` 进行子类化，并完全替换 `write()` 和 `writeString()` 方法。但是，如果 feed 格式是 RSS 或 Atom 的一个衍生格式（即 GeoRSS，（链接到网站 [`georss.org/`](http://georss.org/)），苹果的 iTunes podcast 格式（链接到网站 [`www.apple.com/itunes/podcasts/specs.html`](http://www.apple.com/itunes/podcasts/specs.html)）等），您有更好的选择。

这些类型的 feed 通常会向底层格式添加额外的元素和/或属性，并且有一组方法，`SyndicationFeed` 调用这些额外的属性。因此，您可以对适当的 feed 生成器类（`Atom1Feed` 或 `Rss201rev2Feed`）进行子类化，并扩展这些回调。它们是：

### SyndicationFeed.root_attributes(self, )

返回要添加到根源元素（`feed`/`channel`）的属性字典。

### SyndicationFeed.add_root_elements(self, handler)

回调以在根源元素（`feed`/`channel`）内添加元素。`handler`是 Python 内置 SAX 库中的`XMLGenerator`；您将在其上调用方法以添加到正在处理的 XML 文档中。

### SyndicationFeed.item_attributes(self, item)

返回要添加到每个条目（`item`/`entry`）元素的属性字典。参数`item`是传递给`SyndicationFeed.add_item()`的所有数据的字典。

### SyndicationFeed.add_item_elements(self, handler, item)

回调以向每个条目（`item`/`entry`）元素添加元素。`handler`和`item`与上述相同。

### 注意

如果您覆盖了这些方法中的任何一个，请确保调用超类方法，因为它们会为每个 feed 格式添加所需的元素。

例如，您可以开始实现一个 iTunes RSS feed 生成器，如下所示：

```py
class iTunesFeed(Rss201rev2Feed): 
    def root_attributes(self): 
        attrs = super(iTunesFeed, self).root_attributes() 
        attrs['xmlns:itunes'] =  
          'http://www.itunes.com/dtds/podcast-1.0.dtd' 
        return attrs 

    def add_root_elements(self, handler): 
        super(iTunesFeed, self).add_root_elements(handler) 
        handler.addQuickElement('itunes:explicit', 'clean') 

```

显然，要创建一个完整的自定义 feed 类还有很多工作要做，但上面的例子应该演示了基本思想。

# 站点地图框架

**站点地图**是您网站上的一个 XML 文件，告诉搜索引擎索引器您的页面更改的频率以及与站点上其他页面的重要性。这些信息有助于搜索引擎索引您的站点。有关站点地图的更多信息，请参阅 sitemaps.org 网站。

Django 站点地图框架通过让您在 Python 代码中表达此信息来自动创建此 XML 文件。它的工作方式与 Django 的 Syndication 框架类似。要创建站点地图，只需编写一个`Sitemap`类并在 URLconf 中指向它。

## 安装

要安装站点地图应用，请按照以下步骤进行：

+   将`"django.contrib.sitemaps"`添加到您的`INSTALLED_APPS`设置中。

+   确保您的`TEMPLATES`设置包含一个`DjangoTemplates`后端，其`APP_DIRS`选项设置为 True。默认情况下就在那里，所以只有在更改了该设置时才需要更改这一点。

+   确保您已安装了站点框架。

## 初始化

要在 Django 站点上激活站点地图生成，请将此行添加到您的 URLconf 中：

```py
from django.contrib.sitemaps.views import sitemap 

url(r'^sitemap\.xml$', sitemap, {'sitemaps': sitemaps}, 
    name='django.contrib.sitemaps.views.sitemap') 

```

这告诉 Django 在客户端访问`/sitemap.xml`时构建站点地图。站点地图文件的名称并不重要，但位置很重要。搜索引擎只会索引站点地图中当前 URL 级别及以下的链接。例如，如果`sitemap.xml`位于根目录中，它可以引用站点中的任何 URL。但是，如果您的站点地图位于`/content/sitemap.xml`，它只能引用以`/content/`开头的 URL。

站点地图视图需要一个额外的必需参数：`{'sitemaps': sitemaps}`。`sitemaps`应该是一个将短部分标签（例如`blog`或`news`）映射到其`Sitemap`类（例如`BlogSitemap`或`NewsSitemap`）的字典。它也可以映射到`Sitemap`类的实例（例如`BlogSitemap(some_var)`）。

## 站点地图类

`Sitemap`类是一个简单的 Python 类，表示站点地图中的条目部分。例如，一个`Sitemap`类可以表示您博客的所有条目，而另一个可以表示您事件日历中的所有事件。

在最简单的情况下，所有这些部分都被合并到一个`sitemap.xml`中，但也可以使用框架生成引用各个站点地图文件的站点地图索引，每个部分一个文件。（请参阅下面的创建站点地图索引。）

`Sitemap`类必须是`django.contrib.sitemaps.Sitemap`的子类。它们可以存在于代码库中的任何位置。

## 一个简单的例子

假设您有一个博客系统，其中有一个`Entry`模型，并且您希望您的站点地图包括到您个人博客条目的所有链接。以下是您的站点地图类可能如何看起来：

```py
from django.contrib.sitemaps import Sitemap 
from blog.models import Entry 

class BlogSitemap(Sitemap): 
    changefreq = "never" 
    priority = 0.5 

    def items(self): 
        return Entry.objects.filter(is_draft=False) 

    def lastmod(self, obj): 
        return obj.pub_date 

```

**注意：**

+   `changefreq`和`priority`是对应于`<changefreq>`和`<priority>`元素的类属性。它们可以作为函数调用，就像上面的`lastmod`一样。

+   `items()`只是返回对象列表的方法。返回的对象将传递给与站点地图属性（`location`，`lastmod`，`changefreq`和`priority`）对应的任何可调用方法。

+   `lastmod`应返回 Python `datetime`对象。

+   在此示例中没有`location`方法，但您可以提供它以指定对象的 URL。默认情况下，`location()`调用每个对象上的`get_absolute_url()`并返回结果。

## 站点地图类参考

`Sitemap`类可以定义以下方法/属性：

### items

**必需。**返回对象列表的方法。框架不关心它们是什么*类型*的对象；重要的是这些对象传递给`location()`，`lastmod()`，`changefreq()`和`priority()`方法。

### 位置

**可选。**可以是方法或属性。如果是方法，它应该返回`items()`返回的给定对象的绝对路径。如果是属性，其值应该是表示`items()`返回的每个对象使用的绝对路径的字符串。

在这两种情况下，绝对路径表示不包括协议或域的 URL。示例：

+   好的：`'/foo/bar/'`

+   不好：`'example.com/foo/bar/'`

+   不好：`'http://example.com/foo/bar/'`

如果未提供`location`，框架将调用`items()`返回的每个对象上的`get_absolute_url()`方法。要指定除`http`之外的协议，请使用`protocol`。

### lastmod

**可选。**可以是方法或属性。如果是方法，它应该接受一个参数-`items()`返回的对象-并返回该对象的最后修改日期/时间，作为 Python `datetime.datetime`对象。

如果它是一个属性，其值应该是一个 Python `datetime.datetime`对象，表示`items()`返回的*每个*对象的最后修改日期/时间。如果站点地图中的所有项目都有`lastmod`，则`views.sitemap()`生成的站点地图将具有等于最新`lastmod`的`Last-Modified`标头。

您可以激活`ConditionalGetMiddleware`，使 Django 对具有`If-Modified-Since`标头的请求做出适当响应，这将防止在站点地图未更改时发送站点地图。

### changefreq

**可选。**可以是方法或属性。如果是方法，它应该接受一个参数-`items()`返回的对象-并返回该对象的更改频率，作为 Python 字符串。如果是属性，其值应该是表示`items()`返回的每个对象的更改频率的字符串。无论您使用方法还是属性，`changefreq`的可能值是：

+   `'always'`

+   `'hourly'`

+   `'daily'`

+   `'weekly'`

+   `'monthly'`

+   `'yearly'`

+   `'never'`

### priority

**可选。**可以是方法或属性。如果是方法，它应该接受一个参数-`items()`返回的对象-并返回该对象的优先级，作为字符串或浮点数。

如果它是一个属性，其值应该是一个字符串或浮点数，表示`items()`返回的每个对象的优先级。`priority`的示例值：`0.4`，`1.0`。页面的默认优先级为`0.5`。有关更多信息，请参阅 sitemaps.org 文档。

### 协议

**可选。**此属性定义站点地图中 URL 的协议（`http`或`https`）。如果未设置，将使用请求站点地图的协议。如果站点地图是在请求的上下文之外构建的，则默认值为`http`。

### i18n

**可选。**一个布尔属性，定义此站点地图的 URL 是否应使用所有`LANGUAGES`生成。默认值为`False`。

## 快捷方式

网站地图框架为常见情况提供了一个方便的类-`django.contrib.syndication.GenericSitemap`

`django.contrib.sitemaps.GenericSitemap`类允许您通过向其传递至少包含`queryset`条目的字典来创建站点地图。此查询集将用于生成站点地图的项目。它还可以具有指定从`queryset`检索的对象的日期字段的`date_field`条目。

这将用于生成的站点地图中的`lastmod`属性。您还可以将`priority`和`changefreq`关键字参数传递给`GenericSitemap`构造函数，以指定所有 URL 的这些属性。

### 例子

以下是使用`GenericSitemap`的 URLconf 示例：

```py
from django.conf.urls import url 
from django.contrib.sitemaps import GenericSitemap 
from django.contrib.sitemaps.views import sitemap 
from blog.models import Entry 

info_dict = { 
    'queryset': Entry.objects.all(), 
    'date_field': 'pub_date', 
} 

urlpatterns = [ 
    # some generic view using info_dict 
    # ... 

    # the sitemap 
    url(r'^sitemap\.xml$', sitemap, 
        {'sitemaps': {'blog': GenericSitemap(info_dict, priority=0.6)}},  
        name='django.contrib.sitemaps.views.sitemap'), 
] 

```

## 静态视图的站点地图

通常，您希望搜索引擎爬虫索引既不是对象详细页面也不是平面页面的视图。解决方案是在`sitemap`的`items`中显式列出这些视图的 URL 名称，并在`sitemap`的`location`方法中调用`reverse()`。例如：

```py
# sitemaps.py 
from django.contrib import sitemaps 
from django.core.urlresolvers import reverse 

class StaticViewSitemap(sitemaps.Sitemap): 
    priority = 0.5 
    changefreq = 'daily' 

    def items(self): 
        return ['main', 'about', 'license'] 

    def location(self, item): 
        return reverse(item) 

# urls.py 
from django.conf.urls import url 
from django.contrib.sitemaps.views import sitemap 

from .sitemaps import StaticViewSitemap 
from . import views 

sitemaps = { 
    'static': StaticViewSitemap, 
} 

urlpatterns = [ 
    url(r'^$', views.main, name='main'), 
    url(r'^about/$', views.about, name='about'), 
    url(r'^license/$', views.license, name='license'), 
    # ... 
    url(r'^sitemap\.xml$', sitemap, {'sitemaps': sitemaps}, 
        name='django.contrib.sitemaps.views.sitemap') 
] 

```

## 创建站点地图索引

站点地图框架还具有创建引用各自`sitemaps`字典中定义的每个部分的单独站点地图文件的站点地图索引的功能。使用的唯一区别是：

+   您在 URLconf 中使用了两个视图：`django.contrib.sitemaps.views.index()`和`django.contrib.sitemaps.views.sitemap()`。

+   `django.contrib.sitemaps.views.sitemap()`视图应该接受一个`section`关键字参数。

以下是上述示例的相关 URLconf 行的样子：

```py
from django.contrib.sitemaps import views 

urlpatterns = [ 
    url(r'^sitemap\.xml$', views.index, {'sitemaps': sitemaps}), 
    url(r'^sitemap-(?P<section>.+)\.xml$', views.sitemap,  
        {'sitemaps': sitemaps}), 
] 

```

这将自动生成一个`sitemap.xml`文件，其中引用了`sitemap-flatpages.xml`和`sitemap-blog.xml`。`Sitemap`类和`sitemaps`字典完全不会改变。

如果您的站点地图中有超过 50,000 个 URL，则应创建一个索引文件。在这种情况下，Django 将自动对站点地图进行分页，并且索引将反映这一点。如果您没有使用原始站点地图视图-例如，如果它被缓存装饰器包装-您必须为您的站点地图视图命名，并将`sitemap_url_name`传递给索引视图：

```py
from django.contrib.sitemaps import views as sitemaps_views 
from django.views.decorators.cache import cache_page 

urlpatterns = [ 
    url(r'^sitemap\.xml$', 
        cache_page(86400)(sitemaps_views.index), 
        {'sitemaps': sitemaps, 'sitemap_url_name': 'sitemaps'}), 
    url(r'^sitemap-(?P<section>.+)\.xml$', 
        cache_page(86400)(sitemaps_views.sitemap), 
        {'sitemaps': sitemaps}, name='sitemaps'), 
] 

```

## 模板自定义

如果您希望在站点上可用的每个站点地图或站点地图索引使用不同的模板，您可以通过在 URLconf 中向`sitemap`和`index`视图传递`template_name`参数来指定它：

```py
from django.contrib.sitemaps import views 

urlpatterns = [ 
    url(r'^custom-sitemap\.xml$', views.index, { 
        'sitemaps': sitemaps, 
        'template_name': 'custom_sitemap.html' 
    }), 
    url(r'^custom-sitemap-(?P<section>.+)\.xml$', views.sitemap, { 
    'sitemaps': sitemaps, 
    'template_name': 'custom_sitemap.html' 
}), 
] 

```

## 上下文变量

在自定义`index()`和`sitemap()`视图的模板时，您可以依赖以下上下文变量。

### 索引

变量`sitemaps`是每个站点地图的绝对 URL 的列表。

### 站点地图

变量`urlset`是应该出现在站点地图中的 URL 列表。每个 URL 都公开了`Sitemap`类中定义的属性：

+   `changefreq`

+   `item`

+   `lastmod`

+   `位置`

+   `priority`

已为每个 URL 添加了`item`属性，以允许对模板进行更灵活的自定义，例如 Google 新闻站点地图。假设 Sitemap 的`items()`将返回一个具有`publication_data`和`tags`字段的项目列表，类似这样将生成一个与 Google 兼容的站点地图：

```py
{% spaceless %} 
{% for url in urlset %} 
    {{ url.location }} 
    {% if url.lastmod %}{{ url.lastmod|date:"Y-m-d" }}{% endif %} 
    {% if url.changefreq %}{{ url.changefreq }}{% endif %} 
    {% if url.priority %}{{ url.priority }}{% endif %} 

      {% if url.item.publication_date %}{{ url.item.publication_date|date:"Y-m-d" }}{% endif %} 
      {% if url.item.tags %}{{ url.item.tags }}{% endif %} 

{% endfor %} 
{% endspaceless %} 

```

## ping google

当您的站点地图发生更改时，您可能希望向 Google 发送 ping，以便让它知道重新索引您的站点。站点地图框架提供了一个函数来实现这一点：

### django.contrib.syndication.ping_google()

`ping_google()`接受一个可选参数`sitemap_url`，它应该是站点地图的绝对路径（例如`'/sitemap.xml'`）。如果未提供此参数，`ping_google()`将尝试通过在 URLconf 中执行反向查找来确定您的站点地图。如果无法确定您的站点地图 URL，`ping_google()`会引发异常`django.contrib.sitemaps.SitemapNotFound`。

从模型的`save()`方法中调用`ping_google()`的一个有用的方法是：

```py
from django.contrib.sitemaps import ping_google 

class Entry(models.Model): 
    # ... 
    def save(self, force_insert=False, force_update=False): 
        super(Entry, self).save(force_insert, force_update) 
        try: 
            ping_google() 
        except Exception: 
            # Bare 'except' because we could get a variety 
            # of HTTP-related exceptions. 
            pass 

```

然而，更有效的解决方案是从 cron 脚本或其他计划任务中调用`ping_google()`。该函数会向 Google 的服务器发出 HTTP 请求，因此您可能不希望在每次调用`save()`时引入网络开销。

### 通过 manage.py 向 Google 发送 ping

一旦站点地图应用程序添加到您的项目中，您还可以使用`ping_google`管理命令来 ping Google：

```py
python manage.py ping_google [/sitemap.xml] 

```

### 注意

**首先向 Google 注册！**只有在您已经在 Google 网站管理员工具中注册了您的站点时，`ping_google()`命令才能起作用。

# 接下来是什么？

接下来，我们将继续深入研究 Django 提供的内置工具，通过更仔细地查看 Django 会话框架。
