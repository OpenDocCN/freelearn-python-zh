# 构建您的第一个 Web 抓取应用程序

在本章中，我们将涵盖以下内容：

+   下载网页

+   解析 HTML

+   爬取网络

+   订阅源

+   访问 Web API

+   与表单交互

+   使用 Selenium 进行高级交互

+   访问受密码保护的页面

+   加速网络抓取

# 介绍

互联网和**WWW**（**万维网**）可能是当今最重要的信息来源。大部分信息可以通过 HTTP 协议检索。**HTTP**最初是为了共享超文本页面而发明的（因此称为**超文本传输协议**），这开创了 WWW。

这个操作非常熟悉，因为它是任何网络浏览器中发生的事情。但我们也可以以编程方式执行这些操作，自动检索和处理信息。Python 在标准库中包含了一个 HTTP 客户端，但是 fantastic `requests`模块使它变得非常容易。在本章中，我们将看到如何做到这一点。

# 下载网页

下载网页的基本能力涉及对 URL 发出 HTTP `GET`请求。这是任何网络浏览器的基本操作。让我们快速回顾一下这个操作的不同部分：

1.  使用 HTTP 协议。

1.  使用最常见的 HTTP 方法`GET`。我们将在*访问 Web API*配方中看到更多。

1.  URL 描述页面的完整地址，包括服务器和路径。

该请求将由服务器处理，并发送回一个响应。这个响应将包含一个**状态码**，通常是 200，如果一切顺利的话，以及一个包含结果的 body，通常是一个包含 HTML 页面的文本。

大部分由用于执行请求的 HTTP 客户端自动处理。在这个配方中，我们将看到如何发出简单的请求以获取网页。

HTTP 请求和响应也可以包含头部。头部包含额外的信息，如请求的总大小，内容的格式，请求的日期以及使用的浏览器或服务器。

# 准备工作

使用 fantastic `requests`模块，获取网页非常简单。安装模块：

```py
$ echo "requests==2.18.3" >> requirements.txt
$ source .venv/bin/activate
(.venv) $ pip install -r requirements.txt 
```

我们将下载页面在[`www.columbia.edu/~fdc/sample.html`](http://www.columbia.edu/~fdc/sample.html)，因为它是一个简单的 HTML 页面，很容易在文本模式下阅读。

# 如何做...

1.  导入`requests`模块：

```py
>>> import requests
```

1.  对 URL 发出请求，这将花费一两秒钟：

```py
>>> url = 'http://www.columbia.edu/~fdc/sample.html'
>>> response = requests.get(url)
```

1.  检查返回的对象状态码：

```py
>>> response.status_code
200
```

1.  检查结果的内容：

```py
>>> response.text
'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n<html>\n<head>\n
...
FULL BODY
...
<!-- close the <html> begun above -->\n'
```

1.  检查进行中和返回的头部：

```py
>>> response.request.headers
{'User-Agent': 'python-requests/2.18.4', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive'}
>>> response.headers
{'Date': 'Fri, 25 May 2018 21:51:47 GMT', 'Server': 'Apache', 'Last-Modified': 'Thu, 22 Apr 2004 15:52:25 GMT', 'Accept-Ranges': 'bytes', 'Vary': 'Accept-Encoding,User-Agent', 'Content-Encoding': 'gzip', 'Content-Length': '8664', 'Keep-Alive': 'timeout=15, max=85', 'Connection': 'Keep-Alive', 'Content-Type': 'text/html', 'Set-Cookie': 'BIGipServer~CUIT~www.columbia.edu-80-pool=1764244352.20480.0000; expires=Sat, 26-May-2018 03:51:47 GMT; path=/; Httponly'}
```

# 它是如何工作的...

`requests`的操作非常简单；在 URL 上执行操作，这种情况下是`GET`，返回一个可以分析的`result`对象。主要元素是`status_code`和 body 内容，可以呈现为`text`。

可以在`request`字段中检查完整的请求：

```py
>>> response.request
<PreparedRequest [GET]>
>>> response.request.url
'http://www.columbia.edu/~fdc/sample.html'
```

完整的请求文档可以在这里找到：[`docs.python-requests.org/en/master/`](http://docs.python-requests.org/en/master/)。在本章中，我们将展示更多功能。

# 还有更多...

所有 HTTP 状态码可以在这个网页上检查：[`httpstatuses.com/`](https://httpstatuses.com/)。它们也在`httplib`模块中以方便的常量名称进行描述，如`OK`，`NOT_FOUND`或`FORBIDDEN`。

最著名的错误状态码可能是 404，当 URL 未找到时会发生。通过执行`requests.get('http://www.columbia.edu/invalid')`来尝试。

请求可以使用**HTTPS**协议（**安全 HTTP**）。它是等效的，但确保请求和响应的内容是私有的。`requests`会自动处理它。

任何处理任何私人信息的网站都将使用 HTTPS 来确保信息没有泄漏。HTTP 容易受到窃听。尽可能使用 HTTPS。

# 另请参阅

+   在第一章的*让我们开始自动化之旅*中的*安装第三方包*配方中

+   *解析 HTML*配方

# 解析 HTML

下载原始文本或二进制文件是一个很好的起点，但是网页的主要语言是 HTML。

HTML 是一种结构化语言，定义文档的不同部分，如标题和段落。HTML 也是分层的，定义了子元素。将原始文本解析为结构化文档的能力基本上是能够从网页中自动提取信息的能力。例如，如果在特定的`class div`中或在标题`h3`标签后面包含一些文本，则该文本可能是相关的。

# 准备就绪

我们将使用优秀的 Beautiful Soup 模块将 HTML 文本解析为可以分析的内存对象。我们需要使用`beautifulsoup4`包来使用可用的最新 Python 3 版本。将包添加到您的`requirements.txt`并在虚拟环境中安装依赖项：

```py
$ echo "beautifulsoup4==4.6.0" >> requirements.txt
$ pip install -r requirements.txt
```

# 如何做...

1.  导入`BeautifulSoup`和`requests`：

```py
>>> import requests >>> from bs4 import BeautifulSoup
```

1.  设置要下载并检索的页面的 URL：

```py
>>> URL = 'http://www.columbia.edu/~fdc/sample.html'
>>> response = requests.get(URL)
>>> response
<Response [200]>
```

1.  解析下载的页面：

```py
>>> page = BeautifulSoup(response.text, 'html.parser')
```

1.  获取页面的标题。注意它与浏览器中显示的内容相同：

```py
>>> page.title
<title>Sample Web Page</title>
>>> page.title.string
'Sample Web Page'
```

1.  在页面中查找所有的`h3`元素，以确定现有的部分：

```py
>>> page.find_all('h3')
[<h3><a name="contents">CONTENTS</a></h3>, <h3><a name="basics">1\. Creating a Web Page</a></h3>, <h3><a name="syntax">2\. HTML Syntax</a></h3>, <h3><a name="chars">3\. Special Characters</a></h3>, <h3><a name="convert">4\. Converting Plain Text to HTML</a></h3>, <h3><a name="effects">5\. Effects</a></h3>, <h3><a name="lists">6\. Lists</a></h3>, <h3><a name="links">7\. Links</a></h3>, <h3><a name="tables">8\. Tables</a></h3>, <h3><a name="install">9\. Installing Your Web Page on the Internet</a></h3>, <h3><a name="more">10\. Where to go from here</a></h3>]
```

1.  提取部分链接上的文本。当达到下一个`<h3>`标签时停止：

```py
>>> link_section = page.find('a', attrs={'name': 'links'})
>>> section = []
>>> for element in link_section.next_elements:
...     if element.name == 'h3':
...         break
...     section.append(element.string or '')
...
>>> result = ''.join(section)
>>> result
'7\. Links\n\nLinks can be internal within a Web page (like to\nthe Table of ContentsTable of Contents at the top), or they\ncan be to external web pages or pictures on the same website, or they\ncan be to websites, pages, or pictures anywhere else in the world.\n\n\n\nHere is a link to the Kermit\nProject home pageKermit\nProject home page.\n\n\n\nHere is a link to Section 5Section 5 of this document.\n\n\n\nHere is a link to\nSection 4.0Section 4.0\nof the C-Kermit\nfor Unix Installation InstructionsC-Kermit\nfor Unix Installation Instructions.\n\n\n\nHere is a link to a picture:\nCLICK HERECLICK HERE to see it.\n\n\n'
```

注意没有 HTML 标记；这都是原始文本。

# 它是如何工作的...

第一步是下载页面。然后，可以像第 3 步那样解析原始文本。生成的`page`对象包含解析的信息。

`html.parser`解析器是默认的，但是对于特定操作可能会出现问题。例如，对于大页面，它可能会很慢，或者在渲染高度动态的网页时可能会出现问题。您可以使用其他解析器，例如`lxml`，它速度更快，或者`html5lib`，它将更接近浏览器的操作，包括 HTML5 产生的动态更改。它们是外部模块，需要添加到`requirements.txt`文件中。

`BeautifulSoup`允许我们搜索 HTML 元素。它可以使用`.find()`搜索第一个元素，或者使用`.find_all()`返回一个列表。在第 5 步中，它搜索了一个具有特定属性`name=link`的特定标签`<a>`。之后，它继续在`.next_elements`上迭代，直到找到下一个`h3`标签，标志着该部分的结束。

提取每个元素的文本，最后组合成单个文本。注意`or`，它避免存储`None`，当元素没有文本时返回。

HTML 非常灵活，可以有多种结构。本配方中介绍的情况是典型的，但是在划分部分方面的其他选项可能是将相关部分组合在一个大的`<div>`标签或其他元素内，甚至是原始文本。需要进行一些实验，直到找到从网页中提取重要部分的特定过程。不要害怕尝试！

# 还有更多...

正则表达式也可以用作`.find()`和`.find_all()`方法的输入。例如，此搜索使用`h2`和`h3`标签：

```py
>>> page.find_all(re.compile('^h(2|3)'))
[<h2>Sample Web Page</h2>, <h3><a name="contents">CONTENTS</a></h3>, <h3><a name="basics">1\. Creating a Web Page</a></h3>, <h3><a name="syntax">2\. HTML Syntax</a></h3>, <h3><a name="chars">3\. Special Characters</a></h3>, <h3><a name="convert">4\. Converting Plain Text to HTML</a></h3>, <h3><a name="effects">5\. Effects</a></h3>, <h3><a name="lists">6\. Lists</a></h3>, <h3><a name="links">7\. Links</a></h3>,
```

```py
<h3><a name="tables">8\. Tables</a></h3>, <h3><a name="install">9\. Installing Your Web Page on the Internet</a></h3>, <h3><a name="more">10\. Where to go from here</a></h3>]
```

另一个有用的 find 参数是包含`class_`参数的 CSS 类。这将在本书的后面显示。

完整的 Beautiful Soup 文档可以在这里找到：[`www.crummy.com/software/BeautifulSoup/bs4/doc/`](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)。

# 另请参阅

+   在第一章的*让我们开始自动化之旅*中的*安装第三方包*配方

+   在第一章的*让我们开始自动化之旅*中的*介绍正则表达式*配方

+   *下载网页*配方

# 爬取网页

考虑到超链接页面的性质，从已知位置开始并跟随链接到其他页面是在抓取网页时的重要工具。

为此，我们爬取页面寻找一个小短语，并打印包含它的任何段落。我们只会在属于同一网站的页面中搜索。即只有以 www.somesite.com 开头的 URL。我们不会跟踪外部网站的链接。

# 准备工作

这个食谱是基于介绍的概念构建的，因此它将下载和解析页面以搜索链接并继续下载。

在爬取网页时，记得在下载时设置限制。很容易爬取太多页面。任何查看维基百科的人都可以证实，互联网是潜在无限的。

我们将使用一个准备好的示例，该示例可在 GitHub 存储库中找到：[`github.com/PacktPublishing/Python-Automation-Cookbook/tree/master/Chapter03/test_site`](https://github.com/PacktPublishing/Python-Automation-Cookbook/tree/master/Chapter03/test_site)。下载整个站点并运行包含的脚本。

```py
$ python simple_delay_server.py
```

它在 URL`http://localhost:8000`中提供站点。您可以在浏览器上查看它。这是一个简单的博客，有三篇文章。大部分内容都不那么有趣，但我们添加了一些包含关键字`python`的段落。

![](img/28ceb1d7-d5a3-47b8-b776-e6a0d1bf8bcb.png)

# 如何做到这一点...

1.  完整的脚本`crawling_web_step1.py`可以在 GitHub 的以下链接找到：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter03/crawling_web_step1.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter03/crawling_web_step1.py)。最相关的部分显示在这里：

```py
...

def process_link(source_link, text):
    logging.info(f'Extracting links from {source_link}')
    parsed_source = urlparse(source_link)
    result = requests.get(source_link)
    # Error handling. See GitHub for details
    ...
    page = BeautifulSoup(result.text, 'html.parser')
    search_text(source_link, page, text)
    return get_links(parsed_source, page)

def get_links(parsed_source, page):
    '''Retrieve the links on the page'''
    links = []
    for element in page.find_all('a'):
        link = element.get('href')
        # Validate is a valid link. See GitHub for details
        ...
        links.append(link)
    return links
```

1.  搜索对`python`的引用，返回包含它和段落的 URL 列表。请注意，由于损坏的链接，有一些错误：

```py
$ python crawling_web_step1.py https://localhost:8000/ -p python
Link http://localhost:8000/: --> A smaller article , that contains a reference to Python
Link http://localhost:8000/files/5eabef23f63024c20389c34b94dee593-1.html: --> A smaller article , that contains a reference to Python
Link http://localhost:8000/files/33714fc865e02aeda2dabb9a42a787b2-0.html: --> This is the actual bit with a python reference that we are interested in.
Link http://localhost:8000/files/archive-september-2018.html: --> A smaller article , that contains a reference to Python
Link http://localhost:8000/index.html: --> A smaller article , that contains a reference to Python
```

1.  另一个很好的搜索词是`crocodile`。试一下：

```py
$ python crawling_web_step1.py http://localhost:8000/ -p crocodile
```

# 它是如何工作的...

让我们看看脚本的每个组件：

1.  一个循环，遍历`main`函数中找到的所有链接：

请注意，有 10 页的检索限制，并且正在检查是否已经添加了要添加的任何新链接。

请注意这两件事是有限制的。我们不会下载相同的链接两次，我们会在某个时候停止。

1.  在`process_link`函数中下载和解析链接：

它下载文件，并检查状态是否正确，以跳过诸如损坏链接之类的错误。它还检查类型（如`Content-Type`中描述的）是否为 HTML 页面，以跳过 PDF 和其他格式。最后，它将原始 HTML 解析为`BeautifulSoup`对象。

它还使用`urlparse`解析源链接，以便在步骤 4 中跳过所有对外部来源的引用。`urlparse`将 URL 分解为其组成元素：

```py
>>> from urllib.parse import urlparse
>>> >>> urlparse('http://localhost:8000/files/b93bec5d9681df87e6e8d5703ed7cd81-2.html')
ParseResult(scheme='http', netloc='localhost:8000', path='/files/b93bec5d9681df87e6e8d5703ed7cd81-2.html', params='', query='', fragment='')
```

1.  在`search_text`函数中找到要搜索的文本：

它在解析的对象中搜索指定的文本。请注意，搜索是作为`regex`进行的，仅在文本中进行。它打印出结果的匹配项，包括`source_link`，引用找到匹配项的 URL：

```py
for element in page.find_all(text=re.compile(text)):
    print(f'Link {source_link}: --> {element}')
```

1.  **`get_links`**函数检索页面上的所有链接：

它在解析页面中搜索所有`<a>`元素，并检索`href`元素，但只有具有这些`href`元素并且是完全合格的 URL（以`http`开头）的元素。这将删除不是 URL 的链接，例如`'#'`链接，或者是页面内部的链接。

还进行了额外的检查，以检查它们是否与原始链接具有相同的来源，然后将它们注册为有效链接。`netloc`属性允许检测链接是否来自与步骤 2 中生成的解析 URL 相同的 URL 域。

我们不会跟踪指向不同地址的链接（例如[`www.google.com`](http://www.google.com)）。

最后，链接被返回，它们将被添加到步骤 1 中描述的循环中。

# 还有更多...

还可以进一步强制执行其他过滤器，例如丢弃所有以`.pdf`结尾的链接，这意味着它们是 PDF 文件：

```py
# In get_links
if link.endswith('pdf'):
  continue
```

还可以使用`Content-Type`来确定以不同方式解析返回的对象。例如，PDF 结果（`Content-Type: application/pdf`）将没有有效的`response.text`对象进行解析，但可以用其他方式解析。其他类型也是如此，例如 CSV 文件（`Content-Type: text/csv`）或可能需要解压缩的 ZIP 文件（`Content-Type: application/zip`）。我们将在后面看到如何处理这些。

# 另请参阅

+   *下载网页*食谱

+   *解析 HTML*食谱

# 订阅 Feed

RSS 可能是互联网上最大的“秘密”。虽然它的辉煌时刻似乎是在 2000 年代，现在它不再处于聚光灯下，但它可以轻松订阅网站。它存在于许多地方，非常有用。

在其核心，RSS 是一种呈现有序引用（通常是文章，但也包括其他元素，如播客剧集或 YouTube 出版物）和发布时间的方式。这使得很自然地知道自上次检查以来有哪些新文章，以及呈现一些关于它们的结构化数据，如标题和摘要。

在这个食谱中，我们将介绍`feedparser`模块，并确定如何从 RSS Feed 中获取数据。

**RSS**不是唯一可用的 Feed 格式。还有一种称为**Atom**的格式，但两者几乎是等效的。`feedparser`也能够解析它，因此两者可以不加区分地使用。

# 准备工作

我们需要将`feedparser`依赖项添加到我们的`requirements.txt`文件中并重新安装它：

```py
$ echo "feedparser==5.2.1" >> requirements.txt
$ pip install -r requirements.txt
```

几乎所有涉及出版物的页面上都可以找到 Feed URL，包括博客、新闻、播客等。有时很容易找到它们，但有时它们会隐藏得有点深。可以通过`feed`或`RSS`进行搜索。

大多数报纸和新闻机构都将它们的 RSS Feed 按主题划分。我们将使用**纽约时报**主页 Feed 作为示例，[`rss.nytimes.com/services/xml/rss/nyt/HomePage.xml`](http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml)。主要 Feed 页面上还有更多可用的 Feed：[`archive.nytimes.com/www.nytimes.com/services/xml/rss/index.html`](https://archive.nytimes.com/www.nytimes.com/services/xml/rss/index.html?mcubz=0)。

请注意，Feed 可能受到使用条款和条件的约束。在纽约时报的情况下，它们在主要 Feed 页面的末尾有描述。

请注意，此 Feed 经常更改，这意味着链接的条目将与本书中的示例不同。

# 如何做...

1.  导入`feedparser`模块，以及`datetime`、`delorean`和`requests`：

```py
import feedparser
import datetime
import delorean
import requests
```

1.  解析 Feed（它将自动下载）并检查其上次更新时间。Feed 信息，如 Feed 的标题，可以在`feed`属性中获取：

```py
>>> rss = feedparser.parse('http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml')
```

```py
>>> rss.updated
'Sat, 02 Jun 2018 19:50:35 GMT'
```

1.  获取新于六小时的条目：

```py
>>> time_limit = delorean.parse(rss.updated) - datetime.timedelta(hours=6)
>>> entries = [entry for entry in rss.entries if delorean.parse(entry.published) > time_limit]
```

1.  条目将比总条目少，因为返回的条目中有些条目的时间已经超过六个小时：

```py
>>> len(entries)
10
>>> len(rss.entries)
44
```

1.  检索条目的信息，如`title`。完整的条目 URL 可作为`link`获取。探索此特定 Feed 中的可用信息：

```py
>>> entries[5]['title']
'Loose Ends: How to Live to 108'
>>> entries[5]['link']
'https://www.nytimes.com/2018/06/02/opinion/sunday/how-to-live-to-108.html?partner=rss&emc=rss'
>>> requests.get(entries[5].link)
<Response [200]>
```

# 工作原理...

解析的`feed`对象包含条目的信息，以及有关 Feed 本身的一般信息，例如更新时间。`feed`信息可以在`feed`属性中找到：

```py
>>> rss.feed.title
'NYT > Home Page'
```

每个条目都像一个字典，因此很容易检索字段。它们也可以作为属性访问，但将它们视为键可以获取所有可用的字段：

```py
>>> entries[5].keys()
dict_keys(['title', 'title_detail', 'links', 'link', 'id', 'guidislink', 'media_content', 'summary', 'summary_detail', 'media_credit', 'credit', 'content', 'authors', 'author', 'author_detail', 'published', 'published_parsed', 'tags'])
```

处理 Feed 的基本策略是解析它们并浏览条目，快速检查它们是否有趣，例如通过检查*描述*或*摘要*。如果它们有趣，就使用“链接”字段下载整个页面。然后，为了避免重新检查条目，存储最新的发布日期，下次只检查更新的条目。

# 还有更多...

完整的`feedparser`文档可以在这里找到：[`pythonhosted.org/feedparser/`](https://pythonhosted.org/feedparser/)。

可用的信息可能因源而异。在纽约时报的例子中，有一个带有标签信息的`tag`字段，但这不是标准的。至少，条目将有一个标题，一个描述和一个链接。

RSS 订阅也是筛选自己的新闻来源的好方法。有很好的订阅阅读器。

# 另请参阅

+   第一章中的*安装第三方软件包*配方，*让我们开始我们的自动化之旅*

+   *下载网页*的配方

# 访问 Web API

通过 Web 可以创建丰富的接口，通过 HTTP 进行强大的交互。最常见的接口是使用 JSON 的 RESTful API。这些基于文本的接口易于理解和编程，并使用通用技术，**与语言无关**，这意味着它们可以在任何具有 HTTP`client`模块的编程语言中访问，当然包括 Python。

除了 JSON 之外，还使用了其他格式，例如 XML，但 JSON 是一种非常简单和可读的格式，非常适合转换为 Python 字典（以及其他语言的等价物）。JSON 目前是 RESTful API 中最常见的格式。在这里了解更多关于 JSON 的信息：[`www.json.org/`](https://www.json.org/)。

RESTful 的严格定义需要一些特征，但更非正式的定义可能是通过 URL 访问资源。这意味着 URL 代表特定的资源，例如报纸上的文章或房地产网站上的属性。然后可以通过 HTTP 方法（`GET`查看，`POST`创建，`PUT`/`PATCH`编辑和`DELETE`删除）来操作资源。

适当的 RESTful 接口需要具有某些特征，并且是创建接口的一种方式，不严格限于 HTTP 接口。您可以在这里阅读更多信息：[`codewords.recurse.com/issues/five/what-restful-actually-means`](https://codewords.recurse.com/issues/five/what-restful-actually-means)。

使用`requests`与它们非常容易，因为它包含本机 JSON 支持。

# 准备就绪

为了演示如何操作 RESTful API，我们将使用示例站点[`jsonplaceholder.typicode.com/`](https://jsonplaceholder.typicode.com/)。它模拟了帖子，评论和其他常见资源的常见情况。我们将使用帖子和评论。要使用的 URL 如下：

```py
# The collection of all posts
/posts
# A single post. X is the ID of the post
/posts/X
# The comments of post X
/posts/X/comments
```

网站为它们中的每一个返回了适当的结果。非常方便！

因为这是一个测试站点，数据不会被创建，但站点将返回所有正确的响应。

# 如何做...

1.  导入`requests`：

```py
>>> import requests
```

1.  获取所有帖子的列表并显示最新帖子：

```py
>>> result = requests.get('https://jsonplaceholder.typicode.com/posts')
>>> result
<Response [200]>
>>> result.json()
# List of 100 posts NOT DISPLAYED HERE
>>> result.json()[-1]
{'userId': 10, 'id': 100, 'title': 'at nam consequatur ea labore ea harum', 'body': 'cupiditate quo est a modi nesciunt soluta\nipsa voluptas error itaque dicta in\nautem qui minus magnam et distinctio eum\naccusamus ratione error aut'}
```

1.  创建一个新帖子。查看新创建资源的 URL。调用还返回资源：

```py
>>> new_post = {'userId': 10, 'title': 'a title', 'body': 'something something'}
>>> result = requests.post('https://jsonplaceholder.typicode.com/posts',
              json=new_post)
>>> result
<Response [201]>
>>> result.json()
{'userId': 10, 'title': 'a title', 'body': 'something something', 'id': 101}
>>> result.headers['Location']
'http://jsonplaceholder.typicode.com/posts/101'
```

注意，创建资源的`POST`请求返回 201，这是创建的适当状态。

1.  使用`GET`获取现有帖子：

```py
>>> result = requests.get('https://jsonplaceholder.typicode.com/posts/2')
>>> result
<Response [200]>
>>> result.json()
{'userId': 1, 'id': 2, 'title': 'qui est esse', 'body': 'est rerum tempore vitae\nsequi sint nihil reprehenderit dolor beatae ea dolores neque\nfugiat blanditiis voluptate porro vel nihil molestiae ut reiciendis\nqui aperiam non debitis possimus qui neque nisi nulla'}
```

1.  使用`PATCH`更新其值。检查返回的资源：

```py
>>> update = {'body': 'new body'}
>>> result = requests.patch('https://jsonplaceholder.typicode.com/posts/2', json=update)
>>> result
<Response [200]>
>>> result.json()
{'userId': 1, 'id': 2, 'title': 'qui est esse', 'body': 'new body'}
```

# 它是如何工作的...

通常访问两种资源。单个资源（`https://jsonplaceholder.typicode.com/posts/X`）和集合（`https://jsonplaceholder.typicode.com/posts`）：

+   集合接受`GET`以检索它们所有，并接受`POST`以创建新资源

+   单个元素接受`GET`以获取元素，`PUT`和`PATCH`以编辑，`DELETE`以删除它们

所有可用的 HTTP 方法都可以在`requests`中调用。在以前的配方中，我们使用了`.get()`，但`.post()`，`.patch()`，`.put()`和`.delete()`也可用。

返回的响应对象具有`.json()`方法，用于解码 JSON 结果。

同样，发送信息时，有一个`json`参数可用。这将字典编码为 JSON 并将其发送到服务器。数据需要遵循资源的格式，否则可能会引发错误。

`GET`和`DELETE`不需要数据，而`PATCH`、`PUT`和`POST`需要数据。

将返回所引用的资源，其 URL 在标头位置可用。这在创建新资源时非常有用，因为其 URL 事先是未知的。

`PATCH`和`PUT`之间的区别在于后者替换整个资源，而前者进行部分更新。

# 还有更多...

RESTful API 非常强大，但也具有巨大的可变性。请查看特定 API 的文档，了解其详细信息。

# 另请参阅

+   *下载网页*配方

+   第一章中的*安装第三方软件包*配方，*让我们开始我们的自动化之旅*

# 与表单交互

网页中常见的一个元素是表单。表单是将值发送到网页的一种方式，例如，在博客文章上创建新评论或提交购买。

浏览器呈现表单，以便您可以输入值并在按下提交或等效按钮后以单个操作发送它们。我们将在此配方中看到如何以编程方式创建此操作。

请注意，向网站发送数据通常比从网站接收数据更明智。例如，向网站发送自动评论非常符合**垃圾邮件**的定义。这意味着自动化和包含安全措施可能更加困难。请仔细检查您尝试实现的是否是有效的、符合道德的用例。

# 准备就绪

我们将针对测试服务器[`httpbin.org/forms/post`](https://httpbin.org/forms/post)进行操作，该服务器允许我们发送测试表单并返回提交的信息。

以下是一个订购比萨的示例表单：

![](img/3fe93fc3-a5a2-450e-ba88-003566953c79.png)

您可以手动填写表单并查看它以 JSON 格式返回信息，包括浏览器使用等额外信息。

以下是生成的 Web 表单的前端：

![](img/32f55d90-7967-4733-9b79-a3a6eca11557.png)

以下图像是生成的 Web 表单的后端：

![](img/1f516fd0-d5de-484c-8875-9369cbe29b1b.png)

我们需要分析 HTML 以查看表单的接受数据。检查源代码，显示如下：

![](img/a2f505fc-424b-47f5-a94f-93541d72e0b8.png)源代码

检查输入的名称，`custname`、`custtel`、`custemail`、`size`（单选按钮选项）、`topping`（多选复选框）、`delivery`（时间）和`comments`。

# 如何做...

1.  导入`requests`、`BeautifulSoup`和`re`模块：

```py
>>> import requests
>>> from bs4 import BeautifulSoup
>>> import re
```

1.  检索表单页面，解析它，并打印输入字段。检查发布 URL 是否为`/post`（而不是`/forms/post`）：

```py
>>> response = requests.get('https://httpbin.org/forms/post')
>>> page = BeautifulSoup(response.text)
>>> form = soup.find('form')
>>> {field.get('name') for field in form.find_all(re.compile('input|textarea'))}
{'delivery', 'topping', 'size', 'custemail', 'comments', 'custtel', 'custname'}
```

请注意，`textarea`是有效输入，也是在 HTML 格式中定义的。

1.  准备要发布的数据作为字典。检查值是否与表单中定义的相同：

```py
>>> data = {'custname': "Sean O'Connell", 'custtel': '123-456-789', 'custemail': 'sean@oconnell.ie', 'size': 'small', 'topping': ['bacon', 'onion'], 'delivery': '20:30', 'comments': ''}
```

1.  发布值并检查响应是否与浏览器中返回的相同：

```py
>>> response = requests.post('https://httpbin.org/post', data)
>>> response
<Response [200]>
>>> response.json()
{'args': {}, 'data': '', 'files': {}, 'form': {'comments': '', 'custemail': 'sean@oconnell.ie', 'custname': "Sean O'Connell", 'custtel': '123-456-789', 'delivery': '20:30', 'size': 'small', 'topping': ['bacon', 'onion']}, 'headers': {'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate', 'Connection': 'close', 'Content-Length':
```

```py
'140', 'Content-Type': 'application/x-www-form-urlencoded', 'Host': 'httpbin.org', 'User-Agent': 'python-requests/2.18.3'}, 'json': None, 'origin': '89.100.17.159', 'url': 'https://httpbin.org/post'}
```

# 它是如何工作的...

`requests`直接接受以正确方式发送数据。默认情况下，它以`application/x-www-form-urlencoded`格式发送`POST`数据。

将其与*访问 Web API*配方进行比较，其中数据是使用参数`json`以 JSON 格式明确发送的。这使得`Content-Type`为`application/json`而不是`application/x-www-form-urlencoded`。

这里的关键是尊重表单的格式和可能返回错误的可能值，通常是 400 错误。

# 还有更多...

除了遵循表单的格式和输入有效值之外，在处理表单时的主要问题是防止垃圾邮件和滥用行为的多种方式。

非常常见的限制是确保在提交表单之前下载了表单，以避免提交多个表单或**跨站点请求伪造**（**CSRF**）。

CSRF，意味着从一个页面对另一个页面发出恶意调用，利用您的浏览器已经经过身份验证，这是一个严重的问题。例如，进入一个利用您已经登录到银行页面执行操作的小狗网站。这是一个很好的描述：[`stackoverflow.com/a/33829607`](https://stackoverflow.com/a/33829607)。浏览器中的新技术默认情况下有助于解决这些问题。

要获取特定令牌，您需要首先下载表单，如配方中所示，获取 CSRF 令牌的值，并重新提交。请注意，令牌可以有不同的名称；这只是一个例子：

```py
>>> form.find(attrs={'name': 'token'}).get('value')
'ABCEDF12345'
```

# 另请参阅

+   *下载网页*的配方

+   *解析 HTML*的配方

# 使用 Selenium 进行高级交互

有时，除了真实的东西外，什么都行不通。 Selenium 是一个实现 Web 浏览器自动化的项目。它被构想为一种自动测试的方式，但也可以用于自动化与网站的交互。

Selenium 可以控制 Safari、Chrome、Firefox、Internet Explorer 或 Microsoft Edge，尽管它需要为每种情况安装特定的驱动程序。我们将使用 Chrome。

# 准备工作

我们需要为 Chrome 安装正确的驱动程序，称为`chromedriver`。它在这里可用：[`sites.google.com/a/chromium.org/chromedriver/`](https://sites.google.com/a/chromium.org/chromedriver/)。它适用于大多数平台。它还要求您已安装 Chrome：[`www.google.com/chrome/`](https://www.google.com/chrome/)。

将`selenium`模块添加到`requirements.txt`并安装它：

```py
$ echo "selenium==3.12.0" >> requirements.txt
$ pip install -r requirements.txt
```

# 如何做...

1.  导入 Selenium，启动浏览器，并加载表单页面。将打开一个反映操作的页面：

```py
>>> from selenium import webdriver
>>> browser = webdriver.Chrome()
>>> browser.get('https://httpbin.org/forms/post')
```

请注意，Chrome 中的横幅由自动化测试软件控制。

1.  在“客户名称”字段中添加一个值。请记住它被称为`custname`：

```py
>>> custname = browser.find_element_by_name("custname")
>>> custname.clear()
>>> custname.send_keys("Sean O'Connell")
```

表单将更新：

![](img/13d8d74f-9f2e-4ef9-87e1-29deb1c70b95.png)

1.  选择披萨大小为`medium`：

```py
>>> for size_element in browser.find_elements_by_name("size"):
...     if size_element.get_attribute('value') == 'medium':
...         size_element.click()
...
>>>
```

这将改变披萨大小比例框。

1.  添加`bacon`和`cheese`：

```py
>>> for topping in browser.find_elements_by_name('topping'):
...     if topping.get_attribute('value') in ['bacon', 'cheese']:
...         topping.click()
...
>>>
```

最后，复选框将显示为已标记：

![](img/789403c5-3c8f-4c14-b9c5-6318286ae9c1.png)

1.  提交表单。页面将提交，结果将显示：

```py
>>> browser.find_element_by_tag_name('form').submit()
```

表单将被提交，服务器的结果将显示：

![](img/b9063025-2521-4820-a446-90e9d89b4dbd.png)

1.  关闭浏览器：

```py
>>> browser.quit()
```

# 它是如何工作的...

*如何做...*部分的第 1 步显示了如何创建一个 Selenium 页面并转到特定的 URL。

Selenium 的工作方式与 Beautiful Soup 类似。选择适当的元素，然后操纵它。 Selenium 中的选择器的工作方式与 Beautiful Soup 中的选择器的工作方式类似，最常见的选择器是`find_element_by_id`、`find_element_by_class_name`、`find_element_by_name`、`find_element_by_tag_name`和`find_element_by_css_selector`。还有等效的`find_elements_by_X`，它们返回一个列表而不是第一个找到的元素（`find_elements_by_tag_name`、`find_elements_by_name`等）。当检查元素是否存在时，这也很有用。如果没有元素，`find_element`将引发错误，而`find_elements`将返回一个空列表。

可以通过`.get_attribute()`获取元素上的数据，用于 HTML 属性（例如表单元素上的值）或`.text`。

可以通过模拟发送按键输入文本来操作元素，方法是`.send_keys()`，点击是`.click()`，如果它们接受，可以使用`.submit()`进行提交。`.submit()`将在表单上搜索适当的提交，`.click()`将以与鼠标点击相同的方式选择/取消选择。

最后，第 6 步关闭浏览器。

# 还有更多...

这是完整的 Selenium 文档：[`selenium-python.readthedocs.io/`](http://selenium-python.readthedocs.io/)。

对于每个元素，都可以提取额外的信息，例如`.is_displayed()`或`.is_selected()`。可以使用`.find_element_by_link_text()`和`.find_element_by_partial_link_text()`来搜索文本。

有时，打开浏览器可能会不方便。另一种选择是以无头模式启动浏览器，并从那里操纵它，就像这样：

```py
>>> from selenium.webdriver.chrome.options import Options
>>> chrome_options = Options()
>>> chrome_options.add_argument("--headless")
>>> browser = webdriver.Chrome(chrome_options=chrome_options)
>>> browser.get('https://httpbin.org/forms/post')
```

页面不会显示。但是可以使用以下命令保存截图：

```py
>>> browser.save_screenshot('screenshot.png')
```

# 另请参阅

+   *解析 HTML*配方

+   *与表单交互*配方

# 访问受密码保护的页面

有时，网页对公众不开放，而是以某种方式受到保护。最基本的方面是使用基本的 HTTP 身份验证，它集成到几乎每个 Web 服务器中，并且是用户/密码模式。

# 准备就绪

我们可以在[`httpbin.org`](https://httpbin.org)中测试这种身份验证。

它有一个路径，`/basic-auth/{user}/{password}`，强制进行身份验证，用户和密码已声明。这对于理解身份验证的工作原理非常方便。

# 如何做...

1.  导入`requests`：

```py
>>> import requests
```

1.  使用错误的凭据对 URL 进行`GET`请求。注意，我们在 URL 上设置了凭据为`user`和`psswd`：

```py
>>> requests.get('https://httpbin.org/basic-auth/user/psswd', 
                 auth=('user', 'psswd'))
<Response [200]>
```

1.  使用错误的凭据返回 401 状态码（未经授权）：

```py
>>> requests.get('https://httpbin.org/basic-auth/user/psswd', 
                 auth=('user', 'wrong'))
<Response [401]>
```

1.  凭据也可以直接通过 URL 传递，在服务器之前用冒号和`@`符号分隔，就像这样：

```py
>>> requests.get('https://user:psswd@httpbin.org/basic-auth/user/psswd')
<Response [200]>
>>> requests.get('https://user:wrong@httpbin.org/basic-auth/user/psswd')
<Response [401]>
```

# 它的工作原理...

由于 HTTP 基本身份验证在各处都受支持，因此从“请求”获得支持非常容易。

*如何做...*部分的第 2 步和第 4 步显示了如何提供正确的密码。第 3 步显示了密码错误时会发生什么。

请记住始终使用 HTTPS，以确保密码的发送保密。如果使用 HTTP，密码将在网络上以明文发送。

# 还有更多...

将用户和密码添加到 URL 中也适用于浏览器。尝试直接访问页面，看到一个框显示要求输入用户名和密码：

![](img/b0d91d6e-fa37-4084-82f1-d825d15441bc.png)用户凭据页面

在使用包含用户和密码的 URL 时，`https://user:psswd@httpbin.org/basic-auth/user/psswd`，对话框不会出现，它会自动进行身份验证。

如果您需要访问多个页面，可以在“请求”中创建一个会话，并将身份验证参数设置为避免在各处输入它们：

```py
>>> s = requests.Session()
>>> s.auth = ('user', 'psswd')
>>> s.get('https://httpbin.org/basic-auth/user/psswd')
<Response [200]>
```

# 另请参阅

+   *下载网页*配方

+   *访问 Web API*配方

# 加快网页抓取速度

从网页下载信息所花费的大部分时间通常是在等待。请求从我们的计算机发送到将处理它的任何服务器，直到响应被组成并返回到我们的计算机，我们无法做太多事情。

在书中执行配方时，您会注意到`requests`调用通常需要等待大约一两秒。但是计算机可以在等待时做其他事情，包括同时发出更多请求。在这个配方中，我们将看到如何并行下载页面列表并等待它们全部准备就绪。我们将使用一个故意缓慢的服务器来说明这一点。

# 准备就绪

我们将获得代码来爬取和搜索关键字，利用 Python 3 的`futures`功能同时下载多个页面。

`future`是表示值承诺的对象。这意味着在代码在后台执行时，您立即收到一个对象。只有在明确请求其`.result()`时，代码才会阻塞，直到获取它。

要生成一个`future`，你需要一个后台引擎，称为**executor**。一旦创建，`submit`一个函数和参数给它以检索一个`future`。结果的检索可以被延迟，直到需要，允许连续生成多个`futures`，并等待直到所有都完成，以并行执行它们，而不是创建一个，等待它完成，再创建另一个，依此类推。

有几种方法可以创建一个 executor；在这个示例中，我们将使用`ThreadPoolExecutor`，它将使用线程。

我们将以一个准备好的示例为例，该示例可在 GitHub 存储库中找到：[`github.com/PacktPublishing/Python-Automation-Cookbook/tree/master/Chapter03/test_site`](https://github.com/PacktPublishing/Python-Automation-Cookbook/tree/master/Chapter03/test_site)。下载整个站点并运行包含的脚本。

```py
$ python simple_delay_server.py -d 2
```

这是 URL 为`http://localhost:8000`的站点。你可以在浏览器上查看它。这是一个简单的博客，有三篇文章。大部分内容都不那么有趣，但我们添加了几段包含关键字`python`的段落。参数`-d 2`使服务器故意变慢，模拟一个糟糕的连接。

# 如何做...

1.  编写以下脚本`speed_up_step1.py`。完整的代码可以在 GitHub 的`Chapter03`目录下找到：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter03/speed_up_step1.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter03/speed_up_step1.py)。这里只列出了最相关的部分。它基于`crawling_web_step1.py`。

```py
...
def process_link(source_link, text):
    ...
    return source_link, get_links(parsed_source, page)
...

def main(base_url, to_search, workers):
    checked_links = set()
    to_check = [base_url]
    max_checks = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while to_check:
            futures = [executor.submit(process_link, url, to_search)
                       for url in to_check]
            to_check = []
            for data in concurrent.futures.as_completed(futures):
                link, new_links = data.result()

                checked_links.add(link)
                for link in new_links:
                    if link not in checked_links and link not in to_check:
                        to_check.append(link)

                max_checks -= 1
                if not max_checks:
                    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ...
    parser.add_argument('-w', type=int, help='Number of workers',
                        default=4)
    args = parser.parse_args()

    main(args.u, args.p, args.w)
```

1.  请注意`main`函数中的差异。还添加了一个额外的参数（并发工作人员的数量），并且`process_link`函数现在返回源链接。

1.  运行`crawling_web_step1.py`脚本以获取时间基准。注意这里已经删除了输出以保持清晰：

```py
$ time python crawling_web_step1.py http://localhost:8000/
... REMOVED OUTPUT
real 0m12.221s
user 0m0.160s
sys 0m0.034s
```

1.  使用比原始版本慢的一个工作人员运行新脚本：

```py
$ time python speed_up_step1.py -w 1
... REMOVED OUTPUT
real 0m16.403s
user 0m0.181s
sys 0m0.068s
```

1.  增加工作人员的数量：

```py
$ time python speed_up_step1.py -w 2
... REMOVED OUTPUT
real 0m10.353s
user 0m0.199s
sys 0m0.068s
```

1.  增加更多的工作人员会减少时间。

```py
$ time python speed_up_step1.py -w 5
... REMOVED OUTPUT
real 0m6.234s
user 0m0.171s
sys 0m0.040s
```

# 它是如何工作的...

创建并发请求的主要引擎是主函数。请注意，代码的其余部分基本上没有改动（除了在`process_link`函数中返回源链接）。

这种变化在适应并发时实际上是相当常见的。并发任务需要返回所有相关的数据，因为它们不能依赖于有序的上下文。

这是处理并发引擎的代码的相关部分：

```py
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    while to_check:
        futures = [executor.submit(process_link, url, to_search)
                   for url in to_check]
        to_check = []
        for data in concurrent.futures.as_completed(futures):
            link, new_links = data.result()

            checked_links.add(link)
            for link in new_links:
                if link not in checked_links and link not in to_check:
                    to_check.append(link)

             max_checks -= 1
             if not max_checks:
                return
```

`with`上下文创建了一个指定数量的工作人员池。在内部，创建了一个包含所有要检索的 URL 的`futures`列表。`.as_completed()`函数返回已完成的`futures`，然后进行一些工作，处理获取到的新链接，并检查它们是否需要被添加到检索中。这个过程类似于*Crawling the web*示例中呈现的过程。

该过程会再次开始，直到检索到足够的链接或没有链接可检索为止。请注意，链接是批量检索的；第一次，处理基本链接并检索所有链接。在第二次迭代中，将请求所有这些链接。一旦它们都被下载，将处理一个新的批次。

处理并发请求时，请记住它们可以在两次执行之间改变顺序。如果一个请求花费的时间稍微多一点或少一点，那可能会影响检索信息的顺序。因为我们在下载 10 页后停止，这也意味着这 10 页可能是不同的。

# 还有更多...

Python 的完整`futures`文档可以在这里找到：[`docs.python.org/3/library/concurrent.futures.html`](https://docs.python.org/3/library/concurrent.futures.html)。

正如您在*如何做…*部分的第 4 和第 5 步中所看到的，正确确定工作人员的数量可能需要一些测试。一些数字可能会使过程变慢，因为管理增加了。不要害怕尝试！

在 Python 世界中，还有其他方法可以进行并发的 HTTP 请求。有一个原生请求模块，允许我们使用`futures`，名为`requests-futures`。它可以在这里找到：[`github.com/ross/requests-futures`](https://github.com/ross/requests-futures)。

另一种选择是使用异步编程。最近，这种工作方式引起了很多关注，因为在处理许多并发调用时可以非常高效，但编码的方式与传统方式不同，需要一些时间来适应。Python 包括`asyncio`模块来进行这种工作，还有一个名为`aiohttp`的好模块来处理 HTTP 请求。您可以在这里找到有关`aiohttp`的更多信息：[`aiohttp.readthedocs.io/en/stable/client_quickstart.html`](https://aiohttp.readthedocs.io/en/stable/client_quickstart.html)。

关于异步编程的良好介绍可以在这篇文章中找到：[`djangostars.com/blog/asynchronous-programming-in-python-asyncio/`](https://djangostars.com/blog/asynchronous-programming-in-python-asyncio/)。

# 另请参阅

+   *爬取网页*配方

+   *下载网页*配方
