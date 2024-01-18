# 网络爬虫-从网站提取有用的数据

在本章中，您将学习有关网络爬虫的知识。您还将学习Python中的`beautifulsoup`库，该库用于从网站提取信息。

在本章中，我们将涵盖以下主题：

+   什么是网络爬虫？

+   数据提取

+   从维基百科提取信息

# 什么是网络爬虫？

网络爬虫是从网站提取信息的技术。这种技术用于将非结构化数据转换为结构化数据。

网络爬虫的用途是从网站提取数据。提取的信息以本地文件的形式保存在您的系统上，您也可以以表格格式将其存储到数据库中。网络爬虫软件直接使用HTTP或Web浏览器访问**万维网**（**WWW**）。这是使用网络爬虫或机器人实施的自动化过程。

爬取网页涉及获取页面，然后提取数据。网络爬虫获取网页。网络爬虫是网络爬取中的一个必不可少的组件。获取后，进行提取。您可以搜索、解析、将数据保存到表中，并重新格式化页面。

# 数据提取

在本节中，我们将看到实际的数据提取过程。Python具有`beautifulsoup`库来执行数据提取任务。我们还将使用Python的requests库。

首先，我们必须安装这两个库。运行以下命令以安装`requests`和`beautifulsoup`库：

```py
$ pip3 install requests $ pip3 install beautifulsoup4
```

# requests库

使用`requests`库是在我们的Python脚本中以人类可读的格式使用HTTP。我们可以使用Python中的`requests`库下载页面。`requests`库有不同类型的请求。在这里，我们将学习`GET`请求。`GET`请求用于从Web服务器检索信息。`GET`请求下载指定网页的HTML内容。每个请求都有一个状态代码。状态代码与我们向服务器发出的每个请求一起返回。这些状态代码为我们提供了关于请求发生了什么的信息。状态代码的类型在此列出：

+   `200`：表示一切正常，并返回结果（如果有的话）

+   `301`：表示服务器正在重定向到不同的端点，如果已经切换了域名或端点名称必须更改

+   `400`：表示您发出了一个错误的请求

+   `401`：表示我们未经授权

+   `403`：表示您正在尝试访问被禁止的资源

+   `404`：表示您正在尝试访问的资源在服务器上不可用

# beautifulsoup库

`beautifulsoup`是Python中用于网络爬虫的库。它具有用于搜索、导航和修改的简单方法。它只是一个工具包，用于从网页中提取所需的数据。

现在，要在脚本中使用`requests`和`beautifulsoup`功能，您必须使用`import`语句导入这两个库。现在，我们将看一个解析网页的例子。在这里，我们将解析一个网页，这是来自IMDb网站的头条新闻页面。为此，请创建一个`parse_web_page.py`脚本，并在其中编写以下内容：

```py
import requests from bs4 import BeautifulSoup page_result = requests.get('https://www.imdb.com/news/top?ref_=nv_nw_tp') parse_obj = BeautifulSoup(page_result.content, 'html.parser') print(parse_obj)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 parse_web_page.py Output: <!DOCTYPE html> <html  > <head> <meta charset="utf-8"/> <meta content="IE=edge" http-equiv="X-UA-Compatible"/> <meta content="app-id=342792525, app-argument=imdb:///?src=mdot" name="apple-itunes-app"/> <script type="text/javascript">var IMDbTimer={starttime: new Date().getTime(),pt:'java'};</script> <script>
 if (typeof uet == 'function') { uet("bb", "LoadTitle", {wb: 1}); } </script> <script>(function(t){ (t.events = t.events || {})["csm_head_pre_title"] = new Date().getTime(); })(IMDbTimer);</script> <title>Top News - IMDb</title> <script>(function(t){ (t.events = t.events || {})["csm_head_post_title"] = new Date().getTime(); })(IMDbTimer);</script> <script>
 if (typeof uet == 'function') { uet("be", "LoadTitle", {wb: 1}); } </script> <script>
 if (typeof uex == 'function') { uex("ld", "LoadTitle", {wb: 1}); } </script> <link href="https://www.imdb.com/news/top" rel="canonical"/> <meta content="http://www.imdb.com/news/top" property="og:url"> <script>
 if (typeof uet == 'function') { uet("bb", "LoadIcons", {wb: 1}); }
```

在前面的示例中，我们收集了一个页面并使用`beautifulsoup`解析了它。首先，我们导入了`requests`和`beautifulsoup`模块。然后，我们使用`GET`请求收集了URL，并将该URL分配给`page_result`变量。接下来，我们创建了一个`beautifulsoup`对象`parse_obj`。这个对象将使用来自requests的`page_result`.content作为参数，然后使用`html.parser`解析页面。

现在，我们将从一个类和一个标签中提取内容。要执行此操作，请转到您的网络浏览器，右键单击要提取的内容，然后向下滚动，直到您看到**检查**选项。单击它，您将获得类名。在程序中提到它并运行您的脚本。为此，请创建一个`extract_from_class.py`脚本，并在其中编写以下内容：

```py
import requests from bs4 import BeautifulSoup page_result = requests.get('https://www.imdb.com/news/top?ref_=nv_nw_tp') parse_obj = BeautifulSoup(page_result.content, 'html.parser') top_news = parse_obj.find(class_='news-article__content') print(top_news)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 extract_from_class.py Output : <div class="news-article__content"> <a href="/name/nm4793987/">Issa Rae</a> and <a href="/name/nm0000368/">Laura Dern</a> are teaming up to star in a limited series called “The Dolls” currently in development at <a href="/company/co0700043/">HBO</a>.<br/><br/>Inspired by true events, the series recounts the aftermath of Christmas Eve riots in two small Arkansas towns in 1983, riots which erupted over Cabbage Patch Dolls. The series explores class, race, privilege and what it takes to be a “good mother.”<br/><br/>Rae will serve as a writer and executive producer on the series in addition to starring, with Dern also executive producing. <a href="/name/nm3308450/">Laura Kittrell</a> and <a href="/name/nm4276354/">Amy Aniobi</a> will also serve as writers and co-executive producers. <a href="/name/nm0501536/">Jayme Lemons</a> of Dern’s <a href="/company/co0641481/">Jaywalker Pictures</a> and <a href="/name/nm3973260/">Deniese Davis</a> of <a href="/company/co0363033/">Issa Rae Productions</a> will also executive produce.<br/><br/>Both Rae and Dern currently star in HBO shows, with Dern appearing in the acclaimed drama “<a href="/title/tt3920596/">Big Little Lies</a>” and Rae starring in and having created the hit comedy “<a href="/title/tt5024912/">Insecure</a>.” Dern also recently starred in the film “<a href="/title/tt4015500/">The Tale</a>,
 </div>
```

在上面的例子中，我们首先导入了requests和`beautifulsoup`模块。然后，我们创建了一个请求对象并为其分配了一个URL。接下来，我们创建了一个`beautifulsoup`对象`parse_obj`。这个对象以requests的`page_result.content`作为参数，然后使用`html.parser`解析页面。接下来，我们使用beautifulsoup的`find()`方法从`'news-article__content'`类中获取内容。

现在，我们将看到从特定标签中提取内容的示例。在这个例子中，我们将从`<a>`标签中提取内容。创建一个`extract_from_tag.py`脚本，并在其中编写以下内容：

```py
import requests from bs4 import BeautifulSoup page_result = requests.get('https://www.imdb.com/news/top?ref_=nv_nw_tp') parse_obj = BeautifulSoup(page_result.content, 'html.parser') top_news = parse_obj.find(class_='news-article__content') top_news_a_content = top_news.find_all('a') print(top_news_a_content)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 extract_from_tag.py Output: [<a href="/name/nm4793987/">Issa Rae</a>, <a href="/name/nm0000368/">Laura Dern</a>, <a href="/company/co0700043/">HBO</a>, <a href="/name/nm3308450/">Laura Kittrell</a>, <a href="/name/nm4276354/">Amy Aniobi</a>, <a href="/name/nm0501536/">Jayme Lemons</a>, <a href="/company/co0641481/">Jaywalker Pictures</a>, <a href="/name/nm3973260/">Deniese Davis</a>, <a href="/company/co0363033/">Issa Rae Productions</a>, <a href="/title/tt3920596/">Big Little Lies</a>, <a href="/title/tt5024912/">Insecure</a>, <a href="/title/tt4015500/">The Tale</a>]
```

在上面的例子中，我们正在从`<a>`标签中提取内容。我们使用`find_all()`方法从`'news-article__content'`类中提取所有`<a>`标签内容。

# 从维基百科中提取信息

在本节中，我们将看到维基百科上興舞形式列表的一个示例。我们将列出所有古典印度舞蹈。为此，请创建一个`extract_from_wikipedia.py`脚本，并在其中编写以下内容：

```py
import requests from bs4 import BeautifulSoup page_result = requests.get('https://en.wikipedia.org/wiki/Portal:History') parse_obj = BeautifulSoup(page_result.content, 'html.parser') h_obj = parse_obj.find(class_='hlist noprint')
h_obj_a_content = h_obj.find_all('a') print(h_obj) print(h_obj_a_content)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 extract_from_wikipedia.py
Output:
<div class="hlist noprint" id="portals-browsebar" style="text-align: center;">
<dl><dt><a href="/wiki/Portal:Contents/Portals" title="Portal:Contents/Portals">Portal topics</a></dt>
<dd><a href="/wiki/Portal:Contents/Portals#Human_activities" title="Portal:Contents/Portals">Activities</a></dd>
<dd><a href="/wiki/Portal:Contents/Portals#Culture_and_the_arts" title="Portal:Contents/Portals">Culture</a></dd>
<dd><a href="/wiki/Portal:Contents/Portals#Geography_and_places" title="Portal:Contents/Portals">Geography</a></dd>
<dd><a href="/wiki/Portal:Contents/Portals#Health_and_fitness" title="Portal:Contents/Portals">Health</a></dd>
<dd><a href="/wiki/Portal:Contents/Portals#History_and_events" title="Portal:Contents/Portals">History</a></dd>
<dd><a href="/wiki/Portal:Contents/Portals#Mathematics_and_logic" title="Portal:Contents/Portals">Mathematics</a></dd>
<dd><a href="/wiki/Portal:Contents/Portals#Natural_and_physical_sciences" title="Portal:Contents/Portals">Nature</a></dd>
<dd><a href="/wiki/Portal:Contents/Portals#People_and_self" title="Portal:Contents/Portals">People</a></dd>
In the preceding example, we extracted the content from Wikipedia. In this example also, we extracted the content from class as well as tag.
....
```

# 摘要

在本章中，您了解了网络爬取的内容。我们了解了用于从网页中提取数据的两个库。我们还从维基百科中提取了信息。

在下一章中，您将学习有关统计数据收集和报告的内容。您将学习有关NumPy模块、数据可视化以及使用图表、图形和图表显示数据的内容。

# 问题

1.  什么是网络爬虫？

1.  什么是网络爬虫？

1.  您能够在登录页面后面抓取数据吗？

1.  你能爬Twitter吗？

1.  是否可能抓取JavaScript页面？如果是，如何？

# 进一步阅读

+   Urllib文档：[https://docs.python.org/3/library/urllib.html](https://docs.python.org/3/library/urllib.html)

+   Mechanize：[https://mechanize.readthedocs.io/en/latest/](https://mechanize.readthedocs.io/en/latest/)

+   Scrapemark：[https://pypi.org/project/scrape/](https://pypi.org/project/scrape/)

+   Scrapy：[https://doc.scrapy.org/en/latest/index.html](https://doc.scrapy.org/en/latest/index.html)
