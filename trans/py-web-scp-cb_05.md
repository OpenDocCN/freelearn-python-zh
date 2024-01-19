# 抓取 - 行为准则

在本章中，我们将涵盖：

+   抓取的合法性和有礼貌的抓取

+   尊重 robots.txt

+   使用站点地图进行爬行

+   带延迟的爬行

+   使用可识别的用户代理

+   设置每个域的并发请求数量

+   使用自动节流

+   缓存响应

# 介绍

虽然您在技术上可以抓取任何网站，但重要的是要知道抓取是否合法。我们将讨论抓取的法律问题，探讨一般的法律原则，并了解有礼貌地抓取和最大程度地减少对目标网站的潜在损害的最佳做法。

# 抓取的合法性和有礼貌的抓取

这个配方中没有真正的代码。这只是对涉及抓取的法律问题的一些概念的阐述。我不是律师，所以不要把我在这里写的任何东西当作法律建议。我只是指出在使用抓取器时需要关注的一些事情。

# 准备就绪

抓取的合法性分为两个问题：

+   内容所有权

+   拒绝服务

基本上，网上发布的任何内容都是公开阅读的。每次加载页面时，您的浏览器都会从网络服务器下载内容并将其可视化呈现给您。因此，在某种意义上，您和您的浏览器已经在网上查看任何内容。由于网络的性质，因为有人在网上公开发布内容，他们本质上是在要求您获取这些信息，但通常只是为了特定目的。

大问题在于创建直接寻找并复制互联网上的*事物*的自动化工具，*事物*可以是数据、图像、视频或音乐 - 基本上是由他人创建并代表对创建者或所有者有价值的东西。当明确复制项目供您个人使用时，这些项目可能会产生问题，并且在复制并将其用于您或他人的利益时，可能会更有可能产生问题。

视频、书籍、音乐和图像是一些明显引起关注的项目，涉及制作个人或商业用途的副本的合法性。一般来说，如果您从无需授权访问或需要付费访问内容的开放网站（如不需要授权访问或需要付费访问内容的网站）上抓取此类内容，那么您就没问题。还有*公平使用*规则允许在某些情况下重复使用内容，例如在课堂场景中共享少量文件，其中发布供人们学习的知识并没有真正的经济影响。

从网站上抓取*数据*通常是一个更加模糊的问题。我的意思是作为服务提供的信息。从我的经验来看，一个很好的例子是能源价格，这些价格发布在供应商的网站上。这些通常是为了方便客户而提供的，而不是供您自由抓取并将数据用于自己的商业分析服务。如果您只是为了非公开数据库而收集数据，或者只是为了自己的使用而收集数据，那么可能没问题。但是，如果您使用该数据库来驱动自己的网站并以自己的名义分享该内容，那么您可能需要小心。

重点是，查看网站的免责声明/服务条款，了解您可以如何使用这些信息。这应该有记录，但如果没有，那并不意味着您可以肆意妄为。始终要小心并运用常识，因为您正在为自己的目的获取他人的内容。

另一个关注点是我归为拒绝服务的概念，它涉及到收集信息的实际过程以及你收集信息的频率。在网站上手动阅读内容的过程与编写自动机器人不断骚扰网络服务器以获取内容的过程有很大的不同。如果访问频率过高，可能会拒绝其他合法用户访问内容，从而拒绝为他们提供服务。这也可能会增加内容的主机的成本，增加他们的带宽成本，甚至是运行服务器的电费。

一个良好管理的网站将识别这些重复和频繁的访问，并使用诸如基于 IP 地址、标头和 cookie 的规则的 Web 应用程序防火墙关闭它们。在其他情况下，这些可能会被识别，并联系您的 ISP 要求您停止执行这些任务。请记住，您永远不是真正匿名的，聪明的主机可以找出您是谁，确切地知道您访问了什么内容以及何时访问。

# 如何做到这一点

那么，你如何成为一个好的爬虫呢？在本章中，我们将涵盖几个因素：

+   您可以从尊重`robots.txt`文件开始

+   不要爬取您在网站上找到的每个链接，只爬取站点地图中给出的链接。

+   限制您的请求，就像汉·索洛对丘巴卡说的那样：放轻松；或者，不要看起来像您在重复爬取内容

+   让自己被网站识别

# 尊重 robots.txt

许多网站希望被爬取。这是兽性的本质：Web 主机将内容放在其网站上供人类查看。但同样重要的是其他计算机也能看到内容。一个很好的例子是搜索引擎优化（SEO）。SEO 是一个过程，您实际上设计您的网站以便被 Google 等搜索引擎的爬虫爬取，因此您实际上是在鼓励爬取。但与此同时，发布者可能只希望网站的特定部分被爬取，并告诉爬虫不要爬取网站的某些部分，要么是因为不适合分享，要么是因为不重要而浪费了 Web 服务器资源。

通常，您被允许和不被允许爬取的规则包含在大多数网站上的一个名为`robots.txt`的文件中。`robots.txt`是一个可读但可解析的文件，可用于识别您被允许和不被允许爬取的位置。

`robots.txt`文件的格式不幸地不是标准的，任何人都可以进行自己的修改，但是对于格式有很强的共识。`robots.txt`文件通常位于站点的根 URL。为了演示`robots.txt`文件，以下代码包含了亚马逊在[`amazon.com/robots.txt`](http://amazon.com/robots.txt)上提供的摘录。我编辑了它，只显示了重要的概念：

```py
User-agent: *
Disallow: /exec/obidos/account-access-login
Disallow: /exec/obidos/change-style
Disallow: /exec/obidos/flex-sign-in
Disallow: /exec/obidos/handle-buy-box
Disallow: /exec/obidos/tg/cm/member/
Disallow: /gp/aw/help/id=sss
Disallow: /gp/cart
Disallow: /gp/flex

...

Allow: /wishlist/universal*
Allow: /wishlist/vendor-button*
Allow: /wishlist/get-button*

...

User-agent: Googlebot
Disallow: /rss/people/*/reviews
Disallow: /gp/pdp/rss/*/reviews
Disallow: /gp/cdp/member-reviews/
Disallow: /gp/aw/cr/

...
Allow: /wishlist/universal*
Allow: /wishlist/vendor-button*
Allow: /wishlist/get-button*

```

可以看到文件中有三个主要元素：

+   用户代理声明，以下行直到文件结束或下一个用户代理声明将被应用

+   允许爬取的一组 URL

+   禁止爬取的一组 URL

语法实际上非常简单，Python 库存在以帮助我们实现`robots.txt`中包含的规则。我们将使用`reppy`库来尊重`robots.txt`。

# 准备工作

让我们看看如何使用`reppy`库来演示`robots.txt`。有关`reppy`的更多信息，请参阅其 GitHub 页面[`github.com/seomoz/reppy`](https://github.com/seomoz/reppy)。

可以这样安装`reppy`：

```py
pip install reppy
```

但是，我发现在我的 Mac 上安装时出现了错误，需要以下命令：

```py
CFLAGS=-stdlib=libc++ pip install reppy
```

在 Google 上搜索`robots.txt` Python 解析库的一般信息通常会引导您使用 robotparser 库。此库适用于 Python 2.x。对于 Python 3，它已移至`urllib`库。但是，我发现该库在特定情况下报告不正确的值。我将在我们的示例中指出这一点。

# 如何做到这一点

要运行该示例，请执行`05/01_sitemap.py`中的代码。脚本将检查 amazon.com 上是否允许爬取多个 URL。运行时，您将看到以下输出：

```py
True: http://www.amazon.com/
False: http://www.amazon.com/gp/dmusic/
True: http://www.amazon.com/gp/dmusic/promotions/PrimeMusic/
False: http://www.amazon.com/gp/registry/wishlist/
```

# 它是如何工作的

1.  脚本首先通过导入`reppy.robots`开始：

```py
from reppy.robots import Robots
```

1.  然后，代码使用`Robots`来获取 amazon.com 的`robots.txt`。

```py
url = "http://www.amazon.com" robots = Robots.fetch(url + "/robots.txt")
```

1.  使用获取的内容，脚本检查了几个 URL 的可访问性：

```py
paths = [
  '/',
  '/gp/dmusic/', '/gp/dmusic/promotions/PrimeMusic/',
 '/gp/registry/wishlist/'  ]   for path in paths:
  print("{0}: {1}".format(robots.allowed(path, '*'), url + path))
```

此代码的结果如下：

```py
True: http://www.amazon.com/
False: http://www.amazon.com/gp/dmusic/
True: http://www.amazon.com/gp/dmusic/promotions/PrimeMusic/
False: http://www.amazon.com/gp/registry/wishlist/
```

对`robots.allowed`的调用给出了 URL 和用户代理。它根据 URL 是否允许爬取返回`True`或`False`。在这种情况下，指定的 URL 的结果为 True、False、True 和 False。让我们看看如何。

/ URL 在`robots.txt`中没有条目，因此默认情况下是允许的。但是，在*用户代理组下的文件中有以下两行：

```py
Disallow: /gp/dmusic/
Allow: /gp/dmusic/promotions/PrimeMusic
```

不允许/gp/dmusic，因此返回 False。/gp/dmusic/promotions/PrimeMusic 是明确允许的。如果未指定 Allowed:条目，则 Disallow:/gp/dmusic/行也将禁止从/gp/dmusic/进一步的任何路径。这基本上表示以/gp/dmusic/开头的任何 URL 都是不允许的，除了允许爬取/gp/dmusic/promotions/PrimeMusic。

在使用`robotparser`库时存在差异。`robotparser`报告`/gp/dmusic/promotions/PrimeMusic`是不允许的。该库未正确处理此类情况，因为它在第一次匹配时停止扫描`robots.txt`，并且不会进一步查找文件以寻找此类覆盖。

# 还有更多...

首先，有关`robots.txt`的详细信息，请参阅[`developers.google.com/search/reference/robots_txt`](https://developers.google.com/search/reference/robots_txt)。

请注意，并非所有站点都有`robots.txt`，其缺失并不意味着您有权自由爬取所有内容。

此外，`robots.txt`文件可能包含有关在网站上查找站点地图的信息。我们将在下一个示例中检查这些站点地图。

Scrapy 还可以读取`robots.txt`并为您找到站点地图。

# 使用站点地图进行爬行

站点地图是一种允许网站管理员通知搜索引擎有关可用于爬取的网站上的 URL 的协议。网站管理员希望使用此功能，因为他们实际上希望他们的信息被搜索引擎爬取。网站管理员希望使该内容可通过搜索引擎找到，至少通过搜索引擎。但您也可以利用这些信息。

站点地图列出了站点上的 URL，并允许网站管理员指定有关每个 URL 的其他信息：

+   上次更新时间

+   内容更改的频率

+   URL 在与其他 URL 的关系中有多重要

站点地图在以下情况下很有用：

+   网站的某些区域无法通过可浏览的界面访问；也就是说，您无法访问这些页面

+   Ajax、Silverlight 或 Flash 内容通常不会被搜索引擎处理

+   网站非常庞大，网络爬虫有可能忽略一些新的或最近更新的内容

+   当网站具有大量孤立或链接不良的页面时

+   当网站具有较少的外部链接时

站点地图文件具有以下结构：

```py
<?xml version="1.0" encoding="utf-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" 
   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
   xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd">
    <url>
        <loc>http://example.com/</loc>
        <lastmod>2006-11-18</lastmod>
        <changefreq>daily</changefreq>
        <priority>0.8</priority>
    </url>
</urlset>
```

站点中的每个 URL 都将用`<url></url>`标签表示，所有这些标签都包裹在外部的`<urlset></urlset>`标签中。始终会有一个指定 URL 的`<loc></loc>`标签。其他三个标签是可选的。

网站地图文件可能非常庞大，因此它们经常被分成多个文件，然后由单个网站地图索引文件引用。该文件的格式如下：

```py
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
   <sitemap>
      <loc>http://www.example.com/sitemap1.xml.gz</loc>
      <lastmod>2014-10-01T18:23:17+00:00</lastmod>
   </sitemap>
</sitemapindex>
```

在大多数情况下，`sitemap.xml` 文件位于域的根目录下。例如，对于 nasa.gov，它是[`www.nasa.gov/sitemap.xml`](https://www.nasa.gov/sitemap.xml)。但请注意，这不是一个标准，不同的网站可能在不同的位置拥有地图或地图。

特定网站的网站地图也可能位于该网站的 `robots.txt` 文件中。例如，microsoft.com 的 `robots.txt` 文件以以下内容结尾：

```py
Sitemap: https://www.microsoft.com/en-us/explore/msft_sitemap_index.xml
Sitemap: https://www.microsoft.com/learning/sitemap.xml
Sitemap: https://www.microsoft.com/en-us/licensing/sitemap.xml
Sitemap: https://www.microsoft.com/en-us/legal/sitemap.xml
Sitemap: https://www.microsoft.com/filedata/sitemaps/RW5xN8
Sitemap: https://www.microsoft.com/store/collections.xml
Sitemap: https://www.microsoft.com/store/productdetailpages.index.xml
```

因此，要获取 microsoft.com 的网站地图，我们首先需要读取 `robots.txt` 文件并提取该信息。

现在让我们来看看如何解析网站地图。

# 准备工作

你所需要的一切都在 `05/02_sitemap.py` 脚本中，以及与其在同一文件夹中的 `sitemap.py` 文件。`sitemap.py` 文件实现了一个基本的网站地图解析器，我们将在主脚本中使用它。在这个例子中，我们将获取 nasa.gov 的网站地图数据。

# 如何做

首先执行 `05/02_sitemap.py` 文件。确保相关的 `sitemap.py` 文件与其在同一目录或路径下。运行后，几秒钟后，你将会得到类似以下的输出：

```py
Found 35511 urls
{'lastmod': '2017-10-11T18:23Z', 'loc': 'http://www.nasa.gov/centers/marshall/history/this-week-in-nasa-history-apollo-7-launches-oct-11-1968.html', 'tag': 'url'}
{'lastmod': '2017-10-11T18:22Z', 'loc': 'http://www.nasa.gov/feature/researchers-develop-new-tool-to-evaluate-icephobic-materials', 'tag': 'url'}
{'lastmod': '2017-10-11T17:38Z', 'loc': 'http://www.nasa.gov/centers/ames/entry-systems-vehicle-development/roster.html', 'tag': 'url'}
{'lastmod': '2017-10-11T17:38Z', 'loc': 'http://www.nasa.gov/centers/ames/entry-systems-vehicle-development/about.html', 'tag': 'url'}
{'lastmod': '2017-10-11T17:22Z', 'loc': 'http://www.nasa.gov/centers/ames/earthscience/programs/MMS/instruments', 'tag': 'url'}
{'lastmod': '2017-10-11T18:15Z', 'loc': 'http://www.nasa.gov/centers/ames/earthscience/programs/MMS/onepager', 'tag': 'url'}
{'lastmod': '2017-10-11T17:10Z', 'loc': 'http://www.nasa.gov/centers/ames/earthscience/programs/MMS', 'tag': 'url'}
{'lastmod': '2017-10-11T17:53Z', 'loc': 'http://www.nasa.gov/feature/goddard/2017/nasa-s-james-webb-space-telescope-and-the-big-bang-a-short-qa-with-nobel-laureate-dr-john', 'tag': 'url'}
{'lastmod': '2017-10-11T17:38Z', 'loc': 'http://www.nasa.gov/centers/ames/entry-systems-vehicle-development/index.html', 'tag': 'url'}
{'lastmod': '2017-10-11T15:21Z', 'loc': 'http://www.nasa.gov/feature/mark-s-geyer-acting-deputy-associate-administrator-for-technical-human-explorations-and-operations', 'tag': 'url'}
```

程序在所有 nasa.gov 的网站地图中找到了 35,511 个 URL！代码只打印了前 10 个，因为输出量会相当大。使用这些信息来初始化对所有这些 URL 的爬取肯定需要相当长的时间！

但这也是网站地图的美妙之处。许多，如果不是所有的结果都有一个 `lastmod` 标签，告诉你与该关联 URL 末端的内容上次修改的时间。如果你正在实现一个有礼貌的爬虫来爬取 nasa.gov，你会想把这些 URL 及其时间戳保存在数据库中，然后在爬取该 URL 之前检查内容是否实际上已经改变，如果没有改变就不要爬取。

现在让我们看看这实际是如何工作的。

# 工作原理

该方法的工作如下：

1.  脚本开始调用 `get_sitemap()`：

```py
map = sitemap.get_sitemap("https://www.nasa.gov/sitemap.xml")
```

1.  给定一个指向 sitemap.xml 文件（或任何其他文件 - 非压缩）的 URL。该实现简单地获取 URL 处的内容并返回它：

```py
def get_sitemap(url):
  get_url = requests.get(url)    if get_url.status_code == 200:
  return get_url.text
    else:
  print ('Unable to fetch sitemap: %s.' % url) 
```

1.  大部分工作是通过将该内容传递给 `parse_sitemap()` 来完成的。在 nasa.gov 的情况下，这个网站地图包含以下内容，即网站地图索引文件：

```py
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="//www.nasa.gov/sitemap.xsl"?>
<sitemapindex >
<sitemap><loc>http://www.nasa.gov/sitemap-1.xml</loc><lastmod>2017-10-11T19:30Z</lastmod></sitemap>
<sitemap><loc>http://www.nasa.gov/sitemap-2.xml</loc><lastmod>2017-10-11T19:30Z</lastmod></sitemap>
<sitemap><loc>http://www.nasa.gov/sitemap-3.xml</loc><lastmod>2017-10-11T19:30Z</lastmod></sitemap>
<sitemap><loc>http://www.nasa.gov/sitemap-4.xml</loc><lastmod>2017-10-11T19:30Z</lastmod></sitemap>
</sitemapindex>
```

1.  `process_sitemap()` 从调用 `process_sitemap()` 开始：

```py
def parse_sitemap(s):
  sitemap = process_sitemap(s)
```

1.  这个函数开始调用 `process_sitemap()`，它返回一个包含 `loc`、`lastmod`、`changeFreq` 和 priority 键值对的 Python 字典对象列表：

```py
def process_sitemap(s):
  soup = BeautifulSoup(s, "lxml")
  result = []    for loc in soup.findAll('loc'):
  item = {}
  item['loc'] = loc.text
        item['tag'] = loc.parent.name
        if loc.parent.lastmod is not None:
  item['lastmod'] = loc.parent.lastmod.text
        if loc.parent.changeFreq is not None:
  item['changeFreq'] = loc.parent.changeFreq.text
        if loc.parent.priority is not None:
  item['priority'] = loc.parent.priority.text
        result.append(item)    return result
```

1.  这是通过使用 `BeautifulSoup` 和 `lxml` 解析网站地图来执行的。`loc` 属性始终被设置，如果有相关的 XML 标签，则会设置 `lastmod`、`changeFreq` 和 priority。.tag 属性本身只是指出这个内容是从 `<sitemap>` 标签还是 `<url>` 标签中检索出来的（`<loc>` 标签可以在任何一个标签上）。

`parse_sitemap()` 然后继续逐一处理这些结果：

```py
while sitemap:
  candidate = sitemap.pop()    if is_sub_sitemap(candidate):
  sub_sitemap = get_sitemap(candidate['loc'])
  for i in process_sitemap(sub_sitemap):
  sitemap.append(i)
  else:
  result.append(candidate)
```

1.  检查每个项目。如果它来自网站地图索引文件（URL 以 .xml 结尾且 .tag 是网站地图），那么我们需要读取该 .xml 文件并解析其内容，然后将结果放入我们要处理的项目列表中。在这个例子中，识别出了四个网站地图文件，每个文件都被读取、处理、解析，并且它们的 URL 被添加到结果中。

为了演示一些内容，以下是 sitemap-1.xml 的前几行：

```py
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="//www.nasa.gov/sitemap.xsl"?>
<urlset >
<url><loc>http://www.nasa.gov/</loc><changefreq>daily</changefreq><priority>1.0</priority></url>
<url><loc>http://www.nasa.gov/connect/apps.html</loc><lastmod>2017-08-14T22:15Z</lastmod><changefreq>yearly</changefreq></url>
<url><loc>http://www.nasa.gov/socialmedia</loc><lastmod>2017-09-29T21:47Z</lastmod><changefreq>monthly</changefreq></url>
<url><loc>http://www.nasa.gov/multimedia/imagegallery/iotd.html</loc><lastmod>2017-08-21T22:00Z</lastmod><changefreq>yearly</changefreq></url>
<url><loc>http://www.nasa.gov/archive/archive/about/career/index.html</loc><lastmod>2017-08-04T02:31Z</lastmod><changefreq>yearly</changefreq></url>
```

总的来说，这一个网站地图有 11,006 行，所以大约有 11,000 个 URL！而且总共，正如报道的那样，所有三个网站地图中共有 35,511 个 URL。

# 还有更多...

网站地图文件也可能是经过压缩的，并以 .gz 扩展名结尾。这是因为它可能包含许多 URL，压缩将节省大量空间。虽然我们使用的代码不处理 gzip 网站地图文件，但可以使用 gzip 库中的函数轻松添加这个功能。

Scrapy 还提供了使用网站地图开始爬取的功能。其中之一是 Spider 类的一个特化，SitemapSpider。这个类有智能来解析网站地图，然后开始跟踪 URL。为了演示，脚本`05/03_sitemap_scrapy.py`将从 nasa.gov 的顶级网站地图索引开始爬取：

```py
import scrapy
from scrapy.crawler import CrawlerProcess

class Spider(scrapy.spiders.SitemapSpider):
  name = 'spider'
  sitemap_urls = ['https://www.nasa.gov/sitemap.xml']    def parse(self, response):
  print("Parsing: ", response)   if __name__ == "__main__":
  process = CrawlerProcess({
  'DOWNLOAD_DELAY': 0,
  'LOG_LEVEL': 'DEBUG'
  })
  process.crawl(Spider)
  process.start()
```

运行时，会有大量输出，因为爬虫将开始爬取所有 30000 多个 URL。在输出的早期，您将看到以下输出：

```py
2017-10-11 20:34:27 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.nasa.gov/sitemap.xml> (referer: None)
2017-10-11 20:34:27 [scrapy.downloadermiddlewares.redirect] DEBUG: Redirecting (301) to <GET https://www.nasa.gov/sitemap-4.xml> from <GET http://www.nasa.gov/sitemap-4.xml>
2017-10-11 20:34:27 [scrapy.downloadermiddlewares.redirect] DEBUG: Redirecting (301) to <GET https://www.nasa.gov/sitemap-2.xml> from <GET http://www.nasa.gov/sitemap-2.xml>
2017-10-11 20:34:27 [scrapy.downloadermiddlewares.redirect] DEBUG: Redirecting (301) to <GET https://www.nasa.gov/sitemap-3.xml> from <GET http://www.nasa.gov/sitemap-3.xml>
2017-10-11 20:34:27 [scrapy.downloadermiddlewares.redirect] DEBUG: Redirecting (301) to <GET https://www.nasa.gov/sitemap-1.xml> from <GET http://www.nasa.gov/sitemap-1.xml>
2017-10-11 20:34:27 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.nasa.gov/sitemap-4.xml> (referer: None)
```

Scrapy 已经找到了所有的网站地图并读取了它们的内容。不久之后，您将开始看到许多重定向和通知，指出正在解析某些页面：

```py
2017-10-11 20:34:30 [scrapy.downloadermiddlewares.redirect] DEBUG: Redirecting (302) to <GET https://www.nasa.gov/image-feature/jpl/pia21629/neptune-from-saturn/> from <GET https://www.nasa.gov/image-feature/jpl/pia21629/neptune-from-saturn>
2017-10-11 20:34:30 [scrapy.downloadermiddlewares.redirect] DEBUG: Redirecting (302) to <GET https://www.nasa.gov/centers/ames/earthscience/members/nasaearthexchange/Ramakrishna_Nemani/> from <GET https://www.nasa.gov/centers/ames/earthscience/members/nasaearthexchang
```

```py
e/Ramakrishna_Nemani>
Parsing: <200 https://www.nasa.gov/exploration/systems/sls/multimedia/sls-hardware-being-moved-on-kamag-transporter.html>
Parsing: <200 https://www.nasa.gov/exploration/systems/sls/M17-057.html>
```

# 带延迟的爬取

快速抓取被认为是一种不良实践。持续不断地访问网站页面可能会消耗 CPU 和带宽，而且强大的网站会识别到您这样做并阻止您的 IP。如果您运气不好，可能会因违反服务条款而收到一封恶意的信！

在爬虫中延迟请求的技术取决于您的爬虫是如何实现的。如果您使用 Scrapy，那么您可以设置一个参数，告诉爬虫在请求之间等待多长时间。在一个简单的爬虫中，只需按顺序处理 URL 的列表，您可以插入一个 thread.sleep 语句。

如果您实施了一个分布式爬虫集群，以分散页面请求的负载，比如使用具有竞争消费者的消息队列，情况可能会变得更加复杂。这可能有许多不同的解决方案，这超出了本文档提供的范围。

# 准备工作

我们将使用带延迟的 Scrapy。示例在`o5/04_scrape_with_delay.py`中。

# 如何做

Scrapy 默认在页面请求之间强加了 0 秒的延迟。也就是说，默认情况下不会在请求之间等待。

1.  这可以使用`DOWNLOAD_DELAY`设置来控制。为了演示，让我们从命令行运行脚本：

```py
05 $ scrapy runspider 04_scrape_with_delay.py -s LOG_LEVEL=WARNING
Parsing: <200 https://blog.scrapinghub.com>
Parsing: <200 https://blog.scrapinghub.com/page/2/>
Parsing: <200 https://blog.scrapinghub.com/page/3/>
Parsing: <200 https://blog.scrapinghub.com/page/4/>
Parsing: &lt;200 https://blog.scrapinghub.com/page/5/>
Parsing: <200 https://blog.scrapinghub.com/page/6/>
Parsing: <200 https://blog.scrapinghub.com/page/7/>
Parsing: <200 https://blog.scrapinghub.com/page/8/>
Parsing: <200 https://blog.scrapinghub.com/page/9/>
Parsing: <200 https://blog.scrapinghub.com/page/10/>
Parsing: <200 https://blog.scrapinghub.com/page/11/>
Total run time: 0:00:07.006148
Michaels-iMac-2:05 michaelheydt$ 
```

这将爬取 blog.scrapinghub.com 上的所有页面，并报告执行爬取的总时间。`LOG_LEVEL=WARNING`会删除大部分日志输出，只会输出打印语句的输出。这使用了默认的页面等待时间为 0，爬取大约需要七秒钟。

1.  页面之间的等待时间可以使用`DOWNLOAD_DELAY`设置。以下在页面请求之间延迟五秒：

```py
05 $ scrapy runspider 04_scrape_with_delay.py -s DOWNLOAD_DELAY=5 -s LOG_LEVEL=WARNING
Parsing: <200 https://blog.scrapinghub.com>
Parsing: <200 https://blog.scrapinghub.com/page/2/>
Parsing: <200 https://blog.scrapinghub.com/page/3/>
Parsing: <200 https://blog.scrapinghub.com/page/4/>
Parsing: <200 https://blog.scrapinghub.com/page/5/>
Parsing: <200 https://blog.scrapinghub.com/page/6/>
Parsing: <200 https://blog.scrapinghub.com/page/7/>
Parsing: <200 https://blog.scrapinghub.com/page/8/>
Parsing: <200 https://blog.scrapinghub.com/page/9/>
Parsing: <200 https://blog.scrapinghub.com/page/10/>
Parsing: <200 https://blog.scrapinghub.com/page/11/>
Total run time: 0:01:01.099267
```

默认情况下，这实际上并不会等待 5 秒。它将等待`DOWNLOAD_DELAY`秒，但是在`DOWNLOAD_DELAY`的 0.5 到 1.5 倍之间有一个随机因素。为什么这样做？这会让您的爬虫看起来“不那么机械化”。您可以通过使用`RANDOMIZED_DOWNLOAD_DELAY=False`设置来关闭这个功能。

# 它是如何工作的

这个爬虫是作为一个 Scrapy 爬虫实现的。类定义从声明爬虫名称和起始 URL 开始：

```py
class Spider(scrapy.Spider):
  name = 'spider'
  start_urls = ['https://blog.scrapinghub.com']
```

解析方法查找 CSS 'div.prev-post > a'，并跟踪这些链接。

爬虫还定义了一个 close 方法，当爬取完成时，Scrapy 会调用这个方法：

```py
def close(spider, reason):
  start_time = spider.crawler.stats.get_value('start_time')
  finish_time = spider.crawler.stats.get_value('finish_time')
  print("Total run time: ", finish_time-start_time)
```

这访问了爬虫的统计对象，检索了爬虫的开始和结束时间，并向用户报告了差异。

# 还有更多...

脚本还定义了在直接使用 Python 执行脚本时的代码：

```py
if __name__ == "__main__":
  process = CrawlerProcess({
  'DOWNLOAD_DELAY': 5,
  'RANDOMIZED_DOWNLOAD_DELAY': False,
  'LOG_LEVEL': 'DEBUG'
  })
  process.crawl(Spider)
  process.start()
```

这是通过创建一个 CrawlerProcess 对象开始的。这个对象可以传递一个表示设置和值的字典，以配置爬取。这默认为五秒的延迟，没有随机化，并且输出级别为 DEBUG。

# 使用可识别的用户代理

如果您违反了服务条款并被网站所有者标记了怎么办？您如何帮助网站所有者联系您，以便他们可以友好地要求您停止对他们认为合理的抓取级别？

为了方便这一点，您可以在请求的 User-Agent 标头中添加有关自己的信息。我们已经在`robots.txt`文件中看到了这样的例子，比如来自 amazon.com。在他们的`robots.txt`中明确声明了一个用于 Google 的用户代理：GoogleBot。

在爬取过程中，您可以在 HTTP 请求的 User-Agent 标头中嵌入自己的信息。为了礼貌起见，您可以输入诸如'MyCompany-MyCrawler（mybot@mycompany.com）'之类的内容。如果远程服务器标记您违规，肯定会捕获这些信息，如果提供了这样的信息，他们可以方便地与您联系，而不仅仅是关闭您的访问。

# 如何做到

根据您使用的工具，设置用户代理会有所不同。最终，它只是确保 User-Agent 标头设置为您指定的字符串。在使用浏览器时，这通常由浏览器设置为标识浏览器和操作系统。但您可以在此标头中放入任何您想要的内容。在使用请求时，这非常简单：

```py
url = 'https://api.github.com/some/endpoint'
headers = {'user-agent': 'MyCompany-MyCrawler (mybot@mycompany.com)'}
r = requests.get(url, headers=headers) 
```

在使用 Scrapy 时，只需配置一个设置即可：

```py
process = CrawlerProcess({
 'USER_AGENT': 'MyCompany-MyCrawler (mybot@mycompany.com)'  }) process.crawl(Spider) process.start()
```

# 它是如何工作的

传出的 HTTP 请求有许多不同的标头。这些确保 User-Agent 标头对目标网站的所有请求都设置为此值。

# 还有更多...

虽然可能将任何内容设置为 User-Agent 标头，但有些 Web 服务器会检查 User-Agent 标头并根据内容做出响应。一个常见的例子是使用标头来识别移动设备以提供移动展示。

但有些网站只允许特定 User-Agent 值访问内容。设置自己的值可能导致 Web 服务器不响应或返回其他错误，比如未经授权。因此，在使用此技术时，请确保检查它是否有效。

# 设置每个域的并发请求数量

一次只爬取一个网址通常效率低下。因此，通常会同时向目标站点发出多个页面请求。通常情况下，远程 Web 服务器可以相当有效地处理多个同时的请求，而在您的端口，您只是在等待每个请求返回数据，因此并发通常对您的爬虫工作效果很好。

但这也是聪明的网站可以识别并标记为可疑活动的模式。而且，您的爬虫端和网站端都有实际限制。发出的并发请求越多，双方都需要更多的内存、CPU、网络连接和网络带宽。这都涉及成本，并且这些值也有实际限制。

因此，通常最好的做法是设置您将同时向任何 Web 服务器发出的请求的数量限制。

# 它是如何工作的

有许多技术可以用来控制并发级别，这个过程通常会相当复杂，需要控制多个请求和执行线程。我们不会在这里讨论如何在线程级别进行操作，只提到了内置在 Scrapy 中的构造。

Scrapy 在其请求中天生是并发的。默认情况下，Scrapy 将最多同时向任何给定域发送八个请求。您可以使用`CONCURRENT_REQUESTS_PER_DOMAIN`设置来更改这一点。以下将该值设置为 1 个并发请求：

```py
process = CrawlerProcess({
 'CONCURRENT_REQUESTS_PER_DOMAIN': 1  }) process.crawl(Spider) process.start()
```

# 使用自动节流

与控制最大并发级别紧密相关的是节流的概念。不同的网站在不同时间对请求的处理能力不同。在响应时间较慢的时期，减少请求的数量是有意义的。这可能是一个繁琐的过程，需要手动监控和调整。

幸运的是，对于我们来说，scrapy 还提供了通过名为`AutoThrottle`的扩展来实现这一点的能力。

# 如何做到

可以使用`AUTOTHROTTLE_TARGET_CONCURRENCY`设置轻松配置 AutoThrottle。

```py
process = CrawlerProcess({
 'AUTOTHROTTLE_TARGET_CONCURRENCY': 3  }) process.crawl(Spider) process.start()
```

# 它是如何工作的

scrapy 跟踪每个请求的延迟。利用这些信息，它可以调整请求之间的延迟，以便在特定域上同时活动的请求不超过`AUTOTHROTTLE_TARGET_CONCURRENCY`，并且请求在任何给定的时间跨度内均匀分布。

# 还有更多...

有很多控制节流的选项。您可以在以下网址上获得它们的概述：[`doc.scrapy.org/en/latest/topics/autothrottle.html?&_ga=2.54316072.1404351387.1507758575-507079265.1505263737#settings.`](https://doc.scrapy.org/en/latest/topics/autothrottle.html?&_ga=2.54316072.1404351387.1507758575-507079265.1505263737#settings)

# 使用 HTTP 缓存进行开发

网络爬虫的开发是一个探索过程，将通过各种细化来迭代检索所需的信息。在开发过程中，您经常会反复访问远程服务器和这些服务器上的相同 URL。这是不礼貌的。幸运的是，scrapy 也通过提供专门设计用于帮助解决这种情况的缓存中间件来拯救您。

# 如何做到这一点

Scrapy 将使用名为 HttpCacheMiddleware 的中间件模块缓存请求。启用它就像将`HTTPCACHE_ENABLED`设置为`True`一样简单：

```py
process = CrawlerProcess({
 'AUTOTHROTTLE_TARGET_CONCURRENCY': 3  }) process.crawl(Spider) process.start()
```

# 它是如何工作的

HTTP 缓存的实现既简单又复杂。Scrapy 提供的`HttpCacheMiddleware`根据您的需求有大量的配置选项。最终，它归结为将每个 URL 及其内容存储在存储器中，并附带缓存过期的持续时间。如果在过期间隔内对 URL 进行第二次请求，则将检索本地副本，而不是进行远程请求。如果时间已经过期，则从 Web 服务器获取内容，存储在缓存中，并设置新的过期时间。

# 还有更多...

有许多配置 scrapy 缓存的选项，包括存储内容的方式（文件系统、DBM 或 LevelDB）、缓存策略以及如何处理来自服务器的 Http 缓存控制指令。要探索这些选项，请查看以下网址：[`doc.scrapy.org/en/latest/topics/downloader-middleware.html?_ga=2.50242598.1404351387.1507758575-507079265.1505263737#dummy-policy-default.`](https://doc.scrapy.org/en/latest/topics/downloader-middleware.html?_ga=2.50242598.1404351387.1507758575-507079265.1505263737#dummy-policy-default.)
