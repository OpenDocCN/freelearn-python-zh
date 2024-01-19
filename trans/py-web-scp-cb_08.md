# 搜索、挖掘和可视化数据

在本章中，我们将涵盖：

+   IP 地址地理编码

+   收集维基百科编辑的 IP 地址

+   在维基百科上可视化贡献者位置频率

+   从 StackOverflow 工作列表创建词云

+   在维基百科上爬取链接

+   在维基百科上可视化页面关系

+   计算维基百科页面之间的分离度

# 介绍

在本章中，我们将研究如何搜索 Web 内容，推导分析结果，并可视化这些结果。我们将学习如何定位内容的发布者并可视化其位置的分布。然后，我们将研究如何爬取、建模和可视化维基百科页面之间的关系。

# IP 地址地理编码

地理编码是将地址转换为地理坐标的过程。这些地址可以是实际的街道地址，可以使用各种工具进行地理编码，例如 Google 地图地理编码 API（[`developers.google.com/maps/documentation/geocoding/intro`](https://developers.google.com/maps/documentation/geocoding/intro)）。 IP 地址可以通过各种应用程序进行地理编码，以确定计算机及其用户的位置。一个非常常见和有价值的用途是分析 Web 服务器日志，以确定您网站的用户来源。

这是可能的，因为 IP 地址不仅代表计算机的地址，可以与该计算机进行通信，而且通常还可以通过在 IP 地址/位置数据库中查找来转换为大致的物理位置。有许多这些数据库可用，所有这些数据库都由各种注册机构（如 ICANN）维护。还有其他工具可以报告公共 IP 地址的地理位置。

有许多免费的 IP 地理位置服务。我们将研究一个非常容易使用的服务，即 freegeoip.net。

# 准备工作

Freegeoip.net 是一个免费的地理编码服务。如果您在浏览器中转到[`www.freegeoip.net`](http://www.freegeoip.net)，您将看到一个类似以下的页面：

![](img/9f56e0c3-c452-4ab1-b951-aa839cfdc831.png)freegeoip.net 主页

默认页面报告您的公共 IP 地址，并根据其数据库给出 IP 地址的地理位置。这并不准确到我家的实际地址，实际上相差几英里，但在世界上的一般位置是相当准确的。我们可以使用这种分辨率甚至更低的数据做重要的事情。通常，只知道 Web 请求的国家来源对于许多目的已经足够了。

Freegeoip 允许您每小时进行 15000 次调用。每次页面加载都算一次调用，正如我们将看到的，每次 API 调用也算一次。

# 如何做到这一点

我们可以爬取这个页面来获取这些信息，但幸运的是，freegeoip.net 为我们提供了一个方便的 REST API 来使用。在页面下方滚动，我们可以看到 API 文档：

![](img/89c10af1-bbcf-4f28-ba1f-5d13fb0f706f.png)freegeoio.net API 文档

我们可以简单地使用 requests 库使用正确格式的 URL 进行 GET 请求。例如，只需在浏览器中输入以下 URL，即可返回给定 IP 地址的地理编码数据的 JSON 表示：

![](img/f87e77ef-910e-4011-b99f-c7942469aa52.png)IP 地址的示例 JSON

一个 Python 脚本，用于演示这一点，可以在`08/01_geocode_address.py`中找到。这很简单，包括以下内容：

```py
import json
import requests

raw_json = requests.get("http://www.freegeoip.net/json/63.153.113.92").text
parsed = json.loads(raw_json) print(json.dumps(parsed, indent=4, sort_keys=True)) 
```

这有以下输出：

```py
{
    "city": "Deer Lodge",
    "country_code": "US",
    "country_name": "United States",
    "ip": "63.153.113.92",
    "latitude": 46.3797,
    "longitude": -112.7202,
    "metro_code": 754,
    "region_code": "MT",
    "region_name": "Montana",
    "time_zone": "America/Denver",
    "zip_code": "59722"
}
```

请注意，对于这个 IP 地址，您的输出可能会有所不同，并且不同的 IP 地址肯定会有所不同。

# 如何收集维基百科编辑的 IP 地址

处理地理编码 IP 地址的聚合结果可以提供有价值的见解。这在服务器日志中非常常见，也可以在许多其他情况下使用。许多网站包括内容贡献者的 IP 地址。维基百科提供了他们所有页面的更改历史。由维基百科未注册用户创建的编辑在历史中公布其 IP 地址。我们将研究如何创建一个爬虫，以浏览给定维基百科主题的历史，并收集未注册编辑的 IP 地址。

# 准备工作

我们将研究对维基百科的 Web 抓取页面所做的编辑。此页面位于：[`en.wikipedia.org/wiki/Web_scraping`](https://en.wikipedia.org/wiki/Web_scraping)。以下是此页面的一小部分：

![](img/533c554f-85a6-4f21-bea7-c382ae64db39.png)查看历史选项卡

注意右上角的查看历史。单击该链接可访问编辑历史：

![](img/11286403-7354-45f0-9786-49148c1f52dc.png)检查 IP 地址

我把这个滚动了一点，以突出一个匿名编辑。请注意，我们可以使用源中的`mw-userling mw-anonuserlink`类来识别这些匿名编辑条目。

还要注意，您可以指定要列出的每页编辑的数量，可以通过向 URL 添加参数来指定。以下 URL 将给我们最近的 500 次编辑：

[`en.wikipedia.org/w/index.php?title=Web_scraping&offset=&limit=500&action=history`](https://en.wikipedia.org/w/index.php?title=Web_scraping&offset=&limit=500&action=history)

因此，我们不是爬行多个不同的页面，每次走 50 个，而是只做一个包含 500 个页面。

# 操作方法

我们按以下步骤进行：

1.  执行抓取的代码在脚本文件`08/02_geocode_wikipedia_edits.py`中。运行脚本会产生以下输出（截断到前几个地理 IP）：

```py
Reading page: https://en.wikipedia.org/w/index.php?title=Web_scraping&offset=&limit=500&action=history
Got 106 ip addresses
{'ip': '2601:647:4a04:86d0:1cdf:8f8a:5ca5:76a0', 'country_code': 'US', 'country_name': 'United States', 'region_code': 'CA', 'region_name': 'California', 'city': 'Sunnyvale', 'zip_code': '94085', 'time_zone': 'America/Los_Angeles', 'latitude': 37.3887, 'longitude': -122.0188, 'metro_code': 807}
{'ip': '194.171.56.13', 'country_code': 'NL', 'country_name': 'Netherlands', 'region_code': '', 'region_name': '', 'city': '', 'zip_code': '', 'time_zone': 'Europe/Amsterdam', 'latitude': 52.3824, 'longitude': 4.8995, 'metro_code': 0}
{'ip': '109.70.55.226', 'country_code': 'DK', 'country_name': 'Denmark', 'region_code': '85', 'region_name': 'Zealand', 'city': 'Roskilde', 'zip_code': '4000', 'time_zone': 'Europe/Copenhagen', 'latitude': 55.6415, 'longitude': 12.0803, 'metro_code': 0}
{'ip': '101.177.247.131', 'country_code': 'AU', 'country_name': 'Australia', 'region_code': 'TAS', 'region_name': 'Tasmania', 'city': 'Lenah Valley', 'zip_code': '7008', 'time_zone': 'Australia/Hobart', 'latitude': -42.8715, 'longitude': 147.2751, 'metro_code': 0}

```

脚本还将地理 IP 写入`geo_ips.json`文件。下一个示例将使用该文件，而不是再次进行所有页面请求。

# 工作原理

解释如下。脚本首先执行以下代码：

```py
if __name__ == "__main__":
  geo_ips = collect_geo_ips('Web_scraping', 500)
  for geo_ip in geo_ips:
  print(geo_ip)
  with open('geo_ips.json', 'w') as outfile:
  json.dump(geo_ips, outfile)
```

调用`collect_geo_ips`，该函数将请求指定主题的页面和最多 500 次编辑。然后将这些地理 IP 打印到控制台，并写入`geo_ips.json`文件。

`collect_geo_ips`的代码如下：

```py
def collect_geo_ips(article_title, limit):
  ip_addresses = get_history_ips(article_title, limit)
  print("Got %s ip addresses" % len(ip_addresses))
  geo_ips = get_geo_ips(ip_addresses)
  return geo_ips
```

此函数首先调用`get_history_ips`，报告找到的数量，然后对每个 IP 地址重复请求`get_geo_ips`。

`get_history_ips`的代码如下：

```py
def get_history_ips(article_title, limit):
  history_page_url = "https://en.wikipedia.org/w/index.php?title=%s&offset=&limit=%s&action=history" % (article_title, limit)
  print("Reading page: " + history_page_url)
  html = requests.get(history_page_url).text
    soup = BeautifulSoup(html, "lxml")    anon_ip_anchors = soup.findAll("a", {"class": "mw-anonuserlink"})
  addresses = set()
  for ip in anon_ip_anchors:
  addresses.add(ip.get_text())
  return addresses
```

这个函数构建了历史页面的 URL，检索页面，然后提取所有具有`mw-anonuserlink`类的不同 IP 地址。

然后，`get_geo_ips`获取这组 IP 地址，并对每个 IP 地址调用`freegeoip.net`以获取数据。

```py
def get_geo_ips(ip_addresses):
  geo_ips = []
  for ip in ip_addresses:
  raw_json = requests.get("http://www.freegeoip.net/json/%s" % ip).text
        parsed = json.loads(raw_json)
  geo_ips.append(parsed)
  return geo_ips
```

# 还有更多...

虽然这些数据很有用，但在下一个示例中，我们将读取写入`geo_ips.json`的数据（使用 pandas），并使用条形图可视化用户按国家的分布。

# 在维基百科上可视化贡献者位置频率

我们可以使用收集的数据来确定来自世界各地的维基百科文章的编辑频率。这可以通过按国家对捕获的数据进行分组并计算与每个国家相关的编辑数量来完成。然后，我们将对数据进行排序并创建一个条形图来查看结果。

# 操作方法

这是一个使用 pandas 执行的非常简单的任务。示例的代码在`08/03_visualize_wikipedia_edits.py`中。

1.  代码开始导入 pandas 和`matplotlib.pyplot`：

```py
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
```

1.  我们在上一个示例中创建的数据文件已经以可以直接被 pandas 读取的格式。这是使用 JSON 作为数据格式的好处之一；pandas 内置支持从 JSON 读取和写入数据。以下使用`pd.read_json()`函数读取数据并在控制台上显示前五行：

```py
>>> df = pd.read_json("geo_ips.json") >>> df[:5]) city country_code country_name ip latitude \
0 Hanoi VN Vietnam 118.70.248.17 21.0333 
1 Roskilde DK Denmark 109.70.55.226 55.6415 
2 Hyderabad IN India 203.217.144.211 17.3753 
3 Prague CZ Czechia 84.42.187.252 50.0833 
4 US United States 99.124.83.153 37.7510

longitude metro_code region_code region_name time_zone \
0 105.8500 0 HN Thanh Pho Ha Noi Asia/Ho_Chi_Minh 
1 12.0803 0 85 Zealand Europe/Copenhagen 
2 78.4744 0 TG Telangana Asia/Kolkata 
3 14.4667 0 10 Hlavni mesto Praha Europe/Prague 
4 -97.8220 0
zip_code 
0 
1 4000 
2 
3 130 00 
4
```

1.  对于我们的直接目的，我们只需要`country_code`列，我们可以用以下方法提取它（并显示该结果中的前五行）：

```py
>>> countries_only = df.country_code
>>> countries_only[:5]

0 VN
1 DK
2 IN
3 CZ
4 US
Name: country_code, dtype:object
```

1.  现在我们可以使用`.groupby('country_code')`来对这个系列中的行进行分组，然后在结果上，`调用.count()`将返回每个组中的项目数。该代码还通过`调用.sort_values()`将结果从最大到最小值进行排序：

```py
>>> counts = df.groupby('country_code').country_code.count().sort_values(ascending=False) >>> counts[:5]

country_code
US 28
IN 12
BR 7
NL 7
RO 6
Name: country_code, dtype: int64 
```

仅从这些结果中，我们可以看出美国在编辑方面绝对领先，印度是第二受欢迎的。

这些数据可以很容易地可视化为条形图：

```py
counts.plot(kind='bar') plt.show()
```

这导致以下条形图显示所有国家的总体分布：

![](img/3c71176f-4d78-4576-9b9d-63f6df781f6a.png)编辑频率的直方图

# 从 StackOverflow 职位列表创建词云

现在让我们来看看如何创建一个词云。词云是一种展示一组文本中关键词频率的图像。图像中的单词越大，它在文本中的重要性就越明显。

# 准备工作

我们将使用 Word Cloud 库来创建我们的词云。该库的源代码可在[`github.com/amueller/word_cloud`](https://github.com/amueller/word_cloud)上找到。这个库可以通过`pip install wordcloud`安装到你的 Python 环境中。

# 如何做到这一点

创建词云的脚本在`08/04_so_word_cloud.py`文件中。这个示例是从第七章的堆栈溢出示例中继续提供数据的可视化。

1.  首先从 NLTK 中导入词云和频率分布函数：

```py
from wordcloud import WordCloud
from nltk.probability import FreqDist
```

1.  然后，词云是从我们从职位列表中收集的单词的概率分布生成的：

```py
freq_dist = FreqDist(cleaned) wordcloud = WordCloud(width=1200, height=800).generate_from_frequencies(freq_dist) 
```

现在我们只需要显示词云：

```py
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear') plt.axis("off") plt.show()
```

生成的词云如下：

![](img/02a14763-b72f-4386-ae29-ebed2d7d7219.png)职位列表的词云

位置和大小都有一些内置的随机性，所以你得到的结果可能会有所不同。

# 在维基百科上爬取链接

在这个示例中，我们将编写一个小程序来利用爬取维基百科页面上的链接，通过几个深度级别。在这个爬取过程中，我们将收集页面之间以及每个页面引用的页面之间的关系。在此过程中，我们将建立这些页面之间的关系，最终在下一个示例中进行可视化。

# 准备工作

这个示例的代码在`08/05_wikipedia_scrapy.py`中。它引用了代码示例中`modules`/`wikipedia`文件夹中的一个模块的代码，所以确保它在你的 Python 路径中。

# 如何做到这一点

你可以使用示例 Python 脚本。它将使用 Scrapy 爬取单个维基百科页面。它将爬取的页面是 Python 页面，网址为[`en.wikipedia.org/wiki/Python_(programming_language)`](https://en.wikipedia.org/wiki/Python_(programming_language))，并收集该页面上的相关链接。

运行时，你将看到类似以下的输出：

```py
/Users/michaelheydt/anaconda/bin/python3.6 /Users/michaelheydt/Dropbox/Packt/Books/PyWebScrCookbook/code/py/08/05_wikipedia_scrapy.py
parsing: https://en.wikipedia.org/wiki/Python_(programming_language)
parsing: https://en.wikipedia.org/wiki/C_(programming_language)
parsing: https://en.wikipedia.org/wiki/Object-oriented_programming
parsing: https://en.wikipedia.org/wiki/Ruby_(programming_language)
parsing: https://en.wikipedia.org/wiki/Go_(programming_language)
parsing: https://en.wikipedia.org/wiki/Java_(programming_language)
------------------------------------------------------------
0 Python_(programming_language) C_(programming_language)
0 Python_(programming_language) Java_(programming_language)
0 Python_(programming_language) Go_(programming_language)
0 Python_(programming_language) Ruby_(programming_language)
0 Python_(programming_language) Object-oriented_programming
```

输出的第一部分来自 Scrapy 爬虫，并显示传递给解析方法的页面。这些页面以我们的初始页面开头，并通过该页面的前五个最常见的链接。

此输出的第二部分是对被爬取的页面以及在该页面上找到的链接的表示，这些链接被认为是未来处理的。第一个数字是找到关系的爬取级别，然后是父页面和在该页面上找到的链接。对于每个找到的页面/链接，都有一个单独的条目。由于这是一个深度爬取，我们只显示从初始页面找到的页面。

# 它是如何工作的

让我们从主脚本文件`08/05_wikipedia_scrapy.py`中的代码开始。这是通过创建一个`WikipediaSpider`对象并运行爬取开始的：

```py
process = CrawlerProcess({
    'LOG_LEVEL': 'ERROR',
    'DEPTH_LIMIT': 1 })

process.crawl(WikipediaSpider)
spider = next(iter(process.crawlers)).spider
process.start()
```

这告诉 Scrapy 我们希望运行一层深度，我们得到一个爬虫的实例，因为我们想要检查其属性，这些属性是爬取的结果。然后用以下方法打印结果：

```py
print("-"*60)

for pm in spider.linked_pages:
    print(pm.depth, pm.title, pm.child_title)
```

爬虫的每个结果都存储在`linked_pages`属性中。每个对象都由几个属性表示，包括页面的标题（维基百科 URL 的最后部分）和在该页面的 HTML 内容中找到的每个页面的标题。

现在让我们来看一下爬虫的功能。爬虫的代码在`modules/wikipedia/spiders.py`中。爬虫首先定义了一个 Scrapy `Spider`的子类：

```py
class WikipediaSpider(Spider):
    name = "wikipedia"
  start_urls = [ "https://en.wikipedia.org/wiki/Python_(programming_language)" ]
```

我们从维基百科的 Python 页面开始。接下来是定义一些类级变量，以定义爬取的操作方式和要检索的结果：

```py
page_map = {}
linked_pages = []
max_items_per_page = 5 max_crawl_depth = 1
```

这次爬取的每个页面都将由爬虫的解析方法处理。让我们来看一下。它从以下开始：

```py
def parse(self, response):
    print("parsing: " + response.url)

    links = response.xpath("//*/a[starts-with(@href, '/wiki/')]/@href")

    link_counter = {}
```

在每个维基百科页面中，我们寻找以`/wiki`开头的链接。页面中还有其他链接，但这些是这次爬取将考虑的重要链接。

这个爬虫实现了一个算法，其中页面上找到的所有链接都被计算为相似。有相当多的重复链接。其中一些是虚假的。其他代表了多次链接到其他页面的真正重要性。

`max_items_per_page`定义了我们将进一步调查当前页面上有多少链接。每个页面上都会有相当多的链接，这个算法会计算所有相似的链接并将它们放入桶中。然后它会跟踪`max_items_per_page`最受欢迎的链接。

这个过程是通过使用`links_counter`变量来管理的。这是当前页面和页面上找到的所有链接之间的映射字典。对于我们决定跟踪的每个链接，我们计算它在页面上被引用的次数。这个变量是该 URL 和计数引用次数的对象之间的映射：

```py
class LinkReferenceCount:
    def __init__(self, link):
        self.link = link
  self.count = 0
```

然后，代码遍历所有识别的链接：

```py
for l in links:
    link = l.root
    if ":" not in link and "International" not in link and link != self.start_urls[0]:
        if link not in link_counter:
            link_counter[link] = LinkReferenceCount(link)
        link_counter[link].count += 1
```

这个算法检查每个链接，并根据规则（链接中没有“：”，也没有“国际”因为它非常受欢迎所以我们排除它，最后我们不包括起始 URL）只考虑它们进行进一步的爬取。如果链接通过了这一步，那么就会创建一个新的`LinkReferenceCounter`对象（如果之前没有看到这个链接），或者增加它的引用计数。

由于每个页面上可能有重复的链接，我们只想考虑`max_items_per_page`最常见的链接。代码通过以下方式实现了这一点：

```py
references = list(link_counter.values())
s = sorted(references, key=lambda x: x.count, reverse=True)
top = s[:self.max_items_per_page]
```

从`link_counter`字典中，我们提取所有的`LinkReferenceCounter`对象，并按计数排序，然后选择前`max_items_per_page`个项目。

下一步是对这些符合条件的项目进行记录，记录在类的`linked_pages`字段中。这个列表中的每个对象都是`PageToPageMap`类型。这个类有以下定义：

```py
class PageToPageMap:
    def __init__(self, link, child_link, depth): #, parent):
  self.link = link
  self.child_link = child_link
  self.title = self.get_page_title(self.link)
        self.child_title = self.get_page_title(self.child_link)
        self.depth = depth    def get_page_title(self, link):
        parts = link.split("/")
        last = parts[len(parts)-1]
        label = urllib.parse.unquote(last)
        return label
```

从根本上说，这个对象表示一个源页面 URL 到一个链接页面 URL，并跟踪爬取的当前级别。标题属性是维基百科 URL 最后部分的 URL 解码形式，代表了 URL 的更加人性化的版本。

最后，代码将新的页面交给 Scrapy 进行爬取。

```py
for item in top:
    new_request = Request("https://en.wikipedia.org" + item.link,
                          callback=self.parse, meta={ "parent": pm })
    yield new_request
```

# 还有更多...

这个爬虫/算法还跟踪爬取中当前的**深度**级别。如果认为新链接超出了爬取的最大深度。虽然 Scrapy 可以在一定程度上控制这一点，但这段代码仍然需要排除超出最大深度的链接。

这是通过使用`PageToPageMap`对象的深度字段来控制的。对于每个爬取的页面，我们检查响应是否具有元数据，这是表示给定页面的“父”`PageToPageMap`对象的属性。我们可以通过以下代码找到这个：

```py
depth = 0 if "parent" in response.meta:
    parent = response.meta["parent"]
    depth = parent.depth + 1
```

页面解析器中的此代码查看是否有父对象。只有爬取的第一个页面没有父页面。如果有一个实例，这个爬取的深度被认为是更高的。当创建新的`PageToPageMap`对象时，这个值被传递并存储。

代码通过使用请求对象的 meta 属性将此对象传递到爬取的下一级别：

```py
meta={ "parent": pm }
```

通过这种方式，我们可以将数据从 Scrapy 蜘蛛的一个爬取级别传递到下一个级别。

# 在维基百科上可视化页面关系

在这个示例中，我们使用之前收集的数据，并使用 NetworkX Python 库创建一个力导向网络可视化页面关系。

# 准备工作

NetworkX 是用于建模、可视化和分析复杂网络关系的软件。您可以在[`networkx.github.io`](https://networkx.github.io/)找到更多关于它的信息。它可以通过`pip install networkx`在您的 Python 环境中安装。

# 如何做到这一点

此示例的脚本位于`08/06_visualizze_wikipedia_links.py`文件中。运行时，它会生成维基百科上初始 Python 页面上找到的链接的图表：

![](img/54f24aea-344a-42b8-b02b-a68c1a8287c4.png)链接的图表

现在我们可以看到页面之间的关系了！

# 工作原理

爬取从定义一级深度爬取开始：

```py
crawl_depth = 1 process = CrawlerProcess({
    'LOG_LEVEL': 'ERROR',
    'DEPTH_LIMIT': crawl_depth
})
process.crawl(WikipediaSpider)
spider = next(iter(process.crawlers)).spider
spider.max_items_per_page = 5 spider.max_crawl_depth = crawl_depth
process.start()

for pm in spider.linked_pages:
    print(pm.depth, pm.link, pm.child_link)
print("-"*80)
```

这些信息与之前的示例类似，现在我们需要将其转换为 NetworkX 可以用于图的模型。这始于创建一个 NetworkX 图模型：

```py
g = nx.Graph()
```

NetworkX 图由节点和边组成。从收集的数据中，我们必须创建一组唯一的节点（页面）和边（页面引用另一个页面的事实）。可以通过以下方式执行：

```py
nodes = {}
edges = {}

for pm in spider.linked_pages:
    if pm.title not in nodes:
        nodes[pm.title] = pm
        g.add_node(pm.title)

    if pm.child_title not in nodes:
        g.add_node(pm.child_title)

    link_key = pm.title + " ==> " + pm.child_title
    if link_key not in edges:
        edges[link_key] = link_key
        g.add_edge(pm.title, pm.child_title)
```

这通过遍历我们爬取的所有结果，并识别所有唯一节点（不同的页面），以及页面之间的所有链接。对于每个节点和边，我们使用 NetworkX 进行注册。

接下来，我们使用 Matplotlib 创建绘图，并告诉 NetworkX 如何在绘图中创建可视化效果：

```py
plt.figure(figsize=(10,8))

node_positions = nx.spring_layout(g)

nx.draw_networkx_nodes(g, node_positions, g.nodes, node_color='green', node_size=50)
nx.draw_networkx_edges(g, node_positions)

labels = { node: node for node in g.nodes() }
nx.draw_networkx_labels(g, node_positions, labels, font_size=9.5)

plt.show()
```

其中重要的部分首先是使用 NetworkX 在节点上形成弹簧布局。这计算出节点的实际位置，但不渲染节点或边。这是接下来的两行的目的，它们给出了 NetworkX 如何渲染节点和边的指令。最后，我们需要在节点上放置标签。

# 还有更多...

这次爬取只进行了一级深度的爬取。可以通过对代码进行以下更改来增加爬取的深度：

```py
crawl_depth = 2 process = CrawlerProcess({
    'LOG_LEVEL': 'ERROR',
    'DEPTH_LIMIT': crawl_depth
})
process.crawl(WikipediaSpider)
spider = next(iter(process.crawlers)).spider
spider.max_items_per_page = 5 spider.max_crawl_depth = crawl_depth
process.start()
```

基本上唯一的变化是增加一级深度。然后得到以下图表（任何弹簧图都会有随机性，因此实际结果会有不同的布局）：

![](img/6ddc2576-434d-411d-9d3f-15e9482e48f6.png)链接的蜘蛛图

这开始变得有趣，因为我们现在开始看到页面之间的相互关系和循环关系。

我敢你进一步增加深度和每页的链接数。

# 计算分离度

现在让我们计算任意两个页面之间的分离度。这回答了从源页面到另一个页面需要浏览多少页面的问题。这可能是一个非平凡的图遍历问题，因为两个页面之间可能有多条路径。幸运的是，对于我们来说，NetworkX 使用完全相同的图模型，具有内置函数来解决这个问题。

# 如何做到这一点

这个示例的脚本在`08/07_degrees_of_separation.py`中。代码与之前的示例相同，进行了 2 层深度的爬取，只是省略了图表，并要求 NetworkX 解决`Python_(programming_language)`和`Dennis_Ritchie`之间的分离度：

```py
Degrees of separation: 1
 Python_(programming_language)
   C_(programming_language)
    Dennis_Ritchie
```

这告诉我们，要从`Python_(programming_language)`到`Dennis_Ritchie`，我们必须通过另一个页面：`C_(programming_language)`。因此，一度分离。如果我们直接到`C_(programming_language)`，那么就是 0 度分离。

# 它是如何工作的

这个问题的解决方案是由一种称为**A***的算法解决的。**A***算法确定图中两个节点之间的最短路径。请注意，这条路径可以是不同长度的多条路径，正确的结果是最短路径。对我们来说好消息是，NetworkX 有一个内置函数来为我们做这个。它可以用一条简单的语句完成：

```py
path = nx.astar_path(g, "Python_(programming_language)", "Dennis_Ritchie")
```

从这里我们报告实际路径：

```py
degrees_of_separation = int((len(path) - 1) / 2)
print("Degrees of separation: {}".format(degrees_of_separation))
for i in range(0, len(path)):
    print(" " * i, path[i])
```

# 还有更多...

有关**A***算法的更多信息，请查看[此页面](https://en.wikipedia.org/wiki/A*_search_algorithm)。
