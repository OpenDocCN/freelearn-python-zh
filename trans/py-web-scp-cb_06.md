# 第六章：爬取挑战和解决方案

在本章中，我们将涵盖：

+   重试失败的页面下载

+   支持页面重定向

+   等待 Selenium 中的内容可用

+   将爬行限制为单个域

+   处理无限滚动页面

+   控制爬行的深度

+   控制爬行的长度

+   处理分页网站

+   处理表单和基于表单的授权

+   处理基本授权

+   通过代理防止被禁止爬取

+   随机化用户代理

+   缓存响应

# 介绍

开发可靠的爬虫从来都不容易，我们需要考虑很多*假设*。如果网站崩溃了怎么办？如果响应返回意外数据怎么办？如果您的 IP 被限制或阻止了怎么办？如果需要身份验证怎么办？虽然我们永远无法预测和涵盖所有*假设*，但我们将讨论一些常见的陷阱、挑战和解决方法。

请注意，其中几个配方需要访问我提供的作为 Docker 容器的网站。它们需要比我们在早期章节中使用的简单静态站点更多的逻辑。因此，您需要使用以下 Docker 命令拉取和运行 Docker 容器：

```py
docker pull mheydt/pywebscrapecookbook
docker run -p 5001:5001 pywebscrapecookbook
```

# 重试失败的页面下载

使用重试中间件，Scrapy 可以轻松处理失败的页面请求。安装后，Scrapy 将在接收以下 HTTP 错误代码时尝试重试：

`[500, 502, 503, 504, 408]`

可以使用以下参数进一步配置该过程：

+   `RETRY_ENABLED`（True/False-默认为 True）

+   `RETRY_TIMES`（在任何错误上重试的次数-默认为 2）

+   `RETRY_HTTP_CODES`（应该重试的 HTTP 错误代码列表-默认为[500, 502, 503, 504, 408]）

# 如何做到

`06/01_scrapy_retry.py`脚本演示了如何配置 Scrapy 进行重试。脚本文件包含了以下 Scrapy 的配置：

```py
process = CrawlerProcess({
  'LOG_LEVEL': 'DEBUG',
  'DOWNLOADER_MIDDLEWARES':
 {  "scrapy.downloadermiddlewares.retry.RetryMiddleware": 500
  },
  'RETRY_ENABLED': True,
  'RETRY_TIMES': 3 }) process.crawl(Spider) process.start()
```

# 它是如何工作的

Scrapy 在运行蜘蛛时会根据指定的重试配置进行重试。在遇到错误时，Scrapy 会在放弃之前最多重试三次。

# 支持页面重定向

Scrapy 中的页面重定向是使用重定向中间件处理的，默认情况下已启用。可以使用以下参数进一步配置该过程：

+   `REDIRECT_ENABLED`：（True/False-默认为 True）

+   `REDIRECT_MAX_TIMES`：（对于任何单个请求要遵循的最大重定向次数-默认为 20）

# 如何做到

`06/02_scrapy_redirects.py`脚本演示了如何配置 Scrapy 来处理重定向。这为任何页面配置了最多两次重定向。运行该脚本会读取 NASA 站点地图并爬取内容。其中包含大量重定向，其中许多是从 HTTP 到 HTTPS 版本的 URL 的重定向。输出会很多，但以下是一些演示输出的行：

```py
Parsing: <200 https://www.nasa.gov/content/earth-expeditions-above/>
['http://www.nasa.gov/content/earth-expeditions-above', 'https://www.nasa.gov/content/earth-expeditions-above']
```

此特定 URL 在重定向后被处理，从 URL 的 HTTP 版本重定向到 HTTPS 版本。该列表定义了涉及重定向的所有 URL。

您还将能够看到输出页面中重定向超过指定级别（2）的位置。以下是一个例子：

```py
2017-10-22 17:55:00 [scrapy.downloadermiddlewares.redirect] DEBUG: Discarding <GET http://www.nasa.gov/topics/journeytomars/news/index.html>: max redirections reached
```

# 它是如何工作的

蜘蛛被定义为以下内容：

```py
class Spider(scrapy.spiders.SitemapSpider):
  name = 'spider'
  sitemap_urls = ['https://www.nasa.gov/sitemap.xml']    def parse(self, response):
  print("Parsing: ", response)
  print (response.request.meta.get('redirect_urls'))
```

这与我们之前基于 NASA 站点地图的爬虫相同，只是增加了一行打印`redirect_urls`。在对`parse`的任何调用中，此元数据将包含到达此页面所发生的所有重定向。

爬行过程使用以下代码进行配置：

```py
process = CrawlerProcess({
  'LOG_LEVEL': 'DEBUG',
  'DOWNLOADER_MIDDLEWARES':
 {  "scrapy.downloadermiddlewares.redirect.RedirectMiddleware": 500
  },
  'REDIRECT_ENABLED': True,
  'REDIRECT_MAX_TIMES': 2 }) 
```

重定向默认已启用，但这将将最大重定向次数设置为 2，而不是默认值 20。

# 等待 Selenium 中的内容可用

动态网页的一个常见问题是，即使整个页面已经加载完成，因此 Selenium 中的`get()`方法已经返回，仍然可能有我们需要稍后访问的内容，因为页面上仍有未完成的 Ajax 请求。这个的一个例子是需要点击一个按钮，但是在加载页面后，直到所有数据都已异步加载到页面后，按钮才被启用。

以以下页面为例：[`the-internet.herokuapp.com/dynamic_loading/2`](http://the-internet.herokuapp.com/dynamic_loading/2)。这个页面加载非常快，然后呈现给我们一个开始按钮：

![](img/08dd65a4-9018-4136-9bb9-0b7f74e17aff.png)屏幕上呈现的开始按钮

按下按钮后，我们会看到一个进度条，持续五秒：

![](img/530d1355-e1cc-4551-ab0f-9374c029b03e.png)等待时的状态栏

当这个完成后，我们会看到 Hello World！

![](img/d37ade1b-f857-4a5f-a7ce-5157238e9e09.png)页面完全渲染后

现在假设我们想要爬取这个页面，以获取只有在按下按钮并等待后才暴露的内容？我们该怎么做？

# 如何做到这一点

我们可以使用 Selenium 来做到这一点。我们将使用 Selenium 的两个功能。第一个是点击页面元素的能力。第二个是等待直到页面上具有特定 ID 的元素可用。

1.  首先，我们获取按钮并点击它。按钮的 HTML 如下：

```py
<div id='start'>
   <button>Start</button>
</div>
```

1.  当按下按钮并加载完成后，以下 HTML 将被添加到文档中：

```py
<div id='finish'>
   <h4>Hello World!"</h4>
</div>
```

1.  我们将使用 Selenium 驱动程序来查找开始按钮，点击它，然后等待直到`div`中的 ID 为`'finish'`可用。然后我们获取该元素并返回封闭的`<h4>`标签中的文本。

您可以通过运行`06/03_press_and_wait.py`来尝试这个。它的输出将是以下内容：

```py
clicked
Hello World!
```

现在让我们看看它是如何工作的。

# 它是如何工作的

让我们分解一下解释：

1.  我们首先从 Selenium 中导入所需的项目：

```py
from selenium import webdriver
from selenium.webdriver.support import ui
```

1.  现在我们加载驱动程序和页面：

```py
driver = webdriver.PhantomJS() driver.get("http://the-internet.herokuapp.com/dynamic_loading/2")
```

1.  页面加载后，我们可以检索按钮：

```py
button = driver.find_element_by_xpath("//*/div[@id='start']/button")
```

1.  然后我们可以点击按钮：

```py
button.click() print("clicked")
```

1.  接下来我们创建一个`WebDriverWait`对象：

```py
wait = ui.WebDriverWait(driver, 10)
```

1.  有了这个对象，我们可以请求 Selenium 的 UI 等待某些事件。这也设置了最长等待 10 秒。现在使用这个，我们可以等到我们满足一个标准；使用以下 XPath 可以识别一个元素：

```py
wait.until(lambda driver: driver.find_element_by_xpath("//*/div[@id='finish']"))
```

1.  当这完成后，我们可以检索 h4 元素并获取其封闭文本：

```py
finish_element=driver.find_element_by_xpath("//*/div[@id='finish']/h4") print(finish_element.text)
```

# 限制爬行到单个域

我们可以通知 Scrapy 将爬行限制在指定域内的页面。这是一个重要的任务，因为链接可以指向网页的任何地方，我们通常希望控制爬行的方向。Scrapy 使这个任务非常容易。只需要设置爬虫类的`allowed_domains`字段即可。 

# 如何做到这一点

这个示例的代码是`06/04_allowed_domains.py`。您可以使用 Python 解释器运行脚本。它将执行并生成大量输出，但如果您留意一下，您会发现它只处理 nasa.gov 上的页面。

# 它是如何工作的

代码与之前的 NASA 网站爬虫相同，只是我们包括`allowed_domains=['nasa.gov']`：

```py
class Spider(scrapy.spiders.SitemapSpider):
  name = 'spider'
  sitemap_urls = ['https://www.nasa.gov/sitemap.xml']
  allowed_domains=['nasa.gov']    def parse(self, response):
  print("Parsing: ", response) 
```

NASA 网站在其根域内保持相当一致，但偶尔会有指向其他网站的链接，比如 boeing.com 上的内容。这段代码将阻止转移到这些外部网站。

# 处理无限滚动页面

许多网站已经用无限滚动机制取代了“上一页/下一页”分页按钮。这些网站使用这种技术在用户到达页面底部时加载更多数据。因此，通过点击“下一页”链接进行爬行的策略就会崩溃。

虽然这似乎是使用浏览器自动化来模拟滚动的情况，但实际上很容易找出网页的 Ajax 请求，并使用这些请求来爬取，而不是实际页面。让我们以`spidyquotes.herokuapp.com/scroll`为例。

# 准备就绪

在浏览器中打开[`spidyquotes.herokuapp.com/scroll`](http://spidyquotes.herokuapp.com/scroll)。当你滚动到页面底部时，页面将加载更多内容：

![](img/5aedcf3b-b3dd-4e67-8328-7093d70c2db4.png)要抓取的引用的屏幕截图

页面打开后，进入开发者工具并选择网络面板。然后，滚动到页面底部。您将在网络面板中看到新内容：

![](img/b8f8c31d-706b-4f11-bab8-901263f7fdfc.png)开发者工具选项的屏幕截图

当我们点击其中一个链接时，我们可以看到以下 JSON：

```py
{
"has_next": true,
"page": 2,
"quotes": [{
"author": {
"goodreads_link": "/author/show/82952.Marilyn_Monroe",
"name": "Marilyn Monroe",
"slug": "Marilyn-Monroe"
},
"tags": ["friends", "heartbreak", "inspirational", "life", "love", "sisters"],
"text": "\u201cThis life is what you make it...."
}, {
"author": {
"goodreads_link": "/author/show/1077326.J_K_Rowling",
"name": "J.K. Rowling",
"slug": "J-K-Rowling"
},
"tags": ["courage", "friends"],
"text": "\u201cIt takes a great deal of bravery to stand up to our enemies, but just as much to stand up to our friends.\u201d"
},
```

这很棒，因为我们只需要不断生成对`/api/quotes?page=x`的请求，增加`x`直到回复文档中存在`has_next`标签。如果没有更多页面，那么这个标签将不在文档中。

# 如何做到这一点

`06/05_scrapy_continuous.py`文件包含一个 Scrapy 代理，它爬取这组页面。使用 Python 解释器运行它，你将看到类似以下的输出（以下是输出的多个摘录）：

```py
<200 http://spidyquotes.herokuapp.com/api/quotes?page=2>
2017-10-29 16:17:37 [scrapy.core.scraper] DEBUG: Scraped from <200 http://spidyquotes.herokuapp.com/api/quotes?page=2>
{'text': "“This life is what you make it. No matter what, you're going to mess up sometimes, it's a universal truth. But the good part is you get to decide how you're going to mess it up. Girls will be your friends - they'll act like it anyway. But just remember, some come, some go. The ones that stay with you through everything - they're your true best friends. Don't let go of them. Also remember, sisters make the best friends in the world. As for lovers, well, they'll come and go too. And baby, I hate to say it, most of them - actually pretty much all of them are going to break your heart, but you can't give up because if you give up, you'll never find your soulmate. You'll never find that half who makes you whole and that goes for everything. Just because you fail once, doesn't mean you're gonna fail at everything. Keep trying, hold on, and always, always, always believe in yourself, because if you don't, then who will, sweetie? So keep your head high, keep your chin up, and most importantly, keep smiling, because life's a beautiful thing and there's so much to smile about.”", 'author': 'Marilyn Monroe', 'tags': ['friends', 'heartbreak', 'inspirational', 'life', 'love', 'sisters']}
2017-10-29 16:17:37 [scrapy.core.scraper] DEBUG: Scraped from <200 http://spidyquotes.herokuapp.com/api/quotes?page=2>
{'text': '“It takes a great deal of bravery to stand up to our enemies, but just as much to stand up to our friends.”', 'author': 'J.K. Rowling', 'tags': ['courage', 'friends']}
2017-10-29 16:17:37 [scrapy.core.scraper] DEBUG: Scraped from <200 http://spidyquotes.herokuapp.com/api/quotes?page=2>
{'text': "“If you can't explain it to a six year old, you don't understand it yourself.”", 'author': 'Albert Einstein', 'tags': ['simplicity', 'understand']}
```

当它到达第 10 页时，它将停止，因为它会看到内容中没有设置下一页标志。

# 它是如何工作的

让我们通过蜘蛛来看看这是如何工作的。蜘蛛从以下开始 URL 的定义开始：

```py
class Spider(scrapy.Spider):
  name = 'spidyquotes'
  quotes_base_url = 'http://spidyquotes.herokuapp.com/api/quotes'
  start_urls = [quotes_base_url]
  download_delay = 1.5
```

然后解析方法打印响应，并将 JSON 解析为数据变量：

```py
  def parse(self, response):
  print(response)
  data = json.loads(response.body)
```

然后它循环遍历 JSON 对象的引用元素中的所有项目。对于每个项目，它将向 Scrapy 引擎产生一个新的 Scrapy 项目：

```py
  for item in data.get('quotes', []):
  yield {
  'text': item.get('text'),
  'author': item.get('author', {}).get('name'),
  'tags': item.get('tags'),
 } 
```

然后它检查数据 JSON 变量是否具有`'has_next'`属性，如果有，它将获取下一页并向 Scrapy 产生一个新的请求来解析下一页：

```py
if data['has_next']:
    next_page = data['page'] + 1
  yield scrapy.Request(self.quotes_base_url + "?page=%s" % next_page)
```

# 还有更多...

也可以使用 Selenium 处理无限滚动页面。以下代码在`06/06_scrape_continuous_twitter.py`中：

```py
from selenium import webdriver
import time

driver = webdriver.PhantomJS()   print("Starting") driver.get("https://twitter.com") scroll_pause_time = 1.5   # Get scroll height last_height = driver.execute_script("return document.body.scrollHeight") while True:
  print(last_height)
  # Scroll down to bottom
  driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")    # Wait to load page
  time.sleep(scroll_pause_time)    # Calculate new scroll height and compare with last scroll height
  new_height = driver.execute_script("return document.body.scrollHeight")
  print(new_height, last_height)    if new_height == last_height:
  break
  last_height = new_height
```

输出将类似于以下内容：

```py
Starting
4882
8139 4882
8139
11630 8139
11630
15055 11630
15055
15055 15055
Process finished with exit code 0
```

这段代码首先从 Twitter 加载页面。调用`.get()`将在页面完全加载时返回。然后检索`scrollHeight`，程序滚动到该高度并等待新内容加载片刻。然后再次检索浏览器的`scrollHeight`，如果与`last_height`不同，它将循环并继续处理。如果与`last_height`相同，则没有加载新内容，然后您可以继续检索已完成页面的 HTML。

# 控制爬取的深度

可以使用 Scrapy 的`DepthMiddleware`中间件来控制爬取的深度。深度中间件限制了 Scrapy 从任何给定链接获取的跟随数量。这个选项对于控制你深入到特定爬取中有多有用。这也用于防止爬取过长，并且在你知道你要爬取的内容位于从爬取开始的页面的一定数量的分离度内时非常有用。

# 如何做到这一点

深度控制中间件默认安装在中间件管道中。深度限制的示例包含在`06/06_limit_depth.py`脚本中。该脚本爬取源代码提供的端口 8080 上的静态站点，并允许您配置深度限制。该站点包括三个级别：0、1 和 2，并且每个级别有三个页面。文件名为`CrawlDepth<level><pagenumber>.html`。每个级别的第 1 页链接到同一级别的其他两页，以及下一级别的第 1 页。到达第 2 级的链接结束。这种结构非常适合检查 Scrapy 中如何处理深度处理。

# 它是如何工作的

深度限制可以通过设置`DEPTH_LIMIT`参数来执行：

```py
process = CrawlerProcess({
    'LOG_LEVEL': 'CRITICAL',
    'DEPTH_LIMIT': 2,
    'DEPT_STATS': True })
```

深度限制为 1 意味着我们只会爬取一层，这意味着它将处理`start_urls`中指定的 URL，然后处理这些页面中找到的任何 URL。使用`DEPTH_LIMIT`我们得到以下输出：

```py
Parsing: <200 http://localhost:8080/CrawlDepth0-1.html>
Requesting crawl of: http://localhost:8080/CrawlDepth0-2.html
Requesting crawl of: http://localhost:8080/Depth1/CrawlDepth1-1.html
Parsing: <200 http://localhost:8080/Depth1/CrawlDepth1-1.html>
Requesting crawl of: http://localhost:8080/Depth1/CrawlDepth1-2.html
Requesting crawl of: http://localhost:8080/Depth1/depth1/CrawlDepth1-2.html
Requesting crawl of: http://localhost:8080/Depth1/depth2/CrawlDepth2-1.html
Parsing: <200 http://localhost:8080/CrawlDepth0-2.html>
Requesting crawl of: http://localhost:8080/CrawlDepth0-3.html
<scrapy.statscollectors.MemoryStatsCollector object at 0x109f754e0>
Crawled: ['http://localhost:8080/CrawlDepth0-1.html', 'http://localhost:8080/Depth1/CrawlDepth1-1.html', 'http://localhost:8080/CrawlDepth0-2.html']
Requested: ['http://localhost:8080/CrawlDepth0-2.html', 'http://localhost:8080/Depth1/CrawlDepth1-1.html', 'http://localhost:8080/Depth1/CrawlDepth1-2.html', 'http://localhost:8080/Depth1/depth1/CrawlDepth1-2.html', 'http://localhost:8080/Depth1/depth2/CrawlDepth2-1.html', 'http://localhost:8080/CrawlDepth0-3.html']
```

爬取从`CrawlDepth0-1.html`开始。该页面有两行，一行到`CrawlDepth0-2.html`，一行到`CrawlDepth1-1.html`。然后请求解析它们。考虑到起始页面在深度 0，这些页面在深度 1，是我们深度的限制。因此，我们将看到这两个页面被解析。但是，请注意，这两个页面的所有链接，虽然请求解析，但由于它们在深度 2，超出了指定的限制，因此被 Scrapy 忽略。

现在将深度限制更改为 2：

```py
process = CrawlerProcess({
  'LOG_LEVEL': 'CRITICAL',
  'DEPTH_LIMIT': 2,
  'DEPT_STATS': True })
```

然后输出变成如下：

```py
Parsing: <200 http://localhost:8080/CrawlDepth0-1.html>
Requesting crawl of: http://localhost:8080/CrawlDepth0-2.html
Requesting crawl of: http://localhost:8080/Depth1/CrawlDepth1-1.html
Parsing: <200 http://localhost:8080/Depth1/CrawlDepth1-1.html>
Requesting crawl of: http://localhost:8080/Depth1/CrawlDepth1-2.html
Requesting crawl of: http://localhost:8080/Depth1/depth1/CrawlDepth1-2.html
Requesting crawl of: http://localhost:8080/Depth1/depth2/CrawlDepth2-1.html
Parsing: <200 http://localhost:8080/CrawlDepth0-2.html>
Requesting crawl of: http://localhost:8080/CrawlDepth0-3.html
Parsing: <200 http://localhost:8080/Depth1/depth2/CrawlDepth2-1.html>
Parsing: <200 http://localhost:8080/CrawlDepth0-3.html>
Parsing: <200 http://localhost:8080/Depth1/CrawlDepth1-2.html>
Requesting crawl of: http://localhost:8080/Depth1/CrawlDepth1-3.html
<scrapy.statscollectors.MemoryStatsCollector object at 0x10d3d44e0>
Crawled: ['http://localhost:8080/CrawlDepth0-1.html', 'http://localhost:8080/Depth1/CrawlDepth1-1.html', 'http://localhost:8080/CrawlDepth0-2.html', 'http://localhost:8080/Depth1/depth2/CrawlDepth2-1.html', 'http://localhost:8080/CrawlDepth0-3.html', 'http://localhost:8080/Depth1/CrawlDepth1-2.html']
Requested: ['http://localhost:8080/CrawlDepth0-2.html', 'http://localhost:8080/Depth1/CrawlDepth1-1.html', 'http://localhost:8080/Depth1/CrawlDepth1-2.html', 'http://localhost:8080/Depth1/depth1/CrawlDepth1-2.html', 'http://localhost:8080/Depth1/depth2/CrawlDepth2-1.html', 'http://localhost:8080/CrawlDepth0-3.html', 'http://localhost:8080/Depth1/CrawlDepth1-3.html']
```

请注意，之前被忽略的三个页面，当`DEPTH_LIMIT`设置为 1 时，现在被解析了。现在，这个深度下找到的链接，比如`CrawlDepth1-3.html`页面的链接，现在被忽略了，因为它们的深度超过了 2。

# 控制爬取的长度

爬取的长度，即可以解析的页面数量，可以通过`CLOSESPIDER_PAGECOUNT`设置来控制。

# 如何操作

我们将使用`06/07_limit_length.py`中的脚本。该脚本和爬虫与 NASA 站点地图爬虫相同，只是增加了以下配置来限制解析的页面数量为 5：

```py
if __name__ == "__main__":
  process = CrawlerProcess({
  'LOG_LEVEL': 'INFO',
  'CLOSESPIDER_PAGECOUNT': 5
  })
  process.crawl(Spider)
  process.start()
```

当运行时，将生成以下输出（在日志输出中交错）：

```py
<200 https://www.nasa.gov/exploration/systems/sls/multimedia/sls-hardware-being-moved-on-kamag-transporter.html>
<200 https://www.nasa.gov/exploration/systems/sls/M17-057.html>
<200 https://www.nasa.gov/press-release/nasa-awards-contract-for-center-protective-services-for-glenn-research-center/>
<200 https://www.nasa.gov/centers/marshall/news/news/icymi1708025/>
<200 https://www.nasa.gov/content/oracles-completed-suit-case-flight-series-to-ascension-island/>
<200 https://www.nasa.gov/feature/goddard/2017/asteroid-sample-return-mission-successfully-adjusts-course/>
<200 https://www.nasa.gov/image-feature/jpl/pia21754/juling-crater/>
```

# 工作原理

请注意，我们将页面限制设置为 5，但实际示例解析了 7 页。`CLOSESPIDER_PAGECOUNT`的值应被视为 Scrapy 将至少执行的值，但可能会略微超出。

# 处理分页网站

分页将大量内容分成多个页面。通常，这些页面有一个供用户点击的上一页/下一页链接。这些链接通常可以通过 XPath 或其他方式找到，然后跟随以到达下一页（或上一页）。让我们来看看如何使用 Scrapy 遍历页面。我们将看一个假设的例子，爬取自动互联网搜索结果。这些技术直接适用于许多具有搜索功能的商业网站，并且很容易修改以适应这些情况。

# 准备工作

我们将演示如何处理分页，示例将从提供的容器网站中爬取一组页面。该网站模拟了五个页面，每个页面上都有上一页和下一页的链接，以及每个页面中的一些嵌入数据，我们将提取这些数据。

这个集合的第一页可以在`http://localhost:5001/pagination/page1.html`中看到。以下图片显示了这个页面的打开情况，我们正在检查下一页按钮：

![](img/2c9af8d6-9b76-47e9-965e-1875830119d4.png)检查下一页按钮

页面有两个感兴趣的部分。第一个是下一页按钮的链接。这个链接通常有一个类来标识链接作为下一页的链接。我们可以使用这个信息来找到这个链接。在这种情况下，我们可以使用以下 XPath 找到它：

```py
//*/a[@class='next']
```

第二个感兴趣的部分实际上是从页面中检索我们想要的数据。在这些页面上，这是由具有`class="data"`属性的`<div>`标签标识的。这些页面只有一个数据项，但在这个搜索结果页面爬取的示例中，我们将提取多个项目。

现在让我们实际运行这些页面的爬虫。

# 如何操作

有一个名为`06/08_scrapy_pagination.py`的脚本。用 Python 运行此脚本，Scrapy 将输出大量内容，其中大部分将是标准的 Scrapy 调试输出。但是，在这些输出中，您将看到我们提取了所有五个页面上的数据项：

```py
Page 1 Data
Page 2 Data
Page 3 Data
Page 4 Data
Page 5 Data
```

# 工作原理

代码从定义`CrawlSpider`和起始 URL 开始：

```py
class PaginatedSearchResultsSpider(CrawlSpider):
    name = "paginationscraper"
  start_urls = [
"http://localhost:5001/pagination/page1.html"
  ]
```

然后定义了规则字段，它告诉 Scrapy 如何解析每个页面以查找链接。此代码使用前面讨论的 XPath 来查找页面中的下一个链接。Scrapy 将使用此规则在每个页面上查找下一个要处理的页面，并将该请求排队等待处理。对于找到的每个页面，回调参数告诉 Scrapy 调用哪个方法进行处理，在本例中是`parse_result_page`：

```py
rules = (
# Extract links for next pages
  Rule(LinkExtractor(allow=(),
restrict_xpaths=("//*/a[@class='next']")),
callback='parse_result_page', follow=True),
)
```

声明了一个名为`all_items`的单个列表变量来保存我们找到的所有项目：

```py
all_items = []
```

然后定义了`parse_start_url`方法。Scrapy 将调用此方法来解析爬行中的初始 URL。该函数简单地将处理推迟到`parse_result_page`：

```py
def parse_start_url(self, response):
  return self.parse_result_page(response)
```

然后，`parse_result_page`方法使用 XPath 来查找`<div class="data">`标签中`<h1>`标签内的文本。然后将该文本附加到`all_items`列表中：

```py
def parse_result_page(self, response):
    data_items = response.xpath("//*/div[@class='data']/h1/text()")
for data_item in data_items:
 self.all_items.append(data_item.root)
```

爬行完成后，将调用`closed()`方法并写出`all_items`字段的内容：

```py
def closed(self, reason):
  for i in self.all_items:
  print(i) 
```

使用 Python 作为脚本运行爬虫，如下所示：

```py
if __name__ == "__main__":
  process = CrawlerProcess({
  'LOG_LEVEL': 'DEBUG',
  'CLOSESPIDER_PAGECOUNT': 10   })
  process.crawl(ImdbSearchResultsSpider)
  process.start()
```

请注意，`CLOSESPIDER_PAGECOUNT`属性被设置为`10`。这超过了该网站上的页面数量，但在许多（或大多数）情况下，搜索结果可能会有数千个页面。在适当数量的页面后停止是一个很好的做法。这是爬虫的良好行为，因为在前几页之后，与您的搜索相关的项目的相关性急剧下降，因此在前几页之后继续爬行会大大减少回报，通常最好在几页后停止。

# 还有更多...

正如在本教程开始时提到的，这很容易修改为在各种内容网站上进行各种自动搜索。这种做法可能会推动可接受使用的极限，因此这里进行了泛化。但是，要获取更多实际示例，请访问我的博客：`www.smac.io`。

# 处理表单和基于表单的授权

我们经常需要在爬取网站内容之前登录网站。这通常是通过一个表单完成的，我们在其中输入用户名和密码，按*Enter*，然后获得以前隐藏的内容的访问权限。这种类型的表单认证通常称为 cookie 授权，因为当我们授权时，服务器会创建一个 cookie，用于验证您是否已登录。Scrapy 尊重这些 cookie，所以我们所需要做的就是在爬行过程中自动化表单。

# 准备工作

我们将在容器网站的页面上爬行以下 URL：`http://localhost:5001/home/secured`。在此页面上，以及从该页面链接出去的页面，有我们想要抓取的内容。但是，此页面被登录阻止。在浏览器中打开页面时，我们会看到以下登录表单，我们可以在其中输入`darkhelmet`作为用户名，`vespa`作为密码：

![](img/518cef3e-91c8-47a3-b978-020504dcc4ca.png)输入用户名和密码凭证

按下*Enter*后，我们将得到验证，并被带到最初想要的页面。

那里没有太多的内容，但这条消息足以验证我们已经登录，我们的爬虫也知道这一点。

# 如何操作

我们按照以下步骤进行：

1.  如果您检查登录页面的 HTML，您会注意到以下表单代码：

```py
<form action="/Account/Login" method="post"><div>
 <label for="Username">Username</label>
 <input type="text" id="Username" name="Username" value="" />
 <span class="field-validation-valid" data-valmsg-for="Username" data-valmsg-replace="true"></span></div>
<div>
 <label for="Password">Password</label>
 <input type="password" id="Password" name="Password" />
 <span class="field-validation-valid" data-valmsg-for="Password" data-valmsg-replace="true"></span>
 </div> 
 <input type="hidden" name="returnUrl" />
<input name="submit" type="submit" value="Login"/>
 <input name="__RequestVerificationToken" type="hidden" value="CfDJ8CqzjGWzUMJKkKCmxuBIgZf3UkeXZnVKBwRV_Wu4qUkprH8b_2jno5-1SGSNjFqlFgLie84xI2ZBkhHDzwgUXpz6bbBwER0v_-fP5iTITiZi2VfyXzLD_beXUp5cgjCS5AtkIayWThJSI36InzBqj2A" /></form>
```

1.  要使 Scrapy 中的表单处理器工作，我们需要该表单中用户名和密码字段的 ID。它们分别是`Username`和`Password`。现在我们可以使用这些信息创建一个蜘蛛。这个蜘蛛在脚本文件`06/09_forms_auth.py`中。蜘蛛定义以以下内容开始：

```py
class Spider(scrapy.Spider):
  name = 'spider'
  start_urls = ['http://localhost:5001/home/secured']
  login_user = 'darkhelmet'
  login_pass = 'vespa'
```

1.  我们在类中定义了两个字段`login_user`和`login_pass`，用于保存我们要使用的用户名。爬行也将从指定的 URL 开始。

1.  然后更改`parse`方法以检查页面是否包含登录表单。这是通过使用 XPath 来查看页面是否有一个类型为密码的输入表单，并且具有`id`为`Password`的方式来完成的：

```py
def parse(self, response):
  print("Parsing: ", response)    count_of_password_fields = int(float(response.xpath("count(//*/input[@type='password' and @id='Password'])").extract()[0]))
  if count_of_password_fields > 0:
  print("Got a password page") 
```

1.  如果找到了该字段，我们将返回一个`FormRequest`给 Scrapy，使用它的`from_response`方法生成：

```py
return scrapy.FormRequest.from_response(
 response,
  formdata={'Username': self.login_user, 'Password': self.login_pass},
  callback=self.after_login)
```

1.  这个函数接收响应，然后是一个指定需要插入数据的字段的 ID 的字典，以及这些值。然后定义一个回调函数，在 Scrapy 执行这个 FormRequest 后执行，并将生成的表单内容传递给它：

```py
def after_login(self, response):
  if "This page is secured" in str(response.body):
  print("You have logged in ok!")
```

1.  这个回调函数只是寻找单词`This page is secured`，只有在登录成功时才会返回。当成功运行时，我们将从我们的爬虫的打印语句中看到以下输出：

```py
Parsing: <200 http://localhost:5001/account/login?ReturnUrl=%2Fhome%2Fsecured>
Got a password page
You have logged in ok!
```

# 它是如何工作的

当您创建一个`FormRequest`时，您正在指示 Scrapy 代表您的进程构造一个表单 POST 请求，使用指定字典中的数据作为 POST 请求中的表单参数。它构造这个请求并将其发送到服务器。在收到 POST 的答复后，它调用指定的回调函数。

# 还有更多...

这种技术在许多其他类型的表单输入中也很有用，不仅仅是登录表单。这可以用于自动化，然后执行任何类型的 HTML 表单请求，比如下订单，或者用于执行搜索操作的表单。

# 处理基本授权

一些网站使用一种称为*基本授权*的授权形式。这在其他授权方式（如 cookie 授权或 OAuth）出现之前很流行。它也常见于企业内部网络和一些 Web API。在基本授权中，一个头部被添加到 HTTP 请求中。这个头部，`Authorization`，传递了 Basic 字符串，然后是值`<username>:<password>`的 base64 编码。所以在 darkhelmet 的情况下，这个头部会如下所示：

```py
Authorization: Basic ZGFya2hlbG1ldDp2ZXNwYQ==, with ZGFya2hlbG1ldDp2ZXNwYQ== being darkhelmet:vespa base 64 encoded.
```

请注意，这并不比以明文发送更安全（尽管通过 HTTPS 执行时是安全的）。然而，大部分情况下，它已经被更健壮的授权表单所取代，甚至 cookie 授权允许更复杂的功能，比如声明：

# 如何做到

在 Scrapy 中支持基本授权是很简单的。要使其对爬虫和爬取的特定网站起作用，只需在您的爬虫中定义`http_user`，`http_pass`和`name`字段。以下是示例：

```py
class SomeIntranetSiteSpider(CrawlSpider):
    http_user = 'someuser'
    http_pass = 'somepass'
    name = 'intranet.example.com'
    # .. rest of the spider code omitted ...
```

# 它是如何工作的

当爬虫爬取由名称指定的网站上的任何页面时，它将使用`http_user`和`http_pass`的值来构造适当的标头。

# 还有更多...

请注意，这个任务是由 Scrapy 的`HttpAuthMiddleware`模块执行的。有关基本授权的更多信息也可以在[`developer.mozilla.org/en-US/docs/Web/HTTP/Authentication`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication)上找到。

# 通过代理来防止被屏蔽

有时候您可能会因为被识别为爬虫而被屏蔽，有时候是因为网站管理员看到来自统一 IP 的爬取请求，然后他们会简单地屏蔽对该 IP 的访问。

为了帮助防止这个问题，可以在 Scrapy 中使用代理随机化中间件。存在一个名为`scrapy-proxies`的库，它实现了代理随机化功能。

# 准备工作

您可以从 GitHub 上获取`scrapy-proxies`，网址为[`github.com/aivarsk/scrapy-proxies`](https://github.com/aivarsk/scrapy-proxies)，或者使用`pip install scrapy_proxies`进行安装。

# 如何做到

使用`scrapy-proxies`是通过配置完成的。首先要配置`DOWNLOADER_MIDDLEWARES`，并确保安装了`RetryMiddleware`，`RandomProxy`和`HttpProxyMiddleware`。以下是一个典型的配置：

```py
# Retry many times since proxies often fail
RETRY_TIMES = 10
# Retry on most error codes since proxies fail for different reasons
RETRY_HTTP_CODES = [500, 503, 504, 400, 403, 404, 408]

DOWNLOADER_MIDDLEWARES = {
 'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
 'scrapy_proxies.RandomProxy': 100,
 'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
}
```

`PROXY_LIST`设置被配置为指向一个包含代理列表的文件：

```py
PROXY_LIST = '/path/to/proxy/list.txt'
```

然后，我们需要让 Scrapy 知道`PROXY_MODE`：

```py
# Proxy mode
# 0 = Every requests have different proxy
# 1 = Take only one proxy from the list and assign it to every requests
# 2 = Put a custom proxy to use in the settings
PROXY_MODE = 0
```

如果`PROXY_MODE`是`2`，那么您必须指定一个`CUSTOM_PROXY`：

```py
CUSTOM_PROXY = "http://host1:port"
```

# 它是如何工作的

这个配置基本上告诉 Scrapy，如果对页面的请求失败，并且每个 URL 最多重试`RETRY_TIMES`次中的任何一个`RETRY_HTTP_CODES`，则使用`PROXY_LIST`指定的文件中的代理，并使用`PROXY_MODE`定义的模式。通过这种方式，您可以让 Scrapy 退回到任意数量的代理服务器，以从不同的 IP 地址和/或端口重试请求。

# 随机化用户代理

您使用的用户代理可能会影响爬虫的成功。一些网站将直接拒绝为特定的用户代理提供内容。这可能是因为用户代理被识别为被禁止的爬虫，或者用户代理是不受支持的浏览器（即 Internet Explorer 6）的用户代理。

对爬虫的控制另一个原因是，根据指定的用户代理，内容可能会在网页服务器上以不同的方式呈现。目前移动网站通常会这样做，但也可以用于桌面，比如为旧版浏览器提供更简单的内容。

因此，将用户代理设置为默认值以外的其他值可能是有用的。Scrapy 默认使用名为`scrapybot`的用户代理。可以通过使用`BOT_NAME`参数进行配置。如果使用 Scrapy 项目，Scrapy 将把代理设置为您的项目名称。

对于更复杂的方案，有两个常用的扩展可以使用：`scrapy-fake-agent`和`scrapy-random-useragent`。

# 如何做到这一点

我们按照以下步骤进行操作：

1.  `scrapy-fake-useragent`可在 GitHub 上找到，网址为[`github.com/alecxe/scrapy-fake-useragent`](https://github.com/alecxe/scrapy-fake-useragent)，而`scrapy-random-useragent`可在[`github.com/cnu/scrapy-random-useragent`](https://github.com/cnu/scrapy-random-useragent)找到。您可以使用`pip install scrapy-fake-agent`和/或`pip install scrapy-random-useragent`来包含它们。

1.  `scrapy-random-useragent`将从文件中为每个请求选择一个随机用户代理。它配置在两个设置中：

```py
DOWNLOADER_MIDDLEWARES = {
    'scrapy.contrib.downloadermiddleware.useragent.UserAgentMiddleware': None,
    'random_useragent.RandomUserAgentMiddleware': 400
}
```

1.  这将禁用现有的`UserAgentMiddleware`，并用`RandomUserAgentMiddleware`中提供的实现来替换它。然后，您需要配置一个包含用户代理名称列表的文件的引用：

```py
USER_AGENT_LIST = "/path/to/useragents.txt"
```

1.  配置完成后，每个请求将使用文件中的随机用户代理。

1.  `scrapy-fake-useragent`使用了不同的模型。它从在线数据库中检索用户代理，该数据库跟踪使用最普遍的用户代理。配置 Scrapy 以使用它的设置如下：

```py
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 400,
}
```

1.  它还具有设置使用的用户代理类型的能力，例如移动或桌面，以强制选择这两个类别中的用户代理。这是使用`RANDOM_UA_TYPE`设置执行的，默认为随机。

1.  如果使用`scrapy-fake-useragent`与任何代理中间件，那么您可能希望对每个代理进行随机化。这可以通过将`RANDOM_UA_PER_PROXY`设置为 True 来实现。此外，您还需要将`RandomUserAgentMiddleware`的优先级设置为大于`scrapy-proxies`，以便在处理之前设置代理。

# 缓存响应

Scrapy 具有缓存 HTTP 请求的能力。如果页面已经被访问过，这可以大大减少爬取时间。通过启用缓存，Scrapy 将存储每个请求和响应。

# 如何做到这一点

`06/10_file_cache.py`脚本中有一个可用的示例。在 Scrapy 中，默认情况下禁用了缓存中间件。要启用此缓存，将`HTTPCACHE_ENABLED`设置为`True`，将`HTTPCACHE_DIR`设置为文件系统上的一个目录（使用相对路径将在项目的数据文件夹中创建目录）。为了演示，此脚本运行了 NASA 网站的爬取，并缓存了内容。它的配置如下：

```py
if __name__ == "__main__":
  process = CrawlerProcess({
  'LOG_LEVEL': 'CRITICAL',
  'CLOSESPIDER_PAGECOUNT': 50,
  'HTTPCACHE_ENABLED': True,
  'HTTPCACHE_DIR': "."
  })
  process.crawl(Spider)
  process.start()
```

我们要求 Scrapy 使用文件进行缓存，并在当前文件夹中创建一个子目录。我们还指示它将爬取限制在大约 500 页。运行此操作时，爬取将大约需要一分钟（取决于您的互联网速度），并且大约会有 500 行的输出。

第一次执行后，您会发现您的目录中现在有一个`.scrapy`文件夹，其中包含缓存数据。 结构将如下所示：

![](img/5c6f662d-595c-46d0-95f7-c118ffe6afc4.png)

再次运行脚本只需要几秒钟，将产生相同的输出/报告已解析的页面，只是这次内容将来自缓存而不是 HTTP 请求。

# 还有更多...

在 Scrapy 中有许多缓存的配置和选项。默认情况下，由`HTTPCACHE_EXPIRATION_SECS`指定的缓存过期时间设置为 0。 0 表示缓存项永远不会过期，因此一旦写入，Scrapy 将永远不会通过 HTTP 再次请求该项。实际上，您可能希望将其设置为某个会过期的值。

文件存储仅是缓存的选项之一。通过将`HTTPCACHE_STORAGE`设置为`scrapy.extensions.httpcache.DbmCacheStorage`或`scrapy.extensions.httpcache.LeveldbCacheStorage`，也可以将项目缓存在 DMB 和 LevelDB 中。如果您愿意，还可以编写自己的代码，将页面内容存储在另一种类型的数据库或云存储中。

最后，我们来到缓存策略。Scrapy 自带两种内置策略：Dummy（默认）和 RFC2616。这可以通过将`HTTPCACHE_POLICY`设置更改为`scrapy.extensions.httpcache.DummyPolicy`或`scrapy.extensions.httpcache.RFC2616Policy`来设置。

RFC2616 策略通过以下操作启用 HTTP 缓存控制意识：

+   不要尝试存储设置了 no-store 缓存控制指令的响应/请求

+   如果设置了 no-cache 缓存控制指令，即使是新鲜的响应，也不要从缓存中提供响应

+   从 max-age 缓存控制指令计算新鲜度生存期

+   从 Expires 响应标头计算新鲜度生存期

+   从 Last-Modified 响应标头计算新鲜度生存期（Firefox 使用的启发式）

+   从 Age 响应标头计算当前年龄

+   从日期标头计算当前年龄

+   根据 Last-Modified 响应标头重新验证陈旧的响应

+   根据 ETag 响应标头重新验证陈旧的响应

+   为任何接收到的响应设置日期标头

+   支持请求中的 max-stale 缓存控制指令
