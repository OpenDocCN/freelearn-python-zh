# 开始爬取

在本章中，我们将涵盖以下主题：

+   设置Python开发环境

+   使用Requests和Beautiful Soup爬取Python.org

+   使用urllib3和Beautiful Soup爬取Python.org

+   使用Scrapy爬取Python.org

+   使用Selenium和PhantomJs爬取Python.org

# 介绍

网上可用的数据量在数量和形式上都在持续增长。企业需要这些数据来做决策，特别是随着需要大量数据进行训练的机器学习工具的爆炸式增长。很多数据可以通过应用程序编程接口获得，但同时也有很多有价值的数据仍然只能通过网页抓取获得。

本章将重点介绍设置爬取环境的几个基本原理，并使用行业工具进行基本数据请求。Python是本书的首选编程语言，也是许多进行爬取系统构建的人的首选语言。它是一种易于使用的编程语言，拥有丰富的工具生态系统，适用于许多任务。如果您使用其他语言进行编程，您会发现很容易上手，也许永远不会回头！

# 设置Python开发环境

如果您以前没有使用过Python，拥有一个可用的开发环境是很重要的。本书中的示例将全部使用Python，并且是交互式示例的混合，但主要是作为脚本实现，由Python解释器解释。这个示例将向您展示如何使用`virtualenv`设置一个隔离的开发环境，并使用`pip`管理项目依赖。我们还会获取本书的代码并将其安装到Python虚拟环境中。

# 准备工作

我们将专门使用Python 3.x，特别是在我的情况下是3.6.1。虽然Mac和Linux通常已安装了Python 2版本，而Windows系统没有安装。因此很可能需要安装Python 3。您可以在www.python.org找到Python安装程序的参考资料。

您可以使用`python --version`检查Python的版本

![](assets/e9039d11-8e50-44c6-8204-3199ae5d7b1e.png)`pip`已经随Python 3.x一起安装，因此我们将省略其安装说明。此外，本书中的所有命令行示例都在Mac上运行。对于Linux用户，命令应该是相同的。在Windows上，有替代命令（如dir而不是ls），但这些替代命令将不会被涵盖。

# 如何做...

我们将使用`pip`安装许多包。这些包将被安装到一个Python环境中。通常可能会与其他包存在版本冲突，因此在跟着本书的示例进行操作时，一个很好的做法是创建一个新的虚拟Python环境，确保我们将使用的包能够正常工作。

虚拟Python环境是用`virtualenv`工具管理的。可以用以下命令安装它：

```py
~ $ pip install virtualenv
Collecting virtualenv
 Using cached virtualenv-15.1.0-py2.py3-none-any.whl
Installing collected packages: virtualenv
Successfully installed virtualenv-15.1.0
```

现在我们可以使用`virtualenv`。但在那之前，让我们简要地看一下`pip`。这个命令从PyPI安装Python包，PyPI是一个拥有成千上万个包的包存储库。我们刚刚看到了使用pip的install子命令，这可以确保一个包被安装。我们也可以用`pip list`来查看当前安装的所有包：

```py
~ $ pip list
alabaster (0.7.9)
amqp (1.4.9)
anaconda-client (1.6.0)
anaconda-navigator (1.5.3)
anaconda-project (0.4.1)
aniso8601 (1.3.0)
```

我截取了前几行，因为安装了很多包。对我来说，安装了222个包。

也可以使用`pip uninstall`命令卸载包。我留给您去尝试一下。

现在回到`virtualenv`。使用`virtualenv`非常简单。让我们用它来创建一个环境并安装来自github的代码。让我们一步步走过这些步骤：

1.  创建一个代表项目的目录并进入该目录。

```py
~ $ mkdir pywscb
~ $ cd pywscb
```

1.  初始化一个名为env的虚拟环境文件夹：

```py
pywscb $ virtualenv env
Using base prefix '/Users/michaelheydt/anaconda'
New python executable in /Users/michaelheydt/pywscb/env/bin/python
copying /Users/michaelheydt/anaconda/bin/python => /Users/michaelheydt/pywscb/env/bin/python
copying /Users/michaelheydt/anaconda/bin/../lib/libpython3.6m.dylib => /Users/michaelheydt/pywscb/env/lib/libpython3.6m.dylib
Installing setuptools, pip, wheel...done.
```

1.  这将创建一个env文件夹。让我们看看安装了什么。

```py
pywscb $ ls -la env
total 8
drwxr-xr-x 6  michaelheydt staff 204 Jan 18 15:38 .
drwxr-xr-x 3  michaelheydt staff 102 Jan 18 15:35 ..
drwxr-xr-x 16 michaelheydt staff 544 Jan 18 15:38 bin
drwxr-xr-x 3  michaelheydt staff 102 Jan 18 15:35 include
drwxr-xr-x 4  michaelheydt staff 136 Jan 18 15:38 lib
-rw-r--r-- 1  michaelheydt staff 60 Jan 18 15:38  pip-selfcheck.json
```

1.  现在我们激活虚拟环境。这个命令使用`env`文件夹中的内容来配置Python。之后所有的python活动都是相对于这个虚拟环境的。

```py
pywscb $ source env/bin/activate
(env) pywscb $
```

1.  我们可以使用以下命令检查python是否确实使用了这个虚拟环境：

```py
(env) pywscb $ which python
/Users/michaelheydt/pywscb/env/bin/python
```

有了我们创建的虚拟环境，让我们克隆书籍的示例代码并看看它的结构。

```py
(env) pywscb $ git clone https://github.com/PacktBooks/PythonWebScrapingCookbook.git
 Cloning into 'PythonWebScrapingCookbook'...
 remote: Counting objects: 420, done.
 remote: Compressing objects: 100% (316/316), done.
 remote: Total 420 (delta 164), reused 344 (delta 88), pack-reused 0
 Receiving objects: 100% (420/420), 1.15 MiB | 250.00 KiB/s, done.
 Resolving deltas: 100% (164/164), done.
 Checking connectivity... done.
```

这创建了一个`PythonWebScrapingCookbook`目录。

```py
(env) pywscb $ ls -l
 total 0
 drwxr-xr-x 9 michaelheydt staff 306 Jan 18 16:21 PythonWebScrapingCookbook
 drwxr-xr-x 6 michaelheydt staff 204 Jan 18 15:38 env
```

让我们切换到它并检查内容。

```py
(env) PythonWebScrapingCookbook $ ls -l
 total 0
 drwxr-xr-x 15 michaelheydt staff 510 Jan 18 16:21 py
 drwxr-xr-x 14 michaelheydt staff 476 Jan 18 16:21 www
```

有两个目录。大部分Python代码都在`py`目录中。`www`包含一些我们将使用的网络内容，我们将使用本地web服务器不时地访问它。让我们看看`py`目录的内容：

```py
(env) py $ ls -l
 total 0
 drwxr-xr-x 9  michaelheydt staff 306 Jan 18 16:21 01
 drwxr-xr-x 25 michaelheydt staff 850 Jan 18 16:21 03
 drwxr-xr-x 21 michaelheydt staff 714 Jan 18 16:21 04
 drwxr-xr-x 10 michaelheydt staff 340 Jan 18 16:21 05
 drwxr-xr-x 14 michaelheydt staff 476 Jan 18 16:21 06
 drwxr-xr-x 25 michaelheydt staff 850 Jan 18 16:21 07
 drwxr-xr-x 14 michaelheydt staff 476 Jan 18 16:21 08
 drwxr-xr-x 7  michaelheydt staff 238 Jan 18 16:21 09
 drwxr-xr-x 7  michaelheydt staff 238 Jan 18 16:21 10
 drwxr-xr-x 9  michaelheydt staff 306 Jan 18 16:21 11
 drwxr-xr-x 8  michaelheydt staff 272 Jan 18 16:21 modules
```

每个章节的代码都在与章节匹配的编号文件夹中（第2章没有代码，因为它都是交互式Python）。

请注意，有一个`modules`文件夹。本书中的一些食谱使用这些模块中的代码。确保你的Python路径指向这个文件夹。在Mac和Linux上，你可以在你的`.bash_profile`文件中设置这一点（在Windows上是在环境变量对话框中）：

```py
export PYTHONPATH="/users/michaelheydt/dropbox/packt/books/pywebscrcookbook/code/py/modules"
export PYTHONPATH
```

每个文件夹中的内容通常遵循与章节中食谱顺序相匹配的编号方案。以下是第6章文件夹的内容：

```py
(env) py $ ls -la 06
 total 96
 drwxr-xr-x 14 michaelheydt staff 476 Jan 18 16:21 .
 drwxr-xr-x 14 michaelheydt staff 476 Jan 18 16:26 ..
 -rw-r--r-- 1  michaelheydt staff 902 Jan 18 16:21  01_scrapy_retry.py
 -rw-r--r-- 1  michaelheydt staff 656 Jan 18 16:21  02_scrapy_redirects.py
 -rw-r--r-- 1  michaelheydt staff 1129 Jan 18 16:21 03_scrapy_pagination.py
 -rw-r--r-- 1  michaelheydt staff 488 Jan 18 16:21  04_press_and_wait.py
 -rw-r--r-- 1  michaelheydt staff 580 Jan 18 16:21  05_allowed_domains.py
 -rw-r--r-- 1  michaelheydt staff 826 Jan 18 16:21  06_scrapy_continuous.py
 -rw-r--r-- 1  michaelheydt staff 704 Jan 18 16:21  07_scrape_continuous_twitter.py
 -rw-r--r-- 1  michaelheydt staff 1409 Jan 18 16:21 08_limit_depth.py
 -rw-r--r-- 1  michaelheydt staff 526 Jan 18 16:21  09_limit_length.py
 -rw-r--r-- 1  michaelheydt staff 1537 Jan 18 16:21 10_forms_auth.py
 -rw-r--r-- 1  michaelheydt staff 597 Jan 18 16:21  11_file_cache.py
 -rw-r--r-- 1  michaelheydt staff 1279 Jan 18 16:21 12_parse_differently_based_on_rules.py
```

在食谱中，我会说明我们将使用`<章节目录>`/`<食谱文件名>`中的脚本。

恭喜，你现在已经配置了一个带有书籍代码的Python环境！

现在，如果你想退出Python虚拟环境，你可以使用以下命令退出：

```py
(env) py $ deactivate
 py $
```

检查一下python，我们可以看到它已经切换回来了：

```py
py $ which python
 /Users/michaelheydt/anaconda/bin/python
```

我不会在本书的其余部分使用虚拟环境。当你看到命令提示时，它们将是以下形式之一"<目录> $"或者简单的"$"。

现在让我们开始爬取一些数据。

# 使用Requests和Beautiful Soup从Python.org上爬取数据

在这个食谱中，我们将安装Requests和Beautiful Soup，并从www.python.org上爬取一些内容。我们将安装这两个库，并对它们有一些基本的了解。在随后的章节中，我们将深入研究它们。

# 准备好了...

在这个食谱中，我们将从[https://www.python.org/events/pythonevents](https://www.python.org/events/pythonevents)中爬取即将到来的Python事件。以下是`Python.org事件页面`的一个示例（它经常更改，所以你的体验会有所不同）：

![](assets/c4caf889-b8fa-4f5e-87dc-d6d78921bddb.png)

我们需要确保Requests和Beautiful Soup已安装。我们可以使用以下命令来安装：

```py
pywscb $ pip install requests
Downloading/unpacking requests
 Downloading requests-2.18.4-py2.py3-none-any.whl (88kB): 88kB downloaded
Downloading/unpacking certifi>=2017.4.17 (from requests)
 Downloading certifi-2018.1.18-py2.py3-none-any.whl (151kB): 151kB downloaded
Downloading/unpacking idna>=2.5,<2.7 (from requests)
 Downloading idna-2.6-py2.py3-none-any.whl (56kB): 56kB downloaded
Downloading/unpacking chardet>=3.0.2,<3.1.0 (from requests)
 Downloading chardet-3.0.4-py2.py3-none-any.whl (133kB): 133kB downloaded
Downloading/unpacking urllib3>=1.21.1,<1.23 (from requests)
 Downloading urllib3-1.22-py2.py3-none-any.whl (132kB): 132kB downloaded
Installing collected packages: requests, certifi, idna, chardet, urllib3
Successfully installed requests certifi idna chardet urllib3
Cleaning up...
pywscb $ pip install bs4
Downloading/unpacking bs4
 Downloading bs4-0.0.1.tar.gz
 Running setup.py (path:/Users/michaelheydt/pywscb/env/build/bs4/setup.py) egg_info for package bs4
```

# 如何做...

现在让我们去学习一下爬取一些事件。对于这个食谱，我们将开始使用交互式python。

1.  用`ipython`命令启动它：

```py
$ ipython
Python 3.6.1 |Anaconda custom (x86_64)| (default, Mar 22 2017, 19:25:17)
Type "copyright", "credits" or "license" for more information.
IPython 5.1.0 -- An enhanced Interactive Python.
? -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help -> Python's own help system.
object? -> Details about 'object', use 'object??' for extra details.
In [1]:
```

1.  接下来导入Requests

```py
In [1]: import requests
```

1.  我们现在使用requests来对以下url进行GET HTTP请求：[https://www.python.org/events/python-events/](https://www.python.org/events/python-events/)，通过进行`GET`请求：

```py
In [2]: url = 'https://www.python.org/events/python-events/'
In [3]: req = requests.get(url)
```

1.  这下载了页面内容，但它存储在我们的requests对象req中。我们可以使用`.text`属性检索内容。这打印了前200个字符。

```py
req.text[:200]
Out[4]: '<!doctype html>\n<!--[if lt IE 7]> <html class="no-js ie6 lt-ie7 lt-ie8 lt-ie9"> <![endif]-->\n<!--[if IE 7]> <html class="no-js ie7 lt-ie8 lt-ie9"> <![endif]-->\n<!--[if IE 8]> <h'
```

现在我们有了页面的原始HTML。我们现在可以使用beautiful soup来解析HTML并检索事件数据。

1.  首先导入Beautiful Soup

```py
In [5]: from bs4 import BeautifulSoup
```

1.  现在我们创建一个`BeautifulSoup`对象并传递HTML。

```py
In [6]: soup = BeautifulSoup(req.text, 'lxml')
```

1.  现在我们告诉Beautiful Soup找到最近事件的主要`<ul>`标签，然后获取其下的所有`<li>`标签。

```py
In [7]: events = soup.find('ul', {'class': 'list-recent-events'}).findAll('li')
```

1.  最后，我们可以循环遍历每个`<li>`元素，提取事件详情，并将每个打印到控制台：

```py
In [13]: for event in events:
 ...: event_details = dict()
 ...: event_details['name'] = event_details['name'] = event.find('h3').find("a").text
 ...: event_details['location'] = event.find('span', {'class', 'event-location'}).text
 ...: event_details['time'] = event.find('time').text
 ...: print(event_details)
 ...:
{'name': 'PyCascades 2018', 'location': 'Granville Island Stage, 1585 Johnston St, Vancouver, BC V6H 3R9, Canada', 'time': '22 Jan. – 24 Jan. 2018'}
{'name': 'PyCon Cameroon 2018', 'location': 'Limbe, Cameroon', 'time': '24 Jan. – 29 Jan. 2018'}
{'name': 'FOSDEM 2018', 'location': 'ULB Campus du Solbosch, Av. F. D. Roosevelt 50, 1050 Bruxelles, Belgium', 'time': '03 Feb. – 05 Feb. 2018'}
{'name': 'PyCon Pune 2018', 'location': 'Pune, India', 'time': '08 Feb. – 12 Feb. 2018'}
{'name': 'PyCon Colombia 2018', 'location': 'Medellin, Colombia', 'time': '09 Feb. – 12 Feb. 2018'}
{'name': 'PyTennessee 2018', 'location': 'Nashville, TN, USA', 'time': '10 Feb. – 12 Feb. 2018'}
```

整个示例都在`01/01_events_with_requests.py`脚本文件中可用。以下是它的内容，它逐步汇总了我们刚刚做的所有内容：

```py
import requests
from bs4 import BeautifulSoup

def get_upcoming_events(url):
    req = requests.get(url)

    soup = BeautifulSoup(req.text, 'lxml')

    events = soup.find('ul', {'class': 'list-recent-events'}).findAll('li')

    for event in events:
        event_details = dict()
        event_details['name'] = event.find('h3').find("a").text
        event_details['location'] = event.find('span', {'class', 'event-location'}).text
        event_details['time'] = event.find('time').text
        print(event_details)

get_upcoming_events('https://www.python.org/events/python-events/')
```

你可以在终端中使用以下命令运行它：

```py
$ python 01_events_with_requests.py
{'name': 'PyCascades 2018', 'location': 'Granville Island Stage, 1585 Johnston St, Vancouver, BC V6H 3R9, Canada', 'time': '22 Jan. – 24 Jan. 2018'}
{'name': 'PyCon Cameroon 2018', 'location': 'Limbe, Cameroon', 'time': '24 Jan. – 29 Jan. 2018'}
{'name': 'FOSDEM 2018', 'location': 'ULB Campus du Solbosch, Av. F. D. Roosevelt 50, 1050 Bruxelles, Belgium', 'time': '03 Feb. – 05 Feb. 2018'}
{'name': 'PyCon Pune 2018', 'location': 'Pune, India', 'time': '08 Feb. – 12 Feb. 2018'}
{'name': 'PyCon Colombia 2018', 'location': 'Medellin, Colombia', 'time': '09 Feb. – 12 Feb. 2018'}
{'name': 'PyTennessee 2018', 'location': 'Nashville, TN, USA', 'time': '10 Feb. – 12 Feb. 2018'}
```

# 它的工作原理...

我们将在下一章节详细介绍Requests和Beautiful Soup，但现在让我们总结一下关于它的一些关键点。关于Requests的一些重要点：

+   Requests用于执行HTTP请求。我们用它来对事件页面的URL进行GET请求。

+   Requests对象保存了请求的结果。不仅包括页面内容，还有很多其他关于结果的项目，比如HTTP状态码和头部信息。

+   Requests仅用于获取页面，不进行解析。

我们使用Beautiful Soup来解析HTML和在HTML中查找内容。

要理解这是如何工作的，页面的内容具有以下HTML来开始“即将到来的事件”部分：

![](assets/9c3b8d5a-57e7-4cab-b868-b2362f805cc8.png)

我们利用Beautiful Soup的强大功能：

+   找到代表该部分的`<ul>`元素，通过查找具有值为`list-recent-events`的`class`属性的`<ul>`来找到。

+   从该对象中，我们找到所有`<li>`元素。

每个`<li>`标签代表一个不同的事件。我们遍历每一个，从子HTML标签中找到事件数据，制作一个字典：

+   名称从`<h3>`标签的子标签`<a>`中提取

+   位置是具有`event-location`类的`<span>`的文本内容

+   时间是从`<time>`标签的`datetime`属性中提取的。

# 使用urllib3和Beautiful Soup爬取Python.org

在这个配方中，我们将使用requests替换为另一个库`urllib3`。这是另一个常见的用于从URL检索数据以及处理URL的各个部分和处理各种编码的库。

# 准备工作...

这个配方需要安装`urllib3`。所以用`pip`安装它：

```py
$ pip install urllib3
Collecting urllib3
 Using cached urllib3-1.22-py2.py3-none-any.whl
Installing collected packages: urllib3
Successfully installed urllib3-1.22
```

# 如何做...

该代码在`01/02_events_with_urllib3.py`中实现。代码如下：

```py
import urllib3
from bs4 import BeautifulSoup

def get_upcoming_events(url):
    req = urllib3.PoolManager()
    res = req.request('GET', url)

    soup = BeautifulSoup(res.data, 'html.parser')

    events = soup.find('ul', {'class': 'list-recent-events'}).findAll('li')

    for event in events:
        event_details = dict()
        event_details['name'] = event.find('h3').find("a").text
        event_details['location'] = event.find('span', {'class', 'event-location'}).text
        event_details['time'] = event.find('time').text
        print(event_details)

get_upcoming_events('https://www.python.org/events/python-events/')
```

使用Python解释器运行它。你将得到与前一个配方相同的输出。

# 它的工作原理

这个配方唯一的区别是我们如何获取资源：

```py
req = urllib3.PoolManager()
res = req.request('GET', url)
```

与`Requests`不同，`urllib3`不会自动应用头部编码。前面示例中代码片段能够工作的原因是因为BS4能够很好地处理编码。但你应该记住编码是爬取的一个重要部分。如果你决定使用自己的框架或其他库，请确保编码得到很好的处理。

# 还有更多...

Requests和urllib3在功能方面非常相似。一般建议在进行HTTP请求时使用Requests。以下代码示例说明了一些高级功能：

```py
import requests

# builds on top of urllib3's connection pooling
# session reuses the same TCP connection if 
# requests are made to the same host
# see https://en.wikipedia.org/wiki/HTTP_persistent_connection for details
session = requests.Session()

# You may pass in custom cookie
r = session.get('http://httpbin.org/get', cookies={'my-cookie': 'browser'})
print(r.text)
# '{"cookies": {"my-cookie": "test cookie"}}'

# Streaming is another nifty feature
# From http://docs.python-requests.org/en/master/user/advanced/#streaming-requests
# copyright belongs to reques.org
r = requests.get('http://httpbin.org/stream/20', stream=True) 
```

```py
for line in r.iter_lines():
  # filter out keep-alive new lines
  if line:
     decoded_line = line.decode('utf-8')
     print(json.loads(decoded_line))
```

# 使用Scrapy爬取Python.org

**Scrapy**是一个非常流行的开源Python爬虫框架，用于提取数据。它最初是为了爬取而设计的，但它也发展成了一个强大的网络爬虫解决方案。

在我们之前的配方中，我们使用Requests和urllib2来获取数据，使用Beautiful Soup来提取数据。Scrapy提供了所有这些功能以及许多其他内置模块和扩展。在使用Python进行爬取时，这也是我们的首选工具。

Scrapy提供了一些强大的功能值得一提：

+   内置扩展，用于进行HTTP请求和处理压缩、认证、缓存、操作用户代理和HTTP头部

+   内置支持使用选择器语言（如CSS和XPath）选择和提取数据，以及支持利用正则表达式选择内容和链接

+   编码支持以处理语言和非标准编码声明

+   灵活的API，可以重用和编写自定义中间件和管道，提供了一种干净简单的方式来执行任务，比如自动下载资源（例如图片或媒体）并将数据存储在文件系统、S3、数据库等中

# 准备工作...

有几种方法可以使用Scrapy创建一个爬虫。一种是编程模式，我们在代码中创建爬虫和爬虫。还可以从模板或生成器配置一个Scrapy项目，然后使用`scrapy`命令从命令行运行爬虫。本书将遵循编程模式，因为它可以更有效地将代码放在一个文件中。这将有助于我们在使用Scrapy时组合特定的、有针对性的配方。

这并不一定是比使用命令行执行Scrapy爬虫更好的方式，只是这本书的设计决定。最终，这本书不是关于Scrapy的（有其他专门讲Scrapy的书），而是更多地阐述了在爬取时可能需要做的各种事情，以及在云端创建一个功能齐全的爬虫服务。

# 如何做...

这个配方的脚本是`01/03_events_with_scrapy.py`。以下是代码：

```py
import scrapy
from scrapy.crawler import CrawlerProcess

class PythonEventsSpider(scrapy.Spider):
    name = 'pythoneventsspider'    start_urls = ['https://www.python.org/events/python-events/',]
    found_events = []

    def parse(self, response):
        for event in response.xpath('//ul[contains(@class, "list-recent-events")]/li'):
            event_details = dict()
            event_details['name'] = event.xpath('h3[@class="event-title"]/a/text()').extract_first()
            event_details['location'] = event.xpath('p/span[@class="event-location"]/text()').extract_first()
            event_details['time'] = event.xpath('p/time/text()').extract_first()
            self.found_events.append(event_details)

if __name__ == "__main__":
    process = CrawlerProcess({ 'LOG_LEVEL': 'ERROR'})
    process.crawl(PythonEventsSpider)
    spider = next(iter(process.crawlers)).spider
    process.start()

    for event in spider.found_events: print(event)
```

以下是运行脚本并显示输出的过程：

```py
~ $ python 03_events_with_scrapy.py
{'name': 'PyCascades 2018', 'location': 'Granville Island Stage, 1585 Johnston St, Vancouver, BC V6H 3R9, Canada', 'time': '22 Jan. – 24 Jan. '}
{'name': 'PyCon Cameroon 2018', 'location': 'Limbe, Cameroon', 'time': '24 Jan. – 29 Jan. '}
{'name': 'FOSDEM 2018', 'location': 'ULB Campus du Solbosch, Av. F. D. Roosevelt 50, 1050 Bruxelles, Belgium', 'time': '03 Feb. – 05 Feb. '}
{'name': 'PyCon Pune 2018', 'location': 'Pune, India', 'time': '08 Feb. – 12 Feb. '}
{'name': 'PyCon Colombia 2018', 'location': 'Medellin, Colombia', 'time': '09 Feb. – 12 Feb. '}
{'name': 'PyTennessee 2018', 'location': 'Nashville, TN, USA', 'time': '10 Feb. – 12 Feb. '}
{'name': 'PyCon Pakistan', 'location': 'Lahore, Pakistan', 'time': '16 Dec. – 17 Dec. '}
{'name': 'PyCon Indonesia 2017', 'location': 'Surabaya, Indonesia', 'time': '09 Dec. – 10 Dec. '}
```

使用另一个工具得到相同的结果。让我们快速回顾一下它是如何工作的。

# 它是如何工作的

我们将在后面的章节中详细介绍Scrapy，但让我们快速浏览一下这段代码，以了解它是如何完成这个爬取的。Scrapy中的一切都围绕着创建**spider**。蜘蛛根据我们提供的规则在互联网上爬行。这个蜘蛛只处理一个单独的页面，所以它并不是一个真正的蜘蛛。但它展示了我们将在后面的Scrapy示例中使用的模式。

爬虫是通过一个类定义创建的，该类继承自Scrapy爬虫类之一。我们的类继承自`scrapy.Spider`类。

```py
class PythonEventsSpider(scrapy.Spider):
    name = 'pythoneventsspider'    start_urls = ['https://www.python.org/events/python-events/',]
```

每个爬虫都有一个`name`，还有一个或多个`start_urls`，告诉它从哪里开始爬行。

这个爬虫有一个字段来存储我们找到的所有事件：

```py
    found_events = []
```

然后，爬虫有一个名为parse的方法，它将被调用来处理爬虫收集到的每个页面。

```py
def parse(self, response):
        for event in response.xpath('//ul[contains(@class, "list-recent-events")]/li'):
            event_details = dict()
            event_details['name'] = event.xpath('h3[@class="event-title"]/a/text()').extract_first()
            event_details['location'] = event.xpath('p/span[@class="event-location"]/text()').extract_first()
            event_details['time'] = event.xpath('p/time/text()').extract_first()
            self.found_events.append(event_details)
```

这个方法的实现使用了XPath选择器来从页面中获取事件（XPath是Scrapy中导航HTML的内置方法）。它构建了`event_details`字典对象，类似于其他示例，然后将其添加到`found_events`列表中。

剩下的代码执行了Scrapy爬虫的编程执行。

```py
    process = CrawlerProcess({ 'LOG_LEVEL': 'ERROR'})
    process.crawl(PythonEventsSpider)
    spider = next(iter(process.crawlers)).spider
    process.start()
```

它从创建一个CrawlerProcess开始，该过程执行实际的爬行和许多其他任务。我们传递了一个ERROR的LOG_LEVEL来防止大量的Scrapy输出。将其更改为DEBUG并重新运行以查看差异。

接下来，我们告诉爬虫进程使用我们的Spider实现。我们从爬虫中获取实际的蜘蛛对象，这样当爬取完成时我们就可以获取项目。然后我们通过调用`process.start()`来启动整个过程。

当爬取完成后，我们可以迭代并打印出找到的项目。

```py
    for event in spider.found_events: print(event)
```

这个例子并没有涉及到Scrapy的任何强大功能。我们将在本书的后面更深入地了解一些更高级的功能。

# 使用Selenium和PhantomJS来爬取Python.org

这个配方将介绍Selenium和PhantomJS，这两个框架与之前的配方中的框架非常不同。实际上，Selenium和PhantomJS经常用于功能/验收测试。我们想展示这些工具，因为它们从爬取的角度提供了独特的好处。我们将在本书的后面看到一些，比如填写表单、按按钮和等待动态JavaScript被下载和执行的能力。

Selenium本身是一个与编程语言无关的框架。它提供了许多编程语言绑定，如Python、Java、C#和PHP（等等）。该框架还提供了许多专注于测试的组件。其中三个常用的组件是：

+   用于录制和重放测试的IDE

+   Webdriver实际上启动了一个Web浏览器（如Firefox、Chrome或Internet Explorer），通过发送命令并将结果发送到所选的浏览器来运行脚本

+   网格服务器在远程服务器上执行带有Web浏览器的测试。它可以并行运行多个测试用例。

# 准备工作

首先，我们需要安装Selenium。我们可以使用我们信赖的`pip`来完成这个过程：

```py
~ $ pip install selenium
Collecting selenium
 Downloading selenium-3.8.1-py2.py3-none-any.whl (942kB)
 100% |████████████████████████████████| 952kB 236kB/s
Installing collected packages: selenium
Successfully installed selenium-3.8.1
```

这将安装Python的Selenium客户端驱动程序（语言绑定）。如果你将来想要了解更多信息，可以在[https://github.com/SeleniumHQ/selenium/blob/master/py/docs/source/index.rst](https://github.com/SeleniumHQ/selenium/blob/master/py/docs/source/index.rst)找到更多信息。

对于这个配方，我们还需要在目录中有Firefox的驱动程序（名为`geckodriver`）。这个文件是特定于操作系统的。我已经在文件夹中包含了Mac的文件。要获取其他版本，请访问[https://github.com/mozilla/geckodriver/releases](https://github.com/mozilla/geckodriver/releases)。

然而，当运行这个示例时，你可能会遇到以下错误：

```py
FileNotFoundError: [Errno 2] No such file or directory: 'geckodriver'
```

如果你这样做了，将geckodriver文件放在系统的PATH中，或者将`01`文件夹添加到你的路径中。哦，你还需要安装Firefox。

最后，需要安装PhantomJS。你可以在[http://phantomjs.org/](http://phantomjs.org/)下载并找到安装说明。

# 如何做...

这个配方的脚本是`01/04_events_with_selenium.py`。

1.  以下是代码：

```py
from selenium import webdriver

def get_upcoming_events(url):
    driver = webdriver.Firefox()
    driver.get(url)

    events = driver.find_elements_by_xpath('//ul[contains(@class, "list-recent-events")]/li')

    for event in events:
        event_details = dict()
        event_details['name'] = event.find_element_by_xpath('h3[@class="event-title"]/a').text
        event_details['location'] = event.find_element_by_xpath('p/span[@class="event-location"]').text
        event_details['time'] = event.find_element_by_xpath('p/time').text
        print(event_details)

    driver.close()

get_upcoming_events('https://www.python.org/events/python-events/')
```

1.  然后用Python运行脚本。你会看到熟悉的输出：

```py
~ $ python 04_events_with_selenium.py
{'name': 'PyCascades 2018', 'location': 'Granville Island Stage, 1585 Johnston St, Vancouver, BC V6H 3R9, Canada', 'time': '22 Jan. – 24 Jan.'}
{'name': 'PyCon Cameroon 2018', 'location': 'Limbe, Cameroon', 'time': '24 Jan. – 29 Jan.'}
{'name': 'FOSDEM 2018', 'location': 'ULB Campus du Solbosch, Av. F. D. Roosevelt 50, 1050 Bruxelles, Belgium', 'time': '03 Feb. – 05 Feb.'}
{'name': 'PyCon Pune 2018', 'location': 'Pune, India', 'time': '08 Feb. – 12 Feb.'}
{'name': 'PyCon Colombia 2018', 'location': 'Medellin, Colombia', 'time': '09 Feb. – 12 Feb.'}
{'name': 'PyTennessee 2018', 'location': 'Nashville, TN, USA', 'time': '10 Feb. – 12 Feb.'}
```

在这个过程中，Firefox将弹出并打开页面。我们重用了之前的配方并采用了Selenium。

![](assets/05feca6d-bf9f-4938-9cb7-1392310dc374.png)Firefox弹出的窗口

# 它的工作原理

这个配方的主要区别在于以下代码：

```py
driver = webdriver.Firefox()
driver.get(url)
```

这个脚本获取了Firefox驱动程序，并使用它来获取指定URL的内容。这是通过启动Firefox并自动化它去到页面，然后Firefox将页面内容返回给我们的应用程序。这就是为什么Firefox弹出的原因。另一个区别是，为了找到东西，我们需要调用`find_element_by_xpath`来搜索结果的HTML。

# 还有更多...

在许多方面，PhantomJS与Selenium非常相似。它对各种Web标准有快速和本地支持，具有DOM处理、CSS选择器、JSON、Canvas和SVG等功能。它经常用于Web测试、页面自动化、屏幕捕捉和网络监控。

Selenium和PhantomJS之间有一个关键区别：PhantomJS是**无头**的，使用WebKit。正如我们所看到的，Selenium打开并自动化浏览器。如果我们处于一个连续集成或测试环境中，浏览器没有安装，我们也不希望打开成千上万个浏览器窗口或标签，那么这并不是很好。无头浏览器使得这一切更快更高效。

PhantomJS的示例在`01/05_events_with_phantomjs.py`文件中。只有一行代码需要更改：

```py
driver = webdriver.PhantomJS('phantomjs')
```

运行脚本会产生与Selenium/Firefox示例类似的输出，但不会弹出浏览器，而且完成时间更短。
