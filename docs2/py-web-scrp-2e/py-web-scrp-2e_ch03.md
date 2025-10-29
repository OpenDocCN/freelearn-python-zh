# 缓存下载

在上一章中，我们学习了如何从爬取的网页中抓取数据并将结果保存到 CSV 文件中。如果我们现在想抓取额外的字段，比如标志 URL，为了抓取额外的字段，我们需要再次下载整个网站。这对我们的示例网站来说不是一个重大的障碍；然而，其他网站可能有数百万个网页，这可能需要几周的时间才能重新抓取。爬虫避免这些问题的方法之一是从一开始就缓存爬取的网页，这样它们就只需要下载一次。

在本章中，我们将介绍几种使用我们的网络爬虫来实现这一功能的方法。

本章将涵盖以下主题：

+   何时使用缓存

+   为链接爬虫添加缓存支持

+   测试缓存

+   使用 requests - 缓存

+   Redis 缓存实现

# 何时使用缓存？

要缓存还是不要缓存？这是一个许多程序员、数据科学家和网络爬虫需要回答的问题。在本章中，我们将向您展示如何为您的网络爬虫使用缓存；但你应该使用缓存吗？

如果你需要进行大规模的爬取，这可能会因为错误或异常而中断，缓存可以帮助你不必重新爬取你可能已经覆盖的所有页面。缓存还可以通过允许你在离线状态下访问这些页面（用于你的数据分析或开发目的）来帮助你。

然而，如果你将获取网站最新和最准确的信息作为最高优先级，那么缓存可能就不合适了。此外，如果你不打算进行大量或重复的爬取，你可能只想每次都抓取页面。

在实施之前，你可能想要概述你正在抓取的页面更改的频率或你应该多久抓取一次新页面并清除缓存；但首先，让我们学习如何使用缓存！

# 为链接爬虫添加缓存支持

要支持缓存，需要在第一章中开发的`download`函数中进行修改，以在下载 URL 之前检查缓存。我们还需要将节流操作移入此函数中，并且只在下载时进行节流，而不是在从缓存中加载时。为了避免每次下载都需要传递各种参数，我们将利用这个机会将`download`函数重构为类，以便可以在构造函数中设置参数并多次重用。以下是支持此功能的更新实现：

```py
from chp1.throttle import Throttle
from random import choice
import requests

class Downloader:
    def __init__(self, delay=5, user_agent='wswp', proxies=None, cache={}):
        self.throttle = Throttle(delay)
        self.user_agent = user_agent
        self.proxies = proxies
        self.num_retries = None  # we will set this per request
        self.cache = cache

    def __call__(self, url, num_retries=2):
        self.num_retries = num_retries
        try:
             result = self.cache[url]
             print('Loaded from cache:', url)
        except KeyError:
             result = None
        if result and self.num_retries and 500 <= result['code'] < 600:
            # server error so ignore result from cache
            # and re-download
            result = None
        if result is None:
             # result was not loaded from cache
             # so still need to download
             self.throttle.wait(url)
             proxies = choice(self.proxies) if self.proxies else None
             headers = {'User-Agent': self.user_agent}
             result = self.download(url, headers, proxies)
             if self.cache:
                 # save result to cache
                 self.cache[url] = result
        return result['html']

    def download(self, url, headers, proxies, num_retries): 
        ... 
        return {'html': html, 'code': resp.status_code } 

```

下载类的完整源代码可在[`github.com/kjam/wswp/blob/master/code/chp3/downloader.py`](https://github.com/kjam/wswp/blob/master/code/chp3/downloader.py)找到。

在前面代码中使用的`Download`类的有趣部分在于`__call__`特殊方法，在下载之前检查缓存。该方法首先检查此 URL 是否之前已放入缓存。默认情况下，缓存是一个 Python 字典。如果 URL 已缓存，它检查在之前的下载中是否遇到了服务器错误。最后，如果没有遇到服务器错误，可以使用缓存的缓存结果。如果这些检查中的任何一个失败，则需要像往常一样下载 URL，并将结果添加到缓存中。

这个类的`download`方法几乎与之前的`download`函数相同，但现在它返回 HTTP 状态码，因此错误代码可以存储在缓存中。此外，它不再调用自身并测试`num_retries`，而是首先减少`self.num_retries`，然后在还有重试剩余的情况下递归地使用`self.download`。如果你只想进行简单的下载而不需要节流或缓存，可以使用这个方法代替`__call__`。

通过调用`result = cache[url]`从`cache`中加载和`cache[url] = result`将结果保存到`cache`，我们使用 Python 内置字典数据类型的方便接口。为了支持此接口，我们的`cache`类需要定义`__getitem__()`和`__setitem__()`特殊类方法。

链接爬虫也需要稍作更新以支持缓存，通过添加`cache`参数、移除节流并替换`download`函数为新类，如下所示：

```py
def link_crawler(..., num_retries=2, cache={}): 
    crawl_queue = [seed_url] 
    seen = {seed_url: 0}  
    rp = get_robots(seed_url) 
    D = Downloader(delay=delay, user_agent=user_agent, proxies=proxies, cache=cache) 

    while crawl_queue: 
        url = crawl_queue.pop() 
        # check url passes robots.txt restrictions 
        if rp.can_fetch(user_agent, url): 
            depth = seen.get(url, 0)
            if depth == max_depth: 
                continue
            html = D(url, num_retries=num_retries)
            if not html:
                continue
            ...

```

你会注意到`num_retries`现在与我们的调用相关联。这允许我们根据每个 URL 利用请求重试次数。如果我们简单地使用相同的重试次数而不重置`self.num_retries`值，一旦遇到一个页面的`500`错误，我们将耗尽重试次数。

你可以在书籍仓库中再次查看完整代码（[`github.com/kjam/wswp/blob/master/code/chp3/advanced_link_crawler.py`](https://github.com/kjam/wswp/blob/master/code/chp3/advanced_link_crawler.py)）。现在，我们的网络爬取基础设施已经准备就绪，我们可以开始构建实际的缓存。

# 磁盘缓存

为了缓存下载，我们首先尝试明显的解决方案，并将网页保存到文件系统中。为此，我们需要一种将 URL 映射到安全的跨平台文件名的方法。以下表格列出了某些流行文件系统的限制：

| **操作系统** | **文件系统** | **无效文件名字符** | **最大文件名长度** |
| --- | --- | --- | --- |
| Linux | Ext3/Ext4 | / and \0 | 255 bytes |
| OS X | HFS Plus | : and \0 | 255 UTF-16 code units |
| Windows | NTFS | \, /, ?, :, *, ", >, <, and &#124; | 255 characters |

为了在这些文件系统中保持文件路径的安全，它需要限制为数字、字母和基本标点符号，并且应该将所有其他字符替换为下划线，如下所示：

```py
>>> import re 
>>> url = 'http://example.webscraping.com/default/view/Australia-1' 
>>> re.sub('[^/0-9a-zA-Z\-.,;_ ]', '_', url) 
'http_//example.webscraping.com/default/view/Australia-1' 

```

此外，文件名和父目录需要限制为 255 个字符（如下所示），以满足前面表格中描述的长度限制：

```py
>>> filename = re.sub('[^/0-9a-zA-Z\-.,;_ ]', '_', url)
>>> filename = '/'.join(segment[:255] for segment in filename.split('/'))
>>> print(filename)
'http_//example.webscraping.com/default/view/Australia-1' 

```

在这里，我们的 URL 的任何部分都没有超过 255 个字符；因此，我们的文件路径没有改变。还有一个需要考虑的边缘情况，即 URL 路径以斜杠（`/`）结尾，斜杠后面的空字符串将是一个无效的文件名。然而，删除此斜杠以使用父目录作为文件名将阻止保存其他 URL。考虑以下 URL：

+   http://example.webscraping.com/index/

+   http://example.webscraping.com/index/1

如果你需要保存这些，索引需要是一个目录，以保存具有文件名 1 的子页面。我们的磁盘缓存将使用的方法是在 URL 路径以斜杠结尾时将`index.html`附加到文件名。当 URL 路径为空时也适用。为了解析 URL，我们将使用`urlsplit`函数，该函数将 URL 分割成其组成部分：

```py
>>> from urllib.parse import urlsplit 
>>> components = urlsplit('http://example.webscraping.com/index/') 
>>> print(components) 
SplitResult(scheme='http', netloc='example.webscraping.com', path='/index/', query='', fragment='') 
>>> print(components.path) 
'/index/' 

```

此函数提供了一个方便的接口来解析和操作 URL。以下是一个使用此模块为这种边缘情况添加`index.html`的示例：

```py
>>> path = components.path 
>>> if not path: 
>>>     path = '/index.html' 
>>> elif path.endswith('/'): 
>>>     path += 'index.html' 
>>> filename = components.netloc + path + components.query 
>>> filename 
'example.webscraping.com/index/index.html' 

```

根据你正在抓取的网站，你可能需要修改这种边缘情况的处理。例如，一些网站会由于 Web 服务器期望 URL 的发送方式，在每个 URL 后附加`/`。对于这些网站，你可能只需简单地为每个 URL 删除尾随的反斜杠。再次评估并更新你的网络爬虫代码，以最好地适应你打算抓取的网站。

# 实现 DiskCache

在上一节中，我们讨论了在构建基于磁盘的缓存时需要考虑的文件系统限制，即可以使用的字符限制、文件名长度限制，以及确保文件和目录不在同一位置创建。将此代码与将 URL 映射到文件名的逻辑相结合，将形成磁盘缓存的主要部分。以下是`DiskCache`类的初始实现：

```py
import os 
import re 
from urllib.parse import urlsplit 

class DiskCache: 
    def __init__(self, cache_dir='cache', max_len=255): 
        self.cache_dir = cache_dir 
        self.max_len = max_len 

    def url_to_path(self, url): 
        """ Return file system path string for given URL""" 
        components = urlsplit(url) 
        # append index.html to empty paths 
        path = components.path 
        if not path: 
            path = '/index.html' 
        elif path.endswith('/'): 
            path += 'index.html' 
        filename = components.netloc + path + components.query 
        # replace invalid characters 
        filename = re.sub('[^/0-9a-zA-Z\-.,;_ ]', '_', filename) 
        # restrict maximum number of characters 
        filename = '/'.join(seg[:self.max_len] for seg in filename.split('/')) 
        return os.path.join(self.cache_dir, filename) 

```

前面代码中显示的类构造函数接受一个参数来设置缓存的位置，然后`url_to_path`方法应用了之前讨论的文件名限制。现在我们只需要提供使用此文件名加载数据和保存数据的方法。

这里是这些缺失方法的实现：

```py
import json 
class DiskCache: 
    ... 
    def __getitem__(self, url): 
        """Load data from disk for given URL""" 
        path = self.url_to_path(url) 
        if os.path.exists(path): 
            return json.load(path)
        else: 
            # URL has not yet been cached 
            raise KeyError(url + ' does not exist') 

    def __setitem__(self, url, result): 
        """Save data to disk for given url""" 
        path = self.url_to_path(url) 
        folder = os.path.dirname(path) 
        if not os.path.exists(folder): 
            os.makedirs(folder) 
        json.dump(result, path) 

```

在`__setitem__()`中，使用`url_to_path()`将 URL 映射到安全的文件名，然后根据需要创建父目录。使用`json`模块序列化 Python 对象，然后将其保存到磁盘。在`__getitem__()`中，也将 URL 映射到安全的文件名。如果文件名存在，则使用`json`加载内容以恢复原始数据类型。如果文件名不存在（即，对于此 URL 没有缓存中的数据），则引发`KeyError`异常。

# 测试缓存

现在我们准备通过将`DiskCache`传递给`cache`关键字参数来在我们的爬虫中尝试使用它。这个类的源代码可在[`github.com/kjam/wswp/blob/master/code/chp3/diskcache.py`](https://github.com/kjam/wswp/blob/master/code/chp3/diskcache.py)找到，并且可以在任何 Python 解释器中测试缓存。

IPython 附带了一套出色的工具，用于编写和解释 Python，特别是使用[IPython 魔法命令](https://ipython.org/ipython-doc/3/interactive/magics.html)进行 Python 调试。您可以使用 pip 或 conda 安装 IPython（`pip install ipython`）。

在这里，我们使用[IPython](https://ipython.org/)来帮助我们计时以测试其性能：

```py
In [1]: from chp3.diskcache import DiskCache

In [2]: from chp3.advanced_link_crawler import link_crawler

In [3]: %time link_crawler('http://example.webscraping.com/', '/(index|view)', cache=DiskCache())
Downloading: http://example.webscraping.com/
Downloading: http://example.webscraping.com/index/1
Downloading: http://example.webscraping.com/index/2
...
Downloading: http://example.webscraping.com/view/Afghanistan-1
CPU times: user 300 ms, sys: 16 ms, total: 316 ms
Wall time: 1min 44s

```

第一次运行此命令时，缓存为空，因此所有网页都会正常下载。然而，当我们第二次运行此脚本时，页面将从缓存中加载，因此爬取应该会更快完成，如下所示：

```py
In [4]: %time link_crawler('http://example.webscraping.com/', '/(index|view)', cache=DiskCache())
Loaded from cache: http://example.webscraping.com/
Loaded from cache: http://example.webscraping.com/index/1
Loaded from cache: http://example.webscraping.com/index/2
...
Loaded from cache: http://example.webscraping.com/view/Afghanistan-1
CPU times: user 20 ms, sys: 0 ns, total: 20 ms
Wall time: 1.1 s

```

如预期的那样，这次爬取完成得更快。在我电脑上使用空缓存下载时，爬虫花费了一分钟多；第二次，使用完整缓存，只需 1.1 秒（大约快 95 倍！）。

在您的电脑上的确切时间将取决于您硬件的速度和互联网连接速度。然而，磁盘缓存无疑会比通过 HTTP 下载更快。

# 节省磁盘空间

为了最小化我们缓存所需的磁盘空间，我们可以压缩下载的 HTML 文件。通过在保存到磁盘之前使用`zlib`压缩序列化的字符串来实现这一点很简单。使用我们当前的实现方式的好处是文件可读性高。我可以查看任何缓存页面，并看到以 JSON 形式显示的字典。如果需要，我还可以重用这些文件，并将它们移动到不同的操作系统上，用于非 Python 代码。添加压缩会使这些文件仅通过打开它们就不再可读，并且如果我们使用其他编码语言下载的页面，可能会引入一些编码问题。为了允许压缩可以开启或关闭，我们可以将其添加到构造函数中，与文件编码一起，我们将默认设置为 UTF-8：

```py
class DiskCache:
    def __init__(self, cache_dir='../data/cache', max_len=255, compress=True, 
                 encoding='utf-8'):
        ...
        self.compress = compress
        self.encoding = encoding

```

然后，应该更新`__getitem__`和`__setitem__`方法：

```py
# in __getitem__ method for DiskCache class
mode = ('rb' if self.compress else 'r')
with open(path, mode) as fp:
    if self.compress:
        data = zlib.decompress(fp.read()).decode(self.encoding)
        return json.loads(data)
    return json.load(fp)

# in __setitem__ method for DiskCache class
mode = ('wb' if self.compress else 'w')
with open(path, mode) as fp:
    if self.compress:
        data = bytes(json.dumps(result), self.encoding)
        fp.write(zlib.compress(data))
 else:
 json.dump(result, fp)

```

通过添加压缩每个网页的功能，缓存从 416 KB 减少到 156 KB，在我的电脑上爬取缓存的示例网站需要 260 毫秒。

根据您的操作系统和 Python 安装，未压缩的缓存等待时间可能会稍微长一些（我的实际上更短）。根据您对约束条件的优先级（速度与内存、调试的简便性等）进行有信息和量化的决策，决定是否为您的爬虫使用压缩。

您可以在书籍的代码仓库中看到更新的磁盘缓存代码（[`github.com/kjam/wswp/blob/master/code/chp3/diskcache.py`](https://github.com/kjam/wswp/blob/master/code/chp3/diskcache.py)）。

# 过期无效数据

我们当前版本的磁盘缓存会将值保存到磁盘上的一个键，并在将来请求此键时返回它。由于在线内容的变化，这种功能在缓存网页时可能不是理想的，因此我们缓存中的数据会变得过时。在本节中，我们将为我们的缓存数据添加一个过期时间，这样爬虫就知道何时下载网页的新副本。为了支持存储每个网页被缓存的时间戳，这是直截了当的。

这里是这个实现的示例：

```py
from datetime import datetime, timedelta 

class DiskCache:
     def __init__(..., expires=timedelta(days=30)):
         ...
         self.expires = expires

## in __getitem___ for DiskCache class
with open(path, mode) as fp:
    if self.compress:
        data = zlib.decompress(fp.read()).decode(self.encoding)
        data = json.loads(data)
    else:
        data = json.load(fp)
    exp_date = data.get('expires')
    if exp_date and datetime.strptime(exp_date,
                                      '%Y-%m-%dT%H:%M:%S') <= datetime.utcnow():
        print('Cache expired!', exp_date)
        raise KeyError(url + ' has expired.')
    return data

## in __setitem___ for DiskCache class
result['expires'] = (datetime.utcnow() + self.expires).isoformat(timespec='seconds')

```

在构造函数中，默认的过期时间设置为 30 天，使用`timedelta`对象。然后，`__set__`方法将过期时间戳作为键存储在`result`字典中，而`__get__`方法将当前 UTC 时间与过期时间进行比较。为了测试这个过期时间，我们可以尝试一个短暂的超时时间，例如 5 秒，如下所示：

```py
 >>> cache = DiskCache(expires=timedelta(seconds=5)) 
 >>> url = 'http://example.webscraping.com' 
 >>> result = {'html': '...'} 
 >>> cache[url] = result 
 >>> cache[url] 
 {'html': '...'} 
 >>> import time; time.sleep(5) 
 >>> cache[url] 
 Traceback (most recent call last): 
 ... 
 KeyError: 'http://example.webscraping.com has expired' 

```

如预期，缓存的结果最初是可用的，然后，在睡眠五秒后，调用相同的键会引发一个`KeyError`，以显示这个缓存的下载已经过期。

# DiskCache 的缺点

我们的基于磁盘的缓存系统相对简单易实现，不需要安装额外的模块，并且结果可以在我们的文件管理器中查看。然而，它有一个缺点，即依赖于本地文件系统的限制。在本章的早期部分，我们应用了各种限制来将 URL 映射到安全的文件名，但这个系统的不幸后果是，一些 URL 会映射到相同的文件名。例如，替换以下 URL 中的不受支持的字符将使它们都映射到相同的文件名：

+   [`example.com/?a+b`](http://example.com/?a+b)

+   [`example.com/?a*b`](http://example.com/?a*b)

+   [`example.com/?a=b`](http://example.com/?a=b)

+   [`example.com/?a!b`](http://example.com/?a!b)

这意味着，如果这些 URL 中的任何一个被缓存，它们看起来就像其他三个 URL 也被缓存了一样，因为它们映射到相同的文件名。或者，如果一些长 URL 在 255^(th)个字符之后才有所不同，缩短的版本也会映射到相同的文件名。这是一个特别重要的问题，因为 URL 的最大长度没有定义的限制。然而，在实践中，超过 2,000 个字符的 URL 很少见，而且旧版本的 Internet Explorer 不支持超过 2,083 个字符。

避免这些限制的一个潜在解决方案是取 URL 的哈希值，并使用哈希值作为文件名。这可能是一个改进；然而，我们最终会面临许多文件系统都有的一个更大的问题，即每个卷和每个目录允许的文件数量限制。如果在这个缓存中使用 FAT32 文件系统，每个目录允许的文件最大数量仅为 65,535。通过将缓存分散到多个目录中可以避免这种限制；然而，文件系统也可能限制文件的总数。我的当前`ext4`分区支持略超过 3100 万文件，而一个大型网站可能有超过 1 亿个网页。不幸的是，`DiskCache`方法有太多的限制，不能被普遍使用。我们真正需要的是将多个缓存的网页合并成一个文件，并使用`B+`树或类似的数据结构进行索引。我们不会实现自己的，而是在下一节中使用现有的键值存储。

# 键值存储缓存

为了避免对基于磁盘的缓存的预期限制，我们现在将在现有的键值存储系统之上构建我们的缓存。在爬取时，我们可能需要缓存大量数据，并且不需要任何复杂的连接操作，因此我们将使用高可用性的键值存储，这比传统的数据库或大多数 NoSQL 数据库更容易扩展。具体来说，我们的缓存将使用 Redis，这是一个非常流行的键值存储。

# 什么是键值存储？

**键值存储**与 Python 字典非常相似，因为存储中的每个元素都有一个键和一个值。在设计`DiskCache`时，键值模型非常适合这个问题。实际上，Redis 代表远程字典服务器。Redis 首次发布于 2009 年，其 API 支持多种不同语言的客户端（包括 Python）。它与一些更简单的键值存储（如 memcache）不同，因为其值可以是几种不同的结构化数据类型。Redis 可以通过集群轻松扩展，并被大型公司（如 Twitter）用于大量缓存存储（例如，一个大约有 65TB 分配堆内存的 Twitter BTree [highscalability.com/blog/2014/9/8/how-twitter-uses-redis-to-scale-105tb-ram-39mm-qps-10000-ins.html](http://highscalability.com/blog/2014/9/8/how-twitter-uses-redis-to-scale-105tb-ram-39mm-qps-10000-ins.html)）。

对于您的抓取和爬取需求，可能会有一些情况需要为每个文档提供更多信息，或者需要能够根据文档中的数据进行搜索和选择。对于这些情况，我推荐使用基于文档的数据库，如 ElasticSearch 或 MongoDB。键值存储和基于文档的数据库都能够以比具有模式的传统 SQL 数据库（如 PostgreSQL 和 MySQL）更清晰、更简单的方式扩展并快速查询非关系型数据。

# 安装 Redis

可以按照 Redis 网站上的说明编译最新源代码来安装 Redis（[`redis.io/topics/quickstart`](https://redis.io/topics/quickstart)）。如果你正在运行 Windows，你需要使用 MSOpenTech 的项目（[`github.com/MSOpenTech/redis`](https://github.com/MSOpenTech/redis)）或者简单地通过虚拟机（使用 Vagrant）或 docker 实例安装 Redis。然后需要单独使用以下命令安装 Python 客户端：

```py
    pip install redis

```

要测试安装是否正常工作，请使用以下命令在本地（或虚拟机或容器）启动 Redis：

```py
    $ redis-server

```

你应该会看到一些带有版本号和 Redis 符号的文本。在文本的末尾，你会看到如下信息：

```py
1212:M 18 Feb 20:24:44.590 * The server is now ready to accept connections on port 6379

```

很可能，你的 Redis 服务器将使用相同的端口，这是默认端口（6379）。为了测试我们的 Python 客户端并连接到 Redis，我们可以使用 Python 解释器（在下面的代码中，我使用 IPython），如下所示：

```py
In [1]: import redis

In [2]: r = redis.StrictRedis(host='localhost', port=6379, db=0)

In [3]: r.set('test', 'answer')
Out[3]: True

In [4]: r.get('test')
Out[4]: b'answer'

```

在前面的代码中，我们能够轻松地连接到我们的 Redis 服务器，然后使用键`'test'`和值`'answer'`来`set`一条记录。我们能够使用`get`命令轻松检索该记录。

要查看如何将 Redis 设置为后台进程运行的更多选项，我建议使用官方的 Redis 快速入门（[`redis.io/topics/quickstart`](https://redis.io/topics/quickstart)）或使用你喜欢的搜索引擎查找特定操作系统的具体说明或安装说明。

# Redis 概述

这里有一个示例，说明如何在 Redis 中保存一些示例网站数据，然后加载它：

```py
In [5]: url = 'http://example.webscraping.com/view/United-Kingdom-239' 

In [6]: html = '...'

In [7]: results = {'html': html, 'code': 200}

In [8]: r.set(url, results)
Out[8]: True

In [9]: r.get(url)
Out[9]: b"{'html': '...', 'code': 200}"

```

我们可以通过`get`输出看到，即使我们插入了一个字典或字符串，我们也会从 Redis 存储中接收到`bytes`。我们可以像管理我们的`DiskCache`类一样管理这些序列化，通过使用`json`模块。

如果我们需要更新 URL 的内容会发生什么？

```py
In [10]: r.set(url, {'html': 'new html!', 'code': 200})
Out[10]: True

In [11]: r.get(url)
Out[11]: b"{'html': 'new html!', 'code': 200}"

```

从上面的输出中我们可以看到，Redis 中的`set`命令会简单地覆盖之前的值，这使得它非常适合像我们的网络爬虫这样的简单存储。对于我们的需求，我们只想为每个 URL 有一组内容，因此它与键值存储很好地映射。

让我们看看我们的存储中有什么，并清理我们不需要的内容：

```py
In [12]: r.keys()
Out[12]: [b'test', b'http://example.webscraping.com/view/United-Kingdom-239']

In [13]: r.delete('test')
Out[13]: 1

In [14]: r.keys()
Out[14]: [b'http://example.webscraping.com/view/United-Kingdom-239']

```

`keys`方法返回所有可用键的列表，`delete`方法允许我们传递一个（或多个）键并将它们从我们的存储中删除。我们还可以删除所有键：

```py
In [15]: r.flushdb()
Out[15]: True

In [16]: r.keys()
Out[16]: []

```

Redis 有许多更多的命令和用法，所以请随意阅读文档中的更多内容。目前，我们应该有创建带有 Redis 后端的缓存所需的所有内容，用于我们的网络爬虫。

Python Redis 客户端 [`github.com/andymccurdy/redis-py`](https://github.com/andymccurdy/redis-py) 提供了出色的文档和多个使用 Python 与 Redis 一起使用的案例（例如 PubSub 管道或作为大连接池）。官方 Redis 文档 [`redis.io/documentation`](https://redis.io/documentation) 列出了大量的教程、书籍、参考资料和用例；因此，如果您想了解更多关于如何扩展、安全性和部署 Redis 的信息，我建议从那里开始。如果您在云端或服务器上使用 Redis，别忘了为您的 Redis 实例实现安全性 ([`redis.io/topics/security`](https://redis.io/topics/security))！

# Redis 缓存实现

现在我们准备使用与早期 `DiskCache` 类相同的类接口在 Redis 上构建我们的缓存：

```py
import json
from datetime import timedelta 
from redis import StrictRedis

class RedisCache: 
    def __init__(self, client=None, expires=timedelta(days=30), encoding='utf-8'): 
        # if a client object is not passed then try 
        # connecting to redis at the default localhost port 
        self.client = StrictRedis(host='localhost', port=6379, db=0) 
            if client is None else client 
        self.expires = expires
        self.encoding = encoding

    def __getitem__(self, url): 
        """Load value from Redis for the given URL""" 
        record = self.client.get(url) 
        if record: 
            return json.loads(record.decode(self.encoding))
        else: 
            raise KeyError(url + ' does not exist') 

    def __setitem__(self, url, result): 
        """Save value in Redis for the given URL""" 
        data = bytes(json.dumps(result), self.encoding)
        self.client.setex(url, self.expires, data)

```

这里的 `__getitem__` 和 `__setitem__` 方法应该与上一节中关于如何在 Redis 中获取和设置键的讨论中提到的内容相似，只是我们使用 `json` 模块来控制序列化，并使用 `setex` 方法，这允许我们设置带有过期时间的键和值。`setex` 可以接受 `datetime.timedelta` 或秒数。这是一个方便的 Redis 功能，它将自动在指定秒数后删除记录。这意味着我们不需要像在 `DiskCache` 类中那样手动检查记录是否在过期指南内。让我们在 IPython（或您选择的解释器）中使用 20 秒的时间跨度来尝试它，这样我们就可以看到缓存过期：

```py
In [1]: from chp3.rediscache import RedisCache

In [2]: from datetime import timedelta

In [3]: cache = RedisCache(expires=timedelta(seconds=20))

In [4]: cache['test'] = {'html': '...', 'code': 200}

In [5]: cache['test']
Out[5]: {'code': 200, 'html': '...'}

In [6]: import time; time.sleep(20)

In [7]: cache['test']
---------------------------------------------------------------------------
KeyError Traceback (most recent call last)
...
KeyError: 'test does not exist'

```

结果显示，我们的缓存按预期工作，能够将数据在 JSON、字典和 Redis 键值存储之间进行序列化和反序列化，并且能够使结果过期。

# 压缩

为了使这个缓存功能与原始磁盘缓存相比更加完整，我们需要添加一个最终的功能：**压缩**。这可以通过与磁盘缓存类似的方式实现，通过序列化数据，然后使用 `zlib` 进行压缩，如下所示：

```py
import zlib 
from bson.binary import Binary 

class RedisCache:
    def __init__(..., compress=True):
        ...
        self.compress = compress

    def __getitem__(self, url): 
        record = self.client.get(url)
        if record:
            if self.compress:
                record = zlib.decompress(record)
            return json.loads(record.decode(self.encoding))
        else: 
            raise KeyError(url + ' does not exist') 

    def __setitem__(self, url, result): 
        data = bytes(json.dumps(result), self.encoding)
        if self.compress:
            data = zlib.compress(data)
        self.client.setex(url, self.expires, data)

```

# 测试缓存

`RedisCache` 类的源代码可在 [`github.com/kjam/wswp/blob/master/code/chp3/rediscache.py`](https://github.com/kjam/wswp/blob/master/code/chp3/rediscache.py) 找到，并且与 `DiskCache` 类一样，缓存可以通过链接爬虫在任何 Python 解释器中进行测试。在这里，我们使用 IPython 来使用 `%time` 命令：

```py
In [1]: from chp3.advanced_link_crawler import link_crawler

In [2]: from chp3.rediscache import RedisCache

In [3]: %time link_crawler('http://example.webscraping.com/', '/(index|view)', cache=RedisCache())
Downloading: http://example.webscraping.com/
Downloading: http://example.webscraping.com/index/1
Downloading: http://example.webscraping.com/index/2
...
Downloading: http://example.webscraping.com/view/Afghanistan-1
CPU times: user 352 ms, sys: 32 ms, total: 384 ms
Wall time: 1min 42s

In [4]: %time link_crawler('http://example.webscraping.com/', '/(index|view)', cache=RedisCache())
Loaded from cache: http://example.webscraping.com/
Loaded from cache: http://example.webscraping.com/index/1
Loaded from cache: http://example.webscraping.com/index/2
...
Loaded from cache: http://example.webscraping.com/view/Afghanistan-1
CPU times: user 24 ms, sys: 8 ms, total: 32 ms
Wall time: 282 ms

```

这里所用的时间与我们的 `DiskCache` 在第一次迭代时大致相同。然而，一旦缓存被加载，Redis 的速度优势就真正显现出来，与我们的非压缩磁盘缓存系统相比，速度提高了 3 倍以上。我们缓存代码的可读性提高以及将 Redis 集群扩展到高可用大数据解决方案的能力，简直是锦上添花！

# 探索 requests-cache

有时，你可能想要缓存一个内部使用`requests`的库，或者你可能不想自己管理缓存类和处理。如果是这种情况，`requests-cache`（[`github.com/reclosedev/requests-cache`](https://github.com/reclosedev/requests-cache)）是一个非常好的库，它实现了为`requests`库创建缓存的一些建后端选项。当使用`requests-cache`时，所有通过`requests`库访问 URL 的`get`请求将首先检查缓存，只有在未找到时才会请求页面。

`requests-cache`支持包括 Redis、MongoDB（一个 NoSQL 数据库）、SQLite（一个轻量级的关系型数据库）和内存（它不是持久的，因此不建议使用）在内的几个后端。由于我们已经有 Redis 设置好了，我们可以将其用作后端。要开始，我们首先需要安装这个库：

```py
pip install requests-cache

```

现在，我们可以简单地使用 IPython 中的几个简单命令来安装和测试我们的缓存：

```py
In [1]: import requests_cache

In [2]: import requests

In [3]: requests_cache.install_cache(backend='redis')

In [4]: requests_cache.clear()

In [5]: url = 'http://example.webscraping.com/view/United-Kingdom-239'

In [6]: resp = requests.get(url)

In [7]: resp.from_cache
Out[7]: False

In [8]: resp = requests.get(url)

In [9]: resp.from_cache
Out[9]: True

```

如果我们使用这个代替我们自己的缓存类，我们只需要使用`install_cache`命令实例化缓存，然后每个请求（只要我们正在使用`requests`库）都会在我们的 Redis 后端中维护。我们还可以使用一些简单的命令设置过期时间：

```py
from datetime import timedelta
requests_cache.install_cache(backend='redis', expire_after=timedelta(days=30))

```

为了测试使用`requests-cache`与我们的实现相比的速度，我们构建了一个新的下载器和链接爬虫来使用。这个下载器还实现了在`requests-cache`用户指南中记录的推荐的`requests`钩子，以允许节流：[`requests-cache.readthedocs.io/en/latest/user_guide.html`](https://requests-cache.readthedocs.io/en/latest/user_guide.html)。

要查看完整的代码，请查看新的下载器（[`github.com/kjam/wswp/blob/master/code/chp3/downloader_requests_cache.py`](https://github.com/kjam/wswp/blob/master/code/chp3/downloader_requests_cache.py)）和链接爬虫（[`github.com/kjam/wswp/blob/master/code/chp3/requests_cache_link_crawler.py`](https://github.com/kjam/wswp/blob/master/code/chp3/requests_cache_link_crawler.py)）。我们可以使用 IPython 来测试它们的性能：

```py
In [1]: from chp3.requests_cache_link_crawler import link_crawler
...
In [3]: %time link_crawler('http://example.webscraping.com/', '/(index|view)')
Returning from cache: http://example.webscraping.com/
Returning from cache: http://example.webscraping.com/index/1
Returning from cache: http://example.webscraping.com/index/2
...
Returning from cache: http://example.webscraping.com/view/Afghanistan-1
CPU times: user 116 ms, sys: 12 ms, total: 128 ms
Wall time: 359 ms

```

我们发现`requests-cache`解决方案在性能上略逊于我们自己的 Redis 解决方案，但它也使用了更少的代码行数，并且仍然相当快速（而且比我们的 DiskCache 解决方案快得多）。特别是如果你正在使用另一个可能内部管理`requests`的库，那么`requests-cache`实现是一个非常有用的工具。

# 摘要

在本章中，我们了解到缓存下载的网页可以在重新爬取网站时节省时间和最小化带宽。然而，缓存页面会占用磁盘空间，其中一些可以通过压缩来缓解。此外，基于现有的存储系统，如 Redis，可以用来避免速度、内存和文件系统限制。

在下一章中，我们将为我们的爬虫添加更多功能，以便我们可以并发下载网页，从而使网络爬取速度更快。
