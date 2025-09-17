# 第六章 屏幕抓取和其他实用应用

在本章中，我们将涵盖以下主题：

+   使用 Google Maps API 搜索商业地址

+   使用 Google Maps URL 搜索地理坐标

+   在维基百科中搜索文章

+   搜索 Google 股票报价

+   在 GitHub 上搜索源代码仓库

+   从 BBC 读取新闻源

+   爬取网页中存在的链接

# 简介

本章展示了您可以编写的某些有趣的 Python 脚本，用于从网络中提取有用的信息，例如，搜索商业地址、特定公司的股票报价或新闻机构的最新新闻。这些脚本展示了 Python 如何在不与复杂的 API 通信的情况下以更简单的方式提取简单信息。

按照这些配方，您应该能够编写用于复杂场景的代码，例如，查找有关业务的信息，包括位置、新闻、股票报价等。

# 使用 Google Maps API 搜索商业地址

您想搜索您所在地区一家知名企业的地址。

## 准备工作

您可以使用 Python 地理编码库`pygeocoder`来搜索本地商业。您需要使用`pip`或`easy_install`从**PyPI**安装此库，通过输入`$ pip install pygeocoder`或`$ easy_install pygeocoder`。

## 如何做到这一点...

让我们使用几行 Python 代码找到知名英国零售商 Argos Ltd.的地址。

列表 6.1 提供了一个简单的地理编码示例，用于搜索商业地址，如下所示：

```py
#!/usr/bin/env python

# Python Network Programming Cookbook -- Chapter - 6
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

from pygeocoder import Geocoder

def search_business(business_name):

  results = Geocoder.geocode(business_name)

  for result in results:
    print result

if __name__ == '__main__':
  business_name =  "Argos Ltd, London" 
  print "Searching %s" %business_name
  search_business(business_name)
```

此配方将打印出 Argos Ltd.的地址，如所示。输出可能会根据您安装的地理编码库的输出略有不同：

```py
$ python 6_1_search_business_addr.py
Searching Argos Ltd, London 

Argos Ltd, 110-114 King Street, London, Greater London W6 0QP, UK

```

## 它是如何工作的...

此配方依赖于 Python 第三方地理编码库。

此配方定义了一个简单的函数`search_business()`，它接受业务名称作为输入并将其传递给`geocode()`函数。`geocode()`函数可以根据您的搜索词返回零个或多个搜索结果。

在此配方中，`geocode()`函数将业务名称 Argos Ltd.，伦敦作为搜索查询。作为回报，它给出了 Argos Ltd.的地址，即 110-114 King Street，伦敦，大伦敦 W6 0QP，英国。

## 参见

`pygeocoder`库功能强大，具有许多有趣和有用的地理编码功能。您可以在开发者的网站上找到更多详细信息，网址为[`bitbucket.org/xster/pygeocoder/wiki/Home`](https://bitbucket.org/xster/pygeocoder/wiki/Home)。

# 使用 Google Maps URL 搜索地理坐标

有时您可能需要一个简单的函数，通过仅提供该城市的名称即可给出该城市的地理坐标。您可能对安装任何第三方库来完成此简单任务不感兴趣。

## 如何做到这一点...

在这个简单的屏幕抓取示例中，我们使用谷歌地图 URL 查询城市的纬度和经度。用于查询的 URL 可以在对谷歌地图页面进行自定义搜索后找到。我们可以执行以下步骤从谷歌地图中提取一些信息。

让我们使用`argparse`模块从命令行获取一个城市的名称。

我们可以使用`urllib`模块的`urlopen()`函数打开地图搜索 URL。如果 URL 正确，这将给出 XML 输出。

现在，处理 XML 输出以获取该城市的地理坐标。

列表 6.2 帮助使用谷歌地图查找城市的地理坐标，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 6
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.
import argparse
import os
import urllib

ERROR_STRING = '<error>'

def find_lat_long(city):
  """ Find geographic coordinates """
  # Encode query string into Google maps URL
    url = 'http://maps.google.com/?q=' + urllib.quote(city) + 
'&output=js'
    print 'Query: %s' % (url)

  # Get XML location from Google maps
    xml = urllib.urlopen(url).read()

    if ERROR_STRING in xml:
      print '\nGoogle cannot interpret the city.'
      return
    else:
    # Strip lat/long coordinates from XML
      lat,lng = 0.0,0.0
      center = xml[xml.find('{center')+10:xml.find('}',xml.find('{center'))]
      center = center.replace('lat:','').replace('lng:','')
      lat,lng = center.split(',')
      print "Latitude/Longitude: %s/%s\n" %(lat, lng)

    if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='City Geocode 
Search')
      parser.add_argument('--city', action="store", dest="city", 
required=True)
      given_args = parser.parse_args() 

      print "Finding geographic coordinates of %s" 
%given_args.city
      find_lat_long(given_args.city)
```

如果您运行此脚本，您应该看到以下类似的内容：

```py
$ python 6_2_geo_coding_by_google_maps.py --city=London 
Finding geograhic coordinates of London 
Query: http://maps.google.com/?q=London&output=js 
Latitude/Longitude: 51.511214000000002/-0.119824 

```

## 它是如何工作的...

此配方从命令行获取一个城市的名称并将其传递给`find_lat_long()`函数。此函数使用`urllib`模块的`urlopen()`函数查询谷歌地图服务并获取 XML 输出。然后，搜索错误字符串`'<error>'`。如果没有出现，这意味着有一些好的结果。

如果您打印出原始 XML，它是一长串为浏览器生成的字符流。在浏览器中，显示地图的层可能很有趣。但在我们的情况下，我们只需要纬度和经度。

从原始 XML 中，使用字符串方法`find()`提取纬度和经度。这是搜索关键字"center"。此列表键具有地理坐标信息。但它还包含额外的字符，这些字符使用字符串方法`replace()`被移除。

您可以尝试这个配方来找出世界上任何已知城市的纬度/经度。

# 在维基百科中搜索文章

维基百科是一个收集关于几乎任何事物的信息的绝佳网站，例如，人物、地点、技术等等。如果您想从 Python 脚本中在维基百科上搜索某些内容，这个配方就是为您准备的。

这里有一个例子：

![在维基百科中搜索文章](img/3463OS_06_01.jpg)

## 准备工作

您需要使用`pip`或`easy_install`通过输入`$ pip install pyyaml`或`$ easy_install pyyaml`从 PyPI 安装`pyyaml`第三方库。

## 如何做...

让我们在维基百科中搜索关键字`Islam`并按行打印每个搜索结果。

列表 6.3 解释了如何在维基百科中搜索一篇文章，如下所示：

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python Network Programming Cookbook -- Chapter - 6
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications

import argparse
import re
import yaml
import urllib
import urllib2

SEARCH_URL = 'http://%s.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s&sroffset=%d&srlimit=%d&format=yaml'

class Wikipedia:

  def __init__(self, lang='en'):
    self.lang = lang

  def _get_content(self, url):
    request = urllib2.Request(url)
    request.add_header('User-Agent', 'Mozilla/20.0')

    try:
      result = urllib2.urlopen(request)
      except urllib2.HTTPError, e:
        print "HTTP Error:%s" %(e.reason)
      except Exception, e:
        print "Error occurred: %s" %str(e)
      return result

  def search_content(self, query, page=1, limit=10):
    offset = (page - 1) * limit
    url = SEARCH_URL % (self.lang, urllib.quote_plus(query), 
offset, limit)
    content = self._get_content(url).read()

    parsed = yaml.load(content)
    search = parsed['query']['search']
    if not search:
    return

    results = []
    for article in search:
      snippet = article['snippet']
      snippet = re.sub(r'(?m)<.*?>', '', snippet)
      snippet = re.sub(r'\s+', ' ', snippet)
      snippet = snippet.replace(' . ', '. ')
      snippet = snippet.replace(' , ', ', ')
      snippet = snippet.strip()

    results.append({
      'title' : article['title'].strip(),
'snippet' : snippet
    })

    return results

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Wikipedia search')
  parser.add_argument('--query', action="store", dest="query", 
required=True)
  given_args = parser.parse_args()

  wikipedia = Wikipedia()
  search_term = given_args.query
  print "Searching Wikipedia for %s" %search_term 
  results = wikipedia.search_content(search_term)
  print "Listing %s search results..." %len(results)
  for result in results:
    print "==%s== \n \t%s" %(result['title'], result['snippet'])
  print "---- End of search results ----"
```

运行此配方查询维基百科关于伊斯兰的结果如下：

```py
$ python 6_3_search_article_in_wikipedia.py --query='Islam' 
Searching Wikipedia for Islam 
Listing 10 search results... 
==Islam== 
 Islam. (
ˈ
 | 
ɪ
 | s | l | 
ɑː
 | m 
الإسلام
, ar | ALA | al-
ʾ
Isl
ā
m  æl
ʔɪ
s
ˈ
læ
ː
m | IPA | ar-al_islam. ... 

==Sunni Islam== 
 Sunni Islam (
ˈ
 | s | u
ː
 | n | i or 
ˈ
 | s | 
ʊ
 | n | i |) is the 
largest branch of Islam ; its adherents are referred to in Arabic as ... 
==Muslim== 
 A Muslim, also spelled Moslem is an adherent of Islam, a monotheistic Abrahamic religion based on the Qur'an —which Muslims consider the ... 
==Sharia== 
 is the moral code and religious law of Islam. Sharia deals with 
many topics addressed by secular law, including crime, politics, and ... 
==History of Islam== 
 The history of Islam concerns the Islamic religion and its 
adherents, known as Muslim s. " "Muslim" is an Arabic word meaning 
"one who ... 

==Caliphate== 
 a successor to Islamic prophet Muhammad ) and all the Prophets 
of Islam. The term caliphate is often applied to successions of 
Muslim ... 
==Islamic fundamentalism== 
 Islamic ideology and is a group of religious ideologies seen as 
advocating a return to the "fundamentals" of Islam : the Quran and 
the Sunnah. ... 
==Islamic architecture== 
 Islamic architecture encompasses a wide range of both secular 
and religious styles from the foundation of Islam to the present day. ... 
---- End of search results ---- 

```

## 它是如何工作的...

首先，我们收集搜索文章的维基百科 URL 模板。我们创建了一个名为`Wikipedia`的类，它有两个方法：`_get_content()`和`search_content()`。默认情况下，初始化时，该类将语言属性`lang`设置为`en`（英语）。

命令行查询字符串被传递给`search_content()`方法。然后它通过插入变量（如语言、查询字符串、页面偏移和要返回的结果数量）来构建实际的搜索 URL。`search_content()`方法可以可选地接受参数，偏移量由`(page -1) * limit`表达式确定。

搜索结果的内容是通过`_get_content()`方法获取的，该方法调用`urllib`的`urlopen()`函数。在搜索 URL 中，我们设置了结果格式`yaml`，这基本上是为了纯文本文件。然后使用 Python 的`pyyaml`库解析`yaml`搜索结果。

搜索结果通过替换每个结果项中找到的正则表达式进行处理。例如，`re.sub(r'(?m)<.*?>', '', snippet)`表达式将替换片段字符串中的原始模式`(?m)<.*?>`。要了解更多关于正则表达式的信息，请访问 Python 文档页面，网址为[`docs.python.org/2/howto/regex.html`](http://docs.python.org/2/howto/regex.html)。

在维基百科术语中，每篇文章都有一个片段或简短描述。我们创建了一个字典项列表，其中每个项包含每个搜索结果的标题和片段。通过遍历这个字典项列表，结果被打印在屏幕上。

# 搜索谷歌股票报价

如果您对任何公司的股票报价感兴趣，此配方可以帮助您找到该公司的今日股票报价。

## 准备工作

我们假设您已经知道您喜欢的公司用于在任何证券交易所上市的符号。如果您不知道，可以从公司网站获取符号，或者直接在谷歌上搜索。

## 如何操作...

在这里，我们使用谷歌财经([`finance.google.com/`](http://finance.google.com/))来搜索给定公司的股票报价。您可以通过命令行输入符号，如下所示。

列表 6.4 描述了如何搜索谷歌股票报价，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 6
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications. 

import argparse
import urllib
import re
from datetime import datetime

SEARCH_URL = 'http://finance.google.com/finance?q='

def get_quote(symbol):
  content = urllib.urlopen(SEARCH_URL + symbol).read()
  m = re.search('id="ref_694653_l".*?>(.*?)<', content)
  if m:
    quote = m.group(1)
  else:
    quote = 'No quote available for: ' + symbol
  return quote

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Stock quote 
search')
  parser.add_argument('--symbol', action="store", dest="symbol", 
required=True)
  given_args = parser.parse_args() 
  print "Searching stock quote for symbol '%s'" %given_args.symbol 
  print "Stock  quote for %s at %s: %s" %(given_args.symbol , 
datetime.today(),  get_quote(given_args.symbol))
```

如果你运行此脚本，你将看到类似以下输出。在此，通过输入符号`goog`来搜索谷歌的股票报价，如下所示：

```py
$ python 6_4_google_stock_quote.py --symbol=goog 
Searching stock quote for symbol 'goog' 
Stock quote for goog at 2013-08-20 18:50:29.483380: 868.86 

```

## 它是如何工作的...

此配方使用`urllib`的`urlopen()`函数从谷歌财经网站获取股票数据。

通过使用正则表达式库`re`，它定位到第一个项目组中的股票报价数据。`re`的`search()`函数足够强大，可以搜索内容并过滤特定公司的 ID 数据。

使用此配方，我们搜索了谷歌的股票报价，该报价在 2013 年 8 月 20 日为`868.86`。

# 在 GitHub 上搜索源代码仓库

作为一名 Python 程序员，您可能已经熟悉 GitHub ([`www.github.com`](http://www.github.com))，一个源代码共享网站，如下面的截图所示。您可以使用 GitHub 将源代码私密地分享给团队或公开地分享给全世界。它有一个很好的 API 接口，可以查询任何源代码仓库。这个食谱可能为您创建自己的源代码搜索引擎提供了一个起点。

![在 GitHub 上搜索源代码仓库](img/3463OS_06_02.jpg)

## 准备工作

要运行此食谱，您需要通过输入 `$ pip install requests` 或 `$ easy_install requests` 来安装第三方 Python 库 `requests`。

## 如何操作...

我们希望定义一个 `search_repository()` 函数，它将接受作者名称（也称为程序员）、仓库和搜索键。作为回报，它将根据搜索键返回可用的结果。从 GitHub API 来看，以下是可以用的搜索键：`issues_url`、`has_wiki`、`forks_url`、`mirror_url`、`subscription_url`、`notifications_url`、`collaborators_url`、`updated_at`、`private`、`pulls_url`、`issue_comment_url`、`labels_url`、`full_name`、`owner`、`statuses_url`、`id`、`keys_url`、`description`、`tags_url`、`network_count`、`downloads_url`、`assignees_url`、`contents_url`、`git_refs_url`、`open_issues_count`、`clone_url`、`watchers_count`、`git_tags_url`、`milestones_url`、`languages_url`、`size`、`homepage`、`fork`、`commits_url`、`issue_events_url`、`archive_url`、`comments_url`、`events_url`、`contributors_url`、`html_url`、`forks`、`compare_url`、`open_issues`、`git_url`、`svn_url`、`merges_url`、`has_issues`、`ssh_url`、`blobs_url`、`master_branch`、`git_commits_url`、`hooks_url`、`has_downloads`、`watchers`、`name`、`language`、`url`、`created_at`、`pushed_at`、`forks_count`、`default_branch`、`teams_url`、`trees_url`、`organization`、`branches_url`、`subscribers_url` 和 `stargazers_url`。

列表 6.5 给出了在 GitHub 上搜索源代码仓库详细信息的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 6
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

SEARCH_URL_BASE = 'https://api.github.com/repos'

import argparse
import requests
import json

def search_repository(author, repo, search_for='homepage'):
  url = "%s/%s/%s" %(SEARCH_URL_BASE, author, repo)
  print "Searching Repo URL: %s" %url
  result = requests.get(url)
  if(result.ok):
    repo_info = json.loads(result.text or result.content)
    print "Github repository info for: %s" %repo
    result = "No result found!"
    keys = [] 
    for key,value in repo_info.iteritems():
      if  search_for in key:
          result = value
      return result

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Github search')
  parser.add_argument('--author', action="store", dest="author", 
required=True)
  parser.add_argument('--repo', action="store", dest="repo", 
required=True)
  parser.add_argument('--search_for', action="store", 
dest="search_for", required=True)

  given_args = parser.parse_args() 
  result = search_repository(given_args.author, given_args.repo, 
given_args.search_for)
  if isinstance(result, dict):
    print "Got result for '%s'..." %(given_args.search_for)
    for key,value in result.iteritems():
    print "%s => %s" %(key,value)
  else:
    print "Got result for %s: %s" %(given_args.search_for, 
result)
```

如果您运行此脚本以搜索 Python 网络框架 Django 的所有者，您可以得到以下结果：

```py
$ python 6_5_search_code_github.py --author=django --repo=django --search_for=owner 
Searching Repo URL: https://api.github.com/repos/django/django 
Github repository info for: django 
Got result for 'owner'... 
following_url => https://api.github.com/users/django/following{/other_user} 
events_url => https://api.github.com/users/django/events{/privacy} 
organizations_url => https://api.github.com/users/django/orgs 
url => https://api.github.com/users/django 
gists_url => https://api.github.com/users/django/gists{/gist_id} 
html_url => https://github.com/django 
subscriptions_url => https://api.github.com/users/django/subscriptions 
avatar_url => https://1.gravatar.com/avatar/fd542381031aa84dca86628ece84fc07?d=https%3A%2F%2Fidenticons.github.com%2Fe94df919e51ae96652259468415d4f77.png 
repos_url => https://api.github.com/users/django/repos 
received_events_url => https://api.github.com/users/django/received_events 
gravatar_id => fd542381031aa84dca86628ece84fc07 
starred_url => https://api.github.com/users/django/starred{/owner}{/repo} 
login => django 
type => Organization 
id => 27804 
followers_url => https://api.github.com/users/django/followers 

```

## 工作原理...

此脚本接受三个命令行参数：仓库作者（`--author`）、仓库名称（`--repo`）和要搜索的项目（`--search_for`）。这些参数通过 `argpase` 模块进行处理。

我们的 `search_repository()` 函数将命令行参数追加到固定的搜索 URL，并通过调用 `requests` 模块的 `get()` 函数接收内容。

默认情况下，搜索结果以 JSON 格式返回。然后使用 `json` 模块的 `loads()` 方法处理此内容。然后在结果中查找搜索键，并将该键的对应值返回给 `search_repository()` 函数的调用者。

在主用户代码中，我们检查搜索结果是否是 Python 字典的实例。如果是，则迭代打印键/值。否则，只打印值。

# 从 BBC 读取新闻源

如果你正在开发一个包含新闻和故事的社交网络网站，你可能对展示来自各种世界新闻机构（如 BBC 和路透社）的新闻感兴趣。让我们尝试通过 Python 脚本从 BBC 读取新闻。

## 准备中

此菜谱依赖于 Python 的第三方`feedparser`库。你可以通过运行以下命令来安装它：

```py
$ pip install feedparser

```

或者

```py
$ easy_install feedparser

```

## 如何操作...

首先，我们从 BBC 网站收集 BBC 的新闻源 URL。这个 URL 可以用作模板来搜索各种类型的新闻，如世界、英国、健康、商业和技术。因此，我们可以将显示的新闻类型作为用户输入。然后，我们依赖于`read_news()`函数，它将从 BBC 获取新闻。

列表 6.6 解释了如何从 BBC 读取新闻源，如下面的代码所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 6
# This program is optimized for Python 2.7\. 
# It may run on any other version with/without modifications.

from datetime import datetime
import feedparser 
BBC_FEED_URL = 'http://feeds.bbci.co.uk/news/%s/rss.xml'

def read_news(feed_url):
  try:
    data = feedparser.parse(feed_url)
  except Exception, e:
    print "Got error: %s" %str(e)

  for entry in data.entries:
    print(entry.title)
    print(entry.link)
    print(entry.description)
    print("\n") 

if __name__ == '__main__':
  print "==== Reading technology news feed from bbc.co.uk 
(%s)====" %datetime.today()

  print "Enter the type of news feed: "
  print "Available options are: world, uk, health, sci-tech, 
business, technology"
  type = raw_input("News feed type:")
  read_news(BBC_FEED_URL %type)
  print "==== End of BBC news feed ====="
```

运行此脚本将显示可用的新闻类别。如果我们选择技术作为类别，你可以获取最新的技术新闻，如下面的命令所示：

```py
$ python 6_6_read_bbc_news_feed.py 
==== Reading technology news feed from bbc.co.uk (2013-08-20 19:02:33.940014)==== 
Enter the type of news feed:
Available options are: world, uk, health, sci-tech, business, technology 
News feed type:technology 
Xbox One courts indie developers 
http://www.bbc.co.uk/news/technology-23765453#sa-ns_mchannel=rss&ns_source=PublicRSS20-sa 
Microsoft is to give away free Xbox One development kits to encourage independent developers to self-publish games for its forthcoming console. 

Fast in-flight wi-fi by early 2014 
http://www.bbc.co.uk/news/technology-23768536#sa-ns_mchannel=rss&ns_source=PublicRSS20-sa 
Passengers on planes, trains and ships may soon be able to take advantage of high-speed wi-fi connections, says Ofcom. 

Anonymous 'hacks council website' 
http://www.bbc.co.uk/news/uk-england-surrey-23772635#sa-ns_mchannel=rss&ns_source=PublicRSS20-sa 
A Surrey council blames hackers Anonymous after references to a Guardian journalist's partner detained at Heathrow Airport appear on its website. 

Amazon.com website goes offline 
http://www.bbc.co.uk/news/technology-23762526#sa-ns_mchannel=rss&ns_source=PublicRSS20-sa 
Amazon's US website goes offline for about half an hour, the latest high-profile internet firm to face such a problem in recent days. 

[TRUNCATED]

```

## 它是如何工作的...

在这个菜谱中，`read_news()`函数依赖于 Python 的第三方模块`feedparser`。`feedparser`模块的`parser()`方法以结构化的方式返回源数据。

在这个菜谱中，`parser()`方法解析给定的源 URL。这个 URL 由`BBC_FEED_URL`和用户输入构成。

在调用`parse()`获取一些有效的源数据后，然后打印数据的内容，例如每个源条目的标题、链接和描述。

# 爬取网页中存在的链接

有时你希望在网页中找到特定的关键词。在网页浏览器中，你可以使用浏览器的页面搜索功能来定位术语。一些浏览器可以突出显示它。在复杂的情况下，你可能想深入挖掘并跟随网页中存在的每个 URL，以找到那个特定的术语。这个菜谱将为你自动化这个任务。

## 如何操作...

让我们编写一个`search_links()`函数，它将接受三个参数：搜索 URL、递归搜索的深度以及搜索关键词/术语，因为每个 URL 的内容中可能包含链接，而该内容可能包含更多要爬取的 URL。为了限制递归搜索，我们定义了一个深度。达到那个深度后，将不再进行递归搜索。

列表 6.7 给出了爬取网页中存在的链接的代码，如下面的代码所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 6
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import argparse
import sys
import httplib
import re

processed = []

def search_links(url, depth, search):
  # Process http links that are not processed yet
  url_is_processed = (url in processed)
  if (url.startswith("http://") and (not url_is_processed)):
    processed.append(url)
    url = host = url.replace("http://", "", 1)
    path = "/"

    urlparts = url.split("/")
    if (len(urlparts) > 1):
      host = urlparts[0]
      path = url.replace(host, "", 1)

     # Start crawling
     print "Crawling URL path:%s%s " %(host, path)
     conn = httplib.HTTPConnection(host)
     req = conn.request("GET", path)
     result = conn.getresponse()

    # find the links
    contents = result.read()
    all_links = re.findall('href="(.*?)"', contents)

    if (search in contents):
      print "Found " + search + " at " + url

      print " ==> %s: processing %s links" %(str(depth), 
str(len(all_links)))
      for href in all_links:
      # Find relative urls
      if (href.startswith("/")):
        href = "http://" + host + href

        # Recurse links
        if (depth > 0):
          search_links(href, depth-1, search)
    else:
      print "Skipping link: %s ..." %url

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Webpage link 
crawler')
  parser.add_argument('--url', action="store", dest="url", 
required=True)
  parser.add_argument('--query', action="store", dest="query", 
required=True)
  parser.add_argument('--depth', action="store", dest="depth", 
default=2)

  given_args = parser.parse_args() 

  try:
    search_links(given_args.url,  
given_args.depth,given_args.query)
    except KeyboardInterrupt:
      print "Aborting search by user request."
```

如果你运行此脚本来搜索[www.python.org](http://www.python.org)中的`python`，你将看到以下类似的输出：

```py
$ python 6_7_python_link_crawler.py --url='http://python.org' --query='python' 
Crawling URL path:python.org/ 
Found python at python.org 
 ==> 2: processing 123 links 
Crawling URL path:www.python.org/channews.rdf 
Found python at www.python.org/channews.rdf 
 ==> 1: processing 30 links 
Crawling URL path:www.python.org/download/releases/3.4.0/ 
Found python at www.python.org/download/releases/3.4.0/ 
 ==> 0: processing 111 links 
Skipping link: https://ep2013.europython.eu/blog/2013/05/15/epc20145-call-proposals ... 
Crawling URL path:www.python.org/download/releases/3.2.5/ 
Found python at www.python.org/download/releases/3.2.5/ 
 ==> 0: processing 113 links 
...
Skipping link: http://www.python.org/download/releases/3.2.4/ ... 
Crawling URL path:wiki.python.org/moin/WikiAttack2013 
^CAborting search by user request. 

```

## 它是如何工作的...

此菜谱可以接受三个命令行输入：搜索 URL（`--url`）、查询字符串（`--query`）和递归深度（`--depth`）。这些输入由`argparse`模块处理。

当使用之前的参数调用 `search_links()` 函数时，它将递归地遍历该给定网页上找到的所有链接。如果完成时间过长，你可能希望提前退出。因此，`search_links()` 函数被放置在一个 try-catch 块中，该块可以捕获用户的键盘中断操作，例如 *Ctrl* + *C*。

`search_links()` 函数通过一个名为 `processed` 的列表来跟踪已访问的链接。这样做是为了使其全局可用，以便在所有递归函数调用中都能访问。

在单个搜索实例中，确保只处理 HTTP URL，以避免潜在的 SSL 证书错误。URL 被拆分为主机和路径。主要的爬取操作使用 `httplib` 的 `HTTPConnection()` 函数启动。它逐渐发起 `GET` 请求，然后使用正则表达式模块 `re` 处理响应。这收集了响应中的所有链接。然后检查每个响应以查找搜索词。如果找到搜索词，它将打印该事件。

收集到的链接以相同的方式递归访问。如果找到任何相对 URL，该实例将通过在主机和路径前添加 `http://` 转换为完整 URL。如果搜索深度大于 0，则激活递归。它将深度减少 1，并再次运行搜索函数。当搜索深度变为 0 时，递归结束。
