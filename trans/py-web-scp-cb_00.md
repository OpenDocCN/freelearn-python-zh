# 前言

互联网包含大量数据。这些数据既通过结构化API提供，也通过网站直接提供。虽然API中的数据高度结构化，但在网页中找到的信息通常是非结构化的，需要收集、提取和处理才能有价值。收集数据只是旅程的开始，因为这些数据还必须存储、挖掘，然后以增值形式向他人展示。

通过这本书，您将学习从网站收集各种信息所需的核心任务。我们将介绍如何收集数据，如何执行几种常见的数据操作（包括存储在本地和远程数据库中），如何执行常见的基于媒体的任务，如将图像和视频转换为缩略图，如何使用NTLK清理非结构化数据，如何检查几种数据挖掘和可视化工具，以及构建基于微服务的爬虫和API的核心技能，这些技能可以并且将在云上运行。

通过基于配方的方法，我们将学习独立的技术，以解决不仅仅是爬取，还包括数据操作和管理、数据挖掘、可视化、微服务、容器和云操作中涉及的特定任务。这些配方将以渐进和整体的方式建立技能，不仅教授如何执行爬取的基础知识，还将带您从爬取的结果到通过云向他人提供的服务。我们将使用Python、容器和云生态系统中的常用工具构建一个实际的网络爬虫服务。

# 这本书适合谁

这本书适合那些想要学习使用爬取过程从网站提取数据以及如何使用各种数据管理工具和云服务的人。编码将需要基本的Python编程语言技能。

这本书还适合那些希望了解更大的工具生态系统，用于检索、存储和搜索数据，以及使用现代工具和Python库创建数据API和云服务的人。您可能还会使用Docker和Amazon Web Services在云上打包和部署爬虫。

# 本书涵盖内容

第1章，“开始爬取”，介绍了网页爬取的几个概念和工具。我们将研究如何安装并使用工具，如requests、urllib、BeautifulSoup、Scrapy、PhantomJS和Selenium进行基本任务。

第2章，“数据获取和提取”，基于对HTML结构的理解以及如何查找和提取嵌入式数据。我们将涵盖DOM中的许多概念以及如何使用BeautifulSoup、XPath、LXML和CSS选择器查找和提取数据。我们还简要介绍了Unicode / UTF8的工作。

第3章，“处理数据”，教你如何以多种格式加载和操作数据，然后如何将数据存储在各种数据存储中（S3、MySQL、PostgreSQL和ElasticSearch）。网页中的数据以各种格式表示，最常见的是HTML、JSON、CSV和XML。我们还将研究使用消息队列系统，主要是AWS SQS，来帮助构建强大的数据处理管道。

第4章，“处理图像、音频和其他资产”，研究了检索多媒体项目的方法，将它们存储在本地，并执行诸如OCR、生成缩略图、制作网页截图、从视频中提取音频以及在YouTube播放列表中找到所有视频URL等多项任务。

[第5章](713f8046-658b-4346-9ece-59e2e92a2c53.xhtml)，*爬取-行为准则*，涵盖了与爬取的合法性有关的几个概念，以及进行礼貌爬取的实践。我们将研究处理robots.txt和站点地图的工具，以尊重网络主机对可接受行为的要求。我们还将研究爬行的几个方面的控制，比如使用延迟、包含爬行的深度和长度、使用用户代理以及实施缓存以防止重复请求。

[第6章](390110fd-80c7-42e7-bc1a-b93fe5b87f31.xhtml)，*爬取挑战与解决方案*，涵盖了编写健壮爬虫时面临的许多挑战，以及如何处理许多情况。这些情况包括分页、重定向、登录表单、保持爬虫在同一域内、请求失败时重试以及处理验证码。

[第7章](90200c61-a13c-4384-925a-e9adbac1eaf0.xhtml)，*文本整理和分析*，探讨了各种工具，比如使用NLTK进行自然语言处理，以及如何去除常见的噪音词和标点符号。我们经常需要处理网页的文本内容，以找到页面上作为文本一部分的信息，既不是结构化/嵌入式数据，也不是多媒体。这需要使用各种概念和工具来清理和理解文本。

[第8章](8a120fd5-e37d-4d54-9c12-93cf320cee47.xhtml)，*搜索、挖掘和可视化数据*，涵盖了在网上搜索数据、存储和组织数据，以及从已识别的关系中得出结果的几种方法。我们将看到如何理解维基百科贡献者的地理位置，找到IMDB上演员之间的关系，以及在Stack Overflow上找到与特定技术匹配的工作。

[第9章](c91fe735-b615-42c9-9513-29121a7f022a.xhtml)，*创建一个简单的数据API*，教会我们如何创建一个爬虫作为服务。我们将使用Flask为爬虫创建一个REST API。我们将在这个API后面运行爬虫作为服务，并能够提交请求来爬取特定页面，以便从爬取和本地ElasticSearch实例中动态查询数据。

[第10章](ae07f922-9f75-4b94-9902-e5a5ebeec167.xhtml)，*使用Docker创建爬虫微服务*，通过将服务和API打包到Docker集群中，并通过消息队列系统（AWS SQS）分发请求，继续扩展我们的爬虫服务。我们还将介绍使用Docker集群工具来扩展和缩减爬虫实例。

[第11章](05a687e5-e5d7-4ffb-aa28-6078fbab5fea.xhtml)，*使爬虫成为真正的服务*，通过充实上一章中创建的服务来结束，添加一个爬虫，汇集了之前介绍的各种概念。这个爬虫可以帮助分析StackOverflow上的职位发布，以找到并比较使用指定技术的雇主。该服务将收集帖子，并允许查询以找到并比较这些公司。

# 为了充分利用本书

本书中所需的主要工具是Python 3解释器。这些配方是使用Anaconda Python发行版的免费版本编写的，具体版本为3.6.1。其他Python 3发行版应该也能很好地工作，但尚未经过测试。

配方中的代码通常需要使用各种Python库。这些都可以使用`pip`进行安装，并且可以使用`pip install`进行访问。在需要的地方，这些安装将在配方中详细说明。

有几个配方需要亚马逊AWS账户。AWS账户在第一年可以免费使用免费层服务。配方不需要比免费层服务更多的东西。可以在[https://portal.aws.amazon.com/billing/signup](https://portal.aws.amazon.com/billing/signup)上创建一个新账户。

几个食谱将利用Elasticsearch。GitHub上有一个免费的开源版本，网址是[https://github.com/elastic/elasticsearch](https://github.com/elastic/elasticsearch)，该页面上有安装说明。Elastic.co还提供了一个完全功能的版本（还带有Kibana和Logstash），托管在云上，并提供为期14天的免费试用，网址是[http://info.elastic.co](http://info.elastic.co)（我们将使用）。还有一个docker-compose版本，具有所有x-pack功能，网址是[https://github.com/elastic/stack-docker](https://github.com/elastic/stack-docker)，所有这些都可以通过简单的`docker-compose up`命令启动。

最后，一些食谱使用MySQL和PostgreSQL作为数据库示例，以及这些数据库的几个常见客户端。对于这些食谱，这些都需要在本地安装。 MySQL Community Server可在[https://dev.mysql.com/downloads/mysql/](https://dev.mysql.com/downloads/mysql/)上找到，而PostgreSQL可在[https://www.postgresql.org/](https://www.postgresql.org/)上找到。

我们还将研究创建和使用多个食谱的docker容器。 Docker CE是免费的，可在[https://www.docker.com/community-edition](https://www.docker.com/community-edition)上获得。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，文件将直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在GitHub上，网址是[https://github.com/PacktPublishing/Python-Web-Scraping-Cookbook](https://github.com/PacktPublishing/Python-Web-Scraping-Cookbook)。我们还有其他代码包，来自我们丰富的书籍和视频目录，可在**[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟URL、用户输入和Twitter句柄。这是一个例子：“这将循环遍历多达20个字符，并将它们放入`sw`索引中，文档类型为`people`”

代码块设置如下：

```py
from elasticsearch import Elasticsearch
import requests
import json

if __name__ == '__main__':
    es = Elasticsearch(
        [
```

任何命令行输入或输出都按如下方式编写：

```py
$ curl https://elastic:tduhdExunhEWPjSuH73O6yLS@7dc72d3327076cc4daf5528103c46a27.us-west-2.aws.found.io:9243
```

**粗体**：表示一个新术语、一个重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会出现在文本中。这是一个例子：“从管理面板中选择系统信息。”

警告或重要说明会出现在这样的地方。提示和技巧会出现在这样的地方。
