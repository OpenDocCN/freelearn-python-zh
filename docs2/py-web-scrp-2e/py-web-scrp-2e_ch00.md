# 前言

互联网包含了有史以来最有用的数据集，大部分数据都可以免费公开访问。然而，这些数据并不容易重用。它们嵌入在网站的结构和风格中，需要提取出来才能变得有用。从网页中提取数据的过程被称为*网络爬虫*，随着越来越多的信息在线上可用，它变得越来越有用。

所有使用的代码都已用 Python 3.4+测试，可在[`github.com/kjam/wswp`](https://github.com/kjam/wswp)下载。

# 这本书涵盖了什么

第一章，*网络爬虫简介*，介绍了什么是网络爬虫以及如何爬取网站。

第二章，*爬取数据*，展示了如何使用几个库从网页中提取数据。

第三章，*缓存下载*，教授如何通过缓存结果来避免重新下载。

第四章，*并发下载*，指导你如何通过并行下载网站来加快数据爬取速度。

第五章，*动态内容*，通过多种方式学习如何从动态网站中提取数据。

第六章，*与表单交互*，展示了如何处理输入和导航等表单，以进行搜索和登录。

第七章，*解决验证码*，详细阐述了如何访问受 CAPTCHA 图像保护的资料。

第八章，*Scrapy*，学习如何使用 Scrapy 爬虫蜘蛛进行快速并行爬取，以及使用 Portia 网络界面构建网络爬虫。

第九章，*综合运用*，概述了通过本书学习到的网络爬虫技术。

# 你需要这本书的内容

为了帮助说明爬取示例，我们创建了一个示例网站[`example.webscraping.com`](http://example.webscraping.com)。用于生成此网站的源代码可在[`bitbucket.org/WebScrapingWithPython/website`](http://bitbucket.org/WebScrapingWithPython/website)找到，其中包含了如果你愿意自己托管网站的说明。

我们决定为示例构建一个自定义网站，而不是爬取实时网站，这样我们可以完全控制环境。这为我们提供了稳定性——实时网站更新比书籍更频繁，等你尝试爬取示例时，它可能已经不再工作。此外，自定义网站允许我们制作展示特定技能的示例，避免干扰。最后，实时网站可能不欢迎我们使用它们来学习网络爬虫，并可能阻止我们的爬虫。使用我们自己的自定义网站可以避免这些风险，然而在这些示例中学到的技能当然也可以应用于实时网站。

# 这本书面向的对象

本书假设您有先前的编程经验，可能不适合完全初学者。网络爬取示例需要具备 Python 和 pip 安装模块的能力。如果您需要复习，Mark Pilgrim 有一本出色的免费在线书籍，可在[`www.diveintopython.net`](http://www.diveintopython.net)找到。这是我最初学习 Python 时使用的资源。

这些示例还假设您了解网页如何使用 HTML 构建以及如何使用 JavaScript 更新。对 HTTP、CSS、AJAX、WebKit 和 Redis 的了解也会很有用，但不是必需的，并且将在需要时介绍。许多这些主题的详细参考资料可在[`developer.mozilla.org/`](https://developer.mozilla.org/)找到。

# 约定

在这本书中，您将找到许多不同风格的文本，以区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词显示如下：“我们可以通过使用`include`指令来包含其他上下文。”

代码块设置为如下：

```py
from urllib.request import urlopen
from urllib.error import URLError

url = 'http://example.webscraping.com'
try:
    html = urlopen(url).read()
except urllib2.URLError as e:
    html = None

```

任何命令行输入或输出将按以下方式编写：

```py
python script.py

```

我们将偶尔显示正常 Python 解释器使用的 Python 解释器提示，例如：

```py
>>> import urllib

```

或者 IPython 解释器，例如：

```py
In [1]: import urllib

```

**新术语**和**重要词汇**将以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，将以文本中的这种方式显示：“点击“下一步”按钮将您移动到下一屏幕”。

警告或重要说明将以这样的框显示。

小贴士和技巧将以这样的形式出现。

# 读者反馈

我们读者的反馈总是受欢迎的。告诉我们您对这本书的看法——您喜欢什么或可能不喜欢什么。读者反馈对我们开发您真正从中获得最大收益的标题非常重要。

要发送给我们一般性的反馈，只需发送一封电子邮件到`feedback@packtpub.com`，并在邮件的主题中提及书名。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您已经是 Packt 图书的骄傲拥有者，我们有一些事情可以帮助您从购买中获得最大收益。

# 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的“支持”选项卡上。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名。

1.  选择您想要下载代码文件的书籍。

1.  从下拉菜单中选择您购买此书的来源。

1.  点击“代码下载”。

您也可以通过点击 Packt Publishing 网站上书籍网页上的“代码文件”按钮来下载代码文件。您可以通过在搜索框中输入书籍名称来访问此页面。请注意，您需要登录到您的 Packt 账户。

文件下载完成后，请确保您使用最新版本解压或提取文件夹：

+   Windows 版本的 WinRAR / 7-Zip

+   Mac 版本的 Zipeg / iZip / UnRarX

+   Linux 版本的 7-Zip / PeaZip

本书代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Python-Web-Scraping-Second-Edition`](https://github.com/PacktPublishing/Python-Web-Scraping-Second-Edition)。我们还有其他来自我们丰富图书和视频目录的代码包，可在 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/) 找到。查看它们吧！

# 勘误表

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告，我们将不胜感激。这样做可以帮助其他读者避免挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入您的勘误详情。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误表部分。

要查看之前提交的勘误表，请访问 [`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在勘误表部分。

# 盗版

互联网上版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上遇到任何形式的非法复制我们的作品，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `copyright@packtpub.com` 联系我们，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面的帮助。

# 问题

如果您在这本书的任何方面遇到问题，您可以通过 `questions@packtpub.com` 联系我们，我们将尽力解决问题。
