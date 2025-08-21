# 前言

网页抓取是许多组织中使用的一种重要技术，用于从网页中抓取有价值的数据。网页抓取是为了从网站中提取和收集数据而进行的。网页抓取在模型开发中非常有用，这需要实时收集数据。它也适用于真实且与主题相关的数据，其中准确性是短期内所需的，而不是实施数据集。收集的数据存储在包括 JSON、CSV 和 XML 在内的文件中，也写入数据库以供以后使用，并作为数据集在线提供。本书将为您打开网页抓取技术和方法的大门，使用 Python 库和其他流行工具，如 Selenium。通过本书，您将学会如何高效地抓取不同的网站。

# 本书适合对象

这本书适用于 Python 程序员、数据分析师、网页抓取新手以及任何想要学习如何从头开始进行网页抓取的人。如果你想开始学习如何将网页抓取技术应用于各种网页，那么这本书就是你需要的！

# 本书内容

第一章，网页抓取基础知识，探讨了一些与 WWW 相关的核心技术和工具，这些技术和工具对网页抓取是必需的。

第二章，Python 和 Web-使用 URLlib 和 Requests，演示了 Python 库中可用的一些核心功能，如`requests`和`urllib`，并探索了各种格式和结构的页面内容。

第三章，使用 LXML、XPath 和 CSS 选择器，描述了使用 LXML 的各种示例，实现了处理元素和 ElementTree 的各种技术和库特性。

第四章，使用 pyquery 进行抓取-一个 Python 库，更详细地介绍了网页抓取技术和一些部署这些技术的新 Python 库。

第五章，使用 Scrapy 和 Beautiful Soup 进行网页抓取，检查了使用 Beautiful Soup 遍历网页文档的各个方面，同时还探索了一个专为使用蜘蛛进行爬行活动而构建的框架，换句话说，Scrapy。

第六章，处理安全网页，涵盖了许多常见的基本安全措施和技术，这些措施和技术经常遇到，并对网页抓取构成挑战。

第七章，使用基于 Web 的 API 进行数据提取，涵盖了 Python 编程语言以及如何与 Web API 交互以进行数据提取。

第八章，使用 Selenium 进行网页抓取，涵盖了 Selenium 以及如何使用它从网页中抓取数据。

第九章，使用正则表达式提取数据，更详细地介绍了使用正则表达式进行网页抓取技术。

第十章，下一步，介绍并探讨了使用文件进行数据管理，使用 pandas 和 matplotlib 进行分析和可视化的基本概念，同时还介绍了机器学习和数据挖掘，并探索了一些相关资源，这些资源对进一步学习和职业发展都有帮助。

# 充分利用本书

读者应该具有一定的 Python 编程语言工作知识。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在“搜索”框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Web-Scraping-with-Python`](https://github.com/PacktPublishing/Hands-On-Web-Scraping-with-Python)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有来自丰富书籍和视频目录的其他代码包，可以在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。去看看吧！

# 下载彩色图片

我们还提供了一份 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789533392_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789533392_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子：“`<p>`和`<h1>` HTML 元素包含与它们一起的一般文本信息（元素内容）。”

代码块设置如下：

```py
import requests
link="http://localhost:8080/~cache"

queries= {'id':'123456','display':'yes'}

addedheaders={'user-agent':''}
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
import requests
link="http://localhost:8080/~cache"

queries= {'id':'123456','display':'yes'}

addedheaders={'user-agent':''}
```

任何命令行输入或输出都以以下方式编写：

```py
C:\> pip --version

pip 18.1 from c:\python37\lib\site-packages\pip (python 3.7)
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这样的方式出现在文本中。这是一个例子：“如果通过 Chrome 菜单访问开发者工具，请单击更多工具|开发者工具”

警告或重要说明会以这种方式出现。提示和技巧会以这种方式出现。
