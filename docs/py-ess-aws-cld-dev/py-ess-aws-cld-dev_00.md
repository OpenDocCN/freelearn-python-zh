# 前言

云计算是实现应用程序最受欢迎的方法之一，具有巨大的优势。有多个云服务提供商，例如 AWS、GCP 和 Azure。AWS 是最常用的云服务提供商之一，许多公司正在迁移到那里。云的使用量显著增长，开发者需要具备云知识。

大多数应用程序正在迁移到云端。AWS 提供了不同的服务来实现 Python 应用程序，因此对于没有 AWS 背景的人来说，配置和选择正确的服务是一个挑战。通过购买本书，您将走上正确的道路，并开始学习如何使用 AWS 服务实现酷炫的 Python 应用程序。

# 本书面向的对象

本书是为云计算开发者、软件开发人员和打算在 AWS 上开发 Python 应用程序并了解适当 AWS 服务概念的 IT 专业人士编写的。您应该具备 Python 编程经验，以便在 AWS 上实现应用程序。

# 本书涵盖的内容

*第一章*, *在 AWS 上使用 Python*。本章将教您如何安装和使用 Python IDE，并了解 AWS 云的优势。

*第二章*, *创建 AWS 账户*。要开始云计算，AWS 需要一个账户来实现 Python 编程。在本章中，您将学习如何创建 AWS 账户。

*第三章*, *使用 Lambda 进行云计算*。Lambda 是实现 Python 函数的一种非常有效的方法。本章将帮助您了解 Lambda 服务，并展示如何实现代码。

*第四章*, *在 EC2 上运行 Python 应用程序*。EC2 是您可以在云上配置的关键服务之一。本章将帮助您了解 EC2 服务，并展示如何配置服务器以及随后部署 Python 应用程序。

*第五章*, *使用 PyCharm 运行 Python 应用程序*。调试 Python 应用程序对于测试应用程序非常重要。本章将帮助您以简单的方式在本地调试 Python 应用程序。

*第六章*, *在 Elastic Beanstalk 上部署 Python 应用程序*。Elastic Beanstalk 是一个有用的服务，允许部署应用程序。本章将帮助您了解 Elastic Beanstalk 服务，并展示如何创建服务以及随后部署 Python 应用程序。

*第七章*, *通过 CloudWatch 监控应用程序*。CloudWatch 允许您在 AWS 中监控您的应用程序。本章将帮助您了解 CloudWatch 服务，并展示如何监控 Python 应用程序。

*第八章*，*使用 RDS 进行数据库操作*。RDS 用于在 AWS 中创建数据库。本章将帮助您了解 RDS 服务，并展示如何通过 Python 应用程序创建数据库和执行 SQL 操作。

*第九章*，*在 AWS 中创建 API*。API 是应用程序的重要接口。本章将帮助您在 AWS 中创建 API 并将 API 发布以访问 Python 应用程序。

*第十章*，*使用 Python 与 NoSQL（DynamoDB）*。NoSQL 用于存储非结构化和半结构化数据。本章将帮助您创建 NoSQL 数据库并在 DynamoDB 上执行 SQL 操作。

*第十一章*，*使用 Python 与 Glue*。Glue 是 AWS 中的无服务器数据集成服务。本章将帮助您将 Python 应用程序嵌入到 Glue 服务中。

*第十二章*，*AWS 参考项目*。实施一个示例项目是了解应用程序编程的最佳方式。本章将帮助您使用最佳实践实施示例 AWS 项目。

# 要充分利用本书

您需要了解 Python 编程语言的基础知识，才能在 AWS 上实施应用程序。

| **书中涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Python | Windows、macOS 或 Linux |
| **亚马逊网络服务**（**AWS**） |  |

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件[`github.com/PacktPublishing/Python-Essentials-for-AWS-Cloud-Developers`](https://github.com/PacktPublishing/Python-Essentials-for-AWS-Cloud-Developers)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图和图表彩色图像的 PDF 文件。您可以从这里下载：[`packt.link/hWfW6`](https://packt.link/hWfW6)

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“从命令行执行`python --version`。”

代码块按以下方式设置：

```py
from flask import Flask
app = Flask(__name__)
@app.route('/')
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
from flask import Flask
app = Flask(__name__)
@app.route('/')
```

任何命令行输入或输出都按以下方式编写：

```py
wget https://raw.githubusercontent.com/PacktPublishing/Python-Essentials-for-AWS-Cloud-Developers/main/fileprocessor.py
```

**粗体**: 表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以**粗体**显示。以下是一个例子：“在左侧点击**实例**，然后点击**启动实例**。”

小贴士或重要提示

它看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**: 如果您对本书的任何方面有疑问，请通过电子邮件发送至 customercare@packtpub.com，并在邮件主题中提及书名。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**: 如果您在互联网上发现我们作品的任何非法副本，我们将不胜感激，如果您能提供位置地址或网站名称，请通过电子邮件联系 copyright@packt.com 并提供材料链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读了*Python Essentials for AWS Cloud Developers*，我们非常乐意听到您的想法！请[点击此处直接进入本书的亚马逊评论页面](https://packt.link/r/1804610062)并分享您的反馈。

您的评论对我们和科技社区非常重要，并将帮助我们确保我们提供高质量的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

您喜欢在路上阅读，但无法携带您的印刷书籍到处走？您的电子书购买是否与您选择的设备不兼容？

不要担心，现在每本 Packt 书籍都附赠一本无 DRM 的 PDF 版本，无需额外费用。

在任何地方、任何设备上阅读。直接从您最喜欢的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

优惠远不止这些，您还可以获得独家折扣、时事通讯和每天收件箱中的优质免费内容。

按照以下简单步骤获取优惠：

1.  扫描二维码或访问以下链接

![二维码](img/B19195_QR_Free_PDF.jpg)

https://packt.link/free-ebook/9781804610060

1.  提交您的购买证明

1.  就这些！我们将直接将您的免费 PDF 和其他优惠发送到您的电子邮件。

# 第一部分：Python 安装和云

在本部分，您将学习如何安装和使用 Python IDE，并了解云基础知识。为了通过 Python 编程在 AWS 中进入云计算，我们还将开设一个 AWS 账户。

本部分包含以下章节：

+   *第一章*，*在 AWS 上使用 Python*

+   *第二章*, *创建 AWS 账户*
