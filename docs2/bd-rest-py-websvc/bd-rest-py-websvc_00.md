# 前言

**REST**（**表征状态转移**）是推动现代 Web 开发和移动应用架构风格。实际上，开发和使用 RESTful Web 服务是任何现代软件开发工作必备的技能。有时，你必须与现有的 API 交互，在其他情况下，你必须从头设计一个 RESTful API 并使其与**JSON**（**JavaScript 对象表示法**）兼容。

Python 是最受欢迎的编程语言之一。Python 3.5 是 Python 最现代的版本。它是开源的、多平台的，你可以用它来开发任何类型的应用程序，从网站到极其复杂的科学计算应用程序。总有一个 Python 包可以使事情变得更容易，避免重复造轮子并更快地解决问题。最重要的和最受欢迎的云计算提供商使得使用 Python 及其相关的 Web 框架变得容易。因此，Python 是开发 RESTful Web 服务的理想选择。本书涵盖了你需要知道的所有内容，以选择最合适的 Python Web 框架并从头开始开发 RESTful API。

你将使用三个最受欢迎的 Python Web 框架，这些框架使得开发 RESTful Web 服务变得简单：Django、Flask 和 Tornado。每个 Web 框架都有其优势和权衡。你将使用代表这些 Web 框架适当案例的示例，结合额外的 Python 包来简化最常见的任务。你将学习使用不同的工具来测试和开发高质量、一致且可扩展的 RESTful Web 服务。你还将利用面向对象编程（也称为 OOP）来最大化代码重用并最小化维护成本。

你将始终为书中开发的每个 RESTful Web 服务编写单元测试并提高测试覆盖率。你不仅会运行示例代码，还会确保为你的 RESTful API 编写测试。

这本书将帮助你学习如何利用许多简化与 RESTful Web 服务相关常见任务的软件包。你将能够开始为任何领域创建自己的 RESTful API，这些 API 可以在 Python 3.5 或更高版本的任何覆盖的 Web 框架中实现。

# 本书涵盖内容

第一章，*使用 Django 开发 RESTful API*，在本章中，我们将开始使用 Django 和 Django REST 框架，并创建一个对简单 SQLite 数据库执行**CRUD**（**创建、读取、更新和删除**）操作的 RESTful Web API。

第二章, *在 Django 中使用基于类的视图和超链接 API*，在本章中，我们将扩展上一章中开始构建的 RESTful API 的功能。我们将更改 ORM 设置以使用更强大的 PostgreSQL 数据库，并利用 Django REST Framework 中包含的先进功能，这些功能允许我们减少复杂 API（如基于类的视图）的样板代码。

第三章, *在 Django API 中改进和添加认证*，在本章中，我们将改进上一章中开始构建的 RESTful API。我们将向模型添加唯一约束并更新数据库。我们将通过 PATCH 方法简化单个字段的更新，并利用分页功能。我们将开始处理认证、权限和限制。

第四章, *使用 Django REST Framework 限制、过滤、测试和部署 API*，在本章中，我们将利用 Django REST Framework 中包含的许多功能来定义限制策略。我们将使用过滤、搜索和排序类来简化配置过滤器、搜索查询和结果排序的 HTTP 请求。我们将使用可浏览的 API 功能来测试我们 API 中包含的新特性。我们将编写第一轮单元测试，测量测试覆盖率，然后编写额外的单元测试以提高测试覆盖率。最后，我们将学习许多关于部署和可扩展性的考虑因素。

第五章 , *使用 Flask 开发 RESTful API*，在本章中，我们将开始使用 Flask 及其 Flask-RESTful 扩展。我们将创建一个执行简单列表 CRUD 操作的 RESTful Web API。

第六章, *在 Flask 中使用模型、SQLAlchemy 和超链接 API*，在本章中，我们将扩展上一章中开始构建的 RESTful API 的功能。我们将使用 SQLAlchemy 作为我们的 ORM 来与 PostgreSQL 数据库交互，并利用 Flask 和 Flask-RESTful 中包含的先进功能，这些功能将使我们能够轻松组织复杂 API（如模型和蓝图）的代码。

第七章, *使用 Flask 改进和添加认证到 API*，在本章中，我们将从多个方面改进 RESTful API。当资源不唯一时，我们将添加用户友好的错误消息。我们将测试如何使用 PATCH 方法更新单个或多个字段，并创建我们自己的通用分页类。然后，我们将开始处理认证和权限。我们将添加用户模型并更新数据库。我们将对代码的不同部分进行许多更改以实现特定的安全目标，并利用 Flask-HTTPAuth 和 passlib 在我们的 API 中使用 HTTP 认证。

第八章, *使用 Flask 测试和部署 API*，在本章中，我们将设置测试环境。我们将安装 nose2 以简化单元测试的发现和执行，并创建一个新的数据库用于测试。我们将编写第一轮单元测试，测量测试覆盖率，然后编写额外的单元测试以提高测试覆盖率。最后，我们将学习许多关于部署和可扩展性的考虑因素。

第九章,*使用 Tornado 开发 RESTful API*，我们将与 Tornado 一起创建一个 RESTful Web API。我们将设计一个 RESTful API 来与慢速传感器和执行器交互。我们将定义 API 的需求，并理解每个 HTTP 方法执行的任务。我们将创建代表无人机的类，并编写代码来模拟每个 HTTP 请求方法所需的慢速 I/O 操作。我们将编写代表请求处理器的类，处理不同的 HTTP 请求，并配置 URL 模式将 URL 路由到请求处理器及其方法。

第十章, *使用 Tornado 处理异步代码、测试和部署 API*，在本章中，我们将了解同步执行和异步执行之间的区别。我们将创建一个利用 Tornado 的非阻塞特性结合异步执行的 RESTful API 新版本。我们将提高现有 API 的可扩展性，并使其在等待传感器和执行器的慢速 I/O 操作时能够启动执行其他请求。然后，我们将设置测试环境。我们将安装 nose2 以简化单元测试的发现和执行，并创建一个新的数据库用于测试。我们将编写第一轮单元测试，测量测试覆盖率，然后编写额外的单元测试以提高测试覆盖率。我们将创建所有必要的测试，以确保对所有代码行的全面覆盖。

# 你需要这本书的

为了使用 Python 3.5.x 的不同样本，您需要任何配备 Intel Core i3 或更高 CPU 以及至少 4 GB RAM 的计算机。您可以使用以下任何一种操作系统：

+   Windows 7 或更高版本（Windows 8、Windows 8.1 或 Windows 10）

+   macOS Mountain Lion 或更高版本

+   任何能够运行 Python 3.5.x 的 Linux 版本以及任何支持 JavaScript 的现代浏览器

您需要在计算机上安装 Python 3.5 或更高版本。

# 本书面向对象

这本书是为那些对 Python 有一定了解并希望通过利用 Python 的各种框架来构建令人惊叹的 Web 服务而编写的 Web 开发者。您应该对 RESTful API 有一些了解。

# 惯例

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号将以以下方式显示：“如果没有游戏匹配指定的 id 或主键，服务器将返回 `404 Not Found` 状态。”

代码块将如下设置：

```py
from django.apps import AppConfig
class GamesConfig(AppConfig):
    name = 'games'
```

任何命令行输入或输出将如下所示：

```py
python3 -m venv ~/PythonREST/Django01
```

### 注意

警告或重要注意事项将以这样的框显示。

### 小贴士

小贴士和技巧将以这样的形式显示。

# 读者反馈

我们欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢或不喜欢什么。读者反馈对我们来说很重要，因为它帮助我们开发出您真正能从中获得最大收益的标题。要发送一般反馈，请简单地发送电子邮件至 feedback@packtpub.com，并在邮件主题中提及书籍的标题。如果您在某个主题领域有专业知识，并且对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经是 Packt 书籍的骄傲拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

## 下载示例代码

您可以从您的 [`www.packtpub.com`](http://www.packtpub.com) 账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [`www.packtpub.com/support`](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的 **支持** 选项卡上。

1.  点击 **代码下载与勘误**。

1.  在 **搜索** 框中输入书籍名称。

1.  选择您想要下载代码文件的书籍。

1.  从下拉菜单中选择您购买此书的来源。

1.  点击 **代码下载**。

文件下载后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   Windows 的 WinRAR / 7-Zip

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Building-RESTful-Python-Web-Services`](https://github.com/PacktPublishing/Building-RESTful-Python-Web-Services)。我们还有其他来自我们丰富图书和视频目录的代码包，可在 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/) 找到。查看它们吧！

## 勘误

尽管我们已经尽一切努力确保我们内容的准确性，但错误仍然会发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问 [`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将在**勘误**部分显示。

## 盗版

互联网上版权材料的盗版是一个持续存在的问题，涉及所有媒体。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 copyright@packtpub.com 联系我们，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面提供的帮助。

## 问题

如果您对本书的任何方面有问题，您可以通过 questions@packtpub.com 联系我们，我们将尽力解决问题。
