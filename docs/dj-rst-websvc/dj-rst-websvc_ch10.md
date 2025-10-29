# 前言

Python 无疑是最受欢迎的编程语言之一。它是开源的、多平台的，您可以使用它来开发任何类型的应用程序，从网站和 Web 服务到人工智能和机器学习应用程序。您总会找到在 Python 中简化任何领域工作的框架或一系列包。在最重要的和最受欢迎的云计算提供商中，与 Python 及其最受欢迎的 Web 框架 Django 一起工作极为容易。因此，Python 是开发将在云上运行的现代和可扩展的 RESTful Web 服务的绝佳选择。

REST（代表表示状态传输）是近年来推动现代和可扩展网络开发的架构风格。如果您想成为构建复杂网络应用和移动应用的世界的一部分，您将需要开发和交互 RESTful Web 服务。在许多情况下，您将不得不从头开始设计和开发一个 RESTful Web 服务，并在一段时间内维护该 API。对 RESTful Web 服务的深入了解是任何软件开发工作的必备技能。

本书涵盖了您需要了解的一切，以从头开始使用最新版本的 Django、Django REST 框架和 Python 开发和测试 RESTful Web 服务。您将结合 Python 包的实际示例来简化任务。

您将学会使用大量工具来测试和开发统一、高质量和可扩展的 RESTful Web 服务。您将使用面向对象编程和现代 Python 3.6 代码来促进代码重用并简化未来的维护。您将利用自动化测试来确保编码的 RESTful Web 服务按预期运行。

这本书将使您能够使用 Python 3.6 或更高版本中的 Django 和 Django REST 框架为任何领域创建自己的 RESTful Web 服务。您将学习在最受欢迎的 Python 平台上的流程：Linux、Windows 和 macOS。

# 本书面向对象

本书面向想要使用 Python 3.6 或更高版本开发 RESTful Web 服务（也称为 RESTful 网络 API）的 Python 开发者，并希望学习如何使用最受欢迎的 Python 网络框架——Django。

# 本书涵盖内容

第一章，*安装所需的软件和工具*，展示了如何开始使用 Python 及其最受欢迎的 Web 框架 Django 创建 RESTful Web Service 的旅程。我们将安装和配置创建 Django 和 Django REST framework RESTful Web Service 所需的环境、软件和工具。我们将学习在 Linux、macOS 和 Windows 上必要的步骤。我们将使用 Django 创建我们的第一个应用程序，我们将首次查看 Django 文件夹、文件和配置，并将进行必要的更改以激活 Django REST framework。此外，我们将介绍和安装我们将用于与我们在后续章节中设计、编码和测试的 RESTful Web Service 交互的命令行和 GUI 工具。

第二章，*使用模型、迁移、序列化和反序列化*，介绍了如何设计一个 RESTful Web Service 以与简单的 SQLite 数据库交互，并使用玩具执行 CRUD 操作。我们将定义我们 Web 服务的需求，并理解每个 HTTP 方法执行的任务和不同的范围。我们将创建一个模型来表示和持久化玩具，并在 Django 中执行迁移以在数据库中创建所需的表。我们将分析这些表，并学习如何使用 Django REST 框架管理玩具实例到 JSON 表示的序列化以及反向过程。

第三章，*创建 API 视图*，是关于执行一个简单的 Django RESTful Web Service 的第一版本，该服务与 SQLite 数据库交互。我们将编写 API 视图来处理对玩具集合和特定玩具的多种 HTTP 请求。我们将使用以下 HTTP 动词：GET、POST 和 PUT。我们将配置 URL 模式列表以将 URL 路由到视图。我们将启动 Django 开发服务器，并使用命令行工具（curl 和 HTTPie）来构建并发送各种 HTTP 请求到我们的 RESTful Web Service。我们将学习 HTTP 请求在 Django 和我们的代码中是如何处理的。此外，我们将使用 Postman，一个 GUI 工具，来构建并发送其他 HTTP 请求到我们的 RESTful Web Service。

第四章，*从 APIView 类使用通用行为*，介绍了提高我们简单 Django RESTful Web Service 的不同方法。我们将利用 Django REST 框架中包含的许多功能来删除重复代码并为 Web 服务添加许多功能。我们将使用模型序列化器，了解接受和返回的不同内容类型，以及为 HTTP OPTIONS 请求提供准确响应的重要性。我们将对现有代码进行必要的修改，以启用多种解析器和渲染器。我们将学习在 Django REST 框架中内部的工作原理。我们将处理不同的内容类型，并注意与之前版本相比，RESTful Web Service 的改进。

第五章，*理解和自定义可浏览 API 功能*，解释了如何使用 Django REST 框架添加到我们的 RESTful Web Service 中的一个附加功能——可浏览的 API。我们将使用网络浏览器来处理我们用 Django 构建的第一个 Web 服务。我们将学习如何使用可浏览的 API 进行 HTTP GET、POST、PUT、OPTIONS 和 DELETE 请求。我们将能够轻松地使用网络浏览器测试 CRUD 操作。可浏览的 API 将使我们能够轻松地与我们的 RESTful Web Service 进行交互。

第六章，*与高级关系和序列化一起工作*，展示了如何定义一个复杂 RESTful Web Service 的需求，其中我们需要处理无人机类别、无人机、飞行员和比赛。我们将使用 Django 创建一个新应用并配置新的 Web 服务。我们将定义模型之间的多对一关系，并将 Django 配置为与 PostgreSQL 数据库一起工作。我们将执行迁移以生成具有相互关系表。我们还将分析生成的数据库并为模型配置序列化和反序列化。我们将定义超链接并使用基于类的视图进行工作。然后，我们将利用通用类和通用视图，这些视图将通用和混合预定义的行为。我们将使用路由和端点，并准备我们的 RESTful Web Service 以与可浏览的 API 一起工作。我们将进行许多不同的 HTTP 请求来创建和检索具有相互关系资源。

第七章，*使用约束、过滤、搜索、排序和分页*，描述了如何使用可浏览的 API 功能通过资源和关系在 API 中进行导航。我们将添加唯一约束来提高我们 RESTful Web Service 中模型的一致性。我们将理解分页结果的重要性，并使用 Django REST 框架配置和测试一个全局限制/偏移分页方案。然后，我们将创建自己的定制分页类，以确保请求不会要求在单页上显示大量元素。我们将配置过滤器后端类，并将代码集成到模型中，以向基于类的视图添加过滤、搜索和排序功能。我们将创建一个定制的过滤器，并发送请求以过滤、搜索和排序结果。最后，我们将使用可浏览的 API 来测试分页、过滤和排序。

第八章，*使用认证和权限保护 API*，展示了在 Django、Django REST 框架和 RESTful Web 服务中认证和权限之间的区别。我们将分析 Django REST 框架中包含的内置认证类。我们将遵循向模型提供安全和权限相关数据的步骤。

我们将通过定制的权限类处理对象级权限，并保存有关发起请求的用户的信息。我们将配置权限策略，并组合并发送认证请求以了解权限策略的工作方式。我们将使用命令行工具和 GUI 工具来组合并发送认证请求。我们将使用可浏览的 API 功能浏览安全的 RESTful Web 服务，并使用 Django REST 框架提供的简单基于令牌的认证来了解另一种认证请求的方式。

*第九章*，*应用节流规则和版本管理*，重点关注节流规则的重要性以及我们如何将它们与 Django、Django REST 框架和 RESTful Web 服务中的认证和权限相结合。我们将分析 Django REST 框架中包含的内置节流类。我们将遵循在 Django REST 框架中配置多个节流策略的必要步骤。我们将处理全局和范围相关的设置。然后，我们将使用命令行工具来组合并发送多个请求以测试节流规则的工作方式。我们将理解版本类，并配置一个 URL 路径版本控制方案，以便我们能够处理我们 RESTful Web 服务的两个版本。我们将使用命令行工具和可浏览的 API 来了解两个版本之间的差异。

第十章，*自动化测试*，展示了如何自动化使用 Django 和 Django REST 框架开发的 RESTful Web 服务的测试。我们将使用不同的包、工具和配置来执行测试。我们将为 RESTful Web 服务编写第一轮单元测试，运行它们，并测量测试代码覆盖率。然后，我们将分析测试代码覆盖率报告，并编写新的单元测试以提高测试代码覆盖率。我们将理解新的测试代码覆盖率报告，并学习良好测试代码覆盖率的好处。

附录，*解决方案*，包含每章“测试你的知识”部分的正确答案。这些内容包含在附录中。

# 要充分利用本书

任何能够运行 Python 3.6.3 或更高版本的 Linux、macOS 或 Windows 的计算机或设备。

任何能够运行与现代 Web 浏览器兼容的 HTML 5 和 CSS 3 的计算机或设备，都可以用于与 Django REST 框架中包含的可浏览 API 功能一起工作。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择支持选项卡。

1.  点击代码下载和勘误表。

1.  在搜索框中输入本书的名称，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Django-RESTful-Web-Services`](https://github.com/PacktPublishing/Django-RESTful-Web-Services/)。我们还有其他来自我们丰富图书和视频目录的代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/DjangoRESTfulWebServices_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/Bookname_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“将下载的`WebStorm-10*.dmg`磁盘映像文件作为系统中的另一个磁盘挂载。”

代码块设置如下：

```py
from django.shortcuts import render 

# Create your views here.
```

当我们希望您注意代码块中的特定部分时，相关的行或项目会以粗体显示：

```py
from django.conf.urls import url, include

urlpatterns = [
    url(r'^', include('drones.urls')),
 url(r'^api-auth/', include('rest_framework.urls')) ]
```

任何命令行输入或输出都应如下所示：

```py
   http :8000/toys/
   curl -iX GET localhost:8000/toys/3
```

**粗体**: 表示新术语、重要单词或屏幕上出现的单词。例如，菜单或对话框中的单词在文本中会这样显示。例如：“从管理面板中选择系统信息。”

警告或重要注意事项会像这样显示。

技巧和窍门会像这样显示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**: 请发送电子邮件至 `feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请发送电子邮件至 `questions@packtpub.com`。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告，我们将非常感谢。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果您在互联网上以任何形式遇到我们作品的非法副本，我们将不胜感激，如果您能提供位置地址或网站名称，我们将非常感谢。请通过 `copyright@packtpub.com` 联系我们，并提供材料的链接。

**如果您有兴趣成为作者**: 如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用过这本书，为何不在购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问：[packtpub.com](https://www.packtpub.com/). [packtpub.com](https://www.packtpub.com/)
