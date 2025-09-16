# 前言

《*FastAPI 食谱*》是希望掌握 FastAPI 框架以构建 API 的 Python 开发者的宝贵资源。由 Sebastián Ramírez Montaño 创建，FastAPI 首次发布于 2018 年 12 月。它迅速获得了人气，并成为构建 API 最广泛使用的 Python 框架之一。

本书首先介绍 FastAPI，展示其优势，并帮助你设置开发环境。然后，它转向数据处理，展示数据库集成和**创建、读取、更新和删除**（**CRUD**）操作，以帮助你有效地在 API 中管理数据。

随着本书的深入，它探讨了如何创建**RESTful** API，涵盖了高级主题，如复杂查询、版本控制和广泛的文档。安全性同样重要，本书有一章专门介绍实现认证机制，例如**OAuth2**和**JWT**令牌，以保护 FastAPI 应用程序的安全。

测试是开发的重要组成部分，本书提供了确保 FastAPI 应用程序质量和可靠性的策略。讨论了部署策略，强调了生产环境中的最佳实践。对于高流量应用程序，本书探讨了扩展技术以提高性能。

通过中间件扩展 FastAPI 的功能是可能的，本书还展示了如何通过将其与其他 Python 工具和框架集成来增强 FastAPI 的能力，以适应机器学习模型并公开**LLM** **RAG**应用程序。

实时通信通过**WebSockets**章节进行处理，并提供了高级数据处理技术来管理大量数据集和文件管理。

本书以使用 FastAPI 处理现实世界流量结束，强调部署策略和打包发货。每一章都精心设计，以构建你的专业知识，使《*FastAPI 食谱*》成为专业级 API 开发的宝贵指南。

# 本书面向的对象

本书针对对网络开发概念有基础理解的初级到高级 Python 开发者。对于那些寻求使用现代 FastAPI 框架构建高效、可扩展 API 的人来说，本书特别有益。对于希望提高 API 开发技能并将实际解决方案应用于现实编程挑战的开发者来说，本书是宝贵的资源。无论你是想保护 API、有效管理数据还是优化性能，本书都提供了知识和动手示例，以提升你在 FastAPI 方面的专业知识。

# 本书涵盖的内容

*第一章*，*FastAPI 的第一步*，作为框架的介绍，强调其速度、易用性和全面的文档。这一章是你设置开发环境、创建第一个 FastAPI 项目并探索其基本概念的入门。

*第二章*, *处理数据*，致力于掌握网络应用程序中数据处理的关键方面。它涵盖了使用 SQL 和 NoSQL 数据库集成、管理和优化数据存储的复杂性。

*第三章*, *使用 FastAPI 构建 RESTful API*，深入探讨了构建 RESTful API 的基本要素，这对于网络服务至关重要，它使应用程序能够高效地通信和交换数据。

*第四章*, *身份验证和授权*，深入探讨了保护您的网络应用程序免受未经授权访问的关键领域。它涵盖了用户注册和认证的基础、将 OAuth2 协议与 JWT 集成以增强安全性，以及创建 API 的基本组件。

*第五章*, *测试和调试 FastAPI 应用程序*，转向软件开发的一个关键方面，确保您应用程序的可靠性、健壮性和质量——测试和调试。

*第六章*, *将 FastAPI 与 SQL 数据库集成*，开始了一段在 FastAPI 应用程序中充分利用 SQL 数据库潜力的旅程。它精心设计，旨在指导您利用 SQLAlchemy（一个强大的 Python SQL 工具包和**对象关系映射器**（ORM））。

*第七章*, *将 FastAPI 与 NoSQL 数据库集成*，通过指导您设置和使用 MongoDB（一个流行的 NoSQL 数据库）与 FastAPI 的过程，探讨了 FastAPI 与 NoSQL 数据库的集成。它涵盖了 CRUD 操作、使用索引进行性能优化以及处理 NoSQL 数据库中的关系。此外，本章还讨论了将 FastAPI 与 Elasticsearch 集成以实现强大的搜索功能，以及使用 Redis 实现缓存。

*第八章*, *高级特性和最佳实践*，探讨了优化 FastAPI 应用程序功能、性能和可扩展性的高级技术和最佳实践。它涵盖了依赖注入、自定义中间件、国际化、性能优化、速率限制和后台任务执行等基本主题。

*第九章*, *使用 WebSockets*，是一本全面指南，介绍了在 FastAPI 应用程序中使用 WebSockets 实现实时通信功能。它涵盖了设置 WebSocket 连接、发送和接收消息、处理连接和断开连接、错误处理以及实现聊天功能。

*第十章*, *将 FastAPI 与其他 Python 库集成*，深入探讨了 FastAPI 与外部库结合时的潜力，增强了其核心功能之外的能力。它提供了一种基于食谱的方法，将 FastAPI 与各种技术（如 Cohere 和 LangChain）集成，以构建 LLM RAG 应用。

*第十一章*, *中间件和 Webhooks*，深入探讨了 FastAPI 中中间件和 Webhooks 的先进和关键方面。中间件允许你全局处理请求和响应，而 Webhooks 则使你的 FastAPI 应用程序能够通过发送实时数据更新与其他服务进行通信。

*第十二章*, *部署和管理 FastAPI 应用程序*，涵盖了无缝部署 FastAPI 应用程序所需的知识和工具，利用各种技术和最佳实践。你将学习如何利用 FastAPI CLI 高效运行服务器，启用 HTTPS 以保护你的应用程序，以及使用 Docker 容器化你的 FastAPI 项目。

# 为了充分利用本书

你应该对 Python 编程有基本的了解，因为本书假设你对 Python 语法和概念熟悉。此外，了解 Web 开发原则，包括 HTTP、RESTful API 和 JSON，将有所帮助。熟悉 SQL 和 NoSQL 数据库，以及使用 Git 等版本控制系统的经验，将帮助你完全理解内容。

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Python 3.9 或更高版本 | Windows、macOS 或 Linux（任何） |

**如果你使用的是本书的数字版，我们建议你亲自输入代码或通过 GitHub 仓库（下一节中提供链接）访问代码。这样做将有助于避免与代码复制和粘贴相关的任何潜在错误** **。**

## 下载示例代码文件

你可以从 GitHub 下载本书的示例代码文件[`github.com/PacktPublishing/FastAPI-Cookbook`](https://github.com/PacktPublishing/FastAPI-Cookbook)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“此外，你将在应用程序自动创建的新`app.log`文件中找到我们`logger_client`的消息。”

代码块设置如下：

```py
from locust import HttpUser, task
class ProtoappUser(HttpUser):
    host = "http://localhost:8000"
    @task
    def hello_world(self):
        self.client.get("/home")
```

当我们希望将您的注意力引向代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
from pydantic import BaseModel, Field
class Book(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    author: str = Field(..., min_length=1, max_length=50)
    year: int = Field(..., gt=1900, lt=2100)
```

任何命令行输入或输出都应按照以下方式编写：

```py
$ pytest –-cov protoapp tests
```

在本书中，我们将一般使用类 Unix 终端命令。这可能会导致 Windows 系统在多行命令上出现兼容性问题。如果你使用的是 Windows 终端，请考虑将换行符`\`调整为以下形式：

```py
$ python -m grpc_tools.protoc \ 
--proto_path=. ./grpcserver.proto \ 
--python_out=. \ 
--grpc_python_out=.
```

这是 CMD 中的相同行：

```py
$ python -m grpc_tools.protoc ^
--proto_path=. ./grpcserver.proto ^
--python_out=. ^
--grpc_python_out=.
```

这是 PowerShell 中的相同行：

```py
$ python -m grpc_tools.protoc `
--proto_path=. ./grpcserver.proto `
--python_out=. `
--grpc_python_out=.
```

**粗体**：表示新术语、重要词汇或屏幕上看到的词汇。例如，菜单或对话框中的文字会以这种方式显示。以下是一个示例：“此限制可以在设置中调整（**设置** | **高级设置** | **运行/调试** | **临时** **配置限制**）。”

小贴士或重要注意事项

显示如下。

# 部分

在本书中，你会发现一些频繁出现的标题（*准备就绪*，*如何操作…*，*它是如何工作的…*，*还有更多…*，以及*另请参阅*）。

为了清楚地说明如何完成食谱，请按照以下方式使用这些部分。

## 准备就绪

本节告诉你可以在食谱中期待什么，并描述如何设置任何所需的软件或初步设置。

## 如何操作…

本节包含遵循食谱所需的步骤。

## 它是如何工作的…

本节通常包含对上一节发生事件的详细解释。

## 还有更多…

本节包含有关食谱的附加信息，以便让你对食谱有更深入的了解。

## 另请参阅

本节提供了对其他有用信息的链接，以帮助理解食谱。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果你对本书的任何方面有疑问，请在邮件主题中提及书名，并请通过 customercare@packtpub.com 给我们发送邮件。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果你在这本书中发现了错误，我们将非常感激你向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)，选择你的书，点击**勘误提交表单**链接，并输入详细信息。

**盗版**：如果你在互联网上发现我们作品的任何非法副本，我们将非常感激你提供位置地址或网站名称。请通过 copyright@packt.com 与我们联系，并提供材料的链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且你感兴趣的是撰写或为本书做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享你的想法

一旦你阅读了*FastAPI 食谱*，我们很乐意听到你的想法！请[点击此处直接进入此书的亚马逊评论页面](https://packt.link/r/1-805-12785-3)并分享你的反馈。

您的评论对我们和科技社区非常重要，并将帮助我们确保我们提供高质量的内容。

# 下载此书的免费 PDF 副本

感谢您购买此书！

您喜欢在路上阅读，但无法随身携带您的印刷书籍吗？

您的电子书购买是否与您选择的设备不兼容？

别担心，现在每购买一本 Packt 书籍，您都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何时间、任何设备上阅读。直接从您最喜欢的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

优惠远不止这些，您还可以获得独家折扣、时事通讯和每日免费内容的访问权限。

按照以下简单步骤获取优惠：

1.  扫描二维码或访问以下链接

![二维码](img/B21025_QR_Free_PDF.jpg)

[`packt.link/free-ebook/978-1-80512-785-7`](https://packt.link/free-ebook/978-1-80512-785-7)

1.  提交您的购买证明

1.  就这些！我们将直接将您的免费 PDF 和其他优惠发送到您的邮箱。
