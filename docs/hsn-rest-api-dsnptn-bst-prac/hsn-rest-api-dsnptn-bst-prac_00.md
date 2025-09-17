# 前言

本书旨在赋予你 API 设计原则和最佳实践的知识，以便你能够准备好设计高度可扩展、可重用、可适应和安全的 RESTful API。本书还介绍了 RESTful API 最不可或缺领域的一些常见和新兴模式。

RESTful 模式影响跨越多个功能的网络服务各个层次，例如 CRUD 操作、数据库、表示层、应用程序和基础设施层。在 RESTful 领域，其他突出和主导的模式包括通信、集成、编排、安全、管理、软件部署和交付。本书将帮助你熟悉最重要的模式，如客户端/服务器发现、API 网关、API 组合、断路器、企业安全、内容协商、端点重定向、幂等能力、API 外观等许多基本模式。

虽然本书主要涵盖关于 RESTful API 的中级到高级主题，但它也涵盖了服务导向架构和面向资源的 Web 服务架构的一些基础知识，以帮助你更快地理解所涵盖的内容。

# 本书面向读者

本书面向任何需要全面且易于理解的学习资源来帮助他们提升 RESTful API 设计和开发技能，并展示如何构建高度可采用的 RESTful API，这些 API 能够提供最佳实践和关键原则的见解，以及经过验证的 RESTful API 设计模式。

# 本书涵盖内容

第一章，*RESTful 架构基础介绍*，旨在更新你对网络、其架构以及其演变方式的一些基本概念的理解，希望为 RESTful 服务设计和应用打下坚实的基础。我们将讨论万维网层和架构、Web API 开发模型以及基于 REST 的服务通信。你还将了解到面向服务和面向资源的架构原则和特性，然后进一步探讨 REST 原则、约束、限定符和目标的基础。

第二章，*设计策略、指南和最佳实践*，讨论了一些基本的 API 设计指南，如一致性、标准化、可重用性和通过 REST 接口的可访问性，旨在为 API 设计者提供更好的思维过程进行 API 建模。此外，本章还旨在介绍一些更好的 REST API 实现实践，以及一些常见但可避免的 API 策略错误。

第三章，《RESTful API 模式精华》，不仅提供了关于概念的信息，还提供了与 RESTful API 的常见和基本设计模式相关的实际代码示例，以便您更好地理解和增强您的 RESTful API 服务。作为本章的一部分，您将学习一些常见和基本 API 设计模式，例如内容协商、URI 模板、分页、Unicode 等，以及这些模式的代码实现。每个模式都针对 RESTful 约束，并帮助您确保这些基本模式在您的 API 设计和实现中得到考虑。

第四章，《高级 RESTful API 模式》，是我们对 API 设计模式的第二次审视，旨在讨论一些高级设计模式，如版本控制、前后端分离、授权、幂等及其重要性，以及如何通过批量操作 API 来增强 API 并取悦客户。

第五章，《微服务 API 网关》，主要讨论 API 网关解决方案在使微服务对生产企业级、任务关键型、云托管、事件驱动、生产级和业务中心型应用至关重要的贡献。本章讨论了流行的 API 网关解决方案，并探讨了通过 API 网关实现聚合服务的实现。

第六章，《RESTful 服务 API 测试与安全》，旨在带您踏上 API 测试之旅，探讨 API 测试的类型、API 测试的挑战以及 API 测试中的安全问题。您将一瞥各种 API 测试工具、API 安全工具和框架，并学习如何在 API 测试、质量和安全措施中暴露任何安全问题和 API 漏洞。

第七章，《智能应用 RESTful 服务组合》，专门为您讲述 RESTful 服务范式在设计和开发下一代以微服务为中心的企业级应用以及部署中的应用。它探讨了能够相互发现和绑定在一起的 RESTful 服务如何导致过程感知、业务关键和以人为本的复合服务。您将了解服务组合的需求，各种组合方法，如编排和协奏，以及混合版本的编排和协奏在智能应用中的使用。

第八章，*RESTful API 设计技巧*，讨论了构建能够轻松跟上技术和业务变化的强大且兼容的 RESTful API 所需的设计模式和最佳实践。本章探讨了 API 的重要性；API 设计模式和最佳实践；API 安全指南；API 设计、开发、集成、安全和管理的各种工具和相关平台；以及 API 驱动的数字世界趋势。

第九章，*对 RESTful 服务范式的更深入观察*，专注于传达产生 RESTful 服务及其相应 API 的新兴技术和技巧。我们讨论了软件定义和驱动世界的方法，以及有助于快速轻松部署 API 的新兴应用类型。此外，本章还讨论了应用现代化和集成中的 REST 范式、用于数字化转型和智能的 RESTful 服务以及基于 REST 的微服务的最佳实践。

第十章，*框架、标准语言和工具包*，向您介绍了一些在决定 API 开发需求时可能很有用的突出框架。它讨论了几种对应用开发者来说，使用熟悉的编程语言启动他们的 RESTful API 和微服务的突出框架。本章试图为您提供有关一些编程语言友好型框架的信息，以便您可以选择最适合您 RESTful API 开发需求的框架。此外，本章还包含了一个各种框架及其支持的语言的参考表，以及它们的突出特性。

第十一章，*从遗留系统现代化到以微服务为中心的应用*，讨论了**微服务架构**（**MSA**）是现代应用前进的方向，这些应用具有高度敏捷、灵活和弹性。本章提供了遗留应用程序现代化的原因，阐述了为什么应用程序必须现代化才能迁移并在云环境中运行，讨论了微服务和容器的组合是实现遗留系统现代化的最佳方式，并详细介绍了遗留系统现代化的方法。

# 为了充分利用这本书

由于本书介绍了许多网络服务和 RESTful 服务概念，您不需要遵循任何特定要求；然而，如果您想运行和执行书中提供的代码示例（您应该这样做），那么您需要对 Java 编程语言、Maven 或任何构建工具有所了解。

包含示例代码的章节对如何运行和测试示例有清晰的说明，并附带构建和运行脚本。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)上的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择 SUPPORT 标签页。

1.  点击代码下载与勘误。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

下载完文件后，请确保您使用以下软件的最新版本解压缩或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Hands-On-RESTful-API-Design-Patterns-and-Best-Practices`](https://github.com/PacktPublishing/Hands-On-RESTful-API-Design-Patterns-and-Best-Practices)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包可供在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 处获取。查看它们吧！

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“四个基本的 HTTP 操作：`GET`、`POST`、`PUT` 和 `DELETE`。”

代码块设置如下：

```py
@GetMapping({"/v1/investors","/v1.1/investors","/v2/investors"})
  public List<Investor> fetchAllInvestors()
    {
       return investorService.fetchAllInvestors();
    }
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
public interface DeleteServiceFacade {
 boolean deleteAStock(String investorId, String stockTobeDeletedSymbol);
    boolean deleteStocksInBulk(String investorId, List<String> stocksSymbolsList);
}
```

任何命令行输入或输出都按以下方式编写：

```py
$ mkdir css
$ cd css
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“**Pipeline** 实体完全负责协调控制和数据流”

警告或重要注意事项如下所示。

小贴士和技巧如下所示。

# 联系我们

我们读者的反馈总是受欢迎的。

**一般反馈**：请发送电子邮件至 `feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 发送电子邮件给我们。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您向我们报告。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现任何形式的我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过发送链接至 `copyright@packtpub.com` 与我们联系。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
