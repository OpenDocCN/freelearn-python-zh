# 前言

*全栈 FastAPI、React 和 MongoDB，第二版*是一本快速、简洁、实用的入门指南，旨在提升 Web 开发者的潜力，并帮助他们利用 FARM 栈的灵活性、适应性和稳健性，在快速发展的 Web 开发和 AI 领域中保持领先。本书介绍了栈的每个元素，然后解释了如何使它们协同工作以构建中型 Web 应用程序。

本书通过实际操作示例和真实世界的用例，展示了如何使用 MongoDB 设置文档存储，使用 FastAPI 构建简单的 API，以及使用 React 创建应用程序。此外，它深入探讨了使用 Next.js，确保 MongoDB 中的数据完整性和安全性，以及将第三方服务与应用程序集成。

# 这本书将如何帮助您

这本书采用实际操作的方法，通过使用 FARM 栈的真实世界示例来展示 Web 应用程序开发。到本书结束时，您将能够自信地使用 FARM 栈以快速的速度开发功能齐全的 Web 应用程序。

# 这本书面向的对象是谁

这本书适合具有基本 JavaScript 和 Python 知识的初级 Web 开发者，他们希望提高自己的开发技能，掌握一个强大且灵活的栈，并更快地编写更好的应用程序。

# 这本书涵盖了哪些内容

*第一章*，*Web 开发与 FARM 栈*，通过快速浏览广泛使用的各种技术，为您提供了对 Web 开发领域的深入理解。它介绍了最受欢迎的选项——FARM 栈。它突出了 FARM 栈组件的优势，它们之间的关系，以及为什么这一组特定技术非常适合 Web 应用程序。

*第二章*，*使用 MongoDB 设置数据库*，提供了 MongoDB 的概述，然后展示了如何为 FARM 应用程序设置数据存储层。它帮助您了解创建、更新和删除文档的基本知识。此外，本章详细介绍了聚合管道框架——一个强大的分析工具。

*第三章*，*Python 类型提示和 Pydantic*，包括一些示例，教您更多关于 FastAPI 的 Web 特定方面以及如何无缝地在 MongoDB、Python 数据结构和 JSON 之间混合数据。

*第四章*，*FastAPI 入门*，专注于介绍 FastAPI 框架，以及标准的 REST API 实践及其在 FastAPI 中的实现方式。它涵盖了 FastAPI 实现最常见 REST API 任务的一些非常简单的示例，以及它如何通过利用现代 Python 功能和库（如 Pydantic）来帮助您。

*第五章*，*设置 React 工作流程*，展示了如何使用 React 框架设计一个由几个组件组成的应用程序。它讨论了探索 React 及其各种功能所需的工具。

*第六章*，*身份验证和授权*，详细介绍了基于 **JSON Web Tokens**（**JWTs**）的简单、健壮且可扩展的 FastAPI 后端配置。它展示了如何将基于 JWT 的身份验证方法集成到 React 中，利用 React 的强大功能——特别是 Hooks、Context 和 React Router。

*第七章*，*使用 FastAPI 构建 Backend*，帮助您处理一个简单的业务需求并将其转化为一个完全功能、部署在互联网上的 API。它展示了如何定义 Pydantic 模型、执行 CRUD 操作、构建 FastAPI 后端并连接到 MongoDB。

*第八章*，*构建应用程序的前端*，说明了构建全栈 FARM 应用程序前端的步骤。它展示了如何使用现代 Vite 设置创建 React 应用程序并实现基本功能。

*第九章*，*使用 FastAPI 和 Beanie 集成第三方服务*，介绍了 Beanie，这是一个基于 Motor 和 Pydantic 的流行 ODM 库，用于 MongoDB。它展示了如何定义模型和映射到 MongoDB 集合的 Beanie 文档。您将看到如何构建另一个 FastAPI 应用程序，并使用后台任务集成第三方服务。

*第十章*，*使用 Next.js 14 进行 Web 开发*，对重要的 Next.js 概念进行了概述，例如服务器操作、表单处理和 Cookie，以帮助创建新的 Next.js 项目。您还将学习如何在 Netlify 上部署您的 Next.js 应用程序。

*第十一章*，*有用的资源和项目想法*，在工作与 FARM 栈时提供了一些实用建议，以及 FARM 栈或非常相似的栈可能适用且有帮助的项目想法。

# 为了充分利用本书

您需要了解 JavaScript 和 Python 的基础知识。对 MongoDB 的先验知识更佳，但不是必需的。您将需要以下软件：

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| MongoDB 版本 7.0 或更高 | Windows、macOS 或 Linux |
| MongoDB Atlas Search | Windows、macOS 或 Linux |
| MongoDB Shell 2.2.15 或更高版本 | Windows、macOS 或 Linux |
| Node.js 版本 18.17 或更高 | Windows、macOS 或 Linux |
| Python 3.11.7 或更高版本 | Windows、macOS 或 Linux |
| Next.js 14 或更高版本 | Windows、macOS 或 Linux |
| FastAPI 0.111.1 | Windows、macOS 或 Linux |
| React 18 或更高版本 | Windows、macOS 或 Linux |

如果您使用的是本书的数字版，我们建议您亲自输入代码或从书的 GitHub 仓库（下一节中提供链接）获取代码。这样做将帮助您避免与代码复制和粘贴相关的任何潜在错误。

# 下载示例代码文件

您可以从 GitHub（[`github.com/PacktPublishing/Full-Stack-FastAPI-React-and-MongoDB-2nd-Edition`](https://github.com/PacktPublishing/Full-Stack-FastAPI-React-and-MongoDB-2nd-Edition)）下载本书的示例代码文件。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图和图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/Bookname_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/Bookname_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`文本中的代码`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“可选地，您可以创建一个`middleware.js`函数，该函数将包含将在每个（或仅选定的）请求上应用的中间件。”

代码块设置如下：

```py
const Cars = () => {
    return (
        <div>Cars</div>
    )
}
export default Cars
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
      <body>
        <Navbar />
        {children}
      </body>
```

任何命令行输入或输出都应如下编写：

```py
git push -u origin main
```

**粗体**: 表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词以**粗体**显示。以下是一个示例：“选择 Windows 版本，然后点击**下载**。”

小贴士或重要注意事项

看起来是这样的。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**: 如果您对本书的任何方面有疑问，请通过电子邮件发送至 customercare@packtpub.com，并在邮件主题中提及书名。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**: 如果您在互联网上发现我们作品的任何形式的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过电子邮件发送至 copyright@packt.com 并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 下载此书的免费 PDF 副本

感谢您购买此书！

您喜欢在路上阅读，但无法随身携带您的印刷书籍吗？

您的电子书购买是否与您选择的设备不兼容？

别担心，现在，每购买一本 Packt 书籍，您都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何设备上阅读。直接从您最喜欢的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

优惠远不止于此，您还可以获得独家折扣、新闻通讯以及每天收件箱中的优质免费内容。

按照以下简单步骤获取这些好处：

1.  扫描二维码或访问以下链接

![](img/B22406_QR_Free_PDF.png)

[`packt.link/free-ebook/9781835886762`](https://packt.link/free-ebook/9781835886762)

1.  提交您的购买证明

1.  就这样！我们将直接将免费 PDF 和其他优惠发送到您的电子邮件。
