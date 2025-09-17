# 前言

**REST**（代表**表示状态转移**）是推动现代 Web 开发和移动应用开发的架构风格。实际上，开发和使用 RESTful Web 服务是任何现代软件开发工作的一项必备技能。有时，你必须与现有的 API 交互，在其他情况下，你必须从头设计一个 RESTful API 并使其与 JSON（代表 JavaScript 对象表示法）兼容。

Python 是最受欢迎的编程语言之一。Python 3.6 和 3.7 是 Python 最现代的版本。Python 是开源和多平台的，你可以用它来开发任何类型的应用程序，从网站到极其复杂的科学计算应用程序。总有一个 Python 包可以使事情变得更容易，避免重复造轮子并更快地解决问题。最重要的和最受欢迎的云计算提供商使使用 Python 及其相关 Web 框架变得容易。因此，Python 是开发 RESTful Web 服务的理想选择。本书涵盖了你需要知道的所有内容，以选择最合适的 Python Web 框架并从头开始开发 RESTful API。

你将使用四个最受欢迎的 Python Web 框架的最新版本，这些框架使开发 RESTful Web 服务变得容易：Flask、Django、Pyramid 和 Tornado。每个 Web 框架都有其优点和缺点。你将使用代表每个这些 Web 框架适当案例的示例，结合额外的 Python 包，这些包将简化最常见任务。你将学习如何使用不同的工具来测试和开发高质量、一致性和可扩展的 RESTful Web 服务。

你将编写单元测试并提高你将在本书中开发的 RESTful Web 服务的测试覆盖率。你不仅会运行示例代码；你还将确保为你的 RESTful API 编写测试。你将始终编写现代 Python 代码，并利用最新 Python 版本引入的功能。

本书将帮助你学习如何利用许多简化与 RESTful Web 服务相关的最常见任务的软件包。你将能够开始为任何领域创建自己的 RESTful API，无论是在 Python 3.6、3.7 或更高版本中覆盖的任何 Web 框架中。

# 本书面向对象

本书面向那些对 Python 有实际了解并希望利用 Python 的各种框架构建令人惊叹的 Web 服务的 Web 开发者。你应该对 RESTful API 有所了解。

# 为了充分利用本书

为了使用 Python 3.6 和 Python 3.7 的不同示例，你需要一台具有 Intel Core i3 或更高 CPU 和至少 4GB RAM 的计算机。你可以使用以下任何一种操作系统：

+   Windows 7 或更高版本（Windows 8、Windows 8.1 或 Windows 10）

+   Windows Server 2012 或更高版本（Windows Server 2016 或 Windows Server 2019）

+   macOS Mountain Lion 或更高版本

+   任何能够运行 Python 3.7.1 的 Linux 版本以及任何支持 JavaScript 的现代浏览器

您需要在计算机上安装 Python 3.7.1 或更高版本。

# 下载示例代码文件

您可以从 [www.packt.com](http://www.packt.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

下载完成后，请确保您使用最新版本的软件解压缩或提取文件夹。

+   Windows 的 WinRAR/7-Zip

+   Mac 的 Zipeg/iZip/UnRarX

+   Linux 的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，地址为 [`github.com/PacktPublishing/Hands-On-RESTful-Python-Web-Services-Second-Edition ...`](https://github.com/PacktPublishing/Hands-On-RESTful-Python-Web-Services-Second-Edition)。

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789532227_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789532227_ColorImages.pdf)。

# 使用的约定

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。以下是一个示例：“我们可以通过使用 `include` 指令来包含其他上下文。”

代码块设置如下：

```py
html, body, #map { height: 100%;  margin: 0; padding: 0}
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
[default]
```

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢什么或不喜欢什么。读者反馈对我们来说很重要，因为它帮助我们开发出您真正能从中获得最大价值的书籍。

要向我们发送一般反馈，请简单地发送电子邮件到 `feedback@packtpub.com`，并在邮件主题中提及书籍的标题。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南 [www.packtpub.com/authors](https://www.packtpub.com/books/info/packt/authors)。

# 客户支持

现在您已经是 Packt 书籍的骄傲拥有者，我们有一些事情可以帮助您从购买中获得最大价值。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**: 如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过 `customercare@packtpub.com` 发送邮件给我们。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**侵权**: 如果您在互联网上发现任何形式的我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过发送链接至 `copyright@packt.com` 与我们联系。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 勘误

尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入您的勘误详细信息。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看...

# 评论

请留下评论。一旦您阅读并使用了这本书，为什么不在您购买它的网站上留下评论呢？潜在的读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/)。

# 侵权

在互联网上对版权材料的侵权是一个跨所有媒体的持续问题。在 Packt，我们非常重视保护我们的版权和许可证。如果您在互联网上发现任何形式的我们作品的非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过发送链接至 `copyright@packt.com` 与我们联系，以提供疑似侵权材料的链接。

我们感谢您的帮助，以保护我们的作者和提供有价值内容的能力。

# 问题

如果您对本书的任何方面有问题，您可以通过 `questions@packtpub.com` 与我们联系，我们将尽力解决问题。
