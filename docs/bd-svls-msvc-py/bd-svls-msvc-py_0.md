# 前言

这本书将让您对微服务和无服务器计算有很好的理解，以及它们与现有架构相比的优缺点。您将对部署完整的无服务器堆栈的威力有所认识，不仅在节省运行成本方面，还在支持维护和升级方面。这有效地使您的公司能够更快地推出任何新产品，并在这个过程中击败竞争对手。您还将能够创建、测试和部署一个可扩展的无服务器微服务，其成本是按使用量支付的，而不是按其正常运行时间支付的。此外，这将允许您根据请求的数量进行自动扩展，同时安全性是由 AWS 本地构建和支持的。因此，既然我们知道前方有什么，让我们立即开始阅读这本书吧。

# 这本书是为谁准备的

如果您是一名具有 Python 基础知识的开发人员，并且想要学习如何构建、测试、部署和保护微服务，那么这本书适合您。不需要先前构建微服务的知识。

# 本书涵盖的内容

第一章，*无服务器微服务架构和模式*，提供了单片和微服务架构的概述。您将了解设计模式和原则，以及它们与无服务器微服务的关系。

第二章，*创建您的第一个无服务器数据 API*，讨论了安全性及其重要性。我们将讨论 IAM 角色，并概述一些安全概念和原则，涉及到保护您的无服务器微服务，特别是关于 Lambda、API Gateway 和 DynamoDB。

第三章，*部署您的无服务器堆栈*，向您展示如何仅使用代码和配置部署所有基础设施。您将了解不同的部署选项。

第四章，*测试您的无服务器微服务*，涵盖了测试的概念。我们将探讨许多类型的测试，从使用模拟进行单元测试，使用 Lambda 和 API Gateway 进行集成测试，本地调试 Lambda，并使本地端点可用，到负载测试。

第五章，*保护您的微服务*，涵盖了如何使您的微服务安全的重要主题。

# 充分利用本书

一些先前的编程知识将会有所帮助。

所有其他要求将在各自章节的相关点中提到。

# 下载示例代码文件

您可以从您在[www.packt.com](http://www.packt.com)的账户中下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   Windows 的 WinRAR/7-Zip

+   Mac 的 Zipeg/iZip/UnRarX

+   Linux 的 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Building-Serverless-Microservices-in-Python`](https://github.com/PacktPublishing/Building-Serverless-Microservices-in-Python)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有来自我们丰富书籍和视频目录的其他代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。查看一下！

# 下载彩色图片

本书中使用的屏幕截图/图表的彩色图像也可以在 PDF 文件中找到。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789535297_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789535297_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这里有一个例子：“在这里，您可以看到我们将 `EventId` 作为资源 `1234`，以及格式为 `YYYYMMDD` 的 `startDate` 参数。”

代码块设置如下：

```py
  "phoneNumbers": [
    {
      "type": "home",
      "number": "212 555-1234"
    },
    {
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 27,
  "address": {
```

任何命令行输入或输出都以以下方式编写：

```py
$ cd /mnt/c/
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单中的单词或对话框中的单词会在文本中以这种方式出现。这里有一个例子：“在 DynamoDB 导航窗格中，选择 Tables，然后选择 user-visits。”

警告或重要说明会以这种方式出现。

提示和技巧会出现在这样。
