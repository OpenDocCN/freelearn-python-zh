# 前言

本书基于以现代方式开发基于 Python 的无服务器 Web 或微服务的方式。无服务器涉及云服务提供商提供的无服务器基础设施。本书展示了如何使用亚马逊网络服务来实现无服务器基础设施。此外，它还涵盖了使用 Zappa 的部署过程。Zappa 消除了手动干预，为您提供了自动化的部署方式，并帮助您维护多个部署阶段。

# 本书适合谁

本书适用于初学者到有经验的 Python 开发人员，他们想要了解在无服务器基础设施上开发 Python Web 服务或微服务的方式。有经验的 Python 开发人员可以通过学习无服务器技术和了解无服务器部署来提升他们的技能。

# 本书涵盖的内容

第一章，*用于无服务器的亚马逊网络服务*，涵盖了理解 AWS Lambda 和 API Gateway 服务的基础知识。还介绍了通过与 AWS 控制台和 CLI 工具交互创建无服务器服务的手动过程。

第二章，*开始使用 Zappa*，解释了 Zappa 工具的概念，并详细说明了使用 Zappa 相对于 AWS 服务的手动过程的好处。

第三章，*使用 Zappa 构建 Flask 应用程序*，探讨了基本的 Flask 应用程序开发，并使用 Zappa 作为无服务器应用程序进行部署。

第四章，*使用 Zappa 构建基于 Flask 的 REST API*，介绍了基于 Flask 的 RESTful API 开发和使用 Zappa 的部署过程。

第五章，*使用 Zappa 构建 Django 应用程序*，讨论了 Django 核心应用程序开发，并使用 Zappa 将应用程序部署为 AWS Lambda 上的无服务器应用程序。

第六章，*使用 Zappa 构建 Django REST API*，专注于使用 Django REST 框架实现 RESTful API 以及使用 Zappa 的部署过程。

第七章，*使用 Zappa 构建 Falcon 应用程序*，带您了解使用 Falcon 框架开发 RESTful API 作为微服务的过程，以及使用 Zappa 的部署过程。

第八章，*使用 SSL 的自定义域*，解释了如何使用 Zappa 配置自定义域，并涵盖了使用 AWS 生成 SSL。

第九章，*在 AWS Lambda 上执行异步任务*，展示了使用 Zappa 执行耗时任务的异步操作的实现。

第十章，*高级 Zappa 设置*，让您熟悉 Zappa 工具的附加设置，以增强应用部署过程。

第十一章，*使用 Zappa 保护无服务器应用程序*，概述了使用 Zappa 在 AWS Lambda 上保护无服务器应用程序的安全方面。

第十二章，*使用 Docker 的 Zappa*，介绍了在 AWS Lambda 上下文环境中使用 Docker 容器化进行应用程序开发。

# 充分利用本书

在开始之前，读者需要一些先决条件。读者应具备以下条件：

+   对虚拟环境有很好的理解

+   理解 Python 包安装

+   了解使用 Apache 或 NGINX 进行传统部署的知识

+   对 Web 服务或微服务的基本了解

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，文件将直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的以下工具解压或提取文件夹：

+   WinRAR/7-Zip 适用于 Windows

+   Zipeg/iZip/UnRarX 适用于 Mac

+   7-Zip/PeaZip 适用于 Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Building-Serverless-Python-Web-Services-with-Zappa`](https://github.com/PacktPublishing/Building-Serverless-Python-Web-Services-with-Zappa)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有来自丰富书籍和视频目录的其他代码包可供下载，网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。快去看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/BuildingServerlessPythonWebServiceswithZappa_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/BuildingServerlessPythonWebServiceswithZappa_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子：“Zappa 部署需要生成`zappa_settings.json`文件，该文件生成`zappa init`命令。”

代码块设置如下：

```py
client = boto3.client('lambda')
response = client.invoke(
    FunctionName='MyFunction',
    InvocationType='Event'
)
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
$ curl https://quote-api.abdulwahid.info/daily
{"quote": "May the Force be with you.", "author": "Star Wars", "category": "Movies"}
```

任何命令行输入或输出都以以下方式编写：

```py
$ pip install awscli
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单中的单词或对话框中的单词会在文本中以这种方式出现。这是一个例子：“单击“创建函数”按钮。”

警告或重要说明会出现在这样的地方。

提示和技巧会出现在这样的地方。
