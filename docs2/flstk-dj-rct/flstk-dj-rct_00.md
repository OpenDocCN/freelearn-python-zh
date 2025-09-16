# 前言

使用 Python 或 JavaScript 开始全栈开发可能会令人望而却步，尤其是如果你已经是使用这些语言之一的开发生态中的一员，并希望将第二种语言添加到你的技能集中。如果你是一名正在使用 Django 或 React 的开发者，或者是一名了解 Python 或 JavaScript 的开发者，并希望学习如何从头开始构建具有身份验证、CRUD 操作等功能的全栈应用程序，同时你还在寻找如何使用 Docker 在 AWS 上部署 Web 应用程序的方法，本书涵盖了你需要的一切。

本书将帮助你发现结合两个最受欢迎的框架——React 和 Django——的双重力量的全部潜力实践。我们将构建全栈应用程序，包括后端的 RESTful API 和直观的前端，同时探索这两个框架的高级功能。我们将从头开始构建一个名为 Postagram 的社交媒体网络应用程序，同时涵盖端到端开发的重要概念、技术和最佳实践。

我们将看到 React 框架的动态功能如何用于构建你的前端系统，以及 Django 的 ORM 层如何帮助简化数据库，从而提高构建全栈应用程序的后端开发过程。

在本书结束时，你将能够从头开始独立创建一个动态的全栈应用程序。

# 本书面向的对象

这本书是为那些熟悉 Django 但不知道如何开始构建全栈应用程序（更确切地说，构建 RESTful API）的 Python 开发者而写的。如果你是一名了解 JavaScript 的前端开发者，并希望学习全栈开发，这本书也会对你很有帮助。如果你是一名经验丰富的全栈开发者，正在使用不同的技术，并希望探索和学习新的技术，这本书也是为你而写的。

# 本书涵盖的内容

*第一章*，*创建 Django 项目*，展示了如何创建 Django 项目以及与数据库服务器进行必要的配置。

*第二章*，*使用 JWT 进行身份验证和授权*，解释了如何使用 JSON Web Tokens 实现身份验证系统以及如何编写自定义权限。

*第三章*，*社交媒体帖子管理*，展示了如何使用序列化器和视图集实现复杂的 CRUD 操作。

*第四章*，*在社交媒体帖子中添加评论*，展示了如何使用数据库关系、序列化器和视图集向帖子添加评论。

*第五章*，*测试 REST API*，介绍了使用 Django 和 Pytest 进行测试。

*第六章*, *使用 React 创建项目*，解释了如何在配置良好的开发环境中创建 React 项目。

*第七章*, *构建注册和登录表单*，解释了如何在全栈应用程序的前端实现认证表单和逻辑。 

*第八章*, *社交媒体帖子*，展示了如何在 React 前端实现社交媒体帖子的 CRUD 操作。

*第九章*, *评论*，展示了如何在 React 前端实现社交媒体评论的 CRUD 操作。

*第十章*, *用户资料*，解释了如何在 React 前端实现与资料相关的 CRUD 操作以及如何上传图片。

*第十一章*, *React 组件的有效 UI 测试*，向您介绍使用 Jest 和 React Testing Library 进行组件测试。

*第十二章*, *部署基础 - Git、GitHub 和 AWS*，介绍了 DevOps 工具和术语以及如何在 AWS EC2 上直接部署 Django 应用程序。

*第十三章*, *将 Django 项目 Docker 化*，展示了如何使用 Docker 和 Docker Compose 将 Django 应用程序 Docker 化。

*第十四章*, *在 AWS 上自动化部署*，展示了如何使用 GitHub Actions 在 EC2 上部署 Docker 化的应用程序。

*第十五章*, *在 AWS 上部署我们的 React 应用*，演示了如何在 AWS S3 上部署 React 应用程序并使用 GitHub Actions 自动化部署。

*第十六章*, *性能、优化和安全*，向您展示如何使用 webpack 优化应用程序，优化数据库查询，并增强后端安全性。

# 为了最大限度地利用本书

为了使用本书，您需要在您的机器上安装 Python 3.8+、Node.js 16+和 Docker。本书中的所有代码和示例都是使用 Django 4.1 和 React 18 在 Ubuntu 上测试的。在安装任何 React 或 JavaScript 库时，请确保您有它们文档中提供的最新安装命令（`npm`、`yarn`和`pnpm`），并检查是否有与本书中使用版本相关的任何重大更改。

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Python | Windows, macOS, 或 Linux |
| JavaScript | Windows, macOS, 或 Linux |
| PostgreSQL | Windows, macOS, 或 Linux |
| Django | Windows, macOS, 或 Linux |
| React | Windows, macOS, 或 Linux |
| Docker | Windows, macOS, 或 Linux |

**如果您正在使用本书的数字版，我们建议您自己输入代码或从本书的 GitHub 仓库（下一节中提供链接）获取代码。这样做将帮助您避免与代码的复制和粘贴相关的任何潜在错误。**

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件：[`github.com/PacktPublishing/Full-stack-Django-and-React`](https://github.com/PacktPublishing/Full-stack-Django-and-React)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们！

# 下载彩色图像

我们还提供了一个包含本书中使用的截图和图表的彩色图像 PDF 文件。您可以从这里下载：[`packt.link/jdEHp`](https://packt.link/jdEHp)。

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“一旦安装了包，就在 Django 项目的根目录下创建一个名为`pytest.ini`的新文件。”

代码块设置如下：

```py
>>> comment = Comment.objects.create(**comment_data)
>>> comment
<Comment: Dingo Dog>
>>> comment.body
'A comment.'
```

当我们希望将您的注意力引向代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
ENV = os.environ.get("ENV")
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get(
   "SECRET_KEY", default="qkl+xdr8aimpf-&x(mi7)dwt^-q77aji#j*d#02-5usa32r9!y"
)
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False if ENV == "PROD" else True
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", default="*").split(",")
```

任何命令行输入或输出都应如下所示：

```py
pip install drf-nested-routers
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词以**粗体**显示。以下是一个示例：“最后，选择**权限**选项卡并选择**存储桶策略**。”

小贴士或重要提示

看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请通过电子邮件发送给我们，电子邮件地址为 customercare@packtpub.com，并在邮件主题中提及书名。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您能向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，我们将非常感激您能向我们提供位置地址或网站名称。请通过电子邮件发送给我们，电子邮件地址为 copyright@packt.com，并附上材料的链接。

**如果您有兴趣成为作者**：如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享你的想法

一旦你阅读了《Full Stack Django and React》，我们很乐意听听你的想法！请[点击此处直接进入此书的亚马逊评论页面](https://packt.link/r/1803242973)并分享你的反馈。

你的评论对我们和科技社区非常重要，并将帮助我们确保我们提供高质量的内容。

# 下载此书的免费 PDF 副本

感谢您购买此书！

你喜欢在路上阅读，但无法携带你的印刷书籍到处走吗？ 你的电子书购买是否与你的选择设备不兼容？

别担心，现在每购买一本 Packt 书籍，你都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何设备上阅读。直接从你最喜欢的技术书籍中搜索、复制和粘贴代码到你的应用程序中。

优惠远不止这些，你还可以获得独家折扣、时事通讯和每日免费内容的每日电子邮件。

按照以下简单步骤获取好处：

1.  扫描二维码或访问以下链接

![](img/B18221_QR_Free_PDF.jpg)

[`packt.link/free-ebook/9781803242972`](https://packt.link/free-ebook/9781803242972)

1.  提交你的购买证明

1.  就这样！我们将直接将免费 PDF 和其他优惠发送到您的电子邮件。

# 第一部分：技术背景

在本书的这一部分，你将学习如何使用 Django 和 Django REST 构建 REST API。本部分提供了将 Django 连接到 PostgreSQL 数据库、使用 JSON Web Tokens 添加身份验证、创建支持复杂 CRUD 操作的 RESTful 资源以及向 Django 应用程序添加测试所需的知识。我们将特别构建一个名为 Postagram 的社交媒体网络应用程序的后端，该应用程序具有社交媒体应用程序最常见的特点，如帖子管理、评论管理和帖子点赞。

本节包括以下章节：

+   *第一章*, *创建 Django 项目*

+   *第二章*, *使用 JWT 进行身份验证和授权*

+   *第三章*, *社交媒体帖子管理*

+   *第四章*, *为社交媒体帖子添加注释*

+   *第五章*, *测试 REST API*
