# 前言

Flask 是一个设计精良的微框架，旨在提供创建 Web 应用程序所需的最小功能量。它确实做到了它设计要做的。与其他 Web 框架不同，Flask 没有捆绑整个生态系统，没有现成的功能来处理数据库、缓存、安全或表单处理。

这一概念的目标是允许程序员以任何他们想要的方式设计他们的应用程序或工具，不施加任何结构或设计。然而，由于 Flask 社区相当庞大，你可以找到各种各样的扩展，这些扩展可以帮助你利用 Flask 与大量技术相结合。本书的主要重点之一是介绍这些扩展，并找出它们如何帮助避免重复造轮子。这些扩展的最好之处在于，如果你不需要它们的额外功能，你就不需要包含它们，你的应用程序将保持小巧。

本书将帮助您构建应用程序，以便轻松扩展到任何规模。使用包和简单、可预测的命名空间对于保持可维护性和提高团队生产力至关重要。这就是本书的另一大重点是介绍如何使用 Flask 应用创建 **模型-视图-控制器**（**MVC**）架构的原因。

现代应用程序必须超越良好的代码结构。安全性、依赖隔离、环境配置、开发/生产一致性以及负载扩展能力是必须考虑的因素。在整个本书中，你将学习如何解决这些问题，识别可能的风险，并提前思考。

本书融入了大量研究和大量关于在开发和部署 Web 应用程序时可能出错的第一手经验。我真诚地希望你会喜欢阅读它。

# 本书面向对象

本书理想的读者对象是希望使用 Flask 及其高级功能来创建企业级和轻量级应用的 Python 开发者。本书面向那些已有一定 Flask 经验，并希望将技能从入门级提升到精通级的人。

# 本书涵盖内容

第一章，*入门指南*，帮助读者使用 Python 项目的最佳实践设置 Flask 开发环境。你将获得一个非常基础的 Flask 应用骨架，这个骨架将在整本书中逐步构建。

第二章，*使用 SQLAlchemy 创建模型*，展示了如何结合 Flask 使用 Python 数据库库 SQLAlchemy 来创建数据库的对象化 API。

第三章，*使用模板创建视图*，展示了如何利用 SQLAlchemy 模型通过 Flask 的模板系统 Jinja 动态创建 HTML。

第四章，*使用蓝图创建控制器*，介绍了如何使用 Flask 的蓝图功能来组织视图代码，同时避免重复。

第五章，*高级应用结构*，利用前四章学到的知识，解释了如何重新组织代码文件以创建更易于维护和测试的应用程序结构。

第六章，*保护您的应用*，解释了如何使用各种 Flask 扩展来添加带有基于权限访问的登录系统。

第七章，*在 Flask 中使用 NoSQL*，展示了 NoSQL 数据库是什么，以及如何在它允许更强大的功能时将其集成到您的应用程序中。

第八章，*构建 RESTful API*，展示了如何以安全且易于使用的方式将应用程序数据库中存储的数据提供给第三方。

第九章，*使用 Celery 创建异步任务*，解释了如何将耗时的程序移到后台，以便应用程序不会变慢。

第十章，*有用的 Flask 扩展*，解释了如何利用流行的 Flask 扩展来使您的应用更快，添加更多功能，并使调试更容易。

第十一章，*构建您的扩展*，教您如何了解 Flask 扩展的工作原理以及如何创建自己的扩展。

第十二章，*测试 Flask 应用*，解释了如何添加单元测试和用户界面测试到您的应用中，以确保质量并减少错误代码的数量。

第十三章，*部署 Flask 应用*，解释了如何将完成的应用程序从开发状态部署到实时服务器上。

# 为了充分利用本书

要开始使用本书，您只需要选择一个文本编辑器，一个网络

浏览器，以及机器上安装的 Python。

Windows、Mac OS X 和 Linux 用户都应能够轻松地跟随

阅读本书的内容。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入本书的名称，并遵循屏幕上的说明。

下载文件后，请确保使用最新版本的软件解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

书籍的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Mastering-Flask-Web-Development-Second-Edition`](https://github.com/PacktPublishing/Mastering-Flask-Web-Development-Second-Edition)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。去看看吧！

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“将下载的`WebStorm-10*.dmg`磁盘映像文件作为系统中的另一个磁盘挂载。”

代码块如下设置：

```py
from flask import g
....
# Set some key with some value on a request context
g.some_key = "some_value"
# Get a key
v = g.some_key
# Get and remove a key
v = g.pop('some_key', "default_if_not_present")
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()
```

任何命令行输入或输出都应如下编写：

```py
$ source env/bin/activate
$ pip install -r requirements.txt
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要注意事项看起来是这样的。

小贴士和技巧看起来是这样的。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`给我们发邮件。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告，我们将非常感谢。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解有关 Packt 的更多信息，请访问[packt.com](http://www.packt.com/)。
