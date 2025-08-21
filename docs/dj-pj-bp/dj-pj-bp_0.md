# 前言

Django 可能是当今最流行的 Web 开发框架之一。这是大多数 Python 开发人员在开发任何规模的 Web 应用程序时会选择的框架。

凭借其经过验证的性能、可扩展性和安全性记录，以及其著名的一揽子方法，Django 被一些行业巨头使用，包括 Instagram、Pinterest 和 National Geographic。

本书适用于对 Django 有初步了解并对如何使用它创建简单网站有基本概念的人。它将向您展示如何将您的技能提升到下一个级别，开发像电子商务网站这样复杂的应用程序，并实现快速搜索。

# 本书涵盖的内容

第一章，“Blueblog – 一个博客平台”，带您开始使用 Django，并介绍如何使用该框架的基本概念。它还向您介绍了本书其余部分使用的开发技术。

第二章，“Discuss – 一个 Hacker News 克隆”，带您创建一个类似流行的 Hacker News 讨论论坛的 Web 应用程序。我们将介绍高级技术，根据用户反馈对 Web 应用程序的内容进行排序和排名，然后介绍防止垃圾邮件的技术。

第三章，“Djagios – 一个基于 Django 的 Nagios 克隆”，涵盖了使用 Django 创建类似 Nagios 的应用程序，可以监视和报告远程服务器系统状态。

第四章，“汽车租赁应用程序”，向您展示如何创建汽车租赁应用程序，并自定义 Django 管理应用程序，为我们的用户提供功能齐全的内容管理系统。

第五章，“多语言电影数据库”，帮助您创建类似 IMDB 的电影网站列表，允许用户对电影进行评论和评价。本章的主要重点是允许您的 Web 应用程序以多种语言提供国际化和本地化版本。

第六章，“Daintree – 一个电子商务网站”，向您展示如何使用 Elasticsearch 搜索服务器软件和 Django 创建类似亚马逊的电子商务网站，实现快速搜索。

第七章，“Form Mason – 自己的猴子”，帮助您创建一个复杂而有趣的 Web 应用程序，允许用户动态定义 Web 表单，然后要求其他人回答这些表单，这与 SurveyMonkey 和其他类似网站的性质相似。

附录，“开发环境设置详细信息和调试技术”，在这里我们将深入研究设置的细节，并解释我们采取的每个步骤。我们还将看到一种调试 Django 应用程序的技术。

# 本书所需内容

要创建和运行本书中将开发的所有 Web 应用程序，您需要以下软件的工作副本：

+   Python 编程语言

+   pip：用于安装 Python 包的软件包管理器

+   virtualenv：用于创建 Python 包的隔离环境的工具

您可以从[`www.python.org/downloads/`](https://www.python.org/downloads/)下载适用于您操作系统的 Python 编程语言。您需要 Python 3 来跟随本书中的示例。

您可以在[`pip.pypa.io/en/stable/installing/`](https://pip.pypa.io/en/stable/installing/)找到安装 pip 软件包管理工具的说明。

您可以按照以下链接中的说明安装 virtualenv：[`virtualenv.pypa.io/en/latest/installation.html`](https://virtualenv.pypa.io/en/latest/installation.html)。

# 这本书适合谁

如果您是一名 Django 网络开发人员，能够使用该框架构建基本的网络应用程序，那么这本书适合您。本书将通过引导您开发六个令人惊叹的网络应用程序，帮助您更深入地了解 Django 网络框架。

# 约定

在本书中，您会发现一些文本样式，用于区分不同类型的信息。以下是这些样式的一些示例以及它们的含义解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名显示如下：“我们可以通过使用`include`指令来包含其他上下文。”

代码块设置如下：

```py
[default]
exten => s,1,Dial(Zap/1|30)
exten => s,2,Voicemail(u100)
exten => s,102,Voicemail(b100)
exten => i,1,Voicemail(s0)
```

当我们希望引起您对代码块的特定部分的注意时，相关的行或项目会以粗体显示：

```py
[default]
exten => s,1,Dial(Zap/1|30)
exten => s,2,Voicemail(u100)
exten => s,102,Voicemail(b100)
exten => i,1,Voicemail(s0)
```

任何命令行输入或输出都以以下方式编写：

```py
# cp /usr/src/asterisk-addons/configs/cdr_mysql.conf.sample
 /etc/asterisk/cdr_mysql.conf

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，会在文本中显示为：“单击**下一步**按钮将您移动到下一个屏幕。”

### 注意

警告或重要提示会以这样的方式出现在一个框中。

### 提示

提示和技巧会以这样的方式出现。
