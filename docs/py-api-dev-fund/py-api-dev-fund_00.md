# 前言

## 关于

本节简要介绍了作者、本书涵盖的内容、开始学习所需的技术技能，以及完成所有包含的活动和练习所需的硬件和软件要求。

## 关于本书

Python 是一种灵活的语言，它可以用于比脚本开发更多的用途。通过了解 Python RESTful API 的工作原理，你可以使用 Python 构建强大的后端，用于 Web 应用和移动应用。

你将通过构建一个简单的 API 并学习前端 Web 界面如何与后端通信来迈出第一步。你还将学习如何使用 marshmallow 库序列化和反序列化对象。然后，你将学习如何使用 Flask-JWT 进行用户认证和授权。除了所有这些，你还将学习如何通过添加有用的功能来增强你的 API，例如电子邮件、图片上传、搜索和分页。你将通过将 API 部署到云中来结束整本书的学习。

到本书结束时，你将拥有信心和技能，利用 RESTful API 和 Python 的力量构建高效的 Web 应用。

### 关于作者

**陈杰**在 10 岁时开始编程。他在大学期间是全世界编程竞赛的积极参与者。毕业后，他在金融和 IT 行业工作了 10 多年，构建了分析数百万笔交易和头寸以发现可疑活动的系统。他利用强大的 Python 分析库为工作在纳秒级的交易系统进行数据分析性能优化。他对现代软件开发生命周期有深入的了解，该生命周期使用自动化测试、持续集成和敏捷方法。在所有编程语言中，他发现 Python 是最具表现力和功能性的。他创建了课程并在世界各地教授学生，使用 Python 作为教学语言。激励有志于软件工程职业道路的开发者一直是陈杰的目标。

**钟瑞**是一位开发者和讲师。他热爱帮助学生学习编程和掌握软件开发。他现在是自雇人士，使用 Python 开发 Web 应用、网络应用和聊天机器人。他出售的第一个程序是一个网络应用，帮助客户配置、维护和测试数千个多厂商网络设备。他参与过大型项目，如马拉松在线注册系统、租车管理系统等。他与 Google App Engine、PostgreSQL 和高级系统架构设计有广泛的工作经验。他多年来一直是一位自学成才的开发者，并知道学习新技能的最高效方法。

**黄杰**是一位拥有超过 7 年经验的程序员，擅长使用 Python、Javascript 和.NET 开发 Web 应用程序。他精通 Flask、Django 和 Vue 等 Web 框架，以及 PostgreSQL、DynamoDB、MongoDB、RabbitMQ、Redis、Elasticsearch、RESTful API 设计、支付处理、系统架构设计、数据库设计和 Unix 系统。他为配件商店平台、ERP 系统、占卜 Web 应用程序、播客平台、求职服务、博客系统、沙龙预约系统、电子商务服务等多个项目编写了应用程序。他还拥有处理大量数据和优化支付处理的经验。他是一位热爱编码并不断跟踪最新技术的专家级 Web 应用程序开发者。

### 学习目标

在本书结束时，您将能够：

+   理解 RESTful API 的概念

+   使用 Flask 和 Flask-Restful 扩展构建 RESTful API

+   使用 Flask-SQLAlchemy 和 Flask-Migrate 操作数据库

+   使用 Mailgun API 发送纯文本和 HTML 格式的电子邮件

+   使用 Flask-SQLAlchemy 实现分页功能

+   使用缓存来提高 API 性能并高效地获取最新信息

+   将应用程序部署到 Heroku 并使用 Postman 进行测试

### 目标受众

本书非常适合对 Python 编程有基础到中级知识的软件开发初学者，他们希望使用 Python 开发 Web 应用程序。了解 Web 应用程序的工作原理将有所帮助，但不是必需的。

### 方法

本书采用实践学习的方法向您解释概念。您将通过实现您在理论上学到的每个概念来构建一个真实的 Web 应用程序。这样，您将巩固您的新技能。

### 硬件要求

为了获得最佳体验，我们建议以下硬件配置：

+   处理器：Intel Core i5 或等效处理器

+   内存：4 GB RAM（8 GB 更佳）

+   存储：35 GB 可用空间

### 软件要求

我们还建议您提前安装以下软件：

+   操作系统：Windows 7 SP1 64 位、Windows 8.1 64 位或 Windows 10 64 位、Ubuntu Linux 或最新版本的 OS X

+   浏览器：Google Chrome/Mozilla Firefox（最新版本）

+   Python 3.4+（最新版本是 Python 3.8：从`https://python.org`）

+   Pycharm

+   Postman

+   Postgres 数据库

### 惯例

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称的显示方式如下：

"接下来，我们将处理`create_recipe`函数，该函数在内存中创建一个食谱。使用`/recipes`路由来触发`create_recipe`函数，并通过`methods = [POST]`参数指定该路由装饰器仅响应 POST 请求。"

新术语和重要词汇以粗体显示。屏幕上看到的单词，例如在菜单或对话框中，在文本中显示如下：“然后，选择**定义**并设置密码。点击**保存**”。

代码块设置如下：

```py
    if not recipe:
        return jsonify({'message': 'recipe not found'}), HTTPStatus.NOT_FOUND
```

### 安装和设置

在我们能够用数据做些酷的事情之前，我们需要准备好最高效的环境。在本节中，我们将看到如何做到这一点。

**安装 Python**

访问 [`www.python.org/downloads/`](https://www.python.org/downloads/) 并按照您平台的具体说明进行操作。

**安装 Pycharm 社区版**

访问 [`www.jetbrains.com/pycharm/download/`](https://www.jetbrains.com/pycharm/download/) 并按照您平台的具体说明进行操作。

**安装 Postman**

访问 [`www.getpostman.com/downloads/`](https://www.getpostman.com/downloads/) 并按照您平台的具体说明进行操作。

**安装 Postgres 数据库**

我们将在本地机器上安装 Postgres：

1.  访问 [`www.postgresql.org`](http://www.postgresql.org) 并点击 **下载** 进入下载页面。

1.  根据您的操作系统选择 macOS 或 Windows。

1.  在 **EnterpriseDB 的交互式安装程序** 下，下载最新版本的安装程序。安装程序包含 PostgreSQL 以及 pgAdmin，这是一个用于管理和开发您数据库的图形工具。

1.  安装 Postgres 版本 11.4。按照屏幕上的说明安装 Postgres 并设置密码。

1.  安装完成后，您将被带到 pgAdmin。请设置 pgAdmin 密码。

### 其他资源

本书代码包托管在 GitHub 上，网址为 [`github.com/TrainingByPackt/Python-API-Development-Fundamentals`](https://github.com/TrainingByPackt/Python-API-Development-Fundamentals)。我们还有其他来自我们丰富课程和视频目录的代码包，可在 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/) 找到。查看它们！
