# 前言

如果你在过去 20 年里一直在软件开发行业中，那么你肯定听说过一种名为 Python 的编程语言。Python 由 Guido van Rossum 创建，于 1991 年首次亮相，并自那时起就一直受到全球许多软件开发人员的喜爱。

然而，一个已经有 20 多年历史的语言为何仍然存在，并且每天都在变得越来越受欢迎呢？

嗯，这个问题的答案很简单。Python 对于一切（或几乎一切）都很棒。Python 是一种通用编程语言，这意味着你可以创建简单的终端应用程序、Web 应用程序、微服务、游戏，以及复杂的科学应用程序。尽管可以用 Python 来实现不同的目的，但 Python 是一种以易学著称的语言，非常适合初学者以及没有计算机科学背景的人。

Python 是一种“电池包含”的编程语言，这意味着大多数时候在开发项目时你不需要使用任何外部依赖。Python 的标准库功能丰富，大多数时候包含了你创建程序所需的一切，而且即使你需要标准库中没有的东西，PyPI（Python 包索引）目前也包含了 117,652 个包。

Python 社区是一个欢迎、乐于助人、多元化且对这门语言非常热情的社区，社区中的每个人都乐意互相帮助。

如果你还不相信，知名网站 StackOverflow 发布了今年关于编程语言受欢迎程度的统计数据，基于用户在网站上提出的问题数量，Python 是排名前列的语言，仅次于 JavaScript、Java、C#和 PHP。

现在是成为 Python 开发者的完美时机，所以让我们开始吧！

# 本书适合对象

这本书适用于熟悉 Python 并希望通过网络和软件开发项目获得实践经验的软件开发人员。需要有 Python 编程的基础知识。

# 本书内容包括

第一章，*实现天气应用程序*，将指导你开发一个终端应用程序，显示特定地区的当前天气和未来 5 天的预报。本章将介绍 Python 编程的基本概念。你将学习如何解析命令行参数以增加程序的交互性，并最终学会如何使用流行的 Beautiful Soup 框架从网站上抓取数据。

第二章，*使用 Spotify 创建远程控制应用程序*，将教你如何使用 OAuth 对 Spotify API 进行身份验证。我们将使用 curses 库使应用程序更有趣和用户友好。

第三章，*在 Twitter 上投票*，将教你如何使用 Tkinter 库使用 Python 创建美观的用户界面。我们将使用 Python 的 Reactive Extensions 来检测后端的投票情况，然后在用户界面中发布更改。

第四章，*汇率和货币转换工具*，将使你能够实现一个货币转换器，它将实时从不同来源获取外汇汇率，并使用数据进行货币转换。我们将开发一个包含辅助函数来执行转换的 API。首先，我们将使用开源外汇汇率和货币转换 API（[`fixer.io/`](http://fixer.io/)）。

本章的第二部分将教你如何创建一个命令行应用程序，利用我们的 API 从数据源获取数据，并使用一些参数获取货币转换结果。

第五章《使用微服务构建 Web Messenger》将教您如何使用 Nameko，这是 Python 的微服务框架。您还将学习如何为外部资源（如 Redis）创建依赖项提供程序。本章还将涉及对 Nameko 服务进行集成测试以及对 API 的基本 AJAX 请求。

第六章《使用用户认证微服务扩展 TempMessenger》将在第五章《使用微服务构建 Web Messenger》的基础上构建您的应用程序。您将创建一个用户认证微服务，将用户存储在 Postgres 数据库中。使用 Bcrypt，您还将学习如何安全地将密码存储在数据库中。本章还涵盖了创建 Flask Web 界面以及如何利用 cookie 存储 Web 会话数据。通过这些章节的学习，您将能够创建可扩展和协调的微服务。

第七章《使用 Django 创建在线视频游戏商店》将使您能够创建一个在线视频游戏商店。它将包含浏览不同类别的视频游戏、使用不同标准进行搜索、查看每个游戏的详细信息，最后将游戏添加到购物车并下订单等功能。在这里，您将学习 Django 2.0、管理 UI、Django 数据模型等内容。

第八章《订单微服务》将帮助您构建一个负责接收来自我们在上一章中开发的 Web 应用程序的订单的微服务。订单微服务还提供其他功能，如更新订单状态和使用不同标准提供订单信息。

第九章《通知无服务器应用》将教您有关无服务器函数架构以及如何使用 Flask 构建通知服务，并使用伟大的项目 Zappa 将最终应用部署到 AWS Lambda。您还将学习如何将在第七章《使用 Django 创建在线视频游戏商店》中开发的 Web 应用程序和在第八章《订单微服务》中开发的订单微服务与无服务器通知应用集成。

# 为了充分利用本书

为了在本地计算机上执行本书中的代码，您需要以下内容：

+   互联网连接

+   Virtualenv

+   Python 3.6

+   MongoDB 3.2.11

+   pgAdmin（参考官方文档[`url.marcuspen.com/pgadmin`](http://url.marcuspen.com/pgadmin)进行安装）

+   Docker（参考官方文档[`url.marcuspen.com/docker-install`](http://url.marcuspen.com/docker-install)进行安装）

随着我们逐步学习，所有其他要求都将被安装。

本章中的所有说明都针对 macOS 或 Debian/Ubuntu 系统；但是，作者已经注意只使用跨平台依赖项。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“SUPPORT”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名并按照屏幕上的说明进行操作。

下载文件后，请确保使用以下最新版本解压或提取文件夹：

+   Windows 的 WinRAR/7-Zip

+   Mac 的 Zipeg/iZip/UnRarX

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Python-Programming-Blueprints`](https://github.com/PacktPublishing/Python-Programming-Blueprints)。我们还有来自丰富书籍和视频目录的其他代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子：“这个方法将调用`Runner`的`exec`方法来执行执行请求 Twitter API 的函数。”

代码块设置如下：

```py
def set_header(self):
    title = Label(self,
                  text='Voting for hasthags',
                  font=("Helvetica", 24),
                  height=4)
    title.pack()
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
def start_app(args):
    root = Tk()
    app = Application(hashtags=args.hashtags, master=root)
    app.master.title("Twitter votes")
    app.master.geometry("400x700+100+100")
    app.mainloop()
```

任何命令行输入或输出都以以下方式编写：

```py
python app.py --hashtags debian ubuntu arch
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这是一个例子：“它说，以您的用户名登录，然后在其后有一个注销链接。试一试，点击链接注销。”

警告或重要说明会显示为这样。提示和技巧会显示为这样。
