# 前言

Python 是一种动态类型的解释语言，可以快速构建各种领域的应用程序，包括人工智能、桌面和 Web 应用程序。

随着 Python 生态系统的最新进展和支持高度可重用性并使模块化代码编译成为可能的大量库的可用性，Python 可以用于构建能够解决组织问题的应用程序。这些应用程序可以在短时间内开发，并且如果开发得当，可以以一种解决组织需求的方式进行扩展。

Python 3.7 版本带来了几项改进和新功能，使应用程序开发变得轻松。连同...

# 这本书适合谁

企业应用程序是旨在解决组织特定业务需求的关键应用程序。企业应用程序的要求与个人通常需要的要求大不相同。这些应用程序应提供高性能和可扩展性，以满足组织日益增长的需求。

考虑到这一点，本书适用于具有 Python 编程中级知识并愿意深入了解根据组织需求进行扩展的应用程序构建的开发人员。本书提供了几个示例，可以在运行在 Linux 发行版上的 Python 3.7 上执行，但也适用于其他操作系统。

为了充分利用本书，您必须对基本操作系统概念有基本的了解，例如进程管理和多线程。除此之外，对数据库系统的基本工作知识可能有益，但不是强制性的。

熟悉 Python 应用程序构建不同方面的开发人员可以学习有助于构建可扩展应用程序的工具和技术，并了解企业应用程序开发方法的想法。

# 充分利用本书

除了对编程有一般了解外，不需要特定的专业知识才能利用本书。

Odoo 是使用 Python 构建的，因此对该语言有扎实的了解是个好主意。我们还选择在 Ubuntu 主机上运行 Odoo（一种流行的云托管选项），并且将在命令行上进行一些工作，因此一些熟悉将是有益的。

为了充分利用本书，我们建议您找到关于 Python 编程语言、Ubuntu/Debian Linux 操作系统和 PostgreSQL 数据库的辅助阅读。

尽管我们将在 Ubuntu 主机上运行 Odoo，但我们还将提供关于如何在...设置开发环境的指导

# 下载示例代码文件

您可以从[www.packt.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packt.com/support](http://www.packtpub.com/support)并注册，以便直接将文件发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packtpub.com/support)上登录或注册。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   Windows 的 WinRAR/7-Zip

+   Mac 的 Zipeg/iZip/UnRarX

+   Linux 的 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python`](https://github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python)。我们还有其他代码包来自我们丰富的书籍和视频目录，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄。以下是一个例子：“除了这三个包，读者还需要`sqlalchemy`包，它提供了我们将在整个章节中使用的 ORM，以及`psycopg2`，它提供了`postgres`数据库绑定，允许`sqlalchemy`连接到`postgres`。”

代码块设置如下：

```py
username = request.args.get('username')email = request.args.get('email')password = request.args.get('password')user_record = User(username=username, email=email, password=password)
```

当我们希望绘制...
