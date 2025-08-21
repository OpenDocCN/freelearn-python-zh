# 前言

谁没有一个想要推出的下一个伟大应用或服务的想法？然而，大多数应用程序、服务和网站最终都依赖于服务器能够接受请求，然后根据这些请求创建、读取、更新和删除记录。Django 使得构建和启动网站、服务和后端变得容易。然而，尽管它在大规模成功的初创公司和企业中被使用的历史，但要收集实际将一个想法从空目录到运行生产服务器所需的所有资源可能是困难的。

在三个项目的过程中，《构建 Django Web 应用程序》指导您从一个空目录到创建全功能应用程序，以复制一些最受欢迎的网络应用程序的核心功能。在第一部分，您将创建自己的在线电影数据库。在第二部分，您将创建一个让用户提问和回答问题的网站。在第三部分，您将创建一个用于管理邮件列表和发送电子邮件的 Web 应用程序。所有三个项目都将最终部署到服务器上，以便您可以看到自己的想法变为现实。在开始每个项目和部署它之间，我们将涵盖重要的实用概念，如如何构建 API、保护您的项目、使用 Elasticsearch 添加搜索、使用缓存和将任务卸载到工作进程以帮助您的项目扩展。

《构建 Django Web 应用程序》适用于已经了解 Python 基础知识，但希望将自己的技能提升到更高水平的开发人员。还建议具有基本的 HTML 和 CSS 理解，因为这些语言将被提及，但不是本书的重点。

阅读完本书后，您将熟悉使用 Django 启动惊人的 Web 应用程序所需的一切。

# 这本书是为谁准备的

这本书是为熟悉 Python 的开发人员准备的。读者应该知道如何在 Bash shell 中运行命令。假定具有一些基本的 HTML 和 CSS 知识。最后，读者应该能够自己连接到 PostgreSQL 数据库。

# 充分利用本书

要充分利用本书，您应该：

1.  对 Python 有一定了解，并已安装 Python3.6+

1.  能够在计算机上安装 Docker 或其他新软件

1.  知道如何从计算机连接到 Postgres 服务器

1.  可以访问 Bash shell

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip 适用于 Windows

+   Zipeg/iZip/UnRarX 适用于 Mac

+   7-Zip/PeaZip 适用于 Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Building-Django-2.0-Web-Applications`](https://github.com/PacktPublishing/Building-Django-2.0-Web-Applications)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有来自我们丰富的图书和视频目录的其他代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。去看看吧！

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：指示文本中的代码字词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。以下是一个示例：“它还提供了一个`create()`方法，用于创建和保存实例。”

代码块设置如下：

```py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
```

当我们希望引起您对代码块的特定部分的注意时，相关的行或项目会以粗体显示：

```py
DATABASES = {
  'default': {
 'ENGINE': 'django.db.backends.sqlite3',
     'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),  }
}
```

任何命令行输入或输出都以以下方式书写：

```py
$ pip install -r requirements.dev.txt
```

**粗体**：表示一个新术语，一个重要的词，或者屏幕上看到的词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这是一个例子：“点击 MOVIES 将显示给我们一个电影列表。”

警告或重要说明会以这种方式出现。

提示和技巧会以这种方式出现。
