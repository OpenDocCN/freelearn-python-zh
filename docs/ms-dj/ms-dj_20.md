# 第二十章：更多关于安装 Django 的信息

本章涵盖了与安装和维护 Django 相关的一些常见附加选项和场景。首先，我们将看看除了 SQLite 之外使用其他数据库的安装配置，然后我们将介绍如何升级 Django 以及如何手动安装 Django。最后，我们将介绍如何安装 Django 的开发版本，以防您想要尝试 Django 开发的最前沿。

# 运行其他数据库

如果您计划使用 Django 的数据库 API 功能，则需要确保数据库服务器正在运行。Django 支持许多不同的数据库服务器，并且官方支持 PostgreSQL、MySQL、Oracle 和 SQLite。

第二十一章，*高级数据库管理*，包含了连接 Django 到这些数据库的额外信息，但是，本书的范围不包括向您展示如何安装它们；请参考每个项目网站上的数据库文档。

如果您正在开发一个简单的项目或者您不打算在生产环境中部署，SQLite 通常是最简单的选择，因为它不需要运行单独的服务器。但是，SQLite 与其他数据库有许多不同之处，因此，如果您正在开发一些实质性的东西，建议使用与生产环境中计划使用的相同数据库进行开发。

除了数据库后端，您还需要确保安装了 Python 数据库绑定。

+   如果您使用 PostgreSQL，则需要`postgresql_psycopg2`（[`initd.org/psycopg/`](http://initd.org/psycopg/)）包。您可能需要参考 PostgreSQL 的注意事项，以获取有关此数据库的进一步技术细节。如果您使用 Windows，请查看非官方编译的 Windows 版本（[`stickpeople.com/projects/python/win-psycopg/`](http://stickpeople.com/projects/python/win-psycopg/)）。

+   如果您正在使用 MySQL，则需要`MySQL-python`包，版本为 1.2.1p2 或更高版本。您还需要阅读 MySQL 后端的特定于数据库的注意事项。

+   如果您使用 SQLite，您可能需要阅读 SQLite 后端的注意事项。

+   如果您使用 Oracle，则需要`cx_Oracle`的副本（[`cx-oracle.sourceforge.net/`](http://cx-oracle.sourceforge.net/)），但请阅读有关 Oracle 后端的特定于数据库的注意事项，以获取有关 Oracle 和`cx_Oracle`支持版本的重要信息。

+   如果您使用非官方的第三方后端，请查阅所提供的文档以获取任何额外要求。

如果您计划使用 Django 的`manage.py migrate`命令自动为模型创建数据库表（在安装 Django 并创建项目后），您需要确保 Django 有权限在您使用的数据库中创建和更改表；如果您计划手动创建表，您可以简单地授予 Django`SELECT`、`INSERT`、`UPDATE`和`DELETE`权限。在创建具有这些权限的数据库用户后，您将在项目的设置文件中指定详细信息，请参阅`DATABASES`以获取详细信息。

如果您使用 Django 的测试框架来测试数据库查询，Django 将需要权限来创建测试数据库。

# 手动安装 Django

1.  从 Django 项目下载页面下载最新版本的发布版（[`www.djangoproject.com/download/`](https://www.djangoproject.com/download/)）。

1.  解压下载的文件（例如，`tar xzvf Django-X.Y.tar.gz`，其中`X.Y`是最新发布版的版本号）。如果您使用 Windows，您可以下载命令行工具`bsdtar`来执行此操作，或者您可以使用基于 GUI 的工具，如 7-zip（[`www.7-zip.org/`](http://www.7-zip.org/)）。

1.  切换到步骤 2 中创建的目录（例如，`cd Django-X.Y`）。

1.  如果您使用 Linux、Mac OS X 或其他 Unix 变种，请在 shell 提示符下输入`sudo python setup.py install`命令。如果您使用 Windows，请以管理员权限启动命令 shell，并运行`python setup.py install`命令。这将在 Python 安装的`site-packages`目录中安装 Django。

### 注意

**删除旧版本**

如果您使用此安装技术，特别重要的是首先删除任何现有的 Django 安装（请参见下文）。否则，您可能会得到一个包含自 Django 已删除的以前版本的文件的损坏安装。

# 升级 Django

## 删除任何旧版本的 Django

如果您正在从以前的版本升级 Django 安装，您需要在安装新版本之前卸载旧的 Django 版本。

如果以前使用`pip`或`easy_install`安装了 Django，则再次使用`pip`或`easy_install`安装将自动处理旧版本，因此您无需自己操作。

如果您以前手动安装了 Django，卸载就像删除 Python `site-packages`中的`django`目录一样简单。要找到需要删除的目录，您可以在 shell 提示符（而不是交互式 Python 提示符）下运行以下命令：

`python -c "import sys; sys.path = sys.path[1:]; import django; print(django.__path__)"`

# 安装特定于发行版的软件包

检查特定于发行版的说明，看看您的平台/发行版是否提供官方的 Django 软件包/安装程序。发行版提供的软件包通常允许自动安装依赖项和简单的升级路径；但是，这些软件包很少包含 Django 的最新版本。

# 安装开发版本

如果您决定使用 Django 的最新开发版本，您需要密切关注开发时间表，并且需要关注即将发布的版本的发布说明。这将帮助您了解您可能想要使用的任何新功能，以及在更新 Django 副本时需要进行的任何更改。（对于稳定版本，任何必要的更改都在发布说明中记录。）

如果您希望偶尔能够使用最新的错误修复和改进更新 Django 代码，请按照以下说明操作：

1.  确保已安装 Git，并且可以从 shell 运行其命令。（在 shell 提示符处输入`git help`来测试这一点。）

1.  像这样查看 Django 的主要开发分支（*trunk*或*master*）：

```py
 git clone 
      git://github.com/django/django.git django-trunk

```

1.  这将在当前目录中创建一个名为`django-trunk`的目录。

1.  确保 Python 解释器可以加载 Django 的代码。最方便的方法是通过 pip。运行以下命令：

```py
 sudo pip install -e django-trunk/

```

1.  （如果使用`virtualenv`，或者运行 Windows，可以省略`sudo`。）

这将使 Django 的代码可导入，并且还将使`django-admin`实用程序命令可用。换句话说，您已经准备好了！

### 注意

不要运行`sudo python setup.py install`，因为您已经在第 3 步中执行了相应的操作。

当您想要更新 Django 源代码的副本时，只需在`django-trunk`目录中运行`git pull`命令。这样做时，Git 将自动下载任何更改。

# 接下来是什么？

在下一章中，我们将介绍有关在特定数据库上运行 Django 的附加信息
