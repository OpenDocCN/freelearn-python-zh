# 第十章：现在怎么办？

Flask 目前是最受欢迎的 Web 框架，因此为它找到在线阅读材料并不难。例如，快速在谷歌上搜索肯定会给你找到一两篇你感兴趣的好文章。尽管如此，像部署这样的主题，尽管在互联网上讨论得很多，但仍然会在我们的网页战士心中引起怀疑。因此，我们在最后一章中提供了一个很好的“像老板一样部署你的 Flask 应用程序”的逐步指南。除此之外，我们还会建议你一些非常特别的地方，那里的知识就在那里，浓厚而丰富，等着你去获取智慧。通过这一章，你将能够将你的产品从代码部署到服务器，也许，只是也许，能够得到一些应得的高分！欢迎来到这一章，在这里代码遇见服务器，你遇见世界！

# 你的部署比我的前任好

部署不是每个人都熟悉的术语；如果你最近还不是一个 Web 开发人员，你可能对它不太熟悉。以一种粗犷的斯巴达方式，我们可以将部署定义为准备和展示你的应用程序给世界的行为，确保所需的资源可用，并对其进行调整，因为适合开发阶段的配置与适合部署的配置是不同的。在 Web 开发的背景下，我们谈论的是一些非常具体的行动：

+   将你的代码放在服务器上

+   设置你的数据库

+   设置你的 HTTP 服务器

+   设置你可能使用的其他服务

+   将所有内容联系在一起

## 将你的代码放在服务器上

首先，什么是服务器？我们所说的服务器是指具有高可靠性、可用性和可维护性（RAS）等服务器特性的计算机。这些特性使服务器中运行的应用程序获得一定程度的信任，即使在出现任何环境问题（如硬件故障）之后，服务器也会继续运行。

在现实世界中，人们有预算，一个普通的计算机（你在最近的商店买的那种）很可能是运行小型应用程序的最佳选择，因为“真正的服务器”非常昂贵。对于小项目预算（现在也包括大项目），创建了一种称为服务器虚拟化的强大解决方案，其中昂贵的高 RAS 物理服务器的资源（内存、CPU、硬盘等）被虚拟化成虚拟机（VM），它们就像真实硬件的较小（更便宜）版本一样。像 DigitalOcean（https://digitalocean.com/）、Linode（https://www.linode.com/）和 RamNode（https://www.ramnode.com/）这样的公司专注于向公众提供廉价可靠的虚拟机。

现在，鉴于我们的 Web 应用程序已经准备就绪（我的意思是，我们的最小可行产品已经准备就绪），我们必须在某个对我们的目标受众可访问的地方运行代码。这通常意味着我们需要一个 Web 服务器。从前面一段提到的公司中选择两台便宜的虚拟机，使用 Ubuntu 进行设置，然后让我们开始吧！

## 设置你的数据库

关于数据库，部署过程中你应该知道的最基本的事情之一是，最好的做法是让你的数据库和 Web 应用程序在不同的（虚拟）机器上运行。你不希望它们竞争相同的资源，相信我。这就是为什么我们雇了两台虚拟服务器——一台将运行我们的 HTTP 服务器，另一台将运行我们的数据库。

让我们开始设置数据库服务器；首先，我们将我们的 SSH 凭据添加到远程服务器，这样我们就可以在不需要每次输入远程服务器用户密码的情况下进行身份验证。在此之前，如果你没有它们，生成你的 SSH 密钥，就像这样：

```py
# ref: https://help.github.com/articles/generating-ssh-keys/
# type a passphrase when asked for one
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

```

现在，假设您的虚拟机提供程序为您的远程机器提供了 IP 地址、root 用户和密码，我们将如下创建一个无密码的 SSH 身份验证与我们的服务器：

```py
# type the root password when requested
ssh-copy-id root@ipaddress

```

现在，退出您的远程终端，尝试 SSH `root@ipaddress`。密码将不再被请求。

第二步！摆脱非数据库的东西，比如 Apache，并安装 Postgres（[`www.postgresql.org/`](http://www.postgresql.org/)），迄今为止最先进的开源数据库：

```py
# as root
apt-get purge apache2-*
apt-get install postgresql
# type to check which version of postgres was installed (most likely 9.x)
psql -V

```

现在我们设置数据库。

将默认用户 Postgres 连接到角色`postgres`：

```py
sudo -u postgres psql

```

为我们的项目创建一个名为`mydb`的数据库：

```py
CREATE DATABASE mydb;

```

创建一个新的用户角色来访问我们的数据库：

```py
CREATE USER you WITH PASSWORD 'passwd'; # please, use a strong password
# We now make sure "you" can do whatever you want with mydb
# You don't want to keep this setup for long, be warned
GRANT ALL PRIVILEGES ON DATABASE mydb TO you;

```

到目前为止，我们已经完成了很多工作。首先，我们删除了不必要的包（只有很少）；安装了我们的数据库 Postgres 的最新支持版本；创建了一个新的数据库和一个新的“用户”；并授予了我们的用户对我们的新数据库的完全权限。让我们了解每一步。

我们首先删除 Apache2 等内容，因为这是一个数据库服务器设置，所以没有必要保留 Apache2 包。根据安装的 Ubuntu 版本，您甚至需要删除其他包。这里的黄金法则是：安装的包越少，我们就要关注的包就越少。只保留最少的包。

然后我们安装 Postgres。根据您的背景，您可能会问——为什么是 Postgres 而不是 MariaDB/MySQL？嗯，嗯，亲爱的读者，Postgres 是一个完整的解决方案，支持 ACID，文档（JSONB）存储，键值存储（使用 HStore），索引，文本搜索，服务器端编程，地理定位（使用 PostGIS）等等。如果您知道如何安装和使用 Postgres，您就可以在一个单一的解决方案中访问所有这些功能。我也更喜欢它比其他开源/免费解决方案，所以我们将坚持使用它。

安装 Postgres 后，我们必须对其进行配置。与我们迄今为止使用的关系数据库解决方案 SQLite 不同，Postgres 具有基于角色的强大权限系统，控制着可以被访问或修改的资源，以及由谁访问或修改。这里的主要概念是，角色是一种非常特殊的组，它可能具有称为**权限**的权限，或者与之相关或包含它的其他组。例如，在`psql`控制台内运行的`CREATE USER`命令（Postgres 的交互式控制台，就像 Python 的）实际上并不是创建一个用户；实际上，它是在创建一个具有登录权限的新角色，这类似于用户概念。以下命令等同于`psql`内的创建用户命令：

```py
CREATE ROLE you WITH LOGIN;

```

现在，朝着我们最后的目标，有`GRANT`命令。为了允许角色执行操作，我们授予它们权限，比如登录权限，允许我们的“用户”登录。在我们的示例中，我们授予您对数据库`mydb`的所有可用权限。我们这样做是为了能够创建表，修改表等等。通常您不希望您的生产 Web 应用程序数据库用户（哇！）拥有所有这些权限，因为在发生安全漏洞时，入侵者将能够对您的数据库执行任何操作。因为通常（咳咳从不！）不会在用户交互中更改数据库结构，所以在 Web 应用程序中使用一个权限较低的用户并不是问题。

### 提示

PgAdmin 是一个令人惊叹的、用户友好的、Postgres 管理应用程序。只需使用 SSH 隧道（[`www.pgadmin.org/docs/dev/connect.html`](http://www.pgadmin.org/docs/dev/connect.html)），就可以快乐了！

现在测试您的数据库设置是否正常工作。从控制台连接到它：

```py
psql -U user_you -d database_mydb -h 127.0.0.1 -W

```

在被要求时输入你的密码。我们之前的命令实际上是我们在使用 Postgres 时使用的一个技巧，因为我们是通过网络接口连接到数据库的。默认情况下，Postgres 假设你试图使用与你的系统用户名相同的角色和数据库进行连接。除非你像我们一样通过网络接口连接，否则你甚至不能以与你的系统用户名不同的角色名称进行连接。

## 设置 web 服务器

设置你的 web 服务器会更加复杂，因为它涉及修改更多的文件，并确保它们之间的配置是稳固的，但我们会做到的，你会看到的。

首先，我们要确保我们的项目代码在我们的 web 服务器上（这不是与数据库服务器相同的服务器，对吧？）。我们可以以多种方式之一来做到这一点：使用 FTP（请不要），简单的 fabric 加 rsync，版本控制，或者版本加 fabric（开心脸！）。让我们看看如何做后者。

假设你已经在你的 web 服务器虚拟机中创建了一个名为`myuser`的常规用户，请确保已经安装了 fabric：

```py
sudo apt-get install python-dev
pip install fabric

```

还有，在你的项目根目录中创建一个名为`fabfile.py`的文件：

```py
# coding:utf-8

from fabric.api import *
from fabric.contrib.files import exists

env.linewise = True
# forward_agent allows you to git pull from your repository
# if you have your ssh key setup
env.forward_agent = True
env.hosts = ['your.host.ip.address']

def create_project():
    if not exists('~/project'):
        run('git clone git://path/to/repo.git')

def update_code():
    with cd('~/project'):
        run('git pull')
def reload():
    "Reloads project instance"
    run('touch --no-dereference /tmp/reload')
```

有了上述代码和安装了 fabric，假设你已经将你的 SSH 密钥复制到了远程服务器，并已经与你的版本控制提供商（例如`github`或`bitbucket`）进行了设置，`create_project`和`update_code`就可以使用了。你可以像这样使用它们：

```py
fab create_project  # creates our project in the home folder of our remote web server
fab update_code  # updates our project code from the version control repository

```

这非常容易。第一条命令将你的代码放入存储库，而第二条命令将其更新到你的最后一次提交。

我们的 web 服务器设置将使用一些非常流行的工具：

+   **uWSGI**：这用于应用服务器和进程管理

+   **Nginx**：这用作我们的 HTTP 服务器

+   **UpStart**：这用于管理我们的 uWSGI 生命周期

UpStart 已经随 Ubuntu 一起提供，所以我们以后会记住它。对于 uWSGI，我们需要像这样安装它：

```py
pip install uwsgi

```

现在，在你的虚拟环境`bin`文件夹中，会有一个 uWSGI 命令。记住它的位置，因为我们很快就会需要它。

在你的项目文件夹中创建一个`wsgi.py`文件，内容如下：

```py
# coding:utf-8
from main import app_factory

app = app_factory(name="myproject")
```

uWSGI 使用上面的文件中的应用实例来连接到我们的应用程序。`app_factory`是一个创建应用程序的工厂函数。到目前为止，我们已经看到了一些。只需确保它返回的应用程序实例已经正确配置。就应用程序而言，这就是我们需要做的。接下来，我们将继续将 uWSGI 连接到我们的应用程序。

我们可以在命令行直接调用我们的 uWSGI 二进制文件，并提供加载 wsgi.py 文件所需的所有参数，或者我们可以创建一个`ini`文件，其中包含所有必要的配置，并将其提供给二进制文件。正如你可能猜到的那样，第二种方法通常更好，因此创建一个看起来像这样的 ini 文件：

```py
[uwsgi]
user-home = /home/your-system-username
project-name = myproject
project-path = %(user-home)/%(myproject)

# make sure paths exist
socket = %(user-home)/%(project-name).sock
pidfile = %(user-home)/%(project-name).pid
logto = /var/tmp/uwsgi.%(prj).log
touch-reload = /tmp/reload
chdir = %(project-path)
wsgi-file = %(project-path)/wsgi.py
callable = app
chmod-socket = 664

master = true
processes = 5
vacuum = true
die-on-term = true
optimize = 2
```

`user-home`，`project-name`和`project-path`是我们用来简化我们工作的别名。`socket`选项指向我们的 HTTP 服务器将用于与我们的应用程序通信的套接字文件。我们不会讨论所有给定的选项，因为这不是 uWSGI 概述，但一些更重要的选项，如`touch-reload`，`wsgi-file`，`callable`和`chmod-socket`，将得到详细的解释。Touch-reload 特别有用；你指定为它的参数的文件将被 uWSGI 监视，每当它被更新/触摸时，你的应用程序将被重新加载。在一些代码更新之后，你肯定想重新加载你的应用程序。Wsgi-file 指定了哪个文件有我们的 WSGI 兼容应用程序，而`callable`告诉 uWSGI wsgi 文件中实例的名称（通常是 app）。最后，我们有 chmod-socket，它将我们的套接字权限更改为`-rw-rw-r--`，即对所有者和组的读/写权限；其他人可能只读取这个。我们需要这样做，因为我们希望我们的应用程序在用户范围内，并且我们的套接字可以从`www-data`用户读取，这是服务器用户。这个设置非常安全，因为应用程序不能干扰系统用户资源之外的任何东西。

我们现在可以设置我们的 HTTP 服务器，这是一个非常简单的步骤。只需按照以下方式安装 Nginx：

```py
sudo apt-get install nginx-full

```

现在，您的 http 服务器在端口 80 上已经运行起来了。让我们确保 Nginx 知道我们的应用程序。将以下代码写入`/etc/nginx/sites-available`中的名为`project`的文件：

```py
server {
    listen 80;
    server_name PROJECT_DOMAIN;

    location /media {
        alias /path/to/media;
    }
    location /static {
        alias /path/to/static;
    }

    location / {
        include         /etc/nginx/uwsgi_params;
        uwsgi_pass      unix:/path/to/socket/file.sock;
    }
}
```

前面的配置文件创建了一个虚拟服务器，运行在端口 80 上，监听域`server_name`，通过`/static`和`/media`提供静态和媒体文件，并监听将所有访问指向`/`的路径，使用我们的套接字处理。我们现在打开我们的配置并关闭 nginx 的默认配置：

```py
sudo rm /etc/nginx/sites-enabled/default
ln -s /etc/nginx/sites-available/project /etc/nginx/sites-enabled/project

```

我们刚刚做了什么？虚拟服务器的配置文件位于`/etc/nginx/sites-available/`中，每当我们希望 nginx 看到一个配置时，我们将其链接到已启用的站点。在前面的配置中，我们刚刚禁用了`default`并通过符号链接启用了`project`。Nginx 不会自行注意到并加载我们刚刚做的事情；我们需要告诉它重新加载其配置。让我们把这一步留到以后。

我们需要在`/etc/init`中创建一个最后的文件，它将使用 upstart 将我们的 uWSGI 进程注册为服务。这部分非常简单；只需创建一个名为`project.conf`（或任何其他有意义的名称）的文件，内容如下：

```py
description "uWSGI application my project"

start on runlevel [2345]
stop on runlevel [!2345]

setuid your-user
setgid www-data

exec /path/to/uwsgi --ini /path/to/ini/file.ini
```

前面的脚本使用我们的项目`ini`文件（我们之前创建的）作为参数运行 uWSGI，用户为"your-user"，组为 www-data。用您的用户替换`your-user`（…），但不要替换`www-data`组，因为这是必需的配置。前面的运行级别配置只是告诉 upstart 何时启动和停止此服务。您不必进行干预。

运行以下命令行来启动您的服务：

```py
start project

```

接下来重新加载 Nginx 配置，就像这样：

```py
sudo /etc/init.d/nginx reload

```

如果一切顺利，媒体路径和静态路径存在，项目数据库设置指向私有网络内的远程服务器，并且上帝对您微笑，您的项目应该可以从您注册的域名访问。击掌！

# StackOverflow

StackOverflow 是新的谷歌术语，用于黑客和软件开发。很多人使用它，所以有很多常见问题和很好的答案供您使用。只需花几个小时阅读关于[`stackoverflow.com/search?q=flask`](http://stackoverflow.com/search?q=flask)的最新趋势，您肯定会学到很多！

# 结构化您的项目

由于 Flask 不强制执行项目结构，您有很大的自由度来尝试最适合您的方式。大型单文件项目可行，类似 Django 的结构化项目可行，平面架构也可行；可能性很多！因此，许多项目都提出了自己建议的架构；这些项目被称为样板或骨架。它们专注于为您提供一个快速启动新 Flask 项目的方法，利用他们建议的代码组织方式。

如果您计划使用 Flask 创建一个大型 Web 应用程序，强烈建议您至少查看其中一个这些项目，因为它们可能已经面临了一些您可能会遇到的问题，并提出了解决方案：

+   Flask-Empty ([`github.com/italomaia/flask-empty`](https://github.com/italomaia/flask-empty))

+   Flask-Boilerplate ([`github.com/mbr/flask-bootstrap`](https://github.com/mbr/flask-bootstrap))

+   Flask-Skeleton ([`github.com/sean-/flask-skeleton`](https://github.com/sean-/flask-skeleton))

# 总结

我必须承认，我写这本书是为了自己。在一个地方找到构建 Web 应用程序所需的所有知识是如此困难，以至于我不得不把我的笔记放在某个地方，浓缩起来。我希望，如果您读到这一段，您也和我一样觉得，这本书是为您写的。这是一次愉快的挑战之旅。

你现在能够构建功能齐全的 Flask 应用程序，包括安全表单、数据库集成、测试，并利用扩展功能，让你能够在短时间内创建强大的软件。我感到非常自豪！现在，去告诉你的朋友你有多棒。再见！

# 附言

作为一个个人挑战，拿出你一直梦想编码的项目，但从未有勇气去做的，然后制作一个 MVP（最小可行产品）。创建你想法的一个非常简单的实现，并将其发布到世界上看看；然后，给我留言。我很乐意看看你的作品！
