# 第十三章：部署 Flask 应用程序

现在我们已经到达了这本书的最后一章，并制作了一个完全功能性的 Flask 网络应用程序，我们开发周期的最后一步是将应用程序提供给全世界。托管您的 Flask 应用程序有众多不同的方法，每种方法都有其自身的优缺点。本章将涵盖最佳解决方案，并指导您在何种情况下选择一种方法而不是另一种。

在本章中，我们将涵盖以下内容：

+   对最常用的网络服务器和网关接口的简要介绍

+   如何在各种云服务上部署

+   如何构建 Docker 镜像

+   如何使用 Docker Compose 描述服务

+   如何使用 AWS CloudFormation (IaC) 描述您的基础设施

+   如何设置和使用 CI/CD 系统以轻松构建、测试、审查和部署我们的应用程序

# 网络服务器和网关接口

在本节中，我们将快速介绍最常用的网络服务器和**Web Server Gateway Interfaces**（**WSGI**），以及它们之间的区别和配置。WSGI 是位于网络服务器和 Python 应用程序本身之间的一种与应用程序无关的层。

# Gevent

获取网络服务器运行的最简单选项是使用名为 `gevent` 的 Python 库来托管您的应用程序。`Gevent` 是一个 Python 库，它为 Python 线程库之外提供了另一种并发编程方式，称为协程。Gevent 提供了一个运行 WSGI 应用程序的接口，它既简单又具有良好性能。一个简单的 gevent 服务器可以轻松处理数百个并发用户，这比互联网上网站的用户多出 99%。这种选择的缺点是它的简单性意味着缺乏配置选项。例如，无法向服务器添加速率限制或添加 HTTPS 流量。这种部署选项纯粹适用于您不期望接收大量流量的网站。记住 YAGNI：只有当您真的需要时，才升级到不同的网络服务器。

协程超出了本书的范围，但可以在[`en.wikipedia.org/wiki/Coroutine`](https://en.wikipedia.org/wiki/Coroutine)找到良好的解释。

要安装 `gevent`，我们将使用以下命令的 `pip`：

```py
    $ pip install gevent
```

在项目目录的根目录下，在名为 `gserver.py` 的新文件中添加以下内容：

```py
    from gevent.wsgi import WSGIServer
    from webapp import create_app

    app = create_app('webapp.config.ProdConfig')

    server = WSGIServer(('', 80), app)
    server.serve_forever()
```

要使用 supervisor 运行服务器，只需将命令值更改为以下内容：

```py
    [program:webapp]
    command=python gserver.py 
    directory=/home/deploy/webapp
    user=deploy
```

现在当您部署时，`gevent` 将在每次部署时通过运行您的 `requirements.txt` 自动为您安装；也就是说，如果您在添加每个新依赖项后正确地使用 pip 冻结。

# Tornado

**Tornado** 是另一种仅使用 Python 部署 WSGI 应用程序的方法。Tornado 是一个设计用来处理数千个并发连接的网络服务器。如果您的应用程序需要实时数据，Tornado 还支持 WebSocket 以实现到服务器的持续、长连接。

不要在 Windows 服务器上以生产模式使用 Tornado。Tornado 的 Windows 版本不仅速度较慢，而且被认为是处于测试阶段的软件。

要使用 Tornado 与我们的应用程序一起使用，我们将使用 Tornado 的`WSGIContainer`来包装应用程序对象，使其与 Tornado 兼容。然后，Tornado 将开始监听端口*80*上的请求，直到进程终止。在新的文件`tserver.py`中添加以下内容：

```py
    from tornado.wsgi import WSGIContainer
    from tornado.httpserver import HTTPServer
    from tornado.ioloop import IOLoop
    from webapp import create_app

    app = WSGIContainer(create_app("webapp.config.ProdConfig"))
    http_server = HTTPServer(app)
    http_server.listen(80)
    IOLoop.instance().start()
```

要以管理员的权限运行 Tornado，只需更改命令值为以下内容：

```py
    [program:webapp]
    command=python tserver.py 
    directory=/home/deploy/webapp
    user=deploy
```

# Nginx 和 uWSGI

如果您需要更好的性能或更多自定义选项，最流行的部署 Python 网络应用程序的方式是使用 Nginx 网络服务器作为基于 WSGI 的 uWSGI 服务器的代理前端。*反向代理*是一种网络程序，它从服务器获取客户端的内容，就像它从代理本身返回一样。这个过程在以下图中展示：

![图片](img/6547d721-3adc-45eb-baa1-850483f73ff3.png)

**Nginx**和**uWSGI**的使用方式是这样的，因为这样我们既获得了 Nginx 前端的强大功能，又具有 uWSGI 的定制性。

**Nginx**是一个非常强大的网络服务器，因其提供了速度和定制的最佳组合而变得流行。Nginx 始终比其他网络服务器，如 Apache 的 httpd，要快，并且具有对 WSGI 应用程序的原生支持。它之所以能够达到这种速度，是因为开发者做出了几个良好的架构决策，并且没有像 Apache 那样试图覆盖大量用例。后一点是在 Nginx 开发早期就做出的决定。具有较小的功能集使得维护和优化代码变得更加容易。从程序员的视角来看，配置 Nginx 也更容易，因为没有巨大的默认配置文件（`httpd.conf`），每个项目目录中都可以用`.htaccess`文件覆盖。

**uWSGI**是一个支持多种不同类型服务器接口的网络服务器，包括 WSGI。uWSGI 处理应用程序内容的提供，以及诸如在多个不同进程和线程之间进行流量负载均衡等事情。

要安装 uWSGI，我们将使用以下`pip`命令：

```py
    $ pip install uwsgi
```

为了运行我们的应用程序，uWSGI 需要一个包含可访问 WSGI 应用程序的文件。在项目目录顶层名为`wsgi.py`的文件中。

要测试 uWSGI，我们可以通过以下命令从**命令行界面**（**CLI**）运行它：

```py
    $ uwsgi --socket 127.0.0.1:8080 
    --wsgi-file wsgi.py 
    --callable app 
    --processes 4 
    --threads 2 
```

如果您在自己的服务器上运行此程序，您应该能够访问端口 8080 并看到您的应用程序（如果您没有防火墙的话）。

此命令的作用是从 `wsgi.py` 文件中加载 app 对象，并使其在 `localhost` 的端口 *8080* 上可访问。它还启动了四个不同的进程，每个进程有两个线程，这些进程由主进程自动负载均衡。对于绝大多数网站来说，这样的进程数量是过度的。为了开始，使用一个进程和两个线程，然后根据需要扩展。

我们不必在 CLI 上添加所有配置选项，可以创建一个文本文件来保存我们的配置，这为我们提供了与 *Gevent* 部分中提到的 supervisor 相同的配置优势。在项目目录的根目录下创建一个名为 `uwsgi.ini` 的文件，并添加以下代码：

```py
    [uwsgi]
    socket = 127.0.0.1:8080
    wsgi-file = wsgi.py
    callable = app
    processes = 4
    threads = 2

```

uWSGI 支持数百个配置选项，以及一些官方和非官方插件。要充分利用 uWSGI 的全部功能，您可以查阅 [`uwsgi-docs.readthedocs.org/`](http://uwsgi-docs.readthedocs.org/) 上的文档。

现在，让我们从 supervisor 运行服务器：

```py
    [program:webapp]
    command=uwsgi uwsgi.ini
    directory=/home/deploy/webapp
    user=deploy

```

由于我们是从操作系统的软件包管理器中安装 Nginx，操作系统将为我们处理 Nginx 的运行。

在撰写本文时，官方 Debian 软件包管理器中的 Nginx 版本已经过时好几年。要安装最新版本，请遵循 [`wiki.nginx.org/Install`](http://wiki.nginx.org/Install) 上提供的说明。

接下来，我们需要创建一个 Nginx 配置文件，然后，当我们推送代码时，我们需要将配置文件复制到 `/etc/nginx/sites-available/` 目录。在项目目录的根目录下创建一个名为 `nginx.conf` 的新文件，并添加以下内容：

```py
server { 
    listen 80; 
    server_name your_domain_name; 

    location / { 
        include uwsgi_params; 
        uwsgi_pass 127.0.0.1:8080; 
    } 

    location /static { 
        alias /home/deploy/webapp/webapp/static; 
    } 
} 
```

此配置文件的作用是告诉 Nginx 监听端口 *80* 上的传入请求，并将所有请求转发到监听端口 *8080* 的 WSGI 应用程序。此外，它为静态文件请求设置了一个例外，并将这些请求直接发送到文件系统。绕过 uWSGI 处理静态文件可以极大地提高性能，因为 Nginx 在快速服务静态文件方面非常出色。

# Apache 和 uWSGI

使用 Apache httpd 与 uWSGI 的设置基本上是相同的。首先，我们需要一个 Apache 配置文件，因此让我们在我们的项目目录根目录下创建一个名为 `apache.conf` 的新文件，并添加以下代码：

```py
<VirtualHost *:80> 
    <Location /> 
        ProxyPass / uwsgi://127.0.0.1:8080/ 
    </Location> 
</VirtualHost> 
```

此文件仅告知 Apache 将所有端口 *80* 上的请求转发到监听端口 *8080* 的 uWSGI 网络服务器。然而，此功能需要从 uWSGI 获取一个额外的 Apache 插件，名为 `mod-proxy-uwsgi`。

接下来，我们将介绍几种在 **平台即服务**（**PaaS**）和 **基础设施即服务**（**IaaS**）工具上部署我们的应用程序的解决方案。您将学习如何创建几种不同类型的环境，并使我们的示例 Blog 应用程序对全世界可用。

# 在 Heroku 上部署

**Heroku** 是本章将要介绍的第一个 **平台即服务**（**PaaS**）提供商。PaaS 是一种服务，提供给网络开发者，使他们能够将网站托管在由他人控制和维护的平台之上。以牺牲一些自由为代价，你可以获得保证，你的网站将自动根据网站的用户数量进行扩展，而无需你做额外的工作。然而，使用 PaaS 工具可能会比运行自己的服务器更昂贵。

Heroku 是一种 PaaS 工具，旨在通过挂钩到现有的工具来为网络开发者提供易用性，而不需要应用程序进行任何大的更改。Heroku 通过读取名为 `Procfile` 的文件来工作，该文件包含你的 Heroku dyno（基本上是位于服务器上的虚拟机）将要运行的命令。在我们开始之前，你需要一个 Heroku 账户。如果你只想进行实验，有一个免费账户可供使用。

在目录的根目录下，在名为 `Procfile` 的新文件中，我们有以下内容：

```py
web: uwsgi heroku-uwsgi.ini
```

这告诉 Heroku 我们有一个名为 `web` 的进程，该进程将运行 uWSGI 命令并传递 `uwsgi.ini` 文件。Heroku 还需要一个名为 `runtime.txt` 的文件，该文件将告诉 Heroku 你希望使用哪个 Python 运行时——在撰写本文时，最新的 Python 版本是 3.7.0：

```py
python-3.7.0
```

接下来，确保 **uwsgi** 已存在于 `requirements.txt` 文件中。

最后，我们需要对之前创建的 `uwsgi.ini` 文件进行一些修改：

```py
    [uwsgi]
    http-socket = :$(PORT)
    die-on-term = true
    wsgi-file = wsgi.py
    callable = app
    processes = 4
    threads = 2
```

我们将 uWSGI 监听的端口设置为环境变量 port，因为 Heroku 并不会直接将 dyno 暴露给互联网。相反，它有一个非常复杂的负载均衡器和反向代理系统，因此我们需要让 uWSGI 监听 Heroku 需要我们监听的端口。此外，我们还设置了 die-on-term 为 true，以便 uWSGI 能够正确地监听来自操作系统的信号终止事件。

要使用 Heroku 的命令行工具，我们首先需要安装它们，这可以通过 [`toolbelt.heroku.com`](https://toolbelt.heroku.com) 完成。

接下来，你需要登录到你的账户：

```py
$ heroku login
```

在部署之前，我们可以使用 `foreman` 命令测试我们的设置，以确保它将在 Heroku 上正常工作：

```py
$ foreman start web
```

`foreman` 命令模拟了 Heroku 用来运行我们的应用程序的相同生产环境。为了创建将在 Heroku 服务器上运行应用程序的 dyno，我们将使用 `create` 命令。然后，我们可以将 Heroku 推送到 Git 仓库上的远程分支，以便 Heroku 服务器自动拉取我们的更改：

```py
$ heroku create
$ git push heroku master
```

如果一切顺利，你现在应该有一个在新的 Heroku dyno 上运行的应用程序。你可以使用以下命令打开新标签页，访问你的新网络应用程序：

```py
$ heroku open
```

要查看在 Heroku 部署中的应用程序运行情况，请访问 [`mastering-flask.herokuapp.com/`](https://mastering-flask.herokuapp.com/).

# 使用 Heroku Postgres

正确维护数据库是一项全职工作。幸运的是，我们可以使用 Heroku 的内置功能来自动化这一过程。Heroku Postgres 提供了一个由 Heroku 维护和完全托管的数据库。因为我们使用 SQLAlchemy，所以使用 Heroku Postgres 非常简单。在您的 dyno 仪表板中，有一个链接到您的 Heroku Postgres 信息。通过点击它，您将进入一个类似于以下截图的页面：

![图片](img/d7d77ee8-61c8-46ff-9354-bea69d15beca.png)

通过点击 URL 字段，您将获得一个 SQLAlchemy URL，您可以直接将其复制到您的生产配置对象中。

# 在 Heroku 上使用 Celery

我们已经设置了生产 Web 服务器和数据库，但我们仍然需要设置 Celery。使用 Heroku 的众多插件之一，我们可以在云中托管一个 RabbitMQ 实例，同时在 dyno 上运行 Celery 工作进程。第一步是告诉 Heroku 在`Procfile`中运行您的 Celery 工作进程：

```py
web: uwsgi heroku-uwsgi.ini celery: celery worker -A celery_runner
```

接下来，为了使用免费计划（`lemur`计划）安装 Heroku RabbitMQ 插件，请使用以下命令：

```py
$  heroku addons:create cloudamqp:lemur
```

要获取 Heroku 附加组件的完整列表，请访问[`elements.heroku.com/addons`](https://elements.heroku.com/addons)。

在仪表板上列出 Heroku Postgres 的位置，您现在将找到 CloudAMQP：

![图片](img/a0579d3f-1e34-4b2f-b01d-5b8ebfb9ac06.png)

点击 CloudAMQP 也会给您一个带有 URL 的屏幕，您可以将它复制并粘贴到您的生产配置中：

![图片](img/ed113cc9-8bb1-4a24-b6be-0dce512a0284.png)

# 在 Amazon Web Services 上部署

**亚马逊网络服务**（**AWS**）是由亚马逊维护的一组服务，建立在运行 Amazon.com 的相同基础设施之上。在本节中，我们将使用 Amazon Elastic Beanstalk 来部署我们的 Flask 代码，而数据库将托管在亚马逊的**关系数据库服务**（**RDS**）上，我们的 Celery 消息队列将托管在亚马逊的**简单队列服务**（**SQS**）上。

# 在 Amazon Elastic Beanstalk 上使用 Flask

Elastic Beanstalk 是一个为 Web 应用程序提供的平台，它为开发者提供了许多强大的功能，因此他们不必担心维护服务器。例如，随着同时使用您应用程序的人数增加，您的 Elastic Beanstalk 应用程序将自动通过利用更多服务器进行扩展。对于 Python 应用程序，Elastic Beanstalk 使用 Apache，结合`mod_wsgi`连接到 WSGI 应用程序——如果您的部署简单，负载中等或较低，则不需要额外配置。

在我们开始之前，您需要一个 Amazon.com 账户来登录控制台。接下来，您需要安装**awscli**并使用您的凭证配置它——您必须生成 AWS 访问密钥和密钥：前往 AWS 控制台，选择 IAM 服务，选择您的用户，然后选择“安全凭证”选项卡，点击“创建访问密钥”。接下来，我们需要安装 awsebcli 来从命令行管理 Elastic Beanstalk：

```py
$ pip install awsebcli --upgrade --user
```

接下来，从我们项目的根目录开始，我们将配置 CLI 并创建一个新的 Elastic Beanstalk 应用程序：

```py
$ eb init Enter Application Name
(default is "Chapter-13"): myblog Application myblog has been created.

It appears you are using Python. Is this correct?
(Y/n): Y Select a platform version.
1) Python 3.6
2) Python 3.4
3) Python 3.4 (Preconfigured - Docker)
4) Python 2.7
5) Python
(default is 1): 1 Cannot setup CodeCommit because there is no Source Control setup, continuing with initialization
Do you want to set up SSH for your instances?
(Y/n): Y Select a keypair.
1) aws-sshkey
2) [ Create new KeyPair ]
(default is 1): 1 
```

Elastic Beanstalk 会查找项目目录中的 `application.py` 文件，并期望在该文件中找到一个名为 `application` 的 WSGI 应用程序：

```py
import os
from webapp import create_app
from webapp.cli import register

env = os.environ.get('WEBAPP_ENV', 'dev')
application = create_app('config.%sConfig' % env.capitalize())
register(application)
```

接下来，我们将创建一个开发环境。每个 Elastic Beanstalk 应用程序可以包含一个或多个环境。但就目前情况来看，我们的应用程序将会失败——我们需要告诉 Elastic Beanstalk 如何在 Python 的虚拟环境中安装 Flask-YouTube 并初始化数据库。为此，我们需要扩展默认设置。

在根目录下，我们需要一个名为 `.ebextensions` 的目录。这是我们在其中创建大量额外配置和设置脚本的地方。在 `.ebextensions` 中，我们创建两个将在部署后阶段运行的 shell 脚本。因此，在 `.ebextensions/10_post_deploy.config` 文件中，添加以下代码：

```py
files:
   "/opt/elasticbeanstalk/hooks/appdeploy/post/01_install_flask_youtube.sh":
        mode: "000755"
        owner: root
        group: root
        content: |
            #!/usr/bin/env bash

            cd /opt/python/current/app
            . /opt/python/current/env
            source /opt/python/run/venv/bin/activate
            sh install_flask_youtube.sh

    "/opt/elasticbeanstalk/hooks/appdeploy/post/02_migrate_database.sh":
        mode: "000755"
        owner: root
        group: root
        content: |
            #!/usr/bin/env bash
...
```

使用 YAML 语法，我们告诉 Elastic Beanstalk 创建两个 shell 脚本来安装 Flask-YouTube 并创建或迁移数据库。这些文件的位置是特殊的——`/opt/elasticbeanstalk/hooks/appdeploy/post` 是我们可以放置在部署后执行的脚本的地方。这些脚本按字母顺序执行。同时，请注意以下位置：

+   `/opt/python/current/app`: 这是应用程序的部署位置。

+   `/opt/python/current/env`: 这是一个包含 Elastic Beanstalk 上定义的环境变量的文件。

+   `/opt/python/run/venv`: 这是指定的 Python 的 `virtualenv`，也是 Elastic Beanstalk 安装所有定义的依赖项的地方。

现在，为了创建我们的环境，运行以下命令：

```py
$ eb create myblog-dev
$ # Setup this environment variable
$ eb setenv WEBAPP_ENV=Dev
```

最后，在环境完成基础设施和部署配置后，我们可以使用以下命令检查我们的应用程序：

```py
$ eb open
```

要部署我们应用程序的新版本，我们只需运行此命令：

```py
$ eb deploy
```

注意，我们的开发环境使用 SQLite，因此数据库位于 Web 服务器上的一个文件中。在每次部署或实例重建时，此数据库都会被重新创建。

# 使用 Amazon RDS

**Amazon RDS** 是一个云数据库托管平台，它自动管理多个方面，例如节点故障时的恢复、计划备份和主/从设置。

要使用 RDS，请转到 AWS 控制台的“服务”选项卡，并点击“关系数据库服务”。

现在，创建并配置一个新的数据库——确保在**公开访问**选项中选择否。选择与实例相同的 VPC，并仔细登记您的管理员凭证。现在，等待几分钟以创建实例。之后，选择您的实例，转到详细配置，并找到**端点**字段——它应该看起来像 `myblog.c7pdwgffmbqdm.eu-central-1.rds.amazonaws.com`。我们的生产配置使用系统环境变量来设置数据库 URI，因此我们必须配置 Elastic Beanstalk 以设置 `DB_URI` 环境变量。

要使用这些环境变量，我们需要将博客的 `config.py` 文件更改为使用实际的 OS 环境变量，如下所示：

```py
class ProdConfig(Config):
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DB_URI', '')

    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', '')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_BROKER_URL', '')

    CACHE_TYPE = 'redis'
    CACHE_REDIS_HOST = os.environ.get('REDIS_HOST', '')
    CACHE_REDIS_PORT = '6379'
    CACHE_REDIS_PASSWORD = ''
    CACHE_REDIS_DB = '0'
```

确保您的实例可以连接到数据库。如果您选择了安全组默认选项和 RDS 创建，那么向导将为您创建一个安全组（默认名称为 '`rds-launch-wizard`'）。在 EC2 上，编辑此安全组并打开 3306 端口到您的实例的 VPC CIDR。

在 `.ebextensions` 中查看 `01_env.config`——这是我们设置环境变量的地方：

```py
option_settings:
  aws:elasticbeanstalk:application:environment:
    WEBAPP_ENV: Prod
    DB_URI: mysql://admin:password@myblog.c4pdwhkmbyqm.eu-central-1.rds.amazonaws.com:3306/myblog
    CELERY_BROKER_URL: sqs://sqs.us-east-1.amazonaws.com/arn:aws:sqs:eu-central-1:633393569065:myblog-sqs/myblog-sqs
```

最后，让我们使用以下命令创建生产环境：

```py
$ eb create myblog-prod 
```

# 使用 Celery 和 Amazon SQS

为了在 AWS 上使用 Celery，我们需要让我们的 Elastic Beanstalk 实例在后台运行 Celery 工作进程，并设置 SQS 消息队列。为了使 Celery 支持 SQS，它需要从 `pip` 安装一个辅助库。再次确认我们的 `requirements.txt` 文件包含 **boto3** 包。Elastic Beanstalk 将查看此文件并从中创建一个虚拟环境。

在 SQS 上设置新的消息队列非常简单。转到“服务”选项卡，然后在“应用程序”选项卡中点击“简单队列服务”，然后点击**创建新队列**。在非常简短的配置屏幕后，你应该会看到一个类似于以下屏幕的界面：

![图片](img/f22c7501-612e-4e62-b7cc-0f12df21d56a.png)

接下来，我们必须让我们的实例访问新创建的 SQS。最简单的方法是编辑 Elastic Beanstalk 默认实例配置文件（不过这并不推荐——您应该创建一个单独的实例配置文件，并使用 `.ebextensions` 选项设置将所有实例与其关联）。默认 IAM 实例配置文件名为 [aws-elasticbeanstalk-ec2-role](https://console.aws.amazon.com/iam/home#/roles/aws-elasticbeanstalk-ec2-role)。转到 IAM 服务，然后选择角色，然后选择 [aws-elasticbeanstalk-ec2-role](https://console.aws.amazon.com/iam/home#/roles/aws-elasticbeanstalk-ec2-role) 角色。接下来，点击**添加内联策略**并按照向导为新建的 SQS 提供访问权限。

现在我们必须将我们的 `CELERY_BROKER_URL` 改为新 URL，其格式如下：

```py
$ eb setenv CELERY_BROKER_URL=sqs://sqs.us-east-1.amazonaws.com/arn:aws:sqs:us-east-1:<AWS_ACCOUNT_ID>:myblog-sqs/myblog-sqs
```

将 `AWS_ACCOUNT_ID` 的值更改为您的 AWS 账户 ID。

最后，我们需要告诉 Elastic Beanstalk 在后台运行 Celery 工作进程。再一次，我们可以在`.ebextensions`中这样做。创建一个名为`11_celery_start.config`的文件，并将以下代码插入其中：

```py
commands:
    celery_start:
        command: |
              #!/usr/bin/env bash
              cd /opt/python/current/app
              . /opt/python/current/env
              source /opt/python/run/venv/bin/activate
              celery multi start worker1 -A celery_runner
```

注意，这种 Celery 工作部署存在于 Web 服务器上（这并不推荐），并且会随着 Web 服务器按需扩展。更好的选择是探索 Elastic Beanstalk 的工作功能，但这将意味着对功能进行彻底的重构，我们可能会遭受后续的供应商锁定。

# 使用 Docker

Docker 是 Docker, Inc. 在 2013 年创建的一种基于容器的技术。容器技术并不新鲜，Unix 操作系统上已经存在一段时间了，1982 年就有 chroot，2004 年有 Solaris Zones，AIX 或 OS400 系统上也有 WPAR（尽管 WPAR 更像是一种虚拟化技术而不是容器）。后来，Linux 集成了两个重要的特性：**namespaces**，它隔离了 OS 功能名称，以及 **cgroups**，这是一个受配置和资源限制约束的进程集合。这些新特性催生了 Linux 容器，那么为什么还要使用 Docker 呢？

主要是因为 Docker 使配置定义变得简单。使用非常容易编写的 Dockerfile，您可以描述如何配置您的容器并使用它创建一个新的镜像。每个 Dockerfile 行都会使用 UnionFS 创建一个新的文件系统层，这使得更改非常快速地应用，并且同样容易回滚和前进到更改。此外，Docker, Inc. 创建了一个开放的镜像仓库，您可以在其中找到几乎所有 Linux 软件的质量镜像。我们已经在第九章，*使用 Celery 创建异步任务*中使用了一些这些镜像。

Docker 已经获得了巨大的关注和炒作。其中一些最好的特性如下：

+   解决来自操作系统的依赖问题：由于我们正在将一个薄的操作系统打包到您的容器镜像中，因此可以安全地假设在您的笔记本电脑上运行的内容在生产环境中也能运行。

+   容器非常轻量，用户可以在同一虚拟机或硬件主机上运行多个容器，这可以降低运营成本并提高效率。

+   容器启动非常快，使您的基础设施能够快速扩展，例如，如果您需要处理工作负载的增加。

+   开发者可以轻松地使用容器与其他开发者共享他们的应用程序。

+   Docker 支持 DevOps 原则：开发者和运维人员可以在镜像和架构定义上共同工作，使用 Dockerfile 或 Docker Compose。

如果我们考虑 Docker 容器与虚拟机提供的功能差异，让我们记住容器共享相同的内核并且通常运行单个进程，而虚拟机运行一个完整的客户操作系统：

![](img/5ef2a593-2af8-4fb8-9282-79e7c50ac68a.png)

这种架构使容器非常轻量且快速启动。

# 创建 Docker 镜像

在前面的章节中，我们的 Blog 应用程序已经从简单的三层架构发展到多层架构。我们现在需要处理 web 服务器、数据库、缓存系统和队列。我们将定义每个这些层为 Docker 容器。

首先，让我们从我们的 web 服务器和 Flask 应用程序开始。为此，我们将使用 Nginx 前端，以及后端使用的 WSGI，称为 uWSGI。

Dockerfile 是一个包含特殊指令的文本文件，我们使用它来指定我们的 Docker 镜像以及如何运行它。构建过程将逐个执行命令，在每个命令上创建一个新的层。一些最常用的 Dockerfile 命令包括以下内容：

+   `FROM`**: **指定我们的新镜像基于的基础镜像。我们可以从一个非常薄的操作系统开始，例如 Alpine，或者直接从 RabbitMQ 镜像开始。

+   `EXPOSE`: 通知 Docker 容器监听指定的网络端口/协议。

+   `ENV`: 设置环境变量。

+   `WORKDIR`: 为 Dockerfile 建立基本目录。

+   `RUN`: 在新层上运行 bash Linux 命令。这通常用于安装额外的包。

+   `COPY`: 从本地文件系统复制文件或目录到 Docker 镜像。

+   `CMD`: 只能有一个 CMD 实例。它指定了如何运行容器。

+   `ENTRYPOINT`: 这与 CMD 有相同的目标，但在 Docker 中是一个脚本。

要查看 Dockerfile 命令的完整参考，请参阅[`docs.docker.com/engine/reference/builder/#usage`](https://docs.docker.com/engine/reference/builder/#usage)文档。

我们为 Docker 部署的目录结构将是以下内容：

```py
/
  deploy/
  docker/
    docker-compose.yml -> Compose file
    ecs-docker-compose.yml -> Specific compose file for AWS ECS
    Dockerfile_frontend -> Dockerfile for the frontends
    Dockerfile_worker -> Dockerfile for the workers
    prod.env -> Production environment variables
    worker_entrypoing.sh -> entrypoint for the celery worker
  supervisor_worker.sh -> Supervisor conf file for the celery worker
  uwsgi.ini -> Conf. file for uWSGI
```

我们将要创建的镜像将用于 Docker Compose（本章后面将详细介绍），因此它们不能独立工作。如果您不想使用 Docker Compose，对镜像进行少量修改即可使其工作——您只需更改`prod.env`文件即可。

首先，让我们为我们的 web 服务器创建一个 Dockerfile。我们将使用一个已经包含 NGINX 和 uWSGI 的先前镜像，这样我们就可以节省安装和配置它们的工作。我们的`Dockerfile_frontend`是包含创建前端镜像定义的 Dockerfile：

```py
FROM tiangolo/uwsgi-nginx:python3.6

# Create and set directory where the code will live
RUN mkdir /srv/app
WORKDIR /srv/app

# Copy our code
COPY . .
# Install all python packages required
RUN pip install -r requirements.txt
RUN sh install_flask_youtube.sh

# Setup NGINX and uWSGI
COPY ./deploy/uwsgi.ini /etc/uwsgi/uwsgi.ini
ENV NGINX_WORKER_OPEN_FILES 2048
ENV NGINX_WORKER_CONNECTIONS 2048
ENV LISTEN_PORT 80

EXPOSE 80
```

首先，在前面的片段中，我们将我们的镜像基于`uwsgi-nginx:python3.6`，这意味着我们将使用 Python 3.6。接下来，我们创建并设置应用程序将驻留的目录——这将是`/srv/app`。然后，我们使用`COPY . .`将所有本地内容（myblog 代码）复制到镜像本身。接下来，我们复制 WSGI 的配置文件，最后配置 NGINX 将使用的工人数。最后，我们通知 Docker 该镜像将在端口 80 上监听，使用`EXPOSE 80`。

接下来，让我们看一下我们的 Celery 工作器 Dockerfile：

```py
FROM ubuntu
RUN  apt-get update && \
     apt-get install -y supervisor python3-pip python3-dev libmysqlclient-dev mysql-client
RUN mkdir /srv/app
WORKDIR /srv/app
COPY . .
RUN pip3 install -r requirements.txt
RUN sh install_flask_youtube.sh

COPY ./deploy/supervisor_worker.conf /etc/supervisor/conf.d/celery_worker.conf
COPY ./deploy/docker/worker_entrypoint.sh .
ENTRYPOINT ["sh", "./worker_entrypoint.sh"]
```

这次，我们的基础镜像将是 Ubuntu（特别是，一个非常薄的 Ubuntu 版本用于 Docker）。我们将使用 **supervisor** Python 包来监控和启动我们的 Celery 进程，所以如果 Celery 由于某种原因崩溃，supervisor 将重新启动它。因此，在操作系统级别，我们正在安装 supervisor、Python 3 和 MySQL 客户端包。看看前面代码块中的 `worker_entrypoint.sh` shell 脚本，我们在那里做一些有趣的事情：

+   我们正在等待 MySQL 变得可用。当使用 Docker Compose 时，我们可以定义每个任务（即每个 Docker 容器）启动的顺序，但我们没有方法知道服务是否已经可用。

+   接下来，我们使用 Flask CLI 和 Alembic 创建或迁移我们的数据库。

+   最后，我们将测试数据插入到我们的数据库中（简单来说，对读者来说是个好东西），这样当你启动应用程序时，它已经处于可工作状态，并且已经存在一些假帖子数据。

要构建和创建我们的镜像，请在项目根目录的 shell 中执行以下 Docker 命令：

```py
$ docker build -f deploy/docker/Dockerfile_frontend -t myblog:latest . 
```

这将创建一个名为 **myblog** 的镜像，带有 **latest** 标签。作为生产最佳实践的组成部分，你应该使用项目版本给你的镜像打标签，也使用 **git** 标签。这样，我们总能确保代码在哪个镜像中；例如，`myblog:1.0` 和 `myblog:1.1` 之间有什么变化。

最后，使用以下命令创建 Celery 工作进程镜像：

```py
$ docker build -f deploy/docker/Dockerfile_worker -t myblog_worker:latest . 
```

现在我们已经创建了自定义镜像，我们可以进入下一节，在那里我们将定义所有基础设施并将容器相互连接。

# Docker Compose

**Docker Compose** 是定义我们多层应用程序的工具。这是我们定义运行应用程序所需的所有服务、配置它们并将它们连接在一起的地方。

Docker Compose 基于 YAML 文件，所有定义都发生在这里，所以让我们直接进入它，看看 `deploy/docker/docker-compose.yaml` 文件：

```py
version: '3'
services:
  db:
    image: mysql:5.7
    env_file:
      - prod.env
  rmq:
    image: rabbitmq:3-management
    env_file:
      - prod.env
    ports:
      - 15672:15672
  redis:
      image: redis
  worker:
    image: myblog_worker:latest
    depends_on:
      - db
      - rmq
    env_file:
      - prod.env
  frontend:
    image: myblog
    depends_on:
      - db
      - rmq
    env_file:
      - prod.env
    restart: always
    ports:
      - 80:80
```

在 Docker Compose 中，我们定义了以下服务：

+   **mysql**：这是基于 Docker Hub 社区版的 MySQL 5.7 镜像。所有自定义配置都使用环境变量完成，如 `prod.env` 文件中定义的那样。

+   **rmq**：Rabbit MQ 基于我们定制的 Docker Hub 社区镜像，创建用户凭据、cookies 和 VHOST。这将安装管理界面，可以通过 `http://localhost:15672` 访问。

+   **redis**：这是我们缓存用的 Redis 服务。

+   **工作进程**：这使用我们之前构建的 `myblog_worker` Docker 镜像。

+   **前端**：这使用我们之前构建的 `myblog_worker` Docker 镜像。

这是一个非常简单的 composer 定义。注意 `depends_on`，其中我们定义了哪些服务依赖于其他服务。例如，我们的前端服务将依赖于数据库和 Rabbit MQ。`ports` 键是一个公开端口的列表；在这种情况下，前端端口 80 将由 Docker 主机上的端口 80 公开。这样，我们就可以通过 Docker 主机的 IP 端口 80 或通过在 Docker 主机前面使用负载均衡器来访问我们的应用程序。在已经安装 Docker 的机器上，您可以通过 `http://localhost` 访问应用程序。

使用 `prod.env` 文件非常重要，因为这样我们可以为不同的环境定义不同的配置，同时仍然使用相同的 compose 文件。在环境中使用相同的 compose 文件遵守了十二要素应用规则中关于使基础设施组件在所有环境中相同的规定。

让我们看看 `prod.env` 文件：

```py
WEBAPP_ENV=Prod
DB_HOST=db
DB_URI=mysql://myblog:password@db:3306/myblog
CELERY_BROKER_URL=amqp://rabbitmq:rabbitmq@rmq//
REDIS_HOST=redis
MYSQL_ROOT_PASSWORD=rootpassword
MYSQL_DATABASE=myblog
MYSQL_USER=myblog
MYSQL_PASSWORD=password
RABBITMQ_ERLANG_COOKIE=SWQOKODSQALRPCLNMEQG
RABBITMQ_DEFAULT_USER=rabbitmq
RABBITMQ_DEFAULT_PASS=rabbitmq
RABBITMQ_DEFAULT_VHOST=/
```

此文件环境变量将设置实际的操作系统级别环境变量，以便在应用程序的配置文件中简单使用。这将符合来自 `https://12factor.net/` 的十二要素应用规则之一。

在顶部，我们使用 `WEBAPP_ENV=Prod` 设置我们的应用程序生产配置环境。

`MYSQL_*` 变量是我们配置 MySQL 5.7 容器的地方。我们设置了 root 密码，并设置了一个初始数据库来创建（如果需要）为该数据库创建用户和密码。

需要注意的是，`REDIS_HOST`、`DB_URI`、`CELERY_BROKER_URL` 这些变量正在使用每个容器实际使用的实际主机名来与其他容器进行通信。默认情况下，这些是服务名称，这使得一切变得相当简单。因此，前端容器使用 `db` 网络主机名来访问数据库。

最后，让我们开始我们的应用程序：

```py
$ docker-compose -f deploy/docker/docker-compose.yml up
```

等待所有容器启动，然后打开您的浏览器并转到 `http://localhost`。

# 在 AWS 上部署 Docker 容器

要在 AWS 上部署，我们将使用 **Amazon Elastic Container Service**（**ECS**）。ECS 是一个为 Docker 提供可扩展集群的服务，无需安装任何软件来编排您的容器。它基于 **AWS Auto Scaling Groups**（**ASG**），该组使用 Docker 安装实例进行扩展或缩减。这种扩展是由监控指标触发的，例如 CPU 使用率或网络负载。ECS 还会将所有容器从某个原因终止或服务受损的实例迁移出来。因此，ECS 充当集群的角色。之后，ASG 将启动一个新实例来替换有故障的实例。

# CloudFormation 基础

AWS 提供了许多服务，每个服务都有许多配置选项。您还需要将这些服务连接起来。为了有效地和可靠地创建、配置、更新或销毁这些服务，我们将向您展示如何使用 AWS 的**IaC**（**基础设施即代码**）技术，称为 CloudFormation。**CloudFormation**并非复杂的技术，而是遵循所有 AWS 服务和配置选项的扩展。CloudFormation 的详细信息和操作可能需要一本书来专门介绍。

CloudFormation 是一个扩展的数据结构，您可以使用 JSON 或 YAML 编写。我说扩展，因为可以使用引用、函数和条件。一个 CloudFormation 文件由以下部分组成：

```py
AWSTemplateFormatVersion: "version date"
Description: "Some description about the stack"
Parameters: Input parameters to configure the stack
Metadata: Aditional data about the template, also useful to group parameters on the UI
Mappings: Data mappings definitions
Conditions: Setup conditions to setup resources or configuration
Transform: Mainly used for AWS serverless
Resources: Resource definitions, this is the only required section
Output: Section to output data, you can use it return the DNS name to access the created application

```

让我们快速查看提供的 CloudFormation 文件`./deploy/docker/cfn_myblog.yml`。我们将逐个遵循所有 CloudFormation 部分。首先，让我们检查**参数**部分：

```py
...
Parameters:
  ApplicationName:
    Description: The application name
    Type: String
    Default: ecs001
  Environment:
    Description: Application environment that will use the Stack
    Type: String
    Default: prod
    AllowedValues:
    - dev
    - stg
    - prod
  InstanceType:
    Description: Which instance type should we use to build the ECS cluster?
    Type: String
    Default: t2.medium
...
```

不深入细节，在这个文件中，一个输入参数由一个名称定义，可能包含描述、类型、默认值和接受值的规则。所有这些值将在配置我们的基础设施时被引用。这些值将在部署或更新 CloudFormation 堆栈时填写。

接下来，看看**映射**部分：

```py
...
Mappings:
  AWSRegionToAMI:
    us-east-2:
        AMI: ami-b86a5ddd
    us-east-1:
        AMI: ami-a7a242da
    us-west-2:
        AMI: ami-92e06fea
...
```

这只是一个方便的数据结构，用于将 AWS 区域映射到 AMI。AMI 是我们用于我们的 Docker VMs 的基础操作系统镜像。每个区域中的 AMI 都有一个不同的标识符，因此我们需要将它们映射出来，以便我们的堆栈可以在任何 AWS 区域上部署。在我们的案例中，我们将使用 Amazon ECS 优化的 Linux。

现在，让我们考虑**元数据**部分：

```py
...
Metadata:
  AWS::CloudFormation::Interface:
    ParameterGroups:
    - Label:
        default: System Information (Tags)
      Parameters:
      - Environment
      - ApplicationName
    - Label:
        default: Networking
      Parameters:
      - VPC
      - Subnets
...
```

在这里，我们声明了一个`Interface`来分组我们的参数。这只是为了让参数以更美观的方式显示给将要部署堆栈的人。记住，参数部分是一个字典，且该字典的键是无序的。

主要且更为重要的部分是**资源**。我们不会对此进行详细说明，而是快速概述我们将要创建的主要基础设施资源及其连接方式。首先，对于数据库，我们将使用另一个 AWS 服务，称为**RDS**，并创建一个 MySQL 服务器：

```py
Resources:
...
DB:
 Type: AWS::RDS::DBInstance
  Properties:
    AllocatedStorage: "30"
    DBInstanceClass: "db.t2.medium"
    Engine: "MariaDB"
    EngineVersion: "10.2.11"
    MasterUsername: !Ref DBUsername
    MasterUserPassword: !Ref DBPassword
    DBSubnetGroupName: !Ref DBSubnetGroup
    VPCSecurityGroups:
      - Ref: DBSecurityGroup
```

每个资源都有一个类型。对于 RDS，这是`AWS::RDS:DBInstance`。每种类型都有自己的特定属性集。注意`!Ref`如何声明来自其他资源或参数的值。"DBUsername"和"DBPassword"是参数，但"DBSubnetGroup"和"DBSecurityGroup"是由 CloudFormation 创建的资源，用于设置数据库的网络 ACL 和子网放置。

ECS 集群资源声明如下：

```py
ECSCluster:
  Type: "AWS::ECS::Cluster"
  Properties:
    ClusterName: !Sub ${Environment}-${ApplicationName}

ECSAutoScalingGroup:
  Type: AWS::AutoScaling::AutoScalingGroup
  Properties:
...

ECSLaunchConfiguration:
  Type: AWS::AutoScaling::LaunchConfiguration
  Properties:
...

ECSRole:
  Type: AWS::IAM::Role
  Properties:
...
ECSInstanceProfile:
  Type: AWS::IAM::InstanceProfile
  Properties:
...
ECSServiceRole:
  Type: AWS::IAM::Role
  Properties:
...
```

所有这些定义都属于 ECS 集群。这个集群可以用于部署许多不同的应用程序，因此在这些定义上声明单独的 CloudFormation 文件或使用嵌套堆栈是有意义的。为了简化部署，我们将使用单个文件来创建我们的应用程序。首先，我们创建 ECS 集群，并设置其名称为 `Environment` 和 `ApplicationName` 参数的连接。这是通过 `!Sub` CloudFormation 函数完成的。

接下来，我们声明我们集群的 **自动扩展组**（**ASG**），并设置 AWS 将如何为属于此 ASG 的每个实例进行配置。这些是 `ECSAutoScalingGroup` 和 `ECSLaunchConfiguration` 资源。最后，使用 `ECSRole`、`ECSInstanceProfile` 和 `ECSServiceRole` 来设置 ECS 集群获取 Docker 镜像、与 AWS 负载均衡器（ELB）、S3 等进行工作所需的安全权限。这些权限是 AWS 作为示例使用的标准，当然也可以降低权限级别。

现在，针对我们的应用，我们将定义 ECS 服务和 ECS 任务定义。任务定义是我们定义一个或多个容器定义的地方，这些定义引用要使用的 Docker 镜像，并包括环境变量。然后，ECS 服务引用一个 ECS 任务定义，并且可以将其与负载均衡器关联，并设置部署配置选项，例如性能限制和自动扩展选项（是的，ECS 集群可以根据负载变化进行扩展或缩减，但我们的容器也可以独立地进行扩展或缩减）：

```py
FrontEndTask:
  DependsOn: WorkerTask
  Type: "AWS::ECS::TaskDefinition"
  Properties:
    ContainerDefinitions:
      -
        Name: "frontend"
        Image: !Ref DockerFrontEndImageArn
        Cpu: "10"
        Memory: "500"
        PortMappings:
          -
            ContainerPort: "80"
            HostPort: "80"
        Environment:
          -
            Name: "WEBAPP_ENV"
            Value: !Ref Environment
          -
            Name: "CELERY_BROKER_URL"
            Value: !Sub "amqp://${RMQUsername}:${RMQPassword}@${ELBRMQ.DNSName}:5672//"
          -
            Name: "DB_URI"
            Value: !Sub "mysql://${DBUsername}:${DBPassword}@${DB.Endpoint.Address}:3306/myblog"
          -
            Name: "REDIS_HOST"
            Value: !Sub ${ELBRedis.DNSName}
```

这是我们的前端容器的任务定义。你可能注意到，这是我们已经看到的 Docker Compose 服务的 CloudFormation 版本。我们为我们的容器声明一个名称，`Name: "frontend"`，稍后将在负载均衡器中引用。接下来，image：`!Ref DockerFrontEndImageArn` 是对输入参数的引用。这将使我们能够轻松部署我们博客应用程序的新版本。Docker 的端口映射在 `PortMappings` 中声明，这是一个键值对的列表，重复了 `ContainerPort` 和 `HostPort` 的键。环境也是一个键值对的列表，在这里我们从我们创建的其他资源中为 DB、RMQ 和 Redis 进行“连接”。例如，这里是如何使用 `DB_URI` 的：

```py
-
            Name: "DB_URI"
            Value: !Sub "mysql://${DBUsername}:${DBPassword}@${DB.Endpoint.Address}:3306/myblog"
```

这个 `Value` 是我们构建数据库 URI 的地方，使用我们已知的 `!Sub` 函数和 `DBUsername` 以及 `DBPassword` 的引用。`DB.Endpoint.Address` 是我们如何引用 AWS 为我们新创建的 MySQL 服务器创建的 DNS 名称。

在服务定义中，我们将我们的容器与 AWS 弹性负载均衡器关联，并做一些部署配置：

```py
MyBlogFrontendService:
  Type: "AWS::ECS::Service"
  Properties:
    Cluster: !Ref ECSCluster
    DeploymentConfiguration:
      MaximumPercent: 200
      MinimumHealthyPercent: 50
    DesiredCount: 2
    TaskDefinition: !Ref FrontEndTask
    LoadBalancers:
      -
        ContainerName: 'frontend'
        ContainerPort: 80
        LoadBalancerName: !Ref ELBFrontEnd
  DependsOn:
    - ECSServiceRole
    - FrontEndTask
```

首先，我们声明这个服务将在我们新创建的 ECS 集群上运行，使用`Cluster: !Ref ECSCluster`。然后，使用`DeploymentConfiguration`和`DesiredCount`，我们说明这个服务将以两个容器（为了高可用性）启动，并允许它在 4 到 1 之间进行扩展和缩减。这遵循以下公式：

+   最大容器数量 = DesiredCount * (MaximumPercent / 100)

+   最小容器数量 = DesiredCount * (MinimumPercent / 100)

因此，将公式应用于我们的案例，我们得到以下结果：

+   4 = 2 * (200/100)

+   1 = 2 * (50/100)

通过`TaskDefinition: !RefFrontEndTask`，我们说明这个服务使用我们之前的前端任务定义。最后，通过`LoadBalancers`键属性，我们将我们的服务与负载均衡器绑定。这意味着我们两个新创建的容器将均匀地接收来自用户的请求，并且随着容器的创建，新的容器将自动在负载均衡器上注册。

最后，让我们看看负载均衡器定义：

```py
ELBFrontEnd:
  Type: AWS::ElasticLoadBalancing::LoadBalancer
  Properties:
    SecurityGroups:
    - Fn::GetAtt:
      - ELBFrontEndSecurityGroup
      - GroupId
    Subnets:
      Ref: Subnets
    Scheme: internet-facing
    CrossZone: true
    Listeners:
    - LoadBalancerPort: '80'
      InstancePort: '80'
      Protocol: HTTP
      InstanceProtocol: HTTP
    HealthCheck:
      Target: TCP:80
      HealthyThreshold: '2'
      UnhealthyThreshold: '3'
      Interval: '10'
      Timeout: '5'
```

这是一个 AWS 经典 ELB 定义，其中我们将 ELB 与网络安全组关联，这大致相当于防火墙。这是通过`SecurityGroups`键属性完成的。接下来，我们定义 ELB 将在哪些子网中提供服务。每个子网都在不同的 AWS 可用区中创建，每个可用区代表 AWS 区域中的一个数据中心（每个区域包含两个或多个数据中心，或可用区）。然后，我们定义这个 ELB 将通过`Scheme: internet-facing`暴露给互联网。对于`Listeners`，我们说 ELB 的 80 端口映射到 Docker 主机的 80 端口。最后，我们定义服务的健康检查，以及这种检查将持续的时间。

在[`docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-ec2-elb.html`](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-ec2-elb.html)查看 ELB CloudFormation 定义的更多详细信息。

我们还在 CloudFormation 提供的`./deploy/docker/cfn_myblog.yml` YAML 文件中创建了以下资源：

+   为 ELBs 和 Docker 主机设置几个安全组

+   我们 myblog Celery 工作者的任务定义和相应的服务

+   我们 RabbitMQ 容器的任务定义和相应的服务

+   我们 Redis 容器的任务定义和相应的服务

+   Redis 容器的负载均衡器

+   RabbitMQ 的负载均衡器

使用 RabbitMQ 的负载均衡器是一种获取服务发现功能的经济方式——在单个实例上平衡负载似乎很奇怪，但如果我们的 RabbitMQ 所在位置的 Docker 主机因某种原因崩溃，那么 RabbitMQ 容器将创建在另一个 Docker 主机上，并且应用程序需要能够动态地找到它。

# 创建和更新一个 CloudFormation 堆栈

我们可以使用控制台或 CLI 创建和部署我们的 CloudFormation 堆栈。要使用控制台创建，请选择 AWS CloudFormation 服务，然后点击创建堆栈按钮。你会看到以下表单：

![图片](img/462ffb4f-b781-49f1-8851-98dfca713ea6.png)

选择“上传模板到 Amazon S3”选项，然后从提供的代码中选择`deploy/docker/cfn_myblog.yaml`文件，并点击下一步。现在，我们需要按照以下方式填写堆栈参数：

+   Stack Name: 提供一个名称来识别此堆栈；使用你想要的任何名称。

+   Environment: 选择此堆栈的生产、预发布和开发环境。

+   ApplicationName: 在这里，使用你想要用来识别 ECS 集群的任何名称。

+   VPC: 选择一个 AWS VPC。

+   Subnets: 从下拉菜单中选择属于 VPC 的所有子网（如果你有公共和私有子网，请只选择公共子网，记住 ELB 是面向互联网的）。

+   ClusterSize: 这是 ECS 集群的大小；在这里保留默认设置`2`。

+   InstanceType: 这是 Docker 主机使用的 AWS 实例类型。

+   KeyName: 这是 AWS 密钥对，需要是我们之前创建的。我们可以使用私钥通过 SSH 连接到 Docker 主机。

+   DockerFrontEndImageArn: 这是上传我们前端 Docker 镜像的 ECR 存储库的 ARN。

+   DockerWorkerImageArn: 这是上传我们工作 Docker 镜像的 ECR 存储库的 ARN。

+   DBUsername, DBPassword, RMQUsername, 和 RMQPassword: 这些都是数据库和 RabbitMQ 的凭证；选择你想要的任何值。

填写所有参数后，点击下一步。出现一个选项表单——再次点击下一步。出现一个带有我们的参数和可能的堆栈更改的审查页面。在这里，我们需要检查**我承认 AWS CloudFormation 可能会创建具有自定义名称的 IAM 资源**选项，并点击创建。所有资源的创建将需要几分钟时间——等待 CREATE_COMPLETED 状态。要检查我们的应用程序，只需转到输出选项卡并点击 URL。

现在，让我们看看我们如何轻松地开发和部署代码更改。首先，进行简单的代码更改。例如，在`webapp/templates/head.html`文件中，找到以下行：

```py
...
<h1><a class="text-white" href="{{ url_for('blog.home') }}">My Blog</a></h1>
...
```

现在，将前面的行更改为以下内容：

```py
...
<h1><a class="text-white" href="{{ url_for('blog.home') }}">My Blog v2</a></h1>
...
```

然后创建一个新的 Docker 镜像，并使用`v2`进行标记，如下所示：

```py
$ docker build -f deploy/docker/Dockerfile_frontend -t myblog:v2 .
```

接下来，使用以下命令将此镜像推送到 AWS ECR：

```py
$ ecs-cli push myblog:v2
```

然后，转到 AWS 控制台并选择我们之前创建的堆栈。在操作中，选择更新堆栈。在第一个表单中，选择使用当前模板。然后，在输入参数中，我们需要更改`DockerFrontEndImageArn`——使用新标签更新它，并在其后加上`:v2`后缀。新的 ARN 应类似于这样：`XXXXXXXX.dkr.ecr.eu-central-1.amazonaws.com/myblog:v2`**。**然后，点击下一步，在选项表单中再次点击下一步。在预览表单中，注意在预览您的更改部分，更新器会准确地识别需要更新的内容。在这种情况下，`FrontEndTask`和`MyBlogFrontendService`被选中进行更新，所以让我们更新它们。在我们等待 UPDATE_COMPLETE 状态时，只需继续使用应用程序——注意没有停机时间发生。一到两分钟后，注意我们的博客显示的主要标题为 My Blog v2。

在下一节中，我们将看到如何将这种方法与现代 CI/CD 系统集成，以构建、运行测试、检查代码质量，并在不同的环境中部署。

# 快速构建和部署高可用性应用

无论我们的 Web 应用是在云端还是数据中心，我们都应该追求可靠性。可靠性可以通过多种方式影响用户，无论是通过停机、数据丢失、应用错误、响应时间下降，甚至是用户部署延迟。接下来，我们将介绍一些方面，帮助您思考架构和可靠性，以便您提前规划以处理问题，例如故障或负载增加。首先，我们将介绍您快速且可靠部署所需的必要步骤。

# 可靠地构建和部署

在今天竞争激烈的市场中，我们需要快速且轻松地构建和部署。但我们的部署速度也必须保证可靠性。实现这一目标的一个步骤是使用脚本或 CI/CD 工具进行自动化。

为了帮助我们设置整个流程，我们应该使用 CI/CD 工具，如 Jenkins、Bamboo、TeamCity 或 Travis。首先，CI/CD 究竟是什么？

**CI**代表**持续集成**，是指将许多开发者做出的软件更改集成到主仓库的过程——当然，这样做要快速且可靠。让我们从下到上列举我们需要的东西：

+   首先，使用源代码控制和版本控制系统至关重要，例如 Git，以及一个良好建立且内部定义的分支模型，例如**GitFlow**。这将让我们清楚地看到代码更改，并能够在功能或热修复级别接受和测试它们。这将使我们能够轻松回滚到之前的版本。

+   在批准任何由 pull request 提出的合并之前，请确保设置测试的自动触发和代码审查。然后，pull-request 审查员在批准合并之前可以做出更明智的决定。失败的测试当然是我们希望在合并最终进入生产环境的代码之前看到的警告信号。快速失败，不要害怕经常失败。

如前所述，我们有几个工具来自动化此过程。一种简单的方法是使用 GitHub、Travis 和 landscape.io。你可以在所有三个平台上自由创建账户并尝试它们。之后，只需在你的仓库中创建以下两个文件。

创建一个`.travis.yml`文件，它应该包含以下内容：

```py
language: python
python:
  - "3.6"
  - "3.3"
  - "2.7"
install:
  - "pip install --upgrade"
  - "pip -V"
  - "pip install -r requirements.txt"
  - "pip install coveralls"
script:
  - coverage run --source webapp --branch -m unittest discover
after_success:
  coveralls
```

这是我们需要的所有内容，以便在每次提交时自动运行自动化测试。此外，我们的测试将使用 Python 版本 3.6、3.3 和 2.7 独立运行。GitHub 和 Travis 集成还将给我们提供每次 pull request 的测试结果。

对于代码质量控制，landscape.io 与 GitHub（其他工具包括 flake8、Sonarqube 和 Codacy 等）非常容易使用。

要设置 landscape.io，我们只需在我们的项目根目录下创建以下`.landscape.yml`文件：

```py
ignore-paths:
  - migrations
  - deploy
  - babel
```

通过将每个分支自动合并到 develop 分支，例如，可以实现进一步的自动化，但我们需要一个第三方工具来自动化 GitHub 上的此过程。

**CD**代表**持续交付**，它基于缩短的开发周期和实际交付变更。这必须快速且可靠地完成，并且应该始终考虑回滚。为了帮助我们定义和执行此过程，我们可以使用**Jenkins/Blue Ocean 管道**。

使用 Jenkins 管道，我们可以定义从构建到部署的整个管道过程。此过程使用位于我们项目根目录的`Jenkinsfile`定义。首先，让我们从 CLI 创建并启动我们的 Jenkins CI 服务器，如下所示：

```py

docker run \
  --rm \
  -u root \
  -p 8080:8080 \
  -v jenkins-data:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HOME":/home \
  jenkinsci/blueocean
```

在启动时，Docker 的输出将显示以下内容：

```py
...
INFO: Pre-instantiating singletons in org.springframework.beans.factory.support.DefaultListableBeanFactory@340c828a: defining beans [filter,legacy]; root of factory hierarchy
Sep 16, 2018 11:39:39 AM jenkins.install.SetupWizard init
INFO:

*************************************************************
*************************************************************
*************************************************************

Jenkins initial setup is required. An admin user has been created and a password generated.
Please use the following password to proceed to installation:

476c3b81f2gf4n30a7f9325568dec9f7

This may also be found at: /var/jenkins_home/secrets/initialAdminPassword

*************************************************************
*************************************************************
*************************************************************

```

从输出中复制密码，并通过访问`http://localhost:8080`在浏览器中打开 Jenkins。在启动时，Jenkins 将要求一次性密码——粘贴由 Docker 输出提供的密码。接下来，Jenkins 将要求你进行一些初始配置。这包括创建一个管理员用户，并安装插件（在我们的示例中，你可以简单地接受建议的插件）。

要设置自动构建和部署我们的 Docker 镜像到 AWS ECR 的方法，我们需要一个额外的插件，称为 Amazon ECR。要安装此插件，请转到“管理 Jenkins”，然后选择“管理插件”，并点击“可用”选项卡以查看可用和尚未安装的插件列表。从该列表中选择 Amazon ECR 插件，并最终点击“不重启安装”选项。

接下来，我们必须配置一组凭据，以便 Jenkins 可以在 AWS 上进行身份验证并推送我们新构建的 Docker 镜像。为此，在左侧菜单中选择凭据，然后选择 Jenkins 凭据范围和全局凭据。现在，在左侧面板中选择添加凭据并填写以下信息：

+   类型：AWS 凭据

+   范围：全局

+   ID：ecr-credentials

+   描述：ecr-credentials

+   访问密钥 ID：使用你之前章节中创建的 AWS 访问密钥 ID 来推送你的 Docker 镜像

+   密钥访问密钥：使用你之前章节中创建的 AWS 密钥访问密钥来推送你的 Docker 镜像

由于安全原因，最好选择 IAM 角色方法。然而，为了简化，我们在这里使用 AWS 密钥。如果你仍然想使用 AWS 密钥，请记住永远不要在自动化过程中使用你的个人密钥——相反，为该过程创建一个具有受限和管理权限的特定用户。

现在我们已经准备好创建我们的第一个 CI/CD 管道。按照以下步骤操作：

1.  在主页上，选择创建新作业链接

1.  在“输入项目名称”的输入框中，写`myblog`

1.  选择多分支管道选项。然后点击确定

在作业配置中，你需要填写以下字段：

+   分支源：为你的 GitHub 账户创建新的 Jenkins 凭据，或者使用你从私有 Git 仓库的凭据设置。然后，选择这本书的 GitHub 仓库，或者使用你的私有仓库 URL。

+   然后，目前，请移除所有行为，除了“发现分支”，如下所示：

![图片](img/2fd29995-b9d8-4b05-af7d-6d2cdf3e6895.png)

在“构建配置”作业部分，如果你使用这本书的 GitHub 仓库，将“脚本路径”更改为`Chapter-13/Jenkinsfile`。这是必需的，因为仓库是按章节组织的，而`Jenkinsfile`不在仓库的根目录中。

这就是全部，因为繁重的工作是通过`Jenkinsfile`管道定义完成的。让我们看看这个文件：

```py
pipeline {
    agent any

    parameters {
        string(description: 'Your AWS ECR URL: http://<AWS ACCOUNT NUMBER>.dkr.ecr.<REGION>.amazonaws.com', name: 'ecrURL')
    }

    environment {
        CHAPTER = 'Chapter-13'
        ECRURL = "${params.ecrURL}"
        ECRCRED = 'ecr:eu-central-1:ecr-credentials'
    }
...
```

Jenkins 管道定义为你提供了大量的配置选项。我们甚至可以使用嵌入其中的 Groovy 脚本。请查看文档以获取更多详细信息，文档可在[`jenkins.io/doc/book/pipeline/jenkinsfile/`](https://jenkins.io/doc/book/pipeline/jenkinsfile/)找到。

在`pipeline`主部分，我们为你创建了一个手动参数，以便填写应推送图像的 AWS ECR URL。此部分还配置了一些必要的环境变量，使我们的阶段更加动态。

接下来，让我们看看管道阶段部分：

```py
....
stages {
    stage('Build') {
        steps {
            echo "Building"
            checkout scm
        }
    }
    stage('Style') {
        agent {
            docker 'python:3'
        }

        steps {
            sh '''
                #!/bin/bash

                cd "${CHAPTER}"
                python -m pip install -r requirements.txt
                cd Flask-YouTube
                python setup.py build
                python setup.py install
                cd ..
                python -m pip install flake8
                flake8 --max-line-length 120 webapp
            '''
        }
    }
...
```

`stages`部分将包含构建、测试、检查和部署我们的应用程序所需的所有阶段。使用`stage('Build')`声明的构建只是使用`checkout scm`执行我们的仓库的检出。

在*风格*阶段，我们将使用**flake8**检查代码风格。我们假设一个关键的风格问题足以使管道失败，并且永远不会部署应用程序。要运行它，我们告诉 Jenkins 使用`docker 'python:3'`命令运行一个 Python 3 的 Docker 容器，并在其中安装所有必要的依赖项并运行**flake8**对代码进行检查。

接下来，你将找到一个*测试*阶段，它与*风格*阶段非常相似。请注意，我们可以轻松地定义用于 Python 3 和 2.7 的测试，使用特定的 Docker 容器来运行它。

Docker 构建阶段如下：

```py
stage('Build docker images') {
    agent any
    steps {
        echo 'Creating new images...'
        script {
             def frontend = docker.build("myblog:${env.BUILD_ID}", "-f ${CHAPTER}/deploy/docker/Dockerfile_frontend ${CHAPTER}")
             def worker = docker.build("myblog_worker:${env.BUILD_ID}", "-f ${CHAPTER}/deploy/docker/Dockerfile_worker ${CHAPTER}")
        }
    }
}
```

在这个阶段，我们使用 Groovy 为前端和 Celery 工作进程构建镜像。这些镜像将被生成并带有 Jenkins 构建标识符，我们可以将其用作`env.BUILD_ID`环境变量。

在最终阶段，我们将新创建的镜像推送到 AWS ECR Docker 镜像存储库，如下所示：

```py
stage('Publish Docker Image') {
    agent any
    steps {
        echo 'Publishing new images...'
        script {
            docker.withRegistry(ECRURL, ECRCRED)
            {
                docker.image("myblog:${env.BUILD_ID}").push()
                docker.image("myblog_worker:${env.BUILD_ID}").push()
            }
        }
    }
}
```

最后，为了运行我们的任务，选择“myblog”任务，然后选择“master”，在左侧面板中选择“带参数构建”。填写您的 AWS ECR URL（此 URL 的格式为`http://<ACCOUNT_NUMBER>.dkr.ecr.<REGION>.amazonaws.com`），然后点击构建。构建完成后，我们只需更新我们的 CloudFormation，以包含新创建的 Docker 镜像。

一个很好的最终阶段将是更新之前部署的 CloudFormation，使用我们在本书中之前测试的脚本过程。为此，我们可以使用“pipeline: AWS steps”插件。

# 创建高度可用的可扩展应用程序

**高可用性**（**HA**）和可扩展性是一个越来越重要的主题。它应该从开发阶段一直考虑到发布阶段。单体架构，其中所有组成应用程序的功能和服务都无法分离或安装在单个实例上，将无法抵抗故障，也无法扩展。垂直扩展只能走这么远，而且在故障的情况下，将增加恢复时间，以及用户的影响。这是一个重要且复杂的问题，正如你可能猜到的，没有单一的解决方案可以解决这个问题。

要考虑 HA，我们必须持悲观态度。记住——故障无法消除，但可以识别故障点，并应制定恢复计划，以便停机时间只需几秒或几分钟，而不是几小时甚至几天。

首先，让我们考虑我们的博客应用程序的所有组件，并识别无状态的组件：

+   **前端**: Web 服务器和 uWSGI – 无状态

+   **Celery 工作进程**: Celery – 无状态

+   **消息队列**: RabbitMQ 或 AWS SQS – 有状态

+   **缓存**: Redis – 有状态

+   **数据库**: SQL 或 NoSQL – 有状态

我们的首要目标是识别我们应用程序中的所有**单点故障**（**SPOF**），并尝试消除它们。为此，我们必须考虑冗余：

+   **前端**: 这是一个无状态服务，直接接收来自用户的请求。我们可以使用负载均衡器来平衡这些请求，并且始终至少有两个实例。如果一个实例失败了，另一个立即开始接收所有负载。看起来不错？也许吧，但单个实例能支持所有负载吗？巨大的响应时间也是一种失败，所以考虑一下——你可能至少需要三个实例。接下来，你的负载均衡器也会失败吗？当使用某种基于云的负载均衡器，如 AWS ELB 或 ALB 时，这不是问题，但如果你没有使用这些，那么在这个层面上也要设置冗余。

+   **Celery 工作进程**: 工作进程是无状态的，完全失败不会立即对用户产生影响。只要恢复是自动完成的，或者失败可以轻松识别，并且一个失败的实例可以快速用新实例替换，你至少可以有一个实例。

+   **消息队列**: 如果使用 AWS SQS 或 CloudMQ，已经考虑了失败的情况。如果没有，集群化的 RabbitMQ 可以是一个选择，或者你可以确保消息丢失是一个选项，并且 RabbitMQ 的替换是自动的，或者至少可以快速执行。

+   **缓存**: 确保你有多于一个的 memcached 实例（使用集群键分片），或者你的应用程序可以优雅地处理失败。记住，memcached 的替换会带来一个冷缓存，这可能会对你的数据库产生巨大影响，这取决于你的负载。

+   **数据库**: 确保你有一个 SQL 或 NoSQL 从属/集群就绪，以替换失败的 master 的写入。

包含状态的层更成问题，即使是小的失败（几秒或毫秒）也可能是不可避免的。热备用或冷备用应该被考虑。在负载测试的同时测试你所有服务的系统故障非常有用。冗余就像一个软件特性——如果不测试，它可能已经损坏了。

可以通过负载测试来验证扩展性。在生产管道发布过程中包含它是一个非常好的主意。**Locust**是一个出色的 Python 工具，可以实施高度可配置的负载测试，可以扩展到任何你想要的负载级别。这类测试是验证你的高可用性设置的好机会。在模拟预期负载的同时关闭实例，并负载测试直到你的堆栈崩溃。这样你将知道你的极限——在它在生产中崩溃之前知道什么会先崩溃，将有助于你测试性能调整。

Locust Python 包的文档可在[`docs.locust.io/en/stable/`](https://docs.locust.io/en/stable/)找到。

使用云基础设施，如 AWS、Azure 和 GCP 进行扩展，全部关于自动化。你需要自动设置你的实例，以便监控指标可以自动触发新虚拟机或 Docker 容器的创建。

最后，请确保您定期备份您的数据库。备份之间的时间差是可能数据丢失的点，所以请识别它并报告回来。同样，恢复您的生产备份也非常重要——如果未经测试，那么它们可能已经损坏。

# 监控和收集日志

监控所有系统和组件，收集操作系统级别的指标，并生成应用程序指标。您有很好的工具来做这件事，包括 DataDog；NewRelic；StatsD、Graphana、InfluxDB 和 Prometheus 的组合；以及 ELK。

根据指标阈值设置故障警报。非常重要的一点是不要过度设置您创建的警报数量——确保一个关键警报确实意味着系统已关闭或严重受损。设置时间图表，以便您可以提前识别问题或升级需求。

从操作系统、应用程序和云服务中收集日志。解析、结构化和向您的日志添加元数据丰富了您的数据，并使适当的日志聚合、过滤和图表成为可能。能够轻松地根据特定用户、IP 或国家过滤所有日志是一个进步。

日志收集在云环境中变得更加重要，在容器中更是如此，因为它们是短暂的，并将您的应用程序分解成微服务，因此当发生某些事情时，您的日志可能已经不存在，或者您可能需要手动浏览数十个，甚至数千个日志文件来找出发生了什么。这越来越难以做到。然而，有许多好的解决方案：您可以使用 ELK（ElasticSearch、logstash 和 Kibana）或 EFK（ElasticSearch、Fluentd 和 Kibana）堆栈、Sumo logic 或 DataDog。

# 摘要

正如本章所解释的，托管您的应用程序有许多不同的选项，每个选项都有其自身的优缺点。选择哪一个取决于您愿意投入的时间和金钱，以及您预期的总用户数量。

现在，我们已经到达了本书的结尾。我希望这本书在构建您对 Flask 的理解以及如何用它来创建任何复杂度的应用程序，既简单又易于维护方面有所帮助。

网络应用程序开发是一个快速发展的领域，涉及不同的技术和概念。不要止步于此——继续提高您的 Python 技能，了解用户体验设计，提高您对 CSS 和 HTML 的知识，掌握 SQL 和查询性能，并使用 Flask 和 JavaScript 开发单页应用程序。本书的每一章都是进一步了解知识的邀请。
