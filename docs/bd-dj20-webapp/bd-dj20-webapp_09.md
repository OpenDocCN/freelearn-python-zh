# 第九章：部署 Answerly

在前一章中，我们了解了 Django 的测试 API，并为 Answerly 编写了一些测试。作为最后一步，让我们使用 Apache Web 服务器和 mod_wsgi 在 Ubuntu 18.04（Bionic Beaver）服务器上部署 Answerly。

本章假设您的服务器上有代码位于`/answerly`下，并且能够推送更新到该代码。您将在本章中对代码进行一些更改。尽管进行了更改，但您需要避免养成直接在生产环境中进行更改的习惯。例如，您可能正在使用版本控制系统（如 git）来跟踪代码的更改。然后，您可以在本地工作站上进行更改，将其推送到远程存储库（例如，托管在 GitHub 或 GitLab 上），并在服务器上拉取它们。这些代码在 GitHub 的版本控制中可用（[`github.com/tomarayn/Answerly`](https://github.com/tomarayn/Answerly)）。

在本章中，我们将做以下事情：

+   组织我们的配置代码以分离生产和开发设置

+   准备我们的 Ubuntu Linux 服务器

+   使用 Apache 和 mod_wsgi 部署我们的项目

+   看看 Django 如何让我们将项目部署为十二要素应用程序

让我们开始组织我们的配置，将开发和生产设置分开。

# 组织生产和开发的配置

到目前为止，我们一直保留了一个`requirements`文件和一个`settings.py`。这使得开发变得方便。但是，我们不能在生产中使用我们的开发设置。

当前的最佳实践是为每个环境单独创建一个文件。然后，每个环境的文件都导入具有共享值的公共文件。我们将使用这种模式来处理我们的要求和设置文件。

让我们首先拆分我们的要求文件。

# 拆分我们的要求文件

首先，让我们在项目的根目录创建`requirements.common.txt`：

```py
django<2.1
psycopg2==2.7.3.2
django-markdownify==0.2.2
django-crispy-forms==1.7.0
elasticsearch==6.0.0
```

无论我们的环境如何，这些都是我们运行 Answerly 所需的共同要求。然而，这个`requirements`文件从未直接使用过。我们的开发和生产要求文件将会引用它。

接下来，让我们在`requirements.development.txt`中列出我们的开发要求：

```py
-r requirements.common.txt
ipython==6.2.1
coverage==4.4.2
factory-boy==2.9.2
selenium==3.8.0
```

前面的文件将安装`requirements.common.txt`中的所有内容（感谢`-r`），以及我们的测试包（`coverage`，`factory-boy`和`selenium`）。我们将这些文件放在我们的开发文件中，因为我们不希望在生产环境中运行这些测试。如果我们在生产环境中运行测试，那么我们可能会将它们移动到`requirements.common.txt`中。

对于生产环境，我们的`requirements.production.txt`文件非常简单：

```py
-r requirements.common.txt
```

Answerly 不需要任何特殊的软件包。但是，为了清晰起见，我们仍将创建一个。

要在生产环境中安装软件包，我们现在执行以下命令：

```py
$ pip install -r requirements.production.txt
```

接下来，让我们按类似的方式拆分设置文件。

# 拆分我们的设置文件

同样，我们将遵循当前 Django 最佳实践，将我们的设置文件分成三个文件：`common_settings.py`，`production_settings.py`和`dev_settings.py`。

# 创建 common_settings.py

我们将通过重命名我们当前的`settings.py`文件并进行一些更改来创建`common_settings.py`。

让我们将`DEBUG = False`更改为不会*意外*处于调试模式的新设置文件。然后，让我们通过更新`SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')`来从环境变量中获取密钥。

让我们还添加一个新的设置，`STATIC_ROOT`。`STATIC_ROOT`是 Django 将从我们安装的应用程序中收集所有静态文件的目录，以便更容易地提供它们：

```py
STATIC_ROOT = os.path.join(BASE_DIR, 'static_root')
```

在数据库配置中，我们可以删除所有凭据并保留`ENGINE`的值（以明确表明我们打算在任何地方使用 Postgres）：

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
    }
}
```

接下来，让我们创建一个开发设置文件。

# 创建 dev_settings.py

我们的开发设置将在`django/config/dev_settings.py`中。让我们逐步构建它。

首先，我们将从`common_settings`中导入所有内容：

```py
from config.common_settings import *
```

然后，我们将覆盖一些设置：

```py
DEBUG = True
SECRET_KEY = 'some secret'
```

在开发中，我们总是希望以调试模式运行。此外，我们可以放心地硬编码一个密钥，因为我们知道它不会在生产中使用：

```py
DATABASES['default'].update({
    'NAME': 'mymdb',
    'USER': 'mymdb',
    'PASSWORD': 'development',
    'HOST': 'localhost',
    'PORT': '5432',
})
```

由于我们的开发数据库是本地的，我们可以在设置中硬编码值，以使设置更简单。如果您的数据库不是本地的，请避免将密码检入版本控制，并像在生产中一样使用`os.getenv()`。

我们还可以添加我们的开发专用应用程序可能需要的更多设置。例如，在第五章中，*使用 Docker 部署*，我们有缓存和 Django Debug Toolbar 应用程序的设置。Answerly 目前不使用这些，所以我们不会包含这些设置。

接下来，让我们添加生产设置。

# 创建 production_settings.py

让我们在`django/config/production_settings.py`中创建我们的生产设置。

`production_settings.py`类似于`dev_settings.py`，但通常使用`os.getenv()`从环境变量中获取值。这有助于我们将机密（例如密码、API 令牌等）排除在版本控制之外，并将设置与特定服务器分离。我们将在*Factor 3 – config*部分再次提到这一点。

```py
from config.common_settings import * 
DEBUG = False
assert SECRET_KEY is not None, (
    'Please provide DJANGO_SECRET_KEY '
    'environment variable with a value')
ALLOWED_HOSTS += [
    os.getenv('DJANGO_ALLOWED_HOSTS'),
]
```

首先，我们导入通用设置。出于谨慎起见，我们确保调试模式关闭。

设置`SECRET_KEY`对于我们的系统保持安全至关重要。我们使用`assert`来防止 Django 在没有`SECRET_KEY`的情况下启动。`common_settings.py`文件应该已经从环境变量中设置了它。

生产网站将在`localhost`之外的域上访问。我们将通过将`DJANGO_ALLOWED_HOSTS`环境变量附加到`ALLOWED_HOSTS`列表来告诉 Django 我们正在提供哪些其他域。

接下来，让我们更新数据库配置：

```py
DATABASES['default'].update({
    'NAME': os.getenv('DJANGO_DB_NAME'),
    'USER': os.getenv('DJANGO_DB_USER'),
    'PASSWORD': os.getenv('DJANGO_DB_PASSWORD'),
    'HOST': os.getenv('DJANGO_DB_HOST'),
    'PORT': os.getenv('DJANGO_DB_PORT'),
})
```

我们使用环境变量的值更新了数据库配置。

现在我们的设置已经整理好了，让我们准备我们的服务器。

# 准备我们的服务器

现在我们的代码已经准备好投入生产，让我们准备我们的服务器。在本章中，我们将使用 Ubuntu 18.04（Bionic Beaver）。如果您使用其他发行版，则某些软件包名称可能不同，但我们将采取的步骤将是相同的。

为了准备我们的服务器，我们将执行以下步骤：

1.  安装所需的操作系统软件包

1.  设置 Elasticsearch

1.  创建数据库

让我们从安装我们需要的软件包开始。

# 安装所需的软件包

要在我们的服务器上运行 Answerly，我们需要确保正确的软件正在运行。

让我们创建一个我们将在`ubuntu/packages.txt`中需要的软件包列表：

```py
python3
python3-pip
virtualenv

apache2
libapache2-mod-wsgi-py3

postgresql
postgresql-client

openjdk-8-jre-headless
```

前面的代码将为以下内容安装软件包：

+   完全支持 Python 3

+   Apache HTTP 服务器

+   mod_wsgi，用于运行 Python Web 应用程序的 Apache HTTP 模块

+   PostgreSQL 数据库服务器和客户端

+   Java 8，Elasticsearch 所需

要安装软件包，请运行以下命令：

```py
$ sudo apt install -y $(cat /answerly/ubuntu/packages.txt)
```

接下来，我们将把我们的 Python 软件包安装到虚拟环境中：

```py
$ mkvirutalenv /opt/answerly.venv
$ source /opt/answerly.venv/bin/activate
$ pip install -r /answerly/requirements.production.txt
```

太好了！现在我们有了所有的软件包，我们需要设置 Elasticsearch。不幸的是，Ubuntu 没有提供最新版本的 Elasticsearch，所以我们将直接从 Elastic 安装它。

# 配置 Elasticsearch

我们将直接从 Elastic 获取 Elasticsearch。Elastic 通过在具有 Ubuntu 兼容的`.deb`软件包的服务器上运行来简化此过程（如果对您更方便，Elastic 还提供并支持 RPM）。最后，我们必须记住将 Elasticsearch 重新绑定到 localhost，否则我们将在开放的公共端口上运行一个不安全的服务器。

# 安装 Elasticsearch

让我们通过运行以下三个命令将 Elasticsearch 添加到我们信任的存储库中：

```py
$ wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
$ sudo apt install apt-transport-https
$ echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
$ sudo apt update
```

前面的命令执行以下四个步骤：

1.  将 Elastic GPG 密钥添加到受信任的 GPG 密钥列表中

1.  通过安装`apt-transport-https`软件包，确保`apt`通过`HTTPS`获取软件包

1.  添加一个新的源文件，列出 Elastic 软件包服务器，以便`apt`知道如何从 Elastic 获取 Elasticsearch 软件包

1.  更新可用软件包列表（现在将包括 Elasticsearch）

现在我们有了 Elasticsearch，让我们安装它：

```py
$ sudo apt install elasticsearch
```

接下来，让我们配置 Elasticsearch。

# 运行 Elasticsearch

默认情况下，Elasticsearch 配置为绑定到公共 IP 地址，并且不包括身份验证。

要更改 Elasticsearch 运行的地址，让我们编辑`/etc/elasticsearch/elasticsearch.yml`。找到带有`network.host`的行并更新如下：

```py
network.host: 127.0.0.1
```

如果您不更改`network.host`设置，那么您将在公共 IP 上运行没有身份验证的 Elasticsearch。您的服务器被黑客攻击将是不可避免的。

最后，我们要确保 Ubuntu 启动 Elasticsearch 并保持其运行。为了实现这一点，我们需要告诉 systemd 启动 Elasticsearch：

```py
$ sudo systemctl daemon-reload
$ sudo systemctl enable elasticsearch.service
$ sudo systemctl start elasticsearch.service
```

上述命令执行以下三个步骤：

1.  完全重新加载 systemd，然后它将意识到新安装的 Elasticsearch 服务

1.  启用 Elasticsearch 服务，以便在服务器启动时启动（以防重新启动或关闭）

1.  启动 Elasticsearch

如果您需要停止 Elasticsearch 服务，可以使用`systemctl`：`sudo systemctl stop elasticsearch.service`。

现在我们已经运行了 Elasticsearch，让我们配置数据库。

# 创建数据库

Django 支持迁移，但不能自行创建数据库或数据库用户。我们现在将编写一个脚本来为我们执行这些操作。

让我们将数据库创建脚本添加到我们的项目中的`postgres/make_database.sh`：

```py
#!/usr/bin/env bash

psql -v ON_ERROR_STOP=1 <<-EOSQL
    CREATE DATABASE $DJANGO_DB_NAME;
    CREATE USER $DJANGO_DB_USER;
    GRANT ALL ON DATABASE $DJANGO_DB_NAME to "$DJANGO_DB_USER";
    ALTER USER $DJANGO_DB_USER PASSWORD '$DJANGO_DB_PASSWORD';
    ALTER USER $DJANGO_DB_USER CREATEDB;
EOSQL
```

要创建数据库，请运行以下命令：

```py
$ sudo su postgres
$ export DJANGO_DB_NAME=answerly
$ export DJANGO_DB_USER=answerly
$ export DJANGO_DB_PASSWORD=password
$ bash /answerly/postgres/make_database.sh
```

上述命令执行以下三件事：

1.  切换到`postgres`用户，该用户被信任可以连接到 Postgres 数据库而无需任何额外的凭据。

1.  设置环境变量，描述我们的新数据库用户和模式。**记得将`password`的值更改为一个强密码。**

1.  执行`make_database.sh`脚本。

现在我们已经配置了服务器，让我们使用 Apache 和 mod_wsgi 部署 Answerly。

# 使用 Apache 部署 Answerly

我们将使用 Apache 和 mod_wsgi 部署 Answerly。mod_wsgi 是一个开源的 Apache 模块，允许 Apache 托管实现**Web 服务器网关接口**（**WSGI**）规范的 Python 程序。

Apache web 服务器是部署 Django 项目的众多优秀选项之一。许多组织都有一个运维团队，他们部署 Apache 服务器，因此使用 Apache 可以消除在项目中使用 Django 时的一些组织障碍。Apache（带有 mod_wsgi）还知道如何运行多个 web 应用程序并在它们之间路由请求，与我们在第五章中的先前配置不同，*使用 Docker 部署*，我们需要一个反向代理（NGINX）和 web 服务器（uWSGI）。使用 Apache 的缺点是它比 uWSGI 使用更多的内存。此外，Apache 没有一种将环境变量传递给我们的 WSGI 进程的方法。总的来说，使用 Apache 进行部署可以成为 Django 开发人员工具中非常有用和重要的一部分。

要部署，我们将执行以下操作：

1.  创建虚拟主机配置

1.  更新`wsgi.py`

1.  创建一个环境配置文件

1.  收集静态文件

1.  迁移数据库

1.  启用虚拟主机

让我们为我们的 Apache web 服务器开始创建一个虚拟主机配置。

# 创建虚拟主机配置

一个 Apache web 服务器可以使用来自不同位置的不同技术托管许多网站。为了保持每个网站的独立性，Apache 提供了定义虚拟主机的功能。每个虚拟主机是一个逻辑上独立的站点，可以为一个或多个域和端口提供服务。

由于 Apache 已经是一个很好的 Web 服务器，我们将使用它来提供静态文件。提供静态文件的 Web 服务器和我们的 mod_wsgi 进程不会竞争，因为它们将作为独立的进程运行，这要归功于 mod_wsgi 的守护进程模式。mod_wsgi 守护进程模式意味着 Answerly 将在与 Apache 的其余部分分开的进程中运行。Apache 仍然负责启动/停止这些进程。

让我们在项目的`apache/answerly.apache.conf`下添加 Apache 虚拟主机配置：

```py
<VirtualHost *:80>

    WSGIDaemonProcess answerly \
        python-home=/opt/answerly.venv \
        python-path=/answerly/django \
        processes=2 \
        threads=15
    WSGIProcessGroup answerly
    WSGIScriptAlias / /answerly/django/config/wsgi.py

    <Directory /answerly/django/config>
        <Files wsgi.py>
            Require all granted
        </Files>
    </Directory>

    Alias /static/ /answerly/django/static_root
    <Directory /answerly/django/static_root>
        Require all granted
    </Directory>

    ErrorLog ${APACHE_LOG_DIR}/error.log
    CustomLog ${APACHE_LOG_DIR}/access.log combined

</VirtualHost>
```

让我们仔细看一下其中的一些指令：

+   `<VirtualHost *:80>`：这告诉 Apache，直到关闭的`</VirtualHost>`标签之前的所有内容都是虚拟主机定义的一部分。

+   `WSGIDaemonProcess`：这配置 mod_wsgi 以守护进程模式运行。守护进程将被命名为`answerly`。`python-home`选项定义了守护进程将使用的 Python 进程的虚拟环境。`python-path`选项允许我们将我们的模块添加到守护进程的 Python 中，以便它们可以被导入。`processes`和`threads`选项告诉 Apache 要维护多少个进程和线程。

+   `WSGIProcessGroup`：这将虚拟主机与 Answerly mod_wsgi 守护进程关联起来。记住要保持`WSGIDaemonProcess`名称和`WSGIProcessGroup`名称相同。

+   `WSGIScriptAlias`：这描述了应该将哪些请求路由到哪个 WSGI 脚本。在我们的情况下，所有请求都应该转到 Answerly 的 WSGI 脚本。

+   `<Directory /answerly/django/config>`：这个块允许所有用户访问我们的 WSGI 脚本。

+   `Alias /static/ /answerly/django/static_root`：这将任何以`/static/`开头的请求路由到我们的静态文件根目录，而不是 mod_wsgi。

+   `<Directory /answerly/django/static_root>`：这个块允许用户访问`static_root`中的文件。

+   `ErrorLog`和`CustomLog`：它们描述了 Apache 应该将其日志发送到这个虚拟主机的位置。在我们的情况下，我们希望将其记录在 Apache 的`log`目录中（通常是`/var/log/apache`）。

我们现在已经配置 Apache 来运行 Answerly。然而，如果你比较一下你的 Apache 配置和第五章中的 uWSGI 配置，*使用 Docker 部署*，你会注意到一个区别。在 uWSGI 配置中，我们提供了我们的`production_settings.py`依赖的环境变量。然而，mod_wsgi 并没有为我们提供这样的功能。相反，我们将更新`django/config/wsgi.py`，以提供`production_settings.py`需要的环境变量。

# 更新 wsgi.py 以设置环境变量

现在，我们将更新`django/config/wsgi.py`，以提供`production_settings.py`想要的环境变量，但 mod_wsgi 无法提供。我们还将更新`wsgi.py`，在启动时读取配置文件，然后自己设置环境变量。这样，我们的生产设置不会与 mod_wsgi 或配置文件耦合。

让我们更新`django/config/wsgi.py`：

```py
import os
import configparser
from django.core.wsgi import get_wsgi_application

if not os.environ.get('DJANGO_SETTINGS_MODULE'):
    parser = configparser.ConfigParser()
    parser.read('/etc/answerly/answerly.ini')
    for name, val in parser['mod_wsgi'].items():
        os.environ[name.upper()] = val

application = get_wsgi_application()
```

在更新的`wsgi.py`中，我们检查是否有`DJANGO_SETTINGS_MODULE`环境变量。如果没有，我们解析我们的配置文件并设置环境变量。我们的`for`循环将变量的名称转换为大写，因为`ConfigParser`默认会将它们转换为`小写`。

接下来，让我们创建我们的环境配置文件。

# 创建环境配置文件

我们将把环境配置存储在`/etc/answerly/answerly.ini`下。我们不希望它存储在`/answerly`下，因为它不是我们代码的一部分。这个文件描述了*只有*这台服务器的设置。我们永远不应该将这个文件提交到版本控制中。

让我们在服务器上创建`/etc/answerly/answerly.ini`：

```py
[mod_wsgi]
DJANGO_ALLOWED_HOSTS=localhost
DJANGO_DB_NAME=answerly
DJANGO_DB_USER=answerly
DJANGO_DB_PASSWORD=password
DJANGO_DB_HOST=localhost
DJANGO_DB_PORT=5432
DJANGO_ES_INDEX=answerly
DJANGO_ES_HOST=localhost
DJANGO_ES_PORT=9200
DJANGO_LOG_FILE=/var/log/answerly/answerly.log
DJANGO_SECRET_KEY=a large random value
DJANGO_SETTINGS_MODULE=config.production_settings
```

以下是关于这个文件的两件事需要记住的：

+   记得将`DJANGO_DB_PASSWORD`设置为你在运行`make_database.sh`脚本时设置的相同值。*记得确保这个密码是强大和保密的*。

+   记得设置一个强大的`DJANGO_SECRET_KEY`值。

我们现在应该已经为 Apache 设置好了环境。接下来，让我们迁移数据库。

# 迁移数据库

我们在之前的步骤中为 Answerly 创建了数据库，但我们没有创建表。现在让我们使用 Django 内置的迁移工具迁移数据库。

在服务器上，我们希望执行以下命令：

```py
$ cd /answerly/django
$ source /opt/answerly.venv/bin/activate
$ export DJANGO_SECRET_KEY=anything
$ export DJANGO_DB_HOST=127.0.0.1 
$ export DJANGO_DB_PORT=5432 
$ export DJANGO_LOG_FILE=/var/log/answerly/answerly.log 
$ export DJANGO_DB_USER=myqa 
$ export DJANGO_DB_NAME=myqa 
$ export DJANGO_DB_PASSWORD=password 
$ sudo python3 manage.py migrate --settings=config.production_settings
```

我们的`django/config/production_settings.py`将要求我们提供带有值的`DJANGO_SECRET_KEY`，但在这种情况下不会使用它。但是，为`DJANGO_DB_PASSWORD`和其他`DJANGO_DB`变量提供正确的值至关重要。

一旦我们的`migrate`命令返回成功，那么我们的数据库将拥有我们需要的所有表。

接下来，让我们让我们的静态（JavaScript/CSS/图像）文件对我们的用户可用。

# 收集静态文件

在我们的虚拟主机配置中，我们配置了 Apache 来提供我们的静态（JS，CSS，图像等）文件。为了让 Apache 提供这些文件，我们需要将它们全部收集到一个父目录下。让我们使用 Django 内置的`manage.py collectstatic`命令来做到这一点。

在服务器上，让我们运行以下命令：

```py
$ cd /answerly/django
$ source /opt/answerly.venv/bin/activate
$ export DJANGO_SECRET_KEY=anything
$ export DJANGO_LOG_FILE=/var/log/answerly/answerly.log
$ sudo python3 manage.py collectstatic --settings=config.production_settings --no-input
```

上述命令将从所有已安装的应用程序复制静态文件到`/answerly/django/static_root`（根据`production_settings.py`中的`STATIC_ROOT`定义）。我们的虚拟主机配置告诉 Apache 直接提供这些文件。

现在，让我们告诉 Apache 开始提供 Answerly。

# 启用 Answerly 虚拟主机

为了让 Apache 向用户提供 Answerly，我们需要启用我们在上一节创建的虚拟主机配置，创建虚拟主机配置。要在 Apache 中启用虚拟主机，我们将在虚拟主机配置上添加一个软链接指向 Apache 的`site-enabled`目录，并告诉 Apache 重新加载其配置。

首先，让我们将我们的软链接添加到 Apache 的`site-enabled`目录：

```py
$ sudo ln -s /answerly/apache/answerly.apache.conf /etc/apache/site-enabled/000-answerly.conf
```

我们使用`001`作为软链接的前缀来控制我们的配置加载顺序。Apache 按字符顺序加载站点配置文件（例如，在 Unicode/ASCII 编码中，`B`在`a`之前）。前缀用于使顺序更加明显。

Apache 经常与默认站点捆绑在一起。查看`/etc/apache/sites-enabled/`以查找不想运行的站点。由于其中的所有内容都应该是软链接，因此可以安全地删除它们。

要激活虚拟主机，我们需要重新加载 Apache 的配置：

```py
$ sudo systemctl reload  apache2.service
```

恭喜！您已经在服务器上部署了 Answerly。

# 快速回顾本节

到目前为止，在本章中，我们已经了解了如何使用 Apache 和 mod_wsgi 部署 Django。首先，我们通过从 Ubuntu 和 Elastic（用于 Elasticsearch）安装软件包来配置了我们的服务器。然后，我们配置了 Apache 以将 Answerly 作为虚拟主机运行。我们的 Django 代码将由 mod_wsgi 执行。

到目前为止，我们已经看到了两种非常不同的部署方式，一种使用 Docker，一种使用 Apache 和 mod_wsgi。尽管是非常不同的环境，但我们遵循了许多相似的做法。让我们看看 Django 最佳实践是如何符合流行的十二要素应用方法论的。

# 将 Django 项目部署为十二要素应用

*十二要素应用*文档解释了一种开发 Web 应用和服务的方法论。这些原则是由 Adam Wiggins 和其他人在 2011 年主要基于他们在 Heroku（一家知名的平台即服务提供商）的经验而记录的。Heroku 是最早帮助开发人员构建易于扩展的 Web 应用和服务的 PaaS 之一。自发布以来，十二要素应用的原则已经塑造了很多关于如何构建和部署 SaaS 应用（如 Web 应用）的思考。

十二要素提供了许多好处，如下：

+   使用声明性格式来简化自动化和入职

+   强调在部署环境中的可移植性

+   鼓励生产/开发环境的一致性和持续部署和集成

+   简化扩展而无需重新架构

然而，在评估十二因素时，重要的是要记住它们与 Heroku 的部署方法紧密相关。并非所有平台（或 PaaS 提供商）都有完全相同的方法。这并不是说十二因素是正确的，其他方法是错误的，反之亦然。相反，十二因素是要牢记的有用原则。您应该根据需要调整它们以帮助您的项目，就像您对待任何方法论一样。

单词*应用程序*的十二因素用法与 Django 的可用性不同：

+   Django 项目相当于十二因素应用程序

+   Django 应用程序相当于十二因素库

在本节中，我们将研究十二个因素的每个含义以及它们如何应用到您的 Django 项目中。

# 因素 1 - 代码库

“一个代码库在修订控制中跟踪，多个部署” - [12factor.net](http://12factor.net)

这个因素强调了以下两点：

+   所有代码都应该在版本控制的代码存储库（repo）中进行跟踪

+   每次部署都应该能够引用该存储库中的单个版本/提交

这意味着当我们遇到错误时，我们确切地知道是哪个代码版本负责。如果我们的项目跨越多个存储库，十二因素方法要求共享代码被重构为库并作为依赖项进行跟踪（参见*因素 2 - 依赖关系*部分）。如果多个项目使用同一个存储库，那么它们应该被重构为单独的存储库（有时称为*多存储库*）。自十二因素首次发布以来，多存储库与单存储库（一个存储库用于多个项目）的使用已经越来越受到争议。一些大型项目发现使用单存储库有益处。其他项目通过多个存储库取得了成功。

基本上，这个因素努力确保我们知道在哪个环境中运行什么。

我们可以以可重用的方式编写我们的 Django 应用程序，以便它们可以作为使用`pip`安装的库进行托管（多存储库样式）。或者，您可以通过修改 Django 项目的 Python 路径，将所有 Django 项目和应用程序托管在同一个存储库（单存储库）中。

# 因素 2 - 依赖关系

“明确声明和隔离依赖关系” - [12 factor.net](https://12factor.net)

十二因素应用程序不应假设其环境的任何内容。项目使用的库和工具必须由项目声明并作为部署的一部分安装（参见*因素 5 - 构建、发布和运行*部分）。所有运行的十二因素应用程序都应该相互隔离。

Django 项目受益于 Python 丰富的工具集。 “在 Python 中，这些步骤有两个单独的工具 - Pip 用于声明，Virtualenv 用于隔离”（[`12factor.net/dependencies`](https://12factor.net/dependencies)）。在 Answerly 中，我们还使用了一系列我们用`apt`安装的 Ubuntu 软件包。

# 因素 3 - 配置

将配置存储在环境中 - [12factor.net](http://12factor.net)

十二因素应用程序方法提供了一个有用的配置定义：

“应用程序的配置是在部署之间可能变化的所有内容（暂存、生产、开发环境等）” - [`12factor.net/config`](https://12factor.net/config)

十二因素应用程序方法还鼓励使用环境变量来传递配置值给我们的代码。这意味着如果出现问题，我们可以测试确切部署的代码（由因素 1 提供）以及使用的确切配置。我们还可以通过使用不同的配置部署相同的代码来检查错误是配置问题还是代码问题。

在 Django 中，我们的配置由我们的`settings.py`文件引用。在 MyMDB 和 Answerly 中，我们看到了一些常见的配置值，如`SECRET_KEY`、数据库凭据和 API 密钥（例如 AWS 密钥），通过环境变量传递。

然而，这是一个领域，Django 最佳实践与十二要素应用的最严格解读有所不同。Django 项目通常为分别用于分阶段、生产和本地开发的设置文件创建一个单独的设置文件，大多数设置都是硬编码的。主要是凭据和秘密作为环境变量传递。

# Factor 4 – 后备服务

"将后备服务视为附加资源" – [12factor.net](https://12factor.net)

十二要素应用不应关心后备服务（例如数据库）的位置，并且应始终通过 URL 访问它。这样做的好处是我们的代码不与特定环境耦合。这种方法还允许我们架构的每个部分独立扩展。

在本章中部署的 Answerly 与其数据库位于同一服务器上。然而，我们没有使用本地身份验证机制，而是向 Django 提供了主机、端口和凭据。这样，我们可以将数据库移动到另一台服务器上，而不需要更改任何代码。我们只需要更新我们的配置。

Django 的编写假设我们会将大多数服务视为附加资源（例如，大多数数据库文档都是基于这一假设）。在使用第三方库时，我们仍然需要遵循这一原则。

# Factor 5 – 构建、发布和运行

"严格分离构建和运行阶段" – [12factor.net](https://12factor.net)

十二要素方法鼓励将部署分为三个明确的步骤：

1.  **构建**：代码和依赖项被收集到一个单一的捆绑包中（一个*构建*）

1.  **发布**：构建与配置组合在一起，准备执行

1.  **运行**：组合构建和配置的执行位置

十二要素应用还要求每个发布都有一个唯一的 ID，以便可以识别它。

这种部署细节已经超出了 Django 的范围，对这种严格的三步模型的遵循程度有各种各样。在第五章中看到的使用 Django 和 Docker 的项目可能会非常严格地遵循这一原则。MyMDB 有一个清晰的构建，所有依赖项都捆绑在 Docker 镜像中。然而，在本章中，我们从未进行捆绑构建。相反，我们在代码已经在服务器上之后安装依赖项（运行`pip install`）。许多项目都成功地使用了这种简单的模型。然而，随着项目规模的扩大，这可能会引起复杂性。Answerly 的部署展示了十二要素原则如何可以被弯曲，但对于某些项目仍然有效。

# Factor 6 – 进程

"将应用程序作为一个或多个无状态进程执行" – [12factor.net](https://12factor.net)

这一因素的重点是应用进程应该是*无状态*的。每个任务都是在不依赖前一个任务留下数据的情况下执行的。相反，状态应该存储在后备服务中（参见*Factor 4 – 后备服务*部分），比如数据库或外部缓存。这使得应用能够轻松扩展，因为所有进程都同样有资格处理请求。

Django 是围绕这一假设构建的。即使是会话，用户的登录状态也不是保存在进程中，而是默认保存在数据库中。视图类的实例永远不会被重用。Django 接近违反这一点的唯一地方是缓存后端之一（本地内存缓存）。然而，正如我们讨论过的，那是一个低效的后端。通常，Django 项目会为它们的缓存使用一个后备服务（例如 memcached）。

# Factor 7 – 端口绑定

"通过端口绑定导出服务" – [12factor.net](https://12factor.net)

这个因素的重点是我们的进程应该通过其端口直接访问。访问一个项目应该是向`app.example.com:1234`发送一个正确形成的请求。此外，十二要素应用程序不应该作为 Apache 模块或 Web 服务器容器运行。如果我们的项目需要解析 HTTP 请求，应该使用库（参见*因素 2-依赖*部分）来解析它们。

Django 遵循这个原则的部分。用户通过 HTTP 端口使用 HTTP 访问 Django 项目。与十二要素有所不同的是，Django 的一个方面几乎总是作为 Web 服务器的子进程运行（无论是 Apache、uWSGI 还是其他什么）。进行端口绑定的是 Web 服务器，而不是 Django。然而，这种微小的差异并没有阻止 Django 项目有效地扩展。

# 因素 8-并发

“通过进程模型扩展”- [12factor.net](https://12factor.net)

十二要素应用程序的原则侧重于扩展（对于像 Heroku 这样的 PaaS 提供商来说是一个重要的关注点）。在因素 8 中，我们看到之前做出的权衡和决策如何帮助项目扩展。

由于项目作为无状态进程运行（参见*因素 6-进程*部分），作为端口（参见*因素 7-端口绑定*部分）可用，并发性只是拥有更多进程（跨一个或多个机器）的问题。进程不需要关心它们是否在同一台机器上，因为任何状态（比如问题的答案）都存储在后备服务（参见*因素 4-后备服务*部分）中，比如数据库。因素 8 告诉我们要相信 Unix 进程模型来运行服务，而不是创建守护进程或创建 PID 文件。

由于 Django 项目作为 Web 服务器的子进程运行，它们经常遵循这个原则。需要扩展的 Django 项目通常使用反向代理（例如 Nginx）和轻量级 Web 服务器（例如 uWSGI 或 Gunicorn）的组合。Django 项目不直接关注进程的管理方式，而是遵循它们正在使用的 Web 服务器的最佳实践。

# 因素 9-可处置性

“通过快速启动和优雅关闭来最大限度地提高鲁棒性”- [12factor.net](https://12factor.net)

可处置性因素有两个部分。首先，十二要素应用程序应该能够在进程启动后不久开始处理请求。记住，所有它的依赖关系（参见*因素 2-依赖*部分）已经被安装（参见*因素 5-构建、发布和运行*部分）。十二要素应用程序应该处理进程停止或优雅关闭。进程不应该使十二要素应用程序处于无效状态。

Django 项目能够优雅地关闭，因为 Django 默认会将每个请求包装在一个原子事务中。如果一个 Django 进程（无论是由 uWSGI、Apache 还是其他任何东西管理的）在处理请求时停止，事务将永远不会被提交。数据库将放弃该事务。当我们处理其他后备服务（例如 S3 或 Elasticsearch）不支持事务时，我们必须确保在设计中考虑到这一点。

# 因素 10-开发/生产对等性

“尽量使开发、分期和生产尽可能相似”- [12factor.net](https://12factor.net)

十二要素应用程序运行的所有环境应尽可能相似。当十二要素应用程序是一个简单的进程时（参见*因素 6-进程*部分），这就容易得多。这还包括十二要素应用程序使用的后备服务（参见*因素 4-后备服务*部分）。例如，十二要素应用程序的开发环境应该包括与生产环境相同的数据库。像 Docker 和 Vagrant 这样的工具可以使今天实现这一点变得更加容易。

Django 的一般最佳实践是在开发和生产中使用相同的数据库（和其他后端服务）。在本书中，我们一直在努力做到这一点。然而，Django 社区通常在开发中使用`manage.py runserver`命令，而不是运行 uWSGI 或 Apache。

# 11 因素 - 日志

"将日志视为事件流" - [12factor.net](https://12factor.net)

日志应该只作为无缓冲的`stdout`流输出，*十二因素应用程序永远不会关心其输出流的路由或存储*（[`12factor.net/logs`](https://12factor.net/logs)）。当进程运行时，它应该只输出无缓冲的内容到`stdout`。然后启动进程的人（无论是开发人员还是生产服务器的 init 进程）可以适当地重定向该流。

Django 项目通常使用 Python 的日志模块。这可以支持写入日志文件或输出无缓冲流。一般来说，Django 项目会追加到一个文件中。该文件可以单独处理或旋转（例如，使用`logrotate`实用程序）。

# 12 因素 - 管理流程

"将管理/管理任务作为一次性进程运行" - [12factor.net](https://12factor.net)

所有项目都需要不时运行一次性任务（例如，数据库迁移）。当十二因素应用程序的一次性任务运行时，它应该作为一个独立的进程运行，而不是处理常规请求的进程。但是，一次性进程应该与所有其他进程具有相同的环境。

在 Django 中，这意味着在运行我们的`manage.py`任务时使用相同的虚拟环境、设置文件和环境变量作为我们的正常进程。这就是我们之前迁移数据库时所做的。

# 快速审查本节

在审查了十二因素应用程序的所有原则之后，我们将看看 Django 项目如何遵循这些原则，以帮助我们的项目易于部署、扩展和自动化。

Django 项目和严格的十二因素应用程序之间的主要区别在于，Django 应用程序是由 Web 服务器而不是作为独立进程运行的（因素 6）。然而，只要我们避免复杂的 Web 服务器配置（就像在本书中所做的那样），我们就可以继续获得作为十二因素应用程序的好处。

# 摘要

在本章中，我们专注于将 Django 部署到运行 Apache 和 mod_wsgi 的 Linux 服务器上。我们还审查了十二因素应用程序的原则以及 Django 应用程序如何使用它们来实现易于部署、扩展和自动化。

恭喜！您已经推出了 Answerly。

在下一章中，我们将看看如何创建一个名为 MailApe 的邮件列表管理应用程序。
