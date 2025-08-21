# 第五章：使用 Docker 部署

在本章中，我们将看看如何使用托管在亚马逊的**电子计算云**（**EC2**）上的 Docker 容器将 MyMDB 部署到生产环境。我们还将使用**亚马逊网络服务**（**AWS**）的**简单存储服务**（**S3**）来存储用户上传的文件。

我们将做以下事情：

+   将我们的要求和设置文件拆分为单独的开发和生产设置

+   为 MyMDB 构建一个 Docker 容器

+   构建数据库容器

+   使用 Docker Compose 启动两个容器

+   在云中的 Linux 服务器上将 MyMDB 启动到生产环境

首先，让我们拆分我们的要求和设置，以便保持开发和生产值分开。

# 为生产和开发组织配置

到目前为止，我们保留了一个要求文件和一个`settings.py`文件。这使得开发变得方便。但是，我们不能在生产中使用我们的开发设置。

当前的最佳实践是为每个环境使用单独的文件。然后，每个环境的文件都导入具有共享值的公共文件。我们将使用此模式进行要求和设置文件。

让我们首先拆分我们的要求文件。

# 拆分要求文件

让我们在项目的根目录下创建`requirements.common.txt`：

```py
django<2.1
psycopg2
Pillow<4.4.0
```

无论我们处于哪种环境，我们始终需要 Django、Postgres 驱动程序和 Pillow（用于`ImageField`类）。但是，此要求文件永远不会直接使用。

接下来，让我们在`requirements.dev.txt`中列出我们的开发要求：

```py
-r requirements.common.txt
django-debug-toolbar==1.8
```

上述文件将安装来自`requirements.common.txt`（感谢`-r`）和 Django 调试工具栏的所有内容。

对于我们的生产软件包，我们将使用`requirements.production.txt`：

```py
-r requirements.common.txt
django-storages==1.6.5
boto3==1.4.7
uwsgi==2.0.15
```

这也将安装来自`requirements.common.txt`的软件包。它还将安装`boto3`和`django-storages`软件包，以帮助我们轻松地将文件上传到 S3。`uwsgi`软件包将提供我们用于提供 Django 的服务器。

要为生产环境安装软件包，我们现在可以执行以下命令：

```py
$ pip install -r requirements.production.txt
```

接下来，让我们按类似的方式拆分设置文件。

# 拆分设置文件

再次，我们将遵循当前的 Django 最佳实践，将我们的设置文件分成以下三个文件：`common_settings.py`，`production_settings.py`和`dev_settings.py`。

# 创建 common_settings.py

我们将通过将当前的`settings.py`文件重命名为`common_settings.py`，然后进行本节中提到的更改来创建`common_settings.py`。

让我们将`DEBUG = False`更改为不会*意外*处于调试模式的新设置文件。然后，让我们更改`SECRET_KEY`设置，以便通过更改其行来从环境变量获取其值：

```py
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')
```

让我们还添加一个新的设置`STATIC_ROOT`。`STATIC_ROOT`是 Django 将从已安装的应用程序中收集所有静态文件的目录，以便更容易地提供它们：

```py
STATIC_ROOT = os.path.join(BASE_DIR, 'gathered_static_files')
```

在数据库配置中，我们可以删除所有凭据，但保留`ENGINE`值（为了明确起见，我们打算在任何地方都使用 Postgres）：

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
    }
}
```

最后，让我们删除`CACHES`设置。这将在每个环境中以不同的方式配置。

接下来，让我们创建一个开发设置文件。

# 创建 dev_settings.py

我们的开发设置将在`django/config/dev_settings.py`中。我们将逐步构建它。

首先，我们将从`common_settings`中导入所有内容：

```py
from config.common_settings import *
```

然后，我们将覆盖`DEBUG`和`SECRET_KEY`设置：

```py
DEBUG = True
SECRET_KEY = 'some secret'
```

在开发中，我们希望以调试模式运行。我们还会感到安全，硬编码一个秘密密钥，因为我们知道它不会在生产中使用。

接下来，让我们更新`INSTALLED_APPS`列表：

```py
INSTALLED_APPS += [
    'debug_toolbar',
]
```

在开发中，我们可以通过将一系列仅用于开发的应用程序附加到`INSTALLED_APPS`列表中来运行额外的应用程序（例如 Django 调试工具栏）。

然后，让我们更新数据库配置：

```py
DATABASES['default'].update({
    'NAME': 'mymdb',
    'USER': 'mymdb',
    'PASSWORD': 'development',
    'HOST': 'localhost',
    'PORT': '5432',
})
```

由于我们的开发数据库是本地的，我们可以在设置中硬编码值，使文件更简单。如果您的数据库不是本地的，请避免将密码检入版本控制，并在生产中使用`os.getenv()`。

接下来，让我们更新缓存配置：

```py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'default-locmemcache',
        'TIMEOUT': 5,
    }
}
```

在我们的开发缓存中，我们将使用非常短的超时时间。

最后，我们需要设置文件上传目录：

```py
# file uploads
MEDIA_ROOT = os.path.join(BASE_DIR, '../media_root')
```

在开发中，我们将在本地文件系统中存储上传的文件。我们将使用`MEDIA_ROOT`指定要上传到的目录。

Django Debug Toolbar 也需要一些配置：

```py
# Django Debug Toolbar
INTERNAL_IPS = [
    '127.0.0.1',
]
```

Django Debug Toolbar 只会在预定义的 IP 上呈现，所以我们会给它我们的本地 IP，这样我们就可以在本地使用它。

我们还可以添加我们的开发专用应用程序可能需要的更多设置。

接下来，让我们添加生产设置。

# 创建 production_settings.py

让我们在`django/config/production_settings.py`中创建我们的生产设置。

`production_settings.py`类似于`dev_settings.py`，但通常使用`os.getenv()`从环境变量中获取值。这有助于我们将秘密信息（例如密码、API 令牌等）排除在版本控制之外，并将设置与特定服务器解耦：

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

首先，我们导入通用设置。出于谨慎起见，我们确保调试模式已关闭。

设置`SECRET_KEY`对于我们的系统保持安全至关重要。我们使用`assert`来防止 Django 在没有`SECRET_KEY`的情况下启动。`common_settings`模块应该已经从环境变量中设置了它。

生产网站将从除`localhost`之外的域访问。然后我们通过将`DJANGO_ALLOWED_HOSTS`环境变量附加到`ALLOWED_HOSTS`列表来告诉 Django 我们正在服务的其他域。

接下来，我们将更新数据库配置：

```py
DATABASES['default'].update({
    'NAME': os.getenv('DJANGO_DB_NAME'),
    'USER': os.getenv('DJANGO_DB_USER'),
    'PASSWORD': os.getenv('DJANGO_DB_PASSWORD'),
    'HOST': os.getenv('DJANGO_DB_HOST'),
    'PORT': os.getenv('DJANGO_DB_PORT'),
})
```

我们使用来自环境变量的值更新数据库配置。

然后，需要设置缓存配置。

```py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'default-locmemcache',
        'TIMEOUT': int(os.getenv('DJANGO_CACHE_TIMEOUT'), ),
    }
}
```

在生产中，我们将接受本地内存缓存的权衡。我们使用另一个环境变量在运行时配置超时时间。

接下来，需要设置文件上传配置设置。

```py
# file uploads
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY_ID')
AWS_STORAGE_BUCKET_NAME = os.getenv('DJANGO_UPLOAD_S3_BUCKET')
```

在生产中，我们不会将上传的图像存储在容器的本地文件系统上。Docker 的一个核心概念是容器是短暂的。停止和删除容器并用另一个替换应该是可以接受的。如果我们将上传的图像存储在本地，我们将违背这一理念。

不将上传的文件存储在本地的另一个原因是，它们也应该从不同的域提供服务（我们在第三章中讨论过这个问题，*海报、头像和安全性*）。我们将使用 S3 存储，因为它便宜且易于使用。

`django-storages`应用程序为许多 CDN 提供文件存储后端，包括 S3。我们告诉 Django 使用 S3，方法是更改`DEFAULT_FILE_STORAGE`设置。`S3Boto3Storage`后端需要一些额外的设置才能与 AWS 一起工作，包括 AWS 访问密钥、AWS 秘密访问密钥和目标存储桶的名称。我们将在 AWS 部分稍后讨论这两个访问密钥。

现在我们的设置已经组织好了，我们可以创建我们的 MyMDB `Dockerfile`。

# 创建 MyMDB Dockerfile

在本节中，我们将为 MyMDB 创建一个 Dockerfile。Docker 基于镜像运行容器。镜像由 Dockerfile 定义。Dockerfile 必须扩展另一个 Dockerfile（保留的`scratch`镜像是这个周期的结束）。

Docker 的理念是每个容器应该只有一个关注点（目的）。这可能意味着它运行一个单一进程，或者它可能运行多个一起工作的进程。在我们的情况下，它将运行 uWSGI 和 Nginx 进程来提供 MyMDB。

令人困惑的是，Dockerfile 既指预期的*文件名*，也指*文件类型*。所以`Dockerfile`是一个 Dockerfile。

让我们在项目的根目录中创建一个名为`Dockerfile`的文件。 Dockerfile 使用自己的语言来定义图像中的文件/目录，以及在制作图像时需要运行的任何命令。编写 Dockerfile 的完整指南超出了本章的范围。相反，我们将逐步构建我们的`Dockerfile`，仅讨论最相关的元素。

我们将通过以下六个步骤构建我们的`Dockerfile`：

1.  初始化基础镜像并将源代码添加到镜像中

1.  安装软件包

1.  收集静态文件

1.  配置 Nginx

1.  配置 uWSGI

1.  清理不必要的资源

# 启动我们的 Dockerfile

我们的`Dockerfile`的第一部分告诉 Docker 要使用哪个镜像作为基础，添加我们的代码，并创建一些常见的目录：

```py
FROM phusion/baseimage

# add code and directories
RUN mkdir /mymdb
WORKDIR /mymdb
COPY requirements* /mymdb/
COPY django/ /mymdb/django
COPY scripts/ /mymdb/scripts
RUN mkdir /var/log/mymdb/
RUN touch /var/log/mymdb/mymdb.log
```

让我们更详细地看看这些说明：

+   `FROM`：Dockerfile 中需要这个。`FROM`告诉 Docker 我们的镜像要使用哪个基础镜像。我们将使用`phusion/baseimage`，因为它提供了许多方便的设施并且占用的内存很少。它是一个专为 Docker 定制的 Ubuntu 镜像，具有一个更小、易于使用的 init 服务管理器，称为 runit（而不是 Ubuntu 的 upstart）。

+   `RUN`：这在构建图像的过程中执行命令。`RUN mkdir /mymdb`创建我们将存储文件的目录。

+   `WORKDIR`：这为我们所有未来的`RUN`命令设置了工作目录。

+   `COPY`：这将文件（或目录）从我们的文件系统添加到图像中。源路径是相对于包含我们的`Dockerfile`的目录的。最好将目标路径设置为绝对路径。

我们还将引用一个名为`scripts`的新目录。让我们在项目目录的根目录中创建它：

```py
$ mkdir scripts
```

作为配置和构建新镜像的一部分，我们将创建一些小的 bash 脚本，我们将保存在`scripts`目录中。

# 在 Dockerfile 中安装软件包

接下来，我们将告诉我们的`Dockerfile`安装我们将需要的所有软件包：

```py
RUN apt-get -y update
RUN apt-get install -y \
    nginx \
    postgresql-client \
    python3 \
    python3-pip
RUN pip3 install virtualenv
RUN virtualenv /mymdb/venv
RUN bash /mymdb/scripts/pip_install.sh /mymdb
```

我们使用`RUN`语句来安装 Ubuntu 软件包并创建虚拟环境。要将我们的 Python 软件包安装到虚拟环境中，我们将在`scripts/pip_install.sh`中创建一个小脚本：

```py
#!/usr/bin/env bash

root=$1
source $root/venv/bin/activate

pip3 install -r $root/requirements.production.txt
```

上述脚本只是激活虚拟环境并在我们的生产需求文件上运行`pip3 install`。

在 Dockerfile 的中间调试长命令通常很困难。将命令包装在脚本中可以使它们更容易调试。如果某些内容不起作用，您可以使用`docker exec -it bash -l`命令连接到容器并像平常一样调试脚本。

# 在 Dockerfile 中收集静态文件

静态文件是支持我们网站的 CSS、JavaScript 和图像。静态文件可能并非总是由我们创建。一些静态文件来自安装的 Django 应用程序（例如 Django 管理）。让我们更新我们的`Dockerfile`以收集静态文件：

```py
# collect the static files
RUN bash /mymdb/scripts/collect_static.sh /mymdb
```

再次，我们将命令包装在脚本中。让我们将以下脚本添加到`scripts/collect_static.sh`中：

```py
#!/usr/bin/env bash

root=$1
source $root/venv/bin/activate

export DJANGO_CACHE_TIMEOUT=100
export DJANGO_SECRET_KEY=FAKE_KEY
export DJANGO_SETTINGS_MODULE=config.production_settings

cd $root/django/

python manage.py collectstatic
```

上述脚本激活了我们在前面的代码中创建的虚拟环境，并设置了所需的环境变量。在这种情况下，大多数这些值都不重要，只要变量存在即可。但是，`DJANGO_SETTINGS_MODULE`环境变量非常重要。`DJANGO_SETTINGS_MODULE`环境变量用于 Django 查找设置模块。如果我们不设置它并且没有`config/settings.py`，那么 Django 将无法启动（甚至`manage.py`命令也会失败）。

# 将 Nginx 添加到 Dockerfile

要配置 Nginx，我们将添加一个配置文件和一个 runit 服务脚本：

```py
COPY nginx/mymdb.conf /etc/nginx/sites-available/mymdb.conf
RUN rm /etc/nginx/sites-enabled/*
RUN ln -s /etc/nginx/sites-available/mymdb.conf /etc/nginx/sites-enabled/mymdb.conf

COPY runit/nginx /etc/service/nginx
RUN chmod +x /etc/service/nginx/run
```

# 配置 Nginx

让我们将一个 Nginx 配置文件添加到`nginx/mymdb.conf`中：

```py
# the upstream component nginx needs
# to connect to
upstream django {
    server 127.0.0.1:3031;
}

# configuration of the server
server {

    # listen on all IPs on port 80
    server_name 0.0.0.0;
    listen      80;
    charset     utf-8;

    # max upload size
    client_max_body_size 2M;

    location /static {
        alias /mymdb/django/gathered_static_files;
    }

    location / {
        uwsgi_pass  django;
        include     /etc/nginx/uwsgi_params;
    }

}
```

Nginx 将负责以下两件事：

+   提供静态文件（以`/static`开头的 URL）

+   将所有其他请求传递给 uWSGI

`upstream`块描述了我们 Django（uWSGI）服务器的位置。在`location /`块中，nginx 被指示使用 uWSGI 协议将请求传递给上游服务器。`include /etc/nginx/uwsgi_params`文件描述了如何映射标头，以便 uWSGI 理解它们。

`client_max_body_size`是一个重要的设置。它描述了文件上传的最大大小。将这个值设置得太大可能会暴露漏洞，因为攻击者可以用巨大的请求压倒服务器。

# 创建 Nginx runit 服务

为了让`runit`知道如何启动 Nginx，我们需要提供一个`run`脚本。我们的`Dockerfile`希望它在`runit/nginx/run`中：

```py
#!/usr/bin/env bash

exec /usr/sbin/nginx \
    -c /etc/nginx/nginx.conf \
    -g "daemon off;"
```

`runit`不希望其服务分叉出一个单独的进程，因此我们使用`daemon off`来运行 Nginx。此外，`runit`希望我们使用`exec`来替换我们脚本的进程，新的 Nginx 进程。

# 将 uWSGI 添加到 Dockerfile

我们使用 uWSGI，因为它通常被评为最快的 WSGI 应用服务器。让我们通过添加以下代码到我们的`Dockerfile`中来设置它：

```py
# configure uwsgi
COPY uwsgi/mymdb.ini /etc/uwsgi/apps-enabled/mymdb.ini
RUN mkdir -p /var/log/uwsgi/
RUN touch /var/log/uwsgi/mymdb.log
RUN chown www-data /var/log/uwsgi/mymdb.log
RUN chown www-data /var/log/mymdb/mymdb.log

COPY runit/uwsgi /etc/service/uwsgi
RUN chmod +x /etc/service/uwsgi/run
```

这指示 Docker 使用`mymdb.ini`文件配置 uWSGI，创建日志目录，并添加 uWSGI runit 服务。为了让 runit 启动 uWSGI 服务，我们使用`chmod`命令给予 runit 脚本执行权限。

# 配置 uWSGI 运行 MyMDB

让我们在`uwsgi/mymdb.ini`中创建 uWSGI 配置：

```py
[uwsgi]
socket = 127.0.0.1:3031
chdir = /mymdb/django/
virtualenv = /mymdb/venv
wsgi-file = config/wsgi.py
env = DJANGO_SECRET_KEY=$(DJANGO_SECRET_KEY)
env = DJANGO_LOG_LEVEL=$(DJANGO_LOG_LEVEL)
env = DJANGO_ALLOWED_HOSTS=$(DJANGO_ALLOWED_HOSTS)
env = DJANGO_DB_NAME=$(DJANGO_DB_NAME)
env = DJANGO_DB_USER=$(DJANGO_DB_USER)
env = DJANGO_DB_PASSWORD=$(DJANGO_DB_PASSWORD)
env = DJANGO_DB_HOST=$(DJANGO_DB_HOST)
env = DJANGO_DB_PORT=$(DJANGO_DB_PORT)
env = DJANGO_CACHE_TIMEOUT=$(DJANGO_CACHE_TIMEOUT)
env = AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)
env = AWS_SECRET_ACCESS_KEY_ID=$(AWS_SECRET_ACCESS_KEY_ID)
env = DJANGO_UPLOAD_S3_BUCKET=$(DJANGO_UPLOAD_S3_BUCKET)
env = DJANGO_LOG_FILE=$(DJANGO_LOG_FILE)
processes = 4
threads = 4
```

让我们更仔细地看一下其中一些设置：

+   `socket`告诉 uWSGI 在`127.0.0.1:3031`上使用其自定义的`uwsgi`协议打开一个套接字（令人困惑的是，协议和服务器的名称相同）。

+   `chdir`改变了进程的工作目录。所有路径都需要相对于这个位置。

+   `virtualenv`告诉 uWSGI 项目虚拟环境的路径。

+   每个`env`指令为我们的进程设置一个环境变量。我们可以在我们的代码中使用`os.getenv()`访问这些变量（例如，`production_settings.py`）。

+   `$(...)`是从 uWSGI 进程自己的环境中引用的环境变量（例如，`$(DJANGO_SECRET_KEY )`）。

+   `proccesses`设置我们应该运行多少个进程。

+   `threads`设置每个进程应该有多少线程。

`processes`和`threads`设置将根据生产性能进行微调。

# 创建 uWSGI runit 服务

为了让 runit 知道如何启动 uWSGI，我们需要提供一个`run`脚本。我们的`Dockerfile`希望它在`runit/uwsgi/run`中。这个脚本比我们用于 Nginx 的要复杂：

```py
#!/usr/bin/env bash

source /mymdb/venv/bin/activate

export PGPASSWORD="$DJANGO_DB_PASSWORD"
psql \
    -h "$DJANGO_DB_HOST" \
    -p "$DJANGO_DB_PORT" \
    -U "$DJANGO_DB_USER" \
    -d "$DJANGO_DB_NAME"

if [[ $? != 0 ]]; then
    echo "no db server"
    exit 1
fi

pushd /mymdb/django

python manage.py migrate

if [[ $? != 0 ]]; then
    echo "can't migrate"
    exit 2
fi
popd

exec /sbin/setuser www-data \
    uwsgi \
    --ini /etc/uwsgi/apps-enabled/mymdb.ini \
    >> /var/log/uwsgi/mymdb.log \
    2>&1
```

这个脚本做了以下三件事：

+   检查是否可以连接到数据库，否则退出

+   运行所有迁移或失败时退出

+   启动 uWSGI

runit 要求我们使用`exec`来启动我们的进程，以便 uWSGI 将替换`run`脚本的进程。

# 完成我们的 Dockerfile

作为最后一步，我们将清理并记录我们正在使用的端口：

```py
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 80
```

`EXPOSE`语句记录了我们正在使用的端口。重要的是，它实际上并不打开任何端口。当我们运行容器时，我们将不得不这样做。

接下来，让我们为我们的数据库创建一个容器。

# 创建数据库容器

我们需要一个数据库来在生产中运行 Django。PostgreSQL Docker 社区为我们提供了一个非常强大的 Postgres 镜像，我们可以扩展使用。

让我们在`docker/psql/Dockerfile`中为我们的数据库创建另一个容器：

```py
FROM postgres:10.1

ADD make_database.sh /docker-entrypoint-initdb.d/make_database.sh
```

这个`Dockerfile`的基本镜像将使用 Postgres 10.1。它还有一个方便的设施，它将执行`/docker-entrypoint-initdb.d`中的任何 shell 或 SQL 脚本作为 DB 初始化的一部分。我们将利用这一点来创建我们的 MyMDB 数据库和用户。

让我们在`docker/psql/make_database.sh`中创建我们的数据库初始化脚本：

```py
#!/usr/bin/env bash

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE $DJANGO_DB_NAME;
    CREATE USER $DJANGO_DB_USER;
    GRANT ALL ON DATABASE $DJANGO_DB_NAME TO "$DJANGO_DB_USER";
    ALTER USER $DJANGO_DB_USER PASSWORD '$DJANGO_DB_PASSWORD';
    ALTER USER $DJANGO_DB_USER CREATEDB;
EOSQL
```

我们在前面的代码中使用了一个 shell 脚本，以便我们可以使用环境变量来填充我们的 SQL。

现在我们的两个容器都准备好了，让我们确保我们实际上可以通过注册并配置 AWS 来启动它们。

# 在 AWS S3 上存储上传的文件

我们期望我们的 MyMDB 将文件保存到 S3。为了实现这一点，我们需要注册 AWS，然后配置我们的 shell 以便能够使用 AWS。

# 注册 AWS

要注册，请转到[`aws.amazon.com`](https://aws.amazon.com)并按照其说明操作。请注意，注册是免费的。

我们将使用的资源在撰写本书时都在 AWS 免费层中。免费层的一些元素仅在第一年对新帐户可用。在执行任何 AWS 命令之前，请检查您的帐户的资格。

# 设置 AWS 环境

为了与 AWS API 交互，我们将需要以下两个令牌——访问密钥和秘密访问密钥。这对密钥定义了对帐户的访问。

要生成一对令牌，转到[`console.aws.amazon.com/iam/home?region=us-west-2#/security_credential_`](https://console.aws.amazon.com/iam/home?region=us-west-2#/security_credential)，单击访问密钥，然后单击创建新的访问密钥按钮。如果您丢失了秘密访问密钥，将无法检索它，因此请确保将其保存在安全的地方。

上述的 AWS 控制台链接将为您的根帐户生成令牌。在我们测试时这没问题。将来，您应该使用 AWS IAM 权限系统创建具有有限权限的用户。

接下来，让我们安装 AWS 命令行界面（CLI）：

```py
$ pip install awscli
```

然后，我们需要使用我们的密钥和区域配置 AWS 命令行工具。`aws`命令提供一个交互式`configure`子命令来执行此操作。让我们在命令行上运行它：

```py
$ aws configure
 AWS Access Key ID [None]: <Your ACCESS key>
 AWS Secret Access Key [None]: <Your secret key>
 Default region name [None]: us-west-2
 Default output format [None]: json
```

`aws configure`命令将存储您在家目录中的`.aws`目录中输入的值。

要确认您的新帐户是否设置正确，请请求 EC2 实例的列表（不应该有）：

```py
$ aws ec2 describe-instances
{
    "Reservations": []
}
```

# 创建文件上传存储桶

S3 被组织成存储桶。每个存储桶必须有一个唯一的名称（在整个 AWS 中唯一）。每个存储桶还将有一个控制访问的策略。

通过执行以下命令来创建我们的文件上传存储桶（将`BUCKET_NAME`更改为您自己的唯一名称）：

```py
$ export AWS_ACCESS_KEY=#your value
$ export AWS_SECRET_ACCESS_KEY=#yourvalue
$ aws s3 mb s3://BUCKET_NAME
```

为了让未经身份验证的用户访问我们存储桶中的文件，我们必须设置一个策略。让我们在`AWS/mymdb-bucket-policy.json`中创建策略：

```py
{
    "Version": "2012-10-17",
    "Id": "mymdb-bucket-policy",
    "Statement": [
        {
            "Sid": "allow-file-download-stmt",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::BUCKET_NAME/*"
        }
    ]
}
```

确保将`BUCKET_NAME`更新为您的存储桶的名称。

现在，我们可以使用 AWS CLI 在您的存储桶上应用策略：

```py
$ aws s3api put-bucket-policy --bucket BUCKET_NAME --policy "$(cat AWS/mymdb-bucket-policy.json)"
```

确保您记住您的存储桶名称，AWS 访问密钥和 AWS 秘密访问密钥，因为我们将在下一节中使用它们。

# 使用 Docker Compose

我们现在已经准备好生产部署的所有部分。 Docker Compose 是 Docker 让多个容器一起工作的方式。 Docker Compose 由一个命令行工具`docker-compose`，一个配置文件`docker-compose.yml`和一个环境变量文件`.env`组成。我们将在项目目录的根目录中创建这两个文件。

永远不要将您的`.env`文件检入版本控制。那里是您的秘密所在。不要让它们泄漏。

首先，让我们在`.env`中列出我们的环境变量：

```py
# Django settings
DJANGO_SETTINGS_MODULE=config.production_settings
DJANGO_SECRET_KEY=#put your secret key here
DJANGO_LOG_LEVEL=DEBUG
DJANGO_LOG_FILE=/var/log/mymdb/mymdb.log
DJANGO_ALLOWED_HOSTS=# put your domain here
DJANGO_DB_NAME=mymdb
DJANGO_DB_USER=mymdb
DJANGO_DB_PASSWORD=#put your password here
DJANGO_DB_HOST=db
DJANGO_DB_PORT=5432
DJANGO_CACHE_TIMEOUT=200

AWS_ACCESS_KEY_ID=# put aws key here
AWS_SECRET_ACCESS_KEY_ID=# put your secret key here
DJANGO_UPLOAD_S3_BUCKET=# put BUCKET_NAME here

# Postgres settings
POSTGRES_PASSWORD=# put your postgress admin password here
```

这些值中的许多值都可以硬编码，但有一些值需要为您的项目设置：

+   `DJANGO_SECRET_KEY`：Django 秘密密钥用作 Django 加密种子的一部分

+   `DJANGO_DB_PASSWORD`：这是 Django 的 MyMDB 数据库用户的密码

+   `AWS_ACCESS_KEY_ID`：您的 AWS 访问密钥

+   `AWS_SECRET_ACCESS_KEY_ID`：您的 AWS 秘密访问密钥

+   `DJANGO_UPLOAD_S3_BUCKET`：您的存储桶名称

+   `POSTGRES_PASSWORD`：Postgres 数据库超级用户的密码（与 MyMDB 数据库用户不同）

+   `DJANGO_ALLOWED_HOSTS`：我们将提供服务的域（一旦我们启动 EC2 实例，我们将填写这个）

接下来，我们在`docker-compose.yml`中定义我们的容器如何一起工作：

```py
version: '3'

services:
  db:
    build: docker/psql
    restart: always
    ports:
      - "5432:5432"
    environment:
      - DJANGO_DB_USER
      - DJANGO_DB_NAME
      - DJANGO_DB_PASSWORD
  web:
    build: .
    restart: always
    ports:
      - "80:80"
    depends_on:
      - db
    environment:
      - DJANGO_SETTINGS_MODULE
      - DJANGO_SECRET_KEY
      - DJANGO_LOG_LEVEL
      - DJANGO_LOG_FILE
      - DJANGO_ALLOWED_HOSTS
      - DJANGO_DB_NAME
      - DJANGO_DB_USER
      - DJANGO_DB_PASSWORD
      - DJANGO_DB_HOST
      - DJANGO_DB_PORT
      - DJANGO_CACHE_TIMEOUT
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY_ID
      - DJANGO_UPLOAD_S3_BUCKET
```

此 Compose 文件描述了构成 MyMDB 的两个服务（`db`和`web`）。让我们回顾一下我们使用的配置选项：

+   `build`：构建上下文的路径。一般来说，构建上下文是一个带有`Dockerfile`的目录。因此，`db`使用`psql`目录，`web`使用`.`目录（项目根目录，其中有一个`Dockerfile`）。

+   `ports`：端口映射列表，描述如何将主机端口上的连接路由到容器上的端口。在我们的情况下，我们不会更改任何端口。

+   `environment`：每个服务的环境变量。我们使用的格式意味着我们从我们的`.env`文件中获取值。但是，您也可以使用`MYVAR=123`语法硬编码值。

+   `restart`：这是容器的重启策略。`always`表示如果容器因任何原因停止，Docker 应该始终尝试重新启动容器。

+   `depends_on`：这告诉 Docker 在启动`web`容器之前启动`db`容器。然而，我们仍然不能确定 Postgres 是否能在 uWSGI 之前成功启动，因此我们需要在我们的 runit 脚本中检查数据库是否已经启动。

# 跟踪环境变量

我们的生产配置严重依赖于环境变量。让我们回顾一下在 Django 中使用`os.getenv()`之前必须遵循的步骤：

1.  在`.env`中列出变量

1.  在`docker-compose.yml`中的`environment`选项下包括变量

1.  在`env`中包括 uWSGI ini 文件变量

1.  使用`os.getenv`访问变量

# 在本地运行 Docker Compose

现在我们已经配置了我们的 Docker 容器和 Docker Compose，我们可以运行这些容器。Docker Compose 的一个优点是它可以在任何地方提供相同的环境。这意味着我们可以在本地运行 Docker Compose，并获得与我们在生产环境中获得的完全相同的环境。不必担心在不同环境中有额外的进程或不同的分发。让我们在本地运行 Docker Compose。

# 安装 Docker

要继续阅读本章的其余部分，您必须在您的机器上安装 Docker。Docker, Inc.提供免费的 Docker 社区版，可以从其网站上获得：[`docker.com`](https://docker.com)。Docker 社区版安装程序在 Windows 和 Mac 上是一个易于使用的向导。Docker, Inc.还为大多数主要的 Linux 发行版提供官方软件包。

安装完成后，您将能够按照接下来的所有步骤进行操作。

# 使用 Docker Compose

要在本地启动我们的容器，请运行以下命令：

```py
$ docker-compose up -d 
```

`docker-compose up`构建然后启动我们的容器。`-d`选项将 Compose 与我们的 shell 分离。

要检查我们的容器是否正在运行，我们可以使用`docker ps`：

```py
$ docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                          NAMES
0bd7f7203ea0        mymdb_web           "/sbin/my_init"          52 seconds ago      Up 51 seconds       0.0.0.0:80->80/tcp, 8031/tcp   mymdb_web_1
3b9ecdcf1031        mymdb_db            "docker-entrypoint..."   46 hours ago        Up 52 seconds       0.0.0.0:5432->5432/tcp         mymdb_db_1
```

要检查 Docker 日志，您可以使用`docker logs`命令来记录启动脚本的输出：

```py
$ docker logs mymdb_web_1
```

要访问容器内部的 shell（以便您可以检查文件或查看应用程序日志），请使用此`docker exec`命令启动 bash：

```py
$ docker exec -it mymdb_web_1 bash -l
```

要停止容器，请使用以下命令：

```py
$ docker-compose stop
```

要停止容器并*删除*它们，请使用以下命令：

```py
$ docker-compose down
```

当您删除一个容器时，您会删除其中的所有数据。对于 Django 容器来说这不是问题，因为它不保存数据。然而，如果您删除 db 容器，您将*丢失数据库的数据*。在生产环境中要小心。

# 通过容器注册表共享您的容器

现在我们有一个可工作的容器，我们可能希望使其更广泛地可访问。Docker 有一个容器注册表的概念。您可以将您的容器推送到容器注册表，以便将其公开或仅提供给您的团队。

最受欢迎的 Docker 容器注册表是 Docker Hub（[`hub.docker.com`](https://hub.docker.com)）。您可以免费创建一个帐户，并且在撰写本书时，每个帐户都附带一个免费的私有存储库和无限的公共存储库。大多数云提供商也提供 docker 存储库托管设施（尽管价格可能有所不同）。

本节的其余部分假设您已配置了主机。我们将以 Docker Hub 为例，但无论谁托管您的容器存储库，所有步骤都是相同的。

要共享您的容器，您需要做以下事情：

1.  登录到 Docker 注册表

1.  标记我们的容器

1.  推送到 Docker 注册表

让我们首先登录到 Docker 注册表：

```py
$ docker login -u USERNAME -p PASSWORD docker.io
```

`USERNAME` 和 `PASSWORD` 的值需要与您在 Docker Hub 帐户上使用的相同。 `docker.io` 是 Docker Hub 容器注册表的域。如果您使用不同的容器注册表主机，则需要更改域。

现在我们已经登录，让我们重新构建并标记我们的容器：

```py
$ docker build . -t USERNAME/REPOSITORY:latest
```

其中 `USERNAME` 和 `REPOSITORY` 的值将被替换为您的值。 `:latest` 后缀是构建的标签。我们可以在同一个存储库中有许多不同的标签（例如 `development`，`stable` 和 `1.x`）。Docker 中的标签很像版本控制中的标签；它们帮助我们快速轻松地找到特定的项目。 `:latest` 是给最新构建的常见标签（尽管它可能不稳定）。

最后，让我们将标记的构建推送到我们的存储库：

```py
$ docker push USERNAME/REPOSITORY:latest
```

Docker 将显示其上传的进度，然后在成功时显示 SHA256 摘要。

当我们将 Docker 镜像推送到远程存储库时，我们需要注意镜像中存储的任何私人数据。我们在 `Dockerfile` 中创建或添加的所有文件都包含在推送的镜像中。就像我们不希望在存储在远程存储库中的代码中硬编码密码一样，我们也不希望在可能存储在远程服务器上的 Docker 镜像中存储敏感数据（如密码）。这是我们强调将密码存储在环境变量而不是硬编码它们的另一个原因。

太好了！现在你可以与其他团队成员分享存储库，以运行你的 Docker 容器。

接下来，让我们启动我们的容器。

# 在云中的 Linux 服务器上启动容器

现在我们已经让一切运转起来，我们可以将其部署到互联网上。我们可以使用 Docker 将我们的容器部署到任何 Linux 服务器上。大多数使用 Docker 的人都在使用云提供商来提供 Linux 服务器主机。在我们的情况下，我们将使用 AWS。

在前面的部分中，当我们使用 `docker-compose` 时，实际上是在向运行在我们的机器上的 Docker 服务发送命令。Docker Machine 提供了一种管理运行 Docker 的远程服务器的方法。我们将使用 `docker-machine` 来启动一个 EC2 实例，该实例将托管我们的 Docker 容器。

启动 EC2 实例可能会产生费用。在撰写本书时，我们将使用符合 AWS 免费套餐资格的实例 `t2.micro`。但是，您有责任检查 AWS 免费套餐的条款。

# 启动 Docker EC2 VM

我们将在我们的帐户的**虚拟私有云**（**VPC**）中启动我们的 EC2 VM（称为 EC2 实例）。但是，每个帐户都有一个唯一的 VPC ID。要获取您的 VPC ID，请运行以下命令：

```py
$ export AWS_ACCESS_KEY=#your value
$ export AWS_SECRET_ACCESS_KEY=#yourvalue
$ export AWS_DEFAULT_REGION=us-west-2
$ aws ec2 describe-vpcs | grep VpcId
            "VpcId": "vpc-a1b2c3d4",
```

上述代码中使用的值不是真实值。

现在我们知道我们的 VPC ID，我们可以使用 `docker-machine` 来启动一个 EC2 实例：

```py
$ docker-machine create \
     --driver amazonec2 \
     --amazonec2-instance-type t2.micro \
     --amazonec2-vpc-id vpc-a1b2c3d4 \
     --amazonec2-region us-west-2 \
     mymdb-host
```

这告诉 Docker Machine 在`us-west-2`地区和提供的 VPC 中启动一个 EC2 `t2.micro`实例。Docker Machine 负责确保服务器上安装并启动了 Docker 守护程序。在 Docker Machine 中引用此 EC2 实例时，我们使用名称 `mymdb-host`。

当实例启动时，我们可以向 AWS 请求我们实例的公共 DNS 名称：

```py
$ aws ec2 describe-instances | grep -i publicDnsName
```

即使只有一个实例运行，上述命令可能会返回相同值的多个副本。将结果放入 `.env` 文件中作为 `DJANGO_ALLOWED_HOSTS`。

所有 EC2 实例都受其安全组确定的防火墙保护。Docker Machine 在启动我们的实例时自动为我们的服务器创建了一个安全组。为了使我们的 HTTP 请求到达我们的机器，我们需要在 `docker-machine` 安全组中打开端口 `80`，如下所示：

```py
$ aws ec2 authorize-security-group-ingress \
    --group-name docker-machine \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0
```

现在一切都设置好了，我们可以配置`docker-compose`与我们的远程服务器通信，并启动我们的容器：

```py
$ eval $(docker-machine env mymdb-host)
$ docker-compose up -d
```

恭喜！MyMDB 已经在生产环境中运行起来了。通过导航到`DJANGO_ALLOWED_HOSTS`中使用的地址来查看它。

这里的说明重点是启动 AWS Linux 服务器。然而，所有的 Docker 命令都有等效的选项适用于 Google Cloud、Azure 和其他主要的云服务提供商。甚至还有一个*通用*选项，可以与任何 Linux 服务器配合使用，尽管根据 Linux 发行版和 Docker 版本的不同，效果可能有所不同。

# 关闭 Docker EC2 虚拟机

Docker Machine 也可以用于停止运行 Docker 的虚拟机，如下面的代码片段所示：

```py
$ export AWS_ACCESS_KEY=#your value
$ export AWS_SECRET_ACCESS_KEY=#yourvalue
$ export AWS_DEFAULT_REGION=us-west-2
$ eval $(docker-machine env mymdb-host)
$ docker-machine stop mymdb-host 
```

这将停止 EC2 实例并销毁其中的所有容器。如果您希望保留您的数据库，请确保通过运行前面的`eval`命令来备份您的数据库，然后使用`docker exec -it mymdb_db_1 bash -l`打开一个 shell。

# 总结

在这一章中，我们已经将 MyMDB 部署到了互联网上的生产 Docker 环境中。我们使用 Dockerfile 为 MyMDB 创建了一个 Docker 容器。我们使用 Docker Compose 使 MyMDB 与 PostgreSQL 数据库（也在 Docker 容器中）配合工作。最后，我们使用 Docker Machine 在 AWS 云上启动了这些容器。

恭喜！你现在已经让 MyMDB 运行起来了。

在下一章中，我们将实现 Stack Overflow。
