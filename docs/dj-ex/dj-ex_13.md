# 第十三章：上线

上一章中，你为你的项目创建了 RESTful API。在本章中，你会学习以下知识点：

- 配置一个生产环境
- 创建一个自定义的中间件
- 实现自定义的管理命令

## 13.1 在生产环境上线

是时候把你的 Django 项目部署到生产环境了。我们将按以下步骤上线我们的项目：

1. 为生产环境配置项目设置。
2. 使用 PostgreSQL 数据库。
3. 使用`uWSGI`和`Ngnix`设置一个 web 服务器。
4. 为静态资源提供服务。
5. 用 SSL 保护我们的网站。

### 13.1.1 为多个环境管理设置

在实际项目中，你可能需要处理多个环境。你最少会有一个本地环境和一个生产环境，但是很可能还有别的环境。有些项目设置是所有环境通用的，有些可能需要被每个环境覆盖。让我们为多个环境配置项目设置，同时保持项目的良好组织。

在`educa`项目目录中创建`settings/`目录。把项目的`settings.py`文件移动到`settings/`目录中，并重命名为`base.py`，然后在新目录中创建以下文件结构：

```py
settings/
	__init__.py
	base.py
	local.py
	pro.py
```

这些文件分别是：

- `base.py`：包括通用和默认设置的基础设置文件
- `local.py`：你本地环境的自定义设置
- `pro.py`：生产环境的自定义设置

编辑`settings/base.py`文件，找到这一行代码：

```py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

替换为下面这一行代码：

```py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
```

我们已经把设置文件移动到了低一级的目录中，所以我们需要`BASE_DIR`正确的指向父目录。我们使用`os.pardir`指向父目录。

编辑`settings/local.py`文件，并添加以下代码：

```py
from .base import *

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
```

这是我们本地环境的设置文件。我们导入`base.py`文件中定义的所有设置，只为这个生产环境定义特定设置。我们从`base.py`文件中拷贝了`DEBUG`和`DATABASES`设置，因为每个环境会设置这些选项。你可以从`base.py`文件中移除`DEBUG`和`DATABASES`设置。

编辑`settings/pro.py`文件，并添加以下代码：

```py
from .base import *

DEBUG = False

ADMINS = {
    ('Antonio M', 'email@mydomain.com'),
}

ALLOWED_HOSTS = ['educaproject.com', 'www.educaproject.com']

DATABASES = {
    'default': {
        
    }
}
```

这些是生产环境的设置。它们分别是：

- `DEBUG`：设置`DEBUG`为`False`对任何生产环境都是强制的。不这么做会导致追踪信息和敏感的配置数据暴露给每一个人。
- `ADMINS`：当`DEBUG`为`False`，并且一个视图抛出异常时，所有信息会通过邮件发送给`ADMINS`设置中列出的所有人。确保用你自己的信息替换上面的`name/e-mail`元组。
- `ALLOWED_HOST`：因为`DEBUG`为`False`，Django 只允许这个列表中列出的主机为应用提供服务。这是一个安全措施。我们包括了`educaproject.com`和`www.educaproject.com`域名，我们的网站会使用这两个域名。
- `DATABASES`：我们保留这个设置为空。我们将在下面讨论生产环境的数据库设置。

> 处理多个环境时，创建一个基础的设置文件，并为每个环境创建一个设置文件。环境设置文件应用从通用设置继承，并覆写环境特定设置。

我们已经把项目设置从默认的`settings.py`文件放到了不同位置。除非你指定使用的设置模块，否则不能用`manage.py`工具执行任何命令。在终端执行管理命令时，你需要添加`--settings`标记，或者设置`DJANGO_SETTINGS_MODULE`环境变量。打开终端执行以下命令：

```py
export DJANGO_SETTINGS_MODULE=educa.settings.pro
```

这会为当前终端会话是设置`DJANGO_SETTINGS_MODULE`环境变量。如果你不想为每个新终端都执行这个命令，可以在`.bashrc`或`.bash_profile`文件中，把这个命令添加到你的终端配置。如果你不想设置这个变量，则必须使用`--settings`标记运行管理命令，比如：

```py
python manage.py migrate --settings=educa.settings.pro
```

现在你已经成功的为多个环境组织好了设置。

### 13.1.2 安装 PostgreSQL

在本书中，我们一直使用 SQLite 数据库。它的设置简单快捷，但对于生产环境，你需要一个更强大的数据库，比如 PostgreSQL，MySQL 或者 Oracle。我们将在生产环境使用 PostgreSQL。因为 PostgreSQL 提供的特性和性能，所以它是 Django 的推荐数据库。Django 还自带`django.contrib.postgres`包，允许你利用 PostgreSQL 的特定特性。你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/contrib/postgres/)阅读更多关于这个模块的信息。

如果你正在使用 Linux，使用以下命令安装 PostgreSQL 的依赖：

```py
sudo apt-get install libpq-dev python-dev
```

然后使用以下命令安装 PostgreSQL：

```py
sudo apt-get install postgresql postgresql-contrib
```

如果你正在使用 Mac OS X 或者 Windows，你可以在[这里](http://www.postgresql.org/download/)下载 PostgreSQL。

让我们创建一个 PostgreSQL 用户。打开终端，并执行以下命令：

```py
su postgres
createuser -dP educa
```

会提示你输入密码和给予这个用户的权限。输入密码和权限，然后使用以下命令创建一个新的数据库：

```py
createdb -E utf8 -U educa educa
```

然后编辑`settings/pro.py`文件，并修改`DATABASES`设置：

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'educa',
        'USER': 'educa',
        'PASSWORD': '****',
    }
}
```

用你创建的数据库名和用户凭证替换上面的数据。现在新数据库是空的。执行以下命令应用所有数据库迁移：

```py
python manage.py migrate
```

最后使用以下命令创建一个超级用户：

```py
python manage.py createsuperuser
```

### 13.1.3 检查你的项目

Django 包括`check`命令，可以在任何时候检查你的项目。这个命令检查 Django 项目中安装的应用，并输出所有错误或警告。如果你包括了`--deploy`选项，则只会触发生成环境相关的额外检查。打开终端，运行以下命令执行一次检查：

```py
python manage.py check --deploy
```

你会看到没有错误，但是有几个警告的输出。这意味着检查成功了，但你应该查看警告，看看是否可以做一些工作，让你的项目在生产环境上是安全的。我们不会深入其中，你需要记住，在生产环境中使用之前，你应该检查项目所有相关的问题。

## 13.2 通过 WSGI 为 Django 提供服务

Django 的主要部署平台是 WSGI。WSGI 是`Web Server Gateway Interface`的缩写，它是在网络上为 Python 应用提供服务的标准。

当你使用`startproject`命令创建一个新项目时，Django 会在项目目录中创建一个`wsgi.py`文件。这个文件包含一个 WSGI 应用的可调用对象，它是你应用的访问点。使用 Django 开发服务器运行你的项目，以及在生产环境中用你选择的服务器部署应用都会使用 WSGI。

你可以在[这里](http://wsgi.readthedocs.org/en/latest/)进一步学习 WSGI。

### 13.2.1 安装 uWSGI

在这本书中，你一直使用 Django 开发服务器在本地环境运行项目。但是你需要一个实际的 Web 服务器在生产环境部署你的应用。

`uWSGI`是一个非常快速的 Python 应用服务器。它使用 WSGI 规范与你的 Python 应用通信。`uWSGI`把 Web 请求转换为 Django 项目可用处理的格式。

使用以下命令安装`uWSGI`：

```py
pip install uwsgi
```

如果你正在使用 Mac OS X，你可以使用`brew install uwsgi`命令安装`uWSGI`。如果你想在 Windows 上安装`uWSGI`，则需要[Cygwin](https://www.cygwin.com/)。但是推荐你在基于 Unix 的环境中使用`uWSGI`。

### 13.2.2 配置 uWSGI

你可以从命令行中启动`uWSGI`。打开终端，在`educa`项目目录中执行以下命令：

```py
uwsgi --module=educa.wsgi:application \
--env=DJANGO_SETTINGS_MODULE=educa.settings.pro \
--http=127.0.0.1:80 \
--uid=1000 \
--virtualenv=/home/zenx/env/educa/
```

如果你没有权限，则需要在命令前加上`sudo`。

通过这个命令，我们用以下选项在本地运行`uWSGI`：

- 我们使用`educa.wsgi:application`作为 WSGI 的可调用对象。
- 我们为生产环境加载设置。
- 我们使用我们的虚拟环境。用你的实际虚拟环境目录替换`virtualenv`选项的路径。如果你没有使用虚拟环境，则跳过这个选项。

如果你没有在项目目录下运行命令，则需要用你的项目目录包括`--chdir=/path/to/educa/`选项。

在浏览器中打开`http://127.0.0.1:80/`。你会看到没有加载 CSS 或者图片的 HTML。这是有道理的，因为我们还没有配置`uWSGI`为静态文件提供服务。

`uWSGI`允许你在`.ini`文件中定义自定义配置。它比在命令行中传递选项更方便。在主`educa/`目录下创建以下文件结构：

```py
config/
	uwsgi.ini
```

编辑`uwsgi.ini`文件，添加以下代码：

```py
[uwsgi]
# variables
projectname = educa
base = /home/zenx/educa

# configuration
master = true
virtualenv = /home/zenx/env/%(projectname)
pythonpath = %(base)
chdir = %(base)
env = DJANGO_SETTINGS_MODULE=%(projectname).settings.pro
module = educa.wsgi:application
socket = /tmp/%(projectname).sock
```

我们定义了以下变量：

- `projectname`：你的 Django 项目名称，这里是`educa`。
- `base`：`educa`项目的绝对路径。用你的绝对路径替换它。

还有一些会在`uWSGI`选项中使用的自定义变量。你可以定义任意变量，只要它跟`uWSGI`选项名不同就行。我们设置了以下选项：

- `master`：启用主进程。
- `virtualenv`：你的虚拟环境路径。用响应的路径替换它。
- `pythonpath`：添加到 Python 路径的路径。
- `chdir`：项目目录的路径，加载应用之前，`uWSGI`改变到这个目录。
- `env`：环境变量。我们包括了`DJANGO_SETTINGS_MODULE`变量，指向生产环境的设置。
- `module`：使用的 WSGI 模块。我们把它设置为`application`可调用对象，它包含在项目的`wsgi`模块中。
- `socket`：绑定到服务器的 UNIX/TCP 套接字。

`socket`选项用于与第三方路由（比如 Nginx）通信，而`http`选项用于`uWGSI`接收传入的 HTTP 请求，并自己进行路由。因为我们将配置 Nginx 作为 Web 服务器，并通过套接字与`uWSGI`通信，所以我们将使用套接字运行`uWSGI`。

你可以在[这里](http://readthedocs.org/en/latest/Options.html)找到所有可用的`uWSGI`选项列表。

现在你可以使用自定义配置运行`uWSGI`：

```py
uwsgi --ini config/uwsgi.ini
```

因为`uWSGI`通过套接字运行，所以你现在不能在浏览器中访问`uWSGI`实例。让我们完成生产环境。

### 13.2.3 安装 Nginx

当你为一个网站提供服务时，你必须为动态内容提供服务，同时还需要为静态文件，比如 CSS，JavaScript 文件和图片提供服务。虽然`uWSGI`可以为静态文件提供服务，但它会在 HTTP 请求上增加不必要的开销。因此，推荐在`uWSGI`之前设置一个 Web 服务器（比如 Nginx），为静态文件提供服务。

Nginx 是一个专注于高并发，高性能和低内存使用的 Web 服务器。Nginx 还可以充当反向代理，接收 HTTP 请求，并把它们路由到不同的后台。通常情况下，你会在前端使用一个 Web 服务器（比如 Nginx），高效快速的为静态文件提供服务，并且把动态请求转发到`uWSGI`的工作线程。通过使用 Nginx，你还可以应用规则，并从它的反向代理功能中获益。

使用以下命令安装 Nginx：

```py
sudo apt-get install nginx
```

如果你正在使用 Mac OS X，你可以使用`brew install nginx`命令安装 Nginx。你可以在[这里](http://nginx.org/en/download.html)找到 Windows 的二进制版本。

### 13.2.4 生产环境

下图展示了我们最终的生产环境：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE13.1.png)

当客户端浏览器发起一个 HTTP 请求时，会发生以下事情：

1. Nginx 接收 HTTP 请求。
2. 如果请求的是静态文件，则 Nginx 直接为静态文件提供服务。如果请求的是动态页面，则 Nginx 通过套接字把请求转发给`uWSGI`。
3. `uWSGI`把请求传递给 Django 处理。返回的 HTTP 响应传递回 Nginx，然后再返回到客户端浏览器。

### 13.2.5 配置 Nginx

在`config/`目录中创建`nginx.conf`文件，并添加以下代码：

```py
# the upstream component nginx needs to connect to
upstream educa {
    server unix:///tmp/educa.sock;
}

server {
    listen 80;
    server_name www.educaproject.com educaproject.com;

    location / {
        include /etc/nginx/uwsgi_params;
        uwsgi_pass educa;
    }
}
```

这是 Nginx 的基础配置。我们设置了一个名为`educa`的上游（upstream），指向`uWSGI`创建的套接字。我们使用`server`指令，并添加以下配置：

1. 我们告诉 Nginx 监听 80 端口。
2. 我们设置服务名为`www.educaproject.com`和`educaproject.com`。Nginx 会为来自这两个域名的请求服务。
3. 最后，我们指定`/`路径下的所有请求都路由到`educa`套接字（`uWSGI`）。我们还包括了 Nginx 自带的默认`uWSGI`配置参数。

你可以在[这里](http://nginx.org/en/docs/)阅读 Nginx 文档。主要的 Nginx 配置文件位于`/etc/nginx/nginx.conf`。它包括了`/etc/nginx/sites-enabled/`下找到的所有配置文件。要让 Nginx 加载你的自定义配置文件，需要如下创建一个符号链接：

```py
sudo ln -s /home/zenx/educa/config/nginx.conf /etc/nginx/sites-enabled/educa.conf
```

用你项目的绝对路径替换`/home/zenx/educa/`。然后打开终端启动`uWSGI`：

```py
uwsgi --ini config/uwsgi.ini
```

打开第二个终端，用以下命令启动 Nginx：

```py
service nginx start
```

因为我们正在使用简单的主机名，所以需要把它重定向到本机。编辑你的`/etc/hosts`文件，添加下面两行：

```py
127.0.0.1 educaproject.com
127.0.0.1 www.educaproject.com
```

这样，我们把两个主机名路由到我们的本地服务器。在生产环境中你不需要这么用，因为你会在域名的 DNS 配置中把主机名指向你的服务器。

在浏览器中打开`http://educaproject.com`。你可以看到你的网站，仍然没有加载任何静态资源。我们的生产环境马上就好了。

### 13.2.6 为静态资源和多媒体资源提供服务

为了最好的性能，我们将直接使用 Nginx 为静态资源提供服务。

编辑`settings/base.py`文件，并添加以下代码：

```py
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static/')
```

我们需要用 Django 导出静态资源。`collectstatic`命令从所有应用中拷贝静态文件，并把它们存储到`STATIC_ROOT`目录中。打开终端执行以下命令：

```py
python manage.py collectstatic
```

你会看到以下输出：

```py
You have requested to collect static files at the destination location as specified in your settings:

    /educa/static
    
This will overwrite existing files!
Are you sure you want to do this?
```

输入`yes`让 Django 拷贝这些文件。你会看到以下输出：

```py
78 static files copied to /educa/static
```

现在编辑`config/nginx.conf`文件，并在`server`指令中添加以下代码：

```py
location /static/ {
    alias /home/zenx/educa/static/;
}
location /media/ {
    alias /home/zenx/educa/media/;
}
```

记得把`/home/zenx/educa/`路径替换为你项目目录的绝对路径。这些指令告诉 Nginx，直接在`/static/`或`/media/`路径下为静态资源提供服务。

使用以下命令重新加载 Nginx 的配置：

```py
server nginx reload
```

在浏览器中打开`http://educaproject.com/`。现在你可以看到静态文件了。我们成功的配置了 Nginx 来提供静态文件。

## 13.3 使用 SSL 保护链接

SSL 协议（Secure Sockets Layer）通过安全连接成为了为网站提供服务的标准。强烈鼓励你在 HTTPS 下为网站提供服务。我们将在 Nginx 中配置一个 SSL 证书，安全的为我们网站提供服务。

### 13.3.1 创建 SSL 证书

在`educa`项目目录中创建`ssl`目录。然后使用以下命令生成一个 SSL 证书：

```py
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ssl/educa.key -out ssl/educa.crt
```

我们生成一个私有 key 和一个有效期是 1 年的 2048 个字节的证书。你将被要求输入以下信息：

```py
Country Name (2 letter code) [AU]:
State or Province Name (full name) [Some-State]:
Locality Name (eg, city) []: Madrid
Organization Name (eg, company) [Internet Widgits Pty Ltd]: Zenx IT
Organizational Unit Name (eg, section) []:
Common Name (e.g. server FQDN or YOUR name) []: educaproject.com
Email Address []: email@domain.com
```

你可以用自己的信息填写要求的数据。最重要的字段是`Common Name`。你必须制定证书的域名。我们将使用`educaproject.com`。

这会在`ssl/`目录中生成`educa.key`的私有 key 和实际证书`educa.crt`文件。

### 13.3.2 配置 Nginx 使用 SSL

编辑`nginx.conf`文件，并修改`server`指令，让它包括以下 SSL 指令：

```py
server {
    listen 80;
    listen 443 ssl;
    ssl_certificate /home/zenx/educa/ssl/educa.crt;
    ssl_certificate_key /home/zenx/educa/ssl/educa.key;
    server_name www.educaproject.com educaproject.com;
    # ...
```

现在我们的服务器在 80 端口监听 HTTP，在 443 端口监听 HTTPS。我们用`ssl_certificate`制定 SSL 证书，用`ssl_certificate_key`制定证书 key。

使用以下命令重启 Nginx：

```py
sudo service nginx restart
```

Nginx 将会加载新的配置。在浏览器中打开`http://educaproject.com`。你会看到一个类似这样的静态消息：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE13.2.png)

不同的浏览器，这个消息可能会不同。它警告你，你的网站没有使用受信任的证书；浏览器不能验证网站的身份。这是因为我们签署了自己的证书，而不是从受信任的认证机构获得证书。当你拥有真正的域名时，你可以申请受信任的 CA 为其颁发 SSL 证书，以便浏览器可以验证其身份。

如果你想为真正域名获得受信任的证书，你可以参数 Linux Foundation 创建的`Let's Encrypt`项目。这是一个协作项目，目的是免费的简化获取和更新受信任的 SSL 证书。你可以在[这里](https://letsencrypt.org/)阅读更多信息。

点击`Add Exception`按钮，让浏览器知道你信任这个证书。你会看到浏览器在 URL 旁显示一个锁的图标，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE13.3.png)

如果你点击锁图标，则会显示 SSL 证书的详情。

### 13.3.3 为 SSL 配置我们的项目

Django 包括一些 SSL 的特定设置。编辑`settings/pro.py`设置文件，添加以下代码：

```py
SECURE_SSL_REDIRECT = True
CSRF_COOKIE_SECURE = True
```

这些设置分别是：

- `SECURE_SSL_REDIRECT`：HTTP 请求是否必须重定义到 HTTPS 请求
- `CSRF_COOKIE_SECURE`：这必须为跨站点请求保护设置为建立一个安全的 cookie

非常棒！你已经配置了一个生产环境，它会为你的项目提供高性能的服务。

## 13.4 创建自定义的中间件

你已经了解了`MIDDLEWARE`设置，其中包括项目的中间件。一个中间件是一个类，其中包括一些在全局执行的特定方法。你可以把它看成一个低级别的插件系统，允许你实现在请求或响应过程中执行的钩子。每个中间件负责会在所有请求或响应中执行的一些特定操作。

> 避免在中间件中添加开销昂贵的处理，因为它们为在每个请求中执行。

当收到一个 HTTP 请求时，中间件会以`MIDDLEWARE`设置中的出现顺序执行。当一个 HTTP 响应由 Django 生成时，中间件的方法会逆序执行。

下图展示了请求和响应阶段时，中间件方法的执行顺序。它还展示了可能被调用的中间件方法：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE13.4.png)

在请求阶段，会执行中间件的以下方法：

1. `process_request(request)`：在 Django 决定执行哪个视图之前，在每个请求上调用。`request`是一个`HttpRequest`实例。
2. `process_view(request, view_func, view_args, view_kwargs)`：在 Django 调用视图之前调用。它可以访问视图函数及其收到的参数

在响应阶段，会执行中间件的以下方法：

1. `process_exception(request, exception)`：只有视图函数抛出`Exception`异常时才会调用。
2. `process_template_response(request, response)`：视图执行完成后调用，只有当`response`对象有`render()`方法时才调用（比如它是`TemplateResponse`或者等价对象）
3. `process_response(request, response)`：响应返回到浏览器之前，在所有响应上调用。

因为中间件可以依赖之前已经执行的其它中间件方法在请求中设置的数据，所以`MIDDLEWARE`设置中的顺序很重要。请注意，即使因为前一个中间件返回了 HTTP 响应，导致`process_request()`和`process_view()`被跳过，中间件的`process_response()`方法也会被调用。这意味着`process_response()`不能依赖于请求阶段设置的数据。如果一个异常被某个中间件处理，并返回了一个响应，则之前的中间件类不会被调用。

> 当添加新的中间件到`MIDDLEWARE`设置中时，确保把它放在了正确的位置。在请求阶段，中间件方法按设置中的出现顺序执行，响应阶段则是逆序执行。

你可以在[这里](https://www.djangoproject.com/en/1.11/topics/http/middleware/)查看更多关于中间件的信息。

我们将创建一个自定义的中间件，允许通过自定义子域名访问课程。每个课程详情视图的 URL（比如`http://educaproject.com/courses/django/`）也可以通过子域名（用课程的`slug`字段构建）访问，比如`http://django.educaproject.com/`。

### 13.4.1 创建子域名中间件

中间件可以位于项目的任何地方。但是，推荐的方式是在应用目录中创建一个`middleware.py`文件。

在`courses`应用目录中创建`middleware.py`文件，并添加以下代码：

```py
from django.core.urlresolvers import reverse
from django.shortcuts import get_object_or_404, redirect
from .models import Course

class SubdomainCourseMiddleware:
    def process_request(self, request):
        host_parts = request.get_host().split('.')
        if len(host_parts) > 2 and host_parts[0] != 'www':
            # get course for the given subdomain
            course = get_object_or_404(Course, slug=host_parts[0])
            course_url = reverse('course_detail', args=[course.slug])
            # redirect current request to the course_detail view
            url = '{}://{}{}'.format(
                request.scheme,
                '.'.join(host_parts[1:]),
                course_url
            )
            return redirect(url)
```

我们创建了一个实现了`process_request()`的中间件。当收到 HTTP 请求时，我们执行以下任务：

1. 我们获得请求中使用的主机名，并把它拆分为多个部分。比如，如果用户访问的是`mycourse.educaproject.com`，则会生成`['mycourse', 'educaproject', 'com']`列表。
2. 通过检查拆分后是否生成两个以上的元素，我们核实包括子域名的主机名。如果主机名包括子域名，并且它不是`www`，则尝试使用子域名提供的`slug`获得课程。
3. 如果没有找到课程，我们抛出`Http 404`异常。否则，我们使用主域名重定向到课程详情的 URL。

编辑项目的`settings/base.py`文件，在`MIDDLEWARE`设置底部添加`courses.middleware.SubdomainCourseMiddleware`：

```py
MIDDLEWARE = [
    # ...
    'courses.middleware.SubdomainCourseMiddleware',
]
```

现在我们的中间件会在每个请求上执行。

### 13.4.2 使用 Nginx 为多个子域名服务

我们需要 Nginx 为带任意可能子域名的我们的网站提供服务。编辑`config/nginx.conf`文件，找到这一行代码：

```py
server_name www.educaproject.com educaproject.com;
```

替换为下面这一行代码：

```py
server_name *.educaproject.com educaproject.com;
```

通过使用星号，这条规则会应用与`educaproject.com`的所有子域名。为了在本地测试我们的中间件，我们需要在`/etc/hosts`中添加想要测试的子域名。要用别名为`django`的`Course`对象测试中间件，需要在`/etc/hosts`文件添加这一行：

```py
127.0.0.1 django.educaproject.com
```

然后在浏览器中打开`https://django.educaproject.com/`。中间件会通过子域名找到课程，并重定向到`https://educaproject.com/course/django/`。

## 13.5 实现自定义管理命令

Django 允许你的应用为`manage.py`工具注册自定义管理命令。例如，我们在第九章使用`makemessages`和`compilemessages`管理命令来创建和编译转换文件。

一个管理命令由一个 Python 模块组成，其中 Python 模块包括一个从`django.core.management.BaseCommand`继承的`Command`类。你可以创建简单命令，或者让它们接收位置和可选参数作为输入。

Django 在`INSTALLED_APPS`设置中激活的每个应用的`management/commands/`目录中查找管理命令。发现的每个模块注册为以其命名的管理命令。

你可以在[这里](http://djangoproject.com/en/1.11/howto/custom-management-commands/)进一步学习自定义管理命令。

我们将注册一个自定义管理命令，提供学生至少报名一个课程。该命令会给注册时间长于指定时间，但尚未报名任何课程的用户发送一封提醒邮件。

在`students`应用目录中创建以下文件结构：

```py
management/
	__init__.py
	commands/
		__init__.py
		enroll_reminder.py
```

编辑`enroll_reminder.py`文件，并添加以下代码：

```py
import datetime
from django.conf import settings
from django.core.management.base import BaseCommand
from django.core.mail import send_mass_mail
from django.contrib.auth.models import User
from django.db.models import Count

class Command(BaseCommand):
    help = 'Send an e-mail reminder to users registered more \
            than N days that are not enrolled into any courses yet'

    def add_arguments(self, parser):
        parser.add_argument('--days', dest='days', typy=int)

    def handle(self, *args, **kwargs):
        emails = []
        subject = 'Enroll in a course'
        date_joined = datetime.date.today() - datetime.timedelta(days=options['days'])
        users = User.objects.annotate(
            course_count=Count('courses_enrolled')
        ).filter(
            course_count=0, date_joined__lte=date_joined
        )
        for user in users:
            message = 'Dear {},\n\nWe noticed that you didn\'t enroll in any courses yet.'.format(user.first_name)
            emails.append((
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                [user.email]
            ))
        send_mass_mail(emails)
        self.stdout.write('Sent {} reminders' % len(emails))
```

这是我们的`enroll_reminder`命令。这段代码完成以下任务：

- `Command`类从`BaseCommand`继承。
- 我们包括了一个`help`属性。该属性为命令提供了一个简单描述，如果你执行`python manage.py help enroll_reminder`命令，则会打印这个描述。
- 我们使用`add_arguments()`方法添加`--days`命名参数。该参数用于指定用户注册了，但没有报名参加任何课程，从而需要接收提醒邮件的最小天数。
- `handle()`方法包括实际命令。我们从命令行解析中获得`days`属性。我们检索注册天数超过指定天数，当仍没有参加任何课程的用户。我们用一个用户报名参加的总课程数量注解（annotate）QuerySet 实现此目的。我们为每个用户生成一封提醒邮件，并把它添加到`emails`列表中。最后，我们用`send_mass_mail()`函数发送邮件，这个函数打开单个 SMTP 连接发送所有邮件，而不是每发送一封邮件打开一个连接。

你已经创建了第一个管理命令。打开终端执行你的命令：

```py
python manage.py enroll_reminder --days=20
```

如果你没有正在运行的本地 SMTP 服务器，你可以参考第二章，我们为第一个 Django 项目配置了 SMTP 设置。另外，你可以添加以下行到`settings/local.py`文件，让 Django 在开发期间输出邮件到标准输出：

```py
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
```

让我们调度管理命令，让服务器没有早上 8 点运行它。如果你正在使用基于 Unix 的系统，比如 Linux 或者 Mac OS X，打开终端执行`crontab -e`来编辑计划任务。在其中添加下面这一行：

```py
0 8 * * * python /path/to/educa/manage.py enroll_reminder --days=20 --settings=educa.settings.pro
```

如果你不熟悉`Cron`，你可以在[这里](https://en.wikipedia.org/wiki/Cron)学习它。

如果你正在使用 Windows，你可以使用`Task scheduler`调度任务。你可以在[这里](http://windows.microsoft.com/en-au/windows/schedule-task#1TC=windows-7)进一步学习它。

定期执行操作的另一个方法是用 Celery 创建和调度任务。记住，我们在第七章使用 Celery 执行了异步任务。除了使用`Cron`创建和调用管理命令，你还可以使用`Celery beat scheduler`创建异步任务并执行它们。你可以在[这里](http://celery.readthedocs.io/en/latest/userguide/periodic-tasks.html)进一步学习使用 Celery 调度定时任务。

> 对要使用`Cron`或者 Windows 调度任务控制面板调度的独立脚本使用管理命令。

Django 还包括一个用 Python 调用管理命令的工具。你可以在代码中如下执行管理命令：

```py
from django.core import management
management.call_command('enroll_reminder', days=20)
```

恭喜你！现在你已经为你的应用创建了自定义管理命令，并在需要时调度它们。

## 13.6 总结

在这一章中，你使用`uWSGI`和`Nginx`配置了一个生产环境。你还实现了一个自定义中间件，并学习了如何创建自定义管理命令。