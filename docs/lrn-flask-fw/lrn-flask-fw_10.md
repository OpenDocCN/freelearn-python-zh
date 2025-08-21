# 第十章：部署您的应用程序

在本章中，我们将学习如何以安全和自动化的可重复方式部署我们的 Flask 应用程序。我们将看到如何配置常用的**WSGI**（**Web 服务器网关接口**）能力服务器，如 Apache、Nginx，以及 Python Web 服务器 Gunicorn。然后，我们将看到如何使用 SSL 保护部分或整个站点，最后将我们的应用程序包装在配置管理工具中，以自动化我们的部署。

在本章中，我们将学习以下主题：

+   配置常用的 WSGI 服务器

+   高效地提供静态文件

+   使用 SSL 保护您的网站

+   使用 Ansible 自动化部署

# 使用 WSGI 服务器运行 Flask

重要的是要注意，Flask 本身并不是一个 Web 服务器。Web 服务器是面向互联网的工具，经过多年的开发和修补，并且可以同时运行多个服务。

在互联网上仅运行 Flask 作为 Web 服务器可能会很好，这要归功于 Werkzeug WSGI 层。然而，Flask 在页面路由和渲染系统上的真正重点是开发。作为 Web 服务器运行 Flask 可能会产生意想不到的影响。理想情况下，Flask 将位于 Web 服务器后面，并在服务器识别到对您的应用程序的请求时被调用。为此，Web 服务器和 Flask 需要能够使用相同的语言进行通信。

幸运的是，Flask 构建在 Werkzeug 堆栈之上，该堆栈旨在使用 WSGI 协议。WSGI 是一个常见的协议，被诸如 Apache 的 httpd 和 Nginx 之类的 Web 服务器使用。它可以用来管理 Flask 应用程序的负载，并以 Python 可以理解的方式传达关于请求来源和请求头的重要信息。

然而，要让 Werkzeug 使用 WSGI 协议与您的 Web 服务器通信，我们必须使用一个网关。这将接收来自您的 Web 服务器和 Python 应用程序的请求，并在它们之间进行转换。大多数 Web 服务器都会使用 WSGI，尽管有些需要一个模块，有些需要一个单独的网关，如 uWSGI。

首先要做的一件事是为 WSGI 网关创建一个 WSGI 文件以进行通信。这只是一个具有已知结构的 Python 文件，以便 WSGI 网关可以访问它。我们需要在与您的博客应用程序的其余部分相同的目录中创建一个名为`wsgi.py`的文件，它将包含：

```py
from app import app as application
```

Flask 默认是与 WSGI 兼容的，因此我们只需要以正确的方式声明对象，以便 WSGI 网关理解。现在，Web 服务器需要配置以找到此文件。

## Apache 的 httpd

Apache 的 httpd 目前可能是互联网上使用最广泛的 Web 服务器。该程序的名称实际上是 httpd，并由 Apache 软件基金会维护。然而，大多数人都将其称为*Apache*，因此我们也将称其为*Apache*。

要确保在基于 Debian 和 Ubuntu 的系统上安装了 Apache 和 WSGI 模块，请运行以下命令：

```py
sudo apt-get install apache2 libapache2-mod-wsgi

```

但是，在基于 Red Hat 和 Fedora 的系统上运行以下命令：

```py
sudo yum install httpd mod_wsgi

```

要设置 Apache 配置，我们必须创建一个指定新 VirtualHost 的配置文件。您必须找到系统上存放这些文件的目录。在基于 Debian 的系统（如 Ubuntu）中，这将在`/etc/apache2/sites-available`中；在基于 Red Hat/Fedora 的系统中，我们需要在`/etc/apache2/conf.d`目录中创建一个名为`blog.conf`的文件。

在该配置文件中，使用以下代码更新内容：

```py
<VirtualHost *:80>

    WSGIScriptAlias / <path to app>/wsgi.py

    <Directory <path to app>/>
        Order deny,allow
        Allow from all
    </Directory>

</VirtualHost>
```

此配置指示 Apache，对于对端口`80`上主机的每个请求，都要尝试从`wsgi.py`脚本加载。目录部分告诉 Apache 如何处理对该目录的请求，并且默认情况下，最好拒绝任何访问 Web 服务器的人对源目录中的文件的访问。请注意，在这种情况下，`<path to app>`是存储`wsgi.py`文件的目录的完整绝对路径。

现在我们需要为 Apache 的 httpd 服务器启用 WSGI 模块。这样 Apache 就知道在指定 WSGI 配置时要使用它。在基于 Debian 和 Ubuntu 的系统中，我们只需运行此命令：

```py
sudo a2enmod wsgi

```

然而，在 Red Hat 和 CentOS 系统上，情况会复杂一些。我们需要创建或修改文件`/etc/httpd/conf.d/wsgi.conf`，并包含以下行：

```py
LoadModule wsgi_module modules/mod_wsgi.so
```

现在我们需要通过运行以下命令在基于 Debian 和 Ubuntu 的系统上启用我们的新站点：

```py
sudo a2ensite blog

```

这指示 Apache 在`/etc/apache2/sites-available`和`/etc/apache2/sites-enabled`之间创建符号链接，Apache 实际上从中获取其配置。现在我们需要重新启动 Apache。在您的特定环境或分发中，可以以许多方式执行此操作。最简单的方法可能只是运行以下命令：

```py
sudo service apache2 restart

```

所以我们需要做的就是通过浏览器连接到 Web 服务器，访问`http://localhost/`。

在 Debian 和 Ubuntu 系统的`/var/log/apache2/error.log`和基于 Red Hat 和 CentOS 的系统的`/var/log/httpd/error_log`中检查是否有任何问题。

请注意，一些 Linux 发行版默认配置必须禁用。这可能可以通过在 Debian 和 Ubuntu 系统中输入以下命令来禁用：

```py
sudo a2dissite default

```

然而，在基于 Red Hat 和 CentOS 的系统中，我们需要删除`/etc/httpd/conf.d/welcome.conf`文件：

```py
sudo rm /etc/httpd/conf.d/welcome.conf

```

当然，我们需要再次重启 Debian 和 Ubuntu 系统的服务器：

```py
sudo service apache2 restart

```

在基于 Red Hat 和 CentOS 的系统中：

```py
sudo service httpd restart

```

Apache 还有一个重新加载选项，而不是重新启动。这告诉服务器再次查看配置文件并与其一起工作。这通常比重新启动更快，并且可以保持现有连接打开。而重新启动会退出服务器并重新启动，带走打开的连接。重新启动的好处是更明确，对于设置目的更一致。

### 提供静态文件

在使用 Flask 时，通过 Web 服务器，非常重要的一步是通过为站点的静态内容创建一个快捷方式来减少应用程序的负载。这将把相对琐碎的任务交给 Web 服务器，使得处理过程更快速、更响应。这也是一件简单的事情。

编辑您的`blog.conf`文件，在`<VirtualHost *:80>`标签内添加以下行：

```py
Alias /static <path to app>/static
```

在这里，`<path to app>`是静态目录存在的完整绝对路径。然后按照以下步骤重新加载 Debian 和 Ubuntu 系统的 Apache 配置：

```py
sudo service apache2 restart

```

对于基于 Red Hat 和 CentOS 的系统如下：

```py
sudo service httpd restart

```

这将告诉 Apache 在浏览器请求`/static`时在何处查找文件。您可以通过查看 Apache 日志文件来看到这一点，在 Debian 和 Ubuntu 系统中为`/var/log/apache2/access.log`，在基于 Red Hat 和 CentOS 的系统中为`/var/log/httpd/access.log`。

## Nginx

Nginx 正迅速成为取代 Apache 的 httpd 的事实标准 Web 服务器。它被证明更快，更轻量级，尽管配置有所不同，但更容易理解。

尽管 Nginx 已经支持 WSGI 有一段时间了，但即使是更新的 Linux 发行版也可能没有更新到它，因此我们必须使用一个称为 **uWSGI** 的接口层来访问 Python web 应用程序。uWSGI 是一个用 Python 编写的 WSGI 网关，可以通过套接字在 WSGI 和您的 Web 服务器之间进行翻译。我们需要安装 Nginx 和 uWSGI。在基于 Debian 和 Ubuntu 的系统中运行以下命令：

```py
sudo apt-get install nginx

```

在基于 Red Hat 或 Fedora 的系统中，以下

```py
sudo yum install nginx

```

现在由于 uWSGI 是一个 Python 模块，我们可以使用 `pip` 安装它：

```py
sudo pip install uwsgi

```

要在基于 Debian 和 Ubuntu 的系统中配置 Nginx，需要在 `/etc/nginx/sites-available` 中创建一个名为 `blog.conf` 的文件，或者在基于 Red Hat 或 Fedora 的系统中，在 `/etc/nginx/conf.d` 中创建文件，并添加以下内容：

```py
server {
    listen      80;
    server_name _;

    location / { try_files $uri @blogapp; }
    location @blogapp {
        include uwsgi_params;
        uwsgi_pass unix:/var/run/blog.wsgi.sock;
    }
}
```

这个配置与 Apache 配置非常相似，尽管是以 Nginx 形式表达的。它在端口 `80` 上接受连接，并且对于任何服务器名称，它都会尝试访问 `blog.wsgi.sock`，这是一个用于与 uWSGI 通信的 Unix 套接字文件。您会注意到 `@blogapp` 被用作指向位置的快捷方式引用。

只有在基于 Debian 和 Ubuntu 的系统中，我们现在需要通过从可用站点创建符号链接到已启用站点来启用新站点：

```py
sudo ln -s /etc/nginx/sites-available/blog.conf /etc/nginx/sites-enabled

```

然后我们需要告诉 uWSGI 在哪里找到套接字文件，以便它可以与 Nginx 通信。为此，我们需要在 `blog app` 目录中创建一个名为 `uwsgi.ini` 的 uWSGI 配置文件，其中包含以下内容：

```py
[uwsgi]
base = <path to app>
app = app
module = app
socket = /var/run/blog.wsgi.sock

```

您将需要将 `<path to app>` 更改为您的 `app.py` 文件存在的路径。还要注意套接字是如何设置在与 Nginx 站点配置文件中指定的相同路径中的。

### 注意

您可能会注意到 INI 文件的格式和结构非常类似于 Windows 的 INI 文件。

我们可以通过运行以下命令来验证此配置是否有效：

```py
uwsgi –ini uwsgi.ini

```

现在 Nginx 知道如何与网关通信，但还没有使用站点配置文件；我们需要重新启动它。在您特定的环境中可以通过多种方式执行此操作。最简单的方法可能就是运行以下命令：

```py
sudo service nginx restart

```

所以我们需要做的就是通过浏览器连接到 Web 服务器，访问 `http://localhost/`。

请注意，一些 Linux 发行版附带了必须禁用的默认配置。在基于 Debian 和 Ubuntu 的系统以及基于 Red Hat 和 CentOS 的系统中，通常可以通过删除 `/etc/nginx/conf.d/default.conf` 文件来完成此操作。

```py
sudo rm /etc/nginx/conf.d/default.conf

```

并重新启动 `nginx` 服务：

```py
sudo service nginx restart

```

### 注意

Nginx 还有一个重新加载选项，而不是重新启动。这告诉服务器再次查看配置文件并与其一起工作。这通常比重新启动更快，并且可以保持现有的连接打开。而重新启动会退出服务器并重新启动，带走打开的连接。重新启动的好处在于它更加明确，并且对于设置目的更加一致。

### 提供静态文件

在使用 Flask 通过 Web 服务器时，非常重要的一步是通过为站点上的静态内容创建一个快捷方式，以减轻应用程序的负载。这将使 Web 服务器从相对琐碎的任务中解脱出来，使得向最终浏览器提供基本文件的过程更快速、更响应。这也是一个简单的任务。

编辑您的 `blog.conf` 文件，在 server `{` 标签内添加以下行：

```py
location /static {
    root <path to app>/static;
}
```

其中 `<path to app>` 是静态目录存在的完整绝对路径。重新加载 Nginx 配置：

```py
sudo service nginx restart

```

这将告诉 Nginx 在浏览器请求 `/static` 时在哪里查找文件。您可以通过查看 Nginx 日志文件 `/var/log/nginx/access.log` 来看到这一点。

## Gunicorn

Gunicorn 是一个用 Python 编写的 Web 服务器。它已经理解了 WSGI，Flask 也是如此，因此让 Gunicorn 运行起来就像输入以下代码一样简单：

```py
pip install gunicorn
gunicorn app:app

```

其中`app:app`是您的应用程序，模块名称是我们在其中使用的（与 uWSGI 配置基本相同）。除此之外还有更多选项，但例如，从中工作并设置端口和绑定是有用的：

```py
gunicorn --bind 127.0.0.1:8000 app:app

```

`--bind`标志告诉 Gunicorn 要连接到哪个接口以及在哪个端口。如果我们只需要在内部使用 Web 应用程序，这是有用的。

另一个有用的标志是`--daemon`标志，它告诉 Gunicorn 在后台运行并与您的 shell 分离。这意味着我们不再直接控制该进程，但它正在运行，并且可以通过设置的绑定接口和端口进行访问。

# 使用 SSL 保护您的网站

在一个日益残酷的互联网上，通过证明其真实性来提高网站的安全性是很重要的。改善网站安全性的常用工具是使用 SSL，甚至更好的是 TLS。

SSL 和 TLS 证书允许您的服务器通过受信任的第三方基于您的浏览器连接的域名进行验证。这意味着，作为网站用户，我们可以确保我们正在交谈的网站在传输过程中没有被更改，是我们正在交谈的正确服务器，并且在服务器和我们的浏览器之间发送的数据不能被嗅探。当我们想要验证用户发送给我们的信息是否有效和受保护时，这显然变得重要，而我们的用户希望知道我们的数据在传输过程中受到保护。

## 获取您的证书

首先要做的是生成您的 SSL 证书请求。这与第三方一起使用，该第三方签署请求以验证您的服务器与任何浏览器。有几种方法可以做到这一点，取决于您的系统，但最简单的方法是运行以下命令：

```py
openssl req -nodes -newkey rsa:2048 -sha256 -keyout private.key -out public.csr

```

现在将询问您有关您所属组织的一些问题，但重要的是通用名称。这是您的服务器将被访问的域名（不带`https://`）：

```py
Country Name (2 letter code) [AU]: GB
State or Province Name (full name) [Some-State]: London
Locality Name (eg, city) []: London
Organization Name (eg, company) [Internet Widgits Pty Ltd]: Example Company
Organizational Unit Name (eg, section) []: IT
Common Name (eg, YOUR name) []: blog.example.com
Email Address []:
A challenge password []:
An optional company name []:
```

在这里，您可以看到我们使用`blog.example.com`作为我们示例域名，我们的博客应用将在该域名下访问。您必须在这里使用您自己的域名。电子邮件地址和密码并不是非常重要的，可以留空，但您应该填写“组织名称”字段，因为这将是您的 SSL 证书被识别为的名称。如果您不是一家公司，只需使用您自己的名字。

该命令为我们生成了两个文件；一个是`private.key`文件，这是我们的服务器用来与浏览器签署通信的文件，另一个是`public.csr`，这是发送给处理服务器和浏览器之间验证的第三方服务的证书请求文件。

### 注意

公钥/私钥加密是一个广泛但深入研究的主题。鉴于 Heartbleed 攻击，如果您希望保护服务器，了解这个是值得的。

下一步是使用第三方签署您的`public.csr`请求。有许多服务可以为您执行此操作，有些免费，有些略有成本；例如**Let's Encrypt**等一些服务可以完全免费地自动化整个过程。它们都提供基本相同的服务，但它们可能不会全部内置到所有浏览器中，并且为不同成本的不同程度的支持提供不同程度的支持。

这些服务将与您进行验证过程，要求您的`public.csr`证书请求，并为您的主机名返回一个已签名的`.crt`证书文件。

### 注意

请注意，将您的`.crt`和`.key`文件命名为其中申请证书的站点主机名可能会对您有所帮助。在我们的情况下，这将是`blog.example.com.crt`。

您的新`.crt`文件和现有的`.key`文件可以放在服务器的任何位置。但是，通常`.crt`文件放在`/etc/ssl/certs`中，而`.key`文件放在`/etc/ssl/private`中。

所有正确的文件都放在正确的位置后，我们需要重新打开用于我们的博客服务的现有 Apache 配置。最好运行一个正常的 HTTP 和 HTTPS 服务。但是，由于我们已经努力设置了 HTTPS 服务，强制执行它以重定向我们的用户是有意义的。这可以通过一个称为 HSTS 的新规范来实现，但并非所有的 Web 服务器构建都支持这一点，所以我们将使用重定向。

### 提示

您可以通过向操作系统的主机文件添加一个条目来在本地机器上运行带有 SSL 证书的测试域。只是不要忘记在完成后将其删除。

## Apache httpd

首先要更改的是`VirtualHost`行上的端口，从默认的 HTTP 端口`80`更改为默认的 HTTPS 端口`443`：

```py
<VirtualHost *:443>
```

我们还应该指定服务器的主机名正在使用的 SSL 证书；因此，在 VirtualHost 部分添加一个`ServerName`参数。这将确保证书不会在错误的域中使用。

```py
ServerName blog.example.com
```

您必须用您将要使用的主机名替换`blog.example.com`。

我们还需要设置 SSL 配置，以告诉 Apache 如何响应：

```py
SSLEngine on
SSLProtocol -all +TLSv1 +SSLv2
SSLCertificateFile /etc/ssl/certs/blog.example.com.crt
SSLCertificateKeyFile /etc/ssl/private/blog.example.com.key
SSLVerifyClient None
```

这里的情况是，Apache 中的 SSL 模块被启用，为该站点指定了公共证书和私钥文件，并且不需要客户端证书。禁用默认的 SSL 协议并启用 TLS 非常重要，因为 TLS 被认为比 SSL 更安全。但是，仍然启用 SSLv2 以支持旧版浏览器。

现在我们需要测试它。让我们重新启动 Apache：

```py
sudo service apache2 restart

```

尝试使用浏览器连接到 Web 服务器，不要忘记您现在正在使用`https://`。

现在它正在工作，最后一步是将普通的 HTTP 重定向到 HTTPS。在配置文件中，再次添加以下内容：

```py
<VirtualHost *:80>
  ServerName blog.example.com
  RewriteEngine On
  RewriteRule (.*) https://%{HTTP_HOST}%{REQUEST_URI}
</VirtualHost>
```

我们为端口`80`创建一个新的`VirtualHost`，并指定它是为`ServerName blog.example.com`主机名而设的。然后我们使用 Apache 中的`Rewrite`模块简单地将浏览器重定向到相同的 URL，但是在开头使用 HTTPS。

再次重启 Apache：

```py
sudo service apache2 restart

```

现在在网站上用浏览器测试这个配置；验证您被重定向到 HTTPS，无论您访问哪个页面。

## Nginx

Nginx 的配置非常简单。与 Apache 配置非常相似，我们需要更改 Nginx 将监听我们站点的端口。由于 HTTPS 在端口`443`上运行，这里的区别在于告诉 Nginx 期望 SSL 连接。在配置中，我们必须更新以下行：

```py
listen   443 ssl;
```

现在要将 SSL 配置添加到配置的服务器元素中，输入以下内容：

```py
server_name blog.example.com;
ssl_certificate /etc/ssl/certs/blog.example.com.crt;
ssl_certificate_key /etc/ssl/private/blog.example.com.key;
ssl_protocols TLSv1 SSLv2;
```

这告诉 Nginx 将此配置应用于对`blog.example.com`主机名的请求（不要忘记用您自己的替换它），因为我们不希望为不适用的域发送 SSL 证书。我们还指定了公共证书文件位置和文件系统上的私有 SSL 密钥文件位置。最后，我们指定了要使用的 SSL 协议，这意味着启用 TLS（被认为比 SSL 更安全）。但是 SSLv2 仍然启用以支持旧版浏览器。

现在来测试它。让我们重新启动 Nginx 服务：

```py
sudo service nginx restart

```

尝试使用浏览器连接到 Web 服务器，不要忘记您现在正在使用`https://`。

一旦我们证明它正在工作，最后一步是将普通的 HTTP 重定向到 HTTPS。再次在配置文件中添加以下内容：

```py
server {
    listen 80;
    server_name blog.example.com;
    rewrite ^ https://$server_name$request_uri? permanent;
}
```

这与以前的普通 HTTP 配置基本相同；只是我们使用`rewrite`命令告诉 Nginx 捕获所有 URL，并向访问 HTTP 端口的浏览器发送重定向命令，以转到 HTTPS，使用他们在 HTTP 上尝试使用的确切路径。

最后一次，重新启动 Nginx：

```py
sudo service nginx restart

```

最后，在您被重定向到 HTTPS 的网站上测试您的浏览器，无论您访问哪个页面。

## Gunicorn

从 0.17 版本开始，Gunicorn 也添加了 SSL 支持。要从命令行启用 SSL，我们需要一些标志：

```py
gunicorn --bind 0.0.0.0:443 --certfile /etc/ssl/certs/blog.example.com.crt --keyfile /etc/ssl/private/blog.example.com.key --ssl-version 2 --ciphers TLSv1  app:app

```

这与 Nginx 和 Apache SSL 配置的工作方式非常相似。它指定要绑定的端口，以及在这种情况下的所有接口。然后，它将 Gunicorn 指向公共证书和私钥文件，并选择在旧版浏览器中使用 SSLv2 和（通常被认为更安全的）TLS 密码协议。

通过在浏览器中输入主机名和 HTTPS 来测试这个。

现在准备好了，让我们将端口`80`重定向到端口`443`。这在 Gunicorn 中相当复杂，因为它没有内置的重定向功能。一个解决方案是创建一个非常简单的 Flask 应用程序，在 Gunicorn 上的端口`80`启动，并重定向到端口`443`。这将是一个新的应用程序，带有一个新的`app.py`文件，其内容如下：

```py
from flask import Flask,request, redirect
import urlparse

app = Flask(__name__)

@app.route('/')
@app.route('/<path:path>')
def https_redirect(path='/'):
    url = urlparse.urlunparse((
        'https',
        request.headers.get('Host'),
        path,
        '','',''
    ))

    return redirect(url, code=301)
if __name__ == '__main__':
    app.run()
```

这是一个非常简单的 Flask 应用程序，可以在任何地方使用，将浏览器重定向到等效的 URL，但在前面加上 HTTPS。它通过使用标准的 Python `urlparse`库，使用浏览器发送到服务器的标头中的请求主机名，以及路由中的通用路径变量来构建 URL。然后，它使用 Flask 的`redirect`方法告诉浏览器它真正需要去哪里。

### 注意

请注意，空字符串对于 urlunparse 函数很重要，因为它期望一个完整的 URL 元组，就像由 urlparse 生成的那样。

您现在可能已经知道如何在 Gunicorn 中运行这个，尽管如此，要使用的命令如下：

```py
gunicorn --bind 0.0.0.0:80 app:app

```

现在使用浏览器连接到旧的 HTTP 主机，您应该被重定向到 HTTPS 版本。

# 使用 Ansible 自动化部署

Ansible 是一个配置管理工具。它允许我们以可重复和可管理的方式自动化部署我们的应用程序，而无需每次考虑如何部署我们的应用程序。

Ansible 可以在本地和通过 SSH 工作。您可以使用 Ansible 的一个聪明之处是让 Ansible 配置自身。根据您自己的配置，然后可以告诉它部署它需要的其他机器。

然而，我们只需要专注于使用 Apache、WSGI 和 Flask 构建我们自己的本地 Flask 实例。

首先要做的是在我们要部署 Flask 应用的机器上安装 Ansible。由于 Ansible 是用 Python 编写的，我们可以通过使用`pip`来实现这一点：

```py
sudo pip install ansible

```

现在我们有了一个配置管理器，既然配置管理器是用来设置服务器的，让我们建立一个 playbook，Ansible 可以用来构建整个机器。

在一个新项目或目录中，创建一个名为`blog.yml`的文件。我们正在创建一个 Ansible 称为 Playbook 的文件；它是一个按顺序运行的命令列表，并构建我们在 Apache 下运行的博客。为简单起见，在这个文件中假定您使用的是一个 Ubuntu 衍生操作系统：

```py
---

- hosts: webservers
  user: ubuntu
  sudo: True

  vars:
    app_src: ../blog
    app_dest: /srv/blog

  tasks:
    - name: install necessary packages
      action: apt pkg=$item state=installed
      with_items:
        - apache2
        - libapache2-mod-wsgi
        - python-setuptools
    - name: Enable wsgi module for Apache
      action: command a2enmod wsgi
    - name: Blog app configuration for Apache
      action: template src=templates/blog dest=/etc/apache/sites-available/blog
    - name: Copy blog app in
      action: copy src=${app_src} dest=${app_dest}
    - name: Enable site
 action: command a2ensite blog
    - name: Reload Apache
      action: service name=apache2 state=reloaded
```

Ansible Playbook 是一个 YAML 文件，包含几个部分；主要部分描述了“play”。`hosts`值描述了后续设置应该应用于哪组机器。`user`描述了 play 应该以什么用户身份运行；对于您来说，这应该是 Ansible 可以运行以安装您的应用程序的用户。`sudo`设置告诉 Ansible 以`sudo`权限运行此 play，而不是以 root 身份运行。

`vars`部分描述了 playbook 中常见的变量。这些设置很容易找到，因为它们位于顶部，但也可以在 playbook 配置中以`${example_variable}`的格式稍后使用，如果`example_variable`在这里的`vars`部分中定义。这里最重要的变量是`app_src`变量，它告诉 Ansible 在将应用程序复制到正确位置时在哪里找到我们的应用程序。在这个例子中，我们假设它在一个名为`blog`的目录中，但对于您来说，它可能位于文件系统的其他位置，您可能需要更新此变量。

最后一个最重要的部分是`tasks`部分。这告诉 Ansible 在更新它控制的机器时要运行什么。如果您熟悉 Ubuntu，这些任务应该有些熟悉。例如，`action: apt`告诉 apt 确保`with_items`列表中指定的所有软件包都已安装。您将注意到`$item`变量与`pkg`参数。`$item`变量由 Ansible 自动填充，因为它在`with_items`命令和`apt`命令上进行迭代，`apt`命令使用`pkg`参数来验证软件包是否已安装。

随后的任务使用命令行命令`a2enmod wsgi`启用 WSGI 模块，这是 Debian 系统中启用模块的简写，通过填充模板设置我们博客站点的 Apache 配置。幸运的是，Ansible 用于模板的语言是 Jinja，您很可能已经熟悉。我们的模板文件的内容应该与此`blog.yml`相关，在一个名为`templates`的目录中，一个名为`blog`的文件。内容应该如下所示：

```py
NameVirtualHost *:80

<VirtualHost *:80>
    WSGIScriptAlias / {{ app_dest }}/wsgi.py

    <Directory {{ app_dest }}/>
        Order deny,allow
        Allow from all
    </Directory>
</VirtualHost>
```

这应该很熟悉，这是 Apache 部分示例的直接剽窃；但是，我们已经利用了 Ansible 变量来填充博客应用程序的位置。这意味着，如果我们想将应用程序安装到另一个位置，只需更新`app_dest`变量即可。

最后，在 Playbook 任务中，它将我们非常重要的博客应用程序复制到机器上，使用 Debian 简写在 Apache 中启用站点，并重新加载 Apache，以便可以使用该站点。

所以剩下的就是在那台机器上运行 Ansible，并让它为您构建系统。

```py
ansible-playbook blog.yml --connection=local

```

这告诉 Ansible 运行我们之前创建的 Playbook 文件`blog.yml`，并在`local`连接类型上使用它，这意味着应用于本地机器。

### 提示

**Ansible 提示**

值得注意的是，这可能不是在大型分布式环境中使用 Ansible 的最佳方式。首先，您可能希望将其应用于远程机器，或者将 Apache 配置、Apache WSGI 配置、Flask 应用程序配置和博客配置分开成 Ansible 称为角色的单独文件；这将使它们可重用。

另一个有用的提示是指定使用的配置文件并在 Apache 中设置静态目录。阅读 Ansible 文档，了解更多有关改进部署的方法的想法：

[`docs.ansible.com/`](http://docs.ansible.com/)

# 阅读更多

有关如何在 Apache 和 WSGI 中更有效地保护您的 Flask 部署，通过创建只能运行 Flask 应用程序的无 shell 用户，详细信息请参见[`www.subdimension.co.uk/2012/04/24/Deploying_Flask_to_Apache.html`](http://www.subdimension.co.uk/2012/04/24/Deploying_Flask_to_Apache.html)。

此指南还提供了更多针对 CentOS 系统的示例，以及通过 Ansible 在 Lighttpd 和 Gunicorn 上部署的所有示例[`www.zufallsheld.de/2014/11/19/deploying-lighttpd-your-flask-apps-gunicorn-and-supervisor-with-ansible-on-centos/`](https://www.zufallsheld.de/2014/11/19/deploying-lighttpd-your-flask-apps-gunicorn-and-supervisor-with-ansible-on-centos/)。

# 摘要

在本章中，我们已经看到了许多运行 Flask 应用程序的方法，包括在多个 Web 服务器中保护隐私和安全，并提供静态文件以减少 Flask 应用程序的负载。我们还为 Ansible 制作了一个配置文件，以实现可重复的应用程序部署，因此，如果需要重新构建机器，这将是一个简单的任务。
