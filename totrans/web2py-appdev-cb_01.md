# 第一章。部署 web2py

在本章中，我们将涵盖以下食谱：

+   在 Windows 上安装 web2py（从源代码）

+   在 Ubuntu 上安装 web2py

+   在 Ubuntu 上设置生产部署

+   使用 Apache、`mod_proxy` 和 `mod_rewrite` 运行 web2py

+   使用 `Lighttpd` 运行 web2py

+   使用 Cherokee 运行 web2py

+   使用 Nginx 和 uWSGI 运行 web2py

+   使用 CGI 在共享主机上运行 web2py

+   在共享主机上使用 `mod_proxy` 运行 web2py

+   从用户定义的文件夹运行 web2py

+   在 Ubuntu 上将 web2py 安装为服务

+   使用 IIS 作为代理运行 web2py

+   使用 ISAPI 运行 web2py

# 简介

在本章中，我们将讨论如何在不同的系统和不同的 Web 服务器上下载、设置和安装 web2py。

### 注意

它们都要求您从网站下载最新的 web2py 源代码：[`web2py.com`](http://web2py.com)，在 Unix 和 Linux 系统的 `/home/www-data/web2py` 下解压，在 Windows 系统的 `c:/web2py` 下解压。在各个地方，我们将假设主机的公共 IP 地址为 `192.168.1.1`；用您自己的 IP 地址或主机名替换它。我们还将假设 web2py 在端口 `8000` 上启动，但这个数字并没有什么特殊之处；如果需要，请更改它。

# 在 Windows 上安装 web2py（从源代码）

虽然为 Windows 环境提供了二进制发行版（打包可执行文件和标准库），但 web2py 是开源的，并且可以与正常的 Python 安装一起使用。

此方法允许使用 web2py 的最新版本，并自定义要使用的 python 模块。

## 准备工作

首先，您必须安装 **Python**。从以下网址下载您喜欢的 2.x 版本（不是 3.x）：[`www.python.org/download/releases/`](http://www.python.org/download/releases/)。

虽然较新版本包括更多增强功能和错误修复，但旧版本具有更高的稳定性和第三方库覆盖范围。Python 2.5.4 在功能和经过验证的稳定性历史之间取得了良好的平衡，具有良好的二进制库支持。Python 2.7.2 是在撰写本文时此平台上的最新生产版本，因此我们将使用它进行示例。

下载您选择的 Windows Python 安装程序（即 **python-2.7.2.msi**），双击安装它。对于大多数情况，默认值都是可以的，所以按 **下一步** 直至安装完成。

您需要 **Python Win32 扩展** 来使用 web2py 任务栏或 Windows 服务。您可以从以下网址安装 **pywin32**：[`starship.python.net/~skippy/win32/Downloads.html`](http://starship.python.net/~skippy/win32/Downloads.html)。

在使用 web2py 之前，您可能还需要一些依赖项来连接到数据库。SQLite 和 MySQL 驱动程序包含在 web2py 中。如果您计划使用其他 RDBMS，您需要安装其驱动程序。

对于 **PostgreSQL**，您可以安装 **psycopg2** 二进制包（对于 Python 2.7，您应使用 `psycopg2-2.3.1.win32-py2.7-pg9.0.1-release.exe`）：[`www.stickpeople.com/projects/python/win-psycopg/`](http://www.stickpeople.com/projects/python/win-psycopg/)（请注意，web2py 需要 **psycopg2** 而不是 **psycopg**）。

对于 MS SQLServer 或 DB2，您需要 **pyodbc**：[`code.google.com/p/pyodbc/downloads/list`](http://code.google.com/p/pyodbc/downloads/list%20)。

## 如何操作...

在这一点上，您可以使用您首选的数据库使用 web2py。

1.  从 web2py 官方网站下载源代码包：[`www.web2py.com/examples/static/web2py_src.zip`](http://www.web2py.com/examples/static/web2py_src.zip)，并解压它。

    由于 web2py 不需要安装，您可以在任何文件夹中解压它。使用 `c:\web2py` 很方便，以保持路径名短。

1.  要启动它，双击 `web2py.py`。您也可以从控制台启动它：

    ```py
    cd c:\web2py
    c:\python27\python.exe web2py.py

    ```

1.  在这里，您可以添加命令行参数（例如 `-a` 用于设置管理员密码，`-p` 用于指定备用端口等）。您可以使用以下命令查看所有启动选项：

```py
C:\web2py>c:\python27\python.exe web2py.py --help

```

## 工作原理...

web2py 是用 Python 编写的，Python 是一种便携、解释和动态的语言，不需要编译或复杂的安装即可运行。它使用虚拟机（如 Java 和 .Net），并且可以在运行脚本时透明地即时字节编译您的源代码。

为了方便新手用户，官方网站上提供了 web2py 的 Windows 二进制发行版，它预先编译成字节码，打包在 zip 文件中，包含所有必需的库（dll/pyd），并附带一个可执行入口点文件（web2py.exe），但使用源代码运行 web2py 并没有明显的区别。

## 还有更多...

在 Windows 中从源代码包运行 web2py 有许多优点，以下列出其中一些：

+   您可以更轻松地使用第三方库，例如 Python Imaging（查看 Python 软件包索引，您可以在那里安装超过一万个模块！）。

+   您可以从其他 Python 程序中导入 web2py 功能（例如，**数据库抽象层 (DAL**)）。

+   您可以使用最新的更改保持 web2py 更新，帮助测试它，并提交补丁。

+   您可以浏览 web2py 的源代码，根据您的定制需求进行调整等。

# 在 Ubuntu 中安装 web2py

本教程涵盖如何在 Ubuntu 桌面环境中安装 web2py。在生产系统中的安装将在下一教程中介绍。

我们假设您知道如何使用控制台和通过控制台安装应用程序。我们将使用最新的 Ubuntu 桌面，即本文撰写时的 Ubuntu Desktop 10.10。

## 准备工作

我们将在您的家目录中安装 web2py，因此请启动控制台。

## 如何操作...

1.  下载 web2py。

    ```py
    cd /home
    mkdir www-dev
    cd www-dev
    wget http://www.web2py.com/examples/static/web2py_src.zip
    (get web2py)

    ```

1.  下载完成后，解压它：

    ```py
    unzip -x web2py_src.zip

    ```

1.  如果您想使用 GUI，可以选择安装 Python 的 `tk` 库。

    ```py
    sudo apt-get install python-tk

    ```

    ### 注意

    **下载示例代码**

    您可以从您在 [`www.PacktPub.com`](http://www.PacktPub.com) 的账户中下载您购买的所有 Packt 书籍的示例代码文件。如果您在其他地方购买了此书，您可以访问 [`www.PacktPub.com/support`](http://www.PacktPub.com/support)，并注册以将文件直接通过电子邮件发送给您。代码文件也上传到了以下存储库：[`github.com/mdipierro/web2py-recipes-source`](http://https://github.com/mdipierro/web2py-recipes-source)。

    所有代码均在 BSD 许可下发布([`www.opensource.org/licenses/bsd-license.php`](http://www.opensource.org/licenses/bsd-license.php))，除非源文件中另有说明。

1.  要启动 web2py，请访问 web2py 目录并运行 web2py。

    ```py
    cd web2py
    python web2py.py

    ```

    ![如何做到这一点...](img/5467OS_01_00.jpg)

    +   安装后，每次运行它时，web2py 都会要求您选择一个密码。这个密码是您的管理员密码。如果密码留空，则管理界面将被禁用。

1.  在您的浏览器中输入 `127.0.0.1:8000/` 以检查一切是否正常工作。

### 注意

管理界面：`http://127.0.0.1:8000/admin/default/index` 只能通过 `localhost` 访问，并且始终需要密码。它也可以通过 SSH 隧道访问。

## 还有更多...

您可以使用一些其他选项。例如，您可以使用选项 `-p port` 指定端口，使用选项 `-i 127.0.0.1` 指定 IP 地址。指定密码很有用，这样您就不必每次启动 web2py 时都输入它；使用选项 `-a` 指定密码。如果您需要其他选项的帮助，请使用带有 `-h` 或 `help` 选项的 web2py 运行。

例如：

```py
python web2py.py -i 127.0.0.1 -p 8000 -a mypassword --nogui

```

# 在 Ubuntu 上设置生产部署

本菜谱描述了如何在 Ubuntu 服务器上使用生产环境安装 web2py。这是在生产环境中部署 web2py 的推荐方法。

## 准备工作

我们假设您知道如何使用控制台，并使用存储库和命令安装应用程序。我们将使用写作时的最新 Ubuntu 服务器：Ubuntu Server 10.04 LTS。

在这个菜谱中，我们将学习如何：

+   在 Ubuntu 上安装运行 web2py 所需的所有模块

+   在 `/home/www-data/` 中安装 web2py

+   创建自签名 SSL 证书

+   使用 `mod_wsgi` 设置 web2py

+   覆盖 `/etc/apache2/sites-available/default`

+   重启 Apache

![准备工作](img/5467OS_01_01.jpg)

首先，我们需要确保系统是最新的。使用以下命令升级系统：

```py
sudo apt-get update
sudo apt-get upgrade

```

## 如何做到这一点...

1.  让我们从安装 `postgreSQL:` 开始。

    ```py
    sudo apt-get install postgresql

    ```

1.  如果尚未安装，我们需要解压并打开 `ssh-server`。

    ```py
     sudo apt-get install unzip
    sudo apt-get install openssh-server

    ```

1.  安装 Apache 2 和 `mod-wsgi:`

    ```py
    sudo apt-get install apache2
    sudo apt-get install libapache2-mod-wsgi

    ```

1.  可选地，如果您计划操作图像，我们可以安装 **Python Imaging Library (PIL)** :

    ```py
    sudo apt-get install python-imaging

    ```

1.  现在我们需要安装 web2py。我们将在 `/home` 中创建 `www-data` 并在那里提取 web2py 源代码。

    ```py
    cd /home
    sudo mkdir www-data
    cd www-data

    ```

1.  从 web2py 网站获取 web2py 源代码：

    ```py
    sudo wget http://web2py.com/examples/static/web2py_src.zip
    sudo unzip web2py_src.zip
    sudo chown -R www-data:www-data web2py

    ```

1.  启用 Apache SSL 和 EXPIRES 模块：

    ```py
    sudo a2enmod expires
    sudo a2enmod ssl

    ```

1.  创建自签名证书：

    您应该从受信任的**证书颁发机构**获取您的 SSL 证书，例如 `verisign.com`，但出于测试目的，您可以生成自己的自签名证书。您可以在：[`help.ubuntu.com/10.04/serverguide/C/certificates-and-security.html.`](https://help.ubuntu.com/10.04/serverguide/C/certificates-and-security.html.%20) 了解更多。

1.  创建 `SSL` 文件夹，并将 SSL 证书放入其中：

    ```py
    sudo openssl req -new -x509 -nodes -sha1 -days 365 -key \
    /etc/apache2/ssl/self_signed.key > \
    /etc/apache2/ssl/self_signed.cert
    sudo openssl x509 -noout -fingerprint -text < \
    /etc/apache2/ssl/self_signed.cert > \
    /etc/apache2/ssl/self_signed.info

    ```

1.  如果您遇到权限问题，请使用 `sudo -i`。

1.  使用您的编辑器编辑默认的 Apache 配置。

    ```py
    sudo nano /etc/apache2/sites-available/default

    ```

1.  将以下代码添加到配置中：

    ```py
    NameVirtualHost *:80
    NameVirtualHost *:443

    <VirtualHost *:80>
    	WSGIDaemonProcess web2py user=www-data group=www-data
    	WSGIProcessGroup web2py
    	WSGIScriptAlias / /home/www-data/web2py/wsgihandler.py

    	<Directory /home/www-data/web2py>
    		AllowOverride None
    		Order Allow,Deny
    		Deny from all
    		<Files wsgihandler.py>
    			Allow from all
    		</Files>
    	</Directory>

    	AliasMatch ^/([^/]+)/static/(.*) \
    		/home/www-data/web2py/applications/$1/static/$2

    	<Directory /home/www-data/web2py/applications/*/static/>
    		Options -Indexes
    		Order Allow,Deny
    		Allow from all
    	</Directory>

    	<Location /admin>
    		Deny from all
    	</Location>

    	<LocationMatch ^/([^/]+)/appadmin>
    		Deny from all
    	</LocationMatch>

    	CustomLog /var/log/apache2/access.log common
    	ErrorLog /var/log/apache2/error.log
    </VirtualHost>

    <VirtualHost *:443>
    	SSLEngine on
    	SSLCertificateFile /etc/apache2/ssl/self_signed.cert
    	SSLCertificateKeyFile /etc/apache2/ssl/self_signed.key

    	WSGIProcessGroup web2py

    	WSGIScriptAlias / /home/www-data/web2py/wsgihandler.py

    	<Directory /home/www-data/web2py>
    		AllowOverride None
    		Order Allow,Deny
    		Deny from all
    		<Files wsgihandler.py>
    			Allow from all
    		</Files>
    	</Directory>

    	AliasMatch ^/([^/]+)/static/(.*) \
    		/home/www-data/web2py/applications/$1/static/$2
    	<Directory /home/www-data/web2py/applications/*/static/>
    		Options -Indexes
    		ExpiresActive On
    		ExpiresDefault "access plus 1 hour"
    		Order Allow,Deny
    		Allow from all
    	</Directory>

    	CustomLog /var/log/apache2/access.log common
    	ErrorLog /var/log/apache2/error.log
    </VirtualHost>

    ```

1.  重新启动 Apache 服务器：

    ```py
     sudo /etc/init.d/apache2 restart
    	cd /home/www-data/web2py
    	sudo -u www-data python -c "from gluon.widget import console; \
    console();"
    	sudo -u www-data python -c "from gluon.main \
    import save_password; \
    save_password(raw_input('admin password: '),443)"

    ```

1.  在您的浏览器中输入 `http://192.168.1.1/` 以检查一切是否正常工作，将 `192.168.1.1` 替换为您的公网 IP 地址。

## 还有更多...

我们所做的一切都可以使用 web2py 提供的脚本自动完成：

```py
 wget http://web2py.googlecode.com/hg/scripts/setup-web2py-\
	ubuntu.sh
chmod +x setup-web2py-ubuntu.sh
sudo ./setup-web2py-ubuntu.sh

```

# 使用 Apache、mod_proxy 和 mod_rewrite 运行 web2py

**Apache httpd** 是最受欢迎的 HTTP 服务器，在大型安装中拥有 Apache httpd 是必需的，就像在意大利圣诞节那天必须有潘内托尼一样。就像潘内托尼一样，Apache 有很多口味和不同的填充物。您必须找到您喜欢的。

在此配方中，我们使用 `mod_proxy` 配置 Apache，并通过 `mod_rewrite` 规则对其进行优化。这是一个简单但稳健的解决方案。它可以用来提高 web2py 的可伸缩性、吞吐量、安全性和灵活性。这些规则应该能满足专家和初学者的需求。

此配方将向您展示如何在主机上创建一个 web2py 安装，使其看起来像网站的一部分，即使它托管在其他地方。我们还将展示如何使用 Apache 来提高您的 web2py 应用程序的性能，而不需要接触 web2py。

## 准备工作

您应该有以下内容：

+   web2py 已安装在 `localhost` 上并运行，使用内置的 Rocket 服务器（端口 8000）

+   Apache HTTP 服务器 (`httpd`) 版本 2.2.x 或更高版本

+   `mod_proxy` 和 `mod_rewrite`（包含在标准的 Apache 发行版中）

在 Ubuntu 或其他基于 Debian 的服务器上，您可以使用以下命令安装 Apache：

```py
apt-get install apache

```

在 CentOS 或其他基于 Fedora 的 Linux 发行版上，您可以使用以下命令安装 Apache：

```py
yum install httpd

```

对于大多数其他系统，您可以从网站 [`httpd.apache.org/`](http://httpd.apache.org/) 下载 Apache，并按照提供的说明自行安装。

## 如何操作...

现在我们已经本地运行了 Apache HTTP 服务器（从现在起我们将简单地称之为 Apache）和 web2py，我们必须对其进行配置。

Apache 通过在纯文本配置文件中放置指令来配置。主要的配置文件通常称为 `httpd.conf`。此文件的默认位置在编译时设置，但可以使用 `-f` 命令行标志进行覆盖。`httpd.conf` 可能包含其他配置文件。额外的指令可以放置在这些配置文件中的任何一个。

配置文件可能位于 `/etc/apache2`、`/etc/apache` 或 `/etc/httpd`，具体取决于操作系统和 Apache 版本的细节。

1.  在编辑任何文件之前，请确保从命令行 shell (`bash`) 启用了所需的模块，输入：

    ```py
    a2enmod proxy
    a2enmod rewrite

    ```

    +   在 `mod_proxy` 和 `mod_rewrite` 启用后，我们现在可以设置一个简单的重写规则，将 Apache 收到的 HTTP 请求代理转发到我们希望的其他任何 HTTP 服务器。Apache 支持多个 `VirtualHosts`，也就是说，它能够在单个 Apache 实例中处理不同的虚拟主机名称和端口。默认的 `VirtualHost` 配置位于名为 `/etc/<apache>/ sites-available/default` 的文件中，其中 `<apache>` 是 apache、apache2 或 httpd。`

1.  在此文件中，每个 `VirtualHost` 都是通过创建以下条目来定义的：

    ```py
     <VirtualHost *:80>
    	...
    </VirtualHost>

    ```

    +   您可以在 `http://httpd.apache.org/docs/2.2/vhosts/.` 阅读关于 `VirtualHost` 的深入文档。

1.  要使用 `RewriteRules`，我们需要在 `VirtualHost:` 内部激活 **Rewrite Engine**。

    ```py
     <VirtualHost *:80>
    	RewriteEngine on
    	...
    </VirtualHost>

    ```

1.  然后，我们可以配置重写规则：

    ```py
     <VirtualHost *:80>
    	RewriteEngine on
    	# make sure we handle the case with no / at the end of URL
    	RewriteRule ^/web2py$ /web2py/ [R,L]

    	# when matching a path starting with /web2py/ do use a reverse
    	# proxy
    	RewriteRule ^/web2py/(.*) http://localhost:8000/$1 [P,L]
    	...
    </VirtualHost>

    ```

    +   第二条规则告诉 Apache 对 `http://localhost:8000` 执行反向代理连接，传递用户调用的 URL 的所有路径组件，除了第一个，即 web2py。规则使用的语法基于正则表达式 (`regex`)，其中第一个表达式与传入的 URL（用户请求的 URL）进行比较。

        如果有匹配，则使用第二个表达式来构建一个新的 URL。`[`and`]` 内部的标志决定了如何处理生成的 URL。前面的例子匹配任何以 `/web2py` 开头的默认 `VirtualHost` 路径的传入请求，并生成一个新的 URL，将 `http://localhost:8000/` 预先添加到匹配路径的剩余部分；与表达式 .* 匹配的传入 URL 的部分替换第二个表达式中的 `$1`。

        标志 `P` 告诉 Apache 在将其传递回请求的浏览器之前，使用其代理检索 URL 所指向的内容。

        假设 Apache 服务器响应域名 [www.example.com](http://www.example.com)；那么如果用户的浏览器请求 [`www.example.com/web2py/welcome`](http://www.example.com/web2py/welcome)，它将收到来自 web2py 框架应用的响应内容。也就是说，这就像浏览器请求了 `http://localhost:8000/welcome` 一样。

1.  有一个陷阱：web2py 可能会发送一个 HTTP 重定向，例如将用户的浏览器指向默认页面。问题是重定向是相对于 web2py 的应用程序布局的，即 Apache 代理试图隐藏的那个布局，因此重定向很可能会指向错误的位置。为了避免这种情况，我们必须配置 Apache 来拦截重定向并纠正它们。

    ```py
     <VirtualHost *:80>
    	...
    	#make sure that HTTP redirects generated by web2py are reverted
    		/ -> /web2py/
    	ProxyPassReverse /web2py/ http://localhost:8000/
    	ProxyPassReverse /web2py/ /

    	# transform cookies also
    	ProxyPassReverseCookieDomain localhost localhost
    	ProxyPassReverseCookiePath / /web2py/
    	...
    </VirtualHost>

    ```

1.  还有一个问题。由 web2py 生成的许多 URL 也相对于 web2py 的上下文。这包括图像或 CSS 样式表的 URL。我们必须指导 web2py 如何编写正确的 URL，当然，由于它是 web2py，所以很简单，我们不需要修改应用程序代码中的任何代码。我们需要在 web2py 安装根目录下定义一个名为 `routes.py` 的文件，如下所示：

    ```py
    routes_out=((r'^/(?P<any>.*)', r'/web2py/\g<any>'),)

    ```

1.  在此阶段，Apache 可以在将内容发送回客户端之前对其进行转换。我们有几种方法可以提高网站速度。例如，如果浏览器接受压缩内容，我们可以在将内容发送回浏览器之前对其进行压缩。

```py
 # Enable content compression on the fly,
# speeding up the net transfer on the reverse proxy.
<Location /web2py/>
	# Insert filter
	SetOutputFilter DEFLATE
	# Netscape 4.x has some problems...
	BrowserMatch ^Mozilla/4 gzip-only-text/html
	# Netscape 4.06-4.08 have some more problems
	BrowserMatch ^Mozilla/4\.0[678] no-gzip
	# MSIE masquerades as Netscape, but it is fine
	BrowserMatch \bMSIE !no-gzip !gzip-only-text/html
	# Don't compress images
	SetEnvIfNoCase Request_URI \
		\.(?:gif|jpe?g|png)$ no-gzip dont-vary
	# Make sure proxies don't deliver the wrong content
	Header append Vary User-Agent env=!dont-vary
</Location>

```

+   同样，只需配置 Apache，就可以执行其他有趣的任务，例如 SSL 加密、负载均衡、通过内容缓存加速，以及许多其他事情。您可以在以下网站找到有关这些和许多其他配置的信息：[`httpd.apache.org.`](http://httpd.apache.org.%20)

这里是以下配方中使用的默认虚拟主机完整配置：

```py
 <VirtualHost *:80>
	ServerName localhost
	# ServerAdmin: Your address, where problems with the server
	# should
	# be e-mailed. This address appears on some server-generated
	# pages,
	# such as error documents. e.g. admin@your-domain.com
	ServerAdmin root@localhost

	# DocumentRoot: The directory out of which you will serve your
	# documents. By default, all requests are taken from this
	# directory,
	# but symbolic links and aliases may be used to point to other
	# locations.
	# If you change this to something that isn't under /var/www then
	# suexec will no longer work.
	DocumentRoot "/var/www/localhost/htdocs"

	# This should be changed to whatever you set DocumentRoot to.
	<Directory "/var/www/localhost/htdocs">
		# Possible values for the Options directive are "None", "All",
		# or any combination of:
		# 	Indexes Includes FollowSymLinks
		# 	SymLinksifOwnerMatch ExecCGI MultiViews
		#
		# Note that "MultiViews" must be named *explicitly* ---
		# "Options All"
		# doesn't give it to you.
		#
		# The Options directive is both complicated and important.
		# Please
		# see http://httpd.apache.org/docs/2.2/mod/core.html#options
		# for more information.
		Options Indexes FollowSymLinks
		# AllowOverride controls what directives may be placed in
		# .htaccess
		# It can be "All", "None", or any combination of the keywords:
		# 	Options FileInfo AuthConfig Limit
		AllowOverride All

		# Controls who can get stuff from this server.
			Order allow,deny
			Allow from all
		</Directory>

		### WEB2PY EXAMPLE PROXY REWRITE RULES
		RewriteEngine on
		# make sure we handle when there is no / at the end of URL
		RewriteRule ^/web2py$ /web2py/ [R,L]

		# when matching a path starting with /web2py/ do a reverse proxy
		RewriteRule ^/web2py/(.*) http://localhost:8000/$1 [P,L]

		# make sure that HTTP redirects generated by web2py are reverted
		# / -> /web2py/
		ProxyPassReverse /web2py/ http://localhost:8000/
		ProxyPassReverse /web2py/ /

		# transform cookies also
		ProxyPassReverseCookieDomain localhost localhost
		ProxyPassReverseCookiePath / /web2py/

		# Enable content compression on the fly speeding up the net
		# transfer on the reverse proxy.
		<Location /web2py/>
			# Insert filter
			SetOutputFilter DEFLATE
			# Netscape 4.x has some problems...
			BrowserMatch ^Mozilla/4 gzip-only-text/html
			# Netscape 4.06-4.08 have some more problems
			BrowserMatch ^Mozilla/4\.0[678] no-gzip
			# MSIE masquerades as Netscape, but it is fine
			BrowserMatch \bMSIE !no-gzip !gzip-only-text/html
			# Don't compress images
			SetEnvIfNoCase Request_URI \
				\.(?:gif|jpe?g|png)$ no-gzip dont-vary
			# Make sure proxies don't deliver the wrong content
			Header append Vary User-Agent env=!dont-vary
		</Location>
</VirtualHost>

```

您必须重新启动 Apache 以使任何更改生效。您可以使用以下命令进行相同操作：

```py
apachectl restart

```

# 使用 Lighttpd 运行 web2py

**Lighttpd** 是一个安全、快速、兼容且非常灵活的 Web 服务器，它针对高性能环境进行了优化。与其他 Web 服务器相比，它具有非常低的内存占用，并关注 cpu-load。其高级功能集（FastCGI、CGI、认证、输出压缩、URL 重写等）使 Lighttpd 成为每个遭受负载问题的服务器的完美 Web 服务器软件。

这个配方是从官方 web2py 书籍中提取的，但尽管书中使用 FastCGI `mod_fcgi` 在 Ligthttpd Web 服务器后面公开 web2py 功能，这里我们使用 SCGI。我们在这里使用的 SCGI 协议在意图上与 FastCGI 类似，但更简单、更快。它描述在以下网站上：

[`python.ca/scgi`](http://python.ca/scgi)

**SCGI** 是一种用于 IP 上进程间通信的二进制协议。SCGI 专为 Web 服务器与 CGI 应用程序之间的通信任务量身定制。CGI 标准定义了 Web 服务器如何将动态生成 HTTP 响应的任务委托给外部应用程序。

CGI 的问题在于，对于每个传入的请求，都必须创建一个新的进程。在某些情况下，进程创建可能比响应生成所需的时间更长。这在大多数解释语言环境中都是正确的，其中加载新解释器实例的时间可能比程序本身的执行时间更长。

**FastCGI** 通过使用长时间运行的进程来回答多个请求而不退出，从而解决了这个问题。这对于解释程序特别有益，因为每次不需要重新启动解释器。SCGI 是在 FastCGI 经验之后开发的，以减少将 CGI 转换为 FastCGI 应用程序所需的复杂性，从而提高性能。SCGI 是 Lighttpd 的标准模块，也适用于 Apache。

## 准备工作。

您应该有：

+   web2py 已安装在本地主机（端口 `8000`）上。

+   Lighttpd（从 [`www.lighttpd.net`](http://www.lighttpd.net) 下载并安装）。

+   SCGI（从 [`python.ca/scgi`](http://python.ca/scgi) 下载并安装）。

+   Python Paste（从 [`pythonpaste.org/`](http://pythonpaste.org/) 下载并安装），或 WSGITools（http://subdivi.de/helmut/wsgitools）。

如果您有 `setuptools`，您可以安装 SCGI、paste 和 wsgitools，如下所示：

```py
easy_install scgi
easy_install paste
easy_install wsgitools

```

您还需要一个脚本来启动一个配置为 web2py 的 SCGI 服务器，这个脚本可能随 web2py 一起提供，也可能不提供，这取决于版本，因此我们为这个配方提供了一个。

## 如何做到这一点...

现在，您必须编写一个脚本来启动将监听 Lighttpd 请求的 SCGI 服务器。别担心，即使它非常短且简单，我们在这里提供了一个可以复制的示例：

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

LOGGING = False
SOFTCRON = False

import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
sys.path = [path]+[p for p in sys.path if not p==path]

import gluon.main

if LOGGING:
application = gluon.main.appfactory(
	wsgiapp=gluon.main.wsgibase,
	logfilename='httpserver.log',
profilerfilename=None)
else:
	application = gluon.main.wsgibase

if SOFTCRON:
	from gluon.settings import global_settings
	global_settings.web2py_crontype = 'soft'

try:
	import paste.util.scgiserver as scgi
	scgi.serve_application(application, '', 4000).run()
except ImportError:
from wsgitools.scgi.forkpool import SCGIServer
SCGIServer(application, port=4000).run()

```

1.  复制前面的脚本，并将其放在您的 web2py 安装根目录中，命名为 `scgihandler.py`。启动 SCGI 服务器，并在后台运行：

    ```py
    $ nohup python ./scgihandler.py &

    ```

    +   现在，我们已经准备好配置 `lighttpd`。

        我们提供了一个简单的 `lighttpd.conf` 配置文件作为示例。当然，现实世界的配置可能更加复杂，但重要部分不会有太大差异。

1.  将以下行追加到您的 `lighttpd.conf` 文件中：

    ```py
    server.modules += ( "mod_scgi" )
    server.document-root="/var/www/web2py/"
    # for >= linux-2.6
    server.event-handler = "linux-sysepoll"

    url.rewrite-once = (
    	"^(/.+?/static/.+)$" => "/applications$1",
    	"(^|/.*)$" => "/handler_web2py.scgi$1",
    )
    scgi.server = ( "/handler_web2py.scgi" =>
    	("handler_web2py" =>
    		( "host" => "127.0.0.1",
    		"port" => "4000",
    		"check-local" => "disable", # important!
    		)
    	)
    )

    ```

1.  此配置执行以下操作：

    +   将 SCGI 模块加载到 Lighttpd 中。

    +   将服务器文档根配置为 web2py 安装根目录。

    +   使用 `mod_rewrite` 重写 URL，以便静态文件的请求直接由 Lighttpd 服务，而所有其他请求都被重写到以 `/handler_web2py.scgi` 开头的 **假** URL。

    +   **创建一个 SCGI 服务器段落：** 对于以 `/handler_web2py.scgi` 开头的每个请求，请求会被路由到运行在 `127.0.0.1` 的 `4000` 端口的 SCGI 服务器，跳过检查文件系统上是否存在相应的本地文件。

1.  现在，检查您的配置是否正确：

    ```py
    $ lighttpd -t -f lighttpd.conf

    ```

1.  然后启动服务器进行测试：

    ```py
    $ lighttpd -D -f lighttpd.conf

    ```

1.  您可以使用以下命令启动/停止/重启服务器：

```py
$ /etc/init.d/lighttpd start|stop|restart

```

您将看到您的 web2py 应用程序达到光速（ttpd）的速度。

# 使用 Cherokee 运行 web2py。

此配方解释了如何在 Cherokee 网络服务器后面运行 web2py，使用 **uWSGI**。

**Cherokee** 是用 C 语言编写的 web 服务器，其意图与 Lighttpd 相似：快速、紧凑且模块化。Cherokee 附带一个管理界面，允许用户管理其配置，否则配置难以阅读和修改。uWSGI 在其网站上描述为一种快速（纯 C）、自我修复、开发者/系统管理员友好的应用程序容器服务器。Cherokee 包含一个用于与 uWSGI 服务器通信的模块。

## 如何操作...

1.  安装软件包或下载、编译和安装所需的组件。在 web2py 安装根目录中创建以下文件，并将其命名为 `uwsgi.xml`: 

    ```py
     <uwsgi>
    	<pythonpath>/home/web2py</pythonpath>
    	<module>wsgihandler</module>
    	<socket>127.0.0.1:37719</socket>
    	<master/>
    	<processes>8</processes>
    	<memory-report/>
    </uwsgi>

    ```

    +   此配置会启动八个进程来管理来自 HTTP 服务器的多个请求。根据需要更改它，并将 `<pythonpath>` 配置为 web2py 的安装根目录。

1.  以拥有 web2py 安装的用户的身份启动 uWSGI 服务器：

    ```py
    $ uWSGI -d uwsgi.xml

    ```

1.  现在，启动 Cherokee 管理界面以创建新的配置：

    ```py
    $ cherokee-admin

    ```

1.  使用浏览器通过以下链接连接到管理界面：`http://localhost:9090/`。

    ![如何操作...](img/5467OS_01_02.jpg)

1.  前往**源**部分 - **(A)**，然后点击**+**按钮 - **(B)**。

1.  在 **(C)** 处选择**远程主机**，然后在 **(D)** 处的文本字段中填写 IP 地址和端口，以匹配上一个 `uswgi.xml` 文件中的配置。

    配置了 uWGI 源之后，现在可以配置一个虚拟主机，并通过它重定向请求。在这个菜谱中，我们选择当没有其他虚拟主机更适合传入请求时使用的**默认**虚拟主机。

1.  点击按钮 `(C)` 进入**规则管理**。

    ![如何操作...](img/5467OS_01_03.jpg)

1.  删除左侧列出的所有规则。只有**默认**规则将保留。

    ![如何操作...](img/5467OS_01_04.jpg)

1.  使用 uWSGI **处理器**配置**默认**规则。其他值保持不变。

    ![如何操作...](img/5467OS_01_05.jpg)

1.  如果你想让 Cherokee 直接从 web2py 文件夹中服务静态文件，你可以添加一个**正则表达式**规则。点击按钮 **(A)**，然后从 **(B)** 处的下拉菜单中选择**正则表达式**。请注意，此配置仅在 web2py 目录位于同一文件系统且可由 Cherokee 访问时才有效。

    ![如何操作...](img/5467OS_01_06.jpg)

1.  配置**正则表达式**：

    ![如何操作...](img/5467OS_01_07.jpg)

1.  现在，您可以配置指向您的 web2py 安装的应用程序子目录的静态处理器：

    ![如何操作...](img/5467OS_01_08.jpg)

    +   记得保存配置，并从管理界面重新加载或重启 Cherokee；然后您就可以开始启动 uWSGI 服务器了。

1.  切换到用于安装 web2py 的正确用户 ID；请注意，不建议使用 root 用户。

1.  进入 web2py 安装根目录，其中保存了配置文件 `uwsgi.xml`。

1.  使用 `-d <logfile>` 选项运行 uWSGI，使其在后台运行：

```py
$ su - <web2py user>
$ cd <web2py root>
$ uwsgi -x uwsgi.xml -d /tmp/uwsgi.log

```

享受速度！

## 准备工作

您应该具备以下条件：

+   web2py（已安装但未运行）

+   uWSGI（从[`projects.unbit.it/uwsgi/wiki`](http://projects.unbit.it/uwsgi/wiki)下载并安装）

+   Cherokee（从[`www.cherokee-project.com/`](http://www.cherokee-project.com/)下载并安装）

# 使用 Nginx 和 uWSGI 运行 web2py

本菜谱解释了如何使用 uWSGI 在 Nginx 网络服务器上运行 web2py。

**Nginx**是一个免费、开源、高性能的 HTTP 服务器和反向代理，由**Igor Sysoev**编写。

与传统服务器不同，Nginx 不依赖于线程来处理请求，而是实现了一个异步架构。这意味着 Nginx 即使在重负载下也能使用可预测的内存量，从而实现更高的稳定性和低资源消耗。Nginx 现在托管了全球所有域的超过七个百分点。

应强调的是，即使 Nginx 是异步的，web2py 也不是。因此，web2py 处理的并发请求越多，它使用的资源就越多。uWSGI 在其网站上被描述为一个快速（纯 C）、自我修复、开发者/系统管理员友好的应用程序容器服务器。我们将配置 Nginx 通过 uWSGI 服务动态的 web2py 页面，并直接服务静态页面，利用其低内存占用能力。

## 准备工作

您应该具备以下条件：

+   web2py（已安装但未运行）

+   uWSGI（从[`projects.unbit.it/uwsgi/wiki`](http://projects.unbit.it/uwsgi/wiki)下载并安装）

+   Nginx（从[`nginx.net/`](http://nginx.net/)下载并安装）

在 Ubuntu 10.04 LTS 上，您可以使用`apt-get`安装 uWSGI 和 Nginx，如下所示：

```py
apt-get update
apt-get -y upgrade
apt-get install python-software-properties
add-apt-repository ppa:nginx/stable
add-apt-repository ppa:uwsgi/release
apt-get update
apt-get -y install nginx-full
apt-get -y install uwsgi-python

```

## 如何操作...

1.  首先，我们需要配置 Nginx。创建或编辑一个名为`/etc/nginx/sites-available/web2py`的文件。

1.  在文件中，写入以下内容：

    ```py
     server {
    	listen 			80;
    	server_name 	$hostname;
    	location ~* /(\w+)/static/ {
    		root /home/www-data/web2py/applications/;
    	}
    	location / {
    		uwsgi_pass 	127.0.0.1:9001;
    		include 	uwsgi_params;
    	}
    }

    server {
    	listen 			443;
    	server_name 	$hostname;
    	ssl 					on;
    	ssl_certificate 		/etc/nginx/ssl/web2py.crt;
    	ssl_certificate_key 	/etc/nginx/ssl/web2py.key;
    	location / {
    		uwsgi_pass 	127.0.0.1:9001;
    		include 	uwsgi_params;
    		uwsgi_param UWSGI_SCHEME $scheme;
    	}
    }

    ```

    +   如您所见，它将所有动态请求传递到`127.0.0.1:9001`。我们需要在那里运行 uWSGI。

1.  在 web2py 的安装根目录下创建以下文件，并将其命名为`web2py.xml:`

    ```py
     <uwsgi>
    	<socket>127.0.0.1:9001</socket>
    	<pythonpath>/home/www-data/web2py/</pythonpath>
    	<app mountpoint="/">
    		<script>wsgihandler</script>
    	</app>
    </uwsgi>

    ```

    +   此脚本假设 web2py 通常安装在`/home/www-data/web2py/.`。

1.  现在禁用默认配置，并启用新的配置：

    ```py
    rm /etc/nginx/sites-enabled/default
    rm /etc/nginx/sites-available/default
    ln -s /etc/nginx/sites-available/web2py /etc/nginx/sites-enabled/\
    web2py
    ln -s /etc/uwsgi-python/apps-available/web2py.xml /etc/uwsgi-\
    python/apps-enabled/web2py.xml

    ```

1.  为了使用 HTTPS，您可能需要创建一个自签名证书：

    ```py
    mkdir /etc/nginx/ssl
    cd /etc/nginx/ssl
    openssl genrsa -out web2py.key 1024
    openssl req -batch -new -key web2py.key -out web2py.csr
    openssl x509 -req -days 1780 -in web2py.csr -signkey web2py.key \
    -out web2py.crt

    ```

1.  您还需要启用 web2py 管理员：

    ```py
    cd /var/web2py
    sudo -u www-data python -c "from gluon.main import save_password;\
    save_password('$PW', 443)"

    ```

1.  完成后，重新启动 uWSGI 和 Nginx：

```py
/etc/init.d/uwsgi-python restart
/etc/init.d/nginx restart

```

web2py 附带一个脚本，可以自动为您完成此设置：

`scrips/setup-web2py-nginx-uwsgi-ubuntu.sh`

# 在共享主机上使用 CGI 运行 web2py

本菜谱解释了如何配置 web2py 在具有登录（但不是 root）访问权限的共享主机上运行。

使用共享主机的登录或 FTP 访问，用户无法配置网络服务器，必须遵守主机的配置限制。本菜谱假设运行 Apache 的典型基于 Unix 或 Linux 的共享主机。

根据系统的配置，有两种部署方法。如果 Apache 的`mod_proxy`可用，并且主机允许长时间运行进程，将 web2py 的内置服务器作为 Apache 代理运行是简单且高效的。如果`mod_proxy`不可用，或者主机禁止长时间运行进程，我们将局限于 CGI 接口，该接口配置简单且几乎无处不在，但速度较慢，因为 Python 解释器必须为每个请求运行和加载 web2py。

我们将从 CGI 部署开始，这是一个简单的情况。

## 准备工作

我们假设你的网站根目录是`/usr/www/users/username`，而`/usr/www/users/username/cgi-bin`是你的 CGI 二进制目录。如果你的详细信息不同，从你的服务提供商那里获取实际值，并相应地修改这些说明。

由于安全原因，这里我们假设你的主机支持以本地用户（cgiwrap）的身份运行 CGI 脚本。如果它可用，这个程序可能因主机而异；请咨询你的服务提供商。

将 web2py 源代码下载到你的`cgi-bin`目录。例如：

```py
cd cgi-bin
wget http://www.web2py.com/examples/static/web2py_src.zip
unzip web2py_src.zip
rm web2py_src.zip

```

或者，在本地解压 web2py 源代码，并通过 FTP 上传到主机。

## 如何操作...

1.  在你的 Web 根目录中，如果需要，创建文件`.htaccess`，并添加以下行（根据需要更改路径）：

    ```py
     SuexecUserGroup <yourusername> <yourgroup>
    RewriteEngine on
    RewriteBase /usr/www/users/username
    RewriteRule ^(welcome|examples|admin)(/.*)?$ \
    			/cgi-bin/cgiwrap/username/web2py/cgihandler.py

    ```

1.  使用以下命令更改其权限：

    ```py
    chown 644 .htaccess

    ```

1.  现在访问[`yourdomain.com/welcome`](http://yourdomain.com/welcome)，或者（根据你的服务提供商）[`hostingdomain.com/username/welcome`](http://hostingdomain.com/username/welcome)。

1.  如果你在这个阶段遇到访问错误，请使用`tail`命令检查`web2py/applications/welcome/errors/`目录中最新的文件。这种格式并不特别友好，但它可以提供有用的线索。如果`errors`目录为空，你可能需要再次检查`errors`目录是否可由 Web 服务器写入。

# 在共享主机上使用 mod_proxy 运行 web2py

使用`mod_proxy`比之前菜谱中讨论的 CGI 部署有两个主要优势：web2py 持续运行，因此性能显著更好，并且它以你的本地用户身份运行，这提高了安全性。因为从 web2py 的角度来看，它似乎是在 localhost 上运行的，所以管理应用程序可以运行，但如果你没有 SSL 操作可用，你可能出于安全原因想要禁用管理。SSL 设置在*在 Ubuntu 上设置生产部署*菜谱中讨论。

## 准备工作

这里我们假设你已经将 web2py 下载并解压到你的家目录中的某个位置。我们还假设你的 Web 托管提供商已启用 mod_proxy，支持长时间运行进程，允许你打开一个端口（例如示例中的 8000，但如果你发现该端口已被其他用户占用，你可以更改它）。

## 如何操作...

1.  在你的基本 Web 目录中，如果需要，创建文件`.htaccess`，并添加以下行：

    ```py
     RewriteEngine on
    RewriteBase /usr/www/users/username
    RewriteRule ^((welcome|examples|admin)(/.*)?)$ \
    			http://127.0.0.1:8000/$1 [P]

    ```

1.  按照之前描述的 CGI 操作方式下载和解压 web2py，除了 web2py 不需要安装在你的`cgi-bin`目录中，甚至不需要在你的网页文档树中。对于这个配方，我们假设你将其安装在登录主目录`$HOME`中。

1.  使用以下命令在本地主机和端口`8000`上启动 web2py 运行：

```py
nohup python web2py.py -a password -p 8000 -N

```

+   `password`是你选择的单次管理员密码。`-N`是可选的，它禁用了`web2py`的 cron 以节省内存。（注意，最后一步不能通过 FTP 完成，因此需要登录访问。）

# 从用户定义的文件夹运行 web2py

这个配方解释了如何移动 web2py 的`applications`文件夹。

在 web2py 中，每个应用程序都位于`applications/`文件夹下的一个文件夹中，而`applications/`文件夹又位于 web2py 的`base`或`root`文件夹中（该文件夹还包含`gluon/`，web2py 的核心代码）。

当使用 web2py 内置的 web 服务器部署时，`applications/`文件夹可以被移动到文件系统中的其他位置。当`applications/`被移动时，某些其他文件也会随之移动，包括`logging.conf`、`routes.py`和`parameters_port.py`。此外，位于移动后的`applications/`文件夹同一目录下的`site-packages`会被插入到`sys.path`中（这个`site-packages`目录不必存在）。

## 如何操作...

当 web2py 从命令行运行时，文件夹移动位置可以通过`-f`选项指定，该选项应指定移动后的`applications/`文件夹的父文件夹，例如：

```py
python web2py.py -i 127.0.0.1 -p 8000 -f /path/to/apps

```

## 还有更多...

当 web2py 以 Windows 服务的方式运行（`web2py.exe -W`）时，移动位置可以在 web2py 主文件夹中的`options.py`文件中指定。将默认文件夹`os.getcwd()`改为指定移动后的`applications/`文件夹的父文件夹。以下是一个`options.py`文件的示例：

```py
 import socket
import os
ip = '0.0.0.0'
port = 80
interfaces=[('0.0.0.0',80),
	('0.0.0.0',443,'ssl_key.pem','ssl_certificate.pem')]
password = '<recycle>' # <recycle> means use the previous password
pid_filename = 'httpserver.pid'
log_filename = 'httpserver.log'
profiler_filename = None
#ssl_certificate = 'ssl_cert.pem' # certificate file
#ssl_private_key = 'ssl_key.pem' # private key file
#numthreads = 50 # ## deprecated; remove
minthreads = None
maxthreads = None
server_name = socket.gethostname()
request_queue_size = 5
timeout = 30
shutdown_timeout = 5
folder = "/path/to/apps" # <<<<<<<< edit this line
extcron = None
nocron = None

```

当 web2py 与外部 web 服务器一起部署时，不可用应用程序移动功能。

## 如何操作...

1.  首先，创建一个 web2py 无权限用户：

    ```py
    sudo adduser web2py

    ```

1.  为了安全起见，禁用 web2py 用户密码以防止远程登录：

    ```py
    sudo passwd -l web2py

    ```

1.  从 web2py 的官方网站下载源代码包，将其解压到合适的目录中（例如`/opt/web2py`），并适当地设置访问权限：

    ```py
    wget http://www.web2py.com/examples/static/web2py_src.zip
    sudo unzip -x web2py_src.zip -d /opt
    sudo chown -Rv web2py. /opt/web2py

    ```

1.  在`/etc/inid.d/web2py`中创建一个`init`脚本（你可以使用`web2py/scripts/`中的作为起点）：

    ```py
    sudo cp /opt/web2py/scripts/web2py.ubuntu.sh /etc/init.d/web2py

    ```

1.  编辑`init`脚本：

    ```py
    sudo nano /etc/init.d/web2py

    ```

1.  设置基本配置参数：

    ```py
    PIDDIR=/opt/$NAME
    DAEMON_DIR=/opt/$NAME
    APPLOG_FILE=$DAEMON_DIR/web2py.log
    DAEMON_ARGS="web2py.py -p 8001 -i 127.0.0.1 -c server.crt -k
    server.key -a<recycle> --nogui --pid_filename=$PIDFILE -l \
    $APPLOG_FILE"

    ```

1.  将`127.0.0.1`和`8001`改为你想要的 IP 和端口。你可以使用`0.0.0.0`作为通配符 IP，以匹配所有接口。

1.  如果计划远程使用管理员权限，创建一个自签名证书：

    ```py
    sudo openssl genrsa -out /opt/web2py/server.key 1024
    sudo openssl req -new -key /opt/web2py/server.key -out /opt/\
    web2py/server.csr
    sudo openssl x509 -req -days 365 -in /opt/web2py/server.csr \
    -signkey /opt/web2py/server.key -out /opt/web2py/server.crt

    ```

1.  如果你使用`print`语句进行调试，或者想要记录 web2py 的输出消息，你可以在`web2py.py`中的`imports`之后添加以下行来重定向标准输出：

    ```py
    sys.stdout = sys.stderr = open("/opt/web2py/web2py.err","wa", 0)

    ```

1.  最后，启动你的 web2py 服务：

    ```py
    sudo /etc/init.d/web2py start

    ```

1.  要永久安装（使其与操作系统的其他服务一起自动启动和停止），请执行以下命令：

```py
sudo update-rc.d web2py defaults

```

如果一切正常，你将能够打开你的 web2py 管理员：

```py
https://127.0.0.1:8001/welcome/default/index

```

# 在 Ubuntu 上安装 web2py 作为服务

对于简单的网站和内部网络，你可能需要一个简单的安装方法，以保持 web2py 运行。这个菜谱展示了如何以简单的方式启动 web2py，而不需要进一步的依赖（没有 Apache Web 服务器！）。

## 还有更多...

你可以使用`bash`来调试`init`脚本，查看正在发生的事情：

```py
sudo bash -x /etc/init.d/web2py start

```

此外，你也可以更改`start-stop-daemon`选项以更详细地输出，并使用 web2py 用户来防止与其他 Python 守护进程的干扰：

```py
 start-stop-daemon --start \
	${DAEMON_USER:+--chuid $DAEMON_USER} --chdir $DAEMON_DIR \
	--background --user $DAEMON_USER --verbose --exec $DAEMON \
	--$DAEMON_ARGS || return 2

```

记住设置一个密码，以便能够使用管理界面。可以通过执行以下命令来完成（将`mypass`更改为你想要的密码）：

```py
sudo -u web2py python /opt/web2py/web2py.py -p 8001 -a mypasswd

```

# 使用 IIS 作为代理运行 web2py

IIS 是 Windows 操作系统的首选 Web 服务器。它可以运行多个并发域和多个应用程序池。当你将 web2py 部署到 IIS 上时，你想要设置一个新的站点，并为它的根应用程序创建一个单独的应用程序池。这样，你就有独立的日志和启动/停止应用程序池的能力，独立于其他应用程序池。以下是具体操作方法。

这是三个菜谱中的第一个，我们将使用不同的配置重复这个过程。在这个第一个菜谱中，我们将 IIS 设置为作为 web2py **Rocket** Web 服务器的代理。

当 IIS 默认站点已经处于生产状态，并且启用了 ASP.NET、ASP 或 PHP 应用程序时，这种配置是可取的，同时，你的 web2py 站点可能处于开发中，可能需要频繁重启（例如，由于`routes.py`中的更改）。

## 准备工作

在这个菜谱中，我们假设你已经安装了 IIS 7 或更高版本。我们不讨论安装 IIS7 的步骤，因为它是商业产品，并且它们在其他地方有很好的文档记录。

你还需要将 web2py 解压缩到本地文件夹中。在端口`8081`上启动 web2py。

```py
python web2py -p 8081 -i 127.0.0.1 -a 'password'

```

注意，当以代理方式运行 web2py 时，你应该小心不要无意中暴露未加密的 admin。

最后，你需要能够使用 IIS 代理。为此，你需要**应用程序请求路由（ARR）** **2.5**。ARR 可以从以下 Microsoft Web 平台安装程序下载和安装：

```py
http://www.microsoft.com/web/downloads/platform.aspx

```

## 如何操作...

1.  下载 ARR 的 Web 平台安装程序后，打开应用程序，浏览到屏幕左侧的**产品**，如下面的截图所示：

    ![如何操作...](img/5467OS_01_09.jpg)

1.  接下来，点击**添加** - **应用程序请求路由 2.5**，然后点击**安装**。这将带您到一个新屏幕，如下面的截图所示；点击**我接受**：

    ![如何操作...](img/5467OS_01_10.jpg)

1.  Web 平台安装程序将自动选择并安装**应用程序请求路由 2.5**运行所需的所有依赖项。点击**完成**，这将带您到**下载和安装**屏幕。

    ![如何操作...](img/5467OS_01_11.jpg)

1.  一旦收到成功消息，您可以关闭 Microsoft Web 平台应用程序。

1.  现在打开 IIS 管理器，并按照指示创建一个新的网站。

1.  首先，在**IIS 管理器**的左上角右键点击**网站**，然后选择**新建网站**。这将带您进入以下屏幕。按照此处所示填写详细信息：

    ![如何操作...](img/5467OS_01_12.jpg)

    +   确保您选择了您的网站将要运行的正确 IP。

1.  一旦创建了网站，双击以下截图所示的**URL 重写**：

    ![如何操作...](img/5467OS_01_13.jpg)

1.  一旦进入**URL 重写**模块，点击右上角的**添加规则**，如图所示。

1.  在**入站和出站规则**下选择**反向代理**模板。

1.  按照此处所示填写详细信息：

    ![如何操作...](img/5467OS_01_14.jpg)

1.  由于**服务器 IP**字段是最重要的，它必须包含 web2py 运行的 IP 和端口：`127.0.0.1:8081`。同时，请确保**SSL 卸载**被勾选。在**TO**字段的出站规则中，写入分配给网站的域名。完成后，点击**确定**。

    在这个阶段，你的 web2py 安装应该一切正常，除了管理界面。当非 localhost 服务器的请求指向管理界面时，Web2py 要求我们使用 HTTPS。在我们的例子中，web2py 的 localhost 是`127.0.0.1:8081`，而 IIS 目前运行在`127.0.0.1:80.`

1.  要启用管理，您需要一个证书。创建一个证书并将其添加到 IIS 7 的服务器证书中，然后重复之前的步骤将`443`绑定到之前创建的 web2py 网站。

1.  现在，访问：`https://yourdomain.com/admin/`，您将能够浏览 web2py 管理 Web 界面。输入您的 web2py 管理界面的密码，然后正常进行。

# 使用 ISAPI 运行 web2py

在这里，我们展示了一个生产质量的配置，它使用一个在 IIS 中本地运行的专用应用程序池，并使用 ISAPI 处理程序。它与典型的 Linux/Apache 配置类似，但它是 Windows 本地的。

## 准备工作

如前所述，您将需要安装 IIS。

您应该已经下载并解压了 web2py。如果您已经在 localhost 的**8081**（或其它端口）上运行，您可以保留它，因为它不应该干扰此安装。我们将假设 web2py 已安装到`C:\path\to\web2py`。

您可以将其放置在您喜欢的任何位置。

然后您需要下载并安装`isapi-wsgi`。这将在下面解释。

## 如何操作...

1.  首先，您需要从：[`code.google.com/p/isapi-wsgi/`](http://code.google.com/p/isapi-wsgi/)下载`isapi-wsgi`。

    它是基于 pywin32 的成熟 WSGI 适配器，大部分的配置基于`isapi-wsgi`的文档和示例。

    您可以使用 win32 安装程序安装 isapi-wsgi：`http://code.google.com/p/isapi-wsgi/downloads/detail?name=isapi_wsgi-0.4.2\. win32.exe.`

    你也可以简单地下载 Python 文件并将其放置在`"c:\Python\Lib\site-packages"`中安装它。

    [`isapi-wsgi.googlecode.com/svn/tags/isapi_wsgi-0.4.2/isapi_wsgi.py.`](http://isapi-wsgi.googlecode.com/svn/tags/isapi_wsgi-0.4.2/isapi_wsgi.py)

    `isapi_wsgi`在 IIS 5.1、6.0 和 7.0 上运行。但 IIS 7.x 必须安装**IIS 6.0 管理兼容性**。

    你可能想尝试运行以下测试来确认它已正确安装：

    ```py
    cd C:\Python\Lib\site-packages
    C:\Python\Lib\site-packages> python isapi_wsgi.py install
    Configured Virtual Directory: isapi-wsgi-test
    Extension installed
    Installation complete.

    ```

1.  现在转到`http://localhost/isapi-wsgi-test/`。

1.  如果你遇到一个显示“这不是一个有效的 Win32 应用程序”的`500 错误`，那么可能存在问题，这里进行了讨论：[`support.microsoft.com/kb/895976/en-us`](http://support.microsoft.com/kb/895976/en-us)。

1.  如果你看到一个正常的`Hello`响应，那么安装就成功了，你可以移除测试：

    ```py
    C:\Python\Lib\site-packages> python isapi_wsgi.py remove

    ```

    +   我们还没有准备好配置 web2py 处理器。你需要启用 32 位模式。

1.  我们现在准备好配置 web2py 处理器。将你的 web2py 安装添加到`PYTHONPATH:`。

    ```py
    set PYTHONPATH=%PYTHONPATH%;C:\path\to\web2py

    ```

1.  如果它还不存在，请在`C:\path\to\web2py`文件夹中创建一个名为`isapiwsgihandler.py`的文件，其中包含以下内容：

    ```py
     import os
    import sys
    import isapi_wsgi

    # The entry point for the ISAPI extension.
    def __ExtensionFactory__():
    	path = os.path.dirname(os.path.abspath(__file__))
    	os.chdir(path)
    	sys.path = [path]+[p for p in sys.path if not p==path]
    	import gluon.main
    	application = gluon.main.wsgibase
    	return isapi_wsgi.ISAPISimpleHandler(application)

    # ISAPI installation:
    if __name__=='__main__':
    	from isapi.install import ISAPIParameters
    	from isapi.install import ScriptMapParams
    	from isapi.install import VirtualDirParameters
    	from isapi.install import HandleCommandLine
    	params = ISAPIParameters()
    	sm = [ScriptMapParams(Extension="*", Flags=0)]
    vd = VirtualDirParameters(Name="appname",
    	Description = "Web2py in Python", ScriptMaps = sm,
    	ScriptMapUpdate = "replace")
    params.VirtualDirs = [vd]
    HandleCommandLine(params)

    ```

    +   web2py 的最近版本可能已经包含了这个文件，甚至是一个更好的版本。

1.  第一部分是处理器，第二部分将允许从命令行自动安装：

```py
 cd c:\path\to\web2py
python isapiwsgihandler.py install --server=sitename

```

+   默认情况下，这将在`Default Web Site`下的虚拟目录`appname`中安装扩展。

## 更多...

检查 Web 应用程序的当前模式（32 位或 64 位）：

```py
cd C:\Inetpub\AdminScripts
cscript.exe adsutil.vbs get W3SVC/AppPools/Enable32BitAppOnWin64
cscript %systemdrive%\inetpub\AdminScripts\adsutil.vbs get w3svc/\
AppPools/Enable32bitAppOnWin64

```

如果答案是`"Enable32BitAppOnWin64"参数在此节点未设置`或`Enable32BitAppOnWin64 : (BOOLEAN) False`，那么你必须将 Web 服务器从 64 位模式切换到 32 位模式。ISAPI 在 64 位模式的 IIS 上不工作。你可以使用以下命令进行切换：

```py
cscript %systemdrive%\inetpub\AdminScripts\adsutil.vbs set w3svc/\
AppPools/Enable32bitAppOnWin64 1

```

然后按照以下步骤重新启动应用程序池：

```py
IIsExt /AddFile %systemroot%\syswow64\inetsrv\httpext.dll 1 ^
WEBDAV32 1 "WebDAV (32-bit)"

```

或者按照以下步骤设置一个单独的池：

```py
system.webServer/applicationPool/add@enable32BitAppOnWin64.

```
