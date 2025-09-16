# *第十五章*：部署和维护生产实例

在本章中，您将学习如何为生产环境准备 Odoo 服务器的基础知识。

设置和维护服务器本身就是一个非平凡的话题，应由专业人士完成。这里提供的信息不足以确保普通用户能够创建一个能够承载敏感数据和服务的弹性且安全的环境。

本章的目标是介绍最重要的配置方面和针对 Odoo 部署的最佳实践。这将帮助系统管理员为他们的 Odoo 服务器主机做好准备。

您将首先设置主机系统，然后安装 Odoo 的先决条件和 Odoo 本身。**Ubuntu** 是云服务器的流行选择，这里将使用它。然后，需要准备 Odoo 配置文件。到目前为止，设置与开发环境使用的设置类似。

接下来，需要将 Odoo 配置为系统服务，以便在服务器启动时自动启动。

对于托管在公共云上的服务器，Odoo 应通过 **HTTPS** 提供服务。为此，您将学习如何使用自签名证书安装和配置 **Nginx** 反向代理。

最后一个部分讨论了如何执行服务器升级并准备一个预演环境，以便在实际更新应用之前进行测试运行。

本章讨论的主题如下：

+   准备主机系统

+   从源代码安装 Odoo

+   配置 Odoo

+   将 Odoo 配置为系统服务

+   设置 Nginx 反向代理

+   配置和强制执行 HTTPS

+   维护 Odoo 服务和模块

到本章结束时，您将能够设置一个相当安全的 Odoo 服务器，这对于低调的生产使用已经足够好。然而，本章中给出的食谱并不是部署 Odoo 的唯一有效方法——还有其他方法也是可能的。

# 技术要求

要跟随本章，您需要一个干净的 Ubuntu 20.04 服务器——例如，一个在云上托管的 **虚拟专用服务器**（**VPS**）。

本章中使用的代码和脚本可以在 [`github.com/PacktPublishing/Odoo-15-Development-Essentials`](https://github.com/PacktPublishing/Odoo-15-Development-Essentials) 的 **GitHub** 仓库的 `ch15/` 目录中找到。

# 准备主机系统

Odoo 通常部署在基于 **Debian** 的 **Linux** 系统上。Ubuntu 是一个流行的选择，最新的 **长期支持**（**LTS**）版本是 20.04 **Focal Fossa**。

其他 Linux 发行版也可以使用。在商业领域，**CentOS**/**Red Hat Enterprise Linux**（**RHEL**）系统也很受欢迎。

安装过程需要提升访问权限，使用`root`超级用户或`sudo`命令。在 Debian 发行版中，默认登录是`root`，它具有管理访问权限，命令提示符显示`#`。在 Ubuntu 系统上，`root`账户被禁用。相反，在安装过程中配置了初始用户，并且是`sudo`命令来提升访问权限并使用`root`权限运行命令。

在开始 Odoo 安装之前，必须在主机系统上安装系统依赖，并创建一个特定的用户来运行 Odoo 服务。

下一个部分解释了在 Debian 系统上所需的系统依赖。

## 安装系统依赖

当从源运行 Odoo 时，需要安装一些系统依赖。

在开始之前，更新软件包索引并执行升级以确保所有已安装的程序都是最新的，这是一个好习惯，如下所示：

```py
$ sudo apt update
$ sudo apt upgrade -y
```

接下来，可以安装**PostgreSQL**数据库。我们的用户应该被设置为数据库超级用户，以便他们能够获得对数据库的管理访问权限。以下是这些命令：

```py
$ sudo apt install postgresql -y
$ sudo su -c "createuser -s $USER" postgres
```

注意

Odoo 可以使用安装在它自己的服务器上的现有 PostgreSQL 数据库。如果是这种情况，则不需要在 Odoo 服务器上安装 PostgreSQL 服务，并且应在 Odoo 配置文件中设置相应的连接详情。

这些是运行 Odoo 所需的 Debian 依赖项：

```py
$ sudo apt install git python3-dev python3-pip \
python3-wheel python3-venv -y
$ sudo apt install build-essential libpq-dev libxslt-dev \
libzip-dev libldap2-dev libsasl2-dev libssl-dev
```

为了拥有报表打印功能，必须安装`wkhtmltox`。对于 Odoo 10 及以后的版本，推荐的版本是 0.12.5-1。下载链接可以在[`github.com/wkhtmltopdf/wkhtmltopdf/releases/tag/0.12.5`](https://github.com/wkhtmltopdf/wkhtmltopdf/releases/tag/0.12.5)找到。Ubuntu 的**代号**对于 18.04 版本是**bionic**，对于 20.04 版本是**focal**。

以下命令为 Ubuntu 20.04 Focal 版本执行此安装：

```py
$ wget "https://github.com/wkhtmltopdf/wkhtmltopdf\
/releases""/download/0.12.5/\
wkhtmltox_0.12.5-1.focal_amd64.deb" \
-O /tmp/wkhtml.deb
$ sudo dpkg -i /tmp/wkhtml.deb
$ sudo apt-get -fy install  # Fix dependency errors
```

软件包安装可能会报告缺少依赖项错误。在这种情况下，最后一个命令将强制安装这些依赖项并正确完成安装。

接下来，你将创建一个系统用户用于 Odoo 进程。

## 准备专用系统用户

一个好的安全实践是使用一个专用用户来运行 Odoo，该用户在主机系统上没有特殊权限。

对于用户名的一个流行选择是`odoo`。这是创建它的命令：

```py
$ sudo adduser --home=/opt/odoo --disabled-password \
--gecos "Odoo" odoo
```

Linux 系统用户可以有一个`家`目录。对于 Odoo 用户来说，这是一个方便的地方来存储 Odoo 文件。这个选择的流行选项是`/opt/odoo`。自动使用的`--home`选项会创建这个目录并将其设置为`odoo`用户的家目录。

此用户目前还没有访问 PostgreSQL 数据库的权限。以下命令添加了这种访问权限并为它创建数据库以初始化 Odoo 生产环境：

```py
$ sudo su -c "createuser odoo" postgres
$ createdb --owner=odoo odoo-prod
```

在这里，`odoo` 是用户名，`odoo-prod` 是支持我们的 Odoo 实例的数据库名称。`odoo` 用户被设置为 `odoo-prod` 数据库的所有者。这意味着它对该数据库具有 *创建和删除* 权限，包括删除它的能力。

小贴士

要运行，Odoo 不需要使用数据库的特权权限。这些权限可能仅在某些维护操作中需要，例如安装或升级模块。因此，为了提高安全性，Odoo 系统用户可以是非所有者数据库用户。请注意，在这种情况下，维护应使用与数据库所有者不同的用户运行 Odoo。

要使用 Odoo 系统用户启动会话，请使用以下命令：

```py
$ sudo su - odoo
$ exit
```

这将用于以 Odoo 用户身份运行安装步骤。完成后，`exit` 命令将终止该会话并返回到原始用户。

在下一节中，我们将继续安装 Odoo 代码和 `/opt/odoo` 目录。

# 从源代码安装 Odoo

虽然 Odoo 提供了 Debian/Ubuntu 和 CentOS/RHEL 系统包，但由于其提供的灵活性和控制，从源代码安装是一个流行的选项。

使用源代码可以更好地控制部署的内容，并在生产环境中更容易地管理更改和修复。例如，它允许我们将部署过程与 Git 工作流程相关联。

到目前为止，Odoo 的系统依赖项已经安装，数据库已准备好使用。现在，可以下载并安装 Odoo 源代码，以及所需的 Python 依赖项。

让我们看看如何下载 Odoo 源代码。

## 下载 Odoo 源代码

总有一天，您的服务器将需要升级和补丁。在需要的时候，版本控制仓库可以提供极大的帮助。我们使用 `git` 从仓库获取代码，就像我们在安装开发环境时做的那样。

接下来，我们将模拟 `odoo` 用户，并将代码下载到其主目录中，如下所示：

```py
$ sudo su - odoo
$ git clone https://github.com/odoo/odoo.git \
/opt/odoo/odoo15 \
-b 15.0 --depth=1
```

`-b` 选项确保我们获取正确的分支，而 `--depth=1` 选项仅检索最新的代码修订版，忽略（长）变更历史，使下载更小、更快。

小贴士

**Git** 是管理 Odoo 部署代码版本的重要工具。如果您不熟悉 Git，值得了解更多关于它的信息。一个好的起点是 [`git-scm.com/doc`](http://git-scm.com/doc)。

自定义模块通常也会使用 Git 管理，并且也应该克隆到生产服务器上。例如，以下代码将库自定义模块添加到 `/opt/odoo/odoo15/library` 目录：

```py
$ git clone https://github.com/PacktPublishing/Odoo-15-Development-Essentials/opt/odoo/library
```

Odoo 源代码已位于服务器上，但还不能运行，因为所需的 Python 依赖项尚未安装。让我们在下一节中安装这些依赖项。

## 安装 Python 依赖项

下载 Odoo 源代码后，应安装 Odoo 所需的 Python 包。

其中许多也都有 Debian 或 Ubuntu 系统包。官方 Odoo Debian 安装包使用它们，依赖包的名称可以在 Odoo 源代码的 `debian/control` 文件中找到：[`github.com/odoo/odoo/blob/15.0/debian/control`](https://github.com/odoo/odoo/blob/15.0/debian/control)。

这些 Python 依赖项也可以直接从 **Python 包索引**（**PyPI**）安装。使用 Python **虚拟环境**来做这件事可以更好地保护主机系统免受更改的影响。

以下命令创建一个虚拟环境，激活它，然后从源代码安装 Odoo 以及所有必需的 Python 依赖项：

```py
$ python3 -m venv /opt/odoo/env15
$ source /opt/odoo/env15/bin/activate
(env15) $ pip install -r /opt/odoo/odoo15/requirements.txt
(env15) $ pip install -e /opt/odoo/odoo15
```

现在，Odoo 应该已经准备好了。可以使用以下任何命令来确认这一点：

```py
(env15) $ odoo --version
Odoo Server 15.0
(env15) $ /opt/odoo/odoo15/odoo-bin --version
Odoo Server 15.0
$ /opt/odoo/env15/bin/python3 /opt/odoo/odoo15/odoo-bin --version
Odoo Server 15.0
$ /opt/odoo/env15/bin/odoo --version
Odoo Server 15.0
```

让我们逐个理解这些命令：

+   第一个命令依赖于由 `pip install -e /opt/odoo/odoo15` 提供的 `odoo` 命令。

+   第二个命令不依赖于 `odoo` 命令，它直接调用 Odoo 启动脚本，`/opt/odoo/odoo15/odoo-bin`。

+   第三个命令不需要事先激活虚拟环境，因为它直接使用相应的 Python 可执行文件，这具有相同的效果。

+   最后一个命令以更紧凑的方式执行相同的操作。它直接使用该虚拟环境中可用的 `odoo` 命令。这对某些脚本可能很有用。

现在 Odoo 已经准备好运行了。下一步是注意要使用的配置文件，我们将在下一节中解释。

# 配置 Odoo

一旦安装了 Odoo，就需要准备用于生产服务的配置文件。

下一个子节提供了如何做到这一点的指导。

## 设置配置文件

预期配置文件位于 `/etc` 系统目录中。因此，Odoo 生产配置文件将存储在 `/etc/odoo/odoo.conf`。

为了更容易地看到所有可用的选项，可以生成一个默认配置文件。这应该由将运行该服务的用户来完成。

如果尚未完成，为 `odoo` 用户创建一个会话并激活虚拟环境：

```py
$ sudo su - odoo
$ python3 -m venv /opt/odoo/env15
```

现在，可以使用以下命令创建一个默认配置文件：

```py
(env15) $ odoo -c /opt/odoo/odoo.conf --save --stop-after-init
```

在上一个命令中，`-c` 选项设置配置文件的位置。如果没有提供，则默认为 `~/.odoorc`。`--save` 选项将选项写入其中。如果文件不存在，它将使用所有默认选项创建。如果它已经存在，它将使用命令中使用的选项进行更新。

以下命令设置了该文件的一些重要选项：

```py
(env15) $ odoo -c /opt/odoo/odoo.conf --save \
--stop-after-init \
-d odoo-prod --db-filter="^odoo-prod$" \
--without-demo=all --proxy-mode
```

设置的选项如下：

+   `-d`: 这是默认要使用的数据库。

+   `--db-filter`: 这是一个正则表达式，用于过滤 Odoo 服务可用的数据库。使用的表达式仅使 `odoo-prod` 数据库可用。

+   `--without-demo=all`: 这将禁用演示数据，以便 Odoo 初始化数据库从零开始。

+   `--proxy-mode`：这启用了代理模式，意味着 Odoo 应该期望来自反向代理的请求。

下一步是将此默认文件复制到 `/etc` 目录，并设置必要的访问权限，以便 Odoo 用户可以读取它：

```py
$ exit  # exit from the odoo user session
$ sudo mkdir /etc/odoo
$ sudo cp /opt/odoo/odoo.conf /etc/odoo/odoo.conf
$ sudo chown -R odoo /etc/odoo
$ sudo chmod u=r,g=rw,o=r /etc/odoo/odoo.conf  # for extra hardening
```

最后一条命令确保运行 Odoo 进程的用户可以读取但不能更改配置文件，从而提供更好的安全性。

还需要创建 Odoo 日志文件目录，并授予 `odoo` 用户访问权限。这应该在 `/var/log` 目录内完成。以下命令可以完成此操作：

```py
$ sudo mkdir /var/log/odoo
$ sudo chown odoo /var/log/odoo
```

最后，应该编辑 Odoo 配置文件，以确保一些重要的参数被正确配置。例如，以下命令使用 `nano` 编辑器打开文件：

```py
$ sudo nano /etc/odoo/odoo.conf
```

这些是一些最重要的参数的建议值：

```py
[options]
addons_path = /opt/odoo/odoo15/odoo/addons,/opt/odoo/odoo15/addons,/opt/odoo/library
admin_passwd = StrongRandomPassword
db_name = odoo-prod
dbfilter = ^odoo-prod$
http_interface = 127.0.0.1
http_port = 8069
limit_time_cpu = 600
limit_time_real = 1200
list_db = False
logfile = /var/log/odoo/odoo-server.log
proxy_mode = True
without_demo = all
workers = 6
```

让我们详细解释一下：

+   `addons_path`：这是一个逗号分隔的路径列表，其中将查找附加模块。它从左到右读取，最左边的目录被视为优先级更高。

+   `admin_passwd`：这是用于访问网络客户端数据库管理功能的密码。使用强密码设置此密码至关重要，或者更好的是将其设置为 `False` 以禁用此功能。

+   `db_name`：这是在服务器启动序列中初始化的数据库实例。

+   `dbfilter`：这是用于使数据库可访问的过滤器。它是一个 Python 解释的正则表达式表达式。为了用户不被提示选择数据库，并且未认证的 URL 能够正常工作，它应该设置为 `^dbname$`，例如，`dbfilter=^odoo-prod$`。它支持 `%h` 和 `%d` 占位符，它们将被 HTTP 请求的主机名和子域名名称替换。

+   `http_interface`：这是 Odoo 将监听的 TCP/IP 地址。默认情况下，它是 `0.0.0.0`，意味着所有地址。对于位于反向代理后面的部署，可以将它设置为反向代理地址，以便只考虑来自那里的请求。如果反向代理与 Odoo 服务在同一服务器上，请使用 `127.0.0.1`。

+   `http_port`：这是服务器将监听的端口号。默认情况下，使用端口号 `8069`。

+   `limit_time_cpu` / `limit_time_real`：这为工作者设置了 CPU 时间限制。默认设置 `60` 和 `120` 可能太低，可能需要将它们提高。

+   `list_db = False`：这阻止了数据库列表，无论是在远程过程调用（RPC）级别还是在 UI 中，它还阻止了数据库管理屏幕和底层的 RPC 函数。

+   `logfile`：这是服务器日志应该写入的位置。对于系统服务，预期的位置是在 `/var/log` 内的某个地方。如果为空，则日志将打印到标准输出。

+   `proxy_mode`：当 Odoo 通过反向代理访问时，应该将其设置为 `True`，正如我们将要做的。

+   `without_demo`：在生产环境中，此选项应设置为`all`，以便新数据库上不包含演示数据。

+   `workers`：此选项，当值为两个或更多时，启用多进程模式。我们将在稍后详细讨论这一点。

从安全角度考虑，`admin_passwd`和`list_db=False`选项尤为重要。它们阻止对数据库管理功能的 Web 访问，并且应在任何生产或面向互联网的 Odoo 服务器上设置。

小贴士

可以使用`openssl rand -base64 32`命令在命令行中生成随机密码。将`32`数字更改为您喜欢的密码大小。

以下参数也可能很有帮助：

+   `data_dir`：这是会话数据和附件文件存储的路径；请记住备份此目录。

+   `http_interface`：此选项设置将监听的地址。默认情况下，它监听`0.0.0.0`，但在使用反向代理时，可以将其设置为`127.0.0.1`以仅响应本地请求。

我们可以通过以下方式运行 Odoo 手动检查配置的效果：

```py
$ sudo su - odoo
$ source /opt/odoo/env15/bin/activate
$ odoo -c /etc/odoo/odoo.conf
```

最后一条命令不会在控制台显示任何输出，因为日志消息正在写入日志文件而不是标准输出。

要跟踪运行中的 Odoo 服务器的日志，可以使用`tail`命令：

```py
$ tail -f /var/log/odoo/odoo-server.log
```

这可以在运行手动命令的原终端窗口之外的不同终端窗口中完成。

要在同一终端窗口中运行多个终端会话，可以使用`tmux`或`screen`。Ubuntu 还提供了`tmux`或`screen`。有关更多详细信息，请参阅[`help.ubuntu.com/community/Byobu`](https://help.ubuntu.com/community/Byobu)。

注意

不幸的是，无法直接从 Odoo 命令中取消`logfile`配置选项。如果我们想暂时将日志输出发送回标准输出，最佳解决方案是使用未设置`logfile`选项的配置文件副本。

可能的情况是`odoo-prod`数据库尚未由 Odoo 初始化，这需要手动完成。在这种情况下，可以通过安装`base`模块来完成初始化：

```py
$ /opt/odoo/env15/bin/odoo -c /etc/odoo/odoo.conf -i base \
--stop-after-init
```

到目前为止，Odoo 配置应该已经准备好了。在继续之前，值得了解更多关于 Odoo 中的多进程工作者的信息。

## 理解多进程工作者

预期的生产实例应处理大量的工作负载。由于 Python 语言的**全局解释器锁**（**GIL**），默认情况下，服务器运行一个进程，并且只能使用一个 CPU 核心进行处理。然而，有一个多进程模式可供使用，以便可以处理并发请求，从而让我们可以利用多个核心。

`workers=N` 选项设置要使用的工人数。作为一个指导原则，它可以设置为 `1+2*P`，其中 `P` 是处理器核心数。找到最佳设置可能需要通过使用不同的数字并检查服务器处理器的繁忙程度进行一些实验。在同一台机器上运行 PostgreSQL 也会对此产生影响，这将减少应该启用的工人数。

对于负载过高的情况，设置过多的工人数比设置过少的情况更好。最小值应该是六个，因为大多数浏览器使用的并行连接数。最大值通常由机器上的 RAM 量限制，因为每个工人都将消耗一些服务器内存。对于正常的使用模式，Odoo 服务器应该能够处理 `(1+2*P)*6` 个并发用户，其中 `P` 是处理器的数量。

有几个 `limit-` 配置参数可以用来调整工作参数。当工人数达到这些限制时，会回收工人数，相应的进程将被停止并启动一个新的进程。这可以保护服务器免受内存泄漏和特定进程过载服务器资源的影响。

官方文档提供了关于如何调整工作参数的额外建议。可以在 [`www.odoo.com/documentation/15.0/setup/deploy.html#builtin-server`](https://www.odoo.com/documentation/15.0/setup/deploy.html#builtin-server) 找到。

到目前为止，Odoo 已经安装、配置并准备好运行。下一步是让它作为无人值守的系统服务运行。我们将在下一节中详细探讨这一点。

# 设置 Odoo 为系统服务

Odoo 应该作为系统服务运行，以便在系统启动时自动启动并无人值守运行，不需要用户会话。

在 Debian/Ubuntu 系统中，`init` 系统负责启动服务。历史上，Debian 及其衍生操作系统使用 `sysvinit`。但现在已经改变，最近的 Debian/Ubuntu 系统使用 `systemd`。这同样适用于 Ubuntu 16.04 及以后的版本。

要确认您的系统中使用的是 `systemd`，请尝试以下命令：

```py
$ man init
```

此命令将打开当前正在使用的 `init` 系统的文档，以便您可以检查正在使用的内容。在手册页的顶部，您应该看到提到了 `SYSTEMD`。

让我们继续配置 `systemd` 服务。

## 创建 systemd 服务

如果操作系统较新，例如 Debian 8 和 Ubuntu 16.04 或更新的版本，`systemd` 应该是正在使用的 `init` 系统。

要向系统中添加新的服务，只需创建一个描述它的文件。创建一个包含以下内容的 `/lib/systemd/system/odoo.service` 文件：

```py
[Unit]
Description=Odoo Open Source ERP and CRM
After=network.target
[Service]
Type=simple
User=odoo
Group=odoo
ExecStart=/opt/odoo/env/bin/odoo -c /etc/odoo/odoo.conf --log-file=/var/log/odoo/odoo-server.log
KillMode=mixed
[Install]
WantedBy=multi-user.target
```

此服务配置文件基于 Odoo 源代码中提供的示例，可以在 [`github.com/odoo/odoo/blob/15.0/debian/odoo.service`](https://github.com/odoo/odoo/blob/15.0/debian/odoo.service) 找到。`ExecStart` 选项应调整为此系统要使用的特定路径。

接下来，可以使用以下命令将新服务注册：

```py
$ sudo systemctl enable odoo.service
```

要启动这个新服务，请运行以下命令：

```py
$ sudo systemctl start odoo
```

要检查其状态，请使用以下命令：

```py
$ sudo systemctl status odoo
```

可以使用以下命令停止它：

```py
$ sudo systemctl stop odoo
```

当以系统服务运行 Odoo 时，确认客户端可以访问它是有用的。让我们看看如何在命令行中做到这一点。

## 从命令行检查 Odoo 服务

要确认 Odoo 服务运行良好且响应迅速，我们可以检查它是否正在响应请求。我们应该能够从它那里获得响应，并在日志文件中看不到错误。

我们可以使用以下命令检查 Odoo 是否在服务器内部响应 HTTP 请求：

```py
$ curl http://localhost:8069
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<title>Redirecting...</title>
<h1>Redirecting...</h1>
<p>You should be redirected automatically to target URL: <a href="/web">/web</a>.  If not click the link.
```

此外，要查看`log`文件中的内容，请使用以下命令：

```py
$ less /var/log/odoo/odoo-server.log
```

要实时跟踪添加到日志文件的内容，可以使用`tail -f`，如下所示：

```py
$ tail -f /var/log/odoo/odoo-server.log
```

Odoo 现在已安装并作为服务运行。接下来，可以通过添加反向代理来改进设置。下一节将解释这一点。

# 设置 Nginx 反向代理

虽然 Odoo 本身可以提供网页服务，但建议在其前面设置一个反向代理。反向代理接收来自客户端的流量，然后将它转发到响应客户端的 Odoo 服务器。这样做有几个好处。

在安全方面，它可以提供以下功能：

+   处理（并强制执行）HTTPS 协议以加密流量。

+   隐藏内部网络特征。

+   充当应用程序防火墙，限制接受处理的 URL。

在性能方面，它可以提供以下功能：

+   缓存静态内容，避免让 Odoo 服务承受这些请求的负担，从而减少其负载。

+   压缩内容以加快加载时间。

+   充当负载均衡器，在多个 Odoo 服务之间分配负载。

有几种选项可以作为反向代理使用。历史上，**Apache**是一个流行的选择。近年来，Nginx 已被广泛使用，并在 Odoo 官方文档中提到。在我们的示例中，将使用 Nginx 进行反向代理，并使用它实现所提供的安全和性能功能。

首先，Nginx 应该被安装并设置为监听默认的 HTTP 端口。可能这个端口已经被另一个已安装的服务占用。为确保端口空闲且可用，请使用以下命令，它应该导致错误：

```py
$ curl http://localhost
curl: (7) Failed to connect to localhost port 80: Connection refused
```

如果它没有返回之前的错误消息，则已安装的服务正在使用端口`80`，应该被禁用或卸载。

例如，如果已安装 Apache 服务器，请使用`sudo service apache2 stop`命令停止它，或者甚至使用`sudo apt remove apache2`命令将其卸载。

在端口`80`空闲的情况下，可以安装和配置 Nginx。以下命令安装 Nginx：

```py
$ sudo apt-get install nginx
$ sudo service nginx start  # start nginx, if not already started
```

要确认`nginx`运行正确，请使用浏览器访问服务器地址或使用服务器上的`curl http://localhost`命令。这应该返回一个**欢迎来到 nginx**页面。

Nginx 配置文件存储在`/etc/nginx/available-sites/`，通过将它们添加到`/etc/nginx/enabled-sites/`来激活，这通常是通过在可用站点目录中的文件创建符号链接来完成的。

为了准备 Odoo Nginx 配置，应删除默认配置并添加 Odoo 配置文件，如下所示：

```py
$ sudo rm /etc/nginx/sites-enabled/default
$ sudo touch /etc/nginx/sites-available/odoo
$ sudo ln -s /etc/nginx/sites-available/odoo \
/etc/nginx/sites-enabled/odoo
```

接下来，使用`nano`或`vi`等编辑器，按照以下方式编辑配置文件：

```py
$ sudo nano /etc/nginx/sites-available/odoo
```

以下示例提供了一个基本的 Nginx 配置，用于 Odoo：

```py
upstream odoo {
  server 127.0.0.1:8069;
}
upstream odoochat {
  server 127.0.0.1:8072;
}
server {
  listen 80;
  server_name odoo.mycompany.com;
  proxy_read_timeout 720s;
  proxy_connect_timeout 720s;
  proxy_send_timeout 720s;
  # Add Headers for odoo proxy mode
  proxy_set_header X-Forwarded-Host  $host;
  proxy_set_header X-Forwarded-For   $proxy_add_x_
    forwarded_for;
  proxy_set_header X-Forwarded-Proto $scheme;
  proxy_set_header X-Real-IP         $remote_addr;
  # log
  access_log /var/log/nginx/odoo.access.log;
  error_log /var/log/nginx/odoo.error.log;
  # Redirect longpoll requests to odoo longpolling port
  location /longpolling {
    proxy_pass http://odoochat;
  }
  # Redirect requests to odoo backend server
  location / {
    proxy_redirect off;
    proxy_pass http://odoo;
  }
  # common gzip
  gzip_types text/css text/scss text/plain text/xml 
   application/xml application/json application/javascript;
  gzip on;
}
```

在配置文件顶部，有`upstream`配置部分。这些指向默认监听端口`8069`和`8072`的 Odoo 服务。`8069`端口服务于 Web 客户端和 RPC 请求，而`8072`服务于即时消息功能使用的长轮询请求。

`server`配置部分定义了在`80`默认 HTTP 端口上接收到的流量将发生什么。在这里，它通过`proxy_pass`配置指令重定向到上游 Odoo 服务。任何针对`/longpolling`地址的流量都会传递给`odoochat`上游，而剩余的`/`流量会传递给`odoo`上游。

几个`proxy_set_header`指令向请求头添加信息，以便让 Odoo 后端服务知道它正在被代理。

小贴士

由于安全原因，确保 Odoo 的`proxy_mode`参数设置为`True`非常重要。这样做的原因是，在 Nginx 中，所有击中 Odoo 的请求都来自 Nginx 服务器，而不是原始的远程 IP 地址。在代理中设置`X-Forwarded-For`头并启用`--proxy-mode`允许 Odoo 了解请求的原始来源。请注意，在没有在代理级别强制设置头的情况下启用`--proxy-mode`允许恶意客户端伪造其请求地址。

在配置文件的末尾，可以找到一些与`gzip`相关的指令。这些指令启用了某些文件的压缩，从而提高了性能。

一旦编辑并保存，可以使用以下命令验证 Nginx 配置的正确性：

```py
$ sudo nginx -t
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

现在，可以使用以下命令之一重新加载 Nginx 服务的新配置，具体取决于使用的`init`系统：

```py
$ sudo /etc/init.d/nginx reload
$ sudo systemctl reload nginx  # using systemd
$ sudo service nginx reload  # on Ubuntu systems
```

这将使 Nginx 在不中断服务的情况下重新加载配置，而如果使用`restart`而不是`reload`，则可能会发生中断。

为了确保安全，应通过 HTTPS 访问 Odoo。下一节将讨论这个问题。

# 配置和强制执行 HTTPS

网络流量不应以纯文本形式通过互联网传输。当在网络上公开 Odoo 服务器时，应使用 HTTPS 加密流量。

在某些情况下，使用自签名证书可能是可接受的。请记住，使用自签名证书提供有限的安全性。虽然它允许加密流量，但它有一些安全限制，例如无法防止中间人攻击，或者无法在最新的网页浏览器上显示安全警告。

一个更稳健的解决方案是使用由认可机构签发的证书。这在运行电子商务网站时尤为重要。另一个选项是使用 **Let's Encrypt** 证书，**Certbot** 程序可以自动化获取该证书的 SSL 证书。有关更多信息，请参阅 [`certbot.eff.org/instructions`](https://certbot.eff.org/instructions)。

接下来，我们将看到如何创建自签名证书，以防这是首选的选择。

## 创建自签名 SSL 证书

Nginx 需要安装一个证书来启用 SSL。我们可以选择使用证书机构提供的证书，或者生成一个自签名的证书。

要创建自签名证书，请使用以下命令：

```py
$ sudo mkdir /etc/ssl/nginx && cd /etc/ssl/nginx
$ sudo openssl req -x509 -newkey rsa:2048 \
-keyout server.key -out server.crt -days 365 -nodes
$ sudo chmod a-wx *            # make files read only
$ sudo chown www-data:root *   # access only to www-data group
```

上述代码创建了一个 `/etc/ssl/nginx` 目录和一个无密码的 SSL 证书。当运行 `openssl` 命令时，用户将被要求提供一些额外的信息，然后生成证书和密钥文件。最后，这些文件的拥有权被赋予 `www-data` 用户，该用户用于运行网页服务器。

准备好要使用的 SSL 证书后，下一步是将它安装到 Nginx 服务上。

## 在 Nginx 上配置 HTTPS 访问

为了强制使用 HTTPS，需要一个 SSL 证书。Nginx 服务将使用它来加密服务器和网页浏览器之间的流量。

对于这一点，需要重新检查 Odoo Nginx 的配置文件。编辑它，将 `server` 指令替换为以下内容：

```py
server {
  listen 80;
  rewrite ^(.*) https://$host$1 permanent;
}
```

通过这个更改，对 `http://` 地址的请求将被转换为 `https://` 相应的地址，确保不会意外地使用非安全传输。

HTTPS 服务仍然需要配置。这可以通过向配置中添加以下 `server` 指令来完成：

```py
# odoo server
upstream odoo {
  server 127.0.0.1:8069;
}
upstream odoochat {
  server 127.0.0.1:8072;
}
# http -> https
server {
  listen 80;
  server_name odoo.mycompany.com;
  rewrite ^(.*) https://$host$1 permanent;
}
server {
  listen 443;
  server_name odoo.mycompany.com;
  proxy_read_timeout 720s;
  proxy_connect_timeout 720s;
  proxy_send_timeout 720s;
  # Add Headers for odoo proxy mode
  proxy_set_header X-Forwarded-Host $host;
  proxy_set_header X-Forwarded-For $proxy_add_x_for
    warded_for;
  proxy_set_header X-Forwarded-Proto $scheme;
  proxy_set_header X-Real-IP $remote_addr;
  # SSL parameters
  ssl on;
  ssl_certificate /etc/ssl/nginx/server.crt;
  ssl_certificate_key /etc/ssl/nginx/server.key;
  ssl_session_timeout 30m;
  ssl_protocols TLSv1.2;
  ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-
    AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-
    RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-
    POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-
    GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
  ssl_prefer_server_ciphers off;
  # log
  access_log /var/log/nginx/odoo.access.log;
  error_log /var/log/nginx/odoo.error.log;
  # Redirect longpoll requests to odoo longpolling port
  location /longpolling {
    proxy_pass http://odoochat;
  }
  # Redirect requests to odoo backend server
  location / {
    proxy_redirect off;
    proxy_pass http://odoo;
  }
  # common gzip
  gzip_types text/css text/scss text/plain text/xml 
   application/xml application/json application/javascript;
  gzip on;
}
```

这个额外的 `server` 指令监听 HTTPS 端口，并使用 `/etc/ssl/nginx/` 下的证书文件来加密流量。

注意

这里提出的 Nginx 配置基于在 [`www.odoo.com/documentation/15.0/administration/install/deploy.html#https`](https://www.odoo.com/documentation/15.0/administration/install/deploy.html#https) 找到的官方文档。

一旦重新加载此配置，Odoo 应该只通过 HTTPS 运行，如下面的命令所示：

```py
$ sudo nginx -t
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
$ sudo service nginx reload  # or: sudo systemctl reload nginx
* Reloading nginx configuration nginx
...done.
$ curl -k https://localhost
```

加密网络流量并不是 Nginx 为我们做的唯一事情。它还可以帮助我们减少 Odoo 上游服务的负载。让我们在下一节中详细探讨这一点。

## 缓存静态内容

Nginx 可以缓存提供的静态文件——这意味着后续对缓存文件的请求将直接由 Nginx 提供，无需上游 Odoo 服务请求。

这不仅提高了响应时间，还提高了 Odoo 服务能力，以服务更多用户，因为它现在专注于响应动态请求。

要启用静态内容缓存，请在 Nginx 配置文件中`# comming gzip`指令之后添加以下部分：

```py
  # cache static data
  location ~* /web/static/ {
    proxy_cache_valid 200 60m;
    proxy_buffering on;
    expires 864000;
    proxy_pass http://odoo;
  }
```

使用此配置，静态数据将缓存 60 分钟。Odoo 静态内容定义为从`/web/static`路径提供的任何文件。

到这一点，服务器应该完全功能，Nginx 通过 HTTPS 处理请求，然后将它们传递给 Odoo 服务进行处理。

Odoo 服务需要维护和更新，所以下一节将讨论如何进行此操作。

# 维护 Odoo 服务和模块

一旦 Odoo 服务器启动并运行，预计需要一些维护工作——例如，安装或更新模块。

这些操作涉及生产系统的一些风险，因此在生产环境中应用之前，最好在预发布环境中进行测试。让我们从一个基本的配方开始，创建一个预发布环境。

## 创建预发布环境

预发布环境应该是生产系统的副本，理想情况下应该有自己的专用服务器。

一种简化方法，对于大多数情况来说足够安全，是将预发布环境放在与生产系统相同的服务器上。

要将`odoo-prod`生产数据库的副本作为`odoo-stage`数据库创建，请使用以下命令：

```py
$ dropdb odoo-stage
$ createdb --owner=odoo odoo-stage
$ pg_dump odoo-prod | psql -d odoo-stage
$ sudo su - odoo
$ cd ~/.local/share/Odoo/filestore/
$ cp -r odoo-prod odoo-stage
$ exit
```

注意，一些配置被复制过来，例如连接到电子邮件服务器，您可能希望有额外的命令来禁用它们。具体需要采取的操作取决于数据库设置，但很可能可以通过脚本自动化。为此，了解`psql`命令可以直接从命令行运行 SQL 很有用，例如，`psql -d odoo-stage -c "<SQL command>"`。

小贴士

可以使用以下命令以更快的速度创建数据库副本：

`$ createdb --owner=odoo --template=odoo-prod odoo-stage`。

这里需要注意的是，为了使其运行，不能有任何对`odoo-prod`数据库的开放连接，因此在使用命令之前，需要停止 Odoo 生产服务器。

现在我们已经有了生产数据库的副本用于预发布，下一步是创建要使用的源副本。例如，这可以放在名为`/opt/odoo/stage`的子目录中。

以下 shell 命令复制相关文件并创建预发布环境：

```py
$ sudo su - odoo
$ mkdir /opt/odoo/stage
$ cp -r /opt/odoo/odoo15/ /opt/odoo/stage/
$ cp -r /opt/odoo/library/ /opt/odoo/stage/  # custom code
$ python3 -m venv /opt/odoo/env-stage
$ source /opt/odoo/env-stage/bin/activate
(env-stage) $ pip install -r \
/opt/odoo/stage/odoo15/requirements.txt
(env-stage) $ pip install -e /opt/odoo/stage/odoo15
(env-stage) $ exit
```

最后，应该为预发布环境准备一个特定的 Odoo 配置文件，因为文件使用的路径不同。使用的 HTTP 端口也应更改，以便预发布环境可以与主生产服务同时运行。

现在，这个预演环境可以用于测试目的。因此，下一节将描述如何应用生产更新。

## 更新 Odoo 源代码

Odoo 和自定义模块的代码通常通过 Git 进行版本管理。

要从 GitHub 仓库获取最新的 Odoo 源代码，请使用`git pull`命令。在此之前，可以使用`git tag`命令为当前使用的提交创建一个标签，以便更容易回滚代码更新，如下所示：

```py
$ sudo su - odoo
$ cd /opt/odoo/odoo15
$ git tag --force 15-last-prod
$ git pull
$ exit
```

要使代码更改生效，应重启 Odoo 服务。要使数据文件更改生效，需要升级模块。

小贴士

作为一般规则，对 Odoo 稳定版本的更改被认为是代码修复，因此通常不值得冒进行模块升级的风险。然而，如果你需要执行模块升级，可以使用`-u <module>`附加选项（或`-u base`），这将升级所有模块。

在将操作应用到生产数据库之前，我们可以使用预演数据库来测试这些操作，如下所示：

```py
$ source /opt/odoo/env15/bin/activate
(env15) $ odoo -c /etc/odoo/odoo.conf -d odoo-stage \
--http-port=8080 -u library  # modules to updgrade
(env15) $ exit
```

这个 Odoo 预演服务器被配置为监听端口`8080`。我们可以用我们的网络浏览器导航到那里，检查升级后的代码是否正确工作。

如果出现问题，可以使用以下命令将代码回滚到早期版本：

```py
$ sudo su - odoo
$ cd /opt/odoo/odoo15
$ git checkout 15-last-prod
$ exit
```

如果一切按预期进行，那么在生产服务上执行升级应该是安全的，这通常是通过重启来完成的。如果你想执行实际的模块升级，建议的方法是停止服务器，运行升级，然后重启服务，如下所示：

```py
$ sudo service odoo stop
$ sudo su -c "/opt/odoo/env15/bin/odoo -c /etc/odoo/odoo.conf" \
" -u base --stop-after-init" odoo
$ sudo service odoo start
```

在运行升级之前备份数据库也是建议的。

在本节中，你学习了如何创建一个与主 Odoo 环境并行的预演环境，用于测试。在将更新应用到生产系统之前，可以在预演环境中尝试对 Odoo 代码或自定义模块的更新。这使我们能够在升级前识别并纠正可能发现的问题。

# 摘要

在本章中，我们学习了在基于 Debian 的生产服务器上设置和运行 Odoo 所需的额外步骤。我们查看配置文件中的最重要的设置，并学习了如何利用多进程模式。

为了提高安全性和可扩展性，我们还学习了如何在前端 Odoo 服务器进程前使用 Nginx 作为反向代理，以及如何配置它以使用 HTTPS 加密流量。

最后，提供了一些关于如何创建预演环境以及如何对 Odoo 代码或自定义模块进行更新的建议。

这涵盖了运行 Odoo 服务器和为用户提供一个相对稳定和安全的服务所需的基本要素。现在我们可以用它来托管我们的图书馆管理系统！

# 进一步阅读

要了解更多关于 Odoo 的信息，你应该查看官方文档 [`www.odoo.com/documentation`](https://www.odoo.com/documentation)。那里有更详细的某些主题，你还会发现本书未涉及的主题。

此外，还有一些关于 Odoo 的已出版书籍可能对你有帮助。**Packt Publishing** 在其目录中收录了一些，特别是 *Odoo 开发食谱* 提供了更多关于本书未讨论主题的进阶材料。在撰写本文时，可用的最后一版是为 Odoo 14 定制的，可在 [`www.packtpub.com/product/odoo-14-development-cookbook-fourth-edition/9781800200319`](https://www.packtpub.com/product/odoo-14-development-cookbook-fourth-edition/9781800200319) 获取。

最后，Odoo 是一个具有活跃社区的开源产品。参与其中，提问和贡献是一个不仅能够学习，还能建立商业网络的绝佳方式。考虑到这一点，我们还应该提到 **Odoo 社区协会**（**OCA**），它促进协作和高质量的开源代码。你可以在 [`odoo-community.org/`](https://odoo-community.org/) 或 [`github.com/OCA`](https://github.com/OCA) 上了解更多信息。

享受你的 Odoo 之旅！
