# 第十二章：使用 Django 进行生产

当网站的开发阶段完成并且您希望使其对用户可访问时，您必须部署它。以下是要执行此操作的步骤：

+   完成开发

+   选择物理服务器

+   选择服务器软件

+   选择服务器数据库

+   安装 PIP 和 Python 3

+   安装 PostgreSQL

+   安装 Nginx

+   安装 virtualenv 并创建虚拟环境

+   安装 Django，South，Gunicorn 和 psycopg2

+   配置 PostgreSQL

+   将 Work_manager 调整为生产

+   初始 South 迁移

+   使用 Gunicorn

+   启动 Nginx

# 完成开发

在部署之前进行一些测试非常重要。实际上，当网站部署后，问题更难解决；这对开发人员和用户来说可能是巨大的时间浪费。这就是为什么我再次强调：您必须进行充分的测试！

# 选择物理服务器

物理服务器是托管您的网站的机器。在家中托管自己的网站是可能的，但这不适合专业网站。事实上，由于许多网站用户使用该网站，因此需要使用网络主机。有许多不同类型的住宿，如下所示：

+   **简单托管**：这种类型的托管适用于需要高质量服务但没有很多功率的网站。通过这种住宿，您无需处理系统管理，但它不允许与专用服务器一样的灵活性。这种类型的托管在 Django 网站上也有另一个缺点：尚未有许多提供与 Django 兼容的住宿。

+   **专用服务器**：这是最灵活的住宿类型。我们租用（或购买）一台服务器，由提供互联网连接和其他服务的网络主机提供。根据所需的配置不同，价格也不同，但功能强大的服务器非常昂贵。这种类型的住宿要求您处理系统管理，除非您订阅外包服务。外包服务允许您使用系统管理员来照顾服务器，并获得报酬。

+   **虚拟服务器**：虚拟服务器与专用服务器非常相似。它们通常价格较低，因为一些虚拟服务器可以在单个物理服务器上运行。主机经常提供额外的服务，如服务器热备份或复制。

选择住宿类型应基于您的需求和财政资源。

以下是提供 Django 的主机的非详尽列表：

+   alwaysdata

+   WebFaction

+   DjangoEurope

+   DjangoFoo Hosting

# 选择服务器软件

在开发阶段，我们使用了 Django 附带的服务器。该服务器在开发过程中非常方便，但不适合生产网站。事实上，开发服务器既不高效也不安全。您必须选择另一种类型的服务器来安装它。有许多 Web 服务器；我们选择了其中两个：

+   **Apache HTTP 服务器**：根据 Netcraft 的数据，自 1996 年以来，这一直是最常用的 Web 服务器。这是一个模块化服务器，允许您安装模块而无需编译服务器。近年来，它的使用越来越少。根据 Netcraft 的数据，2013 年 4 月，市场份额为 51％。

+   **Nginx**：Nginx 以其性能和低内存消耗而闻名。它也是模块化的，但模块需要在编译中集成。2013 年 4 月，Netcraft 知道的所有网站中有 14％使用了 Nginx 作为其 Web 服务器。

# 选择服务器数据库

选择服务器数据库非常重要。实际上，该服务器将存储网站的所有数据。在数据库中寻求的主要特征是性能，安全性和可靠性。

选择取决于以下三个标准的重要性：

+   **Oracle**：这个数据库是由 Oracle Corporation 开发的系统数据库。有这个数据库的免费开源版本，但其功能有限。这不是一个免费的数据库。

+   **MySQL**：这是属于 Oracle 的数据库系统（自从收购 Sun Microsystems 以来）。它是 Web 上广泛使用的数据库，包括**LAMP**（**Linux Apache MySQL PHP**）平台。它以双 GPL 和专有许可证进行分发。

+   **PostgreSQL**：这是一个根据 BSD 许可证分发的免费数据库系统。这个系统被认为是稳定的，并提供高级功能（如数据类型的创建）。

+   **SQLite**：这是我们在开发网站期间使用的系统。它不适合访问量很大的网站。事实上，整个数据库都在一个 SQLite 文件中，并且不允许竞争对手访问数据。此外，没有用户或系统没有安全机制。但是，完全可以用它来向客户演示。

+   **MongoDB**：这是一个面向文档的数据库。这个数据库系统被归类为 NoSQL 数据库，因为它使用了使用**BSON**（**二进制 JSON**）格式的存储架构。这个系统在数据库分布在多台服务器之间的环境中很受欢迎。

# 部署 Django 网站

在本书的其余部分，我们将使用 HTTP Nginx 服务器和 PostgreSQL 数据库。本章的解释将在 GNU / Linux Debian 7.3.0 32 位系统上进行。我们将从一个没有任何安装的新的 Debian 操作系统开始。

## 安装 PIP 和 Python 3

对于以下命令，您必须使用具有与超级用户帐户相同特权的用户帐户登录。为此，请运行以下命令：

```py
su

```

在此命令之后，您必须输入 root 密码。

首先，我们更新 Debian 存储库：

```py
apt-get update

```

然后，我们安装 Python 3 和 PIP，就像在第二章中所做的那样，*创建一个 Django 项目*：

```py
apt-get install python3
apt-get install python3-pip
alias pip=pip-3.2

```

## 安装 PostgreSQL

我们将安装四个软件包以便使用 PostgreSQL：

```py
apt-get install libpq-dev python-dev postgresql postgresql-contrib

```

然后，我们将安装我们的 web Nginx 服务器：

```py
apt-get install nginx

```

## 安装 virtualenv 并创建虚拟环境

我们已经像在第二章中所做的那样安装了 Python 和 PIP，但在安装 Django 之前，我们将安装 virtualenv。这个工具用于为 Python 创建虚拟环境，并在同一个操作系统上拥有不同的库版本。事实上，在许多 Linux 系统中，Debian 已经安装了 Python 2 的一个版本。建议您不要卸载它以保持系统的稳定。我们将安装 virtualenv 来设置我们自己的环境，并简化我们未来的 Django 迁移：

```py
pip install virtualenv

```

然后，您必须创建一个将托管您的虚拟环境的文件夹：

```py
mkdir /home/env

```

以下命令在`/home/env/`文件夹中创建一个名为`django1.6`的虚拟环境：

```py
virtualenv /home/env/django1.6

```

然后，我们将通过发出以下命令为所有用户提供访问环境文件夹的所有权限。从安全的角度来看，最好限制用户或组的访问，但这将花费很多时间：

```py
cd /home/
chmod -R 777 env/
exit

```

# 安装 Django、South、Gunicorn 和 psycopg2

我们将安装 Django 和所有 Nginx 和 Django 通信所需的组件。我们首先激活我们的虚拟环境。以下命令将连接我们到虚拟环境。因此，从此环境中执行的所有 Python 命令只能使用此环境中安装的软件包。在我们的情况下，我们将安装四个仅安装在我们的虚拟环境中的库。对于以下命令，您必须以没有超级用户特权的用户登录。我们不能从 root 帐户执行以下命令，因为我们需要 virtualenv。但是，root 帐户有时会覆盖虚拟环境，以使用系统中的 Python，而不是虚拟环境中存在的 Python。

```py
source /home/env/django1.6/bin/activate
pip install django=="1.6"
pip install South

```

Gunicorn 是一个扮演 Python 和 Nginx 之间 WSGI 接口角色的 Python 包。要安装它，请发出以下命令：

```py
pip install gunicorn 

```

psycopg2 是一个允许 Python 和 PostgreSQL 相互通信的库：

```py
pip install psycopg2

```

要重新连接为超级用户，我们必须断开与虚拟环境的连接：

```py
deactivate

```

## 配置 PostgreSQL

对于以下命令，您必须使用具有与超级用户相同特权的用户帐户登录。我们将连接到 PostgreSQL 服务器：

```py
su
su - postgres

```

以下命令创建一个名为`workmanager`的数据库：

```py
createdb workmanager

```

然后，我们将为 PostgreSQL 创建一个用户。输入以下命令后，会要求更多信息：

```py
createuser -P 

```

以下行是 PostgreSQL 要求的新用户信息和响应（用于本章）：

```py
Role name : workmanager
Password : workmanager
Password confirmation : workmanager
Super user : n
Create DB : n
Create new roles : n

```

然后，我们必须连接到 PostgreSQL 解释器：

```py
psql 

```

我们在新数据库上给予新用户所有权限：

```py
GRANT ALL PRIVILEGES ON DATABASE workmanager TO workmanager;

```

然后，我们退出 SQL 解释器和与 PostgreSQL 的连接：

```py
\q
exit

```

## 将 Work_manager 适应到生产环境

对于以下命令，您必须以没有超级用户特权的用户登录。

在部署的这个阶段，我们必须复制包含我们的 Django 项目的文件夹。要复制的文件夹是`Work_manager`文件夹（其中包含`Work_manager`和`TasksManager`文件夹以及`manage.py`文件）。我们将其复制到虚拟环境的根目录，即`/home/env/django1.6`。

要复制它，您可以使用您拥有的手段：USB 键，SFTP，FTP 等。然后，我们需要编辑项目的`settings.py`文件以使其适应部署。

定义数据库连接的部分变为以下内容：

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2', 
        'NAME':  'workmanager',      
        'USER': 'workmanager',                     
        'PASSWORD': 'workmanager',                 
        'HOST': '127.0.0.1',                     
        'PORT': '',                     
    }
}
```

我们必须修改`ALLOWED_HOSTS`行如下：

```py
ALLOWED_HOSTS = ['*']
```

另外，重要的是不要使用`DEBUG`模式。实际上，`DEBUG`模式可以为黑客提供宝贵的数据。为此，我们必须以以下方式更改`DEBUG`和`TEMPLATE_DEBUG`变量：

```py
DEBUG = False
TEMPLATE_DEBUG = False
```

## 初始 South 迁移

我们激活我们的虚拟环境以执行迁移并启动 Gunicorn：

```py
cd /home/env/django1.6/Work_manager/
source /home/env/django1.6/bin/activate
python3.2 manage.py schemamigration TasksManager --initial
python3.2 manage.py syncdb -–migrate

```

有时，使用 PostgreSQL 创建数据库时会出现错误，即使一切顺利。要查看数据库的创建是否顺利，我们必须以 root 用户身份运行以下命令，并验证表是否已创建：

```py
su
su - postgres
psql -d workmanager
\dt
\q
exit

```

如果它们被正确创建，您必须进行一个虚假的 South 迁移，手动告诉它一切顺利：

```py
python3.2 manage.py migrate TasksManager --fake

```

## 使用 Gunicorn

然后，我们启动我们的 WSGI 接口，以便 Nginx 进行通信：

```py
gunicorn Work_manager.wsgi

```

## 启动 Nginx

另一个命令提示符作为 root 用户必须使用以下命令运行 Nginx：

```py
su
service nginx start

```

现在，我们的 Web 服务器是功能性的，并且已准备好与许多用户一起工作。

# 总结

在本章中，我们学习了如何使用现代架构部署 Django 网站。此外，我们使用了 virtualenv，它允许您在同一系统上使用多个版本的 Python 库。

在这本书中，我们学习了什么是 MVC 模式。我们已经为我们的开发环境安装了 Python 和 Django。我们学会了如何创建模板、视图和模型。我们还使用系统来路由 Django 的 URL。我们还学会了如何使用一些特定的元素，比如 Django 表单、CBV 或认证模块。然后，我们使用了会话变量和 AJAX 请求。最后，我们学会了如何在 Linux 服务器上部署 Django 网站。
