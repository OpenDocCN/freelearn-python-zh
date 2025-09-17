# 第七章：编程跨越机器边界

在本章中，我们将介绍以下配方：

+   使用 telnet 执行远程 shell 命令

+   通过 SFTP 将文件复制到远程机器

+   打印远程机器的 CPU 信息

+   远程安装 Python 包

+   远程运行 MySQL 命令

+   通过 SSH 将文件传输到远程机器

+   远程配置 Apache 以托管网站

# 简介

本章推荐一些有趣的 Python 库。这些配方旨在面向系统管理员和喜欢编写连接到远程系统并执行命令的代码的高级 Python 程序员。本章从使用内置的 Python 库`telnetlib`的轻量级配方开始。然后引入了著名的远程访问库`Paramiko`。最后，介绍了功能强大的远程系统管理库`fabric`。`fabric`库受到经常为自动部署编写脚本的开发者的喜爱，例如部署 Web 应用程序或构建自定义应用程序的二进制文件。

# 使用 telnet 执行远程 shell 命令

如果您需要通过 telnet 连接到旧的网络交换机或路由器，您可以从 Python 脚本而不是使用 bash 脚本或交互式 shell 中这样做。此配方将创建一个简单的 telnet 会话。它将向您展示如何向远程主机执行 shell 命令。

## 准备工作

您需要在您的机器上安装 telnet 服务器并确保它已启动并运行。您可以使用针对您操作系统的包管理器来安装 telnet 服务器包。例如，在 Debian/Ubuntu 上，您可以使用`apt-get`或`aptitude`来安装`telnetd`包，如下面的命令所示：

```py
$ sudo apt-get install telnetd
$ telnet localhost

```

## 如何操作...

让我们定义一个函数，该函数将从命令提示符获取用户的登录凭证并连接到 telnet 服务器。

连接成功后，它将发送 Unix 的`'ls'`命令。然后，它将显示命令的输出，例如，列出目录的内容。

列表 7.1 显示了执行远程 Unix 命令的 telnet 会话的代码如下：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 7
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import getpass
import sys
import telnetlib

def run_telnet_session():
  host = raw_input("Enter remote hostname e.g. localhost:")
  user = raw_input("Enter your remote account: ")
  password = getpass.getpass()

  session = telnetlib.Telnet(host)

  session.read_until("login: ")
  session.write(user + "\n")
  if password:
    session.read_until("Password: ")
    session.write(password + "\n")

  session.write("ls\n")
  session.write("exit\n")

  print session.read_all()

if __name__ == '__main__':
  run_telnet_session()
```

如果您在本地机器上运行 telnet 服务器并运行此代码，它将要求您输入远程用户账户和密码。以下输出显示了在 Debian 机器上执行的 telnet 会话：

```py
$ python 7_1_execute_remote_telnet_cmd.py 
Enter remote hostname e.g. localhost: localhost
Enter your remote account: faruq
Password: 

ls
exit
Last login: Mon Aug 12 10:37:10 BST 2013 from localhost on pts/9
Linux debian6 2.6.32-5-686 #1 SMP Mon Feb 25 01:04:36 UTC 2013 i686

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.
You have new mail.
faruq@debian6:~$ ls 
down              Pictures               Videos
Downloads         projects               yEd
Dropbox           Public
env               readme.txt
faruq@debian6:~$ exit
logout

```

## 它是如何工作的...

此配方依赖于 Python 的内置`telnetlib`网络库来创建一个 telnet 会话。`run_telnet_session()`函数从命令提示符获取用户名和密码。使用`getpass`模块的`getpass()`函数来获取密码，因为这个函数不会让您看到屏幕上输入的内容。

为了创建一个 telnet 会话，您需要实例化一个`Telnet()`类，该类需要一个主机名参数来初始化。在这种情况下，使用`localhost`作为主机名。您可以使用`argparse`模块将主机名传递给此脚本。

可以使用`read_until()`方法捕获 telnet 会话的远程输出。在第一种情况下，使用此方法检测到登录提示。然后，通过`write()`方法（在这种情况下，与远程访问相同的机器）将带有新行换行的用户名发送到远程机器。同样，密码也被提供给远程主机。

然后，将`ls`命令发送以执行。最后，为了从远程主机断开连接，发送`exit`命令，并使用`read_all()`方法在屏幕上打印从远程主机接收到的所有会话数据。

# 通过 SFTP 将文件复制到远程机器

如果您想安全地将文件从本地机器上传或复制到远程机器，您可以通过**安全文件传输协议**（**SFTP**）来实现。

## 准备工作

这个菜谱使用了一个强大的第三方网络库`Paramiko`，展示了如何通过 SFTP 进行文件复制的示例，如下所示命令。您可以从 GitHub([`github.com/paramiko/paramiko`](https://github.com/paramiko/paramiko))或 PyPI 获取`Paramiko`的最新代码：

```py
$ pip install paramiko

```

## 如何做...

这个菜谱接受一些命令行输入：远程主机名、服务器端口、源文件名和目标文件名。为了简单起见，我们可以为这些输入参数使用默认值或硬编码值。

为了连接到远程主机，我们需要用户名和密码，这些可以从命令行中的用户那里获取。

列表 7.2 解释了如何通过 SFTP 远程复制文件，如下所示代码：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 7
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications. 

import argparse
import paramiko
import getpass

SOURCE = '7_2_copy_remote_file_over_sftp.py'
DESTINATION ='/tmp/7_2_copy_remote_file_over_sftp.py '

def copy_file(hostname, port, username, password, src, dst):
  client = paramiko.SSHClient()
  client.load_system_host_keys()
  print " Connecting to %s \n with username=%s... \n" 
%(hostname,username)
  t = paramiko.Transport((hostname, port)) 
  t.connect(username=username,password=password)
  sftp = paramiko.SFTPClient.from_transport(t)
  print "Copying file: %s to path: %s" %(SOURCE, DESTINATION)
  sftp.put(src, dst)
  sftp.close()
  t.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Remote file copy')
  parser.add_argument('--host', action="store", dest="host", 
default='localhost')
  parser.add_argument('--port', action="store", dest="port", 
default=22, type=int)
  parser.add_argument('--src', action="store", dest="src", 
default=SOURCE)
  parser.add_argument('--dst', action="store", dest="dst", 
default=DESTINATION)

  given_args = parser.parse_args()
  hostname, port =  given_args.host, given_args.port
  src, dst = given_args.src, given_args.dst

  username = raw_input("Enter the username:")
  password = getpass.getpass("Enter password for %s: " %username)

  copy_file(hostname, port, username, password, src, dst)
```

如果您运行此脚本，您将看到类似以下输出的结果：

```py
$ python 7_2_copy_remote_file_over_sftp.py 
Enter the username:faruq
Enter password for faruq: 
 Connecting to localhost 
 with username=faruq... 
Copying file: 7_2_copy_remote_file_over_sftp.py to path: 
/tmp/7_2_copy_remote_file_over_sftp.py 

```

## 工作原理...

这个菜谱可以接受连接到远程机器和通过 SFTP 复制文件的多种输入。

这个菜谱将命令行输入传递给`copy_file()`函数。然后，它创建一个 SSH 客户端，调用`paramiko`的`SSHClient`类。客户端需要加载系统主机密钥。然后，它连接到远程系统，从而创建`transport`类的实例。实际的 SFTP 连接对象`sftp`是通过调用`paramiko`的`SFTPClient.from_transport()`函数创建的。这个函数需要一个`transport`实例作为输入。

在 SFTP 连接就绪后，使用`put()`方法将本地文件通过此连接复制到远程主机。

最后，通过分别在每个对象上调用`close()`方法来清理 SFTP 连接和底层对象是个好主意。

# 打印远程机器的 CPU 信息

有时候，我们需要通过 SSH 在远程机器上运行一个简单的命令。例如，我们需要查询远程机器的 CPU 或 RAM 信息。这可以通过如下的 Python 脚本实现。

## 准备工作

您需要安装第三方包`Paramiko`，如下所示命令，从 GitHub 仓库[`github.com/paramiko/paramiko`](https://github.com/paramiko/paramiko)提供的源代码中安装：

```py
$ pip install paramiko

```

## 如何做...

我们可以使用`paramiko`模块创建到 Unix 机器的远程会话。

然后，从本次会话中，我们可以读取远程机器的`/proc/cpuinfo`文件以提取 CPU 信息。

列表 7.3 给出了打印远程机器 CPU 信息的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 7
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import argparse
import getpass
import paramiko

RECV_BYTES = 4096
COMMAND = 'cat /proc/cpuinfo'

def print_remote_cpu_info(hostname, port, username, password):
  client = paramiko.Transport((hostname, port))
  client.connect(username=username, password=password)

  stdout_data = []
  stderr_data = []
  session = client.open_channel(kind='session')
  session.exec_command(COMMAND)
  while True:
    if session.recv_ready():
      stdout_data.append(session.recv(RECV_BYTES))
      if session.recv_stderr_ready():
        stderr_data.append(session.recv_stderr(RECV_BYTES))
      if session.exit_status_ready():
        break

  print 'exit status: ', session.recv_exit_status()
  print ''.join(stdout_data)
  print ''.join(stderr_data)

  session.close()
  client.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Remote file copy')
  parser.add_argument('--host', action="store", dest="host", 
default='localhost')
  parser.add_argument('--port', action="store", dest="port", 
default=22, type=int)    
  given_args = parser.parse_args()
  hostname, port =  given_args.host, given_args.port

  username = raw_input("Enter the username:")
  password = getpass.getpass("Enter password for %s: " %username)
  print_remote_cpu_info(hostname, port, username, password)
```

运行此脚本将显示指定主机的 CPU 信息，在本例中，为本地计算机，如下所示：

```py
$ python 7_3_print_remote_cpu_info.py 
Enter the username:faruq
Enter password for faruq: 
exit status:  0
processor    : 0
vendor_id    : GenuineIntel
cpu family    : 6
model        : 42
model name    : Intel(R) Core(TM) i5-2400S CPU @ 2.50GHz
stepping    : 7
cpu MHz        : 2469.677
cache size    : 6144 KB
fdiv_bug    : no
hlt_bug        : no
f00f_bug    : no
coma_bug    : no
fpu        : yes
fpu_exception    : yes
cpuid level    : 5
wp        : yes
flags        : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 syscall nx rdtscp lm constant_tsc up pni monitor ssse3 lahf_lm
bogomips    : 4939.35
clflush size    : 64
cache_alignment    : 64
address sizes    : 36 bits physical, 48 bits virtual
power management:

```

## 它是如何工作的...

首先，我们收集连接参数，如主机名、端口、用户名和密码。然后，将这些参数传递给`print_remote_cpu_info()`函数。

此函数通过调用`paramiko`的`transport`类创建 SSH 客户端会话。之后使用提供的用户名和密码建立连接。我们可以使用 SSH 客户端上的`open_channel()`创建原始通信会话。为了在远程主机上执行命令，可以使用`exec_command()`。

在向远程主机发送命令后，可以通过阻塞会话对象的`recv_ready()`事件来捕获远程主机的响应。我们可以创建两个列表，`stdout_data`和`stderr_data`，并使用它们来存储远程输出和错误消息。

当命令在远程机器上退出时，可以使用`exit_status_ready()`方法检测，并使用`join()`字符串方法连接远程会话数据。

最后，可以使用每个对象的`close()`方法关闭会话和客户端连接。

# 远程安装 Python 包

在处理远程主机的前一个示例中，您可能已经注意到我们需要做很多与连接设置相关的事情。为了高效执行，最好将它们抽象化，只将相关的高级部分暴露给程序员。总是明确设置连接以执行远程命令既繁琐又慢。

Fabric ([`fabfile.org/`](http://fabfile.org/))，一个第三方 Python 模块，解决了这个问题。它只暴露了可以用来高效与远程机器交互的 API。

在本例中，将展示使用 Fabric 的简单示例。

## 准备工作

我们需要首先安装 Fabric。您可以使用 Python 打包工具`pip`或`easy_install`安装 Fabric，如下所示。Fabric 依赖于`paramiko`模块，它将自动安装。

```py
$ pip install fabric

```

在这里，我们将使用 SSH 协议连接远程主机。因此，在远程端运行 SSH 服务器是必要的。如果您想使用本地计算机进行测试（假装访问远程机器），您可以在本地安装`openssh`服务器包。在 Debian/Ubuntu 机器上，可以使用包管理器`apt-get`完成此操作，如下所示：

```py
$ sudo apt-get install openssh-server

```

## 如何操作...

这是使用 Fabric 安装 Python 包的代码。

列表 7.4 给出了远程安装 Python 包的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 7
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

from getpass import getpass
from fabric.api import settings, run, env, prompt

def remote_server():
  env.hosts = ['127.0.0.1']
  env.user = prompt('Enter user name: ')
  env.password = getpass('Enter password: ')

def install_package():
  run("pip install yolk")
```

与常规 Python 脚本相比，Fabric 脚本的运行方式不同。所有使用 `fabric` 库的函数都必须引用一个名为 `fabfile.py` 的 Python 脚本。在这个脚本中没有传统的 `__main__` 指令。相反，你可以使用 Fabric API 定义你的方法，并使用命令行工具 `fab` 执行这些方法。因此，你不需要调用 `python <script>.py`，而是可以通过调用 `fab one_function_name another_function_name` 来运行定义在 `fabfile.py` 脚本中并位于当前目录下的 Fabric 脚本。

因此，让我们创建一个 `fabfile.py` 脚本，如下所示命令。为了简化，你可以从任何文件创建一个到 `fabfile.py` 脚本的文件快捷方式或链接。首先，删除任何之前创建的 `fabfile.py` 文件，并创建一个到 `fabfile` 的快捷方式：

```py
$ rm -rf fabfile.py
$ ln -s 7_4_install_python_package_remotely.py fabfile.py

```

如果你现在调用 fabfile，它将在远程安装 Python 包 `yolk` 后产生以下输出：

```py
$ ln -sfn 7_4_install_python_package_remotely.py fabfile.py
$ fab remote_server install_package
Enter user name: faruq
Enter password:
[127.0.0.1] Executing task 'install_package'
[127.0.0.1] run: pip install yolk
[127.0.0.1] out: Downloading/unpacking yolk
[127.0.0.1] out:   Downloading yolk-0.4.3.tar.gz (86kB): 
[127.0.0.1] out:   Downloading yolk-0.4.3.tar.gz (86kB): 100%  86kB
[127.0.0.1] out:   Downloading yolk-0.4.3.tar.gz (86kB): 
[127.0.0.1] out:   Downloading yolk-0.4.3.tar.gz (86kB): 86kB 
downloaded
[127.0.0.1] out:   Running setup.py egg_info for package yolk
[127.0.0.1] out:     Installing yolk script to /home/faruq/env/bin
[127.0.0.1] out: Successfully installed yolk
[127.0.0.1] out: Cleaning up...
[127.0.0.1] out: 

Done.
Disconnecting from 127.0.0.1... done.

```

## 它是如何工作的...

这个食谱演示了如何使用 Python 脚本远程执行系统管理任务。在这个脚本中有两个函数。`remote_server()` 函数设置 Fabric `env` 环境变量，例如主机名、用户、密码等。

另一个功能，`install_package()`，调用 `run()` 函数。这接受你在命令行中通常输入的命令。在这种情况下，命令是 `pip install yolk`。这使用 `pip` 安装 Python 包 `yolk`。与之前描述的食谱相比，使用 Fabric 运行远程命令的方法更简单、更高效。

# 在远程运行 MySQL 命令

如果你需要远程管理 MySQL 服务器，这个食谱就适合你。它将展示如何从 Python 脚本向远程 MySQL 服务器发送数据库命令。如果你需要设置一个依赖于后端数据库的 Web 应用程序，这个食谱可以作为你的 Web 应用程序设置过程的一部分使用。

## 准备工作

这个食谱也需要首先安装 Fabric。你可以使用 Python 打包工具 `pip` 或 `easy_install` 来安装 Fabric，如下所示命令。Fabric 依赖于 `paramiko` 模块，它将被自动安装。

```py
$ pip install fabric

```

在这里，我们将使用 SSH 协议连接到远程主机。因此，在远程端运行 SSH 服务器是必要的。你还需要在远程主机上运行一个 MySQL 服务器。在 Debian/Ubuntu 系统上，可以使用包管理器 `apt-get` 来完成，如下所示命令：

```py
$ sudo apt-get install openssh-server mysql-server

```

## 如何操作...

我们定义了 Fabric 环境设置和一些用于远程管理 MySQL 的函数。在这些函数中，我们不是直接调用 `mysql` 可执行文件，而是通过 `echo` 将 SQL 命令发送到 `mysql`。这确保了参数被正确传递给 `mysql` 可执行文件。

列表 7.5 给出了运行 MySQL 命令的远程代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 7
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

from getpass import getpass 
from fabric.api import run, env, prompt, cd

def remote_server():
  env.hosts = ['127.0.0.1']
# Edit this list to include remote hosts
  env.user =prompt('Enter your system username: ')
  env.password = getpass('Enter your system user password: ')
  env.mysqlhost = 'localhost'
  env.mysqluser = 'root'prompt('Enter your db username: ')
  env.password = getpass('Enter your db user password: ')
  env.db_name = ''

def show_dbs():
  """ Wraps mysql show databases cmd"""
  q = "show databases"
  run("echo '%s' | mysql -u%s -p%s" %(q, env.mysqluser, 
env.mysqlpassword))

def run_sql(db_name, query):
  """ Generic function to run sql"""
  with cd('/tmp'):
    run("echo '%s' | mysql -u%s -p%s -D %s" %(query, 
env.mysqluser, env.mysqlpassword, db_name))

def create_db():
  """Create a MySQL DB for App version"""
  if not env.db_name:
    db_name = prompt("Enter the DB name:")
  else:
    db_name = env.db_name
  run('echo "CREATE DATABASE %s default character set utf8 collate 
utf8_unicode_ci;"|mysql --batch --user=%s --password=%s --
host=%s'\
    % (db_name, env.mysqluser, env.mysqlpassword, env.mysqlhost), 
pty=True)

def ls_db():
  """ List a dbs with size in MB """
  if not env.db_name:
    db_name = prompt("Which DB to ls?")
  else:
    db_name = env.db_name
  query = """SELECT table_schema                                        
"DB Name", 
  Round(Sum(data_length + index_length) / 1024 / 1024, 1) "DB Size 
in MB" 
    FROM   information_schema.tables         
    WHERE table_schema = \"%s\" 
    GROUP  BY table_schema """ %db_name
  run_sql(db_name, query)

def empty_db():
  """ Empty all tables of a given DB """
  db_name = prompt("Enter DB name to empty:")
  cmd = """
  (echo 'SET foreign_key_checks = 0;'; 
  (mysqldump -u%s -p%s --add-drop-table --no-data %s | 
  grep ^DROP); 
  echo 'SET foreign_key_checks = 1;') | \
  mysql -u%s -p%s -b %s
  """ %(env.mysqluser, env.mysqlpassword, db_name, env.mysqluser, 
env.mysqlpassword, db_name)
  run(cmd)
```

为了运行此脚本，你应该创建一个快捷方式，`fabfile.py`。从命令行，你可以通过输入以下命令来完成：

```py
$ ln -sfn 7_5_run_mysql_command_remotely.py fabfile.py

```

然后，你可以以各种形式调用 `fab` 可执行文件。

以下命令将显示数据库列表（使用 SQL 查询，`show databases`）：

```py
$ fab remote_server show_dbs

```

以下命令将创建一个新的 MySQL 数据库。如果你没有定义 Fabric 环境变量 `db_name`，将显示提示输入目标数据库名称。此数据库将使用 SQL 命令 `CREATE DATABASE <database_name> default character set utf8 collate utf8_unicode_ci;` 创建。

```py
$ fab remote_server create_db

```

这个 Fabric 命令将显示数据库的大小：

```py
$ fab remote_server ls_db()

```

以下 Fabric 命令将使用 `mysqldump` 和 `mysql` 可执行文件来清空数据库。此函数的行为类似于数据库截断，但会删除所有表。结果是创建了一个没有任何表的全新数据库：

```py
$ fab remote_server empty_db()

```

以下将是输出：

```py
$ $ fab remote_server show_dbs
[127.0.0.1] Executing task 'show_dbs'
[127.0.0.1] run: echo 'show databases' | mysql -uroot -p<DELETED>
[127.0.0.1] out: Database
[127.0.0.1] out: information_schema
[127.0.0.1] out: mysql
[127.0.0.1] out: phpmyadmin
[127.0.0.1] out: 

Done.
Disconnecting from 127.0.0.1... done.

$ fab remote_server create_db
[127.0.0.1] Executing task 'create_db'
Enter the DB name: test123
[127.0.0.1] run: echo "CREATE DATABASE test123 default character set utf8 collate utf8_unicode_ci;"|mysql --batch --user=root --password=<DELETED> --host=localhost

Done.
Disconnecting from 127.0.0.1... done.
$ fab remote_server show_dbs
[127.0.0.1] Executing task 'show_dbs'
[127.0.0.1] run: echo 'show databases' | mysql -uroot -p<DELETED>
[127.0.0.1] out: Database
[127.0.0.1] out: information_schema
[127.0.0.1] out: collabtive
[127.0.0.1] out: test123
[127.0.0.1] out: testdb
[127.0.0.1] out: 

Done.
Disconnecting from 127.0.0.1... done.

```

## 它是如何工作的...

此脚本定义了一些与 Fabric 一起使用的函数。第一个函数 `remote_server()` 设置环境变量。将本地回环 IP (`127.0.0.1`) 放到主机列表中。设置本地系统用户和 MySQL 登录凭证，并通过 `getpass()` 收集。

另一个函数利用 Fabric 的 `run()` 函数通过将命令回显到 `mysql` 可执行文件来向远程 MySQL 服务器发送 MySQL 命令。

`run_sql()` 函数是一个通用函数，可以用作其他函数的包装器。例如，`empty_db()` 函数调用它来执行 SQL 命令。这可以使你的代码更加有组织且更干净。

# 通过 SSH 将文件传输到远程机器

在使用 Fabric 自动化远程系统管理任务时，如果你想通过 SSH 在本地机器和远程机器之间传输文件，你可以使用 Fabric 内置的 `get()` 和 `put()` 函数。这个菜谱展示了我们如何通过在传输前后检查磁盘空间来创建自定义函数，以智能地传输文件。

## 准备工作

此菜谱也需要首先安装 Fabric。你可以使用 Python 打包工具 `pip` 或 `easy_install` 来安装 Fabric，如下所示：

```py
$ pip install fabric

```

在这里，我们将使用 SSH 协议连接远程主机。因此，在远程主机上安装和运行 SSH 服务器是必要的。

## 如何操作...

让我们先设置 Fabric 环境变量，然后创建两个函数，一个用于下载文件，另一个用于上传文件。

列表 7.6 给出了通过 SSH 将文件传输到远程机器的代码如下：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 7
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

from getpass import getpass
from fabric.api import local, run, env, get, put, prompt, open_shell

def remote_server():
  env.hosts = ['127.0.0.1']
  env.password = getpass('Enter your system password: ')
  env.home_folder = '/tmp'

def login():
  open_shell(command="cd %s" %env.home_folder)

def download_file():
  print "Checking local disk space..."
  local("df -h")
  remote_path = prompt("Enter the remote file path:")
  local_path = prompt("Enter the local file path:")
  get(remote_path=remote_path, local_path=local_path)
  local("ls %s" %local_path)

def upload_file():
  print "Checking remote disk space..."
  run("df -h")
  local_path = prompt("Enter the local file path:")
  remote_path = prompt("Enter the remote file path:")
  put(remote_path=remote_path, local_path=local_path)
  run("ls %s" %remote_path)
```

为了运行此脚本，你应该创建一个快捷方式，`fabfile.py`。从命令行，你可以通过输入以下命令来完成：

```py
$ ln -sfn 7_6_transfer_file_over_ssh.py fabfile.py

```

然后，你可以以各种形式调用 `fab` 可执行文件。

首先，为了使用你的脚本登录到远程服务器，你可以运行以下 Fabric 函数：

```py
$ fab remote_server login

```

这将为您提供最小化的 shell-like 环境。然后，您可以使用以下命令从远程服务器下载文件到本地机器：

```py
$ fab remote_server download_file

```

同样，要上传文件，可以使用以下命令：

```py
$ fab remote_server upload_file

```

在此示例中，通过 SSH 使用本地机器。因此，您必须在本地上安装 SSH 服务器才能运行这些脚本。否则，您可以修改`remote_server()`函数并将其指向远程服务器，如下所示：

```py
$ fab remote_server login
[127.0.0.1] Executing task 'login'
Linux debian6 2.6.32-5-686 #1 SMP Mon Feb 25 01:04:36 UTC 2013 i686

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.
You have new mail.
Last login: Wed Aug 21 15:08:45 2013 from localhost
cd /tmp
faruq@debian6:~$ cd /tmp
faruq@debian6:/tmp$ 

<CTRL+D>
faruq@debian6:/tmp$ logout

Done.
Disconnecting from 127.0.0.1... done.

$ fab remote_server download_file
[127.0.0.1] Executing task 'download_file'
Checking local disk space...
[localhost] local: df -h
Filesystem            Size  Used Avail Use% Mounted on
/dev/sda1              62G   47G   12G  81% /
tmpfs                 506M     0  506M   0% /lib/init/rw
udev                  501M  160K  501M   1% /dev
tmpfs                 506M  408K  505M   1% /dev/shm
Z_DRIVE              1012G  944G   69G  94% /media/z
C_DRIVE               466G  248G  218G  54% /media/c
Enter the remote file path: /tmp/op.txt
Enter the local file path: .
[127.0.0.1] download: chapter7/op.txt <- /tmp/op.txt
[localhost] local: ls .
7_1_execute_remote_telnet_cmd.py   7_3_print_remote_cpu_info.py           7_5_run_mysql_command_remotely.py  7_7_configure_Apache_for_hosting_website_remotely.py  fabfile.pyc  __init__.py  test.txt
7_2_copy_remote_file_over_sftp.py  7_4_install_python_package_remotely.py  7_6_transfer_file_over_ssh.py      fabfile.py                        index.html     op.txt       vhost.conf

Done.
Disconnecting from 127.0.0.1... done.

```

## 工作原理...

在此配方中，我们使用了一些 Fabric 的内置函数在本地机和远程机之间传输文件。`local()`函数在本地机上执行操作，而远程操作由`run()`函数执行。

在上传文件之前检查目标机器上的可用磁盘空间非常有用，反之亦然。

这是通过使用 Unix 命令`df`实现的。源文件路径和目标文件路径可以通过命令提示符指定，或者在无人值守的自动执行的情况下，可以在源文件中硬编码。

# 远程配置 Apache 托管网站

Fabric 函数可以作为普通用户和超级用户运行。如果您需要在远程 Apache Web 服务器上托管网站，则需要管理员用户权限来创建配置文件和重新启动 Web 服务器。此配方介绍了 Fabric 的`sudo()`函数，该函数在远程机器上以超级用户身份运行命令。在此，我们希望配置 Apache 虚拟主机以运行网站。

## 准备工作

此配方需要首先在您的本地机器上安装 Fabric。您可以使用 Python 打包工具`pip`或`easy_install`安装 Fabric，如下所示：

```py
$ pip install fabric

```

在这里，我们将使用 SSH 协议连接远程主机。因此，在远程主机上安装和运行 SSH 服务器是必要的。还假设 Apache Web 服务器已安装在远程服务器上。在 Debian/Ubuntu 机器上，可以使用包管理器`apt-get`完成此操作，如下所示：

```py
$ sudo apt-get install openssh-server apache2

```

## 如何操作...

首先，我们收集 Apache 安装路径和一些配置参数，例如，Web 服务器用户、组、虚拟主机配置路径和初始化脚本。这些参数可以定义为常量。

然后，我们设置两个函数，`remote_server()`和`setup_vhost()`，使用 Fabric 执行 Apache 配置任务。

列表 7.7 提供了以下配置 Apache 远程托管网站的代码：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 7
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

from fabric.api import env, put, sudo, prompt
from fabric.contrib.files import exists

WWW_DOC_ROOT = "/data/apache/test/"
WWW_USER = "www-data"
WWW_GROUP = "www-data"
APACHE_SITES_PATH = "/etc/apache2/sites-enabled/"
APACHE_INIT_SCRIPT = "/etc/init.d/apache2 "

def remote_server():
  env.hosts = ['127.0.0.1']
  env.user = prompt('Enter user name: ')
  env.password = getpass('Enter your system password: ')

def setup_vhost():
  """ Setup a test website """
  print "Preparing the Apache vhost setup..."

  print "Setting up the document root..."
  if exists(WWW_DOC_ROOT):
    sudo("rm -rf %s" %WWW_DOC_ROOT)
  sudo("mkdir -p %s" %WWW_DOC_ROOT)

  # setup file permissions
  sudo("chown -R %s.%s %s" %(env.user, env.user, WWW_DOC_ROOT))

  # upload a sample index.html file
  put(local_path="index.html", remote_path=WWW_DOC_ROOT)
  sudo("chown -R %s.%s %s" %(WWW_USER, WWW_GROUP, WWW_DOC_ROOT))

  print "Setting up the vhost..."
  sudo("chown -R %s.%s %s" %(env.user, env.user, 
APACHE_SITES_PATH))

  # upload a pre-configured vhost.conf
  put(local_path="vhost.conf", remote_path=APACHE_SITES_PATH)
  sudo("chown -R %s.%s %s" %('root', 'root', APACHE_SITES_PATH))

  # restart Apache to take effect
  sudo("%s restart" %APACHE_INIT_SCRIPT)
  print "Setup complete. Now open the server path 
http://abc.remote-server.org/ in your web browser."
```

为了运行此脚本，应在您的宿主文件中添加以下行，例如，`/etc/hosts`：

```py
127.0.0.1 abc.remote-server.org abc 

```

您还应该创建一个快捷方式，`fabfile.py`。从命令行，您可以通过输入以下命令来完成此操作：

```py
$ ln -sfn 7_7_configure_Apache_for_hosting_website_remotely.py 
fabfile.py

```

然后，您可以通过多种形式调用`fab`可执行文件。

首先，要使用您的脚本登录到远程服务器，您可以运行以下 Fabric 函数。这将产生以下输出：

```py
$ fab remote_server setup_vhost
[127.0.0.1] Executing task 'setup_vhost'
Preparing the Apache vhost setup...
Setting up the document root...
[127.0.0.1] sudo: rm -rf /data/apache/test/
[127.0.0.1] sudo: mkdir -p /data/apache/test/
[127.0.0.1] sudo: chown -R faruq.faruq /data/apache/test/
[127.0.0.1] put: index.html -> /data/apache/test/index.html
[127.0.0.1] sudo: chown -R www-data.www-data /data/apache/test/
Setting up the vhost...
[127.0.0.1] sudo: chown -R faruq.faruq /etc/apache2/sites-enabled/
[127.0.0.1] put: vhost.conf -> /etc/apache2/sites-enabled/vhost.conf
[127.0.0.1] sudo: chown -R root.root /etc/apache2/sites-enabled/
[127.0.0.1] sudo: /etc/init.d/apache2 restart
[127.0.0.1] out: Restarting web server: apache2apache2: Could not reliably determine the server's fully qualified domain name, using 127.0.0.1 for ServerName
[127.0.0.1] out:  ... waiting apache2: Could not reliably determine the server's fully qualified domain name, using 127.0.0.1 for ServerName
[127.0.0.1] out: .
[127.0.0.1] out: 

Setup complete. Now open the server path http://abc.remote-server.org/ in your web browser.

Done.
Disconnecting from 127.0.0.1... done.

```

运行此配方后，您可以在浏览器中打开并尝试访问您在主机文件（例如，`/etc/hosts`）上设置的路径。它应该在您的浏览器上显示以下输出：

```py
It works! 
This is the default web page for this server.
The web server software is running but no content has been added, 
yet.

```

## 它是如何工作的...

此配方将初始 Apache 配置参数设置为常量，然后定义两个函数。在 `remote_server()` 函数中，放置了常用的 Fabric 环境参数，例如，主机、用户、密码等。

`setup_vhost()` 函数执行一系列特权命令。首先，它使用 `exists()` 函数检查网站的文档根路径是否已经创建。如果已存在，它将删除该路径并在下一步中重新创建。使用 `chown` 命令确保该路径由当前用户拥有。

在下一步中，它将一个裸骨 HTML 文件，`index.html`，上传到文档根路径。上传文件后，它将文件的权限重置为 web 服务器用户。

在设置文档根目录后，`setup_vhost()` 函数将提供的 `vhost.conf` 文件上传到 Apache 网站配置路径。然后，它将该路径的所有者设置为 root 用户。

最后，脚本重新启动 Apache 服务，以便激活配置。如果配置成功，当您在浏览器中打开 URL [`abc.remote-server.org/`](http://abc.remote-server.org/) 时，您应该看到之前显示的示例输出。
