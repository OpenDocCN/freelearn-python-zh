# 前言

欢迎来到使用Python进行网络编程的世界。Python是一种功能齐全的面向对象的编程语言，具有一个标准库，其中包括了快速构建强大网络应用所需的一切。此外，它还有大量的第三方库和包，将Python扩展到网络编程的各个领域。结合使用Python的乐趣，我们希望通过这本书让您开始您的旅程，以便掌握这些工具并产生一些出色的网络代码。

在本书中，我们专注于Python 3。尽管Python 3仍在确立自己作为Python 2的继任者，但第3版是该语言的未来，我们希望证明它已经准备好用于网络编程。它相对于以前的版本有许多改进，其中许多改进都提高了网络编程体验，包括增强的标准库模块和新的添加。

我们希望您喜欢这本关于使用Python进行网络编程的介绍。

# 本书内容

第1章，网络编程和Python，介绍了对网络编程新手来说的核心网络概念，并介绍了Python中的网络编程方法。

第2章，HTTP和网络工作，向您介绍了HTTP协议，并介绍了如何使用Python作为HTTP客户端检索和操作Web内容。我们还研究了标准库`urllib`和第三方`Requests`模块。

第3章，API的实际应用，向您介绍了使用HTTP使用Web API。我们还介绍了XML和JSON数据格式，并指导您使用Amazon Web Services Simple Storage Service（S3）和Twitter API开发应用程序。

第4章，与电子邮件互动，涵盖了发送和接收电子邮件时使用的主要协议，如SMTP、POP3和IMAP，以及如何在Python 3中使用它们。

第5章，与远程系统交互，指导您如何使用Python连接服务器并执行常见的管理任务，包括通过SSH执行shell命令，使用FTP和SMB进行文件传输，使用LDAP进行身份验证以及使用SNMP监视系统。

第6章，IP和DNS，讨论了Internet Protocol（IP）的细节，以及在Python中处理IP的方法，以及如何使用DNS解析主机名。

第7章，使用套接字编程，涵盖了使用Python编写TCP和UDP套接字以编写低级网络应用程序。我们还介绍了用于安全数据传输的HTTPS和TLS。

第8章，客户端和服务器应用程序，介绍了为基于套接字的通信编写客户端和服务器程序。通过编写一个回显应用程序和一个聊天应用程序，我们研究了开发基本协议、构建网络数据的方法，并比较了多线程和基于事件的服务器架构。

第9章，Web应用程序，向您介绍了如何在Python中编写Web应用程序。我们涵盖了主要方法，Python Web应用程序的托管方法，并在Flask微框架中开发了一个示例应用程序。

附录，使用Wireshark，涵盖了数据包嗅探器、Wireshark的安装以及如何使用Wireshark应用程序捕获和过滤数据包。

# 本书所需内容

本书针对Python 3。虽然许多示例可以在Python 2中运行，但使用最新版本的Python 3来完成本书的学习会获得最佳体验。在撰写本文时，最新版本是3.4.3，并且示例已针对此版本进行了测试。

尽管Python 3.4是首选版本，所有示例都应该在Python 3.1或更高版本上运行，除了以下情况：

+   [第8章](ch08.html "第8章。客户端和服务器应用")中的`asyncio`示例，*客户端和服务器应用*，因为`asyncio`模块只包含在3.4版本中

+   [第9章](ch09.html "第9章。网络应用")中的Flask示例，*网络应用*，需要Python 3.3或更高版本

我们还针对Linux操作系统，并假设您正在使用Linux操作系统。尽管示例已在Windows上进行了测试，但我们会注意到在需求或结果方面可能存在差异的地方。

## 虚拟环境

强烈建议您在使用本书时使用Python虚拟环境，或者“venvs”，实际上，在使用Python进行任何工作时都应该使用。venv是Python可执行文件和相关文件的隔离副本，为安装Python模块提供了一个独立的环境，独立于系统Python安装。您可以拥有尽可能多的venv，这意味着您可以设置多个模块配置，并且可以轻松地在它们之间切换。

从3.3版本开始，Python包括一个`venv`模块，提供了这个功能。文档和示例可以在[https://docs.python.org/3/using/scripts.html](https://docs.python.org/3/using/scripts.html)找到。还有一个独立的工具可用于早期版本，可以在[https://virtualenv.pypa.io/en/latest/](https://virtualenv.pypa.io/en/latest/)找到。

## 安装Python 3

大多数主要的Linux发行版都预装了Python 2。在这样的系统上安装Python 3时，重要的是要注意我们并没有替换Python 2的安装。许多发行版使用Python 2进行核心系统操作，并且这些操作将针对系统Python的主要版本进行调整。替换系统Python可能会对操作系统的运行产生严重后果。相反，当我们安装Python 3时，它会与Python 2并存。安装Python 3后，可以使用`python3.x`可执行文件来调用它，其中的`x`会被相应安装的次要版本替换。大多数软件包还提供了指向这个可执行文件的`symlink`，名为`python3`，可以代替运行。

大多数最新发行版都提供了安装Python 3.4的软件包，我们将在这里介绍主要的发行版。如果软件包不可用，仍然有一些选项可以用来安装一个可用的Python 3.4环境。

### Ubuntu和Debian

Ubuntu 15.04和14.04已经预装了Python 3.4；所以如果您正在运行这些版本，您已经准备就绪。请注意，14.04中存在一个错误，这意味着必须手动安装pip在使用捆绑的`venv`模块创建的任何venv中。您可以在[http://askubuntu.com/questions/488529/pyvenv-3-4-error-returned-non-zero-exit-status-1](http://askubuntu.com/questions/488529/pyvenv-3-4-error-returned-non-zero-exit-status-1)找到解决此问题的信息。

对于Ubuntu的早期版本，Felix Krull维护了一个最新的Ubuntu Python安装的存储库。完整的细节可以在[https://launchpad.net/~fkrull/+archive/ubuntu/deadsnakes](https://launchpad.net/~fkrull/+archive/ubuntu/deadsnakes)找到。

在Debian上，Jessie有一个Python 3.4包（`python3.4`），可以直接用`apt-get`安装。Wheezy有一个3.2的包（`python3.2`），Squeeze有`python3.1`，可以类似地安装。为了在后两者上获得可用的Python 3.4安装，最简单的方法是使用Felix Krull的Ubuntu存储库。

### RHEL、CentOS、Scientific Linux

这些发行版不提供最新的Python 3软件包，因此我们需要使用第三方存储库。对于Red Hat Enterprise Linux、CentOS和Scientific Linux，可以从社区支持的软件集合（SCL）存储库获取Python 3。有关使用此存储库的说明可以在[https://www.softwarecollections.org/en/scls/rhscl/python33/](https://www.softwarecollections.org/en/scls/rhscl/python33/)找到。撰写时，Python 3.3是最新可用版本。

Python 3.4可从另一个存储库IUS社区存储库中获得，由Rackspace赞助。安装说明可以在[https://iuscommunity.org/pages/IUSClientUsageGuide.html](https://iuscommunity.org/pages/IUSClientUsageGuide.html)找到。

### Fedora

Fedora 21和22提供带有`python3`软件包的Python 3.4：

```py
**$ yum install python3**

```

对于早期版本的Fedora，请使用前面列出的存储库。

## 备用安装方法

如果您正在使用的系统不是前面提到的系统之一，并且找不到适用于您的系统安装最新的Python 3的软件包，仍然有其他安装方法。我们将讨论两种方法，`Pythonz`和`JuJu`。

### Pythonz

Pythonz是一个管理从源代码编译Python解释器的程序。它从源代码下载并编译Python，并在您的主目录中安装编译的Python解释器。然后可以使用这些二进制文件创建虚拟环境。这种安装方法的唯一限制是您需要在系统上安装构建环境（即C编译器和支持软件包），以及编译Python的依赖项。如果这不包含在您的发行版中，您将需要root访问权限来最初安装这些。完整的说明可以在[https://github.com/saghul/pythonz](https://github.com/saghul/pythonz)找到。

### JuJu

JuJu可以作为最后的手段使用，它允许在任何系统上安装工作的Python 3.4，而无需root访问权限。它通过在您的主目录中的文件夹中创建一个微型Arch Linux安装，并提供工具，允许我们切换到此安装并在其中运行命令。使用此方法，我们可以安装Arch的Python 3.4软件包，并且可以使用此软件包运行Python程序。Arch环境甚至与您的系统共享主目录，因此在环境之间共享文件很容易。JuJu主页位于[https://github.com/fsquillace/juju](https://github.com/fsquillace/juju)。

JuJu应该适用于任何发行版。要安装它，我们需要这样做：

```py
**$ mkdir ~/.juju**
**$ curl https:// bitbucket.org/fsquillace/juju-repo/raw/master/juju- x86_64.tar.gz | tar -xz -C ~/.juju**

```

这将下载并提取JuJu映像到`~/.juju`。如果您在32位系统上运行，需要将`x86_64`替换为`x86`。接下来，设置`PATH`以获取JuJu命令：

```py
**$ export PATH=~/.juju/opt/juju/bin:$PATH**

```

将此添加到您的`.bashrc`是个好主意，这样您就不需要每次登录时都运行它。接下来，我们在`JuJu`环境中安装Python，我们只需要这样做一次：

```py
**$ juju -f**
**$ pacman --sync refresh**
**$ pacman --sync --sysupgrade**
**$ pacman --sync python3**
**$ exit**

```

这些命令首先以root身份激活`JuJu`环境，然后使用`pacman` Arch Linux软件包管理器更新系统并安装Python 3.4。最后的`exit`命令退出`JuJu`环境。最后，我们可以以普通用户的身份访问`JuJu`环境：

```py
**$ juju**

```

然后我们可以开始使用安装的Python 3：

```py
**$ python3** 
**Python 3.4.3 (default, Apr 28 2015, 19:59:08)**
**[GCC 4.7.2] on linux**
**Type "help", "copyright", "credits" or "license" for more information.**
**>>>**

```

## Windows

与一些较旧的Linux发行版相比，在Windows上安装Python 3.4相对容易；只需从[http://www.python.org](http://www.python.org)下载Python 3.4安装程序并运行即可。唯一的问题是它需要管理员权限才能这样做，因此如果您在受限制的计算机上，事情就会更加棘手。目前最好的解决方案是WinPython，可以在[http://winpython.github.io](http://winpython.github.io)找到。

## 其他要求

我们假设您有一个正常工作的互联网连接。几章使用互联网资源广泛，而且没有真正的方法来离线模拟这些资源。拥有第二台计算机也对探索一些网络概念以及在真实网络中尝试网络应用程序非常有用。

我们还在几章中使用Wireshark数据包嗅探器。这将需要一台具有root访问权限（或Windows中的管理员访问权限）的机器。Wireshark安装程序和安装说明可在[https://www.wireshark.org](https://www.wireshark.org)找到。有关使用Wireshark的介绍可以在[附录](apa.html "附录 A. 使用Wireshark")中找到，*使用Wireshark*。

# 这本书是为谁写的

如果您是Python开发人员，或者具有Python经验的系统管理员，并且希望迈出网络编程的第一步，那么这本书适合您。无论您是第一次使用网络还是希望增强现有的网络和Python技能，您都会发现这本书非常有用。

# 约定

在本书中，您会发现一些文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟URL、用户输入和Twitter句柄显示如下：“通过在Windows上运行`ip addr`或`ipconfig /all`命令为您的计算机分配了IP地址。”

代码块设置如下：

```py
import sys, urllib.request

try:
    rfc_number = int(sys.argv[1])
except (IndexError, ValueError):
    print('Must supply an RFC number as first argument')
    sys.exit(2)

template = 'http://www.ietf.org/rfc/rfc{}.txt'
url = template.format(rfc_number)
rfc_raw = urllib.request.urlopen(url).read()
rfc = rfc_raw.decode()
print(rfc)
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目会被突出显示：

```py
<body>
...
<div id="content">
<h1>Debian &ldquo;jessie&rdquo; Release Information</h1>
<p>**Debian 8.0** was
released October 18th, 2014.
The release included many major
changes, described in
...
```

任何命令行输入或输出都是这样写的：

```py
**$ python RFC_downloader.py 2324 | less**

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如菜单或对话框中的单词，会在文本中出现，如：“我们可以看到**开始**按钮下面有一个接口列表。”

### 注意

警告或重要说明会出现在这样的框中。

### 提示

提示和技巧会以这种方式出现。

我们尽量遵循PEP 8，但我们也遵循实用性胜过纯粹的原则，并在一些领域偏离。导入通常在一行上执行以节省空间，而且我们可能不严格遵守换行约定，因为这是印刷媒体的特性；我们的目标是“可读性至关重要”。

我们还选择专注于过程式编程风格，而不是使用面向对象的示例。这样做的原因是，熟悉面向对象编程的人通常更容易将过程式示例重新制作为面向对象的格式，而对于不熟悉面向对象编程的人来说，反过来做则更困难。
