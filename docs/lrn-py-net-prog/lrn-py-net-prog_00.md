# 前言

欢迎来到使用 Python 进行网络编程的世界。Python 是一种功能齐全的面向对象的编程语言，具有一个标准库，其中包括了快速构建强大网络应用所需的一切。此外，它还有大量的第三方库和包，将 Python 扩展到网络编程的各个领域。结合使用 Python 的乐趣，我们希望通过这本书让您开始您的旅程，以便掌握这些工具并产生一些出色的网络代码。

在本书中，我们专注于 Python 3。尽管 Python 3 仍在确立自己作为 Python 2 的继任者，但第 3 版是该语言的未来，我们希望证明它已经准备好用于网络编程。它相对于以前的版本有许多改进，其中许多改进都提高了网络编程体验，包括增强的标准库模块和新的添加。

我们希望您喜欢这本关于使用 Python 进行网络编程的介绍。

# 本书内容

第一章，网络编程和 Python，介绍了对网络编程新手来说的核心网络概念，并介绍了 Python 中的网络编程方法。

第二章，HTTP 和网络工作，向您介绍了 HTTP 协议，并介绍了如何使用 Python 作为 HTTP 客户端检索和操作 Web 内容。我们还研究了标准库`urllib`和第三方`Requests`模块。

第三章，API 的实际应用，向您介绍了使用 HTTP 使用 Web API。我们还介绍了 XML 和 JSON 数据格式，并指导您使用 Amazon Web Services Simple Storage Service（S3）和 Twitter API 开发应用程序。

第四章，与电子邮件互动，涵盖了发送和接收电子邮件时使用的主要协议，如 SMTP、POP3 和 IMAP，以及如何在 Python 3 中使用它们。

第五章，与远程系统交互，指导您如何使用 Python 连接服务器并执行常见的管理任务，包括通过 SSH 执行 shell 命令，使用 FTP 和 SMB 进行文件传输，使用 LDAP 进行身份验证以及使用 SNMP 监视系统。

第六章，IP 和 DNS，讨论了 Internet Protocol（IP）的细节，以及在 Python 中处理 IP 的方法，以及如何使用 DNS 解析主机名。

第七章，使用套接字编程，涵盖了使用 Python 编写 TCP 和 UDP 套接字以编写低级网络应用程序。我们还介绍了用于安全数据传输的 HTTPS 和 TLS。

第八章，客户端和服务器应用程序，介绍了为基于套接字的通信编写客户端和服务器程序。通过编写一个回显应用程序和一个聊天应用程序，我们研究了开发基本协议、构建网络数据的方法，并比较了多线程和基于事件的服务器架构。

第九章，Web 应用程序，向您介绍了如何在 Python 中编写 Web 应用程序。我们涵盖了主要方法，Python Web 应用程序的托管方法，并在 Flask 微框架中开发了一个示例应用程序。

附录，使用 Wireshark，涵盖了数据包嗅探器、Wireshark 的安装以及如何使用 Wireshark 应用程序捕获和过滤数据包。

# 本书所需内容

本书针对 Python 3。虽然许多示例可以在 Python 2 中运行，但使用最新版本的 Python 3 来完成本书的学习会获得最佳体验。在撰写本文时，最新版本是 3.4.3，并且示例已针对此版本进行了测试。

尽管 Python 3.4 是首选版本，所有示例都应该在 Python 3.1 或更高版本上运行，除了以下情况：

+   第八章中的`asyncio`示例，*客户端和服务器应用*，因为`asyncio`模块只包含在 3.4 版本中

+   第九章中的 Flask 示例，*网络应用*，需要 Python 3.3 或更高版本

我们还针对 Linux 操作系统，并假设您正在使用 Linux 操作系统。尽管示例已在 Windows 上进行了测试，但我们会注意到在需求或结果方面可能存在差异的地方。

## 虚拟环境

强烈建议您在使用本书时使用 Python 虚拟环境，或者“venvs”，实际上，在使用 Python 进行任何工作时都应该使用。venv 是 Python 可执行文件和相关文件的隔离副本，为安装 Python 模块提供了一个独立的环境，独立于系统 Python 安装。您可以拥有尽可能多的 venv，这意味着您可以设置多个模块配置，并且可以轻松地在它们之间切换。

从 3.3 版本开始，Python 包括一个`venv`模块，提供了这个功能。文档和示例可以在[`docs.python.org/3/using/scripts.html`](https://docs.python.org/3/using/scripts.html)找到。还有一个独立的工具可用于早期版本，可以在[`virtualenv.pypa.io/en/latest/`](https://virtualenv.pypa.io/en/latest/)找到。

## 安装 Python 3

大多数主要的 Linux 发行版都预装了 Python 2。在这样的系统上安装 Python 3 时，重要的是要注意我们并没有替换 Python 2 的安装。许多发行版使用 Python 2 进行核心系统操作，并且这些操作将针对系统 Python 的主要版本进行调整。替换系统 Python 可能会对操作系统的运行产生严重后果。相反，当我们安装 Python 3 时，它会与 Python 2 并存。安装 Python 3 后，可以使用`python3.x`可执行文件来调用它，其中的`x`会被相应安装的次要版本替换。大多数软件包还提供了指向这个可执行文件的`symlink`，名为`python3`，可以代替运行。

大多数最新发行版都提供了安装 Python 3.4 的软件包，我们将在这里介绍主要的发行版。如果软件包不可用，仍然有一些选项可以用来安装一个可用的 Python 3.4 环境。

### Ubuntu 和 Debian

Ubuntu 15.04 和 14.04 已经预装了 Python 3.4；所以如果您正在运行这些版本，您已经准备就绪。请注意，14.04 中存在一个错误，这意味着必须手动安装 pip 在使用捆绑的`venv`模块创建的任何 venv 中。您可以在[`askubuntu.com/questions/488529/pyvenv-3-4-error-returned-non-zero-exit-status-1`](http://askubuntu.com/questions/488529/pyvenv-3-4-error-returned-non-zero-exit-status-1)找到解决此问题的信息。

对于 Ubuntu 的早期版本，Felix Krull 维护了一个最新的 Ubuntu Python 安装的存储库。完整的细节可以在[`launchpad.net/~fkrull/+archive/ubuntu/deadsnakes`](https://launchpad.net/~fkrull/+archive/ubuntu/deadsnakes)找到。

在 Debian 上，Jessie 有一个 Python 3.4 包（`python3.4`），可以直接用`apt-get`安装。Wheezy 有一个 3.2 的包（`python3.2`），Squeeze 有`python3.1`，可以类似地安装。为了在后两者上获得可用的 Python 3.4 安装，最简单的方法是使用 Felix Krull 的 Ubuntu 存储库。

### RHEL、CentOS、Scientific Linux

这些发行版不提供最新的 Python 3 软件包，因此我们需要使用第三方存储库。对于 Red Hat Enterprise Linux、CentOS 和 Scientific Linux，可以从社区支持的软件集合（SCL）存储库获取 Python 3。有关使用此存储库的说明可以在[`www.softwarecollections.org/en/scls/rhscl/python33/`](https://www.softwarecollections.org/en/scls/rhscl/python33/)找到。撰写时，Python 3.3 是最新可用版本。

Python 3.4 可从另一个存储库 IUS 社区存储库中获得，由 Rackspace 赞助。安装说明可以在[`iuscommunity.org/pages/IUSClientUsageGuide.html`](https://iuscommunity.org/pages/IUSClientUsageGuide.html)找到。

### Fedora

Fedora 21 和 22 提供带有`python3`软件包的 Python 3.4：

```py
**$ yum install python3**

```

对于早期版本的 Fedora，请使用前面列出的存储库。

## 备用安装方法

如果您正在使用的系统不是前面提到的系统之一，并且找不到适用于您的系统安装最新的 Python 3 的软件包，仍然有其他安装方法。我们将讨论两种方法，`Pythonz`和`JuJu`。

### Pythonz

Pythonz 是一个管理从源代码编译 Python 解释器的程序。它从源代码下载并编译 Python，并在您的主目录中安装编译的 Python 解释器。然后可以使用这些二进制文件创建虚拟环境。这种安装方法的唯一限制是您需要在系统上安装构建环境（即 C 编译器和支持软件包），以及编译 Python 的依赖项。如果这不包含在您的发行版中，您将需要 root 访问权限来最初安装这些。完整的说明可以在[`github.com/saghul/pythonz`](https://github.com/saghul/pythonz)找到。

### JuJu

JuJu 可以作为最后的手段使用，它允许在任何系统上安装工作的 Python 3.4，而无需 root 访问权限。它通过在您的主目录中的文件夹中创建一个微型 Arch Linux 安装，并提供工具，允许我们切换到此安装并在其中运行命令。使用此方法，我们可以安装 Arch 的 Python 3.4 软件包，并且可以使用此软件包运行 Python 程序。Arch 环境甚至与您的系统共享主目录，因此在环境之间共享文件很容易。JuJu 主页位于[`github.com/fsquillace/juju`](https://github.com/fsquillace/juju)。

JuJu 应该适用于任何发行版。要安装它，我们需要这样做：

```py
**$ mkdir ~/.juju**
**$ curl https:// bitbucket.org/fsquillace/juju-repo/raw/master/juju- x86_64.tar.gz | tar -xz -C ~/.juju**

```

这将下载并提取 JuJu 映像到`~/.juju`。如果您在 32 位系统上运行，需要将`x86_64`替换为`x86`。接下来，设置`PATH`以获取 JuJu 命令：

```py
**$ export PATH=~/.juju/opt/juju/bin:$PATH**

```

将此添加到您的`.bashrc`是个好主意，这样您就不需要每次登录时都运行它。接下来，我们在`JuJu`环境中安装 Python，我们只需要这样做一次：

```py
**$ juju -f**
**$ pacman --sync refresh**
**$ pacman --sync --sysupgrade**
**$ pacman --sync python3**
**$ exit**

```

这些命令首先以 root 身份激活`JuJu`环境，然后使用`pacman` Arch Linux 软件包管理器更新系统并安装 Python 3.4。最后的`exit`命令退出`JuJu`环境。最后，我们可以以普通用户的身份访问`JuJu`环境：

```py
**$ juju**

```

然后我们可以开始使用安装的 Python 3：

```py
**$ python3** 
**Python 3.4.3 (default, Apr 28 2015, 19:59:08)**
**[GCC 4.7.2] on linux**
**Type "help", "copyright", "credits" or "license" for more information.**
**>>>**

```

## Windows

与一些较旧的 Linux 发行版相比，在 Windows 上安装 Python 3.4 相对容易；只需从[`www.python.org`](http://www.python.org)下载 Python 3.4 安装程序并运行即可。唯一的问题是它需要管理员权限才能这样做，因此如果您在受限制的计算机上，事情就会更加棘手。目前最好的解决方案是 WinPython，可以在[`winpython.github.io`](http://winpython.github.io)找到。

## 其他要求

我们假设您有一个正常工作的互联网连接。几章使用互联网资源广泛，而且没有真正的方法来离线模拟这些资源。拥有第二台计算机也对探索一些网络概念以及在真实网络中尝试网络应用程序非常有用。

我们还在几章中使用 Wireshark 数据包嗅探器。这将需要一台具有 root 访问权限（或 Windows 中的管理员访问权限）的机器。Wireshark 安装程序和安装说明可在[`www.wireshark.org`](https://www.wireshark.org)找到。有关使用 Wireshark 的介绍可以在附录中找到，*使用 Wireshark*。

# 这本书是为谁写的

如果您是 Python 开发人员，或者具有 Python 经验的系统管理员，并且希望迈出网络编程的第一步，那么这本书适合您。无论您是第一次使用网络还是希望增强现有的网络和 Python 技能，您都会发现这本书非常有用。

# 约定

在本书中，您会发现一些文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“通过在 Windows 上运行`ip addr`或`ipconfig /all`命令为您的计算机分配了 IP 地址。”

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

我们尽量遵循 PEP 8，但我们也遵循实用性胜过纯粹的原则，并在一些领域偏离。导入通常在一行上执行以节省空间，而且我们可能不严格遵守换行约定，因为这是印刷媒体的特性；我们的目标是“可读性至关重要”。

我们还选择专注于过程式编程风格，而不是使用面向对象的示例。这样做的原因是，熟悉面向对象编程的人通常更容易将过程式示例重新制作为面向对象的格式，而对于不熟悉面向对象编程的人来说，反过来做则更困难。
