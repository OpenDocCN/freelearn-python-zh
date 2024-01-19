# 第十四章：评估

# 第一章：使用 Python 脚本

1.  Python 2.x 和 3.x 之间有什么区别？

Python 3.x 中的 Unicode 支持已经得到改进。其他更改涉及`print`和`exec`函数，这些函数已经调整为更易读和一致。

1.  Python 开发人员使用的主要编程范式是什么？

面向对象编程。

1.  Python 中的哪种数据结构允许我们将值与键关联起来？

Python 字典数据结构提供了一个哈希表，可以存储任意数量的 Python 对象。字典由包含键和值的项目对组成。

1.  Python 脚本的主要开发环境是什么？

PyCharm、Wing IDE 和 Python IDLE。

1.  作为 Python 开发安全工具的一套最佳实践，我们可以遵循什么方法论？

**安全工具开发的开放方法论**（**OMSTD**）

1.  帮助创建隔离的 Python 环境的 Python 模块是什么？

`virtualenv`

1.  哪个工具允许我们创建一个基础项目，从而可以开始开发我们自己的工具？

**安全工具构建者**（**SBT**）

1.  在 Python 开发环境中如何调试变量？

通过添加断点。这样，我们可以在我们设置断点的地方调试并查看变量的内容。

1.  在 PyCharm 中如何添加断点？

我们可以在调试工具窗口中使用`call`函数设置断点。

1.  如何在 Wing IDE 中添加断点？

我们可以在调试选项菜单中使用`call`函数设置断点。

# 第二章：系统编程包

1.  允许我们与 Python 解释器交互的主要模块是什么？

系统（`sys`）模块。

1.  允许我们与操作系统环境、文件系统和权限交互的主要模块是什么？

操作系统（`os`）模块

1.  用于列出当前工作目录内容的模块和方法是什么？

操作系统（`os`）模块和`getcwd()`方法。

1.  哪个模块用于通过`call()`函数执行命令或调用进程？

`>>> subprocess.call("cls", shell=True)`

1.  在 Python 中，我们可以采用什么方法来轻松安全地处理文件和异常？

我们可以使用上下文管理器方法和`with`语句。

1.  进程和线程之间有什么区别？

进程是完整的程序。线程类似于进程：它们也是正在执行的代码。但是，线程在进程内执行，并且进程的线程之间共享资源，如内存。

1.  Python 中用于创建和管理线程的主要模块是什么？

有两个选项：

`thread`模块提供了编写多线程程序的原始操作。

`threading`模块提供了更方便的接口。

1.  Python 在处理线程时存在的限制是什么？

Python 中的线程执行受全局解释器锁（GIL）控制，因此一次只能执行一个线程，而不受机器处理器数量的影响。

1.  提供了一个高级接口，以异步方式执行输入/输出任务的类是哪个？`ThreadPoolExecutors`提供了一个简单的抽象，可以同时启动多个线程，并使用这些线程以并发方式执行任务。

1.  `threading`模块中用于确定哪个线程执行了的函数是什么？

我们可以使用`threading.current_thread()`函数来确定哪个线程执行了当前任务。

# 第三章：套接字编程

1.  `sockets`模块的哪个方法允许从 IP 地址获取域名？

通过`gethostbyaddr(address)`方法，我们可以从 IP 地址获取域名。

1.  `socket`模块的哪个方法允许服务器套接字接受来自另一台主机的客户端套接字的请求？

`socket.accept()`用于接受来自客户端的连接。此方法返回两个值：`client_socket`和`client_address`，其中`client_socket`是用于在连接上发送和接收数据的新套接字对象。

1.  `socket`模块的哪种方法允许将数据发送到给定的地址？

`socket.sendto(data, address)`用于将数据发送到给定的地址。

1.  `socket`模块的哪种方法允许您将主机和端口与特定的套接字关联起来？

`bind(IP,PORT)`方法允许将主机和端口与特定的套接字关联；例如，

`>>> server.bind((“localhost”, 9999))`.

1.  TCP 和 UDP 协议之间的区别是什么，以及如何在 Python 中使用`socket`模块实现它们？

TCP 和 UDP 之间的主要区别是 UDP 不是面向连接的。这意味着我们的数据包没有保证会到达目的地，并且如果传递失败，也不会收到错误通知。

1.  `socket`模块的哪种方法允许您将主机名转换为 IPv4 地址格式？

`socket.gethostbyname(hostname)`

1.  `socket`模块的哪种方法允许您使用套接字实现端口扫描并检查端口状态？

`socket.connect_ex(address)`用于使用套接字实现端口扫描。

1.  `socket`模块的哪个异常允许您捕获与等待时间到期相关的异常？

`socket.timeout`

1.  `socket`模块的哪个异常允许您捕获在搜索 IP 地址信息时发生的错误？

`socket.gaierror`异常，带有消息“连接到服务器的错误：[Errno 11001] getaddrinfo 失败”。

1.  `socket`模块的哪个异常允许您捕获通用输入和输出错误和通信？

`socket.error`

# 第四章：HTTP 编程

1.  哪个模块最容易使用，因为它旨在简化对 REST API 的请求？

`requests`模块。

1.  通过传递一个字典类型的数据结构来进行 POST 请求的正确方法是什么，该数据结构将被发送到请求的正文中？

`response = requests.post(url, data=data)`

1.  通过代理服务器正确地进行 POST 请求并同时修改标头信息的方法是什么？

`requests.post(url,headers=headers,proxies=proxy)`

1.  如果我们需要通过代理发送请求，需要挂载哪种数据结构？

字典数据结构；例如，`proxy = {“protocol”:”ip:port”}`。

1.  如果在`response`对象中有服务器的响应，我们如何获得服务器返回的 HTTP 请求代码？

`response.status_code`

1.  我们可以使用哪个模块来指示我们将使用`PoolManager`类预留的连接数？

`urllib3`

1.  `requests`库的哪个模块提供执行摘要类型身份验证的可能性？

`HTTPDigestAuth`

1.  基本身份验证机制使用什么编码系统发送用户名和密码？

HTTP 基本身份验证机制基于表单，使用`Base64`对由冒号分隔的用户名和密码组合进行编码。

1.  使用一种单向哈希加密算法（MD5）来改进基本身份验证过程的机制是什么？

HTTP 摘要身份验证机制使用 MD5 加密用户、密钥和领域哈希。

1.  用于识别我们用于向 URL 发送请求的浏览器和操作系统的标头是哪个？

**User-Agent**标头。

# 第五章：分析网络流量

1.  Scapy 函数可以以与`tcpdump`和 Wireshark 等工具相同的方式捕获数据包是什么？

`scapy> pkts = sniff (iface = "eth0", count = n)`，其中`n`是数据包的数量。

1.  使用 Scapy 以循环的形式每五秒发送一个数据包的最佳方法是什么？

`scapy> sendp (packet, loop=1, inter=5)`

1.  在 Scapy 中必须调用哪个方法来检查某个机器上的某个端口（`port`）是否打开或关闭，并且显示有关数据包发送方式的详细信息？

`scapy> sr1(IP(dst=host)/TCP(dport=port), verbose=True)`

1.  在 Scapy 中实现 `traceroute` 命令需要哪些函数？

`IP`/`UDP`/`sr1`

1.  哪个 Python 扩展模块与 `libpcap` 数据包捕获库进行接口？

`Pcapy.`

1.  `Pcapy` 接口中的哪个方法允许我们在特定设备上捕获数据包？

我们可以使用 Pcapy 接口中的 `open_live` 方法来捕获特定设备上的数据包，并且可以指定每次捕获的字节数和其他参数，比如混杂模式和超时。

1.  在 Scapy 中发送数据包的方法有哪些？

`send(): sends layer-3 packets`

`sendp(): sends layer-2 packets`

1.  `sniff` 函数的哪个参数允许我们定义一个将应用于每个捕获数据包的函数？

`prn` 参数将出现在许多其他函数中，并且如文档中所述，是指函数作为输入参数。以下是一个例子：

`>>> packet=sniff(filter="tcp", iface="eth0", prn=lambda x:x.summary())`

1.  Scapy 支持哪种格式来应用网络数据包过滤器？

**伯克利数据包过滤器**（**BPFs**）

1.  哪个命令允许您跟踪数据包（IP 数据包）从计算机 A 到计算机 B 的路径？

`**traceroute**`

# 第六章：从服务器收集信息

1.  我们需要什么来访问 Shodan 开发者 API？

在 Shodan 网站注册并使用 `API_KEY`，这将使您访问他们的服务。

1.  在 Shodan API 中应该调用哪个方法以获取有关给定主机的信息，该方法返回什么数据结构？

该方法是 `host()` 方法，它返回字典数据结构。

1.  哪个模块可以用于获取服务器的横幅？

我们需要使用 `sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)` 创建一个套接字，使用 `sock.sendall(http_get)` 发送 GET 请求，最后使用 `data = sock.recvfrom(1024)` 接收数据。

1.  应该调用哪个方法并传递哪些参数以获取 `DNSPython` 模块中的 IPv6 地址记录？

`dns.resolver.query('domain','AAAA')`

1.  应该调用哪个方法并传递哪些参数以获取 `DNSPython` 模块中邮件服务器的记录？

`dns.resolver.query('domain','MX')`

1.  应该调用哪个方法并传递哪些参数以获取 `DNSPython` 模块中的名称服务器记录？

`dns.resolver.query('domain','NS')`

1.  哪个项目包含文件和文件夹，其中包含在各种渗透测试中收集的已知攻击模式？

`FuzzDB` 项目提供了分为不同目录的类别，这些目录包含可预测的资源位置模式和用于检测带有恶意有效负载或易受攻击路由的漏洞模式。

1.  应该使用哪个模块来查找可能存在漏洞的服务器上的登录页面？

`fuzzdb.Discovery.PredictableRes.Logins`

1.  哪个 FuzzDB 项目模块允许我们获取用于检测 SQL 注入型漏洞的字符串？

`fuzzdb.attack_payloads.sql_injection.detect.GenericBlind`

1.  DNS 服务器用哪个端口来解析邮件服务器名称的请求？

`53(UDP)`

# 第七章：与 FTP、SSH 和 SNMP 服务器交互

1.  如何使用 `ftplib` 模块通过 `connect()` 和 `login()` 方法连接到 FTP 服务器？

`ftp = FTP()`

`ftp.connect(host, 21)`

`ftp.login(‘user’, ‘password’)`

1.  `ftplib` 模块的哪个方法允许列出 FTP 服务器的文件？

`FTP.dir()`

1.  Paramiko 模块的哪个方法允许我们连接到 SSH 服务器，并使用哪些参数（主机、用户名、密码）？

`ssh = paramiko.SSHClient()`

`ssh.connect(host, username=’username’, password=’password’)`

1.  Paramiko 模块的哪种方法允许我们打开一个会话以便随后执行命令？

`ssh_session = client.get_transport().open_session()`

1.  我们如何使用找到的路由和密码从 RSA 证书登录到 SSH 服务器？

`rsa_key= RSAKey.from_private_key_file('path_key_rsa',password)`

`client.connect('host',username='',pkey= rsa_key,password='')`

1.  `PySNMP`模块的哪个主要类允许对 SNMP 代理进行查询？

`CommandGenerator`。以下是其使用示例：

`from pysnmp.entity.rfc3413.oneliner import cmdgen`

`cmdGen = cmdgen.CommandGenerator()`

1.  在不中断会话或提示用户的情况下，通知 Paramiko 首次接受服务器密钥的指令是什么？

`ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())`

1.  通过`Transport()`方法连接到 SSH 服务器的另一种对象类型提供了另一种身份验证方式？

`transport = paramiko.Transport(ip_address)`

`transport.start_client()`

1.  基于 Paramiko 的 Python FTP 模块，以安全方式与 FTP 服务器建立连接是什么？

`pysftp`，它基于 paramiko。

1.  我们需要使用`ftplib`的哪种方法来下载文件，以及需要执行哪个`ftp`命令？

`file_handler = open(DOWNLOAD_FILE_NAME, 'wb')`

`ftp_cmd = 'RETR %s' %DOWNLOAD_FILE_NAME`

`ftp_client.retrbinary(ftp_cmd,file_handler.write)`

# 第八章：使用 Nmap 扫描器

1.  哪种方法允许我们查看已被定位扫描的机器？

`nmap.all_hosts()`

1.  如果我们想执行异步扫描并在扫描结束时执行脚本，我们如何调用`scan`函数？

`nmasync.scan('ip','ports',arguments='--script=/usr/local/share/nmap/scripts/')`

1.  我们可以使用哪种方法以字典格式获取扫描结果？

`nmap.csv()`

1.  用于执行异步扫描的 Nmap 模块是什么类型的？

`nma = nmap.PortScannerAsync()`

1.  用于执行同步扫描的 Nmap 模块是什么类型的？

`nma = nmap.PortScanner()`

1.  如何在给定的主机和给定的端口上启动同步扫描，如果我们使用`self.nmsync = nmap.PortScanner()`指令初始化对象？

`self.nmsync.scan(hostname, port)`

1.  我们可以使用哪种方法来检查特定网络中的主机是否在线？

我们可以使用`state()`函数来查看主机是否在线。以下是其使用示例：

`nmap[‘127.0.0.1’].state()`

1.  当我们使用`PortScannerAsync()`类执行异步扫描时，需要定义哪个函数？

在执行扫描时，我们可以指定一个额外的回调参数，其中定义了`return`函数，该函数将在扫描结束时执行。以下是一个例子：

`def callback_result(host, scan_result)`

`nmasync.scan(hosts=’127.0.0.1’, arguments=’-sP’, callback=callback_result)`

1.  如果我们需要知道 FTP 服务是否允许匿名身份验证而无需输入用户名和密码，我们需要在端口`21`上运行哪个脚本？

`ftp-anon.nse`

1.  如果我们需要知道 MySQL 服务是否允许匿名身份验证而无需输入用户名和密码，我们需要在端口`3306`上运行哪个脚本？

`mysql-enum.nse`

# 第九章：与 Metasploit 框架连接

1.  与模块交互和执行 Metasploit 中的利用的接口是什么？

`msfconsole`

1.  使用 Metasploit 框架利用系统的主要步骤是什么？

使用 Metasploit 框架利用系统的五个步骤如下：

1. 配置主动利用

2. 验证利用选项

3. 选择目标

4. 选择有效载荷

5. 启动利用

1.  Metasploit 框架用于客户端和 Metasploit 服务器实例之间交换信息的接口名称是什么？

MSGRPC 接口使用`MessagePack`格式在 Metasploit Framework 实例和客户端之间交换信息。

1.  `generic/shell_bind_tcp`和`generic/shell_reverse_tcp`之间的区别是什么？

它们之间的区别在于，使用`generic/shell_bind_tcp`时，连接是从攻击者的机器到受害者的机器建立的，而使用`generic/shell_reverse_tcp`时，连接是从受害者的机器建立的，这需要攻击者的机器有一个监听以检测该连接的程序。

1.  我们可以执行哪个命令来连接`msfconsole`？

`./msfrpcd -U user -P password -p 55553 -n -f`

这样，Metasploit 的 RPC 接口在端口`55553`上监听。

1.  我们需要使用哪个函数以与`msfconsole`实用程序相同的方式与框架交互？

1.  我们使用`console.create`函数，然后使用该函数返回的控制台标识符，如下所示：

导入 msfrpc

`client = msfrpc.Msfrpc({'uri':'/msfrpc', 'port':'5553', 'host':'127.0.0.1', 'ssl': True})`

`client.call('console.create')`

1.  使用 Metasploit Framework 在客户端和 Metasploit 服务器实例之间交换信息的远程访问接口的名称是什么？

`MSGRPC`

1.  我们如何可以从 Metasploit 服务器获取所有利用的列表？

要获取利用，可以在使用该工具时使用`**show exploits**`命令。

1.  Metasploit Framework 中哪些模块可以访问 Apache Tomcat 中的应用程序管理器并利用 Apache Tomcat 服务器以获取会话 meterpreter？

在 Metasploit Framework 中，有一个名为`tomcat_mgr_login`的辅助模块，它为攻击者提供用户名和密码以访问 Tomcat Manager。

1.  在 Tomcat 服务器上执行利用时建立 meterpreter 会话的有效负载名称是什么？

`java/meterpreter/bind_tcp`

# 第十章：与漏洞扫描仪交互

1.  考虑一组标准化和易于衡量的标准，评分漏洞的主要机制是什么？

**通用漏洞评分系统**（**CVSS**）

1.  我们使用哪个包和类来与 Python 交互 Nessus？

`from nessrest import ness6rest`

1.  `nessrest`模块中哪个方法在特定目标上启动扫描？

`scan = ness6rest.Scanner(url="https://nessusscanner:8834", login="username", password="password")`

1.  `nessrest`模块中哪个方法获取特定目标扫描的详细信息？

`scan_details(self, name)`方法获取所请求扫描的详细信息。

1.  用 Python 连接`nexpose`服务器的主要类是什么？

要使用`nexpose`服务器连接 Python，我们使用`pynexpose.py`文件中的`NeXposeServer`类。

1.  负责列出所有检测到的漏洞并返回特定漏洞详情的方法是什么？

`vulnerability_listing()`和`vulnerability_details()`方法负责列出所有检测到的漏洞并返回特定漏洞的详情。

1.  允许我们解析并获取从`nexpose`服务器获取的信息的 Python 模块的名称是什么？

`BeautifulSoup`。

1.  允许我们连接到`NexPose`漏洞扫描仪的 Python 模块的名称是什么？

`Pynexpose`模块允许从 Python 对位于 Web 服务器上的漏洞扫描器进行编程访问。

1.  允许我们连接到`Nessus`漏洞扫描仪的 Python 模块的名称是什么？

`nessrest`。

1.  `Nexpose`服务器以何种格式返回响应以便从 Python 中简单处理？

XML。

# 第十一章：识别 Web 应用程序中的服务器漏洞

1.  哪种类型的漏洞是一种攻击，将恶意脚本注入到网页中，以重定向用户到假网站或收集个人信息？

**跨站脚本**（**XSS**）允许攻击者在受害者的浏览器中执行脚本，从而允许他们劫持用户会话或将用户重定向到恶意站点。

1.  攻击者将 SQL 数据库命令插入到 Web 应用程序使用的订单表单的数据输入字段的技术是什么？

SQL 注入是一种利用`未经验证`输入漏洞来窃取数据的技术。基本上，它是一种代码注入技术，攻击者执行恶意 SQL 查询，控制 Web 应用程序的数据库。

您希望防止浏览器运行潜在有害的 JavaScript 命令。什么工具可以帮助您检测与 JavaScript 相关的 Web 应用程序中的漏洞？

您可以使用`xssscrapy`来检测 XSS 漏洞。

1.  什么工具允许您从网站获取数据结构？

`Scrapy`是 Python 的一个框架，允许您执行网络抓取任务和网络爬行过程以及数据分析。它允许您递归扫描网站的内容，并对内容应用一组规则，以提取可能对您有用的信息。

1.  什么工具允许您检测与 JavaScript 相关的 Web 应用程序中的漏洞？

`Sqlmap`和`xsscrapy`。

1.  w3af 工具的哪个配置文件执行扫描以识别更高风险的漏洞，如 SQL 注入和 XSS？

`audit_high_risk`配置文件执行扫描以识别更高风险的漏洞，如 SQL 注入和 XSS。

1.  w3af API 中的主要类是包含启用插件、建立攻击目标和管理配置文件所需的所有方法和属性的类是什么？

在整个攻击过程中，最重要的是管理`core.controllers.w3afCore`模块的`w3afCore`类。该类的实例包含启用插件、建立攻击目标、管理配置文件以及启动、中断和停止攻击过程所需的所有方法和属性。

1.  哪个`slmap`选项列出所有可用的数据库？

`dbs`选项。以下是其使用示例：

`>>>sqlmap -u http://testphp.productweb.com/showproducts.php?cat=1 –dbs`

1.  允许在服务器中扫描 Heartbleed 漏洞的 Nmap 脚本的名称是什么？

`ssl-heartbleed`

1.  哪个过程允许我们与服务器建立 SSL 连接，包括对称和非对称密钥的交换，以在客户端和服务器之间建立加密连接？

`HandShake`确定将用于加密通信的密码套件，验证服务器，并在实际数据传输之前建立安全连接。

# 第十二章：从文档、图像和浏览器中提取地理位置和元数据

1.  哪个 Python 模块允许我们从 IP 地址检索地理信息？

`pygeoip`允许您从 IP 地址检索地理信息。它基于 GeoIP 数据库，这些数据库根据类型（类型为`city`、`region`、`country`、`ISP`）分布在几个文件中。

1.  哪个模块使用 Google Geocoding API v3 服务来检索特定地址的坐标？

`pygeocoder`是一个简化使用 Google 地理位置功能的 Python 模块。使用此模块，您可以轻松找到与坐标对应的地址，反之亦然。我们还可以使用它来验证和格式化地址。

1.  允许根据地点的描述和特定位置进行查询的`pygeocoder`模块的主要类是什么？

该模块的主要类是`Geocoder`类，它允许根据地点的描述和特定位置进行查询。

1.  哪个方法允许反转过程，从对应于纬度和经度的坐标中恢复给定站点的地址？

`results = Geocoder.reverse_geocode(results.latitude, results.longitude)`

1.  `pygeoip`模块中的哪个方法允许我们从传递的 IP 地址获取国家名称的值？

`country_name_by_addr(<ip_address>)`

1.  `pygeoip`模块中的哪个方法允许我们从 IP 地址获取以字典形式的地理数据（国家、城市、地区、纬度、经度）？

`record_by_addr(<ip_address>)`

1.  `pygeoip`模块中的哪个方法允许我们从域名中获取组织的名称？

`org_by_name(<domain_name>)`

1.  哪个 Python 模块允许我们从 PDF 文档中提取元数据？

`PyPDF2`

1.  我们可以使用哪个类和方法来从 PDF 文档中获取信息？

`PyPDF2`模块提供了提取文档信息以及加密和解密文档的功能。要提取元数据，我们可以使用`PdfFileReader`类和`getDocumentInfo()`方法，它返回一个包含文档数据的字典。

1.  哪个模块允许我们从 EXIF 格式的标签中提取图像信息？

`PIL.ExifTags`用于获取图像的 EXIF 标签信息；图像对象的`_getexif()`方法可用。

# 第十三章：密码学和隐写术

1.  哪种算法类型使用相同的密钥来加密和解密数据？

对称加密。

1.  哪种算法类型使用两个不同的密钥，一个用于加密，另一个用于解密？

公钥算法使用两个不同的密钥：一个用于加密，另一个用于解密。这项技术的用户发布其公钥，同时保持其私钥保密。这使得任何人都可以使用公钥加密发送给他们的消息，只有私钥的持有者才能解密。

1.  在`pycrypto`中，我们可以使用哪个包来使用诸如 AES 之类的加密算法？

`from Crypto.Cipher import AES`

1.  对于哪种算法，我们需要确保数据的长度是 16 字节的倍数？

AES 加密。

1.  对于密码模块，我们可以使用哪个包进行对称加密？

`fernet`包是对称加密的一种实现，并保证加密的消息在没有密钥的情况下无法被篡改或阅读。以下是其使用示例：

`from cryptography.fernet import Fernet`

1.  从密码中派生加密密钥使用哪种算法？

**基于密码的密钥派生函数 2**（**PBKDF2**）。对于密码模块，我们可以使用`cryptography.hazmat.primitives.kdf.pbkdf2`包中的`PBKDF2HMAC`。

1.  `fernet`包为对称加密提供了什么，并且用于生成密钥的方法是什么？

`fernet`包是对称加密的一种实现，并保证加密的消息在没有密钥的情况下无法被篡改或阅读。要生成密钥，我们可以使用以下代码：

`from cryptography.fernet import Fernet`

`key = Fernet.generate_key()`

1.  哪个类提供了`ciphers`包的对称加密？

`cryptography.hazmat.primitives.ciphers.Cipher`

1.  `stepic`中的哪个方法生成带有隐藏数据的图像，从现有图像和任意数据开始？

`encode(image,data)`

1.  `pycrypto`中的哪个包包含一些哈希函数，允许单向加密？

`from Crypto.Hash import [Hash Type]`
