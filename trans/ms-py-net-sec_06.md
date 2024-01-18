# 从服务器中收集信息

在本章中，我们将研究主要模块，这些模块允许我们提取服务器以公开方式暴露的信息。通过我们讨论过的工具，我们可以获取可能对我们的渗透测试或审计过程的后续阶段有用的信息。我们将看到诸如Shodan和Banner Grabbing之类的工具，使用`DNSPython`模块获取DNS服务器的信息，以及使用`pywebfuzz`模块进行模糊处理。

本章将涵盖以下主题：

+   收集信息的介绍

+   `Shodan`包作为从服务器中提取信息的工具

+   `Shodan`包作为应用过滤器和在Shodan中搜索的工具

+   如何通过`socket`模块从服务器中提取横幅信息

+   `DNSPython`模块作为从DNS服务器中提取信息的工具

+   `pywebfuzz`模块作为获取特定服务器上可能存在的漏洞地址的工具

# 技术要求

本章的示例和源代码可在GitHub存储库的`chapter 6`文件夹中找到：[https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)。

您需要在本地计算机上安装Python，并且需要一些关于TCP协议和请求的基本知识。

# 收集信息的介绍

收集信息的过程可以使用Python分发中默认安装的模块和简单安装的外部模块来自动化。我们将看到的一些模块允许我们提取服务器和服务的信息，例如域名和横幅。

有许多方法可以从服务器中收集信息：

+   我们可以使用Shodan从公共服务器中提取信息

+   我们可以使用`socket`模块从公共和私人服务器中提取横幅信息

+   我们可以使用`DNSPython`模块从DNS服务器中提取信息

+   我们可以使用`pywebfuzz`模块获取可能的漏洞

# 使用Shodan从服务器中提取信息

在本节中，您将学习使用Shodan从端口扫描、横幅服务器和操作系统版本中获取信息的基础知识。它不是索引网页内容，而是索引有关标头、横幅和操作系统版本的信息。

# Shodan的介绍

Shodan是Sentient Hyper-Optimized Data Access Network的缩写。与传统的搜索引擎不同，Shodan尝试从端口中获取数据。免费版本提供50个结果。如果你知道如何创造性地使用它，你可以发现Web服务器的漏洞。

Shodan是一个搜索引擎，可以让您从路由器、服务器和任何具有IP地址的设备中找到特定信息。我们可以从这项服务中提取的所有信息都是公开的。

Shodan索引了大量的数据，这在搜索连接到互联网的特定设备时非常有帮助。我们可以从这项服务中提取的所有信息都是公开的。

使用Shodan，我们还可以使用REST API进行搜索、扫描和查询：[https://developer.shodan.io/api](https://developer.shodan.io/api)。

# 访问Shodan服务

Shodan是一个搜索引擎，负责跟踪互联网上的服务器和各种类型的设备（例如IP摄像头），并提取有关这些目标上运行的服务的有用信息。

与其他搜索引擎不同，Shodan不搜索网页内容，而是从HTTP请求的标头中搜索有关服务器的信息，例如操作系统、横幅、服务器类型和版本。

Shodan的工作方式与互联网上的搜索引擎非常相似，不同之处在于它不索引找到的服务器的内容，而是索引服务返回的标头和横幅。

它被称为“黑客的谷歌”，因为它允许我们通过应用不同类型的筛选器进行搜索，以恢复使用特定协议的服务器。

要从Python以编程方式使用Shodan，需要在Shodan中拥有一个带有开发人员Shodan密钥的帐户，这样可以让Python开发人员通过其API自动化搜索其服务。如果我们注册为开发人员，我们会获得`SHODAN_API_KEY`，我们将在Python脚本中使用它来执行与[https://developer.shodan.io](https://developer.shodan.io)服务相同的搜索。如果我们注册为开发人员，除了能够获得`API_KEY`之外，我们还有其他优势，比如获得更多结果或使用搜索筛选器。

我们还有一些供开发人员使用的选项，可以让我们发现Shodan服务：

![](assets/64c53d0e-3b41-4761-a549-b76241dadd99.png)

要安装`Python`模块，可以运行`pip install shodan`命令。

Shodan还有一个REST API，可以向其服务发出请求，您可以在[https://developer.shodan.io/api](https://developer.shodan.io/api)找到。

![](assets/e0b54dc4-0fae-482c-aa4d-f7d9a68dd32d.png)

例如，如果我们想进行搜索，我们可以使用`/shodan/host/`端点搜索。为了正确地进行请求，需要指定我们注册时获得的`API_KEY`。

例如，通过这个请求，我们可以获得带有“apache”搜索的搜索结果，返回JSON格式的响应：[https://api.shodan.io/shodan/host/search?key=<your_api_key>&query=apache](https://api.shodan.io/shodan/host/search?key=v4YpsPUJ3wjDxEqywwu6aF5OZKWj8kik&query=apache)。

您可以在官方文档中找到更多信息：

![](assets/cce092a9-3099-4371-91b9-0d940c6855e0.png)

# Shodan筛选器

Shodan有一系列特殊的筛选器，可以让我们优化搜索结果。在这些筛选器中，我们可以突出显示：

+   **after/before**：按日期筛选结果

+   **country**：按两位国家代码筛选结果

+   **city**：通过城市筛选结果

+   **geo**：通过纬度/经度筛选结果

+   **hostname**：通过主机名或域名筛选结果

+   **net**：通过特定范围的IP或网络段筛选结果

+   **os**：执行特定操作系统的搜索

+   **port**：允许我们按端口号筛选

您可以在[http://www.shodanhq.com/help/filters](http://www.shodanhq.com/help/filters)找到更多关于shodan筛选器的信息。

# Shodan搜索与Python

通过Python API提供的`search`函数，可以以与Web界面相同的方式进行搜索。如果我们从Python解释器执行以下示例，我们会发现如果搜索“apache”字符串，我们会得到15,684,960个结果。

在这里，我们可以看到总结果和从解释器执行的`Shodan`模块：

![](assets/1d719031-9e79-4c2e-94a7-5dc5972ab966.png)

我们还可以创建自己的类（**ShodanSearch**），该类具有`__init__`方法，用于初始化我们注册时获得的`API_KEY`的Shodan对象。我们还可以有一个方法，通过参数搜索搜索字符串，并调用shodan的API的搜索方法。

您可以在github存储库的shodan文件夹中的`ShodanSearch.py`文件中找到以下代码：

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shodan
import re

class ShodanSearch:
    """ Class for search in Shodan """
    def __init__(self,API_KEY):
        self.api =  shodan.Shodan(API_KEY)    

    def search(self,search):
        """ Search from the search string"""
        try:
            result = self.api.search(str(search))
            return result
        except Exception as e:
            print 'Exception: %s' % e
            result = []
            return result
```

# 通过给定主机执行搜索

在这个例子中，从Python解释器执行，我们可以看到使用`shodan.host()`方法，可以获取特定IP的信息，比如国家、城市、服务提供商、服务器或版本：

![](assets/64355c0a-012a-4927-abfc-6a9980492d62.png)

我们可以通过**数据数组**进行详细了解，其中可以获取更多关于**ISP**、**位置、纬度和经度**的信息：

![](assets/f89dfd7f-41eb-436a-af00-0cc0c66bf1c2.png)

在先前定义的`ShodanSearch`类中，我们可以定义一个方法，该方法通过主机的IP参数传递，并调用shodan API的`host()`方法：

```py
def get_host_info(self,IP):
""" Get the information that may have shodan on an IP""
    try:
        host = self.api.host(IP)
        return host
    except Exception as e:
        print 'Exception: %s' % e
        host = []
        return host
```

`ShodanSearch`脚本接受搜索字符串和主机的IP地址：

![](assets/ddf0845c-4e96-4e66-a02d-7f13067e2c86.png)

在此示例执行中，我们正在测试IP地址22.253.135.79，以获取此服务器的所有公共信息：

**`python .\ShodanSearch.py -h 23.253.135.79`**

![](assets/c0ab91b6-103b-4a65-abb4-cf9c4c2194e1.png)

# 搜索FTP服务器

您可以搜索具有匿名用户的FTP访问权限并且可以在没有用户名和密码的情况下访问的服务器。

如果我们使用“**端口：21匿名用户登录**”字符串进行搜索，我们将获得那些易受攻击的FTP服务器：

![](assets/81811d2e-2261-434f-b97b-ab5d020a7767.png)

此脚本允许您获取允许匿名FTP访问的服务器中的IP地址列表。

您可以在`ShodanSearch_FTP_Vulnerable.py`文件中找到以下代码：

```py
import shodan
import re
sites =[]
shodanKeyString = 'v4YpsPUJ3wjDxEqywwu6aF5OZKWj8kik'
shodanApi = shodan.Shodan(shodanKeyString)
results = shodanApi.search("port: 21 Anonymous user logged in")
print "hosts number: " + str(len( results['matches']))
for match in results['matches']:
    if match['ip_str'] is not None:
        print match['ip_str']
        sites.append(match['ip_str'])
```

通过执行上述脚本，我们获得了一个IP地址列表，其中包含容易受到匿名登录FTP服务的服务器：

![](assets/fa6007ba-a01f-4818-ac2a-5b62be918960.png)

# 使用Python获取服务器信息

在本节中，您将学习使用套接字和`python-whois`模块从服务器获取横幅和whois信息的基础知识。

# 使用Python提取服务器横幅

横幅显示与Web服务器的名称和正在运行的版本相关的信息。有些暴露了使用的后端技术（PHP、Java、Python）及其版本。生产版本可能存在公共或非公共的故障，因此测试公开暴露的服务器返回的横幅是否暴露了我们不希望公开的某些信息，这总是一个很好的做法。

使用标准的Python库，可以创建一个简单的程序，连接到服务器并捕获响应中包含的服务的横幅。获取服务器横幅的最简单方法是使用`socket`模块。我们可以通过`recvfrom()`方法发送一个get请求并获取响应，该方法将返回一个带有结果的元组。

您可以在`BannerServer.py`文件中找到以下代码：

```py
import socket
import argparse
import re
parser = argparse.ArgumentParser(description='Get banner server')
# Main arguments
parser.add_argument("-target", dest="target", help="target IP", required=True)
parser.add_argument("-port", dest="port", help="port", type=int, required=True)
parsed_args = parser.parse_args()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((parsed_args.target, parsed_args.port))
sock.settimeout(2)
http_get = b"GET / HTTP/1.1\nHost: "+parsed_args.target+"\n\n"
data = ''
try:
    sock.sendall(http_get)
    data = sock.recvfrom(1024)
    data = data[0]
    print data
    headers = data.splitlines()
    #  use regular expressions to look for server header
    for header in headers:
        if re.search('Server:', header):
            print(header)
except socket.error:
    print ("Socket error", socket.errno)
finally:
    sock.close()
```

上述脚本接受**目标**和**端口**作为**参数**：

![](assets/b52f0624-d259-4b14-9aa8-08109c1110ea.png)

在这种情况下，我们获得了端口80上的Web服务器版本：

`**python .\BannerServer.py -target www.google.com -port 80**`

![](assets/9a3bcd8a-75d9-4f68-82eb-90a05b492014.png)

# 查找有关服务器的whois信息

我们可以使用WHOIS协议来查看域名的注册所有者。有一个名为python-whois的Python模块，用于此协议，文档位于[https://pypi.python.org/pypi/python-whois](https://pypi.python.org/pypi/python-whois)，可以使用`pip install python-whois`命令安装。

例如，如果我们想查询某个域的服务器名称和所有者，我们可以通过`get_whois()`方法来查询。该方法返回一个字典结构（`键->值`）。

```py
>>> import pythonwhois
>>> whois = pythonwhois.get_whois(domain)
>>> for key in whois.keys():
>>  print "%s : %s \n" %(key, whois[key])
```

使用`pythonwhois.net.get_root_server()`方法，可以恢复给定域的根服务器：

```py
>>> whois = pythonwhois.net.get_root_server(domain)
```

使用`pythonwhois.net.get_whois_raw()`方法，可以检索给定域的所有信息：

```py
>>> whois = pythonwhois.net.get_whois_raw(domain)
```

在下面的脚本中，我们看到一个完整的示例，我们从中提取信息的域作为参数传递。

您可以在`PythonWhoisExample.py`文件中找到以下代码：

```py
if len(sys.argv) != 2:
    print “[-] usage python PythonWhoisExample.py <domain_name>”
    sys.exit()
print sys.argv[1]
whois = pythonwhois.get_whois(sys.argv[1])
for key in whois.keys():
    print “[+] %s : %s \n” %(key, whois[key])
whois = pythonwhois.net.get_root_server(sys.argv[1])
print whois
whois = pythonwhois.net.get_whois_raw(sys.argv[1])
print whois
```

# 使用DNSPython获取DNS服务器信息

在本节中，我们将在Python中创建一个DNS客户端，并查看此客户端将如何获取有关名称服务器、邮件服务器和IPV4/IPV6地址的信息。

# DNS协议

DNS代表域名服务器，域名服务用于将IP地址与域名链接起来。 DNS是一个全球分布的映射主机名和IP地址的数据库。 它是一个开放和分层的系统，许多组织选择运行自己的DNS服务器。

DNS协议用于不同的目的。 最常见的是：

+   名称解析：给定主机的完整名称，可以获取其IP地址。

+   反向地址解析：这是与上一个相反的机制。 它可以根据IP地址获取与之关联的名称。

+   邮件服务器解析：给定邮件服务器域名（例如gmail.com），可以通过它来进行通信的服务器（例如gmail-smtp-in.l.google.com）。

DNS还是设备用于查询DNS服务器以将主机名解析为IP地址（反之亦然）的协议。 `nslookup`工具附带大多数Linux和Windows系统，并且它允许我们在命令行上查询DNS。 在这里，我们确定python.org主机具有IPv4地址`23.253.135.79`：

`$ nslookup python.org`

这是python.org域的地址解析：

![](assets/bc740de9-7a55-40d3-accb-1884f928fb65.png)

# DNS服务器

人类更擅长记住与对象相关的名称，而不是长序列的数字。 记住google.com域名比IP地址要容易得多。 此外，IP地址可能会因网络基础设施的变动而发生变化，而域名保持不变。

它的操作基于使用分布式和分层数据库，其中存储了域名和IP地址，以及提供邮件服务器位置服务的能力。

DNS服务器位于应用层，通常使用端口53（UDP）。 当客户端发送DNS数据包以执行某种类型的查询时，必须发送要查询的记录类型。 一些最常用的记录是：

+   A：允许您查询IPv4地址

+   AAAA：允许您查询IPv6地址

+   MX：允许您查询邮件服务器

+   NS：允许您查询服务器的名称（名称服务器）

+   TXT：允许您以文本格式查询信息

# DNSPython模块

DnsPython是一个用Python编写的开源库，允许对DNS服务器进行查询记录操作。 它允许访问高级和低级。 在高级别允许查询DNS记录，而在低级别允许直接操作区域，名称和寄存器。

PyPI提供了一些DNS客户端库。 我们将重点关注`dnspython`库，该库可在[http://www.dnspython.org](http://www.dnspython.org)上找到。

安装可以通过python存储库或通过下载github源代码（[https://github.com/rthalley/dnspython](https://github.com/rthalley/dnspython)）并运行`setup.py`安装文件来完成。

您可以使用`easy_install`命令或`pip`命令安装此库：

```py
$ pip install dnspython
```

此模块的主要包括：

```py
import dns
import dns.resolver
```

我们可以从特定域名获取的信息是：

+   邮件服务器记录：ansMX = dns.resolver.query（'domain'，'MX'）

+   名称服务器记录：ansNS = dns.resolver.query（'domain'，'NS'）

+   IPV4地址记录：ansipv4 = dns.resolver.query（'domain'，'A'）

+   IPV6地址记录：ansipv6 = dns.resolver.query（'domain'，'AAAA'）

在此示例中，我们正在对具有`dns.resolver`子模块的主机的IP地址进行简单查询：

```py
import dns.resolver
answers = dns.resolver.query('python.org', 'A')
for rdata in answers:
    print('IP', rdata.to_text())
```

我们可以使用`is_subdomain（）`方法检查一个域是否是另一个域的**子域**：

```py
domain1= dns.name.from_text('domain1')
domain2= dns.name.from_text('domain2')
domain1.is_subdomain(domain2)
```

从IP地址获取域名：

```py
import dns.reversename
domain = dns.reversename.from_address("ip_address")
```

从域名获取IP：

```py
import dns.reversename
ip = dns.reversename.to_address("domain")
```

如果要进行**反向查找**，需要使用`dns.reversename`子模块，如下例所示：

您可以在`DNSPython-reverse-lookup.py`文件中找到以下代码：

```py
import dns.reversename

name = dns.reversename.from_address("ip_address")
print name
print dns.reversename.to_address(name)
```

在这个完整的示例中，我们将域作为参数传递，从中提取信息。

您可以在`DNSPythonServer_info.py`文件中找到以下代码：

```py
import dns
import dns.resolver
import dns.query
import dns.zone
import dns.name
import dns.reversename
import sys

if len(sys.argv) != 2:
    print "[-] usage python DNSPythonExample.py <domain_name>"
    sys.exit()

domain = sys.argv[1]
ansIPV4,ansMX,ansNS,ansIPV6=(dns.resolver.query(domain,'A'), dns.resolver.query(domain,'MX'),
dns.resolver.query(domain, 'NS'),
dns.resolver.query(domain, 'AAAA'))

print('Name Servers: %s' % ansNS.response.to_text())
print('Name Servers: %s' %[x.to_text() for x in ansNS])
print('Ipv4 addresses: %s' %[x.to_text() for x in ansIPV4])
print('Ipv4 addresses: %s' % ansIPV4.response.to_text())
print('Ipv6 addresses: %s' %[x.to_text() for x in ansIPV6])
print('Ipv6 addresses: %s' % ansIPV6.response.to_text())
print('Mail Servers: %s' % ansMX.response.to_text())
for data in ansMX:
    print('Mailserver', data.exchange.to_text(), 'has preference', data.preference)
```

例如，如果我们尝试从python.org域获取信息，我们会得到以下结果。

使用上一个脚本，我们可以从python.org域中获取NameServers：

![](assets/1924876b-2efb-4126-a3e6-06b75175d6e1.png)

在这个截图中，我们可以看到从python.org解析出的**IPV4和IPV6地址**：

![](assets/948e0d86-308e-4b28-a7e9-581c5a719d3b.png)

在这个截图中，我们可以看到从`python.org`解析出的**邮件服务器**：

![](assets/546541b0-aea9-4f57-a64d-21422822a1f7.png)

# 使用模糊测试获取服务器中的易受攻击的地址

在本节中，我们将学习有关模糊测试过程以及如何使用此实践与Python项目来获取容易受到攻击者攻击的URL和地址。

# 模糊测试过程

模糊器是一个程序，其中包含可以针对特定应用程序或服务器可预测的URL的文件。基本上，我们对每个可预测的URL进行请求，如果我们看到响应正常，这意味着我们找到了一个不公开或隐藏的URL，但后来我们发现我们可以访问它。

像大多数可利用的条件一样，模糊测试过程只对不正确地对输入进行消毒或接受超出其处理能力的数据的系统有用。

总的来说，模糊测试过程包括以下**阶段**：

+   **识别目标**：要对应用程序进行模糊测试，我们必须确定目标应用程序。

+   **识别输入**：漏洞存在是因为目标应用程序接受了格式不正确的输入并在未经消毒的情况下处理它。

+   **创建模糊数据**：在获取所有输入参数后，我们必须创建无效的输入数据发送到目标应用程序。

+   **模糊测试**：创建模糊数据后，我们必须将其发送到目标应用程序。我们可以使用模糊数据来监视调用服务时的异常。

+   **确定可利用性**：模糊测试后，我们必须检查导致崩溃的输入。

# FuzzDB项目

FuzzDB是一个项目，其中我们可以找到一组包含已在多次渗透测试中收集的已知攻击模式的文件夹，主要是在Web环境中：[https://github.com/fuzzdb-project/fuzzdb](https://github.com/fuzzdb-project/fuzzdb)。

FuzzDB类别分为不同的目录，这些目录包含可预测的资源位置模式、用于检测带有恶意有效负载或易受攻击的路由的模式：

![](assets/f74dd986-38d2-4cda-9888-b16e47fa734f.png)

# 使用pywebfuzz进行模糊测试

pywebfuzz是一个Python模块，通过暴力方法帮助识别Web应用程序中的漏洞，并提供测试服务器和Web应用程序（如apache服务器、jboss和数据库）漏洞的资源。

该项目的目标之一是简化对Web应用程序的测试。pywebfuzz项目提供了用于测试用户、密码和代码与Web应用程序的值和逻辑。

在Python中，我们找到`pywebfuzz`模块，其中有一组类，允许访问FuzzDB目录并使用它们的有效负载。PyWebFuzz中创建的类结构是按不同的攻击方案组织的；这些方案代表FuzzDB中可用的不同有效负载。

它有一个类结构，负责读取FuzzDB中可用的文件，以便稍后我们可以在Python中使用它们在我们的脚本中。

首先，我们需要导入`fuzzdb`模块：

```py
from pywebfuzz import fuzzdb
```

例如，如果我们想在服务器上搜索登录页面，我们可以使用`fuzzdb.Discovery.PredictableRes.Logins`模块：

```py
logins = fuzzdb.Discovery.PredictableRes.Logins
```

这将返回一个可预测资源的列表，其中每个元素对应于Web服务器中存在的URL，可能是易受攻击的：

![](assets/67c1c29f-5f68-4ad9-852d-a763e089e12c.png)

我们可以在Python中编写一个脚本，在分析的URL给定的情况下，我们可以测试连接到每个登录路由，如果请求返回代码`200`，则页面已在服务器中找到。

在此脚本中，我们可以获取可预测的URL，例如登录、管理员、管理员和默认页面，对于每个组合域+可预测的URL，我们验证返回的状态代码。

您可以在`pywebfuzz_folder`内的`demofuzzdb.py`文件中找到以下代码：

```py
from pywebfuzz import fuzzdb
import requests

logins = fuzzdb.Discovery.PredictableRes.Logins
domain = "http://testphp.vulnweb.com"
  for login in logins:
 print("Testing... "+ domain + login)
 response = requests.get(domain + login)
 if response.status_code == 200:
 print("Login Resource detected: " +login)
```

您还可以获取服务器支持的HTTP方法：

```py
httpMethods= fuzzdb.attack_payloads.http_protocol.http_protocol_methods
```

从python解释器的先前命令的输出显示了可用的HTTP方法：

![](assets/8674b58d-92e1-4f44-b6a2-27eaae40d3fa.png)

您可以在`pywebfuzz_folder`内的`demofuzzdb2.py`文件中找到以下代码：

```py
from pywebfuzz import fuzzdb
import requests
httpMethods= fuzzdb.attack_payloads.http_protocol.http_protocol_methods
domain = "http://www.google.com" for method in httpMethods:
    print("Testing... "+ domain +"/"+ method)
    response = requests.get(domain, method)
    if response.status_code not in range(400,599):
        print(" Method Allowed: " + method)
```

有一个模块允许您在Apache tomcat服务器上搜索可预测的资源：

```py
tomcat = fuzzdb.Discovery. PredictableRes.ApacheTomcat
```

此子模块允许您获取字符串以检测SQL注入漏洞：

```py
fuzzdb.attack_payloads.sql_injection.detect.GenericBlind
```

在这个屏幕截图中，我们可以看到`fuzzdb sql_injection`模块的执行：

![](assets/97ebbc4b-840a-4b52-84a1-eda6490ebeba.png)

在这种情况下返回的信息与项目的GitHub存储库中找到的信息相匹配。[https://github.com/fuzzdb-project/fuzzdb/tree/master/attack/sql-injection/detect](https://github.com/fuzzdb-project/fuzzdb/tree/master/attack/sql-injection/detect)包含许多用于检测SQL注入情况的文件，例如，我们可以找到**GenericBlind.txt**文件，其中包含与Python模块返回的相同字符串。

在GitHub存储库中，我们看到一些文件取决于我们正在测试的SQL攻击和数据库类型：

![](assets/1107a246-1469-4197-8a84-cdf0fe899375.png)

我们还可以找到其他用于测试MySQL数据库中SQL注入的文件：[https://github.com/fuzzdb-project/fuzzdb/blob/master/attack/sql-injection/detect/MySQL.txt](https://github.com/fuzzdb-project/fuzzdb/blob/master/attack/sql-injection/detect/MySQL.txt)。

在`Mysql.txt`文件中，我们可以看到所有可用的攻击向量，以发现SQL注入漏洞：

![](assets/f3866661-f9d6-4654-b485-30ff8e20c8af.png)

我们可以使用先前的文件来检测特定站点中的SQL注入漏洞：testphp.vulnweb.com。

您可以在`pywebfuzz_folder`内的`demofuzz_sql.py`文件中找到以下代码：

```py
from pywebfuzz import fuzzdb
import requests

mysql_attacks= fuzzdb.attack_payloads.sql_injection.detect.MySQL

domain = "http://testphp.vulnweb.com/listproducts.php?cat="

for attack in mysql_attacks:
    print "Testing... "+ domain + attack
    response = requests.get(domain + attack)
    if "mysql" in response.text.lower(): 
        print("Injectable MySQL detected")
        print("Attack string: "+attack)
```

先前脚本的执行显示了输出：

![](assets/5035d6a0-6ff4-41a0-ad90-2cc3ada3e8bf.png)

以下示例将创建一个包含来自fuzzdb的所有值的Python列表，用于LDAP注入：

```py
from pywebfuzz import fuzzdb ldap_values=fuzzdb.attack_payloads.ldap.ldap_injection
```

现在`ldap_values`变量将是一个包含来自fuzzdb的`ldap_injection`文件的Python字典。然后，您可以使用您的测试迭代此变量的顶部。

我们可以在fuzzbd项目中找到ldap文件夹：[https://github.com/fuzzdb-project/fuzzdb/tree/master/attack/ldap](https://github.com/fuzzdb-project/fuzzdb/tree/master/attack/ldap)。

# 总结

本章的目标之一是了解允许我们提取服务器以公开方式暴露的信息的模块。使用我们讨论过的工具，我们可以获得足够的信息，这些信息可能对我们的后续渗透测试或审计过程有用。

在下一个[章节](321a63e9-bf32-449a-9673-4991ab97234f.xhtml)中，我们将探讨与FTP、SSH和SNMP服务器交互的python编程包。

# 问题

1.  我们需要什么来访问Shodan开发者API？

1.  应该在shodan API中调用哪个方法以获取有关给定主机的信息，该方法返回什么数据结构？

1.  哪个模块可以用来获取服务器的横幅？

1.  应该调用哪个方法并传递什么参数来获取`DNSPython`模块中的IPv6地址记录？

1.  应该调用哪个方法并传递什么参数以获取`DNSPython`模块中邮件服务器的记录？

1.  使用`DNSPython`模块应调用哪个方法以及应传递哪些参数以获取名称服务器的记录？

1.  哪个项目包含文件和文件夹，其中包含在各种网页应用程序的渗透测试中收集的已知攻击模式？

1.  应使用哪个模块来查找可能易受攻击的服务器上的登录页面？

1.  `FuzzDB`项目模块允许我们获取字符串以检测SQL注入类型的漏洞是哪个？

1.  DNS服务器用于解析邮件服务器名称的请求的端口是多少？

# 进一步阅读

在这些链接中，您将找到有关上述工具的更多信息以及一些被评论模块的官方Python文档：

[https://developer.shodan.io/api](https://developer.shodan.io/api)

[http://www.dnspython.org](http://www.dnspython.org)

您可以使用python `dnslib`模块创建自己的DNS服务器：[https://pypi.org/project/dnslib/](https://pypi.org/project/dnslib/)

[https://github.com/fuzzdb-project/fuzzdb](https://github.com/fuzzdb-project/fuzzdb).

在Python生态系统中，我们可以找到其他模糊器，例如**wfuzz**。

Wfuzz是一个Web应用程序安全模糊测试工具，您可以从命令行或使用Python库进行编程：[https://github.com/xmendez/wfuzz](https://github.com/xmendez/wfuzz)。

官方文档可在[http://wfuzz.readthedocs.io](http://wfuzz.readthedocs.io/)找到。

使用`python Shodan`模块的项目示例：

+   [https://www.programcreek.com/python/example/107467/shodan.Shodan](https://www.programcreek.com/python/example/107467/shodan.Shodan)

+   [https://github.com/NullArray/Shogun](https://github.com/NullArray/Shogun)

+   [https://github.com/RussianOtter/networking/blob/master/8oScanner.py](https://github.com/RussianOtter/networking/blob/master/8oScanner.py)

+   [https://github.com/Va5c0/Shodan_cmd](https://github.com/Va5c0/Shodan_cmd)

+   [https://github.com/sjorsng/osint-combinerhttps://github.com/carnal0wnage/pentesty_scripts](https://github.com/sjorsng/osint-combinerhttps://github.com/carnal0wnage/pentesty_scripts)

+   [https://github.com/ffmancera/pentesting-multitool](https://github.com/ffmancera/pentesting-multitool)

+   [https://github.com/ninj4c0d3r/ShodanCli](https://github.com/ninj4c0d3r/ShodanCli)

如果我们有兴趣在没有暴力破解过程的情况下查找网页目录，我们可以使用名为`dirhunt`的工具，基本上是一个用于搜索和分析网站中目录的网络爬虫。

[https://github.com/Nekmo/dirhunt](https://github.com/Nekmo/dirhunt)

您可以使用命令`**pip install dirhunt**`来安装它

这个工具支持Python 2.7版本和3.x版本，但建议使用Python 3.x
