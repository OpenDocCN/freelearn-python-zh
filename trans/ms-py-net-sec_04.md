# HTTP 编程

本章将向您介绍 HTTP 协议，并介绍如何使用 Python 检索和操作 Web 内容。我们还将回顾`urllib`标准库和`requests`包。`urllib2`是用于获取 URL 的 Python 模块。它提供了一个非常简单的接口，以`urlopen`函数的形式。如果我们想要向 API 端点发出请求以简化 HTTP 工作流程，请求包是一个非常有用的工具。

本章将涵盖以下主题：

+   理解 HTTP 协议和在 Python 中构建 HTTP 客户端

+   理解`urllib`包以查询 REST API

+   理解`requests`包以查询 REST API

+   理解不同的身份验证机制以及它们在 Python 中的实现方式

# 技术要求

本章的示例和源代码可在 GitHub 存储库的`第四章`文件夹中找到：[`github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security.`](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security.)

您需要在本地计算机上安装 Python 发行版，并对 HTTP 协议有一些基本的了解。

# HTTP 协议和在 Python 中构建 HTTP 客户端

在本节中，我们将介绍 HTTP 协议以及如何使用 httplib 构建 HTTP 客户端。HTTP 是一个应用层协议，基本上由两个元素组成：客户端发出的请求，该请求从服务器请求由 URL 指定的特定资源，以及服务器发送的响应，提供客户端请求的资源。

# HTTP 协议介绍

HTTP 协议是一种无状态的超文本数据传输协议，不存储客户端和服务器之间交换的信息。该协议定义了客户端、代理和服务器必须遵循的规则以交换信息。

作为存储与 HTTP 事务相关信息的无状态协议，有必要采用其他技术，如 cookie（存储在客户端上的值）或会话（用于在服务器端临时存储有关一个或多个 HTTP 事务的信息的临时内存空间）。

服务器返回一个 HTTP 代码，指示客户端请求的操作结果；此外，头部可以在请求中使用，以在请求和响应中包含额外信息。

HTTP 协议在最低级别使用套接字来建立客户端和服务器之间的连接。在 Python 中，我们有可能使用一个更高级别的模块，它将我们从低级别套接字的操作中抽象出来。

# 使用 httplib 构建 HTTP 客户端

Python 提供了一系列模块来创建 HTTP 客户端。Python 提供的标准库中的模块有`httplib`、`urllib`和`urllib2`。这些模块在所有模块中具有不同的功能，但它们对于大多数 Web 测试都是有用的。我们还可以找到提供一些改进的`httplib`模块和请求的包。

该模块定义了一个实现`HTTPConnection`类的类。

该类接受主机和端口作为参数。主机是必需的，端口是可选的。该类的实例表示与 HTTP 服务器的交易。必须通过传递服务器标识符和可选的端口号来实例化它。如果未指定端口号，则如果服务器标识字符串具有主机：端口的形式，则提取端口号，否则使用默认的 HTTP 端口（80）。

您可以在`request_httplib.py`文件中找到以下代码：

```py
import httplib

connection = httplib.HTTPConnection("www.packtpub.com")
connection.request("GET", "/networking-and-servers/mastering-python-networking-and-security")
response = connection.getresponse()
print response
print response.status, response.reason
data = response.read()
print data
```

# 使用 urllib2 构建 HTTP 客户端

在本节中，我们将学习如何使用`urllib2`以及如何使用该模块构建 HTTP 客户端。

# 介绍 urllib2

`urllib2`可以使用各种协议（如 HTTP、HTTPS、FTP 或 Gopher）从 URL 读取数据。该模块提供了`urlopen`函数，用于创建类似文件的对象，可以从 URL 读取数据。该对象具有诸如`read()`、`readline()`、`readlines()`和`close()`等方法，其工作方式与文件对象完全相同，尽管实际上我们正在使用一个抽象我们免于使用底层套接字的包装器。

`read`方法，正如您记得的那样，用于读取完整的“文件”或作为参数指定的字节数，readline 用于读取一行，readlines 用于读取所有行并返回一个包含它们的列表。

我们还有一些`geturl`方法，用于获取我们正在读取的 URL（这对于检查是否有重定向很有用），以及返回一个带有服务器响应头的对象的 info（也可以通过 headers 属性访问）。

在下一个示例中，我们使用`urlopen()`打开一个网页。当我们将 URL 传递给`urlopen()`方法时，它将返回一个对象，我们可以使用`read()`属性以字符串格式获取该对象的数据。

您可以在`urllib2_basic.py`文件中找到以下代码：

```py
import urllib2
try:
    response = urllib2.urlopen("http://www.python.org")
    print response.read()
    response.close()
except HTTPError, e:
    print e.code
except URLError, e:
    print e.reason
```

使用`urllib2`模块时，我们还需要处理错误和异常类型`URLError`。如果我们使用 HTTP，还可以在`URLError`的子类`HTTPError`中找到错误，当服务器返回 HTTP 错误代码时会抛出这些错误，比如当资源未找到时返回 404 错误。

`urlopen`函数有一个可选的数据参数，用于使用 POST 发送信息到 HTTP 地址（参数在请求本身中发送），例如响应表单。该参数是一个正确编码的字符串，遵循 URL 中使用的格式。

# 响应对象

让我们详细探讨响应对象。我们可以在前面的示例中看到`urlopen()`返回`http.client.HTTPResponse`类的实例。响应对象返回有关请求的资源数据以及响应的属性和元数据。

以下代码使用 urllib2 进行简单的请求：

```py
>>> response = urllib2.urlopen('http://www.python.org')
>>> response.read()
b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"
"http://www.w3.org/TR/html4/strict.dtd">\n<html
>>> response.read(100)
```

`read()`方法允许我们读取请求的资源数据并返回指定的字节数。

# 状态码

我们可以使用其**status**属性读取响应的状态码。200 的值是一个告诉我们请求 OK 的 HTTP 状态码：

```py
>>> response.status
200
```

状态码分为以下几组：

+   **100:** 信息

+   **200:** 成功

+   **300:** 重定向

+   **400:** 客户端错误

+   **500:** 服务器错误

# 使用 urllib2 检查 HTTP 头

HTTP 请求由两个主要部分组成：头部和主体。头部是包含有关响应的特定元数据的信息行，告诉客户端如何解释它。使用此模块，我们可以检查头部是否可以提供有关 Web 服务器的信息。

`http_response.headers`语句提供了 Web 服务器的头部。在访问此属性之前，我们需要检查响应代码是否等于`200`。

您可以在`urllib_headers_basic.py`文件中找到以下代码：

```py
import urllib2
url = raw_input("Enter the URL ")
http_response = urllib2.urlopen(url)
print 'Status Code: '+ str(http_response.code)
if http_response.code == 200:
    print http_response.headers
```

在下面的截图中，我们可以看到脚本在 python.org 域上执行：

![](img/3bff218f-4c01-4cae-8d3d-815f784da3ca.png)

此外，您还可以获取头部的详细信息：

![](img/8da41655-5f6d-4a6f-bdda-0625c41acb8d.png)

检索响应头的另一种方法是使用响应对象的`info()`方法，它将返回一个字典：

![](img/89591d31-8be7-41e9-8a6b-043834987c4a.png)

我们还可以使用`**keys()**`方法获取所有响应头键：

```py
>>> print response_headers.keys()
['content-length', 'via', 'x-cache', 'accept-ranges', 'x-timer', 'vary', 'strict-transport-security', 'server', 'age', 'connection', 'x-xss-protection', 'x-cache-hits', 'x-served-by', 'date', 'x-frame-options', 'content-type', 'x-clacks-overhead']
```

# 使用 urllib2 的 Request 类

`urllib2`的`urlopen`函数还可以将 Request 对象作为参数，而不是 URL 和要发送的数据。Request 类定义了封装与请求相关的所有信息的对象。通过这个对象，我们可以进行更复杂的请求，添加我们自己的头部，比如 User-Agent。

Request 对象的最简构造函数只接受一个字符串作为参数，指示要连接的 URL，因此将此对象作为 urlopen 的参数将等同于直接使用 URL 字符串。

但是，Request 构造函数还有一个可选参数，用于通过 POST 发送数据的数据字符串和标头字典。

# 使用 urllib2 自定义请求

我们可以自定义请求以检索网站的特定版本。为此任务，我们可以使用 Accept-Language 标头，告诉服务器我们首选的资源语言。

在本节中，我们将看到如何使用 User-Agent 标头添加我们自己的标头。User-Agent 是一个用于识别我们用于连接到该 URL 的浏览器和操作系统的标头。默认情况下，urllib2 被标识为“Python-urllib / 2.5”；如果我们想要将自己标识为 Chrome 浏览器，我们可以重新定义标头参数。

在这个例子中，我们使用 Request 类创建相同的 GET 请求，通过将自定义的 HTTP User-Agent 标头作为参数传递：

您可以在`urllib_requests_headers.py`文件中找到以下代码：

```py
import urllib2
url = "http://www.python.org"
headers= {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.117 Safari/537.36'}
request = urllib2.Request(url,headers=headers)
response = urllib2.urlopen(request)
# Here we check response headers
if response.code == 200:
    print(response.headers)
```

使用`urllib`模块的 Request 类，可以创建自定义标头，为此需要在标头参数中定义一个带有键和值格式的标头字典。在上一个例子中，我们设置了代理标头配置，并将其分配为 Chrome 值，并将标头作为字典提供给 Request 构造函数。

# 使用 urllib2 从 URL 获取电子邮件

在这个例子中，我们可以看到如何使用 urllib2 和正则表达式提取电子邮件。

您可以在`get_emails_from_url.py`文件中找到以下代码：

```py
import urllib2
import re
#enter url
web =  raw_input("Enter url: ")
#https://www.packtpub.com/books/info/packt/terms-and-conditions
#get response form url
response = urllib2.Request('http://'+web)
#get content page from response
content = urllib2.urlopen(response).read()
#regular expression
pattern = re.compile("[-a-zA-Z0-9._]+@[-a-zA-Z0-9_]+.[a-zA-Z0-9_.]+")
#get mails from regular expression
mails = re.findall(pattern,content)
print(mails)

```

在这个屏幕截图中，我们可以看到 packtpub.com 域的脚本正在执行：

![](img/44ec2eea-dcbe-4720-bd07-2d0de1bdd9dd.png)

# 使用 urllib2 从 URL 获取链接

在这个脚本中，我们可以看到如何使用`urllib2`和`HTMLParser`提取链接。`HTMLParser`是一个允许我们解析 HTML 格式的文本文件的模块。

您可以在[`docs.python.org/2/library/htmlparser.html`](https://docs.python.org/2/library/htmlparser.html)获取更多信息。

您可以在`get_links_from_url.py`文件中找到以下代码：

```py
#!/usr/bin/python
import urllib2
from HTMLParser import HTMLParser
class myParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if (tag == "a"):
            for a in attrs:
                if (a[0] == 'href'):
                    link = a[1]
                    if (link.find('http') >= 0):
                        print(link)
                        newParse = myParser()
                        newParse.feed(link)

web =  raw_input("Enter url: ")
url = "http://"+web
request = urllib2.Request(url)
handle = urllib2.urlopen(request)
parser = myParser()
parser.feed(handle.read().decode('utf-8'))
```

在以下截图中，我们可以看到 python.org 域的脚本正在执行：

![](img/276cc13e-834b-4a2a-bedf-942e292279d9.png)

# 使用 requests 构建 HTTP 客户端

能够与基于 HTTP 的 RESTful API 进行交互是任何编程语言项目中越来越常见的任务。在 Python 中，我们还可以使用`Requests`模块以简单的方式与 REST API 进行交互。在本节中，我们将回顾使用`Python Requests`包与基于 HTTP 的 API 进行交互的不同方式。

# 请求简介

在 Python 生态系统中进行 HTTP 请求的最佳选择之一是第三方请求库。您可以使用`pip`命令轻松在系统中安装 requests 库：

```py
pip install requests
```

该模块在 PyPi 存储库中可用，名称为`requests`包。它可以通过 Pip 安装，也可以从[`docs.python-requests.org`](http://docs.python-requests.org)下载，该网站提供了文档。

要在我们的脚本中测试该库，只需像其他模块一样导入它。基本上，request 是`urllib2`的一个包装器，以及其他 Python 模块，为我们提供了与 REST 结构的简单方法，因为我们有“post”，“get”，“put”，“patch”，“delete”，“head”和“options”方法，这些都是与 RESTful API 通信所需的方法。

这个模块有一个非常简单的实现形式，例如，使用 requests 进行`GET`查询将是：

```py
>>> import requests
>>> response = requests.get('http://www.python.org')
```

正如我们在这里看到的，requests.get 方法返回一个“response”对象；在这个对象中，您将找到与我们请求的响应对应的所有信息。

这些是响应对象的主要属性：

+   **response.status_code**：这是服务器返回的 HTTP 代码。

+   **response.content**：在这里我们将找到服务器响应的内容。

+   **response.json()**：如果答案是 JSON，这个方法会序列化字符串并返回一个带有相应 JSON 结构的字典结构。如果每个响应都没有收到 JSON，该方法会触发一个异常。

在这个脚本中，我们还可以通过 python.org 域中的响应对象查看请求属性。

您可以在**`requests_headers.py`**文件中找到以下代码：

```py
import requests, json
print("Requests Library tests.")
response = requests.get("http://www.python.org")
print(response.json)
print("Status code: "+str(response.status_code))
print("Headers response: ")
for header, value in response.headers.items():
    print(header, '-->', value)

print("Headers request : ")
for header, value in response.request.headers.items():
    print(header, '-->', value)

```

在下面的屏幕截图中，我们可以看到 python.org 域的脚本正在执行。

在执行的最后一行，我们可以看到**User-Agent**标头中存在**python-requests**：

![](img/93c9b235-8b16-44cd-b09d-2aeb8299b1ae.png)

以类似的方式，我们只能从对象响应字典中获得`keys()`。

您可以在`requests_headers_keys.py`文件中找到以下代码：

```py
import requests
if __name__ == "__main__":
 response = requests.get("http://www.python.org")
 for header in response.headers.keys():
 print(header + ":" + response.headers[header])
```

# 请求的优势

在`requests`模块的主要优势中，我们可以注意到以下几点：

+   一个专注于创建完全功能的 HTTP 客户端的库。

+   支持 HTTP 协议中定义的所有方法和特性。

+   它是“Pythonic”的，也就是说，它完全是用 Python 编写的，所有操作都是以简单的方式和只有几行代码完成的。

+   诸如与 web 服务集成、HTTP 连接的汇集、在表单中编码 POST 数据以及处理 cookies 等任务。所有这些特性都是使用 Requests 自动处理的。

# 使用 REST API 进行 GET 请求

为了使用这个模块进行请求测试，我们可以使用[`httpbin.org`](http://httpbin.org)服务并尝试这些请求，分别执行每种类型。在所有情况下，执行以获得所需输出的代码将是相同的，唯一变化的将是请求类型和发送到服务器的数据：

![](img/36067bfe-7767-42c3-9d83-1dfe4f7538b4.png)

[`httpbin.org`](http://httpbin.org) [提供了一个服务，让您通过预定义的端点使用 get、post、patch、put 和 delete 方法来测试 REST 请求。](http://httpbin.org)

您可以在`testing_api_rest_get_method.py`文件中找到以下代码：

```py
import requests,json
response = requests.get("http://httpbin.org/get",timeout=5)
# we then print out the http status_code
print("HTTP Status Code: " + str(response.status_code))
print(response.headers)
if response.status_code == 200:
    results = response.json()
    for result in results.items():
        print(resul)

    print("Headers response: ")
    for header, value in response.headers.items():
        print(header, '-->', value)

    print("Headers request : ")
    for header, value in response.request.headers.items():
        print(header, '-->', value)
    print("Server:" + response.headers['server'])
else:
    print("Error code %s" % response.status_code)
```

当您运行上述代码时，您应该看到为请求和响应获取的标头的以下输出：

![](img/a6a89921-2970-40b9-9434-a6c6a7ab02af.png)

# 使用 REST API 进行 POST 请求

与将数据发送到 URL 的 GET 方法不同，POST 方法允许我们将数据发送到请求的正文中。

例如，假设我们有一个用于注册用户的服务，您必须通过数据属性传递 ID 和电子邮件。这些信息将通过数据属性通过字典结构传递。post 方法需要一个额外的字段叫做“data”，我们通过这个字段发送一个包含我们将通过相应方法发送到服务器的所有元素的字典。

在这个例子中，我们将模拟通过 POST 请求发送 HTML 表单，就像浏览器在向网站发送表单时所做的那样。表单数据总是以键值字典格式发送。

POST 方法在[`httpbin.org/post`](http://httpbin.org/post)服务中可用：

![](img/1cc80100-be73-4344-9850-b8124eb1d97e.png)

在下面的代码中，我们定义了一个数据字典，我们正在使用它与 post 方法一起传递请求正文中的数据：

```py
>>> data_dictionary = {"id": "0123456789"}
>>> url = "http://httpbin.org/post"
>>> response = requests.post(url, data=data_dictionary)
```

有些情况下，服务器要求请求包含标头，指示我们正在使用 JSON 格式进行通信；对于这些情况，我们可以添加自己的标头或使用**“headers”**参数修改现有的标头：

```py
>>> data_dictionary = {"id": "0123456789"}
>>> headers = {"Content-Type" : "application/json","Accept":"application/json"}
>>> url = "http://httpbin.org/post"
>>> response = requests.post(url, data=data_dictionary,headers=headers)
```

在这个例子中，除了使用 POST 方法，您还必须将要发送到服务器的数据作为数据属性中的参数传递。在答案中，我们看到 ID 是如何被发送到表单对象中的。

# 进行代理请求

`requests`模块提供的一个有趣功能是可以通过代理或内部网络与外部网络之间的中间机器进行请求。

代理的定义方式如下：

```py
>>> proxy = {"protocol":"ip:port", ...}
```

通过代理进行请求时，使用 get 方法的 proxies 属性：

```py
>>> response = requests.get(url,headers=headers,proxies=proxy)
```

代理参数必须以字典形式传递，即必须创建一个指定协议、IP 地址和代理监听端口的字典类型：

```py
import requests
http_proxy = "http://<ip_address>:<port>"
proxy_dictionary = { "http" : http_proxy}
requests.get("http://example.org", proxies=proxy_dictionary)
```

# 使用 requests 处理异常

请求中的错误与其他模块处理方式不同。以下示例生成了一个 404 错误，表示无法找到请求的资源：

```py
>>> response = requests.get('http://www.google.com/pagenotexists')
>>> response.status_code
404
```

在这种情况下，`requests`模块返回了一个 404 错误。要查看内部生成的**异常**，我们可以使用`raise_for_status()`方法：

```py
>>> response.raise_for_status()
requests.exceptions.HTTPError: 404 Client Error
```

如果向不存在的主机发出请求，并且一旦产生了超时，我们会得到一个`ConnectionError`异常：

```py
>>> r = requests.get('http://url_not_exists')
requests.exceptions.ConnectionError: HTTPConnectionPool(...
```

在这个屏幕截图中，我们可以看到在 Python 空闲中执行之前的命令：

![](img/9f4f5df0-2856-4f83-8f5d-ec1217064cd3.png)

与 urllib 相比，请求库使得在 Python 中使用 HTTP 请求更加容易。除非有使用 urllib 的要求，我总是建议在 Python 项目中使用 Requests。

# Python 中的身份验证机制

HTTP 协议本身支持的身份验证机制是**HTTP 基本**和**HTTP 摘要**。这两种机制都可以通过 Python 的 requests 库来支持。

HTTP 基本身份验证机制基于表单，并使用 Base64 对由“冒号”（用户：密码）分隔的用户组成进行编码。

HTTP 摘要身份验证机制使用 MD5 加密用户、密钥和领域哈希。两种方法之间的主要区别在于基本只进行编码，而不实际加密，而摘要会以 MD5 格式加密用户信息。

# 使用 requests 模块进行身份验证

使用`requests`模块，我们可以连接支持基本和摘要身份验证的服务器。使用基本身份验证，用户和密码的信息以`base64`格式发送，而使用摘要身份验证，用户和密码的信息以`md5`或`sha1`算法的哈希形式发送。

# HTTP 基本身份验证

HTTP 基本是一种简单的机制，允许您在 HTTP 资源上实现基本身份验证。其主要优势在于可以在 Apache Web 服务器中轻松实现，使用标准 Apache 指令和 httpasswd 实用程序。

这种机制的问题在于，使用 Wireshark 嗅探器相对简单地获取用户凭据，因为信息是以明文发送的；对于攻击者来说，解码 Base64 格式的信息就足够了。如果客户端知道资源受到此机制的保护，可以使用 Base64 编码的 Authorization 标头发送登录名和密码。

基本访问身份验证假定客户端将通过用户名和密码进行标识。当浏览器客户端最初使用此系统访问站点时，服务器会以包含“**WWW-Authenticate**”标签的 401 响应进行回复，其中包含“Basic”值和受保护域的名称（例如 WWW-Authenticate：Basic realm =“www.domainProtected.com”）。

浏览器用“Authorization”标签回应服务器，其中包含“Basic”值和登录名、冒号标点符号（“：”）和密码的 Base64 编码连接（例如，Authorization：Basic b3dhc3A6cGFzc3dvcmQ =）。

假设我们有一个受到此类身份验证保护的 URL，在 Python 中使用`requests`模块，如下所示：

```py
import requests
encoded = base64.encodestring(user+":"+passwd)
response = requests.get(protectedURL, auth=(user,passwd))
```

我们可以使用此脚本来测试对受保护资源的访问，使用**基本身份验证。**在此示例中，我们应用了**暴力破解过程**来获取受保护资源上的用户和密码凭据。

您可以在`BasicAuthRequests.py`文件中找到以下代码：

```py
import base64
import requests
users=['administrator', 'admin']
passwords=['administrator','admin']
protectedResource = 'http://localhost/secured_path'
foundPass = False
for user in users:
    if foundPass:
        break
    for passwd in passwords:
        encoded = base64.encodestring(user+':'+passwd)
        response = requests.get(protectedResource, auth=(user,passwd))
        if response.status_code != 401:
            print('User Found!')
            print('User: %s, Pass: %s' %(user,passwd))
            foundPass=True
            break
```

# HTTP 摘要身份验证

HTTP 摘要是用于改进 HTTP 协议中基本身份验证过程的机制。通常使用 MD5 来加密用户信息、密钥和领域，尽管其他算法，如 SHA，也可以在其不同的变体中使用，从而提高安全性。它在 Apache Web 服务器中实现了`mod_auth_digest`模块和`htdigest`实用程序。

客户端必须遵循的过程以发送响应，从而获得对受保护资源的访问是：

+   `Hash1= MD5(“user:realm:password”)`

+   `Hash2 = MD5(“HTTP-Method-URI”)`

+   `response = MD5(Hash1:Nonce:Hash2)`

基于摘要的访问身份验证通过使用单向哈希加密算法（MD5）扩展基本访问身份验证，首先加密认证信息，然后添加唯一的连接值。

客户端浏览器在计算密码响应的哈希格式时使用该值。尽管密码通过使用加密哈希和唯一值的使用来防止重放攻击的威胁，但登录名以明文形式发送。

假设我们有一个受此类型身份验证保护的 URL，在 Python 中将如下所示：

```py
import requests
from requests.auth import HTTPDigestAuth
response = requests.get(protectedURL, auth=HTTPDigestAuth(user,passwd))
```

我们可以使用此脚本来测试对受保护资源的访问**摘要身份验证。**在此示例中，我们应用了**暴力破解过程**来获取受保护资源上的用户和密码凭据。该脚本类似于基本身份验证的先前脚本。主要区别在于我们发送用户名和密码的部分，这些用户名和密码是通过 protectedResource URL 发送的。

您可以在`DigestAuthRequests.py`文件中找到以下代码：

```py
import requests
from requests.auth import HTTPDigestAuth
users=['administrator', 'admin']
passwords=['administrator','admin']
protectedResource = 'http://localhost/secured_path'
foundPass = False
for user in users:
 if foundPass:
     break
 for passwd in passwords:
     res = requests.get(protectedResource)
     if res.status_code == 401:
         resDigest = requests.get(protectedResource, auth=HTTPDigestAuth(user, passwd))
         if resDigest.status_code == 200:
             print('User Found...')
             print('User: '+user+' Pass: '+passwd)
             foundPass = True
```

# 摘要

在本章中，我们研究了`httplib`和`urllib`模块，以及用于构建 HTTP 客户端的请求。如果我们想要从 Python 应用程序消耗 API 端点，`requests`模块是一个非常有用的工具。在最后一节中，我们回顾了主要的身份验证机制以及如何使用`request`模块实现它们。在这一点上，我想强调的是，始终阅读我们使用的所有工具的官方文档非常重要，因为那里可以解决更具体的问题。

在下一章中，我们将探索 Python 中的网络编程包，使用`pcapy`和`scapy`模块来分析网络流量。

# 问题

1.  哪个模块是最容易使用的，因为它旨在简化对 REST API 的请求？

1.  如何通过传递字典类型的数据结构来进行 POST 请求，该请求将被发送到请求的正文中？

1.  通过代理服务器正确进行 POST 请求并同时修改标头信息的方法是什么？

1.  如果我们需要通过代理发送请求，需要构建哪种数据结构？

1.  如果在响应对象中有服务器的响应，我们如何获得服务器返回的 HTTP 请求的代码？

1.  我们可以使用哪个模块来指示我们将使用 PoolManager 类预留的连接数？

1.  请求库的哪个模块提供了执行摘要类型身份验证的可能性？

1.  基本身份验证机制使用哪种编码系统来发送用户名和密码？

1.  通过使用单向哈希加密算法（MD5）来改进基本身份验证过程使用了哪种机制？

1.  哪个标头用于识别我们用于向 URL 发送请求的浏览器和操作系统？

# 进一步阅读

在这些链接中，您将找到有关提到的工具的更多信息，以及一些被注释模块的官方 Python 文档：

+   [`docs.python.org/2/library/httplib.html`](https://docs.python.org/2/library/httplib.html)

+   [`docs.python.org/2/library/urllib2.html`](https://docs.python.org/2/library/urllib2.html)

+   [`urllib3.readthedocs.io/en/latest/`](http://urllib3.readthedocs.io/en/latest/)

+   [`docs.python.org/2/library/htmlparser.html`](https://docs.python.org/2/library/htmlparser.html)

+   [`docs.python-requests.org/en/latest`](http://docs.python-requests.org/en/latest)
