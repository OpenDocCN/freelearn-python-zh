# 第2章。HTTP和网络应用

**超文本传输协议**（**HTTP**）可能是最广泛使用的应用层协议。最初开发是为了让学者分享HTML文档。如今，它被用作互联网上无数应用程序的核心协议，并且是万维网的主要协议。

在本章中，我们将涵盖以下主题：

+   HTTP协议结构

+   使用Python通过HTTP与服务通信

+   下载文件

+   HTTP功能，如压缩和cookies

+   处理错误

+   URL

+   Python标准库`urllib`包

+   Kenneth Reitz的第三方`Requests`包

`urllib`包是Python标准库中用于HTTP任务的推荐包。标准库还有一个名为`http`的低级模块。虽然这提供了对协议几乎所有方面的访问，但它并不是为日常使用而设计的。`urllib`包有一个更简单的接口，并且处理了我们将在本章中涵盖的所有内容。

第三方`Requests`包是`urllib`的一个非常受欢迎的替代品。它具有优雅的界面和强大的功能集，是简化HTTP工作流的绝佳工具。我们将在本章末讨论它如何替代`urllib`使用。

# 请求和响应

HTTP是一个应用层协议，几乎总是在TCP之上使用。HTTP协议被故意定义为使用人类可读的消息格式，但仍然可以用于传输任意字节数据。

一个HTTP交换包括两个元素。客户端发出的**请求**，请求服务器提供由URL指定的特定资源，以及服务器发送的**响应**，提供客户端请求的资源。如果服务器无法提供客户端请求的资源，那么响应将包含有关失败的信息。

这个事件顺序在HTTP中是固定的。所有交互都是由客户端发起的。服务器不会在没有客户端明确要求的情况下向客户端发送任何内容。

这一章将教你如何将Python用作HTTP客户端。我们将学习如何向服务器发出请求，然后解释它们的响应。我们将在[第9章](ch09.html "第9章。网络应用")中讨论编写服务器端应用程序，*网络应用*。

到目前为止，最广泛使用的HTTP版本是1.1，定义在RFC 7230到7235中。HTTP 2是最新版本，正式批准时本书即将出版。版本1.1和2之间的语义和语法大部分保持不变，主要变化在于TCP连接的利用方式。目前，HTTP 2的支持并不广泛，因此本书将专注于版本1.1。如果你想了解更多，HTTP 2在RFC 7540和7541中有记录。

HTTP版本1.0，记录在RFC 1945中，仍然被一些较老的软件使用。版本1.1与1.0向后兼容，`urllib`包和`Requests`都支持HTTP 1.1，所以当我们用Python编写客户端时，不需要担心连接到HTTP 1.0服务器。只是一些更高级的功能不可用。几乎所有现在的服务都使用版本1.1，所以我们不会在这里讨论差异。如果需要更多信息，可以参考堆栈溢出的问题：[http://stackoverflow.com/questions/246859/http-1-0-vs-1-1](http://stackoverflow.com/questions/246859/http-1-0-vs-1-1)。

# 使用urllib进行请求

在讨论RFC下载器时，我们已经看到了一些HTTP交换的例子，[第1章](ch01.html "第1章。网络编程和Python")*网络编程和Python*。`urllib`包被分成几个子模块，用于处理我们在使用HTTP时可能需要执行的不同任务。为了发出请求和接收响应，我们使用`urllib.request`模块。

使用`urllib`从URL检索内容是一个简单的过程。打开你的Python解释器，然后执行以下操作：

```py
**>>> from urllib.request import urlopen**
**>>> response = urlopen('http://www.debian.org')**
**>>> response**
**<http.client.HTTPResponse object at 0x7fa3c53059b0>**
**>>> response.readline()**
**b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n'**

```

我们使用`urllib.request.urlopen()`函数发送请求并接收[http://www.debian.org](http://www.debian.org)上资源的响应，这里是一个HTML页面。然后我们将打印出我们收到的HTML的第一行。

# 响应对象

让我们更仔细地看一下我们的响应对象。从前面的例子中我们可以看到，`urlopen()`返回一个`http.client.HTTPResponse`实例。响应对象使我们能够访问请求资源的数据，以及响应的属性和元数据。要查看我们在上一节中收到的响应的URL，可以这样做：

```py
**>>> response.url**
**'http://www.debian.org'**

```

我们通过类似文件的接口使用`readline()`和`read()`方法获取请求资源的数据。我们在前一节看到了`readline()`方法。这是我们使用`read()`方法的方式：

```py
**>>> response = urlopen('http://www.debian.org')**
**>>> response.read(50)**
**b'g="en">\n<head>\n  <meta http-equiv="Content-Type" c'**

```

`read()`方法从数据中返回指定数量的字节。这里是前50个字节。调用`read()`方法而不带参数将一次性返回所有数据。

类似文件的接口是有限的。一旦数据被读取，就无法使用上述函数之一返回并重新读取它。为了证明这一点，请尝试执行以下操作：

```py
**>>> response = urlopen('http://www.debian.org')**
**>>> response.read()**
**b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n<html lang="en">\n<head>\n  <meta http-equiv**
**...**
**>>> response.read()**
**b''**

```

我们可以看到，当我们第二次调用`read()`函数时，它返回一个空字符串。没有`seek()`或`rewind()`方法，所以我们无法重置位置。因此，最好将`read()`输出捕获在一个变量中。

`readline()`和`read()`函数都返回字节对象，`http`和`urllib`都不会对接收到的数据进行解码为Unicode。在本章的后面，我们将看到如何利用`Requests`库来处理这个问题。

# 状态码

如果我们想知道我们的请求是否发生了意外情况怎么办？或者如果我们想知道我们的响应在读取数据之前是否包含任何数据怎么办？也许我们期望得到一个大的响应，我们想快速查看我们的请求是否成功而不必读取整个响应。

HTTP响应通过**状态码**为我们提供了这样的方式。我们可以通过使用其`status`属性来读取响应的状态码。

```py
**>>> response.status**
**200**

```

状态码是告诉我们请求的情况的整数。`200`代码告诉我们一切都很好。

有许多代码，每个代码传达不同的含义。根据它们的第一个数字，状态码被分为以下几组：

+   100：信息

+   200：成功

+   300：重定向

+   400：客户端错误

+   500：服务器错误

一些常见的代码及其消息如下：

+   `200`：`OK`

+   `404`：`未找到`

+   `500`：`内部服务器错误`

状态码的官方列表由IANA维护，可以在[https://www.iana.org/assignments/http-status-codes](https://www.iana.org/assignments/http-status-codes)找到。我们将在本章中看到各种代码。

# 处理问题

状态码帮助我们查看响应是否成功。200范围内的任何代码表示成功，而400范围或500范围内的代码表示失败。

应该始终检查状态码，以便我们的程序在出现问题时能够做出适当的响应。`urllib`包通过在遇到问题时引发异常来帮助我们检查状态码。

让我们看看如何捕获这些异常并有用地处理它们。为此，请尝试以下命令块：

```py
**>>> import urllib.error**
**>>> from urllib.request import urlopen**
**>>> try:**
**...   urlopen('http://www.ietf.org/rfc/rfc0.txt')**
**... except urllib.error.HTTPError as e:**
**...   print('status', e.code)**
**...   print('reason', e.reason)**
**...   print('url', e.url)**
**...**
**status: 404**
**reason: Not Found**
**url: http://www.ietf.org/rfc/rfc0.txt**

```

在这里，我们请求了不存在的RFC 0。因此服务器返回了404状态代码，`urllib`已经发现并引发了`HTTPError`。

您可以看到`HTTPError`提供了有关请求的有用属性。在前面的示例中，我们使用了`status`、`reason`和`url`属性来获取有关响应的一些信息。

如果网络堆栈中出现问题，那么适当的模块将引发异常。`urllib`包捕获这些异常，然后将它们包装为`URLErrors`。例如，我们可能已经指定了一个不存在的主机或IP地址，如下所示：

```py
**>>> urlopen('http://192.0.2.1/index.html')**
**...**
**urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>**

```

在这种情况下，我们已经从`192.0.2.1`主机请求了`index.html`。`192.0.2.0/24` IP地址范围被保留供文档使用，因此您永远不会遇到使用前述IP地址的主机。因此TCP连接超时，`socket`引发超时异常，`urllib`捕获，重新包装并为我们重新引发。我们可以像在前面的例子中一样捕获这些异常。

# HTTP头部

请求和响应由两个主要部分组成，**头部**和**正文**。当我们在[第1章](ch01.html "第1章。网络编程和Python")中使用TCP RFC下载器时，我们简要地看到了一些HTTP头部，*网络编程和Python*。头部是出现在通过TCP连接发送的原始消息开头的协议特定信息行。正文是消息的其余部分。它与头部之间由一个空行分隔。正文是可选的，其存在取决于请求或响应的类型。以下是一个HTTP请求的示例：

```py
GET / HTTP/1.1
Accept-Encoding: identity
Host: www.debian.com
Connection: close
User-Agent: Python-urllib/3.4
```

第一行称为**请求行**。它由请求**方法**组成，在这种情况下是`GET`，资源的路径，在这里是`/`，以及HTTP版本`1.1`。其余行是请求头。每行由一个头部名称后跟一个冒号和一个头部值组成。前述输出中的请求只包含头部，没有正文。

头部用于几个目的。在请求中，它们可以用于传递额外的数据，如cookies和授权凭据，并询问服务器首选资源格式。

例如，一个重要的头部是`Host`头部。许多Web服务器应用程序提供了在同一台服务器上使用相同的IP地址托管多个网站的能力。为各个网站域名设置了DNS别名，因此它们都指向同一个IP地址。实际上，Web服务器为每个托管的网站提供了多个主机名。IP和TCP（HTTP运行在其上）不能用于告诉服务器客户端想要连接到哪个主机名，因为它们都仅仅在IP地址上操作。HTTP协议允许客户端在HTTP请求中提供主机名，包括`Host`头部。

我们将在下一节中查看一些更多的请求头部。

以下是响应的一个示例：

```py
HTTP/1.1 200 OK
Date: Sun, 07 Sep 2014 19:58:48 GMT
Content-Type: text/html
Content-Length: 4729
Server: Apache
Content-Language: en

<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n...
```

第一行包含协议版本、状态代码和状态消息。随后的行包含头部、一个空行，然后是正文。在响应中，服务器可以使用头部通知客户端有关正文长度、响应正文包含的内容类型以及客户端应存储的cookie数据等信息。

要查看响应对象的头部，请执行以下操作：

```py
**>>> response = urlopen('http://www.debian.org)**
**>>> response.getheaders()**
**[('Date', 'Sun, 07 Sep 2014 19:58:48 GMT'), ('Server', 'Apache'), ('Content-Location', 'index.en.html'), ('Vary', 'negotiate,accept- language,Accept-Encoding')...**

```

`getheaders()`方法以元组列表的形式返回头部（`头部名称`，`头部值`）。HTTP 1.1头部及其含义的完整列表可以在RFC 7231中找到。让我们看看如何在请求和响应中使用一些头部。

# 自定义请求

利用标头提供的功能，我们在发送请求之前向请求添加标头。为了做到这一点，我们不能只是使用`urlopen()`。我们需要按照以下步骤进行：

+   创建一个`Request`对象

+   向请求对象添加标头

+   使用`urlopen()`发送请求对象

我们将学习如何自定义一个请求，以检索Debian主页的瑞典版本。我们将使用`Accept-Language`标头，告诉服务器我们对其返回的资源的首选语言。请注意，并非所有服务器都保存多种语言版本的资源，因此并非所有服务器都会响应`Accept-Language`。

首先，我们创建一个`Request`对象：

```py
**>>> from urllib.request import Request**
**>>> req = Request('http://www.debian.org')**

```

接下来，添加标头：

```py
**>>> req.add_header('Accept-Language', 'sv')**

```

`add_header()`方法接受标头的名称和标头的内容作为参数。`Accept-Language`标头采用两字母的ISO 639-1语言代码。瑞典语的代码是`sv`。

最后，我们使用`urlopen()`提交定制的请求：

```py
**>>> response = urlopen(req)**

```

我们可以通过打印前几行来检查响应是否是瑞典语：

```py
**>>> response.readlines()[:5]**
**[b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n',**
 **b'<html lang="sv">\n',**
 **b'<head>\n',**
 **b'  <meta http-equiv="Content-Type" content="text/html; charset=utf-  8">\n',**
 **b'  <title>Debian -- Det universella operativsystemet </title>\n']**

```

Jetta bra！`Accept-Language`标头已经告知服务器我们对响应内容的首选语言。

要查看请求中存在的标头，请执行以下操作：

```py
**>>> req = Request('http://www.debian.org')**
**>>> req.add_header('Accept-Language', 'sv')**
**>>> req.header_items()**
**[('Accept-language', 'sv')]**

```

当我们在请求上运行`urlopen()`时，`urlopen()`方法会添加一些自己的标头：

```py
**>>> response = urlopen(req)**
**>>> req.header_items()**
**[('Accept-language', 'sv'), ('User-agent': 'Python-urllib/3.4'), ('Host': 'www.debian.org')]**

```

添加标头的一种快捷方式是在创建请求对象的同时添加它们，如下所示：

```py
**>>> headers = {'Accept-Language': 'sv'}**
**>>> req = Request('http://www.debian.org', headers=headers)**
**>>> req.header_items()**
**[('Accept-language', 'sv')]**

```

我们将标头作为`dict`提供给`Request`对象构造函数，作为`headers`关键字参数。通过这种方式，我们可以一次性添加多个标头，通过向`dict`添加更多条目。

让我们看看我们可以用标头做些什么其他事情。

## 内容压缩

`Accept-Encoding`请求标头和`Content-Encoding`响应标头可以一起工作，允许我们临时对响应主体进行编码，以便通过网络传输。这通常用于压缩响应并减少需要传输的数据量。

这个过程遵循以下步骤：

+   客户端发送一个请求，其中在`Accept-Encoding`标头中列出了可接受的编码

+   服务器选择其支持的编码方法

+   服务器使用这种编码方法对主体进行编码

+   服务器发送响应，指定其在`Content-Encoding`标头中使用的编码

+   客户端使用指定的编码方法解码响应主体

让我们讨论如何请求一个文档，并让服务器对响应主体使用`gzip`压缩。首先，让我们构造请求：

```py
**>>> req = Request('http://www.debian.org')**

```

接下来，添加`Accept-Encoding`标头：

```py
**>>> req.add_header('Accept-Encoding', 'gzip')**

```

然后，借助`urlopen()`提交请求：

```py
**>>> response = urlopen(req)**

```

我们可以通过查看响应的`Content-Encoding`标头来检查服务器是否使用了`gzip`压缩：

```py
**>>> response.getheader('Content-Encoding')**
**'gzip'**

```

然后，我们可以使用`gzip`模块对主体数据进行解压：

```py
**>>> import gzip**
**>>> content = gzip.decompress(response.read())**
**>>> content.splitlines()[:5]**
**[b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">',**
 **b'<html lang="en">',**
 **b'<head>',**
 **b'  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">',**
 **b'  <title>Debian -- The Universal Operating System </title>']**

```

编码已在IANA注册。当前列表包括：`gzip`、`compress`、`deflate`和`identity`。前三个是指特定的压缩方法。最后一个允许客户端指定不希望对内容应用任何编码。

让我们看看如果我们使用`identity`编码来请求不进行压缩会发生什么：

```py
**>>> req = Request('http://www.debian.org')**
**>>> req.add_header('Accept-Encoding', 'identity')**
**>>> response = urlopen(req)**
**>>> print(response.getheader('Content-Encoding'))**
**None**

```

当服务器使用`identity`编码类型时，响应中不包括`Content-Encoding`标头。

## 多个值

为了告诉服务器我们可以接受多种编码，我们可以在`Accept-Encoding`标头中添加更多值，并用逗号分隔它们。让我们试试。我们创建我们的`Request`对象：

```py
**>>> req = Request('http://www.debian.org')**

```

然后，我们添加我们的标头，这次我们包括更多的编码：

```py
**>>> encodings = 'deflate, gzip, identity'**
**>>> req.add_header('Accept-Encoding', encodings)**

```

现在，我们提交请求，然后检查响应的编码：

```py
**>>> response = urlopen(req)**
**>>> response.getheader('Content-Encoding')**
**'gzip'**

```

如果需要，可以通过添加`q`值来给特定编码分配相对权重：

```py
**>>> encodings = 'gzip, deflate;q=0.8, identity;q=0.0'**

```

`q`值跟随编码名称，并且由分号分隔。最大的`q`值是`1.0`，如果没有给出`q`值，则默认为`1.0`。因此，前面的行应该被解释为我的首选编码是`gzip`，我的第二个首选是`deflate`，如果没有其他可用的编码，则我的第三个首选是`identity`。

# 内容协商

使用`Accept-Encoding`标头进行内容压缩，使用`Accept-Language`标头进行语言选择是**内容协商**的例子，其中客户端指定其关于所请求资源的格式和内容的首选项。以下标头也可以用于此目的：

+   `Accept`：请求首选文件格式

+   `Accept-Charset`：请求以首选字符集获取资源

内容协商机制还有其他方面，但由于支持不一致并且可能变得相当复杂，我们不会在本章中进行介绍。RFC 7231包含您需要的所有详细信息。如果您发现您的应用程序需要此功能，请查看3.4、5.3、6.4.1和6.5.6等部分。

## 内容类型

HTTP可以用作任何类型文件或数据的传输。服务器可以在响应中使用`Content-Type`头来通知客户端有关它在主体中发送的数据类型。这是HTTP客户端确定如何处理服务器返回的主体数据的主要手段。

要查看内容类型，我们检查响应标头的值，如下所示：

```py
**>>> response = urlopen('http://www.debian.org')**
**>>> response.getheader('Content-Type')**
**'text/html'**

```

此标头中的值取自由IANA维护的列表。这些值通常称为**内容类型**、**互联网媒体类型**或**MIME类型**（**MIME**代表**多用途互联网邮件扩展**，在该规范中首次建立了这种约定）。完整列表可以在[http://www.iana.org/assignments/media-types](http://www.iana.org/assignments/media-types)找到。

对于通过互联网传输的许多数据类型都有注册的媒体类型，一些常见的类型包括：

| 媒体类型 | 描述 |
| --- | --- |
| text/html | HTML文档 |
| text/plain | 纯文本文档 |
| image/jpeg | JPG图像 |
| application/pdf | PDF文档 |
| application/json | JSON数据 |
| application/xhtml+xml | XHTML文档 |

另一个感兴趣的媒体类型是`application/octet-stream`，在实践中用于没有适用的媒体类型的文件。这种情况的一个例子是一个经过pickle处理的Python对象。它还用于服务器不知道格式的文件。为了正确处理具有此媒体类型的响应，我们需要以其他方式发现格式。可能的方法如下：

+   检查已下载资源的文件名扩展名（如果有）。然后可以使用`mimetypes`模块来确定媒体类型（转到[第3章](ch03.html "第3章。APIs in Action")，*APIs in Action*，以查看此示例）。

+   下载数据，然后使用文件类型分析工具。对于图像，可以使用Python标准库的`imghdr`模块，对于其他类型，可以使用第三方的`python-magic`包或`GNU`文件命令。

+   检查我们正在下载的网站，看看文件类型是否已经在任何地方有文档记录。

内容类型值可以包含可选的附加参数，提供有关类型的进一步信息。这通常用于提供数据使用的字符集。例如：

```py
Content-Type: text/html; charset=UTF-8.
```

在这种情况下，我们被告知文档的字符集是UTF-8。参数在分号后面包括，并且它总是采用键/值对的形式。

让我们讨论一个例子，下载Python主页并使用它返回的`Content-Type`值。首先，我们提交我们的请求：

```py
**>>> response = urlopen('http://www.python.org')**

```

然后，我们检查响应的`Content-Type`值，并提取字符集：

```py
**>>> format, params = response.getheader('Content-Type').split(';')**
**>>> params**
**' charset=utf-8'**
**>>> charset = params.split('=')[1]**
**>>> charset**
**'utf-8'**

```

最后，我们通过使用提供的字符集来解码我们的响应内容：

```py
**>>> content = response.read().decode(charset)**

```

请注意，服务器通常要么在`Content-Type`头中不提供`charset`，要么提供错误的`charset`。因此，这个值应该被视为一个建议。这是我们稍后在本章中查看`Requests`库的原因之一。它将自动收集关于解码响应主体应该使用的字符集的所有提示，并为我们做出最佳猜测。

# 用户代理

另一个值得了解的请求头是`User-Agent`头。使用HTTP通信的任何客户端都可以称为**用户代理**。RFC 7231建议用户代理应该在每个请求中使用`User-Agent`头来标识自己。放在那里的内容取决于发出请求的软件，尽管通常包括一个标识程序和版本的字符串，可能还包括操作系统和运行的硬件。例如，我当前版本的Firefox的用户代理如下所示：

```py
Mozilla/5.0 (X11; Linux x86_64; rv:24.0) Gecko/20140722 Firefox/24.0 Iceweasel/24.7.0
```

尽管这里被分成了两行，但它是一个单独的长字符串。正如你可能能够解释的那样，我正在运行Iceweasel（Debian版的Firefox）24版本，运行在64位Linux系统上。用户代理字符串并不是用来识别个别用户的。它们只标识用于发出请求的产品。

我们可以查看`urllib`使用的用户代理。执行以下步骤：

```py
**>>> req = Request('http://www.python.org')**
**>>> urlopen(req)**
**>>> req.get_header('User-agent')**
**'Python-urllib/3.4'**

```

在这里，我们创建了一个请求并使用`urlopen`提交了它，`urlopen`添加了用户代理头到请求中。我们可以使用`get_header()`方法来检查这个头。这个头和它的值包含在`urllib`发出的每个请求中，所以我们向每个服务器发出请求时都可以看到我们正在使用Python 3.4和`urllib`库。

网站管理员可以检查请求的用户代理，然后将这些信息用于各种用途，包括以下内容：

+   为了他们的网站统计分类访问

+   阻止具有特定用户代理字符串的客户端

+   发送给已知问题的用户代理的资源的替代版本，比如在解释某些语言（如CSS）时出现的错误，或者根本不支持某些语言（比如JavaScript）。

最后两个可能会给我们带来问题，因为它们可能会阻止或干扰我们访问我们想要的内容。为了解决这个问题，我们可以尝试设置我们的用户代理，使其模拟一个知名的浏览器。这就是所谓的**欺骗**，如下所示：

```py
**>>> req = Request('http://www.debian.org')**
**>>> req.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64; rv:24.0) Gecko/20140722 Firefox/24.0 Iceweasel/24.7.0')**
**>>> response = urlopen(req)**

```

服务器将会响应，就好像我们的应用程序是一个普通的Firefox客户端。不同浏览器的用户代理字符串可以在网上找到。我还没有找到一个全面的资源，但是通过谷歌搜索浏览器和版本号通常会找到一些信息。或者你可以使用Wireshark来捕获浏览器发出的HTTP请求，并查看捕获的请求的用户代理头。

# Cookies

Cookie是服务器在响应的一部分中以`Set-Cookie`头发送的一小段数据。客户端会将cookie存储在本地，并在以后发送到服务器的任何请求中包含它们。

服务器以各种方式使用cookie。它们可以向其中添加一个唯一的ID，这使它们能够跟踪客户端访问站点的不同区域。它们可以存储一个登录令牌，这将自动登录客户端，即使客户端离开站点然后以后再次访问。它们也可以用于存储客户端的用户偏好或个性化信息的片段，等等。

Cookie是必需的，因为服务器没有其他方式在请求之间跟踪客户端。HTTP被称为**无状态**协议。它不包含一个明确的机制，让服务器确切地知道两个请求是否来自同一个客户端。如果没有cookie允许服务器向请求添加一些唯一标识信息，像购物车（这是cookie开发的最初问题）这样的东西将变得不可能构建，因为服务器将无法确定哪个篮子对应哪个请求。

我们可能需要在Python中处理cookie，因为没有它们，一些网站的行为不如预期。在使用Python时，我们可能还想访问需要登录的站点的部分，登录会话通常通过cookie来维护。

## Cookie处理

我们将讨论如何使用`urllib`处理cookie。首先，我们需要创建一个存储服务器将发送给我们的cookie的地方：

```py
**>>> from http.cookiejar import CookieJar**
**>>> cookie_jar = CookieJar()**

```

接下来，我们构建一个名为`urllib` `opener` **的东西。这将自动从我们收到的响应中提取cookie，然后将它们存储在我们的cookie jar中：

```py
**>>> from urllib.request import build_opener, HTTPCookieProcessor**
**>>> opener = build_opener(HTTPCookieProcessor(cookie_jar))**

```

然后，我们可以使用我们的opener来发出HTTP请求：

```py
**>>> opener.open('http://www.github.com')**

```

最后，我们可以检查服务器是否发送了一些cookie：

```py
**>>> len(cookie_jar)**
**2**

```

每当我们使用`opener`发出进一步的请求时，`HTTPCookieProcessor`功能将检查我们的`cookie_jar`，看它是否包含该站点的任何cookie，然后自动将它们添加到我们的请求中。它还将接收到的任何进一步的cookie添加到cookie jar中。

`http.cookiejar`模块还包含一个`FileCookieJar`类，它的工作方式与`CookieJar`相同，但它提供了一个额外的函数，用于轻松地将cookie保存到文件中。这允许在Python会话之间持久保存cookie。

## 了解您的cookie

值得更详细地查看cookie的属性。让我们来检查GitHub在前一节中发送给我们的cookie。

为此，我们需要从cookie jar中取出cookie。`CookieJar`模块不允许我们直接访问它们，但它支持迭代器协议。因此，一个快速获取它们的方法是从中创建一个`list`：

```py
**>>> cookies = list(cookie_jar)**
**>>> cookies**
**[Cookie(version=0, name='logged_in', value='no', ...),**
 **Cookie(version=0, name='_gh_sess', value='eyJzZxNzaW9uX...', ...)**
**]**

```

您可以看到我们有两个`Cookie`对象。现在，让我们从第一个对象中提取一些信息：

```py
**>>> cookies[0].name**
**'logged_in'**
**>>> cookies[0].value**
**'no'**

```

cookie的名称允许服务器快速引用它。这个cookie显然是GitHub用来查明我们是否已经登录的机制的一部分。接下来，让我们做以下事情：

```py
**>>> cookies[0].domain**
**'.github.com'**
**>>> cookies[0].path**
**'/'**

```

域和路径是此cookie有效的区域，因此我们的`urllib` opener将在发送到[www.github.com](http://www.github.com)及其子域的任何请求中包含此cookie，其中路径位于根目录下方的任何位置。

现在，让我们来看一下cookie的生命周期：

```py
**>>> cookies[0].expires**
**2060882017**

```

这是一个Unix时间戳；我们可以将其转换为`datetime`：

```py
**>>> import datetime**
**>>> datetime.datetime.fromtimestamp(cookies[0].expires)**
**datetime.datetime(2035, 4, 22, 20, 13, 37)**

```

因此，我们的cookie将在2035年4月22日到期。到期日期是服务器希望客户端保留cookie的时间。一旦到期日期过去，客户端可以丢弃cookie，并且服务器将在下一个请求中发送一个新的cookie。当然，没有什么能阻止客户端立即丢弃cookie，尽管在一些站点上，这可能会破坏依赖cookie的功能。

让我们讨论两个常见的cookie标志：

```py
**>>> print(cookies[0].get_nonstandard_attr('HttpOnly'))**
**None**

```

存储在客户端上的cookie可以通过多种方式访问：

+   由客户端作为HTTP请求和响应序列的一部分

+   由客户端中运行的脚本，比如JavaScript

+   由客户端中运行的其他进程，比如Flash

`HttpOnly`标志表示客户端只有在HTTP请求或响应的一部分时才允许访问cookie。其他方法应该被拒绝访问。这将保护客户端免受跨站脚本攻击的影响（有关这些攻击的更多信息，请参见[第9章](ch09.html "第9章。Web应用程序")*Web应用程序*）。这是一个重要的安全功能，当服务器设置它时，我们的应用程序应该相应地行事。

还有一个`secure`标志：

```py
**>>> cookies[0].secure**
**True**

```

如果值为true，则`Secure`标志表示cookie只能通过安全连接发送，例如HTTPS。同样，如果已设置该标志，我们应该遵守这一点，这样当我们的应用程序发送包含此cookie的请求时，它只会将它们发送到HTTPS URL。

您可能已经发现了一个不一致之处。我们的URL已经请求了一个HTTP响应，然而服务器却发送了一个cookie给我们，要求它只能在安全连接上发送。网站设计者肯定没有忽视这样的安全漏洞吧？请放心，他们没有。实际上，响应是通过HTTPS发送的。但是，这是如何发生的呢？答案就在于重定向。

# 重定向

有时服务器会移动它们的内容。它们还会使一些内容过时，并在不同的位置放上新的东西。有时他们希望我们使用更安全的HTTPS协议而不是HTTP。在所有这些情况下，他们可能会得到请求旧URL的流量，并且在所有这些情况下，他们可能更愿意能够自动将访问者发送到新的URL。

HTTP状态码的300系列是为此目的而设计的。这些代码指示客户端需要采取进一步的行动才能完成请求。最常见的操作是在不同的URL上重试请求。这被称为**重定向**。

我们将学习在使用`urllib`时如何工作。让我们发出一个请求：

```py
**>>> req = Request('http://www.gmail.com')**
**>>> response = urlopen(req)**

```

很简单，但现在，看一下响应的URL：

```py
**>>> response.url**
**'https://accounts.google.com/ServiceLogin?service=mail&passive=true&r m=false...'**

```

这不是我们请求的URL！如果我们在浏览器中打开这个新的URL，我们会发现这实际上是Google的登录页面（如果您已经有缓存的Google登录会话，则可能需要清除浏览器的cookie才能看到这一点）。Google将我们从[http://www.gmail.com](http://www.gmail.com)重定向到其登录页面，`urllib`自动跟随了重定向。此外，我们可能已经被重定向了多次。看一下我们请求对象的`redirect_dict`属性：

```py
**>>> req.redirect_dict**
**{'https://accounts.google.com/ServiceLogin?service=...': 1, 'https://mail.google.com/mail/': 1}**

```

`urllib`包将我们通过的每个URL添加到这个`dict`中。我们可以看到我们实际上被重定向了两次，首先是到[https://mail.google.com](https://mail.google.com)，然后是到登录页面。

当我们发送第一个请求时，服务器会发送一个带有重定向状态代码的响应，其中之一是301、302、303或307。所有这些都表示重定向。此响应包括一个`Location`头，其中包含新的URL。`urllib`包将向该URL提交一个新的请求，在上述情况下，它将收到另一个重定向，这将导致它到达Google登录页面。

由于`urllib`为我们跟随重定向，它们通常不会影响我们，但值得知道的是，`urllib`返回的响应可能是与我们请求的URL不同的URL。此外，如果我们对单个请求进行了太多次重定向（对于`urllib`超过10次），那么`urllib`将放弃并引发`urllib.error.HTTPError`异常。

# URL

统一资源定位符，或者**URL**是Web操作的基础，它们已经在RFC 3986中正式描述。URL代表主机上的资源。URL如何映射到远程系统上的资源完全取决于系统管理员的决定。URL可以指向服务器上的文件，或者在收到请求时资源可能是动态生成的。只要我们请求时URL有效，URL映射到什么并不重要。

URL由几个部分组成。Python使用`urllib.parse`模块来处理URL。让我们使用Python将URL分解为其组成部分：

```py
**>>> from urllib.parse import urlparse**
**>>> result = urlparse('http://www.python.org/dev/peps')**
**>>> result**
**ParseResult(scheme='http', netloc='www.python.org', path='/dev/peps', params='', query='', fragment='')**

```

`urllib.parse.urlparse()`函数解释了我们的URL，并识别`http`作为**方案**，[https://www.python.org/](https://www.python.org/)作为**网络位置**，`/dev/peps`作为**路径**。我们可以将这些组件作为`ParseResult`的属性来访问：

```py
**>>> result.netloc**
**'www.python.org'**
**>>> result.path**
**'/dev/peps'**

```

对于网上几乎所有的资源，我们将使用`http`或`https`方案。在这些方案中，要定位特定的资源，我们需要知道它所在的主机和我们应该连接到的TCP端口（这些组合在一起是`netloc`组件），我们还需要知道主机上资源的路径（`path`组件）。

可以通过将端口号附加到主机后来在URL中明确指定端口号。它们与主机之间用冒号分隔。让我们看看当我们尝试使用`urlparse`时会发生什么。

```py
**>>> urlparse('http://www.python.org:8080/')**
**ParseResult(scheme='http', netloc='www.python.org:8080', path='/', params='', query='', fragment='')**

```

`urlparse`方法只是将其解释为netloc的一部分。这没问题，因为这是`urllib.request.urlopen()`等处理程序期望它格式化的方式。

如果我们不提供端口（通常情况下），那么`http`将使用默认端口80，`https`将使用默认端口443。这通常是我们想要的，因为这些是HTTP和HTTPS协议的标准端口。

## 路径和相对URL

URL中的路径是指主机和端口之后的任何内容。路径总是以斜杠(`/`)开头，当只有一个斜杠时，它被称为**根**。我们可以通过以下操作来验证这一点：

```py
**>>> urlparse('http://www.python.org/')**
**ParseResult(scheme='http', netloc='www.python.org', path='/', params='', query='', fragment='')**

```

如果请求中没有提供路径，默认情况下`urllib`将发送一个请求以获取根目录。

当URL中包含方案和主机时（如前面的例子），该URL被称为**绝对URL**。相反，也可能有**相对URL**，它只包含路径组件，如下所示：

```py
**>>> urlparse('../images/tux.png')**
**ParseResult(scheme='', netloc='', path='../images/tux.png', params='', query='', fragment='')**

```

我们可以看到`ParseResult`只包含一个`path`。如果我们想要使用相对URL请求资源，那么我们需要提供缺失的方案、主机和基本路径。

通常，我们在已从URL检索到的资源中遇到相对URL。因此，我们可以使用该资源的URL来填充缺失的组件。让我们看一个例子。

假设我们已经检索到了[http://www.debian.org](http://www.debian.org)的URL，并且在网页源代码中找到了“关于”页面的相对URL。我们发现它是`intro/about`的相对URL。

我们可以通过使用原始页面的URL和`urllib.parse.urljoin()`函数来创建绝对URL。让我们看看我们可以如何做到这一点：

```py
**>>> from urllib.parse import urljoin**
**>>> urljoin('http://www.debian.org', 'intro/about')**
**'http://www.debian.org/intro/about'**

```

通过向`urljoin`提供基本URL和相对URL，我们创建了一个新的绝对URL。

在这里，注意`urljoin`是如何在主机和路径之间填充斜杠的。只有当基本URL没有路径时，`urljoin`才会为我们填充斜杠，就像前面的例子中所示的那样。让我们看看如果基本URL有路径会发生什么。

```py
**>>> urljoin('http://www.debian.org/intro/', 'about')**
**'http://www.debian.org/intro/about'**
**>>> urljoin('http://www.debian.org/intro', 'about')**
**'http://www.debian.org/about'**

```

这将给我们带来不同的结果。请注意，如果基本URL以斜杠结尾，`urljoin`会将其附加到路径，但如果基本URL不以斜杠结尾，它将替换基本URL中的最后一个路径元素。

我们可以通过在路径前加上斜杠来强制路径替换基本URL的所有元素。按照以下步骤进行：

```py
**>>> urljoin('http://www.debian.org/intro/about', '/News')**
**'http://www.debian.org/News'**

```

如何导航到父目录？让我们尝试标准的点语法，如下所示：

```py
**>>> urljoin('http://www.debian.org/intro/about/', '../News')**
**'http://www.debian.org/intro/News'**
**>>> urljoin('http://www.debian.org/intro/about/', '../../News')**
**'http://www.debian.org/News'**
**>>> urljoin('http://www.debian.org/intro/about', '../News')**
**'http://www.debian.org/News'**

```

它按我们的预期工作。注意基本URL是否有尾随斜杠的区别。

最后，如果“相对”URL实际上是绝对URL呢：

```py
**>>> urljoin('http://www.debian.org/about', 'http://www.python.org')**
**'http://www.python.org'**

```

相对URL完全替换了基本URL。这很方便，因为这意味着我们在使用`urljoin`时不需要担心URL是相对的还是绝对的。

## 查询字符串

RFC 3986定义了URL的另一个属性。它们可以包含在路径之后以键/值对形式出现的附加参数。它们通过问号与路径分隔，如下所示：

[http://docs.python.org/3/search.html?q=urlparse&area=default](http://docs.python.org/3/search.html?q=urlparse&area=default)

这一系列参数称为查询字符串。多个参数由`&`分隔。让我们看看`urlparse`如何处理它：

```py
**>>> urlparse('http://docs.python.org/3/search.html? q=urlparse&area=default')**
**ParseResult(scheme='http', netloc='docs.python.org', path='/3/search.html', params='', query='q=urlparse&area=default', fragment='')**

```

因此，`urlparse`将查询字符串识别为`query`组件。

查询字符串用于向我们希望检索的资源提供参数，并且通常以某种方式自定义资源。在上述示例中，我们的查询字符串告诉Python文档搜索页面，我们要搜索术语`urlparse`。

`urllib.parse`模块有一个函数，可以帮助我们将`urlparse`返回的`query`组件转换为更有用的内容：

```py
**>>> from urllib.parse import parse_qs**
**>>> result = urlparse ('http://docs.python.org/3/search.html?q=urlparse&area=default')**
**>>> parse_qs(result.query)**
**{'area': ['default'], 'q': ['urlparse']}**

```

`parse_qs()` 方法读取查询字符串，然后将其转换为字典。看看字典值实际上是以列表的形式存在的？这是因为参数可以在查询字符串中出现多次。尝试使用重复参数：

```py
**>>> result = urlparse ('http://docs.python.org/3/search.html?q=urlparse&q=urljoin')**
**>>> parse_qs(result.query)**
**{'q': ['urlparse', 'urljoin']}**

```

看看这两个值都已添加到列表中？由服务器决定如何解释这一点。如果我们发送这个查询字符串，那么它可能只选择一个值并使用它，同时忽略重复。您只能尝试一下，看看会发生什么。

通常，您可以通过使用Web浏览器通过Web界面提交查询并检查结果页面的URL来弄清楚对于给定页面需要在查询字符串中放置什么。您应该能够找到搜索文本的文本，从而推断出搜索文本的相应键。很多时候，查询字符串中的许多其他参数实际上并不需要获得基本结果。尝试仅使用搜索文本参数请求页面，然后查看发生了什么。然后，如果预期的结果没有实现，添加其他参数。

如果您向页面提交表单并且结果页面的URL没有查询字符串，则该页面将使用不同的方法发送表单数据。我们将在接下来的*HTTP方法*部分中查看这一点，同时讨论POST方法。

## URL编码

URL仅限于ASCII字符，并且在此集合中，许多字符是保留字符，并且需要在URL的不同组件中进行转义。我们通过使用称为URL编码的东西来对它们进行转义。它通常被称为**百分比编码**，因为它使用百分号作为转义字符。让我们对字符串进行URL编码：

```py
**>>> from urllib.parse import quote**
**>>> quote('A duck?')**
**'A%20duck%3F'**

```

特殊字符`' '`和`?`已被转换为转义序列。转义序列中的数字是十六进制中的字符ASCII代码。

需要转义保留字符的完整规则在RFC 3986中给出，但是`urllib`为我们提供了一些帮助我们构造URL的方法。这意味着我们不需要记住所有这些！

我们只需要：

+   对路径进行URL编码

+   对查询字符串进行URL编码

+   使用`urllib.parse.urlunparse()`函数将它们组合起来

让我们看看如何在代码中使用上述步骤。首先，我们对路径进行编码：

```py
**>>> path = 'pypi'**
**>>> path_enc = quote(path)**

```

然后，我们对查询字符串进行编码：

```py
**>>> from urllib.parse import urlencode**
**>>> query_dict = {':action': 'search', 'term': 'Are you quite sure this is a cheese shop?'}**
**>>> query_enc = urlencode(query_dict)**
**>>> query_enc**
**'%3Aaction=search&term=Are+you+quite+sure+this+is+a+cheese+shop%3F'**

```

最后，我们将所有内容组合成一个URL：

```py
**>>> from urllib.parse import urlunparse**
**>>> netloc = 'pypi.python.org'**
**>>> urlunparse(('http', netloc, path_enc, '', query_enc, ''))**
**'http://pypi.python.org/pypi?%3Aaction=search&term=Are+you+quite+sure +this+is+a+cheese+shop%3F'**

```

`quote()`函数已经设置用于特定编码路径。默认情况下，它会忽略斜杠字符并且不对其进行编码。在前面的示例中，这并不明显，尝试以下内容以查看其工作原理：

```py
**>>> from urllib.parse import quote**
**>>> path = '/images/users/+Zoot+/'**
**>>> quote(path)**
**'/images/users/%2BZoot%2B/'**

```

请注意，它忽略了斜杠，但转义了`+`。这对路径来说是完美的。

`urlencode()`函数类似地用于直接从字典编码查询字符串。请注意，它如何正确地对我们的值进行百分比编码，然后使用`&`将它们连接起来，以构造查询字符串。

最后，`urlunparse()`方法期望包含与`urlparse()`结果匹配的元素的6元组，因此有两个空字符串。

对于路径编码有一个注意事项。如果路径的元素本身包含斜杠，那么我们可能会遇到问题。示例在以下命令中显示：

```py
**>>> username = '+Zoot/Dingo+'**
**>>> path = 'images/users/{}'.format(username)**
**>>> quote(path)**
**'images/user/%2BZoot/Dingo%2B'**

```

注意用户名中的斜杠没有被转义吗？这将被错误地解释为额外的目录结构，这不是我们想要的。为了解决这个问题，首先我们需要单独转义可能包含斜杠的路径元素，然后手动连接它们：

```py
**>>> username = '+Zoot/Dingo+'**
**>>> user_encoded = quote(username, safe='')**
**>>> path = '/'.join(('', 'images', 'users', username))**
**'/images/users/%2BZoot%2FDingo%2B'**

```

注意用户名斜杠现在是百分比编码了吗？我们单独对用户名进行编码，告诉`quote`不要忽略斜杠，通过提供`safe=''`参数来覆盖其默认的忽略列表`/`。然后，我们使用简单的`join()`函数组合路径元素。

在这里，值得一提的是，通过网络发送的主机名必须严格遵循ASCII，但是`socket`和`http`模块支持将Unicode主机名透明地编码为ASCII兼容的编码，因此在实践中我们不需要担心编码主机名。关于这个过程的更多细节可以在`codecs`模块文档的`encodings.idna`部分找到。

## URL总结

在前面的部分中，我们使用了相当多的函数。让我们简要回顾一下我们每个函数的用途。所有这些函数都可以在`urllib.parse`模块中找到。它们如下：

+   将URL拆分为其组件：`urlparse`

+   将绝对URL与相对URL组合：`urljoin`

+   将查询字符串解析为`dict`：`parse_qs`

+   对路径进行URL编码：`quote`

+   从`dict`创建URL编码的查询字符串：`urlencode`

+   从组件创建URL（`urlparse`的反向）：`urlunparse`

# HTTP方法

到目前为止，我们一直在使用请求来请求服务器向我们发送网络资源，但是HTTP提供了更多我们可以执行的操作。我们请求行中的`GET`是一个HTTP **方法**，有几种方法，比如`HEAD`、`POST`、`OPTION`、`PUT`、`DELETE`、`TRACE`、`CONNECT`和`PATCH`。

我们将在下一章中详细讨论其中的一些，但现在我们将快速查看两种方法。

## HEAD方法

`HEAD`方法与`GET`方法相同。唯一的区别是服务器永远不会在响应中包含正文，即使在请求的URL上有一个有效的资源。`HEAD`方法用于检查资源是否存在或是否已更改。请注意，一些服务器不实现此方法，但当它们这样做时，它可以证明是一个巨大的带宽节省者。

我们使用`urllib`中的替代方法，通过在创建`Request`对象时提供方法名称：

```py
**>>> req = Request('http://www.google.com', method='HEAD')**
**>>> response = urlopen(req)**
**>>> response.status**
**200**
**>>> response.read()**
**b''**

```

这里服务器返回了一个`200 OK`响应，但是正文是空的，这是预期的。

## POST方法

`POST`方法在某种意义上是`GET`方法的相反。我们使用`POST`方法向服务器发送数据。然而，服务器仍然可以向我们发送完整的响应。`POST`方法用于提交HTML表单中的用户输入和向服务器上传文件。

在使用`POST`时，我们希望发送的数据将放在请求的正文中。我们可以在那里放入任何字节数据，并通过在我们的请求中添加`Content-Type`头来声明其类型，使用适当的MIME类型。

让我们通过一个例子来看看如何通过POST请求向服务器发送一些HTML表单数据，就像浏览器在网站上提交表单时所做的那样。表单数据始终由键/值对组成；`urllib`让我们可以使用常规字典来提供这些数据（我们将在下一节中看到这些数据来自哪里）：

```py
**>>> data_dict = {'P': 'Python'}**

```

在发布HTML表单数据时，表单值必须以与URL中的**查询字符串**相同的方式进行格式化，并且必须进行URL编码。还必须设置`Content-Type`头为特殊的MIME类型`application/x-www-form-urlencoded`。

由于这种格式与查询字符串相同，我们可以在准备数据时使用`urlencode()`函数：

```py
**>>> data = urlencode(data_dict).encode('utf-8')**

```

在这里，我们还将结果额外编码为字节，因为它将作为请求的主体发送。在这种情况下，我们使用UTF-8字符集。

接下来，我们将构建我们的请求：

```py
**>>> req = Request('http://search.debian.org/cgi-bin/omega', data=data)**

```

通过将我们的数据作为`data`关键字参数添加，我们告诉`urllib`我们希望我们的数据作为请求的主体发送。这将使请求使用`POST`方法而不是`GET`方法。

接下来，我们添加`Content-Type`头：

```py
**>>> req.add_header('Content-Type', 'application/x-www-form-urlencode;  charset=UTF-8')**

```

最后，我们提交请求：

```py
**>>> response = urlopen(req)**

```

如果我们将响应数据保存到文件并在网络浏览器中打开它，那么我们应该会看到一些与Python相关的Debian网站搜索结果。

# 正式检查

在前一节中，我们使用了URL`http://search.debian.org/cgibin/omega`，和字典`data_dict = {'P': 'Python'}`。但这些是从哪里来的呢？

我们通过访问包含我们手动提交以获取结果的表单的网页来获得这些信息。然后我们检查网页的HTML源代码。如果我们在网络浏览器中进行上述搜索，那么我们很可能会在[http://www.debian.org](http://www.debian.org)页面上，并且我们将通过在右上角的搜索框中输入搜索词然后点击**搜索**来进行搜索。

大多数现代浏览器允许您直接检查页面上任何元素的源代码。要做到这一点，右键单击元素，这种情况下是搜索框，然后选择**检查元素**选项，如此屏幕截图所示：

![正式检查](graphics/6008OS_02_01.jpg)

源代码将在窗口的某个部分弹出。在前面的屏幕截图中，它位于屏幕的左下角。在这里，您将看到一些代码行，看起来像以下示例：

```py
<form action="http://search.debian.org/cgi-bin/omega"
method="get" name="P">
  <p>
    <input type="hidden" value="en" name="DB"></input>
    **<input size="27" value="" name="P"></input>**
    <input type="submit" value="Search"></input>
  </p>
</form>
```

您应该看到第二个高亮显示的`<input>`。这是对应于搜索文本框的标签。高亮显示的`<input>`标签上的`name`属性的值是我们在`data_dict`中使用的键，这种情况下是`P`。我们`data_dict`中的值是我们要搜索的术语。

要获取URL，我们需要在高亮显示的`<input>`上方查找包围的`<form>`标签。在这里，我们的URL将是`action`属性的值，[http://search.debian.org/cgi-bin/omega](http://search.debian.org/cgi-bin/omega)。本书的源代码下载中包含了此网页的源代码，以防Debian在您阅读之前更改他们的网站。

这个过程可以应用于大多数HTML页面。要做到这一点，找到与输入文本框对应的`<input>`，然后从包围的`<form>`标签中找到URL。如果您不熟悉HTML，那么这可能是一个反复试验的过程。我们将在下一章中看一些解析HTML的更多方法。

一旦我们有了我们的输入名称和URL，我们就可以构建并提交POST请求，就像在前一节中所示的那样。

# HTTPS

除非另有保护，所有HTTP请求和响应都是以明文发送的。任何可以访问消息传输的网络的人都有可能拦截我们的流量并毫无阻碍地阅读它。

由于网络用于传输大量敏感数据，已经创建了一些解决方案，以防止窃听者阅读流量，即使他们能够拦截它。这些解决方案在很大程度上采用了某种形式的加密。

加密HTTP流量的标准方法称为HTTP安全，或**HTTPS**。它使用一种称为TLS/SSL的加密机制，并应用于HTTP流量传输的TCP连接上。HTTPS通常使用TCP端口443，而不是默认的HTTP端口80。

对于大多数用户来说，这个过程几乎是透明的。原则上，我们只需要将URL中的http更改为https。由于`urllib`支持HTTPS，因此对于我们的Python客户端也是如此。

请注意，并非所有服务器都支持HTTPS，因此仅将URL方案更改为`https:`并不能保证适用于所有站点。如果是这种情况，连接尝试可能会以多种方式失败，包括套接字超时、连接重置错误，甚至可能是HTTP错误，如400范围错误或500范围错误。然而，越来越多的站点正在启用HTTPS。许多其他站点正在切换到HTTPS并将其用作默认协议，因此值得调查它是否可用，以便为应用程序的用户提供额外的安全性。

# `Requests`库

这就是关于`urllib`包的全部内容。正如你所看到的，访问标准库对于大多数HTTP任务来说已经足够了。我们还没有涉及到它的所有功能。还有许多处理程序类我们没有讨论，而且打开接口是可扩展的。

然而，API并不是最优雅的，已经有几次尝试来改进它。其中一个是非常受欢迎的第三方库**Requests**。它作为`requests`包在PyPi上可用。它可以通过Pip安装，也可以从[http://docs.python-requests.org](http://docs.python-requests.org)下载，该网站提供了文档。

`Requests`库自动化并简化了我们一直在研究的许多任务。最快的说明方法是尝试一些示例。

使用`Requests`检索URL的命令与使用`urllib`包检索URL的命令类似，如下所示：

```py
**>>> import requests**
**>>> response = requests.get('http://www.debian.org')**

```

我们可以查看响应对象的属性。尝试：

```py
**>>> response.status_code**
**200**
**>>> response.reason**
**'OK'**
**>>> response.url**
**'http://www.debian.org/'**
**>>> response.headers['content-type']**
**'text/html'**

```

请注意，前面命令中的标头名称是小写的。`Requests`响应对象的`headers`属性中的键是不区分大小写的。

响应对象中添加了一些便利属性：

```py
**>>> response.ok**
**True**

```

`ok`属性指示请求是否成功。也就是说，请求包含的状态码在200范围内。另外：

```py
**>>> response.is_redirect**
**False**

```

`is_redirect`属性指示请求是否被重定向。我们还可以通过响应对象访问请求属性：

```py
**>>> response.request.headers**
**{'User-Agent': 'python-requests/2.3.0 CPython/3.4.1 Linux/3.2.0-4- amd64', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*'}**

```

请注意，`Requests`会自动处理压缩。它在`Accept-Encoding`头中包括`gzip`和`deflate`。如果我们查看`Content-Encoding`响应，我们会发现响应实际上是`gzip`压缩的，而`Requests`会自动为我们解压缩：

```py
**>>> response.headers['content-encoding']**
**'gzip'**

```

我们可以以更多的方式查看响应内容。要获得与`HTTPResponse`对象相同的字节对象，执行以下操作：

```py
**>>> response.content**
**b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n<html lang="en">...**

```

但是，`Requests`还会自动解码。要获取解码后的内容，请执行以下操作：

```py
**>>> response.text**
**'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n<html lang="en">\n<head>\n**
**...**

```

请注意，这现在是`str`而不是`bytes`。`Requests`库使用头中的值来选择字符集并将内容解码为Unicode。如果无法从头中获取字符集，则使用`chardet`库（[http://pypi.python.org/pypi/chardet](http://pypi.python.org/pypi/chardet)）从内容本身进行估计。我们可以看到`Requests`选择了哪种编码：

```py
**>>> response.encoding**
**'ISO-8859-1'**

```

我们甚至可以要求它更改已使用的编码：

```py
**>>> response.encoding = 'utf-8'**

```

更改编码后，对于此响应的`text`属性的后续引用将返回使用新编码设置解码的内容。

`Requests`库会自动处理Cookie。试试这个：

```py
**>>> response = requests.get('http://www.github.com')**
**>>> print(response.cookies)**
**<<class 'requests.cookies.RequestsCookieJar'>**
**[<Cookie logged_in=no for .github.com/>,**
 **<Cookie _gh_sess=eyJzZxNz... for ..github.com/>]>**

```

“Requests”库还有一个“Session”类，允许重复使用cookie，这类似于使用“http”模块的“CookieJar”和“urllib”模块的“HTTPCookieHandler”对象。要在后续请求中重复使用cookie，请执行以下操作：

```py
**>>> s = requests.Session()**
**>>> s.get('http://www.google.com')**
**>>> response = s.get('http://google.com/preferences')**

```

“Session”对象具有与“requests”模块相同的接口，因此我们可以像使用“requests.get()”方法一样使用其“get()”方法。现在，遇到的任何cookie都将存储在“Session”对象中，并且在将来使用“get()”方法时将随相应的请求发送。

重定向也会自动跟随，方式与使用“urllib”时相同，并且任何重定向的请求都会被捕获在“history”属性中。

不同的HTTP方法很容易访问，它们有自己的功能：

```py
**>>> response = requests.head('http://www.google.com')**
**>>> response.status_code**
**200**
**>>> response.text**
**''**

```

自定义标头以类似于使用“urllib”时的方式添加到请求中：

```py
**>>> headers = {'User-Agent': 'Mozilla/5.0 Firefox 24'}**
**>>> response = requests.get('http://www.debian.org', headers=headers)**

```

使用查询字符串进行请求是一个简单的过程：

```py
**>>> params = {':action': 'search', 'term': 'Are you quite sure this is a cheese shop?'}**
**>>> response = requests.get('http://pypi.python.org/pypi', params=params)**
**>>> response.url**
**'https://pypi.python.org/pypi?%3Aaction=search&term=Are+you+quite+sur e+this+is+a+cheese+shop%3F'**

```

“Requests”库为我们处理所有的编码和格式化工作。

发布也同样简化，尽管我们在这里使用“data”关键字参数：

```py
**>>> data = {'P', 'Python'}**
**>>> response = requests.post('http://search.debian.org/cgi- bin/omega', data=data)**

```

## 使用Requests处理错误

“Requests”中的错误处理与使用“urllib”处理错误的方式略有不同。让我们通过一些错误条件来看看它是如何工作的。通过以下操作生成一个404错误：

```py
**>>> response = requests.get('http://www.google.com/notawebpage')**
**>>> response.status_code**
**404**

```

在这种情况下，“urllib”会引发异常，但请注意，“Requests”不会。 “Requests”库可以检查状态代码并引发相应的异常，但我们必须要求它这样做：

```py
**>>> response.raise_for_status()**
**...**
**requests.exceptions.HTTPError: 404 Client Error**

```

现在，尝试在成功的请求上进行测试：

```py
**>>> r = requests.get('http://www.google.com')**
**>>> r.status_code**
**200**
**>>> r.raise_for_status()**
**None**

```

它不做任何事情，这在大多数情况下会让我们的程序退出“try/except”块，然后按照我们希望的方式继续。

如果我们遇到协议栈中较低的错误会发生什么？尝试以下操作：

```py
**>>> r = requests.get('http://192.0.2.1')**
**...**
**requests.exceptions.ConnectionError: HTTPConnectionPool(...**

```

我们已经发出了一个主机不存在的请求，一旦超时，我们就会收到一个“ConnectionError”异常。

与“urllib”相比，“Requests”库简化了在Python中使用HTTP所涉及的工作量。除非您有使用“urllib”的要求，我总是建议您在项目中使用“Requests”。

# 总结

我们研究了HTTP协议的原则。我们看到如何使用标准库“urllib”和第三方“Requests”包执行许多基本任务。

我们研究了HTTP消息的结构，HTTP状态代码，我们可能在请求和响应中遇到的不同标头，以及如何解释它们并用它们来定制我们的请求。我们看了URL是如何形成的，以及如何操作和构建它们。

我们看到了如何处理cookie和重定向，如何处理可能发生的错误，以及如何使用安全的HTTP连接。

我们还介绍了如何以提交网页表单的方式向网站提交数据，以及如何从页面源代码中提取我们需要的参数。

最后，我们看了第三方的“Requests”包。我们发现，与“urllib”包相比，“Requests”自动化并简化了我们可能需要用HTTP进行的许多常规任务。这使得它成为日常HTTP工作的绝佳选择。

在下一章中，我们将运用我们在这里学到的知识，与不同的网络服务进行详细的交互，查询API以获取数据，并将我们自己的对象上传到网络。
