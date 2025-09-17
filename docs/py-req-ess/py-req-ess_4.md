# 第四章：使用 HTTPretty 模拟 HTTP 请求

使用 `Requests` 模块，我们获得了打开 URL、发送数据以及从网络服务获取数据的手段。让我们以构建一个应用程序的实例来举例，该应用程序使用 RESTful API，不幸的是，服务器运行的 API 出现了故障。尽管我们通过 Requests 实现了与网络的交互，但这次我们失败了，因为我们从服务器端没有得到任何响应。这种状况可能会让我们感到恼火并阻碍我们的进展，因为我们找不到进一步测试我们代码的方法。

因此，产生了创建一个 HTTP 请求模拟工具的想法，它可以模拟客户端的 Web 服务器来为我们提供服务。即使 HTTPretty 与 Requests 没有直接连接，我们仍然希望介绍一个模拟工具，以帮助我们在之前提到的情况下。

### 注意事项

HTTP 模拟工具通过伪造请求来模拟网络服务。

我们在本章中将探讨以下主题：

+   理解 HTTPretty

+   安装 HTTPretty

+   详细用法

+   设置标题

+   与响应一起工作

# 理解 HTTPretty

HTTPretty 是一个用于 Python 的 HTTP 客户端模拟库。HTTPretty 的基本理念受到了 Ruby 社区中广为人知的 FakeWeb 的启发，它通过模拟请求和响应来重新实现 HTTP 协议。

本质上，HTTPretty 在套接字级别上工作，这使得它具有与大多数 HTTP 客户端库协同工作的内在优势，并且它特别针对像 `Requests`、`httplib2` 和 `urlib2` 这样的 HTTP 客户端库进行了实战测试。因此，我们可以毫无困难地模拟我们的请求库中的交互。

这里是 HTTPretty 提供帮助的两个案例：

+   API 服务器宕机的情况

+   API 内容发生变化的条件

# 安装 HTTPretty

我们可以轻松地从**Python 软件包索引**（**PyPi**）安装 HTTPretty。

```py
pip install HTTPretty

```

在熟悉 HTTPretty 的过程中，我们将通过实例学习到更多内容；在这个过程中，我们将使用 mock、sure 以及显然的 Requests 等库。现在，让我们开始这些安装：

```py
>>> pip install requests sure mock

```

让我们一探究竟，看看那些包具体处理了什么：

+   `mock`: 它是一个测试库，允许我们用模拟对象替换被测试系统中的部分组件

+   `sure`: 它是一个用于进行断言的 Python 库

# 使用 HTTPretty

处理 HTTPretty 时需要遵循三个主要步骤：

1.  启用 HTTPretty

1.  将统一资源定位符注册到 HTTPretty

1.  禁用 HTTPretty

我们应该最初启用 HTTPretty，这样它就会应用猴子补丁；也就是说，动态替换套接字模块的属性。我们将使用`register_uri`函数来注册统一资源定位符。`register_uri`函数接受`class`、`uri`和`body`作为参数：

```py
 method: register_uri(class, uri, body)

```

在我们的测试过程结束时，我们应该禁用 HTTPretty，以免它改变其他组件的行为。让我们通过一个示例来看看如何使用 HTTPretty：

```py
import httpretty
import requests
from sure import expect

def example():
 httpretty.enable()
 httpretty.register_uri(httpretty.GET, "http://google.com/",
 body="This is the mocked body",
 status=201)
 response = requests.get("http://google.com/")
 expect(response.status_code).to.equal(201)
 httpretty.disable()

```

在这个例子中，我们使用了`httpretty.GET`类在`register_uri`函数中注册了`uri`值为`"http://google.com/"`。在下一行，我们使用 Request 从 URI 获取信息，然后使用 expect 函数断言预期的状态码。总的来说，前面的代码试图模拟 URI 并测试我们是否得到了预期的相同状态码。

我们可以使用装饰器来简化前面的代码。正如第一步和第三步，即启用和禁用 HTTPretty 始终相同，我们可以使用装饰器，这样那些函数就可以在我们需要它们出现的时候被包装起来。装饰器看起来是这样的：`@httpretty.activate`。之前的代码示例可以用以下方式使用装饰器重写：

```py
import httpretty
import requests

from sure import expect

@httpretty.activate
def example():
 httpretty.register_uri(httpretty.GET, "http://google.com/",
 body="This is the mocked body",
 status=201)
 response = requests.get("http://google.com/")
 expect(response.status_code).to.equal(201)

```

# 设置标题

HTTP 头部字段提供了关于请求或响应的必要信息。我们可以通过使用 HTTPretty 来模拟任何 HTTP 响应头部。为了实现这一点，我们将它们作为关键字参数添加。我们应该记住，关键字参数的键始终是小写字母，并且使用下划线（_）而不是破折号（-）。

例如，如果我们想模拟返回 Content-Type 的服务器，我们可以使用`content_type`参数。请注意，在下面的部分，我们使用一个不存在的 URL 来展示语法：

```py
import httpretty
import requests

from sure import expect

@httpretty.activate
def setting_header_example():
 httpretty.register_uri(httpretty.GET,
 "http://api.example.com/some/path",
 body='{"success": true}',
 status=200,
 content_type='text/json')

 response = requests.get("http://api.example.com/some/path")

 expect(response.json()).to.equal({'success': True})
 expect(response.status_code).to.equal(200)

```

同样，HTTPretty 会接收所有关键字参数并将其转换为 RFC2616 的等效名称。

# 与响应一起工作

当我们使用 HTTPretty 模拟 HTTP 请求时，它会返回一个`httpretty.Response`对象。我们可以通过回调函数生成以下响应：

+   旋转响应

+   流式响应

+   动态响应

## 旋转响应

旋转响应是我们向服务器发送请求时，按照给定顺序收到的响应。我们可以使用响应参数定义我们想要的任意数量的响应。

以下片段解释了旋转响应的模拟过程：

```py
import httpretty
import requests

from sure import expect

@httpretty.activate
def rotating_responses_example():
 URL = "http://example.com/some/path"
 RESPONSE_1 = "This is Response 1."
 RESPONSE_2 = "This is Response 2."
 RESPONSE_3 = "This is Last Response."

 httpretty.register_uri(httpretty.GET,
 URL,
 responses=[
 httpretty.Response(body=RESPONSE_1,
 status=201),
 httpretty.Response(body=RESPONSE_2,
 status=202),
 httpretty.Response(body=RESPONSE_3,
 status=201)])

 response_1 = requests.get(URL)
 expect(response_1.status_code).to.equal(201)
 expect(response_1.text).to.equal(RESPONSE_1)

 response_2 = requests.get(URL)
 expect(response_2.status_code).to.equal(202)
 expect(response_2.text).to.equal(RESPONSE_2)

 response_3 = requests.get(URL)
 expect(response_3.status_code).to.equal(201)
 expect(response_3.text).to.equal(RESPONSE_3)

 response_4 = requests.get(URL)
 expect(response_4.status_code).to.equal(201)
 expect(response_4.text).to.equal(RESPONSE_3)

```

在这个例子中，我们使用`httpretty.register_uri`方法通过`responses`参数注册了三种不同的响应。然后，我们向服务器发送了四个不同的请求，这些请求具有相同的 URI 和相同的方法。结果，我们按照注册的顺序收到了前三个响应。从第四个请求开始，我们将获得`responses`对象中定义的最后一个响应。

## 流式响应

流式响应将不会包含`Content-Length`头部。相反，它们有一个值为`chunked`的`Transfer-Encoding`头部，以及由一系列数据块组成的内容体，这些数据块由它们各自的大小值 precede。这类响应也被称为**分块响应**。

我们可以通过注册一个生成器响应体来模拟一个 Streaming 响应：

```py
import httpretty
import requests
from time import sleep
from sure import expect

def mock_streaming_repos(repos):
 for repo in repos:
 sleep(.5)
 yield repo

@httpretty.activate
def streaming_responses_example():
 URL = "https://api.github.com/orgs/python/repos"
 REPOS = ['{"name": "repo-1", "id": 1}\r\n',
 '\r\n',
 '{"name": "repo-2", "id": 2}\r\n']

 httpretty.register_uri(httpretty.GET,
 URL,
 body=mock_streaming_repos(REPOS),
 streaming=True)

 response = requests.get(URL,
 data={"track": "requests"})

 line_iter = response.iter_lines()
 for i in xrange(len(REPOS)):
 expect(line_iter.next().strip()).to.equal(REPOS[i].strip())

```

为了模拟流式响应，我们需要在注册`uri`时将流式参数设置为`True`。在之前的示例中，我们使用生成器`mock_streaming_repos`来模拟流式响应，该生成器将列表作为参数，并且每半秒产生列表中的一个项目。

## 通过回调函数实现动态响应

如果 API 服务器的响应是根据请求的值生成的，那么我们称之为动态响应。为了根据请求模拟动态响应，我们将使用以下示例中定义的回调方法：

```py
import httpretty
import requests

from sure import expect

@httpretty.activate
def dynamic_responses_example():
 def request_callback(method, uri, headers):
 return (200, headers, "The {} response from {}".format(method, uri)
 httpretty.register_uri(
 httpretty.GET, "http://example.com/sample/path",
 body=request_callback)

 response = requests.get("http://example.com/sample/path")

 expect(response.text).to.equal(' http://example.com/sample/path')

```

在此示例中，在模拟响应时注册了`request_callback`方法，以便生成动态响应内容。

# 摘要

在本章中，我们学习了与 HTTPretty 相关的基本概念。我们了解了 HTTPretty 是什么，以及为什么我们需要 HTTPretty。我们还详细介绍了模拟库的使用，设置头部信息和模拟不同类型的响应。这些主题足以让我们开始并保持进展。

在下一章中，我们将学习如何使用 requests 库与社交网络如 Facebook、Twitter 和 Reddit 进行交互。
