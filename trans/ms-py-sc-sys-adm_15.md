# SOAP 和 REST API 通信

在本章中，我们将了解 SOAP 和 REST API 的基础知识。我们还将了解 Python 用于 SOAP 和 REST API 的库。我们将学习有关 SOAP 的 Zeep 库和 REST API 的请求。您将学习如何处理 JSON 数据。我们将看到处理 JSON 数据的简单示例，例如将 JSON 字符串转换为 Python 对象和将 Python 对象转换为 JSON 字符串。

在本章中，您将学习以下内容：

+   SOAP 是什么？

+   使用 SOAP 的库

+   什么是 RESTful API？

+   使用标准库进行 RESTful API

+   处理 JSON 数据

# SOAP 是什么？

**SOAP**是**简单对象访问协议**。SOAP 是允许进程使用不同操作系统的标准通信协议系统。这些通过 HTTP 和 XML 进行通信。它是一种 Web 服务技术。SOAP API 主要用于创建，更新，删除和恢复数据等任务。SOAP API 使用 Web 服务描述语言来描述 Web 服务提供的功能。SOAP 描述所有功能和数据类型。它构建了一个基于 XML 的协议。

# 使用 SOAP 的库

在本节中，我们将学习有关 Python 用于 SOAP 的库。以下是用于 SOAP 的各种库：

+   SOAPpy

+   `Zeep`

+   `Ladon`

+   `suds-jurko`

+   `pysimplesoap`

这些是 Python 的 SOAP API 库。在本节中，我们将只学习有关 Zeep 库的知识。

要使用 Zeep 的功能，您首先需要安装它。在终端中运行以下命令以安装 Zeep：

```py
 $ pip3 install Zeep
```

`Zeep`模块用于 WSDL 文档。它为服务和文档生成代码，并为 SOAP 服务器提供编程接口。`lxml`库用于解析 XML 文档。

现在，我们将看一个例子。创建一个`soap_example.py`脚本，并在其中写入以下代码：

```py
import zeep w = 'http://www.soapclient.com/xml/soapresponder.wsdl' c = zeep.Client(wsdl=w) print(c.service.Method1('Hello', 'World'))
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~$ python3 soap_example.py Output : Your input parameters are Hello and World
```

在上面的例子中，我们首先导入了`zeep`模块。我们首先提到了网站名称。然后我们创建了`zeep`客户端对象。我们之前使用的 WSDL 定义了一个简单的`Method1`函数，通过`zeep`通过`client.service.Method1`提供。它接受两个参数并返回一个字符串。

# 什么是 RESTful API？

**REST**代表**表述性状态转移**。RESTful API 是在 Web 服务开发中使用的一种通信方法。它是一种作为互联网上不同系统之间通信渠道的 Web 服务风格。它是一个应用程序接口，用于使用`HTTP`请求`GET`，`PUT`，`POST`和`DELETE`数据。

REST 的优势在于它使用的带宽较少，适合互联网使用。REST API 使用统一的接口。所有资源都由`GET`，`POST`，`PUT`和`DELETE`操作处理。`REST` API 使用`GET`来检索资源，使用`PUT`来更新资源或更改资源状态，使用`POST`来创建资源，使用`DELETE`来删除资源。使用 REST API 的系统提供快速性能和可靠性。

REST API 独立处理每个请求。从客户端到服务器的请求必须包含理解该请求所需的所有信息。

# 使用标准库进行 RESTful API

在本节中，我们将学习如何使用 RESTful API。为此，我们将使用 Python 的`requests`和 JSON 模块。我们现在将看一个例子。首先，我们将使用`requests`模块从 API 获取信息。我们将看到`GET`和`POST`请求。

首先，您必须安装`requests`库，如下所示：

```py
 $ pip3 install requests
```

现在，我们将看一个例子。创建一个`rest_get_example.py`脚本，并在其中写入以下内容：

```py
import requests req_obj = requests.get('https://www.imdb.com/news/top?ref_=nv_tp_nw') print(req_obj)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 rest_get_example.py Output: <Response [200]>
```

在前面的示例中，我们导入了`requests`模块来获取请求。接下来，我们创建了一个请求对象`req_obj`，并指定了我们想要获取请求的链接。然后，我们打印了它。我们得到的输出是状态码`200`，表示成功。

现在，我们将看到`POST`请求的示例。`POST`请求用于向服务器发送数据。创建一个`rest_post_example.py`脚本，并在其中写入以下内容：

```py
import requests import json url_name = 'http://httpbin.org/post' data = {"Name" : "John"} data_json = json.dumps(data) headers = {'Content-type': 'application/json'} response = requests.post(url_name, data=data_json, headers=headers) print(response)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 rest_post_example.py Output: <Response [200]>
```

在前面的示例中，我们学习了关于`POST`请求。首先，我们导入了必要的模块 requests 和 JSON。接下来，我们提到了 URL。此外，我们以字典格式输入了要发布的数据。接下来，我们提到了标头。然后，我们使用`POST`请求发布。我们得到的输出是状态码`200`，这是一个成功的代码。

# 处理 JSON 数据

在本节中，我们将学习有关 JSON 数据。**JSON**代表**JavaScript 对象表示**。JSON 是一种数据交换格式。它将 Python 对象编码为 JSON 字符串，并将 JSON 字符串解码为 Python 对象。Python 有一个 JSON 模块，用于格式化 JSON 输出。它具有用于序列化和反序列化 JSON 的函数。

+   `json.dump(obj, fileObj)`: 这个函数将一个对象序列化为一个 JSON 格式的流。

+   `json.dumps(obj)`: 这个函数将一个对象序列化为一个 JSON 格式的字符串。

+   `json.load(JSONfile)`: 这个函数将一个 JSON 文件反序列化为一个 Python 对象。

+   `json.loads(JSONfile)`: 这个函数将一个字符串类型的 JSON 文件反序列化为一个 Python 对象。

它还列出了编码和解码的两个类：

+   `JSONEncoder`: 用于将 Python 对象转换为 JSON 格式。

+   `JSONDecoder`: 用于将 JSON 格式的文件转换为 Python 对象。

现在，我们将看到一些使用 JSON 模块的示例。首先，我们将看到从 JSON 到 Python 的转换。为此，创建一个名为`json_to_python.py`的脚本，并在其中写入以下代码：

```py
import json j_obj =  '{ "Name":"Harry", "Age":26, "Department":"HR"}' p_obj = json.loads(j_obj) print(p_obj["Name"]) print(p_obj["Department"])
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 json_to_python.py Output: Harry HR
```

在前面的示例中，我们编写了一个代码，将 JSON 字符串转换为 Python 对象。`json.loads()`函数用于将 JSON 字符串转换为 Python 对象。

现在，我们将看到如何将 Python 转换为 JSON。为此，创建一个`python_to_json.py`脚本，并在其中写入以下代码：

```py
import json emp_dict1 =  '{ "Name":"Harry", "Age":26, "Department":"HR"}' json_obj = json.dumps(emp_dict1) print(json_obj)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 python_to_json.py Output: "{ \"Name\":\"Harry\", \"Age\":26, \"Department\":\"HR\"}"
```

在前面的示例中，我们将 Python 对象转换为 JSON 字符串。`json.dumps()`函数用于这种转换。

现在，我们将看到如何将各种类型的 Python 对象转换为 JSON 字符串。为此，创建一个`python_object_to_json.py`脚本，并在其中写入以下内容：

```py
import json python_dict =  {"Name": "Harry", "Age": 26} python_list =  ["Mumbai", "Pune"] python_tuple =  ("Basketball", "Cricket") python_str =  ("hello_world") python_int =  (150) python_float =  (59.66) python_T =  (True) python_F =  (False) python_N =  (None) json_obj = json.dumps(python_dict) json_arr1 = json.dumps(python_list) json_arr2 = json.dumps(python_tuple) json_str = json.dumps(python_str) json_num1 = json.dumps(python_int) json_num2 = json.dumps(python_float) json_t = json.dumps(python_T) json_f = json.dumps(python_F) json_n = json.dumps(python_N) print("json object : ", json_obj) print("json array1 : ", json_arr1) print("json array2 : ", json_arr2) print("json string : ", json_str) print("json number1 : ", json_num1) print("json number2 : ", json_num2) print("json true", json_t) print("json false", json_f) print("json null", json_n)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 python_object_to_json.py Output: json object :  {"Name": "Harry", "Age": 26} json array1 :  ["Mumbai", "Pune"] json array2 :  ["Basketball", "Cricket"] json string :  "hello_world" json number1 :  150 json number2 :  59.66 json true true json false false json null null
```

在前面的示例中，我们使用`json.dumps()`函数将各种类型的 Python 对象转换为 JSON 字符串。转换后，Python 列表和元组被转换为数组。整数和浮点数在 JSON 中被视为数字。以下是从 Python 到 JSON 的转换图表：

| **Python** | **JSON** |
| --- | --- |
| `dict` | Object |
| `list` | Array |
| `tuple` | Array |
| `str` | String |
| `int` | Number |
| `float` | Number |
| `True` | true |
| `False` | false |
| `None` | null |

# 总结

在这一章中，您学习了关于 SOAP API 和 RESTful API。您学习了关于`zeep` Python 库用于 SOAP API 和 requests 库用于 REST API。您还学会了如何处理 JSON 数据，例如将 JSON 转换为 Python，反之亦然。

在下一章中，您将学习有关网页抓取和用于执行此任务的 Python 库。

# 问题

1.  SOAP 和 REST API 之间有什么区别？

1.  `json.loads`和`json.load`之间有什么区别？

1.  JSON 支持所有平台吗？

1.  以下代码片段的输出是什么？

```py
boolean_value = False
print(json.dumps(boolean_value))
```

1.  以下代码片段的输出是什么？

```py
>> weird_json = '{"x": 1, "x": 2, "x": 3}'
>>> json.loads(weird_json)
```

# 进一步阅读

+   JSON 文档：[`docs.python.org/3/library/json.html`](https://docs.python.org/3/library/json.html)

+   REST API 信息：[`searchmicroservices.techtarget.com/definition/REST-representational-state-transfer`](https://searchmicroservices.techtarget.com/definition/REST-representational-state-transfer)
