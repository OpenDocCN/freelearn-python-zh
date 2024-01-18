# 第三章。API 的实际应用

当我们谈论 Python 中的 API 时，通常指的是模块向我们呈现的类和函数，以便与之交互。在本章中，我们将谈论一些不同的东西，即 Web API。

Web API 是一种通过 HTTP 协议与之交互的 API 类型。如今，许多 Web 服务提供一组 HTTP 调用，旨在由客户端以编程方式使用，也就是说，它们是为机器而不是人类设计的。通过这些接口，可以自动化与服务的交互，并执行诸如提取数据、以某种方式配置服务以及将自己的内容上传到服务中等任务。

在本章中，我们将看到：

+   Web API 使用的两种流行数据交换格式：XML 和 JSON

+   如何与两个主要 Web API 进行交互：Amazon S3 和 Twitter

+   在 API 不可用时如何从 HTML 页面中提取数据

+   如何为提供这些 API 和网站的网络管理员简化工作

有数百种提供 Web API 的服务。这些服务的相当全面且不断增长的列表可以在[`www.programmableweb.com`](http://www.programmableweb.com)找到。

我们将首先介绍 Python 中如何使用 XML，然后解释一种基于 XML 的 API，称为 Amazon S3 API。

# 开始使用 XML

**可扩展标记语言**（**XML**）是一种以标准文本格式表示分层数据的方式。在使用基于 XML 的 Web API 时，我们将创建 XML 文档，并将其作为 HTTP 请求的主体发送，并接收 XML 文档作为响应的主体。

以下是 XML 文档的文本表示，也许代表奶酪店的库存：

```py
<?xml version='1.0'?>
<inventory>
    <cheese id="c01">
        <name>Caerphilly</name>
        <stock>0</stock>
    </cheese>
    <cheese id="c02">
        <name>Illchester</name>
        <stock>0</stock>
    </cheese>
</inventory>
```

如果您以前使用过 HTML 编码，那么这可能看起来很熟悉。 XML 是一种基于标记的格式。它来自与 HTML 相同语言系列。数据以元素形式的层次结构进行组织。每个元素由两个标签表示，例如开始标签`<name>`和匹配的结束标签，例如`</name>`。在这两个标签之间，我们可以放置数据，例如`Caerphilly`，或者添加更多标签，代表子元素。

与 HTML 不同，XML 被设计成我们可以定义自己的标签并创建自己的数据格式。此外，与 HTML 不同，XML 语法始终严格执行。在 HTML 中，小错误（例如标签以错误顺序关闭，完全缺少关闭标签或属性值缺少引号）是可以容忍的，但在 XML 中，这些错误将导致完全无法阅读的 XML 文档。格式正确的 XML 文档称为格式良好的。

## XML API

处理 XML 数据有两种主要方法：

+   读取整个文档并创建基于对象的表示，然后使用面向对象的 API 进行操作。

+   从头到尾处理文档，并在遇到特定标签时执行操作

现在，我们将专注于使用名为**ElementTree**的 Python XML API 的基于对象的方法。第二种所谓的拉或事件驱动方法（也经常称为**SAX**，因为 SAX 是这一类别中最流行的 API 之一）设置更加复杂，并且仅在处理大型 XML 文件时才需要。我们不需要这个来处理 Amazon S3。

## ElementTree 的基础知识

我们将使用 Python 标准库中的`ElementTree` API 实现，该 API 位于`xml.etree.ElementTree`模块中。

让我们看看如何使用`ElementTree`创建上述示例 XML 文档。打开 Python 解释器并运行以下命令：

```py
**>>> import xml.etree.ElementTree as ET**
**>>> root = ET.Element('inventory')**
**>>> ET.dump(root)**
**<inventory />**

```

我们首先创建根元素，也就是文档的最外层元素。我们在这里创建了一个根元素“<inventory>”，然后将其字符串表示打印到屏幕上。“<inventory />”表示是“<inventory></inventory>”的 XML 快捷方式。它用于显示一个空元素，即没有数据和子标签的元素。

我们通过创建一个新的“ElementTree.Element”对象来创建“<inventory>”元素。您会注意到我们给“Element（）”的参数是创建的标签的名称。

我们的“<inventory>”元素目前是空的，所以让我们往里面放点东西。这样做：

```py
**>>> cheese = ET.Element('cheese')**
**>>> root.append(cheese)**
**>>> ET.dump(root)**
**<inventory><cheese /></inventory>**

```

现在，在我们的“<inventory>”元素中有一个“<cheese>”元素。当一个元素直接嵌套在另一个元素内时，那么嵌套的元素称为外部元素的**子元素**，外部元素称为**父元素**。同样，处于同一级别的元素称为**兄弟元素**。

让我们再添加另一个元素，这次给它一些内容。添加以下命令：

```py
**>>> name = ET.SubElement(cheese, 'name')**
**>>> name.text = 'Caerphilly'**
**>>> ET.dump(root)**
**<inventory><cheese><name>Caerphilly</name></cheese></inventory>**

```

现在，我们的文档开始成形了。我们在这里做了两件新事情：首先，我们使用了快捷类方法“ElementTree.SubElement（）”来创建新的“<name>”元素，并将其作为“<cheese>”的子元素一次性插入树中。其次，我们通过将一些文本赋给元素的“text”属性来为其赋予一些内容。

我们可以使用父元素上的“remove（）”方法来删除元素，如下面的命令所示：

```py
**>>> temp = ET.SubElement(root, 'temp')**
**>>> ET.dump(root)**
**<inventory><cheese><name>Caerphilly</name></cheese><temp /></inventory>**
**>>> root.remove(temp)**
**>>> ET.dump(root)**
**<inventory><cheese><name>Caerphilly</name></cheese></inventory>**

```

### 漂亮打印

我们能够以更易读的格式生成输出将会很有用，比如在本节开头展示的例子。ElementTree API 没有用于执行此操作的函数，但标准库提供的另一个 XML API“minidom”有，并且使用起来很简单。首先，导入“minidom”：

```py
**>>> import xml.dom.minidom as minidom**

```

其次，使用以下命令打印一些格式良好的 XML：

```py
**>>> print(minidom.parseString(ET.tostring(root)).toprettyxml())**
**<?xml version="1.0" ?>**
**<inventory>**
 **<cheese>**
 **<name>Caerphilly</name>**
 **</cheese>**
**</inventory>**

```

这些乍一看不是最容易的代码行，所以让我们来分解一下。 “minidom”库不能直接处理 ElementTree 元素，因此我们使用 ElementTree 的“tostring（）”函数来创建我们的 XML 的字符串表示。我们通过使用“minidom.parseString（）”将字符串加载到“minidom” API 中，然后使用“toprettyxml（）”方法输出我们格式化的 XML。

这可以封装成一个函数，使其更加方便。在 Python shell 中输入以下命令块：

```py
**>>> def xml_pprint(element):**
**...     s = ET.tostring(element)**
**...     print(minidom.parseString(s).toprettyxml())**

```

现在，只需执行以下操作进行漂亮的打印：

```py
**>>> xml_pprint(root)**
**<?xml version="1.0" ?>**
**<inventory>**
 **<cheese>**
**...**

```

### 元素属性

在本节开头展示的例子中，您可能已经注意到了“<cheese>”元素的开标签中的内容，“id =“c01””。这被称为**属性**。我们可以使用属性来附加额外的信息到元素上，元素可以拥有的属性数量没有限制。属性始终由属性名称组成，在本例中是“id”，以及一个值，在本例中是“c01”。值可以是任何文本，但必须用引号括起来。

现在，按照以下方式为“<cheese>”元素添加“id”属性：

```py
**>>> cheese.attrib['id'] = 'c01'**
**>>> xml_pprint(cheese)**
**<?xml version="1.0" ?>**
**<cheese id="c01">**
 **<name>Caerphilly</name>**
**</cheese>**

```

元素的“attrib”属性是一个类似字典的对象，保存着元素的属性名称和值。我们可以像操作常规“dict”一样操作 XML 属性。

到目前为止，您应该能够完全重新创建本节开头展示的示例文档。继续尝试吧。

### 转换为文本

一旦我们有了满意的 XML 树，通常我们会希望将其转换为字符串以便通过网络发送。我们一直在使用的“ET.dump（）”函数不适用于此。 “dump（）”函数所做的只是将标签打印到屏幕上。它不会返回我们可以使用的字符串。我们需要使用“ET.tostring（）”函数，如下面的命令所示：

```py
**>>> text = ET.tostring(name)**
**>>> print(text)**
**b'<name>Caerphilly</name>'**

```

请注意它返回一个字节对象。它为我们编码字符串。默认字符集是`us-ascii`，但最好使用 UTF-8 进行 HTTP 传输，因为它可以编码完整的 Unicode 字符范围，并且得到了 Web 应用的广泛支持。

```py
**>>> text = ET.tostring(name, encoding='utf-8')**

```

目前，这就是我们需要了解有关创建 XML 文档的所有内容，让我们看看如何将其应用到 Web API。

# 亚马逊 S3 API

亚马逊 S3 是一个数据存储服务。它支撑了今天许多知名的网络服务。尽管提供了企业级的弹性、性能和功能，但它非常容易上手。它价格合理，并且提供了一个简单的 API 用于自动访问。它是不断增长的**亚马逊网络服务**（**AWS**）组合中的众多云服务之一。

API 不断变化，通常会被赋予一个版本号，以便我们可以跟踪它们。我们将使用当前版本的 S3 REST API，“2006-03-01”。

您会注意到在 S3 文档和其他地方，S3 Web API 被称为**REST API**。**REST**代表**表述性状态转移**，这是 Roy Fielding 在他的博士论文中最初提出的关于如何使用 HTTP 进行 API 的相当学术的概念。尽管一个 API 应该具有被认为是 RESTful 的属性是非常具体的，但实际上几乎任何基于 HTTP 的 API 现在都被贴上了 RESTful 的标签。S3 API 实际上是最具有 RESTful 特性的高调 API 之一，因为它适当地使用了 HTTP 方法的广泛范围。

### 注意

如果您想了解更多关于这个主题的信息，Roy Fielding 的博士论文可以在这里找到[`ics.uci.edu/~fielding/pubs/dissertation`](http://ics.uci.edu/~fielding/pubs/dissertation)，而最初提出这个概念并且是一本很好的读物的书籍之一，*RESTful Web Services*由*Leonard Richardson*和*Sam Ruby*，现在可以从这个页面免费下载[`restfulwebapis.org/rws.html`](http://restfulwebapis.org/rws.html)。

## 注册 AWS

在我们可以访问 S3 之前，我们需要在 AWS 上注册。API 通常要求在允许访问其功能之前进行注册。您可以使用现有的亚马逊账户或在[`www.amazonaws.com`](http://www.amazonaws.com)上创建一个新账户。虽然 S3 最终是一个付费服务，但如果您是第一次使用 AWS，那么您将获得一年的免费试用，用于低容量使用。一年的时间足够完成本章的学习！试用提供 5GB 的免费 S3 存储空间。

## 认证

接下来，我们需要讨论认证，这是在使用许多 Web API 时的一个重要讨论话题。我们使用的大多数 Web API 都会指定一种提供认证凭据的方式，允许向它们发出请求，通常我们发出的每个 HTTP 请求都必须包含认证信息。

API 需要这些信息有以下原因：

+   确保其他人无法滥用应用程序的访问权限

+   应用每个应用程序的速率限制

+   管理访问权限的委托，以便应用程序可以代表服务的其他用户或其他服务进行操作

+   收集使用统计数据

所有的 AWS 服务都使用 HTTP 请求签名机制进行认证。为了签署一个请求，我们使用加密密钥对 HTTP 请求中的唯一数据进行哈希和签名，然后将签名作为标头添加到请求中。通过在服务器上重新创建签名，AWS 可以确保请求是由我们发送的，并且在传输过程中没有被更改。

AWS 签名生成过程目前处于第 4 版，需要进行详细讨论，因此我们将使用第三方库，即`requests-aws4auth`。这是一个`Requests`模块的伴侣库，可以自动处理签名生成。它可以在 PyPi 上获得。因此，请在命令行上使用`pip`安装它：

```py
**$ pip install requests-aws4auth**
**Downloading/unpacking requests-aws4auth**
**...**

```

### 设置 AWS 用户

要使用身份验证，我们需要获取一些凭据。

我们将通过 AWS 控制台进行设置。注册 AWS 后，登录到[`console.aws.amazon.com`](https://console.aws.amazon.com)控制台。

一旦您登录，您需要执行这里显示的步骤：

1.  点击右上角的您的名称，然后选择**安全凭据**。

1.  点击屏幕左侧列表中的**用户**，然后点击顶部的**创建新用户**按钮。

1.  输入**用户名**，确保已选中**为每个用户生成访问密钥**，然后点击右下角的**创建**按钮。

您将看到一个新页面，显示用户已成功创建。点击右下角的**下载凭据**按钮下载一个 CSV 文件，其中包含此用户的**访问 ID**和**访问密钥**。这些很重要，因为它们将帮助我们对 S3 API 进行身份验证。请确保将它们安全地存储，因为它们将允许完全访问您的 S3 文件。

然后，点击屏幕底部的**关闭**，点击将出现的列表中的新用户，然后点击**附加策略**按钮。将显示一系列策略模板。滚动此列表并选择**AmazonS3FullAccess**策略，如下图所示：

![设置 AWS 用户](img/6008OS_03_01.jpg)

最后，当它出现时，点击右下角的**附加策略**按钮。现在，我们的用户已完全访问 S3 服务。

## 区域

AWS 在世界各地都有数据中心，因此当我们在 AWS 中激活服务时，我们选择希望其存在的区域。S3 的区域列表在[`docs.aws.amazon.com/general/latest/gr/rande.html#s3_region`](http://docs.aws.amazon.com/general/latest/gr/rande.html#s3_region)上。

最好选择离将使用该服务的用户最近的区域。目前，您将是唯一的用户，所以只需为我们的第一个 S3 测试选择离您最近的区域。

## S3 存储桶和对象

S3 使用两个概念来组织我们存储在其中的数据：存储桶和对象。对象相当于文件，即具有名称的数据块，而存储桶相当于目录。存储桶和目录之间唯一的区别是存储桶不能包含其他存储桶。

每个存储桶都有自己的 URL 形式：

`http://<bucketname>.s3-<region>.amazonaws.com`。

在 URL 中，`<bucketname>`是存储桶的名称，`<region>`是存储桶所在的 AWS 区域，例如`eu-west-1`。存储桶名称和区域在创建存储桶时设置。

存储桶名称在所有 S3 用户之间是全局共享的，因此它们必须是唯一的。如果您拥有域名，则该域名的子域名将成为适当的存储桶名称。您还可以使用您的电子邮件地址，将`@`符号替换为连字符或下划线。

对象在我们首次上传时命名。我们通过将对象名称作为路径添加到存储桶的 URL 末尾来访问对象。例如，如果我们在`eu-west-1`区域有一个名为`mybucket.example.com`的存储桶，其中包含名为`cheeseshop.txt`的对象，那么我们可以通过 URL[`mybucket.example.com.s3-eu-west-1.amazonaws.com/cheeseshop.txt`](http://mybucket.example.com.s3-eu-west-1.amazonaws.com/cheeseshop.txt)来访问它。

让我们通过 AWS 控制台创建我们的第一个存储桶。我们可以通过这个网页界面手动执行 API 公开的大多数操作，并且这是检查我们的 API 客户端是否执行所需任务的好方法：

1.  登录到[`console.aws.amazon.com`](https://console.aws.amazon.com)控制台。

1.  转到 S3 服务。您将看到一个页面，提示您创建一个存储桶。

1.  点击**创建存储桶**按钮。

1.  输入存储桶名称，选择一个区域，然后点击**创建**。

1.  您将被带到存储桶列表，并且您将能够看到您的存储桶。

## 一个 S3 命令行客户端

好了，准备工作足够了，让我们开始编码。在接下来的 S3 部分中，我们将编写一个小的命令行客户端，这将使我们能够与服务进行交互。我们将创建存储桶，然后上传和下载文件。

首先，我们将设置我们的命令行解释器并初始化身份验证。创建一个名为`s3_client.py`的文件，并将以下代码块保存在其中：

```py
import sys
import requests
import requests_aws4auth as aws4auth
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

access_id = '<ACCESS ID>'
access_key = '<ACCESS KEY>'
region = '<REGION>'
endpoint = 's3-{}.amazonaws.com'.format(region)
auth = aws4auth.AWS4Auth(access_id, access_key, region, 's3')
ns = 'http://s3.amazonaws.com/doc/2006-03-01/'

def xml_pprint(xml_string):
    print(minidom.parseString(xml_string).toprettyxml())

def create_bucket(bucket):
    print('Bucket name: {}'.format(bucket))

if __name__ == '__main__':
    cmd, *args = sys.argv[1:]
    globals()cmd
```

### 提示

**下载示例代码**

您可以从[`www.packtpub.com`](http://www.packtpub.com)的帐户中下载您购买的所有 Packt 图书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便直接通过电子邮件接收文件。

您需要用之前下载的凭据 CSV 中的值替换`<ACCESS ID>`和`<ACCESS KEY>`，并用您选择的 AWS 区域替换`<REGION>`。

那么，我们在这里做什么呢？首先，我们设置了我们的端点。端点是一个通用术语，用于访问 API 的 URL。一些 Web API 只有一个端点，一些有多个端点，这取决于 API 的设计方式。我们在这里生成的端点实际上只是我们在使用存储桶时将使用的完整端点的一部分。我们的实际端点是由存储桶名称前缀的端点。

接下来，我们创建我们的`auth`对象。我们将与`Requests`一起使用它来为我们的 API 请求添加 AWS 身份验证。

`ns`变量是一个字符串，我们需要用它来处理来自 S3 API 的 XML。我们将在使用它时讨论这个。

我们已经包含了我们的`xml_pprint()`函数的修改版本，以帮助调试。目前，`create_bucket()`函数只是一个占位符。我们将在下一节中了解更多。

最后，我们有命令解释器本身 - 它只是获取脚本在命令行上给出的第一个参数，并尝试运行一个同名的函数，将任何剩余的命令行参数传递给函数。让我们进行一次测试。在命令提示符中输入以下内容：

```py
**$ python3.4 s3_client.py create_bucket mybucket**
**Bucket name: mybucket**

```

您可以看到脚本从命令行参数中提取`create_bucket`，因此调用`create_bucket()`函数，将`myBucket`作为参数传递。

这个框架使得添加功能来扩展我们客户的能力成为一个简单的过程。让我们从使`create_bucket()`做一些有用的事情开始。

### 使用 API 创建一个存储桶

每当我们为 API 编写客户端时，我们的主要参考点是 API 文档。文档告诉我们如何构造执行操作的 HTTP 请求。S3 文档可以在[`docs.aws.amazon.com/AmazonS3/latest/API/APIRest.html`](http://docs.aws.amazon.com/AmazonS3/latest/API/APIRest.html)找到。[`docs.aws.amazon.com/AmazonS3/latest/API/RESTBucketPUT.html`](http://docs.aws.amazon.com/AmazonS3/latest/API/RESTBucketPUT.html) URL 将提供存储桶创建的详细信息。

这份文档告诉我们，要创建一个存储桶，我们需要通过使用 HTTP `PUT`方法向我们新存储桶的端点发出 HTTP 请求。它还告诉我们，请求正文必须包含一些 XML，其中指定了我们希望创建存储桶的 AWS 区域。

所以，现在我们知道我们的目标是什么，让我们讨论我们的功能。首先，让我们创建 XML。用以下代码块替换`create_bucket()`的内容：

```py
def create_bucket(bucket):
    XML = ET.Element('CreateBucketConfiguration')
    XML.attrib['xmlns'] = ns
    location = ET.SubElement(XML, 'LocationConstraint')
    location.text = auth.region
    data = ET.tostring(XML, encoding='utf-8')
    xml_pprint(data)
```

在这里，我们创建一个遵循 S3 文档中给出的格式的 XML 树。如果我们现在运行我们的客户端，那么我们将看到这里显示的 XML：

```py
**$ python3.4 s3_client.py create_bucket mybucket.example.com**
**<?xml version="1.0" ?>**
**<CreateBucketConfiguration >**
 **<LocationConstraint>eu-west-1</LocationConstraint>**
**</CreateBucketConfiguration>**

```

这与文档中指定的格式相匹配。您可以看到我们使用`ns`变量来填充`xmlns`属性。这个属性在整个 S3 XML 中都会出现，预定义`ns`变量使得更快地处理它。

现在，让我们添加代码来发出请求。将`create_bucket()`末尾的`xml_pprint(data)`替换为以下内容：

```py
    url = 'http://{}.{}'.format(bucket, endpoint)
    r = requests.put(url, data=data, auth=auth)
    if r.ok:
        print('Created bucket {} OK'.format(bucket))
    else:
        xml_pprint(r.text)
```

这里显示的第一行将从我们的存储桶名称和端点生成完整的 URL。第二行将向 S3 API 发出请求。请注意，我们使用`requests.put()`函数使用 HTTP `PUT`方法进行此请求，而不是使用`requests.get()`方法或`requests.post()`方法。还要注意，我们已经提供了我们的`auth`对象给调用。这将允许`Requests`为我们处理所有 S3 身份验证！

如果一切顺利，我们将打印出一条消息。如果一切不如预期，我们将打印出响应正文。S3 将错误消息作为 XML 返回到响应正文中。因此，我们使用我们的`xml_pprint()`函数来显示它。稍后我们将在*处理错误*部分讨论处理这些错误。

现在运行客户端，如果一切正常，那么我们将收到确认消息。确保您选择的存储桶尚未创建：

```py
**$ python3.4 s3_client.py create_bucket mybucket.example.com**
**Created bucket mybucket.example.com OK**

```

当我们在浏览器中刷新 S3 控制台时，我们将看到我们的存储桶已创建。

### 上传文件

现在我们已经创建了一个存储桶，我们可以上传一些文件。编写一个上传文件的函数类似于创建一个存储桶。我们查看文档以了解如何构建我们的 HTTP 请求，找出应该在命令行收集哪些信息，然后编写函数。

我们需要再次使用 HTTP `PUT`。我们需要存储文件的存储桶名称以及我们希望文件在 S3 中存储的名称。请求的正文将包含文件数据。在命令行中，我们将收集存储桶名称，我们希望文件在 S3 服务中存储的名称以及要上传的本地文件的名称。

在`create_bucket()`函数之后将以下函数添加到您的`s3_client.py`文件中：

```py
def upload_file(bucket, s3_name, local_path):
    data = open(local_path, 'rb').read()
    url = 'http://{}.{}/{}'.format(bucket, endpoint, s3_name)
    r = requests.put(url, data=data, auth=auth)
if r.ok:
        print('Uploaded {} OK'.format(local_path))
    else:
        xml_pprint(r.text)
```

在创建此函数时，我们遵循了与创建存储桶类似的模式：

1.  准备要放入请求正文中的数据。

1.  构建我们的 URL。

1.  发出请求。

1.  检查结果。

请注意，我们以二进制模式打开本地文件。文件可以包含任何类型的数据，因此我们不希望应用文本转换。我们可以从任何地方获取这些数据，例如数据库或另一个 Web API。在这里，我们只是简单地使用本地文件。

URL 与我们在`create_bucket()`中构建的端点相同，并且 S3 对象名称附加到 URL 路径。稍后，我们可以使用此 URL 检索对象。

现在，运行这里显示的命令来上传一个文件：

```py
**$ python3.4 s3_client.py mybucket.example.com test.jpg ~/test.jpg**
**Uploaded ~/test.jpg OK**

```

您需要将`mybucket.example.com`替换为您自己的存储桶名称。一旦文件上传完成，您将在 S3 控制台中看到它。

我使用了一个存储在我的主目录中的 JPEG 图像作为源文件。您可以使用任何文件，只需将最后一个参数更改为适当的路径。但是，使用 JPEG 图像将使您更容易重现以下部分。

### 通过 Web 浏览器检索已上传的文件

默认情况下，S3 对存储桶和对象应用限制权限。创建它们的帐户具有完全的读写权限，但对于其他人完全拒绝访问。这意味着我们刚刚上传的文件只有在下载请求包括我们帐户的身份验证时才能下载。如果我们在浏览器中尝试结果 URL，那么我们将收到访问被拒绝的错误。如果我们试图使用 S3 与其他人共享文件，这并不是很有用。

解决此问题的方法是使用 S3 的一种机制来更改权限。让我们看看使我们上传的文件公开的简单任务。将`upload_file()`更改为以下内容：

```py
def upload_file(bucket, s3_name, local_path, acl='private'):
    data = open(local_path, 'rb').read()
    url = 'http://{}.{}/{}'.format(bucket, endpoint, s3_name)
    headers = {'x-amz-acl': acl}
    r = requests.put(url, data=data, headers=headers, auth=auth)
if r.ok:
        print('Uploaded {} OK'.format(local_path))
    else:
        xml_pprint(r.text)
```

我们现在在我们的 HTTP 请求中包含了一个头部，`x-amz-acl`，它指定了要应用于对象的权限集。我们还在函数签名中添加了一个新的参数，这样我们就可以在命令行上指定权限集。我们使用了 S3 提供的所谓的**预设** **ACLs**（**预设** **访问控制列表**），并在[`docs.aws.amazon.com/AmazonS3/latest/dev/acl-overview.html#canned-acl`](http://docs.aws.amazon.com/AmazonS3/latest/dev/acl-overview.html#canned-acl)中进行了记录。

我们感兴趣的 ACL 称为`public-read`。这将允许任何人下载文件而无需任何形式的身份验证。现在，我们可以重新运行我们的上传，但这次会将这个 ACL 应用到它上面：

```py
**$ python3.4 s3_client.py mybucket.example.com test.jpg ~/test.jpg public-read**
**Uploaded test.jpg OK**

```

现在，在浏览器中访问文件的 S3 URL 将给我们下载文件的选项。

### 在 Web 浏览器中显示上传的文件

如果你上传了一张图片，那么你可能会想知道为什么浏览器要求我们保存它而不是直接显示它。原因是我们没有设置文件的`Content-Type`。

如果你还记得上一章，HTTP 响应中的`Content-Type`头部告诉客户端，这里是我们的浏览器，正文中的文件类型。默认情况下，S3 应用`binary/octet-stream`的内容类型。由于这个`Content-Type`，浏览器无法知道它正在下载一个图像，所以它只是将它呈现为一个可以保存的文件。我们可以通过在上传请求中提供`Content-Type`头部来解决这个问题。S3 将存储我们指定的类型，并在随后的下载响应中使用它作为`Content-Type`。

在`s3_client.py`的开头添加以下代码块到导入中：

```py
import mimetypes
```

然后将`upload_file()`更改为以下内容：

```py
def upload_file(bucket, s3_name, local_path, acl='private'):
    data = open(local_path, 'rb').read()
    url = 'http://{}.{}/{}'.format(bucket, endpoint, s3_name)
    headers = {'x-amz-acl': acl}
    mimetype = mimetypes.guess_type(local_path)[0]
    if mimetype:
        headers['Content-Type'] = mimetype
    r = requests.put(url, data=data, headers=headers, auth=auth)
if r.ok:
        print('Uploaded {} OK'.format(local_path))
    else:
        xml_pprint(r.text)
```

在这里，我们使用了`mimetypes`模块来猜测一个适合的`Content-Type`，通过查看`local_path`的文件扩展名。如果`mimetypes`无法从`local_path`确定`Content-Type`，那么我们就不包括`Content-Type`头部，让 S3 应用默认的`binary/octet-stream`类型。

不幸的是，在 S3 中，我们无法通过简单的`PUT`请求覆盖现有对象的元数据。可以通过使用`PUT`复制请求来实现，但这超出了本章的范围。现在，最好的方法是在上传文件之前使用 AWS 控制台从 S3 中删除文件。我们只需要做一次。现在，我们的代码将自动为我们上传的任何新文件添加`Content-Type`。

一旦你删除了文件，就像上一节所示重新运行客户端，也就是说，用新的`Content-Type`上传文件并尝试在浏览器中再次下载文件。如果一切顺利，那么图像将被显示。

### 使用 API 下载文件

通过 S3 API 下载文件与上传文件类似。我们只需要再次提供存储桶名称、S3 对象名称和本地文件名，但是发出一个`GET`请求而不是`PUT`请求，然后将接收到的数据写入磁盘。

在你的程序中添加以下函数，放在`upload_file()`函数下面：

```py
def download_file(bucket, s3_name, local_path):
    url = 'http://{}.{}/{}'.format(bucket, endpoint, s3_name)
    r = requests.get(url, auth=auth)
    if r.ok:
        open(local_path, 'wb').write(r.content)
        print('Downloaded {} OK'.format(s3_name))
    else:
        xml_pprint(r.text)
```

现在，运行客户端并下载一个文件，你之前上传的文件，使用以下命令：

```py
**$ python3.4 s3_client.py download_file mybucket.example.com test.jpg ~/test_downloaded.jpg**
**Downloaded test.jpg OK**

```

## 解析 XML 和处理错误

如果在运行上述代码时遇到任何错误，那么你会注意到清晰的错误消息不会被显示。S3 将错误消息嵌入到响应体中返回的 XML 中，直到现在我们只是将原始 XML 转储到屏幕上。我们可以改进这一点，并从 XML 中提取文本。首先，让我们生成一个错误消息，这样我们就可以看到 XML 的样子。在`s3_client.py`中，将你的访问密钥替换为空字符串，如下所示：

```py
access_secret = ''
```

现在，尝试在服务上执行以下操作：

```py
**$ python3.4 s3_client.py create_bucket failbucket.example.com**
**<?xml version="1.0" ?>**
**<Error>**
 **<Code>SignatureDoesNotMatch</Code>**
 **<Message>The request signature we calculated does not match the signature you provided. Check your key and signing method.</Message>**
 **<AWSAccessKeyId>AKIAJY5II3SZNHZ25SUA</AWSAccessKeyId>**
 **<StringToSign>AWS4-HMAC-SHA256...</StringToSign>**
 **<SignatureProvided>e43e2130...</SignatureProvided>**
 **<StringToSignBytes>41 57 53 34...</StringToSignBytes>**
 **<CanonicalRequest>PUT...</CanonicalRequest>**
 **<CanonicalRequestBytes>50 55 54...</CanonicalRequestBytes>**
 **<RequestId>86F25A39912FC628</RequestId>**
 **<HostId>kYIZnLclzIW6CmsGA....</HostId>**
**</Error>**

```

前面的 XML 是 S3 错误信息。我已经截断了几个字段以便在这里显示。你的代码块会比这个稍微长一点。在这种情况下，它告诉我们它无法验证我们的请求，这是因为我们设置了一个空的访问密钥。

### 解析 XML

打印所有的 XML 对于错误消息来说太多了。有很多无用的额外信息对我们来说没有用。最好的办法是只提取错误消息的有用部分并显示出来。

嗯，`ElementTree`为我们从 XML 中提取这样的信息提供了一些强大的工具。我们将回到 XML 一段时间，来探索这些工具。

首先，我们需要打开一个交互式的 Python shell，然后使用以下命令再次生成上述错误消息：

```py
**>>> import requests**
**>>> import requests_aws4auth**
**>>> auth = requests_aws4auth.AWS4Auth('<ID>', '', 'eu-west-1', '')**
**>>> r = requests.get('http://s3.eu-west-1.amazonaws.com', auth=auth)**

```

你需要用你的 AWS 访问 ID 替换`<ID>`。打印出`r.text`以确保你得到一个错误消息，类似于我们之前生成的那个。

现在，我们可以探索我们的 XML。将 XML 文本转换为`ElementTree`树。一个方便的函数是：

```py
**>>> import xml.etree.ElementTree as ET**
**>>> root = ET.fromstring(r.text)**

```

现在我们有了一个 ElementTree 实例，`root`作为根元素。

### 查找元素

通过使用元素作为迭代器来浏览树的最简单方法。尝试做以下事情：

```py
**>>> for element in root:**
**...     print('Tag: ' + element.tag)**
**Tag: Code**
**Tag: Message**
**Tag: AWSAccessKeyId**
**Tag: StringToSign**
**Tag: SignatureProvided**
**...**

```

迭代`root`会返回它的每个子元素，然后我们通过使用`tag`属性打印出元素的标签。

我们可以使用以下命令对我们迭代的标签应用过滤器：

```py
**>>> for element in root.findall('Message'):**
**...     print(element.tag + ': ' + element.text)**
**Message: The request signature we calculated does not match the signature you provided. Check your key and signing method.**

```

在这里，我们使用了`root`元素的`findall()`方法。这个方法将为我们提供与指定标签匹配的`root`元素的所有直接子元素的列表，在这种情况下是`<Message>`。

这将解决我们只提取错误消息文本的问题。现在，让我们更新我们的错误处理。

### 处理错误

我们可以回去并将这添加到我们的`s3_client.py`文件中，但让我们在输出中包含更多信息，并结构化代码以允许重用。将以下函数添加到`download_file()`函数下面的文件中：

```py
def handle_error(response):
    output = 'Status code: {}\n'.format(response.status_code)
    root = ET.fromstring(response.text)
    code =  root.find('Code').text
    output += 'Error code: {}\n'.format(code)
    message = root.find('Message').text
    output += 'Message: {}\n'.format(message)
    print(output)
```

你会注意到我们在这里使用了一个新的函数，即`root.find()`。这与`findall()`的工作方式相同，只是它只返回第一个匹配的元素，而不是所有匹配的元素列表。

然后，用`handle_error(r)`替换文件中每个`xml_pprint(r.text)`的实例，然后再次使用错误的访问密钥运行客户端。现在，你会看到一个更详细的错误消息：

```py
**$ python3.4 s3_client.py create_bucket failbucket.example.com**
**Status code: 403**
**Error code: SignatureDoesNotMatch**
**Message: The request signature we calculated does not match the signature you provided. Check your key and signing method.**

```

## 进一步的增强

这就是我们要为客户提供的服务。我们编写了一个命令行程序，可以在 Amazon S3 服务上执行创建存储桶、上传和下载对象等基本操作。还有很多操作可以实现，这些可以在 S3 文档中找到；例如列出存储桶内容、删除对象和复制对象等操作。

我们可以改进一些其他东西，特别是如果我们要将其制作成一个生产应用程序。命令行解析机制虽然紧凑，但从安全角度来看并不令人满意，因为任何有权访问命令行的人都可以运行任何内置的 python 命令。最好是有一个函数白名单，并使用标准库模块之一，如`argparse`来实现一个适当的命令行解析器。

将访问 ID 和访问密钥存储在源代码中也是安全问题。由于密码存储在源代码中，然后上传到云代码仓库，发生了几起严重的安全事件。最好在运行时从外部来源加载密钥，比如文件或数据库。

## Boto 包

我们已经讨论了直接使用 S3 REST API，并且这给了我们一些有用的技术，让我们能够在将来编写类似 API 时进行编程。在许多情况下，这将是我们与 Web API 交互的唯一方式。

然而，一些 API，包括 AWS，有现成的包可以暴露服务的功能，而无需处理 HTTP API 的复杂性。这些包通常使代码更清晰、更简单，如果可用的话，应该优先用于生产工作。

AWS 包被称为**Boto**。我们将快速浏览一下`Boto`包，看看它如何提供我们之前编写的一些功能。

`boto`包在 PyPi 中可用，所以我们可以用`pip`安装它：

```py
**$ pip install boto**
**Downloading/unpacking boto**
**...**

```

现在，启动一个 Python shell，让我们试一试。我们需要先连接到服务：

```py
**>>> import boto**
**>>> conn = boto.connect_s3('<ACCESS ID>', '<ACCESS SECRET>')**

```

您需要用您的访问 ID 和访问密钥替换`<ACCESS ID>`和`<ACCESS SECRET>`。现在，让我们创建一个存储桶：

```py
**>>> conn.create_bucket('mybucket.example.com')**

```

这将在默认的标准美国地区创建存储桶。我们可以提供不同的地区，如下所示：

```py
**>>> from boto.s3.connection import Location**
**>>> conn.create_bucket('mybucket.example.com', location=Location.EU)**

```

我们需要使用不同的区域名称来执行此功能，这些名称与我们之前创建存储桶时使用的名称不同。要查看可接受的区域名称列表，请执行以下操作：

```py
**>>> [x for x in dir(Location) if x.isalnum()]**
**['APNortheast', 'APSoutheast', 'APSoutheast2', 'CNNorth1', 'DEFAULT', 'EU', 'SAEast', 'USWest', 'USWest2']**

```

执行以下操作以显示我们拥有的存储桶列表：

```py
**>>> buckets = conn.get_all_buckets()**
**>>> [b.name for b in buckets]**
**['mybucket.example.com', 'mybucket2.example.com']**

```

我们还可以列出存储桶的内容。为此，首先我们需要获取对它的引用：

```py
**>>> bucket = conn.get_bucket('mybucket.example.com')**

```

然后列出内容：

```py
**>>> [k.name for k in bucket.list()]**
**['cheesehop.txt', 'parrot.txt']**

```

上传文件是一个简单的过程。首先，我们需要获取要放入的存储桶的引用，然后我们需要创建一个`Key`对象，它将代表我们在存储桶中的对象：

```py
**>>> bucket = conn.get_bucket('mybucket.example.com')**
**>>> from boto.s3.key import Key**
**>>> key = Key(bucket)**

```

接下来，我们需要设置`Key`名称，然后上传我们的文件数据：

```py
**>>> key.key = 'lumberjack_song.txt'**
**>>> key.set_contents_from_filename('~/lumberjack_song.txt')**

```

`boto`包在上传文件时会自动设置`Content-Type`，它使用了我们之前用于确定类型的`mimetypes`模块。

下载也遵循类似的模式。尝试以下命令：

```py
**>>> bucket = conn.get_bucket('mybucket.example.com')**
**>>> key = bucket.get_key('parrot.txt')**
**>>> key.get_contents_to_filename('~/parrot.txt')**

```

这将下载`mybucket.example.com`存储桶中的`parrot.txt` S3 对象，然后将其存储在`~/parrot.txt`本地文件中。

一旦我们有了对`Key`的引用，只需使用以下内容来设置 ACL：

```py
**>>> key.set_acl('public-read')**

```

我将让您通过教程进一步探索`boto`包的功能，该教程可以在[`boto.readthedocs.org/en/latest/s3_tut.html`](https://boto.readthedocs.org/en/latest/s3_tut.html)找到。

显然，对于 Python 中的日常 S3 工作，`boto`应该是您的首选包。

## 结束 S3

因此，我们已经讨论了 Amazon S3 API 的一些用途，并学到了一些关于在 Python 中使用 XML 的知识。这些技能应该让您在使用任何基于 XML 的 REST API 时有一个良好的开端，无论它是否有像`boto`这样的预构建库。

然而，XML 并不是 Web API 使用的唯一数据格式，S3 处理 HTTP 的方式也不是 Web API 使用的唯一模型。因此，我们将继续并看一看今天使用的另一种主要数据格式，JSON 和另一个 API：Twitter。

# JSON

**JavaScript 对象表示法（JSON）**是一种用文本字符串表示简单对象（如`列表`和`字典`）的标准方式。尽管最初是为 JavaScript 开发的，但 JSON 是与语言无关的，大多数语言都可以使用它。它轻巧，但足够灵活，可以处理广泛的数据范围。这使得它非常适合在 HTTP 上传数据，许多 Web API 使用它作为其主要数据格式。

## 编码和解码

我们使用`json`模块来处理 Python 中的 JSON。通过以下命令，让我们创建一个 Python 列表的 JSON 表示：

```py
**>>> import json**
**>>> l = ['a', 'b', 'c']**
**>>> json.dumps(l)**
**'["a", "b", "c"]'**

```

我们使用`json.dumps()`函数将对象转换为 JSON 字符串。在这种情况下，我们可以看到 JSON 字符串似乎与 Python 对列表的表示相同，但请注意这是一个字符串。通过以下操作确认：

```py
**>>> s = json.dumps(['a', 'b', 'c'])**
**>>> type(s)**
**<class 'str'>**
**>>> s[0]**
**'['**

```

将 JSON 转换为 Python 对象也很简单，如下所示：

```py
**>>> s = '["a", "b", "c"]'**
**>>> l = json.loads(s)**
**>>> l**
**['a', 'b', 'c']**
**>>> l[0]**
**'a'**

```

我们使用`json.loads()`函数，只需传递一个 JSON 字符串。正如我们将看到的，这在与 Web API 交互时非常强大。通常，我们将收到一个 JSON 字符串作为 HTTP 响应的主体，只需使用`json.loads()`进行解码，即可提供可立即使用的 Python 对象。

## 使用 JSON 的字典

JSON 本身支持映射类型对象，相当于 Python 的`dict`。这意味着我们可以直接通过 JSON 使用`dicts`。

```py
**>>> json.dumps({'A':'Arthur', 'B':'Brian', 'C':'Colonel'})**
**'{"A": "Arthur", "C": "Colonel", "B": "Brian"}'**

```

此外，了解 JSON 如何处理嵌套对象也是有用的。

```py
**>>> d = {**
**...     'Chapman': ['King Arthur', 'Brian'],**
**...     'Cleese': ['Sir Lancelot', 'The Black Knight'],**
**...     'Idle': ['Sir Robin', 'Loretta'],**
**... }**
**>>> json.dumps(d)**
**'{"Chapman": ["King Arthur", "Brian"], "Idle": ["Sir Robin", "Loretta"], "Cleese": ["Sir Lancelot", "The Black Knight"]}'**

```

不过有一个需要注意的地方：JSON 字典键只能是字符串形式。

```py
**>>> json.dumps({1:10, 2:20, 3:30})**
**'{"1": 10, "2": 20, "3": 30}'**

```

注意，JSON 字典中的键如何成为整数的字符串表示？要解码使用数字键的 JSON 字典，如果我们想将它们作为数字处理，我们需要手动进行类型转换。执行以下操作来实现这一点：

```py
**>>> j = json.dumps({1:10, 2:20, 3:30})**
**>>> d_raw = json.loads(j)**
**>>> d_raw**
**{'1': 10, '2': 20, '3': 30}**
**>>> {int(key):val for key,val in d_raw.items()}**
**{1: 10, 2: 20, 3: 30}**

```

我们只需使用字典推导将`int()`应用于字典的键。

## 其他对象类型

JSON 只能干净地处理 Python 的`lists`和`dicts`，对于其他对象类型，`json`可能会尝试将对象类型转换为其中一个，或者完全失败。尝试一个元组，如下所示：

```py
**>>> json.dumps(('a', 'b', 'c'))**
**'["a", "b", "c"]'**

```

JSON 没有元组数据类型，因此`json`模块将其转换为`list`。如果我们将其转换回：

```py
**>>> j = json.dumps(('a', 'b', 'c'))**
**>>> json.loads(j)**
**['a', 'b', 'c']**

```

它仍然是一个`list`。`json`模块不支持`sets`，因此它们也需要重新转换为`lists`。尝试以下命令：

```py
**>>> s = set(['a', 'b', 'c'])**
**>>> json.dumps(s)**
**...**
**TypeError: {'a', 'c', 'b'} is not JSON serializable**
**>>> json.dumps(list(s))**
**'["a", "b", "c"]'**

```

这将导致类似于元组引起的问题。如果我们将 JSON 转换回 Python 对象，那么它将是一个`list`而不是`set`。

我们几乎从不遇到需要这些专门的 Python 对象的 Web API，如果我们确实遇到，那么 API 应该提供一些处理它的约定。但是，如果我们将数据存储在除`lists`或`dicts`之外的任何格式中，我们需要跟踪我们需要应用于传出或传入对象的任何转换。

现在我们对 JSON 有了一定的了解，让我们看看它在 Web API 中是如何工作的。

# Twitter API

Twitter API 提供了访问我们可能希望 Twitter 客户端执行的所有功能。使用 Twitter API，我们可以创建搜索最新推文、查找趋势、查找用户详细信息、关注用户时间线，甚至代表用户发布推文和直接消息的客户端。

我们将查看 Twitter API 版本 1.1，这是撰写本章时的当前版本。

### 注意

Twitter 为其 API 提供了全面的文档，可以在[`dev.twitter.com/overview/documentation`](https://dev.twitter.com/overview/documentation)找到。

## 一个 Twitter 世界时钟

为了说明 Twitter API 的一些功能，我们将编写一个简单的 Twitter 世界时钟的代码。我们的应用程序将定期轮询其 Twitter 账户，寻找包含可识别城市名称的提及，如果找到，则会回复推文并显示该城市的当前当地时间。在 Twitter 中，提及是指包含我们账户名前缀`@`的任何推文，例如`@myaccount`。

## Twitter 的身份验证

与 S3 类似，我们需要确定在开始之前如何管理身份验证。我们需要注册，然后了解 Twitter 希望我们如何对请求进行身份验证。

### 为 Twitter API 注册您的应用程序

我们需要创建一个 Twitter 账户，注册我们的应用程序，并且我们将收到我们应用程序的身份验证凭据。另外，建立一个第二个账户也是一个好主意，我们可以用它来向应用程序账户发送测试推文。这提供了一种更干净的方式来检查应用程序是否正常工作，而不是让应用程序账户向自己发送推文。您可以创建的 Twitter 账户数量没有限制。

要创建帐户，请转到[`www.twitter.com`](http://www.twitter.com)并完成注册过程。一旦您拥有 Twitter 帐户，执行以下操作注册您的应用程序：

1.  使用您的主要 Twitter 帐户登录[`apps.twitter.com`](http://apps.twitter.com)，然后创建一个新应用程序。

1.  填写新应用程序表格，注意 Twitter 应用程序名称需要在全球范围内是唯一的。

1.  转到应用程序设置，然后更改应用程序权限以具有读写访问权限。您可能需要注册您的手机号码以启用此功能。即使您不愿意提供这个信息，我们也可以创建完整的应用程序；但是，最终发送回复推文的最终功能将不会激活。

现在我们需要获取我们的访问凭证，如下所示：

1.  转到**Keys and Access Tokens**部分，然后记下**Consumer Key**和**Access Secret**。

1.  生成一个**访问令牌**。

1.  记下**访问令牌**和**访问密钥**。

### 认证请求

我们现在有足够的信息来进行请求认证。Twitter 使用一个称为**oAuth**的认证标准，版本 1.0a。详细描述在[`oauth.net/core/1.0a/`](http://oauth.net/core/1.0a/)。

oAuth 认证标准有点棘手，但幸运的是，`Requests`模块有一个名为`requests-oauthlib`的伴侣库，它可以为我们处理大部分复杂性。这在 PyPi 上可用，因此我们可以使用`pip`下载和安装它。

```py
**$ pip install requests-oauthlib**
**Downloading/unpacking requests-oauthlib**
**...**

```

现在，我们可以为我们的请求添加认证，然后编写我们的应用程序。

### 一个 Twitter 客户端

将此处提到的代码保存到文件中，并将其保存为`twitter_worldclock.py`。您需要用从上述 Twitter 应用程序配置中获取的值替换`<CONSUMER_KEY>`，`<CONSUMER_SECRET>`，`<ACCESS_TOKEN>`和`<ACCESS_SECRET>`：

```py
import requests, requests_oauthlib, sys

consumer_key = '<CONSUMER_KEY>'
consumer_secret = '<CONSUMER_SECRET>'
access_token = '<ACCESS_TOKEN>'
access_secret = '<ACCESS_KEY>'

def init_auth():
    auth_obj = requests_oauthlib.OAuth1(
                    consumer_key, consumer_secret,
                    access_token, access_secret)

    if verify_credentials(auth_obj):
        print('Validated credentials OK')
        return auth_obj
    else:
        print('Credentials validation failed')
        sys.exit(1)	

def verify_credentials(auth_obj):
    url = 'https://api.twitter.com/1.1/' \
          'account/verify_credentials.json'
    response = requests.get(url, auth=auth_obj)
    return response.status_code == 200

if __name__ == '__main__':
    auth_obj = init_auth()
```

请记住，`consumer_secret`和`access_secret`充当您的 Twitter 帐户的密码，因此在生产应用程序中，它们应该从安全的外部位置加载，而不是硬编码到源代码中。

在上述代码中，我们通过使用我们的访问凭证在`init_auth()`函数中创建`OAuth1`认证实例`auth_obj`。每当我们需要发出 HTTP 请求时，我们将其传递给`Requests`，通过它`Requests`处理认证。您可以在`verify_credentials()`函数中看到这个例子。

在`verify_credentials()`函数中，我们测试 Twitter 是否识别我们的凭据。我们在这里使用的 URL 是 Twitter 专门用于测试我们的凭据是否有效的终点。如果它们有效，则返回 HTTP 200 状态代码，否则返回 401 状态代码。

现在，让我们运行`twitter_worldclock.py`，如果我们已经注册了我们的应用程序并正确填写了令牌和密钥，那么我们应该会看到`验证凭据 OK`。现在认证已经工作，我们程序的基本流程将如下图所示：

![Twitter 客户端](img/6008OS_03_02.jpg)

我们的程序将作为守护程序运行，定期轮询 Twitter，查看是否有任何新的推文需要我们处理和回复。当我们轮询提及时间线时，我们将下载自上次轮询以来接收到的任何新推文，以便我们可以处理所有这些推文而无需再次轮询。

## 轮询推文

让我们添加一个函数来检查并从我们的提及时间线中检索新推文。在我们添加循环之前，我们将使其工作。在`verify_credentials()`下面添加新函数，然后在主部分中添加对此函数的调用；同时，在文件开头的导入列表中添加`json`：

```py
def get_mentions(since_id, auth_obj):
    params = {'count': 200, 'since_id': since_id,
              'include_rts':  0, 'include_entities': 'false'}
    url = 'https://api.twitter.com/1.1/' \
          'statuses/mentions_timeline.json'
    response = requests.get(url, params=params, auth=auth_obj)
    response.raise_for_status()
    return json.loads(response.text)

if __name__ == '__main__':
    auth_obj = init_auth()
    since_id = 1
    for tweet in get_mentions(since_id, auth_obj):
        print(tweet['text'])
```

使用`get_mentions()`，我们通过连接到`statuses/mentions_timeline.json`端点来检查并下载提及我们应用账户的任何推文。我们提供了一些参数，`Requests`将其作为查询字符串传递。这些参数由 Twitter 指定，它们控制推文将如何返回给我们。它们如下：

+   `'count'`：这指定将返回的最大推文数。Twitter 将允许通过单个请求接收 200 条推文。

+   `'include_entities'`：这用于从检索到的推文中删除一些多余的信息。

+   `'include_rts'`：这告诉 Twitter 不要包括任何转发。如果有人转发我们的回复，我们不希望用户收到另一个时间更新。

+   `'since_id'`：这告诉 Twitter 只返回 ID 大于此值的推文。每条推文都有一个唯一的 64 位整数 ID，后来的推文比先前的推文具有更高的值 ID。通过记住我们处理的最后一条推文的 ID，然后将其作为此参数传递，Twitter 将过滤掉我们已经看过的推文。

在运行上述操作之前，我们希望为我们的账户生成一些提及，这样我们就有东西可以下载。登录您的 Twitter 测试账户，然后创建一些包含`@username`的推文，其中您将`username`替换为您的应用账户用户名。之后，当您进入应用账户的**通知**选项卡的**提及**部分时，您将看到这些推文。

现在，如果我们运行上述代码，我们将在屏幕上打印出我们提及的文本。

## 处理推文

下一步是解析我们的提及，然后生成我们想要包含在回复中的时间。解析是一个简单的过程。在这里，我们只需检查推文的“text”值，但生成时间需要更多的工作。实际上，为此，我们需要一个城市及其时区的数据库。这在`pytz`包中可用，在 PyPi 上可以找到。为此，请安装以下包：

```py
**$ pip install pytz**
**Downloading/unpacking pytz**
**...**

```

然后，我们可以编写我们的推文处理函数。将此函数添加到`get_mentions()`下方，然后在文件开头的导入列表中添加`datetime`和`pytz`：

```py
def process_tweet(tweet):
    username = tweet['user']['screen_name']
    text = tweet['text']
    words = [x for x in text.split() if
                        x[0] not in ['@', '#']]
    place = ' '.join(words)
    check = place.replace(' ', '_').lower()
    found = False
    for tz in pytz.common_timezones:
        tz_low = tz.lower()
        if check in tz_low.split('/'):
            found = True
            break
    if found:
        timezone = pytz.timezone(tz)
        time = datetime.datetime.now(timezone).strftime('%H:%M')
        reply = '@{} The time in {} is currently {}'.format(username, place, time)
    else:
        reply = "@{} Sorry, I didn't recognize " \
                        "'{}' as a city".format(username, place)
    print(reply)

if __name__ == '__main__':
    auth_obj = init_auth()
    since_id = 1
    for tweet in get_mentions(since_id, auth_obj):
        process_tweet(tweet)
```

`process_tweet()`的大部分内容用于格式化推文文本和处理时区数据。首先，我们将从推文中删除任何`@username`提及和`#hashtags`。然后，我们准备剩下的推文文本与时区名称数据库进行比较。时区名称数据库存储在`pytz.common_timezones`中，但名称中还包含地区，用斜杠(`/`)与名称分隔。此外，在这些名称中，下划线用于代替空格。

我们通过扫描数据库来检查格式化的推文文本。如果找到匹配项，我们将构建一个包含匹配时区的当地时间的回复。为此，我们使用`datetime`模块以及由`pytz`生成的时区对象。如果在时区数据库中找不到匹配项，我们将组成一个回复，让用户知道这一点。然后，我们将我们的回复打印到屏幕上，以检查它是否按预期工作。

同样，在运行此操作之前，我们可能希望创建一些只包含城市名称并提及我们的世界时钟应用账户的推文，以便函数有东西可以处理。在时区数据库中出现的一些城市包括都柏林、纽约和东京。

试一试！当您运行它时，您将在屏幕上得到一些包含这些城市和这些城市当前当地时间的推文回复文本。

## 速率限制

如果我们多次运行上述操作，然后我们会发现它在一段时间后会停止工作。要么凭据暂时无法验证，要么`get_mentions()`中的 HTTP 请求将失败。

这是因为 Twitter 对其 API 应用速率限制，这意味着我们的应用程序只允许在一定时间内对端点进行一定数量的请求。限制在 Twitter 文档中列出，根据认证路线（稍后讨论）和端点的不同而有所不同。我们使用`statuses/mentions_timeline.json`，因此我们的限制是每 15 分钟 15 次请求。如果我们超过这个限制，那么 Twitter 将以`429` `Too many requests`状态代码做出响应。这将迫使我们等待下一个 15 分钟窗口开始之前，才能让我们获得任何有用的数据。

速率限制是 Web API 的常见特征，因此在使用它们时，有一些有效的测试方法是很有用的。使用速率限制的 API 数据进行测试的一种方法是下载一些数据，然后将其存储在本地。之后，从文件中加载它，而不是从 API 中拉取它。通过使用 Python 解释器下载一些测试数据，如下所示：

```py
**>>> from twitter_worldclock import ***
**>>> auth_obj = init_auth()**
**Credentials validated OK**
**>>> mentions = get_mentions(1, auth_obj)**
**>>> json.dump(mentions, open('test_mentions.json', 'w'))**

```

当您运行此时，您需要在与`twitter_worldclock.py`相同的文件夹中。这将创建一个名为`test_mentions.json`的文件，其中包含我们的 JSON 化提及。在这里，`json.dump()`函数将提供的数据写入文件，而不是将其作为字符串返回。

我们可以通过修改程序的主要部分来使用这些数据，看起来像下面这样：

```py
if __name__ == '__main__':
    mentions = json.load(open('test_mentions.json'))
    for tweet in mentions:
        process_tweet(tweet)
```

## 发送回复

我们需要执行的最后一个函数是对提及进行回复。为此，我们使用`statuses/update.json`端点。如果您尚未在应用帐户中注册您的手机号码，则这将无法工作。因此，只需将程序保持原样。如果您已经注册了手机号码，则在`process_tweets()`下添加此功能：

```py
def post_reply(reply_to_id, text, auth_obj):
    params = {
        'status': text,
        'in_reply_to_status_id': reply_to_id}
    url = 'https://api.twitter.com/1.1./statuses/update.json'
    response = requests.post(url, params=params, auth=auth_obj)
    response.raise_for_status()
```

并在`process_tweet()`末尾的`print()`调用下面，与相同的缩进级别：

```py
post_reply(tweet['id'], reply, auth_obj)
```

现在，如果您运行此程序，然后检查您的测试帐户的 Twitter 通知，您将看到一些回复。

`post_reply()`函数只是使用以下参数调用端点，通知 Twitter 要发布什么：

+   `status`：这是我们回复推文的文本。

+   `in_reply_to_status_id`：这是我们要回复的推文的 ID。我们提供这个信息，以便 Twitter 可以将推文链接为对话。

在测试时，我们可能会收到一些`403`状态代码响应。这没关系，只是 Twitter 拒绝让我们连续发布两条相同文本的推文，这可能会发生在这个设置中，具体取决于我们发送了什么测试推文。

## 最后的修饰

建筑模块已经就位，我们可以添加主循环使程序成为守护进程。在顶部导入`time`模块，然后将主要部分更改为以下内容：

```py
if __name__ == '__main__':
    auth_obj = init_auth()
    since_id = 1
    error_count = 0
    while error_count < 15:
        try:
            for tweet in get_mentions(since_id, auth_obj):
                process_tweet(tweet)
                since_id = max(since_id, tweet['id'])
            error_count =  0
        except requests.exceptions.HTTPError as e:
            print('Error: {}'.format(str(e)))
            error_count += 1
        time.sleep(60)
```

这将每 60 秒调用`get_mentions()`，然后处理已下载的任何新推文。如果出现任何 HTTP 错误，它将在退出程序之前重试 15 次。

现在，如果我们运行程序，它将持续运行，回复提及世界时钟应用帐户的推文。试一试，运行程序，然后从您的测试帐户发送一些推文。一分钟后，您将看到一些回复您的通知。

## 进一步进行

现在我们已经编写了一个基本的功能 Twitter API 客户端，肯定有一些可以改进的地方。虽然本章没有空间详细探讨增强功能，但值得提到一些，以便通知您可能想要承担的未来项目。

### 轮询和 Twitter 流 API

您可能已经注意到一个问题，即我们的客户端每次轮询最多只能拉取 200 条推文。在每次轮询中，Twitter 首先提供最近的推文。这意味着如果我们在 60 秒内收到超过 200 条推文，那么我们将永久丢失最先收到的推文。实际上，使用`statuses/mentions_timeline.json`端点没有完整的解决方案。

Twitter 针对这个问题的解决方案是提供一种另类的 API，称为**流式 API**。连接到这些 API 时，HTTP 响应连接实际上是保持打开状态的，并且传入的推文会不断通过它进行流式传输。`Requests`包提供了处理这种情况的便捷功能。`Requests`响应对象具有`iter_lines()`方法，可以无限运行。它能够在服务器发送数据时输出一行数据，然后我们可以对其进行处理。如果您发现您需要这个功能，那么在 Requests 文档中有一个示例可以帮助您入门，可以在[`docs.python-requests.org/en/latest/user/advanced/#streaming-requests`](http://docs.python-requests.org/en/latest/user/advanced/#streaming-requests)找到。

## 替代 oAuth 流程

我们的设置是让我们的应用程序针对我们的主账户进行操作，并为发送测试推文使用第二个账户，这有点笨拙，特别是如果您将您的应用账户用于常规推文。有没有更好的办法，专门有一个账户来处理世界时钟的推文？

嗯，是的。理想的设置是在一个主账户上注册应用程序，并且您也可以将其用作常规 Twitter 账户，并且让应用程序处理第二个专用世界时钟账户的推文。

oAuth 使这成为可能，但需要一些额外的步骤才能使其正常工作。我们需要世界时钟账户来授权我们的应用代表其行事。您会注意到之前提到的 oAuth 凭据由两个主要元素组成，**消费者**和**访问**。消费者元素标识我们的应用程序，访问元素证明了访问凭据来自授权我们的应用代表其行事的账户。在我们的应用程序中，我们通过让应用程序代表注册时的账户，也就是我们的应用账户，来简化完整的账户授权过程。当我们这样做时，Twitter 允许我们直接从[dev.twitter.com](http://dev.twitter.com)界面获取访问凭据。要使用不同的用户账户，我们需要插入一个步骤，让用户转到 Twitter，这将在 Web 浏览器中打开，用户需要登录，然后明确授权我们的应用程序。

### 注意

这个过程在`requests-oauthlib`文档中有演示，可以在[`requests-oauthlib.readthedocs.org/en/latest/oauth1_workflow.html`](https://requests-oauthlib.readthedocs.org/en/latest/oauth1_workflow.html)找到。

# HTML 和屏幕抓取

尽管越来越多的服务通过 API 提供其数据，但当一个服务没有这样做时，以编程方式获取数据的唯一方法是下载其网页，然后解析 HTML 源代码。这种技术称为**屏幕抓取**。

虽然原则上听起来很简单，但屏幕抓取应该被视为最后的手段。与 XML 不同，XML 的语法严格执行，数据结构通常是相对稳定的，有时甚至有文档记录，而网页源代码的世界却是一个混乱的世界。这是一个不断变化的地方，代码可能会意外改变，以一种完全破坏你的脚本并迫使你从头开始重新设计解析逻辑的方式。

尽管如此，有时这是获取基本数据的唯一方法，因此我们将简要讨论开发一种抓取方法。我们将讨论在 HTML 代码发生变化时减少影响的方法。

在抓取之前，您应该始终检查网站的条款和条件。一些网站明确禁止自动解析和检索。违反条款可能导致您的 IP 地址被禁止。然而，在大多数情况下，只要您不重新发布数据并且不进行过于频繁的请求，您应该没问题。

## HTML 解析器

我们将解析 HTML 就像我们解析 XML 一样。我们再次可以选择拉取式 API 和面向对象的 API。我们将使用`ElementTree`，原因与之前提到的相同。

有几个可用的 HTML 解析库。它们的区别在于它们的速度、在 HTML 文档中导航的接口，以及它们处理糟糕构建的 HTML 的能力。Python 标准库不包括面向对象的 HTML 解析器。这方面普遍推荐的第三方包是`lxml`，它主要是一个 XML 解析器。但是，它确实包含一个非常好的 HTML 解析器。它快速，提供了几种浏览文档的方式，并且对破碎的 HTML 宽容。

`lxml`库可以通过`python-lxml`包在 Debian 和 Ubuntu 上安装。如果您需要一个最新版本，或者无法安装系统包，那么可以通过`pip`安装`lxml`。请注意，您需要一个构建环境。Debian 通常带有一个已经设置好的环境，但如果缺少，那么以下内容将为 Debian 和 Ubuntu 都安装一个：

```py
**$ sudo apt-get install build-essential**

```

然后你应该能够像这样安装`lxml`：

```py
**$ sudo STATIC_DEPS=true pip install lxml**

```

如果您在 64 位系统上遇到编译问题，那么您也可以尝试：

```py
**$ CFLAGS="$CFLAGS -fPIC" STATIC_DEPS=true pip install lxml**

```

在 Windows 上，可以从`lxml`网站[`lxml.de/installation.html`](http://lxml.de/installation.html)获取安装程序包。如果您的 Python 版本没有安装程序，可以在页面上查找第三方安装程序的链接。

如果`lxml`对您不起作用，下一个最好的库是 BeautifulSoup。BeautifulSoup 是纯 Python，因此可以使用`pip`安装，并且应该可以在任何地方运行。尽管它有自己的 API，但它是一个备受尊重和有能力的库，实际上它可以使用`lxml`作为后端库。

## 给我看数据

在开始解析 HTML 之前，我们需要解析的东西！让我们从 Debian 网站上获取最新稳定版 Debian 发行版的版本和代号。有关当前稳定版发行版的信息可以在[`www.debian.org/releases/stable/`](https://www.debian.org/releases/stable/)找到。

我们想要的信息显示在页面标题和第一句中：

![给我看数据](img/6008OS_03_03.jpg)

因此，我们应该提取*"jessie"*代号和 8.0 版本号。

## 使用 lxml 解析 HTML

让我们打开一个 Python shell 并开始解析。首先，我们将使用`Requests`下载页面。

```py
**>>> import requests**
**>>> response = requests.get('https://www.debian.org/releases/stable')**

```

接下来，我们将源代码解析成`ElementTree`树。这与使用标准库的`ElementTree`解析 XML 相同，只是这里我们将使用`lxml`专家`HTMLParser`。

```py
**>>> from lxml.etree import HTML**
**>>> root = HTML(response.content)**

```

`HTML()`函数是一个快捷方式，它读取传递给它的 HTML，然后生成一个 XML 树。请注意，我们传递的是`response.content`而不是`response.text`。`lxml`库在使用原始响应而不是解码的 Unicode 文本时会产生更好的结果。

`lxml`库的`ElementTree`实现已经被设计为与标准库的 100%兼容，因此我们可以像处理 XML 一样开始探索文档：

```py
**>>> [e.tag for e in root]**
**['head', 'body']**
**>>> root.find('head').find('title').text**
**'Debian –- Debian \u201cjessie\u201d Release Information'**

```

在上面的代码中，我们已经打印出了文档的`<title>`元素的文本内容，这是在上面截图的标签中显示的文本。我们已经看到它包含了我们想要的代号。

## 聚焦

屏幕抓取是一种寻找明确地址 HTML 元素的艺术，这些元素包含我们想要的信息，并且只从这些元素中提取信息。

然而，我们也希望选择标准尽可能简单。我们依赖文档的内容越少，页面的 HTML 发生变化时就越不容易破坏。

让我们检查页面的 HTML 源代码，看看我们正在处理什么。为此，可以在 Web 浏览器中使用`查看源代码`，或者将 HTML 保存到文件中并在文本编辑器中打开。本书的源代码下载中也包含了页面的源代码。搜索文本`Debian 8.0`，这样我们就可以直接找到我们想要的信息。对我来说，它看起来像以下代码块：

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

我跳过了`<body>`和`<div>`之间的 HTML，以显示`<div>`是`<body>`元素的直接子元素。从上面可以看出，我们想要`<div>`元素的`<p>`标签子元素的内容。

如果我们使用之前使用过的`ElementTree`函数导航到此元素，那么我们最终会得到类似以下的内容：

```py
**>>> root.find('body').findall('div')[1].find('p').text**
**Debian 8.0 was.**
**...**

```

但这并不是最佳方法，因为它相当大程度上依赖于 HTML 结构。例如，插入一个我们需要的`<div>`标签之前的变化会破坏它。此外，在更复杂的文档中，这可能导致可怕的方法调用链，难以维护。我们在上一节中使用`<title>`标签来获取代号的方法是一个很好的技巧的例子，因为文档中始终只有一个`<head>`和一个`<title>`标签。找到我们的`<div>`的更好方法是利用它包含的`id="content"`属性。将页面分成几个顶级`<div>`，如页眉、页脚和内容，并为`<div>`赋予标识它们的`id`属性，是一种常见的网页设计模式。

因此，如果我们可以搜索具有`id`属性为`"content"`的`<div>`，那么我们将有一种干净的方法来选择正确的`<div>`。文档中只有一个匹配的`<div>`，并且不太可能会添加另一个类似的`<div>`到文档中。这种方法不依赖于文档结构，因此不会受到对结构所做的任何更改的影响。我们仍然需要依赖于`<div>`中的`<p>`标签是出现的第一个`<p>`标签，但鉴于没有其他方法来识别它，这是我们能做的最好的。

那么，我们如何运行这样的搜索来找到我们的内容`<div>`呢？

## 使用 XPath 搜索

为了避免穷举迭代和检查每个元素，我们需要使用**XPath**，它比我们迄今为止使用的更强大。它是一种专门为 XML 开发的查询语言，并且得到了`lxml`的支持。此外，标准库实现对其提供了有限的支持。

我们将快速了解 XPath，并在此过程中找到之前提出的问题的答案。

要开始使用 Python shell，可以执行以下操作：

```py
**>>> root.xpath('body')**
**[<Element body at 0x39e0908>]**

```

这是 XPath 表达式的最简单形式：它搜索当前元素的子元素，其标签名称与指定的标签名称匹配。当前元素是我们在其上调用`xpath()`的元素，在本例中是`root`。`root`元素是 HTML 文档中的顶级`<html>`元素，因此返回的元素是`<body>`元素。

XPath 表达式可以包含多个级别的元素。搜索从进行`xpath()`调用的节点开始，并随着它们在表达式中匹配连续元素而向下工作。我们可以利用这一点来仅查找`<body>`的`<div>`子元素。

```py
**>>> root.xpath('body/div')**
**[<Element div at 0x39e06c8>, <Element div at 0x39e05c8>, <Element div at 0x39e0608>]**

```

`body/div`表达式意味着匹配当前元素的`<body>`子元素的`<div>`子元素。在 XML 文档中，具有相同标签的元素可以在同一级别出现多次，因此 XPath 表达式可以匹配多个元素，因此`xpath()`函数始终返回一个列表。

前面的查询是相对于我们称之为`xpath()`的元素的，但我们可以通过在表达式开头添加斜杠来强制从树的根部进行搜索。我们还可以通过双斜杠来对元素的所有后代进行搜索。要做到这一点，请尝试以下操作：

```py
**>>> root.xpath('//h1')**
**[<Element h1 at 0x2ac3b08>]**

```

在这里，我们只通过指定单个标记就直接找到了我们的`<h1>`元素，即使它在`root`下面几个级别。表达式开头的双斜杠将始终从根目录搜索，但如果我们希望从上下文元素开始搜索，可以在前面加上一个点。

```py
**>>> root.find('head').xpath('.//h1')**
**[]**

```

这将找不到任何内容，因为`<head>`没有`<h1>`的后代。

### XPath 条件

因此，通过提供路径，我们可以非常具体，但 XPath 的真正力量在于对路径中的元素应用附加条件。特别是，我们前面提到的问题，即测试元素属性。

```py
**>>> root.xpath('//div[@id="content"]')**
**[<Element div at 0x39e05c8>]**

```

在`div`后面的方括号`[@id="content"]`形成了我们放在匹配的`<div>`元素上的条件。`id`之前的`@`符号表示`id`是一个属性，因此条件的含义是：只有`id`属性等于`"content"`的元素。这就是我们如何找到我们的内容`<div>`。

在我们使用它来提取信息之前，让我们简单介绍一下我们可以使用条件做的一些有用的事情。我们可以只指定一个标记名称，如下所示：

```py
**>>> root.xpath('//div[h1]')**
**[<Element div at 0x39e05c8>]**

```

这将返回所有具有`<h1>`子元素的`<div>`元素。也可以尝试：

```py
**>>> root.xpath('body/div[2]'):**
**[<Element div at 0x39e05c8>]**

```

将数字作为条件将返回匹配列表中的该位置的元素。在这种情况下，这是`<body>`的第二个`<div>`子元素。请注意，这些索引从`1`开始，而不像 Python 索引从`0`开始。

XPath 还有很多功能，完整的规范是**万维网联盟**（**W3C**）的标准。最新版本可以在[`www.w3.org/TR/xpath-3/`](http://www.w3.org/TR/xpath-3/)上找到。

## 汇总

现在我们已经将 XPath 添加到我们的超能力中，让我们通过编写一个脚本来获取我们的 Debian 版本信息来完成。创建一个新文件`get_debian_version.py`，并将以下内容保存到其中：

```py
import re
import requests
from lxml.etree import HTML

response = requests.get('http://www.debian.org/releases/stable/')
root = HTML(response.content)
title_text = root.find('head').find('title').text
release = re.search('\u201c(.*)\u201d', title_text).group(1)
p_text = root.xpath('//div[@id="content"]/p[1]')[0].text
version = p_text.split()[1]

print('Codename: {}\nVersion: {}'.format(release, version))
```

在这里，我们通过 XPath 下载和解析了网页，通过 XPath 提取我们想要的文本。我们使用了正则表达式来提取*jessie*，并使用`split`来提取版本 8.0。最后我们将其打印出来。

因此，像这里显示的那样运行它：

```py
**$ python3.4 get_debian_version.py**
**Codename: jessie**
**Version: 8.0**

```

了不起。至少非常巧妙。有一些第三方包可用于加快抓取和表单提交的速度，其中两个流行的包是 Mechanize 和 Scrapy。请在[`wwwsearch.sourceforge.net/mechanize/`](http://wwwsearch.sourceforge.net/mechanize/)和[`scrapy.org`](http://scrapy.org)上查看它们。

# 伟大的力量……

作为 HTTP 客户端开发人员，您可能有不同的优先级，与运行网站的网络管理员不同。网络管理员通常会为人类用户提供网站；可能提供旨在产生收入的服务，并且很可能所有这些都需要在非常有限的资源的帮助下完成。他们将对分析人类如何使用他们的网站感兴趣，并且可能有他们希望自动客户端不要探索的网站区域。

自动解析和下载网站页面的 HTTP 客户端被称为各种各样的东西，比如*机器人*、*网络爬虫*和*蜘蛛*。机器人有许多合法的用途。所有的搜索引擎提供商都大量使用机器人来爬取网页并构建他们庞大的页面索引。机器人可以用来检查死链接，并为存储库存档网站，比如 Wayback Machine。但是，也有许多可能被认为是非法的用途。自动遍历信息服务以提取其页面上的数据，然后在未经网站所有者许可的情况下重新打包这些数据以在其他地方展示，一次性下载大批量的媒体文件，而服务的精神是在线查看等等，这些都可能被认为是非法的。一些网站有明确禁止自动下载的服务条款。尽管一些行为，比如复制和重新发布受版权保护的材料，显然是非法的，但其他一些行为则需要解释。这个灰色地带是一个持续辩论的话题，而且不太可能会得到所有人的满意解决。

然而，即使它们确实有合法的目的，总的来说，机器人确实使网站所有者的生活变得更加困难。它们污染了 Web 服务器日志，而网站所有者用这些日志来计算他们的人类受众如何使用他们的网站的统计数据。机器人还会消耗带宽和其他服务器资源。

使用本章中我们正在研究的方法，编写一个执行许多前述功能的机器人是非常简单的。网站所有者为我们提供了我们将要使用的服务，因此，作为回报，我们应该尊重上述领域，并设计我们的机器人，使它们对他们的影响尽可能小。

## 选择用户代理

我们可以做一些事情来帮助我们的网站所有者。我们应该为我们的客户端选择一个合适的用户代理。网站所有者从日志文件中过滤出机器人流量的主要方法是通过用户代理分析。

有已知机器人的用户代理列表，例如，可以在[`www.useragentstring.com/pages/Crawlerlist/`](http://www.useragentstring.com/pages/Crawlerlist/)找到这样的列表。

网站所有者可以在他们的过滤器中使用这些。许多网站所有者也会简单地过滤掉包含*bot*、*spider*或*crawler*等词的用户代理。因此，如果我们编写的是一个自动化机器人而不是一个浏览器，那么如果我们使用包含这些词中的一个的用户代理，那么这将使网站所有者的生活变得更加轻松。搜索引擎提供商使用的许多机器人都遵循这个惯例，这里列举了一些例子：

+   `Mozilla/5.0` `compatible; bingbot/2.0; http://www.bing.com/bingbot.htm`

+   `Baiduspider: http://www.baidu.com/search/spider.htm`

+   `Mozilla/5.0 compatible; Googlebot/2.1; http://www.google.com/bot.html`

在 HTTP RFC 7231 的第 5.5.3 节中也有一些指南。

## Robots.txt 文件

有一个非官方但标准的机制，可以告诉机器人网站的哪些部分不应该被爬取。这个机制称为`robots.txt`，它采用一个名为`robots.txt`的文本文件的形式。这个文件总是位于网站的根目录，以便机器人总是可以找到它。它包含描述网站可访问部分的规则。文件格式在[`www.robotstxt.org`](http://www.robotstxt.org)中有描述。

Python 标准库提供了`urllib.robotparser`模块，用于解析和处理`robots.txt`文件。您可以创建一个解析器对象，将`robots.txt`文件传递给它，然后可以简单地查询它，以查看给定用户代理是否允许给定 URL。在标准库的文档中可以找到一个很好的例子。如果您在访问之前检查客户端可能想要访问的每个 URL，并遵守网站所有者的意愿，那么您将会帮助他们。

最后，由于我们可能会频繁地进行请求来测试我们新建的客户端，最好是在本地复制你想让客户端解析和测试的网页或文件。这样，我们既可以为自己节省带宽，也可以为网站节省带宽。

# 总结

在本章中，我们涵盖了很多内容，但现在你应该能够开始真正利用你遇到的 Web API 了。

我们研究了 XML，如何构建文档，解析它们并通过使用`ElementTree` API 从中提取数据。我们研究了 Python 的`ElementTree`实现和`lxml`。我们还研究了 XPath 查询语言如何有效地从文档中提取信息。

我们研究了 Amazon S3 服务，并编写了一个客户端，让我们可以执行基本操作，比如创建存储桶，通过 S3 REST API 上传和下载文件。我们学习了如何设置访问权限和内容类型，使文件在 Web 浏览器中正常工作。

我们讨论了 JSON 数据格式，如何将 Python 对象转换为 JSON 数据格式，以及如何将它们转换回 Python 对象。

然后，我们探索了 Twitter API，并编写了一个按需的世界时钟服务，通过这个服务，我们学会了如何阅读和处理账户的推文，以及如何发送推文作为回复。

我们看到了如何从网页的 HTML 源代码中提取信息。我们学习了在使用`ElementTree`和`lxml` HTML 解析器时如何处理 HTML。我们还学习了如何使用 XPath 来帮助使这个过程更加高效。

最后，我们研究了如何回报给为我们提供所有数据的网站管理员。我们讨论了一些编写客户端的方式，使网站管理员的生活变得更轻松，并尊重他们希望我们如何使用他们的网站。

所以，暂时就介绍这么多关于 HTTP 了。我们将在第九章中重新讨论 HTTP，*Web 应用程序*，届时我们将学习如何使用 Python 构建 Web 应用程序的服务器端。在下一章中，我们将讨论互联网的另一个重要工具：电子邮件。
