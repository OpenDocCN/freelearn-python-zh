# *第九章*：外部 API – 与其他系统集成

Odoo 服务器提供了一个外部 API，该 API 被其 Web 客户端使用，并且也适用于其他客户端应用程序。在本章中，我们将学习如何使用 Odoo 外部 API 通过将其作为后端来实现与 Odoo 服务器交互的外部应用程序。

这可以用来编写脚本以加载或修改 Odoo 数据，或者与现有的 Odoo 业务应用程序集成，这互补且不能被 Odoo 应用程序取代。

我们将描述如何使用 OdooRPC 调用，然后利用这些知识使用 Python 构建一个简单的命令行应用程序，用于*图书馆*Odoo 应用程序。

本章将涵盖以下主题：

+   介绍学习项目 – 一个用于编目书籍的客户端应用程序

+   在客户端机器上设置 Python

+   探索 Odoo 外部 API

+   实现客户端应用程序的 XML-RPC 接口

+   实现客户端应用程序的用户界面

+   使用 OdooRPC 库

到本章结束时，您应该已经创建了一个简单的 Python 应用程序，该应用程序可以使用 Odoo 作为后端来查询和存储数据。

# 技术要求

本章中的代码需要我们在*第三章*，*您的第一个 Odoo 应用程序*中创建的`library_app` Odoo 模块。相应的代码可以在此书的 GitHub 仓库中找到：[`github.com/PacktPublishing/Odoo-15-Development-Essentials`](https://github.com/PacktPublishing/Odoo-15-Development-Essentials)。

Git 克隆仓库的路径应该在 Odoo 插件路径中，并且应该安装`library_app`模块。代码示例将假设您正在使用的 Odoo 数据库是`library`，以与提供的安装说明保持一致，见*第二章*，*准备开发环境*。

本章中的代码可以在同一仓库中的`ch09/client_app/`目录中找到。

# 介绍学习项目 – 一个用于编目书籍的客户端应用程序

在本章中，我们将开发一个简单的客户端应用程序来管理图书馆书籍编目。这是一个**命令行界面**（CLI）应用程序，使用 Odoo 作为其后端。我们将实现的功能将是基本的，以保持对与 Odoo 服务器交互所使用技术的关注。

这个简单的命令行应用程序应该能够完成以下操作：

+   通过标题搜索和列出书籍。

+   向编目中添加新书。

+   编辑书籍标题。

目标是专注于如何使用 Odoo 外部 API，因此我们希望避免引入您可能不熟悉的额外编程语言。通过引入这个限制，最合理的选择是使用 Python 来实现客户端应用程序。然而，一旦我们理解了特定语言的 XML-RPC 库，处理 RPC 调用的技术也将适用。

应用程序将是一个 Python 脚本，它期望执行特定命令。以下是一个示例：

```py
$ python3 library.py add "Moby-Dick"
$ python3 library.py list "moby"
3 Moby-Dick
$ python3 library.py set-title 3 "Moby Dick"
```

此示例会话演示了使用客户端应用程序添加、列出和修改书名。

此客户端应用程序将使用 Python 运行。在我们开始查看客户端应用程序的代码之前，我们必须确保 Python 已安装在客户端机器上。

# 在客户端机器上设置 Python

Odoo API 可以通过两种不同的协议从外部访问：XML-RPC 和 JSON-RPC。任何能够实现这些协议之一客户端的外部程序都将能够与 Odoo 服务器交互。为了避免引入额外的编程语言，我们将使用 Python 来探索外部 API。

到目前为止，Python 代码仅在服务器端使用。对于客户端应用程序，Python 代码将在客户端运行，因此工作站可能需要额外的设置。

要遵循本章中的示例，您所使用的系统需要能够运行 Python 3 代码。如果您已经遵循了本书其他章节中使用的相同开发环境，这可能已经实现。然而，如果尚未实现，我们应该确保 Python 已安装。

要确保开发工作站上已安装 Python 3，请在终端窗口中运行`python3 --version`命令。如果没有安装，请参考官方网站以找到适用于您系统的安装包，网址为[`www.python.org/downloads/`](https://www.python.org/downloads/)。

对于 Ubuntu，有很大可能性它已经预安装在您的系统上。如果没有，可以使用以下命令安装：

```py
$ sudo apt-get install python3 python3-pip
```

对于 Windows 10，可以从 Microsoft Store 安装。

在 PowerShell 中运行`python3`将引导您到相应的下载页面。

如果您是 Windows 用户并且已使用一站式安装程序安装了 Odoo，您可能会想知道为什么 Python 解释器尚未对您可用。在这种情况下，您需要额外的安装。简短的回答是，Odoo 一站式安装程序包含一个嵌入的 Python 解释器，它不会直接提供给通用系统。

现在 Python 已经安装并可用，它可以用来探索 Odoo 外部 API。

# 探索 Odoo 外部 API

在我们实现客户端应用程序之前，应该先熟悉 Odoo 外部 API。以下章节将使用*Python 解释器*探索 XML-RPC API。

## 使用 XML-RPC 连接到 Odoo 外部 API

访问 Odoo 服务器的最简单方法是使用 XML-RPC。Python 标准库中的`xmlrpc`库可以用于此目的。

请记住，正在开发的应用程序是一个连接到服务器的客户端。因此，需要一个正在运行的 Odoo 服务器实例供客户端连接。代码示例将假设 Odoo 服务器实例在同一台机器上运行，`http://localhost:8069`，但如果您要使用的服务器在不同的机器上运行，则可以使用任何可到达的 URL。

Odoo `xmlrpc/2/common` 端点公开了公共方法，并且可以在不登录的情况下访问这些方法。这些方法可以用来检查服务器版本和验证登录凭证。让我们使用 `xmlrpc` 库来探索公开的 `common` Odoo API。

首先，启动 Python 3 控制台并输入以下内容：

```py
>>> from xmlrpc import client
>>> srv = "http://localhost:8069"
>>> common = client.ServerProxy("%s/xmlrpc/2/common" % srv)
>>> common.version()
{'server_version': '15.0', 'server_version_info': [15, 0, 0, 'final', 0, ''], 'server_serie': '15.0', 'protocol_version': 1}
```

之前的代码导入了 `xmlrpc` 库，并设置了一个包含服务器地址和监听端口的变量。这可以根据要连接的 Odoo 服务器的特定 URL 进行调整。

接下来，创建一个 XML-RPC 客户端对象以访问在 `/xmlrpc/2/common` 端点公开的服务器公共服务。您不需要登录。那里可用的方法之一是 `version()`，它用于检查 Odoo 服务器版本。这是一种简单的方法来确认与服务器通信是否正常。

另一个有用的公共方法是 `authenticate()`。此方法确认用户名和密码被接受，并返回在请求中应使用的用户 ID。以下是一个示例：

```py
>>> db, user, password = "library", "admin", "admin"
>>> uid = common.authenticate(db, user, password, {})
>>> print(uid)
2
```

`authenticate()` 方法期望四个参数：数据库名称、用户名、密码和用户代理。之前的代码使用变量来存储这些信息，然后将这些变量作为参数传递。

Odoo 14 的变化

Odoo 14 支持使用 API 密钥，这可能对于外部访问 Odoo API 是必需的。API 密钥可以在用户的**偏好**表单中设置，在**账户安全**选项卡中。XML-RPC 的使用方式相同，只是应该使用 API 密钥作为密码。更多详细信息请参阅官方文档[`www.odoo.com/documentation/15.0/developer/misc/api/odoo.html#api-keys`](https://www.odoo.com/documentation/15.0/developer/misc/api/odoo.html#api-keys)。

应使用用户代理环境来提供有关客户端的一些元数据。这是强制性的，至少应该是一个空字典 `{}`。

如果身份验证失败，将返回 `False` 值。

`common` 公共端点相当有限，因此要访问 ORM API 或其他端点，需要使用所需的身份验证。

## 使用 XML-RPC 运行服务器方法

要访问 Odoo 模型和它们的方法，需要使用 `xmlrpc/2/object` 端点。对该端点的请求需要登录详细信息。

此端点公开了一个通用的 `execute_kw` 方法，并接收模型名称、要调用的方法以及一个包含传递给该方法的参数列表。

下面是一个 `execute_kw` 的工作示例。它调用 `search_count` 方法，该方法返回与域过滤器匹配的记录数：

```py
>>> api = xmlrpc.client.ServerProxy('%s/xmlrpc/2/object' % srv)
>>> api.execute_kw(db, uid, password, "res.users", "search_count", [[]])
3
```

此代码使用 `xmlrpc/2/endpoint` 对象来访问服务器 API。使用以下参数调用 `execute_kw()` 方法：

+   要连接到的数据库名称

+   连接用户 ID

+   用户密码（或 API 密钥）

+   目标模型标识符

+   要调用的方法

+   位置参数列表

+   可选的包含关键字参数的字典（在此示例中未使用）

可以调用所有模型方法，除了以下划线 (`_`) 开头的，这些被认为是私有的。某些方法可能不适用于 XML-RPC 协议，如果它们返回的值无法通过 XML-RPC 协议发送。这是 `browse()` 的情况，它返回一个记录集对象。尝试通过 XML-RPC 使用 `browse()` 会返回 `TypeError: cannot marshal objects` 错误。而不是 `browse()`，XML-RPC 调用应使用 `read` 或 `search_read`，它们返回的数据格式是 XML-RPC 协议可以发送到客户端的格式。

现在，让我们看看如何使用 `search` 和 `read` 来查询 Odoo 数据。

## 使用搜索和读取 API 方法

Odoo 服务器端代码使用 `browse` 来查询记录。RPC 客户端不能使用它，因为记录集对象不能通过 RPC 协议传输。相反，应使用 `read` 方法。

`read([<ids>, [<fields>])` 与 `browse` 方法类似，但它返回的是记录列表，而不是记录集。每个记录都是一个字典，包含请求的字段及其数据。

让我们看看如何使用 `read()` 方法从 Odoo 中检索数据：

```py
>>> api = xmlrpc.client.ServerProxy("%s/xmlrpc/2/object" % srv)
>>> api.execute_kw(db, uid, password, "res.users", "read", [2, ["login", "name", "company_id"]])
[{'id': 2, 'login': 'admin', 'name': 'Mitchell Admin', 'company_id': [1, 'YourCompany']}]
```

上述示例调用 `res.users` 模型的 `read` 方法，带有两个位置参数——记录 ID `2`（也可以使用 ID 列表）和要检索的字段列表 `["login", "name", "company_id"]`，以及没有关键字参数。

结果是一个字典列表，其中每个字典都是一个记录。多对多字段的值遵循特定的表示。它们是一对值，包含记录 ID 和记录显示名称。例如，之前返回的 `company_id` 值是 `[1, 'YourCompany']`。

记录 ID 可能未知，在这种情况下，需要搜索调用以找到匹配域过滤器的记录 ID。

例如，如果我们想找到管理员用户，我们可以使用 `[("login", "=", "admin")]`。这个 RPC 调用如下所示：

```py
>>> domain = [("login", "=", "admin")]
>>> api.execute_kw(db, uid, password, "res.users", "search", [domain])
[2]
```

结果是一个只有一个元素的列表，`2`，这是 `admin` 用户的 ID。

常见的操作是使用 `search` 和 `read` 方法的组合来查找符合域过滤器的记录 ID，然后检索它们的数据。对于客户端应用程序来说，这意味着对服务器进行两次往返。为了简化这个过程，`search_read` 方法可用，它可以在单步中执行这两个操作。

这里有一个使用 `search_read` 来查找管理员用户并返回其名称的示例：

```py
>>> api.execute_kw(db, uid, password, "res.users", "search_read", [domain, ["login", "name"]])
[{'id': 2, 'login': 'admin', 'name': 'Mitchell Admin'}]
```

`search_read`方法使用了两个位置参数：一个包含域过滤器的列表，以及一个包含要检索的字段的第二个列表。

`search_read`的参数如下：

+   `domain`：一个包含域过滤表达式的列表

+   `fields`：一个包含要检索的字段名称的列表

+   `offset`：要跳过的记录数或用于记录分页的记录数

+   `limit`：要返回的最大记录数

+   `order`：用于数据库的`ORDER BY`子句的字符串

`fields`参数对于`read`和`search_read`都是可选的。如果没有提供，将检索所有模型字段。但这可能会导致昂贵的函数字段计算和检索大量可能不需要的数据。因此，建议提供显式的字段列表。

`execute_kw`调用可以使用位置参数和关键字参数。以下是在使用关键字参数而不是位置参数时，相同的调用看起来是什么样子：

```py
>>> api.execute_kw(db, uid, password, "res.users", "search_read", [], {"domain": domain, "fields": ["login", "name"]})
```

`search_read`是检索数据最常用的方法，但还有更多方法可用于写入数据或触发其他业务逻辑。

## 调用其他 API 方法

除了前缀为下划线的那些方法被认为是私有的之外，所有其他模型方法都通过 RPC 公开。这意味着`create`、`write`和`unlink`可以调用以在服务器上修改数据。

让我们看看一个例子。以下代码创建了一个新的合作伙伴记录，修改了它，读取以确认修改已被写入，并最终删除它：

```py
>>> x = api.execute_kw(db, uid, password, "res.partner", "create", 
[{'name': 'Packt Pub'}])
>>> print(x)
49
>>> api.execute_kw(db, uid, password, "res.partner", "write", 
[[x], {'name': 'Packt Publishing'}]) 
True
>>> api.execute_kw(db, uid, password, "res.partner", "read", 
[[x], ["name"]])
[{'id': 49, 'name': 'Packt Publishing'}]
>>> api.execute_kw(db, uid, password, "res.partner", "unlink", [[x]])
True
>>> api.execute_kw(db, uid, password, "res.partner", "read", [[x]])
[]
```

XML-RPC 协议的一个限制是它不支持`None`值。有一个支持`None`值的 XML-RPC 扩展，但这是否可用将取决于客户端应用程序使用的特定 XML-RPC 库。不返回任何内容的方法可能无法通过 XML-RPC 使用，因为它们隐式返回`None`。这就是为什么方法始终返回某些内容，如`True`值是一个好习惯。另一个选择是使用 JSON-RPC。`OdooRPC`库支持此协议，它将在本章的“使用 OdooRPC 库”部分中使用。

前缀为下划线的`Model`方法被认为是私有的，并且不会通过 XML-RPC 公开。

小贴士

通常，客户端应用程序希望在一个 Odoo 表单上复制手动用户输入。调用`create()`方法可能不足以完成这项任务，因为表单可以使用`onchange`方法来自动化一些字段，这些方法是由表单的交互触发的，而不是由`create()`触发的。解决方案是在 Odoo 服务器上创建一个自定义方法，该方法使用`create()`然后运行所需的`onchange`方法。

值得重复的是，Odoo 外部 API 可以被大多数编程语言使用。官方文档提供了 Ruby、PHP 和 Java 的示例。这些信息可在[`www.odoo.com/documentation/15.0/webservices/odoo.html`](https://www.odoo.com/documentation/15.0/webservices/odoo.html)找到。

到目前为止，我们已经看到了如何使用 XML-RPC 协议调用 Odoo 方法。现在，我们可以使用这个来构建书籍目录客户端应用程序。

# 实现客户端应用程序 XML-RPC 接口

让我们先从实现图书馆书籍目录客户端应用程序开始。

这可以分成两个文件：一个包含服务器后端`library_xmlrpc.py`的 Odoo 后端接口，另一个是用户界面`library.py`。这将允许我们为后端接口使用替代实现。

从 Odoo 后端组件开始，将使用`LibraryAPI`类来设置与支持与 Odoo 交互所需方法的 Odoo 服务器的连接。要实现的方法如下：

+   `search_read(<title>)`用于通过标题搜索书籍数据

+   `create(<title>)`用于创建具有特定标题的书

+   `write(<id>, <title>)`用于使用书籍 ID 更新书籍标题

+   `unlink(<id>)`用于使用 ID 删除一本书

选择一个目录来存放应用程序文件，并创建`library_xmlrpc.py`文件。首先添加类构造函数，如下所示：

```py
import xmlrpc.client
class LibraryAPI(): 
    def __init__(self, host, port, db, user, pwd):
        common = xmlrpc.client.ServerProxy(
            "http://%s:%d/xmlrpc/2/common" % (host, port))
        self.api = xmlrpc.client.ServerProxy(
            "http://%s:%d/xmlrpc/2/object" % (host, port))
        self.uid = common.authenticate(db, user, pwd, {})
        self.pwd = pwd
        self.db = db
        self.model = "library.book"
```

这个类存储了执行目标模型调用所需的所有信息：API XML-RPC 引用、`uid`、密码、数据库名和模型名。

对 Odoo 的 RPC 调用都将使用相同的`execute_kw` RPC 方法。在它周围添加了一个薄薄的包装器，在`_execute()`私有方法中。这利用了存储在对象中的数据，提供了一个更小的函数签名，如下面的代码块所示：

```py
    def _execute(self, method, arg_list, kwarg_dict=None): 
        return self.api.execute_kw( 
            self.db, self.uid, self.pwd, self.model,
            method, arg_list, kwarg_dict or {})
```

这个`_execute()`私有方法现在可以用于更简洁的高层方法实现。

第一个公共方法是`search_read()`方法。它将接受一个可选的字符串，用于搜索书籍标题。如果没有提供标题，将返回所有记录。这是相应的实现：

```py
    def search_read(self, title=None):
        domain = [("name", "ilike", title)] if title else 
                   [] 
        fields = ["id", "name"]
        return self._execute("search_read", [domain, 
          fields])
```

`create()`方法将创建一个具有给定标题的新书，并返回创建记录的 ID：

```py
    def create(self, title):
        vals = {"name": title}
        return self._execute("create", [vals])
```

`write()`方法将接受新的标题和书籍 ID 作为参数，并在此书籍上执行写操作：

```py
    def write(self, id, title): 
        vals = {"name": title} 
        return self._execute("write", [[id], vals])
```

最后，使用`unlink()`方法根据相应的 ID 删除一本书：

```py
    def unlink(self, id): 
        return self._execute("unlink", [[id]])
```

我们在文件末尾添加一小段测试代码，如果运行 Python 文件，将执行这些代码，有助于测试已实现的方法，如下所示：

```py
if __name__ == "__main__": 
    # Sample test configurations 
    host, port, db = "localhost", 8069, "library" 
    user, pwd = "admin", "admin"
    api = LibraryAPI(host, port, db, user, pwd) 
    from pprint import pprint 
    pprint(api.search_read())
```

如果我们运行这个 Python 脚本，我们应该看到我们的图书馆书籍内容被打印出来：

```py
$ python3 library_xmlrpc.py
[{'id': 1, 'name': 'Odoo Development Essentials 11'},
 {'id': 2, 'name': 'Odoo 11 Development Cookbook'},
 {'id': 3, 'name': 'Brave New World'}]
```

现在我们已经围绕我们的 Odoo 后端有一个简单的包装器，让我们处理命令行用户界面。

# 实现客户端应用程序用户界面

我们的目标是学习如何编写外部应用程序和 Odoo 服务器之间的接口，我们已经在上一节中做到了这一点。但让我们更进一步，为这个简约客户端应用程序构建用户界面。

为了尽可能保持简单，我们将使用简单的命令行用户界面，并避免使用额外的依赖。这使我们能够利用 Python 的内置功能来实现命令行应用程序，以及`ArgumentParser`库。

现在，在`library_xmlrpc.py`文件旁边，创建一个新的`library.py`文件。这个文件将导入 Python 的命令行参数解析器和`LibraryAPI`类，如下面的代码所示：

```py
from argparse import ArgumentParser
from library_xmlrpc import LibraryAPI
```

接下来，我们必须描述参数解析器期望的命令。有四个命令：

+   `list`用于搜索和列出书籍

+   `add`用于添加一本书

+   `set`用于更新书籍标题

+   `del`用于删除一本书

实现上述命令的命令行解析器代码如下：

```py
parser = ArgumentParser()
parser.add_argument(
    "command",
    choices=["list", "add", "set", "del"])
parser.add_argument("params", nargs="*")  # optional args
args = parser.parse_args()
```

`args`对象代表用户给出的命令行选项。`args.command`是正在使用的命令，而`args.params`包含用于命令的附加参数，如果提供了任何参数。

如果没有给出或给出了错误的命令，参数解析器将处理这种情况，并将向用户显示预期的输入。`argparse`的完整参考可以在官方文档中找到，网址为[`docs.python.org/3/library/argparse.html`](https://docs.python.org/3/library/argparse.html)。

下一步是执行与`user`命令相对应的操作。我们将首先创建一个`LibraryAPI`实例。这需要 Odoo 连接细节，在这个简单的实现中，这些细节将被硬编码，如下所示：

```py
host, port, db = "localhost", 8069, "library"
user, pwd = "admin", "admin"
api = LibraryAPI(host, port, db, user, pwd)
```

第一行设置了服务器实例和要连接的数据库的一些固定参数。在这种情况下，连接到的是本地 Odoo 服务器，监听默认的`8069`端口，连接到`library`数据库。要连接到不同的服务器和数据库，这些参数应相应地进行调整。

必须添加新的特定代码来处理每个命令。我们将从`list`命令开始，它返回书籍列表：

```py
if args.command == "list":
    title = args.params[:1]
    books = api.search_read(title)
    for book in books:
        print("%(id)d %(name)s" % book)
```

在前面的代码中使用了`LibraryAPI.search_read()`方法来检索书籍记录的列表。然后迭代返回的列表以打印出每个元素。

接下来是`add`命令：

```py
if args.command == "add":
    title = args.params[0]
    book_id = api.create(title)
    print("Book added with ID %d for title %s." % (
      book_id, title))
```

由于在`LibraryAPI`对象中已经完成了艰苦的工作，实现只需要调用`create()`方法并向最终用户显示结果。

`set`命令允许我们更改现有书籍的标题。它应该有两个参数——书籍的 ID 和新的标题：

```py
if args.command == "set":
    if len(args.params) != 2:
        print("set command requires a Title and ID.")
    else:
        book_id, title = int(args.params[0]), 
          args.params[1]
        api.write(book_id, title)
        print("Title of Book ID %d set to %s." % (book_id, 
          title))
```

最后，是`del`命令的实现，用于删除书籍记录。这与之前的命令没有太大区别：

```py
if args.command == "del":
    book_id = int(args.params[0])
    api.unlink(book_id)
    print("Book with ID %s was deleted." % book_id)
```

客户端应用程序已完成，您可以使用您选择的命令尝试它。特别是，我们应该能够运行本章开头所示的示例命令。

小贴士

在 Linux 系统上，可以通过运行 `chmod +x library.py` 命令并将 `#!/usr/bin/env python3` 添加到文件的第一行来使 `library.py` 可执行。之后，在命令行中运行 `./library.py` 应该可以工作。

这是一个相当基础的应用程序，很容易想到几种改进它的方法。这里的目的是使用 Odoo RPC API 构建一个最小可行应用。

# 使用 OdooRPC 库

另一个需要考虑的相关客户端库是 `OdooRPC`。它是一个完整的客户端库，使用 JSON-RPC 协议而不是 XML-RPC。尽管 XML-RPC 仍然得到支持，但 Odoo 官方网页客户端也使用 JSON-RPC。

`OdooRPC` 库现在在 Odoo 社区协会的伞下维护。源代码仓库可以在 [`github.com/OCA/odoorpc`](https://github.com/OCA/odoorpc) 找到。

可以使用以下命令从 PyPI 安装 `OdooRPC` 库：

```py
$ pip3 install odoorpc
```

当创建一个新的 `odoorpc.ODOO` 对象时，`OdooRPC` 库会设置一个服务器连接。在这个时候，我们应该使用 `ODOO.login()` 方法来创建一个用户会话。就像在服务器端一样，会话有一个 `env` 属性，包含会话的环境，包括用户 ID、`uid` 和 `context`。

`OdooRPC` 库可以用来为服务器提供 `library_xmlrpc.py` 接口的替代实现。它应该提供相同的功能，但使用 JSON-RPC 而不是 XML-RPC 来实现。

为了实现这一点，将创建一个名为 `library_odoorpc.py` 的 Python 模块，它为 `library_xmlrpc.py` 模块提供了一个即插即用的替代品。为此，创建一个名为 `library_odoorpc.py` 的新文件，并将其放在旁边，该文件包含以下代码：

```py
import odoorpc
class LibraryAPI():
    def __init__(self, host, port, db, user, pwd):
        self.api = odoorpc.ODOO(host, port=port)
        self.api.login(db, user, pwd)
        self.uid = self.api.env.uid
        self.model = "library.book"
        self.Model = self.api.env[self.model]
    def _execute(self, method, arg_list, kwarg_dict=None):
        return self.api.execute(
            self.model,
            method, *arg_list, **kwarg_dict)
```

`OdooRPC` 库实现了 `Model` 和 `Recordset` 对象，它们模仿了服务器端对应对象的行为。目标是使使用此库的代码与 Odoo 服务器端使用的代码相似。客户端使用的方法利用这一点，并在 `self.Model` 属性中存储对 `library.book` 模型对象的引用，该属性由 OdooRPC 的 `env["library.book"]` 调用提供。

`_execute()` 方法也在这里实现；它允许我们将其与普通的 XML-RPC 版本进行比较。OdooRPC 库有一个 `execute()` 方法来运行任意的 Odoo 模型方法。

接下来是 `search_read()`、`create()`、`write()` 和 `unlink()` 客户端方法的实现。在同一个文件中，将这些方法添加到 `LibraryAPI()` 类内部：

```py
    def search_read(self, title=None):
        domain = [("name", "ilike", title)] if title else 
                  []
        fields = ["id", "name"]
        return self.Model.search_read(domain, fields)
    def create(self, title):
        vals = {"name": title}
        return self.Model.create(vals)
    def write(self, id, title):
        vals = {"name": title}
        self.Model.write(id, vals)
    def unlink(self, id):
        return self.Model.unlink(id)
```

注意这个客户端代码与 Odoo 服务器端代码的相似之处。

这个 `LibraryAPI` 对象可以用作 `library_xmlrpc.py` 的即插即用替代品。可以通过编辑 `library.py` 文件并将 `from library_xmlrpc import LibraryAPI` 行更改为 `from library_odoorpc import LibraryAPI` 来用作 RPC 连接层。现在，测试驱动 `library.py` 客户端应用程序；它应该表现得和以前一样！

# 摘要

本章的目标是了解外部 API 的工作原理及其功能。我们首先通过使用 Python XML-RPC 客户端编写简单脚本来探索它，尽管外部 API 可以从任何编程语言中使用。官方文档提供了 Java、PHP 和 Ruby 的代码示例。

然后，我们学习了如何使用 XML-RPC 调用来搜索和读取数据，以及如何调用任何其他方法。例如，我们可以创建、更新和删除记录。

接下来，我们介绍了 OdooRPC 库。它提供了一个在 RPC 基础库（XML-RPC 或 JSON-RPC）之上的层，以提供类似于服务器端可找到的 API 的本地 API。这降低了学习曲线，减少了编程错误，并使得在服务器端和客户端代码之间复制代码变得更加容易。

通过这些，我们已经完成了关于编程 API 和业务逻辑的章节。现在，是时候看看视图和用户界面了。在下一章中，我们将更详细地探讨后端视图以及网络客户端可以提供的开箱即用的用户体验。

# 进一步阅读

以下附加参考资料可能有助于补充本章中描述的主题：

+   Odoo 网络服务的官方文档，包括除 Python 之外编程语言的代码示例：[`www.odoo.com/documentation/15.0/developer/misc/api/odoo.html`](https://www.odoo.com/documentation/15.0/developer/misc/api/odoo.html)

+   OdooRPC 文档：[`pythonhosted.org/OdooRPC`](https://pythonhosted.org/OdooRPC)
