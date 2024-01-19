# 第九章：使用 Python 构建网络 Web 服务

在之前的章节中，我们是各种工具提供的 API 的消费者。在第三章中，*API 和意图驱动的网络*，我们看到我们可以使用`HTTP POST`方法到`http://<your router ip>/ins` URL 上的 NX-API，其中`CLI`命令嵌入在主体中，以远程执行 Cisco Nexus 设备上的命令；然后设备返回命令执行输出。在第八章中，*使用 Python 进行网络监控-第 2 部分*，我们使用`GET`方法来获取我们 sFlow-RT 的`http://<your host ip>:8008/version`上的版本，主体为空。这些交换是 RESTful Web 服务的例子。

根据维基百科（[`en.wikipedia.org/wiki/Representational_state_transfer`](https://en.wikipedia.org/wiki/Representational_state_transfer)）：

“表征状态转移（REST）或 RESTful Web 服务是提供互操作性的一种方式，用于互联网上的计算机系统。符合 REST 标准的 Web 服务允许请求系统使用一组统一和预定义的无状态操作来访问和操作 Web 资源的文本表示。”

如前所述，使用 HTTP 协议的 REST Web 服务只是网络上信息交换的许多方法之一；还存在其他形式的 Web 服务。然而，它是今天最常用的 Web 服务，具有相关的`GET`，`POST`，`PUT`和`DELETE`动词作为信息交换的预定义方式。

使用 RESTful 服务的优势之一是它可以让您隐藏用户对内部操作的了解，同时仍然为他们提供服务。例如，在 sFlow-RT 的情况下，如果我们要登录安装了我们软件的设备，我们需要更深入地了解工具，才能知道在哪里检查软件版本。然而，通过以 URL 的形式提供资源，软件将版本检查操作从请求者中抽象出来，使操作变得更简单。抽象还提供了一层安全性，因为现在可以根据需要仅打开端点。

作为网络宇宙的大师，RESTful Web 服务提供了许多显着的好处，我们可以享受，例如以下：

+   您可以将请求者与网络操作的内部细节分离。例如，我们可以提供一个 Web 服务来查询交换机版本，而无需请求者知道所需的确切 CLI 命令或 API 格式。

+   我们可以整合和定制符合我们网络需求的操作，例如升级所有顶部交换机的资源。

+   我们可以通过仅在需要时公开操作来提供更好的安全性。例如，我们可以为核心网络设备提供只读 URL（`GET`），并为访问级别交换机提供读写 URL（`GET` / `POST` / `PUT` / `DELETE`）。

在本章中，我们将使用最流行的 Python Web 框架之一**Flask**来为我们的网络创建自己的 REST Web 服务。在本章中，我们将学习以下内容：

+   比较 Python Web 框架

+   Flask 简介

+   静态网络内容的操作

+   涉及动态网络操作的操作

让我们开始看看可用的 Python Web 框架以及为什么我们选择了 Flask。

# 比较 Python Web 框架

Python 以其众多的 web 框架而闻名。在 PyCon 上有一个笑话，即你永远不能成为全职 Python 开发者而不使用任何 Python web 框架。甚至为 Django 举办了一年一度的会议，这是最受欢迎的 Python 框架之一，叫做 DjangoCon。每年都吸引数百名与会者。如果你在[`hotframeworks.com/languages/python`](https://hotframeworks.com/languages/python)上对 Python web 框架进行排序，你会发现在 Python 和 web 框架方面选择是不缺乏的。

![](img/15528b62-5084-4ef6-936e-5f53f2f6faa0.png)Python web 框架排名

有这么多选择，我们应该选择哪个框架呢？显然，自己尝试所有的框架将非常耗时。关于哪个 web 框架更好的问题也是网页开发者之间的一个热门话题。如果你在任何论坛上问这个问题，比如 Quora，或者在 Reddit 上搜索，准备好接受一些充满个人意见的答案和激烈的辩论。

说到 Quora 和 Reddit，这里有一个有趣的事实：Quora 和 Reddit 都是用 Python 编写的。Reddit 使用 Pylons（[`www.reddit.com/wiki/faq#wiki_so_what_python_framework_do_you_use.3F`](https://www.reddit.com/wiki/faq#wiki_so_what_python_framework_do_you_use.3F.)），而 Quora 最初使用 Pylons，但用他们自己的内部代码替换了部分框架（[`www.quora.com/What-languages-and-frameworks-are-used-to-code-Quora`](https://www.quora.com/What-languages-and-frameworks-are-used-to-code-Quora)）。

当然，我对编程语言（Python！）和 web 框架（Flask！）有自己的偏见。在这一部分，我希望向你传达我选择一个而不是另一个的理由。让我们选择前面 HotFrameworks 列表中的前两个框架并进行比较：

+   **Django**：这个自称为“完美主义者与截止日期的 web 框架”是一个高级 Python web 框架，鼓励快速开发和清晰的实用设计（[`www.djangoproject.com/`](https://www.djangoproject.com/)）。它是一个大型框架，提供了预先构建的代码，提供了管理面板和内置内容管理。

+   **Flask**：这是一个基于 Werkzeug，Jinja2 和良好意图的 Python 微框架（[`flask.pocoo.org/`](http://flask.pocoo.org/)）。作为一个微框架，Flask 的目标是保持核心小，需要时易于扩展。微框架中的“微”并不意味着 Flask 功能不足，也不意味着它不能在生产环境中工作。

就我个人而言，我觉得 Django 有点难以扩展，大部分时间我只使用预先构建的代码的一小部分。Django 框架对事物应该如何完成有着强烈的意见；任何偏离这些意见的行为有时会让用户觉得他们在“与框架作斗争”。例如，如果你看一下 Django 数据库文档，你会注意到这个框架支持多种不同的 SQL 数据库。然而，它们都是 SQL 数据库的变体，比如 MySQL，PostgreSQL，SQLite 等。如果你想使用 NoSQL 数据库，比如 MongoDB 或 CouchDB 呢？这可能是可能的，但可能会让你自己摸索。成为一个有主见的框架当然不是坏事，这只是一个观点问题（无意冒犯）。

我非常喜欢保持核心代码简洁，并在需要时进行扩展的想法。文档中让 Flask 运行的初始示例只包含了八行代码，即使你没有任何经验，也很容易理解。由于 Flask 是以扩展为核心构建的，编写自己的扩展，比如装饰器，非常容易。尽管它是一个微框架，但 Flask 核心仍然包括必要的组件，比如开发服务器、调试器、与单元测试的集成、RESTful 请求分发等等，可以让你立即开始。正如你所看到的，除了 Django，Flask 是按某些标准来说第二受欢迎的 Python 框架。社区贡献、支持和快速发展带来的受欢迎程度有助于进一步扩大其影响力。

出于上述原因，我觉得 Flask 是我们在构建网络 Web 服务时的理想选择。

# Flask 和实验设置

在本章中，我们将使用`virtualenv`来隔离我们将要工作的环境。顾名思义，virtualenv 是一个创建虚拟环境的工具。它可以将不同项目所需的依赖项保存在不同的位置，同时保持全局 site-packages 的清洁。换句话说，当你在虚拟环境中安装 Flask 时，它只会安装在本地`virtualenv`项目目录中，而不是全局 site-packages。这使得将代码移植到其他地方变得非常容易。

很有可能在之前使用 Python 时，你已经接触过`virtualenv`，所以我们会快速地浏览一下这个过程。如果你还没有接触过，可以随意选择在线的优秀教程之一，比如[`docs.python-guide.org/en/latest/dev/virtualenvs/`](http://docs.python-guide.org/en/latest/dev/virtualenvs/)。

要使用，我们首先需要安装`virtualenv`。

```py
# Python 3
$ sudo apt-get install python3-venv
$ python3 -m venv venv

# Python 2
$ sudo apt-get install python-virtualenv
$ virtualenv venv-python2
```

下面的命令使用`venv`模块（`-m venv`）来获取一个带有完整 Python 解释器的`venv`文件夹。我们可以使用`source venv/bin/activate`和`deactivate`来进入和退出本地 Python 环境：

```py
$ source venv/bin/activate
(venv) $ python
$ which python
/home/echou/Master_Python_Networking_second_edition/Chapter09/venv/bin/python
$ python
Python 3.5.2 (default, Nov 23 2017, 16:37:01)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
>>> exit()
(venv) $ deactivate
```

在本章中，我们将安装相当多的 Python 包。为了让生活更轻松，我在书的 GitHub 存储库中包含了一个`requirements.txt`文件；我们可以使用它来安装所有必要的包（记得激活你的虚拟环境）。在过程结束时，你应该看到包被下载并成功安装：

```py
(venv) $ pip install -r requirements.txt
Collecting Flask==0.10.1 (from -r requirements.txt (line 1))
  Downloading https://files.pythonhosted.org/packages/db/9c/149ba60c47d107f85fe52564133348458f093dd5e6b57a5b60ab9ac517bb/Flask-0.10.1.tar.gz (544kB)
    100% |████████████████████████████████| 552kB 2.0MB/s
Collecting Flask-HTTPAuth==2.2.1 (from -r requirements.txt (line 2))
  Downloading https://files.pythonhosted.org/packages/13/f3/efc053c66a7231a5a38078a813aee06cd63ca90ab1b3e269b63edd5ff1b2/Flask-HTTPAuth-2.2.1.tar.gz
... <skip>
  Running setup.py install for Pygments ... done
  Running setup.py install for python-dateutil ... done
Successfully installed Flask-0.10.1 Flask-HTTPAuth-2.2.1 Flask-SQLAlchemy-1.0 Jinja2-2.7.3 MarkupSafe-0.23 Pygments-1.6 SQLAlchemy-0.9.6 Werkzeug-0.9.6 httpie-0.8.0 itsdangerous-0.24 python-dateutil-2.2 requests-2.3.0 six-1.11.0 
```

对于我们的网络拓扑，我们将使用一个简单的四节点网络，如下所示：

![](img/1647347c-bb46-4301-82d5-4cd6b61096bc.png) 实验拓扑

让我们在下一节中看一下 Flask。

请注意，从现在开始，我将假设你总是在虚拟环境中执行，并且已经安装了`requirements.txt`文件中的必要包。

# Flask 简介

像大多数流行的开源项目一样，Flask 有非常好的文档，可以在[`flask.pocoo.org/docs/0.10/`](http://flask.pocoo.org/docs/0.10/)找到。如果任何示例不清楚，你可以肯定会在项目文档中找到答案。

我还强烈推荐 Miguel Grinberg（[`blog.miguelgrinberg.com/`](https://blog.miguelgrinberg.com/)）关于 Flask 的工作。他的博客、书籍和视频培训让我对 Flask 有了很多了解。事实上，Miguel 的课程*使用 Flask 构建 Web API*启发了我写这一章。你可以在 GitHub 上查看他发布的代码：[`github.com/miguelgrinberg/oreilly-flask-apis-video`](https://github.com/miguelgrinberg/oreilly-flask-apis-video)。

我们的第一个 Flask 应用程序包含在一个单独的文件`chapter9_1.py`中：

```py
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_networkers():
    return 'Hello Networkers!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
```

这几乎总是 Flask 最初的设计模式。我们使用 Flask 类的实例作为应用程序模块包的第一个参数。在这种情况下，我们使用了一个单一模块；在自己操作时，输入您选择的名称，以指示它是作为应用程序启动还是作为模块导入。然后，我们使用路由装饰器告诉 Flask 哪个 URL 应该由`hello_networkers()`函数处理；在这种情况下，我们指定了根路径。我们以通常的名称结束文件（[`docs.python.org/3.5/library/__main__.html`](https://docs.python.org/3.5/library/__main__.html)）。我们只添加了主机和调试选项，允许更详细的输出，并允许我们监听主机的所有接口（默认情况下，它只监听回环）。我们可以使用开发服务器运行此应用程序：

```py
(venv) $ python chapter9_1.py
 * Running on http://0.0.0.0:5000/
 * Restarting with reloader
```

既然我们有一个运行的服务器，让我们用一个 HTTP 客户端测试服务器的响应。

# HTTPie 客户端

我们已经安装了 HTTPie ([`httpie.org/`](https://httpie.org/)) 作为从阅读`requirements.txt`文件安装的一部分。尽管本书是黑白文本打印的，所以这里看不到，但在您的安装中，您可以看到 HTTPie 对 HTTP 事务有更好的语法高亮。它还具有更直观的 RESTful HTTP 服务器命令行交互。我们可以用它来测试我们的第一个 Flask 应用程序（后续将有更多关于 HTTPie 的例子）：

```py
$ http GET http://172.16.1.173:5000/
HTTP/1.0 200 OK
Content-Length: 17
Content-Type: text/html; charset=utf-8
Date: Wed, 22 Mar 2017 17:37:12 GMT
Server: Werkzeug/0.9.6 Python/3.5.2

Hello Networkers!
```

或者，您也可以使用 curl 的`-i`开关来查看 HTTP 头：`curl -i http://172.16.1.173:5000/`。

我们将在本章中使用`HTTPie`作为我们的客户端；值得花一两分钟来看一下它的用法。我们将使用免费的网站 HTTP Bin ([`httpbin.org/`](https://httpbin.org/)) 来展示`HTTPie`的用法。`HTTPie`的用法遵循这种简单的模式：

```py
$ http [flags] [METHOD] URL [ITEM]
```

按照前面的模式，`GET`请求非常简单，就像我们在 Flask 开发服务器中看到的那样：

```py
$ http GET https://httpbin.org/user-agent
...
{
 "user-agent": "HTTPie/0.8.0"
}
```

JSON 是`HTTPie`的默认隐式内容类型。如果您的 HTTP 主体只包含字符串，则不需要进行其他操作。如果您需要应用非字符串 JSON 字段，请使用`:=`或其他文档化的特殊字符：

```py
$ http POST https://httpbin.org/post name=eric twitter=at_ericchou married:=true 
HTTP/1.1 200 OK
...
Content-Type: application/json
...
{
 "headers": {
...
 "User-Agent": "HTTPie/0.8.0"
 },
 "json": {
 "married": true,
 "name": "eric",
 "twitter": "at_ericchou"
 },
 ...
 "url": "https://httpbin.org/post"
}
```

正如您所看到的，`HTTPie`是传统 curl 语法的一个重大改进，使得测试 REST API 变得轻而易举。

更多的用法示例可在[`httpie.org/doc#usage`](https://httpie.org/doc#usage)找到。

回到我们的 Flask 程序，API 构建的一个重要部分是基于 URL 路由的流程。让我们更深入地看一下`app.route()`装饰器。

# URL 路由

我们添加了两个额外的函数，并将它们与`chapter9_2.py`中的适当的`app.route()`路由配对：

```py
$ cat chapter9_2.py
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return 'You are at index()'

@app.route('/routers/')
def routers():
    return 'You are at routers()'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
```

结果是不同的端点传递给不同的函数。我们可以通过两个`http`请求来验证这一点：

```py
# Server
$ python chapter9_2.py

# Client
$ http GET http://172.16.1.173:5000/
...

You are at index()

$ http GET http://172.16.1.173:5000/routers/
...

You are at routers()
```

当然，如果我们一直保持静态，路由将会非常有限。有办法将变量从 URL 传递给 Flask；我们将在接下来的部分看一个例子。

# URL 变量

如前所述，我们也可以将变量传递给 URL，就像在`chapter9_3.py`中讨论的例子中看到的那样：

```py
...
@app.route('/routers/<hostname>')
def router(hostname):
    return 'You are at %s' % hostname

@app.route('/routers/<hostname>/interface/<int:interface_number>')
def interface(hostname, interface_number):
    return 'You are at %s interface %d' % (hostname, interface_number)
...
```

请注意，在`/routers/<hostname>` URL 中，我们将`<hostname>`变量作为字符串传递；`<int:interface_number>`将指定该变量应该是一个整数：

```py
$ http GET http://172.16.1.173:5000/routers/host1
...
You are at host1

$ http GET http://172.16.1.173:5000/routers/host1/interface/1
...
You are at host1 interface 1

# Throws exception
$ http GET http://172.16.1.173:5000/routers/host1/interface/one
HTTP/1.0 404 NOT FOUND
...
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<title>404 Not Found</title>
<h1>Not Found</h1>
<p>The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.</p>
```

转换器包括整数、浮点数和路径（它接受斜杠）。

除了匹配静态路由之外，我们还可以动态生成 URL。当我们事先不知道端点变量，或者端点基于其他条件，比如从数据库查询的值时，这是非常有用的。让我们看一个例子。

# URL 生成

在`chapter9_4.py`中，我们想要在代码中动态创建一个形式为`'/<hostname>/list_interfaces'`的 URL：

```py
from flask import Flask, url_for
...
@app.route('/<hostname>/list_interfaces')
def device(hostname):
    if hostname in routers:
        return 'Listing interfaces for %s' % hostname
    else:
        return 'Invalid hostname'

routers = ['r1', 'r2', 'r3']
for router in routers:
    with app.test_request_context():
        print(url_for('device', hostname=router))
...
```

执行后，您将得到一个漂亮而合乎逻辑的 URL，如下所示：

```py
(venv) $ python chapter9_4.py
/r1/list_interfaces
/r2/list_interfaces
/r3/list_interfaces
 * Running on http://0.0.0.0:5000/
 * Restarting with reloader 
```

目前，您可以将`app.text_request_context()`视为一个虚拟的`request`对象，这对于演示目的是必要的。如果您对本地上下文感兴趣，请随时查看[`werkzeug.pocoo.org/docs/0.14/local/`](http://werkzeug.pocoo.org/docs/0.14/local/)。

# jsonify 返回

Flask 中的另一个时间节省器是`jsonify()`返回，它包装了`json.dumps()`并将 JSON 输出转换为具有`application/json`作为 HTTP 标头中内容类型的`response`对象。我们可以稍微调整最后的脚本，就像我们将在`chapter9_5.py`中做的那样：

```py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/routers/<hostname>/interface/<int:interface_number>')
def interface(hostname, interface_number):
    return jsonify(name=hostname, interface=interface_number)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
```

我们将看到返回的结果作为`JSON`对象，并带有适当的标头：

```py
$ http GET http://172.16.1.173:5000/routers/r1/interface/1
HTTP/1.0 200 OK
Content-Length: 36
Content-Type: application/json
...

{
 "interface": 1,
 "name": "r1"
}
```

在 Flask 中查看了 URL 路由和`jsonify()`返回后，我们现在准备为我们的网络构建 API。

# 网络资源 API

通常，您的网络由一旦投入生产就不经常更改的网络设备组成。例如，您将拥有核心设备、分发设备、脊柱、叶子、顶部交换机等。每个设备都有特定的特性和功能，您希望将这些信息存储在一个持久的位置，以便以后可以轻松检索。通常是通过将数据存储在数据库中来实现的。但是，您通常不希望将其他用户直接访问数据库；他们也不想学习所有复杂的 SQL 查询语言。对于这种情况，我们可以利用 Flask 和 Flask-SQLAlchemy 扩展。

您可以在[`flask-sqlalchemy.pocoo.org/2.1/`](http://flask-sqlalchemy.pocoo.org/2.1/)了解更多关于 Flask-SQLAlchemy 的信息。

# Flask-SQLAlchemy

当然，SQLAlchemy 和 Flask 扩展都是数据库抽象层和对象关系映射器。这是一种使用`Python`对象作为数据库的高级方式。为了简化事情，我们将使用 SQLite 作为数据库，它是一个充当独立 SQL 数据库的平面文件。我们将查看`chapter9_db_1.py`的内容，作为使用 Flask-SQLAlchemy 创建网络数据库并将表条目插入数据库的示例。

首先，我们将创建一个 Flask 应用程序，并加载 SQLAlchemy 的配置，比如数据库路径和名称，然后通过将应用程序传递给它来创建`SQLAlchemy`对象：

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Create Flask application, load configuration, and create
# the SQLAlchemy object
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///network.db'
db = SQLAlchemy(app)
```

然后，我们可以创建一个`database`对象及其关联的主键和各种列：

```py
class Device(db.Model):
    __tablename__ = 'devices'
    id = db.Column(db.Integer, primary_key=True)
    hostname = db.Column(db.String(120), index=True)
    vendor = db.Column(db.String(40))

    def __init__(self, hostname, vendor):
        self.hostname = hostname
        self.vendor = vendor

    def __repr__(self):
        return '<Device %r>' % self.hostname
```

我们可以调用`database`对象，创建条目，并将它们插入数据库表中。请记住，我们添加到会话中的任何内容都需要提交到数据库中才能永久保存：

```py
if __name__ == '__main__':
    db.create_all()
    r1 = Device('lax-dc1-core1', 'Juniper')
    r2 = Device('sfo-dc1-core1', 'Cisco')
    db.session.add(r1)
    db.session.add(r2)
    db.session.commit()
```

我们将运行 Python 脚本并检查数据库文件是否存在：

```py
$ python chapter9_db_1.py
$ ls network.db
network.db
```

我们可以使用交互式提示来检查数据库表条目：

```py
>>> from flask import Flask
>>> from flask_sqlalchemy import SQLAlchemy
>>>
>>> app = Flask(__name__)
>>> app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///network.db'
>>> db = SQLAlchemy(app)
>>> from chapter9_db_1 import Device
>>> Device.query.all()
[<Device 'lax-dc1-core1'>, <Device 'sfo-dc1-core1'>]
>>> Device.query.filter_by(hostname='sfo-dc1-core1')
<flask_sqlalchemy.BaseQuery object at 0x7f1b4ae07eb8>
>>> Device.query.filter_by(hostname='sfo-dc1-core1').first()
<Device 'sfo-dc1-core1'>
```

我们也可以以相同的方式创建新条目：

```py
>>> r3 = Device('lax-dc1-core2', 'Juniper')
>>> db.session.add(r3)
>>> db.session.commit()
>>> Device.query.all()
[<Device 'lax-dc1-core1'>, <Device 'sfo-dc1-core1'>, <Device 'lax-dc1-core2'>]
```

# 网络内容 API

在我们深入代码之前，让我们花一点时间考虑我们要创建的 API。规划 API 通常更多是一种艺术而不是科学；这确实取决于您的情况和偏好。我建议的下一步绝不是正确的方式，但是现在，为了开始，跟着我走。

回想一下，在我们的图表中，我们有四个 Cisco IOSv 设备。假设其中两个，`iosv-1`和`iosv-2`，是网络角色的脊柱。另外两个设备，`iosv-3`和`iosv-4`，在我们的网络服务中作为叶子。这显然是任意选择，可以稍后修改，但重点是我们想要提供关于我们的网络设备的数据，并通过 API 公开它们。

为了简化事情，我们将创建两个 API：设备组 API 和单个设备 API：

![](img/0cf9e61a-6b19-4746-b968-9ef5830d08ab.png)网络内容 API

第一个 API 将是我们的`http://172.16.1.173/devices/`端点，支持两种方法：`GET`和`POST`。`GET`请求将返回当前设备列表，而带有适当 JSON 主体的`POST`请求将创建设备。当然，您可以选择为创建和查询设置不同的端点，但在这个设计中，我们选择通过 HTTP 方法来区分这两种情况。

第二个 API 将特定于我们的设备，形式为`http://172.16.1.173/devices/<device id>`。带有`GET`请求的 API 将显示我们输入到数据库中的设备的详细信息。`PUT`请求将修改更新条目。请注意，我们使用`PUT`而不是`POST`。这是 HTTP API 使用的典型方式；当我们需要修改现有条目时，我们将使用`PUT`而不是`POST`。

到目前为止，您应该对您的 API 的外观有一个很好的想法。为了更好地可视化最终结果，我将快速跳转并展示最终结果，然后再看代码。

对`/devices/`API 的`POST`请求将允许您创建一个条目。在这种情况下，我想创建我们的网络设备，其属性包括主机名、回环 IP、管理 IP、角色、供应商和运行的操作系统：

```py
$ http POST http://172.16.1.173:5000/devices/ 'hostname'='iosv-1' 'loopback'='192.168.0.1' 'mgmt_ip'='172.16.1.225' 'role'='spine' 'vendor'='Cisco' 'os'='15.6'
HTTP/1.0 201 CREATED
Content-Length: 2
Content-Type: application/json
Date: Fri, 24 Mar 2017 01:45:15 GMT
Location: http://172.16.1.173:5000/devices/1
Server: Werkzeug/0.9.6 Python/3.5.2

{}
```

我可以重复前面的步骤来添加另外三个设备：

```py
$ http POST http://172.16.1.173:5000/devices/ 'hostname'='iosv-2' 'loopback'='192.168.0.2' 'mgmt_ip'='172.16.1.226' 'role'='spine' 'vendor'='Cisco' 'os'='15.6'
...
$ http POST http://172.16.1.173:5000/devices/ 'hostname'='iosv-3', 'loopback'='192.168.0.3' 'mgmt_ip'='172.16.1.227' 'role'='leaf' 'vendor'='Cisco' 'os'='15.6'
...
$ http POST http://172.16.1.173:5000/devices/ 'hostname'='iosv-4', 'loopback'='192.168.0.4' 'mgmt_ip'='172.16.1.228' 'role'='leaf' 'vendor'='Cisco' 'os'='15.6'
```

如果我们可以使用相同的 API 和`GET`请求，我们将能够看到我们创建的网络设备列表：

```py
$ http GET http://172.16.1.173:5000/devices/
HTTP/1.0 200 OK
Content-Length: 188
Content-Type: application/json
Date: Fri, 24 Mar 2017 01:53:15 GMT
Server: Werkzeug/0.9.6 Python/3.5.2

{
 "device": [
 "http://172.16.1.173:5000/devices/1",
 "http://172.16.1.173:5000/devices/2",
 "http://172.16.1.173:5000/devices/3",
 "http://172.16.1.173:5000/devices/4"
 ]
}
```

类似地，使用`GET`请求对`/devices/<id>`将返回与设备相关的特定信息：

```py
$ http GET http://172.16.1.173:5000/devices/1
HTTP/1.0 200 OK
Content-Length: 188
Content-Type: application/json
...
{
 "hostname": "iosv-1",
 "loopback": "192.168.0.1",
 "mgmt_ip": "172.16.1.225",
 "os": "15.6",
 "role": "spine",
 "self_url": "http://172.16.1.173:5000/devices/1",
 "vendor": "Cisco"
}
```

假设我们将`r1`操作系统从`15.6`降级到`14.6`。我们可以使用`PUT`请求来更新设备记录：

```py
$ http PUT http://172.16.1.173:5000/devices/1 'hostname'='iosv-1' 'loopback'='192.168.0.1' 'mgmt_ip'='172.16.1.225' 'role'='spine' 'vendor'='Cisco' 'os'='14.6'
HTTP/1.0 200 OK

# Verification
$ http GET http://172.16.1.173:5000/devices/1
...
{
 "hostname": "r1",
 "loopback": "192.168.0.1",
 "mgmt_ip": "172.16.1.225",
 "os": "14.6",
 "role": "spine",
 "self_url": "http://172.16.1.173:5000/devices/1",
 "vendor": "Cisco"
}
```

现在，让我们看一下`chapter9_6.py`中的代码，这些代码帮助创建了前面的 API。在我看来，很酷的是，所有这些 API 都是在单个文件中完成的，包括数据库交互。以后，当我们需要扩展现有的 API 时，我们总是可以将组件分离出来，比如为数据库类单独创建一个文件。

# 设备 API

`chapter9_6.py`文件以必要的导入开始。请注意，以下请求导入是来自客户端的`request`对象，而不是我们在之前章节中使用的 requests 包：

```py
from flask import Flask, url_for, jsonify, request
from flask_sqlalchemy import SQLAlchemy
# The following is deprecated but still used in some examples
# from flask.ext.sqlalchemy import SQLAlchemy
```

我们声明了一个`database`对象，其`id`为主键，`hostname`、`loopback`、`mgmt_ip`、`role`、`vendor`和`os`为字符串字段：

```py
class Device(db.Model):
    __tablename__ = 'devices'
  id = db.Column(db.Integer, primary_key=True)
    hostname = db.Column(db.String(64), unique=True)
    loopback = db.Column(db.String(120), unique=True)
    mgmt_ip = db.Column(db.String(120), unique=True)
    role = db.Column(db.String(64))
    vendor = db.Column(db.String(64))
    os = db.Column(db.String(64))
```

`get_url()`函数从`url_for()`函数返回一个 URL。请注意，调用的`get_device()`函数尚未在`'/devices/<int:id>'`路由下定义：

```py
def get_url(self):
    return url_for('get_device', id=self.id, _external=True)
```

`export_data()`和`import_data()`函数是彼此的镜像。一个用于从数据库获取信息到用户（`export_data()`），当我们使用`GET`方法时。另一个用于将用户的信息放入数据库（`import_data()`），当我们使用`POST`或`PUT`方法时：

```py
def export_data(self):
    return {
        'self_url': self.get_url(),
  'hostname': self.hostname,
  'loopback': self.loopback,
  'mgmt_ip': self.mgmt_ip,
  'role': self.role,
  'vendor': self.vendor,
  'os': self.os
    }

def import_data(self, data):
    try:
        self.hostname = data['hostname']
        self.loopback = data['loopback']
        self.mgmt_ip = data['mgmt_ip']
        self.role = data['role']
        self.vendor = data['vendor']
        self.os = data['os']
    except KeyError as e:
        raise ValidationError('Invalid device: missing ' + e.args[0])
    return self
```

有了`database`对象以及创建的导入和导出函数，设备操作的 URL 分发就变得简单了。`GET`请求将通过查询设备表中的所有条目返回设备列表，并返回每个条目的 URL。`POST`方法将使用全局`request`对象作为输入，使用`import_data()`函数，然后将设备添加到数据库并提交信息：

```py
@app.route('/devices/', methods=['GET'])
def get_devices():
    return jsonify({'device': [device.get_url() 
                              for device in Device.query.all()]})

@app.route('/devices/', methods=['POST'])
def new_device():
    device = Device()
    device.import_data(request.json)
    db.session.add(device)
    db.session.commit()
    return jsonify({}), 201, {'Location': device.get_url()}
```

如果您查看`POST`方法，返回的主体是一个空的 JSON 主体，状态码为`201`（已创建），以及额外的标头：

```py
HTTP/1.0 201 CREATED
Content-Length: 2
Content-Type: application/json
Date: ...
Location: http://172.16.1.173:5000/devices/4
Server: Werkzeug/0.9.6 Python/3.5.2
```

让我们来看一下查询和返回有关单个设备的信息的 API。

# 设备 ID API

单个设备的路由指定 ID 应该是一个整数，这可以作为我们对错误请求的第一道防线。这两个端点遵循与我们的`/devices/`端点相同的设计模式，我们在这里使用相同的`import`和`export`函数：

```py
@app.route('/devices/<int:id>', methods=['GET'])
def get_device(id):
    return jsonify(Device.query.get_or_404(id).export_data())

@app.route('/devices/<int:id>', methods=['PUT'])
def edit_device(id):
    device = Device.query.get_or_404(id)
    device.import_data(request.json)
    db.session.add(device)
    db.session.commit()
    return jsonify({})
```

注意`query_or_404()`方法；如果数据库查询对传入的 ID 返回负值，它提供了一个方便的方法来返回`404（未找到）`。这是一个相当优雅的方式来快速检查数据库查询。

最后，代码的最后部分创建数据库表并启动 Flask 开发服务器：

```py
if __name__ == '__main__':
    db.create_all()
    app.run(host='0.0.0.0', debug=True)
```

这是本书中较长的 Python 脚本之一，这就是为什么我们花了更多的时间详细解释它。该脚本提供了一种说明我们如何利用后端数据库来跟踪网络设备，并将它们仅作为 API 暴露给外部世界的方法，使用 Flask。

在下一节中，我们将看看如何使用 API 对单个设备或一组设备执行异步任务。

# 网络动态操作

我们的 API 现在可以提供关于网络的静态信息；我们可以将数据库中存储的任何内容返回给请求者。如果我们可以直接与我们的网络交互，比如查询设备信息或向设备推送配置更改，那将是很棒的。

我们将通过利用我们已经在第二章中看到的脚本，*低级网络设备交互*，来开始这个过程，通过 Pexpect 与设备进行交互。我们将稍微修改脚本，将其转换为一个我们可以在`chapter9_pexpect_1.py`中重复使用的函数：

```py
# We need to install pexpect for our virtual env
$ pip install pexpect

$ cat chapter9_pexpect_1.py
import pexpect

def show_version(device, prompt, ip, username, password):
 device_prompt = prompt
 child = pexpect.spawn('telnet ' + ip)
 child.expect('Username:')
 child.sendline(username)
 child.expect('Password:')
 child.sendline(password)
 child.expect(device_prompt)
 child.sendline('show version | i V')
 child.expect(device_prompt)
 result = child.before
 child.sendline('exit')
 return device, result
```

我们可以通过交互式提示来测试新的函数：

```py
$ pip3 install pexpect
$ python
>>> from chapter9_pexpect_1 import show_version
>>> print(show_version('iosv-1', 'iosv-1#', '172.16.1.225', 'cisco', 'cisco'))
('iosv-1', b'show version | i V\r\nCisco IOS Software, IOSv Software (VIOS-ADVENTERPRISEK9-M), Version 15.6(3)M2, RELEASE SOFTWARE (fc2)\r\n')
>>> 
```

确保您的 Pexpect 脚本在继续之前能够正常工作。以下代码假定您已经输入了前一节中的必要数据库信息。

我们可以在`chapter9_7.py`中添加一个新的 API 来查询设备版本：

```py
from chapter9_pexpect_1 import show_version
...
@app.route('/devices/<int:id>/version', methods=['GET'])
def get_device_version(id):
    device = Device.query.get_or_404(id)
    hostname = device.hostname
    ip = device.mgmt_ip
    prompt = hostname+"#"
  result = show_version(hostname, prompt, ip, 'cisco', 'cisco')
    return jsonify({"version": str(result)})
```

结果将返回给请求者：

```py
$ http GET http://172.16.1.173:5000/devices/4/version
HTTP/1.0 200 OK
Content-Length: 210
Content-Type: application/json
Date: Fri, 24 Mar 2017 17:05:13 GMT
Server: Werkzeug/0.9.6 Python/3.5.2

{
 "version": "('iosv-4', b'show version | i V\r\nCisco IOS Software, IOSv Software (VIOS-ADVENTERPRISEK9-M), Version 15.6(2)T, RELEASE SOFTWARE (fc2)\r\nProcessor board ID 9U96V39A4Z12PCG4O6Y0Q\r\n')"
}
```

我们还可以添加另一个端点，允许我们根据它们的共同字段对多个设备执行批量操作。在下面的示例中，端点将在 URL 中获取`device_role`属性，并将其与相应的设备匹配：

```py
@app.route('/devices/<device_role>/version', methods=['GET'])
def get_role_version(device_role):
    device_id_list = [device.id for device in Device.query.all() if device.role == device_role]
    result = {}
    for id in device_id_list:
        device = Device.query.get_or_404(id)
        hostname = device.hostname
        ip = device.mgmt_ip
        prompt = hostname + "#"
  device_result = show_version(hostname, prompt, ip, 'cisco', 'cisco')
        result[hostname] = str(device_result)
    return jsonify(result)
```

当然，像在前面的代码中那样循环遍历所有的设备`Device.query.all()`是不高效的。在生产中，我们将使用一个专门针对设备角色的 SQL 查询。

当我们使用 REST API 时，可以同时查询所有的骨干和叶子设备：

```py
$ http GET http://172.16.1.173:5000/devices/spine/version
HTTP/1.0 200 OK
...
{
 "iosv-1": "('iosv-1', b'show version | i V\r\nCisco IOS Software, IOSv Software (VIOS-ADVENTERPRISEK9-M), Version 15.6(2)T, RELEASE SOFTWARE (fc2)\r\n')",
 "iosv-2": "('iosv-2', b'show version | i V\r\nCisco IOS Software, IOSv Software (VIOS-ADVENTERPRISEK9-M), Version 15.6(2)T, RELEASE SOFTWARE (fc2)\r\nProcessor board ID 9T7CB2J2V6F0DLWK7V48E\r\n')"
}

$ http GET http://172.16.1.173:5000/devices/leaf/version
HTTP/1.0 200 OK
...
{
 "iosv-3": "('iosv-3', b'show version | i V\r\nCisco IOS Software, IOSv Software (VIOS-ADVENTERPRISEK9-M), Version 15.6(2)T, RELEASE SOFTWARE (fc2)\r\nProcessor board ID 9MGG8EA1E0V2PE2D8KDD7\r\n')",
 "iosv-4": "('iosv-4', b'show version | i V\r\nCisco IOS Software, IOSv Software (VIOS-ADVENTERPRISEK9-M), Version 15.6(2)T, RELEASE SOFTWARE (fc2)\r\nProcessor board ID 9U96V39A4Z12PCG4O6Y0Q\r\n')"
}
```

正如所示，新的 API 端点实时查询设备，并将结果返回给请求者。当您可以保证在事务的超时值（默认为 30 秒）内获得操作的响应，或者如果您可以接受 HTTP 会话在操作完成之前超时，这种方法相对有效。解决超时问题的一种方法是异步执行任务。我们将在下一节中看看如何做到这一点。

# 异步操作

在我看来，异步操作是 Flask 的一个高级主题。幸运的是，Miguel Grinberg（[`blog.miguelgrinberg.com/`](https://blog.miguelgrinberg.com/)）是我非常喜欢的 Flask 工作的作者，他在博客和 GitHub 上提供了许多帖子和示例。对于异步操作，`chapter9_8.py`中的示例代码引用了 Miguel 在 GitHub 上的`Raspberry Pi`文件上的代码（[`github.com/miguelgrinberg/oreilly-flask-apis-video/blob/master/camera/camera.py`](https://github.com/miguelgrinberg/oreilly-flask-apis-video/blob/master/camera/camera.py)）来使用 background 装饰器。我们将开始导入一些额外的模块：

```py
from flask import Flask, url_for, jsonify, request,
    make_response, copy_current_request_context
...
import uuid
import functools
from threading import Thread
```

background 装饰器接受一个函数，并使用线程和 UUID 作为任务 ID 在后台运行它。它返回状态码`202` accepted 和新资源的位置，供请求者检查。我们将创建一个新的 URL 用于状态检查：

```py
@app.route('/status/<id>', methods=['GET'])
def get_task_status(id):   global background_tasks
    rv = background_tasks.get(id)
    if rv is None:
        return not_found(None)
   if isinstance(rv, Thread):
        return jsonify({}), 202, {'Location': url_for('get_task_status', id=id)}
   if app.config['AUTO_DELETE_BG_TASKS']:
        del background_tasks[id]
    return rv
```

一旦我们检索到资源，它就会被删除。这是通过在应用程序顶部将`app.config['AUTO_DELETE_BG_TASKS']`设置为`true`来完成的。我们将在我们的版本端点中添加这个装饰器，而不改变代码的其他部分，因为所有的复杂性都隐藏在装饰器中（这多酷啊！）：

```py
@app.route('/devices/<int:id>/version', methods=['GET'])
@**background** def get_device_version(id):
    device = Device.query.get_or_404(id)
...

@app.route('/devices/<device_role>/version', methods=['GET'])
@**background** def get_role_version(device_role):
    device_id_list = [device.id for device in Device.query.all() if device.role == device_role]
...
```

最终结果是一个两部分的过程。我们将为端点执行`GET`请求，并接收位置头：

```py
$ http GET http://172.16.1.173:5000/devices/spine/version
HTTP/1.0 202 ACCEPTED
Content-Length: 2
Content-Type: application/json
Date: <skip>
Location: http://172.16.1.173:5000/status/d02c3f58f4014e96a5dca075e1bb65d4
Server: Werkzeug/0.9.6 Python/3.5.2

{}
```

然后我们可以发出第二个请求以检索结果的位置：

```py
$ http GET http://172.16.1.173:5000/status/d02c3f58f4014e96a5dca075e1bb65d4
HTTP/1.0 200 OK
Content-Length: 370
Content-Type: application/json
Date: <skip>
Server: Werkzeug/0.9.6 Python/3.5.2

{
 "iosv-1": "('iosv-1', b'show version | i V\r\nCisco IOS Software, IOSv Software (VIOS-ADVENTERPRISEK9-M), Version 15.6(2)T, RELEASE SOFTWARE (fc2)\r\n')",
 "iosv-2": "('iosv-2', b'show version | i V\r\nCisco IOS Software, IOSv Software (VIOS-ADVENTERPRISEK9-M), Version 15.6(2)T, RELEASE SOFTWARE (fc2)\r\nProcessor board ID 9T7CB2J2V6F0DLWK7V48E\r\n')"
}
```

为了验证当资源尚未准备好时是否返回状态码`202`，我们将使用以下脚本`chapter9_request_1.py`立即向新资源发出请求：

```py
import requests, time

server = 'http://172.16.1.173:5000' endpoint = '/devices/1/version'   # First request to get the new resource r = requests.get(server+endpoint)
resource = r.headers['location']
print("Status: {} Resource: {}".format(r.status_code, resource))

# Second request to get the resource status r = requests.get(resource)
print("Immediate Status Query to Resource: " + str(r.status_code))

print("Sleep for 2 seconds")
time.sleep(2)
# Third request to get the resource status r = requests.get(resource)
print("Status after 2 seconds: " + str(r.status_code))
```

如您在结果中所见，当资源仍在后台运行时，状态码以`202`返回：

```py
$ python chapter9_request_1.py
Status: 202 Resource: http://172.16.1.173:5000/status/1de21f5235c94236a38abd5606680b92
Immediate Status Query to Resource: 202
Sleep for 2 seconds
Status after 2 seconds: 200
```

我们的 API 正在很好地进行中！因为我们的网络资源对我们很有价值，所以我们应该只允许授权人员访问 API。我们将在下一节为我们的 API 添加基本的安全措施。

# 安全

对于用户身份验证安全，我们将使用 Flask 的`httpauth`扩展，由 Miguel Grinberg 编写，以及 Werkzeug 中的密码函数。`httpauth`扩展应该已经作为`requirements.txt`安装的一部分。展示安全功能的新文件名为`chapter9_9.py`；我们将从几个模块导入开始：

```py
...
from werkzeug.security import generate_password_hash, check_password_hash
from flask.ext.httpauth import HTTPBasicAuth
...
```

我们将创建一个`HTTPBasicAuth`对象以及`用户数据库`对象。请注意，在用户创建过程中，我们将传递密码值；但是，我们只存储`password_hash`而不是密码本身。这确保我们不会为用户存储明文密码：

```py
auth = HTTPBasicAuth()

class User(db.Model):
    __tablename__ = 'users'
  id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
```

`auth`对象有一个`verify_password`装饰器，我们可以使用它，以及 Flask 的`g`全局上下文对象，该对象在请求开始时创建，用于密码验证。因为`g`是全局的，如果我们将用户保存到`g`变量中，它将在整个事务中存在：

```py
@auth.verify_password def verify_password(username, password):
    g.user = User.query.filter_by(username=username).first()
    if g.user is None:
        return False
 return g.user.verify_password(password)
```

有一个方便的`before_request`处理程序，可以在调用任何 API 端点之前使用。我们将结合`auth.login_required`装饰器和`before_request`处理程序，将其应用于所有 API 路由：

```py
@app.before_request @auth.login_required def before_request():
    pass 
```

最后，我们将使用`未经授权`错误处理程序返回`401`未经授权错误的`response`对象：

```py
@auth.error_handler def unauthorized():
    response = jsonify({'status': 401, 'error': 'unauthorized', 
 'message': 'please authenticate'})

    response.status_code = 401
  return response
```

在我们测试用户身份验证之前，我们需要在我们的数据库中创建用户：

```py
>>> from chapter9_9 import db, User
>>> db.create_all()
>>> u = User(username='eric')
>>> u.set_password('secret')
>>> db.session.add(u)
>>> db.session.commit()
>>> exit()
```

一旦启动 Flask 开发服务器，请尝试发出请求，就像我们之前做的那样。您应该看到，这次服务器将以`401`未经授权的错误拒绝请求：

```py
$ http GET http://172.16.1.173:5000/devices/
HTTP/1.0 401 UNAUTHORIZED
Content-Length: 81
Content-Type: application/json
Date: <skip>
Server: Werkzeug/0.9.6 Python/3.5.2
WWW-Authenticate: Basic realm="Authentication Required"

{
 "error": "unauthorized",
 "message": "please authenticate",
 "status": 401
}
```

现在我们需要为我们的请求提供身份验证头：

```py
$ http --auth eric:secret GET http://172.16.1.173:5000/devices/
HTTP/1.0 200 OK
Content-Length: 188
Content-Type: application/json
Date: <skip>
Server: Werkzeug/0.9.6 Python/3.5.2

{
 "device": [
 "http://172.16.1.173:5000/devices/1",
 "http://172.16.1.173:5000/devices/2",
 "http://172.16.1.173:5000/devices/3",
 "http://172.16.1.173:5000/devices/4"
 ]
}
```

现在我们已经为我们的网络设置了一个不错的 RESTful API。用户现在可以与 API 交互，而不是与网络设备。他们可以查询网络的静态内容，并为单个设备或一组设备执行任务。我们还添加了基本的安全措施，以确保只有我们创建的用户能够从我们的 API 中检索信息。很酷的是，这一切都在不到 250 行代码的单个文件中完成了（如果减去注释，不到 200 行）！

我们现在已经将底层供应商 API 从我们的网络中抽象出来，并用我们自己的 RESTful API 替换了它们。我们可以在后端自由使用所需的内容，比如 Pexpect，同时为我们的请求者提供统一的前端。

让我们看看 Flask 的其他资源，这样我们就可以继续构建我们的 API 框架。

# 其他资源

毫无疑问，Flask 是一个功能丰富的框架，功能和社区都在不断增长。在本章中，我们涵盖了许多主题，但我们仍然只是触及了框架的表面。除了 API，你还可以将 Flask 用于 Web 应用程序以及你的网站。我认为我们的网络 API 框架仍然有一些改进的空间：

+   将数据库和每个端点分开放在自己的文件中，以使代码更清晰，更易于故障排除。

+   从 SQLite 迁移到其他适用于生产的数据库。

+   使用基于令牌的身份验证，而不是为每个交易传递用户名和密码。实质上，我们将在初始身份验证时收到一个具有有限过期时间的令牌，并在之后的交易中使用该令牌，直到过期。

+   将 Flask API 应用程序部署在生产 Web 服务器后面，例如 Nginx，以及 Python WSGI 服务器用于生产环境。

+   使用自动化过程控制系统，如 Supervisor ([`supervisord.org/`](http://supervisord.org/))，来控制 Nginx 和 Python 脚本。

显然，推荐的改进选择会因公司而异。例如，数据库和 Web 服务器的选择可能会对公司的技术偏好以及其他团队的意见产生影响。如果 API 仅在内部使用，并且已经采取了其他形式的安全措施，那么使用基于令牌的身份验证可能并不必要。因此，出于这些原因，我想为您提供额外的链接作为额外资源，以便您选择继续使用前述任何项目。

以下是一些我认为在考虑设计模式、数据库选项和一般 Flask 功能时有用的链接：

+   Flask 设计模式的最佳实践: [`flask.pocoo.org/docs/0.10/patterns/`](http://flask.pocoo.org/docs/0.10/patterns/)

+   Flask API: [`flask.pocoo.org/docs/0.12/api/`](http://flask.pocoo.org/docs/0.12/api/)

+   部署选项: [`flask.pocoo.org/docs/0.12/deploying/`](http://flask.pocoo.org/docs/0.12/deploying/)

由于 Flask 的性质以及它依赖于其小核心之外的扩展，有时你可能会发现自己从一个文档跳到另一个文档。这可能令人沮丧，但好处是你只需要了解你正在使用的扩展，我觉得这在长远来看节省了时间。

# 摘要

在本章中，我们开始着手构建网络的 REST API。我们研究了不同流行的 Python Web 框架，即 Django 和 Flask，并对比了两者。选择 Flask，我们能够从小处着手，并通过使用 Flask 扩展来扩展功能。

在我们的实验室中，我们使用虚拟环境将 Flask 安装基础与全局 site-packages 分开。实验室网络由四个节点组成，其中两个被指定为脊柱路由器，另外两个被指定为叶子路由器。我们对 Flask 的基础知识进行了介绍，并使用简单的 HTTPie 客户端来测试我们的 API 设置。

在 Flask 的不同设置中，我们特别强调了 URL 分发以及 URL 变量，因为它们是请求者和我们的 API 系统之间的初始逻辑。我们研究了使用 Flask-SQLAlchemy 和 SQLite 来存储和返回静态网络元素。对于操作任务，我们还创建了 API 端点，同时调用其他程序，如 Pexpect，来完成配置任务。我们通过添加异步处理和用户身份验证来改进 API 的设置。在本章的最后，我们还查看了一些额外的资源链接，以便添加更多安全性和其他功能。

在第十章中，*AWS 云网络*，我们将转向使用**Amazon Web Services**（**AWS**）进行云网络的研究。
