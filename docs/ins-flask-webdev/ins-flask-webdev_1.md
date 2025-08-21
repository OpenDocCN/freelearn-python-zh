# 第一章：即时 Flask 网络开发

欢迎来到*即时 Flask 网络开发*。我们将逐步学习使用 Flask 这种小而富有表现力的框架在 Python 编程语言中进行网络开发，该框架提供了基本功能，并使您能够构建自己的代码模式。我们将构建一个简单的日程安排应用程序来跟踪约会，包括数据库和用户界面，我们将一步一步地构建。我们将从底层构建我们的应用程序，您将在后面的章节中看到一切是如何组合在一起的。这种自下而上的方法将为您提供使用 Flask 扩展项目所需的基本组件。

# 准备开发（简单）

我们将在 Python 中设置开发环境，开始探索 Flask 网络编程。

## 准备工作

转到[python.org](http://python.org)获取 Python。我们的应用程序将在 Python 2.7 和 Python 3.3+上运行。一旦您安装了 Python，您就不需要对开发机器进行管理访问，甚至可以根据[python.org](http://python.org)上的安装说明仅为您的用户安装 Python。

## 如何做...

Python 项目可以使用**virtualenv**来管理软件包，这是机器上专门用于特定项目的用户可写区域。您可以在 Unix/类 Unix 系统、Mac OS X 和 Windows 上使用 virtualenv。

我们将使用命令行来启动 virtualenv，然后讨论您在开发环境中需要的工具。在开始之前，请注意*常见错误及其解决方法*小节中的*更多信息*部分，以防遇到问题。

类 Unix 系统附带一个终端仿真器，我们将在本书中使用。在 Unix、类 Unix 系统和 Mac OS X 上安装 Flask 所需的步骤如下：

1.  在您希望项目位于的目录中打开一个终端。

1.  从`pypi.python.org/pypi/virtualenv/1.9.1`下载 virtualenv。在终端中，您可以使用`curl -O https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.9.1.tar.gz`来完成这个步骤。

1.  使用`tar xvzf virtualenv-1.9.1.tar.gz`解压 virtualenv。

1.  使用`python virtualenv-1.9.1/virtualenv.py env`创建一个名为`env`的 virtualenv 工具。您可以使用任何名称；只需根据您选择的名称更改这里的命令。

1.  使用`. env/bin/activate`激活 virtualenv 工具。

1.  使用`pip install Flask`安装 Flask。

1.  使用简单的烟雾测试`python -m flask.config`验证 Flask 是否已安装。如果已安装 Flask，则不会显示任何内容。

您可以使用`deactivate`关闭 virtualenv，以便继续在终端上使用其他项目。每当您恢复对 Flask 项目的工作时，都要使用`. path/to/env/bin/activate`再次激活 virtualenv。`.`命令会源化激活脚本，设置指向 virtualenv 中的`python`可执行文件的环境变量，并启用在那里安装的软件包的导入。

对于 Windows，我们将使用`cmd.exe`。在 Windows 上安装 Flask 的步骤如下：

1.  确保 Python 安装及其`scripts`目录（默认为`C:\Python33\`和`C:\Python33\Scripts\`）在您当前的`PATH`变量中。如果您使用 Python 2.7，则使用`Python27`，请注意 Windows 在`PATH`中的目录之间使用分号。使用`echo %PATH%`查看您的`PATH`变量。通常，通过导航到**计算机** | **属性** | **高级** | **环境变量**来编辑`PATH`。

1.  在您希望项目位于的目录中打开一个 shell。从历史上看，您可能会遇到在文件路径中存在空格时运行失败的脚本，这就是为什么`C:\Python27\`比`C:\Program Files\Python27\`更受青睐；如果您遇到奇怪的错误，请记住这一点。

1.  从[`raw.github.com/pypa/virtualenv/1.9.1/virtualenv.py`](https://raw.github.com/pypa/virtualenv/1.9.1/virtualenv.py)下载`virtualenv.py`到相同的目录。

1.  使用`python virtualenv.py env`创建名为`env`的虚拟环境。您可以使用任何名称；只需确保根据您选择的名称更改这里的命令。

1.  使用`env\Scripts\activate.bat`激活 virtualenv。

1.  使用`pip install Flask`安装 Flask。

1.  使用简单的烟雾测试`python -m flask.config`验证 Flask 是否已安装。如果已安装 Flask，则不会显示任何内容。

您可以使用`deactivate`关闭 virtualenv 工具，以便继续在命令提示符中使用其他项目。每当您恢复对 Flask 项目的工作时，再次使用`env\Scripts\activate.bat`激活 virtualenv 工具，这是一个批处理文件，用于设置指向 virtualenv 中的 python 可执行文件的环境变量，并启用在其中安装的软件包的导入。

### 提示

如果您使用 Windows，可以参考[technet.microsoft.com/en-us/library/bb978526.aspx](http://technet.microsoft.com/en-us/library/bb978526.aspx)使用 PowerShell 获得更丰富的命令行环境，或者如果您无法访问 PowerShell，则可以使用`cmd.exe`，网址为[docs.python.org/2/using/windows.html](http://docs.python.org/2/using/windows.html)。

## 它是如何工作的...

Python 本身附带了一个大量的实用程序集合。使用 Python 即可立即使用[docs.python.org/2/library/](http://docs.python.org/2/library/)中列出的任何代码。Python 解释器具有几个始终可用的函数和对象，以及几个通过导入语句显示的其他函数和对象，如下所示：

```py
import antigravity
```

Python 社区通过在 Python 软件包索引 PyPI（发音为"pie p.i."）上发布的框架（包括 Flask）和工具包的合作，添加了更多要导入的模块，网址为[pypi.python.org](http://pypi.python.org)。打包变得复杂，pip 和 virtualenv 旨在使在 PyPI 上使用第三方软件包的工作流程变得可管理。

## 还有更多...

还有更多可以实现的内容：

### 开发环境清单

以下是开发 Web 应用程序所需的项目：

+   用于添加和编辑源代码的文本编辑器。我使用 emacs 和 vim，它们具有高效的键盘快捷键，Sublime Text 是一个受欢迎的选择。有许多选择；只需确保编辑器是为代码设计的。

+   使用本章中的说明，带有虚拟环境工具的 Python。

+   一个终端控制台用于运行您的 Flask 代码，每个运行的 Flask 应用程序一个，如下一节所述。

+   一个终端控制台用于运行 Python 的交互式解释器。在这里，您可以查看 API，尝试代码，验证单元功能，并使用 Python 内置的`help(arg)`函数获取在线文档。

+   一个现代的 Web 浏览器。Firefox 和 Chrome 具有方便访问的 JavaScript 控制台和网页文档检查器。JavaScript 控制台和文档检查器对于 Web 用户界面开发至关重要。

+   Flask 文档可在[flask.pocoo.org/docs/](http://flask.pocoo.org/docs/)上获得。

+   Werkzeug 文档可在[werkzeug.pocoo.org/docs/](http://werkzeug.pocoo.org/docs/)上获得。Werkzeug 是 Flask 的底层 Web 服务实现。

+   Jinja 文档可在[jinja.pocoo.org/docs/](http://jinja.pocoo.org/docs/)上获得。Jinja 是 Flask 的默认模板引擎。

+   Python 文档可在[docs.python.org/2/](http://docs.python.org/2/)上获得。

+   版本控制系统。跟踪对项目的更改。我个人推荐 git，可在[git-scm.com](http://git-scm.com)上获得。

集成开发环境（IDE）可以将这些工具捆绑在一个应用程序中，Python 有几种可用的 IDE。我个人发现清单中每一项的最佳工具，并将它们组合成一个独立工具和应用程序的集合作为我的环境。我强烈建议拥有一个独立的命令行，以便进行代码交互，这样即使您喜欢的 IDE，也可以随时访问代码的核心。

（我发现自己在命令行和终端上做任何事情都更有效率，但这是个人偏好的问题。一定要喜欢您的开发环境，这样您才能清晰地思考您的工作。）

### 常见错误及其解决方法

在使用 Python 虚拟环境时，您会遇到两个常见错误：

+   尝试运行 Python 时找不到`python`命令

+   在您的代码中尝试导入 Flask 时出现“没有名为 flask 的模块”

这两个问题都是由于命令行会话中的一个名为**PATH**的环境变量设置不正确引起的，它是一个有序的目录列表，系统在加载命令时应该搜索这些目录。如果找不到`python`，则在安装 Python 时未更新 PATH，应该重新安装 Python。如果找不到`flask`，则要么没有激活虚拟环境，要么没有安装 Flask，应该重新查看安装说明。

# 运行一个简单的应用程序（简单）

让我们开始运行一个最简单的 Flask 应用程序。

## 准备工作

确保您已激活虚拟环境工具，如前一节所述。

## 如何做到...

1.  将此代码放入名为`hello.py`的文件中：

```py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, world!'

if __name__ == '__main__':
    app.run()
```

1.  从控制台运行此应用程序：

```py
$ python hello.py
```

1.  如果这个运行正确，您将看到：

*** 在 http://127.0.0.1:5000/ 上运行**

这是一个 URL，您可以在浏览器中访问。当您这样做时，您的浏览器将显示一个简单的页面，上面写着“**Hello, world!**”（不带引号）。请注意，地址`127.0.0.1`指的是本地主机，因此 Flask 只会响应来自运行代码的主机的请求。

## 它是如何工作的...

逐行来看，这段代码做了什么：

```py
app = Flask(__name__)
```

这将创建一个名为`app`的 Python 对象，它是一个**WSGI**应用程序。WSGI 代表**Web Service Gateway Interface**，是 Python 社区托管 Web 项目的标准接口。您找到的任何 WSGI 资源或工具，都可以应用到这个`app`对象上，包括 WSGI 中间件（如果这样做，请包装`app.wsgi_app`）和 HTTP 服务器，如 gunicorn 或带有`mod_wsgi`的 Apache `httpd`。

传递给`Flask`类的参数告诉 Flask 如何找到与您的应用关联的资源：静态文件和模板。传递`__name__`告诉 Flask 查看当前的 Python 模块——包含此代码的`.py`文件。大多数应用程序将使用`__name__`；如果您怀疑`__name__`未满足您的需求，Flask 文档会详细描述此参数。

下一个代码块设置一个处理对“/'”URL 的 Web 请求的函数：

```py
@app.route('/')
def hello():
  return 'Hello, world!'
```

每当 Flask 收到对“/'”URL 的请求时，它将调用`hello`函数。Python 网络社区将这些路由函数称为**视图**函数。一般来说，Flask 中的视图函数返回字符串以进行 Web 响应，这里我们为所有请求提供一个简单的“Hello, world!”响应，以证明事情正在运行。

最后一个代码块告诉 Python 运行一个开发 Web 服务器，但只有在当前的`.py`文件被直接调用时才这样做：

```py
if __name__ == '__main__':
  app.run()
```

也就是说，这段代码块将在运行命令`python hello.py`时运行，但如果您从另一个 Python 模块中使用`import hello`，则不会运行。

## 还有更多...

一旦您对 Hello World 应用程序满意，让我们构建我们的项目结构。

### 项目布局

我们将构建一个简单的调度应用程序来管理约会并显示它们。让我们将我们的单文件应用程序移动到一个为更大的应用程序设置的目录中。在您的项目中创建以下文件布局，其中包含一个`sched`目录，其中包含当前为空的子目录`static`和`templates`。

```py
manage.py
requirements.txt
sched/
  app.py
  config.py
  filters.py
  forms.py
  __init__.py
  models.py
  static/
  templates/
```

将`hello.py`移动到`sched`目录中的`app.py`。`__init__.py`文件是一个空文件，告诉 Python`sched`是一个包含 Python 文件的包。`config.py`、`filters.py`、`forms.py`和`models.py`文件目前是空文件，我们将在接下来的部分中填写它们。

### 开发服务器

如果您需要从网络上的另一台机器访问 Flask 应用程序，请使用：

```py
app.run('0.0.0.0')
```

Flask 开发服务器的默认端口是`5000`，在本书中我们将在端口`5000`上使用`localhost`。如果此端口已被占用，或者您想使用不同的端口，可以通过`run`方法的第二个参数进行设置。

```py
app.run('0.0.0.0', 8080)
```

`app.run`提供的开发服务器仅用于开发。当您准备发布应用程序时，请使用工业强度的 Web 服务器，如后面关于部署和日志记录的部分所讨论的。因为我们仅在开发中使用`app.run`，让我们打开调试模式，当未捕获的异常发生时，它将在 Web 浏览器中提供交互式调试器，并且将在项目中现有的 Python 文件更改时重新加载代码。您应该只在受信任的网络上使用`'0.0.0.0'`和/或调试模式。在运行行中添加`debug=True`，保留您已经添加的任何其他参数：

```py
app.run(debug=True)
```

### 使用 Flask-Script 的命令行界面

我们将使用**Flask-Script**的`manage.py`来管理我们的应用程序，内容如下：

```py
from flask.ext.script import Manager
from sched.app import app

manager = Manager(app)
app.config['DEBUG'] = True # Ensure debugger will load.

if __name__ == '__main__':
  manager.run()
```

`requirements.txt`文件是跟踪所有第三方 Python 包的常规方式。使用以下列表设置文件，每行一个包名。在您的虚拟环境处于活动状态并且有工作的互联网连接时，运行`pip install -r requirements.txt`以安装这些依赖项。

```py
Flask
Flask-Script
```

现在，您可以使用`python manage.py runserver`运行应用程序。运行`python manage.py -h`以获取帮助，运行`python manage.py runserver -h`以获取有关开发服务器选项的帮助。使用`python manage.py shell`获取加载了 Flask 应用程序的交互式 Python 解释器，您可以使用它来检查代码并尝试新的交互式功能。默认情况下，Flask-Script 为您提供`runserver`和`shell`命令；您可以添加自定义的`manage.py`子命令；请参阅 Flask-Script 文档。

如我们所讨论的，您可以使开发服务器从网络上的另一台机器访问，并更改端口；以下是命令模式：

```py
$ python manage.py runserver –t 0.0.0.0 –p 8080

```

### 提示

Flask-Script 在本月早些时候添加了 Python 3 支持。本书现在完全支持 Python 2 和 Python 3。

# 路由 URL 并接受请求（简单）

本节将在我们的 Web 服务中将 URL 连接到 Python 函数。

## 准备工作

使用“运行简单应用程序”食谱中列出的*项目布局*部分下的项目布局。

## 如何做…

1.  Flask `app`对象包括一个`route`装饰器，用于方便地指定用于视图函数的 URL 规则，为 Python 可调用的请求提供了一种声明式样式的路由。以下代码为我们的调度应用程序中的约会记录的列表、详细信息、创建、更新和删除路由了五个 URL 处理函数。目前，我们只是返回一个描述处理程序将要执行的字符串，稍后我们将实现它，这样您就可以从浏览器中看到 URL 路由的操作。放入`app.py`：

```py
@app.route('/appointments/')
def appointment_list():
  return 'Listing of all appointments we have.'

@app.route('/appointments/<int:appointment_id>/')
def appointment_detail(appointment_id):
  return 'Detail of appointment     #{}.'.format(appointment_id)

@app.route( 
  '/appointments/<int:appointment_id>/edit/',
  methods=['GET', 'POST'])

@app.route(...) and def appointment_edit(...).

def appointment_edit(appointment_id):
  return 'Form to edit appointment #.'.format(appointment_id)

@app.route(
  '/appointments/create/', 
  methods=['GET', 'POST'])
def appointment_create():
  return 'Form to create a new appointment.'

@app.route(
  '/appointments/<int:appointment_id>/delete/,
  methods=['DELETE'])
def appointment_delete(appointment_id):
    raise NotImplementedError('DELETE')
```

### 提示

**下载示例代码**

您可以从[`www.packtpub.com`](http://www.packtpub.com)的帐户中下载您购买的所有 Packt 图书的示例代码文件。如果您在其他地方购买了本书，可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便直接通过电子邮件接收文件。

1.  使用`python manage.py runserver`，您可以访问以下 URL：

+   `http://localhost:5000/appointments/`

+   `http://localhost:5000/appointments/1/`

+   `http://localhost:5000/appointments/1/edit/`

+   `http://localhost:5000/appointments/create/`

+   `http://localhost:5000/appointments/1/delete/`

### 注意

如果您在浏览器中测试这些 URL，您会发现删除 URL 会响应**405 Method Not Allowed**错误。这是有意为之的，因为浏览器默认发送`GET`请求，我们只允许`DELETE`方法。我们不希望在`GET`请求上删除记录，而是只有在按下删除按钮时才删除记录（在后面的部分中构建）。

1.  您可以使用`flask.url_for`函数构建到视图函数的 URL，它返回 URL 的字符串表示。这样可以使用干净的标识符进行编程，以防 URL 发生变化，使用目标函数的名称和其参数：

```py
from flask import url_for

@app.route('/appointments/<int:appointment_id>/')
def appointment_detail(appointment_id):
  edit_url = url_for('appointment_edit',
    appointment_id=appointment_id)
  # Return the URL string just for demonstration.
  return edit_url
```

1.  `url_for`的第一个参数称为端点，默认情况下是由`app.route`装饰器包装的 Python 函数的名称。您可以使用`app.route`的`endpoint`关键字参数来覆盖默认值：

```py
@app.route(
  '/appointments/<int:appointment_id>/',
  endpoint='some_name')
def appointment_detail(appointment_id):
  # Use url_for('some_name', appointment_id=x)
  # to build a URL for this.
  return 'Just to demonstrate...'
```

## 它是如何工作的...

`app.route`的主要参数是字符串 URL 规则，由 Flask 的底层工具包**Werkzeug**实现。在尖括号`<argument>`中列出的项目被解析为命名参数，传递到视图函数中。Flask 在 URL 规则中使用`<converter:argument>`的约定来解析参数值，然后将其传递给视图函数，并仅在正确解析值时路由 URL。默认情况下，Flask 将参数视为字符串。其他内置转换器包括：

+   `int`：此转换器的值是整数

+   `float`：此转换器的值是浮点数

+   `path`：此转换器的值是字符串，如默认值，但也接受斜杠

您可以像 Flask 的`app.url_map`文档中所示那样定义自己的转换器，但您可能不需要这样做，这样您的视图函数可以检查字符串参数并解析所需的内容。

默认情况下，如果 URL 规则以斜杠`/`结尾，Flask 将重定向不带斜杠的请求到包含它的处理程序。

## 还有更多...

一旦您开始路由，您应该了解有关 HTTP 和 Flask 路由的一些事项。

### 处理 HTTP 方法

`app.route`的最常见关键字参数是`methods`，它给 Flask 一个接受的 HTTP 方法列表，当路由时默认为`GET`。有效值为`GET`、`POST`、`PUT`、`DELETE`、`HEAD`和`OPTIONS`。RFC2068 是一个定义这些方法的标准文档。简而言之，它们是：

+   `GET`：此选项用于回复有关资源的信息，最常见

+   `POST`：此选项用于从浏览器/客户端接收资源的更新信息

+   `PUT`：此选项类似于`POST`，但对资源进行重复的`PUT`调用不应产生影响

+   `DELETE`：使用此选项删除资源

+   `HEAD`：此选项类似于`GET`，但仅回复 HTTP 标头而不是内容

+   `OPTIONS`：此选项用于确定资源可用的方法

### 注意

如果`GET`存在，Flask 会为您实现`HEAD`，并在所有情况下为您实现`OPTIONS`。旧的 HTTP 浏览器实现只支持`GET`和`POST`，但我们将在后面的部分通过 JavaScript 添加`DELETE`请求。在某些情况下，使用其他 HTTP 方法取决于您和您的项目，特别是在方法的语义很重要的情况下。

### 装饰函数的替代方法

`@app.route`方法是 Python 装饰器，在定义函数时使用。作为替代，您可以使用`app.add_url_rule`，它的工作方式与`app.route`完全相同，但是它是一个简单的方法调用，而不是一个装饰器。您可以向`app.add_url_rule`提供任何 Python 可调用对象，接受 URL 规则中的参数，在请求时将成为`request.view_args`。如果您对如何指定应用程序的 URL 路由有自己的想法，您可以使用`app.add_url_rule`与您喜欢的任何工具来连接您的 Flask 应用程序。

### 路由冲突

当 Flask 将您的请求路由到意外的位置时，请查找`app.route`调用中的冲突。也就是说，如果您有路由`/<path:foopath>`和`/foo/bar/baz/`，两者都将匹配`/foo/bar/baz/`。解决方案是在路由参数中尽可能具体，并避免过于通用的参数。

### 使用子域路由

您可以使用`app.route`的`subdomain`关键字参数路由子域：

```py
@app.route('/', subdomain='<spam_eggs>')
def subdomain_example(spam_eggs):
  return '...'
```

这个子域参数使用与其他 URL 相同的参数方法，但请注意适当的子域在接受的内容方面是有限制的。简单名称很直接，但如果您有特定要求，请参阅 RFC2181 以获取名称语法，并注意一些 HTTP 客户端不支持子域中的完整规范。

当您使用子域时，Flask 需要知道服务器名称以便从 URL 中解析子域。提供`SERVER_NAME`给`app.config`，如配置部分所述。

在本地开发时，您也会遇到限制，因为`localhost`不接受子域。您可以设置操作系统的 hosts 文件（通常为`/etc/hosts`）。如果您的`SERVER_NAME`是`example.com`，并且您的子域参数是`foo`，您可以设置：

```py
127.0.0.1    localhost local.example.com foo.local.example.com
```

`hosts`文件不接受通配符。如果您可以访问域名服务，hosts 文件的替代方法是在 DNS 中设置 A 记录及其所有子域（通配符）为`127.0.0.1`，就像以下示例一样。这将在所有机器上将`local.example.com`的所有子域路由到本地主机。

```py
local.example.com A 127.0.0.1
*.local.example.com A 127.0.0.1
```

这种技术需要进行 DNS 网络调用，因此如果您离线工作，您需要退回到 hosts 文件。

# 处理请求和响应（简单）

本节将演示 Flask 如何处理 HTTP 请求上的传入数据以及您如何发送响应。

## 准备工作

将调度应用程序搁置一边，打开一个新的 Python 文件来探索 Flask。

## 如何做...

1.  在核心上，Flask 和 Werkzeug 提供请求和响应对象来表示您的 Web 应用程序的传入和传出数据。Flask 提供了三种不同的模式来返回视图函数的返回值：

+   字符串，可以选择使用模板引擎（稍后介绍）

+   响应实例，一个具有表示 HTTP 响应细节的属性的对象

+   元组`(string, status)`或`(string, status, http_headers)`，以方便起见，这样您就不必创建响应实例

1.  让我们看看每个响应模式的实际操作。每个响应都将以 200 OK 状态码（成功的典型 HTTP 响应）说**Hello, world!**，后两个函数告诉浏览器将响应显示为纯文本。第一个函数的响应显示为 HTML，因为仅字符串返回对象无法告诉 Flask 如何设置响应的内容类型。

```py
from flask import Flask, make_response

app = Flask(__name__)

@app.route('/string/')
def return_string():
  return 'Hello, world!'

@app.route('/object/')
def return_object():
  headers = {'Content-Type': 'text/plain'}
  return make_response('Hello, world!', status=200,
    headers=headers)

@app.route('/tuple/')
def return_tuple():
  return 'Hello, world!', 200, {'Content-Type':
    'text/plain'}
```

1.  当 Flask 准备调用您的视图函数并接受其返回值时，它会遍历您提供的每个在请求之前和请求之后的回调。在这里，我们设置了一个在请求之前的函数和一个在请求之后的函数，纯粹是为了演示 Flask 的请求和响应处理。您可以使用这些处理程序来探索 Flask 在您想要与 Flask 文档交互时的行为。当您运行此代码时，请记住，浏览器通常会自动寻找`/favicon.ico`，这可能会在测试时导致对您的应用程序的额外请求。

```py
from flask import request

def dump_request_detail(request):
  request_detail = """
# Before Request #
request.endpoint: {request.endpoint}
request.method: {request.method}
request.view_args: {request.view_args}
request.args: {request.args}
request.form: {request.form}
request.user_agent: {request.user_agent}
request.files: {request.files}
request.is_xhr: {request.is_xhr}

## request.headers ##
{request.headers}
  """.format(request=request).strip()
  return request_detail

@app.before_request
def callme_before_every_request():
  # Demo only: the before_request hook.
  app.logger.debug(dump_request_detail(request))

@app.after_request
def callme_after_every_response(response):
  # Demo only: the after_request hook.
  app.logger.debug('# After Request #\n' + repr(response))
  return response
```

## 它是如何工作的...

以下是请求对象的常用特性。请参阅 Flask 文档以获取传入请求数据的完整列表。为了提供一个示例，每个描述都包括一个示例值，用于从我们的浏览器命中`/string/?foo=bar&foo=baz`的请求。

+   `endpoint`：请求对象的这个特性指定路由的请求端点的名称，例如`return_string`。

+   `method`：请求对象的这个特性指定当前请求的 HTTP 方法，例如`GET`。

+   `view_args`：请求对象的这个特性指定从 URL 路由规则解析的视图函数参数字典，例如`{}`。

+   `args`：请求对象的这个特性指定从 URL 查询字符串解析的参数字典，例如`request.args['foo']`是`'bar'`，`request.args.getlist('foo')`是`['bar', 'baz']`。

+   `form`：请求对象的这个特性指定来自`POST`或`PUT`请求的表单数据字典，例如`{}`。

+   `user_agent`：请求对象的这个特性指定了浏览器提供的版本标识。

+   `文件`：请求对象的这个特性指定了来自`POST`或`PUT`请求的文件上传的字典，这些文件上传代替了`request.form`，例如，`{}`。字典中的每个值都是一个`FileStorage`对象，它的行为类似于 Python 的`file`对象，但也包括一个`save(filepath)`方法来存储上传的文件（在验证目标路径之后）。

+   `is_xhr`：`True`：请求对象的这个特性指定了传入请求是否为 JavaScript `XMLHttpRequest`，否则为`False`。这适用于提供`X-Requested-With` HTTP 头的 JavaScript 库，设置为`XMLHttpRequest`。

Flask 使用自定义的`dict`类型`ImmutableMultiDict`，它支持每个键的多个值（通过`getlist`方法访问），在 HTTP 允许给定参数的多个值的情况下，例如 URL 中的查询字符串，例如，`?foo=bar&foo=baz`。

## 还有更多...

我们在演示代码中使用了`app.before_request`和`app.after_request`。最终，您将希望在请求之前和之后的处理程序之间保留对象。

### 请求之前和之后

如果您需要在请求之前和之后的处理程序之间保留一个对象，可以在`flask.g`对象上设置一个属性。`g`对象只在请求的持续时间内存在，您可以在`g`对象上设置任何您想要的东西。这是为了方便起见，对于您在整个请求过程中需要的东西，但目前还没有归属地。不要滥用`g`对象，不要给它属于其他地方的对象，比如数据库系统。

这是`g`对象的示例，其中在请求之前和之后设置了`g.x`上的 0 到 9 之间的随机整数，并在请求之前和之后记录：

```py
import random

from flask import g

@app.before_request
def set_on_g_object():
  x = random.randint(0, 9)
  app.logger.debug('before request: g.x is {x}'.format(x=x))
  g.x = x

@app.after_request
def get_on_g_object(response):
  app.logger.debug(
    'after request: g.x is {g.x}'.format(g=g))
  return response
```

# 处理静态文件（简单）

Flask 准备好在您提供第一个请求时从磁盘上提供文件。

## 准备工作

转到您的调度器项目，查看`sched`目录中的`static`文件夹。

## 如何做...

1.  将文件放入`static`文件夹中。

1.  使用`flask.url_for('static', filename='path/to/filename')`为它们构建 URL，其中`path/to/filename`是静态文件夹中的文件路径，使用`/`，无论您使用的操作系统是什么。

## 工作原理...

按照惯例，Flask 会在应用程序旁边查找名为`static`的文件夹，并在`/static/<path:filename>` URL 处提供该文件夹中及其子文件夹中的所有文件。也就是说，如果您的应用程序在`app.py`中，则默认情况下，Flask 会查看`app.py`旁边的`static`文件夹。将文件放置在`app.py`旁边的`/static/img/favicon.ico`处，可以在 URL`/static/img/favicon.ico`处使用，默认开发服务器中的`http://localhost:5000/static/img/favicon.ico`。（这个 favicon 在下一节关于模板的部分中有详细说明。）

## 还有更多...

您可以自定义 Flask 处理静态文件的方式。

### 在生产环境中提供静态文件

当您部署应用程序时，您可能希望从诸如 nginx 或 Apache httpd 之类的工业强度 HTTP 服务器中提供静态文件，这些服务器对静态文件进行了高度优化。您可以配置这些服务器从项目中的相同`/static/`文件夹中提供`/static/` URL。如果您无法访问这些服务器，如果需要，您可以在生产中使用 Flask 的静态文件处理。

### 在其他地方托管静态文件

如果您正在使用由另一个服务器提供的 URL 不是`/static/`的静态文件，您可以在 Flask 应用程序中添加`build_only`选项的路由。这告诉 Flask 如何使用`flask.url_for`函数构建 URL，而无需`view`函数或让 Flask 尝试提供这些文件。

```py
app.add_url_rule('/attachments/<path:filename>', endpoint='attachments', build_only=True)
```

### 自定义静态文件处理程序

如果需要在从文件系统提供文件时进行任何自定义处理（例如，仅向付费用户提供文件），可以构建自己的静态文件处理程序，用于静态文件夹之外的静态文件（例如，`/robots.txt`），或者覆盖 Flask 的内置静态视图函数：

```py
import os.path
from flask import Flask, send_from_directory

# Setting static_folder=None disables built-in static handler.
app = Flask(__name__, static_folder=None)
# Use folder called 'assets' sitting next to app.py module.
assets_folder = os.path.join(app.root_path, 'assets')

@app.route('/assets/<path:filename>')
def assets(filename):
  # Add custom handling here.
  # Send a file download response.
  return send_from_directory(assets_folder, filename)
```

使用硬编码的目录和`send_from_directory`将减轻目录遍历攻击，因为底层实现使用`flask.safe_join(directory, filename)`来清理输入，您可以在处理作为用户输入传递的文件名时在自己的代码中使用它。

使用自定义静态视图函数来自定义处理静态文件，并不是更改`static`文件夹的配置。如果只需要重命名`static`文件夹，请使用：

```py
app = Flask(__name__, static_folder='foldername')
```

默认情况下，Flask 在`/static/`处提供静态文件，其中`static`是在`static_folder`参数中提供的文件夹的名称，默认为`static`。如果要从文件夹名称使用不同的 URL 路径，请使用：

```py
app = Flask(__name__, static_url_path='/assets')
```

### HTTP 缓存

默认情况下，Flask 发送一个带有静态文件的 HTTP 头，告诉浏览器将文件缓存 43200 秒，即 12 小时。您可以使用`SEND_FILE_MAX_AGE_DEFAULT`配置变量进行配置。例如，如果您希望对所有静态文件进行积极缓存，可以将此值设置为 2592000 秒，即 30 天：

```py
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 2592000
```

增加 HTTP 缓存长度时要小心，因为在部署更改时浏览器会有过时的文件。为了解决这个问题，在重新部署时更改静态 URL 路径：

```py
app = Flask(__name__, static_url_path='/static/v2')
```

### 目录索引

Flask 不会列出静态目录中的文件内容供用户浏览；您必须直接链接到静态文件。如果需要目录索引视图，请考虑使用像 nginx 或 Apache httpd 这样的工业级静态文件 Web 服务器。

# 使用数据库（简单）

我们的调度程序应用程序需要数据，我们希望正确存储这些数据，以便我们可以在不丢失约会的情况下停止和启动我们的应用程序。

## 准备工作

我们正在从项目的`sched`目录中的`models.py`文件中进行工作。

## 如何做...

SQLAlchemy 提供了一个 Python 工具包和对象关系管理器，用于使用 SQL 的关系数据库。SQLAlchemy 独立存在，不管你使用什么 web 框架，我们可以将其集成到 Flask 中，使用 Flask-SQLAlchemy 来管理 Flask 请求生命周期中的数据库连接。

更新`requirements.txt`：

```py
Flask
Flask-Script
Flask-SQLAlchemy
```

然后：

```py
$ pip install -r requirements.txt

```

让我们使用 SQLAlchemy 的声明性扩展来建模我们的约会数据结构，这允许我们编写 Python 类来表示数据库表。请注意，Flask-SQLAlchemy 包括一些声明性功能，但我们的应用程序使用纯 SQLAlchemy 来使代码可移植到任何 Python 项目，并让您引用核心 SQLAlchemy 文档，[docs.sqlalchemy.org](http://docs.sqlalchemy.org)。

我们正在建模一个`Appointment`类，它将对象映射到`appointment`表。从一个`Base`类开始，定义一个`Appointment`子类。当与底层数据库系统一起使用时，SQLAlchemy 将找到`Base`的所有子类。然后在`Appointment`类上定义列，这些列将映射到我们在代码中使用的每个`Appointment`对象的属性，将值存储和检索为数据库中`appointment`表中的列。在`models.py`中：

```py
from datetime import datetime

from sqlalchemy import Boolean, Column
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class Appointment(Base):
  """An appointment on the calendar."""
  __tablename__ = 'appointment'

  id = Column(Integer, primary_key=True)
  created = Column(DateTime, default=datetime.now)
  modified = Column(DateTime, default=datetime.now,
    onupdate=datetime.now)

  title = Column(String(255))
  start = Column(DateTime, nullable=False)
  end = Column(DateTime, nullable=False)
  allday = Column(Boolean, default=False)
  location = Column(String(255))
  description = Column(Text)
```

要在我们的应用程序中加载领域模型，我们在我们的 Flask 应用程序上配置我们的数据库，并设置一个`db`对象，以准备在我们的请求处理程序中进行查询，在`app.py`中：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

from sched.models import Base
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sched.db'

# Use Flask-SQLAlchemy for its engine and session
# configuration. Load the extension, giving it the app object,
# and override its default Model class with the pure
# SQLAlchemy declarative Base class.
db = SQLAlchemy(app)
db.Model = Base
```

我们可以将一些辅助方法添加为`Appointment`类的方法。`duration`属性提供了约会的长度的计算，以秒为单位，使用`appointment`对象的开始和结束时间。`__repr__`方法告诉 Python 如何在打印时表示`appointment`对象。这个实现将说`<Appointment: 1>`而不是 Python 的默认形式`<__main__.Appointment object at 0x26cf2d0>`。

在`models.py`中，在`Appointment`类声明中，添加：

```py
  @property
  def duration(self):
    delta = self.end - self.start
      return delta.days * 24 * 60 * 60 + delta.seconds

  def __repr__(self):
    return (u'<{self.__class__.__name__}: {self.id}>'
      .format(self=self))
```

## 它是如何工作的...

每个`Column`都采用一个类型，为`appointment`记录提供结构。我们给`Column`关键字参数定义行为如下：

+   `primary_key`：`True`表示该字段是记录标识符。

+   `default`：当没有提供数据时，使用此值。这可以是返回值的函数，例如，将`created`设置为`datetime.now()`的输出，以提供数据库记录创建时的日期/时间。

+   `onupdate`：当记录被存储或更新时，将其值设置为给定函数的返回值，例如，将`modified`设置为数据库记录更新时的当前日期/时间。

+   `nullable`：当为`False`时，不允许存储没有为此属性设置值的记录，通过引发异常。

`sqlite:///sched.db` URL 告诉 SQLAlchemy 在当前工作目录中使用 SQLite 数据库。Python 内置支持 SQLite，它是一种嵌入式关系数据库管理系统，也可以在[sqlite.org](http://sqlite.org)上找到。

## 还有更多...

我们只对数据进行了建模，并告诉 Flask 如何连接到 SQLite 数据库。让我们尝试一些查询。

### 使用 SQLAlchemy 进行查询

有了`Appointment`模型定义，我们可以从 Python 运行一些查询。在构建应用程序的视图函数之前，我们可以进行一些示例查询。将以下代码添加到`models.py`的主脚本中，在`if __name__ == '__main__'`块内。添加任何语句或`print`调用，以便查看 SQLAlchemy 的工作原理并观察它的运行方式：

```py
$ python models.py

```

### 提示

Web 请求在哪里？这一部分直接使用 SQLAlchemy，而不是在请求上下文中，以便说明 SQLAlchemy 的工作原理。这种自下而上的方法有助于理解构建模块。

我们从一个引擎开始，它连接到数据库并执行查询。如果您不想在磁盘上创建文件，可以使用临时内存数据库。`sqlite://` URL 告诉 SQLAlchemy 连接到 SQLite 数据库，并且因为省略了文件路径，它应该连接到内存中的临时数据库。也就是说，`sqlite://` URL 将提供一个仅在 Python 进程执行期间存在的数据库，并且不会在调用 Python 时持久存在。

```py
from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///sched.db', echo=True)
```

接下来我们创建一个会话并创建数据库表。当引擎连接到数据库并执行查询时，会话表示与数据库的持续对话，并且是应用程序使用 SQLAlchemy 中的关系数据库的主要入口点。

```py
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
```

现在我们添加一些示例数据。我们使用当前时间`now`加上或减去一些`timedelta`来生成我们虚假预约的时间，`timedelta`接受天数和秒数。

```py
now = datetime.now()

session.add(Appointment(
  title='Important Meeting',
  start=now + timedelta(days=3),
  end=now + timedelta(days=3, seconds=3600),
  allday=False,
  location='The Office'))
session.commit()

session.add(Appointment(
  title='Past Meeting',
  start=now - timedelta(days=3),
  end=now - timedelta(days=3, seconds=3600),
  allday=False,
  location='The Office'))
session.commit()

session.add(Appointment(
  title='Follow Up',
  start=now + timedelta(days=4),
  end=now + timedelta(days=4, seconds=3600),
  allday=False,
  location='The Office'))
session.commit()

session.add(Appointment(
  title='Day Off',
  start=now + timedelta(days=5),
  end=now + timedelta(days=5),
  allday=True))
session.commit()
```

要创建、更新和删除预约记录：

```py
# Create. Add a new model instance to the session.
appt = Appointment(
  title='My Appointment',
  start=now,
  end=now + timedelta(seconds=1800),
  allday=False)

session.add(appt)
session.commit()

# Update. Update the object in place, then commit.
appt.title = 'Your Appointment'
session.commit()

# Delete. Tell the session to delete the object.
session.delete(appt)
session.commit()
```

以下是一些示例查询，您可以运行以了解 SQLAlchemy 的工作原理。每个`appt`示例都是`Appointment`类型的 Python 对象。每个`appts`示例都是`Appointment`对象的 Python 列表。

```py
# Get an appointment by ID.
appt = session.query(Appointment).get(1)

# Get all appointments.
appts = session.query(Appointment).all()

# Get all appointments before right now, after right now.
appts = session.query(Appointment).filter(Appointment.start < datetime.now()).all()
appts = session.query(Appointment).filter(Appointment.start >= datetime.now()).all()

# Get all appointments before a certain date.
appts = session.query(Appointment).filter(Appointment.start <= datetime(2013, 5, 1)).all()

# Get the first appointment matching the filter query.
appt = session.query(Appointment).filter(Appointment.start <= datetime(2013, 5, 1)).first()
```

SQLAlchemy 提供完整的 SQL 功能。确保在从 Python 代码生成 SQL 查询时参考 SQLAlchemy 文档。

### 提示

关于`session`对象：您将使用`db.session`（在本节中我们在`app.py`中添加的`db`对象）而不是直接使用会话。Flask-SQLAlchemy 将确保正确实例化`db.session`，并且通过数据库会话访问的数据不会干扰其他请求中的`session`。

### 生产数据库

SQLite 非常适合开发，并且可以在生产中使用，但是当您部署代码时，您可能希望使用 PostgreSQL 或 MySQL 作为您的数据库管理系统。您可以通过指定不同的数据库 URL 并安装必要的包来更改底层数据库，同时保持模型不变。

对于 PostgreSQL：

+   URL 模式：`postgresql://user:pass@localhost:5432/database_name`

+   `pip install psycopg2`

对于 MySQL：

+   URL 模式：`mysql://user:pass@localhost:3306/database_name`

+   `pip install mysql-python`

记得在连接到一个干净的数据库时创建表。

SQLAlchemy 知道如何将数据定义转换为所有主要的关系数据库管理系统，当您需要使用特定的数据库功能时，它支持 SQL 方言，如[docs.sqlalchemy.org/en/rel_0_8/dialects/](http://docs.sqlalchemy.org/en/rel_0_8/dialects/)中所述。

### 命名约定

我们的`Appointment`类使用名为`appointment`的基础表。我建议使用带有主键的单数表名，主键始终命名为`id`，但您可以自行决定如何命名您的表和列。

### 自定义 SQL

在其核心，SQLAlchemy 是一个用于构建参数化 SQL 表达式的 Python 工具包。在任何时候，如果您需要编写自定义查询，您不需要通过传递字符串来降级到原始执行语句—SQLAlchemy 为您提供了工具。

请参阅[docs.sqlalchemy.org/en/rel_0_8/core/](http://docs.sqlalchemy.org/en/rel_0_8/core/)中的 SQL 表达语言文档。

### 文档存储和非关系数据库

如果您更喜欢另一个数据库管理系统，也许是文档导向而不是关系型的，那么可以使用 Python 连接到该数据库的可用工具。正如您所看到的，我们已经将域模型声明隔离在一个文件中，并用几行代码将其连接到 Flask 应用程序对象中。本书中的应用程序代码将使用 SQLAlchemy API 来查询数据，但您可以将这些调用转换为您选择的数据库。

# 处理表单和文件上传（简单）

让我们从用户那里获取数据。在开始之前，请记住绝对不要相信用户输入，因为迟早有人（或某人的脚本）会试图破坏您的应用程序。

## 准备工作

我们正在从项目的`sched`目录中的`forms.py`文件中工作。

## 如何做...

1.  现在我们有了数据模型，我们需要向用户呈现一个表单，以便填写我们的数据库并验证用户输入，以确保它与我们的模式匹配。您可以使用任何您喜欢的工具来验证传入的数据。我们将在我们的调度程序中使用**WTForms**。

1.  更新`requirements.txt`：

```py
Flask
Flask-Script
Flask-SQLAlchemy
WTForms
```

1.  然后：

```py
$ pip install -r requirements.txt

```

1.  WTForms 使用类似于 SQLAlchemy 的声明性扩展来模拟表单模型的形式。它的理念是表单应该与数据分开建模，这样表单和数据就是分开的关注点，通常用户表单（在 HTML 用户界面中）与使用 SQLAlchemy 建模的域数据并不完全匹配。WTForms 提供 HTML 表单生成和表单数据验证。在 Flask 中，您将在`POST`和`PUT`请求的`request.form`中找到提交的表单数据。

1.  这是我们的预约表单。每个属性都有一个标签和一组验证器。`Length(max=255)`验证器确保最多输入 255 个字符，`required()`验证器拒绝空输入。在`forms.py`中：

```py
from wtforms import Form, BooleanField, DateTimeField
from wtforms import TextAreaField, TextField
from wtforms.validators import Length, required

class AppointmentForm(Form):
  title = TextField('Title', [Length(max=255)])
  start = DateTimeField('Start', [required()])
  end = DateTimeField('End')
  allday = BooleanField('All Day')
  location = TextField('Location', [Length(max=255)])
  description = TextAreaField('Description')
```

## 它是如何工作的...

`AppointmentForm`类的目的是双重的：在 HTML 中呈现输入表单并验证提交的数据。这与基于 SQLAlchemy 的`Appointment`类非常接近。`Appointment`模型代表了域及其持久性，而这个表单类代表了如何在 HTML 中显示表单并接受或拒绝结果。

### 提示

在我们的自下而上的方法中，这是一个在将代码添加到我们的 Flask 应用程序之前查看我们的代码的机会。本节通过一些示例代码解释了 WTForms 的工作原理。您可以将此代码添加到`forms.py`的底部的`if name == '__main__'`块中，并使用`python forms.py`运行它。

在后面的部分中，我们将在应用程序中添加表单之前，让我们看看表单在自己身上的作用。我们将打印 WTForms 创建的值，以便我们可以理解它在做什么。如果您使用 Python 2，您需要在 Python 文件的顶部添加一行，以使您的打印表达式在 Python 2 和 Python 3 之间兼容。

```py
from __future__ import print_function
```

以下代码打印了`title`字段的 HTML 表示：

```py
# Demonstration of a WTForms form by itself.
form = AppointmentForm()
print('Here is how a form field displays:')
print(form.title.label)
print(form.title)
```

输出：

```py
Here is how a form field displays:
<label for="title">Title</label>
<input id="title" name="title" type="text" value="">
```

在浏览器中显示时，此表单字段将呈现为非常简单的输入，如下面的屏幕截图所示。我们将在 HTML 模板的后面部分为我们的表单添加样式。

![工作原理...](img/9628OS_01_06.jpg)

以下代码显示了如何直接与表单验证进行交互。您可以使用表单数据的字典来实例化`AppointmentForm`。您可以将 Flask 的`request.form`对象传递给`AppointmentForm`，但在这里，我们将构建自己的字典来查看它是如何工作的。由于 HTML 表单支持单个参数的多个值，我们不能使用 Python 的内置`dict`。相反，我们使用 Werkzeug 的`ImmutableMultiDict`类型，并为`title`字段制作虚拟数据，并省略所有其他字段。

```py
# Give it some data.
from werkzeug.datastructures import \
  ImmutableMultiDict as multidict

data = multidict([('title', 'Hello, form!')])
form = AppointmentForm(data)
print('Here is validation...')
print('Does it validate: {}'.format(form.validate()))
print('There is an error attached to the field...')
print('form.start.errors: {}'.format(form.start.errors))
```

输出：

```py
Here is validation...
Does it validate: False
There is an error attached to the field...
form.start.errors: [u'This field is required.']
```

请注意，当`AppointmentForm`获得字典表单数据时，`form.validate()`将处理它，并根据表单数据是否有效返回`True`或`False`（`True`表示有效）。验证器的错误将加载在`errors`列表中的每个字段上，以便您可以在上下文中显示错误，这是我们在后面的模板渲染中将要做的。

## 还有更多...

我们以稍有不同的方式处理文件上传。

### 处理文件上传

如果您正在处理文件上传，您需要直接与 Flask 进行交互。以下代码将帮助您入门。首先，HTML 表单将包括这些基本元素：`form`元素上的`enctype=multipart/form-data`属性和表单中的文件输入，格式为`"<input type=file name=...>"`。

```py
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Attach>
</form>
```

接收端的代码应该在`request.files`字典中查找 Werkzeug 的`FileStorage`对象，名称与文件输入元素上给定的名称相匹配，然后在保存文件到磁盘之前对文件名进行清理。请注意，如果您在响应中使用与文件名扩展名匹配的 Content-Type，您应该只对您信任的文件扩展名这样做。否则，攻击者可以利用您的文件上传功能在您的应用程序中嵌入 JavaScript，这将导致用户的浏览器信任其他人的代码。

```py
import os

from flask import request
from werkzeug import secure_filename

# '...' here and below need to be set according to your project.
# In your configuration, set a root folder for file uploads.
FILE_FOLDER = '...'

@app.route('...', methods=['POST'])
def file_attach():
  # Match name from <input type=file name=file>: 'file'.
  filestorage = request.files['file']
  # Do not allow '..' in the filename.
  filename = secure_filename(filestorage.filename)
  dest = os.path.join(FILE_FOLDER, filename)
  filestorage.save(dest)
  return 'Put your response here.'
```

# Jinja 模板化-设置基本模板（简单）

Flask 附带了 Jinja 模板引擎，可以呈现您需要的任何文本格式。我们的调度程序将通过 Jinja 呈现 HTML5 页面，稍加样式和一点 JavaScript。

## 准备工作

我们将使用 HTML、CSS 和 JavaScript 构建模板。您需要对这些内容有一定的了解，才能理解我们在 Flask 中构建的响应。我们将使用 Twitter 的 Bootstrap ([getbootstrap.com/2.3.2/](http://getbootstrap.com/2.3.2/))框架进行 CSS 和 jQuery ([jquery.com](http://jquery.com))库进行 JavaScript。两者都包括在线文档。Bootstrap 还包括图标，我们可以在我们的应用程序中使用，提供者是 Glyphicons ([glyphicons.com](http://glyphicons.com))。我们将使用 Bootswatch ([bootswatch.com](http://bootswatch.com))为 Bootstrap 提供免费主题。

我们是在`sched`目录中的`templates`目录内的`base.html`中工作。

### 提示

这段文字通过示例展示了模板化。在我们的自下而上的方法中，我们将从基本页面结构开始，然后构建特定的应用程序页面。Jinja 有关其所有模板功能的文档。请务必参考[`jinja.pocoo.org/docs/`](http://jinja.pocoo.org/docs/)上的文档，了解 Jinja 提供了什么。

## 如何做...

我们将使用公共**内容传送网络**（**CDN**）来为 Bootstrap 和 jQuery 提供服务，这将加快我们项目的启动时间。NetDNA 为 Bootstrap 提供了一个公共 CDN。jQuery 项目通过 MediaTemple 提供了一个 CDN。每当你准备好或者更喜欢远离 CDN 时，只需下载你需要的文件，并将它们与应用的静态文件一起提供，更新基础模板中的链接。CDN 的 URL 如下，用于基础模板（其中`... thing ...`标识符）：

+   [`netdna.bootstrapcdn.com/bootswatch/2.3.2/united/bootstrap.min.css`](http://netdna.bootstrapcdn.com/bootswatch/2.3.2/united/bootstrap.min.css)

+   [`netdna.bootstrapcdn.com/twitter-bootstrap/2.3.2/css/bootstrap-responsive.min.css`](http://netdna.bootstrapcdn.com/twitter-bootstrap/2.3.2/css/bootstrap-responsive.min.css)

+   [`code.jquery.com/jquery-1.9.1.min.js`](http://code.jquery.com/jquery-1.9.1.min.js)

+   [`netdna.bootstrapcdn.com/twitter-bootstrap/2.3.2/js/bootstrap.min.js`](http://netdna.bootstrapcdn.com/twitter-bootstrap/2.3.2/js/bootstrap.min.js)

Jinja 支持模板继承，其中模板文件可以扩展现有模板。我们将使用这个功能来布局我们的基本结构，使用`block`模板标签来指示子模板将填充的部分。我们提供了一个块来设置页面标题，向页面头部添加任何额外的内容（对于额外的样式和脚本很有用），并填充页面的主要内容。Jinja 使用模板标签`{% ... %}`来指示 Jinja 特定的标记和指令。以下是我们的基础模板，在`templates/base.html`中，屏幕截图在接下来的部分中：

```py
<!DOCTYPE html>
<html lang="en">
<head>
  <title>{% block title %}{% endblock title %}</title>
  <link rel="shortcut icon" type="image/x-icon"
    href="{{ url_for('static', filename='img/favicon.ico') }}"/>
  <link href="... bootstrap css ..." rel="stylesheet">
  <link href="... responsive css ..." rel="stylesheet">
  <script src="img/... jQuery js ..."></script>
  <script src="img/... bootstrap js ..."></script>
  {# Link any other .css or .js found in app static folder. #}
  <style>
    {# Add simple CSS style tweaks here. #}
  </style>
  <script>
    $(function() {
      {# Add page-load JavaScript here. #}
    });
  </script>
  {% block extra_head %}{% endblock extra_head %}
</head>
<body>
  <div id="main">
    <div class="utility-nav navbar navbar-fixed-top">
    <div class="navbar-inner">
    <div class="container">
      {# Navbar goes here. #}
    </div>
   </div>
  </div>
  <div class="content container">
    {% block main %}{% endblock main %}
  </div>
  </div>
</body>
</html>
```

## 它是如何工作的...

这个基础模板允许我们编写专注的页面，这些页面共享一个共同的结构和样式。你在模板目录中创建的任何模板都可以继承这个基础模板：

```py
{% extends 'base.html' %}
{% block title %}Page Title{% endblock title %}
{% block main %}
  <h2>This is a child template.</h2>
{% endblock main %}
```

使用`flask.render_template`函数进行渲染。如果文件名为`index.html`，在模板目录中，你可以使用以下代码将该模板渲染为字符串：

```py
from flask import render_template
#...
@app.route('/')
def index():
  return render_template('index.html')
```

## 还有更多...

Jinja 模板还有很多其他功能

### 使用 Jinja 以外的模板引擎

你可以选择任何你喜欢的模板引擎。当你调用`render_template`时，你会得到一个 Python 字符串，Flask 会将该字符串转换为一个`Response`对象。如果你更喜欢不同的模板方法，可以构建一个字符串并从你的视图函数中返回它。Flask 捆绑了 Jinja，为快速启动提供了一个默认的模板环境，并为社区构建 Flask 扩展提供了一个共同的环境。

### 使用 Bower 管理 Web 包

你可以使用 Bower 来管理下载的静态文件，Bower 是一个用于 Web 工具的包管理器，[bower.io](http://bower.io)。使用 Bower，你可以使用`bower.json`文件列出依赖项，使用`bower install`加载这些文件，然后在应用的静态区域提供这些文件。

# 创建新记录（中级）

这里我们提供一个网页表单，用于创建新的预约，如下截图所示：

![创建新记录（中级）](img/9628OS_01_01.jpg)

## 准备工作

我们将继续从`sched`的`app.py`文件和`templates`目录中进行工作。

## 如何做...

1.  我们提供了一个`view`函数来处理表单的 GET 和 POST。在这里，我们从数据库部分提取了`db.session`，从表单部分提取了`AppointmentForm`。在`app.py`中：

```py
from flask import abort, jsonify, redirect, render_template
from flask import request, url_for
from sched.forms import AppointmentForm
from sched.models import Appointment

# ... skipping ahead. Keep previous code from app.py here.

@app.route('/appointments/create/', methods=['GET', 'POST'])
def appointment_create():
  """Provide HTML form to create a new appointment."""
  form = AppointmentForm(request.form)
  if request.method == 'POST' and form.validate():
    appt = Appointment()
    form.populate_obj(appt)
    db.session.add(appt)
    db.session.commit()
    # Success. Send user back to full appointment list.
    return redirect(url_for('appointment_list'))
  # Either first load or validation error at this point.
  return render_template('appointment/edit.html', form=form)
```

1.  在构建输入字段之前，我们可以为自己创建一个实用程序，它将显示给定字段的所有 WTForms 功能：标签、输入和错误。Jinja 有宏，类似于 Python 函数。我们将创建一个宏来从`AppointmentForm`中呈现一个`edit`字段。

1.  对于输入，我们可以遵循 Bootstrap 的约定，使用`control-group`和`controls`页面元素，这将让我们完全控制表单流程的 CSS。我们想要用我们的宏开始一个新的模板，这样我们可以在我们创建的其他模板中重用它。在`templates/appointment/common.html`中：

```py
{% macro edit_field(field, catch_kwargs=true) %}
  <div class="control-group{% if field.errors %} error{% endif %}">
    {{ field.label(class="control-label") }}
  <div class="controls">
    {{ field(**kwargs) }}
    {% for error in field.errors %}
     <span class="help-inline">{{ error }}</span>
    {% endfor %}
  </div>
  </div>
{% endmacro %}
```

现在我们可以使用我们的新宏构建一个表单。开始一个新模板，我们可以扩展基础并导入宏。在扩展基础时，我们提供标题和主要块，使用与基础模板相同的语法。您可以在模板中设置变量，以便在模板内使用，使用`{% set ... %}`。 

1.  在`templates/appointment/edit.html`中：

```py
{% extends 'base.html' %}

{% from 'appointment/common.html' import edit_field %}

{% set title = 'Add Appointment' %}

{% block title %}{{ title }}{% endblock title %}

{% block main %}
<div class="row">
  <div class="appointment-edit well offset2 span8">
  <form method="post" class="form-horizontal">
    <legend>{{ title }}</legend>
    {{ edit_field(form.title, maxlength=255, class="span3",
      placeholder="Important Meeting") }}
    {{ edit_field(form.start, class="span3", type="datetime",
      placeholder="yyyy-mm-dd hh:mm:ss") }}
    {{ edit_field(form.end, class="span3", type="datetime",
      placeholder="yyyy-mm-dd hh:mm:ss") }}
    {{ edit_field(form.allday) }}
    {{ edit_field(form.location, maxlength=255, class="span3",
      placeholder="The Office") }}
    {{ edit_field(form.description, rows="5", class="span5",
      placeholder="Bring a sharpened #2 pencil.") }}
    <div class="form-actions">
    <button type="submit" class="btn">Save</button>
    or <a href="{{ url_for('appointment_list') }}">Cancel</a>
    </div>
  </form>
  </div>
</div>
{% endblock main %}
```

## 它是如何工作的...

在 GET 时，表单将呈现为将数据 POST 到相同 URL 的操作。在 POST 时，视图函数验证数据并将其添加到数据库中。如果验证失败，POST 将再次呈现模板，但这次表单对象有错误。

Jinja 使用模板语法`{{ ... }}`将 Python 对象打印到模板输出的上下文中。我们使用一个简单的`if`语句，如果字段有错误，则添加错误类，这将在存在错误时突出显示输入（感谢 Bootstrap 样式）。我们使用`for`循环为字段上的每个错误放置一个帮助 span。`**kwargs`的使用将捕获超出宏调用签名的所有参数；这使我们能够通过宏传递所有 WTForm 字段选项。`kwargs`特性仅在 Jinja 宏中`catch_kwargs=true`时起作用。

CSS 类`form-horizontal`、`span3`和`span5`告诉 Bootstrap 如何在其网格系统上布局表单。`placeholder`属性是 HTML5 中的一个特性，用于在输入没有内容时显示一个水印。

# 显示记录（中级）

随着内容进入数据库，让我们按照下图所示将其取回：

![显示记录（中级）](img/9628OS_01_03.jpg)

## 准备就绪

我们将继续从`sched`文件夹和`templates`目录的`app.py`文件中进行工作。

## 如何做...

1.  我们提供一个简单的查询，通过数据库 ID 获取一个约会。在`app.py`中：

```py
# Note the import statements from the previous section.

@app.route('/appointments/<int:appointment_id>/')
def appointment_detail(appointment_id):
  """Provide HTML page with a given appointment."""
  # Query: get Appointment object by ID.
  appt = db.session.query(Appointment).get(appointment_id)
  if appt is None:
    # Abort with Not Found.
    abort(404)
  return render_template('appointment/detail.html',
    appt=appt)
```

1.  让我们采用相同的宏方法来显示一个约会。模板宏将为我们提供一个工具，以便在任何需要时显示一个约会。在`templates/appointment/common.html`中：

```py
{% macro detail(appt,
  link_title=false,
  show_edit=true,
  show_description=true) %}
<div class="appointment-detail">
  {% if link_title %}
    <h3>
      <a href="{{ url_for('appointment_detail',
        appointment_id=appt.id) }}">
        {{ appt.title }}</a>
    </h3>
  {% else %}
    <h3>{{ appt.title }}</h3>
  {% endif %}
  {% if appt.location %}
    <p><i class="icon-home"></i> {{ appt.location }}</p>
  {% endif %}
  {% if appt.allday %}
    <p><i class="icon-calendar"></i> {{ appt.start | date }}</p>
  {% else %}
    <p><i class="icon-calendar"></i> {{ appt.start | datetime }}
      for {{ appt.duration | duration }}</p>
  {% endif %}
  {% if show_edit %}
  <div class="edit-controls">
    <i class="icon-pencil"></i>
    <a href="{{ url_for('appointment_edit',
      appointment_id=appt.id) }}">Edit</a>
    <span class="inline-pad"></span>
    <i class="icon-trash"></i>
    <a class="appointment-delete-link" href="#"
      data-delete-url="{{ url_for('appointment_delete',
      appointment_id=appt.id) }}">Delete</a>
  </div>
  {% endif %}
  {% if show_description and appt.description %}
  <div class="row">
    <p class="span5">{{ appt.description | nl2br }}</p>
  </div>
  {% endif %}
</div>
{% endmacro %}
```

1.  然后我们可以在`templates/appointment/detail.html`中使用宏：

```py
{% extends 'base.html' %}

{% from 'appointment/common.html' import detail %}

{% block title %}
  {{ appt.title }}
{% endblock title %}

{% block main %}
<div class="row">
  <div class="span12">
    {{ detail(appt) }}
  </div>
</div>
{% endblock main %}
```

## 它是如何工作...

如果请求的数据库 ID 不存在，我们告诉 Flask 中止并返回**404 Not Found**响应。`flask.abort`函数被实现为一个异常，因此当调用`abort`时，Python 将停止当前函数的执行。

我们还将实现一些用于 Jinja 的实用程序，名为过滤器，它可以在 Python 对象进入模板输出之前格式化 Python 对象的输出，格式为`{{ foo | filter }}`。我们将构建`date`、`datetime`、`duration`和`nl2br`过滤器。在首次实现这些过滤器时，可以简单地从模板中省略它们；使用`{{ appt.start }}`代替`{{ appt.start | date }}`。这将让您更清楚地了解我们为什么要构建这个过滤器。

对于可能不存在的单个字段，我们使用 Python 的`or`行为。当用于显示或赋值时，`or`表达式是一种"快捷方式"，直到命中一个真值，并且使用该值。这种方法让我们以一种简单的方式提供默认的显示值。

# 编辑记录（中级）

现在我们提供一个用于现有约会的编辑页面，如下截图所示：

![编辑记录（中级）](img/9628OS_01_02.jpg)

## 准备就绪

我们将继续从`sched`目录和`templates`目录的`app.py`文件中进行工作。

## 如何做...

1.  编辑工作流是添加和详细视图的混合。我们获取一个约会，如果它存在，我们显示一个表单来编辑它。在`app.py`中：

```py
@app.route('/appointments/
  <int:appointment_id>/edit/,
  methods=['GET', 'POST'])

def appointment_edit(appointment_id):
  """Provide HTML form to edit a given appointment."""
  appt = db.session.query(Appointment).get(appointment_id)
  if appt is None:
    abort(404)
  form = AppointmentForm(request.form, appt)
  if request.method == 'POST' and form.validate():
    form.populate_obj(appt)
    db.session.commit()
    # Success. Send the user back to the detail view.
    return redirect(url_for('appointment_detail',
      appointment_id=appt.id))
    return render_template('appointment/edit.html', form=form)
```

1.  在`templates/appointment/edit.html`中：

```py
{% if request.endpoint.endswith('edit') %}
  {% set title = 'Edit Appointment' %}
{% else %}
  {% set title = 'Add Appointment' %}
{% endif %}
```

## 它是如何工作的...

我们重用与创建表单相同的模板，但是我们使用**编辑**而不是**添加**。由于这是非常简单的逻辑，我们可以在模板中使用`if`语句。Flask 允许我们在模板中检查请求对象。如果您发现您的项目需要大量的请求检查，您应该考虑一种设计模式，将这种逻辑移回到您的 Python 代码中。

# 列出所有记录（简单）

我们不希望用户跟踪他们约会的数据库 ID，因此我们提供了一个视图，它将列出他们所有的约会，如下面的屏幕截图所示：

![列出所有记录（简单）](img/9628OS_01_04.jpg)

## 准备工作

我们将继续从`sched`目录的`app.py`文件和`templates`目录中进行操作。

## 如何做...

1.  我们在`app.py`中查询所有约会：

```py
@app.route('/appointments/')
def appointment_list():
  """Provide HTML listing of all appointments."""
  # Query: Get all Appointment objects, sorted by date.
  appts = (db.session.query(Appointment)
    .order_by(Appointment.start.asc()).all())
  return render_template('appointment/index.html',
    appts=appts)
```

1.  在`templates/appointment/index.html`中：

```py
{% extends 'base.html' %}

{% from 'appointment/common.html' import detail %}

{% block title %}Appointments{% endblock title %}

{% block main %}
<div class="row">
  {% for appt in appts %}
    <div class="span12">
    {{ detail(appt,
      link_title=true,
      show_edit=false,
      show_description=false) }}
    </div>
  {% else %}
    <h3 class="span12">No appointments.</h3>
  {% endfor %}
</div>
{% endblock main %}
```

## 它是如何工作的...

我们按约会开始时间的升序查询所有约会。SQLAlchemy 声明类包括在列属性上的助手，让我们提供 SQL 子句，这里使用`.asc()`来指示`sort`字段。在约会列表模板中，我们遍历所有约会记录，并重用我们的显示宏，并关闭仅适用于详细视图的功能，使用我们创建的调用签名。

## 还有更多...

我们保持了约会列表的简单。如果你对数据库标识符不熟悉，你可能会想知道约会的 URL 是如何生成的。

### 数据库标识符

关于这些数据库 ID 的说明：它们是由我们的数据库管理系统和 SQLAlchemy 系统生成的，它们在我们的应用程序之外是没有意义的。ID 对于 URL 很有用，但对于人类的记忆来说并不好。这个视图是一个可以让你发挥创意并使用日、周、月和年的日历隐喻的地方。你可以这样做，但这主要存在于 HTML、CSS 和 JavaScript 中。对于我们简单的调度程序应用程序，我们只是在列表中列出约会。

# 删除记录（高级）

我们已经完成了我们的约会交互。等等，我们如何删除约会？

## 准备工作

我们将继续从`sched`的`app.py`文件和`templates`目录中进行操作。

## 如何做...

1.  在`app.py`中构建一个 Ajax 处理程序：

```py
@app.route('/appointments/
  <int:appointment_id>/delete/',
  methods=['DELETE'])

def appointment_delete(appointment_id):
  """Delete record using HTTP DELETE, respond with JSON."""
  appt = db.session.query(Appointment).get(appointment_id)
  if appt is None:
    # Abort with Not Found, but with simple JSON response.
    response = jsonify({'status': 'Not Found'})
    response.status = 404
    return response
  db.session.delete(appt)
  db.session.commit()
  return jsonify({'status': 'OK'})
```

1.  一点 jQuery 将使所有删除链接使用这个处理程序，放在`templates/base.html`中：

```py
<script>
 $(function() {
  $(".appointment-delete-link").on("click", function() {
    var delete_url = $(this).attr('data-delete-url');
    $.ajax({
      url: delete_url,
      type: 'DELETE',
      success: function(response) {
        if (response.status == 'OK') {
          window.location = {{ url_for('appointment_list') }};
        } else {
        alert('Delete failed.')
      }
    }
    });
    return false;
  });
});
</script>
```

## 它是如何工作的...

我们提供了我们的第一个视图函数，它不会呈现 HTML，而是使用最小的 Ajax 与 JSON 交互。它接受 HTTP DELETE 请求并执行工作。我们不必通过 GET 请求公开删除功能，这样用户就不会因为浏览而意外删除数据库记录。当我们发布的代码可能被搜索引擎和其他机器人访问时，这一点尤为重要。爬取我们所有的页面将删除我们整个数据库！Flask 提供了一个`jsonify`函数，将 Python 字典转换为 JSON 响应。

如何让浏览器发送 DELETE 请求？用 JavaScript。jQuery 库使得 Ajax 调用比单独使用 JavaScript 更简单。我们添加一个 jQuery 钩子，它将获取所有删除链接并在点击时提交一个 ajax 调用。点击回调从删除链接获取删除 URL，并将其作为 DELETE 请求发送。成功后，它将当前浏览器窗口重定向到约会列表。通过将此脚本放入基本模板中，它将使约会列表和详细页面上的所有删除链接都可用。

## 还有更多...

我们在这里只简要介绍了 Ajax；它值得一本整书。

### Ajax

Ajax 简单地意味着我们呈现一个网页，并让浏览器与我们的网络服务通信，而不一定要重新加载页面。也就是说，呈现页面一次，然后更新页面上显示的内容，而不刷新整个页面。最初这被称为 AJAX，意为“异步 JavaScript 和 XML”，但已经成为 Web 开发人员之间的常用词，并且可以表示任何序列化格式，这里使用 JSON 而不是 XML。

# 在 Jinja 中使用自定义模板过滤器（高级）

为了完成我们的模板，我们需要清理一些字段的显示。

## 准备工作

我们从`sched`目录中的`filters.py`文件转到`app.py`并再次进行操作。

## 如何做...

1.  我们希望以清晰的格式显示时间。在`filters.py`中：

```py
def do_datetime(dt, format=None):
  """Jinja template filter to format a datetime object."""
  if dt is None:
    # By default, render an empty string.
    return ''
  if format is None:
    # No format is given in the template call.
    # Use a default format.
    #
    # Format time in its own strftime call in order to:
    # 1\. Left-strip leading 0 in hour display.
    # 2\. Use 'am'/'pm' (lower case) instead of 'AM'/'PM'.
    formatted_date = dt.strftime('%Y-%m-%d - %A')
    formatted_time =\
      dt.strftime('%I:%M%p').lstrip('0').lower()
    formatted = '%s at %s' %\
      (formatted_date, formatted_time)
  else:
    formatted = dt.strftime(format)
  return formatted
```

1.  在这里添加更多的过滤器，并提供一个钩子来初始化 Flask 应用程序。

```py
def init_app(app):
  """Initialize a Flask application with custom filters."""
  app.jinja_env.filters['datetime'] = do_datetime
```

1.  在`app.py`中使用初始化钩子：

```py
from sched import filters
filters.init_app(app)
```

## 它是如何工作的...

Jinja 支持一个环境，其中 Python 函数作为过滤器提供在模板中。我们的预约中有一个`datetime`字段，因此我们为其提供了一些格式。您可以在模板中直接调用属性和方法，`{{ dt.strftime('%Y-%m-%d') }}`，但是定义一个过滤器让我们指定如何在中心格式化所有日期，同时仍然暴露参数。您可以使用我们的 datetime 过滤器调用`{{ dt | datetime }}`来获得我们定义的默认功能，以及`{{ dt | datetime('%Y-%m-%d') }}`，其中模板调用中的参数作为值之后作为参数传递。

## 还有更多...

您可以在 Packt 网站上提供的源代码文件中找到过滤器的完整代码。

### 附加过滤器

`sched`的完整源代码包括默认日期格式化的过滤器，预约记录的持续时间格式化，以及`nl2br`过滤器，以保留预约描述字段中用户输入的换行符（因为 HTML 会规范化空格）。

# 发送错误响应（简单）

在本节中，我们将向用户发送一个带有样式的错误页面。

## 准备工作

我们正在从项目的`sched`目录中的`app.py`文件中工作，并且在`sched`内的`templates`目录中。在`templates`目录内创建一个名为`error`的目录。

## 如何做...

1.  当用户访问**未找到**页面时，我们希望显示一个页面，指引他们返回预约列表。您可以告诉 Flask 如何渲染错误情况的响应，通常是针对 HTTP 错误代码。我们将提供一个自定义的 404 未找到页面，如下面的截图所示：![如何做...](img/9628OS_01_05.jpg)

1.  Flask 提供了`app.errorhandler`来告诉 Flask 如何处理某些类型的错误，可以使用`4xx`或`5xx`的 HTTP 状态码或 Python 异常类。装饰一个接受`Exception`实例并返回响应的函数。确保在 Flask 响应中包含 HTTP 状态码，通过在渲染模板字符串的元组响应中提供`404`。在下面的代码中，我们渲染一个未找到的模板，其中提供了返回预约列表的链接。在`app.py`中：

```py
@app.errorhandler(404)
def error_not_found(error):
  return render_template('error/not_found.html'), 404
```

1.  在`templates/error/not_found.html`中：

```py
{% extends 'base.html' %}

{% block title %}Not Found{% endblock title %}

{% block main %}
<h2>Not Found</h2>
{% endblock main %}
```

基本模板中的`navbar`元素将帮助用户导航回已知页面。

## 它是如何工作的...

`app.errorhandler`装饰器接受异常类或 HTTP 状态码作为整数。以下是一些 HTTP 状态码，供您开始使用。您可以在视图函数中使用`flask.abort(status_code)`触发这些状态码的错误响应，并定义一个自定义的`app.errorhandler`来提供样式化的响应。

+   400 错误的请求

+   401 未经授权

+   403 禁止

+   404 未找到

+   405 方法不允许（您可能在路由中忘记了方法）

+   410 已经消失（不会再回来）

+   500 内部服务器错误

查看[www.w3.org/Protocols/rfc2616/rfc2616-sec10.html](http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html)获取完整列表。

`500`错误处理程序将在您的代码出现未捕获的异常时显示。

## 还有更多...

Flask 提供了一个非常有用的调试器，用于检查开发中的错误。

### 处理特定的异常

您可以在 Python 异常上提供自定义错误页面：

```py
class SchedulingException(Exception):
  """Some application-specific error has occurred."""
  pass

@app.errorhandler(SchedulingException)
def scheduling_exception_handler(error):
  return 'Just a demo of app.errorhandler...', 500

@app.route('/test_error/')
def test_error():
  raise SchedulingException('Use my custom error handler.')
```

### Flask 的调试器

这是讨论 Flask 调试器的好地方。当 Flask 应用程序的`app.config['DEBUG']`设置为`True`时，应用程序代码中的任何未捕获异常都将在浏览器中显示一个开发人员友好的调试器。这个调试器使 Python 堆栈跟踪交互式，让您在堆栈帧的任何点上运行代码并查看上下文中的代码。

在`app.py`中：

```py
app = Flask(__name__)
app.config['DEBUG'] = True
```

您可以故意触发异常，快速查看您的堆栈。在异常中写入一个异常并在 Web 浏览器中访问路由。我最喜欢的两个异常是：

```py
1/0 # Divide by zero to get an exception.
raise Exception("Don't Panic!")
```

只在受保护的机器上进行开发时使用这个，因为交互式解释器允许从浏览器执行 Python 代码。如果交互式解释器在浏览器中变得无响应，要么开发服务器已关闭，要么已经转移到新的请求，忘记了你的错误。

# 用户认证（高级）

到目前为止，我们假设任何可以访问`sched`应用程序的人都应该能够在数据库中创建、显示、编辑、列出和删除约会记录。我们的 Flask 应用程序需要对用户进行身份验证并保护我们的数据库。

## 准备工作

我们回到项目的`sched`目录中的`models.py`文件。

## 如何做...

1.  我们从`models.py`中具有唯一电子邮件地址的用户的数据库记录开始：

```py
from datetime import datetime

from sqlalchemy import Column
from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import synonym
from sqlalchemy.ext.declarative import declarative_base
from werkzeug import check_password_hash
from werkzeug import generate_password_hash

Base = declarative_base()

class User(Base):
  """A user login, with credentials and authentication."""
  __tablename__ = 'user'

  id = Column(Integer, primary_key=True)
  created = Column(DateTime, default=datetime.now)
  modified = Column(DateTime, default=datetime.now,
    onupdate=datetime.now)

  name = Column('name', String(200))
  email = Column(String(100), unique=True, nullable=False)
  active = Column(Boolean, default=True)
```

1.  然后，我们添加一种存储密码哈希的方法（继续在用户定义中）：

```py
  _password = Column('password', String(100))

  def _get_password(self):
    return self._password

  def _set_password(self, password):
    if password:
      password = password.strip()
    self._password = generate_password_hash(password)

  password_descriptor = property(_get_password,
    _set_password)
  password = synonym('_password',
    descriptor=password_descriptor)
```

1.  最后，我们添加一种验证密码的方法（继续在用户定义中）：

```py
  def check_password(self, password):
    if self.password is None:
      return False
    password = password.strip()
    if not password:
      return False
    return check_password_hash(self.password, password)

  @classmethod
  def authenticate(cls, query, email, password):
    email = email.strip().lower()
    user = query(cls).filter(cls.email==email).first()
    if user is None:
      return None, False
    if not user.active:
      return user, False
    return user, user.check_password(password)
```

## 工作原理...

我们需要一个地方来存储用户记录，这样我们就可以验证认证请求是否针对我们应用程序中的有效用户。然而，我们不希望以明文存储密码，这样任何具有对数据库表的读取权限的人都会知道如何以有效用户的身份登录应用程序。第一次遇到这个想法时可能看起来不重要，但是你不希望承担存储每个人密码的责任，特别是如果用户在不同应用程序中重复使用密码的话。（这应该是为什么这是一个坏主意的提示。）

Werkzeug 提供了哈希密码和验证提供的密码是否与哈希匹配的实用程序。我们存储哈希而不是明文密码。每当在`User`实例上设置`user.password`时，密码立即使用 Python 描述符和 SQLAlchemy 的同义钩子进行哈希。同义词让我们可以让`user.password`的行为就像每个其他 SQLAlchemy 对象属性一样，但是当访问`user.password`时，获取值将调用`_get_password`，设置它将调用`_set_password`，这将存储一个哈希值。

现在我们可以使用电子邮件、密码和 SQLAlchemy 查询对象进行身份验证，`User.authenticate(query, email, password)`，它返回一个`user, bool`对（元组），其中`user`是匹配的用户对象，布尔值表示身份验证是否有效。

## 还有更多...

我们需要将电子邮件/密码身份验证集成到我们的应用程序中。

### 约会关系

我们可以建立约会和用户之间的关系，这样当我们加载日程表时，只显示当前用户的约会。在`models.py`中更新`Appointment`类，包括与`User`的外键关系：

```py
class Appointment(Base):
  # ... existing code.
  user_id = Column(Integer,
    ForeignKey('user.id'), nullable=False)
  user = relationship(User, lazy='joined', join_depth=1)
  # ... existing code.
```

现在访问`Appointment`对象将提供`appointment.user`属性来访问该约会的所有者的`User`对象。因为我们使用了`ForeignKey`，所以当访问`Appointment`对象时，SQLAlchemy 知道如何获取相关的`User`记录。

### 打开一个会话

HTTP 是无状态的，这意味着一个请求没有关于先前请求的信息。我们需要打开一个会话，该会话从一个请求保留到另一个请求，这样我们就可以在一个请求中对用户进行身份验证，并在用户在给定的会话中进行的每个后续请求中使用该身份验证。

在下一节中，我们将看到如何处理会话，向我们的代码添加一个`current_user`对象，该对象将表示与当前经过身份验证的用户匹配的`User`数据库对象。现在我们已经在`models.py`中为`User`和`Appointment`之间添加了关系，我们需要在每种情况下更新`app.py`，在那里我们创建或检索`Appointment`对象。将`Appointment`对象创建行更改为：

```py
appt = Appointment(user_id=current_user.id)
```

然后验证当前`Appointment`是否适用于当前用户在检索记录时：

```py
if appt.user_id != current_user.id:
  abort(403)
```

### 数据库迁移

更改`models.py`意味着我们需要更新我们在开发中使用的数据库的数据库结构，或者如果我们已经部署了`sched`应用程序，则是生产数据库。如果我们所做的唯一更改是添加表，那么我们可以简单地再次调用`Base.metadata.create_all(engine)`。

有时在开发中，这是最简单的方法：只需销毁开发数据库，再次运行`create_all`，然后重新开始添加数据。设置初始开发或测试数据可能很麻烦，因此您可能希望投资于**fixtures**，在运行应用程序开发服务器之前加载它们。Python 社区中有几个项目可以做到这一点，包括`fixture`在[pypi.python.org/pypi/fixture](http://pypi.python.org/pypi/fixture)。

对于列的更改，我们需要提供一个自动化来更改实时数据库，特别是对于正在使用的生产数据库。如果您熟悉 SQL，可以编写自己的`ALTER TABLE`语句并提供自己的脚本。如果您想要一个工具包来管理这个，SQLAlchemy 的创始人 Mike Bayer 已经编写了 Alembic 作为在 SQLAlchemy 中执行迁移的工具，[alembic.readthedocs.org](http://alembic.readthedocs.org)。

### 使用现有服务进行身份验证

新用户如何开始？对于`sched`中的用户身份验证，您需要提供一种注册用户的方法。这可以是从 Python 交互解释器中创建`User`对象，也可以是应用程序内的用户注册表单。

或者，您可以将`User`实现减少到您需要的核心应用程序数据，并使用现有服务来在外部对您的用户进行身份验证，首次进行此类身份验证时创建新的`User`对象。这将使用户无需记住另一个密码，并加快用户获取过程。

OpenID（[openid.net](http://openid.net)）提供了一个开放标准来执行这种类型的身份验证。流行的服务，如谷歌，有他们自己的身份验证和授权的文档，使用现有的用户账户，[developers.google.com/accounts/](http://developers.google.com/accounts/)。

# 处理会话和用户（中级）

由于 HTTP 是无状态的，我们需要使用会话在请求之间跟踪一些数据。

## 准备就绪

我们将继续从`sched`目录中的`app.py`文件和`models.py`文件中进行工作。

## 如何做...

1.  Flask 提供了一个`session`对象，它的行为类似于 Python 字典，并且在请求之间自动持久化。您可以在您的 Flask 应用程序代码中：

```py
from flask import session
# ... in a request ...
session['spam'] = 'eggs'
# ... in another request ...
spam = session.get('spam') # 'eggs'
```

1.  Flask-Login 提供了一种简单的方法来跟踪 Flask 会话中的用户。更新 requirements.txt：

```py
Flask
Flask-Login
Flask-Script
Flask-SQLAlchemy
WTForms
```

1.  然后：

```py
$ pip install -r requirements.txt

```

1.  然后我们可以将 Flask-Login 加载到`sched`的请求处理中，在`app.py`中：

```py
from flask.ext.login import LoginManager, current_user
from flask.ext.login import login_user, logout_user
from sched.models import User

# Use Flask-Login to track current user in Flask's session.
login_manager = LoginManager()
login_manager.setup_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
  """Flask-Login hook to load a User instance from ID."""
  return db.session.query(User).get(user_id)
```

1.  Flask-Login 在`models.py`中的`class User`内需要`User`对象上的四种方法：

```py
  def get_id(self):
    return str(self.id)

  def is_active(self):
    return True

  def is_anonymous(self):
    return False

  def is_authenticated(self):
    return True
```

### 提示

如果您喜欢使用其默认实现，Flask-Login 提供了一个**UserMixin**（`flask.ext.login.UserMixin`）。

1.  然后我们提供路由来在经过身份验证时登录用户并注销。在`app.py`中：

```py
@app.route('/login/', methods=['GET', 'POST'])
def login():
  if current_user.is_authenticated():
    return redirect(url_for('appointment_list'))
  form = LoginForm(request.form)
  error = None
  if request.method == 'POST' and form.validate():
    email = form.username.data.lower().strip()
    password = form.password.data.lower().strip()
    user, authenticated = \
      User.authenticate(db.session.query, email,
        password)
    if authenticated:
      login_user(user)
      return redirect(url_for('appointment_list'))
    else:
      error = 'Incorrect username or password.'
    return render_template('user/login.html',
      form=form, error=error)

@app.route('/logout/')
def logout():
  logout_user()
  return redirect(url_for('login'))
```

1.  然后我们在`app.py`中装饰每个需要有效用户的视图函数：

```py
from flask.ext.login import login_required
@app.route('/appointments/')
@login_required
def appointment_list():
  # ...
```

## 它是如何工作的...

在`login_user`中，Flask-Login 从`User.get_id`获取用户对象的 ID，并将其存储在 Flask 的会话中。然后，Flask-Login 设置一个`before_request`处理程序，将用户实例加载到`current_user`对象中，使用我们提供的`load_user`钩子。`logout_user`函数然后从会话中删除相关的部分。

如果没有用户登录，那么`current_user`将提供一个匿名用户对象，导致`current_user.is_anonymous()`返回`True`，`current_user.is_authenticated()`返回`False`，这允许应用程序和模板代码基于用户是否有效进行逻辑。 （Flask-Login 将`current_user`放入所有模板上下文中。）您可以使用`User.is_active`使用户帐户无效，而不必实际删除它们，返回适当的`False`。

使用`login_required`装饰的视图函数将在当前用户未经身份验证时将用户重定向到登录视图，而不调用装饰的函数。

## 还有更多...

Flask 的会话支持显示消息和防止请求伪造。

### 闪存消息

当您想要显示一个简单的消息来指示成功的操作或快速失败时，您可以使用 Flask 的闪存消息，它将消息加载到会话中直到被检索。在应用程序代码中，在请求处理代码内：

```py
from flask import flash
flash('Sucessfully did that thing.', 'success')

```

在模板代码中，您可以使用`'success'`类别进行条件显示：

```py
{% for cat, m in get_flashed_messages(with_categories=true) %}
 <div class="alert">{{ m }}</div>
{% endfor %}
```

### 跨站点请求伪造保护

恶意网络代码将尝试伪造数据更改请求以访问其他网络服务。为了防止伪造，您可以将一个随机化的令牌加载到会话和 HTML 表单中，当两者不匹配时拒绝请求。这在 Flask-SeaSurf 扩展中提供，[pythonhosted.org/Flask-SeaSurf/](http://pythonhosted.org/Flask-SeaSurf/)或 Flask-WTF 扩展（集成了 WTForms），[pythonhosted.org/Flask-ETF/](http://pythonhosted.org/Flask-ETF/)。

# 部署到世界（高级）

一旦满意您的应用程序，您可以部署您的应用程序，使其对外开放。

## 准备好

您需要一台联机且可供目标用户访问的计算机。安装 Python 并将`requirements.txt`放在您可以访问命令行的文件夹中，就像我们为您的开发环境所做的那样（virtualenv 适用于生产环境），然后创建数据库表。您可以部署到任何操作系统：Windows 和任何类 Unix 系统（包括 Mac OS X）。

## 如何做...

1.  从干净的 Ubuntu 12.04 服务器安装开始：

```py
$ sudo apt-get update && sudo apt-get install nginx
$ # Load config into /etc/nginx/sites-available/default
$ sudo service nginx restart
$ pip install gunicorn
$ gunicorn --access-logfile - -b 127.0.0.1:5000 sched.app:app

```

1.  在 virtualenv 中使用`pip`和`gunicorn`程序。将其加载到 nginx 配置中，如果您有域名，请将 localhost 更改为您的域名：

```py
server {
  listen 80;
  server_name localhost;
  access_log  /var/log/nginx/access.log;
  error_log  /var/log/nginx/error.log;
  location / {
    proxy_pass http://127.0.0.1:5000/;
    proxy_redirect off;
    proxy_set_header Host            $host;
    proxy_set_header X-Real-IP       $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  }
}
```

1.  然后将 gunicorn 进程添加到服务器的启动顺序中，而不是手动运行它。请参阅您操作系统的文档。对于类 Unix 系统，supervisord 提供了长时间运行应用程序的管理，[supervisord.org](http://supervisord.org)。

## 它是如何工作的...

每当 nginx 接收到匹配请求时，它会将代理请求发送到运行在端口`5000`上的 HTTP 服务器，这在我们的应用程序中是 gunicorn，并传递响应。请注意，gunicorn 不会在代码更新时自动重新加载您的应用程序。您可以按照[docs.gunicorn.org/en/latest/faq.html](http://docs.gunicorn.org/en/latest/faq.html)中描述的过程信号重新加载。

有关其他文档，请参阅[docs.gunicorn.org/en/latest/deploy.html](http://docs.gunicorn.org/en/latest/deploy.html)。

## 还有更多...

部署有许多选项；配置完全取决于您的项目。

### 使用 nginx 处理静态文件

Flask 提供了一个方便的静态文件处理程序。现在我们已经有了一个优化的 HTTP 服务器，我们可以让它直接提供静态文件。以下配置块显示了如何使用 nginx 提供静态文件并将所有其他请求代理到 Flask。根据您的设置调整配置：

+   将`/var/www/myproject`更改为包含我们项目布局中`sched`文件夹的文件路径。

+   如果您更改了 Flask 中的默认静态路由，请更改`/static/`。

更新的服务器配置，用于处理静态文件：

```py
server {
  listen 80;
  server_name localhost;
  access_log  /var/log/nginx/access.log;
  error_log  /var/log/nginx/error.log;
  location  /static/ {
    # /static/ is appended to root path on the request.
    root /var/www/myproject/sched/;
  }
  location / {
    proxy_pass http://127.0.0.1:5000/;
    proxy_redirect off;
    proxy_set_header Host            $host;
    proxy_set_header X-Real-IP       $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  }
}
```

### 可在任何 OS 上进行 HTTP 代理部署

我更喜欢使用独立的操作系统进程通过 HTTP 代理来运行应用程序。大多数强大的 HTTP 服务器都包括一种方式来代理请求到另一个 web 服务，就像我们用 nginx 演示的那样。gunicorn 只能在类 Unix 系统上运行，但 cherrypy ([cherrypy.org](http://cherrypy.org))提供了一个跨平台的 WSGI 服务器。您可以使用以下示例中的脚本启动应用程序：

```py
from cherrypy.wsgiserver import CherryPyWSGIServer
from sched.app import app
server = CherryPyWSGIServer(('127.0.0.1', 5000), app)
server.start()
```

### Windows 服务器部署

要在 Windows Server IIS 中直接运行 Python 应用程序，请参阅[NWSGI 项目](http://nwsgi.codeplex.com)。

### 其他部署选项

请参阅[flask.pocoo.org/docs/deploying/](http://flask.pocoo.org/docs/deploying/)上的 Flask 文档。
