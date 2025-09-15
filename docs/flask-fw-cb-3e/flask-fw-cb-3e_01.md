

# 第一章：Flask 配置

这章入门指南将帮助我们了解 Flask 可以以不同的方式配置，以满足项目的各种需求。Flask 是 “*The Python micro framework for building web applications*” (pallets/Flask, [`github.com/pallets/flask`](https://github.com/pallets/flask))。

那么，为什么 Flask 被称为 **microframework**？这难道意味着 Flask 缺乏功能，或者意味着您的网络应用程序的完整代码必须包含在一个文件中？并非如此！microframework 这个术语仅仅指的是 Flask 旨在保持其框架核心小而高度可扩展。这使得编写应用程序或扩展既容易又灵活，并赋予开发者选择他们希望为应用程序使用的配置的能力，而不对数据库、模板引擎、管理界面等选择施加任何限制。在本章中，您将学习几种设置和配置 Flask 的方法。

重要信息

这本书整个使用 *Python 3* 作为 Python 的默认版本。Python 2 在 2019 年 12 月 31 日停止了支持，因此本书不支持 Python 2。建议您在学习本书时使用 Python 3，因为许多配方可能在 Python 2 上无法工作。

同样，在编写本书时，Flask 2.2.x 是最新版本。尽管本书中的许多代码可以在 Flask 的早期版本上运行，但建议您使用 2.2.x 及以上版本。

开始使用 Flask 只需几分钟。设置一个简单的 *Hello World* 应用程序就像做饼一样简单。只需在您的计算机上任何可以访问 `python` 或 `python3` 的位置创建一个文件，例如 `app.py`，然后包含以下脚本：

```py
from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello to the World of Flask!'
if __name__ == '__main__':
    app.run()
```

现在，需要安装 Flask；这可以通过 `pip` 或 `pip3` 完成。如果您遇到访问问题，可能需要在基于 Unix 的机器上使用 `sudo`：

```py
$ pip3 install Flask
```

重要

这里提供的代码和 Flask 安装示例只是为了展示 Flask 可以多么容易地使用。要设置适当的开发生态，请遵循本章中的配方。

前面的代码片段是一个完整的基于 Flask 的网络应用程序。在这里，导入的 `Flask` 类的实例在这个代码中成为 `app`，它成为我们的 WSGI 应用程序，并且由于这是一个独立模块，我们将 `__name__` 字符串设置为 `'__main__'`。如果我们将其保存为名为 `app.py` 的文件，那么应用程序可以通过以下命令简单地运行：

```py
$ python3 app.py
* Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

现在，如果我们打开浏览器并输入 `http://127.0.0.1:5000/`，我们就可以看到我们的应用程序正在运行。

或者，可以通过使用 `flask run` 或 Python 的 `-m` 开关与 Flask 来运行应用程序。在遵循此方法时，可以跳过 `app.py` 的最后两行。请注意，以下命令仅在当前目录中存在名为 `app.py` 或 `wsgi.py` 的文件时才有效。如果没有，则包含 `app` 对象的文件应作为环境变量导出，即 `FLASK_APP`。作为最佳实践，在两种情况下都应这样做：

```py
$ export FLASK_APP=app.py
$ flask run
* Running on http://127.0.0.1:5000/
```

或者，如果你决定使用 `-m` 开关，它将如下所示：

```py
$ export FLASK_APP=app.py
$ python3 -m flask run
* Running on http://127.0.0.1:5000/
```

小贴士

不要将你的应用程序文件保存为 `flask.py`；如果你这样做，在导入时它将与 Flask 本身冲突。

在本章中，我们将介绍以下食谱：

+   使用 `virtualenv` 设置我们的环境

+   处理基本配置

+   配置基于类的设置

+   组织静态文件

+   使用 `instance` 文件夹进行特定于部署的操作

+   视图和模型的组合

+   使用蓝图创建模块化 Web 应用程序

+   使用 `setuptools` 使 Flask 应用程序可安装

# 技术要求

在一般情况下，与 Flask 和 **Python** 一起工作相当简单，不需要很多依赖项和配置。在本书的大部分章节中，所有必需的软件包都将在相关食谱中提及。我将在相关章节中提及更具体的要求。一般来说，您需要以下内容：

+   一台不错的计算机，最好是基于 *UNIX* 的操作系统，如 *Linux* 或 *macOS*。您也可以使用 Windows，但这需要一些额外的设置，但这本书的范围之外。

+   选择一个代码编辑器作为 IDE。我使用 *Vim* 和 *Visual Studio Code*，但任何支持 *Python* 的编辑器都可以，只要它支持即可。

+   一个良好的互联网连接，因为您将下载软件包及其依赖项。

所有代码均可在 GitHub 上免费获取，网址为 [`github.com/PacktPublishing/Flask-Framework-Cookbook-Third-Edition`](https://github.com/PacktPublishing/Flask-Framework-Cookbook-Third-Edition)。此 GitHub 仓库包含本书所有章节的代码，分别存放在相应的文件夹中。

# 设置虚拟环境

Flask 可以简单地使用 `pip`/`pip3` 或 `easy_install` 在全局范围内安装，但最好使用 `venv` 设置应用程序环境，它将管理在单独的环境中，并且不会让任何库的不正确版本影响任何应用程序。在本食谱中，我们将学习如何创建和管理这些环境。

## 如何操作...

使用 `venv` 模块创建虚拟环境。因此，只需在您选择的文件夹中创建一个名为 `my_flask_env`（或您选择的任何其他名称）的新环境即可，您希望您的开发环境所在的位置。这将创建一个具有相同名称的新文件夹，如下所示：

```py
$ python3 -m venv my_flask_env
```

从 `my_flask_env` 文件夹内部运行以下命令：

```py
$ source my_flask_env/bin/activate
$ pip3 install flask
```

这将激活我们的环境并在其中安装`flask`。现在，我们可以在该环境中对应用程序进行任何操作，而不会影响任何其他 Python 环境。

## 它是如何工作的...

到目前为止，我们已经多次使用`pip3 install flask`。正如其名所示，该命令指的是安装 Flask，就像安装任何 Python 包一样。如果我们稍微深入到通过`pip3`安装 Flask 的过程，我们会看到安装了几个包。以下是 Flask 包安装过程的概述：

```py
$ pip3 install flask
Collecting Flask
...........
...........
Many more lines.........
...........
Installing collected packages: zipp, Werkzeug, MarkupSafe, itsdangerous, click, Jinja2, importlib-metadata, Flask
Successfully installed Flask-2.1.2 Jinja2-3.1.2 MarkupSafe-2.1.1 Werkzeug-2.1.2 click-8.1.3 importlib-metadata-4.11.4 itsdangerous-2.1.2 zipp-3.8.0
```

如果我们仔细查看前面的代码片段，我们会看到已经安装了多个包。在这些包中，有五个包，即`Werkzeug`、`Jinja2`、`click`、`itsdangerous`和`markupsafe`，是 Flask 所依赖的包，如果其中任何一个缺失，Flask 将无法工作。其他的是 Flask 依赖项所需的子依赖项。

## 还有更多...

在`venv`在*Python 3.3*中引入之前，`virtualenv`是用于创建和管理虚拟环境的标准库。`venv`是`virtualenv`的一个子集，并缺少`virtualenv`提供的某些高级功能。为了简化并保持本书的上下文，我将使用`venv`，但你也可以自由探索`virtualenv`和`virtualenvwrapper`。

## 参考信息

本节的相关参考资料如下：

+   [`pypi.python.org/pypi/Flask`](https://pypi.python.org/pypi/Flask)

+   [`pypi.python.org/pypi/Werkzeug`](https://pypi.python.org/pypi/Werkzeug)

+   [`pypi.python.org/pypi/Jinja2`](https://pypi.python.org/pypi/Jinja2)

+   [`pypi.python.org/pypi/itsdangerous`](https://pypi.python.org/pypi/itsdangerous)

+   [`pypi.python.org/pypi/MarkupSafe`](https://pypi.python.org/pypi/MarkupSafe)

+   [`pypi.python.org/pypi/click`](https://pypi.python.org/pypi/click)

更多关于`virtualenv`和`virtualenvwrapper`的信息，请参阅[`virtualenv.pypa.io/en/latest/`](https://virtualenv.pypa.io/en/latest/)和[`pypi.org/project/virtualenvwrapper/`](https://pypi.org/project/virtualenvwrapper/)。

# 处理基本配置

Flask 的一个优点是它很容易根据项目的需求配置 Flask 应用程序。在这个菜谱中，我们将尝试了解 Flask 应用程序可以以哪些不同的方式配置，包括如何从环境变量、Python 文件或甚至`config`对象中加载配置。

## 准备工作

在 Flask 中，配置变量存储在名为`config`的字典样式的`Flask`对象属性中。`config`属性是 Python 字典的子类，我们可以像任何字典一样修改它。

## 如何做到这一点...

要以调试模式运行我们的应用程序，例如，我们可以编写以下代码：

```py
app = Flask(__name__)
app.config['DEBUG'] = True
```

小贴士

`debug`布尔值也可以在 Flask`对象`级别而不是在`config`级别设置，如下所示：

`app.debug =` `True`

或者，我们可以将`debug`作为命名参数传递给`app.run`，如下所示：

`app.run(debug=True)`

在 Flask 的新版本中，调试模式也可以通过环境变量设置，`FLASK_DEBUG=1`。然后，我们可以使用`flask run`或 Python 的`-m`开关来运行应用程序：

`$` `export FLASK_DEBUG=1`

启用调试模式会在代码发生任何更改时自动重新加载服务器，并且在出现问题时还提供了非常有用的*Werkzeug*调试器。

Flask 提供了许多配置值。在本章的相关食谱中，我们将遇到它们。

随着应用程序的增大，需要将应用程序的配置管理在一个单独的文件中，如下例所示。在您使用的多数操作系统和开发环境中，这个文件不太可能是版本控制系统的一部分。因此，Flask 为我们提供了多种获取配置的方法。最常用的方法如下：

+   从 Python 配置文件（`*.cfg`），其中配置可以通过以下语句获取：

    ```py
    app.config.from_pyfile('myconfig.cfg')
    ```

+   从一个对象，其中配置可以通过以下语句获取：

    ```py
    app.config.from_object('myapplication.default_settings')
    ```

+   或者，要从运行此命令的同一文件中加载，我们可以使用以下语句：

    ```py
    app.config.from_object(__name__)
    ```

+   从环境变量，配置可以通过以下语句获取：

    ```py
    app.config.from_envvar('PATH_TO_CONFIG_FILE')
    ```

+   Flask 版本*2.0*新增了从通用配置文件格式（如**JSON**或**TOML**）加载的能力：

    ```py
    app.config.from_file('config.json', load=json.load)
    ```

```py
Alternatively, we can do the following:
app.config.from_file('config.toml', load=toml.load)
```

## 它是如何工作的...

Flask 设计为仅拾取以大写字母编写的配置变量。这允许我们在配置文件和对象中定义任何局部变量，其余的由 Flask 处理。

使用配置的最佳实践是在`app.py`中或通过应用程序中的任何对象设置一些默认设置，然后通过从配置文件中加载来覆盖它们。因此，代码将如下所示：

```py
app = Flask(__name__)
DEBUG = True
TESTING = True
app.config.from_object(__name__)
app.config.from_pyfile('/path/to/config/file')
```

# 使用基于类的设置进行配置

对于不同的部署模式，如生产、测试、预发布等，使用类的继承模式来布局配置是一种有效的方法。随着项目的增大，您可以有不同部署模式，每种模式可以有不同的配置设置，或者有一些设置将保持不变。在本食谱中，我们将学习如何使用基于类的设置来实现这种模式。

## 如何实现...

我们可以有一个具有默认设置的基类；然后，其他类可以简单地从基类继承，并覆盖或添加特定于部署的配置变量，如下例所示：

```py
class BaseConfig(object):
    'Base config class'
    SECRET_KEY = 'A random secret key'
    DEBUG = True
    TESTING = False
    NEW_CONFIG_VARIABLE = 'my value'
class ProductionConfig(BaseConfig):
    'Production specific config'
    DEBUG = False
    SECRET_KEY = open('/path/to/secret/file').read()
class StagingConfig(BaseConfig):
    'Staging specific config'
    DEBUG = True
class DevelopmentConfig(BaseConfig):
    'Development environment specific config'
    DEBUG = True
    TESTING = True
    SECRET_KEY = 'Another random secret key'
```

重要信息

在生产配置中，密钥通常存储在一个单独的文件中，因为出于安全原因，它不应成为版本控制系统的一部分。这应该在机器的本地文件系统中保留，无论是你的机器还是服务器。

## 它是如何工作的...

现在，我们可以在通过`from_object()`加载应用程序配置时使用任何前面的类。假设我们将前面的基于类的配置保存在一个名为`configuration.py`的文件中，如下所示：

```py
app.config.from_object('configuration.DevelopmentConfig')
```

总体来说，这使得管理不同部署环境下的配置更加灵活和容易。

# 组织静态文件

高效地组织静态文件，如 JavaScript、样式表、图像等，一直是所有 Web 框架关注的焦点。在本教程中，我们将学习如何在 Flask 中实现这一点。

## 如何操作...

Flask 推荐一种特定的方式来组织应用程序中的静态文件，如下所示：

```py
my_app/
    app.py
    config.py
    __init__.py
    static/
        css/
        js/
        images/
            logo.png
```

在模板中渲染时（比如，`logo.png`文件），我们可以使用以下代码引用静态文件：

```py
<img src='/static/images/logo.png'>
```

## 它是如何工作的...

如果在应用程序的根级别存在一个名为`static`的文件夹——即与`app.py`在同一级别——那么 Flask 将自动读取该文件夹的内容，无需任何额外配置。

## 还有更多...

或者，我们可以在定义`app.py`中的应用程序时提供一个名为`static_folder`的参数，如下所示：

```py
app = Flask(__name__,
    static_folder='/path/to/static/folder')
```

在前面的代码行中，`static`指的是应用程序对象上的`static_folder`的值。这可以通过提供 URL 前缀来修改，如下所示：

```py
app = Flask(
    _name_, static_url_path='/differentstatic',
    static_folder='/path/to/static/folder'
)
```

现在，要渲染静态文件，我们可以使用以下代码：

```py
<img src='/differentstatic/logo.png'>
```

总是使用`url_for`为静态文件创建 URL，而不是显式定义它们，这是一个好习惯，如下所示：

```py
<img src="img/{{ url_for('static', filename='logo.png') }}">
```

# 使用实例文件夹进行部署特定操作

Flask 还提供了一个用于配置的另一种方法，我们可以通过它有效地管理部署特定的部分。实例文件夹允许我们将部署特定的文件从受版本控制的应用程序中分离出来。我们知道配置文件可以针对不同的部署环境分开，例如开发和生产，但还有许多其他文件，如数据库文件、会话文件、缓存文件和其他运行时文件。在本教程中，我们将创建一个实例文件夹，它将充当此类文件的容器。按照设计，实例文件夹不会成为版本控制系统的一部分。

## 如何操作...

默认情况下，如果我们在应用程序级别有一个名为`instance`的文件夹，应用程序会自动选择实例文件夹，如下所示：

```py
my_app/
    app.py
    instance/
        config.cfg
```

我们还可以通过在应用程序对象上使用`instance_path`参数显式定义实例文件夹的绝对路径，如下所示：

```py
app = Flask(
    __name__,
    instance_path='/absolute/path/to/instance/folder')
```

要从实例文件夹中加载配置文件，我们可以在应用程序对象上使用 `instance_relative_config` 参数，如下所示：

```py
app = Flask(__name__, instance_relative_config=True)
```

这告诉应用程序从实例文件夹中加载配置文件。以下示例显示了如何配置此操作：

```py
app = Flask(
    __name__, instance_path='path/to/instance/folder',
    instance_relative_config=True
)
app.config.from_pyfile('config.cfg', silent=True)
```

## 它是如何工作的...

在前面的代码中，首先从给定的路径加载实例文件夹；然后从给定的实例文件夹中的 `config.cfg` 文件加载配置文件。在这里，`silent=True` 是可选的，用于抑制如果实例文件夹中没有找到 `config.cfg` 时出现的错误。如果没有提供 `silent=True` 并且文件未找到，则应用程序将失败，并给出以下错误：

```py
IOError: [Errno 2] Unable to load configuration file (No such file or directory): '/absolute/path/to/config/file'
```

信息

可能看起来使用 `instance_relative_config` 从实例文件夹加载配置是重复的工作，并且可以将其移动到配置方法之一。然而，这个过程的美妙之处在于实例文件夹的概念与配置完全独立，而 `instance_relative_config` 只是补充了配置对象。

# 视图和模型的组合

随着我们的应用程序变得更大，我们可能希望以模块化的方式对其进行结构化。在这个菜谱中，我们将通过重构我们的 *Hello* *World* 应用程序来实现这一点。

## 如何做到这一点...

首先，在应用程序中创建一个新的文件夹，并将所有文件移动到这个新文件夹中。然后，在这些文件夹中创建 `__init__.py`，这些文件夹将被用作模块。

之后，在顶级文件夹中创建一个名为 `run.py` 的新文件。正如其名所示，此文件将用于运行应用程序。

最后，创建单独的文件夹来充当模块。

参考以下文件结构以获得更好的理解：

```py
flask_app/
    run.py
    my_app/
        __init__.py
        hello/
            __init__.py
            models.py
            views.py
```

让我们看看前面的每个文件将如何看起来。

`flask_app/run.py` 文件将类似于以下代码行：

```py
from my_app import app
app.run(debug=True)
```

`flask_app/my_app/__init__.py` 文件将类似于以下代码行：

```py
from flask import Flask
app = Flask(__name__)
import my_app.hello.views
```

接下来，我们有一个空文件，只是为了使封装文件夹成为一个 Python 包，`flask_app/my_app/hello/__init__.py`：

```py
# No content.
# We need this file just to make this folder a python module.
```

模型文件，`flask_app/my_app/hello/models.py`，有一个非持久的键值存储，如下所示：

```py
MESSAGES = {
    'default': 'Hello to the World of Flask!',
}
```

最后，以下是一个视图文件，`flask_app/my_app/hello/views.py`。在这里，我们获取与请求的键对应的消息，也可以创建或更新一个消息：

```py
from my_app import app
from my_app.hello.models import MESSAGES
@app.route('/')
@app.route('/hello')
def hello_world():
    return MESSAGES['default']
@app.route('/show/<key>')
def get_message(key):
    return MESSAGES.get(key) or "%s not found!" % key
@app.route('/add/<key>/<message>')
def add_or_update_message(key, message):
    MESSAGES[key] = message
    return "%s Added/Updated" % key
```

## 它是如何工作的...

在这个菜谱中，我们有一个循环导入，在 `my_app/__init__.py` 和 `my_app/hello/views.py` 之间，在前者中，我们从后者导入 `views`，在后者中，我们从前者导入 `app`。尽管这使得两个模块相互依赖，但没有任何问题，因为我们不会在 `my_app/__init__.py` 中使用视图。请注意，最好在文件的底部导入视图，这样它们就不会在这个文件中使用。这确保了当你在视图中引用 `app` 对象时，不会导致空指针异常。

在这个配方中，我们使用一个非常简单的非持久内存中的键值存储来演示模型的布局结构。我们可以在 `views.py` 中直接编写 `MESSAGES` 哈希表的字典，但最佳实践是将模型和视图层分开。

因此，我们可以仅使用 `run.py` 来运行此应用程序，如下所示：

```py
$ python run.py
Serving Flask app "my_app" (lazy loading)
Environment: production
WARNING: Do not use the development server in a production environment.
Use a production WSGI server instead.
Debug mode: on
Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
Restarting with stat
Debugger is active!
* Debugger PIN: 111-111-111
```

小贴士

注意块中的前一个 `WARNING`。这个警告发生是因为我们没有指定应用程序环境，默认情况下假设为 `production`。要在 `development` 环境中运行应用程序，请使用以下内容修改 `run.py` 文件：

`from my_app import app`

`app.env="development"`

`app.run(debug=True)`

信息

重载指示应用程序正在调试模式下运行，并且每当代码中发生更改时，应用程序将重新加载。

如我们所见，我们已经在 `MESSAGES` 中定义了一个默认消息。我们可以通过打开 `http://127.0.0.1:5000/show/default` 来查看它。要添加一条新消息，我们可以输入 `http://127.0.0.1:5000/add/great/Flask%20is%20greatgreat!!`。这将更新 `MESSAGES` 键值存储，使其看起来像这样：

```py
MESSAGES = {
    'default': 'Hello to the World of Flask!',
    'great': 'Flask is great!!',
}
```

现在，如果我们在一个浏览器中打开 `http://127.0.0.1:5000/show/great`，我们将看到我们的消息，否则它将显示为一个未找到的消息。

## 参见

下一个配方 *使用蓝图创建模块化 Web 应用* 提供了一种组织 Flask 应用程序的更好方法，并且是循环导入的现成解决方案。

# 使用蓝图创建模块化 Web 应用

**蓝图** 是 Flask 中的一项功能，有助于使大型应用程序模块化。它通过提供一个中央位置来注册应用程序中的所有组件，从而简化了应用程序的分派。蓝图看起来像一个应用程序对象，但它不是一个应用程序。它也看起来像一个可插入的应用程序或更大应用程序的一个较小部分，但它不是。蓝图是一组可以在应用程序上注册的操作，它代表了如何构建或构建应用程序。另一个好处是，它允许我们在多个应用程序之间创建可重用的组件。

## 准备工作

在这个配方中，我们将以前一个配方 *视图和模型的组合* 中的应用程序作为参考，并修改它，使其使用蓝图工作。

## 如何做到这一点...

以下是一个使用 `Blueprint` 的简单 *Hello World* 应用程序的示例。它将像上一个配方中那样工作，但将更加模块化和可扩展。

首先，我们将从以下 `flask_app/my_app/__init__.py` 文件开始：

```py
from flask import Flask
from my_app.hello.views import hello
app = Flask(__name__)
app.register_blueprint(hello)
```

接下来，我们将在视图文件 `my_app/hello/views.py` 中添加一些代码，它应该看起来如下：

```py
from flask import Blueprint
from my_app.hello.models import MESSAGES
hello = Blueprint('hello', __name__)
@hello.route('/')
@hello.route('/hello')
def hello_world():
    return MESSAGES['default']
@hello.route('/show/<key>')
def get_message(key):
    return MESSAGES.get(key) or "%s not found!" % key
@hello.route('/add/<key>/<message>')
def add_or_update_message(key, message):
    MESSAGES[key] = message
    return "%s Added/Updated" % key
```

我们现在在`flask_app/my_app/hello/views.py`文件中定义了一个蓝图。我们不再需要这个文件中的应用程序对象，并且我们的完整路由定义在一个名为`hello`的蓝图中。我们使用`@hello.route`而不是`@app.route`。相同的蓝图被导入到`flask_app/my_app/__init__.py`中，并在应用程序对象上注册。

我们可以在应用程序中创建任意数量的蓝图，并完成我们通常会做的许多活动，例如提供不同的模板路径或不同的静态路径。我们甚至可以为我们的蓝图设置不同的 URL 前缀或子域名。

## 它是如何工作的...

此应用程序将以与上一个应用程序完全相同的方式运行。唯一的区别在于代码的组织方式。

# 使用 setuptools 使 Flask 应用程序可安装

我们现在有一个 Flask 应用程序，但如何像使用`setuptools`创建可安装的 Python 包一样安装它。

什么是 Python 包？

可以简单地将 Python 包视为一个程序，可以在虚拟环境或基于其安装范围全局使用 Python 的`import`语句导入。

## 如何做到...

使用`setuptools` Python 库可以轻松地安装 Flask 应用程序。为了实现这一点，在应用程序文件夹中创建一个名为`setup.py`的文件，并配置它以运行应用程序的设置脚本。这将处理任何依赖项、描述、加载测试包等。

以下是一个简单的`setup.py`脚本示例，用于之前配方中的*Hello World*应用程序：

```py
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from setuptools import setup
setup(
    name = 'my_app',
    version='1.0',
    license='GNU General Public License v3',
    author='Shalabh Aggarwal',
    author_email='contact@shalabhaggarwal.com',
    description='Hello world application for Flask',
    packages=['my_app'],
    platforms='any',
    install_requires=[
        'Flask',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public
          License v3',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries ::
          Python Modules'
],
)
```

## 它是如何工作的...

在前面的脚本中，大部分配置都是不言自明的。分类器在应用程序在**PyPI**上提供时使用。这些将帮助其他用户使用相关分类器搜索应用程序。

现在，我们可以使用`install`关键字运行此文件，如下所示：

```py
$ python setup.py install
```

前面的命令将安装应用程序以及`install_requires`中提到的所有依赖项——也就是说，Flask 及其所有依赖项。现在，该应用程序可以在 Python 环境中像任何 Python 包一样使用。

要验证包安装成功，在 Python 环境中导入它：

```py
$ python
Python 3.8.13 (default, May  8 2022, 17:52:27)
>>> import my_app
>>>
```

## 相关内容

有效 trove 分类器的列表可以在[`pypi.python.org/pypi?%3Aaction=list_classifiers`](https://pypi.python.org/pypi?%3Aaction=list_classifiers)找到。
