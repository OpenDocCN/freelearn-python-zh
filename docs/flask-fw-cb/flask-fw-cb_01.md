# 第一章：Flask 配置

第一章将会帮助你去理解不同的 Flask 配置方法来满足每个项目各式各样的需求。

在这一章，将会涉及到以下方面：

*   用 virtualenv 搭建环境
*   处理基本配置
*   基于类的配置
*   组织静态文件
*   用实例文件夹（instance floders）进行部署
*   视图和模型的融合（composition）
*   用蓝本（blueprint）创建一个模块化的 web 应用
*   使用 setuptools 使 Flask 应用可安装

## 介绍

> “Flask is a microframework for Python based on Werkzeug, Jinja2 and good intentions.”

何为微小？是不是意味着 Flask 在功能性上有所欠缺或者必须只能用一个文件来完成 web 应用？并不是这样！它说明的事实是 Flask 目的在于保持核心框架的微小但是高度可扩展。这使得编写应用或者扩展非常的容易和灵活，同时也给了开发者为他们的应用选择他们想要配置的余地，没有在数据库，模板引擎和其他方面做出强制性的限制。通过这一章你将会学到一些建立和配置 Flask 的方法。
开始 Flask 几乎不需要 2 分钟。建立一个简单的 Hello World 应用就和烤派一样简单：

```py
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello to the World of Flask!'

if __name__ == '__main__':
    app.run() 
```

现在需要安装 Flask，这可以通过 pip 实现：

```py
$ pip install Flask 
```

之前的一小段就是完整的基于 Flask 的 web 应用。导入的 Flask 类创建的实例是一个 web 服务器网关接口(Web Server Gateway Interface WSGI)应用。所以代码里的 app 成为了我们的 WSGI 应用。因为这个一个独立的模块，我们用`__name__` 和`'__main__'` 字符串做比较。如果我们将这些保存为名字是 app.py 的文件，这个应用可以使用下面的命令来运行：

```py
$ python app.py 
 * Running on http://127.0.0.1:5000/ 
```

现在如果在浏览器中输入 http:/127.0.0.1:5000/，将会看见应用在运行。

###### 提示

千万不要将你的文件保存为 flask.py，如果你这样做了，将会和导入的 Flask 冲突。

## 用 virtualenv 搭建环境

Flask 能够通过使用 pip 或者 easy_install 进行安装，但我们应该使用 virtualenv 来创建应用环境。通过为应用创建一个单独的环境可以防止全局 Python 被我们自定义的安装所影响。单独的环境是有用的，因为你可以多个应用程序有同一个库的多个版本，或者一些包可能有相同库的不同版本作为它们的依赖。virtualenv 在单独的环境里管理这些，不会让任何错误版本的库影响到任何其他应用。

#### 怎么做

首先用 pip 安装 virtualenv，然后创建一个名字为 my_flask_env 的环境。这同时会创建一个相同名字的文件夹：

```py
$ pip install virtualenv
$ virtualenv my_flask_env 
```

现在运行下面命令：

```py
$ cd my_flask_env
$ source bin/activate
$ pip install flask 
```

这将激活环境并且在其中安装 Flask。现在可以在这个环境中对我们的应用做任何事情，而不会影响到任何其他 Python 环境。

#### 原理

直到现在，我们已经使用 pip install flask 多次了。顾名思义，这个命令的意思是安装 Flask，就像安装其他 Python 包一样。如果仔细观察一下通过 pip 安装 Flask 的过程，我们将会看到一些包被安装了。下面是 Flask 包安装过程的一些摘要：

```py
$ pip install -U flask
Downloading/unpacking flask
......
......
Many more lines......
......
Successfully installed flask Werkzeug Jinja2 itsdangerous markupsafe
Cleaning up... 
```

###### 提示

在前面的命令中，-U 指的是安装与升级。这将会用最新的版本覆盖已经存在的安装。
如果观察的够仔细，总共有五个包被安装了，分别是 flask，Werkzeug，Jinja2，itsdangerous，markupsafe。Flask 依赖这些包，如果这些包缺失了，Flask 将不会工作。

#### 其他

为了更美好的生活，我们可以使用 virtualenvwrapper。顾名思义，这是对 virtualenv 的封装，使得处理多个 virtualenv 更容易。

###### 提示

记住应该通过全局的方式安装 virtualenvwrapper。所以需要停用还处在激活状态的 virtualenv，可以用下面的命令：

$ deactivate

同时，你可能因为权限问题不被允许在全局环境安装 virtualenvwrapper。这种情况下需要切换到超级用户或者使用 sudo。

可以用下面的命令来安装 virtualenvwrapper：

```py
$ pip install virtualenvwrapper
$ export WORKON_HOME=~/workspace
$ source /usr/local/bin/virtualenvwrapper.sh 
```

在上面的代码里，我们安装了 virtualenvwrapper，创建了一个名字为 WORKON_HOME 的环境变量，同时给它赋值了一个路径，当用 virtualenvwrapper 创建虚拟环境时，虚拟环境将会安装在这个路径下面。安装 Flask 可以使用下面的命令：

```py
$ mkvirtualenv flask
$ pip install flask 
```

停用虚拟环境，只需运行下面的命令：

```py
$ deactivate 
```

激活已经存在的 virtualenv，可以运行下面的命令：

```py
$ workon flask 
```

#### 其他

参考和安装链接如下：

*   [`pypi.python.org/pypi/virtualenv`](https://pypi.python.org/pypi/virtualenv)
*   [`pypi.python.org/pypi/virtualenvwrapper`](https://pypi.python.org/pypi/virtualenvwrapper)
*   [`pypi.python.org/pypi/Flask`](https://pypi.python.org/pypi/Flask)
*   [`pypi.python.org/pypi/Werkzeug`](https://pypi.python.org/pypi/Werkzeug)
*   [`pypi.python.org/pypi/Jinja2`](https://pypi.python.org/pypi/Jinja2)
*   [`pypi.python.org/pypi/itsdangerous`](https://pypi.python.org/pypi/itsdangerous)
*   [`pypi.python.org/pypi/MarkupSafe`](https://pypi.python.org/pypi/MarkupSafe)

## 处理基本配置

首先想到的应该是根据每个需求去配置一个 Flask 应用。这一小节，我们将会去理解 Flask 不同的配置方法。

#### 准备

在 Flask 中，配置能够通过 Flask 的一个名为 config 的属性来完成。config 是字典数据类型的一个子集，我们能够像字典一样修改它。

#### 怎么做

举个例子，需要将我们的应用运行在调试模式下，需要写出下面这样的代码：

```py
app = Flask(__name__)
app.config['DEBUG'] = True 
```

###### 提示

debug 布尔变量可以从 Flask 对象而不是 config 角度来设置：

```py
app.debug = True 
```

同样也可以使用下面这行代码：

```py
app.run(debug=True) 
```

使用调试模将会使服务器在有代码改变的时候自动重载，同时它也在出错的时候提供了非常有用的调试信息。

Flask 还提供了许多配置变量，我们将会在相关的章节接触他们。
当应用越来越大的时候，就产生了在一个文件中管理这些配置的需要。在大部分案例中特定于机器基础的配置都不是版本控制系统的一部分。因为这些，Flask 提供了多种方式去获取配置。常用的几种是：

*   通 pyhton 配置文件(*.cfg)，通过使用:`app.config.from_pyfile('myconfig.cfg')`获取配置

*   通过一个对象，通过使用:`app.config.from_object('myapplication.default_settings')`获取配置或者也可以使用:`app.config.from_object(__name__)` #从当前文件加载配置

*   通过环境变量，通过使用:`app.config.from_envvar('PATH_TO_CONFIG_FILE')`获取配置

#### 原理

Flask 足够智能去找到那些用大写字母写的配置变量。同时这也允许我们在配置文件/对象里定义任何局部变量，剩下的就交给 Flask。

###### 提示

最好的配置方式是在 app.py 里定义一些默认配置，或者通过应用本身的任何对象，然后从配置文件里加载同样的配置去覆盖它们。所以代码看起来像这样：

```py
app = Flask(__name__)
DEBUG = True
TESTING = True
app.config.from_object(__name__) #译者注：这句话作用是导入当前文件里定义的配置，比如 DEBUG 和 TESTING
app.config.from_pyfile('/path/to/config/file') 
```

## 基于类的配置

配置生产，测试等不同模式的方式是通过使用类继承。当项目越来越大，可以有不同的部署模式，比如开发环境，staging，生产等等，每种模式都有一些不同的配置，也会存在一些相同的配置。

#### 怎么做

我们可以有一个默认配置基类，其他类可以继承基类也可以重载或者增加特定发布环境的配置变量。
下面是一个使用默认配置基类的例子：

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

###### 提示

SECRET KEY 应该被存储在单独的文件里，因为从安全角度考虑，它不应该是版本控制系统的一部分。应该被保存在机器自身的本地文件系统，或者个人电脑或者服务器。

#### 原理

现在，通过 from_object()可以使用任意一个刚才写的类来加载应用配置。前提是我们将刚才基于类的配置保存在了名字为 configuration.py 的文件里：

```py
app.config.from_object('configuration.DevelopmentConfig') 
```

总体上，这使得管理不同环境下的配置更加灵活和容易。

###### 提示

书源码下载地址：
[`pan.baidu.com/s/1o7GyZUi`](https://pan.baidu.com/s/1o7GyZUi) 密码：x9rw
[`download.csdn.net/download/liusple/10186764`](http://download.csdn.net/download/liusple/10186764)

## 组织静态文件

将 JavaScript，stylesheets，图像等静态文件高效的组织起来是所有 web 框架需要考虑的事情。

#### 怎么做

Flask 推荐一个特定的方式组织静态文件：

```py
my_app/
    - app.py
    - config.py
    - __init__.py
    - static/
        - css/
        - js/
        - images/
            - logo.png 
```

当需要在模板中渲染他们的时候（比如 logo.png），我们可以通过下面方式使用静态文件：

```py
<img src='/statimg/logo.png'> 
```

#### 原理

如果在应用根目录存在一个和 app.py 同一层目录的名字为 static 的文件夹，Flask 会自动的去读这个文件夹下的内容，而不需要任何其他配置。

#### 其它

与此同时，我们可以在 app.py 定义应用的时候为应用对象提供一个名为 static_folder 的参数：

```py
app=Flask(__name__, static_folder='/path/to/static/folder') 
```

在怎么做一节里的 img src 中，static 指的是这个应用 static_url_path 的值。可以通过下面方法修改：

```py
app = Flask(
    __name__, static_url_path='/differentstatic',
    static_folder='/path/to/static/folder'
) 
```

现在，渲染静态文件，可以使用：

```py
<img src='/differentstatic/logo.png'> 
```

###### 提示

通常一个好的方式是使用 url_for 去为静态文件创建 URLS，而不是明确的定义他们：

```py
<img src='{{ url_for('static', filename="logo.png") }}'> 
```

我们将会在下面章节看到更多这样的用法。

## 使用实例文件夹（instance folders）进行特定部署

Flask 也提供了高效管理特定部署的其他方式。实例文件夹允许我们从版本控制系统中费力出特定的部署文件。我们知道不同部署环境比如开发，生产，他们的配置文件是分开的。但还有很多其他文件比如数据库文件，会话文件，缓存文件，其他运行时文件。所以我们可以用实例文件夹像一个 holder bin 一样来存放这些文件。

#### 怎么做

通常，如果在我们的应用里有一个名字问 instance 的文件夹，应用可以自动的识别出实例文件夹：

```py
my_app/
    - app.py
    - instance/
        - config.cfg 
```

在应用对象里，我们可以用 instance_path 明确的定义实例文件夹的绝对路径：

```py
app = Flask(
    __name__, instance_path='/absolute/path/to/instance/folder'
) 
```

为了从实例文件夹加载配置文件，可以在应用对象里使用 instance_relative_config 参数：

```py
app = Flask(__name__, instance_relative_config=True) 
```

这告诉我们的应用从实例文件夹加载配置。下面的实例演示了它如何工作：

```py
app = Flask(
    __name__, instance_path='path/to/instance/folder',
    instance_relative_config=True
)
app.config.from_pyfile('config.cfg', silent=True) 
```

#### 原理

前面的代码，首先，实例文件夹从给定的路径被加载了，然后，配置从实例文件夹里一个名为 config.cfg 的文件中加载。silent=True 是可选的，用来在实例文件夹里没发现 config.cfg 时不报错误。如果 silent=True 没有给出，并且 config.cfg 没有找到，应用将失败，给出下面的错误：

```py
IOError: [Errno 2] Unable to load configuration file (No such file or directory): '/absolute/path/to/config/file' 
```

###### 提示

用 instance_relative_config 从实例文件夹加载配置好像是一个对于的工作，可以使用一个配置方法代替。但是这个过程的美妙之处在于，实例文件夹的概念是完全独立于配置的。

译者注:可以参考[这篇博客](https://www.cnblogs.com/m0m0/p/5624315.html)理解实例文件夹。

## 视图和模型的结合(composition)

随着应用的变大，我们需要用模块化的方式组织我们的应用。下面我们将重构 Hello World 应用。

#### 怎么做

1.  首先在我们的应用里创建一个文件夹，移动所有的文件到这个新的文件夹里。
2.  然后在新建的文件夹里创建一个名为`__init__.py`的文件，这将使得文件夹变成一个模块。
3.  之后，在顶层目录创建一个新的名为 run.py 的文件。从名字可以看出，这个文件将会用来运行这个应用。
4.  最后，创建单独的文件夹作为模块。

通过下面的文件结构可以更好的理解：

```py
flask_app/
    - run.py
    - my_app/
        – __init__.py
        - hello/
            - __init__.py
            - models.py
            - views.py 
```

首先，`flask_app/run.py`文件里的内容看起来像这样：

```py
from my_app import app
app.run(debug=True) 
```

然后，`flask_app/my_app/__init__.py`文件里的内容看起来像这样：

```py
from flask import Flask
app = Flask(__name__)

import my_app.hello.views 
```

然后，存在一个空文件使得文件夹可以作为一个 Python 包，`flask_app/my_app/hello/__init__.py`:

```py
# No content.
# We need this file just to make this folder a python module. 
```

模型文件，`flask_app/my_app/hello/models.py`，有一个非持久性的键值存储：

```py
MESSAGES = {
    'default': 'Hello to the World of Flask!',
} 
```

最后是视图文件，`flask_app/my_app/hello/views.py`。这里，我们获取与请求键相对于的消息，并提供相应的服务创建或更新一条消息：

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

###### 提示

记住上面的实例代码不能用在生产环境下。仅仅为了让 Flask 初学者更容易理解进行的示范。

#### 原理

可以看到在`my_app/__init__.py`和`my_app/hello/views.py`之间有一个循环导入，前者从后者导入 views，后者从前者导入 app。所以，这实际上将会使得这两个模块相互依赖，但是在这里是没问题的，因为我们不会在`my_app/__init__.py`里使用 views。我们在文件的底部导入 views，所以它们不会被使用到。

在这个实例中，我们使用了一个非常简单的基于内存的非持久化键值对。当然我们能够在 views.py 文件里重写 MESSAGES，但是最好的方式是保持模型层和视图层的相互独立。

现在，可以用 run.py 就可以运行 app 了：

```py
$ python run.py
* Running on http://127.0.0.1:5000/
* Restarting with reloader 
```

###### 提示

上面加载信息表示应用正运行在调试模式下，这个应用将会在代码更改的时候重新加载。

现在可以看到我们在 MESSAGES 里面定义的默认消息。可以通过打开`http://127.0.0.1:5000/show/default`来看到这些消息。通过`http://127.0.0.1:5000/add/great/Flask%20is%20great`增加一个新的消息。这将会更新 MESSAGES 键值对，看起来像这样：

```py
MESSAGES = {
    'default': 'Hello to the World of Flask!',
    'great': 'Flask is great!!',
} 
```

现在可以在浏览器打开`http://127.0.0.1:5000/show/great`，我们将会看到我们的消息，否则会看到一个 not-found 消息。

#### 其他

下一章节，使用蓝图创建一个模块化的 web 应用，提供了一个更好的方式来组织你的 Flask 应用，也是一个对循环导入的现成解决方案。

## 使用蓝图（blueprint）创建一个模块化的 web 应用

蓝图是 Flask 的一个概念用来帮助大型应用真正的模块化。通过提供一个集中的位置来注册应用中的所有组件，使得应用调度变得简单。蓝本看起来像是一个应用对象，但却不是。它看上去更像是一个可插拔（pluggable）的应用或者是一个更大应用的一小部分。一个蓝本实际上是一组可以注册到应用上的操作集合，并且表示了如何构建一个应用。

#### 准备

我们将会利用上一小节的应用做为例子，通过使用蓝图修改它，使它正常工作。

#### 怎么做

下面是一个使用蓝图的 Hello World 例子。它的效果和前一章节相似，但是更加模块化和可扩展。
首先，从`flask_app/my_app/__init__.py`文件开始：

```py
from flask import Flask
from my_app.hello.views import hello

app = Flask(__name__)
app.register_blueprint(hello) 
```

接下来，视图文件，`my_app/hello/views.py`，将会看起来像下面这些代码：

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

我们在`flask_app/my_app/hello/views.py`文件里定义了一个蓝本。我们不需要在这里使用任何应用对象，完整的路由是通过使用名为 hello 的蓝图定义的。我们用@hello.route 替代了@app.route。这个蓝本在`flask_app/my_app/__init__.py`被导入了，并且注册在了应用对象上。

我们可以在应用里创建任意数量的蓝图和做大部分的活动(activities)，比如提供不同的模板路径和静态文件夹路径。我们甚至为蓝图创建不同的 URL 前缀或者子域。

#### 原理

这个应用的工作方式和上一个应用完全一样。唯一的差别是代码组织方式的不同。

#### 其他

*   理解上一小节，视图和模型的组合，对理解这一章节有所帮助。

## 使用 setuptools 使 Flask 应用可安装

现在我们已经有了一个 Flask 应用了，但是怎么去像安装其他 Python 包一样来安装它呢？

#### 怎么做

使用 Python 的 setuptools 库可以很容易使 Flask 应用可安装。我们需要在应用文件夹里创建一个名为 setup.py 的文件，并且配置它去为应用运行一个安装脚本。它将处理任何依赖，描述，加载测试包，等等。
下面是 Hello World 应用安装脚本 setup.py 的一个简单实例：

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
        'flask',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
) 
```

#### 原理

前面的脚本里大部分的配置都是不言而喻的。当我们的应用可从 PyPI 可获取时，分类器(classifiers)是有用的。这将会帮助其他用户通过使用分类器(classiflers)来搜索我们的应用。
现在我们可以用 install 关键字来运行这个文件：

```py
$ python setup.py install 
```

这将会安装我们的应用，并且也会安装在 install_requires 里提到的依赖，所以 Flask 和所有 Flask 的依赖都会被安装。现在这个应用可以在 Python 环境里像使用其他 Python 包一样来使用了。

