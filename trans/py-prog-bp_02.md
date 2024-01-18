# 使用Spotify创建远程控制应用程序

Spotify是一家总部位于瑞典斯德哥尔摩的音乐流媒体服务。第一个版本于2008年发布，如今它不仅提供音乐，还提供视频和播客。Spotify从瑞典的初创公司迅速发展成为世界上最大的音乐服务，其应用程序在视频游戏机和手机上运行，并与许多社交网络集成。

该公司确实改变了我们消费音乐的方式，也使得不仅是知名艺术家，而且小型独立艺术家也能与世界分享他们的音乐。

幸运的是，Spotify也是开发人员的绝佳平台，并提供了一个非常好的和有文档的REST API，可以通过艺术家、专辑、歌曲名称进行搜索，还可以创建和分享播放列表。

在本书的第二个应用程序中，我们将开发一个终端应用程序，其中我们可以：

+   搜索艺术家

+   搜索专辑

+   搜索曲目

+   播放音乐

除了所有这些功能之外，我们将实现一些函数，以便通过终端控制Spotify应用程序。

首先，我们将经历在Spotify上创建新应用程序的过程；然后，将是开发一个小框架的时间，该框架将包装Spotify的REST API的某些部分。我们还将致力于实现Spotify支持的不同类型的身份验证，以便消耗其REST API。

当所有这些核心功能都就位后，我们将使用Python附带的`curses`软件包来开发终端用户界面。

在本章中，您将学习：

+   如何创建`Spotify`应用程序

+   如何使用`OAuth`

+   面向对象的编程概念

+   使用流行的`Requests`软件包来消耗REST API

+   使用curses设计终端用户界面的方法

我不知道你们，但我真的很想写代码并听一些好听的音乐，所以让我们开始吧！

# 设置环境

让我们继续配置我们的开发环境。我们需要做的第一件事是创建一个新的虚拟环境，这样我们就可以工作并安装我们需要的软件包，而不会干扰全局Python安装。

我们的应用程序将被称为`musicterminal`，因此我们可以创建一个同名的虚拟环境。

要创建一个新的虚拟环境，请运行以下命令：

```py
$ python3 -m venv musicterminal
```

确保您使用的是Python 3.6或更高版本，否则本书中的应用程序可能无法正常工作。

要激活虚拟环境，可以运行以下命令：

```py
$ . musicterminal/bin/activate
```

太好了！现在我们已经设置好了虚拟环境，我们可以创建项目的目录结构。它应该具有以下结构：

```py
musicterminal
├── client
├── pytify
│   ├── auth
│   └── core
└── templates
```

与第一章中的应用程序一样，我们创建一个项目目录（这里称为`musicterminal`）和一个名为`pytify`的子目录，其中将包含包装Spotify的REST API的框架。

在框架目录中，我们将`auth`拆分为两个模块，这两个模块将包含Spotify支持的两种身份验证流程的实现——授权代码和客户端凭据。最后，`core`模块将包含从REST API获取数据的所有方法。

客户端目录将包含与我们将构建的客户端应用程序相关的所有脚本。

最后，`templates`目录将包含一些HTML文件，这些文件将在我们构建一个小的Flask应用程序来执行Spotify身份验证时使用。

现在，让我们在`musicterminal`目录中创建一个`requirements.txt`文件，内容如下：

```py
requests==2.18.4
PyYAML==3.12
```

要安装依赖项，只需运行以下命令：

```py
$ pip install -r requirements.txt
```

![](assets/c51878ae-cdd2-4c1c-aba5-a1fb2df3386e.png)

如您在输出中所见，其他软件包已安装在我们的虚拟环境中。这是因为我们项目所需的软件包也需要其他软件包，因此它们也将被安装。

Requests是由Kenneth Reitz创建的[https://www.kennethreitz.org/](https://www.kennethreitz.org/)，它是Python生态系统中使用最广泛且备受喜爱的软件包之一。它被微软、谷歌、Mozilla、Spotify、Twitter和索尼等大公司使用，它是Pythonic且非常直观易用的。

查看Kenneth的其他项目，尤其是`pipenv`项目，这是一个很棒的Python打包工具。

我们将使用的另一个模块是curses。curses模块只是curses C函数的包装器，相对于在C中编程，它相对简单。如果您之前使用过curses C库，那么Python中的curses模块应该是熟悉且易于学习的。

需要注意的一点是，Python在Linux和Mac上包含curses模块；但是，在Windows上，默认情况下不包含它。如果您使用Windows，curses文档在[https://docs.python.org/3/howto/curses.html](https://docs.python.org/3/howto/curses.html)上推荐由Fredrik Lundh开发的UniCurses包。

在我们开始编码之前，还有一件事。在尝试导入curses时，您可能会遇到问题；最常见的原因是您的系统中未安装`libncurses`。在安装Python之前，请确保您的系统上已安装`libncurses`和`libncurses-dev`。

如果您使用Linux，您很可能会在我们首选发行版的软件包存储库中找到`libncurses`。在Debian/Ubuntu中，您可以使用以下命令安装它：

```py
$ sudo apt-get install libncurses5 libncurses5-dev
```

太好了！现在，我们已经准备好开始实施我们的应用程序了。

# 创建Spotify应用程序

我们需要做的第一件事是创建一个Spotify应用程序；之后，我们将获取访问密钥，以便我们可以进行身份验证并使用REST API。

前往[https://beta.developer.spotify.com/dashboard/](https://beta.developer.spotify.com/dashboard/)，在页面下方您可以找到登录按钮，如果您没有帐户，可以创建一个新帐户。

![](assets/1557b618-e930-456a-bf79-f539bf015194.png)在撰写本文时，Spotify开始更改其开发者网站，并且目前处于测试阶段，因此登录地址和一些截图可能会有所不同。

如果您没有Spotify帐户，您首先需要创建一个。如果您注册免费帐户，应该能够创建应用程序，但我建议您注册高级帐户，因为它是一个拥有丰富音乐目录的优秀服务。

当您登录Spotify开发者网站时，您将看到类似以下页面：

![](assets/d051e689-8f73-40a9-b6bf-7e005dc6fd66.png)

目前，我们还没有创建任何应用程序（除非您已经创建了一个），所以继续点击“CREATE AN APP”按钮。将显示一个对话框屏幕来创建应用程序：

![](assets/50e3e47c-e7b5-484f-8617-203af23e550a.png)

在这里，我们有三个必填字段：应用程序名称、描述，以及一些复选框，您需要告诉Spotify您正在构建什么。名称应该是`pytify`，在描述中，您可以随意填写，但让我们添加类似“用于从终端控制Spotify客户端的应用程序”的内容。我们正在构建的应用程序类型将是网站。

完成后，点击对话框屏幕底部的“NEXT”按钮。

应用程序创建过程的第二步是告知Spotify您是否正在创建商业集成。对于本书的目的，我们将选择**NO**；但是，如果您要创建一个将实现货币化的应用程序，您应该选择**YES**。

在下一步中，将显示以下对话框：

![](assets/0f932dc3-70ea-4696-91e7-7ddc19dd3cef.png)

如果您同意所有条件，只需选择所有复选框，然后点击“SUBMIT”按钮。

如果应用程序已成功创建，您将被重定向到应用程序的页面，如下所示：

![](assets/ed9c9c57-1548-4c53-8414-9a973dc370df.png)

单击“显示客户端密钥”链接，并复制客户端ID和客户端密钥的值。我们将需要这些密钥来使用Spotify的REST API。

# 应用程序的配置

为了使应用程序更灵活且易于配置，我们将创建一个配置文件。这样，我们就不需要硬编码URL和访问密钥；而且，如果需要更改这些设置，也不需要更改源代码。

我们将创建一个YAML格式的配置文件，用于存储我们的应用程序用于认证、向Spotify RESP API端点发出请求等的信息。

# 创建配置文件

让我们继续在`musicterminal`目录中创建一个名为`config.yaml`的文件，内容如下：

```py
client_id: '<your client ID>'
client_secret: '<your client secret>'
access_token_url: 'https://accounts.spotify.com/api/token'
auth_url: 'http://accounts.spotify.com/authorize'
api_version: 'v1'
api_url: 'https://api.spotify.com'
auth_method: 'AUTHORIZATION_CODE'
```

`client_id`和`client_secret`是我们创建Spotify应用程序时为我们创建的密钥。这些密钥将用于获取访问令牌，每次我们需要向Spotify的REST API发送新请求时都必须获取访问令牌。只需用您自己的密钥替换`<your client ID>`和`<your client secret>`。

请记住，这些密钥必须保存在安全的地方。不要与任何人分享密钥，如果您在GitHub等网站上有项目，请确保不要提交带有您的秘密密钥的配置文件。我通常会将配置文件添加到我的`.gitignore`文件中，这样它就不会被源代码控制；否则，您可以像我一样提交文件，使用占位符而不是实际密钥。这样，就很容易记住您需要在哪里添加密钥。

在`client_id`和`client_secret`键之后，我们有`access_token_url`。这是我们必须执行请求的API端点的URL，以便获取访问令牌。

`auth_url`是Spotify的账户服务的端点；当我们需要获取或刷新授权令牌时，我们将使用它。

`api_version`，顾名思义，指定了Spotify的REST API版本。在执行请求时，这将附加到URL上。

最后，我们有`api_url`，这是Spotify的REST API端点的基本URL。

# 实现配置文件读取器

在实现读取器之前，我们将添加一个枚举，表示Spotify提供给我们的两种认证流程。让我们继续在`musicterminal/pytify/auth`目录中创建一个名为`auth_method.py`的文件，内容如下：

```py
from enum import Enum, auto

class AuthMethod(Enum):
    CLIENT_CREDENTIALS = auto()
    AUTHORIZATION_CODE = auto()
```

这将定义一个枚举，具有`CLIENT_CREDENTIALS`和`AUTHORIZATION_CODE`属性。现在，我们可以在配置文件中使用这些值。我们还需要做的另一件事是在`musicterminal/pytify/auth`目录中创建一个名为`__init__.py`的文件，并导入我们刚刚创建的枚举：

```py
from .auth_method import AuthMethod
```

现在，我们可以继续创建将为我们读取配置的函数。在`musicterminal/pytify/core`目录中创建一个名为`config.py`的文件，然后让我们开始添加一些导入语句：

```py
import os
import yaml
from collections import namedtuple

from pytify.auth import AuthMethod
```

首先，我们导入`os`模块，这样我们就可以访问一些函数，这些函数将帮助我们构建YAML配置文件所在的路径。我们还导入`yaml`包来读取配置文件，最后，我们从collections模块导入`namedtuple`。稍后我们将更详细地讨论`namedtuple`的作用。

我们最后导入的是我们刚刚在`pytify.auth`模块中创建的`AuthMethod`枚举。

现在，我们需要一个表示配置文件的模型，因此我们创建一个名为`Config`的命名元组，如下所示：

```py
Config = namedtuple('Config', ['client_id',
                               'client_secret',
                               'access_token_url',
                               'auth_url',
                               'api_version',
                               'api_url',
                               'base_url',
                               'auth_method', ])
```

`namedtuple`不是Python中的新功能，自2.6版本以来一直存在。`namedtuple`是类似元组的对象，具有名称，并且可以通过属性查找访问字段。可以以两种不同的方式创建`namedtuple`；让我们开始Python REPL并尝试一下：

```py
>>> from collections import namedtuple
>>> User = namedtuple('User', ['firstname', 'lastname', 'email'])
>>> u = User('Daniel','Furtado', 'myemail@test.com')
User(firstname='Daniel', lastname='Furtado', email='myemail@test.com')
>>>
```

此结构有两个参数；第一个参数是`namedtuple`的名称，第二个是表示`namedtuple`中每个字段的`str`元素数组。还可以通过传递一个由空格分隔的每个字段名的字符串来指定`namedtuple`的字段，例如：

```py
>>> from collections import namedtuple
>>> User = namedtuple('User', 'firstname lastname email')
>>> u = User('Daniel', 'Furtado', 'myemail@test.com')
>>> print(u)
User(firstname='Daniel', lastname='Furtado', email='myemail@test.com')
```

`namedtuple`构造函数还有两个关键字参数：

`Verbose`，当设置为`True`时，在终端上显示定义`namedtuple`的类。在幕后，`namedtuple`是类，`verbose`关键字参数让我们一睹`namedtuple`类的构造方式。让我们在REPL上实践一下：

```py
>>> from collections import namedtuple
>>> User = namedtuple('User', 'firstname lastname email', verbose=True)
from builtins import property as _property, tuple as _tuple
from operator import itemgetter as _itemgetter
from collections import OrderedDict

class User(tuple):
    'User(firstname, lastname, email)'

    __slots__ = ()

    _fields = ('firstname', 'lastname', 'email')

    def __new__(_cls, firstname, lastname, email):
        'Create new instance of User(firstname, lastname, email)'
        return _tuple.__new__(_cls, (firstname, lastname, email))

    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):
        'Make a new User object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != 3:
            raise TypeError('Expected 3 arguments, got %d' % 
            len(result))
        return result

    def _replace(_self, **kwds):
        'Return a new User object replacing specified fields with  
         new values'
        result = _self._make(map(kwds.pop, ('firstname', 'lastname',  
                             'email'), _self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' %  
                              list(kwds))
        return result

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + '(firstname=%r,  
                                           lastname=%r, email=%r)' 
        % self

    def _asdict(self):
        'Return a new OrderedDict which maps field names to their  
          values.'
        return OrderedDict(zip(self._fields, self))

    def __getnewargs__(self):
        'Return self as a plain tuple. Used by copy and pickle.'
        return tuple(self)

    firstname = _property(_itemgetter(0), doc='Alias for field  
                          number 0')

    lastname = _property(_itemgetter(1), doc='Alias for field number  
                         1')

    email = _property(_itemgetter(2), doc='Alias for field number  
                      2')
```

另一个关键字参数是`rename`，它将重命名`namedtuple`中具有不正确命名的每个属性，例如：

```py
>>> from collections import namedtuple
>>> User = namedtuple('User', 'firstname lastname email 23445', rename=True)
>>> User._fields
('firstname', 'lastname', 'email', '_3')
```

如您所见，字段`23445`已自动重命名为`_3`，这是字段位置。

要访问`namedtuple`字段，可以使用与访问类中的属性相同的语法，使用`namedtuple`——`User`，如前面的示例所示。如果我们想要访问`lastname`属性，只需写`u.lastname`。

现在我们有了代表我们配置文件的`namedtuple`，是时候添加执行加载YAML文件并返回`namedtuple`——`Config`的工作的函数了。在同一个文件中，让我们实现`read_config`函数如下：

```py
def read_config():
    current_dir = os.path.abspath(os.curdir)
    file_path = os.path.join(current_dir, 'config.yaml')

    try:
        with open(file_path, mode='r', encoding='UTF-8') as file:
            config = yaml.load(file)

            config['base_url'] = 
 f'{config["api_url"]}/{config["api_version"]}'    auth_method = config['auth_method']
            config['auth_method'] = 
            AuthMethod.__members__.get(auth_method)

            return Config(**config)

    except IOError as e:
        print(""" Error: couldn''t file the configuration file 
        `config.yaml`
 'on your current directory.   Default format is:',   client_id: 'your_client_id' client_secret: 'you_client_secret' access_token_url: 'https://accounts.spotify.com/api/token' auth_url: 'http://accounts.spotify.com/authorize' api_version: 'v1' api_url: 'http//api.spotify.com' auth_method: 'authentication method'   * auth_method can be CLIENT_CREDENTIALS or  
          AUTHORIZATION_CODE""")
        raise   
```

`read_config`函数首先使用`os.path.abspath`函数获取当前目录的绝对路径，并将其赋给`current_dir`变量。然后，我们将存储在`current_dir`变量上的路径与文件名结合起来，即YAML配置文件。

在`try`语句中，我们尝试以只读方式打开文件，并将编码设置为UTF-8。如果失败，将向用户打印帮助消息，说明无法打开文件，并显示描述YAML配置文件结构的帮助。

如果配置文件可以成功读取，我们调用`yaml`模块中的load函数来加载和解析文件，并将结果赋给`config`变量。我们还在配置中包含了一个额外的项目`base_url`，它只是一个辅助值，包含了`api_url`和`api_version`的连接值。

`base_url`的值将如下所示：[https://api.spotify.com/v1.](https://api.spotify.com/v1)

最后，我们创建了一个`Config`的实例。请注意我们如何在构造函数中展开值；这是可能的，因为`namedtuple`——`Config`具有与`yaml.load()`返回的对象相同的字段。这与执行以下操作完全相同：

```py
return Config(
    client_id=config['client_id'],
    client_secret=config['client_secret'],
    access_token_url=config['access_token_url'],
    auth_url=config['auth_url'],
    api_version=config['api_version'],
    api_url=config['api_url'],
    base_url=config['base_url'],
    auth_method=config['auth_method'])
```

最后一步是在`pytify/core`目录中创建一个`__init__.py`文件，并导入我们刚刚创建的`read_config`函数：

```py
from .config import read_config
```

# 使用Spotify的Web API进行身份验证

现在我们已经有了加载配置文件的代码，我们将开始编写框架的认证部分。Spotify目前支持三种认证方式：授权码、客户端凭据和隐式授权。在本章中，我们将实现授权码和客户端凭据，首先实现客户端凭据流程，这是最容易开始的。

客户端凭据流程与授权码流程相比有一些缺点，因为该流程不包括授权，也无法访问用户的私人数据以及控制播放。我们现在将实现并使用此流程，但在开始实现终端播放器时，我们将改为授权码。

首先，我们将在`musicterminal/pytify/auth`目录中创建一个名为`authorization.py`的文件，内容如下：

```py
from collections import namedtuple

Authorization = namedtuple('Authorization', [
    'access_token',
    'token_type',
    'expires_in',
    'scope',
    'refresh_token',
])
```

这将是认证模型，它将包含我们在请求访问令牌后获得的数据。在下面的列表中，您可以看到每个属性的描述：

+   `access_token`：必须与每个对Web API的请求一起发送的令牌

+   `token_type`：令牌的类型，通常为`Bearer`

+   `expires_in`：`access_token`的过期时间，为3600秒（1小时）

+   `scope`：范围基本上是Spotify用户授予我们应用程序的权限

+   `refresh_token`：在过期后可以用来刷新`access_token`的令牌

最后一步是在`musicterminal/pytify/auth`目录中创建一个`__init__.py`文件，并导入`Authorization`，这是一个`namedtuple`：

```py
from .authorization import Authorization
```

# 实施客户端凭据流

客户端凭据流非常简单。让我们分解一下直到获得`access_token`的所有步骤：

1.  我们的应用程序将从Spotify帐户服务请求访问令牌；请记住，在我们的配置文件中，有`api_access_token`。这是我们需要发送请求以获取访问令牌的URL。我们需要发送请求的三件事是客户端ID、客户端密钥和授权类型，在这种情况下是`client_credentials`。

1.  Spotify帐户服务将验证该请求，检查密钥是否与我们在开发者网站注册的应用程序的密钥匹配，并返回一个访问令牌。

1.  现在，我们的应用程序必须使用此访问令牌才能从REST API中获取数据。

1.  Spotify REST API将返回我们请求的数据。

在开始实现将进行身份验证并获取访问令牌的函数之前，我们可以添加一个自定义异常，如果从Spotify帐户服务获得了错误请求（HTTP `400`）时，我们将抛出该异常。

让我们在`musicterminal/pytify/core`目录中创建一个名为`exceptions.py`的文件，内容如下：

```py
class BadRequestError(Exception):
    pass
```

这个类并没有做太多事情；我们只是继承自`Exception`。我们本可以只抛出一个通用异常，但是在开发其他开发人员将使用的框架和库时，最好创建自己的自定义异常，并使用良好的名称和描述。

因此，不要像这样抛出异常：

`raise Exception('some message')`

我们可以更明确地抛出`BadRequestError`，如下所示：

`raise BadRequestError('some message')`

现在，使用此代码的开发人员可以在其代码中正确处理此类异常。

打开`musicterminal/pytify/core`目录中的`__init__.py`文件，并添加以下导入语句：

```py
from .exceptions import BadRequestError
```

太好了！现在是时候在`musicterminal/pytify/auth`目录中添加一个名为`auth.py`的新文件了，我们要添加到此文件的第一件事是一些导入：

```py
import requests
import base64
import json

from .authorization import Authorization
from pytify.core import BadRequestError
```

我通常首先放置来自标准库模块的所有导入，然后是来自我的应用程序文件的函数导入。这不是必需的，但我认为这样可以使代码更清晰、更有组织。这样，我可以轻松地看出哪些是标准库项目，哪些不是。

现在，我们可以开始添加将发送请求到`Spotify`帐户服务并返回访问令牌的函数。我们要添加的第一个函数称为`get_auth_key`：

```py
def get_auth_key(client_id, client_secret):
    byte_keys = bytes(f'{client_id}:{client_secret}', 'utf-8')
    encoded_key = base64.b64encode(byte_keys)
    return encoded_key.decode('utf-8')
```

客户端凭据流要求我们发送`client_id`和`client_secret`，它必须是base 64编码的。首先，我们将字符串转换为`client_id:client_secret`格式的字节。然后，我们使用base 64对其进行编码，然后解码它，返回该编码数据的字符串表示，以便我们可以将其与请求有效负载一起发送。

我们要在同一文件中实现的另一个函数称为`_client_credentials`：

```py
def _client_credentials(conf):

    auth_key = get_auth_key(conf.client_id, conf.client_secret)

    headers = {'Authorization': f'Basic {auth_key}', }

    options = {
        'grant_type': 'client_credentials',
        'json': True,
        }

    response = requests.post(
        'https://accounts.spotify.com/api/token',
        headers=headers,
        data=options
    )

    content = json.loads(response.content.decode('utf-8'))

    if response.status_code == 400:
        error_description = content.get('error_description','')
        raise BadRequestError(error_description)

    access_token = content.get('access_token', None)
    token_type = content.get('token_type', None)
    expires_in = content.get('expires_in', None)
    scope = content.get('scope', None)    

    return Authorization(access_token, token_type, expires_in, 
    scope, None)
```

这个函数接收配置作为参数，并使用`get_auth_key`函数传递`client_id`和`client_secret`来构建一个base 64编码的`auth_key`。这将被发送到Spotify的账户服务以请求`access_token`。

现在，是时候准备请求了。首先，我们在请求头中设置`Authorization`，值将是`Basic`字符串后跟`auth_key`。这个请求的载荷将是`grant_type`，在这种情况下是`client_credentials`，`json`将设置为`True`，告诉API我们希望以JSON格式获取响应。

我们使用requests包向Spotify的账户服务发出请求，传递我们配置的头部和数据。

当我们收到响应时，我们首先解码并将JSON数据加载到变量content中。

如果HTTP状态码是`400 (BAD_REQUEST)`，我们会引发一个`BadRequestError`；否则，我们会获取`access_token`、`token_type`、`expires_in`和`scope`的值，最后创建一个`Authorization`元组并返回它。

请注意，当创建一个`Authentication`的`namedtuple`时，我们将最后一个参数设置为`None`。这样做的原因是，当身份验证类型为`CLIENT_CREDENTIALS`时，Spotify的账户服务不会返回`refresh_token`。

到目前为止，我们创建的所有函数都是私有的，所以我们要添加的最后一个函数是`authenticate`函数。这是开发人员将调用以开始身份验证过程的函数：

```py
def authenticate(conf):
    return _client_credentials(conf)
```

这个函数非常直接；函数接收一个`Config`的实例作为参数，`namedtuple`，其中包含了从配置文件中读取的所有数据。然后我们将配置传递给`_client_credentials`函数，该函数将使用客户端凭据流获取`access_token`。

让我们在`musicterminal/pytify/auth`目录中打开`__init__.py`文件，并导入`authenticate`和`get_auth_key`函数：

```py
from .auth import authenticate
from .auth import get_auth_key
```

很好！让我们在Python REPL中尝试一下：

```py
Python 3.6.2 (default, Oct 15 2017, 01:15:28)
[GCC 6.3.0 20170516] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from pytify.core import read_config
>>> from pytify.auth import authenticate
>>> config = read_config()
>>> auth = authenticate(config)
>>> auth
Authorization(access_token='BQDM_DC2HcP9kq5iszgDwhgDvq7zm1TzvzXXyJQwFD7trl0Q48DqoZirCMrMHn2uUml2YnKdHOszAviSFGtE6w', token_type='Bearer', expires_in=3600, scope=None, refresh_token=None)
>>>
```

正是我们所期望的！下一步是开始创建将消耗Spotify的REST API的函数。

# 实现授权码流程

在这一部分，我们将实现授权码流程，这是我们将在客户端中使用的流程。我们需要使用这种身份验证流程，因为我们需要从用户那里获得特殊的访问权限，以便使用我们的应用程序执行某些操作。例如，我们的应用程序将能够向Spotify的Web API发送请求，在用户的活动设备上播放某个曲目。为了做到这一点，我们需要请求`user-modify-playback-state`。

以下是授权码流程中涉及的步骤：

1.  我们的应用程序将请求授权以访问数据，并将用户重定向到Spotify网页上的登录页面。在那里，用户可以看到应用程序需要的所有访问权限。

1.  如果用户批准，Spotify账户服务将向回调URI发送一个请求，发送一个代码和状态。

1.  当我们获得了代码后，我们发送一个新的请求，传递`client_id`、`client_secret`、`grant_type`和`code`来获取`access_token`。这一次，它将与客户端凭据流不同；我们将获得`scope`和`refresh_token`。

1.  现在，我们可以正常地向Web API发送请求，如果访问令牌已过期，我们可以发送另一个请求来刷新访问令牌并继续执行请求。

说到这里，在`musicterminal/pytify/auth`目录中打开`auth.py`文件，让我们添加一些更多的函数。首先，我们将添加一个名为`_refresh_access_token`的函数；你可以在`get_auth_key`函数之后添加这个函数：

```py
def _refresh_access_token(auth_key, refresh_token):

    headers = {'Authorization': f'Basic {auth_key}', }

    options = {
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token',
        }

    response = requests.post(
        'https://accounts.spotify.com/api/token',
        headers=headers,
        data=options
    )

    content = json.loads(response.content.decode('utf-8'))

    if not response.ok:
        error_description = content.get('error_description', None)
        raise BadRequestError(error_description)

    access_token = content.get('access_token', None)
    token_type = content.get('token_type', None)
    scope = content.get('scope', None)
    expires_in = content.get('expires_in', None)

    return Authorization(access_token, token_type, expires_in, 
    scope, None)
```

它基本上与处理客户端凭据流的函数做同样的事情，但这次我们发送`refresh_token`和`grant_type`。我们从响应对象中获取数据并创建一个`Authorization`，`namedtuple`。

我们接下来要实现的下一个函数将利用标准库的`os`模块，因此在开始实现之前，我们需要在`auth.py`文件的顶部添加以下导入语句：

```py
import os
```

现在，我们可以继续添加一个名为`_authorization_code`的函数。您可以在`get_auth_key`函数之后添加此函数，并包含以下内容：

```py
def _authorization_code(conf):

    current_dir = os.path.abspath(os.curdir)
    file_path = os.path.join(current_dir, '.pytify')

    auth_key = get_auth_key(conf.client_id, conf.client_secret)

    try:
        with open(file_path, mode='r', encoding='UTF-8') as file:
            refresh_token = file.readline()

            if refresh_token:
                return _refresh_access_token(auth_key, 
                 refresh_token)

    except IOError:
        raise IOError(('It seems you have not authorize the 
                       application '
                       'yet. The file .pytify was not found.'))
```

在这里，我们尝试在`musicterminal`目录中打开一个名为`.pytify`的文件。这个文件将包含我们将用来刷新`access_token`的`refresh_token`。

从文件中获取`refresh_token`后，我们将其与`auth_key`一起传递给`_refresh_access_token`函数。如果由于某种原因我们无法打开文件或文件不存在于`musicterminal`目录中，将引发异常。

我们现在需要做的最后修改是在同一文件中的`authenticate`函数中。我们将为两种身份验证方法添加支持；它应该是这样的：

```py
def authenticate(conf):
    if conf.auth_method == AuthMethod.CLIENT_CREDENTIALS:
        return _client_credentials(conf)

    return _authorization_code(conf)
```

现在，我们将根据配置文件中的指定开始不同的身份验证方法。

由于身份验证函数引用了`AuthMethod`，我们需要导入它：

```py
from .auth_method import AuthMethod
```

在我们尝试这种类型的身份验证之前，我们需要创建一个小型的Web应用程序，它将为我们授权我们的应用程序。我们将在下一节中进行这方面的工作。

# 使用授权码流授权我们的应用程序

为了使我们的Spotify终端客户端正常工作，我们需要特殊的访问权限来操作用户的播放。我们通过使用授权码来做到这一点，我们需要专门请求`user-modify-playback-state`访问权限。

如果您打算为此应用程序添加更多功能，最好从一开始就添加一些其他访问权限；例如，如果您想要能够操作用户的私人和公共播放列表，您可能希望添加`playlist-modify-private`和`playlist-modify-public`范围。

您可能还希望在客户端应用程序上显示用户关注的艺术家列表，因此您还需要将`user-follow-read`包含在范围内。

对于我们将在客户端应用程序中实现的功能，请求`user-modify-playback-state`访问权限将足够。

我们的想法是使用授权码流授权我们的应用程序。我们将使用Flask框架创建一个简单的Web应用程序，该应用程序将定义两个路由。`/`根将只呈现一个简单的页面，其中包含一个链接，该链接将重定向我们到Spotify认证页面。

第二个根将是`/callback`，这是Spotify在我们的应用程序用户授权我们的应用程序访问其Spotify数据后将调用的端点。

让我们看看这是如何实现的，但首先，我们需要安装Flask。打开终端并输入以下命令：

```py
pip install flask
```

安装后，您甚至可以将其包含在`requirements.txt`文件中，如下所示：

```py
$ pip freeze | grep Flask >> requirements.txt
```

命令`pip freeze`将以requirements格式打印所有已安装的软件包。输出将返回更多项目，因为它还将包含我们已安装的软件包的所有依赖项，这就是为什么我们使用grep `Flask`并将其附加到`requirements.txt`文件中。

下次您要设置虚拟环境来处理这个项目时，只需运行：

```py
pip install -r requirements.txt
```

太棒了！现在，我们可以开始创建Web应用程序。创建一个名为`spotify_auth.py`的文件。

首先，我们添加所有必要的导入：

```py
from urllib.parse import urlencode

import requests
import json

from flask import Flask
from flask import render_template
from flask import request

from pytify.core import read_config
from pytify.core import BadRequestError
from pytify.auth import Authorization
from pytify.auth import get_auth_key
```

我们将使用`urllib.parse`模块中的`urlencode`函数来对要附加到授权URL的参数进行编码。我们还将使用requests来发送请求，以在用户授权我们的应用程序后获取`access_token`，并使用`json`包来解析响应。

然后，我们将导入与Flask相关的内容，以便创建一个Flask应用程序，`render_template`，以便将渲染的HTML模板返回给用户，最后是请求，以便我们可以访问Spotify授权服务返回给我们的数据。

我们还将导入一些我们在`pytify`模块的核心和auth子模块中包含的函数：`read_config`用于加载和读取YAML配置文件，以及`_authorization_code_request`。后者将在稍后详细解释。

我们将创建一个Flask应用程序和根路由：

```py
app = Flask(__name__)

@app.route("/")
def home():
    config = read_config()

    params = {
        'client_id': config.client_id,
        'response_type': 'code',
        'redirect_uri': 'http://localhost:3000/callback',
        'scope': 'user-read-private user-modify-playback-state',
    }

    enc_params = urlencode(params)
    url = f'{config.auth_url}?{enc_params}'

    return render_template('index.html', link=url)
```

太棒了！从头开始，我们读取配置文件，以便获取我们的`client_id`，还有Spotify授权服务的URL。我们使用`client_id`构建参数字典；授权代码流的响应类型需要设置为`code`；`redirect_uri`是回调URI，Spotify授权服务将用它来将授权代码发送回给我们。最后，由于我们将向REST API发送指令来播放用户活动设备中的曲目，应用程序需要具有`user-modify-playback-state`权限。

现在，我们对所有参数进行编码并构建URL。

返回值将是一个渲染的HTML。在这里，我们将使用`render_template`函数，将模板作为第一个参数传递。默认情况下，Flask将在一个名为`templates`的目录中搜索这个模板。这个函数的第二个参数是模型。我们传递了一个名为`link`的属性，并设置了变量URL的值。这样，我们可以在HTML模板中渲染链接，比如：`{{link}}`。

接下来，我们将添加一个函数，以在从Spotify的帐户服务获取授权代码后为我们获取`access_token`和`refresh_token`。创建一个名为`_authorization_code_request`的函数，内容如下：

```py
def _authorization_code_request(auth_code):
    config = read_config()

    auth_key = get_auth_key(config.client_id, config.client_secret)

    headers = {'Authorization': f'Basic {auth_key}', }

    options = {
        'code': auth_code,
        'redirect_uri': 'http://localhost:3000/callback',
        'grant_type': 'authorization_code',
        'json': True
    }

    response = requests.post(
        config.access_token_url,
        headers=headers,
        data=options
    )

    content = json.loads(response.content.decode('utf-8'))

    if response.status_code == 400:
        error_description = content.get('error_description', '')
        raise BadRequestError(error_description)

    access_token = content.get('access_token', None)
    token_type = content.get('token_type', None)
    expires_in = content.get('expires_in', None)
    scope = content.get('scope', None)
    refresh_token = content.get('refresh_token', None)

    return Authorization(access_token, token_type, expires_in, 
    scope, refresh_token)
```

这个函数与我们之前在`auth.py`文件中实现的`_refresh_access_token`函数基本相同。这里唯一需要注意的是，在选项中，我们传递了授权代码，`grant_type`设置为`authorization_code`：

```py
@app.route('/callback')
def callback():
    config = read_config()
    code = request.args.get('code', '')
    response = _authorization_code_request(config, code)

    file = open('.pytify', mode='w', encoding='utf-8')
    file.write(response.refresh_token)
    file.close()

    return 'All set! You can close the browser window and stop the 
    server.'
```

在这里，我们定义了将由Spotify授权服务调用以发送授权代码的路由。

我们首先读取配置，解析请求数据中的代码，并调用`_authorization_code_request`，传递我们刚刚获取的代码。

这个函数将使用这个代码发送另一个请求，并获取一个我们可以用来发送请求的访问令牌，以及一个将存储在`musicterminal`目录中名为`.pytify`的文件中的刷新令牌。

我们获取的用于向Spotify REST API发出请求的访问令牌有效期为3,600秒，或1小时，这意味着在一个小时内，我们可以使用相同的访问令牌发出请求。之后，我们需要刷新访问令牌。我们可以通过使用存储在`.pytify`文件中的刷新令牌来实现。

最后，我们向浏览器发送一个成功消息。

现在，为了完成我们的Flask应用程序，我们需要添加以下代码：

```py
if __name__ == '__main__':
    app.run(host='localhost', port=3000)
```

这告诉Flask在本地主机上运行服务器，并使用端口`3000`。

我们的Flash应用程序的`home`函数将作为响应返回一个名为index.html的模板化HTML文件。我们还没有创建该文件，所以让我们继续创建一个名为`musicterminal/templates`的文件夹，并在新创建的目录中添加一个名为`index.html`的文件，内容如下：

```py
<html>
    <head>
    </head>
    <body>
       <a href={{link}}> Click here to authorize </a>
    </body>
</html>
```

这里没有太多解释的地方，但请注意我们正在引用链接属性，这是我们在Flask应用程序的主页函数中传递给`render_template`函数的。我们将锚元素的`href`属性设置为链接的值。

太好了！在我们尝试这个并查看一切是否正常工作之前，还有一件事情。我们需要更改Spotify应用程序的设置；更具体地说，我们需要配置应用程序的回调函数，以便我们可以接收授权码。

说到这一点，前往[https://beta.developer.spotify.com/dashboard/](https://beta.developer.spotify.com/dashboard/)网站，并使用你的凭据登录。仪表板将显示我们在本章开头创建的`pytify`应用程序。点击应用程序名称，然后点击页面右上角的`EDIT SETTINGS`按钮。

向下滚动直到找到重定向URI，在文本框中输入http://localhost:3000/callback，然后点击添加按钮。你的配置应该如下所示：

![](assets/bc6aecdc-cc8d-41c2-b322-0e5104dee0e5.png)

太好了！滚动到对话框底部，点击保存按钮。

现在，我们需要运行我们刚刚创建的Flask应用程序。在终端中，进入项目的根目录，输入以下命令：

```py
python spotify_auth.py
```

你应该会看到类似于这样的输出：

```py
* Running on http://localhost:3000/ (Press CTRL+C to quit)
```

打开你选择的浏览器，转到`http://localhost:3000`；你将看到一个简单的页面，上面有我们创建的链接：

![](assets/6bc6cdd8-da51-453f-a117-5ac4fe18ee67.png)

点击链接，你将被发送到Spotify的授权服务页面。

一个对话框将显示，要求将`Pytify`应用程序连接到我们的账户。一旦你授权了它，你将被重定向回`http://localhost:3000/callback`。如果一切顺利，你应该在页面上看到`All set! You can close the browser window and stop the server`的消息。

现在，只需关闭浏览器，你就可以停止Flask应用程序了。

请注意，现在在`musicterminal`目录中有一个名为`.pytify`的文件。如果你查看内容，你会看到一个类似于这样的加密密钥：

```py
AQB2jJxziOvuj1VW_DOBeJh-uYWUYaR03nWEJncKdRsgZC6ql2vaUsVpo21afco09yM4tjwgt6Kkb_XnVC50CR0SdjWrrbMnr01zdemN0vVVHmrcr_6iMxCQSk-JM5yTjg4
```

现在，我们准备开始编写播放器。

接下来，我们将添加一些函数，用于向Spotify的Web API发送请求，搜索艺术家，获取艺术家专辑的列表和专辑中的曲目列表，并播放所选的曲目。

# 查询Spotify的Web API

到目前为止，我们只是准备了地形，现在事情开始变得更有趣了。在这一部分，我们将创建基本函数来向Spotify的Web API发送请求；更具体地说，我们想要能够搜索艺术家，获取艺术家专辑的列表，获取该专辑中的曲目列表，最后我们想要发送一个请求来实际播放Spotify客户端中当前活动的曲目。可以是浏览器、手机、Spotify客户端，甚至是视频游戏主机。所以，让我们马上开始吧！

首先，我们将在`musicterminal/pytify/core`目录中创建一个名为`request_type.py`的文件，内容如下：

```py
from enum import Enum, auto

class RequestType(Enum):
    GET = auto()
    PUT = auto()
```

我们之前已经讨论过枚举，所以我们不会详细讨论。可以说我们创建了一个包含`GET`和`PUT`属性的枚举。这将用于通知为我们执行请求的函数，我们想要进行`GET`请求还是`PUT`请求。

然后，我们可以在相同的`musicterminal/pytify/core`目录中创建另一个名为`request.py`的文件，并开始添加一些导入语句，并定义一个名为`execute_request`的函数：

```py
import requests
import json

from .exceptions import BadRequestError
from .config import read_config
from .request_type import RequestType

def execute_request(
        url_template,
        auth,
        params,
        request_type=RequestType.GET,
        payload=()):

```

这个函数有一些参数：

+   `url_template`：这是将用于构建执行请求的URL的模板；它将使用另一个名为`params`的参数来构建URL

+   `auth`：是`Authorization`对象

+   `params`：这是一个包含我们将放入我们将要执行请求的URL中的所有参数的`dict`

+   `request`：这是请求类型；可以是`GET`或`PUT`

+   `payload`：这是可能与请求一起发送的数据

随着我们继续实现相同的功能，我们可以添加：

```py
conf = read_config()

params['base_url'] = conf.base_url

url = url_template.format(**params)

headers = {
    'Authorization': f'Bearer {auth.access_token}'
}
```

我们读取配置并将基本URL添加到参数中，以便在`url_template`字符串中替换它。我们在请求标头中添加`Authorization`，以及认证访问令牌：

```py
if request_type is RequestType.GET:
    response = requests.get(url, headers=headers)
else:
    response = requests.put(url, headers=headers, data=json.dumps(payload))

    if not response.text:
        return response.text

result = json.loads(response.text)
```

在这里，我们检查请求类型是否为`GET`。如果是，我们执行来自requests的`get`函数；否则，我们执行`put`函数。函数调用非常相似；这里唯一不同的是数据参数。如果返回的响应为空，我们只返回空字符串；否则，我们将JSON数据解析为`result`变量：

```py
if not response.ok:
    error = result['error']
    raise BadRequestError(
        f'{error["message"]} (HTTP {error["status"]})')

return result
```

解析JSON结果后，我们测试请求的状态是否不是`200`（OK）；在这种情况下，我们引发`BadRequestError`。如果是成功的响应，我们返回结果。

我们还需要一些函数来帮助我们准备要传递给Web API端点的参数。让我们继续在`musicterminal/pytify/core`文件夹中创建一个名为`parameter.py`的文件，内容如下：

```py
from urllib.parse import urlencode

def validate_params(params, required=None):

    if required is None:
        return

    partial = {x: x in params.keys() for x in required}
    not_supplied = [x for x in partial.keys() if not partial[x]]

    if not_supplied:
        msg = f'The parameter(s) `{", ".join(not_supplied)}` are 
        required'
        raise AttributeError(msg)

def prepare_params(params, required=None):

    if params is None and required is not None:
        msg = f'The parameter(s) `{", ".join(required)}` are 
        required'
        raise ValueErrorAttributeError(msg)
    elif params is None and required is None:
        return ''
    else:
        validate_params(params, required)

    query = urlencode(
        '&'.join([f'{key}={value}' for key, value in 
         params.items()])
    )

    return f'?{query}'
```

这里有两个函数，`prepare_params`和`validate_params`。`validate_params`函数用于识别是否有参数需要进行某种操作，但它们尚未提供。`prepare_params`函数首先调用`validate_params`，以确保所有参数都已提供，并将所有参数连接在一起，以便它们可以轻松附加到URL查询字符串中。

现在，让我们添加一个枚举，列出可以执行的搜索类型。在`musicterminal/pytify/core`目录中创建一个名为`search_type.py`的文件，内容如下：

```py
from enum import Enum

class SearchType(Enum):
    ARTIST = 1
    ALBUM = 2
    PLAYLIST = 3
    TRACK = 4
```

这只是一个简单的枚举，列出了四个搜索选项。

现在，我们准备创建执行搜索的函数。在`musicterminal/pytify/core`目录中创建一个名为`search.py`的文件：

```py
import requests
import json
from urllib.parse import urlencode

from .search_type import SearchType
from pytify.core import read_config

def _search(criteria, auth, search_type):

    conf = read_config()

    if not criteria:
        raise AttributeError('Parameter `criteria` is required.')

    q_type = search_type.name.lower()
    url = urlencode(f'{conf.base_url}/search?q={criteria}&type=
    {q_type}')

    headers = {'Authorization': f'Bearer {auth.access_token}'}
    response = requests.get(url, headers=headers)

    return json.loads(response.text)

def search_artist(criteria, auth):
    return _search(criteria, auth, SearchType.ARTIST)

def search_album(criteria, auth):
    return _search(criteria, auth, SearchType.ALBUM)

def search_playlist(criteria, auth):
    return _search(criteria, auth, SearchType.PLAYLIST)

def search_track(criteria, auth):
    return _search(criteria, auth, SearchType.TRACK)
```

我们首先解释`_search`函数。这个函数获取三个标准参数（我们要搜索的内容），`Authorization`对象，最后是搜索类型，这是我们刚刚创建的枚举中的一个值。

这个函数非常简单；我们首先验证参数，然后构建URL以进行请求，我们使用我们的访问令牌设置`Authorization`头，最后，我们执行请求并返回解析后的响应。

其他功能`search_artist`，`search_album`，`search_playlist`和`search_track`只是获取相同的参数，标准和`Authorization`对象，并将其传递给`_search`函数，但它们传递不同的搜索类型。

现在我们可以搜索艺术家，我们必须获取专辑列表。在`musicterminal/pytify/core`目录中添加一个名为`artist.py`的文件，内容如下：

```py
from .parameter import prepare_params
from .request import execute_request

def get_artist_albums(artist_id, auth, params=None):

    if artist_id is None or artist_id is "":
        raise AttributeError(
            'Parameter `artist_id` cannot be `None` or empty.')

    url_template = '{base_url}/{area}/{artistid}/{postfix}{query}'
    url_params = {
        'query': prepare_params(params),
        'area': 'artists',
        'artistid': artist_id,
        'postfix': 'albums',
        }

    return execute_request(url_template, auth, url_params)
```

因此，给定一个`artist_id`，我们只需定义URL模板和我们要发出请求的参数，并运行`execute_request`函数，它将负责为我们构建URL，获取和解析结果。

现在，我们想要获取给定专辑的曲目列表。在`musicterminal/pytify/core`目录中添加一个名为`album.py`的文件，内容如下：

```py
from .parameters import prepare_params
from .request import execute_request

def get_album_tracks(album_id, auth, params=None):

    if album_id is None or album_id is '':
        raise AttributeError(
            'Parameter `album_id` cannot be `None` or empty.')

    url_template = '{base_url}/{area}/{albumid}/{postfix}{query}'
    url_params = {
        'query': prepare_params(params),
        'area': 'albums',
        'albumid': album_id,
        'postfix': 'tracks',
        }

    return execute_request(url_template, auth, url_params)
```

`get_album_tracks`函数与我们刚刚实现的`get_artist_albums`函数非常相似。

最后，我们希望能够向Spotify的Web API发送指令，告诉它播放我们选择的曲目。在`musicterminal/pytify/core`目录中添加一个名为`player.py`的文件，并添加以下内容：

```py
from .parameter import prepare_params
from .request import execute_request

from .request_type import RequestType

def play(track_uri, auth, params=None):

    if track_uri is None or track_uri is '':
        raise AttributeError(
            'Parameter `track_uri` cannot be `None` or empty.')

    url_template = '{base_url}/{area}/{postfix}'
    url_params = {
        'query': prepare_params(params),
        'area': 'me',
        'postfix': 'player/play',
        }

    payload = {
        'uris': [track_uri],
        'offset': {'uri': track_uri}
    }

    return execute_request(url_template,
                           auth,
                           url_params,
                           request_type=RequestType.PUT,
                           payload=payload)
```

这个函数与之前的函数（`get_artist_albums`和`get_album_tracks`）非常相似，只是它定义了一个有效负载。有效负载是一个包含两个项目的字典：`uris`，是应该添加到播放队列的曲目列表，和`offset`，其中包含另一个包含应该首先播放的曲目的URI的字典。由于我们只对一次播放一首歌感兴趣，`uris`和`offset`将包含相同的`track_uri`。

这里的最后一步是导入我们实现的新函数。在`musicterminal/pytify/core`目录下的`__init__.py`文件中，添加以下代码：

```py
from .search_type import SearchType

from .search import search_album
from .search import search_artist
from .search import search_playlist
from .search import search_track

from .artist import get_artist_albums
from .album import get_album_tracks
from .player import play
```

让我们尝试在python REPL中搜索艺术家的函数，以检查一切是否正常工作：

```py
Python 3.6.2 (default, Dec 22 2017, 15:38:46)
[GCC 6.3.0 20170516] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from pytify.core import search_artist
>>> from pytify.core import read_config
>>> from pytify.auth import authenticate
>>> from pprint import pprint as pp
>>>
>>> config = read_config()
>>> auth = authenticate(config)
>>> results = search_artist('hot water music', auth)
>>> pp(results)
{'artists': {'href': 'https://api.spotify.com/v1/search?query=hot+water+music&type=artist&market=SE&offset=0&limit=20',
 'items': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4dmaYARGTCpChLhHBdr3ff'},
 'followers': {'href': None, 'total': 56497},
 'genres': ['alternative emo',
 'emo',
 'emo punk', 
```

其余输出已被省略，因为太长了，但现在我们可以看到一切都正如预期地工作。

现在，我们准备开始构建终端播放器。

# 创建播放器

现在我们已经拥有了认证和使用Spotify Rest API所需的一切，我们将创建一个小型终端客户端，可以在其中搜索艺术家，浏览他/她的专辑，并选择要在Spotify客户端中播放的曲目。请注意，要使用客户端，我们将不得不从高级账户中发出访问令牌，并且我们需要在这里使用的认证流程是`AUTHENTICATION_CODE`。

我们还需要从我们应用程序的用户那里要求`user-modify-playback-state`范围，这将允许我们控制播放。说到这里，让我们开始吧！

首先，我们需要创建一个新目录，将所有客户端相关的文件保存在其中，所以继续创建一个名为`musicterminal/client`的目录。

我们的客户端只有三个视图。在第一个视图中，我们将获取用户输入并搜索艺术家。当艺术家搜索完成后，我们将切换到第二个视图，在这个视图中，将呈现所选艺术家的专辑列表。在这个视图中，用户将能够使用键盘的*上*和*下*箭头键选择列表上的专辑，并通过按*Enter*键选择专辑。

最后，当选择了一个专辑后，我们将切换到我们应用程序的第三个和最后一个视图，用户将看到所选专辑的曲目列表。与之前的视图一样，用户还可以使用键盘的*上*和*下*箭头键选择曲目；按*Enter*将向Spotify API发送请求，在用户可用设备上播放所选曲目。

一种方法是使用`curses.panel`。面板是一种窗口，非常灵活，允许我们堆叠、隐藏和显示、切换面板，返回到面板堆栈的顶部等等，非常适合我们的目的。

因此，让我们在`musicterminal/client`目录下创建一个名为`panel.py`的文件，内容如下：

```py
import curses
import curses.panel
from uuid import uuid1

class Panel:

    def __init__(self, title, dimensions):
        height, width, y, x = dimensions

        self._win = curses.newwin(height, width, y, x)
        self._win.box()
        self._panel = curses.panel.new_panel(self._win)
        self.title = title
        self._id = uuid1()

        self._set_title()

        self.hide()
```

我们所做的就是导入我们需要的模块和函数，并创建一个名为`Panel`的类。我们还导入`uuid`模块，以便为每个新面板创建一个GUID。

面板的初始化器有两个参数：`title`，是窗口的标题，和`dimensions`。`dimensions`参数是一个元组，遵循curses的约定。它由`height`、`width`和面板应该开始绘制的位置`y`和`x`组成。

我们解包`dimensions`元组的值，以便更容易处理，然后我们使用`newwin`函数创建一个新窗口；它将具有我们在类初始化器中传递的相同尺寸。接下来，我们调用box函数在终端的四个边上绘制线条。

现在我们已经创建了窗口，是时候为我们刚刚创建的窗口创建面板了，调用`curses.panel.new_panel`并传递窗口。我们还设置窗口标题并创建一个GUID。

最后，我们将面板的状态设置为隐藏。继续在这个类上工作，让我们添加一个名为`hide`的新方法：

```py
def hide(self):
    self._panel.hide()
```

这个方法非常简单；它所做的唯一的事情就是调用我们面板中的`hide`方法。

我们在初始化器中调用的另一个方法是`_set_title`；现在让我们创建它：

```py
def _set_title(self):
    formatted_title = f' {self._title} '
    self._win.addstr(0, 2, formatted_title, curses.A_REVERSE)
```

在`_set_title`中，我们通过在标题字符串的两侧添加一些额外的填充来格式化标题，然后我们调用窗口的`addstr`方法在零行、二列打印标题，并使用常量`A_REVERSE`，它将颠倒字符串的颜色，就像这样：

![](assets/ae53d748-1e23-4730-8538-9be17dde71de.png)

我们有一个隐藏面板的方法；现在，我们需要一个显示面板的方法。让我们添加`show`方法：

```py
def show(self):
    self._win.clear()
    self._win.box()
    self._set_title()
    curses.curs_set(0)
    self._panel.show()
```

`show`方法首先清除窗口并用`box`方法绘制其周围的边框。然后，我们再次设置`title`。`cursers.curs_set(0)`调用将禁用光标；我们在这里这样做是因为当我们在列表中选择项目时，我们不希望光标可见。最后，我们在面板中调用`show`方法。

也很好有一种方法来知道当前面板是否可见。因此，让我们添加一个名为`is_visible`的方法：

```py
def is_visible(self):
    return not self._panel.hidden()
```

在这里，我们可以在面板上使用`hidden`方法，如果面板隐藏则返回`true`，如果面板可见则返回`false`。

在这个类中的最后一步是添加比较面板的可能性。我们可以通过覆盖一些特殊方法来实现这一点；在这种情况下，我们想要覆盖`__eq__`方法，每当使用`==`运算符时都会调用它。记住我们为每个面板创建了一个`id`吗？我们现在可以使用那个`id`来测试相等性：

```py
def __eq__(self, other):
    return self._id == other._id
```

太好了！现在我们有了`Panel`基类，我们准备创建一个特殊的面板实现，其中将包含选择项目的菜单。

# 为专辑和曲目选择添加菜单

现在，我们将在`musicterminal/client/`目录中创建一个名为`menu_item.py`的文件，并且我们将从中导入一些我们需要的函数开始：

```py
from uuid import uuid1
```

我们只需要从`uuid`模块中导入`uuid1`函数，因为和面板一样，我们将为列表中的每个菜单项创建一个`id（GUID）`。

让我们首先添加类和构造函数：

```py
class MenuItem:
    def __init__(self, label, data, selected=False):
        self.id = str(uuid1())
        self.data = data
        self.label = label

        def return_id():
            return self.data['id'], self.data['uri']

        self.action = return_id
        self.selected = selected
```

`MenuItem`初始化器有三个参数，`label`项，`data`将包含Spotify REST API返回的原始数据，以及一个指示项目当前是否被选中的标志。

我们首先为项目创建一个id，然后使用传递给类初始化器的参数值设置数据和标签属性的值。

列表中的每个项目都将有一个在选择列表项时执行的操作，因此我们创建一个名为`return_id`的函数，它返回一个包含项目id的元组（不同于我们刚刚创建的id）。这是Spotify上项目的id，URI是Spotify上项目的URI。当我们选择并播放一首歌时，后者将会很有用。

现在，我们将实现一些特殊方法，这些方法在执行项目比较和打印项目时将对我们很有用。我们要实现的第一个方法是`__eq__`：

```py
def __eq__(self, other):
    return self.id == other.id
```

这将允许我们使用`index`函数在`MenuItem`对象列表中找到特定的`MenuItem`。

我们要实现的另一个特殊方法是`__len__`方法：

```py
def __len__(self):
    return len(self.label)
```

它返回`MenuItem`标签的长度，当测量列表中菜单项标签的长度时将会用到。稍后，当我们构建菜单时，我们将使用`max`函数来获取具有最长标签的菜单项，并基于此，我们将为其他项目添加额外的填充，以便列表中的所有项目看起来对齐。

我们要实现的最后一个方法是`__str__`方法：

```py
def __str__(self):
    return self.label
```

这只是在打印菜单项时的便利性；我们可以直接调用`print(menuitem)`而不是`print(menuitem.label)`，它将调用`__str__`，返回`MenuItem`标签的值。

# 实现菜单面板

现在，我们将实现菜单面板，它将是一个容器类，容纳所有菜单项，处理事件，并在终端屏幕上执行呈现。

在我们开始实现菜单面板之前，让我们添加一个枚举，表示不同的项目对齐选项，这样我们就可以更灵活地显示菜单中的菜单项。

在`musicterminal/client`目录中创建一个名为`alignment.py`的文件，内容如下：

```py
from enum import Enum, auto

class Alignment(Enum):
    LEFT = auto()
    RIGHT = auto()
```

如果您在第一章中跟随代码，您应该是一个枚举专家。这里没有什么复杂的；我们定义了一个从Enum继承的`Alignment`类，并定义了两个属性，`LEFT`和`RIGHT`，它们的值都设置为`auto()`，这意味着值将自动设置为`1`和`2`。

现在，我们准备创建菜单。让我们继续在`musicterminal/client`目录中创建一个名为`menu.py`的最终类。

让我们添加一些导入和构造函数：

```py
import curses
import curses.panel

from .alignment import Alignment
from .panel import Panel

class Menu(Panel):

    def __init__(self, title, dimensions, align=Alignment.LEFT, 
                 items=[]):
        super().__init__(title, dimensions)
        self._align = align
        self.items = items
```

`Menu`类继承自我们刚刚创建的`Panel`基类，类初始化器接收一些参数：`title`，`dimensions`（包含`height`，`width`，`y`和`x`值的元组），默认为`LEFT`的`alignment`设置，以及`items`。items参数是一个`MenuItems`对象的列表。这是可选的，如果没有指定值，它将设置为空列表。

在类初始化器中的第一件事是调用基类的`__init__`方法。我们可以使用`super`函数来做到这一点。如果您记得，`Panel`类上的`__init__`方法有两个参数，`title`和`dimension`，所以我们将它传递给基类初始化器。

接下来，我们为属性`align`和`items`赋值。

我们还需要一个方法，返回菜单项列表中当前选定的项目：

```py
def get_selected(self):
    items = [x for x in self.items if x.selected]
    return None if not items else items[0]
```

这个方法非常简单；推导返回一个选定项目的列表，如果没有选定项目，则返回`None`；否则，返回列表中的第一个项目。

现在，我们可以实现处理项目选择的方法。让我们添加另一个名为`_select`的方法：

```py
def _select(self, expr):
    current = self.get_selected()
    index = self.items.index(current)
    new_index = expr(index)

    if new_index < 0:
        return

    if new_index > index and new_index >= len(self.items):
        return

    self.items[index].selected = False
    self.items[new_index].selected = True
```

在这里，我们开始获取当前选定的项目，然后立即使用数组中的索引方法获取菜单项列表中项目的索引。这是因为我们在`Panel`类中实现了`__eq__`方法。

然后，我们开始运行作为参数传递的函数`expr`，传递当前选定项目索引的值。

`expr`将确定下一个当前项目索引。如果新索引小于`0`，这意味着我们已经到达菜单项列表的顶部，因此我们不采取任何行动。

如果新索引大于当前索引，并且新索引大于或等于列表中菜单项的数量，则我们已经到达列表底部，因此此时不需要采取任何操作，我们可以继续选择相同的项目。

但是，如果我们还没有到达列表的顶部或底部，我们需要交换选定的项目。为此，我们将当前项目的selected属性设置为`False`，并将下一个项目的selected属性设置为`True`。

`_select`方法是一个`private`方法，不打算在外部调用，因此我们定义了两个方法——`next`和`previous`：

```py
def next(self):
    self._select(lambda index: index + 1)

def previous(self):
    self._select(lambda index: index - 1)
```

下一个方法将调用`_select`方法，并传递一个lambda表达式，该表达式将接收一个索引并将其加一，而上一个方法将执行相同的操作，但是不是增加索引`1`，而是减去。因此，在`_select`方法中，当我们调用：

```py
new_index = expr(index)
```

我们要么调用`lambda index: index + 1`，要么调用`lambda index: index + 1`。

太好了！现在，我们将添加一个负责在屏幕上呈现菜单项之前格式化菜单项的方法。创建一个名为`_initialize_items`的方法，如下所示：

```py
def _initialize_items(self):
    longest_label_item = max(self.items, key=len)

    for item in self.items:
        if item != longest_label_item:
            padding = (len(longest_label_item) - len(item)) * ' '
            item.label = (f'{item}{padding}'
                          if self._align == Alignment.LEFT
                          else f'{padding}{item}')

        if not self.get_selected():
            self.items[0].selected = True
```

首先，我们获取具有最大标签的菜单项；我们可以通过使用内置函数`max`并传递`items`，以及作为键的另一个内置函数`len`来实现这一点。这将起作用，因为我们在菜单项中实现了特殊方法`__len__`。

在发现具有最大标签的菜单项之后，我们循环遍历列表的项目，在`LEFT`或`RIGHT`上添加填充，具体取决于对齐选项。最后，如果列表中没有被选中标志设置为`True`的菜单项，我们将选择第一个项目作为选定项目。

我们还想提供一个名为`init`的方法，它将为我们初始化列表上的项目：

```py
def init(self):
    self._initialize_items()
```

我们还需要处理键盘事件，这样当用户特别按下*上*和*下*箭头键以及*Enter*键时，我们就可以执行一些操作。

首先，我们需要在文件顶部定义一些常量。您可以在导入和类定义之间添加这些常量：

```py
NEW_LINE = 10 CARRIAGE_RETURN = 13
```

让我们继续包括一个名为`handle_events`的方法：

```py
    def handle_events(self, key):
        if key == curses.KEY_UP:
            self.previous()
        elif key == curses.KEY_DOWN:
            self.next()
        elif key == curses.KEY_ENTER or key == NEW_LINE or key == 
         CARRIAGE_RETURN:
            selected_item = self.get_selected()
            return selected_item.action
```

这个方法非常简单；它获取一个`key`参数，如果键等于`curses.KEY_UP`，那么我们调用`previous`方法。如果键等于`curses.KEY_DOWN`，那么我们调用`next`方法。现在，如果键是`ENTER`，那么我们获取选定的项目并返回其操作。操作是一个将执行另一个函数的函数；在我们的情况下，我们可能会在列表上选择艺术家或歌曲，或执行一个将播放音乐曲目的函数。

除了测试`key`是否为`curses.KEY_ENTER`之外，我们还需要检查键是否为换行符`\n`或回车符`\r`。这是必要的，因为*Enter*键的代码可能会根据应用程序运行的终端的配置而有所不同。

我们将实现`__iter__`方法，这将使我们的`Menu`类表现得像一个可迭代的对象：

```py
    def __iter__(self):
        return iter(self.items)
```

这个类的最后一个方法是`update`方法。这个方法将实际工作渲染菜单项并刷新窗口屏幕：

```py
def update(self):
    pos_x = 2
    pos_y = 2

    for item in self.items:
        self._win.addstr(
                pos_y,
                pos_x,
                item.label,
                curses.A_REVERSE if item.selected else 
                curses.A_NORMAL)
        pos_y += 1

    self._win.refresh()
```

首先，我们将`x`和`y`坐标设置为`2`，这样窗口上的菜单将从第`2`行和第`2`列开始。我们循环遍历菜单项，并调用`addstr`方法在屏幕上打印项目。

`addstr`方法获取`y`位置，`x`位置，将在屏幕上写入的字符串，在我们的例子中是`item.label`，最后一个参数是`style`。如果项目被选中，我们希望以突出显示的方式显示它；否则，它将以正常颜色显示。以下截图说明了渲染列表的样子：

![](assets/270dbea3-ceb9-4922-8216-c30ab85c7688.png)

# 创建DataManager类

我们已经实现了身份验证和从Spotify REST API获取数据的基本功能，但现在我们需要创建一个类，利用这些功能，以便获取我们需要在客户端中显示的信息。

我们的Spotify终端客户端将执行以下操作：

+   按名称搜索艺术家

+   列出艺术家的专辑

+   列出专辑的曲目

+   请求播放一首曲目

我们要添加的第一件事是一个自定义异常，我们可以引发，而且没有从Spotify REST API返回结果。在`musicterminal/client`目录中创建一个名为`empty_results_error.py`的新文件，内容如下：

```py
class EmptyResultsError(Exception):
    pass
```

为了让我们更容易，让我们创建一个称为`DataManager`的类，它将为我们封装所有这些功能。在`musicterminal/client`目录中创建一个名为`data_manager.py`的文件：

```py
from .menu_item import MenuItem

from pytify.core import search_artist
from pytify.core import get_artist_albums
from pytify.core import get_album_tracks
from pytify.core import play

from .empty_results_error import EmptyResultsError

from pytify.auth import authenticate
from pytify.core import read_config

class DataManager():

    def __init__(self):
        self._conf = read_config()
        self._auth = authenticate(self._conf)
```

首先，我们导入`MenuItem`，这样我们就可以返回带有请求结果的`MenuItem`对象。之后，我们从`pytify`模块导入函数来搜索艺术家，获取专辑，列出专辑曲目，并播放曲目。此外，在`pytify`模块中，我们导入`read_config`函数并对其进行身份验证。

最后，我们导入刚刚创建的自定义异常`EmptyResultsError`。

`DataManager`类的初始化器开始读取配置并执行身份验证。身份验证信息将存储在`_auth`属性中。

接下来，我们将添加一个搜索艺术家的方法：

```py
def search_artist(self, criteria):
    results = search_artist(criteria, self._auth)
    items = results['artists']['items']

    if not items:
        raise EmptyResultsError(f'Could not find the artist: 
        {criteria}')

    return items[0]
```

`_search_artist`方法将`criteria`作为参数，并调用`python.core`模块中的`search_artist`函数。如果没有返回项目，它将引发一个`EmptyResultsError`；否则，它将返回第一个匹配项。

在我们继续创建将获取专辑和音轨的方法之前，我们需要两个实用方法来格式化`MenuItem`对象的标签。

第一个方法将格式化艺术家标签：

```py
def _format_artist_label(self, item):
    return f'{item["name"]} ({item["type"]})'
```

在这里，标签将是项目的名称和类型，可以是专辑、单曲、EP等。

第二个方法格式化音轨的名称：

```py
def _format_track_label(self, item):

    time = int(item['duration_ms'])
    minutes = int((time / 60000) % 60)
    seconds = int((time / 1000) % 60)

    track_name = item['name']

    return f'{track_name} - [{minutes}:{seconds}]'
```

在这里，我们提取音轨的持续时间（以毫秒为单位），将其转换为`分钟：秒`的格式，并使用音轨的名称和持续时间在方括号之间格式化标签。

之后，让我们创建一个获取艺术家专辑的方法：

```py
def get_artist_albums(self, artist_id, max_items=20):

     albums = get_artist_albums(artist_id, self._auth)['items']

     if not albums:
         raise EmptyResultsError(('Could not find any albums for'
                                  f'the artist_id: {artist_id}'))

     return [MenuItem(self._format_artist_label(album), album)
             for album in albums[:max_items]]
```

`get_artist_albums`方法接受两个参数，`artist_id`和`max_item`，它是该方法返回的专辑最大数量。默认情况下，它设置为`20`。

我们在这里首先使用`pytify.core`模块中的`get_artist_albums`方法，传递`artist_id`和`authentication`对象，并从结果中获取项目的属性，将其分配给变量专辑。如果`albums`变量为空，它将引发一个`EmptyResultsError`；否则，它将为每个专辑创建一个`MenuItem`对象的列表。

我们还可以为音轨添加另一个方法：

```py
def get_album_tracklist(self, album_id):

    results = get_album_tracks(album_id, self._auth)

    if not results:
        raise EmptyResultsError('Could not find the tracks for this 
        album')

    tracks = results['items']

    return [MenuItem(self._format_track_label(track), track)
            for track in tracks]
```

`get_album_tracklist`方法以`album_id`作为参数，我们首先使用`pytify.core`模块中的`get_album_tracks`函数获取该专辑的音轨。如果没有返回结果，我们会引发一个`EmptyResultsError`；否则，我们会构建一个`MenuItem`对象的列表。

最后一个方法实际上是将命令发送到Spotify REST API播放音轨的方法：

```py
def play(self, track_uri):
    play(track_uri, self._auth)
```

非常直接。在这里，我们只是将`track_uri`作为参数，并将其传递给`pytify.core`模块中的`play`函数，以及`authentication`对象。这将使音轨开始在可用设备上播放；可以是手机、您计算机上的Spotify客户端、Spotify网络播放器，甚至您的游戏机。

接下来，让我们把我们建立的一切放在一起，并运行Spotify播放器终端。

# 是时候听音乐了！

现在，我们拥有了开始构建终端播放器所需的所有部件。我们有`pytify`模块，它提供了Spotify RESP API的包装器，并允许我们搜索艺术家、专辑、音轨，甚至控制运行在手机或计算机上的Spotify客户端。

`pytify`模块还提供了两种不同类型的身份验证——客户端凭据和授权代码——在之前的部分中，我们实现了构建使用curses的应用程序所需的所有基础设施。因此，让我们将所有部分粘合在一起，听一些好音乐。

在`musicterminal`目录中，创建一个名为`app.py`的文件；这将是我们应用程序的入口点。我们首先添加导入语句：

```py
import curses
import curses.panel
from curses import wrapper
from curses.textpad import Textbox
from curses.textpad import rectangle

from client import Menu
from client import DataManager
```

我们当然需要导入`curses`和`curses.panel`，这次我们还导入了`wrapper`。这用于调试目的。在开发curses应用程序时，它们极其难以调试，当出现问题并抛出异常时，终端将无法返回到其原始状态。

包装器接受一个`callable`，当`callable`函数返回时，它将返回终端的原始状态。

包装器将在try-catch块中运行可调用项，并在出现问题时恢复终端。在开发应用程序时非常有用。让我们使用包装器，这样我们就可以看到可能发生的任何问题。

我们将导入两个新函数，`Textbox`和`rectangle`。我们将使用它们创建一个搜索框，用户可以在其中搜索他们喜欢的艺术家。

最后，我们导入在前几节中实现的`Menu`类和`DataManager`。

让我们开始实现一些辅助函数；第一个是`show_search_screen`：

```py
def show_search_screen(stdscr):
    curses.curs_set(1)
    stdscr.addstr(1, 2, "Artist name: (Ctrl-G to search)")

    editwin = curses.newwin(1, 40, 3, 3)
    rectangle(stdscr, 2, 2, 4, 44)
    stdscr.refresh()

    box = Textbox(editwin)
    box.edit()

    criteria = box.gather()
    return criteria
```

它以窗口实例作为参数，这样我们就可以在屏幕上打印文本并添加我们的文本框。

`curses.curs_set`函数用于打开和关闭光标；当设置为`1`时，光标将在屏幕上可见。我们希望在搜索屏幕上这样做，以便用户知道可以从哪里开始输入搜索条件。然后，我们打印帮助文本，以便用户知道应输入艺术家的名称；最后，他们可以按*Ctrl* + *G*或*Enter*执行搜索。

创建文本框时，我们创建一个新的小窗口，高度为`1`，宽度为`40`，并且它从终端屏幕的第`3`行，第`3`列开始。之后，我们使用`rectangle`函数在新窗口周围绘制一个矩形，并刷新屏幕以使我们所做的更改生效。

然后，我们创建`Textbox`对象，传递我们刚刚创建的窗口，并调用`edit`方法，它将设置框为文本框并进入编辑模式。这将`停止`应用程序，并允许用户在文本框中输入一些文本；当用户点击*Ctrl* + *G*或*Enter*时，它将退出。

当用户完成编辑文本后，我们调用`gather`方法，它将收集用户输入的数据并将其分配给`criteria`变量，最后返回`criteria`。

我们还需要一个函数来轻松清理屏幕，让我们创建另一个名为`clean_screen`的函数：

```py
def clear_screen(stdscr):
    stdscr.clear()
    stdscr.refresh()
```

太好了！现在，我们可以开始应用程序的主入口，并创建一个名为main的函数，内容如下：

```py
def main(stdscr):

    curses.cbreak()
    curses.noecho()
    stdscr.keypad(True)

    _data_manager = DataManager()

    criteria = show_search_screen(stdscr)

    height, width = stdscr.getmaxyx()

    albums_panel = Menu('List of albums for the selected artist',
                        (height, width, 0, 0))

    tracks_panel = Menu('List of tracks for the selected album',
                        (height, width, 0, 0))

    artist = _data_manager.search_artist(criteria)

    albums = _data_manager.get_artist_albums(artist['id'])

    clear_screen(stdscr)

    albums_panel.items = albums
    albums_panel.init()
    albums_panel.update()
    albums_panel.show()

    current_panel = albums_panel

    is_running = True

    while is_running:
        curses.doupdate()
        curses.panel.update_panels()

        key = stdscr.getch()

        action = current_panel.handle_events(key)

        if action is not None:
            action_result = action()
            if current_panel == albums_panel and action_result is 
            not None:
                _id, uri = action_result
                tracks = _data_manager.get_album_tracklist(_id)
                current_panel.hide()
                current_panel = tracks_panel
                current_panel.items = tracks
                current_panel.init()
                current_panel.show()
            elif current_panel == tracks_panel and action_result is  
            not None:
                _id, uri = action_result
                _data_manager.play(uri)

        if key == curses.KEY_F2:
            current_panel.hide()
            criteria = show_search_screen(stdscr)
            artist = _data_manager.search_artist(criteria)
            albums = _data_manager.get_artist_albums(artist['id'])

            clear_screen(stdscr)
            current_panel = albums_panel
            current_panel.items = albums
            current_panel.init()
            current_panel.show()

        if key == ord('q') or key == ord('Q'):
            is_running = False

        current_panel.update()

try:
    wrapper(main)
except KeyboardInterrupt:
    print('Thanks for using this app, bye!')
```

让我们将其分解为其组成部分：

```py
curses.cbreak()
curses.noecho()
stdscr.keypad(True)
```

在这里，我们进行一些初始化。通常，curses不会立即注册按键。当按键被输入时，这称为缓冲模式；用户必须输入一些内容，然后按*Enter*。在我们的应用程序中，我们不希望出现这种行为；我们希望按键在用户输入后立即注册。这就是`cbreak`的作用；它关闭curses的缓冲模式。

我们还使用`noecho`函数来读取按键并控制何时在屏幕上显示它们。

我们做的最后一个curses设置是打开键盘，这样curses将负责读取和处理按键，并返回表示已按下的键的常量值。这比尝试自己处理并测试键码数字要干净得多，更易于阅读。

我们创建`DataManager`类的实例，以便获取我们需要在菜单上显示的数据并执行身份验证：

```py
_data_manager = DataManager()
```

现在，我们创建搜索对话框：

```py
criteria = show_search_screen(stdscr)
```

我们调用`show_search_screen`函数，传递窗口的实例；它将在屏幕上呈现搜索字段并将结果返回给我们。当用户输入完成时，用户输入将存储在`criteria`变量中。

在获取条件后，我们调用`get_artist_albums`，它将首先搜索艺术家，然后获取艺术家专辑列表并返回`MenuItem`对象的列表。

当专辑列表返回时，我们可以创建其他带有菜单的面板：

```py
height, width = stdscr.getmaxyx()

albums_panel = Menu('List of albums for the selected artist',
                    (height, width, 0, 0))

tracks_panel = Menu('List of tracks for the selected album',
                    (height, width, 0, 0))

artist = _data_manager.search_artist(criteria)

albums = _data_manager.get_artist_albums(artist['id'])

clear_screen(stdscr)
```

在这里，我们获取主窗口的高度和宽度，以便我们可以创建具有相同尺寸的面板。`albums_panel`将显示专辑，`tracks_panel`将显示曲目；如前所述，它将具有与主窗口相同的尺寸，并且两个面板将从第`0`行，第`0`列开始。

之后，我们调用`clear_screen`准备窗口以渲染带有专辑的菜单窗口：

```py
albums_panel.items = albums
albums_panel.init()
albums_panel.update()
albums_panel.show()

current_panel = albums_panel

is_running = True
```

我们首先使用专辑搜索结果设置项目的属性。我们还在面板上调用`init`，这将在内部运行`_initialize_items`，格式化标签并设置当前选定的项目。我们还调用`update`方法，这将实际打印窗口中的菜单项；最后，我们展示如何将面板设置为可见。

我们还定义了`current_panel`变量，它将保存当前在终端上显示的面板的实例。

`is_running`标志设置为`True`，并将在应用程序的主循环中使用。当我们想要停止应用程序的执行时，我们将其设置为`False`。

现在，我们进入应用程序的主循环：

```py
while is_running:
    curses.doupdate()
    curses.panel.update_panels()

    key = stdscr.getch()

    action = current_panel.handle_events(key)
```

首先，我们调用`doupdate`和`update_panels`：

+   `doupdate`：Curses保留两个表示物理屏幕（在终端屏幕上看到的屏幕）和虚拟屏幕（保持下一个更新的屏幕）的数据结构。`doupdate`更新物理屏幕，使其与虚拟屏幕匹配。

+   `update_panels`：在面板堆栈中的更改后更新虚拟屏幕，例如隐藏、显示面板等。

更新屏幕后，我们使用`getch`函数等待按键按下，并将按下的键值分配给`key`变量。然后将`key`变量传递给当前面板的`handle_events`方法。

如果您还记得`Menu`类中`handle_events`的实现，它看起来像这样：

```py
def handle_events(self, key):
    if key == curses.KEY_UP:
        self.previous()
    elif key == curses.KEY_DOWN:
        self.next()
    elif key == curses.KEY_ENTER or key == NEW_LINE or key ==  
    CARRIAGE_RETURN:
    selected_item = self.get_selected()
    return selected_item.action
```

它处理`KEY_DOWN`，`KEY_UP`和`KEY_ENTER`。如果键是`KEY_UP`或`KEY_DOWN`，它将只更新菜单中的位置并设置新选择的项目，这将在下一个循环交互中更新在屏幕上。如果键是`KEY_ENTER`，我们获取所选项目并返回其操作函数。

请记住，对于两个面板，它将返回一个函数，当执行时，将返回包含项目ID和项目URI的元组。

接下来，我们处理返回的操作：

```py
if action is not None:
    action_result = action()
    if current_panel == albums_panel and action_result is not None:
        _id, uri = action_result
        tracks = _data_manager.get_album_tracklist(_id)
        current_panel.hide()
        current_panel = tracks_panel
        current_panel.items = tracks
        current_panel.init()
        current_panel.show()
    elif current_panel == tracks_panel and action_result is not 
    None:
        _id, uri = action_result
        _data_manager.play(uri)
```

如果当前面板的`handle_events`方法返回一个可调用的`action`，我们执行它并获取结果。然后，我们检查活动面板是否是第一个面板（带有专辑）。在这种情况下，我们需要获取所选专辑的曲目列表，因此我们在`DataManager`实例中调用`get_album_tracklist`。

我们隐藏`current_panel`，将当前面板切换到第二个面板（曲目面板），使用曲目列表设置项目属性，调用init方法使项目正确格式化并设置列表中的第一个项目为选定项目，最后我们调用`show`以便曲目面板可见。

在当前面板是`tracks_panel`的情况下，我们获取操作结果并在`DataManager`上调用play，传递曲目URI。它将请求在Spotify上活跃的设备上播放所选的曲目。

现在，我们希望有一种方法返回到搜索屏幕。当用户按下*F12*功能键时，我们这样做：

```py
if key == curses.KEY_F2:
    current_panel.hide()
    criteria = show_search_screen(stdscr)
    artist = _data_manager.search_by_artist_name(criteria)
    albums = _data_manager.get_artist_albums(artist['id'])

    clear_screen(stdscr)
    current_panel = albums_panel
    current_panel.items = albums
    current_panel.init()
    current_panel.show()
```

对于上面的`if`语句，测试用户是否按下了*F12*功能键；在这种情况下，我们希望返回到搜索屏幕，以便用户可以搜索新的艺术家。当按下*F12*键时，我们隐藏当前面板。然后，我们调用`show_search_screen`函数，以便呈现搜索屏幕，并且文本框将进入编辑模式，等待用户的输入。

当用户输入完成并按下*Ctrl*+ *G*或*Enter*时，我们搜索艺术家。然后，我们获取艺术家的专辑，并显示带有专辑列表的面板。

我们想要处理的最后一个事件是用户按下`q`或`Q`键，将`is_running`变量设置为`False`，应用程序关闭：

```py
if key == ord('q') or key == ord('Q'):
    is_running = False
```

最后，我们在当前面板上调用`update`，以便重新绘制项目以反映屏幕上的更改：

```py
current_panel.update()
```

在主函数之外，我们有代码片段，其中我们实际执行`main`函数：

```py
try:
    wrapper(main)
except KeyboardInterrupt:
    print('Thanks for using this app, bye!')
```

我们用`try` catch包围它，所以如果用户按下*Ctrl* + *C*，将会引发`KeyboardInterrupt`异常，我们只需优雅地完成应用程序，而不会在屏幕上抛出异常。

我们都完成了！让我们试试吧！

打开终端并输入命令—`python app.py`。

您将看到的第一个屏幕是搜索屏幕：

![](assets/526e4ad4-49d1-401d-8c0c-d58064a728fd.png)

让我搜索一下我最喜欢的艺术家：

![](assets/fdc3a925-18ee-43ea-b8e1-65fe12d2ae54.png)

按下*Enter*或*Ctrl* + *G*后，您应该会看到专辑列表：

![](assets/131e336d-88eb-4ca7-a629-e2670ede056f.png)

在这里，您可以使用箭头键（*上*和*下*）来浏览专辑，并按*Enter*来选择一个专辑。然后，您将看到屏幕显示所选专辑的所有曲目：

![](assets/71f792f1-dc7c-4300-a96e-3983f7576669.png)

如果这个屏幕是一样的，您可以使用箭头键（*上*和*下*）来选择曲目，*Enter*将发送请求在您的Spotify活动设备上播放这首歌曲。

# 总结

在本章中，我们涵盖了很多内容；我们首先在Spotify上创建了一个应用程序，并学习了其开发者网站的使用方法。然后，我们学习了如何实现Spotify支持的两种认证流程：客户端凭据流程和授权流程。

在本章中，我们还实现了一个完整的模块包装器，其中包含了一些来自Spotify的REST API的功能。

然后，我们实现了一个简单的终端客户端，用户可以在其中搜索艺术家，浏览艺术家的专辑和曲目，最后在用户的活动设备上播放一首歌曲，这可以是计算机、手机，甚至是视频游戏主机。

在下一章中，我们将创建一个桌面应用程序，显示通过Twitter标签的投票数。
