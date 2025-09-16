# 第三章：2 组织项目

这是第 0 天。你手头有一个项目。你充满激情，准备构建一个新的 Web 应用程序。你的脑海中充满了想法，你的手指迫不及待地想要开始敲击键盘。是时候坐下来开始编码了！

或者是吗？在我们头脑中关于我们想要构建的想法开始形成时，立即开始构建应用程序是很诱人的。在这样做之前，我们应该考虑如何为成功做好准备。有一个坚实的基础将使这个过程更容易，减少错误，并产生更干净的应用程序。

开始任何 Python Web 应用程序项目的三个基础是：

+   你的 IDE/代码编辑器

+   运行你的开发应用程序的环境

+   一个项目应用程序结构

这三个元素考虑了很多个人喜好。有如此多的优秀工具和方法。没有任何一本书能够涵盖所有这些。如果你是一个经验丰富的开发者并且已经有一套偏好：太好了，继续前进并跳到下一章。

在本章中，我们将探索一些现代选项，帮助你快速启动。重点将放在第二个基础（环境）和第三个基础（应用程序结构）。我们跳过第一个基础，并假设你正在使用自己选择的现代 IDE。Python 世界中的流行选择包括 VS Code、PyCharm 和 Sublime Text。如果你没有使用这些或类似的产品，去查找并找到一个适合你的。

在我们设置好环境后，我们将探讨在 Sanic 中实现的一些模式，这将有助于定义你的应用程序架构。这不是一本软件架构的书。我强烈建议你学习像“领域驱动设计”和“清洁架构”这样的方法。这本书更多地关注在 Sanic 中构建 Web 应用程序的实用方面和决策，所以请根据需要自由调整模式。

在本章中，我们将探讨以下主题：

+   设置环境和目录

+   有效使用蓝图

+   整合所有元素

+   运行我们的应用程序

## 技术要求

在我们开始之前，我们将假设你已经在你的电脑上设置了以下内容：

+   现代 Python 安装（Python 3.7 或更高版本）

+   终端（以及如何执行程序的基本知识）

+   IDE（如上所述）

## 设置环境和目录

当你开始任何项目时，你采取的前几步将对整个项目产生重大影响。无论你是开始一个可能持续多年的项目，还是只需要几个小时就能完成的项目，这些早期的决定将塑造你和他人如何在这个项目上工作。但是，尽管这些选择很重要，不要陷入认为你需要找到*完美*解决方案的陷阱。没有单一的正确方式来设置环境或项目目录。记住我们之前章节的讨论：我们想要做出适合当前项目的选择。

### 环境

对于 Python 开发来说，一个良好的实践是将它的运行环境与其他项目隔离开来。这通常是通过虚拟环境来实现的。在最基本的理解中，虚拟环境是一个工具，它允许你在隔离的环境中安装 Python 依赖项。这很重要，因为当我们开始开发我们的应用程序时，我们可以控制所使用的需求和依赖项。如果没有它，我们可能会错误地运行我们的应用程序，导致其他项目的需求渗透到应用程序中，造成错误和意外的行为。

在 Python 开发世界中，虚拟环境的使用是如此基础，以至于在创建 Python 脚本或应用程序时，它已经成为预期的“规范”。当你开始一个新的项目时，你应该做的第一步是为它创建一个新的虚拟环境。它们的替代方案是使用操作系统安装的 Python 运行你的应用程序。不要这样做。这可能在一开始是可行的，但最终，你将遇到冲突的需求、命名冲突或其他困难，所有这些困难都源于缺乏隔离。成为更好的 Python 开发者的第一步是使用虚拟环境，如果你还没有这样做的话。

熟悉 IDE 提供的将虚拟环境连接到你的不同工具也非常有帮助。这些工具通常包括代码补全等功能，并指导你开始使用依赖项的功能。

我们最终希望使用容器来运行我们的应用程序。能够在 Docker 容器中运行我们的应用程序将大大减少未来部署应用程序的复杂性。这将在*第九章，提高你的 Web 应用程序的最佳实践*中进一步讨论。然而，我也相信我们的应用程序应该能够在多个环境中运行（因此可测试）。即使我们打算将来使用 Docker，我们首先也需要在没有它的本地环境中运行我们的应用程序。当我们的应用程序不依赖于过于复杂的依赖项来运行时，调试变得更加容易。因此，让我们花些时间思考如何设置虚拟环境。

关于如何使用虚拟环境有许多优秀的教程和资源。还有许多工具被创建出来以帮助管理这个过程。虽然我个人是 `virtualenv` 加 `virtuanenvwrapper` 这种简单、可靠方法的粉丝，但许多人喜欢 `pipenv` 或 `poetry`。这些后者的工具旨在成为您运行环境的更“完整”封装。如果它们适合您，那很好。我们鼓励您花些时间看看什么与您的开发模式和需求产生共鸣。

我们现在暂时不考虑虚拟环境，而是简要探索 Python 中一个相对较新的模式。在 Python 3.8 中，Python 采用了 PEP 582 中的一种新模式，该模式将需求正式包含到项目内部的一个特殊 `__pypackages__` 目录中，从而在一个隔离的环境中。虽然这个概念与虚拟环境类似，但它的工作方式略有不同。

为了实现 `__pypackages__`，我们要求我们的虚构开发团队强制使用 `pdm`。这是一个相对较新的工具，它使得遵循现代 Python 开发的一些最新实践变得非常简单。如果您对此方法感兴趣，请花些时间阅读 PEP 582 ([`www.python.org/dev/peps/pep-0582/`](https://www.python.org/dev/peps/pep-0582/)) 并查看 `pdm` ([`pdm.fming.dev/`](https://pdm.fming.dev/))。

您可以通过使用 `pip` 来安装它：

```py
$ pip install --user pdm
```

请参考他们网站上的安装说明以获取更多详细信息：[`pdm.fming.dev/#installation`](https://pdm.fming.dev/#installation)。请特别注意像 shell 完成和 IDE 集成这样的有用功能。

现在我们继续进行设置：

1.  要开始，我们为我们的应用程序创建一个新的目录，并从这个目录运行以下命令，然后按照提示设置基本结构。

    ```py
    $ mkdir booktracker
    $ cd booktracker
    $ pdm init
    ```

1.  现在我们将安装 Sanic。

    ```py
    $ pdm add sanic
    ```

1.  我们现在可以访问 Sanic。为了确认我们确实在一个隔离的环境中，让我们快速跳入 Python REPL，并使用以下命令检查 Sanic 的位置：sanic.__file__。

    ```py
    $ python
    >>> import sanic
    >>> sanic.__file__
    '/path/to/booktracker/__pypackages__/3.9/lib/sanic/__init__.py'
    ```

### Sanic CLI

如第八章所述，关于如何部署和运行 Sanic 有许多考虑因素。除非我们特别关注这些替代方案之一，否则在本书中您可以假设我们正在使用 Sanic CLI 运行 Sanic。这将使用集成的 Sanic 网络服务器启动我们的应用程序。

首先，我们将检查我们正在运行哪个版本：

```py
$ sanic -v
Sanic 21.3.4
```

然后查看我们可以使用 CLI 的选项：

```py
$ sanic -h
usage: sanic [-h] [-H HOST] [-p PORT] [-u UNIX] [--cert CERT] [--key KEY] [-w WORKERS] [--debug] [--access-logs | --no-access-logs] [-v] module
                 Sanic
         Build Fast. Run Fast.
positional arguments:
  module                path to your Sanic app. Example: path.to.server:app
optional arguments:
  -h, --help            show this help message and exit
  -H HOST, --host HOST  host address [default 127.0.0.1]
  -p PORT, --port PORT  port to serve on [default 8000]
  -u UNIX, --unix UNIX  location of unix socket
  --cert CERT           location of certificate for SSL
  --key KEY             location of keyfile for SSL.
  -w WORKERS, --workers WORKERS
                        number of worker processes [default 1]
  --debug
  --access-logs         display access logs
  --no-access-logs      no display access logs
  -v, --version         show program's version number and exit
```

我们现在运行应用程序的标准形式将是：

```py
$ sanic src.server:app -p 7777 --debug --workers=2
```

在使用此命令的决定中考虑了哪些因素？让我们看看。

#### 为什么是 src.server:app？

首先，我们将从这个 `./booktracker` 目录运行这个命令。我们所有的代码都将嵌套在 `src` 目录中。

其次，这是一个相对标准的做法，我们的应用程序创建一个单一的`Sanic()`实例，并将其分配给一个名为`app`的变量：

```py
app = Sanic("BookTracker")
```

如果我们将它放入一个名为`app.py`的文件中，那么我们的模块和变量开始变得混乱。

```py
from app import app
```

上述导入语句，嗯，很丑。尽可能避免模块和该模块内容之间的命名冲突是有益的。

标准库中存在一个这样的坏例子。你有没有不小心做过这个：

```py
>>> import datetime
>>> datetime(2021, 1, 1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'module' object is not callable
```

哎呀，我们应该使用`from datetime import datetime`。我们希望最小化模块名称和属性的重复，并使我们的导入易于记忆和直观。

因此，我们将我们的全局`app`变量放入一个名为`server.py`的文件中。Sanic 会在你传入以下形式时查找我们的应用程序实例：`<module>:<variable>`。

#### 为什么是-p 7777？

我们当然可以选择任何任意的端口。许多网络服务器将使用端口`8000`，如果我们完全忽略它，那么这就是 Sanic 的默认端口。然而，正是因为它是标准的，我们想要选择另一个端口。通常，选择一个不太可能与你的机器上运行的其它端口冲突的端口是有益的。我们能够保留的常用端口越多，我们遇到冲突的可能性就越小。

#### 为什么是--debug？

在开发过程中，启用`DEBUG`模式可以提供：来自 Sanic 的更冗余的输出和一个自动重新加载的服务器。查看更多日志可能会有所帮助，但请确保在生产环境中关闭此功能。

自动重新加载功能特别有益，因为你可以在一个窗口中开始编写你的应用程序，然后在另一个终端会话中运行它。然后，每次你做出更改并保存应用程序时，Sanic 都会重新启动服务器，你的新代码立即可用于测试。

如果你想要自动重新加载，但又不想有所有额外的冗余信息，考虑使用`--auto-reload`代替。

#### 为什么是--workers=2？

这并不是一个罕见的问题，有人开始构建一个应用程序，然后意识到他们没有为横向扩展做准备，最终犯了一个错误。也许他们添加了全局状态，这些状态无法在单个进程之外访问。

```py
sessions = set()
@app.route("/login")
async def login(request):
    new_session = await create_session(request)
    sessions.add(new_session)
```

哎呀，现在那个人需要回去重新设计解决方案，如果他们想要扩展应用程序的话。这可能是一项代价高昂的任务。幸运的是，我们比那样聪明。

通过强制我们的开发模式从一开始就包括多个工作者，这将有助于我们在解决问题时提醒自己，我们的应用程序*必须*考虑到扩展。即使我们的最终部署不使用每个实例的多个 Sanic 工作者（例如，使用多个 Kubernetes pod，每个 pod 只有一个工作者实例–参见*第九章，提高你的 Web 应用程序的最佳实践*），这种持续的保护措施是保持最终目标与设计过程一致的有帮助方式。

### 目录结构

对于组织 Web 应用程序，你可以遵循许多不同的模式。可能最简单的是单文件`server.py`，其中包含所有的逻辑。由于显而易见的原因，这不是更大、更实际的项目的实际解决方案。所以我们将忽略这一点。

有哪些类型的解决方案？也许我们可以使用 Django 偏好的“apps”结构，将应用程序的离散部分组合成一个模块。或者，也许你更喜欢按类型分组，例如，将所有的视图控制器放在一起。在这里，我们不做任何关于什么更适合你需求的判断，但我们需要了解我们决策的一些后果。

在做决定时，你可能想了解一些常见的做法。这可能是一个查找以下模式的好机会：

+   **模型视图控制器**（**MVC**）

+   **模型视图视图模型**（**MVVM**）

+   **领域驱动设计**（**DDD**）

+   **清洁架构**（**CA**）

只是为了让你了解这些差异（或者至少是我对它们的解释），你可能会以以下方式之一来构建你的项目：

你可能会使用 MVC：

```py
./booktracker
├── controllers
│   ├── book.py
│   └── author.py
├── models
│   ├── book.py
│   └── author.py
├── views
│   ├── static
│   └── templates
└── services
```

或者你可能使用 DDD：

```py
./booktracker
└── domains
    ├── author
    │   ├── view.py
    │   └── model.py
    ├── book
    │   ├── view.py
    │   └── model.py
    └── universal
        └── middleware.py
```

在这本书中，我们将采用一种类似于混合方法的东西。应用这些理论结构的时间和地点是存在的。我敦促你们去学习它们。这些信息是有用的。但我们在这里是为了学习**如何**使用 Sanic 实际构建一个应用程序。

这是修改后的结构：

```py
./booktracker
├── blueprints
│   ├── author
│   │   ├── view.py
│   │   └── model.py
│   └── book
│       ├── view.py
│       └── model.py
├── middleware
│   └── thing.py
├── common
│   ├── utilities
│   └── base
└── server.py
```

让我们逐一分析这些，看看它们可能是什么样子，并理解这个应用程序设计的背后的思考过程。

#### ./blueprints

这可能让你觉得有些奇怪，因为最终这个目录看起来包含的不仅仅是蓝图。而且，你会是对的。查看树状结构，你会看到“blueprints”包括`view.py`和`model.py`。这个目录的目标是将你的应用程序分成逻辑组件或领域。它的工作方式与 Django 应用程序中的`apps`目录非常相似。如果你可以将某些结构或应用程序的一部分隔离为独立的实体，那么它可能应该有一个子目录。

本目录下的单个模块可能包含验证传入请求的模型、从数据库获取数据的实用工具以及带有附加路由处理器的蓝图。这样做可以将相关代码放在一起。

但为什么叫它`blueprints`？每个子目录将包含比单个`Blueprint`对象多得多的内容。重点是强化这样一个观点：这个目录中的所有内容都围绕这些离散组件之一。在 Sanic 中组织所谓的组件的标准方法是`Blueprint`（我们将在下一节中了解更多）。因此，每个子目录将有一个——而且只有一个——`Blueprint`对象。

另一个重要规则：`./bluprints`目录内的**任何内容**都不会引用我们的 Sanic 应用程序。这意味着在这个目录内禁止使用`Sanic.get_app()`和`from server import app`。

通常，将蓝图视为与您的 API 设计模式的一部分相对应是有帮助的。

+   `example.com/auth -> ./blueprints/auth`

+   `example.com/cake -> ./blueprints/cake`

+   `example.com/pie -> ./blueprints/pie`

+   `example.com/user -> ./blueprints/user`

#### ./middleware

此目录应包含任何旨在具有全局范围的中间件。

```py
@app.on_request
async def extract_user(request):
    user = await get_user_from_request(request)
    request.ctx.user = user
```

如本章后面和*第六章，响应周期之外*以及 Sanic 用户指南（[`sanicframework.org/en/guide/best-practices/blueprints.html#middleware`](https://sanicframework.org/en/guide/best-practices/blueprints.html#middleware)）中讨论的那样，中间件可以是全局的或附加到蓝图上。如果您需要将中间件应用于特定路由，基于蓝图的中间件可能是有意义的。在这种情况下，您应该将它们嵌套在适当的`./blueprints`目录中，而不是这里。

#### ./common

此模块旨在存储用于构建应用程序的类定义和函数。它是关于将跨越您的蓝图并在您的应用程序中普遍使用的所有内容。

> **提示**
> 
> 尝试根据您的需求扩展这里的目录结构。但是，尽量不要添加太多的顶级目录。如果您开始使文件夹杂乱无章，考虑一下您如何能够将目录嵌套在彼此内部。通常，您会发现这会导致更清晰的架构。过度嵌套也是一件事情。例如，如果您需要在应用程序代码中导航到十层深度，可能应该适当减少。

这仍然是第一天。您头脑中还有很多关于您想要构建的伟大想法。多亏了一些深思熟虑的预先规划，我们现在已经为在本地构建应用程序提供了一个有效的设置。在这个时候，我们应该知道应用程序如何在本地运行，以及项目通常是如何组织的。接下来我们将学习的是从应用程序结构到业务逻辑的过渡步骤。

## 有效使用蓝图

如果您已经知道什么是蓝图，那么请暂时想象一下您不知道。当我们构建应用程序并试图以逻辑和可维护的模式结构化我们的代码库时，我们意识到我们需要不断传递我们的`app`对象：

```py
from some.location import app
@app.route("/my/stuff")
async def stuff_handler(...):
    ...
@app.route("/my/profile")
async def profile_handler(...):
    ...
```

如果我们需要对我们的端点进行更改，这可能会变得非常繁琐。您可以想象一个场景，我们需要更新多个单独的文件，反复重复相同的更改。

可能更令人沮丧的是，我们可能会陷入一个存在循环导入的场景。

```py
# server.py
from user import *
app = Sanic(...)
# user.py
from server import app
@app.route("/user")
...
```

蓝图解决了这两个问题，并允许我们抽象出一些内容，以便组件可以独立存在。回到上面的例子，我们将端点的公共部分（`/my`）添加到`Blueprint`定义中。

```py
from sanic import Blueprint
bp = Blueprint("MyInfo", url_prefix="/my")
@bp.route("/stuff")
async def stuff_handler(...):
    ...
@bp.route("/profile")
async def profile_handler(...):
    ...
```

在这个例子中，我们能够将这些路由组合成一个单独的蓝图。重要的是，这允许我们将 URL 路径的公共部分（`/my`）拉到`Blueprint`中，这为我们提供了在将来进行更改的灵活性。

无论你决定如何组织你的文件结构，你可能应该始终使用蓝图。它们使组织更容易，甚至可以嵌套。我个人只会在我最简单的 Web 应用中使用`@app.route`。对于任何*真实*的项目，我总是将路由附加到蓝图上。

### 蓝图注册

仅创建我们的蓝图是不够的。Python 将无法知道它们的存在。我们需要导入我们的蓝图并将它们附加到我们的应用程序上。这是通过一个简单的注册方法完成的：`app.blueprint()`。

```py
# server.py
from user import bp as user_bp
app = Sanic(...)
app.blueprint(user_bp)
```

常见的“陷阱”是误解了`blueprint`的作用。以下这样的代码不会按预期工作：

```py
from sanic import Sanic, Blueprint
app = Sanic("MyApp")
bp = Blueprint("MyBrokenBp")
app.blueprint(bp)
@bp.route("/oops")
```

在我们注册蓝图的那一刻，所有附加到它的内容都将重新附加到应用程序上。这意味着在调用`app.blueprint()`之后添加到蓝图中的任何内容都不会应用。在上面的例子中，`/oops`将不会存在于应用程序中。因此，你应该尽可能晚地注册你的蓝图。

> **提示**
> 
> 我认为始终将蓝图变量命名为`bp`非常方便。当我打开一个文件时，我自动就知道`bp`代表什么。有些人可能觉得将变量命名得更有意义更有帮助：`user_bp`或`auth_bp`。对我来说，我更愿意在我总是查看的文件中保持一致性，并在导入时重命名它们：`from user import bp as user_bp`。

### 蓝图版本控制

在 API 设计中，版本控制是一个非常强大且常见的结构。让我们想象一下，我们正在开发我们的书籍 API，该 API 将被客户使用。他们已经创建了他们的集成，也许他们已经使用了一段时间的 API。

你有一些新的业务需求，或者你想要支持的新功能。完成这一点的唯一方法是改变特定端点的工作方式。但这将破坏用户的向后兼容性。这是一个困境。

API 设计者通常通过版本控制他们的路由来解决此问题。Sanic 通过向路由定义添加一个关键字参数或（可能更有用）一个蓝图来简化这一点。

你可以在用户指南中了解更多关于版本控制的信息（[`sanicframework.org/en/guide/advanced/versioning.html`](https://sanicframework.org/en/guide/advanced/versioning.html)），我们将在第三章“路由和接收 HTTP 请求”中更深入地讨论它。现在，我们将满足于知道我们的原始 API 设计需要修改，我们将在下一节中看到我们如何实现这一点。

### 分组蓝图

当你开始开发你的应用程序时，你可能会开始看到蓝图之间的相似性。就像我们看到了我们可以将路由的公共部分提取到 `Blueprint` 中一样，我们也可以将 `Blueprint` 的公共部分提取到 `BlueprintGroup` 中。这提供了相同的目的。

```py
from myinfo import bp as myinfo_bp
from somethingelse import bp as somethingelse_bp
from sanic import Blueprint
bp = Blueprint.group(myinfo_bp, somethingelse_bp, url_prefix="/api")
```

我们现在已经将 `/api` 添加到了 `myinfo` 和 `somethingelse` 中定义的每个路由路径的开头。

通过分组蓝图，我们正在压缩我们的逻辑，减少重复。在上面的例子中，通过给整个组添加前缀，我们不再需要管理单个端点或甚至蓝图。我们确实需要在设计端点和项目结构布局时牢记嵌套的可能性。

在上一节中，我们提到了使用版本来提供灵活升级 API 的简单路径。让我们回到我们的图书跟踪应用程序，看看这可能会是什么样子。如果你还记得，我们的应用程序看起来是这样的：

```py
./booktracker
└── blueprints
    ├── author
    │   └── view.py
    └── book
        └── view.py
```

以及 `view.py` 文件：

+   `# ./blueprints/book/view.py`

+   `bp = Blueprint("book", url_prefix="/book")`

+   `# ./blueprints/author/view.py`

+   `bp = Blueprint("author", url_prefix="/author")`

让我们设想这样一个场景：当我们的新业务需求是 `/v2/books` 路由时，这个 API 已经部署并为客户所使用。

我们将其添加到现有的架构中，它立即开始看起来丑陋且杂乱：

```py
└── blueprints
    ├── author
    │   └── view.py
    ├── book
    │   └── view.py
    └── book_v2
        └── view.py
```

让我们重构这个。我们不会改变 `./blueprints/author` 或 `./blueprints/book`，只是让它们嵌套得深一点。这部分应用程序已经构建好了，我们不想去动它。但是，既然我们已经从错误中吸取了教训，我们想要修改我们的 `/v2` 端点的策略，使其看起来像这样：

```py
└── blueprints
    ├── v1
    │   ├── author
    │   │   └── view.py
    │   ├── book
    │   │   └── view.py
    │   └── group.py
    └── v2
        ├── book
        │   └── view.py
        └── group.py
```

我们刚刚创建了一个新的文件，`group.py`：

```py
# ./blurprints/v2/group.py
from .book.view import bp as book_bp
from sanic import Blueprint
group = Blueprint.group(book_bp, version=2)
```

在构建复杂的 API 时，分组蓝图是一个强大的概念。它允许我们根据需要嵌套蓝图，同时为我们提供路由和组织控制。在这个例子中，注意我们是如何将 `version=2` 分配给组的。这意味着现在，这个组中每个与蓝图关联的路由都将有一个 `/v2` 路径前缀。

## 连接所有组件

正如我们所学的，创建一个实用的目录结构会导致可预测且易于导航的源代码。因为对我们开发者来说是可预测的，对计算机运行来说也是可预测的。也许我们可以利用这一点。

之前我们讨论了在尝试将应用程序从单文件结构扩展时经常遇到的一个问题：循环导入。我们可以用我们的蓝图很好地解决这个问题，但这仍然让我们对在应用程序级别可能想要附加的东西（如中间件、监听器和信号）感到困惑。现在让我们看看这些用例。

### 控制导入

通常我们更倾向于使用嵌套目录和文件将代码拆分成模块，这有助于我们逻辑上思考我们的代码，同时也便于导航。但这并非没有代价。当有两个相互依赖的模块时会发生什么？这将导致循环导入异常，我们的 Python 应用程序将崩溃。我们需要不仅考虑如何逻辑上组织我们的代码，还要考虑代码的不同部分如何在其他位置导入和使用。

考虑以下示例。首先，创建一个名为`./server.py`的文件，如下所示：

```py
app = Sanic(__file__)
```

其次，创建一个名为`./services/db.py`的第二个文件。

```py
app = Sanic.get_app()
@app.before_server_start
async def setup_db_pool(app, _):
    ...
```

这个例子说明了问题。当我们运行我们的应用程序时，我们需要在`Sanic.get_app()`之前运行`Sanic(__file__)`。但是，我们需要导入`.services.db`以便它能够附加到我们的应用程序上。哪个文件先评估？由于 Python 解释器将按顺序执行指令，我们需要确保在导入`db`模块之前实例化`Sanic()`对象。

这将有效：

```py
app = Sanic(__file__)
from .services.db import *
```

但是，它看起来有点丑陋，也不符合 Python 的风格。确实，如果你运行像`flake8`这样的工具，你将开始注意到你的环境也不太喜欢这种模式。它打破了在文件顶部放置导入的正常做法。在这里了解更多关于这种反模式的信息：[`www.flake8rules.com/rules/E402.html`](https://www.flake8rules.com/rules/E402.html)。

你可能觉得这并不重要，这是完全可以接受的。记住，我们做这件事是为了找到适合你应用程序的解决方案。然而，在我们做出决定之前，让我们看看一些其他的替代方案。

我们可以有一个单一的“启动”文件，这将是一个受控的导入顺序集合：

```py
# ./startup.py
from .server import app
from .services.db import *
```

现在，我们不想运行`sanic server:app`，而是想将我们的服务器指向新的`startup.py`。

```py
sanic startup:app
```

让我们继续寻找替代方案。

> **提示**
> 
> `Sanic.get_app()`结构是一个非常有用的模式，可以在不通过导入传递的情况下访问你的应用程序实例。这是朝着正确方向迈出的一个非常有帮助的步骤，你可以在用户指南中了解更多关于它的信息。[`sanicframework.org/en/guide/basics/app.html#app-registry`](https://sanicframework.org/en/guide/basics/app.html#app-registry)

### 工厂模式

我们将把应用程序创建移动到工厂模式。如果你来自 Flask，你可能熟悉这个概念，因为许多示例和教程都使用了类似的构造。在这里这样做的主要原因是我们希望为未来的良好开发实践设置我们的应用程序。这最终也将解决循环导入问题。在第九章后面，我们将讨论测试。如果没有好的工厂，测试将变得非常困难。

我们需要创建一个新的文件`./utilities/app_factory.py`，并重新做我们的`./server.py`。

```py
# ./utilities/app_factory.py
from typing import Optional, Sequence
from sanic import Sanic
from importlib import import_module
DEFAULT_BLUEPRINTS = [
    "src.blueprints.v1.book.view",
    "src.blueprints.v1.author.view",
    "src.blueprints.v2.group",
]
def create_app(
    init_blueprints: Optional[Sequence[str]] = None,
) -> Sanic:
    app = Sanic("BookTracker")
    if not init_blueprints:
        init_blueprints = DEFAULT_BLUEPRINTS
    for module_name in init_blueprints:
        module = import_module(module_name)
        app.blueprint(getattr(module, "bp"))
    return app
from .utilities.app_factory import create_app
app = create_app()
```

如你所见，我们的新工厂将创建`app`实例，并将一些蓝图附加到它上。我们特别允许工厂覆盖它将使用的蓝图。也许这并不必要，我们完全可以一直将它们硬编码。但我喜欢这种灵活性，并且发现它在以后的道路上很有帮助，当我想要开始测试我的应用程序时。

可能会跳出一个问题，那就是它要求我们的模块有一个全局的`bp`变量。虽然我提到这是我的标准做法，但它可能并不适用于所有场景。

### 自动发现

Sanic 用户指南在“如何做…”部分给了我们另一个想法。见 https://sanicframework.org/en/guide/how-to/autodiscovery.html。它建议我们创建一个`autodiscover`工具，该工具将为我们处理一些导入，并且还有自动附加蓝图的好处。记得我曾经说过我喜欢可预测的文件夹结构吗？我们即将利用这个模式。

让我们创建`./utilities/autodiscovery.py`：

```py
# ./utilities/autodiscovery.py
from importlib import import_module
from inspect import getmembers
from types import ModuleType
from typing import Union
from sanic.blueprints import Blueprint
def autodiscover(app, *module_names: Union[str, ModuleType]) -> None:
    mod = app.__module__
    blueprints = set()
    def _find_bps(module: ModuleType) -> None:
        nonlocal blueprints
        for _, member in getmembers(module):
            if isinstance(member, Blueprint):
                blueprints.add(member)
    for module in module_names:
        if isinstance(module, str):
            module = import_module(module, mod)
        _find_bps(module)
    for bp in blueprints:
        app.blueprint(bp)
```

此文件与用户指南中建议的内容非常相似（[`sanicframework.org/en/guide/how-to/autodiscovery.html#utility.py`](https://sanicframework.org/en/guide/how-to/autodiscovery.html#utility.py)）。值得注意的是，那里展示的代码中缺少递归的概念。如果你在用户指南中查找该函数，你会看到它包括递归搜索我们的源代码以查找`Blueprint`实例的能力。虽然很方便，但在我们正在构建的应用程序中，我们希望通过声明每个蓝图的位置来获得明确控制。再次引用 Tim Peters 的《Python 之禅》：

明确优于隐晦。

自动发现工具的作用是允许我们将位置传递给模块，并将导入它们的任务交给应用程序。加载模块后，它将检查任何蓝图。最后，它将自动将发现的蓝图注册到我们的应用程序实例中。

现在，我们的`server.py`看起来是这样的：

```py
from typing import Optional, Sequence
from sanic import Sanic
from .autodiscovery import autodiscover
DEFAULT_BLUEPRINTS = [
    "src.blueprints.v1.book.view",
    "src.blueprints.v1.author.view",
    "src.blueprints.v2.group",
]
def create_app(
    init_blueprints: Optional[Sequence[str]] = None,
) -> Sanic:
    app = Sanic("BookTracker")
    if not init_blueprints:
        init_blueprints = DEFAULT_BLUEPRINTS
    autodiscover(app, *init_blueprints)
    return app
```

> **提示**
> 
> 在这个例子中，我们使用字符串作为导入路径。我们同样可以在这里导入模块，并传递这些对象，因为`autodiscover`工具既适用于模块对象也适用于字符串。我们更喜欢字符串，因为它可以避免讨厌的循环导入异常。

另一个需要注意的事情是，这个自动发现工具可以用于包含中间件或监听器的模块。给出的例子仍然相当简单，不会涵盖所有用例。例如，我们应该如何处理深度嵌套的蓝图组？这是一个很好的机会让你进行实验，我强烈鼓励你花些时间玩转应用程序结构和自动发现工具，以找出最适合你的方法。

## 运行我们的应用程序

现在我们已经奠定了应用程序的基础，我们几乎准备好运行我们的服务器了。我们将在`server.py`中进行一个小改动，以包含一个在启动时运行的实用工具，显示已注册的路由。

```py
from .utilities.app_factory import create_app
from sanic.log import logger
app = create_app()
@app.main_process_start
def display_routes(app, _):
    logger.info("Registered routes:")
    for route in app.router.routes:
        logger.info(f"> /{route.path}")
```

你可以前往 GitHub 仓库[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/02`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/02)查看完整的源代码。

我们现在可以首次启动我们的应用程序。记住，这将是我们的模式：

```py
$ sanic src.server:app -p 7777 --debug --workers=2
```

我们应该看到类似这样的情况：

```py
[2021-05-30 11:34:54 +0300] [36571] [INFO] Goin' Fast @ http://127.0.0.1:7777
[2021-05-30 11:34:54 +0300] [36571] [INFO] Registered routes:
[2021-05-30 11:34:54 +0300] [36571] [INFO] > /v2/book
[2021-05-30 11:34:54 +0300] [36571] [INFO] > /book
[2021-05-30 11:34:54 +0300] [36571] [INFO] > /author
[2021-05-30 11:34:54 +0300] [36572] [INFO] Starting worker [36572]
[2021-05-30 11:34:54 +0300] [36573] [INFO] Starting worker [36573]
```

欢呼！

现在，到了最诱人的部分。我们的代码实际上做了什么？打开你最喜欢的网络浏览器并访问：[`127.0.0.1:7777/book`](http://127.0.0.1:7777/book)。现在可能看起来不多，但你应该能看到一些 JSON 数据。接下来，尝试访问`/author`和`/v2/book`。你现在应该能看到我们上面创建的内容。你可以随意玩转这些路由，向它们添加内容。每次你这样做，你都应该在浏览器中看到你的更改。

我们正式开始了网络应用程序开发的旅程。

## 摘要

我们已经审视了我们关于设置环境和项目组织所做的一些早期决策的重要影响。我们可以——并且应该——不断地调整我们的环境和应用程序以满足不断变化的需求。我们使用了`pdm`来利用运行服务器的新工具，在定义良好且隔离的环境中运行。

在我们的例子中，我们随后开始构建我们的应用程序。也许我们在添加`/book`路由时过于仓促，因为我们很快意识到我们需要端点以不同的方式执行。为了避免破坏现有用户的程序，我们简单地创建了一个新的蓝图组，这将是我们的 API `/v2`的开始。通过嵌套和分组蓝图，我们正在为未来的灵活性和开发可维护性设置应用程序。从现在开始，让我们尽可能坚持这个模式。

我们还考察了几种组织应用程序逻辑的替代方法。这些早期的决策将影响导入顺序并塑造应用程序的外观。我们决定采用一种工厂方法，这将在我们开始测试应用程序时对我们有所帮助。

基本应用结构确定后，我们将在下一章开始探讨网络服务器和框架最重要的方面：处理请求/响应周期。我们知道我们将使用蓝图，但现在是我们深入探究使用 Sanic 路由和处理器的可能性的时候了。在这一章中，我们已经对 API 版本化有所了解。在下一章中，我们还将更广泛地探讨路由，并尝试理解一些在 Web API 中设计应用程序逻辑的策略。
