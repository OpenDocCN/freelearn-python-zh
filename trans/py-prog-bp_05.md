# 使用微服务构建Web Messenger

在当今的应用程序开发世界中，微服务已成为设计和构建分布式系统的标准。像Netflix这样的公司开创了这一变革，并从拥有小型自治团队到设计轻松扩展的系统的方式，彻底改变了软件公司的运营方式。

在本章中，我将指导您完成创建两个微服务的过程，这两个微服务将共同工作，创建一个使用Redis作为数据存储的消息传递Web应用程序。消息在可配置的时间后会自动过期，因此在本章中，让我们称其为TempMessenger。

在本章中，我们将涵盖以下主题：

+   什么是Nameko？

+   创建您的第一个Nameko微服务

+   存储消息

+   Nameko依赖提供程序

+   保存消息

+   检索所有消息

+   在Web浏览器中显示消息

+   通过`POST`请求发送消息

+   浏览器轮询消息

# TempMessenger目标

在开始之前，让我们为我们的应用定义一些目标：

+   用户可以访问网站并发送消息

+   用户可以查看其他人发送的消息

+   消息在可配置的时间后会自动过期

为了实现这一点，我们将使用Nameko - 一个用于Python的微服务框架。

如果在本章的任何时候，您想要参考本章中的所有代码，请随时查看：[http://url.marcuspen.com/github-ppb](http://url.marcuspen.com/github-ppb)。

# 要求

为了参与本章，您的本地计算机将需要以下内容：

+   互联网连接

+   Docker - 如果您尚未安装Docker，请参阅官方文档：[http://url.marcuspen.com/docker-install](http://url.marcuspen.com/docker-install)

随着我们在本章中的进展，所有其他要求都将被安装。

本章中的所有说明都针对macOS或Debian/Ubuntu系统。但是，我已经注意到只使用跨平台依赖项。

在本章中，将会有代码块。不同类型的代码将有它们自己的前缀，如下所示：

`$`：在您的终端中执行，始终在您的虚拟环境中

`>>>`：在您的Nameko/Python shell中执行

无前缀：要在您的编辑器中使用的Python代码块

# 什么是Nameko？

Nameko是一个用于构建Python微服务的开源框架。使用Nameko，您可以创建使用**AMQP**（**高级消息队列协议**）通过**RPC**（**远程过程调用**）相互通信的微服务。

# RPC

RPC代表远程过程调用，我将用一个基于电影院预订系统的简短示例来简要解释这一点。在这个电影院预订系统中，有许多微服务，但我们将专注于预订服务，它负责管理预订，以及电子邮件服务，它负责发送电子邮件。预订服务和电子邮件服务都存在于不同的机器上，并且都不知道对方在哪里。在进行新预订时，预订服务需要向用户发送电子邮件确认，因此它对电子邮件服务进行远程过程调用，可能看起来像这样：

```py
def new_booking(self, user_id, film, time): 
    ... 
    self.email_service.send_confirmation(user_id, film, time) 
    ... 
```

请注意在上述代码中，预订服务如何进行调用，就好像它在本地执行代码一样？它不关心网络或协议，甚至不提供需要发送到哪个电子邮件地址的详细信息。对于预订服务来说，电子邮件地址和任何其他与电子邮件相关的概念都是无关紧要的！这使得预订服务能够遵守**单一责任原则**，这是由Robert C. Martin在他的文章*面向对象设计原则*（[http://url.marcuspen.com/bob-ood](http://url.marcuspen.com/bob-ood)）中介绍的一个术语，该原则规定：

“一个类应该只有一个变化的原因”

这个引用的范围也可以扩展到微服务，并且在开发它们时我们应该牢记这一点。这将使我们能够保持我们的微服务自包含和内聚。如果电影院决定更改其电子邮件提供商，那么唯一需要更改的服务应该是电子邮件服务，从而最小化所需的工作量，进而减少错误和可能的停机时间的风险。

然而，与其他技术（如REST）相比，RPC确实有其缺点，主要是很难看出调用是否是远程的。一个人可能在不知情的情况下进行不必要的远程调用，这可能很昂贵，因为它们需要通过网络并使用外部资源。因此，在使用RPC时，重要的是要使它们在视觉上有所不同。

# Nameko如何使用AMQP

AMQP代表高级消息队列协议，Nameko将其用作RPC的传输。当我们的Nameko服务相互进行RPC时，请求被放置在消息队列中，然后被目标服务消耗。Nameko服务使用工作程序来消耗和执行请求；当进行RPC时，目标服务将生成一个新的工作程序来执行任务。任务完成后，工作程序就会终止。由于可以有多个工作程序同时执行任务，Nameko可以扩展到其可用的工作程序数量。如果所有工作程序都被耗尽，那么消息将保留在队列中，直到有空闲的工作程序可用。

您还可以通过增加运行服务的实例数量来水平扩展Nameko。这被称为集群，也是*Nameko*名称的由来，因为Nameko蘑菇是成簇生长的。

Nameko还可以响应来自其他协议（如HTTP和Websockets）的请求。

# RabbitMQ

RabbitMQ被用作Nameko的消息代理，并允许其利用AMQP。在开始之前，您需要在您的机器上安装它；为此，我们将使用Docker，在所有主要操作系统上都可用。

对于那些不熟悉Docker的人来说，它允许我们在一个独立的、自包含的环境中运行我们的代码，称为容器。容器中包含了代码独立运行所需的一切。您还可以下载和运行预构建的容器，这就是我们将要运行RabbitMQ的方式。这样可以避免在本地机器上安装它，并最大程度地减少在不同平台（如macOS或Windows）上运行RabbitMQ时可能出现的问题。

如果您尚未安装Docker，请访问[http://url.marcuspen.com/docker-install](http://url.marcuspen.com/docker-install)获取详细的安装说明。本章的其余部分将假定您已经安装了Docker。

# 启动RabbitMQ容器

在您的终端中执行以下命令：

```py
$ docker run -d -p 5672:5672 -p 15672:15672 --name rabbitmq rabbitmq 
```

这将使用以下设置启动一个RabbitMQ容器：

+   `-d`：指定我们要在守护进程模式（后台进程）下运行容器。

+   `-p`：允许我们在容器上将端口`5672`和`15672`暴露到本地机器。这些端口是Nameko与RabbitMQ通信所需的。

+   `--name`：将容器名称设置为`rabbitmq`。

您可以通过执行以下命令来检查您的新RabbitMQ容器是否正在运行：

```py
$ docker ps
```

# 安装Python要求

对于这个项目，我将使用Python 3.6，这是我写作时的最新稳定版本的Python。我建议始终使用最新的稳定版本的Python，不仅可以获得新功能，还可以确保环境中始终应用最新的安全更新。

Pyenv是一种非常简单的安装和切换不同版本的Python的方法：[http://url.marcuspen.com/pyenv](http://url.marcuspen.com/pyenv)。

我还强烈建议使用virtualenv创建一个隔离的环境来安装我们的Python要求。在没有虚拟环境的情况下安装Python要求可能会导致与其他Python应用程序或更糟糕的操作系统产生意外的副作用！

要了解有关virtualenv及其安装方法的更多信息，请访问：[http://url.marcuspen.com/virtualenv](http://url.marcuspen.com/virtualenv)

通常，在处理Python包时，您会创建一个`requirements.txt`文件，填充它的要求，然后安装它。我想向您展示一种不同的方法，可以让您轻松地跟踪Python包的版本。

要开始，请在您的虚拟环境中安装`pip-tools`：

```py
pip install pip-tools 
```

现在创建一个名为`requirements`的新文件夹，并创建两个新文件：

```py
base.in 
test.in 
```

`base.in`文件将包含运行我们服务核心所需的要求，而`test.in`文件将包含运行我们测试所需的要求。将这些要求分开是很重要的，特别是在微服务架构中部署代码时。我们的本地机器可以安装测试包，但是部署版本的代码应尽可能简洁和轻量。

在`base.in`文件中，输入以下行：

```py
nameko 
```

在`test.in`文件中，输入以下行：

```py
pytest 
```

假设您在包含`requirements`文件夹的目录中，运行以下命令：

```py
pip-compile requirements/base.in 
pip-compile requirements/test.in 
```

这将生成两个文件，`base.txt`和`test.txt`。以下是`base.txt`的一个小样本：

```py
... 
nameko==2.8.3 
path.py==10.5             # via nameko 
pbr==3.1.1                # via mock 
pyyaml==3.12              # via nameko 
redis==2.10.6 
requests==2.18.4          # via nameko 
six==1.11.0               # via mock, nameko 
urllib3==1.22             # via requests 
... 
```

注意我们现在有一个文件，其中包含了Nameko的所有最新依赖和子依赖。它指定了所需的版本，还指出了每个子依赖被安装的原因。例如，`six`是由`nameko`和`mock`需要的。

这样可以非常容易地通过跟踪每个代码发布版本之间的版本更改来解决未来的升级问题。

在撰写本文时，Nameko当前版本为2.8.3，Pytest为3.4.0。如果有新版本可用，请随时使用，但如果在本书中遇到任何问题，请通过在您的`base.in`或`test.in`文件中附加版本号来恢复到这些版本：

```py
nameko==2.8.3 
```

要安装这些要求，只需运行：

```py
$ pip-sync requirements/base.txt requirements/test.txt 
```

`pip-sync`命令安装文件中指定的所有要求，同时删除环境中未指定的任何包。这是保持您的虚拟环境干净的好方法。或者，您也可以使用：

```py
$ pip install -r requirements/base.txt -r requirements/test.txt 
```

# 创建您的第一个Nameko微服务

让我们首先创建一个名为`temp_messenger`的新文件夹，并在其中放置一个名为`service.py`的新文件，其中包含以下代码：

```py
from nameko.rpc import rpc 

class KonnichiwaService: 

    name = 'konnichiwa_service' 

    @rpc 
    def konnichiwa(self): 
        return 'Konnichiwa!' 
```

我们首先通过从`nameko.rpc`导入`rpc`开始。这将允许我们使用`rpc`装饰器装饰我们的方法，并将它们公开为我们服务的入口点。入口点是Nameko服务中的任何方法，它作为我们服务的网关。

要创建一个Nameko服务，我们只需创建一个名为`KonnichiwaService`的新类，并为其分配一个`name`属性。`name`属性为其提供了一个命名空间；这将在我们尝试远程调用服务时使用。

我们在服务上编写了一个简单返回单词`Konnichiwa!`的方法。注意这个方法是用`rpc`装饰的。`konnichiwa`方法现在将通过RPC公开。

在测试这段代码之前，我们需要创建一个小的`config`文件，告诉Nameko在哪里访问RabbitMQ以及使用什么RPC交换。创建一个新文件`config.yaml`：

```py
AMQP_URI: 'pyamqp://guest:guest@localhost' 
rpc_exchange: 'nameko-rpc' 
```

这里的`AMQP_URI`配置对于使用先前给出的说明启动RabbitMQ容器的用户是正确的。如果您已调整了用户名、密码或位置，请确保您的更改在这里反映出来。

现在你应该有一个类似以下的目录结构：

```py
. 
├── config.yaml 
├── requirements 
│   ├── base.in 
│   ├── base.txt 
│   ├── test.in 
│   └── test.txt 
├── temp_messenger 
    └── service.py 
```

现在在您的终端中，在项目目录的根目录中，执行以下操作：

```py
$ nameko run temp_messenger.service --config config.yaml 
```

您应该会得到以下输出：

```py
starting services: konnichiwa_service 
Connected to amqp://guest:**@127.0.0.1:5672// 
```

# 对我们的服务进行调用

我们的微服务现在正在运行！为了进行我们自己的调用，我们可以启动一个集成了 Nameko 的 Python shell，以允许我们调用我们的入口点。要访问它，请打开一个新的终端窗口并执行以下操作：

```py
$ nameko shell 
```

这将为您提供一个 Python shell，可以进行远程过程调用。让我们试一试：

```py
>>> n.rpc.konnichiwa_service.konnichiwa() 
'Konnichiwa!' 
```

成功了！我们已成功调用了我们的 Konnichiwa 服务，并收到了一些输出。当我们在 Nameko shell 中执行此代码时，我们将一条消息放入队列，然后由我们的 `KonnichiwaService` 接收。然后它生成一个新的 worker 来执行 `konnichiwa` RPC 的工作。

# 单元测试 Nameko 微服务

根据文档，[http://url.marcuspen.com/nameko](http://url.marcuspen.com/nameko)，Nameko 是：

“一个用于 Python 的微服务框架，让服务开发人员专注于应用逻辑并鼓励可测试性。”

现在我们将专注于 Nameko 的可测试性部分；它提供了一些非常有用的工具，用于隔离和测试其服务。

创建一个新文件夹 `tests`，并在其中放入两个新文件，`__init__.py`（可以留空）和 `test_service.py`：

```py
from nameko.testing.services import worker_factory 
from temp_messenger.service import KonnichiwaService 

def test_konnichiwa(): 
    service = worker_factory(KonnichiwaService) 
    result = service.konnichiwa() 
    assert result == 'Konnichiwa!' 
```

在测试环境之外运行时，Nameko 会为每个被调用的入口点生成一个新的 worker。之前，当我们测试我们的 `konnichiwa` RPC 时，Konnichiwa 服务会监听 Rabbit 队列上的新消息。一旦它收到了 `konnichiwa` 入口点的新消息，它将生成一个新的 worker 来执行该方法，然后消失。

要了解更多关于 Nameko 服务解剖学的信息，请参阅：[http://url.marcuspen.com/nam-key](http://url.marcuspen.com/nam-key)。

对于我们的测试，Nameko 提供了一种通过 `woker_factory` 模拟的方法。正如您所看到的，我们的测试使用了 `worker_factory`，我们将我们的服务类 `KonnichiwaService` 传递给它。这将允许我们调用该服务上的任何入口点并访问结果。

要运行测试，只需从代码目录的根目录执行：

```py
pytest 
```

就是这样。测试套件现在应该通过了。尝试一下并尝试使其出错。

# 暴露 HTTP 入口点

现在我们将创建一个新的微服务，负责处理 HTTP 请求。首先，让我们在 `service.py` 文件中修改我们的导入：

```py
from nameko.rpc import rpc, RpcProxy 
from nameko.web.handlers import http 
```

在我们之前创建的 `KonnichiwaService` 下面，插入以下内容：

```py
class WebServer: 

    name = 'web_server' 
    konnichiwa_service = RpcProxy('konnichiwa_service') 

    @http('GET', '/') 
    def home(self, request): 
        return self.konnichiwa_service.konnichiwa() 
```

注意它如何遵循与 `KonnichiwaService` 类似的模式。它有一个 `name` 属性和一个用于将其公开为入口点的方法。在这种情况下，它使用 `http` 入口点进行装饰。我们在 `http` 装饰器内指定了它是一个 `GET` 请求以及该请求的位置 - 在这种情况下，是我们网站的根目录。

还有一个至关重要的区别：这个服务通过 `RpcProxy` 对象持有对 Konnichiwa 服务的引用。`RpcProxy` 允许我们通过 RPC 调用另一个 Nameko 服务。我们使用 `name` 属性来实例化它，这是我们之前在 `KonnichiwaService` 中指定的。

让我们试一试 - 只需使用之前的命令重新启动 Nameko（这是为了考虑代码的任何更改），然后在您选择的浏览器中转到 `http://localhost:8000/`：

![](assets/10d861cc-6456-4160-b51e-71bc67b982c9.png)

成功了！我们现在成功地创建了两个微服务——一个负责显示消息，一个负责提供 Web 请求。

# Nameko 微服务的集成测试

之前我们通过生成单个 worker 来测试隔离的服务。这对于单元测试来说很好，但不适用于集成测试。

Nameko 使我们能够在单个测试中测试多个服务协同工作的能力。看看下面的内容：

```py
def test_root_http(web_session, web_config, container_factory): 
    web_config['AMQP_URI'] = 'pyamqp://guest:guest@localhost' 

    web_server = container_factory(WebServer, web_config) 
    konnichiwa = container_factory(KonnichiwaService, web_config) 
    web_server.start() 
    konnichiwa.start() 

    result = web_session.get('/') 

    assert result.text == 'Konnichiwa!' 
```

正如您在前面的代码中所看到的，Nameko 还为我们提供了以下测试装置：

+   `web_session`：为我们提供了一个会话，用于向服务发出 HTTP 请求

+   `web_config`：允许我们访问服务的配置（在测试之外，这相当于`config.yaml`文件）

+   `container_factory`：这允许我们模拟整个服务而不仅仅是一个工作实例，这在集成测试时是必要的

由于这是运行实际服务，我们需要通过将其注入到`web_config`中来指定AMQP代理的位置。使用`container_factory`，我们创建两个容器：`web_server`和`konnichiwa`。然后启动两个容器。

然后只需使用`web_session`发出`GET`请求到我们网站的根目录，并检查结果是否符合我们的预期。

当我们继续阅读本章的其余部分时，我鼓励您为代码编写自己的测试，因为这不仅可以防止错误，还可以帮助巩固您对这个主题的知识。这也是一个尝试您自己的想法和对代码进行修改的好方法，因为它们可以快速告诉您是否有任何错误。

有关测试Nameko服务的更多信息，请参见：[http://url.marcuspen.com/nam-test](http://url.marcuspen.com/nam-test)。

# 存储消息

我们希望应用程序显示的消息需要是临时的。我们可以使用关系数据库，如PostgreSQL，但这意味着必须为像文本这样简单的东西设计和维护数据库。

# Redis简介

Redis是一个内存数据存储。整个数据集可以存储在内存中，使读写速度比关系数据库快得多，这对于不需要持久性的数据非常有用。此外，我们可以在不制作模式的情况下存储数据，如果我们不需要复杂的查询，这是可以接受的。在我们的情况下，我们只需要一个数据存储，它将允许我们存储消息，获取消息并使消息过期。Redis完全符合我们的用例！

# 启动Redis容器

在您的终端中，执行以下操作：

```py
$ docker run -d -p 6379:6379 --name redis redis
```

这将使用以下设置启动一个Redis容器：

+   `-d`：指定我们要在守护程序模式（后台进程）中运行容器。

+   `-p`：允许我们将容器上的端口`6379`暴露到本地机器上。这对于Nameko与Redis通信是必需的。

+   `--name`：将容器名称设置为`redis`。

您可以通过执行以下操作来检查新的Redis容器是否正在运行：

```py
$ docker ps
```

# 安装Python Redis客户端

您还需要安装Python Redis客户端，以便通过Python与Redis进行交互。为此，我建议您修改之前的`base.in`文件以包括`redis`并重新编译它以生成新的`base.txt`文件。或者，您可以运行`pip install redis`。

# 使用Redis

让我们简要地看一下对于TempMessenger对我们有用的Redis命令类型：

+   `SET`：设置给定键来保存给定的字符串。它还允许我们设置以秒或毫秒为单位的过期时间。

+   `GET`：获取存储在给定键处的数据的值。

+   `TTL`：以秒为单位获取给定键的生存时间。

+   `PTTL`：以毫秒为单位获取给定键的生存时间。

+   `KEYS`：返回数据存储中的所有键的列表。

要尝试它们，我们可以使用`redis-cli`，这是随Redis容器一起提供的程序。要访问它，首先通过在终端中执行以下操作登录到容器：

```py
docker exec -it redis /bin/bash
```

然后通过简单地运行以下内容在同一个终端窗口中访问`redis-cli`：

```py
redis-cli 
```

以下是如何使用`redis-cli`的一些示例；如果您对Redis不熟悉，我鼓励您自己尝试使用这些命令。

将一些数据`hello`设置为键`msg1`：

```py
127.0.0.1:6379> SET msg1 hello
OK
```

获取存储在键`msg1`处的数据：

```py
127.0.0.1:6379> GET msg1
"hello"
```

在键`msg2`处设置一些更多的数据`hi there`并检索它：

```py
127.0.0.1:6379> SET msg2 "hi there"
OK
127.0.0.1:6379> GET msg2
"hi there"
```

检索当前存储在Redis中的所有键：

```py
127.0.0.1:6379> KEYS *
1) "msg2"
2) "msg1"
```

在`msg3`上保存数据，过期时间为15秒：

```py
127.0.0.1:6379> SET msg3 "this message will die soon" EX 15
OK
```

获取`msg3`的生存时间（以秒为单位）：

```py
127.0.0.1:6379> TTL msg3
(integer) 10
```

以毫秒为单位获取`msg3`的生存时间：

```py
127.0.0.1:6379> PTTL msg3
(integer) 6080
```

在`msg3`过期之前检索：

```py
127.0.0.1:6379> GET msg3
"this message will die soon"
```

在`msg3`过期后检索：

```py
127.0.0.1:6379> GET msg3
(nil)
```

# Nameko依赖提供者

在构建微服务时，Nameko鼓励使用依赖提供程序与外部资源进行通信，如数据库、服务器或我们的应用程序所依赖的任何东西。通过使用依赖提供程序，您可以隐藏掉只对该依赖性特定的逻辑，使您的服务级代码保持干净，并且对于与这个外部资源进行交互的细节保持不可知。

通过这种方式构建我们的微服务，我们可以轻松地在其他服务中交换或重用依赖提供程序。

Nameko提供了一系列开源依赖提供程序，可以直接使用：[http://url.marcuspen.com/nam-ext](http://url.marcuspen.com/nam-ext)。

# 添加一个Redis依赖提供程序

由于Redis是我们应用程序的外部资源，我们将为它创建一个依赖提供程序。

# 设计客户端

首先，让我们在`temp_messenger`文件夹内创建一个名为`dependencies`的新文件夹。在里面，放置一个新文件`redis.py`。我们现在将创建一个Redis客户端，其中包含一个简单的方法，将根据一个键获取一条消息：

```py
from redis import StrictRedis 

class RedisClient: 

    def __init__(self, url): 
        self.redis = StrictRedis.from_url( 
            url, decode_responses=True 
        ) 
```

我们通过实现`__init__`方法来开始我们的代码，该方法创建我们的Redis客户端并将其分配给`self.redis`。`StrictRedis`可以接受许多可选参数，但是我们只指定了以下参数：

+   `url`：我们可以使用`StrictRedis`的`from_url`，而不是分别指定主机、端口和数据库号，这将允许我们使用单个字符串指定所有三个，如`redis://localhost:6379/0`。当将其存储在我们的`config.yaml`中时，这将更加方便。

+   `decode_responses`：这将自动将我们从Redis获取的数据转换为Unicode字符串。默认情况下，数据以字节形式检索。

现在，在同一个类中，让我们实现一个新的方法：

```py
def get_message(self, message_id): 
    message = self.redis.get(message_id) 

    if message is None: 
        raise RedisError( 
            'Message not found: {}'.format(message_id) 
        ) 

    return message 
```

在我们的新类之外，让我们还实现一个新的错误类：

```py
class RedisError(Exception): 
    pass 
```

在这里，我们有一个方法`get_message`，它接受一个`message_id`作为我们的Redis键。我们使用Redis客户端的`get`方法来检索具有给定键的消息。当从Redis检索值时，如果键不存在，它将简单地返回`None`。由于这个方法期望有一条消息，我们应该自己处理引发错误。在这种情况下，我们制作了一个简单的异常`RedisError`。

# 创建依赖提供程序

到目前为止，我们已经创建了一个具有单个方法的Redis客户端。现在我们需要创建一个Nameko依赖提供程序，以利用这个客户端与我们的服务一起使用。在同一个`redis.py`文件中，更新您的导入以包括以下内容：

```py
from nameko.extensions import DependencyProvider 
```

现在，让我们实现以下代码：

```py
class MessageStore(DependencyProvider): 

    def setup(self): 
        redis_url = self.container.config['REDIS_URL'] 
        self.client = RedisClient(redis_url) 

    def stop(self): 
        del self.client 

    def get_dependency(self, worker_ctx): 
        return self.client 
```

在上述代码中，您可以看到我们的新`MessageStore`类继承自`DependencyProvider`类。我们在新的MessageStore类中指定的方法将在我们的微服务生命周期的某些时刻被调用。

+   `setup`：这将在我们的Nameko服务启动之前调用。在这里，我们从`config.yaml`获取Redis URL，并使用我们之前制作的代码创建一个新的`RedisClient`。

+   `stop`：当我们的Nameko服务开始关闭时，这将被调用。

+   `get_dependency`：所有依赖提供程序都需要实现这个方法。当入口点触发时，Nameko创建一个worker，并将`get_dependency`的结果注入到服务中指定的每个依赖项的worker中。在我们的情况下，这意味着我们的worker都将可以访问`RedisClient`的一个实例。

Nameko提供了更多的方法来控制您的依赖提供程序在服务生命周期的不同时刻如何运行：[http://url.marcuspen.com/nam-writ](http://url.marcuspen.com/nam-writ)。

# 创建我们的消息服务

在我们的`service.py`中，我们现在可以利用我们的新的Redis依赖提供程序。让我们首先创建一个新的服务，它将替换我们之前的Konnichiwa服务。首先，我们需要在文件顶部更新我们的导入：

```py
from .dependencies.redis import MessageStore 
```

现在我们可以创建我们的新服务：

```py
class MessageService: 

    name = 'message_service' 
    message_store = MessageStore() 

    @rpc 
    def get_message(self, message_id): 
        return self.message_store.get_message(message_id) 
```

这与我们之前的服务类似；但是，这次我们正在指定一个新的类属性`message_store`。我们的RPC入口点`get_message`现在可以使用这个属性，并调用我们的`RedisClient`中的`get_message`，然后简单地返回结果。

我们本可以通过在我们的RPC入口点内创建一个新的Redis客户端并实现Redis的`GET`来完成所有这些。然而，通过创建一个依赖提供者，我们促进了可重用性，并隐藏了Redis在键不存在时返回`None`的不需要的行为。这只是一个小例子，说明了为什么依赖提供者非常擅长将我们的服务与外部依赖解耦。

# 将所有内容整合在一起

让我们尝试一下我们刚刚创建的代码。首先使用`redis-cli`将一个新的键值对保存到Redis中：

```py
127.0.0.1:6379> set msg1 "this is a test"
OK
```

现在启动我们的Nameko服务：

```py
$ nameko run temp_messenger.service --config config.yaml
```

我们现在可以使用`nameko shell`来远程调用我们的新`MessageService`：

```py
>>> n.rpc.message_service.get_message('msg1') 
'this is a test' 
```

如预期的那样，我们能够通过我们的`MessageService`入口点使用`redis-cli`检索到我们之前设置的消息。

现在让我们尝试获取一个不存在的消息：

```py
    >>> n.rpc.message_service.get_message('i_dont_exist')
    Traceback (most recent call last):
      File "<console>", line 1, in <module>
      File "/Users/marcuspen/.virtualenvs/temp_messenger/lib/python3.6/site-packages/nameko/rpc.py", line 393, in __call__
        return reply.result()
      File "/Users/marcuspen/.virtualenvs/temp_messenger/lib/python3.6/site-packages/nameko/rpc.py", line 379, in result
        raise deserialize(error)
    nameko.exceptions.RemoteError: RedisError Message not found: i_dont_exist
```

这并不是最漂亮的错误，有一些事情我们可以做来减少这个回溯，但最后一行说明了我们之前定义的异常，并清楚地显示了为什么该请求失败。

我们现在将继续保存消息。

# 保存消息

之前，我介绍了Redis的`SET`方法。这将允许我们将消息保存到Redis，但首先，我们需要在我们的依赖提供者中创建一个新的方法来处理这个问题。

我们可以简单地创建一个调用`redis.set(message_id, message)`的新方法，但是我们如何处理新的消息ID呢？如果我们期望用户为他们想要发送的每条消息输入一个新的消息ID，那将会有些麻烦，对吧？另一种方法是让消息服务在调用依赖提供者之前生成一个新的随机消息ID，但这样会使我们的服务充斥着依赖本身可以处理的逻辑。

我们将通过让依赖创建一个随机字符串来解决这个问题，以用作消息ID。

# 在我们的Redis客户端中添加保存消息的方法

在`redis.py`中，让我们修改我们的导入以包括`uuid4`：

```py
from uuid import uuid4 
```

`uuid4`为我们生成一个唯一的随机字符串，我们可以用它来作为我们消息的ID。

现在我们可以将我们的新的`save_message`方法添加到`RedisClient`中：

```py
    def save_message(self, message): 
        message_id = uuid4().hex 
        self.redis.set(message_id, message) 

        return message_id 
```

首先，我们使用`uuid4().hex`生成一个新的消息ID。`hex`属性将UUID作为一个32字符的十六进制字符串返回。然后我们将其用作键来保存消息并返回它。

# 添加一个保存消息的RPC

现在让我们创建一个RPC方法，用来调用我们的新客户端方法。在我们的`MessageService`中，添加以下方法：

```py
    @rpc 
    def save_message(self, message): 
        message_id = self.message_store.save_message(message) 
        return message_id 
```

这里没有什么花哨的，但请注意，向我们的服务添加新功能变得如此容易。我们正在将属于依赖的逻辑与我们的入口点分离，并同时使我们的代码可重用。如果我们将来创建的另一个RPC方法需要将消息保存到Redis中，我们可以轻松地这样做，而不必再次创建相同的代码。

让我们通过使用`nameko shell`来测试一下 - 记得重新启动Nameko服务以使更改生效！

```py
>>> n.rpc.message_service.save_message('Nameko is awesome!')
    'd18e3d8226cd458db2731af8b3b000d9'
```

这里返回的ID是随机的，与您在会话中获得的ID不同。

```py
>>>n.rpc.message_service.get_message
   ('d18e3d8226cd458db2731af8b3b000d9')
    'Nameko is awesome!'
```

正如您所看到的，我们已成功保存了一条消息，并使用返回的UUID检索了我们的消息。

这一切都很好，但是为了我们应用的目的，我们不希望用户必须提供消息UUID才能读取消息。让我们把这变得更实用一点，看看我们如何获取我们Redis存储中的所有消息。

# 检索所有消息

与我们之前的步骤类似，为了添加更多功能，我们需要在我们的Redis依赖中添加一个新的方法。这次，我们将创建一个方法，它将遍历Redis中的所有键，并以列表的形式返回相应的消息。

# 在我们的Redis客户端中添加一个获取所有消息的方法

让我们将以下内容添加到我们的`RedisClient`中：

```py
def get_all_messages(self): 
    return [ 
        { 
            'id': message_id, 
            'message': self.redis.get(message_id) 
        } 
        for message_id in self.redis.keys() 
    ] 
```

我们首先使用`self.redis.keys()`来收集存储在Redis中的所有键，这在我们的情况下是消息ID。然后，我们有一个列表推导式，它将遍历所有消息ID并为每个消息ID创建一个字典，其中包含消息ID本身和存储在Redis中的消息，使用`self.redis.get(message_id)`。

对于生产环境中的大型应用程序，不建议使用Redis的`KEYS`方法，因为这将阻塞服务器直到完成操作。更多信息，请参阅：[http://url.marcuspen.com/rediskeys](http://url.marcuspen.com/rediskeys)。

就我个人而言，我更喜欢在这里使用列表推导式来构建消息列表，但如果您在理解这种方法方面有困难，我建议将其编写为标准的for循环。

为了举例说明，可以查看以下代码，该代码是使用for循环构建的相同方法：

```py
def get_all_messages(self): 
    message_ids = self.redis.keys() 
    messages = [] 

    for message_id in message_ids: 
        message = self.redis.get(message_id) 
        messages.append( 
            {'id': message_id, 'message': message} 
        ) 

    return messages 
```

这两种方法都是完全相同的。你更喜欢哪个？我把这个选择留给你...

每当我编写列表或字典推导式时，我总是从测试函数或方法的输出开始。然后我用推导式编写我的代码并测试它以确保输出是正确的。然后，我将我的代码更改为for循环并确保测试仍然通过。之后，我会查看我的代码的两个版本，并决定哪个看起来最可读和干净。除非代码需要非常高效，我总是选择阅读良好的代码，即使这意味着多写几行。当以后需要阅读和维护该代码时，这种方法在长远来看是值得的！

我们现在有一种方法可以获取Redis中的所有消息。在上述代码中，我本可以简单地返回一个消息列表，而不涉及任何字典，只是消息的字符串值。但是，如果我们以后想要为每条消息添加更多数据呢？例如，一些元数据来表示消息创建的时间或消息到期的时间...我们以后会涉及到这部分！在这里为每条消息使用字典将使我们能够轻松地以后演变我们的数据结构。

我们现在可以考虑向我们的`MessageService`中添加一个新的RPC，以便我们可以获取所有消息。

# 添加获取所有消息的RPC

在我们的`MessageService`类中，只需添加：

```py
@rpc 
def get_all_messages(self): 
    messages = self.message_store.get_all_messages() 
    return messages 
```

我相信到现在为止，我可能不需要解释这里发生了什么！我们只是调用了我们之前在Redis依赖中制作的方法，并返回结果。

# 将所有内容放在一起

在您的虚拟环境中，使用`nameko shell`，我们现在可以测试这个功能。

```py
>>> n.rpc.message_service.save_message('Nameko is awesome!')
'bf87d4b3fefc49f39b7dd50e6d693ae8'
>>> n.rpc.message_service.save_message('Python is cool!')
'd996274c503b4b57ad5ee201fbcca1bd'
>>> n.rpc.message_service.save_message('To the foo bar!')
'69f99e5863604eedaf39cd45bfe8ef99'
>>> n.rpc.message_service.get_all_messages()
[{'id': 'd996274...', 'message': 'Python is cool!'},
{'id': 'bf87d4b...', 'message': 'Nameko is awesome!'},
{'id': '69f99e5...', 'message': 'To the foo bar!'}]
```

我们现在可以检索数据存储中的所有消息了。（出于空间和可读性考虑，我已经截断了消息ID。）

这里返回的消息存在一个问题-你能发现是什么吗？我们将消息放入Redis的顺序与我们再次取出它们的顺序不同。我们以后会回到这个问题，但现在让我们继续在我们的Web浏览器中显示这些消息。

# 在Web浏览器中显示消息

之前，我们添加了`WebServer`微服务来处理HTTP请求；现在我们将对其进行修改，以便当用户登陆根主页时，他们会看到我们数据存储中的所有消息。

其中一种方法是使用Jinja2等模板引擎。

# 添加Jinja2依赖提供程序

Jinja2是Python的模板引擎，与Django中的模板引擎非常相似。对于熟悉Django的人来说，使用它应该感觉非常熟悉。

在开始之前，您应该修改您的`base.in`文件，包括`jinja2`，重新编译您的要求并安装它们。或者，只需运行`pip install jinja2`。

# 创建模板渲染器

在Jinja2中生成简单的HTML模板需要以下三个步骤：

+   创建模板环境

+   指定模板

+   渲染模板

通过这三个步骤，重要的是要确定哪些部分在我们的应用程序运行时永远不会改变（或者至少极不可能改变）...以及哪些会改变。在我解释以下代码时，请记住这一点。

在您的依赖目录中，添加一个新文件`jinja2.py`，并从以下代码开始：

```py
from jinja2 import Environment, PackageLoader, select_autoescape 

class TemplateRenderer: 

    def __init__(self, package_name, template_dir): 
        self.template_env = Environment( 
            loader=PackageLoader(package_name, template_dir), 
            autoescape=select_autoescape(['html']) 
        ) 

    def render_home(self, messages): 
        template = self.template_env.get_template('home.html') 
        return template.render(messages=messages) 
```

在我们的`__init__`方法中，我们需要一个包名称和一个模板目录。有了这些，我们就可以创建模板环境。环境需要一个加载器，这只是一种能够从给定的包和目录加载我们的模板文件的方法。我们还指定我们要在我们的HTML文件上启用自动转义以确保安全。

然后我们创建了一个`render_home`方法，它将允许我们渲染我们的`home.html`模板。请注意我们如何使用`messages`来渲染我们的模板...稍后你会明白的！

你能看出我为什么以这种方式构建代码吗？由于`__init__`方法总是被执行，我把我们的模板环境的创建放在那里，因为这在我们的应用程序运行时几乎不会改变。

然而，我们要渲染的模板以及我们给该模板的变量总是会改变的，这取决于用户尝试访问的页面以及在那个特定时刻可用的数据。有了上述结构，为我们应用程序的每个网页添加一个新方法变得微不足道。

# 创建我们的主页模板

现在让我们看看我们模板所需的HTML。让我们首先在我们的依赖旁边创建一个名为`templates`的新目录。

在我们的新目录中，创建以下`home.html`文件：

```py
<!DOCTYPE html> 

<body> 
    {% if messages %} 
        {% for message in messages %} 
            <p>{{ message['message'] }}</p> 
        {% endfor %} 
    {% else %} 
        <p>No messages!</p> 
    {% endif %} 
</body> 
```

这个HTML并不花哨，模板逻辑也不复杂！如果你对Jinja2或Django模板不熟悉，那么你可能会觉得这个HTML看起来很奇怪，到处都是花括号。Jinja2使用这些花括号允许我们在模板中输入类似Python的语法。

在上面的例子中，我们首先使用`if`语句来查看是否有任何消息（`messages`的格式和结构将与我们之前制作的`get_all_messages` RPC返回的消息相同）。如果有，那么我们有一些更多的逻辑，包括一个for循环，它将迭代并显示我们`messages`列表中每个字典的`'message'`的值。

如果没有消息，那么我们将只显示`没有消息！`文本。

要了解更多关于Jinja2的信息，请访问：[http://url.marcuspen.com/jinja2](http://url.marcuspen.com/jinja2)。

# 创建依赖提供者

现在我们需要将我们的`TemplateRenderer`公开为Nameko依赖提供者。在我们之前创建的`jinja2.py`文件中，更新我们的导入以包括以下内容：

```py
from nameko.extensions import DependencyProvider 
```

然后添加以下代码：

```py
class Jinja2(DependencyProvider): 

    def setup(self): 
        self.template_renderer = TemplateRenderer( 
            'temp_messenger', 'templates' 
        ) 

    def get_dependency(self, worker_ctx): 
        return self.template_renderer 
```

这与我们之前的Redis依赖非常相似。我们指定了一个`setup`方法，用于创建我们的`TemplateRenderer`的实例，以及一个`get_dependency`方法，用于将其注入到worker中。

现在可以被我们的`WebServer`使用了。

# 创建HTML响应

现在我们可以在我们的`WebServer`中使用我们的新的Jinja2依赖项。首先，我们需要在`service.py`的导入中包含它：

```py
from .dependencies.jinja2 import Jinja2 
```

现在让我们修改我们的`WebServer`类如下：

```py
class WebServer: 

    name = 'web_server' 
    message_service = RpcProxy('message_service') 
    templates = Jinja2() 

    @http('GET', '/') 
    def home(self, request): 
        messages = self.message_service.get_all_messages() 
        rendered_template = self.templates.render_home(messages) 

        return rendered_template 
```

请注意，我们已经像之前在我们的`MessageService`中使用`message_store`一样，为它分配了一个新的属性`templates`。我们的HTTP入口现在与我们的`MessageService`通信，从Redis中检索所有消息，并使用它们来使用我们的新Jinja2依赖项创建一个渲染模板。然后我们返回结果。

# 将所有内容放在一起

重新启动您的Nameko服务，让我们在浏览器中尝试一下：

![](assets/cb3800d5-490c-4191-b33d-48391301b765.png)

它起作用了...有点！我们之前存储在Redis中的消息现在存在，这意味着我们模板中的逻辑正常运行，但我们也有来自`home.html`的所有HTML标签和缩进。

这是因为我们还没有为我们的HTTP响应指定任何头部，以表明它是HTML。为了做到这一点，让我们在`WebServer`类之外创建一个小的辅助函数，它将把我们的渲染模板转换为一个带有正确头部和状态码的响应。

在我们的`service.py`中，修改我们的导入以包括：

```py
from werkzeug.wrappers import Response 
```

然后在我们的类之外添加以下函数：

```py
def create_html_response(content): 
    headers = {'Content-Type': 'text/html'} 
    return Response(content, status=200, headers=headers) 
```

这个函数创建一个包含正确内容类型HTML的头部字典。然后我们创建并返回一个`Response`对象，其中包括HTTP状态码`200`，我们的头部和内容，而在我们的情况下，内容将是渲染的模板。

我们现在可以修改我们的HTTP入口点以使用我们的新的辅助函数：

```py
@http('GET', '/') 
def home(self, request): 
    messages = self.message_service.get_all_messages() 
    rendered_template = self.templates.render_home(messages) 
    html_response = create_html_response(rendered_template) 

    return html_response 
```

我们的`home` HTTP入口点现在使用`create_html_reponse`，给它渲染的模板，然后返回所做的响应。让我们在浏览器中再试一次：

![](assets/4fa4561f-41cd-471f-9798-b7b696ff5924.png)

现在你可以看到，我们的消息现在按我们的期望显示，没有找到任何HTML标签！尝试使用`redis-cli`中的`flushall`命令删除Redis中的所有数据，然后重新加载网页。会发生什么？

我们现在将继续发送消息。

# 通过POST请求发送消息

到目前为止，我们取得了很好的进展；我们有一个网站，它有能力显示我们数据存储中的所有消息，还有两个微服务。一个微服务处理我们消息的存储和检索，另一个充当我们用户的Web服务器。我们的`MessageService`已经有了保存消息的能力；让我们通过`POST`请求在我们的`WebServer`中暴露它。

# 添加发送消息的POST请求

在我们的`service.py`中，添加以下导入：

```py
import json 
```

现在在我们的`WebServer`类中添加以下内容：

```py
@http('POST', '/messages') 
def post_message(self, request): 
    data_as_text = request.get_data(as_text=True) 

    try: 
        data = json.loads(data_as_text) 
    except json.JSONDecodeError: 
        return 400, 'JSON payload expected' 

    try: 
        message = data['message'] 
    except KeyError: 
        return 400, 'No message given' 

    self.message_service.save_message(message) 

    return 204, '' 
```

有了我们的新的`POST`入口点，我们首先从请求中提取数据。我们指定参数`as_text=True`，因为否则我们会得到数据的字节形式。

一旦我们有了那些数据，我们就可以尝试将其从JSON加载到Python字典中。如果数据不是有效的JSON，那么这可能会在我们的服务中引发`JSONDecodeError`，因此最好处理得体，并返回一个`400`的错误请求状态码。如果没有这个异常处理，我们的服务将返回一个内部服务器错误，状态码为`500`。

现在数据以字典格式存在，我们可以获取其中的消息。同样，我们有一些防御性代码，它将处理任何缺少`'message'`键的情况，并返回另一个`400`。

然后我们继续使用我们之前在`MessageService`中创建的`save_message` RPC来保存消息。

有了这个，TempMessenger现在有了通过HTTP `POST`请求保存新消息的能力！如果你愿意，你可以使用curl或其他API客户端来测试这一点，就像这样：

```py
$ curl -d '{"message": "foo"}' -H "Content-Type: application/json" -X POST http://localhost:8000/messages
```

我们现在将更新我们的`home.html`模板，以包括使用这个新的`POST`请求的能力。

# 在jQuery中添加一个AJAX POST请求

在我们开始之前，让我说一下，写作时，我绝对不是JavaScript专家。我的专长更多地在于后端编程而不是前端。话虽如此，如果你在网页开发中工作超过10分钟，你就会知道试图避免JavaScript几乎是不可能的。在某个时候，我们可能会不得不涉足一些JavaScript来完成一些工作。

有了这个想法，请*不要被吓到*！

你即将阅读的代码是我仅仅通过阅读jQuery文档学到的，所以它非常简单。如果你对前端代码感到舒适，我相信可能有一百万种不同的，可能更好的方法来用JavaScript做到这一点，所以请根据自己的需要进行修改。

你首先需要在`<!DOCTYPE html>`之后添加以下内容：

```py
<head> 
  <script src="https://code.jquery.com/jquery-latest.js"></script> 
</head> 
```

这将在浏览器中下载并运行最新版本的jQuery。

在我们的`home.html`中，在闭合的`</body>`标签之前，添加以下内容：

```py
<form action="/messages" id="postMessage"> 
  <input type="text" name="message" placeholder="Post message"> 
  <input type="submit" value="Post"> 
</form> 
```

我们从一个简单的HTML开始，添加一个基本的表单。这只有一个文本输入和一个提交按钮。单独使用时，它将呈现一个文本框和一个提交按钮，但不会做任何事情。

现在让我们用一些jQuery JavaScript跟随这段代码：

```py
<script> 

$( "#postMessage" ).submit(function(event) { # ① 
  event.preventDefault(); # ② 

  var $form = $(this), 
    message = $form.find( "input[name='message']" ).val(), 
    url = $form.attr("action"); # ③ 

  $.ajax({ # ④ 
    type: 'POST', 
    url: url, 
    data: JSON.stringify({message: message}), # ⑤ 
    contentType: "application/json", # ⑥ 
    dataType: 'json', # ⑦ 
    success: function() {location.reload();} # ⑧ 
  }); 
}); 
</script> 
```

现在，这将为我们的提交按钮添加一些功能。让我们简要地介绍一下这里发生了什么：

1.  这将为我们的页面创建一个监听器，监听`postMessage`事件。

1.  我们还使用`event.preventDefault();`阻止了提交按钮的默认行为。在这种情况下，它将提交我们的表单，并尝试在`/messages?message=I%27m+a+new+message`上执行`GET`。

1.  一旦触发了，我们就可以在我们的表单中找到消息和URL。

1.  有了这些，我们就构建了我们的AJAX请求，这是一个POST请求。

1.  我们使用`JSON.stringify`将我们的有效负载转换为有效的JSON数据。

1.  还记得之前，当我们需要构建一个响应并提供头信息以说明我们的内容类型是`text/html`时吗？好吧，我们在我们的AJAX请求中也在做同样的事情，但这次我们的内容类型是`application/json`。

1.  我们将`datatype`设置为`json`。这告诉浏览器我们期望从服务器返回的数据类型。

1.  我们还注册了一个回调函数，如果请求成功，就重新加载网页。这将允许我们在页面上看到我们的新消息（和任何其他新消息），因为它将再次获取所有消息。这种强制页面重新加载并不是处理这个问题的最优雅方式，但现在可以这样做。

让我们重新启动Nameko并在浏览器中尝试一下：

![](assets/e58ffe52-a0b1-40b3-9733-27e950d20384.png)

只要您没有清除Redis中的数据（可以通过手动删除或简单地重新启动您的机器来完成），您应该仍然可以看到之前的旧消息。

输入消息后，点击“发布”按钮提交您的新消息：

![](assets/19fa394e-4f12-420b-916c-91e819e7edcf.png)

看起来好像成功了！我们的应用程序现在可以发送新消息了。我们现在将继续进行我们应用程序的最后一个要求，即在一定时间后过期消息。

# 在Redis中过期的消息

现在我们要实现应用程序的最后一个要求，即过期消息。由于我们使用Redis来存储消息，这变得非常简单。

让我们回顾一下我们Redis依赖中的`save_message`方法。Redis的`SET`有一些可选参数；我们在这里最感兴趣的是`ex`和`px`。两者都允许我们设置要保存的数据的过期时间，但有一个区别：`ex`是以秒为单位的，而`px`是以毫秒为单位的：

```py
def save_message(self, message): 
    message_id = uuid4().hex 
    self.redis.set(message_id, message, ex=10) 

    return message_id 
```

在上面的代码中，您可以看到我对代码所做的唯一修改是在`redis.set`方法中添加了`ex=10`；这将导致我们所有的消息在10秒后过期。现在重新启动您的Nameko服务并尝试一下。当您发送新消息后，等待10秒并刷新页面，它应该消失了。

**请注意**，如果在您进行此更改之前Redis中有任何消息，它们仍将存在，因为它们是在没有过期时间的情况下保存的。要删除它们，请使用`redis-cli`使用`flushall`命令删除Redis中的所有数据。

随意尝试设置过期时间，使用`ex`或`px`参数将其设置为您希望的任何时间。您可以将过期时间常量移到配置文件中，然后在启动Nameko时加载，这样可以使其更好，但现在这样就足够了。

# 排序消息

您很快会注意到我们应用程序的当前状态是，消息根本没有任何顺序。当您发送新消息时，它可能会被插入到消息线程的任何位置，这使得我们的应用程序非常不方便，至少可以这么说！

为了解决这个问题，我们将按剩余时间对消息进行排序。首先，我们将不得不修改我们的Redis依赖中的`get_all_messages`方法，以便为每条消息获取剩余时间：

```py
def get_all_messages(self): 
    return [ 
        { 
            'id': message_id, 
            'message': self.redis.get(message_id), 
            'expires_in': self.redis.pttl(message_id), 
        } 
        for message_id in self.redis.keys() 
    ] 
```

如前面的代码中所示，我们为每条消息添加了一个新的`expires_in`值。这使用了Redis的PTTL命令，该命令返回给定键的存活时间（以毫秒为单位）。或者，我们也可以使用Redis的TTL命令，该命令返回以秒为单位的存活时间，但我们希望尽可能精确，以使我们的排序更准确。

现在，当我们的`MessageService`调用`get_all_messages`时，它还将知道每条消息的存活时间。有了这个，我们可以创建一个新的辅助函数来对消息进行排序。

首先，将以下内容添加到我们的导入中：

```py
from operator import itemgetter 
```

在`MessageService`类之外，创建以下函数：

```py
def sort_messages_by_expiry(messages, reverse=False): 
    return sorted( 
        messages, 
        key=itemgetter('expires_in'), 
        reverse=reverse 
    ) 
```

这使用了Python内置的`sorted`函数，该函数能够从给定的可迭代对象返回一个排序后的列表；在我们的情况下，可迭代对象是`messages`。我们使用`key`来指定我们希望`messages`按照什么进行排序。由于我们希望`messages`按照`expires_in`进行排序，因此我们使用`itemgetter`来提取它以用作比较。我们给`sort_messages_by_expiry`函数添加了一个可选参数`reverse`，如果设置为`True`，则会使`sorted`以相反的顺序返回排序后的列表。

有了这个新的辅助函数，我们现在可以修改我们`MessageService`中的`get_all_messages` RPC：

```py
@rpc 
def get_all_messages(self): 
    messages = self.message_store.get_all_messages() 
    sorted_messages = sort_messages_by_expiry(messages) 
    return sorted_messages 
```

我们的应用现在将返回我们的消息，按照最新的消息在底部排序。如果您希望最新的消息在顶部，则只需将`sorted_messages`更改为：

```py
sorted_messages = sort_messages_by_expiry(messages, reverse=True) 
```

我们的应用现在符合我们之前指定的所有验收标准。我们有发送消息和获取现有消息的能力，并且它们在可配置的时间后都会过期。不太理想的一点是，我们依赖浏览器刷新来获取消息的最新状态。我们可以通过多种方式来解决这个问题，但我将演示解决这个问题的最简单的方法之一；通过轮询。

通过轮询，浏览器可以不断地向服务器发出请求，以获取最新的消息，而无需强制刷新页面。为了实现这一点，我们将不得不引入一些更多的JavaScript，但任何其他方法也都需要。

# 浏览器轮询消息

当浏览器进行轮询以获取最新消息时，我们的服务器应以JSON格式返回消息。为了实现这一点，我们需要创建一个新的HTTP端点，以JSON格式返回消息，而不使用Jinja2模板。我们首先构建一个新的辅助函数来创建一个JSON响应，设置正确的标头。

在我们的WebServer之外，创建以下函数：

```py
def create_json_response(content): 
    headers = {'Content-Type': 'application/json'} 
    json_data = json.dumps(content) 
    return Response(json_data, status=200, headers=headers) 
```

这类似于我们之前的`create_html_response`，但是这里将Content-Type设置为`'application/json'`，并将我们的数据转换为有效的JSON对象。

现在，在WebServer中，创建以下HTTP入口点：

```py
@http('GET', '/messages') 
def get_messages(self, request): 
    messages = self.message_service.get_all_messages() 
    return create_json_response(messages) 
```

这将调用我们的`get_all_messages` RPC，并将结果作为JSON响应返回给浏览器。请注意，我们在这里使用与我们在端点中使用的相同URL`/messages`，来发送新消息。这是RESTful的一个很好的例子。我们使用POST请求到`/messages`来创建新消息，我们使用GET请求到`/messages`来获取所有消息。

# 使用JavaScript进行轮询

为了使我们的消息在没有浏览器刷新的情况下自动更新，我们将创建两个JavaScript函数——`messagePoll`，用于获取最新消息，以及`updateMessages`，用于使用这些新消息更新HTML。

从我们的`home.html`中替换Jinja2 `if`块开始，该块遍历我们的消息列表，并使用以下行替换：

```py
<div id="messageContainer"></div> 
```

这将在稍后用于保存我们的jQuery函数生成的新消息列表。

在我们的`home.html`的`<script>`标签中，编写以下代码：

```py
function messagePoll() { 
  $.ajax({ 
    type: "GET", # ① 
    url: "/messages", 
    dataType: "json", 
    success: function(data) { # ② 
      updateMessages(data); 
    }, 
    timeout: 500, # ③ 
    complete: setTimeout(messagePoll, 1000), # ④ 
  }) 
} 
```

这是另一个AJAX请求，类似于我们之前发送新消息时所做的请求，但有一些不同之处：

1.  在这里，我们执行了一个`GET`请求到我们在`WebServer`中创建的新端点，而不是一个`POST`请求。

1.  如果成功，我们使用`success`回调来调用我们稍后将创建的`updateMessages`函数。

1.  将`timeout`设置为500毫秒 - 这是我们应该在放弃之前从服务器收到响应的时间。

1.  使用`complete`，它允许我们定义`success`或`error`回调完成后发生的事情 - 在这种情况下，我们设置它在1000毫秒后再次调用`poll`，使用`setTimeout`函数。

现在我们将创建`updateMessages`函数：

```py
function updateMessages(messages) { 
  var $messageContainer = $('#messageContainer'); # ① 
  var messageList = []; # ② 
  var emptyMessages = '<p>No messages!</p>'; # ③ 

  if (messages.length === 0) { # ④ 
    $messageContainer.html(emptyMessages); # 
  } else { 
    $.each(messages, function(index, value) { 
      var message = $(value.message).text() || value.message; 
      messageList.push('<p>' + message + '</p>'); # 
    }); 
    $messageContainer.html(messageList); # ⑤ 
  } 
} 
```

通过使用这个函数，我们可以替换Jinja2模板中生成消息列表的所有代码。让我们一步一步来：

1.  首先，我们获取HTML中的`messageContainer`，以便我们可以更新它。

1.  我们生成一个空的`messageList`数组。

1.  我们生成`emptyMessages`文本。

1.  我们检查消息的数量是否等于0：

1.  如果是，我们使用`.html()`将`messageContainer`的HTML替换为`"没有消息！"`。

1.  否则，对于`messages`中的每条消息，我们首先使用jQuery的内置`.text()`函数去除可能存在的任何HTML标签。然后我们将消息包装在`<p>`标签中，并使用`.push()`将它们附加到`messageList`中。

1.  最后，我们使用`.html()`将`messageContainer`的HTML替换为`messagesList`。

在*4b*点，重要的是要转义消息中可能存在的任何HTML标签，因为恶意用户可能发送一条恶意脚本作为消息，这将被每个使用该应用程序的人执行！

这绝不是解决不得不强制刷新浏览器以更新消息的问题的最佳方法，但对我来说，这是在本书中演示的最简单的方法之一。可能有更优雅的方法来实现轮询，如果你真的想要做到这一点，那么WebSockets绝对是你在这里的最佳选择。

# 总结

这样，我们就结束了编写TempMessenger应用程序的指南。如果你以前从未使用过Nameko或编写过微服务，我希望我已经为你提供了一个很好的基础，以便在保持服务小而简洁方面进行构建。

我们首先创建了一个具有单个RPC方法的服务，然后通过HTTP在另一个服务中使用它。然后我们看了一下我们如何使用允许我们生成工作者甚至服务本身的固定装置来测试Nameko服务。

我们引入了依赖提供程序，并创建了一个Redis客户端，具有获取单个消息的能力。然后，我们扩展了Redis依赖，增加了允许我们保存新消息、过期消息并以列表形式返回它们的方法。

我们看了如何使用Jinja2将HTML返回给浏览器，并创建了一个依赖提供程序。我们甚至看了一些JavaScript和JQuery，使我们能够从浏览器发出请求。

你可能已经注意到的一个主题是需要将依赖逻辑与服务代码分开。通过这样做，我们使我们的服务对只有该依赖特定的工作保持不可知。如果我们决定将Redis替换为MySQL数据库呢？在我们的代码中，只需创建一个新的MySQL依赖提供程序和映射到我们的`MessageService`期望的方法的新客户端方法。然后我们只需最小的更改，将Redis替换为MySQL。如果我们没有以这种方式编写代码，那么我们将不得不投入更多的时间和精力来对我们的服务进行更改。我们还会引入更多的错误可能性。

如果你熟悉其他Python框架，你现在应该看到Nameko如何让我们轻松创建可扩展的微服务，同时与像Django这样的框架相比，它给我们提供了更多的*不包括电池*的方法。当涉及编写专注于后端任务的小服务时，Nameko可能是一个完美的选择。

在下一章中，我们将使用PostgreSQL数据库来扩展TempMessenger，添加一个用户认证微服务。
