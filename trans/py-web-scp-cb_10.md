# 使用 Docker 创建爬虫微服务

在本章中，我们将涵盖：

+   安装 Docker

+   从 Docker Hub 安装 RabbitMQ 容器

+   运行一个 Docker 容器（RabbitMQ）

+   停止和删除容器和镜像

+   创建一个 API 容器

+   使用 Nameko 创建一个通用微服务

+   创建一个爬取微服务

+   创建一个爬虫容器

+   创建后端（ElasticCache）容器

+   使用 Docker Compose 组合和运行爬虫容器

# 介绍

在本章中，我们将学习如何将我们的爬虫容器化，使其准备好进入现实世界，开始为真正的、现代的、云启用的操作打包。这将涉及将爬虫的不同元素（API、爬虫、后端存储）打包为可以在本地或云中运行的 Docker 容器。我们还将研究将爬虫实现为可以独立扩展的微服务。

我们将主要关注使用 Docker 来创建我们的容器化爬虫。Docker 为我们提供了一种方便和简单的方式，将爬虫的各个组件（API、爬虫本身以及其他后端，如 Elasticsearch 和 RabbitMQ）打包为一个服务。通过使用 Docker 对这些组件进行容器化，我们可以轻松地在本地运行容器，编排组成服务的不同容器，还可以方便地发布到 Docker Hub。然后我们可以轻松地部署它们到云提供商，以在云中创建我们的爬虫。

关于 Docker（以及容器一般）的一大好处是，我们既可以轻松地安装预打包的容器，而不必费力地获取应用程序的安装程序并处理所有配置的麻烦。我们还可以将我们编写的软件打包到一个容器中，并在不必处理所有这些细节的情况下运行该容器。此外，我们还可以发布到私有或公共存储库以分享我们的软件。

Docker 真正伟大的地方在于容器在很大程度上是平台无关的。任何基于 Linux 的容器都可以在任何操作系统上运行，包括 Windows（它在虚拟化 Linux 时使用 VirtualBox，并且对 Windows 用户来说基本上是透明的）。因此，一个好处是任何基于 Linux 的 Docker 容器都可以在任何 Docker 支持的操作系统上运行。不再需要为应用程序创建多个操作系统版本了！

让我们学习一些 Docker 知识，并将我们的爬虫组件放入容器中。

# 安装 Docker

在这个教程中，我们将学习如何安装 Docker 并验证其是否正在运行。

# 准备工作

Docker 支持 Linux、macOS 和 Windows，因此它覆盖了主要平台。Docker 的安装过程因您使用的操作系统而异，甚至在不同的 Linux 发行版之间也有所不同。

Docker 网站对安装过程有很好的文档，因此本教程将快速浏览 macOS 上安装的重要要点。安装完成后，至少从 CLI 方面来看，Docker 的用户体验是相同的。

参考文献，Docker 的安装说明主页位于：[`docs.docker.com/engine/installation/`](https://docs.docker.com/engine/installation/)

# 如何做

我们将按照以下步骤进行：

1.  我们将使用一个名为 Docker 社区版的 Docker 变体，并在 macOS 上进行安装。在 macOS 的下载页面上，您将看到以下部分。点击稳定频道的下载，除非您感到勇敢并想使用 Edge 频道。

![](img/3fe42443-f9f8-4dc5-8679-221293a203fa.png)Docker 下载页面

1.  这将下载一个`Docker.dmg`文件。打开 DMG，您将看到以下窗口：

![](img/b2a7044a-5f06-4874-96c6-112496770357.png)Docker for Mac 安装程序窗口

1.  将*Moby*鲸鱼拖到您的应用程序文件夹中。然后打开`Docker.app`。您将被要求验证安装，因此输入密码，安装将完成。完成后，您将在状态栏中看到 Moby：

![](img/db37ccda-6c94-44dd-b327-f943fe3afbdb.png)Moby 工具栏图标

1.  点击 Moby 可以获得许多配置设置、状态和信息。我们将主要使用命令行工具。要验证命令行是否正常工作，请打开终端并输入命令 docker info。Docker 将为您提供有关其配置和状态的一些信息。

# 从 Docker Hub 安装 RabbitMQ 容器

可以从许多容器存储库获取预构建的容器。Docker 预先配置了与 Docker Hub 的连接，许多软件供应商和爱好者在那里发布一个或多个配置的容器。

在这个教程中，我们将安装 RabbitMQ，这将被我们在另一个教程中使用的另一个工具 Nameko 所使用，以作为我们的抓取微服务的消息总线。

# 准备工作

通常，安装 RabbitMQ 是一个相当简单的过程，但它确实需要几个安装程序：一个用于 Erlang，然后一个用于 RabbitMQ 本身。如果需要管理工具，比如基于 Web 的管理 GUI，那就是另一步（尽管是一个相当小的步骤）。通过使用 Docker，我们可以简单地获取所有这些预配置的容器。让我们去做吧。

# 如何做

我们按照以下步骤进行教程：

1.  可以使用`docker pull`命令获取容器。此命令将检查并查看本地是否已安装容器，如果没有，则为我们获取。从命令行尝试该命令，包括`--help`标志。您将得到以下信息，告诉您至少需要一个参数：容器的名称和可能的标签：

```py
$ docker pull --help

Usage: docker pull [OPTIONS] NAME[:TAG|@DIGEST]

Pull an image or a repository from a registry

Options:
  -a, --all-tags Download all tagged images in the repository
      --disable-content-trust Skip image verification (default true)
      --help Print usage
```

1.  我们将拉取`rabbitmq:3-management`容器。冒号前的部分是容器名称，第二部分是标签。标签通常代表容器的版本或特定配置。在这种情况下，我们希望获取带有 3-management 标签的 RabbitMQ 容器。这个标签意味着我们想要带有 RabbitMQ 版本 3 和管理工具安装的容器版本。

在我们这样做之前，您可能会想知道这是从哪里来的。它来自 Docker Hub（`hub.docker.com`），来自 RabbitMQ 存储库。该存储库的页面位于[`hub.docker.com/_/rabbitmq/`](https://hub.docker.com/_/rabbitmq/)，并且看起来像下面这样：

![](img/2210e62e-0f9d-4a21-b71c-41c37d0dd8c6.png)RabbitMQ 存储库页面请注意显示标签的部分，以及它具有 3-management 标签。如果您向下滚动，还会看到有关容器和标签的更多信息，以及它们的组成。

1.  现在让我们拉取这个容器。从终端发出以下命令：

```py
$docker pull rabbitmq:3-management
```

1.  Docker 将访问 Docker Hub 并开始下载。您将在类似以下的输出中看到这一过程，这可能会根据您的下载速度运行几分钟：

```py
3-management: Pulling from library/rabbitmq
e7bb522d92ff: Pull complete 
ad90649c4d84: Pull complete 
5a318b914d6c: Pull complete 
cedd60f70052: Pull complete 
f4ec28761801: Pull complete 
b8fa44aa9074: Pull complete 
e3b16d5314a0: Pull complete 
7d93dd9659c8: Pull complete 
356c2fc6e036: Pull complete 
3f52408394ed: Pull complete 
7c89a0fb0219: Pull complete 
1e37a15bd7aa: Pull complete 
9313c22c63d5: Pull complete 
c21bcdaa555d: Pull complete 
Digest: sha256:c7466443efc28846bb0829d0f212c1c32e2b03409996cee38be4402726c56a26 
Status: Downloaded newer image for rabbitmq:3-management 
```

恭喜！如果这是您第一次使用 Docker，您已经下载了您的第一个容器镜像。您可以使用 docker images 命令验证它是否已下载和安装。

```py
$ docker images 
REPOSITORY TAG IMAGE    ID           CREATED     SIZE 
rabbitmq   3-management 6cb6e2f951a8 10 days ago 151MB
```

# 运行 Docker 容器（RabbitMQ）

在这个教程中，我们将学习如何运行 docker 镜像，从而创建一个容器。

# 准备工作

我们将启动我们在上一个教程中下载的 RabbitMQ 容器镜像。这个过程代表了许多容器的运行方式，因此它是一个很好的例子。

# 如何做

我们按照以下步骤进行教程：

1.  到目前为止，我们已经下载了一个可以运行以创建实际容器的镜像。容器是使用特定参数实例化的镜像，这些参数需要配置容器中的软件。我们通过运行 docker run 并传递镜像名称/标签以及运行镜像所需的任何其他参数来运行容器（这些参数特定于镜像，通常可以在 Docker Hub 页面上找到镜像的参数）。

我们需要使用以下特定命令来运行 RabbitMQ 使用此镜像：

```py
$ docker run -d -p 15672:15672 -p 5672:5672 rabbitmq:3-management
094a138383764f487e5ad0dab45ff64c08fe8019e5b0da79cfb1c36abec69cc8
```

1.  `docker run`告诉 Docker 在容器中运行一个镜像。我们要运行的镜像在语句的末尾：`rabbitmq:3-management`。`-d`选项告诉 Docker 以分离模式运行容器，这意味着容器的输出不会路由到终端。这允许我们保留对终端的控制。`-p`选项将主机端口映射到容器端口。RabbitMQ 使用 5672 端口进行实际命令，15672 端口用于 Web UI。这将在您的实际操作系统上的相同端口映射到容器中运行的软件使用的端口。

大的十六进制值输出是容器的标识符。第一部分，094a13838376，是 Docker 创建的容器 ID（对于每个启动的容器都会有所不同）。

1.  我们可以使用 docker ps 来检查正在运行的容器，这会给我们每个容器的进程状态：

```py
$ docker ps
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
094a13838376 rabbitmq:3-management "docker-entrypoint..." 5 minutes ago Up 5 minutes 4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp dreamy_easley
```

我们可以看到容器 ID 和其他信息，例如它基于哪个镜像，它已经运行了多长时间，容器暴露了哪些端口，我们定义的端口映射，以及 Docker 为我们创建的友好名称，以便我们引用容器。

1.  检查是否正在运行的真正方法是打开浏览器，导航到`localhost:15672`，即 RabbitMQ 管理 UI 的 URL：

![](img/d8715fbc-5686-48c9-9f38-16e8c54aa325.png)RabbitMQ 管理 UI 登录页面

1.  该容器的默认用户名和密码是 guest:guest。输入这些值，您将看到管理 UI：

![](img/20179fcc-8e5c-467f-8421-49f1dd7b3e9c.png)管理 UI

# 还有更多...

这实际上是我们将在 RabbitMQ 中取得的进展。在以后的教程中，我们将使用 Nameko Python 微服务框架，它将在我们不知情的情况下透明地使用 RabbitMQ。我们首先需要确保它已安装并正在运行。

# 创建和运行 Elasticsearch 容器

当我们正在查看拉取容器镜像和启动容器时，让我们去运行一个 Elasticsearch 容器。

# 如何做

像大多数 Docker 一样，有很多不同版本的 Elasticsearch 容器可用。我们将使用 Elastic 自己的 Docker 存储库中提供的官方 Elasticsearch 镜像：

1.  要安装镜像，请输入以下内容：

```py
$docker pull docker.elastic.co/elasticsearch/elasticsearch:6.1.1
```

请注意，我们正在使用另一种指定要拉取的镜像的方式。由于这是在 Elastic 的 Docker 存储库上，我们包括了包含容器镜像 URL 的限定名称，而不仅仅是镜像名称。 :6.1.1 是标签，指定了该镜像的特定版本。

1.  在处理此过程时，您将看到一些输出，显示下载过程。完成后，您将看到几行让您知道已完成：

```py
Digest: sha256:9e6c7d3c370a17736c67b2ac503751702e35a1336724741d00ed9b3d00434fcb 
Status: Downloaded newer image for docker.elastic.co/elasticsearch/elasticsearch:6.1.1
```

1.  现在让我们检查 Docker 中是否有可用的镜像：

```py
$ docker images 
REPOSITORY TAG IMAGE ID CREATED SIZE 
rabbitmq 3-management 6cb6e2f951a8 12 days ago 151MB docker.elastic.co/elasticsearch/elasticsearch 6.1.1 06f0d8328d66 2 weeks ago 539MB
```

1.  现在我们可以使用以下 Docker 命令运行 Elasticsearch：

```py
docker run -e ELASTIC_PASSWORD=MagicWord -p 9200:9200 -p 9300:9300 docker.elastic.co/elasticsearch/elasticsearch:6.1.1
```

1.  环境变量`ELASTIC_PASSWORD`传递密码，两个端口将主机端口映射到容器中暴露的 Elasticsearch 端口。

1.  接下来，检查容器是否在 Docker 中运行：

```py
$ docker ps
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
308a02f0e1a5 docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 7 seconds ago Up 6 seconds 0.0.0.0:9200->9200/tcp, 0.0.0.0:9300->9300/tcp romantic_kowalevski
094a13838376 rabbitmq:3-management "docker-entrypoint..." 47 hours ago Up 47 hours 4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp dreamy_easley
```

1.  最后，执行以下 curl。如果 Elasticsearch 正在运行，您将收到`You Know, for Search`消息：

```py
$ curl localhost:9200
{
 "name" : "8LaZfMY",
 "cluster_name" : "docker-cluster",
 "cluster_uuid" : "CFgPERC8TMm5KaBAvuumvg",
 "version" : {
 "number" : "6.1.1",
 "build_hash" : "bd92e7f",
 "build_date" : "2017-12-17T20:23:25.338Z",
 "build_snapshot" : false,
 "lucene_version" : "7.1.0",
 "minimum_wire_compatibility_version" : "5.6.0",
 "minimum_index_compatibility_version" : "5.0.0"
 },
 "tagline" : "You Know, for Search"
}
```

# 停止/重新启动容器并删除镜像

让我们看看如何停止和删除一个容器，然后也删除它的镜像。

# 如何做

我们按照以下步骤进行：

1.  首先查询正在运行的 Docker 容器：

```py
$ docker ps
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
308a02f0e1a5 docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 7 seconds ago Up 6 seconds 0.0.0.0:9200->9200/tcp, 0.0.0.0:9300->9300/tcp romantic_kowalevski
094a13838376 rabbitmq:3-management "docker-entrypoint..." 47 hours ago Up 47 hours 4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp dreamy_easley
```

1.  让我们停止 Elasticsearch 容器。要停止一个容器，我们使用`docker stop <container-id>`。Elasticsearch 的容器 ID 是`308a02f0e1a5`。以下停止容器

```py
$ docker stop 30
30
```

为了确认容器已停止，Docker 将回显您告诉它停止的容器 ID

请注意，我不必输入完整的容器 ID，只输入了 30。你只需要输入容器 ID 的前几位数字，直到你输入的内容在所有容器中是唯一的。这是一个很好的快捷方式！

1.  检查运行的容器状态，Docker 只报告其他容器：

```py
$ docker ps
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
094a13838376 rabbitmq:3-management "docker-entrypoint..." 2 days ago Up 2 days 4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp dreamy_easley
```

1.  容器没有运行，但也没有被删除。让我们来使用`docker ps -a`命令：

```py
$ docker ps -a
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
308a02f0e1a5 docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 11 minutes ago Exited (143) 5 minutes ago romantic_kowalevski
548fc19e8b8d docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 12 minutes ago Exited (130) 12 minutes ago competent_keller
15c83ca72108 docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 15 minutes ago Exited (130) 14 minutes ago peaceful_jennings
3191f204c661 docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 18 minutes ago Exited (130) 16 minutes ago thirsty_hermann
b44f1da7613f docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 25 minutes ago Exited (130) 19 minutes ago
```

这列出了当前系统上的所有容器。实际上，我截断了我的列表，因为我有很多这样的容器！

1.  我们可以使用`docker restart`来重新启动我们的 Elasticsearch 容器：

```py
$ docker restart 30
30
```

1.  如果你检查`docker ps`，你会看到容器再次运行。

这很重要，因为这个容器在容器的文件系统中存储了 Elasticsearch 数据。通过停止和重新启动，这些数据不会丢失。因此，您可以停止以回收容器使用的资源（CPU 和内存），然后在以后的某个时间重新启动而不会丢失。

1.  无论是运行还是停止，容器都会占用磁盘空间。可以删除容器以回收磁盘空间。这可以使用`docker container rm <container-id>`来完成，但是只有在容器没有运行时才能删除容器。让我们尝试删除正在运行的容器：

```py
$ docker container rm 30
Error response from daemon: You cannot remove a running container 308a02f0e1a52fe8051d1d98fa19f8ac01ff52ec66737029caa07a8358740bce. Stop the container before attempting removal or force remove
```

1.  我们收到了有关容器运行的警告。我们可以使用一个标志来强制执行，但最好先停止它。停止可以确保容器内的应用程序干净地关闭：

```py
$ docker stop 30
30
$ docker rm 30
30
```

1.  现在，如果你回到 docker `ps -a`，Elasticsearch 容器不再在列表中，容器的磁盘空间被回收。

请注意，我们现在已经丢失了存储在该容器中的任何数据！这超出了本书的范围，但大多数容器可以被告知将数据存储在主机的文件系统上，因此我们不会丢失数据。

1.  容器的磁盘空间已经被删除，但是容器的镜像仍然在磁盘上。如果我们想创建另一个容器，这是很好的。但是如果你也想释放那个空间，你可以使用`docker images rm <image-id>`。回到 Docker 镜像结果，我们可以看到该镜像的 ID 是`06f0d8328d66`。以下删除该镜像，我们可以获得那个空间（在这种情况下是 539MB）：

```py
$ docker image rm 06
Untagged: docker.elastic.co/elasticsearch/elasticsearch:6.1.1
Untagged: docker.elastic.co/elasticsearch/elasticsearch@sha256:9e6c7d3c370a17736c67b2ac503751702e35a1336724741d00ed9b3d00434fcb
Deleted: sha256:06f0d8328d66a0f620075ee689ddb2f7535c31fb643de6c785deac8ba6db6a4c
Deleted: sha256:133d33f65d5a512c5fa8dc9eb8d34693a69bdb1a696006628395b07d5af08109
Deleted: sha256:ae2e02ab7e50b5275428840fd68fced2f63c70ca998a493d200416026c684a69
Deleted: sha256:7b6abb7badf2f74f1ee787fe0545025abcffe0bf2020a4e9f30e437a715c6d6a
```

现在镜像已经消失，我们也已经回收了那个空间。

请注意，如果还存在任何使用该镜像运行的容器，那么这将失败，这些容器可能正在运行或已停止。只是做一个`docker ps -a`可能不会显示有问题的容器，所以你可能需要使用`docker ps -a`来找到已停止的容器并首先删除它们。

# 还有更多...

在这一点上，你已经了解了足够多关于 Docker 的知识，可以变得非常危险！所以让我们继续研究如何创建我们自己的容器，并安装我们自己的应用程序。首先，让我们去看看如何将爬虫变成一个可以在容器中运行的微服务。

# 使用 Nameko 创建通用微服务

在接下来的几个步骤中，我们将创建一个可以作为 Docker 容器内的微服务运行的爬虫。但在直接进入火坑之前，让我们先看看如何使用一个名为 Nameko 的 Python 框架创建一个基本的微服务。

# 准备工作

我们将使用一个名为 Nameko 的 Python 框架（发音为[nah-meh-koh]）来实现微服务。与 Flask-RESTful 一样，使用 Nameko 实现的微服务只是一个类。我们将指示 Nameko 如何将该类作为服务运行，并且 Nameko 将连接一个消息总线实现，以允许客户端与实际的微服务进行通信。

默认情况下，Nameko 使用 RabbitMQ 作为消息总线。RabbitMQ 是一个高性能的消息总线，非常适合在微服务之间进行消息传递。它与我们之前在 SQS 中看到的模型类似，但更适合于位于同一数据中心的服务，而不是跨云。这实际上是 RabbitMQ 的一个很好的用途，因为我们现在倾向于在相同的环境中集群/扩展微服务，特别是在容器化集群中，比如 Docker 或 Kubernetes。

因此，我们需要在本地运行一个 RabbitMQ 实例。确保你有一个 RabbitMQ 容器运行，就像在之前的示例中展示的那样。

还要确保你已经安装了 Nameko：

```py
pip install Nameko
```

# 如何做到这一点

我们按照以下步骤进行操作：

1.  示例微服务实现在`10/01/hello_microservice.py`中。这是一个非常简单的服务，可以传递一个名字，微服务会回复`Hello, <name>!`。

1.  要运行微服务，我们只需要从终端执行以下命令（在脚本所在的目录中）：

```py
$nameko run hello_microservice
```

1.  Nameko 打开与指定微服务名称匹配的 Python 文件，并启动微服务。启动时，我们会看到几行输出：

```py
starting services: hello_microservice
Connected to amqp://guest:**@127.0.0.1:5672//
```

1.  这表明 Nameko 已经找到了我们的微服务，并且已经连接到了一个 AMQP 服务器（RabbitMQ）的 5672 端口（RabbitMQ 的默认端口）。微服务现在已经启动并且正在等待请求。

如果你进入 RabbitMQ API 并进入队列选项卡，你会看到 Nameko 已经自动为微服务创建了一个队列。

1.  现在我们必须做一些事情来请求微服务。我们将看两种方法来做到这一点。首先，Nameko 带有一个交互式 shell，让我们可以交互地向 Nameko 微服务发出请求。你可以在一个单独的终端窗口中使用以下命令启动 shell，与运行微服务的窗口分开：

```py
nameko shell
```

1.  你会看到一个交互式的 Python 会话开始，输出类似于以下内容：

```py
Nameko Python 3.6.1 |Anaconda custom (x86_64)| (default, Mar 22 2017, 19:25:17)
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)] shell on darwin
Broker: pyamqp://guest:guest@localhost
In [1]:
```

1.  在这个 shell 中，我们可以简单地将 Nameko 称为'n'。要与我们的服务交谈，我们发出以下声明：

```py
n.rpc.hello_microservice.hello(name='Mike')
```

1.  这告诉 Nameko 我们想要调用`hello_microservice`的`hello`方法。按下*Enter*后，你会得到以下结果：

```py
Out[1]: 'Hello, Mike!'
```

1.  如果你在运行服务的终端窗口中检查，你应该会看到额外的一行输出：

```py
Received a request from: Mike
```

1.  也可以在 Python 代码中调用微服务。在`10/01/say_hi.py`中有一个实现。用 Python 执行这个脚本会得到以下输出：

```py
$python say_hi.py
Hello, Micro-service Client!
```

那么让我们去看看这些是如何实现的。

# 它是如何工作的

让我们首先看一下`hello_microservice.py`中微服务的实现。实际上并没有太多的代码，所以这里是全部代码：

```py
from nameko.rpc import rpc

class HelloMicroService:
    name = "hello_microservice"    @rpc
  def hello(self, name):
        print('Received a request from: ' + name)
        return "Hello, {}!".format(name)
```

有两件事情要指出关于这个类。第一是声明`name = "hello_microservice"`。这是微服务的实际名称声明。这个成员变量被用来代替类名。

第二个是在`hello`方法上使用`@rpc`属性。这是一个 Nameko 属性，指定这个方法应该作为`rpc`风格的方法被微服务公开。因此，调用者会一直等待，直到从微服务接收到回复。还有其他实现方式，但是对于我们的目的，这是我们将使用的唯一方式。

当使用 nameko run 命令运行时，该模块将检查文件中带有 Nameko 属性的方法，并将它们连接到底层总线。

`say_hi.py`中的实现构建了一个可以调用服务的动态代理。代码如下：

```py
from nameko.standalone.rpc import ClusterRpcProxy

CONFIG = {'AMQP_URI': "amqp://guest:guest@localhost"}

with ClusterRpcProxy(CONFIG) as rpc:
    result = rpc.hello_microservice.hello("Micro-service Client")
    print(result)
```

动态代理是由`ClusterRpcProxy`类实现的。创建该类时，我们传递一个配置对象，该对象指定了服务所在的 AMQP 服务器的地址，在这种情况下，我们将这个实例称为变量`rpc`。然后，Nameko 动态识别下一个部分`.hello_microservice`作为微服务的名称（如在微服务类的名称字段中指定的）。

接下来的部分`.hello`代表要调用的方法。结合在一起，Nameko 调用`hello_microservice`的`hello`方法，传递指定的字符串，由于这是一个 RPC 代理，它会等待接收到回复。

远程过程调用，简称 RPC，会一直阻塞，直到结果从其他系统返回。与发布模型相比，发布模型中消息被发送后发送应用程序继续进行。

# 还有更多...

在 Nameko 中有很多好东西，我们甚至还没有看到。一个非常有用的因素是，Nameko 运行多个微服务实例的监听器。撰写本文时，默认值为 10。在底层，Nameko 将来自微服务客户端的请求发送到 RabbitMQ 队列，其中将有 10 个同时的请求处理器监听该队列。如果有太多的请求需要同时处理，RabbitMQ 将保留消息，直到 Nameko 回收现有的微服务实例来处理排队的消息。为了增加微服务的可伸缩性，我们可以通过微服务的配置简单地增加工作人员的数量，或者在另一个 Docker 容器中运行一个单独的 Nameko 微服务容器，或者在另一台计算机系统上运行。

# 创建一个抓取微服务

现在让我们把我们的抓取器变成一个 Nameko 微服务。这个抓取微服务将能够独立于 API 的实现而运行。这将允许抓取器独立于 API 的实现进行操作、维护和扩展。

# 如何做

我们按照以下步骤进行：

1.  微服务的代码很简单。代码在`10/02/call_scraper_microservice.py`中，如下所示：

```py
from nameko.rpc import rpc
import sojobs.scraping 

class ScrapeStackOverflowJobListingsMicroService:
    name = "stack_overflow_job_listings_scraping_microservice"    @rpc
  def get_job_listing_info(self, job_listing_id):
        listing = sojobs.scraping.get_job_listing_info(job_listing_id)
        print(listing)
        return listing

if __name__ == "__main__":
    print(ScrapeStackOverflowJobListingsMicroService("122517"))
```

1.  我们创建了一个类来实现微服务，并给它一个单一的方法`get_job_listing_info`。这个方法简单地包装了`sojobs.scraping`模块中的实现，但是给它一个`@rpc`属性，以便 Nameko 在微服务总线上公开该方法。这可以通过打开终端并使用 Nameko 运行服务来运行。

```py
$ nameko run scraper_microservice
 starting services: stack_overflow_job_listings_scraping_microservice
 Connected to amqp://guest:**@127.0.0.1:5672//
```

1.  现在我们可以使用`10/02/call_scraper_microservice.py`脚本中的代码运行抓取器。文件中的代码如下：

```py
from nameko.standalone.rpc import ClusterRpcProxy

CONFIG = {'AMQP_URI': "amqp://guest:guest@localhost"}

with ClusterRpcProxy(CONFIG) as rpc:
    result = rpc.stack_overflow_job_listings_scraping_microservice.get_job_listing_info("122517")
    print(result)
```

1.  这基本上与上一个教程中客户端的代码相同，但是更改了微服务和方法的名称，并当然传递了特定的工作列表 ID。运行时，您将看到以下输出（已截断）：

```py
{"ID": "122517", "JSON": {"@context": "http://schema.org", "@type": "JobPosting", "title": "SpaceX Enterprise Software Engineer, Full Stack", "skills": ["c#", "sql", "javascript", "asp.net", "angularjs"], 

...
```

1.  就像这样，我们已经创建了一个从 StackOverflow 获取工作列表的微服务！

# 还有更多...

这个微服务只能使用`ClusterRpcProxy`类调用，不能被任何人通过互联网甚至本地使用 REST 调用。我们将在即将到来的教程中解决这个问题，在那里我们将在一个容器中创建一个 REST API，该 API 将与另一个运行在另一个容器中的微服务进行通信。

# 创建一个抓取容器

现在我们为我们的抓取微服务创建一个容器。我们将学习 Dockerfile 以及如何指示 Docker 如何构建容器。我们还将研究如何为我们的 Docker 容器提供主机名，以便它们可以通过 Docker 集成的 DNS 系统相互找到。最后但并非最不重要的是，我们将学习如何配置我们的 Nameko 微服务，以便与另一个容器中的 RabbitMQ 通信，而不仅仅是在本地主机上。

# 准备工作

我们要做的第一件事是确保 RabbitMQ 在一个容器中运行，并分配给一个自定义的 Docker 网络，连接到该网络的各种容器将相互通信。除了许多其他功能外，它还提供了软件定义网络（SDN）功能，以在容器、主机和其他系统之间提供各种类型的集成。

Docker 自带了几个预定义的网络。您可以使用`docker network ls`命令查看当前安装的网络：

```py
$ docker network ls
NETWORK ID   NAME                                     DRIVER  SCOPE
bc3bed092eff bridge                                   bridge  local
26022f784cc1 docker_gwbridge                          bridge  local
448d8ce7f441 dockercompose2942991694582470787_default bridge  local
4e549ce87572 dockerelkxpack_elk                       bridge  local
ad399a431801 host                                     host    local
rbultxlnlhfb ingress                                  overlay swarm
389586bebcf2 none                                     null    local
806ff3ec2421 stackdockermaster_stack                  bridge  local
```

为了让我们的容器相互通信，让我们创建一个名为`scraper-net`的新桥接网络。

```py
$ docker network create --driver bridge scraper-net
e4ea1c48395a60f44ec580c2bde7959641c4e1942cea5db7065189a1249cd4f1
```

现在，当我们启动一个容器时，我们使用`--network`参数将其连接到`scraper-net`：

```py
$docker run -d --name rabbitmq --network scrape-rnet -p 15672:15672 -p 5672:5672 rabbitmq:3-management
```

这个容器现在连接到`scraper-net`网络和主机网络。因为它也连接到主机，所以仍然可以从主机系统连接到它。

还要注意，我们使用了`--name rabbitmq`作为一个选项。这给了这个容器名字`rabbitmq`，但 Docker 也会解析来自连接到`scraper-net`的其他容器的 DNS 查询，以便它们可以找到这个容器！

现在让我们把爬虫放到一个容器中。

# 如何做到这一点

我们按照以下步骤进行配方：

1.  我们创建容器的方式是创建一个`dockerfile`，然后使用它告诉 Docker 创建一个容器。我在`10/03`文件夹中包含了一个 Dockerfile。内容如下（我们将在*它是如何工作*部分检查这意味着什么）：

```py
FROM python:3 WORKDIR /usr/src/app

RUN pip install nameko BeautifulSoup4 nltk lxml
RUN python -m nltk.downloader punkt -d /usr/share/nltk_data all

COPY 10/02/scraper_microservice.py .
COPY modules/sojobs sojobs

CMD ["nameko", "run", "--broker", "amqp://guest:guest@rabbitmq", "scraper_microservice"]
```

1.  要从这个 Dockerfile 创建一个镜像/容器，在终端中，在`10/03`文件夹中，运行以下命令：

```py
$docker build ../.. -f Dockerfile  -t scraping-microservice
```

1.  这告诉 Docker，我们想要根据给定的 Dockerfile 中的指令*构建*一个容器（用-f 指定）。创建的镜像由指定

`-t scraping-microservice`。`build`后面的`../..`指定了构建的上下文。在构建时，我们将文件复制到容器中。这个上下文指定了复制相对于的主目录。当你运行这个命令时，你会看到类似以下的输出：

```py
Sending build context to Docker daemon 2.128MB
Step 1/8 : FROM python:3
 ---> c1e459c00dc3
Step 2/8 : WORKDIR /usr/src/app
 ---> Using cache
 ---> bf047017017b
Step 3/8 : RUN pip install nameko BeautifulSoup4 nltk lxml
 ---> Using cache
 ---> a30ce09e2f66
Step 4/8 : RUN python -m nltk.downloader punkt -d /usr/share/nltk_data all
 ---> Using cache
 ---> 108b063908f5
Step 5/8 : COPY 10/07/. .
 ---> Using cache
 ---> 800a205d5283
Step 6/8 : COPY modules/sojobs sojobs
 ---> Using cache
 ---> 241add5458a5
Step 7/8 : EXPOSE 5672
 ---> Using cache
 ---> a9be801d87af
Step 8/8 : CMD nameko run --broker amqp://guest:guest@rabbitmq scraper_microservice
 ---> Using cache
 ---> 0e1409911ac9
Successfully built 0e1409911ac9
Successfully tagged scraping-microservice:latest
```

1.  这可能需要一些时间，因为构建过程需要将所有的 NLTK 文件下载到容器中。要检查镜像是否创建，可以运行以下命令：

```py
$ docker images | head -n 2
REPOSITORY            TAG    IMAGE ID     CREATED     SIZE
scraping-microservice latest 0e1409911ac9 3 hours ago 4.16GB
```

1.  请注意，这个容器的大小是 4.16GB。这个镜像是基于`Python:3`容器的，可以看到大小为`692MB`：

```py
$ docker images | grep python
 python 3 c1e459c00dc3 2 weeks ago 692MB
```

这个容器的大部分大小是因为包含了 NTLK 数据文件。

1.  现在我们可以使用以下命令将这个镜像作为一个容器运行：

```py
03 $ docker run --network scraper-net scraping-microservice
starting services: stack_overflow_job_listings_scraping_microservice
Connected to amqp://guest:**@rabbitmq:5672//
```

我们组合的爬虫现在在这个容器中运行，这个输出显示它已经连接到一个名为`rabbitmq`的系统上的 AMQP 服务器。

1.  现在让我们测试一下这是否有效。在另一个终端窗口中运行 Nameko shell：

```py
03 $ nameko shell
Nameko Python 3.6.1 |Anaconda custom (x86_64)| (default, Mar 22 2017, 19:25:17)
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)] shell on darwin
Broker: pyamqp://guest:guest@localhost
In [1]:
```

1.  现在，在提示符中输入以下内容来调用微服务：

```py
n.rpc.stack_overflow_job_listings_scraping_microservice.get_job_listing_info("122517")
```

1.  由于抓取的结果，你会看到相当多的输出（以下是截断的）：

```py
Out[1]: '{"ID": "122517", "JSON": {"@context": "http://schema.org", "@type": "JobPosting", "title": "SpaceX Enterprise Software Engineer, Full Stack", "skills": ["c#", "sql", "javascript", "asp.net"
```

恭喜！我们现在已经成功调用了我们的爬虫微服务。现在，让我们讨论这是如何工作的，以及 Dockerfile 是如何构建微服务的 Docker 镜像的。

# 它是如何工作的

让我们首先讨论 Dockerfile，通过在构建过程中告诉 Docker 要做什么来逐步了解它的内容。第一行：

```py
FROM python:3
```

这告诉 Docker，我们想要基于 Docker Hub 上找到的`Python:3`镜像构建我们的容器镜像。这是一个预先构建的 Linux 镜像，安装了 Python 3。下一行告诉 Docker，我们希望所有的文件操作都是相对于`/usr/src/app`文件夹的。

```py
WORKDIR /usr/src/app
```

在构建镜像的这一点上，我们已经安装了一个基本的 Python 3。然后我们需要安装我们的爬虫使用的各种库，所以下面告诉 Docker 运行 pip 来安装它们：

```py
RUN pip install nameko BeautifulSoup4 nltk lxml
```

我们还需要安装 NLTK 数据文件：

```py
RUN python -m nltk.downloader punkt -d /usr/share/nltk_data all
```

接下来，我们将实现我们的爬虫复制进去。以下是将`scraper_microservice.py`文件从上一个配方的文件夹复制到容器镜像中。

```py
COPY 10/02/scraper_microservice.py .
```

这也取决于`sojobs`模块，因此我们也复制它：

```py
COPY modules/sojobs sojobs
```

最后一行告诉 Docker 在启动容器时要运行的命令：

```py
CMD ["nameko", "run", "--broker", "amqp://guest:guest@rabbitmq", "scraper_microservice"]
```

这告诉 Nameko 在`scraper_microservice.py`中运行微服务，并且还与名为`rabbitmq`的系统上的 RabbitMQ 消息代理进行通信。由于我们将 scraper 容器附加到 scraper-net 网络，并且还对 RabbitMQ 容器执行了相同操作，Docker 为我们连接了这两个容器！

最后，我们从 Docker 主机系统中运行了 Nameko shell。当它启动时，它报告说它将与 AMQP 服务器（RabbitMQ）通信`pyamqp://guest:guest@localhost`。当我们在 shell 中执行命令时，Nameko shell 将该消息发送到 localhost。

那么它如何与容器中的 RabbitMQ 实例通信呢？当我们启动 RabbitMQ 容器时，我们告诉它连接到`scraper-net`网络。它仍然连接到主机网络，因此只要我们在启动时映射了`5672`端口，我们仍然可以与 RabbitMQ 代理进行通信。

我们在另一个容器中的微服务正在 RabbitMQ 容器中监听消息，然后响应该容器，然后由 Nameko shell 接收。这很酷，不是吗？

# 创建 API 容器

此时，我们只能使用 AMQP 或使用 Nameko shell 或 Nameko `ClusterRPCProxy`类与我们的微服务进行通信。因此，让我们将我们的 Flask-RESTful API 放入另一个容器中，与其他容器一起运行，并进行 REST 调用。这还需要我们运行一个 Elasticsearch 容器，因为该 API 代码还与 Elasticsearch 通信。

# 准备就绪

首先让我们在附加到`scraper-net`网络的容器中启动 Elasticsearch。我们可以使用以下命令启动它：

```py
$ docker run -e ELASTIC_PASSWORD=MagicWord --name=elastic --network scraper-net  -p 9200:9200 -p 9300:9300 docker.elastic.co/elasticsearch/elasticsearch:6.1.1
```

Elasticsearch 现在在我们的`scarper-net`网络上运行。其他容器中的应用程序可以使用名称 elastic 访问它。现在让我们继续创建 API 的容器。

# 如何做

我们按照以下步骤进行：

1.  在`10/04`文件夹中有一个`api.py`文件，该文件实现了一个修改后的 Flask-RESTful API，但进行了几处修改。让我们检查 API 的代码：

```py
from flask import Flask
from flask_restful import Resource, Api
from elasticsearch import Elasticsearch
from nameko.standalone.rpc import ClusterRpcProxy

app = Flask(__name__)
api = Api(app)

CONFIG = {'AMQP_URI': "amqp://guest:guest@rabbitmq"}

class JobListing(Resource):
    def get(self, job_listing_id):
        print("Request for job listing with id: " + job_listing_id)

        es = Elasticsearch(hosts=["elastic"])
        if (es.exists(index='joblistings', doc_type='job-listing', id=job_listing_id)):
            print('Found the document in Elasticsearch')
            doc =  es.get(index='joblistings', doc_type='job-listing', id=job_listing_id)
            return doc['_source']

        print('Not found in Elasticsearch, trying a scrape')
        with ClusterRpcProxy(CONFIG) as rpc:
            listing = rpc.stack_overflow_job_listings_scraping_microservice.get_job_listing_info(job_listing_id)
            print("Microservice returned with a result - storing in Elasticsearch")
            es.index(index='joblistings', doc_type='job-listing', id=job_listing_id, body=listing)
            return listing

api.add_resource(JobListing, '/', '/joblisting/<string:job_listing_id>')

if __name__ == '__main__':
    print("Starting the job listing API ...")
    app.run(host='0.0.0.0', port=8080, debug=True)
```

1.  第一个变化是 API 上只有一个方法。我们现在将重点放在`JobListing`方法上。在该方法中，我们现在进行以下调用以创建 Elasticsearch 对象：

```py
es = Elasticsearch(hosts=["elastic"])
```

1.  默认构造函数假定 Elasticsearch 服务器在 localhost 上。此更改现在将其指向 scraper-net 网络上名为 elastic 的主机。

1.  第二个变化是删除对 sojobs 模块中函数的调用。相反，我们使用`Nameko ClusterRpcProxy`对象调用在 scraper 容器内运行的 scraper 微服务。该对象传递了一个配置，将 RPC 代理指向 rabbitmq 容器。

1.  最后一个变化是 Flask 应用程序的启动：

```py
    app.run(host='0.0.0.0', port=8080, debug=True)
```

1.  默认连接到 localhost，或者 127.0.0.1。在容器内部，这不会绑定到我们的`scraper-net`网络，甚至不会绑定到主机网络。使用`0.0.0.0`将服务绑定到所有网络接口，因此我们可以通过容器上的端口映射与其通信。端口也已移至`8080`，这是比 5000 更常见的 REST API 端口。

1.  将 API 修改为在容器内运行，并与 scraper 微服务通信后，我们现在可以构建容器。在`10/04`文件夹中有一个 Dockerfile 来配置容器。其内容如下：

```py
FROM python:3 WORKDIR /usr/src/app

RUN pip install Flask-RESTful Elasticsearch Nameko

COPY 10/04/api.py .

CMD ["python", "api.py"]
```

这比以前容器的 Dockerfile 简单。该容器没有 NTLK 的所有权重。最后，启动只需执行`api.py`文件。

1.  使用以下内容构建容器：

```py
$docker build ../.. -f Dockerfile -t scraper-rest-api
```

1.  然后我们可以使用以下命令运行容器：

```py
$docker run -d -p 8080:8080 --network scraper-net scraper-rest-api
```

1.  现在让我们检查一下我们的所有容器是否都在运行：

```py
$ docker ps
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
55e438b4afcd scraper-rest-api "python -u api.py" 46 seconds ago Up 45 seconds 0.0.0.0:8080->8080/tcp vibrant_sammet
bb8aac5b7518 docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 3 hours ago Up 3 hours 0.0.0.0:9200->9200/tcp, 0.0.0.0:9300->9300/tcp elastic
ac4f51c1abdc scraping-microservice "nameko run --brok..." 3 hours ago Up 3 hours thirsty_ritchie
18c2f01f58c7 rabbitmq:3-management "docker-entrypoint..." 3 hours ago Up 3 hours 4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp rabbitmq
```

1.  现在，从主机终端上，我们可以向 REST 端点发出 curl 请求（输出已截断）：

```py
$ curl localhost:8080/joblisting/122517
"{\"ID\": \"122517\", \"JSON\": {\"@context\": \"http://schema.org\", \"@type\": \"JobPosting\", \"title\": \"SpaceX Enterprise Software Engineer, Full Stack\", \"skills\": [\"c#\", \"sql\", \"javas
```

然后我们就完成了。我们已经将 API 和功能容器化，并在容器中运行了 RabbitMQ 和 Elasticsearch。

# 还有更多...

这种类型的容器化对于操作的设计和部署是一个巨大的优势，但是我们仍然需要创建许多 Docker 文件、容器和网络来连接它们，并独立运行它们。幸运的是，我们可以使用 docker-compose 来简化这个过程。我们将在下一个步骤中看到这一点。

# 使用 docker-compose 在本地组合和运行爬虫

Compose 是一个用于定义和运行多容器 Docker 应用程序的工具。使用 Compose，您可以使用 YAML 文件配置应用程序的服务。然后，通过一个简单的配置文件和一个命令，您可以从配置中创建和启动所有服务。

# 准备就绪

使用 Compose 的第一件事是确保已安装。Compose 会随 Docker for macOS 自动安装。在其他平台上，可能已安装或未安装。您可以在以下网址找到说明：[`docs.docker.com/compose/install/#prerequisites`](https://docs.docker.com/compose/install/#prerequisites)。

此外，请确保我们之前创建的所有现有容器都没有在运行，因为我们将创建新的容器。

# 如何做到这一点

我们按照以下步骤进行：

1.  Docker Compose 使用`docker-compose.yml`文件告诉 Docker 如何将容器组合为`services`。在`10/05`文件夹中有一个`docker-compose.yml`文件，用于将我们的爬虫的所有部分作为服务启动。以下是文件的内容：

```py
version: '3' services:
 api: image: scraper-rest-api
  ports:
  - "8080:8080"
  networks:
  - scraper-compose-net    scraper:
 image: scraping-microservice
  depends_on:
  - rabbitmq
  networks:
  - scraper-compose-net    elastic:
 image: docker.elastic.co/elasticsearch/elasticsearch:6.1.1
  ports:
  - "9200:9200"
  - "9300:9300"
  networks:
  - scraper-compose-net    rabbitmq:
 image: rabbitmq:3-management
  ports:
  - "15672:15672"
  networks:
  - scraper-compose-net   networks:
 scraper-compose-net: driver: bridge
```

使用 Docker Compose，我们不再考虑容器，而是转向与服务一起工作。在这个文件中，我们描述了四个服务（api、scraper、elastic 和 rabbitmq）以及它们的创建方式。每个服务的图像标签告诉 Compose 要使用哪个 Docker 图像。如果需要映射端口，那么我们可以使用`ports`标签。`network`标签指定要连接服务的网络，在这种情况下，文件中还声明了一个`bridged`网络。最后要指出的一件事是 scraper 服务的`depends_on`标签。该服务需要在之前运行`rabbitmq`服务，这告诉 docker compose 确保按指定顺序进行。

1.  现在，要启动所有内容，打开一个终端并从该文件夹运行以下命令：

```py
    $ docker-compose up
```

1.  Compose 在读取配置并弄清楚要做什么时会暂停一会儿，然后会有相当多的输出，因为每个容器的输出都将流式传输到这个控制台。在输出的开头，您将看到类似于以下内容：

```py
Starting 10_api_1 ...
 Recreating elastic ...
 Starting rabbitmq ...
 Starting rabbitmq
 Recreating elastic
 Starting rabbitmq ... done
 Starting 10_scraper_1 ...
 Recreating elastic ... done
 Attaching to rabbitmq, 10_api_1, 10_scraper_1, 10_elastic_1
```

1.  在另一个终端中，您可以发出`docker ps`命令来查看已启动的容器：

```py
$ docker ps
 CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
 2ed0d456ffa0 docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 3 minutes ago Up 2 minutes 0.0.0.0:9200->9200/tcp, 0.0.0.0:9300->9300/tcp 10_elastic_1
 8395989fac8d scraping-microservice "nameko run --brok..." 26 minutes ago Up 3 minutes 10_scraper_1
 4e9fe8479db5 rabbitmq:3-management "docker-entrypoint..." 26 minutes ago Up 3 minutes 4369/tcp, 5671-5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp rabbitmq
 0b0df48a7201 scraper-rest-api "python -u api.py" 26 minutes ago Up 3 minutes 0.0.0.0:8080->8080/tcp 10_api_1
```

注意服务容器的名称。它们被两个不同的标识符包裹。前缀只是运行组合的文件夹，本例中为 10（用于'10_'前缀）。您可以使用-p 选项来更改这个，以指定其他内容。尾随的数字是该服务的容器实例编号。在这种情况下，我们每个服务只启动了一个容器，所以这些都是 _1。不久之后，当我们进行扩展时，我们将看到这一点发生变化。

您可能会问：如果我的服务名为`rabbitmq`，而 Docker 创建了一个名为`10_rabbitmq_1`的容器，那么使用`rabbitmq`作为主机名的微服务如何连接到 RabbitMQ 实例？在这种情况下，Docker Compose 已经为您解决了这个问题，因为它知道`rabbitmq`需要被转换为`10_rabbitmq_1`。太棒了！

1.  作为启动此环境的一部分，Compose 还创建了指定的网络：

```py
$ docker network ls | head -n 2
 NETWORK ID NAME DRIVER SCOPE
 0e27be3e30f2 10_scraper-compose-net bridge local
```

如果我们没有指定网络，那么 Compose 将创建一个默认网络并将所有内容连接到该网络。在这种情况下，这将正常工作。但在更复杂的情况下，这个默认值可能不正确。

1.  现在，此时一切都已经启动并运行。让我们通过调用 REST 抓取 API 来检查一切是否正常运行：

```py
$ curl localhost:8080/joblisting/122517
 "{\"ID\": \"122517\", \"JSON\": {\"@context\": \"http://schema.org\", \"@type\": \"JobPosting\", \"title\": \"SpaceX Enterprise Software Engineer, Full Stack\", \"
...
```

1.  同时，让我们通过检查工作列表的索引来确认 Elasticsearch 是否正在运行，因为我们已经请求了一个：

```py
$ curl localhost:9200/joblisting
{"error":{"root_cause":{"type":"index_not_found_exception","reason":"no such index","resource.type":"index_or_alias","resource.id":"joblisting","index_uuid":"_na_","index":"j
...
```

1.  我们还可以使用 docker-compose 来扩展服务。如果我们想要添加更多微服务容器以增加处理请求的数量，我们可以告诉 Compose 增加 scraper 服务容器的数量。以下命令将 scraper 容器的数量增加到 3 个：

```py
docker-compose up --scale scraper=3
```

1.  Compose 将会考虑一会儿这个请求，然后发出以下消息，说明正在启动另外两个 scraper 服务容器（随后会有大量输出来自这些容器的初始化）：

```py
10_api_1 is up-to-date
10_elastic_1 is up-to-date
10_rabbitmq_1 is up-to-date
Starting 10_scraper_1 ... done
Creating 10_scraper_2 ...
Creating 10_scraper_3 ...
Creating 10_scraper_2 ... done
Creating 10_scraper_3 ... done
Attaching to 10_api_1, 10_elastic_1, 10_rabbitmq_1, 10_scraper_1, 10_scraper_3, 10_scraper_2
```

1.  `docker ps`现在将显示三个正在运行的 scraper 容器：

```py
Michaels-iMac-2:09 michaelheydt$ docker ps
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
b9c2da0c9008 scraping-microservice "nameko run --brok..." About a minute ago Up About a minute 10_scraper_2
643221f85364 scraping-microservice "nameko run --brok..." About a minute ago Up About a minute 10_scraper_3
73dc31fb3d92 scraping-microservice "nameko run --brok..." 6 minutes ago Up 6 minutes 10_scraper_1
5dd0db072483 scraper-rest-api "python api.py" 7 minutes ago Up 7 minutes 0.0.0.0:8080->8080/tcp 10_api_1
d8e25b6ce69a rabbitmq:3-management "docker-entrypoint..." 7 minutes ago Up 7 minutes 4369/tcp, 5671-5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp 10_rabbitmq_1
f305f81ae2a3 docker.elastic.co/elasticsearch/elasticsearch:6.1.1 "/usr/local/bin/do..." 7 minutes ago Up 7 minutes 0.0.0.0:9200->9200/tcp, 0.0.0.0:9300->9300/tcp 10_elastic_1
```

1.  现在我们可以看到我们有三个名为`10_scraper_1`、`10_scraper_2`和`10_scraper_3`的容器。很酷！如果你进入 RabbitMQ 管理界面，你会看到有三个连接：

![RabbitMQ 中的 Nameko 队列请注意每个队列都有不同的 IP 地址。在像我们创建的桥接网络上，Compose 会在`172.23.0`网络上分配 IP 地址，从`.2`开始。

操作上，所有来自 API 的抓取请求都将被路由到 rabbitmq 容器，实际的 RabbitMQ 服务将把消息传播到所有活动连接，因此传播到所有三个容器，帮助我们扩展处理能力。

服务实例也可以通过发出一个较小数量的容器的规模值来缩减，Compose 将会响应并删除容器，直到达到指定的值。

当一切都完成时，我们可以告诉 Docker Compose 关闭所有内容：

```py
$ docker-compose down
Stopping 10_scraper_1 ... done
Stopping 10_rabbitmq_1 ... done
Stopping 10_api_1 ... done
Stopping 10_elastic_1 ... done
Removing 10_scraper_1 ... done
Removing 10_rabbitmq_1 ... done
Removing 10_api_1 ... done
Removing 10_elastic_1 ... done
Removing network 10_scraper-compose-net
```

执行`docker ps`现在将显示所有容器都已被移除。

# 还有更多...

我们几乎没有涉及 Docker 和 Docker Compose 的许多功能，甚至还没有开始研究使用 Docker swarm 等服务。虽然 docker Compose 很方便，但它只在单个主机上运行容器，最终会有可扩展性的限制。Docker swarm 将执行类似于 Docker Compose 的操作，但是在集群中跨多个系统进行操作，从而实现更大的可扩展性。但希望这让你感受到了 Docker 和 Docker Compose 的价值，以及在创建灵活的抓取服务时它们的价值。
