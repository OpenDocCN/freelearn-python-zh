

# 第十二章：部署和管理 FastAPI 应用程序

在本章中，我们将深入探讨部署和管理 FastAPI 应用程序的基本方面。随着您开发 FastAPI 项目，了解如何有效地运行、安全和扩展它们对于确保生产环境中的性能和可靠性至关重要。本章将为您提供部署 FastAPI 应用程序所需的知识和工具，利用各种技术和最佳实践，使部署过程无缝进行。

您将学习如何利用 **FastAPI CLI** 高效地运行您的服务器，启用 **HTTPS** 以保护您的应用程序，以及使用 **Docker** 容器化您的 FastAPI 项目。此外，我们将探讨跨多个工作进程扩展应用程序、打包应用程序以进行分发以及在 **Railway** 等云平台上部署的技术。本章中的每个配方都提供了逐步说明、实际示例以及优化部署工作流程的见解。

到本章结束时，您将熟练使用现代工具和方法部署 FastAPI 应用程序。您将能够始终使用 HTTPS 保护应用程序，在 Docker 容器中运行它们，使用多个工作进程进行扩展，并在云平台上部署它们。这些技能对于任何希望将 FastAPI 应用程序从开发推向生产的开发者来说是无价的。

在本章中，我们将介绍以下配方：

+   使用 FastAPI CLI 运行服务器

+   在 FastAPI 应用程序上启用 HTTPS

+   在 Docker 容器中运行 FastAPI 应用程序

+   跨多个工作进程运行服务器

+   在云上部署您的 FastAPI 应用程序

+   使用 Hatch 打包 FastAPI 应用程序

# 技术要求

本章是为希望学习如何在云上部署 FastAPI 应用程序的高级用户而编写的。如果您是 FastAPI 或 Python 的新手，您可能想查看本书的前两章。

您可以在 GitHub 上找到本章的代码：[`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter12`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter12).

为了管理依赖项并隔离项目，在项目根目录中设置虚拟环境。

对于 *在 Docker 容器中运行 FastAPI 应用程序* 和 *跨多个工作进程运行服务器* 的配方，我们将使用 Docker。请确保在您的机器上安装它。

# 使用 FastAPI CLI 运行服务器

FastAPI 的 `$ fastapi` 命令用于运行 FastAPI 应用程序、管理 FastAPI 项目以及执行其他操作。这个功能最近在版本 0.111.0 中添加。

在本配方中，我们将探讨如何使用 FastAPI CLI 运行 FastAPI 应用程序。这种方法可以简化您的开发工作流程，并提供一种更直观的方式来管理您的服务器。

## 准备工作

要运行配方，请确保你有一个至少包含一个端点的最小 FastAPI 模块的应用程序。我们将工作于一个名为`Live Application`的新应用程序，因此创建一个名为`live_application`的新项目文件夹，其中包含一个`app`子文件夹，并包含一个`main.py`模块，如下所示：

```py
from fastapi import FastAPI
app = FastAPI(title="FastAPI Live Application")
@app.get("/")
def read_root():
    return {"Hello": "World"}
```

此外，确保你的环境中安装的 FastAPI 版本高于`0.111.0`，可以通过在命令行中运行以下命令来检查：

```py
$ pip install "fastapi~=0.111.0"
```

如果你已经安装了它，请确保你的环境中安装的是`fastapi`的最新版本。你可以通过运行以下命令来完成：

```py
$ pip install fastapi --upgrade
```

安装或升级完成后，我们可以开始执行配方。

## 如何操作…

在设置好你的应用程序后，只需从命令行运行以下命令：

```py
$ fastapi dev
```

你将在终端上看到打印的详细信息。让我们检查其中最重要的几个。

第一条消息是这样的：

```py
INFO    Using path app\main.py
```

在`fastapi dev`命令中，我们没有指定`app.main:app`参数，就像我们以前使用`uvicorn`命令时做的那样。FastAPI CLI 会根据一组默认路径自动检测代码中的`FastAPI`对象类。

以下消息是关于通过查看要考虑的包和模块来构建服务器。然后，它明确显示了`FastAPI`对象类的解析导入：

```py
╭─ Python module file ─╮
│                      │
│   main.py            │
│                      │
╰──────────────────────╯
INFO    Importing module main
INFO    Found importable FastAPI app
╭─ Importable FastAPI app ─╮
│                          │
│  from main import app    │
│                          │
╰──────────────────────────╯
INFO    Using import string main:app
```

然后，你会看到指定运行模式的主地址类似于以下内容：

```py
╭────────── FastAPI CLI - Development mode ───────────╮
│                                                     │
│  Serving at: http://127.0.0.1:8000                  │
│                                                     │
│  API docs: http://127.0.0.1:8000/docs               │
│                                                     │
│  Running in development mode, for production use:   │
│                                                     │
│  fastapi run                                        │
│                                                     │
╰─────────────────────────────────────────────────────╯
```

这条消息表明应用程序正在开发模式下运行。

这意味着当有代码更新时，服务器将自动重启，并且服务器将在本地地址`127.0.0.1`上运行。

你可以选择通过运行以下命令以生产模式运行服务器：

```py
$ fastapi run
```

这不会应用任何重载，服务器将使应用程序对托管服务器的本地网络可见。

这些是一些你可以使用的基本命令，以不同的设置和选项运行你的 FastAPI 应用程序。对于更高级的功能和配置，你可以参考 FastAPI 文档。

## 还有更多…

FastAPI CLI 依赖于`uvicorn`命令来运行。一些参数是相似的。例如，如果我们想在不同于`8000`的端口号上运行服务，我们可以使用`--port`参数，或者要指定主机地址，我们可以使用`--host`。你可以使用`--help`参数来查看带有所有可用参数的命令行文档。例如，你可以运行以下命令：

```py
$ fastapi run --help
```

例如，要运行对网络可见的应用程序，你可以将未指定的地址`0.0.0.0`传递给主机，如下所示：

```py
$ fastapi run
```

这相当于以下命令：

```py
$ uvicorn app.main:app --host 0.0.0.0
```

你的应用程序现在将可见于托管本地网络。

## 参见

你可以在官方文档页面上了解更多关于 FastAPI CLI 的功能：

+   *FastAPI* *CLI*: https://fastapi.tiangolo.com/fastapi-cli/

# 在 FastAPI 应用程序上启用 HTTPS

网络应用程序需要安全性，**超文本传输协议安全** (**HTTPS**) 是在客户端和服务器之间安全通信的基本方式。

HTTPS 对通过网络发送的数据进行加密，防止未经授权的访问和修改。

在本配方中，我们将学习如何在 FastAPI 应用程序上启用 HTTPS 以进行本地测试。我们将使用 `mkcert` 为本地开发创建 **安全套接字层/传输层安全性** (**SSL/TLS**) 证书，并为生产部署提供一些建议。到配方结束时，您将能够使用 HTTPS 保护您的 FastAPI 应用程序，提高其安全性和可靠性。

## 准备工作

关于 HTTPS 和 SSL/TLS 证书的一些背景信息可以帮助您完成此配方。从消费者角度来看，您可以在以下链接中找到良好的概述：[`howhttps.works/`](https://howhttps.works/)。

我们还将使用一个现有的应用程序作为示例。您可以将配方应用于您自己的应用程序，或者使用 `Live Application` 作为参考。

您还需要 `mkcert`，因此请确保在您的机器上正确安装它。安装取决于您的操作系统，您可以在以下位置查看说明：[`github.com/FiloSottile/mkcert?tab=readme-ov-file#installation`](https://github.com/FiloSottile/mkcert?tab=readme-ov-file#installation)。

安装后，从您的终端运行此命令以了解如何使用它并检查它是否正常工作：

```py
$ mkcert
```

安装完成后，我们可以开始执行配方。

## 如何操作...

让我们按照以下步骤设置我们的证书。

1.  让我们先允许我们的浏览器信任使用 `mkcert` 本地创建的证书。运行以下简单命令：

    ```py
    $ mkcert -install
    The local CA is now installed in the system trust store! ⚡
    ```

    此命令已将本地证书添加到您的操作系统信任存储中，以便您的浏览器会自动将其作为证书的可靠来源接受。

1.  然后，我们可以通过运行以下命令创建服务器将用于某些域名范围的证书和私钥：

    ```py
    example.com+5-key.pem for the key and example.com+5.pem for the certificate.
    ```

警告

为了确保安全，在创建证书和密钥时，不要将它们包含在您的 Git 历史记录中。将 `*.pem` 文件扩展名添加到 `.gitignore` 文件中

1.  当服务器启动时，我们必须将密钥和证书提供给服务器。在撰写本文时，`fastapi` 命令不支持将密钥和证书传递给服务器的参数，因此我们将通过运行以下命令使用 `uvicorn` 启动服务器：

    ```py
    $ uvicorn app.main:app --port 443  \
    --ssl-keyfile example.com+5-key.pem \
    --ssl-certfile example.com+5.pem
    ```

    此命令将使用证书和密钥启动服务器。

这是您设置 HTTPS 服务器连接所需的所有内容。

要测试它，请打开您的浏览器，并转到 `localhost` 地址。

您将在地址栏中看到锁形图标，这意味着连接是 HTTPS。

然而，如果您尝试通过 `http://localhost:443` 使用 HTTP 连接访问地址，您将收到错误响应。

您可以通过使用 FastAPI 提供的专用中间件将服务器的 HTTPS 自动重定向来修复此问题。按照以下方式更改 `main.py` 文件：

```py
from fastapi import FastAPI
from fastapi.middleware.httpsredirect import (
    HTTPSRedirectMiddleware,
)
app = FastAPI(title="FastAPI Live Application")
app.add_middleware(HTTPSRedirectMiddleware)
# rest of the module
```

然后，重新启动服务器。如果您尝试使用 HTTP 连接连接到 `localhost`（例如，`http://localhost:443`），它将自动将您重定向到 HTTPS 连接，`https://localhost`。然而，由于它不支持端口重定向，您仍然需要指定端口 `443`。

您已在服务器内为您的 FastAPI 应用程序启用 HTTPS 连接。通过为您的 FastAPI 应用程序启用 HTTPS，您已经迈出了增强网络安全和用户体验的重要一步。现在您可以更有信心和信任地享受 FastAPI 的功能。

## 还有更多…

我们已经看到了如何为本地测试生成 TLS/SSL 证书。在生产环境中，这将是类似的，区别在于这将涉及域名系统（**DNS**）托管提供商。

这里有一些关于如何操作的通用指南：

1.  为您的域名生成一个私钥和一个 **证书签名请求（CSR**）。使用 **OpenSSL** 或 **mkcert** 等工具。请保密私钥。CSR 包含证书颁发机构将验证的关于您的域名和组织的信息。

1.  将 CSR 提交给证书颁发机构并获取已签名的证书。证书颁发机构是一个受信任的实体，负责颁发和验证 TLS/SSL 证书。有自签名、免费或付费的证书颁发机构。根据证书颁发机构的不同，您可能需要提供更多关于您身份和域名所有权的信息。一些流行的证书颁发机构包括 **Let’s Encrypt**、**DigiCert** 和 **Comodo**。

1.  在您的 Web 服务器上安装证书和私钥。根据服务器软件和操作系统，程序可能会有所不同。您可能还需要从证书颁发机构安装中间证书。配置您的 Web 服务器以使用 HTTPS 并将 HTTP 重定向到 HTTPS。

通常，您的托管服务提供商可能会为您处理 TLS/SSL 证书和配置。一些提供商使用 **Certbot** 等工具从 Let’s Encrypt 获取和更新证书，或者他们使用自己的证书颁发机构。请咨询您的提供商以了解他们是否提供此类选项以及如何使用它们。

## 参考以下内容

以下链接中的 GitHub 仓库展示了 `mkcert` 的更多可能性：

+   *mkcert:* [`github.com/FiloSottile/mkcert`](https://github.com/FiloSottile/mkcert)

在 FastAPI 官方文档中，您可以在以下页面上查看 HTTPS 的工作原理：

+   *关于 HTTPS:* [`fastapi.tiangolo.com/deployment/https/`](https://fastapi.tiangolo.com/deployment/https/)

在以下链接中可以找到如何以 HTTPS 模式运行 `uvicorn` 的说明：

+   *使用 HTTPS 运行*：[`www.uvicorn.org/deployment/#running-with-https`](https://www.uvicorn.org/deployment/#running-with-https)

您可以在以下链接的官方文档页面上找到有关 `HTTPSRedirectMiddle` 的详细信息：

+   *HTTPSRedirectMiddleware:* [`fastapi.tiangolo.com/advanced/middleware/#httpsredirectmiddleware`](https://fastapi.tiangolo.com/advanced/middleware/#httpsredirectmiddleware)

# 在 Docker 容器中运行 FastAPI 应用程序

**Docker** 是一个有用的工具，它允许开发者将应用程序及其依赖项打包到容器中。这种方法确保应用程序在不同的环境中可靠运行，避免了常见的“在我的机器上工作”问题。在本教程中，我们将了解如何创建 Dockerfile 并在 Docker 容器中运行 FastAPI 应用程序。到本指南结束时，您将知道如何将 FastAPI 应用程序放入容器中，使其更加灵活且易于部署。

## 准备工作

为了更好地遵循本教程，您需要了解一些容器技术知识，特别是 Docker。但首先，请检查您的机器上是否已正确设置 **Docker Engine**。您可以通过以下链接了解如何操作：[`docs.docker.com/engine/install/`](https://docs.docker.com/engine/install/)。

如果您使用 Windows，则最好安装 **Docker Desktop**，这是一个带有内置图形界面的 Docker 虚拟机发行版。

无论您使用 Docker Engine 还是 Docker Desktop，请通过输入以下命令确保守护程序正在运行：

```py
$ docker images
```

如果您没有看到关于守护程序的任何错误，这意味着 Docker 已安装并在机器上运行。启动 Docker 守护程序的方式取决于您选择的安装方式。查看相关文档了解如何操作。

您可以使用此教程为您的应用程序或跟随我们在第一个教程中介绍的“实时应用程序”应用程序，我们将在本章中使用它。

## 如何操作...

在 Docker 容器中运行简单的 FastAPI 应用程序并不复杂。该过程包括三个步骤：

1.  创建 Dockerfile。

1.  构建镜像。

1.  生成容器。

然后，您只需运行容器即可使应用程序工作。

### 创建 Dockerfile

Dockerfile 包含从操作系统和我们要指定的文件构建镜像所需的指令。

为开发环境创建一个单独的 Dockerfile 是一种良好的实践。我们将将其命名为 `Dockerfile.dev` 并将其放置在项目根目录下。

我们通过指定以下基本镜像来开始文件：

```py
FROM python:3.10
```

这将从 Docker Hub 拉取一个镜像，该镜像已经集成了 Python 3.10。然后，我们创建一个名为 `/code` 的文件夹，它将托管我们的代码：

```py
WORKDIR /code
```

接下来，我们将 `requirements.txt` 复制到镜像中，并在镜像内安装这些包：

```py
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt
```

`pip install` 命令使用 `--no-cache-dir` 参数运行，以避免在容器内进行不会带来益处的 `pip` 缓存操作。此外，在生产环境中，对于较大的应用程序，建议在 `requirements.txt` 中固定包的版本，以避免由于包升级而可能出现的兼容性问题。

然后，我们可以使用以下命令将包含应用程序的 `app` 文件夹复制到镜像中：

```py
COPY ./app /code/app
```

最后，我们定义服务器启动指令如下：

```py
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
```

这就是我们需要创建`Dockerfile.dev`文件的所有内容。

### 构建镜像

一旦我们有了`Dockerfile.dev`，我们就可以构建镜像。我们可以在项目根目录级别通过以下命令进行构建：

```py
$ docker build -f Dockerfile.dev -t live-application .
```

由于我们命名了 Dockerfile 为`Dockerfile.dev`，我们应该在参数中指定它。一旦构建完成，你可以通过运行以下命令来检查镜像是否已正确构建：

```py
$ docker images live-application
```

你应该能在输出打印中看到镜像的详细信息，如下所示：

```py
REPOSITORY      TAG    IMAGE ID    CREATED        SIZE
live-application latest  7ada80a535c2 43 seconds ago 1.06GB
```

镜像构建完成后，我们可以继续创建容器。

### 创建容器

要创建容器并运行它，只需运行以下命令：

```py
$ docker run -p 8000:80 live-application
```

这将创建容器并运行它。我们可以通过运行以下命令来查看容器：

```py
$ docker ps -a
```

由于我们没有指定容器名称，它将自动赋予一个花哨的名称。例如，我的名称是`bold_robinson`。

打开浏览器到`http://localhost:8000`，你将看到我们应用程序的主页响应。

这就是你在 Docker 容器内运行 FastAPI 应用程序所需的所有内容。在 Docker 容器中运行 FastAPI 应用程序是利用两种技术优势的绝佳方式。你可以轻松地进行扩展、更新和部署你的 Web 应用程序，而配置量最小。

## 参见

Dockerfile 可以用来指定镜像的几个特性。请查看官方文档中的命令列表：

+   *Dockerfile* *参考*: [`docs.docker.com/reference/dockerfile/`](https://docs.docker.com/reference/dockerfile/)

此外，你还可以查看以下页面上的 Docker CLI 文档：

+   *Docker:* [`docs.docker.com/reference/cli/docker/`](https://docs.docker.com/reference/cli/docker/)

你可以在此链接查看 FastAPI 与 Docker 集成的专用文档页面：

+   *容器中的 FastAPI - Docker:* [`fastapi.tiangolo.com/deployment/docker/`](https://fastapi.tiangolo.com/deployment/docker/)

# 在多个工作进程中运行服务器

在高流量环境中，使用单个工作进程运行 FastAPI 应用程序可能不足以高效处理所有传入请求。为了提高性能并确保更好的资源使用，你可以在多个工作进程中运行你的 FastAPI 实例。这可以通过使用如**Gunicorn**等工具实现。

在这个配方中，我们将探讨如何使用 Gunicorn 在 Docker 容器中运行具有多个工作进程的 FastAPI 应用程序，我们还将讨论 Uvicorn 处理多个工作进程的能力及其局限性。

## 准备工作

`gunicorn`包与 Windows 不兼容。为了确保操作系统兼容性，我们将在我们的`Live Application`中运行 Docker 容器。

这个配方将基于上一个配方中创建的项目，*在 Docker 容器中运行 FastAPI 应用程序*。

## 如何操作…

使用多个工作进程的 FastAPI 会同时在不同的 CPU 进程中运行应用程序的多个副本。

为了更好地查看，让我们让端点显示`main.py`，添加以下行：

```py
import logging
from os import getpid
# rest of the module
logger = logging.getLogger("uvicorn")
# rest of the module
@app.get("/")
def read_root():
    logger.info(f"Processd by worker {getpid()}")
    return {"Hello": "World"}
```

让我们在`requirements.txt`文件中添加`gunicorn`依赖项，如下所示：

```py
fastapi
gunicorn instead of uvicorn to run the server.
If you are on Linux or macOS, you simply install `gunicorn` in your environment like this:

```

$ pip install gunicorn

```py

 Then, run the server with four workers with the following command:

```

$ gunicorn app.main:app --workers 4 \

--worker-class uvicorn.workers.UvicornWorker

```py

 If you are on Windows, we will use Docker. In the `Dockerfile.dev` file, add the new `CMD` instruction below the existing one, which will be ignored:

```

CMD ["gunicorn",\

"app.main:app",\

"--bind", "0.0.0.0:80",\

"--workers", "4",\

"--worker-class",\

"uvicorn.workers.UvicornWorker",\

"--log-level", "debug"]

```py

 Then, build the Docker image with the following:

```

$ docker build -t live-application-gunicorn \

-f Dockerfile.dev .

```py

 Next, run the container from the image:

```

$ docker run -p 8000:80 -i live-application-gunicorn

```py

 The `-i` parameter allows you to run the container in interactive mode to see the logs.
After the server is running, open the browser on `http://localhost:8000/docs` and use the interactive documentation to make calls. On the terminal output, you will notice different PIDs that vary for each call.
This shows that Gunicorn can distribute the load among different processes, and you can take advantage of multiple CPU cores.
You have learned how to run a FastAPI app with Gunicorn and multiple workers, which can improve the performance and scalability of your web service. You can experiment with different settings and options to find the optimal configuration for your needs.
Important note
You can run multiple workers with Uvicorn as well. However, Uvicorn’s worker process management is not as advanced as Gunicorn’s at the moment.
There’s more…
One of the benefits of running Gunicorn with multiple workers is that it can handle more concurrent requests and improve the performance and availability of the web application. However, there are also some challenges and trade-offs that come with this approach.
For example, when using multiple workers, each worker process has its own memory space and cannot share data with other workers. This means that any stateful components of the application, such as caches or sessions, need to be stored in a centralized or distributed service, such as Redis or Memcached. Moreover, multiple workers may increase resource consumption and the risk of contention on the server machine, especially if the application is CPU-intensive or input/output-bound. Therefore, it is important to choose the optimal number of workers based on the characteristics of the application and the available resources.
A common heuristic is to use the formula *workers = (2 x cores) + 1*, where *cores* means the number of CPU cores on the server. However, this may not be suitable for all scenarios and may require some experimentation and fine-tuning.
See also
You can discover more about Gunicorn in the official documentation at this link: 

*   *gunicorn:* [`gunicorn.org/`](https://gunicorn.org/)

Also, you can have a look at the page in the FastAPI documentation dedicated to server workers:

*   *Server Workers – Gunicorn with* *Uvicorn:* [`fastapi.tiangolo.com/deployment/server-workers/`](https://fastapi.tiangolo.com/deployment/server-workers/)

Deploying your FastAPI application on the cloud
Deploying your FastAPI application on the cloud is an essential step to make it accessible to users worldwide. In this recipe, we will demonstrate how to deploy a FastAPI application on Railway.
Railway is a versatile and user-friendly platform that enables developers to deploy, manage, and scale their applications with ease. By the end of the recipe, you will have a FastAPI application running on Railway, ready to serve users on the internet.
Getting started
Before we begin, ensure that you have already set up an application, as we will be deploying it on the cloud. The recipe will be applied to our `Live Application`, the basic application created in the *Running the server with the FastAPI* *CLI* recipe.
Also, put the project folder on GitHub, since it will be used as a reference for the deployment.
You will also need to set up an account at [`railway.app`](https://railway.app). The creation is straightforward, and you can use your GitHub account as well. When you sign up, you will receive a $5 credit, which is more than enough to cover the recipe.
How to do it…
We will demonstrate how to deploy the application on Railway through the following steps:

1.  Create the configuration file.
2.  Connect the Git repository.
3.  Configure the deployment.

    Although we will demonstrate it specifically for Railway, these steps are also common for other cloud services.

Creating the configuration file
Every deployment tool requires a configuration file that contains specifications for the deployment. To deploy on Railway, under our project root folder, let’s create a file called `Procfile`. The file content will be as follows:

```

web: fastapi run --port $PORT

```py

 Remember to push the file to the GitHub repository hosting your project to be visible to Railway.
Connecting the Git repository
Once the configuration file is set up, log in to Railway ([`railway.app/login`](https://railway.app/login)) with your account and you will be redirected to your dashboard ([`railway.app/dashboard`](https://railway.app/dashboard)).
Then, click on the `FastAPI-Cookbook` repository ([`github.com/PacktPublishing/FastAPI-Cookbook`](https://github.com/PacktPublishing/FastAPI-Cookbook)), you can select it.
Then select `profound-enchantment`.
Once finished, the *deployment* icon will appear on the project dashboard. By default, the deployment takes the name of the chosen GitHub repository. In my case, it’s `FastAPI-Cookbook`.
Configuring the deployment
When you click on the *deployment* icon, you can see a warning indicating that the deployment has failed. To resolve this, we need to add some parameters.
Click on the *deployment* icon, which will open a window on the left. Then, click on the **Settings** tab. This will display a list of configurations with sections such as **Source**, **Networking**, **Build**, and **Deploy**.
Begin with the `FastAPI-Cookbook` repository or if your project’s root directory is not the repository root, click on **Add Root Directory** under the **Source** repository specification and enter the path.
For the `FastAPI-Cookbook` repository, the path will be `/Chapter12/live_application`. After adding the path, click on the *save* icon.
Leave the branch selected as **main**.
Moving on to the `fastapi-cookbook-production.up.railway.app`. You will have a slightly different domain.
Leave the remaining settings as they are.
At the top left of the screen, you will see a text bar with the text **Apply 2 changes** with a **Deploy** button. Click on it to apply the modification we have done.
After the deployment process is complete, your application will begin to handle live web traffic. The public address is defined in the **Networking** section of the **Settings** panel.
Open the address in a new browser tab, and check the response. You should see the implemented response:

```

{

"Hello": "World"

}

```py

 In your web browser’s address bar, you can see a *lock* icon, which indicates that the connection is secure and has a certificate. Usually, when you expose your service to the web, the hosting platform provides you with certificates.
You have just deployed your FastAPI application to be accessible on the World Wide Web. Now, users from all over the world can access your service.
There’s more…
To deploy your service, Railway creates an image and then a container to run your service. You can specify a custom image with a Dockerfile and it will be automatically detected.
See also
You can discover more about Railway services on the official documentation website: 

*   *Railway* *Docs:* [`docs.railway.app/`](https://docs.railway.app/)

You can check the official FastAPI template used for Railway at this link: 

*   *FastAPI* *Example:* [`github.com/railwayapp-templates/fastapi`](https://github.com/railwayapp-templates/fastapi)

FastAPI is one of the fastest-growing production applications, especially on the major public cloud service providers. That’s why you can find extensive documentation on how to use it:
For **Google Cloud Platform** (**GCP**), you can follow the article at the link: 

*   *Deploying FastAPI app with Google Cloud Run* article at the following link: [`dev.to/0xnari/deploying-fastapi-app-with-google-cloud-run-13f3`](https://dev.to/0xnari/deploying-fastapi-app-with-google-cloud-run-13f3)

For **Amazon Web Services** (**AWS**), check this Medium article:

*   *Deploy FastAPI on AWS* *EC2*: [`medium.com/@shreyash966977/deploy-fastapi-on-aws-ec2-quick-and-easy-steps-954d4a1e4742`](https://medium.com/@shreyash966977/deploy-fastapi-on-aws-ec2-quick-and-easy-steps-954d4a1e4742)

For Microsoft Azure, you can check the official documentation page:

*   *Using FastAPI Framework with Azure* *Functions*: [`learn.microsoft.com/en-us/samples/azure-samples/fastapi-on-azure-functions/fastapi-on-azure-functions/`](https://learn.microsoft.com/en-us/samples/azure-samples/fastapi-on-azure-functions/fastapi-on-azure-functions/)

On the FastAPI website, you can check other examples for other cloud providers at the following link:

*   *Deploy FastAPI on Cloud* *Providers*: [`fastapi.tiangolo.com/deployment/cloud/`](https://fastapi.tiangolo.com/deployment/cloud/)

A useful tool is the Porter platform, which allows you to deploy your applications on different cloud services such as AWS, GCP, and Azure from one centralized platform. Have a look at this link:

*   *Deploy a FastAPI* *app:* [`docs.porter.run/guides/fastapi/deploy-fastapi`](https://docs.porter.run/guides/fastapi/deploy-fastapi)

Shipping FastAPI applications with Hatch
Packaging and shipping a FastAPI application as a distributable package are essential for deploying and sharing your application efficiently.
**Hatch** is a modern Python project management tool that simplifies the packaging, versioning, and distribution process. In this recipe, we’ll explore how to use Hatch to build and ship a package containing a FastAPI application. This will ensure that your application is portable, easy to install, and maintainable, making it easier to deploy and share with others.
Getting ready
Hatch facilitates the use of multiple virtual environments for our project. It uses the `venv` package under the hood.
To run the recipe, you need to install Hatch on your local machine. The installation process may vary depending on your operating system. Detailed instructions can be found on the official documentation page: [`hatch.pypa.io/1.9/install/`](https://hatch.pypa.io/1.9/install/).
Once the installation is complete, verify that it has been correctly installed by running the following from the command-line terminal:

```

$ hatch --version

```py

 You should have the version printed on the output like this:

```

Hatch, version 1.11.1

```py

 Make sure that you installed a version higher than `1.11.1`. We can then start creating our package.
How to do it…
We divide the process of shipping our FastAPI package into five steps:

1.  Initialize the project.
2.  Install dependencies.
3.  Create the app.
4.  Build the distribution.
5.  Test the package.

Let’s start building our package.
Initializing the project
We start by creating our project by bootstrapping the structure. Let’s call our application `FCA`, which stands for **FastAPI Cookbook Application**. Let’s bootstrap our project by running the following command:

```

$ hatch new "FCA Server"

```py

 The command will create a project bootstrap under the `fca-server` folder as follows:

```

fca-server

├──src

│  └── fca_server

│      ├── __about__.py

│      └── __init__.py

├──tests

│  └── __init__.py

├──LICENSE.txt

├──README.md

└──pyproject.tomt

```py

 We can then directly use a virtual environment by entering the `fca-server` directory and running the following:

```

$ hatch shell

```py

 The command will automatically create a default virtual environment and activate it. You will see your command-line terminal with a prepend value, `(fca-server)`, like so:

```

(fca-server) path/to/fca-server $

```py

 Verify that the environment is correctly activated by checking the Python executable. You do it by running the following:

```

$ python -c "import sys; print(sys.executable)"

```py

 The executable should come from the virtual environment called `fca-server`, which will present a path such as `<virtual` `environment locations>\fca-server\Scripts\python`.
This will give you information on the virtual environment that you can also provide to your **integrated development environment** (**IDE**) to work with the code.
You can exit from the shell by typing `exit` in the terminal. Also, you can run commands in the virtual environment without spawning the shell. For example, you can check the Python executable of the default environment by running the following:

```

$ hatch run python -c "import sys; print(sys.executable)"

```py

 We can now proceed to install the package dependencies in our environment.
Installing dependencies
Now that you have created a virtual environment, let’s add the `fastapi` dependency to our project. We can do it by modifying the `pyproject.toml` file. Add it in the `dependencies` field under the `[project]` section like so:

```

[project]

...

dependencies = [

"fastapi"

]

```py

 Next time you spawn a shell, the dependencies will synchronized and the `fastapi` package will be installed.
Let’s see, for example, whether the `fastapi` command works by running the following:

```

$ hatch run fastapi --help

```py

 If you see the help documentation of the command, the dependency has been added correctly.
Creating the app
Now that we have the environment with the `fastapi` package installed, we can develop our application.
Let’s create the `main.py` module under the `src/fca_server` folder and initialize the `APIRouter` object with one endpoint like this:

```

from fastapi import APIRouter

app = APIRouter()

@app.get("/")

def read_root():

return {

"message":

"欢迎来到 FastAPI 食谱应用程序！"

}

```py

 Then, let’s import the router into the `src/fca_server.__init__.py` file as follows:

```

from fca_server.main import router

```py

 This will allow us to directly import the router from the `fca_server` package from an external project.
Building the distribution
Now that we have finalized the package, let’s leverage Hatch to build the package distribution.
We will generate the package in the form of a `.tar.gz` file by running the following:

```

$ hatch build -t sdist ../dist

```py

 It will generate the `fca_server-0.0.1.tar.gz` file placed outside of the project in a `dist` folder. We will then use the file in an external project.
Testing the package
Next, we will make a different project that uses the `fca_server` package we made.
Create an `import-fca-server` folder outside of the `fca-server` folder for the package and use it as the project root folder.
In the folder, make a local virtual environment with `venv` by running the following:

```

$ python -m venv .venv

```py

 Activate the environment. On Linux or macOS, type the following:

```

$ source .venv/Scripts/activate

```py

 On Windows, type this instead:

```

$ .venv\Scripts\activate

```py

 Install the `fca_server` package with `pip`:

```

$ pip install ..\dist\fca_server-0.0.1.tar.gz

```py

 Use the path where the `fca_server-0.0.1.tar.gz` file is.
Now, try to import the package.
Make a `main.py` file and import the router from the `fca_server` package:

```

from fastapi import FastAPI

from fca_server import router

app = FastAPI(

title="Import FCA Server Application"

)

app.include_router(router)

```py

 Run the server from the command line:

```

$ fastapi run

```py

 Go to the interactive documentation at `http://localhost:8000/docs` and see the endpoint in the external package. You have just created a custom package and imported it into another project.
You have learned how to use Hatch to create and manage your Python projects with ease. This is a powerful tool that can save you time and effort and help you write better code. Now, you can experiment with different options and features of Hatch and see what else you can do with it.
There’s more…
Hatch is a versatile packaging system for Python that allows you to create scripts and multiple environments for your projects.
With Hatch, you can also customize the location of the virtual environment files, such as whether you want them to be centralized or in the project folder. You can specify this option in the `config.toml` file, which contains the configuration settings for Hatch.
To find the location of the `config.toml` file, you can run the following command in your terminal:

```

$ hatch config find

```py

 Hatch also lets you create the build of your package in a wheel format, which is a binary distribution format that is more efficient and compatible than the traditional source distribution.
Moreover, you can publish your package directly to the **Python Package Index** (**PyPI**), where other users can find and install it. Hatch makes it easy to share your code with the world.
See also
You can find more information about Hatch in the official documentation at 

*   *Hatch*: [`hatch.pypa.io/latest/`](https://hatch.pypa.io/latest/)

We learned how to create a project bootstrap, but with Hatch, you can also initialize an existing project. Check out the documentation page: 

*   *Existing project:* https://hatch.pypa.io/1.9/intro/#existing-project 

One of the greatest advantages of using Hatch is the flexibility of running the project for several virtual environments. Check more on the documentation page: 

*   *Environments:* [`hatch.pypa.io/1.9/environment/`](https://hatch.pypa.io/1.9/environment/)

The `pyproject.toml` file is a configuration file for Python projects, introduced in `PEP 518` ([`peps.python.org/pep-0518/`](https://peps.python.org/pep-0518/)). It aims to standardize and simplify the configuration of Python projects by providing a single place to specify build system requirements and other project metadata. It is used by other build tools. You can have a look at the Python Package User Guide page at the following link: 

*   *Writing your* *pyproject.toml:* [`packaging.python.org/en/latest/guides/writing-pyproject-toml/`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

You can see more on how to manage Python dependencies on this page: 

*   *Dependency* *configuration:* [`hatch.pypa.io/dev/config/dependency/`](https://hatch.pypa.io/dev/config/dependency/)

```
