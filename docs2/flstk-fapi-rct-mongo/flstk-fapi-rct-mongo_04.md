

# 第四章：快速入门 FastAPI

**应用程序编程接口**（**API**）是您的 FARM 堆栈的基石，作为系统的“大脑”。它实现了业务逻辑，决定了数据如何进出系统，但更重要的是，它如何与系统内的业务需求相关联。

如同 FastAPI 这样的框架，通过示例更容易展示。在本章中，您将探索一些简单的端点，这些端点构成了一个最小、自包含的 REST API。这些示例将帮助您了解 FastAPI 如何处理请求和响应。

本章重点介绍该框架，以及标准 REST API 实践及其在 FastAPI 中的实现。您将学习如何发送请求并根据您的需求修改它们，以及如何从 HTTP 请求中检索所有数据，包括参数和请求体。您还将了解如何处理响应，以及您如何可以使用 FastAPI 轻松设置 cookies、headers 和其他标准网络相关主题。

本章将涵盖以下主题：

+   FastAPI 框架概述

+   简单 FastAPI 应用的设置和需求

+   FastAPI 中的 Python 特性，例如类型提示、注解和`async/await`语法

+   FastAPI 如何处理典型的 REST API 任务

+   处理表单数据

+   FastAPI 项目的结构和路由

# 技术要求

对于本章，您需要以下内容：

+   Python 设置

+   虚拟环境

+   代码编辑器和插件

+   REST 客户端

以下部分将更详细地介绍这些要求。

## Python 设置

如果您还没有安装 Python，请访问 Python 下载网站（[`www.python.org/downloads/`](https://www.python.org/downloads/））以获取您操作系统的安装程序。在本书中，您将使用**版本** **3.11.7**或更高版本。

FastAPI 严重依赖于 Python 提示和注解，Python 3.6 之后的版本以类似现代的方式处理类型提示；因此，虽然理论上任何高于 3.6 的版本都应该可以工作，但本书中的代码使用 Python 版本 3.11.7，出于兼容性的原因。

确保您的 Python 安装已升级到最新的 Python 版本之一——如前所述，至少为版本 3.11.7——并且是可访问的且是默认版本。您可以通过以下方式进行检查：

+   在您选择的终端中键入`python`。

+   使用**pyenv**，一个方便的工具，可以在同一台机器上管理多个 Python 版本。

## 虚拟环境

如果您之前曾经参与过 Python 项目，那么您可能需要包含一些，如果不是几十个，Python 第三方包。毕竟，Python 的主要优势之一在于其庞大的生态系统，这也是它被选为 FARM 堆栈的主要原因之一。

不深入探讨 Python 如何管理第三方包的安装细节，让我们先概述一下，如果你决定为所有项目仅使用一个 Python 安装，或者更糟糕的是，如果这个安装是默认操作系统的 Python 安装，可能会出现的主要问题。

下面是一些挑战：

+   操作系统在 Python 版本方面通常滞后，所以最新的几个版本可能不可用。

+   包将安装到相同的命名空间或相同的包文件夹中，这会在任何依赖于该包的应用程序或包中造成混乱。

+   Python 包依赖于其他包，这些包也有版本。假设你正在使用包 A，它依赖于包 B 和 C，并且由于某种原因，你需要将包 B 保持在一个特定的版本（即 1.2.3）。你可能需要包 B 用于完全不同的项目，而这个项目可能需要不同的版本。

+   减少或无法复现：没有单独的 Python 虚拟环境，将很难快速复制所有必需的包所需的功能。

Python 虚拟环境是解决上述问题的解决方案，因为它们允许你在一个纯净的 Python 开发环境中工作，只包含你需要的包和包版本。在我们的例子中，虚拟环境将肯定包括核心包：FastAPI 和 Uvicorn。另一方面，FastAPI 依赖于 Starlette、Pydantic 等，因此控制包版本非常重要。

Python 开发的最佳实践指出，无论项目大小如何，每个项目都应该有自己的虚拟环境。虽然有多种创建虚拟环境的方法，它是一个分离和独立的 Python 环境，但你将使用`virtualenv`。

使用`virtualenv`创建新虚拟环境的基本语法如下所示。一旦你处于项目文件夹中，将你的文件夹命名为`FARM`或`chapter4`，打开一个终端，并输入以下命令：

```py
python – m venv venv
```

此命令将为你的项目创建一个新的虚拟环境，Python 解释器的副本（或者在 macOS 上，一个全新的 Python 解释器），必要的文件夹结构，以及一些激活和停用环境的命令，以及`pip`安装程序的副本（pip 用于安装包）。

为了激活你的新虚拟环境，你将根据你的操作系统选择以下命令之一。对于 Windows 系统，在 shell 中输入以下内容：

```py
venv/Scripts/activate
```

在 Linux 或 macOS 系统上，使用以下命令：

```py
source venv/bin/activate
```

在这两种情况下，你的 shell 现在应该以你为环境所取的名字作为前缀。在创建新虚拟环境的命令中，最后一个参数是环境名称，所以在这个例子中是`venv`。

在使用虚拟环境时，以下是一些需要考虑的事项：

+   在虚拟环境放置方面，存在不同的观点。目前，如果你像之前那样将它们保存在项目文件夹内就足够了。

+   与 `activate` 命令类似，还有一个 `deactivate` 命令可以退出你的虚拟环境。

+   在 `requirements.txt` 文件中保存确切的包版本并固定依赖项不仅有用，而且在部署时通常是必需的。

Python 社区中有许多 `virtualenv` 的替代方案，以及许多互补的包。Poetry 是一个同时管理虚拟环境和依赖项的工具，`virtualenvwrapper` 是一组进一步简化环境管理过程的实用工具。`pyenv` 稍微复杂一些——它管理 Python 版本，并允许你根据不同的 Python 版本拥有不同的虚拟环境。

## 代码编辑器

虽然有许多优秀的 Python 代码编辑器和 **集成开发环境**（**IDE**），但一个常见的选择是微软的 **Visual Studio Code**（**VS Code**）。2015 年发布，它是跨平台的，提供了许多集成工具，例如用于运行开发服务器的集成终端。它轻量级，提供了数百个插件，几乎可以满足你任何编程任务的需求。由于你将使用 JavaScript、Python、React 和 CSS 进行样式设计，以及运行命令行进程，因此使用 VS Code 是最简单的方法。

也有一个名为 **MongoDB for VS Code** 的优秀 MongoDB 插件，它允许你连接到 MongoDB 或 Atlas 集群，浏览数据库和集合，快速查看模式索引，以及查看集合中的文档。这在全栈场景中非常有用，当你发现自己正在处理 Python 的后端代码、JavaScript 和 React 或 Next.js 的前端代码、运行外壳，并需要快速查看 MongoDB 数据库的状态时。扩展程序可在以下链接找到：[https://marketplace.visualstudio.com/items?itemName=mongodb.mongodb-vscode](https://marketplace.visualstudio.com/items?itemName=mongodb.mongodb-vscode)。你还可以在 Visual Studio Code 的 **扩展** 选项卡中通过搜索 MongoDB 来安装它。

## 终端

除了 Python 和 Git 之外，你还需要一个外壳程序。Linux 和 Mac 用户通常已经预装了一个。对于 Windows，你可以使用 Windows PowerShell 或像 **Cmder** ([`cmder.app`](https://cmder.app/)) 这样的控制台模拟器，它提供了额外的功能。

## REST 客户端

为了有效地测试您的 REST API，您需要一个 REST 客户端。虽然**Postman**([`www.postman.com/`](https://www.postman.com/))功能强大且可定制，但还有其他可行的替代方案。**Insomnia**()和 REST GUI 提供了一个更简单的界面，而**HTTPie**()，一个命令行 REST API 客户端，允许在不离开 shell 的情况下快速测试。它提供了诸如表达性语法、表单和上传处理以及会话等功能。

HTTPie 可能是安装最简单的 REST 客户端，因为它可以使用`pip`或其他包管理器，如 Chocolatey、apt（用于 Linux）或 Homebrew。

安装 HTTPie 的最简单方法是激活您的虚拟环境并使用`pip`，如下面的命令所示：

```py
pip install httpie
```

安装完成后，您可以使用以下命令测试 HTTPie：

```py
(venv) http GET "http://jsonplaceholder.typicode.com/todos/1"
```

输出应该以`HTTP/1.1 200` `OK`响应开始。

`venv`表示虚拟环境已激活。HTTPie 通过简单地添加`POST`来简化 HTTP 请求，包括有效载荷、表单值等。

## 安装必要的包

在设置虚拟环境之后，您应该激活它并安装运行第一个简单应用程序所需的 Python 库：FastAPI 和 Uvicorn。

为了使 FastAPI 运行，它需要一个服务器。在这种情况下，服务器是一种用于提供 Web 应用程序（或 REST API）的软件。FastAPI 依赖于**异步服务器网关接口**（**ASGI**），它使异步非阻塞应用程序成为可能，这是您可以完全利用 FastAPI 功能的地方。您可以在以下文档中了解更多关于 ASGI 的信息：[`asgi.readthedocs.io/`](https://asgi.readthedocs.io/)。

目前，FastAPI 文档列出了三个兼容 Python ASGI 的服务器：**Uvicorn**、**Hypercorn**和**Daphne**。本书将重点介绍 Uvicorn，这是与 FastAPI 一起使用最广泛和推荐的选择。Uvicorn 提供高性能，如果您遇到困难，网上有大量的文档可供参考。

要安装前两个依赖项，请确保您位于工作目录中，并激活了所需的虚拟环境，然后执行以下命令：

```py
pip install fastapi uvicorn
```

现在，您拥有了一个包含 shell、一个或两个 REST 客户端、一个优秀的编辑器和优秀的 REST 框架的 Python 编码环境。如果您之前开发过**Django**或**Flask**应用程序，这些都应该很熟悉。

最后，选择一个文件夹或克隆这本书的 GitHub 仓库，并激活一个虚拟环境。通常，在工作目录中创建一个名为`venv`的文件夹来创建环境，但请随意根据您的喜好来组织您的目录和代码。

在此之后，本章将简要讨论一些结构化您的 FastAPI 代码的选项。现在，请确保您在一个已激活新创建的虚拟环境的文件夹中。

# 快速了解 FastAPI

在 *第一章*，*Web 开发和 FARM 栈*中，提到了为什么 FastAPI 是 FARM 栈中首选的 REST 框架。使 FastAPI 独特的是其编码速度和由此产生的干净代码，这使得你可以快速发现并修复错误。该框架的作者 *Sebastian Ramirez* 经常谦逊地强调，FastAPI 只是 Starlette 和 Pydantic 的混合，同时大量依赖现代 Python 特性，特别是类型提示。

在深入示例和构建 FastAPI 应用程序之前，快速回顾 FastAPI 所基于的框架是有用的。

## Starlette

Starlette 是一个以高性能和众多特性著称的 ASGI 框架，这些特性在 FastAPI 中也有提供。这些包括 WebSocket 支持、启动和关闭时的事件、会话和 Cookie 支持、后台任务、中间件实现和模板。虽然你不会直接在 Starlette 中编码，但了解 FastAPI 内部的工作原理及其起源是很重要的。

如果你对其功能感兴趣，请访问 Starlette 优秀的文档（https://www.starlette.io/）。

## 异步编程

你可能在学习使用 Node.js 开发应用程序时已经接触过异步编程范式。这涉及到执行慢速操作，例如网络调用和文件读取，使得系统可以在不阻塞的情况下响应其他请求。这是通过使用事件循环，一个异步任务管理器来实现的，它允许系统将请求移动到下一个，即使前一个请求尚未完成并返回响应。

Python 在 3.4 版本中增加了对异步 I/O 编程的支持，并在 3.6 版本中引入了 `async`/`await` 关键字。ASGI 在 Python 世界中随后出现，概述了应用程序应该如何构建和调用，并定义了可以发送和接收的事件。FastAPI 依赖于 ASGI 并返回一个 ASGI 兼容的应用程序。

在这本书中，所有端点函数都带有 `async` 关键字前缀，甚至在它们成为必要之前，因为你会使用异步的 Motor Python MongoDB 驱动程序。

注意

如果你正在开发一个不需要高压力的简单应用程序，你可以使用简单的同步代码和官方的 PyMongo 驱动程序。

带有 `async` 关键字的函数是协程；它们在事件循环上运行。虽然本章中的简单示例可能不需要 `async` 就能工作，但当你通过一个异步驱动程序，如 **Motor** (https://motor.readthedocs.io/en/stable/)，连接到你的 MongoDB 服务器时，FastAPI 中异步编程的真正力量将变得明显。

## 标准的 REST API 操作

本节将讨论 API 开发中的一些常见术语。通常，通信通过 HTTP 协议进行，通过 HTTP 请求和响应。你将探索 FastAPI 如何处理这些方面，并利用 Pydantic 和类型提示等额外库来提高效率。在示例中，你将使用 Uvicorn 作为服务器。

任何 REST API 通信的基础是一个 URL 和路径的系统。你的本地 Web 开发服务器的 URL 将是`http://localhost:8000`，因为`8000`是 Uvicorn 使用的默认端口。端点的路径部分（可选）可以是`/cars`，而`http`是方案。你将看到 FastAPI 如何处理路径、查询字符串、请求和响应正文，定义端点函数的特定顺序的重要性，以及如何有效地从动态路径段中提取变量。

在每个路径或地址中，URL 和路径的组合，都有一组可以执行的操作—HTTP 动词。例如，一个页面或 URL 可能列出所有待售的汽车，但你不能发出`POST`请求，因为这不被允许。

在 FastAPI 中，这些动词作为 Python**装饰器**实现。换句话说，它们被公开为装饰器，并且只有当你，即开发者，实现它们时，它们才会被实现。

FastAPI 鼓励正确和语义化地使用 HTTP 动词进行数据资源操作。例如，在创建新资源时，你应该始终使用`POST`（或`@post`装饰器），对于读取数据（单个或项目列表），使用`GET`，对于更新使用`PATCH`等等。

HTTP 消息由请求/状态行、头部和可选的正文数据组成。FastAPI 提供了工具，可以轻松创建和修改头部、设置响应代码以及以干净直观的方式操作请求和响应正文。

本节描述了支撑 FastAPI 性能的编程概念和特定的 Python 特性，使代码易于维护。在下一节中，你将了解标准的 REST API 操作，并了解它们是如何通过 FastAPI 实现的。

## FastAPI 是如何表达 REST 的？

观察一个最小的 FastAPI 应用程序，例如经典的**Hello World**示例，你可以开始检查 FastAPI 如何构建端点。在这个上下文中，端点指定以下详细信息：

+   一个独特的 URL 组合：这将在你的开发服务器中保持一致—`localhost:8000`。

+   路径：斜杠后面的部分。

+   HTTP 方法。

例如，在名为`Chapter4`的新文件夹中，使用 Visual Studio Code 创建一个名为`chapter4_01.py`的新 Python 文件：

```py
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello FastAPI"}
```

使用这段代码，你可以完成几件事情。以下是每个部分的作用分解：

+   在`chapter4_01.py`的第一行中，你从`fastapi`包中导入了 FastAPI 类。

+   接下来，你实例化了一个应用程序对象。这只是一个具有所有 API 功能并暴露一个 ASGI 兼容应用程序的 Python 类，这个应用程序必须传递给 Uvicorn。

现在，应用程序已经准备就绪并实例化。但没有端点，它无法做或说很多。它有一个端点，即根端点，你可以在 `http://127.0.0.1:8000/` 上查看。FastAPI 提供了用于 HTTP 方法的装饰器，以告诉应用程序如何以及是否响应。然而，你必须实现它们。

之后，你使用了 `@get` 装饰器，它对应于 `GET` 方法，并传递了一个 URL——在这种情况下，使用了根路径 `/`。

装饰的函数，命名为 `root`，负责响应请求。它接受任何参数（在这种情况下，没有参数）。函数返回的值，通常是 Python 字典，将由 ASGI 服务器转换为 **JavaScript 对象表示法**（**JSON**）响应，并作为 HTTP 响应返回。这看起来可能很显然，但将其分解以了解基础知识是有用的。

上述代码定义了一个具有单个端点的完整功能应用程序。要测试它，你需要一个 Uvicorn 服务器。现在，你必须使用 Uvicorn 在你的命令行中运行实时服务器：

```py
uvicorn chapter4_01:app --reload
```

当你使用 FastAPI 进行开发时，你将非常频繁地使用此代码片段，所以以下说明将对其进行分解。

注意

Uvicorn 是你的 ASGI 兼容的 Web 服务器。你可以通过传递可执行 Python 文件（不带扩展名）和实例化应用（FastAPI 实例）的组合（由冒号 `:` 分隔）来直接调用它。`--reload` 标志指示 Uvicorn 在你保存代码时每次重新加载服务器，类似于 Node.js 中的 *Nodemon*。除非指定其他方式，否则你可以使用此语法运行本书中包含 FastAPI 应用的所有示例。

这是使用 HTTPie 测试唯一端点时的输出。记住，当你省略方法的关键字时，它默认为 `GET` 请求：

```py
(venv) http http://localhost:8000/
HTTP/1.1 200 OK
content-length: 27
content-type: application/json date: Fri, 01 Apr 2022 17:35:48 GMT
server: uvicorn
{
  "message": "Hello FastAPI"
}
```

HTTPie 通知你你的简单端点正在运行。你将获得 `200 OK` 状态码，`content-type` 设置正确为 `application/json`，并且响应是一个包含所需消息的 JSON 对象。

每个 REST API 指南都以类似的 *hello world* 示例开始，但使用 FastAPI，这尤其有用。只需几行代码，你就可以看到简单端点的结构。这个端点仅覆盖针对根 URL (`/`) 的 `GET` 方法。因此，如果你尝试使用 `POST` 请求测试此应用，你应该会收到 `405 Method Not Allowed` 错误（或任何非 `GET` 方法）。

如果你想要创建一个对 `POST` 请求返回相同消息的端点，你只需更改装饰器。将以下代码添加到文件末尾（`chapter4_01.py`）：

```py
@app.post("/")
async def post_root():
    return {"message": "Post request success!"}
```

HTTPie 将在终端中相应地响应：

```py
(venv) http POST http://localhost:8000 HTTP/1.1 200 OK
content-length: 35
content-type: application/json date: Sat, 26 Mar 2022 12:49:25 GMT
server: uvicorn
{
    "message": "Post request success!"
}
```

现在你已经创建了一些端点，请转到 `http://localhost:8000/docs`，看看 FastAPI 为你生成了什么。

## 自动文档

在开发 REST API 时，你会发现你需要不断执行 API 调用——`GET` 和 `POST` 请求——分析响应，设置有效载荷和头信息，等等。选择一个可行的 REST 客户端在很大程度上是一个个人喜好问题，这是一件应该仔细考虑的事情。虽然市场上有很多客户端——从功能齐全的 API IDE，如 Postman ([`www.postman.com/`](https://www.postman.com/))，到稍微轻量级的 Insomnia ([`insomnia.rest/`](https://insomnia.rest/)) 或 Visual Studio Code 的 REST 客户端 ([`marketplace.visualstudio.com/items?itemName=humao.rest-client`](https://marketplace.visualstudio.com/items?itemName=humao.rest-client))——本书主要使用非常简单的基于命令行的 HTTPie 客户端，它提供了一个简约的命令行界面。

然而，这正是介绍 FastAPI 最受欢迎的另一个特性的正确时机——交互式文档——这是一个有助于在 FastAPI 中开发 REST API 的工具。

随着你开发的每个端点或路由器，FastAPI 会自动生成文档。它是交互式的，允许你在开发过程中测试你的 API。FastAPI 列出你定义的所有端点，并提供有关预期输入和响应的信息。该文档基于 OpenAPI 规范，并大量依赖于 Python 提示和 Pydantic 库。它允许设置要发送到端点的 JSON 或表单数据，显示响应或错误，与 Pydantic 紧密耦合，并且能够处理简单的授权程序，例如将在 *第六章**，认证和授权* 中实现的携带令牌流。你无需使用 REST 客户端，只需打开文档，选择要测试的端点，方便地将测试数据输入到标准网页中，然后点击 **提交** 按钮！

在本节中，你创建了一个最小化但功能齐全的 API，具有单个端点，让你了解了应用程序的语法和结构。在下一节中，你将了解 REST API 请求-响应周期的基本元素以及如何控制过程的每个方面。标准 REST 客户端提供了一种更可移植的体验，并允许你比较不同的 API，即使它们不是基于 Python 的。

# 构建展示 API

REST API 围绕 HTTP 请求和响应展开，这些是网络的动力，并且在每个使用 HTTP 协议的 Web 框架中实现。为了展示 FastAPI 的功能，你现在将创建简单的端点，专注于实现所需功能的特定代码部分。而不是常规的 CRUD 操作，接下来的部分将专注于检索和设置请求和响应元素的过程。

## 获取路径和查询参数

第一个端点将用于通过其唯一的 ID 获取一个虚构的汽车。

1.  创建一个名为 `chapter4_02.py` 的文件，并插入以下代码：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/car/{id}")
    async def root(id):
        return {"car_id": id}
    car/:id, while {id} is a standard Python string-formatted dynamic parameter in the sense that it can be anything—a string or a number since you haven’t used any hinting.
    ```

1.  尝试一下，并使用等于 `1` 的 ID 来测试端点：

    ```py
    (venv) http "http://localhost:8000/car/1"
    HTTP/1.1 200 OK
    content-length: 14
    content-type: application/json date: Mon, 28 Mar 2022 20:31:58 GMT
    server: uvicorn
    {
        "car_id": "1"
    }
    ```

1.  你收到了你的 JSON 响应，但在这里，响应中的 `1` 是一个字符串（提示：引号）。你可以尝试用等于字符串的 ID 来执行相同的路由：

    ```py
    (venv) http http://localhost:8000/car/billy HTTP/1.1 200 OK
    {
        "car_id": "billy"
    }
    ```

    FastAPI 返回你提供的字符串，它是作为动态参数的一部分提供的。然而，Python 的新特性，如类型提示，也派上用场。

1.  回到你的 FastAPI 路径（或端点），使汽车 ID 成为整数，只需对变量参数的类型进行提示即可。端点将看起来像这样：

    ```py
    @app.get("/carh/{id}")
    async def hinted_car_id(id: int):
        return {"car_id": id}
    ```

你已经给它指定了一个新的路径：`/carh/{id}`（`car` 后面的 `h` 表示提示）。除了函数名（`hinted_car_id`）外，唯一的区别在于参数：紧跟在 `int` 后面的分号表示你可以期望一个整数，但 FastAPI 对此非常认真，你已经在框架中看到了如何很好地使用提示系统。

如果你查看 `http://localhost:8000/docs` 上的交互式文档，并尝试在 `/carh/` 端点的 `id` 字段中插入一个字符串，你会得到一个错误。

现在，在你的 REST 客户端中尝试运行它，并通过传递一个字符串来测试 `/carh/` 路径。首先，FastAPI 为你正确地设置了状态码——即 `422 Unprocessable Entity`——并在响应体中指出问题所在——值不是一个有效的整数。它还告知你错误发生的确切位置：在 `id` 路径中。

这是一个简单的例子，但想象一下你正在发送一个复杂的请求，路径复杂，有多个查询字符串，也许还有头部中的附加信息。使用类型提示可以快速解决这些问题。

如果你尝试访问端点而不指定任何 ID，你将得到另一个错误：

```py
(venv) http http://localhost:8000/carh/ HTTP/1.1 404 Not Found
{
    "detail": "Not Found"
}
```

FastAPI 再次正确地设置了状态码，给你一个 `404 Not Found` 错误，并在响应体中重复了此消息。你访问的端点不存在；你必须在斜杠后指定一个值。

可能会出现你拥有类似路径的情况：既有动态路径也有静态路径。一个典型的情况是拥有众多用户的应用程序。将 API 定向到由 `/users/id` 定义的 URL 将会给你一些关于选定 ID 的用户信息，而 `/users/me` 通常是一个显示你的信息并允许你以某种方式修改它的端点。

在这种情况下，重要的是要记住，与其他 Web 框架一样，顺序很重要。由于路径处理程序声明的顺序，以下代码将不会产生预期的结果，因为应用程序会尝试将 `/me` 路径与它遇到的第一个端点匹配——需要 ID 的那个端点——由于 `/me` 部分不是一个有效的 ID，你会得到一个错误。

创建一个名为 `chapter4_03.py` 的新文件，并将以下代码粘贴进去：

```py
from fastapi import FastAPI
app = FastAPI()
@app.get("/user/{id}")
async def user(id: int):
    return {"User_id": id}
@app.get("/user/me")
async def me_user():
    return {"User_id": "This is me!"}
```

当你运行应用程序并测试 `/user/me` 端点时，你将得到一个与之前相同的 `422 Unprocessable Entity` 错误。一旦你记住顺序很重要——FastAPI 会找到第一个匹配的 URL，检查类型，并抛出错误。如果第一个匹配的是具有固定路径的那个，那么一切都会按预期工作。只需更改两个路由的顺序，一切就会按预期工作。

FastAPI 对路径处理的一个强大功能是它如何限制路径到一组特定的值和一个从 FastAPI 导入的路径函数，这使你能够在路径上执行额外的验证。

假设你想要一个 URL 路径，它接受两个值并允许以下操作：

+   `account_type`：可以是 `free` 或 `pro`。

+   `months`：这必须是一个介于 3 和 12 之间的整数。

FastAPI 通过让你创建一个基于 `Enum` 的类来解决这个问题，用于账户类型。这个类定义了账户变量所有可能的值。在这种情况下，只有两个——`free` 和 `pro`。创建一个新的文件，并将其命名为 `chapter4_04.py`，然后编辑它：

```py
from enum import Enum
from fastapi import FastAPI, Path
app = FastAPI()
class AccountType(str, Enum):
    FREE = "free"
    PRO = "pro"
```

最后，在实际的端点中，你可以将这个类与 `Path` 函数的实用工具结合起来（不要忘记与 FastAPI 一起从 `fastapi` 导入它）。将以下代码粘贴到文件的末尾：

```py
@app.get("/account/{acc_type}/{months}")
async def account(acc_type: AccountType, months: int = Path(..., ge=3, le=12)):
    return {"message": "Account created", "account_type": acc_type, "months": months}
```

在前面的代码中，FastAPI 将路径的 `acc_type` 部分的类型设置为之前定义的类，并确保只能传递 `free` 或 `pro` 值。然而，`months` 变量是由 `Path` 实用函数处理的。当你尝试访问这个端点时，`account_type` 将显示只有两个值可用，而实际的枚举值可以通过 `.value` 语法访问。

FastAPI 允许你使用标准的 Python 类型声明路径参数。如果没有声明类型，FastAPI 将假设你正在使用字符串。

关于这些主题的更多详细信息，你可以访问优秀的文档网站，看看其他可用的选项（https://fastapi.tiangolo.com/tutorial/path-params/）。在这种情况下，`Path` 函数接收了三个参数。三个点表示该值是必需的，并且没有提供默认值，`ge=3` 表示该值可以大于或等于 `3`，而 `le=12` 表示它可以小于或等于 `12`。这种语法允许你在路径函数中快速定义验证。

### 查询参数

现在你已经学会了如何验证、限制和正确排序你的路径参数和端点，是时候看看**查询参数**了。这些参数是通过 URL 将数据传递给服务器的简单机制，它们以键值对的形式表示，由等号（=）分隔。你可以有多个键值对，由与号（&）分隔。

查询参数通过在 URL 的末尾使用问号/等号记法添加：`?min_price=2000&max_price=4000`。

问号（`?`）是一个分隔符，它告诉您查询字符串从哪里开始，而与号（`&`）允许您添加多个（等号`=`）赋值。

查询参数通常用于应用过滤器、排序、排序或限制查询集、分页长列表的结果以及类似任务。FastAPI 将它们处理得与路径参数非常相似，因为它会自动提取它们并在您的端点函数中使它们可用于处理。

1.  创建一个简单的端点，接受两个查询参数，用于汽车的最低价和最高价，并将其命名为`chapter4_05.py`：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/cars/price")
    async def cars_by_price(min_price: int = 0, max_price: int = 100000):
        return {"Message": f"Listing cars with prices between {min_price} and {max_price}"}
    ```

1.  使用 HTTPie 测试此端点：

    ```py
    (venv) http "http://localhost:8000/cars/price?min_price=2000&max_price=4000"
    HTTP/1.1 200 OK
    content-length: 60
    content-type: application/json date: Mon, 28 Mar 2022 21:20:24 GMT
    server: uvicorn
    {
    "Message": "Listing cars with prices between 2000 and 4000"
    }
    ```

在这个解决方案中，您无法确保基本条件，即最低价格应低于最高价格。这是由 Pydantic 的对象级验证处理的。

FastAPI 选择您的查询参数，并执行与之前相同的解析和验证检查。它提供了`Query`函数，就像`Path`函数一样。您可以使用**大于**、**小于**或**等于**条件，以及设置默认值。它们也可以设置为默认为`None`。根据需要，查询参数将被转换为布尔值。您可以编写相当复杂的路径和查询参数的组合，因为 FastAPI 可以区分它们并在函数内部处理它们。

通过这样，您已经看到了 FastAPI 如何使您能够处理通过路径和查询参数传递的数据，以及它使用的工具在幕后尽快进行解析和验证。现在，您将检查 REST API 的主要数据载体：**请求体**。

### 请求体——数据的大部分

REST API 允许客户端（一个网页浏览器或移动应用程序）与 API 服务器之间进行双向通信。大部分数据都通过请求和响应体传输。请求体包含从客户端发送到您的 API 的数据，而响应体是从 API 服务器发送到客户端（们）的数据。

这些数据可以用各种方式编码，但许多用户更喜欢使用 JSON 编码数据，因为它与我们的数据库解决方案 MongoDB 非常出色——MongoDB 使用 BSON，与 JSON 非常相似。

当在服务器上修改数据时，您应该始终使用：

+   `POST`请求：用于创建新资源

+   `PUT`和`PATCH`：用于更新资源

+   `DELETE`：用于删除资源

由于请求体将包含原始数据——在这种情况下，MongoDB 文档或文档数组——您可以使用 Pydantic 模型。但首先，看看这个机制是如何工作的，没有任何验证或建模。在 HTTP 术语中，`GET`方法应该是**幂等的**，这意味着它应该总是为同一组参数返回相同的值。

在以下用于将新车插入未来数据库的假设端点的代码中，你可以将通用的请求体作为数据传递。它可以是字典，无需进入该字典应该如何构建的细节。创建一个名为 `chapter4_06.py` 的新文件，并将以下代码粘贴进去：

```py
from typing import Dict
from fastapi import FastAPI, Body
app = FastAPI()
@app.post("/cars")
async def new_car(data: Dict = Body(...)):
    print(data)
    return {"message": data}
```

直观来看，`Body` 函数与之前介绍的 `Path` 和 `Query` 函数类似。然而，区别在于，当处理请求体时，此函数是强制性的。

三个点表示请求体是必需的（你必须发送一些内容），但这仅是唯一的要求。尝试插入一辆车（2015 年制造的菲亚特 500）：

```py
(venv) http POST "http://localhost:8000/cars" brand="FIAT" model="500" year=2015
HTTP/1.1 200 OK
content-length: 56
content-type: application/json date: Mon, 28 Mar 2022 21:27:31 GMT
server: uvicorn
{
  "message": {
  "brand": "FIAT",
  "model": "500",
  "year": "2015"
}
```

FastAPI 会做繁重的工作。你可以检索传递给请求体的所有数据，并将其提供给函数以进行进一步处理——数据库插入、可选预处理等。

另一方面，你可以向请求体传递任何键值对。当然，这只是一个说明一般机制的例子——在现实中，Pydantic 将成为你的数据守护者，确保你只让正确的数据进入。

虽然一切顺利，但 FastAPI 仍然会发送一个 `200` 响应状态，尽管 `201 Resource Created` 错误更为合适和准确。例如，你可以在函数末尾将一些文档插入 MongoDB，并使用 `201 CREATED` 状态消息。你将看到修改响应体是多么容易，但就目前而言，你将能够看到 Pydantic 在处理请求体时的优势。

要创建新的汽车条目，你只需要 `brand`、`model` 和生产 `year` 字段。

因此，在 `chapter4_07.py` 文件中创建一个简单的 Pydantic 模型，其中包含所需的数据类型：

```py
from fastapi import FastAPI, Body
from pydantic import BaseModel
class InsertCar(BaseModel):
    brand: str
    model: str
    year: int
app = FastAPI()
@app.post("/cars")
async def new_car(data: InsertCar):
    print(data)
    return {"message": data}
```

到现在为止，你知道前两个参数应该是字符串，而年份必须是整数；它们都是必需的。

现在，如果你尝试发送之前相同的数据，但带有额外的字段，你将只会收到这三个字段。此外，这些字段将经过 Pydantic 解析和验证，如果某些内容不符合数据规范，将抛出有意义的错误信息。

Pydantic 模型验证和 `Body` 函数的组合，在处理请求数据时提供了所有必要的灵活性。这是因为你可以将它们结合起来，并通过相同的请求总线传递不同的信息片段。

如果你想要传递与用户关联的促销代码以及新车数据，你可以尝试定义一个用于用户的 Pydantic 模型，并使用 `Body` 函数提取促销代码。首先，在新的文件中定义一个最小的用户模型，并将其命名为 `chapter4_08.py`：

```py
class UserModel(BaseModel):
    username: str
    name: str
```

现在，创建一个更复杂的函数，该函数将处理两个 Pydantic 模型和可选的用户促销代码——将默认值设置为 `None`：

```py
@app.post("/car/user")
async def new_car_model(car: InsertCar, user: UserModel, code: int = Body(None)):
    return {"car": car, "user": user, "code": code}
```

对于这个请求，它包含一个完整的 JSON 对象，其中有两个嵌套对象和一些代码，你可能选择使用 Insomnia 或类似的图形用户界面客户端，因为这样做比在命令提示符中输入 JSON 或使用管道要容易。虽然这主要是一个个人偏好的问题，但在开发和测试 REST API 时，拥有一个如 Insomnia 或 Postman 之类的图形用户界面工具以及一个命令行客户端（如 cURL 或 HTTPie）是非常有用的。

`Body`类构造函数的参数与`Path`和`Query`构造函数非常相似，并且由于它们通常会更加复杂，因此尝试使用 Pydantic 来驯服它们是有用的。解析、验证和有意义的错误消息——Pydantic 在允许请求数据到达真实数据处理功能之前为我们提供了整个包。`POST`请求几乎总是以适当的 Pydantic 模型作为参数传入。

在尝试了请求体和 Pydantic 模型的组合之后，你已经看到你可以控制数据的流入，并且可以确信提供给你的 API 端点的数据将是你想要和期望的数据。然而，有时你可能想要直接与裸金属打交道，并处理原始请求对象。FastAPI 也覆盖了这种情况，如下一节所述。

### 请求对象

FastAPI 建立在 Starlette 网络框架之上。FastAPI 中的原始请求对象是 Starlette 的请求对象，一旦从 FastAPI 直接导入，你就可以在你的函数中访问它。通过直接使用请求对象，你错过了 FastAPI 最重要的功能：Pydantic 的解析和验证以及自文档化！然而，可能存在你需要拥有原始请求的情况。

看看`chapter4_09.py`文件中的以下示例：

```py
from fastapi import FastAPI, Request
app = FastAPI()
@app.get("/cars")
async def raw_request(request: Request):
    return {"message": request.base_url, "all": dir(request)}
```

在前面的代码中，你创建了一个最小的 FastAPI 应用程序，导入了`Request`类，并在端点中使用它。如果你使用`REST`客户端测试此端点，你将只得到基础 URL 作为消息，而`all`部分列出了`Request`对象的全部方法和属性，以便你了解可用的内容。

所有这些方法和属性都可供你在你的应用程序中使用。

有了这些，你已经看到了 FastAPI 如何帮助你与主要的 HTTP 传输机制——请求体、查询字符串和路径——一起工作。接下来，你将探索任何网络框架解决方案同样重要的方面——cookies、headers、表单数据和文件。

### Cookies 和 headers，表单数据，和文件

说到网络框架如何摄取数据，处理表单数据、处理文件以及操作 Cookies 和 headers 等主题必须包括在内。本节将提供 FastAPI 如何处理这些任务的简单示例。

#### Headers

标头参数的处理方式与查询和路径参数类似，正如你稍后将会看到的，还有 cookie。你可以通过使用`Header`函数来收集它们，可以说。在诸如身份验证和授权等主题中，标头是必不可少的，因为它们经常携带**JSON Web Tokens**（JWTs），这些用于识别用户及其权限。

尝试使用新文件`chapter4_10.py`中的`Header`函数读取用户代理：

```py
from typing import Annotated
from fastapi import FastAPI, Header
app = FastAPI()
@app.get("/headers")
async def read_headers(user_agent: Annotated[str | None, Header()] = None):
    return {"User-Agent": user_agent}
```

根据你使用的软件来执行端点的测试，你将得到不同的结果。以下是一个使用 HTTPie 的示例：

```py
(venv) http GET "http://localhost:8000/headers"
HTTP/1.1 200 OK
content-length: 29
content-type: application/json date: Sun, 27 Mar 2022 09:26:49 GMT
server: uvicorn
{
"User-Agent": "HTTPie/3.2.2"
}
```

你可以用这种方式提取所有标头，FastAPI 将提供进一步的帮助——它将名称转换为小写，将键转换为蛇形命名法，等等。

#### Cookie

Cookie 的工作方式类似，尽管它们可以从`Cookies`标头中手动提取。该框架提供了一个名为`Cookie`的实用函数，它以类似于`Query`、`Path`和`Header`的方式完成所有工作。

#### 表单（和文件）

到目前为止，你只处理了 JSON 数据。它是网络上的通用语言，也是你在数据往返中的主要载体。然而，有些情况需要不同的数据编码——表单可能直接由你的 API 处理，数据编码为`multipart/form-data`或`form-urlencoded`。随着现代 React Server Actions 的出现，表单数据在前端开发中也变得更加流行。

注意

尽管你可以在路径操作中声明多个表单参数，但你不能声明 JSON 中预期的`Body`字段。HTTP 请求将使用仅`application/x-www-form-urlencoded`而不是`application/json`进行编码。这种限制是 HTTP 协议的一部分，并不特定于 FastAPI。

覆盖这两种表单情况——包括和不包括上传文件的最简单方法是首先安装`python-multipart`，这是一个 Python 的流式多部分解析器。为此，你必须停止你的服务器并使用`pip`来安装它：

```py
pip install python-multipart==0.0.9
```

`Form`函数与之前检查的实用函数类似，但不同之处在于它寻找表单编码的参数。对于简单字段，数据通常使用媒体类型（`application/x-www-form-urlencoded`）进行编码，而如果包含文件，编码对应于`multipart/form-data`。

看一个简单的例子，你希望上传一张图片和一些表单字段，比如品牌和型号。

你将使用一张可以在 Pexels 上找到的照片（[`www.pexels.com/photo/white-`](https://www.pexels.com/photo/white-vintage-car-parked-on-green-grass-8746027/)），重命名为`car.jpeg`并保存在当前目录中。

创建一个名为`chapter4_11.py`的文件，并将以下代码粘贴进去：

```py
from fastapi import FastAPI, Form, File, UploadFile
app = FastAPI()
@app.post("/upload")
async def upload(
    file: UploadFile = File(...), brand: str = Form(...), model: str = Form(...)
):
    return {"brand": brand, "model": model, "file_name": file.filename}
```

上一段代码通过`Form`函数处理表单参数，并通过使用`UploadFile`实用类上传文件。

然而，照片并没有保存在磁盘上——它的存在只是被确认，并返回文件名。在 HTTPie 中测试具有文件上传的端点如下所示：

```py
http -f POST localhost:8000/upload  brand='Ferrari' model='Testarossa'  file@car.jpeg
```

前面的 HTTPie 调用返回以下输出：

```py
HTTP/1.1 200 OK
content-length: 63
content-type: application/json
date: Fri, 22 Mar 2024 11:01:38 GMT
server: uvicorn
{
    "brand": "Ferrari",
    "file_name": "car.jpeg",
    "model": "Testarossa"
}
```

要将图像保存到磁盘，你必须将缓冲区复制到磁盘上的实际文件中。以下代码实现了这一点（`chapter4_12.py`）：

```py
import shutil
from fastapi import FastAPI, Form, File, UploadFile
app = FastAPI()
@app.post("/upload")
async def upload(
    picture: UploadFile = File(...),
    brand: str = Form(...),
    model: str = Form(...)
):
    with open("saved_file.png", "wb") as buffer:
        shutil.copyfileobj(picture.file, buffer)
    return {"brand": brand, "model": model, "file_name": picture.filename}
```

`open`块使用指定的文件名在磁盘上打开一个文件，并复制通过表单发送的 FastAPI 文件。你将硬编码文件名，因此任何新的上传将简单地覆盖现有文件，但你可以使用**通用唯一识别码**（**UUID**）库等随机生成文件名。

文件上传可以通过不同的方式实现——文件上传也可以由 Python 的`async`文件库`aiofiles`或作为后台任务处理，这是 FastAPI 的另一个特性，将在第五章中展示，*设置 React 工作流程*。

## FastAPI 响应自定义

前几节讨论了 FastAPI 请求的许多示例，说明了你可以如何触及请求的每一个角落——路径、查询字符串、请求体、头部和 Cookies，以及如何处理表单编码的请求。

现在，让我们更仔细地看看 FastAPI 的响应对象。在所有之前的案例中，你返回了一个由 FastAPI 序列化为 JSON 的 Python 字典。该框架允许对响应进行自定义。

在 HTTP 响应中，你可能首先想要更改的是状态码，例如，在事情没有按计划进行时提供一些有意义的错误。当存在 HTTP 错误时，FastAPI 方便地引发经典的 Python 异常。它还使用符合标准的、有意义的响应代码，以最大限度地减少创建自定义有效负载消息的需求。例如，你不希望为所有内容发送`200 OK`状态码，然后通过有效负载通知用户错误——FastAPI 鼓励良好的实践。

#### 设置状态码

HTTP 状态码表示操作是否成功或存在错误。这些代码还提供了关于操作类型的信息，并且可以根据几个组来划分：信息性、成功、客户端错误、服务器错误等。虽然不需要记住状态码，但你可能知道`404`或`500`代码的含义。

FastAPI 使设置状态码变得非常简单——只需将所需的`status_code`变量传递给装饰器即可。在这里，你正在使用`208 status`代码为一个简单的端点（`chapter4_13.py`）：

```py
from fastapi import FastAPI, status
app = FastAPI()
@app.get("/", status_code=status.HTTP_208_ALREADY_REPORTED)
async def raw_fa_response():
    return {"message": "fastapi response"}
```

在 HTTPie 中测试根路由产生以下输出：

```py
(venv) http GET "http://localhost:8000"
HTTP/1.1 208 Already Reported content-length: 30
content-type: application/json date: Sun, 27 Mar 2022 20:14:25 GMT
server: uvicorn
{
    "message": "fastapi response"
}
```

类似地，你可以为`delete`、`update`和`create`操作设置状态码。

FastAPI 默认设置`200 状态码`，如果没有遇到异常，因此设置各种 API 操作的正确代码取决于你，例如删除时使用`204 No Content`，创建时使用`201`。这是一个特别值得鼓励的良好实践。

Pydantic 可用于响应建模。您可以使用`response_model`参数限制或修改应在响应中出现的字段，并执行与请求体类似的检查。

FastAPI 不启用自定义响应，但修改和设置头和 cookie 与从 HTTP 请求和框架中读取它们一样简单。

虽然这超出了本书的范围，但值得注意的是，JSON 绝不是 FastAPI 可以提供的唯一响应：您可以输出`HTMLResponse`并使用经典的 Flask-like Jinja 模板，`StreamingResponse`，`FileResponse`，`RedirectResponse`等等。

#### HTTP 错误

错误是不可避免的。例如，用户可能以某种方式向查询发送了错误的参数，前端发送了错误的请求体，或者数据库离线（尽管在 MongoDB 中这种情况不太可能）——任何情况都可能发生。尽快检测这些错误（这是 FastAPI 的一个主题）并向前端以及用户发送清晰完整的消息，通过抛出异常至关重要。

FastAPI 依赖于网络标准，并在开发过程的各个方面强制执行良好实践，因此它非常重视使用 HTTP 状态码。这些代码提供了对出现问题的清晰指示，而有效载荷可以用来进一步阐明问题的原因。

FastAPI 使用一个称为`HTTPException`的 Python 异常来引发 HTTP 错误。这个类允许您设置状态码并设置错误消息。

回到将新汽车插入数据库的例子，您可以设置一个自定义异常，如下所示（`chapter4_14.py`）：

```py
from pydantic import BaseModel
from fastapi import Fastapi, HTTPException, status
app = FastAPI()
class InsertCar(BaseModel):
    brand: str
    model: str
    year: int
@app.post("/carsmodel")
async def new_car_model(car: InsertCar):
    if car.year > 2022:
        raise HTTPException(
            status.HTTP_406_NOT_ACCEPTABLE, detail="The car doesn't exist yet!"
        )
    return {"message": car}
```

当尝试插入尚未建造的汽车时，响应如下：

```py
(venv) λ http POST http://localhost:8000/carsmodel brand="fiat" mode3
l="500L" year=2023
HTTP/1.1 406 Not Acceptable content-length: 39
content-type: application/json date: Tue, 29 Mar 2022 18:37:42 GMT
server: uvicorn
{
    "detail": "The car doesn't exist yet!"
}
```

这是一个相当牵强的例子，用于为可能出现的潜在问题创建自定义异常。然而，这很好地说明了可能实现的内容以及 FastAPI 提供的灵活性。

#### 依赖注入

为了简要但自包含地介绍 FastAPI，必须提到依赖注入系统。从广义上讲，**依赖注入**（**DI**）是在适当的时间向路径操作函数提供必要功能（类、函数、数据库连接、授权状态等）的一种方式。FastAPI 的 DI 系统对于在端点之间共享逻辑、共享数据库连接等非常有用，正如您在连接到您的 MongoDB Atlas 实例时将看到的——执行安全性和身份验证检查等。

依赖项并不特殊；它们只是可以接受与路径操作相同参数的正常函数。实际上，官方文档将它们与未使用装饰器的路径操作进行比较。尽管如此，依赖项的使用方式略有不同。它们被赋予一个单一参数（通常是可调用的），并且不是直接调用；它们只是作为参数传递给 `Depends()`。

一个受官方 FastAPI 文档启发的示例如下；你可以使用分页依赖并在不同的资源中使用它（`chapter4_15.py`）：

```py
from typing import Annotated
from fastapi import Depends, FastAPI
app = FastAPI()
async def pagination(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}
@app.get("/cars/")
async def read_items(commons: Annotated[dict, Depends(pagination)]):
    return commons
@app.get("/users/")
async def read_users(commons: Annotated[dict, Depends(pagination)]):
    return commons
```

在全栈 FastAPI 项目中，DI（依赖注入）最常见的情况之一是身份验证；你可以使用相同的身份验证逻辑，即检查头部的授权令牌并将其应用于所有需要身份验证的路由或路由器，正如你将在*第六章**，身份验证* *和授权* *中看到的那样。

#### 使用路由器结构化 FastAPI 应用程序

虽然将所有我们的 `request`/`response` 逻辑放在一个大文件中是可能的，但随着你开始构建一个中等规模的项目，你很快就会看到这并不可行，不可维护，也不便于工作。FastAPI，就像 Node.js 世界的 Express.js 或 Flask 的蓝图，在 `/cars` 路径上提供了 `cars`，另一个用于处理 `/users` 路径上的用户创建和管理，等等。FastAPI 提出了一种简单直观的项目结构，足以容纳最常见的情况。

### API 路由器

FastAPI 提供了一个名为 APIRouter 的类，用于分组路由，通常与同一类型的资源（用户、购物项目等）相关。这个概念在 Flask 中被称为 **Blueprints**，并且在每个现代 Web 框架中都存在，它允许代码更加模块化和分散在更小的单元中，每个路由器只管理一种类型的资源。这些 APIRouter 最终包含在主要的 FastAPI 实例中，并提供非常相似的功能。

而不是直接在主应用程序实例（通常称为 app）上应用路径装饰器（`@get`、`@post` 等），它们被应用于 APIRouter 实例。下面是一个简单的示例，将应用程序拆分为两个 APIRouter：

1.  首先，创建一个名为 `chapter4_16.py` 的文件，该文件将托管主要的 FastAPI 实例：

    ```py
    from fastapi import FastAPI
    from routers.cars import router as cars_router
    from routers.user import router as users_router
    app = FastAPI()
    app.include_router(cars_router, prefix="/cars", tags=["cars"])
    app.include_router(users_router, prefix="/users", tags=["users"])
    ```

1.  现在，创建一个名为 `/routers` 的新文件夹，并在该文件夹中创建一个名为 `users.py` 的文件，用于创建 APIRouter：

    ```py
    from fastapi import APIRouter
    router = APIRouter()
    @router.get("/")
    async def get_users():
        return {"message": "All users here"}
    ```

1.  在同一 `/routers` 目录中创建另一个文件，命名为 `cars.py`：

    ```py
    from fastapi import APIRouter
    router = APIRouter()
    @router.get("/")
    async def get_cars():
        return {"message": "All cars here"}
    ```

当在 `chapter4_17.py` 文件中将路由器连接到主应用程序时，你可以向 APIRouter 提供不同的可选参数——标签和一组依赖项，例如身份验证要求。然而，前缀是强制性的，因为应用程序需要知道在哪个 URL 上挂载 APIRouter。

如果你使用以下命令使用 Uvicorn 测试此应用程序：

```py
uvicorn chapter4_17:app
```

然后，前往自动生成的文档，您会看到两个 APIRouter 被挂载，就像您定义了两个单独的端点一样。然而，它们被分别归类在各自的标签下，以便于导航和测试。

如果您现在导航到文档，您确实应该找到在`/cars`上定义的一个路由，并且只响应`GET`请求。直观地，这个程序可以让您在短时间内构建并行或同一级别的路由，但使用 APIRouters 的最大好处之一是它们支持嵌套，这使得管理端点的复杂层次结构变得轻而易举！

路由是应用程序的子系统，并不打算独立使用，尽管您可以在特定路径下自由挂载整个独立的 FastAPI 应用程序，但这超出了本书的范围。

#### 中间件

FastAPI 实现了`请求`/`响应`周期的概念，拦截请求，以某种期望的方式对其进行操作，然后在将其发送到浏览器或客户端之前获取响应，如果需要，执行额外的操作，最后返回最终的响应。

中间件基于 ASGI 规范，并在 Starlette 中实现，因此 FastAPI 允许您在所有路由中使用它，并且可以选择将其绑定到应用程序的一部分（通过 APIRouter）或整个应用程序。

与提到的框架类似，FastAPI 的中间件只是一个接收请求和`call_next`函数的函数。创建一个名为`chapter4_17.py`的新文件：

```py
from fastapi import FastAPI, Request
from random import randint
app = FastAPI()
@app.middleware("http")
async def add_random_header(request: Request, call_next):
    number = randint(1,10)
    response = await call_next(request)
    response.headers["X-Random-Integer "] = str(number)
    return response
@app.get("/")
async def root():
    return {"message": "Hello World"}
```

如果您现在启动这个小型应用程序，并测试唯一的路由，即`http://127.0.0.1:8000/`上的路由，您会注意到返回的头部包含一个介于 1 到 10 之间的整数，并且每次请求这个整数都会不同。

中间件在**跨源资源共享**（**CORS**）认证中扮演着重要角色，这是您在开发全栈应用程序时必然会遇到的问题，同时也用于重定向、管理代理等。这是一个非常强大的概念，可以极大地简化并提高您的应用程序效率。

# 概述

本章介绍了 FastAPI 如何通过利用现代 Python 功能和库（如 Pydantic）实现最常用的 REST API 任务的一些简单示例，以及它如何帮助您。

本章还详细介绍了 FastAPI 如何使您能够通过 HTTP 执行请求和响应，以及您如何在任何时候利用它，自定义和访问请求以及响应的元素。最后，它还详细介绍了如何将 API 拆分为路由，以及如何将应用程序组织成基于资源的逻辑单元。

下一章将为您快速介绍 React——FARM 堆栈中首选的用户界面库。
