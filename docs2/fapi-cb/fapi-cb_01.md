

# 第一章：FastAPI 的第一步

欢迎来到激动人心的 **FastAPI** 世界，这是一个用于在 **Python** 中构建 API 和 Web 应用程序的现代、高性能框架。本章是您了解和利用 FastAPI 力量的入门。在这里，您将迈出第一步，设置您的开发环境，创建您的第一个 FastAPI 项目，并探索其基本概念。

FastAPI 以其速度、易用性和全面的文档而脱颖而出，成为希望构建可扩展和高效 Web 应用的开发者的首选选择。在本章中，您将实际参与设置 FastAPI，学习如何导航其架构，并了解其核心组件。通过定义简单的 API 端点、处理 HTTP 方法以及学习请求和响应处理，您将获得实践经验。这些基础技能对于任何进入使用 FastAPI 构建现代 Web 开发世界的开发者至关重要。

到本章结束时，您将对 FastAPI 的基本结构和功能有扎实的理解。您将能够设置新项目、定义 API 端点，并掌握使用 FastAPI 处理数据。这些基础知识为您在阅读本书的过程中遇到更高级主题和复杂应用奠定了基础。

在本章中，我们将介绍以下配方：

+   设置您的开发环境

+   创建新的 FastAPI 项目

+   理解 FastAPI 基础

+   定义您的第一个 API 端点

+   使用路径和查询参数

+   定义和使用请求和响应模型

+   处理错误和异常

每个配方都是为了提供给您实用的知识和直接经验，确保在本章结束时，您将准备好开始构建自己的 FastAPI 应用程序。

# 技术要求

要开始您的 FastAPI 之旅，您需要设置一个支持 Python 开发和 FastAPI 功能的环境。以下是本章所需的技术要求和安装列表：

+   **Python**：FastAPI 是基于 Python 构建的，因此您需要一个与您的 FastAPI 版本兼容的 Python 版本。您可以从 [python.org](http://python.org) 下载最新版本。

+   `pip`，Python 的包管理器。您可以通过在命令行中运行 `pip install fastapi` 来完成此操作。

+   `pip install uvicorn`。

+   **集成开发环境（IDE）**：一个支持 Python 开发的 IDE，例如 **Visual Studio Code**（**VS Code**）、PyCharm 或任何其他支持 Python 开发的 IDE，对于编写和测试您的代码将是必要的。

+   **Postman 或 Swagger UI**：用于测试 API 端点。FastAPI 自动生成并托管 Swagger UI，因此您可以直接使用它。

+   **Git**：版本控制是必不可少的，Git 是一个广泛使用的系统。如果尚未安装，您可以从 [git-scm.com](http://git-scm.com) 获取它。

+   **GitHub 账户**：要访问代码仓库，需要 GitHub 账户。如果您还没有，请前往 [github.com](http://github.com) 注册。

本章中使用的代码可在以下地址的 GitHub 上找到：[`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter01`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter01)。您可以在本地机器上通过 [`github.com/PacktPublishing/FastAPI-Cookbook`](https://github.com/PacktPublishing/FastAPI-Cookbook) 克隆或下载存储库以进行跟随。

# 设置您的开发环境

这份食谱，致力于设置您的开发环境，是任何成功的 Web 开发项目的关键基础。在这里，您将学习如何安装和配置所有必需的工具，以便开始使用 FastAPI 进行构建。

我们首先将指导您安装 Python，这是 FastAPI 背后的核心语言。接下来，我们将继续安装 FastAPI 本身，以及 Uvicorn，一个闪电般的 ASGI 服务器，它是运行您的 FastAPI 应用程序的基础。

设置 IDE 是我们的下一个目标。无论您更喜欢 VS Code、PyCharm 还是其他任何 Python 友好的 IDE，我们都会提供一些提示，使您的开发过程更加顺畅和高效。

最后，我们将向您介绍 Git 和 GitHub – 这些是现代软件开发中版本控制和协作不可或缺的工具。了解如何使用这些工具不仅可以帮助您有效地管理代码，而且还能打开通往社区驱动开发和资源的广阔世界的大门。

## 准备工作

FastAPI 与 Python 兼容，因此在使用之前您需要检查您的 Python 版本。这是设置 FastAPI 的重要步骤。我们将指导您如何安装它。

#### Windows 安装

如果您在 Windows 上工作，请按照以下步骤安装 Python：

1.  访问官方 Python 网站：[python.org](http://python.org)。

1.  下载 Python 的最新版本或任何高于 3.9 的版本。

1.  运行安装程序。在点击“立即安装”之前，请确保勾选“将 Python 添加到 PATH”的复选框。

1.  安装后，打开命令提示符并输入 `python --version` 以确认安装。

#### macOS/Linux 安装

macOS 通常预装了 Python，但可能不是最新版本。

您可以使用 `Homebrew`（macOS 的包管理器）。要安装它，打开终端并运行以下命令：

```py
$ /bin/bash -c "$(curl –fsSL https://raw.githubusercontent.com/\Homebrew/install/HEAD/install.sh)"
```

然后，您可以使用以下命令在终端中安装 Python：

```py
$ brew install python
```

在 Linux 上，您可以通过运行以下命令使用包管理器安装 Python：

```py
$ sudo apt-get install python3
```

这就是您在 macOS 和 Linux 系统上安装 Python 所需要的一切。

#### 检查安装

安装后，在终端中运行以下命令以检查 Python 是否正确安装：

```py
$ python --version
```

如果您在 Linux 上安装了它，二进制命令是 `python3`，因此您可以通过运行以下命令来检查 Python 是否正确安装：

```py
$ python3 --version
```

一旦安装了 Python，我们想要确保 Python 的包管理器已正确安装。它随 Python 的安装一起提供，被称为 `pip`。

从终端窗口运行以下命令：

```py
$ pip --version
```

在 Linux 上，运行以下命令：

```py
$ pip3 --version
```

一旦您的计算机上安装了 Python，您现在可以考虑安装 FastAPI。

## 如何做到这一点...

当您已经准备好 Python 和 `pip` 后，我们可以继续安装 FastAPI 和 IDE。然后，我们将配置 Git。

我们将按照以下步骤进行：

1.  安装 FastAPI 和 Uvicorn

1.  设置您的 IDE（VS Code 或 PyCharm）

1.  设置 Git 和 GitHub 以跟踪您的项目

### 安装 FastAPI 和 Uvicorn

在设置好 Python 后，下一步是安装 FastAPI 和 Uvicorn。FastAPI 是我们将用来构建应用程序的框架，而 Uvicorn 是一个 ASGI 服务器，用于运行和提供我们的 FastAPI 应用程序。

打开您的命令行界面，通过运行以下命令一起安装 FastAPI 和 Uvicorn：

```py
$ pip install fastapi[all]
```

此命令将安装 FastAPI 以及其推荐的依赖项，包括 Uvicorn。

为了验证安装，您只需在终端中运行 `uvicorn --version` 即可。

### 设置您的 IDE

选择正确的 IDE 是您 FastAPI 之旅中的关键步骤。IDE 不仅仅是一个文本编辑器；它是一个您编写、调试和测试代码的空间。

一个好的 IDE 可以显著提高您的编码体验和生产力。对于 FastAPI 开发以及 Python 的一般使用，两个流行的选择是 VS Code 和 PyCharm。

#### VS Code

**VS Code** 是一个免费、开源、轻量级的 IDE，具有强大的功能。它提供了出色的 Python 支持，并且高度可定制。

您可以从官方网站（`code.visualstudio.com`）下载并安装 VS Code。安装过程相当简单。安装完成后，打开 VS Code，转到 `python`。安装微软版本，然后完成。

#### PyCharm

**PyCharm** 由 JetBrains 创建，专门针对 Python 开发。它为专业开发者提供了一系列工具，包括对 FastAPI 等网络开发框架的优秀支持。

您可以选择社区免费版和专业付费版。对于本书的范围，社区版已经足够，您可以在 JetBrains 网站上下载：[`www.jetbrains.com/pycharm/download/`](https://www.jetbrains.com/pycharm/download/).

对于 PyCharm，安装过程也很简单。

#### 提高您的开发体验

对于这两个 IDE - 以及如果您使用其他 IDE - 确保利用基本优势来提高您的开发体验并提高效率。以下是我接近新的 IDE 环境时使用的简短清单：

+   **代码补全和分析**：好的 IDE 提供智能代码补全、错误突出显示和修复，这对于高效开发非常有价值。

+   **调试工具**：利用 IDE 提供的调试功能来诊断和解决代码中的问题

+   **版本控制集成**：一个好的 IDE 提供了对 Git 的支持，简化了代码更改跟踪和仓库管理

+   **自定义**：通过调整主题、快捷键和设置来自定义你的 IDE，使你的开发体验尽可能舒适和高效

### 设置 Git 和 GitHub

版本控制是软件开发的一个基本方面。Git 与 GitHub 结合，形成了一套强大的工具集，用于跟踪更改、协作和维护项目的历史。你可以从官方网站 [git-scm.com](http://git-scm.com) 下载 Git 安装程序并安装它。

安装完成后，使用以下命令在命令行中配置 Git 的用户名和电子邮件：

```py
$ git config --global user.name "Your Name"
$ git config --global user.email "your.email@example.com"
```

GitHub 是本书中使用的代码示例存储的平台。如果你还没有，请在 [github.com](http://github.com) 上注册一个 GitHub 账户。

# 创建一个新的 FastAPI 项目

设置一个有组织的项目结构对于维护干净的代码库至关重要，尤其是在你的应用程序增长和演变时。这个食谱将指导你如何创建你的第一个基本的 FastAPI 项目。一个结构化的项目简化了导航、调试和协作。对于 FastAPI，遵循结构化的最佳实践可以显著提高可扩展性和可维护性。

## 准备工作

要遵循这个食谱，你需要确保你的开发环境已经设置好。

## 如何做到...

我们首先创建一个名为 `fastapi_start` 的项目文件夹，我们将使用它作为根项目文件夹。

1.  在根项目文件夹级别的终端中，我们将通过运行以下命令设置我们的虚拟环境：

    ```py
    .venv folder that will contain all packages required for the project within our project's root folder.
    ```

1.  现在，你需要激活环境。如果你使用的是 Mac 或 Linux，请运行以下命令：

    ```py
    (.venv) $. Alternatively, if you check the location of the python binary command, it should be located within the .venv folder. From now on, each time you install a module with pip, it will be installed in the .venv folder, and it will be activated only if the environment is active.
    ```

1.  现在，你可以通过运行以下命令在你的环境中安装 `fastapi` 包和 `uvicorn`：

    ```py
    main.py.
    ```

1.  此文件是你的 FastAPI 应用程序开始的地方。首先，编写 `FastAPI` 模块的导入。然后，创建 `FastAPI` 类的一个实例：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    ```

    此实例存放着你的应用程序代码。

1.  接下来，定义你的第一个路由。在 FastAPI 中，路由就像路标，将请求导向相应的函数。从一个简单的返回问候世界的路由开始：

    ```py
    @app.get("/")
    def read_root():
        return {"Hello": "World"}
    ```

    你刚刚创建了你的第一个 FastAPI 应用程序的代码。

如果你想要跟踪项目，你可以按照以下方式设置 Git：

1.  在你的项目根目录中，打开终端或命令提示符并运行以下命令：

    ```py
    .gitignore file to specify untracked files to ignore (such as __pychache__, .venv, or IDE-specific folders). You can also have a look at the one on the GitHub repository of the project at the link: https://github.com/PacktPublishing/FastAPI-Cookbook/blob/main/.gitignore.
    ```

1.  然后，使用以下命令添加你的文件：

    ```py
    $ git add .
    ```

1.  然后，使用以下命令提交它们：

    ```py
    $ git commit –m "Initial commit"
    ```

就这样。你现在可以使用 Git 跟踪你的项目了。

## 还有更多...

一个结构良好的项目不仅仅是关于整洁；它关乎创建一个可持续和可扩展的环境，让你的应用程序可以增长和演变。在 FastAPI 中，这意味着以逻辑和高效的方式组织你的项目，以分离应用程序的不同方面。

对于 FastAPI 项目来说，没有独特和完美的结构；然而，一个常见的做法是将你的项目分为几个关键目录：

+   `/src`：这是你的主要应用程序代码所在的地方。在 `/src` 内，你可能会有不同模块的应用程序的子目录。例如，你可以有一个 `models` 目录用于你的数据库模型，一个 `routes` 目录用于你的 FastAPI 路由，以及一个 `services` 目录用于业务逻辑。

+   `/tests`：将你的测试代码与应用程序代码分开是一个好的做法。这使得管理它们变得更容易，并确保你的生产构建不包括测试代码。

+   `/docs`：对于任何项目来说，文档都是至关重要的。无论是 API 文档、安装指南还是使用说明，为文档保留一个专门的目录有助于保持清晰。

## 参见

你可以在以下链接中找到有关如何使用 `venv` 管理虚拟环境的详细信息：

+   *虚拟环境的创建*：[`docs.python.org/3/library/venv.xhtml`](https://docs.python.org/3/library/venv.xhtml)

为了用 Git 提升你的知识并熟悉添加、暂存和提交操作，请查看此指南：

+   *Git 简明指南*：[`rogerdudler.github.io/git-guide/`](https://rogerdudler.github.io/git-guide/)

# 理解 FastAPI 基础

在我们开始使用 FastAPI 的旅程时，建立一个坚实的基础至关重要。FastAPI 不仅仅是一个另一个网络框架；它是一个强大的工具，旨在使开发者的生活更轻松，使应用程序更快，使代码更健壮和易于维护。

在这个菜谱中，我们将揭示 FastAPI 的核心概念，深入探讨其独特的功能，如异步编程，并指导你创建和组织你的第一个端点。到菜谱结束时，你将拥有一个运行中的第一个 FastAPI 服务器——这是一个标志着现代网络开发激动人心旅程开始的里程碑。

FastAPI 是一个基于 Python 的现代、快速网络框架，用于构建基于标准 Python 类型提示的 API。

定义 FastAPI 的关键特性如下：

+   **速度**：它是构建 Python API 中最快的框架之一，归功于其底层的 Starlette 框架和 Pydantic 数据处理

+   **易用性**：FastAPI 被设计为易于使用，具有直观的编码，可以加速你的开发时间

+   **自动文档**：使用 FastAPI，API 文档是自动生成的，这是一个既节省时间又对开发者有益的特性

## 如何做到这一点…

现在，我们将探讨如何有效地使用这些功能，并提供一些一般性的指导。

我们将按照以下步骤进行：

+   将异步编程应用于现有端点以提高时间效率

+   探索路由器和端点以更好地组织大型代码库

+   使用基本配置运行你的第一个 FastAPI 服务器

+   探索自动文档

### 应用异步编程

FastAPI 最强大的功能之一是其对异步编程的支持。这允许你的应用程序同时处理更多请求，使其更高效。异步编程是一种并发编程风格，其中任务在没有阻塞其他任务执行的情况下执行，从而提高了应用程序的整体性能。为了顺利集成异步编程，FastAPI 利用 `async`/`await` 语法 ([`fastapi.tiangolo.com/async/`](https://fastapi.tiangolo.com/async/)) 并自动集成异步函数。

因此，从 *创建新的 FastAPI 项目* 菜单中的前一个代码片段中，`main.py` 中的 `read_root()` 函数可以写成如下所示：

```py
@app.get("/")
async def read_root():
    return {"Hello": "World"}
```

在这个例子中，代码的行为将与之前完全相同。

### 探索路由器和端点

在 FastAPI 中，将代码组织成路由器和端点是基本实践。这种组织有助于使代码更整洁、更模块化。

#### 端点

端点是 API 交互发生的点。在 FastAPI 中，通过使用 HTTP 方法（如 `@app.get("/")`）装饰一个函数来创建端点。

这表示对应用程序根的 `GET` 请求。

考虑以下代码片段：

```py
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def read_root():
    return {"Hello": "World"}
```

在这个片段中，我们定义了一个针对根 URL (`"/"`) 的端点。当对 URL 发起 `GET` 请求时，`read_root` 函数被调用，返回一个 JSON 响应。

#### 路由器

当我们需要处理位于不同文件中的多个端点时，我们可以从使用路由器中受益。路由器帮助我们将端点分组到不同的模块中，这使得我们的代码库更容易维护和理解。例如，我们可以为与用户相关的操作使用一个路由器，为与产品相关的操作使用另一个路由器。

要定义一个路由器，首先在 `fastapi_start` 文件夹中创建一个名为 `router_example.py` 的新文件。然后，创建路由器如下所示：

```py
from fastapi import APIRouter
router = APIRouter()
@router.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

你现在可以重用它，并将路由器附加到 `main.py` 中的 FastAPI 服务器实例：

```py
import router_example
from fastapi import FastAPI
app = FastAPI()
app.include_router(router_example.router)
@app.get("/")
async def read_root():
    return {"Hello": "World"}
```

现在你已经有了运行服务器的代码，其中包括来自另一个模块的 `GET /items` 端点的路由器导入。

### 运行你的第一个 FastAPI 服务器

要运行你的 FastAPI 应用程序，你需要将 Uvicorn 指向你的应用实例。如果你的文件名为 `main.py`，并且你的 FastAPI 实例名为 `app`，你可以在 `fastapi_start` 文件夹级别像这样启动你的服务器：

```py
$ uvicorn main:app --reload
```

`--reload` 标志在代码更改后使服务器重新启动，这使得它非常适合开发。

服务器启动后，你可以在 `http://127.0.0.1:8000` 访问你的 API。如果你在浏览器中访问这个 URL，你会看到来自我们刚刚创建的 `"/"` 端点的 JSON 响应。

### 探索自动文档

FastAPI 最令人兴奋的特性之一是其自动文档。当你运行 FastAPI 应用程序时，会自动生成两个文档接口：**Swagger UI** 和 **Redoc**。

你可以通过 `http://127.0.0.1:8000/docs` 访问 Swagger UI，通过 `http://127.0.0.1:8000/redoc` 访问 Redoc。

这些接口提供了一种交互式的方式来探索你的 API 并测试其功能。

## 相关内容

你可以在以下链接中了解更多关于我们在菜谱中涵盖的内容：

+   *第一步*：[`fastapi.tiangolo.com/tutorial/first-steps/`](https://fastapi.tiangolo.com/tutorial/first-steps/)

+   *文档 URL*：[`fastapi.tiangolo.com/tutorial/metadata/#docs-urls`](https://fastapi.tiangolo.com/tutorial/metadata/#docs-urls)

+   *并发和异步 /* *await*：[`fastapi.tiangolo.com/async/`](https://fastapi.tiangolo.com/async/)

# 定义你的第一个 API 端点

现在你已经对 FastAPI 有了一个基本的了解，你的开发环境也已经全部设置好，是时候迈出下一个激动人心的步骤：创建你的第一个 API 端点。

这就是 FastAPI 真正的魔力开始显现的地方。你会看到你可以多么轻松地构建一个功能性的 API 端点，准备好响应 HTTP 请求。

在这个菜谱中，你将创建一个书店后端服务的基本草案。

## 准备工作

确保你知道如何从 *创建一个新的 FastAPI 项目* 菜谱中启动一个基本的 FastAPI 项目。

## 如何做到这一点...

在 Web API 领域中，`GET` 请求可能是最常见的一种。它用于从服务器检索数据。在 FastAPI 中，处理 `GET` 请求既简单又直观。让我们创建一个基本的 `GET` 端点。

假设你正在构建一个书店的 API。你的第一个端点将在给定其 ID 的情况下提供关于一本书的信息。下面是如何做到这一点：

1.  创建一个名为 `bookstore` 的新文件夹，它将包含你将要编写的代码。

1.  在其中创建一个包含服务器实例的 `main.py` 文件：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/books/{book_id}")
    async def read_book(book_id: int):
        return {
            "book_id": book_id,
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald"
        }
    ```

在前面的代码片段中，`@app.get("/books/{book_id}")` 装饰器告诉 FastAPI 这个函数将响应 `/books/{book_id}` 路径上的 `GET` 请求。路径中的 `{book_id}` 是一个路径参数，你可以用它来动态传递值。FastAPI 会自动提取 `book_id` 参数并将其传递给你的函数。

类型提示和自动数据验证

注意到使用了类型提示（`book_id: int`）。FastAPI 使用这些提示来进行数据验证。如果请求中带有非整数的 `book_id` 参数，FastAPI 会自动发送一个有用的错误响应。

## 它是如何工作的…

定义了 `GET` 端点后，使用 Uvicorn 运行你的 FastAPI 应用程序，就像你之前做的那样：

```py
$ uvicorn main:app --reload
```

在终端上，您可以阅读描述服务器正在端口`8000`上运行的日志消息。

FastAPI 最受欢迎的特性之一是其使用 Swagger UI 自动生成交互式 API 文档。这个工具允许您直接从浏览器中测试您的 API 端点，而无需编写任何额外的代码，并且您可以直接检查其中新创建的端点是否存在。

### 使用 Swagger UI

要测试您的新`GET`端点，请在浏览器中导航到`http://127.0.0.1:8000/docs`。这个 URL 会显示您的 FastAPI 应用的 Swagger UI 文档。在这里，您会看到您的`/books/{book_id}`端点被列出。点击它，您将能够从界面直接执行测试请求。尝试输入一个书籍 ID，看看您的 API 生成的响应。

### Postman – 一种多功能的替代方案

虽然 Swagger UI 对于快速测试来说很方便，但您可能希望使用像 Postman 这样的更健壮的工具来处理更复杂的场景。Postman 是一个 API 客户端，它允许您更广泛地构建、测试和记录您的 API。

要使用 Postman，请从 Postman 网站下载并安装它（[`www.postman.com/downloads/`](https://www.postman.com/downloads/))。

安装完成后，创建一个新的请求。将方法设置为`GET`，并将请求 URL 设置为您的 FastAPI 端点，`http://127.0.0.1:8000/books/1`。点击**发送**，Postman 将显示您的 FastAPI 服务器的响应。

# 使用路径和查询参数进行工作

API 开发中最关键的一个方面是处理参数。参数允许您的 API 接受用户的输入，使您的端点变得动态和响应。

在这个食谱中，我们将探讨如何捕获和处理路径、查询参数，并高效地测试它们，从而增强 FastAPI 应用的灵活性和功能性。

## 准备工作

要遵循这个食谱，请确保您知道如何从上一个食谱中创建一个基本的端点。

## 如何操作...

路径参数是 URL 中预期会变化的组成部分。例如，在一个如`/books/{book_id}`的端点中，`book_id`是一个路径参数。FastAPI 允许您轻松捕获这些参数并在函数中使用它们。

1.  让我们通过添加一个新的端点来扩展我们的书店 API，这个端点使用路径参数。这次，我们将创建一个获取特定作者信息的路由：

    ```py
    @app.get("/authors/{author_id}")
    async def read_author(author_id: int):
        return {
            "author_id": author_id,
            "name": "Ernest Hemingway"
        }
    ```

    名称将不会改变；然而，`author_id`的值将是查询请求提供的那个。

    查询参数用于细化或自定义 API 端点的响应。它们可以包含在 URL 中的问号（`?`）之后。例如，`/books?genre=fiction&year=2010`可能会返回只有 2010 年发布的属于小说类别的书籍。

1.  让我们在现有的端点中添加查询参数。假设我们想允许用户通过出版年份过滤书籍：

    ```py
    @app.get("/books")
    async def read_books(year: int = None):
        if year:
            return {
                "year": year,
                "books": ["Book 1", "Book 2"]
            }
        return {"books": ["All Books"]}
    ```

在这里，`year` 是一个可选的查询参数。通过将其默认值设置为 `None`，我们使其成为可选的。如果指定了年份，端点将返回该年的书籍；否则，它将返回所有书籍。

练习

使用 `APIRouter` 类，将每个端点重构到单独的文件中，并将路由添加到 FastAPI 服务器。

## 它是如何工作的…

现在，从命令行终端，通过运行以下命令启动服务器：

```py
$ uvicorn main:app
```

使用 Swagger UI 或 Postman 测试带有路径参数的端点，类似于我们测试基本 `GET` 端点的方式。

在 Swagger UI 中，在 `http://localhost:8000/docs`，导航到您的 `/authors/{author_id}` 端点。您会注意到在您可以尝试之前，它会提示您输入 `author_id` 值。输入一个有效的整数并执行请求。您应该会看到一个包含作者信息的响应。

`GET /books` 端点现在将显示一个可选的 `year` 查询参数字段。您可以通过输入不同的年份来测试它，并观察不同的响应。

如果您使用 Postman，创建一个新的 `GET` 请求，URL 为 `http://127.0.0.1:8000/authors/1`。发送此请求应该会产生类似的响应。

在 Postman 中，将查询参数附加到 URL，如下所示：`http://127.0.0.1:8000/books?year=2021`。发送此请求应该返回 2021 年出版的书籍。

## 参见

您可以在 FastAPI 官方文档中找到更多关于路径和查询参数的信息，以下是一些链接：

+   *路径* *参数*：[`fastapi.tiangolo.com/tutorial/path-params/`](https://fastapi.tiangolo.com/tutorial/path-params/)

+   *查询* *参数*：[`fastapi.tiangolo.com/tutorial/query-params/`](https://fastapi.tiangolo.com/tutorial/query-params/)

# 定义和使用请求和响应模型

在 API 开发的世界里，数据处理是决定您应用程序健壮性和可靠性的关键方面。FastAPI 通过与 **Pydantic** 的无缝集成简化了这一过程，Pydantic 是一个使用 Python 类型注解进行数据验证和设置管理的库。这个菜谱将向您展示如何在 FastAPI 中定义和使用请求和响应模型，确保您的数据结构良好、经过验证且定义清晰。

Pydantic 模型是数据验证和转换的强大功能。它们允许您定义应用程序处理的数据的结构、类型和约束，无论是传入请求还是传出响应。

在这个菜谱中，我们将看到如何使用 Pydantic 确保您的数据符合指定的模式，提供一层自动的安全性和清晰性。

## 准备工作

这个菜谱要求您知道如何在 FastAPI 中设置基本端点。

## 如何做到这一点…

我们将把整个过程分解为以下步骤：

1.  创建模型

1.  定义请求体

1.  验证请求数据

1.  管理响应格式

### 创建模型

让我们在名为 `models.py` 的新文件中为我们的书店应用程序创建一个 Pydantic `BaseModel` 类。

假设我们想要一个包含标题、作者和出版年份的书的模型：

```py
from pydantic import BaseModel
class Book(BaseModel):
    title: str
    author: str
    year: int
```

在这里，`Book` 是一个具有三个字段：`title`、`author` 和 `year` 的 Pydantic `BaseModel` 类。每个字段都有类型，确保任何符合此模型的数据都将具有这些属性和指定的数据类型。

### 定义请求体

在 FastAPI 中，Pydantic 模型不仅用于验证，还作为请求体。让我们在我们的应用程序中添加一个端点，让用户可以添加新书：

```py
from models import Book
@app.post("/book")
async def create_book(book: Book):
    return book
```

在此端点中，当用户向 `/book` 端点发送带有 JSON 数据的 `POST` 请求时，FastAPI 会自动解析并验证它是否与 `Book` 模型相匹配。如果数据无效，用户会收到自动的错误响应。

### 验证请求数据

Pydantic 提供了高级验证功能。例如，你可以添加正则表达式验证、默认值等：

```py
from pydantic import BaseModel, Field
class Book(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    author: str = Field(..., min_length=1, max_length=50)
    year: int = Field(..., gt=1900, lt=2100)
```

要查看完整的验证功能列表，请查看 Pydantic 的官方文档：[`docs.pydantic.dev/latest/concepts/fields/`](https://docs.pydantic.dev/latest/concepts/fields/)

接下来，你可以继续管理响应格式。

### 管理响应格式

FastAPI 允许你显式地定义响应模型，确保你的 API 返回的数据与特定的模式相匹配。这可以特别有用，用于过滤敏感数据或重新结构化响应。

例如，假设你想要 `/allbooks` `GET` 端点返回一本书的列表，但只包含它们的标题和作者，省略出版年份。在 `main.py` 中相应地添加以下内容：

```py
from pydantic import BaseModel
class BookResponse(BaseModel):
    title: str
    author: str
@app.get("/allbooks")
async def read_all_books() -> list[BookResponse]:
    return [
        {
            "id": 1,
            "title": "1984",
            "author": "George Orwell"},
        {
            "id": 1,
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
        },
    ]
```

在这里，`-> list[BookResponse]` 函数类型提示告诉 FastAPI 使用 `BookResponse` 模型进行响应，确保响应 JSON 中只包含标题和作者字段。或者，你可以在端点装饰器的参数中指定响应类型，如下所示：

```py
@app.get("/allbooks", response_model= list[BookResponse])
async def read_all_books() -> Any:
# rest of the endpoint content
```

`response_model` 参数具有优先级，可以用作替代类型提示来解决可能出现的类型检查问题。

请查阅文档，网址为 `http://127.0.0.1:8000/docs`。展开 `/allbooks` 端点详情，你会注意到基于以下模式的示例值响应：

```py
[
  {
    "title": "string",
    "author": "string"
  }
]
```

通过掌握 FastAPI 中的 Pydantic 模型，你现在可以轻松且精确地处理复杂的数据结构。你已经学会了如何定义请求体和管理响应格式，确保在整个应用程序中保持数据的一致性和完整性。

## 参考也

**Pydantic** 是一个独立的项目，主要用于 Python 中的数据验证，具有比示例中展示的更多功能。请自由查看以下链接的官方文档：

+   *Pydantic*：[`docs.pydantic.dev/latest/`](https://docs.pydantic.dev/latest/)

你可以在 FastAPI 官方文档链接中了解更多关于响应模型的使用：[`fastapi.tiangolo.com/`](https://fastapi.tiangolo.com/)

+   *响应模型 - 返回* *类型*: [`fastapi.tiangolo.com/tutorial/response-model/`](https://fastapi.tiangolo.com/tutorial/response-model/)

# 处理错误和异常

错误处理是开发健壮和可靠的 Web 应用程序的一个基本方面。在 FastAPI 中，管理错误和异常不仅涉及捕获意外问题，还包括积极设计您的应用程序以优雅地应对各种错误场景。

本教程将指导您通过自定义错误处理、验证数据和处理异常，以及测试这些场景，以确保您的 FastAPI 应用程序具有弹性和用户友好性。

## 如何做到这一点...

FastAPI 提供了处理异常和错误的内置支持。

当发生错误时，FastAPI 会返回一个包含错误详细信息的 JSON 响应，这对于调试非常有用。然而，在某些情况下，您可能希望自定义这些错误响应以提供更好的用户体验或安全性。

让我们创建一个自定义错误处理器来捕获特定类型的错误并返回自定义响应。例如，如果请求的资源未找到，您可能希望返回一个更友好的错误消息。

要实现这一点，在`main.py`文件中相应地添加以下代码：

```py
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": "Oops! Something went wrong"
        },
    )
```

在本例中，将使用`http_exception_handler`函数来处理`HTTPException`错误。只要您的应用程序中任何地方引发`HTTPException`错误，FastAPI 就会使用此处理器来返回自定义响应。

您可以通过创建一个新的端点来引发 HTTP 异常来测试响应：

```py
@app.get("/error_endpoint")
async def raise_exception():
    raise HTTPException(status_code=400)
```

该端点将明确抛出 HTTP 错误响应，以展示之前步骤中定义的自定义消息。

现在，使用以下命令从命令行启动服务器：

```py
http://localhost:8000/error_endpoint, and you will have a JSON response like this:

```

{

"message": "Oops! Something went wrong"

}

```py

 The response returns the default message we defined for any HTTP exception returned by the code.
There’s more…
As discussed in the previous recipe, *Defining and using request and response models*, FastAPI uses Pydantic models for data validation. When a request is made with data that does not conform to the defined model, FastAPI automatically raises an exception and returns an error response.
In some cases, you might want to customize the response for validation errors. FastAPI makes this quite straightforward:

```

import json

from fastapi import Request, status

from fastapi.exceptions import RequestValidationError

from fastapi.responses import PlainTextResponse

@app.exception_handler(RequestValidationError)

async def validation_exception_handler(

request: Request,

exc: RequestValidationError

):

return PlainTextResponse(

"这是一个纯文本响应："

f" \n{json.dumps(exc.errors(), indent=2)}",

status_code=status.HTTP_400_BAD_REQUEST,

)

```py

 This custom handler will catch any `RequestValidationError` error and return a plain text response with the details of the error.
If you try, for example, to call the `POST /book` endpoint with a number type of `title` instead of a string, you will get a response with a status code of `400` and body:

```

这是一个纯文本响应：

[

{

"type": "string_type",

"loc": [

"body",

"author"

],

"msg": "输入应是一个有效的字符串",

"input": 3,

"url": "https://errors.pydantic.dev/2.5/v/string_type"

},

{

"type": "greater_than",

"loc": [

"body",

"year"

],

"msg": "输入应大于 1900",

"input": 0,

"ctx": {

"gt": 1900

},

"url": "https://errors.pydantic.dev/2.5/v/greater_than"

}

]

```py

 You can also, for example, mask the message to add a layer of security to protect from unwanted users using it incorrectly.
This is all you need to customize responses when a request validation error occurs.
You will use this basic knowledge as you move to the next chapter. *Chapter 2* will teach you more about data management in web applications, showing you how to set up and use SQL and NoSQL databases and stressing data security. This will not only improve your technical skills but also increase your awareness of creating scalable and reliable FastAPI applications.
See also
You can find more information about customizing errors and exceptions using FastAPI in the official documentation:

*   *Handling* *Errors*: [`fastapi.tiangolo.com/tutorial/handling-errors/`](https://fastapi.tiangolo.com/tutorial/handling-errors/)

```
