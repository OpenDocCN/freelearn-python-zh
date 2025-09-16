

# 第八章：高级特性和最佳实践

欢迎来到 *第八章*，我们将探讨优化 FastAPI 应用程序功能、性能和可扩展性的高级技术和最佳实践。

在本章中，通过构建一个旅行社平台，您将深入了解依赖注入、自定义中间件、国际化、性能优化、速率限制和后台任务执行等基本主题。通过掌握这些高级特性，您将能够使用 FastAPI 构建强大、高效和高性能的 API。

到本章结束时，您将对高级 FastAPI 功能和最佳实践有一个全面的理解，这将使您能够构建高效、可扩展和安全的 API，以满足现代 Web 应用程序的需求。让我们深入探讨这些高级技术，以提升您的 FastAPI 开发技能。

在本章中，我们将涵盖以下配方：

+   实现依赖注入

+   创建自定义中间件

+   国际化和本地化

+   优化应用程序性能

+   实现速率限制

+   实现后台任务

# 技术要求

要能够跟随本章的配方，您必须对以下基本概念有很好的掌握：

+   **Python**：您应该对 Python 3.7 或更高版本有很好的理解。您应该了解注解的工作原理以及基本类继承。

+   `fastapi`、`asyncio`、`async`/`await` 语法。

本章中使用的代码托管在 GitHub 上，网址为 [`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter08`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter08)。

为了更高效地管理依赖项并保持项目隔离，考虑在 `project` 根目录下创建一个虚拟环境。您可以通过使用 GitHub 仓库中 `project` 文件夹提供的 `requirements.txt` 文件，轻松地同时安装所有依赖项：

```py
$  pip install –r requirements.txt
```

您可以从第一个配方开始，高效地在您的 FastAPI 应用程序中实现依赖注入。

# 实现依赖注入

**依赖注入**是一种在软件开发中用于管理组件之间依赖关系的强大设计模式。在 FastAPI 的上下文中，依赖注入允许您高效地管理和注入依赖项，例如数据库连接、认证服务和配置设置，到您的应用程序的端点和中间件中。尽管我们已经在之前的配方中使用了依赖注入，例如在 *第二章* *设置 SQL 数据库*、*处理数据* 或在 *第四章* *设置用户注册*、*认证和授权* 中，这个配方将向您展示如何在 FastAPI 中实现依赖注入，以及如何处理具有嵌套依赖注入的更复杂的使用案例。

## 准备工作

要跟随这个配方，你只需要在你的环境中安装了 Python，并且安装了 `fastapi` 和 `uvicorn` 包，以及 `pytest`。如果你还没有使用 GitHub 仓库中提供的 `requirements.txt` 文件安装这些包，你可以从命令行使用 `pip` 安装它们：

```py
$ pip install fastapi uvicorn pytest
```

此外，了解如何在 FastAPI 中创建一个简单的服务器将很有帮助。你可以参考 *第一章* 中的 *Creating a* *new* *FastAPI* *project* 配方，了解更多详情。

## 如何做到这一点……

让我们从创建一个名为 `trip_platform` 的项目根文件夹开始，该文件夹包含 `app` 文件夹。然后按照以下步骤继续进行。

1.  在 `app` 文件夹中，创建一个包含服务器的 `main.py` 模块，如下所示：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    ```

    我们将在 `app` 文件夹内创建一个名为 `dependencies.py` 的单独模块来编写依赖项。

1.  让我们假设我们需要创建一个端点来检索从开始日期到结束日期之间的所有行程。我们需要处理两个参数，即开始日期和结束日期，并检查开始日期是否早于结束日期。这两个参数都可以是可选的；如果未提供开始日期，则默认为当前日期。

    在 `app` 文件夹的 `dependencies.py` 专用模块中，让我们定义一个条件函数，该函数检查开始日期是否早于结束日期：

    ```py
    from fastapi import HTTPException
    def check_start_end_condition(start: date, end: date):
        if end and end < start:
            raise HTTPException(
                status_code=400,
                detail=(
                    "End date must be "
                    "greater than start date"
                ),
            )
    ```

1.  我们使用 `check_start_end_condition` 函数来定义 `dependable` 函数——即用作依赖项的函数——如下所示：

    ```py
    from datetime import date, timedelta
    from fastapi import Query
    def time_range(
        start: date | None = Query(
            default=date.today(),
            description=(
                "If not provided the current date is used"
            ),
            example=date.today().isoformat(),
        ),
        end: date | None = Query(
            None,
            example=date.today() + timedelta(days=7),
        ),
    ) -> Tuple[date, date | None]:
        check_start_end_condition(start, end)
        return start, end
    ```

    `Query` 对象用于管理查询参数的元数据，例如在生成文档时使用的默认值、描述和示例。

1.  我们可以使用可信赖的 `time_range` 函数在 `main.py` 模块中创建端点。为了指定它是一个依赖项，我们使用 `Depends` 对象，如下所示：

    ```py
    from fastapi import Depends
    @app.get("/v1/trips")
    def get_tours(
        time_range = Depends(time_range),
    ):
        start, end = time_range
        message = f"Request trips from {start}"
        if end:
            return f"{message} to {end}"
        return message
    ```

    你还可以使用 `typing` 包中的 `Annotated` 类来如下定义依赖项：

    ```py
    from typing import Annotated
    from fastapi import Depends
    @app.get("/v1/trips")
    def get_tours(
        time_range: Annotated[time_range, Depends()]
    ):
    ```

重要提示

FastAPI 中 `Annotated` 的使用目前正在演变，以避免重复并提高可读性；请参阅专门的文档部分：[`fastapi.tiangolo.com/tutorial/dependencies/#share-annotated-dependencies`](https://fastapi.tiangolo.com/tutorial/dependencies/#share-annotated-dependencies)。

对于本章的其余部分，我们将使用最新的 `Annotated` 习惯用法。

现在，如果你在终端中运行 `uvicorn app.main:app` 来启动服务器，你将在交互式文档的 `http://localhost:8000/docs` 中找到端点。你会看到你刚刚创建的端点，参数得到了正确的文档说明。示例中用字符串构造替换了数据库逻辑，返回了一条重要信息。

你刚刚实现了一个依赖注入策略来定义端点的查询参数。你可以使用相同的策略来编写路径或体参数，以编写模块化和可读的代码。

使用依赖注入的一个优点是逻辑上分离可以替换为其他东西的代码片段，比如在测试中。让我们看看如何做到这一点。

### 在测试中覆盖依赖注入

让我们为`GET /v1/trips`端点创建一个测试。如果你环境中没有`pytest`，请使用`pip install pytest`进行安装。然后，在项目根目录下创建`pytest.ini`文件，包含`pytest`的`pythonpath`，如下所示：

```py
[pytest]
pythonpath=.
```

测试将在`tests`文件夹下的`test_main.py`测试模块中进行。让我们通过覆盖客户端的依赖来编写一个单元测试：

```py
from datetime import date
from fastapi.testclient import TestClient
from app.dependencies import time_range
from app.main import app
def test_get_v1_trips_endpoint():
    client = TestClient(app)
    app.dependency_overrides[time_range] = lambda: (
        date.fromisoformat("2024-02-01"),
        None,
)
    response = client.get("/v1/trips")
    assert (
        response.json()
        == "Request trips from 2024-02-01"
    )
```

通过覆盖`time_range`依赖，我们不需要在调用端点时传递参数，响应将取决于定义的 lambda 函数。

然后，你可以从命令行运行测试：

```py
$ pytest tests
```

当编写不应干扰生产数据库的测试时，这项技术非常有用。如果测试不关心，最终的重计算逻辑也可以被模拟。

依赖注入的使用可以通过提高模块化程度显著提高测试质量。

## 它是如何工作的…

`Depends`对象和依赖注入利用 Python 强大的函数注解和类型提示功能。

当你定义一个依赖函数并用`Depends`进行注解时，FastAPI 将其解释为在执行端点函数之前需要解决的依赖。当对依赖于一个或多个依赖项的端点发出请求时，FastAPI 会检查端点函数的签名，识别依赖项，并通过调用正确的顺序中的相应依赖函数来解析它们。

FastAPI 使用 Python 的类型提示机制来确定每个依赖参数的类型，并自动将解析后的依赖注入到端点函数中。这个过程确保了所需的数据或服务在运行时对端点函数可用，从而实现了外部服务、数据库连接、身份验证机制和其他依赖与 FastAPI 应用程序的无缝集成。

总体而言，`Depends`类和 FastAPI 中的依赖注入提供了一种干净且高效的方式来管理依赖项，并促进模块化和可维护的代码架构。一个优点是它们可以在测试中重写，以便轻松模拟或替换。

## 还有更多…

我们可以通过利用子依赖来进一步推进。

让我们创建一个端点，该端点返回三个类别（巡游、城市之旅和度假住宿）之一的行程，并同时检查该类别的优惠券有效性。

在`dependencies.py`模块中，让我们为该类别创建一个`dependable`函数。

想象一下，我们可以将我们的旅行分为三类——邮轮之旅、城市短途游和度假村住宿。我们需要添加一个参数来检索特定类别的旅行。我们需要一个`dependable`函数，如下所示：

```py
def select_category(
    category: Annotated[
        str,
        Path(
            description=(
                "Kind of travel "
                "you are interested in"
            ),
            enum=[
                "Cruises",
                "City Breaks",
                "Resort Stay",
            ],
        ),
    ],
) -> str:
    return category
```

现在，让我们想象我们需要验证一个优惠券以获取折扣。

`dependable`函数将作为另一个用于检查优惠券的`dependable`函数的依赖项。让我们定义它，如下所示：

```py
def check_coupon_validity(
    category: Annotated[select_category, Depends()],
    code: str | None = Query(
        None, description="Coupon code"
    ),
) -> bool:
    coupon_dict = {
        "cruises": "CRUISE10",
        "city-breaks": "CITYBREAK15",
        "resort-stays": "RESORT20",
    }
    if (
        code is not None
        and coupon_dict.get(category, ...) == code
    ):
        return True
    return False
```

在`main.py`模块中，让我们定义一个新的端点，`GET /v2/trips/{category}`，它返回指定类别的旅行：

```py
@app.get("/v2/trips/{category}")
def get_trips_by_category(
    category: Annotated[select_category, Depends()],
    discount_applicable: Annotated[
        bool, Depends(check_coupon_validity)
    ],
):
    category = category.replace("-", " ").title()
    message = f"You requested {category} trips."
    if discount_applicable:
        message += (
            "\n. The coupon code is valid! "
            "You will get a discount!"
        )
    return message
```

如果你使用`uvicorn app.main:app`命令运行服务器，并在`http://localhost:8000/docs`打开交互式文档，你会看到新的端点。接受的参数`category`和`code`都来自依赖项，并且`category`参数在代码中不会重复。

重要提示

你可以使用`def`和`async def`关键字来声明依赖项，无论是同步函数还是异步函数。FastAPI 将自动处理它们。

你刚刚创建了一个使用嵌套依赖项的端点。通过使用嵌套依赖项和子依赖项，你将能够编写清晰且模块化的代码，这使得代码更容易阅读和维护。

练习

在 FastAPI 中，依赖项也可以作为一个类来创建。查看[`fastapi.tiangolo.com/tutorial/dependencies/classes-as-dependencies/#classes-as-dependencies`](https://fastapi.tiangolo.com/tutorial/dependencies/classes-as-dependencies/#classes-as-dependencies)的文档，并创建一个新的端点，该端点使用我们在配方中定义的所有参数（`time_range`、`category`和`code`）。

将所有参数组合到一个类中，并定义和使用它作为端点的依赖项。

## 参见

我们已经使用了`Query`和`Path`描述符对象来设置`query`和`path`参数的元数据和文档相关数据。你可以在这些文档链接中了解更多关于它们潜力的信息：

+   *查询参数和字符串* *验证*：[`fastapi.tiangolo.com/tutorial/query-params-str-validations/`](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/)

+   *路径参数和数字* *验证*：[`fastapi.tiangolo.com/tutorial/path-params-numeric-validations/`](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/)

对于 FastAPI 中的依赖注入，你可以找到广泛的文档，涵盖了所有可能的用法，解释了这一强大功能的潜力：

+   *依赖项*：[`fastapi.tiangolo.com/tutorial/dependencies/`](https://fastapi.tiangolo.com/tutorial/dependencies/)

+   *高级* *依赖项*：[`fastapi.tiangolo.com/advanced/advanced-dependencies/`](https://fastapi.tiangolo.com/advanced/advanced-dependencies/)

+   *使用* *覆盖* *测试依赖项*：[`fastapi.tiangolo.com/advanced/testing-dependencies/`](https://fastapi.tiangolo.com/advanced/testing-dependencies/)

# 创建自定义中间件

**中间件** 是一个 API 组件，允许您拦截和修改传入的请求和传出的响应，使其成为实现跨切面关注点（如身份验证、日志记录和错误处理）的强大工具。

在本菜谱中，我们将探讨如何在 FastAPI 应用程序中开发自定义中间件以处理请求和响应，并检索客户端信息。

## 准备中…

您只需要有一个运行的 FastAPI 应用程序。菜谱将使用我们在之前的菜谱中定义的旅行平台，*实现依赖注入*。然而，中间件适用于通用运行的应用程序。

## 如何实现…

我们将通过以下步骤向您展示如何创建一个自定义中间件对象类，我们将在我们的应用程序中使用它。

1.  让我们在应用文件夹中创建一个名为 `middleware.py` 的专用模块。

    我们希望中间件能够拦截请求并打印客户端的主机和方法到输出终端。在实际的应用场景中，这些信息可以存储在数据库中进行分析或用于安全检查。

    让我们使用 FastAPI 默认使用的相同 `uvicorn` 日志记录器：

    ```py
    import logging
    logger = logging.getLogger("uvicorn.error")
    ```

1.  然后，让我们创建我们的 `ClientInfoMiddleware` 类，如下所示：

    ```py
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    class ClientInfoMiddleware(BaseHTTPMiddleware):
        async def dispatch(
            self, request: Request, call_next
        ):
            host_client = request.client.host
            requested_path = request.url.path
            method = request.method
            logger.info(
                f"host client {host_client} "
                f"requested {method} {requested_path} "
                "endpoint"
            )
            return await call_next(request)
    ```

1.  然后，我们需要在 `main.py` 中将我们的中间件添加到 FastAPI 服务器。在定义应用服务器后，我们可以使用 `add_middleware` 方法添加中间件：

    ```py
    # main.py import modules
    from app.middleware import ClientInfoMiddleware
    app = FastAPI()
    app.add_middleware(ClientInfoMiddleware)
    # rest of the code
    ```

现在，使用 `uvicorn app.main:app` 命令启动服务器，并尝试连接到 `http://localhost:8000/v1/trips` 的子路径。您甚至不需要调用现有的端点。您将在应用程序输出终端中看到日志消息：

```py
INFO:host client 127.0.0.1 requested GET /v1/trips endpoint
```

您刚刚实现了一个基本的自定义中间件来检索客户端信息。您可以通过添加更多操作来增加其复杂性，例如根据 IP 重定向请求或集成 IP 阻止或过滤。

## 工作原理…

FastAPI 使用来自 `Starlette` 库的 `BasicHTTPMiddleware` 类。菜谱中展示的策略创建了一个从 `BasicHTTPMiddleware` 派生的类，它具有一个特定的 `dispatch` 方法，用于实现拦截操作。

要在 FastAPI 中创建中间件，您可以将 FastAPI 类方法中的一个装饰器添加到一个简单的函数上。然而，建议创建一个类，因为它允许更好的模块化和代码组织。通过创建一个类，您最终可以创建自己的中间件集合模块。

## 参考信息

您可以查看以下链接的官方文档页面，了解如何创建自定义中间件：

+   *FastAPI 中间件* 文档：[`fastapi.tiangolo.com/tutorial/middleware/`](https://fastapi.tiangolo.com/tutorial/middleware/)

在 **Stack Overflow** 网站上可以找到一个有趣的讨论，关于如何在 FastAPI 中创建中间件类：

+   *创建 FastAPI 自定义中间件类* *讨论*：[`stackoverflow.com/questions/71525132/how-to-write-a-custom-fastapi-middleware-class`](https://stackoverflow.com/questions/71525132/how-to-write-a-custom-fastapi-middleware-class)

# 国际化和本地化

**国际化**（**i18n**）和**本地化**（**l10n**）是软件开发中的基本概念，它使得应用程序能够适应不同的语言、地区和文化。

**i18n**指的是设计和开发能够适应不同语言和文化的软件或产品的过程。这个过程主要涉及提供特定语言的内容。相反，**l10n**涉及将产品或内容适应特定地区或市场，例如货币或度量单位。

在我们的旅行平台中实现 i18n 和 l10n 的`Accept-Language`头。这将使我们的平台能够向客户端提供有针对性的内容。

## 准备工作

了解`Accept-Language`头信息会有所帮助；请查看 Mozilla 文档中的这篇有趣文章：[`developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Language`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Language).

你需要有一个正在运行的 FastAPI 应用程序来遵循这个配方。你可以跟随本章中使用的旅行平台应用程序。

我们将使用依赖注入，因此完成本章的*实现依赖注入*配方会有所帮助。

此外，我们还将使用`babel`包来解析语言代码引用，所以如果你还没有通过`requirements.txt`文件安装这些包，请确保通过运行以下命令在你的环境中安装`babel`：

```py
$ pip install babel
```

安装完成后，你将拥有启动所需的一切。

## 如何实现它...

首先，我们必须确定我们希望服务哪些地区和语言。在这个例子中，我们将关注两个——**美国英语**（**en_US**）和**来自法国的法语**（**fr_FR**）。所有与语言相关的内 容都将使用这两种语言之一。

在主机客户端侧管理`Accept-Language`头是必要的，它是一个带有偏好权重参数的语言列表。

该头的示例如下：

```py
Accept-Language: en
Accept-Language: en, fr
Accept-Language: en-US
Accept-Language: en-US;q=0.8, fr;q=0.5
Accept-Language: en, *
Accept-Language: en-US, en-GB
Accept-Language: zh-Hans-CN
```

我们需要一个函数，它接受头信息和我们的应用程序中可用的语言列表作为参数，并返回最合适的一个。让我们通过以下步骤来实现它。

1.  在`app`文件夹下创建一个专门的模块，`internationalization.py`。

    首先，我们将支持的语言存储在一个变量中，如下所示：

    ```py
    SUPPORTED_LOCALES = [
        "en_US",
        "fr_FR",
    ]
    ```

1.  然后，我们开始定义`resolve_accept_language`函数，如下所示：

    ```py
    from babel import Locale, negotiate_locale
    def resolve_accept_language(
        accept_language: str = Header("en-US"),
    ) -> Locale:
    ```

1.  在函数内部，我们将字符串解析成一个列表：

    ```py
        client_locales = []
        for language_q in accept_language.split(","):
            if ";q=" in language_q:
                language, q = language_q.split(";q=")
            else:
                language, q = (language_q, float("inf"))
            try:
                Locale.parse(language, sep="-")
                client_locales.append(
                    (language, float(q))
                )
            except ValueError:
                continue
    ```

1.  然后我们根据偏好`q`参数对字符串进行排序：

    ```py
        client_locales.sort(
            key=lambda x: x[1], reverse=True
        )
        locales = [locale for locale, _ in client_locales]
    ```

1.  然后，我们使用`babel`包中的`negotiate_locale`来获取最合适的语言：

    ```py
        locale = negotiate_locale(
            [str(locale) for locale in locales],
            SUPPORTED_LOCALES,
        )
    ```

1.  如果没有匹配项，我们返回默认的 `en_US`：

    ```py
        if locale is None:
            locale = "en_US"
        return locale
    ```

    `resolve_accept_language` 函数将被用作根据语言返回内容的端点的依赖项。

1.  在相同的 `internationalization.py` 模块中，让我们创建一个返回欢迎字符串的 `GET /homepage` 端点，这个字符串取决于语言。我们将在一个单独的 `APIRouter` 中完成它，因此路由器将如下所示：

    ```py
    from fastapi import APIRouter
    router = APIRouter(
        tags=["Localizad Content Endpoints"]
    )
    ```

    `tags` 参数指定路由器的端点将在交互式文档中根据指定的标签名称单独分组。

    `GET /home` 端点将如下所示：

    ```py
    home_page_content = {
        "en_US": "Welcome to Trip Platform",
        "fr_FR": "Bienvenue sur Trip Platform",
    }
    @router.get("/homepage")
    async def home(
        request: Request,
        language: Annotated[
            resolve_accept_language, Depends()
        ],
    ):
        return {"message": home_page_content[language]}
    ```

    在示例中，内容已被硬编码为一个以语言代码为字典键的 `dict` 对象。

    在实际场景中，内容应该存储在数据库中，每个语言一个。

    类似地，你可以定义一个本地化策略来检索货币。

1.  让我们创建一个 `GET /show/currency` 端点作为示例，该端点使用依赖项从 `Accept-Language` 标头检索货币。`dependency` 函数可以定义如下：

    ```py
    async def get_currency(
        language: Annotated[
            resolve_accept_language, Depends()
        ],
    ):
        currencies = {
            "en_US": "USD",
            "fr_FR": "EUR",
        }
        return currencies[language]
    ```

    端点将如下所示：

    ```py
    from babel.numbers import get_currency_name
    @router.get("/show/currency")
    async def show_currency(
        currency: Annotated[get_currency, Depends()],
        language: Annotated[
            resolve_accept_language,
            Depends(use_cache=True)
        ],
    ):
        currency_name = get_currency_name(
            currency, locale=language
        )
        return {
            "currency": currency,
            "currency_name": currency_name,
        }
    ```

1.  要使用这两个端点，我们需要在 `main.py` 中将路由器添加到 FastAPI 对象中：

    ```py
    from app import internationalization
    # rest of the code
    app.include_router(internationalization.router)
    ```

这就是实现国际化本地化的全部内容。要测试它，请在命令行中运行以下命令来启动服务器：

```py
http:localhost:8000/docs, you will find the GET /homepage and GET /show/currency endpoints. Both accept the Accept-Language header to provide the language choice; if you don’t, it will get the default language from the browser. To test the implementation, try experimenting with different values for the header.
You have successfully implemented internationalization and localization from scratch for your API. Using the recipe provided, you have integrated i18n and l10n into your applications, making them easily understandable worldwide.
See also
You can find out more about the potential of `Babel` package on the official documentation page: [`babel.pocoo.org/en/latest/`](https://babel.pocoo.org/en/latest/).
Optimizing application performance
Optimizing FastAPI applications is crucial for ensuring high performance and scalability, especially under heavy loads.
In this recipe, we’ll see a technique to profile our FastAPI application and explore actionable strategies to optimize performances. By the end of the recipe, you will be able to detect code bottlenecks and optimize your application.
Getting ready
Before starting the recipe, make sure to have a FastAPI application running with some endpoints already set up. You can follow along with our trip platform application.
We will be using the `pyinstrument` package to set up a profiler for the application. If you haven’t installed the packages with `requirements.txt`, you can install `pyinstrument` in your environment by running the following:

```

$ pip install pyinstrument

```py

 Also, it can be useful to have a look at the *Creating* *custom middleware* recipe from earlier in the chapter.
How to do it…
Let's implement the profiler in simple steps.

1.  Under the app folder, create a `profiler.py` module as follows:

    ```

    from pyinstrument import Profiler

    profiler = Profiler(

    interval=0.001, async_mode="enabled"

    )

    ```py

    The `async_mode="enabled"` parameter specifies that the profiler logs the time each time it encounters an `await` statement in the function being awaited, rather than observing other coroutines or the event loop. The `interval` specifies the time between two samples.

     2.  Before using the profiler, we should plan what we want to profile. Let’s plan to profile only the code executed in the endpoints. To do this, we can create simple middleware in a separate module that starts and stops the profiler before and after each call, respectively. We can create the middleware in the same `profiler.py` module, as follows:

    ```

    from starlette.middleware.base import (

    BaseHTTPMiddleware

    )

    class ProfileEndpointsMiddleWare(

    BaseHTTPMiddleware

    ):

    async def dispatch(

    self, request: Request, call_next

    ):

    if not profiler.is_running:

    profiler.start()

    response = await call_next(request)

    if profiler.is_running:

    profiler.stop()

    profiler.write_html(

    os.getcwd() + "/profiler.xhtml"

    )

    profiler.start()

    return response

    ```py

    The profiler is initiated every time an endpoint is requested, and it is terminated after the request is complete. However, since the server operates asynchronously, there is a possibility that the profiler may already be running, due to another endpoint request. This can result in errors during the start and stop of the profiler. To prevent this, we verify before each request whether the profiler is not already running. After the request, we check whether the profiler is running before terminating it.

     3.  You can attach the profiler to the FastAPI server by adding the middleware in the `main.py` module, as we did in the *Creating custom* *middleware* recipe:

    ```

    app.add_middleware(ProfileEndpointsMiddleWare)

    ```py

To test the profiler, spin up the server by running `uvicorn app.main:app`. Once you start making some calls, you can do it from the interactive documentation at http://localhost:8000/docs. Then, a `profiler.xhtml` file will be created. You can open the file with a simple browser and check the status of the code.
You have just integrated a profiler into your FastAPI application.
There’s more...
Integrating a profiler is the first step that allows you to spot code bottlenecks and optimize the performance of your application.
Let’s explore some techniques to optimize the performance of your FastAPI performances:

*   `Starlette` library and supports asynchronous request handlers, using the `async` and `await` keywords. By leveraging asynchronous programming, you can maximize CPU and **input/output** (**I/O**) utilization, reducing response times and improving scalability.
*   **Scaling Uvicorn workers**: Increasing the number of Uvicorn workers distributes incoming requests across multiple processes. However, it might not be always the best solution. For purely I/O operations, asynchronous programming massively reduces CPU usage, and additional workers remain idle. Before adding additional workers, check the CPU usage of the main process.
*   **Caching**: Implement caching mechanisms to store and reuse frequently accessed data, reducing database queries and computation overhead. Use dedicated libraries l to integrate caching into your FastAPI applications.

Other techniques are related to external libraries or tools, and whatever strategy you use, make sure to properly validate it with proper profiling configuration.
Also, for high-traffic testing, take a look at the *Performance testing for high traffic applications* recipe in *Chapter 5*, *Testing and Debugging* *FastAPI Applications*.
Exercise
We learned how to configure middleware to profile applications; however, it is more common to create tests to profile specific use cases. We learned how to configure middleware to profile applications; however, it is more common to create test scripts to profile specific use cases. Try to create one by yourself that attaches the profiler to the server, runs the server, makes API calls that reproduce the use case, and finally, writes the profiler output. The solution is provided on the GitHub repository in the `profiling_application.py` file. You can find it at [`github.com/PacktPublishing/FastAPI-Cookbook/blob/main/Chapter08/trip_platform/profiling_application.py`](https://github.com/PacktPublishing/FastAPI-Cookbook/blob/main/Chapter08/trip_platform/profiling_application.py).
See also
You can discover more about the potential of **pyinstrument** profiler on the official documentation:

*   *pyinstrument* *documentation*: [`pyinstrument.readthedocs.io/en/latest/`](https://pyinstrument.readthedocs.io/en/latest/)

Also, you can find a different approach to profile FastAPI endpoints on the page:

*   *pyinstrument – profiling FastAPI* *requests*: [`pyinstrument.readthedocs.io/en/latest/guide.xhtml#profile-a-web-request-in-fastapi`](https://pyinstrument.readthedocs.io/en/latest/guide.xhtml#profile-a-web-request-in-fastapi)

Implementing rate limiting
**Rate limiting** is an essential technique used to control and manage the flow of traffic to web applications, ensuring optimal performance, resource utilization, and protection against abuse or overload. In this recipe, we’ll explore how to implement rate limiting in FastAPI applications to safeguard against potential abuse, mitigate security risks, and optimize application responsiveness. By the end of this recipe, you’ll have a solid understanding of how to leverage rate limiting to enhance the security, reliability, and scalability of your FastAPI applications, ensuring optimal performance under varying traffic conditions and usage patterns.
Getting ready
To follow the recipe, you need a running FastAPI application with some endpoints to use for rate limiting. To implement rate limiting, we will use the `slowapi` package; if you haven’t installed the packages with the `requirements.txt` file provided in the GitHub repository, you can install `slowapi` in your environment with `pip` by running the following:

```

$ pip install slowapi

```py

 Once the installation is completed, you are ready to start the recipe.
How to do it…
We will start by applying a rate limiter to a single endpoint in simple steps.

1.  Let’s create the `rate_limiter.py` module under the `app` folder that contains our limiter object class defined as follows:

    ```

    from slowapi import Limiter

    from slowapi.util import get_remote_address

    limiter = Limiter(

    key_func=get_remote_address,

    )

    ```py

    The limiter is designed to restrict the number of requests from a client based on their IP address. It is possible to create a function that can detect a user’s credentials and limit their calls according to their specific user profile. However, for the purpose of this example, we will use the client’s IP address to implement the limiter.

     2.  Now, we need to configure the FastAPI server to implement the limiter. In `main.py`, we have to add the following configuration:

    ```

    from slowapi import _rate_limit_exceeded_handler

    from slowapi.errors import RateLimitExceeded

    # 代码的其余部分

    app.state.limiter = limiter

    app.add_exception_handler(

    RateLimitExceeded, _rate_limit_exceeded_handler

    )

    # 代码的其余部分

    ```py

     3.  Now, we will apply a rate limit of two requests per minute to the `GET /homepage` endpoint defined in the `internalization.py` module:

    ```

    from fastapi import Request

    from app.rate_limiter import limiter

    @router.get("/homepage")

    @limiter.limit("2/minute")

    async def home(

    request: Request,

    language: Annotated[

    resolve_accept_language, Depends()

    ],

    ):

    return {"message": home_page_content[language]}

    ```py

    The rate limit is applied as a decorator. Also, the request parameter needs to be added to make the limiter work.

Now, spin up the server from the command line by running the following:

```

http://localhost:8000/homepage; 你将获得主页内容，第三次调用时，你将获得一个包含以下内容的 429 响应：

```py
{
    "error": "Rate limit exceeded: 2 per 1 minute"
}
```

你刚刚为 `GET /homepage` 端点添加了速率限制。使用相同的策略，你可以为每个端点添加特定的速率限制器。

还有更多...

你可以通过添加全局速率限制到整个应用程序来做更多，如下所示。

在 `main.py` 中，你需要添加一个专门的中间件，如下所示：

```py
# rest of the code in main.py
from slowapi.middleware import SlowAPIMiddleware
# rest of the code
app.add_exception_handler(
    RateLimitExceeded, _rate_limit_exceeded_handler
)
app.add_middleware(SlowAPIMiddleware)
```

然后，你只需在 `rate_limiter.py` 模块的 `Limiter` 对象实例化中指定默认限制即可：

```py
limiter = Limiter(
    key_func=get_remote_address,
default_limits=["5/minute"],
)
```

就这样。现在，如果你重新启动服务器并连续调用任何端点超过五次，你将收到 `429` 响应。

你已经成功为你的 FastAPI 应用程序设置了一个全局速率限制器。

相关内容

你可以在官方文档中找到更多关于 **Slowapi** 功能的信息，例如共享限制、限制策略等，链接如下：

+   *SlowApi* *文档*：[`slowapi.readthedocs.io/en/latest/`](https://slowapi.readthedocs.io/en/latest/)

你可以在 **Limits** 项目文档中找到更多关于速率限制表示法语法的详细信息，链接如下：

+   *速率限制字符串* *表示法*：[`limits.readthedocs.io/en/stable/quickstart.xhtml#rate-limit-string-notation`](https://limits.readthedocs.io/en/stable/quickstart.xhtml#rate-limit-string-notation)

实现背景任务

背景任务是一个有用的功能，它允许你将资源密集型操作委托给单独的进程。使用背景任务，你的应用程序可以保持响应性并同时处理多个请求。这对于处理长时间运行的过程而不阻塞主请求-响应周期尤为重要。这提高了应用程序的整体效率和可扩展性。在本教程中，我们将探讨如何在 FastAPI 应用程序中执行背景任务。

准备工作

要遵循这个教程，你只需要一个运行着至少一个端点以应用背景任务 FastAPI 应用程序。然而，我们将把背景任务实现到我们的行程平台中的 `GET /v2/trips/{category}` 端点，该端点在 *实现依赖注入* 教程中定义。

如何操作...

让我们假设我们想要将 `GET /v2/trips/{category}` 端点的消息存储在外部数据库中，用于分析目的。让我们分两步简单完成。

1.  首先，我们定义一个函数，在 `app` 文件夹中的专用模块 `background_tasks.py` 中模拟存储操作。该函数看起来如下：

    ```py
    import asyncio
    import logging
    logger = logging.getLogger("uvicorn.error")
    async def store_query_to_external_db(message: str):
        logger.info(f"Storing message '{message}'.")
        await asyncio.sleep(2)
        logger.info(f"Message '{message}' stored!")
    ```

    存储操作通过 `asyncio.sleep` 非阻塞操作进行模拟。我们还添加了一些日志消息以跟踪执行情况。

    2.  现在，我们需要将 `store_query_to_external_db` 函数作为我们端点的背景任务执行。在 `main.py` 中，让我们修改 `GET /v2/trips/cruises`，如下所示：

    ```py
    from fastapi import BackgroundTasks
    @app.get("/v2/trips/{category}")
    def get_trips_by_category(
        background_tasks: BackgroundTasks,
        category: Annotated[select_category, Depends()],
        discount_applicable: Annotated[
            bool, Depends(check_coupon_validity)
        ],
    ):
        category = category.replace("-", " ").title()
        message = f"You requested {category} trips."
        if discount_applicable:
            message += (
                "\n. The coupon code is valid! "
                "You will get a discount!"
            )
        background_tasks.add_task(
            store_query_to_external_db, message
        )
        logger.info(
            "Query sent to background task, "
            "end of request."
        )
        return message
    ```

现在，如果你使用 `uvicorn app.main:app` 启动服务器并尝试调用 `GET /v2/trips/cruises` 端点，你将在终端输出中看到 `store_query_to_external_db` 函数的日志。

```py
INFO:  Query sent to background task, end of request.
INFO:  127.0.0.1:58544 - "GET /v2/trips/cruises
INFO:  Storing message 'You requested Cruises trips.'
INFO:  Message 'You requested Cruises trips.' Stored!
```

这就是你在 FastAPI 中实现后台任务所需的所有内容！然而，如果你必须执行大量的后台计算，你可能想要使用专门的工具来处理队列任务执行。这将允许你在单独的进程中运行任务，避免在相同进程中运行时可能出现的任何性能问题。

它是如何工作的…

当对端点发起请求时，后台任务会被排队到`BackgroundTasks`对象。所有任务都会传递给事件循环，以便它们可以并发执行，从而允许非阻塞 I/O 操作。

如果你有一个需要大量处理能力且不一定需要由相同过程完成的任务，你可能想要考虑使用像 Celery 这样的大型工具。

参见

你可以在官方文档页面上的此链接找到更多关于在 FastAPI 中创建后台任务的信息：

+   *后台* *任务*：[`fastapi.tiangolo.com/tutorial/background-tasks/`](https://fastapi.tiangolo.com/tutorial/background-tasks/)

```py

```
