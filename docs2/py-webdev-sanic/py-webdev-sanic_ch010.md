# 提高你的 Web 应用程序的 9 个最佳实践

从*第一章*到*第八章*，我们学习了如何从构思到部署构建 Web 应用程序。给自己鼓掌，给自己来个满堂彩。构建和部署 Web 应用程序不是一件简单的事情。那么，我们学到了什么呢？我们当然花了时间学习 Sanic 提供的所有基本工具：路由处理程序、蓝图、中间件、信号、监听器、装饰器、异常处理程序等等。然而，更重要的是，我们花了时间思考 HTTP 是如何工作的，以及我们如何可以使用这些工具来设计和构建安全、可扩展、可维护且易于部署的应用程序。

这本书中有许多特定的模式供你使用，但我故意留下很多模糊性。你不断地读到这样的声明：“*这取决于你的应用程序需求*。”毕竟，Sanic 项目的目标之一是保持“*无偏见*”。

这听起来很好，而且灵活性是极好的。但是，如果你是一个尚未确定哪些模式有效，哪些无效的开发者呢？编写一个“*Hello, world*”应用程序和编写一个生产就绪、真实世界的应用程序之间的差异是巨大的。如果你在编写应用程序方面只有有限的经验，那么你在犯错误方面的经验也是有限的。正是通过这些错误（无论是你自己犯的，还是从犯过错误的其他人那里学到的教训），我真正相信我们成为了更好的开发者。就像生活中许多其他事情一样，失败导致成功。

因此，本章的目的是包括我从 20 多年的构建 Web 应用程序中学到的几个示例和*偏好*。这意味着在本章中你将学习的每一个最佳实践，可能都伴随着我犯过的某个*错误*。这是一套基础级别的*最佳实践*，我认为对于任何专业级应用程序从一开始就包括在内是至关重要的。

在本章中，我们将探讨：

+   实用的真实世界异常处理程序

+   如何设置可测试的应用程序

+   真实世界日志和跟踪的好处

+   管理数据库连接

## 技术要求

没有新的技术要求是你之前没有见过的。到这一点，你希望有一个适合构建 Sanic 的环境，以及我们一直在使用的所有工具，如 Docker、Git、Kubernetes 和 Curl。你可以在 GitHub 仓库上的代码示例中跟随：[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09)。

## 实施实用的真实世界异常处理程序

到目前为止，异常处理不是一个新概念。我们在第六章的 *实现适当的异常处理* 部分探讨了这一主题。我强调了创建我们自己的异常集的重要性，这些异常包括默认状态消息和响应代码。这个有用的模式旨在让你能够快速启动并运行，以便能够向用户发送 *有用的* 消息。

例如，设想我们正在为旅行代理人开发一个应用程序，用于为客户预订机票。你可以想象操作步骤之一可能是协助通过连接机场匹配航班。

好吧，如果客户选择了两个时间间隔太短的航班，你可能会这样做：

```py
from sanic.exceptions import SanicException
class InsufficientConnection(SanicException):
    status_code = 400
    message = "Selected flights do no leave enough time for connection to be made."
```

我喜欢这个模式，因为它使我们现在能够重复性地抛出 `InsufficientConnection` 异常，并为用户提供已知的响应。但是，正确地响应用户只是战斗的一半。在我们的应用程序的 *现实世界* 中，当发生错误时，我们想要知道。我们的应用程序需要能够报告问题，以便如果确实存在问题，我们可以修复它。

那么，我们如何解决这个问题呢？日志记录当然是必不可少的（我们将在后面的 *从日志和跟踪中获得洞察力* 部分很快探讨这一点）。有一个可靠的方式来获取你的系统日志是绝对必须的，出于很多原因。但你真的想每天整天监控你的日志，寻找跟踪信息吗？当然不想！

以某种方式，你需要设置一些警报来通知你发生了异常。你会了解到并非所有异常都是平等的，而且只有有时你才真正希望你的注意力被吸引到异常发生的事实上。如果客户忘记输入有效数据，你不需要在凌晨 3 点被你的手机吵醒。虽然设置系统监控和警报工具超出了本书的范围，但我试图说明的是，你的应用程序应该主动警告你在某些事情发生时。有时会发生不好的事情，你想要确保你能够从噪音中筛选出真正重要的问题。这种简单形式可能是在发生特别糟糕的事情时发送一封电子邮件。

根据你迄今为止对 Sanic 的了解，如果我来找你并要求你构建一个系统，每当抛出 `PinkElephantError` 异常时，就给我发送一封电子邮件，你会怎么做？

我希望这并不是你的答案：

```py
if there_is_a_pink_elephant():
    await send_adam_an_email()
    raise PinkElephantError
```

“*为什么？*”你可能会问。首先，如果需要在几个地方实现这个功能，然后我们需要将通知从 `send_adam_an_email()` 更改为 `build_a_fire_and_send_a_smoke_signal()`，那会怎样？现在你需要搜索所有代码以确保它是一致的，并希望你没有错过任何东西。

你还能做什么？你如何在应用程序中简单地编写以下代码，并让它知道它需要给我发送电子邮件？

```py
if there_is_a_pink_elephant():
    raise PinkElephantError
```

我们下次再学习这个。

### 使用中间件捕获错误

在我们抛出异常的地方旁边添加通知机制（在这种情况下，`send_adam_an_email()`）并不是最好的解决方案。一个解决方案是使用响应中间件捕获异常，并从那里发送警报。响应不太可能有一个易于解析的异常供你使用。如果 `PinkElephantError` 抛出一个 400 响应，你如何能够将它与其他任何 400 响应区分开来？当然，你可以有 JSON 格式，并检查异常类型。但这只能在 `DEBUG` 模式下工作，因为在 `PRODUCTION` 模式下，你无法获得这些信息。

一个创造性的解决方案可能是附加一个任意的异常代码，并在中间件中按如下方式重写：

```py
class PinkElephantError(SanicException):
    status_code = 4000
    message = "There is a pink elephant in the room"
@app.on_response
async def exception_middleware(request, response):
    if response.status == 4000:
        response.status = 400
        await send_adam_an_email()
```

对于一些非常具体的用例，我可以想象这可能会很有用，如果你觉得它对你有用，我会鼓励你这样做。它确实让我想起了老式的错误编码风格。你知道的，那种你需要查找表将数字转换为错误，而这个错误仍然有些难以理解，因为没有标准化或文档的情况？仅仅想象在我四处寻找用户手册来查找 *E19* 的含义时，我的咖啡机上的这个错误代码就足以提高我的压力水平。我想说的是：“*省去麻烦，尝试找到一种比附加一些难以理解的错误代码更好的方法来识别异常，这些错误代码你后来还需要翻译成其他东西。*”

### 使用信号捕获错误

记得我们以前的老朋友信号吗？在 *第六章* 的 *利用信号进行工作间通信* 部分中提到的？如果你还记得，Sanic 在某些事情发生时会派发事件信号。其中之一就是当抛出异常时。更好的是，信号上下文包括异常实例，这使得识别哪个异常发生了变得 *非常* 容易。

对于上述代码的一个更干净、更易于维护的解决方案如下：

```py
@app.signal("http.lifecycle.exception")
async def exception_signal(request, exception):
    if isinstance(exception, PinkElephantError):
        await send_adam_an_email()
```

我想你已经可以看出这是一个更优雅、更合适的解决方案。对于许多用例来说，这可能是你最好的解决方案。因此，我建议你记住这个简单的 4 行模式。现在，当我们需要将 `send_adam_an_email()` 更改为 `build_a_fire_and_send_a_smoke_signal()` 时，这将是一个超级简单的代码更改。

长期以来一直在构建 Sanic 应用程序的开发者可能会看到我的这个例子，想知道我们是否可以直接使用 `app.exception`？这当然是一个可接受的模式，但并非没有潜在的风险。我们接下来看看这个问题。

### 捕获错误并手动响应

当异常被抛出时，Sanic 会停止常规的路由处理过程，并将其移动到`ErrorHandler`实例。这是一个在整个应用程序实例生命周期中存在的单个对象。它的目的是充当一种迷你路由器，接收传入的异常并确保它们被传递到适当的异常处理程序。如果没有，则使用默认的异常处理程序。正如我们之前所看到的，默认处理程序是我们可以通过使用`error_format`参数来修改的。

这里有一个快速示例，展示了 Sanic 中的异常处理程序是什么样的：

```py
@app.exception(PinkElephantError)
async def handle_pink_elephants(request, exception):
    ...
```

这个模式的问题在于，因为你接管了实际的异常处理，现在你需要负责提供适当的响应。如果你构建了一个包含 10 个、20 个甚至更多这些异常处理程序的应用程序，保持它们的响应一致性就会变成一项繁琐的工作。

正是因为这个原因，我真正地尽量避免自定义异常处理，除非我需要。根据我的经验，通过控制格式化（如第六章“操作在请求处理程序之外”的*Fallback 处理*部分中讨论的）可以获得更好的结果。然而，这里有一个需要注意的例外，我们将在下一节中探讨。 

我尽量避免只针对单一用例进行一次性响应定制。在构建应用程序时，我们可能需要为许多类型的异常构建错误处理程序，而不仅仅是`PinkElephantError`。因此，当需要处理错误（如发送邮件）而不是仅仅处理用户输出的格式时，我通常不倾向于使用异常处理程序。

好吧，好吧，我认输了。我会告诉你一个秘密：你仍然可以使用`app.exception`模式来拦截错误，对它进行一些操作，然后使用内置的错误格式化。如果你更喜欢异常处理程序模式而不是信号，那么你可以使用它，而无需担心我关于格式化太多自定义错误响应的担忧。让我们看看我们如何实现这一点。

1.  首先，让我们创建一个简单的端点来抛出我们的错误，并以文本格式报告：

    ```py
    class PinkElephantError(SanicException):
        status_code = 400
        message = "There is a pink elephant in the room"
        quiet = True
    @app.get("/", error_format="text")
    async def handler(request: Request):
        raise PinkElephantError
    ```

    我已经将`quiet=True`添加到异常中，这样就会抑制跟踪信息被记录。当跟踪信息对你来说不重要，只是妨碍了你的工作，这是一个有用的技术。

1.  接下来，创建一个异常处理程序来发送邮件，但仍然使用默认的错误响应：

    ```py
    async def send_adam_an_email():
        print("EMAIL ADAM")
    @app.exception(PinkElephantError)
    async def handle_pink_elephants(request, exception):
        await send_adam_an_email()
        return request.app.error_handler.default(request, exception)
    ```

我们可以使用应用程序实例来访问默认的`ErrorHandler`实例，如前述代码所示。

我希望你能使用`curl`来访问那个端点，这样你就可以看到它按预期工作。你应该得到默认的文本响应，并看到日志中记录了一个模拟的邮件发送给我。

正如你也能看到的，我们正在使用存在于应用程序范围内的`error_handler`对象。在我们下一节中，我们将探讨如何修改该对象。

### 修改 ErrorHandler

当 Sanic 启动时，它首先做的事情之一就是创建一个`ErrorHandler`实例。我们在前面的例子中看到，我们可以从应用程序实例中访问它。它的目的是确保当你定义异常处理程序时，请求会从正确的位置得到响应。

这个对象的另一个好处是它易于定制，并且会在每个异常上触发。因此，在 Sanic 引入信号之前的日子里，这是在每次异常上运行任意代码的最简单方法，就像我们的错误报告工具一样。

修改默认的`ErrorHandler`实例可能看起来像这样：

1.  创建一个`ErrorHandler`并注入报告代码。

    ```py
    from sanic.handlers import ErrorHandler
    class CustomErrorHandler(ErrorHandler):
        def default(self, request, exception):
            ...
    ```

1.  使用你的新处理程序实例化你的应用程序。

    ```py
    from sanic import Sanic
    app = Sanic(..., error_handler=CustomErrorHandler())
    ```

就这样。就我个人而言，在处理警报或其他错误报告时，我几乎*总是*会选择信号解决方案。信号的好处是它是一个更简洁、更有针对性的解决方案。它不需要我子类化或修补任何对象。然而，了解如何创建自定义`ErrorHandler`实例是有帮助的，因为你在野外会看到它。

例如，你会在第三方错误报告服务中看到它们。这些服务是你可以订阅的平台，它们会聚合和跟踪你的应用程序中的异常。它们在识别和调试生产应用程序中的问题方面可以非常有帮助。通常，它们通过挂钩到你的正常异常处理流程来实现。由于在 Sanic 中，覆盖`ErrorHandler`曾经是访问所有异常的底层访问的最佳方法，因此许多这些提供商将提供示例代码或库来使用该策略。

无论你使用自定义的`ErrorHandler`还是信号，这仍然是一个个人喜好的问题。然而，信号的最大好处是它们在单独的`asyncio`任务中运行。这意味着 Sanic 将有效地管理对用户的报告（前提是你没有引入其他阻塞代码）的并发响应。

```py
Does this mean that subclassing ErrorHandler is not a worthwhile effort? Of course not. In fact, if you are unhappy with the default error formats that Sanic uses,  I would recommend that you change it using the previous example with CustomErrorHandler. 
```

考虑到这一点，你现在有权限根据需要格式化所有的错误。与此策略不同的替代策略是使用异常处理程序来管理。这种方法的问题是你可能会失去 Sanic 内置的自动格式化逻辑。提醒一下，默认`ErrorHandler`的一个巨大好处是它会尝试根据情况以适当的格式（如 HTML、JSON 或纯文本）做出响应。

异常处理可能不是构建中最激动人心的事情。然而，它无疑是任何专业级 Web 应用程序的一个极其重要的基本组件。在设计策略时，请确保考虑你的应用程序需求。你可能会发现你需要信号、异常处理程序和自定义`ErrorHandler`的混合。

现在，我们将注意力转向专业级应用程序开发的另一个重要方面，这可能对一些人来说构建起来并不令人兴奋：测试。

## 设置可测试的应用程序

想象一下这个场景：灵感突然降临，你有一个很棒的应用程序想法。当你开始在脑海中构思要构建的内容时，你的兴奋和创造力都在流淌。当然，你不会直接冲进去构建它，因为你已经阅读了这本书的所有前面的章节。你花了一些时间来规划它，然后在咖啡因的驱动下，你开始着手编写代码。慢慢地，你开始看到应用程序的轮廓，它运行得非常完美。几个小时过去了，也许是一天或一周——你不确定，因为你完全沉浸在其中。最后，经过所有这些工作，你得到了一个**最小可行产品(MVP**)。你部署了它，然后去享受一些应得的睡眠。

问题在于你从未设置过测试。毫无疑问，当你现在上线并检查你根据上一节建议设置的错误处理系统时，你会发现它被错误信息淹没了。哎呀。用户在你的应用程序中做了你没有预料到的事情。数据没有按照你想象的方式表现。你的应用程序出问题了。///

我敢打赌，大多数开发过 Web 应用程序或进行过任何软件开发的人都能对这个故事表示同情。我们之前都经历过这种情况。对于许多新手和经验丰富的开发者来说，测试并不有趣。也许你是那些少数喜欢设置测试环境的工程师之一。如果是这样，我真诚地向你致敬。对于其他人来说，简单地说，如果你想构建一个专业应用程序，你需要找到内在的动力来开发测试套件。

测试是一个*巨大的*领域，我这里不会详细讲解。有许多测试策略，包括经常被称赞的**测试驱动设计**(TDD)。如果你知道它是什么并且它对你有效：太好了。如果你不知道：我不会评判你。如果你不熟悉它，我建议你花些时间在网上对这个主题进行一些研究。它是许多专业发展工作流程的基础部分，许多公司已经采用了它。

同样，有许多测试术语，如**单元测试**和**集成测试**。再次强调，这本书不是关于测试理论的，所以我们将使用简化的定义：单元测试是在测试单个组件或端点时进行的，集成测试是在测试与另一个系统（如数据库）交互的组件或端点时进行的。我知道有些人可能不喜欢我的定义，但术语的语义对我们当前的需求并不重要。

在本书中，我们关注的是您如何在单元测试和集成测试中测试您的 Sanic 应用程序。因此，虽然我希望这里的一般思想和方法是有用的，但要真正拥有一个经过良好测试的应用程序，您需要超越本书的页面。

我们需要解决的最后一个基本规则是，这里的测试都将假设您正在使用`pytest`。它是最广泛使用的测试框架之一，拥有许多插件和资源。

### 开始使用 sanic-testing

Sanic 社区组织（维护此项目的开发者社区）还维护了一个用于 Sanic 的测试库。尽管它的主要效用是 Sanic 项目本身用来实现高水平的测试覆盖率，但它仍然为与 Sanic 一起工作的开发者找到了一个归宿和用例。我们将广泛使用它，因为它提供了一个方便的接口来与 Sanic 交互。

首先，我们需要将它安装到您的虚拟环境中。在此过程中，我们还将安装`pytest`：

```py
$ pip install sanic-testing pytest
```

那么，`sanic-testing`到底做了什么？它提供了一个 HTTP 客户端，您可以使用它来访问您的端点。

一个典型的基本实现可能看起来像这样：

1.  首先，您将在某个模块或工厂中定义您的应用程序。目前，它将是一个全局作用域的变量，但稍后在本章中，我们将开始使用工厂模式应用程序，其中应用程序实例是在一个函数内部定义的。

    ```py
    # server.py
    from sanic import Sanic
    app = Sanic(__name__)
    @app.get("/")
    async def handler(request):
        return text("...")
    ```

1.  然后，在您的测试环境中，您初始化一个测试客户端。由于我们正在使用`pytest`，让我们在`conftest.py`文件中将其设置为一个固定装置，这样我们就可以轻松访问它：

    ```py
    # conftest.py
    import pytest
    from sanic_testing.testing import SanicTestClient
    from server import app
    @pytest.fixture
    def test_client():
        return SanicTestClient(app)
    ```

1.  现在，您将能够在单元测试中访问 HTTP 客户端：

    ```py
    # test_sample.py
    def test_sample(test_client):
        request, response = test_client.get("/")
        assert response.status == 200
    ```

1.  现在运行您的测试只是执行 pytest 命令。它应该看起来像这样：

    ```py
    $ pytest
    ================= test session starts =================
    platform linux -- Python 3.9.7, pytest-6.2.5, py-1.11.0, pluggy-1.0.0
    rootdir: /path/to/testing0
    plugins: anyio-3.3.4
    collected 1 item 
    test_sample.py . [100%]
    ================= 1 passed in 0.09s ===================
    ```

那么，这里到底发生了什么？发生的事情是测试客户端获取了您的应用程序实例，并在您的操作系统上实际运行了它。它启动了 Sanic 服务器，将其绑定到操作系统上的主机和端口地址，并运行了附加到您的应用程序上的任何事件监听器。然后，一旦服务器运行起来，它使用`httpx`作为接口向服务器发送实际的 HTTP 请求。然后，它将`Request`和`HTTPResponse`对象捆绑在一起，并将它们作为返回值提供。

这个示例的代码可以在 GitHub 仓库中找到：[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/testing0`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/testing0)。

这是我无法强调得足够的事情。几乎每次有人向我提出关于或使用`sanic-testing`的问题时，都是因为那个人没有理解测试客户端实际上是在运行您的应用程序。这种情况在每次调用中都会发生。

例如，考虑以下内容：

```py
request, response = test_client.get("/foo")
request, response = test_client.post("/bar")
```

当您运行此操作时，它将首先启动应用程序并向 `/foo` 发送 `GET` 请求。然后服务器完成完全关闭。接下来，它再次启动应用程序并向 `/bar` 发送 `POST` 请求。

对于大多数测试用例，这种启动和停止服务器的操作是首选的。这确保了每次您的应用程序都在一个干净的环境中运行。这个过程非常快，您仍然可以快速完成一系列单元测试，而不会感到性能上的惩罚。

在接下来的几节中，我们将探索一些其他选项。

### 更实用的测试客户端实现

现在您已经看到了测试客户端是如何工作的，我要向您透露一个小秘密：您实际上并不需要实例化测试客户端。实际上，除了之前的例子之外，我**从未**在真实的应用程序中以这种方式使用 `sanic-testing`。

Sanic 应用实例有一个内置属性，如果已经安装了 `sanic-testing`，则可以为您设置测试客户端。由于我们已经安装了该包，我们可以直接开始使用它。您所需的一切就是访问您的应用实例。

#### 设置应用程序 fixture

在继续之前，我们将重新审视 `pytest` 的 fixture。如果您不熟悉它们，它们可能对您来说有些神奇。简而言之，它们是 `pytest` 中声明一个将返回值的函数的模式。该值然后可以用来将对象注入到您的单个测试中。

因此，例如，在我们的上一个用例中，我们在一个名为 `conftest.py` 的特殊文件中定义了一个 fixture。在那里定义的任何 fixture 都将在您的测试环境中任何地方可用。这就是我们能够在测试用例中将 `test_client` 作为参数注入的原因。

我发现几乎总是有益于使用应用实例来做这件事。无论您是使用全局定义的实例，还是使用工厂模式，使用 fixture 都会使测试变得更容易。

因此，我总是在我的 `conftest.py` 中做类似的事情：

```py
import pytest
from server import app as application_instance
@pytest.fixture
def app():
    return application_instance
```

我现在可以在测试环境的任何地方访问我的应用实例，而无需导入它：

```py
def test_sample(app):
    ...
```

> **提示**
> 
> 您还需要了解关于 fixture 的一个快速技巧。您可以使用 yield 语法在这里帮助您在测试前后注入代码。这对于应用程序尤其有用，如果您需要在测试运行后进行任何清理操作。为了实现这一点，请执行以下操作：

```py
@pytest.fixture
def app():
    print("Running before the test")
    yield application_instance
    print("Running after the test")
```

通过使用 fixture 访问我们的应用实例，我们现在可以像这样重写之前的单元测试：

```py
def test_sample(app: Sanic):
    request, response = app.test_client.get("/")
    assert response.status == 200
```

为了让我们的生活变得简单一些，我为 fixture 添加了类型注解，这样我的 **集成开发环境**（**IDE**）就知道它是一个 Sanic 实例。尽管类型提示的主要目的是尽早捕捉错误，但我还喜欢在类似的情况下使用它来使我的 IDE 体验更佳。

这个例子表明，访问测试客户端只是使用 `app.test_client` 属性的问题。通过这样做，只要安装了包，Sanic 就会自动为您实例化客户端。这使得编写这样的单元测试变得非常简单。

#### 测试蓝图

有时候，你可能会遇到一个场景，你想测试蓝图上存在的某些功能。在这种情况下，我们假设在蓝图之前运行的任何应用程序范围的中间件或监听器都与我们的测试无关。这意味着我们正在测试一些完全包含在蓝图边界内的功能。

我喜欢这种情况，并且会积极寻找它们。原因在于，正如我们将在下一分钟看到的那样，这些测试非常容易进行。这些类型的测试模式最好理解为与我们在 *测试完整应用程序部分* 中将要做的对比。主要区别在于，在这些测试中，我们的端点不依赖于第三方系统（如数据库）的存在。也许更准确地说，我应该说我应该说的是，它们不依赖于第三方系统可能产生的影响。功能性和业务逻辑是自包含的，因此非常适合单元测试。

当我发现这种情况时，我首先会在我 的 `conftest.py` 文件中添加一个新的 fixture。它将作为一个我可以用于测试的虚拟应用程序。我创建的每个单元测试都可以使用这个虚拟应用程序，并附加我的目标蓝图，而无需其他任何内容。这使我的单元测试能够更专注于单个示例。让我们看看接下来会是什么样子。

1.  在这里，我们将创建一个新的 fixture，它将创建一个新的应用程序实例。

    ```py
    # conftest.py
    import pytest
    from sanic import Sanic
    @pytest.fixture
    def dummy_app():
        return Sanic("DummyApp")
    ```

1.  我们现在可以在蓝图测试中创建一个测试桩：

    ```py
    # test_some_blueprint.py
    import pytest
    from path.to.some_blueprint import bp
    @pytest.fixture
    def app_with_bp(dummy_app):
        dummy_app.blueprint(bp)
        return dummy_app
    def test_some_blueprint_foobar(app_with_bp):
        ...
    ```

在这个例子中，我们看到我创建了一个仅限于这个模块的 fixture。这样做是为了创建一个可重用的应用程序实例，该实例附加了我的目标蓝图。

这种类型测试的一个简单用例可能是输入验证。让我们添加一个执行一些输入验证的蓝图。该蓝图将有一个简单的 `POST` 处理程序，它检查传入的 JSON 主体，并仅检查键是否存在，以及类型是否与预期匹配。

1.  首先，我们将创建一个模式，它将是我们的端点预期能够测试的键和值类型。

    ```py
    from typing import NamedTuple
    class ExpectedTypes(NamedTuple):
        a_string: str
        an_int: int
    ```

1.  第二，我们将创建一个简单的类型检查器，根据值是否存在以及是否为预期的类型，响应三个值之一：

    ```py
    def _check(
        exists: bool,
        value: Any, 
        expected: Type[object],
    ) -> str:
        if not exists:
            return "missing"
        return "OK" if type(value) is expected else "WRONG"
    ```

1.  最后，我们将创建我们的端点，它将接收请求 JSON 并响应一个字典，说明传递的数据是否有效。

    ```py
    from sanic import Blueprint, Request, json
    bp = Blueprint("Something", url_prefix="/some")
    @bp.post("/validation")
    async def check_types(request: Request):
        valid = {
            field_name: _check(
                field_name in request.json,
                request.json.get(field_name), field_type
            )
            for field_name, field_type in
            ExpectedTypes.__annotations__.items()
        }
        expected_length = len(ExpectedTypes.__annotations__)
        status = (
            200
            if all(value == "OK" for value in valid.values())
            and len(request.json) == expected_length
            else 400
        )
        return json(valid, status=status)
    ```

    如您所见，我们现在创建了一个非常简单的数据检查器。我们遍历模式中的定义，并检查每个定义是否符合预期。所有值都应该是 `"OK"`，并且请求数据的长度应该与模式相同。

我们现在可以在我们的测试套件中测试这一点。我们首先可以测试的是确保所有必需的字段都存在。这里有三种可能的场景：输入缺少字段，输入只有正确的字段，以及输入有额外的字段。让我们看看这些场景并为它们创建一些测试：

1.  首先，我们将创建一个测试来检查没有缺少字段。

    ```py
    def test_some_blueprint_no_missing(app_with_bp):
        _, response = app_with_bp.test_client.post(
            "/some/validation",
            json={
                "a_string": "hello",
                "an_int": "999",
            },
        )
        assert not any(
            value == "MISSING"
            for value in response.json.values()
        )
        assert len(response.json) == 2
    ```

    在这个测试中，我们发送了一些不良数据。注意`an_int`值实际上是一个`str`。但我们现在并不关心这一点。这意味着要测试的是所有适当的字段都已发送。

1.  接下来是一个应该包含所有输入、正确类型但没有更多内容的测试。

    ```py
    def test_some_blueprint_correct_data(app_with_bp):
        _, response = app_with_bp.test_client.post(
            "/some/validation",
            json={
                "a_string": "hello",
                "an_int": 999,
            },
        )
        assert response.status == 200
    ```

    在这里，我们只需要断言响应是 200，因为我们知道如果数据有问题，它将是 400。

1.  最后，我们创建一个测试来检查我们没有发送多余的信息。

    ```py
    def test_some_blueprint_bad_data(app_with_bp):
        _, response = app_with_bp.test_client.post(
            "/some/validation",
            json={
                "a_string": "hello",
                "an_int": 999,
                "a_bool": True,
            },
        )
        assert response.status == 400
    ```

    在这个最终测试中，我们发送了已知的不良数据，因为它包含与上一个测试完全相同的有效载荷，除了额外的`"a_bool": True`。因此，我们应该断言响应将是 400。

看这些测试，似乎非常重复。虽然“不要重复自己”的原则（简称 DRY）通常被引用为抽象逻辑的理由，但在测试中要小心这一点。我更愿意看到重复的测试代码，而不是一些高度抽象、美丽、闪亮的工厂模式。根据我的经验——是的，我过去曾多次因此受到伤害——在测试代码中添加花哨的抽象层是灾难的配方。一些抽象可能是有帮助的（创建`dummy_app`固定值是一个好的抽象例子），但过多可能会造成灾难。在未来需要更改某些功能时，解开这些抽象将变得是一场噩梦。这确实是在开发中科学与艺术之间界限模糊的一个领域。创建一个功能强大的测试套件，在重复和抽象之间取得适当的平衡，需要一些实践，并且非常主观。

在消除这些警告之后，有一个抽象层我确实很喜欢。它利用了`pytest.parametrize`。这是一个超级有用的功能，允许你创建一个测试并针对多个输入运行它。我们并不是在抽象测试本身，而是在使用各种输入测试相同的代码。

使用`pytest.parametrize`，我们实际上可以将这三个测试压缩成一个测试：

1.  我们创建了一个有两个参数的装饰器：一个包含逗号分隔的参数名称列表的字符串，以及一个包含要注入测试中的值的可迭代对象。

    ```py
    @pytest.mark.parametrize(
    "input,has_missing,expected_status",
    (
        (
            {
                "a_string": "hello",
            }, True, 400,
        ),
        (
            {
                "a_string": "hello",
                "an_int": "999",
            }, False, 400,
        ),
        (
            {
                "a_string": "hello",
                "an_int": 999,
            }, False, 200,
        ),
        (
            {
                "a_string": "hello","an_int": 999,
                "a_bool": True,
            }, False, 400,
        ),
    ),
    )
    ```

    我们有三个值要注入到我们的测试中：`input`、`has_missing`和`expected_status`。测试将运行多次，每次它都会从参数的元组中抽取一个来注入到测试函数中。

1.  我们的测试函数现在可以抽象化以使用这些参数：

    ```py
    def test_some_blueprint_data_validation(
        app_with_bp,
        input,
        has_missing,
        expected_status,
    ):
        _, response = app_with_bp.test_client.post(
            "/some/validation",
            json=input,
        )
        assert any(
            value == "MISSING"
            for value in response.json.values()
        ) is has_missing
        assert response.status == expected_status
    ```

    这样做，我们更容易为不同的用例编写多个单元测试。您可能已经注意到，我实际上创建了一个第四个测试。由于使用这种方法添加更多测试非常简单，我包含了一个我们之前没有测试过的用例。我希望您能看到这种做法带来的巨大好处，并学会喜欢使用`@pytest.mark.parametrize`进行测试。

在这个例子中，我们定义了输入和预期的结果。通过参数化单个测试，它实际上在`pytest`内部转换成了多个测试。

这些示例的代码可以在 GitHub 仓库中找到：[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/testing2`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/testing2)。

#### 模拟服务

我们测试的样本蓝图显然不是我们会在现实生活中使用的。在那个例子中，我们实际上并没有对数据进行任何操作。我之所以简化它，是为了我们不必担心如何处理与数据库访问层等服务的交互。当我们测试一个真实端点时怎么办？而且，我所说的真实端点是指那些旨在与数据库接口的端点。例如，注册端点怎么样？我们如何测试注册端点实际上执行了它应该执行的操作，并按预期注入了数据？

即使我知道我的端点需要执行一些数据库操作，我仍然喜欢使用`dummy_app`模式进行测试。我们将探讨如何使用 Python 的模拟工具来模拟我们有一个真实的数据库层：

1.  首先，我们需要重构我们的蓝图，使其看起来像您可能在野外遇到的东西：

    ```py
    @bp.post("/")
    async def check_types(request: Request):
        _validate(request.json, RegistrationSchema)
        connection: FakeDBConnection = request.app.ctx.db
        service = RegistrationService(connection)
        await service.register_user(request.json["username"], request.json["email"])
        return json(True, status=201)
    ```

    我们仍在进行输入验证。然而，我们不会仅仅将注册详情存储在内存中，而是将它们发送到数据库以写入磁盘。您可以在[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/testing3`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/testing3)查看完整的代码，以了解输入验证。这里需要注意的重要事项是我们有一个`RegistrationService`，并且它调用了一个`register_user`方法。

1.  由于我们还没有探讨**对象关系映射**（**ORM**）的使用，我们的数据库存储函数最终只是调用一些原始的 SQL 查询。我们将在“管理数据库连接”中更详细地探讨 ORM，但现在，让我们创建注册服务：

    ```py
    from .some_db_connection import FakeDBConnection
    class RegistrationService:
        def __init__(self, connection: FakeDBConnection) -> None:
            self.connection = connection
        async def register_user(
            self, username: str, email: str
        ) -> None:
            query = "INSERT INTO users VALUES ($1, $2);"
            await self.connection.execute(query, username, email)
    ```

1.  注册服务调用我们的数据库来执行一些 SQL 语句。我们还需要一个到数据库的连接。为了举例说明，我使用了一个假类，但这个类（并且应该是）您的应用程序用来连接数据库的实际对象。因此，想象这是一个合适的 DB 客户端：

    ```py
    from typing import Any
    class FakeDBConnection:
        async def execute(self, query: str, *params: Any):
            ...
    ```

1.  在此基础上，我们现在可以创建一个新的测试用例，它将取代我们的数据访问层。通常，你会创建类似的东西来实例化客户端：

    ```py
    from sanic import Sanic
    from .some_db_connection import FakeDBConnection
    app = Sanic.get_app()
    @app.before_server_start
    async def setup_db_connection(app, _):
        app.ctx.db = FakeDBConnection()
    ```

    发挥你的想象力，想象一下上述代码片段存在于我们的*实际*应用程序中。它初始化数据库连接，并允许我们在端点内访问客户端，如前述代码所示，因为我们的连接使用了应用程序的`ctx`对象。由于我们的单元测试无法访问数据库，我们需要创建一个*模拟*数据库并将其附加到我们的模拟应用程序上。

1.  为了做到这一点，我们将创建我们的`dummy_app`，然后导入实际应用程序使用的实际监听器来实例化模拟客户端。

    ```py
    @pytest.fixture
    def dummy_app():
        app = Sanic("DummyApp")
        import_module("testing3.path.to.some_startup")
        return app
    ```

1.  为了强制我们的客户端使用模拟方法而不是实际向数据库发送网络请求，我们将使用 pytest 的一个功能来 monkeypatch DB 客户端。设置一个类似的测试用例：

    ```py
    from unittest.mock import AsyncMock
    @pytest.fixture
    def mocked_execute(monkeypatch):
        execute = AsyncMock()
        monkeypatch.setattr(
            testing3.path.to.some_db_connection.FakeDBConnection, "execute", execute
        )
        return execute
    ```

    现在我们已经用模拟的`execute`方法替换了真实的`execute`方法，我们可以继续构建我们的注册蓝图测试。使用`unittest.mock`库的一个巨大好处是，它允许我们创建断言，表明数据库客户端已被调用。我们将在下一部分看到这会是什么样子。

1.  在这里，我们创建了一个包含一些断言的测试，这些断言帮助我们了解正确数据将如何到达数据访问层：

    ```py
    @pytest.mark.parametrize(
        "input,expected_status",
        (
            (
                {
                    "username": "Alice",
                    "email": "alice@bob.com",
                },
                201,
            ),
        ),
    )
    def test_some_blueprint_data_validation(
        app_with_bp,
        mocked_execute,
        input,
        expected_status,
    ):
        _, response = app_with_bp.test_client.post(
            "/registration",
            json=input,
        )
        assert response.status == expected_status
        if expected_status == 201:
            mocked_execute.assert_awaited_with(
                "INSERT INTO users VALUES ($1, $2);", input["username"], input["email"]
            )
    ```

    就像之前一样，我们使用`parametrize`，这样我们就可以使用不同的输入运行多个测试。关键要点是，由于我们正在使用模拟的`execute`方法，我们可以要求 pytest 为我们提供它，以便我们的测试可以断言它被调用，就像我们预期的那样。

这对于测试隔离问题当然很有帮助，但当我们需要进行应用程序范围内的测试时怎么办？我们将在下一部分探讨这个问题。

#### 测试完整的应用程序

随着应用程序从其婴儿期发展，可能会开始形成一个由中间件、监听器和信号组成的网络，这些网络处理请求，不仅限于路由处理器。此外，还可能与其他服务（如数据库）建立连接，这会复杂化整个过程。典型的 Web 应用程序不能在真空中运行。当它启动时，它需要连接到其他服务。这些连接对于应用程序的正确性能至关重要，因此如果它们不存在，则应用程序无法启动。测试这些连接可能会非常麻烦。不要只是举手投降并放弃。抵制诱惑。在前面的测试中，我们已经看到了如何简单实现这一点。事实上，我们已经成功测试了我们的数据库。但那还不够？

有时候仅对`dummy_app`进行测试是不够的。

这也是为什么我非常喜欢由工厂模式创建的应用程序。本章的 GitHub 仓库就是一个我经常使用的工厂模式的例子。它包含了一些非常有用的功能。本质上，最终结果是返回一个带有所有附加内容的 Sanic 实例的函数。通过 Sanic 标准库的实现，该函数会遍历你的源代码，寻找可以附加到其上的内容（路由、蓝图、中间件、信号、监听器等等），并且被设置为避免循环导入问题。我们之前在*第二章*，*组织项目*中讨论了工厂模式和它们的优点。

目前特别重要的是，GitHub 仓库中的工厂可以选择性选择要实例化的内容。这意味着我们可以使用具有针对性功能的有效应用程序。让我提供一个例子。

以前，我正在构建一个应用程序。了解它在现实世界中的确切性能至关重要。因此，我创建了一个中间件，它会计算一些性能指标，然后将它们发送给供应商进行分析。性能至关重要——这也是我最初选择使用 Sanic 的原因之一。当我尝试进行一些测试时，我意识到如果它没有连接到供应商，我就无法在我的测试套件中运行应用程序。是的，我可以模拟它。然而，更好的策略是根本跳过这个操作。有时候，真的没有必要测试每一个功能点。

为了使这一点更具体，这里是对我所谈论的内容的一个快速解释。这是一个中间件代码片段，它在请求的开始和结束时计算运行时间，并将它发送出去：

```py
from time import time
from sanic import Sanic
app = Sanic.get_app()
@app.on_request
async def start_timer(request: Request) -> None:
    request.ctx.start_time = time()
@app.on_response
async def stop_timer(request: Request, _) -> None:
    end_time = time()
    total = end_time - request.ctx.start_time
    async send_the_value_somewhere(total)
```

解决我的测试与生产行为对比问题的解决方案之一可能是将应用程序代码更改为仅在生产中运行：

```py
if app.config.ENVIRONMENT == "PRODUCTION":
    ...
```

但在我看来，更好的解决方案是根本跳过这个中间件。使用仓库中显示的工厂模式，我可以这样做：

```py
from importlib import import_module
from typing import Optional, Sequence
from sanic import Sanic
DEFAULT = ("path.to.some_middleware.py",)
def create_app(modules: Optional[Sequence[str]] = None) -> Sanic:
    app = Sanic("MyApp")
    if modules is None:
        modules = DEFAULT
    for module in modules:
        import_module(module)
    return app
```

在这个工厂中，我们正在创建一个新的应用程序实例，并遍历一个已知模块列表来导入它们。在正常使用中，我们会通过调用`create_app()`来创建一个应用程序，工厂会导入`DEFAULT`已知模块。通过导入它们，它们将附加到我们的应用程序实例上。更重要的是，这个工厂允许我们发送一个任意模块列表来加载。这使我们能够在测试中创建一个使用我们应用程序的实际工厂模式的固定装置，同时拥有选择加载内容的控制权。

在我们的用例中，我们决定我们不想测试性能中间件。我们可以通过创建一个简单地忽略该模块的测试固定装置来跳过它：

```py
from path.to.factory import create_app
@pytest.fixture
def dummy_app():
    return create_app(modules=[])
```

如你所见，这为我创建针对我实际应用程序特定部分的测试打开了大门，而不仅仅是模拟应用程序。通过使用包含和排除，我可以创建只包含我需要的功能的单元测试，并避免不必要的功能。

我希望你的脑海中现在充满了这个为你打开的可能性。当应用程序本身是可组合的时候，测试变得容易得多。这个神奇的技巧是真正将你的应用程序开发提升到下一个水平的一种方式。一个容易组合的应用程序成为一个容易测试的应用程序。这导致应用程序得到良好的测试，现在你真正走上了成为高级开发者的道路。

如果你还没有开始，我强烈建议你使用像我这样的工厂。大胆地复制它。只需向我承诺你会用它来创建一些单元测试。

### 使用可重用客户端进行测试

到目前为止，我们一直在使用一个测试客户端，每次调用它都会启动和停止一个服务。`sanic-testing`包附带另一个可以手动启动和停止的测试客户端。因此，可以在调用之间或测试之间重用它。在下一小节中，我们将了解这个可重用测试客户端。

#### 每个测试运行一个单独的测试服务器

你有时可能需要在同一实例上运行多个 API 调用。例如，如果你在内存中在调用之间存储一些临时状态，这可能是有用的。显然，在大多数情况下，这不是一个好的解决方案，因为将状态存储在内存中会使水平扩展变得困难。不谈这个问题，让我们快速看看你可能如何实现它：

1.  我们首先创建一个只输出计数器的端点：

    ```py
    from sanic import Sanic, Request, json
    from itertools import count
    app = Sanic("test")
    @app.before_server_start
    def setup(app, _):
        app.ctx.counter = count()
    @app.get("")
    async def handler(request: Request):
        return json(next(request.app.ctx.counter))
    ```

    在这个简化的例子中，每次你点击端点时，它都会增加一个数字。

1.  我们可以通过以下方式使用`ReusableClient`实例来测试维护内部状态的端点：

    ```py
    from sanic_testing.reusable import ReusableClient
    def test_reusable_context():
        client = ReusableClient(app, host="localhost", port=9999)
        with client:
            _, response = client.get("/")
            assert response.json == 0
            _, response = client.get("/")
            assert response.json == 1
            _, response = client.get("/")
            assert response.json == 2
    ```

    只要你在那个`with`上下文管理器内部使用客户端，那么你每次调用都会遇到你应用程序的相同实例。

1.  我们可以通过使用固定值来简化前面的代码：

    ```py
    from sanic_testing.reusable import ReusableClient
    import pytest
    @pytest.fixture
    def test_client():
        client = ReusableClient(app, host="localhost", port=9999)
        client.run()
        yield client
        client.stop()
    ```

    现在，当你设置单元测试时，它将保持服务器在测试函数执行期间运行。

1.  如前所述的单元测试可以写成以下形式：

    ```py
    def test_reusable_fixture(test_client):
        _, response = test_client.get("/")
        assert response.json == 0
        _, response = test_client.get("/")
        assert response.json == 1
        _, response = test_client.get("/")
        assert response.json == 2
    ```

    如你所见，如果你只想在测试函数的整个运行期间运行单个服务器，这是一个潜在的强大策略。那么，如果你想在整个测试期间保持实例运行呢？最简单的方法就是将固定值的`scope`改为`session`：

    ```py
    @pytest.fixture(scope="session")
    def test_client():
        client = ReusableClient(app, host="localhost", port=9999)
        client.run()
        yield client
        client.stop()
    ```

使用这个设置，无论你在`pytest`中运行测试在哪里，它都会使用相同的应用程序。虽然我个人从未觉得有必要使用这种模式，但我确实看到了它的实用性。

这个示例的代码可以在 GitHub 仓库中找到：[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/testing4`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/testing4)。

在适当的异常管理和测试完成之后，任何真正的专业应用程序的下一个关键添加是日志。

## 从日志和跟踪中获得洞察

当谈到日志时，我认为大多数 Python 开发者可以分为三大类：

+   总是使用`print`语句的人。

+   对日志设置有极端意见和极其复杂的人。

+   知道不应该使用`print`，但没有时间或精力去理解 Python 的`logging`模块的人。

如果你属于第二类，你大可以跳过这一节。里面对你来说没有什么内容，除非你只是想批评我的解决方案，并告诉我有更好的方法。

如果你属于第一类，那么你真的需要学会改变你的习惯。不要误会，`print`非常棒。然而，它不适合专业级 Web 应用，因为它不提供日志模块提供的灵活性。

“*等一下!*”我听到第一类人已经开始喊叫了。“*如果我使用容器和 Kubernetes 部署我的应用程序，它可以从那里获取我的输出并将其重定向。*”/// 如果你坚决反对使用日志，那么我想我可能无法改变你的想法。然而，抛开配置复杂性不谈，考虑一下日志模块提供了丰富的 API 来发送不同级别和元上下文的消息。

看一下标准的 Sanic 访问日志。访问记录器发送的消息实际上是空的。如果你想查看 Sanic 代码库，请自行查看。访问日志是这样的：

```py
access_logger.info("")
```

你实际上看到的是更类似于以下的内容：

```py
[2021-10-21 09:39:14 +0300] - (sanic.access)[INFO][127.0.0.1:58388]: GET http://localhost:9999/  200 13
```

在那一行中嵌入了一堆既适合机器读取又适合人类阅读的元数据，这要归功于`logging`模块。实际上，你可以使用日志存储任意数据，一些日志配置会为你存储这些数据。例如：

```py
log.info("Some message", extra={"arbitrary": "data"})
```

如果我已经说服了你，并且你想了解更多关于如何在 Sanic 中使用日志的信息，那么让我们继续。

### Sanic 日志记录器的类型

Sanic 自带了三个日志记录器。你可以在`log`模块中访问它们：

```py
from sanic.log import access_logger, error_logger, logger
```

随意使用这些在你的应用程序中。特别是在较小的项目中，我经常为了方便而使用 Sanic 的`logger`对象。当然，这些实际上是供 Sanic 本身使用的，但没有任何阻止你使用它们的。事实上，这可能很方便，因为你知道所有的日志都是格式一致的。我唯一的警告是，最好让`access_logger`对象保持原样，因为它有一个非常具体的工作。

为什么你想同时使用`error_logger`和常规 logger 呢？我认为答案取决于你希望你的日志发生什么。有许多选项可供选择。最简单的形式当然是直接输出到控制台。然而，对于错误日志来说，这并不是一个好主意，因为你没有方法持久化消息，在发生错误时无法查看它们。因此，你可能采取下一步，将你的`error_logger`输出到文件。当然，这可能会变得繁琐，所以你决定使用第三方系统将日志发送到另一个应用程序以存储并使其可访问。无论你希望设置什么，使用多个 logger 可能在处理和分发日志消息中扮演特定的角色。

### 创建自己的 logger，我应用程序开发的第一个步骤

当我面对一个新的项目时，我会问自己一个问题：我的生产日志会发生什么？当然，这是一个高度依赖于你应用程序的问题，你需要自己决定。尽管如此，提出这个问题却突出了一个非常重要的观点：开发日志和生产日志之间有一个区别。很多时候，我甚至不知道在生产环境中我想对它们做什么。我们可以将这个问题推迟到另一天。

在我开始编写应用程序之前，我会创建一个日志框架。我知道目标是拥有两套配置，所以我从我的开发日志开始。

我想再次强调：构建应用程序的第一步是创建一个超级简单的框架，用于设置带有日志的应用程序。那么，我们现在就来看看这个设置过程：

1.  第一件事，我们将按照在*第二章*，“组织项目”中确立的模式，创建一个超级基本的脚手架：

    ```py
    .
    ├── Dockerfile
    ├── myapp
    │   ├── common
    │   │   ├── __init__.py
    │   │   └── log.py
    │   ├── __init__.py
    │   └── server.py
    └── tests
    ```

    这是我喜欢与之工作的应用程序结构，因为它使我开发起来非常容易。使用这种结构，我们可以轻松地创建一个专注于本地运行应用程序、测试应用程序、记录日志和构建镜像的开发环境。在这里，我们显然关注的是本地运行应用程序并记录日志。

1.  接下来，我喜欢创建我的应用程序工厂，并在上面放置一个我将稍后删除的虚拟路由。以下是`server.py`的创建方式。我们将继续添加内容：

    ```py
    from sanic import Sanic, text
    from myapp.common.log import setup_logging, app_logger
    def create_app():
        app = Sanic(__name__)
        setup_logging()
        @app.route("")
        async def dummy(_):
            app_logger.debug("This is a DEBUG message")
            app_logger.info("This is a INFO message")
            app_logger.warning("This is a WARNING message")
            app_logger.error("This is a ERROR message")
            app_logger.critical("This is a CRITICAL message")
            return text("")
        return app
    ```

    在创建我的应用程序实例之后，我之所以称之为`setup_logging`，是因为我想能够使用 Sanic 的配置逻辑来加载可能用于创建我的日志设置的环境变量。在继续之前，我想先提一下。在创建 Python `logger`对象时，有两种不同的观点。一方面认为，在每一个模块中创建一个新的`logger`是最佳实践。在这种情况下，你会在*每个 Python 文件*的顶部放置以下代码：

    ```py
    from logging import getLogger
    logger = getLogger(__name__)
    ```

    这种方法的优点是创建它的模块名称与日志记录器名称紧密相关。这当然有助于追踪日志的来源。然而，另一方却认为应该是一个单一的全球变量，被导入并重复使用，因为这可能更容易配置和控制。此外，我们可以通过适当的日志格式快速获取特定的文件名和行号，因此没有必要在日志记录器名称中包含模块名称。虽然我不否认本地化、按模块的方法，但我也更喜欢导入单个实例的简单性：

    ```py
    from logging import getLogger
    logger = getLogger("myapplogger")
    ```

    如果你深入研究日志记录，这也为你提供了更大的能力来控制不同的日志记录实例如何操作。类似于关于异常处理器的讨论，我更愿意限制我需要控制的实例数量。在我刚刚展示的`server.py`示例中，我选择了第二个选项，使用单个全局`logging`实例。这是一个个人选择，在我看来没有错误答案。两种策略都有其利弊，所以选择对你来说有意义的那个。

1.  下一步是创建基本的`log.py`。目前，让我们保持它非常简单，然后我们将从那里开始构建：

    ```py
    import logging
    app_logger = logging.getLogger("myapplogger")
    def setup_logging():
        ...
    ```

1.  在此基础上，我们就可以运行应用程序并对其进行测试。但是等等？！我们传递给`sanic`命令的应用程序在哪里？我们之前使用这个来运行我们的应用程序：

    ```py
    $ sanic src.server:app -p 7777 --debug --workers=2
    ```

    相反，我们将告诉 Sanic CLI `create_app`函数的位置，并让它为我们运行。将你的启动方式改为如下：

    ```py
    $ sanic myapp.server:create_app --factory -p 7777 --debug --workers=2
    ```

你现在应该能够到达你的端点并看到一些基本的消息输出到你的终端。你很可能没有`DEBUG`消息，因为日志记录器可能仍然设置为只记录`INFO`级别及以上。你应该看到一些非常基础的类似如下内容：

```py
This is a WARNING message
This is a ERROR message
This is a CRITICAL message
```

### 配置日志记录

前面的日志消息正是使用`print`可以提供的内容。接下来我们需要添加一些配置，以便输出一些元数据并格式化消息。重要的是要记住，某些日志细节可能需要根据生产环境进行定制：

1.  因此，我们将首先创建一个简单的配置。

    ```py
    DEFAULT_LOGGING_FORMAT = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s"
    def setup_logging(app: Sanic):
        formatter = logging.Formatter(
            fmt=app.config.get("LOGGING_FORMAT", DEFAULT_LOGGING_FORMAT),
            datefmt="%Y-%m-%d %H:%M:%S %z",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        app_logger.addHandler(handler)
    ```

    请注意，我们已经将`setup_logging`函数的签名函数更改为现在接受应用程序实例作为参数。请确保回到更新你的`server.py`文件以反映这一变化。作为旁注，有时你可能想简化你的日志记录，以强制 Sanic 使用相同的处理器。虽然你当然可以通过更新 Sanic 日志记录器配置的过程（见[`sanicframework.org/en/guide/best-practices/logging.html#changing-sanic-loggers`](https://sanicframework.org/en/guide/best-practices/logging.html#changing-sanic-loggers)）来完成，但我发现这太过繁琐。一个更简单的方法是设置日志处理器，然后将它们简单地应用到 Sanic 日志记录器上，如下所示：

    ```py
    from sanic.log import logger, error_logger
    def setup_logging(app: Sanic):
        ...
        logger.handlers = app_logger.handlers
        error_logger.handlers = app_logger.handlers
    ```

    总是保留一个`StreamHandler`是一个好的实践。这将用于将日志输出到控制台。但是，当我们想要为生产添加一些额外的日志工具时怎么办？由于我们还不完全确定我们的生产需求是什么，我们将暂时将日志设置到文件中。这总可以在另一个时间替换。

1.  将你的`log.py`修改如下：

    ```py
    def setup_logging(app: Sanic):
        formatter = logging.Formatter(
            fmt=app.config.get("LOGGING_FORMAT", DEFAULT_LOGGING_FORMAT),
            datefmt="%Y-%m-%d %H:%M:%S %z",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        app_logger.addHandler(handler)
        if app.config.get("ENVIRONMENT", "local") == "production":
            file_handler = logging.FileHandler("output.log")
            file_handler.setFormatter(formatter)
            app_logger.addHandler(file_handler)
    ```

你可以很容易地看到如何配置不同的日志处理程序或格式，这可能与不同环境中的需求更接近。

所示的所有配置都使用了日志实例的程序控制。`logging`库的一个巨大灵活性是所有这些都可以通过一个单一的`dict`配置对象来控制。因此，你会发现保留包含日志配置的 YAML 文件是一个非常常见的做法。这些文件易于更新，可以在构建环境中互换以控制生产设置。

### 添加颜色上下文

上述设置完全有效，你可以在这里停止。然而，对我来说，这还不够。当我开发一个网络应用程序时，我总是打开终端输出日志。在消息的海洋中，可能很难筛选出所有文本。我们如何使它更好？我们将通过适当使用颜色来实现这一点。

由于我通常不需要在我的生产输出中添加颜色，所以我们只会在本地环境中添加颜色格式化：

1.  我们将首先设置一个自定义的日志格式化器，该格式化器将根据日志级别添加颜色。任何调试信息都是蓝色，警告是黄色，错误是红色，而一个关键信息将以红色和白色背景显示，以便突出显示（在深色终端中）：

    ```py
    class ColorFormatter(logging.Formatter):
        COLORS = {
            "DEBUG": "\033[34m",
            "WARNING": "\033[01;33m",
            "ERROR": "\033[01;31m",
            "CRITICAL": "\033[02;47m\033[01;31m",
        }
        def format(self, record) -> str:
            prefix = self.COLORS.get(record.levelname)
            message = super().format(record)
            if prefix:
                message = f"{prefix}{message}\033[0m"
            return message
    ```

    我们使用大多数终端都理解的标准的颜色转义码来应用颜色。这将使整个消息着色。当然，你可以通过只着色消息的一部分来变得更为复杂，如果你对此感兴趣，我建议你尝试使用这个格式化器看看你能实现什么。

1.  在创建这个之后，我们将快速创建一个内部函数来决定使用哪个格式化器：

    ```py
    import sys
    def _get_formatter(is_local, fmt, datefmt):
        formatter_type = logging.Formatter
        if is_local and sys.stdout.isatty():
            formatter_type = ColorFormatter
        return formatter_type(
            fmt=fmt,
            datefmt=datefmt,
        )
    ```

    如果我们处于一个 TTY 终端的本地环境，那么我们使用我们的颜色格式化器。

1.  我们需要更改`setup_logging`函数的开始部分，以考虑到这些变化。我们还将抽象出更多细节到我们的配置中，以便可以轻松地按环境更改它们：

    ```py
    DEFAULT_LOGGING_FORMAT = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s"
    DEFAULT_LOGGING_DATEFORMAT = "%Y-%m-%d %H:%M:%S %z"
    def setup_logging(app: Sanic):
        environment = app.config.get("ENVIRONMENT", "local")
        logging_level = app.config.get(
            "LOGGING_LEVEL", logging.DEBUG if environment == "local" else logging.INFO
        )
        fmt = app.config.get("LOGGING_FORMAT", DEFAULT_LOGGING_FORMAT)
        datefmt = app.config.get("LOGGING_DATEFORMAT", DEFAULT_LOGGING_DATEFORMAT)
        formatter = _get_formatter(environment == "local", fmt, datefmt)
        ...
    ```

除了动态获取格式化器之外，这个示例还为这个谜题增加了一个新的部分。它是使用配置值来确定你的日志记录器的日志级别。

### 添加一些基本的跟踪请求 ID

日志中常见的问题之一是它们可能会变得嘈杂。可能很难将特定的日志与特定的请求相关联。例如，你可能同时处理多个请求。如果出现错误，并且你想回顾早先的消息，你该如何知道哪些日志应该被分组在一起？

完全有第三方应用程序添加了所谓的**跟踪**功能。如果你正在构建一个由相互关联的微服务组成的系统，这些微服务协同工作以响应传入的请求，这将特别有帮助。虽然不一定深入探讨微服务架构，但在这里提一下，跟踪是一个重要的概念，应该添加到你的应用程序中。这适用于无论你的应用程序架构是否使用微服务。

对于我们的目的，我们想要实现的是为每个请求添加一个请求标识符。每当该请求尝试记录某些内容时，该标识符将自动注入到我们的请求格式中。为了实现这个目标：

+   首先，我们需要一种机制将请求对象注入到每个日志操作中。

+   其次，我们需要一种方法来显示标识符如果它存在，或者如果不存在则忽略它。

在我们进入代码实现之前，我想指出，第二部分可以通过几种方式处理。最简单的方法可能是创建一个特定的记录器，它只会在请求上下文中使用。这意味着你将有一个用于启动和关闭操作的记录器，另一个仅用于请求。我见过这种方法被很好地使用。

问题是我们再次使用了多个记录器。坦白说，我真的很喜欢只有一个实例，它可以适用于我所有的用例的简单性。这样我就不需要费心考虑我应该使用哪个记录器。因此，我将在这里向你展示如何构建第二种选择：一个可以在应用程序的任何地方使用的全能型记录器。如果你更喜欢更具体类型，那么我挑战你在这里使用我的概念，构建两个记录器而不是一个。

我们将从处理传递请求上下文的问题开始。记住，由于 Sanic 是异步操作的，我们无法保证哪个请求将以什么顺序被处理。幸运的是，Python 标准库有一个与 `asyncio` 工作得很好的实用工具，那就是 `contextvars` 模块。我们将开始创建一个监听器，设置一个我们可以用来共享请求对象并将其传递给日志框架的上下文：

1.  创建一个名为 `./middleware/request_context.py` 的文件。它应该看起来像这样：

    ```py
    from contextvars import ContextVar
    from sanic import Request, Sanic
    app = Sanic.get_app()
    @app.after_server_start
    async def setup_request_context(app, _):
        app.ctx.request = ContextVar("request")
    @app.on_request
    async def attach_request(request: Request):
        request.app.ctx.request.set(request)
    ```

    这里发生的事情是我们正在创建一个可以在任何有权访问我们的应用程序的地方访问的上下文对象。然后，在每次请求中，我们将当前请求附加到上下文变量中，使其在任何应用程序实例可访问的地方都可以访问。

1.  接下来需要做的事情是创建一个日志过滤器，它会获取请求（如果存在）并将其添加到我们的日志记录中。为了做到这一点，我们实际上会在`log.py`文件中覆盖 Python 创建日志记录的函数：

    ```py
    old_factory = logging.getLogRecordFactory()
    def _record_factory(*args, app, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_info = ""
        if hasattr(app.ctx, "request"):
            request = app.ctx.request.get(None)
            if request:
                display = " ".join([str(request.id), request.method, request.path])
                record.request_info = f"[{display}] "
        return record
    ```

    确保你注意到我们需要保存默认的记录工厂，因为我们想利用它。然后当这个函数执行时，它会检查是否有当前请求，通过查看我们刚刚设置的请求上下文。

1.  我们还需要更新我们的格式以使用这条新信息。确保更新这个值：

    ```py
    DEFAULT_LOGGING_FORMAT = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] %(request_info)s%(message)s"
    ```

1.  最后，我们可以按照以下方式注入新的工厂：

    ```py
    from functools import partial
    def setup_logging(app: Sanic):
        ...
        logging.setLogRecordFactory(partial(_record_factory, app=app))
    ```

    随意检查这本书的 GitHub 仓库，以确保你的`log.py`看起来像我的一样：[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/tracing`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/tracing).

1.  在所有这些准备就绪后，是时候访问我们的端点了。你现在应该在终端看到一些漂亮的颜色，以及一些请求信息被插入：

    ```py
    [2021-10-21 12:22:48 +0300] [DEBUG] [server.py:12] [b5e7da51-68b0-4add-a850-9855c0a16814 GET /] This is a DEBUG message
    [2021-10-21 12:22:48 +0300] [INFO] [server.py:13] [b5e7da51-68b0-4add-a850-9855c0a16814 GET /] This is a INFO message
    [2021-10-21 12:22:48 +0300] [WARNING] [server.py:14] [b5e7da51-68b0-4add-a850-9855c0a16814 GET /] This is a WARNING message
    [2021-10-21 12:22:48 +0300] [ERROR] [server.py:15] [b5e7da51-68b0-4add-a850-9855c0a16814 GET /] This is a ERROR message
    [2021-10-21 12:22:48 +0300] [CRITICAL] [server.py:16] [b5e7da51-68b0-4add-a850-9855c0a16814 GET /] This is a CRITICAL message
    ```

在运行完这些示例后，你可能注意到了一些之前没有注意到的事情，那就是`request.id`。这是什么，它从哪里来？

### 使用 X-Request-ID

使用 UUID 来跟踪请求是一种常见的做法。这使得客户端应用程序跟踪请求并将它们与特定实例相关联变得非常容易。这就是为什么你经常会听到它们被称为关联 ID。如果你听到这个术语，它们就是同一回事。

作为关联请求实践的一部分，许多客户端应用程序会发送一个 X-Request-ID 头。如果 Sanic 在传入的请求中看到这个头，那么它将获取这个 ID 并使用它来识别请求。如果没有，那么它将自动为你生成一个 UUID。因此，你应该能够发送以下请求到我们的日志应用，并看到日志中填充了这个 ID：

```py
$ curl localhost:7777 -H 'x-request-id: abc123'
```

为了简化，我没有使用 UUID。

你的日志现在应该反映这一点：

```py
[2021-10-21 12:36:00 +0300] [DEBUG] [server.py:12] [abc123 GET /] This is a DEBUG message
[2021-10-21 12:36:00 +0300] [INFO] [server.py:13] [abc123 GET /] This is a INFO message
[2021-10-21 12:36:00 +0300] [WARNING] [server.py:14] [abc123 GET /] This is a WARNING message
[2021-10-21 12:36:00 +0300] [ERROR] [server.py:15] [abc123 GET /] This is a ERROR message
[2021-10-21 12:36:00 +0300] [CRITICAL] [server.py:16] [abc123 GET /] This is a CRITICAL message
```

记录日志是专业级 Web 应用的一个关键组件。它实际上并不需要那么复杂。我见过超级冗长且过于冗词的配置，老实说这让我感到害怕。然而，只要稍微注意细节，你就可以轻松地获得一个真正出色的日志体验。我鼓励你获取这个源代码，并修改它直到满足你的需求。

我们接下来将注意力转向 Web 应用的另一个关键组件：数据库管理。

## 管理数据库连接

这本书首先希望为你提供信心，让你能够以自己的方式构建应用程序。这意味着我们正在积极努力消除复制粘贴式开发。你知道我的意思。你访问**Stackoverflow**或另一个网站，复制代码，粘贴，然后继续你的日子，没有多想。

这种复制粘贴的心态在数据库连接方面可能最为普遍。是时候接受挑战了。启动一个新的 Sanic 应用程序并将其连接到数据库。一些开发者可能会通过前往其他代码库（来自另一个项目、文章、文档或帮助网站）来应对这个挑战，复制一些基本的连接函数，更改凭证，然后结束。他们可能从未深入思考过连接到数据库意味着什么：如果它工作，那么它就是好的。我知道我确实长时间这样做过。

这不是我们在这里要做的。相反，我们将考虑一些常见的场景，思考我们的担忧，并围绕它们开发解决方案。

### 使用 ORM 还是不要使用 ORM，这是一个问题

为了让那些不知道 ORM 是什么的人受益，这里有一个简短的定义：

**ORM** 是一个用于构建 Python 原生对象的框架。这些对象直接与数据库模式相关联，并且也用于构建查询以从数据库中检索数据，以便在构建 Python 对象时使用。换句话说，它们是一个具有从 Python 和数据库双向翻译能力的数据访问层。当人们谈论 ORM 时，他们通常指的是旨在与基于 SQL 的数据库一起使用的 ORM。

关于是否使用 ORM 的疑问充满了强烈的观点。在某些情况下，如果不用 ORM 而是手动编写 SQL 查询，人们可能会认为你生活在石器时代。另一方面，有些人可能会认为 ORM 是一种麻烦，会导致过于简单化，同时又复杂且低效的查询。我想在一定程度上，这两组人都是正确的。

理想情况下，我无法告诉你应该做什么或不应该做什么。实现细节和用例与任何决策都高度相关。在我的项目中，我倾向于避免使用它们。我喜欢使用 `databases` 项目（[`github.com/encode/databases`](https://github.com/encode/databases)）来构建自定义 SQL 查询，然后将结果映射到 `dataclass` 对象。在手工编写我的 SQL 之后，我使用一些实用工具将它们从原始、非结构化的值转换为模式定义的 Python 对象。我过去也广泛使用了像 peewee ([`github.com/coleifer/peewee`](https://github.com/coleifer/peewee)) 和 SQLAlchemy ([`github.com/sqlalchemy/sqlalchemy`](https://github.com/sqlalchemy/sqlalchemy)) 这样的 ORM。当然，由于我多年来一直在 Django 中开发，我在其内部 ORM 上也做了很多工作。

你应该在什么时候使用 ORM？首先，对于大多数项目来说，使用 ORM 可能应该是默认选项。它们在增加所需的安全性和安全性方面非常出色，以确保你不会意外地引入安全漏洞。通过强制类型，它们在维护数据完整性方面可以极为有益。当然，还有抽象化大量数据库知识的优势。它们可能不足之处在于处理复杂性的能力。随着项目表的数量和相互关系的增加，可能更难继续使用 ORM。此外，SQL 语言（如 Postgresql）中还有许多更高级的选项，这些选项你无法通过 ORM 构建查询来完成。我发现它们在更简单的 CRUD（创建/读取/更新/删除）应用中表现得非常出色，但实际上会阻碍更复杂的数据库模式。

ORM 的另一个潜在缺点是，它们使你自己的项目变得非常容易受到破坏。在构建低效查询时犯的一个小错误可能会导致极端长的响应时间，以及超级快的响应。作为一名经历过这种错误的人，我发现使用 ORM 构建的应用程序往往会过度获取数据，并且运行比所需的更多网络调用。如果你对 SQL 感到舒适，并且知道你的数据将变得相当复杂，那么你可能最好编写自己的 SQL 查询。使用手工编写的 SQL 的最大好处是它克服了 ORM 的复杂性扩展问题。

尽管这本书不是关于 SQL 的，但经过深思熟虑，我认为我们最好的时间是构建一个自定义数据层，而不是使用现成的 ORM。这个选项将迫使我们做出良好的选择，以维护我们的连接池并开发安全且实用的 SQL 查询。此外，这里讨论的任何关于实现的内容都可以轻松地替换为功能更全面的 ORM。如果你更熟悉并舒适地使用 SQLAlchemy（现在有异步支持），那么请随意相应地替换我的代码。

### 在 Sanic 中创建自定义数据访问层

在决定这本书使用哪种策略时，我探索了那里的大量选项。我查看了我看到人们与 Sanic 一起使用的所有流行的 ORM。一些选项（如 SQLAlchemy）有如此多的材料，我根本无法做到公正。其他选项鼓励较低质量的设计模式。因此，我们转向我最喜欢的之一，使用`asyncpg`连接到 Postgres（我选择的 SQL 关系数据库）。目标是实现良好的连接管理，

我强烈建议您查看 GitHub 仓库中的代码[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/hikingapp`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/09/hikingapp)。这是我们第一次创建一个**完整**的应用程序。我的意思是，这是一个示例应用程序，它会出去获取一些数据。回到*第二章，组织项目*中关于项目布局的讨论，您将看到我们可能如何构建一个真实世界的应用程序的示例。那里还有很多内容，与这里的讨论（主要关注数据库连接）有些不同，所以我们现在不会深入探讨。但请放心，当我们构建完整的应用程序时，我们将在*第十一章*再次回到应用程序的模式。同时，现在可能是您回顾源代码的好机会。尝试理解项目的结构，运行它，然后测试一些端点。说明在仓库中：[`github.com/PacktPublishing/Web-Development-with-Sanic/blob/main/chapters/09/hikingapp/README.md`](https://github.com/PacktPublishing/Web-Development-with-Sanic/blob/main/chapters/09/hikingapp/README.md)。

我还想指出，随着我们新增了一个服务，我们的应用程序正在增长，所以我打算开始使用 docker-compose 和 Docker 容器在本地上运行它。所有构建材料都存放在 GitHub 仓库中，供您复制以满足自己的需求。但当然，您不会仅仅复制粘贴代码而不真正理解它，所以让我们确保您确实做到了。

我们正在讨论的应用程序是一个用于存储徒步旅行详情的 Web API。它将其已知的徒步旅行数据库连接到可以跟踪他们徒步旅行总距离以及何时徒步旅行某些路线的用户。当您启动数据库时，应该有一些预先填充的信息供您使用。

我们必须做的第一件事是确保我们的连接细节来自环境变量。永远不要将它们存储在项目文件中。除了与这种做法相关的安全顾虑之外，如果您需要更改连接池的大小或轮换密码，通过重新部署应用程序并使用不同的值来更改这些值非常有帮助。让我们开始：

1.  使用 docker-compose、kubernetes 或其他您用于运行容器的工具来存储您的连接设置。如果您不在容器中运行 Sanic（例如，您计划部署到一个通过 GUI 提供环境变量的 PAAS），那么我喜欢的用于本地开发的一个选项是`dotenv` ([`github.com/theskumar/python-dotenv`](https://github.com/theskumar/python-dotenv))。目前我们关心的配置值是**数据源名称**（**DSN**）和池设置。如果您不熟悉 DSN，它是一个包含连接到数据库所需所有信息的字符串，其形式可能对您来说像 URL 一样熟悉。*什么是连接池？*想象一个场景，当网络请求到来时，您的应用程序会打开一个网络套接字到您的数据库。它获取信息，序列化并发送回客户端。但是，它也会关闭该连接。下次发生这种情况时，您的应用程序需要重新打开到数据库的连接。这非常低效。相反，您的应用程序可以通过打开它们并保留在备用中，来预热几个连接。

1.  然后，当应用程序需要连接时，它不需要打开新的连接，而可以直接通过连接池连接到您的数据库，并将该对象存储在您的应用程序`ctx`上：

    ```py
    # ./application/hiking/worker/postgres.py
    @app.before_server_start
    async def setup_postgres(app, _):
        app.ctx.postgres = Database(
            app.config.POSTGRES_DSN,
            min_size=app.config.POSTGRES_MIN,
            max_size=app.config.POSTGRES_MAX,
        )
    @app.after_server_start
    async def connect_postgres(app, _):
        await app.ctx.postgres.connect()
    @app.after_server_stop
    async def shutdown_postgres(app, _):
        await app.ctx.postgres.disconnect()
    ```

    如您所见，有三个主要的事情正在发生：

    1.  第一件事是我们创建了一个数据库对象，它存储我们的连接池并作为查询的接口。我们将其存储在`app.ctx`对象上，以便它可以在应用程序的任何地方轻松访问。这被放置在`before_server_start`监听器中，因为它改变了我们应用程序的状态。

    1.  第二个是监听器实际上打开了到数据库的连接，并在需要之前保持它们就绪。我们提前预热连接池，这样我们就不需要在查询时间上花费开销。

    1.  当然，我们做的最重要的步骤是确保我们的应用程序正确关闭其连接。

1.  下一步我们需要做的是创建我们的端点。在这个例子中，我们将使用基于类的视图：

    ```py
    from sanic import Blueprint, json, Request
    from sanic.views import HTTPMethodView
    from .executor import TrailExecutor
    bp = Blueprint("Trails", url_prefix="/trails")
    class TrailListView(HTTPMethodView, attach=bp):
        async def get(self, request: Request):
            executor = TrailExecutor(request.app.ctx.postgres)
            trails = await executor.get_all_trails()
            return json({"trails": trails})
    ```

    在这里，根级别的 `/trails` 端点的 `GET` 端点旨在提供数据库中所有小径的列表（不考虑分页）。`TrailExecutor` 是那些我不打算现在深入探讨的对象之一。但正如你可能从这段代码中猜到的，它接受我们数据库的实例（我们在上一步中启动的）并提供从数据库获取数据的方法。我非常喜欢数据库包的一个原因是因为它使得处理连接池和会话管理变得极其简单。它基本上在幕后为你做了所有事情。但有一个好习惯是无论你使用什么系统都应该养成（将多个连续写入数据库的操作包裹在一个事务中）。想象一下，你需要做类似这样的事情：

    ```py
    executor = FoobarExecutor(app.ctx.postgres)
    await executor.update_foo(value=3.141593)
    await executor.update_bar(value=1.618034)
    ```

经常在单个函数中有多个数据库写入时，你可能希望它们全部成功或全部失败。例如，成功和失败的混合可能会使你的应用程序处于不良状态。当你识别出这种情况时，几乎总是有益于在单个事务中嵌套你的函数。为了在我们的示例中实现此类事务，它看起来可能像这样：

```py
executor = FoobarExecutor(app.ctx.postgres)
async with app.ctx.postgres.transaction():
    await executor.update_foo(value=3.141593)
    await executor.update_bar(value=1.618034)
```

现在，如果由于任何原因查询失败，数据库状态将回滚到更改之前的状态。我强烈建议你无论使用什么框架连接到数据库，都采用类似的实践。

当然，关于数据库的讨论并不一定局限于 SQL 数据库。有许多 NoSQL 选项可供选择，而你当然应该找出适合你需求的那一个。接下来，我们将看看如何将我最喜欢的数据库选项 Redis 连接到 Sanic：

### 将 Sanic 连接到 Redis

Redis 是一个快速且简单的数据库，易于工作。许多人认为它仅仅是一个键/值存储，这是它做得非常好的事情。它还具有许多其他可以被视为某种共享原始数据类型的功能。例如，Redis 有哈希表、列表和集合。这些与 Python 的 `dict`、`list` 和 `set` 对应得很好。正因为如此，我经常向需要跨水平扩展共享数据的人推荐这个解决方案。

在我们的示例中，我们将使用 Redis 作为缓存层。为此，我们依赖于其哈希表功能来存储一个类似 `dict` 的结构，其中包含有关响应的详细信息。我们有一个可能需要几秒钟才能生成响应的端点。现在让我们模拟一下：

1.  首先创建一个需要一段时间才能生成响应的路由：

    ```py
    @app.get("/slow")
    async def wow_super_slow(request):
        wait_time = 0
        for _ in range(10):
            t = random.random()
            await asyncio.sleep(t)
            wait_time += t
        return text(f"Wow, that took {wait_time:.2f}s!")
    ```

1.  检查它是否工作：

    ```py
    $ curl localhost:7777/slow
    Wow, that took 5.87s!
    ```

为了解决这个问题，我们将创建一个装饰器，其任务是查找预缓存响应并在存在的情况下提供该响应：

1.  首先，我们将安装 `aioredis`：

    ```py
    $ pip install aioredis
    ```

1.  创建一个类似于上一节中我们所做的数据库连接池：

    ```py
    from sanic import Sanic
    import aioredis
    app = Sanic.get_app()
    @app.before_server_start
    async def setup_redis(app, _):
        app.ctx.redis_pool = aioredis.BlockingConnectionPool.from_url(
            app.config.REDIS_DSN, max_connections=app.config.REDIS_MAX
        )
        app.ctx.redis = aioredis.Redis(connection_pool=app.ctx.redis_pool)
    @app.after_server_stop
    async def shutdown_redis(app, _):
        await app.ctx.redis_pool.disconnect()
    ```

1.  接下来，我们将创建一个用于端点的装饰器。

    ```py
    def cache_response(build_key, exp: int = 60 * 60 * 72):
        def decorator(f):
            @wraps(f)
            async def decorated_function(request, *handler_args, **handler_kwargs):
                cache: Redis = request.app.ctx.redis
                key = make_key(build_key, request)
                if cached_response := await get_cached_response(request, cache, key):
                    response = raw(**cached_response)
                else:
                    response = await f(request, *handler_args, **handler_kwargs)
                    await set_cached_response(response, cache, key, exp)
                return response
            return decorated_function
        return decorator
    ```

    这里发生的事情相当简单。首先，我们生成一些键，这些键将用于查找和存储值。然后我们检查是否有任何与该键相关的内容。如果有，则使用它来构建响应。如果没有，则执行实际的路由处理程序（我们知道这需要一些时间）。

1.  让我们看看我们在实际操作中取得了什么成果。首先，我们将再次访问端点。为了强调我的观点，我将包括一些来自`curl`的统计数据：

    ```py
    $ curl localhost:7777/v1/slow
    Wow, that took 5.67s!
    status=200  size=21 time=5.686937 content-type="text/plain; charset=utf-8"
    ```

1.  现在，我们将再次尝试：

    ```py
    $ curl localhost:7777/v1/slow           
    Wow, that took 5.67s!
    status=200  size=21 time=0.004090 content-type="text/plain; charset=utf-8"
    ```

哇！它几乎是瞬间返回的！在第一次尝试中，它只用了不到 6 秒来响应。在第二次，因为信息已经被存储在 Redis 中，我们大约在 4/1000 秒内得到了相同的响应。而且，别忘了在这 4/1000 秒内，Sanic 已经去 Redis 获取数据了。太神奇了！

使用 Redis 作为缓存层非常强大，因为它可以显著提高你的性能。然而，正如之前使用过缓存的人所知道的那样，你需要有一个合适的用例和一个用于使缓存失效的机制。在上面的例子中，这通过两种方式实现。如果你查看 GitHub 上的源代码([`github.com/PacktPublishing/Web-Development-with-Sanic/blob/main/chapters/09/hikingapp/application/hiking/common/cache.py#L43`](https://github.com/PacktPublishing/Web-Development-with-Sanic/blob/main/chapters/09/hikingapp/application/hiking/common/cache.py#L43))，你会看到我们自动在 72 小时后过期值，或者如果有人向端点发送`?refresh=1`查询参数。

## 摘要

既然我们已经超越了讨论应用开发基本概念的阶段，我们已经提升到了探索我在多年开发 Web 应用过程中所学到的最佳实践的水平。这显然只是冰山一角，但它们是一些非常重要的基础实践，我鼓励你采纳。本章的例子可以成为开始你的下一个 Web 应用流程的绝佳基础。

首先，我们看到了如何使用智能和可重复的异常处理来为你的用户提供一致和周到的体验。其次，我们探讨了创建可测试应用程序的重要性，以及一些使其易于接近的技术。第三，我们讨论了在开发和生产环境中实现日志记录，以及如何使用这些日志轻松地调试和跟踪应用程序中的请求。最后，我们花了时间学习如何将数据库集成到你的应用程序中。

在下一章中，我们将继续扩展我们已经建立的基本平台。你将看到很多相同的模式（如日志记录）在我们的例子中继续出现，因为我们查看 Sanic 的一些常见用例。
