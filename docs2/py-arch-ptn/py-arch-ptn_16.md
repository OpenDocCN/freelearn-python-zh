# 12

# 日志记录

监控和可观察性的基本要素之一就是日志。日志使我们能够检测正在运行的系统中的动作。这些信息可以用来分析系统的行为，特别是可能出现的任何错误或缺陷，从而为我们提供对实际发生情况的宝贵见解。

正确使用日志看似困难，实则不然。很容易收集过多或过少的信息，或者记录错误的信息。在本章中，我们将探讨一些关键元素，以及确保日志发挥最佳效果的通用策略。

在本章中，我们将涵盖以下主题：

+   日志基础

+   在 Python 中生成日志

+   通过日志检测问题

+   日志策略

+   在开发时添加日志

+   日志限制

让我们从日志的基本原则开始。

# 日志基础

日志基本上是系统运行时产生的消息。这些消息是在代码执行时由特定的代码片段产生的，使我们能够跟踪代码中的动作。

日志可以是完全通用的，如“*函数 X 被调用*”，或者可以包含一些执行具体细节的上下文，如“*函数 X 使用参数 Y 被调用.*”

通常，日志以纯文本消息的形式生成。虽然还有其他选项，但纯文本处理起来非常简单，易于阅读，格式灵活，可以使用像`grep`这样的纯文本工具进行搜索。这些工具通常运行得非常快，大多数开发人员和系统管理员都知道如何使用它们。

除了主要的消息文本外，每个日志还包含一些元数据，例如产生日志的系统、日志创建的时间等。如果日志是文本格式，这些信息通常附加到行的开头。

标准和一致的日志格式有助于您进行消息的搜索、过滤和排序。请确保您在不同系统之间使用一致的格式。

另一个重要的元数据值是日志的严重性。这使我们能够根据相对重要性对不同的日志进行分类。标准严重性级别，按从低到高的顺序，是`DEBUG`、`INFO`、`WARNING`和`ERROR`。

`CRITICAL`级别使用较少，但用于显示灾难性错误是有用的。

重要的是要对日志进行适当的分类，并过滤掉不重要的消息，以便关注更重要的消息。每个日志设施都可以配置为仅生成一个或多个严重性级别的日志。

可以添加自定义日志级别，而不是预定义的级别。这通常不是一个好主意，在大多数情况下应避免，因为所有工具和工程师都很好地理解了日志级别。我们将在本章后面描述如何为每个级别定义策略，以充分利用每个级别。

在一个处理请求的系统（无论是请求-响应还是异步）中，大部分日志都会作为处理请求的一部分生成，这将产生几个日志，指示请求正在做什么。由于通常会有多个请求同时进行，日志将会混合生成。例如，考虑以下日志：

```py
Sept 16 20:42:04.130 10.1.0.34 INFO web: REQUEST GET /login

Sept 16 20:42:04.170 10.1.0.37 INFO api: REQUEST GET /api/login

Sept 16 20:42:04.250 10.1.0.37 INFO api: REQUEST TIME 80 ms

Sept 16 20:42:04.270 10.1.0.37 INFO api: REQUEST STATUS 200

Sept 16 20:42:04.360 10.1.0.34 INFO web: REQUEST TIME 230 ms

Sept 16 20:42:04.370 10.1.0.34 INFO web: REQUEST STATUS 200 
```

前面的日志显示了两个不同的服务，如不同的 IP 地址（`10.1.0.34` 和 `10.1.0.37`）和两种不同的服务类型（`web` 和 `api`）所示。虽然这足以区分请求，但创建一个单一的请求 ID 来能够以以下方式分组请求是个好主意：

```py
Sept 16 20:42:04.130 10.1.0.34 INFO web: [4246953f8] REQUEST GET /login

Sept 16 20:42:04.170 10.1.0.37 INFO api: [fea9f04f3] REQUEST GET /api/login

Sept 16 20:42:04.250 10.1.0.37 INFO api: [fea9f04f3] REQUEST TIME 80 ms

Sept 16 20:42:04.270 10.1.0.37 INFO api: [fea9f04f3] REQUEST STATUS 200

Sept 16 20:42:04.360 10.1.0.34 INFO web: [4246953f8] REQUEST TIME 230 ms

Sept 16 20:42:04.370 10.1.0.34 INFO web: [4246953f8] REQUEST STATUS 200 
```

在微服务环境中，请求将从一项服务流向另一项服务，因此创建一个跨服务共享的请求 ID 是个好主意，以便可以理解完整的跨服务流程。为此，请求 ID 需要由第一个服务创建，然后传输到下一个服务，通常作为 HTTP 请求的头部。

如我们在*第五章*，*十二要素应用方法*中看到的，在十二要素应用方法中，日志应该被视为一个事件流。这意味着应用程序本身不应该关心日志的存储和处理。相反，日志应该被导向 `stdout`。从那里，在开发应用程序时，开发者可以在运行时提取信息。

在生产环境中，`stdout` 应该被捕获，以便其他工具可以使用它，然后进行路由，将不同的来源合并到一个单一的流中，然后存储或索引以供以后查阅。这些工具应该在生产环境中配置，而不是在应用程序本身中配置。

可能用于此重路由的工具包括类似 Fluentd ([`github.com/fluent/fluentd`](https://github.com/fluent/fluentd)) 或甚至旧爱的直接到 `logger` Linux 命令来创建系统日志，然后将这些日志发送到配置的 `rsyslog` ([`www.rsyslog.com/`](https://www.rsyslog.com/)) 服务器，该服务器可以转发和聚合它们。

无论我们如何收集日志，一个典型的系统都会产生大量的日志，并且需要存储在某个地方。虽然每个单独的日志都很小，但聚合数千个日志会占用大量的空间。任何日志系统都应该配置为有一个政策，以确定它应该接受多少数据以避免无限增长。一般来说，基于时间（例如保留过去 15 天的日志）的保留策略是最好的方法，因为它将很容易理解。在需要回溯多远的历史和系统使用的空间量之间找到平衡是很重要的。

在启用任何新的日志服务时，无论是本地还是基于云的，务必检查保留策略，以确保它与您定义的保留期兼容。您将无法分析时间窗口之前发生的事情。请再次确认日志创建速率是否符合预期，以及空间消耗是否没有使您可以收集日志的有效时间窗口变小。您不希望在你追踪一个错误时意外地超出配额。

生成日志条目很简单，正如我们将在下一节中看到的，*在 Python 中生成日志*。

# 在 Python 中生成日志

Python 包含一个用于生成日志的标准模块。这个模块易于使用，具有非常灵活的配置，但如果你不了解它的操作方式，可能会令人困惑。

一个创建日志的基本程序看起来像这样。这个程序可以在 GitHub 上的 [`github.com/PacktPublishing/Python-Architecture-Patterns/tree/main/chapter_12_logging`](https://github.com/PacktPublishing/Python-Architecture-Patterns/tree/main/chapter_12_logging) 找到，文件名为 `basic_logging.py`。

```py
import logging

# Generate two logs with different severity levels

logging.warning('This is a warning message')

logging.info('This is an info message') 
```

`.warning` 和 `.info` 方法创建具有相应严重性消息的日志。消息是一个文本字符串。

当执行时，它会显示以下内容：

```py
$ python3 basic_logging.py

WARNING:root:This is a warning message 
```

默认情况下，日志会被路由到 `stdout`，这是我们想要的，但它被配置为不显示 `INFO` 级别的日志。日志的格式也是默认的，不包含时间戳。

要添加所有这些信息，我们需要了解 Python 中用于日志记录的三个基本元素：

+   一个 *格式化器*，它描述了完整的日志将如何呈现，附加元数据如时间戳或严重性。

+   一个 *处理器*，它决定了日志如何传播。它通过格式化器设置日志的格式，如上面定义的那样。

+   一个 *记录器*，它生成日志。它有一个或多个处理器，描述了日志如何传播。

使用这些信息，我们可以配置日志以指定我们想要的全部细节：

```py
import sys

import logging

# Define the format

FORMAT = '%(asctime)s.%(msecs)dZ:APP:%(name)s:%(levelname)s:%(message)s'

formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")

# Create a handler that sends the logs to stdout

handler = logging.StreamHandler(stream=sys.stdout)

handler.setFormatter(formatter)

# Create a logger with name 'mylogger', adding the handler and setting

# the level to INFO

logger = logging.getLogger('mylogger')

logger.addHandler(handler)

logger.setLevel(logging.INFO)

# Generate three logs

logger.warning('This is a warning message')

logger.info('This is an info message')

logger.debug('This is a debug message, not to be displayed') 
```

我们按照之前看到的顺序定义这三个元素。首先是 `formatter`，然后是 `handler`，它设置 `formatter`，最后是 `logger`，它添加 `handler`。

`formatter` 的格式如下：

```py
FORMAT = '%(asctime)s.%(msecs)dZ:APP:%(name)s:%(levelname)s:%(message)s'

formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%dT%H:%M:%S") 
```

`FORMAT` 是由 Python `%` 格式组成的，这是一种描述字符串的旧方法。大多数元素都描述为 `%(name)s`，其中最后的 `s` 字符表示字符串格式。以下是每个元素的描述：

+   `asctime` 将时间戳设置为可读的格式。我们在 `datefmt` 参数中描述它，以遵循 ISO 8601 格式。我们还添加了毫秒数和一个 `Z` 来获取完整的 ISO 8601 格式的时间戳。`%(msecs)d` 末尾的 `d` 表示我们将值打印为整数。这是为了将值限制在毫秒，而不显示任何额外的分辨率，这可以作为分数值提供。

+   `name` 是记录器的名称，正如我们稍后将要描述的那样。我们还添加了 `APP` 以区分不同的应用程序。

+   `levelname`是日志的严重性，例如`INFO`、`WARNING`或`ERROR`。

+   最后，`message`是日志消息。

一旦我们定义了`formatter`，我们就可以转向`handler`：

```py
handler = logging.StreamHandler(stream=sys.stdout)

handler.setFormatter(formatter) 
```

处理器是一个`StreamHandler`，我们将流的目的地设置为`sys.stdout`，这是 Python 定义的指向`stdout`的变量。

有更多可用的处理器，如`FileHandler`，可以将日志发送到文件，`SysLogHandler`可以将日志发送到`syslog`目的地，还有更高级的案例，如`TimeRotatingFileHandler`，它根据时间旋转日志，这意味着它存储最后定义的时间，并归档旧版本。您可以在文档[`docs.python.org/3/howto/logging.html#useful-handlers`](https://docs.python.org/3/howto/logging.html#useful-handlers)中查看所有可用处理器的更多信息。

一旦定义了`handler`，我们就可以创建`logger`：

```py
logger = logging.getLogger('mylogger')

logger.addHandler(handler)

logger.setLevel(logging.INFO) 
```

首件事是为 logger 创建一个名称，这里我们将其定义为`mylogger`。这允许我们将应用程序的日志划分为子部分。我们使用`.addHandler`附加处理器。

最后，我们使用`.setLevel`方法将日志级别定义为`INFO`。这将显示所有`INFO`级别及以上的日志，而低于此级别的日志则不会显示。

如果我们运行文件，我们会看到整个配置组合在一起：

```py
$ python3 configured_logging.py

2021-09-18T23:15:24.563Z:APP:mylogger:WARNING:This is a warning message

2021-09-18T23:15:24.563Z:APP:mylogger:INFO:This is an info message 
```

我们可以看到：

+   时间定义为 ISO 8601 格式，即`2021-09-18T23:15:24.563Z`。这是`asctime`和`msec`参数的组合。

+   `APP`和`mylogger`参数允许我们通过应用程序和子模块进行筛选。

+   显示了严重性。请注意，有一个未显示的`DEBUG`消息，因为配置的最小级别是`INFO`。

Python 中的`logging`模块能够进行高级别的配置。有关更多信息，请参阅官方文档[`docs.python.org/3/library/logging.html`](https://docs.python.org/3/library/logging.html)。

# 通过日志检测问题

对于运行中的系统中的任何问题，都可能发生两种错误：预期错误和意外错误。在本节中，我们将通过日志来了解它们之间的区别，以及我们如何处理它们。

## 检测预期错误

预期错误是通过在代码中创建`ERROR`日志来显式检测到的错误。例如，以下代码在访问的 URL 返回的状态码不是`200 OK`时产生`ERROR`日志：

```py
import logging

import requests

URL = 'https://httpbin.org/status/500'

response = requests.get(URL)

status_code = response.status_code

if status_code != 200:

    logging.error(f'Error accessing {URL} status code {status_code}') 
```

当执行此代码时，会触发一个`ERROR`日志：

```py
$ python3 expected_error.py

ERROR:root:Error accessing https://httpbin.org/status/500 status code 500 
```

这是一种常见的模式，用于访问外部 URL 并验证其是否正确访问。生成日志的块可以执行一些补救措施或重试，以及其他操作。

在这里，我们使用[`httpbin.org`](https://httpbin.org)服务，这是一个简单的 HTTP 请求和响应服务，可用于测试代码。特别是，`https://httpbin.org/status/<code>`端点返回指定的状态码，这使得生成错误变得容易。

这是一个预期错误的例子。我们事先计划了某些我们不希望发生的事情，但理解了它可能发生的可能性。通过提前规划，代码可以准备好处理错误并充分捕获它。

在这种情况下，我们可以清楚地描述情况，并提供上下文来理解正在发生的事情。问题很明显，即使解决方案可能不是。

这类错误相对容易处理，因为它们描述的是预见的问题。

例如，网站可能不可用，可能存在认证问题，或者可能是基本 URL 配置错误。

请记住，在某些情况下，代码可能能够处理某种情况而不失败，但它仍然被视为错误。例如，你可能想检测是否有某人仍在使用旧的认证系统。当检测到已弃用的操作时，添加`ERROR`或`WARNING`日志的方法可以让你采取行动来纠正情况。

这种类型错误的其他例子包括数据库连接和以已弃用的格式存储的数据。

## 捕获意外错误

但预期错误并不是唯一可能发生的错误。不幸的是，任何正在运行的系统都会以各种意想不到的方式让你感到惊讶，从而以创新的方式破坏代码。Python 中的意外错误通常是在代码的某个点抛出异常时产生的，而这个异常不会被捕获。

例如，想象当我们对某些代码进行小改动时，我们引入了一个拼写错误：

```py
import logging

import requests

URL = 'https://httpbin.org/status/500'

logging.info(f'GET {URL}')

response = requests.ge(URL)

status_code = response.status_code

if status_code != 200:

    logging.error(f'Error accessing {URL} status code {status_code}') 
```

注意，在第 8 行中，我们引入了一个拼写错误：

```py
response = requests.ge(URL) 
```

正确的`.get`调用已被`.ge`替换。当我们运行它时，会产生以下错误：

```py
$ python3 unexpected_error.py

Traceback (most recent call last):

  File "./unexpected_error.py", line 8, in <module>

    response = requests.ge(URL)

AttributeError: module 'requests' has no attribute 'ge' 
```

在 Python 中默认情况下，它会在`stdout`中显示错误和堆栈跟踪。当代码作为 Web 服务器的一部分执行时，这有时足以将这些消息作为`ERROR`日志发送，具体取决于配置如何设置。

任何网络服务器都会捕获并正确地将这些消息路由到日志中，并生成适当的 500 状态码，指示发生了意外错误。服务器仍然可以处理下一个请求。

如果你需要创建一个需要无限期运行并保护不受任何意外错误影响的脚本，请确保使用`try..except`块，因为它是一般性的，所以任何可能的异常都将被捕获和处理。

任何使用特定`except`块正确捕获的 Python 异常都可以被认为是预期错误。其中一些可能需要生成`ERROR`消息，但其他可能被捕获和处理，而不需要此类信息。

例如，让我们调整代码，使其每隔几秒发送一次请求。代码在 GitHub 上可用，链接为 [`github.com/PacktPublishing/Python-Architecture-Patterns/tree/main/chapter_12_logging`](https://github.com/PacktPublishing/Python-Architecture-Patterns/tree/main/chapter_12_logging):

```py
import logging

import requests

from time import sleep

logger = logging.getLogger()

logger.setLevel(logging.INFO)

while True:


    try:

        sleep(3)

        logging.info('--- New request ---')


        URL = 'https://httpbin.org/status/500'

        logging.info(f'GET {URL}')

        response = requests.ge(URL)

        scode = response.status_code

        if scode != 200:

            logger.error(f'Error accessing {URL} status code {scode}')

    except Exception as err:

        logger.exception(f'ERROR {err}') 
```

关键元素是以下无限循环：

```py
while True:

    try:

        code

    except Exception as err:

        logger.exception(f'ERROR {err}') 
```

`try..except`块在循环内部，所以即使有错误，循环也不会中断。如果有任何错误，`except Exception`将捕获它，无论异常是什么。

这有时被称为*宝可梦异常处理*，就像“抓到它们所有”。这应该限制为一种“最后的救命网”。一般来说，不精确地捕获异常是一个坏主意，因为你可以通过错误地处理它们来隐藏错误。错误永远不应该无声地通过。

为了确保不仅记录了错误，还记录了完整的堆栈跟踪，我们使用`.exception`而不是`.error`来记录它。这通过`ERROR`严重性记录扩展了信息，而不会超过单条文本消息。

当我们运行命令时，我们会得到这些日志。确保通过按*Ctrl* + *C*来停止它：

```py
$ python3 protected_errors.py

INFO:root:--- New request ---

INFO:root:GET https://httpbin.org/status/500

ERROR:root:ERROR module 'requests' has no attribute 'ge'

Traceback (most recent call last):

  File "./protected_errors.py", line 18, in <module>

    response = requests.ge(URL)

AttributeError: module 'requests' has no attribute 'ge'

INFO:root:--- New request ---

INFO:root:GET https://httpbin.org/status/500

ERROR:root:ERROR module 'requests' has no attribute 'ge'

Traceback (most recent call last):

  File "./protected_errors.py", line 18, in <module>

    response = requests.ge(URL)

AttributeError: module 'requests' has no attribute 'ge'

^C

...

KeyboardInterrupt 
```

正如你所见，日志中包含了`Traceback`，这使我们能够通过添加异常产生位置的信息来检测特定的问题。

任何意外错误都应该记录为`ERROR`。理想情况下，它们还应该被分析，并更改代码以修复错误或至少将它们转换为预期的错误。有时这由于其他紧迫的问题或问题发生的频率低而不可行，但应该实施一些策略以确保处理错误的连贯性。

处理意外错误的一个优秀工具是 Sentry ([`sentry.io/`](https://sentry.io/))。这个工具在许多常见的平台上为每个错误创建触发器，包括 Python Django、Ruby on Rails、Node、JavaScript、C#、iOS 和 Android。它聚合检测到的错误，并允许我们更策略性地处理它们，这在仅仅能够访问日志时有时是困难的。

有时，意外错误会提供足够的信息来描述问题，这可能涉及到外部问题，如网络问题或数据库问题。解决方案可能位于服务本身之外。

# 日志策略

处理日志时常见的一个问题是确定每个单独服务的适当严重性。这条消息是`WARNING`还是`ERROR`？这个声明应该添加为`INFO`消息吗？

大多数日志严重性描述都有定义，例如*程序显示一个可能有害的情况*或*应用程序突出显示请求的进度*。这些是模糊的定义，在现实生活中的情况下很难采取行动。与其使用这些模糊的定义，不如尝试将每个级别与如果遇到问题应该采取的任何后续行动相关联来定义。这有助于向开发者阐明在发现给定的错误日志时应该做什么。例如：“*我是否希望每次这种情况发生时都得到通知？*”

下表显示了不同严重级别的示例以及可能采取的行动：

| 日志级别 | 应采取的操作 | 备注 |
| --- | --- | --- |
| `DEBUG` | 无。 | 不跟踪。仅在开发期间有用。 |
| `INFO` | 无。 | `INFO` 日志显示有关应用程序中操作流程的通用信息，以帮助跟踪系统。 |
| `WARNING` | 跟踪日志数量。当级别上升时发出警报。 | `WARNING` 日志跟踪可以自动修复的错误，如尝试连接外部服务或数据库中的可修复格式错误。突然增加可能需要调查。 |
| `ERROR` | 跟踪日志数量。当级别上升时发出警报。审查所有错误。 | `ERROR` 日志跟踪无法恢复的错误。突然增加可能需要立即采取行动。所有这些错误都应定期审查，以修复常见问题并减轻它们，可能将它们提升到 `WARNING` 级别。 |
| `CRITICAL` | 立即响应。 | `CRITICAL` 日志表明应用程序发生了灾难性故障。单个 `CRITICAL` 日志表示系统完全无法工作且无法恢复。 |

这明确了如何响应的期望。请注意，这是一个示例，你可能需要根据你特定组织的需要对其进行调整和修改。

不同严重程度的层次结构非常清晰，在我们的示例中，我们接受将产生一定数量的 `ERROR` 日志。为了开发团队的理智，不是所有问题都需要立即修复，但应强制执行一定的顺序和优先级。

在生产环境中，`ERROR` 日志通常会被分类从“我们完了”到“嗯”。开发团队应积极修复“嗯”日志或停止记录问题，以从监控工具中去除噪音。这可能包括降低日志级别，如果它们不值得检查的话。你希望尽可能少的 `ERROR` 日志，这样所有的日志都是有意义的。

记住，`ERROR` 日志将包括通常需要修复以完全解决或明确捕获并降低其严重性（如果它不重要）的意外错误。

随着应用程序的增长，这种后续工作无疑是一个挑战，因为 `ERROR` 日志的数量将显著增加。这需要投入时间进行主动维护。如果不认真对待，并且过于频繁地因其他任务而放弃，则会在中期损害应用程序的可靠性。

`WARNING` 日志表明某些事情可能没有像预期那样顺利运行，但情况仍在控制之中，除非此类日志的数量突然增加。`INFO` 日志仅在出现问题时提供上下文，否则可以忽略。

一个常见的错误是在存在错误输入参数的操作中生成`ERROR`日志，例如在 Web 请求中返回`400 BAD REQUEST`状态码时。一些开发者可能会争辩说，客户发送的格式不正确的请求是一个错误。但如果请求被正确检测并返回，开发团队就没有什么需要做的。这是正常的业务，唯一可能采取的行动可能是向请求者返回一个有意义的消息，以便他们可以修复他们的请求。

如果这种行为在某些关键请求中持续存在，例如反复发送错误的密码，可以创建一个`WARNING`日志。当应用程序按预期运行时，创建`ERROR`日志是没有意义的。

在 Web 应用程序中，一般来说，只有在状态码是 50X 变体之一（如 500、502 和 503）时才应创建`ERROR`日志。记住，40X 错误意味着发送者有问题，而 50X 意味着应用程序有问题，这是你团队的责任去修复。

在团队中采用常见和共享的日志级别定义，所有工程师都将对错误严重性有一个共同的理解，这将有助于形成改进代码的有意义行动。

允许时间对任何定义进行调整和微调。也可能需要处理在定义之前创建的日志，这可能需要工作。在遗留系统中，最大的挑战之一是创建一个适当的日志系统来分类问题，因为这些问题可能非常嘈杂，使得区分真正的问题、烦恼甚至非问题变得困难。

# 在开发过程中添加日志

任何测试运行器都会在运行测试时捕获日志，并将其作为跟踪的一部分显示出来。

我们在*第十章*，*测试和 TDD*中介绍的`pytest`将显示失败的测试结果中的日志。

这是一个检查在功能开发阶段是否生成预期日志的好机会，尤其是在它是作为 TDD 过程的一部分完成的情况下，在 TDD 过程中，失败的测试和错误作为过程的一部分常规产生，正如我们在*第十章*，*测试和 TDD*中看到的。任何检查错误的测试都应该添加相应的日志，并且在开发功能时检查它们是否被生成。

您可以使用像`pytest-catchlog`（[`pypi.org/project/pytest-catchlog/`](https://pypi.org/project/pytest-catchlog/)）这样的工具显式地向测试添加一个检查，以验证日志是否被生成。

通常，我们只需稍加注意，并在使用 TDD 实践作为初始检查的一部分时，将检查日志的做法纳入其中。然而，确保开发者理解为什么在开发过程中拥有日志是有用的，以便养成习惯。

在开发过程中，可以使用`DEBUG`日志来添加关于代码流程的额外信息，这些信息对于生产来说可能过多。在开发中，这些额外信息可以帮助填补`INFO`日志之间的差距，并帮助开发者养成添加日志的习惯。如果测试期间发现`DEBUG`日志在生产中跟踪问题很有用，则可以将`DEBUG`日志提升为`INFO`。

此外，在特殊情况下，可以在受控的情况下在生产环境中启用`DEBUG`日志来跟踪难以理解的问题。请注意，这将对生成的日志数量产生重大影响，可能导致存储问题。因此，请非常谨慎。

对于显示在`INFO`和更高严重性日志中的消息，要有理智。在显示的信息方面，应避免敏感数据，如密码、密钥、信用卡号码和个人信息。

在生产过程中，要注意任何大小的限制以及日志生成的速度。当新功能生成、请求数量增加或系统中工人的数量增加时，系统可能会经历日志爆炸。这三种情况可以在系统增长时出现。

总是检查日志是否被正确捕获并在不同的环境中可用是一个好主意。确保日志被正确捕获的所有配置可能需要一些时间，因此最好事先完成。这包括在生产环境中捕获意外错误和其他日志，并检查所有管道是否正确完成。另一种选择是在遇到真正的问题后才发现它没有正确工作。

# 日志限制

日志对于理解正在运行的系统中的情况非常有用，但它们有一些重要的局限性需要了解：

+   *日志的价值取决于其消息。一个好的、描述性的消息对于使日志有用至关重要。用批判性的眼光审查日志消息，并在需要时进行纠正，对于在生产问题上节省宝贵时间非常重要。

+   *应保持适当的日志数量。过多的日志可能会使流程混乱，而过少的日志可能不会包含足够的信息，使我们能够理解问题。大量的日志也会引起存储问题。

+   *日志应作为问题上下文的指示，但很可能不会精确指出问题所在。*试图生成能够完全解释错误的特定日志将是一项不可能的任务。相反，应专注于展示动作的一般流程和周围上下文，以便可以在本地复制并调试。例如，对于请求，确保记录请求及其参数，以便可以复制情况。

+   *日志使我们能够跟踪单个实例的执行过程*。当使用请求 ID 或类似方式分组时，日志可以根据执行进行分组，使我们能够跟踪请求或任务的流程。然而，日志并不直接显示汇总信息。日志回答的问题是“*在这个任务中发生了什么？*”，而不是“*系统中正在发生什么？*”对于这类信息，最好使用指标。

    有可用的工具可以根据日志创建指标。我们将在第十三章*指标*中更多地讨论指标。

+   *日志仅具有回顾性功能*。当检测到任务中的问题时，日志只能显示事先准备好的信息。这就是为什么批判性地分析和精炼信息很重要，移除无用的日志，并添加包含相关上下文信息的其他日志，以帮助重现问题。

日志是一个出色的工具，但它们需要维护以确保它们可以用来检测错误和问题，并允许我们尽可能高效地采取行动。

# 摘要

在本章中，我们首先介绍了日志的基本元素。我们定义了日志包含消息以及一些元数据，如时间戳，并考虑了不同的严重级别。我们还描述了定义请求 ID 以分组与同一任务相关的日志的需求。此外，我们还讨论了在十二要素应用方法中，日志应发送到`stdout`，以便将日志生成与处理和路由到适当目的地的过程分离，从而允许收集系统中的所有日志。

然后，我们展示了如何使用标准的`logging`模块在 Python 中生成日志，描述了`logger`、`handler`和`formatter`的三个关键元素。接下来，我们展示了系统中可能产生的两种不同错误：*预期的*，理解为可以预见并得到处理的错误；和*意外的*，意味着那些我们没有预见并且超出我们控制范围的错误。然后我们探讨了这些错误的不同策略和案例。

我们描述了不同的严重性以及当检测到特定严重性的日志时，应采取哪些行动的策略，而不是根据“它们有多关键”来对日志进行分类，这最终会产生模糊的指南，并且不太有用。

我们讨论了几个习惯，通过在 TDD 工作流程中包含它们来提高日志的有用性。这允许开发者在编写测试和产生错误时考虑日志中呈现的信息，这为确保生成的日志正确工作提供了完美的机会。

最后，我们讨论了日志的限制以及我们如何处理它们。

在下一章中，我们将探讨如何通过使用指标来处理汇总信息，以找出系统的总体状态。
