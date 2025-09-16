# 13

# 错误处理

**错误处理**是任何 Web 应用程序用户体验中的关键组件。**Flask**提供了几个内置的工具和选项，用于以干净和高效的方式处理错误。错误处理的目标是捕获和响应在应用程序执行过程中可能发生的错误，例如运行时错误、异常和无效的用户输入。

Flask 提供了一个内置的调试器，可以在开发过程中用于捕获和诊断错误。那么，为什么错误处理在任何 Web 应用程序中都是一个如此重要的概念呢？错误处理机制在预期向北发展却向南发展时，向用户提供有意义的错误消息，有助于维护用户体验的整体质量。此外，主动的错误处理使得调试变得容易。

如果错误处理实现得很好，那么调试问题和识别应用程序中问题的根本原因就会变得更容易。作为开发者，你也会希望通过预测和处理潜在的错误来提高应用程序的可靠性。这无疑使得你的应用程序更加可靠，并且不太可能在意外情况下崩溃。

在本章中，我们将探讨处理 Flask Web 应用程序中错误的不同策略和技术。你将了解并学习如何使用内置的**Flask 调试器**、实现**错误处理器**以及创建自定义的**错误页面**，以便向用户提供有意义的反馈。

在本章中，你将学习以下主题：

+   使用 Flask 调试器

+   创建错误处理器

+   创建自定义错误页面

+   跟踪应用程序中的事件

+   向管理员发送错误邮件

# 技术要求

本章的完整代码可在 GitHub 上找到：[`github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter13`](https://github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter13)。

# 使用 Flask 调试器

Flask 作为一个轻量级的 Python 网络框架，被广泛用于构建 Web 应用程序。使用 Flask 的一个即用即得的好处是其内置的调试器，它为识别和修复应用程序中的错误提供了一个强大的工具。

当你的 Flask 应用程序发生错误时，调试器会自动激活。调试器将提供关于错误的详细信息，包括堆栈跟踪、源代码上下文以及错误发生时在作用域内的任何变量。这些信息对于确定错误的根本原因和修复它的可能想法至关重要。

Flask 调试器还提供了一些交互式工具，可以用来检查应用程序的状态并理解正在发生什么。例如，你可以评估表达式并检查变量的值。你还可以在代码中设置断点，逐行执行代码以查看其执行情况。

让我们通过以下代码片段来进行分析：

```py
import pdb@app.route("/api/v1/debugging")
def debug():
    a = 10
    b = 20
    pdb.set_trace()
    c = a + b
    return f"The result is: {c}"
```

在这种情况下，你可以在`c = a + b`之前的行设置一个断点，就像前面的代码中所做的那样，并运行应用程序。当断点被触发时，你可以进入调试器并检查`a`、`b`和`c`的值。你还可以评估表达式并查看其结果。例如，要评估表达式`a + b`，你可以在调试器的命令提示符中输入`a + b`并按*Enter*。结果`30`将被显示。你还可以使用`n`命令逐行执行代码，使用`c`命令继续执行直到下一个断点。

通过这种方式，你可以使用 Flask 调试器的交互式工具来了解应用程序中发生的情况，并更有效地进行调试。这对于处理大型或复杂的代码库特别有用。当没有额外工具和信息时，Flask 调试器的交互式工具在难以理解导致错误的根本原因时非常有用。

除了交互式工具之外，Flask 还提供了一个可以启用的调试模式，可以提供更详细的错误信息。当启用调试模式时，Flask 将显示包含错误信息、堆栈跟踪和源代码上下文的详细错误页面。这些信息对于调试复杂问题非常有帮助。

要启用 Flask 调试器，只需在你的 Flask 应用程序中将`debug`配置值设置为`True`。在本项目书中，我们在`.env`文件中设置了此参数。你应该只在开发中使用它，因为它可能会向任何有权访问它的人透露你应用程序的敏感信息。

此外，Flask 还允许使用第三方扩展来增强调试体验。例如，`Flask-DebugToolbar`提供了一个工具栏，可以添加到你的应用程序中，以显示有关当前请求及其上下文的信息。

Flask 的内置调试器是一个强大的工具，可以帮助你快速识别和修复应用程序中的错误。无论你是在处理小型项目还是企业级应用程序，调试器都提供了有助于解决问题和提高应用程序可靠性和性能的有价值信息。

接下来，我们将讨论并实现 Flask Web 应用程序中的错误处理器。

# 创建错误处理器

Flask 还提供了一个处理错误的机制，称为错误处理器。错误处理器是在你的应用程序中发生特定错误时被调用的函数。这些函数可以用来返回自定义错误页面、记录有关错误的日志，或者执行任何适合错误的操作。要在 Flask Web 应用程序中定义错误处理器，你需要使用`errorhandler`装饰器。

装饰器接受错误代码作为其参数，并装饰的函数是当发生该错误时将被调用的错误处理器。错误处理器函数接受一个错误对象作为参数，该对象提供了关于发生错误的信息。这些信息可以用来向客户端提供更详细的错误响应，或者为了调试目的记录有关错误的附加信息。

在 Flask 后端和 **React** 前端应用程序中，错误处理是确保流畅用户体验的关键步骤。如前所述，错误处理器的目标是当出现问题时向用户提供有意义的反馈，而不仅仅是返回一个通用的错误消息。

例如，您可以定义错误处理器来处理错误 `400`、`404` 和 `500`。

## Flask 后端

以下代码展示了为 HTTP 错误代码 `404`（未找到）、`400`（错误请求）和 `500`（内部服务器错误）创建的错误处理器：

```py
from flask import jsonify@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400
@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'internal server error'}), 500
```

`not_found`、`bad_request` 和 `internal_server_error` 函数返回包含错误消息的 `JSON` 响应，以及相应的 HTTP 错误代码。

## React 前端

在 React 前端，您可以通过向 Flask 后端发送 HTTP 请求并检查响应中的错误来处理这些错误。例如，您可以在 React 中使用 **Axios**：

```py
import React, { useState, useEffect } from 'react';import axios from 'axios';
const Speakers = () => {
    const [error, setError] = useState(null);
    useEffect(() => {
        axios.get('/api/v1/speakers')
        .then(response => {
            // handle success
        })
        .catch(error => {
            switch (error.response.status) {
                case 404:
                    setError('Resource not found.');
                    break;
                case 400:
                    setError('Bad request');
                    break;
                case 500:
                    setError('An internal server error
                        occurred.');
                    break;
                default:
                    setError('An unexpected error
                        occurred.');
                    break;
            }
        });
    }, []);
    return (
        <div>
            {error ? <p>{error}</p> : <p>No error</p>}
        </div>
    );
};
export default Speakers;
```

上述错误处理代码展示了 React 前端与 Flask 后端 API 的通信。代码导入了 `React`、`useState` 和 `useEffect` 钩子，以及用于发送 API 请求的 `axios` 库。然后，代码定义了一个功能组件 `Speakers`，该组件向后端的 `/api/v1/speakers` 端点发送 API `GET` 请求。

`useEffect` 钩子用于管理 API 调用，响应在 `.then()` 块中处理成功，在 `.catch()` 块中处理错误。在 `.catch()` 块中，检查错误响应的状态并根据状态码设置特定的错误消息。例如，如果状态码是 `404`，则将“资源未找到”设置为错误。

错误信息随后通过条件渲染在 UI 中显示，如果没有错误，则显示“无错误”文本。错误信息使用 `useState` 钩子在状态中存储，初始值为 `null`。

接下来，我们将讨论和实现 Flask 网络应用程序中的自定义错误页面。

# 创建自定义错误页面

除了 Flask 中的错误处理器外，您还可以创建自定义错误页面，以提供更好的用户体验。当您的应用程序发生错误时，错误处理器可以返回一个包含错误信息、解决问题说明或任何其他适当内容的自定义错误页面。

在 Flask 中创建自定义错误页面，只需创建一个如前文所述的错误处理器，并返回一个包含错误页面内容的 `JSON` 响应。

例如，让我们看一下以下代码中包含自定义错误消息的 `JSON` 响应：

```py
@app.errorhandler(404)def not_found(error):
    return jsonify({'error': 'Not found'}), 404
```

以下代码在发生 `404` 错误时返回包含错误消息的 `JSON` 响应，以及相应的 HTTP 错误代码。让我们定义 React 前端以使用 `ErrorPage` 组件处理 UI：

```py
import React from 'react';const ErrorPage = ({ error }) => (
    <div>
        <h1>An error has occurred</h1>
        <p>{error}</p>
    </div>
);
export default ErrorPage;
```

以下代码显示了 `ErrorPage` 组件，它接受一个错误属性并在错误消息中显示它。您可以在应用程序中使用此组件在发生错误时显示自定义错误页面。

您可以直接将 `ErrorPage` 组件添加到应用程序的其余部分。例如，使用以下代码将 `ErrorPage` 组件添加到 `Speaker` 组件：

```py
import React, { useState, useEffect } from 'react';import axios from 'axios';
import ErrorPage from './ErrorPage';
const Speakers = () => {
    const [error, setError] = useState(null);
    useEffect(() => {
        axios.get('/api/v1/speakers')
            .then(response => {
                // handle success
            })
            .catch(error => {
                setError(error.response.data.error);
            });
    }, []);
    if (error) {
        return <ErrorPage error={error} />;
    }
    return (
        // rest of your application
    );
};
export default Speakers;
```

接下来，我们将讨论如何在 Flask 网络应用程序中跟踪和记录事件。

# 在您的应用程序中跟踪事件

Flask 允许您以优雅的方式跟踪应用程序中的事件。这对于识别潜在问题至关重要。通过跟踪事件，您可以更好地了解应用程序中正在发生的事情，并就如何改善情况做出明智的决策。

在 Flask 中跟踪事件有几种方法，包括使用内置的日志记录功能、第三方日志记录服务或自定义代码跟踪。例如，您可以使用 Python 的 `logging` 模块将有关应用程序活动的信息记录到文件或控制台。

使用日志记录模块很简单；只需将 `logging` 导入到您的 Flask 应用程序中，并配置它以适当的级别记录信息。例如，以下代码配置了日志记录模块以将信息记录到名为 `error.log` 的文件中：

```py
import loggingfrom flask import Flask
app = Flask(__name__)
# Set up a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Specify the log file
file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.DEBUG)
# Add the handler to the logger
logger.addHandler(file_handler)
@app.route('/logger')
def logger():
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    return 'Log messages have been written to the log file'
if __name__ == '__main__':
    app.run()
```

以下代码演示了在 Flask 网络应用程序中实现日志记录模块的方法。代码使用 `logging.getLogger(__name__)` 方法设置了一个日志记录器对象。日志记录器设置为调试级别，使用 `logger.setLevel(logging.DEBUG)`。使用 `file_handler = logging.FileHandler('error.log')` 创建了一个 `FileHandler` 对象，并将处理程序设置为调试级别，同样使用 `file_handler.setLevel(logging.DEBUG)`。

使用 `logger.addHandler(file_handler)` 将处理程序添加到日志记录器对象。在 `logger()` 函数中，调用了四个日志记录方法：`debug()`、`info()`、`warning()` 和 `error()`。这些方法将消息记录到日志文件中，并带有相应的日志级别（调试、信息、警告和错误）。记录的消息是简单的字符串消息。

此外，在跟踪 Flask 应用程序中的事件时，您可以使用第三方日志记录服务。使用 Flask 与第三方日志记录服务结合可以提供更高级的日志记录功能，如集中式日志管理、实时日志搜索和警报。

例如，您可以使用基于云的日志管理服务，如 **AWS CloudWatch**、**Loggly** 和 **Papertrail**。

简单地考察一下 AWS CloudWatch 的实现。AWS CloudWatch 是一种日志服务，为 AWS 资源提供日志管理和监控。要使用 Flask 与 AWS CloudWatch，你可以使用 **CloudWatch Logs** API 直接将日志数据发送到 AWS CloudWatch。

以下步骤实现了使用 AWS CloudWatch 在 Flask 应用程序中进行日志记录：

1.  设置 AWS 账户并创建一个 **CloudWatch** **日志组**。

1.  安装 `boto3` 库，它提供了对 AWS CloudWatch API 的 Python 接口。使用 `pip install boto3` 安装 `Boto2` 并确保你的虚拟环境已激活。

1.  在你的 Flask 应用程序中导入 `boto3` 库，并使用你的 AWS 凭据进行配置。

1.  创建一个记录器并将其日志级别设置为所需的详细程度。

1.  在你的应用程序代码中，使用记录器以各种级别（如 info、warning、error 等）记录消息。

1.  配置记录器将日志发送到 AWS CloudWatch。这可以通过创建一个自定义处理程序来实现，该处理程序使用 `boto3` 库将日志消息发送到 CloudWatch。

1.  将你的 Flask 应用程序部署并监控 AWS CloudWatch 中的日志。

让我们来看看代码实现：

```py
import boto3import logging
from flask import Flask
app = Flask(__name__)
boto3.setup_default_session(
    aws_access_key_id='<your-access-key-id>',
    aws_secret_access_key='<your-secret-access-key>',
    region_name='<your-region>')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
cloudwatch = boto3.client('logs')
log_group_name = '<your-log-group-name>'
class CloudWatchHandler(logging.Handler):
    def emit(self, record):
        log_message = self.format(record)
        cloudwatch.put_log_events(
            logGroupName=log_group_name,
            logStreamName='<your-log-stream-name>',)
if __name__ == '__main__':
    app.run()
```

完整的源代码可以在 GitHub 上找到。

上述代码展示了如何使用 `boto3` 库将 Flask 应用程序的日志发送到 AWS CloudWatch 的实现方式。它的工作原理如下：

1.  导入 `boto3` 库并设置一个默认会话，指定 `AWS 访问密钥 ID`、`秘密访问密钥` 和 `区域名称`。

1.  使用 `logging` 模块创建一个记录器对象，并将日志级别设置为 `DEBUG`。

1.  使用 `boto3` 库创建一个 `CloudWatch` 客户端对象。

1.  创建一个名为 `CloudWatchHandler` 的自定义处理程序类，它继承自 `logging.Handler` 类并重写了其 `emit` 方法。在 `emit` 方法中，将日志消息格式化并发送到 `AWS CloudWatch`，使用 `CloudWatch` 客户端的 `put_log_events` 方法。

1.  创建一个 `CloudWatchHandler` 类的实例，并将其日志级别设置为 `DEBUG`。然后将此处理程序添加到记录器对象。

1.  创建一个名为 `/logging_with_aws_cloudwatch` 的路由，该路由使用记录器对象生成不同级别（`debug`、`info`、`warning` 和 `error`）的日志消息。

在 Flask 应用程序中处理错误和跟踪事件对于确保其可靠性和健壮性至关重要。有了 Flask 的内置调试器、错误处理器、自定义错误页面、日志记录和第三方日志库，你可以轻松诊断和解决在 Flask 应用程序开发中出现的问题。

现在你已经能够实现 Flask 内置的调试器、错误处理器、自定义错误页面、日志记录和第三方日志库，如果管理员能够实时收到关于应用程序日志中错误的通知，那岂不是很好？

让我们来看看如何在 Flask 中实现这一点。

# 向管理员发送错误邮件

向管理员发送错误邮件提供了一种高效的通知方式，让他们了解 Flask 应用程序中的错误和问题。这允许你在问题升级成更大问题并负面影响用户体验之前快速识别和解决问题。其好处包括及时识别和解决错误、提高系统可靠性和减少停机时间。

让我们深入探讨一个向管理员发送错误邮件的示例：

```py
import smtplibfrom email.mime.text import MIMEText
from flask import Flask, request
app = Flask(__name__)
def send_email(error):
    try:
        msg = MIMEText(error)
        msg['Subject'] = 'Error in Flask Application'
        msg['From'] = 'from@example.com'
        msg['To'] = 'to@example.com'
        s = smtplib.SMTP('localhost')
        s.send_message(msg)
        s.quit()
    except Exception as e:
        print(f'Error sending email: {e}')
@app.errorhandler(500)
def internal_server_error(error):
    send_email(str(error))
    return 'An error occurred and an email was sent to the
        administrator.', 500
if __name__ == '__main__':
    app.run()
```

上一段代码展示了在 Flask 应用程序中发送错误邮件以通知管理员错误实现的示例。它的工作原理如下：

1.  该代码使用 `smtplib` 和 `email.mime.text` 库来创建和发送电子邮件消息。

1.  `send_email(error)` 函数接受一个错误消息作为参数，并使用 `MIMEText` 对象创建一个电子邮件消息。邮件的主题、发件人电子邮件地址、收件人电子邮件地址和错误消息被设置为邮件内容。然后，通过本地邮件服务器使用 `smtplib` 库发送邮件。

Flask 的 `errorhandler` 装饰器用于捕获应用程序中发生的任何 `500` 内部服务器错误。当发生错误 `500` 时，会调用 `internal_server_error` 函数，并使用错误消息作为参数调用 `send_email` 函数。该函数返回一个响应给用户，表明发生了错误，并向管理员发送了电子邮件。

# 摘要

错误处理自古以来就是软件开发的一个基本方面。确保你的 Flask Web 应用程序能够有效地处理错误至关重要。我们讨论了 Flask 调试器、错误处理程序和自定义错误页面。有了这些，你可以向用户提供有意义的反馈，并帮助维护应用程序的稳定性和可靠性。

作为全栈开发者，我们强调了持续关注错误处理的重要性。你应该定期审查和更新你的错误处理策略，以确保你的应用程序保持健壮和弹性。我们还考虑了记录错误并向管理员发送通知，以便你可以快速识别和解决可能出现的任何问题。

简而言之，无 bug 的开发体验对于任何专业开发者来说都只是一个幻象。你应该准备好有效地处理你的 Web 应用程序中的预期和意外错误。通过这样做，即使面对意外错误和故障，你的应用程序也能继续为用户提供价值。

接下来，我们将探讨在 Flask 中使用 **Blueprints** 进行模块化开发。通过 Blueprints 和模块化架构，你可以轻松维护和扩展你的 React-Flask Web 应用程序。
