

# 第十章：测试模式

在前面的章节中，我们介绍了架构模式和针对特定用例（如并发或性能）的模式。

在本章中，我们将探讨特别适用于测试的设计模式。这些模式有助于隔离组件，使测试更加可靠，并促进代码重用。

在本章中，我们将介绍以下主要主题：

+   模拟对象模式

+   依赖注入模式

# 技术要求

请参阅在*第一章*中提出的需求。

# 模拟对象模式

**模拟对象**模式是一种在测试期间通过模拟其行为来隔离组件的强大工具。模拟对象有助于创建受控的测试环境并验证组件之间的交互。

模拟对象模式提供了三个功能：

1.  **隔离**：模拟将正在测试的代码单元隔离，确保测试在受控环境中运行，其中依赖项是可预测的，并且没有外部副作用。

1.  **行为验证**：通过使用模拟对象，您可以在测试期间验证某些行为是否发生，例如方法调用或属性访问。

1.  **简化**：它们通过替换可能需要大量设置的复杂真实对象来简化测试的设置。

与存根的比较

存根也替换了真实实现，但仅用于向被测试的代码提供间接输入。相比之下，模拟可以验证交互，使它们在许多测试场景中更加灵活。

## 现实世界示例

我们可以想到以下现实世界的类比概念或工具：

+   飞行模拟器，这是一种旨在复制实际驾驶飞机体验的工具。它允许飞行员在受控和安全的环境中学习如何处理各种飞行场景。

+   **心肺复苏**（**CPR**）模拟人，用于教授学生如何有效地进行心肺复苏。它模拟人体以提供一个真实但受控的学习环境。

+   碰撞测试模拟人，由汽车制造商用于模拟人类对车辆碰撞的反应。它提供了关于汽车碰撞影响和安全特性的宝贵数据，而无需冒实际人类生命危险。

## 模拟对象模式的用例

在**单元测试**中，模拟对象用于替换被测试代码的复杂、不可靠或不可用的依赖项。这允许开发者仅关注单元本身，而不是它与外部系统的交互。例如，当测试一个从 API 获取数据的服务时，模拟对象可以通过返回预定义的响应来模拟 API，确保服务能够处理各种数据场景或错误，而无需与实际 API 交互。

虽然与单元测试类似，但使用模拟对象的**集成测试**侧重于组件之间的交互，而不是单个单元。模拟可以用来模拟尚未开发或成本过高而无法参与每个测试的组件。例如，在微服务架构中，模拟可以代表一个正在开发或暂时不可用的服务，允许其他服务在如何集成和与其通信方面进行测试。

模拟对象模式对于**行为验证**也非常有用。此用例涉及验证对象之间是否发生预期的某些交互。模拟对象可以被编程为期望特定的调用、参数甚至交互顺序，这使得它们成为行为测试的强大工具；例如，测试控制器在**模型-视图-控制器**（**MVC**）架构中在处理用户请求之前是否正确地调用了身份验证和日志记录服务。模拟可以验证控制器是否以正确的顺序进行了正确的调用，例如在尝试记录请求之前检查凭证。

## 实现模拟对象模式

假设我们有一个将消息记录到文件的函数。我们可以模拟文件写入机制，以确保我们的日志记录函数将预期的内容写入日志，而不写入文件。让我们看看如何使用 Python 的`unittest`模块来实现这一点。

首先，我们导入示例中需要的模块：

```py
import unittest
from unittest.mock import mock_open, patch
```

然后，我们创建一个表示简单日志记录器的类，该记录器将消息写入初始化期间指定的文件：

```py
class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
    def log(self, message):
        with open(self.filepath, "a") as file:
            file.write(f"{message}\n")
```

接下来，我们创建一个继承自`unittest.TestCase`类的测试用例类，就像通常一样。在这个类中，我们需要`test_log()`方法来测试日志记录器的`log()`方法，如下所示：

```py
class TestLogger(unittest.TestCase):
    def test_log(self):
        msg = "Hello, logging world!"
```

接下来，我们将在测试范围内直接模拟 Python 内置的`open()`函数。模拟函数是通过使用`unittest.mock.patch()`来完成的，它临时用模拟对象（调用`mock_open()`的结果）替换了目标对象，即`builtins.open`。通过调用`unittest.mock.patch()`函数获得的上下文管理器，我们创建一个`Logger`对象并调用其`.log()`方法，这应该会触发`open()`函数：

```py
        m_open = mock_open()
        with patch("builtins.open", m_open):
            logger = Logger("dummy.log")
            logger.log(msg)
```

关于`builtins`

根据 Python 文档，`builtins`模块提供了对 Python 所有内置标识符的直接访问；例如，`builtins.open`是`open()`内置函数的全名。见[`docs.python.org/3/library/builtins.html`](https://docs.python.org/3/library/builtins.html)。

关于`mock_open`

当你调用`mock_open()`时，它返回一个配置为像内置的`open()`函数一样行为的 Mock 对象。此模拟被设置为模拟文件操作，如读取和写入。

关于`unittest.mock.patch`

它用于在测试期间用模拟对象替换对象。它的参数包括 `target`，用于指定要替换的对象，以及可选参数：`new` 用于可选的替换对象，`spec` 和 `autospec` 用于将模拟限制在真实对象的属性上以提高准确性，`spec_set` 用于更严格的属性指定，`side_effect` 用于定义条件行为或异常，`return_value` 用于设置固定的响应，`wraps` 用于在修改某些方面时允许原始对象的行为。这些选项使测试场景中的精确控制和灵活性成为可能。

现在，我们检查日志文件是否正确打开，我们使用两种验证方法来完成。对于第一个验证，我们在模拟对象上使用 `assert_called_once_with()` 方法，以检查 `open()` 函数是否以预期的参数被调用。对于第二个验证，我们需要从 `unittest.mock.mock_open` 中获取更多技巧；我们的 `m_open` 模拟对象，通过调用 `mock_open()` 函数获得，也是一个可调用对象，每次被调用时都像是一个创建新模拟文件句柄的工厂。我们使用它来获取一个新的文件句柄，然后在该文件句柄上的 `write()` 方法调用上使用 `assert_called_once_with()`，这有助于我们检查 `write()` 方法是否以正确的消息被调用。测试函数的这一部分如下：

```py
            m_open.assert_called_once_with(
                "dummy.log", "a"
            )
            m_open().write.assert_called_once_with(
                f"{msg}\n"
            )
```

最后，我们调用 `unitest.main()`：

```py
if __name__ == "__main__":
    unittest.main()
```

要执行示例（在 `ch10/mock_object.py` 文件中），像往常一样，运行以下命令：

```py
python ch10/mock_object.py
```

你应该得到以下输出：

```py
.
---------------------------------------------------------
Ran 1 test in 0.012s
OK
```

这只是一个快速演示，展示了如何在单元测试中使用模拟来模拟系统的一部分。我们可以看到，这种方法隔离了副作用（即文件 I/O），确保单元测试不会创建或需要实际文件。它允许测试类的内部行为，而不需要为了测试目的而改变类的结构。

# 依赖注入模式

依赖注入模式涉及将类的依赖项作为外部实体传递，而不是在类内创建它们。这促进了松散耦合、模块化和可测试性。

## 现实世界的例子

在现实生活中，我们会遇到以下例子：

+   **电器和电源插座**：各种电器可以插入不同的电源插座，使用电力而无需直接和永久布线

+   **相机镜头**：摄影师可以在不改变相机本身的情况下，根据不同的环境和需求更换相机的镜头

+   **模块化列车系统**：在模块化列车系统中，可以根据每次旅行的需求添加或移除单个车厢（如卧铺车厢、餐厅车厢或行李车厢）

## 依赖注入模式的用例

在 Web 应用程序中，将数据库连接对象注入到组件（如仓库或服务）中，可以增强模块化和可维护性。这种做法允许轻松地在不同的数据库引擎或配置之间切换，而无需直接修改组件的代码。它还通过允许注入模拟数据库连接，从而简化了单元测试过程，从而在不影响实时数据库的情况下测试各种数据场景。

另一种使用场景是管理跨各种环境（开发、测试、生产等）的配置设置。通过动态将设置注入到模块中，**依赖注入**（**DI**）减少了模块与其配置源之间的耦合。这种灵活性使得在不进行大量重新配置的情况下，更容易管理和切换环境。在单元测试中，这意味着你可以注入特定的设置来测试模块在不同配置下的表现，确保其健壮性和功能。

## 实现依赖注入模式 - 使用模拟对象

在这个第一个例子中，我们将创建一个简单的场景，其中`WeatherService`类依赖于`WeatherApiClient`接口来获取天气数据。对于示例的单元测试代码，我们将注入该 API 客户端的模拟版本。

我们首先定义任何天气 API 客户端实现应遵守的接口，使用 Python 的`Protocol`功能：

```py
from typing import Protocol
class WeatherApiClient(Protocol):
    def fetch_weather(self, location):
        """Fetch weather data for a given location"""
        ...
```

然后，我们添加一个`RealWeatherApiClient`类，该类实现了该接口，并将与我们的天气服务进行交互。在实际场景中，在提供的`fetch_weather()`方法中，我们会调用天气服务，但为了使示例简单并专注于本章的主要概念；所以我们提供了一个模拟，简单地返回一个表示天气数据结果的字符串。代码如下：

```py
class RealWeatherApiClient:
    def fetch_weather(self, location):
        return f"Real weather data for {location}"
```

接下来，我们创建一个天气服务，它使用实现`WeatherApiClient`接口的对象来获取天气数据：

```py
class WeatherService:
    def __init__(self, weather_api: WeatherApiClient):
        self.weather_api = weather_api
    def get_weather(self, location):
        return self.weather_api.fetch_weather(location)
```

最后，我们准备好通过`WeatherService`构造函数注入 API 客户端的依赖。我们添加代码来帮助手动测试示例，使用以下真实服务：

```py
if __name__ == "__main__":
    ws = WeatherService(RealWeatherApiClient())
    print(ws.get_weather("Paris"))
```

在我们的示例的这一部分（在`ch10/dependency_injection/di_with_mock.py`文件中）可以通过以下命令手动测试：

```py
python ch10/dependency_injection/di_with_mock.py
```

你应该得到以下输出：

```py
ch10/dependency_injection/test_di_with_mock.py).
			First, we import the `unittest` module, as well as the `WeatherService` class (from our `di_with_mock` module), as follows:

```

导入 unittest 模块

from di_with_mock import WeatherService

```py

			Then, we create a mock version of the weather API client implementation that will be useful for unit testing, simulating responses without making real API calls:

```

class MockWeatherApiClient:

def fetch_weather(self, location):

return f"为 {location} 的模拟天气数据"

```py

			Next, we write the test case class, with a test function. In that function, we inject the mock API client instead of the real API client, passing it to the `WeatherService` constructor, as follows:

```

class TestWeatherService(unittest.TestCase):

def test_get_weather(self):

mock_api = MockWeatherApiClient()

weather_service = WeatherService(mock_api)

self.assertEqual(

weather_service.get_weather("Anywhere"),

"为任何地方的模拟天气数据",

)

```py

			We finish by adding the usual lines for executing unit tests when the file is interpreted by Python:

```

if __name__ == "__main__":

unittest.main()

```py

			Executing this part of the example (in the `ch10/dependency_injection/test_di_with_mock.py` file), using the `python ch10/dependency_injection/test_di_with_mock.py` command, gives the following output:

```

.

---------------------------------------------------------

执行了 1 个测试，耗时 0.000 秒

OK

```py

			The test with the dependency injected using a mock object succeeded.
			Through this example, we were able to see that the `WeatherService` class doesn’t need to know whether it’s using a real or a mock API client, making the system more modular and easier to test.
			Implementing the Dependency Injection pattern – using a decorator
			It is also possible to use decorators for DI, which simplifies the injection process. Let’s see a simple example demonstrating how to do that, where we’ll create a notification system that can send notifications through different channels (for example, email or SMS). The first part of the example will show the result based on manual testing, and the second part will provide unit tests.
			First, we define a `NotificationSender` interface, outlining the methods any notification sender should have:

```

from typing import Protocol

class NotificationSender(Protocol):

def send(self, message: str):

"""使用给定消息发送通知"""

...

```py

			Then, we implement two specific notification senders: the `EmailSender` class implements sending a notification using email, and the `SMSSender` class implements sending using SMS. This part of the code is as follows:

```

class EmailSender:

def send(self, message: str):

打印(f"发送电子邮件：{message}")

class SMSSender:

def send(self, message: str):

打印(f"发送短信：{message}")

```py

			We also define a notification service class, `NotificationService`, with a class attribute sender and a `.notify()` method, which takes in a message and calls `.send()` on the provided sender object to send the message, as follows:

```

class NotificationService:

sender: NotificationSender = None

def notify(self, message):

self.sender.send(message)

```py

			What is missing is the decorator that will operate the DI, to provide the specific sender object to be used. We create our decorator to decorate the `NotificationService` class for injecting the sender. It will be used by calling `@inject_sender(EmailSender)` if we want to inject the email sender, or `@inject_sender(SMSSender)` if we want to inject the SMS sender. The code for the decorator is as follows:

```

def inject_sender(sender_cls):

def decorator(cls):

cls.sender = sender_cls()

return cls

return decorator

```py

			Now, if we come back to the notification service’s class, the code would be as follows:

```

@inject_sender(EmailSender)

class NotificationService:

sender: NotificationSender = None

def notify(self, message):

self.sender.send(message)

```py

			Finally, we can instantiate the `NotificationService` class in our application and notify a message for testing the implementation, as follows:

```

if __name__ == "__main__":

service = NotificationService()

service.notify("Hello, this is a test notification!")

```py

			That first part of our example (in the `ch10/dependency_injection/di_with_decorator.py` file) can be manually tested by using the following command:

```

python ch10/dependency_injection/di_with_decorator.py

```py

			You should get the following output:

```

发送电子邮件：您好，这是一条测试通知！

```py

			If you change the decorating line, replace the `EmailSender` class with `SMSSender`, and rerun that command, you will get the following output:

```

发送短信：您好，这是一条测试通知！

```py

			That shows the DI is effective.
			Next, we want to write unit tests for that implementation. We could use the mocking technique, but to see other ways, we are going to use the stub classes approach. The stubs manually implement the dependency interfaces and include additional mechanisms to verify that methods have been called correctly. Let’s start by importing what we need:

```

import unittest

from di_with_decorator import (

NotificationSender,

NotificationService,

inject_sender,

)

```py

			Then, we create stub classes that implement the `NotificationSender` interface. These classes will help record calls to their `send()` method, using the `messages_sent` attribute on their instances, allowing us to check whether the correct methods were called during the test. Both stub classes are as follows:

```

class EmailSenderStub:

def __init__(self):

self.messages_sent = []

def send(self, message: str):

self.messages_sent.append(message)

class SMSSenderStub:

def __init__(self):

self.messages_sent = []

def send(self, message: str):

self.messages_sent.append(message)

```py

			Next, we are going to use both stubs in our test case to verify the functionality of `NotificationService`. In the test function, `test_notify_with_email`, we create an instance of `EmailSenderStub`, inject that stub into the service, send a notification message, and then verify that the message was sent by the email stub. That part of the code is as follows:

```

class TestNotifService(unittest.TestCase):

def test_notify_with_email(self):

email_stub = EmailSenderStub()

service = NotificationService()

service.sender = email_stub

service.notify("Test Email Message")

self.assertIn(

"Test Email Message",

email_stub.messages_sent,

)

```py

			We need another function for the notification with SMS functionality, `test_notify_with_sms`. Similarly to the previous case, we create an instance of `SMSSenderStub`. Then, we need to inject that stub into the notification service. But, for that, in the scope of the test, we define a custom notification service class, and decorate it with `@inject_sender(SMSSenderStub)`, as follows:

```

@inject_sender(SMSSenderStub)

class CustomNotificationService:

sender: NotificationSender = None

def notify(self, message):

self.sender.send(message)

```py

			Based on that, we inject the SMS sender stub into the custom service, send a notification message, and then verify that the message was sent by the SMS stub. The complete code for the second unit test is as follows:

```

def test_notify_with_sms(self):

sms_stub = SMSSenderStub()

@inject_sender(SMSSenderStub)

class CustomNotificationService:

sender: NotificationSender = None

def notify(self, message):

self.sender.send(message)

service = CustomNotificationService()

service.sender = sms_stub

service.notify("Test SMS Message")

self.assertIn(

"Test SMS Message", sms_stub.messages_sent

)

```py

			Finally, we should not forget to add the lines needed for executing unit tests when the file is interpreted by Python:

```

if __name__ == "__main__":

unittest.main()

```py

			Executing the unit test code (in the `ch10/dependency_injection/test_di_with_decorator.py` file), using the `python ch10/dependency_injection/test_di_with_decorator.py` command, gives the following output:

```

..

---------------------------------------------------------

测试完成，运行了 2 个测试用例，耗时 0.000 秒

OK

```py

			This is what was expected.
			So, this example showed how using a decorator to manage dependencies allows for easy changes without modifying the class internals, which not only keeps the application flexible but also encapsulates the dependency management outside of the core business logic of your application. In addition, we saw how DI can be tested with unit tests using the stubs technique, ensuring the application’s components work as expected in isolation.
			Summary
			In this chapter, we’ve explored two pivotal patterns essential for writing clean code and enhancing our testing strategies: the Mock Object pattern and the Dependency Injection pattern.
			The Mock Object pattern is crucial for ensuring test isolation, which helps avoid unwanted side effects. It also facilitates behavior verification and simplifies test setup. We discussed how mocking, particularly through the `unittest.mock` module, allows us to simulate components within a unit test, demonstrating this with a practical example.
			The Dependency Injection pattern, on the other hand, offers a robust framework for managing dependencies in a way that enhances flexibility, testability, and maintainability. It’s applicable not only in testing scenarios but also in general software design. We illustrated this pattern with an initial example that integrates mocking for either unit or integration tests. Subsequently, we explored a more advanced implementation using a decorator to streamline dependency management across both the application and its tests.
			As we conclude this chapter and prepare to enter the final one, we’ll shift our focus slightly to discuss Python anti-patterns, identifying common pitfalls, and learning how to avoid them.

```
