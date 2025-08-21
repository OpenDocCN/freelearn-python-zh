# 第五章：使用装饰器改进我们的代码

在本章中，我们将探讨装饰器，并看到它们在许多情况下如何有用，我们想要改进我们的设计。我们将首先探讨装饰器是什么，它们是如何工作的，以及它们是如何实现的。

有了这些知识，我们将重新审视我们在以前章节中学到的关于软件设计的一般良好实践，并看看装饰器如何帮助我们遵守每个原则。

本章的目标如下：

+   了解 Python 中装饰器的工作原理

+   学习如何实现适用于函数和类的装饰器

+   有效实现装饰器，避免常见的实现错误

+   分析如何通过装饰器避免代码重复（DRY 原则）

+   研究装饰器如何有助于关注点分离

+   分析良好装饰器的示例

+   审查常见情况、习语或模式，以确定装饰器是正确的选择

# Python 中的装饰器是什么？

装饰器在 Python 中很久以前就被引入了（PEP-318），作为一种简化函数和方法定义的机制，当它们在原始定义之后需要被修改时。

最初的动机之一是因为诸如`classmethod`和`staticmethod`之类的函数被用来转换方法的原始定义，但它们需要额外的一行，修改函数的原始定义。

更一般地说，每当我们必须对函数应用转换时，我们必须使用`modifier`函数调用它，然后将其重新分配给与函数最初定义的相同名称。

例如，如果我们有一个名为`original`的函数，然后我们有一个在其上更改`original`行为的函数，称为`modifier`，我们必须编写类似以下的内容：

```py
def original(...):
    ...
original = modifier(original)
```

注意我们如何更改函数并将其重新分配给相同的名称。这很令人困惑，容易出错（想象有人忘记重新分配函数，或者确实重新分配了函数，但不是在函数定义后的下一行，而是在更远的地方），而且很麻烦。因此，语言中添加了一些语法支持。

前面的示例可以这样重写：

```py
@modifier
def original(...):
   ...
```

这意味着装饰器只是调用装饰器后面的内容作为装饰器本身的第一个参数的语法糖，结果将是装饰器返回的内容。

根据 Python 术语和我们的示例，`modifier`是我们称之为装饰器，`original`是被装饰的函数，通常也称为“wrapped”对象。

虽然最初的功能是为方法和函数设计的，但实际的语法允许对任何类型的对象进行装饰，因此我们将探讨应用于函数、方法、生成器和类的装饰器。

最后一点是，虽然装饰器的名称是正确的（毕竟，装饰器实际上是在对`wrapped`函数进行更改、扩展或处理），但它不应与装饰器设计模式混淆。

# 装饰函数

函数可能是 Python 对象的最简单表示形式，可以对函数使用装饰器来应用各种逻辑——我们可以验证参数、检查前提条件、完全改变行为、修改其签名、缓存结果（创建原始函数的记忆版本），等等。

例如，我们将创建一个实现“重试”机制的基本装饰器，控制特定领域级别的异常并重试一定次数：

```py
# decorator_function_1.py
class ControlledException(Exception):
    """A generic exception on the program's domain."""

def retry(operation):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        last_raised = None
        RETRIES_LIMIT = 3
        for _ in range(RETRIES_LIMIT):
            try:
                return operation(*args, **kwargs)
            except ControlledException as e:
                logger.info("retrying %s", operation.__qualname__)
                last_raised = e
        raise last_raised

    return wrapped
```

现在可以忽略`@wraps`的使用，因为它将在名为*有效装饰器-避免常见错误*的部分中进行介绍。在 for 循环中使用`_`，意味着这个数字被赋值给一个我们目前不感兴趣的变量，因为它在 for 循环内没有被使用（在 Python 中命名`_`的值被忽略是一个常见的习惯）。

`retry`装饰器不接受任何参数，因此可以轻松地应用到任何函数，如下所示：

```py
@retry
def run_operation(task):
    """Run a particular task, simulating some failures on its execution."""
    return task.run()
```

正如在开头解释的那样，在`run_operation`的顶部定义`@retry`只是 Python 提供的语法糖，实际上执行`run_operation = retry(run_operation)`。

在这个有限的例子中，我们可以看到装饰器如何被用来创建一个通用的`retry`操作，根据一定的条件（在这种情况下，表示为可能与超时相关的异常），允许调用被装饰的代码多次。

# 装饰类

类也可以使用相同的语法装饰（PEP-3129）作用于函数。唯一的区别是，在编写这个装饰器的代码时，我们必须考虑到我们接收到的是一个类，而不是一个函数。

一些从业者可能会认为装饰一个类是相当复杂的，这种情况可能会危及可读性，因为我们会在类中声明一些属性和方法，但在幕后，装饰器可能会应用会使一个完全不同的类。

这个评估是正确的，但只有在这种技术被滥用的情况下。客观地说，这与装饰函数没有什么不同；毕竟，类只是 Python 生态系统中的另一种对象类型，就像函数一样。我们将在标题为*装饰器和关注点分离*的部分中审查这个问题的利弊，但现在我们将探讨特别适用于类的装饰器的好处：

+   所有重用代码和 DRY 原则的好处。类装饰器的一个有效案例是强制多个类符合某个接口或标准（通过在将应用于这些多个类的装饰器中只进行一次检查）。

+   我们可以创建更小或更简单的类，稍后可以通过装饰器进行增强。

+   我们需要应用到某个类的转换逻辑，如果我们使用装饰器，将会更容易维护，而不是使用更复杂（通常是被不鼓励的）方法，比如元类。

在所有可能的装饰器应用中，我们将探讨一个简单的例子，以给出它们可以有用的事情的一些想法。请记住，这不是类装饰器的唯一应用类型，但我们展示的代码也可能有许多其他多种解决方案，都有各自的利弊，但我们选择了装饰器，目的是说明它们的用处。

回顾我们的监控平台的事件系统，现在我们需要为每个事件转换数据并将其发送到外部系统。然而，每种类型的事件在选择如何发送其数据时可能有其自己的特殊之处。

特别是，登录的`event`可能包含诸如我们想要隐藏的凭据之类的敏感信息。其他字段，比如`timestamp`，也可能需要一些转换，因为我们想以特定格式显示它们。满足这些要求的第一次尝试可能就像有一个映射到每个特定`event`的类，并且知道如何对其进行序列化：

```py
class LoginEventSerializer:
    def __init__(self, event):
        self.event = event

    def serialize(self) -> dict:
        return {
            "username": self.event.username,
            "password": "**redacted**",
            "ip": self.event.ip,
            "timestamp": self.event.timestamp.strftime("%Y-%m-%d 
             %H:%M"),
        }

class LoginEvent:
    SERIALIZER = LoginEventSerializer

    def __init__(self, username, password, ip, timestamp):
        self.username = username
        self.password = password
        self.ip = ip
        self.timestamp = timestamp

    def serialize(self) -> dict:
        return self.SERIALIZER(self).serialize()
```

在这里，我们声明了一个类，它将直接与登录事件进行映射，包含了它的逻辑——隐藏`password`字段，并按要求格式化`timestamp`。

虽然这种方法有效，看起来可能是一个不错的选择，但随着时间的推移，当我们想要扩展我们的系统时，我们会发现一些问题：

+   **类太多**：随着事件数量的增加，序列化类的数量也会按同等数量级增长，因为它们是一对一映射的。

+   **解决方案不够灵活**：如果我们需要重用组件的部分（例如，我们需要隐藏另一种类型的`event`中也有的`password`），我们将不得不将其提取到一个函数中，但也要从多个类中重复调用它，这意味着我们实际上并没有重用太多代码。

+   **样板代码**：`serialize()`方法必须存在于所有`event`类中，调用相同的代码。尽管我们可以将其提取到另一个类中（创建一个 mixin），但这似乎不是继承的好用法。

另一种解决方案是能够动态构造一个对象，给定一组过滤器（转换函数）和一个`event`实例，能够通过将这些过滤器应用于其字段来对其进行序列化。然后，我们只需要定义转换每种字段类型的函数，序列化器通过组合许多这些函数来创建。

有了这个对象后，我们可以装饰类以添加`serialize()`方法，它将只调用这些`Serialization`对象本身：

```py

def hide_field(field) -> str:
    return "**redacted**"

def format_time(field_timestamp: datetime) -> str:
    return field_timestamp.strftime("%Y-%m-%d %H:%M")

def show_original(event_field):
    return event_field

class EventSerializer:
    def __init__(self, serialization_fields: dict) -> None:
        self.serialization_fields = serialization_fields

    def serialize(self, event) -> dict:
        return {
            field: transformation(getattr(event, field))
            for field, transformation in 
            self.serialization_fields.items()
        }

class Serialization:

    def __init__(self, **transformations):
        self.serializer = EventSerializer(transformations)

    def __call__(self, event_class):
        def serialize_method(event_instance):
            return self.serializer.serialize(event_instance)
        event_class.serialize = serialize_method
        return event_class

@Serialization(
    username=show_original,
    password=hide_field,
    ip=show_original,
    timestamp=format_time,
)
class LoginEvent:

    def __init__(self, username, password, ip, timestamp):
        self.username = username
        self.password = password
        self.ip = ip
        self.timestamp = timestamp
```

请注意，装饰器使用户更容易知道每个字段将如何处理，而无需查看另一个类的代码。只需阅读传递给类装饰器的参数，我们就知道`username`和 IP 地址将保持不变，`password`将被隐藏，`timestamp`将被格式化。

现在，类的代码不需要定义`serialize()`方法，也不需要扩展实现它的 mixin，因为装饰器将添加它。实际上，这可能是唯一证明创建类装饰器的部分，因为否则，`Serialization`对象可以是`LoginEvent`的类属性，但它正在通过向其添加新方法来更改类的事实使其成为不可能。

此外，我们可以有另一个类装饰器，只需定义类的属性，就可以实现`init`方法的逻辑，但这超出了本示例的范围。这就是诸如`attrs`（ATTRS 01）这样的库所做的事情，标准库中的（PEP-557）也提出了类似的功能。

通过使用 Python 3.7+中的（PEP-557）中的这个类装饰器，可以以更紧凑的方式重写先前的示例，而不需要`init`的样板代码，如下所示：

```py
from dataclasses import dataclass
from datetime import datetime

@Serialization(
    username=show_original,
    password=hide_field,
    ip=show_original,
    timestamp=format_time,
)
@dataclass
class LoginEvent:
    username: str
    password: str
    ip: str
    timestamp: datetime
```

# 其他类型的装饰器

现在我们知道了装饰器的`@`语法实际上意味着什么，我们可以得出结论，不仅可以装饰函数、方法或类；实际上，任何可以定义的东西，例如生成器、协程，甚至已经被装饰的对象，都可以被装饰，这意味着装饰器可以被堆叠。

先前的示例展示了装饰器如何链接。我们首先定义了类，然后对其应用了`@dataclass`，将其转换为数据类，充当这些属性的容器。之后，`@Serialization`将对该类应用逻辑，从而产生一个新的类，其中添加了新的`serialize()`方法。

装饰器的另一个很好的用途是用于应该用作协程的生成器。我们将在第七章中探讨生成器和协程的细节，但主要思想是，在向新创建的生成器发送任何数据之前，必须通过调用`next()`将其推进到下一个`yield`语句。这是每个用户都必须记住的手动过程，因此容易出错。我们可以轻松地创建一个装饰器，它以生成器作为参数，调用`next()`，然后返回生成器。

# 将参数传递给装饰器

到目前为止，我们已经将装饰器视为 Python 中的强大工具。但是，如果我们可以向它们传递参数，使其逻辑更加抽象，它们可能会更加强大。

实现装饰器的几种方法，可以接受参数，但我们将介绍最常见的方法。第一种方法是将装饰器创建为嵌套函数，增加一个新的间接层，使装饰器中的所有内容深入一层。第二种方法是使用类作为装饰器。

一般来说，第二种方法更有利于可读性，因为以对象的方式思考比使用三个或更多个嵌套函数与闭包更容易。然而，为了完整起见，我们将探讨两种方法，读者可以决定对于手头的问题哪种方法更好。

# 带有嵌套函数的装饰器

大致来说，装饰器的一般思想是创建一个返回函数的函数（通常称为高阶函数）。在装饰器主体中定义的内部函数将是实际被调用的函数。

现在，如果我们希望向其传递参数，那么我们需要另一个间接层。第一个将接受参数，并在该函数内部，我们将定义一个新函数，这将是装饰器，然后将定义另一个新函数，即作为装饰过程的结果返回的函数。这意味着我们将至少有三个级别的嵌套函数。

如果到目前为止这还不清楚，不要担心。在查看即将出现的示例之后，一切都会变得清晰起来。

我们看到的第一个装饰器的示例是在一些函数上实现`retry`功能。这是一个好主意，但是有一个问题；我们的实现不允许我们指定重试次数，而是在装饰器内部是一个固定的数字。

现在，我们希望能够指示每个实例将具有多少次重试，也许我们甚至可以为此参数添加一个默认值。为了做到这一点，我们需要另一个级别的嵌套函数——首先是参数，然后是装饰器本身。

这是因为我们现在将有以下形式的东西：

```py
 @retry(arg1, arg2,... )
```

并且必须返回一个装饰器，因为`@`语法将该计算的结果应用于要装饰的对象。从语义上讲，它将转换为以下内容：

```py
  <original_function> = retry(arg1, arg2, ....)(<original_function>)
```

除了所需的重试次数，我们还可以指示我们希望控制的异常类型。支持新要求的代码的新版本可能如下所示：

```py
RETRIES_LIMIT = 3

def with_retry(retries_limit=RETRIES_LIMIT, allowed_exceptions=None):
    allowed_exceptions = allowed_exceptions or (ControlledException,)

    def retry(operation):

        @wraps(operation)
        def wrapped(*args, **kwargs):
            last_raised = None
            for _ in range(retries_limit):
                try:
                    return operation(*args, **kwargs)
                except allowed_exceptions as e:
                    logger.info("retrying %s due to %s", operation, e)
                    last_raised = e
            raise last_raised

        return wrapped

    return retry
```

以下是如何将此装饰器应用于函数的一些示例，显示它接受的不同选项：

```py
# decorator_parametrized_1.py
@with_retry()
def run_operation(task):
    return task.run()

@with_retry(retries_limit=5)
def run_with_custom_retries_limit(task):
    return task.run()

@with_retry(allowed_exceptions=(AttributeError,))
def run_with_custom_exceptions(task):
    return task.run()

@with_retry(
    retries_limit=4, allowed_exceptions=(ZeroDivisionError, AttributeError)
)
def run_with_custom_parameters(task):
    return task.run()
```

# 装饰器对象

前面的示例需要三个级别的嵌套函数。第一个将是一个接收我们想要使用的装饰器的参数的函数。在这个函数内部，其余的函数都是使用这些参数以及装饰器的逻辑的闭包。

更干净的实现方法是使用类来定义装饰器。在这种情况下，我们可以在`__init__`方法中传递参数，然后在名为`__call__`的魔术方法上实现装饰器的逻辑。

装饰器的代码看起来像以下示例中的样子：

```py
class WithRetry:

    def __init__(self, retries_limit=RETRIES_LIMIT, allowed_exceptions=None):
        self.retries_limit = retries_limit
        self.allowed_exceptions = allowed_exceptions or (ControlledException,)

    def __call__(self, operation):

        @wraps(operation)
        def wrapped(*args, **kwargs):
            last_raised = None

            for _ in range(self.retries_limit):
                try:
                    return operation(*args, **kwargs)
                except self.allowed_exceptions as e:
                    logger.info("retrying %s due to %s", operation, e)
                    last_raised = e
            raise last_raised

        return wrapped
```

这个装饰器可以应用得和之前的一个差不多，像这样：

```py
@WithRetry(retries_limit=5)
def run_with_custom_retries_limit(task):
    return task.run()
```

重要的是要注意 Python 语法在这里的作用。首先，我们创建对象，因此在应用`@`操作之前，对象已经被创建，并且其参数传递给它。这将创建一个新对象，并使用`init`方法中定义的这些参数进行初始化。之后，调用`@`操作，因此这个对象将包装名为`run_with_custom_reries_limit`的函数，这意味着它将被传递给`call`魔术方法。

在这个`call`魔术方法中，我们像往常一样定义了装饰器的逻辑-我们包装原始函数，返回一个具有我们想要的逻辑的新函数。

# 装饰器的好处

在本节中，我们将看一些常见的模式，这些模式充分利用了装饰器。这些都是装饰器是一个不错选择的常见情况。

从装饰器可以使用的无数应用中，我们将列举一些最常见或相关的：

+   **转换参数**：更改函数的签名以公开更好的 API，同时封装有关如何处理和转换参数的细节

+   **跟踪代码**：记录函数的执行及其参数

+   **验证参数**

+   **实现重试操作**

+   **通过将一些（重复的）逻辑移入装饰器来简化类**

让我们在下一节详细讨论前两个应用。

# 转换参数

我们之前提到过，装饰器可以用于验证参数（甚至在 DbC 的概念下强制执行一些前置条件或后置条件），因此您可能已经得到这样的想法，即在处理或操作参数时，使用装饰器是很常见的。

特别是，在某些情况下，我们发现自己反复创建类似的对象，或者应用类似的转换，我们希望将其抽象化。大多数情况下，我们可以通过简单地使用装饰器来实现这一点。

# 跟踪代码

在本节讨论“跟踪”时，我们将指的是处理我们希望监视的函数的执行的更一般的内容。这可能涉及到我们希望的一些情况：

+   实际上跟踪函数的执行（例如，通过记录它执行的行）

+   监视函数的一些指标（如 CPU 使用率或内存占用）

+   测量函数的运行时间

+   记录函数调用的时间和传递给它的参数

在下一节中，我们将探讨一个简单的例子，即记录函数的执行情况，包括其名称和运行所花费的时间的装饰器。

# 有效的装饰器-避免常见错误

虽然装饰器是 Python 的一个很棒的特性，但如果使用不当，它们也不免有问题。在本节中，我们将看到一些常见的问题，以避免创建有效的装饰器。

# 保留有关原始包装对象的数据

将装饰器应用于函数时最常见的问题之一是，原始函数的某些属性或属性未得到保留，导致不希望的、难以跟踪的副作用。

为了说明这一点，我们展示了一个负责记录函数即将运行时的装饰器：

```py
# decorator_wraps_1.py

def trace_decorator(function):
    def wrapped(*args, **kwargs):
        logger.info("running %s", function.__qualname__)
        return function(*args, **kwargs)

    return wrapped
```

现在，让我们想象一下，我们有一个应用了这个装饰器的函数。我们可能最初会认为该函数的任何部分都没有修改其原始定义：

```py
@trace_decorator
def process_account(account_id):
    """Process an account by Id."""
    logger.info("processing account %s", account_id)
    ...
```

但也许有一些变化。

装饰器不应该改变原始函数的任何内容，但事实证明，由于它包含一个缺陷，它实际上修改了其名称和`docstring`等属性。

让我们尝试为这个函数获取`help`：

```py
>>> help(process_account)
Help on function wrapped in module decorator_wraps_1:

wrapped(*args, **kwargs) 
```

让我们检查它是如何被调用的：

```py
>>> process_account.__qualname__
'trace_decorator.<locals>.wrapped'
```

我们可以看到，由于装饰器实际上是将原始函数更改为一个新函数（称为`wrapped`），我们实际上看到的是这个函数的属性，而不是原始函数的属性。

如果我们将这样一个装饰器应用于多个函数，它们都有不同的名称，它们最终都将被称为`wrapped`，这是一个主要问题（例如，如果我们想要记录或跟踪函数，这将使调试变得更加困难）。

另一个问题是，如果我们在这些函数上放置了带有测试的文档字符串，它们将被装饰器的文档字符串覆盖。结果，我们希望的带有测试的文档字符串在我们使用`doctest`模块调用我们的代码时将不会运行（正如我们在第一章中所看到的，*介绍、代码格式和工具*）。

修复很简单。我们只需在内部函数（`wrapped`）中应用`wraps`装饰器，告诉它实际上是在包装`function`：

```py
# decorator_wraps_2.py
def trace_decorator(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        logger.info("running %s", function.__qualname__)
        return function(*args, **kwargs)

    return wrapped
```

现在，如果我们检查属性，我们将得到我们最初期望的结果。像这样检查函数的`help`：

```py
>>> Help on function process_account in module decorator_wraps_2:

process_account(account_id)
    Process an account by Id. 
```

并验证其合格的名称是否正确，如下所示：

```py
>>> process_account.__qualname__
'process_account'
```

最重要的是，我们恢复了可能存在于文档字符串中的单元测试！通过使用`wraps`装饰器，我们还可以在`__wrapped__`属性下访问原始的未修改的函数。虽然不应该在生产中使用，但在一些单元测试中，当我们想要检查函数的未修改版本时，它可能会派上用场。

通常，对于简单的装饰器，我们使用`functools.wraps`的方式通常遵循以下一般公式或结构：

```py
def decorator(original_function):
    @wraps(original_function)
    def decorated_function(*args, **kwargs):
        # modifications done by the decorator ...
        return original_function(*args, **kwargs)

    return decorated_function
```

在创建装饰器时，通常对包装的函数应用`functools.wraps`，如前面的公式所示。

# 处理装饰器中的副作用

在本节中，我们将了解在装饰器的主体中避免副作用是明智的。有些情况下可能是可以接受的，但最重要的是，如果有疑问，最好不要这样做，原因将在后面解释。

尽管如此，有时这些副作用是必需的（甚至是期望的）在导入时运行，反之亦然。

我们将看到两者的示例，以及每种情况的适用情况。如果有疑问，最好谨慎行事，并将所有副作用延迟到最后，就在`wrapped`函数将被调用之后。

接下来，我们将看到在`wrapped`函数之外放置额外逻辑不是一个好主意的情况。

# 装饰器中副作用的处理不正确

让我们想象一个创建目的是在函数开始运行时记录日志，然后记录其运行时间的装饰器的情况：

```py
def traced_function_wrong(function):
    logger.info("started execution of %s", function)
    start_time = time.time()

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        logger.info(
            "function %s took %.2fs",
            function,
            time.time() - start_time
        )
        return result
    return wrapped
```

现在，我们将装饰器应用到一个常规函数上，认为它会正常工作：

```py
@traced_function_wrong
def process_with_delay(callback, delay=0):
    time.sleep(delay)
    return callback()
```

这个装饰器有一个微妙但关键的错误。

首先，让我们导入函数，多次调用它，看看会发生什么：

```py
>>> from decorator_side_effects_1 import process_with_delay
INFO:started execution of <function process_with_delay at 0x...>
```

通过导入函数，我们会注意到有些地方不对劲。日志行不应该出现在那里，因为函数没有被调用。

现在，如果我们运行函数，看看运行需要多长时间？实际上，我们期望多次调用相同的函数会得到类似的结果：

```py
>>> main()
...
INFO:function <function process_with_delay at 0x> took 8.67s

>>> main()
...
INFO:function <function process_with_delay at 0x> took 13.39s

>>> main()
...
INFO:function <function process_with_delay at 0x> took 17.01s
```

每次运行相同的函数，都会花费更长的时间！此时，您可能已经注意到（现在显而易见的）错误。

除了装饰的函数之外，装饰器需要做的一切都应该放在最内部的函数定义中，否则在导入时会出现问题。

```py
process_with_delay = traced_function_wrong(process_with_delay)
```

这将在模块导入时运行。因此，函数中设置的时间将是模块导入时的时间。连续调用将计算从运行时间到原始开始时间的时间差。它还将在错误的时刻记录，而不是在实际调用函数时。

幸运的是，修复也很简单——我们只需将代码移到`wrapped`函数内部以延迟其执行：

```py
def traced_function(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        logger.info("started execution of %s", function.__qualname__)
        start_time = time.time()
        result = function(*args, **kwargs)
        logger.info(
            "function %s took %.2fs",
            function.__qualname__,
            time.time() - start_time
        )
        return result
    return wrapped
```

记住装饰器的语法。`@traced_function_wrong`实际上意味着以下内容：

如果装饰器的操作不同，结果可能会更加灾难性。例如，如果它要求您记录事件并将其发送到外部服务，除非在导入此模块之前正确运行了配置，否则肯定会失败，而这是我们无法保证的。即使我们可以，这也是不好的做法。如果装饰器具有其他任何形式的副作用，例如从文件中读取、解析配置等，也是一样。

# 需要具有副作用的装饰器

有时，装饰器上的副作用是必要的，我们不应该延迟它们的执行直到最后可能的时间，因为这是它们工作所需的机制的一部分。

当我们不想延迟装饰器的副作用时，一个常见的情况是，我们需要将对象注册到一个将在模块中可用的公共注册表中。

例如，回到我们之前的`event`系统示例，现在我们只想在模块中使一些事件可用，而不是所有事件。在事件的层次结构中，我们可能希望有一些中间类，它们不是我们想要在系统上处理的实际事件，而是它们的一些派生类。

我们可以通过装饰器显式注册每个类，而不是根据它是否要被处理来标记每个类。

在这种情况下，我们有一个与用户活动相关的所有事件的类。然而，这只是我们实际想要的事件类型的中间表，即`UserLoginEvent`和`UserLogoutEvent`：

```py
EVENTS_REGISTRY = {}

def register_event(event_cls):
    """Place the class for the event into the registry to make it 
    accessible in
    the module.
    """
    EVENTS_REGISTRY[event_cls.__name__] = event_cls
    return event_cls

class Event:
    """A base event object"""

class UserEvent:
    TYPE = "user"

@register_event
class UserLoginEvent(UserEvent):
    """Represents the event of a user when it has just accessed the system."""

@register_event
class UserLogoutEvent(UserEvent):
    """Event triggered right after a user abandoned the system."""
```

当我们查看前面的代码时，似乎`EVENTS_REGISTRY`是空的，但在从这个模块导入一些内容之后，它将被填充为所有在`register_event`装饰器下的类。

```py
>>> from decorator_side_effects_2 import EVENTS_REGISTRY
>>> EVENTS_REGISTRY
{'UserLoginEvent': decorator_side_effects_2.UserLoginEvent,
 'UserLogoutEvent': decorator_side_effects_2.UserLogoutEvent}
```

这可能看起来很难阅读，甚至具有误导性，因为`EVENTS_REGISTRY`将在运行时具有其最终值，就在模块导入后，我们无法仅通过查看代码来轻松预测其值。

虽然在某些情况下这种模式是合理的。事实上，许多 Web 框架或知名库使用这种模式来工作和公开对象或使它们可用。

在这种情况下，装饰器并没有改变`wrapped`对象，也没有以任何方式改变它的工作方式。然而，这里需要注意的是，如果我们进行一些修改并定义一个修改`wrapped`对象的内部函数，我们可能仍然希望在外部注册生成的对象的代码。

注意使用*outside*这个词。它不一定意味着之前，它只是不属于同一个闭包；但它在外部范围，因此不会延迟到运行时。

# 创建始终有效的装饰器

装饰器可能适用于几种不同的情况。也可能出现这样的情况，我们需要对落入这些不同多种情况的对象使用相同的装饰器，例如，如果我们想重用我们的装饰器并将其应用于函数、类、方法或静态方法。

如果我们创建装饰器，只考虑支持我们想要装饰的第一种对象类型，我们可能会注意到相同的装饰器在不同类型的对象上效果不同。典型的例子是，我们创建一个用于函数的装饰器，然后想将其应用于类的方法，结果发现它不起作用。如果我们为方法设计了装饰器，然后希望它也适用于静态方法或类方法，可能会发生类似的情况。

在设计装饰器时，我们通常考虑重用代码，因此我们也希望将该装饰器用于函数和方法。

使用`*args`和`**kwargs`签名定义我们的装饰器将使它们在所有情况下都起作用，因为这是我们可以拥有的最通用的签名。然而，有时我们可能不想使用这个，而是根据原始函数的签名定义装饰器包装函数，主要是因为两个原因：

+   它将更易读，因为它类似于原始函数。

+   它实际上需要对参数进行一些处理，因此接收`*args`和`**kwargs`将不方便。

考虑我们的代码库中有许多函数需要从参数创建特定对象的情况。例如，我们传递一个字符串，并重复使用它初始化一个驱动程序对象。然后我们认为可以通过使用一个装饰器来消除这种重复。

在下一个例子中，我们假设`DBDriver`是一个知道如何连接和在数据库上运行操作的对象，但它需要一个连接字符串。我们在我们的代码中有的方法，都设计为接收包含数据库信息的字符串，并且总是需要创建一个`DBDriver`实例。装饰器的想法是它将自动进行这种转换——函数将继续接收一个字符串，但装饰器将创建一个`DBDriver`并将其传递给函数，因此在内部我们可以假设我们直接接收到了我们需要的对象。

在下一个清单中展示了在函数中使用这个的例子：

```py
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class DBDriver:
    def __init__(self, dbstring):
        self.dbstring = dbstring

    def execute(self, query):
        return f"query {query} at {self.dbstring}"

def inject_db_driver(function):
    """This decorator converts the parameter by creating a ``DBDriver``
    instance from the database dsn string.
    """
    @wraps(function)
    def wrapped(dbstring):
        return function(DBDriver(dbstring))
    return wrapped

@inject_db_driver
def run_query(driver):
    return driver.execute("test_function")
```

很容易验证，如果我们将一个字符串传递给函数，我们会得到一个`DBDriver`实例完成的结果，所以装饰器的工作是符合预期的：

```py
>>> run_query("test_OK")
'query test_function at test_OK'
```

但现在，我们想在类方法中重用这个相同的装饰器，我们发现了同样的问题：

```py
class DataHandler:
    @inject_db_driver
    def run_query(self, driver):
        return driver.execute(self.__class__.__name__)
```

我们尝试使用这个装饰器，只是意识到它不起作用：

```py
>>> DataHandler().run_query("test_fails")
Traceback (most recent call last):
 ...
TypeError: wrapped() takes 1 positional argument but 2 were given
```

问题是什么？

类中的方法是用额外的参数`self`定义的。

方法只是一种特殊类型的函数，它接收`self`（它们所定义的对象）作为第一个参数。

因此，在这种情况下，装饰器（设计为仅适用于名为`dbstring`的参数）将解释`self`是所说的参数，并调用该方法传递字符串作为 self 的位置，以及在第二个参数的位置上什么都不传，即我们正在传递的字符串。

为了解决这个问题，我们需要创建一个装饰器，它可以同时适用于方法和函数，我们通过将其定义为一个装饰器对象来实现这一点，该对象还实现了协议描述符。

描述符在第七章中有详细解释，*使用生成器*，所以，现在，我们可以将其视为一个可以使装饰器工作的配方。

解决方案是将装饰器实现为一个类对象，并使该对象成为一个描述符，通过实现`__get__`方法。

```py
from functools import wraps
from types import MethodType

class inject_db_driver:
    """Convert a string to a DBDriver instance and pass this to the 
       wrapped function."""

    def __init__(self, function):
        self.function = function
        wraps(self.function)(self)

    def __call__(self, dbstring):
        return self.function(DBDriver(dbstring))

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.__class__(MethodType(self.function, instance))
```

描述符的详细信息将在第六章中解释，*使用描述符更充分地利用我们的对象*，但是对于这个例子的目的，我们现在可以说它实际上是将它装饰的可调用对象重新绑定到一个方法，这意味着它将函数绑定到对象，然后使用这个新的可调用对象重新创建装饰器。

对于函数，它仍然有效，因为它根本不会调用`__get__`方法。

# 装饰器与 DRY 原则

我们已经看到装饰器如何允许我们将某些逻辑抽象成一个单独的组件。这样做的主要优势是我们可以多次应用装饰器到不同的对象中，以便重用代码。这遵循了**不要重复自己**（**DRY**）原则，因为我们只定义了某些知识一次。

在前面的部分中实现的“重试”机制是一个很好的例子，它是一个可以多次应用以重用代码的装饰器。我们不是让每个特定的函数包含其“重试”逻辑，而是创建一个装饰器并多次应用它。一旦我们确保装饰器可以同样适用于方法和函数，这就是有意义的。

定义了事件如何表示的类装饰器也符合 DRY 原则，因为它定义了一个特定的位置来序列化事件的逻辑，而无需在不同的类中重复代码。由于我们希望重用这个装饰器并将其应用于许多类，它的开发（和复杂性）是值得的。

当尝试使用装饰器来重用代码时，这最后一点很重要——我们必须绝对确定我们实际上将节省代码。

任何装饰器（特别是如果设计不慎）都会给代码增加另一层间接性，因此会增加更多的复杂性。代码的读者可能希望跟踪装饰器的路径以充分理解函数的逻辑（尽管这些考虑在下一节中有所解决），因此请记住这种复杂性必须得到回报。如果不会有太多的重用，那么不要选择装饰器，而选择一个更简单的选项（也许只是一个单独的函数或另一个小类就足够了）。

但我们如何知道太多的重用是什么？有没有规则来确定何时将现有代码重构为装饰器？在 Python 中，没有特定于装饰器的规则，但我们可以应用软件工程中的一个经验法则（GLASS 01），该法则规定在考虑创建可重用组件之前，应该至少尝试三次使用组件。从同一参考资料（GLASS 01）中还得出了一个观点，即创建可重用组件比创建简单组件困难三倍。

底线是，通过装饰器重用代码是可以接受的，但只有在考虑以下几点时才可以：

+   不要从头开始创建装饰器。等到模式出现并且装饰器的抽象变得清晰时再进行重构。

+   考虑到装饰器必须被应用多次（至少三次）才能实施。

+   将装饰器中的代码保持在最小限度。

# 装饰器和关注点分离

前面列表中的最后一点非常重要，值得单独一节来讨论。我们已经探讨了重用代码的想法，并注意到重用代码的一个关键元素是具有内聚性的组件。这意味着它们应该具有最小的责任水平——只做一件事，只做一件事，并且做得很好。我们的组件越小，就越可重用，也越能在不同的上下文中应用，而不会带有额外的行为，这将导致耦合和依赖，使软件变得僵化。

为了向您展示这意味着什么，让我们回顾一下我们在先前示例中使用的装饰器之一。我们创建了一个装饰器，用类似以下代码的方式跟踪了某些函数的执行：

```py
def traced_function(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        logger.info("started execution of %s", function.__qualname__)
        start_time = time.time()
        result = function(*args, **kwargs)
        logger.info(
            "function %s took %.2fs",
            function.__qualname__,
            time.time() - start_time
        )
        return result
    return wrapped
```

现在，这个装饰器虽然有效，但存在一个问题——它做了不止一件事。它记录了特定函数的调用，并记录了运行所花费的时间。每次使用这个装饰器，我们都要承担这两个责任，即使我们只想要其中一个。

这应该被分解成更小的装饰器，每个装饰器都有更具体和有限的责任：

```py
def log_execution(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        logger.info("started execution of %s", function.__qualname__)
        return function(*kwargs, **kwargs)
    return wrapped

def measure_time(function):
 @wraps(function)
 def wrapped(*args, **kwargs):
 start_time = time.time()
 result = function(*args, **kwargs)

 logger.info("function %s took %.2f", function.__qualname__,
 time.time() - start_time)
 return result
 return wrapped
```

请注意，我们之前所拥有的相同功能可以通过简单地将它们结合起来来实现：

```py
@measure_time
@log_execution
def operation():
    ....
```

注意装饰器的应用顺序也很重要。

不要在一个装饰器中放置多个责任。单一责任原则也适用于装饰器。

# 分析好的装饰器

作为本章的结束语，让我们回顾一些好的装饰器的示例以及它们在 Python 本身以及流行库中的用法。这个想法是获得如何创建好的装饰器的指导方针。

在跳入示例之前，让我们首先确定好的装饰器应该具有的特征：

+   **封装，或关注点分离**：一个好的装饰器应该有效地将它所做的事情和它所装饰的事物之间的不同责任分开。它不能是一个有漏洞的抽象，这意味着装饰器的客户端应该只以黑盒模式调用它，而不知道它实际上是如何实现其逻辑的。

+   **正交性**：装饰器所做的事情应该是独立的，并且尽可能与它所装饰的对象解耦。

+   **可重用性**：希望装饰器可以应用于多种类型，而不仅仅出现在一个函数的一个实例上，因为这意味着它本来可以只是一个函数。它必须足够通用。

装饰器的一个很好的例子可以在 Celery 项目中找到，其中通过将应用程序的`task`装饰器应用到一个函数来定义`task`：

```py
@app.task
def mytask():
   ....
```

这是一个好的装饰器的原因之一是因为它在封装方面非常出色。库的用户只需要定义函数体，装饰器就会自动将其转换为一个任务。`"@app.task"`装饰器肯定包含了大量的逻辑和代码，但这些对`"mytask()"`的主体来说都不相关。这是完全的封装和关注点分离——没有人需要查看装饰器在做什么，因此它是一个不泄漏任何细节的正确抽象。

装饰器的另一个常见用法是在 Web 框架（例如 Pyramid，Flask 和 Sanic 等）中，通过装饰器将视图的处理程序注册到 URL：

```py
@route("/", method=["GET"])
def view_handler(request):
 ...
```

这些类型的装饰器与之前的考虑相同；它们也提供了完全的封装，因为 Web 框架的用户很少（如果有的话）需要知道`"@route"`装饰器在做什么。在这种情况下，我们知道装饰器正在做更多的事情，比如将这些函数注册到 URL 的映射器上，并且它还改变了原始函数的签名，以便为我们提供一个更好的接口，接收一个已经设置好所有信息的请求对象。

前面的两个例子足以让我们注意到关于装饰器的这种用法的另一点。它们符合 API。这些库或框架通过装饰器向用户公开其功能，结果表明装饰器是定义清晰的编程接口的绝佳方式。

这可能是我们应该考虑装饰器的最佳方式。就像在告诉我们事件属性将如何被处理的类装饰器的示例中一样，一个好的装饰器应该提供一个清晰的接口，以便代码的用户知道可以从装饰器中期望什么，而不需要知道它是如何工作的，或者它的任何细节。

# 总结

装饰器是 Python 中强大的工具，可以应用于许多事物，如类、方法、函数、生成器等。我们已经演示了如何以不同的方式创建装饰器，以及不同的目的，并在这个过程中得出了一些结论。

在为函数创建装饰器时，尝试使其签名与被装饰的原始函数匹配。与使用通用的`*args`和`**kwargs`不同，使签名与原始函数匹配将使其更容易阅读和维护，并且它将更接近原始函数，因此对于代码的读者来说更加熟悉。

装饰器是重用代码和遵循 DRY 原则的非常有用的工具。然而，它们的有用性是有代价的，如果不明智地使用，复杂性可能会带来更多的害处。因此，我们强调装饰器应该在实际上会被多次应用（三次或更多次）时使用。与 DRY 原则一样，我们发现关注点分离的想法，目标是尽可能保持装饰器的小巧。

另一个很好的装饰器用法是创建更清晰的接口，例如，通过将类的一部分逻辑提取到装饰器中来简化类的定义。在这个意义上，装饰器还通过提供关于特定组件将要做什么的信息来帮助可读性，而不需要知道如何做（封装）。

在下一章中，我们将看看 Python 的另一个高级特性——描述符。特别是，我们将看到如何借助描述符创建更好的装饰器，并解决本章遇到的一些问题。

# 参考资料

以下是您可以参考的信息列表：

+   *PEP-318*：函数和方法的装饰器（[`www.python.org/dev/peps/pep-0318/`](https://www.python.org/dev/peps/pep-0318/)）

+   *PEP-3129*：类装饰器（[`www.python.org/dev/peps/pep-3129/`](https://www.python.org/dev/peps/pep-3129/)）

+   *WRAPT 01*：[`pypi.org/project/wrapt/`](https://pypi.org/project/wrapt/)

+   *WRAPT 02*：[`wrapt.readthedocs.io/en/latest/decorators.html#universal-decorators`](https://wrapt.readthedocs.io/en/latest/decorators.html#universal-decorators)

+   *Functools 模块*：Python 标准库中`functools`模块中的`wraps`函数（[`docs.python.org/3/library/functools.html#functools.wrap`](https://docs.python.org/3/library/functools.html#functools.wraps)）

+   *ATTRS 01*：`attrs`库（[`pypi.org/project/attrs/`](https://pypi.org/project/attrs/)）

+   *PEP-557*：数据类（[`www.python.org/dev/peps/pep-0557/`](https://www.python.org/dev/peps/pep-0557/)）

+   *GLASS 01*：Robert L. Glass 撰写的书籍*软件工程的事实和谬误*
