

# 第五章：使用装饰器改进我们的代码

在本章中，我们将探讨装饰器，并了解它们在许多我们想要改进设计的情况中的有用之处。我们将首先探索装饰器是什么，它们是如何工作的，以及它们是如何实现的。

带着这些知识，我们将回顾我们在前几章中学到的关于软件设计的一般良好实践的概念，并看看装饰器如何帮助我们遵守每个原则。

本章的目标如下：

+   理解装饰器在 Python 中的工作方式

+   学习如何实现适用于函数和类的装饰器

+   为了有效地实现装饰器，避免常见的实现错误

+   分析如何使用装饰器避免代码重复（DRY 原则）

+   研究装饰器如何有助于关注点的分离

+   为了分析好的装饰器的例子

+   为了回顾当装饰器是正确选择时的常见情况、习语或模式

# Python 中的装饰器是什么？

装饰器是在 Python 中很久以前引入的，在 PEP-318 中，作为一种在函数和方法的原始定义之后需要修改时的简化定义方式的机制。

我们首先必须理解，在 Python 中，函数就像几乎所有其他东西一样是常规对象。这意味着你可以将它们赋给变量，通过参数传递它们，甚至将其他函数应用于它们。通常，人们会想写一个小函数，然后对其应用一些转换，生成该函数的新修改版本（类似于数学中函数组合的工作方式）。

引入装饰器的原始动机之一是因为像`classmethod`和`staticmethod`这样的函数用于转换方法的原始定义，它们需要一个额外的行，在单独的语句中修改函数的原始定义。

更普遍地说，每次我们需要对一个函数应用一个转换时，我们必须使用`modifier`函数来调用它，然后将其重新赋值回原来定义该函数的同一名称。

例如，如果我们有一个名为`original`的函数，然后我们有一个在它之上改变`original`行为的函数，称为`modifier`，我们必须编写如下内容：

```py
def original(...):
    ...
original = modifier(original) 
```

注意我们是如何更改函数并将其重新赋值回同一名称的。这很令人困惑，容易出错（想象一下有人忘记重新赋值函数，或者虽然重新赋值了，但不是在函数定义后的下一行，而是在更远的地方），而且很麻烦。因此，语言增加了一些语法支持。

之前的例子可以重写如下：

```py
@modifier
def original(...):
   ... 
```

这意味着装饰器只是将装饰器之后的内容作为装饰器本身的第一个参数调用的语法糖，结果将是装饰器返回的内容。

装饰器的语法大大提高了代码的可读性，因为现在代码的读者可以在一个地方找到函数的整个定义。请记住，手动修改函数如以前一样仍然是允许的。

通常情况下，尽量避免在不使用装饰器语法的情况下重新分配已经设计好的函数的值。特别是，如果函数被重新分配为其他内容，并且这种情况发生在代码的远程部分（远离函数最初定义的地方），这将使你的代码更难以阅读。

根据 Python 术语和我们的示例，`modifier`是我们所说的**装饰器**，而`original`是被装饰的函数，通常也称为**包装**对象。

虽然最初的功能是为方法和函数考虑的，但实际的语法允许任何类型的对象被装饰，因此我们将探讨应用于函数、方法、生成器和类的装饰器。

最后一点需要注意的是，虽然装饰器的名字是正确的（毕竟，装饰器正在对包装函数进行更改、扩展或在其之上工作），但它不应与装饰器设计模式混淆。

## 函数装饰器

函数可能是 Python 中可以装饰的最简单对象表示。我们可以使用装饰器在函数上应用各种逻辑——我们可以验证参数，检查先决条件，完全改变其行为，修改其签名，缓存结果（创建原始函数的缓存版本），等等。

作为例子，我们将创建一个基本的装饰器，实现一个`retry`机制，控制特定的域级别异常，并尝试一定次数：

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

目前可以忽略`@wraps`的使用，因为它将在*有效的装饰器 - 避免常见错误*部分进行讲解。

在`for`循环中使用`_`表示该数字被分配给一个我们目前不感兴趣的变量，因为它在`for`循环内没有使用（在 Python 中，命名被忽略的`_`值是一个常见的习惯用法）。

`retry`装饰器不接受任何参数，因此它可以很容易地应用于任何函数，如下所示：

```py
@retry
def run_operation(task):
    """Run a particular task, simulating some failures on its execution."""
    return task.run() 
```

在`run_operation`上定义`@retry`只是 Python 提供的语法糖，用于执行`run_operation = retry(run_operation)`。

在这个有限的示例中，我们可以看到装饰器如何被用来创建一个通用的`retry`操作，在特定条件下（在这种情况下，表示可能相关的超时异常），将允许多次调用被装饰的代码。

## 类装饰器

在 Python 中，类也是对象（坦白说，在 Python 中几乎一切都是对象，很难找到反例；然而，有一些技术上的细微差别）。这意味着相同的考虑也适用；它们也可以通过参数传递，分配给变量，调用某些方法，或者被转换（装饰）。

类装饰器是在 PEP-3129 中引入的，并且它们与我们已经探索过的函数装饰器有非常相似的考虑。唯一的区别是，在编写这种装饰器的代码时，我们必须考虑到我们正在接收一个类作为包装方法的参数，而不是另一个函数。

当我们在“第二章”，“Pythonic 代码”中看到`dataclasses.dataclass`装饰器时，我们看到了如何使用类装饰器。在本章中，我们将学习如何编写我们自己的类装饰器。

一些从业者可能会认为装饰类是一种相当复杂的事情，并且这种场景可能会危及可读性，因为我们在类中声明了一些属性和方法，但幕后，装饰器可能正在应用一些会使其成为完全不同类的更改。

这种评估是正确的，但只有当这种技术被过度使用时。客观上，这与装饰函数没有区别；毕竟，类只是 Python 生态系统中的另一种类型对象，就像函数一样。我们将在标题为“装饰器和关注点分离”的章节中回顾使用装饰器的利弊，但现在，我们将探讨装饰器对类特别有益的益处：

+   代码重用和 DRY 原则的所有好处。一个有效的类装饰器用例是强制多个类遵守某个接口或标准（通过在将被应用于许多类的装饰器中只写一次这些检查）。

+   我们可以创建更小或更简单的类，这些类可以通过装饰器在以后进行增强。

+   如果我们使用装饰器而不是更复杂（并且通常被正确劝阻）的方法，如元类，那么我们需要应用于特定类的转换逻辑将更容易维护。

在所有可能的装饰器应用中，我们将探索一个简单的例子来展示它们可能有用的情况。请记住，这并不是类装饰器的唯一应用类型，而且我向你展示的代码也可以有其他许多解决方案，所有这些解决方案都有其优缺点，但我选择装饰器是为了说明它们的有用性。

回顾我们的监控平台的事件系统，我们现在需要转换每个事件的 数据并将其发送到外部系统。然而，每种类型的事件在选择如何发送其数据时可能都有自己的特性。

特别是，登录事件的`event`可能包含敏感信息，例如我们想要隐藏的凭证。其他字段，如`timestamp`，也可能需要一些转换，因为我们希望以特定的格式显示它们。满足这些要求的一个初步尝试就是拥有一个映射到每个特定事件并知道如何序列化它的类：

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
@dataclass
class LoginEvent:
    SERIALIZER = LoginEventSerializer
    username: str
    password: str
    ip: str
    timestamp: datetime
    def serialize(self) -> dict:
        return self.SERIALIZER(self).serialize() 
```

在这里，我们声明一个类，它将直接与登录事件映射，包含其逻辑——隐藏`password`字段，并按要求格式化`timestamp`。

虽然这可行，并且可能看起来是一个好的起点，但随着时间的推移，当我们想要扩展我们的系统时，我们会发现一些问题：

+   **类太多**：随着事件数量的增长，序列化类的数量将以相同的数量级增长，因为它们是一对一映射的。

+   **解决方案不够灵活**：如果我们需要重用组件的部分（例如，我们需要在另一个类型的事件中隐藏密码，该事件也有密码），我们必须将其提取到一个函数中，但还需要从多个类中重复调用它，这意味着我们最终并没有重用很多代码。

+   **模板代码**：`serialize()`方法将必须存在于所有事件类中，调用相同的代码。虽然我们可以将其提取到另一个类中（创建一个混入类），但这似乎并不是继承的好用法。

另一种解决方案是动态构建一个对象，给定一组过滤器（转换函数）和事件实例，可以通过应用过滤器到其字段来序列化它。然后我们只需要定义转换每种字段类型的函数，序列化器通过组合这些函数中的许多来创建。

一旦我们有了这个对象，我们就可以装饰类来添加`serialize()`方法，这个方法将只调用这些`Serialization`对象本身：

```py
from dataclasses import dataclass
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
            for field, transformation
            in self.serialization_fields.items()
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
    username=str.lower, 
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

注意装饰器如何让用户更容易知道每个字段将被如何处理，而无需查看另一个类的代码。只需阅读传递给类装饰器的参数，我们就可以知道`username`和 IP 地址将保持不变，`password`将被隐藏，而`timestamp`将被格式化。

现在，类的代码不需要定义`serialize()`方法，也不需要从实现它的混入类扩展，因为装饰器会添加它。这可能是唯一可以证明创建类装饰器合理性的部分，因为否则`Serialization`对象可以是`LoginEvent`的类属性，但它是通过向其添加新方法来改变类的，这使得它变得不可能。

## 其他类型的装饰器

既然我们已经知道了装饰器的`@`语法意味着什么，我们可以得出结论，不仅仅是函数、方法或类可以被装饰；实际上，任何可以被定义的东西，比如生成器、协程，甚至已经装饰过的对象，都可以被装饰，这意味着装饰器可以堆叠。

之前的例子展示了装饰器可以如何链式使用。我们首先定义了类，然后应用`@dataclass`到它上面，这将其转换成了一个数据类，作为一个容器的属性。之后，`@Serialization`将对那个类应用逻辑，结果产生一个新的类，其中添加了新的`serialize()`方法。

现在我们已经了解了装饰器的基础知识以及如何编写它们，我们可以继续到更复杂的例子。在接下来的几节中，我们将看到如何拥有具有参数的更灵活的装饰器以及实现它们的不同方法。

# 更高级的装饰器

通过我们刚刚的介绍，我们现在已经了解了装饰器的基础知识：它们是什么，以及它们的语法和语义。

现在，我们感兴趣的是更高级的装饰器用法，这将帮助我们更干净地组织代码。

我们将看到我们可以使用装饰器将关注点分离成更小的函数，并重用代码，但为了有效地这样做，我们希望对装饰器进行参数化（否则，我们最终会重复代码）。为此，我们将探讨如何将参数传递给装饰器的不同选项。

之后，我们可以看到一些装饰器良好用法的例子。

## 将参数传递给装饰器

到目前为止，我们已经将装饰器视为 Python 中的一个强大工具。然而，如果我们能够向它们传递参数，以便它们的逻辑被进一步抽象化，它们可能会更加强大。

有几种实现可以接受参数的装饰器的方法，但我们将介绍最常见的一些。第一种方法是将装饰器作为嵌套函数创建，增加一个间接层，使得装饰器中的所有内容都深入一个层次。第二种方法是使用一个类作为装饰器（即实现一个仍然充当装饰器的可调用对象）。

通常，第二种方法在可读性方面更有优势，因为相对于三个或更多使用闭包工作的嵌套函数，更容易从对象的角度思考。然而，为了完整性，我们将探讨两种方法，你可以决定哪种最适合当前的问题。

### 带嵌套函数的装饰器

大体来说，装饰器的一般思想是创建一个返回另一个函数的函数（在函数式编程中，接受其他函数作为参数的函数被称为高阶函数，这指的是我们在这里讨论的相同概念）。装饰器体内定义的内部函数将是将被调用的函数。

现在，如果我们希望向它传递参数，我们则需要另一个间接层。第一个函数将接受参数，在该函数内部，我们将定义一个新的函数，这个新的函数将是装饰器，它反过来将定义另一个新的函数，即装饰过程的结果要返回的函数。这意味着我们将至少有三个嵌套函数的层次。

如果到目前为止这还不清楚，请不要担心。在查看即将到来的示例之后，一切都会变得清晰。

我们看到的第一批装饰器示例之一是在某些函数上实现了重试功能。这是一个好主意，但有一个问题；我们的实现不允许我们指定重试次数，而是在装饰器内部有一个固定的数字。

现在，我们希望能够表明每个实例将要尝试的次数，也许我们甚至可以为此参数添加一个默认值。为了做到这一点，我们需要另一层嵌套函数——首先是参数，然后是装饰器本身。

这是因为我们现在将会有以下形式的内容：

```py
@retry(arg1, arg2,... ) 
```

而这必须返回一个装饰器，因为`@`语法将计算结果应用于要装饰的对象。从语义上讲，它将翻译成以下类似的内容：

```py
 <original_function> = retry(arg1, arg2, ....)(<original_function>) 
```

除了想要的尝试次数之外，我们还可以指出我们希望控制的异常类型。支持新要求的代码的新版本可能看起来像这样：

```py
_DEFAULT_RETRIES_LIMIT = 3
    def with_retry(
        retries_limit: int = _DEFAULT_RETRIES_LIMIT,
        allowed_exceptions: Optional[Sequence[Exception]] = None,
    ):
        allowed_exceptions = allowed_exceptions or
        (ControlledException,) # type: ignore
        def retry(operation):
            @wraps(operation)
            def wrapped(*args, **kwargs):
                last_raised = None
                for _ in range(retries_limit):
                    try:
                        return operation(*args, **kwargs)
                    except allowed_exceptions as e:
                        logger.warning(
                            "retrying %s due to %s",
                            operation.__qualname__, e
                        )
                        last_raised = e
                raise last_raised
            return wrapped
        return retry 
```

这里有一些如何将这个装饰器应用于函数的示例，展示了它接受的不同选项：

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

使用嵌套函数来实现装饰器可能是我们首先想到的事情。这在大多数情况下都很好用，但正如你可能已经注意到的，每创建一个新的函数，缩进就会增加，所以很快可能会导致嵌套函数过多。此外，函数是无状态的，所以以这种方式编写的装饰器不一定能保留内部数据，就像对象可以做到的那样。

实现装饰器还有另一种方式，它不是使用嵌套函数，而是使用对象，我们将在下一节中探讨。

### 装饰器对象

之前的例子需要三个级别的嵌套函数。第一个是一个接收我们想要使用的装饰器参数的函数。在这个函数内部，其余的函数都是闭包，它们使用这些参数，以及装饰器的逻辑。

一种更干净的实现方法是使用类来定义装饰器。在这种情况下，我们可以在`__init__`方法中传递参数，然后在实际的`__call__`魔法方法上实现装饰器的逻辑。

装饰器的代码看起来就像以下示例中的那样：

```py
_DEFAULT_RETRIES_LIMIT = 3
class WithRetry:
    def __init__(
        self,
        retries_limit: int = _DEFAULT_RETRIES_LIMIT,
        allowed_exceptions: Optional[Sequence[Exception]] = None,
    ) -> None:
    self.retries_limit = retries_limit
    self.allowed_exceptions = allowed_exceptions or
(ControlledException,)
    def __call__(self, operation):
        @wraps(operation)
        def wrapped(*args, **kwargs):
            last_raised = None
            for _ in range(self.retries_limit):
                try:
                    return operation(*args, **kwargs)
                except self.allowed_exceptions as e:
                logger.warning(
                    "retrying %s due to %s",
                    operation.__qualname__, e
                )
                    last_raised = e
            raise last_raised
      return wrapped 
```

这个装饰器可以像之前的那个一样应用，如下所示：

```py
@WithRetry(retries_limit=5)
def run_with_custom_retries_limit(task):
    return task.run() 
```

重要的是要注意 Python 语法在这里是如何起作用的。首先，我们创建对象，因此在`@`操作应用之前，对象就已经被创建，并且将参数传递给它。这将创建一个新的对象，并使用`init`方法中的这些参数对其进行初始化。之后，调用`@`操作，这意味着这个对象将包装名为`run_with_custom_retries_limit`的函数，意味着它将被传递给`call`魔法方法。

在这个调用魔法方法中，我们像平常一样定义了装饰器的逻辑——我们包装了原始函数，返回了一个具有我们想要逻辑的新函数。

## 带有默认值的装饰器

在上一个例子中，我们看到了一个接受参数的装饰器，但那些参数有默认值。之前装饰器的写法确保了只要用户在使用装饰器时没有忘记在函数调用时使用括号，它们就能正常工作。

例如，如果我们只想使用默认值，这将有效：

```py
@retry()
def my function(): ... 
```

但这不会：

```py
@retry
def my function(): ... 
```

你可能会争论这是否必要，并接受（可能需要适当的文档）第一个例子是装饰器预期使用的方式，第二个例子是错误的。这将是可行的，但需要密切注意，否则将发生运行时错误。

当然，如果装饰器接受没有默认值的参数，那么第二种语法就没有意义，只有一个可能性，这可能会使事情变得更简单。

或者，你可以使装饰器同时支持这两种语法。正如你可能猜到的，这需要额外的努力，而且像往常一样，你应该权衡这是否值得。

让我们用一个简单的例子来说明这一点，这个例子使用了一个带参数的装饰器来将参数注入到函数中。我们定义了一个接受两个参数的函数和一个执行相同操作的装饰器，我们的想法是在不带参数的情况下调用函数，让它使用装饰器传递的参数：

```py
 @decorator(x=3, y=4)
        def my_function(x, y):
            return x + y
        my_function()  # 7 
```

当然，我们为装饰器的参数定义了默认值，这样我们就可以不带值调用它。我们也希望不带括号调用它。

最简单且最直接的方法是使用条件语句来区分这两种情况：

```py
def decorator(function=None, *, x=DEFAULT_X, y=DEFAULT_Y):
    if function is None:  # called as `@decorator(...)`

        def decorated(function):
            @wraps(function)
            def wrapped():
                return function(x, y)

            return wrapped

        return decorated
    else:  # called as `@decorator`

        @wraps(function)
        def wrapped():
            return function(x, y)

        return wrapped 
```

注意关于装饰器签名的一个重要事项：参数只能是关键字参数。这大大简化了装饰器的定义，因为我们可以在没有参数调用函数时假设函数是`None`（否则，如果我们按位置传递值，我们传递的第一个参数会被误认为是函数）。如果我们想更加小心，而不是使用`None`（或任何哨兵值），我们可以检查参数类型，断言我们期望的函数对象类型，然后相应地调整参数，但这会使装饰器变得更加复杂。

另一个替代方案是将包装装饰器的一部分抽象出来，然后对函数进行部分应用（使用`functools.partial`）。为了更好地解释这一点，让我们考虑一个中间状态，并使用一个`lambda`函数来展示装饰器的参数是如何应用的，以及装饰器的参数是如何“移动”的：

```py
def decorator(function=None, *, x=DEFAULT_X, y=DEFAULT_Y):
    if function is None:
        return lambda f: decorator(f, x=x, y=y)

    @wraps(function)
    def wrapped():
        return function(x, y)

    return wrapped 
```

这与前面的例子类似，从意义上讲，我们有`wrapped`函数的定义（它是如何被装饰的）。然后，如果没有提供函数，我们返回一个新的函数，该函数接受一个函数作为参数（`f`），并返回应用了该函数的装饰器以及其余绑定参数。然后，在第二次递归调用中，函数将存在，并将返回常规的装饰器函数（wrapped）。

您可以通过更改函数部分应用的`lambda`定义来达到相同的结果：

```py
return partial(decorator, x=x, y=y) 
```

如果这对我们的用例来说太复杂，我们总是可以决定让我们的装饰器参数接受强制值。

在任何情况下，定义装饰器的参数为关键字参数（无论是否有默认值）可能是一个好主意。这是因为，通常在应用装饰器时，关于每个值的作用的上下文并不多，使用位置值可能不会产生非常有意义的表达式，因此最好更具有表达性，并将参数的名称与值一起传递。

如果您正在使用参数定义装饰器，请优先使用关键字参数。

同样，如果我们的装饰器不打算接受参数，并且我们想明确这一点，我们可以使用我们在*第二章*中学到的语法来定义我们的装饰器接收的单个位置参数。

对于我们的第一个例子，语法将是：

```py
def retry(operation, /): ... 
```

但请记住，这并不是严格推荐的，只是让您更明确地了解装饰器应该如何被调用。

## 协程装饰器

如介绍中所述，由于 Python 中几乎所有东西都是一个对象，因此几乎所有东西都可以被装饰，这包括协程。

然而，这里有一个注意事项，正如前几章所解释的，Python 中的异步编程引入了一些语法上的差异。因此，这些语法差异也将传递到装饰器中。

简而言之，如果我们为协程编写装饰器，我们可以简单地适应新的语法（记得等待包装的协程并将包装对象本身定义为协程，这意味着内部函数可能需要使用 '`async def`' 而不是仅仅使用 '`def`'）。

问题在于如果我们想要一个广泛适用于函数和协程的装饰器。在大多数情况下，创建两个装饰器可能是最简单（也许是最好的）的方法，但如果我们想为用户提供一个更简单的接口（通过让用户记住更少的对象），我们可以创建一个薄的包装器，充当两个内部（未公开）装饰器的调度器。这就像创建一个*外观*，但使用装饰器。

关于为函数和协程创建装饰器的难度，没有一般规则，因为这取决于我们想在装饰器中放入的逻辑。例如，下面的代码中有一个装饰器，它会改变它接收到的函数的参数，并且这将对常规函数或协程都有效：

```py
X, Y = 1, 2

def decorator(callable):
    """Call <callable> with fixed values"""

    @wraps(callable)
    def wrapped():
        return callable(X, Y)

    return wrapped

@decorator
def func(x, y):
    return x + y

@decorator
async def coro(x, y):
    return x + y 
```

然而，关于协程，我们需要做出一个重要的区分。装饰器将接收协程作为其`callable`参数，然后使用这些参数调用它。这创建了协程对象（将进入事件循环的任务），但它不会等待它，这意味着调用`await coro()`的人最终会等待装饰器包装的结果。这意味着，在像这种情况这样的简单情况下，我们不需要用另一个协程替换协程（尽管这通常是推荐的）。

但同样，这取决于我们想要做什么。如果我们需要一个`计时`函数，那么我们必须等待函数或协程完成以测量时间，为此我们必须在它上面调用`await`，这意味着包装器对象反过来必须是一个协程（但不是主要的装饰器）。

下面的代码示例使用一个装饰器来选择性地决定如何包装调用函数来说明这一点：

```py
import inspect
def timing(callable):
    @wraps(callable)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = callable(*args, **kwargs)
        latency = time.time() - start
        return {"latency": latency, "result": result}

    @wraps(callable)
    async def wrapped_coro(*args, **kwargs):
        start = time.time()
        result = await callable(*args, **kwargs)
        latency = time.time() - start
        return {"latency": latency, "result": result}

    if inspect.iscoroutinefunction(callable):
        return wrapped_coro

    return wrapped 
```

第二个包装器对于协程是必需的。如果我们没有它，那么代码会有两个问题。首先，对`callable`的调用（没有`await`）实际上不会等待操作完成，这意味着结果将是错误的。更糟糕的是，字典中`result`键的值不会是结果本身，而是创建的协程。因此，响应将是一个字典，任何尝试调用它的人都会尝试等待一个字典，这将导致错误。

作为一条一般规则，你应该用一个同类的另一个对象来替换装饰过的对象，也就是说，用一个函数替换另一个函数，用一个协程替换另一个协程。

我们还应该研究最近添加到 Python 中的一个最后增强功能，它消除了其语法的一些限制。

## 装饰器的扩展语法

Python 3.9 为装饰器引入了一个新特性，即 PEP-614 ([`www.python.org/dev/peps/pep-0614/`](https://www.python.org/dev/peps/pep-0614/)），因为允许了更通用的语法。在此增强之前，调用装饰器（在`@`之后）的语法被限制在非常有限的表达式上，并不是每个 Python 表达式都被允许。

解除这些限制后，我们现在可以编写更复杂的表达式，并在我们的装饰器中使用它们，如果我们认为这样可以节省一些代码行（但就像往常一样，要小心不要过度复杂化，得到一个更紧凑但难以阅读的行）。

例如，我们可以简化一些通常用于简单装饰器（记录函数调用及其参数）的嵌套函数。在这里（仅用于说明目的），我将装饰器中典型的嵌套函数定义替换为两个`lambda`表达式：

```py
def _log(f, *args, **kwargs):
    print(f"calling {f.__qualname__!r} with {args=} and {kwargs=}")
    return f(*args, **kwargs)

@(lambda f: lambda *args, **kwargs: _log(f, *args, **kwargs))
def func(x):
    return x + 1 
```

```py
>>> func(3)
calling 'func' with args=(3,) and kwargs={} 
```

PEP 文档引用了一些示例，说明此功能何时可能有用（例如，简化无操作函数以评估其他表达式，或避免使用`eval`函数）。

本书对此功能的推荐与所有可以通过更紧凑的语句实现的情况一致：只要不影响可读性，就编写更紧凑的代码版本。如果装饰器表达式难以阅读，则优先选择更冗长但更简单的替代方案，即编写两个或更多函数。

# 装饰器的良好用途

在本节中，我们将探讨一些常见的模式，这些模式很好地使用了装饰器。这些是装饰器是一个好选择时的常见情况。

从装饰器可以用于的无数应用中，我们将列举一些，其中最常见或相关的：

+   **转换参数**：更改函数的签名以提供更友好的 API，同时在下面封装如何处理和转换参数的细节。我们必须小心使用装饰器的这种用途，因为它只有在故意使用时才是好的特性。这意味着，如果我们明确使用装饰器为具有相当复杂签名的函数提供良好的签名，那么通过装饰器实现更干净的代码是一种很好的方法。另一方面，如果函数的签名由于装饰器而意外更改，那么这是我们想要避免的（我们将在本章末尾讨论如何避免）。

+   **跟踪代码**：记录带有其参数的函数执行。你可能熟悉提供跟踪功能的多个库，并且通常将这些功能作为装饰器暴露出来，以便添加到我们的函数中。这是一个很好的抽象，也是一个很好的接口，可以作为将代码与外部各方集成而不会造成太大干扰的方式。此外，它也是一个很好的灵感来源，因此我们可以编写自己的日志或跟踪功能作为装饰器。

+   **验证参数**：装饰器可以以透明的方式验证参数类型（例如，与预期值或其注释进行比较）。使用装饰器，我们可以为我们的抽象强制执行先决条件，遵循设计合同的思路。

+   **实现重试操作**：与我们在上一节中探讨的示例类似。

+   **通过将一些（重复的）逻辑移动到装饰器中来简化类**：这与 DRY 原则相关，我们将在本章末尾重新回顾。

在接下来的几节中，我将更详细地讨论这些主题。

## 适配函数签名

在面向对象设计中，有时存在具有不同接口的对象需要交互的情况。解决这个问题的一个方案是适配器设计模式，我们将在第七章“生成器、迭代器和异步编程”中讨论这个模式，当我们回顾一些主要设计模式时。

然而，本节的主题在某种程度上是相似的，即有时我们需要适应的不是对象，而是函数签名。

想象一下这样一个场景：你正在处理遗留代码，并且有一个包含大量使用复杂签名（许多参数、样板代码等）定义的函数的模块。有一个更干净的接口来与这些定义交互会很不错，但改变许多函数意味着需要进行大规模重构。

我们可以使用装饰器来将更改的差异降到最低。

有时我们可以使用装饰器作为我们代码和使用的框架之间的适配器，如果，例如，该框架有上述考虑。

想象一下这样一个框架，它期望调用我们定义的函数，保持一定的接口：

```py
def resolver_function(root, args, context, info): ... 
```

现在，我们到处都有这个签名，并决定最好从所有这些参数中创建一个抽象，它封装了它们，并暴露了我们在应用程序中需要的操作。

因此，现在我们有很多函数，它们的第一行重复创建相同的对象，而函数的其余部分只与我们的域对象交互：

```py
def resolver_function(root, args, context, info):
    helper = DomainObject(root, args, context, info)
    ...
    helper.process() 
```

在这个例子中，我们可以有一个改变函数签名的装饰器，这样我们就可以假设直接传递`helper`对象来编写我们的函数。在这种情况下，装饰器的任务将是拦截原始参数，创建域对象，然后将`helper`对象传递给我们的函数。然后我们定义我们的函数，假设我们只接收我们需要的对象，并且已经初始化。

也就是说，我们希望以这种形式编写我们的代码：

```py
@DomainArgs
def resolver_function(helper):
    helper.process()
   ... 
```

这也可以反过来，例如，如果我们有的遗留代码需要太多参数，而我们总是解构已经创建的对象，因为重构遗留代码会有风险，那么我们可以通过装饰器作为中间层来实现这一点。

这种使用装饰器的想法可以帮助你编写具有更简单、更紧凑签名的函数。

## 参数验证

我们之前提到过，装饰器可以用来验证参数（甚至在“设计由合同”**DbC**）的概念下强制某些先决条件或后置条件），所以从这个角度来看，你可能已经意识到在处理或操作参数时使用装饰器是相当常见的。

特别是，有些情况下，我们会发现自己反复创建相似的对象或应用相似的转换，而我们希望将这些抽象出来。大多数时候，我们可以通过简单地使用装饰器来实现这一点。

## 跟踪代码

在本节中谈到**跟踪**时，我们将指代更一般的东西，这与处理我们希望监控的函数执行有关。这可能包括我们想要的情况：

+   跟踪函数的执行（例如，通过记录它执行的行）

+   监控函数的一些指标（如 CPU 使用率或内存占用）

+   测量函数的运行时间

+   记录函数被调用时及其传递的参数

在下一节中，我们将探讨一个简单的装饰器示例，该装饰器记录函数的执行，包括其名称和运行时间。

# 有效的装饰器 - 避免常见错误

尽管装饰器是 Python 的一个伟大特性，但如果使用不当，它们也不会免除问题。在本节中，我们将看到一些常见的避免问题，以便创建有效的装饰器。

## 保留原始包装对象的有关数据

在将装饰器应用于函数时，最常见的问题之一是原始函数的一些属性或属性没有被保留，导致不希望出现且难以追踪的副作用。

为了说明这一点，我们展示了一个负责在函数即将运行时记录日志的装饰器：

```py
# decorator_wraps_1.py
def trace_decorator(function):
    def wrapped(*args, **kwargs):
        logger.info("running %s", function.__qualname__)
        return function(*args, **kwargs)
    return wrapped 
```

现在，让我们假设我们有一个应用了此装饰器的函数。我们可能会最初认为这个函数与它的原始定义相比没有任何修改：

```py
@trace_decorator
def process_account(account_id: str):
    """Process an account by Id."""
    logger.info("processing account %s", account_id)
    ... 
```

但也许有变化。

装饰器不应该改变原始函数的任何内容，但，结果证明，由于它包含缺陷，它实际上正在修改其名称和 docstring，以及其他属性。

让我们尝试获取这个函数的`help`信息：

```py
>>> help(process_account)
Help on function wrapped in module decorator_wraps_1:
wrapped(*args, **kwargs) 
```

然后让我们检查它的调用方式：

```py
>>> process_account.__qualname__
'trace_decorator.<locals>.wrapped' 
```

此外，原始函数的注解也丢失了：

```py
>>> process_account.__annotations__
{} 
```

我们可以看到，由于装饰器实际上是将原始函数替换为一个新的函数（称为`wrapped`），所以我们看到的是这个函数的属性，而不是原始函数的属性。

如果我们将这样的装饰器应用于多个具有不同名称的函数，它们最终都会被调用为`wrapped`，这是一个主要问题（例如，如果我们想记录或跟踪函数，这将使调试更加困难）。

另一个问题是我们如果在这些函数上放置带有测试的 docstrings，它们将被装饰器的那些覆盖。结果，我们用`doctest`模块调用我们的代码时，我们想要的测试的 docstrings 将不会运行（如我们在*第一章*，*介绍、代码格式化和工具*中看到的）。

修复方法很简单。我们只需在内部函数（`wrapped`）中应用`wraps`装饰器，告诉它这实际上是一个包装函数：

```py
# decorator_wraps_2.py
def trace_decorator(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        logger.info("running %s", function.__qualname__)
        return function(*args, **kwargs)
    return wrapped 
```

现在，如果我们检查属性，我们将得到我们最初预期的结果。检查函数的帮助信息，如下所示：

```py
>>> from decorator_wraps_2 import process_account
>>> help(process_account)
Help on function process_account in module decorator_wraps_2:
process_account(account_id)
    Process an account by Id. 
```

并验证其限定名称是否正确，如下所示：

```py
>>> process_account.__qualname__
'process_account' 
```

最重要的是，我们恢复了可能存在于文档字符串上的单元测试！通过使用 `wraps` 装饰器，我们还可以通过 `__wrapped__` 属性访问原始的、未修改的函数。尽管在生产环境中不应使用它，但在某些单元测试中，当我们想要检查函数的未修改版本时，它可能很有用。

通常情况下，对于简单的装饰器，我们使用 `functools.wraps` 的方式会遵循以下通用公式或结构：

```py
def decorator(original_function):
    @wraps(original_function)
    def decorated_function(*args, **kwargs):
        # modifications done by the decorator ...
        return original_function(*args, **kwargs)
    return decorated_function 
```

在创建装饰器时，始终使用 `functools.wraps` 对包装函数进行应用，如前述公式所示。

## 处理装饰器中的副作用

在本节中，我们将了解到在装饰器的主体中避免副作用是明智的。在某些情况下，这可能是可以接受的，但底线是，如果有疑问，应决定不这样做，原因将在下面解释。装饰器除了装饰的函数之外需要做的所有事情都应该放在最内层的函数定义中，否则在导入时会出现问题。尽管如此，有时这些副作用是必需的（甚至可能是期望的）在导入时运行，反之亦然。

我们将看到这两种情况的示例，以及它们各自适用的场景。如果有疑问，应谨慎行事，并将所有副作用推迟到最后一刻，即在 `wrapped` 函数将要被调用之后。

接下来，我们将看到在 `wrapped` 函数外部放置额外逻辑不是什么好主意的情况。

### 装饰器中对副作用的不正确处理

让我们想象一个装饰器的例子，它的目的是在函数开始运行时记录日志，然后记录其运行时间：

```py
def traced_function_wrong(function):
    logger.info("started execution of %s", function)
    start_time = time.time()
    @wraps(function)
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

现在，我们将装饰器应用到普通函数上，认为它将正常工作：

```py
@traced_function_wrong
def process_with_delay(callback, delay=0):
    time.sleep(delay)
    return callback() 
```

这个装饰器中存在一个微妙但关键的错误。

首先，让我们导入函数，多次调用它，看看会发生什么：

```py
>>> from decorator_side_effects_1 import process_with_delay
INFO:started execution of <function process_with_delay at 0x...> 
```

只需导入函数，我们就会注意到有问题。日志行不应该在那里，因为函数没有被调用。

现在，如果我们运行函数，看看它运行需要多长时间？实际上，我们预计多次调用相同的函数将给出相似的结果：

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

每次我们运行相同的函数，它所需的时间会越来越长！在这个时候，你可能已经注意到了（现在很明显）的错误。

记住装饰器的语法。`@traced_function_wrong` 实际上意味着以下内容：

```py
process_with_delay = traced_function_wrong(process_with_delay) 
```

因此，函数中设置的时间将是模块导入的时间。随后的调用将计算从运行时间到原始起始时间的差异。它还会在错误的时间记录日志，而不是在函数实际被调用时。

幸运的是，修复也很简单——我们只需将 `wrapped` 函数内部的代码移动，以延迟其执行：

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

使用这个新版本，之前的问题都得到了解决。

如果装饰器的行为不同，结果可能会更加灾难性。例如，如果它要求你记录事件并将它们发送到外部服务，那么除非在导入之前已经正确运行了配置，否则它肯定会失败。即使我们可以做到，这也是一种不良的做法。如果装饰器有任何其他类型的副作用，例如从文件中读取、解析配置等，也同样适用。

### 需要具有副作用的装饰器

有时，装饰器的副作用是必要的，我们不应该将它们的执行延迟到最后时刻，因为这是它们正常工作所需机制的一部分。

当我们不想延迟装饰器的副作用时，一个常见的场景是我们需要将对象注册到一个公共注册表中，该注册表将在模块中可用。

例如，回到我们之前的`event`系统示例，我们现在只想在模块中使某些事件可用，而不是所有事件。在事件层次结构中，我们可能希望有一些中间类，它们不是我们希望在系统中处理的真实事件，而是它们的派生类。

而不是根据每个类是否将被处理来标记每个类，我们可以通过装饰器显式地注册每个类。

在这种情况下，我们有一个与用户活动相关的所有事件的类。然而，这只是一个中间表，用于我们实际想要的事件类型，即`UserLoginEvent`和`UserLogoutEvent`：

```py
EVENTS_REGISTRY = {}
def register_event(event_cls):
    """Place the class for the event into the registry to make it     accessible in the module.
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

当我们查看前面的代码时，似乎`EVENTS_REGISTRY`是空的，但在从该模块导入某些内容后，它将填充所有在`register_event`装饰器下的类：

```py
>>> from decorator_side_effects_2 import EVENTS_REGISTRY
>>> EVENTS_REGISTRY
{'UserLoginEvent': decorator_side_effects_2.UserLoginEvent,
 'UserLogoutEvent': decorator_side_effects_2.UserLogoutEvent} 
```

这可能看起来很难读懂，甚至具有误导性，因为`EVENTS_REGISTRY`将在模块导入后立即具有其最终值，在运行时，我们无法仅通过查看代码就轻易预测其值。

虽然如此，在某些情况下，这种模式是有道理的。事实上，许多 Web 框架或知名库都使用这种方法来工作，暴露对象或使它们可用。话虽如此，请注意这个风险，如果你在自己的项目中实现类似的功能：大多数情况下，选择一个替代方案会更受欢迎。

在这个情况下，装饰器也没有改变`wrapped`对象或以任何方式改变其工作方式。然而，这里的重要提示是，如果我们对某些内容进行修改并定义一个内部函数来修改`wrapped`对象，我们仍然可能希望将注册结果的代码放在外部。

注意到“*outside*”这个词的使用。它并不一定意味着“之前”，它只是不属于同一个闭包的一部分；但它在外部作用域中，所以它不会被延迟到运行时。

## 创建始终有效的装饰器

装饰器可能适用于几种不同的场景。也可能存在这样的情况，我们需要为属于这些不同多个场景的对象使用相同的装饰器，例如，如果我们想重用我们的装饰器并将其应用于函数、类、方法或静态方法。

如果我们只考虑支持我们想要装饰的第一种类型的对象来创建装饰器，我们可能会注意到，同样的装饰器在另一种类型的对象上工作得并不一样好。一个典型的例子是，我们创建一个用于函数的装饰器，然后我们想将其应用于类的某个方法，结果发现它不起作用。如果我们为方法设计了装饰器，然后我们希望它也能应用于静态方法或类方法，也可能出现类似的场景。

在设计装饰器时，我们通常考虑代码的重用，因此我们希望将这个装饰器用于函数和方法。

使用签名 `*args` 和 `**kwargs` 定义我们的装饰器将使它们在所有情况下都能工作，因为这是最通用的签名类型。然而，有时我们可能不想使用这种签名，而是根据原始函数的签名定义装饰器包装函数，这主要是因为两个原因：

+   它将更易于阅读，因为它类似于原始函数。

+   实际上需要处理这些参数，因此接收 `*args` 和 `**kwargs` 并不方便。

考虑到我们代码库中有许多需要从参数创建特定对象的功能。例如，我们传递一个字符串，并使用它初始化一个驱动对象，反复进行。然后我们认为我们可以通过使用一个装饰器来处理这个参数的转换来消除重复。

在下一个例子中，我们假设 `DBDriver` 是一个知道如何连接和运行数据库操作的对象，但它需要一个连接字符串。我们代码中的方法被设计为接收包含数据库信息的字符串，并要求我们始终创建 `DBDriver` 实例。装饰器的想法是它会自动替换这个转换——函数将继续接收一个字符串，但装饰器将创建一个 `DBDriver` 实例并将其传递给函数，因此我们可以假设我们直接接收所需的对象。

在下一个列表中展示了如何在函数中使用这个例子：

```py
# src/decorator_universal_1.py
from functools import wraps
from log import logger
class DBDriver:
    def __init__(self, dbstring: str) -> None:
        self.dbstring = dbstring
    def execute(self, query: str) -> str:
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

如果我们将一个字符串传递给函数，我们可以得到由 `DBDriver` 实例执行的结果，所以装饰器按预期工作：

```py
>>> run_query("test_OK")
'query test_function at test_OK' 
```

但现在，我们想在类方法中重用这个相同的装饰器，我们发现同样的问题：

```py
class DataHandler:
    @inject_db_driver
    def run_query(self, driver):
        return driver.execute(self.__class__.__name__) 
```

我们尝试使用这个装饰器，但发现它不起作用：

```py
>>> DataHandler().run_query("test_fails")
Traceback (most recent call last):
  ...
TypeError: wrapped() takes 1 positional argument but 2 were given 
```

问题是什么？

类中的方法定义了一个额外的参数——`self`。

方法只是接收`self`（它们定义的对象）作为第一个参数的特定类型的函数。

因此，在这种情况下，这个装饰器（设计为只与一个名为`dbstring`的参数一起工作）将解释`self`是这个参数，并且将调用方法，用字符串代替`self`，在第二个参数的位置上没有内容，即我们传递的字符串。

为了解决这个问题，我们需要创建一个既适用于方法也适用于函数的装饰器，我们通过定义这个作为装饰器对象并实现协议描述符来实现这一点。

描述符在*第七章*，*生成器、迭代器和异步编程*中得到了全面解释，所以现在我们只需将其视为一个使装饰器工作的配方。

解决方案是实现一个作为类对象的装饰器，并通过实现`__get__`方法使其成为一个描述符：

```py
from functools import wraps
from types import MethodType
class inject_db_driver:
    """Convert a string to a DBDriver instance and pass this to the 
       wrapped function."""
    def __init__(self, function) -> None:
        self.function = function
        wraps(self.function)(self)
    def __call__(self, dbstring):
        return self.function(DBDriver(dbstring))
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.__class__(MethodType(self.function, instance)) 
```

描述符的详细信息将在*第六章*，*使用描述符获取更多对象功能*中解释，但为了这个示例，我们可以说它实际上是将它装饰的调用重新绑定到方法上，这意味着它将函数绑定到对象上，然后使用这个新的调用重新创建装饰器。

对于函数来说，这仍然有效，因为它根本不会调用`__get__`方法。

# 装饰器和整洁代码

现在我们对装饰器有了更多的了解，如何编写它们，以及如何避免常见问题，是时候将它们提升到下一个层次，看看我们如何利用所学知识来制作更好的软件。

我们在前面几节中简要地提到了这个主题，但那些更接近代码的示例，因为建议是关于如何使特定的代码行（或部分）更易读。

从现在开始讨论的主题与更普遍的设计原则相关。其中一些想法我们在前面的章节中已经接触过，但这里的观点是理解我们如何为了这样的目的使用装饰器。

## 组合优于继承

我们已经简要讨论过，一般来说，组合比继承更好，因为后者会带来一些问题，使得代码组件更加耦合。

在书籍《设计模式：可复用面向对象软件元素》（DESIG01）中，围绕设计模式的大部分想法都是基于以下观点：

> 优先组合而非类继承

在*第二章*，*Pythonic 代码*中，我介绍了使用魔法方法`__getattr__`在对象上动态解析属性的想法。我还给出了一个例子，说明这可以用来根据命名约定自动解析属性，如果外部框架需要的话。让我们探索两种不同的解决方案。

对于这个例子，让我们假设我们正在与一个命名约定为使用前缀"`resolve_`"来解析属性的框架交互，但我们的领域对象只有那些没有"`resolve_`"前缀的属性。

显然，我们不想为每个属性编写很多重复的名为"`resolve_x`"的方法，所以第一个想法是利用前面提到的`__getattr__`魔法方法，并将其放置在基类中：

```py
class BaseResolverMixin:
    def __getattr__(self, attr: str):
        if attr.startswith("resolve_"):
            *_, actual_attr = attr.partition("resolve_")
        else:
            actual_attr = attr
        try:
            return self.__dict__[actual_attr]
        except KeyError as e:
            raise AttributeError from e

@dataclass
class Customer(BaseResolverMixin):
    customer_id: str
    name: str
    address: str 
```

这将解决问题，但我们能做得更好吗？

我们可以设计一个类装饰器来直接设置这个方法：

```py
from dataclasses import dataclass

def _resolver_method(self, attr):
    """The resolution method of attributes that will replace __getattr__."""
    if attr.startswith("resolve_"):
        *_, actual_attr = attr.partition("resolve_")
    else:
        actual_attr = attr
    try:
        return self.__dict__[actual_attr]
    except KeyError as e:
        raise AttributeError from e

def with_resolver(cls):
    """Set the custom resolver method to a class."""
    cls.__getattr__ = _resolver_method
    return cls

@dataclass
@with_resolver
class Customer:
    customer_id: str
    name: str
    address: str 
```

两个版本都将符合以下行为：

```py
>>> customer = Customer("1", "name", "address")
>>> customer.resolve_customer_id
'1'
>>> customer.resolve_name
'name' 
```

首先，我们有一个独立的方法`resolve`，它遵循原始`__getattr__`的签名（这就是为什么我甚至保留了`self`作为第一个参数的名字，以便有意识地让这个函数成为一个方法）。

其余的代码看起来相当简单。我们的装饰器只将方法设置为我们通过参数接收的类，然后我们只需将装饰器应用于我们的类，就不再需要使用继承。

这与之前的例子相比有什么好处呢？首先，我们可以争论使用装饰器意味着我们正在使用组合（取一个类，修改它，然后返回一个新的类）而不是继承，因此我们的代码与最初的基本类耦合度更低。

此外，我们可以说，第一个例子中使用继承（通过混合类）是相当虚构的。我们并没有使用继承来创建类的更专用版本，而是为了利用`__getattr__`方法。这有两个（互补的）原因：首先，继承不是重用代码的最佳方式。好的代码是通过拥有小的、内聚的抽象来重用的，而不是创建层次结构。

其次，记得从之前的章节中，创建子类应该遵循特殊化的想法，“是一种”的关系。从概念上考虑，一个客户是否真的是一个`BaseResolverMivin`（顺便问一下，那是什么？）。

为了更清楚地说明这个第二点，想象你有一个像这样的层次结构：

```py
class Connection: pass
class EncryptedConnection(Connection): pass 
```

在这种情况下，使用继承可以说是正确的，毕竟加密连接是一种更具体的连接类型。但什么是比`BaseResolverMixin`更具体的类型呢？这是一个混合类，所以它被期望与其他类（使用多重继承）一起混合在层次结构中。使用这种混合类纯粹是实用主义的，并且出于实现目的。请别误会，这是一本实用主义的书，所以你会在你的专业经验中处理混合类，使用它们是完全正常的，但如果我们可以避免这种纯粹实现上的抽象，并用不损害我们的领域对象（在这种情况下是`Customer`类）的东西来替换它，那就更好了。

新设计还有一个令人兴奋的能力，那就是可扩展性。我们已经看到装饰器可以参数化。想象一下，如果我们允许装饰器设置任何解析函数，而不仅仅是定义的那个，我们能在我们的设计中实现多大的灵活性。

## 装饰器与 DRY 原则

我们已经看到装饰器如何允许我们将某些逻辑抽象到一个单独的组件中。这样做的主要优势是，我们可以将装饰器多次应用于不同的对象，以重用代码。这遵循了**不要重复自己**（**DRY**）原则，因为我们只定义某些知识一次。

在前几节中实现的`retry`机制是一个很好的例子，它是一个可以多次应用以重用代码的装饰器。我们不是让每个特定的函数都包含自己的`retry`逻辑，而是创建一个装饰器并多次应用它。一旦我们确保装饰器可以与方法和函数一样工作，这样做是有意义的。

定义事件表示方式的类装饰器也符合 DRY 原则，因为它定义了一个特定的地方来处理序列化事件的逻辑，而不需要在不同的类中重复代码。由于我们预计会重用这个装饰器并将其应用于许多类，因此其开发（及其复杂性）是有回报的。

在尝试使用装饰器重用代码时，这一点非常重要：我们必须绝对确信我们实际上会节省代码。

任何装饰器（尤其是如果它没有经过精心设计的话）都会给代码增加一个额外的间接层，从而增加更多的复杂性。代码的读者可能想要追踪装饰器的路径，以便完全理解函数的逻辑（尽管这些考虑在下一节中已经讨论过），因此请记住，这种复杂性必须得到回报。如果不太可能大量重用，那么就不应该选择装饰器，而应该选择更简单的选项（可能只是一个单独的函数或另一个小型类就足够了）。

但我们如何知道过度重用是什么意思？有没有一个规则来确定何时将现有代码重构为装饰器？Python 中并没有针对装饰器的特定内容，但我们可以应用软件工程中的一个通用经验法则（GLASS 01），即一个组件至少应该尝试三次，然后再考虑创建一个通用的抽象，即可重用组件。同样来自同一参考（GLASS 01）（我鼓励所有读者阅读《软件工程的真相与谬误》，因为它是一个极好的参考资料）的想法是，创建可重用组件比创建简单的组件难三倍。

核心观点是，通过装饰器重用代码是可以接受的，但前提是你必须考虑到以下因素：

+   不要一开始就从头开始创建装饰器。等待模式出现，装饰器的抽象变得清晰，然后再进行重构。

+   考虑到装饰器可能需要多次（至少三次）应用，然后再进行实现。

+   将装饰器中的代码保持到最小。

既然我们已经从装饰器的角度重新审视了 DRY 原则，我们仍然可以讨论应用于装饰器的关注点分离，正如下一节所探讨的那样。

## 装饰器与关注点分离

上一个列表中的最后一点非常重要，以至于它值得单独成节。我们已经探讨了代码重用的概念，并注意到重用代码的关键要素是具有凝聚性的组件。这意味着它们应该具有最小的职责水平——只做一件事，只做一件事，并且做好。我们的组件越小，它们就越可重用，并且可以在不同的上下文中应用，而不会携带额外的行为，这会导致耦合和依赖，从而使软件变得僵化。

为了展示这意味着什么，让我们回顾一下我们在前一个示例中使用的一个装饰器。我们创建了一个装饰器，它使用类似于以下代码的方式来追踪某些函数的执行：

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

现在，这个装饰器虽然有效，但有一个问题——它做了不止一件事。它记录了一个特定函数已被调用，并记录了运行所需的时间。每次我们使用这个装饰器时，我们都在承担这两项职责，即使我们只想承担其中之一。

这应该被分解成更小的装饰器，每个装饰器都有更具体和有限的职责：

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
        logger.info(
            "function %s took %.2f",
            function.__qualname__,
            time.time() - start_time,
        )
        return result
    return wrapped 
```

注意，我们之前拥有的相同功能可以通过简单地结合两者来实现：

```py
@measure_time
@log_execution
def operation():
    .... 
```

注意装饰器应用顺序的重要性。

不要在装饰器中放置超过一个职责。**单一职责原则**（SRP）同样适用于装饰器。

最后，我们可以分析优秀的装饰器，以了解它们在实际中的应用。下一节将通过分析装饰器来总结本章所学内容。

## 优秀装饰器的分析

作为本章的结束语，让我们回顾一些优秀的装饰器示例以及它们在 Python 本身以及流行库中的使用方式。目的是获取有关如何创建优秀装饰器的指导原则。

在进入示例之前，让我们首先确定优秀装饰器应具备的特征：

+   **封装，或关注点分离**：一个好的装饰器应该有效地将其所做之事与所装饰的内容之间的不同职责分开。它不能是一个有漏洞的抽象，这意味着装饰器的客户端应该只以黑盒模式调用它，而不了解其实际实现逻辑。

+   **正交性**：装饰器所做的事情应该是独立的，并且尽可能与它所装饰的对象解耦。

+   **可重用性**：希望装饰器可以应用于多种类型，而不仅仅是出现在一个函数的一个实例上，因为这意味着它可能只是一个函数。它必须足够通用。

在`Celery`项目中可以找到一个装饰器的美好例子，其中任务是通过将应用中的任务装饰器应用到函数上来定义的：

```py
@app.task
def mytask():
   .... 
```

这个装饰器之所以好，其中一个原因是因为它在某一方面非常出色——封装。库的用户只需要定义函数体，装饰器就会自动将其转换为任务。`@app.task`装饰器无疑封装了大量的逻辑和代码，但这些都无关紧要`mytask()`函数的主体。这是完整的封装和关注点的分离——没有人需要查看装饰器做了什么，因此它是一个正确的抽象，不会泄露任何细节。

装饰器的另一个常见用途是在 Web 框架中（例如`Pyramid`、`Flask`和`Sanic`，仅举几个例子），在这些框架中，视图的处理程序通过装饰器注册到 URL：

```py
@route("/", method=["GET"])
def view_handler(request):
 ... 
```

这类装饰器与之前的有相同的考虑因素；它们也提供了完全的封装，因为使用 Web 框架的用户很少（如果有的话）需要知道`@route`装饰器正在做什么。在这种情况下，我们知道装饰器正在做更多的事情，比如将这些函数注册到 URL 的映射器，并且它还在改变原始函数的签名，为我们提供一个更友好的接口，该接口接收一个已经设置好所有信息的`request`对象。

前两个例子已经足够让我们注意到关于这种装饰器使用的一个其他方面。它们遵循一个 API。这些库和框架通过装饰器向用户公开其功能，结果证明装饰器是定义干净编程接口的一种极好的方式。

这可能是我们应该考虑装饰器的最好方式。就像在类装饰器的例子中，它告诉我们事件属性将如何被处理一样，一个好的装饰器应该提供一个干净的接口，这样代码的用户就知道从装饰器可以期待什么，而无需了解它是如何工作的，或者它的任何细节。

# 摘要

装饰器是 Python 中的强大工具，可以应用于许多事物，如类、方法、函数、生成器等等。我们已经展示了如何以不同的方式创建装饰器，用于不同的目的，并在过程中得出了一些结论。

在为函数创建装饰器时，尽量使它的签名与被装饰的原函数相匹配。而不是使用通用的`*args`和`**kwargs`，使签名与原函数相匹配将使其更容易阅读和维护，并且会更接近原函数，因此对代码的读者来说会更加熟悉。

装饰器是重用代码和遵循 DRY 原则的一个非常有用的工具。然而，它们的实用性是有代价的，如果使用不当，复杂性可能会带来比好处更多的坏处。因此，我们强调，装饰器应该在它们将被多次应用（三次或更多次）时使用。与 DRY 原则一样，我们支持关注点分离的理念，目标是使装饰器尽可能小。

装饰器的另一个良好用途是创建更清晰的接口，例如，通过将部分逻辑提取到装饰器中来简化类的定义。从这个意义上讲，装饰器也通过向用户提供有关特定组件将执行什么操作的信息来帮助提高可读性，而无需了解如何实现（封装）。

在下一章中，我们将探讨 Python 的另一个高级特性——描述符。特别是，我们将看到如何借助描述符，我们可以创建更好的装饰器并解决本章中遇到的一些问题。

# 参考文献

这里是一份您可以参考的信息列表：

+   *PEP-318*: *函数和方法的装饰器* ([`www.python.org/dev/peps/pep-0318/`](https://www.python.org/dev/peps/pep-0318/))

+   *PEP-3129*: *类装饰器* ([`www.python.org/dev/peps/pep-3129/`](https://www.python.org/dev/peps/pep-3129/))

+   *WRAPT 01*: [`pypi.org/project/wrapt/`](https://pypi.org/project/wrapt/)

+   *WRAPT 02*: [`wrapt.readthedocs.io/en/latest/decorators.html#universal-decorators`](https://wrapt.readthedocs.io/en/latest/decorators.html#universal-decorators)

+   *Python 标准库中的 functools 模块：wraps 函数* ([`docs.python.org/3/library/functools.html#functools.wrap`](https://docs.python.org/3/library/functools.html#functools.wrap))

+   *ATTRS 01*: *attrs 库* ([`pypi.org/project/attrs/`](https://pypi.org/project/attrs/))

+   *PEP-557*: *数据类* ([`www.python.org/dev/peps/pep-0557/`](https://www.python.org/dev/peps/pep-0557/))

+   *GLASS 01*: 由*Robert L. Glass*所著的名为*软件工程的事实与谬误*的书籍

+   *DESIG01*: 由*Erich Gamma*所著的名为*设计模式：可复用面向对象软件元素*的书籍

+   *PEP-614*: *放宽装饰器的语法限制* ([`www.python.org/dev/peps/pep-0614/`](https://www.python.org/dev/peps/pep-0614/))
