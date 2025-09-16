

# 第一章：基础设计原则

设计原则是任何良好架构的软件的基础。它们作为指导之光，帮助开发者导航创建可维护、可扩展和健壮应用程序的道路，同时避免不良设计带来的陷阱。

在本章中，我们将探讨所有开发者都应该了解并应用于其项目的核心设计原则。我们将探讨四个基础原则。第一个，“封装变化”，教您如何隔离代码中易变的部分，使修改和扩展应用程序变得更加容易。接下来，“优先使用组合”，让您理解为什么通常从简单对象组装复杂对象比继承功能更好。第三个，“面向接口编程”，展示了面向接口而非具体类编码的力量，增强了灵活性和可维护性。最后，通过“松耦合”原则，您将掌握减少组件之间依赖关系的重要性，使代码更容易重构和测试。

在本章中，我们将涵盖以下主要主题：

+   遵循“封装变化”原则

+   遵循“优先使用组合而非继承”原则

+   遵循“面向接口而非实现编程”原则

+   遵循“松耦合”原则

到本章结束时，您将对这些原则及其在 Python 中的实现有一个扎实的理解，为本书的其余部分打下基础。

# 技术要求

对于本书中的章节，您需要一个运行中的 Python 3.12 环境或在某些章节的某些特殊情况中，3.11 版本。

此外，通过运行以下命令安装 Mypy 静态类型检查器（[`www.mypy-lang.org`](https://www.mypy-lang.org)）：

```py
python3.12 –m pip install -–user mypy
```

示例可以在以下 GitHub 仓库中找到：[`github.com/PacktPublishing/Mastering-Python-Design-Patterns-Third-Edition`](https://github.com/PacktPublishing/Mastering-Python-Design-Patterns-Third-Edition)

关于 Python 可执行文件

在整本书中，我们将引用 Python 可执行文件来执行代码示例，作为 `python3.12` 或 `python`。根据您的具体环境、实践和/或工作流程进行适配。

# 遵循“封装变化”原则

软件开发中最常见的挑战之一是处理变化。需求演变、技术进步，以及用户需求也会发生变化。因此，编写能够适应变化而不会在您的程序或应用程序中引起连锁修改的代码至关重要。这就是“封装变化”原则发挥作用的地方。

## 这是什么意思？

这一原则背后的思想很简单：隔离您代码中最可能发生变化的部分，并将它们封装起来。通过这样做，您创建了一个保护屏障，保护代码的其他部分免受这些可能发生变化元素的影响。这种封装允许您在不影响其他部分的情况下对系统的一部分进行更改。

## 优点

封装变化的部分提供了几个好处，主要包括以下：

+   **易于维护**：当需要更改时，您只需修改封装的部分，从而降低在其他地方引入错误的风险

+   **增强灵活性**：封装的组件可以轻松交换或扩展，提供更适应的架构

+   **提高可读性**：通过隔离变化元素，您的代码变得更加有组织，更容易理解

## 实现封装的技术

正如我们所介绍的，封装有助于数据隐藏，仅暴露必要的功能。在这里，我们将介绍增强 Python 中封装的关键技术：多态性和 *getters* 和 *setters* 技术。

### 多态性

在编程中，多态性允许将不同类的对象视为公共超类对象。它是 **面向对象编程** **OOP** 的核心概念之一，它使单个接口能够表示不同类型。多态性允许实现优雅的软件设计模式，如策略模式，并且是实现 Python 中干净、可维护代码的一种方式。

### Getters and Setters

这些是在一个类中使用的特殊方法，它们允许对属性值进行受控访问。*getters* 允许读取属性的值，而 *setters* 允许修改它们。通过使用这些方法，您可以添加验证逻辑或副作用，如记录日志，从而遵循封装的原则。它们提供了一种控制和保护对象状态的方法，并且在您想要封装从其他实例变量派生的复杂属性时特别有用。

还有更多。为了补充 *getters* 和 *setters* 技术，Python 提供了一种更优雅的方法，称为 *property* 技术。这是 Python 的内置功能，允许您无缝地将属性访问转换为方法调用。使用属性，您可以在不显式定义 *getter* 和 *setter* 方法的情况下，确保对象在不受正确或有害操作的情况下保持其内部状态。

`@property` 装饰器允许您定义一个方法，当访问属性时自动调用，有效地充当 *getter*。同样，`@attribute_name.setter` 装饰器允许您定义一个方法，充当 *setter*，在您尝试更改属性值时调用。这样，您可以直接在这些方法中嵌入验证或其他操作，使代码更加简洁。

通过使用*属性*技术，你可以实现与传统的*获取器*和*设置器*相同级别的数据封装和验证，但以一种更符合 Python 设计哲学的方式。它允许你编写不仅功能性强，而且干净、易于阅读的代码，从而增强封装和 Python 程序的整体质量。

接下来，我们将通过示例更好地理解这些技术。

## 举例——使用多态进行封装

多态是实现变化行为封装的强大方式。让我们通过一个支付处理系统的例子来看一下，在这个系统中，支付方式选项可以变化。在这种情况下，你可能会将每种支付方式封装在其自己的类中：

1.  你首先定义支付方法的基类，提供一个`process_payment()`方法，每个具体的支付方法都将实现它。这就是我们封装变化的部分——支付处理逻辑。这部分代码如下：

    ```py
    class PaymentBase:
        def __init__(self, amount: int):
            self.amount: int = amount
        def process_payment(self):
            pass
    ```

1.  接下来，我们将介绍`CreditCard`和`PayPal`类，它们继承自`PaymentBase`，每个类都提供了自己的`process_payment`实现。这是一种经典的多态方式，因为你可以将`CreditCard`和`PayPal`对象视为它们共同超类实例。代码如下：

    ```py
    class CreditCard(PaymentBase):
        def process_payment(self):
            msg = f"Credit card payment: {self.amount}"
            print(msg)
    class PayPal(PaymentBase):
        def process_payment(self):
            msg = f"PayPal payment: {self.amount}"
            print(msg)
    ```

1.  为了使测试我们刚刚创建的类成为可能，让我们添加一些代码，为每个对象调用`process_payment()`。当你使用这些类时，多态的美丽之处显而易见，如下所示：

    ```py
    if __name__ == "__main__":
        payments = [CreditCard(100), PayPal(200)]
        for payment in payments:
            payment.process_payment()
    ```

本例的完整代码（`ch01/encapsulate.py`）如下：

```py
class PaymentBase:
    def __init__(self, amount: int):
        self.amount: int = amount
    def process_payment(self):
        pass
class CreditCard(PaymentBase):
    def process_payment(self):
        msg = f"Credit card payment: {self.amount}"
        print(msg)
class PayPal(PaymentBase):
    def process_payment(self):
        msg = f"PayPal payment: {self.amount}"
        print(msg)
if __name__ == "__main__":
    payments = [CreditCard(100), PayPal(200)]
    for payment in payments:
        payment.process_payment()
```

要测试代码，请运行以下命令：

```py
python3.12 ch01/encapsulate.py
```

你应该得到以下输出：

```py
Credit card payment: 100
PayPal payment: 200
```

如你所见，当支付方式改变时，程序会适应以产生预期的结果。

通过封装变化的部分——在这里是*支付方式*——你可以轻松地添加新选项或修改现有选项，而不会影响核心支付处理逻辑。

## 举例——使用属性进行封装

让我们定义一个`Circle`类，并展示如何使用 Python 的`@property`技术为其`radius`属性创建一个*获取器*和一个*设置器*。

注意，底层属性实际上会被称为`_radius`，但它被隐藏/保护在名为`radius`的*属性*后面。

让我们一步步编写代码：

1.  我们首先定义`Circle`类及其初始化方法，其中我们将`_radius`属性初始化如下：

    ```py
    class Circle:
        def __init__(self, radius: int):
            self._radius: int = radius
    ```

1.  我们添加半径属性：一个`radius()`方法，其中我们从底层属性返回值，使用`@property`装饰器装饰，如下所示：

    ```py
        @property
        def radius(self):
            return self._radius
    ```

1.  我们添加半径设置器部分：另一个`radius()`方法，其中我们在验证检查后实际修改底层属性，因为我们不希望允许半径为负值；此方法由特殊的`@radius.setter`装饰器装饰。这部分代码如下：

    ```py
        @radius.setter
        def radius(self, value: int):
            if value < 0:
                raise ValueError("Radius cannot be negative!")
            self._radius = value
    ```

1.  最后，我们添加一些将帮助我们测试类的代码，如下所示：

    ```py
    if __name__ == "__main__":
        circle = Circle(10)
        print(f"Initial radius: {circle.radius}")
        circle.radius = 15
        print(f"New radius: {circle.radius}")
    ```

此示例的完整代码（`ch01/encapsulate_bis.py`）如下所示：

```py
class Circle:
    def __init__(self, radius: int):
        self._radius: int = radius
    @property
    def radius(self):
        return self._radius
    @radius.setter
    def radius(self, value: int):
        if value < 0:
            raise ValueError("Radius cannot be negative!")
        self._radius = value
if __name__ == "__main__":
    circle = Circle(10)
    print(f"Initial radius: {circle.radius}")
    circle.radius = 15
    print(f"New radius: {circle.radius}")
```

要测试此示例，请运行以下命令：

```py
python3.12 ch01/encapsulate_bis.py
```

您应该得到以下输出：

```py
Initial radius: 10
New radius: 15
```

在这个第二个示例中，我们看到了如何封装圆的半径组件，以便在需要时更改技术细节，而不会破坏类。例如，*setter* 的验证代码可以演变。我们甚至可以更改基础属性 `_radius`，而我们的代码用户的行为将保持不变。

# 遵循“优先使用组合而非继承”原则

在面向对象编程中，通过继承创建复杂的类层次结构是很诱人的。虽然继承有其优点，但它可能导致代码紧密耦合，难以维护和扩展。这就是“优先使用组合而非继承”原则发挥作用的地方。

## 这是什么意思？

此原则建议您应该优先从更简单的部分组合对象，而不是从基类继承功能。换句话说，通过组合更简单的对象来构建复杂对象。

## 优点

选择组合而非继承提供了一些优点：

+   **灵活性**：组合允许您在运行时更改对象的行为，使代码更具适应性

+   **可重用性**：较小的、简单的对象可以在应用程序的不同部分重用，从而促进代码的可重用性

+   **易于维护**：使用组合，您可以轻松地替换或更新单个组件，而不会影响整体系统，避免边界效应

## 组合的技术

在 Python 中，组合通常通过面向对象编程实现，即在类中包含其他类的实例。这有时被称为被组合的类和被包含的类之间的“具有”关系。Python 通过不需要显式类型声明，特别容易使用组合。您可以通过在类的 `__init__` 方法中实例化它们或作为参数传递来包含其他对象。

## 例子 - 使用发动机组合汽车

在 Python 中，您可以通过在您的类中包含其他类的实例来实现组合。例如，考虑一个包含 `Engine` 类实例的 `Car` 类：

1.  让我们先定义如下所示的 `Engine` 类，其中包含其 `start` 方法：

    ```py
    class Engine:
        def start(self):
            print("Engine started")
    ```

1.  然后，让我们定义如下所示的 `Car` 类：

    ```py
    class Car:
        def __init__(self):
            self.engine = Engine()
        def start(self):
            self.engine.start()
            print("Car started")
    ```

1.  最后，在程序执行时，添加以下代码行以创建 `Car` 类的实例，并在该实例上调用 `start` 方法：

    ```py
    if __name__ == "__main__":
        my_car = Car()
        my_car.start()
    ```

此示例的完整代码（`ch01/composition.py`）如下所示：

```py
class Engine:
    def start(self):
        print("Engine started")
class Car:
    def __init__(self):
        self.engine = Engine()
    def start(self):
        self.engine.start()
        print("Car started")
if __name__ == "__main__":
    my_car = Car()
    my_car.start()
```

要测试代码，请运行以下命令：

```py
python3.12 ch01/composition.py
```

您应该得到以下输出：

```py
Engine started
Car started
```

正如您在这个示例中所看到的，`Car` 类由一个 `Engine` 对象组成，这是通过 `self.engine = Engine()` 这一行实现的，这使得您能够轻松地更换发动机类型，而无需更改 `Car` 类本身。

# 遵循“面向接口而非实现”原则

在软件设计中，很容易陷入如何实现一个功能的细节。然而，过分关注实现细节可能导致代码紧密耦合且难以修改。*面向接口，而非实现*的原则为解决这个问题提供了解决方案。

## 这是什么意思？

接口定义了类的一个**契约**，指定了一组必须实现的方法。

这个原则鼓励你针对接口而不是具体类进行编码。通过这样做，你将代码从提供所需行为的特定类解耦，使得在不影响系统其他部分的情况下更容易交换或扩展实现。

## 优点

面向接口编程提供了几个优点：

+   **灵活性**：你可以轻松地在不同的实现之间切换，而无需更改使用它们的代码

+   **可维护性**：失去特定实现中的代码使得更新或替换组件变得更加容易

+   **可测试性**：接口使得编写单元测试更加简单，因为在测试期间你可以轻松地模拟接口

## 接口技术

在 Python 中，*接口*可以通过两种主要技术实现：**抽象基类**（**ABCs**）和协议。

### 抽象基类

由`abc`模块提供的*ABCs*（抽象基类），允许你定义必须由任何具体（即非抽象）子类实现的*抽象方法*。

让我们通过一个示例来理解这个概念，我们将定义一个抽象类（作为接口）然后使用它：

1.  首先，我们需要按照以下方式导入`ABC`类和`abstractmethod`装饰器函数：

    ```py
    from abc import ABC, abstractmethod
    ```

1.  然后，我们定义接口类如下：

    ```py
    class MyInterface(ABC):
        @abstractmethod
        def do_something(self, param: str):
            pass
    ```

1.  现在，为该接口定义一个具体类；它从接口类继承并提供了`do_something`方法的实现，如下所示：

    ```py
    class MyClass(MyInterface):
        def do_something(self, param: str):
            print(f"Doing something with: '{param}'")
    ```

1.  为了测试目的，添加以下行：

    ```py
    if __name__ == "__main__":
        MyClass().do_something("some param")
    ```

完整的代码（`ch01/abstractclass.py`）如下：

```py
from abc import ABC, abstractmethod
class MyInterface(ABC):
    @abstractmethod
    def do_something(self, param: str):
        pass
class MyClass(MyInterface):
    def do_something(self, param: str):
        print(f"Doing something with: '{param}'")
if __name__ == "__main__":
    MyClass().do_something("some param")
```

要测试代码，请运行以下命令：

```py
python3.12 ch01/abstractclass.py
```

你应该得到以下输出：

```py
Doing something with: 'some param'
```

现在你已经知道如何在 Python 中定义接口和实现该接口的具体类。

### 协议

通过`typing`模块在 Python 3.8 中引入的*协议*，比 ABCs 提供了更灵活的方法，称为*结构化鸭子类型*，其中如果对象具有某些属性或方法，则认为它是有效的，而不管其实际继承关系如何。

与在运行时确定类型兼容性的传统鸭子类型不同，结构化鸭子类型允许在编译时进行类型检查。这意味着你可以在代码甚至运行之前（例如在 IDE 中）捕获类型错误，使你的程序更加健壮且更容易调试。

使用*协议*的关键优势是它们关注对象能做什么，而不是它是什么。换句话说，*如果一个对象像鸭子走路，像鸭子嘎嘎叫，那么它就是一只鸭子*，无论它的实际继承层次结构如何。这在像 Python 这样的动态类型语言中尤其有用，其中对象的行为比其实际类型更重要。

例如，你可以定义一个`Drawable`协议，它需要一个`draw()`方法。任何实现此方法的类都会隐式满足协议，而无需明确从它继承。

这里有一个快速示例来说明这个概念。假设你需要一个名为`Flyer`的协议，它需要一个`fly()`方法。你可以这样定义它：

```py
from typing import Protocol
class Flyer(Protocol):
    def fly(self) -> None:
        ...
```

就这样！现在，任何具有`fly()`方法的类都会被认为是`Flyer`，无论它是否明确地从`Flyer`类继承。这是一个强大的功能，允许你编写更通用和可重用的代码，并遵循我们之前在*遵循“优先使用组合而非继承”*原则部分讨论的原则。

在后面的示例中，我们将看到协议的实际应用。

## 举例说明 – 不同类型的记录器

使用 ABCs，让我们创建一个允许不同类型日志记录机制的日志接口。以下是实现方式：

1.  从`abc`导入所需的模块：

    ```py
    from abc import ABC, abstractmethod
    ```

1.  使用`log`方法定义`Logger`接口：

    ```py
    class Logger(ABC):
        @abstractmethod
        def log(self, message: str):
            pass
    ```

1.  然后，定义两个具体的类，它们实现了`Logger`接口，用于两种不同的`Logger`类型：

    ```py
    class ConsoleLogger(Logger):
        def log(self, message: str):
            print(f"Console: {message}")
    class FileLogger(Logger):
        def log(self, message: str):
            with open("log.txt", "a") as f:
                f.write(f"File: {message}\n")
    ```

1.  接下来，为了使用每种类型的记录器，请定义一个如下所示的功能：

    ```py
    def log_message(logger: Logger, message: str):
        logger.log(message)
    ```

    注意，该函数将其第一个参数作为类型为`Logger`的对象，这意味着一个实现了`Logger`接口的具体类的实例（即`ConsoleLogger`或`FileLogger`）。

1.  最后，添加测试代码所需的行，如下调用`log_message`函数：

    ```py
    if __name__ == "__main__":
        log_message(ConsoleLogger(), "A console log.")
        log_message(FileLogger(), "A file log.")
    ```

这个示例的完整代码（`ch01/interfaces.py`）如下所示：

```py
from abc import ABC, abstractmethod
class Logger(ABC):
    @abstractmethod
    def log(self, message: str):
        pass
class ConsoleLogger(Logger):
    def log(self, message: str):
        print(f"Console: {message}")
class FileLogger(Logger):
    def log(self, message: str):
        with open("log.txt", "a") as f:
            f.write(f"File: {message}\n")
def log_message(logger: Logger, message: str):
    logger.log(message)
if __name__ == "__main__":
    log_message(ConsoleLogger(), "A console log.")
    log_message(FileLogger(), "A file log.")
```

要测试代码，请运行以下命令：

```py
python3.12 ch01/interfaces.py
```

你应该得到以下输出：

```py
Console: A console log.
```

除了那个输出之外，查看你运行命令的文件夹，你会发现一个名为`log.txt`的文件已经被创建，其中包含以下行：

```py
File: A file log.
```

正如你刚才在`log_message`函数中看到的，你可以轻松地在不同的日志记录机制之间切换，而无需更改函数本身。

## 举例说明 – 使用协议的不同类型的记录器

让我们用*协议*方式重新审视之前的示例：

1.  首先，我们需要如下导入`Protocol`类：

    ```py
    from typing import Protocol
    ```

1.  然后，通过从`Protocol`类继承来定义`Logger`接口，如下所示：

    ```py
    class Logger(Protocol):
        def log(self, message: str):
            ...
    ```

    并且其余的代码保持不变。

因此，完整的代码（`ch01/interfaces_bis.py`）如下所示：

```py
from typing import Protocol
class Logger(Protocol):
    def log(self, message: str):
        ...
class ConsoleLogger:
    def log(self, message: str):
        print(f"Console: {message}")
class FileLogger:
    def log(self, message: str):
        with open("log.txt", "a") as f:
            f.write(f"File: {message}\n")
def log_message(logger: Logger, message: str):
    logger.log(message)
if __name__ == "__main__":
    log_message(ConsoleLogger(), "A console log.")
    log_message(FileLogger(), "A file log.")
```

要根据我们定义的协议检查代码的静态类型，请运行以下命令：

```py
mypy ch01/interfaces_bis.py
```

你应该得到以下输出：

```py
Success: no issues found in 1 source file
```

要测试代码，请运行以下命令：

```py
python3.12 ch01/interfaces_bis.py
```

你应该得到与运行上一个版本相同的结果——换句话说，创建的`log.txt`文件和 shell 中的以下输出：

```py
Console: A console log.
```

这是正常的，因为我们唯一改变的是定义接口的方式。而且，接口（协议）的效果在运行时没有强制执行，这意味着它不会改变代码执行的实际情况。

# 遵循松散耦合原则

随着软件的复杂性增加，其组件之间的关系可能会变得复杂，导致系统难以理解、维护和扩展。松散耦合原则旨在减轻这一问题。

## 这是什么意思？

松散耦合指的是最小化程序不同部分之间的依赖关系。在松散耦合系统中，组件是独立的，并通过定义良好的接口进行交互，这使得修改一个部分而不影响其他部分变得更容易。

## 优点

松散耦合提供了几个优点：

+   **可维护性**：由于依赖项较少，更新或替换单个组件更容易

+   **可扩展性**：松散耦合的系统可以更容易地通过添加新功能或组件来扩展

+   **可测试性**：独立的组件更容易在隔离状态下进行测试，从而提高软件的整体质量

## 松散耦合的技术

实现松散耦合的两个主要技术是**依赖注入**和**观察者模式**。依赖注入允许组件从外部源接收其依赖项，而不是创建它们，这使得交换或模拟这些依赖项更容易。另一方面，观察者模式允许一个对象发布其状态的变化，以便其他对象可以相应地做出反应，而无需紧密绑定在一起。

这两种技术都旨在减少组件之间的相互依赖性，使你构建的系统更加模块化，更容易管理。

我们将在*第五章*中详细讨论*观察者模式*，*行为设计模式*。现在，让我们通过一个例子来了解如何使用*依赖注入*技术。

## 举例来说，一个消息服务

在 Python 中，你可以通过使用*依赖注入*来实现松散耦合。让我们看看一个涉及`MessageService`类的简单例子：

1.  首先，我们定义`MessageService`类如下：

    ```py
    class MessageService:
        def __init__(self, sender):
            self.sender = sender
        def send_message(self, message):
            self.sender.send(message)
    ```

    如你所见，该类将通过传递一个发送对象给它来初始化；该对象有一个`send`方法，允许发送消息。

1.  第二，让我们定义一个`EmailSender`类：

    ```py
    class EmailSender:
        def send(self, message):
            print(f"Sending email: {message}")
    ```

1.  第三，让我们定义一个`SMSSender`类：

    ```py
    class SMSSender:
        def send(self, message):
            print(f"Sending SMS: {message}")
    ```

1.  现在我们可以使用一个`EmailSender`对象实例化`MessageService`并使用它来发送消息。我们也可以使用一个`SMSSender`对象来实例化`MessageService`。我们添加了以下代码来测试这两个操作：

    ```py
    if __name__ == "__main__":
        email_service = MessageService(EmailSender())
        email_service.send_message("Hello via Email")
        sms_service = MessageService(SMSSender())
        sms_service.send_message("Hello via SMS")
    ```

此示例的完整代码，保存在`ch01/loose_coupling.py`文件中，如下所示：

```py
class MessageService:
    def __init__(self, sender):
        self.sender = sender
    def send_message(self, message: str):
        self.sender.send(message)
class EmailSender:
    def send(self, message: str):
        print(f"Sending email: {message}")
class SMSSender:
    def send(self, message: str):
        print(f"Sending SMS: {message}")
if __name__ == "__main__":
    email_service = MessageService(EmailSender())
    email_service.send_message("Hello via Email")
    sms_service = MessageService(SMSSender())
    sms_service.send_message("Hello via SMS")
```

要测试代码，请运行以下命令：

```py
python3.12 ch01/loose_coupling.py
```

你应该得到以下输出：

```py
Sending email: Hello via Email
Sending SMS: Hello via SMS
```

在这个例子中，`MessageService`通过依赖注入与`EmailSender`和`SMSSender`松散耦合。这允许你轻松地在不同的发送机制之间切换，而无需修改`MessageService`类。

# 摘要

我们从本书开始，介绍了开发者应该遵循的基础设计原则，以编写可维护、灵活和健壮的软件。从封装变化的部分到偏好组合、面向接口编程以及追求松散耦合，这些原则为任何 Python 开发者提供了一个强大的基础。

正如你所见，这些原则不仅仅是理论上的构建，而是可以显著提高你代码质量的实用指南。它们为接下来要讨论的内容奠定了基础：深入探讨更多专门化的原则集合，这些原则指导着面向对象的设计。

在下一章中，我们将深入探讨 SOLID 原则，这是一组旨在使软件设计更易于理解、灵活和可维护的五个设计原则。
