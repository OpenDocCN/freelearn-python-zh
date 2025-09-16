

# 第二章：SOLID 原则

在软件工程的世界里，原则和最佳实践是构建健壮、可维护和高效代码库的基石。在前一章中，我们介绍了每个开发者都需要遵循的基础原则。

在本章中，我们继续探讨设计原则，重点关注由罗伯特·C·马丁提出的 SOLID，这是一个代表他提出的五个设计原则的首字母缩略词，旨在使软件更易于理解、灵活和可维护。

本章中，我们将涵盖以下主要内容：

+   **单一职责原则**（**SRP**）

+   **开放封闭原则**（**OCP**）

+   **里氏替换原则**（**LSP**）

+   **接口隔离原则**（**ISP**）

+   **依赖倒置原则**（**DIP**）

到本章结束时，你将理解这五个额外的设计原则以及如何在 Python 中应用它们。

# 技术要求

请参阅第一章中提出的要求。

# SRP

SRP 是软件设计中的一个基本概念。它主张在定义一个类以提供功能时，该类应该只有一个存在的理由，并且只负责功能的一个方面。用更简单的话说，它提倡每个类应该有一个工作或职责，并且这个工作应该封装在该类中。

因此，通过遵循 SRP，你实际上是在努力使类专注于功能、具有凝聚力和专业化。这种方法在提高代码库的可维护性和可理解性方面发挥着至关重要的作用。当每个类都有一个明确且单一的目的时，它就更容易管理、理解和扩展。

当然，你没有义务遵循 SRP。但了解这个原则，并带着这个想法思考你的代码，将随着时间的推移提高你的代码库。

在实践中，应用单一职责原则（SRP）通常会导致更小、更专注的类，这些类可以组合和组合以创建复杂的系统，同时保持清晰和有序的结构。

注意

SRP 并非关于最小化类中的代码行数，而是确保一个类只有一个改变的理由，减少在修改时产生意外副作用的可能性。

让我们通过一个小例子来使事情更清晰。

## 跟随 SRP 的软件设计示例

让我们想象一些可以在许多不同类型的应用程序中使用的代码，例如内容或文档管理工具或专门的 Web 应用程序，这些应用程序包括生成 PDF 文件并将其保存到磁盘的功能。为了帮助理解 SRP（单一职责原则），让我们考虑一个初始版本，其中代码不遵循此原则。在这种情况下，开发者可能会定义一个处理报告的类，称为`Report`，并以使其负责生成报告并保存到文件的方式实现它。此类典型的代码可能如下所示：

```py
class Report:
    def __init__(self, content):
        self.content = content
    def generate(self):
        print(f"Report content: {self.content}")
    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            file.write(self.content)
```

如您所见，`Report`类有两个职责。首先，生成报告，然后，将报告内容保存到文件。

当然，这是可以的。但设计原则鼓励我们考虑为未来改进事物，因为需求会演变，代码会增长以处理复杂性和变化。在这里，SRP（单一职责原则）教导我们分离事物。为了遵循 SRP，我们可以重构代码以使用两个不同的类，每个类将各自有一个职责，如下所示：

1.  创建第一个类，负责生成报告内容：

    ```py
    class Report:
        def __init__(self, content: str):
            self.content: str = content
        def generate(self):
            print(f"Report content: {self.content}")
    ```

1.  创建一个第二类来处理将报告保存到文件的需求：

    ```py
    class ReportSaver:
        def __init__(self, report: Report):
            self.report: Report = report
        def save_to_file(self, filename: str):
            with open(filename, 'w') as file:
                file.write(self.report.content)
    ```

1.  为了确认我们的重构版本可以正常工作，让我们添加以下代码以便立即进行测试：

    ```py
    if __name__ == "__main__":
        report_content = "This is the content."
        report = Report(report_content)
        report.generate()
        report_saver = ReportSaver(report)
        report_saver.save_to_file("report.txt")
    ```

为了总结，以下是完整的代码，保存在`ch02/srp.py`文件中：

```py
class Report:
    def __init__(self, content: str):
        self.content: str = content
    def generate(self):
        print(f"Report content: {self.content}")
class ReportSaver:
    def __init__(self, report: Report):
        self.report: Report = report
    def save_to_file(self, filename: str):
        with open(filename, "w") as file:
            file.write(self.report.content)
if __name__ == "__main__":
    report_content = "This is the content."
    report = Report(report_content)
    report.generate()
    report_saver = ReportSaver(report)
    report_saver.save_to_file("report.txt")
```

要查看代码的结果，请运行以下命令：

```py
python ch02/srp.py
```

您将得到以下输出：

```py
report.txt file has been created. So, everything works as expected.
			As you can see, by following the SRP, you can achieve cleaner, more maintainable, and adaptable code, which contributes to the overall quality and longevity of your software projects.
			OCP
			The OCP is another fundamental principle in software design. It emphasizes that software entities, such as classes and modules, should be open for extension but closed for modification. What does that mean? It means that once a software entity is defined and implemented, it should not be changed to add new functionality. Instead, the entity should be extended through inheritance or interfaces to accommodate new requirements and behaviors.
			When thinking about this principle and if you have some experience writing code for non-trivial programs, you can see how it makes sense, since modifying an entity introduces a risk of breaking some other part of the code base relying on it.
			The OCP provides a robust foundation for building flexible and maintainable software systems. It allows developers to introduce new features or behaviors without altering the existing code base. By adhering to the OCP, you can minimize the risk of introducing bugs or unintended side effects when making changes to your software.
			An example of design following the OCP
			Consider a `Rectangle` class defined for rectangle shapes. Let’s say we add a way to calculate the area of different shapes, maybe by using a function. The hypothetical code for the definition of both the class and the function could look like the following:

```

class Rectangle:

def __init__(self, width:float, height: float):

self.width: float = width

self.height: float = height

def calculate_area(shape) -> float:

if isinstance(shape, Rectangle):

return shape.width * shape.height

```py

			Note
			This code is not in the example code files. It is a hypothetical idea to start with in our thinking, and not the code you would end up using. Keep reading.
			Given that code, if we want to add more shapes, we have to modify the `calculate_area` function. That is not ideal as we will keep coming back to change that code and that means more time testing things to avoid bugs.
			As we aim to become good at writing maintainable code, let’s see how we could improve that code by adhering to the OCP, while extending it to support another type of shape, the circle (using a `Circle` class):

				1.  Start by importing what we will need:

    ```

    import math

    from typing import Protocol

    ```py

    				2.  Define a `Shape` protocol for an interface providing a method for the shape’s area:

    ```

    class Shape(Protocol):

    def area(self) -> float:

    ...

    ```py

			Note
			Refer to *Chapter 1*, *Foundational Design Principles*, to understand Python’s `Protocol` concept and technique.

				1.  Define the `Rectangle` class, which conforms to the `Shape` protocol:

    ```

    class Rectangle:

    def __init__(self, width: float, height: float):

    self.width: float = width

    self.height: float = height

    def area(self) -> float:

    return self.width * self.height

    ```py

    				2.  Also define the `Circle` class, which also conforms to the `Shape` protocol:

    ```

    class Circle:

    def __init__(self, radius: float):

    self.radius: float = radius

    def area(self) -> float:

    return math.pi * (self.radius**2)

    ```py

    				3.  Implement the `calculate_area` function in such a way that adding a new shape won’t require us to modify it:

    ```

    def calculate_area(shape: Shape) -> float:

    return shape.area()

    ```py

    				4.  Add some code for testing the `calculate_area` function on the two types of shape objects:

    ```

    if __name__ == "__main__":

    rect = Rectangle(12, 8)

    rect_area = calculate_area(rect)

    print(f"Rectangle area: {rect_area}")

    circ = Circle(6.5)

    circ_area = calculate_area(circ)

    print(f"Circle area: {circ_area:.2f}")

    ```py

			The following is the complete code for this example, saved in the `ch02/ocp.py` file:

```

import math

from typing import Protocol

class Shape(Protocol):

def area(self) -> float:

...

class Rectangle:

def __init__(self, width: float, height: float):

self.width: float = width

self.height: float = height

def area(self) -> float:

return self.width * self.height

class Circle:

def __init__(self, radius: float):

self.radius: float = radius

def area(self) -> float:

return math.pi * (self.radius**2)

def calculate_area(shape: Shape) -> float:

return shape.area()

if __name__ == "__main__":

rect = Rectangle(12, 8)

`rect_area = calculate_area(rect)`

`print(f"Rectangle area: {rect_area}")`

`circ = Circle(6.5)`

`circ_area = calculate_area(circ)`

`print(f"Circle area: {circ_area:.2f}")`

```py

			To see the result of this code, run the following command:

```

`python ch02/ocp.py`

```py

			You should get the following output:

```

`Rectangle area: 96`

`calculate_area` 函数。新的设计优雅，并且由于遵循了 OCP，易于维护。

            因此，你现在已经发现了另一个你应该每天使用的原则，它既促进了适应不断变化的需求的设计，又保持了现有功能的不变性。

            LSP

            LSP 是面向对象编程中的另一个基本概念。它规定了子类应该如何与它们的超类相关联。根据 LSP，如果一个程序使用超类的对象，那么用子类的对象替换这些对象不应该改变程序的正确性和预期的行为。

            遵循这一原则对于保持软件系统的健壮性非常重要。它确保在使用继承时，子类在不改变其外部行为的情况下扩展其父类。例如，如果一个函数与超类对象一起工作正确，那么它也应该与这个超类的任何子类对象一起工作正确。

            LSP 允许开发者引入新的子类类型，而不会破坏现有功能的风险。这在大型系统中尤为重要，因为一个部分的更改可能会影响系统的其他部分。通过遵循 LSP，开发者可以安全地修改和扩展类，知道他们的新子类将与既定的层次结构和功能无缝集成。

            LSP 设计的一个例子

            让我们考虑一个 `Bird` 类和一个继承自它的 `Penguin` 类：

```py
class Bird:
    def fly(self):
        print("I can fly")
class Penguin(Bird):
    def fly(self):
        print("I can't fly")
```

            然后，为了满足一个假设的使鸟类飞行的程序的需求，我们添加了一个 `make_bird_fly` 函数：

```py
def make_bird_fly(bird):
    bird.fly()
```

            根据当前代码，我们可以看到，如果我们向函数传递 `Bird` 类的实例，我们会得到预期的行为（“鸟会飞”），而如果我们传递 `Penguin` 类的实例，我们会得到另一种行为（“它不会飞”）。你可以分析 `ch02/lsp_violation.py` 文件中提供的代表这种第一个设计代码，并运行它来测试这个结果。这至少给我们提供了 LSP 希望帮助我们避免的直觉。那么，我们如何通过遵循 LSP 来改进设计呢？

            为了遵循 LSP，我们可以重构代码并引入新的类，以确保行为保持一致：

                1.  我们保留 `Bird` 类，但使用更好的方法来表示我们想要的行为；让我们称它为 `move()`。现在这个类将看起来如下：

    ```py
    class Bird:
        def move(self):
            print("I'm moving")
    ```

                    1.  然后，我们引入一个 `FlyingBird` 类和一个 `FlightlessBird` 类，它们都继承自 `Bird` 类：

    ```py
    class FlyingBird(Bird):
        def move(self):
            print("I'm flying")
    class FlightlessBird(Bird):
        def move(self):
            print("I'm walking")
    ```

                    1.  现在，`make_bird_move` 函数可以定义为以下内容：

    ```py
    def make_bird_move(bird):
        bird.move()
    ```

                    1.  如往常一样，我们添加一些必要的代码来测试设计：

    ```py
    if __name__ == "__main__":
        generic_bird = Bird()
        eagle = FlyingBird()
        penguin = FlightlessBird()
        make_bird_move(generic_bird)
        make_bird_move(eagle)
        make_bird_move(penguin)
    ```

            这个新设计的完整代码，保存在 `ch02/lsp.py` 文件中，如下所示：

```py
class Bird:
    def move(self):
        print("I'm moving")
class FlyingBird(Bird):
    def move(self):
        print("I'm flying")
class FlightlessBird(Bird):
    def move(self):
        print("I'm walking")
def make_bird_move(bird):
    bird.move()
if __name__ == "__main__":
    generic_bird = Bird()
    eagle = FlyingBird()
    penguin = FlightlessBird()
    make_bird_move(generic_bird)
    make_bird_move(eagle)
    make_bird_move(penguin)
```

            To test the example, run the following command:

```py
python ch02/lsp.py
```

            You should get the following output:

```py
I'm moving
I'm flying
Bird class with a Penguin class or with an Eagle class; that is, each object moves whether it is an instance of a Bird class or an instance of a subclass. And that result was possible thanks to following the LSP.
			This example demonstrates that all subclasses (`FlyingBird` and `FlightlessBird`) can be used in place of their superclass (`Bird`) without disrupting the expected behavior of the program. This conforms to the LSP.
			ISP
			The ISP advocates for designing smaller, more specific interfaces rather than broad, general-purpose ones. This principle states that a class should not be forced to implement interfaces it does not use. In the context of Python, this implies that a class shouldn’t be forced to inherit and implement methods that are irrelevant to its purpose.
			The ISP suggests that when designing software, one should avoid creating large, monolithic interfaces. Instead, the focus should be on creating smaller, more focused interfaces. This allows classes to only inherit or implement what they need, ensuring that each class only contains relevant and necessary methods.
			Following this principle helps us build software with modularity, code readability and maintainability qualities, reduced side effects, and software that benefits from easier refactoring and testing, among other things.
			An example of design following the ISP
			Let’s consider an `AllInOnePrinter` class that implements functionalities for printing, scanning, and faxing documents. The definition for that class would look like the following:

```

class AllInOnePrinter:

def print_document(self):

print("打印中")

def scan_document(self):

print("扫描中")

def fax_document(self):

print("发送传真")

```py

			If we wanted to introduce a specialized `SimplePrinter` class that only prints, it would have to implement or inherit the `scan_document` and `fax_document` methods (even though it only prints). That is not ideal.
			To adhere to the ISP, we can create a separate interface for each functionality so that each class implements only the interfaces it needs.
			Note about interfaces
			Refer to the presentation in *Chapter 1*, *Foundational Design Principles*, of the **program to interfaces, not implementations principle**, to understand the importance of interfaces and the techniques we use in Python to define them (abstract base classes, protocols, etc.). In particular, here is the situation where protocols are the natural answer, that is, they help define small interfaces where each interface is created for doing only one thing.

				1.  Let’s start by defining the three interfaces:

    ```

    from typing import Protocol

    class Printer(Protocol):

    def print_document(self):

    ...

    class Scanner(Protocol):

    def scan_document(self):

    ...

    class Fax(Protocol):

    def fax_document(self):

    ...

    ```py

    				2.  Then, we keep the `AllInOnePrinter` class, which already implements the interfaces:

    ```

    class AllInOnePrinter:

    def print_document(self):

    print("打印中")

    def scan_document(self):

    print("扫描中")

    def fax_document(self):

    print("发送传真")

    ```py

    				3.  We add the `SimplePrinter` class, implementing the `Printer` interface, as follows:

    ```

    class SimplePrinter:

    def print_document(self):

    print("简单打印")

    ```py

    				4.  We also add a function that, when passed an object that implements the `Printer` interface, calls the right method on it to do the printing:

    ```

    def do_the_print(printer: Printer):

    printer.print_document()

    ```py

    				5.  Finally, we add code for testing the classes and the implemented interfaces:

    ```

    if __name__ == "__main__":

    all_in_one = AllInOnePrinter()

    all_in_one.scan_document()

    all_in_one.fax_document()

    do_the_print(all_in_one)

    simple = SimplePrinter()

    do_the_print(simple)

    ```py

			Here is the complete code for this new design (`ch02/isp.py`):

```

from typing import Protocol

class Printer(Protocol):

def print_document(self):

...

class Scanner(Protocol):

def scan_document(self):

...

class Fax(Protocol):

def fax_document(self):

...

class AllInOnePrinter:

def print_document(self):

print("打印中")

def scan_document(self):

print("扫描中")

def fax_document(self):

print("发送传真")

class SimplePrinter:

def print_document(self):

print("简单打印")

def do_the_print(printer: Printer):

printer.print_document()

if __name__ == "__main__":

all_in_one = AllInOnePrinter()

all_in_one.scan_document()

all_in_one.fax_document()

do_the_print(all_in_one)

simple = SimplePrinter()

do_the_print(simple)

```py

			To test this code, run the following command:

```

python ch02/isp.py

```py

			You will get the following output:

```

扫描中

发送传真

Printing

简单打印

```py

			Because of the new design, each class only needs to implement the methods relevant to its behavior. This illustrates the ISP.
			DIP
			The DIP advocates that high-level modules should not depend directly on low-level modules. Instead, both should depend on abstractions or interfaces. By doing so, you decouple the high-level components from the details of the low-level components.
			This principle allows for the reduction of the coupling between different parts of the system you are building, making it more maintainable and extendable, as we will see in an example.
			Following the DIP brings loose coupling within a system because it encourages the use of interfaces as intermediaries between different parts of the system. When high-level modules depend on interfaces, they remain isolated from the specific implementations of low-level modules. This separation of concerns enhances maintainability and extensibility.
			In essence, the DIP is closely linked to the loose coupling principle, which was covered in *Chapter 1*, *Foundational Design Principles*, by promoting a design where components interact through interfaces rather than concrete implementations. This reduces the interdependencies between modules, making it easier to modify or extend one part of the system without affecting others.
			An example of design following the ISP
			Consider a `Notification` class responsible for sending notifications via email, using an `Email` class. The code for both classes would look like the following:

```

class Email:

def send_email(self, message):

print(f"发送邮件: {message}")

class Notification:

def __init__(self):

self.email = Email()

def send(self, message):

self.email.send_email(message)

```py

			Note about the code
			This is not yet the final version of the example.
			Currently, the high-level `Notification` class is dependent on the low-level `Email` class, and that is not ideal. To adhere to the DIP, we can introduce an abstraction, with a new code, as follows:

				1.  Define a `MessageSender` interface:

    ```

    from typing import Protocol

    class MessageSender(Protocol):

    def send(self, message: str):

    ...

    ```py

    				2.  Define the `Email` class, which implements the `MessageSender` interface, as follows:

    ```

    class Email:

    def send(self, message: str):

    print(f"发送邮件: {message}")

    ```py

    				3.  Define the `Notification` class, which also implements the `MessageSender` interface, and has an object that implements `MessageSender` stored in its `sender` attribute, for handling the actual message sending. The code for that definition is as follows:

    ```

    class Notification:

    def __init__(self, sender: MessageSender):

    self.sender: MessageSender = sender

    def send(self, message: str):

    self.sender.send(message)

    ```py

    				4.  Finally, add some code for testing the design:

    ```

    if __name__ == "__main__":

    email = Email()

    notif = Notification(sender=email)

    notif.send(message="这是消息。")

    ```py

			The complete code for the implementation we just proposed is as follows (`ch02/dip.py`):

```

from typing import Protocol

class MessageSender(Protocol):

def send(self, message: str):

...

class Email:

def send(self, message: str):

print(f"发送邮件: {message}")

class Notification:

def __init__(self, sender: MessageSender):

self.sender = sender

def send(self, message: str):

self.sender.send(message)

if __name__ == "__main__":

email = Email()

notif = Notification(sender=email)

notif.send(message="这是消息。")

```py

			To test the code, run the following command:

```

python ch02/dip.py

```py

			You should get the following output:

```

Notification and Email are based on the MessageSender abstraction, so this design adheres to the DIP.

            Summary

            在本章中，我们探讨了比在*第一章*“基础设计原则”中介绍的原则更多的内容。理解和应用 SOLID 原则对于编写可维护、健壮和可扩展的 Python 代码至关重要。这些原则为良好的软件设计提供了坚实的基础，使得管理复杂性、减少错误和提升代码的整体质量变得更加容易。

            在下一章中，我们将开始探索 Python 中的设计模式，这是追求卓越的 Python 开发者不可或缺的一个主题。

```py

```

```py

```

# 第二部分：来自四人帮

本部分探讨了来自四人帮（GoF）的经典设计模式，这些模式用于解决日常问题，以及如何作为 Python 开发者应用它们。本部分包括以下章节：

+   *第三章*“创建型设计模式”

+   *第四章*“结构设计模式”

+   *第五章*“行为设计模式”
