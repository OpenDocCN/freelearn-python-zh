# 12

# 类型提示简介

> “知己知彼，百战不殆。”
> 
> – 亚里士多德

在本章中，我们将探讨**类型提示**这一主题。类型提示可能是自 Python 2.2 以来 Python 引入的最大变化，它实现了类型和类的统一。

具体来说，我们将研究以下主题：

+   Python 对类型的处理方法。

+   可用于注解的类型。

+   协议（简要介绍）。

+   Mypy，Python 的静态类型检查器。

# Python 对类型的处理方法

Python 是一种**强类型**和**动态类型**的语言。

*强类型*意味着 Python 不允许可能导致意外行为的隐式类型转换。考虑以下**php**代码：

```py
<?php
$a = 2;
$b = "2";
echo $a + $b; // prints: 4
?> 
```

在 php 中，变量前面有一个`$`符号。在上面的代码中，我们将`$a`设置为整数`2`，将`$b`设置为字符串`"2"`。要将它们相加，php 会执行从字符串到整数的隐式转换。这被称为**类型魔术**。这看起来可能很方便，但 php 是弱类型的，这可能导致代码中的 bug。如果我们尝试在 Python 中做同样的事情，结果将大不相同：

```py
# example.strongly.typed.py
a = 2
b = "2"
print(a + b) 
```

运行上述代码会产生：

```py
$ python ch12/example.strongly.typed.py
Traceback (most recent call last):
  File "ch12/example.strongly.typed.py", line 3, in <module>
    print(a + b)
          ~~^~~
TypeError: unsupported operand type(s) for +: 'int' and 'str' 
```

Python 是强类型的，所以当我们尝试将整数加到字符串上——或者任何不兼容类型的组合——我们会得到`TypeError`。

*动态类型*意味着 Python 在运行时确定变量的类型，这意味着我们不需要在代码中显式指定类型。

相比之下，像 C++、Java、C#和 Swift 这样的语言都是**静态类型**的。当我们在这类语言中声明变量时，我们必须指定它们的类型。例如，在 Java 中，常见的变量声明如下：

```py
String name = "John Doe";
int age = 60; 
```

两种方法都有优点和缺点，所以很难说哪一种最好。Python 被设计成简洁、精炼和优雅。其设计的一个优点就是**鸭子类型**。

## 鸭子类型

另一个 Python 帮助普及的概念是**鸭子类型**。本质上，这意味着一个对象的数据类型或类不如它定义的方法或支持的运算重要。俗话说：“如果它看起来像鸭子，游泳像鸭子，叫起来像鸭子，那么它可能就是一只鸭子。”

由于 Python 是动态类型的语言，鸭子类型在 Python 中被广泛使用。它提供了更大的灵活性和代码重用。考虑以下示例：

```py
# duck.typing.py
class Circle:
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14159 * (self.radius**2)
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height
def print_shape_info(shape):
    print(f"{shape.__class__.__name__} area: {shape.area()}")
circle = Circle(5)
rectangle = Rectangle(4, 6)
print_shape_info(circle)  # Circle area: 78.53975
print_shape_info(rectangle)  # Rectangle area: 24 
```

在上述代码中，`print_shape_info()`函数不关心`shape`的具体类型。它只关心`shape`有一个名为`area()`的方法。

# 类型提示的历史

虽然 Python 对类型的处理方法是导致其广泛采用成功的一个特征，但在 Python 3 中，我们看到了一个逐渐且精心设计的演变，旨在整合类型安全特性，同时保持 Python 的动态特性。

这始于 Python 3.0，随着函数注解的引入，由 PEP 3107 ([`peps.python.org/pep-3107/`](https://peps.python.org/pep-3107/)) 提出。这一新增功能允许开发者向函数参数和返回值添加任意元数据。这些注解最初是作为文档工具而设计的，没有语义意义。这一步是引入显式类型提示支持的基础层。

在 Python 3.5 中，随着 PEP 484 ([`peps.python.org/pep-0484/`](https://peps.python.org/pep-0484/)) 的落地，类型提示的真正起点出现了。PEP 484 正式化了类型提示的添加，建立在 PEP 3107 提出的语法之上。它定义了声明函数参数、返回值和变量类型的标准方式。

这使得可选的静态类型检查成为可能，开发者现在可以使用 **Mypy** 等工具在运行前检测类型相关的错误。

在 Python 3.6 中，我们看到了变量声明注解的引入。这是由 PEP 526 ([`peps.python.org/pep-0526/`](https://peps.python.org/pep-0526/)) 带来的。这一新功能意味着类型可以在整个代码中显式声明，而不仅仅是函数中。它还包括类属性和模块级变量。这进一步提高了 Python 的类型提示能力，并使得静态分析代码变得更加容易。

随后的增强和 PEP 进一步精炼和扩展了 Python 的类型系统。主要的有以下几项：

+   PEP 544 ([`peps.python.org/pep-0544/`](https://peps.python.org/pep-0544/)) 在 Python 3.8 中落地，引入了协议的概念，这实现了鸭子类型和进一步的静态类型检查。

+   PEP 585 ([`peps.python.org/pep-0585/`](https://peps.python.org/pep-0585/)) 在 Python 3.9 中落地，并设立了另一个里程碑。它通过直接与 Python 核心集合集成，彻底改变了类型提示。这消除了从 `typing` 模块导入类型以用于常见数据结构（如字典和列表）的需要。

+   PEP 586 ([`peps.python.org/pep-0586/`](https://peps.python.org/pep-0586/)) 在 Python 3.8 中落地，并添加了字面量类型，允许函数指定字面量值作为参数。

+   PEP 589 ([`peps.python.org/pep-0589/`](https://peps.python.org/pep-0589/)) 在 Python 3.8 中落地，引入了 `TypedDict`，它为具有固定键集的字典提供了精确的类型提示。

+   PEP 604 ([`peps.python.org/pep-0604/`](https://peps.python.org/pep-0604/)) 在 Python 3.10 中落地，简化了联合类型的语法，从而简化了注解并提高了代码的可读性。

其他值得注意的 PEP 包括：

+   PEP 561 ([`peps.python.org/pep-0561/`](https://peps.python.org/pep-0561/)) 规定了如何分发支持类型检查的包。

+   PEP 563 ([`peps.python.org/pep-0563/`](https://peps.python.org/pep-0563/))，它改变了注解的评估方式，使得它们不在函数定义时进行评估。这种延迟在 Python 3.10 中被设置为默认行为。

+   PEP 593 ([`peps.python.org/pep-0593/`](https://peps.python.org/pep-0593/))，它介绍了一种方法来增强现有的类型提示，使用任意元数据，可能用于第三方工具。

+   PEP 612 ([`peps.python.org/pep-0612/`](https://peps.python.org/pep-0612/))，它引入了参数规范，允许更复杂的变量注解类型，特别是对于修改函数签名的装饰器非常有用。

+   PEP 647 ([`peps.python.org/pep-0647/`](https://peps.python.org/pep-0647/))，它引入了类型守卫，这些函数可以在条件块中实现更精确的类型推断。

+   PEP 673 ([`peps.python.org/pep-0673/`](https://peps.python.org/pep-0673/))，它引入了 `Self` 类型来表示类体和方法返回中的实例类型，使得涉及类的类型提示更加表达性和准确。

Python 类型提示的演变是由对代码更强大鲁棒性和可扩展性的需求所驱动的，利用了 Python 的动态特性。

虽然 Python 的类型提示流行度似乎持续增加，但重要的是要注意，根据 PEP 484（Guido van Rossum、Jukka Lehtosalo 和 Łukasz Langa）的作者：

> “Python 将继续是一种动态类型语言，作者们没有意愿将类型提示强制化，即使是按照惯例。”

Python 类型提示的引入方式，以及该 PEP 的作者和 Python 的主要开发者的哲学方法，表明使用类型提示的选择，现在和将来都将由开发者决定。

让我们看看类型提示的一些主要好处。

# 类型提示的好处

采用类型提示带来了一些关键好处，如增强代码质量、可维护性和提高开发者效率。我们可以将它们总结成一个列表：

+   **改进代码可读性和文档**：类型提示是一种文档形式。它们阐明了函数期望的参数类型以及它返回的类型。这有助于开发者立即理解代码，无需阅读冗长的注释或大量代码。

+   **增强错误检测**：静态类型检查器，如 Mypy，可以扫描代码库并在运行前标记错误。这意味着一些错误可以在它们成为问题之前被消除。

+   **更好的 IDE 体验和自动完成**：现代 IDE 利用类型提示提供更好的自动完成和增强重构功能。此外，有了类型提示，IDE 可以建议对象适当的方法和属性。

+   **改进的协作和代码审查**：类型提示的文档质量使得一眼就能理解代码，这在阅读 pull request 中的更改时可能很有用。

+   **代码灵活性和可重用性**：Python 的类型提示包括泛型、自定义类型和协议等特性，这些特性有助于开发者编写结构更清晰、更灵活的代码。

Python 类型提示系统的另一个重要方面是它可以逐步引入。实际上，在代码库中逐步引入类型提示是很常见的，最初将努力限制在最重要的地方。

# 类型注释

现在我们已经对 Python 的动态特性和其类型提示方法有了基础的了解，让我们开始探索一些示例和概念，看看它在实践中是如何应用的。

虽然我们展示了我们可以用来注释代码的主要类型，但如果您之前在代码中使用过类型注释，您可能会发现一些细微的差异。这很可能是由于 Python 中的类型提示目前仍在不断发展，因此它根据您使用的 Python 版本而有所不同。在本章中，我们将坚持 Python 3.12 的规则。

让我们从基本示例开始。

## 函数注释

在 Python 中，我们可以注释函数参数及其返回值。让我们从一个基本的`greeter`函数开始：

```py
# annotations/basic.py
def greeter(name):
    return f"Hello, {name}!" 
```

这是一个简单的 Python 函数，它接受一个`name`并返回一个问候语。我们现在可以注释这个函数，指定我们期望`name`是一个字符串，并且返回的问候语也将是一个字符串。

```py
# annotations/basic.py
def **greeter_annotated****(name:** **str****) ->** **str****:**
    return f"Hello, {name}!" 
```

如您从代码中高亮显示的部分所示，注释函数的语法很简单。我们使用参数名后的冒号来指定类型。`return`值通过一个箭头（`->`）后跟返回的对象类型进行注释。

现在我们添加另一个参数`age`，我们期望它是一个整数。

```py
# annotations/basic.py
def greeter_annotated_age(name: str, age: int) -> str:
    return f"Hello, {name}! You are {age} years old." 
```

如您所预期的那样，我们对`age`参数做了同样的处理，只是这次我们指定了`int`，而不是`str`。

如果您使用的是相对现代的 IDE，请尝试运行此代码。如果您输入`name`或`age`然后输入一个点（`.`），编辑器应该只会建议与您正在使用的对象类型相关的方法和属性。

这是一个基本示例，旨在向您展示类型注释的语法。请忽略人工命名的部分，如`greeter_annotated()`和`greeter_annotated_age()`；它们不是好的名字，但应该有助于您更快地发现差异。

我们现在将在此基础上进行扩展，向您展示 Python 类型提示的实际功能。

## 任意类型

这是一种特殊类型的类型。任何没有返回类型或参数类型的函数将隐式地默认使用`Any`。因此，以下函数声明是等效的。

```py
# annotations/any.py
from typing import Any
def square(n):
    return n**2
def square_annotated(n: Any) -> Any:
    return n**2 
```

上述声明完全等效。`Any` 在某些情况下可能很有用，例如在注释数据容器、函数装饰器或当函数设计为处理多种类型的输入时。一个简单的例子可以是：

```py
# annotations/any.py
from typing import Any
def reverse(items: list) -> list:
    return list(reversed(items))
def reverse_any(items: list[Any]) -> list[Any]:
    return list(reversed(items)) 
```

在这种情况下，这两个定义再次完全等效。唯一的区别是，在第二个定义中，通过使用 `Any`，我们明确指出列表 `items` 应该包含 *任何* 类型的对象。

## 类型别名

我们几乎可以在类型提示中使用任何 Python 类型。此外，`typing` 模块引入了几个我们可以利用的结构来扩展它们。

其中一种结构是 **类型别名**，它使用 `type` 语句定义。结果是 `TypeAliasType` 的一个实例。它们是简化代码阅读的便捷方式。让我们看一个例子：

```py
# annotations/type.aliases.py
type DatabasePort = int
def connect_to_database(host: str, port: DatabasePort):
    print(f"Connecting to {host} on port {port}...") 
```

如上例所示，我们可以定义自己的类型别名。从静态类型检查器的角度来看，`DatabasePort` 和 `int` 将被同等对待。

尽管从这样一个示例中可能不明显，但使用类型别名可以增强可读性并简化重构。想象一下，有几个函数期望 `DatabasePort` 类型：如果我们想重构代码库，使用字符串来表示端口号，我们只需更改定义 `DatabasePort` 的那一行。如果我们简单地使用 `int`，我们就需要重构所有函数定义。

## 特殊形式

特殊形式可以用作注释中的类型。它们都支持使用 [] 进行索引，但每个都有独特的语法。在这里，我们将看到 `Optional` 和 `Union`；有关所有特殊形式的完整列表，请参阅官方文档[`docs.python.org/3/library/typing.html#special-forms`](https://docs.python.org/3/library/typing.html#special-forms)。

### Optional

Python 允许在函数参数有默认值时使用可选参数。在这里，我们需要做出区分。让我们重新引入 `greeter` 函数，并给它添加一个默认值：

```py
# annotations/optional.py
def greeter(name: str = "stranger") -> str:
    return f"Hello, {name}!" 
```

这次，我们在 `greeter()` 函数中添加了一个默认值。这意味着如果我们不带任何参数调用它，它将返回字符串 `"Hello, stranger!"`。

这个定义假设当我们调用 `greeter()` 时，如果我们传递 `name`，它将是一个字符串。有时，这并不是我们想要的，我们需要在函数调用时提供一个 `None` 参数，如果没有提供。对于这种情况，`typing` 模块为我们提供了 `Optional` 特殊形式：

```py
# annotations/optional.py
def greeter_optional(name: Optional[str] = None) -> str:
    if name is not None:
        return f"Hello, {name}!"
    return "No-one to greet!" 
```

在 `greeter_optional()` 函数中，我们不想在没有传递名字的情况下返回问候语。因为 `None` 不是一个字符串，我们将 `name` 标记为可选，并将其默认值设置为 `None`。

### Union

有时，一个参数可以有多种类型。例如，在连接数据库时，端口号可以指定为一个整数或一个字符串。在这些情况下，拥有 `Union` 特殊形式是有用的：

```py
# annotations/union.py
from typing import Union
def connect_to_database(host: str, port: Union[int, str]):
    print(f"Connecting to {host} on port {port}...") 
```

在上述例子中，对于 `port` 参数，我们希望接受 `int` 和 `str`，`Union` 特殊形式允许我们做到这一点。

自 Python 3.10 以来，我们不需要导入 `Union`，而是可以使用管道（`|`）来指定联合类型。

```py
# annotations/union.py
def connect_to_db(host: str, port: int | str):
    print(f"Connecting to {host} on port {port}...") 
```

这看起来更加简洁。

`Union` 或其管道等价物，顺便提一下，使我们能够避免必须导入 `Optional`，例如，`Optional[str]` 可以写成 `Union[str, None]`，或者简单地写成 `str | None`。让我们通过一个例子来看看后者形式：

```py
# annotations/optional.py
def another_greeter(name: str | None = None) -> str:
    if name is not None:
        return f"Hello, {name}!"
    return "No-one to greet!" 
```

你有没有注意到，上面定义的一些函数在参数上有类型注解，但在返回值上没有？这是为了向你展示注解是完全可选的。如果我们愿意，甚至可以只注解函数接受的某些参数。

## 泛型

让我们继续探讨使用类型提示可以实现的内容，通过探索泛型的概念。

假设我们想要编写一个 `last()` 函数，它接受任何类型的项的列表，并返回最后一个项（如果有），或者返回 `None`。我们知道列表中的所有项都是同一类型，但我们不知道它是什么类型。我们如何正确地注解这个函数呢？泛型的概念帮助我们做到这一点。语法只是比我们之前看到的稍微复杂一点，但并不复杂。让我们编写这个函数：

```py
# annotations/generics.py
def last**[T]****list****[T]****T**(items: ) ->  | None:
    return items[-1] if items else None 
```

请特别注意上述代码中突出显示的部分。首先，我们需要在函数名称后添加一个 `[T]` 作为后缀。然后，我们可以指定 `items` 是一个对象列表，其类型为 `T`，无论它是什么类型。最后，我们还可以使用 `T` 来指定返回值，尽管在这种情况下，我们已经指定了返回类型为 `T` 和 `None` 的联合，以应对 `items` 为空的情况。

尽管函数签名使用了泛型语法，但你会这样调用它：`last([1, 2, 3])`。

泛型的语法支持是 Python 3.12 的新特性。在此之前，为了达到相同的效果，人们会求助于使用 `TypeVar` 工厂，如下所示：

```py
# annotations/generics.py
from typing import TypeVar
U = TypeVar("U")
def first(items: list[U]) -> U | None:
    return items[0] if items else None 
```

注意，在这种情况下，`first()` 并没有定义为 `firstU`。

由于 Python 3.12 语法上的增强，泛型的使用现在变得更加简单。

## 变量注解

现在，让我们暂时从函数转向，讨论变量注解。我们可以快速展示一个不需要太多解释的例子：

```py
# annotations/variables.py
x: int = 10
x: float = 3.14
x: bool = True
x: str = "Hello!"
x: bytes = b"Hello!"
# Python 3.9+
x: list[int] = [7, 14, 21]
x: set[int] = {1, 2, 3}
x: dict[str, float] = {"planck": 6.62607015e-34}
# Python 3.8 and earlier
from typing import List, Set, Dict
x: List[int] = [7, 14, 21]
x: Set[int] = {1, 2, 3}
x: Dict[str, float] = {"planck": 6.62607015e-34}
# Python 3.10+
x: list[int | str] = [0, 1, 1, 2, "fibonacci", "rules"]
# Python 3.9 and earlier
from typing import Union
x: list[Union[int, str]] = [0, 1, 1, 2, "fibonacci", "rules"] 
```

在上面的代码中，我们声明 `x` 是许多东西，仅作为一个例子。正如你所见，语法很简单：我们声明变量的名称，其类型（冒号之后），以及其值（等号之后）。我们还为你提供了一些如何在前几版 Python 中注解变量的例子。方便的是，在 Python 3.12 中，我们可以直接使用内置类型，而无需从 `typing` 模块导入很多内容。

## 容器注解

类型系统假定 Python 容器中的所有元素都将具有相同的类型。对于大多数容器来说这是真的。例如，考虑以下代码：

```py
# annotations/containers.py
# The type checker assumes all elements of the list are integers
a: list[int] = [1, 2, 3]
# We cannot specify two types for the elements of the list
# it only accepts a single type argument
b: list[int, str] = [1, 2, 3, "four"]  # Wrong!
# The type checker will infer that all keys in `c` are strings
# and all values are integers or strings
c: dict[str, int | str] = {"one": 1, "two": "2"} 
```

如上面的代码所示，`list`期望一个类型参数。在这种情况下，一个联合，如`c`注释中的`int | str`，仍然算作一个类型。然而，类型检查器会对`b`提出抱怨。这反映了 Python 中的列表通常用于存储任意数量的同一类型的项目。

包含相同类型元素的容器被称为*同构的*。

注意，尽管语法相似，`dict`期望其键和值都有类型。

## 注释元组

与大多数其他容器类型不同，元组通常包含固定数量的项目，每个位置都期望有特定的类型。包含不同类型的元组被称为*异构的*。正因为如此，元组在类型系统中被特殊处理。

有三种方式来注释元组类型：

+   固定长度的元组，可以进一步分为：

    +   没有命名字段的元组

    +   带有命名字段的元组

+   任意长度的元组

### 固定长度元组

让我们看看一个固定长度元组的例子，没有命名字段。

```py
# annotations/tuples.fixed.py
# Tuple `a` is assigned to a tuple of length 1,
# with a single string element.
a: tuple[str] = ("hello",)
# Tuple `b` is assigned to a tuple of length 2,
# with an integer and a string element.
b: tuple[int, str] = (1, "one")
# Type checker error: the annotation indicates a tuple of
# length 1, but the tuple has 3 elements.
c: tuple[float] = (3.14, 1.0, -1.0)  # Wrong! 
```

在上面的代码中，`a`和`b`都被正确注释了。然而，`c`是不正确的，因为注释表明这是一个长度为`1`的元组，但`c`的长度是`3`。

### 带有命名字段的元组

当元组有多个或两个以上的字段，或者它们在代码库的多个地方使用时，使用`typing.NamedTuple`来注释它们可能很有用。这里有一个简单的例子：

```py
# annotations/tuples.named.py
from typing import NamedTuple
class Person(NamedTuple):
    name: str
    age: int
fab = Person("Fab", 48)
print(fab)  # Person(name='Fab', age=48)
print(fab.name)  # Fab
print(fab.age)  # 48
print(fab[0])  # Fab
print(fab[1])  # 48 
```

如通过`print()`调用的结果所见，这相当于声明一个元组，正如我们在*第二章*，*内置数据类型*中学到的：

```py
# annotations/tuples.named.py
import collections
Person = collections.namedtuple("Person", ["name", "age"]) 
```

使用`typing.NamedTuple`不仅允许我们正确注释元组，如果我们愿意，甚至可以指定默认值：

```py
# annotations/tuples.named.py
class Point(NamedTuple):
    x: int
    y: int
    z: int = 0
p = Point(1, 2)
print(p)  # Point(x=1, y=2, z=0) 
```

注意，在上面的代码中，当我们创建`p`时没有指定第三个参数，但`z`仍然正确地分配给了`0`。

### 任意长度的元组

如果我们想要指定一个任意长度的元组，其中所有元素都是同一类型，我们可以使用特定的语法。这可能在某些情况下很有用，例如，当我们使用元组作为不可变序列时。让我们看看几个例子：

```py
# annotations/tuples.any.length.py
from typing import Any
# We use the ellipsis to indicate that the tuple can have any
# number of elements.
a: tuple[int, ...] = (1, 2, 3)
# All the following are valid, because the tuple can have any
# number of elements.
a = ()
a = (7,)
a = (7, 8, 9, 10, 11)
# But this is an error, because the tuple can only have integers
a = ("hello", "world")
# We can specify a tuple that must be empty
b: tuple[()] = ()
# Finally, if we annotate a tuple like this:
c: tuple = (1, 2, 3)
# The type checker will treat it as if we had written:
c: tuple[Any, ...] = (1, 2, 3)
# And because of that, all the below are valid:
c = ()
c = ("hello", "my", "friend") 
```

如你所见，有无数种方式可以注释一个元组。你想要有多严格的选择取决于你。记住要与其他代码库保持一致。同时，要意识到注释为你的代码带来的价值。如果你正在编写一个打算发布并供其他开发者使用的库，那么在注释中非常精确可能是有意义的。另一方面，没有充分的理由就过于严格或限制可能会损害你的生产力，尤其是在这种精确度不是必需的情况下。

## 抽象基类（ABCs）

在 Python 的旧版本中，`typing` 模块会提供几种类型的泛型版本。例如，列表、元组、集合和字典可以使用泛型具体集合 `List` 、`Tuple` 、`Set` 、`Dict` 等进行注释。

从 Python 3.9 开始，这些泛型集合已经被弃用，转而使用相应的内置类型，这意味着，例如，可以使用 `list` 本身来注释列表，而不需要 `typing.List` 。

文档还指出，这些泛型集合应该用于注释返回值，而参数应该使用抽象集合进行注释。例如，我们应该使用 `collections.abc.Sequence` 来注释只读和可变序列，如 `list` 、`tuple` 、`str` 、`bytes` 等等。

这与 **Postel 的法则** 一致，也称为 **鲁棒性原则** ，它假设：

> “发送时要保守，接受时要宽容。”

*接受时要宽容* 意味着我们在注释参数时不应过于限制。如果一个函数接受一个名为 `items` 的参数，并且它所做的只是迭代或根据位置访问一个项目，那么 `items` 是列表还是元组无关紧要。因此，我们不应使用 `tuple` 或 `list` 进行注释，而应使用 `collections.abc.Sequence` 允许 `items` 以元组或列表的形式传递。

想象一下这样的场景：你使用 `tuple` 来注释 `items` 。过了一段时间，你重构了代码，现在 `items` 作为列表传递。由于 `items` 已不再是元组，函数的注释现在是不正确的。如果我们使用了 `collections.abc.Sequence` ，那么函数就不需要任何修改，因为 `tuple` 和 `list` 都是可行的。

让我们看看一个例子：

```py
# annotations/collections.abcs.py
from collections.abc import Mapping, Sequence
def average_bad(v: list[float]) -> float:
    return sum(v) / len(v)
def average(v: Sequence[float]) -> float:
    return sum(v) / len(v)
def greet_user_bad(user: dict[str, str]) -> str:
    return f"Hello, {user['name']}!"
def greet_user(user: Mapping[str, str]) -> str:
    return f"Hello, {user['name']}!" 
```

上述函数应该有助于澄清问题。以 `average_bad()` 为例。如果我们传递 `v` 作为元组，它就会与我们所使用的函数注释不符，该注释是 `list` 。另一方面，`average()` 没有这个问题。当然，我们也可以用同样的推理来分析 `greet_user_bad()` 和 `greet_user()` 。

回到 Postel 的法则，当涉及到返回值时，最好是保守的，这意味着要具体。返回值应该精确地表明函数返回了什么。

这非常重要，特别是对于调用者来说，他们需要知道在调用函数时将接收到的对象类型。

这里有一个简单的例子，来自同一文件，应该有助于理解：

```py
# annotations/collections.abcs.py
def add_defaults_bad(
    data: Mapping[str, str]
) -> **Mapping**[str, str]:
    defaults = {"host": "localhost", "port": "5432"}
    return {**defaults, **data}
def add_defaults(data: Mapping[str, str]) -> **dict**[str, str]:
    defaults = {"host": "localhost", "port": "5432"}
    return {**defaults, **data} 
```

在上述两个函数中，我们只是将一些假设的连接默认值添加到 `data` 参数传递的任何内容上（如果 `data` 中缺少 `"host"` 和 `"port"` 键）。`add_defaults_bad()` 函数指定返回类型为 `Mapping`。问题是这太泛了。例如，`dict` 和来自 `collections` 模块的 `defaultdict`、`Counter`、`OrderedDict`、`ChainMap` 和 `UserDict` 等对象都实现了 `Mapping` 接口。这使得调用者感到非常困惑。

另一方面，`add_defaults()` 是一个更好的函数，因为它精确地指定了返回类型：`dict`。

常用的 ABC 包括 `Iterable`、`Iterator`、`Collection`、`Sequence`、`Mapping`、`Set`、`MutableSequence`、`MutableMapping`、`MutableSet` 和 `Callable`。

让我们看看一个使用 `Iterable` 的示例：

```py
# annotations/collections.abc.iterable.py
from collections.abc import Iterable
def process_items(items: Iterable) -> None:
    for item in items:
        print(item) 
```

`process_items()` 函数需要做的只是遍历 `items`；因此，我们使用 `Iterable` 来注释它。

对于 `Callable` 可以提供一个更有趣的例子：

```py
# annotations/collections.abc.iterable.py
from collections.abc import Callable
def process_callback(
    arg: str, callback: Callable[[str], str]
) -> str:
    return callback(arg)
def greeter(name: str) -> str:
    return f"Hello, {name}!"
def reverse(name: str) -> str:
    return name[::-1] 
```

这里，我们有 `process_callback()` 函数，它定义了一个字符串参数 `arg` 和一个 `callback` 可调用对象。接下来有两个函数，它们的签名指定了输入参数为字符串，返回值为字符串对象。注意 `callback` 的类型注解 `Callable[[str], str]`，它表明 `callback` 参数应该接受一个字符串输入参数并返回一个字符串输出。当我们用以下代码调用这些函数时，我们得到的是内联注释中指示的输出。

```py
# annotations/collections.abc.iterable.py
print(process_callback("Alice", greeter))  # Hello, Alice!
print(process_callback("Alice", reverse))  # ecilA 
```

这就结束了我们对抽象基类的巡礼。

## 特殊类型原语

在 `typing` 模块中，还有一个称为 **特殊类型原语** 的对象类别，它们非常有趣，了解其中最常见的一些是有用的。我们已经看到了一个例子：`Any`。

其他值得注意的例子包括：

+   `AnyStr`：用于注释可以接受 `str` 或 `bytes` 参数但不能混合两者的函数。这被称为 **约束类型变量**，意味着类型只能是给定的约束之一。在 `AnyStr` 的情况下，它要么是 `str`，要么是 `bytes`。

+   `LiteralString`：一个只包含字面字符串的特殊类型。

+   `Never` / `NoReturn`：可以用来表示函数永远不会返回——例如，它可能会引发异常。

+   `TypeAlias`：已被 `type` 语句取代。

最后，`Self` 类型值得更多关注。

### `Self` 类型

`Self` 类型是在 Python 3.11 中添加的，它是一个特殊类型，用于表示当前封装的类。让我们看一个例子：

```py
# annotations/self.py
from typing import Self
from collections.abc import Iterable
from dataclasses import dataclass
@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5
    @classmethod
    def sum_points(cls, points: Iterable[**Self**]) -> **Self**:
        return cls(
            sum(p.x for p in points),
            sum(p.y for p in points),
            sum(p.z for p in points),
        ) 
```

在上面的代码中，我们创建了一个简单的类`Point`，它表示空间中的一个三维点。为了展示如何使用`Self`类型，我们创建了一个`sum_points()`类方法，它接受一个可迭代的`points`，并返回一个`Point`对象，其坐标是`points`中所有项对应坐标的总和。

要注解`points`参数，我们将`Self`传递给`Iterable`，并且对于方法的返回值也做同样的处理。在引入`Self`类型之前，我们不得不为每个需要它的类创建一个独特的“self”类型变量。你可以在官方文档中找到一个例子，见[`docs.python.org/3/library/typing.html#typing.Self`](https://docs.python.org/3/library/typing.html#typing.Self)。

注意，按照惯例，`self`和`cls`参数都没有类型注解。

现在，让我们继续看看如何注解可变参数。

## 注解变量参数

要注解可变位置参数和关键字参数，我们使用迄今为止看到的相同语法。一个快速示例胜过任何解释：

```py
# annotations/variable.parameters.py
def add_query_params(
    ***urls:** **str**, ****query_params:** **str**
) -> list[str]:
    params = "&".join(f"{k}={v}" for k, v in query_params.items())
    return [f"{url}?{params}" for url in urls]
urls = add_query_params(
    "https://example1.com",
    "https://example2.com",
    "https://example3.com",
    limit="10",
    offset="20",
    sort="desc",
)
print(urls)
# ['https://example1.com?limit=10&offset=20&sort=desc',
#  'https://example2.com?limit=10&offset=20&sort=desc',
#  'https://example3.com?limit=10&offset=20&sort=desc'] 
```

在上面的例子中，我们编写了一个虚拟函数`add_query_params()`，它向一组 URL 添加一些查询参数。注意，在函数定义中，我们只需要指定元组`urls`中包含的对象的类型，以及`query_params`字典中值的类型。声明`*urls: str`等同于`tuple[str, …]`，而`**query_params: str`等同于`dict[str, str]`。

## 协议

让我们通过讨论**协议**来结束我们对类型注解的探索。在面向对象编程中，协议定义了一组类必须实现的方法，而不强制从任何特定类继承。它们类似于其他语言中的接口概念，但更加灵活和非正式。它们通过允许不同类在遵循相同协议的情况下可以互换使用来促进多态的使用，即使它们没有共享一个共同的基类。

这个概念从 Python 的开始就一直是其一部分。这种类型的协议通常被称为**动态协议**，并在**Python 语言参考**的“数据模型”章节（[`docs.python.org/3/reference/datamodel.html`](https://docs.python.org/3/reference/datamodel.html)）中进行了描述。

然而，在类型提示的上下文中，协议是`typing.Protocol`的子类，它定义了一个类型检查器可以验证的接口。

由 PEP 544（[`peps.python.org/pep-0544/`](https://peps.python.org/pep-0544/)）引入，它们实现了结构化子类型化（非正式地称为*静态鸭子类型*），我们在本章的开头简要探讨了这一点。

一个对象与协议的兼容性是由某些方法或属性的存在来确定的，而不是从特定类继承。

因此，在那些我们无法轻松定义类型的情况下，协议非常有用，并且更方便以“它应该支持某些方法或具有某些属性”的形式表达注释。

由 PEP 544 定义的协议通常被称为**静态协议**。

动态和静态协议存在两个关键区别：

+   动态协议允许部分实现。这意味着一个对象可以只为协议中的一部分方法提供实现，并且仍然是有用的。然而，静态协议要求对象提供协议中声明的所有方法，即使软件不需要它们全部。

+   静态协议可以被静态类型检查器验证，而动态协议则不能。

你可以在这里找到`typing`模块提供的协议列表：[`docs.python.org/3/library/typing.html#protocols`](https://docs.python.org/3/library/typing.html#protocols)。它们的名称以前缀词`Supports`开头，后面跟着它们声明的支持方法的标题化版本。一些例子包括：

+   `SupportsAbs`：一个具有一个抽象方法`__abs__`的 ABC。

+   `SupportsBytes`：一个具有一个抽象方法`__bytes__`的 ABC。

+   `SupportsComplex`：一个具有一个抽象方法`__complex__`的 ABC。

其他曾经存在于`typing`模块但现在已迁移到`collections.abc`的协议包括`Iterable`、`Iterator`、`Sized`、`Container`、`Collection`、`Reversible`和`ContextManager`，仅举几个例子。

你可以在 Mypy 文档中找到完整的列表：[`mypy.readthedocs.io/en/stable/protocols.html#predefined-protocols-reference`](https://mypy.readthedocs.io/en/stable/protocols.html#predefined-protocols-reference)。

既然我们已经对协议有了概念，在类型提示的背景下，让我们看看一个例子，展示如何创建一个简单的自定义协议：

```py
# annotations/protocols.py
from typing import Iterable, Protocol
class SupportsStart(Protocol):
    def start(self) -> None: ...
class Worker:  # No SupportsStart base class.
    def __init__(self, name: str) -> None:
        self.name = name
    def start(self) -> None:
        print(f"Starting worker {self.name}")
def start_workers(workers: Iterable[SupportsStart]) -> None:
    for worker in workers:
        worker.start()
workers = [Worker("Alice"), Worker("Bob")]
start_workers(workers) 
```

在上面的代码中，我们定义了一个协议类`SupportsStart`，它有一个方法：`start()`。为了使其成为静态协议，`SupportsStart`从`Protocol`继承。有趣的部分就在这里，当我们创建`Worker`类时。请注意，没有必要从`SupportsStart`类继承。`Worker`类只需要满足协议，这意味着它需要一个`start()`方法。

我们还编写了一个函数`start_workers()`，它接受一个参数`workers`，被注释为`Iterable[SupportsStart]`。这就是使用协议所需的所有内容。我们定义了几个工人，`Alice`和`Bob`，并将它们传递给函数调用。

运行上面的例子将产生以下输出：

```py
Starting worker Alice
Starting worker Bob 
```

现在想象一下，我们还想停止一个工人。这是一个更有趣的情况，因为它允许我们讨论如何子类化协议。让我们看看一个例子：

```py
# annotations/protocols.subclassing.py
from typing import Iterable, Protocol
class SupportsStart(Protocol):
    def start(self) -> None: ...
class SupportsStop(Protocol):
    def stop(self) -> None: ...
class SupportsWorkCycle(**SupportsStart, SupportsStop, Protocol**):
    pass
class Worker:
    def __init__(self, name: str) -> None:
        self.name = name
    def start(self) -> None:
        print(f"Starting worker {self.name}")
    def stop(self) -> None:
        print(f"Stopping worker {self.name}")
def start_workers(workers: Iterable[SupportsWorkCycle]) -> None:
    for worker in workers:
        worker.start()
        worker.stop()
workers = [Worker("Alice"), Worker("Bob")]
start_workers(workers) 
```

在上面的示例中，我们了解到我们可以像使用混入（mixins）一样组合协议。一个关键的区别是，当我们从协议类继承，例如在`SupportsWorkCycle`的情况下，我们仍然需要显式地将`Protocol`添加到基类列表中。如果我们不这样做，静态类型检查器会报错。这是因为从现有协议继承并不会自动将子类转换为协议。它只会创建一个实现给定协议的常规类或 ABC。

您可以在 Mypy 文档中找到有关协议的更多信息：[`mypy.readthedocs.io/en/stable/protocols.html`](https://mypy.readthedocs.io/en/stable/protocols.html)。

现在，让我们讨论 Mypy，这是 Python 社区最广泛采用的静态类型检查器。

# Mypy 静态类型检查器

目前，Python 有几种静态类型检查器。目前最广泛采用的是：

+   **Mypy**：设计用于与 Python 的 PEP 484 定义的类型注解无缝工作，支持渐进式类型，与现有代码库集成良好，并拥有广泛的文档。您可以在[`mypy.readthedocs.io/`](https://mypy.readthedocs.io/)找到它。

+   **Pyright**：由 Microsoft 开发，这是一个快速的类型检查器，针对 Visual Studio Code 进行了优化。它进行增量分析以实现快速类型检查，并支持 TypeScript 和 Python。您可以在[`github.com/microsoft/pyright`](https://github.com/microsoft/pyright)找到它。

+   **Pylint**：一个综合的静态分析工具，包括类型检查、代码质量检查以及代码风格检查。它高度可配置，支持自定义插件，并生成详细的代码质量报告。您可以在[`pylint.org/`](https://pylint.org/)找到它。

+   **Pyre**：由 Facebook 开发，它速度快，可扩展，并且与大型代码库配合良好。它支持渐进式类型，并拥有强大的类型推断引擎。它还很好地与持续集成系统集成。您可以在[`pyre-check.org/`](https://pyre-check.org/)找到它。

+   **Pytype**：由 Google 开发，它自动推断类型并减少了对显式注解的需求，它可以生成这些注解。它与 Google 开源工具集成良好。您可以在[`github.com/google/pytype`](https://github.com/google/pytype)找到它。

对于本章的这一部分，我们决定使用 Mypy，因为它目前似乎是最受欢迎的。

要在虚拟环境中安装它，您可以运行以下命令：

```py
$ pip install mypy 
```

Mypy 也被包含在本章的要求文件中。当 Mypy 安装后，您可以对任何文件或文件夹运行它。Mypy 将递归遍历任何文件夹以查找 Python 模块（`*.py`文件）。以下是一个示例：

```py
$ mypy program.py some_folder another_folder 
```

命令提供了一组庞大的选项，我们鼓励您通过运行以下命令来探索：

```py
$ mypy --help 
```

让我们从一个非常简单的没有注解的函数示例开始，看看运行`mypy`后的结果。

```py
# mypy_src/simple_function.py
def hypothenuse(a, b):
    return (a**2 + b**2) ** 0.5 
```

在此模块上运行 `mypy` 得到以下结果：

```py
$ mypy simple_function.py
Success: no issues found in 1 source file 
```

这可能不是您预期的结果，但 Mypy 被设计成支持逐步向现有代码库添加类型注解。为未注解的代码输出错误信息会阻止开发者以这种方式使用它。因此，默认行为是忽略没有注解的函数。如果我们想无论如何都让 Mypy 检查 `hypothenuse()` 函数，我们可以这样运行它（注意我们已重新格式化输出以适应书籍的宽度）：

```py
$ mypy --strict mypy_src/simple_function.py
mypy_src/simple_function.py:4:
    error: Function is missing a type annotation  [no-untyped-def]
Found 1 error in 1 file (checked 1 source file) 
```

现在 Mypy 告诉我们，该函数缺少类型注解，因此让我们修复它。

```py
# mypy_src/simple_function_annotated.py
def hypothenuse(a: float, b: float) -> float:
    return (a**2 + b**2) ** 0.5 
```

我们可以再次运行 `mypy`：

```py
$ mypy simple_function_annotated.py 
Success: no issues found in 1 source file 
```

优秀——现在函数已添加注解，`mypy` 运行成功。让我们尝试一些函数调用：

```py
print(hypothenuse(3, 4))  # This is fine
print(hypothenuse(3.5, 4.9))  # This is also fine
print(hypothenuse(complex(1, 2), 10))  # Type checker error 
```

前两个调用是好的，但最后一个产生了错误：

```py
$ mypy mypy_src/simple_function_annotated.py
mypy_src/simple_function_annotated.py:10:
    error: Argument 1 to "hypothenuse" has incompatible
    type "complex"; expected "float"  [arg-type]
Found 1 error in 1 file (checked 1 source file) 
```

Mypy 通知我们，传递一个 `complex` 类型而不是所需的 `float` 类型是不可以的。这两种类型不兼容。

让我们尝试一个稍微复杂一点的例子（无意中打趣）：

```py
# mypy_src/case.py
from collections.abc import Iterable
def title(names: Iterable[str]) -> list[str]:
    return [name.title() for name in names]
print(title(["ALICE", "bob"]))  # ['Alice', 'Bob'] - mypy OK
print(title([b"ALICE", b"bob"]))  # [b'Alice', b'Bob'] - mypy ERR 
```

上述函数将 `names` 中的每个字符串应用为首字母大写格式。首先，我们用字符串 `"ALICE"` 和 `"bob"` 调用它一次，然后我们用 `bytes` 对象 `b"ALICE"` 和 `b"bob"` 调用它。这两个调用都成功了，因为 `str` 和 `bytes` 对象都有 `title()` 方法。然而，运行 `mypy` 得到以下结果：

```py
$ mypy mypy_src/case.py
mypy_src/case.py:10:
    error: List item 0 has incompatible type "bytes";
    expected "str"  [list-item]
mypy_src/case.py:10:
    error: List item 1 has incompatible type "bytes";
    expected "str"  [list-item]
Found 2 errors in 1 file (checked 1 source file) 
```

再次强调，Mypy 指出两种类型的兼容性问题——这次是 `str` 和 `bytes`。我们可以很容易地修复这个问题，通过修改第二个调用或更改函数上的类型注解。让我们做后者：

```py
# mypy_src/case.fixed.py
from collections.abc import Iterable
def title(names: Iterable[**str** **|** **bytes**]) -> list[**str** **|** **bytes**]:
    return [name.title() for name in names]
print(title(["ALICE", "bob"]))  # ['Alice', 'Bob'] - mypy OK
print(title([b"ALICE", b"bob"]))  # [b'Alice', b'Bob'] - mypy OK 
```

现在，我们在注解中使用 `str` 和 `bytes` 类型的联合，`mypy` 运行成功。

我们的建议是安装 Mypy 并将其运行在您可能拥有的任何现有代码库上。尝试逐步引入类型注解，并使用 Mypy 检查您代码的正确性。这项练习将帮助您熟悉类型提示，并且对您的代码也有益处。

# 一些有用的资源

我们建议您阅读本章开头列出的所有 PEP（Python Enhancement Proposals）。我们还建议您研究我们在过程中提到的各种资源，其中一些列在下面供您方便查阅：

+   Python 类型提示文档：

[`typing.readthedocs.io/en/latest/`](https://typing.readthedocs.io/en/latest/)

+   使用 Python 进行静态类型：

[`docs.python.org/3/library/typing.html`](https://docs.python.org/3/library/typing.html)

+   抽象基类：

[`docs.python.org/3/library/abc.html`](https://docs.python.org/3/library/abc.html)

+   容器的抽象基类：

[`docs.python.org/3/library/collections.abc.html`](https://docs.python.org/3/library/collections.abc.html)

+   Mypy 文档：

[`mypy.readthedocs.io/en/stable/`](https://mypy.readthedocs.io/en/stable/)

在 FastAPI 框架文档中还有一个快速的 *Python 类型简介* 部分，我们建议您阅读：[`fastapi.tiangolo.com/python-types/`](https://fastapi.tiangolo.com/python-types/)。

**FastAPI** 是一个用于构建 API 的现代 Python 框架。*第十四章*，*API 开发简介*，专门介绍它，所以我们建议在阅读该章节之前至少阅读关于类型的介绍。

# 摘要

在本章中，我们探讨了 Python 中的类型提示主题。我们首先理解了 Python 对类型的原生方法，并回顾了类型提示的历史，这些类型提示是从 Python 3 逐步引入的，并且仍在不断发展。

我们研究了类型提示的好处，然后学习了如何注释函数、类和变量。我们探讨了基础知识，并讨论了主要内置类型，同时也涉猎了更高级的主题，如泛型、抽象基类和协议。

最后，我们提供了一些如何使用最受欢迎的静态类型检查器 Mypy 的示例，以逐步引入代码库中的类型检查，并以一个简短的回顾结束本章，回顾了您进一步研究此主题最有用的资源。

这本书的理论部分到此结束。剩余的章节是项目导向的，采用更实际的方法，从数据科学的介绍开始。通过学习第一部分的内容所获得的知识应该足以支持您在阅读下一章时的学习。

# 加入我们的 Discord 社区

加入我们的 Discord 社区空间，与作者和其他读者进行讨论：

`discord.com/invite/uaKmaz7FEC`

![img](img/QR_Code119001106417026468.png)
