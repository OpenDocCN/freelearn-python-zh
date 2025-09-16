

# 第三章：Python 类型提示和 Pydantic

在探索 FastAPI 之前，了解一些将在 FastAPI 旅程中大量使用的 Python 概念是有用的。

Python 类型提示是语言中非常重要且相对较新的特性，它有助于开发者提高工作效率，为开发流程带来更大的健壮性和可维护性。类型使你的代码更易于阅读和理解，最重要的是，它们促进了良好的编程实践。

FastAPI 高度基于 Python 类型提示。因此，在深入研究框架之前，回顾类型提示的基本概念、它们是什么、如何实现以及它们的目的是有用的。这些基础知识将帮助你使用 FastAPI 创建健壮、可维护和可扩展的 API。

到本章结束时，你将对 Python 中类型注解在 FastAPI 和 Pydantic 中的作用有深入的理解。Pydantic 是一个现代 Python 库，它在运行时强制执行类型提示，当数据无效时提供可定制且用户友好的错误，并允许使用 Python 类型注解定义数据结构。

你将能够精确地建模你的数据，利用 Pydantic 的高级功能，使你成为一个更好的、更高效的 FastAPI 开发者。

本章将涵盖以下主题：

+   Python 类型提示及其用法

+   Pydantic 的概述及其主要功能，包括解析和验证数据

+   数据反序列化和序列化，包括高级和特殊情况

+   验证和数据转换、别名以及字段和模型级别的验证

+   高级 Pydantic 使用，例如嵌套模型、字段和模型设置

# 技术要求

要运行本章中的示例应用程序，你应在本地计算机上安装 Python 版本 3.11.7（https://www.python.org/downloads/）或更高版本，一个虚拟环境，以及一些包。由于本章的示例不会使用 FastAPI，如果你愿意，你可以创建一个干净的虚拟环境，并使用以下命令安装 Pydantic：

```py
pip install pydantic==2.7.1 pydantic_settings==2.2.1
```

在本章中，你将使用 Pydantic 以及一些与 Pydantic 相关的包，例如`pydantic_settings`。

# Python 类型

编程语言中存在的不同类型定义了语言本身——它们定义了其边界，并为可能实现的方式设定了一些基本规则，更重要的是，它们推荐了实现某种功能的方法。不同类型的变量有完全不同的方法和属性集合。例如，将字符串大写是有意义的，但将浮点数或整数列表大写则没有意义。

如果你已经使用 Python 一段时间了，即使是完成最平凡的任务，你也已经知道，就像每一种编程语言一样，它支持不同类型的数据——字符串和不同的数值类型，如整数和浮点数。它还拥有一个相当丰富的数据结构库：从字典到列表，从集合到元组，等等。

Python 是一种**动态类型语言**。这意味着变量的类型不是在编译时确定的，而是在运行时确定的。这个特性使得语言本身具有很大的灵活性，并允许你将一个变量声明为字符串，使用它，然后稍后将其重新赋值为列表。然而，改变变量类型的便捷性可能会使得更大、更复杂的代码库更容易出错。动态类型意味着变量的类型与其本身嵌入，并且易于修改。

在另一端的是所谓的静态类型语言：C、C++、Java、Rust、Go 等等。在这些语言中，变量的类型是在编译时已知的，并且不能随时间改变。类型检查是在编译时（即在运行时之前）进行的，错误是在运行时之前捕获的，因为编译器会阻止程序编译。

编程语言根据另一个不同的轴划分为不同的类别：强类型语言和弱类型语言。这个特性告诉我们语言对其类型限制到多大程度，以及从一个类型强制转换为另一个类型有多容易。例如，与 JavaScript 不同，Python 被认为是在这个光谱的较强一侧，当你在 Python 解释器中尝试执行非法操作时，解释器会发送强烈的消息，例如在 Python 解释器中输入以下内容以将`dict`类型添加到数字中：

```py
>>>{}+3
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: unsupported operand type(s) for +: 'dict' and 'int'
```

因此，虽然 Python 会在你尝试执行不支持的操作时抱怨，但它只会在运行时这样做，而不是在执行代码之前。实际上，对于开发者来说，没有任何提示表明你正在编写的代码违反了 Python 的类型系统。

# 类型提示

正如你在上一节中看到的，Python 是一种动态类型语言，类型是在运行时才知道的。由于变量类型嵌入在变量的值中，作为一个开发者，仅通过查看它或使用你选择的 IDE 检查它，你无法知道代码库中遇到的变量的类型。幸运的是，Python 从 3.5 版本开始引入了一个非常受欢迎的特性——类型注解（https://peps.python.org/pep-0484/）。

类型注解或提示在 Python 中是额外的语法，它通知你，开发者，变量的预期类型。它们在运行时不被 Python 语言使用，并且以任何方式修改或影响你的程序的行为。你可能想知道，如果 Python 解释器甚至看不到它们，这些提示有什么用。

结果表明，几个重要的好处将使几乎任何代码库都更加健壮、更易于维护和面向未来：

+   **更快的代码开发**：任何阅读你代码的开发者都会确切知道任何注释变量的类型——无论是整数还是浮点数，列表还是集合，这有助于加快开发速度。

+   **方法和属性知识**：你将确切知道任何给定变量可用的哪些方法和属性。在大型代码库中意外更改变量的类型将被立即检测到。

+   **简化代码开发**：代码编辑器和 IDE（如 Visual Studio Code）将提供出色的支持和自动完成（IntelliSense），进一步简化开发并减少开发者的认知负荷。

+   **自动代码生成**：FastAPI 提供基于 Python 类型提示的自动和交互式（如完全功能的 REST API）文档，完全基于 Python 类型提示。

+   **类型检查器**：这是最重要的好处。这些是在后台运行的程序，对你的代码进行静态分析，发现潜在问题并立即通知你。

+   **更易于阅读和更小的认知负荷**：注释的代码更容易阅读，并且当你作为开发者需要处理代码并试图弄清楚它应该做什么时，它对你的认知负荷要小得多。

+   **强类型且灵活**：保留了语言强类型和动态类型灵活性的特点，同时允许施加必要的安全要求和约束。虽然推荐用于大型代码库，但 Python 类型提示已深入 FastAPI 和 Pydantic，因此即使是小型项目，也至少需要了解类型以及如何使用它们。

类型提示是 FastAPI 的基础。结合 MongoDB 的灵活文档模式，它是 FARM 栈开发的支柱。类型提示确保你的应用程序数据流在系统中的每个时刻都保持正确的数据类型。虽然对于简单的端点来说这可能看起来微不足道——数量应该是整数，名称应该是字符串等——但是当你的数据结构变得更加复杂时，调试类型错误可能会变得非常繁琐。

类型提示也可以被定义为一种形式化——一种在运行时之前（静态）向类型检查器（在你的情况下是 **Mypy**）指示值类型的正式解决方案，这将确保当 Python 运行时遇到你的程序时，类型不会成为问题。

下一个部分将详细说明类型提示的语法、如何注释函数以及如何使用 Mypy 检查代码。

## 实现类型提示

让我们看看如何实现类型提示。创建一个名为 `Chapter3` 的目录，并在其中创建一个虚拟环境，如前所述。在此目录内，如果你想要能够精确地重现章节中的示例，请添加一个包含以下内容的 `requirements.txt` 文件：

```py
mypy==1.10.0
pydantic==2.7.4
```

使用 `requirements.txt` 安装包：

```py
pip install -r requirements.txt
```

现在，你已经准备好探索 Python 类型提示的世界了。

虽然有许多 Python 类型检查器——基本上是执行源代码静态分析而不运行它的工具——但我们将使用 `mypy`，因为它易于安装。稍后，你将拥有 Black 或 Ruff 等工具，这些工具会对你的源代码执行不同的操作，包括类型检查。

为了展示 Python 类型注解语法，一个简单的函数，如下所示，就足够了：

1.  创建一个名为 `chapter3_01.py` 的文件并定义一个简单的函数：

    ```py
    def print_name_x_times(name: str, times: int) -> None:
        for _ in range(times):
            print(name)
    ```

    之前的函数接受两个参数，`name`（一个字符串）和 `times`（一个整数），并返回 `None`，同时该函数会在控制台打印给定名称指定次数。如果你尝试在代码中调用该函数并开始输入参数，Visual Studio Code（或任何具有 Python 类型检查支持的 IDE）会立即建议第一个位置参数为字符串，第二个位置参数为整数。

1.  你可以尝试输入错误的参数类型，例如，先输入一个整数，然后输入一个字符串，保存文件，并在命令行上运行 `mypy`：

    ```py
    mypy chapter3_01.py
    ```

1.  Mypy 将会通知你存在两个错误：

    ```py
    types_testing.py:8: error: Argument 1 to "print_name_x_times" has incompatible type "int"; expected "str"  [arg-type]
    types_testing.py:8: error: Argument 2 to "print_name_x_times" has incompatible type "str"; expected "int"  [arg-type]
    Found 2 errors in 1 file (checked 1 source file)
    ```

这个例子足够简单，但再次看看 **Python 增强提案 8**（**PEP 8**）在另一个例子中对类型提示语法的建议：

1.  插入一个具有值的简单变量：

    ```py
    text: str = "John"
    ```

    冒号紧接在变量后面（没有空格），冒号后有一个空格，并且在提供值的情况下，等号周围有空格。

1.  当注释函数的输出时，由破折号和大于号组成的“箭头”（`->`）应该被一个空格包围，如下所示：

    ```py
    def count_users(users: list[str]) -> int:
        return len(users)
    ```

    到目前为止，你已经看到了简单的注解，这些注解将变量限制为一些 Python 原始类型，包括整数和字符串。类型注解可以更加灵活：你可能希望允许变量接受几种不同的变量类型，例如整数和字符串。

1.  你可以使用 `typing` 模块的 `Union` 包来实现这一点：

    ```py
    from typing import Union
    x: Union(str, int)
    ```

1.  之前定义的 `x` 变量可以接受字符串或整数值。实现相同功能的一种更现代和简洁的方式如下：

    ```py
    x: str | int
    ```

这些注解意味着变量 `x` 可以是整数，也可以接受 `string` 类型的值，这与整数的类型不同。

`typing` 模块包含几种所谓的泛型，包括以下几种：

+   `List`：用于应该为列表类型的变量

+   `Dict`：用于字典

+   `Sequence`：用于任何类型的值序列

+   `Callable`：用于可调用对象，例如函数

+   `Iterator`：表示一个函数或变量接受一个迭代器对象（一个实现迭代器协议并可用于 `for` 循环的对象）

注意

鼓励你探索 `typing` 模块，但请记住，该模块中的类型正在逐渐被导入到 Python 的代码功能中。

例如，`List` 类型在处理 FastAPI 时非常有用，因为它允许你快速高效地将项目或资源的列表序列化为 JSON 输出。

`List` 类型的例子如下，在一个名为 `chapter3_02.py` 的新文件中：

```py
from typing import List
def square_numbers(numbers: List[int]) -> List[int]:
    return [number ** 2 for number in numbers]
# Example usage
input_numbers = [1, 2, 3, 4, 5]
squared_numbers = square_numbers(input_numbers)
print(squared_numbers)  # Output: [1, 4, 9, 16, 25]
```

另一个有用的类型是 `Literal`，它将变量的可能值限制为几个可接受的状态：

```py
from typing import Literal
account_type: Literal["personal", "business"]
account_type = "name"
```

前面的几行展示了类型提示的力量。将 `account_type` 变量分配给字符串本身并没有什么错误，但这个字符串不是可接受状态集的一部分，因此 Mypy 会抱怨并返回一个 `Incompatible types in` `assignment` 错误。

现在，看看一个包含 `datetime` 参数的例子。创建一个名为 `chapter3_03.py` 的新文件：

```py
from datetime import datetime
def format_datetime(dt: datetime) -> str:
     return dt.strftime("%Y-%m-%d %H:%M:%S")
now = datetime.now()
print(format_datetime(now))
```

之前定义的函数接受一个参数——一个 datetime 对象，并输出一个字符串：一个格式良好的日期和时间，适用于在网站上显示。如果你在 Visual Studio Code 编辑器中尝试输入 *dt* 然后一个点，你将收到自动完成系统的提示，提供与 datetime 对象相关的所有方法和属性。

要声明一个结构为字典列表（对任何使用基于 JSON 的 API 的人来说都非常熟悉），你可以使用如下方式，在一个名为 `chapter3_04.py` 的文件中：

```py
def get_users(id: int) -> list[dict]:
    return [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ]
```

在介绍了 Python 中的基本注解类型之后，接下来的几节将探讨一些更高级的类型，这些类型在处理 FastAPI 和 Pydantic 时非常有用。

## 高级注解

你迄今为止看到的注解非常简单，仅传达与变量、函数、类参数或输出相关的特定所需类型的基本信息。Python 的类型系统功能更强大，它可以用来进一步限制允许的变量状态，并防止你，作为开发者，在代码中创建不可能或非法的状态。

最常用的类型如下：

+   `Optional` 类型用于以明确和开发者友好的方式处理可选值和 `None` 值。

+   `Union` 类型允许你定义可能类型的联合，例如整数和字符串。现代 Python 使用管道运算符（`|`），如前例所示。

+   `self` 类型用于表示值将是某个类的实例，这在 Pydantic 模型验证器中非常有用，正如我们稍后将要看到的。

+   `New` 类型允许开发者基于现有类型定义全新的类型。

本节详细介绍了 Python 类型提示、它们的目的以及它们的实现方式。下一节将更深入地探讨 Pydantic，FastAPI 数据验证的得力助手。

# Pydantic

Pydantic 是一个数据验证库，在其网站上被标记为 Python 最广泛使用的验证库。它允许您以细粒度的方式对数据进行建模，并在 Python 类型提示系统中牢固地扎根，同时执行各种类型的验证。实际版本 V2 将代码的关键部分重写为**Rust**以提高速度，并提供了出色的开发者体验。以下列表描述了使用 Pydantic 的一些好处：

+   **基于标准库中的类型提示**：您无需学习虚构的新系统或术语，只需学习纯 Python 类型即可。

+   **卓越的速度**：FastAPI 和 MongoDB 的各个方面都围绕着速度——以创纪录的时间交付快速且响应迅速的应用程序——因此拥有一个快速的验证和解析库是强制性的。Pydantic 的核心是用 Rust 编写的，这确保了数据操作的高速运行。

+   **庞大的社区支持和广泛采用**：当与 Django Ninja、SQLModel、LangChain 等流行包一起工作时，学习 Pydantic 将非常有用。

+   **JSON schema 的发射可能性**：它有助于与其他系统集成。

+   **更多灵活性**：Pydantic 支持不同的模式（在强制转换方面严格和宽松）以及几乎无限定制的选项和灵活性。

+   **深受开发者喜爱**：它已被下载超过 7000 万次，PyPI 上有超过 8000 个包依赖于 Pydantic（截至 2024 年 7 月）。

注意

您可以在其文档中详细了解 Pydantic：[`docs.pydantic.dev/latest/`](https://docs.pydantic.dev/latest/)。

从广义上讲，Pydantic 在现代 Web 开发工作流程中解决了许多重要问题。它确保输入到您的应用程序中的数据是正确形成和格式化的，位于期望的范围内，具有适当类型和尺寸，并且安全且无错误地到达文档存储库。

Pydantic 还确保您的应用程序输出的数据与预期和规范完全一致，省略了不应公开的字段（如用户密码），甚至包括与不兼容系统交互等更复杂的任务。

FastAPI 站在两个强大的 Python 库——Starlette 和 Pydantic 的肩膀上。虽然 Starlette 负责框架的 Web 相关方面，通常通过 FastAPI 提供的薄包装、实用函数和类来实现，但 Pydantic 负责 FastAPI 的非凡开发者体验。Pydantic 是 FastAPI 的基础，利用其强大的功能为所有 FARM 堆栈开发者打开了竞技场。

虽然类型检查是在静态（不运行代码）的情况下执行的，但 Pydantic 在运行时的作用很明显，并扮演着输入数据的守护者角色。你的 FastAPI 应用将从用户那里接收数据，从灵活的 MongoDB 数据库模式中接收数据，以及通过 API 从其他系统接收数据——Pydantic 将简化解析和数据验证。你不需要为每个可能的无效情况编写复杂的验证逻辑，只需创建与你的应用程序需求尽可能匹配的 Pydantic 模型即可。

在接下来的部分中，你将通过具有递增复杂性和要求的示例来探索 Pydantic 的大部分功能，因为我们认为这是熟悉库的最佳和最有效的方式。

## Pydantic 基础知识

与一些提供类似功能的其他库（如`dataclasses`）不同，Pydantic 提供了一个基类（恰当地命名为`BaseModel`），通过继承实现了解析和验证功能。由于你将在接下来的部分中构建用户模型，你可以先列出需要与你的用户关联的最基本数据。至少，你需要以下内容：

+   用户名

+   电子邮件地址

+   一个 ID（目前保持为整数）

+   出生日期

在 Pydantic 中，一个与该规范相关联的用户模型可能如下所示，在一个名为`chapter3_05.py`的文件中：

```py
from datetime import datetime
from pydantic import BaseModel
class User(BaseModel):
    id: int
    username: str
    email: str
    dob: datetime
```

`User`类已经为你处理了很多工作——在类实例化时立即执行验证和解析，因此不需要执行验证检查。

构建类的过程相当直接：每个字段都有一个类型声明，Pydantic 准备好通知你任何可能遇到的错误类型。

如果你尝试创建一个用户，你不应该看到任何错误：

```py
Pu = User(id=1, username="freethrow", email="email@gmail.com", dob=datetime(1975, 5, 13))
```

然而，如果你创建了一个包含错误数据的用户，并且方便地导入了 Pydantic 的`ValidationError`：

```py
from pydantic import BaseModel, ValidationError
try:
    u = User(
        id="one",
        username="freethrow",
        email="email@gmail.com",
        dob=datetime(1975, 5, 13),
    )
    print(u)
except ValidationError as e:
    print(e)
```

当你运行程序时，Pydantic 会通知你数据无法验证：

```py
1 validation error for User
id
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='one', input_type=str]
```

Pydantic 的错误信息，源自`ValidationError`，是故意设计得信息丰富且精确的。出现错误的字段被称作`id`，错误类型也会被描述。首先想到的有用之处是，如果有多个错误——例如，你可能提供了一个无效的`datetime`——Pydantic 不会在第一个错误处停止。它会继续解析整个实例，并输出一个错误列表，这个列表可以很容易地以 JSON 格式输出。这实际上是在处理 API 时的期望行为；你希望能够列出所有错误，例如，向发送了错误数据的后端用户。异常包含了一个遇到的所有错误的列表。

模型保证实例在验证通过后，将包含所需的字段，并且它们的类型是正确的。

你还可以根据类型提示约定提供默认值和可空类型：

```py
class User(BaseModel):
    id: int = 2
    username: str
    email: str
    dob: datetime
    fav_colors: list[str] | None = ["red", "blue"]
```

之前的模型有一个默认的 `id` 值（这在实践中可能不是你想要做的）以及一个作为字符串的喜欢的颜色列表，这些也可以是 `None`。

当你创建并打印一个模型（或者更准确地说，当你通过 `print` 函数调用它的表示时），你会得到一个漂亮的输出：

```py
id=2 username='marko' email='email@gmail.com' dob=datetime.datetime(1975, 5, 13, 0, 0) fav_colors=None
```

Pydantic 默认以宽松模式运行，这意味着它会尝试将提供的类型强制转换为模型中声明的类型。例如，如果你将用户 ID 作为字符串 `"2"` 传递给模型，将不会出现任何错误，因为 Pydantic 会自动将 ID 转换为整数。

虽然字段可以通过点符号（`user.id`）访问并且可以轻松修改，但这不建议这样做，因为验证规则将不会应用。你可以创建一个具有 `id` 值为 `5` 的用户实例，访问 `user.id`，并将其设置为字符串 `"five"`，但这可能不是你想要的。

除了纯粹的数据验证之外，Pydantic 还为你的应用程序提供了其他重要的功能。Pydantic 模型中最广泛使用的操作包括以下内容：

+   **数据反序列化**：将数据摄入模型

+   **数据序列化**：将验证后的数据从模型输出到 Python 数据结构或 JSON

+   **数据修改**：实时清理或修改数据

接下来的几节将更详细地查看这些操作的每一个。

## 反序列化

反序列化是指向模型提供数据的过程，这是输入阶段，与序列化过程相对，序列化意味着以期望的形式输出模型数据。反序列化与验证紧密相关，因为验证和解析过程是在实例化模型时执行的，尽管这可以被覆盖。

在 Pydantic 中，`ValidationError` 是 Pydantic 类型，当数据无法成功解析为模型实例时会被抛出。

虽然你已经通过实例化基于 Pydantic 的用户模型执行了一些验证，但待验证的数据通常以字典的形式传递。以下是一个将数据作为字典传递的示例，文件名为 `chapter3_06.py`：

创建你的用户模型的另一个版本，并传递一个包含数据的字典：

```py
class User(BaseModel):
    id: int
    username: str
    email: str
    password: str
user = User.model_validate(
    {
        "id": 1,
        "username": "freethrow",
        "email": "email@gmail.com",
        "password": "somesecret",
    }
)
print(user)
```

`.model_validate()` 方法是一个辅助方法，它接受一个 Python 字典并执行类实例化和验证。这个方法在一步中创建你的 `user` 实例并验证数据类型。

类似地，`model_validate_json()` 接受一个 JSON 字符串（当与 API 一起工作时很有用）。

也可以使用 `model_construct()` 方法不进行验证来构建模型实例，但这有非常特定的用户场景，并且在大多数情况下不推荐使用。

你已经学会了如何将数据传递给你的简单 Pydantic 模型。下一个部分将更详细地查看模型字段及其属性。

## 模型字段

Pydantic 字段基于 Python 类型，设置它们为必需或可空并提供默认值是直观的。例如，要为字段创建默认值，只需在模型中提供它作为值即可，而可空字段遵循你在 *Python 类型* 部分中看到的相同约定——通过使用 `typing` 模块的旧联合语法，或使用带有管道操作符的新语法。

以下是一个名为 `chapter3_07.py` 的文件中另一个用户模型的示例：

1.  插入一些默认值：

    ```py
    from pydantic import BaseModel
    from typing import Literal
    class UserModel(BaseModel):
        id: int
        username: str
        email: str
        account: Literal["personal", "business"] | None = None
        nickname: str | None = None
    ```

    之前定义的 `UserModel` 类定义了一些标准的字符串类型字段：一个账户可以有两个确切值或等于 `None`，以及一个昵称可以是字符串或 `None`。

1.  你可以使用 `model_fields` 属性如下检查模型：

    ```py
    print(UserModel.model_fields)
    ```

    你将获得一个方便的列表，其中包含属于该模型的所有字段及其信息，包括它们的类型和是否为必需项：

    ```py
    {'id': FieldInfo(annotation=int, required=True), 'username': FieldInfo(annotation=str, required=True), 'email': FieldInfo(annotation=str, required=True), 'account': FieldInfo(annotation=Union[Literal['personal', 'business'], NoneType], required=False, default=None), 'nickname': FieldInfo(annotation=Union[str, NoneType], required=False, default=None)}
    ```

下一个部分将详细介绍 Pydantic 特定的类型，这些类型使得使用库更加容易和快速。

## Pydantic 类型

虽然 Pydantic 基于标准 Python 类型，如字符串、整数、字典和集合，这使得它对于初学者来说非常直观和简单，但该库还提供了一系列针对常见情况的定制和解决方案。在本节中，你将了解其中最有用的。

严格的类型，如 `StrictBool`、`StrictInt`、`StrictStr` 和其他 Pydantic 特定类型，是只有当验证的值属于这些类型时才会通过验证的类型，没有任何强制转换：例如，`StrictInt` 必须是 `Integer` 类型，而不是 `"1"` 或 `1.0`。

限制类型为现有类型提供额外的约束。例如，`condate()` 是一个具有大于、大于等于、小于和小于等于约束的日期类型。`conlist()` 包装列表类型并添加长度验证，或可以强制规则，即包含的项必须是唯一的。

Pydantic 不仅限于验证原始类型，如字符串和整数。许多额外的验证器涵盖了你在建模业务逻辑时可能遇到的大多数使用情况。例如，`email` 验证器验证电子邮件地址，由于它不是 Pydantic 核心包的一部分，因此需要使用以下命令单独安装：

```py
pip install pydantic[email]
```

Pydantic 网站（https://docs.pydantic.dev/latest/api/types/）提供了一个全面的附加验证类型列表，这些类型扩展了功能——列表可以有最小和最大长度，唯一性可以是必需的，整数可以是正数或负数，等等，例如 CSS 颜色代码。

## Pydantic 字段

虽然简单的 Python 类型注解在许多情况下可能足够，但 Pydantic 的真正力量开始在你开始使用 `Field` 类为字段定制模型并添加元数据到模型字段时显现出来。

让我们看看如何使用上一节中探讨的 `UserModel` 的 `Field` 类。创建一个文件，并将其命名为 `chapter3_08.py`。

首先，使用 `Field` 类重写你之前的 `UserModel`：

```py
from typing import Literal
from pydantic import BaseModel, Field
class UserModelFields(BaseModel):
    id: int = Field(…)
    username: str = Field(…)
    email: str = Field(…)
    account: Literal["personal", "business"] | None = Field(default=None)
    nickname: str | None = Field(default=None)
```

此模型与之前定义的没有字段的模型等效。第一个语法差异可以在提供默认值的方式中看到——`Field` 类接受一个显式定义的默认值。

字段还通过使用别名提供了额外的模型灵活性，正如你将在下一节中看到的。

### 字段别名

字段允许你创建和使用别名，这在处理需要与你的基于 Pydantic 的数据定义兼容的不同系统时非常有用。创建一个名为 `chapter3_09.py` 的文件。假设你的应用程序使用 `UserModelFields` 模型来处理用户，但也需要能够从另一个系统接收数据，可能通过基于 JSON 的 API，而这个其他系统发送的数据格式如下：

```py
external_api_data = {
    "user_id": 234,
    "name": "Marko",
    "email": "email@gmail.com",
    "account_type": "personal",
    "nick": "freethrow",
}
```

这种格式明显不符合你的 `UserModelFields` 模型，而别名提供了一种优雅地处理这种不兼容性的方法：

```py
class UserModelFields(BaseModel):
    id: int = Field(alias="user_id")
    username: str = Field(alias="name")
    email: str = Field()
    account: Literal["personal", "business"] | None = Field(
        default=None, alias="account_type"
    )
    nickname: str | None = Field(default=None, alias="nick")
```

此更新后的模型为所有具有不同名称的字段提供了别名，因此可以验证你的外部数据：

```py
user = UserModelFields.model_validate(external_api_data)
```

在这种情况下，你已经使用了简单的 `alias` 参数，但还有其他选项用于别名，用于序列化或仅用于验证。

此外，`Field` 类允许以不同的方式约束数值，这是 FastAPI 中广泛使用的一个特性。创建一个名为 `chapter3_10.py` 的文件并开始填充它。

假设你需要模拟一个具有以下字段的棋类活动：

```py
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
class ChessTournament(BaseModel):
    id: int = Field(strict=True)
    dt: datetime = Field(default_factory=datetime.now)
    name: str = Field(min_length=10, max_length=30)
    num_players: int = Field(ge=4, le=16, multiple_of=2)
    code: str = Field(default_factory=uuid4)
```

在这个相对简单的课程中，Pydantic 字段引入了一些复杂的验证规则，否则这些规则将非常冗长且难以编写：

+   `dt`：锦标赛的 `datetime` 对象使用 `default_factory` 参数，这是一个在实例化时调用的函数，它提供了默认值。在这种情况下，值等于 `datetime.now`。

+   `name`：此字段有一些长度约束，例如最小和最大长度。

+   **注册球员的数量受到限制**：它必须大于或等于 4，小于或等于 16，并且还必须是偶数——2 的倍数，以便所有球员都能在每一轮比赛中进行比赛。

+   `uuid` 库。

+   `id`：此字段是一个整数，但这次你应用了 `strict` 标志，这意味着你覆盖了 Pydantic 的默认行为，不允许像 `"3"` 这样的字符串通过验证，即使它们可以被转换为整数。

注意

Pydantic 文档中的一个有用页面专门介绍了字段：https://docs.pydantic.dev/latest/concepts/fields/。`Field` 类提供了许多验证选项，建议在开始建模过程之前浏览一下。

下一节将详细介绍如何通过反序列化过程从模型中获取数据。

## 序列化

任何解析和验证库最重要的任务是数据序列化（或数据导出）。这是将模型实例转换为 Python 字典或 JSON 编码字符串的过程。生成 Python 字典的方法是 `model_dump()`，如下面的用户模型示例所示，在一个名为 `chapter3_11.py` 的新文件中。

要在 Pydantic 中使用电子邮件验证，请将以下行添加到 `requirements.txt` 文件中：

```py
email_validator==2.1.1
```

然后，重新运行用户模型：

```py
pip install -r requirements.txt
class UserModel(BaseModel):
    id: int = Field()
    username: str = Field(min_length=5, max_length=20)
    email: EmailStr = Field()
    password: str = Field(min_length=5, max_length=20, pattern="^[a-zA-Z0-9]+$")
```

您正在使用的用户模型是一个相当标准的模型，并且，凭借您对 Pydantic 字段的了解，您已经可以理解它。有几个新的验证，但它们是直观的：从 Pydantic 导入的 `EmailStr` 对象是一个验证电子邮件地址的字符串，而 `password` 字段包含一个额外的正则表达式，以确保该字段只包含字母数字字符，没有空格。以下是一个例子：

1.  创建模型的一个实例并将其序列化为 Python 字典：

    ```py
    u = UserModel(
        id=1,
        username="freethrow",
        email="email@gmail.com",
        password="password123",
    )
    print(u.model_dump())
    ```

    结果是一个简单的 Python 字典：

    ```py
    {'id': 1, 'username': 'freethrow', 'email': 'email@gmail.com', 'password': 'password123'}
    ```

1.  尝试将模型导出为 JSON 表示形式并出于安全原因省略密码：

    ```py
    print(u.model_dump_json(exclude=set("password"))
    ```

    结果是一个省略密码的 JSON 字符串：

    ```py
    {"id":1,"username":"freethrow","email":"email@gmail.com"}
    ```

序列化默认使用字段名而不是别名，但这是可以通过将 `by_alias` 标志设置为 `True` 来轻松覆盖的另一个设置。

在使用 FastAPI 和 MongoDB 时，一个使用的别名示例是 MongoDB 的 `ObjectId` 字段，它通常序列化为字符串。另一个有用的方法是 `model_json_schema()`，它为模型生成 JSON 模式。

模型可以通过 `ConfigDict` 对象进行额外配置，以及一个名为 `model_config` 的特殊字段——该名称是保留的且必须的。在以下名为 `chapter3_12.py` 的文件中，您使用 `model_config` 字段允许通过名称填充模型并防止向模型传递额外的数据：

```py
from pydantic import BaseModel, Field, ConfigDict, EmailStr
class UserModel(BaseModel):
    id: int = Field()
    username: str = Field(min_length=5, max_length=20, alias="name")
    email: EmailStr = Field()
    password: str = Field(min_length=5, max_length=20, pattern="^[a-zA-Z0-9]+$")
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
```

`model_config` 字段允许对模型进行额外配置。例如，`extra` 关键字指的是传递给反序列化过程的数据字段：默认行为是简单地忽略这些数据。

在此示例中，我们将 `extra` 设置为 `forbid`，因此任何传递的额外数据（未在模型中声明）将引发验证错误。"populate_by_name" 是另一个有用的设置，因为它允许我们使用字段名而不是仅使用别名来填充模型，实际上是将两者混合使用。您将看到，当构建需要与不同系统通信的 API 时，此功能非常方便。

### 自定义序列化器

当涉及到序列化时，Pydantic 几乎可以提供无限的能力，并且还提供了不同的序列化方法，用于 Python 和 JSON 输出，这通过使用`@field_serializer`装饰器来实现。

注意

Python 装饰器是一种强大而优雅的特性，允许您在不更改实际代码的情况下修改或扩展函数或方法的行为。

装饰器是高阶函数，它接受一个函数作为输入，添加一些功能，并返回一个新的、装饰过的函数。这种方法促进了 Python 程序的可重用性、模块化和关注点的分离。

在以下示例中，您将创建一个非常简单的银行账户模型，并使用不同类型的序列化器。您的需求是将余额精确四舍五入到两位小数，并且在序列化为 JSON 时，将`updated`字段格式化为 ISO 格式：

1.  创建一个名为`chapter3_13.py`的新文件，并添加一个简单的银行账户模型，该模型只包含两个字段：余额和最后账户更新时间：

    ```py
    from datetime import datetime
    from pydantic import BaseModel, field_serializer
    class Account(BaseModel):
        balance: float
        updated: datetime
        @field_serializer("balance", when_used="always")
        def serialize_balance(self, value: float) -> float:
            return round(value, 2)
        @field_serializer("updated", when_used="json")
        def serialize_updated(self, value: datetime) -> str:
           return value.isoformat()
    ```

    您已添加了两个自定义序列化器。第一个是余额序列化器（如字符串`"balance"`所示），它将始终被使用。这个序列化器简单地将余额四舍五入到两位小数。第二个序列化器仅用于 JSON 序列化，并将日期返回为 ISO 格式的日期时间字符串。

1.  如果您尝试填充模型并检查序列化，您将看到序列化器如何修改了初始默认输出：

    ```py
    account_data = {
        "balance": 123.45545,
        "updated": datetime.now(),
    }
    account = Account.model_validate(account_data)
    print("Python dictionary:", account.model_dump())
    print("JSON:", account.model_dump_json())
    ```

    您将得到类似的输出：

    ```py
    Python dictionary: {'balance': 123.46, 'updated': datetime.datetime(2024, 5, 2, 21, 34, 11, 917378)}
    JSON: {"balance":123.46,"updated":"2024-05-02T21:34:11.917378"}
    ```

在本章的早期部分，您已经看到了通过仅实例化模型类所提供的 Pydantic 基本验证。下一节将讨论 Pydantic 的各种自定义验证方法，以及如何借助 Pydantic 装饰器来利用这些方法，从而超越序列化并提供强大的自定义验证功能。

## 自定义数据验证

与自定义字段序列化器类似，自定义字段验证器作为装饰器实现，使用`@field_validator`装饰器。

字段验证器是类方法，因此它们必须接收整个类作为第一个参数，而不是实例，第二个值是要验证的字段名称（或字段列表，或`*`符号表示所有字段）。

字段验证器应返回解析后的值或一个`ValueError`响应（或`AssertionError`），如果传递给验证器的数据不符合验证规则。与其他 Pydantic 功能一样，从示例开始要容易得多。创建一个名为`chapter3_14.py`的新文件，并插入以下代码：

```py
from pydantic import BaseModel,  field_validator
class Article(BaseModel):
    id: int
    title: str
    content: str
    published: bool
    @field_validator("title")
    @classmethod
    def check_title(cls, v: str) -> str:
        if "FARM stack" not in v:
            raise ValueError('Title must contain "FARM stack"')
        return v.title()
```

验证器在类实例化之前运行，并接受类和验证的字段名称作为参数。`check_title`验证器检查标题是否包含字符串`"FARM stack"`，如果不包含，则抛出`ValueError`。此外，验证器返回标题大写的字符串，因此我们可以在字段级别执行数据转换。

虽然字段验证器提供了很大的灵活性，但它们并没有考虑字段之间的交互和字段值的组合。这就是模型验证器发挥作用的地方，下一节将详细说明。

## 模型验证器

在执行与网络相关数据的验证时，另一个有用的功能是模型验证——在模型级别编写验证函数的可能性，允许各种字段之间进行复杂的交互。

模型验证器可以在实例化模型类之前或之后运行。我们再次将关注一个相当简单的例子：

1.  首先，创建一个新文件，并将其命名为`chapter3_15.py`。

1.  假设你有一个具有以下结构的用户模型：

    ```py
    from pydantic import BaseModel, EmailStr, ValidationError, model_validator
    from typing import Any, Self
    class UserModelV(BaseModel):
        id: int
        username: str
        email: EmailStr
        password1: str
        password2: str
    ```

    该模型与之前的模型一样简单，它包含两个密码字段，这两个字段必须匹配才能注册新用户。此外，你还想施加另一个验证——通过反序列化进入模型的 数据不得包含私有数据（如社会保险号码或卡号）。模型验证器允许你执行此类灵活的验证。

1.  继续上一个模型，你可以在类定义下编写以下模型验证器：

    ```py
    @model_validator(mode='after')
    def check_passwords_match(self) -> Self:
        pw1 = self.password1
        pw2 = self.password2
        if pw1 is not None and pw2 is not None and pw1 != pw2:
            raise ValueError('passwords do not match')
        return self
    @model_validator(mode='before')
    @classmethod
    def check_private_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            assert (
                'private_data' not in data
            ), 'Private data should not be included'
        return data
    ```

1.  现在，尝试验证以下数据：

    ```py
    usr_data = {
        "id": 1,
        "username": "freethrow",
        "email": "email@gmail.com",
        "password1": "password123",
        "password2": "password456",
        "private_data": "some private data",
    }
    try:
        user = UserModelV.model_validate(usr_data)
        print(user)
    except ValidationError as e:
        print(e)
    ```

    你只会被告知一个错误——与`before`模式相关的错误，指出不应包含私有数据。

1.  如果你取消注释或删除设置`private_data`字段的行并重新运行示例，错误将变为以下内容：

    ```py
    Value error, passwords do not match [type=value_error, input_value={'id': 1, 'username': 'fr...ssword2': 'password456'}, input_type=dict]
    ```

在上一个例子中涉及了一些新概念；你正在使用 Python 的`Self`类型，它是为了表示包装类的实例而引入的，因此你实际上期望输出是`UserModelV`类的实例。

在`check_private_data`函数中，还有一个新概念，它检查传递给类的数据是否是字典的实例，然后继续验证字典中是否包含不希望的`private_data`字段——这只是 Pydantic 检查传递数据的途径，因为它存储在字典内部。

下一节将详细介绍如何使用 Pydantic 组合嵌套模型以验证越来越复杂的模型。

## 嵌套模型

如果你来自基本的 MongoDB 背景，那么通过组合在 Pydantic 中对嵌套模型的处理非常简单直观。要了解如何实现嵌套模型，最简单的方法是从需要验证的现有数据结构开始，并通过 Pydantic 进行操作：

1.  从返回汽车品牌和型号（或模型）的 JSON 文档结构开始。创建一个名为 `chapter3_16.py` 的新文件，并添加以下代码行：

    ```py
    car_data = {
        "brand": "Ford",
        "models": [
            {"model": "Mustang", "year": 1964},
            {"model": "Focus", "year": 1975},
            {"model": "Explorer", "year": 1999},
        ],
        "country": "USA",
    }
    ```

    您可以从数据结构内部开始，识别最小的单元或最深层嵌套的结构——在这个例子中，最小的单元是 1964 年的福特野马车型。

1.  这可以是第一个 Pydantic 模型：

    ```py
    class CarModel(BaseModel):
        model: str
        year: int
    ```

1.  一旦完成这个初步的抽象，创建品牌模型就变得容易了：

    ```py
    class CarBrand(BaseModel):
        brand: str
        models: List[CarModel]
        country: str
    ```

汽车品牌型号有独特的名称和产地，并包含一系列车型。

模型字段可以是其他模型（或列表、集合或其他序列）并且这个特性使得将 Pydantic 数据结构映射到数据，尤其是 MongoDB 文档，变得非常愉快和直观。

虽然 MongoDB 可以支持多达 100 层的嵌套，但在您的数据建模过程中，您可能不会达到这个限制。然而，值得注意的是，Pydantic 将在您深入数据结构时支持您。从 Python 端嵌入数据也变得更加容易管理，因为您可以确信进入您集合的数据是按照预期存储的。

下一节和最后一节将详细介绍 Pydantic 提供的另一个有用工具——在处理环境变量和设置时提供一些帮助，这是每个与网络相关的项目都会遇到的问题。

## Pydantic Settings

Pydantic Settings 是一个外部包，需要单独安装。它提供了从环境变量或秘密文件中加载设置或配置类的 Pydantic 功能。

这基本上是 Pydantic 网站上的定义（[`docs.pydantic.dev/latest/concepts/pydantic_settings/`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)），整个概念围绕着 `BaseSettings` 类展开。

尝试从此类继承的模型会尝试通过扫描环境来读取任何作为关键字参数传递的字段值。

这种简单的功能允许您从环境变量中定义清晰和直接的配置类。Pydantic 设置也可以自动获取环境修改，并在需要时手动覆盖测试、开发或生产中的设置。

在接下来的练习中，您将创建一个简单的 `pydantic_settings` 设置，这将允许您读取环境变量，并在必要时轻松覆盖它们：

1.  使用 `pip` 安装 Pydantic settings：

    ```py
    pip install pydantic-settings
    ```

1.  在与项目文件同一级别的位置创建一个 `.env` 文件：

    ```py
    API_URL=https://api.com/v2
    SECRET_KEY=s3cretstr1n6
    ```

1.  现在，您可以设置一个简单的 `Settings` 配置（`chapter3_17.py` 文件）：

    ```py
    from pydantic import Field
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        api_url: str = Field(default="")
        secret_key: str = Field(default="")
        class Config:
            env_file = ".env"
    print(Settings().model_dump())
    ```

1.  如果您运行此代码，Python 和 `.env` 文件位于同一路径，您将看到 Pydantic 能够从 `.env` 文件中读取环境变量：

    ```py
    {'api_url': 'https://api.com/v2', 'secret_key': 's3cretstr1n6'}
    ```

    然而，如果您设置了环境变量，它将优先于 `.env` 文件。

1.  你可以通过在 `Settings()` 调用之前添加此行来测试它，并观察程序的输出：

    ```py
    os.environ["API_URL"] = 'http://localhost:8000'
    ```

Pydantic 设置使得管理配置，如 Atlas 和 MongoDB 的 URL、密码散列的秘密以及其他配置，变得更加结构化和有序。

# 摘要

本章详细介绍了 Python 的一些方面，这些方面要么是新的且仍在发展中，要么通常被简单地忽视，例如类型提示，以及它们的使用可能对你的项目产生的影响。

FastAPI 基于 Pydantic 和类型提示。与这些稳固的原则和约定一起工作，将使你的代码更加健壮、可维护和面向未来，即使在与其他框架一起工作时也是如此。你已经拥有坚实的 Python 类型基础，并学习了 Pydantic 提供的基本功能——验证、序列化和反序列化。

你已经学会了如何通过 Pydantic 反序列化、序列化和验证数据，甚至在过程中添加一些转换，创建更复杂的结构。

本章已为你提供了学习更多 FastAPI 的网络特定方面的能力，以及如何在 MongoDB、Python 数据结构和 JSON 之间无缝混合数据。

下一章将探讨 FastAPI 及其 Pythonic 基础。
