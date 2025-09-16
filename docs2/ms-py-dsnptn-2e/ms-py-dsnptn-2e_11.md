

# 第十一章：Python 反模式

在本章的最后部分，我们将探讨 Python 反模式。这些是常见的编程实践，虽然它们并不一定错误，但往往会导致代码效率低下、可读性差和难以维护。通过了解这些陷阱，你可以为你的 Python 应用程序编写更干净、更高效的代码。

在本章中，我们将涵盖以下主要内容：

+   代码风格违规

+   正确性反模式

+   可维护性反模式

+   性能反模式

# 技术要求

请参阅在*第一章*中提出的要求数据。

# 代码风格违规

Python 风格指南，也称为**Python 增强提案第 8 号**（**PEP 8**），为你的代码的可读性和一致性提供了建议，使得开发者能够更容易地在长时间内协作和维护项目。你可以在其官方页面找到风格指南的详细信息：[`peps.python.org/pep-0008`](https://peps.python.org/pep-0008)。在本节中，我们将介绍风格指南的一些建议，以便你在编写应用程序或库的代码时避免它们。

# 修复代码风格违规的工具

注意，我们有如 *Black* ([`black.readthedocs.io/en/stable/`](https://black.readthedocs.io/en/stable/))、*isort* ([`pycqa.github.io/isort/`](https://pycqa.github.io/isort/)) 和/或 *Ruff* ([`docs.astral.sh/ruff/`](https://docs.astral.sh/ruff/)) 等格式化工具，可以帮助你修复不符合风格指南建议的代码。我们不会在这里花费时间讲解如何使用这些工具，因为你可以找到所有需要的文档在它们的官方文档页面上，并且可以在几分钟内开始使用它们。

现在，让我们来探讨我们选定的代码风格建议。

## 缩进

你应该使用每个缩进级别四个空格，并应避免混合使用制表符和空格。

## 最大行长度和空行

风格指南建议将所有代码行限制在最多 79 个字符，以提高可读性。

此外，还有一些与空行相关的规则。首先，你应该在顶级函数和类定义周围使用两个空行。其次，类内部的方法定义应使用单个空行。

例如，以下代码片段的格式是不正确的：

```py
class MyClass:
    def method1(self):
        pass
    def method2(self):
        pass
def top_level_function():
    pass
```

正确的格式如下：

```py
class MyClass:
    def method1(self):
        pass
    def method2(self):
        pass
def top_level_function():
    pass
```

## 导入

你编写、组织和排序导入语句的方式也很重要。根据风格指南，导入应单独成行，并按以下顺序分为三类：标准库导入、相关第三方库导入以及应用程序或库代码库中的本地特定导入。此外，每个组之间应有一个空行。

例如，以下内容不符合风格指南：

```py
import os, sys
import numpy as np
from mymodule import myfunction
```

对于相同的导入，最佳实践如下：

```py
import os
import sys
import numpy as np
from mymodule import myfunction
```

## 命名约定

你应该为变量、函数、类和模块使用描述性的名称。以下是为不同类型情况的具体命名约定：

+   `lower_case_with_underscores`

+   `CapWords`

+   `ALL_CAPS_WITH_UNDERSCORES`

例如，以下不是良好的实践：

```py
def calculateSum(a, b):
    return a + b
class my_class:
    pass
maxValue = 100
```

最佳实践如下：

```py
def calculate_sum(a, b):
    return a + b
class MyClass:
    pass
MAX_VALUE = 100
```

## 注释

注释应该是完整的句子，首字母大写，应该清晰简洁。我们对注释的两种情况有具体的建议——块注释和行内注释：

+   块注释通常适用于其后的某些（或所有）代码，并且缩进与该代码相同级别。块注释的每一行都以 `#` 和一个空格开始。

+   行内注释应谨慎使用。行内注释位于语句的同一行上，至少与语句有两个空格的距离。

例如，以下是一个不良的注释风格：

```py
#This is a poorly formatted block comment.
def foo():  #This is a poorly formatted inline comment.
    pass
```

下面是修复了风格的等效代码：

```py
# This is a block comment.
# It spans multiple lines.
def foo():
    pass  # This is an inline comment.
```

## 表达式和语句中的空白

在以下情况下，你应该避免多余的空白：

+   立即位于括号、方括号或花括号内

+   立即位于逗号、分号或冒号之前

+   在赋值运算符周围留出多个空格以对齐

这结束了我们对最常见代码风格违规的审查。正如之前所说，有工具可以帮助以生产性的方式检测和修复这些违规，它们通常包含在开发工作流程中（例如，通过 `git commit` 钩子以及/或在项目的 CI/CD 流程中）。

# 正确性反模式

如果不解决这些问题，这些反模式可能会导致错误或意外的行为。我们将讨论这些反模式中最常见的一些，以及推荐的替代方法和途径。我们将重点关注以下反模式：

+   使用 `type()` 函数比较类型

+   可变默认参数

+   从类外部访问受保护的成员

注意，使用如 *Visual Studio Code* 或 *PyCharm* 这样的 IDE 或如 *Flake8* 这样的命令行工具可以帮助你在代码中找到这样的不良实践，但了解每个建议及其背后的原因同样重要。

## 使用 `type()` 函数比较类型

有时，为了我们的算法，我们需要通过比较来识别值的类型。人们可能会想到的常见技术是使用 `type()` 函数。但使用 `type()` 来比较对象类型不考虑子类化，并且不如基于使用 `isinstance()` 函数的替代方案灵活。

假设我们有两个类，`CustomListA` 和 `CustomListB`，它们是 `UserList` 类的子类，当定义自定义列表的类时，推荐从该类继承，如下所示：

```py
from collections import UserList
class CustomListA(UserList):
    pass
class CustomListB(UserList):
    pass
```

如果我们想检查一个对象是否是自定义列表类型之一，使用第一种方法，我们会测试 `type(obj) in (CustomListA, CustomListB)` 条件。

或者，我们只需简单地测试`isinstance(obj, UserList)`，这已经足够了，因为`CustomListA`和`CustomListB`是`UserList`的子类。

作为演示，我们编写一个`compare()`函数，使用以下第一种方法：

```py
def compare(obj):
    if type(obj) in (CustomListA, CustomListB):
        print("It's a custom list!")
    else:
        print("It's a something else!")
```

然后，我们编写一个`better_compare()`函数，使用以下替代方法执行等效操作：

```py
def better_compare(obj):
    if isinstance(obj, UserList):
        print("It's a custom list!")
    else:
        print("It's a something else!")
```

以下代码行可以帮助测试这两个函数：

```py
obj1 = CustomListA([1, 2, 3])
obj2 = CustomListB(["a", "b", "c"])
compare(obj1)
compare(obj2)
better_compare(obj1)
better_compare(obj2)
```

完整的演示代码在`ch11/compare_types.py`文件中。运行`python ch11/compare_types.py`命令应给出以下输出：

```py
It's a custom list!
It's a custom list!
It's a custom list!
It's a custom list!
```

这表明两个函数都可以产生预期的结果。但使用推荐技术`isinstance()`的函数更简单易写，并且更灵活，因为它考虑了子类。

## 可变默认参数

当你定义一个带有期望可变值参数的函数，例如列表或字典时，你可能想提供一个默认参数（分别为`[]`或`{}`）。但这样的函数会在调用之间保留更改，这会导致意外的行为。

建议的做法是使用`None`作为默认值，并在需要时在函数内部将其设置为可变数据结构。

让我们创建一个名为`manipulate()`的函数，其`mylist`参数的默认值为`[]`。该函数将`"test"`字符串追加到`mylist`列表中，然后返回它，如下所示：

```py
def manipulate(mylist=[]):
    mylist.append("test")
    return mylist
```

在另一个名为`better_manipulate()`的函数中，其`mylist`参数的默认值为`None`，我们首先将`mylist`设置为`[]`，如果它是`None`，然后在返回之前将`"test"`字符串追加到`mylist`中，如下所示：

```py
def better_manipulate(mylist=None):
    if not mylist:
        mylist = []
    mylist.append("test")
    return mylist
```

以下行帮助我们通过多次使用默认参数调用每个函数来测试每个函数：

```py
if __name__ == "__main__":
    print("function manipulate()")
    print(manipulate())
    print(manipulate())
    print(manipulate())
    print("function better_manipulate()")
    print(better_manipulate())
    print(better_manipulate())
```

运行`python ch11/mutable_default_argument.py`命令应给出以下输出：

```py
function manipulate()
['test']
['test', 'test']
['test', 'test', 'test']
function better_manipulate()
['test']
"test" string several times in the list returned; the string is accumulating because each subsequent time the function has been called, the mylist argument kept its previous value instead of being reset to the empty list. But, with the recommended solution, we see with the result that we get the expected behavior.
			Accessing a protected member from outside a class
			Accessing a protected member (an attribute prefixed with `_`) of a class from outside that class usually calls for trouble since the creator of that class did not intend this member to be exposed. Someone maintaining the code could change or rename that attribute later down the road, and parts of the code accessing it could result in unexpected behavior.
			If you have code that accesses a protected member that way, the recommended practice is to refactor that code so that it is part of the public interface of the class.
			To demonstrate this, let’s define a `Book` class with two protected attributes, `_title` and `_author`, as follows:

```

class Book:

def __init__(self, title, author):

self._title = title

self._author = author

```py

			Now, let’s create another class, `BetterBook`, with the same attributes and a `presentation_line()` method that accesses the `_title` and `_author` attributes and returns a concatenated string based on them. The class definition is as follows:

```

class BetterBook:

def __init__(self, title, author):

self._title = title

self._author = author

def presentation_line(self):

return f"{self._title} by {self._author}"

```py

			Finally, in the code for testing both classes, we get and print the presentation line for an instance of each class, accessing the protected members for the first one (instance of `Book`) and calling the `presentation_line()` method for the second one (instance of `BetterBook`), as follows:

```

if __name__ == "__main__":

b1 = Book(

"Mastering Object-Oriented Python",

"Steven F. Lott",

)

print(

"不良做法：直接访问受保护的成员"

)

print(f"{b1._title} by {b1._author}")

b2 = BetterBook(

"Python 算法",

"Magnus Lie Hetland",

)

print(

"推荐：通过公共接口访问"

)

print(b2.presentation_line())

```py

			The complete code is in the `ch11/ protected_member_of_class.py` file. Running the `python ch11/ protected_member_of_class.py` command gives the following output:

```

不良做法：直接访问受保护的成员

《精通面向对象 Python》由 Steven F. Lott 著

推荐：通过公共接口访问

"Python Algorithms" by Magnus Lie Hetland

```py

			This shows that we get the same result, without any error, in both cases, but using the `presentation_line()` method, as done in the case of the second class, is the best practice. The `_title` and `_author` attributes are protected, so it is not recommended to call them directly. The developer could change those attributes in the future. That is why they must be encapsulated in a public method.
			Also, it is good practice to provide an attribute that encapsulates each protected member of the class using the `@property` decorator, as we have seen in the *Techniques for achieving encapsulation* section of *Chapter 1*, *Foundational* *Design Principles*.
			Maintainability anti-patterns
			These anti-patterns make your code difficult to understand or maintain over time. We are going to discuss several anti-patterns that should be avoided for better quality in your Python application or library’s code base. We will focus on the following points:

				*   Using a wildcard import
				*   **Look Before You Leap** (**LBYL**) versus **Easier to Ask for Forgiveness than** **Permission** (**EAFP**)
				*   Overusing inheritance and tight coupling
				*   Using global variables for sharing data between functions

			As mentioned for the previous category of anti-patterns, using tools such as Flake8 as part of your developer workflow can be handy to help find some of those potential issues when they are already present in your code.
			Using a wildcard import
			This way of importing (`from mymodule import *`) can clutter the namespace and make it difficult to determine where an imported variable or function came from. Also, the code may end up with bugs because of name collision.
			The best practice is to use specific imports or import the module itself to maintain clarity.
			LBYL versus EAFP
			LBYL often leads to more cluttered code, while EAFP makes use of Python’s handling of exceptions and tends to be cleaner.
			For example, we may want to check if a file exists, before opening it, with code such as the following:

```

if os.path.exists(filename):

with open(filename) as f:

print(f.text)

```py

			This is LBYL, and when new to Python, you would think that it is the right way to treat such situations. But in Python, it is recommended to favor EAFP, where appropriate, for cleaner, more Pythonic code. So, the recommended way for the expected result would give the following code:

```

try:

with open(filename) as f:

print(f.text)

except FileNotFoundError:

print("此处无文件")

```py

			As a demonstration, let’s write a `test_open_file()` function that uses the LBYL approach, as follows:

```

def test_open_file(filename):

if os.path.exists(filename):

with open(filename) as f:

print(f.text)

else:

print("此处无文件")

```py

			Then, we add a function that uses the recommended approach:

```

def better_test_open_file(filename):

try:

with open(filename) as f:

print(f.text)

except FileNotFoundError:

print("No file there")

```py

			We can then test these functions with the following code:

```

filename = "no_file.txt"

test_open_file(filename)

better_test_open_file(filename)

```py

			You can check the complete code of the example in the `ch11/lbyl_vs_eafp.py` file, and running it should give the following output:

```

没有该文件

try/except 方法使我们的代码更简洁。

            过度使用继承和紧密耦合

            继承是面向对象编程的一个强大功能，但过度使用它——例如，为每个轻微的行为变化创建一个新的类——会导致类之间的紧密耦合。这增加了复杂性，并使代码更不灵活，更难以维护。

            不推荐创建如下的深层继承层次结构（作为一个简化的例子）：

```py
class GrandParent:
    pass
class Parent(GrandParent):
    pass
class Child(Parent):
    Pass
```

            最佳实践是创建更小、更专注的类，并将它们组合起来以实现所需的行为，如下所示：

```py
class Parent:
    pass
class Child:
    def __init__(self, parent):
        self.parent = parent
```

            如您所记得，这是组合方法，我们在 *第一章* 的 *遵循组合优于继承原则* 部分进行了讨论，*基础* *设计原则*。

            使用全局变量在函数之间共享数据

            全局变量是可以在整个程序中访问的变量，这使得它们在函数之间共享数据时很有吸引力——例如，跨多个模块使用的配置设置或共享资源，如数据库连接。

            然而，它们可能导致应用程序的不同部分意外地修改全局状态，从而导致错误。此外，它们使得扩展应用程序变得更加困难，因为它们可能导致多线程环境中的问题，在多线程环境中，多个线程可能会尝试同时修改全局变量。

            下面是一个不推荐的做法示例：

```py
# Global variable
counter = 0
def increment():
    global counter
    counter += 1
def reset():
    global counter
    counter = 0
```

            而不是使用全局变量，你应该将所需的数据作为参数传递给函数或封装状态在类中，这提高了代码的模块化和可测试性。因此，对于反例的最佳实践是定义一个包含 `counter` 属性的 `Counter` 类，如下所示：

```py
class Counter:
    def __init__(self):
        self.counter = 0
    def increment(self):
        self.counter += 1
    def reset(self):
        self.counter = 0
```

            接下来，我们添加测试 `Counter` 类的代码如下：

```py
if __name__ == "__main__":
    c = Counter()
    print(f"Counter value: {c.counter}")
    c.increment()
    print(f"Counter value: {c.counter}")
    c.reset()
```

            你可以在 `ch11/instead_of_global_variable.py` 文件中查看示例的完整代码，运行它应该会给出以下输出：

```py
Counter value: 0
Counter value: 1
```

            这表明使用类而不是全局变量是有效且可扩展的，因此是推荐的做法。

            性能反模式

            这些反模式会导致效率低下，尤其是在大型应用程序或数据密集型任务中，这会降低性能。我们将关注以下此类反模式：

                +   在循环中不使用 `.join()` 连接字符串

                +   使用全局变量进行缓存

            让我们开始吧。

            在循环中不使用 .join() 连接字符串

            在循环中使用 `+` 或 `+=` 连接字符串会每次创建一个新的字符串对象，这是低效的。最好的解决方案是使用字符串的 `.join()` 方法，该方法专为从序列或可迭代对象中连接字符串时的效率而设计。

            让我们创建一个`concatenate()`函数，其中我们使用`+=`来连接字符串列表中的项，如下所示：

```py
def concatenate(string_list):
    result = ""
    for item in string_list:
        result += item
    return result
```

            然后，让我们创建一个`better_concatenate()`函数，以实现相同的结果，但使用`str.join()`方法，如下所示：

```py
def better_concatenate(string_list):
    result = "".join(string_list)
    return result
```

            我们可以使用以下方式测试这两个函数：

```py
if __name__ == "__main__":
    string_list = ["Abc", "Def", "Ghi"]
    print(concatenate(string_list))
    print(better_concatenate(string_list))
```

            运行代码（在`ch11/concatenate_strings_in_loop.py`文件中）会得到以下输出：

```py
AbcDefGhi
.join() is the recommended practice for performance reasons.
			Using global variables for caching
			Using global variables for caching can seem like a quick and easy solution but often leads to poor maintainability, potential data consistency issues, and difficulties in managing the cache life cycle effectively. A more robust approach involves using specialized caching libraries designed to handle these aspects more efficiently.
			In this example (in the `ch11/caching/using_global_var.py` file), a global dictionary is used to cache results from a function that simulates a time-consuming operation (for example, a database query) done in the `perform_expensive_operation()` function. The complete code for this demonstration is as follows:

```

import time

import random

# 全局变量作为缓存

_cache = {}

def get_data(query):

if query in _cache:

return _cache[query]

else:

result = perform_expensive_operation(query)

_cache[query] = result

return result

def perform_expensive_operation(user_id):

time.sleep(random.uniform(0.5, 2.0))

user_data = {

1: {"name": "Alice", "email": "alice@example.com"},

2: {"name": "Bob", "email": "bob@example.com"},

3: {"name": "Charlie", "email": "charlie@example.com"},

}

result = user_data.get(user_id, {"error": "User not found"})

return result

if __name__ == "__main__":

print(get_data(1))

print(get_data(1))

```py

			Testing the code by running the `python ch11/caching/using_global_var.py` command gives the following output:

```

{'name': 'Alice', 'email': 'alice@example.com'}

functools.lru_cache() 函数。lru_cache 装饰器提供的 lru_cache 是针对性能优化的，使用高效的数据结构和算法来管理缓存。

            这就是如何使用`functools.lru_cache`实现从耗时函数缓存结果的功能。完整的代码（在`ch11/caching/using_lru_cache.py`文件中）如下所示：

```py
import random
import time
from functools import lru_cache
@lru_cache(maxsize=100)
def get_data(user_id):
    return perform_expensive_operation(user_id)
def perform_expensive_operation(user_id):
    time.sleep(random.uniform(0.5, 2.0))
    user_data = {
        1: {"name": "Alice", "email": "alice@example.com"},
        2: {"name": "Bob", "email": "bob@example.com"},
        3: {"name": "Charlie", "email": "charlie@example.com"},
    }
    result = user_data.get(user_id, {"error": "User not found"})
    return result
if __name__ == "__main__":
    print(get_data(1))
    print(get_data(1))
    print(get_data(2))
    print(get_data(99))
```

            要测试此代码，请运行`python ch11/caching/using_lru_cache.py`命令。你应该得到以下输出：

```py
{'name': 'Alice', 'email': 'alice@example.com'}
{'name': 'Alice', 'email': 'alice@example.com'}
{'name': 'Bob', 'email': 'bob@example.com'}
{'error': 'User not found'}
```

            如我们所见，这种方法不仅增强了缓存机制的鲁棒性，还提高了代码的可读性和可维护性。

            摘要

            理解和避免常见的 Python 反模式将帮助你编写更干净、更高效、更易于维护的代码。

            首先，我们介绍了常见的 Python 代码风格违规。然后，我们讨论了几种与正确性相关的反模式，这些反模式可能导致错误。接下来，我们介绍了除了代码风格本身之外，对代码可读性和可维护性不利的实践。最后，我们看到了一些应该避免的反模式，以编写具有良好性能的代码。

            总是记住——最好的代码不仅仅是让它工作，还要让它工作得很好。更进一步，理想情况下，它应该易于维护。

            我们终于到达了这本书的结尾。这是一段旅程。我们从主要设计原则开始，然后转向介绍最流行的设计模式，以及它们如何应用于 Python，最后简要介绍了 Python 的反模式。这有很多！我们讨论的思想和例子帮助我们思考不同的实现选项或技术，以便在遇到用例时选择。无论你选择哪种解决方案，都要记住 Python 倾向于简单性，尽量使用被认为是 Pythonic 的模式和技巧，并避免 Python 的反模式。

```py

```

```py

```
