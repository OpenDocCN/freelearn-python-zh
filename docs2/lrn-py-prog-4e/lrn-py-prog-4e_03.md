

# 第三章：条件和迭代

> “你能告诉我，请，我应该从这里走哪条路吗？”
> 
> “这很大程度上取决于你想要去哪里。”
> 
> ——刘易斯·卡罗尔，《爱丽丝梦游仙境》

在上一章中，我们探讨了 Python 的内置数据类型。现在你已经熟悉了各种形式和形状的数据，是时候开始了解程序如何使用它了。

根据维基百科：

> 在计算机科学中，**控制流**（或**控制流程**）是指 imperative 程序中各个语句、指令或函数调用的执行或评估的顺序。

控制程序流程的两种主要方式是**条件编程**（也称为**分支**）和**循环**。这些技术可以组合起来产生无数种程序。我们不会尝试记录所有组合循环和分支的方法，而是会给你一个概述 Python 中可用的流程控制结构。然后，我们将带你通过几个示例程序。这样，你应该能更好地理解条件编程和循环是如何被使用的。

在本章中，我们将涵盖以下内容：

+   条件编程

+   Python 中的循环

+   赋值表达式

+   快速浏览 `itertools` 模块

# 条件编程

条件编程，或分支，是你每天每时每刻都在做的事情。本质上，它包括评估条件和决定采取什么行动：*如果绿灯亮了，那么我可以过马路*；*如果下雨了，那么我会带伞*；*如果我上班迟到了，那么我会给经理打电话*。

## if 语句

Python 中条件编程的主要工具是 `if` 语句。它的功能是评估一个表达式，并根据结果选择执行代码的哪个部分。像往常一样，让我们来看一个例子：

```py
# conditional.1.py
late = True
if late:
    print("I need to call my manager!") 
```

这是最简单的例子：if 语句在布尔上下文中评估 `late` 表达式（就像我们调用 `bool(late)` 一样）。如果评估结果为 `True`，则立即进入 if 语句之后的代码体。注意，`print` 指令是缩进的，这意味着它属于由 if 子句定义的作用域。执行此代码会产生以下结果：

```py
$ python conditional.1.py
I need to call my manager! 
```

由于 `late` 是 `True`，执行了 `print()` 语句。我们可以通过添加 `else` 子句来扩展基本的 `if` 语句。这提供了一组替代指令，在 if 子句中的表达式评估为 `False` 时执行。

```py
# conditional.2.py
late = False
if late:
    print("I need to call my manager!")  # 1
else:
    print("no need to call my manager...")  # 2 
```

这次，我们设置 `late = False`，因此当我们执行代码时，结果会有所不同：

```py
$ python conditional.2.py
no need to call my manager... 
```

根据 `late` 的评估结果，我们可以进入块 `# 1` 或块 `# 2`，但不能同时进入。当 `late` 评估为 `True` 时，执行块 `# 1`，而当 `late` 评估为 `False` 时，执行块 `# 2`。尝试将 `False` / `True` 值分配给 `late` 并观察输出如何变化。

## 特殊的 else: elif

到目前为止，当你只有一个条件要评估，并且最多只有两条替代路径（`if`和`else`子句）时，我们所看到的是足够的。然而，有时你可能需要评估多个条件以从多个路径中选择。为了演示这一点，我们需要一个有更多选项可供选择的例子。

这次，我们将创建一个简单的税务计算器。假设税收是这样确定的：如果你的收入低于$10,000，你不需要支付任何税款。如果它在$10,000 和$30,000 之间，你必须支付 20%的税款。如果它在$30,000 和$100,000 之间，你支付 35%的税款，如果你有幸赚得超过$100,000，你必须支付 45%的税款。让我们把这个翻译成 Python 代码：

```py
# taxes.py
income = 15000
if income < 10000:
    tax_coefficient = 0.0  # 1
elif income < 30000:
    tax_coefficient = 0.2  # 2
elif income < 100000:
    tax_coefficient = 0.35  # 3
else:
    tax_coefficient = 0.45  # 4
print(f"You will pay: ${income * tax_coefficient} in taxes") 
```

当我们执行这段代码时，会得到以下输出：

```py
$ python taxes.py
You will pay: $3000.0 in taxes 
```

让我们逐行分析这个例子。我们首先设置收入值。在这个例子中，你的收入是$15,000。我们进入`if`语句。注意，这次我们还引入了`elif`子句，它是`else-if`的缩写。它与普通的`else`子句不同，因为它也有自己的条件。`income < 10000`的`if`表达式评估为`False`；因此，代码块`# 1`没有被执行。

控制权传递到下一个条件：`elif income < 30000`。这个条件评估为`True`；因此，代码块`# 2`被执行，因此 Python 随后在完整的`if` / `elif` / `elif` / `else`结构（从现在起我们可以简单地称之为`if`语句）之后继续执行。`if`语句之后只有一个指令：`print()`调用，它产生输出告诉我们今年我们将支付$3000.0 的税款（*15,000 * 20%*）。注意，顺序是强制性的：`if`首先，然后（可选地）尽可能多的`elif`子句，最后（可选地）一个单独的`else`子句。

无论每个代码块中有多少行代码，只要其中一个条件评估为`True`，相关的代码块就会被执行，然后执行会继续到整个子句之后。如果没有条件评估为`True`（例如，`income = 200000`），则`else`子句的主体将被执行（代码块`# 4`）。这个例子扩展了我们对于`else`子句行为的理解。它的代码块会在前面的`if` / `elif` /.../ `elif`表达式都没有评估为`True`时执行。

尝试修改 `income` 的值，直到你可以随意执行任何代码块。还要测试在 `if` 和 `elif` 子句中布尔表达式的值发生变化的 **边界** 上的行为。彻底测试边界对于确保代码的正确性至关重要。我们应该允许你在 18 岁或 17 岁时驾驶吗？我们是用 `age < 18` 还是 `age <= 18` 来检查你的年龄？你无法想象我们有多少次不得不修复由使用错误的运算符引起的微妙错误，所以请继续实验代码。将一些 `<` 改为 `<=`，并将 `income` 设置为边界值之一（10,000、30,000 或 100,000），以及任何介于这些值之间的值。看看结果如何变化，并在继续之前对其有一个良好的理解。

## 嵌套 `if` 语句

你也可以嵌套 `if` 语句。让我们看看另一个例子来展示如何做到这一点。比如说，如果你的程序遇到了错误。如果警报系统是控制台，我们就打印错误信息。

如果警报系统是电子邮件，错误的严重性决定了我们应该将警报发送到哪个地址。如果警报系统不是控制台或电子邮件，我们不知道该怎么做，所以我们什么也不做。让我们把这个写成代码：

```py
# errorsalert.py
alert_system = "console"  # other value can be "email"
error_severity = "critical"  # other values: "medium" or "low"
error_message = "Something terrible happened!"
if alert_system == "console":  # outer
    print(error_message)  # 1
elif alert_system == "email":
    if error_severity == "critical":  # inner
        send_email("admin@example.com", error_message)  # 2
    elif error_severity == "medium":
        send_email("support.1@example.com", error_message)  # 3
    else:
        send_email("support.2@example.com", error_message)  # 4 
```

这里，我们有一个嵌套在 *外部* `if` 语句的 `elif` 子句体中的 *内部* `if` 语句。注意，嵌套是通过缩进内部 `if` 语句来实现的。

让我们逐步执行代码，看看会发生什么。我们首先为 `alert_system`、`error_severity` 和 `error_message` 赋值。当我们进入外部 `if` 语句时，如果 `alert_system == "console"` 评估为 `True`，则执行代码块 `# 1`，然后不再发生其他事情。另一方面，如果 `alert_system == "email"` 评估为 `True`，那么我们就进入内部 `if` 语句。在内部 `if` 语句中，`error_severity` 决定了我们应该向管理员、一级支持还是二级支持发送电子邮件（代码块 `# 2`、`# 3` 和 `# 4`）。在这个例子中，`send_email()` 函数没有定义，所以尝试运行它会给你一个错误。在本书源代码中可以找到的 `errorsalert.py` 模块中，我们包括了一个技巧来将那个调用重定向到一个普通的 `print()` 函数，这样你就可以在控制台上进行实验，而实际上并不发送电子邮件。尝试更改值，看看它如何工作。

## 三元运算符

我们接下来想展示的是 **三元运算符**。在 Python 中，这也被称为 **条件表达式**。它看起来和表现就像是一个简短的、内联的 `if` 语句。当你只想根据某个条件在两个值之间进行选择时，使用三元运算符有时比使用完整的 `if` 语句更容易和更易读。例如，而不是：

```py
# ternary.py
order_total = 247  # GBP
# classic if/else form
if order_total > 100:
    discount = 25  # GBP
else:
    discount = 0  # GBP
print(order_total, discount) 
```

我们可以写成：

```py
# ternary.py
# ternary operator
discount = 25 if order_total > 100 else 0
print(order_total, discount) 
```

对于这种简单的情况，我们觉得能够用一行而不是四行来表示这种逻辑非常方便。记住，作为一个程序员，你花在阅读代码上的时间要比编写代码的时间多得多，所以 Python 的简洁性是无价的。

在某些语言（如 C 或 JavaScript）中，三元运算符甚至更加简洁。例如，上面的代码可以写成：

```py
discount = order_total > 100 ? 25 : 0; 
```

虽然 Python 的版本稍微有点冗长，但我们认为它通过更容易阅读和理解来弥补了这一点。

你清楚三元运算符的工作原理吗？它相当简单；`something if condition else something-else`在条件`condition`评估为`True`时评估为`something`。否则，如果`condition`是`False`，表达式评估为`something-else`。

## 模式匹配

**结构化模式匹配**，通常简称为**模式匹配**，是一个相对较新的特性，它是在 Python 3.10 版本中通过 PEP 634（[`peps.python.org/pep-0634`](https://peps.python.org/pep-0634)）引入的。它部分受到了像 Haskel、Erlang、Scala、Elixir 和 Ruby 等语言的模式匹配能力的影响。

简而言之，`match`语句将一个值与一个或多个**模式**进行比较，然后执行与第一个匹配的模式关联的代码块。让我们看一个简单的例子：

```py
# match.py
day_number = 4
**match** day_number:
    **case** 1 | 2 | 3 | 4 | 5:
        print("Weekday")
    **case** 6:
        print("Saturday")
    **case** 7:
        print("Sunday")
    **case** _:
        print(f"{day_number} is not a valid day number") 
```

我们在进入`match`语句之前初始化`day_number`。`match`语句将尝试将`day_number`的值与一系列模式匹配，每个模式都由**case**关键字引入。在我们的例子中，我们有四个模式。第一个`1 | 2 | 3 | 4 | 5`将匹配任何值`1`、`2`、`3`、`4`或`5`。这被称为**或模式**；它由多个通过`|`分隔的子模式组成。当任何子模式（在本例中，是字面值`1`、`2`、`3`、`4`和`5`）匹配时，它就会匹配。我们例子中的第二个和第三个模式分别只包含整数字面量`6`和`7`。最后一个模式`_`是一个**通配符模式**；它是一个通用的匹配任何值的模式。一个`match`语句最多只能有一个通配符模式，如果存在，它必须是最后一个模式。

将会执行第一个与模式匹配的 case 块的主体。之后，执行将继续在`match`语句下方进行，而不会评估任何剩余的模式。如果没有任何模式匹配，执行将继续在`match`语句下方进行，而不会执行任何 case 主体。在我们的例子中，第一个模式匹配，所以`print("Weekday")`被执行。花点时间实验这个例子。尝试改变`day_number`的值，看看运行时会发生什么。

`match` 语句类似于 C++ 和 JavaScript 等语言中的 `switch` / `case` 语句。然而，它比这更强大。可用的不同类型的模式种类繁多，以及组合模式的能力，让你能够做比简单的 C++ `switch` 语句更多的事情。例如，Python 允许你匹配序列、字典，甚至自定义类。你还可以在模式中捕获并将值分配给名称。我们在这里没有足够的空间涵盖你可以用模式匹配做的一切，但我们鼓励你学习 PEP 636 中的教程（[`peps.python.org/pep-0636`](https://peps.python.org/pep-0636)）以了解更多信息。

现在你已经了解了关于控制代码路径的一切，让我们继续下一个主题：*循环*。

# 循环

如果你在其他编程语言中有任何循环的经验，你会发现 Python 的循环方式略有不同。首先，什么是循环？**循环**意味着能够根据循环参数重复执行代码块多次。有不同的循环结构，用于不同的目的，Python 将它们简化为只有两个，你可以使用它们来实现你需要的一切。这些是 `for` 和 `while` 语句。

虽然技术上可以使用任何一个来完成需要循环的任务，但它们确实有不同的用途。我们将在本章中彻底探讨这种差异。到本章结束时，你将知道何时使用 `for` 循环，何时使用 `while` 循环。

## for 循环

当需要遍历序列，如列表、元组或对象集合时，使用 `for` 循环。让我们从一个简单的例子开始，并在此基础上扩展概念，看看 Python 语法允许我们做什么：

```py
# simple.for.py
for number in [0, 1, 2, 3, 4]:
    print(number) 
```

这段简单的代码，当执行时，会打印出从 `0` 到 `4` 的所有数字。`for` 循环的主体（`print()` 行）对于列表 `[0, 1, 2, 3, 4]` 中的每个值都会执行一次。在第一次迭代中，`number` 被分配给序列中的第一个值；在第二次迭代中，`number` 取第二个值；依此类推。在序列的最后一个项目之后，循环结束，执行恢复正常，继续执行循环之后的代码。

### 遍历范围

我们经常需要遍历一系列数字，如果必须通过硬编码列表来这样做，将会非常繁琐。在这种情况下，`range()` 函数就派上用场了。让我们看看之前代码片段的等效代码：

```py
# simple.for.py
for number in range(5):
    print(number) 
```

`range()` 函数在 Python 程序中广泛用于创建序列。你可以用单个值调用它，该值作为 `stop`（计数将从 `0` 开始）。你也可以传递两个值（`start` 和 `stop`），甚至三个（`start`、`stop` 和 `step`）。查看以下示例：

```py
>>> list(range(10))  # one value: from 0 to value (excluded)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> list(range(3, 8))  # two values: from start to stop (excluded)
[3, 4, 5, 6, 7]
>>> list(range(-10, 10, 4))  # three values: step is added
[-10, -6, -2, 2, 6] 
```

目前，忽略我们需要将 `range(...)` 包裹在一个列表中。我们将在 *第五章* *列表推导式和生成器* 中解释这样做的原因。你可以看到，其行为类似于切片（我们在上一章中描述过）：`start` 包含在内，`stop` 不包含，你可以添加一个可选的 `step` 参数，默认为 `1`。

尝试修改 `simple.for.py` 代码中 `range()` 调用的参数，看看它会打印出什么。

### 遍历一个序列

现在我们有了遍历序列的所有工具，让我们在此基础上构建一个例子：

```py
# simple.for.2.py
surnames = ["Rivest", "Shamir", "Adleman"]
for position in range(len(surnames)):
    print(position, surnames[position]) 
```

之前的代码给游戏增加了一点点复杂性。执行结果将显示：

```py
$ python simple.for.2.py
0 Rivest
1 Shamir
2 Adleman 
```

让我们使用**由内而外**的技术来分解它。我们从我们试图理解的内部最深处开始，然后向外扩展。所以 `len(surnames)` 是姓氏列表的长度：`3`。因此，`range(len(surnames))` 实际上被转换成了 `range(3)`。这给我们 `[0, 3)` 的范围，即序列 `(0, 1, 2)`。这意味着 `for` 循环将运行三次迭代。在第一次迭代中，`position` 将取值为 `0`，在第二次迭代中，它将取值为 `1`，在第三次和最后一次迭代中取值为 `2`。在这里，`(0, 1, 2)` 代表 `surnames` 列表可能的索引位置。在位置 `0` 处，我们找到 `"Rivest"`；在位置 `1` 处，`"Shamir"`；在位置 `2` 处，`"Adleman"`。如果你对这三位男士共同创造了什么感到好奇，将 `print(position, surnames[position])` 改为 `print(surnames[position][0], end="")`，在循环外部添加一个 `print()`，然后再次运行代码。

现在，这种循环风格更接近于 Java 或 C 等语言。在 Python 中，很少看到这样的代码。你可以直接遍历任何序列或集合，因此没有必要在每次迭代中获取位置列表并从序列中检索元素。让我们将示例改为更 Pythonic 的形式：

```py
# simple.for.3.py
surnames = ["Rivest", "Shamir", "Adleman"]
for surname in surnames:
    print(surname) 
```

`for` 循环可以遍历 `surnames` 列表，并在每次迭代中按顺序返回每个元素。运行此代码将逐个打印出三个姓氏，这使阅读更加容易。

然而，如果你想要打印位置呢？或者如果你需要它呢？你应该回到 `range(len(...))` 的形式吗？不。你可以使用内置的 `enumerate()` 函数，如下所示：

```py
# simple.for.4.py
surnames = ["Rivest", "Shamir", "Adleman"]
for position, surname in enumerate(surnames):
    print(position, surname) 
```

这段代码也非常有趣。注意，`enumerate()` 在每次迭代中返回一个包含 `(position, surname)` 的二元组，但它仍然比 `range(len(...))` 例子更易读（并且更高效）。你可以使用 `start` 参数调用 `enumerate()`，例如 `enumerate(iterable, start)`，它将从 `start` 开始，而不是 `0`。这只是另一个小细节，展示了在设计 Python 时投入了多少思考，以便让生活更轻松。

你可以使用`for`循环遍历列表、元组，以及在 Python 中称为**可迭代对象**的任何东西。这是一个重要的概念，所以让我们更详细地讨论它。

## 迭代器和可迭代对象

根据 Python 文档（[`docs.python.org/3.12/glossary.html#term-iterable`](https://docs.python.org/3.12/glossary.html#term-iterable)），**可迭代对象**是：

> 能够一次返回其成员的一个对象。可迭代对象包括所有序列类型（如列表、str 和 tuple）以及一些非序列类型，如 dict 和文件对象。

简而言之，当你写下`for k in sequence: ... body ...`时，`for`循环会请求`sequence`的下一个元素，得到一些东西，将其称为`k`，然后执行其主体。然后，再次，`for`循环会请求`sequence`的下一个元素，再次将其称为`k`，再次执行主体，依此类推，直到序列耗尽。空序列将导致主体执行零次。

一些数据结构在迭代时按顺序产生它们的元素，例如列表、元组、字典和字符串，而其他数据结构，如集合，则不按顺序。Python 通过一种称为**迭代器**的对象类型为我们提供了遍历可迭代对象的能力，迭代器是一个表示数据流的对象。

在实践中，整个可迭代/迭代器机制都隐藏在代码背后。除非你需要出于某种原因编写自己的可迭代或迭代器，否则你不必过多担心这一点。然而，了解 Python 如何处理这个关键的控制流方面非常重要，因为它决定了我们编写代码的方式。

我们将在第五章*理解与生成器*和第六章*面向对象编程、装饰器和迭代器*中更详细地介绍迭代。

## 遍历多个序列

让我们看看另一个如何遍历相同长度的两个序列并成对处理它们各自的元素的例子。假设我们有一个包含人名的列表和一个表示他们年龄的数字列表。我们想要为每个人打印一行包含人名/年龄的配对。让我们从一个例子开始，我们将逐步对其进行改进：

```py
# multiple.sequences.py
people = ["Nick", "Rick", "Roger", "Syd"]
ages = [23, 24, 23, 21]
for position in range(len(people)):
    person = people[position]
    age = ages[position]
    print(person, age) 
```

到现在为止，这段代码应该是直截了当的。我们遍历位置列表（`0`、`1`、`2`、`3`），因为我们想要从两个不同的列表中检索元素。执行它，我们得到以下结果：

```py
$ python multiple.sequences.py
Nick 23
Rick 24
Roger 23
Syd 21 
```

代码是可行的，但并不非常符合 Python 风格。必须获取`people`的长度、构造一个`range`，然后遍历它，这显得有些繁琐。对于某些数据结构，按位置检索项目可能也很昂贵。如果能直接遍历序列，就像处理单个序列一样，那就更好了。让我们尝试使用`enumerate()`来改进它：

```py
# multiple.sequences.enumerate.py
people = ["Nick", "Rick", "Roger", "Syd"]
ages = [23, 24, 23, 21]
for position, person in enumerate(people):
    age = ages[position]
    print(person, age) 
```

这样更好，但仍然不完美。我们正确地迭代了`people`，但我们仍然使用位置索引获取`age`，这是我们想要丢弃的。我们可以通过使用我们在上一章中遇到的`zip()`函数来实现这一点。让我们使用它：

```py
# multiple.sequences.zip.py
people = ["Nick", "Rick", "Roger", "Syd"]
ages = [23, 24, 23, 21]
for person, age in zip(people, ages):
    print(person, age) 
```

这比原始版本要优雅得多。当`for`循环请求`zip(sequenceA, sequenceB)`的下一个元素时，它得到一个元组，该元组被解包到`person`和`age`中。元组将包含与提供给`zip()`函数的序列数量一样多的元素。让我们在先前的例子上稍作扩展：

```py
# multiple.sequences.unpack.py
people = ["Nick", "Rick", "Roger", "Syd"]
ages = [23, 24, 23, 21]
instruments = ["Drums", "Keyboards", "Bass", "Guitar"]
for person, age, instrument in zip(people, ages, instruments):
    print(person, age, instrument) 
```

在前面的代码中，我们添加了`instruments`列表。现在，我们向`zip()`函数提供了三个序列，每次迭代`for`循环都会返回一个*三元组*。元组的元素被解包并分配给`person`、`age`和`instrument`。请注意，元组中元素的顺序与`zip()`调用中序列的顺序一致。执行代码将产生以下结果：

```py
$ python multiple.sequences.unpack.py
Nick 23 Drums
Rick 24 Keyboards
Roger 23 Bass
Syd 21 Guitar 
```

注意，在遍历多个序列时，不需要解包元组。你可能在`for`循环体内部将元组作为一个整体进行操作。当然，这样做是完全可能的：

```py
# multiple.sequences.tuple.py
people = ["Nick", "Rick", "Roger", "Syd"]
ages = [23, 24, 23, 21]
instruments = ["Drums", "Keyboards", "Bass", "Guitar"]
for data in zip(people, ages, instruments):
    print(data) 
```

这几乎与先前的例子相同。不同之处在于，我们不是解包从`zip(...)`得到的元组，而是将整个元组赋值给`data`。

## `while`循环

在前面的页面中，我们看到了`for`循环的实际应用。当你需要遍历一个序列或集合时，它很有用。当你需要决定使用哪种循环结构时，要记住的关键点是`for`循环最适合在必须遍历容器对象或其他可迭代对象的元素的情况下使用。

然而，还有其他情况，你可能只需要循环直到满足某个条件，或者无限循环直到应用程序停止。在这种情况下，我们没有可以迭代的，所以`for`循环可能不是一个好的选择。对于这种情况，`while`循环更为合适。

`while`循环与`for`循环类似，因为两者都会重复执行一系列指令。不同之处在于`while`循环不是遍历一个序列。相反，只要满足某个条件，它就会循环。当条件不再满足时，循环结束。

如同往常，让我们看一个例子，以帮助我们更清晰地理解。我们想要打印一个正数的二进制表示。为此，我们可以使用一个简单的算法，通过不断除以二直到为零，并收集余数。当我们反转收集到的余数列表时，我们得到我们开始时的数字的二进制表示。例如，如果我们想要十进制数 6 的二进制表示，步骤如下：

1.  *6 / 2 = 3* 余 *0* 。

1.  *3 / 2 = 1* 余 *1* 。

1.  *1 / 2 = 0* 余 *1* 。

1.  余数列表是*0, 1, 1*。

1.  反转这个结果，我们得到*1, 1, 0*，这也是*6*的二进制表示*110*。

让我们将这个例子翻译成 Python 代码。我们将计算数字 39 的二进制表示，它是 100111：

```py
# binary.py
n = 39
remainders = []
while **n >** **0**:
    remainder = n % 2  # remainder of division by 2
    remainders.append(remainder)  # we keep track of remainders
    n //= 2  # we divide n by 2
remainders.reverse()
print(remainders) 
```

在前面的代码中，我们突出了`n > 0`，这是保持循环的条件。注意代码如何与我们所描述的算法相匹配：只要`n`大于`0`，我们就除以`2`并将余数添加到列表中。在最后（当`n`达到`0`时），我们反转余数列表以获取`n`原始值的二进制表示。

我们可以使用`divmod()`函数使代码更简洁（并且更符合 Python 风格）。`divmod()`函数接受一个数和一个除数，并返回一个元组，包含整数除法的结果及其余数。例如，`divmod(13, 5)`将返回`(2, 3)`，确实，*5 * 2 + 3 = 13*。

```py
# binary.2.py
n = 39
remainders = []
while n > 0:
    **n, remainder =** **divmod****(n,** **2****)**
    remainders.append(remainder)
remainders.reverse()
print(remainders) 
```

现在，我们将`n`重新赋值为除以`2`的结果，并将余数添加到余数列表中，这一行就完成了。

内置函数`bin()`返回一个数的二进制表示。所以，除了示例或作为练习，你不需要在 Python 中自己实现它。

注意，`while`循环中的条件是继续循环的条件。如果它评估为`True`，则执行主体，然后进行另一个评估，依此类推，直到条件评估为`False`。当这种情况发生时，循环会立即停止，而不会执行其主体。如果条件永远不会评估为`False`，则循环会变成所谓的**无限循环**。无限循环在从网络设备轮询时使用，例如：你询问套接字是否有数据，如果有，你将对其进行一些操作，然后你等待一小段时间，然后再次询问套接字，如此反复，永远不会停止。

为了更好地说明`for`循环和`while`循环之间的区别，让我们使用`while`循环修改之前的例子（`multiple.sequences.py`）：

```py
# multiple.sequences.while.py
people = ["Nick", "Rick", "Roger", "Syd"]
ages = [23, 24, 23, 21]
**position =** **0**
while **position <** **len****(people)**:
    person = people[position]
    age = ages[position]
    print(person, age)
    **position +=** **1** 
```

在前面的代码中，我们突出了`position`变量的*初始化*、*条件*和*更新*，这使得通过手动处理迭代来模拟等效的`for`循环代码成为可能。任何可以用`for`循环完成的事情也可以用`while`循环完成，尽管你可以看到为了达到相同的结果，你需要经过一些样板代码。反之亦然，但除非你有这样做的原因，否则你应该使用适合的工具来完成工作。

总结一下，当你需要遍历可迭代对象时使用`for`循环，当你需要根据条件是否满足来循环时使用`while`循环。如果你记住这两个目的之间的区别，你就永远不会选择错误的循环结构。

现在我们来看看如何改变循环的正常流程。

## break 和 continue 语句

有时候你需要改变循环的正常流程。你可以跳过单个迭代（你想跳过多少次就跳过多少次），或者你可以完全跳出循环。跳过迭代的常见用例是，例如，当你正在遍历一个项目列表，但你只需要处理满足某些条件的那些项目。另一方面，如果你正在遍历一个集合以搜索满足某些要求的项，你可能想在找到你想要的东西时立即跳出循环。有无数可能的情况；让我们一起分析几个例子，以展示这在实践中是如何工作的。

假设你想要对今天到期的所有产品应用 20%的折扣。你可以通过使用`continue`语句来实现这一点，它告诉循环结构（`for`或`while`）立即停止执行体并转到下一个迭代（如果有的话）：

```py
# discount.py
from datetime import date, timedelta
today = date.today()
tomorrow = today + timedelta(days=1)  # today + 1 day is tomorrow
products = [
    {"sku": "1", "expiration_date": today, "price": 100.0},
    {"sku": "2", "expiration_date": tomorrow, "price": 50},
    {"sku": "3", "expiration_date": today, "price": 20},
]
for product in products:
    print("Processing sku", product["sku"])
    if product["expiration_date"] != today:
        **continue**
    product["price"] *= 0.8  # equivalent to applying 20% discount
    print("Sku", product["sku"], "price is now", product["price"]) 
```

我们首先导入`date`和`timedelta`对象，然后设置我们的产品。那些`sku`为`1`和`3`的产品有一个到期日期为`today`，这意味着我们想要对它们应用 20%的折扣。我们遍历每个`product`并检查到期日期。如果到期日期不匹配`today`，我们不想执行体中的其余部分，所以我们执行`continue`语句。循环体的执行停止，继续到下一个迭代。如果我们运行`discount.py`模块，这是输出：

```py
$ python discount.py
Processing sku 1
Sku 1 price is now 80.0
Processing sku 2
Processing sku 3
Sku 3 price is now 16.0 
```

如你所见，对于`sku`编号为`2`，体中的最后两行并没有被执行。

现在我们来看一个跳出循环的例子。假设我们想知道列表中的至少一个元素在传递给`bool()`函数时是否评估为`True`。既然我们需要知道是否至少有一个，当我们找到它时，我们就不需要继续扫描列表了。在 Python 代码中，这相当于使用`break`语句。让我们把这个写下来：

```py
# any.py
items = [0, None, 0.0, True, 0, 7]  # True and 7 evaluate to True
found = False  # this is called a "flag"
for item in items:
    print("scanning item", item)
    if item:
        found = True  # we update the flag
        **break**
if found:  # we inspect the flag
    print("At least one item evaluates to True")
else:
    print("All items evaluate to False") 
```

上述代码使用了常见的编程模式；你在开始检查项目之前设置一个**标志**变量。如果你找到一个符合你标准的元素（在这个例子中，评估为`True`），你更新标志并停止迭代。迭代完成后，你检查标志并根据情况采取行动。执行结果如下：

```py
$ python any.py
scanning item 0
scanning item None
scanning item 0.0
scanning item True
At least one item evaluates to True 
```

看看执行在找到`True`后是如何停止的？`break`语句与`continue`类似，因为它立即停止执行循环体，但它还阻止了进一步的迭代运行，实际上是从循环中跳出的。

没有必要编写代码来检测序列中是否至少有一个元素评估为`True`，因为内置函数`any()`正是做这个。

你可以在循环体（`for`或`while`）的任何地方使用你需要的任意多个`continue`或`break`语句，你甚至可以在同一个循环中使用两者。

## 特殊的 else 子句

我们在 Python 语言中看到的一个独特功能是在循环之后有一个`else`子句。它很少被使用，但很有用。如果循环正常结束，因为迭代器（`for`循环）耗尽或因为条件最终没有满足（`while`循环），那么（如果存在）`else`子句将被执行。如果执行被`break`语句中断，则不会执行`else`子句。

让我们以一个`for`循环为例，它遍历一组项目，寻找符合某些条件的一个。如果我们找不到至少一个满足条件的，我们希望抛出一个**异常**。这意味着我们希望阻止程序的正常执行，并发出错误或异常的信号。异常将是*第七章*，*异常和上下文管理器*的主题，所以如果你现在不完全理解它们，不要担心。只需记住，它们会改变代码的正常流程。

让我们先看看在没有`for...else`语法的情况下会如何做。假设我们想在人群中发现一个能够开车的人：

```py
# for.no.else.py
class DriverException(Exception):
    pass
people = [("James", 17), ("Kirk", 9), ("Lars", 13), ("Robert", 8)]
driver = None
for person, age in people:
    if age >= 18:
        driver = (person, age)
        break
if driver is None:
    raise DriverException("Driver not found.") 
```

再次注意*标志*模式。我们将`driver`设置为`None`，然后如果我们找到一个人，我们更新`driver`标志。在循环结束时，我们检查它以查看是否找到了一个人。注意，如果没有找到驾驶员，将抛出`DriverException`，向程序发出信号，表示无法继续执行（我们缺少驾驶员）。

现在，让我们看看如何在`for`循环中使用`else`子句来完成这个操作：

```py
# for.else.py
class DriverException(Exception):
    pass
people = [("James", 17), ("Kirk", 9), ("Lars", 13), ("Robert", 8)]
for person, age in people:
    if age >= 18:
        driver = (person, age)
        break
else:
    raise DriverException("Driver not found.") 
```

注意，我们不再需要*标志*模式。异常作为循环逻辑的一部分被抛出，这很有意义，因为循环会检查某些条件。我们唯一需要做的是设置一个`driver`对象，以防我们找到它；这样，其余的代码就可以使用`driver`对象进行进一步处理。注意，代码变得更短、更优雅，因为逻辑现在被正确地组合在一起，放在了合适的位置。

在他的*将代码转换为优美、惯用 Python*视频中，Raymond Hettinger 建议为与`for`循环关联的`else`语句起一个更好的名字：`nobreak`。如果你在记住`for`循环的`else`是如何工作的方面有困难，只需记住这个事实应该就能帮助你。

# 赋值表达式

在我们查看一些更复杂的例子之前，我们想简要介绍一下 Python 3.8 中添加的一个功能，该功能通过 PEP 572（[`peps.python.org/pep-0572`](https://peps.python.org/pep-0572)）实现。赋值表达式允许我们在不允许正常赋值语句的地方将值绑定到名称上。而不是正常的赋值运算符`=`，赋值表达式使用`:=`（被称为**海象运算符**，因为它与海象的眼睛和獠牙相似）。

## 语句和表达式

要理解正常赋值和赋值表达式之间的区别，我们需要理解语句和表达式之间的区别。根据 Python 文档（[`docs.python.org/3.12/glossary.html#term-statement`](https://docs.python.org/3.12/glossary.html#term-statement)），一个 **语句** 是：

> …是代码块（一个“代码块”）的一部分。一个语句要么是一个表达式，要么是具有关键字的一些构造之一，例如 `if`、`while` 或 `for`。

另一方面，一个 **表达式** 是：

> 一个可以评估为某个值的语法。换句话说，一个表达式是像字面量、名字、属性访问、运算符或函数调用这样的表达式元素的累积，所有这些都会返回一个值。

表达式的关键区分特征是它有一个值。注意，一个表达式可以是一个语句，但并不是所有的语句都是表达式。特别是，像 `name = "heinrich"` 这样的赋值不是表达式，因此它们没有值。这意味着你不能在 `while` 循环或 `if` 语句的条件表达式中（或任何需要值的地方）使用赋值语句。

这就是为什么当你在 Python 控制台中给一个名字赋值时，它不会打印值的原因。例如：

```py
>>> name = "heinrich"
>>> 
```

是一个语句，它没有返回值以打印。

## 使用 walrus 运算符

如果你想将一个值绑定到一个名字并使用该值在一个表达式中，没有赋值表达式，你就必须使用两个独立的语句。例如，我们经常看到这样的代码：

```py
# walrus.if.py
remainder = value % modulus
if remainder:
    print(f"Not divisible! The remainder is {remainder}.") 
```

使用赋值表达式，我们可以将这段代码重写为：

```py
# walrus.if.py
if remainder := value % modulus:
    print(f"Not divisible! The remainder is {remainder}.") 
```

赋值表达式允许我们编写更少的代码行。谨慎使用，它们还可以使代码更简洁、更易于理解。让我们看一个稍微大一点的例子，看看赋值表达式如何简化 `while` 循环。

在交互式脚本中，我们经常需要让用户在多个选项之间进行选择。例如，假设我们正在编写一个交互式脚本，允许冰淇淋店的顾客选择他们想要的口味。为了避免在准备订单时产生混淆，我们希望确保用户选择了一个可用的口味。如果没有赋值表达式，我们可能会写出类似这样的代码：

```py
# menu.no.walrus.py
flavors = ["pistachio", "malaga", "vanilla", "chocolate"]
prompt = "Choose your flavor: "
print(flavors)
while **True**:
    choice = input(prompt)
    if **choice** **in** **flavors**:
        **break**
    print(f"Sorry, '{choice}' is not a valid option.")
print(f"You chose '{choice}'.") 
```

请花一点时间仔细阅读这段代码。注意循环的条件：`while True` 表示“无限循环”，这并不是我们想要的。我们希望在用户输入一个有效的口味（`choice in flavors`）时停止循环。为了实现这一点，我们在循环中有一个 `if` 语句和一个 `break`。控制循环的逻辑并不立即明显。尽管如此，当需要控制循环的值只能在循环内部获得时，这实际上是一种相当常见的模式。

`input()` 函数在交互式脚本中非常有用。它提示用户输入并返回一个字符串。

我们如何改进这一点？让我们尝试使用赋值表达式：

```py
# menu.walrus.py
flavors = ["pistachio", "malaga", "vanilla", "chocolate"]
prompt = "Choose your flavor: "
print(flavors)
while **(choice :=** **input****(prompt))** not in flavors:
    print(f"Sorry, '{choice}' is not a valid option.")
print(f"You chose '{choice}'.") 
```

现在，循环条件正好是我们想要的。这要容易理解得多。代码也短了三行。

在这个例子中，我们需要在赋值表达式周围加上括号，因为 `:=` 运算符的优先级低于 `not in` 运算符。试着去掉它们，看看会发生什么。

我们已经看到了在`if`和`while`语句中使用赋值表达式的例子。除了这些用例之外，赋值表达式在*lambda 表达式*（你将在*第四章*，*函数，代码的构建块*中遇到）以及*推导式*和*生成器*（你将在*第五章*，*推导式和生成器*中学习）中也非常有用。

## 一个警告

Python 中引入 walrus 运算符有些有争议。有些人担心这会让编写丑陋的非 Pythonic 代码变得过于容易。我们认为这些担忧并不完全合理。正如你上面看到的，walrus 运算符可以*改进*代码并使其更容易阅读。然而，像任何强大的功能一样，它也可能被滥用来编写*晦涩难懂*的代码。我们建议你谨慎使用。始终仔细思考它对你的代码可读性的影响。

# 把所有这些放在一起

现在我们已经涵盖了条件语句和循环的基础，我们可以继续到本章开头承诺的示例程序。我们将混合使用，这样你就可以看到如何将这些概念一起使用。

## 一个素数生成器

让我们先写一些代码来生成一个包含素数的列表，直到（包括）某个限制。请记住，我们将编写一个非常低效和原始的算法来寻找素数。重要的是要专注于代码中属于本章主题的部分。

根据 Wolfram MathWorld：

> **素数**（或素整数，通常简称为“**素数**”）是一个大于 1 的正整数 p，它除了 1 和它本身之外没有其他正整数除数。更简洁地说，素数 p 是一个只有一个正除数（除了 1）的正整数，这意味着它是一个不能分解的数。

根据这个定义，如果我们考虑前 10 个自然数，我们可以看到 2、3、5 和 7 是素数，而 1、4、6、8、9 和 10 不是。要确定一个数，*N*，是否为素数，你可以将它除以范围*[2, N)*内的每一个自然数。如果任何除法的余数为零，则该数不是素数。

要生成素数的序列，我们将考虑从 2 开始的自然数，直到限制，并测试它是否是素数。我们将编写两个版本，第二个版本将利用`for...else`语法：

```py
# primes.py
primes = []  # this will contain the primes at the end
upto = 100  # the limit, inclusive
for n in range(2, upto + 1):
    is_prime = True  # flag, new at each iteration of outer for
    for divisor in range(2, n):
        if n % divisor == 0:
            is_prime = False
            break
    if is_prime:  # check on flag
        primes.append(n)
print(primes) 
```

这段代码中发生了很多事情。我们首先设置一个空的`primes`列表，它将包含最后的素数。我们将限制设置为`100`，因为我们希望它是包含的，所以我们必须在最外层`for`循环中遍历`range(2, upto + 1)`（记住`range(2, upto)`*将停止在`upto - 1`*）。最外层循环遍历候选素数——即从`2`到`upto`的所有自然数。这个循环的每次迭代都会测试一个数字，以确定它是否是素数。在最外层循环的每次迭代中，我们设置一个标志（每次迭代都设置为`True`），然后开始将当前值`n`除以从`2`到`n - 1`的所有数字。如果我们找到`n`的一个合适的除数，这意味着`n`是合数，因此我们将标志设置为`False`并退出循环。请注意，当我们退出内层循环时，外层循环会像往常一样继续进行。我们在找到`n`的合适除数后退出是因为我们不需要任何进一步的信息就能判断出`n`不是素数。

在内层循环之后检查`is_prime`标志，如果它仍然是`True`，这意味着我们在*[2, n)*范围内没有找到任何是`n`的合适除数的数字；因此，`n`是素数。我们将`n`添加到`primes`列表中，并继续下一次迭代，直到`n`等于`100`。

运行此代码会输出：

```py
$ python primes.py
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
67, 71, 73, 79, 83, 89, 97] 
```

在继续之前，我们将提出以下问题：外层循环的某次迭代与其他迭代不同。你能指出这是哪一次迭代——为什么？花一点时间思考一下，回到代码中，尝试自己解决它，然后再继续阅读。

你找到答案了吗？如果你没有找到，请不要感到难过，仅通过观察代码就能理解其功能的技能需要时间和经验来学习。尽管如此，这对于程序员来说是一个重要的技能，所以尽量在可能的情况下练习它。现在我们将告诉你答案：第一次迭代与其他所有迭代的行为不同。原因是第一次迭代中，`n`是`2`。因此，最内层的`for`循环根本不会运行，因为它是一个遍历`range(2, 2)`的`for`循环，这是一个空的范围。自己试一试，写一个简单的`for`循环，使用那个可迭代对象，在循环体中放一个`print`语句，看看运行时会发生什么。

我们不会尝试从算法的角度使此代码更高效。但让我们利用本章所学的一些知识，至少让它更容易阅读：

```py
# primes.else.py
primes = []
upto = 100
for n in range(2, upto + 1):
    for divisor in range(2, n):
        if n % divisor == 0:
            break
    else:
        primes.append(n)
print(primes) 
```

在内层循环中使用`else`子句，我们可以去除`is_prime`标志。相反，当我们知道内层循环没有遇到任何`break`语句时，我们将`n`添加到`primes`列表中。这仅仅减少了两个代码行，但代码更简单、更干净，也更易于阅读。这在编程中非常重要，因为简洁性和可读性非常重要。始终寻找简化代码并使其更容易阅读的方法。当你几个月后再次回到它时，你将感谢自己当时所做的努力，以便试图理解你之前做了什么。

## 应用折扣

在这个例子中，我们想向您展示一种称为**查找表**的技术，我们非常喜欢。我们将从简单地编写一些代码开始，根据客户的优惠券价值为它们分配折扣。我们将尽量简化逻辑——记住，我们真正关心的是理解条件语句和循环：

```py
# coupons.py
customers = [
    dict(id=1, total=200, coupon_code="F20"),  # F20: fixed, £20
    dict(id=2, total=150, coupon_code="P30"),  # P30: percent, 30%
    dict(id=3, total=100, coupon_code="P50"),  # P50: percent, 50%
    dict(id=4, total=110, coupon_code="F15"),  # F15: fixed, £15
]
for customer in customers:
    match customer["coupon_code"]:
        case "F20":
            customer["discount"] = 20.0
        case "F15":
            customer["discount"] = 15.0
        case "P30":
            customer["discount"] = customer["total"] * 0.3
        case "P50":
            customer["discount"] = customer["total"] * 0.5
        case _:
            customer["discount"] = 0.0
for customer in customers:
    print(customer["id"], customer["total"], customer["discount"]) 
```

我们首先设置一些客户。他们有一个订单总额、一个优惠券代码和一个 ID。我们编造了四种类型的优惠券：两种是固定金额的，两种是基于百分比的。我们使用一个`match`语句，每个优惠券代码都有一个`case`，以及一个通配符来处理无效的优惠券。我们计算折扣，并将其设置为`customer`字典中的`"discount"`键。

最后，我们只打印出部分数据，以查看我们的代码是否正常工作：

```py
$ python coupons.py
1 200 20.0
2 150 45.0
3 100 50.0
4 110 15.0 
```

这段代码易于理解，但所有这些`match`情况都在逻辑上造成了混乱。添加更多优惠券代码需要添加额外的案例，并为每个案例实现折扣计算。在大多数情况下，折扣计算非常相似，这使得代码重复，违反了**不要重复自己**（**DRY**）原则。在这种情况下，你可以利用字典的优势，如下所示：

```py
# coupons.dict.py
customers = [
    dict(id=1, total=200, coupon_code="F20"),  # F20: fixed, £20
    dict(id=2, total=150, coupon_code="P30"),  # P30: percent, 30%
    dict(id=3, total=100, coupon_code="P50"),  # P50: percent, 50%
    dict(id=4, total=110, coupon_code="F15"),  # F15: fixed, £15
]
discounts = {
    "F20": (0.0, 20.0),  # each value is (percent, fixed)
    "P30": (0.3, 0.0),
    "P50": (0.5, 0.0),
    "F15": (0.0, 15.0),
}
for customer in customers:
    code = customer["coupon_code"]
    percent, fixed = discounts.get(code, (0.0, 0.0))
    customer["discount"] = percent * customer["total"] + fixed
for customer in customers:
    print(customer["id"], customer["total"], customer["discount"]) 
```

运行前面的代码会产生与之前代码片段完全相同的输出。代码减少了两个代码行，但更重要的是，我们在可读性方面取得了很大的进步，因为`for`循环的主体现在只有三行长，易于理解。这里的关键思想是使用字典作为**查找表**。换句话说，我们尝试根据代码（我们的`coupon_code`）从字典中获取一些东西（折扣计算的参数）。我们使用`dict.get(key, default)`来确保我们可以处理不在字典中的代码，并提供一个默认值。

除了可读性之外，这种方法的另一个主要优点是，我们可以轻松地添加新的优惠券代码（或删除旧的代码），而无需更改实现；我们只需要更改查找表中的`*数据*`。在实际应用中，我们甚至可以将查找表存储在数据库中，并为用户提供一个界面，以便在运行时添加或删除优惠券代码。

注意，我们不得不应用一些简单的线性代数来计算折扣。每个折扣在字典中都有一个百分比和固定部分，由一个二元组表示。通过应用 `percent * total + fixed`，我们得到正确的折扣。当 `percent` 为 `0` 时，公式仅给出固定金额；当 `fixed` 为 `0` 时，它给出 `percent * total`。

这种技术与 **调度表** 非常相关，调度表将函数作为表中的值存储。这提供了更大的灵活性。一些面向对象的编程语言在内部使用这种技术来实现诸如虚拟方法等特性。

如果你仍然不清楚这是如何工作的，我们建议你花些时间亲自实验。更改值并添加 `print()` 语句，以查看程序运行时的具体情况。

# 快速浏览 itertools 模块

一章关于可迭代对象、迭代器、条件逻辑和循环的内容，如果没有几句话关于 `itertools` 模块，就不会完整。根据 Python 官方文档（[`docs.python.org/3.12/library/itertools.html`](https://docs.python.org/3.12/library/itertools.html)），`itertools` 模块：

> …实现了许多由 APL、Haskell 和 SML 构造启发的迭代器构建块。每个构建块都已被重新塑形，以适应 Python。
> 
> 该模块标准化了一组快速、内存高效的工具，这些工具本身或组合使用都很有用。它们共同构成了一种“迭代器代数”，使得在纯 Python 中简洁且高效地构建专用工具成为可能。

我们没有足够的空间向您展示这个模块所能提供的一切，所以我们鼓励您自己进一步探索。然而，我们可以保证您会喜欢它。它为您提供了三种广泛的迭代器类别。作为介绍，我们将给出每个类别中一个迭代器的小示例。

## 无限迭代器

无限迭代器允许你使用 `for` 循环作为无限循环，遍历一个永远不会结束的序列：

```py
# infinite.py
from itertools import count
for n in count(5, 3):
    if n > 20:
        break
    print(n, end=", ") # instead of newline, comma and space 
```

运行代码输出：

```py
$ python infinite.py
5, 8, 11, 14, 17, 20, 
```

`count` 工厂类创建一个简单的迭代器，它不断地计数。在这个例子中，它从 `5` 开始，每次迭代都增加 `3`。如果我们不想陷入无限循环，我们需要手动停止它。

## 输入序列最短时终止的迭代器

这个类别非常有趣。它允许您基于多个迭代器创建一个迭代器，根据某些逻辑组合它们的值。关键点在于，如果其中一个输入迭代器比其他迭代器短，结果迭代器不会中断。它会在最短的迭代器耗尽时停止。这听起来可能相当抽象，所以让我们用一个 `compress()` 的例子来说明。这个迭代器接受一个 *data* 序列和一个 *selectors* 序列，只产生与 selectors 序列中的 `True` 值相对应的数据序列中的值。例如，`compress("ABC", (1, 0, 1))` 会返回 `"A"` 和 `"C"`，因为它们对应于 `1`。让我们看看一个简单的例子：

```py
# compress.py
from itertools import compress
data = range(10)
even_selector = [1, 0] * 10
odd_selector = [0, 1] * 10
even_numbers = list(compress(data, even_selector))
odd_numbers = list(compress(data, odd_selector))
print(odd_selector)
print(list(data))
print(even_numbers)
print(odd_numbers) 
```

注意到 `odd_selector` 和 `even_selector` 的长度都是 20 个元素，而 `data` 只有 10 个。`compress()` 函数会在 `data` 产生最后一个元素时停止。运行此代码会产生以下结果：

```py
$ python compress.py
[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[0, 2, 4, 6, 8]
[1, 3, 5, 7, 9] 
```

这是一个快速方便地从可迭代对象中选择元素的方法。代码很简单，但请注意，我们不是使用 `for` 循环来遍历 `compress()` 调用返回的每个值，而是使用了 `list()`，它做的是同样的事情，但它不是执行一系列指令，而是将所有值放入一个列表中并返回它。

## 组合生成器

`itertools` 的第三类迭代器是组合生成器。让我们看看排列的一个简单例子。根据 Wolfram MathWorld：

> 排列，也称为“排列数”或“顺序”，是将有序列表 S 的元素重新排列，使其与 S 本身形成一一对应关系。

例如，`ABC` 有六种排列：`ABC`、`ACB`、`BAC`、`BCA`、`CAB` 和 `CBA`。

如果一个集合有 *N* 个元素，那么这些元素的排列数是 *N!*（*N* 的阶乘）。例如，字符串 `ABC` 有 *3! = 3 * 2 * 1 = 6* 种排列。让我们用 Python 来看看这个例子：

```py
# permutations.py
from itertools import permutations
print(list(permutations("ABC"))) 
```

这段简短的代码会产生以下结果：

```py
$ python permutations.py
[('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'),
('C', 'A', 'B'), ('C', 'B', 'A')] 
```

在玩排列时请小心。它们的数量以与您正在排列的元素数量的阶乘成比例的速度增长，而且这个数字可以变得非常大，非常快。

有一个名为 `more-itertools` 的第三方库扩展了 `itertools` 模块。您可以在 [`more-itertools.readthedocs.io/`](https://more-itertools.readthedocs.io/) 找到它的文档。

# 摘要

在本章中，我们朝着扩展我们的 Python 词汇表又迈出了另一步。我们看到了如何通过评估条件来驱动代码的执行，以及如何循环遍历序列和对象集合。这赋予了我们控制代码运行时发生什么的能力，这意味着我们得到了如何塑造它以实现我们想要的功能，并使其能够对动态变化的数据做出反应的想法。

我们也看到了如何在几个简单的例子中将所有内容结合起来，最后，我们简要地浏览了 `itertools` 模块，它充满了有趣的迭代器，可以让我们用 Python 的能力得到更大的丰富。

现在，是时候转换方向，再迈出一步，来谈谈函数。下一章全部都是关于它们的，它们非常重要。确保你对到目前为止的内容感到舒适。我们想给你提供一些有趣的例子，让我们开始吧。

# 加入我们的 Discord 社区

加入我们的 Discord 空间，与作者和其他读者进行讨论：

`discord.com/invite/uaKmaz7FEC`

![img](img/QR_Code119001106417026468.png)
