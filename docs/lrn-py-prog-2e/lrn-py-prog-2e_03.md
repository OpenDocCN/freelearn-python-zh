# 第三章：迭代和做决定

“疯狂：一遍又一遍地做同样的事情，却期待不同的结果。”- 阿尔伯特·爱因斯坦

在上一章中，我们看了 Python 的内置数据类型。现在你已经熟悉了数据的各种形式和形状，是时候开始看看程序如何使用它了。

根据维基百科：

在计算机科学中，控制流（或者叫控制流程）是指规定命令式程序的各个语句、指令或函数调用的执行或评估顺序。

为了控制程序的流程，我们有两个主要的工具：**条件编程**（也称为**分支**）和**循环**。我们可以以许多不同的组合和变体使用它们，但在本章中，我不打算以*文档*的方式介绍这两个结构的所有可能形式，而是给你一些基础知识，然后和你一起编写一些小脚本。在第一个脚本中，我们将看到如何创建一个简单的素数生成器，而在第二个脚本中，我们将看到如何根据优惠券给顾客打折。这样，你应该更好地了解条件编程和循环如何使用。

在本章中，我们将涵盖以下内容：

+   条件编程

+   Python 中的循环

+   快速浏览`itertools`模块

# 条件编程

条件编程，或者分支，是你每天、每时每刻都在做的事情。它涉及评估条件：*如果交通灯是绿色的，那么我可以过去；* *如果下雨，那么我会带伞；* *如果我上班迟到了，那么我会打电话给我的经理*。

主要工具是`if`语句，它有不同的形式和颜色，但基本上它评估一个表达式，并根据结果选择要执行的代码部分。像往常一样，让我们看一个例子：

```py
# conditional.1.py
late = True 
if late: 
    print('I need to call my manager!') 
```

这可能是最简单的例子：当传递给`if`语句时，`late`充当条件表达式，在布尔上下文中进行评估（就像我们调用`bool(late)`一样）。如果评估的结果是`True`，那么我们就进入`if`语句后面的代码体。请注意，`print`指令是缩进的：这意味着它属于由`if`子句定义的作用域。执行这段代码会产生：

```py
$ python conditional.1.py
I need to call my manager!
```

由于`late`是`True`，`print`语句被执行了。让我们扩展一下这个例子：

```py
# conditional.2.py
late = False 
if late: 
    print('I need to call my manager!')  #1 
else: 
    print('no need to call my manager...')  #2 
```

这次我设置了`late = False`，所以当我执行代码时，结果是不同的：

```py
$ python conditional.2.py
no need to call my manager...
```

根据评估`late`表达式的结果，我们可以进入块`#1`或块`#2`，*但不能同时进入*。当`late`评估为`True`时，执行块`#1`，而当`late`评估为`False`时，执行块`#2`。尝试给`late`名称分配`False`/`True`值，并看看这段代码的输出如何相应地改变。

前面的例子还介绍了`else`子句，当我们想要在`if`子句中的表达式评估为`False`时提供一组备用指令时，它非常方便。`else`子句是可选的，通过比较前面的两个例子可以明显看出。

# 一个专门的 else - elif

有时，您只需要在满足条件时执行某些操作（简单的`if`子句）。在其他时候，您需要提供一个替代方案，以防条件为`False`（`if`/`else`子句），但有时您可能有更多的选择路径，因此，由于调用经理（或不调用他们）是一种二进制类型的示例（要么您打电话，要么您不打电话），让我们改变示例的类型并继续扩展。这次，我们决定税收百分比。如果我的收入低于$10,000，我将不支付任何税。如果在$10,000 和$30,000 之间，我将支付 20%的税。如果在$30,000 和$100,000 之间，我将支付 35%的税，如果超过$100,000，我将（很高兴）支付 45%的税。让我们把这一切都写成漂亮的 Python 代码：

```py
# taxes.py
income = 15000 
if income < 10000: 
    tax_coefficient = 0.0  #1 
elif income < 30000: 
    tax_coefficient = 0.2  #2 
elif income < 100000: 
    tax_coefficient = 0.35  #3 
else: 
    tax_coefficient = 0.45  #4 

print('I will pay:', income * tax_coefficient, 'in taxes') 
```

执行上述代码产生的结果：

```py
$ python taxes.py
I will pay: 3000.0 in taxes
```

让我们逐行通过这个例子：我们首先设置收入值。在这个例子中，我的收入是$15,000。我们进入`if`子句。请注意，这一次我们还引入了`elif`子句，它是`else-if`的缩写，与裸的`else`子句不同，它也有自己的条件。因此，`income < 10000`的`if`表达式评估为`False`，因此块`#1`不被执行。

控制传递给下一个条件评估器：`elif income < 30000`。这个评估为`True`，因此块`#2`被执行，因此，Python 在整个`if`/`elif`/`elif`/`else`子句之后恢复执行（我们现在可以称之为`if`子句）。在`if`子句之后只有一条指令，即`print`调用，它告诉我们今年我将支付`3000.0`的税（*15,000 * 20%*）。请注意，顺序是强制的：`if`首先出现，然后（可选）是尽可能多的`elif`子句，然后（可选）是一个`else`子句。

有趣，对吧？无论每个块内有多少行代码，当其中一个条件评估为`True`时，相关的块将被执行，然后在整个子句之后恢复执行。如果没有一个条件评估为`True`（例如，`income = 200000`），那么`else`子句的主体将被执行（块`#4`）。这个例子扩展了我们对`else`子句行为的理解。当之前的`if`/`elif`/.../`elif`表达式没有评估为`True`时，它的代码块被执行。

尝试修改`income`的值，直到您可以随意执行所有块（每次执行一个）。然后尝试**边界**。这是至关重要的，每当您将条件表达为**相等**或**不等**（`==`，`!=`，`<`，`>`，`<=`，`>=`）时，这些数字代表边界。彻底测试边界是至关重要的。我是否允许您在 18 岁或 17 岁时开车？我是用`age < 18`还是`age <= 18`来检查您的年龄？您无法想象有多少次我不得不修复由于使用错误的运算符而产生的微妙错误，因此继续并尝试修改上述代码。将一些`<`更改为`<=`，并将收入设置为边界值之一（10,000，30,000，100,000）以及之间的任何值。看看结果如何变化，并在继续之前对其有一个很好的理解。

现在让我们看另一个示例，向我们展示如何嵌套`if`子句。假设您的程序遇到错误。如果警报系统是控制台，我们打印错误。如果警报系统是电子邮件，我们根据错误的严重程度发送它。如果警报系统不是控制台或电子邮件之外的任何其他东西，我们不知道该怎么办，因此我们什么也不做。让我们把这写成代码：

```py
# errorsalert.py
alert_system = 'console'  # other value can be 'email' 
error_severity = 'critical'  # other values: 'medium' or 'low' 
error_message = 'OMG! Something terrible happened!' 

if alert_system == 'console': 
    print(error_message)  #1 
elif alert_system == 'email': 
    if error_severity == 'critical': 
        send_email('admin@example.com', error_message)  #2 
    elif error_severity == 'medium': 
        send_email('support.1@example.com', error_message)  #3 
    else: 
        send_email('support.2@example.com', error_message)  #4 
```

上面的例子非常有趣，因为它很愚蠢。它向我们展示了两个嵌套的`if`子句（**外部**和**内部**）。它还向我们展示了外部`if`子句没有任何`else`，而内部`if`子句有。请注意，缩进是允许我们将一个子句嵌套在另一个子句中的原因。

如果`alert_system == 'console'`，则执行`#1`部分，其他情况则不执行。另一方面，如果`alert_system == 'email'`，那么我们进入另一个`if`子句，我们称之为内部。在内部`if`子句中，根据`error_severity`，我们向管理员、一级支持或二级支持发送电子邮件（块`#2`，`#3`和`#4`）。在此示例中未定义`send_email`函数，因此尝试运行它会导致错误。在本书的源代码中，您可以从网站下载，我包含了一个技巧，将该调用重定向到常规的`print`函数，这样您就可以在控制台上进行实验，而不必实际发送电子邮件。尝试更改值，看看它是如何工作的。

# 三元运算符

在转移到下一个主题之前，我想向您展示的最后一件事是**三元运算符**，或者通俗地说，是`if`/`else`子句的简短版本。当根据某个条件来分配名称的值时，有时使用三元运算符而不是适当的`if`子句更容易阅读。在下面的示例中，两个代码块完全做同样的事情：

```py
# ternary.py
order_total = 247  # GBP 

# classic if/else form 
if order_total > 100: 
    discount = 25  # GBP 
else: 
    discount = 0  # GBP 
print(order_total, discount) 

# ternary operator 
discount = 25 if order_total > 100 else 0 
print(order_total, discount) 
```

对于这样简单的情况，我发现能够用一行代码来表达这种逻辑非常好，而不是用四行。记住，作为编码人员，您花在阅读代码上的时间远远多于编写代码的时间，因此 Python 的简洁性是无价的。

您清楚三元运算符是如何工作的吗？基本上，`name = something if condition else something-else`。因此，如果`condition`评估为`True`，则`name`被分配为`something`，如果`condition`评估为`False`，则为`something-else`。

现在您已经了解了如何控制代码的路径，让我们继续下一个主题：*循环*。

# 循环

如果您在其他编程语言中有循环的经验，您会发现 Python 的循环方式有些不同。首先，什么是循环？**循环**意味着能够根据给定的循环参数多次重复执行代码块。有不同的循环结构，它们有不同的目的，Python 已将它们全部简化为只有两种，您可以使用它们来实现您需要的一切。这些是`for`和`while`语句。

虽然使用它们中的任何一个都可以做你需要做的一切，但它们有不同的目的，因此它们通常在不同的上下文中使用。我们将在本章中深入探讨这种差异。

# `for`循环

当循环遍历序列时，例如列表、元组或对象集合时，使用`for`循环。让我们从一个简单的示例开始，扩展概念，看看 Python 语法允许我们做什么：

```py
# simple.for.py
for number in [0, 1, 2, 3, 4]: 
    print(number) 
```

这段简单的代码片段在执行时打印从`0`到`4`的所有数字。`for`循环接收到列表`[0, 1, 2, 3, 4]`，在每次迭代时，`number`从序列中获得一个值（按顺序迭代），然后执行循环体（打印行）。`number`的值在每次迭代时都会更改，根据序列中接下来的值。当序列耗尽时，`for`循环终止，代码的执行在循环后恢复正常。

# 遍历范围

有时我们需要遍历一系列数字，将其硬编码到某个地方将会很不方便。在这种情况下，`range`函数就派上用场了。让我们看看前面代码片段的等价物：

```py
# simple.for.py
for number in range(5): 
    print(number) 
```

在 Python 程序中，`range`函数在创建序列时被广泛使用：您可以通过传递一个值来调用它，该值充当`stop`（从`0`开始计数），或者您可以传递两个值（`start`和`stop`），甚至三个值（`start`，`stop`和`step`）。查看以下示例：

```py
>>> list(range(10))  # one value: from 0 to value (excluded)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> list(range(3, 8))  # two values: from start to stop (excluded)
[3, 4, 5, 6, 7]
>>> list(range(-10, 10, 4))  # three values: step is added
[-10, -6, -2, 2, 6]
```

暂时忽略我们需要在`range(...)`内部包装一个`list`。`range`对象有点特殊，但在这种情况下，我们只对了解它将向我们返回什么值感兴趣。您可以看到，切片的处理方式与之相同：`start`包括在内，`stop`排除在外，还可以添加一个`step`参数，其默认值为`1`。

尝试修改我们的`simple.for.py`代码中`range()`调用的参数，并查看它打印出什么。熟悉它。

# 在序列上进行迭代

现在我们有了迭代序列的所有工具，所以让我们在此基础上构建示例：

```py
# simple.for.2.py
surnames = ['Rivest', 'Shamir', 'Adleman'] 
for position in range(len(surnames)): 
    print(position, surnames[position]) 
```

前面的代码给游戏增加了一点复杂性。执行将显示此结果：

```py
$ python simple.for.2.py
0 Rivest
1 Shamir
2 Adleman
```

让我们使用**从内到外**的技术来分解它，好吗？我们从我们试图理解的最内部部分开始，然后向外扩展。因此，`len(surnames)`是`surnames`列表的长度：`3`。因此，`range(len(surnames))`实际上被转换为`range(3)`。这给我们提供了范围[0, 3)，基本上是一个序列（`0`，`1`，`2`）。这意味着`for`循环将运行三次迭代。在第一次迭代中，`position`将取值`0`，而在第二次迭代中，它将取值`1`，最后在第三次和最后一次迭代中取值`2`。如果不是`surnames`列表的可能索引位置（`0`，`1`，`2`），那是什么？在位置`0`，我们找到`'Rivest'`，在位置`1`，`'Shamir'`，在位置`2`，`'Adleman'`。如果您对这三个人一起创造了什么感到好奇，请将`print(position, surnames[position])`更改为`print(surnames[position][0], end='')`，在循环之外添加最后一个`print()`，然后再次运行代码。

现在，这种循环的风格实际上更接近于 Java 或 C++等语言。在 Python 中，很少见到这样的代码。您可以只是迭代任何序列或集合，因此没有必要在每次迭代时获取位置列表并从序列中检索元素。这是昂贵的，没有必要的昂贵。让我们将示例改为更符合 Python 风格的形式：

```py
# simple.for.3.py
surnames = ['Rivest', 'Shamir', 'Adleman'] 
for surname in surnames: 
    print(surname) 
```

现在这就是了！它几乎是英语。`for`循环可以在`surnames`列表上进行迭代，并且在每次交互中按顺序返回每个元素。运行此代码将逐个打印出三个姓氏。阅读起来更容易，对吧？

但是，如果您想要打印位置呢？或者如果您实际上需要它呢？您应该回到`range(len(...))`形式吗？不。您可以使用`enumerate`内置函数，就像这样：

```py
# simple.for.4.py
surnames = ['Rivest', 'Shamir', 'Adleman'] 
for position, surname in enumerate(surnames): 
    print(position, surname) 
```

这段代码也非常有趣。请注意，`enumerate`在每次迭代时都会返回一个两元组（`position，surname`），但仍然比`range(len(...))`示例更可读（更有效）。您可以使用`start`参数调用`enumerate`，例如`enumerate(iterable, start)`，它将从`start`开始，而不是`0`。这只是另一件小事，向您展示了 Python 在设计时考虑了多少，以便使您的生活更轻松。

您可以使用`for`循环来迭代列表、元组和一般 Python 称为可迭代的任何东西。这是一个非常重要的概念，所以让我们再谈一谈。

# 迭代器和可迭代对象

根据 Python 文档（[`docs.python.org/3/glossary.html`](https://docs.python.org/3/glossary.html)）的说法，可迭代对象是：

能够逐个返回其成员的对象。可迭代对象的示例包括所有序列类型（如列表、str 和元组）和一些非序列类型，如字典、文件对象和您使用 __iter__()或 __getitem__()方法定义的任何类的对象。可迭代对象可以在 for 循环和许多其他需要序列的地方使用（zip()、map()等）。当将可迭代对象作为参数传递给内置函数 iter()时，它会返回该对象的迭代器。该迭代器对值集合进行一次遍历。在使用可迭代对象时，通常不需要调用 iter()或自己处理迭代器对象。for 语句会自动为您执行这些操作，为循环的持续时间创建一个临时的无名变量来保存迭代器。

简而言之，当你写`for k in sequence: ... body ...`时，`for`循环会向`sequence`请求下一个元素，它会得到一些返回值，将返回值称为`k`，然后执行其主体。然后，再次，`for`循环会向`sequence`请求下一个元素，再次将其称为`k`，并再次执行主体，依此类推，直到序列耗尽。空序列将导致主体执行零次。

一些数据结构在进行迭代时按顺序产生它们的元素，例如列表、元组和字符串，而另一些则不会，例如集合和字典（Python 3.6 之前）。Python 让我们能够迭代可迭代对象，使用一种称为**迭代器**的对象类型。

根据官方文档（[`docs.python.org/3/glossary.html`](https://docs.python.org/3/glossary.html)）的说法，迭代器是：

代表数据流的对象。对迭代器的 __next__()方法进行重复调用（或将其传递给内置函数 next()）会返回数据流中的连续项目。当没有更多数据可用时，会引发 StopIteration 异常。此时，迭代器对象已耗尽，对其 __next__()方法的任何进一步调用都会再次引发 StopIteration。迭代器需要具有一个返回迭代器对象本身的 __iter__()方法，因此每个迭代器也是可迭代的，并且可以在大多数其他可接受可迭代对象的地方使用。一个值得注意的例外是尝试多次迭代传递的代码。容器对象（如列表）每次将其传递给 iter()函数或在 for 循环中使用时都会产生一个全新的迭代器。尝试对迭代器执行此操作将只返回上一次迭代传递中使用的相同耗尽的迭代器对象，使其看起来像一个空容器。

如果你不完全理解前面的法律术语，不要担心，你以后会理解的。我把它放在这里作为将来的方便参考。

实际上，整个可迭代/迭代器机制在代码后面有些隐藏。除非出于某种原因需要编写自己的可迭代或迭代器，否则你不必过多担心这个问题。但是理解 Python 如何处理这一关键的控制流方面非常重要，因为它将塑造你编写代码的方式。

# 遍历多个序列

让我们看另一个例子，如何迭代两个相同长度的序列，以便处理它们各自的元素对。假设我们有一个人员列表和一个代表第一个列表中人员年龄的数字列表。我们想要打印所有人员的姓名/年龄对。让我们从一个例子开始，然后逐渐完善它：

```py
# multiple.sequences.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
for position in range(len(people)):
    person = people[position]
    age = ages[position]
    print(person, age)
```

到目前为止，这段代码对你来说应该很容易理解。我们需要遍历位置列表（`0`，`1`，`2`，`3`），因为我们想要从两个不同的列表中检索元素。执行后，我们得到以下结果：

```py
$ python multiple.sequences.py
Conrad 29
Deepak 30
Heinrich 34
Tom 36
```

这段代码既低效又不符合 Python 风格。它是低效的，因为根据位置检索元素可能是一个昂贵的操作，并且我们在每次迭代时都是从头开始做的。邮递员在递送信件时不会每次都回到路的起点，对吧？他们从一户到另一户。让我们尝试使用`enumerate`使其更好：

```py
# multiple.sequences.enumerate.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
for position, person in enumerate(people):
    age = ages[position]
    print(person, age)
```

这样做更好，但还不完美。而且还有点丑陋。我们在`people`上进行了适当的迭代，但仍然使用位置索引获取`age`，我们也想要摆脱。别担心，Python 给了你`zip`函数，记得吗？让我们使用它：

```py
# multiple.sequences.zip.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
for person, age in zip(people, ages):
    print(person, age)
```

啊！好多了！再次将前面的代码与第一个示例进行比较，并欣赏 Python 的优雅之处。我想展示这个例子的原因有两个。一方面，我想让您了解 Python 中较短的代码与其他语言相比有多么简洁，其他语言的语法不允许您像这样轻松地迭代序列或集合。另一方面，更重要的是，请注意，当`for`循环请求`zip(sequenceA, sequenceB)`的下一个元素时，它会得到一个元组，而不仅仅是一个单一对象。它会得到一个元组，其中包含与我们提供给`zip`函数的序列数量一样多的元素。让我们通过两种方式扩展前面的示例，使用显式和隐式赋值：

```py
# multiple.sequences.explicit.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
nationalities = ['Poland', 'India', 'South Africa', 'England']
for person, age, nationality in zip(people, ages, nationalities):
    print(person, age, nationality)
```

在前面的代码中，我们添加了国籍列表。现在我们向`zip`函数提供了三个序列，for 循环在每次迭代时都会返回一个*三元组*。请注意，元组中元素的位置与`zip`调用中序列的位置相对应。执行代码将产生以下结果：

```py
$ python multiple.sequences.explicit.py
Conrad 29 Poland
Deepak 30 India
Heinrich 34 South Africa
Tom 36 England
```

有时，由于在前面的简单示例中可能不太清楚的原因，您可能希望在`for`循环的主体中分解元组。如果这是您的愿望，完全可以做到：

```py
# multiple.sequences.implicit.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
nationalities = ['Poland', 'India', 'South Africa', 'England']
for data in zip(people, ages, nationalities):
    person, age, nationality = data
    print(person, age, nationality)
```

它基本上是在某些情况下自动为您执行`for`循环的操作，但是在某些情况下，您可能希望自己执行。在这里，来自`zip(...)`的三元组`data`在`for`循环的主体中被分解为三个变量：`person`、`age`和`nationality`。

# while 循环

在前面的页面中，我们看到了`for`循环的运行情况。当您需要循环遍历一个序列或集合时，它非常有用。需要记住的关键点是，当您需要能够区分使用哪种循环结构时，`for`循环在必须迭代有限数量的元素时非常有效。它可以是一个巨大的数量，但是仍然是在某个点结束的东西。

然而，还有其他情况，当您只需要循环直到满足某个条件，甚至无限循环直到应用程序停止时，例如我们实际上没有东西可以迭代，因此`for`循环将是一个不好的选择。但是不用担心，对于这些情况，Python 为我们提供了`while`循环。

`while`循环类似于`for`循环，因为它们都循环，并且在每次迭代时执行一组指令。它们之间的不同之处在于`while`循环不会循环遍历一个序列（它可以，但您必须手动编写逻辑，而且这没有任何意义，您只想使用`for`循环），而是只要满足某个条件就会循环。当条件不再满足时，循环结束。

像往常一样，让我们看一个示例，以便更好地理解。我们想要打印一个正数的二进制表示。为了做到这一点，我们可以使用一个简单的算法，它收集除以`2`的余数（以相反的顺序），结果就是数字本身的二进制表示：

```py
6 / 2 = 3 (remainder: 0) 
3 / 2 = 1 (remainder: 1) 
1 / 2 = 0 (remainder: 1) 
List of remainders: 0, 1, 1\. 
Inverse is 1, 1, 0, which is also the binary representation of 6: 110
```

让我们编写一些代码来计算数字 39 的二进制表示：100111[2]：

```py
# binary.py
n = 39
remainders = []
while n > 0:
    remainder = n % 2  # remainder of division by 2
    remainders.insert(0, remainder)  # we keep track of remainders
    n //= 2  # we divide n by 2

print(remainders)
```

在前面的代码中，我突出显示了`n > 0`，这是保持循环的条件。我们可以通过使用`divmod`函数使代码变得更短（更符合 Python 风格），该函数使用一个数字和一个除数调用，并返回一个包含整数除法结果及其余数的元组。例如，`divmod(13, 5)`将返回`(2, 3)`，确实*5 * 2 + 3 = 13*：

```py
# binary.2.py
n = 39
remainders = []
while n > 0:
    n, remainder = divmod(n, 2)
    remainders.insert(0, remainder)

print(remainders)
```

在前面的代码中，我们已经将`n`重新分配为除以`2`的结果，并在一行中得到了余数。

请注意，在`while`循环中的条件是继续循环的条件。如果评估为`True`，则执行主体，然后进行另一个评估，依此类推，直到条件评估为`False`。当发生这种情况时，循环立即退出，而不执行其主体。

如果条件永远不会评估为`False`，则循环变成所谓的**无限循环**。无限循环用于例如从网络设备轮询：您询问套接字是否有任何数据，如果有任何数据，则对其进行某些操作，然后您休眠一小段时间，然后再次询问套接字，一遍又一遍，永远不停止。

拥有循环条件或无限循环的能力，这就是为什么仅使用`for`循环是不够的原因，因此 Python 提供了`while`循环。

顺便说一句，如果您需要数字的二进制表示，请查看`bin`函数。

只是为了好玩，让我们使用`while`逻辑来调整一个例子（`multiple.sequences.py`）：

```py
# multiple.sequences.while.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
position = 0
while position < len(people):
    person = people[position]
    age = ages[position]
    print(person, age)
    position += 1
```

在前面的代码中，我突出显示了`position`变量的*初始化*、*条件*和*更新*，这使得可以通过手动处理迭代变量来模拟等效的`for`循环代码。所有可以使用`for`循环完成的工作也可以使用`while`循环完成，尽管您可以看到为了实现相同的结果，您需要经历一些样板文件。反之亦然，但除非您有理由这样做，否则您应该使用正确的工具来完成工作，99.9%的时间您都会没问题。

因此，总结一下，当您需要遍历可迭代对象时，请使用`for`循环，当您需要根据满足或不满足的条件循环时，请使用`while`循环。如果您记住了两种目的之间的区别，您将永远不会选择错误的循环结构。

现在让我们看看如何改变循环的正常流程。

# 中断和继续语句

根据手头的任务，有时您需要改变循环的常规流程。您可以跳过单个迭代（任意次数），或者完全退出循环。跳过迭代的常见用例是，例如，当您遍历项目列表并且只有在验证了某些条件时才需要处理每个项目时。另一方面，如果您正在遍历项目集，并且找到了满足您某些需求的项目，您可能决定不继续整个循环，因此退出循环。有无数种可能的情况，因此最好看一些例子。

假设您想对购物篮列表中所有今天到期的产品应用 20%的折扣。您实现这一点的方式是使用`continue`语句，它告诉循环结构（`for`或`while`）立即停止执行主体并继续下一个迭代（如果有的话）。这个例子将带我们深入了解，所以准备好跳下去：

```py
# discount.py
from datetime import date, timedelta

today = date.today()
tomorrow = today + timedelta(days=1)  # today + 1 day is tomorrow
products = [
    {'sku': '1', 'expiration_date': today, 'price': 100.0},
    {'sku': '2', 'expiration_date': tomorrow, 'price': 50},
    {'sku': '3', 'expiration_date': today, 'price': 20},
]

for product in products:
    if product['expiration_date'] != today:
        continue
    product['price'] *= 0.8  # equivalent to applying 20% discount
    print(
        'Price for sku', product['sku'],
        'is now', product['price'])
```

我们首先导入`date`和`timedelta`对象，然后设置我们的产品。`sku`为`1`和`3`的产品具有“今天”的到期日期，这意味着我们希望对它们应用 20%的折扣。我们遍历每个产品并检查到期日期。如果它不是（不等运算符，`!=`）“今天”，我们不希望执行其余的主体套件，因此我们`continue`。

请注意，在代码块中放置`continue`语句的位置并不重要（甚至可以使用多次）。当到达它时，执行停止并返回到下一次迭代。如果我们运行`discount.py`模块，这是输出：

```py
$ python discount.py
Price for sku 1 is now 80.0
Price for sku 3 is now 16.0
```

这向你展示了循环体的最后两行没有被执行，对于`sku`编号`2`。

现在让我们看一个中断循环的例子。假设我们想要判断列表中的至少一个元素在传递给`bool`函数时是否评估为`True`。鉴于我们需要知道是否至少有一个，当我们找到它时，就不需要继续扫描列表。在 Python 代码中，这意味着使用`break`语句。让我们把这写成代码：

```py
# any.py
items = [0, None, 0.0, True, 0, 7]  # True and 7 evaluate to True

found = False  # this is called "flag"
for item in items:
    print('scanning item', item)
    if item:
        found = True  # we update the flag
        break

if found:  # we inspect the flag
    print('At least one item evaluates to True')
else:
    print('All items evaluate to False')
```

前面的代码在编程中是一个常见的模式，你会经常看到它。当你以这种方式检查项目时，基本上你是设置一个`flag`变量，然后开始检查。如果你找到一个符合你标准的元素（在这个例子中，评估为`True`），然后你更新标志并停止迭代。迭代后，你检查标志并相应地采取行动。执行结果是：

```py
$ python any.py
scanning item 0
scanning item None
scanning item 0.0
scanning item True
At least one item evaluates to True
```

看到了吗？在找到`True`后执行停止了吗？`break`语句的作用与`continue`相同，即立即停止循环体的执行，但也阻止其他迭代运行，有效地跳出循环。`continue`和`break`语句可以在`for`和`while`循环结构中一起使用，数量上没有限制。

顺便说一句，没有必要编写代码来检测序列中是否至少有一个元素评估为`True`。只需查看内置的`any`函数。

# 特殊的 else 子句

我在 Python 语言中看到的一个特性是在`while`和`for`循环后面有`else`子句的能力。它很少被使用，但绝对是一个不错的功能。简而言之，你可以在`for`或`while`循环后面有一个`else`代码块。如果循环正常结束，因为迭代器耗尽（`for`循环）或者因为条件最终不满足（`while`循环），那么`else`代码块（如果存在）会被执行。如果执行被`break`语句中断，`else`子句就不会被执行。让我们来看一个`for`循环的例子，它遍历一组项目，寻找一个满足某些条件的项目。如果我们找不到至少一个满足条件的项目，我们想要引发一个**异常**。这意味着我们想要中止程序的正常执行，并且表示出现了一个错误或异常，我们无法处理。异常将在第八章中讨论，*测试、分析和处理异常*，所以如果你现在不完全理解它们，不用担心。只要记住它们会改变代码的正常流程。

现在让我向你展示两个做同样事情的例子，但其中一个使用了特殊的`for...else`语法。假设我们想在一群人中找到一个能开车的人：

```py
# for.no.else.py
class DriverException(Exception):
    pass

people = [('James', 17), ('Kirk', 9), ('Lars', 13), ('Robert', 8)]
driver = None
for person, age in people:
    if age >= 18:
        driver = (person, age)
        break

if driver is None:
    raise DriverException('Driver not found.')
```

再次注意`flag`模式。我们将驾驶员设置为`None`，然后如果我们找到一个，我们会更新`driver`标志，然后在循环结束时检查它是否找到了。我有一种感觉，那些孩子可能会开一辆非常*金属感*的车，但无论如何，请注意，如果找不到驾驶员，将会引发`DriverException`，向程序表示执行无法继续（我们缺少驾驶员）。

相同的功能可以使用以下代码更加优雅地重写：

```py
# for.else.py
class DriverException(Exception):
    pass

people = [('James', 17), ('Kirk', 9), ('Lars', 13), ('Robert', 8)]
for person, age in people:
    if age >= 18:
        driver = (person, age)
        break
else:
    raise DriverException('Driver not found.')
```

请注意，我们不再被迫使用`flag`模式。异常是作为`for`循环逻辑的一部分引发的，这是合理的，因为`for`循环正在检查某些条件。我们只需要在找到一个时设置一个`driver`对象，因为代码的其余部分将在某个地方使用该信息。请注意，代码更短、更优雅，因为逻辑现在正确地组合在一起。

在*将代码转换为优美、成语化的 Python*视频中，Raymond Hettinger 建议为与 for 循环关联的`else`语句取一个更好的名字：`nobreak`。如果你在记住`else`在`for`循环中的工作原理方面有困难，只需记住这个事实就应该能帮助你。

# 把所有这些放在一起

现在你已经看到了关于条件和循环的所有内容，是时候稍微调剂一下，看看我在本章开头预期的那两个例子。我们将在这里混合搭配，这样你就可以看到如何将所有这些概念结合起来使用。让我们先写一些代码来生成一个质数列表，直到某个限制为止。请记住，我将写一个非常低效和基本的算法来检测质数。对你来说重要的是集中精力关注代码中属于本章主题的部分。

# 质数生成器

根据维基百科：

质数（或质数）是大于 1 的自然数，除了 1 和它本身之外没有其他正因子。大于 1 的自然数如果不是质数，则称为合数。

根据这个定义，如果我们考虑前 10 个自然数，我们可以看到 2、3、5 和 7 是质数，而 1、4、6、8、9 和 10 不是。为了让计算机告诉你一个数*N*是否是质数，你可以将该数除以范围[2，*N*)内的所有自然数。如果其中任何一个除法的余数为零，那么这个数就不是质数。废话够多了，让我们开始吧。我将写两个版本，第二个版本将利用`for...else`语法：

```py
# primes.py
primes = []  # this will contain the primes in the end
upto = 100  # the limit, inclusive
for n in range(2, upto + 1):
    is_prime = True  # flag, new at each iteration of outer for
    for divisor in range(2, n):
        if n % divisor == 0:
            is_prime = False
            break
```

```py
    if is_prime:  # check on flag
        primes.append(n)
print(primes)
```

在前面的代码中有很多需要注意的事情。首先，我们设置了一个空的`primes`列表，它将在最后包含质数。限制是`100`，你可以看到我们在外部循环中调用`range()`的方式是包容的。如果我们写`range(2, upto)`，那么是*[2, upto)*，对吧？因此`range(2, upto + 1)`给我们*[2, upto + 1) == [2, upto]*。

因此，有两个`for`循环。在外部循环中，我们循环遍历候选质数，即从`2`到`upto`的所有自然数。在外部循环的每次迭代中，我们设置一个标志（在每次迭代时设置为`True`），然后开始将当前的`n`除以从`2`到`n-1`的所有数字。如果我们找到`n`的一个适当的除数，那么意味着`n`是合数，因此我们将标志设置为`False`并中断循环。请注意，当我们中断内部循环时，外部循环会继续正常进行。我们之所以在找到`n`的适当除数后中断，是因为我们不需要任何进一步的信息就能判断`n`不是质数。

当我们检查`is_prime`标志时，如果它仍然是`True`，这意味着我们在[2，*n*)中找不到任何是`n`的适当除数的数字，因此`n`是质数。我们将`n`添加到`primes`列表中，然后继续下一个迭代，直到`n`等于`100`。

运行这段代码会产生：

```py
$ python primes.py
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97] 
```

在我们继续之前，有一个问题：在外部循环的所有迭代中，其中一个与其他所有迭代不同。你能告诉哪一个，以及为什么吗？想一想，回到代码，试着自己找出答案，然后继续阅读。

你弄清楚了吗？如果没有，不要感到难过，这是完全正常的。我让你做这个小练习，因为这是程序员一直在做的事情。通过简单地查看代码来理解代码的功能是您随着时间建立的技能。这非常重要，所以尽量在您能做到的时候进行练习。我现在告诉你答案：与所有其他迭代不同的是第一个迭代。原因是因为在第一次迭代中，`n`是`2`。因此，最内层的`for`循环甚至不会运行，因为它是一个迭代`range(2, 2)`的`for`循环，那不就是[2, 2)吗？自己试一下，用这个可迭代对象编写一个简单的`for`循环，将`print`放在主体套件中，看看是否会发生任何事情（不会...）。

现在，从算法的角度来看，这段代码是低效的，所以让我们至少让它更美观一些：

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

漂亮多了，对吧？`is_prime`标志消失了，当我们知道内部`for`循环没有遇到任何`break`语句时，我们将`n`附加到`primes`列表中。看看代码看起来更清晰，阅读起来更好了吗？

# 应用折扣

在这个例子中，我想向你展示一种我非常喜欢的技术。在许多编程语言中，除了`if`/`elif`/`else`结构之外，无论以什么形式或语法，你都可以找到另一个语句，通常称为`switch`/`case`，在 Python 中缺少。它相当于一系列`if`/`elif`/.../`elif`/`else`子句，其语法类似于这样（警告！JavaScript 代码！）：

```py
/* switch.js */
switch (day_number) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
        day = "Weekday";
        break;
    case 6:
        day = "Saturday";
        break;
    case 0:
        day = "Sunday";
        break;
    default:
        day = "";
```

```py
        alert(day_number + ' is not a valid day number.')
}
```

在前面的代码中，我们根据名为`day_number`的变量进行`switch`。这意味着我们获取它的值，然后决定它适用于哪种情况（如果有的话）。从`1`到`5`有一个级联，这意味着无论数字如何，[`1`，`5`]都会进入将`day`设置为“工作日”的逻辑部分。然后我们有`0`和`6`的单个情况，以及一个`default`情况来防止错误，它会提醒系统`day_number`不是有效的日期数字，即不在[`0`，`6`]中。Python 完全能够使用`if`/`elif`/`else`语句实现这样的逻辑：

```py
# switch.py
if 1 <= day_number <= 5:
    day = 'Weekday'
elif day_number == 6:
    day = 'Saturday'
elif day_number == 0:
    day = 'Sunday'
else:
    day = ''
    raise ValueError(
        str(day_number) + ' is not a valid day number.')
```

在前面的代码中，我们使用`if`/`elif`/`else`语句在 Python 中复制了 JavaScript 片段的相同逻辑。我最后提出了`ValueError`异常，如果`day_number`不在[`0`，`6`]中，这只是一个例子。这是将`switch`/`case`逻辑转换的一种可能方式，但还有另一种方式，有时称为分派，我将在下一个示例的最后版本中向您展示。

顺便说一下，你有没有注意到前面片段的第一行？你有没有注意到 Python 可以进行双重（实际上甚至多重）比较？这太棒了！

让我们通过简单地编写一些代码来开始新的示例，根据客户的优惠券价值为他们分配折扣。我会尽量保持逻辑的最低限度，记住我们真正关心的是理解条件和循环：

```py
# coupons.py
customers = [
    dict(id=1, total=200, coupon_code='F20'),  # F20: fixed, £20
    dict(id=2, total=150, coupon_code='P30'),  # P30: percent, 30%
    dict(id=3, total=100, coupon_code='P50'),  # P50: percent, 50%
    dict(id=4, total=110, coupon_code='F15'),  # F15: fixed, £15
]
for customer in customers:
    code = customer['coupon_code']
    if code == 'F20':
        customer['discount'] = 20.0
    elif code == 'F15':
        customer['discount'] = 15.0
    elif code == 'P30':
        customer['discount'] = customer['total'] * 0.3
    elif code == 'P50':
        customer['discount'] = customer['total'] * 0.5
    else:
        customer['discount'] = 0.0

for customer in customers:
    print(customer['id'], customer['total'], customer['discount'])
```

我们首先设置一些客户。他们有订单总额、优惠券代码和 ID。我编造了四种不同类型的优惠券，两种是固定的，两种是基于百分比的。你可以看到，在`if`/`elif`/`else`级联中，我相应地应用折扣，并将其设置为`customer`字典中的`'discount'`键。

最后，我只打印出部分数据，看看我的代码是否正常工作：

```py
$ python coupons.py
1 200 20.0
2 150 45.0
3 100 50.0
4 110 15.0
```

这段代码很容易理解，但所有这些子句有点混乱。一眼看上去很难看出发生了什么，我不喜欢。在这种情况下，你可以利用字典来发挥你的优势，就像这样：

```py
# coupons.dict.py
customers = [
    dict(id=1, total=200, coupon_code='F20'),  # F20: fixed, £20
    dict(id=2, total=150, coupon_code='P30'),  # P30: percent, 30%
    dict(id=3, total=100, coupon_code='P50'),  # P50: percent, 50%
    dict(id=4, total=110, coupon_code='F15'),  # F15: fixed, £15
]
discounts = {
    'F20': (0.0, 20.0),  # each value is (percent, fixed)
    'P30': (0.3, 0.0),
    'P50': (0.5, 0.0),
    'F15': (0.0, 15.0),
}
for customer in customers:
    code = customer['coupon_code']
    percent, fixed = discounts.get(code, (0.0, 0.0))
    customer['discount'] = percent * customer['total'] + fixed

for customer in customers:
    print(customer['id'], customer['total'], customer['discount'])
```

运行前面的代码产生了与之前片段相同的结果。我们节省了两行，但更重要的是，我们在可读性上获得了很多好处，因为`for`循环的主体现在只有三行，而且非常容易理解。这里的概念是将字典用作**分发器**。换句话说，我们尝试从字典中根据代码（我们的`coupon_code`）获取一些东西，并通过`dict.get(key, default)`，我们确保当`code`不在字典中时，我们也需要一个默认值。

请注意，我必须应用一些非常简单的线性代数来正确计算折扣。字典中的每个折扣都有一个百分比和固定部分，由一个二元组表示。通过应用`percent * total + fixed`，我们得到正确的折扣。当`percent`为`0`时，该公式只给出固定金额，当固定为`0`时，它给出`percent * total`。

这种技术很重要，因为它也用在其他情境中，比如函数，它实际上比我们在前面片段中看到的要强大得多。使用它的另一个优势是，你可以以这样的方式编码，使得`discounts`字典的键和值可以动态获取（例如，从数据库中获取）。这将使代码能够适应你所拥有的任何折扣和条件，而无需修改任何内容。

如果你不完全明白它是如何工作的，我建议你花点时间来试验一下。更改值并添加打印语句，看看程序运行时发生了什么。

# 快速浏览 itertools 模块

关于可迭代对象、迭代器、条件逻辑和循环的章节，如果没有提到`itertools`模块，就不完整了。如果你喜欢迭代，这是一种天堂。

根据 Python 官方文档（[`docs.python.org/2/library/itertools.html`](https://docs.python.org/2/library/itertools.html)），`itertools`模块是：

这个模块实现了一些受 APL、Haskell 和 SML 构造启发的迭代器构建块。每个都已经被重塑成适合 Python 的形式。该模块标准化了一组核心的快速、内存高效的工具，这些工具本身或组合在一起都很有用。它们一起形成了一个“迭代器代数”，使得可以在纯 Python 中简洁高效地构建专门的工具。

在这里我无法向你展示这个模块中所有的好东西，所以我鼓励你自己去查看，我保证你会喜欢的。简而言之，它为您提供了三种广泛的迭代器类别。我将给你一个非常小的例子，来自每一个迭代器，只是为了让你稍微流口水。

# 无限迭代器

无限迭代器允许您以不同的方式使用`for`循环，就像它是一个`while`循环一样：

```py
# infinite.py
from itertools import count

for n in count(5, 3):
    if n > 20:
        break
    print(n, end=', ') # instead of newline, comma and space
```

运行代码会得到这个结果：

```py
$ python infinite.py
5, 8, 11, 14, 17, 20,
```

`count`工厂类创建一个迭代器，它只是不断地计数。它从`5`开始，然后不断加`3`。如果我们不想陷入无限循环，我们需要手动中断它。

# 在最短输入序列上终止的迭代器

这个类别非常有趣。它允许您基于多个迭代器创建一个迭代器，根据某种逻辑组合它们的值。关键点在于，在这些迭代器中，如果有任何一个比其余的短，那么生成的迭代器不会中断，它将在最短的迭代器耗尽时停止。这非常理论化，我知道，所以让我用`compress`给你举个例子。这个迭代器根据选择器中的相应项目是`True`还是`False`，给你返回数据：

`compress('ABC', (1, 0, 1))`会返回`'A'`和`'C'`，因为它们对应于`1`。让我们看一个简单的例子：

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

请注意，`odd_selector`和`even_selector`的长度为 20 个元素，而`data`只有 10 个元素。`compress`将在`data`产生最后一个元素时停止。运行此代码会产生以下结果：

```py
$ python compress.py
[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[0, 2, 4, 6, 8]
[1, 3, 5, 7, 9]
```

这是一种非常快速和方便的从可迭代对象中选择元素的方法。代码非常简单，只需注意，我们使用`list()`而不是使用`for`循环来迭代压缩调用返回的每个值，`list()`做的事情是一样的，但是它不执行一系列指令，而是将所有的值放入一个列表并返回它。

# 组合生成器

最后但并非最不重要的，组合生成器。如果你对这种事情感兴趣，这些真的很有趣。让我们看一个关于排列的简单例子。

根据 Wolfram Mathworld：

排列，也称为“排列数”或“顺序”，是有序列表 S 的元素重新排列成与 S 本身一一对应的过程。

例如，ABC 有六种排列：ABC，ACB，BAC，BCA，CAB 和 CBA。

如果一个集合有*N*个元素，那么它们的排列数就是*N!*（*N*的阶乘）。对于 ABC 字符串，排列数为*3! = 3 * 2 * 1 = 6*。让我们用 Python 来做一下：

```py
# permutations.py
from itertools import permutations 
print(list(permutations('ABC'))) 
```

这段非常简短的代码产生了以下结果：

```py
$ python permutations.py
[('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]
```

当你玩排列时要非常小心。它们的数量增长速度与你进行排列的元素的阶乘成正比，而这个数字可能会变得非常大，非常快。

# 总结

在本章中，我们又迈出了一步，扩展了我们的编码词汇。我们已经看到如何通过评估条件来驱动代码的执行，以及如何循环和迭代序列和对象集合。这赋予了我们控制代码运行时发生的事情的能力，这意味着我们正在了解如何塑造代码，使其按照我们的意愿进行操作，并对动态变化的数据做出反应。

我们还看到了如何在几个简单的例子中将所有东西结合在一起，最后，我们简要地看了一下`itertools`模块，这个模块充满了有趣的迭代器，可以进一步丰富我们使用 Python 的能力。

现在是时候换个方式，向前迈进一步，谈谈函数。下一章将全面讨论它们，因为它们非常重要。确保你对到目前为止所涵盖的内容感到舒适。我想为你提供有趣的例子，所以我将不得不加快速度。准备好了吗？翻页吧。
