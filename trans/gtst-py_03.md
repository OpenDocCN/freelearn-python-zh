# 迭代和做决定

“疯狂就是一遍又一遍地做同样的事情，却期待不同的结果。”- 阿尔伯特·爱因斯坦

在上一章中，我们看过了 Python 的内置数据类型。现在你已经熟悉了数据的各种形式和类型，是时候开始看看程序如何使用它了。

根据维基百科：

在计算机科学中，控制流（或者另一种说法是控制流程）指的是规定命令式程序的各个语句、指令或函数调用的执行或评估顺序。

为了控制程序的流程，我们有两个主要的武器：**条件编程**（也称为**分支**）和**循环**。我们可以以许多不同的组合和变化来使用它们，但在本章中，我不想以*文档*的方式介绍这两个结构的所有可能形式，我宁愿先给你一些基础知识，然后和你一起编写一些小脚本。在第一个脚本中，我们将看到如何创建一个基本的质数生成器，而在第二个脚本中，我们将看到如何根据优惠券为客户提供折扣。这样，你应该更好地了解条件编程和循环如何被使用。

在本章中，我们将涵盖以下内容：

+   条件编程

+   Python 中的循环

+   快速浏览 itertools 模块

# 条件编程

条件编程，或者分支，是你每天、每时每刻都在做的事情。它涉及评估条件：*如果交通灯是绿色的，那么我可以过马路；* *如果下雨了，那么我就带伞；* *如果我上班迟到了，那么我会打电话给我的经理*。

主要工具是`if`语句，它有不同的形式和颜色，但基本上它评估一个表达式，并根据结果选择要执行的代码部分。像往常一样，让我们看一个例子：

```py
# conditional.1.py
late = True 
if late: 
    print('I need to call my manager!') 
```

这可能是最简单的例子：当`late`被传递给`if`语句时，`late`充当条件表达式，在布尔上下文中进行评估（就像我们调用`bool(late)`一样）。如果评估的结果是`True`，那么我们就进入`if`语句后面的代码体。请注意，`print`指令是缩进的：这意味着它属于由`if`子句定义的作用域。执行这段代码会产生：

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

这次我将`late = False`，所以当我执行代码时，结果是不同的：

```py
$ python conditional.2.py
no need to call my manager...
```

根据评估`late`表达式的结果，我们可以进入块`#1`或块`#2`，*但不能同时进入*。当`late`评估为`True`时，执行块`#1`，而当`late`评估为`False`时，执行块`#2`。尝试为`late`名称分配`False`/`True`值，并看看这段代码的输出如何相应地改变。

前面的例子还介绍了`else`子句，当我们想要在`if`子句中的表达式求值为`False`时提供一组替代指令时，它就非常方便。`else`子句是可选的，通过比较前面的两个例子就可以看出来。

# 一个特殊的 else - elif

有时，您只需要在满足条件时执行某些操作（简单的`if`子句）。在其他时候，您需要提供一个替代方案，以防条件为`False`（`if`/`else`子句），但有时候您可能有多于两条路径可供选择，因此，由于调用经理（或不调用他们）是一种二进制类型的示例（要么您打电话，要么您不打电话），让我们改变示例的类型并继续扩展。这次，我们决定税收百分比。如果我的收入低于$10,000，我将不支付任何税款。如果在$10,000 和$30,000 之间，我将支付 20%的税款。如果在$30,000 和$100,000 之间，我将支付 35%的税款，如果超过$100,000，我将（很高兴）支付 45%的税款。让我们把这一切都写成漂亮的 Python 代码：

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

执行上述代码产生：

```py
$ python taxes.py
I will pay: 3000.0 in taxes
```

让我们逐行来看这个例子：我们首先设置收入值。在这个例子中，我的收入是$15,000。我们进入`if`子句。请注意，这次我们还引入了`elif`子句，它是`else-if`的缩写，与裸的`else`子句不同，它还有自己的条件。因此，`income < 10000`的`if`表达式评估为`False`，因此块`#1`不会被执行。

控制权转移到下一个条件评估器：`elif income < 30000`。这个评估为`True`，因此块`#2`被执行，因此，Python 在整个`if`/`elif`/`elif`/`else`子句之后恢复执行（我们现在可以称之为`if`子句）。`if`子句之后只有一条指令，即`print`调用，它告诉我们我今年将支付`3000.0`的税款（*15,000 * 20%*）。请注意，顺序是强制性的：`if`首先出现，然后（可选）是您需要的尽可能多的`elif`子句，然后（可选）是一个`else`子句。

有趣，对吧？无论每个块内有多少行代码，只要其中一个条件评估为`True`，相关块就会被执行，然后在整个子句之后执行。如果没有一个条件评估为`True`（例如，`income = 200000`），那么`else`子句的主体将被执行（块`#4`）。这个例子扩展了我们对`else`子句行为的理解。当前面的`if`/`elif`/.../`elif`表达式没有评估为`True`时，它的代码块将被执行。

尝试修改`income`的值，直到您可以轻松地按需执行所有块（每次执行一个块，当然）。然后尝试**边界**。这是至关重要的，每当您将条件表达为**相等**或**不等式**（`==`，`!=`，`<`，`>`，`<=`，`>=`）时，这些数字代表边界。彻底测试边界是至关重要的。我是否允许您在 18 岁或 17 岁时开车？我是否用`age < 18`或`age <= 18`检查您的年龄？您无法想象有多少次我不得不修复由于使用错误运算符而产生的微妙错误，因此继续并尝试使用上述代码进行实验。将一些`<`更改为`<=`，并将收入设置为边界值之一（10,000，30,000，100,000）以及之间的任何值。看看结果如何变化，并在继续之前对其有一个很好的理解。

现在让我们看另一个例子，它向我们展示了如何嵌套`if`子句。假设您的程序遇到错误。如果警报系统是控制台，我们打印错误。如果警报系统是电子邮件，我们根据错误的严重程度发送它。如果警报系统不是控制台或电子邮件，我们不知道该怎么办，因此我们什么也不做。让我们把这写成代码：

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

如果`alert_system == 'console'`，则执行`#1`部分，其他情况下，如果`alert_system == 'email'`，则进入另一个`if`子句，我们称之为内部。在内部`if`子句中，根据`error_severity`，我们向管理员、一级支持或二级支持发送电子邮件（块`#2`、`#3`和`#4`）。在本例中，`send_email`函数未定义，因此尝试运行它会导致错误。在书的源代码中，您可以从网站上下载，我包含了一个技巧，将该调用重定向到常规的`print`函数，这样您就可以在控制台上进行实验，而不必实际发送电子邮件。尝试更改值，看看它是如何工作的。

# 三元运算符

在继续下一个主题之前，我想向您展示的最后一件事是**三元运算符**，或者通俗地说，`if`/`else`子句的简短版本。当根据某些条件分配名称的值时，有时使用三元运算符而不是适当的`if`子句更容易和更可读。在以下示例中，两个代码块完全相同：

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

对于这种简单情况，我发现能够用一行代码来表达逻辑非常好，而不是用四行。记住，作为编码人员，您花在阅读代码上的时间要比编写代码多得多，因此 Python 的简洁性是无价的。

您清楚三元运算符的工作原理吗？基本上，`name = something if condition else something-else`。因此，如果`condition`评估为`True`，则将`name`分配为`something`，如果`condition`评估为`False`，则将`something-else`分配给`name`。

现在您已经了解了如何控制代码的路径，让我们继续下一个主题：*循环*。

# 循环

如果您在其他编程语言中有循环的经验，您会发现 Python 的循环方式有些不同。首先，什么是循环？**循环**意味着能够根据给定的循环参数多次重复执行代码块。有不同的循环结构，用于不同的目的，Python 已将它们全部简化为只有两种，您可以使用它们来实现您需要的一切。这些是`for`和`while`语句。

虽然使用任何一种都可以实现您需要的一切，但它们用途不同，因此通常在不同的上下文中使用。我们将在本章中彻底探讨这种差异。

# `for`循环

`for`循环用于循环遍历序列，例如列表、元组或一组对象。让我们从一个简单的例子开始，扩展概念，看看 Python 语法允许我们做什么：

```py
# simple.for.py
for number in [0, 1, 2, 3, 4]: 
    print(number) 
```

当执行时，这段简单的代码打印出从`0`到`4`的所有数字。`for`循环接收列表`[0, 1, 2, 3, 4]`，在每次迭代时，`number`从序列中获得一个值（按顺序迭代），然后执行循环体（打印行）。`number`的值在每次迭代时都会改变，根据序列中接下来的值。当序列耗尽时，`for`循环终止，代码的执行会在循环后恢复正常。

# 迭代范围

有时我们需要迭代一系列数字，如果在某处硬编码列表将会很不方便。在这种情况下，`range`函数就派上用场了。让我们看看前面代码片段的等价物：

```py
# simple.for.py
for number in range(5): 
    print(number)
```

在 Python 程序中，当涉及创建序列时，`range`函数被广泛使用：您可以通过传递一个值来调用它，该值充当`stop`（从`0`开始计数），或者您可以传递两个值（`start`和`stop`），甚至三个值（`start`、`stop`和`step`）。看看以下示例：

```py
>>> list(range(10))  # one value: from 0 to value (excluded)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> list(range(3, 8))  # two values: from start to stop (excluded)
[3, 4, 5, 6, 7]
>>> list(range(-10, 10, 4))  # three values: step is added
[-10, -6, -2, 2, 6]
```

暂时忽略我们需要在`list`中包装`range(...)`的事实。`range`对象有点特殊，但在这种情况下，我们只是想了解它将向我们返回什么值。您可以看到，切片的处理方式也是一样的：`start`包括在内，`stop`不包括在内，还可以添加一个`step`参数，其默认值为`1`。

尝试修改我们`simple.for.py`代码中`range()`调用的参数，并查看打印出什么。熟悉一下。

# 在序列上进行迭代

现在我们有了所有迭代序列的工具，让我们在此基础上构建示例：

```py
# simple.for.2.py
surnames = ['Rivest', 'Shamir', 'Adleman'] 
for position in range(len(surnames)): 
    print(position, surnames[position]) 
```

前面的代码给游戏增加了一些复杂性。执行将显示以下结果：

```py
$ python simple.for.2.py
0 Rivest
1 Shamir
2 Adleman
```

让我们使用**从内到外**的技术来分解它，好吗？我们从我们试图理解的最内部部分开始，然后向外扩展。因此，`len(surnames)`是`surnames`列表的长度：`3`。因此，`range(len(surnames))`实际上被转换为`range(3)`。这给我们一个范围[0, 3)，基本上是一个序列（`0`，`1`，`2`）。这意味着`for`循环将运行三次迭代。在第一次迭代中，`position`将取值`0`，而在第二次迭代中，它将取值`1`，最后在第三次和最后一次迭代中取值`2`。如果不是`（0`，`1`，`2`），那么对`surnames`列表的可能索引位置是什么？在位置`0`，我们找到`'Rivest'`，在位置`1`，`'Shamir'`，在位置`2`，`'Adleman'`。如果您对这三个人一起创造了什么感到好奇，请将`print(position, surnames[position])`更改为`print(surnames[position][0], end='')`，在循环外添加最后一个`print()`，然后再次运行代码。

现在，这种循环方式实际上更接近于 Java 或 C++等语言。在 Python 中，很少看到这样的代码。您可以只是迭代任何序列或集合，因此无需获取位置列表并在每次迭代时从序列中检索元素。这是昂贵的，没有必要的昂贵。让我们将示例更改为更符合 Python 风格的形式：

```py
# simple.for.3.py
surnames = ['Rivest', 'Shamir', 'Adleman'] 
for surname in surnames: 
    print(surname) 
```

现在这就是！它几乎是英语。`for`循环可以迭代`surnames`列表，并且它会在每次交互中按顺序返回每个元素。运行此代码将打印出三个姓氏，一个接一个。阅读起来更容易，对吧？

但是，如果您想要打印位置呢？或者如果您确实需要它呢？您应该回到`range(len(...))`形式吗？不。您可以使用`enumerate`内置函数，就像这样：

```py
# simple.for.4.py
surnames = ['Rivest', 'Shamir', 'Adleman'] 
for position, surname in enumerate(surnames): 
    print(position, surname) 
```

这段代码也很有趣。请注意，`enumerate`在每次迭代时返回一个二元组`(position, surname)`，但仍然比`range(len(...))`示例更可读（更有效）。您可以使用`start`参数调用`enumerate`，例如`enumerate(iterable, start)`，它将从`start`开始，而不是从`0`开始。这只是另一个小事情，表明 Python 在设计时考虑了多少，以便使您的生活更轻松。

您可以使用`for`循环来迭代列表、元组和一般 Python 称为可迭代的任何东西。这是一个非常重要的概念，所以让我们再谈谈它。

# 迭代器和可迭代对象

根据 Python 文档（[`docs.python.org/3/glossary.html`](https://docs.python.org/3/glossary.html)）的说法，可迭代对象是：

一个能够逐个返回其成员的对象。可迭代对象的示例包括所有序列类型（如列表、字符串和元组）和一些非序列类型，比如字典、文件对象和你用 __iter__()或 __getitem__()方法定义的任何类的对象。可迭代对象可以在 for 循环和许多其他需要序列的地方使用（zip()、map()等）。当将可迭代对象作为参数传递给内置函数 iter()时，它会返回该对象的迭代器。这个迭代器对一组值进行一次遍历。在使用可迭代对象时，通常不需要调用 iter()或自己处理迭代器对象。for 语句会自动为你创建一个临时的未命名变量来保存迭代器，以便在循环期间使用。

简而言之，当你写`for k in sequence: ... body ...`时，`for`循环会询问`sequence`下一个元素，得到返回值后，将其命名为`k`，然后执行其主体。然后，`for`循环再次询问`sequence`下一个元素，再次将其命名为`k`，再次执行主体，依此类推，直到序列耗尽。空序列将导致主体不执行。

一些数据结构在迭代时按顺序产生它们的元素，比如列表、元组和字符串，而另一些则不会，比如集合和字典（Python 3.6 之前）。Python 给了我们迭代可迭代对象的能力，使用一种称为**迭代器**的对象类型。

根据官方文档（[`docs.python.org/3/glossary.html`](https://docs.python.org/3/glossary.html)），迭代器是：

表示数据流的对象。对迭代器的 __next__()方法进行重复调用（或将其传递给内置函数 next()）会返回数据流中的连续项。当没有更多数据可用时，会引发 StopIteration 异常。此时，迭代器对象已耗尽，任何进一步调用其 __next__()方法都会再次引发 StopIteration。迭代器需要有一个返回迭代器对象本身的 __iter__()方法，因此每个迭代器也是可迭代的，并且可以在大多数接受其他可迭代对象的地方使用。一个值得注意的例外是尝试多次迭代的代码。容器对象（如列表）每次传递给 iter()函数或在 for 循环中使用时都会产生一个全新的迭代器。尝试对迭代器进行这样的操作只会返回相同的已耗尽的迭代器对象，使其看起来像一个空容器。

如果你不完全理解前面的法律术语，不要担心，你以后会理解的。我把它放在这里作为将来的方便参考。

实际上，整个可迭代/迭代器机制在代码后面有些隐藏。除非出于某种原因需要编写自己的可迭代或迭代器，否则你不必过多担心这个问题。但理解 Python 如何处理这一关键的控制流方面非常重要，因为它将塑造你编写代码的方式。

# 迭代多个序列

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

到目前为止，这段代码应该对你来说非常简单。我们需要迭代位置列表（`0`、`1`、`2`、`3`），因为我们想要从两个不同的列表中检索元素。执行后我们得到以下结果：

```py
$ python multiple.sequences.py
Conrad 29
Deepak 30
Heinrich 34
Tom 36
```

这段代码既低效又不符合 Python 的风格。它是低效的，因为根据位置检索元素可能是一个昂贵的操作，并且我们在每次迭代时都是从头开始做这个操作。邮递员在递送信件时不会每次都回到路的起点，对吧？他们是从一户到另一户。让我们尝试使用`enumerate`来改进一下：

```py
# multiple.sequences.enumerate.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
for position, person in enumerate(people):
    age = ages[position]
    print(person, age)
```

这样好一些，但还不完美。而且还有点丑。我们在`people`上进行了适当的迭代，但我们仍然使用位置索引来获取`age`，我们也想摆脱这一点。别担心，Python 给了你`zip`函数，记得吗？让我们使用它：

```py
# multiple.sequences.zip.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
for person, age in zip(people, ages):
    print(person, age)
```

啊！好多了！再次比较前面的代码和第一个例子，欣赏 Python 的优雅之处。我想展示这个例子的原因有两个。一方面，我想给你一个概念，即 Python 中更短的代码可以与其他语言相比，其中的语法不允许你像这样轻松地迭代序列或集合。另一方面，更重要的是，注意当`for`循环请求`zip(sequenceA, sequenceB)`的下一个元素时，它会得到一个元组，而不仅仅是一个单一对象。它会得到一个元组，其中包含与我们提供给`zip`函数的序列数量相同的元素。让我们通过两种方式扩展前面的例子，使用显式和隐式赋值：

```py
# multiple.sequences.explicit.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
nationalities = ['Poland', 'India', 'South Africa', 'England']
for person, age, nationality in zip(people, ages, nationalities):
    print(person, age, nationality)
```

在前面的代码中，我们添加了 nationalities 列表。现在我们向`zip`函数提供了三个序列，for 循环在每次迭代时都会返回一个*三元组*。请注意，元组中元素的位置与`zip`调用中序列的位置相对应。执行代码将产生以下结果：

```py
$ python multiple.sequences.explicit.py
Conrad 29 Poland
Deepak 30 India
Heinrich 34 South Africa
Tom 36 England
```

有时，出于某些在简单示例中可能不太清楚的原因，你可能希望在`for`循环的主体中分解元组。如果这是你的愿望，完全可以这样做：

```py
# multiple.sequences.implicit.py
people = ['Conrad', 'Deepak', 'Heinrich', 'Tom']
ages = [29, 30, 34, 36]
nationalities = ['Poland', 'India', 'South Africa', 'England']
for data in zip(people, ages, nationalities):
    person, age, nationality = data
    print(person, age, nationality)
```

基本上，它在某些情况下会自动为你做`for`循环所做的事情。但在某些情况下，你可能希望自己做。在这里，来自`zip(...)`的三元组`data`在`for`循环的主体中被分解为三个变量：`person`、`age`和`nationality`。

# while 循环

在前面的页面中，我们看到了`for`循环的运行情况。当你需要循环遍历一个序列或集合时，它非常有用。需要记住的关键一点是，当你需要能够区分使用哪种循环结构时，`for`循环在你需要迭代有限数量的元素时非常有效。它可以是一个巨大的数量，但仍然是在某个点结束的东西。

然而，还有其他情况，当你只需要循环直到满足某个条件，甚至是无限循环直到应用程序停止时，比如我们真的没有东西可以迭代的情况，因此`for`循环会是一个不好的选择。但不用担心，对于这些情况，Python 为我们提供了`while`循环。

`while`循环类似于`for`循环，因为它们都会循环，并且在每次迭代时执行一系列指令。它们之间的不同之处在于`while`循环不会循环遍历一个序列（它可以，但你必须手动编写逻辑，而且没有任何意义，你只想使用`for`循环），而是在某个条件满足时循环。当条件不再满足时，循环结束。

和往常一样，让我们看一个例子，这将为我们澄清一切。我们想要打印一个正数的二进制表示。为了做到这一点，我们可以使用一个简单的算法，它收集除以`2`的余数（逆序），结果就是数字本身的二进制表示：

```py
6 / 2 = 3 (remainder: 0) 
3 / 2 = 1 (remainder: 1) 
1 / 2 = 0 (remainder: 1) 
List of remainders: 0, 1, 1\. 
Inverse is 1, 1, 0, which is also the binary representation of 6: 110
```

让我们写一些代码来计算数字 39 的二进制表示：100111[2]：

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

在上面的代码中，我突出了`n > 0`，这是保持循环的条件。我们可以通过使用`divmod`函数使代码变得更短（更符合 Python 风格），该函数使用一个数字和一个除数调用，并返回一个包含整数除法结果及其余数的元组。例如，`divmod(13, 5)`将返回`(2, 3)`，确实*5 * 2 + 3 = 13*。

```py
# binary.2.py
n = 39
remainders = []
while n > 0:
    n, remainder = divmod(n, 2)
    remainders.insert(0, remainder)

print(remainders)
```

在上面的代码中，我们已经将`n`重新分配为除以`2`的结果和余数，一行代码完成。

请注意，在`while`循环中的条件是继续循环的条件。如果条件评估为`True`，则执行主体，然后进行另一个评估，依此类推，直到条件评估为`False`。当发生这种情况时，循环立即退出，而不执行其主体。

如果条件永远不会评估为`False`，则循环将成为所谓的**无限循环**。无限循环的用途包括从网络设备轮询时使用：您询问套接字是否有任何数据，如果有，则对其进行某些操作，然后您休眠一小段时间，然后再次询问套接字，一遍又一遍，永不停止。

能够循环遍历条件或无限循环是`for`循环单独不足的原因，因此 Python 提供了`while`循环。

顺便说一句，如果您需要数字的二进制表示，请查看`bin`函数。

只是为了好玩，让我们使用 while 逻辑来调整一个示例（`multiple.sequences.py`）：

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

在上面的代码中，我突出了`position`变量的*初始化*、*条件*和*更新*，这使得可以通过手动处理迭代变量来模拟等效的`for`循环代码。所有可以使用`for`循环完成的工作也可以使用`while`循环完成，尽管您可以看到为了实现相同的结果，需要经历一些样板文件。相反的也是如此，但除非您有理由这样做，否则您应该使用正确的工具，99.9%的时间您都会没问题。

因此，总结一下，当您需要遍历可迭代对象时，请使用`for`循环，当您需要根据满足或不满足条件来循环时，请使用`while`循环。如果您记住这两种目的之间的区别，您将永远不会选择错误的循环结构。

现在让我们看看如何改变循环的正常流程。

# 中断和继续语句

根据手头的任务，有时您需要改变循环的正常流程。您可以跳过单个迭代（多次），也可以完全退出循环。跳过迭代的常见用例是，例如，当您遍历一个项目列表并且只有在验证了某些条件时才需要处理每个项目时。另一方面，如果您正在遍历一组项目，并且找到了满足某些需求的项目，您可能决定不再继续整个循环，因此退出循环。有无数可能的情景，因此最好看一些例子。

假设您想要对购物篮列表中所有今天到期的产品应用 20%的折扣。您实现这一点的方式是使用`continue`语句，该语句告诉循环结构（`for`或`while`）立即停止执行主体并转到下一个迭代（如果有的话）。这个例子将让我们深入了解一点，所以准备好跳下去：

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

我们首先导入`date`和`timedelta`对象，然后设置我们的产品。那些`sku`为`1`和`3`的产品具有今天的到期日，这意味着我们希望对它们应用 20%的折扣。我们循环遍历每个产品并检查到期日。如果它不是（不等运算符，`!=`）`today`，我们不希望执行其余的主体套件，因此我们`continue`。

注意，`continue`语句放在主体套件的哪里并不重要（你甚至可以使用它多次）。当你到达它时，执行停止并返回到下一个迭代。如果我们运行`discount.py`模块，输出如下：

```py
$ python discount.py
Price for sku 1 is now 80.0
Price for sku 3 is now 16.0
```

这向你展示了主体的最后两行没有被执行给`sku`编号为`2`。

现在让我们看一个退出循环的例子。假设我们想要判断列表中是否至少有一个元素在传递给`bool`函数时评估为`True`。鉴于我们需要知道是否至少有一个，当我们找到它时，就不需要继续扫描列表。在 Python 代码中，这意味着使用`break`语句。让我们把这写成代码：

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

前面的代码在编程中是一个常见的模式，你会经常看到它。当你以这种方式检查项目时，基本上你是设置一个`flag`变量，然后开始检查。如果你找到一个符合你条件的元素（在这个例子中，评估为`True`），然后你更新`flag`并停止迭代。迭代后，你检查`flag`并相应地采取行动。执行结果是：

```py
$ python any.py
scanning item 0
scanning item None
scanning item 0.0
scanning item True
At least one item evaluates to True
```

看到`True`被找到后执行停止了吗？`break`语句的作用和`continue`一样，它立即停止执行循环主体，但也阻止其他迭代运行，有效地跳出循环。`continue`和`break`语句可以在`for`和`while`循环结构中一起使用，没有数量限制。

顺便说一下，没有必要编写代码来检测序列中是否至少有一个元素评估为`True`。只需要查看内置的`any`函数。

# 特殊的 else 子句

在 Python 语言中我看到的一个特性是在`while`和`for`循环后面能够有`else`子句的能力。这种用法非常少见，但是确实很有用。简而言之，你可以在`for`或`while`循环后面有一个`else`子句。如果循环正常结束，因为迭代器耗尽（`for`循环）或者条件最终不满足（`while`循环），那么`else`子句（如果存在）会被执行。如果执行被`break`语句中断，`else`子句就不会被执行。让我们举一个例子，一个`for`循环遍历一组项目，寻找满足某个条件的项目。如果我们找不到至少一个满足条件的项目，我们想要引发一个**异常**。这意味着我们想要中止程序的正常执行，并且表示出现了一个我们无法处理的错误或异常。异常将在后面的章节中讨论，所以如果你现在不完全理解它们也不用担心。只需要记住它们会改变代码的正常流程。

现在让我展示给你两个做同样事情的例子，但其中一个使用了特殊的`for...else`语法。假设我们想在一组人中找到一个可以开车的人：

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

再次注意`flag`模式。我们将驱动程序设置为`None`，然后如果我们找到一个，我们更新`driver`标志，然后在循环结束时，我们检查它是否找到了一个。我有一种感觉，那些孩子会开一辆非常*金属*的车，但无论如何，注意如果没有找到驱动程序，将会引发`DriverException`，表示程序无法继续执行（我们缺少驱动程序）。

相同的功能可以用以下代码更加优雅地重写：

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

请注意，我们不再被迫使用`flag`模式。异常是作为`for`循环逻辑的一部分引发的，这是有道理的，因为`for`循环正在检查某些条件。我们只需要设置一个`driver`对象，以防我们找到一个，因为代码的其余部分将在某个地方使用这些信息。请注意，代码更短更优雅，因为逻辑现在正确地组合在一起。

在*将代码转换为美观的 Python*视频中，Raymond Hettinger 建议将与 for 循环关联的`else`语句的名称改为`nobreak`。如果你在记住`else`如何用于`for`循环时感到困难，只需记住这个事实就应该帮助你了。

# 把这一切放在一起

现在你已经看到关于条件和循环的所有内容，是时候稍微调剂一下，看看我在本章开头预期的那两个例子了。我们将在这里混合搭配，这样你就可以看到如何将所有这些概念结合起来使用。让我们先写一些代码来生成一个质数列表，直到某个限制为止。请记住，我将写一个非常低效和基本的算法来检测质数。对你来说重要的是要集中精力关注本章主题的代码部分。

# 一个质数生成器

根据维基百科：

质数（或质数）是大于 1 的自然数，除了 1 和它本身之外没有正的除数。大于 1 的自然数，如果不是质数，则称为合数。

根据这个定义，如果我们考虑前 10 个自然数，我们可以看到 2、3、5 和 7 是质数，而 1、4、6、8、9 和 10 不是。为了让计算机告诉你一个数*N*是否是质数，你可以将该数除以范围[2，*N*)内的所有自然数。如果其中任何一个除法的余数为零，那么这个数就不是质数。废话够多了，让我们开始做生意吧。我将写两个版本的代码，第二个版本将利用`for...else`语法：

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

在前面的代码中有很多值得注意的地方。首先，我们建立了一个空的`primes`列表，它将在最后包含质数。限制是`100`，你可以看到我们在外部循环中调用`range()`的方式是包容的。如果我们写`range(2, upto)`，那就是*[2, upto)*，对吧？因此`range(2, upto + 1)`给我们*[2, upto + 1) == [2, upto]*。

所以，有两个`for`循环。在外部循环中，我们循环遍历候选质数，也就是从`2`到`upto`的所有自然数。在外部循环的每次迭代中，我们设置一个标志（在每次迭代时设置为`True`），然后开始将当前的`n`除以从`2`到`n - 1`的所有数字。如果我们找到`n`的一个适当的除数，这意味着`n`是合数，因此我们将标志设置为`False`并中断循环。请注意，当我们中断内部循环时，外部循环会继续正常进行。我们在找到`n`的适当除数后中断的原因是，我们不需要任何进一步的信息就能告诉`n`不是质数。

当我们检查`is_prime`标志时，如果它仍然是`True`，这意味着我们在[2，*n*)中找不到任何适当的除数，因此`n`是一个质数。我们将`n`附加到`primes`列表中，然后进行另一个迭代，直到`n`等于`100`。

运行这段代码会产生：

```py
$ python primes.py
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97] 
```

在我们继续之前，有一个问题：在外部循环的所有迭代中，有一个与其他所有迭代都不同。你能告诉哪一个，以及为什么吗？想一想，回到代码，试着自己找出答案，然后继续阅读。

你搞清楚了吗？如果没有，不要感到难过，这很正常。我让你做这个小练习，因为这是程序员一直在做的事情。通过简单地查看代码就能理解代码的功能是一种随着时间积累的技能。这非常重要，所以尽量在你能做的时候进行练习。我现在告诉你答案：与所有其他迭代不同的是第一个迭代。原因是因为在第一次迭代中，`n`是`2`。因此，最内层的`for`循环甚至不会运行，因为它是一个遍历`range(2, 2)`的`for`循环，那不就是[2, 2)吗？自己试一下，用这个可迭代对象写一个简单的`for`循环，放一个`print`在主体套件中，看看是否发生了什么（不会...）。

现在，从算法的角度来看，这段代码效率低下，所以让我们至少让它更美观：

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

更好了，对吧？`is_prime`标志已经消失，当我们知道内部的`for`循环没有遇到任何`break`语句时，我们将`n`附加到`primes`列表中。看看代码是不是更清晰，读起来更好了？

# 应用折扣

在这个例子中，我想向你展示一个我非常喜欢的技巧。在许多编程语言中，除了`if`/`elif`/`else`结构之外，无论以什么形式或语法，你都可以找到另一个语句，通常称为`switch`/`case`，在 Python 中缺少。它相当于一系列`if`/`elif`/.../`elif`/`else`子句，语法类似于这样（警告！JavaScript 代码！）：

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

在上面的代码中，我们在一个名为`day_number`的变量上进行`switch`。这意味着我们获取它的值，然后决定它适用于哪种情况（如果有的话）。从`1`到`5`有一个级联，这意味着无论数字是多少，[`1`, `5`]都会进入将`day`设置为`"Weekday"`的逻辑部分。然后我们有`0`和`6`的单个情况，以及一个`default`情况来防止错误，它警告系统`day_number`不是有效的日期数字，即不在[`0`, `6`]中。Python 完全能够使用`if`/`elif`/`else`语句实现这样的逻辑：

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

在上面的代码中，我们使用`if`/`elif`/`else`语句在 Python 中复制了 JavaScript 片段的相同逻辑。我只是举了一个例子，如果`day_number`不在[`0`, `6`]中，就会引发`ValueError`异常。这是一种可能的转换`switch`/`case`逻辑的方式，但还有另一种方式，有时称为分派，我将在下一个例子的最后版本中向你展示。

顺便问一下，你有没有注意到前面片段的第一行？你有没有注意到 Python 可以进行双重（实际上甚至是多重）比较？这太棒了！

让我们通过简单地编写一些代码来开始新的例子，根据顾客的优惠券价值为他们分配折扣。我会保持逻辑最低限度，记住我们真正关心的是理解条件和循环：

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

我们首先设置一些顾客。他们有一个订单总额，一个优惠券代码和一个 ID。我编造了四种不同类型的优惠券，两种是固定的，两种是基于百分比的。你可以看到，在`if`/`elif`/`else`级联中，我相应地应用折扣，并将其设置为`customer`字典中的`'discount'`键。

最后，我只是打印出部分数据，看看我的代码是否正常工作：

```py
$ python coupons.py
1 200 20.0
2 150 45.0
3 100 50.0
4 110 15.0
```

这段代码很容易理解，但所有这些子句有点混乱。一眼看去很难看出发生了什么，我不喜欢。在这种情况下，你可以利用字典来优化，就像这样：

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

运行前面的代码产生了与之前片段完全相同的结果。我们节省了两行，但更重要的是，我们在可读性方面获得了很多好处，因为现在`for`循环的主体只有三行，非常容易理解。这里的概念是使用字典作为**分发器**。换句话说，我们尝试根据一个代码（我们的`coupon_code`）从字典中获取一些东西，并通过使用`dict.get(key, default)`，我们确保当`code`不在字典中并且我们需要一个默认值时，我们也能满足。

请注意，我必须应用一些非常简单的线性代数来正确计算折扣。每个折扣在字典中都有一个百分比和固定部分，由一个二元组表示。通过应用`percent * total + fixed`，我们得到正确的折扣。当`percent`为`0`时，该公式只给出固定金额，当固定为`0`时，它给出`percent * total`。

这种技术很重要，因为它也用于其他上下文中，例如函数，它实际上比我们在前面的片段中看到的要强大得多。使用它的另一个优势是，您可以以这样的方式编写代码，使得`discounts`字典的键和值可以动态获取（例如，从数据库中获取）。这将允许代码适应您拥有的任何折扣和条件，而无需修改任何内容。

如果它对您不是完全清楚，我建议您花时间进行实验。更改值并添加打印语句，以查看程序运行时发生了什么。

# 快速浏览`itertools`模块

关于可迭代对象、迭代器、条件逻辑和循环的章节，如果没有提到`itertools`模块，就不完整。如果您喜欢迭代，这是一种天堂。

根据 Python 官方文档([`docs.python.org/2/library/itertools.html`](https://docs.python.org/2/library/itertools.html))，`itertools`模块是：

这个模块实现了一些迭代器构建块，受到 APL、Haskell 和 SML 中的构造的启发。每个都以适合 Python 的形式重新表达。该模块标准化了一组核心的快速、内存高效的工具，这些工具本身或组合在一起都很有用。它们一起构成了一个“迭代器代数”，使得可以在纯 Python 中简洁高效地构建专门的工具。

在这里我无法向您展示在这个模块中可以找到的所有好东西，所以我鼓励您自己去查看，我保证您会喜欢它。简而言之，它为您提供了三种广泛的迭代器类别。我将给您展示每一种迭代器中取出的一个非常小的例子，只是为了让您稍微流口水。

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

`count`工厂类创建一个不断计数的迭代器。它从`5`开始，然后不断加`3`。如果我们不想陷入无限循环，我们需要手动中断它。

# 在最短输入序列上终止的迭代器

这个类别非常有趣。它允许您基于多个迭代器创建一个迭代器，并根据某种逻辑组合它们的值。这里的关键是，在这些迭代器中，如果有任何一个比其他迭代器短，那么生成的迭代器不会中断，它将在最短的迭代器耗尽时停止。我知道这很理论化，所以让我用`compress`给您举个例子。这个迭代器根据选择器中的相应项目是`True`还是`False`，将数据返回给您：

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

请注意，`odd_selector` 和 `even_selector` 长度为 20，而 `data` 只有 10 个元素。`compress` 会在 `data` 产生最后一个元素时停止。运行此代码会产生以下结果：

```py
$ python compress.py
[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[0, 2, 4, 6, 8]
[1, 3, 5, 7, 9]
```

这是一种非常快速和方便的方法，可以从可迭代对象中选择元素。代码非常简单，只需注意，我们使用 `list()` 而不是 `for` 循环来迭代压缩调用返回的每个值，它们的作用是相同的，但是 `list()` 不执行一系列指令，而是将所有值放入列表并返回。

# 组合生成器

最后但并非最不重要的，组合生成器。如果你喜欢这种东西，这些真的很有趣。让我们来看一个关于排列的简单例子。

根据 Wolfram Mathworld：

排列，也称为“排列数”或“顺序”，是将有序列表 S 的元素重新排列，使其与 S 本身形成一一对应的重新排列。

例如，ABC 有六种排列：ABC、ACB、BAC、BCA、CAB 和 CBA。

如果一个集合有 *N* 个元素，那么它们的排列数是 *N!* (*N* 阶乘)。对于 ABC 字符串，排列数为 *3! = 3 * 2 * 1 = 6*。让我们用 Python 来做一下：

```py
# permutations.py
from itertools import permutations 
print(list(permutations('ABC'))) 
```

这段非常简短的代码片段产生了以下结果：

```py
$ python permutations.py
[('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]
```

当你玩排列时要非常小心。它们的数量增长速度与你要排列的元素的阶乘成比例，而这个数字可能会变得非常大，非常快。

# 总结

在本章中，我们迈出了扩展我们编码词汇的又一步。我们看到了如何通过评估条件来驱动代码的执行，以及如何循环和迭代序列和对象集合。这赋予了我们控制代码运行时发生的事情的能力，这意味着我们正在了解如何塑造它，使其做我们想要的事情，并对动态变化的数据做出反应。

我们还看到了如何在几个简单的例子中将所有东西组合在一起，最后，我们简要地看了一下 `itertools` 模块，其中充满了可以进一步丰富我们使用 Python 的有趣迭代器。

现在是时候换个方式，向前迈进一步，谈谈函数。下一章将全面讨论它们，因为它们非常重要。确保你对到目前为止所涵盖的内容感到舒适。我想给你提供一些有趣的例子，所以我会快一点。准备好了吗？翻页吧。
