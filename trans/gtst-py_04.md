# 函数，代码的构建块

创建架构就是把东西放在一起。把什么放在一起？函数和对象。"– 勒·柯布西耶

在前几章中，我们已经看到 Python 中的一切都是对象，函数也不例外。但是，函数究竟是什么？**函数**是一系列执行任务的指令，打包成一个单元。然后可以在需要的地方导入并使用这个单元。使用函数在代码中有许多优点，我们很快就会看到。

在本章中，我们将涵盖以下内容：

+   函数——它们是什么，为什么我们应该使用它们

+   作用域和名称解析

+   函数签名——输入参数和返回值

+   递归和匿名函数

+   导入对象以重用代码

我相信这句话，*一张图胜过千言万语*，在向一个对这个概念新手解释函数时尤其正确，所以请看一下下面的图表：

![](img/b4903c1d-cfdd-4893-a1f1-e7657a132673.png)

如您所见，函数是一组指令，作为一个整体打包，就像一个盒子。函数可以接受输入参数并产生输出值。这两者都是可选的，我们将在本章的示例中看到。

在 Python 中，使用`def`关键字定义函数，随后是函数名称，后面跟着一对括号（可能包含或不包含输入参数），冒号（`:`）表示函数定义行的结束。紧接着，缩进四个空格，我们找到函数的主体，这是函数在调用时将执行的一组指令。

请注意，缩进四个空格不是强制性的，但这是**PEP 8**建议的空格数量，在实践中是最广泛使用的间距度量。 

函数可能会返回输出，也可能不会。如果函数想要返回输出，它会使用`return`关键字，后面跟着期望的输出。如果您有鹰眼，您可能已经注意到在前面图表的输出部分的**Optional**后面有一个小*****。这是因为在 Python 中，函数总是返回一些东西，即使您没有明确使用`return`子句。如果函数在其主体中没有`return`语句，或者`return`语句本身没有给出值，函数将返回`None`。这种设计选择背后的原因超出了介绍性章节的范围，所以您需要知道的是，这种行为会让您的生活更轻松。一如既往，感谢 Python。

# 为什么要使用函数？

函数是任何语言中最重要的概念和构造之一，所以让我给你几个需要它们的原因：

+   它们减少了程序中的代码重复。通过将特定任务由一个良好的打包代码块处理，我们可以在需要时导入并调用它，无需重复其实现。

+   它们有助于将复杂的任务或过程分解为更小的块，每个块都成为一个函数。

+   它们隐藏了实现细节，使其用户不可见。

+   它们提高了可追溯性。

+   它们提高可读性。

让我们看一些示例，以更好地理解每一点。

# 减少代码重复

想象一下，您正在编写一段科学软件，需要计算素数直到一个限制，就像我们在上一章中所做的那样。您有一个很好的算法来计算它们，所以您将它复制并粘贴到需要的任何地方。然而，有一天，您的朋友*B.黎曼*给了您一个更好的算法来计算素数，这将为您节省大量时间。在这一点上，您需要检查整个代码库，并用新的代码替换旧的代码。

这实际上是一个不好的做法。这容易出错，你永远不知道你是不是误删或者误留了哪些代码行，当你把代码剪切粘贴到其他代码中时，你也可能会错过其中一个计算质数的地方，导致你的软件处于不一致的状态，同样的操作在不同的地方以不同的方式执行。如果你需要修复一个 bug 而不是用更好的版本替换代码，而你错过了其中一个地方呢？那将更糟糕。

那么，你应该怎么做？简单！你写一个函数`get_prime_numbers(upto)`，在任何需要质数列表的地方使用它。当*B. Riemann*给你新代码时，你只需要用新实现替换该函数的主体，就完成了！其余的软件将自动适应，因为它只是调用函数。

你的代码会更短，不会受到旧方法和新方法执行任务的不一致性的影响，也不会因为复制粘贴失败或疏忽而导致未检测到的 bug。使用函数，你只会从中获益，我保证。

# 拆分复杂任务

函数还非常有用，可以将长或复杂的任务分解为较小的任务。最终结果是，代码从中受益，例如可读性、可测试性和重用性。举个简单的例子，想象一下你正在准备一份报告。你的代码需要从数据源获取数据，解析数据，过滤数据，整理数据，然后对其运行一系列算法，以产生将供`Report`类使用的结果。阅读这样的程序并不罕见，它们只是一个大大的`do_report(data_source)`函数。有数十行或数百行代码以`return report`结束。

这些情况在科学代码中更常见，这些代码在算法上可能很出色，但有时在编写风格上缺乏经验丰富的程序员的触觉。现在，想象一下几百行代码。很难跟进，找到事情改变上下文的地方（比如完成一个任务并开始下一个任务）。你有这个画面了吗？好了。不要这样做！相反，看看这段代码：

```py
# data.science.example.py
def do_report(data_source):
    # fetch and prepare data
    data = fetch_data(data_source)
    parsed_data = parse_data(data)
    filtered_data = filter_data(parsed_data)
    polished_data = polish_data(filtered_data)

    # run algorithms on data
    final_data = analyse(polished_data)

    # create and return report
    report = Report(final_data)
    return report
```

前面的例子当然是虚构的，但你能看出来如果需要检查代码会有多容易吗？如果最终结果看起来不对，逐个调试`do_report`函数中的单个数据输出将会非常容易。此外，暂时从整个过程中排除部分过程也更容易（你只需要注释掉需要暂停的部分）。这样的代码更容易处理。

# 隐藏实现细节

让我们继续使用前面的例子来谈谈这一点。你可以看到，通过阅读`do_report`函数的代码，你可以在不阅读一行实现的情况下获得相当好的理解。这是因为函数隐藏了实现细节。这意味着，如果你不需要深入了解细节，你就不必强制自己去了解，就像如果`do_report`只是一个庞大的函数一样。为了理解发生了什么，你必须阅读每一行代码。而使用函数，你就不需要这样做。这减少了你阅读代码的时间，而在专业环境中，阅读代码所花费的时间比实际编写代码的时间要多得多，因此尽可能减少这部分时间非常重要。

# 提高可读性

程序员有时候看不出来为什么要写一个只有一两行代码的函数，所以让我们看一个例子，告诉你为什么你应该这样做。

想象一下，你需要计算两个矩阵的乘积：

![](img/2e838ad6-e5e9-46a0-993e-6ba47929a5a7.png)

你更喜欢阅读这段代码吗：

```py
# matrix.multiplication.nofunc.py
a = [[1, 2], [3, 4]]
b = [[5, 1], [2, 1]]

c = [[sum(i * j for i, j in zip(r, c)) for c in zip(*b)]
     for r in a]
```

或者你更喜欢这个：

```py
# matrix.multiplication.func.py
# this function could also be defined in another module
def matrix_mul(a, b):
    return [[sum(i * j for i, j in zip(r, c)) for c in zip(*b)]
            for r in a]

a = [[1, 2], [3, 4]]
b = [[5, 1], [2, 1]]
c = matrix_mul(a, b)
```

在第二个例子中，更容易理解`c`是`a`和`b`之间的乘法结果。通过代码更容易阅读，如果您不需要修改该乘法逻辑，甚至不需要深入了解实现细节。因此，在这里提高了可读性，而在第一个片段中，您将不得不花时间尝试理解那个复杂的列表推导在做什么。

# 提高可追溯性

想象一下，您已经编写了一个电子商务网站。您在页面上展示了产品价格。假设您的数据库中的价格是不含增值税（销售税）的，但是您希望在网站上以 20%的增值税显示它们。以下是从不含增值税价格计算含增值税价格的几种方法：

```py
# vat.py
price = 100  # GBP, no VAT
final_price1 = price * 1.2
final_price2 = price + price / 5.0
final_price3 = price * (100 + 20) / 100.0
final_price4 = price + price * 0.2
```

这四种不同的计算含增值税价格的方式都是完全可以接受的，我向您保证，这些年来我在同事的代码中找到了它们。现在，想象一下，您已经开始在不同的国家销售您的产品，其中一些国家有不同的增值税率，因此您需要重构您的代码（整个网站）以使增值税计算动态化。

您如何追踪所有进行增值税计算的地方？今天的编码是一个协作任务，您无法确定增值税是否仅使用了这些形式中的一种。相信我，这将是一场噩梦。

因此，让我们编写一个函数，该函数接受输入值`vat`和`price`（不含增值税），并返回含增值税的价格：

```py
# vat.function.py
def calculate_price_with_vat(price, vat):
    return price * (100 + vat) / 100
```

现在您可以导入该函数，并在您的网站的任何地方使用它，需要计算含增值税的价格，并且当您需要跟踪这些调用时，您可以搜索`calculate_price_with_vat`。

请注意，在前面的例子中，假定`price`是不含增值税的，`vat`是一个百分比值（例如 19、20 或 23）。

# 作用域和名称解析

您还记得我们在第一章中谈到的作用域和命名空间吗，*Python 的初级介绍*？我们现在将扩展这个概念。最后，我们可以谈论函数，这将使一切更容易理解。让我们从一个非常简单的例子开始：

```py
# scoping.level.1.py
def my_function():
    test = 1  # this is defined in the local scope of the function
    print('my_function:', test)

test = 0  # this is defined in the global scope
my_function()
print('global:', test)
```

在前面的例子中，我在两个不同的地方定义了`test`名称。实际上它在两个不同的作用域中。一个是全局作用域（`test = 0`），另一个是`my_function`函数的局部作用域（`test = 1`）。如果您执行该代码，您会看到这个：

```py
$ python scoping.level.1.py
my_function: 1
global: 0
```

很明显，`test = 1`覆盖了`my_function`中的`test = 0`赋值。在全局上下文中，`test`仍然是`0`，正如您从程序的输出中所看到的，但是我们在函数体中再次定义了`test`名称，并将其指向值为`1`的整数。因此，这两个`test`名称都存在，一个在全局作用域中，指向值为`0`的`int`对象，另一个在`my_function`作用域中，指向值为`1`的`int`对象。让我们注释掉`test = 1`的那一行。Python 会在下一个封闭的命名空间中搜索`test`名称（回想一下**LEGB**规则：**local**，**enclosing**，**global**，**built-in**，在第一章中描述，*Python 的初级介绍*），在这种情况下，我们将看到值`0`被打印两次。在您的代码中尝试一下。

现在，让我们提高一下难度：

```py
# scoping.level.2.py
def outer():
    test = 1  # outer scope
    def inner():
        test = 2  # inner scope
        print('inner:', test)

    inner()
    print('outer:', test)

test = 0  # global scope
outer()
print('global:', test)
```

在前面的代码中，我们有两个级别的遮蔽。一个级别在函数`outer`中，另一个级别在函数`inner`中。这并不是什么难事，但可能会有些棘手。如果我们运行代码，我们会得到：

```py
$ python scoping.level.2.py
inner: 2
outer: 1
global: 0
```

尝试注释掉`test = 1`行。您能猜到结果会是什么吗？嗯，当达到`print('outer:', test)`行时，Python 将不得不在下一个封闭范围中查找`test`，因此它将找到并打印`0`，而不是`1`。确保您也注释掉`test = 2`，以查看您是否理解发生了什么，以及 LEGB 规则是否清晰，然后再继续。

另一个需要注意的事情是，Python 允许您在另一个函数中定义一个函数。内部函数的名称在外部函数的命名空间中定义，就像任何其他名称一样。

# global 和 nonlocal 语句

回到前面的例子，我们可以通过使用这两个特殊语句之一来更改对`test`名称的遮蔽：`global`和`nonlocal`。正如您从前面的例子中看到的，当我们在`inner`函数中定义`test = 2`时，我们既不会覆盖`outer`函数中的`test`，也不会覆盖全局范围中的`test`。如果我们在不定义它们的嵌套范围中使用它们，我们可以读取这些名称，但是我们不能修改它们，因为当我们编写赋值指令时，实际上是在当前范围中定义一个新名称。

我们如何改变这种行为呢？嗯，我们可以使用`nonlocal`语句。根据官方文档：

“nonlocal 语句使列出的标识符引用最近的封闭范围中先前绑定的变量，不包括全局变量。”

让我们在`inner`函数中引入它，看看会发生什么：

```py
# scoping.level.2.nonlocal.py
def outer():
    test = 1  # outer scope
    def inner():
        nonlocal test
        test = 2  # nearest enclosing scope (which is 'outer')
        print('inner:', test)

    inner()
    print('outer:', test)

test = 0  # global scope
outer()
print('global:', test)
```

请注意，在`inner`函数的主体中，我已经声明了`test`名称为`nonlocal`。运行此代码会产生以下结果：

```py
$ python scoping.level.2.nonlocal.py
inner: 2
outer: 2
global: 0
```

哇，看看那个结果！这意味着，通过在`inner`函数中声明`test`为`nonlocal`，我们实际上得到了将`test`名称绑定到在`outer`函数中声明的名称。如果我们从`inner`函数中删除`nonlocal test`行并尝试在`outer`函数中尝试相同的技巧，我们将得到一个`SyntaxError`，因为`nonlocal`语句在封闭范围上运行，不包括全局范围。

那么有没有办法到达全局命名空间中的`test = 0`呢？当然，我们只需要使用`global`语句：

```py
# scoping.level.2.global.py
def outer():
    test = 1  # outer scope
    def inner():
        global test
        test = 2  # global scope
        print('inner:', test)

    inner()
    print('outer:', test)

test = 0  # global scope
outer()
print('global:', test)
```

请注意，我们现在已经声明了`test`名称为`global`，这基本上将其绑定到我们在全局命名空间中定义的名称（`test = 0`）。运行代码，您应该会得到以下结果：

```py
$ python scoping.level.2.global.py
inner: 2
outer: 1
global: 2
```

这表明受`test = 2`赋值影响的名称现在是`global`。这个技巧在`outer`函数中也会起作用，因为在这种情况下，我们是在引用全局范围。自己尝试一下，看看有什么变化，熟悉作用域和名称解析，这非常重要。此外，您能告诉在前面的例子中如果在`outer`之外定义`inner`会发生什么吗？

# 输入参数

在本章的开始，我们看到函数可以接受输入参数。在我们深入讨论所有可能类型的参数之前，让我们确保您清楚地理解了向函数传递参数的含义。有三个关键点需要记住：

+   参数传递只不过是将对象分配给本地变量名称

+   在函数内部将对象分配给参数名称不会影响调用者

+   更改函数中的可变对象参数会影响调用者

让我们分别看一下这些要点的例子。

# 参数传递

看一下以下代码。我们在全局范围内声明一个名称`x`，然后我们声明一个函数`func(y)`，最后我们调用它，传递`x`：

```py
# key.points.argument.passing.py
x = 3
```

```py
def func(y):
    print(y)
func(x)  # prints: 3
```

当使用`x`调用`func`时，在其局部范围内，创建了一个名为`y`的名称，并且它指向与`x`指向的相同对象。这可以通过以下图表更清楚地解释（不用担心 Python 3.3，这是一个未更改的功能）：

![](img/01591ef9-a866-4954-880d-44fabb616a9e.jpg)

前面图的右侧描述了程序在执行到结束后的状态，即`func`返回(`None`)后的状态。看一下 Frames 列，注意我们在全局命名空间（全局帧）中有两个名称，`x`和`func`，分别指向一个`int`（值为**3**）和一个`function`对象。在下面的名为`func`的矩形中，我们可以看到函数的局部命名空间，其中只定义了一个名称：`y`。因为我们用`x`调用了`func`（图的左侧第 5 行），`y`指向与`x`指向的相同的对象。这就是在将参数传递给函数时发生的情况。如果我们在函数定义中使用名称`x`而不是`y`，情况将完全相同（可能一开始有点混乱），函数中会有一个局部的`x`，而外部会有一个全局的`x`，就像我们在本章前面的*作用域和名称解析*部分中看到的那样。

总之，实际发生的是函数在其局部范围内创建了作为参数定义的名称，当我们调用它时，我们基本上告诉 Python 这些名称必须指向哪些对象。

# 分配给参数名称不会影响调用者

这一点一开始可能有点难以理解，所以让我们看一个例子：

```py
# key.points.assignment.py
x = 3
def func(x):
    x = 7  # defining a local x, not changing the global one
func(x)
print(x)  # prints: 3
```

在前面的代码中，当执行`x = 7`行时，在`func`函数的局部范围内，名称`x`指向一个值为`7`的整数，而全局的`x`保持不变。

# 更改可变对象会影响调用者

这是最后一点，非常重要，因为 Python 在处理可变对象时表现得似乎有所不同（尽管只是表面上）。让我们看一个例子：

```py
# key.points.mutable.py
x = [1, 2, 3]
def func(x):
    x[1] = 42  # this affects the caller!

func(x)
print(x)  # prints: [1, 42, 3]
```

哇，我们实际上改变了原始对象！如果你仔细想想，这种行为并不奇怪。函数中的`x`名称被设置为通过函数调用指向调用者对象，在函数体内，我们没有改变`x`，也就是说，我们没有改变它的引用，换句话说，我们没有改变`x`指向的对象。我们访问该对象在位置 1 的元素，并更改其值。

记住*输入参数*部分下的第 2 点：*在函数内将对象分配给参数名称不会影响调用者*。如果这对你来说很清楚，下面的代码就不会让你感到惊讶：

```py
# key.points.mutable.assignment.py
x = [1, 2, 3]
def func(x):
    x[1] = 42  # this changes the caller!
    x = 'something else'  # this points x to a new string object

func(x)
print(x)  # still prints: [1, 42, 3]
```

看一下我标记的两行。一开始，就像以前一样，我们再次访问调用者对象，位于位置 1，并将其值更改为数字`42`。然后，我们重新分配`x`指向`'something else'`字符串。这样留下了调用者不变，实际上，输出与前面片段的输出相同。

花点时间来玩弄这个概念，并尝试使用打印和调用`id`函数，直到你的思路清晰。这是 Python 的一个关键方面，必须非常清楚，否则你可能会在代码中引入微妙的错误。再次强调，Python Tutor 网站（[`www.pythontutor.com/`](http://www.pythontutor.com/)）将通过可视化这些概念来帮助你很多。

现在我们对输入参数及其行为有了很好的理解，让我们看看如何指定它们。

# 如何指定输入参数

指定输入参数的五种不同方式：

+   位置参数

+   关键字参数

+   变量位置参数

+   可变关键字参数

+   仅关键字参数

让我们逐一来看看它们。

# 位置参数

位置参数从左到右读取，它们是最常见的参数类型：

```py
# arguments.positional.py
def func(a, b, c):
    print(a, b, c)
func(1, 2, 3)  # prints: 1 2 3
```

没有什么别的要说的。它们可以是任意多个，按位置分配。在函数调用中，`1`排在前面，`2`排在第二，`3`排在第三，因此它们分别分配给`a`、`b`和`c`。

# 关键字参数和默认值

**关键字参数**通过使用`name=value`语法进行分配：

```py
# arguments.keyword.py
def func(a, b, c):
    print(a, b, c)
func(a=1, c=2, b=3)  # prints: 1 3 2
```

关键字参数是按名称匹配的，即使它们不遵守定义的原始位置（当我们混合和匹配不同类型的参数时，稍后我们将看到这种行为的限制）。

关键字参数的对应物，在定义方面，是**默认值**。 语法是相同的，`name=value`，并且允许我们不必提供参数，如果我们对给定的默认值感到满意：

```py
# arguments.default.py
def func(a, b=4, c=88):
    print(a, b, c)

func(1)  # prints: 1 4 88
func(b=5, a=7, c=9)  # prints: 7 5 9
func(42, c=9)  # prints: 42 4 9
func(42, 43, 44)  # prints: 42, 43, 44
```

有两件非常重要的事情需要注意。 首先，你不能在位置参数的左边指定默认参数。 其次，在这些例子中，注意当参数在列表中没有使用`argument_name=value`语法时，它必须是列表中的第一个参数，并且总是分配给`a`。 还要注意，以位置方式传递值仍然有效，并且遵循函数签名顺序（示例的最后一行）。

尝试混淆这些参数，看看会发生什么。 Python 错误消息非常擅长告诉你出了什么问题。 例如，如果你尝试了这样的东西：

```py
# arguments.default.error.py
def func(a, b=4, c=88):
    print(a, b, c)
func(b=1, c=2, 42)  # positional argument after keyword one
```

你会得到以下错误：

```py
$ python arguments.default.error.py
 File "arguments.default.error.py", line 4
 func(b=1, c=2, 42) # positional argument after keyword one
 ^
SyntaxError: positional argument follows keyword argument
```

这会告诉你你错误地调用了函数。

# 可变位置参数

有时候你可能想要向函数传递可变数量的位置参数，Python 为你提供了这样的能力。 让我们看一个非常常见的用例，`minimum`函数。 这是一个计算其输入值的最小值的函数：

```py
# arguments.variable.positional.py
def minimum(*n):
    # print(type(n))  # n is a tuple
    if n:  # explained after the code
        mn = n[0]
        for value in n[1:]:
            if value < mn:
                mn = value
        print(mn)

minimum(1, 3, -7, 9)  # n = (1, 3, -7, 9) - prints: -7
minimum()             # n = () - prints: nothing
```

正如你所看到的，当我们在参数名前面加上`*`时，我们告诉 Python 该参数将根据函数的调用方式收集可变数量的位置参数。 在函数内部，`n`是一个元组。 取消注释`print(type(n))`，自己看看并玩一会儿。

你是否注意到我们如何用简单的`if n：`检查`n`是否为空？ 这是因为在 Python 中，集合对象在非空时求值为`True`，否则为`False`。 这对于元组，集合，列表，字典等都是成立的。

还有一件事要注意的是，当我们在没有参数的情况下调用函数时，我们可能希望抛出错误，而不是默默地什么都不做。 在这种情况下，我们不关心使这个函数健壮，而是理解可变位置参数。

让我们举个例子，展示两件事，根据我的经验，对于新手来说是令人困惑的：

```py
# arguments.variable.positional.unpacking.py
def func(*args):
    print(args)

values = (1, 3, -7, 9)
func(values)   # equivalent to: func((1, 3, -7, 9))
func(*values)  # equivalent to: func(1, 3, -7, 9)
```

好好看看前面例子的最后两行。 在第一个例子中，我们用一个参数调用`func`，一个四元组。 在第二个例子中，通过使用`*`语法，我们正在做一种叫做**解包**的事情，这意味着四元组被解包，函数被调用时有四个参数：`1, 3, -7, 9`。

这种行为是 Python 为你做的魔术的一部分，允许你在动态调用函数时做一些惊人的事情。

# 变量关键字参数

可变关键字参数与可变位置参数非常相似。 唯一的区别是语法（`**`而不是`*`），以及它们被收集在一个字典中。 集合和解包的工作方式相同，所以让我们看一个例子：

```py
# arguments.variable.keyword.py
def func(**kwargs):
    print(kwargs)

# All calls equivalent. They print: {'a': 1, 'b': 42}
func(a=1, b=42)
func(**{'a': 1, 'b': 42})
func(**dict(a=1, b=42))
```

在前面的例子中，所有的调用都是等价的。 你可以看到，在函数定义中在参数名前面添加`**`告诉 Python 使用该名称来收集可变数量的关键字参数。 另一方面，当我们调用函数时，我们可以显式地传递`name=value`参数，或者使用相同的`**`语法解包字典。

能够传递可变数量的关键字参数之所以如此重要的原因可能目前还不明显，那么，来一个更现实的例子怎么样？让我们定义一个连接到数据库的函数。我们希望通过简单调用这个函数而连接到默认数据库。我们还希望通过传递适当的参数来连接到任何其他数据库。在继续阅读之前，试着花几分钟时间自己想出一个解决方案：

```py
# arguments.variable.db.py
def connect(**options):
    conn_params = {
        'host': options.get('host', '127.0.0.1'),
        'port': options.get('port', 5432),
        'user': options.get('user', ''),
        'pwd': options.get('pwd', ''),
    }
    print(conn_params)
    # we then connect to the db (commented out)
    # db.connect(**conn_params)

connect()
connect(host='127.0.0.42', port=5433)
connect(port=5431, user='fab', pwd='gandalf')
```

注意在函数中，我们可以准备一个连接参数的字典（`conn_params`），使用默认值作为回退，允许在函数调用中提供时进行覆盖。有更好的方法可以用更少的代码行来实现，但我们现在不关心这个。运行上述代码会产生以下结果：

```py
$ python arguments.variable.db.py
{'host': '127.0.0.1', 'port': 5432, 'user': '', 'pwd': ''}
{'host': '127.0.0.42', 'port': 5433, 'user': '', 'pwd': ''}
{'host': '127.0.0.1', 'port': 5431, 'user': 'fab', 'pwd': 'gandalf'}
```

注意函数调用和输出之间的对应关系。注意默认值是如何根据传递给函数的参数进行覆盖的。

# 仅限关键字参数

Python 3 允许一种新类型的参数：**仅限关键字**参数。我们只会简要地研究它们，因为它们的使用情况并不那么频繁。有两种指定它们的方式，要么在可变位置参数之后，要么在单独的`*`之后。让我们看一个例子：

```py
# arguments.keyword.only.py
def kwo(*a, c):
    print(a, c)

kwo(1, 2, 3, c=7)  # prints: (1, 2, 3) 7
kwo(c=4)  # prints: () 4
# kwo(1, 2)  # breaks, invalid syntax, with the following error
# TypeError: kwo() missing 1 required keyword-only argument: 'c'

def kwo2(a, b=42, *, c):
    print(a, b, c)

kwo2(3, b=7, c=99)  # prints: 3 7 99
kwo2(3, c=13)  # prints: 3 42 13
# kwo2(3, 23)  # breaks, invalid syntax, with the following error
# TypeError: kwo2() missing 1 required keyword-only argument: 'c'
```

如预期的那样，函数`kwo`接受可变数量的位置参数（`a`）和一个仅限关键字参数`c`。调用的结果很直接，您可以取消注释第三个调用以查看 Python 返回的错误。

相同的规则适用于函数`kwo2`，它与`kwo`不同之处在于它接受一个位置参数`a`，一个关键字参数`b`，然后是一个仅限关键字参数`c`。您可以取消注释第三个调用以查看错误。

现在你知道如何指定不同类型的输入参数了，让我们看看如何在函数定义中结合它们。

# 结合输入参数

只要遵循以下顺序规则，就可以结合输入参数：

+   在定义函数时，普通的位置参数先出现（`name`），然后是任何默认参数（`name=value`），然后是可变位置参数（`*name`或简单的`*`），然后是任何关键字参数（`name`或`name=value`形式都可以），最后是任何可变关键字参数（`**name`）。

+   另一方面，在调用函数时，参数必须按照以下顺序给出：先是位置参数（`value`），然后是任意组合的关键字参数（`name=value`），可变位置参数（`*name`），然后是可变关键字参数（`**name`）。

由于这在理论世界中留下来可能有点棘手，让我们看几个快速的例子：

```py
# arguments.all.py
def func(a, b, c=7, *args, **kwargs):
    print('a, b, c:', a, b, c)
    print('args:', args)
    print('kwargs:', kwargs)

func(1, 2, 3, *(5, 7, 9), **{'A': 'a', 'B': 'b'})
func(1, 2, 3, 5, 7, 9, A='a', B='b')  # same as previous one
```

注意函数定义中参数的顺序，两个调用是等价的。在第一个调用中，我们使用了可迭代对象和字典的解包操作符，而在第二个调用中，我们使用了更明确的语法。执行这个函数会产生以下结果（我只打印了一个调用的结果，另一个结果相同）：

```py
$ python arguments.all.py
a, b, c: 1 2 3
args: (5, 7, 9)
kwargs: {'A': 'a', 'B': 'b'}
```

现在让我们看一个关键字参数的例子：

```py
# arguments.all.kwonly.py
def func_with_kwonly(a, b=42, *args, c, d=256, **kwargs):
    print('a, b:', a, b)
    print('c, d:', c, d)
    print('args:', args)
    print('kwargs:', kwargs)

# both calls equivalent
func_with_kwonly(3, 42, c=0, d=1, *(7, 9, 11), e='E', f='F')
func_with_kwonly(3, 42, *(7, 9, 11), c=0, d=1, e='E', f='F')
```

请注意我在函数声明中突出显示了仅限关键字参数。它们出现在`*args`可变位置参数之后，如果它们直接出现在单个`*`之后，情况也是一样的（在这种情况下就没有可变位置参数了）。

执行这个函数会产生以下结果（我只打印了一个调用的结果）：

```py
$ python arguments.all.kwonly.py
a, b: 3 42
c, d: 0 1
args: (7, 9, 11)
kwargs: {'e': 'E', 'f': 'F'}
```

还要注意的一件事是我给可变位置和关键字参数起的名字。你可以选择不同的名字，但要注意`args`和`kwargs`是这些参数的通用约定名称，至少是通用的。

# 额外的解包概括

Python 3.5 中引入的最近的新功能之一是能够扩展可迭代（`*`）和字典（`**`）解包操作符，以允许在更多位置、任意次数和额外情况下解包。我将给你一个关于函数调用的例子：

```py
# additional.unpacking.py
def additional(*args, **kwargs):
    print(args)
    print(kwargs)

args1 = (1, 2, 3)
args2 = [4, 5]
kwargs1 = dict(option1=10, option2=20)
kwargs2 = {'option3': 30}
additional(*args1, *args2, **kwargs1, **kwargs2)
```

在前面的例子中，我们定义了一个简单的函数，打印它的输入参数`args`和`kwargs`。新功能在于我们调用这个函数的方式。注意我们如何解包多个可迭代对象和字典，并且它们在`args`和`kwargs`下正确地合并。这个功能之所以重要的原因在于，它允许我们不必在代码中合并`args1`和`args2`，以及`kwargs1`和`kwargs2`。运行代码会产生：

```py
$ python additional.unpacking.py
(1, 2, 3, 4, 5)
{'option1': 10, 'option2': 20, 'option3': 30}
```

请参考 PEP 448（[`www.python.org/dev/peps/pep-0448/`](https://www.python.org/dev/peps/pep-0448/)）来了解这一新功能的全部内容，并查看更多示例。

# 避免陷阱！可变默认值

在 Python 中需要非常注意的一件事是，默认值是在`def`时创建的，因此，对同一个函数的后续调用可能会根据它们的默认值的可变性而表现得不同。让我们看一个例子：

```py
# arguments.defaults.mutable.py
def func(a=[], b={}):
    print(a)
    print(b)
    print('#' * 12)
    a.append(len(a))  # this will affect a's default value
    b[len(a)] = len(a)  # and this will affect b's one

func()
func()
func()
```

两个参数都有可变的默认值。这意味着，如果你影响这些对象，任何修改都会在后续的函数调用中保留下来。看看你能否理解这些调用的输出：

```py
$ python arguments.defaults.mutable.py
[]
{}
############
[0]
{1: 1}
############
[0, 1]
{1: 1, 2: 2}
############
```

很有趣，不是吗？虽然这种行为一开始可能看起来很奇怪，但实际上是有道理的，而且非常方便，例如，在使用记忆化技术时（如果你感兴趣，可以搜索一下）。更有趣的是，当我们在调用之间引入一个不使用默认值的调用时，会发生什么：

```py
# arguments.defaults.mutable.intermediate.call.py
func()
func(a=[1, 2, 3], b={'B': 1})
func()
```

当我们运行这段代码时，输出如下：

```py
$ python arguments.defaults.mutable.intermediate.call.py
[]
{}
############
[1, 2, 3]
{'B': 1}
############
[0]
{1: 1}
############
```

这个输出告诉我们，即使我们用其他值调用函数，默认值仍然保留。一个让人想到的问题是，我怎样才能每次都得到一个全新的空值？嗯，惯例是这样的：

```py
# arguments.defaults.mutable.no.trap.py
def func(a=None):
    if a is None:
        a = []
    # do whatever you want with `a` ...
```

请注意，通过使用前面的技术，如果在调用函数时没有传递`a`，你总是会得到一个全新的空列表。

好了，输入就到此为止，让我们看看另一面，输出。

# 返回值

函数的返回值是 Python 领先于大多数其他语言的地方之一。通常函数允许返回一个对象（一个值），但在 Python 中，你可以返回一个元组，这意味着你可以返回任何你想要的东西。这个特性允许程序员编写在其他语言中要难得多或者肯定更加繁琐的软件。我们已经说过，要从函数中返回一些东西，我们需要使用`return`语句，后面跟着我们想要返回的东西。在函数体中可以有多个 return 语句。

另一方面，如果在函数体内部我们没有返回任何东西，或者调用了一个裸的`return`语句，函数将返回`None`。这种行为是无害的，尽管我在这里没有足够的空间详细解释为什么 Python 被设计成这样，但我只想告诉你，这个特性允许了几种有趣的模式，并且证实了 Python 是一种非常一致的语言。

我说这是无害的，因为你从来不会被迫收集函数调用的结果。我将用一个例子来说明我的意思：

```py
# return.none.py
def func():
    pass
func()  # the return of this call won't be collected. It's lost.
a = func()  # the return of this one instead is collected into `a`
print(a)  # prints: None
```

请注意，函数的整个主体只由`pass`语句组成。正如官方文档告诉我们的那样，`pass`是一个空操作。当它被执行时，什么都不会发生。当语法上需要一个语句，但不需要执行任何代码时，它是有用的。在其他语言中，我们可能会用一对花括号（`{}`）来表示这一点，定义一个*空作用域*，但在 Python 中，作用域是通过缩进代码来定义的，因此`pass`这样的语句是必要的。

还要注意，`func`函数的第一个调用返回一个值（`None`），我们没有收集。正如我之前所说，收集函数调用的返回值并不是强制性的。

现在，这很好但不是很有趣，那么我们来写一个有趣的函数吧？记住，在第一章中，*Python 的初步介绍*，我们谈到了函数的阶乘。让我们在这里写一个（为简单起见，我将假设函数总是以适当的值正确调用，因此我不会对输入参数进行健全性检查）：

```py
# return.single.value.py
def factorial(n):
    if n in (0, 1):
        return 1
    result = n
    for k in range(2, n):
        result *= k
    return result

f5 = factorial(5)  # f5 = 120
```

请注意我们有两个返回点。如果“n”是“0”或“1”（在 Python 中通常使用`in`类型的检查，就像我所做的那样，而不是更冗长的`if n == 0 or n == 1：`），我们返回“1”。否则，我们执行所需的计算并返回“result”。让我们尝试以更简洁的方式编写这个函数：

```py
# return.single.value.2.py from functools import reduce
from operator import mul

def factorial(n):
    return reduce(mul, range(1, n + 1), 1)

f5 = factorial(5)  # f5 = 120
```

我知道你在想什么：一行？Python 是优雅而简洁的！我认为这个函数即使你从未见过`reduce`或`mul`，也是可读的，但如果你无法阅读或理解它，请花几分钟时间在 Python 文档上做一些研究，直到它的行为对你清晰为止。能够在文档中查找函数并理解他人编写的代码是每个开发人员都需要能够执行的任务，所以把它当作一个挑战。

为此，请确保查找“help”函数，在控制台探索时会非常有帮助。

# 返回多个值

与大多数其他语言不同，在 Python 中很容易从函数返回多个对象。这个特性打开了一个全新的可能性世界，并允许你以其他语言难以复制的风格编码。我们的思维受到我们使用的工具的限制，因此当 Python 给予你比其他语言更多的自由时，实际上也在提高你自己的创造力。返回多个值非常容易，你只需使用元组（显式或隐式）。让我们看一个简单的例子，模仿`divmod`内置函数：

```py
# return.multiple.py
def moddiv(a, b):
    return a // b, a % b

print(moddiv(20, 7))  # prints (2, 6)
```

我本可以将前面代码中的突出部分包装在括号中，使其成为一个显式的元组，但没有必要。前面的函数同时返回了结果和除法的余数。

在这个示例的源代码中，我留下了一个简单的测试函数的示例，以确保我的代码进行了正确的计算。

# 一些建议

在编写函数时，遵循指南非常有用，这样你就可以写得更好。我将快速指出其中一些：

+   函数应该只做一件事：只做一件事的函数很容易用一句简短的话来描述。做多件事的函数可以拆分成做一件事的较小函数。这些较小的函数通常更容易阅读和理解。还记得我们几页前看到的数据科学示例吗？

+   函数应该尽可能小：它们越小，测试和编写它们就越容易，以便它们只做一件事。

+   输入参数越少越好：接受大量参数的函数很快就变得难以管理（除其他问题外）。

+   函数的返回值应该是一致的：返回“False”或“None”并不相同，即使在布尔上下文中它们都会评估为“False”。 “False”表示我们有信息（“False”），而“None”表示没有信息。尝试编写函数以一致的方式返回，无论其主体发生了什么。

+   **函数不应该有副作用**：换句话说，函数不应该影响你调用它们的值。这可能是目前最难理解的陈述，所以我将给你一个使用列表的例子。在下面的代码中，请注意`sorted`函数没有对`numbers`进行排序，它实际上返回了一个已排序的`numbers`的副本。相反，`list.sort()`方法是在`numbers`对象本身上操作，这是可以的，因为它是一个方法（属于对象的函数，因此有权修改它）：

```py
>>> numbers = [4, 1, 7, 5]
>>> sorted(numbers)  # won't sort the original `numbers` list
[1, 4, 5, 7]
>>> numbers  # let's verify
[4, 1, 7, 5]  # good, untouched
```

```py
>>> numbers.sort()  # this will act on the list
>>> numbers
[1, 4, 5, 7]
```

遵循这些准则，你将编写更好的函数，这将为你服务。

# 递归函数

当一个函数调用自身来产生结果时，它被称为**递归**。有时，递归函数非常有用，因为它们使编写代码变得更容易。有些算法使用递归范式编写起来非常容易，而其他一些则不是。没有递归函数无法以迭代方式重写，因此通常由程序员选择处理当前情况的最佳方法。

递归函数的主体通常有两个部分：一个是返回值取决于对自身的后续调用，另一个是不取决于后续调用的情况（称为基本情况）。

例如，我们可以考虑（希望现在已经熟悉的）`factorial`函数，*N*！。基本情况是当*N*为`0`或`1`时。函数返回`1`，无需进一步计算。另一方面，在一般情况下，*N*！返回乘积*1 * 2 * ... * (N-1) * N*。如果你仔细想想，*N*！可以这样重写：*N! = (N-1)! * N*。作为一个实际的例子，考虑*5! = 1 * 2 * 3 * 4 * 5 = (1 * 2 * 3 * 4) * 5 = 4! * 5*。

让我们用代码写下来：

```py
# recursive.factorial.py
def factorial(n):
    if n in (0, 1):  # base case
        return 1
    return factorial(n - 1) * n  # recursive case
```

在编写递归函数时，始终考虑你进行了多少嵌套调用，因为有一个限制。有关此信息的更多信息，请查看`sys.getrecursionlimit()`和`sys.setrecursionlimit()`。

在编写算法时经常使用递归函数，它们编写起来非常有趣。作为练习，尝试使用递归和迭代方法解决一些简单问题。

# 匿名函数

我想谈谈的最后一种函数类型是**匿名**函数。这些函数在 Python 中称为**lambda**，通常在需要一个完全成熟的带有自己名称的函数会显得有些多余时使用，我们只需要一个快速、简单的一行代码来完成任务。

假设你想要一个包含* N *的所有倍数的列表。假设你想使用`filter`函数进行筛选，该函数接受一个函数和一个可迭代对象，并构造一个筛选对象，你可以对其进行迭代，从可迭代对象中返回`True`的元素。如果不使用匿名函数，你可以这样做：

```py
# filter.regular.py
def is_multiple_of_five(n):
    return not n % 5

def get_multiples_of_five(n):
    return list(filter(is_multiple_of_five, range(n)))
```

注意我们如何使用`is_multiple_of_five`来过滤前`n`个自然数。这似乎有点多余，任务很简单，我们不需要保留`is_multiple_of_five`函数以备其他用途。让我们使用 lambda 函数重新编写它：

```py
# filter.lambda.py
def get_multiples_of_five(n):
    return list(filter(lambda k: not k % 5, range(n)))
```

逻辑完全相同，但现在过滤函数是一个 lambda。定义 lambda 非常简单，遵循这种形式：`func_name = lambda [parameter_list]: expression`。返回一个函数对象，等同于这个：`def func_name([parameter_list]): return expression`。

请注意，可选参数在常见的语法中用方括号括起来表示。

让我们再看看两种形式定义的等价函数的另外一些例子：

```py
# lambda.explained.py
# example 1: adder
def adder(a, b):
    return a + b

# is equivalent to:
adder_lambda = lambda a, b: a + b

# example 2: to uppercase
def to_upper(s):
    return s.upper()

# is equivalent to:
to_upper_lambda = lambda s: s.upper()
```

前面的例子非常简单。第一个例子是两个数字相加，第二个例子是生成字符串的大写版本。请注意，我将`lambda`表达式返回的内容分配给了一个名称（`adder_lambda`，`to_upper_lambda`），但在我们在`filter`示例中使用 lambda 时，没有必要这样做。

# 函数属性

每个函数都是一个完整的对象，因此它们有很多属性。其中一些是特殊的，可以用一种内省的方式在运行时检查函数对象。下面的脚本是一个例子，展示了其中一部分属性以及如何显示它们的值，用于一个示例函数：

```py
# func.attributes.py
def multiplication(a, b=1):
    """Return a multiplied by b. """
    return a * b

special_attributes = [
    "__doc__", "__name__", "__qualname__", "__module__",
    "__defaults__", "__code__", "__globals__", "__dict__",
    "__closure__", "__annotations__", "__kwdefaults__",
]

for attribute in special_attributes:
    print(attribute, '->', getattr(multiplication, attribute))
```

我使用内置的`getattr`函数来获取这些属性的值。`getattr(obj, attribute)`等同于`obj.attribute`，在我们需要在运行时使用字符串名称获取属性时非常方便。运行这个脚本会产生：

```py
$ python func.attributes.py
__doc__ -> Return a multiplied by b.
__name__ -> multiplication
__qualname__ -> multiplication
__module__ -> __main__
__defaults__ -> (1,)
__code__ -> <code object multiplication at 0x10caf7660, file "func.attributes.py", line 1>
__globals__ -> {...omitted...}
__dict__ -> {}
```

```py
__closure__ -> None
__annotations__ -> {}
__kwdefaults__ -> None
```

我省略了`__globals__`属性的值，因为它太大了。关于这个属性的含义的解释可以在*Python 数据模型*文档页面的*Callable types*部分找到（[`docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy`](https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy)）。如果你想看到一个对象的所有属性，只需调用`dir(object_name)`，你将得到所有属性的列表。

# 内置函数

Python 自带了很多内置函数。它们可以在任何地方使用，你可以通过检查`builtins`模块的`dir(__builtins__)`来获取它们的列表，或者查看官方 Python 文档。不幸的是，我没有足够的空间在这里介绍所有这些函数。我们已经见过其中一些，比如`any`、`bin`、`bool`、`divmod`、`filter`、`float`、`getattr`、`id`、`int`、`len`、`list`、`min`、`print`、`set`、`tuple`、`type`和`zip`，但还有很多，你至少应该阅读一次。熟悉它们，进行实验，为每个函数编写一小段代码，并确保你能随时使用它们。

# 最后一个例子

在我们结束本章之前，最后一个例子怎么样？我在想我们可以编写一个函数来生成一个小于某个限制的质数列表。我们已经见过这个代码了，所以让我们把它变成一个函数，并且为了保持趣味性，让我们稍微优化一下。

事实证明，你不需要将一个数*N*除以从*2*到*N*-1 的所有数字来判断它是否是质数。你可以停在*√N*。此外，你不需要测试从*2*到*√N*的所有数字的除法，你可以只使用该范围内的质数。如果你感兴趣，我会留给你去弄清楚为什么这样可以，让我们看看代码如何改变：

```py
# primes.py
from math import sqrt, ceil

def get_primes(n):
    """Calculate a list of primes up to n (included). """
    primelist = []
    for candidate in range(2, n + 1):
        is_prime = True
        root = ceil(sqrt(candidate))  # division limit
        for prime in primelist:  # we try only the primes
            if prime > root:  # no need to check any further
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primelist.append(candidate)
    return primelist
```

代码和上一章的一样。我们改变了除法算法，所以我们只使用先前计算出的质数来测试可除性，并且一旦测试除数大于候选数的平方根，我们就停止了。我们使用`primelist`结果列表来获取除法的质数。我们使用一个花哨的公式来计算根值，即候选数的根的天花板的整数值。虽然一个简单的`int(k ** 0.5) + 1`也可以满足我们的目的，但我选择的公式更简洁，需要我使用一些导入，我想向你展示。查看`math`模块中的函数，它们非常有趣！

# 文档化你的代码

我是一个不需要文档的代码的忠实粉丝。当你正确编程，选择正确的名称并注意细节时，你的代码应该是不言自明的，不需要文档。有时注释是非常有用的，文档也是如此。你可以在*PEP 257 - Docstring conventions*中找到 Python 文档的指南（[`www.python.org/dev/peps/pep-0257/`](https://www.python.org/dev/peps/pep-0257/)），但我会在这里向你展示基础知识。

Python 使用字符串进行文档化，这些字符串被称为**docstrings**。任何对象都可以被文档化，你可以使用单行或多行 docstrings。单行的非常简单。它们不应该为函数提供另一个签名，而是清楚地说明其目的：

```py
# docstrings.py
def square(n):
    """Return the square of a number n. """
    return n ** 2

def get_username(userid):
    """Return the username of a user given their id. """
    return db.get(user_id=userid).username
```

使用三个双引号的字符串允许您以后轻松扩展。使用以句点结尾的句子，并且不要在前后留下空行。

多行注释的结构方式类似。应该有一个简短的一行说明对象大意的描述，然后是更详细的描述。例如，我已经使用 Sphinx 符号记录了一个虚构的`connect`函数，在下面的示例中：

```py
def connect(host, port, user, password):
    """Connect to a database.

    Connect to a PostgreSQL database directly, using the given
    parameters.

    :param host: The host IP.
    :param port: The desired port.
    :param user: The connection username.
    :param password: The connection password.
    :return: The connection object.
    """
    # body of the function here...
    return connection
```

**Sphinx** 可能是创建 Python 文档最广泛使用的工具。事实上，官方 Python 文档就是用它编写的。值得花一些时间去了解它。

# 导入对象

现在您已经对函数有了很多了解，让我们看看如何使用它们。编写函数的整个目的是能够以后重用它们，在 Python 中，这意味着将它们导入到需要它们的命名空间中。有许多不同的方法可以将对象导入命名空间，但最常见的是`import module_name`和`from module_name import function_name`。当然，这些都是相当简单的例子，但请暂时忍耐。

`import module_name` 形式会找到`module_name`模块，并在执行`import`语句的本地命名空间中为其定义一个名称。`from module_name import identifier` 形式比这略微复杂一些，但基本上做的是相同的事情。它找到`module_name`并搜索属性（或子模块），并在本地命名空间中存储对`identifier`的引用。

两种形式都可以使用`as`子句更改导入对象的名称：

```py
from mymodule import myfunc as better_named_func 
```

只是为了让您了解导入的样子，这里有一个来自我的一个项目的测试模块的示例（请注意，导入块之间的空行遵循 PEP 8 的指南：标准库、第三方库和本地代码）：

```py
from datetime import datetime, timezone  # two imports on the same line
from unittest.mock import patch  # single import

import pytest  # third party library

from core.models import (  # multiline import
    Exam,
    Exercise,
    Solution,
)
```

当您有一个从项目根目录开始的文件结构时，您可以使用点符号来获取您想要导入到当前命名空间的对象，无论是包、模块、类、函数还是其他任何东西。`from module import`语法还允许使用一个通配符子句，`from module import *`，有时用于一次性将模块中的所有名称导入当前命名空间，但出于多种原因，如性能和潜在的静默屏蔽其他名称的风险，这是不被赞成的。您可以在官方 Python 文档中阅读关于导入的所有内容，但在我们离开这个主题之前，让我给您一个更好的例子。

假设您在`lib`文件夹中的模块`funcdef.py`中定义了一对函数：`square(n)`和`cube(n)`。您希望在与`lib`文件夹处于相同级别的几个模块中使用它们，这些模块称为`func_import.py`和`func_from.py`。显示该项目的树结构会产生以下内容：

```py
├── func_from.py
├── func_import.py
├── lib
 ├── funcdef.py
 └── __init__.py

```

在我展示每个模块的代码之前，请记住，为了告诉 Python 它实际上是一个包，我们需要在其中放置一个`__init__.py`模块。

关于`__init__.py`文件有两件事需要注意。首先，它是一个完整的 Python 模块，因此您可以像对待任何其他模块一样将代码放入其中。其次，从 Python 3.3 开始，不再需要它的存在来使文件夹被解释为 Python 包。

代码如下：

```py
# funcdef.py
def square(n): 
    return n ** 2 
def cube(n): 
    return n ** 3 

# func_import.py import lib.funcdef 
print(lib.funcdef.square(10)) 
print(lib.funcdef.cube(10)) 

# func_from.py
from lib.funcdef import square, cube 
print(square(10)) 
print(cube(10)) 
```

这两个文件在执行时都会打印`100`和`1000`。您可以看到我们如何根据当前范围中导入的内容以及导入的方式和内容来访问`square`和`cube`函数的不同方式。

# 相对导入

到目前为止，我们所见过的导入被称为**绝对**导入，即它们定义了我们要导入的模块的整个路径，或者我们要从中导入对象的模块。在 Python 中还有另一种导入对象的方式，称为**相对导入**。在我们想要重新排列大型包的结构而不必编辑子包时，或者当我们想要使包内的模块能够自我导入时，这种方式非常有帮助。相对导入是通过在模块前面添加与我们需要回溯的文件夹数量相同数量的前导点来完成的，以便找到我们正在搜索的内容。简而言之，它就是这样的。

```py
from .mymodule import myfunc 
```

有关相对导入的完整解释，请参阅 PEP 328 ([`www.python.org/dev/peps/pep-0328/`](https://www.python.org/dev/peps/pep-0328/))。

# 总结

在本章中，我们探讨了函数的世界。它们非常重要，从现在开始，我们基本上会在任何地方使用它们。我们谈到了使用它们的主要原因，其中最重要的是代码重用和实现隐藏。

我们看到函数对象就像一个接受可选输入并产生输出的盒子。我们可以以许多不同的方式向函数提供输入值，使用位置参数和关键字参数，并对两种类型都使用变量语法。

现在你应该知道如何编写一个函数，对它进行文档化，将它导入到你的代码中，并调用它。

在下一章中，我们将看到如何处理文件以及如何以多种不同的方式和格式持久化数据。
