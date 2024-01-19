# 第八章。功能和反应式编程特性

在本章中，我们将研究以下食谱：

+   使用 yield 语句编写生成器函数

+   使用堆叠的生成器表达式

+   将转换应用于集合

+   选择子集-三种过滤方式

+   总结集合-如何减少

+   组合映射和减少转换

+   实现“存在”处理

+   创建部分函数

+   使用不可变数据结构简化复杂算法

+   使用 yield from 语句编写递归生成器函数

# 介绍

**函数式编程**的理念是专注于编写执行所需数据转换的小型、表达力强的函数。组合函数通常可以创建比长串过程语句或复杂、有状态对象的方法更简洁和表达力更强的代码。Python 允许这三种编程方式。

传统数学将许多东西定义为函数。多个函数组合起来，从先前的转换中构建出复杂的结果。例如，我们可能有两个函数*f(x)*和*g(y)*，需要组合起来创建一个有用的结果：

*y = f(x)*

*z = g(y)*

理想情况下，我们可以从这两个函数创建一个复合函数：

*z* = (*g* ∘ *f*)(*x*)

使用复合函数（*g* ∘ *f*）可以帮助澄清程序的工作方式。它允许我们将许多小细节组合成更大的知识块。

由于编程经常涉及数据集合，我们经常会将函数应用于整个集合。这与数学中的**集合构建器**或**集合理解**的概念非常契合。

有三种常见的模式可以将一个函数应用于一组数据：

+   **映射**：这将一个函数应用于集合中的所有元素{*M*(*x*): *x*∈*C*}。我们将一些函数*M*应用于较大集合*C*的每个项目*x*。

+   **过滤**：这使用一个函数从集合中选择元素。{*x*：*c*∈*C* **if** *F*(*x*)}。我们使用一个函数*F*来确定是否从较大的集合*C*中传递或拒绝项目*x*。

+   **减少**：这是对集合进行总结。细节各异，但最常见的减少之一是创建集合*C*中所有项目*x*的总和：![Introduction](img/Image00017.jpg)。

我们经常将这些模式结合起来创建更复杂的应用程序。这里重要的是小函数，如*M(x)*和*F(x)*，通过映射和过滤等高阶函数进行组合。即使各个部分非常简单，组合操作也可以变得复杂。

**反应式编程**的理念是在输入可用或更改时评估处理规则。这符合惰性编程的理念。当我们定义类定义的惰性属性时，我们创建了反应式程序。

反应式编程与函数式编程契合，因为可能需要多个转换来对输入值的变化做出反应。通常，这最清晰地表达为组合或堆叠成响应变化的复合函数。在第六章*类和对象的基础*中查看*使用惰性属性*食谱，了解一些反应式类设计的示例。

# 使用 yield 语句编写生成器函数

我们看过的大多数食谱都是为了与单个集合中的所有项目一起使用而设计的。这种方法是使用`for`语句来遍历集合中的每个项目，要么将值映射到新项目，要么将集合减少到某种摘要值。

从集合中产生单个结果是处理集合的两种方式之一。另一种方式是产生增量结果，而不是单个结果。

这种方法在我们无法将整个集合放入内存的情况下非常有帮助。例如，分析庞大的网络日志文件最好是分批进行，而不是创建一个内存集合。

有没有办法将集合结构与处理函数分离？我们是否可以在每个单独的项目可用时立即产生处理结果？

## 准备工作

我们将查看一些具有日期时间字符串值的网络日志数据。我们需要解析这些数据以创建适当的`datetime`对象。为了保持本食谱的重点，我们将使用 Flask 生成的简化日志。

条目最初是这样的文本行：

```py
 **[2016-05-08 11:08:18,651] INFO in ch09_r09: Sample Message One** 

 **[2016-05-08 11:08:18,651] DEBUG in ch09_r09: Debugging** 

 **[2016-05-08 11:08:18,652] WARNING in ch09_r09: Something might have gone wrong** 

```

我们已经看到了在第七章的*使用更复杂的结构——列表的映射*食谱中处理这种日志的其他示例，*更高级的类设计*。使用第一章的*使用正则表达式进行字符串解析*食谱中的 REs，*数字、字符串和元组*，我们可以将每行分解为以下行集合：

```py
 **>>> data = [ 
...    ('2016-04-24 11:05:01,462', 'INFO', 'module1', 'Sample Message One'), 
...    ('2016-04-24 11:06:02,624', 'DEBUG', 'module2', 'Debugging'), 
...    ('2016-04-24 11:07:03,246', 'WARNING', 'module1', 'Something might have gone wrong') 
... ]** 

```

我们不能使用普通的字符串解析将复杂的日期时间戳转换为更有用的形式。但是，我们可以编写一个生成器函数，它可以处理日志的每一行，产生一个更有用的中间数据结构。

生成器函数是使用`yield`语句的函数。当一个函数有一个 yield 时，它会逐渐构建结果，以一种可以被客户端消耗的方式产生每个单独的值。消费者可能是一个`for`语句，也可能是另一个需要一系列值的函数。

## 如何做到...

1.  这需要`datetime`模块：

```py
        import datetime 

```

1.  定义一个处理源集合的函数：

```py
        def parse_date_iter(source): 

```

我们在后缀`_iter`中包含了这个函数将是一个可迭代对象而不是一个简单集合的提醒。

1.  包括一个`for`语句，访问源集合中的每个项目：

```py
        for item in source: 

```

1.  `for`语句的主体可以将项目映射到一个新项目：

```py
        date = datetime.datetime.strptime( 
            item[0], 
            "%Y-%m-%d %H:%M:%S,%f") 
        new_item = (date,)+item[1:] 

```

在这种情况下，我们将一个字段从字符串映射到`datetime`对象。变量`date`是从`item[0]`中的字符串构建的。

然后，我们将日志消息的三元组映射到一个新的元组，用正确的`datetime`对象替换日期字符串。由于项目的值是一个元组，我们创建了一个带有`(date,)`的单例元组，然后将其与`item[1:]`元组连接起来。

1.  使用`yield`语句产生新项目：

```py
        yield new_item 

```

整个结构看起来是这样的，正确缩进：

```py
    import datetime 
    def parse_date_iter(source): 
        for item in source: 
            date = datetime.datetime.strptime( 
                item[0], 
                "%Y-%m-%d %H:%M:%S,%f") 
            new_item = (date,)+item[1:] 
            yield new_item 

```

`parse_date_iter()`函数期望一个可迭代的输入对象。集合是可迭代对象的一个例子。然而更重要的是，其他生成器也是可迭代的。我们可以利用这一点构建处理来自其他生成器的数据的生成器堆栈。

这个函数不会创建一个集合。它会产生每个项目，以便可以单独处理这些项目。源集合会被分成小块进行处理，从而可以处理大量的数据。在一些示例中，数据将从内存集合开始。在后续的示例中，我们将处理来自外部文件的数据——处理外部文件最能从这种技术中获益。

以下是我们如何使用这个函数的方法：

```py
 **>>> from pprint import pprint 
>>> from ch08_r01 import parse_date_iter 
>>> for item in parse_date_iter(data): 
...     pprint(item) 
(datetime.datetime(2016, 4, 24, 11, 5, 1, 462000), 
 'INFO', 
 'module1', 
 'Sample Message One') 
(datetime.datetime(2016, 4, 24, 11, 6, 2, 624000), 
 'DEBUG', 
 'module2', 
 'Debugging') 
(datetime.datetime(2016, 4, 24, 11, 7, 3, 246000), 
 'WARNING', 
 'module1', 
 'Something might have gone wrong')** 

```

我们使用`for`语句遍历`parse_date_iter()`函数的结果，一次处理一个项目。我们使用`pprint()`函数显示每个项目。

我们也可以使用类似这样的方法将项目收集到一个适当的列表中：

```py
 **>>> details = list(parse_date_iter(data))** 

```

在这个例子中，`list()`函数消耗了`parse_date_iter()`函数产生的所有项目。使用`list()`或`for`语句来消耗生成器中的所有项目是至关重要的。生成器是一个相对被动的结构——直到需要数据时，它不会做任何工作。

如果我们不主动消耗数据，我们会看到类似这样的情况：

```py
 **>>> parse_date_iter(data) 
<generator object parse_date_iter at 0x10167ddb0>** 

```

`parse_date_iter()`函数的值是一个生成器。它不是一个项目的集合，而是一个能够按需生成项目的函数。

## 工作原理...

编写生成器函数可以改变我们对算法的理解方式。有两种常见的模式：映射和归约。映射将每个项目转换为一个新项目，可能计算一些派生值。归约从源集合中累积一个摘要，比如总和、平均值、方差或哈希。这些可以分解为逐项转换或过滤，与处理集合的整体循环分开。

Python 有一个复杂的构造叫做**迭代器**，它是生成器和集合的核心。迭代器会从集合中提供每个值，同时进行所有必要的内部记录以维护进程的状态。生成器函数的行为就像一个迭代器-它提供一系列的值并维护自己的内部状态。

考虑下面这段常见的 Python 代码：

```py
    for i in some_collection: 
        process(i) 

```

在幕后，类似以下的事情正在发生：

```py
    the_iterator = iter(some_collection) 
    try: 
        while True: 
            i = next(the_iterator) 
            process(i) 
    except StopIteration: 
        pass 

```

Python 对集合上的`iter()`函数进行评估，以创建该集合的迭代器对象。迭代器绑定到集合并维护一些内部状态信息。代码在迭代器上使用`next()`来获取每个值。当没有更多的值时，迭代器会引发`StopIteration`异常。

Python 的每个集合都可以产生一个迭代器。`Sequence`或`Set`产生的迭代器会访问集合中的每个项。`Mapping`产生的迭代器会访问映射的每个键。我们可以使用映射的`values()`方法来迭代值而不是键。我们可以使用映射的`items()`方法来访问一个`(key, value)`两元组的序列。`file`的迭代器会访问文件中的每一行。

迭代器的概念也可以应用到函数上。带有`yield`语句的函数被称为**生成器函数**。它符合迭代器的模板。为了实现这一点，生成器在响应`iter()`函数时返回自身。在响应`next()`函数时，它会产生下一个值。

当我们对集合或生成器函数应用`list()`时，与`for`语句使用的相同的基本机制会得到各个值。`iter()`和`next()`函数被`list()`用来获取这些项。然后这些项被转换成一个序列。

评估生成器函数的`next()`是有趣的。生成器函数会被评估，直到它达到一个`yield`语句。这个值就是`next()`的结果。每次评估`next()`时，函数会在`yield`语句之后恢复处理，并继续到下一个`yield`语句。

这里有一个产生两个对象的小函数：

```py
 **>>> def gen_func(): 
...     print("pre-yield") 
...     yield 1 
...     print("post-yield") 
...     yield 2** 

```

当我们评估`next()`函数时会发生什么。在生成器上，这个函数会产生：

```py
 **>>> y = gen_func() 
>>> next(y) 
pre-yield 
1 
>>> next(y) 
post-yield 
2** 

```

第一次评估`next()`时，第一个`print()`函数被评估，然后`yield`语句产生一个值。函数的处理被暂停，然后出现`>>>`提示符。第二次评估`next()`函数时，两个`yield`语句之间的语句被评估。函数再次被暂停，然后会显示一个`>>>`提示符。

接下来会发生什么？我们已经没有`yield`语句了：

```py
 **>>> next(y)  
Traceback (most recent call last): 
  File "<pyshell...>", line 1, in <module> 
    next(y) 
StopIteration** 

```

在生成器函数的末尾会引发`StopIteration`异常。

## 还有更多...

生成器函数的核心价值在于能够将复杂的处理分解为两部分：

+   要应用的转换或过滤

+   要处理的源数据集

这是使用生成器来过滤数据的一个例子。在这种情况下，我们将过滤输入值，只保留质数，拒绝所有合数。

我们可以将处理写成一个 Python 函数，像这样：

```py
    def primeset(source): 
        for i in source: 
            if prime(i): 
                yield prime 

```

对于源中的每个值，我们将评估`prime()`函数。如果结果为`true`，我们将产生源值。如果结果为`false`，源值将被拒绝。我们可以像这样使用`primeset()`：

```py
    p_10 = set(primeset(range(2,2000000))) 

```

`primeset()`函数将从源集合中产生单个素数值。源集合将是范围在 2 到 200 万之间的整数。结果是从提供的值构建的`set`对象。

这里唯一缺少的是`prime()`函数，用于确定一个数字是否为素数。我们将把这留给读者作为练习。

从数学上讲，常见的是使用*集合生成器*或*集合推导*符号来指定从另一个集合构建一个集合的规则。

我们可能会看到类似这样的东西：

*P[10]* = {*i*：*i* ∈ *ℕ* ∧ 2 ≤ 1 < 2,000,000 **if** *P*（*i*）}

这告诉我们*P[10]*是所有数字*i*的集合，在自然数集*ℕ*中，并且在 2 到 200 万之间，如果*P(i)*为`true`。这定义了一个构建集合的规则。

我们也可以用 Python 写出这个：

```py
    p_10 = {i for i in range(2,2000000) if prime(i)} 

```

这是素数子集的 Python 表示。从数学抽象中略微重新排列了子句，但表达式的所有基本部分都存在。

当我们开始看这样的生成器表达式时，我们会发现很多编程都符合一些常见的整体模式：

+   **Map**：*m*（*x*）：*x* ∈ *S*变为`(m(x) for x in S)`。

+   **Filter**：*x*：*x* ∈ *S* **if** *f*（*x*）变为`(x for x in S if f(x))`。

+   **Reduce**：这有点复杂，但常见的缩减包括求和和计数。![更多内容...](img/Image00018.jpg)是`sum(x for x in S)`。其他常见的缩减包括查找一组数据的最大值或最小值。

我们也可以使用`yield`语句编写这些不同的高级函数。以下是通用映射的定义：

```py
    def map(m, S): 
        for s in S: 
            yield m(s) 

```

此函数将某个其他函数`m()`应用于源集合`S`中的每个数据元素。映射函数的结果作为结果值的序列被产生。

我们可以为通用的`filter`函数编写类似的定义：

```py
    def filter(f, S): 
        for s in S: 
            if f(s): 
                yield s 

```

与通用映射一样，我们将函数`f()`应用于源集合`S`中的每个元素。函数为`true`的地方，值被产生。函数为`false`的地方，值被拒绝。

我们可以像这样使用它来创建一个素数集：

```py
    p_10 = set(filter(prime, range(2,2000000))) 

```

这将应用`prime()`函数到数据源范围。请注意，我们只写`prime`，不带`()`字符，因为我们是在命名函数，而不是在评估它。每个单独的值将由`prime()`函数检查。通过的值将被产生以组装成最终集合。那些是合数的值将被拒绝，不会出现在最终集合中。

## 另请参阅

+   在*使用堆叠的生成器表达式*的示例中，我们将结合生成器函数，从简单组件构建复杂的处理堆栈。

+   在*对集合应用转换*的示例中，我们将看到内置的`map()`函数如何被用于从简单函数和可迭代的数据源创建复杂的处理。

+   在*选择子集-三种过滤方式*的示例中，我们将看到内置的`filter()`函数也可以用于从简单函数和可迭代的数据源构建复杂的处理。

+   有关小于 200 万的素数的具有挑战性的问题，请参阅[`projecteuler.net/problem=10`](https://projecteuler.net/problem=10)。问题的部分似乎是显而易见的。然而，测试所有这些数字是否为素数可能很困难。

# 使用堆叠的生成器表达式

在*使用 yield 语句编写生成器函数*的示例中，我们创建了一个简单的生成器函数，对数据进行了单一的转换。实际上，我们经常有几个函数，我们希望将其应用于传入的数据。

我们如何*堆叠*或组合多个生成器函数以创建一个复合函数？

## 准备工作

我们有一个用于记录大帆船燃油消耗的电子表格。它的行看起来像这样：

| **日期** | **启动引擎** | **燃油高度** |
| --- | --- | --- |
|  | **关闭引擎** | **燃油高度** |
|  | **其他说明** |  |
| 10/25/2013 | 08:24 | 29 |
|  | 13:15 | 27 |
|  | 平静的海域 - 锚在所罗门岛 |  |
| 10/26/2013 | 09:12 | 27 |
|  | 18:25 | 22 |
|  | 波涛汹涌 - 锚在杰克逊溪 |  |

有关这些数据的更多背景信息，请参阅第四章的*对列表进行切片和切块*，*内置数据结构 - 列表、集合、字典*。

作为一个侧边栏，我们可以这样获取数据。我们将在第九章的*使用 csv 模块读取分隔文件*中详细讨论这个问题，*输入/输出、物理格式和逻辑布局*：

```py
 **>>> from pathlib import Path 
>>> import csv 
>>> with Path('code/fuel.csv').open() as source_file: 
...    reader = csv.reader(source_file) 
...    log_rows = list(reader) 
>>> log_rows[0] 
['date', 'engine on', 'fuel height'] 
>>> log_rows[-1] 
['', "choppy -- anchor in jackson's creek", '']** 

```

我们已经使用`csv`模块来读取日志详情。`csv.reader()`是一个可迭代对象。为了将项目收集到一个单一列表中，我们对生成器函数应用了`list()`函数。我们打印了列表中的第一个和最后一个项目，以确认我们确实有一个列表的列表结构。

我们想对这个列表的列表应用两个转换：

+   将日期和两个时间转换为两个日期时间值

+   将三行合并成一行，以便对数据进行简单的组织

如果我们创建一对有用的生成器函数，我们可以有这样的软件：

```py
    total_time = datetime.timedelta(0) 
    total_fuel = 0 
    for row in date_conversion(row_merge(source_data)): 
        total_time += row['end_time']-row['start_time'] 
        total_fuel += row['end_fuel']-row['start_fuel'] 

```

组合的生成器函数`date_conversion(row_merge(...))`将产生一系列单行，其中包含起始信息、结束信息和注释。这种结构可以很容易地总结或分析，以创建简单的统计相关性和趋势。

## 如何做到这一点...

1.  定义一个初始的减少操作，将行组合在一起。我们有几种方法可以解决这个问题。一种方法是总是将三行组合在一起。

另一种方法是注意到第零列在组的开头有数据；在组的下两行为空。这给了我们一个稍微更一般的方法来创建行的组。这是一种**头尾合并**算法。我们将收集数据，并在到达下一个组的头部时每次产生数据：

```py
        def row_merge(source_iter): 
            group = [] 
            for row in source_iter: 
                if len(row[0]) != 0: 
                    if group: 
                        yield group 
                    group = row.copy() 
                else: 
                    group.extend(row) 
            if group: 
                yield group 

```

这个算法使用`len(row[0])`来确定这是一个组的头部还是组的尾部中的一行。在头部行的情况下，任何先前的组都会被产生。在那之后被消耗后，`group`集合的值将被重置为头部行的列数据。

组的尾部行简单地附加到`group`集合上。当数据耗尽时，`group`变量中通常会有一个最终组。如果根本没有数据，那么`group`的最终值也将是一个长度为零的列表，应该被忽略。

我们稍后会讨论`copy()`方法。这是必不可少的，因为我们正在处理一个列表的列表数据结构，而列表是可变对象。我们可以编写处理改变数据结构的处理，使得一些处理难以解释。

1.  定义将在合并后的数据上执行的各种映射操作。这些应用于原始行中的数据。我们将使用单独的函数来转换两个时间列，并将时间与日期列合并：

```py
        import datetime 
        def start_datetime(row): 
            travel_date = datetime.datetime.strptime(row[0], "%m/%d/%y").date() 
            start_time = datetime.datetime.strptime(row[1], "%I:%M:%S %p").time() 
            start_datetime = datetime.datetime.combine(travel_date, start_time) 
            new_row = row+[start_datetime] 
            return new_row 

        def end_datetime(row): 
            travel_date = datetime.datetime.strptime(row[0], "%m/%d/%y").date() 
            end_time = datetime.datetime.strptime(row[4], "%I:%M:%S %p").time() 
            end_datetime = datetime.datetime.combine(travel_date, end_time) 
            new_row = row+[end_datetime] 
            return new_row 

```

我们将把第零列中的日期与第一列中的时间结合起来，创建一个起始`datetime`对象。同样，我们将把第零列中的日期与第四列中的时间结合起来，创建一个结束`datetime`对象。

这两个函数有很多重叠之处，可以重构为一个带有列号作为参数值的单个函数。然而，目前我们的目标是编写一些简单有效的东西。效率的重构可以稍后进行。

1.  定义适用于派生数据的映射操作。第八和第九列包含日期时间戳：

```py
        for starting and ending.def duration(row): 
            travel_hours = round((row[10]-row[9]).total_seconds()/60/60, 1) 
            new_row = row+[travel_hours] 
            return new_row 

```

我们使用`start_datetime`和`end_datetime`创建的值作为输入。我们计算了时间差，这提供了以秒为单位的结果。我们将秒转换为小时，这是这组数据更有用的时间单位。

1.  合并任何需要拒绝或排除坏数据的过滤器。在这种情况下，我们必须排除一个标题行：

```py
        def skip_header_date(rows): 
            for row in rows: 
                if row[0] == 'date': 
                    continue 
                yield row 

```

这个函数将拒绝任何第一列中有`date`的行。`continue`语句恢复`for`语句，跳过体中的所有其他语句；它跳过`yield`语句。所有其他行将通过这个过程。输入是一个可迭代对象，这个生成器将产生没有以任何方式转换的行。

1.  将操作组合起来。我们可以编写一系列生成器表达式，也可以使用内置的`map()`函数。以下是使用生成器表达式的示例：

```py
        def date_conversion(source): 
            tail_gen = skip_header_date(source) 
            start_gen = (start_datetime(row) for row in tail_gen) 
            end_gen = (end_datetime(row) for row in start_gen) 
            duration_gen = (duration(row) for row in end_gen) 
            return duration_gen 

```

这个操作由一系列转换组成。每个转换对原始数据集中的一个值进行小的转换。添加或更改操作相对简单，因为每个操作都是独立定义的：

+   `tail_gen`生成器在跳过源的第一行后产生行

+   `start_gen`生成器将一个`datetime`对象附加到每一行的末尾，起始时间是从字符串构建到源列中的

+   `end_gen`生成器将一个`datetime`对象附加到每一行的末尾，结束时间是从字符串构建到源列中的

+   `duration_gen`生成器将一个`float`对象附加到每个腿的持续时间

整体`date_conversion()`函数的输出是一个生成器。可以使用`for`语句消耗它，也可以从项目构建一个`list`。

## 工作原理...

当我们编写一个生成器函数时，参数值可以是一个集合，也可以是另一种可迭代对象。由于生成器函数是可迭代的，因此可以创建一种生成器函数的*管道*。

每个函数可以包含一个小的转换，改变输入的一个特征以创建输出。然后我们将每个小的转换包装在生成器表达式中。因为每个转换都相对独立于其他转换，所以我们可以对其中一个进行更改而不破坏整个处理流水线。

处理是逐步进行的。每个函数都会被评估，直到产生一个单一的值。考虑这个陈述：

```py
    for row in date_conversion(row_merge(data)): 
        print(row[11]) 

```

我们定义了几个生成器的组合。这个组合使用了各种技术：

+   `row_merge()`函数是一个生成器，将产生数据行。为了产生一行，它将从源中读取四行，组装成一个合并的行，并产生它。每次需要另一行时，它将再读取三行输入来组装输出行。

+   `date_conversion()`函数是由多个生成器构建的复杂生成器。

+   `skip_header_date()`旨在产生一个单一的值。有时它必须从源迭代器中读取两个值。如果输入行的第零列有`date`，则跳过该行。在这种情况下，它将读取第二个值，从`row_merge()`获取另一行；而`row_merge()`必须再读取三行输入来产生一个合并的输出行。我们将生成器分配给`tail_gen`变量。

+   `start_gen`、`end_gen`和`duration_gen`生成器表达式将对其输入的每一行应用相对简单的函数，例如`start_datetime()`和`end_datetime()`，产生具有更有用数据的行。

示例中显示的最终`for`语句将通过反复评估`next()`函数来从`date_conversion()`迭代器中收集值。以下是创建所需结果的逐步视图。请注意，这在一个非常小的数据量上运行——每个步骤都会做出一个小的改变：

1.  `date_conversion()` 函数的结果是 `duration_gen` 对象。为了返回一个值，它需要来自其源 `end_gen` 的一行。一旦有了数据，它就可以应用 `duration()` 函数并产生该行。

1.  `end_gen` 表达式需要来自其源 `start_gen` 的一行。然后它可以应用 `end_datetime()` 函数并产生该行。

1.  `start_gen` 表达式需要来自其源 `tail_gen` 的一行。然后它可以应用 `start_datetime()` 函数并产生该行。

1.  `tail_gen` 表达式只是生成器 `skip_header_date()` 。这个函数将从其源中读取所需的行，直到找到一行，其中第零列不是列标题 `date`。它产生一个非日期行。其源是 `row_merge()` 函数的输出。

1.  `row_merge()` 函数将从其源中读取多行，直到可以组装符合所需模式的行集合。它将产生一个组合行，该行在第零列中有一些文本，后面是没有文本的行。其源是原始数据的列表集合。

1.  行集合将由 `row_merge()` 函数内的 `for` 语句处理。这个处理将隐式地为集合创建一个迭代器，以便 `row_merge()` 函数的主体根据需要产生每个单独的行。

数据的每个单独行将通过这些步骤的管道。管道的某些阶段将消耗多个源行以产生单个结果行，重组数据。其他阶段消耗单个值。

这个例子依赖于将项目连接成一个长序列的值。项目由位置标识。对管道中阶段顺序的小改动将改变项目的位置。有许多方法可以改进这一点，我们将在接下来看一下。

这个核心是只处理单独的行。如果源是一个庞大的数据集合，处理可以非常快速。这种技术允许一个小的 Python 程序快速而简单地处理大量的数据。

## 还有更多...

实际上，一组相互关联的生成器是一种复合函数。我们可能有几个函数，像这样分别定义：

*y = f(x)*

*z = g(y)*

我们可以通过将第一个函数的结果应用到第二个函数来将它们组合起来：

*z = g(f(x))*

随着函数数量的增加，这可能变得笨拙。当我们在多个地方使用这对函数时，我们违反了**不要重复自己**（**DRY**）原则。拥有多个这种复杂表达式的副本并不理想。

我们希望有一种方法来创建一个复合函数——类似于这样：

*z* = ( *g* ∘ *f* )( *x* )

在这里，我们定义了一个新函数（*g* ∘ *f*），将两个原始函数组合成一个新的、单一的复合函数。我们现在可以修改这个复合函数以添加或更改功能。

这个概念推动了复合 `date_conversion()` 函数的定义。这个函数由许多函数组成，每个函数都可以应用于集合的项。如果我们需要进行更改，我们可以轻松地编写更多简单的函数并将它们放入 `date_conversion()` 函数定义的管道中。

我们可以看到管道中的函数之间存在一些细微差异。我们有一些类型转换。然而，持续时间计算并不真正是一种类型转换。它是一种基于日期转换结果的独立计算。如果我们想要计算每小时的燃料使用量，我们需要添加几个计算。这些额外的摘要都不是日期转换的正确部分。

我们真的应该将高级 `data_conversion()` 分成两部分。我们应该编写另一个函数来进行持续时间和燃料使用计算，命名为 `fuel_use()`。然后这个其他函数可以包装 `date_conversion()`。

我们可能会朝着这样的目标努力：

```py
    for row in fuel_use(date_conversion(row_merge(data))): 
        print(row[11]) 

```

现在我们有一个非常复杂的计算，它由许多非常小的（几乎）完全独立的部分定义。我们可以修改一个部分而不必深入思考其他部分的工作方式。

### 命名空间而不是列表

一个重要的改变是停止避免使用简单的列表来存储数据值。对`row[10]`进行计算可能是一场潜在的灾难。我们应该适当地将输入数据转换为某种命名空间。

可以使用`namedtuple`。我们将在*Simplifying complex algorithms with immutable data structures*食谱中看到。

在某些方面，`SimpleNamespace`可以进一步简化这个处理过程。`SimpleNamespace`是一个可变对象，可以被更新。改变对象并不总是一个好主意。它的优点是简单，但对于可变对象的状态变化编写测试可能会稍微困难一些。

例如`make_namespace()`这样的函数可以提供一组名称而不是位置。这是一个必须在行合并后但在任何其他处理之前使用的生成器：

```py
    from types import SimpleNamespace 

    def make_namespace(merge_iter): 
        for row in merge_iter: 
            ns = SimpleNamespace( 
                date = row[0], 
                start_time = row[1], 
                start_fuel_height = row[2], 
                end_time = row[4], 
                end_fuel_height = row[5], 
                other_notes = row[7] 
            ) 
            yield ns 

```

这将产生一个允许我们写`row.date`而不是`row[0]`的对象。当然，这将改变其他函数的定义，包括`start_datetime()`、`end_datetime()`和`duration()`。

这些函数中的每一个都可以发出一个新的`SimpleNamespace`对象，而不是更新表示每一行的值列表。然后我们可以编写以下样式的函数：

```py
    def duration(row_ns): 
        travel_time = row_ns.end_timestamp - row_ns.start_timestamp 
        travel_hours = round(travel_time.total_seconds()/60/60, 1) 
        return SimpleNamespace( 
            **vars(row_ns), 
            travel_hours=travel_hours 
        )

```

这个函数处理行作为`SimpleNamespace`对象，而不是`list`对象。列具有清晰而有意义的名称，如`row_ns.end_timestamp`，而不是晦涩的`row[10]`。

构建新的`SimpleNamespace`的三部曲如下：

1.  使用`vars()`函数提取`SimpleNamespace`实例内部的字典。

1.  使用`**vars(row_ns)`对象基于旧命名空间构建一个新的命名空间。

1.  任何额外的关键字参数，如`travel_hours = travel_hours`，都提供了将加载新对象的额外值。

另一种选择是更新命名空间并返回更新后的对象：

```py
    def duration(row_ns): 
        travel_time = row_ns.end_timestamp - row_ns.start_timestamp 
        row_ns.travel_hours = round(travel_time.total_seconds()/60/60, 1) 
        return row_ns 

```

这样做的优点是稍微简单。缺点是有时会让有状态的对象变得混乱。在修改算法时，可能会失败地按正确顺序设置属性，以便懒惰（或反应性）编程能够正常运行。

尽管有状态的对象很常见，但它们应该始终被视为两种选择之一。不可变的`namedtuple`可能比可变的`SimpleNamespace`更好。

## 另请参阅

+   在*Writing generator functions with the yield statement*食谱中，我们介绍了生成器函数

+   在第四章的*Slicing and dicing a list*食谱中，了解有关燃料消耗数据集的更多信息

+   在*Combining map and reduce transformations*食谱中，还有另一种组合操作的方法

# 对集合应用转换

在*Writing generator functions with the yield statement*食谱中，我们看到了编写生成器函数的例子。我们看到的例子结合了两个元素：转换和数据源。它们通常看起来像这样：

```py
    for item in source: 
        new_item = some transformation of item 
        yield new_item 

```

编写生成器函数的这个模板并不是必需的。它只是一个常见的模式。在`for`语句中隐藏了一个转换过程。`for`语句在很大程度上是样板代码。我们可以重构这个代码，使转换函数明确且与`for`语句分离。

在*Using stacked generator expressions*食谱中，我们定义了一个`start_datetime()`函数，它从数据源集合的两个单独列中的字符串值计算出一个新的`datetime`对象。

我们可以在生成器函数的主体中使用这个函数，就像这样：

```py
    def start_gen(tail_gen): 
        for row in tail_gen: 
            new_row = start_datetime(row) 
            yield new_row 

```

这个函数将`start_datetime()`函数应用于数据源`tail_gen`中的每个项目。每个生成的行都被产生，以便另一个函数或`for`语句可以消耗它。

在*使用堆叠的生成器表达式*的示例中，我们看了另一种将这些转换函数应用于更大的数据集的方法。在这个例子中，我们使用了一个生成器表达式。代码看起来像这样：

```py
    start_gen = (start_datetime(row) for row in tail_gen) 

```

这将`start_datetime()`函数应用于数据源`tail_gen`中的每个项目。另一个函数或`for`语句可以消耗`start_gen`可迭代中可用的值。

完整的生成器函数和较短的生成器表达式本质上是相同的，只是语法略有不同。这两者都与数学上的*集合构建器*或*集合推导*的概念相似。我们可以用数学方式描述这个操作：

*s* = [ *S* ( *r* ): *r* ∈ *T* ]

在这个表达式中，*S*是`start_datetime()`函数，*T*是称为`tail_gen`的值序列。结果序列是*S(r)*的值，其中*r*的每个值是集合*T*的一个元素。

生成器函数和生成器表达式都有类似的样板代码。我们能简化这些吗？

## 准备好了...

我们将查看*使用带有 yield 语句的生成器函数*示例中的 web 日志数据。这里有一个`date`作为一个字符串，我们想要将其转换为一个合适的时间戳。

这是示例数据：

```py
 **>>> data = [ 
...    ('2016-04-24 11:05:01,462', 'INFO', 'module1', 'Sample Message One'), 
...    ('2016-04-24 11:06:02,624', 'DEBUG', 'module2', 'Debugging'), 
...    ('2016-04-24 11:07:03,246', 'WARNING', 'module1', 'Something might have gone wrong') 
... ]** 

```

我们可以编写一个这样的函数来转换数据：

```py
    import datetime 
    def parse_date_iter(source): 
        for item in source: 
            date = datetime.datetime.strptime( 
                item[0], 
                "%Y-%m-%d %H:%M:%S,%f") 
            new_item = (date,)+item[1:] 
            yield new_item 

```

这个函数将使用`for`语句检查源中的每个项目。第零列的值是一个`date`字符串，可以转换为一个合适的`datetime`对象。从`datetime`对象和从第一列开始的剩余项目构建一个新项目`new_item`。

因为函数使用`yield`语句产生结果，所以它是一个生成器函数。我们可以像这样使用它与`for`语句：

```py
    for row in parse_date_iter(data): 
        print(row[0], row[3]) 

```

这个语句将收集生成器函数产生的每个值，并打印两个选定的值。

`parse_date_iter()`函数将两个基本元素合并到一个函数中。大纲看起来像这样：

```py
    for item in source: 
        new_item = transformation(item) 
        yield new_item 

```

`for`和`yield`语句在很大程度上是样板代码。`transformation()`函数是这个非常有用和有趣的部分。

## 如何做...

1.  编写应用于数据单行的转换函数。这不是一个生成器，也不使用`yield`语句。它只是修改集合中的单个项目：

```py
        def parse_date(item): 
            date = datetime.datetime.strptime( 
                item[0], 
                "%Y-%m-%d %H:%M:%S,%f") 
            new_item = (date,)+item[1:] 
            return new_item 

```

这可以用三种方式：语句、表达式和`map()`函数。这是语句的显式`for...yield`模式：

```py
        for item in collection: 
            new_item = parse_date(item) 
            yield new_item 

```

这使用了一个`for`语句来使用孤立的`parse_date()`函数处理集合中的每个项目。第二个选择是一个生成器表达式，看起来像这样：

```py
        (parse_date(item) for item in data) 

```

这是一个生成器表达式，将`parse_date()`函数应用于每个项目。第三个选择是`map()`函数。

1.  使用`map()`函数将转换应用于源数据。

```py
        map(parse_date, data) 

```

我们提供函数名`parse_date`，在名称后面没有任何`()`。我们此时不应用函数。我们提供对象名给`map()`函数，以将`parse_date()`函数应用于可迭代的数据源`data`。

我们可以这样使用：

```py
        for row in map(parse_date, data): 
            print(row[0], row[3]) 

```

`map()`函数创建一个可迭代对象，将`parse_date()`函数应用于数据可迭代中的每个项目。它产生每个单独的项目。它使我们不必编写生成器表达式或生成器函数。

## 工作原理...

`map()`函数替换了一些常见的样板代码。我们可以想象定义看起来像这样：

```py
    def map(f, iterable): 
        for item in iterable: 
            yield f(item) 

```

或者，我们可以想象它看起来像这样：

```py
    def map(f, iterable): 
        return (f(item) for item in iterable) 

```

这两个定义总结了`map()`函数的核心特性。它是一个方便的简写，可以消除一些样板代码，用于将函数应用于可迭代的数据源。

## 还有更多...

在这个例子中，我们使用了`map()`函数来将一个接受单个参数的函数应用到单个可迭代对象的每个项目上。原来`map()`函数可以做的事情比这更多。

考虑这个函数：

```py
 **>>> def mul(a, b): 
...    return a*b** 

```

还有这两个数据来源：

```py
 **>>> list_1 = [2, 3, 5, 7] 
>>> list_2 = [11, 13, 17, 23]** 

```

我们可以将`mul()`函数应用于从每个数据源中提取的对：

```py
 **>>> list(map(mul, list_1, list_2)) 
[22, 39, 85, 161]** 

```

这使我们能够使用不同类型的运算符合并两个值序列。例如，我们可以构建一个行为类似于内置的`zip()`函数的映射。

这是一个映射：

```py
 **>>> def bundle(*args): 
...     return args 
>>> list(map(bundle, list_1, list_2)) 
[(2, 11), (3, 13), (5, 17), (7, 23)]** 

```

我们需要定义一个小的辅助函数，`bundle()`，它接受任意数量的参数，并将它们创建为一个元组。

这里是`zip`函数进行比较：

```py
 **>>> list(zip(list_1, list_2)) 
[(2, 11), (3, 13), (5, 17), (7, 23)]** 

```

## 另请参阅...

+   在*使用堆叠的生成器表达式*示例中，我们研究了堆叠生成器。我们从许多单独的映射操作中构建了一个复合函数，这些操作被编写为生成器函数。我们还在堆栈中包含了一个单一的过滤器。

# 选择子集-三种过滤方式

在*使用堆叠的生成器表达式*示例中，我们编写了一个生成器函数，它从一组数据中排除了一些行。我们定义了这样一个函数：

```py
    def skip_header_date(rows): 
        for row in rows: 
            if row[0] == 'date': 
                continue 
            yield row 

```

当条件为`true`——`row[0]`是`date`——`continue`语句将跳过`for`语句体中的其余语句。在这种情况下，只有一个语句`yield row`。

有两个条件：

+   `row[0] == 'date'`：`yield`语句被跳过；该行被拒绝进一步处理

+   `row[0] != 'date'`：`yield`语句意味着该行将被传递给消耗数据的函数或语句

在四行代码中，这似乎有点冗长。`for...if...yield`模式显然是样板文件，只有条件在这种结构中才是真正的材料。

我们可以更简洁地表达这个吗？

## 准备好...

我们有一个用于记录大帆船燃料消耗的电子表格。它的行看起来像这样：

| **日期** | **引擎开启** | **燃料高度** |
| --- | --- | --- |
|  | **关闭引擎** | **燃料高度** |
|  | **其他说明** |  |
| 10/25/2013 | 08:24 | 29 |
|  | 13:15 | 27 |
|  | 平静的海域 - 锚定所罗门岛 |  |
| 10/26/2013 | 09:12 | 27 |
|  | 18:25 | 22 |
|  | 波涛汹涌 - 锚定在杰克逊溪 |  |

有关这些数据的更多背景信息，请参阅*切片和切块列表*示例。

在*使用堆叠的生成器表达式*示例中，我们定义了两个函数来重新组织这些数据。第一个将每个三行组合并为一个具有八列数据的单行：

```py
    def row_merge(source_iter): 
        group = [] 
        for row in source_iter: 
            if len(row[0]) != 0: 
                if group: 
                    yield group 
                group = row.copy() 
            else: 
                group.extend(row) 
        if group: 
            yield group 

```

这是**头尾**算法的变体。当`len(row[0]) != 0`时，这是一个新组的标题行——任何先前完整的组都会被产生，然后`group`变量的工作值将根据这个标题行重置为一个新的、包含此标题行的列表。进行`copy()`操作，以便我们以后可以避免对列表对象进行变异。当`len(row[0]) == 0`时，这是组的尾部；该行被附加到`group`变量的工作值上。在数据源的末尾，通常有一个需要处理的完整组。有一个边缘情况，即根本没有数据；在这种情况下，也没有最终的组需要产生。

我们可以使用这个函数将数据从许多令人困惑的行转换为有用信息的单行：

```py
 **>>> from ch08_r02 import row_merge, log_rows 
>>> pprint(list(row_merge(log_rows))) 

[['date', 
  'engine on', 
  'fuel height', 
  '', 
  'engine off', 
  'fuel height', 
  '', 
  'Other notes', 
  ''], 
 ['10/25/13', 
  '08:24:00 AM', 
  '29', 
  '', 
  '01:15:00 PM', 
  '27', 
  '', 
  "calm seas -- anchor solomon's island", 
  ''], 
 ['10/26/13', 
  '09:12:00 AM', 
  '27', 
  '', 
  '06:25:00 PM', 
  '22', 
  '', 
  "choppy -- anchor in jackson's creek", 
  '']]** 

```

我们看到第一行只是电子表格的标题。我们想跳过这一行。我们将创建一个生成器表达式来处理过滤，并拒绝这一额外的行。

## 如何做...

1.  编写谓词函数，测试一个项目是否应该通过过滤器进行进一步处理。在某些情况下，我们将不得不从拒绝规则开始，然后编写反向规则，使其成为通过规则：

```py
        def pass_non_date(row): 
            return row[0] != 'date' 

```

这可以用三种方式来使用：语句、表达式和`filter()`函数。这是一个显式的`for...if...yield`模式的语句示例，用于传递行：

```py
        for item in collection: 
            if pass_non_date(item): 
                yield item 

```

这使用一个`for`语句来使用过滤函数处理集合中的每个项目。选择的项目被产生。其他项目被拒绝。

使用这个函数的第二种方式是在生成器表达式中使用它：

```py
        (item for item in data if pass_non_date(item)) 

```

这个生成器表达式应用了`filter`函数`pass_non_date()`到每个项目。第三种选择是`filter()`函数。

1.  使用`filter()`函数将函数应用于源数据：

```py
        filter(pass_non_date, data) 

```

我们提供了函数名`pass_non_date`。我们在函数名后面不使用`()`字符，因为这个表达式不会评估函数。`filter()`函数将给定的函数应用于可迭代的数据源`data`。在这种情况下，`data`是一个集合，但它可以是任何可迭代的对象，包括以前生成器表达式的结果。`pass_non_date()`函数为`true`的每个项目将被过滤器传递；所有其他值都被拒绝。

我们可以这样使用：

```py
        for row in filter(pass_non_date, row_merge(data)): 
            print(row[0], row[1], row[4]) 

```

`filter()`函数创建一个可迭代对象，将`pass_non_date()`函数作为规则应用于`row_merge(data)`可迭代对象中的每个项目，它产生了在第零列中没有`date`的行。

## 它是如何工作的...

`filter()`函数替换了一些常见的样板代码。我们可以想象定义看起来像这样：

```py
    def filter(f, iterable): 
        for item in iterable: 
            if f(item): 
                yield f(item) 

```

或者，我们可以想象它看起来像这样：

```py
    def filter(f, iterable): 
        return (item for item in iterable if f(item)) 

```

这两个定义总结了`filter()`函数的核心特性：一些数据被传递，一些数据被拒绝。这是一个方便的简写，消除了一些应用函数到可迭代数据源的样板代码。

## 还有更多...

有时候很难编写一个简单的规则来传递数据。如果我们编写一个拒绝数据的规则，可能会更清晰。例如，这可能更有意义：

```py
    def reject_date(row): 
        return row[0] == 'date' 

```

我们可以以多种方式使用拒绝规则。这是一个`for...if...continue...yield`语句的模式。这将使用 continue 跳过被拒绝的行，并产生剩下的行：

```py
    for item in collection: 
        if reject_date(item): 
            continue 
        yield item 

```

我们还可以使用这种变体。对于一些程序员来说，不拒绝的概念可能会变得令人困惑。这可能看起来像一个双重否定：

```py
    for item in collection: 
        if not reject_date(item): 
            yield item 

```

我们也可以使用类似这样的生成器表达式：

```py
    (item for item in data if not reject_date(item)) 

```

然而，我们不能轻松地使用`filter()`函数来拒绝数据的规则。`filter()`函数只能用于传递规则。

我们在处理这种逻辑时有两种基本选择。我们可以将逻辑包装在另一个表达式中，或者使用`itertools`模块中的函数。当涉及到包装时，我们还有两种选择。我们可以包装一个拒绝函数以创建一个传递函数。我们可以使用类似这样的东西：

```py
    def pass_date(row): 
        return not reject_date(row) 

```

这使得可以创建一个简单的拒绝规则，并在`filter()`函数中使用它。包装逻辑的另一种方法是创建一个`lambda`对象：

```py
     filter(lambda item: not reject_date(item), data) 

```

`lambda`函数是一个小的匿名函数。它是一个被简化为只有两个元素的函数：参数列表和一个单一的表达式。我们用`lambda`对象包装了`reject_date()`函数，以创建一种`not_reject_date`函数。

在`itertools`模块中，我们使用`filterfalse()`函数。我们可以导入`filterfalse()`并使用它来代替内置的`filter()`函数。

## 另请参阅...

+   在*使用堆叠的生成器表达式*配方中，我们将这样的函数放在一堆生成器中。我们从许多单独的映射和过滤操作中构建了一个复合函数，这些操作被编写为生成器函数。

# 总结一个集合-如何减少

在本章的介绍中，我们注意到有三种常见的处理模式：映射、过滤和减少。我们在*将变换应用到集合*配方中看到了映射的例子，在*挑选子集-三种过滤方式*配方中看到了过滤的例子。很容易看出这些是如何变得非常通用的。

映射将一个简单的函数应用于集合的所有元素。{*M*(*x*): *x* ∈ *C*}将函数*M*应用于较大集合*C*的每个项目*x*。在 Python 中，它可以看起来像这样：

```py
    (M(x) for x in C) 

```

或者，我们可以使用内置的`map()`函数来消除样板代码，并简化为这样：

```py
    map(M, c) 

```

类似地，过滤使用一个函数从集合中选择元素。{*x*: *x* ∈ *C* **if** *F*(*x*)}使用函数*F*来确定是否传递或拒绝来自较大集合*C*的项目*x*。我们可以用各种方式在 Python 中表达这一点，其中之一就像这样：

```py
    filter(F, c) 

```

这将一个谓词函数`F()`应用于集合`c`。

第三种常见的模式是缩减。在*设计具有大量处理的类*和*扩展集合：执行统计的列表*的示例中，我们看到了计算许多统计值的类定义。这些定义几乎完全依赖于内置的`sum()`函数。这是比较常见的缩减之一。

我们能否将求和概括为一种允许我们编写多种不同类型的缩减的方式？我们如何以更一般的方式定义缩减的概念？

## 准备就绪

最常见的缩减之一是求和。其他缩减包括乘积、最小值、最大值、平均值、方差，甚至是简单的值计数。

这是一种将数学定义的求和函数+应用于集合*C*中的值的方式：

![准备就绪](img/Image00019.jpg)

我们通过在值序列*C=c[0], c[1], c[2], ..., c[n]*中插入+运算符来扩展了求和的定义。这种在+运算符中进行*fold*的想法捕捉了内置的`sum()`函数的含义。

类似地，乘积的定义如下：

![准备就绪](img/Image00020.jpg)

在这里，我们对一系列值进行了不同的*fold*。通过 fold 扩展缩减涉及两个项目：一个二元运算符和一个基本值。对于求和，运算符是+，基本值是零。对于乘积，运算符是×，基本值是一。

我们可以定义一个通用的高级函数*F*[(⋄, ⊥)]，它捕捉了 fold 的理想。fold 函数定义包括一个运算符⋄的占位符和一个基本值⊥的占位符。给定集合*C*的函数值可以用这个递归规则来定义：

![准备就绪](img/Image00021.jpg)

如果集合*C*为空，则值为基本值⊥。当定义`sum()`时，基本值将为零。如果*C*不为空，则我们首先计算集合中除最后一个值之外的所有值的 fold，F◊, ⊥。然后我们将运算符（例如加法）应用于前一个 fold 结果和集合中的最后一个值*C[n-]*[1]。对于`sum()`，运算符是+。

我们在 Pythonic 意义上使用了*C*[0..*n*]的符号。包括索引 0 到*n*-1 的值，但不包括索引*n*的值。这意味着*C*[0..0]=∅：这个范围*C[0..0]*中没有元素。 

这个定义被称为**fold left**操作，因为这个定义的净效果是在集合中从左到右执行基础操作。这可以改为定义一个**fold right**操作。由于 Python 的`reduce()`函数是一个 fold left，我们将坚持使用它。

我们将定义一个`prod()`函数，用于计算阶乘值：

![准备就绪](img/Image00022.jpg)

*n*的阶乘的值是 1 到*n*之间所有数字的乘积。由于 Python 使用半开区间，使用*1* ≤ *x* < *n* + *1*来定义范围更符合 Python 的风格。这个定义更适合内置的`range()`函数。

使用我们之前定义的 fold 操作符，我们有这个。我们使用乘法*的操作符和基本值为 1 来定义了一个 fold（或者 reduce）：

![准备工作](img/Image00023.jpg)

折叠的概念是 Python 的`reduce()`概念的通用概念。我们可以将这应用于许多算法，可能简化定义。

## 如何做...

1.  从`functools`模块导入`reduce()`函数：

```py
 **>>> from functools import reduce** 

```

1.  选择运算符。对于求和，是`+`。对于乘积，是`*`。这些可以以多种方式定义。这是长版本。其他定义必要的二进制运算符的方法将在后面展示：

```py
 **>>> def mul(a, b): 
      ...     return a * b** 

```

1.  选择所需的基值。对于求和，它是零。对于乘积，它是一。这使我们能够定义一个计算通用乘积的`prod()`函数：

```py
 **>>> def prod(values): 
      ...    return reduce(mul, values, 1)** 

```

1.  对于阶乘，我们需要定义将被减少的数值序列：

```py
 **range(1, n+1)** 

```

这是`prod()`函数的工作原理：

```py
 **>>> prod(range(1, 5+1)) 
      120** 

```

这是整个阶乘函数：

```py
 **>>> def factorial(n): 
...    return prod(range(1, n+1))** 

```

这是一副 52 张牌的排列方式。这是值*52!*：

```py
 **>>> factorial(52) 
80658175170943878571660636856403766975289505440883277824000000000000** 

```

一副牌可以被洗牌的方式有很多种。

有多少种 5 张牌的可能手牌？二项式计算使用阶乘：

![如何做...](img/Image00024.jpg)

```py
 **>>> factorial(52)//(factorial(5)*factorial(52-5)) 
2598960** 

```

对于任何给定的洗牌，大约有 260 万种不同的可能扑克手牌。（是的，这是一种计算二项式的非常低效的方法。）

## 它是如何工作的...

`reduce()`函数的行为就好像它有这个定义：

```py
    def reduce(function, iterable, base): 
        result = base 
        for item in iterable: 
            result = function(result, item) 
        return result 

```

这将从左到右迭代数值。它将在可迭代集合中的前一组数值和下一个项目之间应用给定的二进制函数。

当我们看*递归函数和 Python 的堆栈限制*这个教程时，我们可以看到 fold 的递归定义可以优化为这个`for`语句。

## 还有更多...

在设计`reduce()`函数时，我们需要提供一个二进制运算符。有三种方法来定义必要的二进制运算符。我们使用了一个完整的函数定义，如下所示：

```py
    def mul(a, b): 
        return a * b 

```

还有两个选择。我们可以使用`lambda`对象而不是完整的函数：

```py
 **>>> add = lambda a, b: a + b 
>>> mul = lambda a, b: a * b** 

```

`lambda`函数是一个匿名函数，只包含两个基本元素：参数和返回表达式。lambda 内部没有语句，只有一个表达式。在这种情况下，表达式只是使用所需的运算符。

我们可以像这样使用它：

```py
 **>>> def prod2(values): 
...     return reduce(lambda a, b: a*b, values, 1)** 

```

这提供了乘法函数作为一个`lambda`对象，而不需要额外的函数定义开销。

我们还可以从`operator`模块导入定义：

```py
    from operator import add, mul 

```

这对所有内置的算术运算符都适用。

请注意，使用逻辑运算符**AND**和**OR**的逻辑归约与其他算术归约有所不同。这些运算符会短路：一旦值为`false`，**and-reduce**就可以停止处理。同样，一旦值为`True`，**or-reduce**就可以停止处理。内置函数`any()`和`all()`很好地体现了这一点。使用内置的`reduce()`很难捕捉到这种短路特性。

### 最大值和最小值

我们如何使用`reduce()`来计算最大值或最小值？这更复杂一些，因为没有可以使用的平凡基值。我们不能从零或一开始，因为这些值可能超出被最小化或最大化的值范围。

此外，内置的`max()`和`min()`必须对空序列引发异常。这些函数无法完全适应`sum()`函数和`reduce()`函数的工作方式。

我们必须使用类似这样的东西来提供期望的功能集：

```py
    def mymax(sequence): 
        try: 
            base = sequence[0] 
            max_rule = lambda a, b: a if a > b else b 
            reduce(max_rule, sequence, base) 
        except IndexError: 
            raise ValueError 

```

这个函数将从序列中选择第一个值作为基值。它创建了一个名为`max_rule`的`lambda`对象，它选择两个参数值中较大的那个。然后我们可以使用数据中的这个基值和`lambda`对象。`reduce()`函数将在非空集合中找到最大的值。我们捕获了`IndexError`异常，以便一个空集合会引发`ValueError`异常。

这个例子展示了我们如何发明一个更复杂或精密的最小值或最大值函数，它仍然基于内置的`reduce()`函数。这样做的好处是可以替换减少集合到单个值时的样板`for`语句。

### 滥用的潜力

请注意，折叠（或在 Python 中称为`reduce()`）可能会被滥用，导致性能不佳。我们必须谨慎使用`reduce()`函数，仔细考虑最终算法可能是什么样子。特别是，被折叠到集合中的运算符应该是一个简单的过程，比如加法或乘法。使用`reduce()`会将**O**（1）操作的复杂性改变为**O**（*n*）。

想象一下，如果在减少过程中应用的运算符涉及对集合进行排序会发生什么。在`reduce()`中使用复杂的运算符-具有**O**（*n* log *n*）复杂度-会将整体`reduce()`的复杂度改变为*O*（*n²* log *n*）。

# 组合映射和减少转换

在本章的其他配方中，我们一直在研究映射、过滤和减少操作。我们分别研究了这三个操作：

+   *对集合应用转换*配方显示`map()`函数

+   *选择子集-三种过滤方法*配方显示`filter()`函数

+   *总结集合-如何减少*配方显示`reduce()`函数

许多算法将涉及函数的组合。我们经常使用映射、过滤和减少来生成可用数据的摘要。此外，我们需要看一下使用迭代器和生成器函数的一个深刻限制。即这个限制：

### 提示

迭代器只能产生一次值。

如果我们从生成器函数和集合数据创建一个迭代器，那么迭代器只会产生数据一次。之后，它将看起来是一个空序列。

这是一个例子：

```py
 **>>> typical_iterator = iter([0, 1, 2, 3, 4]) 
>>> sum(typical_iterator) 
10 
>>> sum(typical_iterator) 
0** 

```

我们通过手动将`iter()`函数应用于文字列表对象来创建了一个值序列的迭代器。`sum()`函数第一次使用`typical_iterator`的值时，它消耗了所有五个值。下一次我们尝试将任何函数应用于`typical_iterator`时，将不会有更多的值被消耗-迭代器看起来是空的。

这种基本的一次性限制驱动了在使用多种类型的生成器函数与映射、过滤和减少一起工作时的一些设计考虑。我们经常需要缓存中间结果，以便我们可以对数据执行多次减少。

## 准备好

在*使用堆叠的生成器表达式*配方中，我们研究了需要多个处理步骤的数据。我们使用生成器函数合并了行。我们过滤掉了一些行，将它们从结果数据中删除。此外，我们对数据应用了许多映射，将日期和时间转换为更有用的信息。

我们想要通过两次减少来补充这一点，以获得一些平均值和方差信息。这些统计数据将帮助我们更充分地理解数据。

我们有一个用于记录大帆船燃料消耗的电子表格。它的行看起来像这样：

| **日期** | **引擎开启** | **燃料高度** |
| --- | --- | --- |
|  | **关闭引擎** | **燃料高度** |
|  | **其他说明** |  |
| 10/25/2013 | 08:24 | 29 |
|  | 13:15 | 27 |
|  | 平静的海洋-锚所罗门岛 |  |
| 10/26/2013 | 09:12 | 27 |
|  | 18:25 | 22 |
|  | 波涛汹涌-锚在杰克逊溪 |  |

最初的处理是一系列操作，改变数据的组织，过滤掉标题，并计算一些有用的值。

## 如何做到...

1.  从目标开始。在这种情况下，我们想要一个可以像这样使用的函数：

```py
 **>>> round(sum_fuel(clean_data(row_merge(log_rows))), 3) 
      7.0** 

```

这显示了这种处理的三步模式。这三步将定义我们创建减少的各个部分的方法：

1.  首先，转换数据组织。有时这被称为数据规范化。在这种情况下，我们将使用一个名为`row_merge()`的函数。有关此信息，请参阅*使用堆叠的生成器表达式*食谱。

1.  其次，使用映射和过滤来清洁和丰富数据。这被定义为一个单一函数，`clean_data()`。

1.  最后，使用`sum_fuel()`将数据减少到总和。还有各种其他减少的方法是有意义的。我们可能计算平均值，或其他值的总和。我们可能想应用很多减少。

1.  如果需要，定义数据结构规范化函数。这几乎总是必须是一个生成器函数。结构性的改变不能通过`map()`应用：

```py
        from ch08_r02 import row_merge 

```

如*使用堆叠的生成器表达式*食谱所示，此生成器函数将把每次航行的三行数据重组为每次航行的一行数据。当所有列都在一行中时，数据处理起来更容易。

1.  定义整体数据清洗和增强数据函数。这是一个由简单函数构建的生成器函数。它是一系列`map()`和`filter()`操作，将从源字段派生数据：

```py
        def clean_data(source): 
            namespace_iter = map(make_namespace, source) 
            fitered_source = filter(remove_date, namespace_iter) 
            start_iter = map(start_datetime, fitered_source) 
            end_iter = map(end_datetime, start_iter) 
            delta_iter = map(duration, end_iter) 
            fuel_iter = map(fuel_use, delta_iter) 
            per_hour_iter = map(fuel_per_hour, fuel_iter) 
            return per_hour_iter 

```

每个`map()`和`filter()`操作都涉及一个小函数，对数据进行单个转换或计算。

1.  定义用于清洗和派生其他数据的单个函数。

1.  将合并的数据行转换为`SimpleNamespace`。这将允许我们使用名称，如`start_time`，而不是`row[1]`：

```py
        from types import SimpleNamespace 
        def make_namespace(row): 
            ns = SimpleNamespace( 
                date = row[0], 
                start_time = row[1], 
                start_fuel_height = row[2], 
                end_time = row[4], 
                end_fuel_height = row[5], 
                other_notes = row[7] 
            ) 
            return ns 

```

此函数从源数据的选定列构建一个`SimpleNamspace`。第三列和第六列被省略，因为它们始终是零长度的字符串，`''`。

1.  这是由`filter()`用于删除标题行的函数。如果需要，这可以扩展到从源数据中删除空行或其他不良数据。想法是尽快在处理中删除不良数据：

```py
        def remove_date(row_ns): 
            return not(row_ns.date == 'date') 

```

1.  将数据转换为可用形式。首先，我们将字符串转换为日期。接下来的两个函数依赖于这个`timestamp()`函数，它将一个列中的`date`字符串加上另一个列中的`time`字符串转换为一个适当的`datetime`实例：

```py
        import datetime 
        def timestamp(date_text, time_text): 
            date = datetime.datetime.strptime(date_text, "%m/%d/%y").date() 
            time = datetime.datetime.strptime(time_text, "%I:%M:%S %p").time() 
            timestamp = datetime.datetime.combine(date, time) 
            return timestamp 

```

这使我们能够根据`datetime`库进行简单的日期计算。特别是，减去两个时间戳将创建一个`timedelta`对象，其中包含任何两个日期之间的确切秒数。

这是我们将如何使用此函数为航行的开始和结束创建适当的时间戳：

```py
        def start_datetime(row_ns): 
            row_ns.start_timestamp = timestamp(row_ns.date, row_ns.start_time) 
            return row_ns 

        def end_datetime(row_ns): 
            row_ns.end_timestamp = timestamp(row_ns.date, row_ns.end_time) 
            return row_ns 

```

这两个函数都将向`SimpleNamespace`添加一个新属性，并返回命名空间对象。这允许这些函数在`map()`操作的堆栈中使用。我们还可以重写这些函数，用不可变的`namedtuple()`替换可变的`SimpleNamespace`，并仍然保留`map()`操作的堆栈。

1.  计算派生时间数据。在这种情况下，我们也可以计算持续时间。这是一个必须在前两个之后执行的函数：

```py
        def duration(row_ns): 
            travel_time = row_ns.end_timestamp - row_ns.start_timestamp 
            row_ns.travel_hours = round(travel_time.total_seconds()/60/60, 1) 
            return row_ns 

```

这将把秒数差转换为小时值。它还会四舍五入到最接近的十分之一小时。比这更精确的信息基本上是噪音。出发和到达时间（通常）至少相差一分钟；它们取决于船长记得看手表的时间。在某些情况下，她可能已经估计了时间。

1.  计算分析所需的其他指标。这包括创建转换为浮点数的高度值。最终的计算基于另外两个计算结果：

```py
        def fuel_use(row_ns): 
            end_height = float(row_ns.end_fuel_height) 
            start_height = float(row_ns.start_fuel_height) 
            row_ns.fuel_change = start_height - end_height 
            return row_ns 

        def fuel_per_hour(row_ns): 
            row_ns.fuel_per_hour = row_ns.fuel_change/row_ns.travel_hours 
            return row_ns 

```

每小时燃料消耗量取决于整个前面的计算堆栈。旅行小时数来自分别计算的开始和结束时间戳。

## 它是如何工作的...

想法是创建一个遵循常见模板的复合操作：

1.  规范化结构：这通常需要一个生成器函数，以在不同结构中读取数据并产生数据。

1.  过滤和清洗：这可能涉及一个简单的过滤，就像这个例子中所示的那样。我们稍后会看到更复杂的过滤器。

1.  通过映射或类定义的惰性属性派生数据：具有惰性属性的类是一个反应式对象。对源属性的任何更改都应该导致计算属性的更改。

在某些情况下，我们可能需要将基本事实与其他维度描述相结合。例如，我们可能需要查找参考数据，或解码编码字段。

一旦我们完成了初步步骤，我们就有了可用于各种分析的数据。很多时候，这是一个减少操作。初始示例计算了燃料使用量的总和。这里还有另外两个例子：

```py
    from statistics import * 
    def avg_fuel_per_hour(iterable): 
        return mean(row.fuel_per_hour for row in iterable) 
    def stdev_fuel_per_hour(iterable): 
        return stdev(row.fuel_per_hour for row in iterable) 

```

这些函数将`mean()`和`stdev()`函数应用于丰富数据的每一行的`fuel_per_hour`属性。

我们可以这样使用它：

```py
 **>>> round(avg_fuel_per_hour( 
...    clean_data(row_merge(log_rows))), 3) 
0.48** 

```

我们使用`clean_data(row_merge(log_rows))`映射管道来清理和丰富原始数据。然后我们对这些数据应用了减少以获得我们感兴趣的值。

现在我们知道我们的 30 英寸高的油箱可以支持大约 60 小时的动力。以 6 节的速度，我们可以在满油箱的情况下行驶大约 360 海里。

## 还有更多...

正如我们所指出的，我们只能对可迭代的数据源执行一次减少。如果我们想要计算多个平均值，或者平均值和方差，我们将需要使用稍微不同的模式。

为了计算数据的多个摘要，我们需要创建一种可以重复进行摘要的序列对象：

```py
    data = tuple(clean_data(row_merge(log_rows))) 
    m = avg_fuel_per_hour(data) 
    s = 2*stdev_fuel_per_hour(data) 
    print("Fuel use {m:.2f} ±{s:.2f}".format(m=m, s=s)) 

```

在这里，我们从清理和丰富的数据中创建了一个`tuple`。这个`tuple`将产生一个可迭代对象，但与生成器函数不同，它可以多次产生这个可迭代对象。我们可以使用`tuple`对象计算两个摘要。

这个设计涉及大量的源数据转换。我们使用了一系列 map、filter 和 reduce 操作来构建它。这提供了很大的灵活性。

另一种方法是创建一个类定义。一个类可以设计为具有惰性属性。这将创建一种反应式设计，体现在单个代码块中。请参阅*使用属性进行惰性属性*配方，了解这方面的示例。

我们还可以在`itertools`模块中使用`tee()`函数进行这种处理：

```py
    from itertools import tee 
    data1, data2 = tee(clean_data(row_merge(log_rows)), 2) 
    m = avg_fuel_per_hour(data1) 
    s = 2*stdev_fuel_per_hour(data2) 

```

我们使用`tee()`创建了`clean_data(row_merge(log_rows))`的可迭代输出的两个克隆。我们可以使用这两个克隆来计算平均值和标准差。

## 另请参阅

+   我们已经看过如何在*使用堆叠的生成器表达式*配方中结合映射和过滤。

+   我们在*使用属性进行惰性属性*配方中看过懒惰属性。此外，这个配方还涉及 map-reduce 处理的一些重要变化。

# 实现“存在”处理

我们一直在研究的处理模式都可以用量词*对于所有*来总结。这已经是所有处理定义的一个隐含部分：

+   **映射**：对于源中的所有项目，应用映射函数。我们可以使用量词来明确这一点：{ *M* ( *x* ) ∀ *x* : *x* ∈ *C* }

+   **过滤**：对于源中的所有项目，传递那些过滤函数为`true`的项目。这里也使用了量词来明确这一点。如果某个函数*F(x)*为`true`，我们希望从集合*C*中获取所有值*x*：{ *x* ∀ *x* : *x* ∈ *C* **if** *F* ( *x* )}

+   **减少**：对于源中的所有项目，使用给定的运算符和基本值来计算摘要。这个规则是一个递归，对于源集合或可迭代的所有值都清晰地适用：![实现“存在”处理](img/Image00025.jpg)。

我们在 Pythonic 意义上使用了*C[0..n]*的符号。索引位置为 0 和*n-1*的值是包括在内的，但索引位置为*n*的值不包括在内。这意味着这个范围内没有元素。

更重要的是*C[0..n-1 ]* ∪ *C[n-1]* = *C* 。也就是说，当我们从范围中取出最后一项时，不会丢失任何项——我们总是在处理集合中的所有项。此外，我们不会两次处理项*C[n-1]*。它不是*C[0..n-1]*范围的一部分，而是一个独立的项*C[n-1]*。

我们如何使用生成器函数编写一个进程，当第一个值匹配某个谓词时停止？我们如何避免*对于所有*并用*存在*量化我们的逻辑？

## 准备工作

我们可能需要另一个量词——*存在*，∃。让我们看一个存在性测试的例子。

我们可能想知道一个数是素数还是合数。我们不需要知道一个数的所有因子就能知道它不是素数。只要证明存在一个因子就足以知道一个数不是素数。

我们可以定义一个素数谓词*P(n)*，如下所示：

*P* ( *n* ) = ¬∃ *i* : 2 ≤ *i* < *n* **if** *n* mod *i* = 0

一个数*n*，如果不存在一个值*i*（在 2 和这个数之间），能够整除这个数，那么它是素数。我们可以将否定移到周围，并重新表述如下：

*¬P* ( *n* ) = ∃ *i* : 2 ≤ *i* < *n* **if** *n* mod *i* = 0

一个数*n*，如果存在一个值*i*，在 2 和这个数本身之间，能够整除这个数，那么它是合数（非素数）。我们不需要知道**所有**这样的值。满足谓词的一个值的存在就足够了。

一旦我们找到这样的数字，我们可以从任何迭代中提前中断。这需要在`for`和`if`语句中使用`break`语句。因为我们不处理所有的值，所以我们不能轻易使用高阶函数，比如`map()`、`filter()`或`reduce()`。

## 如何做...

1.  定义一个生成器函数模板，它将跳过项目，直到找到所需的项目。这将产生只有一个通过谓词测试的值：

```py
        def find_first(predicate, iterable): 
            for item in iterable: 
                if predicate(item): 
                    yield item 
                    break 

```

1.  定义一个谓词函数。对于我们的目的，一个简单的`lambda`对象就可以了。此外，lambda 允许我们使用一个绑定到迭代的变量和一个自由于迭代的变量。这是表达式：

```py
        lambda i: n % i == 0 

```

在这个 lambda 中，我们依赖一个非局部值`n`。这将是 lambda 的*全局*值，但仍然是整个函数的局部值。如果`n % i`是`0`，那么`i`是`n`的一个因子，`n`不是素数。

1.  使用给定的范围和谓词应用该函数：

```py
        import math 
        def prime(n): 
            factors = find_first( 
                lambda i: n % i == 0, 
                range(2, int(math.sqrt(n)+1)) ) 
            return len(list(factors)) == 0 

```

如果`factors`可迭代对象有一个项，那么`n`是合数。否则，`factors`可迭代对象中没有值，这意味着`n`是一个素数。

实际上，我们不需要测试两个和`n`之间的每一个数字，以确定`n`是否是素数。只需要测试值`i`，使得*2* ≤ *i* < √ *n*。

## 它是如何工作的...

在`find_first()`函数中，我们引入了一个`break`语句来停止处理源可迭代对象。当`for`语句停止时，生成器将到达函数的末尾，并正常返回。

从这个生成器中消耗值的进程将得到`StopIteration`异常。这个异常意味着生成器不会再产生值。`find_first()`函数会引发一个异常，但这不是一个错误。这是信号一个可迭代对象已经完成了输入值的处理的正常方式。

在这种情况下，信号意味着两种可能：

+   如果产生了一个值，那么这个值是`n`的一个因子

+   如果没有产生值，那么`n`是素数

从`for`语句中提前中断的这个小改变，使得生成器函数的含义发生了巨大的变化。与处理源的**所有**值不同，`find_first()`生成器将在谓词为`true`时停止处理。

这与过滤器不同，过滤器会消耗所有的源值。当使用`break`语句提前离开`for`语句时，一些源值可能不会被处理。

## 还有更多...

在`itertools`模块中，有一个替代`find_first()`函数的方法。`takewhile()`函数使用一个谓词函数来保持从输入中获取值。当谓词变为`false`时，函数停止处理值。

我们可以很容易地将 lambda 从`lambda i: n % i == 0`改为`lambda i: n % i != 0`。这将允许函数在它们不是因子时接受值。任何是因子的值都会通过结束`takewhile()`过程来停止处理。

让我们来看两个例子。我们将测试`13`是否为质数。我们需要检查范围内的数字。我们还将测试`15`是否为质数：

```py
 **>>> from itertools import takewhile 
>>> n = 13 
>>> list(takewhile(lambda i: n % i != 0, range(2, 4))) 
[2, 3] 
>>> n = 15 
>>> list(takewhile(lambda i: n % i != 0, range(2, 4))) 
[2]** 

```

对于质数，所有的测试值都通过了`takewhile()`谓词。结果是给定数字*n*的非因子列表。如果非因子的集合与被测试的值的集合相同，那么`n`是质数。在`13`的情况下，两个值的集合都是`[2, 3]`。

对于合数，一些值通过了`takewhile()`谓词。在这个例子中，`2`不是`15`的因子。然而，`3`是一个因子；这不符合谓词。非因子的集合`[2]`与被测试的值的集合`[2, 3]`不同。

我们最终得到的函数看起来像这样：

```py
    def prime_t(n): 
        tests = set(range(2, int(math.sqrt(n)+1))) 
        non_factors = set( 
            takewhile( 
                lambda i: n % i != 0, 
                tests 
            ) 
        ) 
        return tests == non_factors 

```

这创建了两个中间集合对象`tests`和`non_factors`。如果所有被测试的值都不是因子，那么这个数就是质数。之前展示的函数，基于`find_first()`只创建了一个中间列表对象。那个列表最多只有一个成员，使得数据结构更小。

### itertools 模块

`itertools`模块中还有许多其他函数，我们可以用来简化复杂的映射-归约应用：

+   `filterfalse()`：它是内置`filter()`函数的伴侣。它颠倒了`filter()`函数的谓词逻辑；它拒绝谓词为`true`的项目。

+   `zip_longest()`：它是内置`zip()`函数的伴侣。内置的`zip()`函数在最短的可迭代对象耗尽时停止合并项目。`zip_longest()`函数将提供一个给定的填充值，以使短的可迭代对象与最长的可迭代对象匹配。

+   `starmap()`：这是对基本`map()`算法的修改。当我们执行`map(function, iter1, iter2)`时，每个可迭代对象中的一个项目将作为给定函数的两个位置参数提供。`starmap()`期望一个可迭代对象提供一个包含参数值的元组。实际上：

```py
        map = starmap(function, zip(iter1, iter2)) 

```

还有其他一些我们可能也会用到的：

+   `accumulate()`：这个函数是内置`sum()`函数的一个变体。它会产生在达到最终总和之前产生的每个部分总和。

+   `chain()`：这个函数将按顺序合并可迭代对象。

+   `compress()`：这个函数使用一个可迭代对象作为数据源，另一个可迭代对象作为选择器的数据源。当选择器的项目为 true 时，相应的数据项目被传递。否则，数据项目被拒绝。这是基于真假值的逐项过滤器。

+   `dropwhile()`：只要这个函数的谓词为`true`，它就会拒绝值。一旦谓词变为`false`，它就会传递所有剩余的值。参见`takewhile()`。

+   `groupby()`：这个函数使用一个键函数来控制组的定义。具有相同键值的项目被分组到单独的迭代器中。为了使结果有用，原始数据应该按键的顺序排序。

+   `islice()`：这个函数类似于切片表达式，只不过它适用于可迭代对象，而不是列表。当我们使用`list[1:]`来丢弃列表的第一行时，我们可以使用`islice(iterable, 1)`来丢弃可迭代对象的第一个项目。

+   `takewhile()`：只要谓词为`true`，这个函数就会传递值。一旦谓词变为`false`，就停止处理任何剩余的值。参见`dropwhile()`。

+   `tee()`：这将单个可迭代对象分成多个克隆。然后可以单独消耗每个克隆。这是在单个可迭代数据源上执行多个减少的一种方法。

# 创建一个部分函数

当我们查看`reduce()`、`sorted()`、`min()`和`max()`等函数时，我们会发现我们经常有一些*永久*参数值。例如，我们可能会发现需要在几个地方写类似这样的东西：

```py
    reduce(operator.mul, ..., 1) 

```

对于`reduce()`的三个参数，只有一个-要处理的可迭代对象-实际上会改变。运算符和基本值参数基本上固定为`operator.mul`和`1`。

显然，我们可以为此定义一个全新的函数：

```py
    def prod(iterable): 
        return reduce(operator.mul, iterable, 1) 

```

然而，Python 有一些简化这种模式的方法，这样我们就不必重复使用样板`def`和`return`语句。

我们如何定义一个具有预先提供一些参数的函数？

请注意，这里的目标与提供默认值不同。部分函数不提供覆盖默认值的方法。相反，我们希望创建尽可能多的部分函数，每个函数都提前绑定了特定的参数。

## 准备工作

一些统计建模是用标准分数来完成的，有时被称为**z 分数**。其想法是将原始测量标准化到一个可以轻松与正态分布进行比较的值，并且可以轻松与以不同单位测量的相关数字进行比较。

计算如下：

*z* = ( *x* - μ)/σ

这里，*x*是原始值，μ是总体均值，σ是总体标准差。值*z*的均值为 0，标准差为 1。这使得它特别容易处理。

我们可以使用这个值来发现**异常值**-与均值相距甚远的值。我们期望我们的*z*值(大约)99.7%会在-3 和+3 之间。

我们可以定义一个这样的函数：

```py
    def standarize(mean, stdev, x): 
        return (x-mean)/stdev 

```

这个`standardize()`函数将从原始分数*x*计算出 z 分数。这个函数有两种类型的参数：

+   `mean`和`stdev`的值基本上是固定的。一旦我们计算出总体值，我们将不断地将它们提供给`standardize()`函数。

+   `x`的值更加可变。

假设我们有一系列大块文本中的数据样本：

```py
    text_1 = '''10  8.04 
    8       6.95 
    13      7.58 
    ... 
    5       5.68 
    ''' 

```

我们已经定义了两个小函数来将这些数据转换为数字对。第一个简单地将每个文本块分解为一系列行，然后将每行分解为一对文本项：

```py
    text_parse = lambda text: (r.split() for r in text.splitlines()) 

```

我们已经使用文本块的`splitlines()`方法创建了一系列行。我们将其放入生成器函数中，以便每个单独的行都可以分配给`r`。使用`r.split()`将每行中的两个文本块分开。

如果我们使用`list(text_parse(text_1))`，我们会看到这样的数据：

```py
    [['10', '8.04'], 
     ['8', '6.95'], 
     ['13', '7.58'], 
     ... 
     ['5', '5.68']] 

```

我们需要进一步丰富这些数据，使其更易于使用。我们需要将字符串转换为适当的浮点值。在这样做的同时，我们将从每个项目创建`SimpleNamespace`实例：

```py
    from types import SimpleNamespace 
    row_build = lambda rows: (SimpleNamespace(x=float(x), y=float(y)) for x,y in rows) 

```

`lambda`对象通过将`float()`函数应用于每行中的每个字符串项来创建`SimpleNamespace`实例。这给了我们可以处理的数据。

我们可以将这两个`lambda`对象应用于数据，以创建一些可用的数据集。之前，我们展示了`text_1`。我们假设我们有一个类似的第二组数据分配给`text_2`：

```py
    data_1 = list(row_build(text_parse(text_1))) 
    data_2 = list(row_build(text_parse(text_2))) 

```

这样就创建了两个类似文本块的数据。每个都有数据点对。`SimpleNamespace`对象有两个属性，`x`和`y`，分配给数据的每一行。

请注意，这个过程创建了`types.SimpleNamespace`的实例。当我们打印它们时，它们将使用`namespace`类显示。这些是可变对象，因此我们可以用标准化的 z 分数更新每一个。

打印`data_1`看起来像这样：

```py
    [namespace(x=10.0, y=8.04), namespace(x=8.0, y=6.95), 
namespace(x=13.0, y=7.58), 
    ..., 
    namespace(x=5.0, y=5.68)] 

```

例如，我们将计算`x`属性的标准化值。这意味着获取均值和标准差。然后我们需要将这些值应用于标准化我们两个集合中的数据。看起来是这样的：

```py
    import statistics 
    mean_x = statistics.mean(item.x for item in data_1) 
    stdev_x = statistics.stdev(item.x for item in data_1) 

    for row in data_1: 
        z_x = standardize(mean_x, stdev_x, row.x) 
        print(row, z_x) 

    for row in data_2: 
        z_x = standardize(mean_x, stdev_x, row.x) 
        print(row, z_x) 

```

每次评估`standardize()`时提供`mean_v1`，`stdev_v1`值可能会使算法混乱，因为这些细节并不是非常重要。在一些相当复杂的算法中，这种混乱可能导致更多的困惑而不是清晰。

## 如何做...

除了简单地使用`def`语句创建具有部分参数值的函数之外，我们还有两种其他方法来创建部分函数：

+   使用`functools`模块的`partial()`函数

+   创建`lambda`对象

### 使用 functools.partial()

1.  从`functools`导入`partial`函数：

```py
        from functools import partial 

```

1.  使用`partial()`创建对象。我们提供基本函数，以及需要包括的位置参数。在定义部分时未提供的任何参数在评估部分时必须提供：

```py
        z = partial(standardize, mean_x, stdev_x) 

```

1.  我们已为前两个位置参数`mean`和`stdev`提供了值。第三个位置参数`x`必须在计算值时提供。

### 创建`lambda`对象

1.  定义绑定固定参数的`lambda`对象：

```py
        lambda x: standardize(mean_v1, stdev_v1, x) 

```

1.  使用`lambda`创建对象：

```py
        z = lambda x: standardize(mean_v1, stdev_v1, x) 

```

## 它是如何工作的...

这两种技术都创建了一个可调用对象——一个名为`z()`的函数，其值为`mean_v1`和`stdev_v1`已经绑定到前两个位置参数。使用任一方法，我们的处理看起来可能是这样的：

```py
    for row in data_1: 
        print(row, z(row.x)) 

    for row in data_2: 
        print(row, z(row.x)) 

```

我们已将`z()`函数应用于每组数据。因为函数已经应用了一些参数，所以在这里使用看起来非常简单。

我们还可以这样做，因为每行都是一个可变对象：

```py
    for row in data_1: 
        row.z = z(row.v1) 

    for row in data_2: 
        row.z = z(row.v1) 

```

我们已更新行，包括一个新属性`z`，其值为`z()`函数。在复杂的算法中，调整行对象可能是一个有用的简化。

创建`z()`函数的两种技术之间存在显着差异：

+   `partial()`函数绑定参数的实际值。对使用的变量进行的任何后续更改都不会改变创建的部分函数的定义。创建`z = partial(standardize(mean_v1, stdev_v1))`后，更改`mean_v1`或`stdev_v1`的值不会对部分函数`z()`产生影响。

+   `lambda`对象绑定变量名，而不是值。对变量值的任何后续更改都将改变 lambda 的行为方式。创建`z = lambda x: standardize(mean_v1, stdev_v1, x)`后，更改`mean_v1`或`stdev_v1`的值将改变`lambda`对象`z()`使用的值。

我们可以稍微修改 lambda 以绑定值而不是名称：

```py
    z = lambda x, m=mean_v1, s=stdev_v1: standardize(m, s, x) 

```

这将提取`mean_v1`和`stdev_v1`的值以创建`lambda`对象的默认值。`mean_v1`和`stdev_v1`的值现在与`lambda`对象`z()`的正常操作无关。

## 还有更多...

在创建部分函数时，我们可以提供关键字参数值以及位置参数值。在许多情况下，这很好用。也有一些情况不适用。

特别是`reduce()`函数不能简单地转换为部分函数。参数的顺序不是创建部分的理想顺序。`reduce()`函数具有以下概念定义。这不是它的定义方式——这是它*看起来*的定义方式：

```py
    def reduce(function, iterable, initializer=None) 

```

如果这是实际定义，我们可以这样做：

```py
    prod = partial(reduce(mul, initializer=1)) 

```

实际上，我们无法这样做，因为`reduce()`的定义比看起来更复杂一些。`reduce()`函数不允许命名参数值。这意味着我们被迫使用 lambda 技术：

```py
 **>>> from operator import mul 
>>> from functools import reduce 
>>> prod = lambda x: reduce(mul, x, 1)** 

```

我们使用`lambda`对象定义了一个只有一个参数`prod()`函数。这个函数使用两个固定参数和一个可变参数与`reduce()`一起使用。

有了`prod()`的定义，我们可以定义依赖于计算乘积的其他函数。下面是`factorial`函数的定义：

```py
 **>>> factorial = lambda x: prod(range(2,x+1)) 
>>> factorial(5) 
120** 

```

`factorial()`的定义取决于`prod()`。`prod()`的定义是一种使用`reduce()`和两个固定参数值的部分函数。我们设法使用了一些定义来创建一个相当复杂的函数。

在 Python 中，函数是一个对象。我们已经看到了函数可以作为参数传递的多种方式。接受另一个函数作为参数的函数有时被称为**高阶函数**。

同样，函数也可以返回一个函数对象作为结果。这意味着我们可以创建一个像这样的函数：

```py
    def prepare_z(data): 
        mean_x = statistics.mean(item.x for item in data_1) 
        stdev_x = statistics.stdev(item.x for item in data_1) 
        return partial(standardize, mean_x, stdev_x) 

```

我们已经定义了一个在一组（*x*，*y*）样本上的函数。我们计算了每个样本的*x*属性的均值和标准差。然后我们创建了一个可以根据计算出的统计数据标准化得分的部分函数。这个函数的结果是一个我们可以用于数据分析的函数：

```py
    z = prepare_z(data_1) 
    for row in data_2: 
        print(row, z(row.x)) 

```

当我们评估`prepare_z()`函数时，它返回了一个函数。我们将这个函数赋给一个变量`z`。这个变量是一个可调用对象；它是函数`z()`，它将根据样本均值和标准差标准化得分。

# 使用不可变数据结构简化复杂算法

有状态对象的概念是面向对象编程的一个常见特性。我们在第六章和第七章中看过与对象和状态相关的一些技术，*类和对象的基础*和*更高级的类设计*。面向对象设计的重点之一是创建能够改变对象状态的方法。

我们还在*使用堆叠的生成器表达式*、*组合 map 和 reduce 转换*和*创建部分函数*配方中看过一些有状态的函数式编程技术。我们使用`types.SimpleNamespace`，因为它创建了一个简单的、有状态的对象，具有易于使用的属性名称。

在大多数情况下，我们一直在处理具有 Python `dict`对象定义属性的对象。唯一的例外是*使用 __slots__ 优化小对象*配方，其中属性由`__slots__`属性定义固定。

使用`dict`对象存储对象的属性有几个后果：

+   我们可以轻松地添加和删除属性。我们不仅仅局限于设置和获取已定义的属性；我们也可以创建新属性。

+   每个对象使用的内存量比最小必要量稍微大一些。这是因为字典使用哈希算法来定位键和值。哈希处理通常需要比其他结构（如`list`或`tuple`）更多的内存。对于非常大量的数据，这可能会成为一个问题。

有状态的面向对象编程最重要的问题是有时很难对对象的状态变化写出清晰的断言。与其定义关于状态变化的断言，更容易的方法是创建完全新的对象，其状态可以简单地映射到对象的类型。这与 Python 类型提示结合使用，有时可以创建更可靠、更易于测试的软件。

当我们创建新对象时，数据项和计算之间的关系可以被明确捕获。`mypy`项目提供了工具，可以分析这些类型提示，以确认复杂算法中使用的对象是否被正确使用。

在某些情况下，我们也可以通过避免首先使用有状态对象来减少内存的使用量。我们有两种技术可以做到这一点：

+   使用带有`__slots__`的类定义：有关此内容，请参阅*使用 __slots__ 优化小对象*的示例。这些对象是可变的，因此我们可以使用新值更新属性。

+   使用不可变的`tuples`或`namedtuples`：有关此内容，请参阅*设计具有少量独特处理的类*的示例。这些对象是不可变的。我们可以创建新对象，但无法更改对象的状态。整体内存的成本节约必须平衡创建新对象的额外成本。

不可变对象可能比可变对象稍快。更重要的好处是算法设计。在某些情况下，编写函数从旧的不可变对象创建新的不可变对象可能比处理有状态对象的算法更简单、更容易测试和调试。编写类型提示可以帮助这个过程。

## 准备工作

正如我们在*使用堆叠的生成器表达式*和*实现“存在”处理*的示例中所指出的，我们只能处理生成器一次。如果我们需要多次处理它，可迭代对象的序列必须转换为像列表或元组这样的完整集合。

这通常会导致一个多阶段的过程：

+   **初始提取数据**：这可能涉及数据库查询或读取`.csv`文件。这个阶段可以被实现为一个产生行或甚至返回生成器函数的函数。

+   **清洗和过滤数据**：这可能涉及一系列生成器表达式，可以仅处理一次源。这个阶段通常被实现为一个包含多个映射和过滤操作的函数。

+   **丰富数据**：这也可能涉及一系列生成器表达式，可以一次处理一行数据。这通常是一系列的映射操作，用于从现有数据中创建新的派生数据。

+   **减少或总结数据**：这可能涉及多个摘要。为了使其工作，丰富阶段的输出需要是可以多次处理的集合对象。

在某些情况下，丰富和总结过程可能会交错进行。正如我们在*创建部分函数*示例中看到的，我们可能会先进行一些总结，然后再进行更多的丰富。

处理丰富阶段有两种常见策略：

+   **可变对象**：这意味着丰富处理会添加或设置属性的值。可以通过急切计算来完成，因为属性被设置。请参阅*使用可设置属性更新急切属性*的示例。也可以使用惰性属性来完成。请参阅*使用惰性属性*的示例。我们已经展示了使用`types.SimpleNamespace`的示例，其中计算是在与类定义分开的函数中完成的。

+   **不可变对象**：这意味着丰富过程从旧对象创建新对象。不可变对象源自`tuple`或由`namedtuple()`创建的类型。这些对象的优势在于非常小且非常快。此外，缺乏任何内部状态变化使它们非常简单。

假设我们有一系列大块文本中的数据样本：

```py
    text_1 = '''10  8.04 
    8       6.95 
    13      7.58 
    ... 
    5       5.68 
    ''' 

```

我们的目标是一个包括`get`、`cleanse`和`enrich`操作的三步过程：

```py
    data = list(enrich(cleanse(get(text)))) 

```

`get()`函数从源获取数据；在这种情况下，它会解析大块文本。`cleanse()`函数将删除空行和其他无法使用的数据。`enrich()`函数将对清理后的数据进行最终计算。我们将分别查看此管道的每个阶段。

`get()`函数仅限于纯文本处理，尽量少地进行过滤：

```py
    from typing import * 

    def get(text: str) -> Iterator[List[str]]: 
        for line in text.splitlines(): 
            if len(line) == 0: 
                continue 
            yield line.split() 

```

为了编写类型提示，我们已导入了`typing`模块。这使我们能够对此函数的输入和输出进行明确声明。`get()`函数接受一个字符串`str`。它产生一个`List[str]`结构。输入的每一行都被分解为一系列值。

这个函数将生成所有非空数据行。这里有一个小的过滤功能，但它与数据序列化的一个小技术问题有关，而不是一个特定于应用程序的过滤规则。

`cleanse()`函数将生成命名元组的数据。这将应用一些规则来确保数据是有效的：

```py
    from collections import namedtuple 

    DataPair = namedtuple('DataPair', ['x', 'y']) 

    def cleanse(iterable: Iterable[List[str]]) -> Iterator[DataPair]: 
        for text_items in iterable: 
            try: 
                x_amount = float(text_items[0]) 
                y_amount = float(text_items[1]) 
                yield DataPair(x_amount, y_amount) 
            except Exception as ex: 
                print(ex, repr(text_items)) 

```

我们定义了一个`namedtuple`，名为`DataPair`。这个项目有两个属性，`x`和`y`。如果这两个文本值可以正确转换为`float`，那么这个生成器将产生一个有用的`DataPair`实例。如果这两个文本值无法转换，这将显示一个错误，指出有问题的对。

注意`mypy`项目类型提示中的技术细微之处。带有`yield`语句的函数是一个迭代器。由于正式关系，我们可以将其用作可迭代对象，这种关系表明迭代器是可迭代对象的一种。

这里可以应用额外的清洗规则。例如，`assert`语句可以添加到`try`语句中。任何由意外或无效数据引发的异常都将停止处理给定输入行。

这个初始的`cleanse()`和`get()`处理的结果如下：

```py
    list(cleanse(get(text))) 
    The output looks like this: 
    [DataPair(x=10.0, y=8.04), 
     DataPair(x=8.0, y=6.95), 
     DataPair(x=13.0, y=7.58), 
     ..., 
     DataPair(x=5.0, y=5.68)] 

```

在这个例子中，我们将按每对的`y`值进行排名。这需要首先对数据进行排序，然后产生排序后的值，并陦一个额外的属性值，即`y`排名顺序。

## 如何做...

1.  定义丰富的`namedtuple`：

```py
        RankYDataPair = namedtuple('RankYDataPair', ['y_rank', 'pair']) 

```

请注意，我们特意在这个新的数据结构中将原始对作为数据项包含在内。我们不想复制各个字段；相反，我们将原始对象作为一个整体合并在一起。

1.  定义丰富函数：

```py
        PairIter = Iterable[DataPair] 
        RankPairIter = Iterator[RankYDataPair] 

        def rank_by_y(iterable:PairIter) -> RankPairIter: 

```

我们在这个函数中包含了类型提示，以清楚地表明这个丰富函数期望和返回的类型。我们单独定义了类型提示，这样它们会更短，并且可以在其他函数中重复使用。

1.  编写丰富的主体。在这种情况下，我们将进行排名排序，因此我们需要使用原始`y`属性进行排序。我们从旧对象创建新对象，因此函数会生成`RankYDataPair`的实例：

```py
        all_data = sorted(iterable, key=lambda pair:pair.y) 
        for y_rank, pair in enumerate(all_data, start=1): 
            yield RankYDataPair(y_rank, pair) 

```

我们使用`enumerate()`为每个值创建排名顺序号。对于一些统计处理来说，起始值为`1`有时很方便。在其他情况下，默认的起始值`0`也能很好地工作。

整个函数如下：

```py
    def rank_by_y(iterable: PairIter) -> RankPairIter: 
        all_data = sorted(iterable, key=lambda pair:pair.y) 
        for y_rank, pair in enumerate(all_data, start=1): 
            yield RankYDataPair(y_rank, pair) 

```

我们可以在一个更长的表达式中使用它来获取、清洗，然后排名。使用类型提示可以使这一点比涉及有状态对象的替代方案更清晰。在某些情况下，代码的清晰度可能会有很大的改进。

## 它是如何工作的...

`rank_by_y()`函数的结果是一个包含原始对象和丰富结果的新对象。这是我们如何使用这个堆叠的生成器序列的：`rank_by_y()`，`cleanse()`和`get()`：

```py
 **>>> data = rank_by_y(cleanse(get(text_1))) 
>>> pprint(list(data)) 
[RankYDataPair(y_rank=1, pair=DataPair(x=4.0, y=4.26)), 
 RankYDataPair(y_rank=2, pair=DataPair(x=7.0, y=4.82)), 
 RankYDataPair(y_rank=3, pair=DataPair(x=5.0, y=5.68)), 
 ..., 
 RankYDataPair(y_rank=11, pair=DataPair(x=12.0, y=10.84))]** 

```

数据按`y`值升序排列。我们现在可以使用这些丰富的数据值进行进一步的分析和计算。

在许多情况下，创建新对象可能更能表达算法，而不是改变对象的状态。这通常是一个主观的判断。

Python 类型提示最适合用于创建新对象。因此，这种技术可以提供强有力的证据，证明复杂的算法是正确的。使用`mypy`可以使不可变对象更具吸引力。

最后，当我们使用不可变对象时，有时会看到一些小的加速。这依赖于 Python 的三个特性之间的平衡才能有效：

+   元组是小型数据结构。使用它们可以提高性能。

+   Python 中对象之间的任何关系都涉及创建对象引用，这是一个非常小的数据结构。一系列相关的不可变对象可能比一个可变对象更小。

+   对象的创建可能是昂贵的。创建太多不可变对象会超过其好处。

前两个功能带来的内存节省必须与第三个功能带来的处理成本相平衡。当存在大量数据限制处理速度时，内存节省可以带来更好的性能。

对于像这样的小例子，数据量非常小，对象创建成本与减少内存使用量的任何成本节省相比都很大。对于更大的数据集，对象创建成本可能小于内存不足的成本。

## 还有更多...

这个配方中的`get()`和`cleanse()`函数都涉及到类似的数据结构：`Iterable[List[str]]`和`Iterator[List[str]]`。在`collections.abc`模块中，我们看到`Iterable`是通用定义，而`Iterator`是`Iterable`的特殊情况。

用于本书的`mypy`版本——`mypy 0.2.0-dev`——对具有`yield`语句的函数被定义为`Iterator`非常严格。未来的版本可能会放宽对子类关系的严格检查，允许我们在两种情况下使用同一定义。

`typing`模块包括`namedtuple()`函数的替代品：`NamedTuple()`。这允许对元组中的各个项目进行数据类型的指定。

看起来是这样的：

```py
    DataPair = NamedTuple('DataPair', [ 
            ('x', float), 
            ('y', float) 
        ] 
    ) 

```

我们几乎可以像使用`collection.namedtuple()`一样使用`typing.NamedTuple()`。属性的定义使用了一个两元组的列表，而不是名称的列表。两元组有一个名称和一个类型定义。

这个补充类型定义被`mypy`用来确定`NamedTuple`对象是否被正确填充。其他人也可以使用它来理解代码并进行适当的修改或扩展。

在 Python 中，我们可以用不可变对象替换一些有状态的对象。但是有一些限制。例如，列表、集合和字典等集合必须保持为可变对象。在其他编程语言中，用不可变的单子替换这些集合可能效果很好，但在 Python 中不是这样的。

![](img/614271.jpg)

# 使用 yield from 语句编写递归生成器函数

有许多算法可以清晰地表达为递归。在*围绕 Python 的堆栈限制设计递归函数*配方中，我们看了一些可以优化以减少函数调用次数的递归函数。

当我们查看一些数据结构时，我们发现它们涉及递归。特别是，JSON 文档（以及 XML 和 HTML 文档）可以具有递归结构。JSON 文档可能包含一个包含其他复杂对象的复杂对象。

在许多情况下，使用生成器处理这些类型的结构有很多优势。我们如何编写能够处理递归的生成器？`yield from`语句如何避免我们编写额外的循环？

## 准备工作

我们将看一种在复杂数据结构中搜索有序集合的所有匹配值的方法。在处理复杂的 JSON 文档时，我们经常将它们建模为字典-字典和字典-列表结构。当然，JSON 文档不是一个两级的东西；字典-字典实际上意味着字典-字典-字典...同样，字典-列表实际上意味着字典-列表-...这些都是递归结构，这意味着搜索必须遍历整个结构以寻找特定的键或值。

具有这种复杂结构的文档可能如下所示：

```py
    document = { 
        "field": "value1", 
        "field2": "value", 
        "array": [ 
            {"array_item_key1": "value"}, 
            {"array_item_key2": "array_item_value2"} 
        ], 
        "object": { 
            "attribute1": "value", 
            "attribute2": "value2" 
        } 
    } 

```

这显示了一个具有四个键`field`、`field2`、`array`和`object`的文档。每个键都有一个不同的数据结构作为其关联值。一些值是唯一的，一些是重复的。这种重复是我们的搜索必须在整个文档中找到**所有**实例的原因。

核心算法是深度优先搜索。这个函数的输出将是一个标识目标值的路径列表。每个路径将是一系列字段名或字段名与索引位置混合的序列。

在前面的例子中，值`value`可以在三个地方找到：

+   `["array", 0, "array_item_key1"]`：这个路径从名为`array`的顶级字段开始，然后访问列表的第零项，然后是一个名为`array_item_key1`的字段

+   `["field2"]`：这个路径只有一个字段名，其中找到了值

+   `["object", "attribute1"]`：这个路径从名为`object`的顶级字段开始，然后是该字段的子`attribute1`

`find_value()`函数在搜索整个文档寻找目标值时，会产生这两个路径。这是这个搜索函数的概念概述：

```py
    def find_path(value, node, path=[]): 
        if isinstance(node, dict): 
            for key in node.keys(): 
                # find_value(value, node[key], path+[key]) 
                # This must yield multiple values 
        elif isinstance(node, list): 
            for index in range(len(node)): 
                # find_value(value, node[index], path+[index]) 
                # This will yield multiple values 
        else: 
            # a primitive type 
            if node == value: 
                yield path 

```

在`find_path()`过程中有三种选择：

+   当节点是一个字典时，必须检查每个键的值。值可以是任何类型的数据，因此我们将对每个值递归使用`find_path()`函数。这将产生一系列匹配。

+   如果节点是一个列表，必须检查每个索引位置的项目。项目可以是任何类型的数据，因此我们将对每个值递归使用`find_path()`函数。这将产生一系列匹配。

+   另一种选择是节点是一个原始值。JSON 规范列出了可能出现在有效文档中的许多原始值。如果节点值是目标值，我们找到了一个实例，并且可以产生这个单个匹配。

处理递归有两种方法。一种是这样的：

```py
    for match in find_value(value, node[key], path+[key]): 
        yield match 

```

对于这样一个简单的想法来说，这似乎有太多的样板。另一种方法更简单，也更清晰一些。

## 如何做...

1.  写出完整的`for`语句：

```py
        for match in find_value(value, node[key], path+[key]): 
            yield match 

```

出于调试目的，我们可以在`for`语句的主体中插入一个`print()`函数。

1.  一旦确定事情运行正常，就用`yield from`语句替换这个：

```py
        yield from find_value(value, node[key], path+[key]) 

```

完整的深度优先`find_value()`搜索函数将如下所示：

```py
    def find_path(value, node, path=[]): 
        if isinstance(node, dict): 
            for key in node.keys(): 
                yield from find_path(value, node[key], path+[key]) 
        elif isinstance(node, list): 
            for index in range(len(node)): 
                yield from find_path(value, node[index], path+[index]) 
        else: 
            if node == value: 
                yield path 

```

当我们使用`find_path()`函数时，它看起来像这样：

```py
 **>>> list(find_path('array_item_value2', document)) 
[['array', 1, 'array_item_key2']]** 

```

`find_path()`函数是可迭代的。它可以产生许多值。我们消耗了所有的结果来创建一个列表。在这个例子中，列表只有一个项目，`['array', 1, 'array_item_key2']`。这个项目有指向匹配项的路径。

然后我们可以评估`document['array'][1]['array_item_key2']`来找到被引用的值。

当我们寻找非唯一值时，我们可能会看到这样的列表：

```py
 **>>> list(find_value('value', document)) 
[['array', 0, 'array_item_key1'], 
 ['field2'], 
 ['object', 'attribute1']]** 

```

结果列表有三个项目。每个项目都提供了指向目标值`value`的路径。

## 它是如何工作的...

`yield from X`语句是以下内容的简写：

```py
    for item in X: 
        yield item 

```

这使我们能够编写一个简洁的递归算法，它将作为迭代器运行，并正确地产生多个值。

这也可以在不涉及递归函数的情况下使用。在涉及可迭代结果的任何地方使用`yield from`语句都是完全合理的。然而，对于递归函数来说，这是一个很大的简化，因为它保留了一个明确的递归结构。

## 还有更多...

另一种常见的定义风格是使用追加操作组装列表。我们可以将这个重写为迭代器，避免构建列表对象的开销。

当分解一个数字时，我们可以这样定义质因数集：

![还有更多...](img/Image00026.jpg)

如果值*x*是质数，它在质因数集中只有自己。否则，必须存在某个质数*n*，它是*x*的最小因数。我们可以从*n*开始组装一个因数集，并包括*x/n*的所有因数。为了确保只找到质因数，*n*必须是质数。如果我们按升序搜索，我们会在找到复合因数之前找到质因数。

我们有两种方法在 Python 中实现这个：一种是构建一个列表，另一种是生成因数。这是一个构建列表的函数：

```py
    import math 
    def factor_list(x): 
        limit = int(math.sqrt(x)+1) 
        for n in range(2, limit): 
            q, r = divmod(x, n) 
            if r == 0: 
                return [n] + factor_list(q) 
        return [x] 

```

这个`factor_list()`函数将搜索所有数字*n*，使得 2 ≤ *n* < √ *x*。找到*x*的第一个因子的数字将是最小的因子。它也将是质数。当然，我们会搜索一些复合值，浪费时间。例如，在测试了二和三之后，我们还将测试四和六这样的值，尽管它们是复合数，它们的所有因子都已经被测试过了。

这个函数构建了一个`list`对象。如果找到一个因子`n`，它将以该因子开始一个列表。它将从`x // n`添加因子。如果没有`x`的因子，那么这个值是质数，我们将返回一个只包含该值的列表。

我们可以通过用`yield from`替换递归调用来将其重写为迭代器。函数将看起来像这样：

```py
    def factor_iter(x): 
        limit = int(math.sqrt(x)+1) 
        for n in range(2, limit): 
            q, r = divmod(x, n) 
            if r == 0: 
                yield n 
                yield from factor_iter(q) 
                return 
        yield x 

```

与构建列表版本一样，这将搜索数字*n*，使得。当找到一个因子时，函数将产生该因子，然后通过对`factor_iter()`的递归调用找到任何其他因子。如果没有找到因子，函数将只产生质数，没有其他东西。

使用迭代器可以让我们从因子构建任何类型的集合。我们不再局限于总是创建一个*list*，而是可以使用`collection.Counter`类创建一个多重集。它看起来像这样：

```py
 **>>> from collections import Counter 
>>> Counter(factor_iter(384)) 
Counter({2: 7, 3: 1})** 

```

这向我们表明：

384 = 2⁷ × 3

在某些情况下，这种多重集比因子列表更容易处理。

## 另请参阅

+   在*围绕 Python 的堆栈限制设计递归函数*的配方中，我们涵盖了递归函数的核心设计模式。这个配方提供了创建结果的另一种方法。
