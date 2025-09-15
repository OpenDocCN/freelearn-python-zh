# 迭代器模式

我们已经讨论了 Python 的许多内置函数和惯用法乍一看似乎与面向对象原则相悖，但实际上是在底层提供对真实对象的访问。在本章中，我们将讨论看似结构化的`for`循环实际上是如何围绕一组面向对象原则的轻量级包装。我们还将看到对这个语法的各种扩展，这些扩展可以自动创建更多类型的对象。我们将涵盖以下主题：

+   什么是设计模式

+   迭代器协议——最强大的设计模式之一

+   列表、集合和字典推导式

+   生成器和协程

# 简要介绍设计模式

当工程师和建筑师决定建造一座桥梁、一座塔或一座建筑时，他们会遵循某些原则以确保结构完整性。桥梁（例如，悬索桥和悬臂桥）有各种可能的设计，但如果工程师不使用标准设计，也没有一个出色的全新设计，那么他/她设计的桥梁很可能会倒塌。

设计模式试图将这种对正确设计的正式定义应用到软件工程中。有许多不同的设计模式来解决不同的普遍问题。设计模式通常解决开发者在某些特定情况下面临的具体常见问题。设计模式随后是对该问题的理想解决方案的建议，从面向对象设计的角度来说。 

然而，了解设计模式并选择在我们的软件中使用它并不能保证我们正在创建一个*正确*的解决方案。1907 年，魁北克桥（至今仍是世界上最长的悬臂桥）在建设完成前就倒塌了，因为设计它的工程师们极大地低估了用于建造它的钢材重量。同样，在软件开发中，我们可能会错误地选择或应用设计模式，并创建出在正常操作情况下或在超出原始设计极限的压力下会*崩溃*的软件。

任何一种设计模式都提出了一组以特定方式交互的对象，以解决一个普遍问题。程序员的任务是识别他们面临的是这种问题的特定版本，然后选择并调整通用设计以适应他们的精确需求。

在本章中，我们将介绍迭代器设计模式。这个模式非常强大且普遍，以至于 Python 开发者提供了多种语法来访问模式背后的面向对象原则。我们将在下一章中介绍其他设计模式。其中一些有语言支持，而另一些则没有，但没有任何一个模式像迭代器模式那样内在地成为 Python 程序员日常生活的组成部分。

# 迭代器

在典型的设计模式术语中，迭代器是一个具有 `next()` 方法和 `done()` 方法的对象；后者如果序列中没有剩余的项目则返回 `True`。在没有内置迭代器支持的编程语言中，迭代器会被像这样遍历：

```py
while not iterator.done(): 
    item = iterator.next() 
    # do something with the item 
```

在 Python 中，迭代是一个特殊特性，因此该方法有一个特殊名称，即 `__next__`。此方法可以通过 `next(iterator)` 内置函数访问。而不是 `done` 方法，Python 的迭代器协议通过抛出 `StopIteration` 来通知循环它已经完成。最后，我们有更易读的 `for item in iterator` 语法来实际访问迭代器中的项目，而不是在 `while` 循环中纠缠不清。让我们更详细地看看这些。

# 迭代器协议

`Iterator` 抽象基类，位于 `collections.abc` 模块中，定义了 Python 中的迭代协议。正如之前提到的，它必须有一个 `__next__` 方法，`for` 循环（以及其他支持迭代的特性）可以通过调用该方法从序列中获取新的元素。此外，每个迭代器还必须满足 `Iterable` 接口。任何提供了 `__iter__` 方法的类都是可迭代的。该方法必须返回一个将覆盖该类中所有元素的 `Iterator` 实例。

这可能听起来有些令人困惑，所以请看一下下面的示例，但请注意，这是一种非常冗长的解决问题的方式。它清楚地解释了迭代和相关的两个协议，但我们在本章后面将探讨几种更易读的方式来实现这一效果：

```py
class CapitalIterable: 
    def __init__(self, string): 
        self.string = string 

 def __iter__(self): 
        return CapitalIterator(self.string) 

class CapitalIterator: 
    def __init__(self, string): 
        self.words = [w.capitalize() for w in string.split()] 
        self.index = 0 

 def __next__(self): 
        if self.index == len(self.words): 
 raise StopIteration() 

        word = self.words[self.index] 
        self.index += 1 
        return word 

    def __iter__(self): 
        return self 
```

此示例定义了一个 `CapitalIterable` 类，其任务是遍历字符串中的每个单词，并将它们以首字母大写的方式输出。该可迭代对象的大部分工作都传递给了 `CapitalIterator` 实现。与这个迭代器交互的规范方式如下：

```py
>>> iterable = CapitalIterable('the quick brown fox jumps over the lazy dog')
>>> iterator = iter(iterable)
>>> while True:
...     try:
...         print(next(iterator))
...     except StopIteration:
...         break
... 
The
Quick
Brown
Fox
Jumps
Over
The
Lazy
Dog  
```

此示例首先构建了一个可迭代对象，并从中检索了一个迭代器。这种区别可能需要解释；可迭代对象是一个具有可以遍历的元素的对象。通常，这些元素可以被多次遍历，甚至可能在同一时间或重叠的代码中。另一方面，迭代器代表可迭代对象中的特定位置；一些项目已经被消耗，而一些还没有。两个不同的迭代器可能在单词列表的不同位置，但任何一个迭代器只能标记一个位置。

每次在迭代器上调用 `next()` 时，它都会按顺序返回可迭代对象中的另一个标记。最终，迭代器将会耗尽（没有更多元素可以返回），在这种情况下，会抛出 `Stopiteration` 异常，然后我们退出循环。

当然，我们已经有了一种更简单的语法来从可迭代对象中构建迭代器：

```py
>>> for i in iterable:
...     print(i)
... 
The
Quick
Brown
Fox
Jumps
Over
The
Lazy
Dog  
```

如您所见，尽管`for`语句看起来并不像面向对象，但实际上它是某些显然面向对象设计原则的快捷方式。在我们讨论理解时，请记住这一点，因为它们也似乎与面向对象工具完全相反。然而，它们使用与`for`循环完全相同的迭代协议，只是另一种快捷方式。

# 理解

理解是简单但强大的语法，允许我们在一行代码中转换或过滤可迭代对象。结果对象可以是一个完全正常的列表、集合或字典，或者它可以是生成器表达式，可以在保持一次只保留一个元素在内存中的同时高效地消费。

# 列表理解

列表理解是 Python 中最强大的工具之一，因此人们倾向于认为它们是高级的。实际上并非如此。实际上，我已经在之前的例子中随意使用了理解，假设你会理解它们。虽然高级程序员确实经常使用理解，但这并不是因为它们是高级的。这是因为它们很简单，并且处理软件开发中最常见的操作。

让我们看看那些常见的操作之一；即，将一个项目的列表转换为相关项目的列表。具体来说，假设我们刚刚从一个文件中读取了一个字符串列表，现在我们想要将其转换为整数列表。我们知道列表中的每个项目都是一个整数，我们想要对这些数字进行一些操作（比如，计算平均值）。这里有简单的一种方法来处理它：

```py
input_strings = ["1", "5", "28", "131", "3"]

output_integers = [] 
for num in input_strings: 
    output_integers.append(int(num)) 
```

这工作得很好，而且只有三行代码。如果你不习惯使用理解，你可能甚至不会觉得它看起来很丑！现在，看看使用列表理解的相同代码：

```py
input_strings = ["1", "5", "28", "131", "3"]
output_integers = [int(num) for num in input_strings] 
```

我们只剩下一行，并且对于性能来说非常重要，我们已经为列表中的每个项目省略了`append`方法调用。总的来说，即使你不习惯理解语法，也很容易看出发生了什么。

方括号，一如既往地表示我们正在创建一个列表。在这个列表内部有一个`for`循环，它会遍历输入序列中的每个项目。可能让人困惑的是列表开括号和`for`循环开始之间发生了什么。这里发生的任何操作都会应用到输入列表中的每个项目上。当前的项目通过循环中的`num`变量来引用。因此，它为每个元素调用`int`函数，并将结果整数存储在新列表中。

基本列表理解就是这样。理解是高度优化的 C 代码；列表理解在遍历大量项目时比`for`循环要快得多。如果仅仅从可读性来看不足以说服你尽可能多地使用它们，那么速度应该可以。

将一个项目列表转换为相关列表并不是列表推导所能做的唯一事情。我们还可以选择通过在推导中添加一个`if`语句来排除某些值。看看这个例子：

```py
output_integers = [int(num) for num in input_strings if len(num) < 3]
```

与前一个例子相比，唯一不同的是`if len(num) < 3`这部分。这段额外的代码排除了任何超过两个字符的字符串。`if`语句是在`int`函数**之前**应用于每个元素的，因此它是在测试字符串的长度。由于我们的输入字符串本质上都是整数，所以它排除了任何大于 99 的数字。

列表推导用于将输入值映射到输出值，同时在过程中应用一个过滤器来包含或排除任何满足特定条件的值。

任何可迭代对象都可以作为列表推导的输入。换句话说，我们可以将任何可以放在`for`循环中的东西也放在推导中。例如，文本文件是可迭代的；文件迭代器的每次`__next__`调用都会返回文件的一行。我们可以使用`zip`函数将带有标题行的制表符分隔文件加载到字典中：

```py
import sys

filename = sys.argv[1]

with open(filename) as file:
    header = file.readline().strip().split("\t")
 contacts = [
 dict(
 zip(header, line.strip().split("\t")))
 for line in file
 ]

for contact in contacts:
    print("email: {email} -- {last}, {first}".format(**contact))

```

这次，我添加了一些空白，使其更易于阅读（列表推导**不**一定要放在一行上）。这个例子创建了一个从文件中每个行的压缩标题和分割行生成的字典列表。

哎，什么？如果这段代码或解释让你感到困惑，请不要担心；它确实很复杂。一个列表推导在这里做了很多工作，代码难以理解、阅读，最终也难以维护。这个例子表明列表推导并不总是最佳解决方案；大多数程序员都会同意使用`for`循环比这个版本更易读。

记住：我们提供的工具不应该被滥用！始终选择适合工作的正确工具，那就是始终编写可维护的代码。

# 集合和字典推导

推导不仅限于列表。我们还可以使用类似的大括号语法来创建集合和字典。让我们从集合开始。创建集合的一种方法是将列表推导包裹在`set()`构造函数中，将其转换为集合。但为什么要在被丢弃的中间列表上浪费内存，当我们可以直接创建集合时？

下面是一个使用命名元组来模拟作者/标题/类型三元组的例子，然后检索特定类型的所有作者集合：

```py
from collections import namedtuple

Book = namedtuple("Book", "author title genre")
books = [
    Book("Pratchett", "Nightwatch", "fantasy"),
    Book("Pratchett", "Thief Of Time", "fantasy"),
    Book("Le Guin", "The Dispossessed", "scifi"),
    Book("Le Guin", "A Wizard Of Earthsea", "fantasy"),
    Book("Turner", "The Thief", "fantasy"),
    Book("Phillips", "Preston Diamond", "western"),
    Book("Phillips", "Twice Upon A Time", "scifi"),
]

fantasy_authors = {b.author for b in books if b.genre == "fantasy"}

```

与演示数据的设置相比，高亮的集合推导确实要短得多！如果我们使用列表推导，当然，特里·普拉切特会被列出两次。但事实上，集合的性质消除了重复项，我们最终得到以下结果：

```py
>>> fantasy_authors
{'Turner', 'Pratchett', 'Le Guin'}  
```

仍然使用大括号，我们可以通过添加冒号来创建字典推导式。这会将序列转换为使用*键:值*对的字典。例如，如果我们知道标题，快速查找作者或类型可能很有用。我们可以使用字典推导式将标题映射到`books`对象：

```py
fantasy_titles = {b.title: b for b in books if b.genre == "fantasy"}
```

现在，我们有一个字典，并且可以使用正常语法通过标题查找书籍。

总结来说，列表推导不是高级 Python，也不是应该避免的*非面向对象*工具。它们只是从现有序列创建列表、集合或字典的更简洁和优化的语法。

# 生成器表达式

有时我们想要处理一个新的序列，而不需要将新的列表、集合或字典拉入系统内存。如果我们只是逐个遍历项目，并且实际上不关心创建一个完整的容器（如列表或字典），那么创建该容器就是浪费内存。在逐个处理项目时，我们只需要在任何时刻内存中可用的当前对象。但是，当我们创建一个容器时，所有对象都必须在开始处理之前存储在该容器中。

例如，考虑一个处理日志文件的程序。一个非常简单的日志可能包含以下格式的信息：

```py
Jan 26, 2015 11:25:25 DEBUG This is a debugging message. Jan 26, 2015 11:25:36 INFO This is an information method. Jan 26, 2015 11:25:46 WARNING This is a warning. It could be serious. Jan 26, 2015 11:25:52 WARNING Another warning sent. Jan 26, 2015 11:25:59 INFO Here's some information. Jan 26, 2015 11:26:13 DEBUG Debug messages are only useful if you want to figure something out. Jan 26, 2015 11:26:32 INFO Information is usually harmless, but helpful. Jan 26, 2015 11:26:40 WARNING Warnings should be heeded. Jan 26, 2015 11:26:54 WARNING Watch for warnings. 
```

流行网络服务器、数据库或电子邮件服务器的日志文件可以包含许多 GB 的数据（我曾经不得不从一个行为不端的系统中清理近 2TB 的日志）。如果我们想要处理日志中的每一行，我们不能使用列表推导；它将创建一个包含文件中每一行的列表。这可能不会适合 RAM，并且可能会根据操作系统使计算机瘫痪。

如果我们在日志文件上使用`for`循环，我们可以逐行处理，在将下一行读入内存之前。如果我们可以使用推导式语法达到相同的效果，那岂不是很好？

这就是生成器表达式发挥作用的地方。它们使用与推导式相同的语法，但不会创建一个最终的容器对象。要创建一个生成器表达式，将推导式用`()`括起来而不是`[]`或`{}`。

以下代码解析了之前展示格式的日志文件，并输出一个只包含`WARNING`行的新的日志文件：

```py
import sys 

inname = sys.argv[1] 
outname = sys.argv[2] 

with open(inname) as infile: 
    with open(outname, "w") as outfile: 
 warnings = (l for l in infile if 'WARNING' in l) 
        for l in warnings: 
            outfile.write(l) 
```

这个程序接受命令行上的两个文件名，使用生成器表达式过滤掉警告（在这种情况下，它使用`if`语法并保留行不变），然后将警告输出到另一个文件。如果我们运行它在我们提供的样本文件上，输出看起来像这样：

```py
Jan 26, 2015 11:25:46 WARNING This is a warning. It could be serious.
Jan 26, 2015 11:25:52 WARNING Another warning sent.
Jan 26, 2015 11:26:40 WARNING Warnings should be heeded.
Jan 26, 2015 11:26:54 WARNING Watch for warnings. 
```

当然，对于如此短的输入文件，我们可以安全地使用列表推导，但如果文件有数百万行长，生成器表达式将对内存和速度产生巨大影响。

将`for`表达式用括号括起来创建的是一个生成器表达式，而不是一个元组。

生成器表达式通常在函数调用中非常有用。例如，我们可以对生成器表达式调用 `sum`、`min` 或 `max` 而不是列表，因为这些函数一次处理一个对象。我们只对聚合结果感兴趣，而不是任何中间容器。

通常情况下，在四个选项中，只要可能就应该使用生成器表达式。如果我们实际上不需要列表、集合或字典，但只需要过滤或转换序列中的项，生成器表达式将是最有效的。如果我们需要知道列表的长度，或者对结果进行排序、删除重复项或创建字典，我们就必须使用理解语法。

# 生成器

生成器表达式实际上也是一种理解方式；它们将更高级的（这次确实是更高级！）生成器语法压缩成一行。更高级的生成器语法看起来甚至比我们见过的任何东西都更不面向对象，但我们会再次发现，这只是一个简单的语法快捷方式来创建一种对象。

让我们进一步探讨日志文件示例。如果我们想从输出文件中删除 `WARNING` 列（因为它重复了：这个文件只包含警告），我们有几种不同可读性的选项。我们可以用生成器表达式来做这件事：

```py
import sys

# generator expression
inname, outname = sys.argv[1:3]

with open(inname) as infile:
    with open(outname, "w") as outfile:
 warnings = (
 l.replace("\tWARNING", "") for l in infile if "WARNING" in l
 )
        for l in warnings:
            outfile.write(l)
```

这完全是可以读的，尽管我不希望将表达式复杂化到那种程度。我们也可以用普通的 `for` 循环来做这件事：

```py
with open(inname) as infile:
    with open(outname, "w") as outfile:
        for l in infile:
            if "WARNING" in l:
                outfile.write(l.replace("\tWARNING", ""))
```

这显然是可维护的，但在这么少的行中却有很多缩进级别，看起来有点丑陋。更令人担忧的是，如果我们想做的不仅仅是打印行，我们还得复制循环和条件代码。

现在让我们考虑一个真正面向对象且没有捷径的解决方案：

```py
class WarningFilter:
    def __init__(self, insequence):
        self.insequence = insequence

 def __iter__(self):
        return self

 def __next__(self):
        l = self.insequence.readline()
        while l and "WARNING" not in l:
            l = self.insequence.readline()
        if not l:
 raise StopIteration
        return l.replace("\tWARNING", "")

with open(inname) as infile:
    with open(outname, "w") as outfile:
        filter = WarningFilter(infile)
        for l in filter:
            outfile.write(l)
```

毫无疑问：这看起来如此丑陋且难以阅读，你可能甚至都无法弄清楚发生了什么。我们创建了一个以文件对象为输入的对象，并提供了一个像任何迭代器一样的 `__next__` 方法。

这个 `__next__` 方法从文件中读取行，如果它们不是 `WARNING` 行则忽略它们。当我们遇到 `WARNING` 行时，我们修改并返回它。然后我们的 `for` 循环再次调用 `__next__` 来处理后续的 `WARNING` 行。当我们没有更多的行时，我们引发 `StopIteration` 来告诉循环我们已经完成了迭代。与其它例子相比，这看起来相当丑陋，但它也很强大；现在我们手中有一个类，我们可以用它来做任何事情。

在有了这个背景知识之后，我们终于可以看到真正的生成器在行动了。接下来的这个例子与上一个例子**完全**一样：它创建了一个具有 `__next__` 方法的对象，当它没有输入时将引发 `StopIteration`：

```py
def warnings_filter(insequence):
    for l in insequence:
        if "WARNING" in l:
 yield l.replace("\tWARNING", "")

with open(inname) as infile:
    with open(outname, "w") as outfile:
        filter = warnings_filter(infile)
        for l in filter:
            outfile.write(l)
```

好吧，这看起来挺容易读的，也许吧... 至少它很短。但这里到底发生了什么？这完全说不通。那么，`yield` 究竟是什么意思呢？

事实上，`yield` 是生成器的关键。当 Python 在一个函数中看到 `yield` 时，它会将这个函数包装成一个对象，这个对象与我们在之前的例子中看到的不太一样。将 `yield` 语句视为与 `return` 语句类似；它退出函数并返回一行。然而，与 `return` 不同的是，当函数再次被调用（通过 `next()`）时，它将从上次离开的地方开始——在 `yield` 语句之后的行——而不是从函数的开始处。在这个例子中，`yield` 语句之后没有行，所以它跳到 `for` 循环的下一个迭代。由于 `yield` 语句在 `if` 语句内部，它只产生包含 `WARNING` 的行。

虽然看起来这个函数只是在遍历行，但实际上它正在创建一个特殊类型的对象，一个生成器对象：

```py
>>> print(warnings_filter([]))
<generator object warnings_filter at 0xb728c6bc>  
```

我将一个空列表传递给函数，作为迭代器使用。这个函数所做的只是创建并返回一个生成器对象。这个对象上有 `__iter__` 和 `__next__` 方法，就像我们在之前的例子中创建的那样。（你可以通过调用 `dir` 内置函数来确认。）每当调用 `__next__` 时，生成器会运行函数，直到找到 `yield` 语句。然后它返回 `yield` 的值，下一次调用 `__next__` 时，它会从上次离开的地方继续。

这种生成器的用法并不复杂，但如果你不意识到函数正在创建一个对象，它可能会显得像魔法一样。这个例子相当简单，但通过在单个函数中多次调用 `yield`，你可以得到非常强大的效果；在每次循环中，生成器将简单地从最近的 `yield` 处开始，并继续到下一个。

# 从另一个可迭代对象中产生项目

通常，当我们构建一个生成器函数时，我们会陷入一个想要从另一个可迭代对象（可能是我们在生成器内部构建的列表推导式或生成器表达式，或者可能是传递给函数的外部项目）产生数据的局面。这始终是通过遍历可迭代对象并逐个产生每个项目来实现的。然而，在 Python 3.3 版本中，Python 开发者引入了一种新的语法，使其更加优雅。

让我们稍微修改一下生成器例子，让它接受一个文件名而不是一系列行。这通常会被认为是不好的做法，因为它将对象绑定到特定的范式。当可能的时候，我们应该操作迭代器作为输入；这样，无论日志行是从文件、内存还是网络中来的，都可以使用相同的函数。

这段代码的版本说明了你的生成器可以在从另一个可迭代对象（在这种情况下，是一个生成器表达式）产生信息之前做一些基本的设置：

```py
def warnings_filter(infilename):
    with open(infilename) as infile:
 yield from (
 l.replace("\tWARNING", "") for l in infile if "WARNING" in l
 )

filter = warnings_filter(inname)
with open(outname, "w") as outfile:
    for l in filter:
        outfile.write(l)
```

这段代码将之前例子中的 `for` 循环结合成了一个生成器表达式。注意，这种转换并没有带来任何帮助；之前的 `for` 循环例子更易于阅读。

因此，让我们考虑一个比它的替代方案更易读的例子。构建一个生成器，从多个其他生成器中产生数据，可能是有用的。例如，`itertools.chain`函数按顺序从可迭代对象中产生数据，直到它们全部耗尽。这可以通过`yield from`语法非常容易地实现，所以让我们考虑一个经典的计算机科学问题：遍历一般树。

一般树数据结构的一个常见实现是计算机的文件系统。让我们模拟 Unix 文件系统中的几个文件夹和文件，这样我们就可以使用`yield from`来有效地遍历它们：

```py
class File:
    def __init__(self, name):
        self.name = name

class Folder(File):
    def __init__(self, name):
        super().__init__(name)
        self.children = []

root = Folder("")
etc = Folder("etc")
root.children.append(etc)
etc.children.append(File("passwd"))
etc.children.append(File("groups"))
httpd = Folder("httpd")
etc.children.append(httpd)
httpd.children.append(File("http.conf"))
var = Folder("var")
root.children.append(var)
log = Folder("log")
var.children.append(log)
log.children.append(File("messages"))
log.children.append(File("kernel"))

```

这个设置代码看起来工作量很大，但在实际的文件系统中，它会更复杂。我们不得不从硬盘读取数据并将其结构化到树中。然而，一旦在内存中，输出文件系统中每个文件的代码相当优雅：

```py
def walk(file):
    if isinstance(file, Folder):
        yield file.name + "/"
        for f in file.children:
 yield from walk(f)
    else:
        yield file.name
```

如果这个代码遇到一个目录，它会递归地调用`walk()`来生成其每个子目录下所有文件的列表，然后产生所有这些数据加上它自己的文件名。在它遇到一个普通文件的情况下，它只产生那个名字。

作为旁白，在不使用生成器的情况下解决前面的问题已经足够棘手，以至于它成为了一个常见的面试问题。如果你像这样回答，准备好你的面试官既会印象深刻又会有些恼火，因为你回答得太容易了。他们可能会要求你解释到底发生了什么。当然，有了你在本章中学到的原则，你不会有任何问题。祝你好运！

`yield from`语法在编写链式生成器时是一个有用的快捷方式。它被添加到语言中是为了支持协程。然而，它现在并不怎么使用了，因为它的用法已经被`async`和`await`语法所取代。我们将在下一节中看到这两个语法的示例。

# 协程

协程是非常强大的构造，常常与生成器混淆。许多作者不适当地将协程描述为“带有额外语法的生成器”。这是一个容易犯的错误，因为，在 Python 2.5 中，协程首次引入时，它们被展示为“我们在生成器语法中添加了一个`send`方法”。实际上，这种差异要微妙得多，在你看过几个例子之后会更有意义。

协程相当难以理解。在`asyncio`模块之外，我们将在并发章节中介绍，它们在野外并不常用。你完全可以跳过这一节，并快乐地用 Python 开发多年而不必遇到协程。有几个库广泛使用协程（主要用于并发或异步编程），但它们通常被编写成你可以使用协程而不必真正理解它们是如何工作的！所以，如果你在这一节中迷路了，不要绝望。

如果我没有吓到你，让我们开始吧！这里有一个最简单的协程之一；它允许我们保持一个可以由任意值增加的累计计数：

```py
def tally(): 
    score = 0 
    while True: 
 increment = yield score 
        score += increment 
```

这段代码看起来像是黑魔法，不可能工作，所以在我们逐行描述之前，让我们证明它是有效的。这个简单的对象可以被用于棒球队伍的计分应用程序。可以为每个队伍分别记录计数，并在每个半局结束时累加的得分数量来增加他们的分数。看看这个交互会话：

```py
>>> white_sox = tally()
>>> blue_jays = tally()
>>> next(white_sox)
0
>>> next(blue_jays)
0
>>> white_sox.send(3)
3
>>> blue_jays.send(2)
2
>>> white_sox.send(2)
5
>>> blue_jays.send(4)
6  
```

首先，我们为每个队伍构建两个 `计数器` 对象。是的，它们看起来像函数，但就像前一部分中的生成器对象一样，函数内部存在 `yield` 语句的事实告诉 Python 要投入大量精力将这个简单的函数转换成一个对象。

然后，我们在每个协程对象上调用 `next()`。这与调用任何生成器的 `next()` 方法做的是同样的事情，也就是说，它执行每一行代码，直到遇到 `yield` 语句，返回该点的值，然后 *暂停*，直到下一个 `next()` 调用。

到目前为止，没有什么新的。但是看看我们协程中的 `yield` 语句：

```py
increment = yield score 
```

与生成器不同，这个 `yield` 函数看起来像是应该返回一个值并将其分配给一个变量。实际上，这正是正在发生的事情。协程仍然在 `yield` 语句处暂停，等待通过另一个 `next()` 调用再次被激活。

但是我们没有调用 `next()`。正如你在交互会话中看到的那样，我们而是调用了一个名为 `send()` 的方法。`send()` 方法与 `next()` 完全相同，除了它除了将生成器推进到下一个 `yield` 语句之外，还允许你从生成器外部传递一个值。这个值就是分配给 `yield` 语句左侧的值。

对于许多人来说，真正令人困惑的是这个发生的顺序：

1.  `yield` 发生并且生成器暂停

1.  `send()` 从函数外部发生，生成器醒来

1.  发送的值被分配给 `yield` 语句的左侧

1.  生成器继续处理，直到遇到另一个 `yield` 语句

因此，在这个特定的例子中，在我们构建协程并使用一次`next()`调用将其推进到`yield`语句之后，每次对`send()`的后续调用都会将一个值传递给协程。我们将这个值加到它的分数上。然后我们回到`while`循环的顶部，并继续处理，直到我们遇到`yield`语句。`yield`语句返回一个值，这成为我们最近一次`send`调用的返回值。不要错过这一点：像`next()`一样，`send()`方法不仅将一个值提交给生成器，还返回即将到来的`yield`语句的值。这就是我们定义生成器和协程之间差异的方式：生成器只产生值，而协程也可以消费它们。

`next(i)`、`i.__next__()`和`i.send(value)`的行为和语法相当不直观且令人沮丧。第一个是一个普通函数，第二个是一个特殊方法，最后一个是一个普通方法。但三者都做同样的事情：推进生成器直到它产生一个值并暂停。此外，`next()`函数和相关方法可以通过调用`i.send(None)`来复制。在这里有两个不同的方法名称是有价值的，因为它有助于我们的代码读者轻松地看到他们是在与协程还是生成器交互。我只是觉得在一种情况下它是一个函数调用，而在另一种情况下它是一个普通方法有些令人烦恼。

# 回到日志解析

当然，前面的例子可以很容易地使用几个整数变量和调用`x += increment`来编写。让我们看看第二个例子，其中协程实际上帮我们节省了一些代码。这个例子是一个为了教学目的而简化的（问题）版本，我在 Facebook 工作时必须解决的问题。

Linux 内核日志包含看起来几乎，但又不完全像这样的行：

```py
unrelated log messages 
sd 0:0:0:0 Attached Disk Drive 
unrelated log messages 
sd 0:0:0:0 (SERIAL=ZZ12345) 
unrelated log messages 
sd 0:0:0:0 [sda] Options 
unrelated log messages 
XFS ERROR [sda] 
unrelated log messages 
sd 2:0:0:1 Attached Disk Drive 
unrelated log messages 
sd 2:0:0:1 (SERIAL=ZZ67890) 
unrelated log messages 
sd 2:0:0:1 [sdb] Options 
unrelated log messages 
sd 3:0:1:8 Attached Disk Drive 
unrelated log messages 
sd 3:0:1:8 (SERIAL=WW11111) 
unrelated log messages 
sd 3:0:1:8 [sdc] Options 
unrelated log messages 
XFS ERROR [sdc] 
unrelated log messages 
```

有很多散布的内核日志消息，其中一些与硬盘有关。硬盘消息可能会与其他消息混合，但它们以可预测的格式和顺序出现。对于每一个，一个具有已知序列号的特定驱动器与一个总线标识符（例如`0:0:0:0`）相关联。一个块设备标识符（例如`sda`）也与该总线相关联。最后，如果驱动器有一个损坏的文件系统，它可能会因为 XFS 错误而失败。

现在，鉴于前面的日志文件，我们需要解决的问题是如何获取任何有 XFS 错误的驱动器的序列号。这个序列号可能后来会被数据中心的技术人员用来识别和更换驱动器。

我们知道我们可以使用正则表达式来识别单独的行，但我们需要在循环遍历行时更改正则表达式，因为我们将根据之前找到的内容寻找不同的事情。另一个困难之处在于，如果我们找到一个错误字符串，包含该字符串的总线信息以及序列号已经处理过了。这可以通过按相反的顺序迭代文件中的行来轻松解决。

在你看这个例子之前，要警告你——基于协程的解决方案所需的代码量非常小：

```py
import re

def match_regex(filename, regex):
    with open(filename) as file:
        lines = file.readlines()
    for line in reversed(lines):
        match = re.match(regex, line)
        if match:
 regex = yield match.groups()[0]

def get_serials(filename):
    ERROR_RE = "XFS ERROR (\[sd[a-z]\])"
    matcher = match_regex(filename, ERROR_RE)
    device = next(matcher)
    while True:
        try:
            bus = matcher.send(
                "(sd \S+) {}.*".format(re.escape(device))
            )
            serial = matcher.send("{} \(SERIAL=([^)]*)\)".format(bus))
 yield serial
            device = matcher.send(ERROR_RE)
        except StopIteration:
            matcher.close()
            return

for serial_number in get_serials("EXAMPLE_LOG.log"):
    print(serial_number)
```

这段代码巧妙地将工作分为两个单独的任务。第一个任务是循环遍历所有行并输出任何匹配给定正则表达式的行。第二个任务是与第一个任务交互，并指导它在任何给定时间应该搜索什么正则表达式。

首先看看`match_regex`协程。记住，它在构造时不会执行任何代码；相反，它只是创建一个协程对象。一旦构造完成，协程外部的人最终会调用`next()`来启动代码的运行。然后它存储两个变量`filename`和`regex`的状态。然后它读取文件中的所有行，并按相反的顺序迭代它们。每一行都会与传入的正则表达式进行比较，直到找到匹配项。当找到匹配项时，协程产生正则表达式的第一个组并等待。

在未来的某个时刻，其他代码将发送一个新的正则表达式来搜索。请注意，协程永远不会关心它试图匹配的正则表达式是什么；它只是在循环遍历行并将它们与正则表达式进行比较。决定提供什么正则表达式是别人的责任。

在这种情况下，其他人就是`get_serials`生成器。它不关心文件中的行；实际上，它甚至没有意识到它们的存在。它首先做的事情是从`match_regex`协程构造函数创建一个`matcher`对象，给它一个默认的正则表达式来搜索。它将协程推进到它的第一个`yield`，并存储它返回的值。然后它进入一个循环，指示`matcher`对象根据存储的设备 ID 搜索一个总线 ID，然后根据该总线 ID 搜索一个序列号。

在指示匹配器找到另一个设备 ID 并重复循环之前，它先无用地将序列号产生到外部的`for`循环中。

基本上，协程的任务是搜索文件中的下一个重要行，而生成器（使用`yield`语法但不进行赋值的`get_serial`）的任务是决定哪一行是重要的。生成器有关于这个特定问题的信息，比如文件中行的出现顺序。另一方面，协程可以插入到任何需要搜索文件中给定正则表达式的任何问题中。

# 关闭协程和抛出异常

正常生成器通过抛出`StopIteration`来在内部发出退出信号。如果我们将多个生成器链式连接在一起（例如，通过在一个生成器内部迭代另一个生成器），`StopIteration`异常将会向外传播。最终，它将遇到一个`for`循环，该循环将看到这个异常并知道是时候退出循环了。

尽管它们使用类似的语法，但协程通常不遵循迭代机制。而不是通过一个直到遇到异常，数据通常被推入其中（使用`send`）。执行推入操作的是通常负责告诉协程何时完成的实体。它通过在相关的协程上调用`close()`方法来完成这个操作。

当调用时，`close()`方法将在协程等待发送值的地方抛出`GeneratorExit`异常。对于协程来说，通常是一个好的做法，将它们的`yield`语句包裹在`try`...`finally`块中，这样就可以执行任何清理任务（例如关闭相关的文件或套接字）。

如果我们需要在协程内部抛出异常，我们可以使用类似的方式使用`throw()`方法。它接受一个异常类型，并带有可选的`value`和`traceback`参数。后者在我们遇到一个协程中的异常时很有用，我们想在相邻的协程中引发异常，同时保持跟踪记录。

之前的例子可以不使用协程来编写，并且阅读起来几乎一样。事实上，正确管理协程之间的所有状态相当困难，尤其是在考虑上下文管理器和异常的情况下。幸运的是，Python 标准库中包含一个名为`asyncio`的包，可以为你管理所有这些。我们将在并发章节中介绍这一点。一般来说，我建议你除非你专门为`asyncio`编码，否则避免使用裸协程。日志记录示例几乎可以被认为是一种*反模式*；一种应该避免而不是采纳的设计模式。

# 协程、生成器和函数之间的关系

我们已经看到了协程的实际应用，现在让我们回到它们与生成器相关性的讨论。在 Python 中，正如经常发生的那样，这种区别相当模糊。事实上，所有协程都是生成器对象，作者经常互换使用这两个术语。有时，他们将协程描述为生成器的一个子集（只有从`yield`返回值的生成器才被认为是协程）。在 Python 中，这从先前的章节中我们已经看到，在技术上是真的。

然而，在理论计算机科学的更广泛领域，协程被认为是更通用的原则，生成器是协程的一种特定类型。此外，普通函数是协程的另一个独立的子集。

协程是一种可以在一个或多个点接收数据并在一个或多个点输出数据的例程。在 Python 中，数据传入和退出的点是`yield`语句。

函数，或子程序，是最简单的协程类型。你可以在一个点传入数据，当函数返回时在另一个点获取数据。虽然函数可以有多个`return`语句，但在任何给定函数调用中只能调用其中一个。

最后，生成器是一种可以在一个点传入数据但在多个点输出数据的协程类型。在 Python 中，数据会在`yield`语句处输出，但不能传入数据。如果你调用`send`，数据将被静默丢弃。

因此，从理论上讲，生成器是协程类型，函数是协程类型，还有一些既不是函数也不是生成器的协程。这很简单，对吧？那么，为什么在 Python 中感觉更复杂呢？

在 Python 中，生成器和协程都使用一种**看起来**像我们在构建函数的语法来构建。但生成的对象根本不是函数；它是一种完全不同的对象。当然，函数也是对象。但它们有不同的接口；函数是可调用的并返回值，生成器使用`next()`提取数据，协程使用`send()`推送数据。

协程还有一个使用`async`和`await`关键字的不同语法。这种语法使得代码是协程的事实更加清晰，并进一步打破了协程和生成器之间欺骗性的对称性。这种语法在没有构建完整的事件循环的情况下工作得不是很好，所以我们将在并发章节介绍`asyncio`之前跳过它。

# 案例研究

目前 Python 最受欢迎的领域之一是数据科学。为了纪念这一事实，让我们实现一个基本的机器学习算法。

机器学习是一个巨大的主题，但基本思想是利用从过去数据中获得的知识来对未来的数据进行预测或分类。此类算法的应用非常广泛，数据科学家每天都在寻找新的应用机器学习的方法。一些重要的机器学习应用包括计算机视觉（如图像分类或人脸识别）、产品推荐、识别垃圾邮件和自动驾驶汽车。

为了避免深入到一本关于机器学习的整本书，我们将看看一个更简单的问题：给定一个 RGB 颜色定义，人类会将其识别为哪种颜色？

标准 RGB 颜色空间中有超过 1600 万种颜色，而人类只为其中的一小部分想出了名字。虽然有一些名字（有些相当荒谬；只需去任何汽车经销商或油漆店看看），但让我们构建一个分类器，尝试将 RGB 空间划分为基本颜色：

+   红色

+   紫色

+   蓝色

+   绿色

+   黄色

+   橙色

+   灰色

+   粉色

（在我的测试中，我将白色和黑色颜色分类为灰色，将棕色颜色分类为橙色。）

我们首先需要的是一个用于训练我们算法的数据集。在生产系统中，你可能需要抓取一个 *颜色列表* 网站，或者调查数千人。相反，我创建了一个简单的应用程序，它会渲染一个随机颜色，并要求用户从前面的八个选项中选择一个来对其进行分类。我使用 `tkinter`（Python 附带的用户界面工具包）实现了它。我不会详细介绍这个脚本的细节，但为了完整性，这里提供它的全部内容（它有点长，所以你可能想从 Packt 的 GitHub 仓库中获取这本书的示例，而不是手动输入）：

```py
import random
import tkinter as tk
import csv

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.grid(sticky="news")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.create_widgets()
        self.file = csv.writer(open("colors.csv", "a"))

    def create_color_button(self, label, column, row):
        button = tk.Button(
            self, command=lambda: self.click_color(label), text=label
        )
        button.grid(column=column, row=row, sticky="news")

    def random_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        return f"#{r:02x}{g:02x}{b:02x}"

    def create_widgets(self):
        self.color_box = tk.Label(
            self, bg=self.random_color(), width="30", height="15"
        )
        self.color_box.grid(
            column=0, columnspan=2, row=0, sticky="news"
        )
        self.create_color_button("Red", 0, 1)
        self.create_color_button("Purple", 1, 1)
        self.create_color_button("Blue", 0, 2)
        self.create_color_button("Green", 1, 2)
        self.create_color_button("Yellow", 0, 3)
        self.create_color_button("Orange", 1, 3)
        self.create_color_button("Pink", 0, 4)
        self.create_color_button("Grey", 1, 4)
        self.quit = tk.Button(
            self, text="Quit", command=root.destroy, bg="#ffaabb"
        )
        self.quit.grid(column=0, row=5, columnspan=2, sticky="news")

    def click_color(self, label):
        self.file.writerow([label, self.color_box["bg"]])
        self.color_box["bg"] = self.random_color()

root = tk.Tk()
app = Application(master=root)
app.mainloop()
```

如果你喜欢，可以轻松地添加更多按钮以供其他颜色选择。你可能会在布局上遇到困难；`create_color_button` 函数的第二个和第三个参数代表按钮所在的二维网格的行和列。一旦你将所有颜色放置到位，你可能会想将 **退出** 按钮移至最后一行。

对于本案例研究的目的，了解此应用程序的输出很重要。它创建了一个名为 `colors.csv` 的 **逗号分隔值**（**CSV**）文件。此文件包含两个 CSV 文件：用户分配给颜色的标签以及颜色的十六进制 RGB 值。以下是一个示例：

```py
Green,#6edd13
Purple,#814faf
Yellow,#c7c26d
Orange,#61442c
Green,#67f496
Purple,#c757d5
Blue,#106a98
Pink,#d40491
.
.
.
Blue,#a4bdfa
Green,#30882f
Pink,#f47aad
Green,#83ddb2
Grey,#baaec9
Grey,#8aa28d
Blue,#533eda
```

在我感到无聊并决定开始对我的数据集进行机器学习之前，我制作了超过 250 个数据点。如果你想使用它，这些数据点与本章的示例一起提供（没有人告诉我我是色盲，所以它应该是有一定合理性的）。

我们将实现一种简单的机器学习算法，称为 *k 近邻*。此算法依赖于数据集中点之间的某种 *距离* 计算（在我们的情况下，我们可以使用三维版本的勾股定理）。给定一个新数据点，它找到一定数量的数据点（称为 *k*，即 *k* 近邻中的 *k*），这些数据点在距离计算中被测量为最接近。然后它以某种方式组合这些数据点（对于线性计算，平均值可能适用；对于我们的分类问题，我们将使用众数），并返回结果。

我们不会过多地介绍算法做了什么；相反，我们将关注一些我们可以将迭代器模式或迭代器协议应用于此问题的方法。

现在我们编写一个程序，按照以下步骤顺序执行：

1.  从文件中加载样本数据并据此构建模型。

1.  生成 100 种随机颜色。

1.  将每种颜色进行分类，并以与输入相同的格式输出到文件。

第一步是一个相当简单的生成器，它加载 CSV 数据并将其转换为适合我们需求的形式：

```py
import csv

dataset_filename = "colors.csv"

def load_colors(filename):
    with open(filename) as dataset_file:
        lines = csv.reader(dataset_file)
 for line in lines:
            label, hex_color = line
 yield (hex_to_rgb(hex_color), label)
```

我们之前没有见过`csv.reader`函数。它返回文件中行的迭代器。迭代器返回的每个值都是一个字符串列表，由逗号分隔。因此，行`Green,#6edd13`返回为`["Green", "#6edd13"]`。

`load_colors`生成器随后逐行消费这个迭代器，并产生一个 RGB 值的元组以及标签。以这种方式链式使用生成器是很常见的，其中一个迭代器调用另一个，然后又调用另一个，依此类推。你可能想查看 Python 标准库中的`itertools`模块，那里有许多现成的生成器等待你使用。

在这种情况下，RGB 值是介于 0 到 255 之间的整数元组。从十六进制到 RGB 的转换有点棘手，所以我们将其提取到一个单独的函数中：

```py
def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) for i in range(1, 6, 2))

```

这个生成器表达式正在做很多工作。它接受一个如`"#12abfe"`这样的字符串作为输入，并返回一个如`(18, 171, 254)`这样的元组。让我们从后往前分析它。

`range`调用将返回数字`[1, 3, 5]`。这些代表十六进制字符串中三个颜色通道的索引。索引`0`被跳过，因为它代表字符`"#"`，我们对此不感兴趣。对于这三个数字中的每一个，它提取`i`和`i+2`之间的两个字符字符串。对于前面的示例字符串，这将分别是`12`、`ab`和`fe`。然后它将这个字符串值转换为整数。传递给`int`函数的第二个参数`16`告诉函数在转换时使用十六进制（十六进制）而不是通常的十进制（十进制）基数。

由于生成器表达式难以阅读，你认为它应该以不同的格式表示吗？它可以创建为多个生成器表达式的序列，例如，或者展开为一个带有`yield`语句的正常生成器函数。你更喜欢哪一个？

在这种情况下，我放心地相信函数名可以解释那行丑陋的代码在做什么。

现在我们已经加载了*训练数据*（手动分类的颜色），我们需要一些新的数据来测试算法的效果。我们可以通过生成一百种随机颜色来实现，每种颜色由 0 到 255 之间的三个随机数组成。

有很多种实现方式：

+   一个带有嵌套生成器表达式的列表解析：`[tuple(randint(0,255) for c in range(3)) for r in range(100)]`

+   一个基本的生成器函数

+   一个实现`__iter__`和`__next__`协议的类

+   将数据通过协程管道推送

+   甚至只是一个基本的`for`循环

生成器版本看起来最易读，所以让我们将这个函数添加到我们的程序中：

```py
from random import randint

def generate_colors(count=100):
    for i in range(count):
        yield (randint(0, 255), randint(0, 255), randint(0, 255))
```

注意我们如何参数化要生成的颜色数量。现在我们可以重用这个函数来处理未来的其他颜色生成任务。

现在，在我们进行分类步骤之前，我们需要一个函数来计算两种颜色之间的*距离*。由于颜色可以被认为是三维的（例如，红色、绿色和蓝色可以映射到*x*、*y*和*z*轴），让我们使用一点基本的数学：

```py
def color_distance(color1, color2):
    channels = zip(color1, color2)
    sum_distance_squared = 0
    for c1, c2 in channels:
        sum_distance_squared += (c1 - c2) ** 2
    return sum_distance_squared
```

这个函数看起来相当基础；它看起来甚至没有使用迭代器协议。没有`yield`函数，没有理解。然而，有一个`for`循环，并且那个`zip`函数的调用实际上在进行一些真正的迭代（如果你不熟悉它，`zip`产生元组，每个元组包含来自每个输入迭代器的元素）。

这种距离计算是你可能从学校学到的三维版本的勾股定理：*a² + b² = c²*。由于我们使用三个维度，我想它实际上应该是*a² + b² + c² = d²*。距离在技术上是指*a² + b² + c²*的平方根，但由于平方距离彼此之间相对大小相同，所以没有必要执行相对昂贵的`sqrt`计算。

现在我们已经建立了一些管道，让我们进行实际的 k 最近邻实现。这个例程可以被认为是消费和组合我们之前看到的两个生成器（`load_colors`和`generate_colors`）：

```py
def nearest_neighbors(model_colors, target_colors, num_neighbors=5):
    model_colors = list(model_colors)

    for target in target_colors:
        distances = sorted(
            ((color_distance(c[0], target), c) for c in model_colors)
        )
        yield target, distances[:5]
```

我们首先将`model_colors`生成器转换为列表，因为它必须被多次消费，一次用于每个`target_colors`。如果我们不这样做，我们就必须反复从源文件中加载颜色，这将执行很多不必要的磁盘读取。

这个决定的缺点是整个列表必须一次性存储在内存中。如果我们有一个庞大的数据集，它无法适应内存，那么实际上每次都需要从磁盘重新加载生成器（尽管在这种情况下，我们实际上会查看不同的机器学习算法）。

`nearest_neighbors`生成器遍历每个目标颜色（一个三元组，例如`(255, 14, 168)`），并在生成器表达式中调用它内部的`color_distance`函数。围绕该生成器表达式的`sorted`调用然后按其第一个元素（即距离）对结果进行排序。这是一段复杂的代码，并且根本不是面向对象的。你可能想要将其分解为正常的`for`循环，以确保你理解生成器表达式正在做什么。

`yield`语句稍微简单一些。对于`target_colors`生成器中的每个 RGB 三元组，它产生目标和一个包含`num_neighbors`（也就是*k*，在*k*最近邻中。许多数学家和数据科学家都有一种糟糕的倾向，使用难以理解的单一字母变量名）最近颜色的列表。

列表推导式中的每个元素内容都是`model_colors`生成器中的一个元素；也就是说，是一个包含三个 RGB 值和为该颜色手动输入的字符串名称的元组。所以，一个元素可能看起来像这样：`((104, 195, 77), 'Green')`。当我看到这样的嵌套元组时，我首先想到的是，*这不是正确的数据结构*。RGB 颜色可能应该用命名元组来表示，而这两个属性可能应该放在数据类中。

我们现在可以向链中添加另一个生成器，以确定我们应该给这个目标颜色起什么名字：

```py
from collections import Counter

def name_colors(model_colors, target_colors, num_neighbors=5):
    for target, near in nearest_neighbors(
        model_colors, target_colors, num_neighbors=5
    ):
        print(target, near)
        name_guess = Counter(n[1] for n in near).most_common()[0][0]
        yield target, name_guess
```

这个生成器正在将`nearest_neighbors`返回的元组解包成三个元组的目标和五个最近的数据点。它使用`Counter`来找到在返回的颜色中出现次数最多的名称。在`Counter`构造函数中还有一个另一个生成器表达式；这个生成器从每个数据点中提取第二个元素（颜色名称）。然后它产生一个 RGB 值和猜测的名称。返回值的例子是`(91, 158, 250) Blue`。

我们可以编写一个函数，该函数接受`name_colors`生成器的输出并将其写入 CSV 文件，RGB 颜色以十六进制值表示：

```py
def write_results(colors, filename="output.csv"):
    with open(filename, "w") as file:
        writer = csv.writer(file)
        for (r, g, b), name in colors:
            writer.writerow([name, f"#{r:02x}{g:02x}{b:02x}"])
```

这是一个函数，不是一个生成器。它在`for`循环中消耗生成器，但没有产生任何东西。它构建了一个 CSV 写入器，并为每个目标颜色输出名称、十六进制值（例如，`Purple,#7f5f95`）的行。这里可能让人困惑的是格式字符串的内容。与每个`r`、`g`和`b`通道一起使用的`:02x`修饰符将数字输出为零填充的两个十六进制数字。

现在我们只需要将这些不同的生成器和管道连接起来，并通过单个函数调用启动过程：

```py
def process_colors(dataset_filename="colors.csv"):
    model_colors = load_colors(dataset_filename)
    colors = name_colors(model_colors, generate_colors(), 5)
    write_results(colors)

if __name__ == "__main__":
    process_colors()
```

所以，这个函数，与我们所定义的几乎所有其他函数不同，是一个完全正常的函数，没有任何`yield`语句或`for`循环。它根本不做任何迭代。

它确实构建了三个生成器。你能看到这三个吗？：

+   `load_colors`返回一个生成器

+   `generate_colors`返回一个生成器

+   `name_guess`返回一个生成器

`name_guess`生成器消耗前两个生成器。然后，它被`write_results`函数消耗。

我编写了第二个 Tkinter 应用程序来检查算法的准确性。它与第一个应用程序类似，但它渲染每个颜色及其相关的标签。然后你必须手动点击是或否，如果标签与颜色匹配。对于我的示例数据，我得到了大约 95%的准确性。这可以通过实现以下方法来改进：

+   添加更多颜色名称

+   通过手动分类更多颜色来添加更多训练数据

+   调整`num_neighbors`的值

+   使用更先进的机器学习算法

这里是输出检查应用程序的代码，尽管我建议下载示例代码。这会非常麻烦：

```py
import tkinter as tk
import csv

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.grid(sticky="news")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.csv_reader = csv.reader(open("output.csv"))
        self.create_widgets()
        self.total_count = 0
        self.right_count = 0

    def next_color(self):
        return next(self.csv_reader)

    def mk_grid(self, widget, column, row, columnspan=1):
        widget.grid(
            column=column, row=row, columnspan=columnspan, sticky="news"
        )

    def create_widgets(self):
        color_text, color_bg = self.next_color()
        self.color_box = tk.Label(
            self, bg=color_bg, width="30", height="15"
        )
        self.mk_grid(self.color_box, 0, 0, 2)

        self.color_label = tk.Label(self, text=color_text, height="3")
        self.mk_grid(self.color_label, 0, 1, 2)

        self.no_button = tk.Button(
            self, command=self.count_next, text="No"
        )
        self.mk_grid(self.no_button, 0, 2)

        self.yes_button = tk.Button(
            self, command=self.count_yes, text="Yes"
        )
        self.mk_grid(self.yes_button, 1, 2)

        self.percent_accurate = tk.Label(self, height="3", text="0%")
        self.mk_grid(self.percent_accurate, 0, 3, 2)

        self.quit = tk.Button(
            self, text="Quit", command=root.destroy, bg="#ffaabb"
        )
        self.mk_grid(self.quit, 0, 4, 2)

    def count_yes(self):
        self.right_count += 1
        self.count_next()

    def count_next(self):
        self.total_count += 1
        percentage = self.right_count / self.total_count
        self.percent_accurate["text"] = f"{percentage:.0%}"
        try:
            color_text, color_bg = self.next_color()
        except StopIteration:
            color_text = "DONE"
            color_bg = "#ffffff"
            self.color_box["text"] = "DONE"
            self.yes_button["state"] = tk.DISABLED
            self.no_button["state"] = tk.DISABLED
        self.color_label["text"] = color_text
        self.color_box["bg"] = color_bg

root = tk.Tk()
app = Application(master=root)
app.mainloop()
```

你可能会想知道，“这与面向对象编程有什么关系？这段代码甚至没有使用一个类！”。在某种程度上，你是对的；生成器通常不被认为是面向对象的。然而，创建它们的函数返回对象；实际上，你可以将这些函数视为构造函数。构建的对象有一个适当的`__next__()`方法。基本上，生成器语法是一种语法快捷方式，用于创建一种在没有它的情况下会相当冗长的特定类型的对象。

作为一种历史性的注释，本书的第二版使用协程而不是基本生成器来解决此问题。我决定在第三版中将它改为生成器，原因有几个：

+   除了`asyncio`之外，没有人会在现实生活中使用协程，我们将在并发章节中介绍`asyncio`。我觉得我错误地鼓励人们使用协程来解决问题，而实际上协程极其罕见地是正确的工具。

+   与生成器版本相比，协程版本更长、更复杂，并且有更多的模板代码。

+   协程版本没有展示出本章讨论的其他足够多的特性，例如列表推导和生成器表达式。

如果你可能对基于协程的实现感到历史上有兴趣，我在本章的下载示例代码中包含了这个代码的副本。

# 练习

如果你平时很少在代码中使用推导，你应该首先搜索一些现有的代码，找到一些`for`循环。看看它们是否可以轻易地转换为生成器表达式、列表、集合或字典推导。 

测试列表推导比`for`循环更快的说法。这可以通过内置的`timeit`模块来完成。使用`timeit.timeit`函数的帮助文档来了解如何使用它。基本上，写两个执行相同操作的功能，一个使用列表推导，另一个使用`for`循环遍历数千个项目。将每个函数传递给`timeit.timeit`，并比较结果。如果你觉得冒险，也可以比较生成器和生成器表达式。使用`timeit`测试代码可能会变得上瘾，所以请记住，除非代码被大量执行，例如在巨大的输入列表或文件上，否则代码不需要非常快。

尝试使用生成器函数。从需要多个值的简单迭代器开始（数学序列是典型的例子；如果你想不到更好的例子，斐波那契序列可能会被过度使用）。尝试一些更高级的生成器，它们可以执行诸如合并多个输入列表并从中产生值等操作。生成器也可以用于文件；你能写一个简单的生成器，显示两个文件中相同的行吗？

协程滥用迭代器协议，但实际上并不满足迭代器模式。你能构建一个非协程版本的代码，从日志文件中获取序列号吗？采用面向对象的方法，以便你可以在类上存储额外的状态。如果你能创建一个对象，它可以替代现有的协程，那么你将学到很多关于协程的知识。

本章的案例研究有很多奇特的元组元组传递，难以跟踪。看看你是否可以用更面向对象的方法替换那些返回值。此外，尝试将一些共享数据的函数（例如，`model_colors`和`target_colors`）移动到一个类中。这应该会减少大多数生成器需要传递的参数数量，因为它们可以在`self`中查找它们。

# 摘要

在本章中，我们了解到设计模式是有用的抽象，为常见的编程问题提供了最佳实践解决方案。我们介绍了我们的第一个设计模式——迭代器，以及 Python 使用和滥用此模式的各种方式，以实现其自身的邪恶目的。原始的迭代器模式非常面向对象，但编写代码时既丑陋又冗长。然而，Python 的内置语法抽象掉了这种丑陋，为我们提供了面向对象构造的干净接口。

理解和生成器表达式可以将容器构造与迭代结合到一行中。可以使用`yield`语法构造生成器对象。协程在外观上类似于生成器，但它们服务于完全不同的目的。

在接下来的两章中，我们将介绍更多设计模式。
