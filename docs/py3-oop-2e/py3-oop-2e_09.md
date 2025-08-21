# 第九章：迭代器模式

我们已经讨论了许多 Python 内置和习惯用法，乍一看似乎不是面向对象的，实际上在底层提供了对主要对象的访问。在本章中，我们将讨论`for`循环，看起来如此结构化，实际上是一组面向对象原则的轻量级包装。我们还将看到一系列扩展到这个语法的类型。我们将涵盖：

+   什么是设计模式

+   迭代器协议——最强大的设计模式之一

+   列表、集合和字典的理解

+   生成器和协程

# 简要介绍设计模式

当工程师和建筑师决定建造桥梁、塔楼或建筑物时，他们遵循某些原则以确保结构完整性。桥梁有各种可能的设计（例如悬索桥或悬臂桥），但如果工程师不使用标准设计之一，并且没有一个新的杰出设计，那么他/她设计的桥梁可能会倒塌。

设计模式试图将正确设计的结构的正式定义引入软件工程。有许多不同的设计模式来解决不同的通用问题。创建设计模式的人首先确定开发人员在各种情况下面临的常见问题。然后，他们建议可能被认为是该问题的理想解决方案，从面向对象设计的角度来看。

了解设计模式并选择在我们的软件中使用它并不保证我们正在创建一个“正确”的解决方案。1907 年，魁北克大桥（至今仍是世界上最长的悬臂桥）在建造完成之前倒塌，因为设计它的工程师严重低估了用于建造的钢材重量。同样，在软件开发中，我们可能错误地选择或应用设计模式，并创建在正常操作情况下或在超出其原始设计限制时“倒塌”的软件。

任何一个设计模式都提出了一组以特定方式相互作用的对象，以解决一个通用问题。程序员的工作是识别何时面临特定版本的问题，并在解决方案中调整通用设计。

在本章中，我们将介绍迭代器设计模式。这种模式是如此强大和普遍，以至于 Python 开发人员提供了多种语法来访问该模式的基础面向对象原则。我们将在接下来的两章中介绍其他设计模式。其中一些具有语言支持，一些没有，但没有一个像迭代器模式一样成为 Python 程序员日常生活的固有部分。

# 迭代器

在典型的设计模式术语中，迭代器是一个具有`next()`方法和`done()`方法的对象；后者在序列中没有剩余项目时返回`True`。在没有迭代器的内置支持的编程语言中，迭代器将被循环遍历，如下所示：

```py
while not iterator.done():
    item = iterator.next()
    # do something with the item
```

在 Python 中，迭代是一个特殊的特性，所以该方法得到了一个特殊的名称`__next__`。可以使用内置的`next(iterator)`来访问这个方法。迭代器协议不是使用`done`方法，而是引发`StopIteration`来通知循环已经完成。最后，我们有更加可读的`for item in iterator`语法来实际访问迭代器中的项目，而不是使用`while`循环。让我们更详细地看一下这些。

## 迭代器协议

抽象基类`Iterator`，在`collections.abc`模块中，定义了 Python 中的迭代器协议。正如前面提到的，它必须有一个`__next__`方法，`for`循环（和其他支持迭代的功能）可以调用它来从序列中获取一个新的元素。此外，每个迭代器还必须满足`Iterable`接口。任何提供`__iter__`方法的类都是可迭代的；该方法必须返回一个`Iterator`实例，该实例将覆盖该类中的所有元素。由于迭代器已经在元素上循环，因此它的`__iter__`函数传统上返回它自己。

这可能听起来有点混乱，所以看一下以下的例子，但请注意，这是解决这个问题的一种非常冗长的方式。它清楚地解释了迭代和所讨论的两个协议，但在本章的后面，我们将看到几种更可读的方法来实现这种效果：

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

这个例子定义了一个`CapitalIterable`类，其工作是循环遍历字符串中的每个单词，并输出它们的首字母大写。这个可迭代对象的大部分工作都交给了`CapitalIterator`实现。与这个迭代器互动的规范方式如下：

```py
>>> iterable = CapitalIterable('the quick brown fox jumps over the lazy dog')
>>> iterator = iter(iterable)
>>> while True:
...     try:
...         print(next(iterator))
...     except StopIteration:
...         break
...** 
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

这个例子首先构造了一个可迭代对象，并从中检索了一个迭代器。这种区别可能需要解释；可迭代对象是一个具有可以循环遍历的元素的对象。通常，这些元素可以被多次循环遍历，甚至可能在同一时间或重叠的代码中。另一方面，迭代器代表可迭代对象中的特定位置；一些项目已被消耗，一些项目尚未被消耗。两个不同的迭代器可能在单词列表中的不同位置，但任何一个迭代器只能标记一个位置。

每次在迭代器上调用`next()`时，它会按顺序从可迭代对象中返回另一个标记。最终，迭代器将被耗尽（不再有任何元素返回），在这种情况下会引发`Stopiteration`，然后我们跳出循环。

当然，我们已经知道了一个更简单的语法来从可迭代对象中构造一个迭代器：

```py
>>> for i in iterable:
...     print(i)
...** 
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

正如你所看到的，`for`语句，尽管看起来并不是非常面向对象，实际上是一种显然面向对象设计原则的快捷方式。在我们讨论理解时，请记住这一点，因为它们似乎是面向对象工具的完全相反。然而，它们使用与`for`循环完全相同的迭代协议，只是另一种快捷方式。

# 理解

理解是简单但强大的语法，允许我们在一行代码中转换或过滤可迭代对象。结果对象可以是一个完全正常的列表、集合或字典，也可以是一个生成器表达式，可以在一次性中高效地消耗。

## 列表理解

列表理解是 Python 中最强大的工具之一，所以人们倾向于认为它们是高级的。它们不是。事实上，我已经在以前的例子中使用了理解，并假设你会理解它们。虽然高级程序员确实经常使用理解，但并不是因为它们很高级，而是因为它们很琐碎，并处理软件开发中最常见的一些操作。

让我们来看看其中一个常见的操作；即将一组项目转换为相关项目的列表。具体来说，假设我们刚刚从文件中读取了一个字符串列表，现在我们想将其转换为一个整数列表。我们知道列表中的每个项目都是整数，并且我们想对这些数字进行一些操作（比如计算平均值）。以下是一种简单的方法来解决这个问题：

```py
input_strings = ['1', '5', '28', '131', '3']

output_integers = []
for num in input_strings:
    output_integers.append(int(num))
```

这个例子运行良好，只有三行代码。如果你不习惯理解，你可能甚至不会觉得它看起来很丑！现在，看看使用列表理解的相同代码：

```py
input_strings = ['1', '5', '28', '131', '3']output_integers = [int(num) for num in input_strings]
```

我们只剩下一行，而且，对于性能来说，我们已经删除了列表中每个项目的`append`方法调用。总的来说，很容易看出发生了什么，即使你不习惯理解推导语法。

方括号表示，我们正在创建一个列表。在这个列表中是一个`for`循环，它遍历输入序列中的每个项目。唯一可能令人困惑的是列表的左大括号和`for`循环开始之间发生了什么。这里发生的事情应用于输入列表中的*每个*项目。所讨论的项目由循环中的`num`变量引用。因此，它将每个单独的元素转换为`int`数据类型。

这就是基本列表推导的全部内容。它们并不那么高级。推导是高度优化的 C 代码；当循环遍历大量项目时，列表推导比`for`循环要快得多。如果仅凭可读性不足以说服你尽可能多地使用它们，速度应该是一个令人信服的理由。

将一个项目列表转换为相关列表并不是列表推导唯一能做的事情。我们还可以选择通过在推导中添加`if`语句来排除某些值。看一下：

```py
output_ints = [int(n) for n in input_strings if len(n) < 3]
```

我将变量的名称从`num`缩短为`n`，将结果变量缩短为`output_ints`，这样它仍然可以放在一行上。除此之外，这个例子和前一个例子之间的唯一不同是`if len(n) < 3`部分。这个额外的代码排除了任何长度超过两个字符的字符串。`if`语句应用于`int`函数之前，因此它测试字符串的长度。由于我们的输入字符串本质上都是整数，它排除了任何大于 99 的数字。现在列表推导就是这样了！我们用它们将输入值映射到输出值，同时应用过滤器来包括或排除满足特定条件的任何值。

任何可迭代对象都可以成为列表推导的输入；我们可以将任何可以放在`for`循环中的东西也放在推导中。例如，文本文件是可迭代的；文件的迭代器上的每次调用`__next__`将返回文件的一行。我们可以使用`zip`函数将制表符分隔的文件加载到字典中，其中第一行是标题行：

```py
import sys
filename = sys.argv[1]

with open(filename) as file:
    header = file.readline().strip().split('\t')
 **contacts = [
 **dict(
 **zip(header, line.strip().split('\t'))
 **) for line in file
 **]

for contact in contacts:
    print("email: {email} -- {last}, {first}".format(
        **contact))
```

这次，我添加了一些空格，使其更易读一些（列表推导不一定要放在一行上）。这个例子从文件的标题和分割行创建了一个字典列表。

嗯，什么？如果那段代码或解释没有意义，不要担心；有点令人困惑。一个列表推导在这里做了大量的工作，代码很难理解、阅读，最终也难以维护。这个例子表明列表推导并不总是最好的解决方案；大多数程序员都会同意，`for`循环比这个版本更可读。

### 提示

记住：我们提供的工具不应该被滥用！始终选择合适的工具，即编写可维护的代码。

## 集合和字典推导

推导不仅限于列表。我们也可以使用类似的语法用大括号创建集合和字典。让我们从集合开始。创建集合的一种方法是将列表推导包装在`set()`构造函数中，将其转换为集合。但是，为什么要浪费内存在一个被丢弃的中间列表上，当我们可以直接创建一个集合呢？

这是一个例子，它使用命名元组来模拟作者/标题/流派三元组，然后检索写作特定流派的所有作者的集合：

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

fantasy_authors = {
 **b.author for b in books if b.genre == 'fantasy'}

```

与演示数据设置相比，突出显示的集合推导确实很短！如果我们使用列表推导，特里·普拉切特当然会被列出两次。因为集合的性质消除了重复项，我们最终得到：

```py
>>> fantasy_authors
{'Turner', 'Pratchett', 'Le Guin'}

```

我们可以引入冒号来创建字典理解。这将使用*键:值*对将序列转换为字典。例如，如果我们知道标题，可能会很有用快速查找字典中的作者或流派。我们可以使用字典理解将标题映射到书籍对象：

```py
fantasy_titles = {
        b.title: b for b in books if b.genre == 'fantasy'}
```

现在，我们有了一个字典，可以使用正常的语法按标题查找书籍。

总之，理解不是高级 Python，也不是应该避免使用的“非面向对象”工具。它们只是一种更简洁和优化的语法，用于从现有序列创建列表、集合或字典。

## 生成器表达式

有时我们希望处理一个新的序列，而不将新的列表、集合或字典放入系统内存中。如果我们只是逐个循环遍历项目，并且实际上并不关心是否创建最终的容器对象，那么创建该容器就是对内存的浪费。在逐个处理项目时，我们只需要当前对象在任一时刻存储在内存中。但是当我们创建一个容器时，所有对象都必须在开始处理它们之前存储在该容器中。

例如，考虑一个处理日志文件的程序。一个非常简单的日志文件可能包含以下格式的信息：

```py
Jan 26, 2015 11:25:25    DEBUG        This is a debugging message.
Jan 26, 2015 11:25:36    INFO         This is an information method.
Jan 26, 2015 11:25:46    WARNING      This is a warning. It could be serious.
Jan 26, 2015 11:25:52    WARNING      Another warning sent.
Jan 26, 2015 11:25:59    INFO         Here's some information.
Jan 26, 2015 11:26:13    DEBUG        Debug messages are only useful if you want to figure something out.
Jan 26, 2015 11:26:32    INFO         Information is usually harmless, but helpful.
Jan 26, 2015 11:26:40    WARNING      Warnings should be heeded.
Jan 26, 2015 11:26:54    WARNING      Watch for warnings.
```

流行的网络服务器、数据库或电子邮件服务器的日志文件可能包含大量的数据（我最近不得不清理近 2TB 的日志文件）。如果我们想处理日志中的每一行，我们不能使用列表理解；它会创建一个包含文件中每一行的列表。这可能不适合在 RAM 中，并且可能会使计算机陷入困境，这取决于操作系统。

如果我们在日志文件上使用`for`循环，我们可以在将下一行读入内存之前一次处理一行。如果我们能使用理解语法来达到相同的效果，那不是很好吗？

这就是生成器表达式的用武之地。它们使用与理解相同的语法，但它们不创建最终的容器对象。要创建生成器表达式，将理解包装在`()`中，而不是`[]`或`{}`。

以下代码解析了以前呈现格式的日志文件，并输出了一个只包含`WARNING`行的新日志文件：

```py
import sys

inname = sys.argv[1]
outname = sys.argv[2]

with open(inname) as infile:
    with open(outname, "w") as outfile:
 **warnings = (l for l in infile if 'WARNING' in l)
        for l in warnings:
            outfile.write(l)
```

这个程序在命令行上接受两个文件名，使用生成器表达式来过滤警告（在这种情况下，它使用`if`语法，并且保持行不变），然后将警告输出到另一个文件。如果我们在示例文件上运行它，输出如下：

```py
Jan 26, 2015 11:25:46    WARNING     This is a warning. It could be serious.
Jan 26, 2015 11:25:52    WARNING     Another warning sent.
Jan 26, 2015 11:26:40    WARNING     Warnings should be heeded.
Jan 26, 2015 11:26:54    WARNING     Watch for warnings.
```

当然，对于这样一个短的输入文件，我们可以安全地使用列表理解，但是如果文件有数百万行，生成器表达式将对内存和速度产生巨大影响。

生成器表达式在函数调用内部经常最有用。例如，我们可以对生成器表达式调用`sum`、`min`或`max`，而不是列表，因为这些函数一次处理一个对象。我们只对结果感兴趣，而不关心任何中间容器。

一般来说，应尽可能使用生成器表达式。如果我们实际上不需要列表、集合或字典，而只需要过滤或转换序列中的项目，生成器表达式将是最有效的。如果我们需要知道列表的长度，或对结果进行排序、去除重复项或创建字典，我们将不得不使用理解语法。

# 生成器

生成器表达式实际上也是一种理解；它将更高级（这次确实更高级！）的生成器语法压缩成一行。更高级的生成器语法看起来甚至不那么面向对象，但我们将发现，它再次是一种简单的语法快捷方式，用于创建一种对象。

让我们进一步看一下日志文件的例子。如果我们想要从输出文件中删除`WARNING`列（因为它是多余的：这个文件只包含警告），我们有几种选择，不同的可读性级别。我们可以使用生成器表达式来实现：

```py
import sys
inname, outname = sys.argv[1:3]

with open(inname) as infile:
    with open(outname, "w") as outfile:
 **warnings = (l.replace('\tWARNING', '')
 **for l in infile if 'WARNING' in l)
        for l in warnings:
            outfile.write(l)
```

这是完全可读的，尽管我不想使表达式比那更复杂。我们也可以使用普通的`for`循环来实现：

```py
import sys
inname, outname = sys.argv[1:3]

with open(inname) as infile:
    with open(outname, "w") as outfile:
 **for l in infile:
 **if 'WARNING' in l:
 **outfile.write(l.replace('\tWARNING', ''))

```

这是可维护的，但在如此少的行数中有这么多级别的缩进有点丑陋。更令人担忧的是，如果我们想对这些行做一些不同的事情，而不仅仅是打印它们，我们也必须复制循环和条件代码。现在让我们考虑一个真正面向对象的解决方案，没有任何捷径：

```py
import sys
inname, outname = sys.argv[1:3]

class WarningFilter:
 **def __init__(self, insequence):
 **self.insequence = insequence
 **def __iter__(self):
 **return self
 **def __next__(self):
 **l = self.insequence.readline()
 **while l and 'WARNING' not in l:
 **l = self.insequence.readline()
 **if not l:
 **raise StopIteration
 **return l.replace('\tWARNING', '')

with open(inname) as infile:
    with open(outname, "w") as outfile:
        filter = WarningFilter(infile)
        for l in filter:
            outfile.write(l)
```

毫无疑问：这是如此丑陋和难以阅读，以至于你甚至可能无法理解发生了什么。我们创建了一个以文件对象为输入的对象，并提供了一个像任何迭代器一样的`__next__`方法。

这个`__next__`方法从文件中读取行，如果它们不是`WARNING`行，则将它们丢弃。当它遇到`WARNING`行时，它会返回它。然后`for`循环将再次调用`__next__`来处理下一个`WARNING`行。当我们用尽行时，我们引发`StopIteration`来告诉循环我们已经完成迭代。与其他例子相比，这看起来相当丑陋，但也很强大；既然我们手头有一个类，我们可以随心所欲地使用它。

有了这个背景，我们终于可以看到生成器的实际应用了。下一个例子与前一个例子*完全*相同：它创建了一个带有`__next__`方法的对象，当输入用尽时会引发`StopIteration`。

```py
import sys
inname, outname = sys.argv[1:3]

def warnings_filter(insequence):
 **for l in insequence:
 **if 'WARNING' in l:
 **yield l.replace('\tWARNING', '')

with open(inname) as infile:
    with open(outname, "w") as outfile:
        filter = warnings_filter(infile)
        for l in filter:
            outfile.write(l)
```

好吧，这看起来相当可读，也许...至少很简短。但这到底是怎么回事，一点道理也没有。`yield`又是什么？

事实上，`yield`是生成器的关键。当 Python 在函数中看到`yield`时，它会将该函数包装成一个对象，类似于我们前面例子中的对象。将`yield`语句视为类似于`return`语句；它退出函数并返回一行。然而，当函数再次被调用（通过`next()`）时，它将从上次离开的地方开始——在`yield`语句之后的行——而不是从函数的开头开始。在这个例子中，`yield`语句之后没有行，所以它跳到`for`循环的下一个迭代。由于`yield`语句在`if`语句内，它只会产生包含`WARNING`的行。

虽然看起来像是一个函数在循环处理行，但实际上它创建了一种特殊类型的对象，即生成器对象：

```py
>>> print(warnings_filter([]))
<generator object warnings_filter at 0xb728c6bc>

```

我将一个空列表传递给函数，作为迭代器。函数所做的就是创建并返回一个生成器对象。该对象上有`__iter__`和`__next__`方法，就像我们在前面的例子中创建的那样。每当调用`__next__`时，生成器运行函数，直到找到一个`yield`语句。然后返回`yield`的值，下次调用`__next__`时，它将从上次离开的地方继续。

这种生成器的使用并不是很高级，但如果你没有意识到函数正在创建一个对象，它可能看起来像魔术。这个例子很简单，但通过在单个函数中多次调用`yield`，你可以获得非常强大的效果；生成器将简单地从最近的`yield`开始，并继续到下一个`yield`。

## 从另一个可迭代对象中产生值

通常，当我们构建一个生成器函数时，我们最终会处于这样一种情况：我们希望从另一个可迭代对象中产生数据，可能是我们在生成器内部构造的列表推导或生成器表达式，或者是一些外部传递到函数中的项目。以前一直可以通过循环遍历可迭代对象并逐个产生每个项目来实现这一点。然而，在 Python 3.3 版本中，Python 开发人员引入了一种新的语法，使这一点更加优雅。

让我们稍微调整一下生成器示例，使其不再接受一系列行，而是接受一个文件名。这通常会被认为是不好的，因为它将对象与特定的范例联系在一起。在可能的情况下，我们应该操作输入的迭代器；这样，相同的函数可以在日志行来自文件、内存或基于网络的日志聚合器的情况下使用。因此，以下示例是为了教学目的而人为构造的。

这个版本的代码说明了你的生成器可以在从另一个可迭代对象（在本例中是生成器表达式）产生信息之前做一些基本的设置：

```py
import sys
inname, outname = sys.argv[1:3]

def warnings_filter(infilename):
    with open(infilename) as infile:
 **yield from (
 **l.replace('\tWARNING', '')
 **for l in infile
 **if 'WARNING' in l
 **)

filter = warnings_filter(inname)
with open(outname, "w") as outfile:
    for l in filter:
        outfile.write(l)
```

这段代码将前面示例中的`for`循环合并到了一个生成器表达式中。请注意，我将生成器表达式的三个子句（转换、循环和过滤）放在不同的行上，以使它们更易读。还要注意，这种转换并没有帮助太多；前面的`for`循环示例更易读。

因此，让我们考虑一个比其替代方案更易读的示例。构建一个生成器，从多个其他生成器中产生数据，这是有用的。例如，`itertools.chain`函数按顺序从可迭代对象中产生数据，直到它们全部耗尽。这可以使用`yield from`语法实现得太容易了，因此让我们考虑一个经典的计算机科学问题：遍历一棵通用树。

通用树数据结构的常见实现是计算机的文件系统。让我们模拟 Unix 文件系统中的一些文件夹和文件，以便我们可以使用`yield from`有效地遍历它们：

```py
class File:
    def __init__(self, name):
        self.name = name

class Folder(File):
    def __init__(self, name):
        super().__init__(name)
        self.children = []

root = Folder('')
etc = Folder('etc')
root.children.append(etc)
etc.children.append(File('passwd'))
etc.children.append(File('groups'))
httpd = Folder('httpd')
etc.children.append(httpd)
httpd.children.append(File('http.conf'))
var = Folder('var')
root.children.append(var)
log = Folder('log')
var.children.append(log)
log.children.append(File('messages'))
log.children.append(File('kernel'))
```

这个设置代码看起来很费力，但在一个真实的文件系统中，它会更加复杂。我们需要从硬盘中读取数据，并将其结构化成树。然而，一旦在内存中，输出文件系统中的每个文件的代码就非常优雅。

```py
def walk(file):
    if isinstance(file, Folder):
 **yield file.name + '/'
        for f in file.children:
 **yield from walk(f)
    else:
 **yield file.name

```

如果这段代码遇到一个目录，它会递归地要求`walk()`生成其每个子目录下所有文件的列表，然后产生所有这些数据以及自己的文件名。在它遇到一个普通文件的简单情况下，它只会产生那个文件名。

顺便说一句，解决前面的问题而不使用生成器是非常棘手的，以至于这个问题是一个常见的面试问题。如果你像这样回答，准备好让你的面试官既印象深刻又有些恼火，因为你回答得太容易了。他们可能会要求你解释到底发生了什么。当然，有了你在本章学到的原则，你不会有任何问题。

`yield from`语法在编写链式生成器时是一个有用的快捷方式，但它更常用于不同的目的：通过协程传输数据。我们将在第十三章中看到许多这样的例子，但现在，让我们先了解一下协程是什么。

# 协程

协程是非常强大的构造，经常被误解为生成器。许多作者不恰当地将协程描述为“带有一些额外语法的生成器”。这是一个容易犯的错误，因为在 Python 2.5 时引入协程时，它们被介绍为“我们在生成器语法中添加了一个`send`方法”。这更加复杂的是，当你在 Python 中创建一个协程时，返回的对象是一个生成器。实际上，区别要微妙得多，在你看到一些例子之后会更有意义。

### 注意

虽然 Python 中的协程目前与生成器语法紧密耦合，但它们与我们讨论过的迭代器协议只是表面上相关。即将发布的 Python 3.5 版本将使协程成为一个真正独立的对象，并提供一种新的语法来处理它们。

另一件需要记住的事情是，协程很难理解。它们在实际中并不经常使用，你可能会在 Python 中开发多年而不会错过或甚至遇到它们。有一些库广泛使用协程（主要用于并发或异步编程），但它们通常是这样编写的，以便你可以在不实际理解它们如何工作的情况下使用协程！所以如果你在这一节迷失了方向，不要绝望。

但是你不会迷失方向，因为已经学习了以下示例。这是最简单的协程之一；它允许我们保持一个可以通过任意值增加的累加值：

```py
def tally():
    score = 0
    while True:
 **increment = yield score
        score += increment
```

这段代码看起来像不可能工作的黑魔法，所以我们将在逐行描述之前看到它的工作原理。这个简单的对象可以被棒球队的记分应用程序使用。可以为每个团队保留单独的计分，并且他们的得分可以在每个半局结束时递增。看看这个交互式会话：

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

首先我们构造两个`tally`对象，一个用于每个团队。是的，它们看起来像函数，但与上一节中的生成器对象一样，函数内部有`yield`语句告诉 Python 要花大量精力将简单函数转换为对象。

然后我们对每个协程对象调用`next()`。这与对任何生成器调用`next`的操作相同，也就是说，它执行代码的每一行，直到遇到`yield`语句，返回该点的值，然后*暂停*直到下一个`next()`调用。

到目前为止，没有什么新鲜的。但是回顾一下我们协程中的`yield`语句：

```py
increment = yield score
```

与生成器不同，这个 yield 函数看起来应该返回一个值并将其分配给一个变量。事实上，这正是发生的事情。协程仍然在`yield`语句处暂停，等待通过另一个`next()`调用再次激活。

或者，正如你在交互式会话中看到的那样，调用一个名为`send()`的方法。`send()`方法与`next()`完全相同，只是除了将生成器推进到下一个`yield`语句外，它还允许你从生成器外部传入一个值。这个值被分配给`yield`语句的左侧。

对于许多人来说，真正令人困惑的是这发生的顺序：

+   `yield`发生，生成器暂停

+   `send()`来自函数外部，生成器被唤醒

+   发送的值被分配给`yield`语句的左侧

+   生成器继续处理，直到遇到另一个`yield`语句

因此，在这个特定的示例中，我们构造了协程并通过调用`next()`将其推进到`yield`语句，然后每次调用`send()`都会将一个值传递给协程，协程将这个值加到其分数中，然后返回到`while`循环的顶部，并继续处理直到达到`yield`语句。`yield`语句返回一个值，这个值成为最近一次`send`调用的返回值。不要错过：`send()`方法不仅仅提交一个值给生成器，它还返回即将到来的`yield`语句的值，就像`next()`一样。这就是我们定义生成器和协程之间的区别的方式：生成器只产生值，而协程也可以消耗值。

### 注意

`next(i)`，`i.__next__()`和`i.send(value)`的行为和语法相当不直观和令人沮丧。第一个是普通函数，第二个是特殊方法，最后一个是普通方法。但是这三个都是做同样的事情：推进生成器直到产生一个值并暂停。此外，`next()`函数和相关方法可以通过调用`i.send(None)`来复制。在这里有两个不同的方法名称是有价值的，因为它有助于我们的代码读者轻松地看到他们是在与协程还是生成器进行交互。我只是觉得在某些情况下，它是一个函数调用，而在另一种情况下，它是一个普通方法，有点令人恼火。

## 回到日志解析

当然，前面的示例也可以很容易地使用几个整数变量和在它们上调用`x += increment`来编写。让我们看一个第二个示例，其中协程实际上为我们节省了一些代码。这个例子是我在真实工作中不得不解决的问题的一个简化版本（出于教学目的）。它从先前关于处理日志文件的讨论中逻辑上延伸出来，这完全是偶然的；那些示例是为本书的第一版编写的，而这个问题是四年后出现的！

Linux 内核日志包含看起来有些类似但又不完全相同的行：

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

有许多交错的内核日志消息，其中一些与硬盘有关。硬盘消息可能与其他消息交错，但它们以可预测的格式和顺序出现，其中具有已知序列号的特定驱动器与总线标识符（如`0:0:0:0`）相关联，并且与该总线相关联的块设备标识符（如`sda`）。最后，如果驱动器的文件系统损坏，它可能会出现 XFS 错误。

现在，考虑到前面的日志文件，我们需要解决的问题是如何获取任何存在 XFS 错误的驱动器的序列号。稍后，数据中心技术人员可能会使用这个序列号来识别并更换驱动器。

我们知道可以使用正则表达式识别各行，但是我们必须在循环遍历行时更改正则表达式，因为根据之前找到的内容，我们将寻找不同的内容。另一个困难的地方是，如果我们找到错误字符串，关于包含该字符串的总线以及附加到该总线上的驱动器的序列号的信息已经被处理。通过以相反的顺序迭代文件的行，这个问题很容易解决。

在查看此示例之前，请注意 - 基于协程的解决方案所需的代码量非常少，令人不安：

```py
import re

def match_regex(filename, regex):
    with open(filename) as file:
        lines = file.readlines()
    for line in reversed(lines):
        match = re.match(regex, line)
        if match:
 **regex = yield match.groups()[0]

def get_serials(filename):
    ERROR_RE = 'XFS ERROR (\[sd[a-z]\])'
 **matcher = match_regex(filename, ERROR_RE)
    device = next(matcher)
    while True:
        bus = matcher.send(
            '(sd \S+) {}.*'.format(re.escape(device)))
        serial = matcher.send('{} \(SERIAL=([^)]*)\)'.format(bus))
 **yield serial
        device = matcher.send(ERROR_RE)

for serial_number in get_serials('EXAMPLE_LOG.log'):
    print(serial_number)
```

这段代码将工作分为两个独立的任务。第一个任务是循环遍历所有行，并输出与给定正则表达式匹配的任何行。第二个任务是与第一个任务进行交互，并指导它在任何给定时间搜索什么正则表达式。

首先看一下`match_regex`协程。记住，它在构造时不执行任何代码；相反，它只是创建一个协程对象。一旦构造完成，协程外部的某人最终会调用`next()`来启动代码运行，此时它会存储两个变量`filename`和`regex`的状态。然后它读取文件中的所有行并以相反的顺序对它们进行迭代。将传入的每一行与正则表达式进行比较，直到找到匹配项。当找到匹配项时，协程会产生正则表达式的第一个组并等待。

在将来的某个时候，其他代码将发送一个新的正则表达式进行搜索。请注意，协程从不关心它试图匹配什么正则表达式；它只是循环遍历行并将它们与正则表达式进行比较。决定提供什么正则表达式是其他人的责任。

在这种情况下，那个“别人”是`get_serials`生成器。它不关心文件中的行，事实上它甚至不知道它们。它做的第一件事是从`match_regex`协程构造函数创建一个`matcher`对象，并给它一个默认的正则表达式来搜索。它将协程推进到它的第一个`yield`并存储它返回的值。然后它进入一个循环，指示匹配器对象根据存储的设备 ID 搜索总线 ID，然后根据该总线 ID 搜索序列号。

在向外部的`for`循环中空闲地产生该序列号，然后指示匹配器查找另一个设备 ID 并重复循环。

基本上，协程（`match_regex`，因为它使用`regex = yield`语法）的工作是在文件中搜索下一个重要的行，而生成器（`get_serial`，它使用没有赋值的`yield`语法）的工作是决定哪一行是重要的。生成器有关于这个特定问题的信息，比如文件中行的顺序。另一方面，协程可以插入到任何需要搜索文件以获取给定正则表达式的问题中。

## 关闭协程和抛出异常

普通生成器通过引发`StopIteration`来从内部信号退出。如果我们将多个生成器链接在一起（例如通过在另一个生成器内部迭代一个生成器），`StopIteration`异常将被传播到外部。最终，它将触发一个`for`循环，看到异常并知道是时候退出循环了。

协程通常不遵循迭代机制；而不是通过一个直到遇到异常的数据，通常是将数据推送到其中（使用`send`）。通常是负责推送的实体告诉协程何时完成；它通过在相关协程上调用`close()`方法来实现这一点。

当调用`close()`方法时，将在协程等待发送值的点引发一个`GeneratorExit`异常。通常，协程应该将它们的`yield`语句包装在`try`...`finally`块中，以便执行任何清理任务（例如关闭关联的文件或套接字）。

如果我们需要在协程内部引发异常，我们可以类似地使用`throw()`方法。它接受一个异常类型，可选的`value`和`traceback`参数。当我们在一个协程中遇到异常并希望在相邻的协程中引发异常时，后者是非常有用的，同时保持回溯。

如果您正在构建健壮的基于协程的库，这两个功能都是至关重要的，但在日常编码生活中，我们不太可能遇到它们。

## 协程、生成器和函数之间的关系

我们已经看到协程的运行情况，现在让我们回到讨论它们与生成器的关系。在 Python 中，就像往常一样，这个区别是相当模糊的。事实上，所有的协程都是生成器对象，作者经常交替使用这两个术语。有时，他们将协程描述为生成器的一个子集（只有从`yield`返回值的生成器被认为是协程）。在 Python 中，这在技术上是正确的，正如我们在前面的部分中所看到的。

然而，在更广泛的理论计算机科学领域，协程被认为是更一般的原则，生成器是协程的一种特定类型。此外，普通函数是协程的另一个独特子集。

协程是一个可以在一个或多个点传入数据并在一个或多个点获取数据的例程。在 Python 中，数据传入和传出的点是`yield`语句。

函数，或子例程，是最简单的协程类型。你可以在一个点传入数据，并在函数返回时在另一个点获取数据。虽然函数可以有多个`return`语句，但对于任何给定的函数调用，只能调用其中一个。

最后，生成器是一种协程类型，可以在一个点传入数据，但可以在多个点传出数据。在 Python 中，数据将在`yield`语句处传出，但你不能将数据传回。如果你调用了`send`，数据将被悄悄丢弃。

所以理论上，生成器是协程的一种类型，函数是协程的一种类型，还有一些既不是函数也不是生成器的协程。够简单吧？那为什么在 Python 中感觉更复杂呢？

在 Python 中，生成器和协程都是使用看起来像是构造函数的语法构造的。但是生成的对象根本不是函数；它是一种完全不同类型的对象。函数当然也是对象。但它们有不同的接口；函数是可调用的并返回值，生成器使用`next()`提取数据，协程使用`send`推送数据。

# 案例研究

Python 目前最流行的领域之一是数据科学。让我们实现一个基本的机器学习算法！机器学习是一个庞大的主题，但总体思想是利用从过去数据中获得的知识对未来数据进行预测或分类。这些算法的用途很广泛，数据科学家们每天都在找到新的应用机器学习的方法。一些重要的机器学习应用包括计算机视觉（如图像分类或人脸识别）、产品推荐、识别垃圾邮件和语音识别。我们将研究一个更简单的问题：给定一个 RGB 颜色定义，人们会将该颜色识别为什么名称？

在标准 RGB 颜色空间中有超过 1600 万种颜色，人类只为其中的一小部分制定了名称。虽然有成千上万的名称（有些相当荒谬；只需去任何汽车经销商或化妆品商店），让我们构建一个试图将 RGB 空间划分为基本颜色的分类器：

+   红色

+   紫色

+   蓝色

+   绿色

+   黄色

+   橙色

+   灰色

+   白色

+   粉色

我们需要的第一件事是一个数据集来训练我们的算法。在生产系统中，你可能会从*颜色列表*网站上获取数据，或者对成千上万的人进行调查。相反，我创建了一个简单的应用程序，它会渲染一个随机颜色，并要求用户选择前述九个选项中的一个来对其进行分类。这个应用程序包含在本章的示例代码中的`kivy_color_classifier`目录中，但我们不会详细介绍这段代码，因为它在这里的唯一目的是生成样本数据。

### 注

Kivy 有一个非常精心设计的面向对象的 API，你可能想自己探索一下。如果你想开发可以在许多系统上运行的图形程序，从你的笔记本电脑到你的手机，你可能想看看我的书《在 Kivy 中创建应用》，*O'Reilly*。

在这个案例研究中，该应用程序的重要之处在于输出，这是一个包含每行四个值的**逗号分隔值**（**CSV**）文件：红色、绿色和蓝色值（表示为 0 到 1 之间的浮点数），以及用户为该颜色分配的前述九个名称中的一个。数据集看起来像这样：

```py
0.30928279150905513,0.7536768153744394,0.3244011790604804,Green
0.4991001855115986,0.6394567277907686,0.6340502030888825,Grey
0.21132621004927998,0.3307376167520666,0.704037576789711,Blue
0.7260420945787928,0.4025279573860123,0.49781705131696363,Pink
0.706469868610228,0.28530423638868196,0.7880240251003464,Purple
0.692243900051664,0.7053550777777416,0.1845069151913028,Yellow
0.3628979381122397,0.11079495501215897,0.26924540840045075,Purple
0.611273677646518,0.48798521783547677,0.5346130557761224,Purple
.
.
.
0.4014121109376566,0.42176706818252674,0.9601866228083298,Blue
0.17750449496124632,0.8008214961070862,0.5073944321437429,Green
```

在我感到无聊并决定开始对这个数据集进行机器学习之前，我制作了 200 个数据点（其中很少有不真实的数据）。如果你想使用我的数据（没有人告诉我我是色盲，所以应该是相当合理的），这些数据点已经包含在本章的示例中。

我们将实现一个较简单的机器学习算法，称为 k 最近邻算法。该算法依赖于数据集中点之间的某种“距离”计算（在我们的情况下，我们可以使用三维版本的毕达哥拉斯定理）。给定一个新的数据点，它找到一定数量（称为 k，如 k 最近邻）的数据点，这些数据点在通过该距离计算时最接近它。然后以某种方式组合这些数据点（对于线性计算，平均值可能有效；对于我们的分类问题，我们将使用众数），并返回结果。

我们不会过多地讨论算法的具体内容；相反，我们将专注于如何将迭代器模式或迭代器协议应用于这个问题。

现在，让我们编写一个程序，按顺序执行以下步骤：

1.  从文件中加载样本数据并构建模型。

1.  生成 100 种随机颜色。

1.  对每种颜色进行分类，并将其输出到与输入相同格式的文件中。

一旦有了这个第二个 CSV 文件，另一个 Kivy 程序可以加载文件并渲染每种颜色，要求人类用户确认或否认预测的准确性，从而告诉我们我们的算法和初始数据集的准确性如何。

第一步是一个相当简单的生成器，它加载 CSV 数据并将其转换为符合我们需求的格式：

```py
import csv

dataset_filename = 'colors.csv'

def load_colors(filename):
    with open(filename) as dataset_file:
 **lines = csv.reader(dataset_file)
        for line in lines:
 **yield tuple(float(y) for y in line[0:3]), line[3]

```

我们以前没有见过`csv.reader`函数。它返回文件中行的迭代器。迭代器返回的每个值都是一个字符串列表。在我们的情况下，我们可以只是按逗号分割就可以了，但`csv.reader`还负责处理引号和逗号分隔值格式的各种其他细微差别。

然后我们循环遍历这些行，并将它们转换为颜色和名称的元组，其中颜色是由三个浮点值整数组成的元组。这个元组是使用生成器表达式构造的。可能有更易读的方法来构造这个元组；你认为生成器表达式的代码简洁和速度是否值得混淆？它不是返回一个颜色元组的列表，而是逐个产生它们，从而构造一个生成器对象。

现在，我们需要一百种随机颜色。有很多方法可以做到这一点：

+   使用嵌套生成器表达式的列表推导：`[tuple(random() for r in range(3)) for r in range(100)]`

+   一个基本的生成器函数

+   一个实现`__iter__`和`__next__`协议的类

+   将数据通过一系列协程

+   甚至只是一个基本的`for`循环

生成器版本似乎是最易读的，所以让我们将该函数添加到我们的程序中：

```py
from random import random

def generate_colors(count=100):
    for i in range(count):
 **yield (random(), random(), random())

```

注意我们对要生成的颜色数量进行了参数化。现在我们可以在将来的其他生成颜色任务中重用这个函数。

现在，在进行分类步骤之前，我们需要一个函数来计算两种颜色之间的“距离”。由于可以将颜色看作是三维的（例如，红色、绿色和蓝色可以映射到*x*、*y*和*z*轴），让我们使用一些基本的数学：

```py
import math

def color_distance(color1, color2):
    channels = zip(color1, color2)
    sum_distance_squared = 0
    for c1, c2 in channels:
        sum_distance_squared += (c1 - c2) ** 2
    return math.sqrt(sum_distance_squared)
```

这是一个看起来非常基本的函数；它似乎甚至没有使用迭代器协议。没有`yield`函数，也没有推导。然而，有一个`for`循环，而且`zip`函数的调用也在进行一些真正的迭代（记住`zip`会产生包含每个输入迭代器中一个元素的元组）。

然而，需要注意的是，这个函数将在我们的 k 最近邻算法中被调用很多次。如果我们的代码运行得太慢，并且我们能够确定这个函数是瓶颈，我们可能希望用一个不太易读但更优化的生成器表达式来替换它：

```py
def color_distance(color1, color2):
    return math.sqrt(sum((x[0] - x[1]) ** 2 for x in zip(
    color1, color2)))
```

然而，我强烈建议在证明可读版本太慢之前不要进行这样的优化。

现在我们已经有了一些管道，让我们来实际做 k 最近邻实现。这似乎是使用协程的好地方。下面是一些测试代码，以确保它产生合理的值：

```py
def nearest_neighbors(model_colors, num_neighbors):
    model = list(model_colors)
 **target = yield
    while True:
        distances = sorted(
            ((color_distance(c[0], target), c) for c in model),
        )
 **target = yield [
 **d[1] for d in distances[0:num_neighbors]
 **]

model_colors = load_colors(dataset_filename)
target_colors = generate_colors(3)
get_neighbors = nearest_neighbors(model_colors, 5)
next(get_neighbors)

for color in target_colors:
    distances = get_neighbors.send(color)
    print(color)
    for d in distances:
        print(color_distance(color, d[0]), d[1])
```

该协程接受两个参数，要用作模型的颜色列表和要查询的邻居数。它将模型转换为列表，因为它将被多次迭代。在协程的主体中，它使用`yield`语法接受一个 RGB 颜色值的元组。然后它将`sorted`调用与一个奇怪的生成器表达式结合在一起。看看你是否能弄清楚那个生成器表达式在做什么。

它为模型中的每种颜色返回一个`(distance, color_data)`元组。请记住，模型本身包含`(color, name)`的元组，其中`color`是三个 RGB 值的元组。因此，该生成器返回一个奇怪数据结构的迭代器，看起来像这样：

```py
(distance, (r, g, b), color_name)
```

然后，`sorted`调用按照它们的第一个元素（距离）对结果进行排序。这是一段复杂的代码，根本不是面向对象的。您可能希望将其分解为一个普通的`for`循环，以确保您理解生成器表达式的工作原理。如果您将一个键参数传递给`sorted`函数而不是构造一个元组，想象一下这段代码会是什么样子也是一个很好的练习。

`yield`语句稍微复杂一些；它从前 k 个`(distance, color_data)`元组中提取第二个值。更具体地说，它为距离最近的 k 个值产生了`((r, g, b), color_name)`元组。或者，如果您更喜欢更抽象的术语，它为给定模型中目标的 k 个最近邻产生了值。

剩下的代码只是测试这种方法的样板；它构造了模型和颜色生成器，启动了协程，并在`for`循环中打印结果。

剩下的两个任务是根据最近邻选择颜色，并将结果输出到 CSV 文件。让我们创建两个协程来处理这些任务。我们先做输出，因为它可以独立测试：

```py
def write_results(filename="output.csv"):
    with open(filename, "w") as file:
        writer = csv.writer(file)
        while True:
 **color, name = yield
            writer.writerow(list(color) + [name])

results = write_results()
next(results)
for i in range(3):
    print(i)
    results.send(((i, i, i), i * 10))
```

该协程将一个打开的文件作为状态，并在使用`send()`发送的情况下将代码行写入其中。测试代码确保协程正常工作，所以现在我们可以用第三个协程连接这两个协程了。

第二个协程使用了一个有点奇怪的技巧：

```py
from collections import Counter
def name_colors(get_neighbors):
 **color = yield
    while True:
 **near = get_neighbors.send(color)
        name_guess = Counter(
            n[1] for n in near).most_common(1)[0][0]
 **color = yield name_guess

```

这个协程接受一个现有的协程作为参数。在这种情况下，它是`nearest_neighbors`的一个实例。这段代码基本上通过`nearest_neighbors`实例代理所有发送到它的值。然后它对结果进行一些处理，以获取返回的值中最常见的颜色。在这种情况下，也许将原始协程调整为返回一个名称会更有意义，因为它没有被用于其他任何事情。然而，有许多情况下传递协程是有用的；这就是我们的做法。

现在，我们所要做的就是将这些不同的协程和管道连接在一起，并通过一个单一的函数调用启动整个过程：

```py
def process_colors(dataset_filename="colors.csv"):
    model_colors = load_colors(dataset_filename)
    get_neighbors = nearest_neighbors(model_colors, 5)
 **get_color_name = name_colors(get_neighbors)
    output = write_results()
 **next(output)
 **next(get_neighbors)
 **next(get_color_name)

    for color in generate_colors():
 **name = get_color_name.send(color)
 **output.send((color, name))

process_colors()
```

因此，与我们定义的几乎所有其他函数不同，这个函数是一个完全正常的函数，没有任何`yield`语句。它不会被转换为协程或生成器对象。但是，它确实构造了一个生成器和三个协程。请注意，`get_neighbors`协程是如何传递给`name_colors`构造函数的。注意所有三个协程是如何通过调用`next`推进到它们的第一个`yield`语句的。

一旦所有管道都创建好了，我们就使用`for`循环将生成的每种颜色发送到`get_color_name`协程中，然后将该协程产生的每个值传送到输出协程，将其写入文件。

就是这样！我创建了第二个 Kivy 应用程序，加载了生成的 CSV 文件，并将颜色呈现给用户。用户可以根据他们认为机器学习算法的选择是否与他们的选择相匹配来选择*是*或*否*。这并不科学准确（容易出现观察偏差），但对于玩耍来说已经足够了。用我的眼睛看，它成功率约为 84%，比我 12 年级的平均成绩要好。对于我们第一次的机器学习经历来说，这已经不错了，对吧？

你可能会想，“这与面向对象编程有什么关系？这段代码中甚至没有一个类！”在某些方面，你是对的；协程和生成器通常不被认为是面向对象的。然而，创建它们的函数会返回对象；实际上，你可以将这些函数看作构造函数。构造的对象具有适当的`send()`和`__next__()`方法。基本上，协程/生成器语法是一种特定类型的对象的语法快捷方式，如果没有它，创建这种对象会非常冗长。

这个案例研究是一个自下而上设计的练习。我们创建了各种低级对象，执行特定的任务，并在最后将它们全部连接在一起。我发现这在开发协程时是一个常见的做法。另一种选择，自上而下的设计有时会导致更多的代码块而不是独特的个体。总的来说，我们希望在太大和太小的方法之间找到一个合适的平衡，以及它们如何组合在一起。当然，这是真的，无论是否像我们在这里做的那样使用迭代器协议。

# 练习

如果你在日常编码中很少使用推导，那么你应该做的第一件事是搜索一些现有的代码，找到一些`for`循环。看看它们中是否有任何可以轻松转换为生成器表达式或列表、集合或字典推导的。

测试列表推导是否比`for`循环更快的说法。这可以通过内置的`timeit`模块来实现。使用`timeit.timeit`函数的帮助文档来了解如何使用它。基本上，编写两个做同样事情的函数，一个使用列表推导，一个使用`for`循环。将每个函数传递给`timeit.timeit`，并比较结果。如果你感到有冒险精神，也可以比较生成器和生成器表达式。使用`timeit`测试代码可能会让人上瘾，所以请记住，除非代码被执行了大量次数，比如在一个巨大的输入列表或文件上，否则代码不需要非常快。

玩转生成器函数。从需要多个值的基本迭代器开始（数学序列是典型的例子；如果你想不出更好的例子，斐波那契数列就太过于使用了）。尝试一些更高级的生成器，比如接受多个输入列表并以某种方式产生合并值的生成器。生成器也可以用在文件上；你能写一个简单的生成器来显示两个文件中相同的行吗？

协程滥用迭代器协议，但实际上并不满足迭代器模式。你能否构建一个从日志文件中获取序列号的非协程版本的代码？采用面向对象的方法，这样你就可以在一个类上存储额外的状态。如果你能创建一个可以替换现有协程的对象，你将学到很多关于协程的知识。

看看你是否能将案例研究中使用的协程抽象出来，以便可以在各种数据集上使用 k 最近邻算法。你可能希望构建一个接受其他协程或执行距离和重组计算的函数作为参数的协程，并调用这些函数来找到实际的最近邻。

# 总结

在本章中，我们了解到设计模式是有用的抽象，为常见的编程问题提供了“最佳实践”解决方案。我们介绍了我们的第一个设计模式，迭代器，以及 Python 使用和滥用这种模式的多种方式。原始的迭代器模式非常面向对象，但在代码编写时也相当丑陋和冗长。然而，Python 的内置语法将丑陋的部分抽象出来，为我们留下了一个清晰的接口来使用这些面向对象的构造。

理解和生成器表达式可以在一行中将容器构造与迭代结合起来。生成器对象可以使用`yield`语法来构造。协程看起来像生成器，但用途完全不同。

在接下来的两章中，我们将介绍更多的设计模式。
