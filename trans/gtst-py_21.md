# 迭代器模式

我们已经讨论了 Python 的许多内置功能和习语，乍一看似乎违反了面向对象的原则，但实际上在幕后提供了对真实对象的访问。在本章中，我们将讨论`for`循环，它似乎如此结构化，实际上是一组面向对象原则的轻量级包装。我们还将看到一系列扩展到这种语法，自动创建更多类型的对象。我们将涵盖以下主题：

+   设计模式是什么

+   迭代器协议-最强大的设计模式之一

+   列表、集合和字典推导

+   生成器和协程

# 简要介绍设计模式

当工程师和建筑师决定建造一座桥、一座塔或一座建筑时，他们遵循某些原则以确保结构完整性。桥梁有各种可能的设计（例如悬索和悬臂），但如果工程师不使用标准设计之一，并且没有一个杰出的新设计，那么他/她设计的桥梁可能会坍塌。

设计模式是试图将同样的正确设计结构的正式定义引入到软件工程中。有许多不同的设计模式来解决不同的一般问题。设计模式通常解决开发人员在某些特定情况下面临的特定常见问题。然后，设计模式是对该问题的理想解决方案的建议，从面向对象设计的角度来看。

了解设计模式并选择在软件中使用它并不保证我们正在创建一个*正确*的解决方案。1907 年，魁北克大桥（至今仍是世界上最长的悬臂桥）在建设完成之前坍塌，因为设计它的工程师严重低估了用于建造它的钢材重量。同样，在软件开发中，我们可能会错误地选择或应用设计模式，并创建在正常操作情况下或在超出原始设计限制时*崩溃*的软件。

任何一个设计模式都提出了一组以特定方式相互作用的对象，以解决一般问题。程序员的工作是识别何时面临这样一个特定版本的问题，然后选择和调整通用设计以满足其精确需求。

在本章中，我们将介绍迭代器设计模式。这种模式如此强大和普遍，以至于 Python 开发人员提供了多种语法来访问该模式的基础面向对象原则。我们将在接下来的两章中介绍其他设计模式。其中一些具有语言支持，而另一些则没有，但没有一个像迭代器模式那样成为 Python 程序员日常生活中的固有部分。

# 迭代器

在典型的设计模式术语中，迭代器是一个具有`next()`方法和`done()`方法的对象；后者如果序列中没有剩余项目，则返回`True`。在没有内置迭代器支持的编程语言中，迭代器将像这样循环：

```py
while not iterator.done(): 
    item = iterator.next() 
    # do something with the item 
```

在 Python 中，迭代是一种特殊的特性，因此该方法得到了一个特殊的名称`__next__`。可以使用内置的`next(iterator)`来访问此方法。Python 的迭代器协议不是使用`done`方法，而是引发`StopIteration`来通知循环已完成。最后，我们有更易读的`foriteminiterator`语法来实际访问迭代器中的项目，而不是使用`while`循环。让我们更详细地看看这些。

# 迭代器协议

`Iterator`抽象基类在`collections.abc`模块中定义了 Python 中的迭代器协议。正如前面提到的，它必须有一个`__next__`方法，`for`循环（以及其他支持迭代的功能）可以调用它来从序列中获取一个新元素。此外，每个迭代器还必须满足`Iterable`接口。任何提供`__iter__`方法的类都是可迭代的。该方法必须返回一个`Iterator`实例，该实例将覆盖该类中的所有元素。

这可能听起来有点混乱，所以看看以下示例，但请注意，这是解决这个问题的一种非常冗长的方式。它清楚地解释了迭代和所讨论的两个协议，但在本章的后面，我们将看到几种更易读的方法来实现这种效果：

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

这个例子首先构造了一个可迭代对象，并从中检索了一个迭代器。这种区别可能需要解释；可迭代对象是一个可以循环遍历的对象。通常，这些元素可以被多次循环遍历，甚至可能在同一时间或重叠的代码中。另一方面，迭代器代表可迭代对象中的特定位置；一些项目已被消耗，一些尚未被消耗。两个不同的迭代器可能在单词列表中的不同位置，但任何一个迭代器只能标记一个位置。

每次在迭代器上调用`next()`时，它都会按顺序从可迭代对象中返回另一个标记。最终，迭代器将被耗尽（不再有任何元素返回），在这种情况下会引发`Stopiteration`，然后我们跳出循环。

当然，我们已经知道了一个更简单的语法，用于从可迭代对象构造迭代器：

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

正如你所看到的，`for`语句，尽管看起来并不像面向对象，实际上是一种显而易见的面向对象设计原则的快捷方式。在讨论理解时，请记住这一点，因为它们似乎是面向对象工具的完全相反。然而，它们使用与`for`循环完全相同的迭代协议，只是另一种快捷方式。

# 理解

理解是一种简单但强大的语法，允许我们在一行代码中转换或过滤可迭代对象。结果对象可以是一个完全正常的列表、集合或字典，也可以是一个生成器表达式，可以在保持一次只有一个元素在内存中的情况下高效地消耗。

# 列表理解

列表理解是 Python 中最强大的工具之一，所以人们倾向于认为它们是高级的。事实并非如此。事实上，我已经在以前的例子中使用了理解，假设你会理解它们。虽然高级程序员确实经常使用理解，但并不是因为它们很高级。而是因为它们很简单，并处理了软件开发中最常见的一些操作。

让我们来看看其中一个常见操作；即，将一个项目列表转换为相关项目列表。具体来说，假设我们刚刚从文件中读取了一个字符串列表，现在我们想将其转换为整数列表。我们知道列表中的每个项目都是整数，并且我们想对这些数字进行一些操作（比如计算平均值）。以下是一种简单的方法：

```py
input_strings = ["1", "5", "28", "131", "3"]

output_integers = [] 
for num in input_strings: 
    output_integers.append(int(num)) 
```

这个方法很好用，而且只有三行代码。如果你不习惯理解，你可能不会觉得它看起来很丑陋！现在，看看使用列表理解的相同代码：

```py
input_strings = ["1", "5", "28", "131", "3"]
output_integers = [int(num) for num in input_strings] 
```

我们只剩下一行，而且，对于性能来说很重要的是，我们已经放弃了列表中每个项目的`append`方法调用。总的来说，即使你不习惯推导式语法，也很容易理解发生了什么。

方括号表示，我们正在创建一个列表。在这个列表中是一个`for`循环，它遍历输入序列中的每个项目。唯一可能令人困惑的是在列表的左大括号和`for`循环开始之间发生了什么。这里发生的事情应用于输入列表中的*每个*项目。所讨论的项目由循环中的`num`变量引用。因此，它对每个元素调用`int`函数，并将结果整数存储在新列表中。

这就是基本列表推导式的全部内容。推导式是高度优化的 C 代码；当循环遍历大量项目时，列表推导式比`for`循环要快得多。如果仅仅从可读性的角度来看，不能说服你尽可能多地使用它们，那么速度应该是一个令人信服的理由。

将一个项目列表转换为相关列表并不是列表推导式唯一能做的事情。我们还可以选择通过在推导式中添加`if`语句来排除某些值。看一下：

```py
output_integers = [int(num) for num in input_strings if len(num) < 3]
```

这个例子和前面的例子唯一不同的地方是`if len(num) < 3`部分。这个额外的代码排除了任何超过两个字符的字符串。`if`语句应用于**在**`int`函数之前的每个元素，因此它测试字符串的长度。由于我们的输入字符串在本质上都是整数，它排除了任何超过 99 的数字。

列表推导式用于将输入值映射到输出值，并在途中应用过滤器以包括或排除满足特定条件的任何值。

任何可迭代对象都可以成为列表推导式的输入。换句话说，任何我们可以放入`for`循环中的东西也可以放入推导式中。例如，文本文件是可迭代的；对文件的迭代器每次调用`__next__`都会返回文件的一行。我们可以使用`zip`函数将第一行是标题行的制表符分隔文件加载到字典中：

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

这一次，我添加了一些空白以使其更易读（列表推导式不一定要放在一行上）。这个例子从压缩的标题和分割行中创建了一个字典列表，对文件中的每一行进行了处理。

嗯，什么？如果那段代码或解释没有意义，不要担心；它很令人困惑。一个列表推导式在这里做了一堆工作，代码很难理解、阅读，最终也很难维护。这个例子表明，列表推导式并不总是最好的解决方案；大多数程序员都会同意，`for`循环比这个版本更可读。

记住：我们提供的工具不应该被滥用！始终选择适合工作的正确工具，这总是编写可维护代码。

# 集合和字典推导式

理解并不局限于列表。我们也可以使用类似的语法来创建集合和字典。让我们从集合开始。创建集合的一种方法是将列表推导式放入`set()`构造函数中，将其转换为集合。但是，为什么要浪费内存在一个被丢弃的中间列表上，当我们可以直接创建一个集合呢？

这是一个使用命名元组来模拟作者/标题/流派三元组的例子，然后检索写作特定流派的所有作者的集合：

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

与演示数据设置相比，突出显示的集合推导式确实很短！如果我们使用列表推导式，特里·普拉切特当然会被列出两次。事实上，集合的性质消除了重复项，我们最终得到了以下结果：

```py
>>> fantasy_authors
{'Turner', 'Pratchett', 'Le Guin'}  
```

仍然使用大括号，我们可以引入冒号来创建字典理解。这将使用*键:值*对将序列转换为字典。例如，如果我们知道标题，可能会很快地在字典中查找作者或流派。我们可以使用字典理解将标题映射到`books`对象：

```py
fantasy_titles = {b.title: b for b in books if b.genre == "fantasy"}
```

现在，我们有了一个字典，并且可以使用正常的语法按标题查找书籍。

总之，理解不是高级的 Python，也不是应该避免使用的*非面向对象*工具。它们只是一种更简洁和优化的语法，用于从现有序列创建列表、集合或字典。

# 生成器表达式

有时我们想处理一个新的序列，而不将新的列表、集合或字典拉入系统内存。如果我们只是一个接一个地循环遍历项目，并且实际上并不关心是否创建了一个完整的容器（如列表或字典），那么创建该容器就是浪费内存。当一次处理一个项目时，我们只需要当前对象在内存中的可用性。但是当我们创建一个容器时，所有对象都必须在开始处理它们之前存储在该容器中。

例如，考虑一个处理日志文件的程序。一个非常简单的日志文件可能以这种格式包含信息：

```py
Jan 26, 2015 11:25:25 DEBUG This is a debugging message. Jan 26, 2015 11:25:36 INFO This is an information method. Jan 26, 2015 11:25:46 WARNING This is a warning. It could be serious. Jan 26, 2015 11:25:52 WARNING Another warning sent. Jan 26, 2015 11:25:59 INFO Here's some information. Jan 26, 2015 11:26:13 DEBUG Debug messages are only useful if you want to figure something out. Jan 26, 2015 11:26:32 INFO Information is usually harmless, but helpful. Jan 26, 2015 11:26:40 WARNING Warnings should be heeded. Jan 26, 2015 11:26:54 WARNING Watch for warnings. 
```

流行的网络服务器、数据库或电子邮件服务器的日志文件可能包含大量的数据（我曾经不得不清理近 2TB 的日志文件）。如果我们想处理日志中的每一行，我们不能使用列表理解；它会创建一个包含文件中每一行的列表。这可能不适合在 RAM 中，并且可能会使计算机陷入困境，这取决于操作系统。

如果我们在日志文件上使用`for`循环，我们可以在将下一行读入内存之前一次处理一行。如果我们能使用理解语法来获得相同的效果，那不是很好吗？

这就是生成器表达式的用武之地。它们使用与理解相同的语法，但不创建最终的容器对象。要创建生成器表达式，将理解包装在`()`中，而不是`[]`或`{}`。

以下代码解析了以前介绍的格式的日志文件，并输出了一个只包含`WARNING`行的新日志文件：

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

该程序在命令行上获取两个文件名，使用生成器表达式来过滤警告（在这种情况下，它使用`if`语法并保持行不变），然后将警告输出到另一个文件。如果我们在示例文件上运行它，输出如下：

```py
Jan 26, 2015 11:25:46 WARNING This is a warning. It could be serious.
Jan 26, 2015 11:25:52 WARNING Another warning sent.
Jan 26, 2015 11:26:40 WARNING Warnings should be heeded.
Jan 26, 2015 11:26:54 WARNING Watch for warnings. 
```

当然，对于这样一个简短的输入文件，我们可以安全地使用列表理解，但是如果文件有数百万行，生成器表达式将对内存和速度产生巨大影响。

将`for`表达式括在括号中会创建一个生成器表达式，而不是元组。

生成器表达式通常在函数调用内最有用。例如，我们可以在生成器表达式上调用`sum`、`min`或`max`，而不是列表，因为这些函数一次处理一个对象。我们只对聚合结果感兴趣，而不关心任何中间容器。

总的来说，在四个选项中，尽可能使用生成器表达式。如果我们实际上不需要列表、集合或字典，而只需要过滤或转换序列中的项目，生成器表达式将是最有效的。如果我们需要知道列表的长度，或对结果进行排序、去除重复项或创建字典，我们将不得不使用理解语法。

# 生成器

生成器表达式实际上也是一种理解；它将更高级（这次确实更高级！）的生成器语法压缩成一行。更高级的生成器语法看起来甚至不那么面向对象，但我们将再次发现，这只是一种简单的语法快捷方式，用于创建一种对象。

让我们进一步考虑一下日志文件示例。如果我们想要从输出文件中删除“WARNING”列（因为它是多余的：这个文件只包含警告），我们有几种不同级别的可读性选项。我们可以使用生成器表达式来实现：

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

尽管如此，这是完全可读的，但我不想使表达式比这更复杂。我们也可以使用普通的“for”循环来实现：

```py
with open(inname) as infile:
    with open(outname, "w") as outfile:
        for l in infile:
            if "WARNING" in l:
                outfile.write(l.replace("\tWARNING", ""))
```

这显然是可维护的，但在如此少的行数中有如此多级缩进有点丑陋。更令人担忧的是，如果我们想要做一些其他事情而不是简单地打印出行，我们还必须复制循环和条件代码。

现在让我们考虑一个真正面向对象的解决方案，没有任何捷径：

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

毫无疑问：这太丑陋和难以阅读了，你甚至可能无法理解发生了什么。我们创建了一个以文件对象为输入的对象，并提供了一个像任何迭代器一样的“__next__”方法。

这个“__next__”方法从文件中读取行，如果不是“WARNING”行，则将其丢弃。当我们遇到“WARNING”行时，我们修改并返回它。然后我们的“for”循环再次调用“__next__”来处理后续的“WARNING”行。当我们用完行时，我们引发“StopIteration”来告诉循环我们已经完成了迭代。与其他示例相比，这相当丑陋，但也很强大；现在我们手头有一个类，我们可以随心所欲地使用它。

有了这样的背景，我们终于可以看到真正的生成器在起作用了。下一个示例*完全*与前一个示例相同：它创建了一个具有“__next__”方法的对象，当输入用完时会引发“StopIteration”：

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

好吧，那可能相当容易阅读...至少很简短。但这到底是怎么回事？这根本毫无意义。而且“yield”到底是什么？

实际上，“yield”是生成器的关键。当 Python 在函数中看到“yield”时，它会将该函数包装在一个对象中，类似于我们之前示例中的对象。将“yield”语句视为类似于“return”语句；它退出函数并返回一行。但与“return”不同的是，当函数再次被调用（通过“next()”）时，它将从上次离开的地方开始——在“yield”语句之后的行——而不是从函数的开头开始。

在这个示例中，“yield”语句之后没有行，因此它会跳到“for”循环的下一个迭代。由于“yield”语句位于“if”语句内，它只会产生包含“WARNING”的行。

虽然看起来这只是一个循环遍历行的函数，但实际上它创建了一种特殊类型的对象，即生成器对象：

```py
>>> print(warnings_filter([]))
<generator object warnings_filter at 0xb728c6bc>  
```

我将一个空列表传递给函数，充当迭代器。函数所做的就是创建并返回一个生成器对象。该对象上有“__iter__”和“__next__”方法，就像我们在前面的示例中创建的那样。（你可以调用内置的“dir”函数来确认。）每当调用“__next__”时，生成器运行函数，直到找到“yield”语句。然后它返回“yield”的值，下一次调用“__next__”时，它会从上次离开的地方继续。

这种生成器的使用并不那么高级，但如果你没有意识到函数正在创建一个对象，它可能看起来像魔术一样。这个示例非常简单，但通过在单个函数中多次调用“yield”，你可以获得非常强大的效果；在每次循环中，生成器将简单地从最近的“yield”处继续到下一个“yield”处。

# 从另一个可迭代对象中产生值

通常，当我们构建一个生成器函数时，我们会陷入一种情况，我们希望从另一个可迭代对象中产生数据，可能是我们在生成器内部构造的列表推导或生成器表达式，或者可能是一些传递到函数中的外部项目。以前可以通过循环遍历可迭代对象并逐个产生每个项目来实现。然而，在 Python 3.3 版本中，Python 开发人员引入了一种新的语法，使其更加优雅一些。

让我们稍微调整一下生成器的例子，使其不再接受一系列行，而是接受一个文件名。这通常会被视为不好的做法，因为它将对象与特定的范例联系在一起。如果可能的话，我们应该在输入上操作迭代器；这样，同一个函数可以在日志行来自文件、内存或网络的情况下使用。

这个代码版本说明了你的生成器可以在从另一个可迭代对象（在本例中是一个生成器表达式）产生信息之前做一些基本的设置：

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

这段代码将前面示例中的`for`循环合并为一个生成器表达式。请注意，这种转换并没有帮助任何事情；前面的示例中使用`for`循环更易读。

因此，让我们考虑一个比其替代方案更易读的例子。构建一个生成器，从多个其他生成器中产生数据可能是有用的。例如，`itertools.chain`函数按顺序从可迭代对象中产生数据，直到它们全部耗尽。这可以使用`yield from`语法非常容易地实现，因此让我们考虑一个经典的计算机科学问题：遍历一棵通用树。

通用树数据结构的一个常见实现是计算机的文件系统。让我们模拟 Unix 文件系统中的一些文件夹和文件，这样我们就可以有效地使用`yield from`来遍历它们：

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

这个设置代码看起来很费力，但在一个真实的文件系统中，它会更加复杂。我们需要从硬盘读取数据并将其结构化成树。然而，一旦在内存中，输出文件系统中的每个文件的代码就非常优雅：

```py
def walk(file):
    if isinstance(file, Folder):
        yield file.name + "/"
        for f in file.children:
 yield from walk(f)
    else:
        yield file.name
```

如果这段代码遇到一个目录，它会递归地要求`walk()`生成每个子目录下所有文件的列表，然后产生所有这些数据以及它自己的文件名。在它遇到一个普通文件的简单情况下，它只会产生那个文件名。

顺便说一句，解决前面的问题而不使用生成器是相当棘手的，以至于它是一个常见的面试问题。如果你像这样回答，准备好让你的面试官既印象深刻又有些恼火，因为你回答得如此轻松。他们可能会要求你解释到底发生了什么。当然，凭借你在本章学到的原则，你不会有任何问题。祝你好运！

`yield from`语法在编写链式生成器时是一个有用的快捷方式。它被添加到语言中是出于不同的原因，以支持协程。然而，它现在并没有被那么多地使用，因为它的用法已经被`async`和`await`语法所取代。我们将在下一节看到两者的例子。

# 协程

协程是非常强大的构造，经常被误解为生成器。许多作者不恰当地将协程描述为*带有一些额外语法的生成器*。这是一个容易犯的错误，因为在 Python 2.5 中引入协程时，它们被介绍为*我们在生成器语法中添加了一个* `send` *方法*。实际上，区别要更微妙一些，在看到一些例子之后会更有意义。

协程是相当难以理解的。在`asyncio`模块之外，它们在野外并不经常使用。你绝对可以跳过这一部分，快乐地在 Python 中开发多年，而不必遇到协程。有一些库广泛使用协程（主要用于并发或异步编程），但它们通常是这样编写的，以便你可以使用协程而不必真正理解它们是如何工作的！所以，如果你在这一部分迷失了方向，不要绝望。

如果我还没有吓到你，让我们开始吧！这是一个最简单的协程之一；它允许我们保持一个可以通过任意值增加的累加值：

```py
def tally(): 
    score = 0 
    while True: 
 increment = yield score 
        score += increment 
```

这段代码看起来像是不可能工作的黑魔法，所以在逐行描述之前，让我们证明它可以工作。这个简单的对象可以被棒球队的记分应用程序使用。可以为每个队伍分别保留计分，并且他们的得分可以在每个半局结束时累加的得分增加。看看这个交互式会话：

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

首先，我们构建了两个`tally`对象，一个用于每个队伍。是的，它们看起来像函数，但与上一节中的生成器对象一样，函数内部有`yield`语句告诉 Python 要付出很大的努力将简单的函数转换为对象。

然后我们对每个协程对象调用`next()`。这与调用任何生成器的`next()`做的事情是一样的，也就是说，它执行每一行代码，直到遇到`yield`语句，返回该点的值，然后*暂停*，直到下一个`next()`调用。

到目前为止，没有什么新鲜的。但是回顾一下我们协程中的`yield`语句：

```py
increment = yield score 
```

与生成器不同，这个`yield`函数看起来像是要返回一个值并将其赋给一个变量。事实上，这正是发生的事情。协程仍然在`yield`语句处暂停，等待被另一个`next()`调用再次激活。

除了我们不调用`next()`。正如你在交互式会话中看到的，我们调用一个名为`send()`的方法。`send()`方法和`next()`做*完全*相同的事情，只是除了将生成器推进到下一个`yield`语句之外，它还允许你从生成器外部传入一个值。这个值被分配给`yield`语句的左侧。

对于许多人来说，真正令人困惑的是这发生的顺序：

1.  `yield`发生，生成器暂停

1.  `send()`发生在函数外部，生成器被唤醒

1.  传入的值被分配给`yield`语句的左侧

1.  生成器继续处理，直到遇到另一个`yield`语句

因此，在这个特定的例子中，我们构建了协程并通过单次调用`next()`将其推进到`yield`语句，然后每次调用`send()`都将一个值传递给协程。我们将这个值加到它的得分上。然后我们回到`while`循环的顶部，并继续处理，直到我们遇到`yield`语句。`yield`语句返回一个值，这个值成为我们最近一次调用`send`的返回值。不要错过这一点：像`next()`一样，`send()`方法不仅提交一个值给生成器，还返回即将到来的`yield`语句的值。这就是我们定义生成器和协程之间的区别的方式：生成器只产生值，而协程也可以消耗值。

`next(i)`、`i.__next__()`和`i.send(value)`的行为和语法相当不直观和令人沮丧。第一个是普通函数，第二个是特殊方法，最后一个是普通方法。但这三个都是做同样的事情：推进生成器直到它产生一个值并暂停。此外，`next()`函数和相关的方法可以通过调用`i.send(None)`来复制。在这里有两个不同的方法名是有价值的，因为它有助于我们的代码读者轻松地看到他们是在与协程还是生成器交互。我只是觉得在某些情况下它是一个函数调用，而在另一种情况下它是一个普通方法有点令人恼火。

# 回到日志解析

当然，前面的例子可以很容易地使用一对整数变量编码，并在它们上调用`x += increment`。让我们看一个第二个例子，其中协程实际上节省了我们一些代码。这个例子是我在 Facebook 工作时不得不解决的问题的一个简化版本（出于教学目的）。

Linux 内核日志包含几乎看起来与此类似但又不完全相同的行：

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

有一大堆交错的内核日志消息，其中一些与硬盘有关。硬盘消息可能与其他消息交错，但它们以可预测的格式和顺序出现。对于每个硬盘，已知的序列号与总线标识符（如`0:0:0:0`）相关联。块设备标识符（如`sda`）也与该总线相关联。最后，如果驱动器的文件系统损坏，它可能会出现 XFS 错误。

现在，考虑到前面的日志文件，我们需要解决的问题是如何获取任何出现 XFS 错误的驱动器的序列号。这个序列号可能稍后会被数据中心的技术人员用来识别并更换驱动器。

我们知道我们可以使用正则表达式识别单独的行，但是我们将不得不在循环遍历行时更改正则表达式，因为我们将根据先前找到的内容寻找不同的东西。另一个困难的地方是，如果我们找到一个错误字符串，包含该字符串的总线以及序列号的信息已经被处理过。这可以通过以相反的顺序迭代文件的行来轻松解决。

在查看这个例子之前，请注意——基于协程的解决方案所需的代码量非常少：

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

这段代码将工作分成了两个独立的任务。第一个任务是循环遍历所有行并输出与给定正则表达式匹配的任何行。第二个任务是与第一个任务交互，并为其提供指导，告诉它在任何给定时间应该搜索什么正则表达式。

首先看`match_regex`协程。记住，它在构造时不执行任何代码；相反，它只创建一个协程对象。一旦构造完成，协程外部的某人最终会调用`next()`来启动代码运行。然后它存储两个变量`filename`和`regex`的状态。然后它读取文件中的所有行并以相反的顺序对它们进行迭代。将传入的每一行与正则表达式进行比较，直到找到匹配项。当找到匹配项时，协程会产生正则表达式的第一个组并等待。

在将来的某个时候，其他代码将发送一个新的正则表达式来搜索。请注意，协程从不关心它试图匹配的正则表达式是什么；它只是循环遍历行并将它们与正则表达式进行比较。决定提供什么正则表达式是别人的责任。

在这种情况下，其他人是`get_serials`生成器。它不关心文件中的行；事实上，它甚至不知道它们。它做的第一件事是从`match_regex`协程构造函数创建一个`matcher`对象，给它一个默认的正则表达式来搜索。它将协程推进到它的第一个`yield`并存储它返回的值。然后它进入一个循环，指示`matcher`对象基于存储的设备 ID 搜索总线 ID，然后基于该总线 ID 搜索序列号。

它在向外部`for`循环空闲地产生该序列号之前指示匹配器找到另一个设备 ID 并重复循环。

基本上，协程的工作是在文件中搜索下一个重要的行，而生成器（`get_serial`，它使用`yield`语法而不进行赋值）的工作是决定哪一行是重要的。生成器有关于这个特定问题的信息，比如文件中行的顺序。

另一方面，协程可以插入到需要搜索文件以获取给定正则表达式的任何问题中。

# 关闭协程和引发异常

普通的生成器通过引发`StopIteration`来信号它们的退出。如果我们将多个生成器链接在一起（例如，通过在另一个生成器内部迭代一个生成器），`StopIteration`异常将向外传播。最终，它将遇到一个`for`循环，看到异常并知道是时候退出循环了。

尽管它们使用类似的语法，协程通常不遵循迭代机制。通常不是通过一个直到遇到异常的数据，而是通常将数据推送到其中（使用`send`）。通常是负责推送的实体告诉协程何时完成。它通过在相关协程上调用`close()`方法来做到这一点。

当调用`close()`方法时，它将在协程等待发送值的点引发`GeneratorExit`异常。通常，协程应该将它们的`yield`语句包装在`try`...`finally`块中，以便执行任何清理任务（例如关闭关联文件或套接字）。

如果我们需要在协程内部引发异常，我们可以类似地使用`throw()`方法。它接受一个异常类型，可选的`value`和`traceback`参数。当我们在一个协程中遇到异常并希望在相邻的协程中引发异常时，后者是有用的，同时保持回溯。

前面的例子可以在没有协程的情况下编写，并且读起来几乎一样。事实上，正确地管理协程之间的所有状态是相当困难的，特别是当你考虑到上下文管理器和异常等因素时。幸运的是，Python 标准库包含一个名为`asyncio`的包，可以为您管理所有这些。一般来说，我建议您避免使用裸协程，除非您专门为 asyncio 编写代码。日志示例几乎可以被认为是一种*反模式*；一种应该避免而不是拥抱的设计模式。

# 协程、生成器和函数之间的关系

我们已经看到了协程的运行，现在让我们回到讨论它们与生成器的关系。在 Python 中，就像经常发生的情况一样，这种区别是相当模糊的。事实上，所有的协程都是生成器对象，作者经常交替使用这两个术语。有时，他们将协程描述为生成器的一个子集（只有从`yield`返回值的生成器被认为是协程）。这在 Python 中是技术上正确的，正如我们在前面的部分中看到的。

然而，在更广泛的理论计算机科学领域，协程被认为是更一般的原则，生成器是协程的一种特定类型。此外，普通函数是协程的另一个独特子集。

协程是一个可以在一个或多个点传入数据并在一个或多个点获取数据的例程。在 Python 中，数据传入和传出的点是`yield`语句。

函数，或子例程，是协程的最简单类型。您可以在一个点传入数据，并在函数返回时在另一个点获取数据。虽然函数可以有多个`return`语句，但对于任何给定的函数调用，只能调用其中一个。

最后，生成器是一种可以在一个点传入数据的协程，但可以在多个点传出数据的协程。在 Python 中，数据将在`yield`语句处传出，但无法再传入数据。如果调用`send`，数据将被悄悄丢弃。

因此，理论上，生成器是协程的一种类型，函数是协程的一种类型，还有一些既不是函数也不是生成器的协程。够简单了吧？那么，为什么在 Python 中感觉更复杂呢？

在 Python 中，生成器和协程都是使用类似于构造函数的语法构造的。但是生成的对象根本不是函数；它是一种完全不同类型的对象。函数当然也是对象。但它们有不同的接口；函数是可调用的并返回值，生成器使用`next()`提取数据，协程使用`send`推入数据。

还有一种使用`async`和`await`关键字的协程的替代语法。这种语法使得代码更清晰，表明代码是一个协程，并进一步打破了协程和生成器之间的欺骗性对称性。

# 案例研究

Python 目前最流行的领域之一是数据科学。为了纪念这一事实，让我们实现一个基本的机器学习算法。

机器学习是一个庞大的主题，但总体思想是利用从过去数据中获得的知识对未来数据进行预测或分类。这些算法的用途层出不穷，数据科学家每天都在找到应用机器学习的新方法。一些重要的机器学习应用包括计算机视觉（如图像分类或人脸识别）、产品推荐、识别垃圾邮件和自动驾驶汽车。

为了不偏离整本关于机器学习的书，我们将看一个更简单的问题：给定一个 RGB 颜色定义，人们会将该颜色定义为什么名字？

标准 RGB 颜色空间中有超过 1600 万种颜色，人类只为其中的一小部分取了名字。虽然有成千上万种名称（有些相当荒谬；只需去任何汽车经销商或油漆商店），让我们构建一个试图将 RGB 空间划分为基本颜色的分类器：

+   红色

+   紫色

+   蓝色

+   绿色

+   黄色

+   橙色

+   灰色

+   粉色

（在我的测试中，我将白色和黑色的颜色分类为灰色，棕色的颜色分类为橙色。）

我们需要的第一件事是一个数据集来训练我们的算法。在生产系统中，您可能会从*颜色列表*网站上获取数据，或者对成千上万的人进行调查。相反，我创建了一个简单的应用程序，它会呈现一个随机颜色，并要求用户从前面的八个选项中选择一个来分类。我使用了 Python 附带的用户界面工具包`tkinter`来实现它。我不打算详细介绍这个脚本的内容，但为了完整起见，这是它的全部内容（它有点长，所以您可能想从 Packt 的 GitHub 存储库中获取本书示例的完整内容，而不是自己输入）：

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

如果您愿意，可以轻松添加更多按钮以获取其他颜色。您可能会在布局上遇到问题；`create_color_button`的第二个和第三个参数表示按钮所在的两列网格的行和列。一旦您将所有颜色放在位，您将希望将**退出**按钮移动到最后一行。

对于这个案例研究，了解这个应用程序的重要事情是输出。它创建了一个名为`colors.csv`的**逗号分隔值**（**CSV**）文件。该文件包含两个 CSV：用户为颜色分配的标签和颜色的十六进制 RGB 值。以下是一个示例：

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

在我厌倦并决定开始对我的数据集进行机器学习之前，我制作了 250 多个数据点。如果您想使用它，我的数据点已经与本章的示例一起提供（没有人告诉我我是色盲，所以它应该是合理的）。

我们将实现一种更简单的机器学习算法，称为*k 最近邻*。该算法依赖于数据集中点之间的某种*距离*计算（在我们的情况下，我们可以使用三维版本的毕达哥拉斯定理）。给定一个新的数据点，它找到一定数量（称为*k*，这是*k 最近邻*中的*k*）的数据点，这些数据点在通过该距离计算进行测量时最接近它。然后以某种方式组合这些数据点（对于线性计算，平均值可能有效；对于我们的分类问题，我们将使用模式），并返回结果。

我们不会详细介绍算法的工作原理；相反，我们将专注于如何将迭代器模式或迭代器协议应用于这个问题。

现在让我们编写一个程序，按顺序执行以下步骤：

1.  从文件中加载示例数据并构建模型。

1.  生成 100 种随机颜色。

1.  对每种颜色进行分类，并以与输入相同的格式输出到文件。

第一步是一个相当简单的生成器，它加载 CSV 数据并将其转换为符合我们需求的格式：

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

我们以前没有见过`csv.reader`函数。它返回文件中行的迭代器。迭代器返回的每个值都是一个由逗号分隔的字符串列表。因此，行`Green,#6edd13`返回为`["Green", "#6edd13"]`。

然后`load_colors`生成器逐行消耗该迭代器，并产生 RGB 值的元组以及标签。这种方式将生成器链接在一起是非常常见的，其中一个迭代器调用另一个迭代器，依此类推。您可能希望查看 Python 标准库中的`itertools`模块，其中有许多等待您的现成生成器。

在这种情况下，RGB 值是 0 到 255 之间的整数元组。从十六进制到 RGB 的转换有点棘手，因此我们将其提取到一个单独的函数中：

```py
def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) for i in range(1, 6, 2))

```

这个生成器表达式正在做很多工作。它以`“＃12abfe”`这样的字符串作为输入，并返回一个类似`(18, 171, 254)`的元组。让我们从后往前分解。

`range`调用将返回数字`[1, 3, 5]`。这些数字代表十六进制字符串中三个颜色通道的索引。索引`0`被跳过，因为它代表字符`“＃”`，而我们不关心这个字符。对于这三个数字中的每一个，它提取`i`和`i+2`之间的两个字符的字符串。对于前面的示例字符串，这将是`12`，`ab`和`fe`。然后将此字符串值转换为整数。作为`int`函数的第二个参数传递的`16`告诉函数使用基数 16（十六进制）而不是通常的基数 10（十进制）进行转换。

考虑到生成器表达式的阅读难度，您认为它应该以不同的格式表示吗？例如，它可以被创建为多个生成器表达式的序列，或者展开为一个带有`yield`语句的普通生成器函数。您更喜欢哪种？

在这种情况下，我相信函数名称能够解释这行丑陋代码在做什么。

现在我们已经加载了*训练数据*（手动分类的颜色），我们需要一些新数据来测试算法的工作效果。我们可以通过生成一百种随机颜色来实现这一点，每种颜色由 0 到 255 之间的三个随机数字组成。

有很多方法可以做到这一点：

+   一个带有嵌套生成器表达式的列表推导：``[tuple(randint(0,255) for c in range(3)) for r in range(100)]``

+   一个基本的生成器函数

+   实现`__iter__`和`__next__`协议的类

+   通过一系列协同程序将数据传递

+   即使只是一个基本的`for`循环

生成器版本似乎最易读，所以让我们将该函数添加到我们的程序中：

```py
from random import randint

def generate_colors(count=100):
    for i in range(count):
        yield (randint(0, 255), randint(0, 255), randint(0, 255))
```

注意我们如何对要生成的颜色数量进行参数化。现在我们可以在将来重用这个函数来执行其他生成颜色的任务。

现在，在进行分类之前，我们需要一个计算两种颜色之间*距离*的函数。由于可以将颜色看作是三维的（例如，红色、绿色和蓝色可以映射到*x*、*y*和*z*轴），让我们使用一些基本的数学：

```py
def color_distance(color1, color2):
    channels = zip(color1, color2)
    sum_distance_squared = 0
    for c1, c2 in channels:
        sum_distance_squared += (c1 - c2) ** 2
    return sum_distance_squared
```

这是一个看起来非常基本的函数；它看起来甚至没有使用迭代器协议。没有`yield`函数，也没有推导。但是，有一个`for`循环，`zip`函数的调用也在进行一些真正的迭代（如果您不熟悉它，`zip`会产生元组，每个元组包含来自每个输入迭代器的一个元素）。

这个距离计算是你可能从学校记得的勾股定理的三维版本：*a² + b² = c²*。由于我们使用了三个维度，我猜实际上应该是*a² + b² + c² = d²*。距离在技术上是*a² + b² + c²*的平方根，但没有必要执行相对昂贵的`sqrt`计算，因为平方距离在大小上都是相同的。

现在我们已经有了一些基本的管道，让我们来实现实际的 k-nearest neighbor。这个例程可以被认为是消耗和组合我们已经看到的两个生成器（`load_colors`和`generate_colors`）：

```py
def nearest_neighbors(model_colors, target_colors, num_neighbors=5):
    model_colors = list(model_colors)

    for target in target_colors:
        distances = sorted(
            ((color_distance(c[0], target), c) for c in model_colors)
        )
        yield target, distances[:5]
```

首先，我们将`model_colors`生成器转换为列表，因为它必须被多次使用，每次用于`target_colors`中的一个。如果我们不这样做，就必须重复从源文件加载颜色，这将执行大量不必要的磁盘读取。

这种决定的缺点是整个列表必须一次性全部存储在内存中。如果我们有一个无法放入内存的大型数据集，实际上需要每次从磁盘重新加载生成器（尽管在这种情况下，我们实际上会考虑不同的机器学习算法）。

`nearest_neighbors`生成器循环遍历每个目标颜色（例如`(255, 14, 168)`的三元组），并在生成器表达式中调用`color_distance`函数。然后，`sorted`调用对该生成器表达式的结果按其第一个元素进行排序，即距离。这是一段复杂的代码，一点也不面向对象。您可能需要将其分解为普通的`for`循环，以确保您理解生成器表达式在做什么。

`yield`语句稍微复杂一些。对于`target_colors`生成器中的每个 RGB 三元组，它产生目标和`num_neighbors`（这是*k*在*k-nearest*中，顺便说一下，许多数学家和数据科学家倾向于使用难以理解的单字母变量名）最接近的颜色的列表推导。

列表推导中的每个元素的内容是`model_colors`生成器的一个元素；也就是说，一个包含三个 RGB 值和手动输入的字符串名称的元组。因此，一个元素可能看起来像这样：`((104, 195, 77), 'Green')`。当我看到嵌套元组时，我首先想到的是，*这不是正确的数据结构*。RGB 颜色可能应该表示为一个命名元组，并且这两个属性可能应该放在一个数据类上。 

我们现在可以添加*另一个*生成器到链中，以找出我们应该给这个目标颜色起什么名字：

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

这个生成器将`nearest_neighbors`返回的元组解包成三元组目标和五个最近的数据点。它使用`Counter`来找到在返回的颜色中最常出现的名称。在`Counter`构造函数中还有另一个生成器表达式；这个生成器表达式从每个数据点中提取第二个元素（颜色名称）。然后它产生一个 RGB 值和猜测的名称的元组。返回值的一个例子是`(91, 158, 250) Blue`。

我们可以编写一个函数，接受`name_colors`生成器的输出，并将其写入 CSV 文件，RGB 颜色表示为十六进制值：

```py
def write_results(colors, filename="output.csv"):
    with open(filename, "w") as file:
        writer = csv.writer(file)
        for (r, g, b), name in colors:
            writer.writerow([name, f"#{r:02x}{g:02x}{b:02x}"])
```

这是一个函数，而不是一个生成器。它在`for`循环中消耗生成器，但它不产生任何东西。它构造了一个 CSV 写入器，并为每个目标颜色输出名称、十六进制值（例如`Purple,#7f5f95`）对的行。这里可能会让人困惑的唯一一件事是格式字符串的内容。与每个`r`、`g`和`b`通道一起使用的`:02x`修饰符将数字输出为前导零填充的两位十六进制数。

现在我们所要做的就是将这些不同的生成器和管道连接在一起，并通过一个函数调用启动整个过程：

```py
def process_colors(dataset_filename="colors.csv"):
    model_colors = load_colors(dataset_filename)
    colors = name_colors(model_colors, generate_colors(), 5)
    write_results(colors)

if __name__ == "__main__":
    process_colors()
```

因此，这个函数与我们定义的几乎所有其他函数不同，它是一个完全正常的函数，没有`yield`语句或`for`循环。它根本不进行任何迭代。

然而，它构造了三个生成器。你能看到所有三个吗？：

+   `load_colors`返回一个生成器

+   `generate_colors`返回一个生成器

+   `name_guess`返回一个生成器

`name_guess`生成器消耗了前两个生成器。然后，它又被`write_results`函数消耗。

我写了第二个 Tkinter 应用程序来检查算法的准确性。它与第一个应用程序类似，只是它会渲染每种颜色及与该颜色相关联的标签。然后你必须手动点击是或否，以确定标签是否与颜色匹配。对于我的示例数据，我得到了大约 95%的准确性。通过实施以下内容，这个准确性可以得到提高：

+   添加更多颜色名称

+   通过手动分类更多颜色来添加更多的训练数据

+   调整`num_neighbors`的值

+   使用更高级的机器学习算法

这是输出检查应用的代码，不过我建议下载示例代码。这样打字会很麻烦：

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

你可能会想，*这与面向对象编程有什么关系？这段代码中甚至没有一个类！* 从某些方面来说，你是对的；生成器通常不被认为是面向对象的。然而，创建它们的函数返回对象；实际上，你可以把这些函数看作构造函数。构造的对象有一个适当的`__next__()`方法。基本上，生成器语法是一种特定类型的对象的语法快捷方式，如果没有它，创建这种对象会非常冗长。

# 练习

如果你在日常编码中很少使用推导，那么你应该做的第一件事是搜索一些现有的代码，找到一些`for`循环。看看它们中是否有任何可以轻松转换为生成器表达式或列表、集合或字典推导的。

测试列表推导是否比`for`循环更快。这可以通过内置的`timeit`模块来完成。使用`timeit.timeit`函数的帮助文档找出如何使用它。基本上，编写两个做同样事情的函数，一个使用列表推导，一个使用`for`循环来迭代数千个项目。将每个函数传入`timeit.timeit`，并比较结果。如果你感到有冒险精神，也可以比较生成器和生成器表达式。使用`timeit`测试代码可能会让人上瘾，所以请记住，除非代码被执行了大量次数，比如在一个巨大的输入列表或文件上，否则代码不需要非常快。

玩转生成器函数。从需要多个值的基本迭代器开始（数学序列是典型的例子；如果你想不出更好的例子，斐波那契数列已经被过度使用了）。尝试一些更高级的生成器，比如接受多个输入列表并以某种方式产生合并值的生成器。生成器也可以用在文件上；你能否编写一个简单的生成器，显示两个文件中相同的行？

协程滥用迭代器协议，但实际上并不符合迭代器模式。你能否构建一个非协程版本的代码，从日志文件中获取序列号？采用面向对象的方法，以便在类上存储额外的状态。如果你能创建一个对象，它可以完全替代现有的协程，你将学到很多关于协程的知识。

本章的案例研究中有很多奇怪的元组传递，很难跟踪。看看是否可以用更面向对象的解决方案替换这些返回值。另外，尝试将一些共享数据的函数（例如`model_colors`和`target_colors`）移入一个类中进行实验。这样可以减少大多数生成器需要传入的参数数量，因为它们可以在`self`上查找。

# 总结

在本章中，我们了解到设计模式是有用的抽象，为常见的编程问题提供最佳实践解决方案。我们介绍了我们的第一个设计模式，迭代器，以及 Python 使用和滥用这种模式的多种方式。原始的迭代器模式非常面向对象，但在代码上也相当丑陋和冗长。然而，Python 的内置语法将丑陋抽象化，为我们留下了这些面向对象构造的清晰接口。

理解推导和生成器表达式可以将容器构造与迭代结合在一行中。生成器对象可以使用`yield`语法构造。协程在外部看起来像生成器，但用途完全不同。

我们将在接下来的两章中介绍几种设计模式。
