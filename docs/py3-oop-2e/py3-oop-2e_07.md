# 第七章：Python 面向对象的快捷方式

Python 的许多方面看起来更像是结构化或函数式编程，而不是面向对象编程。尽管面向对象编程在过去的二十年中是最可见的范例，但旧模型最近又出现了。与 Python 的数据结构一样，这些工具大多是在基础面向对象实现之上的语法糖；我们可以将它们看作是在（已经抽象化的）面向对象范例之上构建的进一步抽象层。在本章中，我们将涵盖一些不严格面向对象的 Python 特性。

+   处理常见任务的内置函数

+   文件 I/O 和上下文管理器

+   方法重载的替代方法

+   函数作为对象

# Python 内置函数

Python 中有许多函数，可以在某些类型的对象上执行任务或计算结果，而不是作为基础类的方法。它们通常抽象出适用于多种类型的类的常见计算。这是鸭子类型的最佳体现；这些函数接受具有某些属性或方法的对象，并能够使用这些方法执行通用操作。其中许多，但并非全部，都是特殊的双下划线方法。我们已经使用了许多内置函数，但让我们快速浏览一下重要的函数，并学习一些巧妙的技巧。

## len()函数

最简单的例子是`len()`函数，它计算某种容器对象中的项目数量，例如字典或列表。你以前见过它：

```py
>>> len([1,2,3,4])
4

```

为什么这些对象没有长度属性，而是必须对它们调用一个函数？从技术上讲，它们有。大多数`len()`将适用于的对象都有一个名为`__len__()`的方法，返回相同的值。因此，`len(myobj)`似乎调用了`myobj.__len__()`。

为什么我们应该使用`len()`函数而不是`__len__`方法？显然，`__len__`是一个特殊的双下划线方法，这表明我们不应该直接调用它。这一定有一个解释。Python 开发人员不会轻易做出这样的设计决定。

主要原因是效率。当我们在对象上调用`__len__`时，对象必须在其命名空间中查找该方法，并且如果该对象上定义了特殊的`__getattribute__`方法（每次访问对象的属性或方法时都会调用），还必须调用该方法。此外，该特定方法的`__getattribute__`可能已经被编写为执行一些不好的操作，比如拒绝让我们访问`__len__`之类的特殊方法！`len()`函数不会遇到任何这些问题。它实际上调用了基础类上的`__len__`函数，因此`len(myobj)`映射到`MyObj.__len__(myobj)`。

另一个原因是可维护性。将来，Python 开发人员可能希望更改`len()`，以便它可以计算没有`__len__`的对象的长度，例如，通过计算迭代器返回的项目数量。他们只需要更改一个函数，而不是在整个系统中无数次更改`__len__`方法。

`len()`作为外部函数的另一个极其重要且经常被忽视的原因是向后兼容性。这在文章中经常被引用为“出于历史原因”，这是作者用来表示某事之所以是某种方式的一个轻蔑的短语，因为很久以前犯了一个错误，我们现在被困在这种情况下。严格来说，`len()`并不是一个错误，而是一个设计决策，但这个决策是在一个不那么面向对象的时代做出的。它经受住了时间的考验，并且有一些好处，所以要习惯它。

## 反转

`reversed()`函数接受任何序列作为输入，并返回该序列的一个副本，顺序相反。通常在`for`循环中使用，当我们想要从后到前循环遍历项目时。

与 `len` 类似，`reversed` 调用参数类的 `__reversed__()` 函数。如果该方法不存在，`reversed` 将使用对 `__len__` 和 `__getitem__` 的调用构建反转序列，这些方法用于定义序列。如果我们想要自定义或优化过程，只需要重写 `__reversed__`：

```py
normal_list=[1,2,3,4,5]

class CustomSequence():
 **def __len__(self):
        return 5

 **def __getitem__(self, index):
        return "x{0}".format(index)

class FunkyBackwards():

 **def __reversed__(self):
        return "BACKWARDS!"

for seq in normal_list, CustomSequence(), FunkyBackwards():
    print("\n{}: ".format(seq.__class__.__name__), end="")
    for item in reversed(seq):
        print(item, end=", ")
```

最后的 `for` 循环打印了正常列表的反转版本，以及两个自定义序列的实例。输出显示 `reversed` 在所有三个上都起作用，但当我们自己定义 `__reversed__` 时结果大不相同：

```py
list: 5, 4, 3, 2, 1,
CustomSequence: x4, x3, x2, x1, x0,
FunkyBackwards: B, A, C, K, W, A, R, D, S, !,

```

当我们反转 `CustomSequence` 时，`__getitem__` 方法将为每个项目调用，它只是在索引之前插入一个 `x`。对于 `FunkyBackwards`，`__reversed__` 方法返回一个字符串，其中的每个字符在 `for` 循环中单独输出。

### 注意

前面两个类不是很好的序列，因为它们没有定义 `__iter__` 的正确版本，所以对它们进行正向 `for` 循环将永远不会结束。

## 枚举

有时，当我们在 `for` 循环中循环容器时，我们希望访问索引（列表中的当前位置）的当前项目。`for` 循环不提供索引，但 `enumerate` 函数给了我们更好的东西：它创建了一个元组序列，其中每个元组中的第一个对象是索引，第二个是原始项目。

如果我们需要直接使用索引号，这很有用。考虑一些简单的代码，输出文件中的每一行及其行号：

```py
import sys
filename = sys.argv[1]

with open(filename) as file:
 **for index, line in enumerate(file):
        print("{0}: {1}".format(index+1, line), end='')
```

使用自己的文件名作为输入文件运行此代码可以显示它的工作原理：

```py
1: import sys
2: filename = sys.argv[1]
3:
4: with open(filename) as file:
5:     for index, line in enumerate(file):
6:         print("{0}: {1}".format(index+1, line), end='')
```

`enumerate` 函数返回一个元组序列，我们的 `for` 循环将每个元组拆分为两个值，`print` 语句将它们格式化在一起。它为每行号添加一个索引，因为 `enumerate`，像所有序列一样，是从零开始的。

我们只触及了一些更重要的 Python 内置函数。正如你所看到的，其中许多函数调用了面向对象的概念，而其他一些则遵循纯粹的函数式或过程式范式。标准库中还有许多其他函数；其中一些更有趣的包括：

+   `all` 和 `any`，它们接受一个可迭代对象，并在所有或任何项目评估为真时返回 `True`（例如非空字符串或列表，非零数，不是 `None` 的对象，或字面值 `True`）。

+   `eval`，`exec` 和 `compile`，它们将字符串作为代码在解释器中执行。对于这些要小心；它们不安全，所以不要执行未知用户提供给你的代码（一般来说，假设所有未知用户都是恶意的、愚蠢的或两者兼有）。

+   `hasattr`，`getattr`，`setattr` 和 `delattr`，它们允许通过它们的字符串名称操作对象的属性。

+   `zip`，它接受两个或更多序列，并返回一个新的元组序列，其中每个元组包含来自每个序列的单个值。

+   还有更多！请参阅解释器帮助文档，了解 `dir(__builtins__)` 中列出的每个函数。

## 文件 I/O

到目前为止，我们所涉及的与文件系统有关的示例完全是在文本文件上进行的，没有太多考虑到底层发生了什么。然而，操作系统实际上将文件表示为字节序列，而不是文本。我们将深入探讨字节和文本之间的关系，第八章 *字符串和序列化*。现在，请注意，从文件中读取文本数据是一个相当复杂的过程。Python，特别是 Python 3，在幕后为我们处理了大部分工作。我们是不是很幸运？

文件的概念早在有人创造面向对象编程这个术语之前就已经存在。然而，Python 封装了操作系统提供的接口，提供了一个良好的抽象，使我们能够使用文件（或类似文件，即鸭子类型）对象。

`open()`内置函数用于打开文件并返回文件对象。要从文件中读取文本，我们只需要将文件名传递给函数。文件将被打开以进行读取，并且字节将使用平台默认编码转换为文本。

当然，我们并不总是想要读取文件；通常我们想要向其中写入数据！要打开一个文件进行写入，我们需要将`mode`参数作为第二个位置参数传递，值为`"w"`：

```py
contents = "Some file contents"
file = open("filename", "w")
file.write(contents)
file.close()
```

我们还可以将值`"a"`作为模式参数传递，以追加到文件末尾，而不是完全覆盖现有文件内容。

这些具有内置包装器的文件，用于将字节转换为文本，非常好，但如果我们想要打开的文件是图像、可执行文件或其他二进制文件，那将非常不方便，不是吗？

要打开二进制文件，我们修改模式字符串以附加`'b'`。因此，`'wb'`将打开一个用于写入字节的文件，而`'rb'`允许我们读取它们。它们的行为类似于文本文件，但没有将文本自动编码为字节。当我们读取这样的文件时，它将返回`bytes`对象而不是`str`，当我们向其写入时，如果尝试传递文本对象，它将失败。

### 注意

这些模式字符串用于控制文件的打开方式，相当晦涩，既不符合 Python 的风格，也不是面向对象的。然而，它们与几乎所有其他编程语言保持一致。文件 I/O 是操作系统必须处理的基本工作之一，所有编程语言都必须使用相同的系统调用与操作系统进行通信。幸运的是，Python 返回一个带有有用方法的文件对象，而不是大多数主要操作系统用于标识文件句柄的整数！

一旦文件被打开以进行读取，我们可以调用`read`、`readline`或`readlines`方法来获取文件的内容。`read`方法将整个文件的内容作为`str`或`bytes`对象返回，具体取决于模式中是否有`'b'`。在大文件上不要在没有参数的情况下使用此方法。您不希望尝试将如此多的数据加载到内存中！

还可以从文件中读取固定数量的字节；我们将整数参数传递给`read`方法，描述我们想要读取多少字节。对`read`的下一次调用将加载下一个字节序列，依此类推。我们可以在`while`循环中执行此操作，以管理的方式读取整个文件。

`readline`方法从文件中返回一行（每行以换行符、回车符或两者结尾，具体取决于创建文件的操作系统）。我们可以重复调用它以获取其他行。复数`readlines`方法返回文件中所有行的列表。与`read`方法一样，它不适用于非常大的文件。这两种方法甚至在文件以`bytes`模式打开时也适用，但只有在解析具有合理位置换行符的类文本数据时才有意义。例如，图像或音频文件中不会有换行符（除非换行符字节恰好表示某个像素或声音），因此应用`readline`是没有意义的。

为了可读性，并且避免一次性将大文件读入内存，通常最好直接在文件对象上使用`for`循环。对于文本文件，它将一次读取一行，我们可以在循环体内对其进行处理。对于二进制文件，最好使用`read()`方法读取固定大小的数据块，传递一个参数来指定要读取的最大字节数。

写入文件就像写入文件一样简单；文件对象上的`write`方法将一个字符串（或字节，用于二进制数据）对象写入文件。可以重复调用它来写入多个字符串，一个接一个。`writelines`方法接受一个字符串序列，并将迭代的每个值写入文件。`writelines`方法*不*在序列中的每个项目后附加新行。它基本上是一个命名不当的便利函数，用于写入字符串序列的内容，而无需使用`for`循环显式迭代。

最后，我是指最后，我们来到`close`方法。当我们完成读取或写入文件时，应调用此方法，以确保任何缓冲写入都写入磁盘，文件已经得到适当清理，并且与文件关联的所有资源都释放回操作系统。从技术上讲，当脚本退出时，这将自动发生，但最好是明确地清理自己的东西，特别是在长时间运行的进程中。

## 将其放入上下文中

当我们完成文件时需要关闭文件，这可能会使我们的代码变得相当丑陋。因为在文件 I/O 期间可能随时发生异常，我们应该将对文件的所有调用包装在`try`...`finally`子句中。无论 I/O 是否成功，文件都应在`finally`子句中关闭。这不是很 Pythonic。当然，有一种更优雅的方法来做到这一点。

如果我们在类似文件的对象上运行`dir`，我们会看到它有两个名为`__enter__`和`__exit__`的特殊方法。这些方法将文件对象转换为所谓的**上下文管理器**。基本上，如果我们使用一种称为`with`语句的特殊语法，这些方法将在嵌套代码执行之前和之后被调用。对于文件对象，`__exit__`方法确保文件被关闭，即使引发异常。我们不再需要显式管理文件的关闭。这是`with`语句在实践中的样子：

```py
with open('filename') as file:
    for line in file:
        print(line, end='')
```

`open`调用返回一个文件对象，该对象具有`__enter__`和`__exit__`方法。返回的对象由`as`子句分配给名为`file`的变量。我们知道当代码返回到外部缩进级别时，文件将被关闭，并且即使引发异常，也会发生这种情况。

`with`语句在标准库中的几个地方使用，需要执行启动或清理代码。例如，`urlopen`调用返回一个对象，可以在`with`语句中使用以在完成时清理套接字。线程模块中的锁可以在语句执行后自动释放锁。

最有趣的是，因为`with`语句可以应用于具有适当特殊方法的任何对象，我们可以在自己的框架中使用它。例如，记住字符串是不可变的，但有时您需要从多个部分构建一个字符串。出于效率考虑，通常通过将组件字符串存储在列表中并在最后将它们连接来完成。让我们创建一个简单的上下文管理器，允许我们构建一个字符序列，并在退出时自动将其转换为字符串：

```py
class StringJoiner(list):
    def __enter__(self):
        return self

    **def __exit__(self, type, value, tb):
        self.result = "".join(self)
```

这段代码将两个特殊方法添加到`list`类中，这两个方法是上下文管理器所需的。`__enter__`方法执行任何必需的设置代码（在本例中没有），然后返回将分配给`with`语句中`as`后面的变量的对象。通常情况下，就像我们在这里做的一样，这只是上下文管理器对象本身。`__exit__`方法接受三个参数。在正常情况下，这些参数都被赋予`None`的值。然而，如果在`with`块内发生异常，它们将被设置为与异常类型、值和回溯相关的值。这允许`__exit__`方法执行任何可能需要的清理代码，即使发生异常。在我们的例子中，我们采取了不负责任的路径，并通过连接字符串中的字符来创建一个结果字符串，而不管是否抛出异常。

虽然这是我们可以编写的最简单的上下文管理器之一，而且它的实用性是可疑的，但它确实可以与`with`语句一起使用。看看它的运行情况：

```py
import random, string
with StringJoiner() as joiner:
    for i in range(15):
        joiner.append(random.choice(string.ascii_letters))

print(joiner.result)
```

这段代码构造了一个包含 15 个随机字符的字符串。它使用从`list`继承的`append`方法将这些字符附加到`StringJoiner`上。当`with`语句超出范围（回到外部缩进级别）时，将调用`__exit__`方法，并且连接器对象上的`result`属性变得可用。我们打印这个值来看一个随机字符串。

# 方法重载的替代方法

许多面向对象的编程语言的一个显著特点是一种称为**方法重载**的工具。方法重载简单地指的是具有相同名称但接受不同参数集的多个方法。在静态类型的语言中，如果我们想要一个方法既接受整数又接受字符串，这是很有用的。在非面向对象的语言中，我们可能需要两个函数，称为`add_s`和`add_i`，来适应这种情况。在静态类型的面向对象语言中，我们需要两个方法，都称为`add`，一个接受字符串，一个接受整数。

在 Python 中，我们只需要一个方法，它接受任何类型的对象。它可能需要对对象类型进行一些测试（例如，如果它是一个字符串，将其转换为整数），但只需要一个方法。

然而，方法重载在我们希望一个方法具有相同名称但接受不同数量或一组不同参数时也很有用。例如，电子邮件消息方法可能有两个版本，其中一个接受“from”电子邮件地址的参数。另一个方法可能会查找默认的“from”电子邮件地址。Python 不允许具有相同名称的多个方法，但它提供了一个不同但同样灵活的接口。

我们已经看到了在之前的例子中发送参数给方法和函数的一些可能方式，但现在我们将涵盖所有细节。最简单的函数不接受任何参数。我们可能不需要一个例子，但为了完整起见，这里有一个：

```py
def no_args():
    pass
```

调用方式如下：

```py
no_args()
```

接受参数的函数将在逗号分隔的列表中提供这些参数的名称。只需要提供每个参数的名称。

在调用函数时，这些位置参数必须按顺序指定，不能遗漏或跳过任何一个。这是我们在之前的例子中指定参数的最常见方式：

```py
def mandatory_args(x, y, z):
    pass
```

调用它：

```py
mandatory_args("a string", a_variable, 5)
```

任何类型的对象都可以作为参数传递：对象、容器、原始类型，甚至函数和类。前面的调用显示了一个硬编码的字符串、一个未知的变量和一个整数传递给函数。

## 默认参数

如果我们想要使参数可选，而不是创建具有不同参数集的第二个方法，我们可以在单个方法中指定默认值，使用等号。如果调用代码没有提供此参数，它将被分配一个默认值。但是，调用代码仍然可以选择通过传入不同的值来覆盖默认值。通常，`None`、空字符串或空列表是合适的默认值。

以下是带有默认参数的函数定义：

```py
def default_arguments(x, y, z, a="Some String", b=False):
    pass
```

前三个参数仍然是必需的，并且必须由调用代码传递。最后两个参数有默认参数。

有几种方法可以调用这个函数。我们可以按顺序提供所有参数，就好像所有参数都是位置参数一样：

```py
default_arguments("a string", variable, 8, "", True)
```

或者，我们可以按顺序提供必需的参数，将关键字参数分配它们的默认值：

```py
default_arguments("a longer string", some_variable, 14)
```

我们还可以在调用函数时使用等号语法提供不同顺序的值，或者跳过我们不感兴趣的默认值。例如，我们可以跳过第一个关键字参数并提供第二个参数：

```py
default_arguments("a string", variable, 14, b=True)
```

令人惊讶的是，我们甚至可以使用等号语法来改变位置参数的顺序，只要所有参数都被提供：

```py
>>> default_arguments(y=1,z=2,x=3,a="hi")
3 1 2 hi False

```

有这么多选项，可能很难选择一个，但是如果你把位置参数看作一个有序列表，把关键字参数看作一种字典，你会发现正确的布局往往会自然而然地出现。如果需要要求调用者指定参数，就把它设为必需的；如果有一个合理的默认值，那就把它设为关键字参数。选择如何调用方法通常会自行解决，取决于需要提供哪些值，哪些可以保持默认值。

关键字参数需要注意的一点是，我们提供的默认参数在函数首次解释时进行评估，而不是在调用时。这意味着我们不能有动态生成的默认值。例如，以下代码的行为不会完全符合预期：

```py
number = 5
def funky_function(number=number):
    print(number)

number=6
funky_function(8)
funky_function()
print(number)
```

如果我们运行这段代码，首先输出数字 8，但接着对于没有参数的调用输出数字 5。我们已经将变量设置为数字 6，正如输出的最后一行所证明的那样，但当函数被调用时，打印出数字 5；默认值是在函数定义时计算的，而不是在调用时计算的。

这在空容器（如列表、集合和字典）中有些棘手。例如，通常要求调用代码提供一个我们的函数将要操作的列表，但列表是可选的。我们希望将一个空列表作为默认参数。我们不能这样做；它将在代码首次构造时创建一个列表：

```py
>>> def hello(b=[]):
...     b.append('a')
...     print(b)
...
>>> hello()
['a']
>>> hello()
['a', 'a']

```

哎呀，这不是我们预期的结果！通常的解决方法是将默认值设为`None`，然后在方法内部使用习惯用法`iargument = argument if argument else []`。请注意！

## 可变参数列表

仅仅使用默认值并不能让我们获得方法重载的所有灵活优势。使 Python 真正灵活的是能够编写接受任意数量的位置或关键字参数的方法，而不需要显式命名它们。我们还可以将任意列表和字典传递给这样的函数。

例如，一个接受链接或链接列表并下载网页的函数可以使用这样的可变参数，或者**varargs**。我们可以接受任意数量的参数，其中每个参数都是不同的链接。我们通过在函数定义中指定`*`运算符来实现这一点：

```py
def get_pages(*links):
    for link in links:
        #download the link with urllib
        print(link)
```

`*links`参数表示“我将接受任意数量的参数，并将它们全部放入名为`links`的列表中”。如果我们只提供一个参数，它将是一个具有一个元素的列表；如果我们不提供参数，它将是一个空列表。因此，所有这些函数调用都是有效的：

```py
get_pages()
get_pages('http://www.archlinux.org')
get_pages('http://www.archlinux.org',
        'http://ccphillips.net/')
```

我们也可以接受任意关键字参数。这些参数以字典的形式传入函数。它们在函数声明中用两个星号（如`**kwargs`）指定。这个工具通常用于配置设置。以下类允许我们指定一组具有默认值的选项：

```py
class Options:
    default_options = {
            'port': 21,
            'host': 'localhost',
            'username': None,
            'password': None,
            'debug': False,
            }
    **def __init__(self, **kwargs):
        self.options = dict(Options.default_options)
        self.options.update(kwargs)

    def __getitem__(self, key):
        return self.options[key]
```

这个类中所有有趣的东西都发生在`__init__`方法中。我们在类级别有一个默认选项和值的字典。`__init__`方法的第一件事是复制这个字典。我们这样做是为了避免直接修改字典，以防我们实例化两组不同的选项。（请记住，类级别的变量在类的实例之间是共享的。）然后，`__init__`使用新字典上的`update`方法将任何非默认值更改为提供的关键字参数。`__getitem__`方法简单地允许我们使用新类使用索引语法。以下是演示该类工作的会话：

```py
>>> options = Options(username="dusty", password="drowssap",
 **debug=True)
>>> options['debug']
True
>>> options['port']
21
>>> options['username']
'dusty'

```

我们可以使用字典索引语法访问我们的选项实例，字典包括默认值和我们使用关键字参数设置的值。

关键字参数语法可能是危险的，因为它可能违反“明确胜于隐式”的规则。在前面的示例中，可能会将任意关键字参数传递给`Options`初始化程序，以表示默认字典中不存在的选项。这可能不是一件坏事，这取决于类的目的，但它使得使用该类的人很难发现哪些有效选项是可用的。它还使得很容易输入令人困惑的拼写错误（例如，“Debug”而不是“debug”），从而添加两个选项，而实际上只应该存在一个选项。

当我们需要接受要传递给第二个函数的任意参数时，关键字参数也非常有用，但我们不知道这些参数是什么。我们在第三章中看到了这一点，当我们构建多重继承的支持时。当然，我们可以在一个函数调用中结合可变参数和可变关键字参数语法，并且我们也可以使用正常的位置参数和默认参数。以下示例有些牵强，但演示了这四种类型的作用：

```py
import shutil
import os.path
def augmented_move(target_folder, *filenames,
        **verbose=False, **specific):
    '''Move all filenames into the target_folder, allowing
    specific treatment of certain files.'''

    def print_verbose(message, filename):
        '''print the message only if verbose is enabled'''
        if verbose:
            print(message.format(filename))

    for filename in filenames:
        target_path = os.path.join(target_folder, filename)
        if filename in specific:
            if specific[filename] == 'ignore':
                print_verbose("Ignoring {0}", filename)
            elif specific[filename] == 'copy':
                print_verbose("Copying {0}", filename)
                shutil.copyfile(filename, target_path)
        else:
            print_verbose("Moving {0}", filename)
            shutil.move(filename, target_path)
```

此示例将处理任意文件列表。第一个参数是目标文件夹，默认行为是将所有剩余的非关键字参数文件移动到该文件夹中。然后是一个仅限关键字参数，`verbose`，它告诉我们是否要打印有关每个处理的文件的信息。最后，我们可以提供一个包含要对特定文件名执行的操作的字典；默认行为是移动文件，但如果在关键字参数中指定了有效的字符串操作，它可以被忽略或复制。请注意函数参数的顺序；首先指定位置参数，然后是`*filenames`列表，然后是任何特定的仅限关键字参数，最后是一个`**specific`字典，用于保存剩余的关键字参数。

我们创建一个内部辅助函数`print_verbose`，只有在设置了`verbose`键时才会打印消息。这个函数通过将这个功能封装到一个单一位置来保持代码的可读性。

在常见情况下，假设所讨论的文件存在，可以调用此函数：

```py
>>> augmented_move("move_here", "one", "two")

```

这个命令将文件`one`和`two`移动到`move_here`目录中，假设它们存在（函数中没有错误检查或异常处理，因此如果文件或目标目录不存在，它将失败）。由于`verbose`默认为`False`，移动将在没有任何输出的情况下发生。

如果我们想要看到输出，我们可以这样调用它：

```py
>>> augmented_move("move_here", "three", verbose=True)
Moving three

```

这将移动名为`three`的一个文件，并告诉我们它在做什么。请注意，在这个例子中不可能将`verbose`指定为位置参数；我们必须传递关键字参数。否则，Python 会认为它是`*filenames`列表中的另一个文件名。

如果我们想要复制或忽略列表中的一些文件，而不是移动它们，我们可以传递额外的关键字参数：

```py
>>> augmented_move("move_here", "four", "five", "six",
 **four="copy", five="ignore")

```

这将移动第六个文件并复制第四个文件，但不会显示任何输出，因为我们没有指定`verbose`。当然，我们也可以这样做，关键字参数可以以任何顺序提供：

```py
>>> augmented_move("move_here", "seven", "eight", "nine",
 **seven="copy", verbose=True, eight="ignore")
Copying seven
Ignoring eight
Moving nine

```

## 解包参数

还有一个关于可变参数和关键字参数的巧妙技巧。我们在之前的一些示例中使用过它，但现在解释也不算晚。给定一个值的列表或字典，我们可以将这些值传递到函数中，就好像它们是普通的位置参数或关键字参数。看看这段代码：

```py
def show_args(arg1, arg2, arg3="THREE"):
    print(arg1, arg2, arg3)

some_args = range(3)
more_args = {
        "arg1": "ONE",
        "arg2": "TWO"}

print("Unpacking a sequence:", end=" ")

show_args(*some_args)
print("Unpacking a dict:", end=" ")

show_args(**more_args)

```

当我们运行它时，它看起来是这样的：

```py
Unpacking a sequence: 0 1 2
Unpacking a dict: ONE TWO THREE

```

该函数接受三个参数，其中一个具有默认值。但是当我们有一个包含三个参数的列表时，我们可以在函数调用中使用`*`运算符将其解包为三个参数。如果我们有一个参数字典，我们可以使用`**`语法将其解包为一组关键字参数。

当映射从用户输入或外部来源（例如互联网页面或文本文件）收集的信息到函数或方法调用时，这通常是最有用的。

还记得我们之前的例子吗？它使用文本文件中的标题和行来创建包含联系信息的字典列表？我们可以使用关键字解包将这些参数传递给专门构建的`Contact`对象上的`__init__`方法，该对象接受相同的参数集。看看你是否可以调整示例使其工作。

# 函数也是对象

过分强调面向对象原则的编程语言往往不赞成不是方法的函数。在这种语言中，你应该创建一个对象来包装所涉及的单个方法。有许多情况下，我们希望传递一个简单调用以执行操作的小对象。这在事件驱动编程中最常见，例如图形工具包或异步服务器；我们将在第十章和第十一章中看到一些使用它的设计模式，*Python 设计模式 I*和*Python 设计模式 II*。

在 Python 中，我们不需要将这些方法包装在对象中，因为函数本身就是对象！我们可以在函数上设置属性（尽管这不是常见的活动），并且我们可以传递它们以便在以后调用。它们甚至有一些可以直接访问的特殊属性。这里是另一个刻意的例子：

```py
def my_function():
    print("The Function Was Called")
my_function.description = "A silly function"

def second_function():
    print("The second was called")
second_function.description = "A sillier function."

def another_function(function):
    print("The description:", end=" ")
    print(function.description)
    print("The name:", end=" ")
    print(function.__name__)
    print("The class:", end=" ")
    print(function.__class__)
    print("Now I'll call the function passed in")
    function()

another_function(my_function)
another_function(second_function)

```

如果我们运行这段代码，我们可以看到我们能够将两个不同的函数传递到我们的第三个函数中，并获得不同的输出：

```py
The description: A silly function
The name: my_function
The class: <class 'function'>
Now I'll call the function passed in
The Function Was Called
The description: A sillier function.
The name: second_function
The class: <class 'function'>
Now I'll call the function passed in
The second was called
```

我们在函数上设置了一个名为`description`的属性（诚然，这些描述并不是很好）。我们还能够看到函数的`__name__`属性，并访问它的类，证明了函数确实是一个具有属性的对象。然后我们使用可调用语法（括号）调用了该函数。

函数作为顶级对象的一个最常见的用途是将它们传递以便在以后的某个日期执行，例如当满足某个条件时。让我们构建一个事件驱动的定时器，它正是这样做的：

```py
import datetime
import time

class TimedEvent:
    **def __init__(self, endtime, callback):
        self.endtime = endtime
        self.callback = callback

    def ready(self):
        return self.endtime <= datetime.datetime.now()

class Timer:
    def __init__(self):
        self.events = []

    **def call_after(self, delay, callback):
        end_time = datetime.datetime.now() + \
                datetime.timedelta(seconds=delay)

        self.events.append(TimedEvent(end_time, callback))

    def run(self):
        while True:
            ready_events = (e for e in self.events if e.ready())
            for event in ready_events:
                **event.callback(self)
                self.events.remove(event)
            time.sleep(0.5)
```

在生产中，这段代码应该有额外的文档，使用文档字符串！`call_after`方法至少应该提到`delay`参数是以秒为单位的，并且`callback`函数应该接受一个参数：调用的定时器。

这里有两个类。`TimedEvent`类实际上并不是其他类可以访问的；它只是存储`endtime`和`callback`。我们甚至可以在这里使用`tuple`或`namedtuple`，但是为了方便起见，给对象赋予一个告诉我们事件是否准备运行的行为，我们使用了一个类。

`Timer`类只是存储了一个即将到来的事件列表。它有一个`call_after`方法来添加一个新的事件。这个方法接受一个`delay`参数，表示在执行回调之前等待的秒数，以及`callback`函数本身：在正确的时间执行的函数。这个`callback`函数应该接受一个参数。

`run`方法非常简单；它使用生成器表达式来过滤出任何时间到来的事件，并按顺序执行它们。然后定时器循环无限继续，因此必须使用键盘中断（*Ctrl* + *C*或*Ctrl* + *Break*）来中断它。在每次迭代后，我们休眠半秒，以免使系统陷入停顿。

这里需要注意的重要事情是涉及回调函数的那些行。函数像任何其他对象一样被传递，定时器从不知道或关心函数的原始名称是什么，或者它是在哪里定义的。当该函数被调用时，定时器只是将圆括号语法应用于存储的变量。

这是一组测试定时器的回调函数：

```py
from timer import Timer
import datetime

def format_time(message, *args):
    now = datetime.datetime.now().strftime("%I:%M:%S")
    print(message.format(*args, now=now))

def one(timer):
    format_time("{now}: Called One")

def two(timer):
    format_time("{now}: Called Two")

def three(timer):
    format_time("{now}: Called Three")

class Repeater:
    def __init__(self):
        self.count = 0
    def repeater(self, timer):
        format_time("{now}: repeat {0}", self.count)
        self.count += 1
        timer.call_after(5, self.repeater)

timer = Timer()
timer.call_after(1, one)
timer.call_after(2, one)
timer.call_after(2, two)
timer.call_after(4, two)
timer.call_after(3, three)
timer.call_after(6, three)
repeater = Repeater()
timer.call_after(5, repeater.repeater)
format_time("{now}: Starting")
timer.run()
```

这个例子让我们看到多个回调函数如何与定时器交互。第一个函数是`format_time`函数。它使用字符串的`format`方法将当前时间添加到消息中，并演示了可变参数的作用。`format_time`方法将接受任意数量的位置参数，使用可变参数语法，然后将它们作为位置参数转发给字符串的`format`方法。之后，我们创建了三个简单的回调方法，它们只是输出当前时间和一个简短的消息，告诉我们哪个回调已经被触发。

`Repeater`类演示了方法也可以用作回调，因为它们实际上只是函数。它还展示了回调函数中的`timer`参数的用处：我们可以在当前正在运行的回调中向定时器添加一个新的定时事件。然后，我们创建了一个定时器，并向其中添加了几个在不同时间后调用的事件。最后，我们启动了定时器；输出显示事件按预期顺序运行：

```py
02:53:35: Starting
02:53:36: Called One
02:53:37: Called One
02:53:37: Called Two
02:53:38: Called Three
02:53:39: Called Two
02:53:40: repeat 0
02:53:41: Called Three
02:53:45: repeat 1
02:53:50: repeat 2
02:53:55: repeat 3
02:54:00: repeat 4
```

Python 3.4 引入了类似于这样的通用事件循环架构。我们将在第十三章 *并发*中讨论它。

## 使用函数作为属性

函数作为对象的一个有趣的效果是，它们可以被设置为其他对象的可调用属性。可以向已实例化的对象添加或更改函数：

```py
class A:
    def print(self):
        print("my class is A")

def fake_print():
    print("my class is not A")

a = A()
a.print()
a.print = fake_print
a.print()
```

这段代码创建了一个非常简单的类，其中包含一个不告诉我们任何新信息的`print`方法。然后我们创建了一个新的函数，告诉了我们一些我们不相信的东西。

当我们在`A`类的实例上调用`print`时，它的行为与预期一样。如果我们将`print`方法设置为指向一个新函数，它会告诉我们一些不同的东西：

```py
my class is A
my class is not A
```

还可以替换类的方法而不是对象的方法，尽管在这种情况下，我们必须将`self`参数添加到参数列表中。这将更改该对象的所有实例的方法，即使已经实例化了这些实例。显然，这样替换方法既危险又令人困惑。阅读代码的人会看到已调用一个方法，并在原始类上查找该方法。但是在原始类上的方法并不是被调用的方法。弄清楚到底发生了什么可能会变成一个棘手的，令人沮丧的调试过程。

它确实有其用途。通常，在运行时替换或添加方法（称为**monkey-patching**）在自动化测试中使用。如果测试客户端-服务器应用程序，我们可能不希望在测试客户端时实际连接到服务器；这可能导致资金的意外转移或向真实人员发送尴尬的测试电子邮件。相反，我们可以设置我们的测试代码来替换发送请求到服务器的对象的一些关键方法，这样它只记录这些方法已被调用。

Monkey-patching 也可以用于修复第三方代码中的错误或添加我们正在交互的功能，并且不会以我们需要的方式行为。但是，它应该谨慎应用；它几乎总是一个“混乱的黑客”。不过，有时它是适应现有库以满足我们需求的唯一方法。

## 可调用对象

就像函数是可以在其上设置属性的对象一样，也可以创建一个可以像函数一样被调用的对象。

任何对象都可以通过简单地给它一个接受所需参数的`__call__`方法来使其可调用。让我们通过使其可调用来使我们的`Repeater`类，从计时器示例中，更容易使用：

```py
class Repeater:
    def __init__(self):
        self.count = 0

    **def __call__(self, timer):
        format_time("{now}: repeat {0}", self.count)
        self.count += 1

        timer.call_after(5, self)

timer = Timer()

timer.call_after(5, Repeater())
format_time("{now}: Starting")
timer.run()
```

这个示例与之前的类没有太大的不同；我们所做的只是将`repeater`函数的名称更改为`__call__`并将对象本身作为可调用传递。请注意，当我们进行`call_after`调用时，我们传递参数`Repeater()`。这两个括号创建了一个类的新实例；它们并没有显式调用类。这发生在稍后，在计时器内部。如果我们想要在新实例化的对象上执行`__call__`方法，我们将使用相当奇怪的语法：`Repeater()()`。第一组括号构造了对象；第二组执行了`__call__`方法。如果我们发现自己这样做，我们可能没有使用正确的抽象。只有在对象被视为函数时才实现`__call__`函数。

# 案例研究

为了将本章介绍的一些原则联系起来，让我们构建一个邮件列表管理器。该管理器将跟踪分类为命名组的电子邮件地址。当发送消息时，我们可以选择一个组，并将消息发送到分配给该组的所有电子邮件地址。

现在，在我们开始处理这个项目之前，我们应该有一种安全的方法来测试它，而不会向一群真实的人发送电子邮件。幸运的是，Python 在这方面有所帮助；就像测试 HTTP 服务器一样，它有一个内置的**简单邮件传输协议**（**SMTP**）服务器，我们可以指示它捕获我们发送的任何消息，而不实际发送它们。我们可以使用以下命令运行服务器：

```py
python -m smtpd -n -c DebuggingServer localhost:1025

```

在命令提示符下运行此命令将在本地机器上的端口 1025 上启动运行 SMTP 服务器。但是，我们已经指示它使用`DebuggingServer`类（它与内置的 SMTP 模块一起提供），它不会将邮件发送给预期的收件人，而是在接收到邮件时将其简单地打印在终端屏幕上。好吧，是不是很整洁？

现在，在编写邮件列表之前，让我们编写一些实际发送邮件的代码。当然，Python 也支持标准库中的邮件发送，但它的接口有点奇怪，所以我们将编写一个新的函数来清晰地包装它：

```py
import smtplib
from email.mime.text import MIMEText

def send_email(subject, message, from_addr, *to_addrs,
        host="localhost", port=1025, **headers):

    email = MIMEText(message)
    email['Subject'] = subject
    email['From'] = from_addr
    for header, value in headers.items():
        email[header] = value

    sender = smtplib.SMTP(host, port)
    for addr in to_addrs:
        del email['To']
        email['To'] = addr
        sender.sendmail(from_addr, addr, email.as_string())
    sender.quit()
```

我们不会过分深入地讨论此方法中的代码；标准库中的文档可以为您提供使用`smtplib`和`email`模块的所有信息。

我们在函数调用中使用了变量参数和关键字参数语法。变量参数列表允许我们在默认情况下提供单个`to`地址的字符串，并允许在需要时提供多个地址。任何额外的关键字参数都将映射到电子邮件标头。这是变量参数和关键字参数的一个令人兴奋的用法，但它实际上并不是对调用函数的人来说一个很好的接口。实际上，它使得程序员想要做的许多事情都变得不可能。

传递给函数的标头表示可以附加到方法的辅助标头。这些标头可能包括`Reply-To`、`Return-Path`或*X-pretty-much-anything*。但是为了成为 Python 中的有效标识符，名称不能包括`-`字符。一般来说，该字符表示减法。因此，不可能使用`Reply-To = my@email.com`来调用函数。看来我们太急于使用关键字参数，因为这是我们在本章中刚学到的新工具。

我们将需要将参数更改为普通字典；这将起作用，因为任何字符串都可以用作字典中的键。默认情况下，我们希望这个字典是空的，但我们不能将默认参数设置为空字典。因此，我们将默认参数设置为`None`，然后在方法的开头设置字典：

```py
def send_email(subject, message, from_addr, *to_addrs,
        host="localhost", port=1025, headers=None):

    headers = {} if headers is None else headers
```

如果我们在一个终端中运行我们的调试 SMTP 服务器，我们可以在 Python 解释器中测试这段代码：

```py
>>> send_email("A model subject", "The message contents",
 **"from@example.com", "to1@example.com", "to2@example.com")

```

然后，如果我们检查调试 SMTP 服务器的输出，我们会得到以下内容：

```py
---------- MESSAGE FOLLOWS ----------
Content-Type: text/plain; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Subject: A model subject
From: from@example.com
To: to1@example.com
X-Peer: 127.0.0.1

The message contents
------------ END MESSAGE ------------
---------- MESSAGE FOLLOWS ----------
Content-Type: text/plain; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Subject: A model subject
From: from@example.com
To: to2@example.com
X-Peer: 127.0.0.1

The message contents
------------ END MESSAGE ------------

```

很好，它已经将我们的电子邮件发送到了两个预期的地址，并包括主题和消息内容。现在我们可以发送消息了，让我们来处理电子邮件组管理系统。我们需要一个对象，以某种方式将电子邮件地址与它们所在的组匹配起来。由于这是一对多的关系（任何一个电子邮件地址可以在多个组中；任何一个组可以与多个电子邮件地址相关联），我们学习过的数据结构似乎都不太理想。我们可以尝试一个将组名与相关电子邮件地址列表匹配的字典，但这样会重复电子邮件地址。我们也可以尝试一个将电子邮件地址与组匹配的字典，这样会导致组的重复。两者都不太理想。让我们尝试后一种版本，尽管直觉告诉我，组到电子邮件地址的解决方案可能更直接。

由于我们字典中的值始终是唯一电子邮件地址的集合，我们可能应该将它们存储在`set`容器中。我们可以使用`defaultdict`来确保每个键始终有一个`set`容器可用：

```py
from collections import defaultdict
class MailingList:
    '''Manage groups of e-mail addresses for sending e-mails.'''
    def __init__(self):
        self.email_map = defaultdict(set)

    def add_to_group(self, email, group):
        self.email_map[email].add(group)
```

现在，让我们添加一个方法，允许我们收集一个或多个组中的所有电子邮件地址。这可以通过将组列表转换为一个集合来完成：

```py
    def emails_in_groups(self, *groups):
        groups = set(groups)
        emails = set()
        for e, g in self.email_map.items():
            if g & groups:
                emails.add(e)
        return emails
```

首先，看看我们正在迭代的内容：`self.email_map.items()`。这个方法当然会返回字典中每个项目的键-值对元组。值是表示组的字符串集合。我们将这些拆分成两个变量，命名为`e`和`g`，分别代表电子邮件和组。只有当传入的组与电子邮件地址组相交时，我们才将电子邮件地址添加到返回值集合中。`g & groups`语法是`g.intersection(groups)`的快捷方式；`set`类通过实现特殊的`__and__`方法来调用`intersection`。

### 提示

这段代码可以通过使用集合推导式变得更加简洁，我们将在第九章*迭代器模式*中讨论。

现在，有了这些基本组件，我们可以轻松地向我们的`MailingList`类添加一个发送消息到特定组的方法：

```py
def send_mailing(self, subject, message, from_addr,
        *groups, headers=None):
    emails = self.emails_in_groups(*groups)
    send_email(subject, message, from_addr,
            *emails, headers=headers)
```

这个函数依赖于可变参数列表。作为输入，它接受一个组列表作为可变参数。它获取指定组的电子邮件列表，并将它们作为可变参数传递到`send_email`中，以及传递给这个方法的其他参数。

可以通过确保 SMTP 调试服务器在一个命令提示符中运行，并在第二个提示符中使用以下代码来加载代码来测试程序：

```py
python -i mailing_list.py

```

使用以下命令创建一个`MailingList`对象：

```py
>>> m = MailingList()

```

然后创建一些虚假的电子邮件地址和组，类似于：

```py
>>> m.add_to_group("friend1@example.com", "friends")
>>> m.add_to_group("friend2@example.com", "friends")
>>> m.add_to_group("family1@example.com", "family")
>>> m.add_to_group("pro1@example.com", "professional")

```

最后，使用以下命令发送电子邮件到特定组：

```py
>>> m.send_mailing("A Party",
"Friends and family only: a party", "me@example.com", "friends",
"family", headers={"Reply-To": "me2@example.com"})

```

指定组中每个地址的电子邮件应该显示在 SMTP 服务器的控制台上。

邮件列表的工作正常，但它有点无用；一旦我们退出程序，我们的信息数据库就丢失了。让我们修改它，添加一些方法来从文件中加载和保存电子邮件组列表。

总的来说，当在磁盘上存储结构化数据时，很有必要认真考虑它的存储方式。存在众多数据库系统的原因之一是，如果其他人已经考虑过数据的存储方式，你就不必再考虑了。我们将在下一章中看一些数据序列化机制，但对于这个例子，让我们保持简单，选择可能有效的第一个解决方案。

我心目中的数据格式是存储每个电子邮件地址，后跟一个空格，再跟一个逗号分隔的组列表。这种格式似乎是合理的，我们将采用它，因为数据格式化不是本章的主题。然而，为了说明为什么你需要认真考虑如何在磁盘上格式化数据，让我们来强调一下这种格式的一些问题。

首先，空格字符在技术上是合法的电子邮件地址。大多数电子邮件提供商禁止它（有充分的理由），但定义电子邮件地址的规范说，如果电子邮件地址在引号中，它可以包含空格。如果我们要在我们的数据格式中使用空格作为标记，我们应该在这个空格和电子邮件中的空格之间进行区分。出于简单起见，我们将假装这不是真的，但现实生活中的数据编码充满了这样愚蠢的问题。其次，考虑逗号分隔的组列表。如果有人决定在组名中放一个逗号会发生什么？如果我们决定在组名中禁止逗号，我们应该添加验证以确保这一点到我们的`add_to_group`方法。为了教学上的清晰，我们也将忽略这个问题。最后，有许多安全问题需要考虑：有人是否可以通过在他们的电子邮件地址中放一个虚假的逗号来把自己放入错误的组？如果解析器遇到无效文件会怎么做？

从这次讨论中得出的要点是，尽量使用经过现场测试的数据存储方法，而不是设计自己的数据序列化协议。你可能会忽略很多奇怪的边缘情况，最好使用已经遇到并解决了这些边缘情况的代码。

但是忘了那个，让我们只是写一些基本的代码，使用大量的一厢情愿来假装这种简单的数据格式是安全的：

```py
email1@mydomain.com group1,group2
email2@mydomain.com group2,group3

```

执行此操作的代码如下：

```py
    def save(self):
        with open(self.data_file, 'w') as file:
            for email, groups in self.email_map.items():
                file.write(
                    '{} {}\n'.format(email, ','.join(groups))
                )

    def load(self):
        self.email_map = defaultdict(set)
        try:
            with open(self.data_file) as file:
                for line in file:
                    email, groups = line.strip().split(' ')
                    groups = set(groups.split(','))
                    self.email_map[email] = groups
        except IOError:
            pass
```

在`save`方法中，我们使用上下文管理器打开文件并将文件写入格式化字符串。记住换行符；Python 不会为我们添加它。`load`方法首先重置字典（以防它包含来自对`load`的先前调用的数据），使用`for`...`in`语法，循环遍历文件中的每一行。同样，换行符包含在行变量中，所以我们必须调用`.strip()`来去掉它。我们将在下一章中学习更多关于这种字符串操作的知识。

在使用这些方法之前，我们需要确保对象具有`self.data_file`属性，可以通过修改`__init__`来实现：

```py
    def __init__(self, data_file):
        self.data_file = data_file
        self.email_map = defaultdict(set)
```

我们可以在解释器中测试这两种方法，如下所示：

```py
>>> m = MailingList('addresses.db')
>>> m.add_to_group('friend1@example.com', 'friends')
>>> m.add_to_group('family1@example.com', 'friends')
>>> m.add_to_group('family1@example.com', 'family')
>>> m.save()

```

生成的`addresses.db`文件包含以下行，如预期的那样：

```py
friend1@example.com friends
family1@example.com friends,family

```

我们还可以成功地将这些数据加载回`MailingList`对象中：

```py
>>> m = MailingList('addresses.db')
>>> m.email_map
defaultdict(<class 'set'>, {})
>>> m.load()
>>> m.email_map
defaultdict(<class 'set'>, {'friend2@example.com': {'friends\n'}, 'family1@example.com': {'family\n'}, 'friend1@example.com': {'friends\n'}})

```

正如您所看到的，我忘记了`load`命令，也可能很容易忘记`save`命令。为了使任何想要在自己的代码中使用我们的`MailingList`API 的人更容易一些，让我们提供支持上下文管理器的方法：

```py
    def __enter__(self):
        self.load()
        return self

    def __exit__(self, type, value, tb):
        self.save()
```

这些简单的方法只是将它们的工作委托给 load 和 save，但现在我们可以在交互式解释器中编写这样的代码，并知道以前存储的所有地址都已经被加载，并且在我们完成时整个列表将保存到文件中：

```py
>>> with MailingList('addresses.db') as ml:
...    ml.add_to_group('friend2@example.com', 'friends')
...    ml.send_mailing("What's up", "hey friends, how's it going", 'me@example.com', 'friends')

```

# 练习

如果您以前没有遇到`with`语句和上下文管理器，我鼓励您像往常一样，浏览您的旧代码，并确保所有打开文件的地方都使用`with`语句安全关闭。还要寻找可以编写自己的上下文管理器的地方。丑陋或重复的`try`...`finally`子句是一个很好的起点，但您可能会发现它们在任何需要在上下文中执行之前和/或之后的任务时都很有用。

您可能已经在现在之前使用了许多基本的内置函数。我们涵盖了其中的一些，但没有详细讨论。尝试使用`enumerate`、`zip`、`reversed`、`any`和`all`，直到您知道在合适的情况下会记得使用它们。特别重要的是`enumerate`函数；因为不使用它会导致一些非常丑陋的代码。

还要探索一些将函数作为可调用对象传递的应用程序，以及使用`__call__`方法使您自己的对象可调用。您可以通过将属性附加到函数或在对象上创建`__call__`方法来获得相同的效果。在哪种情况下您会使用一种语法，何时更适合使用另一种语法？

如果有大量的电子邮件需要发送，我们的邮件列表对象可能会使电子邮件服务器不堪重负。尝试重构它，以便您可以为不同的目的使用不同的`send_email`函数。其中一个函数可以是我们在这里使用的版本。另一个版本可能会将电子邮件放入队列，由不同的线程或进程中的服务器发送。第三个版本可能只是将数据输出到终端，从而避免了需要虚拟的 SMTP 服务器。您能否构建带有回调的邮件列表，以便`send_mailing`函数使用传入的任何内容？如果没有提供回调，则默认使用当前版本。

参数、关键字参数、可变参数和可变关键字参数之间的关系可能有点令人困惑。我们看到它们在多重继承时是如何痛苦地相互作用的。设计一些其他示例，看看它们如何能够很好地协同工作，以及了解它们何时不适用。

# 总结

在本章中，我们涵盖了一系列杂项主题。每个主题都代表了 Python 中流行的重要非面向对象的特性。仅仅因为我们可以使用面向对象的原则，并不意味着我们总是应该这样做！

但是，我们也看到 Python 通常通过为传统面向对象的语法提供语法快捷方式来实现这些功能。了解这些工具背后的面向对象原则使我们能够更有效地在自己的类中使用它们。

我们讨论了一系列内置函数和文件 I/O 操作。在调用带有参数、关键字参数和可变参数列表时，我们有许多不同的语法可供选择。上下文管理器对于将一段代码夹在两个方法调用之间的常见模式非常有用。甚至函数也是对象，反之亦然，任何普通对象都可以被调用。

在下一章中，我们将学习更多关于字符串和文件操作的知识，甚至会花一些时间来学习标准库中最不面向对象的主题之一：正则表达式。
