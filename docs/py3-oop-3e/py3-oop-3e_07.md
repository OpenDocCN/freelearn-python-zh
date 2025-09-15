# 第七章：Python 面向对象快捷方式

Python 的许多方面看起来更像是结构化或函数式编程，而不是面向对象编程。尽管面向对象编程在过去二十年里是最明显的范式，但旧模型最近又有所复兴。与 Python 的数据结构一样，这些工具中的大多数都是在底层面向对象实现之上的语法糖；我们可以把它们看作是在（已经抽象化的）面向对象范式之上的进一步抽象层。在本章中，我们将介绍一些不是严格面向对象的 Python 特性：

+   内置函数，可以在一次调用中处理常见任务

+   文件 I/O 和上下文管理器

+   方法重载的替代方案

+   函数作为对象

# Python 内置函数

Python 中有许多函数在底层类上执行任务或计算结果，而不作为底层类的方法。它们通常抽象了适用于多个类类型的常见计算。这是鸭子类型的最典型表现；这些函数接受具有某些属性或方法的对象，并能够使用这些方法执行通用操作。我们已经使用了许多内置函数，但让我们快速浏览一下重要的函数，并在过程中学习一些技巧。

# `len()`函数

最简单的例子是`len()`函数，它计算某种容器对象中项目的数量，例如字典或列表。你之前已经见过它，如下所示：

```py
>>> len([1,2,3,4])
4  
```

你可能会想知道为什么这些对象没有长度属性，而必须调用它们上的函数。技术上，它们确实有。大多数`len()`将应用的对象都有一个名为`__len__()`的方法，它返回相同的值。所以`len(myobj)`看起来是调用`myobj.__len__()`。

为什么我们应该使用`len()`函数而不是`__len__`方法？显然，`__len__`是一个特殊的双下划线方法，暗示我们不应该直接调用它。这肯定有它的原因。Python 开发者不会轻易做出这样的设计决策。

主要原因是效率。当我们对一个对象调用`__len__`时，对象必须在其命名空间中查找该方法，并且如果该对象上定义了特殊的`__getattribute__`方法（每次访问对象上的属性或方法时都会调用该方法），它也必须被调用。此外，该特定方法的`__getattribute__`可能被编写为执行一些令人讨厌的操作，例如拒绝给我们访问特殊方法，如`__len__`！`len()`函数不会遇到这些问题。它实际上在底层类上调用`__len__`函数，所以`len(myobj)`映射到`MyObj.__len__(myobj)`。

另一个原因是可维护性。在未来，Python 开发者可能想要更改`len()`，使其能够计算没有`__len__`的对象的长度，例如，通过计算迭代器返回的项目数量。他们只需更改一个函数，而不是在许多对象中更改无数个`__len__`方法。

另一个非常重要的、经常被忽视的原因是`len()`是一个外部函数：向后兼容性。这通常在文章中引用为“出于历史原因”，这是一个轻微的轻蔑说法，作者会用来表示某事之所以如此，是因为很久以前犯了一个错误，我们不得不忍受它。严格来说，`len()`不是一个错误，而是一个设计决策，但这个决策是在一个不那么面向对象的时代做出的。它经受了时间的考验，并有一些好处，所以请习惯它。

# 逆序

`reversed()`函数接受任何序列作为输入，并返回该序列的逆序副本。它通常在`for`循环中使用，当我们想要从后向前遍历项目时。

与`len`类似，`reversed`在参数的类上调用`__reversed__()`函数。如果该方法不存在，`reversed`将使用定义序列时使用的`__len__`和`__getitem__`调用本身构建逆序序列。我们只需要覆盖`__reversed__`，如果我们想以某种方式自定义或优化这个过程，如下面的代码所示：

```py
normal_list = [1, 2, 3, 4, 5]

class CustomSequence:
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return f"x{index}"

class FunkyBackwards:
 def __reversed__(self):
 return "BACKWARDS!"

for seq in normal_list, CustomSequence(), FunkyBackwards():
    print(f"\n{seq.__class__.__name__}: ", end="")
    for item in reversed(seq):
        print(item, end=", ")
```

结尾的`for`循环打印了正常列表和两个自定义序列的逆序版本。输出显示`reversed`在这三个中都能工作，但当我们自己定义`__reversed__`时，结果非常不同：

```py
list: 5, 4, 3, 2, 1,
CustomSequence: x4, x3, x2, x1, x0,
FunkyBackwards: B, A, C, K, W, A, R, D, S, !,  
```

当我们逆序`CustomSequence`时，`__getitem__`方法会为每个项目被调用，它只是在索引前插入一个`x`。对于`FunkyBackwards`，`__reversed__`方法返回一个字符串，每个字符都在`for`循环中单独输出。

前两个类并不是很好的序列，因为它们没有定义一个合适的`__iter__`版本，所以对它们的正向`for`循环永远不会结束。

# 枚举

有时，当我们在一个`for`循环中遍历一个容器时，我们想要访问当前正在处理的项目索引（列表中的当前位置）。`for`循环不会为我们提供索引，但`enumerate`函数给了我们更好的东西：它创建了一个元组的序列，其中每个元组的第一个对象是索引，第二个是原始项目。

如果我们需要直接使用索引号，这很有用。考虑一些简单的代码，它输出文件中的每一行并带有行号：

```py
import sys

filename = sys.argv[1]

with open(filename) as file:
 for index, line in enumerate(file):
        print(f"{index+1}: {line}", end="")
```

使用其自己的文件名作为输入文件运行此代码，展示了它是如何工作的：

```py
1: import sys
2:
3: filename = sys.argv[1]
4:
5: with open(filename) as file:
6:     for index, line in enumerate(file):
7:         print(f"{index+1}: {line}", end="")
```

`enumerate` 函数返回一个元组序列，我们的 `for` 循环将每个元组拆分为两个值，`print` 语句将它们格式化在一起。它为每一行编号加一，因为 `enumerate`，像所有序列一样，是基于零的。

我们只触及了 Python 内置函数中的一些重要函数。正如你所见，许多函数都调用了面向对象的概念，而其他一些则遵循纯粹的功能或过程式范式。标准库中还有许多其他函数；其中一些有趣的函数包括以下内容：

+   `all` 和 `any`，这两个函数接受一个可迭代对象，如果所有或任何项评估为真（例如非空字符串或列表、非零数字、非 `None` 对象或字面量 `True`），则返回 `True`。

+   `eval`、`exec` 和 `compile`，这三个函数在解释器内部执行字符串作为代码。对这些函数要小心；它们不安全，因此不要执行未知用户提供的代码（通常，假设所有未知用户都是恶意的、愚蠢的或两者兼而有之）。

+   `hasattr`、`getattr`、`setattr` 和 `delattr`，这些函数允许通过对象的字符串名称来操作其属性。

+   `zip` 函数接受两个或更多序列，并返回一个新的元组序列，其中每个元组包含每个序列的单个值。

+   以及更多！请参阅 `dir(__builtins__)` 中列出的每个函数的解释器帮助文档。

# 文件输入/输出

我们迄今为止的示例已经涉及到文件系统，但几乎没有考虑底层发生了什么。然而，操作系统实际上将文件表示为字节序列，而不是文本。我们将在第八章深入探讨字节和文本之间的关系，*字符串和序列化*。现在，请注意，从文件中读取文本数据是一个相当复杂的过程。Python，尤其是 Python 3，在幕后为我们处理了大部分工作。我们不是太幸运吗？!

文件的概念早在有人提出“面向对象编程”这个术语之前就已经存在了。然而，Python 将操作系统提供的接口封装在一个甜美的抽象中，这使得我们可以与文件（或类似文件的，即鸭子类型）对象一起工作。

`open()` 内置函数用于打开文件并返回一个文件对象。为了从文件中读取文本，我们只需要将文件名传递给该函数。文件将以读取模式打开，并且将使用平台默认编码将字节转换为文本。

当然，我们并不总是想读取文件；通常我们希望向它们写入数据！为了写入文件，我们需要将一个 `mode` 参数作为第二个位置参数传递，其值为 `"w"`：

```py
contents = "Some file contents" 
file = open("filename", "w") 
file.write(contents) 
file.close() 
```

我们也可以将值 `"a"` 作为模式参数提供，以将内容追加到文件末尾，而不是完全覆盖现有文件内容。

这些内置了将字节转换为文本的包装器的文件很棒，但如果我们要打开的文件是一个图像、可执行文件或其他二进制文件，那就非常不方便了，不是吗？

要打开一个二进制文件，我们修改模式字符串以附加`'b'`。因此，`'wb'`将打开一个用于写入字节的文件，而`'rb'`允许我们读取它们。它们的行为类似于文本文件，但没有将文本自动编码为字节的操作。当我们读取这样的文件时，它将返回`bytes`对象，而当我们写入它时，如果我们尝试传递一个文本对象，它将失败。

用于控制文件打开方式的这些模式字符串相当晦涩，既不符合 Python 风格，也不是面向对象的。然而，它们与几乎所有其他编程语言都保持一致。文件 I/O 是操作系统必须处理的基本任务之一，所有编程语言都必须使用相同的系统调用来与操作系统通信。庆幸的是，Python 返回了一个带有有用方法的文件对象，而不是像大多数主要操作系统那样使用整数来标识文件句柄！

一旦文件被打开用于读取，我们可以调用`read`、`readline`或`readlines`方法来获取文件的内容。`read`方法返回整个文件的内容，作为一个`str`或`bytes`对象，这取决于模式中是否有`'b'`。小心不要在没有参数的情况下使用这个方法在大型文件上。你不想发现如果你尝试将这么多数据加载到内存中会发生什么！

从文件中读取固定数量的字节也是可能的；我们通过传递一个整数参数给`read`方法，来描述我们想要读取的字节数。下一次调用`read`方法将加载下一个字节序列，依此类推。我们可以在`while`循环中这样做，以分块读取整个文件。

`readline`方法返回文件中的一行（每行以换行符、回车符或两者都结束，具体取决于创建文件的操作系统）。我们可以反复调用它来获取额外的行。复数形式的`readlines`方法返回文件中所有行的列表。像`read`方法一样，在非常大的文件上使用它并不安全。这两个方法在文件以`bytes`模式打开时也能工作，但这只有在我们要解析具有合理位置的新行的类似文本数据时才有意义。例如，图像或音频文件中不会有换行符（除非换行字节恰好代表某个像素或声音），所以应用`readline`就没有意义。

为了提高可读性，并避免一次性将大文件读入内存，通常最好直接在文件对象上使用`for`循环。对于文本文件，它将逐行读取，我们可以在循环体内部处理它。对于二进制文件，最好使用`read()`方法以固定大小的数据块读取，传递一个参数来指定要读取的最大字节数。

向文件写入同样简单；文件对象的 `write` 方法将字符串（或字节，对于二进制数据）对象写入文件。它可以被重复调用以写入多个字符串，一个接一个。`writelines` 方法接受一个字符串序列，并将迭代值中的每个值写入文件。`writelines` 方法不会在序列中的每个项目后添加新行。它基本上是一个命名不佳的便利函数，用于在不显式使用 `for` 循环迭代的情况下写入字符串序列的内容。

最后，而且我确实是指最后，我们来到了 `close` 方法。当我们在完成读取或写入文件后应该调用此方法，以确保任何缓冲的写入都被写入磁盘，文件已经被适当清理，以及所有与文件关联的资源都被释放回操作系统。技术上，当脚本退出时这会自动发生，但最好是明确地清理，尤其是在长时间运行的过程中。

# 放入上下文中

当我们完成文件操作时需要关闭文件，这可能会使我们的代码变得相当丑陋。因为文件 I/O 期间可能随时发生异常，我们应该将所有对文件的调用包裹在 `try`...`finally` 子句中。文件应该在 `finally` 子句中关闭，无论 I/O 是否成功。这并不太符合 Python 风格。当然，还有更优雅的方式来处理。

如果我们在一个文件对象上运行 `dir`，我们会看到它有两个特殊的方法名为 `__enter__` 和 `__exit__`。这些方法将文件对象转换成所谓的 **上下文管理器**。基本上，如果我们使用一个特殊的语法，即 `with` 语句，这些方法将在嵌套代码执行前后被调用。在文件对象上，`__exit__` 方法确保即使在抛出异常的情况下文件也会被关闭。我们不再需要显式地管理文件的关闭。以下是在实践中 `with` 语句的样子：

```py
with open('filename') as file: 
    for line in file: 
        print(line, end='') 
```

`open` 调用返回一个文件对象，该对象具有 `__enter__` 和 `__exit__` 方法。通过 `as` 子句，返回的对象被分配给名为 `file` 的变量。我们知道当代码返回到外层缩进级别时文件将会被关闭，即使抛出了异常也是如此。

`with` 语句在标准库的多个地方被使用，当需要执行启动或清理代码时。例如，`urlopen` 调用返回一个对象，该对象可以在 `with` 语句中使用来清理套接字，当我们完成时。在 `threading` 模块中的锁可以自动释放锁，当语句被执行完毕。

最有趣的是，因为`with`语句可以应用于具有适当特殊方法的任何对象，我们可以在自己的框架中使用它。例如，记住字符串是不可变的，但有时你需要从多个部分构建一个字符串。为了效率，这通常是通过将组件字符串存储在列表中并在最后连接它们来完成的。让我们创建一个简单的上下文管理器，允许我们构建一个字符序列，并在退出时自动将其转换为字符串：

```py
class StringJoiner(list): 
 def __enter__(self): 
        return self 

 def __exit__(self, type, value, tb): 
        self.result = "".join(self) 
```

此代码将上下文管理器所需的两个特殊方法添加到它继承的`list`类中。`__enter__`方法执行任何所需的设置代码（在这种情况下，没有）然后返回将在`with`语句中的`as`之后分配给变量的对象。通常，就像我们在这里做的那样，这仅仅是上下文管理器对象本身。`__exit__`方法接受三个参数。在正常情况下，这些参数都被赋予`None`的值。然而，如果在`with`块内部发生异常，它们将被设置为与异常类型、值和回溯相关的值。这允许`__exit__`方法执行可能需要的任何清理代码，即使发生了异常。在我们的例子中，我们采取了不负责任的做法，通过连接字符串中的字符来创建一个结果字符串，无论是否抛出异常。

虽然这是我们能够编写的最简单的上下文管理器之一，但其实用性值得怀疑，但它确实可以与`with`语句一起工作。看看它是如何运作的：

```py
import random, string 
with StringJoiner() as joiner: 
    for i in range(15): 
        joiner.append(random.choice(string.ascii_letters)) 

print(joiner.result) 
```

此代码构建了一个由 15 个随机字符组成的字符串。它使用从`list`继承的`append`方法将这些字符追加到`StringJoiner`中。当`with`语句超出作用域（回到外层缩进级别）时，会调用`__exit__`方法，并且`result`属性在连接器对象上变得可用。然后我们打印这个值以查看一个随机字符串。

# 方法重载的替代方案

许多面向对象编程语言的一个显著特点是称为**方法重载**的工具。方法重载简单地说就是拥有多个具有相同名称的方法，这些方法接受不同的参数集。在静态类型语言中，如果我们想有一个接受整数或字符串的方法，例如，这很有用。在非面向对象语言中，我们可能需要两个函数，称为`add_s`和`add_i`，以适应这种情况。在静态类型面向对象语言中，我们需要两个方法，都称为`add`，一个接受字符串，另一个接受整数。

在 Python 中，我们已经看到我们只需要一个方法，该方法接受任何类型的对象。它可能需要对对象类型进行一些测试（例如，如果它是一个字符串，将其转换为整数），但只需要一个方法。

然而，当我们想要一个具有相同名称的方法接受不同数量或参数集时，方法重载也是有用的。例如，一个电子邮件消息方法可能有两种版本，其中一种接受一个用于*from*电子邮件地址的参数。另一种方法可能查找默认的*from*电子邮件地址。Python 不允许存在多个同名的方法，但它确实提供了一个不同但同样灵活的接口。

我们在之前的例子中已经看到了向方法和函数发送参数的一些可能方法，但现在我们将涵盖所有细节。最简单的函数不接受任何参数。我们可能不需要示例，但这里有一个为了完整性：

```py
def no_args(): 
    pass 
```

以及它是如何被调用的：

```py
no_args() 
```

接受参数的函数将提供一个逗号分隔的参数名称列表。只需要提供每个参数的名称。

当调用函数时，这些位置参数必须按顺序指定，不能遗漏或跳过。这是我们之前示例中最常见的指定参数的方式：

```py
def mandatory_args(x, y, z): 
    pass 
```

要调用它，请输入以下内容：

```py
mandatory_args("a string", a_variable, 5) 
```

任何类型的对象都可以作为参数传递：一个对象、一个容器、一个原始数据类型，甚至是函数和类。前面的调用显示了硬编码的字符串、一个未知变量和一个整数传递到函数中。

# 默认参数

如果我们想要使一个参数可选，而不是创建一个具有不同参数集的第二种方法，我们可以在一个方法中指定一个默认值，使用等号。如果调用代码没有提供这个参数，它将被分配一个默认值。然而，调用代码仍然可以选择通过传递不同的值来覆盖默认值。通常，`None`、空字符串或空列表的默认值是合适的。

这里有一个带有默认参数的函数定义：

```py
def default_arguments(x, y, z, a="Some String", b=False): 
    pass 
```

前三个参数仍然是强制性的，必须由调用代码提供。最后两个参数提供了默认参数。

我们可以以几种方式调用这个函数。我们可以按顺序提供所有参数，就像所有参数都是位置参数一样，如下所示：

```py
default_arguments("a string", variable, 8, "", True) 
```

或者，我们也可以按顺序提供强制参数，让关键字参数分配它们的默认值：

```py
default_arguments("a longer string", some_variable, 14) 
```

我们也可以在调用函数时使用等号语法来提供不同的顺序的值，或者跳过我们不感兴趣的默认值。例如，我们可以跳过第一个关键字参数并提供第二个参数：

```py
default_arguments("a string", variable, 14, b=True) 
```

惊讶的是，我们甚至可以使用等号语法来混合位置参数的顺序，只要所有参数都提供：

```py
>>> default_arguments(y=1,z=2,x=3,a="hi")
3 1 2 hi False  
```

你可能会偶尔发现，将一个*关键字仅*参数（即必须作为关键字参数提供的参数）是有用的。你可以通过在关键字仅参数之前放置一个`*`来实现这一点：

```py
def kw_only(x, y='defaultkw', *, a, b='only'):
    print(x, y, a, b)
```

这个函数有一个位置参数`x`和三个关键字参数`y`、`a`和`b`。`x`和`y`都是强制性的，但`a`只能作为关键字参数传递。`y`和`b`都是可选的，有默认值，但如果提供了`b`，它只能作为关键字参数。

如果你不传递`a`，这个函数会失败：

```py
>>> kw_only('x')
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
TypeError: kw_only() missing 1 required keyword-only argument: 'a'
```

如果你以位置参数传递`a`，它也会失败：

```py
>>> kw_only('x', 'y', 'a')
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
TypeError: kw_only() takes from 1 to 2 positional arguments but 3 were given
```

但你可以将`a`和`b`作为关键字参数传递：

```py
>>> kw_only('x', a='a', b='b')
x defaultkw a b
```

有这么多选项，可能看起来很难选择，但如果你将位置参数视为有序列表，将关键字参数视为类似字典的东西，你会发现正确的布局往往自然而然地就位。如果你需要要求调用者指定一个参数，就让它成为强制性的；如果你有一个合理的默认值，那么就将其作为关键字参数。如何调用方法通常取决于需要提供哪些值，哪些可以保留为默认值。仅关键字参数相对较少，但在用例出现时，它们可以使 API 更加优雅。

在关键字参数方面需要注意的一点是，我们提供的任何默认参数都是在函数首次解释时评估的，而不是在调用时。这意味着我们不能有动态生成的默认值。例如，以下代码的行为可能不会完全符合预期：

```py
number = 5 
def funky_function(number=number): 
    print(number) 

number=6 
funky_function(8) 
funky_function() 
print(number) 
```

如果我们运行这段代码，它首先输出数字`8`，然后对于不带参数的调用，它输出数字`5`。我们已将变量设置为数字`6`，如输出最后一行所示，但在函数调用时，打印的是数字`5`；默认值是在函数定义时计算的，而不是在调用时。

对于空容器，如列表、集合和字典，这很棘手。例如，通常要求调用代码提供一个列表，我们的函数将要对其进行操作，但这个列表是可选的。我们希望将一个空列表作为默认参数。我们不能这样做；当代码首次构建时，它只会创建一个列表，如下所示::

```py
//DON'T DO THIS
>>> def hello(b=[]):
...     b.append('a')
...     print(b)
...
>>> hello()
['a']
>>> hello()
['a', 'a']  
```

哎呀，这并不是我们预期的！通常绕过这个问题的方法是使默认值`None`，然后在方法内部使用`iargument = argument if argument else []`惯用表达式。请务必注意！

# 可变参数列表

仅默认值并不能给我们带来方法重载的所有灵活好处。使 Python 真正出色的一点是能够编写接受任意数量位置或关键字参数的方法，而无需显式命名它们。我们还可以将任意列表和字典传递到这样的函数中。

例如，一个接受链接或链接列表并下载网页的函数可以使用这样的可变参数，或称为 **varargs**。我们不需要接受一个预期为链接列表的单个值，而是可以接受任意数量的参数，其中每个参数都是不同的链接。我们通过在函数定义中指定 `*` 运算符来实现这一点，如下所示：

```py
def get_pages(*links): 
    for link in links: 
        #download the link with urllib 
        print(link) 
```

`*links` 参数表示，**“我会接受任意数量的参数并将它们全部放入一个名为* `links` 的列表中”。如果我们只提供一个参数，它将是一个只有一个元素的列表；如果我们不提供任何参数，它将是一个空列表。因此，所有这些函数调用都是有效的**：

```py
get_pages() 
get_pages('http://www.archlinux.org') 
get_pages('http://www.archlinux.org', 
        'http://ccphillips.net/') 
```

我们还可以接受任意的关键字参数。这些参数以字典的形式传递给函数。它们在函数声明中用两个星号（如 `**kwargs`）指定。这个工具在配置设置中常用。以下类允许我们指定一组具有默认值的选项：

```py
class Options: 
    default_options = { 
            'port': 21, 
            'host': 'localhost', 
            'username': None, 
            'password': None, 
            'debug': False, 
            } 
 def __init__(self, **kwargs): 
        self.options = dict(Options.default_options) 
        self.options.update(kwargs) 

    def __getitem__(self, key): 
        return self.options[key] 
```

这个类中所有有趣的事情都发生在 `__init__` 方法中。我们在类级别有一个默认选项和值的字典。`__init__` 方法首先做的事情是复制这个字典。我们这样做而不是直接修改字典，以防我们实例化两个不同的选项集。（记住，类级别的变量在类的实例之间是共享的。）然后，`__init__` 使用新字典上的 `update` 方法将任何非默认值更改为作为关键字参数提供的值。`__getitem__` 方法简单地允许我们使用新的类，并使用索引语法。以下是一个演示该类如何工作的会话：

```py
>>> options = Options(username="dusty", password="drowssap",
 debug=True)
>>> options['debug']
True
>>> options['port']
21
>>> options['username']
'dusty'  
```

我们能够使用字典索引语法访问我们的 `options` 实例，并且这个字典包括默认值和我们使用关键字参数设置的值。

关键字参数的语法可能很危险，因为它可能会破坏**“显式优于隐式”**的规则。在上面的例子中，我们可以向 `Options` 初始化器传递任意的关键字参数来表示默认字典中不存在的选项。这可能不是一件坏事，取决于类的用途，但它使得使用该类的人很难发现哪些有效的选项可用。这也使得容易输入令人困惑的拼写错误（例如，*Debug* 而不是 *debug*），这会在只有一个选项应该存在的地方添加两个选项。

当我们需要接受任意参数传递给第二个函数，但我们不知道这些参数将是什么时，关键字参数也非常有用。我们在第三章，*当对象相似时*，看到了这一点，当时我们在构建多重继承的支持。当然，我们可以在一个函数调用中结合可变参数和可变关键字参数的语法，并且我们还可以使用正常的定位参数和默认参数。以下例子有些牵强，但演示了四种类型在实际中的应用：

```py
import shutil
import os.path

def augmented_move(
    target_folder, *filenames, verbose=False, **specific
):
    """Move all filenames into the target_folder, allowing
    specific treatment of certain files."""

    def print_verbose(message, filename):
        """print the message only if verbose is enabled"""
        if verbose:
            print(message.format(filename))

    for filename in filenames:
        target_path = os.path.join(target_folder, filename)
        if filename in specific:
            if specific[filename] == "ignore":
                print_verbose("Ignoring {0}", filename)
            elif specific[filename] == "copy":
                print_verbose("Copying {0}", filename)
                shutil.copyfile(filename, target_path)
        else:
            print_verbose("Moving {0}", filename)
            shutil.move(filename, target_path)
```

此示例处理一个任意文件列表。第一个参数是目标文件夹，默认行为是将所有剩余的非关键字参数文件移动到该文件夹中。然后是一个仅关键字参数，`verbose`，它告诉我们是否要在每个处理的文件上打印信息。最后，我们可以提供一个包含对特定文件名执行操作的字典；默认行为是移动文件，但如果在关键字参数中指定了有效的字符串操作，则可以忽略或复制它。注意函数中参数的顺序；首先指定位置参数，然后是`*filenames`列表，然后是任何特定的仅关键字参数，最后是一个`**specific`字典来保存剩余的关键字参数。

我们创建了一个内部辅助函数`print_verbose`，该函数仅在设置了`verbose`键时打印消息。这个函数通过将此功能封装在单个位置来保持代码的可读性。

在常见情况下，假设相关的文件存在，此函数可以如下调用：

```py
>>> augmented_move("move_here", "one", "two")  
```

此命令会将文件`one`和`two`移动到`move_here`目录中，假设它们存在（函数中没有错误检查或异常处理，所以如果文件或目标目录不存在，它将失败得非常惨烈）。由于默认情况下`verbose`为`False`，移动操作将没有任何输出。

如果我们想查看输出，我们可以使用以下命令来调用它：

```py
>>> augmented_move("move_here", "three", verbose=True)
Moving three  
```

这将移动一个名为`three`的文件，并告诉我们它在做什么。注意，在这个例子中，不可能将`verbose`指定为位置参数；我们必须传递一个关键字参数。否则，Python 会认为它是`*filenames`列表中的另一个文件名。

如果我们想在列表中复制或忽略一些文件，而不是移动它们，我们可以传递额外的关键字参数，如下所示：

```py
>>> augmented_move("move_here", "four", "five", "six",
 four="copy", five="ignore")  
```

这将移动第六个文件并复制第四个，但由于我们没有指定`verbose`，所以不会显示任何输出。当然，我们也可以这样做，并且关键字参数可以以任何顺序提供，如下所示：

```py
>>> augmented_move("move_here", "seven", "eight", "nine",
 seven="copy", verbose=True, eight="ignore")
Copying seven
Ignoring eight
Moving nine  
```

# 参数解包

还有一个涉及可变参数和关键字参数的巧妙技巧。我们在一些之前的例子中使用过它，但解释永远不会太晚。给定一个值列表或字典，我们可以将这些值作为正常的位置参数或关键字参数传递给函数。看看这段代码：

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

当我们运行它时，看起来是这样的：

```py
Unpacking a sequence: 0 1 2
Unpacking a dict: ONE TWO THREE  
```

函数接受三个参数，其中一个具有默认值。但是，当我们有三个参数的列表时，我们可以在函数调用中使用`*`运算符将其解包为三个参数。如果我们有一个参数字典，我们可以使用`**`语法将其解包为关键字参数集合。

这通常在将收集自用户输入或外部来源（例如，网页或文本文件）的信息映射到函数或方法调用时非常有用。

记得我们之前使用文本文件中的标题和行来创建包含联系信息的字典列表的例子吗？我们不仅可以将字典添加到列表中，还可以使用关键字解包将参数传递给一个特别构建的`Contact`对象的`__init__`方法，该对象接受相同的参数集。看看你是否能修改这个例子使其工作。

这种解包语法也可以用在函数调用之外的一些区域。之前提到的`Options`类有一个`__init__`方法，看起来是这样的：

```py
 def __init__(self, **kwargs):
        self.options = dict(Options.default_options)
        self.options.update(kwargs)
```

做这件事的一个更简洁的方法是这样的：

```py
    def __init__(self, **kwargs):
        self.options = {**Options.default_options, **kwargs}
```

因为字典是从左到右按顺序解包的，所以生成的字典将包含所有默认选项，其中任何关键字参数选项将替换一些键。这里有一个例子：

```py
>>> x = {'a': 1, 'b': 2}
>>> y = {'b': 11, 'c': 3}
>>> z = {**x, **y}
>>> z
{'a': 1, 'b': 11, 'c': 3}
```

# 函数也是对象

过度强调面向对象原则的编程语言往往对不是方法的函数持批评态度。在这些语言中，你被期望创建一个对象来包装涉及的单个方法。有许多情况下，我们希望传递一个简单的对象来执行某个动作。这在事件驱动编程中最为常见，例如图形工具包或异步服务器；我们将在第十章设计模式 I 和第十一章设计模式 II 中看到一些使用它的设计模式。

在 Python 中，我们不需要将这些方法包装在对象中，因为函数本身就是对象！我们可以在函数上设置属性（尽管这不是一个常见的活动），并且我们可以将它们传递出去，以便在稍后的日期调用。它们甚至有一些可以直接访问的特殊属性。这里还有一个人为的例子：

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

如果我们运行这段代码，我们可以看到我们能够将两个不同的函数传递给我们的第三个函数，并且为每个函数得到不同的输出：

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

我们在函数上设置了一个名为`description`的属性（诚然，这些描述并不很好）。我们还能够看到函数的`__name__`属性，以及访问其类，这表明函数确实是一个具有属性的实体。然后，我们通过使用可调用语法（括号）调用了该函数。

函数作为顶级对象的事实最常用于将它们传递出去，以便在稍后的日期执行，例如，当满足某个条件时。让我们构建一个事件驱动的计时器来完成这个任务：

```py
import datetime
import time

class TimedEvent:
    def __init__(self, endtime, callback):
        self.endtime = endtime
 self.callback = callback

    def ready(self):
        return self.endtime <= datetime.datetime.now()

class Timer:
    def __init__(self):
        self.events = []

    def call_after(self, delay, callback):
        end_time = datetime.datetime.now() + datetime.timedelta(
            seconds=delay
        )

        self.events.append(TimedEvent(end_time, callback))

    def run(self):
        while True:
            ready_events = (e for e in self.events if e.ready())
            for event in ready_events:
 event.callback(self)
                self.events.remove(event)
            time.sleep(0.5)
```

在生产中，此代码应该使用 docstrings 添加额外的文档！`call_after`方法至少应该提到`delay`参数是以秒为单位的，并且`callback`函数应接受一个参数：执行调用的计时器。

这里有两个类。`TimedEvent`类并不打算被其他类访问；它所做的只是存储`endtime`和`callback`。我们甚至可以使用`tuple`或`namedtuple`，但考虑到方便给对象一个告诉我们事件是否准备好运行的行为，我们使用了一个类。

`Timer`类简单地存储一个即将发生的事件列表。它有一个`call_after`方法来添加新的事件。此方法接受一个表示在执行回调之前等待秒数的`delay`参数，以及`callback`函数本身：在正确时间执行的功能。此`callback`函数应接受一个参数。

`run`方法非常简单；它使用生成器表达式过滤掉任何时间已到的事件，并按顺序执行它们。然后，*计时器*循环无限期地继续，因此必须使用键盘中断（*Ctrl* + *C*，或*Ctrl* + *Break*）来中断它。我们每次迭代后休眠半秒钟，以免系统停止运行。

这里需要注意的重要事项是那些接触回调函数的行。该函数像任何其他对象一样被传递，计时器永远不会知道或关心该函数的原始名称或定义位置。当需要调用该函数时，计时器只需将括号语法应用于存储的变量。

这里有一组测试计时器的回调：

```py
def format_time(message, *args):
    now = datetime.datetime.now()
    print(f"{now:%I:%M:%S}: {message}")

def one(timer):
    format_time("Called One")

def two(timer):
    format_time("Called Two")

def three(timer):
    format_time("Called Three")

class Repeater:
    def __init__(self):
        self.count = 0

    def repeater(self, timer):
        format_time(f"repeat {self.count}")
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
format_time("Starting")
timer.run()
```

此示例让我们看到多个回调如何与计时器交互。第一个函数是`format_time`函数。它使用格式字符串语法将当前时间添加到消息中；我们将在下一章中了解它们。接下来，我们创建了三个简单的回调方法，它们只是输出当前时间和一条简短的消息，告诉我们哪个回调已被触发。

`Repeater`类演示了方法也可以用作回调，因为它们实际上只是绑定到对象的函数。它还展示了回调函数中的`timer`参数为什么有用：我们可以在当前正在运行的回调内部向计时器添加新的定时事件。然后我们创建一个计时器并向它添加了几个不同时间后调用的事件。最后，我们开始运行计时器；输出显示事件按预期顺序运行：

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

Python 3.4 引入了类似于这种的通用事件循环架构。我们将在第十三章中讨论它，*并发*。

# 使用函数作为属性

函数作为对象的一个有趣的效果是，它们可以被设置为其他对象的可调用属性。我们可以在实例化的对象上添加或更改一个函数，如下所示进行演示：

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

这段代码创建了一个非常简单的类，它有一个`print`方法，这个方法并没有告诉我们任何我们不知道的事情。然后，我们创建了一个新的函数，它告诉我们一些我们不相信的事情。

当我们在`A`类的实例上调用`print`时，它表现得如我们所预期。如果我们然后将`print`方法指向一个新的函数，它就会告诉我们一些不同的事情：

```py
my class is A 
my class is not A 
```

还可以在类上而不是对象上替换方法，尽管在这种情况下，我们必须在参数列表中添加`self`参数。这将改变该对象的所有实例的方法，即使是已经实例化的实例。显然，以这种方式替换方法既可能危险也可能令人困惑，难以维护。阅读代码的人会看到调用了某个方法，并会在原始类上查找该方法。但原始类上的方法并不是被调用的那个方法。弄清楚实际发生了什么可能成为一个棘手、令人沮丧的调试会话。

虽然它确实有其用途。通常，在运行时替换或添加方法（称为**猴子补丁**）用于自动化测试。如果我们正在测试客户端-服务器应用程序，我们可能不想在测试客户端时实际连接到服务器；这可能会导致意外转账或尴尬的测试邮件发送给真实的人。相反，我们可以设置我们的测试代码来替换对象上发送请求到服务器的一些关键方法，这样它只会记录这些方法已被调用。

猴子补丁也可以用来修复与我们交互的第三方代码中的错误或添加功能，而这些代码并不完全按照我们的需求运行。然而，应该谨慎使用；它几乎总是*一团糟的解决方案*。有时，尽管如此，它是唯一一种适应现有库以满足我们需求的方法。

# 可调用对象

正如函数是可以设置属性的物体一样，我们也可以创建一个可以被当作函数调用的对象。

任何对象都可以通过简单地给它一个接受所需参数的`__call__`方法来使其可调用。让我们通过使其可调用，使我们的`Repeater`类（来自计时器示例）更容易使用，如下所示：

```py
class Repeater: 
    def __init__(self): 
        self.count = 0 

 def __call__(self, timer): 
        format_time(f"repeat {self.count}") 
        self.count += 1 

        timer.call_after(5, self) 

timer = Timer() 

timer.call_after(5, Repeater()) 
format_time("{now}: Starting") 
timer.run() 
```

这个例子与之前的类并没有太大的不同；我们只是将 `repeater` 函数的名称更改为 `__call__`，并将对象本身作为可调用对象传递。请注意，当我们调用 `call_after` 时，我们传递了 `Repeater()` 参数。这两个括号创建了一个新实例；它们并不是显式地调用类。这发生在计时器内部。如果我们想在新生成的对象上执行 `__call__` 方法，我们将使用一个相当奇怪的语法：`Repeater()()`。第一组括号构建了对象；第二组执行 `__call__` 方法。如果我们发现自己这样做，我们可能没有使用正确的抽象。只有当对象被期望像函数一样处理时，才在对象上实现 `__call__` 函数。

# 案例研究

为了将本章中提出的一些原则结合起来，让我们构建一个邮件列表管理器。管理器将跟踪被分类到命名组中的电子邮件地址。当发送消息的时间到来时，我们可以选择一个组，并将消息发送到分配给该组的所有电子邮件地址。

在我们开始这个项目之前，我们应该有一个安全的方式来测试它，而无需向一大群真实的人发送电子邮件。幸运的是，Python 在这里支持我们；就像测试 HTTP 服务器一样，它有一个内置的 **简单邮件传输协议**（**SMTP**）服务器，我们可以指示它捕获我们发送的任何消息，而实际上并不发送它们。我们可以使用以下命令来运行服务器：

```py
$python -m smtpd -n -c DebuggingServer localhost:1025  
```

在命令提示符下运行此命令将在本地机器上启动一个运行在端口 1025 上的 SMTP 服务器。但我们指示它使用 `DebuggingServer` 类（这个类是内置 SMTP 模块的一部分），它不是将邮件发送给预期的收件人，而是在接收它们时简单地将它们打印在终端屏幕上。

现在，在我们编写邮件列表之前，让我们编写一些实际发送邮件的代码。当然，Python 在标准库中也支持这一点，但它的接口有点奇怪，所以我们将编写一个新函数来干净地包装它，如下面的代码片段所示：

```py
import smtplib
from email.mime.text import MIMEText

def send_email(
    subject,
    message,
    from_addr,
    *to_addrs,
    host="localhost",
    port=1025,
    **headers
):

    email = MIMEText(message)
    email["Subject"] = subject
    email["From"] = from_addr
    for header, value in headers.items():
        email[header] = value

    sender = smtplib.SMTP(host, port)
    for addr in to_addrs:
        del email["To"]
        email["To"] = addr
        sender.sendmail(from_addr, addr, email.as_string())
    sender.quit()
```

我们不会对这个方法内部的代码进行过于详细的介绍；标准库中的文档可以提供你使用 `smtplib` 和 `email` 模块所需的所有信息。

我们在函数调用中使用了变量参数和关键字参数语法。变量参数列表允许我们在只有一个 `to` 地址的默认情况下提供一个单独的字符串，以及在需要时允许提供多个地址。任何额外的关键字参数都映射到电子邮件头。这是变量参数和关键字参数的一个令人兴奋的使用，但它并不是一个很好的函数调用接口。实际上，它使得程序员想要做的许多事情变得不可能。

传递给函数的标题代表可以附加到方法上的辅助标题。这些标题可能包括`Reply-To`、`Return-Path`或*X-pretty-much-anything*。但是，为了在 Python 中成为一个有效的标识符，名称不能包含`-`字符。通常，该字符代表减法。因此，不可能用`Reply-To`=`my@email.com`来调用一个函数。正如经常发生的那样，我们似乎太急于使用关键字参数了，因为它们是我们刚刚学到的闪亮的新工具。

我们必须将参数更改为普通字典；这将有效，因为任何字符串都可以用作字典的键。默认情况下，我们希望这个字典为空，但我们不能将空字典作为默认参数。因此，我们将默认参数设置为`None`，然后在方法的开头设置字典，如下所示：

```py
def send_email(subject, message, from_addr, *to_addrs, 
        host="localhost", port=1025, headers=None): 

    headers = headers if headers else {}
```

如果我们在一个终端中运行调试 SMTP 服务器，我们可以在 Python 解释器中测试这段代码：

```py
>>> send_email("A model subject", "The message contents",
 "from@example.com", "to1@example.com", "to2@example.com")  
```

然后，如果我们检查调试 SMTP 服务器的输出，我们得到以下内容：

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

太好了，它已经*发送*了电子邮件到两个预期的地址，包括主题和消息内容。现在我们可以发送消息了，让我们来处理电子邮件组管理系统。我们需要一个对象，它能够以某种方式将电子邮件地址与它们所在的组匹配起来。由于这是一个多对多关系（任何一个电子邮件地址可以属于多个组；任何一个组可以与多个电子邮件地址相关联），我们研究过的数据结构似乎都不理想。我们可以尝试将组名与相关联的电子邮件地址列表匹配的字典，但这会导致电子邮件地址重复。我们也可以尝试将电子邮件地址与组匹配的字典，这会导致组重复。这两种方法似乎都不太理想。为了好玩，让我们尝试这个后一种版本，尽管直觉告诉我电子邮件地址与组解决方案会更直接。

由于我们字典中的值始终是唯一电子邮件地址的集合，我们可以将它们存储在`set`容器中。我们可以使用`defaultdict`来确保对于每个键都始终有一个`set`容器可用，如下所示：

```py
from collections import defaultdict

class MailingList:
    """Manage groups of e-mail addresses for sending e-mails."""

    def __init__(self):
        self.email_map = defaultdict(set)

    def add_to_group(self, email, group):
        self.email_map[email].add(group)
```

现在，让我们添加一个方法，允许我们收集一个或多个组中的所有电子邮件地址。这可以通过将组列表转换为集合来完成：

```py
def emails_in_groups(self, *groups): groups = set(groups) emails = set() for e, g in self.email_map.items(): if g & groups: emails.add(e) return emails 
```

首先，看看我们正在迭代的：`self.email_map.items()`。这个方法当然返回字典中每个项目的键值对元组。值是表示组的字符串集合。我们将这些值分成两个变量，分别命名为`e`和`g`，分别代表电子邮件和组。只有当传入的组与电子邮件地址组相交时，我们才将电子邮件地址添加到返回值的集合中。`g & groups`语法是`g.intersection(groups)`的快捷方式；`set`类通过实现特殊的`__and__`方法来调用`intersection`。

这段代码可以通过使用集合推导式变得更加简洁，我们将在第九章《迭代器模式》中讨论。

现在，有了这些构建块，我们可以轻易地向我们的`MailingList`类添加一个方法，用于向特定组发送消息：

```py
    def send_mailing(
        self, subject, message, from_addr, *groups, headers=None
    ):
        emails = self.emails_in_groups(*groups)
        send_email(
            subject, message, from_addr, *emails, headers=headers
        )
```

这个函数依赖于可变参数列表。作为输入，它接受一个作为可变参数的组列表。它获取指定组的电子邮件列表，并将这些作为可变参数传递给`send_email`，同时传递给这个方法的其他参数。

可以通过确保 SMTP 调试服务器在一个命令提示符中运行，并在第二个提示符中加载代码来测试程序：

```py
$python -i mailing_list.py  
```

使用以下命令创建一个`MailingList`对象：

```py
>>> m = MailingList()  
```

然后，创建一些伪造的电子邮件地址和组，例如：

```py
>>> m.add_to_group("friend1@example.com", "friends")
>>> m.add_to_group("friend2@example.com", "friends")
>>> m.add_to_group("family1@example.com", "family")
>>> m.add_to_group("pro1@example.com", "professional")  
```

最后，使用类似以下命令将电子邮件发送到特定组：

```py
>>> m.send_mailing("A Party",
"Friends and family only: a party", "me@example.com", "friends",
"family", headers={"Reply-To": "me2@example.com"})  
```

指定组中的每个地址的电子邮件应该显示在 SMTP 服务器的控制台上。

邮件列表本身运行良好，但有点无用；一旦我们退出程序，我们的信息数据库就会丢失。让我们修改它，添加一些方法来从文件中加载和保存电子邮件组列表。

通常，当在磁盘上存储结构化数据时，仔细考虑其存储方式是个好主意。众多数据库系统存在的一个原因就是，如果有人已经考虑过如何存储数据，你就不必再考虑。我们将在下一章探讨一些数据序列化机制，但在这个例子中，让我们保持简单，采用第一个可能可行的解决方案。

我心中所想的数据格式是存储每个电子邮件地址后跟一个空格，然后是逗号分隔的组列表。这种格式看起来是合理的，我们将采用它，因为数据格式化不是本章的主题。然而，为了说明为什么你需要认真思考如何在磁盘上格式化数据，让我们突出一些格式的问题。

首先，空格字符在电子邮件地址中在技术上是被允许的。大多数电子邮件提供商禁止使用它（有很好的理由），但定义电子邮件地址的规范说明，如果电子邮件地址在引号内，则可以包含空格。如果我们打算在我们的数据格式中使用空格作为哨兵，那么在技术上我们应该能够区分这个空格和电子邮件地址中的一部分空格。为了简单起见，我们将假装这不是真的，但现实生活中的数据编码充满了这类愚蠢的问题。

其次，考虑逗号分隔的组列表。如果有人决定在组名中放置一个逗号会发生什么？如果我们决定在组名中使逗号非法，我们应该在`add_to_group`方法中添加验证来强制执行这种命名。为了教学清晰，我们也将忽略这个问题。最后，还有许多我们需要考虑的安全影响：有人可以通过在他们的电子邮件地址中放置一个假的逗号来让自己进入错误的组吗？如果解析器遇到无效的文件，它会做什么？

从这次讨论中得到的启示是，尝试使用经过实地测试的数据存储方法，而不是设计我们自己的数据序列化协议。可能会有很多奇怪的边缘情况被忽视，而且使用已经遇到并修复了这些边缘情况的代码会更好。

但先别管那个。让我们只写一些基本的代码，用不切实际的幻想来假装这种简单的数据格式是安全的，如下所示：

```py
email1@mydomain.com group1,group2
email2@mydomain.com group2,group3  
```

实现这一点的代码如下：

```py
    def save(self):
        with open(self.data_file, "w") as file:
            for email, groups in self.email_map.items():
                file.write("{} {}\n".format(email, ",".join(groups)))

    def load(self):
        self.email_map = defaultdict(set)
        with suppress(IOError):
            with open(self.data_file) as file:
                for line in file:
                    email, groups = line.strip().split(" ")
                    groups = set(groups.split(","))
                    self.email_map[email] = groups
```

在`save`方法中，我们使用上下文管理器打开文件，并将文件写入为格式化的字符串。记住换行符；Python 不会为我们添加它。`load`方法首先重置字典（以防它包含来自之前`load`调用的数据）。它添加了一个对标准库`suppress`上下文管理器的调用，可通过`from contextlib import suppress`访问。这个上下文管理器会捕获任何 I/O 错误并忽略它们。这不是最好的错误处理方式，但比 try...finally...pass 更美观。

然后，加载方法使用`for...in`语法，它遍历文件中的每一行。同样，换行符包含在行变量中，所以我们必须调用`.strip()`来移除它。我们将在下一章中了解更多关于此类字符串操作的内容。

在使用这些方法之前，我们需要确保对象有一个`self.data_file`属性，这可以通过修改`__init__`来实现，如下所示：

```py
    def __init__(self, data_file): 
        self.data_file = data_file 
        self.email_map = defaultdict(set) 
```

我们可以在解释器中如下测试这两个方法：

```py
>>> m = MailingList('addresses.db')
>>> m.add_to_group('friend1@example.com', 'friends')
>>> m.add_to_group('family1@example.com', 'friends')
>>> m.add_to_group('family1@example.com', 'family')
>>> m.save()  
```

结果的`addresses.db`文件包含以下行，正如预期的那样：

```py
friend1@example.com friends
family1@example.com friends,family  
```

我们也可以成功地将这些数据重新加载到`MailingList`对象中：

```py
>>> m = MailingList('addresses.db')
>>> m.email_map
defaultdict(<class 'set'>, {})
>>> m.load()
>>> m.email_map
defaultdict(<class 'set'>, {'friend2@example.com': {'friends\n'}, 
'family1@example.com': {'family\n'}, 'friend1@example.com': {'friends\n'}})  
```

如您所见，我忘记添加了`load`命令，而且也可能容易忘记`save`命令。为了让任何想要在自己的代码中使用我们的`MailingList` API 的人更容易操作，让我们提供支持上下文管理器的函数：

```py
    def __enter__(self): 
        self.load() 
        return self 

    def __exit__(self, type, value, tb): 
        self.save() 
```

这些简单的方法只是将工作委托给加载和保存，但现在我们可以在交互式解释器中编写如下代码，并知道所有之前存储的地址都已被加载，而且当我们完成时，整个列表将被保存到文件中：

```py
>>> with MailingList('addresses.db') as ml:
...    ml.add_to_group('friend2@example.com', 'friends')
...    ml.send_mailing("What's up", "hey friends, how's it going", 'me@example.com', 
       'friends')  
```

# 练习

如果你之前没有遇到过`with`语句和上下文管理器，我鼓励你，像往常一样，检查你的旧代码，找到所有你曾经打开文件的地方，并确保它们使用`with`语句安全关闭。也要寻找可以编写自己的上下文管理器的地方。丑陋或重复的`try`...`finally`子句是一个很好的起点，但你可能会在任何需要执行上下文中的前后任务时发现它们很有用。

你可能之前已经使用过许多基本内置函数。我们介绍了几种，但并没有深入细节。玩转`enumerate`、`zip`、`reversed`、`any`和`all`，直到你确信自己会在需要时记得使用它们。`enumerate`函数尤其重要，因为不使用它会导致一些相当丑陋的`while`循环。

还要探索一些将函数作为可调用对象传递的应用，以及使用`__call__`方法使自己的对象可调用的方法。你可以通过将属性附加到函数或在一个对象上创建`__call__`方法来达到相同的效果。在哪种情况下你会使用一种语法，何时又更适合使用另一种语法？

如果我们的邮件列表对象需要发送大量邮件，可能会压倒邮件服务器。尝试重构它，以便你可以为不同的目的使用不同的`send_email`函数。这里使用的一个版本可能是一个版本，它可能将邮件放入队列，由不同线程或进程的服务器发送。第三个版本可能只是将数据输出到终端，从而消除了需要虚拟 SMTP 服务器的需求。你能构建一个带有回调的邮件列表，使得`send_mailing`函数使用传递的任何内容吗？如果没有提供回调，它将默认使用当前版本。

参数、关键字参数、可变参数和可变关键字参数之间的关系可能会有些令人困惑。我们在介绍多重继承时看到了它们如何痛苦地交互。设计一些其他示例来了解它们如何协同工作，以及了解它们何时不能协同工作。

# 摘要

在本章中，我们涵盖了一系列主题。每个主题都代表了一个在 Python 中流行的、重要的非面向对象特性。仅仅因为我们可以使用面向对象原则，并不意味着我们总是应该这样做！

然而，我们也看到 Python 通常通过提供传统面向对象语法的语法快捷方式来实现这些功能。了解这些工具背后的面向对象原则使我们能够更有效地在我们的类中使用它们。

我们讨论了一系列内置函数和文件输入输出操作。当我们调用带有参数、关键字参数和可变参数列表的函数时，有大量的不同语法可供选择。上下文管理器对于在两个方法调用之间嵌入代码的常见模式非常有用。甚至函数也是对象，反之，任何普通对象都可以被赋予可调用性。

在下一章中，我们将学习更多关于字符串和文件操作的知识，甚至还会花一些时间探讨标准库中最不面向对象的主题之一：正则表达式。
