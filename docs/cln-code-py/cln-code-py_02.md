# 第二章：Pythonic 代码

在本章中，我们将探索在 Python 中表达观念的方式，以及它自己的特点。如果您熟悉编程中一些任务的标准完成方式（例如获取列表的最后一个元素，迭代，搜索等），或者如果您来自更传统的编程语言（如 C、C++和 Java），那么您将发现，总的来说，Python 为大多数常见任务提供了自己的机制。

在编程中，成语是为了执行特定任务而编写代码的一种特定方式。它是一种常见的重复出现并且每次都遵循相同结构的东西。有些人甚至可能争论并称它们为一种模式，但要小心，因为它们不是设计模式（我们稍后将探讨）。主要区别在于设计模式是高级别的想法，独立于语言（在某种程度上），但它们不能立即转化为代码。另一方面，成语实际上是编码的。这是我们想要执行特定任务时应该编写的方式。

由于成语是代码，因此它们是与语言相关的。每种语言都有自己的习语，这意味着在该特定语言中完成任务的方式（例如，在 C、C++等语言中如何打开和写入文件）。当代码遵循这些习语时，它被称为成语化，而在 Python 中通常被称为**Pythonic**。

有多个原因要遵循这些建议并首先编写 Pythonic 代码（我们将看到并分析），以成语化的方式编写代码通常性能更好。它也更紧凑，更容易理解。这些都是我们希望在代码中始终具备的特征，以使其有效运行。其次，正如在上一章中介绍的，整个开发团队能够习惯相同的代码模式和结构非常重要，因为这将帮助他们专注于问题的真正本质，并帮助他们避免犯错。

本章的目标如下：

+   了解索引和切片，并正确实现可以进行索引的对象

+   实现序列和其他可迭代对象

+   学习上下文管理器的良好使用案例

+   通过魔术方法实现更成语化的代码

+   避免导致不良副作用的 Python 常见错误

# 索引和切片

与其他语言一样，在 Python 中，一些数据结构或类型支持通过索引访问其元素。它与大多数编程语言共有的另一点是，第一个元素位于索引号零。然而，与那些语言不同的是，当我们想以与通常不同的顺序访问元素时，Python 提供了额外的功能。

例如，在 C 语言中，如何访问数组的最后一个元素？这是我第一次尝试 Python 时做的事情。以与 C 语言相同的方式思考，我会得到数组长度减一的位置的元素。这可能有效，但我们也可以使用负索引号，它将从最后开始计数，如下面的命令所示：

```py
>>> my_numbers = (4, 5, 3, 9)
>>> my_numbers[-1]
9
>>> my_numbers[-3]
5
```

除了获取单个元素外，我们还可以使用`slice`获取多个元素，如下面的命令所示：

```py
>>> my_numbers = (1, 1, 2, 3, 5, 8, 13, 21)
>>> my_numbers[2:5]
(2, 3, 5)
```

在这种情况下，方括号中的语法意味着我们获取元组中的所有元素，从第一个数字的索引开始（包括该索引），直到第二个数字的索引（不包括该索引）。在 Python 中，切片的工作方式是通过排除所选区间的末尾来实现的。

您可以排除间隔的任一端点，开始或停止，这种情况下，它将分别从序列的开头或结尾起作用，如下面的命令所示：

```py
>>> my_numbers[:3]
(1, 1, 2)
>>> my_numbers[3:]
(3, 5, 8, 13, 21)
>>> my_numbers[::]
(1, 1, 2, 3, 5, 8, 13, 21)
>>> my_numbers[1:7:2]
(1, 3, 8)
```

在第一个示例中，它将获取到索引位置号为`3`的所有内容。在第二个示例中，它将获取从位置`3`（包括）开始到末尾的所有数字。在倒数第二个示例中，两端都被排除，实际上是创建了原始元组的副本。

最后一个例子包括第三个参数，即步长。这表示在迭代间隔时要跳过多少个元素。在这种情况下，它意味着获取位置为一和七之间的元素，每两个跳一次。

在所有这些情况下，当我们将间隔传递给一个序列时，实际上发生的是我们传递了`slice`。注意，`slice`是 Python 中的一个内置对象，你可以自己构建并直接传递：

```py
>>> interval = slice(1, 7, 2)
>>> my_numbers[interval]
(1, 3, 8)

>>> interval = slice(None, 3)
>>> my_numbers[interval] == my_numbers[:3]
True
```

注意，当元素中的一个缺失（开始、停止或步长），它被认为是无。

你应该始终优先使用这种内置的切片语法，而不是手动尝试在`for`循环中迭代元组、字符串或列表，手动排除元素。

# 创建你自己的序列

我们刚刚讨论的功能得益于一个叫做`__getitem__`的魔术方法。当像`myobject[key]`这样的东西被调用时，传递键（方括号内的值）作为参数调用这个方法。特别是，序列是一个实现了`__getitem__`和`__len__`的对象，因此它可以被迭代。列表、元组和字符串是标准库中序列对象的例子。

在这一部分，我们更关心通过关键字从对象中获取特定元素，而不是构建序列或可迭代对象，这是第七章中探讨的主题，*使用生成器*。

如果你要在你的领域的自定义类中实现`__getitem__`，你将不得不考虑一些问题，以便遵循 Pythonic 的方法。

如果你的类是标准库对象的包装器，你可能会尽可能地将行为委托给底层对象。这意味着如果你的类实际上是列表的包装器，调用列表上的所有相同方法，以确保它保持兼容。在下面的列表中，我们可以看到一个对象如何包装一个列表的例子，对于我们感兴趣的方法，我们只是委托给`list`对象上对应的版本：

```py
class Items:
    def __init__(self, *values):
        self._values = list(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        return self._values.__getitem__(item)
```

这个例子使用了封装。另一种方法是通过继承，这种情况下，我们将不得不扩展`collections.UserList`基类，考虑到本章的最后部分提到的注意事项和警告。

然而，如果你正在实现自己的序列，而不是一个包装器或不依赖于任何内置对象，那么请记住以下几点：

+   当通过范围进行索引时，结果应该是与类的相同类型的实例

+   在`slice`提供的范围内，遵守 Python 使用的语义，排除末尾的元素

第一点是一个微妙的错误。想想看——当你得到一个列表的`slice`时，结果是一个列表；当你在元组中请求一个范围时，结果是一个元组；当你请求一个子字符串时，结果是一个字符串。在每种情况下，结果与原始对象的类型相同是有道理的。如果你正在创建一个表示日期间隔的对象，并且你在该间隔上请求一个范围，返回一个列表或元组等都是错误的。相反，它应该返回一个设置了新间隔的相同类的新实例。最好的例子是在标准库中的`range`函数。在 Python 2 中，`range`函数用于构建一个列表。现在，如果你用一个间隔调用`range`，它将构造一个可迭代的对象，知道如何产生所选范围内的值。当你为 range 指定一个间隔时，你得到一个新的 range（这是有道理的），而不是一个列表：

```py
>>> range(1, 100)[25:50]
range(26, 51)
```

第二条规则也是关于一致性 - 代码的用户会发现如果与 Python 本身保持一致，那么使用起来更加熟悉和容易。作为 Python 开发人员，我们已经习惯了切片的工作方式，`range`函数的工作方式等。对自定义类做出异常会造成混乱，这意味着更难记住，可能导致错误。

# 上下文管理器

上下文管理器是 Python 提供的一个非常有用的特性。它们之所以如此有用的原因是它们正确地响应了一种模式。这种模式实际上是我们想要运行一些代码的每种情况，并且具有前置条件和后置条件，这意味着我们想在某个主要操作之前和之后运行一些东西。

大多数情况下，我们在资源管理周围看到上下文管理器。例如，在打开文件时，我们希望在处理后确保它们被关闭（这样我们就不会泄漏文件描述符），或者如果我们打开到服务（甚至是套接字）的连接，我们也希望相应地关闭它，或者在删除临时文件时等等。

在所有这些情况下，通常需要记住释放分配的所有资源，这只是考虑最佳情况，但是异常和错误处理呢？考虑到处理程序的所有可能组合和执行路径会使调试变得更加困难，解决这个问题的最常见方法是将清理代码放在`finally`块中，以确保不会遗漏它。例如，一个非常简单的情况看起来像下面这样：

```py
fd = open(filename)
try:
    process_file(fd)
finally:
    fd.close()
```

尽管如此，有一种更加优雅和 Pythonic 的方法来实现相同的功能：

```py
with open(filename) as fd:
    process_file(fd)
```

`with`语句（PEP-343）进入上下文管理器。在这种情况下，`open`函数实现了上下文管理器协议，这意味着文件将在块完成时自动关闭，即使发生异常也是如此。

上下文管理器由两个魔术方法组成：`__enter__`和`__exit__`。在上下文管理器的第一行，`with`语句将调用第一个方法`__enter__`，并且无论这个方法返回什么都将被分配给`as`后面标记的变量。这是可选的 - 我们不真的需要在`__enter__`方法上返回任何特定的东西，即使我们这样做了，如果不需要，也没有严格的理由将其分配给一个变量。

在执行这行之后，代码进入一个新的上下文，可以运行任何其他 Python 代码。在该块上的最后一条语句完成后，上下文将被退出，这意味着 Python 将调用我们首先调用的原始上下文管理器对象的`__exit__`方法。

如果在上下文管理器块内部发生异常或错误，`__exit__`方法仍然会被调用，这使得安全地管理清理条件变得方便。实际上，如果我们想以自定义方式处理，这个方法会接收在块上触发的异常。

尽管上下文管理器在处理资源时经常出现（比如我们提到的文件、连接等示例），但这并不是它们唯一的应用。我们可以实现自己的上下文管理器来处理我们需要的特定逻辑。

上下文管理器是分离关注点和隔离应该保持独立的代码部分的好方法，因为如果我们混合它们，那么逻辑将变得更难以维护。

举个例子，考虑这样一种情况：我们想要用一个脚本对数据库进行备份。问题在于备份是离线的，这意味着只有在数据库不运行时才能进行备份，为此我们必须停止它。备份完成后，我们希望确保无论备份过程本身如何进行，我们都要重新启动该进程。现在，第一种方法是创建一个巨大的单片函数，试图在同一个地方做所有事情，停止服务，执行备份任务，处理异常和所有可能的边缘情况，然后尝试重新启动服务。你可以想象这样一个函数，因此我将省略细节，而直接提出一种可能的解决这个问题的方式，即使用上下文管理器：

```py
def stop_database():
    run("systemctl stop postgresql.service")

def start_database():
    run("systemctl start postgresql.service")

class DBHandler:
    def __enter__(self):
        stop_database()
        return self

    def __exit__(self, exc_type, ex_value, ex_traceback):
        start_database()

def db_backup():
    run("pg_dump database")

def main():
    with DBHandler():
        db_backup()
```

在这个例子中，我们不需要上下文管理器在块内的结果，这就是为什么我们可以认为，至少对于这种特殊情况，`__enter__`的返回值是无关紧要的。在设计上下文管理器时，这是需要考虑的事情——一旦块开始，我们需要什么？作为一个一般规则，总是在`__enter__`上返回一些东西应该是一个好的做法（尽管不是强制性的）。

在这个块中，我们只运行备份任务，独立于维护任务，就像我们之前看到的那样。我们还提到，即使备份任务出现错误，`__exit__`仍然会被调用。

注意`__exit__`方法的签名。它接收了在块上引发的异常的值。如果块上没有异常，它们都是`None`。

`__exit__`的返回值是需要考虑的。通常，我们希望保持该方法不变，不返回任何特定的内容。如果该方法返回`True`，这意味着潜在引发的异常不会传播到调用者那里，而是在此处停止。有时，这是期望的效果，甚至可能取决于引发的异常类型，但一般来说，吞没异常并不是一个好主意。记住：错误不应该悄悄地传递。

请记住不要在`__exit__`上意外返回`True`。如果你这样做了，请确保这确实是你想要的，并且有一个很好的理由。

# 实现上下文管理器

一般来说，我们可以像前面的例子一样实现上下文管理器。我们只需要一个实现`__enter__`和`__exit__`魔术方法的类，然后该对象就能支持上下文管理器协议。虽然这是实现上下文管理器最常见的方式，但并不是唯一的方式。

在本节中，我们将看到不仅实现上下文管理器的不同（有时更紧凑）的方法，还将看到如何通过使用标准库，特别是`contextlib`模块，充分利用它们。

`contextlib`模块包含了许多辅助函数和对象，可以实现上下文管理器，或者使用一些已经提供的可以帮助我们编写更紧凑代码的上下文管理器。

让我们从看`contextmanager`装饰器开始。

当`contextlib.contextmanager`装饰器应用于一个函数时，它将该函数中的代码转换为上下文管理器。所涉及的函数必须是一种特殊类型的函数，称为**生成器**函数，它将语句分开成分别位于`__enter__`和`__exit__`魔术方法中的内容。

如果你现在对装饰器和生成器不熟悉，这并不是问题，因为我们将要看的例子是独立的，而且这个方法或习惯可以被应用和理解。这些主题在第七章中有详细讨论，*使用生成器*。

前面例子的等价代码可以用`contextmanager`装饰器重写如下：

```py
import contextlib

@contextlib.contextmanager
def db_handler():
    stop_database()
    yield
    start_database()

with db_handler():
    db_backup()
```

在这里，我们定义了生成器函数，并将`@contextlib.contextmanager`装饰器应用到它上面。该函数包含一个`yield`语句，这使它成为一个生成器函数。在这种情况下，生成器的细节并不相关。我们只需要知道，当应用这个装饰器时，`yield`语句之前的所有内容将被视为`__enter__`方法的一部分运行。然后，yield 的值将成为上下文管理器评估的结果（`__enter__`将返回的内容），如果我们选择像`as x:`这样分配它的话，将被分配给变量——在这种情况下，没有任何 yield（这意味着 yield 的值将是隐式的 none），但如果我们想要的话，我们可以 yield 一个语句，这将成为我们可能想要在上下文管理器块中使用的东西。

在那一点上，生成器函数被挂起，进入上下文管理器，在那里，我们再次运行数据库的备份代码。完成后，执行恢复，因此我们可以认为`yield`语句之后的每一行都将成为`__exit__`逻辑的一部分。

像这样编写上下文管理器的优势在于，更容易重构现有函数，重用代码，总的来说，当我们需要一个不属于任何特定对象的上下文管理器时，这是一个好主意。添加额外的魔术方法会使我们领域中的另一个对象更加耦合，责任更多，并支持一些它可能不应该支持的东西。当我们只需要一个上下文管理器函数，而不需要保留许多状态，并且完全独立于我们的其他类时，这可能是一个不错的选择。

然而，我们可以以更多的方式实现上下文管理器，再一次，答案在标准库的`contextlib`包中。

我们还可以使用`contextlib.ContextDecorator`这个辅助工具。这是一个混合基类，提供了将装饰器应用到函数的逻辑，使其在上下文管理器中运行，而上下文管理器本身的逻辑必须通过实现前面提到的魔术方法来提供。

为了使用它，我们必须扩展这个类，并在所需的方法上实现逻辑：

```py
class dbhandler_decorator(contextlib.ContextDecorator):
    def __enter__(self):
        stop_database()

    def __exit__(self, ext_type, ex_value, ex_traceback):
        start_database()

@dbhandler_decorator()
def offline_backup():
    run("pg_dump database")
```

你注意到和之前的例子有什么不同了吗？这里没有`with`语句。我们只需要调用这个函数，`offline_backup()`就会自动在上下文管理器中运行。这是基类提供的逻辑，用作装饰器包装原始函数，使其在上下文管理器中运行。

这种方法的唯一缺点是，对象完全独立运作（这是一个好特点）——装饰器对装饰的函数一无所知，反之亦然。然而，这意味着你无法获得一个你想要在上下文管理器中使用的对象（例如，分配`with offline_backup() as bp:`），所以如果你真的需要使用`__exit__`方法返回的对象，之前的方法将成为首选。

作为一个装饰器，这也带来了一个优势，即逻辑只定义一次，我们可以通过简单地将装饰器应用到其他需要相同不变逻辑的函数上，来重复使用它。

让我们探索`contextlib`的最后一个特性，看看我们可以从上下文管理器中期望什么，并了解我们可以用它们做什么样的事情。

请注意，`contextlib.suppress`是一个`util`包，它进入一个上下文管理器，如果其中一个提供的异常被触发，它不会失败。这类似于在`try`/`except`块上运行相同的代码并传递异常或记录它，但不同之处在于调用`suppress`方法更明确地表明那些作为我们逻辑一部分受控的异常。

例如，考虑以下代码：

```py
import contextlib

with contextlib.suppress(DataConversionException):
      parse_data(input_json_or_dict)
```

在这里，异常的存在意味着输入数据已经是预期格式，因此不需要转换，因此可以安全地忽略它。

# 属性、属性和对象的不同类型的方法

在 Python 中，对象的所有属性和函数都是公开的，这与其他语言不同，其他语言中属性可以是公共的、私有的或受保护的。也就是说，没有必要阻止调用对象调用对象具有的任何属性。这是与其他编程语言的另一个不同之处，其他编程语言可以将一些属性标记为私有或受保护。

没有严格的强制，但有一些约定。以下划线开头的属性意味着它是对象的私有属性，我们期望没有外部代理调用它（但同样，没有阻止这种情况）。

在深入了解属性的细节之前，值得提到 Python 中下划线的一些特点，理解约定和属性的范围。

# Python 中的下划线

Python 中有一些约定和实现细节，使用下划线是一个有趣的话题，值得分析。

正如我们之前提到的，默认情况下，对象的所有属性都是公开的。考虑以下示例来说明这一点：

```py
>>> class Connector:
...     def __init__(self, source):
...         self.source = source
...         self._timeout = 60
... 
>>> conn = Connector("postgresql://localhost")
>>> conn.source
'postgresql://localhost'
>>> conn._timeout
60
>>> conn.__dict__
{'source': 'postgresql://localhost', '_timeout': 60}
```

在这里，创建了一个`Connector`对象与`source`，并且它开始有两个属性——前面提到的`source`和`timeout`。前者是公开的，后者是私有的。然而，正如我们从以下行中看到的，当我们创建这样一个对象时，我们实际上可以访问它们两个。

这段代码的解释是，`_timeout`应该只在`connector`内部访问，而不是从调用者访问。这意味着你应该以一种安全的方式组织代码，以便在所有需要的时间安全地重构超时，依赖于它不是从对象外部调用（只在内部调用），因此保持与以前相同的接口。遵守这些规则使代码更容易维护，更健壮，因为我们在重构代码时不必担心连锁反应，如果我们保持对象的接口不变。同样的原则也适用于方法。

对象应该只公开对外部调用对象相关的属性和方法，即其接口。一切不严格属于对象接口的东西都应该以单下划线为前缀。

这是清晰地界定对象接口的 Python 方式。然而，有一个常见的误解，即一些属性和方法实际上可以被私有化。这又是一个误解。让我们想象一下，现在`timeout`属性定义为双下划线：

```py
>>> class Connector:
...     def __init__(self, source):
...         self.source = source
...         self.__timeout = 60
...
...      def connect(self):
...         print("connecting with {0}s".format(self.__timeout))
...         # ...
... 
>>> conn = Connector("postgresql://localhost")
>>> conn.connect()
connecting with 60s
>>> conn.__timeout
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
AttributeError: 'Connector' object has no attribute '__timeout'
```

一些开发人员使用这种方法来隐藏一些属性，像在这个例子中一样，认为`timeout`现在是`私有的`，没有其他对象可以修改它。现在，看一下尝试访问`__timeout`时引发的异常。它是`AttributeError`，表示它不存在。它没有说像“这是私有的”或“这不能被访问”等等。它说它不存在。这应该给我们一个线索，实际上发生了不同的事情，这种行为只是一个副作用，而不是我们想要的真正效果。

实际上发生的是，使用双下划线，Python 为属性创建了一个不同的名称（这称为**名称混淆**）。它创建的属性的名称如下：`"_<class-name>__<attribute-name>"`。在这种情况下，将创建一个名为`'_Connector__timeout'`的属性，可以通过以下方式访问（和修改）这样的属性：

```py
>>> vars(conn)
{'source': 'postgresql://localhost', '_Connector__timeout': 60}
>>> conn._Connector__timeout
60
>>> conn._Connector__timeout = 30
>>> conn.connect()
connecting with 30s
```

注意我们之前提到的副作用——属性只存在不同的名称，因此在我们第一次尝试访问它时引发了`AttributeError`。

Python 中双下划线的概念完全不同。它被创建为一种方式，用于覆盖将被多次扩展的类的不同方法，而不会出现与方法名称的冲突的风险。即使这是一个牵强的用例，也不能证明使用这种机制的必要性。

双下划线是一种非 Pythonic 的方法。如果需要将属性定义为私有的，请使用单下划线，并遵守 Pythonic 的约定，即它是一个私有属性。

不要使用双下划线。

# 属性

当对象只需要保存值时，我们可以使用常规属性。有时，我们可能希望根据对象的状态和其他属性的值进行一些计算。大多数情况下，属性是一个不错的选择。

当我们需要定义对象中某些属性的访问控制时，属性就应该被使用，这是 Python 在另一个方面有自己的做事方式的地方。在其他编程语言（如 Java）中，你会创建访问方法（getter 和 setter），但惯用的 Python 会使用属性。

想象一下，我们有一个用户可以注册的应用程序，我们希望保护用户的某些信息不被错误地修改，比如他们的电子邮件，如下面的代码所示：

```py
import re

EMAIL_FORMAT = re.compile(r"[^@]+@[^@]+\.[^@]+")

def is_valid_email(potentially_valid_email: str):
    return re.match(EMAIL_FORMAT, potentially_valid_email) is not None

class User:
    def __init__(self, username):
        self.username = username
        self._email = None

    @property
    def email(self):
        return self._email

    @email.setter
    def email(self, new_email):
        if not is_valid_email(new_email):
            raise ValueError(f"Can't set {new_email} as it's not a 
            valid email")
        self._email = new_email
```

通过将电子邮件放在属性下，我们可以免费获得一些优势。在这个例子中，第一个`@property`方法将返回私有属性`email`保存的值。如前所述，前导下划线确定了这个属性是私有的，因此不应该从这个类的外部访问。

然后，第二个方法使用了`@email.setter`，使用了前一个方法已经定义的属性。当调用者代码中运行`<user>.email = <new_email>`时，将调用这个方法，`<new_email>`将成为这个方法的参数。在这里，我们明确定义了一个验证，如果试图设置的值不是实际的电子邮件地址，将失败。如果是，它将使用新值更新属性，如下所示：

```py
>>> u1 = User("jsmith")
>>> u1.email = "jsmith@"
Traceback (most recent call last):
...
ValueError: Can't set jsmith@ as it's not a valid email
>>> u1.email = "jsmith@g.co"
>>> u1.email
'jsmith@g.co'
```

这种方法比使用以`get_`或`set_`为前缀的自定义方法要紧凑得多。因为它只是`email`，所以期望是清晰的。

不要为对象的所有属性编写自定义的`get_*`和`set_*`方法。大多数情况下，将它们作为常规属性留下就足够了。如果需要修改检索或修改属性时的逻辑，那么使用属性。

您可能会发现属性是实现命令和查询分离（CC08）的一种好方法。命令和查询分离表明对象的方法应该要么回答问题，要么执行操作，但不能两者兼而有之。如果对象的方法既在做某事，又同时返回一个回答该操作进行得如何的状态，那么它做了超过一件事，显然违反了函数应该只做一件事的原则。

根据方法的名称，这可能会导致更多的混淆，使读者更难理解代码的实际意图。例如，如果一个方法被称为`set_email`，我们使用它作为`if self.set_email("a@j.com"): ...`，那么这段代码在做什么？它是将电子邮件设置为`a@j.com`吗？它是在检查电子邮件是否已经设置为该值吗？两者（设置然后检查状态是否正确）？

通过属性，我们可以避免这种混淆。`@property`装饰器是回答问题的查询，`@<property_name>.setter`是执行命令的命令。

从这个例子中得出的另一个好建议是——不要在一个方法上做超过一件事。如果你想分配一些东西然后检查值，把它分解成两个或更多个句子。

方法应该只做一件事。如果你必须运行一个动作，然后检查状态，那么应该在不同的语句中调用不同的方法。

# 可迭代对象

在 Python 中，我们有默认可迭代的对象。例如，列表、元组、集合和字典不仅可以以我们想要的结构保存数据，还可以在`for`循环中重复获取这些值。

然而，内置的可迭代对象并不是我们在`for`循环中唯一可以拥有的类型。我们还可以创建自己的可迭代对象，并定义迭代的逻辑。

为了实现这一点，我们再次依赖于魔术方法。

迭代是通过 Python 自己的协议（即迭代协议）工作的。当你尝试以`for e in myobject:...`的形式迭代一个对象时，Python 在非常高的层次上检查以下两件事，按顺序：

+   如果对象包含迭代器方法之一——`__next__`或`__iter__`

+   如果对象是一个序列，并且具有`__len__`和`__getitem__`

因此，作为后备机制，序列可以被迭代，因此有两种方法可以自定义我们的对象以在`for`循环中工作。

# 创建可迭代对象

当我们尝试迭代一个对象时，Python 将在其上调用`iter()`函数。这个函数首先检查的是该对象是否存在`__iter__`方法，如果存在，将执行该方法。

以下代码创建了一个对象，允许在一系列日期上进行迭代，每次循环产生一天：

```py
from datetime import timedelta

class DateRangeIterable:
    """An iterable that contains its own iterator object."""

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self._present_day = start_date

    def __iter__(self):
        return self

    def __next__(self):
        if self._present_day >= self.end_date:
            raise StopIteration
        today = self._present_day
        self._present_day += timedelta(days=1)
        return today
```

该对象旨在使用一对日期创建，并在迭代时，将产生指定日期间隔内的每一天，如下代码所示：

```py
>>> for day in DateRangeIterable(date(2018, 1, 1), date(2018, 1, 5)):
...     print(day)
... 
2018-01-01
2018-01-02
2018-01-03
2018-01-04
>>> 
```

在这里，`for`循环开始对我们的对象进行新的迭代。此时，Python 将在其上调用`iter()`函数，然后`iter()`函数将调用`__iter__`魔术方法。在这个方法中，它被定义为返回 self，表示对象本身是可迭代的，因此在每一步循环中都将在该对象上调用`next()`函数，该函数委托给`__next__`方法。在这个方法中，我们决定如何产生元素并一次返回一个。当没有其他东西可以产生时，我们必须通过引发`StopIteration`异常向 Python 发出信号。

这意味着实际上发生的情况类似于 Python 每次在我们的对象上调用`next()`，直到出现`StopIteration`异常，然后它知道必须停止`for`循环：

```py
>>> r = DateRangeIterable(date(2018, 1, 1), date(2018, 1, 5))
>>> next(r)
datetime.date(2018, 1, 1)
>>> next(r)
datetime.date(2018, 1, 2)
>>> next(r)
datetime.date(2018, 1, 3)
>>> next(r)
datetime.date(2018, 1, 4)
>>> next(r)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File ... __next__
 raise StopIteration
StopIteration
>>> 
```

这个例子是有效的，但存在一个小问题——一旦耗尽，可迭代对象将继续为空，因此引发`StopIteration`。这意味着如果我们在两个或更多连续的`for`循环中使用它，只有第一个会起作用，而第二个会为空：

```py
>>> r1 = DateRangeIterable(date(2018, 1, 1), date(2018, 1, 5))
>>> ", ".join(map(str, r1))
'2018-01-01, 2018-01-02, 2018-01-03, 2018-01-04'
>>> max(r1)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
ValueError: max() arg is an empty sequence
>>> 
```

这是因为迭代协议的工作方式——一个可迭代对象构造一个迭代器，这个迭代器被迭代。在我们的例子中，`__iter__`只是返回了`self`，但我们可以让它每次调用时创建一个新的迭代器。修复这个问题的一种方法是创建`DateRangeIterable`的新实例，这不是一个可怕的问题，但我们可以让`__iter__`使用生成器（它是迭代器对象），每次创建一个：

```py
class DateRangeContainerIterable:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def __iter__(self):
        current_day = self.start_date
        while current_day < self.end_date:
            yield current_day
            current_day += timedelta(days=1)
```

这一次，它起作用了：

```py
>>> r1 = DateRangeContainerIterable(date(2018, 1, 1), date(2018, 1, 5))
>>> ", ".join(map(str, r1))
'2018-01-01, 2018-01-02, 2018-01-03, 2018-01-04'
>>> max(r1)
datetime.date(2018, 1, 4)
>>> 
```

不同之处在于每个`for`循环都会再次调用`__iter__`，并且每个`for`循环都会再次创建生成器。

这被称为**容器**可迭代对象。

一般来说，处理生成器时最好使用容器可迭代对象。

有关生成器的详细信息将在第七章中详细解释，*使用生成器*。

# 创建序列

也许我们的对象没有定义`__iter__()`方法，但我们仍然希望能够对其进行迭代。如果对象上没有定义`__iter__`，`iter()`函数将查找`__getitem__`的存在，如果找不到，将引发`TypeError`。

序列是一个实现`__len__`和`__getitem__`的对象，并期望能够按顺序一次获取它包含的元素，从零开始作为第一个索引。这意味着你应该在逻辑上小心，以便正确实现`__getitem__`以期望这种类型的索引，否则迭代将无法工作。

前一节的示例有一个优点，它使用的内存更少。这意味着它一次只保存一个日期，并且知道如何逐个生成日期。然而，它的缺点是，如果我们想要获取第 n 个元素，我们除了迭代 n 次直到达到它之外别无选择。这是计算机科学中内存和 CPU 使用之间的典型权衡。

使用可迭代的实现会占用更少的内存，但获取一个元素最多需要*O(n)*的时间，而实现一个序列会占用更多的内存（因为我们必须一次性保存所有东西），但支持常数时间的索引，*O(1)*。

这就是新实现可能看起来的样子：

```py
class DateRangeSequence:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self._range = self._create_range()

    def _create_range(self):
        days = []
        current_day = self.start_date
        while current_day < self.end_date:
            days.append(current_day)
            current_day += timedelta(days=1)
        return days

    def __getitem__(self, day_no):
        return self._range[day_no]

    def __len__(self):
        return len(self._range)
```

这是对象的行为：

```py
>>> s1 = DateRangeSequence(date(2018, 1, 1), date(2018, 1, 5))
>>> for day in s1:
...     print(day)
... 
2018-01-01
2018-01-02
2018-01-03
2018-01-04
>>> s1[0]
datetime.date(2018, 1, 1)
>>> s1[3]
datetime.date(2018, 1, 4)
>>> s1[-1]
datetime.date(2018, 1, 4)
```

在前面的代码中，我们可以看到负索引也是有效的。这是因为`DateRangeSequence`对象将所有操作委托给其包装对象（一个`list`），这是保持兼容性和一致行为的最佳方式。

在决定使用哪种可能的实现时，要评估内存和 CPU 使用之间的权衡。一般来说，迭代是可取的（甚至是生成器），但要记住每种情况的要求。

# 容器对象

容器是实现`__contains__`方法的对象（通常返回一个布尔值）。在 Python 中的`in`关键字的存在下会调用这个方法。

类似下面这样的：

```py
element in container
```

在 Python 中使用时变成这样：

```py
container.__contains__(element)
```

当这种方法被正确实现时，你可以想象代码会变得更可读（并且更 Pythonic！）。

假设我们必须在一个具有二维坐标的游戏地图上标记一些点。我们可能期望找到以下函数：

```py
def mark_coordinate(grid, coord):
    if 0 <= coord.x < grid.width and 0 <= coord.y < grid.height:
        grid[coord] = MARKED
```

现在，检查第一个`if`语句条件的部分似乎很复杂；它没有显示代码的意图，不够表达，最糟糕的是它要求代码重复（在代码的每个部分在继续之前都需要重复那个`if`语句）。

如果地图本身（在代码中称为`grid`）能够回答这个问题怎么办？更好的是，如果地图能够将这个动作委托给一个更小（因此更内聚）的对象呢？因此，我们可以问地图是否包含一个坐标，地图本身可以有关于其限制的信息，并询问这个对象以下内容：

```py
class Boundaries:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __contains__(self, coord):
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.limits = Boundaries(width, height)

    def __contains__(self, coord):
        return coord in self.limits
```

这段代码本身就是一个更好的实现。首先，它进行了简单的组合，并使用委托来解决问题。两个对象都非常内聚，具有尽可能少的逻辑；方法很短，逻辑清晰明了——`coord in self.limits`基本上就是对要解决的问题的声明，表达了代码的意图。

从外部来看，我们也能看到好处。几乎就像 Python 在为我们解决问题：

```py
def mark_coordinate(grid, coord):
    if coord in grid:
        grid[coord] = MARKED
```

# 对象的动态属性

可以通过`__getattr__`魔术方法来控制从对象中获取属性的方式。当我们调用类似`<myobject>.<myattribute>`的东西时，Python 会在对象的字典中查找`<myattribute>`，并调用`__getattribute__`。如果没有找到（即对象没有我们要找的属性），那么会调用额外的方法`__getattr__`，并将属性的名称（`myattribute`）作为参数传递。通过接收这个值，我们可以控制返回给我们对象的方式。我们甚至可以创建新的属性等等。

在下面的清单中，演示了`__getattr__`方法：

```py
class DynamicAttributes:

    def __init__(self, attribute):
        self.attribute = attribute

    def __getattr__(self, attr):
        if attr.startswith("fallback_"):
            name = attr.replace("fallback_", "")
            return f"[fallback resolved] {name}"
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute {attr}"
        )
```

这是对该类对象的一些调用：

```py
>>> dyn = DynamicAttributes("value")
>>> dyn.attribute
'value'

>>> dyn.fallback_test
'[fallback resolved] test'

>>> dyn.__dict__["fallback_new"] = "new value"
>>> dyn.fallback_new
'new value'

>>> getattr(dyn, "something", "default")
'default'
```

第一个调用很简单——我们只是请求对象具有的属性，并将其值作为结果。第二个是这个方法发挥作用的地方，因为对象没有任何叫做`fallback_test`的东西，所以`__getattr__`将以该值运行。在该方法内部，我们放置了返回一个字符串的代码，我们得到的是该转换的结果。

第三个例子很有趣，因为这里创建了一个名为`fallback_new`的新属性（实际上，这个调用与运行`dyn.fallback_new = "new value"`是一样的），所以当我们请求该属性时，注意到我们放在`__getattr__`中的逻辑不适用，因为那段代码根本没有被调用。

现在，最后一个例子是最有趣的。这里有一个微妙的细节，这会产生很大的差异。再看一下`__getattr__`方法中的代码。注意当值不可检索时它引发的异常`AttributeError`。这不仅是为了一致性（以及异常中的消息），而且也是内置的`getattr()`函数所要求的。如果这个异常是其他任何异常，它都会引发，而默认值将不会被返回。

在实现`__getattr__`这样动态的方法时要小心，并谨慎使用。在实现`__getattr__`时，要引发`AttributeError`。

# 可调用对象

定义可以作为函数的对象是可能的（而且通常很方便）。其中最常见的应用之一是创建更好的装饰器，但不仅限于此。

当我们尝试执行我们的对象，就好像它是一个常规函数一样时，魔术方法`__call__`将被调用。传递给它的每个参数都将传递给`__call__`方法。

通过这种方式实现函数的主要优势是，对象具有状态，因此我们可以在调用之间保存和维护信息。

当我们有一个对象时，类似这样的语句`object(*args, **kwargs)`在 Python 中被翻译为`object.__call__(*args, **kwargs)`。

当我们想要创建可作为带参数函数的可调用对象时，这种方法非常有用，或者在某些情况下是具有记忆功能的函数。

以下清单使用此方法构建一个对象，当使用参数调用时，返回它已经使用相同值调用的次数：

```py
from collections import defaultdict

class CallCount:

    def __init__(self):
        self._counts = defaultdict(int)

    def __call__(self, argument):
        self._counts[argument] += 1
        return self._counts[argument]
```

这个类的一些示例操作如下：

```py
>>> cc = CallCount()
>>> cc(1)
1
>>> cc(2)
1
>>> cc(1)
2
>>> cc(1)
3
>>> cc("something")
1
```

在本书的后面，我们将发现这种方法在创建装饰器时非常方便。

# 魔术方法总结

我们可以总结前面描述的概念，形成一个类似下面所示的速查表。对于 Python 中的每个操作，都会呈现涉及的魔术方法，以及它所代表的概念：

| **语句** | **魔术方法** | **Python 概念** |
| --- | --- | --- |
| `obj[key]``obj[i:j]``obj[i:j:k]` | `__getitem__(key)` | 可以进行下标操作的对象 |
| `with obj: ...` | `__enter__` / `__exit__` | 上下文管理器 |
| `for i in obj: ...` | `__iter__` / `__next__``__len__` / `__getitem__` | 可迭代对象序列 |
| `obj.<attribute>` | `__getattr__` | 动态属性检索 |
| `obj(*args, **kwargs)` | `__call__(*args, **kwargs)` | 可调用对象 |

# Python 中的注意事项

除了理解语言的主要特性之外，能够编写惯用代码也意味着要意识到一些习语的潜在问题，以及如何避免它们。在本节中，我们将探讨一些常见问题，如果让你措手不及，可能会导致长时间的调试会话。

本节讨论的大部分观点都是要完全避免的，我敢说几乎没有可能的情况能够证明反模式（或者在这种情况下是习语）的存在是合理的。因此，如果你在你正在工作的代码库中发现了这种情况，可以随意按照建议进行重构。如果你在进行代码审查时发现了这些特征，这清楚地表明需要做出一些改变。

# 可变默认参数

简单来说，不要将可变对象用作函数的默认参数。如果您将可变对象用作默认参数，您将得到意料之外的结果。

考虑以下错误的函数定义：

```py
def wrong_user_display(user_metadata: dict = {"name": "John", "age": 30}):
    name = user_metadata.pop("name")
    age = user_metadata.pop("age")

    return f"{name} ({age})"
```

实际上，这有两个问题。除了默认的可变参数外，函数体正在改变一个可变对象，因此产生了副作用。但主要问题是`user_medatada`的默认参数。

实际上，这只会在第一次不带参数调用时起作用。第二次，我们在不明确传递任何内容给`user_metadata`的情况下调用它。它将失败并显示`KeyError`，如下所示：

```py
>>> wrong_user_display()
'John (30)'
>>> wrong_user_display({"name": "Jane", "age": 25})
'Jane (25)'
>>> wrong_user_display()
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File ... in wrong_user_display
 name = user_metadata.pop("name")
KeyError: 'name' 
```

解释很简单 - 在函数定义中将带有默认数据的字典分配给`user_metadata`，实际上是创建了一个字典，并且变量`user_metadata`指向它。函数体修改了这个对象，在程序运行时它会一直存在于内存中。当我们给它传递一个值时，这将取代我们刚刚创建的默认参数。当我们不想要这个对象时再次调用它，并且自上次运行以来它已经被修改；下一次运行它时，将不包含键，因为它们在上一次调用时被移除了。

修复也很简单 - 我们需要使用`None`作为默认的标记值，并在函数体中分配默认值。因为每个函数都有自己的作用域和生命周期，`user_metadata`将在每次出现`None`时被分配给字典：

```py
def user_display(user_metadata: dict = None):
    user_metadata = user_metadata or {"name": "John", "age": 30}

    name = user_metadata.pop("name")
    age = user_metadata.pop("age")

    return f"{name} ({age})"
```

# 扩展内置类型

正确的扩展内置类型（如列表、字符串和字典）的方法是使用`collections`模块。

如果您直接扩展 dict 等类，您将得到可能不是您期望的结果。这是因为在 CPython 中，类的方法不会相互调用（应该调用），因此如果您覆盖其中一个方法，这不会被其他方法反映出来，导致意外的结果。例如，您可能想要覆盖`__getitem__`，然后当您使用`for`循环迭代对象时，您会注意到您在该方法中放置的逻辑没有被应用。

这一切都可以通过使用`collections.UserDict`来解决，它提供了对实际字典的透明接口，并且更加健壮。

假设我们想要一个最初由数字创建的列表将值转换为字符串，并添加前缀。第一种方法可能看起来解决了问题，但是是错误的：

```py
class BadList(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        if index % 2 == 0:
            prefix = "even"
        else:
            prefix = "odd"
        return f"[{prefix}] {value}"
```

乍一看，它看起来像我们想要的对象行为。但是，如果我们尝试迭代它（毕竟，它是一个列表），我们会发现我们得不到我们想要的东西：

```py
>>> bl = BadList((0, 1, 2, 3, 4, 5))
>>> bl[0]
'[even] 0'
>>> bl[1]
'[odd] 1'
>>> "".join(bl)
Traceback (most recent call last):
...
TypeError: sequence item 0: expected str instance, int found
```

`join`函数将尝试迭代（在列表上运行`for`循环），但期望的是字符串类型的值。这应该可以工作，因为这正是我们对列表所做的更改，但显然在迭代列表时，我们修改的`__getitem__`版本没有被调用。

这实际上是 CPython 的一个实现细节（一种 C 优化），在其他平台（如 PyPy）中不会发生（请参阅本章末尾的 PyPy 和 CPython 之间的差异）。

尽管如此，我们应该编写可移植且兼容所有实现的代码，因此我们将通过不是从`list`而是从`UserList`扩展来修复它：

```py
from collections import UserList

class GoodList(UserList):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        if index % 2 == 0:
            prefix = "even"
        else:
            prefix = "odd"
        return f"[{prefix}] {value}"
```

现在事情看起来好多了：

```py
>>> gl = GoodList((0, 1, 2))
>>> gl[0]
'[even] 0'
>>> gl[1]
'[odd] 1'
>>> "; ".join(gl)
'[even] 0; [odd] 1; [even] 2'
```

不要直接从 dict 扩展，而是使用`collections.UserDict`。对于列表，使用`collections.UserList`，对于字符串，使用`collections.UserString`。

# 总结

在本章中，我们已经探讨了 Python 的主要特性，目标是理解其最独特的特性，这些特性使 Python 成为与其他语言相比独特的语言。在这条道路上，我们探索了 Python 的不同方法、协议和它们的内部机制。

与上一章相反，这一章更加关注 Python。本书主题的一个关键要点是，清晰的代码不仅仅是遵循格式规则（当然，这对于良好的代码库是必不可少的）。这是一个必要条件，但不是充分条件。在接下来的几章中，我们将看到更多与代码相关的想法和原则，旨在实现更好的软件解决方案设计和实现。

通过本章的概念和想法，我们探索了 Python 的核心：其协议和魔术方法。现在应该清楚了，编写 Pythonic、惯用的代码的最佳方式不仅仅是遵循格式约定，还要充分利用 Python 提供的所有功能。这意味着有时您应该使用特定的魔术方法，实现上下文管理器等。

在下一章中，我们将把这些概念付诸实践，将软件工程的一般概念与它们在 Python 中的书写方式联系起来。

# 参考资料

读者将在以下参考资料中找到更多关于本章涵盖的一些主题的信息。 Python 中索引如何工作的决定是基于（EWD831），该文分析了数学和编程语言中范围的几种替代方案：

+   *EWD831*：为什么编号应该从零开始（[`www.cs.utexas.edu/users/EWD/transcriptions/EWD08xx/EWD831.html`](https://www.cs.utexas.edu/users/EWD/transcriptions/EWD08xx/EWD831.html)）

+   *PEP-343*： "with"语句（[`www.python.org/dev/peps/pep-0343/`](https://www.python.org/dev/peps/pep-0343/)）

+   *CC08*：由 Robert C. Martin 撰写的书籍*Clean Code: A Handbook of Agile Software Craftsmanship*

+   Python 文档，`iter()`函数（[`docs.python.org/3/library/functions.html#iter`](https://docs.python.org/3/library/functions.html#iter)）

+   PyPy 和 CPython 之间的区别（[`pypy.readthedocs.io/en/latest/cpython_differences.html#subclasses-of-built-in-types`](https://pypy.readthedocs.io/en/latest/cpython_differences.html#subclasses-of-built-in-types)）
