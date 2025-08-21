## 第十章：文件和资源管理

读写文件是许多现实世界程序的关键部分。然而，*文件*的概念有点抽象。在某些情况下，文件可能意味着硬盘上的一系列字节；在其他情况下，它可能意味着例如远程系统上的 HTTP 资源。这两个实体共享一些行为。例如，您可以从每个实体中读取一系列字节。同时，它们并不相同。例如，您通常可以将字节写回本地文件，而无法对 HTTP 资源进行这样的操作。

在本章中，我们将看一下 Python 对文件的基本支持。由于处理本地文件既常见又重要，我们将主要关注与它们一起工作。但请注意，Python 及其库生态系统为许多其他类型的实体提供了类似*文件*的 API，包括基于 URI 的资源、数据库和许多其他数据源。这种使用通用 API 非常方便，使得可以编写可以在各种数据源上工作而无需更改的代码变得容易。

在本章中，我们还将看一下*上下文管理器*，这是 Python 管理资源的主要手段之一。上下文管理器允许您编写在发生异常时健壮且可预测的代码，确保资源（如文件）在发生错误时被正确关闭和处理。

### 文件

要在 Python 中打开本地文件，我们调用内置的`open()`函数。这需要一些参数，但最常用的是：

+   `file`：文件的路径。*这是必需的*。

+   `mode`：读取、写入、追加和二进制或文本。这是可选的，但我们建议始终明确指定以便清晰。显式优于隐式。

+   `encoding`：如果文件包含编码的文本数据，要使用哪种编码。通常最好指定这一点。如果不指定，Python 将为您选择默认编码。

#### 二进制和文本模式

在文件系统级别，当然，文件只包含一系列字节。然而，Python 区分以二进制和文本模式打开的文件，即使底层操作系统没有这样做。当您以二进制模式打开文件时，您正在指示 Python 使用文件中的数据而不进行任何解码；二进制模式文件反映了文件中的原始数据。

另一方面，以文本模式打开的文件将其内容视为包含`str`类型的文本字符串。当您从文本模式文件中获取数据时，Python 首先使用平台相关的编码或者`open()`的`encoding`参数对原始字节进行解码。

默认情况下，文本模式文件还支持 Python 的*通用换行符*。这会导致我们程序字符串中的单个可移植换行符(`'\n'`)与文件系统中存储的原始字节中的平台相关换行表示（例如 Windows 上的回车换行(`'\r\n'`)）之间的转换。

#### 编码的重要性

正确编码对于正确解释文本文件的内容至关重要，因此我们希望重点强调一下。Python^(24)无法可靠地确定文本文件的编码，因此不会尝试。然而，如果不知道文件的编码，Python 就无法正确操作文件中的数据。这就是为什么告诉 Python 要使用哪种编码非常重要。

如果您不指定编码，Python 将使用`sys.getdefaultencoding()`中的默认编码。在我们的情况下，默认编码是`'utf-8'`：

```py
>>> import sys
>>> sys.getdefaultencoding()
'utf-8'

```

但请记住，您的系统上的默认编码与您希望交换文件的另一个系统上的默认编码可能不同。最好是为了所有相关方都明确决定文本到字节的编码，通过在对`open()`的调用中指定它。您可以在[Python 文档](https://docs.python.org/3/library/codecs.html#standard-encodings)中获取支持的文本编码列表。

#### 打开文件进行写入

让我们通过以*写入*模式打开文件来开始处理文件。我们将明确使用 UTF-8 编码，因为我们无法知道您的默认编码是什么。我们还将使用关键字参数使事情更加清晰：

```py
>>> f = open('wasteland.txt', mode='wt', encoding='utf-8')

```

第一个参数是文件名。`mode`参数是一个包含不同含义字母的字符串。在这种情况下，‘w’表示*写入*，‘t’表示*文本*。

所有模式字符串应该由*读取*、*写入*或*追加*模式中的一个组成。此表列出了模式代码以及它们的含义：

| 代码 | 意义 |
| --- | --- |
| `r` | 以读取模式打开文件。流定位在 |
|   | 文件的开头。这是默认设置。 |
| `r+` | 用于读取和写入。流定位在 |
|   | 文件的开头。 |
| `w` | 截断文件至零长度或创建文件以进行写入。 |
|   | 流定位在文件的开头。 |
| `w+` | 用于读取和写入。如果文件不存在，则创建 |
|   | 存在，则截断。流定位在 |
|   | 文件的开头。 |
| `a` | 用于写入。如果文件不存在，则创建 |
|   | 流定位在文件的末尾。后续写入 |
|   | 文件的写入将始终结束在文件的当前末尾 |
|   | 无论有任何寻址或类似。 |
| `a+` | 用于读取和写入。如果文件不存在，则创建文件 |
|   | 存在。流定位在文件的末尾。 |
|   | 对文件的后续写入将始终结束在文件的当前末尾 |
|   | 无论有任何寻址或 |
|   | 类似。 |

前面的内容之一应与下表中的选择器结合使用，以指定*文本*或*二进制*模式：

| 代码 | 意义 |
| --- | --- |
| `t` | 文件内容被解释为编码文本字符串。从文件中接受和返回 |
|   | 文件将根据指定的文本编码进行编码和解码，并进行通用换行符转换 |
|   | 指定的文本编码，并且通用换行符转换将 |
|   | 生效（除非明确禁用）。所有写入方法 |
|   | `str`对象。 |
|   | *这是默认设置*。 |
| `b` | 文件内容被视为原始字节。所有写入方法 |
|   | 从文件中接受和返回`bytes`对象。 |

典型模式字符串的示例可能是`'wb'`表示“写入二进制”，或者`'at'`表示“追加文本”。虽然模式代码的两部分都支持默认设置，但为了可读性起见，我们建议明确指定。

`open()`返回的对象的确切类型取决于文件的打开方式。这就是动态类型的作用！然而，对于大多数目的来说，`open()`返回的实际类型并不重要。知道返回的对象是*类似文件的对象*就足够了，因此我们可以期望它支持某些属性和方法。

#### 向文件写入

我们之前已经展示了如何请求模块、方法和类型的`help()`，但实际上我们也可以请求实例的帮助。当你记住*一切*都是对象时，这是有意义的。

```py
>>> help(f)
. . .
 |  write(self, text, /)
 |      Write string to stream.
 |      Returns the number of characters written (which is always equal to
 |      the length of the string).
. . .

```

浏览帮助文档，我们可以看到`f`支持`write()`方法。使用‘q’退出帮助，并在 REPL 中继续。

现在让我们使用`write()`方法向文件写入一些文本：

```py
>>> f.write('What are the roots that clutch, ')
32

```

对`write()`的调用返回写入文件的代码点或字符数。让我们再添加几行：

```py
>>> f.write('what branches grow\n')
19
>>> f.write('Out of this stony rubbish? ')
27

```

你会注意到我们在写入文件时明确包括换行符。调用者有责任在需要时提供换行符；Python 不提供`writeline()`方法。

#### 关闭文件

当我们完成写入后，应该记得通过调用`close()`方法关闭文件：

```py
>>> f.close()

```

请注意，只有在关闭文件后，我们才能确保我们写入的数据对外部进程可见。关闭文件很重要！

还要记住，关闭文件后就不能再从文件中读取或写入。这样做会导致异常。

#### Python 之外的文件

如果现在退出 REPL，并查看你的文件系统，你会看到你确实创建了一个文件。在 Unix 上使用`ls`命令：

```py
$ ls -l
-rw-r--r--   1 rjs  staff    78 12 Jul 11:21 wasteland.txt

```

你应该看到`wasteland.txt`文件大小为 78 字节。

在 Windows 上使用`dir`：

```py
> dir
 Volume is drive C has no label.
 Volume Serial Number is 36C2-FF83

 Directory of c:\Users\pyfund

12/07/2013  20:54                79 wasteland.txt
 1 File(s)             79 bytes
 0 Dir(s)  190,353,698,816 bytes free

```

在这种情况下，你应该看到`wasteland.txt`大小为 79 字节，因为 Python 对文件的通用换行行为已经将行尾转换为你平台的本地行尾。

`write()`方法返回的数字是传递给`write()`的字符串中的码点（或字符）的数量，而不是编码和通用换行符转换后写入文件的字节数。通常情况下，在处理文本文件时，你不能通过`write()`返回的数量之和来确定文件的字节长度。

#### 读取文件

要读取文件，我们再次使用`open()`，但这次我们以`'rt'`作为模式，表示*读取文本*：

```py
>>> g = open('wasteland.txt', mode='rt', encoding='utf-8')

```

如果我们知道要读取多少字节，或者想要读取整个文件，我们可以使用`read()`。回顾我们的 REPL，我们可以看到第一次写入是 32 个字符长，所以让我们用`read()`方法读取回来：

```py
>>> g.read(32)
'What are the roots that clutch, '

```

在文本模式下，`read()`方法接受要从文件中读取的*字符*数，而不是字节数。调用返回文本并将文件指针移动到所读取内容的末尾。因为我们以文本模式打开文件，返回类型是`str`。

要读取文件中*所有*剩余的数据，我们可以调用`read()`而不带参数：

```py
>>> g.read()
'what branches grow\nOut of this stony rubbish? '

```

这给我们一个字符串中的两行部分 —— 注意中间的换行符。

在文件末尾，进一步调用`read()`会返回一个空字符串：

```py
>>> g.read()
''

```

通常情况下，当我们完成读取文件时，会使用`close()`关闭文件。不过，为了本练习的目的，我们将保持文件处于打开状态，并使用参数为零的`seek()`将文件指针移回文件的开头：

```py
>>> g.seek(0)
0

```

`seek()`的返回值是新的文件指针位置。

##### 逐行读取

对于文本使用`read()`相当麻烦，幸运的是 Python 提供了更好的工具来逐行读取文本文件。其中第一个就是`readline()`函数：

```py
>>> g.readline()
'What are the roots that clutch, what branches grow\n'
>>> g.readline()
'Out of this stony rubbish? '

```

每次调用`readline()`都会返回一行文本。如果文件中存在换行符，返回的行将以单个换行符结尾。

这里的最后一行没有以换行符结尾，因为文件末尾没有换行序列。你不应该*依赖*于`readline()`返回的字符串以换行符结尾。还要记住，通用换行符支持会将平台本地的换行序列转换为`'\n'`。

一旦我们到达文件末尾，进一步调用`readline()`会返回一个空字符串：

```py
>>> g.readline()
''

```

##### 一次读取多行

让我们再次将文件指针倒回并以不同的方式读取文件：

```py
>>> g.seek(0)

```

有时，当我们知道我们想要读取文件中的每一行时 —— 并且如果我们确信有足够的内存来这样做 —— 我们可以使用`readlines()`方法将文件中的所有行读入列表中：

```py
>>> g.readlines()
['What are the roots that clutch, what branches grow\n',
'Out of this stony rubbish? ']

```

如果解析文件涉及在行之间来回跳转，这将特别有用；使用行列表比使用字符流更容易。

这次，在继续之前我们会关闭文件：

```py
>>> g.close()

```

#### 追加到文件

有时我们希望追加到现有文件中，我们可以通过使用模式`'a'`来实现。在这种模式下，文件被打开以进行写入，并且文件指针被移动到任何现有数据的末尾。在这个例子中，我们将`'a'`与`'t'`结合在一起，以明确使用文本模式：

```py
>>> h = open('wasteland.txt', mode='at', encoding='utf-8')

```

虽然 Python 中没有`writeline()`方法，但有一个`writelines()`方法，它可以将可迭代的字符串系列写入流。如果您希望在字符串上有行结束符*，则必须自己提供。这乍一看可能有点奇怪，但它保持了与`readlines()`的对称性，同时也为我们使用`writelines()`将任何可迭代的字符串系列写入文件提供了灵活性：

```py
>>> h.writelines(
... ['Son of man,\n',
... 'You cannot say, or guess, ',
... 'for you know only,\n',
... 'A heap of broken images, ',
... 'where the sun beats\n'])
>>> h.close()

```

请注意，这里只完成了三行——我们说*完成*，因为我们追加的文件本身没有以换行符结束。

#### 文件对象作为迭代器

这些越来越复杂的文本文件读取工具的顶点在于文件对象支持*迭代器*协议。当您在文件上进行迭代时，每次迭代都会产生文件中的下一行。这意味着它们可以在 for 循环和任何其他可以使用迭代器的地方使用。

此时，我们有机会创建一个 Python 模块文件`files.py`：

```py
import sys

def main(filename):
    f = open(filename, mode='rt', encoding='utf-8')
    for line in f:
        print(line)
    f.close()

if __name__ == '__main__':
    main(sys.argv[1])

```

我们可以直接从系统命令行调用它，传递我们的文本文件的名称：

```py
$ python3 files.py wasteland.txt
What are the roots that clutch, what branches grow

Out of this stony rubbish? Son of man,

You cannot say, or guess, for you know only

A heap of broken images, where the sun beats

```

您会注意到诗歌的每一行之间都有空行。这是因为文件中的每一行都以换行符结尾，然后`print()`添加了自己的换行符。

为了解决这个问题，我们可以使用`strip()`方法在打印之前删除每行末尾的空白。相反，我们将使用`stdout`流的`write()`方法。这与我们之前用来写入文件的`write()`方法*完全*相同，因为`stdout`流本身就是一个类似文件的对象，所以可以使用它。

我们从`sys`模块中获得了对`stdout`流的引用：

```py
import sys

def main(filename):
    f = open(filename, mode='rt', encoding='utf-8')
    for line in f:
        sys.stdout.write(line)
    f.close()

if __name__ == '__main__':
    main(sys.argv[1])

```

如果我们重新运行我们的程序，我们会得到：

```py
$ python3 files.py wasteland.txt
What are the roots that clutch, what branches grow
Out of this stony rubbish? Son of man,
You cannot say, or guess, for you know only
A heap of broken images, where the sun beats

```

现在，不幸的是，是时候离开二十世纪最重要的诗歌之一，开始着手处理*几乎*同样令人兴奋的东西，上下文管理器。

### 上下文管理器

对于接下来的一组示例，我们将需要一个包含一些数字的数据文件。使用下面的`recaman.py`中的代码，我们将一个名为[Recaman 序列](http://mathworld.wolfram.com/RecamansSequence.html)的数字序列写入文本文件，每行一个数字：

```py
import sys
from itertools import count, islice

def sequence():
    """Generate Recaman's sequence."""
    seen = set()
    a = 0
    for n in count(1):
        yield a
        seen.add(a)
        c = a - n
        if c < 0 or c in seen:
            c = a + n
        a = c

def write_sequence(filename, num):
    """Write Recaman's sequence to a text file."""
    f = open(filename, mode='wt', encoding='utf-8')
    f.writelines("{0}\n".format(r)
                 for r in islice(sequence(), num + 1))
    f.close()

if __name__ == '__main__':
    write_sequence(filename=sys.argv[1],
                   num=int(sys.argv[2]))

```

Recaman 序列本身对这个练习并不重要；我们只需要一种生成数字数据的方法。因此，我们不会解释`sequence()`生成器。不过，随意进行实验。

该模块包含一个用于产生 Recaman 数的生成器，以及一个使用`writelines()`方法将序列的开头写入文件的函数。生成器表达式用于将每个数字转换为字符串并添加换行符。`itertools.islice()`用于截断否则无限的序列。

通过执行模块，将文件名和序列长度作为命令行参数传递，我们将前 1000 个 Recaman 数写入文件：

```py
$ python3 recaman.py recaman.dat 1000

```

现在让我们创建一个补充模块`series.py`，它可以重新读取这个数据文件：

```py
"""Read and print an integer series."""

import sys

def read_series(filename):
    f = open(filename, mode='rt', encoding='utf-8')
    series = []
    for line in f:
        a = int(line.strip())
        series.append(a)
    f.close()
    return series

def main(filename):
    series = read_series(filename)
    print(series)

if __name__ == '__main__':
    main(sys.argv[1])

```

我们从打开的文件中一次读取一行，使用`strip()`字符串方法去除换行符，并将其转换为整数。如果我们从命令行运行它，一切都应该如预期般工作：

```py
$ python3 series.py recaman.dat
[0, 1, 3, 6, 2, 7, 13,
 ...
,3683, 2688, 3684, 2687, 3685, 2686, 3686]

```

现在让我们故意制造一个异常情况。在文本编辑器中打开`recaman.dat`，并用不是字符串化整数的内容替换其中一个数字：

```py
0
1
3
6
2
7
13
oops!
12
21

```

保存文件，然后重新运行`series.py`：

```py
$ python3 series.py recaman.dat
Traceback (most recent call last):
  File "series.py", line 19, in <module>
    main(sys.argv[1])
  File "series.py", line 15, in main
    series = read_series(filename)
  File "series.py", line 9, in read_series
    a = int(line.strip())
ValueError: invalid literal for int() with base 10: 'oops!'

```

当传递我们的新的无效行时，`int()`构造函数会引发`ValueError`。异常未处理，因此程序以堆栈跟踪终止。

#### 使用`finally`管理资源

这里的一个问题是我们的`f.close()`调用从未执行过。

为了解决这个问题，我们可以插入一个`try`..`finally`块：

```py
def read_series(filename):
    try:
        f = open(filename, mode='rt', encoding='utf-8')
        series = []
        for line in f:
            a = int(line.strip())
            series.append(a)
    finally:
        f.close()
    return series

```

现在文件将始终关闭，即使存在异常。进行这种更改开启了另一种重构的机会：我们可以用列表推导来替换 for 循环，并直接返回这个列表：

```py
def read_series(filename):
    try:
        f = open(filename, mode='rt', encoding='utf-8')
        return [ int(line.strip()) for line in f ]
    finally:
        f.close()

```

即使在这种情况下，`close()`仍然会被调用；无论`try`块如何退出，`finally`块都会被调用。

#### with-blocks

到目前为止，我们的例子都遵循一个模式：`open()`一个文件，处理文件，`close()`文件。`close()`很重要，因为它通知底层操作系统你已经完成了对文件的操作。如果你在完成文件操作后不关闭文件，可能会丢失数据。可能会有待写入的缓冲区，可能不会完全写入。此外，如果你打开了很多文件，你的系统可能会耗尽资源。由于我们总是希望每个`open()`都与一个`close()`配对，我们希望有一个机制，即使我们忘记了，也能强制执行这种关系。

这种资源清理的需求是很常见的，Python 实现了一个特定的控制流结构，称为*with-blocks*来支持它。with-blocks 可以与支持*上下文管理器*协议的任何对象一起使用，这包括`open()`返回的文件对象。利用文件对象是上下文管理器的事实，我们的`read_series()`函数可以变成：

```py
def read_series(filename):
    with open(filename, mode='rt', encoding='utf-8') as f:
        return [int(line.strip()) for line in f]

```

我们不再需要显式调用`close()`，因为`with`结构将在执行退出块时为我们调用它，无论我们如何退出块。

现在我们可以回去修改我们的 Recaman 系列写作程序，也使用一个 with-block，再次消除了显式的`close()`的需要：

```py
def write_sequence(filename, num):
    """Write Recaman's sequence to a text file."""
    with open(filename, mode='wt', encoding='utf-8') as f:
        f.writelines("{0}\n".format(r)
                     for r in islice(sequence(), num + 1))

```

* * *

### 禅的时刻

![](img/zen-beautiful-is-better-than-ugly.png)

with-block 的语法如下：

```py
with EXPR as VAR:
    BLOCK

```

这被称为*语法糖*，用于更复杂的`try...except`和`try...finally`块的安排：

```py
mgr = (EXPR)
exit = type(mgr).__exit__  # Not calling it yet
value = type(mgr).__enter__(mgr)
exc = True
try:
    try:
        VAR = value  # Only if "as VAR" is present
        BLOCK
    except:
        # The exceptional case is handled here
        exc = False
        if not exit(mgr, *sys.exc_info()):
            raise
        # The exception is swallowed if exit() returns true
finally:
    # The normal and non-local-goto cases are handled here
    if exc:
        exit(mgr, None, None, None)

```

^(25)

你更喜欢哪个？

我们中很少有人希望我们的代码看起来如此复杂，但这就是没有`with`语句的情况下它需要看起来的样子。糖可能对你的健康不好，但对你的代码可能非常有益！

* * *

### 二进制文件

到目前为止，我们已经看过文本文件，其中我们将文件内容处理为 Unicode 字符串。然而，有许多情况下，文件包含的数据并不是编码文本。在这些情况下，我们需要能够直接处理文件中存在的确切字节，而不需要任何中间编码或解码。这就是*二进制模式*的用途。

#### BMP 文件格式

为了演示处理二进制文件，我们需要一个有趣的二进制数据格式。BMP 是一种包含设备无关位图的图像文件格式。它足够简单，我们可以从头开始制作一个 BMP 文件写入器。^(26)将以下代码放入一个名为`bmp.py`的模块中：

```py
 1 # bmp.py
 2 
 3 """A module for dealing with BMP bitmap image files."""
 4 
 5 
 6 def write_grayscale(filename, pixels):
 7    """Creates and writes a grayscale BMP file.
 8 
 9    Args:
10         filename: The name of the BMP file to me created.
11 
12         pixels: A rectangular image stored as a sequence of rows.
13             Each row must be an iterable series of integers in the
14             range 0-255.
15 
16     Raises:
17         OSError: If the file couldn't be written.
18     """
19     height = len(pixels)
20     width = len(pixels[0])
21 
22     with open(filename, 'wb') as bmp:
23         # BMP Header
24         bmp.write(b'BM')
25 
26         # The next four bytes hold the filesize as a 32-bit
27         # little-endian integer. Zero placeholder for now.
28         size_bookmark = bmp.tell()
29         bmp.write(b'\x00\x00\x00\x00')
30 
31         # Two unused 16-bit integers - should be zero
32         bmp.write(b'\x00\x00')
33         bmp.write(b'\x00\x00')
34 
35         # The next four bytes hold the integer offset
36         # to the pixel data. Zero placeholder for now.
37         pixel_offset_bookmark = bmp.tell()
38         bmp.write(b'\x00\x00\x00\x00')
39 
40         # Image Header
41         bmp.write(b'\x28\x00\x00\x00')  # Image header size in bytes - 40 decimal
42         bmp.write(_int32_to_bytes(width))   # Image width in pixels
43         bmp.write(_int32_to_bytes(height))  # Image height in pixels
44         # Rest of header is essentially fixed
45         bmp.write(b'\x01\x00')          # Number of image planes
46         bmp.write(b'\x08\x00')          # Bits per pixel 8 for grayscale
47         bmp.write(b'\x00\x00\x00\x00')  # No compression
48         bmp.write(b'\x00\x00\x00\x00')  # Zero for uncompressed images
49         bmp.write(b'\x00\x00\x00\x00')  # Unused pixels per meter
50         bmp.write(b'\x00\x00\x00\x00')  # Unused pixels per meter
51         bmp.write(b'\x00\x00\x00\x00')  # Use whole color table
52         bmp.write(b'\x00\x00\x00\x00')  # All colors are important
53 
54         # Color palette - a linear grayscale
55         for c in range(256):
56             bmp.write(bytes((c, c, c, 0)))  # Blue, Green, Red, Zero
57 
58         # Pixel data
59         pixel_data_bookmark = bmp.tell()
60         for row in reversed(pixels):  # BMP files are bottom to top
61             row_data = bytes(row)
62             bmp.write(row_data)
63             padding = b'\x00' * ((4 - (len(row) % 4)) % 4)  # Pad row to multiple
64                                                             # of four bytes
65             bmp.write(padding)
66 
67         # End of file
68         eof_bookmark = bmp.tell()
69 
70         # Fill in file size placeholder
71         bmp.seek(size_bookmark)
72         bmp.write(_int32_to_bytes(eof_bookmark))
73 
74         # Fill in pixel offset placeholder
75         bmp.seek(pixel_offset_bookmark)
76         bmp.write(_int32_to_bytes(pixel_data_bookmark))

```

这可能看起来很复杂，但你会发现它相对简单。

为了简单起见，我们决定只处理 8 位灰度图像。这些图像有一个很好的特性，即每个像素一个字节。`write_grayscale()`函数接受两个参数：文件名和像素值的集合。正如文档字符串所指出的那样，这个集合应该是整数序列的序列。例如，一个`int`对象的列表列表就可以了。此外：

+   每个`int`必须是从 0 到 255 的像素值

+   每个内部列表都是从左到右的像素行

+   外部列表是从上到下的像素行的列表。

我们要做的第一件事是通过计算行数（第 19 行）来确定图像的大小，以给出高度，并计算零行中的项目数来获得宽度（第 20 行）。我们假设，但不检查，所有行的长度都相同（在生产代码中，这是我们想要进行检查的）。

接下来，我们使用`'wb'`模式字符串在*二进制写入*模式下`open()`（第 22 行）文件。我们不指定编码 - 这对于原始二进制文件没有意义。

在 with 块内，我们开始编写所谓的“BMP 头”，这是 BMP 格式的开始。

头部必须以所谓的“魔术”字节序列`b'BM'`开头，以识别它为 BMP 文件。我们使用`write()`方法（第 24 行），因为文件是以二进制模式打开的，所以我们必须传递一个`bytes`对象。

接下来的四个字节应该包含一个 32 位整数，其中包含文件大小，这是我们目前还不知道的值。我们本可以提前计算它，但我们将采取不同的方法：我们将写入一个占位符值，然后返回到这一点以填写细节。为了能够回到这一点，我们使用文件对象的`tell()`方法（第 28 行）；这给了我们文件指针从文件开头的偏移量。我们将把这个偏移量存储在一个变量中，它将充当一种书签。我们写入四个零字节作为占位符（第 29 行），使用转义语法来指定这些零。

接下来的两对字节是未使用的，所以我们也将零字节写入它们（第 32 和 33 行）。

接下来的四个字节是另一个 32 位整数，应该包含从文件开头到像素数据开始的偏移量（以字节为单位）。我们也不知道这个值，所以我们将使用`tell()`（第 37 行）存储另一个书签，并写入另外四个字节的占位符（第 38 行）；当我们知道更多信息时，我们将很快返回到这里。

接下来的部分称为“图像头”。我们首先要做的是将图像头的长度写入一个 32 位整数（第 41 行）。在我们的情况下，头部总是 40 个字节长。我们只需将其硬编码为十六进制。注意 BMP 格式是小端序的 - 最不重要的字节先写入。

接下来的四个字节是图像宽度，作为小端序的 32 位整数。我们在这里调用一个模块范围的实现细节函数，名为`_int32_to_bytes()`，它将一个`int`对象转换为一个包含恰好四个字节的`bytes`对象（第 42 行）。然后我们再次使用相同的函数来处理图像高度（第 43 行）。

头部的其余部分对于 8 位灰度图像基本上是固定的，这里的细节并不重要，除了要注意整个头部实际上总共是 40 个字节（第 45 行）。

8 位 BMP 图像中的每个像素都是颜色表中 256 个条目的索引。每个条目都是一个四字节的 BGR 颜色。对于灰度图像，我们需要按线性比例写入 256 个 4 字节的灰度值（第 54 行）。这段代码是实验的肥沃土壤，这个函数的一个自然增强功能将是能够单独提供这个调色板作为可选的函数参数。

最后，我们准备写入像素数据，但在这之前，我们要使用`tell()`（第 59 行）方法记录当前文件指针的偏移量，因为这是我们需要稍后返回并填写的位置之一。

写入像素数据本身是相当简单的。我们使用内置函数`reversed()`（第 60 行）来翻转行的顺序；BMP 图像是从底部向顶部写入的。对于每一行，我们将整数的可迭代系列传递给`bytes()`构造函数（第 61 行）。如果任何整数超出了 0-255 的范围，构造函数将引发`ValueError`。

BMP 文件中的每一行像素数据必须是四个字节的整数倍长，与图像宽度无关。为了做到这一点（第 63 行），我们取行长度模四，得到一个介于零和三之间的数字，这是我们行末尾距离*前一个*四字节边界的字节数。为了得到填充字节数，使我们达到*下一个*四字节边界，我们从四中减去这个模数值，得到一个介于 4 到 1 之间的值。然而，我们永远不希望用四个字节填充，只用一、二或三个，所以我们必须再次取模四，将四字节填充转换为零字节填充。

这个值与重复操作符应用于单个零字节一起使用，以产生一个包含零、一个、两个或三个字节的字节对象。我们将这些写入文件，以终止每一行（第 65 行）。

在像素数据之后，我们已经到达了文件的末尾。我们之前承诺记录了这个偏移值，所以我们使用`tell()`（第 68 行）将当前位置记录到一个文件末尾书签变量中。

现在我们可以回来实现我们的承诺，通过用我们记录的真实偏移量替换占位符。首先是文件长度。为此，我们`seek()`（第 71 行）回到我们在文件开头附近记住的`size_bookmark`，并使用我们的`_int32_to_bytes()`函数将存储在`eof_bookmark`中的大小作为小端 32 位整数`write()`（第 72 行）。

最后，我们`seek()`（第 75 行）到由`pixel_offset_bookmark`标记的像素数据偏移量的位置，并将存储在`pixel_data_bookmark`中的 32 位整数（第 76 行）写入。

当我们退出 with 块时，我们可以放心，上下文管理器将关闭文件并将任何缓冲写入文件系统。

#### 位运算符

处理二进制文件通常需要在字节级别拆分或组装数据。这正是我们的`_int32_to_bytes()`函数在做的事情。我们将快速查看它，因为它展示了一些我们以前没有见过的 Python 特性：

```py
def _int32_to_bytes(i):
    """Convert an integer to four bytes in little-endian format."""
    return bytes((i & 0xff,
                  i >> 8 & 0xff,
                  i >> 16 & 0xff,
                  i >> 24 & 0xff))

```

该函数使用`>>`（*位移*）和`&`（*按位与*）运算符从整数值中提取单个字节。请注意，按位与使用和符号来区分它与*逻辑与*，后者是拼写出来的单词“and”。`>>`运算符将整数的二进制表示向右移动指定的位数。该例程在每次移位后使用`&`提取最低有效字节。得到的四个整数用于构造一个元组，然后传递给`bytes()`构造函数以产生一个四字节序列。

#### 写一个 BMP 文件

为了生成一个 BMP 图像文件，我们需要一些像素数据。我们包含了一个简单的模块`fractal.py`，它为标志性的[Mandelbrot 集合分形](https://en.wikipedia.org/wiki/Mandelbrot_set)生成像素值。我们不打算详细解释分形生成代码，更不用说背后的数学。但这段代码足够简单，而且不依赖于我们以前遇到的任何 Python 特性：

```py
# fractal.py

"""Computing Mandelbrot sets."""

import math

def mandel(real, imag):
    """The logarithm of number of iterations needed to
 determine whether a complex point is in the
 Mandelbrot set.

 Args:
 real: The real coordinate
 imag: The imaginary coordinate

 Returns:
 An integer in the range 1-255.
 """
    x = 0
    y = 0
    for i in range(1, 257):
        if x*x + y*y > 4.0:
            break
        xt = real + x*x - y*y
        y = imag + 2.0 * x * y
        x = xt
    return int(math.log(i) * 256 / math.log(256)) - 1

def mandelbrot(size_x, size_y):
    """Make an Mandelbrot set image.

 Args:
 size_x: Image width
 size_y: Image height

 Returns:
 A list of lists of integers in the range 0-255.
 """
    return [ [mandel((3.5 * x / size_x) - 2.5,
                     (2.0 * y / size_y) - 1.0)
              for x in range(size_x) ]
            for y in range(size_y) ]

```

关键是`mandelbrot()`函数使用嵌套的列表推导来生成一个范围在 0-255 的整数列表的列表。这个列表代表了分形的图像。每个点的整数值是由`mandel()`函数产生的。

##### 生成分形图像

让我们启动一个 REPL，并将`fractal`和`bmp`模块一起使用。首先，我们使用`mandelbrot()`函数生成一个 448x256 像素的图像。使用长宽比为 7:4 的图像会获得最佳结果：

```py
>>> import fractal
>>> pixels = fractal.mandelbrot(448, 256)

```

这个对`mandelbrot()`的调用可能需要一秒左右 - 我们的分形生成器简单而不是高效！

我们可以查看返回的数据结构：

```py
>>> pixels
[[31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
  31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
  ...
  49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49]]

```

这是一个整数列表的列表，就像我们所承诺的那样。让我们把这些像素值写入一个 BMP 文件：

```py
>>> import bmp
>>> bmp.write_grayscale("mandel.bmp", pixels)

```

找到文件并在图像查看器中打开它，例如通过在 Web 浏览器中打开它。

![](img/mandel.png)

#### 读取二进制文件

现在我们正在生成美丽的 Mandelbrot 图像，我们应该尝试用 Python 读取这些 BMP 文件。我们不打算编写一个完整的 BMP 阅读器，尽管那将是一个有趣的练习。我们只是制作一个简单的函数来确定 BMP 文件中的像素维度。我们将把代码添加到`bmp.py`中：

```py
def dimensions(filename):
    """Determine the dimensions in pixels of a BMP image.

 Args:
 filename: The filename of a BMP file.

 Returns:
 A tuple containing two integers with the width
 and height in pixels.

 Raises:
 ValueError: If the file was not a BMP file.
 OSError: If there was a problem reading the file.
 """

    with open(filename, 'rb') as f:
        magic = f.read(2)
        if magic != b'BM':
            raise ValueError("{} is not a BMP file".format(filename))

        f.seek(18)
        width_bytes = f.read(4)
        height_bytes = f.read(4)

        return (_bytes_to_int32(width_bytes),
                _bytes_to_int32(height_bytes))

```

当然，我们使用 with 语句来管理文件，所以我们不必担心它是否被正确关闭。在 with 块内，我们通过查找我们在 BMP 文件中期望的前两个魔术字节来执行简单的验证检查。如果不存在，我们会引发`ValueError`，这当然会导致上下文管理器关闭文件。

回顾一下我们的 BMP 写入器，我们可以确定图像尺寸恰好存储在文件开头的 18 个字节处。我们`seek()`到该位置，并使用`read()`方法读取两个四字节的块，分别代表尺寸的两个 32 位整数。因为我们以二进制模式打开文件，`read()`返回一个`bytes`对象。我们将这两个`bytes`对象传递给另一个实现细节函数`_bytes_to_int32()`，它将它们重新组装成一个整数。这两个整数，代表图像的宽度和高度，作为一个元组返回。

`_bytes_to_int32（）`函数使用`<<`（*按位左移*）和`|`（*按位或*），以及对`bytes`对象的索引，来重新组装整数。请注意，对`bytes`对象进行索引返回一个整数：

```py
def _bytes_to_int32(b):
    """Convert a bytes object containing four bytes into an integer."""
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

```

如果我们使用我们的新的读取器代码，我们可以看到它确实读取了正确的值：

```py
>>> bmp.dimensions("mandel.bmp")
(448, 256)

```

### 类似文件的对象

Python 中有一个“类似文件的对象”的概念。这并不像特定的协议^(27)那样正式，但由于鸭子类型所提供的多态性，在实践中它运作良好。

之所以没有严格规定它，是因为不同类型的数据流和设备具有许多不同的功能、期望和行为。因此，实际上定义一组模拟它们的协议将是相当复杂的，而且实际上并没有太多的实际意义，除了一种理论成就感。这就是 EAFP^(28)哲学的优势所在：如果你想在类似文件的对象上执行`seek()`，而事先不知道它是否支持随机访问，那就试试看（字面上！）。只是要做好准备，如果`seek()`方法不存在，或者*存在*但行为不符合你的期望，那么就会失败。

你可能会说“如果它看起来像一个文件，读起来像一个文件，那么它就是一个文件”。

#### 你已经看到了类似文件的对象！

我们已经看到了类似文件的对象的实际应用；当我们以文本和二进制模式打开文件时，返回给我们的对象实际上是不同类型的，尽管都具有明确定义的类似文件的行为。Python 标准库中还有其他类型实现了类似文件的行为，实际上我们在书的开头就看到了其中一个，当时我们使用`urlopen()`从互联网上的 URL 检索数据。

#### 使用类似文件的对象

让我们通过编写一个函数来利用类似文件的对象的多态性，来统计文件中每行的单词数，并将该信息作为列表返回：

```py
>>> def words_per_line(flo):
...    return [len(line.split()) for line in flo.readlines()]

```

现在我们将打开一个包含我们之前创建的 T.S.艾略特杰作片段的常规文本文件，并将其传递给我们的新函数：

```py
>>> with open("wasteland.txt", mode='rt', encoding='utf-8') as real_file:
...     wpl = words_per_line(real_file)
...
>>> wpl
[9, 8, 9, 9]

```

`real_file`的实际类型是：

```py
>>> type(real_file)
<class '_io.TextIOWrapper'>

```

但通常你不应该关心这个具体的类型；这是 Python 内部的实现细节。你只需要关心它的行为“像一个文件”。

现在我们将使用代表 URL 引用的 Web 资源的类似文件对象执行相同的操作：

```py
>>> from urllib.request import urlopen
>>> with urlopen("http://sixty-north.com/c/t.txt") as web_file:
...    wpl = words_per_line(web_file)
...
>>> wpl
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 7, 8, 14, 12, 8]

```

`web_file`的类型与我们刚刚看到的类型相当不同：

```py
>>> type(web_file)
<class 'http.client.HTTPResponse'>

```

然而，由于它们都是类似文件的对象，我们的函数可以与两者一起使用。

类似文件的对象并没有什么神奇之处；它只是一个方便且相当非正式的描述，用于描述我们可以对对象提出的一组期望，这些期望是通过鸭子类型来实现的。

### 其他资源

with 语句结构可以与实现上下文管理器协议的任何类型的对象一起使用。我们不会在本书中向您展示如何实现上下文管理器 - 为此，您需要参考*The Python Journeyman* - 但我们会向您展示一种简单的方法，使您自己的类可以在 with 语句中使用。将这段代码放入模块`fridge.py`中：

```py
# fridge.py

"""Demonstrate raiding a refrigerator."""

class RefrigeratorRaider:
    """Raid a refrigerator."""

    def open(self):
        print("Open fridge door.")

    def take(self, food):
        print("Finding {}...".format(food))
        if food == 'deep fried pizza':
            raise RuntimeError("Health warning!")
        print("Taking {}".format(food))

    def close(self):
        print("Close fridge door.")

def raid(food):
    r = RefrigeratorRaider()
    r.open()
    r.take(food)
    r.close()

```

我们将`raid()`导入 REPL 并开始肆虐：

```py
>>> from fridge import raid
>>> raid("bacon")
Open fridge door.
Finding bacon...
Taking bacon
Close fridge door.

```

重要的是，我们记得关闭了门，所以食物会保存到我们下次袭击。让我们尝试另一次袭击，找一些稍微不那么健康的东西：

```py
>>> raid("deep fried pizza")
Open fridge door.
Finding deep fried pizza...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "./fridge.py", line 23, in raid
    r.take(food)
  File "./fridge.py", line 14, in take
    raise RuntimeError("Health warning!")
RuntimeError: Health warning!

```

这次，我们被健康警告打断，没有来得及关闭门。我们可以通过使用 Python 标准库中的[`contextlib`模块](https://docs.python.org/3/library/contextlib.html)中的`closing()`函数来解决这个问题。导入函数后，我们将`RefrigeratorRaider`构造函数调用包装在`closing()`的调用中。这样可以将我们的对象包装在一个上下文管理器中，在退出之前始终调用包装对象上的`close()`方法。我们使用这个对象来初始化一个 with 块：

```py
"""Demonstrate raiding a refrigerator."""

from contextlib import closing

class RefrigeratorRaider:
    """Raid a refrigerator."""

    def open(self):
        print("Open fridge door.")

    def take(self, food):
        print("Finding {}...".format(food))
        if food == 'deep fried pizza':
            raise RuntimeError("Health warning!")
        print("Taking {}".format(food))

    def close(self):
        print("Close fridge door.")

def raid(food):
    with closing(RefrigeratorRaider()) as r:
        r.open()
        r.take(food)
        r.close()

```

现在当我们执行袭击时：

```py
>>> raid("spam")
Open fridge door.
Finding spam...
Taking spam
Close fridge door.
Close fridge door.

```

我们看到我们对`close()`的显式调用是不必要的，所以让我们来修复一下：

```py
def raid(food):
    with closing(RefrigeratorRaider()) as r:
        r.open()
        r.take(food)

```

更复杂的实现会检查门是否已经关闭，并忽略其他请求。

那么它是否有效呢？让我们再试试吃一些油炸比萨：

```py
>>> raid("deep fried pizza")
Open fridge door.
Finding deep fried pizza...
Close fridge door.
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "./fridge.py", line 23, in raid
    r.take(food)
  File "./fridge.py", line 14, in take
    raise RuntimeError("Health warning!")
RuntimeError: Health warning!

```

这一次，即使触发了健康警告，上下文管理器仍然为我们关闭了门。

### 总结

+   文件是使用内置的`open()`函数打开的，该函数接受文件模式来控制读取/写入/追加行为，以及文件是作为原始二进制数据还是编码文本数据进行处理。

+   对于文本数据，应指定文本编码。

+   文本文件处理字符串对象，并执行通用换行符转换和字符串编码。

+   二进制文件处理`bytes`对象，不进行换行符转换或编码。

+   在写文件时，您有责任为换行符提供换行字符。

+   文件在使用后应始终关闭。

+   文件提供各种面向行的方法进行读取，并且也是迭代器，逐行产生行。

+   文件是上下文管理器，可以与上下文管理器一起使用，以确保执行清理操作，例如关闭文件。

+   文件样对象的概念定义不严格，但在实践中非常有用。尽量使用 EAFP 来充分利用它们。

+   上下文管理器不仅限于类似文件的对象。我们可以使用`contextlib`标准库模块中的工具，例如`closing()`包装器来创建我们自己的上下文管理器。

沿途我们发现：

+   `help()`可以用于实例对象，而不仅仅是类型。

+   Python 支持按位运算符`&`、`|`、`<<`和`>>`。
