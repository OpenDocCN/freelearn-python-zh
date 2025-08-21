## 第四章：模块化

模块化对于除了微不足道的软件系统以外的任何东西都是一个重要的属性，因为它赋予我们能力去创建自包含、可重复使用的部分，这些部分可以以新的方式组合来解决不同的问题。在 Python 中，与大多数编程语言一样，最细粒度的模块化设施是可重复使用函数的定义。但是 Python 还给了我们几种其他强大的模块化机制。

相关函数的集合本身被组合在一起形成了一种称为*模块*的模块化形式。模块是可以被其他模块引用的源代码文件，允许在一个模块中定义的函数在另一个模块中被重用。只要你小心避免任何循环依赖，模块是组织程序的一种简单灵活的方式。

在之前的章节中，我们已经看到我们可以将模块导入 REPL。我们还将向您展示模块如何直接作为程序或脚本执行。作为这一部分的一部分，我们将调查 Python 执行模型，以确保您对代码何时被评估和执行有一个很好的理解。我们将通过向您展示如何使用命令行参数将基本配置数据传递到您的程序中并使您的程序可执行来结束本章。

为了说明本章，我们将从上一章末尾开发的从网络托管的文本文档中检索单词的代码片段开始。我们将通过将代码组织成一个完整的 Python 模块来详细说明该代码。

### 在一个.py 文件中组织代码

让我们从第二章中我们使用的代码片段开始。打开一个文本编辑器 - 最好是一个支持 Python 语法高亮的编辑器 - 并配置它在按下 tab 键时插入四个空格的缩进级别。你还应该检查你的编辑器是否使用 UTF 8 编码保存文件，因为这是 Python 3 运行时的默认设置。

在你的主目录下创建一个名为`pyfund`的目录。这是我们将放置本章代码的地方。

所有的 Python 源文件都使用`.py`扩展名，所以让我们把我们在 REPL 中写的片段放到一个名为`pyfund/words.py`的文本文件中。文件的内容应该是这样的：

```py
from urllib.request import urlopen

with urlopen('http://sixty-north.com/c/t.txt') as story:
    story_words = []
    for line in story:
        line_words = line.decode('utf-8').split()
        for word in line_words:
            story_words.append(word)

```

你会注意到上面的代码和我们之前在 REPL 中写的代码之间有一些细微的差异。现在我们正在使用一个文本文件来编写我们的代码，所以我们可以更加注意可读性，例如，在`import`语句后我们加了一个空行。

在继续之前保存这个文件。

#### 从操作系统 shell 运行 Python 程序

切换到带有操作系统 shell 提示符的控制台，并切换到新的`pyfund`目录：

```py
$ cd pyfund

```

我们可以通过调用 Python 并传递模块的文件名来执行我们的模块：

```py
$ python3 words.py

```

在 Mac 或 Linux 上，或者：

```py
> python words.py

```

在 Windows 上。

当你按下回车键后，经过短暂的延迟，你将返回到系统提示符。并不是很令人印象深刻，但如果你没有得到任何响应，那么程序正在按预期运行。另一方面，如果你看到一些错误，那么就有问题了。例如，`HTTPError`表示有网络问题，而其他类型的错误可能意味着你输入了错误的代码。

让我们在程序的末尾再添加一个 for 循环，每行打印一个单词。将这段代码添加到你的 Python 文件的末尾：

```py
for word in story_words:
    print(word)

```

如果你去命令提示符并再次执行代码，你应该会看到一些输出。现在我们有了一个有用程序的开端！

#### 将模块导入到 REPL 中

我们的模块也可以导入到 REPL 中。让我们试试看会发生什么。启动 REPL 并导入你的模块。当导入一个模块时，你使用`import <module-name>`，省略模块名称的`.py`扩展名。在我们的情况下，看起来是这样的：

```py
$ python
Python 3.5.0 (default, Nov  3 2015, 13:17:02)
[GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import words
It
was
the
best
of
times
. . .

```

当导入模块时，模块中的代码会立即执行！这也许不是你期望的，而且肯定不是很有用。为了更好地控制代码的执行时间，并允许其被重用，我们需要将代码放入一个函数中。

### 定义函数

使用`def`关键字定义函数，后面跟着函数名、括号中的参数列表和一个冒号来开始一个新的块。让我们在 REPL 中快速定义一些函数来了解一下：

```py
>>> def square(x):
...     return x * x
...

```

我们使用`return`关键字从函数中返回一个值。

正如我们之前所看到的，我们通过在函数名后的括号中提供实际参数来调用函数：

```py
>>> square(5)
5

```

函数并不需要显式返回一个值 - 也许它们会产生副作用：

```py
>>> def launch_missiles():
...     print("Missiles launched!")
...
>>> launch_missiles()
Missiles launched!

```

您可以使用`return`关键字而不带参数来提前从函数中返回：

```py
>>> def even_or_odd(n):
...     if n % 2 == 0:
...         print("even")
...         return
...     print("odd")
...
>>> even_or_odd(4)
even
>>> even_or_odd(5)
odd

```

如果函数中没有显式的`return`，Python 会在函数末尾隐式添加一个`return`。这个隐式的返回，或者没有参数的`return`，实际上会导致函数返回`None`。不过要记住，REPL 不会显示`None`结果，所以我们看不到它们。通过将返回的对象捕获到一个命名变量中，我们可以测试是否为`None`：

```py
>>> w = even_or_odd(31)
odd
>>> w is None
True

```

### 将我们的模块组织成函数

让我们使用函数来组织我们的 words 模块。

首先，我们将除了导入语句之外的所有代码移动到一个名为`fetch_words()`的函数中。您可以通过添加`def`语句并将其下面的代码缩进一级来实现这一点：

```py
from urllib.request import urlopen

def fetch_words():
    with urlopen('http://sixty-north.com/c/t.txt') as story:
        story_words = []
        for line in story:
            line_words = line.decode('utf-8').split()
            for word in line_words:
                story_words.append(word)

    for word in story_words:
        print(word)

```

保存模块，并使用新的 Python REPL 重新加载模块：

```py
$ python3
Python 3.5.0 (default, Nov  3 2015, 13:17:02)
[GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import words

```

模块已导入，但直到我们调用`fetch_words()`函数时，单词才会被获取：

```py
>>> words.fetch_words()
It
was
the
best
of
times

```

或者我们可以导入我们的特定函数：

```py
>>> from words import fetch_words
>>> fetch_words()
It
was
the
best
of
times

```

到目前为止一切都很好，但当我们尝试直接从操作系统 shell 运行我们的模块时会发生什么？

从 Mac 或 Linux 使用`Ctrl-D`退出 REPL，或者从 Windows 使用`Ctrl-Z`，然后运行 Python 3 并传递模块文件名：

```py
$ python3 words.py

```

没有单词被打印。这是因为现在模块所做的只是定义一个函数，然后立即退出。为了创建一个我们可以有用地从中导入函数到 REPL *并且*可以作为脚本运行的模块，我们需要学习一个新的 Python 习惯用法。

#### `__name__`和从命令行执行模块

Python 运行时系统定义了一些特殊变量和属性，它们的名称由双下划线分隔。其中一个特殊变量叫做`__name__`，它为我们的模块提供了一种方式来确定它是作为脚本运行还是被导入到另一个模块或 REPL 中。要查看如何操作，请添加：

```py
print(__name__)

```

在`fetch_words()`函数之外的模块末尾添加。

首先，让我们将修改后的 words 模块重新导入到 REPL 中：

```py
$ python3
Python 3.5.0 (default, Nov  3 2015, 13:17:02)
[GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import words
words

```

我们可以看到，当导入`__name__`时，它确实会评估为模块的名称。

顺便说一句，如果再次导入模块，print 语句将*不会*被执行；模块代码只在第一次导入时执行一次：

```py
>>> import words
>>>

```

现在让我们尝试将模块作为脚本运行：

```py
$ python3 words.py
__main__

```

在这种情况下，特殊的`__name__`变量等于字符串“**main**”，也由双下划线分隔。我们的模块可以利用这种行为来检测它的使用方式。我们用一个 if 语句替换 print 语句，该语句测试`__name__`的值。如果值等于“**main**”，那么我们的函数就会被执行：

```py
if __name__ == '__main__':
    fetch_words()

```

现在我们可以安全地导入我们的模块，而不会过度执行我们的函数：

```py
$ python3
>>> import words
>>>

```

我们可以有用地将我们的函数作为脚本运行：

```py
$ python3 words.py
It
was
the
best
of
times

```

### Python 执行模型

为了在 Python 中有一个真正坚实的基础，了解 Python 的*执行模型*是很重要的。我们指的是定义模块导入和执行期间发生的函数定义和其他重要事件的规则。为了帮助你发展这种理解，我们将专注于`def`关键字，因为你已经熟悉它。一旦你了解了 Python 如何处理`def`，你就会知道大部分关于 Python 执行模型的知识。

重要的是要理解这一点：**`def`不仅仅是一个声明，它是一个*语句***。这意味着`def`实际上是在运行时执行的，与其他顶层模块范围代码一起。`def`的作用是将函数体中的代码绑定到`def`后面的名称。当模块被导入或运行时，所有顶层语句都会运行，这是模块命名空间中的函数定义的方式。

重申一下，`def`是在运行时执行的。这与许多其他语言中处理函数定义的方式非常不同，特别是在编译语言如 C++、Java 和 C#中。在这些语言中，函数定义是由编译器在*编译时*处理的，而不是在运行时。^(4)实际执行程序时，这些函数定义已经固定。在 Python 中没有编译器^(5)，函数在执行之前并不存在任何形式，除了源代码。事实上，由于函数只有在导入时处理其`def`时才被定义，因此在从未导入的模块中的函数将永远不会被定义。

理解 Python 函数定义的这种动态特性对于后面本书中的重要概念至关重要，所以确保你对此感到舒适。如果你有 Python 调试器，比如在 IDE 中，你可以花一些时间逐步执行你的`words.py`模块。

#### 模块、脚本和程序之间的区别

有时我们会被问及 Python 模块、Python 脚本和 Python 程序之间的区别。任何`.py`文件都构成一个 Python 模块，但正如我们所见，模块可以被编写为方便导入、方便执行，或者使用`if __name__ == "__main__"`的习惯用法，两者兼而有之。

我们强烈建议即使是简单的脚本也要可导入，因为如果可以从 Python REPL 访问代码，这样可以极大地简化开发和测试。同样，即使是只在生产环境中导入的模块也会受益于具有可执行的测试代码。因此，我们创建的几乎所有模块都采用了定义一个或多个可导入函数的形式，并附有后缀以便执行。

将模块视为 Python 脚本或 Python 程序取决于上下文和用法。将 Python 仅视为脚本工具是错误的，因为许多大型复杂的应用程序都是专门使用 Python 构建的，而不是像 Windows 批处理文件或 Unix shell 脚本那样。

### 设置带有命令行参数的主函数

让我们进一步完善我们的单词获取模块。首先，我们将进行一些小的重构，将单词检索和收集与单词打印分开：

```py
from urllib.request import urlopen

# This fetches the words and returns them as a list.
def fetch_words():
    with urlopen('http://sixty-north.com/c/t.txt') as story:
        story_words = []
        for line in story:
            line_words = line.decode('utf-8').split()
            for word in line_words:
                story_words.append(word)
    return story_words

# This prints a list of words
def print_words(story_words):
    for word in story_words:
      print(word)

if __name__ == '__main__':
    words = fetch_words()
    print_words(words)

```

我们这样做是因为它分离了两个重要的关注点：在导入时，我们宁愿得到单词列表，但在直接运行时，我们更希望单词被打印出来。

接下来，我们将从`if __name__ == '__main__'`块中提取代码到一个名为`main()`的函数中：

```py
def main():
    words = fetch_words()
    print_words(words)

if __name__ == '__main__':
    main()

```

通过将这段代码移到一个函数中，我们可以在 REPL 中测试它，而在模块范围的 if 块中是不可能的。

现在我们可以在 REPL 中尝试这些函数：

```py
>>> from words import (fetch_words, print_words)
>>> print_words(fetch_words())

```

我们利用这个机会介绍了`import`语句的一些新形式。第一种新形式使用逗号分隔的列表从模块中导入多个对象。括号是可选的，但如果列表很长，它们可以允许您将此列表分成多行。这种形式可能是最广泛使用的`import`语句的形式之一。

第二种新形式使用星号通配符从模块中导入所有内容：

```py
>>> from words import *

```

后一种形式仅建议在 REPL 上进行临时使用。它可能会在程序中造成严重破坏，因为导入的内容现在可能超出您的控制范围，从而在将来可能导致潜在的命名空间冲突。

完成这些后，我们可以从 URL 获取单词：

```py
>>> fetch_words()
['It', 'was', 'the', 'best', 'of', 'times', 'it', 'was', 'the', 'worst',
'of', 'times', 'it', 'was', 'the', 'age', 'of', 'wisdom', 'it', 'was',
'the', 'age', 'of', 'foolishness', 'it', 'was', 'the', 'epoch', 'of',
'belief', 'it', 'was', 'the', 'epoch', 'of', 'incredulity', 'it', 'was',
'the', 'season', 'of', 'Light', 'it', 'was', 'the', 'season', 'of',
'Darkness', 'it', 'was', 'the', 'spring', 'of', 'hope', 'it', 'was', 'the',
'winter', 'of', 'despair', 'we', 'had', 'everything', 'before', 'us', 'we',
'had', 'nothing', 'before', 'us', 'we', 'were', 'all', 'going', 'direct',
'to', 'Heaven', 'we', 'were', 'all', 'going', 'direct', 'the', 'other',
'way', 'in', 'short', 'the', 'period', 'was', 'so', 'far', 'like', 'the',
'present', 'period', 'that', 'some', 'of', 'its', 'noisiest', 'authorities',
'insisted', 'on', 'its', 'being', 'received', 'for', 'good', 'or', 'for',
'evil', 'in', 'the', 'superlative', 'degree', 'of', 'comparison', 'only']

```

由于我们已将获取代码与打印代码分开，因此我们还可以打印*任何*单词列表：

```py
>>> print_words(['Any', 'list', 'of', 'words'])
Any
list
of
words

```

事实上，我们甚至可以运行主程序：

```py
>>> main()
It
was
the
best
of
times

```

请注意，`print_words()`函数对列表中的项目类型并不挑剔。它可以很好地打印数字列表：

```py
>>> print_words([1, 7, 3])
1
7
3

```

因此，也许`print_words()`不是最好的名称。实际上，该函数也没有提到列表-它可以很高兴地打印任何 for 循环能够迭代的集合，例如字符串：

```py
>>> print_words("Strings are iterable too")
S
t
r
i
n
g
s

a
r
e

i
t
e
r
a
b
l
e

t
o
o

```

因此，让我们进行一些小的重构，并将此函数重命名为`print_items()`，并相应地更改函数内的变量名：

```py
def print_items(items):
    for item in items:
        print(item)

```

最后，对我们的模块的一个明显改进是用一个可以传递的值替换硬编码的 URL。让我们将该值提取到`fetch_words()`函数的参数中：

```py
def fetch_words(url):
    with urlopen(url) as story:
        story_words = []
        for line in story:
            line_words = line.decode('utf-8').split()
            for word in line_words:
                story_words.append(word)
    return story_words

```

#### 接受命令行参数

最后一次更改实际上破坏了我们的`main()`，因为它没有传递新的`url`参数。当我们将模块作为独立程序运行时，我们需要接受 URL 作为命令行参数。在 Python 中访问命令行参数是通过`sys`模块的一个属性`argv`，它是一个字符串列表。要使用它，我们必须首先在程序顶部导入`sys`模块：

```py
import sys

```

然后我们从列表中获取第二个参数（索引为 1）：

```py
def main():
    url = sys.argv[1]
    words = fetch_words(url)
    print_items(words)

```

当然，这按预期工作：

```py
$ python3 words.py http://sixty-north.com/c/t.txt
It
was
the
best
of
times

```

这看起来很好，直到我们意识到我们无法从 REPL 有用地测试`main()`，因为它引用`sys.argv[1]`，在该环境中这个值不太可能有用：

```py
$ python3
Python 3.5.0 (default, Nov  3 2015, 13:17:02)
[GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from words import *
>>> main()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/sixtynorth/projects/sixty-north/the-python-apprentice/manuscript/code/\
pyfund/words.py", line 21, in main
    url = sys.argv[1]
IndexError: list index out of range
>>>

```

解决方案是允许将参数列表作为`main()`函数的形式参数传递，使用`sys.argv`作为`if __name__ == '__main__'`块中的实际参数：

```py
def main(url):
    words = fetch_words(url)
    print_items(words)

if __name__ == '__main__':
    main(sys.argv[1])

```

再次从 REPL 进行测试，我们可以看到一切都按预期工作：

```py
>>> from words import *
>>> main("http://sixty-north.com/c/t.txt")
It
was
the
best
of
times

```

Python 是开发命令行工具的好工具，您可能会发现您需要处理许多情况的命令行参数。对于更复杂的命令行处理，我们建议您查看[Python 标准库`argparse`](https://docs.python.org/3/library/argparse.html)模块或[受启发的第三方`docopt`模块](http://docopt.org/)。

* * *

### 禅意时刻

![](img/zen-sparse-is-better-than-dense.png)

您会注意到我们的顶级函数之间有两个空行。这是现代 Python 代码的传统。

根据[PEP 8 风格指南](https://www.python.org/dev/peps/pep-0008/)，在模块级函数之间使用两个空行是习惯的。我们发现这种约定对我们有所帮助，使代码更容易导航。同样，我们在函数内使用单个空行进行逻辑分隔。

* * *

### 文档字符串

我们之前看到了如何在 REPL 上询问 Python 函数的帮助。让我们看看如何将这种自我记录的能力添加到我们自己的模块中。

Python 中的 API 文档使用一种称为*docstrings*的设施。 Docstrings 是出现在命名块（例如函数或模块）的第一条语句中的文字字符串。让我们记录`fetch_words()`函数：

```py
def fetch_words(url):
    """Fetch a list of words from a URL."""
    with urlopen(url) as story:
        story_words = []
        for line in story:
            line_words = line.decode('utf-8').split()
            for word in line_words:
                story_words.append(word)
    return story_words

```

我们甚至使用三引号字符串来编写单行文档字符串，因为它们可以很容易地扩展以添加更多细节。

Python 文档字符串的一个约定在[PEP 257](https://www.python.org/dev/peps/pep-0257/)中有记录，尽管它并没有被广泛采用。各种工具，如[Sphinx](http://www.sphinx-doc.org/)，可用于从 Python 文档字符串构建 HTML 文档，每个工具都规定了其首选的文档字符串格式。我们的首选是使用[Google 的 Python 风格指南](https://google.github.io/styleguide/pyguide.html)中提出的形式，因为它适合被机器解析，同时在控制台上仍然可读：

```py
def fetch_words(url):
    """Fetch a list of words from a URL.

 Args:
 url: The URL of a UTF-8 text document.

 Returns:
 A list of strings containing the words from
 the document.
 """
    with urlopen(url) as story:
        story_words = []
        for line in story:
            line_words = line.decode('utf-8').split()
            for word in line_words:
                story_words.append(word)
    return story_words

```

现在我们将从 REPL 中访问这个`help()`：

```py
$ python3
Python 3.5.0 (default, Nov  3 2015, 13:17:02)
[GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from words import *
>>> help(fetch_words)

Help on function fetch_words in module words:

fetch_words(url)
    Fetch a list of words from a URL.

    Args:
        url: The URL of a UTF-8 text document.

    Returns:
        A list of strings containing the words from
        the document.
(END)

```

我们将为其他函数添加类似的文档字符串：

```py
def print_items(items):
    """Print items one per line.

 Args:
 items: An iterable series of printable items.
 """
    for item in items:
        print(item)

def main(url):
    """Print each word from a text document from at a URL.

 Args:
 url: The URL of a UTF-8 text document.
 """
    words = fetch_words(url)
    print_items(words)

```

以及模块本身的文档字符串。模块文档字符串应放在模块的开头，任何语句之前：

```py
"""Retrieve and print words from a URL.

Usage:

 python3 words.py <URL>
"""

import sys
from urllib.request import urlopen

```

现在当我们在整个模块上请求`help()`时，我们会得到相当多有用的信息：

```py
$ python3
Python 3.5.0 (default, Nov  3 2015, 13:17:02)
[GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import words
>>> help(words)

Help on module words:

NAME
    words - Retrieve and print words from a URL.

DESCRIPTION
    Usage:

        python3 words.py <URL>

FUNCTIONS
    fetch_words(url)
        Fetch a list of words from a URL.

        Args:
            url: The URL of a UTF-8 text document.

        Returns:
            A list of strings containing the words from
            the document.

    main(url)
        Print each word from a text document from at a URL.

        Args:
            url: The URL of a UTF-8 text document.

    print_items(items)
        Print items one per line.

        Args:
            items: An iterable series of printable items.

FILE
    /Users/sixtynorth/the-python-apprentice/words.py

(END)

```

### 注释

我们认为文档字符串是 Python 代码中大多数文档的正确位置。它们解释了如何使用模块提供的功能，而不是它的工作原理。理想情况下，您的代码应该足够清晰，不需要辅助解释。尽管如此，有时需要解释为什么选择了特定的方法或使用了特定的技术，我们可以使用 Python 注释来做到这一点。Python 中的注释以`#`开头，直到行尾。

作为演示，让我们记录这样一个事实，即为什么我们在调用`main()`时使用`sys.argv[1]`而不是`sys.argv[0]`可能不是立即明显的：

```py
if __name__ == '__main__':
    main(sys.argv[1])  # The 0th arg is the module filename.

```

### Shebang

在类 Unix 系统上，脚本的第一行通常包括一个特殊的注释`#!`，称为*shebang*。这允许程序加载器识别应该使用哪个解释器来运行程序。Shebang 还有一个额外的目的，方便地在文件顶部记录 Python 代码是 Python 2 还是 Python 3。

您的 shebang 命令的确切细节取决于系统上 Python 的位置。典型的 Python 3 shebang 使用 Unix 的`env`程序来定位您的`PATH`环境变量上的 Python 3，这一点非常重要，它与 Python 虚拟环境兼容：

```py
#!/usr/bin/env python3

```

#### Linux 和 Mac 上可执行的 Python 程序

在 Mac 或 Linux 上，我们必须在 shebang 生效之前使用`chmod`命令将脚本标记为可执行：

```py
$ chmod +x words.py

```

做完这些之后，我们现在可以直接运行我们的脚本：

```py
$ ./words.py http://sixty-north.com/c/t.txt

```

#### Windows 上可执行的 Python 程序

从 Python 3.3 开始，Windows 上的 Python 也支持使用 shebang 来使 Python 脚本直接可执行，即使看起来只能在类 Unix 系统上正常工作的 shebang 也会在 Windows 上按预期工作。这是因为 Windows Python 发行版现在使用一个名为*PyLauncher*的程序。 PyLauncher 的可执行文件名为`py.exe`，它将解析 shebang 并找到适当版本的 Python。

例如，在 Windows 的`cmd`提示符下，这个命令就足以用 Python 3 运行你的脚本（即使你也安装了 Python 2）：

```py
> words.py http://sixty-north.com/c/t.txt

```

在 Powershell 中，等效的是：

```py
PS> .\words.py http://sixty-north.com/c/t.txt

```

您可以在[PEP 397](https://www.python.org/dev/peps/pep-0397/)中了解更多关于 PyLauncher 的信息。

### 总结

+   Python 模块：

+   Python 代码放在名为模块的`*.py`文件中。

+   模块可以通过将它们作为 Python 解释器的第一个参数直接执行。

+   模块也可以被导入到 REPL 中，此时模块中的所有顶级语句将按顺序执行。

+   Python 函数：

+   使用`def`关键字定义命名函数，后面跟着函数名和括号中的参数列表。

+   我们可以使用`return`语句从函数中返回对象。

+   没有参数的返回语句返回`None`，在每个函数体的末尾也是如此。

+   模块执行：

+   我们可以通过检查特殊的`__name__`变量的值来检测模块是否已导入或执行。如果它等于字符串`"__main__"`，我们的模块已直接作为程序执行。通过在模块末尾使用顶层`if __name__ == '__main__'`习语来执行函数，如果满足这个条件，我们的模块既可以被有用地导入，又可以被执行，这是一个重要的测试技术，即使对于短脚本也是如此。

+   模块代码只在第一次导入时执行一次。

+   `def`关键字是一个语句，将可执行代码绑定到函数名。

+   命令行参数可以作为字符串列表访问，通过`sys`模块的`argv`属性。零号命令行参数是脚本文件名，因此索引为 1 的项是第一个真正的参数。

+   Python 的动态类型意味着我们的函数可以非常通用，关于它们参数的类型。

+   文档字符串：

+   作为函数定义的第一行的文字字符串形成函数的文档字符串。它们通常是包含使用信息的三引号多行字符串。

+   在 REPL 中，可以使用`help()`检索文档字符串中提供的函数文档。

+   模块文档字符串应放置在模块的开头，先于任何 Python 语句，如导入语句。

+   注释：

+   Python 中的注释以井号字符开头，并延续到行尾。

+   模块的第一行可以包含一个特殊的注释，称为 shebang，允许程序加载器在所有主要平台上启动正确的 Python 解释器。
