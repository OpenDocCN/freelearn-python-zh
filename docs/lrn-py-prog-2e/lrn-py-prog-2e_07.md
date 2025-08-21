# 第七章：文件和数据持久化

"持久是我们称之为生活的冒险的关键。" - Torsten Alexander Lange

在前几章中，我们已经探讨了 Python 的几个不同方面。由于示例具有教学目的，我们在简单的 Python shell 中运行它们，或者以 Python 模块的形式运行。它们运行，可能在控制台上打印一些内容，然后终止，不留下它们短暂存在的痕迹。

现实世界的应用通常大不相同。当然，它们仍然在内存中运行，但它们与网络、磁盘和数据库进行交互。它们还使用适合情况的格式与其他应用程序和设备交换信息。

在本章中，我们将开始接近真实世界，探索以下内容：

+   文件和目录

+   压缩

+   网络和流

+   JSON 数据交换格式

+   使用 pickle 和 shelve 进行数据持久化，来自标准库

+   使用 SQLAlchemy 进行数据持久化

像往常一样，我会努力平衡广度和深度，以便在本章结束时，您将对基本原理有扎实的理解，并且将知道如何在网络上获取更多信息。

# 处理文件和目录

在处理文件和目录时，Python 提供了许多有用的工具。特别是在以下示例中，我们将利用`os`和`shutil`模块。由于我们将在磁盘上读写，我将使用一个名为`fear.txt`的文件，其中包含了《恐惧》（Thich Nhat Hanh 著）的摘录，作为我们的一些示例的试验品。

# 打开文件

在 Python 中打开文件非常简单和直观。实际上，我们只需要使用`open`函数。让我们看一个快速的例子：

```py
# files/open_try.py
fh = open('fear.txt', 'rt')  # r: read, t: text

for line in fh.readlines():
    print(line.strip())  # remove whitespace and print

fh.close()
```

前面的代码非常简单。我们调用`open`，传递文件名，并告诉`open`我们要以文本模式读取它。文件名之前没有路径信息；因此，`open`将假定文件在运行脚本的同一文件夹中。这意味着如果我们从`files`文件夹外部运行此脚本，那么`fear.txt`将找不到。

一旦文件被打开，我们就会得到一个文件对象`fh`，我们可以用它来处理文件的内容。在这种情况下，我们使用`readlines()`方法来迭代文件中的所有行，并打印它们。我们对每一行调用`strip()`来去除内容周围的任何额外空格，包括末尾的行终止字符，因为`print`会为我们添加一个。这是一个快速而粗糙的解决方案，在这个例子中有效，但是如果文件的内容包含需要保留的有意义的空格，那么您将需要在清理数据时稍微小心。在脚本的结尾，我们刷新并关闭流。

关闭文件非常重要，因为我们不希望冒着无法释放对文件的控制的风险。因此，我们需要采取一些预防措施，并将前面的逻辑包装在`try`/`finally`块中。这样做的效果是，无论我们尝试打开和读取文件时可能发生什么错误，我们都可以放心地确保`close()`会被调用：

```py
# files/open_try.py
try:
    fh = open('fear.txt', 'rt')
    for line in fh.readlines():
        print(line.strip())
finally:
    fh.close()
```

逻辑完全相同，但现在也是安全的。

如果您现在不理解`try`/`finally`，不要担心。我们将在下一章中探讨如何处理异常。现在，可以说在`try`块的主体中放置代码会为该代码添加一个机制，允许我们检测错误（称为*异常*）并决定发生错误时该怎么办。在这种情况下，如果发生错误，我们实际上不做任何事情，但是通过在`finally`块中关闭文件，我们确保该行无论是否发生错误都会被执行。

我们可以通过以下方式简化前面的示例：

```py
# files/open_try.py
try:
    fh = open('fear.txt')  # rt is default
    for line in fh:  # we can iterate directly on fh
        print(line.strip())
finally:
    fh.close()
```

如您所见，`rt`是打开文件的默认模式，因此我们不需要指定它。此外，我们可以直接在`fh`上进行迭代，而不需要显式调用`readlines()`。Python 非常友好，为我们提供了简写，使我们的代码更短，更容易阅读。

所有前面的示例都会在控制台上打印文件的内容（查看源代码以阅读整个内容）：

```py
An excerpt from Fear - By Thich Nhat Hanh

The Present Is Free from Fear

When we are not fully present, we are not really living. We’re not really there, either for our loved ones or for ourselves. If we’re not there, then where are we? We are running, running, running, even during our sleep. We run because we’re trying to escape from our fear.
...
```

# 使用上下文管理器打开文件

让我们承认吧：不得不使用`try`/`finally`块来传播我们的代码并不是最好的选择。通常情况下，Python 给我们提供了一种更好的方式以安全的方式打开文件：使用上下文管理器。让我们先看看代码：

```py
# files/open_with.py
with open('fear.txt') as fh:
    for line in fh:
        print(line.strip())
```

前面的示例等同于前面的一个示例，但读起来更好。`with`语句支持由上下文管理器定义的运行时上下文的概念。这是使用一对方法`__enter__`和`__exit__`实现的，允许用户定义的类定义在执行语句体之前进入的运行时上下文，并在语句结束时退出。`open`函数在由上下文管理器调用时能够产生一个文件对象，但它真正的美妙之处在于`fh.close()`将会自动为我们调用，即使在出现错误的情况下也是如此。

上下文管理器在多种不同的场景中使用，比如线程同步、文件或其他对象的关闭，以及网络和数据库连接的管理。你可以在`contextlib`文档页面中找到关于它们的信息（[`docs.python.org/3.7/library/contextlib.html`](https://docs.python.org/3.7/library/contextlib.html)）。

# 读写文件

现在我们知道如何打开文件了，让我们看看我们有几种不同的方式可以读写文件：

```py
# files/print_file.py
with open('print_example.txt', 'w') as fw:
    print('Hey I am printing into a file!!!', file=fw)
```

第一种方法使用了`print`函数，你在前几章中已经见过很多次。在获取文件对象之后，这次指定我们打算向其写入（"`w`"），我们可以告诉`print`调用将其效果定向到文件，而不是默认的`sys.stdout`，当在控制台上执行时，它会映射到它。

前面的代码的效果是，如果`print_example.txt`文件不存在，则创建它，或者如果存在，则将其截断，并将行`Hey I am printing into a file!!!`写入其中。

这很简单易懂，但并不是我们通常写文件时所做的。让我们看一个更常见的方法：

```py
# files/read_write.py
with open('fear.txt') as f:
    lines = [line.rstrip() for line in f]

with open('fear_copy.txt', 'w') as fw:
    fw.write('\n'.join(lines))
```

在前面的示例中，我们首先打开`fear.txt`并逐行将其内容收集到一个列表中。请注意，这次我调用了一个更精确的方法`rstrip()`，作为一个例子，以确保我只去掉每行右侧的空白。

在代码片段的第二部分中，我们创建了一个新文件`fear_copy.txt`，并将原始文件中的所有行写入其中，用换行符`\n`连接起来。Python 很慷慨，并且默认使用*通用换行符*，这意味着即使原始文件的换行符可能与`\n`不同，它也会在返回行之前自动转换为我们。当然，这种行为是可以自定义的，但通常它正是你想要的。说到换行符，你能想到副本中可能缺少的换行符吗？

# 读写二进制模式

请注意，通过在选项中传递`t`来打开文件（或者省略它，因为它是默认值），我们是以文本模式打开文件。这意味着文件的内容被视为文本进行处理和解释。如果你希望向文件写入字节，可以以二进制模式打开它。当你处理不仅包含原始文本的文件时，这是一个常见的要求，比如图像、音频/视频以及一般的任何其他专有格式。

为了处理二进制模式的文件，只需在打开它们时指定`b`标志，就像以下示例中所示：

```py
# files/read_write_bin.py
with open('example.bin', 'wb') as fw:
    fw.write(b'This is binary data...')

with open('example.bin', 'rb') as f:
    print(f.read())  # prints: b'This is binary data...'
```

在这个示例中，我仍然使用文本作为二进制数据，但它可以是任何你想要的。你可以看到它被视为二进制数据的事实，因为在输出中你会得到`b'This ...'`前缀。

# 防止覆盖现有文件

Python 允许我们打开文件进行写入。通过使用`w`标志，我们打开一个文件并截断其内容。这意味着文件被覆盖为一个空文件，原始内容丢失。如果您希望仅在文件不存在时打开文件进行写入，可以在下面的示例中使用`x`标志：

```py
# files/write_not_exists.py
with open('write_x.txt', 'x') as fw:
    fw.write('Writing line 1')  # this succeeds

with open('write_x.txt', 'x') as fw:
    fw.write('Writing line 2')  # this fails
```

如果您运行前面的片段，您将在目录中找到一个名为`write_x.txt`的文件，其中只包含一行文本。事实上，片段的第二部分未能执行。这是我在控制台上得到的输出：

```py
$ python write_not_exists.py
Traceback (most recent call last):
 File "write_not_exists.py", line 6, in <module>
 with open('write_x.txt', 'x') as fw:
FileExistsError: [Errno 17] File exists: 'write_x.txt'
```

# 检查文件和目录的存在

如果您想确保文件或目录存在（或不存在），则需要使用`os.path`模块。让我们看一个小例子：

```py
# files/existence.py
import os

filename = 'fear.txt'
path = os.path.dirname(os.path.abspath(filename))

print(os.path.isfile(filename))  # True
print(os.path.isdir(path))  # True
print(path)  # /Users/fab/srv/lpp/ch7/files
```

前面的片段非常有趣。在使用相对引用声明文件名之后（因为缺少路径信息），我们使用`abspath`来计算文件的完整绝对路径。然后，我们通过调用`dirname`来获取路径信息（删除末尾的文件名）。正如您所看到的，结果在最后一行打印出来。还要注意我们如何通过调用`isfile`和`isdir`来检查文件和目录的存在。在`os.path`模块中，您可以找到处理路径名所需的所有函数。

如果您需要以不同的方式处理路径，可以查看`pathlib`。虽然`os.path`使用字符串，但`pathlib`提供了表示适合不同操作系统语义的文件系统路径的类。这超出了本章的范围，但如果您感兴趣，请查看 PEP428（[`www.python.org/dev/peps/pep-0428/`](https://www.python.org/dev/peps/pep-0428/)）以及标准库中的页面。

# 操作文件和目录

让我们看几个快速示例，演示如何操作文件和目录。第一个示例操作内容：

```py
# files/manipulation.py
from collections import Counter
from string import ascii_letters

chars = ascii_letters + ' '

def sanitize(s, chars):
    return ''.join(c for c in s if c in chars)

def reverse(s):
    return s[::-1]

with open('fear.txt') as stream:
    lines = [line.rstrip() for line in stream]

with open('raef.txt', 'w') as stream:
    stream.write('\n'.join(reverse(line) for line in lines))

# now we can calculate some statistics
lines = [sanitize(line, chars) for line in lines]
whole = ' '.join(lines)
cnt = Counter(whole.lower().split())
print(cnt.most_common(3))
```

前面的示例定义了两个函数：`sanitize`和`reverse`。它们是简单的函数，其目的是从字符串中删除任何不是字母或空格的内容，并分别生成字符串的反向副本。

我们打开`fear.txt`并将其内容读入列表。然后我们创建一个新文件`raef.txt`，其中包含原始文件的水平镜像版本。我们使用`join`在新行字符上写入`lines`的所有内容。也许更有趣的是最后的部分。首先，我们通过列表推导将`lines`重新分配为其经过清理的版本。然后我们将它们放在`whole`字符串中，最后将结果传递给`Counter`。请注意，我们拆分字符串并将其转换为小写。这样，每个单词都将被正确计数，而不管其大小写如何，并且由于`split`，我们不需要担心任何额外的空格。当我们打印出最常见的三个单词时，我们意识到真正的 Thich Nhat Hanh 的重点在于其他人，因为`we`是文本中最常见的单词：

```py
$ python manipulation.py
[('we', 17), ('the', 13), ('were', 7)]
```

现在让我们看一个更加面向磁盘操作的操作示例，其中我们使用`shutil`模块：

```py
# files/ops_create.py
import shutil
import os

BASE_PATH = 'ops_example'  # this will be our base path
os.mkdir(BASE_PATH)

path_b = os.path.join(BASE_PATH, 'A', 'B')
path_c = os.path.join(BASE_PATH, 'A', 'C')
path_d = os.path.join(BASE_PATH, 'A', 'D')

os.makedirs(path_b)
os.makedirs(path_c)

for filename in ('ex1.txt', 'ex2.txt', 'ex3.txt'):
    with open(os.path.join(path_b, filename), 'w') as stream:
        stream.write(f'Some content here in {filename}\n')

shutil.move(path_b, path_d)

shutil.move(
    os.path.join(path_d, 'ex1.txt'),
    os.path.join(path_d, 'ex1d.txt')
)
```

在前面的代码中，我们首先声明一个基本路径，该路径将安全地包含我们将要创建的所有文件和文件夹。然后我们使用`makedirs`创建两个目录：`ops_example/A/B`和`ops_example/A/C`。（您能想到使用`map`来创建这两个目录的方法吗？）。

我们使用`os.path.join`来连接目录名称，因为使用`/`会使代码专门运行在目录分隔符为`/`的平台上，但是代码将在具有不同分隔符的平台上失败。让我们委托给`join`来确定适当的分隔符。

创建目录后，在一个简单的`for`循环中，我们放入一些代码，创建目录`B`中的三个文件。然后，我们将文件夹`B`及其内容移动到另一个名称：`D`。最后，我们将`ex1.txt`重命名为`ex1d.txt`。如果您打开该文件，您会看到它仍然包含来自`for`循环的原始文本。对结果调用`tree`会产生以下结果：

```py
$ tree ops_example/
ops_example/
└── A
 ├── C
 └── D
 ├── ex1d.txt
 ├── ex2.txt
 └── ex3.txt 
```

# 操作路径名

让我们通过一个简单的例子来更多地探索`os.path`的能力：

```py
# files/paths.py
import os

filename = 'fear.txt'
path = os.path.abspath(filename)

print(path)
print(os.path.basename(path))
print(os.path.dirname(path))
print(os.path.splitext(path))
print(os.path.split(path))

readme_path = os.path.join(
    os.path.dirname(path), '..', '..', 'README.rst')

```

```py
print(readme_path)
print(os.path.normpath(readme_path))
```

阅读结果可能是对这个简单例子的足够好的解释：

```py
/Users/fab/srv/lpp/ch7/files/fear.txt           # path
fear.txt                                        # basename
/Users/fab/srv/lpp/ch7/files                    # dirname
('/Users/fab/srv/lpp/ch7/files/fear', '.txt')   # splitext
('/Users/fab/srv/lpp/ch7/files', 'fear.txt')    # split
/Users/fab/srv/lpp/ch7/files/../../README.rst   # readme_path
/Users/fab/srv/lpp/README.rst                   # normalized
```

# 临时文件和目录

有时，在运行一些代码时，能够创建临时目录或文件非常有用。例如，在编写影响磁盘的测试时，您可以使用临时文件和目录来运行您的逻辑并断言它是正确的，并确保在测试运行结束时，测试文件夹中没有剩余物品。让我们看看在 Python 中如何做到这一点：

```py
# files/tmp.py
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

with TemporaryDirectory(dir='.') as td:
    print('Temp directory:', td)
    with NamedTemporaryFile(dir=td) as t:
        name = t.name
        print(os.path.abspath(name))
```

前面的例子非常简单：我们在当前目录（`.`）中创建一个临时目录，并在其中创建一个命名临时文件。我们打印文件名以及其完整路径：

```py
$ python tmp.py
Temp directory: ./tmpwa9bdwgo
/Users/fab/srv/lpp/ch7/files/tmpwa9bdwgo/tmp3d45hm46 
```

运行此脚本将每次产生不同的结果。毕竟，这里我们创建的是一个临时随机名称，对吧？

# 目录内容

使用 Python，您还可以检查目录的内容。我将向您展示两种方法：

```py
# files/listing.py
import os

with os.scandir('.') as it:
    for entry in it:
        print(
            entry.name, entry.path,
            'File' if entry.is_file() else 'Folder'
        )
```

此片段使用`os.scandir`，在当前目录上调用。我们对结果进行迭代，每个结果都是`os.DirEntry`的一个实例，这是一个暴露有用属性和方法的好类。在代码中，我们访问了其中的一部分：`name`、`path`和`is_file()`。运行代码会产生以下结果（为简洁起见，我省略了一些结果）：

```py
$ python listing.py
fixed_amount.py ./fixed_amount.py File
existence.py ./existence.py File
...
ops_example ./ops_example Folder
...
```

扫描目录树的更强大的方法是由`os.walk`给我们的。让我们看一个例子：

```py
# files/walking.py
import os

for root, dirs, files in os.walk('.'):
    print(os.path.abspath(root))
    if dirs:
        print('Directories:')
        for dir_ in dirs:
            print(dir_)
        print()
    if files:
        print('Files:')
        for filename in files:
            print(filename)
        print()
```

运行前面的片段将产生当前目录中所有文件和目录的列表，并且对每个子目录都会执行相同的操作。

# 文件和目录压缩

在我们离开这一部分之前，让我给你举个创建压缩文件的例子。在书的源代码中，我有两个例子：一个创建一个 ZIP 文件，而另一个创建一个`tar.gz`文件。Python 允许您以几种不同的方式和格式创建压缩文件。在这里，我将向您展示如何创建最常见的一种，ZIP：

```py
# files/compression/zip.py
from zipfile import ZipFile

with ZipFile('example.zip', 'w') as zp:
    zp.write('content1.txt')
    zp.write('content2.txt')
    zp.write('subfolder/content3.txt')
    zp.write('subfolder/content4.txt')

with ZipFile('example.zip') as zp:
    zp.extract('content1.txt', 'extract_zip')
    zp.extract('subfolder/content3.txt', 'extract_zip')
```

在前面的代码中，我们导入`ZipFile`，然后在上下文管理器中，我们向其中写入四个虚拟上下文文件（其中两个在子文件夹中，以显示 ZIP 保留了完整路径）。之后，作为示例，我们打开压缩文件并从中提取了一些文件，放入`extract_zip`目录中。如果您有兴趣了解更多关于数据压缩的信息，请确保查看标准库中的*数据压缩和存档*部分（[`docs.python.org/3.7/library/archiving.html`](https://docs.python.org/3.7/library/archiving.html)），在那里您将能够学习有关此主题的所有内容。

# 数据交换格式

现代软件架构倾向于将应用程序拆分为几个组件。无论您是否采用面向服务的架构范例，或者将其推进到微服务领域，这些组件都必须交换数据。但即使您正在编写一个单体应用程序，其代码库包含在一个项目中，也有可能您必须与 API、其他程序交换数据，或者简单地处理网站前端和后端部分之间的数据流，这些部分很可能不会使用相同的语言。

选择正确的格式来交换信息至关重要。特定于语言的格式的优势在于，语言本身很可能会为您提供使序列化和反序列化变得轻而易举的所有工具。但是，您将失去与使用不同版本的相同语言或完全不同语言编写的其他组件进行通信的能力。无论未来看起来如何，只有在给定情况下这是唯一可能的选择时，才应选择特定于语言的格式。

一个更好的方法是选择一种语言无关的格式，可以被所有（或至少大多数）语言使用。在我领导的团队中，我们有来自英格兰、波兰、南非、西班牙、希腊、印度、意大利等国家的人。我们都说英语，所以无论我们的母语是什么，我们都可以理解彼此（嗯...大多数情况下！）。

在软件世界中，一些流行的格式在最近几年已经成为事实上的标准。最著名的可能是 XML、YAML 和 JSON。Python 标准库包括`xml`和`json`模块，而在 PyPI（[`docs.python.org/3.7/library/archiving.html`](https://docs.python.org/3.7/library/archiving.html)）上，您可以找到一些不同的包来处理 YAML。

在 Python 环境中，JSON 可能是最常用的格式。它胜过其他两种格式，因为它是标准库的一部分，而且它很简单。如果您曾经使用过 XML，您就知道它可能是多么可怕。

# 处理 JSON

**JSON**是**JavaScript 对象表示法**的缩写，它是 JavaScript 语言的一个子集。它已经存在了将近二十年，因此它是众所周知的，并且被基本上所有语言广泛采用，尽管它实际上是与语言无关的。您可以在其网站上阅读有关它的所有信息（[`www.json.org/`](https://www.json.org/)），但我现在要给您一个快速介绍。

JSON 基于两种结构：一组名称/值对和一个有序值列表。您会立即意识到，这两个对象分别映射到 Python 中的字典和列表数据类型。作为数据类型，它提供字符串、数字、对象和值，如 true、false 和 null。让我们看一个快速的例子来开始：

```py
# json_examples/json_basic.py
import sys
import json

data = {
    'big_number': 2 ** 3141,
    'max_float': sys.float_info.max,
    'a_list': [2, 3, 5, 7],
}

json_data = json.dumps(data)
data_out = json.loads(json_data)
assert data == data_out  # json and back, data matches
```

我们首先导入`sys`和`json`模块。然后我们创建一个包含一些数字和一个列表的简单字典。我想测试使用非常大的数字进行序列化和反序列化，所以我放了*2³¹⁴¹*和我的系统可以处理的最大浮点数。

我们使用`json.dumps`进行序列化，它将数据转换为 JSON 格式的字符串。然后将该数据输入`json.loads`，它执行相反的操作：从 JSON 格式的字符串中，将数据重构为 Python。在最后一行，我们确保原始数据和通过 JSON 进行序列化/反序列化的结果匹配。

让我们看看下一个例子中，如果我们打印 JSON 数据会是什么样子：

```py
# json_examples/json_basic.py
import json

info = {
    'full_name': 'Sherlock Holmes',
    'address': {
        'street': '221B Baker St',
        'zip': 'NW1 6XE',
        'city': 'London',
        'country': 'UK',
    }
}

print(json.dumps(info, indent=2, sort_keys=True))
```

在这个例子中，我们创建了一个包含福尔摩斯数据的字典。如果您像我一样是福尔摩斯的粉丝，并且在伦敦，您会在那个地址找到他的博物馆（我建议您去参观，它虽小但非常好）。

请注意我们如何调用`json.dumps`。我们已经告诉它用两个空格缩进，并按字母顺序排序键。结果是这样的：

```py
$ python json_basic.py
{
 "address": {
 "city": "London",
 "country": "UK",
 "street": "221B Baker St",
 "zip": "NW1 6XE"
 },
 "full_name": "Sherlock Holmes"
}
```

与 Python 的相似性非常大。唯一的区别是，如果您在字典的最后一个元素上放置逗号，就像我在 Python 中所做的那样（因为这是习惯的做法），JSON 会抱怨。

让我给您展示一些有趣的东西：

```py
# json_examples/json_tuple.py
import json

data_in = {
    'a_tuple': (1, 2, 3, 4, 5),
}

json_data = json.dumps(data_in)
print(json_data)  # {"a_tuple": [1, 2, 3, 4, 5]}
data_out = json.loads(json_data)
print(data_out)  # {'a_tuple': [1, 2, 3, 4, 5]}
```

在这个例子中，我们放了一个元组，而不是一个列表。有趣的是，从概念上讲，元组也是一个有序的项目列表。它没有列表的灵活性，但从 JSON 的角度来看，它仍然被视为相同的。因此，正如您可以从第一个`print`中看到的那样，在 JSON 中，元组被转换为列表。因此，它是一个元组的信息丢失了，当进行反序列化时，`data_out`中的`a_tuple`实际上是一个列表。在处理数据时，重要的是要记住这一点，因为经历一个涉及仅包括您可以使用的数据结构子集的格式的转换过程意味着会有信息丢失。在这种情况下，我们丢失了类型（元组与列表）的信息。

这实际上是一个常见的问题。例如，您不能将所有 Python 对象序列化为 JSON，因为不清楚 JSON 是否应该还原它（或者如何还原）。想想`datetime`，例如。该类的实例是 JSON 不允许序列化的 Python 对象。如果我们将其转换为字符串，比如`2018-03-04T12:00:30Z`，这是带有日期、时间和时区信息的 ISO 8601 表示，当进行反序列化时，JSON 应该怎么做？它应该说*这实际上可以反序列化为一个 datetime 对象，所以最好这样做*，还是应该简单地将其视为字符串并保留它？那些可以以多种方式解释的数据类型又该怎么办？

答案是，在处理数据交换时，我们经常需要在将对象序列化为 JSON 之前将其转换为更简单的格式。这样，当我们对其进行反序列化时，我们将知道如何正确地重建它们。

然而，在某些情况下，主要是为了内部使用，能够序列化自定义对象是很有用的，因此，只是为了好玩，我将向您展示两个例子：复数（因为我喜欢数学）和*datetime*对象。

# 自定义编码/解码与 JSON

在 JSON 世界中，我们可以将编码/解码这样的术语视为序列化/反序列化的同义词。它们基本上都意味着转换为 JSON，然后再从 JSON 转换回来。在下面的例子中，我将向您展示如何编码复数：

```py
# json_examples/json_cplx.py
import json

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {
                '_meta': '_complex',
                'num': [obj.real, obj.imag],
            }
        return json.JSONEncoder.default(self, obj)

data = {
    'an_int': 42,
    'a_float': 3.14159265,
    'a_complex': 3 + 4j,
}

json_data = json.dumps(data, cls=ComplexEncoder)
print(json_data)

def object_hook(obj):
    try:
        if obj['_meta'] == '_complex':
            return complex(*obj['num'])
    except (KeyError, TypeError):
        return obj

data_out = json.loads(json_data, object_hook=object_hook)
print(data_out)
```

首先，我们定义一个`ComplexEncoder`类，它需要实现`default`方法。这个方法被传递给所有需要被序列化的对象，一个接一个地，在`obj`变量中。在某个时候，`obj`将是我们的复数*3+4j*。当这种情况发生时，我们返回一个带有一些自定义元信息的字典，以及一个包含实部和虚部的列表。这就是我们需要做的，以避免丢失复数的信息。

然后我们调用`json.dumps`，但这次我们使用`cls`参数来指定我们的自定义编码器。结果被打印出来：

```py
{"an_int": 42, "a_float": 3.14159265, "a_complex": {"_meta": "_complex", "num": [3.0, 4.0]}}
```

一半的工作已经完成。对于反序列化部分，我们本可以编写另一个从`JSONDecoder`继承的类，但是，只是为了好玩，我使用了一种更简单的技术，使用了一个小函数：`object_hook`。

在`object_hook`的主体中，我们找到另一个`try`块，但现在不要担心它。我将在下一章节中详细解释它。重要的是`try`块本身的主体中的两行。该函数接收一个对象（请注意，只有当`obj`是一个字典时才调用该函数），如果元数据与我们的复数约定匹配，我们将实部和虚部传递给`complex`函数。`try`/`except`块只是为了防止格式不正确的 JSON 破坏整个过程（如果发生这种情况，我们只需返回对象本身）。

最后的打印返回：

```py
{'an_int': 42, 'a_float': 3.14159265, 'a_complex': (3+4j)}
```

您可以看到`a_complex`已经被正确反序列化。

现在让我们看一个稍微更复杂（没有刻意的双关语）的例子：处理`datetime`对象。我将把代码分成两个部分，序列化部分和之后的反序列化部分：

```py
# json_examples/json_datetime.py
import json
from datetime import datetime, timedelta, timezone

now = datetime.now()
now_tz = datetime.now(tz=timezone(timedelta(hours=1)))

class DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            try:
                off = obj.utcoffset().seconds
            except AttributeError:
                off = None

            return {
                '_meta': '_datetime',
                'data': obj.timetuple()[:6] + (obj.microsecond, ),
                'utcoffset': off,
            }
        return json.JSONEncoder.default(self, obj)

data = {
    'an_int': 42,
    'a_float': 3.14159265,
    'a_datetime': now,
    'a_datetime_tz': now_tz,
}

json_data = json.dumps(data, cls=DatetimeEncoder)
print(json_data)
```

这个例子稍微复杂的原因在于 Python 中的`datetime`对象可以是时区感知的或者不是；因此，我们需要更加小心。流程基本上和之前一样，只是处理不同的数据类型。我们首先获取当前的日期和时间信息，我们既不带（`now`）也带（`now_tz`）时区感知，只是为了确保我们的脚本工作。然后我们继续像之前一样定义一个自定义编码器，并再次实现`default`方法。在该方法中重要的部分是我们如何获取时间偏移（`off`）信息，以秒为单位，并且我们如何构造返回数据的字典。这一次，元数据表示它是*datetime*信息，然后我们保存时间元组的前六个项目（年、月、日、小时、分钟和秒），加上`data`键中的微秒，然后是偏移。你能看出`data`的值是元组的连接吗？如果你能，干得好！

当我们有了自定义编码器后，我们继续创建一些数据，然后进行序列化。`print`语句返回（在我进行了一些美化之后）：

```py
{
 "a_datetime": {
 "_meta": "_datetime",
 "data": [2018, 3, 18, 17, 57, 27, 438792],
 "utcoffset": null
 },
 "a_datetime_tz": {
 "_meta": "_datetime",
 "data": [2018, 3, 18, 18, 57, 27, 438810],
 "utcoffset": 3600
 },
 "a_float": 3.14159265,
 "an_int": 42
}
```

有趣的是，我们发现`None`被翻译为`null`，它的 JavaScript 等价物。此外，我们可以看到我们的数据似乎已经被正确编码。让我们继续进行脚本的第二部分：

```py
# json_examples/json_datetime.py
def object_hook(obj):
    try:
        if obj['_meta'] == '_datetime':
            if obj['utcoffset'] is None:
                tz = None
            else:
                tz = timezone(timedelta(seconds=obj['utcoffset']))
            return datetime(*obj['data'], tzinfo=tz)
    except (KeyError, TypeError):
        return obj

data_out = json.loads(json_data, object_hook=object_hook)
```

再次，我们首先验证元数据告诉我们它是一个`datetime`，然后我们继续获取时区信息。一旦我们有了时区信息，我们将 7 元组（使用`*`来解包其值）和时区信息传递给`datetime`调用，得到我们的原始对象。让我们通过打印`data_out`来验证它：

```py
{
 'a_datetime': datetime.datetime(2018, 3, 18, 18, 1, 46, 54693),
 'a_datetime_tz': datetime.datetime(
 2018, 3, 18, 19, 1, 46, 54711,
 tzinfo=datetime.timezone(datetime.timedelta(seconds=3600))),
 'a_float': 3.14159265,
 'an_int': 42
}
```

正如你所看到的，我们正确地得到了一切。作为一个练习，我想挑战你写相同的逻辑，但是对于一个`date`对象，这应该更简单。

在我们继续下一个主题之前，我想提个小心。也许这是违反直觉的，但是处理`datetime`对象可能是最棘手的事情之一，所以，虽然我很确定这段代码正在做它应该做的事情，我想强调我只是轻微地测试了它。所以如果你打算使用它，请彻底测试它。测试不同的时区，测试夏令时的开启和关闭，测试纪元前的日期等等。你可能会发现这一部分的代码需要一些修改来适应你的情况。

现在让我们转到下一个主题，IO。

# IO、流和请求

**IO**代表**输入**/**输出**，它广泛地指的是计算机与外部世界之间的通信。有几种不同类型的 IO，这超出了本章的范围来解释所有这些，但我仍然想给你举几个例子。

# 使用内存流

第一个将向你展示`io.StringIO`类，它是用于文本 IO 的内存流。而第二个则会逃离我们计算机的局限，并向你展示如何执行 HTTP 请求。让我们看看第一个例子：

```py
# io_examples/string_io.py
import io

stream = io.StringIO()
stream.write('Learning Python Programming.\n')
print('Become a Python ninja!', file=stream)

contents = stream.getvalue()
print(contents)

stream.close()
```

在前面的代码片段中，我们从标准库中导入了`io`模块。这是一个非常有趣的模块，其中包含许多与流和 IO 相关的工具。其中之一是`StringIO`，它是一个内存缓冲区，我们将使用两种不同的方法在其中写入两个句子，就像我们在本章的第一个例子中使用文件一样。我们既可以调用`StringIO.write`，也可以使用`print`，并告诉它将数据定向到我们的流中。

通过调用`getvalue`，我们可以获取流的内容（并打印它），最后我们关闭它。调用`close`会导致文本缓冲立即被丢弃。

有一种更优雅的方法来编写前面的代码（在你看之前，你能猜到吗？）：

```py
# io_examples/string_io.py
with io.StringIO() as stream:
    stream.write('Learning Python Programming.\n')
    print('Become a Python ninja!', file=stream)
    contents = stream.getvalue()
    print(contents)
```

是的，这又是一个上下文管理器。像`open`一样，`io.StringIO`在上下文管理器块内工作得很好。注意与`open`的相似之处：在这种情况下，我们也不需要手动关闭流。

内存对象在许多情况下都很有用。内存比磁盘快得多，对于少量数据来说，可能是完美的选择。

运行脚本时，输出为：

```py
$ python string_io.py
Learning Python Programming.
Become a Python ninja!
```

# 进行 HTTP 请求

现在让我们探索一些关于 HTTP 请求的例子。我将在这些例子中使用`requests`库，你可以使用`pip`安装它。我们将对[httpbin.org](http://httpbin.org/) API 执行 HTTP 请求，有趣的是，这个 API 是由`requests`库的创建者 Kenneth Reitz 开发的。这个库在全世界范围内被广泛采用：

```py
import requests

urls = {
    'get': 'https://httpbin.org/get?title=learn+python+programming',
    'headers': 'https://httpbin.org/headers',
    'ip': 'https://httpbin.org/ip',
    'now': 'https://now.httpbin.org/',
    'user-agent': 'https://httpbin.org/user-agent',
    'UUID': 'https://httpbin.org/uuid',
}

def get_content(title, url):
    resp = requests.get(url)
    print(f'Response for {title}')
    print(resp.json())

for title, url in urls.items():
    get_content(title, url)
    print('-' * 40)
```

前面的片段应该很容易理解。我声明了一个 URL 字典，对这些 URL 我想执行`requests`。我将执行请求的代码封装到一个小函数`get_content`中：如你所见，我们很简单地执行了一个 GET 请求（通过使用`requests.get`），并打印了响应的标题和 JSON 解码版本的响应体。让我多说一句关于最后一点。

当我们对网站或 API 执行请求时，我们会得到一个响应对象，这个对象很简单，就是服务器返回的内容。所有来自[httpbin.org](https://httpbin.org/)的响应体都是 JSON 编码的，所以我们不是通过`resp.text`获取响应体本身，然后手动解码它，而是通过在响应对象上利用`json`方法将两者结合起来。`requests`包变得如此广泛被采用有很多原因，其中一个绝对是它的易用性。

现在，当你在应用程序中执行请求时，你会希望有一个更加健壮的方法来处理错误等等，但是在本章中，一个简单的例子就足够了。别担心，我会在第十四章 *Web Development*中给你一个更全面的 HTTP 请求介绍。

回到我们的代码，最后，我们运行一个`for`循环并获取所有的 URL。当你运行它时，你会在控制台上看到每次调用的结果，就像这样（经过美化和简化）：

```py
$ python reqs.py
Response for get
{
  "args": {
    "title": "learn python programming"
  },
  "headers": {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "close",
    "Host": "httpbin.org",
    "User-Agent": "python-requests/2.19.0"
  },
  "origin": "82.47.175.158",
  "url": "https://httpbin.org/get?title=learn+python+programming"
}
... rest of the output omitted ... 
```

请注意，版本号和 IP 方面的输出可能会有些不同，这没关系。现在，GET 只是 HTTP 动词中的一个，它绝对是最常用的。第二个是无处不在的 POST，当你需要向服务器发送数据时，就会发起这种类型的请求。每当你在网上提交表单时，你基本上就是在发起一个 POST 请求。所以，让我们尝试以编程方式进行一个：

```py
# io_examples/reqs_post.py
import requests

url = 'https://httpbin.org/post'
data = dict(title='Learn Python Programming')

resp = requests.post(url, data=data)
print('Response for POST')
print(resp.json())
```

前面的代码与之前看到的代码非常相似，只是这一次我们不调用`get`，而是调用`post`，因为我们想发送一些数据，我们在调用中指定了这一点。`requests`库提供的远不止这些，它因其美丽的 API 而受到社区的赞扬。这是一个我鼓励你去了解和探索的项目，因为无论如何你最终都会一直使用它。

运行前面的脚本（并对输出进行一些美化处理）会产生以下结果：

```py
$ python reqs_post.py
Response for POST
{ 'args': {},
 'data': '',
 'files': {},
 'form': {'title': 'Learn Python Programming'},
 'headers': { 'Accept': '*/*',
 'Accept-Encoding': 'gzip, deflate',
 'Connection': 'close',
 'Content-Length': '30',
 'Content-Type': 'application/x-www-form-urlencoded',
 'Host': 'httpbin.org',
 'User-Agent': 'python-requests/2.7.0 CPython/3.7.0b2 '
 'Darwin/17.4.0'},
 'json': None,
```

```py
 'origin': '82.45.123.178',
 'url': 'https://httpbin.org/post'}
```

注意现在头部不同了，我们在响应体的`form`键值对中找到了发送的数据。

我希望这些简短的例子足以让你开始，特别是对于请求。网络每天都在变化，所以值得学习基础知识，然后不时地进行复习。

现在让我们转向本章的最后一个主题：以不同格式在磁盘上持久化数据。

# 在磁盘上持久化数据

在本章的最后一节中，我们将探讨如何以三种不同的格式将数据持久化到磁盘。我们将探索`pickle`、`shelve`，以及一个涉及使用 SQLAlchemy 访问数据库的简短示例，SQLAlchemy 是 Python 生态系统中最广泛采用的 ORM 库。

# 使用 pickle 序列化数据

Python 标准库中的`pickle`模块提供了将 Python 对象转换为字节流以及反之的工具。尽管`pickle`和`json`公开的 API 存在部分重叠，但两者是完全不同的。正如我们在本章前面看到的，JSON 是一种文本格式，人类可读，语言无关，并且仅支持 Python 数据类型的受限子集。另一方面，`pickle`模块不是人类可读的，转换为字节，是特定于 Python 的，并且由于 Python 的精彩内省能力，它支持大量的数据类型。

尽管存在这些差异，当你考虑使用其中一个时，你应该知道这些差异，我认为关于`pickle`最重要的关注点在于当你使用它时所面临的安全威胁。从不受信任的来源*unpickling*错误或恶意数据可能非常危险，因此如果你决定在你的应用程序中采用它，你需要格外小心。

话虽如此，让我们通过一个简单的例子来看它的运作方式：

```py
# persistence/pickler.py
import pickle
from dataclasses import dataclass

@dataclass
class Person:
    first_name: str
    last_name: str
    id: int

    def greet(self):
        print(f'Hi, I am {self.first_name} {self.last_name}'
              f' and my ID is {self.id}'
        )

people = [
    Person('Obi-Wan', 'Kenobi', 123),
    Person('Anakin', 'Skywalker', 456),
]

# save data in binary format to a file
with open('data.pickle', 'wb') as stream:
    pickle.dump(people, stream)

# load data from a file
with open('data.pickle', 'rb') as stream:
    peeps = pickle.load(stream)

for person in peeps:
    person.greet()
```

在前面的例子中，我们使用`dataclass`装饰器创建了一个`Person`类，我们在第六章中已经见过，*OOP，Decorators 和 Iterators*。我之所以用数据类写这个例子，只是为了向你展示`pickle`如何毫不费力地处理它，而不需要我们为了更简单的数据类型而做任何事情。

该类有三个属性：`first_name`、`last_name`和`id`。它还公开了一个`greet`方法，简单地打印出带有数据的 hello 消息。

我们创建了一个实例列表，然后将其保存到文件中。为了这样做，我们使用`pickle.dump`，将要*pickled*的内容和要写入的流传递给它。在那之后，我们立即从同一个文件中读取，并通过使用`pickle.load`，将该流的整个内容转换回 Python。为了确保对象已经被正确转换，我们在两个对象上都调用了`greet`方法。结果如下：

```py
$ python pickler.py
Hi, I am Obi-Wan Kenobi and my ID is 123
Hi, I am Anakin Skywalker and my ID is 456 
```

`pickle`模块还允许你通过`dumps`和`loads`函数（注意这两个名称末尾的`s`）将数据转换为（和从）字节对象。在日常应用中，当我们需要持久化不应该与另一个应用程序交换的 Python 数据时，通常会使用`pickle`。我最近遇到的一个例子是`flask`插件中的会话管理，它在将会话对象发送到 Redis 之前对其进行`pickle`。不过，在实践中，你不太可能经常使用这个库。

另一个可能使用得更少，但在资源短缺时非常有用的工具是`shelve`。

# 使用 shelve 保存数据

`shelf`是一个持久的类似字典的对象。它的美妙之处在于，你保存到`shelf`中的值可以是任何你可以`pickle`的对象，因此你不像使用数据库时那样受限制。尽管有趣且有用，但`shelve`模块在实践中很少使用。为了完整起见，让我们快速看一下它是如何工作的：

```py
# persistence/shelf.py
import shelve

class Person:
    def __init__(self, name, id):
        self.name = name
        self.id = id

with shelve.open('shelf1.shelve') as db:
    db['obi1'] = Person('Obi-Wan', 123)
    db['ani'] = Person('Anakin', 456)
    db['a_list'] = [2, 3, 5]
    db['delete_me'] = 'we will have to delete this one...'

    print(list(db.keys()))  # ['ani', 'a_list', 'delete_me', 'obi1']

    del db['delete_me']  # gone!

    print(list(db.keys()))  # ['ani', 'a_list', 'obi1']

    print('delete_me' in db)  # False
    print('ani' in db)  # True

    a_list = db['a_list']
    a_list.append(7)
```

```py
    db['a_list'] = a_list
    print(db['a_list'])  # [2, 3, 5, 7]
```

除了接线和围绕它的样板之外，前面的例子类似于使用字典进行练习。我们创建一个简单的`Person`类，然后在上下文管理器中打开一个`shelve`文件。正如你所看到的，我们使用字典语法来存储四个对象：两个`Person`实例，一个列表和一个字符串。如果我们打印`keys`，我们会得到一个包含我们使用的四个键的列表。打印完后，我们立即从架子上删除了（恰如其名的）`delete_me`键/值对。再次打印`keys`显示删除已成功。然后我们测试了一对键的成员资格，最后，我们将数字`7`附加到`a_list`上。请注意，我们必须从架子上提取列表，修改它，然后再次保存它。

如果这种行为是不希望的，我们可以做一些事情：

```py
# persistence/shelf.py
with shelve.open('shelf2.shelve', writeback=True) as db:
    db['a_list'] = [11, 13, 17]
    db['a_list'].append(19)  # in-place append!
    print(db['a_list'])  # [11, 13, 17, 19]
```

通过以`writeback=True`打开架子，我们启用了`writeback`功能，这使我们可以简单地将`a_list`附加到其中，就好像它实际上是常规字典中的一个值。这个功能之所以不是默认激活的原因是，它会以内存消耗和更慢的关闭架子的方式付出代价。

现在我们已经向与数据持久性相关的标准库模块致敬，让我们来看看 Python 生态系统中最广泛采用的 ORM：*SQLAlchemy*。

# 将数据保存到数据库

对于这个例子，我们将使用内存数据库，这将使事情对我们来说更简单。在书的源代码中，我留下了一些注释，以向您展示如何生成一个 SQLite 文件，所以我希望您也会探索这个选项。

您可以在[sqlitebrowser.org](http://sqlitebrowser.org/)找到一个免费的 SQLite 数据库浏览器。如果您对此不满意，您将能够找到各种工具，有些是免费的，有些不是免费的，您可以用来访问和操作数据库文件。

在我们深入代码之前，让我简要介绍一下关系数据库的概念。

关系数据库是一种允许您按照 1969 年由 Edgar F. Codd 发明的**关系模型**保存数据的数据库。在这个模型中，数据存储在一个或多个表中。每个表都有行（也称为**记录**或**元组**），每个行代表表中的一个条目。表还有列（也称为**属性**），每个列代表记录的一个属性。每个记录通过一个唯一键来标识，更常见的是**主键**，它是表中一个或多个列的联合。举个例子：想象一个名为`Users`的表，有列`id`、`username`、`password`、`name`和`surname`。这样的表非常适合包含我们系统的用户。每一行代表一个不同的用户。例如，具有值`3`、`gianchub`、`my_wonderful_pwd`、`Fabrizio`和`Romano`的行将代表我在系统中的用户。

模型被称为**关系型**的原因是因为您可以在表之间建立关系。例如，如果您向我们虚构的数据库添加一个名为`PhoneNumbers`的表，您可以向其中插入电话号码，然后通过关系建立哪个电话号码属于哪个用户。

为了查询关系数据库，我们需要一种特殊的语言。主要标准称为**SQL**，代表**结构化查询语言**。它源自一种称为**关系代数**的东西，这是一组用于模拟按照关系模型存储的数据并对其进行查询的非常好的代数。你通常可以执行的最常见操作包括对行或列进行过滤、连接表、根据某些标准对结果进行聚合等。举个英文例子，对我们想象中的数据库的查询可能是：*获取所有用户名以“m”开头，最多有一个电话号码的用户（用户名、名字、姓氏）*。在这个查询中，我们要求获取`User`表中的一部分列。我们通过筛选用户，只选择用户名以字母“m”开头的用户，甚至进一步，只选择最多有一个电话号码的用户。

在我在帕多瓦大学上学的时候，我花了一个学期的时间学习关系代数语义和标准 SQL（还有其他东西）。如果不是因为我在考试当天遭遇了一次严重的自行车事故，我会说这是我准备过的最有趣的考试之一。

现在，每个数据库都有自己的 SQL“口味”。它们都在一定程度上遵守标准，但没有一个完全遵守，并且它们在某些方面彼此不同。这在现代软件开发中构成了一个问题。如果我们的应用程序包含 SQL 代码，那么如果我们决定使用不同的数据库引擎，或者可能是同一引擎的不同版本，很可能会发现我们的 SQL 代码需要修改。

这可能会很痛苦，特别是因为 SQL 查询很快就会变得非常复杂。为了稍微减轻这种痛苦，计算机科学家们（感谢他们）创建了将特定语言的对象映射到关系数据库表的代码。毫不奇怪，这种工具的名称是**对象关系映射**（ORM）。

在现代应用程序开发中，通常会通过使用 ORM 来开始与数据库交互，如果你发现自己无法通过 ORM 执行需要执行的查询，那么你会转而直接使用 SQL。这是在完全没有 SQL 和不使用 ORM 之间的一个很好的折衷，这最终意味着专门化与数据库交互的代码，带来了前面提到的缺点。

在这一部分，我想展示一个利用 SQLAlchemy 的例子，这是最流行的 Python ORM。我们将定义两个模型（`Person`和`Address`），它们分别映射到一个表，然后我们将填充数据库并对其执行一些查询。

让我们从模型声明开始：

```py
# persistence/alchemy_models.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column, Integer, String, ForeignKey, create_engine)
from sqlalchemy.orm import relationship
```

一开始，我们导入一些函数和类型。然后，我们需要创建一个引擎。这个引擎告诉 SQLAlchemy 我们选择的数据库类型是什么。

```py
# persistence/alchemy_models.py
engine = create_engine('sqlite:///:memory:')
Base = declarative_base()

class Person(Base):
    __tablename__ = 'person'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

    addresses = relationship(
        'Address',
        back_populates='person',
        order_by='Address.email',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'{self.name}(id={self.id})'

class Address(Base):
    __tablename__ = 'address'

    id = Column(Integer, primary_key=True)
    email = Column(String)
    person_id = Column(ForeignKey('person.id'))
    person = relationship('Person', back_populates='addresses')

    def __str__(self):
        return self.email
    __repr__ = __str__

Base.metadata.create_all(engine)
```

然后，每个模型都继承自`Base`表，在这个例子中，它由`declarative_base()`返回的默认值组成。我们定义了`Person`，它映射到一个名为`person`的表，并公开`id`、`name`和`age`属性。我们还声明了与`Address`模型的关系，通过声明访问`addresses`属性将获取与我们正在处理的特定`Person`实例相关的`address`表中的所有条目。`cascade`选项影响创建和删除的工作方式，但这是一个更高级的概念，所以我建议你现在先略过它，以后可能再进行更深入的调查。

我们声明的最后一件事是`__repr__`方法，它为我们提供了对象的官方字符串表示。这应该是一个可以用来完全重建对象的表示，但在这个例子中，我只是用它来提供输出。Python 将`repr(obj)`重定向到对`obj.__repr__()`的调用。

我们还声明了`Address`模型，它将包含电子邮件地址，以及它们所属的人的引用。你可以看到`person_id`和`person`属性都是关于设置`Address`和`Person`实例之间的关系。注意我在`Address`上声明了`__str__`方法，然后给它分配了一个别名，叫做`__repr__`。这意味着在`Address`对象上调用`repr`和`str`最终都会调用`__str__`方法。这在 Python 中是一种常见的技术，所以我在这里有机会向你展示。

在最后一行，我们告诉引擎根据我们的模型在数据库中创建表。

对这段代码的更深理解需要比我能提供的空间更多，所以我鼓励你阅读**数据库管理系统**（**DBMS**）、SQL、关系代数和 SQLAlchemy。

现在我们有了我们的模型，让我们使用它们来保存一些数据！

让我们看看下面的例子：

```py
# persistence/alchemy.py
from alchemy_models import Person, Address, engine
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()
```

首先我们创建会话，这是我们用来管理数据库的对象。接下来，我们创建了两个人：

```py
anakin = Person(name='Anakin Skywalker', age=32)
obi1 = Person(name='Obi-Wan Kenobi', age=40)
```

然后我们为他们都添加电子邮件地址，使用了两种不同的技术。一种将它们分配给一个列表，另一种只是简单地将它们附加到列表中：

```py
obi1.addresses = [
    Address(email='obi1@example.com'),
    Address(email='wanwan@example.com'),
]

anakin.addresses.append(Address(email='ani@example.com'))
anakin.addresses.append(Address(email='evil.dart@example.com'))
anakin.addresses.append(Address(email='vader@example.com'))
```

我们还没有触及数据库。只有当我们使用会话对象时，它才会发生实际的变化：

```py
session.add(anakin)
session.add(obi1)
session.commit()
```

添加这两个`Person`实例足以添加它们的地址（这要归功于级联效应）。调用`commit`实际上告诉 SQLAlchemy 提交事务并将数据保存到数据库中。事务是在数据库上下文中提供类似沙盒的操作。只要事务没有提交，我们就可以回滚对数据库所做的任何修改，从而恢复到事务开始之前的状态。SQLAlchemy 提供了更复杂和细粒度的处理事务的方式，你可以在它的官方文档中学习，因为这是一个非常高级的话题。现在我们通过使用`like`查询所有以`Obi`开头的人的名字，这将连接到 SQL 中的`LIKE`操作符：

```py
obi1 = session.query(Person).filter(
    Person.name.like('Obi%')
).first()
print(obi1, obi1.addresses)
```

我们获取该查询的第一个结果（我们知道我们只有 Obi-Wan），然后打印出来。然后我们通过使用他的名字进行精确匹配来获取`anakin`（只是为了向你展示另一种过滤的方式）：

```py
anakin = session.query(Person).filter(
    Person.name=='Anakin Skywalker'
).first()
print(anakin, anakin.addresses)
```

然后我们捕获了 Anakin 的 ID，并从全局框架中删除了`anakin`对象：

```py
anakin_id = anakin.id
del anakin
```

我们这样做是因为我想向你展示如何通过 ID 获取对象。在我们这样做之前，我们编写了`display_info`函数，我们将使用它来显示数据库的全部内容（从地址开始获取，以演示如何通过使用 SQLAlchemy 中的关系属性来获取对象）：

```py
def display_info():
    # get all addresses first
    addresses = session.query(Address).all()

    # display results
    for address in addresses:
        print(f'{address.person.name} <{address.email}>')

    # display how many objects we have in total
    print('people: {}, addresses: {}'.format(
        session.query(Person).count(),
        session.query(Address).count())
    )
```

`display_info`函数打印出所有的地址，以及相应的人的名字，并且最后产生了关于数据库中对象数量的最终信息。我们调用这个函数，然后获取并删除`anakin`（想想*Darth Vader*，你就不会因为删除他而难过），然后再次显示信息，以验证他确实已经从数据库中消失了：

```py
display_info()

anakin = session.query(Person).get(anakin_id)
session.delete(anakin)
session.commit()

display_info()
```

所有这些片段一起运行的输出如下（为了方便起见，我已经将输出分成四个块，以反映实际产生该输出的四个代码块）：

```py
$ python alchemy.py
Obi-Wan Kenobi(id=2) [obi1@example.com, wanwan@example.com] 
Anakin Skywalker(id=1) [ani@example.com, evil.dart@example.com, vader@example.com]
 Anakin Skywalker <ani@example.com>
Anakin Skywalker <evil.dart@example.com>
Anakin Skywalker <vader@example.com>
Obi-Wan Kenobi <obi1@example.com>
Obi-Wan Kenobi <wanwan@example.com>
people: 2, addresses: 5
 Obi-Wan Kenobi <obi1@example.com>
Obi-Wan Kenobi <wanwan@example.com>
people: 1, addresses: 2
```

从最后两个代码块可以看出，删除`anakin`已经删除了一个`Person`对象和与之相关的三个地址。再次强调，这是因为在删除`anakin`时发生了级联。

这就结束了我们对数据持久性的简要介绍。这是一个广阔且有时复杂的领域，我鼓励您尽可能多地学习理论。在涉及数据库系统时，缺乏知识或适当的理解可能会带来真正的麻烦。

# 总结

在本章中，我们探讨了如何处理文件和目录。我们学会了如何打开文件进行读写，以及如何通过使用上下文管理器更加优雅地进行操作。我们还探讨了目录：如何递归和非递归地列出它们的内容。我们还学习了路径名，这是访问文件和目录的入口。

我们随后简要介绍了如何创建 ZIP 存档，并提取其内容。该书的源代码还包含了一个不同压缩格式的示例：`tar.gz`。

我们谈到了数据交换格式，并深入探讨了 JSON。我们在为特定的 Python 数据类型编写自定义编码器和解码器时玩得很开心。

然后，我们探讨了 IO，包括内存流和 HTTP 请求。

最后，我们看到了如何使用`pickle`、`shelve`和 SQLAlchemy ORM 库来持久化数据。

现在，您应该对如何处理文件和数据持久性有了相当好的了解，我希望您会花时间自己更深入地探索这些主题。

下一章将讨论测试、性能分析和处理异常。
