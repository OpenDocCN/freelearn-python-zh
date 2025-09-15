# 8

# 文件和数据持久化

> “不是因为我有多聪明，只是我愿意与问题相处更久。”
> 
> – 阿尔伯特·爱因斯坦

在前面的章节中，我们探讨了 Python 的几个不同方面。由于示例具有教学目的，我们在简单的 Python shell 或 Python 模块的形式下运行它们。它们运行，可能在控制台上打印了一些内容，然后终止，没有留下它们短暂存在的痕迹。

实际应用相当不同。自然地，它们仍然在内存中运行，但它们与网络、磁盘和数据库交互。它们还与其他应用程序和设备交换信息，使用适合该情况格式的格式。

在本章中，我们将通过探索以下内容来开始关注现实世界：

+   文件和目录

+   压缩

+   网络和流

+   JSON 数据交换格式

+   使用标准库中的`pickle`和`shelve`进行数据持久化

+   使用 SQLAlchemy 进行数据持久化

+   配置文件

如往常一样，我们将尝试平衡广度和深度，以便到本章结束时，你将牢固掌握基础知识，并知道如何从网络中获取更多信息。

# 与文件和目录一起工作

当涉及到文件和目录时，Python 提供了许多有用的工具。在下面的示例中，我们将使用`os`、`pathlib`和`shutil`模块。由于我们将要在磁盘上进行读写操作，我们将使用一个文件`fear.txt`作为一些示例的基础，该文件包含来自一行禅宗大师一行禅的《恐惧》摘录。

## 打开文件

在 Python 中打开文件简单直观。实际上，我们只需要使用`open()`函数。让我们看看一个快速示例：

```py
# files/open_try.py
fh = open("fear.txt", "rt")  # r: read, t: text
for line in fh.readlines():
    print(line.strip())  # remove whitespace and print
fh.close() 
```

之前的代码很简单。我们调用`open()`，传递文件名，并告诉`open()`我们想要以文本模式（通过`"rt"`标志）读取它。文件名之前没有路径信息；因此，`open()`将假设文件位于脚本运行的同一文件夹中。这意味着如果我们从这个`files`文件夹外部运行此脚本，那么`fear.txt`将找不到。

一旦文件被打开，我们就获得了一个文件对象，`fh`，我们可以用它来处理文件的内容。我们选择这个名字是因为，在 Python 中，文件对象本质上是一个高级抽象，它封装了底层的文件句柄（`fh`）。在这种情况下，我们使用`readlines()`方法遍历文件中的所有行并打印它们。我们对每一行调用`strip()`以去除内容周围的任何额外空格，包括行终止字符，因为`print()`已经为我们添加了一个。这是一个快速且简单的解决方案，在这个例子中有效，但如果文件内容包含需要保留的有意义的空格，你将不得不在清理数据时更加小心。在脚本末尾，我们关闭流。

关闭文件很重要，因为我们不希望冒无法释放我们对它的句柄（`fh`）的风险。当这种情况发生时，你可能会遇到内存泄漏或令人烦恼的 *“你不能删除此文件”* 弹窗，告诉你某些软件仍在使用它。

因此，我们需要采取一些预防措施，并将之前的逻辑包装在 `try/finally` 块中。这意味着无论我们尝试打开和读取文件时可能发生的任何错误，我们都可以确信 `close()` 会被调用：

```py
# files/open_try.py
fh = open("fear.txt", "rt")
try:
    for line in fh.readlines():
        print(line.strip())
finally:
    fh.close() 
```

逻辑是相同的，但现在它也是安全的。

如果你不太熟悉 `try` / `finally` 块，请确保你回到 *第七章* 的 *处理异常* 部分，*异常和上下文管理器*，并学习它。

我们可以进一步简化之前的例子，如下所示：

```py
# files/open_try.py
fh = open("fear.txt")  # rt is default
try:
    for line in fh:  # we can iterate directly on fh
        print(line.strip())
finally:
    fh.close() 
```

打开文件的默认模式是 `"rt"`，因此我们不需要指定它。此外，我们可以简单地迭代 `fh`，而不需要显式调用 `readlines()`。Python 经常给我们提供简写，使我们的代码更加紧凑且易于阅读。

所有的前一个例子都会在控制台上打印出文件的内容（查看源代码以读取全部内容）：

```py
An excerpt from Fear - By Thich Nhat Hanh
The Present Is Free from Fear
When we are not fully present, we are not really living. We are not
really there, either for our loved ones or for ourselves. If we are
not there, then where are we? We are running, running, running,
even during our sleep. We run because we are trying to escape from
our fear. […] 
```

### 使用上下文管理器打开文件

为了避免在代码中到处使用 `try` / `finally` 块，Python 给我们提供了一种更优雅且同样安全的做法：通过使用上下文管理器。让我们先看看代码：

```py
# files/open_with.py
with open("fear.txt") as fh:
    for line in fh:
        print(line.strip()) 
```

这个例子与上一个例子等效，但读起来更好。当通过上下文管理器调用时，`open()` 函数返回一个文件对象，并且当执行退出上下文管理器的作用域时，它会方便地自动调用 `fh.close()`。即使发生错误，也会发生这种情况。

## 从文件中读取和写入

现在我们知道了如何打开文件，让我们看看如何从文件中读取和写入：

```py
# files/print_file.py
with open("print_example.txt", "w") as fw:
    print("Hey I am printing into a file!!!", file=fw) 
```

这种第一种方法使用 `print()` 函数，我们已经在之前的章节中熟悉了它。在获得文件对象后，这次指定我们打算写入它（`"w"`），我们可以告诉 `print()` 调用将输出定向到文件，而不是像通常那样定向到 **标准输出** 流。

在 Python 中，标准输入、输出和错误流由文件对象 `sys.stdin`、`sys.stdout` 和 `sys.stderr` 表示。除非输入或输出被重定向，否则从 `sys.stdin` 读取通常对应于从键盘读取，而将内容写入 `sys.stdout` 或 `sys.stderr` 通常会在控制台屏幕上打印。

之前的代码会在文件不存在时创建 `print_example.txt` 文件，或者如果它已存在，则截断它，并将行 `Hey I am printing into a file!!!` 写入其中。

截断文件意味着在不删除文件的情况下擦除其内容。截断后，文件仍然存在于文件系统中，但它为空。

这个例子完成了工作，但这并不是我们写入文件时通常会做的事情。让我们看看一种更常见的方法：

```py
# files/read_write.py
with open("fear.txt") as f:
    lines = [line.rstrip() for line in f]
with open("fear_copy.txt", "w") as fw:  # w - write
    fw.write("\n".join(lines)) 
```

在这个例子中，我们首先打开 `fear.txt` 并逐行收集其内容到一个列表中。注意，这次我们调用了一个不同的方法 `rstrip()`，作为一个例子，以确保我们只去除每行的右侧空白。

在代码片段的第二部分，我们创建了一个新文件 `fear_copy.txt`，并将 `lines` 中的所有字符串写入其中，通过换行符 `\n` 连接。Python 默认使用 **通用换行符**，这意味着即使原始文件可能有与 `\n` 不同的换行符，它也会在我们返回行之前自动为我们转换。这种行为当然是可以定制的，但通常这正是我们想要的。说到换行符，你能想到在复制中可能缺失的一个吗？

### 以二进制模式读写

注意，通过打开一个文件并传递 `t` 选项（或者省略它，因为它默认是这样），我们是以文本模式打开文件的。这意味着文件的内容被当作文本处理和解释。

如果你希望向文件写入字节，你可以以 **二进制模式** 打开它。当你处理不包含纯文本的文件时，这是一个常见的需求，例如图像、音频/视频，以及通常的任何其他专有格式。

要以二进制模式处理文件，只需在打开时指定 `b` 标志，如下面的例子所示：

```py
# files/read_write_bin.py
with open("example.bin", "wb") as fw:
    fw.write(b"This is binary data...")
with open("example.bin", "rb") as f:
    print(f.read())  # prints: b'This is binary data...' 
```

在这个例子中，我们仍然使用文本作为二进制数据，为了简单起见，但它可以是任何你想要的东西。你可以看到它被当作二进制处理，因为输出字符串中有 `b` 前缀。

### 防止覆盖现有文件

正如我们所见，Python 给我们提供了打开文件进行写入的能力。通过使用 `w` 标志，我们打开一个文件并截断其内容。这意味着文件被一个空文件覆盖，原始内容丢失。如果你希望只有在文件不存在时才打开文件进行写入，你可以使用 `x` 标志，如下面的例子所示：

```py
# files/write_not_exists.py
with open("write_x.txt", "x") as fw:  # this succeeds
    fw.write("Writing line 1")
with open("write_x.txt", "x") as fw:  # this fails
    fw.write("Writing line 2") 
```

如果你运行这个代码片段，你将在你的目录中找到一个名为 `write_x.txt` 的文件，其中只包含一行文本。实际上，代码片段的第二部分未能执行。这是我们控制台上的输出（为了编辑目的，文件路径已被缩短）：

```py
$ python write_not_exists.py
Traceback (most recent call last):
  File "write_not_exists.py", line 6, in <module>
    with open("write_x.txt", "x") as fw:  # this fails
         ^^^^^^^^^^^^^^^^^^^^^^^^
FileExistsError: [Errno 17] File exists: 'write_x.txt' 
```

正如我们所见，打开文件有不同的模式。你可以在 [`docs.python.org/3/library/functions.html#open`](https://docs.python.org/3/library/functions.html#open) 找到完整的标志列表。

## 检查文件和目录是否存在

如果你想要确保一个文件或目录存在（或者不存在），你需要使用 `pathlib` 模块。让我们看一个小例子：

```py
# files/existence.py
from pathlib import Path
p = Path("fear.txt")
path = p.parent.absolute()
print(p.is_file())  # True
print(path)  # /Users/fab/code/lpp4ed/ch08/files
print(path.is_dir())  # True
q = Path("/Users/fab/code/lpp4ed/ch08/files")
print(q.is_dir())  # True 
```

在前面的代码片段中，我们创建了一个 `Path` 对象，我们用要检查的文本文件的名字来设置它。我们使用 `parent()` 方法来检索包含文件的文件夹，并对其调用 `absolute()` 方法以提取绝对路径信息。

我们检查`"fear.txt"`是否是一个文件，以及它所在的文件夹确实是一个文件夹（或目录，两者等价）。

以前执行这些操作的方法是使用标准库中的`os.path`模块。虽然`os.path`在字符串上工作，但`pathlib`提供了表示文件系统路径的类，具有适用于不同操作系统的语义。因此，我们建议尽可能使用`pathlib`，只有在没有其他选择的情况下才回退到旧的方法。

## 文件和目录操作

让我们看看几个快速示例，说明如何操作文件和目录。第一个示例操作内容：

```py
# files/manipulation.py
from collections import Counter
from string import ascii_letters
chars = ascii_letters + " "
def sanitize(s, chars):
    return "".join(c for c in s if c in chars)
def reverse(s):
    return s[::-1]
with open("fear.txt") as stream:
    lines = [line.rstrip() for line in stream]
# let us write the mirrored version of the file
with open("raef.txt", "w") as stream:
    stream.write("\n".join(reverse(line) for line in lines))
# now we can calculate some statistics
lines = [sanitize(line, chars) for line in lines]
whole = " ".join(lines)
# we perform comparisons on the lowercased version of `whole`
cnt = Counter(whole.lower().split())
# we can print the N most common words
print(cnt.most_common(3)) # [('we', 17), ('the', 13), ('were', 7)] 
```

此示例定义了两个函数：`sanitize()`和`reverse()`。它们是简单的函数，其目的是从一个字符串中移除所有非字母或空格的字符，并分别产生字符串的反转副本。

我们打开`fear.txt`并将其内容读取到一个列表中。然后我们创建一个新的文件`raef.txt`，它将包含原始的横向镜像版本。我们通过在换行符上使用`join`操作，一次性写入`lines`中的所有内容。也许更有趣的是结尾的部分。首先，我们通过列表推导式将`lines`重新赋值为它的净化版本。然后我们将这些行组合成一个`whole`字符串，最后，我们将结果传递给一个`Counter`对象。请注意，我们将字符串的小写版本拆分成一个单词列表。这样，每个单词都会被正确计数，无论其大小写如何，而且，多亏了`split()`，我们不需要担心任何地方的额外空格。当我们打印最常见的三个单词时，我们意识到，确实，一行禅的焦点是他人，因为*“我们”*是文本中最常见的单词：

```py
$ python manipulation.py
[('we', 17), ('the', 13), ('were', 7)] 
```

现在我们来看一个更接近磁盘操作的示例，我们将使用`shutil`模块：

```py
# files/ops_create.py
import shutil
from pathlib import Path
base_path = Path("ops_example")
# let us perform an initial cleanup just in case
if base_path.exists() and base_path.is_dir():
    shutil.rmtree(base_path)
# now we create the directory
base_path.mkdir()
path_b = base_path / "A" / "B"
path_c = base_path / "A" / "C"
path_d = base_path / "A" / "D"
path_b.mkdir(parents=True)
path_c.mkdir()  # no need for parents now, as 'A' has been created
# we add three files in `ops_example/A/B`
for filename in ("ex1.txt", "ex2.txt", "ex3.txt"):
    with open(path_b / filename, "w") as stream:
        stream.write(f"Some content here in {filename}\n")
shutil.move(path_b, path_d)
# we can also rename files
ex1 = path_d / "ex1.txt"
ex1.rename(ex1.parent / "ex1.renamed.txt") 
```

在前面的代码中，我们首先声明了一个基础路径，该路径将包含我们即将创建的所有文件和文件夹。然后，我们使用`mkdir()`创建两个目录：`ops_example/A/B`和`ops_example/A/C`。请注意，在调用`path_c.mkdir()`时，我们不需要指定`parents=True`，因为所有父目录都已经在之前的`path_b`调用中创建好了。

我们使用`/`运算符来连接目录名称；`pathlib`会为我们处理背后的正确路径分隔符。

在创建目录后，我们在目录`B`中循环创建三个文件。然后，我们将目录`B`及其内容移动到不同的名称：`D`。我们也可以用另一种方式来做这件事：`path_b.rename(path_d)`。

最后，我们将`ex1.txt`重命名为`ex1.renamed.txt`。如果你打开该文件，你会看到它仍然包含循环逻辑中的原始文本。对结果调用`tree`会产生以下内容：

```py
$ tree ops_example
ops_example
└── A
    ├── C
    └── D
        ├── ex1.renamed.txt
        ├── ex2.txt
        └── ex3.txt 
```

### 操作路径名

让我们通过一个示例来更深入地探索`pathlib`的能力：

```py
# files/paths.py
from pathlib import Path
p = Path("fear.txt")
print(p.absolute())
print(p.name)
print(p.parent.absolute())
print(p.suffix)
print(p.parts)
print(p.absolute().parts)
readme_path = p.parent / ".." / ".." / "README.rst"
print(readme_path.absolute())
print(readme_path.resolve()) 
```

阅读结果可能是对这个简单例子足够好的解释：

```py
$ python paths.py
/Users/fab/code/lpp4ed/ch08/files/fear.txt
fear.txt
/Users/fab/code/lpp4ed/ch08/files
.txt
('fear.txt',)
(
    '/', 'Users', 'fab', 'code', 'lpp4ed',
    'ch08', 'files', 'fear.txt'
)
/Users/fab/code/lpp4ed/ch08/files/../../README.rst
/Users/fab/code/lpp4ed/README.rst 
```

注意，在最后两行中，我们有同一路径的两种不同表示。第一个（`readme_path.absolute()`）显示了两个 `"` `.."`，每个在路径术语中都表示切换到父文件夹。因此，通过连续两次切换到父文件夹，从 `…/lpp4e/ch08/files/`，我们回到了 `…/lpp4e/`。这由示例中的最后一行确认，它显示了 `readme_path.resolve()` 的输出。

## 临时文件和目录

有时候，创建一个临时目录或文件是有用的。例如，当编写影响磁盘的测试时，你可以使用临时文件和目录来运行你的逻辑并断言它是正确的，并且确保在测试运行结束时，测试文件夹没有遗留物。让我们看看如何在 Python 中做到这一点：

```py
# files/tmp.py
from tempfile import NamedTemporaryFile, TemporaryDirectory
with TemporaryDirectory(dir=".") as td:
    print("Temp directory:", td)
    with NamedTemporaryFile(dir=td) as t:
        name = t.name
        print(name) 
```

前面的例子相当简单：我们在当前目录（`"."`）中创建一个临时目录，并在其中创建一个命名的临时文件。我们打印文件名，以及它的完整路径：

```py
$ python tmp.py
Temp directory: /Users/fab/code/lpp4ed/ch08/files/tmpqq4quhbc
/Users/fab/code/lpp4ed/ch08/files/tmpqq4quhbc/tmpypwwhpwq 
```

运行这个脚本将每次产生不同的结果，因为这些是临时的随机名称。

## 目录内容

使用 Python，你还可以检查目录的内容。我们将向你展示两种方法。这是第一种：

```py
# files/listing.py
from pathlib import Path
p = Path(".")
for entry in p.glob("*"):
    print("File:" if entry.is_file() else "Folder:", entry) 
```

这个代码片段使用了 `Path` 对象的 `glob()` 方法，从当前目录应用。我们遍历结果，每个结果都是一个 `Path` 子类的实例（`PosixPath` 或 `WindowsPath`，根据我们运行的操作系统）。对于每个 `entry`，我们检查它是否是目录，并相应地打印。运行代码将产生以下结果（为了简洁，我们省略了一些结果）：

```py
$ python listing.py
File: existence.py
File: manipulation.py
…
File: open_try.py
File: walking.pathlib.py 
```

另一种方法是使用 `Path.walk()` 方法来扫描目录树。让我们看一个例子：

```py
# files/walking.pathlib.py
from pathlib import Path
p = Path(".")
for root, dirs, files in p.walk():
    print(f"{root=}")
    if dirs:
        print("Directories:")
        for dir_ in dirs:
            print(dir_)
        print()
    if files:
        print("Files:")
        for filename in files:
            print(filename)
        print() 
```

运行前面的代码片段将生成当前目录中所有文件和目录的列表，并且它将为每个子目录做同样的事情。在本书的源代码中，你会找到一个名为 `walking.py` 的模块，它做的是完全相同的事情，但使用的是 `os.walk()` 函数。

## 文件和目录压缩

在我们离开这个部分之前，让我们给你一个如何创建压缩文件的例子。在本章的源代码中，在 `files/compression` 文件夹中，我们有两个例子：一个创建 `.zip` 文件，而另一个创建 `tar.gz` 文件。Python 允许你以多种不同的方式和格式创建压缩文件。在这里，我们将向你展示如何创建最常见的一种，**ZIP**：

```py
# files/compression/zip.py
from zipfile import ZipFile
with ZipFile("example.zip", "w") as zp:
    zp.write("content1.txt")
    zp.write("content2.txt")
    zp.write("subfolder/content3.txt")
    zp.write("subfolder/content4.txt")
with ZipFile("example.zip") as zp:
    zp.extract("content1.txt", "extract_zip")
    zp.extract("subfolder/content3.txt", "extract_zip") 
```

在前面的代码中，我们导入`ZipFile`，然后在上下文管理器中写入四个文件（其中两个位于子文件夹中，以展示 ZIP 如何保留完整路径）。之后，作为一个例子，我们打开压缩文件，从中提取一些文件到`extract_zip`目录。如果您对数据压缩感兴趣，请确保查看标准库中的*数据压缩和归档*部分（[`docs.python.org/3.9/library/archiving.html`](https://docs.python.org/3.9/library/archiving.html)），在那里您可以了解有关此主题的所有内容。

# 数据交换格式

现代软件架构倾向于将应用程序拆分为几个组件。无论您是采用面向服务的架构范式，还是将其进一步推进到微服务领域，这些组件都必须要交换数据。但即使您正在编写一个单体应用程序，其代码库包含在一个项目中，您仍然可能需要与 API 或程序交换数据，或者简单地处理网站的前端和后端部分之间的数据流，这些部分可能不会使用相同的语言。

选择正确的信息交换格式至关重要。语言特定的格式具有优势，因为该语言本身很可能为您提供所有工具，使**序列化**和**反序列化**变得轻而易举。然而，您将无法与用同一语言的不同版本或完全不同的语言编写的其他组件进行本地通信。无论未来看起来如何，只有在是给定情况下唯一可能的选择时，才应该采用语言特定的格式。

根据维基百科（[`en.wikipedia.org/wiki/Serialization`](https://en.wikipedia.org/wiki/Serialization)）：

> 在计算机科学中，序列化是将数据结构或对象状态转换为一种可以存储（例如，在文件或内存数据缓冲区中）或传输（例如，通过计算机网络）的格式，并在以后重建（可能在不同的计算机环境中）的过程。

一种更安全的方法是选择一种语言无关的格式。在软件中，一些流行的格式已经成为数据交换的事实标准。最著名的可能是**XML**、**YAML**和**JSON**。Python 标准库提供了`xml`和`json`模块，在 PyPI（[`pypi.org/`](https://pypi.org/)）上，您可以找到一些用于处理 YAML 的不同包。

在 Python 环境中，JSON 可能是最常用的格式。它之所以胜过其他两种，是因为它是标准库的一部分，以及它的简单性。XML 往往相当冗长，难以阅读。

此外，当与像 PostgreSQL 这样的数据库一起工作时，能够使用原生 JSON 字段的能力使得在应用程序中也采用 JSON 具有很大的吸引力。

## 使用 JSON

**JSON**是**JavaScript Object Notation**的缩写，它是 JavaScript 语言的一个子集。它已经存在了近二十年，因此它广为人知，并被大多数语言广泛采用，尽管它实际上是语言无关的。你可以在其网站上阅读所有关于它的信息（[`www.json.org/`](https://www.json.org/)），但现在我们将给你一个快速介绍。

JSON 基于两种结构：

+   一组名称/值对

+   值的有序列表

毫不奇怪，这两个对象分别映射到 Python 中的`dict`和`list`数据类型。作为数据类型，JSON 提供字符串、数字、对象以及由`true`、`false`和`null`组成的值。让我们通过一个快速示例开始：

```py
# json_examples/json_basic.py
import sys
import json
data = {
    "big_number": 2**3141,
    "max_float": sys.float_info.max,
    "a_list": [2, 3, 5, 7],
}
json_data = json.dumps(data)
data_out = json.loads(json_data)
assert data == data_out  # json and back, data matches 
```

我们首先导入`sys`和`json`模块。然后，我们创建一个包含一些数字和一个整数列表的简单字典。我们想测试使用非常大的数字进行序列化和反序列化，包括`int`和`float`，所以我们放入了 2 的 3141 次方以及系统可以处理的最大浮点数。

我们使用`json.dumps()`进行序列化，它将数据转换为 JSON 格式的字符串。然后，该数据被输入到`json.loads()`中，它执行相反的操作：从一个 JSON 格式的字符串中，它将数据重构为 Python。

注意，JSON 模块还提供了`dump`和`load`函数，它们可以将数据转换为文件对象并从文件对象转换数据。

在最后一行，通过断言，我们确保原始数据和通过 JSON 序列化/反序列化的结果相匹配。如果断言语句后面的条件为假，那么该语句将引发`AssertionError`。我们将在第十章*测试*中更详细地介绍断言。

在编程中，术语**falsy**指的是在布尔上下文中评估时被认为是假的对象或条件。

让我们看看如果我们打印 JSON 数据会是什么样子：

```py
# json_examples/json_basic.py
info = {
    "full_name": "Sherlock Holmes",
    "address": {
        "street": "221B Baker St",
        "zip": "NW1 6XE",
        "city": "London",
        "country": "UK",
    },
}
print(json.dumps(info, indent=2, sort_keys=True)) 
```

在这个例子中，我们创建了一个包含福尔摩斯数据的字典。如果你像我们一样是福尔摩斯的粉丝，并且身处伦敦，你会在那个地址找到他的博物馆（我们推荐你去参观；虽然不大，但非常不错）。

注意我们是如何调用`json.dumps()`的。我们指示它使用两个空格缩进并按字母顺序排序键。结果是这个：

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

与 Python 的相似性显而易见。唯一的区别是，如果你在字典中的最后一个元素后面放置一个逗号，这在 Python 中是惯例，JSON 将会抱怨。

让我们展示一些有趣的东西：

```py
# json_examples/json_tuple.py
import json
data_in = {
    "a_tuple": (1, 2, 3, 4, 5),
}
json_data = json.dumps(data_in)
print(json_data)  # {"a_tuple": [1, 2, 3, 4, 5]}
data_out = json.loads(json_data)
print(data_out)  # {'a_tuple': [1, 2, 3, 4, 5]} 
```

在这个例子中，我们使用了一个元组而不是列表。有趣的是，从概念上讲，元组也是一个有序项列表。它没有列表的灵活性，但仍然，从 JSON 的角度来看，它被认为是相同的。因此，正如你通过第一个`print()`看到的，在 JSON 中元组被转换成了列表。自然地，那么，原始对象是一个元组的信息就丢失了，在反序列化发生时，原本是元组的东西被转换成了 Python 列表。在处理数据时，这一点很重要，因为涉及到只包含你可用数据结构子集的格式转换过程可能意味着信息丢失。在这种情况下，我们丢失了关于类型（元组与列表）的信息。

这实际上是一个常见问题。例如，你不能将所有 Python 对象序列化为 JSON，因为并不总是清楚 JSON 应该如何还原那个对象。以`datetime`为例。该类的一个实例是一个 JSON 无法序列化的 Python 对象。如果我们将其转换为如`2018-03-04T12:00:30Z`这样的字符串，这是 ISO 8601 格式的日期和时间以及时区信息，那么在反序列化时 JSON 应该怎么做？它应该决定*这可以反序列化为 datetime 对象，所以我最好这么做*，还是简单地将其视为字符串并保持原样？对于可以有多种解释的数据类型呢？

答案是，在处理数据交换时，我们通常需要在将对象序列化为 JSON 之前将其转换为更简单的格式。我们能够使数据简化得越多，在像 JSON 这样的格式中表示数据就越容易，而 JSON 有其局限性。

在某些情况下，尤其是内部使用时，能够序列化自定义对象非常有用，所以为了好玩，我们将通过两个例子来展示如何实现：复数和*datetime*对象。

### 使用 JSON 进行自定义编码/解码

在 JSON 的世界里，我们可以将编码/解码术语视为序列化/反序列化的同义词。它们基本上意味着转换到和从 JSON 转换回来。

在下面的例子中，我们将学习如何通过编写自定义编码器来编码复数——默认情况下复数不能序列化为 JSON：

```py
# json_examples/json_cplx.py
import json
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        print(f"ComplexEncoder.default: {obj=}")
        if isinstance(obj, complex):
            return {
                "_meta": "complex",
                "num": [obj.real, obj.imag],
            }
        return super().default(obj)
data = {
    "an_int": 42,
    "a_float": 3.14159265,
    "a_complex": 3 + 4j,
}
json_data = json.dumps(data, cls=ComplexEncoder)
print(json_data)
def object_hook(obj):
    print(f"object_hook: {obj=}")
    try:
        if obj["_meta"] == "complex":
            return complex(*obj["num"])
    except KeyError:
        return obj
data_out = json.loads(json_data, object_hook=object_hook)
print(data_out) 
```

我们首先定义一个`ComplexEncoder`类，作为`JSONEncoder`的子类。这个类重写了`default()`方法。每当编码器遇到它无法原生编码的对象时，都会调用这个方法，并期望它返回该对象的可编码表示。

`default()` 方法检查其参数是否是一个 `complex` 对象，如果是的话，它将返回一个包含一些自定义元信息和包含数字实部和虚部的列表的字典。这就是我们避免丢失复数信息所需做的全部工作。如果我们收到除 `complex` 实例之外的其他任何内容，我们将从父类调用 `default()` 方法。

在示例中，我们随后调用了 `json.dumps()`，但这次我们使用 `cls` 参数来指定自定义编码器。最后，结果被打印出来：

```py
$ python json_cplx.py
ComplexEncoder.default: obj=(3+4j)
{
    "an_int": 42, "a_float": 3.14159265,
    "a_complex": {"_meta": "complex", "num": [3.0, 4.0]}
} 
```

有一半的工作已经完成了。对于反序列化部分，我们本可以编写另一个从 `JSONDecoder` 继承的类，但相反，我们选择使用一种更简单的技术，它使用一个小的函数：`object_hook()`。

在 `object_hook()` 的主体中，我们找到一个 `try` 块。重要的是 `try` 块主体中的两行。该函数接收一个对象（注意，只有当 `obj` 是字典时，该函数才会被调用），如果元数据与我们的复数约定相匹配，我们将实部和虚部传递给 `complex()` 函数。`try` / `except` 块的存在是因为我们的函数将为每个解码的字典对象被调用，因此我们需要处理 `_meta` 键不存在的情况。

示例的反序列化部分输出：

```py
object_hook:
  obj={'_meta': 'complex', 'num': [3.0, 4.0]}
object_hook:
  obj={'an_int': 42, 'a_float': 3.14159265, 'a_complex': (3+4j)}
{'an_int': 42, 'a_float': 3.14159265, 'a_complex': (3+4j)} 
```

你可以看到 `a_complex` 已经被正确反序列化。作为练习，我们建议编写你自己的自定义编码器，用于 `Fraction` 和 `Decimal` 对象。

让我们考虑一个稍微复杂一些（不是字面意义上的）例子：处理 `datetime` 对象。我们将把代码分成两个部分，首先是序列化部分，然后是反序列化部分：

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
                "_meta": "datetime",
                "data": obj.timetuple()[:6] + (obj.microsecond,),
                "utcoffset": off,
            }
        return super().default(obj)
data = {
    "an_int": 42,
    "a_float": 3.14159265,
    "a_datetime": now,
    "a_datetime_tz": now_tz,
}
json_data = json.dumps(data, cls=DatetimeEncoder)
print(json_data) 
```

这个例子之所以稍微复杂一些，是因为 Python 中的 `datetime` 对象可以是时区感知的，也可以不是；因此，我们需要小心处理它们。流程与之前相同，只是我们现在处理的是不同的数据类型。我们首先获取当前的日期和时间信息，并且我们在没有（`now`）和有（`now_tz`）时区感知的情况下都这样做。然后我们继续定义一个自定义编码器，就像之前一样，覆盖了 `default()` 方法。该方法中的重要部分是我们如何获取时区偏移量（`off`）信息（以秒为单位），以及我们如何构建返回数据的字典。这次，元数据表明这是 *datetime* 信息。我们将时间元组的前六项（年、月、日、时、分和秒）保存在 `data` 键中，以及之后的微秒，然后是偏移量。如果你能看出 `"data"` 的值是元组的拼接，那么你做得很好。

在自定义编码器之后，我们继续创建一些数据，然后对其进行序列化。`print()` 语句输出了以下内容（我们已重新格式化输出，使其更易于阅读）：

```py
$ python json_datetime.py
{
    "an_int": 42,
    "a_float": 3.14159265,
    "a_datetime": {
        "_meta": "datetime",
        "data": [2024, 3, 29, 23, 24, 22, 232302],
        "utcoffset": null,
    },
    "a_datetime_tz": {
        "_meta": "datetime",
        "data": [2024, 3, 30, 0, 24, 22, 232316],
        "utcoffset": 3600,
    },
} 
```

有趣的是，我们发现 `None` 被翻译成了其 JavaScript 等价物 `null`。此外，我们可以看到数据似乎已经被正确编码。让我们继续脚本的第二部分：

```py
# json_examples/json_datetime.py
def object_hook(obj):
    try:
        if obj["_meta"] == "datetime":
            if obj["utcoffset"] is None:
                tz = None
            else:
                tz = timezone(timedelta(seconds=obj["utcoffset"]))
            return datetime(*obj["data"], tzinfo=tz)
    except KeyError:
        return obj
data_out = json.loads(json_data, object_hook=object_hook)
print(data_out) 
```

再次，我们首先验证元数据告诉我们它是一个 `datetime`，然后我们继续获取时区信息。一旦我们有了它，我们就将 7 元组（使用 `*` 在调用中解包其值）和时区信息传递给 `datetime()` 调用，得到我们原始的对象。让我们通过打印 `data_out` 来验证它：

```py
{
    "an_int": 42,
    "a_float": 3.14159265,
    "a_datetime": datetime.datetime(
        2024, 3, 29, 23, 24, 22, 232302
    ),
    "a_datetime_tz": datetime.datetime(
        2024, 3, 30, 0, 24, 22, 232316,
        tzinfo=datetime.timezone(
            datetime.timedelta(seconds=3600)
        ),
    ),
} 
```

正如你所见，我们正确地获取了所有内容。作为练习，我们建议你编写相同的逻辑，但针对 `date` 对象，这应该会简单一些。

在我们继续下一个主题之前，有一个警告。可能这听起来有些反直觉，但处理 `datetime` 对象可能相当棘手，所以我们虽然相当确信这段代码正在做它应该做的事情，但我们想强调我们只是对其进行了表面测试。因此，如果你打算使用它，请务必彻底测试。测试不同的时区，测试夏令时是否开启或关闭，测试纪元之前的日期，等等。你可能会发现本节中的代码需要一些修改才能适应你的情况。

# I/O、流和请求

**I/O** 代表 **输入/输出**，它广泛地指代计算机与外部世界之间的通信。有几种不同的 I/O 类型，本章的范围不包括解释所有这些类型，但值得通过几个例子来了解。第一个例子将介绍 `io.StringIO` 类，这是一个用于文本 I/O 的内存流。第二个例子将超出我们计算机的本地性，演示如何执行 HTTP 请求。

## 使用内存中的流

内存中的对象在多种情况下都可能很有用。内存比硬盘快得多，它总是可用，对于少量数据来说可能是完美的选择。

让我们看看第一个例子：

```py
# io_examples/string_io.py
import io
stream = io.StringIO()
stream.write("Learning Python Programming.\n")
print("Become a Python ninja!", file=stream)
contents = stream.getvalue()
print(contents)
stream.close() 
```

在前面的代码片段中，我们从标准库中导入了 `io` 模块。这个模块包含了许多与流和 I/O 相关的工具。其中之一是 `StringIO`，它是一个内存缓冲区，我们在其中使用了两种不同的方法写入了两个句子，就像我们在本章的第一个例子中使用文件一样。

当你需要时，`StringIO` 很有用：

+   模拟字符串的文件-like 行为。

+   测试与文件对象一起工作的代码，而不使用实际文件。

+   高效地构建或操作大字符串。

+   为了测试目的捕获或模拟输入/输出。测试运行得更快，因为它们避免了磁盘 I/O。

我们可以调用 `StringIO.write()`，或者我们可以使用 `print()`，指示它将数据导向我们的流。

通过调用 `getvalue()`，我们可以获取流的内 容。然后我们继续打印它，最后关闭它。调用 `close()` 会导致文本缓冲区立即被丢弃。

有一种更优雅的方式来编写之前的代码：

```py
# io_examples/string_io.py
with io.StringIO() as stream:
    stream.write("Learning Python Programming.\n")
    print("Become a Python ninja!", file=stream)
    contents = stream.getvalue()
    print(contents) 
```

就像内置的 `open()` 函数一样，`io.StringIO()` 也在上下文管理器块中工作得很好。注意与 `open()` 的相似性；在这种情况下，我们也不需要手动关闭流。

当运行脚本时，输出如下：

```py
$ python string_io.py
Learning Python Programming.
Become a Python ninja! 
```

现在让我们继续第二个示例。

## 发送 HTTP 请求

在本节中，我们探讨了两个 HTTP 请求的例子。我们将使用 `requests` 库来演示这些例子，你可以使用 `pip` 安装它，并且它包含在本章节的要求文件中。

我们将向 httpbin.org（[`httpbin.org/`](https://httpbin.org/)）API 发起 HTTP 请求，有趣的是，这个 API 是由 `requests` 库的创建者 Kenneth Reitz 开发的。Httpbin 是一个简单的 HTTP 请求和响应服务，当我们想要实验 HTTP 协议时非常有用。

这个库是最广泛采用的之一：

```py
# io_examples/reqs.py
import requests
urls = {
    "get": "https://httpbin.org/get?t=learn+python+programming",
    "headers": "https://httpbin.org/headers",
    "ip": "https://httpbin.org/ip",
    "user-agent": "https://httpbin.org/user-agent",
    "UUID": "https://httpbin.org/uuid",
    "JSON": "https://httpbin.org/json",
}
def get_content(title, url):
    resp = requests.get(url)
    print(f"Response for {title}")
    print(resp.json())
for title, url in urls.items():
    get_content(title, url)
    print("-" * 40) 
```

上述代码片段应该很简单。我们声明了一个字典，其中包含了我们想要对其发起 HTTP 请求的 URL。我们将执行请求的代码封装到了 `get_content()` 函数中。正如你所看到的，我们执行了一个 GET 请求（通过使用 `requests.get()` ），并打印了响应的标题和 JSON 解码后的响应体。让我们花点时间来谈谈最后这部分。

当我们向一个网站或 API 发起请求时，我们会收到一个响应对象，该对象封装了服务器返回的数据。`httpbin.org` 的一些响应体恰好是 JSON 编码的，因此我们不是直接读取 `resp.text` 并手动调用 `json.loads()` 来解码，而是通过使用响应对象的 `json()` 方法将两者结合起来。`requests` 包之所以被广泛采用，有很多原因，其中之一就是它的易用性。

现在，当你在你应用程序中发起请求时，你将希望有一个更健壮的方法来处理错误等，但在这个章节中，一个简单的例子就足够了。我们将在 *第十四章* ，*API 开发简介* 中看到更多请求的例子。

回到我们的代码，最后我们运行一个 `for` 循环并获取所有 URL。当你运行它时，你将在控制台上看到每个调用的结果打印出来，它应该看起来像这样（为了简洁而进行了美化并裁剪）：

```py
$ python reqs.py
Response for get
{
    "args": {"t": "learn python programming"},
    "headers": {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Host": "httpbin.org",
        "User-Agent": "python-requests/2.31.0",
        "X-Amzn-Trace-Id": "Root=1-123abc-123abc",
    },
    "origin": "86.14.44.233",
    "url": "https://httpbin.org/get?t=learn+python+programming",
}
… rest of the output omitted … 
```

注意，你可能会得到一些关于版本号和 IP 的不同输出，这是正常的。现在，`GET` 只是 HTTP 动词之一，尽管是最常用的之一。让我们也看看如何使用 `POST` 动词。当你需要向服务器发送数据时，例如请求创建资源，你会发起一个 `POST` 请求。所以，让我们尝试通过编程来发起一个请求：

```py
# io_examples/reqs_post.py
import requests
url = "https://httpbin.org/post"
data = dict(title="Learn Python Programming")
resp = requests.post(url, data=data)
print("Response for POST")
print(resp.json()) 
```

前面的代码与我们之前看到的非常相似，只是这次我们没有调用`get()`，而是调用`post()`，因为我们想发送一些数据，我们在调用中指定了这一点。`requests`库提供了比这更多的功能。这是一个我们鼓励你检查和探索的项目，因为它很可能你也会用到它。

运行前面的脚本（并对输出应用一些美化魔法）会产生以下结果：

```py
$ python reqs_post.py
Response for POST
{
    "args": {},
    "data": "",
    "files": {},
    "form": {"title": "Learn Python Programming"},
    "headers": {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Content-Length": "30",
        "Content-Type": "application/x-www-form-urlencoded",
        "Host": "httpbin.org",
        "User-Agent": "python-requests/2.31.0",
        "X-Amzn-Trace-Id": "Root=1-123abc-123abc",
    },
    "json": None,
    "origin": "86.14.44.233",
    "url": "https://httpbin.org/post",
} 
```

注意现在头部已经不同了，我们找到了以响应体的键/值对形式发送的数据。

我们希望这些简短的例子足以让你开始，特别是对于请求。网络每天都在变化，因此学习基础知识并时不时地复习是值得的。

# 在磁盘上持久化数据

在本章的这一节中，我们将探讨如何以三种不同的格式在磁盘上持久化数据。持久化数据意味着数据被写入非易失性存储，例如硬盘驱动器，并且当写入它的进程结束其生命周期时，数据不会被删除。我们将探讨`pickle`和`shelve`模块，以及一个简短的例子，该例子将涉及使用**SQLAlchemy**访问数据库，SQLAlchemy 可能是 Python 生态系统中最广泛采用的 ORM 库。

## 使用 pickle 序列化数据

Python 标准库中的`pickle`模块提供了将 Python 对象转换为字节流以及相反的工具。尽管`pickle`和`json`暴露的 API 有部分重叠，但这两个模块相当不同。正如我们在本章前面所见，JSON 是一种人类可读的文本格式，语言无关，仅支持 Python 数据类型的一个受限子集。另一方面，`pickle`模块不是人类可读的，转换为字节，是 Python 特定的，并且，多亏了 Python 出色的内省能力，支持大量数据类型。

除了`pickle`和`json`之间的这些差异之外，还有一些重要的安全问题需要你注意，如果你考虑使用`pickle`的话。从不受信任的来源*反序列化*错误或恶意数据可能是危险的，因此如果我们决定在我们的应用程序中采用它，我们需要格外小心。

如果你确实使用`pickle`，你应该考虑使用加密签名来确保你的序列化数据没有被篡改。我们将在*第九章*，*密码学和令牌*中看到如何在 Python 中生成加密签名。

话虽如此，让我们通过一个简单的例子来看看它的实际应用：

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
        print(
            f"Hi, I am {self.first_name} {self.last_name}"
            f" and my ID is {self.id}"
        )
people = [
    Person("Obi-Wan", "Kenobi", 123),
    Person("Anakin", "Skywalker", 456),
]
# save data in binary format to a file
with open("data.pickle", "wb") as stream:
    pickle.dump(people, stream)
# load data from a file
with open("data.pickle", "rb") as stream:
    peeps = pickle.load(stream)
for person in peeps:
    person.greet() 
```

在这个例子中，我们使用`dataclass`装饰器创建了一个`Person`类，这在*第六章*，*面向对象编程、装饰器和迭代器*中我们见过。我们之所以用`dataclass`写这个例子，只是为了向你展示`pickle`处理它有多么轻松，我们不需要为简单数据类型做任何额外的事情。

这个类有三个属性：`first_name`、`last_name`和`id`。它还公开了一个`greet()`方法，该方法使用实例数据打印一条问候消息。

我们创建一个实例列表并将其保存到文件中。为此，我们使用`pickle.dump()`，向其中提供要*序列化*的内容，以及我们想要写入的流。紧接着，我们使用`pickle.load()`从同一个文件中读取，将流中的整个内容转换回 Python 对象。为了确保对象已正确转换，我们在它们两个上调用`greet()`方法。结果是以下内容：

```py
$ python pickler.py
Hi, I am Obi-Wan Kenobi and my ID is 123
Hi, I am Anakin Skywalker and my ID is 456 
```

`pickle`模块还允许你通过`dumps()`和`loads()`函数（注意两个名称末尾的`s`）将数据转换为（和从）字节对象。在日常应用中，`pickle`通常在我们需要持久化不应与其他应用程序交换的 Python 数据时使用。几年前我们遇到的一个例子是一个`flask`插件的会话管理器，它在将会话对象存储到 Redis 数据库之前将其序列化。然而，在实践中，你不太可能经常需要处理这个库。

另一个可能使用得更少的工具，但在资源不足时证明是有用的，是`shelve`。

## 使用 shelve 保存数据

“书架”是一个持久的类似字典的对象。它的美妙之处在于，你可以将任何可以`pickle`的对象保存到书架中，因此你不会像使用数据库那样受到限制。尽管有趣且有用，但在实际应用中`shelve`模块的使用相当罕见。为了完整性，让我们快速看看它的工作示例：

```py
# persistence/shelf.py
import shelve
class Person:
    def __init__(self, name, id):
        self.name = name
        self.id = id
with shelve.open("shelf1.shelve") as db:
    db["obi1"] = Person("Obi-Wan", 123)
    db["ani"] = Person("Anakin", 456)
    db["a_list"] = [2, 3, 5]
    db["delete_me"] = "we will have to delete this one..."
    print(
        list(db.keys())
    )  # ['ani', 'delete_me', 'a_list', 'obi1']
    del db["delete_me"]  # gone!
    print(list(db.keys()))  # ['ani', 'a_list', 'obi1']
    print("delete_me" in db)  # False
    print("ani" in db)  # True
    a_list = db["a_list"]
    a_list.append(7)
    db["a_list"] = a_list
    print(db["a_list"])  # [2, 3, 5, 7] 
```

除了相关的连接和模板代码之外，这个例子类似于字典练习。我们创建一个`Person`类，然后在上下文管理器中打开一个`shelve`文件。正如你所看到的，我们使用字典语法存储了四个对象：两个`Person`实例、一个列表和一个字符串。如果我们打印键，我们会得到一个包含我们使用的四个键的列表。在打印之后，我们立即从书架中删除（恰如其名）的`delete_me`键/值对。再次打印键时，我们可以看到删除已成功。然后我们测试几个键的成员资格，最后，我们将数字`7`追加到`a_list`。注意，我们必须从书架中提取列表，修改它，然后再保存。

另一种打开书架的方法可以稍微加快这个过程：

```py
# persistence/shelf.py
with shelve.open("shelf2.shelve", writeback=True) as db:
    db["a_list"] = [11, 13, 17]
    db["a_list"].append(19)  # in-place append!
    print(db["a_list"])  # [11, 13, 17, 19] 
```

通过以`writeback=True`打开书架，我们启用了`writeback`功能，这使得我们可以像在常规字典中添加值一样简单地追加到`a_list`。这个功能默认不激活的原因是，它伴随着你在内存消耗和关闭书架时需要付出的代价。

既然我们已经向与数据持久性相关的标准库模块致敬，让我们来看看 Python 生态系统中最广泛采用的 ORM 之一：SQLAlchemy。

## 将数据保存到数据库中

对于这个例子，我们将使用一个内存数据库，这将使事情对我们来说更简单。在本书的源代码中，我们留下了一些注释来展示如何生成 SQLite 文件，所以我们希望您也能探索这个选项。

您可以在[`dbeaver.io/`](https://dbeaver.io/)找到免费的 SQLite 数据库浏览器。DBeaver 是一款免费的跨平台数据库工具，适用于开发者、数据库管理员、分析师以及所有需要与数据库打交道的人。它支持所有流行的数据库：MySQL、PostgreSQL、SQLite、Oracle、DB2、SQL Server、Sybase、MS Access、Teradata、Firebird、Apache Hive、Phoenix、Presto 等。

在我们深入代码之前，让我们简要介绍一下关系数据库的概念。

关系数据库是一种允许您按照 1969 年由爱德华·F·科德发明的**关系模型**保存数据的数据库。在这个模型中，数据存储在一个或多个表中。每个表都有行（也称为**记录**或**元组**），每行代表表中的一个条目。表也有列（也称为**属性**），每列代表记录的一个属性。每个记录通过一个唯一键来识别，更常见的是**主键**，它由表中的一列或多列组成。为了给您一个例子：想象一个名为`Users`的表，有`id`、`username`、`password`、`name`和`surname`列。

这样的表非常适合包含我们系统的用户；每一行代表一个不同的用户。例如，具有值`3`、`fab`、`my_wonderful_pwd`、`Fabrizio`和`Romano`的行将代表系统中的 Fabrizio 用户。

该模型被称为*关系*，因为您可以在表之间建立关系。例如，如果您向这个数据库添加一个名为`PhoneNumbers`的表，您可以将电话号码插入其中，然后通过关系确定哪个电话号码属于哪个用户。

要查询关系型数据库，我们需要一种特殊的语言。主要的标准化语言被称为**SQL**，即**结构化查询语言**。它起源于**关系代数**，这是一种用于操作和查询存储在关系型数据库中的数据的正式系统和理论框架。你可以执行的最常见操作通常涉及对行或列进行过滤、连接表、根据某些标准聚合结果等。以英语为例，对我们假设的数据库进行查询可能是：*检索所有用户名以“m”开头且最多有一个电话号码的用户（username, name, surname）*。在这个例子中，我们正在查询数据库中行的一个子集，并且只对`User`表中的三个列感兴趣的结果。我们通过只选择以字母*m*开头的用户进行过滤，更进一步，只选择最多有一个电话号码的用户。

每个数据库都附带其自己的**风味**SQL。它们都在某种程度上尊重标准，但没有一个完全遵守，它们在某种程度上都各不相同。这在现代软件开发中引发了一个问题。如果我们的应用程序包含原始 SQL 代码，那么如果我们决定使用不同的数据库引擎，或者可能是同一引擎的不同版本，我们可能需要修改应用程序中的 SQL 代码。

这可能会相当痛苦，尤其是由于 SQL 查询可能非常复杂。为了减轻这个问题，计算机科学家们创建了将编程语言的对象映射到关系型数据库表的代码。不出所料，这种工具的名称是**对象关系映射**（**ORM**）。

在现代应用程序开发中，人们通常会通过使用 ORM（对象关系映射）来开始与数据库交互。如果他们发现自己无法通过 ORM 执行某个查询，那么他们就会，而且只有在这种情况下，才会直接使用 SQL。这是在完全没有 SQL 和完全不使用 ORM 之间的一种良好折衷，这意味着专门化与数据库交互的代码，具有上述缺点。

在本节中，我们想展示一个利用 SQLAlchemy 的例子，它是最受欢迎的第三方 Python ORM 之一。您需要将此章节的虚拟环境中安装它。我们将定义两个模型（`Person`和`Email`），每个模型都映射到一个表，然后我们将填充数据库并在其上执行一些查询。

让我们从模型声明开始：

```py
# persistence/alchemy_models.py
from sqlalchemy import ForeignKey, String, Integer
from sqlalchemy.orm import (
    DeclarativeBase,
    mapped_column,
    relationship,
) 
```

在开始时，我们导入一些函数和类型。然后我们继续编写`Person`和`Email`类，以及它们必需的基类。让我们看看这些定义：

```py
# persistence/alchemy_models.py
class Base(DeclarativeBase):
    pass
class Person(Base):
    __tablename__ = "person"
    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String)
    age = mapped_column(Integer)
    emails = relationship(
        "Email",
        back_populates="person",
        order_by="Email.email",
        cascade="all, delete-orphan",
    )
    def __repr__(self):
        return f"{self.name}(id={self.id})"
class Email(Base):
    __tablename__ = "email"
    id = mapped_column(Integer, primary_key=True)
    email = mapped_column(String)
    person_id = mapped_column(ForeignKey("person.id"))
    person = relationship("Person", back_populates="emails")
    def __str__(self):
        return self.email
    __repr__ = __str__ 
```

每个模型都继承自 `Base` 类，在这个例子中，它是一个简单地继承自 SQLAlchemy 的 `DeclarativeBase` 的类。我们定义了 `Person`，它映射到名为 `"person"` 的表，并公开了 `id`、`name` 和 `age` 属性。我们还通过声明访问 `emails` 属性将检索与特定 `Person` 实例相关的 `"Email"` 表中的所有条目来声明与 `"Email"` 模型的关系。`cascade` 选项影响创建和删除的工作方式，但它是一个更高级的概念，所以我们建议你现在忽略它，也许以后再深入研究。

我们最后声明的是 `__repr__()` 方法，它为我们提供了对象的官方字符串表示形式。这个表示形式应该能够用来完全重建对象，但在这个例子中，我们只是简单地用它来提供输出。Python 将 `repr(obj)` 重定向到对 `obj.__repr__()` 的调用。

我们还声明了 `"Email"` 模型，它映射到名为 `"email"` 的表，并将包含电子邮件地址以及属于这些电子邮件地址的人的引用。你可以看到 `person_id` 和 `person` 属性都是关于在 `"Email"` 和 `"Person"` 类之间设置关系。注意我们还如何在 `"Email"` 上声明 `__str__()` 方法，然后将其分配给一个名为 `__repr__()` 的别名。这意味着在 `"Email"` 对象上调用 `repr()` 或 `str()` 最终都会调用 `__str__()` 方法。这在 Python 中是一种相当常见的技巧，用于避免重复相同的代码，所以我们有机会在这里向你展示它。

对这段代码的深入理解需要比我们所能提供的空间更多，所以我们鼓励你阅读有关 **数据库管理系统**（**DBMS**）、SQL、关系代数和 SQLAlchemy 的资料。

现在我们有了我们的模型，让我们使用它们来持久化一些数据。

看看下面的例子（这里展示的所有片段，除非另有说明，都属于 `persistence` 文件夹中的 `alchemy.py` 文件）：

```py
# persistence/alchemy.py
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from alchemy_models import Person, Email, Base
# swap these lines to work with an actual DB file
# engine = create_engine('sqlite:///example.db')
engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine) 
```

首先，我们导入我们需要的函数和类。然后我们继续为应用程序创建一个引擎，最后我们指示 SQLAlchemy 通过给定的引擎创建所有表。

`create_engine()` 函数支持一个名为 `echo` 的参数，可以设置为 `True`、`False` 或字符串 `"debug"`，以启用不同级别的所有语句和它们参数的 `repr()` 的日志记录。请参阅 SQLAlchemy 的官方文档以获取更多信息。

在 SQLAlchemy 中，引擎是一个核心组件，它作为 Python 应用程序和数据库之间的主要接口。它管理数据库交互的两个关键方面：连接和 SQL 语句执行。

在导入和创建引擎以及表之后，我们通过使用我们刚刚创建的引擎设置了一个会话，使用上下文管理器。我们首先创建两个 `Person` 对象：

```py
with Session(engine) as session:
    anakin = Person(name="Anakin Skywalker", age=32)
    obione = Person(name="Obi-Wan Kenobi", age=40) 
```

我们随后使用两种不同的技术给这两个对象添加电子邮件地址。一种是将它们分配给一个列表，另一种则是简单地追加：

```py
 obione.emails = [
        Email(email="obi1@example.com"),
        Email(email="wanwan@example.com"),
    ]
    anakin.emails.append(Email(email="ani@example.com"))
    anakin.emails.append(Email(email="evil.dart@example.com"))
    anakin.emails.append(Email(email="vader@example.com")) 
```

我们还没有接触数据库。只有当我们使用`session`对象时，其中才会发生实际的操作：

```py
 session.add(anakin)
    session.add(obione)
    session.commit() 
```

添加两个`Person`实例也足以添加它们的电子邮件地址（这要归功于级联效应）。调用`commit()`会导致 SQLAlchemy 提交事务并将数据保存到数据库中。

**事务**是一个提供类似沙盒的操作，但在数据库上下文中。只要事务没有被提交，我们就可以回滚对数据库所做的任何修改，并通过这样做，回到开始事务之前的状态。SQLAlchemy 提供了更复杂和更细粒度的处理事务的方法，您可以在其官方文档中学习，这是一个相当高级的话题。

我们现在使用`like()`查询所有名字以`Obi`开头的所有人，这会连接到 SQL 中的`LIKE`运算符：

```py
 obione = session.scalar(
        select(Person).where(Person.name.like("Obi%"))
    )
    print(obione, obione.emails) 
```

我们查询该查询的第一个结果（我们知道我们只有欧比旺）并打印它。然后我们通过使用对名字的精确匹配来获取`anakin`，只是为了展示另一种过滤方式：

```py
 anakin = session.scalar(
        select(Person).where(Person.name == "Anakin Skywalker")
    )
    print(anakin, anakin.emails) 
```

我们随后捕获安纳金的 ID，并从全局框架中删除`anakin`对象（这并不会从数据库中删除条目）：

```py
 anakin_id = anakin.id
    del anakin 
```

我们这样做的原因是我们想向您展示如何通过 ID 获取对象。为了显示数据库的全部内容，我们编写了一个`display_info()`函数。它通过从`Email`的关系中获取电子邮件地址和人员对象来工作，同时也提供了每个模型的所有对象的计数。在这个模块中，这个函数在进入提供会话的上下文管理器之前定义：

```py
def display_info(session):
    # get all emails first
    emails = select(Email)
    # display results
    print("All emails:")
    for email in session.scalars(emails):
        print(f" - {email.person.name} <{email.email}>")
    # display how many objects we have in total
    people = session.scalar(
        select(func.count()).select_from(Person)
    )
    emails = session.scalar(
        select(func.count()).select_from(Email)
    )
    print("Summary:")
    print(f" {people=}, {emails=}") 
```

我们调用这个函数，然后获取并删除`anakin`。最后，我们再次显示信息以验证他确实已经从数据库中消失：

```py
 display_info(session)
    anakin = session.get(Person, anakin_id)
    session.delete(anakin)
    session.commit()
    display_info(session) 
```

所有这些片段的输出合并在一起如下（为了您的方便，我们已经将输出分为四个部分，以反映产生该输出的四个代码块）：

```py
$ python alchemy.py
Obi-Wan Kenobi(id=2) [obi1@example.com, wanwan@example.com]
Anakin Skywalker(id=1) [
    ani@example.com, evil.dart@example.com, vader@example.com
]
All emails:
 - Anakin Skywalker <ani@example.com>
 - Anakin Skywalker <evil.dart@example.com>
 - Anakin Skywalker <vader@example.com>
 - Obi-Wan Kenobi <obi1@example.com>
 - Obi-Wan Kenobi <wanwan@example.com>
Summary:
 people=2, emails=5
All emails:
 - Obi-Wan Kenobi <obi1@example.com>
 - Obi-Wan Kenobi <wanwan@example.com>
Summary:
 people=1, emails=2 
```

如您从最后两个块中看到的，删除`anakin`已经删除了一个`Person`对象及其关联的三个电子邮件地址。再次强调，这是因为当我们删除`anakin`时发生了级联。

这就结束了我们对数据持久性的简要介绍。这是一个庞大且有时复杂的领域，我们鼓励您去探索，尽可能多地学习理论。在数据库系统方面，知识的缺乏或理解不当可能会影响系统中的错误数量以及其性能。

# 配置文件

配置文件是许多 Python 应用程序的关键组成部分。它们允许开发者将主应用程序代码与设置和参数分离。这种分离对于维护、管理和分发软件非常有帮助，尤其是在应用程序需要在不同的环境中运行时——例如开发、生产和测试——并且具有不同的配置。

配置文件允许：

+   **灵活性**：用户可以在不修改应用程序代码的情况下更改应用程序的行为。这对于在不同环境中部署的应用程序或需要数据库、API 密钥等凭证的应用程序特别有用。

+   **安全性**：敏感信息，如认证凭证、API 密钥或秘密令牌，应从源代码中移除，并独立于代码库进行管理。

## 常见格式

配置文件可以写成几种格式，每种格式都有自己的语法和功能。一些流行的格式是 `INI`、`JSON`、`YAML`、`TOML` 和 `.env`。

在本节中，我们将简要探讨 `INI` 和 `TOML` 格式。在 *第十四章*，*API 开发简介* 中，我们还将使用 `.env` 文件。

### INI 配置格式

`INI` 格式是一个简单的文本文件，分为几个部分。每个部分包含以键/值对形式表示的属性。

要了解更多关于此格式的信息，请访问 [`en.wikipedia.org/wiki/INI_file`](https://en.wikipedia.org/wiki/INI_file)。

让我们看看一个示例 INI 配置文件：

```py
# config_files/config.ini
[owner]
name = Fabrizio Romano
dob = 1975-12-29T11:50:00Z
[DEFAULT]
title = Config INI example
host = 192.168.1.1
[database]
host = 192.168.1.255
user = redis
password = redis-password
db_range = [0, 32]
[database.primary]
port = 6379
connection_max = 5000
[database.secondary]
port = 6380
connection_max = 4000 
```

在前面的文本中，有一些部分专门用于数据库连接。常见属性可以在 `database` 部分找到，而特定属性则放在 `.primary` 或 `.secondary` 部分中，分别代表连接到 *主* 和 *次* 数据库的配置。还有一个 `owner` 部分和一个 `DEFAULT` 部分。

要在应用程序中读取此配置，我们可以使用标准库中的 `configparser` 模块（[`docs.python.org/3/library/configparser.html`](https://docs.python.org/3/library/configparser.html)）。它非常直观，因为它将生成一个类似于字典的对象，并且额外的好处是 `DEFAULT` 部分会自动为所有其他部分提供值。

让我们看看一个来自 Python 脚本的一个示例会话：

```py
# config_files/config-ini.txt
>>> import configparser
>>> config = configparser.ConfigParser()
>>> config.read("config.ini")
['config.ini']
>>> config.sections()
['owner', 'database', 'database.primary', 'database.secondary']
>>> config.items("database")
[
    ('title', 'Config INI example'), ('host', '192.168.1.255'),
    ('user', 'redis'), ('password', 'redis-password'),
    ('db_range', '[0, 32]')
]
>>> config["database"]
<Section: database>
>>> dict(config["database"])
{
    'host': '192.168.1.255', 'user': 'redis',
    'password': 'redis-password', 'db_range': '[0, 32]',
    'title': 'Config INI example'
}
>>> config["DEFAULT"]["host"]
'192.168.1.1'
>>> dict(config["database.secondary"])
{
    'port': '6380', 'connection_max': '4000',
    'title': 'Config INI example', 'host': '192.168.1.1'
}
>>> config.getint("database.primary", "port")
6379 
```

注意我们如何导入 `configparser` 并使用它来创建一个 `config` 对象。此对象公开了各种方法；您可以获取部分列表，以及检索其中的任何值。

在内部，`configparser` 将值存储为字符串，因此如果我们想将它们用作它们所代表的 Python 对象，我们需要适当地进行类型转换。`ConfigParser` 对象上有一些方法，例如 `getint()`、`getfloat()` 和 `getboolean()`，它们将检索一个值并将其转换为指定的类型，但如您所见，这个列表相当短。

注意，来自 `DEFAULT` 部分的属性被注入到所有其他部分中。此外，当一个部分定义了一个也存在于 `DEFAULT` 部分的键时，原始部分的值不会被 `DEFAULT` 部分的值覆盖。你可以在高亮显示的代码中看到一个示例，它显示 `title` 属性存在于 `database` 部分中，而 `host` 属性存在于两个部分中，它正确地保留了 `'192.168.1.255'` 的值。

### TOML 配置格式

`TOML` 格式在 Python 应用中相当流行，与 INI 格式相比，它具有更丰富的功能集。如果您想了解其语法，请参阅 [`toml.io/`](https://toml.io/) 。

这里，我们将看到一个快速示例，它遵循之前的示例。

```py
# config_file/config.toml
title = "Config Example"
[owner]
name = "Fabrizio Romano"
dob = 1975-12-29T11:50:00Z
[database]
host = "192.168.1.255"
user = "redis"
password = "redis-password"
db_range = [0, 32]
[database.primary]
port = 6379
connection_max = 5000
[database.secondary]
port = 6380
connection_max = 4000 
```

这次，我们没有 `DEFAULT` 部分，属性指定略有不同，即字符串被引号包围，而数字则不是。

我们将使用标准库中的 `tomllib` 模块（[`docs.python.org/3/library/tomllib.html`](https://docs.python.org/3/library/tomllib.html)）来读取此配置：

```py
# config_files/config-toml.txt
>>> import tomllib
>>> with open("config.toml", "rb") as f:
...     config = tomllib.load(f)
...
>>> config
{
    'title': 'Config Example',
    'owner': {
        'name': 'Fabrizio Romano',
        'dob': datetime.datetime(
            1975, 12, 29, 11, 50, tzinfo=datetime.timezone.utc
        )
    },
    'database': {
        'host': '192.168.1.255',
        'user': 'redis',
        'password': 'redis-password',
        'db_range': [0, 32],
        'primary': {'port': 6379, 'connection_max': 5000},
        'secondary': {'port': 6380, 'connection_max': 4000}
    }
}
>>> config["title"]
'Config Example'
>>> config["owner"]
{
    'name': 'Fabrizio Romano',
    'dob': datetime.datetime(
        1975, 12, 29, 11, 50, tzinfo=datetime.timezone.utc
    )
}
>>> config["database"]["primary"]
{'port': 6379, 'connection_max': 5000}
>>> config["database"]["db_range"]
[0, 32] 
```

注意这次，`config` 对象是一个字典。由于我们指定了 `database.primary` 和 `database.secondary` 部分，`tomllib` 创建了一个嵌套结构来表示它们。

使用 TOML，值会被正确地转换为 Python 对象。我们有字符串、数字、列表，甚至是从代表 Fabrizio 出生日期的 iso 格式化字符串创建的 `datetime` 对象。在 `tomllib` 文档页面上，你可以找到一个包含所有可能转换的表格。

# 摘要

在本章中，我们探讨了与文件和目录一起工作。我们学习了如何读取和写入文件，以及如何通过使用上下文管理器优雅地完成这些操作。我们还探讨了目录：如何递归和非递归地列出其内容。我们还了解了路径，它们是访问文件和目录的门户。

我们简要地看到了如何创建 ZIP 归档并提取其内容。本书的源代码还包含了一个使用不同压缩格式的示例：`tar.gz` 。

我们讨论了数据交换格式，并深入探讨了 JSON。我们编写了一些自定义编码器和解码器，用于特定的 Python 数据类型，并从中获得了乐趣。

然后，我们探讨了 I/O，包括内存流和 HTTP 请求。

我们看到了如何使用 `pickle`、`shelve` 和 SQLAlchemy ORM 库持久化数据。

最后，我们探索了两种配置文件示例，使用了 INI 和 TOML 格式。

现在，你应该已经很好地理解了如何处理文件和数据持久化，我们希望你能花时间自己深入探索这些主题。 

下一章将探讨密码学和令牌。

# 加入我们的 Discord 社区

加入我们的 Discord 社区空间，与作者和其他读者进行讨论：

`discord.com/invite/uaKmaz7FEC`

![img](img/QR_Code119001106417026468.png)
