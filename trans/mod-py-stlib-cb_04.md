# 第四章：文件系统和目录

在本章中，我们将涵盖以下食谱：

+   遍历文件夹-递归遍历文件系统中的路径并检查其内容

+   处理路径-以系统独立的方式构建路径

+   扩展文件名-查找与特定模式匹配的所有文件

+   获取文件信息-检测文件或目录的属性

+   命名临时文件-使用需要从其他进程访问的临时文件

+   内存和磁盘缓冲区-如果临时缓冲区大于阈值，则将其暂存到磁盘上

+   管理文件名编码-处理文件名的编码

+   复制目录-复制整个目录的内容

+   安全地替换文件内容-在发生故障时如何安全地替换文件的内容

# 介绍

使用文件和目录是大多数软件自然而然的，也是我们作为用户每天都在做的事情，但作为开发人员，您很快会发现它可能比预期的要复杂得多，特别是当需要支持多个平台或涉及编码时。

Python 标准库有许多强大的工具可用于处理文件和目录。起初，可能很难在`os`、`shutil`、`stat`和`glob`函数中找到这些工具，但一旦您了解了所有这些工具，就会清楚地知道标准库提供了一套很好的工具来处理文件和目录。

# 遍历文件夹

在文件系统中使用路径时，通常需要查找直接或子文件夹中包含的所有文件。想想复制一个目录或计算其大小；在这两种情况下，您都需要获取要复制的目录中包含的所有文件的完整列表，或者要计算大小的目录中包含的所有文件的完整列表。

# 如何做...

这个食谱的步骤如下：

1.  `os`模块中的`os.walk`函数用于递归遍历目录，其使用方法并不直接，但稍加努力，我们可以将其包装成一个方便的生成器，列出所有包含的文件：

```py
import os

def traverse(path):
    for basepath, directories, files in os.walk(path):
        for f in files:
            yield os.path.join(basepath, f)
```

1.  然后，我们可以遍历`traverse`并对其进行任何操作：

```py
for f in traverse('.'):
    print(f)
```

# 它是如何工作的...

`os.walk`函数遍历目录及其所有子文件夹。对于它找到的每个目录，它返回三个值：目录本身、它包含的子目录和它包含的文件。然后，它将进入所提供的目录的子目录，并为子目录返回相同的三个值。

这意味着在我们的食谱中，`basepath`始终是正在检查的当前目录，`directories`是其子目录，`files`是它包含的文件。

通过迭代当前目录中包含的文件列表，并将它们的名称与目录路径本身连接起来，我们可以获取目录中包含的所有文件的路径。由于`os.walk`将进入所有子目录，因此我们将能够返回直接或间接位于所需路径内的所有文件。

# 处理路径

Python 最初是作为系统管理语言创建的。最初是为 Unix 系统编写脚本，因此在语言的核心部分之一始终是浏览磁盘，但在 Python 的最新版本中，这进一步扩展到了`pathlib`模块，它使得非常方便和容易地构建引用文件或目录的路径，而无需关心我们正在运行的系统。

由于编写多平台软件可能很麻烦，因此非常重要的是有中间层来抽象底层系统的约定，并允许我们编写可以在任何地方运行的代码。

特别是在处理路径时，Unix 和 Windows 系统处理路径的方式之间的差异可能会有问题。一个系统使用`/`，另一个使用`\`来分隔路径的部分本身就很麻烦，但 Windows 还有驱动器的概念，而 Unix 系统没有，因此我们需要一些东西来抽象这些差异并轻松管理路径。

# 如何做...

执行此食谱的以下步骤：

1.  `pathlib`库允许我们根据构成它的部分构建路径，根据您所在的系统正确地执行正确的操作：

```py
>>> import pathlib
>>> 
>>> path = pathlib.Path('somefile.txt')
>>> path.write_text('Hello World')  # Write some text into file.
11
>>> print(path.resolve())  # Print absolute path
/Users/amol/wrk/pythonstlcookbook/somefile.txt
>>> path.read_text()  # Check the file content
'Hello World'
>>> path.unlink()  # Destroy the file
```

1.  有趣的是，即使在 Windows 上进行相同的操作，也会得到完全相同的结果，即使`path.resolve()`会打印出稍微不同的结果：

```py
>>> print(path.resolve())  # Print absolute path
C:\\wrk\\pythonstlcookbook\\somefile.txt
```

1.  一旦我们有了`pathlib.Path`实例，甚至可以使用`/`运算符在文件系统中移动：

```py
>>> path = pathlib.Path('.')
>>> path = path.resolve()
>>> path
PosixPath('/Users/amol/wrk/pythonstlcookbook')
>>> path = path / '..'
>>> path.resolve()
PosixPath('/Users/amol/wrk')
```

即使我是在类 Unix 系统上编写的，上述代码在 Windows 和 Linux/macOS 上都能正常工作并产生预期的结果。

# 还有更多...

`pathlib.Path`实际上会根据我们所在的系统构建不同的对象。在 POSIX 系统上，它将导致一个`pathlib.PosixPath`对象，而在 Windows 系统上，它将导致一个`pathlib.WindowsPath`对象。

在 POSIX 系统上无法构建`pathlib.WindowsPath`，因为它是基于 Windows 系统调用实现的，而这些调用在 Unix 系统上不可用。如果您需要在 POSIX 系统上使用 Windows 路径（或在 Windows 系统上使用 POSIX 路径），可以依赖于`pathlib.PureWindowsPath`和`pathlib.PurePosixPath`。

这两个对象不会实现实际访问文件的功能（读取、写入、链接、解析绝对路径等），但它们将允许您执行与操作路径本身相关的简单操作。

# 扩展文件名

在我们系统的日常使用中，我们习惯于提供路径，例如`*.py`，以识别所有的 Python 文件，因此当我们的用户提供一个或多个文件给我们的软件时，他们能够做同样的事情并不奇怪。

通常，通配符是由 shell 本身扩展的，但假设您从配置文件中读取它们，或者您想编写一个工具来清除当前项目中的`.pyc`文件（编译的 Python 字节码缓存），那么 Python 标准库中有您需要的内容。

# 如何做...

此食谱的步骤是：

1.  `pathlib`能够对您提供的路径执行许多操作。其中之一是解析通配符：

```py
>>> list(pathlib.Path('.').glob('*.py'))
[PosixPath('conf.py')]
```

1.  它还支持递归解析通配符：

```py
>>> list(pathlib.Path('.').glob('**/*.py'))
[PosixPath('conf.py'), PosixPath('venv/bin/cmark.py'), 
 PosixPath('venv/bin/rst2html.py'), ...]
```

# 获取文件信息

当用户提供路径时，您真的不知道路径指的是什么。它是一个文件吗？是一个目录吗？它甚至存在吗？

检索文件信息允许我们获取有关提供的路径的详细信息，例如它是否指向文件以及该文件的大小。

# 如何做...

执行此食谱的以下步骤：

1.  对任何`pathlib.Path`使用`.stat()`将提供有关路径的大部分详细信息：

```py
>>> pathlib.Path('conf.py').stat()
os.stat_result(st_mode=33188, 
               st_ino=116956459, 
               st_dev=16777220, 
               st_nlink=1, 
               st_uid=501, 
               st_gid=20, 
               st_size=9306, 
               st_atime=1519162544, 
               st_mtime=1510786258, 
               st_ctime=1510786258)
```

返回的详细信息是指：

+   +   `st_mode`: 文件类型、标志和权限

+   `st_ino`: 存储文件的文件系统节点

+   `st_dev`: 存储文件的设备

+   `st_nlink`: 对此文件的引用（超链接）的数量

+   `st_uid`: 拥有文件的用户

+   `st_gid`: 拥有文件的组

+   `st_size`: 文件的大小（以字节为单位）

+   `st_atime`: 文件上次访问的时间

+   `st_mtime`: 文件上次修改的时间

+   `st_ctime`: 文件在 Windows 上创建的时间，Unix 上修改元数据的时间

1.  如果我们想要查看其他详细信息，例如路径是否存在或者它是否是一个目录，我们可以依赖于这些特定的方法：

```py
>>> pathlib.Path('conf.py').exists()
True
>>> pathlib.Path('conf.py').is_dir()
False
>>> pathlib.Path('_build').is_dir()
True
```

# 命名临时文件

通常在处理临时文件时，我们不关心它们存储在哪里。我们需要创建它们，在那里存储一些内容，并在完成后摆脱它们。大多数情况下，我们在想要存储一些太大而无法放入内存的东西时使用临时文件，但有时你需要能够提供一个文件给另一个工具或软件，临时文件是避免需要知道在哪里存储这样的文件的好方法。

在这种情况下，我们需要知道通往临时文件的路径，以便我们可以将其提供给其他工具。

这就是`tempfile.NamedTemporaryFile`可以帮助的地方。与所有其他`tempfile`形式的临时文件一样，它将为我们创建，并且在我们完成工作后会自动删除，但与其他类型的临时文件不同，它将有一个已知的路径，我们可以提供给其他程序，这些程序将能够从该文件中读取和写入。

# 如何做...

`tempfile.NamedTemporaryFile`将创建临时文件：

```py
>>> from tempfile import NamedTemporaryFile
>>>
>>> with tempfile.NamedTemporaryFile() as f:
...   print(f.name)
... 
/var/folders/js/ykgc_8hj10n1fmh3pzdkw2w40000gn/T/tmponbsaf34
```

`.name`属性导致完整的文件路径在磁盘上，这使我们能够将其提供给其他外部程序：

```py
>>> with tempfile.NamedTemporaryFile() as f:
...   os.system('echo "Hello World" > %s' % f.name)
...   f.seek(0)
...   print(f.read())
... 
0
0
b'Hello World\n'
```

# 内存和磁盘缓冲

有时，我们需要将某些数据保留在缓冲区中，比如我们从互联网上下载的文件，或者我们动态生成的一些数据。

由于这种数据的大小通常是不可预测的，通常不明智将其全部保存在内存中。

如果你从互联网上下载一个 32GB 的大文件，需要处理它（如解压缩或解析），如果你在处理之前尝试将其存储到字符串中，它可能会耗尽你所有的内存。

这就是为什么通常依赖`tempfile.SpooledTemporaryFile`通常是一个好主意，它将保留内容在内存中，直到达到最大大小，然后如果它比允许的最大大小更大，就将其移动到临时文件中。

这样，我们可以享受保留数据的内存缓冲区的好处，而不会因为内容太大而耗尽所有内存，因为一旦内容太大，它将被移动到磁盘上。

# 如何做...

像其他`tempfile`对象一样，创建`SpooledTemporaryFile`就足以使临时文件可用。唯一的额外部分是提供允许的最大大小，`max_size=`，在此之后内容将被移动到磁盘上：

```py
>>> with tempfile.SpooledTemporaryFile(max_size=30) as temp:
...     for i in range(3):
...         temp.write(b'Line of text\n')
...     
...     temp.seek(0)
...     print(temp.read())
... 
b'Line of text\nLine of text\nLine of text\n'
```

# 它是如何工作的...

`tempfile.SpooledTemporaryFile`有一个`内部 _file`属性，它将真实数据存储在`BytesIO`存储中，直到它可以适应内存，然后一旦它比`max_size`更大，就将其移动到真实文件中。

在写入数据时，你可以通过打印`_file`的值来轻松看到这种行为：

```py
>>> with tempfile.SpooledTemporaryFile(max_size=30) as temp:
...     for i in range(3):
...         temp.write(b'Line of text\n')
...         print(temp._file)
... 
<_io.BytesIO object at 0x10d539ca8>
<_io.BytesIO object at 0x10d539ca8>
<_io.BufferedRandom name=4>
```

# 管理文件名编码

以可靠的方式使用文件系统并不像看起来那么容易。我们的系统必须有特定的编码来表示文本，通常这意味着我们创建的所有内容都是以该编码处理的，包括文件名。

问题在于文件名的编码没有强有力的保证。假设你连接了一个外部硬盘，那个硬盘上的文件名的编码是什么？嗯，这将取决于文件创建时系统的编码。

通常，为了解决这个问题，软件会尝试系统编码，如果失败，它会打印一些占位符（你是否曾经看到过一个充满`?`的文件名，只是因为你的系统无法理解文件的名称？），这通常允许我们看到有一个文件，并且在许多情况下甚至打开它，尽管我们可能不知道它的实际名称。

为了使一切更加复杂，Windows 和 Unix 系统在处理文件名时存在很大的差异。在 Unix 系统上，路径基本上只是字节；你不需要真正关心它们的编码，因为你只是读取和写入一堆字节。而在 Windows 上，文件名实际上是文本。

在 Python 中，文件名通常存储为`str`。它们是需要以某种方式进行编码/解码的文本。

# 如何做...

每当我们处理文件名时，我们应该根据预期的文件系统编码对其进行解码。如果失败（因为它不是以预期的编码存储的），我们仍然必须能够将其放入`str`而不使其损坏，以便我们可以打开该文件，即使我们无法读取其名称：

```py
def decode_filename(fname):
    fse = sys.getfilesystemencoding()
    return fname.decode(fse, "surrogateescape")
```

# 它是如何工作的...

`decode_filename`试图做两件事：首先，它询问 Python 根据操作系统预期的文件系统编码是什么。一旦知道了这一点，它就会尝试使用该编码解码提供的文件名。如果失败，它将使用`surrogateescape`进行解码。

这实际上意味着*如果你无法解码它，就将其解码为假字符，我们将使用它来表示文本*。

这真的很方便，因为这样我们能够将文件名作为文本进行管理，即使我们不知道它的编码，当它使用`surrogateescape`编码回字节时，它将导致回到其原始字节序列。

当文件名以与我们的系统相同的编码进行编码时，很容易看出我们如何能够将其解码为`str`并打印它以读取其内容：

```py
>>> utf8_filename_bytes = 'ùtf8.txt'.encode('utf8')
>>> utf8_filename = decode_filename(utf8_filename_bytes)
>>> type(utf8_filename)
<class 'str'>
>>> print(utf8_filename)
ùtf8.txt
```

如果编码实际上不是我们的系统编码（也就是说，文件来自一个非常古老的外部驱动器），我们实际上无法读取里面写的内容，但我们仍然能够将其解码为字符串，以便我们可以将其保存在一个变量中，并将其提供给任何可能需要处理该文件的函数：

```py
>>> latin1_filename_bytes = 'làtìn1.txt'.encode('latin1')
>>> latin1_filename = decode_filename(latin1_filename_bytes)
>>> type(latin1_filename)
<class 'str'>
>>> latin1_filename
'l\udce0t\udcecn1.txt'
```

`surrogateescape`意味着能够告诉 Python*我不在乎数据是否是垃圾，只需原样传递未知的字节*。

# 复制目录

复制目录的内容是我们可以轻松做到的事情，但是如果我告诉你，像`cp`（在 GNU 系统上复制文件的命令）这样的工具大约有 1200 行代码呢？

显然，`cp`的实现不是基于 Python 的，它已经发展了几十年，它照顾的远远超出了你可能需要的，但是自己编写递归复制目录的代码所需的工作远远超出你的预期。

幸运的是，Python 标准库提供了实用程序，可以直接执行最常见的操作之一。

# 如何做...

此处的步骤如下：

1.  `copydir`函数可以依赖于`shutil.copytree`来完成大部分工作：

```py
import shutil

def copydir(source, dest, ignore=None):
    """Copy source to dest and ignore any file matching ignore 
       pattern."""
    shutil.copytree(source, dest, ignore_dangling_symlinks=True,
                    ignore=shutil.ignore_patterns(*ignore) if 
                    ignore else None)
```

1.  然后，我们可以轻松地使用它来复制任何目录的内容，甚至将其限制为只复制相关部分。我们将复制一个包含三个文件的目录，其中我们实际上只想复制`.pdf`文件：

```py
>>> import glob
>>> print(glob.glob('_build/pdf/*'))
['_build/pdf/PySTLCookbook.pdf', '_build/pdf/PySTLCookbook.rtc', '_build/pdf/PySTLCookbook.stylelog']
```

1.  我们的目标目录目前不存在，因此它不包含任何内容：

```py
>>> print(glob.glob('/tmp/buildcopy/*'))
[]
```

1.  一旦我们执行`copydir`，它将被创建并包含我们期望的内容：

```py
>>> copydir('_build/pdf', '/tmp/buildcopy', ignore=('*.rtc', '*.stylelog'))
```

1.  现在，目标目录存在并包含我们期望的内容：

```py
>>> print(glob.glob('/tmp/buildcopy/*'))
['/tmp/buildcopy/PySTLCookbook.pdf']
```

# 它是如何工作的...

`shutil.copytree`将通过`os.listdir`检索提供的目录的内容。对于`listdir`返回的每个条目，它将检查它是文件还是目录。

如果是文件，它将通过`shutil.copy2`函数进行复制（实际上可以通过提供`copy_function`参数来替换使用的函数），如果是目录，`copytree`本身将被递归调用。

然后使用`ignore`参数构建一个函数，一旦调用，将返回所有需要忽略的文件，给定一个提供的模式：

```py
>>> f = shutil.ignore_patterns('*.rtc', '*.stylelog')
>>> f('_build', ['_build/pdf/PySTLCookbook.pdf', 
                 '_build/pdf/PySTLCookbook.rtc', 
                 '_build/pdf/PySTLCookbook.stylelog'])
{'_build/pdf/PySTLCookbook.stylelog', '_build/pdf/PySTLCookbook.rtc'}
```

因此，`shutil.copytree`将复制除`ignore_patterns`之外的所有文件，这将使其跳过。

最后的`ignore_dangling_symlinks=True`参数确保在`symlinks`损坏的情况下，我们只是跳过文件而不是崩溃。

# 安全地替换文件内容

替换文件的内容是一个非常缓慢的操作。与替换变量的内容相比，通常慢几倍；当我们将某些东西写入磁盘时，需要一些时间才能真正刷新，以及在内容实际写入磁盘之前需要一些时间。这不是一个原子操作，因此如果我们的软件在保存文件时遇到任何问题，文件可能会被写入一半，我们的用户无法恢复其数据的一致状态。

通常有一种常用模式来解决这种问题，该模式基于写入文件是一个缓慢、昂贵、易出错的操作，但重命名文件是一个原子、快速、廉价的操作。

# 如何做...

您需要执行以下操作：

1.  就像`open`可以用作上下文管理器一样，我们可以轻松地推出一个`safe_open`函数，以安全的方式打开文件进行写入：

```py
import tempfile, os

class safe_open:
    def __init__(self, path, mode='w+b'):
        self._target = path
        self._mode = mode

    def __enter__(self):
        self._file = tempfile.NamedTemporaryFile(self._mode, delete=False)
        return self._file

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
        if exc_type is None:
            os.rename(self._file.name, self._target)
        else:
            os.unlink(self._file.name)
```

1.  使用`safe_open`作为上下文管理器允许我们写入文件，就像我们通常会做的那样。

```py
with safe_open('/tmp/myfile') as f:
    f.write(b'Hello World')
```

1.  内容将在退出上下文时正确保存：

```py
>>> print(open('/tmp/myfile').read())
Hello World
```

1.  主要区别在于，如果我们的软件崩溃或在写入时发生系统故障，我们不会得到一个写入一半的文件，而是会保留文件的任何先前状态。在这个例子中，我们在尝试写入`替换 hello world，期望写入更多`时中途崩溃：

```py
with open('/tmp/myfile', 'wb+') as f:
    f.write(b'Replace the hello world, ')
    raise Exception('but crash meanwhile!')
    f.write(b'expect to write some more')
```

1.  使用普通的`open`，结果将只是`"替换 hello world，"`：

```py
>>> print(open('/tmp/myfile').read())
Replace the hello world,
```

1.  在使用`safe_open`时，只有在整个写入过程成功时，文件才会包含新数据：

```py
with safe_open('/tmp/myfile') as f:
    f.write(b'Replace the hello world, ')
    raise Exception('but crash meanwhile!')
    f.write(b'expect to write some more')
```

1.  在所有其他情况下，文件仍将保留其先前的状态：

```py
>>> print(open('/tmp/myfile').read())
Hello World
```

# 工作原理...

`safe_open`依赖于`tempfile`来创建一个新文件，其中实际发生写操作。每当我们在上下文中写入`f`时，实际上是在临时文件中写入。

然后，只有当上下文存在时（`safe_open.__exit__`中的`exc_type`为 none），我们才会使用`os.rename`将旧文件与我们刚刚写入的新文件进行交换。

如果一切如预期般运行，我们应该有新文件，并且所有内容都已更新。

如果任何步骤失败，我们只需向临时文件写入一些或没有数据，并通过`os.unlink`将其丢弃。

在这种情况下，我们以前的文件从未被触及，因此仍保留其先前的状态。
