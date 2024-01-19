# 读/写数据

在本章中，我们将涵盖以下配方：

+   读取和写入文本数据——从文件中读取任何编码的文本

+   读取文本行——逐行读取文本文件

+   读取和写入二进制数据——从文件中读取二进制结构化数据

+   压缩目录——读取和写入压缩的 ZIP 存档

+   Pickling 和 shelving——如何将 Python 对象保存在磁盘上

+   读取配置文件——如何读取`.ini`格式的配置文件

+   写入 XML/HTML 内容——生成 XML/HTML 内容

+   读取 XML/HTML 内容——从文件或字符串解析 XML/HTML 内容

+   读取和写入 CSV——读取和写入类似电子表格的 CSV 文件

+   读取和写入关系数据库——将数据加载到`SQLite`数据库中

# 介绍

您的软件的输入将来自各种来源：命令行选项，标准输入，网络，以及经常是文件。从输入中读取本身很少是处理外部数据源时的问题；一些输入可能需要更多的设置，有些更直接，但通常只是打开它然后从中读取。

问题出在我们读取的数据该如何处理。有成千上万种格式，每种格式都有其自己的复杂性，有些是基于文本的，有些是二进制的。在本章中，我们将设置处理您作为开发人员在生活中可能会遇到的最常见格式的配方。

# 读取和写入文本数据

当读取文本文件时，我们已经知道应该以文本模式打开它，这是 Python 的默认模式。在这种模式下，Python 将尝试根据`locale.getpreferredencoding`返回的作为我们系统首选编码的编码来解码文件的内容。

遗憾的是，任何类型的编码都是我们系统的首选编码与文件内容保存时使用的编码无关。因为它可能是别人写的文件，甚至是我们自己写的，编辑器可能以任何编码保存它。

因此，唯一的解决方案是指定应该用于解码文件的编码。

# 如何做...

Python 提供的`open`函数接受一个`encoding`参数，可以用于正确地编码/解码文件的内容：

```py
# Write a file with latin-1 encoding
with open('/tmp/somefile.txt', mode='w', encoding='latin-1') as f:
    f.write('This is some latin1 text: "è già ora"')

# Read back file with latin-1 encoding.
with open('/tmp/somefile.txt', encoding='latin-1') as f:
    txt = f.read()
    print(txt)
```

# 它是如何工作的...

一旦`encoding`选项传递给`open`，生成的文件对象将知道任何提供给`file.write`的字符串必须在将实际字节存储到文件之前编码为指定的编码。对于`file.read()`也是如此，它将从文件中获取字节，并在将它们返回给您之前使用指定的编码对其进行解码。

这允许您独立于系统声明的首选编码，读/写文件中的内容。

# 还有更多...

如果您想知道如何可能读取编码未知的文件，那么这是一个更加复杂的问题。

事实是，除非文件在头部提供一些指导，或者等效物，可以告诉您内容的编码类型，否则没有可靠的方法可以知道文件可能被编码的方式。

您可以尝试多种不同类型的编码，并检查哪种编码能够解码内容（不会抛出`UnicodeDecodeError`），但是一组字节解码为一种编码并不保证它解码为正确的结果。例如，编码为`utf-8`的`'ì'`字符在`latin-1`中完美解码，但结果完全不同：

```py
>>> 'ì'.encode('utf-8').decode('latin-1')
'Ã¬'
```

如果您真的想尝试猜测内容的类型编码，您可能想尝试一个库，比如`chardet`，它能够检测到大多数常见类型的编码。如果要解码的数据长度足够长且足够多样化，它通常会成功地检测到正确的编码。

# 读取文本行

在处理文本文件时，通常最容易的方法是按行处理；每行文本是一个单独的实体，我们可以通过`'\n'`或`'\r\n'`连接所有行，因此在列表中有文本文件的所有行将非常方便。

有一种非常方便的方法可以立即从文本文件中提取行，Python 可以立即使用。

# 如何做...

由于`file`对象本身是可迭代的，我们可以直接构建一个列表：

```py
with open('/var/log/install.log') as f:
    lines = list(f)
```

# 工作原理...

`open`充当上下文管理器，返回结果对象`file`。依赖上下文管理器非常方便，因为当我们完成文件操作时，我们需要关闭它，使用`open`作为上下文管理器将在我们退出`with`的主体时为我们关闭文件。

有趣的是`file`实际上是一个可迭代对象。当你迭代一个文件时，你会得到其中包含的行。因此，将`list`应用于它将构建所有行的列表，然后我们可以按照我们的意愿导航到结果列表。

# 字符

读取文本数据已经相当复杂，因为它需要解码文件的内容，但读取二进制数据可能会更加复杂，因为它需要解析字节及其内容以重建保存在文件中的原始数据。

在某些情况下，甚至可能需要处理字节顺序，因为当将数字保存到文本文件时，字节的写入顺序实际上取决于写入该文件的系统。

假设我们想要读取 TCP 头的开头，特定的源和目标端口、序列号和确认号，表示如下：

```py
0                   1                   2                   3
0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Source Port          |       Destination Port        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Sequence Number                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Acknowledgment Number                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

# 如何做...

此食谱的步骤如下：

1.  假设有一个包含 TCP 数据包转储的文件（在我的计算机上，我将其保存为`/tmp/packet.dump`），我们可以尝试将其读取为二进制数据并解析其内容。

Python 的`struct`模块是读取二进制结构化数据的完美工具，我们可以使用它来解析我们的 TCP 数据包，因为我们知道每个部分的大小：

```py
>>> import struct
>>> with open('/tmp/packet.dump', 'rb') as f:
...     data = struct.unpack_from('>HHLL', f.read())
>>> data
(50291, 80, 2778997212, 644363807)
```

作为 HTTP 连接，结果是我们所期望的：`源端口：50291，目标端口：80，序列号：2778997212`和`确认号：644363807`。

1.  可以使用`struct.pack`将二进制数据写回：

```py
>>> with open('/tmp/packet.dump', 'wb') as f:
...     data = struct.pack('>HHLL', 50291, 80, 2778997212, 644363807)
...     f.write(data)
>>> data
b'\xc4s\x00P\xa5\xa4!\xdc&h6\x1f'
```

# 工作原理...

首先，我们以*二进制模式*（`rb`参数）打开文件。这告诉 Python 避免尝试解码文件的内容，就像它是文本一样；内容以`bytes`对象的形式返回。

然后，我们使用`f.read()`读取的数据传递给`struct.unpack_from`，它能够解码二进制数据作为一组数字、字符串等。在我们的例子中，我们使用`>`指定我们正在读取的数据是大端排序的（就像所有与网络相关的数据一样），然后使用`HHLL`来说明我们要读取两个无符号 16 位数字和两个无符号 32 位数字（端口和序列/确认号）。

由于我们使用了`unpack_from`，在消耗了指定的四个数字后，任何其他剩余的数据都会被忽略。

写入二进制数据也是一样的。我们以二进制模式打开文件，通过`struct.pack`将四个数字打包成一个字节对象，并将它们写入文件。

# 还有更多...

`struct.pack`和`struct.unpack`函数支持许多选项和格式化程序，以定义应该写入/读取的数据以及应该如何写入/读取。

字节顺序的最常见格式化程序如下：

| 字节顺序 |
| --- |
| 读取和写入二进制数据 |
| 本地 |
| 小端 |
| 大端 |

如果没有指定这些选项中的任何一个，数据将以您系统的本机字节顺序进行编码，并且将按照在系统内存中的自然对齐方式进行对齐。强烈不建议以这种方式保存数据，因为能够读取它的唯一系统是保存它的系统。

对于数据本身，每种数据类型由一个单个字符表示，每个字符定义数据的类型（整数、浮点数、字符串）和其大小：

| 格式 | C 类型 | Python 类型 | 大小（字节） |
| --- | --- | --- | --- |
| `x` | 填充字节 | 无值 |  |
| `c` | `char` | 长度为 1 的字节 | 1 |
| `b` | 有符号`char` | 整数 | 1 |
| `B` | 无符号`char` | 整数 | 1 |
| `?` | `_Bool` | 布尔值 | 1 |
| `h` | `short` | 整数 | 2 |
| `H` | 无符号`short` | 整数 | 2 |
| `i` | `int` | 整数 | 4 |
| `I` | 无符号`int` | 整数 | 4 |
| `l` | `long` | 整数 | 4 |
| `L` | 无符号`long` | 整数 | 4 |
| `q` | `long long` | 整数 | 8 |
| `Q` | 无符号`long long` | 整数 | 8 |
| `n` | `ssize_t` | 整数 |  |
| `N` | `size_t` | 整数 |  |
| `e` | 半精度`float` | 浮点数 | 2 |
| `f` | `float` | 浮点数 | 4 |
| `d` | `double` | 浮点数 | 8 |
| `s` | `char[]` | 字节 |  |
| `p` | `char[]` | 字节 |  |
| `P` | `void *` | 整数 |  |

# 压缩目录

存档文件是以一种好的方式来分发整个目录，就好像它们是单个文件，并且可以减小分发文件的大小。

Python 内置支持创建 ZIP 存档文件，可以利用它来压缩整个目录。

# 如何实现...

这个食谱的步骤如下：

1.  `zipfile`模块允许我们创建由多个文件组成的压缩 ZIP 存档：

```py
import zipfile
import os

def zipdir(archive_name, directory):
    with zipfile.ZipFile(
        archive_name, 'w', compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                abspath = os.path.join(root, filename)
                relpath = os.path.relpath(abspath, directory)
                archive.write(abspath, relpath)        
```

1.  使用`zipdir`就像提供应该创建的`.zip`文件的名称和应该存档的目录的路径一样简单：

```py
zipdir('/tmp/test.zip', '_build/doctrees')
```

1.  在这种情况下，我压缩了包含本书文档树的目录。存档准备好后，我们可以通过再次使用`zipfile`打开它并列出包含的条目来验证其内容：

```py
>>> with zipfile.ZipFile('/tmp/test.zip') as archive:
...     for n in archive.namelist():
...         print(n)
algorithms.doctree
concurrency.doctree
crypto.doctree
datastructures.doctree
datetimes.doctree
devtools.doctree
environment.pickle
filesdirs.doctree
gui.doctree
index.doctree
io.doctree
multimedia.doctree
```

# 如何实现...

`zipfile.ZipFile`首先以`ZIP_DEFLATED`压缩（这意味着用标准 ZIP 格式压缩数据）的写模式打开。这允许我们对存档进行更改，然后在退出上下文管理器的主体时自动刷新并关闭存档。

在上下文中，我们依靠`os.walk`来遍历整个目录及其所有子目录，并找到所有包含的文件。

对于在每个目录中找到的每个文件，我们构建两个路径：绝对路径和相对路径。

绝对路径是必需的，以告诉`ZipFile`从哪里读取需要添加到存档中的数据，相对路径用于为写入存档的数据提供适当的名称。这样，我们写入存档的每个文件都将以磁盘上的名称命名，但是不会存储其完整路径（`/home/amol/pystlcookbook/_build/doctrees/io.doctree`），而是以相对路径（`_build/doctrees/io.doctree`）存储，因此，如果存档被解压缩，文件将相对于我们正在解压缩的目录创建，而不是以长而无意义的路径结束，这个路径类似于文件在我的计算机上的路径。

一旦文件的路径和应该用来存储它的名称准备好，它们就被提供给`ZipFile.write`来实际将文件写入存档。

一旦所有文件都被写入，我们退出上下文管理器，存档最终被刷新。

# Pickling and shelving

如果您的软件需要大量信息，或者如果您希望在不同运行之间保留历史记录，除了将其保存在某个地方并在下次运行时加载它之外，几乎没有其他选择。

手动保存和加载数据可能会很繁琐且容易出错，特别是如果数据结构很复杂。

因此，Python 提供了一个非常方便的模块`shelve`，允许我们保存和恢复任何类型的 Python 对象，只要可以对它们进行`pickle`。

# 如何实现...

执行以下步骤以完成此食谱：

1.  `shelve`，由`shelve`实现，可以像 Python 中的任何其他文件一样打开。一旦打开，就可以像字典一样将键读入其中：

```py
>>> with shelve.open('/tmp/shelf.db') as shelf:
...   shelf['value'] = 5
... 
```

1.  存储到`shelf`中的值也可以作为字典读回：

```py
>>> with shelve.open('/tmp/shelf.db') as shelf:
...   print(shelf['value'])
... 
5
```

1.  复杂的值，甚至自定义类，都可以存储在`shelve`中：

```py
>>> class MyClass(object):
...   def __init__(self, value):
...     self.value = value
... 
>>> with shelve.open('/tmp/shelf.db') as shelf:
...   shelf['value'] = MyClass(5)
... 
>>> with shelve.open('/tmp/shelf.db') as shelf:
...   print(shelf['value'])
... 
<__main__.MyClass object at 0x101e90d30>
>>> with shelve.open('/tmp/shelf.db') as shelf:
...   print(shelf['value'].value)
... 
5
```

# 它的工作原理...

`shelve` 模块被实现为管理`dbm`数据库的上下文管理器。

当上下文进入时，数据库被打开，并且因为`shelf`是一个字典，所以包含的对象变得可访问。

每个对象都作为一个 pickled 对象存储在数据库中。这意味着在存储之前，每个对象都使用`pickle`进行编码，并产生一个序列化字符串：

```py
>>> import pickle
>>> pickle.dumps(MyClass(5))
b'\x80\x03c__main__\nMyClass\nq\x00)\x81q\x01}'
b'q\x02X\x05\x00\x00\x00valueq\x03K\x05sb.'
```

这允许`shelve`存储任何类型的 Python 对象，甚至自定义类，只要它们在读取对象时再次可用。

然后，当上下文退出时，所有已更改的`shelf`键都将通过在关闭`shelf`时调用`shelf.sync`写回磁盘。

# 还有更多...

在使用`shelve`时需要注意一些事项。

首先，`shelve`不跟踪突变。如果您将可变对象（如`dict`或`list`）存储在`shelf`中，则对其进行的任何更改都不会被保存。只有对`shelf`本身的根键的更改才会被跟踪：

```py
>>> with shelve.open('/tmp/shelf.db') as shelf:
...   shelf['value'].value = 10
... 
>>> with shelve.open('/tmp/shelf.db') as shelf:
...   print(shelf['value'].value)
... 
5
```

这只是意味着您需要重新分配您想要改变的任何值：

```py
>>> with shelve.open('/tmp/shelf.db') as shelf:
...   myvalue = shelf['value']
...   myvalue.value = 10
...   shelf['value'] = myvalue
... 
>>> with shelve.open('/tmp/shelf.db') as shelf:
...   print(shelf['value'].value)
... 
10
```

`shelve` 不允许多个进程或线程同时进行并发读/写。如果要从多个进程访问相同的`shelf`，则必须使用锁（例如使用`fcntl.flock`）来包装`shelf`访问。

# 读取配置文件

当您的软件有太多的选项无法通过命令行简单地传递它们，或者当您希望确保用户不必每次启动应用程序时手动提供它们时，从配置文件加载这些选项是最常见的解决方案之一。

配置文件应该易于人类阅读和编写，因为他们经常会与它们一起工作，而最常见的要求之一是允许注释，以便用户可以在配置中写下为什么设置某些选项或如何计算某些值的原因。这样，当用户在六个月后回到配置文件时，他们仍然会知道这些选项的原因。

因此，通常依赖于 JSON 或机器-机器格式来配置选项并不是很好，因此最好使用特定于配置的格式。

最长寿的配置格式之一是`.ini`文件，它允许我们使用`[section]`语法声明多个部分，并使用`name = value`语法设置选项。

生成的配置文件将如下所示：

```py
[main]
debug = true
path = /tmp
frequency = 30
```

另一个很大的优势是我们可以轻松地从 Python 中读取`.ini`文件。

# 如何做到这一点...

本教程的步骤是：

1.  大多数加载和解析`.ini`的工作可以由`configparser`模块本身完成，但我们将扩展它以实现每个部分的默认值和转换器：

```py
import configparser

def read_config(config_text, schema=None):
    """Read options from ``config_text`` applying given ``schema``"""
    schema = schema or {}

    cfg = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    try:
        cfg.read_string(config_text)
    except configparser.MissingSectionHeaderError:
        config_text = '[main]\n' + config_text
        cfg.read_string(config_text)

    config = {}
    for section in schema:
        options = config.setdefault(section, {})
        for option, option_schema in schema[section].items():
            options[option] = option_schema.get('default')
    for section in cfg.sections():
        options = config.setdefault(section, {})
        section_schema = schema.get(section, {})
        for option in cfg.options(section):
            option_schema = section_schema.get(option, {})
            getter = 'get' + option_schema.get('type', '')
            options[option] = getattr(cfg, getter)(section, option)
    return config
```

1.  使用提供的函数就像提供一个应该用于解析它的配置和模式一样容易：

```py
config_text = '''
debug = true

[registry]
name = Alessandro
surname = Molina

[extra]
likes = spicy food
countrycode = 39
'''

config = read_config(config_text, {
    'main': {
        'debug': {'type': 'boolean'}
    },
    'registry': {
        'name': {'default': 'unknown'},
        'surname': {'default': 'unknown'},
        'middlename': {'default': ''},
    },
    'extra': {
        'countrycode': {'type': 'int'},
        'age': {'type': 'int', 'default': 0}
    },
    'more': {
        'verbose': {'type': 'int', 'default': 0}
    }
})
```

生成的配置字典`config`将包含配置中提供的所有选项或在模式中声明的选项，转换为模式中指定的类型：

```py
>>> import pprint
>>> pprint.pprint(config)
{'extra': {'age': 0, 'countrycode': 39, 'likes': 'spicy food'},
 'main': {'debug': True},
 'more': {'verbose': 0},
 'registry': {'middlename': 'unknown',
              'name': 'Alessandro',
              'surname': 'Molina'}}
```

# 它的工作原理...

`read_config`函数执行三件主要事情：

+   允许我们解析没有部分的简单`config`文件的纯列表选项：

```py
option1 = value1
option2 = value2
```

+   为配置的`default`模式中声明的所有选项应用默认值。

+   将所有值转换为模式中提供的`type`。

第一个特性是通过捕获解析过程中引发的任何`MissingSectionHeaderError`异常来提供的，并在缺少时自动添加`[main]`部分。所有未在任何部分中提供的选项都将记录在`main`部分下。

提供默认值是通过首先遍历模式中声明的所有部分和选项，并将它们设置为其`default`中提供的值或者如果没有提供默认值，则设置为`None`来完成的。

在第二次遍历中，所有默认值都将被实际存储在配置中的值所覆盖。

在第二次遍历期间，对于每个被设置的值，该选项的`type`在模式中被查找。通过在类型前加上`get`单词来构建诸如`getboolean`或`getint`的字符串。这导致成为需要用于将配置选项解析为请求的类型的`configparser`方法的名称。

如果没有提供`type`，则使用空字符串。这导致使用普通的`.get`方法，该方法将值读取为文本。因此，不提供`type`意味着将选项视为普通字符串。

然后，所有获取和转换的选项都存储在字典中，这样就可以通过`config[section][name]`的表示法更容易地访问转换后的值，而无需总是调用访问器，例如`.getboolean`。

# 还有更多...

提供给`ConfigParser`对象的`interpolation=configparser.ExtendedInterpolation()`参数还启用了一种插值模式，允许我们引用配置文件中其他部分的值。

这很方便，可以避免一遍又一遍地重复相同的值，例如，当提供应该都从同一个根开始的多个路径时：

```py
[paths]
root = /tmp/test01
images = ${root}/images
sounds = ${root}/sounds
```

此外，该语法允许我们引用其他部分中的选项：

```py
[main]
root = /tmp/test01

[paths]
images = ${main:root}/images
sounds = ${main:root}/sounds
```

`ConfigParser`的另一个便利功能是，如果要使一个选项在所有部分中都可用，只需在特殊的`[DEFAULT]`部分中指定它。

这将使该选项在所有其他部分中都可用，除非在该部分本身中明确覆盖它：

```py
>>> config = read_config('''
... [DEFAULT]
... option = 1
... 
... [section1]
... 
... [section2]
... option = 5
... ''')
>>> config
{'section1': {'option': '1'}, 
 'section2': {'option': '5'}}
```

# 编写 XML/HTML 内容

编写基于 SGML 的语言通常并不是很困难，大多数语言都提供了用于处理它们的实用程序，但是如果文档变得太大，那么在尝试以编程方式构建元素树时很容易迷失。

最终会有数百个`.addChild`或类似的调用，这些调用都是连续的，这样很难理解我们在文档中的位置以及我们当前正在编辑的部分是什么。

幸运的是，通过将 Python 的`ElementTree`模块与上下文管理器结合起来，我们可以拥有一个解决方案，使我们的代码结构能够与我们试图生成的 XML/HTML 的结构相匹配。

# 如何做...

对于这个配方，执行以下步骤：

1.  我们可以创建一个代表 XML/HTML 文档树的`XMLDocument`类，并且通过允许我们插入标签和文本的`XMLDocumentBuilder`来辅助实际构建文档。

```py
import xml.etree.ElementTree as ET
from contextlib import contextmanager

class XMLDocument:
    def __init__(self, root='document', mode='xml'):
        self._root = ET.Element(root)
        self._mode = mode

    def __str__(self):
        return ET.tostring(self._root, encoding='unicode', method=self._mode)

    def write(self, fobj):
        ET.ElementTree(self._root).write(fobj)

    def __enter__(self):
        return XMLDocumentBuilder(self._root)

    def __exit__(self, exc_type, value, traceback):
        return

class XMLDocumentBuilder:
    def __init__(self, root):
        self._current = [root]

    def tag(self, *args, **kwargs):
        el = ET.Element(*args, **kwargs)
        self._current[-1].append(el)
        @contextmanager
        def _context():
            self._current.append(el)
            try:
                yield el
            finally:
                self._current.pop()
        return _context()

    def text(self, text):
        if self._current[-1].text is None:
            self._current[-1].text = ''
        self._current[-1].text += text
```

1.  然后，我们可以使用我们的`XMLDocument`来构建我们想要的文档。例如，我们可以在 HTML 模式下构建网页：

```py
doc = XMLDocument('html', mode='html')

with doc as _:
    with _.tag('head'):
        with _.tag('title'): _.text('This is the title')
    with _.tag('body'):
        with _.tag('div', id='main-div'):
            with _.tag('h1'): _.text('My Document')
            with _.tag('strong'): _.text('Hello World')
            _.tag('img', src='http://via.placeholder.com/150x150')
```

1.  `XMLDocument`支持转换为字符串，因此要查看生成的 XML，我们只需打印它：

```py
>>> print(doc)
<html>
    <head>
        <title>This is the title</title>
    </head>
    <body>
        <div id="main-div">
            <h1>My Document</h1>
            <strong>Hello World</strong>
            <img src="http://via.placeholder.com/150x150">
        </div>
    </body>
</html>
```

正如您所看到的，我们的代码结构与实际 XML 文档的嵌套相匹配，因此很容易看到`_.tag('body')`中的任何内容都是我们 body 标签的内容。

将生成的文档写入实际文件可以依赖于`XMLDocument.write`方法来完成：

```py
doc.write('/tmp/test.html')
```

# 它是如何工作的...

实际的文档生成是由`xml.etree.ElementTree`执行的，但是如果我们必须使用普通的`xml.etree.ElementTree`生成相同的文档，那么将会导致一堆`el.append`调用：

```py
root = ET.Element('html')
head = ET.Element('head')
root.append(head)
title = ET.Element('title')
title.text = 'This is the title'
head.append(title)
```

这使得我们很难理解我们所在的位置。在这个例子中，我们只是构建一个结构，`<html><head><title>This is the title</title></head></html>`，但是已经很难跟踪`title`在 head 中，依此类推。对于更复杂的文档，这将变得不可能。

因此，虽然我们的`XMLDocument`保留了文档树的`root`并支持将其转换为字符串并将其写入文件，但实际工作是由`XMLDocumentBuilder`完成的。

`XMLDocumentBuilder`保持节点堆栈以跟踪我们在树中的位置（`XMLDocumentBuilder._current`）。该列表的尾部将始终告诉我们当前在哪个标签内。

调用`XMLDocumentBuilder.text`将向当前活动标签添加文本：

```py
doc = XMLDocument('html', mode='html')
with doc as _:
    _.text('Some text, ')
    _.text('and even more')
```

上述代码将生成`<html>Some text, and even more</html>`。

`XMLDocumentBuilder.tag`方法将在当前活动标签中添加一个新标签：

```py
doc = XMLDocument('html', mode='html')
with doc as _:
    _.tag('input', type='text', placeholder='Name?')
    _.tag('input', type='text', placeholder='Surname?')
```

这导致以下结果：

```py
<html>
    <input placeholder="Name?" type="text">
    <input placeholder="Surname?" type="text">
</html>
```

有趣的是，`XMLDocumentBuilder.tag`方法还返回一个上下文管理器。进入时，它将设置输入的标签为当前活动标签，退出时，它将恢复先前的活动节点。

这使我们能够嵌套`XMLDocumentBuilder.tag`调用并生成标签树：

```py
doc = XMLDocument('html', mode='html')
with doc as _:
    with _.tag('head'):
        with _.tag('title') as title: title.text = 'This is a title'
```

这导致以下结果：

```py
<html>
    <head>
        <title>This is a title</title>
    </head>
</html>
```

实际文档节点可以通过`as`获取，因此在先前的示例中，我们能够获取刚刚创建的`title`节点并为其设置文本，但`XMLDocumentBuilder.text`也可以工作，因为`title`节点现在是活动元素，一旦我们进入其上下文。

# 还有更多...

在使用此方法时，我经常应用一个技巧。这使得在 Python 端更难理解发生了什么，这就是我在解释配方本身时避免这样做的原因，但通过消除大部分 Python *噪音*，它使 HTML/XML 结构更加可读。

如果您将`XMLDocumentBuilder.tag`和`XMLDocumentBuilder.text`方法分配给一些简短的名称，您几乎可以忽略调用 Python 函数的事实，并使 XML 结构更相关：

```py
doc = XMLDocument('html', mode='html')
with doc as builder:
    _ = builder.tag
    _t = builder.text

    with _('head'):
        with _('title'): _t('This is the title')
    with _('body'):
        with _('div', id='main-div'):
            with _('h1'): _t('My Document')
            with _('strong'): _t('Hello World')
            _('img', src='http://via.placeholder.com/150x150')
```

以这种方式编写，您实际上只能看到 HTML 标签及其内容，这使得文档结构更加明显。

# 阅读 XML/HTML 内容

阅读 HTML 或 XML 文件使我们能够解析网页内容，并阅读 XML 中描述的文档或配置。

Python 有一个内置的 XML 解析器，`ElementTree`模块非常适合解析 XML 文件，但涉及 HTML 时，由于 HTML 的各种怪癖，它很快就会出现问题。

考虑尝试解析以下 HTML：

```py
<html>
    <body class="main-body">
        <p>hi</p>
        <img><br>
        <input type="text" />
    </body>
</html>
```

您将很快遇到错误：

```py
xml.etree.ElementTree.ParseError: mismatched tag: line 7, column 6
```

幸运的是，调整解析器以处理至少最常见的 HTML 文件并不太难，例如自闭合/空标签。

# 如何做...

对于此配方，您需要执行以下步骤：

1.  `ElementTree`默认使用`expat`解析文档，然后依赖于`xml.etree.ElementTree.TreeBuilder`构建文档的 DOM。

我们可以用基于`HTMLParser`的自己的解析器替换基于`expat`的`XMLParser`，并让`TreeBuilder`依赖于它：

```py
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

class ETHTMLParser(HTMLParser):
    SELF_CLOSING = {'br', 'img', 'area', 'base', 'col', 'command',    
                    'embed', 'hr', 'input', 'keygen', 'link', 
                    'menuitem', 'meta', 'param',
                    'source', 'track', 'wbr'}

    def __init__(self, *args, **kwargs):
        super(ETHTMLParser, self).__init__(*args, **kwargs)
        self._builder = ET.TreeBuilder()
        self._stack = []

    @property
    def _last_tag(self):
        return self._stack[-1] if self._stack else None

    def _handle_selfclosing(self):
        last_tag = self._last_tag
        if last_tag in self.SELF_CLOSING:
            self.handle_endtag(last_tag)

    def handle_starttag(self, tag, attrs):
        self._handle_selfclosing()
        self._stack.append(tag)
        self._builder.start(tag, dict(attrs))

    def handle_endtag(self, tag):
        if tag != self._last_tag:
            self._handle_selfclosing()
        self._stack.pop()
        self._builder.end(tag)

    def handle_data(self, data):
        self._handle_selfclosing()
        self._builder.data(data)

    def close(self):
        return self._builder.close()
```

1.  使用此解析器，我们最终可以成功处理我们的 HTML 文档：

```py
text = '''
<html>
    <body class="main-body">
        <p>hi</p>
        <img><br>
        <input type="text" />
    </body>
</html>
'''

parser = ETHTMLParser()
parser.feed(text)
root = parser.close()
```

1.  我们可以验证我们的`root`节点实际上包含我们原始的 HTML 文档，通过将其打印回来：

```py
>>> print(ET.tostring(root, encoding='unicode'))
<html>
    <body class="main-body">
        <p>hi</p>
        <img /><br />
        <input type="text" />
    </body>
</html>
```

1.  然后，生成的`root`文档可以像任何其他`ElementTree.Element`树一样进行导航：

```py
def print_node(el, depth=0):
    print(' '*depth, el)
    for child in el:
        print_node(child, depth + 1)

>>> print_node(root)
 <Element 'html' at 0x102799a48>
  <Element 'body' at 0x102799ae8>
   <Element 'p' at 0x102799a98>
   <Element 'img' at 0x102799b38>
   <Element 'br' at 0x102799b88>
   <Element 'input' at 0x102799bd8>
```

# 它是如何工作的...

为了构建表示 HTML 文档的`ElementTree.Element`对象树，我们一起使用了两个类：`HTMLParser`读取 HTML 文本，`TreeBuilder`构建`ElementTree.Element`对象树。

每次`HTMLParser`遇到打开或关闭标签时，它将调用`handle_starttag`和`handle_endtag`。当我们遇到这些时，我们通知`TreeBuilder`必须启动一个新元素，然后关闭该元素。

同时，我们在`self._stack`中跟踪上次启动的标签（因此我们当前所在的标签）。这样，我们可以知道当前打开的标签尚未关闭。每次遇到新的打开标签或关闭标签时，我们都会检查上次打开的标签是否是自闭合标签；如果是，我们会在打开或关闭新标签之前关闭它。

这将自动转换代码。考虑以下内容：

```py
<br><p></p>
```

它将被转换为以下内容：

```py
In::
<br></br><p></p>
```

在遇到一个新的开放标签后，当遇到一个自关闭标签（`<br>`）时，`<br>`标签会自动关闭。

它还处理以下代码：

```py
<body><br></body>
```

前面的代码转换为以下内容：

```py
<body><br></br></body>
```

当面对`<br>`自关闭标签后，遇到不同的关闭标签（`</body>`），`<br>`会自动关闭。

即使在处理标签内文本时调用`handle_data`，如果最后一个开放标签是自关闭标签，自关闭标签也会自动关闭：

```py
<p><br>Hello World</p>
```

`Hello World`文本被认为是`<p>`的内容，而不是`<br>`的内容，因为代码被转换为以下内容：

```py
<p><br></br>Hello World</p>
```

最后，一旦完整的文档被解析，调用`ETHTMLParser.close()`将终止`TreeBuilder`构建的树，并返回生成的根`Element`。

# 还有更多...

提出的食谱展示了如何使用`HTMLParser`来适应 XML 解析工具以处理 HTML，与 XML 相比，HTML 的规则更加灵活。

虽然这个解决方案主要处理常见的 HTML 写法，但它不会涵盖所有可能的情况。HTML 支持一些奇怪的情况，有时会使用一些没有值的属性：

```py
<input disabled>
```

或者没有引号的属性：

```py
<input type=text>
```

甚至一些带内容但没有任何关闭标签的属性：

```py
<li>Item 1
<li>Item 2
```

尽管大多数这些格式都得到支持，但它们很少被使用（也许除了没有任何值的属性，我们的解析器会报告其值为`None`之外），所以在大多数情况下，它们不会引起麻烦。但是，如果您真的需要解析支持所有可能的奇怪情况的 HTML，那么最好使用外部库，比如`lxml`或`html5lib`，它们在面对奇怪情况时会尽可能地像浏览器一样行为。

# 读写 CSV

CSV 被认为是表格数据的最佳交换格式之一；几乎所有的电子表格工具都支持读写 CSV，并且可以使用任何纯文本编辑器轻松编辑，因为它对人类来说很容易理解。

只需拆分并用逗号设置值，您几乎已经写了一个 CSV 文档。

Python 对于读取 CSV 文件有非常好的内置支持，我们可以通过`csv`模块轻松地写入或读取 CSV 数据。

我们将看到如何读写表格：

```py
"ID","Name","Surname","Language"
1,"Alessandro","Molina","Italian"
2,"Mika","Häkkinen","Suomi"
3,"Sebastian","Vettel","Deutsch"
```

# 如何做...

让我们看看这个食谱的步骤：

1.  首先，我们将看到如何写指定的表：

```py
import csv

with open('/tmp/table.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(("ID","Name","Surname","Language"))
    writer.writerow((1,"Alessandro","Molina","Italian"))
    writer.writerow((2,"Mika","Häkkinen","Suomi"))
    writer.writerow((3,"Sebastian","Vettel","Deutsch"))
```

1.  `table.csv`文件将包含我们之前看到的相同的表，我们可以使用任何`csv`读取器将其读回。当您的 CSV 文件有标题时，最方便的是`DictReader`，它将使用标题作为键读取每一行到一个字典中：

```py
with open('/tmp/table.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
```

1.  迭代`DictReader`将消耗行，应该打印我们写的相同数据：

```py
{'Surname': 'Molina', 'Language': 'Italian', 'ID': '1', 'Name': 'Alessandro'}
{'Surname': 'Häkkinen', 'Language': 'Suomi', 'ID': '2', 'Name': 'Mika'}
{'Surname': 'Vettel', 'Language': 'Deutsch', 'ID': '3', 'Name': 'Sebastian'}
```

# 还有更多...

CSV 文件是纯文本文件，有一些限制。例如，没有任何东西告诉我们如何编码换行符（`\r\n`或`\n`），也没有告诉我们应该使用哪种编码，`utf-8`还是`ucs-2`。理论上，CSV 甚至没有规定必须是逗号分隔的；很多软件会用`:`或`;`来分隔。

这就是为什么在读取 CSV 文件时，您应该注意提供给`open`函数的`encoding`。在我们的例子中，我们确定使用了`utf8`，因为我们自己写了文件，但在其他情况下，不能保证使用了任何特定的编码。

如果您不确定 CSV 文件的格式，可以尝试使用`csv.Sniffer`对象，当应用于 CSV 文件中包含的文本时，它将尝试检测使用的方言。

一旦方言被确定，您可以将其传递给`csv.reader`，告诉读取器使用该方言解析文件。

# 读写数据库

Python 通常被称为一个*内置电池*的语言，这要归功于它非常完整的标准库，它提供的最好的功能之一就是从一个功能齐全的关系型数据库中读取和写入。

Python 内置了`SQLite`库，这意味着我们可以保存和读取由`SQLite`存储的数据库文件。

使用起来非常简单，实际上大部分只涉及发送 SQL 进行执行。

# 如何做到...

对于这些食谱，步骤如下：

1.  使用`sqlite3`模块，可以创建一个新的数据库文件，创建一个表，并向其中插入条目：

```py
import sqlite3

with sqlite3.connect('/tmp/test.db') as db:
    try:
        db.execute('''CREATE TABLE people (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, 
            name TEXT, 
            surname TEXT, 
            language TEXT
        )''')
    except sqlite3.OperationalError:
        # Table already exists
        pass

    sql = 'INSERT INTO people (name, surname, language) VALUES (?, ?, ?)'
    db.execute(sql, ("Alessandro", "Molina", "Italian"))
    db.execute(sql, ("Mika", "Häkkinen", "Suomi"))
    db.execute(sql, ("Sebastian", "Vettel", "Deutsch"))
```

1.  `sqlite3`模块还提供了对`cursors`的支持，它允许我们将查询的结果从数据库流式传输到你自己的代码：

```py
with sqlite3.connect('/tmp/test.db') as db:
    db.row_factory = sqlite3.Row
    cursor = db.cursor()
    for row in cursor.execute('SELECT * FROM people WHERE language 
                              != :language', 
                              {'language': 'Italian'}):
        print(dict(row))
```

1.  前面的片段将打印存储在我们的数据库中的所有行作为`dict`，键与列名匹配，值与行中每个列的值匹配。

```py
{'name': 'Mika', 'language': 'Suomi', 'surname': 'Häkkinen', 'id': 2}
{'name': 'Sebastian', 'language': 'Deutsch', 'surname': 'Vettel', 'id': 3}
```

# 它是如何工作的...

`sqlite3.connect`用于打开数据库文件；返回的对象可以用于对其执行任何查询，无论是插入还是选择。

然后使用`.execute`方法来运行任何 SQL 代码。要运行的 SQL 以纯字符串的形式提供。

在执行查询时，通常不应直接在 SQL 中提供值，特别是如果这些值是由用户提供的。

想象我们写了以下内容：

```py
cursor.execute('SELECT * FROM people WHERE language != %s' % ('Italian',)):
```

如果用户提供的字符串是`Italian" OR 1=1 OR "`，而不是`Italian`，会发生什么？用户不会过滤结果，而是可以访问表的全部内容。很容易看出，如果查询是通过用户 ID 进行过滤，而表中包含来自多个用户的数据，这可能会成为安全问题。

此外，在`executescript`命令的情况下，用户将能够依赖相同的行为来实际执行任何 SQL 代码，从而将代码注入到我们自己的应用程序中。

因此，`sqlite3`提供了一种方法来传递参数到 SQL 查询并转义它们的内容，这样即使用户提供了恶意输入，也不会发生任何不好的事情。

我们的`INSERT`语句中的`?`占位符和我们的`SELECT`语句中的`:language`占位符正是为了这个目的：依赖于`sqlite`的转义行为。

这两者是等价的，你可以选择使用哪一个。一个适用于元组，而另一个适用于字典。

在从数据库中获取结果时，它们是通过`Cursor`提供的。你可以将光标视为从数据库流式传输数据的东西。每当你需要访问它时，才会读取每一行，从而避免将所有行加载到内存中并一次性传输它们的需要。

虽然这对于常见情况不是一个主要问题，但当读取大量数据时可能会出现问题，直到系统可能会因为消耗太多内存而终止你的 Python 脚本。

默认情况下，从光标读取行会返回元组，其中值的顺序与列的声明顺序相同。通过使用`db.row_factory = sqlite3.Row`，我们确保光标返回`sqlite3.Row`对象作为行。

它们比元组更方便，因为它们可以像元组一样进行索引（你仍然可以写`row[0]`），而且还支持通过列名进行访问（`row['name']`）。我们的片段依赖于`sqlite3.Row`对象可以转换为字典，以打印所有带有列名的行值。

# 还有更多...

`sqlite3`模块支持许多其他功能，例如事务、自定义类型和内存数据库。

自定义类型允许我们将结构化数据读取为 Python 对象，但我最喜欢的功能是支持内存数据库。

在编写软件的测试套件时，使用内存数据库非常方便。如果你编写依赖于`sqlite3`模块的软件，请确保编写连接到`":memory:"`数据库的测试。这将使你的测试更快，并且将避免在每次运行测试时在磁盘上堆积测试数据库文件。
