## 第十一章：11

输入/输出、物理格式和逻辑布局

计算通常与持久数据一起工作。可能有源数据需要分析，或者使用 Python 的输入和输出操作创建输出。在游戏中探索的地牢图是游戏应用程序将输入的数据。图像、声音和电影是某些应用程序输出的数据，由其他应用程序输入。甚至通过网络发送的请求也会涉及输入和输出操作。所有这些的共同之处在于数据文件的概念。术语“文件”具有许多含义：

+   操作系统（OS）使用文件作为在设备上组织数据字节的方式。解析字节是应用程序软件的责任。两种常见的设备在操作系统文件的功能方面提供了不同的变体：

    +   块设备，如磁盘或固态驱动器（SSD）：这类设备上的文件可以定位任何特定的字节，这使得它们特别适合数据库，因为任何行都可以在任何时候进行处理。

    +   字符设备，如网络连接、键盘或 GPS 天线。这类设备上的文件被视为传输中的单个字节流。无法向前或向后查找；字节必须被捕获在缓冲区中，并按到达顺序进行处理。

+   “文件”这个词还定义了 Python 运行时使用的数据结构。统一的 Python 文件抽象封装了各种操作系统文件实现。当我们打开 Python 文件时，Python 抽象、操作系统实现以及块设备上的底层字节集合或字符设备的字节流之间存在绑定。

Python 为我们提供了两种常见模式来处理文件内容：

+   在“b”（二进制）模式下，我们的应用程序看到的是字节，而不进行进一步解释。这有助于处理具有复杂编码的媒体数据，如图像、音频和电影。我们通常会导入像 pillow 这样的库来处理图像文件编码为字节以及从字节解码的细节。

+   在“t”（文本）模式下，文件的字节是字符串值的编码。Python 字符串由 Unicode 字符组成，有多种方案用于将字节解码为文本以及将文本编码为字节。通常，操作系统有一个首选的编码，Python 会尊重这一点。UTF-8 编码很受欢迎。从实用主义的角度来看，文件可以具有任何可用的 Unicode 编码，并且可能不清楚使用了哪种编码来创建文件。

此外，Python 模块如 shelve 和 pickle 有独特的方式来表示比简单字符串更复杂的 Python 对象。有几种 pickle 协议可用；它们都基于二进制模式文件操作。

在本章中，我们将讨论 Python 对象的序列化。序列化创建了一系列字节，以表示 Python 对象的状态。反序列化是相反的过程：它从文件的字节中恢复 Python 对象的状态。保存和传输对象状态的表示是 REST 网络服务背后的基本概念。

当我们处理来自文件的数据时，我们有两个常见的问题：

+   数据的物理格式：我们需要知道如何解释文件上的字节来重建 Python 对象。字节可能代表 JPEG 编码的图像或 MPEG 编码的电影。一个非常常见的例子是表示 Unicode 文本的文件字节，组织成行。通常，物理格式问题由 Python 库如 csv、json 和 pickle 等处理。

+   数据的逻辑布局：给定的数据集合可能有灵活的位置来存储数据项。CSV 列或 JSON 字段的排列可以不同。在数据包含标签的情况下，逻辑布局是清晰的。如果没有标签，布局是位置性的，需要一些额外的模式信息来识别哪些数据项占据了各种位置。

物理格式解码和逻辑布局模式对于解释文件上的数据都是必不可少的。我们将探讨处理不同物理格式的多个食谱。我们还将探讨将我们的程序与逻辑布局的一些方面分离的方法。

在本章中，我们将探讨以下食谱：

+   使用 pathlib 处理文件名

+   替换文件同时保留旧版本

+   使用 CSV 模块读取分隔文件

+   使用 dataclasses 简化处理 CSV 文件

+   使用正则表达式读取复杂格式

+   读取 JSON 和 YAML 文档

+   读取 XML 文档

+   读取 HTML 文档

为了处理文件，我们将从帮助控制操作系统文件系统的对象开始。Python 的 pathlib 模块描述了文件和设备的目录结构的常见特性。此模块在多个操作系统上具有一致的行为，使得 Python 程序在 Linux、macOS 和 Windows 上可以以类似的方式工作。

# 11.1 使用 pathlib 处理文件名

大多数操作系统使用包含文件的目录树结构。从根目录到特定文件的路径通常表示为一个字符串。以下是一个示例路径：

```py
/Users/slott/Documents/Writing/Python Cookbook/src/ch11/recipe_01.py
```

这个完整路径名列出了包含在（未命名的）根目录中的七个命名目录。最后的名称有一个 recipe_01 的前缀和 .py 的后缀。

我们可以将其表示为一个字符串，并解析该字符串以定位目录名称、文件名和后缀字符串。这样做在 macOS 和 Linux 操作系统之间不可移植，它们使用"/"作为分隔符，而 Windows 使用"\"作为分隔符。此外，Windows 文件可能还有设备名称作为路径的前缀。

处理文件名中的“/”或目录名中的“.”等边缘情况会使字符串处理变得不必要地困难。我们可以通过使用 pathlib.Path 对象而不是字符串来简化解析和许多文件系统操作。

## 11.1.1 准备工作

重要的是要区分三个概念：

+   一个标识文件的路径，包括文件名

+   文件元数据，如创建时间戳和所有权，存储在目录树中

+   文件内容

文件内容与目录信息无关。多个目录条目链接到相同内容是很常见的。这可以通过硬链接来完成，其中目录信息在多个路径之间共享，以及软链接，其中一种特殊类型的文件包含对另一个文件的引用。

通常，文件名有一个后缀（或扩展名），用作关于物理格式的提示。以.csv 结尾的文件名很可能是一个可以解释为行和列数据的文本文件。这种名称与物理格式之间的绑定不是绝对的。文件后缀只是一个提示，可能会出错。

在 Python 中，pathlib 模块处理所有与路径相关的处理。该模块在路径之间做出几个区分：

+   可能或可能不指向实际文件的纯路径

+   解析的具体路径；这些指的是一个实际文件

这种区分使我们能够为应用可能创建或引用的文件创建纯路径。我们也可以为实际存在于操作系统上的文件创建具体路径。应用程序通常将纯路径解析为具体路径。

虽然 pathlib 模块可以在 Linux 路径对象和 Windows 路径对象之间做出区分，但这种区分很少需要。使用 pathlib 的一个重要原因是，我们希望处理与底层操作系统的细节隔离。

本节中的所有迷你食谱都将利用以下内容：

```py
>>> from pathlib import Path
```

我们还假设使用 argparse 模块来收集文件或目录名称。有关 argparse 的更多信息，请参阅第六章中的使用 argparse 获取命令行输入食谱。我们将使用一个 options 变量作为命名空间，该命名空间包含食谱处理的输入文件名或目录名。作为一个例子，我们将使用以下 Namespace 对象：

```py
>>> from argparse import Namespace 

>>> options = Namespace( 

...     input=’/path/to/some/file.csv’, 

...     file1=’data/ch11_file1.yaml’, 

...     file2=’data/ch11_file2.yaml’, 

... ) 
```

通常，我们会定义 argparse 选项使用 type=Path，以便参数解析为我们创建 Path 对象。为了展示 Path 对象的工作方式，路径信息以字符串值的形式提供。

## 11.1.2 如何操作...

在以下迷你食谱中，我们将展示一些常见的路径名操作：

+   通过更改输入文件名后缀来创建输出文件名

+   创建具有不同名称的多个同级输出文件

+   比较文件日期以查看哪个更新

+   查找所有匹配给定模式的文件

前两个反映了处理目录到文件路径的技术；使用 Path 对象比进行复杂的字符串操作要容易得多。最后两个收集有关计算机上具体路径和相关文件的信息。

### 通过更改输入文件名后缀来创建输出文件名

通过更改输入名称的后缀来创建输出文件名的以下步骤：

1.  从输入文件名字符串创建一个 Path 对象：

    ```py
    >>> input_path = Path(options.input) 

    >>> input_path 

    PosixPath(’/path/to/some/file.csv’)
    ```

    显示 PosixPath 类，因为作者正在使用 macOS。在 Windows 机器上，该类将是 WindowsPath。

1.  使用 with_suffix()方法创建输出 Path 对象：

    ```py
    >>> output_path = input_path.with_suffix(’.out’) 

    >>> output_path 

    PosixPath(’/path/to/some/file.out’)
    ```

所有文件名解析都由 Path 类无缝处理。这不会创建具体的输出文件；它只是为它创建了一个新的 Path 对象。

### 创建具有不同名称的多个同级输出文件

通过更改输入名称的后缀来创建具有不同名称的多个同级输出文件：

1.  从输入文件名字符串创建一个 Path 对象：

    ```py
    >>> input_path = Path(options.input)
    ```

1.  从文件名中提取父目录和基本名称。基本名称是没有后缀的名称：

    ```py
    >>> input_directory = input_path.parent 

    >>> input_stem = input_path.stem
    ```

1.  构建所需的输出名称。对于此示例，我们将追加 _pass 到基本名称并构建完整的 Path 对象：

    ```py
    >>> output_stem_pass = f"{input_stem}_pass" 

    >>> output_stem_pass 

    ’file_pass’
    ```

    ```py
    >>> output_path = ( 

    ...     input_directory / output_stem_pass 

    ... ).with_suffix(’.csv’) 

    >>> output_path 

    PosixPath(’/path/to/some/file_pass.csv’)
    ```

/运算符从 Path 组件组装一个新的 Path。我们需要将/运算放在括号中，以确保它首先执行，在更改后缀之前创建一个新的 Path 对象。

### 比较文件日期以查看哪个更新

以下是比较文件日期以查看哪个更新的步骤：

1.  从输入文件名字符串创建 Path 对象。Path 类将正确解析字符串以确定路径的元素：

    ```py
    >>> file1_path = Path(options.file1) 

    >>> file2_path = Path(options.file2)
    ```

    在探索此示例时，请确保 options 对象中的名称是实际文件。

1.  使用每个 Path 对象的 stat()方法获取文件的戳记。在 stat 对象中，st_mtime 属性提供了文件的最新修改时间：

    ```py
    >>> file1_path.stat().st_mtime 

    1572806032.0 

    >>> file2_path.stat().st_mtime 

    1572806131.0
    ```

这些值是以秒为单位的时间戳。您的值将取决于您系统上的文件。如果我们想要一个对大多数人来说都合理的时间戳，我们可以使用 datetime 模块从这个值创建一个更有用的对象：

```py
>>> import datetime 

>>> mtime_1 = file1_path.stat().st_mtime 

>>> datetime.datetime.fromtimestamp(mtime_1) 

datetime.datetime(2019, 11, 3, 13, 33, 52)
```

我们可以使用多种方法来格式化 datetime 对象。

### 查找所有匹配给定模式的文件

以下是要查找所有匹配给定模式的文件的步骤：

1.  从输入目录名创建 Path 对象：

    ```py
    >>> directory_path = Path(options.file1).parent 

    >>> directory_path 

    PosixPath(’data’)
    ```

1.  使用 Path 对象的 glob() 方法定位此目录中所有匹配给定模式的文件。对于不存在的目录，迭代器将为空。在模式中使用 ** 将递归遍历目录树：

    ```py
    >>> from pprint import pprint 

    >>> pprint(sorted(directory_path.glob("*.csv"))) 

    PosixPath(’data/binned.csv’),
    ```

    我们省略了结果中的许多文件。

glob() 方法是一个迭代器，我们使用了 sorted() 函数来消费这个迭代器的值，并从它们创建一个单独的列表对象。

## 11.1.3 它是如何工作的...

在操作系统中，查找文件的目录序列是通过文件系统路径实现的。在某些情况下，可以使用简单的字符串表示来总结路径。然而，字符串表示使得许多路径操作变成了复杂的字符串解析问题。字符串对于操作操作系统路径来说是一个无用的不透明抽象。

Path 类定义简化了路径操作。Path 实例上的这些属性、方法和运算符包括以下示例：

+   .parent 提取父目录。

+   .parents 列出所有封装的目录。

+   .name 是最终名称。

+   .stem 是最终名称的基名（不带任何后缀）。

+   .suffix 是最终的后缀。

+   .suffixes 是后缀值序列，用于与 file.tag.gz 类型的名称一起使用。

+   .with_suffix() 方法用新的后缀替换文件的后缀。

+   .with_name() 方法用新的名称替换路径中的名称。

+   / 操作符从 Path 和字符串组件构建 Path 对象。

具体路径代表实际的文件系统资源。对于具体的路径对象，我们可以对目录信息进行一系列额外的操作：

+   确定这种目录条目是什么类型；即普通文件、目录、链接、套接字、命名管道（或 FIFO）、块设备或字符设备。

+   获取目录详细信息，包括时间戳、权限、所有权、大小等信息。

+   解除链接（即删除）目录条目。请注意，解除普通文件的链接与删除空目录是不同的。我们将在本食谱的 There’s more... 部分中探讨这一点。

+   将文件重命名以将其放置在新的路径中。我们也会在本食谱的 There’s more... 部分中探讨这一点。

几乎我们可以对文件目录执行的所有操作都可以使用 pathlib 模块来完成。少数例外是 os 模块的一部分，因为它们通常是特定于操作系统的。

## 11.1.4 更多内容...

除了操作路径和收集有关文件的信息外，我们还可以对文件系统进行一些更改。两个常见的操作是重命名文件和解除链接（或删除）文件。我们可以使用多种方法来更改文件系统：

+   .unlink() 方法删除普通文件。它不会删除目录。

+   .rmdir() 方法删除空目录。删除包含文件的目录需要两步操作：首先解除目录中所有文件的联系，然后删除目录。

+   .rename() 方法将文件重命名为新路径。

+   .replace() 方法在目标已存在的情况下不会引发异常来替换文件。

+   .symlink_to() 方法创建一个指向现有文件的软链接文件。

+   .hardlink_to() 方法创建一个操作系统硬链接；现在两个不同的目录条目将拥有底层文件内容。

我们可以通过内置的 open() 函数或 open() 方法来打开一个路径。有些人喜欢看到 open(some_path)，而有些人则更喜欢 some_path.open()。两者都做同样的事情：创建一个打开的文件对象。

我们可以使用 mkdir() 方法创建目录。此方法有两个关键字参数：

+   exist_ok=False 是默认值；如果目录已存在，将引发异常。将此更改为 True 使代码对现有目录具有容错性。

+   parents=False 是默认值；不会创建父目录，只有路径中的最底层目录。将此更改为 True 将创建整个路径，包括父目录和子目录。

我们还可以以大字符串或字节对象的形式读取和写入文件：

+   .read_text() 方法将文件作为单个字符串读取。

+   .write_text() 方法使用给定的字符串创建或替换文件。

+   .read_bytes() 方法将文件作为单个字节实例读取。

+   .write_bytes() 方法使用给定的字节创建或替换文件。

还有更多文件系统操作，如更改所有权或更改权限。这些操作在 os 模块中可用。

## 11.1.5 参见

+   在本章后面的 [在保留先前版本的同时替换文件 菜谱中，我们将探讨如何利用路径对象的功能来创建一个临时文件，然后将临时文件重命名为替换原始文件。

+   在第六章的 使用 argparse 获取命令行输入 菜谱中，我们查看了一种非常常见的方法，即使用字符串来创建路径对象。

+   os 模块提供了一些比 pathlib 提供的更不常用的文件系统操作。

# 11.2 在保留先前版本的同时替换文件

我们可以利用 pathlib 模块的功能来支持各种文件名操作。在第 使用 pathlib 处理文件名 菜谱中，我们查看了一些管理目录、文件名和文件后缀的最常见技术。

一个常见的文件处理需求是以安全的方式创建输出文件；也就是说，无论应用程序如何或在哪里失败，应用程序都应该保留任何先前的输出文件。

考虑以下场景：

1.  在时间 T[0] 时，有一个来自 long_complex.py 应用程序先前运行的 valid output.csv 文件。

1.  在时间 T[1] 时，我们开始使用新数据运行 long_complex.py 应用程序。它开始覆盖 output.csv 文件。直到程序完成，字节将不可用。

1.  在时间 T[2] 时，应用程序崩溃。output.csv 文件的局部内容是无用的。更糟糕的是，时间 T[0] 的有效文件也不再可用，因为它被覆盖了。

在这个菜谱中，我们将探讨在失败情况下创建输出文件的一种安全方法。

## 11.2.1 准备工作

对于不跨越物理设备的文件，安全文件输出通常意味着使用临时名称创建文件的新副本。如果新文件可以成功创建，则应使用单个原子重命名操作替换旧文件。

我们希望有以下功能：

+   重要的输出文件必须始终以有效状态保存。

+   应用程序写入文件的临时版本。命名此文件有许多约定。有时，在文件名上放置额外的字符，如 ~ 或 #，以指示它是临时的工作文件；例如，output.csv~。我们将使用更长的后缀，.new；例如，output.csv.new。

+   文件的先前版本也被保留。有时，先前版本有一个 .bak 后缀，表示“备份”。我们将使用更长的后缀，称为 output.csv.old。这也意味着任何先前的 .old 文件都必须作为最终输出的一部分被删除；只保留一个版本。

为了创建一个具体的例子，我们将使用一个包含非常小但珍贵的部分数据的文件：一系列商对象。以下是商类的定义：

```py
from dataclasses import dataclass, asdict, fields 

@dataclass 

class Quotient: 

    numerator: int 

    denominator: int 
```

以下函数将对象写入 CSV 格式的文件：

```py
import csv 

from collections.abc import Iterable 

from pathlib import Path 

def save_data( 

    output_path: Path, data: Iterable[Quotient] 

) -> None: 

    with output_path.open("w", newline="") as output_file: 

        headers = [f.name for f in fields(Quotient)] 

        writer = csv.DictWriter(output_file, headers) 

        writer.writeheader() 

        for q in data: 

            writer.writerow(asdict(q))
```

如果在将数据对象写入文件时出现问题，我们可能会留下一个损坏的、不可用的文件。我们将用另一个函数包装此函数，以提供可靠的写入。

## 11.2.2 如何操作...

我们通过导入所需的类开始创建一个包装函数：

1.  定义一个函数来封装 save_data() 函数以及一些额外功能。函数签名与 save_data() 函数相同：

1.  保存原始后缀，并在后缀末尾创建一个带有 .new 的新名称。这是一个临时文件。如果它正确写入，没有异常，那么我们可以重命名它，使其成为目标文件：

    ```py
        ext = output_path.suffix 

        output_new_path = output_path.with_suffix(f’{ext}.new’) 

        save_data(output_new_path, data)
    ```

    save_data() 函数是封装在此函数中的创建新文件的原始过程。

1.  在用新文件替换旧文件之前，删除任何先前的备份副本。如果存在，我们将解除 .old 文件的链接：

    ```py
        output_old_path = output_path.with_suffix(f’{ext}.old’) 

        output_old_path.unlink(missing_ok=True)
    ```

1.  现在，我们可以保留任何先前的良好文件，其名称为 .old：

    ```py
        try: 

            output_path.rename(output_old_path) 

        except FileNotFoundError as ex: 

            # No previous file. That’s okay. 

            pass
    ```

1.  最后一步是将临时的 .new 文件变为官方输出：

    ```py
        try: 

            output_new_path.rename(output_path) 

        except IOError as ex: 

            # Possible recovery... 

            output_old_path.rename(output_path)
    ```

这个多步骤过程使用两个重命名操作：

+   将先前的版本重命名为带有后缀 .old 的备份版本。

+   将带有 .new 后缀的新版本重命名为文件的当前版本。

|

* * *

## 11.2.3 它是如何工作的...

此过程涉及三个独立的操作系统操作：一个解除链接和两个重命名。这是为了确保保留一个 .old 文件，并可以使用它来恢复之前良好的状态。

这里是一个显示各种文件状态的时序表。我们将内容标记为版本 0（一些旧数据）、版本 1（当前有效数据）和版本 2（新创建的数据）：

由于这些操作是串行应用的，因此在保留旧文件和重命名新文件之间存在一个非常小的时间间隔，应用程序失败将无法替换新文件。我们将在下一节中探讨这一点。

* * *

* * *

* * *

* * *

* * *

|

* * *

* * *

| T[0] |  | 版本 0 | 版本 1 |  |

|

| * * * |
| --- |

|

|

|

|

* * *

* * *

| T[5] | 将 .csv.new 重命名为 .csv 后 | 版本 1 | 版本 2 |  |

如果有 .csv 文件，它是当前的有效文件。

* * *

* * *

| T[2] | 创建后，关闭 | 版本 0 | 版本 1 | 版本 2 |

| |
| --- |

|

* * *

|

* * *

|

* * *

|

* * *

|

* * *

|

| * * * |
| --- |

|

|

|

|

|

|

* * *

* * *

一个路径对象有一个 replace() 方法。这总是覆盖目标文件，如果覆盖现有文件则没有警告。rename() 和 replace() 之间的选择取决于我们的应用程序如何处理在文件系统中可能留下旧版本文件的情况。在这个菜谱中，我们使用了 rename() 来尝试避免在多个问题的情况下覆盖文件。

* * *

|

| * * * |
| --- |

|

* * *

* * *

虽然有几次失败的机会，但关于哪个文件是有效的没有歧义：

* * *

* * *

* * *

* * *

由于这些操作实际上并不涉及复制文件，因此这些操作都非常快且可靠。然而，它们并不保证一定能工作。文件系统的状态可以被任何具有正确权限的用户更改，因此在创建替换旧文件的新的文件时需要小心。

|

|

| T[3] | 解除链接 .csv.old 后 |  | 版本 1 | 版本 2 |
| --- | --- | --- | --- | --- |

* * *

|

|

* * *

|

* * *

|

为了确保输出文件有效，一些应用程序会采取额外步骤，在文件中写入一个最终的校验和行，以提供明确的证据，表明文件是完整且一致的。

|

|

|

| T[4] | 将 .csv 重命名为 .csv.old 后 | 版本 1 |  | 版本 2 |
| --- | --- | --- | --- | --- |

|

|

|

* * *

|

* * *

|

|

* * *

| T[1] | 创建中 | 版本 0 | 版本 1 | 如果使用将出现损坏 |

|

| * * * |
| --- |

|

|

| 时间 | 操作 | .csv.old | .csv | .csv.new |

|

|

* * *

表 11.1：文件操作时间线

|

|

* * *

|

| * * * |
| --- |

|

|  |  |  |  |  |

+   |

+   如果没有 .csv 文件，那么 .csv.old 文件是一个有效的备份副本，应用于恢复。参见 T[4] 时刻，了解这种情况。

|

|

## 11.2.4 更多内容...

在某些企业应用中，输出文件被组织成基于时间戳命名的目录。这些操作可以通过 pathlib 模块优雅地处理。例如，我们可能有一个用于旧文件的存档目录。这个目录包含带有日期戳的子目录，用于存储临时或工作文件。

然后，我们可以执行以下操作来定义一个工作目录：

[firstline=58,lastline=58,gobble=4][python]src/ch11/recipe˙02.py [firstline=60,lastline=65,gobble=4][python]src/ch11/recipe˙02.py

mkdir() 方法将创建预期的目录。通过包含 parents=True 参数，任何需要的父目录也将被创建。这可以在应用程序第一次执行时创建 archive_path 非常方便。exists_ok=True 将避免在存档目录已存在时引发异常。

对于某些应用，使用 tempfile 模块创建临时文件可能是合适的。此模块可以创建保证唯一的文件名。这允许复杂的服务器进程在无需考虑文件名冲突的情况下创建临时文件。

## 11.2.5 参见

+   在本章前面的使用 pathlib 处理文件名配方中，我们探讨了 Path 类的基本原理。

+   在第十五章，我们将探讨一些编写单元测试的技术，以确保本配方示例代码的部分行为正确。

+   在第六章，创建上下文和上下文管理器配方展示了有关使用 with 语句确保文件操作正确完成以及释放所有 OS 资源的更多细节。

+   shutil 模块提供了一系列用于复制文件和包含文件的目录的方法。这个包反映了 Linux shell 程序（如 cp）以及 Windows 程序（如 copy 和 xcopy）的功能。

# 11.3 使用 CSV 模块读取分隔符文件

一种常用的数据格式是逗号分隔值（CSV）。我们可以将逗号字符视为众多候选分隔符之一。例如，CSV 文件可以使用 | 字符作为数据列之间的分隔符。这种对非逗号分隔符的泛化使得 CSV 文件特别强大。

我们如何处理各种 CSV 格式的数据？

## 11.3.1 准备工作

文件内容的摘要称为模式。区分模式的两个方面是至关重要的。

CSV 文件的字节物理格式编码文本行。对于 CSV 文件，文本使用行分隔符字符（或字符序列）和列分隔符字符组织成行和列。许多电子表格产品将使用 ,（逗号）作为列分隔符，将 \r\n 字符序列作为行分隔符。使用的标点符号字符的具体组合称为 CSV 语法。

此外，当列数据包含分隔符之一时，可以引用列数据。最常见的引用规则是用"字符包围列值。为了在列数据中包含引号字符，引号字符被加倍。例如，"He said, ""Thanks."""。

文件中数据的逻辑布局是一系列存在的数据列。在 CSV 文件中处理逻辑布局有几种常见情况：

+   文件可能有一行标题。这与 csv 模块的工作方式很好地吻合。如果标题也是合适的 Python 变量名，那么这会更有帮助。模式在文件的第一行中明确声明。

+   文件没有标题，但列位置是固定的。在这种情况下，我们可以在打开文件时在文件上施加标题。从实用主义的角度来看，这涉及一些风险，因为很难确认数据符合施加的模式。

+   如果文件没有标题且列位置不固定。在这种情况下，需要额外的外部模式信息来解释数据列。

当然，任何数据都可能出现的某些常见复杂问题。有些文件不是第一范式（1NF）。在 1NF 中，每一行都是独立于所有其他行的。当一个文件不处于这种范式时，我们需要添加一个生成器函数来重新排列数据以形成 1NF 行。请参阅第四章中的 Slicing and dicing a list 配方，以及第九章中的 Using stacked generator expressions 配方，以了解其他显示如何规范化数据结构的配方。

我们将查看一个包含从帆船日志中记录的一些实时数据的 CSV 文件。这是 waypoints.csv 文件。数据如下所示：

```py
lat,lon,date,time 

32.8321666666667,-79.9338333333333,2012-11-27,09:15:00 

31.6714833333333,-80.93325,2012-11-28,00:00:00 

30.7171666666667,-81.5525,2012-11-28,11:35:00
```

这些数据包含文件第一行中命名的四个列：lat、lon、date 和 time。这些描述了一个航点，需要重新格式化以创建更有用的信息。

## 11.3.2 如何实现...

在开始编写任何代码之前，检查数据文件以确认以下功能：

+   列分隔符字符是’,’，这是默认值。

+   行分隔符字符是’\r\n’，在 Windows 和 Linux 中都广泛使用。

+   有一个单行标题。如果不存在，当创建读取器对象时，应单独提供标题。

一旦格式得到确认，我们就可以开始创建所需的函数，如下所示：

1.  导入 csv 模块和 Path 类：

    ```py
    import csv 

    from pathlib import Path
    ```

1.  定义一个 raw()函数，从指向文件的 Path 对象中读取原始数据：

    ```py
    def raw(data_path: Path) -> None:
    ```

1.  使用 Path 对象在 with 语句中打开文件。从打开的文件构建读取器：

    ```py
        with data_path.open() as data_file: 

            data_reader = csv.DictReader(data_file)
    ```

1.  消费（并处理）可迭代读取器的数据行。这正确地缩进在 with 语句内：

    ```py
            for row in data_reader: 

                print(row)
    ```

raw()函数的输出是一系列看起来如下所示的字典：

```py
{’lat’: ’32.8321666666667’, ’lon’: ’-79.9338333333333’, ’date’: ’2012-11-27’, ’time’: ’09:15:00’}
```

我们现在可以通过将列作为字典项来处理数据，使用例如 row[’date’]这样的语法。使用列名比通过位置引用列更具有描述性；例如，row[0]难以理解。

为了确保我们正确地使用了列名，可以使用 typing.TypedDict 类型提示来提供预期的列名。

## 11.3.3 它是如何工作的...

csv 模块处理解析物理格式的工作。这使行彼此分离，并在每行内分离列。默认规则确保每条输入行被视为单独的行，并且列由逗号分隔。

当我们需要将列分隔符字符作为数据的一部分使用时会发生什么？我们可能会有这样的数据：

```py
lan,lon,date,time,notes 

32.832,-79.934,2012-11-27,09:15:00,"breezy, rainy" 

31.671,-80.933,2012-11-28,00:00:00,"blowing ""like stink"""
```

注释列在第一行有数据，包括逗号分隔符字符。CSV 的规则允许列的值被引号包围。默认情况下，引号字符是"。在这些引号字符内，列和行分隔符字符被忽略。

为了在引号字符串中嵌入引号字符，该字符被加倍。第二个示例行显示了如何通过将引号字符加倍来编码值"like stink"，当它们是列值的一部分时。

CSV 文件中的值始终是字符串。像 7331 这样的字符串值可能看起来像数字，但它在 csv 模块处理时始终是文本。这使得处理简单且统一，但可能对我们的 Python 应用程序来说有些尴尬。

当从手动准备的电子表格中保存数据时，数据可能会揭示桌面软件内部数据显示规则的古怪之处。在桌面软件中显示为日期的数据在 CSV 文件中存储为浮点数。

日期作为数字的问题有两个解决方案。一个是向源电子表格中添加一个列，以正确格式化日期数据为字符串。理想情况下，这是使用 ISO 规则完成的，以便日期以 YYYY-MM-DD 格式表示。另一个解决方案是将电子表格中的日期识别为从某个纪元日期以来的秒数。纪元日期随着各种工具的版本略有不同，但通常是 1900 年 1 月 1 日。（一些电子表格应用程序使用 1904 年 1 月 1 日。）

## 11.3.4 更多内容...

正如我们在第九章的结合 map 和 reduce 转换食谱中看到的，通常有一个包括清理和转换源数据的处理流程。这种堆叠生成器函数的想法让 Python 程序能够处理大量数据。一次读取一行可以避免将所有数据读入一个庞大的内存列表。在这个特定例子中，没有需要消除的额外行。然而，每个列都需要转换成更有用的形式。

在第十章中，许多配方使用 Pydantic 来执行这些类型的数据转换。参见 使用 Pydantic 实现更严格的类型检查 配方，了解这种替代方法的示例。

为了将数据转换成更有用的形式，我们将定义一个行级清洗函数。一个函数可以将此清洗函数应用于源数据中的每一行。

在这个例子中，我们将创建一个字典对象，并插入从输入数据派生出的额外值。这个 Waypoint 字典的核心类型提示如下：

```py
import datetime 

from typing import TypeAlias, Any 

Raw: TypeAlias = dict[str, Any] 

Waypoint: TypeAlias = dict[str, Any]
```

基于此对 Waypoint 类型的定义，clean_row() 函数可能看起来像这样：

```py
def clean_row( 

    source_row: Raw 

) -> Waypoint: 

    ts_date = datetime.datetime.strptime( 

        source_row["date"], "%Y-%m-%d").date() 

    ts_time = datetime.datetime.strptime( 

        source_row["time"], "%H:%M:%S").time() 

    return dict( 

        date=source_row["date"], 

        time=source_row["time"], 

        lat=source_row["lat"], 

        lon=source_row["lon"], 

        lat_lon=( 

            float(source_row["lat"]), 

            float(source_row["lon"]) 

        ), 

        ts_date=ts_date, 

        ts_time=ts_time, 

        timestamp = datetime.datetime.combine( 

            ts_date, ts_time 

        ) 

    )
```

clean_row() 函数从原始字符串数据中创建几个新的列值。名为 lat_lon 的列包含一个包含正确浮点值的二元组，而不是字符串。我们还解析了日期和时间值，分别创建了 datetime.date 和 datetime.time 对象。我们将日期和时间合并成一个单一的有用值，即时间戳列的值。

一旦我们有一个用于清洗和丰富我们数据的行级函数，我们就可以将此函数映射到源数据中的每一行。我们可以使用 map(clean_row, reader) 或者我们可以编写一个体现此处理循环的函数：

```py
def cleanse(reader: csv.DictReader[str]) -> Iterator[Waypoint]: 

    for row in reader: 

        yield clean_row(row)
```

这可以用来从每一行提供更有用的数据：

```py
def display_clean(data_path: Path) -> None: 

    with data_path.open() as data_file: 

        data_reader = csv.DictReader(data_file) 

        clean_data_reader = cleanse(data_reader) 

        for row in clean_data_reader: 

            pprint(row)
```

这些清洗和丰富的行看起来如下：

```py
>>> data = Path("data") / "waypoints.csv" 

>>> display_clean(data) 

{’date’: ’2012-11-27’, 

 ’lat’: ’32.8321666666667’, 

 ’lat_lon’: (32.8321666666667, -79.9338333333333), 

 ’lon’: ’-79.9338333333333’, 

 ’time’: ’09:15:00’, 

 ’timestamp’: datetime.datetime(2012, 11, 27, 9, 15), 

 ’ts_date’: datetime.date(2012, 11, 27), 

 ’ts_time’: datetime.time(9, 15)} 

...
```

新的列，如 lat_lon，包含正确的数值而不是字符串。时间戳值包含完整的日期时间值，可用于计算航点之间经过的时间的简单计算。

## 11.3.5 参见

+   参见第九章中的 结合 map 和 reduce 转换 配方，以获取有关处理流程或堆栈的想法的更多信息。

+   参见第四章中的 切片和切块列表 配方，以及第九章中的 使用堆叠生成表达式 配方，以获取有关处理不正确 1NF 的 CSV 文件的更多信息。

+   关于 with 语句的更多信息，请参阅第七章中的 创建上下文和上下文管理器 配方。

+   在第十章中，许多配方使用 Pydantic 来执行这些类型的数据转换。参见 使用 Pydantic 实现更严格的类型检查 配方，了解这种替代方法的示例。

+   查看 [`www.packtpub.com/product/learning-pandas-second-edition/9781787123137`](https://www.packtpub.com/product/learning-pandas-second-edition/9781787123137) 学习 pandas，了解使用 pandas 框架处理 CSV 文件的方法。

# 11.4 使用 dataclasses 简化 CSV 文件的工作

一种常用的数据格式称为逗号分隔值 (CSV)。Python 的 csv 模块有一个非常方便的 DictReader 类定义。当一个文件包含一行标题时，标题行的值成为用于所有后续行的键。这为数据的逻辑布局提供了很大的灵活性。例如，列顺序并不重要，因为每个列的数据都由标题行中的一个名称标识。

使用字典迫使我们编写，例如，row[’lat’] 或 row[’date’] 来引用特定列中的数据。内置的 dict 类没有提供派生数据。如果我们切换到数据类，我们将获得许多好处：

+   更好的属性语法，如 row.lat 或 row.date。

+   派生值可以是延迟属性。

+   冻结的数据类是不可变的，对象可以作为字典的键和集合的成员。

我们如何使用数据类改进数据访问和处理？

## 11.4.1 准备工作

我们将查看一个包含从帆船日志中记录的实时数据的 CSV 文件。此文件是 waypoints.csv 文件。有关更多信息，请参阅本章中的 使用 CSV 模块读取定界文件 烹饪配方。数据如下所示：

```py
lat,lon,date,time 

32.8321666666667,-79.9338333333333,2012-11-27,09:15:00 

31.6714833333333,-80.93325,2012-11-28,00:00:00 

30.7171666666667,-81.5525,2012-11-28,11:35:00
```

第一行包含一个标题，命名了四个列，lat、lon、date 和 time。数据可以通过 csv.DictReader 对象读取。我们希望进行更复杂的工作，因此我们将创建一个 @dataclass 类定义，封装数据和我们需要执行的处理。

## 11.4.2 如何操作...

我们需要从一个反映可用数据的数据类开始，然后我们可以使用这个数据类与字典读取器一起使用：

1.  导入所需的各种库的定义：

    ```py
    from dataclasses import dataclass, field 

    import datetime 

    from collections.abc import Iterator
    ```

1.  定义一个专注于输入的数据类，精确地像源文件中那样出现。我们称这个类为 RawRow。在一个复杂的应用程序中，一个比 RawRow 更有描述性的名称会更合适。这个属性定义可能会随着源文件组织的变化而变化：

    ```py
    @dataclass 

    class RawRow: 

        date: str 

        time: str 

        lat: str 

        lon: str 
    ```

    实际上，企业文件格式很可能在引入新软件版本时发生变化。在发生变化时，将文件模式正式化为类定义通常有助于单元测试和问题解决。

1.  定义第二个数据类，其中对象由源数据类的属性构建。这个第二类专注于应用程序的实际工作。在这个例子中，源数据在一个名为 raw 的单个属性中。从这个源数据计算的字段都使用 field(init=False) 初始化，因为它们将在初始化之后计算：

    ```py
    @dataclass 

    class Waypoint: 

        raw: RawRow 

        lat_lon: tuple[float, float] = field(init=False) 

        ts_date: datetime.date = field(init=False) 

        ts_time: datetime.time = field(init=False) 

        timestamp: datetime.datetime = field(init=False)
    ```

1.  将 `__post_init__()` 方法添加到急切初始化所有派生字段：

    ```py
     def __post_init__(self) -> None: 

            self.ts_date = datetime.datetime.strptime( 

                self.raw.date, "%Y-%m-%d" 

            ).date() 

            self.ts_time = datetime.datetime.strptime( 

                self.raw.time, "%H:%M:%S" 

            ).time() 

            self.lat_lon = ( 

                float(self.raw.lat), 

                float(self.raw.lon) 

            ) 

            self.timestamp = datetime.datetime.combine( 

                self.ts_date, self.ts_time 

            )
    ```

1.  给定这两个数据类定义，我们可以创建一个迭代器，它将接受来自 `csv.DictReader` 对象的单独字典并创建所需的 `Waypoint` 对象。中间表示 `RawRow` 是一个便利，这样我们就可以将属性名称分配给源数据列：

    ```py
    def waypoint_iter(reader: csv.DictReader[str]) -> Iterator[Waypoint]: 

        for row in reader: 

            raw = RawRow(**row) 

            yield Waypoint(raw)
    ```

`waypoint_iter()` 函数从输入字典创建 `RawRow` 对象，然后从 `RawRow` 实例创建最终的 `Waypoint` 对象。这个两步过程有助于隔离对源或处理的代码变更。

我们可以使用以下函数来读取和显示 CSV 数据：

```py
def display(data_path: Path) -> None: 

    with data_path.open() as data_file: 

        data_reader = csv.DictReader(data_file) 

        for waypoint in waypoint_iter(data_reader): 

            pprint(waypoint)
```

## 11.4.3 它是如何工作的...

在这个例子中，源数据类 `RawRow` 类被设计成与输入文档相匹配。字段名称和类型与 CSV 输入类型相匹配。由于名称匹配，`RawRow(**row)` 表达式将从 `DictReader` 字典创建 `RawRow` 类的实例。

从这个初始的或原始的数据中，我们可以推导出更有用的数据，如 `Waypoint` 类定义所示。`__post_init__()` 方法将 `self.raw` 属性中的初始值转换成多个更有用的属性值。

这种分离使我们能够管理应用软件的以下两种常见变更：

1.  由于电子表格是手动调整的，源数据可能会发生变化。这是常见的：一个人可能会更改列名或更改列的顺序。

1.  随着应用程序焦点的扩展或转移，所需的计算可能会发生变化。可能会添加更多派生列，或者算法可能会改变。

将程序的各种方面解开，以便我们可以让它们独立演变，这是很有帮助的。收集、清理和过滤源数据是这种关注点分离的一个方面。由此产生的计算是一个独立的方面，与源数据的格式无关。

## 11.4.4 更多...

在许多情况下，源 CSV 文件将具有不直接映射到有效 Python 属性名称的标题。在这些情况下，源字典中存在的键必须映射到列名。这可以通过扩展 `RawRow` 类定义以包括一个构建 `RawRow` 数据类对象的 `@classmethod` 来管理。

以下示例定义了一个名为 `RawRow_HeaderV2` 的类。这个定义反映了具有不同列名标题的变体电子表格：

```py
@dataclass 

class RawRow_HeaderV2: 

    date: str 

    time: str 

    lat: str 

    lon: str 

    @classmethod 

    def from_csv(cls, csv_row: dict[str, str]) -> "RawRow_HeaderV2": 

        return RawRow_HeaderV2( 

            date = csv_row[’Date of Travel (YYYY-MM-DD)’], 

            time = csv_row[’Arrival Time (HH:MM:SS)’], 

            lat = csv_row[’Latitude (degrees N)’], 

            lon = csv_row[’Logitude (degrees W)’],
```

`RawRow_HeaderV2` 类的实例是通过表达式 `RawRow_HeaderV2.from_csv(row)` 构建的。这些对象与 `RawRow` 类兼容。这两个类中的任何一个对象也可以转换成 `Waypoint` 实例。

对于与各种数据源一起工作的应用程序，这类“原始数据转换”dataclasses 可以方便地将逻辑布局中的细微变化映射到一致的内部结构，以便进一步处理。随着输入转换类的数量增加，需要额外的类型提示。例如，以下类型提示为输入格式的变化提供了一个通用名称：

```py
Raw: TypeAlias = RawRow | RawRow_HeaderV2
```

这种类型提示有助于统一原始的 RawRow 和替代的 RawRow_HeaderV2 类型，它们是具有兼容功能的替代定义。最重要的功能是使用生成器逐行处理数据，以避免创建包含所有数据的庞大列表对象。

## 11.4.5 参考信息

+   本章前面的使用 CSV 模块读取分隔符文件配方也涵盖了 CSV 文件读取。

+   在第六章的使用 dataclasses 处理可变对象配方中，也介绍了使用 Python 的 dataclasses 的方法。

# 11.5 使用正则表达式读取复杂格式

许多文件格式缺乏 CSV 文件那种优雅的规律性。一个相当难以解析的常见文件格式是 Web 服务器日志文件。这些文件往往具有复杂的数据，没有单一的、统一的分隔符字符或一致的引号规则。

当我们在第九章的使用 yield 语句编写生成器函数配方中查看简化的日志文件时，我们看到行如下所示：

```py
[2016-05-08 11:08:18,651] INFO in ch09_r09: Sample Message One 

[2016-05-08 11:08:18,651] DEBUG in ch09_r09: Debugging 

[2016-05-08 11:08:18,652] WARNING in ch09_r09: Something might have gone wrong
```

在这个文件中使用了各种标点符号。csv 模块无法解析这种复杂性。

我们希望编写出像 CSV 处理一样优雅简单的程序。这意味着我们需要封装日志文件解析的复杂性，并将这一方面与分析和汇总处理分开。

## 11.5.1 准备工作

解析具有复杂结构的文件通常涉及编写一个函数，该函数的行为类似于 csv 模块中的 reader()函数。在某些情况下，创建一个类似于 DictReader 类的简单类可能更容易。

读取复杂文件的核心功能是将一行文本转换成字典或单个字段值的元组。这项工作的部分通常可以通过 re 包来完成。

在我们开始之前，我们需要开发（并调试）一个正则表达式，以便正确解析输入文件的每一行。有关此信息，请参阅第一章中的使用正则表达式进行字符串解析配方。

对于这个例子，我们将使用以下代码。我们将定义一个模式字符串，其中包含一系列用于行中各种元素的正则表达式：

```py
import re 

pattern_text = ( 

    r"\[(?P<date>.*?)]\s+" 

    r"(?P<level>\w+)\s+" 

    r"in\s+(?P<module>\S+?)" 

    r":\s+(?P<message>.+)" 

    ) 

pattern = re.compile(pattern_text, re.X)
```

我们使用了 re.X 选项，这样我们可以在正则表达式中包含额外的空白。这可以通过分隔前缀和后缀字符来帮助使其更易于阅读。

当我们编写正则表达式时，我们将要捕获的有兴趣的子字符串用 () 括起来。在执行 match() 或 search() 操作后，生成的 Match 对象将包含匹配的子字符串的捕获文本。Match 对象的 groups() 方法和 Match 对象的 groupdict() 方法将提供捕获的字符串。

这是此模式的工作方式：

```py
>>> sample_data = ’[2016-05-08 11:08:18,651] INFO in ch10_r09: Sample Message One’ 

>>> match = pattern.match(sample_data) 

>>> match.groups() 

(’2016-05-08 11:08:18,651’, ’INFO’, ’ch10_r09’, ’Sample Message One’) 

>>> match.groupdict() 

{’date’: ’2016-05-08 11:08:18,651’, ’level’: ’INFO’, ’module’: ’ch10_r09’, ’message’: ’Sample Message One’}
```

我们在 sample_data 变量中提供了一行样本数据。生成的 Match 对象有一个 groups() 方法，它返回每个有趣的字段。match 对象的 groupdict() 方法的值是一个字典，其中包含在正则表达式中括号内的 ?P<name> 前缀提供的名称。

## 11.5.2 如何做到...

此配方分为两个小配方。第一部分定义了一个 log_parser() 函数来解析单行，而第二部分则将 log_parser() 函数应用于输入的每一行。

### 定义解析函数

执行以下步骤以定义 log_parser() 函数：

1.  定义编译后的正则表达式对象：

    ```py
    import re 

    pattern_text = ( 

        r"\[(?P<date>.*?)]\s+" 

        r"(?P<level>\w+)\s+" 

        r"in\s+(?P<module>\S+?)" 

        r":\s+(?P<message>.+)" 

        ) 

    pattern = re.compile(pattern_text, re.X) 
    ```

1.  定义一个类来模拟生成的复杂数据对象。这可以具有额外的派生属性或其他复杂计算。最小化地，NamedTuple 必须定义解析器提取的字段。字段名称应与正则表达式捕获名称在 (?P<name>...) 前缀中匹配：

    ```py
    from typing import NamedTuple 

    class LogLine(NamedTuple): 

        date: str 

        level: str 

        module: str 

        message: str
    ```

1.  定义一个接受一行文本作为参数并生成解析后的 LogLine 实例的函数：

    ```py
    def log_parser(source_line: str) -> LogLine:
    ```

1.  将正则表达式应用于创建一个匹配对象。我们将其分配给 match 变量，并检查它是否不为 None：

    ```py
        if match := pattern.match(source_line):
    ```

1.  当 match 的值为非 None 时，返回一个包含此输入行各种数据的有用数据结构：

    ```py
            data = match.groupdict() 

            return LogLine(**data)
    ```

1.  当匹配为 None 时，记录问题或引发异常以停止处理：

    ```py
        raise ValueError(f"Unexpected input {source_line=}")
    ```

### 使用 log_parser() 函数

此部分的配方将应用 log_parser() 函数到输入文件的每一行：

1.  从 pathlib 模块中导入有用的类和函数定义：

    ```py
    >>> from pathlib import Path 

    >>> from pprint import pprint
    ```

1.  创建标识文件的 Path 对象：

    ```py
    >>> data_path = Path("data") / "sample.log"
    ```

1.  使用 Path 对象以 with 语句打开文件。从打开的文件对象 data_file 创建日志文件读取器。在这种情况下，我们将使用内置的 map() 函数将 log_parser() 函数应用于源文件的每一行：

    ```py
    >>> with data_path.open() as data_file: 

    ...     data_reader = map(log_parser, data_file)
    ```

1.  读取（并处理）各种数据行。对于此示例，我们将打印每一行：

    ```py
    ...     for row in data_reader: 

    ...         pprint(row)
    ```

输出是一系列看起来如下所示的 LogLine 元组：

```py
LogLine(date=’2016-06-15 17:57:54,715’, level=’INFO’, module=’ch09_r10’, message=’Sample Message One’) 

LogLine(date=’2016-06-15 17:57:54,715’, level=’DEBUG’, module=’ch09_r10’, message=’Debugging’) 

LogLine(date=’2016-06-15 17:57:54,715’, level=’WARNING’, module=’ch09_r10’, message=’Something might have gone wrong’)
```

我们可以对这些元组实例进行比原始文本行更有意义的处理。这允许我们通过严重程度级别过滤数据，或根据提供消息的模块创建计数器。

## 11.5.3 它是如何工作的...

此日志文件处于第一范式（1NF）：数据组织成代表独立实体或事件的行。每一行都有一致的属性或列数，每一列都有原子数据或无法进一步有意义的分解。然而，与 CSV 文件不同的是，这种特定格式需要复杂的正则表达式来解析。

在我们的日志文件示例中，时间戳包含多个单独的元素——年、月、日、小时、分钟、秒和毫秒——但进一步分解时间戳的价值不大。将其用作单个日期时间对象并从中提取详细信息（如一天中的小时）比将单个字段组装成新的复合数据更有帮助。

在复杂的日志处理应用程序中，可能有几种不同类型的消息字段。可能需要使用不同的模式来解析这些消息类型。当我们需要这样做时，我们发现日志中的各种行在格式和属性数量方面并不一致，这违反了 1NF 假设之一。

我们通常遵循使用 CSV 模块读取定界文件配方中的设计模式，这样读取复杂的日志文件几乎与读取简单的 CSV 文件相同。实际上，我们可以看到主要区别在于一行代码：

[firstline=93,lastline=93,gobble=8][python]src/ch11/recipe˙05.py

与以下内容比较：

[firstline=95,lastline=95,gobble=8][python]src/ch11/recipe˙05.py

这种并行结构允许我们在许多输入文件格式之间重用分析函数。这使我们能够创建一个库，该库可以用于多个数据源。它可以帮助使分析应用程序在数据源更改时具有弹性。

## 11.5.4 更多...

在读取非常复杂的文件时，最常见的操作之一是将它们重写为更容易处理的格式。我们通常会想将数据保存为 CSV 格式以供以后处理。

其中一些与第七章中的使用多个资源管理多个上下文配方类似。这个配方展示了多个打开的文件处理上下文。我们将从一个文件中读取并写入到另一个文件中。

文件写入过程如下：

```py
import csv 

def copy(data_path: Path) -> None: 

    target_path = data_path.with_suffix(".csv") 

    with target_path.open("w", newline="") as target_file: 

        writer = csv.DictWriter(target_file, LogLine._fields) 

        writer.writeheader() 

        with data_path.open() as data_file: 

            reader = map(log_parser, data_file) 

            writer.writerows(row._asdict() for row in reader)
```

该脚本的第一个部分定义了一个用于目标文件的 CSV 写入器。输出文件的路径 target_path 基于输入名称 data_path。后缀被更改为.csv。

使用 newline=’’选项关闭了换行符，以打开目标文件。这允许 csv.DictWriter 类插入适合所需 CSV 方言的换行符。

创建一个 DictWriter 对象以写入指定的文件。列标题的序列由 LogLines 类定义提供。这确保输出 CSV 文件将包含正确、一致的列名。

writeheader() 方法将列名写入输出文件的第一行。这使得读取文件稍微容易一些，因为提供了列名。CSV 文件的第一行可以包含显式的模式定义，显示存在哪些数据。

源文件已按前一个菜谱所示打开。由于 csv 模块编写器的工作方式，我们可以将读取生成器表达式提供给 writer 的 writerows() 方法。writerows() 方法将消耗读取生成器产生的所有数据。这将反过来消耗由打开的文件产生的所有行。

我们不需要编写任何显式的 for 语句来确保处理所有输入行。writerows() 函数为我们保证了这一点。

输出文件如下所示：

```py
date,level,module,message 

"2016-06-15 17:57:54,715",INFO,ch09_r10,Sample Message One 

"2016-06-15 17:57:54,715",DEBUG,ch09_r10,Debugging 

"2016-06-15 17:57:54,715",WARNING,ch09_r10,Something might have gone wrong
```

该文件已从相对复杂的输入格式转换为更简单的 CSV 格式，适合进一步分析和处理。

## 11.5.5 参考内容

+   有关 with 语句的更多信息，请参阅第七章的 创建上下文和上下文管理器 菜单。

+   第九章的 使用 yield 语句编写生成器函数 菜谱展示了这种日志格式的其他处理方式。

+   在本章前面的 使用 CSV 模块读取分隔文件 菜谱中，我们探讨了这种通用设计模式的其它应用。

+   在本章前面的 使用数据类简化 CSV 文件处理 菜谱中，我们探讨了其他复杂的 CSV 处理技术。

# 11.6 读取 JSON 和 YAML 文档

JavaScript 对象表示法 (JSON) 通常用于序列化数据。有关详细信息，请参阅 [`json.org`](http://json.org)。Python 包含 json 模块，以便使用这种表示法序列化和反序列化数据。

JSON 文档被广泛应用于网络应用程序。在 RESTful 网络客户端和服务器之间使用 JSON 表示法的文档交换数据是很常见的。这两个应用堆栈的层级通过通过 HTTP 协议发送的 JSON 文档进行通信。

YAML 格式是 JSON 表示法的更复杂和灵活的扩展。有关详细信息，请参阅 [`yaml.org`](https://yaml.org)。任何 JSON 文档都是有效的 YAML 文档。反之则不然：YAML 语法更复杂，包括一些在 JSON 中无效的构造。

要使用 YAML，必须安装一个额外的模块：

```py
(cookbook3) % python -m pip install pyyaml
```

PyYAML 项目提供了一个流行且功能良好的 yaml 模块。请参阅 [`pypi.org/project/PyYAML/`](https://pypi.org/project/PyYAML/).

在本菜谱中，我们将使用 json 模块来解析 Python 中的 JSON 格式数据。

## 11.6.1 准备工作

我们在 race_result.json 文件中收集了一些帆船比赛结果。这个文件包含了关于队伍、比赛段以及各个队伍完成每个单独比赛段顺序的信息。JSON 优雅地处理了这些复杂的数据。

总分可以通过计算每个比赛段的完成位置来得出：得分最低的是总冠军。在某些情况下，当一艘船未参赛、未完成比赛或被取消比赛资格时，会有 null 值。

在计算队伍的总分时，null 值被分配一个比比赛船只数量多一个的分数。如果有七艘船，那么队伍因未能完成比赛而得到八分，这是一个相当大的惩罚。

数据具有以下架构。在整个文档中有两个字段：

+   legs：一个字符串数组，显示起点和终点。

+   teams：一个包含每个队伍详细信息的对象数组。在每个 teams 对象中，有几个数据字段：

    +   name：字符串形式的队伍名称。

    +   position：一个包含整数和 nulls 的数组，表示位置。这个数组中项目的顺序与 legs 数组中项目的顺序相匹配。

数据看起来如下：

```py
{ 

  "teams": [ 

    { 

      "name": "Abu Dhabi Ocean Racing", 

      "position": [ 

        1, 

        3, 

        2, 

        2, 

        1, 

        2, 

        5, 

        3, 

        5 

      ] 

    }, 

... 

  ], 

  "legs": [ 

    "ALICANTE - CAPE TOWN", 

    "CAPE TOWN - ABU DHABI", 

    "ABU DHABI - SANYA", 

    "SANYA - AUCKLAND", 

    "AUCKLAND - ITAJA\u00cd", 

    "ITAJA\u00cd - NEWPORT", 

    "NEWPORT - LISBON", 

    "LISBON - LORIENT", 

    "LORIENT - GOTHENBURG" 

  ] 

}
```

我们只展示了第一支队伍的详细信息。在这场特定的比赛中，总共有七支队伍。每个队伍由一个 Python 字典表示，其中包含队伍的名称和他们在每个比赛段的完成位置历史。对于这里展示的队伍，阿布扎比海洋赛车队，他们在第一段比赛中获得第一名，然后在下一段比赛中获得第三名。他们的最差表现是在第七段和第九段比赛中获得第五名，这两段比赛是从美国罗德岛纽波特到葡萄牙里斯本，以及从法国洛里昂到瑞典哥德堡。

JSON 格式的数据看起来像包含列表的 Python 字典。这种 Python 语法和 JSON 语法的重叠可以被视为一种愉快的巧合：它使得从 JSON 源文档构建的 Python 数据结构更容易可视化。

JSON 有一组小的数据结构：null、布尔值、数字、字符串、列表和对象。这些直接映射到 Python 类型中的对象。json 模块为我们将这些源文本转换为 Python 对象。

其中一个字符串包含一个 Unicode 转义序列，\u00cd，而不是实际的 Unicode 字符Í。这是一种常见的用于编码超出 128 个 ASCII 字符的字符的技术。json 模块中的解析器为我们处理了这一点。

在这个例子中，我们将编写一个函数来解开这个文档，并显示每个比赛段的队伍完成情况。

## 11.6.2 如何实现...

这个食谱将首先导入必要的模块。然后我们将使用这些模块将文件内容转换为有用的 Python 对象：

1.  我们需要 json 模块来解析文本。我们还需要一个 Path 对象来引用文件：

    ```py
    import json 

    from pathlib import Path
    ```

1.  定义一个 race_summary() 函数来从给定的 Path 实例读取 JSON 文档：

    ```py
    def race_summary(source_path: Path) -> None: 
    ```

1.  通过解析 JSON 文档创建一个 Python 对象。通常，使用 source_path.read_text() 读取由 Path 对象命名的文件是最简单的。我们将此字符串提供给 json.loads() 函数进行解析。对于非常大的文件，可以将打开的文件传递给 json.load() 函数：

    ```py
        document = json.loads(source_path.read_text())
    ```

1.  显示数据：文档对象包含一个包含两个键的字典，teams 和 legs。以下是遍历每个 legs 的方法，显示队伍在 legs 中的位置：

    ```py
        for n, leg in enumerate(document[’legs’]): 

            print(leg) 

            for team_finishes in document[’teams’]: 

                print( 

                    team_finishes[’name’], 

                    team_finishes[’position’][n])
    ```

每个队伍的数据将是一个包含两个键的字典：name 和 position。我们可以深入到队伍的详细信息中，以获取第一个队伍的名称：

```py
>>> document[’teams’][6][’name’] 

’Team Vestas Wind’
```

我们可以查看 legs 字段，以查看每个赛段的名称：

```py
>>> document[’legs’][5] 

’ITAJA - NEWPORT’
```

## 11.6.3 它是如何工作的...

JSON 文档是 JavaScript 对象表示法中的数据结构。JavaScript 程序可以轻松解析文档。其他语言必须做更多工作来将 JSON 转换为本地数据结构。

JSON 文档包含三种结构：

+   映射到 Python 字典的对象：JSON 的语法与 Python 类似：{"key": "value", ...}。

+   映射到 Python 列表的数组：JSON 语法使用 [item, ...]，这也与 Python 类似。

+   原始值：有五种值类别：字符串、数字、true、false 和 null。字符串用 " 和 " 包围，并使用各种 \ 转义序列，这与 Python 的类似。数字遵循浮点值规则。其他三个值是简单的字面量；这些与 Python 的 True、False 和 None 字面量平行。

    作为特殊情况，没有小数点的数字成为 Python int 对象。这是 JSON 标准的扩展。

没有提供其他数据类型的支持。这意味着 Python 程序必须将复杂的 Python 对象转换为更简单的表示，以便它们可以用 JSON 语法进行序列化。

相反，我们经常应用额外的转换来从简化的 JSON 表示中重建复杂的 Python 对象。json 模块有一些地方可以应用额外的处理来创建更复杂的 Python 对象。

## 11.6.4 更多...

通常，一个文件包含一个单一的 JSON 文档。JSON 标准没有提供一种简单的方法来将多个文档编码到单个文件中。如果我们想分析一个网络日志，例如，原始的 JSON 标准可能不是保存大量信息的最佳表示法。

有一些常见扩展，如换行符分隔的 JSON（[`ndjson.org`](http://ndjson.org)）和 JSON Lines，[`jsonlines.org`](http://jsonlines.org)，用于定义将多个 JSON 文档编码到单个文件中的方式。

当这些方法处理文档集合时，还有一个额外的问题我们需要解决：序列化（和反序列化）复杂对象，例如 datetime 对象。

当我们将 Python 对象的状态表示为文本字符的字符串时，我们已经序列化了对象的状态。许多 Python 对象需要被保存到文件或传输到另一个进程。这类传输需要对象状态的表示。我们将分别查看序列化和反序列化。

### 序列化复杂的数据结构

如果我们创建的 Python 对象仅限于内置类型 dict、list、str、int、float、bool 和 None 的值，那么序列化到 JSON 的效果最好。这个 Python 类型子集可以用来构建 json 模块可以序列化的对象，并且可以由多种不同语言编写的许多程序广泛使用。

一种常用的、不易序列化的数据结构是 datetime.datetime 对象。

避免在尝试序列化不寻常的 Python 对象时引发 TypeError 异常，可以通过两种方式之一实现。我们可以在构建文档之前将数据转换为 JSON 友好的结构，或者我们可以在 JSON 序列化过程中添加一个默认类型处理器，这样我们就可以提供一个可序列化的数据版本。

在将 datetime 对象序列化为 JSON 之前转换为字符串，我们需要对底层数据进行更改。由于序列化问题而篡改数据或 Python 的数据类型似乎有些尴尬。

序列化复杂数据的另一种技术是提供一个在序列化过程中由 json 模块使用的函数。此函数必须将复杂对象转换为可以安全序列化的东西。在下面的示例中，我们将 datetime 对象转换为简单的字符串值：

```py
def default_date(object: Any) -> Any: 

    match object: 

        case datetime.datetime(): 

            return {"$date$": object.isoformat()} 

    return object
```

我们定义了一个函数，default_date()，它将对 datetime 对象应用特殊的转换规则。任何 datetime.datetime 实例都将被替换为一个具有明显键 – "$date$" – 和字符串值的字典。这个字典可以通过 json 模块进行序列化。

我们将这个序列化辅助函数提供给 json.dumps() 函数。这是通过将 default_date() 函数分配给默认参数来完成的，如下所示：

```py
>>> example_date = datetime.datetime(2014, 6, 7, 8, 9, 10) 

>>> document = {’date’: example_date} 

>>> print( 

...     json.dumps(document, default=default_date, indent=2) 

... ) 

{ 

  "date": { 

    "$date$": "2014-06-07T08:09:10" 

  } 

}
```

当 json 模块无法序列化一个对象时，它将对象传递给给定的默认函数 default_date()。在任何给定的应用程序中，我们需要扩展这个函数以处理我们可能想要在 JSON 表示法中序列化的多种 Python 对象类型。如果没有提供默认函数，当对象无法序列化时将引发异常。

### 反序列化复杂的数据结构

当反序列化 JSON 以创建 Python 对象时，有一个钩子可以用来将数据从 JSON 字典转换为更复杂的 Python 对象。这被称为 object_hook，它在 json.loads() 函数的处理过程中被使用。此钩子用于检查每个 JSON 字典，以查看是否应该从字典实例创建其他内容。

我们提供的函数将创建一个更复杂的 Python 对象，或者简单地返回未修改的原始字典对象：

```py
def as_date(object: dict[str, Any]) -> Any: 

    if {’$date$’} == set(object.keys()): 

        return datetime.datetime.fromisoformat(object[’$date$’]) 

    return object
```

此函数将检查每个解码的对象，以查看对象是否只有一个字段，并且该单个字段是否命名为"$date$"。如果是这种情况，整个对象的价值将替换为 datetime.datetime 对象。返回类型是 Any 和 dict[str, Any]的联合，以反映两种可能的结果：要么是某个对象，要么是原始字典。

我们通过 json.loads()函数使用 object_hook 参数提供了一个函数，如下所示：

```py
>>> source = ’’’{"date": {"$date$": "2014-06-07T08:09:10"}}’’’ 

>>> json.loads(source, object_hook=as_date) 

{’date’: datetime.datetime(2014, 6, 7, 8, 9, 10)}
```

这解析了一个非常小的 JSON 文档。所有对象都提供给 as_date()对象钩子。在这些对象中，有一个字典符合包含日期的标准。从 JSON 序列化中找到的字符串值构建了一个 Python 对象。

Pydantic 包提供了一系列序列化功能。关于如何使用此包的配方在第十章中有展示。

## 11.6.5 参见

+   本章后面的读取 HTML 文档配方将展示我们如何从 HTML 源准备这些数据。

+   第十章中的使用 Pydantic 实现更严格的类型检查配方涵盖了 Pydantic 包的一些功能。

# 11.7 读取 XML 文档

XML 标记语言广泛用于以序列化形式表示对象的状态。有关详细信息，请参阅[`www.w3.org/TR/REC-xml/`](http://www.w3.org/TR/REC-xml/)。Python 包含多个用于解析 XML 文档的库。

XML 被称为标记语言，因为感兴趣的内容被标记为标签，使用起始<tag>和结束</tag>来书写，用于定义数据结构。整个文件文本包括内容和 XML 标记。

由于标记与文本交织在一起，因此必须使用一些额外的语法规则来区分标记和文本。文档必须使用&lt;代替<，&gt;代替>，以及&amp;代替&在文本中。此外，&quot;也用于在属性值中嵌入一个"字符。在大多数情况下，XML 解析器在消费 XML 时会处理这种转换。

因此，示例文档将包含以下项目：

```py
<team><name>Team SCA</name><position>...</position></team>
```

<team>标签包含<name>标签，其中包含团队的名称文本。《position>标签包含关于团队在每个赛段完成位置的数据。

整个文档形成一个大型、嵌套的容器集合。我们可以将文档视为一个树，其根标签包含所有其他标签及其嵌入的内容。在标签之间，可以有额外的内容。在某些应用中，标签结束之间的额外内容完全是空白。

下面是我们将要查看的文档的开始部分：

```py
<?xml version="1.0"?> 

<results> 

    <teams> 

        <team> 

            <name> 

                Abu Dhabi Ocean Racing 

            </name> 

            <position> 

                <leg n="1"> 

                    1 

                </leg> 

               ... 

            </position> 

            ... 

        </team> 

        ... 

   </teams> 

    <legs> 

        <leg n="1"> 

            ALICANTE - CAPE TOWN 

        </leg> 

        ... 

    </legs> 

</results>
```

最高层容器是<results>标签。在这个容器内有一个<teams>标签。在<teams>标签内，每个团队的数据都以<team>标签的形式重复出现。我们使用...来表示文档中省略的部分。

使用正则表达式解析 XML 非常困难。正则表达式不擅长处理 XML 中存在的递归和重复类型。我们需要更复杂的解析器来处理嵌套标签的语法。

有两个二进制库，分别是 xml.sax 和 xml.parsers.expat 模块的一部分，用于解析 XML。这些库的优点是速度非常快。

此外，xml.etree 包中还有一个非常复杂的工具集。我们将专注于使用此包中的 ElementTree 类来解析和分析 XML 文档。这具有提供大量有用功能的优势，如 XPath 搜索，以在复杂文档中查找标签。

## 11.7.1 准备工作

我们已经收集了一些帆船赛成绩，保存在 race_result.xml 文件中。该文件包含有关团队、赛段以及各个团队完成每个赛段顺序的信息。有关此数据的更多信息，请参阅本章中的阅读 JSON 和 YAML 文档配方。

此数据的根标签是一个<results>文档。它具有以下模式：

+   <legs>标签包含单个<leg>标签，命名每条赛道的名称。每个<leg>标签将包含一个起始港和一个结束港的文本。

+   <teams>标签包含多个<team>标签，包含每个团队的详细信息。每个团队的数据都使用内部标签进行结构化：

    +   <name>标签包含团队名称。

    +   <position>标签包含多个<leg>标签，表示给定腿的完成位置。每个腿都有编号，编号与<legs>标签中的腿定义相匹配。

在 XML 表示法中，应用程序数据出现在两种地方。第一种是在起始标签和结束标签之间——例如，<name>阿布扎比海洋赛车</name>，有文本“阿布扎比海洋赛车”，以及<name>和</name>标签。

此外，数据还将作为标签的属性出现；例如，在<leg n="1">中。标签是<leg>，具有属性 n，其值为"1"。一个标签可以有无限数量的属性。

<leg>标签指出了 XML 的一个有趣问题。这些标签包括作为属性的腿号，而腿的位置由标签内的文本给出。没有真正的模式或偏好来指定有用数据的位置。理想情况下，它总是在标签之间，但这通常不是真的。

XML 允许混合内容模型。这反映了 XML 与文本混合的情况，其中文本位于 XML 标签内外。以下是一个混合内容的示例：

```py
<p> 

This has <strong>mixed</strong> content. 

</p>
```

<p> 标签的内容是文本和标签的混合。我们在本食谱中处理的数据不依赖于这种混合内容模型，这意味着所有数据都在单个标签或标签的属性中。标签之间的空白可以忽略。

## 11.7.2 如何实现...

我们将定义一个函数，将 XML 文档转换为包含腿描述和团队结果的字典：

1.  我们需要 xml.etree 模块来解析 XML 文本。我们还需要一个 Path 对象来引用文件。我们将 ElementTree 类的较短名称分配给了 XML：

    ```py
    import xml.etree.ElementTree as XML 

    from pathlib import Path 

    from typing import cast
    ```

    cast() 函数是必需的，用于强制工具如 mypy 将结果视为给定类型。这使得我们可以忽略 None 结果的可能性。

1.  定义一个函数来从给定的 Path 实例读取 XML 文档：

    ```py
    def race_summary(source_path: Path) -> None:
    ```

1.  通过解析 XML 文本创建一个 Python ElementTree 对象。通常，使用 source_path.read_text() 读取由路径指定的文件是最简单的。我们提供了这个字符串给 XML.fromstring() 方法进行解析。对于非常大的文件，增量解析器有时更有帮助。以下是针对较小文件的版本：

    ```py
        source_text = source_path.read_text(encoding=’UTF-8’) 

        document = XML.fromstring(source_text)
    ```

1.  显示数据。XML 元素对象有两个用于导航 XML 结构的有用方法，即 find() 和 findall() 方法，分别用于定位标签的第一个实例和所有实例。使用这些方法，我们可以创建一个包含两个键的字典，"teams" 和 "legs"：

    ```py
        legs = cast(XML.Element, document.find(’legs’)) 

        teams = cast(XML.Element, document.find(’teams’)) 

        for leg in legs.findall(’leg’): 

            print(cast(str, leg.text).strip()) 

            n = leg.attrib[’n’] 

            for team in teams.findall(’team’): 

                position_leg = cast(XML.Element, 

                    team.find(f"position/leg[@n=’{n}’]")) 

                name = cast(XML.Element, team.find(’name’)) 

                print( 

                    cast(str, name.text).strip(), 

                    cast(str, position_leg.text).strip() 

                )
    ```

    在 <legs> 标签内，有许多单独的 <leg> 标签。每个标签都具有以下结构：

    ```py
    <leg n="1">ALICANTE - CAPE TOWN</leg>
    ```

    Python 表达式 leg.attrib[’n’] 从给定元素中提取名为 n 的属性值。表达式 leg.text.strip() 将找到 <leg> 标签内的所有文本，并去除额外的空白。

    元素的 find() 和 findall() 方法使用 XPath 语法来定位标签。我们将在本食谱的 There’s more... 部分中详细检查这些功能。

重要的是要注意，find() 函数的结果具有 XML.Element | None 的类型提示。对于 None 结果的可能性，我们有两种处理方法：

+   使用 if 语句来处理结果为 None 的情况。

+   使用 cast(XML.Element, tag.find(...)) 来声明结果永远不会是 None。如果标签缺失，引发的异常将有助于诊断源文档与我们的消费者应用程序处理期望之间的不匹配。 

对于比赛的每一腿，我们需要打印完成位置，这些位置包含在 <teams> 标签内。在这个标签内，我们需要找到具有给定腿上该团队完成位置的适当 <leg> 标签。为此，我们使用复杂的 XPath 搜索，f"position/leg[@n=’{n}’]"，根据具有特定属性值的 <leg> 标签的存在来定位特定的 <position> 标签。n 的值是腿号。对于第九腿，n=9，f-string 将是 "position/leg[@n=’9’]"。这将定位包含具有属性 n 等于 9 的 <leg> 标签的 <position> 标签。

由于 XML 支持混合内容模型，内容中的所有 \n、\t 和空格字符在解析操作中都被完美保留。我们很少希望保留这些空白字符，因此在使用 strip() 方法在有意义的内容前后删除任何多余的字符是有意义的。

## 11.7.3 它是如何工作的...

XML 解析模块将 XML 文档转换为基于标准化的文档对象模型（DOM）的相当复杂的树结构。在 xml.etree 模块的情况下，文档将由 Element 对象构建，这些对象通常代表标签和文本。

XML 还可以包含处理指令和注释。在这里，我们将忽略它们，专注于文档结构和内容。

每个元素实例都有标签的文本、标签内的文本、标签的部分属性和尾部。标签是 <tag> 内部的名称。属性是跟在标签名称后面的字段，例如，<leg n="1"> 标签有一个名为 leg 的标签名称和一个名为 n 的属性。在 XML 中，值始终是字符串；任何转换为不同数据类型都是使用该数据的应用程序的责任。

文本包含在标签的开始和结束之间。因此，一个如 `<name>Team SCA</name>` 的标签具有 "Team SCA" 作为代表 `<name>` 标签的元素的文本属性值。

注意，标签还有一个尾部属性。考虑以下两个标签的序列：

```py
<name>Team SCA</name> 

<position>...</position>
```

在 </name> 标签关闭后和 <position> 标签打开前有一个 \n 空白字符。这些额外的文本被收集到 <name> 标签的尾部属性中。当在混合内容模型中工作时，这些尾部值可能很重要。在元素内容模型中工作时，尾部值通常是空白字符。

## 11.7.4 更多内容...

由于我们不能简单地将在 XML 文档转换为 Python 字典，我们需要一种方便的方法来搜索文档的内容。ElementTree 类提供了一种搜索技术，这是 XML 路径语言（XPath）的部分实现，用于指定 XML 文档中的位置。XPath 表示法为我们提供了相当大的灵活性。

XPath 查询与 find() 和 findall() 方法一起使用。以下是如何找到所有团队名称的方法：

```py
>>> for tag in document.findall(’teams/team/name’): 

...     print(tag.text.strip()) 

Abu Dhabi Ocean Racing 

Team Brunel 

Dongfeng Race Team 

MAPFRE 

Team Alvimedica 

Team SCA 

Team Vestas Wind
```

XPath 查询查找顶级 <teams> 标签。在该标签内，我们想要 <team> 标签。在这些标签内，我们想要 <name> 标签。这将搜索所有这种嵌套标签结构的实例。

## 11.7.5 参见

+   与 XML 文档相关的安全问题有很多。有关更多信息，请参阅 OWASP [XML 安全速查表](https://cheatsheetseries.owasp.org/cheatsheets/XML_Security_Cheat_Sheet.html)。

+   [lxml](https://pypi.org/project/lxml/) 库扩展了元素树库的核心功能，提供了额外的功能。

+   本章后面的 阅读 HTML 文档 菜谱展示了我们如何从 HTML 源准备这些数据。

# 11.8 阅读 HTML 文档

网络上大量的内容都是使用 HTML 呈现的。浏览器将数据渲染得非常漂亮。我们可以编写应用程序从 HTML 页面中提取内容。

解析 HTML 涉及两个复杂因素：

+   与现代 XML 不同的古老 HTML 方言

+   可以容忍不正确 HTML 并创建正确显示的浏览器

第一个复杂因素是 HTML 和 XML 的历史。现代 HTML 是 XML 的一个特定文档类型。历史上，HTML 从自己的独特文档类型定义开始，基于较老的 SGML。这些原始 SGML/HTML 概念被修订和扩展，以创建一种新的语言，XML。在从遗留 HTML 到基于 XML 的 HTML 的过渡期间，网络服务器使用各种过渡文档类型定义提供内容。大多数现代网络服务器使用<DOCTYPE html>前缀来声明文档是正确结构的 XML 语法，使用 HTML 文档模型。一些网络服务器将在前缀中使用其他 DOCTYPE 引用，并提供不是正确 XML 的 HTML。

解析 HTML 的另一个复杂因素是浏览器的设计。浏览器有义务渲染网页，即使 HTML 结构不佳甚至完全无效。设计目标是向用户提供反映内容的东西——而不是显示错误消息，指出内容无效。

HTML 页面可能充满了问题，但在浏览器中仍然可以显示一个看起来不错的页面。

我们可以使用标准库的 html.parser 模块，但它并不像我们希望的那样有帮助。Beautiful Soup 包提供了更多有用的方法来解析 HTML 页面到有用的数据结构。这个包可以在 Python 包索引（PyPI）上找到。请参阅[`pypi.python.org/pypi/beautifulsoup4`](https://pypi.python.org/pypi/beautifulsoup4)。

这必须使用以下终端命令下载和安装：

```py
(cookbook3) % python -m pip install beautifulsoup4
```

## 11.8.1 准备工作

我们已经收集了一些历史帆船赛结果，保存在 Volvo Ocean Race.html 文件中。这个文件包含有关团队、航段以及各个团队完成每个航段的顺序的信息。它已被从沃尔沃海洋赛网站抓取，并在浏览器中打开时看起来很棒。有关此数据的更多信息，请参阅本章中的阅读 JSON 和 YAML 文档配方。

虽然 Python 的标准库有 urllib 包来获取文档，但通常使用 Requests 包来读取网页。

通常，一个 HTML 页面具有以下整体结构：

```py
<html> 

<head>...</head> 

<body>...</body> 

</html>
```

在<head>标签内，将有元数据、指向 JavaScript 库的链接以及指向层叠样式表（CSS）文档的链接。内容位于<body>标签中。

在这种情况下，比赛结果位于<body>标签内的 HTML<table>标签中。该表格具有以下结构：

```py
<table> 

   <thead> 

      ... 

   </thead> 

   <tbody> 

       ... 

   </tbody> 

</table>
```

<thead>标签定义了表格的列标题。有一个单独的行标签<TR>，其中包含表格标题标签<TH>，这些标签包含列标题。对于示例数据，每个<TH>标签看起来像这样：

```py
<th tooltipster data="<strong>ALICANTE - CAPE TOWN</strong>" data-theme="tooltipster-shadow" data-htmlcontent="true" data-position="top"> 

LEG 1</th>
```

重要的显示是每个赛段的标识符；在这个例子中是 LEG 1。这是<th>标签的文本内容。还有一个由 JavaScript 函数使用的属性值，data。这个属性值是腿的名称，当鼠标悬停在列标题上时显示。

<tbody>标签包含每个团队和比赛的成果行。每个<tr>表格行标签包含带有团队名称及其结果的<td>表格数据标签。以下是从 HTML 中典型的<tr>行：

```py
<tr class="ranking-item"> 

   <td class="ranking-position">3</td> 

   <td class="ranking-avatar"><img src="img/..."></td> 

   <td class="ranking-team"> Dongfeng Race Team</td> 

   <td class="ranking-number">2</td> 

   <td class="ranking-number">2</td> 

   <td class="ranking-number">1</td> 

   <td class="ranking-number">3</td> 

   <td class="ranking-number" tooltipster 

   data="<center><strong>RETIRED</strong><br> Click for more info</center>" data-theme="tooltipster-3" 

   data-position="bottom" data-htmlcontent="true"> 

   <a href="/en/news/8674_Dongfeng-Race-Team-breaks-mast-crew-safe.html" 

   target="_blank">8</a> 

   <div class="status-dot dot-3"></div></td> 

   ... more columns ... 

</tr>
```

<tr>标签有一个 class 属性，它定义了此行的 CSS 样式。这个 class 属性可以帮助我们的数据收集应用程序定位相关内容。

<td>标签也有 class 属性。对于这个精心设计的数据，class 属性阐明了<td>单元格的内容。并不是所有的 CSS 类名都像这些定义得那么好。

其中一个单元格——带有 tooltipster 属性——没有文本内容。相反，这个单元格有一个<a>标签和一个空的<div>标签。这个单元格还包含几个属性，包括 data 等。这些属性由 JavaScript 函数用于在单元格中显示更多信息。

这里还有一个复杂性，即 data 属性包含实际上是 HTML 内容的文本。解析这部分文本需要创建一个单独的 BeautifulSoup 解析器。

## 11.8.2 如何实现...

我们将定义一个函数，将 HTML <table>转换为包含腿描述和团队结果的字典：

1.  从 bs4 模块导入 BeautifulSoup 类以解析文本。我们还需要一个 Path 对象来引用文件：

    ```py
    from bs4 import BeautifulSoup 

    from pathlib import Path 

    from typing import Any
    ```

1.  定义一个函数，从给定的 Path 实例读取 HTML 文档：

    ```py
    def race_extract(source_path: Path) -> dict[str, Any]:
    ```

1.  从 HTML 内容创建 soup 结构。我们将将其分配给一个变量，soup。作为替代，我们也可以使用 Path.read_text()方法来读取内容：

    ```py
        with source_path.open(encoding="utf8") as source_file: 

            soup = BeautifulSoup(source_file, "html.parser")
    ```

1.  从 soup 对象中，我们需要导航到第一个<table>标签。在其内部，我们需要找到第一个<thead>和<tr>标签。通过使用标签名作为属性来导航到第一个实例的标签：

    ```py
        thead_row = soup.table.thead.tr  # type: ignore [union-attr]
    ```

    使用一个特殊的注释来抑制 mypy 警告。# type: ignore [union-attr]是必需的，因为每个标签属性的类型提示为 Tag | None。对于某些应用程序，可以使用额外的 if 语句来确认存在的标签的预期组合。

1.  我们必须从每一行的每个<th>单元格中累积标题数据：

    ```py
        legs: list[tuple[str, str | None]] = [] 

        for tag in thead_row.find_all("th"): # type: ignore [union-attr] 

            leg_description = ( 

                tag.string, tag.attrs.get("data") 

            ) 

            legs.append(leg_description)
    ```

1.  要找到表格的内容，我们导航到<table>和<tbody>标签：

    ```py
        tbody = soup.table.tbody # type: ignore [union-attr]
    ```

1.  我们需要访问所有的<tr>标签。在每一行中，我们希望将所有<td>标签的内容转换为团队名称和团队位置集合，这取决于 td 标签的属性：

    ```py
        teams: list[dict[str, Any]] = [] 

        for row in tbody.find_all("tr"): # type: ignore [union-attr] 

            team: dict[str, Any] = { 

                "name": None, 

                "position": []} 

            for col in row.find_all("td"): 

                if "ranking-team" in col.attrs.get("class"): 

                    team["name"] = col.string 

                elif ( 

                        "ranking-number" in col.attrs.get("class") 

                    ): 

                    team["position"].append(col.string) 

                elif "data" in col.attrs: 

                    # Complicated explanation with nested HTML 

                    # print(col.attrs, col.string) 

                    pass 

            teams.append(team)
    ```

1.  一旦提取了腿和团队，我们就可以创建一个有用的字典，它将包含这两个集合：

    ```py
        document = { 

            "legs": legs, 

            "teams": teams, 

        } 

        return document
    ```

我们创建了一个腿的列表，显示了每个腿的顺序和名称，并解析了表格的主体以创建一个字典-列表结构，其中包含给定团队的每个腿的结果。生成的对象看起来像这样：

```py
>>> source_path = Path("data") / "Volvo Ocean Race.html" 

>>> race_extract(source_path) 

{’legs’: [(None, None), 

          (’LEG 1’, ’<strong>ALICANTE - CAPE TOWN’), 

          (’LEG 2’, ’<strong>CAPE TOWN - ABU DHABI</strong>’), 

          (’LEG 3’, ’<strong>ABU DHABI - SANYA</strong>’), 

          (’LEG 4’, ’<strong>SANYA - AUCKLAND</strong>’), 

          (’LEG 5’, ’<strong>AUCKLAND - ITAJA</strong>’), 

          (’LEG 6’, ’<strong>ITAJA - NEWPORT</strong>’), 

          (’LEG 7’, ’<strong>NEWPORT - LISBON</strong>’), 

          (’LEG 8’, ’<strong>LISBON - LORIENT</strong>’), 

          (’LEG 9’, ’<strong>LORIENT - GOTHENBURG</strong>’), 

          (’TOTAL’, None)], 

 ’teams’: [ 

    {’name’: ’Abu Dhabi Ocean Racing’, 

     ’position’: [’1’, ’3’, 

              ’2’, ’2’, 

              ’1’, ’2’,
```

```py
 {’name’: ’Team Vestas Wind’, 

     ’position’: [’4’, 

                None, 

                None, 

                None, 

                None, 

                None, 

                None, 

                ’2’, 

                ’6’, 

                ’60’]}]}
```

在表格的主体中，许多单元格的最终比赛位置为 None，而特定<TD>标签的数据属性中有一个复杂值。解析此文本中嵌入的 HTML 遵循配方中显示的模式，使用另一个 BeautifulSoup 实例。

## 11.8.3 它是如何工作的...

BeautifulSoup 类将 HTML 文档转换为基于文档对象模型（DOM）的相当复杂对象。生成的结构将由 Tag、NavigableString 和 Comment 类的实例组成。

每个 Tag 对象都有一个名称、字符串和属性。名称是<和>字符内的单词。属性是跟随标签名称的字段。例如，<td class="ranking-number">1</td>有一个名为 td 的标签名称和一个名为 class 的属性。值通常是字符串，但在少数情况下，值可以是字符串列表。Tag 对象的字符串属性是标签内的内容；在这种情况下，它是一个非常短的字符串，1。

HTML 是一个混合内容模型。在查看给定标签的子标签时，将有一个子标签和子 NavigableText 对象的序列，这些子标签和子 NavigableText 对象可以自由混合。

BeautifulSoup 解析器类依赖于一个底层库来完成一些解析工作。使用内置的 html.parser 模块来做这个工作是最简单的。其他替代方案提供了一些优势，如更好的性能或更好地处理损坏的 HTML。

## 11.8.4 更多...

Beautiful Soup 的 Tag 对象代表文档结构的层次结构。在标签之间有几种导航方式。在这个配方中，我们依赖于 soup.html 与 soup.find("html")相同的方式。我们还可以通过属性值进行搜索，包括类和 id。这些通常提供了关于内容的意义信息。

在某些情况下，一个文档将有一个精心设计的组织，通过 id 属性或 class 属性进行搜索将找到相关数据。以下是一个使用 HTML 类属性进行给定结构典型搜索的例子：

```py
>>> ranking_table = soup.find(’table’, class_="ranking-list")
```

注意，我们必须在我们的 Python 查询中使用 class_ 来搜索名为 class 的属性。class 是一个 Python 的保留字，不能用作参数名称。考虑到整个文档，我们正在搜索任何<table class="ranking-list">标签。这将找到网页中的第一个此类标签。由于我们知道只有一个这样的标签，这种基于属性的搜索有助于区分我们试图找到的内容和网页上的任何其他表格数据。

## 11.8.5 参见

+   [Requests](https://pypi.org/project/requests/)包可以极大地简化与复杂网站交互所需的代码。

+   查看关于 robots.txt 文件和 [RFC 9309 机器人排除协议](https://www.rfc-editor.org/rfc/rfc9309.html#name-informative-references) 的信息，请访问 [`www.robotstxt.org`](https://www.robotstxt.org) 网站。

+   阅读 JSON 和 YAML 文档 和 阅读 XML 文档 的配方，如本章前面所示，都使用了类似的数据。示例数据是通过使用这些技术从原始 HTML 页面抓取创建的。

# 加入我们的社区 Discord 空间

加入我们的 Python Discord 工作空间，讨论并了解更多关于本书的信息：[`packt.link/dHrHU`](https://packt.link/dHrHU)

![图片](img/file1.png)
