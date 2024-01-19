# 基本脚本和文件信息配方

本章涵盖了以下配方：

+   像成年人一样处理参数

+   迭代松散文件

+   记录文件属性

+   复制文件、属性和时间戳

+   对文件和数据流进行哈希处理

+   使用进度条跟踪

+   记录结果

+   多人合作，事半功倍

# 介绍

数字取证涉及识别和分析数字媒体，以协助法律、商业和其他类型的调查。我们分析的结果往往对调查的方向产生重大影响。鉴于“摩尔定律”或多或少成立，我们预期要审查的数据量正在稳步增长。因此，可以断定，调查人员必须依赖某种程度的自动化来有效地审查证据。自动化，就像理论一样，必须经过彻底的审查和验证，以免导致错误的结论。不幸的是，调查人员可能使用工具来自动化某些过程，但并不完全了解工具、潜在的取证物件或输出的重要性。这就是 Python 发挥作用的地方。

在《Python 数字取证食谱》中，我们开发和详细介绍了一些典型场景的示例。目的不仅是演示 Python 语言的特性和库，还要说明它的一个巨大优势：即对物件的基本理解。没有这种理解，就不可能首先开发代码，因此迫使您更深入地理解物件。再加上 Python 的相对简单和自动化的明显优势，很容易理解为什么这种语言被社区如此迅速地接受。

确保调查人员理解我们脚本的产品的一种方法是提供有意义的文档和代码解释。这就是本书的目的。本书中演示的示例展示了如何配置参数解析，这既易于开发，又简单易懂。为了增加脚本的文档，我们将介绍有效记录脚本执行过程和遇到的任何错误的技术。

数字取证脚本的另一个独特特性是与文件及其相关元数据的交互。取证脚本和应用程序需要准确地检索和保留文件属性，包括日期、权限和文件哈希。本章将介绍提取和呈现这些数据给审查员的方法。

与操作系统和附加卷上找到的文件进行交互是数字取证中设计的任何脚本的核心。在分析过程中，我们需要访问和解析具有各种结构和格式的文件。因此，准确和正确地处理和与文件交互非常重要。本章介绍的示例涵盖了本书中将继续使用的常见库和技术：

+   解析命令行参数

+   递归迭代文件和文件夹

+   记录和保留文件和文件夹的元数据

+   生成文件和其他内容的哈希值

+   用进度条监视代码

+   记录配方执行信息和错误

+   通过多进程改善性能

访问[www.packtpub.com/books/content/support](http://www.packtpub.com/books/content/support)下载本章的代码包。

# 像成年人一样处理参数

配方难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

A 人：我来这里是为了进行一场好的争论！

B 人：啊，不，你没有，你来这里是为了争论！

A 人：一个论点不仅仅是矛盾。

B 人：好吧！可能吧！

A 人：不，不行！一个论点是一系列相关的陈述

旨在建立一个命题。

B 人：不，不是！

A 人：是的，是的！不仅仅是矛盾。

除了蒙提·派森([`www.montypython.net/scripts/argument.php`](http://www.montypython.net/scripts/argument.php))之外，参数是任何脚本的一个组成部分。参数允许我们为用户提供一个接口，以指定改变代码行为的选项和配置。有效地使用参数，不仅仅是矛盾，可以使工具更加灵活，并成为审查人员喜爱的工具。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。虽然还有其他可用的参数处理库，例如`optparse`和`ConfigParser`，但我们的脚本将利用`argparse`作为我们的事实命令行处理程序。虽然`optparse`是以前版本的 Python 中使用的库，但`argparse`已成为创建参数处理代码的替代品。`ConfigParser`库从配置文件中解析参数，而不是从命令行中解析。这对于需要大量参数或有大量选项的代码非常有用。在本书中，我们不会涵盖`ConfigParser`，但如果发现您的`argparse`配置变得难以维护，值得探索一下。

要了解有关`argparse`库的更多信息，请访问[`docs.python.org/3/library/argparse.html`](https://docs.python.org/3/library/argparse.html)。

# 如何做…

在此脚本中，我们执行以下步骤：

1.  创建位置参数和可选参数。

1.  向参数添加描述。

1.  使用选择选项配置参数。

# 工作原理…

首先，我们导入`print_function`和`argparse`模块。通过从`__future__`库导入`print_function`，我们可以像在 Python 3.X 中编写打印语句一样编写它们，但仍然在 Python 2.X 中运行它们。这使我们能够使配方与 Python 2.X 和 3.X 兼容。在可能的情况下，我们在本书中的大多数配方中都这样做。

在创建有关配方的一些描述性变量之后，我们初始化了我们的`ArgumentParser`实例。在构造函数中，我们定义了`description`和`epilog`关键字参数。当用户指定`-h`参数时，这些数据将显示，并且可以为用户提供有关正在运行的脚本的额外上下文。`argparse`库非常灵活，如果需要，可以扩展其复杂性。在本书中，我们涵盖了该库的许多不同特性，这些特性在其文档页面上有详细说明：

```py
from __future__ import print_function
import argparse

__authors__ = ["Chapin Bryce", "Preston Miller"]
__date__ = 20170815
__description__ = 'A simple argparse example'

parser = argparse.ArgumentParser(
    description=__description__,
    epilog="Developed by {} on {}".format(
        ", ".join(__authors__), __date__)
)
```

创建了解析器实例后，我们现在可以开始向我们的命令行处理程序添加参数。有两种类型的参数：位置参数和可选参数。位置参数以字母开头，与可选参数不同，可选参数以破折号开头，并且需要执行脚本。可选参数以单个或双破折号字符开头，不是位置参数（即，顺序无关紧要）。如果需要，可以手动指定这些特性以覆盖我们描述的默认行为。以下代码块说明了如何创建两个位置参数：

```py
# Add Positional Arguments
parser.add_argument("INPUT_FILE", help="Path to input file")
parser.add_argument("OUTPUT_FILE", help="Path to output file")
```

除了更改参数是否必需，我们还可以指定帮助信息，创建默认值和其他操作。`help`参数有助于传达用户应提供的内容。其他重要参数包括`default`、`type`、`choices`和`action`。`default`参数允许我们设置默认值，而`type`将输入的类型（默认为字符串）转换为指定的 Python 对象类型。`choices`参数使用定义的列表、字典或集合来创建用户可以选择的有效选项。

`action`参数指定应用于给定参数的操作类型。一些常见的操作包括`store`，这是默认操作，用于存储与参数关联的传递值；`store_true`，将`True`分配给参数；以及`version`，打印由版本参数指定的代码版本：

```py
# Optional Arguments
parser.add_argument("--hash", help="Hash the files", action="store_true")

parser.add_argument("--hash-algorithm",
                    help="Hash algorithm to use. ie md5, sha1, sha256",
                    choices=['md5', 'sha1', 'sha256'], default="sha256"
                    )

parser.add_argument("-v", "--version", "--script-version",
                    help="Displays script version information",
                    action="version", version=str(__date__)
                    )

parser.add_argument('-l', '--log', help="Path to log file", required=True)
```

当我们定义和配置了我们的参数后，我们现在可以解析它们并在我们的代码中使用提供的输入。以下片段显示了我们如何访问这些值并测试用户是否指定了可选参数。请注意我们如何通过我们分配的名称来引用参数。如果我们指定了短和长的参数名，我们必须使用长名：

```py
# Parsing and using the arguments
args = parser.parse_args()

input_file = args.INPUT_FILE
output_file = args.OUTPUT_FILE

if args.hash:
    ha = args.hash_algorithm
    print("File hashing enabled with {} algorithm".format(ha))
if not args.log:
    print("Log file not defined. Will write to stdout")
```

当组合成一个脚本并在命令行中使用`-h`参数执行时，上述代码将提供以下输出：

![](img/00005.jpeg)

如此所示，`-h`标志显示了脚本帮助信息，由`argparse`自动生成，以及`--hash-algorithm`参数的有效选项。我们还可以使用`-v`选项来显示版本信息。`--script-version`参数以与`-v`或`-version`参数相同的方式显示版本，如下所示：

![](img/00006.jpeg)

下面的屏幕截图显示了当我们选择我们的一个有效的哈希算法时在控制台上打印的消息：

![](img/00007.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一些建议：

+   探索额外的`argparse`功能。例如，`argparse.FileType`对象可用于接受`File`对象作为输入。

+   我们还可以使用`argparse.ArgumentDefaultsHelpFormatter`类来显示我们为用户设置的默认值。当与可选参数结合使用时，这对于向用户显示如果没有指定将使用什么是有帮助的。

# 迭代松散的文件

示例难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

通常需要迭代一个目录及其子目录以递归处理所有文件。在这个示例中，我们将说明如何使用 Python 遍历目录并访问其中的文件。了解如何递归地浏览给定的输入目录是关键的，因为我们经常在我们的脚本中执行这个操作。

# 入门

这个脚本中使用的所有库都包含在 Python 的标准库中。在大多数情况下，用于处理文件和文件夹迭代的首选库是内置的`os`库。虽然这个库支持许多有用的操作，但我们将专注于`os.path()`和`os.walk()`函数。让我们使用以下文件夹层次结构作为示例来演示 Python 中的目录迭代是如何工作的：

```py
SecretDocs/
|-- key.txt
|-- Plans
|   |-- plans_0012b.txt
|   |-- plans_0016.txt
|   `-- Successful_Plans
|       |-- plan_0001.txt
|       |-- plan_0427.txt
|       `-- plan_0630.txt
|-- Spreadsheets
|   |-- costs.csv
|   `-- profit.csv
`-- Team
    |-- Contact18.vcf
    |-- Contact1.vcf
    `-- Contact6.vcf

4 directories, 11 files
```

# 如何做…

在这个示例中执行以下步骤：

1.  为要扫描的输入目录创建一个位置参数。

1.  遍历所有子目录并将文件路径打印到控制台。

# 它是如何工作的…

我们创建了一个非常基本的参数处理程序，接受一个位置输入`DIR_PATH`，即要迭代的输入目录的路径。例如，我们将使用`~/Desktop`路径作为脚本的输入参数，它是`SecretDocs`的父目录。我们解析命令行参数并将输入目录分配给一个本地变量。现在我们准备开始迭代这个输入目录：

```py
from __future__ import print_function
import argparse
import os

__authors__ = ["Chapin Bryce", "Preston Miller"]
__date__ = 20170815
__description__ = "Directory tree walker"

parser = argparse.ArgumentParser(
    description=__description__,
    epilog="Developed by {} on {}".format(
        ", ".join(__authors__), __date__)
)
parser.add_argument("DIR_PATH", help="Path to directory")
args = parser.parse_args()
path_to_scan = args.DIR_PATH
```

要迭代一个目录，我们需要提供一个表示其路径的字符串给`os.walk()`。这个方法在每次迭代中返回三个对象，我们已经在 root、directories 和 files 变量中捕获了这些对象：

+   `root`：这个值以字符串形式提供了当前目录的相对路径。使用示例目录结构，root 将从`SecretDocs`开始，最终变成`SecretDocs/Team`和`SecretDocs/Plans/SuccessfulPlans`。

+   `directories`：这个值是当前根目录中的子目录列表。我们可以遍历这个目录列表，尽管在后续的`os.walk()`调用中，这个列表中的条目将成为根值的一部分。因此，这个值并不经常使用。

+   `files`：这个值是当前根位置的文件列表。

在命名目录和文件变量时要小心。在 Python 中，`dir`和`file`名称被保留用于其他用途，不应该用作变量名。

```py
# Iterate over the path_to_scan
for root, directories, files in os.walk(path_to_scan):
```

通常会创建第二个 for 循环，如下面的代码所示，以遍历该目录中的每个文件，并对它们执行某些操作。使用`os.path.join()`方法，我们可以将根目录和`file_entry`变量连接起来，以获取文件的路径。然后我们将这个文件路径打印到控制台上。例如，我们还可以将这个文件路径追加到一个列表中，然后对列表进行迭代以处理每个文件：

```py
    # Iterate over the files in the current "root"
    for file_entry in files:
        # create the relative path to the file
        file_path = os.path.join(root, file_entry)
        print(file_path)
```

我们也可以使用`root + os.sep() + file_entry`来实现相同的效果，但这不如我们使用的连接路径的方法那样符合 Python 的风格。使用`os.path.join()`，我们可以传递两个或更多的字符串来形成单个路径，比如目录、子目录和文件。

当我们用示例输入目录运行上述脚本时，我们会看到以下输出：

![](img/00008.jpeg)

如所见，`os.walk()`方法遍历目录，然后会进入任何发现的子目录，从而扫描整个目录树。

# 还有更多...

这个脚本可以进一步改进。以下是一个建议：

+   查看并使用`glob`库实现类似功能，与`os`模块不同，它允许对文件和目录进行通配符模式递归搜索

# 记录文件属性

配方难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

现在我们可以遍历文件和文件夹，让我们学习如何记录这些对象的元数据。文件元数据在取证中扮演着重要的角色，因为收集和审查这些信息是大多数调查中的基本任务。使用单个 Python 库，我们可以跨平台收集一些最重要的文件属性。

# 开始

此脚本中使用的所有库都包含在 Python 的标准库中。`os`库再次可以在这里用于收集文件元数据。收集文件元数据最有帮助的方法之一是`os.stat()`函数。需要注意的是，`stat()`调用仅提供当前操作系统和挂载卷的文件系统可用的信息。大多数取证套件允许检查员将取证图像挂载为系统上的卷，并通常保留 stat 调用可用的`file`属性。在第八章，*使用取证证据容器配方*中，我们将演示如何打开取证获取以直接提取文件信息。

要了解更多关于`os`库的信息，请访问[`docs.python.org/3/library/os.html`](https://docs.python.org/3/library/os.html)。

# 如何做...

我们将使用以下步骤记录文件属性：

1.  获取要处理的输入文件。

1.  打印各种元数据：MAC 时间，文件大小，组和所有者 ID 等。

# 它是如何工作的...

首先，我们导入所需的库：`argparse`用于处理参数，`datetime`用于解释时间戳，`os`用于访问`stat()`方法。`sys`模块用于识别脚本正在运行的平台（操作系统）。接下来，我们创建我们的命令行处理程序，它接受一个参数`FILE_PATH`，表示我们将从中提取元数据的文件的路径。在继续执行脚本之前，我们将这个输入分配给一个本地变量：

```py
from __future__ import print_function
import argparse
from datetime import datetime as dt
import os
import sys

__authors__ = ["Chapin Bryce", "Preston Miller"]
__date__ = 20170815
__description__ = "Gather filesystem metadata of provided file"

parser = argparse.ArgumentParser(
    description=__description__,
    epilog="Developed by {} on {}".format(", ".join(__authors__), __date__)
)
parser.add_argument("FILE_PATH",
                    help="Path to file to gather metadata for")
args = parser.parse_args()
file_path = args.FILE_PATH
```

时间戳是收集的最常见的文件元数据属性之一。我们可以使用`os.stat()`方法访问创建、修改和访问时间戳。时间戳以表示自 1970-01-01 以来的秒数的浮点数返回。使用`datetime.fromtimestamp()`方法，我们将这个值转换为可读格式。

`os.stat()`模块根据平台不同而解释时间戳。例如，在 Windows 上，`st_ctime`值显示文件的创建时间，而在 macOS 和 UNIX 上，这个属性显示文件元数据的最后修改时间，类似于 NTFS 条目的修改时间。然而，`os.stat()`的其余部分在不同平台上是相同的。

```py
stat_info = os.stat(file_path)
if "linux" in sys.platform or "darwin" in sys.platform:
    print("Change time: ", dt.fromtimestamp(stat_info.st_ctime))
elif "win" in sys.platform:
    print("Creation time: ", dt.fromtimestamp(stat_info.st_ctime))
else:
    print("[-] Unsupported platform {} detected. Cannot interpret "
          "creation/change timestamp.".format(sys.platform)
          )
print("Modification time: ", dt.fromtimestamp(stat_info.st_mtime))
print("Access time: ", dt.fromtimestamp(stat_info.st_atime))
```

我们继续打印时间戳后的文件元数据。文件模式和`inode`属性分别返回文件权限和整数`inode`。设备 ID 指的是文件所在的设备。我们可以使用`os.major()`和`os.minor()`方法将这个整数转换为主设备标识符和次设备标识符：

```py
print("File mode: ", stat_info.st_mode)
print("File inode: ", stat_info.st_ino)
major = os.major(stat_info.st_dev)
minor = os.minor(stat_info.st_dev)
print("Device ID: ", stat_info.st_dev)
print("\tMajor: ", major)
print("\tMinor: ", minor)
```

`st_nlink`属性返回文件的硬链接数。我们可以分别使用`st_uid`和`st_gid`属性打印所有者和组信息。最后，我们可以使用`st_size`来获取文件大小，它返回一个表示文件大小的整数（以字节为单位）。

请注意，如果文件是符号链接，则`st_size`属性反映的是指向目标文件的路径的长度，而不是目标文件的大小。

```py
print("Number of hard links: ", stat_info.st_nlink)
print("Owner User ID: ", stat_info.st_uid)
print("Group ID: ", stat_info.st_gid)
print("File Size: ", stat_info.st_size)
```

但等等，这还不是全部！我们可以使用`os.path()`模块来提取更多的元数据。例如，我们可以使用它来确定文件是否是符号链接，就像下面展示的`os.islink()`方法一样。有了这个，我们可以警告用户，如果`st_size`属性不等于目标文件的大小。`os.path()`模块还可以获取绝对路径，检查它是否存在，并获取父目录。我们还可以使用`os.path.dirname()`函数或访问`os.path.split()`函数的第一个元素来获取父目录。`split()`方法更常用于从路径中获取文件名：

```py
# Gather other properties
print("Is a symlink: ", os.path.islink(file_path))
print("Absolute Path: ", os.path.abspath(file_path))
print("File exists: ", os.path.exists(file_path))
print("Parent directory: ", os.path.dirname(file_path))
print("Parent directory: {} | File name: {}".format(
    *os.path.split(file_path)))
```

通过运行脚本，我们可以获取有关文件的相关元数据。请注意，`format()`方法允许我们打印值，而不必担心它们的数据类型。通常情况下，如果我们直接打印变量而不使用字符串格式化，我们需要先将整数和其他数据类型转换为字符串：

![](img/00009.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一些建议：

+   将这个方法与*遍历松散文件*方法结合起来，递归地提取给定一系列目录中文件的元数据

+   实现逻辑以按文件扩展名、修改日期或文件大小进行过滤，以仅收集符合所需条件的文件的元数据信息

# 复制文件、属性和时间戳

方法难度：简单

Python 版本：2.7 或 3.5

操作系统：Windows

保留文件是数字取证中的一项基本任务。通常情况下，最好将文件容器化为可以存储松散文件的哈希和其他元数据的格式。然而，有时我们需要以数字取证的方式从一个位置复制文件到另一个位置。使用这个方法，我们将演示一些可用于复制文件并保留常见元数据字段的方法。

# 入门

这个方法需要安装两个第三方模块`pywin32`和`pytz`。此脚本中使用的所有其他库都包含在 Python 的标准库中。这个方法主要使用两个库，内置的`shutil`和第三方库`pywin32`。`shutil`库是我们在 Python 中复制文件的首选，我们可以使用它来保留大部分时间戳和其他文件属性。然而，`shutil`模块无法保留它复制的文件的创建时间。相反，我们必须依赖于特定于 Windows 的`pywin32`库来保留它。虽然`pywin32`库是特定于平台的，但它非常有用，可以与 Windows 操作系统进行交互。

要了解有关`shutil`库的更多信息，请访问[`docs.python.org/3/library/shutil.html`](https://docs.python.org/3/library/shutil.html)。

要安装`pywin32`，我们需要访问其 SourceForge 页面[`sourceforge.net/projects/pywin32/`](https://sourceforge.net/projects/pywin32/)并下载与我们的 Python 安装相匹配的版本。要检查我们的 Python 版本，我们可以导入`sys`模块并在解释器中调用`sys.version`。在选择正确的`pywin32`安装程序时，版本和架构都很重要。

要了解有关`sys`库的更多信息，请访问[`docs.python.org/3/library/sys.html`](https://docs.python.org/3/library/sys.html)。

除了安装`pywin32`库之外，我们还需要安装`pytz`，这是一个第三方库，用于在 Python 中管理时区。我们可以使用`pip`命令安装这个库：

```py
pip install pytz==2017.2
```

# 如何做…

我们执行以下步骤来在 Windows 系统上进行取证复制文件：

1.  收集源文件和目标参数。

1.  使用`shutil`来复制和保留大多数文件元数据。

1.  使用`win32file`手动设置时间戳属性。

# 它是如何工作的…

现在让我们深入研究复制文件并保留其属性和时间戳。我们使用一些熟悉的库来帮助我们执行这个配方。一些库，如`pytz`，`win32file`和`pywintypes`是新的。让我们在这里简要讨论它们的目的。`pytz`模块允许我们更细致地处理时区，并允许我们为`pywin32`库初始化日期。

为了让我们能够以正确的格式传递时间戳，我们还必须导入`pywintypes`。最后，`win32file`库，通过我们安装的`pywin32`提供了在 Windows 中进行文件操作的各种方法和常量：

```py
from __future__ import print_function
import argparse
from datetime import datetime as dt
import os
import pytz
from pywintypes import Time
import shutil
from win32file import SetFileTime, CreateFile, CloseHandle
from win32file import GENERIC_WRITE, FILE_SHARE_WRITE
from win32file import OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL

__authors__ = ["Chapin Bryce", "Preston Miller"]
__date__ = 20170815
__description__ = "Gather filesystem metadata of provided file"

```

这个配方的命令行处理程序接受两个位置参数，`source`和`dest`，分别代表要复制的源文件和输出目录。这个配方有一个可选参数`timezone`，允许用户指定一个时区。

为了准备源文件，我们存储绝对路径并从路径的其余部分中分离文件名，如果目标是目录，则稍后可能需要使用。我们最后的准备工作涉及从用户那里读取时区输入，这是四个常见的美国时区之一，以及 UTC。这使我们能够为后续在配方中使用初始化`pytz`时区对象：

```py
parser = argparse.ArgumentParser(
    description=__description__,
    epilog="Developed by {} on {}".format(
        ", ".join(__authors__), __date__)
)
parser.add_argument("source", help="Source file")
parser.add_argument("dest", help="Destination directory or file")
parser.add_argument("--timezone", help="Timezone of the file's timestamp",
                    choices=['EST5EDT', 'CST6CDT', 'MST7MDT', 'PST8PDT'],
                    required=True)
args = parser.parse_args()

source = os.path.abspath(args.source)
if os.sep in args.source:
    src_file_name = args.source.split(os.sep, 1)[1]
else:
    src_file_name = args.source

dest = os.path.abspath(args.dest)
tz = pytz.timezone(args.timezone)
```

在这一点上，我们可以使用`shutil.copy2()`方法将源文件复制到目标。这个方法接受目录或文件作为目标。`shutil` `copy()`和`copy2()`方法之间的主要区别在于`copy2()`方法还保留文件属性，包括最后写入时间和权限。这个方法不会在 Windows 上保留文件创建时间，为此我们需要利用`pywin32`绑定。

为此，我们必须通过使用以下`if`语句构建`copy2()`调用复制的文件的目标路径，以便在命令行提供目录时连接正确的路径：

```py
shutil.copy2(source, dest)
if os.path.isdir(dest):
    dest_file = os.path.join(dest, src_file_name)
else:
    dest_file = dest
```

接下来，我们为`pywin32`库准备时间戳。我们使用`os.path.getctime()`方法收集相应的 Windows 创建时间，并使用`datetime.fromtimestamp()`方法将整数值转换为日期。有了我们的`datetime`对象准备好了，我们可以通过使用指定的`timezone`使值具有时区意识，并在将时间戳打印到控制台之前将其提供给`pywintype.Time()`函数：

```py
created = dt.fromtimestamp(os.path.getctime(source))
created = Time(tz.localize(created))
modified = dt.fromtimestamp(os.path.getmtime(source))
modified = Time(tz.localize(modified))
accessed = dt.fromtimestamp(os.path.getatime(source))
accessed = Time(tz.localize(accessed))

print("Source\n======")
print("Created: {}\nModified: {}\nAccessed: {}".format(
    created, modified, accessed))
```

准备工作完成后，我们可以使用`CreateFile()`方法打开文件，并传递表示复制文件的字符串路径，然后是由 Windows API 指定的用于访问文件的参数。这些参数及其含义的详细信息可以在[`msdn.microsoft.com/en-us/library/windows/desktop/aa363858(v=vs.85).aspx﻿`](https://msdn.microsoft.com/en-us/library/windows/desktop/aa363858(v=vs.85).aspx)上进行查看：

```py
handle = CreateFile(dest_file, GENERIC_WRITE, FILE_SHARE_WRITE,
                    None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
SetFileTime(handle, created, accessed, modified)
CloseHandle(handle)
```

一旦我们有了一个打开的文件句柄，我们可以调用`SetFileTime()`函数按顺序更新文件的创建、访问和修改时间戳。设置了目标文件的时间戳后，我们需要使用`CloseHandle()`方法关闭文件句柄。为了向用户确认文件时间戳的复制成功，我们打印目标文件的创建、修改和访问时间：

```py
created = tz.localize(dt.fromtimestamp(os.path.getctime(dest_file)))
modified = tz.localize(dt.fromtimestamp(os.path.getmtime(dest_file)))
accessed = tz.localize(dt.fromtimestamp(os.path.getatime(dest_file)))
print("\nDestination\n===========")
print("Created: {}\nModified: {}\nAccessed: {}".format(
    created, modified, accessed))
```

脚本输出显示了成功保留时间戳的文件从源复制到目标的过程：

![](img/00010.jpeg)

# 还有更多…

这个脚本可以进一步改进。我们在这里提供了一些建议：

+   对源文件和目标文件进行哈希处理，以确保它们被成功复制。哈希处理在下一节的文件和数据流哈希处理配方中介绍。

+   输出文件复制的日志以及在复制过程中遇到的任何异常。

# 对文件和数据流进行哈希处理

配方难度：简单

Python 版本：2.7 或 3.5

操作系统：任意

文件哈希是确定文件完整性和真实性的广泛接受的标识符。虽然一些算法已经容易受到碰撞攻击，但这个过程在这个领域仍然很重要。在这个配方中，我们将介绍对一串字符和文件内容流进行哈希处理的过程。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。为了生成文件和其他数据源的哈希值，我们实现了`hashlib`库。这个内置库支持常见的算法，如 MD5、SHA-1、SHA-256 等。在撰写本书时，许多工具仍然利用 MD5 和 SHA-1 算法，尽管当前的建议是至少使用 SHA-256。或者，可以使用文件的多个哈希值来进一步减少哈希冲突的几率。虽然我们将展示其中一些算法，但还有其他不常用的算法可供选择。

要了解有关`hashlib`库的更多信息，请访问[`docs.python.org/3/library/hashlib.html`](https://docs.python.org/3/library/hashlib.html)。

# 如何做…

我们使用以下步骤对文件进行哈希处理：

1.  使用指定的输入文件和算法打印哈希文件名。

1.  使用指定的输入文件和算法打印哈希文件数据。

# 工作原理…

首先，我们必须像下面所示导入`hashlib`。为了方便使用，我们已经定义了一个算法字典，我们的脚本可以使用`MD5`、`SHA-1`、`SHA-256`和`SHA-512`。通过更新这个字典，我们可以支持其他具有`update()`和`hexdigest()`方法的哈希函数，包括一些不属于`hashlib`库的库中的函数：

```py
from __future__ import print_function
import argparse
import hashlib
import os

__authors__ = ["Chapin Bryce", "Preston Miller"]
__date__ = 20170815
__description__ = "Script to hash a file's name and contents"

available_algorithms = {
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512
}

parser = argparse.ArgumentParser(
    description=__description__,
    epilog="Developed by {} on {}".format(", ".join(__authors__), __date__)
)
parser.add_argument("FILE_NAME", help="Path of file to hash")
parser.add_argument("ALGORITHM", help="Hash algorithm to use",
                    choices=sorted(available_algorithms.keys()))
args = parser.parse_args()

input_file = args.FILE_NAME
hash_alg = args.ALGORITHM
```

注意我们如何使用字典和命令行提供的参数来定义我们的哈希算法对象，然后使用括号来初始化对象。这在添加新的哈希算法时提供了额外的灵活性。

定义了我们的哈希算法后，我们现在可以对文件的绝对路径进行哈希处理，这是在为 iOS 设备的 iTunes 备份命名文件时使用的类似方法，通过将字符串传递到`update()`方法中。当我们准备显示计算出的哈希的十六进制值时，我们可以在我们的`file_name`对象上调用`hexdigest()`方法：

```py
file_name = available_algorithms[hash_alg]()
abs_path = os.path.abspath(input_file)
file_name.update(abs_path.encode())

print("The {} of the filename is: {}".format(
    hash_alg, file_name.hexdigest()))
```

让我们继续打开文件并对其内容进行哈希处理。虽然我们可以读取整个文件并将其传递给 `hash` 函数，但并非所有文件都足够小以适应内存。为了确保我们的代码适用于更大的文件，我们将使用以下示例中的技术以分段方式读取文件并以块的方式进行哈希处理。

通过以 `rb` 打开文件，我们将确保读取文件的二进制内容，而不是可能存在的字符串内容。打开文件后，我们将定义缓冲区大小以读取内容，然后读取第一块数据。

进入 while 循环，我们将根据文件中的内容更新我们的哈希对象。只要文件中有内容，这是可能的，因为 `read()` 方法允许我们传递一个要读取的字节数的整数，如果整数大于文件中剩余的字节数，它将简单地传递给我们剩余的字节。

读取整个文件后，我们调用对象的 `hexdigest()` 方法来向检查员显示文件哈希：

```py
file_content = available_algorithms[hash_alg]()
with open(input_file, 'rb') as open_file:
    buff_size = 1024
    buff = open_file.read(buff_size)

    while buff:
        file_content.update(buff)
        buff = open_file.read(buff_size)

print("The {} of the content is: {}".format(
    hash_alg, file_content.hexdigest()))
```

当我们执行代码时，我们会看到两个打印语句的输出，显示文件的绝对路径和内容的哈希值。我们可以通过在命令行中更改算法来为文件生成额外的哈希：

![](img/00011.jpeg)

# 还有更多…

这个脚本可以进一步改进。以下是一个建议：

+   添加对其他哈希算法的支持，并在 `available_algorithms` 全局变量中创建相应的条目

# 使用进度条进行跟踪

示例难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

不幸的是，处理以千兆字节或兆字节为单位的数据时，长时间运行的脚本是司空见惯的。虽然您的脚本可能在顺利处理这些数据，但用户可能会认为它在三个小时后没有任何进展的情况下已经冻结。幸运的是，一些开发人员构建了一个非常简单的进度条库，让我们没有理由不将其纳入我们的代码中。

# 入门

此示例需要安装第三方模块 `tqdm`。此脚本中使用的所有其他库都包含在 Python 的标准库中。`tqdm` 库，发音为 taqadum，可以通过 `pip` 安装或从 GitHub 下载 [`github.com/tqdm/tqdm`](https://github.com/tqdm/tqdm)。要使用本示例中显示的所有功能，请确保您使用的是 4.11.2 版本，在 `tqdm` GitHub 页面上或使用以下命令通过 `pip` 获取：

```py
pip install tqdm==4.11.2
```

# 如何做…

要创建一个简单的进度条，我们按照以下步骤进行：

1.  导入 `tqdm` 和 `time`。

1.  使用 `tqdm` 和循环创建多个示例。

# 工作原理…

与所有其他示例一样，我们从导入开始。虽然我们只需要 `tqdm` 导入来启用进度条，但我们将使用时间模块来减慢脚本的速度，以更好地可视化进度条。我们使用水果列表作为我们的样本数据，并确定哪些水果的名称中包含 "berry" 或 "berries"：

```py
from __future__ import print_function
from time import sleep
import tqdm

fruits = [
    "Acai", "Apple", "Apricots", "Avocado", "Banana", "Blackberry",
    "Blueberries", "Cherries", "Coconut", "Cranberry", "Cucumber",
    "Durian", "Fig", "Grapefruit", "Grapes", "Kiwi", "Lemon", "Lime",
    "Mango", "Melon", "Orange", "Papaya", "Peach", "Pear", "Pineapple",
    "Pomegranate", "Raspberries", "Strawberries", "Watermelon"
]
```

以下的 for 循环非常简单，遍历我们的水果列表，在休眠一秒钟之前检查水果名称中是否包含子字符串 `berr`。通过在迭代器周围包装 `tqdm()` 方法，我们自动获得一个漂亮的进度条，显示完成百分比、已用时间、剩余时间、完成的迭代次数和总迭代次数。

这些显示选项是 `tqdm` 的默认选项，并且使用我们的列表对象的属性收集所有必要的信息。例如，该库几乎可以通过收集长度并根据每次迭代的时间和已经过的数量来计算其余部分，从而了解进度条的几乎所有细节：

```py
contains_berry = 0
for fruit in tqdm.tqdm(fruits):
    if "berr" in fruit.lower():
        contains_berry += 1
    sleep(.1)
print("{} fruit names contain 'berry' or 'berries'".format(contains_berry))
```

通过指定关键字参数，可以轻松地扩展默认配置以超出进度条。进度条对象也可以在循环开始之前创建，并使用列表对象`fruits`作为可迭代参数。以下代码展示了如何使用列表、描述和提供单位名称定义我们的进度条。

如果我们不是使用列表，而是使用另一种迭代器类型，该类型没有定义`__len__`属性，我们将需要手动使用`total`关键字提供总数。如果迭代的总数不可用，将仅显示有关经过的时间和每秒迭代次数的基本统计信息。

一旦我们进入循环，我们可以使用`set_postfix()`方法显示发现的结果数量。每次迭代都会在进度条右侧提供我们找到的命中数量的更新：

```py
contains_berry = 0
pbar = tqdm.tqdm(fruits, desc="Reviewing names", unit="fruits")
for fruit in pbar:
    if "berr" in fruit.lower():
        contains_berry += 1
    pbar.set_postfix(hits=contains_berry)
    sleep(.1)
print("{} fruit names contain 'berry' or 'berries'".format(contains_berry))
```

进度条的另一个常见用途是在一系列整数中测量执行。由于这是该库的常见用法，开发人员在库中构建了一个称为`trange()`的范围调用。请注意，我们可以在这里指定与之前相同的参数。由于数字较大，我们将在此处使用一个新参数`unit_scale`，它将大数字简化为一个带有字母表示数量的小数字：

```py
for i in tqdm.trange(10000000, unit_scale=True, desc="Trange: "):
    pass
```

当我们执行代码时，将显示以下输出。我们的第一个进度条显示默认格式，而第二个和第三个显示了我们添加的自定义内容：

![](img/00012.jpeg)

# 还有更多…

这个脚本可以进一步改进。以下是一个建议：

+   进一步探索`tqdm`库为开发人员提供的功能。考虑使用`tqdm.write()`方法在不中断进度条的情况下打印状态消息。

# 记录结果

食谱难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

进度条之外，我们通常需要向用户提供消息，描述执行过程中发生的任何异常、错误、警告或其他信息。通过日志记录，我们可以在执行过程中提供这些信息，并在文本文件中供将来参考。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。本食谱将使用内置的`logging`库向控制台和文本文件生成状态消息。

要了解更多关于`logging`库的信息，请访问[`docs.python.org/3/library/logging.html`](https://docs.python.org/3/library/logging.html)。

# 如何做…

以下步骤可用于有效记录程序执行数据：

1.  创建日志格式化字符串。

1.  在脚本执行期间记录各种消息类型。

# 工作原理…

现在让我们学习记录结果。在导入之后，我们通过使用`__file__`属性表示的脚本名称初始化一个实例来创建我们的`logger`对象。通过初始化`logging`对象，我们将为此脚本设置级别并指定各种格式化程序和处理程序。格式化程序提供了灵活性，可以定义每条消息显示哪些字段，包括时间戳、函数名称和消息级别。格式化字符串遵循 Python 字符串格式化的标准，这意味着我们可以为以下字符串指定填充：

```py
from __future__ import print_function
import logging
import sys

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

msg_fmt = logging.Formatter("%(asctime)-15s %(funcName)-20s"
                            "%(levelname)-8s %(message)s")
```

处理程序允许我们指定日志消息应记录在哪里，包括日志文件、标准输出（控制台）或标准错误。在下面的示例中，我们使用标准输出作为我们的流处理程序，并使用脚本名称加上`.log`扩展名作为文件处理程序。最后，我们将这些处理程序注册到我们的记录器对象中：

```py
strhndl = logging.StreamHandler(sys.stdout)
strhndl.setFormatter(fmt=msg_fmt)

fhndl = logging.FileHandler(__file__ + ".log", mode='a')
fhndl.setFormatter(fmt=msg_fmt)

logger.addHandler(strhndl)
logger.addHandler(fhndl)
```

日志库默认使用以下级别，按严重性递增：`NOTSET`、`DEBUG`、`INFORMATION`、`WARNING`、`ERROR`和`CRITICAL`。为了展示格式字符串的一些特性，我们将从函数中记录几种类型的消息：

```py
logger.info("information message")
logger.debug("debug message")

def function_one():
    logger.warning("warning message")

def function_two():
    logger.error("error message")

function_one()
function_two()
```

当我们执行此代码时，我们可以看到从脚本调用中获得的以下消息信息。检查生成的日志文件与在控制台中记录的内容相匹配：

![](img/00013.jpeg)

# 还有更多...

这个脚本可以进一步改进。这是一个建议：

+   在脚本出现错误或用户验证过程时，提供尽可能多的信息通常很重要。因此，我们建议实施额外的格式化程序和日志级别。使用`stderr`流是记录的最佳实践，因为我们可以在控制台上提供输出，而不会中断`stdout`。

# 多人成群，事情好办

食谱难度：中等

Python 版本：2.7 或 3.5

操作系统：任何

虽然 Python 以单线程闻名，但我们可以使用内置库来启动新进程来处理任务。通常，当有一系列可以同时运行的任务并且处理尚未受到硬件限制时，这是首选，例如网络带宽或磁盘速度。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。使用内置的`multiprocessing`库，我们可以处理大多数需要多个进程有效地解决问题的情况。

要了解有关`multiprocessing`库的更多信息，请访问[`docs.python.org/3/library/multiprocessing.html`](https://docs.python.org/3/library/multiprocessing.html)。

# 如何做...

通过以下步骤，我们展示了 Python 中的基本多进程支持：

1.  设置日志以记录`multiprocessing`活动。

1.  使用`multiprocessing`将数据附加到列表。

# 它是如何工作的...

现在让我们看看如何在 Python 中实现多进程。我们导入了`multiprocessing`库，缩写为`mp`，因为它太长了；`logging`和`sys`库用于线程状态消息；`time`库用于减慢我们示例的执行速度；`randint`方法用于生成每个线程应等待的时间：

```py
from __future__ import print_function
import logging
import multiprocessing as mp
from random import randint
import sys
import time
```

在创建进程之前，我们设置一个函数，它们将执行。这是我们在返回主线程之前应该执行的每个进程的任务。在这种情况下，我们将线程睡眠的秒数作为唯一参数。为了打印允许我们区分进程的状态消息，我们使用`current_process()`方法访问每个线程的名称属性：

```py
def sleepy(seconds):
    proc_name = mp.current_process().name
    logger.info("{} is sleeping for {} seconds.".format(
        proc_name, seconds))
    time.sleep(seconds)
```

定义了我们的工作函数后，我们创建了我们的`logger`实例，从上一个食谱中借用代码，并将其设置为仅记录到控制台。

```py
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
msg_fmt = logging.Formatter("%(asctime)-15s %(funcName)-7s "
                            "%(levelname)-8s %(message)s")
strhndl = logging.StreamHandler(sys.stdout)
strhndl.setFormatter(fmt=msg_fmt)
logger.addHandler(strhndl)
```

现在我们定义要生成的工作人员数量，并在 for 循环中创建它们。使用这种技术，我们可以轻松调整正在运行的进程数量。在我们的循环内，我们使用`Process`类定义每个`worker`，并设置我们的目标函数和所需的参数。一旦定义了进程实例，我们就启动它并将对象附加到列表以供以后使用：

```py
num_workers = 5
workers = []
for w in range(num_workers):
    p = mp.Process(target=sleepy, args=(randint(1, 20),))
    p.start()
    workers.append(p)
```

通过将`workers`附加到列表中，我们可以按顺序加入它们。在这种情况下，加入是指在执行继续之前等待进程完成的过程。如果我们不加入我们的进程，其中一个进程可能会在脚本的末尾继续并在其他进程完成之前完成代码。虽然这在我们的示例中不会造成很大问题，但它可能会导致下一段代码过早开始：

```py
for worker in workers:
    worker.join()
    logger.info("Joined process {}".format(worker.name))
```

当我们执行脚本时，我们可以看到进程随着时间的推移开始和加入。由于我们将这些项目存储在列表中，它们将以有序的方式加入，而不管一个工作人员完成需要多长时间。这在下面可见，因为`Process-5`在完成之前睡了 14 秒，与此同时，`Process-4`和`Process-3`已经完成：

![](img/00014.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一个建议：

+   与使用函数参数在线程之间传递数据不同，可以考虑使用管道和队列作为共享数据的替代方法。关于这些对象的更多信息可以在[`docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes`](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes.)找到。[﻿](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes.)
