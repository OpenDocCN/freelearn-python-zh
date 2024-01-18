# 处理数字取证容器配方

在本章中，我们将涵盖以下配方：

+   打开收购

+   收集收购和媒体信息

+   遍历文件

+   处理容器内的文件

+   搜索哈希

# 介绍

Sleuth Kit及其Python绑定`pytsk3`可能是最知名的Python数字取证库。该库提供了丰富的支持，用于访问和操作文件系统。借助支持库（如`pyewf`），它们可以用于处理EnCase流行的`E01`格式等常见数字取证容器。如果没有这些库（以及许多其他库），我们在数字取证中所能完成的工作将受到更多限制。由于其作为一体化文件系统分析工具的宏伟目标，`pytsk3`可能是我们在本书中使用的最复杂的库。

出于这个原因，我们专门制定了一些配方，探索了这个库的基本原理。到目前为止，配方主要集中在松散文件支持上。这种惯例到此为止。我们将会经常使用这个库来与数字取证证据进行交互。了解如何与数字取证容器进行交互将使您的Python数字取证能力提升到一个新的水平。

在本章中，我们将学习如何安装`pytsk3`和`pyewf`，这两个库将允许我们利用Sleuth Kit和`E01`镜像支持。此外，我们还将学习如何执行基本任务，如访问和打印分区表，遍历文件系统，按扩展名导出文件，以及在数字取证容器中搜索已知的不良哈希。您将学习以下内容：

+   安装和设置`pytsk3`和`pyewf`

+   打开数字取证收购，如`raw`和`E01`文件

+   提取分区表数据和`E01`元数据

+   递归遍历活动文件并创建活动文件列表电子表格

+   按文件扩展名从数字取证容器中导出文件

+   在数字取证容器中搜索已知的不良哈希

访问[www.packtpub.com/books/content/support](http://www.packtpub.com/books/content/support)下载本章的代码包。

# 打开收购

配方难度：中等

Python版本：2.7

操作系统：Linux

使用`pyewf`和`pytsk3`将带来一整套新的工具和操作，我们必须首先学习。在这个配方中，我们将从基础知识开始：打开数字取证容器。这个配方支持`raw`和`E01`镜像。请注意，与我们之前的脚本不同，由于在使用这些库的Python 3.X版本时发现了一些错误，这些配方将使用Python 2.X。也就是说，主要逻辑在两个版本之间并没有区别，可以很容易地移植。在学习如何打开容器之前，我们需要设置我们的环境。我们将在下一节中探讨这个问题。

# 入门

除了一些脚本之外，我们在本书的大部分内容中都是与操作系统无关的。然而，在这里，我们将专门提供在Ubuntu 16.04.2上构建的说明。在Ubuntu的新安装中，执行以下命令以安装必要的依赖项：

```py
sudo apt-get update && sudo apt-get -y upgrade 
sudo apt-get install python-pip git autoconf automake autopoint libtool pkg-config  
```

除了前面提到的两个库（`pytsk3`和`pyewf`）之外，我们还将使用第三方模块`tabulate`来在控制台打印表格。由于这是最容易安装的模块，让我们首先完成这个任务，执行以下操作：

```py
pip install tabulate==0.7.7
```

要了解更多关于tabulate库的信息，请访问[https://pypi.python.org/pypi/tabulate](https://pypi.python.org/pypi/tabulate)。

信不信由你，我们也可以使用`pip`安装`pytsk3`：

```py
pip install pytsk3==20170802
```

要了解更多关于`pytsk3`库的信息，请访问[https://github.com/py4n6/pytsk.](https://github.com/py4n6/pytsk)

最后，对于`pyewf`，我们必须采取稍微绕弯的方法，从其GitHub存储库中安装，[https://github.com/libyal/libewf/releases](https://github.com/libyal/libewf/releases)。这些配方是使用`libewf-experimental-20170605`版本编写的，我们建议您在这里安装该版本。一旦包被下载并解压，打开提取目录中的命令提示符，并执行以下操作：

```py
./synclibs.sh 
./autogen.sh 
sudo python setup.py build 
sudo python setup.py install 
```

要了解更多关于`pyewf`库的信息，请访问：[https://github.com/libyal/libewf.](https://github.com/libyal/libewf)

毋庸置疑，对于这个脚本，您需要一个`raw`或`E01`证据文件来运行这些配方。对于第一个脚本，我们建议使用逻辑图像，比如来自[http://dftt.sourceforge.net/test2/index.html](http://dftt.sourceforge.net/test2/index.html)的`fat-img-kw.dd`。原因是这个第一个脚本将缺少一些处理物理磁盘图像及其分区所需的必要逻辑。我们将在*收集获取和媒体信息*配方中介绍这个功能。

# 操作步骤...

我们采用以下方法来打开法证证据容器：

1.  确定证据容器是`raw`图像还是`E01`容器。

1.  使用`pytsk3`访问图像。

1.  在控制台上打印根级文件夹和文件的表格。

# 它是如何工作的...

我们导入了一些库来帮助解析参数、处理证据容器和文件系统，并创建表格式的控制台数据。

```py
from __future__ import print_function
import argparse
import os
import pytsk3
import pyewf
import sys
from tabulate import tabulate
```

这个配方的命令行处理程序接受两个位置参数，`EVIDENCE_FILE`和`TYPE`，它们代表证据文件的路径和证据文件的类型（即`raw`或`ewf`）。请注意，对于分段的`E01`文件，您只需要提供第一个`E01`的路径（假设其他分段在同一个目录中）。在对证据文件进行一些输入验证后，我们将提供两个输入给`main()`函数，并开始执行脚本。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("TYPE",
                        help="Type of evidence: raw (dd) or EWF (E01)",
                        choices=("raw", "ewf"))
    parser.add_argument("-o", "--offset",
                        help="Partition byte offset", type=int)
    args = parser.parse_args()

    if os.path.exists(args.EVIDENCE_FILE) and \
            os.path.isfile(args.EVIDENCE_FILE):
        main(args.EVIDENCE_FILE, args.TYPE, args.offset)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)
```

在`main()`函数中，我们首先检查我们正在处理的证据文件的类型。如果是`E01`容器，我们需要首先使用`pyewf`创建一个句柄，然后才能使用`pytsk3`访问其内容。对于`raw`图像，我们可以直接使用`pytsk3`访问其内容，而无需先执行这个中间步骤。

在这里使用`pyewf.glob()`方法来组合`E01`容器的所有段，如果有的话，并将段的名称存储在一个列表中。一旦我们有了文件名列表，我们就可以创建`E01`句柄对象。然后我们可以使用这个对象来打开`filenames`。

```py
def main(image, img_type, offset):
    print("[+] Opening {}".format(image))
    if img_type == "ewf":
        try:
            filenames = pyewf.glob(image)
        except IOError:
            _, e, _ = sys.exc_info()
            print("[-] Invalid EWF format:\n {}".format(e))
            sys.exit(2)
        ewf_handle = pyewf.handle()
        ewf_handle.open(filenames)
```

接下来，我们必须将`ewf_handle`传递给`EWFImgInfo`类，该类将创建`pytsk3`对象。这里的else语句是为了`raw`图像，可以使用`pytsk3.Img_Info`函数来实现相同的任务。现在让我们看看`EWFImgInfo`类，了解EWF文件是如何稍有不同地处理的。

```py
        # Open PYTSK3 handle on EWF Image
        img_info = EWFImgInfo(ewf_handle)
    else:
        img_info = pytsk3.Img_Info(image)
```

这个脚本组件的代码来自`pyewf`的Python开发页面的*将pyewf与pytsk3结合使用*部分。

了解更多关于`pyewf`函数的信息，请访问[https://github.com/libyal/libewf/wiki/Development](https://github.com/libyal/libewf/wiki/Development)。

这个`EWFImgInfo`类继承自`pytsk3.Img_Info`基类，属于`TSK_IMG_TYPE_EXTERNAL`类型。重要的是要注意，接下来定义的三个函数，`close()`、`read()`和`get_size()`，都是`pytsk3`要求的，以便与证据容器进行适当的交互。有了这个简单的类，我们现在可以使用`pytsk3`来处理任何提供的`E01`文件。

```py
class EWFImgInfo(pytsk3.Img_Info):
    def __init__(self, ewf_handle):
        self._ewf_handle = ewf_handle
        super(EWFImgInfo, self).__init__(url="",
                                         type=pytsk3.TSK_IMG_TYPE_EXTERNAL)

    def close(self):
        self._ewf_handle.close()

    def read(self, offset, size):
        self._ewf_handle.seek(offset)
        return self._ewf_handle.read(size)

    def get_size(self):
        return self._ewf_handle.get_media_size()
```

回到`main()`函数，我们已经成功地为`raw`或`E01`镜像创建了`pytsk3`处理程序。现在我们可以开始访问文件系统。如前所述，此脚本旨在处理逻辑图像而不是物理图像。我们将在下一个步骤中引入对物理图像的支持。访问文件系统非常简单；我们通过在`pytsk3`处理程序上调用`FS_Info()`函数来实现。

```py
    # Get Filesystem Handle
    try:
        fs = pytsk3.FS_Info(img_info, offset)
    except IOError:
        _, e, _ = sys.exc_info()
        print("[-] Unable to open FS:\n {}".format(e))
        exit()
```

有了对文件系统的访问权限，我们可以遍历根目录中的文件夹和文件。首先，我们使用文件系统上的`open_dir()`方法，并指定根目录`**/**`作为输入来访问根目录。接下来，我们创建一个嵌套的列表结构，用于保存表格内容，稍后我们将使用`tabulate`将其打印到控制台。这个列表的第一个元素是表格的标题。

之后，我们将开始遍历图像，就像处理任何Python可迭代对象一样。每个对象都有各种属性和函数，我们从这里开始使用它们。首先，我们使用`f.info.name.name`属性提取对象的名称。然后，我们使用`f.info.meta.type`属性检查我们处理的是目录还是文件。如果这等于内置的`TSK_FS_META_TYPE_DIR`对象，则将`f_type`变量设置为`DIR`；否则，设置为`FILE`。

最后，我们使用更多的属性来提取目录或文件的大小，并创建和修改时间戳。请注意，对象时间戳存储在`Unix`时间中，如果您想以人类可读的格式显示它们，必须进行转换。提取了这些属性后，我们将数据附加到`table`列表中，并继续处理下一个对象。一旦我们完成了对根文件夹中所有对象的处理，我们就使用`tabulate`将数据打印到控制台。通过向`tabulate()`方法提供列表并将`headers`关键字参数设置为`firstrow`，以指示应使用列表中的第一个元素作为表头，可以在一行中完成此操作。

```py
    root_dir = fs.open_dir(path="/")
    table = [["Name", "Type", "Size", "Create Date", "Modify Date"]]
    for f in root_dir:
        name = f.info.name.name
        if f.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR:
            f_type = "DIR"
        else:
            f_type = "FILE"
        size = f.info.meta.size
        create = f.info.meta.crtime
        modify = f.info.meta.mtime
        table.append([name, f_type, size, create, modify])
    print(tabulate(table, headers="firstrow"))
```

当我们运行脚本时，我们可以了解到在证据容器的根目录中看到的文件和文件夹，如下截图所示：

![](../images/00090.jpeg)

# 收集获取和媒体信息

食谱难度：中等

Python版本：2.7

操作系统：Linux

在这个食谱中，我们学习如何使用`tabulate`查看和打印分区表。此外，对于`E01`容器，我们将打印存储在证据文件中的`E01`获取和容器元数据。通常，我们将使用给定机器的物理磁盘镜像。在接下来的任何过程中，我们都需要遍历不同的分区（或用户选择的分区）来获取文件系统及其文件的处理。因此，这个食谱对于我们建立对Sleuth Kit及其众多功能的理解至关重要。

# 入门

有关`pytsk3`、`pyewf`和`tabulate`的构建环境和设置详细信息，请参阅*打开获取*食谱中的*入门*部分。此脚本中使用的所有其他库都包含在Python的标准库中。

# 如何操作...

该食谱遵循以下基本步骤：

1.  确定证据容器是`raw`镜像还是`E01`容器。

1.  使用`pytsk3`访问镜像。

1.  如果适用，将`E01`元数据打印到控制台。

1.  将分区表数据打印到控制台。

# 它是如何工作的...

我们导入了许多库来帮助解析参数、处理证据容器和文件系统，并创建表格式的控制台数据。

```py
from __future__ import print_function
import argparse
import os
import pytsk3
import pyewf
import sys
from tabulate import tabulate
```

这个配方的命令行处理程序接受两个位置参数，`EVIDENCE_FILE`和`TYPE`，它们代表证据文件的路径和证据文件的类型。此外，如果用户在处理证据文件时遇到困难，他们可以使用可选的`p`开关手动提供分区。这个开关在大多数情况下不应该是必要的，但作为一种预防措施已经添加。在执行输入验证检查后，我们将这三个参数传递给`main（）`函数。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("TYPE", help="Type of Evidence",
                        choices=("raw", "ewf"))
    parser.add_argument("-p", help="Partition Type",
                        choices=("DOS", "GPT", "MAC", "SUN"))
    args = parser.parse_args()

    if os.path.exists(args.EVIDENCE_FILE) and \
            os.path.isfile(args.EVIDENCE_FILE):
        main(args.EVIDENCE_FILE, args.TYPE, args.p)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)
```

`main（）`函数在很大程度上与之前的配方相似，至少最初是这样。我们必须首先创建`pyewf`句柄，然后使用`EWFImgInfo`类来创建，如前面在`pytsk3`句柄中所示。如果您想了解更多关于`EWFImgInfo`类的信息，请参阅*打开获取*配方。但是，请注意，我们添加了一个额外的行调用`e01_metadata（）`函数来将`E01`元数据打印到控制台。现在让我们来探索一下这个函数。

```py
def main(image, img_type, part_type):
    print("[+] Opening {}".format(image))
    if img_type == "ewf":
        try:
            filenames = pyewf.glob(image)
        except IOError:
            print("[-] Invalid EWF format:\n {}".format(e))
            sys.exit(2)

        ewf_handle = pyewf.handle()
        ewf_handle.open(filenames)
        e01_metadata(ewf_handle)

        # Open PYTSK3 handle on EWF Image
        img_info = EWFImgInfo(ewf_handle)
    else:
        img_info = pytsk3.Img_Info(image)
```

`e01_metadata（）`函数主要依赖于`get_header_values（）`和`get_hash_values（）`方法来获取`E01`特定的元数据。`get_header_values（）`方法返回各种类型的获取和媒体元数据的`键值`对字典。我们使用循环来遍历这个字典，并将`键值`对打印到控制台。

同样，我们使用`hashes`字典的循环将图像的存储获取哈希打印到控制台。最后，我们调用一个属性和一些函数来打印获取大小的元数据。

```py
def e01_metadata(e01_image):
    print("\nEWF Acquisition Metadata")
    print("-" * 20)
    headers = e01_image.get_header_values()
    hashes = e01_image.get_hash_values()
    for k in headers:
        print("{}: {}".format(k, headers[k]))
    for h in hashes:
        print("Acquisition {}: {}".format(h, hashes[h]))
    print("Bytes per Sector: {}".format(e01_image.bytes_per_sector))
    print("Number of Sectors: {}".format(
        e01_image.get_number_of_sectors()))
    print("Total Size: {}".format(e01_image.get_media_size()))
```

有了这些，我们现在可以回到`main（）`函数。回想一下，在本章的第一个配方中，我们没有为物理获取创建支持（这完全是有意的）。然而，现在，我们使用`Volume_Info（）`函数添加了对此的支持。虽然`pytsk3`一开始可能令人生畏，但要欣赏到目前为止我们介绍的主要函数中使用的命名约定的一致性：`Img_Info`、`FS_Info`和`Volume_Info`。这三个函数对于访问证据容器的内容至关重要。在这个配方中，我们不会使用`FS_Info（）`函数，因为这里的目的只是打印分区表。

我们尝试在`try-except`块中访问卷信息。首先，我们检查用户是否提供了`p`开关，如果是，则将该分区类型的属性分配给一个变量。然后，我们将它与`pytsk3`句柄一起提供给`Volume_Info`方法。否则，如果没有指定分区，我们调用`Volume_Info`方法，并只提供`pytsk3`句柄对象。如果我们尝试这样做时收到`IOError`，我们将捕获异常作为`e`并将其打印到控制台，然后退出。如果我们能够访问卷信息，我们将其传递给`part_metadata（）`函数，以将分区数据打印到控制台。

```py
    try:
        if part_type is not None:
            attr_id = getattr(pytsk3, "TSK_VS_TYPE_" + part_type)
            volume = pytsk3.Volume_Info(img_info, attr_id)
        else:
            volume = pytsk3.Volume_Info(img_info)
    except IOError:
        _, e, _ = sys.exc_info()
        print("[-] Unable to read partition table:\n {}".format(e))
        sys.exit(3)
    part_metadata(volume)
```

`part_metadata（）`函数在逻辑上相对较轻。我们创建一个嵌套的列表结构，如前面的配方中所见，第一个元素代表最终的表头。接下来，我们遍历卷对象，并将分区地址、类型、偏移量和长度附加到`table`列表中。一旦我们遍历了分区，我们使用`tabulate`使用`firstrow`作为表头将这些数据的表格打印到控制台。

```py
def part_metadata(vol):
    table = [["Index", "Type", "Offset Start (Sectors)",
              "Length (Sectors)"]]
    for part in vol:
        table.append([part.addr, part.desc.decode("utf-8"), part.start,
                      part.len])
    print("\n Partition Metadata")
    print("-" * 20)
    print(tabulate(table, headers="firstrow"))
```

运行此代码时，如果存在，我们可以在控制台中查看有关获取和分区信息的信息：

![](../images/00091.jpeg)

# 遍历文件

配方难度：中等

Python版本：2.7

操作系统：Linux

在这个配方中，我们学习如何递归遍历文件系统并创建一个活动文件列表。作为法庭鉴定人，我们经常被问到的第一个问题之一是“设备上有什么数据？”。在这里，活动文件列表非常有用。在Python中，创建松散文件的文件列表是一个非常简单的任务。然而，这将会稍微复杂一些，因为我们处理的是法庭图像而不是松散文件。这个配方将成为未来脚本的基石，因为它将允许我们递归访问和处理图像中的每个文件。正如您可能已经注意到的，本章的配方是相互建立的，因为我们开发的每个函数都需要进一步探索图像。类似地，这个配方将成为未来配方中的一个重要部分，用于迭代目录并处理文件。

# 入门

有关`pytsk3`和`pyewf`的构建环境和设置详细信息，请参考*开始*部分中的*打开获取*配方。此脚本中使用的所有其他库都包含在Python的标准库中。

# 如何做...

我们在这个配方中执行以下步骤：

1.  确定证据容器是`raw`图像还是`E01`容器。

1.  使用`pytsk3`访问法庭图像。

1.  递归遍历每个分区中的所有目录。

1.  将文件元数据存储在列表中。

1.  将`active`文件列表写入CSV。

# 工作原理...

我们导入了许多库来帮助解析参数、解析日期、创建CSV电子表格，以及处理证据容器和文件系统。

```py
from __future__ import print_function
import argparse
import csv
from datetime import datetime
import os
import pytsk3
import pyewf
import sys
```

这个配方的命令行处理程序接受三个位置参数，`EVIDENCE_FILE`、`TYPE`和`OUTPUT_CSV`，分别代表证据文件的路径、证据文件的类型和输出CSV文件。与上一个配方类似，可以提供可选的`p`开关来指定分区类型。我们使用`os.path.dirname()`方法来提取CSV文件的所需输出目录路径，并使用`os.makedirs()`函数，如果不存在，则创建必要的输出目录。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("TYPE", help="Type of Evidence",
                        choices=("raw", "ewf"))
    parser.add_argument("OUTPUT_CSV", 
                        help="Output CSV with lookup results")
    parser.add_argument("-p", help="Partition Type",
                        choices=("DOS", "GPT", "MAC", "SUN"))
    args = parser.parse_args()

    directory = os.path.dirname(args.OUTPUT_CSV)
    if not os.path.exists(directory) and directory != "":
        os.makedirs(directory)
```

一旦我们通过检查输入证据文件是否存在并且是一个文件来验证了输入证据文件，四个参数将被传递给`main()`函数。如果在输入的初始验证中出现问题，脚本将在退出之前将错误打印到控制台。

```py
    if os.path.exists(args.EVIDENCE_FILE) and \
            os.path.isfile(args.EVIDENCE_FILE):
        main(args.EVIDENCE_FILE, args.TYPE, args.OUTPUT_CSV, args.p)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)
```

在`main()`函数中，我们用`None`实例化卷变量，以避免在脚本后面引用它时出错。在控制台打印状态消息后，我们检查证据类型是否为`E01`，以便正确处理它并创建有效的`pyewf`句柄，如在*打开获取*配方中更详细地演示的那样。有关更多详细信息，请参阅该配方。最终结果是为用户提供的证据文件创建`pytsk3`句柄`img_info`。

```py
def main(image, img_type, output, part_type):
    volume = None
    print("[+] Opening {}".format(image))
    if img_type == "ewf":
        try:
            filenames = pyewf.glob(image)
        except IOError:
            _, e, _ = sys.exc_info()
            print("[-] Invalid EWF format:\n {}".format(e))
            sys.exit(2)

        ewf_handle = pyewf.handle()
        ewf_handle.open(filenames)

        # Open PYTSK3 handle on EWF Image
        img_info = EWFImgInfo(ewf_handle)
    else:
        img_info = pytsk3.Img_Info(image)
```

接下来，我们尝试使用`pytsk3.Volume_Info()`方法访问图像的卷，通过提供图像句柄作为参数。如果提供了分区类型参数，我们将其属性ID添加为第二个参数。如果在尝试访问卷时收到`IOError`，我们将捕获异常作为`e`并将其打印到控制台。然而，请注意，当我们收到错误时，我们不会退出脚本。我们将在下一个函数中解释原因。最终，我们将`volume`、`img_info`和`output`变量传递给`open_fs()`方法。

```py
    try:
        if part_type is not None:
            attr_id = getattr(pytsk3, "TSK_VS_TYPE_" + part_type)
            volume = pytsk3.Volume_Info(img_info, attr_id)
        else:
            volume = pytsk3.Volume_Info(img_info)
    except IOError:
        _, e, _ = sys.exc_info()
        print("[-] Unable to read partition table:\n {}".format(e))

    open_fs(volume, img_info, output)
```

`open_fs()`方法尝试以两种方式访问容器的文件系统。如果`volume`变量不是`None`，它会遍历每个分区，并且如果该分区符合某些条件，则尝试打开它。但是，如果`volume`变量是`None`，它将尝试直接在图像句柄`img`上调用`pytsk3.FS_Info()`方法。正如我们所看到的，后一种方法将适用于逻辑图像，并为我们提供文件系统访问权限，而前一种方法适用于物理图像。让我们看看这两种方法之间的区别。

无论使用哪种方法，我们都创建一个`recursed_data`列表来保存我们的活动文件元数据。在第一种情况下，我们有一个物理图像，我们遍历每个分区，并检查它是否大于`2,048`扇区，并且在其描述中不包含`Unallocated`、`Extended`或`Primary Table`这些词。对于符合这些条件的分区，我们尝试使用`FS_Info()`函数访问它们的文件系统，方法是提供`pytsk3 img`对象和分区的偏移量（以字节为单位）。

如果我们能够访问文件系统，我们将使用`open_dir()`方法获取根目录，并将其与分区地址ID、文件系统对象、两个空列表和一个空字符串一起传递给`recurse_files()`方法。这些空列表和字符串将在对此函数进行递归调用时发挥作用，我们很快就会看到。一旦`recurse_files()`方法返回，我们将活动文件的元数据附加到`recursed_data`列表中。我们对每个分区重复这个过程。

```py
def open_fs(vol, img, output):
    print("[+] Recursing through files..")
    recursed_data = []
    # Open FS and Recurse
    if vol is not None:
        for part in vol:
            if part.len > 2048 and "Unallocated" not in part.desc and \
                    "Extended" not in part.desc and \
                    "Primary Table" not in part.desc:
                try:
                    fs = pytsk3.FS_Info(
                        img, offset=part.start * vol.info.block_size)
                except IOError:
                    _, e, _ = sys.exc_info()
                    print("[-] Unable to open FS:\n {}".format(e))
                root = fs.open_dir(path="/")
                data = recurse_files(part.addr, fs, root, [], [], [""])
                recursed_data.append(data)
```

对于第二种情况，我们有一个逻辑图像，卷是`None`。在这种情况下，我们尝试直接访问文件系统，如果成功，我们将其传递给`recurseFiles()`方法，并将返回的数据附加到我们的`recursed_data`列表中。一旦我们有了活动文件列表，我们将其和用户提供的输出文件路径发送到`csvWriter()`方法。让我们深入了解`recurseFiles()`方法，这是本教程的核心。

```py
    else:
        try:
            fs = pytsk3.FS_Info(img)
        except IOError:
            _, e, _ = sys.exc_info()
            print("[-] Unable to open FS:\n {}".format(e))
        root = fs.open_dir(path="/")
        data = recurse_files(1, fs, root, [], [], [""])
        recursed_data.append(data)
    write_csv(recursed_data, output)
```

`recurse_files()`函数基于*FLS*工具的一个示例（[https://github.com/py4n6/pytsk/blob/master/examples/fls.py](https://github.com/py4n6/pytsk/blob/master/examples/fls.py)）和David Cowen的工具DFIR Wizard（[https://github.com/dlcowen/dfirwizard/blob/master/dfirwizard-v9.py](https://github.com/dlcowen/dfirwizard/blob/master/dfirwizard-v9.py)）。为了启动这个函数，我们将根目录`inode`附加到`dirs`列表中。稍后将使用此列表以避免无休止的循环。接下来，我们开始循环遍历根目录中的每个对象，并检查它是否具有我们期望的某些属性，以及它的名称既不是`"**.**"`也不是`"**..**"`。

```py
def recurse_files(part, fs, root_dir, dirs, data, parent):
    dirs.append(root_dir.info.fs_file.meta.addr)
    for fs_object in root_dir:
        # Skip ".", ".." or directory entries without a name.
        if not hasattr(fs_object, "info") or \
                not hasattr(fs_object.info, "name") or \
                not hasattr(fs_object.info.name, "name") or \
                fs_object.info.name.name in [".", ".."]:
            continue
```

如果对象通过了这个测试，我们将使用`info.name.name`属性提取其名称。接下来，我们使用作为函数输入之一提供的`parent`变量手动为此对象创建文件路径。对于我们来说，没有内置的方法或属性可以自动执行此操作。

然后，我们检查文件是否是目录，并将`f_type`变量设置为适当的类型。如果对象是文件，并且具有扩展名，我们将提取它并将其存储在`file_ext`变量中。如果在尝试提取此数据时遇到`AttributeError`，我们将继续到下一个对象。

```py
        try:
            file_name = fs_object.info.name.name
            file_path = "{}/{}".format(
                "/".join(parent), fs_object.info.name.name)
            try:
                if fs_object.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR:
                    f_type = "DIR"
                    file_ext = ""
                else:
                    f_type = "FILE"
                    if "." in file_name:
                        file_ext = file_name.rsplit(".")[-1].lower()
                    else:
                        file_ext = ""
            except AttributeError:
                continue
```

与本章第一个示例类似，我们为对象大小和时间戳创建变量。但是，请注意，我们将日期传递给`convert_time()`方法。此函数用于将`Unix`时间戳转换为人类可读的格式。提取了这些属性后，我们使用分区地址ID将它们附加到数据列表中，以确保我们跟踪对象来自哪个分区。

```py
            size = fs_object.info.meta.size
            create = convert_time(fs_object.info.meta.crtime)
            change = convert_time(fs_object.info.meta.ctime)
            modify = convert_time(fs_object.info.meta.mtime)
            data.append(["PARTITION {}".format(part), file_name, file_ext,
                         f_type, create, change, modify, size, file_path])
```

如果对象是一个目录，我们需要递归遍历它，以访问其所有子目录和文件。为此，我们将目录名称附加到`parent`列表中。然后，我们使用`as_directory()`方法创建一个目录对象。我们在这里使用`inode`，这对于所有目的来说都是一个唯一的数字，并检查`inode`是否已经在`dirs`列表中。如果是这样，那么我们将不处理这个目录，因为它已经被处理过了。

如果需要处理目录，我们在新的`sub_directory`上调用`recurse_files()`方法，并传递当前的`dirs`、`data`和`parent`变量。一旦我们处理了给定的目录，我们就从`parent`列表中弹出该目录。如果不这样做，将导致错误的文件路径细节，因为除非删除，否则所有以前的目录将继续在路径中被引用。

这个函数的大部分内容都在一个大的`try-except`块中。我们传递在这个过程中生成的任何`IOError`异常。一旦我们遍历了所有的子目录，我们将数据列表返回给`open_fs()`函数。

```py
            if f_type == "DIR":
                parent.append(fs_object.info.name.name)
                sub_directory = fs_object.as_directory()
                inode = fs_object.info.meta.addr

                # This ensures that we don't recurse into a directory
                # above the current level and thus avoid circular loops.
                if inode not in dirs:
                    recurse_files(part, fs, sub_directory, dirs, data,
                                  parent)
                parent.pop(-1)

        except IOError:
            pass
    dirs.pop(-1)
    return data
```

让我们简要地看一下`convert_time()`函数。我们以前见过这种类型的函数：如果`Unix`时间戳不是`0`，我们使用`datetime.utcfromtimestamp()`方法将时间戳转换为人类可读的格式。

```py
def convert_time(ts):
    if str(ts) == "0":
        return ""
    return datetime.utcfromtimestamp(ts)
```

有了手头的活动文件列表数据，我们现在准备使用`write_csv()`方法将其写入CSV文件。如果我们找到了数据（即列表不为空），我们打开输出CSV文件，写入标题，并循环遍历`data`变量中的每个列表。我们使用`csvwriterows()`方法将每个嵌套列表结构写入CSV文件。

```py
def write_csv(data, output):
    if data == []:
        print("[-] No output results to write")
        sys.exit(3)

    print("[+] Writing output to {}".format(output))
    with open(output, "wb") as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = ["Partition", "File", "File Ext", "File Type",
                   "Create Date", "Modify Date", "Change Date", "Size",
                   "File Path"]
        csv_writer.writerow(headers)
        for result_list in data:
            csv_writer.writerows(result_list)
```

以下截图演示了这个示例从取证图像中提取的数据类型：

![](../images/00092.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个建议，如下所示：

+   使用`tqdm`或其他库创建进度条，以通知用户当前执行的进度

+   了解可以使用`pytsk3`从文件系统对象中提取的附加元数据值，并将它们添加到输出CSV文件中

# 处理容器内的文件

食谱难度：中等

Python版本：2.7

操作系统：Linux

现在我们可以遍历文件系统，让我们看看如何创建文件对象，就像我们习惯做的那样。在这个示例中，我们创建一个简单的分流脚本，提取与指定文件扩展名匹配的文件，并将它们复制到输出目录，同时保留它们的原始文件路径。

# 入门

有关构建环境和`pytsk3`和`pyewf`的设置详细信息，请参考*入门*部分中的*打开收购*食谱。此脚本中使用的所有其他库都包含在Python的标准库中。

# 如何做...

在这个示例中，我们将执行以下步骤：

1.  确定证据容器是`raw`镜像还是`E01`容器。

1.  使用`pytsk3`访问图像。

1.  递归遍历每个分区中的所有目录。

1.  检查文件扩展名是否与提供的扩展名匹配。

1.  将具有保留文件夹结构的响应文件写入输出目录。

# 它是如何工作的...

我们导入了许多库来帮助解析参数、创建CSV电子表格，并处理证据容器和文件系统。

```py
from __future__ import print_function
import argparse
import csv
import os
import pytsk3
import pyewf
import sys
```

这个示例的命令行处理程序接受四个位置参数：`EVIDENCE_FILE`、`TYPE`、`EXT`和`OUTPUT_DIR`。它们分别是证据文件本身、证据文件类型、要提取的逗号分隔的扩展名列表，以及所需的输出目录。我们还有可选的`p`开关，用于手动指定分区类型。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("TYPE", help="Type of Evidence",
                        choices=("raw", "ewf"))
    parser.add_argument("EXT",
                        help="Comma-delimited file extensions to extract")
    parser.add_argument("OUTPUT_DIR", help="Output Directory")
    parser.add_argument("-p", help="Partition Type",
                        choices=("DOS", "GPT", "MAC", "SUN"))
    args = parser.parse_args()
```

在调用`main()`函数之前，我们创建任何必要的输出目录，并执行我们的标准输入验证步骤。一旦我们验证了输入，我们将提供的参数传递给`main()`函数。

```py
    if not os.path.exists(args.OUTPUT_DIR):
        os.makedirs(args.OUTPUT_DIR)

    if os.path.exists(args.EVIDENCE_FILE) and \
            os.path.isfile(args.EVIDENCE_FILE):
        main(args.EVIDENCE_FILE, args.TYPE, args.EXT, args.OUTPUT_DIR,
             args.p)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)
```

`main()`函数、`EWFImgInfo`类和`open_fs()`函数在之前的配方中已经涵盖过。请记住，本章采用更迭代的方法来构建我们的配方。有关每个函数和`EWFImgInfo`类的更详细描述，请参考之前的配方。让我们简要地再次展示这两个函数，以避免逻辑上的跳跃。

在`main()`函数中，我们检查证据文件是`raw`文件还是`E01`文件。然后，我们执行必要的步骤，最终在证据文件上创建一个`pytsk3`句柄。有了这个句柄，我们尝试访问卷，使用手动提供的分区类型（如果提供）。如果我们能够打开卷，我们将`pytsk3`句柄和卷传递给`open_fs()`方法。

```py
def main(image, img_type, ext, output, part_type):
    volume = None
    print("[+] Opening {}".format(image))
    if img_type == "ewf":
        try:
            filenames = pyewf.glob(image)
        except IOError:
            _, e, _ = sys.exc_info()
            print("[-] Invalid EWF format:\n {}".format(e))
            sys.exit(2)

        ewf_handle = pyewf.handle()
        ewf_handle.open(filenames)

        # Open PYTSK3 handle on EWF Image
        img_info = EWFImgInfo(ewf_handle)
    else:
        img_info = pytsk3.Img_Info(image)

    try:
        if part_type is not None:
            attr_id = getattr(pytsk3, "TSK_VS_TYPE_" + part_type)
            volume = pytsk3.Volume_Info(img_info, attr_id)
        else:
            volume = pytsk3.Volume_Info(img_info)
    except IOError:
        _, e, _ = sys.exc_info()
        print("[-] Unable to read partition table:\n {}".format(e))

    open_fs(volume, img_info, ext, output)
```

在`open_fs()`函数中，我们使用逻辑来支持对文件系统进行逻辑和物理获取。对于逻辑获取，我们可以简单地尝试访问`pytsk3`句柄上文件系统的根。另一方面，对于物理获取，我们必须迭代每个分区，并尝试访问那些符合特定条件的文件系统。一旦我们访问到文件系统，我们调用`recurse_files()`方法来迭代文件系统中的所有文件。

```py
def open_fs(vol, img, ext, output):
    # Open FS and Recurse
    print("[+] Recursing through files and writing file extension matches "
          "to output directory")
    if vol is not None:
        for part in vol:
            if part.len > 2048 and "Unallocated" not in part.desc \
                    and "Extended" not in part.desc \
                    and "Primary Table" not in part.desc:
                try:
                    fs = pytsk3.FS_Info(
                        img, offset=part.start * vol.info.block_size)
                except IOError:
                    _, e, _ = sys.exc_info()
                    print("[-] Unable to open FS:\n {}".format(e))
                root = fs.open_dir(path="/")
                recurse_files(part.addr, fs, root, [], [""], ext, output)
    else:
        try:
            fs = pytsk3.FS_Info(img)
        except IOError:
            _, e, _ = sys.exc_info()
            print("[-] Unable to open FS:\n {}".format(e))
        root = fs.open_dir(path="/")
        recurse_files(1, fs, root, [], [""], ext, output)
```

不要浏览了！这个配方的新逻辑包含在`recurse_files()`方法中。这有点像眨眼就错过的配方。我们已经在之前的配方中做了大部分工作，现在我们基本上可以像处理任何其他Python文件一样处理这些文件。让我们看看这是如何工作的。

诚然，这个函数的第一部分仍然与以前相同，只有一个例外。在函数的第一行，我们使用列表推导来分割用户提供的每个逗号分隔的扩展名，并删除任何空格并将字符串规范化为小写。当我们迭代每个对象时，我们检查对象是目录还是文件。如果是文件，我们将文件的扩展名分离并规范化为小写，并将其存储在`file_ext`变量中。

```py
def recurse_files(part, fs, root_dir, dirs, parent, ext, output):
    extensions = [x.strip().lower() for x in ext.split(',')]
    dirs.append(root_dir.info.fs_file.meta.addr)
    for fs_object in root_dir:
        # Skip ".", ".." or directory entries without a name.
        if not hasattr(fs_object, "info") or \
                not hasattr(fs_object.info, "name") or \
                not hasattr(fs_object.info.name, "name") or \
                fs_object.info.name.name in [".", ".."]:
            continue
        try:
            file_name = fs_object.info.name.name
            file_path = "{}/{}".format("/".join(parent),
                                       fs_object.info.name.name)
            try:
                if fs_object.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR:
                    f_type = "DIR"
                    file_ext = ""
                else:
                    f_type = "FILE"
                    if "." in file_name:
                        file_ext = file_name.rsplit(".")[-1].lower()
                    else:
                        file_ext = ""
            except AttributeError:
                continue
```

接下来，我们检查提取的文件扩展名是否在用户提供的列表中。如果是，我们将文件对象本身及其名称、扩展名、路径和所需的输出目录传递给`file_writer()`方法进行输出。请注意，在这个操作中，我们有逻辑，即在前面的配方中讨论过的逻辑，来递归处理任何子目录，以识别更多符合扩展名条件的潜在文件。到目前为止，一切顺利；现在让我们来看看这最后一个函数。

```py
            if file_ext.strip() in extensions:
                print("{}".format(file_path))
                file_writer(fs_object, file_name, file_ext, file_path,
                            output)
            if f_type == "DIR":
                parent.append(fs_object.info.name.name)
                sub_directory = fs_object.as_directory()
                inode = fs_object.info.meta.addr
                if inode not in dirs:
                    recurse_files(part, fs, sub_directory, dirs,
                                  parent, ext, output)
                    parent.pop(-1)
        except IOError:
            pass
    dirs.pop(-1)
```

`file_writer()`方法依赖于文件对象的`read_random()`方法来访问文件内容。然而，在这之前，我们首先设置文件的输出路径，将用户提供的输出与扩展名和文件的路径结合起来。然后，如果这些目录不存在，我们就创建这些目录。接下来，我们以`"w"`模式打开输出文件，现在准备好将文件的内容写入输出文件。在这里使用的`read_random()`函数接受两个输入：文件中要开始读取的字节偏移量和要读取的字节数。在这种情况下，由于我们想要读取整个文件，我们使用整数`0`作为第一个参数，文件的大小作为第二个参数。

我们直接将其提供给`write()`方法，尽管请注意，如果我们要对这个文件进行任何处理，我们可以将其读入变量中，并从那里处理文件。另外，请注意，对于包含大文件的证据容器，将整个文件读入内存的这个过程可能并不理想。在这种情况下，您可能希望分块读取和写入这个文件，而不是一次性全部读取和写入。

```py
def file_writer(fs_object, name, ext, path, output):
    output_dir = os.path.join(output, ext,
                              os.path.dirname(path.lstrip("//")))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, name), "w") as outfile:
        outfile.write(fs_object.read_random(0, fs_object.info.meta.size))
```

当我们运行这个脚本时，我们会看到基于提供的扩展名的响应文件：

![](../images/00093.jpeg)

此外，我们可以在以下截图中查看这些文件的定义结构：

![](../images/00094.jpeg)

# 搜索哈希

配方难度：困难

Python版本：2.7

操作系统：Linux

在这个配方中，我们创建了另一个分类脚本，这次专注于识别与提供的哈希值匹配的文件。该脚本接受一个文本文件，其中包含以换行符分隔的`MD5`、`SHA-1`或`SHA-256`哈希，并在证据容器中搜索这些哈希。通过这个配方，我们将能够快速处理证据文件，找到感兴趣的文件，并通过将文件路径打印到控制台来提醒用户。

# 入门

参考*打开获取*配方中的*入门*部分，了解有关`build`环境和`pytsk3`和`pyewf`的设置详细信息。此脚本中使用的所有其他库都包含在Python的标准库中。

# 如何做...

我们使用以下方法来实现我们的目标：

1.  确定证据容器是`raw`图像还是`E01`容器。

1.  使用`pytsk3`访问图像。

1.  递归遍历每个分区中的所有目录。

1.  使用适当的哈希算法发送每个文件进行哈希处理。

1.  检查哈希是否与提供的哈希之一匹配，如果是，则打印到控制台。

# 工作原理...

我们导入了许多库来帮助解析参数、创建CSV电子表格、对文件进行哈希处理、处理证据容器和文件系统，并创建进度条。

```py
from __future__ import print_function
import argparse
import csv
import hashlib
import os
import pytsk3
import pyewf
import sys
from tqdm import tqdm
```

该配方的命令行处理程序接受三个位置参数，`EVIDENCE_FILE`，`TYPE`和`HASH_LIST`，分别表示证据文件，证据文件类型和要搜索的换行分隔哈希列表。与往常一样，用户也可以在必要时使用`p`开关手动提供分区类型。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("TYPE", help="Type of Evidence",
                        choices=("raw", "ewf"))
    parser.add_argument("HASH_LIST",
                        help="Filepath to Newline-delimited list of "
                             "hashes (either MD5, SHA1, or SHA-256)")
    parser.add_argument("-p", help="Partition Type",
                        choices=("DOS", "GPT", "MAC", "SUN"))
    parser.add_argument("-t", type=int,
                        help="Total number of files, for the progress bar")
    args = parser.parse_args()
```

在解析输入后，我们对证据文件和哈希列表进行了典型的输入验证检查。如果通过了这些检查，我们调用`main()`函数并提供用户提供的输入。

```py
    if os.path.exists(args.EVIDENCE_FILE) and \
            os.path.isfile(args.EVIDENCE_FILE) and \
            os.path.exists(args.HASH_LIST) and \
            os.path.isfile(args.HASH_LIST):
        main(args.EVIDENCE_FILE, args.TYPE, args.HASH_LIST, args.p, args.t)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)
```

与以前的配方一样，`main()`函数、`EWFImgInfo`类和`open_fs()`函数几乎与以前的配方相同。有关这些函数的更详细解释，请参考以前的配方。`main()`函数的一个新添加是第一行，我们在其中调用`read_hashes()`方法。该方法读取输入的哈希列表并返回哈希列表和哈希类型（即`MD5`、`SHA-1`或`SHA-256`）。

除此之外，`main()`函数的执行方式与我们习惯看到的方式相同。首先，它确定正在处理的证据文件的类型，以便在图像上创建一个`pytsk3`句柄。然后，它使用该句柄并尝试访问图像卷。完成此过程后，变量被发送到`open_fs()`函数进行进一步处理。

```py
def main(image, img_type, hashes, part_type, pbar_total=0):
    hash_list, hash_type = read_hashes(hashes)
    volume = None
    print("[+] Opening {}".format(image))
    if img_type == "ewf":
        try:
            filenames = pyewf.glob(image)
        except IOError:
            _, e, _ = sys.exc_info()
            print("[-] Invalid EWF format:\n {}".format(e))
            sys.exit(2)

        ewf_handle = pyewf.handle()
        ewf_handle.open(filenames)

        # Open PYTSK3 handle on EWF Image
        img_info = EWFImgInfo(ewf_handle)
    else:
        img_info = pytsk3.Img_Info(image)

    try:
        if part_type is not None:
            attr_id = getattr(pytsk3, "TSK_VS_TYPE_" + part_type)
            volume = pytsk3.Volume_Info(img_info, attr_id)
        else:
            volume = pytsk3.Volume_Info(img_info)
    except IOError:
        _, e, _ = sys.exc_info()
        print("[-] Unable to read partition table:\n {}".format(e))

    open_fs(volume, img_info, hash_list, hash_type, pbar_total)
```

让我们快速看一下新函数`read_hashes()`方法。首先，我们将`hash_list`和`hash_type`变量实例化为空列表和`None`对象。接下来，我们打开并遍历输入的哈希列表，并将每个哈希添加到我们的列表中。在这样做时，如果`hash_type`变量仍然是`None`，我们检查行的长度作为识别应该使用的哈希算法类型的手段。

在此过程结束时，如果`hash_type`变量仍然是`None`，则哈希列表必须由我们不支持的哈希组成，因此在将错误打印到控制台后退出脚本。

```py
def read_hashes(hashes):
    hash_list = []
    hash_type = None
    with open(hashes) as infile:
        for line in infile:
            if hash_type is None:
                if len(line.strip()) == 32:
                    hash_type = "md5"
                elif len(line.strip()) == 40:
                    hash_type == "sha1"
                elif len(line.strip()) == 64:
                    hash_type == "sha256"
            hash_list.append(line.strip().lower())
    if hash_type is None:
        print("[-] No valid hashes identified in {}".format(hashes))
        sys.exit(3)

    return hash_list, hash_type
```

`open_fs()`方法函数与以前的配方相同。它尝试使用两种不同的方法来访问物理和逻辑文件系统。一旦成功，它将这些文件系统传递给`recurse_files()`方法。与以前的配方一样，这个函数中发生了奇迹。我们还使用`tqdm`来提供进度条，向用户提供反馈，因为在图像中对所有文件进行哈希可能需要一段时间。

```py
def open_fs(vol, img, hashes, hash_type, pbar_total=0):
    # Open FS and Recurse
    print("[+] Recursing through and hashing files")
    pbar = tqdm(desc="Hashing", unit=" files",
                unit_scale=True, total=pbar_total)
    if vol is not None:
        for part in vol:
            if part.len > 2048 and "Unallocated" not in part.desc and \
                    "Extended" not in part.desc and \
                    "Primary Table" not in part.desc:
                try:
                    fs = pytsk3.FS_Info(
                        img, offset=part.start * vol.info.block_size)
                except IOError:
                    _, e, _ = sys.exc_info()
                    print("[-] Unable to open FS:\n {}".format(e))
                root = fs.open_dir(path="/")
                recurse_files(part.addr, fs, root, [], [""], hashes,
                              hash_type, pbar)
    else:
        try:
            fs = pytsk3.FS_Info(img)
        except IOError:
            _, e, _ = sys.exc_info()
            print("[-] Unable to open FS:\n {}".format(e))
        root = fs.open_dir(path="/")
        recurse_files(1, fs, root, [], [""], hashes, hash_type, pbar)
    pbar.close()
```

在`recurse_files()`方法中，我们遍历所有子目录并对每个文件进行哈希处理。我们跳过`。`和`..`目录条目，并检查`fs_object`是否具有正确的属性。如果是，我们构建文件路径以在输出中使用。

```py
def recurse_files(part, fs, root_dir, dirs, parent, hashes,
                  hash_type, pbar):
    dirs.append(root_dir.info.fs_file.meta.addr)
    for fs_object in root_dir:
        # Skip ".", ".." or directory entries without a name.
        if not hasattr(fs_object, "info") or \
                not hasattr(fs_object.info, "name") or \
                not hasattr(fs_object.info.name, "name") or \
                fs_object.info.name.name in [".", ".."]:
            continue
        try:
            file_path = "{}/{}".format("/".join(parent),
                                       fs_object.info.name.name)
```

在执行每次迭代时，我们确定哪些对象是文件，哪些是目录。对于发现的每个文件，我们将其发送到`hash_file()`方法，以及其路径，哈希列表和哈希算法。`recurse_files()`函数逻辑的其余部分专门设计用于处理目录，并对任何子目录进行递归调用，以确保整个树都被遍历并且不会错过文件。

```py
            if getattr(fs_object.info.meta, "type", None) == \
                    pytsk3.TSK_FS_META_TYPE_DIR:
                parent.append(fs_object.info.name.name)
                sub_directory = fs_object.as_directory()
                inode = fs_object.info.meta.addr

                # This ensures that we don't recurse into a directory
                # above the current level and thus avoid circular loops.
                if inode not in dirs:
                    recurse_files(part, fs, sub_directory, dirs,
                                  parent, hashes, hash_type, pbar)
                    parent.pop(-1)
            else:
                hash_file(fs_object, file_path, hashes, hash_type, pbar)

        except IOError:
            pass
    dirs.pop(-1)
```

`hash_file()`方法首先检查要创建的哈希算法实例的类型，根据`hash_type`变量。确定了这一点，并更新了文件大小到进度条，我们使用`read_random()`方法将文件的数据读入哈希对象。同样，我们通过从第一个字节开始读取并读取整个文件的大小来读取整个文件的内容。我们使用哈希对象上的`hexdigest()`函数生成文件的哈希，然后检查该哈希是否在我们提供的哈希列表中。如果是，我们通过打印文件路径来提醒用户，使用`pbar.write()`来防止进度条显示问题，并将名称打印到控制台。

```py
def hash_file(fs_object, path, hashes, hash_type, pbar):
    if hash_type == "md5":
        hash_obj = hashlib.md5()
    elif hash_type == "sha1":
        hash_obj = hashlib.sha1()
    elif hash_type == "sha256":
        hash_obj = hashlib.sha256()
    f_size = getattr(fs_object.info.meta, "size", 0)
    pbar.set_postfix(File_Size="{:.2f}MB".format(f_size / 1024.0 / 1024))
    hash_obj.update(fs_object.read_random(0, f_size))
    hash_digest = hash_obj.hexdigest()
    pbar.update()

    if hash_digest in hashes:
        pbar.write("[*] MATCH: {}\n{}".format(path, hash_digest))
```

通过运行脚本，我们可以看到一个漂亮的进度条，显示哈希状态和与提供的哈希列表匹配的文件列表，如下面的屏幕截图所示：

![](../images/00095.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个建议，如下所示：

+   而不是打印匹配项，创建一个包含匹配文件的元数据的CSV文件以供审查。

+   添加一个可选开关，将匹配的文件转储到输出目录（保留文件夹路径）
