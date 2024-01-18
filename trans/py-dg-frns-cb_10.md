# 探索Windows取证工件食谱-第二部分

在本章中，将涵盖以下内容：

+   解析预取文件

+   一系列幸运事件

+   索引互联网历史记录

+   昔日的阴影

+   解剖SRUM数据库

# 介绍

微软Windows是在取证分析中发现的机器上最常见的操作系统之一。这导致社区在过去的二十年中付出了大量努力，以开发、共享和记录这个操作系统产生的证据，用于取证工作。

在本章中，我们将继续研究各种Windows取证工件以及如何使用Python处理它们。我们将利用我们在[第8章](part0241.html#75QNI0-260f9401d2714cb9ab693c4692308abe)中开发的框架，直接从取证获取中处理这些工件。我们将使用各种`libyal`库来处理各种文件的底层处理，包括`pyevt`、`pyevtx`、`pymsiecf`、`pyvshadow`和`pyesedb`。我们还将探讨如何使用`struct`和偏移量和感兴趣的数据类型的文件格式表来处理预取文件。在本章中，我们将学习以下内容：

+   解析预取文件以获取应用程序执行信息

+   搜索事件日志并将事件提取到电子表格中

+   从`index.dat`文件中提取互联网历史记录

+   枚举和创建卷影复制的文件列表

+   解剖Windows 10 SRUM数据库

`libyal`存储库的完整列表，请访问[https://github.com/libyal](https://github.com/libyal)。访问[www.packtpub.com/books/content/support](http://www.packtpub.com/books/content/support)下载本章的代码包。

# 解析预取文件

食谱难度：中等

Python版本：2.7

操作系统：Linux

预取文件是一个常见的证据，用于获取有关应用程序执行的信息。虽然它们可能并不总是存在，但在存在的情况下，无疑值得审查。请记住，根据`SYSTEM`注册表中`PrefetchParameters`子键的值，可以启用或禁用预取。此示例搜索具有预取扩展名（`.pf`）的文件，并处理它们以获取有价值的应用程序信息。我们将仅演示这个过程用于Windows XP的预取文件；但请注意，我们使用的基本过程类似于Windows的其他版本。

# 入门

因为我们决定在Ubuntu环境中构建Sleuth Kit及其依赖项，所以我们将继续在该操作系统上进行开发，以便使用。如果尚未安装，此脚本将需要安装三个额外的库：`pytsk3`、`pyewf`和`unicodecsv`。此脚本中使用的所有其他库都包含在Python的标准库中。

有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅*[第8章](part0241.html#75QNI0-260f9401d2714cb9ab693c4692308abe)，与取证证据容器食谱一起工作*。因为我们在Python 2.x中开发这些食谱，所以可能会遇到Unicode编码和解码错误。为了解决这个问题，我们使用`unicodecsv`库在本章中编写所有CSV输出。这个第三方模块处理Unicode支持，不像Python 2.x的标准`csv`模块，并且在这里将得到很好的应用。像往常一样，我们可以使用`pip`来安装`unicodecsv`：

```py
pip install unicodecsv==0.14.1
```

除此之外，我们将继续使用从[第8章](https://cdp.packtpub.com/python_digital_forensics_cookbook/wp-admin/post.php?post=260&action=edit#post_218)开发的`pytskutil`模块，以允许与取证获取进行交互。这个模块与我们之前编写的大致相似，只是对一些细微的更改，以更好地适应我们的目的。您可以通过导航到代码包中的实用程序目录来查看代码。

# 如何做...

我们遵循以下基本原则处理预取文件：

1.  扫描以`.pf`扩展名结尾的文件。

1.  通过签名验证消除误报。

1.  解析Windows XP预取文件格式。

1.  在当前工作目录中创建解析结果的电子表格。

# 它是如何工作的...

我们导入了许多库来帮助解析参数、解析日期、解释二进制数据、编写CSV文件以及自定义的`pytskutil`模块。

```py
from __future__ import print_function
import argparse
from datetime import datetime, timedelta
import os
import pytsk3
import pyewf
import struct
import sys
import unicodecsv as csv
from utility.pytskutil import TSKUtil
```

这个配方的命令行处理程序接受两个位置参数，`EVIDENCE_FILE`和`TYPE`，它们代表证据文件的路径和证据文件的类型（即`raw`或`ewf`）。本章中大多数配方只包括这两个位置输入。这些配方的输出将是在当前工作目录中创建的电子表格。这个配方有一个可选参数`d`，它指定要扫描预取文件的路径。默认情况下，这被设置为`/Windows/Prefetch`目录，尽管用户可以选择扫描整个镜像或其他目录。在对证据文件进行一些输入验证后，我们向`main()`函数提供了三个输入，并开始执行脚本：

```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("TYPE", help="Type of Evidence",
                        choices=("raw", "ewf"))
    parser.add_argument("OUTPUT_CSV", help="Path to write output csv")
    parser.add_argument("-d", help="Prefetch directory to scan",
                        default="/WINDOWS/PREFETCH")
    args = parser.parse_args()

    if os.path.exists(args.EVIDENCE_FILE) and \
            os.path.isfile(args.EVIDENCE_FILE):
        main(args.EVIDENCE_FILE, args.TYPE, args.OUTPUT_CSV, args.d)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)
```

在`main()`函数中，我们首先创建`TSKUtil`对象`tsk_util`，它代表`pytsk3`图像对象。有了`TSKUtil`对象，我们可以调用许多辅助函数直接与证据文件进行交互。我们使用`TSKUtil.query_directory()`函数确认指定的目录是否存在。如果存在，我们使用`TSKUtil.recurse_files()`方法来递归遍历指定目录，并识别以`.pf`扩展名结尾的任何文件。该方法返回一个元组列表，其中每个元组包含许多潜在有用的对象，包括`filename`、路径和对象本身。如果找不到这样的文件，则返回`None`。

```py
def main(evidence, image_type, output_csv, path):
    # Create TSK object and query path for prefetch files
    tsk_util = TSKUtil(evidence, image_type)
    prefetch_dir = tsk_util.query_directory(path)
    prefetch_files = None
    if prefetch_dir is not None:
        prefetch_files = tsk_util.recurse_files(
            ".pf", path=path, logic="endswith")
```

如果我们找到与搜索条件匹配的文件，我们会在控制台上打印状态消息，显示找到的文件数量。接下来，我们设置`prefetch_data`列表，用于存储从每个有效文件中解析的预取数据。当我们遍历搜索中的每个命中时，我们提取文件对象（元组的第二个索引）以进行进一步处理。

在我们对文件对象执行任何操作之前，我们使用`check_signature()`方法验证潜在预取文件的文件签名。如果文件与已知的预取文件签名不匹配，则将`None`作为`pf_version`变量返回，阻止对该特定文件进行进一步处理。在我们进一步深入实际处理文件之前，让我们看看这个`check_signature()`方法是如何工作的。

```py
    if prefetch_files is None:
        print("[-] No .pf files found")
        sys.exit(2)

    print("[+] Identified {} potential prefetch files".format(
          len(prefetch_files)))
    prefetch_data = []
    for hit in prefetch_files:
        prefetch_file = hit[2]
        pf_version = check_signature(prefetch_file)
```

`check_signature()`方法以文件对象作为输入，返回预取版本，如果文件不是有效的预取文件，则返回`None`。我们使用`struct`从潜在的预取文件的前8个字节中提取两个小端`32位`整数。第一个整数代表文件版本，而第二个整数是文件的签名。文件签名应为`0x53434341`，其十进制表示为`1,094,927,187`。我们将从文件中提取的值与该数字进行比较，以确定文件签名是否匹配。如果它们匹配，我们将预取版本返回给`main()`函数。预取版本告诉我们我们正在处理哪种类型的预取文件（Windows XP、7、10等）。我们将此值返回以指示如何处理文件，因为不同版本的Windows中预取文件略有不同。现在，回到`main()`函数！

要了解更多关于预取版本和文件格式的信息，请访问[http://www.forensicswiki.org/wiki/Windows_Prefetch_File_Format](http://www.forensicswiki.org/wiki/Windows_Prefetch_File_Format)。

```py
def check_signature(prefetch_file):
    version, signature = struct.unpack(
        "<2i", prefetch_file.read_random(0, 8))

    if signature == 1094927187:
        return version
    else:
        return None
```

在`main()`函数中，我们检查`pf_version`变量是否不是`None`，这表明它已成功验证。随后，我们将文件名提取到`pf_name`变量中，该变量存储在元组的零索引处。接下来，我们检查我们正在处理哪个版本的预取文件。预取版本及其相关操作系统的详细信息可以在这里查看：

| **预取版本** | **Windows桌面操作系统** |
| 17 | Windows XP |
| 23 | Windows Vista，Windows 7 |
| 26 | Windows 8.1 |
| 30 | Windows 10 |

这个教程只开发了处理Windows XP预取文件的方法，使用的是之前引用的取证wiki页面上记录的文件格式。然而，有占位符可以添加逻辑来支持其他预取格式。它们在很大程度上是相似的，除了Windows 10，可以通过遵循用于Windows XP的相同基本方法来解析。Windows 10预取文件是MAM压缩的，必须先解压缩才能处理--除此之外，它们可以以类似的方式处理。对于版本17（Windows XP格式），我们调用解析函数，提供TSK文件对象和预取文件的名称：

```py
        if pf_version is None:
            continue

        pf_name = hit[0]
        if pf_version == 17:
            parsed_data = parse_pf_17(prefetch_file, pf_name)
            parsed_data.append(os.path.join(path, hit[1].lstrip("//")))
            prefetch_data.append(parsed_data)
```

我们开始处理Windows XP预取文件，将文件本身的`create`和`modify`时间戳存储到本地变量中。这些`Unix`时间戳使用我们之前使用过的`convertUnix()`方法进行转换。除了`Unix`时间戳，我们还遇到了嵌入在预取文件中的`FILETIME`时间戳。在继续讨论`main()`方法之前，让我们简要看一下这些函数：

```py
def parse_pf_17(prefetch_file, pf_name):
    # Parse Windows XP, 2003 Prefetch File
    create = convert_unix(prefetch_file.info.meta.crtime)
    modify = convert_unix(prefetch_file.info.meta.mtime)
```

这两个函数都依赖于`datetime`模块，以适当地将时间戳转换为人类可读的格式。这两个函数都检查提供的时间戳字符串是否等于`"0"`，如果是，则返回空字符串。否则，对于`convert_unix()`方法，我们使用`utcfromtimestamp()`方法将`Unix`时间戳转换为`datetime`对象并返回。对于`FILETIME`时间戳，我们添加自1601年1月1日以来经过的100纳秒数量，并返回结果的`datetime`对象。完成了我们与时间的短暂交往，让我们回到`main()`函数。

```py
def convert_unix(ts):
    if int(ts) == 0:
        return ""
    return datetime.utcfromtimestamp(ts)

def convert_filetime(ts):
    if int(ts) == 0:
        return ""
    return datetime(1601, 1, 1) + timedelta(microseconds=ts / 10)
```

现在我们已经提取了文件元数据，我们开始使用`struct`来提取预取文件中嵌入的数据。我们使用`pytsk3.read_random()`方法和`struct`从文件中读取`136`字节，并将这些数据解包到Python变量中。具体来说，在这`136`字节中，我们提取了五个`32位`整数（`i`），一个`64位`整数（`q`），和一个60字符的字符串（`s`）。在上述句子中的括号中是与这些数据类型相关的`struct`格式字符。这也可以在`struct`格式字符串`"<i60s32x3iq16xi"`中看到，其中在`struct`格式字符之前的数字告诉`struct`有多少个（例如，`60s`告诉`struct`将下一个`60`字节解释为字符串）。同样，`"x"` `struct`格式字符是一个空值。如果`struct`接收到`136`字节要读取，它也必须接收到格式字符来解释每个这`136`字节。因此，我们必须提供这些空值，以确保我们适当地解释我们正在读取的数据，并确保我们正在适当的偏移量上解释值。字符串开头的`"<"`字符确保所有值都被解释为小端。

是的，可能有点多，但我们现在可能都对`struct`有了更好的理解。在`struct`解释数据后，它以解包的数据类型元组的顺序返回。我们将这些分配给一系列本地变量，包括预取文件大小，应用程序名称，最后执行的`FILETIME`和执行计数。我们提取的应用程序的`name`变量，即我们提取的60个字符的字符串，需要进行UTF-16解码，并且我们需要删除填充字符串的所有`x00`值。请注意，我们提取的值之一，`vol_info`，是存储在预取文件中卷信息的指针。我们接下来提取这些信息：

```py
    pf_size, name, vol_info, vol_entries, vol_size, filetime, \
        count = struct.unpack("<i60s32x3iq16xi",
                              prefetch_file.read_random(12, 136))

    name = name.decode("utf-16", "ignore").strip("/x00").split("/x00")[0]
```

让我们看一个更简单的例子，使用`struct`。我们从`vol_info`指针开始读取`20`字节，并提取三个`32位`整数和一个`64位`整数。这些是卷名偏移和长度，卷序列号和卷创建日期。大多数取证程序将卷序列号显示为由破折号分隔的两个四字符十六进制值。我们通过将整数转换为十六进制并删除前置的`"0x"`值来做到这一点，以隔离出八字符十六进制值。接下来，我们使用字符串切片和连接在卷序列号的中间添加一个破折号。

最后，我们使用提取的卷名偏移和长度来提取卷名。我们使用字符串格式化将卷名长度插入`struct`格式字符串中。我们必须将长度乘以二来提取完整的字符串。与应用程序名称类似，我们必须将字符串解码为UTF-16并删除任何存在的`"/x00"`值。我们将从预取文件中提取的元素附加到列表中。请注意，我们在这样做时执行了一些最后一刻的操作，包括转换两个`FILETIME`时间戳并将预取路径与文件名结合在一起。请注意，如果我们不从`filename`中删除前置的`"**/**"`字符，则`os.path.join()`方法将无法正确组合这两个字符串。因此，我们使用`lstrip()`将其从字符串的开头删除：

```py
    vol_name_offset, vol_name_length, vol_create, \
        vol_serial = struct.unpack("<2iqi",
                                   prefetch_file.read_random(vol_info, 20))

    vol_serial = hex(vol_serial).lstrip("0x")
    vol_serial = vol_serial[:4] + "-" + vol_serial[4:]

    vol_name = struct.unpack(
        "<{}s".format(2 * vol_name_length),
        prefetch_file.read_random(vol_info + vol_name_offset,
                                  vol_name_length * 2)
    )[0]

    vol_name = vol_name.decode("utf-16", "ignore").strip("/x00").split(
        "/x00")[0]

    return [
        pf_name, name, pf_size, create,
        modify, convert_filetime(filetime), count, vol_name,
        convert_filetime(vol_create), vol_serial
    ]
```

正如我们在本教程开始时讨论的那样，我们目前仅支持Windows XP格式的预取文件。我们已留下占位符以支持其他格式类型。但是，当前，如果遇到这些格式，将在控制台上打印不支持的消息，然后我们继续到下一个预取文件：

```py
        elif pf_version == 23:
            print("[-] Windows Vista / 7 PF file {} -- unsupported".format(
                pf_name))
            continue
        elif pf_version == 26:
            print("[-] Windows 8 PF file {} -- unsupported".format(
                pf_name))
            continue
        elif pf_version == 30:
            print("[-] Windows 10 PF file {} -- unsupported".format(
                pf_name))
            continue
```

回想一下本教程开始时我们如何检查`pf_version`变量是否为`None`。如果是这种情况，预取文件将无法通过签名验证，因此我们会打印一条相应的消息，然后继续到下一个文件。一旦我们完成处理所有预取文件，我们将包含解析数据的列表发送到`write_output()`方法：

```py
        else:
            print("[-] Signature mismatch - Name: {}\nPath: {}".format(
                hit[0], hit[1]))
            continue

    write_output(prefetch_data, output_csv)
```

`write_output()` 方法接受我们创建的数据列表，并将该数据写入CSV文件。我们使用`os.getcwd()`方法来识别当前工作目录，在那里我们写入CSV文件。在向控制台打印状态消息后，我们创建我们的CSV文件，写入我们列的名称，然后使用`writerows()`方法在数据列表中写入所有解析的预取数据列表。

```py
def write_output(data, output_csv):
    print("[+] Writing csv report")
    with open(output_csv, "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            "File Name", "Prefetch Name", "File Size (bytes)",
            "File Create Date (UTC)", "File Modify Date (UTC)",
            "Prefetch Last Execution Date (UTC)",
            "Prefetch Execution Count", "Volume", "Volume Create Date",
            "Volume Serial", "File Path"
        ])
        writer.writerows(data)
```

当我们运行这个脚本时，我们会生成一个包含以下列的CSV文档：

![](../images/00107.jpeg)

向左滚动，我们可以看到相同条目的以下列（由于其大小，文件路径列未显示）。

![](../images/00108.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一个或多个建议：

+   添加对其他Windows预取文件格式的支持。从Windows 10开始，预取文件现在具有MAM压缩，必须在使用`struct`解析数据之前首先进行解压缩

+   查看`libscca` ([https://github.com/libyal/libscca](https://github.com/libyal/libscca))库及其Python绑定`pyscca`，该库是用于处理预取文件的

# 一系列幸运的事件

示例难度：困难

Python版本：2.7

操作系统：Linux

事件日志，如果配置适当，包含了在任何网络调查中都有用的大量信息。这些日志保留了历史用户活动信息，如登录、RDP访问、Microsoft Office文件访问、系统更改和特定应用程序事件。在这个示例中，我们使用`pyevt`和`pyevtx`库来处理传统和当前的Windows事件日志格式。

# 入门

这个示例需要安装五个第三方模块才能运行：`pytsk3`，`pyewf`，`pyevt`，`pyevtx`和`unicodecsv`。有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅[第8章](part0241.html#75QNI0-260f9401d2714cb9ab693c4692308abe)，*使用取证证据容器* *示例*。同样，有关安装`unicodecsv`的详细信息，请参阅*开始*部分中的*解析预取文件*示例。此脚本中使用的所有其他库都包含在Python的标准库中。在安装大多数`libyal`库的Python绑定时，它们遵循非常相似的路径。

转到GitHub存储库，并下载每个库的所需版本。这个示例是使用`pyevt`和`pyevtx`库的`libevt-alpha-20170120`和`libevtx-alpha-20170122`版本开发的。接下来，一旦提取了发布的内容，打开终端并导航到提取的目录，然后对每个发布执行以下命令：

```py
./synclibs.sh
./autogen.sh
sudo python setup.py install 
```

要了解更多关于`pyevt`库，请访问[https://github.com/libyal/libevt](https://github.com/libyal/libevt)。

要了解更多关于`pyevtx`库，请访问[https://github.com/libyal/libevtx](https://github.com/libyal/libevtx)。

最后，我们可以通过打开Python解释器，导入`pyevt`和`pyevtx`，并运行它们各自的`get_version()`方法来检查库的安装情况，以确保我们有正确的发布版本。

# 如何做...

我们使用以下基本步骤提取事件日志：

1.  搜索与输入参数匹配的所有事件日志。

1.  使用文件签名验证消除误报。

1.  使用适当的库处理找到的每个事件日志。

1.  将所有发现的事件输出到当前工作目录的电子表格中。

# 它是如何工作的...

我们导入了许多库来帮助解析参数、编写CSV、处理事件日志和自定义的`pytskutil`模块。

```py
from __future__ import print_function
import argparse
import unicodecsv as csv
import os
import pytsk3
import pyewf
import pyevt
import pyevtx
import sys
from utility.pytskutil import TSKUtil
```

这个示例的命令行处理程序接受三个位置参数，`EVIDENCE_FILE`，`TYPE`和`LOG_NAME`，分别表示证据文件的路径，证据文件的类型和要处理的事件日志的名称。此外，用户可以使用`"d"`开关指定要扫描的镜像内目录，并使用`"f"`开关启用模糊搜索。如果用户没有提供要扫描的目录，脚本将默认为`"/Windows/System32/winevt"`目录。在比较文件名时，模糊搜索将检查提供的`LOG_NAME`是否是`filename`的子字符串，而不是等于文件名。这种能力允许用户搜索非常特定的事件日志或任何带有`.evt`或`.evtx`扩展名的文件，以及两者之间的任何内容。在执行输入验证检查后，我们将这五个参数传递给`main()`函数：

```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("TYPE", help="Type of Evidence",
                        choices=("raw", "ewf"))
    parser.add_argument("LOG_NAME",
                        help="Event Log Name (SecEvent.Evt, SysEvent.Evt, "
                             "etc.)")
    parser.add_argument("-d", help="Event log directory to scan",
                        default="/WINDOWS/SYSTEM32/WINEVT")
    parser.add_argument("-f", help="Enable fuzzy search for either evt or"
                        " evtx extension", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.EVIDENCE_FILE) and \
            os.path.isfile(args.EVIDENCE_FILE):
        main(args.EVIDENCE_FILE, args.TYPE, args.LOG_NAME, args.d, args.f)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)
```

在`main()`函数中，我们创建了我们的`TSKUtil`对象，我们将与其交互以查询用户提供的路径是否存在。如果路径存在且不为`None`，我们然后检查是否启用了模糊搜索。无论如何，我们都调用相同的`recurse_files()`函数，并将其传递要搜索的日志和要扫描的目录。如果启用了模糊搜索，我们通过将逻辑设置为`"equal"`来向`recurse_files()`方法提供一个额外的可选参数。如果不指定此可选参数，函数将检查日志是否是给定文件的子字符串，而不是精确匹配。我们将任何结果命中存储在`event_log`变量中。

```py
def main(evidence, image_type, log, win_event, fuzzy):
    # Create TSK object and query event log directory for Windows XP
    tsk_util = TSKUtil(evidence, image_type)
    event_dir = tsk_util.query_directory(win_event)
    if event_dir is not None:
        if fuzzy is True:
            event_log = tsk_util.recurse_files(log, path=win_event)
        else:
            event_log = tsk_util.recurse_files(
                log, path=win_event, logic="equal")
```

如果我们确实有日志的命中，我们设置`event_data`列表，它将保存解析后的事件日志数据。接下来，我们开始迭代每个发现的事件日志。对于每个命中，我们提取其文件对象，这是`recurse_files()`方法返回的元组的第二个索引，并将其发送到`write_file()`方法中，暂时写入主机文件系统。这将是以后的常见做法，以便这些第三方库可以更轻松地与文件交互。

```py
        if event_log is not None:
            event_data = []
            for hit in event_log:
                event_file = hit[2]
                temp_evt = write_file(event_file)
```

`write_file()`方法相当简单。它所做的就是以`"w"`模式打开一个Python`File`对象，并使用相同的名称将输入文件的整个内容写入当前工作目录。我们将此输出文件的名称返回给`main()`方法。

```py
def write_file(event_file):
    with open(event_file.info.name.name, "w") as outfile:
        outfile.write(event_file.read_random(0, event_file.info.meta.size))
    return event_file.info.name.name
```

在`main()`方法中，我们使用`pyevt.check_file_signature()`方法来检查我们刚刚缓存的文件是否是有效的`evt`文件。如果是，我们使用`pyevt.open()`方法来创建我们的`evt`对象。在控制台打印状态消息后，我们迭代事件日志中的所有记录。记录可能有许多字符串，因此我们遍历这些字符串，并确保它们被添加到`strings`变量中。然后，我们将一些事件日志属性附加到`event_data`列表中，包括计算机名称、SID、创建和写入时间、类别、来源名称、事件ID、事件类型、字符串和文件路径。

您可能会注意到空字符串添加为列表中倒数第二个项目。由于在`.evtx`文件中找不到等效的对应项，因此需要这个空字符串，以保持输出电子表格的正确间距，因为它设计用于容纳`.evt`和`.evtx`结果。这就是我们处理传统事件日志格式所需做的全部。现在让我们转向日志文件是`.evtx`文件的情况。

```py
                if pyevt.check_file_signature(temp_evt):
                    evt_log = pyevt.open(temp_evt)
                    print("[+] Identified {} records in {}".format(
                        evt_log.number_of_records, temp_evt))
                    for i, record in enumerate(evt_log.records):
                        strings = ""
                        for s in record.strings:
                            if s is not None:
                                strings += s + "\n"

                        event_data.append([
                            i, hit[0], record.computer_name,
                            record.user_security_identifier,
                            record.creation_time, record.written_time,
                            record.event_category, record.source_name,
                            record.event_identifier, record.event_type,
                            strings, "",
                            os.path.join(win_event, hit[1].lstrip("//"))
                        ])
```

值得庆幸的是，`pyevt`和`pyevtx`库的处理方式相似。我们首先使用`pyevtx.check_file_signature()`方法验证日志搜索命中的文件签名。与其`pyevt`对应项一样，该方法根据文件签名检查的结果返回布尔值`True`或`False`。如果文件的签名检查通过，我们使用`pyevtx.open()`方法创建一个`evtx`对象，在控制台写入状态消息，并开始迭代事件日志中的记录。

在将所有字符串存储到`strings`变量后，我们将一些事件日志记录属性附加到事件日志列表中。这些属性包括计算机名称、SID、写入时间、事件级别、来源、事件ID、字符串、任何XML字符串和事件日志路径。请注意，有许多空字符串，这些空字符串用于保持间距，并填补`.evt`等效项不存在的空白。例如，在传统的`.evt`日志中看不到`creation_time`时间戳，因此用空字符串替换它。

```py
                elif pyevtx.check_file_signature(temp_evt):
                    evtx_log = pyevtx.open(temp_evt)
                    print("[+] Identified {} records in {}".format(
                          evtx_log.number_of_records, temp_evt))
                    for i, record in enumerate(evtx_log.records):
                        strings = ""
                        for s in record.strings:
                            if s is not None:
                                strings += s + "\n"

                        event_data.append([
                            i, hit[0], record.computer_name,
                            record.user_security_identifier, "",
                            record.written_time, record.event_level,
                            record.source_name, record.event_identifier,
                            "", strings, record.xml_string,
                            os.path.join(win_event, hit[1].lstrip("//"))
                        ])
```

如果从搜索中获得的日志命中无法验证为`.evt`或`.evtx`日志，则我们会向控制台打印状态消息，使用`os.remove()`方法删除缓存文件，并继续处理下一个命中。请注意，我们只会在无法验证时删除缓存的事件日志。否则，我们会将它们留在当前工作目录中，以便用户可以使用其他工具进一步处理。在处理完所有事件日志后，我们使用`write_output()`方法将解析的列表写入CSV。剩下的两个`else`语句处理了两种情况：要么搜索中没有事件日志命中，要么我们扫描的目录在证据文件中不存在。

```py
                else:
                    print("[-] {} not a valid event log. Removing temp "
                          "file...".format(temp_evt))
                    os.remove(temp_evt)
                    continue
            write_output(event_data)
        else:
            print("[-] {} Event log not found in {} directory".format(
                log, win_event))
            sys.exit(3)

    else:
        print("[-] Win XP Event Log Directory {} not found".format(
            win_event))
        sys.exit(2)
```

`write_output()`方法的行为与前一个示例中讨论的类似。我们在当前工作目录中创建一个CSV，并使用`writerows()`方法将所有解析的结果写入其中。

```py
def write_output(data):
    output_name = "parsed_event_logs.csv"
    print("[+] Writing {} to current working directory: {}".format(
          output_name, os.getcwd()))
    with open(output_name, "wb") as outfile:
        writer = csv.writer(outfile)

        writer.writerow([
            "Index", "File name", "Computer Name", "SID",
            "Event Create Date", "Event Written Date",
            "Event Category/Level", "Event Source", "Event ID",
            "Event Type", "Data", "XML Data", "File Path"
        ])

        writer.writerows(data)
```

以下截图显示了指定日志文件中事件的基本信息：

![](../images/00109.jpeg)

第二个截图显示了这些行的额外列：

![](../images/00110.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一个或多个建议：

+   启用松散文件支持

+   添加事件ID参数以选择性地提取与给定事件ID匹配的事件

# 索引互联网历史

示例难度：中等

Python版本：2.7

操作系统：Linux

在调查过程中，互联网历史记录可能非常有价值。这些记录可以揭示用户的思维过程，并为系统上发生的其他用户活动提供背景。微软一直在努力让用户将Internet Explorer作为他们的首选浏览器。因此，在Internet Explorer使用的`index.dat`文件中经常可以看到互联网历史信息。在这个示例中，我们在证据文件中搜索这些`index.dat`文件，并尝试使用`pymsiecf`处理它们。

# 入门

这个示例需要安装四个第三方模块才能运行：`pytsk3`、`pyewf`、`pymsiecf`和`unicodecsv`。有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅[第8章](part0241.html#75QNI0-260f9401d2714cb9ab693c4692308abe)，*使用取证证据容器* *示例*。同样，有关安装`unicodecsv`的详细信息，请参阅*解析预取文件*示例中的*入门*部分。此脚本中使用的所有其他库都包含在Python的标准库中。

转到GitHub存储库并下载所需版本的`pymsiecf`库。这个示例是使用`libmsiecf-alpha-20170116`版本开发的。提取版本的内容后，打开终端并转到提取的目录，执行以下命令：

```py
./synclibs.sh
./autogen.sh
sudo python setup.py install 
```

要了解更多关于`pymsiecf`库的信息，请访问[https://github.com/libyal/libmsiecf](https://github.com/libyal/libmsiecf)。

最后，我们可以通过打开Python解释器，导入`pymsiecf`，并运行`gpymsiecf.get_version()`方法来检查我们的库是否安装了正确的版本。

# 如何做...

我们按照以下步骤提取Internet Explorer历史记录：

1.  查找并验证图像中的所有`index.dat`文件。

1.  处理互联网历史文件。

1.  将结果输出到当前工作目录的电子表格中。

# 工作原理...

我们导入了许多库来帮助解析参数、编写CSV、处理`index.dat`文件和自定义的`pytskutil`模块：

```py
from __future__ import print_function
import argparse
from datetime import datetime, timedelta
import os
import pytsk3
import pyewf
import pymsiecf
import sys
import unicodecsv as csv
from utility.pytskutil import TSKUtil
```

这个配方的命令行处理程序接受两个位置参数，`EVIDENCE_FILE`和`TYPE`，分别代表证据文件的路径和证据文件的类型。与之前的配方类似，可以提供可选的`d`开关来指定要扫描的目录。否则，配方将从`"/Users"`目录开始扫描。在执行输入验证检查后，我们将这三个参数传递给`main()`函数。

```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("TYPE", help="Type of Evidence",
                        choices=("raw", "ewf"))
    parser.add_argument("-d", help="Index.dat directory to scan",
                        default="/USERS")
    args = parser.parse_args()

    if os.path.exists(args.EVIDENCE_FILE) and os.path.isfile(
            args.EVIDENCE_FILE):
        main(args.EVIDENCE_FILE, args.TYPE, args.d)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)
```

`main()`函数首先创建了一个现在熟悉的`TSKUtil`对象，并扫描指定的目录以确认它是否存在于证据文件中。如果存在，我们会从指定的目录递归扫描任何文件，这些文件等于字符串`"index.dat"`。这些文件以元组的形式从`recurse_files()`方法返回，其中每个元组代表符合搜索条件的特定文件。

```py
def main(evidence, image_type, path):
    # Create TSK object and query for Internet Explorer index.dat files
    tsk_util = TSKUtil(evidence, image_type)
    index_dir = tsk_util.query_directory(path)
    if index_dir is not None:
        index_files = tsk_util.recurse_files("index.dat", path=path,
                                             logic="equal")
```

如果我们找到了潜在的`index.dat`文件要处理，我们会在控制台打印状态消息，并设置一个列表来保留这些文件解析结果。我们开始遍历命中的文件；提取元组的第二个索引，即`index.dat`文件对象；并使用`write_file()`方法将其写入主机文件系统：

```py
        if index_files is not None:
            print("[+] Identified {} potential index.dat files".format(
                  len(index_files)))
            index_data = []
            for hit in index_files:
                index_file = hit[2]
                temp_index = write_file(index_file)
```

`write_file()`方法在之前的配方中有更详细的讨论。它与我们之前讨论的内容相同。本质上，这个函数将证据容器中的`index.dat`文件复制到当前工作目录，以便第三方模块进行处理。一旦创建了这个输出，我们将输出文件的名称，这种情况下总是`index.dat`，返回给`main()`函数：

```py
def write_file(index_file):
    with open(index_file.info.name.name, "w") as outfile:
        outfile.write(index_file.read_random(0, index_file.info.meta.size))
    return index_file.info.name.name
```

与之前的`libyal`库类似，`pymsiecf`模块有一个内置方法`check_file_signature()`，我们用它来确定搜索命中是否是有效的`index.dat`文件。如果是，我们使用`pymsiecf.open()`方法创建一个可以用库操作的对象。我们在控制台打印状态消息，并开始遍历`.dat`文件中的项目。我们首先尝试访问`data`属性。这包含了我们感兴趣的大部分信息，但并不总是可用。然而，如果属性存在且不是`None`，我们会移除追加的`"\x00"`值：

```py
                if pymsiecf.check_file_signature(temp_index):
                    index_dat = pymsiecf.open(temp_index)
                    print("[+] Identified {} records in {}".format(
                        index_dat.number_of_items, temp_index))
                    for i, record in enumerate(index_dat.items):
                        try:
                            data = record.data
                            if data is not None:
                                data = data.rstrip("\x00")
```

正如之前提到的，有些情况下可能没有`data`属性。`pymsiecf.redirected`和`pymsiecf.leak`对象就是两个例子。然而，这些对象仍然可能包含相关的数据。因此，在异常情况下，我们检查记录是否是这两个对象中的一个实例，并将可用的数据追加到我们解析的`index.dat`数据列表中。在我们将这些数据追加到列表中或者记录不是这两种类型的实例时，我们继续处理下一个`record`，除非出现`AttributeError`：

```py
                        except AttributeError:
                            if isinstance(record, pymsiecf.redirected):
                                index_data.append([
                                    i, temp_index, "", "", "", "", "",
                                    record.location, "", "", record.offset,
                                    os.path.join(path, hit[1].lstrip("//"))
                                ])

                            elif isinstance(record, pymsiecf.leak):
                                index_data.append([
                                    i, temp_index, record.filename, "",
                                    "", "", "", "", "", "", record.offset,
                                    os.path.join(path, hit[1].lstrip("//"))
                                ])

                            continue
```

在大多数情况下，`data`属性是存在的，我们可以从记录中提取许多潜在相关的信息点。这包括文件名、类型、若干时间戳、位置、命中次数和数据本身。需要明确的是，`data`属性通常是系统上浏览活动的记录的某种URL：

```py
                        index_data.append([
                            i, temp_index, record.filename,
                            record.type, record.primary_time,
                            record.secondary_time,
                            record.last_checked_time, record.location,
                            record.number_of_hits, data, record.offset,
                            os.path.join(path, hit[1].lstrip("//"))
                        ])
```

如果无法验证`index.dat`文件，我们将删除有问题的缓存文件，并继续迭代所有其他搜索结果。同样，这一次我们选择删除`index.dat`缓存文件，无论它是否有效，因为我们完成处理最后一个后。因为所有这些文件都将具有相同的名称，它们在处理过程中将相互覆盖。因此，在当前工作目录中仅保留一个文件是没有意义的。但是，如果需要，可以做一些更复杂的事情，并将每个文件缓存到主机文件系统，同时保留其路径。剩下的两个`else`语句是用于在取证文件中找不到`index.dat`文件和要扫描的目录不存在的情况：

```py
                else:
                    print("[-] {} not a valid index.dat file. Removing "
                          "temp file..".format(temp_index))
                    os.remove("index.dat")
                    continue

            os.remove("index.dat")
            write_output(index_data)
        else:
            print("[-] Index.dat files not found in {} directory".format(
                path))
            sys.exit(3)

    else:
        print("[-] Directory {} not found".format(win_event))
        sys.exit(2)
```

`write_output()`方法的行为类似于前几个食谱中同名方法的行为。我们创建一个略微描述性的输出名称，在当前工作目录中创建输出CSV，然后将标题和数据写入文件。通过这样，我们已经完成了这个食谱，现在可以将处理过的`index.dat`文件添加到我们的工具箱中：

```py
def write_output(data):
    output_name = "Internet_Indexdat_Summary_Report.csv"
    print("[+] Writing {} with {} parsed index.dat files to current "
          "working directory: {}".format(output_name, len(data),
                                         os.getcwd()))
    with open(output_name, "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Index", "File Name", "Record Name",
                         "Record Type", "Primary Date", "Secondary Date",
                         "Last Checked Date", "Location", "No. of Hits",
                         "Record Data", "Record Offset", "File Path"])
        writer.writerows(data)
```

当我们执行脚本时，可以查看包含数据的电子表格，如下所示：

![](../images/00111.jpeg)

虽然这份报告有很多列，但以下截图显示了同一行的一些额外列的片段：

![](../images/00112.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一个或多个建议：

+   创建可用数据的摘要指标（访问最受欢迎和最不受欢迎的域，互联网使用的平均时间范围等）

# 前任的影子

食谱难度：困难

Python版本：2.7

操作系统：Linux

卷影副本可以包含来自活动系统上不再存在的文件的数据。这可以为检查人员提供一些关于系统随时间如何变化以及计算机上曾经存在哪些文件的历史信息。在这个食谱中，我们将使用`pvyshadow`库来枚举和访问取证图像中存在的任何卷影副本。

# 入门

这个食谱需要安装五个第三方模块才能运行：`pytsk3`、`pyewf`、`pyvshadow`、`unicodecsv`和`vss`。有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅[第8章](part0241.html#75QNI0-260f9401d2714cb9ab693c4692308abe)，*使用取证证据容器* *食谱*。同样，有关安装`unicodecsv`的详细信息，请参阅*解析预取文件*食谱中的*入门*部分。在这个脚本中使用的所有其他库都包含在Python的标准库中。

导航到GitHub存储库并下载所需的`pyvshadow`库的发布版本。这个食谱是使用`libvshadow-alpha-20170715`版本开发的。一旦释放的内容被提取出来，打开一个终端，导航到提取的目录，并执行以下命令：

```py
./synclibs.sh
./autogen.sh
sudo python setup.py install 
```

在[https://github.com/libyal/libvshadow](https://github.com/libyal/libvshadow)了解更多关于`pyvshadow`库的信息。

`pyvshadow`模块仅设计用于处理原始图像，并不支持其他取证图像类型。正如*David Cowen*在[http://www.hecfblog.com/2015/05/automating-dfir-how-to-series-on_25.html](http://www.hecfblog.com/2015/05/automating-dfir-how-to-series-on_25.html)的博客文章中所指出的，plaso项目已经创建了一个辅助库`vss`，可以与`pyvshadow`集成，我们将在这里使用。`vss`代码可以在同一篇博客文章中找到。

最后，我们可以通过打开Python解释器，导入`pyvshadow`，并运行`pyvshadow.get_version()`方法来检查我们是否有正确的发布版本。

# 如何做...

我们使用以下步骤访问卷影副本：

1.  访问原始图像的卷并识别所有NTFS分区。

1.  枚举在有效的NTFS分区上找到的每个卷影副本。

1.  创建快照内数据的文件列表。

# 工作原理...

我们导入了许多库来帮助解析参数、日期解析、编写CSV、处理卷影副本以及自定义的`pytskutil`模块。

```py
from __future__ import print_function
import argparse
from datetime import datetime, timedelta
import os
import pytsk3
import pyewf
import pyvshadow
import sys
import unicodecsv as csv
from utility import vss
from utility.pytskutil import TSKUtil
from utility import pytskutil
```

这个脚本的命令行处理程序接受两个位置参数：`EVIDENCE_FILE`和`OUTPUT_CSV`。它们分别代表证据文件的路径和输出电子表格的文件路径。请注意，这里没有证据类型参数。这个脚本只支持原始镜像文件，不支持`E01s`。要准备一个EWF镜像以便与脚本一起使用，您可以将其转换为原始镜像，或者使用与`libewf`相关的`ewfmount`工具进行挂载，并将挂载点作为输入。

```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("OUTPUT_CSV",
                        help="Output CSV with VSS file listing")
    args = parser.parse_args()
```

解析输入参数后，我们将`OUTPUT_CSV`输入中的目录与文件分开，并确认它存在或者如果不存在则创建它。我们还在将两个位置参数传递给`main()`函数之前，验证输入文件路径的存在。

```py
    directory = os.path.dirname(args.OUTPUT_CSV)
    if not os.path.exists(directory) and directory != "":
        os.makedirs(directory)

    if os.path.exists(args.EVIDENCE_FILE) and \
            os.path.isfile(args.EVIDENCE_FILE):
        main(args.EVIDENCE_FILE, args.OUTPUT_CSV)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)

```

`main()`函数调用了`TSKUtil`对象中的一些新函数，我们还没有探索过。创建了`TSKUtil`对象后，我们使用`return_vol()`方法提取它的卷。与证据文件的卷交互，正如我们在之前的示例中看到的那样，是在我们可以与文件系统交互之前必不可少的步骤之一。然而，这个过程以前在必要时已经在后台执行过。然而，这一次，我们需要访问`pytsk3`卷对象，以便遍历每个分区以识别NTFS文件系统。`detect_ntfs()`方法返回一个布尔值，指示特定分区是否有NTFS文件系统。

对于我们遇到的每个NTFS文件系统，我们将证据文件、发现的NTFS分区的偏移量和输出CSV文件传递给`explore_vss()`函数。如果卷对象是`None`，我们会在控制台打印状态消息，提醒用户证据文件必须是物理设备镜像，而不仅仅是特定分区的逻辑镜像。

```py
def main(evidence, output):
    # Create TSK object and query path for prefetch files
    tsk_util = TSKUtil(evidence, "raw")
    img_vol = tsk_util.return_vol()
    if img_vol is not None:
        for part in img_vol:
            if tsk_util.detect_ntfs(img_vol, part):
                print("Exploring NTFS Partition for VSS")
                explore_vss(evidence, part.start * img_vol.info.block_size,
                            output)
    else:
        print("[-] Must be a physical preservation to be compatible "
              "with this script")
        sys.exit(2)
```

`explore_vss()`方法首先创建一个`pyvshadow.volume()`对象。我们使用这个卷来打开从`vss.VShadowVolume()`方法创建的`vss_handle`对象。`vss.VShadowVolume()`方法接受证据文件和分区偏移值，并公开一个类似卷的对象，与`pyvshadow`库兼容，该库不原生支持物理磁盘镜像。`GetVssStoreCount()`函数返回在证据中找到的卷影副本的数量。

如果有卷影副本，我们使用`pyvshadow vss_volume`打开我们的`vss_handle`对象，并实例化一个列表来保存我们的数据。我们创建一个`for`循环来遍历每个存在的卷影副本，并执行相同的一系列步骤。首先，我们使用`pyvshadow get_store()`方法访问感兴趣的特定卷影副本。然后，我们使用`vss`辅助库`VShadowImgInfo`来创建一个`pytsk3`图像句柄。最后，我们将图像句柄传递给`openVSSFS()`方法，并将返回的数据追加到我们的列表中。`openVSSFS()`方法使用与之前讨论过的类似方法来创建一个`pytsk3`文件系统对象，然后递归遍历当前目录以返回一个活动文件列表。在我们对所有卷影副本执行了这些步骤之后，我们将数据和输出CSV文件路径传递给我们的`csvWriter()`方法。

```py
def explore_vss(evidence, part_offset, output):
    vss_volume = pyvshadow.volume()
    vss_handle = vss.VShadowVolume(evidence, part_offset)
    vss_count = vss.GetVssStoreCount(evidence, part_offset)
    if vss_count > 0:
        vss_volume.open_file_object(vss_handle)
        vss_data = []
        for x in range(vss_count):
            print("Gathering data for VSC {} of {}".format(x, vss_count))
            vss_store = vss_volume.get_store(x)
            image = vss.VShadowImgInfo(vss_store)
            vss_data.append(pytskutil.openVSSFS(image, x))

        write_csv(vss_data, output)
```

`write_csv()`方法的功能与您期望的一样。它首先检查是否有要写入的数据。如果没有，它会在退出脚本之前在控制台上打印状态消息。或者，它使用用户提供的输入创建一个CSV文件，写入电子表格标题，并遍历每个列表，为每个卷影复制调用`writerows()`。为了防止标题多次出现在CSV输出中，我们将检查CSV是否已经存在，并添加新数据进行审查。这使我们能够在处理每个卷影副本后转储信息。

```py
def write_csv(data, output):
    if data == []:
        print("[-] No output results to write")
        sys.exit(3)

    print("[+] Writing output to {}".format(output))
    if os.path.exists(output):
        append = True
    with open(output, "ab") as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = ["VSS", "File", "File Ext", "File Type", "Create Date",
                   "Modify Date", "Change Date", "Size", "File Path"]
        if not append:
            csv_writer.writerow(headers)
        for result_list in data:
            csv_writer.writerows(result_list)
```

运行此脚本后，我们可以查看每个卷影副本中找到的文件，并了解每个项目的元数据：

![](../images/00113.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一个或多个建议：

+   添加对逻辑获取和其他取证获取类型的支持

+   添加支持以处理先前编写的配方中发现的快照中的工件

# 解剖SRUM数据库

配方难度：困难

Python版本：2.7

操作系统：Linux

随着流行操作系统的主要发布，网络社区中的每个人都对潜在的新工件和现有工件的变化感到兴奋（或担忧）。随着Windows 10的出现，我们看到了一些变化（例如对预取文件的MAM压缩）以及新的工件。其中一个工件是**系统资源使用监视器**（**SRUM**），它可以保留应用程序的执行和网络活动。这包括诸如特定应用程序建立连接的时间以及此应用程序发送和接收的字节数等信息。显然，在许多不同的情况下，这可能非常有用。想象一下，在最后一天使用Dropbox桌面应用程序上传了许多千兆字节数据的不满员工手头有这些信息。

在这个配方中，我们利用`pyesedb`库从数据库中提取数据。我们还将实现逻辑来解释这些数据为适当的类型。完成这些后，我们将能够查看存储在Windows 10机器上的`SRUM.dat`文件中的历史应用程序信息。

要了解有关SRUM数据库的更多信息，请访问[https://www.sans.org/summit-archives/file/summit-archive-1492184583.pdf](https://www.sans.org/summit-archives/file/summit-archive-1492184583.pdf)。

# 入门

此配方需要安装四个第三方模块才能运行：`pytsk3`，`pyewf`，`pyesedb`和`unicodecsv`。有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅[第8章](part0241.html#75QNI0-260f9401d2714cb9ab693c4692308abe)，*使用取证证据容器* *配方*。同样，有关安装`unicodecsv`的详细信息，请参阅*解析预取文件*配方中的*入门*部分。此脚本中使用的所有其他库都包含在Python的标准库中。

导航到GitHub存储库，并下载每个库的所需版本。此配方是使用`libesedb-experimental-20170121`版本开发的。提取发布的内容后，打开终端，导航到提取的目录，并执行以下命令：

```py
./synclibs.sh
./autogen.sh
sudo python setup.py install 
```

要了解有关`pyesedb`库的更多信息，请访问[**https://github.com/libyal/libesedb**](https://github.com/libyal/libesedb)**。**最后，我们可以通过打开Python解释器，导入`pyesedb`，并运行`gpyesedb.get_version()`方法来检查我们的库安装，以确保我们有正确的发布版本。

# 如何做...

我们使用以下方法来实现我们的目标：

1.  确定`SRUDB.dat`文件是否存在并执行文件签名验证。

1.  使用`pyesedb`提取表和表数据。

1.  根据适当的数据类型解释提取的表数据。

1.  为数据库中的每个表创建多个电子表格。

# 工作原理...

我们导入了许多库来帮助解析参数、日期解析、编写 CSV、处理 ESE 数据库和自定义的 `pytskutil` 模块：

```py
from __future__ import print_function
import argparse
from datetime import datetime, timedelta
import os
import pytsk3
import pyewf
import pyesedb
import struct
import sys
import unicodecsv as csv
from utility.pytskutil import TSKUtil
```

此脚本在执行过程中使用了两个全局变量。`TABLE_LOOKUP` 变量是一个查找表，将各种 SRUM 表名与更人性化的描述匹配。这些描述是从 *Yogesh Khatri* 的演示文稿中提取的，该演示文稿在配方开头引用。`APP_ID_LOOKUP` 字典将存储来自 SRUM `SruDbIdMapTable` 表的数据，该表将应用程序分配给其他表中引用的整数值。

```py
TABLE_LOOKUP = {
    "{973F5D5C-1D90-4944-BE8E-24B94231A174}": "Network Data Usage",
    "{D10CA2FE-6FCF-4F6D-848E-B2E99266FA86}": "Push Notifications",
    "{D10CA2FE-6FCF-4F6D-848E-B2E99266FA89}": "Application Resource Usage",
    "{DD6636C4-8929-4683-974E-22C046A43763}": "Network Connectivity Usage",
    "{FEE4E14F-02A9-4550-B5CE-5FA2DA202E37}": "Energy Usage"}

APP_ID_LOOKUP = {}
```

这个配方的命令行处理程序接受两个位置参数，`EVIDENCE_FILE` 和 `TYPE`，分别表示证据文件和证据文件的类型。在验证提供的参数后，我们将这两个输入传递给 `main()` 方法，动作就此开始。

```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EVIDENCE_FILE", help="Evidence file path")
    parser.add_argument("TYPE", help="Type of Evidence",
                        choices=("raw", "ewf"))
    args = parser.parse_args()

    if os.path.exists(args.EVIDENCE_FILE) and os.path.isfile(
            args.EVIDENCE_FILE):
        main(args.EVIDENCE_FILE, args.TYPE)
    else:
        print("[-] Supplied input file {} does not exist or is not a "
              "file".format(args.EVIDENCE_FILE))
        sys.exit(1)
```

`main()` 方法首先创建一个 `TSKUtil` 对象，并创建一个变量来引用包含 Windows 10 系统上 SRUM 数据库的文件夹。然后，我们使用 `query_directory()` 方法来确定目录是否存在。如果存在，我们使用 `recurse_files()` 方法从证据中返回 SRUM 数据库（如果存在）：

```py
def main(evidence, image_type):
    # Create TSK object and query for Internet Explorer index.dat files
    tsk_util = TSKUtil(evidence, image_type)
    path = "/Windows/System32/sru"
    srum_dir = tsk_util.query_directory(path)
    if srum_dir is not None:
        srum_files = tsk_util.recurse_files("SRUDB.dat", path=path,
                                            logic="equal")
```

如果我们找到了 SRUM 数据库，我们会在控制台打印状态消息，并遍历每个命中。对于每个命中，我们提取存储在 `recurse_files()` 方法返回的元组的第二个索引中的文件对象，并使用 `write_file()` 方法将文件缓存到主机文件系统以进行进一步处理：

```py
        if srum_files is not None:
            print("[+] Identified {} potential SRUDB.dat file(s)".format(
                len(srum_files)))
            for hit in srum_files:
                srum_file = hit[2]
                srum_tables = {}
                temp_srum = write_file(srum_file)
```

`write_file()` 方法，如前所述，只是在主机文件系统上创建一个同名文件。该方法读取证据容器中文件的全部内容，并将其写入临时文件。完成后，它将文件的名称返回给父函数。

```py
def write_file(srum_file):
    with open(srum_file.info.name.name, "w") as outfile:
        outfile.write(srum_file.read_random(0, srum_file.info.meta.size))
    return srum_file.info.name.name
```

回到 `main()` 方法，我们使用 `pyesedb.check_file_signature()` 方法验证文件命中，然后再进行任何进一步处理。验证文件后，我们使用 `pyesedb.open()` 方法创建 `pyesedb` 对象，并在控制台上打印包含在文件中的表的数量的状态消息。接下来，我们创建一个 `for` 循环来遍历数据库中的所有表。具体来说，我们首先寻找 `SruDbIdMapTable`，因为我们首先需要使用整数到应用程序名称的配对来填充 `APP_ID_LOOKUP` 字典，然后再处理任何其他表。

一旦找到该表，我们就会读取表中的每条记录。感兴趣的整数值存储在第一个索引中，而应用程序名称存储在第二个索引中。我们使用 `get_value_data_as_integer()` 方法来提取和适当解释整数。而使用 `get_value_data()` 方法，我们可以从记录中提取应用程序名称，并尝试替换字符串中的任何填充字节。最后，我们将这两个值存储在全局的 `APP_ID_LOOKUP` 字典中，使用整数作为键，应用程序名称作为值。

```py
                if pyesedb.check_file_signature(temp_srum):
                    srum_dat = pyesedb.open(temp_srum)
                    print("[+] Process {} tables within database".format(
                        srum_dat.number_of_tables))
                    for table in srum_dat.tables:
                        if table.name != "SruDbIdMapTable":
                            continue
                        global APP_ID_LOOKUP
                        for entry in table.records:
                            app_id = entry.get_value_data_as_integer(1)
                            try:
                                app = entry.get_value_data(2).replace(
                                    "\x00", "")
                            except AttributeError:
                                app = ""
                            APP_ID_LOOKUP[app_id] = app
```

创建 `app lookup` 字典后，我们准备再次遍历每个表，并实际提取数据。对于每个表，我们将其名称分配给一个本地变量，并在控制台上打印有关执行进度的状态消息。然后，在将保存我们处理过的数据的字典中，我们使用表的名称创建一个键，以及包含列和数据列表的字典。列列表表示表本身的实际列名。这些是使用列表推导提取的，然后分配给我们字典结构中列的键。

```py
                    for table in srum_dat.tables:
                        t_name = table.name
                        print("[+] Processing {} table with {} records"
                              .format(t_name, table.number_of_records))
                        srum_tables[t_name] = {"columns": [], "data": []}
                        columns = [x.name for x in table.columns]
                        srum_tables[t_name]["columns"] = columns
```

处理完列后，我们将注意力转向数据本身。当我们迭代表中的每一行时，我们使用`number_of_values()`方法创建一个循环来迭代行中的每个值。在这样做时，我们将解释后的值附加到列表中，然后将列表本身分配给字典中的数据键。SRUM数据库存储多种不同类型的数据（`32位`整数、`64位`整数、字符串等）。`pyesedb`库并不一定支持每种数据类型，使用各种`get_value_as`方法。我们必须自己解释数据，并创建了一个新函数`convert_data()`来做到这一点。现在让我们专注于这个方法。

如果搜索失败，文件签名验证，我们将在控制台打印状态消息，删除临时文件，并继续下一个搜索。其余的`else`语句处理了未找到SRUM数据库和SRUM数据库目录不存在的情况。

```py
                        for entry in table.records:
                            data = []
                            for x in range(entry.number_of_values):
                                data.append(convert_data(
                                    entry.get_value_data(x), columns[x],
                                    entry.get_column_type(x))
                                )
                            srum_tables[t_name]["data"].append(data)
                        write_output(t_name, srum_tables)

                else:
                    print("[-] {} not a valid SRUDB.dat file. Removing "
                          "temp file...".format(temp_srum))
                    os.remove(temp_srum)
                    continue

        else:
            print("[-] SRUDB.dat files not found in {} "
                  "directory".format(path))
            sys.exit(3)

    else:
        print("[-] Directory {} not found".format(path))
        sys.exit(2)
```

`convert_data()`方法依赖于列类型来决定如何解释数据。在大多数情况下，我们使用`struct`来解压数据为适当的数据类型。这个函数是一个大的`if-elif-else`语句。在第一种情况下，我们检查数据是否为`None`，如果是，返回一个空字符串。在第一个`elif`语句中，我们检查列名是否为`"AppId"`；如果是，我们解压代表值的`32位`整数，该值来自`SruDbIdMapTable`，对应一个应用程序名称。我们使用之前创建的全局`APP_ID_LOOKUP`字典返回正确的应用程序名称。接下来，我们为各种列值创建情况，返回适当的数据类型，如`8位`无符号整数、`16位`和`32位`有符号整数、`32位`浮点数和`64位`双精度浮点数。

```py
def convert_data(data, column, col_type):
    if data is None:
        return ""
    elif column == "AppId":
        return APP_ID_LOOKUP[struct.unpack("<i", data)[0]]
    elif col_type == 0:
        return ""
    elif col_type == 1:
        if data == "*":
            return True
        else:
            return False
    elif col_type == 2:
        return struct.unpack("<B", data)[0]
    elif col_type == 3:
        return struct.unpack("<h", data)[0]
    elif col_type == 4:
        return struct.unpack("<i", data)[0]
    elif col_type == 6:
        return struct.unpack("<f", data)[0]
    elif col_type == 7:
        return struct.unpack("<d", data)[0]
```

接着上一段，当列类型等于`8`时，我们有一个`OLE`时间戳。我们必须将该值解压为`64位`整数，然后使用`convert_ole()`方法将其转换为`datetime`对象。列类型`5`、`9`、`10`、`12`、`13`和`16`返回为原始值，无需额外处理。大多数其他`elif`语句使用不同的`struct`格式字符来适当解释数据。列类型`15`也可以是时间戳或`64位`整数。因此，针对SRUM数据库，我们检查列名是否为`"EventTimestamp"`或`"ConnectStartTime"`，在这种情况下，该值是`FILETIME`时间戳，必须进行转换。无论列类型如何，可以肯定的是在这里处理并将其作为适当的类型返回到`main()`方法中。

够了，让我们去看看这些时间戳转换方法：

```py
    elif col_type == 8:
        return convert_ole(struct.unpack("<q", data)[0])
    elif col_type in [5, 9, 10, 12, 13, 16]:
        return data
    elif col_type == 11:
        return data.replace("\x00", "")
    elif col_type == 14:
        return struct.unpack("<I", data)[0]
    elif col_type == 15:
        if column in ["EventTimestamp", "ConnectStartTime"]:
            return convert_filetime(struct.unpack("<q", data)[0])
        else:
            return struct.unpack("<q", data)[0]
    elif col_type == 17:
        return struct.unpack("<H", data)[0]
    else:
        return data
```

要了解有关ESE数据库列类型的更多信息，请访问[https://github.com/libyal/libesedb/blob/b5abe2d05d5342ae02929c26475774dbb3c3aa5d/include/libesedb/definitions.h.in](https://github.com/libyal/libesedb/blob/b5abe2d05d5342ae02929c26475774dbb3c3aa5d/include/libesedb/definitions.h.in)。

`convert_filetime()`方法接受一个整数，并尝试使用之前展示的经过验证的方法进行转换。我们观察到输入整数可能太大，超出`datetime`方法的范围，并为这种情况添加了一些错误处理。否则，该方法与之前讨论的类似。

```py
def convert_filetime(ts):
    if str(ts) == "0":
        return ""
    try:
        dt = datetime(1601, 1, 1) + timedelta(microseconds=ts / 10)
    except OverflowError:
        return ts
    return dt
```

在我们的任何食谱中都是`convert_ole()`方法。`OLE`时间戳格式是一个浮点数，表示自1899年12月30日午夜以来的天数。我们将传递给函数的`64位`整数打包和解包为日期转换所需的适当格式。然后，我们使用熟悉的过程，使用`datetime`指定我们的时代和`timedelta`来提供适当的偏移量。如果我们发现这个值太大，我们捕获`OverflowError`并将`64位`整数原样返回。

```py
def convert_ole(ts):
    ole = struct.unpack(">d", struct.pack(">Q", ts))[0]
    try:
        dt = datetime(1899, 12, 30, 0, 0, 0) + timedelta(days=ole)
    except OverflowError:
        return ts
    return dt
```

要了解更多常见的时间戳格式（包括`ole`），请访问[https://blogs.msdn.microsoft.com/oldnewthing/20030905-02/?p=42653](https://blogs.msdn.microsoft.com/oldnewthing/20030905-02/?p=42653)。

对于数据库中的每个表，都会调用`write_output()`方法。我们检查字典，如果给定表没有结果，则返回该函数。只要我们有结果，我们就会创建一个输出名称来区分SRUM表，并将其创建在当前工作目录中。然后，我们打开电子表格，创建CSV写入器，然后使用`writerow()`和`writerows()`方法将列和数据写入电子表格。

```py
def write_output(table, data):
    if len(data[table]["data"]) == 0:
        return
    if table in TABLE_LOOKUP:
        output_name = TABLE_LOOKUP[table] + ".csv"
    else:
        output_name = "SRUM_Table_{}.csv".format(table)
    print("[+] Writing {} to current working directory: {}".format(
        output_name, os.getcwd()))
    with open(output_name, "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data[table]["columns"])
        writer.writerows(data[table]["data"])
```

运行代码后，我们可以在电子表格中查看提取出的数值。以下两个屏幕截图显示了我们应用程序资源使用报告中找到的前几个数值：

![](../images/00114.jpeg)![](../images/00115.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一个或多个建议：

+   通过这个方法进一步研究文件格式，并扩展对其他感兴趣信息的支持

+   查看Mark Baggett的`srum-dump`（[https://github.com/MarkBaggett/srum-dump](https://github.com/MarkBaggett/srum-dump)）

# 结论

无论这是你第一次使用Python，还是之前多次使用过，你都可以看到正确的代码如何在调查过程中起到重要作用。Python让你能够有效地筛选大型数据集，并更有效地找到调查中的关键信息。随着你的发展，你会发现自动化变得自然而然，因此你的工作效率会提高很多倍。

引用“当我们教学时，我们在学习”归因于罗马哲学家塞内卡，即使在引用的概念中最初并没有将计算机作为教学的主题。但写代码有助于通过要求你更深入地理解其结构和内容来完善你对给定工件的知识。

我们希望你已经学到了很多，并且会继续学习。有大量免费资源值得查看和开源项目可以帮助你更好地磨练技能。如果有一件事你应该从这本书中学到：如何编写一个了不起的CSV写入器。但是，真的，我们希望通过这些例子，你已经更好地掌握了何时以及如何利用Python来发挥你的优势。祝你好运。
