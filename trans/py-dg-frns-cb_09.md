# 探索 Windows 取证工件配方-第一部分

本章将涵盖以下配方：

+   一个人的垃圾是取证人员的宝藏

+   一个棘手的情况

+   阅读注册表

+   收集用户活动

+   缺失的链接

+   四处搜寻

# 介绍

长期以来，Windows 一直是 PC 市场上的首选操作系统。事实上，Windows 约占访问政府网站的用户的 47%，而第二受欢迎的 PC 操作系统 macOS 仅占 8.5%。没有理由怀疑这种情况会很快改变，特别是考虑到 Windows 10 受到的热烈欢迎。因此，未来的调查很可能会继续需要分析 Windows 工件。

本章涵盖了许多类型的工件以及如何使用 Python 和各种第一方和第三方库直接从取证证据容器中解释它们。我们将利用我们在第八章中开发的框架，*处理取证证据容器配方*，直接处理这些工件，而不用担心提取所需文件或挂载镜像的过程。具体来说，我们将涵盖：

+   解释`$I`文件以了解发送到回收站的文件的更多信息

+   从 Windows 7 系统的便笺中读取内容和元数据

+   从注册表中提取值，以了解操作系统版本和其他配置细节

+   揭示与搜索、输入路径和运行命令相关的用户活动

+   解析 LNK 文件以了解历史和最近的文件访问

+   检查`Windows.edb`以获取有关索引文件、文件夹和消息的信息

要查看更多有趣的指标，请访问[`analytics.usa.gov/`](https://analytics.usa.gov/)。

访问[www.packtpub.com/books/content/support](http://www.packtpub.com/books/content/support)下载本章的代码包。

# 一个人的垃圾是取证人员的宝藏

配方难度：中等

Python 版本：2.7

操作系统：Linux

虽然可能不是确切的说法，但是对于大多数调查来说，取证检查回收站中已删除文件是一个重要的步骤。非技术保管人可能不明白这些发送到回收站的文件仍然存在，我们可以了解到原始文件的很多信息，比如原始文件路径以及发送到回收站的时间。虽然特定的工件在不同版本的 Windows 中有所不同，但这个配方侧重于 Windows 7 版本的回收站的`$I`和`$R`文件。

# 入门

这个配方需要安装三个第三方模块才能运行：`pytsk3`、`pyewf`和`unicodecsv`。*有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅第八章，处理取证证据容器配方*。*此脚本中使用的所有其他库都包含在 Python 的标准库中*

因为我们正在用 Python 2.x 开发这些配方，我们很可能会遇到 Unicode 编码和解码错误。为了解决这个问题，我们使用`unicodecsv`库来写这一章节中的所有 CSV 输出。这个第三方模块负责 Unicode 支持，不像 Python 2.x 的标准`csv`模块，并且在这里将得到很好的应用。和往常一样，我们可以使用`pip`来安装`unicodecsv`：

```py
pip install unicodecsv==0.14.1
```

要了解更多关于`unicodecsv`库的信息，请访问[`github.com/jdunck/python-unicodecsv`](https://github.com/jdunck/python-unicodecsv)。

除此之外，我们将继续使用从[第八章](https://cdp.packtpub.com/python_digital_forensics_cookbook/wp-admin/post.php?post=260&action=edit#post_218)开发的`pytskutil`模块，*与取证证据容器配方一起工作*，以允许与取证获取进行交互。这个模块在很大程度上类似于我们之前编写的内容，只是对一些细微的更改以更好地适应我们的目的。您可以通过导航到代码包中的实用程序目录来查看代码。

# 如何做...

要解析来自 Windows 7 机器的`$I`和`$R`文件，我们需要：

1.  递归遍历证据文件中的`$Recycle.bin`文件夹，选择所有以`$I`开头的文件。

1.  读取文件的内容并解析可用的元数据结构。

1.  搜索相关的`$R`文件并检查它是文件还是文件夹。

1.  将结果写入 CSV 文件进行审查。

# 它是如何工作的...

我们导入`argparse`，`datetime`，`os`和`struct`内置库来帮助运行脚本并解释这些文件中的二进制数据。我们还引入了我们的 Sleuth Kit 实用程序来处理证据文件，读取内容，并遍历文件夹和文件。最后，我们导入`unicodecsv`库来帮助编写 CSV 报告。

```py
from __future__ import print_function
from argparse import ArgumentParser
import datetime
import os
import struct

from utility.pytskutil import TSKUtil
import unicodecsv as csv
```

这个配方的命令行处理程序接受三个位置参数，`EVIDENCE_FILE`，`IMAGE_TYPE`和`CSV_REPORT`，分别代表证据文件的路径，证据文件的类型和所需的 CSV 报告输出路径。这三个参数被传递给`main()`函数。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument('EVIDENCE_FILE', help="Path to evidence file")
    parser.add_argument('IMAGE_TYPE', help="Evidence file format",
                        choices=('ewf', 'raw'))
    parser.add_argument('CSV_REPORT', help="Path to CSV report")
    args = parser.parse_args()
    main(args.EVIDENCE_FILE, args.IMAGE_TYPE, args.CSV_REPORT)
```

`main()`函数处理与证据文件的必要交互，以识别和提供任何用于处理的`$I`文件。要访问证据文件，必须提供容器的路径和图像类型。这将启动`TSKUtil`实例，我们使用它来搜索图像中的文件和文件夹。要找到`$I`文件，我们在`tsk_util`实例上调用`recurse_files()`方法，指定要查找的文件名模式，开始搜索的`path`和用于查找文件名的字符串`logic`。`logic`关键字参数接受以下值，这些值对应于字符串操作：`startswith`，`endswith`，`contains`和`equals`。这些指定了用于在扫描的文件和文件夹名称中搜索我们的`$I`模式的字符串操作。

如果找到任何`$I`文件，我们将此列表传递给`process_dollar_i()`函数，以及`tsk_util`对象。在它们都被处理后，我们使用`write_csv()`方法将提取的元数据写入 CSV 报告：

```py
def main(evidence, image_type, report_file):
    tsk_util = TSKUtil(evidence, image_type)

    dollar_i_files = tsk_util.recurse_files("$I", path='/$Recycle.bin',
                                            logic="startswith")

    if dollar_i_files is not None:
        processed_files = process_dollar_i(tsk_util, dollar_i_files)

        write_csv(report_file,
                  ['file_path', 'file_size', 'deleted_time',
                   'dollar_i_file', 'dollar_r_file', 'is_directory'],
                  processed_files)
    else:
        print("No $I files found")
```

`process_dollar_i()`函数接受`tsk_util`对象和发现的`$I`文件列表作为输入。我们遍历这个列表并检查每个文件。`dollar_i_files`列表中的每个元素本身都是一个元组列表，其中每个元组元素依次包含文件的名称、相对路径、用于访问文件内容的句柄和文件系统标识符。有了这些可用的属性，我们将调用我们的`read_dollar_i()`函数，并向其提供第三个元组，文件对象句柄。如果这是一个有效的`$I`文件，该方法将从原始文件中返回提取的元数据字典，否则返回`None`。如果文件有效，我们将继续处理它，将文件路径添加到`$I`文件的`file_attribs`字典中：

```py
def process_dollar_i(tsk_util, dollar_i_files):
    processed_files = []
    for dollar_i in dollar_i_files:
        # Interpret file metadata
        file_attribs = read_dollar_i(dollar_i[2])
        if file_attribs is None:
            continue # Invalid $I file
        file_attribs['dollar_i_file'] = os.path.join(
            '/$Recycle.bin', dollar_i[1][1:])
```

接下来，我们在图像中搜索相关的`$R`文件。为此，我们将基本路径与`$I`文件（包括`$Recycle.bin`和`SID`文件夹）连接起来，以减少搜索相应`$R`文件所需的时间。在 Windows 7 中，`$I`和`$R`文件具有类似的文件名，前两个字母分别是`$I`和`$R`，后面是一个共享标识符。通过在我们的搜索中使用该标识符，并指定我们期望找到`$R`文件的特定文件夹，我们已经减少了误报的可能性。使用这些模式，我们再次使用`startswith`逻辑查询我们的证据文件：

```py
        # Get the $R file
        recycle_file_path = os.path.join(
            '/$Recycle.bin',
            dollar_i[1].rsplit("/", 1)[0][1:]
        )
        dollar_r_files = tsk_util.recurse_files(
            "$R" + dollar_i[0][2:],
            path=recycle_file_path, logic="startswith"
        )
```

如果搜索`$R`文件失败，我们尝试查询具有相同信息的目录。如果此查询也失败，我们将附加字典值，指出未找到`$R`文件，并且我们不确定它是文件还是目录。然而，如果我们找到匹配的目录，我们会记录目录的路径，并将`is_directory`属性设置为`True`：

```py
        if dollar_r_files is None:
            dollar_r_dir = os.path.join(recycle_file_path,
                                        "$R" + dollar_i[0][2:])
            dollar_r_dirs = tsk_util.query_directory(dollar_r_dir)
            if dollar_r_dirs is None:
                file_attribs['dollar_r_file'] = "Not Found"
                file_attribs['is_directory'] = 'Unknown'
            else:
                file_attribs['dollar_r_file'] = dollar_r_dir
                file_attribs['is_directory'] = True
```

如果搜索`$R`文件返回一个或多个命中，我们使用列表推导创建一个匹配文件的列表，存储在以分号分隔的 CSV 中，并将`is_directory`属性标记为`False`。

```py
        else:
            dollar_r = [os.path.join(recycle_file_path, r[1][1:])
                        for r in dollar_r_files]
            file_attribs['dollar_r_file'] = ";".join(dollar_r)
            file_attribs['is_directory'] = False
```

在退出循环之前，我们将`file_attribs`字典附加到`processed_files`列表中，该列表存储了所有`$I`处理过的字典。这个字典列表将被返回到`main()`函数，在报告过程中使用。

```py
        processed_files.append(file_attribs)
    return processed_files
```

让我们简要地看一下`read_dollar_i()`方法，用于使用`struct`从二进制文件中解析元数据。我们首先通过使用 Sleuth Kit 的`read_random()`方法来检查文件头，读取签名的前八个字节。如果签名不匹配，我们返回`None`来警告`$I`未通过验证，是无效的文件格式。

```py
def read_dollar_i(file_obj):
    if file_obj.read_random(0, 8) != '\x01\x00\x00\x00\x00\x00\x00\x00':
        return None # Invalid file
```

如果我们检测到一个有效的文件，我们继续从`$I`文件中读取和解压值。首先是文件大小属性，位于字节偏移`8`，长度为`8`字节。我们使用`struct`解压缩这个值，并将整数存储在一个临时变量中。下一个属性是删除时间，存储在字节偏移`16`和`8`字节长。这是一个 Windows `FILETIME`对象，我们将借用一些旧代码来稍后将其处理为可读的时间戳。最后一个属性是以前的文件路径，我们从字节`24`读取到文件的末尾：

```py
    raw_file_size = struct.unpack('<q', file_obj.read_random(8, 8))
    raw_deleted_time = struct.unpack('<q', file_obj.read_random(16, 8))
    raw_file_path = file_obj.read_random(24, 520)
```

提取了这些值后，我们将整数解释为可读的值。我们使用`sizeof_fmt()`函数将文件大小整数转换为可读的大小，包含诸如 MB 或 GB 的大小前缀。接下来，我们使用来自第七章的日期解析配方的逻辑来解释时间戳（在适应该函数仅使用整数后）。最后，我们将路径解码为 UTF-16 并删除空字节值。然后将这些精细的细节作为字典返回给调用函数：

```py
    file_size = sizeof_fmt(raw_file_size[0])
    deleted_time = parse_windows_filetime(raw_deleted_time[0])
    file_path = raw_file_path.decode("utf16").strip("\x00")
    return {'file_size': file_size, 'file_path': file_path,
            'deleted_time': deleted_time}
```

我们的`sizeof_fmt()`函数是从[StackOverflow.com](https://stackoverflow.com/)借来的，这是一个充满了许多编程问题解决方案的网站。虽然我们可以自己起草，但这段代码对我们的目的来说形式良好。它接受整数`num`并遍历列出的单位后缀。如果数字小于`1024`，则数字、单位和后缀被连接成一个字符串并返回；否则，数字除以`1024`并通过下一次迭代。如果数字大于 1 zettabyte，它将以 yottabytes 的形式返回信息。为了你的利益，我们希望数字永远不会那么大。

```py
def sizeof_fmt(num, suffix='B'):
    # From https://stackoverflow.com/a/1094933/3194812
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
```

我们的下一个支持函数是`parse_windows_filetime()`，改编自第七章中的先前日期解析配方，*基于日志的证据配方*。我们借用这个逻辑并将代码压缩为只解释整数并返回给调用函数的格式化日期。像我们刚刚讨论的这两个通用函数一样，它们在你的工具库中是很方便的，因为你永远不知道什么时候会需要这个逻辑。

```py
def parse_windows_filetime(date_value):
    microseconds = float(date_value) / 10
    ts = datetime.datetime(1601, 1, 1) + datetime.timedelta(
        microseconds=microseconds)
    return ts.strftime('%Y-%m-%d %H:%M:%S.%f')
```

最后，我们准备将处理后的结果写入 CSV 文件。毫无疑问，这个函数与我们所有其他的 CSV 函数类似。唯一的区别是它在底层使用了`unicodecsv`库，尽管这里使用的方法和函数名称是相同的：

```py
def write_csv(outfile, fieldnames, data):
    with open(outfile, 'wb') as open_outfile:
        csvfile = csv.DictWriter(open_outfile, fieldnames)
        csvfile.writeheader()
        csvfile.writerows(data)
```

在下面的两个屏幕截图中，我们可以看到这个配方从`$I`和`$R`文件中提取的数据的示例：

![](img/00096.jpeg)![](img/00097.jpeg)

# 一个棘手的情况

配方难度：中等

Python 版本：2.7

操作系统：Linux

计算机已经取代了纸和笔。我们已经将许多过程和习惯转移到了这些机器上，其中一个仅限于纸张的习惯，包括做笔记和列清单。一个复制真实习惯的功能是 Windows 的便利贴。这些便利贴可以让持久的便签漂浮在桌面上，可以选择颜色、字体等选项。这个配方将允许我们探索这些便利贴，并将它们添加到我们的调查工作流程中。

# 开始

这个配方需要安装四个第三方模块才能运行：`olefile`，`pytsk3`，`pyewf`和`unicodecsv`。有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅第八章，*使用法证证据容器* *配方*。同样，有关安装`unicodecsv`的详细信息，请参阅*一个人的垃圾是法医检查员的宝藏*配方中的*入门*部分。此脚本中使用的所有其他库都包含在 Python 的标准库中。

Windows 的便利贴文件存储为`OLE`文件。因此，我们将利用`olefile`库与 Windows 的便利贴进行交互并提取数据。`olefile`库可以通过`pip`安装：

```py
pip install olefile==0.44
```

要了解更多关于`olefile`库的信息，请访问[`olefile.readthedocs.io/en/latest/index.html`](https://olefile.readthedocs.io/en/latest/index.html)。

# 如何做...

为了正确制作这个配方，我们需要采取以下步骤：

1.  打开证据文件并找到所有用户配置文件中的`StickyNote.snt`文件。

1.  解析 OLE 流中的元数据和内容。

1.  将 RTF 内容写入文件。

1.  创建元数据的 CSV 报告。

# 它是如何工作的...

这个脚本，就像其他脚本一样，以导入所需库的导入语句开始执行。这里的两个新库是`olefile`，正如我们讨论的，它解析 Windows 的便利贴 OLE 流，以及`StringIO`，一个内置库，用于将数据字符串解释为类似文件的对象。这个库将用于将`pytsk`文件对象转换为`olefile`库可以解释的流： 

```py
from __future__ import print_function
from argparse import ArgumentParser
import unicodecsv as csv
import os
import StringIO

from utility.pytskutil import TSKUtil
import olefile
```

我们指定一个全局变量，`REPORT_COLS`，代表报告列。这些静态列将在几个函数中使用。

```py
REPORT_COLS = ['note_id', 'created', 'modified', 'note_text', 'note_file']
```

这个配方的命令行处理程序需要三个位置参数，`EVIDENCE_FILE`，`IMAGE_TYPE`和`REPORT_FOLDER`，它们分别代表证据文件的路径，证据文件的类型和期望的输出目录路径。这与之前的配方类似，唯一的区别是`REPORT_FOLDER`，这是一个我们将写入便利贴 RTF 文件的目录：

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument('EVIDENCE_FILE', help="Path to evidence file")
    parser.add_argument('IMAGE_TYPE', help="Evidence file format",
                        choices=('ewf', 'raw'))
    parser.add_argument('REPORT_FOLDER', help="Path to report folder")
    args = parser.parse_args()
    main(args.EVIDENCE_FILE, args.IMAGE_TYPE, args.REPORT_FOLDER)
```

我们的主要函数开始方式与上一个类似，处理证据文件并搜索我们要解析的文件。在这种情况下，我们正在寻找`StickyNotes.snt`文件，该文件位于每个用户的`AppData`目录中。因此，我们将搜索限制为`/Users`文件夹，并寻找与确切名称匹配的文件：

```py
def main(evidence, image_type, report_folder):
    tsk_util = TSKUtil(evidence, image_type)
    note_files = tsk_util.recurse_files('StickyNotes.snt', '/Users',
                                        'equals')
```

然后，我们遍历生成的文件，分离用户的主目录名称，并设置`olefile`库所需的类文件对象。接下来，我们调用`parse_snt_file()`函数处理文件，并返回一个结果列表进行遍历。在这一点上，如果`note_data`不是`None`，我们使用`write_note_rtf()`方法写入 RTF 文件。此外，我们将从`prep_note_report()`处理的数据附加到`report_details`列表中。一旦`for`循环完成，我们使用`write_csv()`方法写入 CSV 报告，提供报告名称、报告列和我们构建的粘贴便笺信息列表。

```py
    report_details = []
    for note_file in note_files:
        user_dir = note_file[1].split("/")[1]
        file_like_obj = create_file_like_obj(note_file[2])
        note_data = parse_snt_file(file_like_obj)
        if note_data is None:
            continue
        write_note_rtf(note_data, os.path.join(report_folder, user_dir))
        report_details += prep_note_report(note_data, REPORT_COLS,
                                           "/Users" + note_file[1])
    write_csv(os.path.join(report_folder, 'sticky_notes.csv'), REPORT_COLS,
              report_details)
```

`create_file_like_obj()`函数获取我们的`pytsk`文件对象并读取文件的大小。这个大小在`read_random()`函数中用于将整个粘贴便笺内容读入内存。我们将`file_content`传递给`StringIO()`类，将其转换为`olefile`库可以读取的类文件对象，然后将其返回给父函数：

```py
def create_file_like_obj(note_file):
    file_size = note_file.info.meta.size
    file_content = note_file.read_random(0, file_size)
    return StringIO.StringIO(file_content)
```

`parse_snt_file()`函数接受类文件对象作为输入，并用于读取和解释粘贴便笺文件。我们首先验证类文件对象是否是 OLE 文件，如果不是，则返回`None`。如果是，我们使用`OleFileIO()`方法打开类文件对象。这提供了一个流列表，允许我们遍历每个粘贴便笺的每个元素。在遍历列表时，我们检查流是否包含三个破折号，因为这表明流包含粘贴便笺的唯一标识符。该文件可以包含一个或多个粘贴便笺，每个粘贴便笺由唯一的 ID 标识。粘贴便笺数据根据流的第一个索引元素的值，直接读取为 RTF 数据或 UTF-16 编码数据。

我们还使用`getctime()`和`getmtime()`函数从流中读取创建和修改的信息。接下来，我们将粘贴便笺的 RTF 或 UTF-16 编码数据提取到`content`变量中。注意，我们必须在存储之前解码 UTF-16 编码的数据。如果有内容要保存，我们将其添加到`note`字典中，并继续处理所有剩余的流。一旦所有流都被处理，`note`字典将返回给父函数：

```py
def parse_snt_file(snt_file):
    if not olefile.isOleFile(snt_file):
        print("This is not an OLE file")
        return None
    ole = olefile.OleFileIO(snt_file)
    note = {}
    for stream in ole.listdir():
        if stream[0].count("-") == 3:
            if stream[0] not in note:
                note[stream[0]] = {
                    # Read timestamps
                    "created": ole.getctime(stream[0]),
                    "modified": ole.getmtime(stream[0])
                }

            content = None
            if stream[1] == '0':
                # Parse RTF text
                content = ole.openstream(stream).read()
            elif stream[1] == '3':
                # Parse UTF text
                content = ole.openstream(stream).read().decode("utf-16")

            if content:
                note[stream[0]][stream[1]] = content

    return note
```

为了创建 RTF 文件，我们将便笺数据字典传递给`write_note_rtf()`函数。如果报告文件夹不存在，我们使用`os`库来创建它。在这一点上，我们遍历`note_data`字典，分离`note_id`键和`stream_data`值。在打开之前，`note_id`用于创建输出 RTF 文件的文件名。

然后将存储在流零中的数据写入输出的 RTF 文件，然后关闭文件并处理下一个粘贴便笺：

```py
def write_note_rtf(note_data, report_folder):
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)
    for note_id, stream_data in note_data.items():
        fname = os.path.join(report_folder, note_id + ".rtf")
        with open(fname, 'w') as open_file:
            open_file.write(stream_data['0'])
```

将粘贴便笺上的内容写好后，我们现在转向`prep_note_report()`函数处理的 CSV 报告本身，这个函数处理方式有点不同。它将嵌套字典转换为一组更有利于 CSV 电子表格的扁平字典。我们通过包括`note_id`键来扁平化它，并使用全局`REPORT_COLS`列表中指定的键来命名字段。

```py
def prep_note_report(note_data, report_cols, note_file):
    report_details = []
    for note_id, stream_data in note_data.items():
        report_details.append({
            "note_id": note_id,
            "created": stream_data['created'],
            "modified": stream_data['modified'],
            "note_text": stream_data['3'].strip("\x00"),
            "note_file": note_file
        })
    return report_details
```

最后，在`write_csv()`方法中，我们创建一个`csv.Dictwriter`对象来创建粘贴便笺数据的概述报告。这个 CSV 写入器还使用`unicodecsv`库，并将字典列表写入文件，使用`REPORT_COLS`列的`fieldnames`。

```py
def write_csv(outfile, fieldnames, data):
    with open(outfile, 'wb') as open_outfile:
        csvfile = csv.DictWriter(open_outfile, fieldnames)
        csvfile.writeheader()
        csvfile.writerows(data)
```

然后我们可以查看输出，因为我们有一个包含导出的粘贴便笺和报告的新目录：

![](img/00098.jpeg)

打开我们的报告，我们可以查看注释元数据并收集一些内部内容，尽管大多数电子表格查看器在处理非 ASCII 字符解释时会遇到困难：

![](img/00099.jpeg)

最后，我们可以打开输出的 RTF 文件并查看原始内容：

![](img/00100.jpeg)

# 读取注册表

食谱难度：中等

Python 版本：2.7

操作系统：Linux

Windows 注册表包含许多与操作系统配置、用户活动、软件安装和使用等相关的重要细节。由于它们包含的文物数量和与 Windows 系统的相关性，这些文件经常受到严格审查和研究。解析注册表文件使我们能够访问可以揭示基本操作系统信息、访问文件夹和文件、应用程序使用情况、USB 设备等的键和值。在这个食谱中，我们专注于从`SYSTEM`和`SOFTWARE`注册表文件中访问常见的基线信息。

# 入门

此食谱需要安装三个第三方模块才能正常运行：`pytsk3`，`pyewf`和`Registry`。有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅第八章，*使用取证证据容器* *食谱*。此脚本中使用的所有其他库都包含在 Python 的标准库中。

在这个食谱中，我们使用`Registry`模块以面向对象的方式与注册表文件进行交互。重要的是，该模块可用于与外部和独立的注册表文件进行交互。可以使用`pip`安装`Registry`模块：

```py
pip install python-registry==1.0.4
```

要了解有关`Registry`库的更多信息，请访问[`github.com/williballenthin/python-registry`](https://github.com/williballenthin/python-registry)。

# 如何做...

要构建我们的注册表系统概述脚本，我们需要：

1.  通过名称和路径查找要处理的注册表文件。

1.  使用`StringIO`和`Registry`模块打开这些文件。

1.  处理每个注册表文件，将解析的值打印到控制台以进行解释。

# 它是如何工作的...

导入与本章其他食谱重叠的导入。这些模块允许我们处理参数解析，日期操作，将文件读入内存以供`Registry`库使用，并解压和解释我们从注册表值中提取的二进制数据。我们还导入`TSKUtil()`类和`Registry`模块以处理注册表文件。

```py
from __future__ import print_function
from argparse import ArgumentParser
import datetime
import StringIO
import struct

from utility.pytskutil import TSKUtil
from Registry import Registry
```

此食谱的命令行处理程序接受两个位置参数，`EVIDENCE_FILE`和`IMAGE_TYPE`，分别表示证据文件的路径和证据文件的类型：

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument('EVIDENCE_FILE', help="Path to evidence file")
    parser.add_argument('IMAGE_TYPE', help="Evidence file format",
                        choices=('ewf', 'raw'))
    args = parser.parse_args()
    main(args.EVIDENCE_FILE, args.IMAGE_TYPE)
```

`main()`函数首先通过从证据中创建一个`TSKUtil`对象，并在`/Windows/System32/config`文件夹中搜索`SYSTEM`和`SOFTWARE`注册表文件。在将它们传递给各自的处理函数之前，我们使用`open_file_as_reg()`函数创建这些注册表文件的`Registry()`类实例。

```py
def main(evidence, image_type):
    tsk_util = TSKUtil(evidence, image_type)
    tsk_system_hive = tsk_util.recurse_files(
        'system', '/Windows/system32/config', 'equals')
    tsk_software_hive = tsk_util.recurse_files(
        'software', '/Windows/system32/config', 'equals')

    system_hive = open_file_as_reg(tsk_system_hive[0][2])
    software_hive = open_file_as_reg(tsk_software_hive[0][2])

    process_system_hive(system_hive)
    process_software_hive(software_hive)
```

要打开注册表文件，我们需要从`pytsk`元数据中收集文件的大小，并将整个文件从字节零到文件末尾读入变量中。然后，我们将此变量提供给`StringIO()`实例，该实例允许我们使用`Registry()`类打开类似文件的对象。我们将`Registry`类实例返回给调用函数进行进一步处理：

```py
def open_file_as_reg(reg_file):
    file_size = reg_file.info.meta.size
    file_content = reg_file.read_random(0, file_size)
    file_like_obj = StringIO.StringIO(file_content)
    return Registry.Registry(file_like_obj)
```

让我们从`SYSTEM` hive 处理开始。这个 hive 主要包含在控制集中的大部分信息。`SYSTEM` hive 通常有两个或更多的控制集，它们充当存储的配置的备份系统。为了简单起见，我们只读取当前的控制集。为了识别当前的控制集，我们通过`root`键在 hive 中找到我们的立足点，并使用`find_key()`方法获取`Select`键。在这个键中，我们读取`Current`值，使用`value()`方法选择它，并在`value`对象上使用`value()`方法来呈现值的内容。虽然方法的命名有点模糊，但键中的值是有名称的，所以我们首先需要按名称选择它们，然后再调用它们所持有的内容。使用这些信息，我们选择当前的控制集键，传递一个适当填充的整数作为当前控制集（如`ControlSet0001`）。这个对象将在函数的其余部分用于导航到特定的`subkeys`和`values`：

```py
def process_system_hive(hive):
    root = hive.root()
    current_control_set = root.find_key("Select").value("Current").value()
    control_set = root.find_key("ControlSet{:03d}".format(
        current_control_set))
```

我们将从`SYSTEM` hive 中提取的第一条信息是关机时间。我们从当前控制集中读取`Control\Windows\ShutdownTime`值，并将十六进制值传递给`struct`来将其转换为`64 位`整数。然后我们将这个整数提供给 Windows `FILETIME`解析器，以获得一个可读的日期字符串，然后将其打印到控制台上。

```py
    raw_shutdown_time = struct.unpack(
        '<Q', control_set.find_key("Control").find_key("Windows").value(
            "ShutdownTime").value()
    )
    shutdown_time = parse_windows_filetime(raw_shutdown_time[0])
    print("Last Shutdown Time: {}".format(shutdown_time))
```

接下来，我们将确定机器的时区信息。这可以在`Control\TimeZoneInformation\TimeZoneKeyName`值中找到。这将返回一个字符串值，我们可以直接打印到控制台上：

```py
    time_zone = control_set.find_key("Control").find_key(
        "TimeZoneInformation").value("TimeZoneKeyName").value()
    print("Machine Time Zone: {}".format(time_zone))
```

接下来，我们收集机器的主机名。这可以在`Control\ComputerName\ComputerName`键的`ComputerName`值下找到。提取的值是一个字符串，我们可以打印到控制台上：

```py
    computer_name = control_set.find_key(
        "Control").find_key("ComputerName").find_key(
            "ComputerName").value("ComputerName").value()
    print("Machine Name: {}".format(computer_name))
```

到目前为止，还是相当容易的，对吧？最后，对于`System` hive，我们解析关于最后访问时间戳配置的信息。这个`registry`键确定了 NTFS 卷的最后访问时间戳是否被维护，并且通常在系统上默认情况下是禁用的。为了确认这一点，我们查找`Control\FileSystem`键中的`NtfsDisableLastAccessUpdate`值，看它是否等于`1`。如果是，最后访问时间戳就不会被维护，并且在打印到控制台之前标记为禁用。请注意这个一行的`if-else`语句，虽然可能有点难以阅读，但它确实有它的用途：

```py
    last_access = control_set.find_key("Control").find_key(
        "FileSystem").value("NtfsDisableLastAccessUpdate").value()
    last_access = "Disabled" if last_access == 1 else "enabled"
    print("Last Access Updates: {}".format(last_access))
```

我们的 Windows `FILETIME`解析器从以前的日期解析配方中借用逻辑，接受一个整数，我们将其转换为可读的日期字符串。我们还从相同的日期解析配方中借用了`Unix` epoch 日期解析器的逻辑，并将用它来解释来自`Software` hive 的日期。

```py
def parse_windows_filetime(date_value):
    microseconds = float(date_value) / 10
    ts = datetime.datetime(1601, 1, 1) + datetime.timedelta(
        microseconds=microseconds)
    return ts.strftime('%Y-%m-%d %H:%M:%S.%f')

def parse_unix_epoch(date_value):
    ts = datetime.datetime.fromtimestamp(date_value)
    return ts.strftime('%Y-%m-%d %H:%M:%S.%f')
```

我们的最后一个函数处理`SOFTWARE` hive，在控制台窗口向用户呈现信息。这个函数也是通过收集 hive 的根开始，然后选择`Microsoft\Windows NT\CurrentVersion`键。这个键包含有关 OS 安装元数据和其他有用的子键的值。在这个函数中，我们将提取`ProductName`、`CSDVersion`、`CurrentBuild number`、`RegisteredOwner`、`RegisteredOrganization`和`InstallDate`值。虽然这些值大多是我们可以直接打印到控制台的字符串，但在打印之前我们需要使用`Unix` epoch 转换器来解释安装日期值。

```py
def process_software_hive(hive):
    root = hive.root()
    nt_curr_ver = root.find_key("Microsoft").find_key(
        "Windows NT").find_key("CurrentVersion")

    print("Product name: {}".format(nt_curr_ver.value(
        "ProductName").value()))
    print("CSD Version: {}".format(nt_curr_ver.value(
        "CSDVersion").value()))
    print("Current Build: {}".format(nt_curr_ver.value(
        "CurrentBuild").value()))
    print("Registered Owner: {}".format(nt_curr_ver.value(
        "RegisteredOwner").value()))
    print("Registered Org: {}".format(nt_curr_ver.value(
        "RegisteredOrganization").value()))

    raw_install_date = nt_curr_ver.value("InstallDate").value()
    install_date = parse_unix_epoch(raw_install_date)
    print("Installation Date: {}".format(install_date))
```

当我们运行这个脚本时，我们可以了解到我们解释的键中存储的信息：

![](img/00101.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个以下建议：

+   添加逻辑来处理在初始搜索中找不到`SYSTEM`或`SOFTWARE` hive 的情况

+   考虑添加对`NTUSER.DAT`文件的支持，提取有关挂载点和 shell bags 查询的基本信息

+   从`System` hive 列出基本的 USB 设备信息

+   解析`SAM` hive 以显示用户和组信息

# 收集用户活动

配方难度：中等

Python 版本：2.7

操作系统：Linux

Windows 存储了大量关于用户活动的信息，就像其他注册表 hive 一样，`NTUSER.DAT`文件是调查中可以依赖的重要资源。这个 hive 存在于每个用户的配置文件中，并存储与特定用户在系统上相关的信息和配置。

在这个配方中，我们涵盖了`NTUSER.DAT`中的多个键，这些键揭示了用户在系统上的操作。这包括在 Windows 资源管理器中运行的先前搜索、输入到资源管理器导航栏的路径以及 Windows“运行”命令中最近使用的语句。这些工件更好地说明了用户如何与系统进行交互，并可能揭示用户对系统的正常或异常使用看起来是什么样子。

# 开始

这个配方需要安装四个第三方模块才能正常工作：`jinja2`、`pytsk3`、`pyewf`和`Registry`。有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅第八章，*使用取证证据容器* *配方*。同样，有关安装`Registry`的详细信息，请参阅*入门*部分*读取注册表*配方。此脚本中使用的所有其他库都包含在 Python 的标准库中。

我们将重新介绍`jinja2`，这是在第二章中首次介绍的，*创建工件报告* *配方*，用于构建 HTML 报告。这个库是一个模板语言，允许我们使用 Python 语法以编程方式构建文本文件。作为提醒，我们可以使用`pip`来安装这个库：

```py
pip install jinja2==2.9.6
```

# 如何做...

要从图像中的`NTUSER.DAT`文件中提取这些值，我们必须：

1.  在系统中搜索所有`NTUSER.DAT`文件。

1.  解析每个`NTUSER.DAT`文件的`WordWheelQuery`键。

1.  读取每个`NTUSER.DAT`文件的`TypedPath`键。

1.  提取每个`NTUSER.DAT`文件的`RunMRU`键。

1.  将每个处理过的工件写入 HTML 报告。

# 它是如何工作的...

我们的导入方式与之前的配方相同，添加了`jinja2`模块：

```py
from __future__ import print_function
from argparse import ArgumentParser
import os
import StringIO
import struct

from utility.pytskutil import TSKUtil
from Registry import Registry
import jinja2
```

这个配方的命令行处理程序接受三个位置参数，`EVIDENCE_FILE`、`IMAGE_TYPE`和`REPORT`，分别代表证据文件的路径、证据文件的类型和 HTML 报告的期望输出路径。这三个参数被传递给`main()`函数。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument('EVIDENCE_FILE',
                        help="Path to evidence file")
    parser.add_argument('IMAGE_TYPE',
                        help="Evidence file format",
                        choices=('ewf', 'raw'))
    parser.add_argument('REPORT',
                        help="Path to report file")
    args = parser.parse_args()
    main(args.EVIDENCE_FILE, args.IMAGE_TYPE, args.REPORT)
```

`main()`函数首先通过读取证据文件并搜索所有`NTUSER.DAT`文件来开始。随后，我们设置了一个字典对象`nt_rec`，虽然复杂，但设计得可以简化 HTML 报告生成过程。然后，我们开始迭代发现的 hive，并从路径中解析出用户名以供处理函数参考。

```py
def main(evidence, image_type, report):
    tsk_util = TSKUtil(evidence, image_type)
    tsk_ntuser_hives = tsk_util.recurse_files('ntuser.dat',
                                              '/Users', 'equals')

    nt_rec = {
        'wordwheel': {'data': [], 'title': 'WordWheel Query'},
        'typed_path': {'data': [], 'title': 'Typed Paths'},
        'run_mru': {'data': [], 'title': 'Run MRU'}
    }
    for ntuser in tsk_ntuser_hives:
        uname = ntuser[1].split("/")[1]
```

接下来，我们将`pytsk`文件句柄传递给`Registry`对象以打开。得到的对象用于收集所有所需值（`Software\Microsoft\Windows\CurrentVersion\Explorer`）中的`root`键。如果未找到此键路径，我们将继续处理下一个`NTUSER.DAT`文件。

```py
        open_ntuser = open_file_as_reg(ntuser[2])
        try:
            explorer_key = open_ntuser.root().find_key(
                "Software").find_key("Microsoft").find_key(
                    "Windows").find_key("CurrentVersion").find_key(
                        "Explorer")
        except Registry.RegistryKeyNotFoundException:
            continue # Required registry key not found for user
```

如果找到了键，我们调用负责每个工件的三个处理函数，并提供共享键对象和用户名。返回的数据存储在字典中的相应数据键中。我们可以通过扩展存储对象定义并添加一个与这里显示的其他函数具有相同配置文件的新函数，轻松扩展代码解析的工件数量：

```py
        nt_rec['wordwheel']['data'] += parse_wordwheel(
            explorer_key, uname)
        nt_rec['typed_path']['data'] += parse_typed_paths(
            explorer_key, uname)
        nt_rec['run_mru']['data'] += parse_run_mru(
            explorer_key, uname)
```

在遍历`NTUSER.DAT`文件之后，我们通过提取数据列表中第一项的键列表来为每种记录类型设置标题。由于数据列表中的所有字典对象都具有统一的键，我们可以使用这种方法来减少传递的参数或变量的数量。这些语句也很容易扩展。

```py
    nt_rec['wordwheel']['headers'] = \
        nt_rec['wordwheel']['data'][0].keys()

    nt_rec['typed_path']['headers'] = \
        nt_rec['typed_path']['data'][0].keys()

    nt_rec['run_mru']['headers'] = \
        nt_rec['run_mru']['data'][0].keys()
```

最后，我们将完成的字典对象和报告文件的路径传递给我们的`write_html()`方法：

```py
    write_html(report, nt_rec)
```

我们之前在上一个示例中见过`open_file_as_reg()`方法。作为提醒，它接受`pytsk`文件句柄，并通过`StringIO`类将其读入`Registry`类。返回的`Registry`对象允许我们以面向对象的方式与注册表交互和读取。

```py
def open_file_as_reg(reg_file):
    file_size = reg_file.info.meta.size
    file_content = reg_file.read_random(0, file_size)
    file_like_obj = StringIO.StringIO(file_content)
    return Registry.Registry(file_like_obj)
```

第一个处理函数处理`WordWheelQuery`键，它存储了用户在 Windows 资源管理器中运行的搜索的信息。我们可以通过从`explorer_key`对象中按名称访问键来解析这个遗物。如果键不存在，我们将返回一个空列表，因为我们没有任何值可以提取。

```py
def parse_wordwheel(explorer_key, username):
    try:
        wwq = explorer_key.find_key("WordWheelQuery")
    except Registry.RegistryKeyNotFoundException:
        return []
```

另一方面，如果这个键存在，我们遍历`MRUListEx`值，它包含一个包含搜索顺序的整数列表。列表中的每个数字都与键中相同数字的值相匹配。因此，我们读取列表的顺序，并按照它们出现的顺序解释剩余的值。每个值的名称都存储为两个字节的整数，所以我们将这个列表分成两个字节的块，并用`struct`读取整数。然后在检查它不存在后，将这个值追加到列表中。如果它存在于列表中，并且是`\x00`或`\xFF`，那么我们已经到达了`MRUListEx`数据的末尾，并且跳出循环：

```py
    mru_list = wwq.value("MRUListEx").value()
    mru_order = []
    for i in xrange(0, len(mru_list), 2):
        order_val = struct.unpack('h', mru_list[i:i + 2])[0]
        if order_val in mru_order and order_val in (0, -1):
            break
        else:
            mru_order.append(order_val)
```

使用我们排序后的值列表，我们遍历它以提取按顺序运行的搜索词。由于我们知道使用的顺序，我们可以将`WordWheelQuery`键的最后写入时间作为搜索词的时间戳。这个时间戳只与最近运行的搜索相关联。所有其他搜索都被赋予值`N/A`。

```py
    search_list = []
    for count, val in enumerate(mru_order):
        ts = "N/A"
        if count == 0:
            ts = wwq.timestamp()
```

之后，在`append`语句中构建字典，添加时间值、用户名、顺序（作为计数整数）、值的名称和搜索内容。为了正确显示搜索内容，我们需要将键名提供为字符串并解码文本为 UTF-16。这个文本一旦去除了空终止符，就可以用于报告。直到所有值都被处理并最终返回为止，列表将被构建出来。

```py
        search_list.append({
            'timestamp': ts,
            'username': username,
            'order': count,
            'value_name': str(val),
            'search': wwq.value(str(val)).value().decode(
                "UTF-16").strip("\x00")
        })
    return search_list
```

下一个处理函数处理输入的路径键，与之前的处理函数使用相同的参数。我们以相同的方式访问键，并在`TypedPaths`子键未找到时返回空列表。

```py
def parse_typed_paths(explorer_key, username):
    try:
        typed_paths = explorer_key.find_key("TypedPaths")
    except Registry.RegistryKeyNotFoundException:
        return []
```

这个键没有 MRU 值来排序输入的路径，所以我们读取它的所有值并直接添加到列表中。我们可以从这个键中获取值的名称和路径，并为了额外的上下文添加用户名值。我们通过将字典值的列表返回给`main()`函数来完成这个函数。

```py
    typed_path_details = []
    for val in typed_paths.values():
        typed_path_details.append({
            "username": username,
            "value_name": val.name(),
            "path": val.value()
        })
    return typed_path_details
```

我们的最后一个处理函数处理`RunMRU`键。如果它在`explorer_key`中不存在，我们将像之前一样返回一个空列表。

```py
def parse_run_mru(explorer_key, username):
    try:
        run_mru = explorer_key.find_key("RunMRU")
    except Registry.RegistryKeyNotFoundException:
        return []
```

由于这个键可能是空的，我们首先检查是否有值可以解析，如果没有，就返回一个空列表，以防止进行任何不必要的处理。

```py
    if len(run_mru.values()) == 0:
        return []
```

与`WordWheelQuery`类似，这个键也有一个 MRU 值，我们处理它以了解其他值的正确顺序。这个列表以不同的方式存储项目，因为它的值是字母而不是整数。这使得我们的工作非常简单，因为我们直接使用这些字符查询必要的值，而无需额外的处理。我们将值的顺序追加到列表中并继续进行。

```py
    mru_list = run_mru.value("MRUList").value()
    mru_order = []
    for i in mru_list:
        mru_order.append(i)
```

当我们遍历值的顺序时，我们开始构建我们的结果字典。首先，我们以与我们的`WordWheelQuery`处理器相同的方式处理时间戳，通过分配默认的`N/A`值并在我们有序列表中的第一个条目时更新它的键的最后写入时间。在此之后，我们附加一个包含相关条目的字典，例如用户名、值顺序、值名称和值内容。一旦我们处理完`Run`键中的所有剩余值，我们将返回这个字典列表。

```py
    mru_details = []
    for count, val in enumerate(mru_order):
        ts = "N/A"
        if count == 0:
            ts = run_mru.timestamp()
        mru_details.append({
            "username": username,
            "timestamp": ts,
            "order": count,
            "value_name": val,
            "run_statement": run_mru.value(val).value()
        })

    return mru_details
```

最后一个函数处理 HTML 报告的创建。这个函数首先准备代码的路径和`jinja2`环境类。这个类用于在库中存储共享资源，并且我们用它来指向库应该搜索模板文件的目录。在我们的情况下，我们希望它在当前目录中查找模板 HTML 文件，所以我们使用`os`库获取当前工作目录并将其提供给`FileSystemLoader()`类。

```py
def write_html(outfile, data_dict):
    cwd = os.path.dirname(os.path.abspath(__file__))
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(cwd))
```

在环境配置好后，我们调用我们想要使用的模板，然后使用`render()`方法创建一个带有我们传递的字典的 HTML 文件。`render`函数返回一个表示渲染的 HTML 输出的字符串，其中包含处理数据插入的结果，我们将其写入输出文件。

```py
    template = env.get_template("user_activity.html")
    rendering = template.render(nt_data=data_dict)
    with open(outfile, 'w') as open_outfile:
        open_outfile.write(rendering)
```

让我们来看一下模板文件，它像任何 HTML 文档一样以`html`、`head`和`body`标签开头。虽然我们在`head`标签中包含了脚本和样式表，但我们在这里省略了不相关的材料。这些信息可以在代码包中完整查看。

我们用一个包含处理过的数据表和部分标题的`div`开始 HTML 文档。为了简化我们需要编写的 HTML 量，我们使用一个`for`循环来收集`nt_data`值中的每个嵌套字典。`jinja2`模板语言允许我们仍然使用 Python 循环，只要它们被包裹在花括号、百分号和空格字符中。我们还可以引用对象的属性和方法，这使我们能够在不需要额外代码的情况下遍历`nt_data`字典的值。

另一个常用的模板语法显示在`h2`标签中，我们在其中访问了`main()`函数中设置的 title 属性。我们希望`jinja2`引擎解释的变量（而不是显示为字面字符串）需要用双花括号和空格字符括起来。现在这将为我们的`nt_data`字典中的每个部分打印部分标题。

```py
<html> 
<head>...</head> 
<body> 
    <div class="container"> 
        {% for nt_content in nt_data.values() %} 
            <h2>{{ nt_content['title'] }}</h2> 
```

在这个循环中，我们使用`data`标签设置我们的数据表，并创建一个新行来容纳表头。为了生成表头，我们遍历收集到的每个表头，并在嵌套的`for`循环中分配值。请注意，我们需要使用`endfor`语句指定循环的结束；这是模板引擎所要求的，因为（与 Python 不同）它对缩进不敏感：

```py
            <table class="table table-hover table-condensed"> 
                <tr> 
                    {% for header in nt_content['headers'] %} 
                        <th>{{ header }}</th> 
                    {% endfor %} 
                <tr/> 
```

在表头之后，我们进入一个单独的循环，遍历我们数据列表中的每个字典。在每个表行内，我们使用与表头相似的逻辑来创建另一个`for`循环，将每个值写入行中的单元格：

```py
                {% for entry in nt_content['data'] %} 
                    <tr> 
                        {% for header in nt_content['headers'] %} 
                            <td>{{ entry[header] }}</td> 
                        {% endfor %} 
                    </tr> 
```

现在 HTML 数据表已经填充，我们关闭当前数据点的`for`循环：我们画一条水平线，并开始编写下一个工件的数据表。一旦我们完全遍历了这些，我们关闭外部的`for`循环和我们在 HTML 报告开头打开的标签。

```py
                {% endfor %} 
            </table> 
            <br /> 
            <hr /> 
            <br /> 
        {% endfor %} 
    </div> 
</body> 
</html> 
```

我们生成的报告如下：

![](img/00102.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了以下一个或多个建议：

+   在仪表板上添加额外的`NTUser`或其他易于审查的工件，以便一目了然地提供更多有用的信息

+   使用各种 JavaScript 和 CSS 元素在仪表板上添加图表、时间轴或其他交互元素

+   从仪表板提供导出选项到 CSV 或 Excel 电子表格，并附加 JavaScript

# 缺失的链接

食谱难度：中等

Python 版本：2.7

操作系统：Linux

快捷方式文件，也称为链接文件，在操作系统平台上很常见。它们使用户可以使用一个文件引用另一个文件，该文件位于系统的其他位置。在 Windows 平台上，这些链接文件还记录了对它们引用的文件的历史访问。通常，链接文件的创建时间代表具有该名称的文件的第一次访问时间，修改时间代表具有该名称的文件的最近访问时间。利用这一点，我们可以推断出一个活动窗口，并了解这些文件是如何以及在哪里被访问的。

# 入门

此食谱需要安装三个第三方模块才能正常运行：`pytsk3`、`pyewf`和`pylnk`。有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅第八章，*使用法证证据容器* *食谱*。此脚本中使用的所有其他库都包含在 Python 的标准库中。

导航到 GitHub 存储库并下载所需版本的`pylnk`库。此处使用的是`pylnk-alpha-20170111`版本。接下来，一旦提取了发布的内容，打开终端并导航到提取的目录，执行以下命令：

```py
./synclibs.sh
./autogen.sh
sudo python setup.py install
```

要了解更多关于`pylnk`库的信息，请访问[`github.com/libyal/liblnk`](https://github.com/libyal/liblnk)。

最后，我们可以通过打开 Python 解释器，导入`pylnk`，并运行`gpylnk.get_version()`方法来检查我们的库的安装，以确保我们有正确的发布版本。

# 如何做...

此脚本将利用以下步骤：

1.  在系统中搜索所有`lnk`文件。

1.  遍历发现的`lnk`文件并提取相关属性。

1.  将所有工件写入 CSV 报告。

# 工作原理...

从导入开始，我们引入 Sleuth Kit 实用程序和`pylnk`库。我们还引入了用于参数解析、编写 CSV 报告和`StringIO`读取 Sleuth Kit 对象作为文件的库：

```py
from __future__ import print_function
from argparse import ArgumentParser
import csv
import StringIO

from utility.pytskutil import TSKUtil
import pylnk
```

此食谱的命令行处理程序接受三个位置参数，`EVIDENCE_FILE`、`IMAGE_TYPE`和`CSV_REPORT`，分别代表证据文件的路径、证据文件的类型和 CSV 报告的期望输出路径。这三个参数将传递给`main()`函数。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument('EVIDENCE_FILE', help="Path to evidence file")
    parser.add_argument('IMAGE_TYPE', help="Evidence file format",
                        choices=('ewf', 'raw'))
    parser.add_argument('CSV_REPORT', help="Path to CSV report")
    args = parser.parse_args()
    main(args.EVIDENCE_FILE, args.IMAGE_TYPE, args.CSV_REPORT)
```

`main()`函数从创建`TSKUtil`对象开始，该对象用于解释证据文件并遍历文件系统以查找以`lnk`结尾的文件。如果在系统上找不到任何`lnk`文件，则脚本会提醒用户并退出。否则，我们指定代表我们要为每个`lnk`文件存储的数据属性的列。虽然还有其他可用的属性，但这些是我们在此食谱中提取的一些更相关的属性：

```py
def main(evidence, image_type, report):
    tsk_util = TSKUtil(evidence, image_type)
    lnk_files = tsk_util.recurse_files("lnk", path="/", logic="endswith")
    if lnk_files is None:
        print("No lnk files found")
        exit(0)

    columns = [
        'command_line_arguments', 'description', 'drive_serial_number',
        'drive_type', 'file_access_time', 'file_attribute_flags',
        'file_creation_time', 'file_modification_time', 'file_size',
        'environmental_variables_location', 'volume_label',
        'machine_identifier', 'local_path', 'network_path',
        'relative_path', 'working_directory'
    ]
```

接下来，我们遍历发现的`lnk`文件，使用`open_file_as_lnk()`函数将每个文件作为文件打开。返回的对象是`pylnk`库的一个实例，可以让我们读取属性。我们使用文件的名称和路径初始化属性字典，然后遍历我们在`main()`函数中指定的列。对于每个列，我们尝试读取指定的属性值，如果无法读取，则存储“N/A”值。这些属性存储在`lnk_data`字典中，一旦提取了所有属性，就将其附加到`parsed_lnks`列表中。完成每个`lnk`文件的这个过程后，我们将此列表与输出路径和列名一起传递给`write_csv()`方法。

```py
    parsed_lnks = []
    for entry in lnk_files:
        lnk = open_file_as_lnk(entry[2])
        lnk_data = {'lnk_path': entry[1], 'lnk_name': entry[0]}
        for col in columns:
            lnk_data[col] = getattr(lnk, col, "N/A")
        lnk.close()
        parsed_lnks.append(lnk_data)

    write_csv(report, columns + ['lnk_path', 'lnk_name'], parsed_lnks)
```

要将我们的`pytsk`文件对象作为`pylink`对象打开，我们使用`open_file_as_lnk()`函数，该函数类似于本章中的其他同名函数。此函数使用`read_random()`方法和文件大小属性将整个文件读入`StringIO`缓冲区，然后将其传递给`pylnk`文件对象。以这种方式读取允许我们以文件的形式读取数据，而无需将其缓存到磁盘。一旦我们将文件加载到我们的`lnk`对象中，我们将其返回给`main()`函数：

```py
def open_file_as_lnk(lnk_file):
    file_size = lnk_file.info.meta.size
    file_content = lnk_file.read_random(0, file_size)
    file_like_obj = StringIO.StringIO(file_content)
    lnk = pylnk.file()
    lnk.open_file_object(file_like_obj)
    return lnk
```

最后一个函数是常见的 CSV 写入器，它使用`csv.DictWriter`类来遍历数据结构，并将相关字段写入电子表格。在`main()`函数中定义的列列表的顺序决定了它们在这里作为`fieldnames`参数的顺序。如果需要，可以更改该顺序，以修改它们在生成的电子表格中显示的顺序。

```py
def write_csv(outfile, fieldnames, data):
    with open(outfile, 'wb') as open_outfile:
        csvfile = csv.DictWriter(open_outfile, fieldnames)
        csvfile.writeheader()
        csvfile.writerows(data)
```

运行脚本后，我们可以在单个 CSV 报告中查看结果，如下两个屏幕截图所示。由于有许多可见列，我们选择仅显示一些以便阅读：

![](img/00103.jpeg)![](img/00104.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个建议如下：

+   添加检查以查看目标文件是否仍然存在

+   识别远程或可移动卷上的目标位置

+   添加对解析跳转列表的支持

# 四处搜寻

食谱难度：困难

Python 版本：2.7

操作系统：Linux

大多数现代操作系统都维护着系统中存储的文件和其他数据内容的索引。这些索引允许在系统卷上更有效地搜索文件格式、电子邮件和其他内容。在 Windows 上，这样的索引可以在`Windows.edb`文件中找到。这个数据库以**可扩展存储引擎**（**ESE**）文件格式存储，并位于`ProgramData`目录中。我们将利用`libyal`项目的另一个库来解析这个文件，以提取有关系统上索引内容的信息。

# 入门

此食谱需要安装四个第三方模块才能运行：`pytsk3`、`pyewf`、`pyesedb`和`unicodecsv`。有关安装`pytsk3`和`pyewf`模块的详细说明，请参阅第八章中的*使用取证证据容器* *食谱*。同样，有关安装`unicodecsv`的详细信息，请参阅*一个人的垃圾是取证人员的宝藏*食谱中的*入门*部分。此脚本中使用的所有其他库都包含在 Python 的标准库中。

转到 GitHub 存储库，并下载每个库的所需版本。此食谱是使用`libesedb-experimental-20170121`版本开发的。提取版本的内容后，打开终端，转到提取的目录，并执行以下命令：

```py
./synclibs.sh
./autogen.sh
sudo python setup.py install 
```

要了解更多关于`pyesedb`库的信息，请访问[**https://github.com/libyal/libesedb**](https://github.com/libyal/libesedb)**。**

最后，我们可以通过打开 Python 解释器，导入`pyesedb`，并运行`epyesedb.get_version()`方法来检查我们的库安装是否正确。

# 操作步骤...

起草此脚本，我们需要：

1.  递归搜索`ProgramData`目录，查找`Windows.edb`文件。

1.  遍历发现的`Windows.edb`文件（虽然实际上应该只有一个），并使用`pyesedb`库打开文件。

1.  处理每个文件以提取关键列和属性。

1.  将这些关键列和属性写入报告。

# 工作原理...

这里导入的库包括我们在本章大多数配方中使用的用于参数解析、字符串缓冲文件样对象和`TSK`实用程序的库。我们还导入`unicodecsv`库来处理 CSV 报告中的任何 Unicode 对象，`datetime`库来辅助时间戳解析，以及`struct`模块来帮助理解我们读取的二进制数据。此外，我们定义了一个全局变量`COL_TYPES`，它将`pyesedb`库中的列类型别名，用于帮助识别我们稍后在代码中将提取的数据类型：

```py
from __future__ import print_function
from argparse import ArgumentParser
import unicodecsv as csv
import datetime
import StringIO
import struct

from utility.pytskutil import TSKUtil
import pyesedb

COL_TYPES = pyesedb.column_types
```

该配方的命令行处理程序接受三个位置参数，`EVIDENCE_FILE`，`IMAGE_TYPE`和`CSV_REPORT`，它们分别表示证据文件的路径，证据文件的类型以及所需的 CSV 报告输出路径。这三个参数被传递给`main()`函数。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument('EVIDENCE_FILE', help="Path to evidence file")
    parser.add_argument('IMAGE_TYPE', help="Evidence file format",
                        choices=('ewf', 'raw'))
    parser.add_argument('CSV_REPORT', help="Path to CSV report")
    args = parser.parse_args()
    main(args.EVIDENCE_FILE, args.IMAGE_TYPE, args.CSV_REPORT)
```

`main()`函数打开证据并搜索`ProgramData`目录中的`Windows.edb`文件。如果找到一个或多个文件，我们会遍历列表并打开每个 ESE 数据库，以便使用`process_windows_search()`函数进行进一步处理。该函数返回要使用的电子表格列标题以及包含报告中要包含的数据的字典列表。然后将此信息写入输出 CSV，供`write_csv()`方法审查：

```py
def main(evidence, image_type, report):
    tsk_util = TSKUtil(evidence, image_type)
    esedb_files = tsk_util.recurse_files(
        "Windows.edb",
        path="/ProgramData/Microsoft/Search/Data/Applications/Windows",
        logic="equals"
    )
    if esedb_files is None:
        print("No Windows.edb file found")
        exit(0)

    for entry in esedb_files:
        ese = open_file_as_esedb(entry[2])
        if ese is None:
            continue # Invalid ESEDB
        report_cols, ese_data = process_windows_search(ese)

    write_csv(report, report_cols, ese_data)
```

读取响应的 ESE 数据库需要`open_file_as_esedb()`函数。此代码块使用与之前配方类似的逻辑，将文件读入`StringIO`对象并使用库打开文件样对象。请注意，如果文件相当大或您的计算机内存较少，这可能会在您的系统上引发错误。您可以使用内置的`tempfile`库将文件缓存到磁盘上的临时位置，然后从那里读取，如果您愿意的话。

```py
def open_file_as_esedb(esedb):
    file_size = esedb.info.meta.size
    file_content = esedb.read_random(0, file_size)
    file_like_obj = StringIO.StringIO(file_content)
    esedb = pyesedb.file()
    try:
        esedb.open_file_object(file_like_obj)
    except IOError:
        return None
    return esedb
```

我们的`process_windows_search()`函数从列定义开始。虽然我们之前的配方使用了一个简单的列列表，但`pyesedb`库需要一个列索引作为输入，以从表中的行中检索值。因此，我们的列列表必须由元组组成，其中第一个元素是数字（索引），第二个元素是字符串描述。由于描述在函数中未用于选择列，我们将其命名为我们希望它们在报告中显示的方式。对于本配方，我们已定义了以下列索引和名称：

```py
def process_windows_search(ese):
    report_cols = [
        (0, "DocID"), (286, "System_KindText"),
        (35, "System_ItemUrl"), (5, "System_DateModified"),
        (6, "System_DateCreated"), (7, "System_DateAccessed"),
        (3, "System_Size"), (19, "System_IsFolder"),
        (2, "System_Search_GatherTime"), (22, "System_IsDeleted"),
        (61, "System_FileOwner"), (31, "System_ItemPathDisplay"),
        (150, "System_Link_TargetParsingPath"),
        (265, "System_FileExtension"), (348, "System_ComputerName"),
        (34, "System_Communication_AccountName"),
        (44, "System_Message_FromName"),
        (43, "System_Message_FromAddress"), (49, "System_Message_ToName"),
        (47, "System_Message_ToAddress"),
        (62, "System_Message_SenderName"),
        (189, "System_Message_SenderAddress"),
        (52, "System_Message_DateSent"),
        (54, "System_Message_DateReceived")
    ]
```

在我们定义感兴趣的列之后，我们访问`SystemIndex_0A`表，其中包含索引文件、邮件和其他条目。我们遍历表中的记录，为每个记录构建一个`record_info`字典，其中包含每个记录的列值，最终将其附加到`table_data`列表中。第二个循环遍历我们之前定义的列，并尝试提取每个记录中的列的值和值类型。

```py
    table = ese.get_table_by_name("SystemIndex_0A")
    table_data = []
    for record in table.records:
        record_info = {}
        for col_id, col_name in report_cols:
            rec_val = record.get_value_data(col_id)
            col_type = record.get_column_type(col_id)
```

使用我们之前定义的`COL_TYPES`全局变量，我们可以引用各种数据类型，并确保我们正确解释值。以下代码块中的逻辑侧重于根据其数据类型正确解释值。首先，我们处理日期，日期可能存储为 Windows `FILETIME`值。我们尝试转换`FILETIME`值（如果可能），或者如果不可能，则以十六进制呈现日期值。接下来的语句检查文本值，使用`pyesedb`的`get_value_data_as_string()`函数或作为 UTF-16 大端，并替换任何未识别的字符以确保完整性。

然后，我们使用`pyesedb`的`get_value_data_as_integer()`函数和一个简单的比较语句分别处理整数和布尔数据类型的解释。具体来说，我们检查`rec_val`是否等于`"\x01"`，并允许根据该比较将`rec_val`设置为`True`或`False`。如果这些数据类型都不合法，我们将该值解释为十六进制，并在将该值附加到表之前将其与相关列名一起存储：

```py
            if col_type in (COL_TYPES.DATE_TIME, COL_TYPES.BINARY_DATA):
                try:
                    raw_val = struct.unpack('>q', rec_val)[0]
                    rec_val = parse_windows_filetime(raw_val)
                except Exception:
                    if rec_val is not None:
                        rec_val = rec_val.encode('hex')

            elif col_type in (COL_TYPES.TEXT, COL_TYPES.LARGE_TEXT):
                try:
                    rec_val = record.get_value_data_as_string(col_id)
                except Exception:
                    rec_val = rec_val.decode("utf-16-be", "replace")

            elif col_type == COL_TYPES.INTEGER_32BIT_SIGNED:
                rec_val = record.get_value_data_as_integer(col_id)

            elif col_type == COL_TYPES.BOOLEAN:
                rec_val = rec_val == '\x01'

            else:
                if rec_val is not None:
                    rec_val = rec_val.encode('hex')

            record_info[col_name] = rec_val
        table_data.append(record_info)
```

然后，我们将一个元组返回给我们的调用函数，其中第一个元素是`report_cols`字典中列的名称列表，第二个元素是数据字典的列表。

```py
    return [x[1] for x in report_cols], table_data
```

借鉴我们在第七章中日期解析食谱中的逻辑，*基于日志的工件食谱*，我们实现了一个将 Windows `FILETIME`值解析为可读状态的函数。这个函数接受一个整数值作为输入，并返回一个可读的字符串：

```py
def parse_windows_filetime(date_value):
    microseconds = float(date_value) / 10
    ts = datetime.datetime(1601, 1, 1) + datetime.timedelta(
        microseconds=microseconds)
    return ts.strftime('%Y-%m-%d %H:%M:%S.%f')
```

最后一个函数是 CSV 报告编写器，它使用`DictWriter`类将收集到的信息的列和行写入到打开的 CSV 电子表格中。虽然我们在一开始选择了一部分可用的列，但还有许多可供选择的列，可能对不同的案例类型有用。因此，我们建议查看所有可用的列，以更好地理解这个食谱，以及哪些列对您可能有用或无用。

```py
def write_csv(outfile, fieldnames, data):
    with open(outfile, 'wb') as open_outfile:
        csvfile = csv.DictWriter(open_outfile, fieldnames)
        csvfile.writeheader()
        csvfile.writerows(data)
```

运行食谱后，我们可以查看这里显示的输出 CSV。由于这份报告有很多列，我们在接下来的两个屏幕截图中突出显示了一些有趣的列：

![](img/00105.jpeg)![](img/00106.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个以下建议：

+   添加支持以检查引用文件和文件夹的存在。

+   使用 Python 的`tempfile`库将我们的`Windows.edb`文件写入临时位置，以减轻解析大型数据库时的内存压力

+   在表中添加更多列或创建单独的（有针对性的）报告，使用表中超过 300 个可用列中的更多列
