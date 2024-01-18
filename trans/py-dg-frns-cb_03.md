# 深入移动取证食谱

本章涵盖以下食谱：

+   解析PLIST文件

+   处理SQLite数据库

+   识别SQLite数据库中的间隙

+   处理iTunes备份

+   将Wi-Fi标记在地图上

+   深入挖掘以恢复消息

# 介绍

也许这已经成为陈词滥调，但事实仍然如此，随着技术的发展，它继续与我们的生活更加紧密地融合。这从未如此明显，如第一部智能手机的发展。这些宝贵的设备似乎永远不会离开其所有者，并且通常比人类伴侣更多地接触。因此，毫不奇怪，智能手机可以为调查人员提供大量关于其所有者的见解。例如，消息可能提供有关所有者心态或特定事实的见解。它们甚至可能揭示以前未知的信息。位置历史是我们可以从这些设备中提取的另一个有用的证据，可以帮助验证个人的不在场证明。我们将学习提取这些信息以及更多内容。

智能手机上证据价值的常见来源是SQLite数据库。这些数据库在大多数智能手机操作系统中作为应用程序的事实存储。因此，本章中的许多脚本将专注于从这些数据库中提取数据并推断。除此之外，我们还将学习如何处理PLIST文件，这些文件通常与苹果操作系统一起使用，包括iOS，并提取相关数据。本章中的脚本专注于解决特定问题，并按复杂性排序：

+   学习处理XML和二进制PLIST文件

+   使用Python与SQLite数据库交互

+   识别SQLite数据库中的缺失间隙

+   将iOS备份转换为人类可读格式

+   处理Cellebrite的输出并执行Wi-Fi MAC地址地理位置查找

+   从SQLite数据库中识别潜在完整的已删除内容

访问[www.packtpub.com/books/content/support](http://www.packtpub.com/books/content/support)下载本章的代码包。

# 解析PLIST文件

食谱难度：简单

Python版本：2.7或3.5

操作系统：任何

这个食谱将处理每个iOS备份中存在的`Info.plist`文件，并提取设备特定信息，如设备名称、IMEI、序列号、产品制造、型号和iOS版本，以及最后备份日期。属性列表，或PLIST，有两种不同的格式：XML或二进制。通常，在处理二进制PLIST时，需要在macOS平台上使用plutil实用程序将其转换为可读的XML格式。然而，我们将介绍一个处理两种类型的Python库，即可轻松处理。一旦我们从`Info.plist`文件中提取相关数据元素，我们将把这些数据打印到控制台上。

# 入门

此食谱需要安装第三方库`biplist`。此脚本中使用的所有其他库都包含在Python的标准库中。`biplist`模块提供了处理XML和二进制PLIST文件的方法。

要了解更多关于`biplist`库的信息，请访问[https://github.com/wooster/biplist](https://github.com/wooster/biplist)。

Python有一个内置的PLIST库，`plistlib`；然而，发现这个库不像`biplist`那样广泛支持二进制PLIST文件。

要了解更多关于`plistlib`库的信息，请访问[https://docs.python.org/3/library/plistlib.html](https://docs.python.org/3/library/plistlib.html)。

使用`pip`可以完成安装`biplist`：

```py
pip install biplist==1.0.2
```

确保获取自己的`Info.plist`文件以便使用此脚本进行处理。如果找不到`Info.plist`文件，任何PLIST文件都应该合适。我们的脚本并不那么具体，理论上应该适用于任何PLIST文件。

# 如何做…

我们将采用以下步骤处理PLIST文件：

1.  打开输入的PLIST文件。

1.  将PLIST数据读入变量。

1.  将格式化的PLIST数据打印到控制台。

# 它是如何工作的...

首先，我们导入所需的库来处理参数解析和处理PLIST文件：

```py
from __future__ import print_function
import argparse
import biplist
import os
import sys
```

该配方的命令行处理程序接受一个位置参数`PLIST_FILE`，表示我们将处理的PLIST文件的路径：

```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("PLIST_FILE", help="Input PList File")
    args = parser.parse_args()
```

我们使用`os.exists()`和`os.path.isfile()`函数来验证输入文件是否存在并且是一个文件，而不是一个目录。我们不对这个文件进行进一步的验证，比如确认它是一个PLIST文件而不是一个文本文件，而是依赖于`biplist`库（和常识）来捕捉这样的错误。如果输入文件通过了我们的测试，我们调用`main()`函数并将PLIST文件路径传递给它：

```py
    if not os.path.exists(args.PLIST_FILE) or \
            not os.path.isfile(args.PLIST_FILE):
        print("[-] {} does not exist or is not a file".format(
            args.PLIST_FILE))
        sys.exit(1)

    main(args.PLIST_FILE)
```

`main()`函数相对简单，实现了读取PLIST文件然后将数据打印到控制台的目标。首先，我们在控制台上打印一个更新，表示我们正在尝试打开文件。然后，我们使用`biplist.readPlist()`方法打开并读取PLIST到我们的`plist_data`变量中。如果PLIST文件损坏或无法访问，`biplist`会引发`InvalidPlistException`或`NotBinaryPlistException`错误。我们在`try`和`except`块中捕获这两种错误，并相应地`exit`脚本：

```py
def main(plist):
    print("[+] Opening {} file".format(plist))
    try:
        plist_data = biplist.readPlist(plist)
    except (biplist.InvalidPlistException,
            biplist.NotBinaryPlistException) as e:
        print("[-] Invalid PLIST file - unable to be opened by biplist")
        sys.exit(2)
```

一旦我们成功读取了PLIST数据，我们遍历结果中的`plist_data`字典中的键，并将它们打印到控制台上。请注意，我们打印`Info.plist`文件中除了`Applications`和`iTunes Files`键之外的所有键。这两个键包含大量数据，会淹没控制台，因此不适合这种类型的输出。我们使用format方法来帮助创建可读的控制台输出：

```py
    print("[+] Printing Info.plist Device "
          "and User Information to Console\n")
    for k in plist_data:
        if k != 'Applications' and k != 'iTunes Files':
            print("{:<25s} - {}".format(k, plist_data[k]))
```

请注意第一个花括号中的额外格式化字符。我们在这里指定左对齐输入字符串，并且宽度为25个字符。正如你在下面的截图中所看到的，这确保了数据以有序和结构化的格式呈现：

![](../images/00022.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一些建议：

+   而不是将数据打印到控制台，添加一个CSV函数将数据写入CSV文件

+   添加支持处理一个目录中的所有PLIST文件

# 处理SQLite数据库

配方难度：简单

Python版本：3.5

操作系统：任何

如前所述，SQLite数据库是移动设备上的主要数据存储库。Python有一个内置的`sqlite3`库，可以用来与这些数据库进行交互。在这个脚本中，我们将与iPhone的`sms.db`文件交互，并从`message`表中提取数据。我们还将利用这个脚本的机会介绍`csv`库，并将消息数据写入电子表格。

要了解更多关于`sqlite3`库的信息，请访问[https://docs.python.org/3/library/sqlite3.html](https://docs.python.org/3/library/sqlite3.html)。

# 入门

此脚本中使用的所有库都包含在Python的标准库中。对于这个脚本，请确保有一个`sms.db`文件可以进行查询。通过一些小的修改，你可以使用这个脚本与任何数据库；然而，我们将特别讨论它与iOS 10.0.1设备的iPhone短信数据库相关。

# 如何做到...

该配方遵循以下基本原则：

1.  连接到输入数据库。

1.  查询表PRAGMA以提取列名。

1.  获取所有表内容。

1.  将所有表内容写入CSV。

# 它是如何工作的...

首先，我们导入所需的库来处理参数解析、写入电子表格和与SQLite数据库交互：

```py
from __future__ import print_function
import argparse
import csv
import os
import sqlite3
import sys
```

该配方的命令行处理程序接受两个位置参数`SQLITE_DATABASE`和`OUTPUT_CSV`，分别表示输入数据库和期望的CSV输出的文件路径：

```py
if __name__ == '__main__':
    # Command-line Argument Parser
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("SQLITE_DATABASE", help="Input SQLite database")
    parser.add_argument("OUTPUT_CSV", help="Output CSV File")
    args = parser.parse_args()
```

接下来，我们使用`os.dirname()`方法仅提取输出文件的目录路径。我们这样做是为了检查输出目录是否已经存在。如果不存在，我们使用`os.makedirs()`方法创建输出路径中尚不存在的每个目录。这样可以避免以后尝试将输出CSV写入不存在的目录时出现问题：

```py
    directory = os.path.dirname(args.OUTPUT_CSV)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
```

一旦我们验证了输出目录存在，我们将提供的参数传递给`main()`函数：

```py
    main(args.SQLITE_DATABASE, args.OUTPUT_CSV)
```

`main()`函数向用户的控制台打印状态更新，然后检查输入文件是否存在且是否为文件。如果不存在，我们使用`sys.exit()`方法退出脚本，使用大于0的值指示脚本由于错误退出：

```py
def main(database, out_csv):
    print("[+] Attempting connection to {} database".format(database))
    if not os.path.exists(database) or not os.path.isfile(database):
        print("[-] Database does not exist or is not a file")
        sys.exit(1)
```

接下来，我们使用`sqlite3.conn()`方法连接到输入数据库。重要的是要注意，`sqlite3.conn()`方法会打开所提供名称的数据库，无论它是否存在。因此，重要的是在尝试打开连接之前检查文件是否存在。否则，我们可能会创建一个空数据库，在与其交互时可能会导致脚本出现问题。一旦建立了连接，我们需要创建一个`Cursor`对象来与数据库交互：

```py
    # Connect to SQLite Database
    conn = sqlite3.connect(database)
    c = conn.cursor()
```

现在，我们可以使用`Cursor`对象的`execute()`命令对数据库执行查询。此时，我们传递给execute函数的字符串只是标准的SQLlite查询。在大多数情况下，您可以运行与与SQLite数据库交互时通常运行的任何查询。从给定命令返回的结果存储在`Cursor`对象中。我们需要使用`fetchall()`方法将结果转储到我们可以操作的变量中：

```py
    # Query DB for Column Names and Data of Message Table
    c.execute("pragma table_info(message)")
    table_data = c.fetchall()
    columns = [x[1] for x in table_data]
```

`fetchall()`方法返回一组结果的元组。每个元组的第一个索引中存储了每列的名称。通过使用列表推导，我们将`message`表的列名存储到列表中。这在稍后将数据结果写入CSV文件时会发挥作用。在获取了`message`表的列名后，我们直接查询该表的所有数据，并将其存储在`message_data`变量中：

```py
    c.execute("select * from message")
    message_data = c.fetchall()
```

提取数据后，我们向控制台打印状态消息，并将输出的CSV和消息表列和数据传递给`write_csv()`方法：

```py
    print("[+] Writing Message Content to {}".format(out_csv))
    write_csv(out_csv, columns, message_data)
```

您会发现大多数脚本最终都会将数据写入CSV文件。这样做有几个原因。在Python中编写CSV非常简单，对于大多数数据集，可以用几行代码完成。此外，将数据放入电子表格中可以根据列进行排序和过滤，以帮助总结和理解大型数据集。

在开始写入CSV文件之前，我们使用`open()`方法创建文件对象及其别名`csvfile`。打开此文件的方式取决于您是否使用Python 2.x或Python 3.x。对于Python 2.x，您以`wb`模式打开文件，而不使用newline关键字参数。对于Python 3.x，您可以以`w`模式打开文件，并将newline关键字设置为空字符串。在可能的情况下，代码是针对Python 3.x编写的，因此我们使用后者。未以这种方式打开文件对象会导致输出的CSV文件在每行之间包含一个空行。

打开文件对象后，我们将其传递给`csv.writer()`方法。我们可以使用该对象的`writerow()`和`writerows()`方法分别写入列标题列表和元组列表。顺便说一句，我们可以遍历`msgs`列表中的每个元组，并为每个元组调用`writerow()`。`writerows()`方法消除了不必要的循环，并在这里使用：

```py
def write_csv(output, cols, msgs):
    with open(output, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(cols)
        csv_writer.writerows(msgs)
```

当我们运行此脚本时，会看到以下控制台消息。在CSV中，我们可以收集有关发送和接收的消息的详细信息，以及包括日期、错误、来源等在内的有趣的元数据：

![](../images/00023.jpeg)![](../images/00024.jpeg)

# 识别 SQLite 数据库中的间隙

食谱难度：简单

Python 版本：2.7 或 3.5

操作系统：任意

这个食谱将演示如何通过编程方式使用主键来识别给定表中的缺失条目。这种技术允许我们识别数据库中不再有效的记录。我们将使用这个方法来识别从 iPhone 短信数据库中删除了哪些消息以及删除了多少条消息。然而，这也适用于使用自增主键的任何表。

要了解更多关于 SQLite 表和主键的信息，请访问 [https://www.sqlite.org/lang_createtable.html](https://www.sqlite.org/lang_createtable.html)。

管理 SQLite 数据库及其表的一个基本概念是主键。主键通常是表中特定行的唯一整数列。常见的实现是自增主键，通常从第一行开始为 `1`，每一行递增 `1`。当从表中删除行时，主键不会改变以适应或重新排序表。

例如，如果我们有一个包含 10 条消息的数据库，并删除了消息 `4` 到 `6`，那么主键列中将会有一个从 `3` 到 `7` 的间隙。通过我们对自增主键的理解，我们可以推断消息 `4` 到 `6` 曾经存在，但现在不再是数据库中的有效条目。通过这种方式，我们可以量化数据库中不再有效的消息数量以及与之相关的主键值。我们将在后续的食谱 *深入挖掘以恢复消息* 中使用这个信息，然后去寻找这些条目，以确定它们是否完整且可恢复。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。这个食谱需要一个数据库来运行。在这个例子中，我们将使用 iPhone `sms.db` 数据库。

# 如何做...

在这个食谱中，我们将执行以下步骤：

1.  连接到输入数据库。

1.  查询表 PRAGMA 以识别表的主键。

1.  获取所有主键值。

1.  计算并在控制台上显示表中的间隙。

# 工作原理...

首先，我们导入所需的库来处理参数解析和与 SQLite 数据库交互：

```py
from __future__ import print_function
import argparse
import os
import sqlite3
import sys
```

这个食谱的命令行处理程序接受两个位置参数，`SQLITE_DATABASE` 和 `TABLE`，分别表示输入数据库的路径和要查看的表的名称。一个可选参数 `column`，由破折号表示，可以用来手动提供主键列（如果已知）：

```py
if __name__ == "__main__":
    # Command-line Argument Parser
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("SQLITE_DATABASE", help="Input SQLite database")
    parser.add_argument("TABLE", help="Table to query from")
    parser.add_argument("--column", help="Optional column argument")
    args = parser.parse_args()
```

如果提供了可选的列参数，我们将它作为关键字参数与数据库和表名一起传递给 `main()` 函数。否则，我们只将数据库和表名传递给 `main()` 函数，而不包括 `col` 关键字参数：

```py
    if args.column is not None:
        main(args.SQLITE_DATABASE, args.TABLE, col=args.column)
    else:
        main(args.SQLITE_DATABASE, args.TABLE)
```

`main()` 函数，与前一个食谱一样，首先执行一些验证，验证输入数据库是否存在且是一个文件。因为我们在这个函数中使用了关键字参数，所以我们必须在函数定义中使用 `**kwargs` 参数来指示这一点。这个参数充当一个字典，存储所有提供的关键字参数。在这种情况下，如果提供了可选的列参数，这个字典将包含一个 `col` 键值对：

```py
def main(database, table, **kwargs):
    print("[+] Attempting connection to {} database".format(database))
    if not os.path.exists(database) or not os.path.isfile(database):
        print("[-] Database does not exist or is not a file")
        sys.exit(1)
```

在验证输入文件后，我们使用 `sqlite3` 连接到这个数据库，并创建我们用来与之交互的 `Cursor` 对象：

```py
    # Connect to SQLite Database
    conn = sqlite3.connect(database)
    c = conn.cursor()
```

为了确定所需表的主键，我们使用带有插入括号的表名的`pragma table_info`命令。我们使用`format()`方法动态地将表的名称插入到否则静态的字符串中。在我们将命令的结果存储在`table_data`变量中后，我们对表名输入进行验证。如果用户提供了一个不存在的表名，我们将得到一个空列表作为结果。我们检查这一点，如果表不存在，就退出脚本。

```py
    # Query Table for Primary Key
    c.execute("pragma table_info({})".format(table))
    table_data = c.fetchall()
    if table_data == []:
        print("[-] Check spelling of table name - '{}' did not return "
              "any results".format(table))
        sys.exit(2)
```

在这一点上，我们为脚本的其余部分创建了一个`if-else`语句，具体取决于用户是否提供了可选的列参数。如果`col`是`kwargs`字典中的一个键，我们立即调用`find_gaps()`函数，并将`Cursor`对象`c`、表名和用户指定的主键列名传递给它。否则，我们尝试在`table_data`变量中识别主键。

先前在`table_data`变量中执行并存储的命令为给定表中的每一列返回一个元组。每个元组的最后一个元素是`1`或`0`之间的二进制选项，其中`1`表示该列是主键。我们遍历返回的元组中的每个最后一个元素，如果它们等于`1`，则将元组的索引一中存储的列名附加到`potential_pks`列表中。

```py
    if "col" in kwargs:
        find_gaps(c, table, kwargs["col"])

    else:
        # Add Primary Keys to List
        potential_pks = []
        for row in table_data:
            if row[-1] == 1:
                potential_pks.append(row[1])
```

一旦我们确定了所有的主键，我们检查列表以确定是否存在零个或多个键。如果存在这些情况中的任何一种，我们会提醒用户并退出脚本。在这些情况下，用户需要指定哪一列应被视为主键列。如果列表包含单个主键，我们将该列的名称与数据库游标和表名一起传递给`find_gaps()`函数。

```py
        if len(potential_pks) != 1:
            print("[-] None or multiple primary keys found -- please "
                  "check if there is a primary key or specify a specific "
                  "key using the --column argument")
            sys.exit(3)

        find_gaps(c, table, potential_pks[0])
```

`find_gaps()`方法首先通过在控制台显示一条消息来提醒用户脚本的当前执行状态。我们尝试在`try`和`except`块中进行数据库查询。如果用户指定的列不存在或拼写错误，我们将从`sqlite3`库接收到`OperationalError`。这是用户提供的参数的最后验证步骤，如果触发了except块，脚本将退出。如果查询成功执行，我们获取所有数据并将其存储在`results`变量中。

```py
def find_gaps(db_conn, table, pk):
    print("[+] Identifying missing ROWIDs for {} column".format(pk))
    try:
        db_conn.execute("select {} from {}".format(pk, table))
    except sqlite3.OperationalError:
        print("[-] '{}' column does not exist -- "
              "please check spelling".format(pk))
        sys.exit(4)
    results = db_conn.fetchall()
```

我们使用列表推导和内置的`sorted()`函数来创建排序后的主键列表。`results`列表包含索引`0`处的一个元素的元组，即主键，对于`sms.db`的`message`表来说，就是名为ROWID的列。有了排序后的ROWID列表，我们可以快速计算表中缺少的条目数。这将是最近的ROWID减去列表中存在的ROWID数。如果数据库中的所有条目都是活动的，这个值将为零。

我们假设最近的ROWID是实际最近的ROWID。有可能删除最后几个条目，而配方只会将最近的活动条目检测为最高的ROWID。

```py
    rowids = sorted([x[0] for x in results])
    total_missing = rowids[-1] - len(rowids)
```

如果列表中没有缺少任何值，我们将这一幸运的消息打印到控制台，并以`0`退出，表示成功终止。另一方面，如果我们缺少条目，我们将其打印到控制台，并显示缺少条目的计数。

```py
    if total_missing == 0:
        print("[*] No missing ROWIDs from {} column".format(pk))
        sys.exit(0)
    else:
        print("[+] {} missing ROWID(s) from {} column".format(
            total_missing, pk))
```

为了计算缺失的间隙，我们使用`range()`方法生成从第一个ROWID到最后一个ROWID的所有ROWIDs的集合，然后将其与我们拥有的排序列表进行比较。`difference()`函数可以与集合一起使用，返回一个新的集合，其中包含第一个集合中不在括号中的对象中的元素。然后我们将识别的间隙打印到控制台，这样脚本的执行就完成了。

```py
    # Find Missing ROWIDs
    gaps = set(range(rowids[0], rowids[-1] + 1)).difference(rowids)
    print("[*] Missing ROWIDS: {}".format(gaps))
```

此脚本的输出示例可能如下截图所示。请注意，控制台可以根据已删除消息的数量迅速变得混乱。然而，这并不是此脚本的预期结束。我们将在本章后面的更高级的食谱“深入挖掘以恢复消息”中使用此脚本的逻辑，来识别并尝试定位潜在可恢复的消息：

![](../images/00025.jpeg)

# 另请参阅

有关SQLite数据库结构和主键的更多信息，请参阅其广泛的文档[https://www.sqlite.org/](https://www.sqlite.org/)。

# 处理iTunes备份

食谱难度：简单

Python版本：2.7或3.5

操作系统：任何

在这个食谱中，我们将把未加密的iTunes备份转换成人类可读的格式，这样我们就可以轻松地探索其内容，而无需任何第三方工具。备份文件可以在主机计算机的`MobileSync\Backup`文件夹中找到。

有关Windows和OS X默认iTunes备份位置的详细信息，请访问[https://support.apple.com/en-us/HT204215](https://support.apple.com/en-us/HT204215)。

如果苹果产品已备份到计算机上，将会有许多文件夹，其名称是表示备份文件夹中特定设备的GUID。这些文件夹包含了一段时间内每个设备的差异备份。

在iOS 10中引入的新备份格式中，文件存储在包含文件名前两个十六进制字符的子文件夹中。每个文件的名称都是设备上路径的`SHA-1`哈希。在设备的备份文件夹的根目录中，有一些感兴趣的文件，例如我们之前讨论过的`Info.plist`文件和`Manifest.db`数据库。此数据库存储了每个备份文件的详细信息，包括其`SHA-1`哈希、文件路径和名称。我们将使用这些信息来使用人类友好的名称重新创建本机备份文件夹结构。

# 入门

此脚本中使用的所有库都包含在Python的标准库中。要跟随操作，您需要获取一个未加密的iTunes备份文件进行操作。确保备份文件是较新的iTunes备份格式（iOS 10+），与之前描述的内容相匹配。

# 如何做...

我们将使用以下步骤来处理此食谱中的iTunes备份：

1.  识别`MobileSync\Backup`文件夹中的所有备份。

1.  遍历每个备份。

1.  读取Manifest.db文件，并将`SHA-1`哈希名称与文件名关联起来。

1.  将备份文件复制并重命名到输出文件夹，使用适当的文件路径和名称。

# 工作原理...

首先，我们导入所需的库来处理参数解析、日志记录、文件复制和与SQLite数据库交互。我们还设置了一个变量，用于稍后构建食谱的日志记录组件：

```py
from __future__ import print_function
import argparse
import logging
import os
from shutil import copyfile
import sqlite3
import sys

logger = logging.getLogger(__name__)
```

此食谱的命令行处理程序接受两个位置参数，`INPUT_DIR`和`OUTPUT_DIR`，分别表示iTunes备份文件夹和所需的输出文件夹。可以提供一个可选参数来指定日志文件的位置和日志消息的冗长程度。

```py
if __name__ == "__main__":
    # Command-line Argument Parser
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument(
        "INPUT_DIR",
        help="Location of folder containing iOS backups, "
        "e.g. ~\Library\Application Support\MobileSync\Backup folder"
    )
    parser.add_argument("OUTPUT_DIR", help="Output Directory")
    parser.add_argument("-l", help="Log file path",
                        default=__file__[:-2] + "log")
    parser.add_argument("-v", help="Increase verbosity",
                        action="store_true")
    args = parser.parse_args()
```

接下来，我们开始为此食谱设置日志。我们检查用户是否提供了可选的冗长参数，如果有，我们将将级别从“INFO”增加到“DEBUG”：

```py
    if args.v:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
```

对于此日志，我们设置消息格式并为控制台和文件输出配置处理程序，并将它们附加到我们定义的`logger`：

```py
    msg_fmt = logging.Formatter("%(asctime)-15s %(funcName)-13s"
                                "%(levelname)-8s %(message)s")
    strhndl = logging.StreamHandler(sys.stderr)
    strhndl.setFormatter(fmt=msg_fmt)
    fhndl = logging.FileHandler(args.l, mode='a')
    fhndl.setFormatter(fmt=msg_fmt)

    logger.addHandler(strhndl)
    logger.addHandler(fhndl)
```

设置好日志文件后，我们向日志记录一些调试详细信息，包括提供给此脚本的参数以及有关主机和Python版本的详细信息。我们排除了`sys.argv`列表的第一个元素，这是脚本的名称，而不是提供的参数之一：

```py
    logger.info("Starting iBackup Visualizer")
    logger.debug("Supplied arguments: {}".format(" ".join(sys.argv[1:])))
    logger.debug("System: " + sys.platform)
    logger.debug("Python Version: " + sys.version)
```

使用`os.makedirs()`函数，如果必要，我们将为所需的输出目录创建任何必要的文件夹，如果它们尚不存在：

```py
    if not os.path.exists(args.OUTPUT_DIR):
        os.makedirs(args.OUTPUT_DIR)
```

最后，如果输入目录存在并且确实是一个目录，我们将提供的输入和输出目录传递给`main()`函数。如果输入目录未通过验证，我们将在退出脚本之前向控制台打印错误并记录：

```py
    if os.path.exists(args.INPUT_DIR) and os.path.isdir(args.INPUT_DIR):
        main(args.INPUT_DIR, args.OUTPUT_DIR)
    else:
        logger.error("Supplied input directory does not exist or is not "
                     "a directory")
        sys.exit(1)
```

`main()`函数首先调用`backup_summary()`函数来识别输入文件夹中存在的所有备份。在继续`main()`函数之前，让我们先看看`backup_summary()`函数并了解它的作用：

```py
def main(in_dir, out_dir):
    backups = backup_summary(in_dir)
```

`backup_summary()`函数使用`os.listdir()`方法列出输入目录的内容。我们还实例化`backups`字典，用于存储每个发现的备份的详细信息：

```py
def backup_summary(in_dir):
    logger.info("Identifying all iOS backups in {}".format(in_dir))
    root = os.listdir(in_dir)
    backups = {}
```

对于输入目录中的每个项目，我们使用`os.path.join()`方法与输入目录和项目。然后我们检查这是否是一个目录，而不是一个文件，以及目录的名称是否为40个字符长。如果目录通过了这些检查，这很可能是一个备份目录，因此我们实例化两个变量来跟踪备份中文件的数量和这些文件的总大小：

```py
    for x in root:
        temp_dir = os.path.join(in_dir, x)
        if os.path.isdir(temp_dir) and len(x) == 40:
            num_files = 0
            size = 0
```

我们使用[第1章](part0029.html#RL0A0-260f9401d2714cb9ab693c4692308abe)中讨论的`os.walk()`方法，并为备份文件夹下的根目录、子目录和文件创建列表。因此，我们可以使用文件列表的长度，并在迭代备份文件夹时继续将其添加到`num_files`变量中。类似地，我们使用一个巧妙的一行代码将每个文件的大小添加到`size`变量中：

```py
            for root, subdir, files in os.walk(temp_dir):
                num_files += len(files)
                size += sum(os.path.getsize(os.path.join(root, name))
                            for name in files)
```

在我们完成对备份的迭代之后，我们使用备份的名称作为键将备份添加到`backups`字典中，并将备份文件夹路径、文件计数和大小作为值存储。一旦我们完成了所有备份的迭代，我们将这个字典返回给`main()`函数。让我们接着来看：

```py
            backups[x] = [temp_dir, num_files, size]

    return backups
```

在`main()`函数中，如果找到了任何备份，我们将每个备份的摘要打印到控制台。对于每个备份，我们打印一个任意的标识备份的数字，备份的名称，文件数量和大小。我们使用`format()`方法并手动指定换行符(`\n`)来确保控制台保持可读性：

```py
    print("Backup Summary")
    print("=" * 20)
    if len(backups) > 0:
        for i, b in enumerate(backups):
            print("Backup No.: {} \n"
                  "Backup Dev. Name: {} \n"
                  "# Files: {} \n"
                  "Backup Size (Bytes): {}\n".format(
                      i, b, backups[b][1], backups[b][2])
                  )
```

接下来，我们使用`try-except`块将`Manifest.db`文件的内容转储到`db_items`变量中。如果找不到`Manifest.db`文件，则识别的备份文件夹可能是旧格式或无效的，因此我们使用`continue`命令跳过它。让我们简要讨论一下`process_manifest()`函数，它使用`sqlite3`连接到并提取`Manifest.db`文件表中的所有数据：

```py
            try:
                db_items = process_manifest(backups[b][0])
            except IOError:
                logger.warn("Non-iOS 10 backup encountered or "
                            "invalid backup. Continuing to next backup.")
                continue
```

`process_manifest()` 方法以备份的目录路径作为唯一输入。对于这个输入，我们连接`Manifest.db`字符串，表示这个数据库应该存在在一个有效的备份中的位置。如果发现这个文件不存在，我们记录这个错误并向`main()`函数抛出一个`IOError`，正如我们刚才讨论的那样，这将导致在控制台上打印一条消息，并继续下一个备份：

```py
def process_manifest(backup):
    manifest = os.path.join(backup, "Manifest.db")

    if not os.path.exists(manifest):
        logger.error("Manifest DB not found in {}".format(manifest))
        raise IOError
```

如果文件确实存在，我们连接到它，并使用`sqlite3`创建`Cursor`对象。`items`字典使用每个条目在`Files`表中的`SHA-1`哈希作为键，并将所有其他数据存储为列表中的值。请注意，这里有一种替代方法来访问查询结果，而不是在以前的示例中使用的`fetchall()`函数。在我们从`Files`表中提取了所有数据之后，我们将字典返回给`main()`函数：

```py
    conn = sqlite3.connect(manifest)
    c = conn.cursor()
    items = {}
    for row in c.execute("SELECT * from Files;"):
        items[row[0]] = [row[2], row[1], row[3]]

    return items
```

回到`main()`函数，我们立即将返回的字典，现在称为`db_items`，传递给`create_files()`方法。我们刚刚创建的字典将被下一个函数用来执行对文件`SHA-1`哈希的查找，并确定其真实文件名、扩展名和本地文件路径。`create_files()`函数执行这些查找，并将备份文件复制到输出文件夹，并使用适当的路径、名称和扩展名。

`else`语句处理了`backup_summary()`函数未找到备份的情况。我们提醒用户应该是适当的输入文件夹，并退出脚本。这完成了`main()`函数；现在让我们继续进行`create_files()`方法：

```py
            create_files(in_dir, out_dir, b, db_items)
        print("=" * 20)

    else:
        logger.warning(
            "No valid backups found. The input directory should be "
            "the parent-directory immediately above the SHA-1 hash "
            "iOS device backups")
        sys.exit(2)
```

我们通过在日志中打印状态消息来启动`create_files()`方法：

```py
def create_files(in_dir, out_dir, b, db_items):
    msg = "Copying Files for backup {} to {}".format(
        b, os.path.join(out_dir, b))
    logger.info(msg)
```

接下来，我们创建一个计数器来跟踪在清单中找到但在备份中找不到的文件数量。然后，我们遍历从`process_manifest()`函数生成的`db_items`字典中的每个键。我们首先检查关联的文件名是否为`None`或空字符串，否则继续到下一个`SHA-1`哈希项：

```py
    files_not_found = 0
    for x, key in enumerate(db_items):
        if db_items[key][0] is None or db_items[key][0] == "":
            continue
```

如果关联的文件名存在，我们创建几个表示输出目录路径和输出文件路径的变量。请注意，输出路径被附加到备份名称`b`的名称上，以模仿输入目录中备份文件夹的结构。我们使用输出目录路径`dirpath`首先检查它是否存在，否则创建它：

```py
        else:
            dirpath = os.path.join(
                out_dir, b, os.path.dirname(db_items[key][0]))
            filepath = os.path.join(out_dir, b, db_items[key][0])
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
```

我们创建了一些路径变量，包括输入目录中备份文件的位置。我们通过创建一个字符串，其中包括备份名称、`SHA-1`哈希键的前两个字符和`SHA-1`键本身，它们之间用斜杠分隔来实现这一点。然后将其连接到输入目录中：

```py
            original_dir = b + "/" + key[0:2] + "/" + key
            path = os.path.join(in_dir, original_dir)
```

有了所有这些路径创建好后，我们现在可以开始执行一些验证步骤，然后将文件复制到新的输出目的地。首先，我们检查输出文件是否已经存在于输出文件夹中。在开发这个脚本的过程中，我们注意到一些文件具有相同的名称，并存储在输出文件夹中的同一文件夹中。这导致数据被覆盖，并且备份文件夹和输出文件夹之间的文件计数不匹配。为了解决这个问题，如果文件已经存在于备份中，我们会附加一个下划线和一个整数`x`，表示循环迭代次数，这对我们来说是一个唯一的值：

```py
            if os.path.exists(filepath):
                filepath = filepath + "_{}".format(x)
```

解决了文件名冲突后，我们使用`shutil.copyfile()`方法来复制由路径变量表示的备份文件，并将其重命名并存储在输出文件夹中，由`filepath`变量表示。如果路径变量指的是不在备份文件夹中的文件，它将引发`IOError`，我们会捕获并记录到日志文件中，并添加到我们的计数器中：

```py
            try:
                copyfile(path, filepath)
            except IOError:
                logger.debug("File not found in backup: {}".format(path))
                files_not_found += 1
```

然后，我们向用户提供一个警告，告知在`Manifest.db`中未找到的文件数量，以防用户未启用详细日志记录。一旦我们将备份目录中的所有文件复制完毕，我们就使用`shutil.copyfile()`方法逐个复制备份文件夹中存在的非混淆的PLIST和数据库文件到输出文件夹中：

```py
    if files_not_found > 0:
        logger.warning("{} files listed in the Manifest.db not"
                       "found in backup".format(files_not_found))

    copyfile(os.path.join(in_dir, b, "Info.plist"),
             os.path.join(out_dir, b, "Info.plist"))
    copyfile(os.path.join(in_dir, b, "Manifest.db"),
             os.path.join(out_dir, b, "Manifest.db"))
    copyfile(os.path.join(in_dir, b, "Manifest.plist"),
             os.path.join(out_dir, b, "Manifest.plist"))
    copyfile(os.path.join(in_dir, b, "Status.plist"),
             os.path.join(out_dir, b, "Status.plist"))
```

当我们运行这段代码时，我们可以在输出中看到以下更新后的文件结构：

![](../images/00026.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们在这里提供了一个建议：

+   添加功能以转换加密的iTunes备份。使用第三方库，如`pycrypto`，可以通过提供正确的密码来解密备份。

# 将Wi-Fi标记在地图上

食谱难度：中等

Python版本：3.5

操作系统：任意

没有与外部世界的连接，移动设备只不过是一块昂贵的纸砖。幸运的是，开放的Wi-Fi网络随处可见，有时移动设备会自动连接到它们。在iPhone上，设备连接过的Wi-Fi网络列表存储在一个名为`com.apple.wifi.plist`的二进制PLIST文件中。这个PLIST记录了Wi-Fi的SSID、BSSID和连接时间等信息。在这个教程中，我们将展示如何从标准的Cellebrite XML报告中提取Wi-Fi详细信息，或者提供Wi-Fi MAC地址的逐行分隔文件。由于Cellebrite报告格式可能随时间而变化，我们基于使用UFED Physical Analyzer版本6.1.6.19生成的报告进行XML解析。

WiGLE是一个在线可搜索的存储库，截至撰写时，拥有超过3亿个Wi-Fi网络。我们将使用Python的`requests`库访问WiGLE的API，以基于Wi-Fi MAC地址执行自动搜索。要安装`requests`库，我们可以使用`pip`，如下所示：

```py
pip install requests==2.18.4
```

如果在WiGLE存储库中找到网络，我们可以获取关于它的大量数据，包括其纬度和经度坐标。有了这些信息，我们可以了解用户设备所在的位置，以及可能的用户本身，以及连接的时间。

要了解更多关于WiGLE并使用WiGLE，请访问网站[https://wigle.net/.](https://wigle.net/)

# 入门

这个教程需要从WiGLE网站获取API密钥。要注册免费的API密钥，请访问[https://wigle.net/account](https://wigle.net/account)并按照说明显示您的API密钥。有两个API值，名称和密钥。对于这个教程，请创建一个文件，其中API名称值在前，后跟一个冒号（没有空格），然后是API密钥。脚本将读取此格式以对您进行WiGLE API身份验证。

在撰写时，为了查询WiGLE API，您必须向服务贡献数据。这是因为整个网站都是建立在社区共享数据的基础上的，这鼓励用户与他人分享信息。有许多贡献数据的方式，如[https://wigle.net](https://wigle.net)上所记录的那样。

# 如何操作...

这个教程遵循以下步骤来实现目标：

1.  将输入标识为Cellebrite XML报告或MAC地址的逐行文本文件。

1.  将任一类型的输入处理为Python数据集。

1.  使用`requests`查询WiGLE API。

1.  将返回的WiGLE结果优化为更方便的格式。

1.  将处理后的输出写入CSV文件。

# 它是如何工作的...

首先，我们导入所需的库来处理参数解析、编写电子表格、处理XML数据以及与WiGLE API交互：

```py
from __future__ import print_function
import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
import requests
```

这个教程的命令行处理程序接受两个位置参数，`INPUT_FILE`和`OUTPUT_CSV`，分别表示带有Wi-Fi MAC地址的输入文件和期望的输出CSV。默认情况下，脚本假定输入文件是Cellebrite XML报告。用户可以使用可选的`-t`标志指定输入文件的类型，并在`xml`或`txt`之间进行选择。此外，我们可以设置包含我们API密钥的文件的路径。默认情况下，这在用户目录的基础上设置，并命名为`.wigle_api`，但您可以更新此值以反映您的环境中最容易的内容。

保存您的API密钥的文件应具有额外的保护措施，通过文件权限或其他方式，以防止您的密钥被盗。

```py
if __name__ == "__main__":
    # Command-line Argument Parser
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("INPUT_FILE", help="INPUT FILE with MAC Addresses")
    parser.add_argument("OUTPUT_CSV", help="Output CSV File")
    parser.add_argument(
        "-t", help="Input type: Cellebrite XML report or TXT file",
        choices=('xml', 'txt'), default="xml")
    parser.add_argument('--api', help="Path to API key file",
                        default=os.path.expanduser("~/.wigle_api"),
                        type=argparse.FileType('r'))
    args = parser.parse_args()
```

我们执行标准的数据验证步骤，并检查输入文件是否存在且为文件，否则退出脚本。我们使用`os.path.dirname（）`来提取目录路径并检查其是否存在。如果目录不存在，我们使用`os.makedirs（）`函数来创建目录。在调用`main（）`函数之前，我们还读取并拆分API名称和密钥：

```py
    if not os.path.exists(args.INPUT_FILE) or \
            not os.path.isfile(args.INPUT_FILE):
        print("[-] {} does not exist or is not a file".format(
            args.INPUT_FILE))
        sys.exit(1)

    directory = os.path.dirname(args.OUTPUT_CSV)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    api_key = args.api.readline().strip().split(":")
```

在我们执行参数验证之后，我们将所有参数传递给`main（）`函数：

```py
    main(args.INPUT_FILE, args.OUTPUT_CSV, args.t, api_key)
```

在`main()`函数中，我们首先确定我们正在处理的输入类型。默认情况下，`type`变量是`"xml"`，除非用户另有指定。根据文件类型，我们将其发送到适当的解析器，该解析器将以字典形式返回提取的Wi-Fi数据元素。然后将此字典与输出CSV一起传递给`query_wigle()`函数。此函数负责查询、处理并将查询结果写入CSV文件。首先，让我们来看看解析器，从`parse_xml()`函数开始：

```py
def main(in_file, out_csv, type, api_key):
    if type == 'xml':
        wifi = parse_xml(in_file)
    else:
        wifi = parse_txt(in_file)

    query_wigle(wifi, out_csv, api_key)
```

我们使用`xml.etree.ElementTree`解析Cellebrite XML报告，我们已将其导入为`ET`。

要了解有关`xml`库的更多信息，请访问[https://docs.python.org/3/library/xml.etree.elementtree.html](https://docs.python.org/3/library/xml.etree.elementtree.html)。

解析由取证工具生成的报告可能是棘手的。这些报告的格式可能会发生变化，并破坏您的脚本。因此，我们不能假设此脚本将继续在未来的Cellebrite Physical Analyzer软件版本中运行。正因为如此，我们已包含了一个选项，可以使用此脚本与包含MAC地址的文本文件一起使用。

与任何XML文件一样，我们需要首先访问文件并使用`ET.parse()`函数对其进行解析。然后我们使用`getroot()`方法返回XML文件的根元素。我们将此根元素作为文件中搜索报告中的Wi-Fi数据标记的初始立足点：

```py
def parse_xml(xml_file):
    wifi = {}
    xmlns = "{http://pa.cellebrite.com/report/2.0}"
    print("[+] Opening {} report".format(xml_file))
    xml_tree = ET.parse(xml_file)
    print("[+] Parsing report for all connected WiFi addresses")
    root = xml_tree.getroot()
```

我们使用`iter()`方法来迭代根元素的子元素。我们检查每个子元素的标记，寻找模型标记。如果找到，我们检查它是否具有位置类型属性：

```py
    for child in root.iter():
        if child.tag == xmlns + "model":
            if child.get("type") == "Location":
```

对于找到的每个位置模型，我们使用`findall()`方法迭代其每个字段元素。此元素包含有关位置工件的元数据，例如网络的时间戳、BSSID和SSID。我们可以检查字段是否具有名称属性，其值为`"Timestamp"`，并将其值存储在`ts`变量中。如果值没有任何文本内容，我们继续下一个字段：

```py
                for field in child.findall(xmlns + "field"):
                    if field.get("name") == "TimeStamp":
                        ts_value = field.find(xmlns + "value")
                        try:
                            ts = ts_value.text
                        except AttributeError:
                            continue
```

类似地，我们检查字段的名称是否与`"Description"`匹配。此字段包含Wi-Fi网络的BSSID和SSID，以制表符分隔的字符串。我们尝试访问此值的文本，并在没有文本时引发`AttributeError`：

```py
                    if field.get("name") == "Description":
                        value = field.find(xmlns + "value")
                        try:
                            value_text = value.text
                        except AttributeError:
                            continue
```

因为Cellebrite报告中可能存在其他类型的`"Location"`工件，我们检查值的文本中是否存在字符串`"SSID"`。如果是，我们使用制表符特殊字符将字符串拆分为两个变量。我们从值的文本中提取的这些字符串包含一些不必要的字符，我们使用字符串切片将其从字符串中删除：

```py
                        if "SSID" in value.text:
                            bssid, ssid = value.text.split("\t")
                            bssid = bssid[7:]
                            ssid = ssid[6:]
```

在从报告中提取时间戳、BSSID和SSID之后，我们可以将它们添加到`wifi`字典中。如果Wi-Fi的BSSID已经存储为其中一个键，我们将时间戳和SSID附加到列表中。这样我们就可以捕获到这个Wi-Fi网络的所有历史连接以及网络名称的任何更改。如果我们还没有将此MAC地址添加到`wifi`字典中，我们将创建键/值对，包括存储API调用结果的WiGLE字典。在解析所有位置模型工件之后，我们将`wifi`字典返回给`main()`函数：

```py
                            if bssid in wifi.keys():
                                wifi[bssid]["Timestamps"].append(ts)
                                wifi[bssid]["SSID"].append(ssid)
                            else:
                                wifi[bssid] = {
                                    "Timestamps": [ts], "SSID": [ssid],
                                    "Wigle": {}}
    return wifi
```

与XML解析器相比，TXT解析器要简单得多。我们遍历文本文件的每一行，并将每一行设置为一个MAC地址，作为一个空字典的键。在处理文件中的所有行之后，我们将字典返回给`main()`函数：

```py
def parse_txt(txt_file):
    wifi = {}
    print("[+] Extracting MAC addresses from {}".format(txt_file))
    with open(txt_file) as mac_file:
        for line in mac_file:
            wifi[line.strip()] = {"Timestamps": ["N/A"], "SSID": ["N/A"],
                                  "Wigle": {}}
    return wifi
```

有了MAC地址的字典，我们现在可以转到`query_wigle()`函数，并使用`requests`进行WiGLE API调用。首先，我们在控制台打印一条消息，通知用户当前的执行状态。接下来，我们遍历字典中的每个MAC地址，并使用`query_mac_addr()`函数查询BSSID的站点：

```py
def query_wigle(wifi_dictionary, out_csv, api_key):
    print("[+] Querying Wigle.net through Python API for {} "
          "APs".format(len(wifi_dictionary)))
    for mac in wifi_dictionary:
        wigle_results = query_mac_addr(mac, api_key)
```

`query_mac_addr()`函数接受我们的MAC地址和API密钥，并构造请求的URL。我们使用API的基本URL，并在其末尾插入MAC地址。然后将此URL提供给`requests.get()`方法，以及`auth kwarg`来提供API名称和密钥。`requests`库处理形成并发送带有正确HTTP基本身份验证的数据包到API。`req`对象现在已准备好供我们解释，因此我们可以调用`json()`方法将数据返回为字典：

```py
def query_mac_addr(mac_addr, api_key):
    query_url = "https://api.wigle.net/api/v2/network/search?" \
        "onlymine=false&freenet=false&paynet=false" \
        "&netid={}".format(mac_addr)
    req = requests.get(query_url, auth=(api_key[0], api_key[1]))
    return req.json()
```

使用返回的`wigle_results`字典，我们检查`resultCount`键，以确定在`Wigle`数据库中找到了多少结果。如果没有结果，我们将一个空列表附加到`Wigle`字典中的结果键。同样，如果有结果，我们直接将返回的`wigle_results`字典附加到数据集中。API确实对每天可以执行的调用次数有限制。当达到限制时，将生成`KeyError`，我们捕获并打印到控制台。我们还提供其他错误的报告，因为API可能会扩展错误报告。在搜索每个地址并将结果添加到字典后，我们将其与输出CSV一起传递给`prep_output()`方法：

```py
        try:
            if wigle_results["resultCount"] == 0:
                wifi_dictionary[mac]["Wigle"]["results"] = []
                continue
            else:
                wifi_dictionary[mac]["Wigle"] = wigle_results
        except KeyError:
            if wigle_results["error"] == "too many queries today":
                print("[-] Wigle daily query limit exceeded")
                wifi_dictionary[mac]["Wigle"]["results"] = []
                continue
            else:
                print("[-] Other error encountered for "
                      "address {}: {}".format(mac, wigle_results['error']))
                wifi_dictionary[mac]["Wigle"]["results"] = []
                continue
    prep_output(out_csv, wifi_dictionary)
```

如果您还没有注意到，数据变得越来越复杂，这使得编写和处理它变得更加复杂。`prep_output()`方法基本上将字典展平为易于编写的块。我们需要这个函数的另一个原因是，我们需要为每个特定Wi-Fi网络连接的实例创建单独的行。虽然该网络的WiGLE结果将是相同的，但连接时间戳和网络SSID可能是不同的。

为了实现这一点，我们首先为最终处理的结果和与Google Maps相关的字符串创建一个字典。我们使用这个字符串来创建一个查询，其中包含纬度和经度，以便用户可以轻松地将URL粘贴到其浏览器中，以在Google Maps中查看地理位置详细信息：

```py
def prep_output(output, data):
    csv_data = {}
    google_map = "https://www.google.com/maps/search/"
```

我们遍历字典中的每个MAC地址，并创建两个额外的循环，以遍历MAC地址的所有时间戳和所有WiGLE结果。通过这些循环，我们现在可以访问到目前为止收集的所有数据，并开始将数据添加到新的输出字典中。

由于初始字典的复杂性，我们创建了一个名为`shortres`的变量，用作输出字典的更深部分的快捷方式。这样可以防止我们在每次需要访问字典的那部分时不必要地写入整个目录结构。`shortres`变量的第一个用法可以看作是我们从WiGLE结果中提取此网络的纬度和经度，并将其附加到Google Maps查询中：

```py
    for x, mac in enumerate(data):
        for y, ts in enumerate(data[mac]["Timestamps"]):
            for z, result in enumerate(data[mac]["Wigle"]["results"]):
                shortres = data[mac]["Wigle"]["results"][z]
                g_map_url = "{}{},{}".format(
                    google_map, shortres["trilat"], shortres["trilong"])
```

在一行中（相当复杂），我们添加一个键值对，其中键是基于循环迭代计数器的唯一键，值是展平的字典。我们首先创建一个新字典，其中包含BSSID、SSID、时间戳和新创建的Google Maps URL。因为我们想简化输出，我们需要合并新字典和存储在`shortres`变量中的WiGLE结果。

我们可以遍历第二个字典中的每个键，并逐个添加其键值对。但是，使用Python 3.5中引入的一个特性会更快，我们可以通过在每个字典之前放置两个`*`符号来合并这两个字典。这将合并两个字典，并且如果有任何重名的键，它将用第二个字典中的数据覆盖第一个字典中的数据。在这种情况下，我们没有任何键重叠，所以这将简单地合并字典。

请参阅以下StackOverflow帖子以了解更多关于字典合并的信息：

[https://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression](https://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression)。

在合并了所有字典之后，我们继续使用`write_csv()`函数最终写入输出：

```py
                csv_data["{}-{}-{}".format(x, y, z)] = {
                    **{
                        "BSSID": mac, "SSID": data[mac]["SSID"][y],
                        "Cellebrite Connection Time": ts,
                        "Google Map URL": g_map_url},
                    **shortres
                }

    write_csv(output, csv_data)
```

在这个示例中，我们重新介绍了`csv.DictWriter`类，它允许我们轻松地将字典写入CSV文件。这比我们之前使用的`csv.writer`类更可取，因为它为我们提供了一些好处，包括对列进行排序。为了利用这一点，我们需要知道我们使用的所有字段。由于WiGLE是动态的，报告的结果可能会改变，我们选择动态查找输出字典中所有键的名称。通过将它们添加到一个集合中，我们确保只有唯一的键：

```py
def write_csv(output, data):
    print("[+] Writing data to {}".format(output))
    field_list = set()
    for row in data:
        for field in data[row]:
            field_list.add(field)
```

一旦我们确定了输出中所有的键，我们就可以创建CSV对象。请注意，使用`csv.DictWriter`对象时，我们使用了两个关键字参数。如前所述，第一个是字典中所有键的列表，我们已经对其进行了排序。这个排序后的列表就是结果CSV中列的顺序。如果`csv.DictWriter`遇到一个不在提供的`field_list`中的键，由于我们的预防措施，它会忽略错误而不是引发异常，这是由`extrasaction kwarg`中的配置决定的：

```py
    with open(output, "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=sorted(
            field_list), extrasaction='ignore')
```

一旦我们设置好写入器，我们可以使用`writeheader()`方法根据提供的字段名称自动写入列。之后，只需简单地遍历数据中的每个字典，并使用`writerow()`函数将其写入CSV文件。虽然这个函数很简单，但想象一下，如果我们没有先简化原始数据结构，我们会有多大的麻烦：

```py
        csv_writer.writeheader()
        for csv_row in data:
            csv_writer.writerow(data[csv_row])
```

运行此脚本后，我们可以在CSV报告中看到各种有用的信息。前几列包括BSSID、Google地图URL、城市和县：

![](../images/00027.jpeg)

然后我们会看到一些时间戳，比如第一次出现的时间、最近出现的时间，以及更具体的位置，比如地区和道路：

![](../images/00028.jpeg)

最后，我们可以了解到SSID、坐标、网络类型和使用的认证方式：

![](../images/00029.jpeg)

# 深入挖掘以恢复消息

示例难度：困难

Python版本：3.5

操作系统：任意

在本章的前面，我们开发了一个从数据库中识别缺失记录的示例。在这个示例中，我们将利用该示例的输出，识别可恢复的记录及其在数据库中的偏移量。这是通过了解SQLite数据库的一些内部机制，并利用这种理解来实现的。

有关SQLite文件内部的详细描述，请查看[https://www.sqlite.org/fileformat.html](https://www.sqlite.org/fileformat.html)。

通过这种技术，我们将能够快速审查数据库并识别可恢复的消息。

当从数据库中删除一行时，类似于文件，条目不一定会被覆盖。根据数据库活动和其分配算法，这个条目可能会持续一段时间。例如，当触发`vacuum`命令时，我们恢复数据的机会会减少。

我们不会深入讨论SQLite结构；可以说每个条目由四个元素组成：有效载荷长度、ROWID、有效载荷头和有效载荷本身。前面的配方识别了缺失的ROWID值，我们将在这里使用它来查找数据库中所有这样的ROWID出现。我们将使用其他数据，例如已知的标准有效载荷头值，与iPhone短信数据库一起验证任何命中。虽然这个配方专注于从iPhone短信数据库中提取数据，但它可以修改为适用于任何数据库。我们稍后将指出需要更改的几行代码，以便将其用于其他数据库。

# 入门

此脚本中使用的所有库都包含在Python的标准库中。如果您想跟着操作，请获取iPhone短信数据库。如果数据库不包含任何已删除的条目，请使用SQLite连接打开它并删除一些条目。这是一个很好的测试，可以确认脚本是否按预期在您的数据集上运行。

# 操作步骤...

这个配方由以下步骤组成：

1.  连接到输入数据库。

1.  查询表PRAGMA并识别活动条目间隙。

1.  将ROWID间隙转换为它们的varint表示。

1.  在数据库的原始十六进制中搜索缺失的条目。

1.  将输出结果保存到CSV文件中。

# 工作原理...

首先，我们导入所需的库来处理参数解析、操作十六进制和二进制数据、编写电子表格、创建笛卡尔积的元组、使用正则表达式进行搜索以及与SQLite数据库交互：

```py
from __future__ import print_function
import argparse
import binascii
import csv
from itertools import product
import os
import re
import sqlite3
import sys
```

这个配方的命令行处理程序有三个位置参数和一个可选参数。这与本章前面的*在SQLite数据库中识别间隙*配方基本相同；但是，我们还添加了一个用于输出CSV文件的参数：

```py
if __name__ == "__main__":
    # Command-line Argument Parser
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("SQLITE_DATABASE", help="Input SQLite database")
    parser.add_argument("TABLE", help="Table to query from")
    parser.add_argument("OUTPUT_CSV", help="Output CSV File")
    parser.add_argument("--column", help="Optional column argument")
    args = parser.parse_args()
```

在解析参数后，我们将提供的参数传递给`main()`函数。如果用户提供了可选的列参数，我们将使用`col`关键字参数将其传递给`main()`函数：

```py
    if args.column is not None:
        main(args.SQLITE_DATABASE, args.TABLE,
             args.OUTPUT_CSV, col=args.column)
    else:
        main(args.SQLITE_DATABASE, args.TABLE, args.OUTPUT_CSV)
```

因为这个脚本利用了我们之前构建的内容，`main()`函数在很大程度上是重复的。我们不会重复关于代码的注释（对于一行代码，只能说这么多），我们建议您参考*在SQLite数据库中识别间隙*配方，以了解代码的这部分内容。

为了让大家回忆起来，以下是该配方的摘要：`main()`函数执行基本的输入验证，从给定表中识别潜在的主键（除非用户提供了列），并调用`find_gaps()`函数。`find_gaps()`函数是前一个脚本的另一个保留部分，几乎与前一个相同，只有一行不同。这个函数现在不再打印所有已识别的间隙，而是将已识别的间隙返回给`main()`函数。`main()`函数的其余部分和此后涵盖的所有其他代码都是新的。这是我们继续理解这个配方的地方。

识别了间隙后，我们调用一个名为`varint_converter()`的函数来处理每个间隙，将其转换为其varint对应项。Varint，也称为可变长度整数，是大小为1到9个字节的大端整数。SQLite使用Varint，因为它们所占的空间比存储ROWID整数本身要少。因此，为了有效地搜索已删除的ROWID，我们必须首先将其转换为varint，然后再进行搜索：

```py
    print("[+] Carving for missing ROWIDs")
    varints = varint_converter(list(gaps))
```

对于小于或等于127的ROWID，它们的varint等价物就是整数的十六进制表示。我们使用内置的`hex()`方法将整数转换为十六进制字符串，并使用字符串切片来删除前置的`0x`。例如，执行`hex(42)`返回字符串`0x2a`；在这种情况下，我们删除了前导的`0x`十六进制标识符，因为我们只对值感兴趣：

```py
def varint_converter(rows):
    varints = {}
    varint_combos = []
    for i, row in enumerate(rows):
        if row <= 127:
            varints[hex(row)[2:]] = row
```

如果缺失的ROWID是`128`或更大，我们开始一个无限的`while`循环来找到相关的varint。在开始循环之前，我们使用列表推导来创建一个包含数字`0`到`255`的列表。我们还实例化一个值为`1`的计数器变量。`while`循环的第一部分创建一个元组列表，其元素数量等于`counter`变量，包含`combos`列表的每个组合。例如，如果counter等于`2`，我们会看到一个元组列表，表示所有可能的2字节varints，如`[(0, 0), (0, 1), (0, 2), ..., (255, 255)]`。完成这个过程后，我们再次使用列表推导来删除所有第一个元素小于或等于`127`的元组。由于`if-else`循环的这部分处理大于或等于`128`的行，我们知道varint不能等于或小于`127`，因此这些值被排除在考虑之外：

```py
        else:
            combos = [x for x in range(0, 256)]
            counter = 1
            while True:
                counter += 1
                print("[+] Generating and finding all {} byte "
                      "varints..".format(counter))
                varint_combos = list(product(combos, repeat=counter))
                varint_combos = [x for x in varint_combos if x[0] >= 128]
```

创建了n字节varints列表后，我们循环遍历每个组合，并将其传递给`integer_converter()`函数。这个函数将这些数字视为varint的一部分，并将它们解码为相应的ROWID。然后，我们可以将返回的ROWID与缺失的ROWID进行比较。如果匹配，我们将一个键值对添加到`varints`字典中，其中键是varint的十六进制表示，值是缺失的ROWID。此时，我们将`i`变量增加`1`，并尝试获取下一个行元素。如果成功，我们处理该ROWID，依此类推，直到我们已经到达将生成`IndexError`的ROWIDs的末尾。我们捕获这样的错误，并将`varints`字典返回给`main()`函数。

关于这个函数需要注意的一件重要的事情是，因为输入是一个排序过的ROWIDs列表，我们只需要计算n字节varint组合一次，因为下一个ROWID只能比前一个更大而不是更小。另外，由于我们知道下一个ROWID至少比前一个大一，我们继续循环遍历我们创建的varint组合，而不重新开始，因为下一个ROWID不可能更小。这些技术展示了`while`循环的一个很好的用例，因为它们大大提高了该方法的执行速度：

```py
                for varint_combo in varint_combos:
                    varint = integer_converter(varint_combo)
                    if varint == row:
                        varints["".join([hex(v)[2:].zfill(2) for v in
                                         varint_combo])] = row
                        i += 1
                        try:
                            row = rows[i]
                        except IndexError:
                            return varints
```

`integer_converter()`函数相对简单。这个函数使用内置的`bin()`方法，类似于已经讨论过的`hex()`方法，将整数转换为其二进制等价物。我们遍历建议的varint中的每个值，首先使用`bin()`进行转换。这将返回一个字符串，这次前缀值为`0b`，我们使用字符串切片去除它。我们再次使用`zfill()`来确保字节具有所有位，因为`bin()`方法默认会去除前导的`0`位。之后，我们移除每个字节的第一位。当我们遍历我们的varint中的每个数字时，我们将处理后的位添加到一个名为`binary`的变量中。

这个过程可能听起来有点混乱，但这是解码varints的手动过程。

有关如何手动将varints转换为整数和其他SQLite内部的更多详细信息，请参阅*Forensics from the sausage factory*上的这篇博文：

[https://forensicsfromthesausagefactory.blogspot.com/2011/05/analysis-of-record-structure-within.html](https://forensicsfromthesausagefactory.blogspot.com/2011/05/analysis-of-record-structure-within.html).[﻿](https://forensicsfromthesausagefactory.blogspot.com/2011/05/analysis-of-record-structure-within.html)

在我们完成对数字列表的迭代后，我们使用`lstrip()`来去除二进制字符串中的任何最左边的零值。如果结果字符串为空，我们返回`0`；否则，我们将处理后的二进制数据转换并返回为从二进制表示的基数2的整数：

```py
def integer_converter(numbs):
    binary = ""
    for numb in numbs:
        binary += bin(numb)[2:].zfill(8)[1:]
    binvar = binary.lstrip("0")
    if binvar != '':
        return int(binvar, 2)
    else:
        return 0
```

回到`main（）`函数，我们将`varints`字典和数据库文件的路径传递给`find_candidates（）`函数：

```py
    search_results = find_candidates(database, varints)
```

我们搜索的两个候选者是“350055”和“360055”。如前所述，在数据库中，跟随单元格的ROWID是有效载荷头长度。iPhone短信数据库中的有效载荷头长度通常是两个值中的一个：要么是0x35，要么是0x36。在有效载荷头长度之后是有效载荷头本身。有效载荷头的第一个序列类型将是0x00，表示为NULL值，数据库的主键--第一列，因此第一个序列类型--将始终被记录为。接下来是序列类型0x55，对应于表中的第二列，消息GUID，它始终是一个21字节的字符串，因此将始终由序列类型0x55表示。任何经过验证的命中都将附加到结果列表中。

通过搜索ROWID varint和这三个附加字节，我们可以大大减少误报的数量。请注意，如果您正在处理的数据库不是iPhone短信数据库，则需要更改这些候选者的值，以反映表中ROWID之前的任何静态内容：

```py
def find_candidates(database, varints):
    results = []
    candidate_a = "350055"
    candidate_b = "360055"
```

我们以`rb`模式打开数据库以搜索其二进制内容。为了做到这一点，我们必须首先读取整个数据库，并使用`binascii.hexlify（）`函数将这些数据转换为十六进制。由于我们已经将varints存储为十六进制，因此现在可以轻松地搜索这些数据集以查找varint和其他周围的数据。我们通过循环遍历每个varint并创建两个不同的搜索字符串来开始搜索过程，以考虑iPhone短信数据库中的两个静态支点之一：

```py
    with open(database, "rb") as infile:
        hex_data = str(binascii.hexlify(infile.read()))
    for varint in varints:
        search_a = varint + candidate_a
        search_b = varint + candidate_b
```

然后，我们使用`re.finditer（）`方法基于`search_a`和`search_b`关键字来迭代每个命中。对于每个结果，我们附加一个包含ROWID、使用的搜索词和文件内的偏移量的列表。我们必须除以2来准确报告字节数，而不是十六进制数字的数量。在完成搜索数据后，我们将结果返回给`main（）`函数：

```py
        for result in re.finditer(search_a, hex_data):
            results.append([varints[varint], search_a, result.start() / 2])

        for result in re.finditer(search_b, hex_data):
            results.append([varints[varint], search_b, result.start() / 2])

    return results
```

最后一次，我们回到`main（）`函数。这次我们检查是否有搜索结果。如果有，我们将它们与CSV输出一起传递给`csvWriter（）`方法。否则，我们在控制台上打印状态消息，通知用户没有识别到完整可恢复的ROWID：

```py
    if search_results != []:
        print("[+] Writing {} potential candidates to {}".format(
            len(search_results), out_csv))
        write_csv(out_csv, ["ROWID", "Search Term", "Offset"],
                  search_results)
    else:
        print("[-] No search results found for missing ROWIDs")
```

`write_csv（）`方法一如既往地简单。我们打开一个新的CSV文件，并为嵌套列表结构中存储的三个元素创建三列。然后，我们使用`writerows（）`方法将结果数据列表中的所有行写入文件：

```py
def write_csv(output, cols, msgs):
    with open(output, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(cols)
        csv_writer.writerows(msgs)
```

当我们查看导出的报告时，我们可以清楚地看到我们的行ID、搜索的十六进制值以及记录被发现的数据库内的偏移量：

![](../images/00030.jpeg)

# 还有更多…

这个脚本可以进一步改进。我们在这里提供了一个建议：

+   而不是硬编码候选者，接受这些候选者的文本文件或命令行条目，以增加该脚本的灵活性
