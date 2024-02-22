# 第七章：基于日志的工件配方

本章涵盖了以下配方：

+   关于时间

+   使用 RegEx 解析 IIS weblogs

+   去探险

+   解释每日日志

+   将`daily.out`解析添加到 Axiom

+   使用 YARA 扫描指标

# 介绍

这些天，遇到配备某种形式的事件或活动监控软件的现代系统并不罕见。这种软件可能被实施以协助安全、调试或合规要求。无论情况如何，这些宝贵的信息宝库通常被广泛利用于各种类型的网络调查。日志分析的一个常见问题是需要筛选出感兴趣的子集所需的大量数据。通过本章的配方，我们将探索具有很大证据价值的各种日志，并演示快速处理和审查它们的方法。具体来说，我们将涵盖：

+   将不同的时间戳格式（UNIX、FILETIME 等）转换为人类可读的格式

+   解析来自 IIS 平台的 Web 服务器访问日志

+   使用 Splunk 的 Python API 摄取、查询和导出日志

+   从 macOS 的`daily.out`日志中提取驱动器使用信息

+   从 Axiom 执行我们的`daily.out`日志解析器

+   使用 YARA 规则识别感兴趣的文件的奖励配方

访问[www.packtpub.com/books/content/support](http://www.packtpub.com/books/content/support)下载本章的代码包。

# 关于时间

配方难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

任何良好日志文件的一个重要元素是时间戳。这个值传达了日志中记录的活动或事件的日期和时间。这些日期值可以以许多格式出现，并且可以表示为数字或十六进制值。除了日志之外，不同的文件和工件以不同的方式存储日期，即使数据类型保持不变。一个常见的区分因素是纪元值，即格式从中计算时间的日期。一个常见的纪元是 1970 年 1 月 1 日，尽管其他格式从 1601 年 1 月 1 日开始计算。在不同格式之间不同的因素是用于计数的间隔。虽然常见的是看到以秒或毫秒计数的格式，但有些格式计算时间块，比如自纪元以来的 100 纳秒数。因此，这里开发的配方可以接受原始日期时间输入，并将格式化的时间戳作为其输出。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。

# 如何做...

为了在 Python 中解释常见的日期格式，我们执行以下操作：

1.  设置参数以获取原始日期值、日期来源和数据类型。

1.  开发一个为不同日期格式提供通用接口的类。

1.  支持处理 Unix 纪元值和 Microsoft 的`FILETIME`日期。

# 它是如何工作的...

我们首先导入用于处理参数和解析日期的库。具体来说，我们需要从`datetime`库中导入`datetime`类来读取原始日期值，以及`timedelta`类来指定时间戳偏移量。

```py
from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime as dt
from datetime import timedelta
```

这个配方的命令行处理程序接受三个位置参数，`date_value`、`source`和`type`，分别代表要处理的日期值、日期值的来源（UNIX、FILETIME 等）和类型（整数或十六进制值）。我们使用`choices`关键字来限制用户可以提供的选项。请注意，源参数使用自定义的`get_supported_formats()`函数，而不是预定义的受支持日期格式列表。然后，我们获取这些参数并初始化`ParseDate`类的一个实例，并调用`run()`方法来处理转换过程，然后将其`timestamp`属性打印到控制台。

```py
if __name__ == '__main__':
    parser = ArgumentParser(
        description=__description__,
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("date_value", help="Raw date value to parse")
    parser.add_argument("source", help="Source format of date",
                        choices=ParseDate.get_supported_formats())
    parser.add_argument("type", help="Data type of input value",
                        choices=('number', 'hex'), default='int')
    args = parser.parse_args()

    date_parser = ParseDate(args.date_value, args.source, args.type)
    date_parser.run()
    print(date_parser.timestamp)
```

让我们看看`ParseDate`类是如何工作的。通过使用一个类，我们可以轻松地扩展和在其他脚本中实现这段代码。从命令行参数中，我们接受日期值、日期源和值类型的参数。这些值和输出变量`timestamp`在`__init__`方法中被定义：

```py
class ParseDate(object):
    def __init__(self, date_value, source, data_type):
        self.date_value = date_value
        self.source = source
        self.data_type = data_type
        self.timestamp = None
```

`run（）`方法是控制器，很像我们许多食谱中的`main（）`函数，并根据日期源选择要调用的正确方法。这使我们能够轻松扩展类并轻松添加新的支持。在这个版本中，我们只支持三种日期类型：Unix 纪元秒，Unix 纪元毫秒和 Microsoft 的 FILETIME。为了减少我们需要编写的方法数量，我们将设计 Unix 纪元方法来处理秒和毫秒格式的时间戳。

```py
    def run(self):
        if self.source == 'unix-epoch':
            self.parse_unix_epoch()
        elif self.source == 'unix-epoch-ms':
            self.parse_unix_epoch(True)
        elif self.source == 'windows-filetime':
            self.parse_windows_filetime()
```

为了帮助未来想要使用这个库的人，我们添加了一个查看支持的格式的方法。通过使用`@classmethod`装饰器，我们可以在不需要先初始化类的情况下公开这个函数。这就是我们可以在命令行处理程序中使用`get_supported_formats（）`方法的原因。只需记住在添加新功能时更新它！

```py
    @classmethod
    def get_supported_formats(cls):
        return ['unix-epoch', 'unix-epoch-ms', 'windows-filetime']
```

`parse_unix_epoch（）`方法处理处理 Unix 纪元时间。我们指定一个可选参数`milliseconds`，以在处理秒和毫秒值之间切换此方法。首先，我们必须确定数据类型是``"hex"``还是``"number"``。如果是``"hex"``，我们将其转换为整数，如果是``"number"``，我们将其转换为浮点数。如果我们不认识或不支持此方法的数据类型，比如`string`，我们向用户抛出错误并退出脚本。

在转换值后，我们评估是否应将其视为毫秒值，如果是，则在进一步处理之前将其除以`1,000`。随后，我们使用`datetime`类的`fromtimestamp（）`方法将数字转换为`datetime`对象。最后，我们将这个日期格式化为人类可读的格式，并将这个字符串存储在`timestamp`属性中。

```py
    def parse_unix_epoch(self, milliseconds=False):
        if self.data_type == 'hex':
            conv_value = int(self.date_value)
            if milliseconds:
                conv_value = conv_value / 1000.0
        elif self.data_type == 'number':
            conv_value = float(self.date_value)
            if milliseconds:
                conv_value = conv_value / 1000.0
        else:
            print("Unsupported data type '{}' provided".format(
                self.data_type))
            sys.exit('1')

        ts = dt.fromtimestamp(conv_value)
        self.timestamp = ts.strftime('%Y-%m-%d %H:%M:%S.%f')
```

`parse_windows_filetime（）`类方法处理`FILETIME`格式，通常存储为十六进制值。使用与之前相似的代码块，我们将``"hex"``或``"number"``值转换为 Python 对象，并对任何其他提供的格式引发错误。唯一的区别是在进一步处理之前，我们将日期值除以`10`而不是`1,000`。

在之前的方法中，`datetime`库处理了纪元偏移，这次我们需要单独处理这个偏移。使用`timedelta`类，我们指定毫秒值，并将其添加到代表 FILETIME 格式纪元的`datetime`对象中。现在得到的`datetime`对象已经准备好供我们格式化和输出给用户了：

```py
    def parse_windows_filetime(self):
        if self.data_type == 'hex':
            microseconds = int(self.date_value, 16) / 10.0
        elif self.data_type == 'number':
            microseconds = float(self.date_value) / 10
        else:
            print("Unsupported data type '{}' provided".format(
                self.data_type))
            sys.exit('1')

        ts = dt(1601, 1, 1) + timedelta(microseconds=microseconds)
        self.timestamp = ts.strftime('%Y-%m-%d %H:%M:%S.%f')
```

当我们运行这个脚本时，我们可以提供一个时间戳，并以易于阅读的格式查看转换后的值，如下所示：

![](img/00076.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个建议如下：

+   为其他类型的时间戳（OLE，WebKit 等）添加支持

+   通过`pytz`添加时区支持

+   使用`dateutil`处理难以阅读的日期格式

# 使用 RegEx 解析 IIS Web 日志

食谱难度：中等

Python 版本：3.5

操作系统：任何

来自 Web 服务器的日志对于生成用户统计信息非常有用，为我们提供了有关使用的设备和访问者的地理位置的深刻信息。它们还为寻找试图利用 Web 服务器或未经授权使用的用户的审查人员提供了澄清。虽然这些日志存储了重要的细节，但以一种不便于高效分析的方式进行。如果您尝试手动分析，字段名称被指定在文件顶部，并且在阅读文本文件时需要记住字段的顺序。幸运的是，有更好的方法。使用以下脚本，我们展示了如何遍历每一行，将值映射到字段，并创建一个正确显示结果的电子表格 - 使得快速分析数据集变得更加容易。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。

# 如何做...

为了正确制作这个配方，我们需要采取以下步骤：

1.  接受输入日志文件和输出 CSV 文件的参数。

1.  为日志的每一列定义正则表达式模式。

1.  遍历日志中的每一行，并以一种我们可以解析单独元素并处理带引号的空格字符的方式准备每一行。

1.  验证并将每个值映射到其相应的列。

1.  将映射的列和值写入电子表格报告。

# 它是如何工作的...

我们首先导入用于处理参数和日志记录的库，然后是我们需要解析和验证日志信息的内置库。这些包括`re`正则表达式库和`shlex`词法分析器库。我们还包括`sys`和`csv`来处理日志消息和报告的输出。我们通过调用`getLogger()`方法初始化了该配方的日志对象。

```py
from __future__ import print_function
from argparse import ArgumentParser, FileType
import re
import shlex
import logging
import sys
import csv

logger = logging.getLogger(__file__)
```

在导入之后，我们为从日志中解析的字段定义模式。这些信息在日志之间可能会有所不同，尽管这里表达的模式应该涵盖日志中的大多数元素。

您可能需要添加、删除或重新排序以下定义的模式，以正确解析您正在使用的 IIS 日志。这些模式应该涵盖 IIS 日志中常见的元素。

我们将这些模式构建为名为`iis_log_format`的元组列表，其中第一个元组元素是列名，第二个是用于验证预期内容的正则表达式模式。通过使用正则表达式模式，我们可以定义数据必须遵循的一组规则以使其有效。这些列必须按它们在日志中出现的顺序来表达，否则代码将无法正确地将值映射到列。

```py
iis_log_format = [
    ("date", re.compile(r"\d{4}-\d{2}-\d{2}")),
    ("time", re.compile(r"\d\d:\d\d:\d\d")),
    ("s-ip", re.compile(
        r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}")),
    ("cs-method", re.compile(
        r"(GET)|(POST)|(PUT)|(DELETE)|(OPTIONS)|(HEAD)|(CONNECT)")),
    ("cs-uri-stem", re.compile(r"([A-Za-z0-1/\.-]*)")),
    ("cs-uri-query", re.compile(r"([A-Za-z0-1/\.-]*)")),
    ("s-port", re.compile(r"\d*")),
    ("cs-username", re.compile(r"([A-Za-z0-1/\.-]*)")),
    ("c-ip", re.compile(
        r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}")),
    ("cs(User-Agent)", re.compile(r".*")),
    ("sc-status", re.compile(r"\d*")),
    ("sc-substatus", re.compile(r"\d*")),
    ("sc-win32-status", re.compile(r"\d*")),
    ("time-taken", re.compile(r"\d*"))
]
```

此配方的命令行处理程序接受两个位置参数，`iis_log`和`csv_report`，分别表示要处理的 IIS 日志和所需的 CSV 路径。此外，此配方还接受一个可选参数`l`，指定配方日志文件的输出路径。

接下来，我们初始化了该配方的日志实用程序，并为控制台和基于文件的日志记录进行了配置。这一点很重要，因为我们应该以正式的方式注意到当我们无法为用户解析一行时。通过这种方式，如果出现问题，他们不应该在错误的假设下工作，即所有行都已成功解析并显示在生成的 CSV 电子表格中。我们还希望记录运行时消息，包括脚本的版本和提供的参数。在这一点上，我们准备调用`main()`函数并启动脚本。有关设置日志对象的更详细解释，请参阅第一章中的日志配方，*基本脚本和文件信息配方*。

```py
if __name__ == '__main__':
    parser = ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument('iis_log', help="Path to IIS Log",
                        type=FileType('r'))
    parser.add_argument('csv_report', help="Path to CSV report")
    parser.add_argument('-l', help="Path to processing log",
                        default=__name__ + '.log')
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    msg_fmt = logging.Formatter("%(asctime)-15s %(funcName)-10s "
                                "%(levelname)-8s %(message)s")

    strhndl = logging.StreamHandler(sys.stdout)
    strhndl.setFormatter(fmt=msg_fmt)
    fhndl = logging.FileHandler(args.log, mode='a')
    fhndl.setFormatter(fmt=msg_fmt)

    logger.addHandler(strhndl)
    logger.addHandler(fhndl)

    logger.info("Starting IIS Parsing ")
    logger.debug("Supplied arguments: {}".format(", ".join(sys.argv[1:])))
    logger.debug("System " + sys.platform)
    logger.debug("Version " + sys.version)
    main(args.iis_log, args.csv_report, logger)
    logger.info("IIS Parsing Complete")
```

`main（）`函数处理了脚本中大部分的逻辑。我们创建一个列表`parsed_logs`，用于在迭代日志文件中的行之前存储解析后的行。在`for`循环中，我们剥离行并创建一个存储字典`log_entry`，以存储记录。通过跳过以注释（或井号）字符开头的行或者行为空，我们加快了处理速度，并防止了列匹配中的错误。

虽然 IIS 日志存储为以空格分隔的值，但它们使用双引号来转义包含空格的字符串。例如，`useragent`字符串是一个单一值，但通常包含一个或多个空格。使用`shlex`模块，我们可以使用`shlex（）`方法解析带有双引号的空格的行，并通过正确地在空格值上分隔数据来自动处理引号转义的空格。这个库可能会减慢处理速度，因此我们只在包含双引号字符的行上使用它。

```py
def main(iis_log, report_file, logger):
    parsed_logs = []
    for raw_line in iis_log:
        line = raw_line.strip()
        log_entry = {}
        if line.startswith("#") or len(line) == 0:
            continue
        if '\"' in line:
            line_iter = shlex.shlex(line_iter)
        else:
            line_iter = line.split(" ")
```

将行正确分隔后，我们使用`enumerate`函数逐个遍历记录中的每个元素，并提取相应的列名和模式。使用模式，我们在值上调用`match（）`方法，如果匹配，则在`log_entry`字典中创建一个条目。如果值不匹配模式，我们记录一个错误，并在日志文件中提供整行。在遍历每个列后，我们将记录字典附加到初始解析日志记录列表，并对剩余行重复此过程。

```py
        for count, split_entry in enumerate(line_iter):
            col_name, col_pattern = iis_log_format[count]
            if col_pattern.match(split_entry):
                log_entry[col_name] = split_entry
            else:
                logger.error("Unknown column pattern discovered. "
                             "Line preserved in full below")
                logger.error("Unparsed Line: {}".format(line))

        parsed_logs.append(log_entry)
```

处理完所有行后，我们在准备`write_csv（）`方法之前向控制台打印状态消息。我们使用一个简单的列表推导表达式来提取`iis_log_format`列表中每个元组的第一个元素，这代表一个列名。有了提取的列，让我们来看看报告编写器。

```py
    logger.info("Parsed {} lines".format(len(parsed_logs)))

    cols = [x[0] for x in iis_log_format]
    logger.info("Creating report file: {}".format(report_file))
    write_csv(report_file, cols, parsed_logs)
    logger.info("Report created")
```

报告编写器使用我们之前探讨过的方法创建一个 CSV 文件。由于我们将行存储为字典列表，我们可以使用`csv.DictWriter`类的四行代码轻松创建报告。

```py
def write_csv(outfile, fieldnames, data):
    with open(outfile, 'w', newline="") as open_outfile:
        csvfile = csv.DictWriter(open_outfile, fieldnames)
        csvfile.writeheader()
        csvfile.writerows(data)
```

当我们查看脚本生成的 CSV 报告时，我们会在样本输出中看到以下字段：

![](img/00077.jpeg)![](img/00078.jpeg)

# 还有更多...

这个脚本可以进一步改进。以下是一个建议：

+   虽然我们可以像在脚本开头看到的那样定义正则表达式模式，但我们可以使用正则表达式管理库来简化我们的生活。一个例子是`grok`库，它用于为模式创建变量名。这使我们能够轻松地组织和扩展模式，因为我们可以按名称而不是字符串值来表示它们。这个库被其他平台使用，比如 ELK 堆栈，用于管理和实现正则表达式。

# 进行洞穴探险

菜谱难度：中等

Python 版本：2.7

操作系统：任何

由于保存的详细级别和时间范围，日志文件很快就会变得相当庞大。正如您可能已经注意到的那样，先前菜谱的 CSV 报告很容易变得过大，以至于我们的电子表格应用程序无法有效地打开或浏览。与其在电子表格中分析这些数据，一个替代方法是将数据加载到数据库中。

**Splunk**是一个将 NoSQL 数据库与摄取和查询引擎结合在一起的平台，使其成为一个强大的分析工具。它的数据库的操作方式类似于 Elasticsearch 或 MongoDB，允许存储文档或结构化记录。因此，我们不需要为了将记录存储在数据库中而提供具有一致键值映射的记录。这就是使 NoSQL 数据库对于日志分析如此有用的原因，因为日志格式可能根据事件类型而变化。

在这个步骤中，我们学习将上一个步骤的 CSV 报告索引到 Splunk 中，从而可以在平台内部与数据交互。我们还设计脚本来针对数据集运行查询，并将响应查询的结果子集导出到 CSV 文件。这些过程分别处理，因此我们可以根据需要独立查询和导出数据。

# 入门

这个步骤需要安装第三方库`splunk-sdk`。此脚本中使用的所有其他库都包含在 Python 的标准库中。此外，我们必须在主机操作系统上安装 Splunk，并且由于`splunk-sdk`库的限制，必须使用 Python 2 来运行脚本。

要安装 Splunk，我们需要转到[Splunk.com](https://www.splunk.com/)，填写表格，并选择 Splunk Enterprise 免费试用下载。这个企业试用版允许我们练习 API，并且可以每天上传 500MB。下载应用程序后，我们需要启动它来配置应用程序。虽然有很多配置可以更改，但现在使用默认配置启动，以保持简单并专注于 API。这样做后，服务器的默认地址将是`localhost:8000`。通过在浏览器中导航到这个地址，我们可以首次登录，设置账户和（*请执行此操作*）更改管理员密码。

新安装的 Splunk 的默认用户名和密码是*admin*和*changeme*。

在 Splunk 实例激活后，我们现在可以安装 API 库。这个库处理从 REST API 到 Python 对象的转换。在撰写本书时，Splunk API 只能在 Python 2 中使用。`splunk-sdk`库可以使用`pip`安装：

```py
pip install splunk-sdk==1.6.2
```

要了解更多关于`splunk-sdk`库的信息，请访问[`dev.splunk.com/python`](http://dev.splunk.com/python)。

# 如何做到...

现在环境已经正确配置，我们可以开始开发代码。这个脚本将新数据索引到 Splunk，对该数据运行查询，并将响应我们查询的数据子集导出到 CSV 文件。为了实现这一点，我们需要：

1.  开发一个强大的参数处理接口，允许用户指定这些选项。

1.  构建一个处理各种属性方法的操作类。

1.  创建处理索引新数据和创建数据存储索引的过程的方法。

1.  建立运行 Splunk 查询的方法，以便生成信息丰富的报告。

1.  提供一种将报告导出为 CSV 格式的机制。

# 它是如何工作的...

首先导入此脚本所需的库，包括新安装的`splunklib`。为了防止由于用户无知而引起不必要的错误，我们使用`sys`库来确定执行脚本的 Python 版本，并在不是 Python 2 时引发错误。

```py
from __future__ import print_function
from argparse import ArgumentParser, ArgumentError
from argparse import ArgumentDefaultsHelpFormatter
import splunklib.client as client
import splunklib.results as results
import os
import sys
import csv

if sys.version_info.major != 2:
    print("Invalid python version. Must use Python 2 due to splunk api "
          "library")
```

下一个逻辑块是开发步骤的命令行参数处理程序。由于这段代码有很多选项和操作需要在代码中执行，我们需要在这一部分花费一些额外的时间。而且因为这段代码是基于类的，所以我们必须在这一部分设置一些额外的逻辑。

这个步骤的命令行处理程序接受一个位置输入`action`，表示要运行的操作（索引、查询或导出）。此步骤还支持七个可选参数：`index`、`config`、`file`、`query`、`cols`、`host`和`port`。让我们开始看看所有这些选项都做什么。

`index`参数实际上是一个必需的参数，用于指定要从中摄取、查询或导出数据的 Splunk 索引的名称。这可以是现有的或新的`index`名称。`config`参数是指包含 Splunk 实例的用户名和密码的配置文件。如参数帮助中所述，此文件应受保护并存储在代码执行位置之外。在企业环境中，您可能需要进一步保护这些凭据。

```py
if __name__ == '__main__':
    parser = ArgumentParser(
        description=__description__,
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument('action', help="Action to run",
                        choices=['index', 'query', 'export'])
    parser.add_argument('--index-name', help="Name of splunk index",
                        required=True)
    parser.add_argument('--config',
                        help="Place where login details are stored."
                        " Should have the username on the first line and"
                        " the password on the second."
                        " Please Protect this file!",
                        default=os.path.expanduser("~/.splunk_py.ini"))
```

`file`参数将用于提供要`index`到平台的文件的路径，或用于指定要将导出的`query`数据写入的文件名。例如，我们将使用`file`参数指向我们希望从上一个配方中摄取的 CSV 电子表格。`query`参数也具有双重作用，它可以用于从 Splunk 运行查询，也可以用于指定要导出为 CSV 的查询 ID。这意味着`index`和`query`操作只需要其中一个参数，但`export`操作需要两个参数。

```py
    parser.add_argument('--file', help="Path to file")
    parser.add_argument('--query', help="Splunk query to run or sid of "
                        "existing query to export")
```

最后一组参数允许用户修改配方的默认属性。例如，`cols`参数可用于指定从源数据中导出的列及其顺序。由于我们将查询和导出 IIS 日志，因此我们已经知道可用的列，并且对我们感兴趣。您可能希望根据正在探索的数据类型指定替代默认列。我们的最后两个参数包括`host`和`port`参数，每个参数默认为本地服务器，但可以配置为允许您与替代实例进行交互。

```py
    parser.add_argument(
        '--cols',
        help="Speficy columns to export. comma seperated list",
        default='_time,date,time,sc_status,c_ip,s_ip,cs_User_Agent')
    parser.add_argument('--host', help="hostname of server",
                        default="localhost")
    parser.add_argument('--port', help="help", default="8089")
    args = parser.parse_args()
```

确定了我们的参数后，我们可以解析它们并在执行配方之前验证所有要求是否满足。首先，我们必须打开并读取包含身份验证凭据的`config`文件，其中`username`在第一行，`password`在第二行。使用这些信息，我们创建一个包含登录详细信息和服务器位置的字典`conn_dict`，并将该字典传递给`splunklib`的`client.connect()`方法。请注意，我们使用`del()`方法删除包含这些敏感信息的变量。虽然用户名和密码仍然可以通过`service`对象访问，但我们希望限制存储这些详细信息的区域数量。在创建`service`变量后，我们测试是否在 Splunk 中安装了任何应用程序，因为默认情况下至少有一个应用程序，并将其用作成功验证的测试。

```py
    with open(args.config, 'r') as open_conf:
        username, password = [x.strip() for x in open_conf.readlines()]
    conn_dict = {'host': args.host, 'port': int(args.port),
                 'username': username, 'password': password}
    del(username)
    del(password)
    service = client.connect(**conn_dict)
    del(conn_dict)

    if len(service.apps) == 0:
        print("Login likely unsuccessful, cannot find any applications")
        sys.exit()
```

我们继续处理提供的参数，将列转换为列表并创建`Spelunking`类实例。要初始化该类，我们必须向其提供`service`变量、要执行的操作、索引名称和列。使用这些信息，我们的类实例现在已经准备就绪。

```py
    cols = args.cols.split(",")
    spelunking = Spelunking(service, args.action, args.index_name, cols)
```

接下来，我们使用一系列`if-elif-else`语句来处理我们预期遇到的三种不同操作。如果用户提供了`index`操作，我们首先确认可选的`file`参数是否存在，如果不存在则引发错误。如果我们找到它，我们将该值分配给`Spelunking`类实例的相应属性。对于`query`和`export`操作，我们重复这种逻辑，确认它们也使用了正确的可选参数。请注意，我们使用`os.path.abspath()`函数为类分配文件的绝对路径。这允许`splunklib`在系统上找到正确的文件。也许这是本书中最长的参数处理部分，我们已经完成了必要的逻辑，现在可以调用类的`run()`方法来启动特定操作的处理。

```py
    if spelunking.action == 'index':
        if 'file' not in vars(args):
            ArgumentError('--file parameter required')
            sys.exit()
        else:
            spelunking.file = os.path.abspath(args.file)

    elif spelunking.action == 'export':
        if 'file' not in vars(args):
            ArgumentError('--file parameter required')
            sys.exit()
        if 'query' not in vars(args):
            ArgumentError('--query parameter required')
            sys.exit()
        spelunking.file = os.path.abspath(args.file)
        spelunking.sid = args.query

    elif spelunking.action == 'query':
        if 'query' not in vars(args):
            ArgumentError('--query parameter required')
            sys.exit()
        else:
            spelunking.query = "search index={} {}".format(args.index_name,
                                                           args.query)

    else:
        ArgumentError('Unknown action required')
        sys.exit()

    spelunking.run()
```

现在参数已经在我们身后，让我们深入研究负责处理用户请求操作的类。这个类有四个参数，包括`service`变量，用户指定的`action`，Splunk 索引名称和要使用的列。所有其他属性都设置为`None`，如前面的代码块所示，如果它们被提供，将在执行时适当地初始化。这样做是为了限制类所需的参数数量，并处理某些属性未使用的情况。所有这些属性都在我们的类开始时初始化，以确保我们已经分配了默认值。

```py
class Spelunking(object):
    def __init__(self, service, action, index_name, cols):
        self.service = service
        self.action = action
        self.index = index_name
        self.file = None
        self.query = None
        self.sid = None
        self.job = None
        self.cols = cols
```

`run()`方法负责使用`get_or_create_index()`方法从 Splunk 实例获取`index`对象。它还检查在命令行指定了哪个动作，并调用相应的类实例方法。

```py
    def run(self):
        index_obj = self.get_or_create_index()
        if self.action == 'index':
            self.index_data(index_obj)
        elif self.action == 'query':
            self.query_index()
        elif self.action == 'export':
            self.export_report()
        return
```

`get_or_create_index()`方法，顾名思义，首先测试指定的索引是否存在，并连接到它，或者如果没有找到该名称的索引，则创建一个新的索引。由于这些信息存储在`service`变量的`indexes`属性中，作为一个类似字典的对象，我们可以很容易地通过名称测试索引的存在。

```py
    def get_or_create_index(self):
        # Create a new index
        if self.index not in self.service.indexes:
            return service.indexes.create(self.index)
        else:
            return self.service.indexes[self.index]
```

要从文件中摄取数据，比如 CSV 文件，我们可以使用一行语句将信息发送到`index_data()`方法中的实例。这个方法使用`splunk_index`对象的`upload()`方法将文件发送到 Splunk 进行摄取。虽然 CSV 文件是一个简单的例子，说明我们可以如何导入数据，但我们也可以使用前面的方法从原始日志中读取数据到 Splunk 实例，而不需要中间的 CSV 步骤。为此，我们希望使用`index`对象的不同方法，允许我们逐个发送每个解析的事件。

```py
    def index_data(self, splunk_index):
        splunk_index.upload(self.file)
```

`query_index()`方法涉及更多，因为我们首先需要修改用户提供的查询。如下面的片段所示，我们需要将用户指定的列添加到初始查询中。这将使在导出阶段未使用的字段在查询中可用。在修改后，我们使用`service.jobs.create()`方法在 Splunk 系统中创建一个新的作业，并记录查询 SID。这个 SID 将在导出阶段用于导出特定查询作业的结果。我们打印这些信息，以及作业在 Splunk 实例中到期之前的时间。默认情况下，这个生存时间值是`300`秒，或五分钟。

```py
    def query_index(self):
        self.query = self.query + "| fields + " + ", ".join(self.cols)
        self.job = self.service.jobs.create(self.query, rf=self.cols)
        self.sid = self.job.sid
        print("Query job {} created. will expire in {} seconds".format(
            self.sid, self.job['ttl']))
```

正如之前提到的，`export_report()`方法使用前面方法中提到的 SID 来检查作业是否完成，并检索要导出的数据。为了做到这一点，我们遍历可用的作业，如果我们的作业不存在，则发出警告。如果找到作业，但`is_ready()`方法返回`False`，则作业仍在处理中，尚未准备好导出结果。

```py
    def export_report(self):
        job_obj = None
        for j in self.service.jobs:
            if j.sid == self.sid:
                job_obj = j

        if job_obj is None:
            print("Job SID {} not found. Did it expire?".format(self.sid))
            sys.exit()

        if not job_obj.is_ready():
            print("Job SID {} is still processing. "
                  "Please wait to re-run".format(self.sir))
```

如果作业通过了这两个测试，我们从 Splunk 中提取数据，并使用`write_csv()`方法将其写入 CSV 文件。在这之前，我们需要初始化一个列表来存储作业结果。接下来，我们检索结果，指定感兴趣的列，并将原始数据读入`job_results`变量。幸运的是，`splunklib`提供了一个`ResultsReader`，它将`job_results`变量转换为一个字典列表。我们遍历这个列表，并将每个字典附加到`export_data`列表中。最后，我们提供文件路径、列名和要导出到 CSV 写入器的数据集。

```py
        export_data = []
        job_results = job_obj.results(rf=self.cols)
        for result in results.ResultsReader(job_results):
            export_data.append(result)

        self.write_csv(self.file, self.cols, export_data)
```

这个类中的`write_csv()`方法是一个`@staticmethod`。这个装饰器允许我们在类中使用一个通用的方法，而不需要指定一个实例。这个方法无疑会让那些在本书的其他地方使用过的人感到熟悉，我们在那里打开输出文件，创建一个`DictWriter`对象，然后将列标题和数据写入文件。

```py
    @staticmethod
    def write_csv(outfile, fieldnames, data):
        with open(outfile, 'wb') as open_outfile:
            csvfile = csv.DictWriter(open_outfile, fieldnames,
                                     extrasaction="ignore")
            csvfile.writeheader()
            csvfile.writerows(data)
```

在我们的假设用例中，第一阶段将是索引前一个食谱中 CSV 电子表格中的数据。如下片段所示，我们提供了前一个食谱中的 CSV 文件，并将其添加到 Splunk 索引中。接下来，我们寻找所有用户代理为 iPhone 的条目。最后，最后一个阶段涉及从查询中获取输出并创建一个 CSV 报告。

![](img/00079.jpeg)

成功执行这三个命令后，我们可以打开并查看过滤后的输出：

![](img/00080.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个建议，如下所示：

+   Python 的 Splunk API（以及一般）还有许多其他功能。此外，还可以使用更高级的查询技术来生成我们可以将其转换为技术和非技术最终用户的图形的数据。了解更多 Splunk API 提供的许多功能。

# 解释 daily.out 日志

食谱难度：中等

Python 版本：3.5

操作系统：任意

操作系统日志通常反映系统上软件、硬件和服务的事件。这些细节可以在我们调查事件时帮助我们，比如可移动设备的使用。一个可以证明在识别这种活动中有用的日志的例子是在 macOS 系统上找到的`daily.out`日志。这个日志记录了大量信息，包括连接到机器上的驱动器以及每天可用和已使用的存储量。虽然我们也可以从这个日志中了解关机时间、网络状态和其他信息，但我们将专注于随时间的驱动器使用情况。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。

# 如何做...

这个脚本将利用以下步骤：

1.  设置参数以接受日志文件和写入报告的路径。

1.  构建一个处理日志各个部分解析的类。

1.  创建一个提取相关部分并传递给进一步处理的方法。

1.  从这些部分提取磁盘信息。

1.  创建一个 CSV 写入器来导出提取的细节。

# 它是如何工作的...

我们首先导入必要的库来处理参数、解释日期和写入电子表格。在 Python 中处理文本文件的一个很棒的地方是你很少需要第三方库。

```py
from __future__ import print_function
from argparse import ArgumentParser, FileType
from datetime import datetime
import csv
```

这个食谱的命令行处理程序接受两个位置参数，`daily_out`和`output_report`，分别代表 daily.out 日志文件的路径和 CSV 电子表格的期望输出路径。请注意，我们通过`argparse.FileType`类传递一个打开的文件对象进行处理。随后，我们用日志文件初始化`ProcessDailyOut`类，并调用`run()`方法，并将返回的结果存储在`parsed_events`变量中。然后我们调用`write_csv()`方法，使用`processor`类对象中定义的列将结果写入到所需输出目录中的电子表格中。

```py
if __name__ == '__main__':
    parser = ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("daily_out", help="Path to daily.out file",
                        type=FileType('r'))
    parser.add_argument("output_report", help="Path to csv report")
    args = parser.parse_args()

    processor = ProcessDailyOut(args.daily_out)
    parsed_events = processor.run()
    write_csv(args.output_report, processor.report_columns, parsed_events)
```

在`ProcessDailyOut`类中，我们设置了用户提供的属性，并定义了报告中使用的列。请注意，我们添加了两组不同的列：`disk_status_columns`和`report_columns`。`report_columns`只是`disk_status_columns`，再加上两个额外的字段来标识条目的日期和时区。

```py
class ProcessDailyOut(object):
    def __init__(self, daily_out):
        self.daily_out = daily_out
        self.disk_status_columns = [
            'Filesystem', 'Size', 'Used', 'Avail', 'Capacity', 'iused',
            'ifree', '%iused', 'Mounted on']
        self.report_columns = ['event_date', 'event_tz'] + \
            self.disk_status_columns
```

`run()`方法首先遍历提供的日志文件。在从每行的开头和结尾去除空白字符后，我们验证内容以识别部分中断。`"-- End of daily output --"`字符串中断了日志文件中的每个条目。每个条目包含几个由新行分隔的数据部分。因此，我们必须使用几个代码块来分割和处理每个部分。

在这个循环中，我们收集来自单个事件的所有行，并将其传递给`process_event()`方法，并将处理后的结果追加到最终返回的`parsed_events`列表中。

```py
    def run(self):
        event_lines = []
        parsed_events = []
        for raw_line in self.daily_out:
            line = raw_line.strip()
            if line == '-- End of daily output --':
                parsed_events += self.process_event(event_lines)
                event_lines = []
            else:
                event_lines.append(line)
        return parsed_events
```

在`process_event()`方法中，我们将定义变量，以便我们可以分割事件的各个部分以进行进一步处理。为了更好地理解代码的下一部分，请花一点时间查看以下事件的示例：

![](img/00081.jpeg)

在此事件中，我们可以看到第一个元素是日期值和时区，后面是一系列子部分。每个子部分标题都是以冒号结尾的行；我们使用这一点来拆分文件中的各种数据元素，如下面的代码所示。我们使用部分标题作为键，其内容（如果存在）作为值，然后进一步处理每个子部分。

```py
    def process_event(self, event_lines):
        section_header = ""
        section_data = []
        event_data = {}
        for line in event_lines:
            if line.endswith(":"):
                if len(section_data) > 0:
                    event_data[section_header] = section_data
                    section_data = []
                    section_header = ""

                section_header = line.strip(":")
```

如果部分标题行不以冒号结尾，我们检查行中是否恰好有两个冒号。如果是这样，我们尝试将此行验证为日期值。为了处理此日期格式，我们需要从日期的其余部分单独提取时区，因为已知 Python 3 版本在解析带有`%Z`格式化程序的时区时存在已知错误。对于感兴趣的人，可以在[`bugs.python.org/issue22377`](https://bugs.python.org/issue22377)找到有关此错误的更多信息。

为了将时区与日期值分开，我们在空格值上分隔字符串，在这个示例中将时区值（元素`4`）放入自己的变量中，然后将剩余的时间值连接成一个新的字符串，我们可以使用`datetime`库解析。如果字符串没有至少`5`个元素，可能会引发`IndexError`，或者如果`datetime`格式字符串无效，可能会引发`ValueError`。如果没有引发这两种错误类型，我们将日期分配给`event_data`字典。如果我们收到这些错误中的任何一个，该行将附加到`section_data`列表中，并且下一个循环迭代将继续。这很重要，因为一行可能包含两个冒号，但不是日期值，所以我们不希望将其从脚本的考虑中移除。

```py
            elif line.count(":") == 2:
                try:
                    split_line = line.split()
                    timezone = split_line[4]
                    date_str = " ".join(split_line[:4] + [split_line[-1]])
                    try:
                        date_val = datetime.strptime(
                            date_str, "%a %b %d %H:%M:%S %Y")
                    except ValueError:
                        date_val = datetime.strptime(
                            date_str, "%a %b %d %H:%M:%S %Y")
                    event_data["event_date"] = [date_val, timezone]
                    section_data = []
                    section_header = ""
                except ValueError:
                    section_data.append(line)
                except IndexError:
                    section_data.append(line)
```

此条件的最后一部分将任何具有内容的行附加到`section_data`变量中，以根据需要进行进一步处理。这可以防止空白行进入，并允许我们捕获两个部分标题之间的所有信息。

```py
            else:
                if len(line):
                    section_data.append(line)
```

通过调用任何子部分处理器来关闭此函数。目前，我们只处理磁盘信息子部分，使用`process_disk()`方法，尽管可以开发代码来提取其他感兴趣的值。此方法接受事件信息和事件日期作为其输入。磁盘信息作为处理过的磁盘信息元素列表返回，我们将其返回给`run()`方法，并将值添加到处理过的事件列表中。

```py
        return self.process_disk(event_data.get("Disk status", []),
                                 event_data.get("event_date", []))
```

要处理磁盘子部分，我们遍历每一行，如果有的话，并提取相关的事件信息。`for`循环首先检查迭代号，并跳过行零，因为它包含数据的列标题。对于任何其他行，我们使用列表推导式，在单个空格上拆分行，去除空白，并过滤掉任何空字段。

```py
    def process_disk(self, disk_lines, event_dates):
        if len(disk_lines) == 0:
            return {}

        processed_data = []
        for line_count, line in enumerate(disk_lines):
            if line_count == 0:
                continue
            prepped_lines = [x for x in line.split(" ")
                             if len(x.strip()) != 0]
```

接下来，我们初始化一个名为`disk_info`的字典，其中包含了此快照的日期和时区详细信息。`for`循环使用`enumerate()`函数将值映射到它们的列名。如果列名包含`"/Volumes/"`（驱动器卷的标准挂载点），我们将连接剩余的拆分项。这样可以确保保留具有空格名称的卷。

```py
            disk_info = {
                "event_date": event_dates[0],
                "event_tz": event_dates[1]
            }
            for col_count, entry in enumerate(prepped_lines):
                curr_col = self.disk_status_columns[col_count]
                if "/Volumes/" in entry:
                    disk_info[curr_col] = " ".join(
                        prepped_lines[col_count:])
                    break
                disk_info[curr_col] = entry.strip()
```

最内层的`for`循环通过将磁盘信息附加到`processed_data`列表来结束。一旦磁盘部分中的所有行都被处理，我们就将`processed_data`列表返回给父函数。

```py
            processed_data.append(disk_info)
        return processed_data
```

最后，我们简要介绍了`write_csv()`方法，它使用`DictWriter`类来打开文件并将标题行和内容写入 CSV 文件。

```py
def write_csv(outfile, fieldnames, data):
    with open(outfile, 'w', newline="") as open_outfile:
        csvfile = csv.DictWriter(open_outfile, fieldnames)
        csvfile.writeheader()
        csvfile.writerows(data)
```

当我们运行这个脚本时，我们可以在 CSV 报告中看到提取出的细节。这里展示了这个输出的一个例子：

![](img/00082.jpeg)

# 将 daily.out 解析添加到 Axiom

教程难度：简单

Python 版本：2.7

操作系统：任意

使用我们刚刚开发的代码来解析 macOS 的`daily.out`日志，我们将这个功能添加到 Axiom 中，由*Magnet Forensics*开发，用于自动提取这些事件。由于 Axiom 支持处理取证镜像和松散文件，我们可以提供完整的获取或只是`daily.out`日志的导出作为示例。通过这个工具提供的 API，我们可以访问和处理其引擎发现的文件，并直接在 Axiom 中返回审查结果。

# 入门

Magnet Forensics 团队开发了一个 API，用于 Python 和 XML，以支持在 Axiom 中创建自定义 artifact。截至本书编写时，Python API 仅适用于运行 Python 版本 2.7 的`IronPython`。虽然我们在这个平台之外开发了我们的代码，但我们可以按照本教程中的步骤轻松地将其集成到 Axiom 中。我们使用了 Axiom 版本 1.1.3.5726 来测试和开发这个教程。

我们首先需要在 Windows 实例中安装 Axiom，并确保我们的代码稳定且可移植。此外，我们的代码需要在沙盒中运行。Axiom 沙盒限制了对第三方库的使用以及对可能导致代码与应用程序外部系统交互的一些 Python 模块和函数的访问。因此，我们设计了我们的`daily.out`解析器，只使用在沙盒中安全的内置库，以演示使用这些自定义 artifact 的开发的便利性。

# 如何做...

要开发和实现自定义 artifact，我们需要：

1.  在 Windows 机器上安装 Axiom。

1.  导入我们开发的脚本。

1.  创建`Artifact`类并定义解析器元数据和列。

1.  开发`Hunter`类来处理 artifact 处理和结果报告。

# 工作原理...

对于这个脚本，我们导入了`axiom`库和 datetime 库。请注意，我们已经删除了之前的`argparse`和`csv`导入，因为它们在这里是不必要的。

```py
from __future__ import print_function
from axiom import *
from datetime import datetime
```

接下来，我们必须粘贴前一个教程中的`ProcessDailyOut`类，不包括`write_csv`或参数处理代码，以在这个脚本中使用。由于当前版本的 API 不允许导入，我们必须将所有需要的代码捆绑到一个单独的脚本中。为了节省页面并避免冗余，我们将在本节中省略代码块（尽管它在本章附带的代码文件中存在）。

下一个类是`DailyOutArtifact`，它是 Axiom API 提供的`Artifact`类的子类。在定义插件的名称之前，我们调用`AddHunter()`方法，提供我们的（尚未显示的）`hHunter`类。

```py
class DailyOutArtifact(Artifact):
    def __init__(self):
        self.AddHunter(DailyOutHunter())

    def GetName(self):
        return 'daily.out parser'
```

这个类的最后一个方法`CreateFragments()`指定了如何处理已处理的 daily.out 日志结果的单个条目。就 Axiom API 而言，片段是用来描述 artifact 的单个条目的术语。这段代码允许我们添加自定义列名，并为这些列分配适当的类别和数据类型。这些类别包括日期、位置和工具定义的其他特殊值。我们 artifact 的大部分列将属于`None`类别，因为它们不显示特定类型的信息。

一个重要的分类区别是`DateTimeLocal`与`DateTime`：`DateTime`将日期呈现为 UTC 值呈现给用户，因此我们需要注意选择正确的日期类别。因为我们从 daily.out 日志条目中提取了时区，所以在这个示例中我们使用`DateTimeLocal`类别。`FragmentType`属性是所有值的字符串，因为该类不会将值从字符串转换为其他数据类型。

```py
    def CreateFragments(self):
        self.AddFragment('Snapshot Date - LocalTime (yyyy-mm-dd)',
                         Category.DateTimeLocal, FragmentType.DateTime)
        self.AddFragment('Snapshot Timezone', Category.None,
                         FragmentType.String)
        self.AddFragment('Volume Name',
                         Category.None, FragmentType.String)
        self.AddFragment('Filesystem Mount',
                         Category.None, FragmentType.String)
        self.AddFragment('Volume Size',
                         Category.None, FragmentType.String)
        self.AddFragment('Volume Used',
                         Category.None, FragmentType.String)
        self.AddFragment('Percentage Used',
                         Category.None, FragmentType.String)
```

接下来的类是我们的`Hunter`。这个父类用于运行处理代码，并且正如你将看到的，指定了将由 Axiom 引擎提供给插件的平台和内容。在这种情况下，我们只想针对计算机平台和一个单一名称的文件运行。`RegisterFileName()`方法是指定插件将请求哪些文件的几种选项之一。我们还可以使用正则表达式或文件扩展名来选择我们想要处理的文件。

```py
class DailyOutHunter(Hunter):
    def __init__(self):
        self.Platform = Platform.Computer

    def Register(self, registrar):
        registrar.RegisterFileName('daily.out')
```

`Hunt()`方法是魔法发生的地方。首先，我们获取一个临时路径，在沙箱内可以读取文件，并将其分配给`temp_daily_out`变量。有了这个打开的文件，我们将文件对象交给`ProcessDailyOut`类，并使用`run()`方法解析文件，就像上一个示例中一样。

```py
    def Hunt(self, context):
        temp_daily_out = open(context.Searchable.FileCopy, 'r')

        processor = ProcessDailyOut(temp_daily_out)
        parsed_events = processor.run()
```

在收集了解析的事件信息之后，我们准备将数据“发布”到软件并显示给用户。在`for`循环中，我们首先初始化一个`Hit()`对象，使用`AddValue()`方法向新片段添加数据。一旦我们将事件值分配给了一个 hit，我们就使用`PublishHit()`方法将 hit 发布到平台，并继续循环直到所有解析的事件都被发布：

```py
        for entry in parsed_events:
            hit = Hit()
            hit.AddValue(
                "Snapshot Date - LocalTime (yyyy-mm-dd)",
                entry['event_date'].strftime("%Y-%m-%d %H:%M:%S"))
            hit.AddValue("Snapshot Timezone", entry['event_tz'])
            hit.AddValue("Volume Name", entry['Mounted on'])
            hit.AddValue("Filesystem Mount", entry["Filesystem"])
            hit.AddValue("Volume Size", entry['Size'])
            hit.AddValue("Volume Used", entry['Used'])
            hit.AddValue("Percentage Used", entry['Capacity'])
            self.PublishHit(hit)
```

最后一部分代码检查文件是否不是`None`，如果是，则关闭它。这是处理代码的结尾，如果在系统上发现另一个`daily.out`文件，可能会再次调用它！

```py
        if temp_daily_out is not None:
            temp_daily_out.close()
```

最后一行注册了我们的辛勤工作到 Axiom 的引擎，以确保它被框架包含和调用。

```py
RegisterArtifact(DailyOutArtifact())
```

要在 Axiom 中使用新开发的工件，我们需要采取一些步骤来导入并针对图像运行代码。首先，我们需要启动 Axiom Process。这是我们将加载、选择并针对提供的证据运行工件的地方。在工具菜单下，我们选择管理自定义工件选项：

![](img/00083.jpeg)

在管理自定义工件窗口中，我们将看到任何现有的自定义工件，并可以像这样导入新的工件：

![](img/00084.jpeg)

我们将添加我们的自定义工件，更新的管理自定义工件窗口应该显示工件的名称：

![](img/00085.jpeg)

现在我们可以按下确定并继续进行 Axiom，添加证据并配置我们的处理选项。当我们到达计算机工件选择时，我们要确认选择运行自定义工件。可能不用说：我们应该只在机器运行 macOS 或者在其上有 macOS 分区时运行这个工件：

![](img/00086.jpeg)

完成剩余的配置选项后，我们可以开始处理证据。处理完成后，我们运行 Axiom Examine 来查看处理结果。如下截图所示，我们可以导航到工件审查的自定义窗格，并看到插件解析的列！这些列可以使用 Axiom 中的标准选项进行排序和导出，而无需我们额外的代码：

![](img/00087.jpeg)

# 使用 YARA 扫描指示器

配方难度：中等

Python 版本：3.5

操作系统：任何

作为一个额外的部分，我们将利用强大的**Yet Another Recursive Algorithm**（**YARA**）正则表达式引擎来扫描感兴趣的文件和妥协指标。YARA 是一种用于恶意软件识别和事件响应的模式匹配实用程序。许多工具使用此引擎作为识别可能恶意文件的基础。通过这个示例，我们学习如何获取 YARA 规则，编译它们，并在一个或多个文件夹或文件中进行匹配。虽然我们不会涵盖形成 YARA 规则所需的步骤，但可以从他们的文档中了解更多关于这个过程的信息[`yara.readthedocs.io/en/latest/writingrules.html`](http://yara.readthedocs.io/en/latest/writingrules.html)。

# 入门

此示例需要安装第三方库`yara`。此脚本中使用的所有其他库都包含在 Python 的标准库中。可以使用`pip`安装此库：

```py
pip install yara-python==3.6.3
```

要了解更多关于`yara-python`库的信息，请访问[`yara.readthedocs.io/en/latest/`](https://yara.readthedocs.io/en/latest/)。

我们还可以使用项目如 YaraRules ([`yararules.com`](http://yararules.com))，并使用行业和 VirusShare ([`virusshare.com`](http://virusshare.com))的预构建规则来使用真实的恶意软件样本进行分析。

# 如何做...

此脚本有四个主要的开发步骤：

1.  设置和编译 YARA 规则。

1.  扫描单个文件。

1.  遍历目录以处理单个文件。

1.  将结果导出到 CSV。

# 它是如何工作的...

此脚本导入所需的库来处理参数解析、文件和文件夹迭代、编写 CSV 电子表格，以及`yara`库来编译和扫描 YARA 规则。

```py
from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import csv
import yara
```

这个示例的命令行处理程序接受两个位置参数，`yara_rules`和`path_to_scan`，分别表示 YARA 规则的路径和要扫描的文件或文件夹。此示例还接受一个可选参数`output`，如果提供，将扫描结果写入电子表格而不是控制台。最后，我们将这些值传递给`main()`方法。

```py
if __name__ == '__main__':
    parser = ArgumentParser(
        description=__description__,
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument(
        'yara_rules',
        help="Path to Yara rule to scan with. May be file or folder path.")
    parser.add_argument(
        'path_to_scan',
        help="Path to file or folder to scan")
    parser.add_argument(
        '--output',
        help="Path to output a CSV report of scan results")
    args = parser.parse_args()

    main(args.yara_rules, args.path_to_scan, args.output)
```

在`main()`函数中，我们接受`yara`规则的路径、要扫描的文件或文件夹以及输出文件（如果有）。由于`yara`规则可以是文件或目录，我们使用`ios.isdir()`方法来确定我们是否在整个目录上使用`compile()`方法，或者如果输入是一个文件，则使用`filepath`关键字将其传递给该方法。`compile()`方法读取规则文件或文件并创建一个我们可以与我们扫描的对象进行匹配的对象。

```py
def main(yara_rules, path_to_scan, output):
    if os.path.isdir(yara_rules):
        yrules = yara.compile(yara_rules)
    else:
        yrules = yara.compile(filepath=yara_rules)
```

一旦规则被编译，我们执行类似的`if-else`语句来处理要扫描的路径。如果要扫描的输入是一个目录，我们将其传递给`process_directory()`函数，否则，我们使用`process_file()`方法。两者都使用编译后的 YARA 规则和要扫描的路径，并返回包含任何匹配项的字典列表。

```py
    if os.path.isdir(path_to_scan):
        match_info = process_directory(yrules, path_to_scan)
    else:
        match_info = process_file(yrules, path_to_scan)
```

正如你可能猜到的，如果指定了输出路径，我们最终将把这个字典列表转换为 CSV 报告，使用我们在`columns`列表中定义的列。然而，如果输出参数是`None`，我们将以不同的格式将这些数据写入控制台。

```py
    columns = ['rule_name', 'hit_value', 'hit_offset', 'file_name',
               'rule_string', 'rule_tag']

    if output is None:
        write_stdout(columns, match_info)
    else:
        write_csv(output, columns, match_info)
```

`process_directory()`函数本质上是遍历目录并将每个文件传递给`process_file()`函数。这减少了脚本中冗余代码的数量。返回的每个处理过的条目都被添加到`match_info`列表中，因为返回的对象是一个列表。一旦我们处理了每个文件，我们将完整的结果列表返回给父函数。

```py
def process_directory(yrules, folder_path):
    match_info = []
    for root, _, files in os.walk(folder_path):
        for entry in files:
            file_entry = os.path.join(root, entry)
            match_info += process_file(yrules, file_entry)
    return match_info
```

`process_file()` 方法使用了 `yrules` 对象的 `match()` 方法。返回的匹配对象是一个可迭代的对象，包含了一个或多个与规则匹配的结果。从匹配结果中，我们可以提取规则名称、任何标签、文件中的偏移量、规则的字符串值以及匹配结果的字符串值。这些信息加上文件路径将形成报告中的一条记录。总的来说，这些信息对于确定匹配结果是误报还是重要的非常有用。在微调 YARA 规则以确保只呈现相关结果进行审查时也非常有帮助。

```py
def process_file(yrules, file_path):
    match = yrules.match(file_path)
    match_info = []
    for rule_set in match:
        for hit in rule_set.strings:
            match_info.append({
                'file_name': file_path,
                'rule_name': rule_set.rule,
                'rule_tag': ",".join(rule_set.tags),
                'hit_offset': hit[0],
                'rule_string': hit[1],
                'hit_value': hit[2]
            })
    return match_info
```

`write_stdout()` 函数如果用户没有指定输出文件，则将匹配信息报告到控制台。我们遍历 `match_info` 列表中的每个条目，并以冒号分隔、换行分隔的格式打印出 `match_info` 字典中的每个列名及其值。在每个条目之后，我们打印 `30` 个等号来在视觉上将条目分隔开。

```py
def write_stdout(columns, match_info):
    for entry in match_info:
        for col in columns:
            print("{}: {}".format(col, entry[col]))
        print("=" * 30)
```

`write_csv()` 方法遵循标准约定，使用 `DictWriter` 类来写入标题和所有数据到表格中。请注意，这个函数已经调整为在 Python 3 中处理 CSV 写入，使用了 `'w'` 模式和 `newline` 参数。

```py
def write_csv(outfile, fieldnames, data):
    with open(outfile, 'w', newline="") as open_outfile:
        csvfile = csv.DictWriter(open_outfile, fieldnames)
        csvfile.writeheader()
        csvfile.writerows(data)
```

使用这段代码，我们可以在命令行提供适当的参数，并生成任何匹配的报告。以下截图显示了用于检测 Python 文件和键盘记录器的自定义规则：

![](img/00088.jpeg)

这些规则显示在输出的 CSV 报告中，如果没有指定报告，则显示在控制台中，如下所示：

![](img/00089.jpeg)
