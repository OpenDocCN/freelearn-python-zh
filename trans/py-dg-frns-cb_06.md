# 阅读电子邮件和获取名称的配方

本章涵盖了以下配方：

+   解析 EML 文件

+   查看 MSG 文件

+   订购外卖

+   盒子里有什么？

+   解析 PST 和 OST 邮箱

# 介绍

一旦计算机证据被添加到混乱中，他说她说的游戏通常就被抛到一边。电子邮件在大多数类型的调查中起着重要作用。电子邮件证据涉及到商业和个人设备，因为它被广泛用于发送文件、与同行交流以及从在线服务接收通知。通过检查电子邮件，我们可以了解托管人使用哪些社交媒体、云存储或其他网站。我们还可以寻找组织外的数据外流，或者调查钓鱼计划的来源。

本章将涵盖揭示此信息以进行调查的配方，包括：

+   使用内置库读取 EML 格式

+   利用`win32com`库从 Outlook MSG 文件中提取信息

+   使用 Takeouts 保存 Google Gmail 并解析保存内容

+   使用内置库从 MBOX 容器中读取

+   使用`libpff`读取 PST 文件

访问[www.packtpub.com/books/content/support](http://www.packtpub.com/books/content/support)下载本章的代码包。

# 解析 EML 文件

配方难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

EML 文件格式被广泛用于存储电子邮件消息，因为它是一个结构化的文本文件，兼容多个电子邮件客户端。这个文本文件以纯文本形式存储电子邮件头部、正文内容和附件数据，使用`base64`来编码二进制数据，使用**Quoted-Printable**（**QP**）编码来存储内容信息。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。我们将使用内置的`email`库来读取和提取 EML 文件中的关键信息。

要了解更多关于`email`库的信息，请访问[`docs.python.org/3/library/email.html`](https://docs.python.org/3/library/email.html)。

# 如何做...

要创建一个 EML 解析器，我们必须：

1.  接受一个 EML 文件的参数。

1.  从头部读取值。

1.  从 EML 的各个部分中解析信息。

1.  在控制台中显示此信息以便审查。

# 它是如何工作的...

我们首先导入用于处理参数、EML 处理和解码 base64 编码数据的库。`email`库提供了从 EML 文件中读取数据所需的类和方法。我们将使用`message_from_file()`函数来解析提供的 EML 文件中的数据。`Quopri`是本书中的一个新库，我们使用它来解码 HTML 正文和附件中的 QP 编码值。`base64`库，正如人们所期望的那样，允许我们解码任何 base64 编码的数据：

```py
from __future__ import print_function
from argparse import ArgumentParser, FileType
from email import message_from_file
import os
import quopri
import base64
```

此配方的命令行处理程序接受一个位置参数`EML_FILE`，表示我们将处理的 EML 文件的路径。我们使用`FileType`类来处理文件的打开：

```py
if __name__ == '__main__':
    parser = ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("EML_FILE",
                        help="Path to EML File", type=FileType('r'))
    args = parser.parse_args()

    main(args.EML_FILE)
```

在`main()`函数中，我们使用`message_from_file()`函数将类似文件的对象读入`email`库。现在我们可以使用结果变量`emlfile`来访问头部、正文内容、附件和其他有效载荷信息。读取电子邮件头部只是通过迭代库的`_headers`属性提供的字典来处理。要处理正文内容，我们必须检查此消息是否包含多个有效载荷，并且如果是这样，将每个传递给指定的处理函数`process_payload()`：

```py
def main(input_file):
    emlfile = message_from_file(input_file)

    # Start with the headers
    for key, value in emlfile._headers:
        print("{}: {}".format(key, value))

    # Read payload
    print("\nBody\n")
    if emlfile.is_multipart():
        for part in emlfile.get_payload():
            process_payload(part)
    else:
        process_payload(emlfile[1])
```

`process_payload()`函数首先通过使用`get_content_type()`方法提取消息的 MIME 类型。我们将这个值打印到控制台上，并在新行上打印一些`"="`字符来区分这个值和消息的其余部分。

在一行中，我们使用`get_payload()`方法提取消息正文内容，并使用`quopri.decodestring()`函数解码 QP 编码的数据。然后，我们检查数据是否有字符集，如果我们确定了字符集，则在指定字符集的同时使用`decode()`方法对内容进行解码。如果编码是未知的，我们将尝试使用 UTF8 对对象进行解码，这是在将`decode()`方法留空时的默认值，以及 Windows-1252：

```py
def process_payload(payload):
    print(payload.get_content_type() + "\n" + "=" * len(
        payload.get_content_type()))
    body = quopri.decodestring(payload.get_payload())
    if payload.get_charset():
        body = body.decode(payload.get_charset())
    else:
        try:
            body = body.decode()
        except UnicodeDecodeError:
            body = body.decode('cp1252')
```

使用我们解码的数据，我们检查内容的 MIME 类型，以便正确处理电子邮件的存储。 HTML 信息的第一个条件，由`text/html` MIME 类型指定，被写入到与输入文件相同目录中的 HTML 文档中。在第二个条件中，我们处理`Application` MIME 类型下的二进制数据。这些数据以`base64`编码的值传输，我们在使用`base64.b64decode()`函数写入到当前目录中的文件之前对其进行解码。二进制数据具有`get_filename()`方法，我们可以使用它来准确命名附件。请注意，输出文件必须以`"w"`模式打开第一种类型，以`"wb"`模式打开第二种类型。如果 MIME 类型不是我们在这里涵盖的类型，我们将在控制台上打印正文：

```py
    if payload.get_content_type() == "text/html":
        outfile = os.path.basename(args.EML_FILE.name) + ".html"
        open(outfile, 'w').write(body)
    elif payload.get_content_type().startswith('application'):
        outfile = open(payload.get_filename(), 'wb')
        body = base64.b64decode(payload.get_payload())
        outfile.write(body)
        outfile.close()
        print("Exported: {}\n".format(outfile.name))
    else:
        print(body)
```

当我们执行此代码时，我们首先在控制台上看到头信息，然后是各种有效载荷。在这种情况下，我们首先有一个`text/plain` MIME 内容，其中包含一个示例消息，然后是一个`application/vnd.ms-excel`附件，我们将其导出，然后是另一个`text/plain`块显示初始消息：

![](img/00064.jpeg)![](img/00065.jpeg)

# 查看 MSG 文件

配方难度：简单

Python 版本：2.7 或 3.5

操作系统：Windows

电子邮件消息可以以许多不同的格式出现。MSG 格式是存储消息内容和附件的另一种流行容器。在这个例子中，我们将学习如何使用 Outlook API 解析 MSG 文件。

# 入门

这个配方需要安装第三方库`pywin32`。这意味着该脚本只能在 Windows 系统上兼容。我们还需要安装`pywin32`，就像我们在第一章中所做的那样，*基本脚本和文件信息配方*。

要安装`pywin32`，我们需要访问其 SourceForge 页面[`sourceforge.net/projects/pywin32/`](https://sourceforge.net/projects/pywin32/)，并下载与您的 Python 安装相匹配的版本。要检查我们的 Python 版本，我们可以导入`sys`模块，并在解释器中调用`sys.version`。在选择正确的`pywin32`安装程序时，版本和架构都很重要。我们还希望确认我们在计算机上安装了有效的 Outlook，因为`pywin32`绑定依赖于 Outlook 提供的资源。在运行`pywin32`安装程序后，我们准备创建脚本。

# 如何做...

要创建 MSG 解析器，我们必须：

1.  接受一个 MSG 文件的参数。

1.  将有关 MSG 文件的一般元数据打印到控制台。

1.  将特定于收件人的元数据打印到控制台。

1.  将消息内容导出到输出文件。

1.  将嵌入在消息中的任何附件导出到适当的输出文件。

# 它是如何工作的...

我们首先导入用于参数处理的库`argparse`和`os`，然后是来自`pywin32`的`win32com`库。我们还导入`pywintypes`库以正确捕获和处理`pywin32`错误：

```py
from __future__ import print_function
from argparse import ArgumentParser
import os
import win32com.client
import pywintypes
```

这个配方的命令行处理程序接受两个位置参数，`MSG_FILE`和`OUTPUT_DIR`，分别表示要处理的 MSG 文件的路径和所需的输出文件夹。我们检查所需的输出文件夹是否存在，如果不存在，则创建它。之后，我们将这两个输入传递给`main()`函数：

```py
if __name__ == '__main__':
    parser = ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("MSG_FILE", help="Path to MSG file")
    parser.add_argument("OUTPUT_DIR", help="Path to output folder")
    args = parser.parse_args()
    out_dir = args.OUTPUT_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    main(args.MSG_FILE, args.OUTPUT_DIR)
```

在 `main()` 函数中，我们调用 `win32com` 库来设置 Outlook API，以便以允许访问 `MAPI` 命名空间的方式进行配置。使用这个 `mapi` 变量，我们可以使用 `OpenSharedItem()` 方法打开一个 `MSG` 文件，并创建一个我们将在本示例中使用的对象。这些函数包括：`display_msg_attribs()`、`display_msg_recipients()`、`extract_msg_body()` 和 `extract_attachments()`。现在让我们依次关注这些函数，看看它们是如何工作的：

```py
def main(msg_file, output_dir):
    mapi = win32com.client.Dispatch(
        "Outlook.Application").GetNamespace("MAPI")
    msg = mapi.OpenSharedItem(os.path.abspath(args.MSG_FILE))
    display_msg_attribs(msg)
    display_msg_recipients(msg)
    extract_msg_body(msg, output_dir)
    extract_attachments(msg, output_dir)
```

`display_msg_attribs()` 函数允许我们显示消息的各种属性（主题、收件人、密件抄送、大小等）。其中一些属性可能不存在于我们解析的消息中，但是我们尝试导出所有值。`attribs` 列表按顺序显示我们尝试从消息中访问的属性。当我们遍历每个属性时，我们在 `msg` 对象上使用内置的 `getattr()` 方法，并尝试提取相关值（如果存在），如果不存在，则为 `"N/A"`。然后我们将属性及其确定的值打印到控制台。需要注意的是，其中一些值可能存在，但仅设置为默认值，例如某些日期的年份为 `4501`：

```py
def display_msg_attribs(msg):
    # Display Message Attributes
    attribs = [
        'Application', 'AutoForwarded', 'BCC', 'CC', 'Class',
        'ConversationID', 'ConversationTopic', 'CreationTime',
        'ExpiryTime', 'Importance', 'InternetCodePage', 'IsMarkedAsTask',
        'LastModificationTime', 'Links', 'OriginalDeliveryReportRequested',
        'ReadReceiptRequested', 'ReceivedTime', 'ReminderSet',
        'ReminderTime', 'ReplyRecipientNames', 'Saved', 'Sender',
        'SenderEmailAddress', 'SenderEmailType', 'SenderName', 'Sent',
        'SentOn', 'SentOnBehalfOfName', 'Size', 'Subject',
        'TaskCompletedDate', 'TaskDueDate', 'To', 'UnRead'
    ]
    print("\nMessage Attributes")
    print("==================")
    for entry in attribs:
        print("{}: {}".format(entry, getattr(msg, entry, 'N/A')))
```

`display_msg_recipients()` 函数遍历消息并显示收件人详细信息。`msg` 对象提供了一个 `Recipients()` 方法，该方法接受一个整数参数以按索引访问收件人。使用 `while` 循环，我们尝试加载和显示可用收件人的值。对于找到的每个收件人，与之前的函数一样，我们使用 `getattr()` 方法与属性列表 `recipient_attrib` 提取和打印相关值，或者如果它们不存在，则赋予它们值 `"N/A"`。尽管大多数 Python 可迭代对象使用零作为第一个索引，但 `Recipients()` 方法从 `1` 开始。因此，变量 `i` 将从 `1` 开始递增，直到找不到更多的收件人为止。我们将继续尝试读取这些值，直到收到 `pywin32` 错误。

```py
def display_msg_recipients(msg):
    # Display Recipient Information
    recipient_attrib = [
        'Address', 'AutoResponse', 'Name', 'Resolved', 'Sendable'
    ]
    i = 1
    while True:
        try:
            recipient = msg.Recipients(i)
        except pywintypes.com_error:
            break

        print("\nRecipient {}".format(i))
        print("=" * 15)
        for entry in recipient_attrib:
            print("{}: {}".format(entry, getattr(recipient, entry, 'N/A')))
        i += 1
```

`extract_msg_body()` 函数旨在从消息中提取正文内容。`msg` 对象以几种不同的格式公开正文内容；在本示例中，我们将导出 HTML（使用 `HTMLBody()` 方法）和纯文本（使用 `Body()` 方法）版本的正文。由于这些对象是字节字符串，我们必须首先解码它们，这是通过使用 `cp1252` 代码页来完成的。有了解码后的内容，我们打开用户指定目录中的输出文件，并创建相应的 `*.body.html` 和 `*.body.txt` 文件：

```py
def extract_msg_body(msg, out_dir):
    # Extract HTML Data
    html_data = msg.HTMLBody.encode('cp1252')
    outfile = os.path.join(out_dir, os.path.basename(args.MSG_FILE))
    open(outfile + ".body.html", 'wb').write(html_data)
    print("Exported: {}".format(outfile + ".body.html"))

    # Extract plain text
    body_data = msg.Body.encode('cp1252')
    open(outfile + ".body.txt", 'wb').write(body_data)
    print("Exported: {}".format(outfile + ".body.txt"))
```

最后，`extract_attachments()` 函数将附件数据从 MSG 文件导出到所需的输出目录。使用 `msg` 对象，我们再次创建一个列表 `attachment_attribs`，表示有关附件的一系列属性。与收件人函数类似，我们使用 `while` 循环和 `Attachments()` 方法，该方法接受一个整数作为参数，以选择要迭代的附件。与之前的 `Recipients()` 方法一样，`Attachments()` 方法从 `1` 开始索引。因此，变量 `i` 将从 `1` 开始递增，直到找不到更多的附件为止：

```py
def extract_attachments(msg, out_dir):
    attachment_attribs = [
        'DisplayName', 'FileName', 'PathName', 'Position', 'Size'
    ]
    i = 1 # Attachments start at 1
    while True:
        try:
            attachment = msg.Attachments(i)
        except pywintypes.com_error:
            break
```

对于每个附件，我们将其属性打印到控制台。我们提取和打印的属性在此函数开始时的 `attachment_attrib` 列表中定义。打印可用附件详细信息后，我们使用 `SaveAsFile()` 方法写入其内容，并提供一个包含输出路径和所需输出附件名称的字符串（使用 `FileName` 属性获取）。之后，我们准备移动到下一个附件，因此我们递增变量 `i` 并尝试访问下一个附件。

```py
        print("\nAttachment {}".format(i))
        print("=" * 15)
        for entry in attachment_attribs:
            print('{}: {}'.format(entry, getattr(attachment, entry,
                                                 "N/A")))
        outfile = os.path.join(os.path.abspath(out_dir),
                               os.path.split(args.MSG_FILE)[-1])
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        outfile = os.path.join(outfile, attachment.FileName)
        attachment.SaveAsFile(outfile)
        print("Exported: {}".format(outfile))
        i += 1
```

当我们执行此代码时，我们将看到以下输出，以及输出目录中的几个文件。这包括正文文本和 HTML，以及任何发现的附件。消息及其附件的属性将显示在控制台窗口中。

![](img/00066.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个建议如下：

+   考虑通过参考 MSDN 上 MSG 对象的属性来向解析器添加更多字段[`msdn.microsoft.com/en-us/library/microsoft.office.interop.outlook.mailitem_properties.aspx`](https://msdn.microsoft.com/en-us/library/microsoft.office.interop.outlook.mailitem_properties.aspx)

# 另请参阅

还存在其他用于访问 MSG 文件的库，包括`Redemption`库。该库提供了访问标头信息的处理程序，以及与此示例中显示的许多相同属性。

# 订购外卖

教程难度：简单

Python 版本：N/A

操作系统：任何

谷歌邮件，通常称为 Gmail，是更广泛使用的网络邮件服务之一。Gmail 帐户不仅可以作为电子邮件地址，还可以作为通往谷歌提供的众多其他服务的入口。除了通过网络或**Internet Message Access Protocol**（**IMAP**）和**Post Office Protocol**（**POP**）邮件协议提供邮件访问外，谷歌还开发了一种用于存档和获取 Gmail 帐户中存储的邮件和其他相关数据的系统。

# 入门

信不信由你，这个教程实际上不涉及任何 Python，而是需要浏览器和对 Google 帐户的访问。这个教程的目的是以 MBOX 格式获取 Google 帐户邮箱，我们将在下一个教程中解析它。

# 如何做...

要启动 Google Takeout，我们按照以下步骤进行：

1.  登录到相关的谷歌帐户。

1.  导航到帐户设置和创建存档功能。

1.  选择要存档的所需谷歌产品并开始该过程。

1.  下载存档数据。

# 它是如何工作的...

我们通过登录帐户并选择“我的帐户”选项来开始 Google Takeout 过程。如果“我的帐户”选项不存在，我们也可以导航到[`myaccount.google.com`](https://myaccount.google.com)：

![](img/00067.jpeg)

在“我的帐户”仪表板上，我们选择“个人信息和隐私”部分下的“控制您的内容”链接：

![](img/00068.jpeg)

在“控制您的内容”部分，我们将看到一个“创建存档”的选项。这是我们开始 Google Takeout 收集的地方：

![](img/00069.jpeg)

选择此选项时，我们将看到管理现有存档或生成新存档的选项。生成新存档时，我们将看到每个我们希望包括的 Google 产品的复选框。下拉箭头提供子菜单，可更改导出格式或内容。例如，我们可以选择将 Google Drive 文档导出为 Microsoft Word、PDF 或纯文本格式。在这种情况下，我们将保留选项为默认值，确保邮件选项设置为收集所有邮件：

![](img/00070.jpeg)

选择所需的内容后，我们可以配置存档的格式。Google Takeout 允许我们选择存档文件类型和最大段大小，以便轻松下载和访问。我们还可以选择如何访问 Takeout。此选项可以设置为将下载链接发送到被存档的帐户（默认选项）或将存档上传到帐户的 Google Drive 或其他第三方云服务，这可能会修改比必要更多的信息以保留这些数据。我们选择接收电子邮件，然后选择“创建存档”以开始该过程！

![](img/00071.jpeg)

现在我们必须等待。根据要保存的数据大小，这可能需要相当长的时间，因为 Google 必须为您收集、转换和压缩所有数据。

当您收到通知电子邮件时，请选择提供的链接下载存档。此存档仅在有限的时间内可用，因此在收到通知后尽快收集它是很重要的。

![](img/00072.jpeg)

下载数据后，提取存档的内容并查看内部文件夹结构和提供的数据。所选的每个产品都有一个包含相关内容或产品的文件夹结构的文件夹。在这种情况下，我们最感兴趣的是以 MBOX 格式提供的邮件。在下一个配方中，我们将展示如何使用 Python 解析这些 MBOX 数据。

# 还有更多...

如果您更喜欢更直接的方式来获取这些数据，您可以在登录账户后导航到[`takeout.google.com/settings/takeout`](https://takeout.google.com/settings/takeout)。在这里，您可以选择要导出的产品。

# 盒子里有什么?!

配方难度：中等

Python 版本：3.5

操作系统：任何

MBOX 文件通常与 UNIX 系统、Thunderbird 和 Google Takeouts 相关联。这些 MBOX 容器是具有特殊格式的文本文件，用于分割存储在其中的消息。由于有几种用于构造 MBOX 文件的格式，我们的脚本将专注于来自 Google Takeout 的格式，使用前一个配方的输出。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。我们使用内置的`mailbox`库来解析 Google Takeout 结构化的 MBOX 文件。

要了解更多关于`mailbox`库的信息，请访问[`docs.python.org/3/library/mailbox.html`](https://docs.python.org/3/library/mailbox.html)。

# 如何做...

要实现这个脚本，我们必须：

1.  设计参数以接受 MBOX 文件的文件路径并输出报告内容。

1.  开发一个处理编码数据的自定义 MBOX 阅读器。

1.  提取消息元数据，包括附件名称。

1.  将附件写入输出目录。

1.  创建一个 MBOX 元数据报告。

# 它是如何工作的...

我们首先导入用于处理参数的库，然后是用于创建脚本输出的`os`、`time`和`csv`库。接下来，我们导入`mailbox`库来解析 MBOX 消息格式和`base64`来解码附件中的二进制数据。最后，我们引入`tqdm`库来提供与消息解析状态相关的进度条：

```py
from __future__ import print_function
from argparse import ArgumentParser
import mailbox
import os
import time
import csv
from tqdm import tqdm
import base64
```

这个配方的命令行处理程序接受两个位置参数，`MBOX`和`OUTPUT_DIR`，分别表示要处理的 MBOX 文件的路径和期望的输出文件夹。这两个参数都传递给`main()`函数来启动脚本：

```py
if __name__ == '__main__':
    parser = ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("MBOX", help="Path to mbox file")
    parser.add_argument("OUTPUT_DIR",
                        help="Path to output directory to write report "
                        "and exported content")
    args = parser.parse_args()

    main(args.MBOX, args.OUTPUT_DIR)
```

`main()`函数从调用`mailbox`库的`mbox`类开始。使用这个类，我们可以通过提供文件路径和一个可选的工厂参数来解析 MBOX 文件，这在我们的情况下是一个自定义阅读器函数。使用这个库，我们现在有一个包含我们可以交互的消息对象的可迭代对象。我们使用内置的`len()`方法来打印 MBOX 文件中包含的消息数量。让我们首先看看`custom_reader()`函数是如何工作的：

```py
def main(mbox_file, output_dir):
    # Read in the MBOX File
    print("Reading mbox file...")
    mbox = mailbox.mbox(mbox_file, factory=custom_reader)
    print("{} messages to parse".format(len(mbox)))
```

这个配方需要一些函数来运行（看到我们做了什么吗...），但`custom_reader()`方法与其他方法有些不同。这个函数是`mailbox`库的一个阅读器方法。我们需要创建这个函数，因为默认的阅读器不能处理诸如`cp1252`之类的编码。我们可以将其他编码添加到这个阅读器中，尽管 ASCII 和`cp1252`是 MBOX 文件的两种最常见的编码。

在输入数据流上使用`read()`方法后，它尝试使用 ASCII 代码页对数据进行解码。如果不成功，它将依赖`cp1252`代码页来完成任务。使用`cp1252`代码页解码时遇到的任何错误都将被替换为替换字符`U+FFFD`，通过向`decode()`方法提供`errors`关键字并将其设置为`"replace"`来实现。我们使用`mailbox.mboxMessage()`函数以适当的格式返回解码后的内容：

```py
def custom_reader(data_stream):
    data = data_stream.read()
    try:
        content = data.decode("ascii")
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        content = data.decode("cp1252", errors="replace")
    return mailbox.mboxMessage(content)
```

回到`main()`函数，在开始处理消息之前，我们准备了一些变量。具体来说，我们设置了`parsed_data`结果列表，为附件创建了一个输出目录，并定义了 MBOX 元数据报告的`columns`。这些列也将用于使用`get()`方法从消息中提取信息。其中两列不会从消息对象中提取信息，而是在处理附件后包含我们分配的数据。为了保持一致性，我们将这些值保留在`columns`列表中，因为它们将默认为`"N/A"`值：

```py
    parsed_data = []
    attachments_dir = os.path.join(output_dir, "attachments")
    if not os.path.exists(attachments_dir):
        os.makedirs(attachments_dir)
    columns = ["Date", "From", "To", "Subject", "X-Gmail-Labels",
               "Return-Path", "Received", "Content-Type", "Message-ID",
               "X-GM-THRID", "num_attachments_exported", "export_path"]
```

当我们开始迭代消息时，我们实现了一个`tqdm`进度条来跟踪迭代过程。由于`mbox`对象具有长度属性，因此我们不需要为`tqdm`提供任何额外的参数。在循环内部，我们定义了`msg_data`字典来存储消息结果，然后尝试通过第二个`for`循环使用`get()`方法在`header_data`字典中查询`columns`键来分配消息属性：

```py
    for message in tqdm(mbox):
        # Preserve header information
        msg_data = dict()
        header_data = dict(message._headers)
        for hdr in columns:
            msg_data[hdr] = header_data.get(hdr, "N/A")
```

接下来，在一个`if`语句中，我们检查`message`是否具有有效载荷，如果有，我们使用`write_payload()`方法，向其提供`message`对象和输出附件目录作为输入。如果`message`没有有效载荷，那么两个与附件相关的列将保持默认的`"N/A"`值。否则，我们计算找到的附件数量，并将它们的路径列表连接成逗号分隔的列表：

```py
        if len(message.get_payload()):
            export_path = write_payload(message, attachments_dir)
            msg_data['num_attachments_exported'] = len(export_path)
            msg_data['export_path'] = ", ".join(export_path)
```

每处理完一条消息，其数据都会被附加到`parsed_data`列表中。在处理完所有消息后，将调用`create_report()`方法，并传递`parsed_data`列表和所需的输出 CSV 名称。让我们回溯一下，首先看一下`write_payload()`方法：

```py
        parsed_data.append(msg_data)

    # Create CSV report
    create_report(
        parsed_data, os.path.join(output_dir, "mbox_report.csv"), columns
    )
```

由于消息可能具有各种各样的有效载荷，我们需要编写一个专门的函数来处理各种`MIME`类型。`write_payload()`方法就是这样一个函数。该函数首先通过`get_payload()`方法提取有效载荷，并进行快速检查，看看有效载荷内容是否包含多个部分。如果是，我们会递归调用此函数来处理每个子部分，通过迭代有效载荷并将输出附加到`export_path`变量中：

```py
def write_payload(msg, out_dir):
    pyld = msg.get_payload()
    export_path = []
    if msg.is_multipart():
        for entry in pyld:
            export_path += write_payload(entry, out_dir)
```

如果有效载荷不是多部分的，我们使用`get_content_type()`方法确定其 MIME 类型，并创建逻辑来根据类别适当地处理数据源。应用程序、图像和视频等数据类型通常表示为`base64`编码数据，允许将二进制信息作为 ASCII 字符传输。因此，大多数格式（包括文本类别中的一些格式）都要求我们在提供写入之前对数据进行解码。在其他情况下，数据已存在为字符串，并且可以按原样写入文件。无论如何，方法通常是相同的，数据被解码（如果需要），并使用`export_content()`方法将其内容写入文件系统。最后，表示导出项目路径的字符串被附加到`export_path`列表中：

```py
    else:
        content_type = msg.get_content_type()
        if "application/" in content_type.lower():
            content = base64.b64decode(msg.get_payload())
            export_path.append(export_content(msg, out_dir, content))
        elif "image/" in content_type.lower():
            content = base64.b64decode(msg.get_payload())
            export_path.append(export_content(msg, out_dir, content))
        elif "video/" in content_type.lower():
            content = base64.b64decode(msg.get_payload())
            export_path.append(export_content(msg, out_dir, content))
        elif "audio/" in content_type.lower():
            content = base64.b64decode(msg.get_payload())
            export_path.append(export_content(msg, out_dir, content))
        elif "text/csv" in content_type.lower():
            content = base64.b64decode(msg.get_payload())
            export_path.append(export_content(msg, out_dir, content))
        elif "info/" in content_type.lower():
            export_path.append(export_content(msg, out_dir,
                                              msg.get_payload()))
        elif "text/calendar" in content_type.lower():
            export_path.append(export_content(msg, out_dir,
                                              msg.get_payload()))
        elif "text/rtf" in content_type.lower():
            export_path.append(export_content(msg, out_dir,
                                              msg.get_payload()))
```

`else` 语句在负载中添加了一个额外的 `if-elif` 语句，以确定导出是否包含文件名。如果有，我们将其视为其他文件，但如果没有，它很可能是存储为 HTML 或文本的消息正文。虽然我们可以通过修改这一部分来导出每个消息正文，但这将为本示例生成大量数据，因此我们选择不这样做。一旦我们完成了从消息中导出数据，我们将导出的数据的路径列表返回给 `main()` 函数：

```py
        else:
            if "name=" in msg.get('Content-Disposition', "N/A"):
                content = base64.b64decode(msg.get_payload())
                export_path.append(export_content(msg, out_dir, content))
            elif "name=" in msg.get('Content-Type', "N/A"):
                content = base64.b64decode(msg.get_payload())
                export_path.append(export_content(msg, out_dir, content))

    return export_path
```

`export_content()` 函数首先调用 `get_filename()` 函数，这个方法从 `msg` 对象中提取文件名。对文件名进行额外处理以提取扩展名（如果有的话），如果没有找到则使用通用的 `.FILE` 扩展名：

```py
def export_content(msg, out_dir, content_data):
    file_name = get_filename(msg)
    file_ext = "FILE"
    if "." in file_name:
        file_ext = file_name.rsplit(".", 1)[-1]
```

接下来，我们进行额外的格式化，通过整合时间（表示为 Unix 时间整数）和确定的文件扩展名来创建一个唯一的文件名。然后将此文件名连接到输出目录，形成用于写入输出的完整路径。这个唯一的文件名确保我们不会错误地覆盖输出目录中已经存在的附件：

```py
    file_name = "{}_{:.4f}.{}".format(
        file_name.rsplit(".", 1)[0], time.time(), file_ext)
    file_name = os.path.join(out_dir, file_name)
```

这个函数中代码的最后一部分处理文件内容的实际导出。这个 `if` 语句处理不同的文件模式（`"w"` 或 `"wb"`），根据源类型。写入数据后，我们返回用于导出的文件路径。这个路径将被添加到我们的元数据报告中：

```py
    if isinstance(content_data, str):
        open(file_name, 'w').write(content_data)
    else:
        open(file_name, 'wb').write(content_data)

    return file_name
```

下一个函数 `get_filename()` 从消息中提取文件名以准确表示这些文件的名称。文件名可以在 `"Content-Disposition"` 或 `"Content-Type"` 属性中找到，并且通常以 `"name="` 或 `"filename="` 字符串开头。对于这两个属性，逻辑基本相同。该函数首先用一个空格替换任何换行符，然后在分号和空格上拆分字符串。这个分隔符通常分隔这些属性中的值。使用列表推导，我们确定哪个元素包含 `name=` 子字符串，并将其用作文件名：

```py
def get_filename(msg):
    if 'name=' in msg.get("Content-Disposition", "N/A"):
        fname_data = msg["Content-Disposition"].replace("\r\n", " ")
        fname = [x for x in fname_data.split("; ") if 'name=' in x]
        file_name = fname[0].split("=", 1)[-1]

    elif 'name=' in msg.get("Content-Type", "N/A"):
        fname_data = msg["Content-Type"].replace("\r\n", " ")
        fname = [x for x in fname_data.split("; ") if 'name=' in x]
        file_name = fname[0].split("=", 1)[-1]
```

如果这两个内容属性为空，我们分配一个通用的 `NO_FILENAME` 并继续准备文件名。提取潜在的文件名后，我们删除任何不是字母数字、空格或句号的字符，以防止在系统中写入文件时出错。准备好我们的文件系统安全文件名后，我们将其返回供前面讨论的 `export_content()` 方法使用：

```py
    else:
        file_name = "NO_FILENAME"

    fchars = [x for x in file_name if x.isalnum() or x.isspace() or
              x == "."]
    return "".join(fchars)
```

最后，我们已经到达了准备讨论 CSV 元数据报告的阶段。`create_report()` 函数类似于本书中我们已经看到的各种变体，它使用 `DictWriter` 类从字典列表创建 CSV 报告。哒哒！

```py
def create_report(output_data, output_file, columns):
    with open(output_file, 'w', newline="") as outfile:
        csvfile = csv.DictWriter(outfile, columns)
        csvfile.writeheader()
        csvfile.writerows(output_data)
```

这个脚本创建了一个 CSV 报告和一个附件目录。第一个截图显示了 CSV 报告的前几列和行以及数据如何显示给用户：

![](img/00073.jpeg)

这第二个截图显示了这些相同行的最后几列，并反映了附件信息的报告方式。这些文件路径可以被跟踪以访问相应的附件：

![](img/00074.jpeg)

# 解析 PST 和 OST 邮箱

食谱难度：困难

Python 版本：2.7

操作系统：Linux

**个人存储表**（**PST**）文件通常在许多系统上找到，并提供对归档电子邮件的访问。这些文件通常与 Outlook 应用程序相关联，包含消息和附件数据。这些文件通常在企业环境中找到，因为许多商业环境继续利用 Outlook 进行内部和外部电子邮件管理。

# 入门指南

该配方需要安装`libpff`及其 Python 绑定`pypff`才能正常运行。这个库在 GitHub 上提供了工具和 Python 绑定，用于处理和提取 PST 文件中的数据。我们将在 Ubuntu 16.04 上为 Python 2 设置这个库以便开发。这个库也可以为 Python 3 构建，不过在本节中我们将使用 Python 2 的绑定。

在安装所需的库之前，我们必须安装一些依赖项。使用 Ubuntu 的`apt`软件包管理器，我们将安装以下八个软件包。您可能希望将这个 Ubuntu 环境保存好，因为我们将在第八章以及以后的章节中广泛使用它：

```py
sudo apt-get install automake autoconf libtool pkg-config autopoint git python-dev
```

安装依赖项后，转到 GitHub 存储库并下载所需的库版本。这个配方是使用`pypff`库的`libpff-experimental-20161119`版本开发的。接下来，一旦提取了发布的内容，打开终端并导航到提取的目录，并执行以下命令以进行发布：

```py
./synclibs.sh
./autogen.sh
sudo python setup.py install 
```

要了解有关`pypff`库的更多信息，请访问[`github.com/libyal/libpff`](https://github.com/libyal/libpff)。

最后，我们可以通过打开 Python 解释器，导入`pypff`并运行`pypff.get_version()`方法来检查库的安装情况，以确保我们有正确的发布版本。

# 如何操作...

我们按照以下步骤提取 PST 消息内容：

1.  使用`pypff`为 PST 文件创建一个句柄。

1.  遍历 PST 中的所有文件夹和消息。

1.  存储每条消息的相关元数据。

1.  根据 PST 的内容创建元数据报告。

# 工作原理...

该脚本首先导入用于处理参数、编写电子表格、执行正则表达式搜索和处理 PST 文件的库：

```py
from __future__ import print_function
from argparse import ArgumentParser
import csv
import pypff
import re
```

此配方的命令行处理程序接受两个位置参数，`PFF_FILE`和`CSV_REPORT`，分别表示要处理的 PST 文件的路径和所需的输出 CSV 路径。在这个配方中，我们不使用`main()`函数，而是立即使用`pypff.file()`对象来实例化`pff_obj`变量。随后，我们使用`open()`方法并尝试访问用户提供的 PST。我们将此 PST 传递给`process_folders()`方法，并将返回的字典列表存储在`parsed_data`变量中。在对`pff_obj`变量使用`close()`方法后，我们使用`write_data()`函数写入 PST 元数据报告，通过传递所需的输出 CSV 路径和处理后的数据字典：

```py
if __name__ == '__main__':
    parser = ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("PFF_FILE", help="Path to PST or OST File")
    parser.add_argument("CSV_REPORT", help="Path to CSV report location")
    args = parser.parse_args()

    # Open file
    pff_obj = pypff.file()
    pff_obj.open(args.PFF_FILE)

    # Parse and close file
    parsed_data = process_folders(pff_obj.root_folder)
    pff_obj.close()

    # Write CSV report
    write_data(args.CSV_REPORT, parsed_data)
```

这个配方由几个处理 PST 文件不同元素的函数组成。`process_folders()`函数处理文件夹处理和迭代。在处理这些文件夹时，我们将它们的名称、子文件夹的数量以及该文件夹中的消息数量打印到控制台。这可以通过在`pff_folder`对象上调用`number_of_sub_folders`和`number_of_sub_messages`属性来实现：

```py
def process_folders(pff_folder):
    folder_name = pff_folder.name if pff_folder.name else "N/A"
    print("Folder: {} (sub-dir: {}/sub-msg: {})".format(folder_name,
          pff_folder.number_of_sub_folders,
          pff_folder.number_of_sub_messages))
```

在打印这些消息后，我们设置了`data_list`，它负责存储处理过的消息数据。当我们遍历文件夹中的消息时，我们调用`process_message()`方法来创建带有处理过的消息数据的字典对象。紧接着，我们将文件夹名称添加到字典中，然后将其附加到结果列表中。

第二个循环通过递归调用`process_folders()`函数并将子文件夹传递给它，然后将结果字典列表附加到`data_list`中。这使我们能够遍历 PST 并提取所有数据，然后返回`data_list`并编写 CSV 报告：

```py
    # Process messages within a folder
    data_list = []
    for msg in pff_folder.sub_messages:
        data_dict = process_message(msg)
        data_dict['folder'] = folder_name
        data_list.append(data_dict)

    # Process folders within a folder
    for folder in pff_folder.sub_folders:
        data_list += process_folders(folder)

    return data_list
```

`process_message()` 函数负责访问消息的各种属性，包括电子邮件头信息。正如在以前的示例中所看到的，我们使用对象属性的列表来构建结果的字典。然后我们遍历`attribs`字典，并使用`getattr()`方法将适当的键值对附加到`data_dict`字典中。最后，如果存在电子邮件头，我们通过使用`transport_headers`属性来确定，我们将从`process_headers()`函数中提取的附加值更新到`data_dict`字典中：

```py
def process_message(msg):
    # Extract attributes
    attribs = ['conversation_topic', 'number_of_attachments',
               'sender_name', 'subject']
    data_dict = {}
    for attrib in attribs:
        data_dict[attrib] = getattr(msg, attrib, "N/A")

    if msg.transport_headers is not None:
        data_dict.update(process_headers(msg.transport_headers))

    return data_dict
```

`process_headers()` 函数最终返回一个包含提取的电子邮件头数据的字典。这些数据以键值对的形式显示，由冒号和空格分隔。由于头部中的内容可能存储在新的一行上，我们使用正则表达式来检查是否在行首有一个键，后面跟着一个值。如果我们找不到与模式匹配的键（任意数量的字母或破折号字符后跟着一个冒号），我们将把新值附加到先前的键上，因为头部以顺序方式显示信息。在这个函数的结尾，我们有一些特定的代码行，使用`isinstance()`来处理字典值的赋值。这段代码检查键的类型，以确保值被分配给键的方式不会覆盖与给定键关联的任何数据：

```py
def process_headers(header):
    # Read and process header information
    key_pattern = re.compile("^([A-Za-z\-]+:)(.*)$")
    header_data = {}
    for line in header.split("\r\n"):
        if len(line) == 0:
            continue

        reg_result = key_pattern.match(line)
        if reg_result:
            key = reg_result.group(1).strip(":").strip()
            value = reg_result.group(2).strip()
        else:
            value = line

        if key.lower() in header_data:
            if isinstance(header_data[key.lower()], list):
                header_data[key.lower()].append(value)
            else:
                header_data[key.lower()] = [header_data[key.lower()],
                                            value]
        else:
            header_data[key.lower()] = value
    return header_data
```

最后，`write_data()` 方法负责创建元数据报告。由于我们可能从电子邮件头解析中有大量的列名，我们遍历数据并提取不在列表中已定义的不同列名。使用这种方法，我们确保来自 PST 的动态信息不会被排除。在`for`循环中，我们还将`data_list`中的值重新分配到`formatted_data_list`中，主要是将列表值转换为字符串，以更容易地将数据写入电子表格。`csv`库很好地确保了单元格内的逗号被转义并由我们的电子表格应用程序适当处理：

```py
def write_data(outfile, data_list):
    # Build out additional columns
    print("Writing Report: ", outfile)
    columns = ['folder', 'conversation_topic', 'number_of_attachments',
               'sender_name', 'subject']
    formatted_data_list = []
    for entry in data_list:
        tmp_entry = {}

        for k, v in entry.items():
            if k not in columns:
                columns.append(k)

            if isinstance(v, list):
                tmp_entry[k] = ", ".join(v)
            else:
                tmp_entry[k] = v
        formatted_data_list.append(tmp_entry)
```

使用`csv.DictWriter`类，我们打开文件，写入头部和每一行到输出文件：

```py
    # Write CSV report
    with open(outfile, 'wb') as openfile:
        csvfile = csv.DictWriter(openfile, columns)
        csvfile.writeheader()
        csvfile.writerows(formatted_data_list)
```

当这个脚本运行时，将生成一个 CSV 报告，其外观应该与以下截图中显示的类似。在水平滚动时，我们可以看到在顶部指定的列名；特别是在电子邮件头列中，大多数这些列只包含少量的值。当您在您的环境中对更多的电子邮件容器运行此代码时，请注意哪些列是最有用的，并且在您处理 PST 时最常见，以加快分析的速度：

![](img/00075.jpeg)

# 还有更多...

这个过程可以进一步改进。我们提供了一个或多个以下建议：

+   这个库还处理**离线存储表**（**OST**）文件，通常与 Outlook 的离线邮件内容存储相关。找到并测试这个脚本在 OST 文件上，并在必要时修改以支持这种常见的邮件格式。

# 另请参阅

在这种情况下，我们还可以利用`Redemtion`库来访问 Outlook 中的信息。
