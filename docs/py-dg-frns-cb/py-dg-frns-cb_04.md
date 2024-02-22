# 第四章：提取嵌入式元数据配方

本章涵盖以下配方：

+   提取音频和视频元数据

+   大局观

+   挖掘 PDF 元数据

+   审查可执行文件元数据

+   阅读办公文档元数据

+   将我们的元数据提取器与 EnCase 集成

# 介绍

当调查仅涉及少数感兴趣的文件时，提取有关文件的每一条可用信息至关重要。经常被忽视的嵌入式元数据可以为我们提供巩固给定文件证据价值的关键信息。无论是从 Microsoft Office 文件中收集作者信息，从图片中映射 GPS 坐标，还是从可执行文件中提取编译信息，我们都可以更多地了解我们正在调查的文件。在本章中，我们将开发脚本来检查这些文件格式以及其他文件格式，以提取我们审查的关键信息。我们将说明如何将这些配方与流行的取证套件 EnCase 集成，并将它们添加到您的调查工作流程中。

特别是，我们将开发突出以下内容的代码：

+   解析音频和视频格式的 ID3 和 QuickTime 格式的元数据

+   揭示嵌入在图像中的 GPS 坐标

+   从 PDF 文件中识别作者和谱系信息

+   从 Windows 可执行文件中提取嵌入的名称、编译日期和其他属性的信息

+   报告 Microsoft Office 文件的创建和来源

+   从 EnCase 启动 Python 脚本

访问[www.packtpub.com/books/content/support](http://www.packtpub.com/books/content/support)下载本章的代码包。

# 提取音频和视频元数据

配方难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

音频和视频文件是常见的文件格式，它们使用嵌入式元数据。例如，您喜欢的媒体播放器使用此信息来显示您导入的内容的艺术家、专辑和曲目名称信息。尽管大多数信息是标准的并专注于向听众提供信息，但我们有时会在文件的这个领域找到重要的细节。我们从提取音频和视频文件的常见属性开始探索嵌入式元数据。

# 入门

此配方需要安装第三方库`mutagen`。此脚本中使用的所有其他库都包含在 Python 的标准库中。此库允许我们从音频和视频文件中提取元数据。可以使用`pip`安装此库：

```py
pip install mutagen==1.38
```

要了解有关`mutagen`库的更多信息，请访问[`mutagen.readthedocs.io/en/latest`](https://mutagen.readthedocs.io/en/latest)。

# 如何做...

在这个脚本中，我们执行以下步骤：

1.  识别输入文件类型。

1.  从文件类型处理器中提取嵌入式元数据。

# 它是如何工作的...

要从示例 MP3 或 MP4 文件中提取信息，我们首先导入此配方所需的三个库：`argparse`、`json`和`mutagen`。`json`库允许我们加载稍后在此配方中使用的 QuickTime MP4 元数据格式的定义。

```py
from __future__ import print_function
import argparse
import json
import mutagen
```

此配方的命令行处理程序接受一个位置参数`AV_FILE`，表示要处理的 MP3 或 MP4 文件的路径。在解析用户提供的参数之后，我们使用`mutagen.File（）`方法打开文件的句柄。根据输入文件的扩展名，我们将此句柄发送到适当的函数：`handle_id3（）`或`handle_mp4（）`。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog="Developed by {} on {}".format(
            ", ".join(__authors__), __date__)
    )
    parser.add_argument("AV_FILE", help="File to extract metadata from")
    args = parser.parse_args()
    av_file = mutagen.File(args.AV_FILE)

    file_ext = args.AV_FILE.rsplit('.', 1)[-1]
    if file_ext.lower() == 'mp3':
        handle_id3(av_file)
    elif file_ext.lower() == 'mp4':
        handle_mp4(av_file)
```

`handle_id3（）`函数负责从 MP3 文件中提取元数据。MP3 格式使用 ID3 标准来存储其元数据。在我们的 ID3 解析函数中，我们首先创建一个名为`id3_frames`的字典，将 ID3 字段（在原始文件中表示）映射到人类可读的字符串。我们可以向此定义添加更多字段，以扩展我们提取的信息。在提取嵌入式元数据之前，我们将适当的列标题打印到控制台。

```py
def handle_id3(id3_file):
    # Definitions from http://id3.org/id3v2.4.0-frames
    id3_frames = {
        'TIT2': 'Title', 'TPE1': 'Artist', 'TALB': 'Album',
        'TXXX': 'Custom', 'TCON': 'Content Type', 'TDRL': 'Date released',
        'COMM': 'Comments', 'TDRC': 'Recording Date'}
    print("{:15} | {:15} | {:38} | {}".format("Frame", "Description",
                                              "Text", "Value"))
    print("-" * 85)
```

接下来，我们使用循环提取每个`id3`帧的名称和各种值。我们查询帧的名称以从`id3_frames`字典中提取其人类可读版本。此外，从每个帧中，我们使用`getattr()`方法提取描述、文本和值（如果存在）。最后，我们将管道分隔的文本打印到控制台进行审查。这样处理了 MP3 文件，现在让我们转到 MP4 文件。

```py
    for frames in id3_file.tags.values():
        frame_name = id3_frames.get(frames.FrameID, frames.FrameID)
        desc = getattr(frames, 'desc', "N/A")
        text = getattr(frames, 'text', ["N/A"])[0]
        value = getattr(frames, 'value', "N/A")
        if "date" in frame_name.lower():
            text = str(text)

        print("{:15} | {:15} | {:38} | {}".format(
            frame_name, desc, text, value))
```

`handle_mp4()`函数负责处理 MP4 文件，并且遵循与之前函数类似的工作流程。我们首先在一个名为`qt_tag`的字典中设置元数据映射，使用版权符号（`u"\u00A9"`）的 Unicode 值作为字段名称的前置字符。这个映射字典被设计成标签名称是键，人类可读的字符串是值。然后，我们使用`json.load()`方法导入了一个大型的媒体类型定义列表（喜剧、播客、乡村等）。通过将 JSON 数据存储到`genre_ids`变量中，这种情况下，我们有一个包含不同类型的键值对的字典，其中键是整数，值是不同的类型。这些定义来自[`www.sno.phy.queensu.ca/~phil/exiftool/TagNames/QuickTime.html#GenreID`](http://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/QuickTime.html#GenreID)。

```py
def handle_mp4(mp4_file):
    cp_sym = u"\u00A9"
    qt_tag = {
        cp_sym + 'nam': 'Title', cp_sym + 'art': 'Artist',
        cp_sym + 'alb': 'Album', cp_sym + 'gen': 'Genre',
        'cpil': 'Compilation', cp_sym + 'day': 'Creation Date',
        'cnID': 'Apple Store Content ID', 'atID': 'Album Title ID',
        'plID': 'Playlist ID', 'geID': 'Genre ID', 'pcst': 'Podcast',
        'purl': 'Podcast URL', 'egid': 'Episode Global ID',
        'cmID': 'Camera ID', 'sfID': 'Apple Store Country',
        'desc': 'Description', 'ldes': 'Long Description'}
    genre_ids = json.load(open('apple_genres.json'))
```

接下来，我们遍历 MP4 文件的嵌入式元数据键值对。对于每个键，我们使用`qt_tag`字典查找键的人类可读版本。如果值是一个列表，我们将其所有元素连接成一个以分号分隔的字符串。或者，如果值是`"geID"`，我们使用`genre_ids`字典查找整数，并为用户打印映射的类型。

```py
    print("{:22} | {}".format('Name', 'Value'))
    print("-" * 40)
    for name, value in mp4_file.tags.items():
        tag_name = qt_tag.get(name, name)
        if isinstance(value, list):
            value = "; ".join([str(x) for x in value])
        if name == 'geID':
            value = "{}: {}".format(
                value, genre_ids[str(value)].replace("|", " - "))
        print("{:22} | {}".format(tag_name, value))
```

使用 MP3 播客作为示例，脚本显示了其他不可用的详细信息。现在我们知道了发布日期，似乎是使用的软件，以及一些标识符，我们可以用作关键字来尝试在其他地方识别文件。

![](img/00031.jpeg)

让我们再看一个播客，但这次是一个 MP4 文件。运行脚本后，我们将得到关于 MP4 文件来源和内容类型的大量信息。同样，由于这个练习，我们可以获得一些有趣的标识符、来源 URL 和其他归因细节。

![](img/00032.jpeg)

# 还有更多...

这个脚本可以进一步改进。这里有一个建议：

+   使用`mutagen`库为其他多媒体格式添加额外支持。

# 大局观

食谱难度：简单

Python 版本：2.7 或 3.5

操作系统：任意

图像可以包含许多元数据属性，取决于文件格式和用于拍摄图像的设备。幸运的是，大多数设备会在它们拍摄的照片中嵌入 GPS 信息。使用第三方库，我们将提取 GPS 坐标并在 Google Earth 中绘制它们。这个脚本专注于这个任务，但是这个食谱可以很容易地调整，以提取所有嵌入的**可交换图像文件格式**（**EXIF**）元数据在 JPEG 和 TIFF 图像中。

# 入门

这个食谱需要安装两个第三方库：`pillow`和`simplekml`。此脚本中使用的所有其他库都包含在 Python 的标准库中。`pillow`库提供了一个清晰的接口，用于从图像中提取嵌入的元数据：

```py
pip install pillow==4.2.1
```

要了解更多关于`pillow`库的信息，请访问[`pillow.readthedocs.io/en/4.2.x/`](https://pillow.readthedocs.io/en/4.2.x/)。

为了给这个食谱增添一些额外的风采，我们将把 GPS 详细信息写入一个 KML 文件，以便在类似 Google Earth 的程序中使用。为了处理这个问题，我们将使用`simplekml`库，可以通过执行以下命令进行安装：

```py
pip install simplekml==1.3.0
```

要了解更多关于`simplekml`库的信息，请访问[`www.simplekml.com/en/latest/`](http://www.simplekml.com/en/latest/)。

# 操作步骤如下...

我们按以下步骤从图像文件中提取元数据：

1.  用`PIL`打开输入照片。

1.  使用`PIL`提取所有`EXIF`标签。

1.  如果找到 GPS 坐标，创建一个 Google Earth KML 文件。

1.  打印 Google Maps URL 以在浏览器中查看 GPS 数据。

# 工作原理...

我们首先导入`argparse`以及新安装的`simplekml`和`PIL`库。在本例中，我们只需要从`PIL`中的`Image`和**`ExifTags.Tags`**类。

```py
from __future__ import print_function
import argparse
from PIL import Image
from PIL.ExifTags import TAGS
import simplekml
import sys
```

这个配方的命令行处理程序接受一个位置参数`PICTURE_FILE`，它代表要处理的照片的文件路径。

```py
parser = argparse.ArgumentParser(
    description=__description__,
    epilog="Developed by {} on {}".format(", ".join(__authors__), __date__)
)
parser.add_argument('PICTURE_FILE', help="Path to picture")
args = parser.parse_args()
```

配置这些参数后，我们指定两个 URL，`gmaps`和`open_maps`，我们将用坐标信息填充它们。由于`PIL`库以**度分秒**（**DMS**）格式的元组元组提供坐标，我们需要一个函数将它们转换为另一种常用的坐标表示格式——十进制。提供的元组中的三个元素分别代表坐标的不同组成部分。此外，在每个元组中，有两个元素：第一个元素代表值，第二个元素是必须用来将值转换为整数的比例。

对于坐标的每个组成部分，我们需要将嵌套元组中的第一个值除以第二个值。这种结构用于描述 DMS 坐标的第二个和第三个元组，此外，我们需要确保通过将每个值除以当前迭代计数的`60`的乘积来正确地将分钟和秒相加（这将是`1`和`2`）。虽然这不会改变第一个值（因为枚举从零开始），但它将确保第二和第三个值被正确表示。

以下代码块突出了`PIL`库提供的坐标格式的示例。请注意，度、分和秒值被分组到它们自己的元组中。第一个元素代表坐标的值，第二个代表比例。例如，对于秒元素（第三个元组），我们需要在执行其他操作之前将整数除以`1000`，以确保值被正确表示。

+   **纬度**：`((41 , 1), (53 , 1), (23487 , 1000))`

+   **经度**：`((12 , 1), (29 , 1), (10362 , 1000))`

+   **GPS 坐标**：`41.8898575 , 12.486211666666666`

```py
gmaps = "https://www.google.com/maps?q={},{}"
open_maps = "http://www.openstreetmap.org/?mlat={}&mlon={}"

def process_coords(coord):
    coord_deg = 0
    for count, values in enumerate(coord):
        coord_deg += (float(values[0]) / values[1]) / 60**count
    return coord_deg
```

配置了 DMS 到十进制坐标转换过程后，我们使用`Image.open()`方法打开图像，以将文件路径作为`PIL`对象打开。然后，我们使用`_getexif()`方法提取包含 EXIF 数据的字典。如果`PIL`无法从照片中提取元数据，这个变量将是`None`。

使用`EXIF`字典，我们遍历键和值，将数值转换为可读的名称。这使用了`PIL`中的`TAGS`字典，它将数值映射到表示标签的`string`。`TAGS`对象的作用方式类似于先前配方中手动指定的映射。

```py
img_file = Image.open(args.PICTURE_FILE)
exif_data = img_file._getexif()

if exif_data is None:
    print("No EXIF data found")
    sys.exit()

for name, value in exif_data.items():
    gps_tag = TAGS.get(name, name)
    if gps_tag is not 'GPSInfo':
        continue
```

一旦找到`GPSInfo`标签，我们提取字典键`1`到`4`中的四个感兴趣的值。成对地，我们存储 GPS 参考并使用先前描述的`process_coords()`方法处理坐标。通过将参考存储为布尔值，我们可以轻松地使用`if`语句来确定 GPS 十进制坐标是正数还是负数。

```py
    lat_ref = value[1] == u'N'
    lat = process_coords(value[2])
    if not lat_ref:
        lat = lat * -1

    lon_ref = value[3] == u'E'
    lon = process_coords(value[4])
    if not lon_ref:
        lon = lon * -1
```

为了添加我们的 KML 支持，我们从`simplekml`库初始化一个`kml`对象。从那里，我们添加一个新的点，带有一个名称和坐标。对于名称，我们简单地使用文件的名称。坐标被提供为一个元组，其中第一个元素是经度，第二个元素是纬度。我们也可以在这个元组中提供第三个元素来指定缩放级别，尽管在这种情况下我们省略了它。为了生成我们的`KML`文件，我们调用`save()`方法并将其写入一个与输入文件同名的`.kml`文件。

```py
    kml = simplekml.Kml()
    kml.newpoint(name=args.PICTURE_FILE, coords=[(lon, lat)])
    kml.save(args.PICTURE_FILE + ".kml")
```

有了处理过的 GPS 信息，我们可以在控制台上打印坐标、KML 文件和 URL。注意我们如何嵌套格式字符串，允许我们打印基本消息以及 URL。

```py
    print("GPS Coordinates: {}, {}".format(lat, lon))
    print("Google Maps URL: {}".format(gmaps.format(lat, lon)))
    print("OpenStreetMap URL: {}".format(open_maps.format(lat, lon)))
    print("KML File {} created".format(args.PICTURE_FILE + ".kml"))
```

当我们在命令行上运行这个脚本时，我们很快就能看到坐标、两个链接以查看地图上的位置，以及 KML 文件的路径。

![](img/00033.jpeg)

根据我们生成的两个链接，我们可以在两个地图上看到标记，并在需要时与其他人分享这些链接。

![](img/00034.jpeg)![](img/00035.jpeg)

最后，我们可以使用 KML 文件来存储和引用图像中找到的位置。Google Earth 允许通过 Web 和桌面客户端查看这个文件。

![](img/00036.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个建议如下：

+   集成文件递归以处理多张照片，创建包含许多 GPS 坐标的更大的 KML 文件。

+   尝试使用`simplekml`库为每个点添加额外的细节，比如描述、时间戳、着色等。

# 挖掘 PDF 元数据

配方难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

虽然 PDF 文档可以表示各种各样的媒体，包括图像、文本和表单，但它们包含结构化的嵌入式元数据，以**可扩展元数据平台**（**XMP**）格式提供了一些额外的信息。通过这个配方，我们使用 Python 访问 PDF 并提取描述文档创建和传承的元数据。

# 入门

这个配方需要安装第三方库`PyPDF2`。这个脚本中使用的所有其他库都包含在 Python 的标准库中。`PyPDF2`模块为我们提供了读写 PDF 文件的绑定。在我们的情况下，我们只会使用这个库来读取以 XMP 格式存储的元数据。要安装这个库，请运行以下命令：

```py
pip install PyPDF2==1.26.0
```

要了解更多关于`PyPDF2`库的信息，请访问[`mstamy2.github.io/PyPDF2/`](http://mstamy2.github.io/PyPDF2/)。

# 如何做...

为了处理这个配方的 PDF，我们按照以下步骤进行：

1.  用`PyPDF2`打开 PDF 文件并提取嵌入式元数据。

1.  为不同的 Python 对象类型定义自定义打印函数。

1.  打印各种嵌入式元数据属性。

# 它是如何工作的...

首先，我们导入`argparse`、`datetime`和`sys`库，以及新安装的`PyPDF2`模块。

```py
from __future__ import print_function
from argparse import ArgumentParser, FileType
import datetime
from PyPDF2 import PdfFileReader
import sys
```

这个配方的命令行处理程序接受一个位置参数`PDF_FILE`，表示要处理的 PDF 文件的文件路径。对于这个脚本，我们需要将一个打开的文件对象传递给`PdfFileReader`类，所以我们使用`argparse.FileType`处理程序来为我们打开文件。

```py
parser = ArgumentParser(
    description=__description__,
    epilog="Developed by {} on {}".format(", ".join(__authors__), __date__)
)
parser.add_argument('PDF_FILE', help='Path to PDF file',
                    type=FileType('rb'))
args = parser.parse_args()
```

将打开的文件提供给`PdfFileReader`类后，我们调用`getXmpMetadata()`方法来提供一个包含可用 XMP 元数据的对象。如果这个方法返回`None`，我们会在退出之前向用户打印一个简洁的消息。

```py
pdf_file = PdfFileReader(args.PDF_FILE)

xmpm = pdf_file.getXmpMetadata()
if xmpm is None:
    print("No XMP metadata found in document.")
    sys.exit()
```

有了`xmpm`对象准备就绪，我们开始提取和打印相关值。我们提取了许多不同的值，包括标题、创建者、贡献者、描述、创建和修改日期。这些值的定义来自[`wwwimages.adobe.com/content/dam/Adobe/en/devnet/xmp/pdfs/XMP%20SDK%20Release%20cc-2016-08/XMPSpecificationPart1.pdf`](http://wwwimages.adobe.com/content/dam/Adobe/en/devnet/xmp/pdfs/XMP%20SDK%20Release%20cc-2016-08/XMPSpecificationPart1.pdf)。尽管这些元素中的许多是不同的数据类型，我们以相同的方式将它们传递给`custom_print()`方法。让我们看看这个函数是如何工作的。

```py
custom_print("Title: {}", xmpm.dc_title)
custom_print("Creator(s): {}", xmpm.dc_creator)
custom_print("Contributors: {}", xmpm.dc_contributor)
custom_print("Subject: {}", xmpm.dc_subject)
custom_print("Description: {}", xmpm.dc_description)
custom_print("Created: {}", xmpm.xmp_createDate)
custom_print("Modified: {}", xmpm.xmp_modifyDate)
custom_print("Event Dates: {}", xmpm.dc_date)
```

由于存储的 XMP 值可能因用于生成 PDF 的软件而异，我们使用一个名为`custom_print()`的自定义打印处理函数。这使我们可以处理列表、字典、日期和其他值的转换为可读格式。这个函数是可移植的，可以根据需要引入其他脚本。该函数通过一系列`if-elif-else`语句检查输入的`value`是否是支持的对象类型，使用内置的`isinstance()`方法并适当处理它们。如果输入的`value`是不受支持的类型，则会将其打印到控制台。

```py
def custom_print(fmt_str, value):
    if isinstance(value, list):
        print(fmt_str.format(", ".join(value)))
    elif isinstance(value, dict):
        fmt_value = [":".join((k, v)) for k, v in value.items()]
        print(fmt_str.format(", ".join(value)))
    elif isinstance(value, str) or isinstance(value, bool):
        print(fmt_str.format(value))
    elif isinstance(value, bytes):
        print(fmt_str.format(value.decode()))
    elif isinstance(value, datetime.datetime):
        print(fmt_str.format(value.isoformat()))
    elif value is None:
        print(fmt_str.format("N/A"))
    else:
        print("warn: unhandled type {} found".format(type(value)))
```

我们的下一个元数据集包括有关文档渊源和创建的更多细节。`xmp_creatorTool`属性存储有关用于创建资源的软件的信息。另外，我们还可以根据以下两个 ID 推断出额外的渊源信息：

+   `文档 ID`表示一个标识符，通常存储为 GUID，通常在将资源保存到新文件时分配。例如，如果我们创建`DocA.pdf`，然后将其另存为`DocB.pdf`，我们将有两个不同的`文档 ID`。

+   在`文档 ID`之后是第二个标识符`实例 ID`。`实例 ID`通常在每次保存时生成一次。当我们使用新的段落更新`DocA.pdf`并以相同的文件名保存时，此标识符更新的一个例子。

在编辑相同的 PDF 时，您期望`文档 ID`保持不变，而`实例 ID`可能会更新，尽管这种行为可能会因所使用的软件而异。

```py
custom_print("Created With: {}", xmpm.xmp_creatorTool)
custom_print("Document ID: {}", xmpm.xmpmm_documentId)
custom_print("Instance ID: {}", xmpm.xmpmm_instanceId)
```

随后，我们继续提取其他常见的 XMP 元数据，包括语言、发布者、资源类型和类型。资源类型字段应该表示**多用途互联网邮件扩展**（**MIME**）值，而类型字段应该存储**都柏林核心元数据倡议**（***DCMI**）值。

```py
custom_print("Language: {}", xmpm.dc_language)
custom_print("Publisher: {}", xmpm.dc_publisher)
custom_print("Resource Type: {}", xmpm.dc_format)
custom_print("Type: {}", xmpm.dc_type)
```

最后，我们提取软件保存的任何自定义属性。由于这应该是一个字典，我们可以在不使用我们的`custom_print()`函数的情况下打印它。

```py
if xmpm.custom_properties:
    print("Custom Properties:")
    for k, v in xmpm.custom_properties.items():
        print("\t{}: {}".format(k, v))
```

当我们执行脚本时，我们可以快速看到 PDF 中存储的许多属性。请注意`文档 ID`与`实例 ID`不匹配，这表明该文档可能已经从原始 PDF 进行了修改。

![](img/00037.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了以下一个或多个建议：

+   探索和集成其他与 PDF 相关的库，如`slate`和`pyocr`：

+   `slate`模块，[`github.com/timClicks/slate`](https://github.com/timClicks/slate)，可以从 PDF 文件中提取文本。

+   `pyocr`模块，[`github.com/openpaperwork/pyocr`](https://github.com/openpaperwork/pyocr)，可用于对 PDF 进行 OCR 以捕获手写文本。

# 审查可执行元数据

食谱难度：简单

Python 版本：2.7 或 3.5

操作系统：任何

在调查过程中，我们可能会识别出潜在可疑或未经授权的可移植可执行文件。这个可执行文件可能很有趣，因为它在系统上的使用时间，它在系统上的位置，或者调查特定的其他属性。无论我们是将其作为恶意软件还是未经授权的实用程序进行调查，我们都需要有能力了解更多关于它的信息。

通过从 Windows 可执行文件中提取嵌入的元数据，我们可以了解构成文件的组件。在这个示例中，我们将公开编译日期，来自节头的有用的**威胁指标**（**IOC**）数据，以及导入和导出的符号。

# 开始

这个示例需要安装第三方库`pefile`。此脚本中使用的所有其他库都包含在 Python 的标准库中。`pefile`模块使我们无需指定 Windows 可执行文件的所有结构。`pefile`库可以这样安装：

```py
pip install pefile==2017.8.1
```

要了解更多关于`pefile`库的信息，请访问[`github.com/erocarrera/pefile`](https://github.com/erocarrera/pefile)。

# 如何做...

我们通过以下步骤从可执行文件中提取元数据：

1.  打开可执行文件并使用`pefile`转储元数据。

1.  如果存在，动态打印元数据到控制台。

# 它是如何工作的...

我们首先导入处理参数、解析日期和与可执行文件交互的库。请注意，我们专门从`pefile`中导入`PE`类，这样我们可以在示例后面直接调用`PE`类的属性和方法。

```py
from __future__ import print_function
import argparse
from datetime import datetime
from pefile import PE
```

这个示例的命令行处理程序接受一个位置参数`EXE_FILE`，即我们将从中提取元数据的可执行文件的路径。我们还将接受一个可选参数`v`，以允许用户决定他们是否希望获得详细或简化的输出。

```py
parser = argparse.ArgumentParser(
    description=__description__,
    epilog="Developed by {} on {}".format(
        ", ".join(__authors__), __date__)
)
parser.add_argument("EXE_FILE", help="Path to exe file")
parser.add_argument("-v", "--verbose", help="Increase verbosity of output",
                    action='store_true', default=False)
args = parser.parse_args()
```

使用`PE`类，我们通过提供文件路径来简单地加载输入可执行文件。使用`dump_dict()`方法，我们将可执行文件数据转储到字典对象中。这个库允许我们通过这个`ped`字典或作为`pe`对象的属性来探索键值对。我们将演示如何使用这两种技术提取嵌入的元数据。

```py
pe = PE(args.EXE_FILE)
ped = pe.dump_dict()
```

让我们从提取基本文件元数据开始，比如嵌入的作者、版本和编译时间。这些元数据存储在`FileInfo`对象的`StringTable`中。使用`for`循环和`if`语句，我们确保提取正确的值，并将字符串`"Unknown"`赋给值为`None`或长度为零的值，以更好地适应将这些数据打印到控制台。提取并打印所有键值对到控制台后，我们继续处理可执行文件的嵌入编译时间，该时间存储在其他地方。

```py
file_info = {}
for structure in pe.FileInfo:
    if structure.Key == b'StringFileInfo':
        for s_table in structure.StringTable:
            for key, value in s_table.entries.items():
                if value is None or len(value) == 0:
                    value = "Unknown"
                file_info[key] = value
print("File Information: ")
print("==================")
for k, v in file_info.items():
    if isinstance(k, bytes):
        k = k.decode()
    if isinstance(v, bytes):
        v = v.decode()
    print("{}: {}".format(k, v))
```

编译时间戳存储在文件中，显示可执行文件的编译日期。`pefile`库为我们解释原始数据，而`Value`键存储原始十六进制值和方括号内解释的日期。我们可以自己解释十六进制值，或者更简单地将解析后的日期字符串转换为`datetime`对象。

我们使用`split()`和`strip()`方法从方括号中提取解析后的日期字符串。在转换之前，必须将缩写的时区（例如 UTC、EST 或 PST）与解析后的日期字符串分开。一旦日期字符串被隔离，我们使用`datetime.strptime()`方法和`datetime`格式化程序来正确转换和打印可执行文件的嵌入编译日期。

```py
# Compile time 
comp_time = ped['FILE_HEADER']['TimeDateStamp']['Value']
comp_time = comp_time.split("[")[-1].strip("]")
time_stamp, timezone = comp_time.rsplit(" ", 1)
comp_time = datetime.strptime(time_stamp, "%a %b %d %H:%M:%S %Y")
print("Compiled on {} {}".format(comp_time, timezone.strip()))
```

我们提取的下一个元素是关于可执行文件部分的元数据。这一次，我们不是使用`pe`对象及其属性，而是使用我们创建的字典对象`ped`来遍历部分并显示部分名称、地址、大小和其内容的`MD5`哈希。这些数据可以添加到您的 IOC 中，以帮助识别环境中此主机和其他主机上的其他恶意文件。

```py
# Extract IOCs from PE Sections 
print("\nSections: ")
print("==========")
for section in ped['PE Sections']:
    print("Section '{}' at {}: {}/{} {}".format(
        section['Name']['Value'], hex(section['VirtualAddress']['Value']),
        section['Misc_VirtualSize']['Value'],
        section['SizeOfRawData']['Value'], section['MD5'])
    )
```

可执行文件中的另一组元数据是其导入和导出的列表。让我们从导入条目开始。首先，我们确保在尝试访问`pe`变量的这个属性之前，该属性存在。如果存在，我们使用两个`for`循环来遍历导入的 DLL，并且如果用户指定了详细输出，遍历 DLL 中的每个导入。如果用户没有指定详细输出，则跳过最内层循环，只向控制台呈现 DLL 名称。从这些循环中，我们提取 DLL 名称、地址和导入名称。我们可以使用`getattr()`内置函数来确保在属性不存在的情况下不会收到任何错误。

```py
if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
    print("\nImports: ")
    print("=========")
    for dir_entry in pe.DIRECTORY_ENTRY_IMPORT:
        dll = dir_entry.dll
        if not args.verbose:
            print(dll.decode(), end=", ")
            continue

        name_list = []
        for impts in dir_entry.imports:
            if getattr(impts, "name", b"Unknown") is None:
                name = b"Unknown"
            else:
                name = getattr(impts, "name", b"Unknown")
            name_list.append([name.decode(), hex(impts.address)])
        name_fmt = ["{} ({})".format(x[0], x[1]) for x in name_list]
        print('- {}: {}'.format(dll.decode(), ", ".join(name_fmt)))
    if not args.verbose:
        print()
```

最后，让我们回顾与导出元数据相关的代码块。因为一些可执行文件可能没有导出，我们使用`hasattr()`函数来确认`DIRECTORY_ENTRY_EXPORT`属性是否存在。如果存在，我们遍历每个符号，并在控制台中以项目符号列表的形式打印每个符号的名称和地址，以更好地区分它们。

```py
# Display Exports, Names, and Addresses 
if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
    print("\nExports: ")
    print("=========")
    for sym in pe.DIRECTORY_ENTRY_EXPORT.symbols:
        print('- {}: {}'.format(sym.name.decode(), hex(sym.address)))
```

以 Firefox 安装程序为例，我们能够从可执行文件中提取大量嵌入的元数据属性。这些信息向我们展示了许多东西，比如编译日期；这似乎是一个打包的可执行文件，可能是用 7-Zip 打包的；以及不同部分的哈希值。

![](img/00038.jpeg)

当我们对 DLL 运行相同的脚本时，我们看到与可执行文件运行中的许多相同字段，另外还有导出部分。由于输出的长度，我们省略了以下截图中的一些文本：

![](img/00039.jpeg)

# 还有更多...

这个脚本可以进一步改进。我们提供了一个或多个建议如下：

+   使用我们在第五章中开发的配方，*网络和威胁指标配方*，查询发现的哈希值与 VirusTotal 等在线资源，并报告其他提交的任何匹配项。

+   集成`pytz`以允许用户在本地或其他指定的时区解释日期

# 读取办公文件元数据

配方难度：中等

Python 版本：2.7 或 3.5

操作系统：任何

从办公文件中读取元数据可以暴露有关这些文件的作者和历史的有趣信息。方便的是，2007 格式的`.docx`、`.xlsx`和`.pptx`文件将元数据存储在 XML 中。XML 标记可以很容易地用 Python 处理。

# 入门

此脚本中使用的所有库都包含在 Python 的标准库中。我们使用内置的`xml`库和`zipfile`库来允许我们访问 ZIP 容器中的 XML 文档。

要了解有关`xml`库的更多信息，请访问[`docs.python.org/3/library/xml.etree.elementtree.html`](https://docs.python.org/3/library/xml.etree.elementtree.html)。

要了解有关`zipfile`库的更多信息，请访问[`docs.python.org/3/library/zipfile.html`](https://docs.python.org/3/library/zipfile.html)[.](https://docs.python.org/3/library/xml.etree.elementtree.html)

# 如何做...

我们通过执行以下步骤提取嵌入的 Office 元数据：

1.  确认输入文件是有效的 ZIP 文件。

1.  从 Office 文件中提取`core.xml`和`app.xml`文件。

1.  解析 XML 数据并打印嵌入的元数据。

# 它是如何工作的...

首先，我们导入`argparse`和`datetime`库，然后是`xml.etree`和`zipfile`库。`ElementTree`类允许我们将 XML 字符串读入一个对象，我们可以通过它进行迭代和解释。

```py
from __future__ import print_function
from argparse import ArgumentParser
from datetime import datetime as dt
from xml.etree import ElementTree as etree
import zipfile
```

这个配方的命令行处理程序接受一个位置参数`Office_File`，即我们将从中提取元数据的办公文件的路径。

```py
parser = argparse.ArgumentParser(
    description=__description__,
    epilog="Developed by {} on {}".format(", ".join(__authors__), __date__)
)
parser.add_argument("Office_File", help="Path to office file to read")
args = parser.parse_args()
```

在处理参数后，我们检查输入文件是否是一个`zipfile`，如果不是，则引发错误。如果是，我们使用`ZipFile`类打开有效的 ZIP 文件，然后访问包含我们感兴趣的元数据的两个 XML 文档。虽然还有其他包含描述文档数据的 XML 文件，但包含最多元数据的是名为`core.xml`和`app.xml`的两个文件。我们将使用`read()`方法从 ZIP 容器中打开这两个 XML 文件，并将返回的字符串直接发送到`etree.fromstring()` XML 解析方法。

```py
# Check if input file is a zipfile
zipfile.is_zipfile(args.Office_File)

# Open the file (MS Office 2007 or later)
zfile = zipfile.ZipFile(args.Office_File)

# Extract key elements for processing
core_xml = etree.fromstring(zfile.read('docProps/core.xml'))
app_xml = etree.fromstring(zfile.read('docProps/app.xml'))
```

有了准备好的 XML 对象，我们可以开始提取感兴趣的数据。我们设置了一个名为`core_mapping`的字典，用于指定我们想要提取的字段作为键名，以及我们想要将它们显示为的值。这种方法使我们能够轻松地打印出对我们重要的值，如果存在的话，还可以使用友好的标题。这个 XML 文件包含了关于文件作者的重要信息。例如，两个作者字段`creator`和`lastModifiedBy`可以显示一个账户修改了另一个用户账户创建的文档的情况。日期值向我们展示了关于文档的创建和修改的信息。此外，像`revision`这样的元数据字段可以给出有关此文档版本数量的一些指示。

```py
# Core.xml tag mapping 
core_mapping = {
    'title': 'Title',
    'subject': 'Subject',
    'creator': 'Author(s)',
    'keywords': 'Keywords',
    'description': 'Description',
    'lastModifiedBy': 'Last Modified By',
    'modified': 'Modified Date',
    'created': 'Created Date',
    'category': 'Category',
    'contentStatus': 'Status',
    'revision': 'Revision'
}
```

在我们的`for`循环中，我们使用`iterchildren()`方法迭代 XML，以访问`core.xml`文件的 XML 根中的每个标签。使用`core_mapping`字典，我们可以有选择地输出特定字段（如果找到的话）。我们还添加了用`strptime()`方法解释日期值的逻辑。

```py
for element in core_xml.getchildren():
    for key, title in core_mapping.items():
        if key in element.tag:
            if 'date' in title.lower():
                text = dt.strptime(element.text, "%Y-%m-%dT%H:%M:%SZ")
            else:
                text = element.text
            print("{}: {}".format(title, text))
```

接下来的列映射集中在`app.xml`文件上。这个文件包含有关文档内容的统计信息，包括总编辑时间和单词、页数和幻灯片的计数。它还包含有关注册在软件中的公司名称和隐藏元素的信息。为了将这些值打印到控制台上，我们使用了与`core.xml`文件相似的一组`for`循环。

```py
app_mapping = {
    'TotalTime': 'Edit Time (minutes)',
    'Pages': 'Page Count',
    'Words': 'Word Count',
    'Characters': 'Character Count',
    'Lines': 'Line Count',
    'Paragraphs': 'Paragraph Count',
    'Company': 'Company',
    'HyperlinkBase': 'Hyperlink Base',
    'Slides': 'Slide count',
    'Notes': 'Note Count',
    'HiddenSlides': 'Hidden Slide Count',
}
for element in app_xml.getchildren():
    for key, title in app_mapping.items():
        if key in element.tag:
            if 'date' in title.lower():
                text = dt.strptime(element.text, "%Y-%m-%dT%H:%M:%SZ")
            else:
                text = element.text
            print("{}: {}".format(title, text))
```

当我们运行脚本并使用示例 Word 文档时，如下所示，关于文档的许多细节都受到质疑。

![](img/00040.jpeg)

另外，我们可以在 PPTX 文档上使用脚本并审查与 PPTX 文件相关的特定格式的元数据：

![](img/00041.jpeg)

# 将我们的元数据提取器与 EnCase 集成

配方难度：中等

Python 版本：2.7 或 3.5

操作系统：Windows

我们设计的嵌入式元数据提取配方适用于松散文件，而不适用于取证图像中的文件。令人恼火的是，这在我们的流程中增加了一个额外的步骤，需要我们从图像中导出感兴趣的文件进行此类审查。在这个配方中，我们展示了如何将我们的脚本连接到取证工具 EnCase，并在无需从取证图像中导出文件的情况下执行它们。

# 入门

安装了 EnCase 后，我们需要创建一个案件并添加证据文件，就像对待其他案件一样。这个配方演示了在 EnCase V6 中执行此操作所需的步骤，尽管相同的技术也可以应用于后续版本。

在开始之前，我们还需要确保机器上安装了 Python 2.7 或 3.5、我们希望使用的脚本以及所需的依赖项。

# 操作步骤...

我们通过以下步骤将元数据配方与 EnCase 集成：

1.  打开 EnCase V6 并将证据添加到案件中。

1.  使用“查看文件查看器”菜单配置自定义文件查看器，其中包括`EXIF`元数据提取器。

1.  使用新创建的文件查看器在 EnCase 中提取嵌入的照片 GPS 坐标。

# 它是如何工作的...

打开案例后，我们可以查看感兴趣的照片的十六进制，以确认我们可以在文件中看到`EXIF`头。在这个头部之后是脚本处理的原始值。确定了一个好的候选者后，让我们看看如何配置 EnCase 来运行脚本。

![](img/00042.jpeg)

在`查看`菜单下，我们选择`文件查看器`选项。这将打开一个列出可用查看器的选项卡。我们使用的 EnCase 实例没有任何查看器，所以我们必须首先添加任何我们希望使用的查看器。

![](img/00043.gif)

在这个选项卡上，右键单击顶层的`文件查看器`元素，然后选择`新建...`来创建我们的自定义查看器。

![](img/00044.jpeg)

一个新窗口，如下面的屏幕截图所示，允许我们指定执行脚本的参数。在这个例子中，我们正在实现 GPS 提取脚本，尽管我们可以以同样的方式添加其他脚本。第一行指定了查看器的名称。我们应该给它取一个容易记住的名字，因为在以后选择文件查看器时，这将是我们唯一可用的描述。第二行是可执行文件的路径。在我们的实例中，我们将启动命令提示符，因为我们的 Python 脚本不是一个独立的可执行文件。我们需要为 EnCase 提供`cmd.exe`的完整路径，以便接受这个参数。

最后一行是我们添加脚本的地方。这一行允许我们指定要传递给命令提示符的参数。我们从`/k`开始，以便在脚本完成后保持我们的命令提示符打开。这不是必需的；尽管如果您的代码在控制台上显示信息（就像我们的代码一样），我们应该实现这个功能。否则，命令提示符将在代码完成后立即关闭。在`/k`参数之后，我们提供启动代码的参数。如图所示，这包括 Python 可执行文件和脚本的完整路径。最后一个元素`[file]`是 EnCase 的占位符，在文件查看器执行时被我们要查看的文件替换。

![](img/00045.gif)

新的文件查看器条目现在显示在`文件查看器`选项卡中，并显示了我们指定的名称、可执行文件和参数。如果一切看起来正确，我们可以返回到文件条目选项卡中感兴趣的照片。

![](img/00046.jpeg)

回到文件条目视图，我们可以右键单击感兴趣的照片，然后从`发送到`子菜单中选择文件查看器。

![](img/00047.jpeg)

当我们选择这个选项时，命令窗口会出现，并显示脚本的输出。请注意，KML 文件会自动放置在案例的`Temp`目录中。这是因为我们正在检查的文件在脚本执行期间被缓存在这个目录中。

![](img/00048.jpeg)

# 还有更多...

这个过程可以进一步改进。我们提供了一个或多个建议如下：

+   虽然与 Python 无关，但可以考虑 EnScripting 作为自动化和解析多个文件并在 EnCase 控制台选项卡中显示输出的另一个选项。

+   通过类似的方法将本章中涵盖的其他脚本添加到 EnCase 中。由于这些脚本的信息被打印到控制台，我们应该使用`/k`参数，或者重新设计逻辑将输出放在一个目录中。
