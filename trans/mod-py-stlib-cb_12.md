# 多媒体

在本章中，我们将涵盖以下配方：

+   确定文件类型——如何猜测文件类型

+   检测图像类型——检查图像以了解其类型

+   检测图像大小——检查图像以检索其大小

+   播放音频/视频/图像——在桌面系统上播放音频、视频或显示图像

# 介绍

多媒体应用程序，如视频、声音和游戏通常需要依赖非常特定的库来管理用于存储数据和播放内容所需的硬件。

由于数据存储格式的多样性，视频和音频存储领域的不断改进导致新格式的出现，以及与本地操作系统功能和特定硬件编程语言的深度集成，多媒体相关功能很少集成在标准库中。

当每隔几个月就会创建一个新的图像格式时，需要维护对所有已知图像格式的支持，这需要全职的工作，而专门的库可以比维护编程语言本身的团队更好地处理这个问题。

因此，Python 几乎没有与多媒体相关的函数，但一些核心函数是可用的，它们可以在多媒体不是主要关注点的应用程序中非常有帮助，但也许它们需要处理多媒体文件以正确工作；例如，一个可能需要检查用户上传的文件是否是浏览器支持的有效格式的 Web 应用程序。

# 确定文件类型

当我们从用户那里收到文件时，通常需要检测其类型。通过文件名而无需实际读取数据就可以实现这一点，这可以通过`mimetypes`模块来实现。

# 如何做...

对于这个配方，需要执行以下步骤：

1.  虽然`mimetypes`模块并不是绝对可靠的，因为它依赖于文件名来检测预期的类型，但它通常足以处理大多数常见情况。

1.  用户通常会为了自己的利益（特别是 Windows 用户，其中扩展名对文件的正确工作至关重要）为其文件分配适当的名称，使用`mimetypes.guess_type`猜测类型通常就足够了：

```py
import mimetypes

def guess_file_type(filename):
    if not getattr(guess_file_type, 'initialised', False):
        mimetypes.init()
        guess_file_type.initialised = True
    file_type, encoding = mimetypes.guess_type(filename)
    return file_type
```

1.  我们可以对任何文件调用`guess_file_type`来获取其类型：

```py
>>> print(guess_file_type('~/Pictures/5565_1680x1050.jpg'))
'image/jpeg'
>>> print(guess_file_type('~/Pictures/5565_1680x1050.jpeg'))
'image/jpeg'
>>> print(guess_file_type('~/Pictures/avatar.png'))
'image/png' 
```

1.  如果类型未知，则返回`None`：

```py
>>> print(guess_file_type('/tmp/unable_to_guess.blob'))
None
```

1.  另外，请注意文件本身并不一定真的存在。您关心的只是它的文件名：

```py
>>> print(guess_file_type('/this/does/not/exists.txt'))
'text/plain'
```

# 它是如何工作的...

`mimetypes`模块保留了与每个文件扩展名关联的 MIME 类型列表。

提供文件名时，只分析扩展名。

如果扩展名在已知 MIME 类型列表中，则返回关联的类型。否则返回`None`。

调用`mimetypes.init()`还会加载系统配置中注册的任何 MIME 类型，通常是从 Linux 系统的`/etc/mime.types`和 Windows 系统的注册表中加载。

这使我们能够涵盖更多可能不为 Python 所知的扩展名，并且还可以轻松支持自定义扩展名，如果您的系统配置支持它们的话。

# 检测图像类型

当您知道正在处理图像文件时，通常需要验证它们的类型，以确保它们是您的软件能够处理的格式。

一个可能的用例是确保它们是浏览器可能能够在网站上上传时显示的格式的图像。

通常可以通过检查文件头部来检测多媒体文件的类型，文件头部是文件的初始部分，存储有关文件内容的详细信息。

标头通常包含有关文件类型、包含图像的大小、每种颜色的位数等的详细信息。所有这些细节都是重现文件内存储的内容所必需的。

通过检查头部，可以确认存储数据的格式。这需要支持特定的头部格式，Python 标准库支持大多数常见的图像格式。

# 如何做...

`imghdr` 模块可以帮助我们了解我们面对的是什么类型的图像文件：

```py
import imghdr

def detect_image_format(filename):
    return imghdr.what(filename)
```

这使我们能够检测磁盘上任何图像的格式或提供的字节流的格式：

```py
>>> print(detect_image_format('~/Pictures/avatar.jpg'))
'jpeg'
>>> with open('~/Pictures/avatar.png', 'rb') as f:
...     print(detect_image_format(f))
'png'
```

# 它是如何工作的...

当提供的文件名是包含文件路径的字符串时，直接在其上调用 `imghdr.what`。

这只是返回文件的类型，如果不支持则返回 `None`。

相反，如果提供了类似文件的对象（例如文件本身或 `io.BytesIO`），则它将查看其前 32 个字节并根据这些字节检测头部。

鉴于大多数图像类型的头部大小在 10 多个字节左右，读取 32 个字节可以确保我们应该有足够的内容来检测任何图像。

读取字节后，它将返回到文件的开头，以便任何后续调用仍能读取文件（否则，前 32 个字节将被消耗并永远丢失）。

# 还有更多...

Python 标准库还提供了一个 `sndhdr` 模块，它的行为很像音频文件的 `imghdr`。

`sndhdr` 识别的格式通常是非常基本的格式，因此当涉及到 `wave` 或 `aiff` 文件时，它通常是非常有帮助的。

# 检测图像大小

如果我们知道我们面对的是什么类型的图像，检测分辨率通常只是从图像头部读取它。

对于大多数图像类型，这相对简单，因为我们可以使用 `imghdr` 来猜测正确的图像类型，然后根据检测到的类型读取头部的正确部分，以提取大小部分。

# 如何做...

一旦 `imghdr` 检测到图像类型，我们就可以使用 `struct` 模块读取头部的内容：

```py
import imghdr
import struct
import os
from pathlib import Path

class ImageReader:
    @classmethod
    def get_size(cls, f):    
        requires_close = False
        if isinstance(f, (str, getattr(os, 'PathLike', str))):
            f = open(f, 'rb')
            requires_close = True
        elif isinstance(f, Path):
            f = f.expanduser().open('rb')
            requires_close = True

        try:
            image_type = imghdr.what(f)
            if image_type not in ('jpeg', 'png', 'gif'):
                raise ValueError('Unsupported image format')

            f.seek(0)
            size_reader = getattr(cls, '_size_{}'.format(image_type))
            return size_reader(f)
        finally:
            if requires_close: f.close()

    @classmethod
    def _size_gif(cls, f):
        f.read(6)  # Skip the Magick Numbers
        w, h = struct.unpack('<HH', f.read(4))
        return w, h

    @classmethod
    def _size_png(cls, f):
        f.read(8)  # Skip Magic Number
        clen, ctype = struct.unpack('>I4s', f.read(8))
        if ctype != b'IHDR':
            raise ValueError('Unsupported PNG format')
        w, h = struct.unpack('>II', f.read(8))
        return w, h

    @classmethod
    def _size_jpeg(cls, f):
        start_of_image = f.read(2)
        if start_of_image != b'\xff\xd8':
            raise ValueError('Unsupported JPEG format')
        while True:
            marker, segment_size = struct.unpack('>2sH', f.read(4))
            if marker[0] != 0xff:
                raise ValueError('Unsupported JPEG format')
            data = f.read(segment_size - 2)
            if not 0xc0 <= marker[1] <= 0xcf:
                continue
            _, h, w = struct.unpack('>cHH', data[:5])
            break
        return w, h
```

然后我们可以使用 `ImageReader.get_size` 类方法来检测任何支持的图像的大小：

```py
>>> print(ImageReader.get_size('~/Pictures/avatar.png'))
(300, 300)
>>> print(ImageReader.get_size('~/Pictures/avatar.jpg'))
(300, 300)
```

# 它是如何工作的...

`ImageReader` 类的四个核心部分共同工作，以提供对读取图像大小的支持。

首先，`ImageReader.get_size` 方法本身负责打开图像文件并检测图像类型。

第一部分与打开文件有关，如果它以字符串形式提供为路径，作为 `Path` 对象，或者如果它已经是文件对象：

```py
requires_close = False
if isinstance(f, (str, getattr(os, 'PathLike', str))):
    f = open(f, 'rb')
    requires_close = True
elif isinstance(f, Path):
    f = f.expanduser().open('rb')
    requires_close = True
```

如果它是一个字符串或路径对象（`os.PathLike` 仅支持 Python 3.6+），则打开文件并将 `requires_close` 变量设置为 `True`，这样一旦完成，我们将关闭文件。

如果它是一个 `Path` 对象，并且我们使用的 Python 版本不支持 `os.PathLike`，那么文件将通过路径本身打开。

如果提供的对象已经是一个打开的文件，则我们什么也不做，`requires_close` 保持 `False`，这样我们就不会关闭提供的文件。

一旦文件被打开，它被传递给 `imghdr.what` 来猜测文件类型，如果它不是受支持的类型之一，它就会被拒绝：

```py
image_type = imghdr.what(f)
if image_type not in ('jpeg', 'png', 'gif'):
    raise ValueError('Unsupported image format')
```

最后，我们回到文件的开头，这样我们就可以读取头部，并调用相关的 `cls._size_png`、`cls._size_jpeg` 或 `cls._size_gif` 方法：

```py
f.seek(0)
size_reader = getattr(cls, '_size_{}'.format(image_type))
return size_reader(f)
```

每种方法都专门用于了解特定文件格式的大小，从最简单的（GIF）到最复杂的（JPEG）。

对于 GIF 本身，我们所要做的就是跳过魔术数字（只有 `imghdr.what` 关心；我们已经知道它是 GIF），并将随后的四个字节读取为无符号短整数（16 位数字），采用小端字节顺序：

```py
@classmethod
def _size_gif(cls, f):
    f.read(6)  # Skip the Magick Numbers
    w, h = struct.unpack('<HH', f.read(4))
    return w, h
```

`png` 几乎和 GIF 一样复杂。我们跳过魔术数字，并将随后的字节作为大端顺序的 `unsigned int`（32 位数字）读取，然后是四字节字符串：

```py
@classmethod
def _size_png(cls, f):
    f.read(8)  # Skip Magic Number
    clen, ctype = struct.unpack('>I4s', f.read(8))
```

这给我们返回了图像头部的大小，后面跟着图像部分的名称，必须是 `IHDR`，以确认我们正在读取图像头部：

```py
if ctype != b'IHDR':
    raise ValueError('Unsupported PNG format')
```

一旦我们知道我们在图像头部内，我们只需读取前两个`unsigned int`数字（仍然是大端）来提取图像的宽度和高度：

```py
w, h = struct.unpack('>II', f.read(8))
return w, h
```

最后一种方法是最复杂的，因为 JPEG 的结构比 GIF 或 PNG 复杂得多。JPEG 头由多个部分组成。每个部分由`0xff`标识，后跟部分标识符和部分长度。

一开始，我们只读取前两个字节并确认我们面对**图像的开始**（**SOI**）部分：

```py
@classmethod
def _size_jpeg(cls, f):
    start_of_image = f.read(2)
    if start_of_image != b'\xff\xd8':
        raise ValueError('Unsupported JPEG format')
```

然后我们寻找一个声明 JPEG 为基线 DCT、渐进 DCT 或无损帧的部分。

这是通过读取每个部分的前两个字节及其大小来完成的：

```py
while True:
    marker, segment_size = struct.unpack('>2sH', f.read(4))
```

由于我们知道每个部分都以`0xff`开头，如果我们遇到以不同字节开头的部分，这意味着图像无效：

```py
if marker[0] != 0xff:
    raise ValueError('Unsupported JPEG format')
```

如果部分有效，我们可以读取它的内容。我们知道大小，因为它是在两个字节的无符号短整数中以大端记法指定的：

```py
data = f.read(segment_size - 2)
```

现在，在能够从我们刚刚读取的数据中读取宽度和高度之前，我们需要检查我们正在查看的部分是否实际上是基线、渐进或无损的帧的开始。这意味着它必须是从`0xc0`到`0xcf`的部分之一。

否则，我们只是跳过这个部分并移动到下一个：

```py
if not 0xc0 <= marker[1] <= 0xcf:
    continue
```

一旦我们找到一个有效的部分（取决于图像的编码方式），我们可以通过查看前五个字节来读取大小。

第一个字节是样本精度。我们真的不关心它，所以我们可以忽略它。然后，剩下的四个字节是图像的高度和宽度，以大端记法的两个无符号短整数：

```py
_, h, w = struct.unpack('>cHH', data[:5])
```

# 播放音频/视频/图像

Python 标准库没有提供打开图像的实用程序，并且对播放音频文件的支持有限。

虽然可以通过结合`wave`和`ossaudiodev`或`winsound`模块以某种格式在一些格式中播放音频文件，但是 OSS 音频系统在 Linux 系统上已经被弃用，而且这两者都不适用于 Mac 系统。

对于图像，可以使用`tkinter`模块显示图像，但我们将受到非常简单的图像格式的限制，因为解码图像将由我们自己完成。

但是有一个小技巧，我们可以用来实际显示大多数图像文件和播放大多数音频文件。

在大多数系统上，尝试使用默认的网络浏览器打开文件将播放文件，我们可以依靠这个技巧和`webbrowser`模块通过 Python 播放大多数文件类型。

# 如何做...

此食谱的步骤如下：

1.  给定一个指向支持的文件的路径，我们可以构建一个`file:// url`，然后使用`webbrowser`模块打开它：

```py
import pathlib
import webbrowser

def playfile(fpath):
    fpath = pathlib.Path(fpath).expanduser().resolve()
    webbrowser.open('file://{}'.format(fpath))
```

1.  打开图像应该会显示它：

```py
>>> playfile('~/Pictures/avatar.jpg')
```

1.  此外，打开音频文件应该会播放它：

```py
>>> playfile('~/Music/FLY_ME_TO_THE_MOON.mp3')
```

因此，我们可以在大多数系统上使用这种方法来向用户显示文件的内容。

# 它是如何工作的...

`webbrowser.open`函数实际上在 Linux 系统上启动浏览器，但在 macOS 和 Windows 系统上，它的工作方式有所不同。

在 Windows 和 macOS 系统上，它将要求系统使用最合适的应用程序打开指定的路径。

如果路径是 HTTP URL，则最合适的应用程序当然是`webbrowser`，但如果路径是本地`file://` URL，则系统将寻找能够处理该文件类型并将文件打开的软件。

这是通过在 Windows 系统上使用`os.startfile`，并通过`osascript`命令在 macOS 上运行一个小的 Apple 脚本片段来实现的。

这使我们能够打开图像和音频文件，由于大多数图像和音频文件格式也受到浏览器支持，因此它也可以在 Linux 系统上运行。
