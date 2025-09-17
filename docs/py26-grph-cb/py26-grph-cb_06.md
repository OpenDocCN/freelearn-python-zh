# 第六章. 处理图片

在本章中，我们将涵盖：

+   原生 Python 中的图片格式

+   打开图像并发现其属性

+   Python 图片库格式转换：`.jpg, .png, .tiff, .gif`，以及 `.bmp`

+   平面内的图像旋转

+   调整图片大小

+   按正确宽高比调整大小

+   旋转图片

+   分离颜色带

+   红色、绿色和蓝色颜色重新混合

+   通过混合组合图片

+   通过调整百分比混合图片

+   使用图像蒙版制作合成图

+   水平和垂直偏移（滚动）图片

+   几何变换：水平和垂直翻转和旋转

+   过滤器：锐化、模糊、边缘增强、浮雕、平滑、轮廓和细节

+   通过调整大小实现的明显旋转

现在我们将处理栅格图像。这些包括照片、位图图像和数字绘画等所有图像类型，它们都不是我们至今一直在使用的矢量图形绘制。栅格图像由像素组成，这是图片元素的简称。矢量图像是定义为数学形状和颜色表达式，可以在您的直接控制下通过代数和算术进行修改。这些矢量图形只是计算机图形世界的一部分。

另一部分涉及照片和绘制的位图图像的表示和处理，通常被称为栅格图像。Python 识别的唯一栅格图像类型是 **GIF**（**图形交换格式**）图像，它具有有限的颜色能力；`GIF` 可以处理 256 种不同的颜色，而 `.png` 或 `.jpg` 则有 1670 万种。优点是 `GIF` 图像在 Python 中的控制允许您使它们动画化，但基本的 Tkinter 提供的库中没有可以操作和改变栅格图像的函数。

然而，有一个非常有用的 Python 模块集合，即 **Python Imaging Library**（**PIL**），它是专门为栅格图像操作设计的。它具有大多数优秀的照片编辑工具所拥有的基本功能。PIL 模块可以轻松地将一种格式转换为另一种格式，包括 `GIF, PNG, TIFF, JPEG, BMP`，并且 `PIL` 还可以与许多其他格式一起工作，但前面提到的可能是最常见的。Python 图片库是您一般图形工具包和技能库的重要组成部分。

为了减少混淆，我们将使用文件扩展名缩写，如 `.gif, .png, .jpg` 等作为 `GIF, PNG` 和 `JPEG` 等文件格式的名称。

# 打开图像文件并发现其属性

首先，我们需要测试 PIL 是否已加载到包含我们其他 Python 模块的库中。最简单的方法是尝试使用 **Image** 模块的 `image_open()` 函数打开一个文件。

## 准备工作

如果 Python Imaging Library (PIL) 还未安装在我们的文件系统中，并且对 Python 可用，我们需要找到并安装它。Tkinter 不需要用于光栅图像处理。你会注意到没有 `from Tkinter import *` 和没有 `root = tK()` 或 `root.mainloop()` 语句。

你可以从 [`www.pythonware.com/products/pil/`](http://www.pythonware.com/products/pil/) 下载 PIL。

这个网站包含源代码、MS Windows 安装可执行文件和 HTML 或 `PDF` 格式的手册。

### 注意

关于 PIL 的最佳解释文档之一是位于新墨西哥技术计算机中心的一个 `PDF` 文件 [` infohost.nmt.edu/tcc/help/pubs/pil.pdf`](http://%20infohost.nmt.edu/tcc/help/pubs/pil.pdf)。它清晰简洁。

在本章接下来的所有示例中，所有保存到我们硬盘上的图像都被放置在 `constr` 文件夹内的 `picsx` 文件夹中。这是为了将结果与包含所有将要用到的输入图像的 `pics1` 文件夹分开。这使我们免于决定保留什么和丢弃什么的困境。你应该保留 `pics1` 中的所有内容，并且可以丢弃 `picsx` 中的任何内容，因为重新运行创建这些文件的程序应该很简单。

## 如何做到...

将以下代码复制到编辑器中，并保存为 `image_getattributes_1.py`，然后像所有之前的程序一样执行。在这个程序中，我们将使用 PIL 来发现 `JPG` 格式图像的属性。请记住，尽管 Python 本身只能识别 `GIF` 图像，但 PIL 模块可以处理许多图像格式。在我们能让这个小程序运行之前，我们无法进一步使用 PIL。

```py
# image_getattributes_1.py
# >>>>>>>>>>>>>>>>>>
import Image
imageFile = "/constr/pics1/canary_a.jpg"
im_1 = Image.open(imageFile)
im_width = im_1.size[0]
im_height = im_1.size[1]
im_mode = im_1.mode
im_format = im_1.format
print "Size: ",im_width, im_height
print "Mode: ",im_mode
print "Format: ",im_format
im_1.show()

```

## 它是如何工作的...

PIL 库的一部分 **Image** 模块有一个 `Image_open()` 方法，它可以打开被识别为图像文件的文件。它不会显示图像。这可能会让人困惑。打开一个文件意味着我们的应用程序已经找到了文件的位置，并处理了加载文件所需的所有权限和管理。当你打开一个文件时，会读取文件头以确定文件格式并提取解码文件所需的东西，如模式、大小和其他属性，但文件的其他部分将在稍后处理。模式是一个术语，用来指代包含图像的数据字节应该如何被解释，比如一个特定的字节是否指的是红色通道或透明度通道等等。只有当 **Image** 模块接收到查看文件、更改其大小、查看某个颜色通道、旋转它或 PIL 模块可以对图像文件执行的数十种操作之一的命令时，它才会从硬盘驱动器实际加载到内存中。

如果我们想查看图像，那么我们使用 `im_1\. show()` 方法。只需在末尾添加一行 `im.show()` 即可。

为什么我们需要获取图像属性？当我们准备更改和操作图像时，我们需要更改属性，因此我们通常需要能够找出它们的原始属性。

## 还有更多...

PIL（Python Imaging Library）的 **Image** 模块可以读取和写入（打开和保存）常见的图像格式。以下格式既可以读取也可以写入：`BMP, GIF, IM, JPG, JPEG, JPE, PCX, PNG, PBM, PPN, TIF, TIFF, XBM, XPM`。

以下文件格式只能读取：`PCD, DCX, PSD`。如果我们需要存储 `PCD, DCX` 或 `PSD` 格式的图像文件，那么我们首先需要将它们转换为像 `PNG, TIFF, JPEG` 或 `BMP` 这样的有效文件格式。Python 本身（没有 PIL 模块）只处理 `GIF` 文件，因此这些将是自包含应用程序的首选文件格式。`JPG` 文件非常普遍，因此我们需要证明我们编写的代码可以使用 `JPG, GIF, PNG` 和 `BMP` 格式。

### 我们需要了解的关于图像格式的知识

了解以下关于文件图像格式的内容是有用的：

+   `GIF` 图像文件是最小且使用和传输速度最快的，它们可能是图像质量和文件大小之间的最佳平衡。缺点是它们颜色范围有限，不适合高质量图片。

+   `JPEG` 图像是网络上最常见的。质量可以从高到低变化，这取决于你指定的压缩程度。大图像可以被大量压缩，但你会失去图像质量。

+   `TIFF` 图像体积大，质量/分辨率高。详细的工程图纸通常以 `TIFF` 文件存档。

+   `PNG` 图像是 `GIF` 文件的现代高质量替代品。但 Tkinter 无法识别它们。

+   `BMP` 图像是不压缩的，有点过时，但仍然有很多。不推荐使用。

    当在 PIL 中处理图像时，PNG 图像是一种方便使用的格式。然而，如果你正在为在多种平台上显示的 Python 程序中的图像做准备，那么在保存之前你需要将它们转换为 GIF 格式。

### 图像与数字游戏

图像格式就像学习古代语言一样，学得越多，事情就越复杂。但这里有一些基本的数字规则，可以给你一些洞察。

+   `GIF` 格式最多支持 256 种颜色，但可以使用更少的颜色。

+   `PNG` 格式最多支持约 14000 种不同的颜色。

+   `JPEG` 可以处理 1600 万种颜色，与上一章中使用的 `#rrggbb` 数字相同数量的颜色。

    大多数数码相机的图像都压缩成 JPG 格式，这会减少颜色的范围。因此，在大多数情况下，我们可以将它们转换为 PNG 图像而不会出现明显的质量损失。

# 以不同的文件格式打开、查看和保存图像

很常见的情况是我们想处理某个图像，但它处于错误的格式。大多数网络图像都是 `JPEG`（`.jpg`）文件。原生 Python 只能识别 `GIF`（`.gif`）格式。

## 准备工作

找到一个 `.jpg` 图像文件，并将其保存或复制到您为这项成像工作创建的目录中。为了进行这些练习，我们将假设有一个名为 `constr` 的目录（代表“建筑工地”）。在代码中，您将看到以 `/constr/pics1/images-name.ext` 形式引用的图像。这意味着 Python 程序期望在名为 `constr` 的系统目录中找到您请求它打开的文件。您可以将此更改为您认为最适合您在类似艺术家工作室等地方弄乱的地方。检索您的图像文件的路径甚至可以是网址。

因此，在这个例子中，有一个名为 `duzi_leo_1.jpg` 的图像，一个存储在名为 `pics1` 的文件夹（目录）中的 `JPEG` 图像，而 `pics1` 文件夹又位于 `constr` 目录中。

## 如何做...

按照通常的方式执行以下所示的程序。

```py
# images_jpg2png_1.py
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import Image
im_1 = Image.open("/constr/pics1/duzi_leo_1.jpg")
im_1.show()
im_1.save('/constr/picsx/duzi_leo_2.png', 'PNG')

```

## 它是如何工作的...

按照典型的 Python 风格，Python 的设计者已经尽可能地为程序员简化了事情。在这里发生的事情是，我们创建了一个名为 `im_1` 的图像对象实例，该对象以 `JPEG` 格式（扩展名为 `.jpg`）存在，并命令将其保存为 `PNG` 格式（扩展名为 `.png`）。复杂的转换在幕后进行。我们展示图像以确信它已被找到。

最后，我们将其转换为 `PNG` 格式并保存为 `duzi_leo_2.png`。

## 还有更多...

我们希望知道我们可以将任何图像格式转换为任何其他格式。不幸的是，图像格式有点像巴别塔现象。由于历史原因、技术演变、专利限制和专有商业霸权，许多图像格式并非旨在公开读取。例如，直到 2004 年，`GIF` 是专有的。`PNG` 是作为替代品开发的。下一个示例将展示用于发现您平台上的哪些转换将有效的代码。

# JPEG、PNG、TIFF、GIF、BMP 图像格式转换

我们从一个 `PNG` 格式的图像开始，然后将其保存为以下每种格式：`JPG、PNG、GIF、TIFF` 和 `BMP`，并将它们保存在本地硬盘上。然后我们取保存的图像格式，逐一将其转换为其他格式。这样我们就测试了所有可能的转换组合。

## 准备工作

我们需要将一个 `JPG` 图像放入 `/constr/pics1` 文件夹中。提供了一个名为 `test_pattern_a.png` 的特定 `PNG` 图像，旨在强调不同格式中的缺陷。

## 如何做...

按照之前所示执行程序。在命令终端阅读它们的描述性 '元数据'

![如何做...](img/3845_06_01.jpg)

```py
# images_one2another_1.py
#>>>>>>>>>>>>>>>>>>>>>
import Image
# Convert a jpg image to OTHER formats
im_1 = Image.open("/constr/pics1/test_pattern_1.jpg")
im_1.save('/constr/picsx/test_pattern_2.png', 'PNG')
im_1.save('/constr/picsx/test_pattern_3.gif', 'GIF')
im_1.save('/constr/picsx/test_pattern_4.tif', 'TIFF')
im_1.save('/constr/picsx/test_pattern_5.bmp', 'BMP')
# Convert a png image to OTHER formats
im_2 = Image.open("/constr/picsx/test_pattern_2.png")
im_2.save('/constr/picsx/test_pattern_6.jpg', 'JPEG')
im_2.save('/constr/picsx/test_pattern_7.gif', 'GIF')
im_2.save('/constr/picsx/test_pattern_8.tif', 'TIFF')
im_2.save('/constr/picsx/test_pattern_9.bmp', 'BMP')
# Convert a gif image to OTHER formats
# It seems that gif->jpg does not work
im_3 = Image.open("/constr/pics1/test_pattern_3.gif")
#im_3.save('/constr/pics1/test_pattern_10.jpg', 'JPEG')
# "IOError "cannot write mode P as JPEG"
im_3.save('/constr/picsx/test_pattern_11.png', 'PNG')
im_3.save('/constr/picsx/test_pattern_12.tif', 'TIFF')
im_3.save('/constr/picsx/test_pattern_13.bmp', 'BMP')
# Convert a tif image to OTHER formats
im_4 = Image.open("/constr/picsx/test_pattern_4.tif")
im_4.save('/constr/picsx/test_pattern_14.png', 'PNG')
im_4.save('/constr/picsx/test_pattern_15.gif', 'GIF')
im_4.save('/constr/picsx/test_pattern_16.tif', 'TIFF')
im_4.save('/constr/picsx/test_pattern_17.bmp', 'BMP')
# Convert a bmp image to OTHER formats
im_5 = Image.open("/constr/picsx/test_pattern_5.bmp")
im_5.save('/constr/picsx/test_pattern_18.png', 'PNG')
im_5.save('/constr/picsx/test_pattern_19.gif', 'GIF')
im_5.save('/constr/picsx/test_pattern_20.tif', 'TIFF')
im_5.save('/constr/picsx/test_pattern_21.jpg', 'JPEG')

```

## 它是如何工作的...

这种转换仅在 PIL 安装的情况下有效。一个例外是，从 `GIF` 到 `JPG` 的转换将不会工作。在执行程序之前，预先打开 `/constr/pics1` 文件夹的内容，并观察图像在执行过程中依次出现，这很有趣。

## 还有更多...

注意，除了`GIF`图像外，很难注意到任何图像质量丢失。问题最明显的是，当`GIF`转换算法必须如图所示在两种相似颜色之间做出选择时。

### 大小重要吗？

原始的`test_pattern_1.jpg`是 77 千字节。所有由此派生的图像大小是四到十倍，即使是低质量的`GIF`图像。原因是只有`JPG`和`GIF`图像是损失性的，这意味着在转换过程中会丢弃一些图像信息，并且无法恢复。

# 图像在图像平面上的旋转

我们有一个侧躺的图像，我们需要通过顺时针旋转 90 度来修复它。我们想要存储修复后的图像副本。

## 准备工作

我们需要将一个`PNG`图像放入文件夹`/constr/pics1`中。在以下代码中，我们使用了图像`dusi_leo.png`。此图像具有突出的红色和黄色成分。

## 如何做到这一点...

执行之前显示的程序。

```py
# image_rotate_1.py
#>>>>>>>>>>>>>>>>
import Image
im_1 = Image.open("/constr/pics1/dusi_leo.png")
im_2= im_1.rotate(-90)
im_2.show()
im_2.save("/constr/picsx/dusi_leo_rightway.png")

```

## 它是如何工作的...

显示的图像将正确对齐。请注意，我们可以将图像旋转到最小为一度，但不能小于这个数值。还有其他变换可以将图像旋转到 90 度的整数倍。这些在标题为“多重变换”的部分进行了演示。

## 还有更多...

我们如何创建平滑旋转图像的效果？仅使用 PIL 是无法实现的。PIL 旨在执行图像操作和变换的计算密集型操作。不可能在时间控制的序列中显示一个图像接着另一个图像。为此，你需要 Tkinter，而 Tkinter 只与`GIF`图像一起工作。

我们会首先创建一系列图像，每个图像比前一个图像稍微旋转一点，并存储每个图像。稍后，我们将运行一个 Python Tkinter 程序，以时间控制的序列显示这些图像系列。这将使旋转动画化。与旋转图像必须放置在具有与原始图像相同大小和方向的框架中的事实相关的问题将在下一章中解决。可以取得一些令人惊讶的有效结果。

# 图像大小调整

我们将一个大型图像文件（1.8 兆字节）的大小减小到 24.7 千字节。但是图像被扭曲了，因为没有考虑到高度与宽度的比例。

## 准备工作

就像我们对之前的食谱所做的那样，我们需要将`dusi_leo.png`图像放入文件夹`/constr/pics1`中。

## 如何做到这一点...

要更改图像的大小，我们需要指定最终尺寸，并指定一个过滤器类型，该类型提供填充由调整大小过程产生的任何像素间隙的规则。执行之前显示的程序。以下图像显示了结果。

![如何做到这一点...](img/3845_06_02.jpg)

```py
# image_resize_1.py
#>>>>>>>>>>>>>>>>
import Image
im_1 = Image.open("/constr/pics1/dusi_leo_1.jpg")
# adjust width and height to desired size
width = 300
height = 300
# NEAREST Filter is compulsory to resize the image
im_2 = im_1.resize((width, height), Image.NEAREST)
im_2.save("/constr/picsx/dusi_leo_2.jpg")

```

## 它是如何工作的...

在这里，我们将图片的大小调整为适合 300x300 像素的矩形。如果图像增大，需要添加额外的像素。如果图像减小，则必须丢弃像素。添加哪些特定的像素，它们的颜色将如何确定，必须由调整大小方法中的算法自动决定。PIL 提供了多种方法来完成这项工作。这些像素添加算法作为 **过滤器** 提供使用。

这正是过滤器设计的目的，在前面的例子中，我们选择使用最近的过滤器。这被记录为使用最近的邻域像素的值。文档有些含糊不清，因为它没有解释将选择哪个最近的像素。在矩形网格中，北、南、东、西的像素距离相等。其他可能的过滤器包括 BILINEAR（在 2x2 环境中的线性插值）、BICUBIC（在 4x4 环境中的三次样条插值）或 ANTIALIAS（高质量的下采样过滤器）。

减小图像也带来了困境。图片元素（像素）需要被丢弃。如果图像中有一个从黑色到白色的锐利边缘，会发生什么？我们是丢弃最后一个黑色像素并用白色像素替换它，还是用介于两者之间的像素替换它？

## 还有更多...

哪些过滤器最好的问题会因图像类型而异。在这个不断变化的图像处理分支中，需要经验和实验。

### 如何保持图像的正确宽高比？

如此例所示，除非我们采取措施选择正确的目标图像尺寸比例，否则瘦阿姨 Milly 的照片将变成宽 Winifred，如果照片被展示出来，家庭成员之间可能永远无法和睦相处。因此，我们在下一个菜谱中展示了如何保持比例和礼仪。

# 正确的比例图像调整大小

我们制作一个缩小尺寸的图像，同时注意保持原始图像的正确宽高比和长宽比。秘诀是使用 `Image.size()` 函数事先获取确切的图像大小，然后确保我们保持相同的长宽比。

## 准备工作

正如我们之前的做法，我们需要将 `dusi_leo.png` 图像放入文件夹 `/constr/pics1`。

## 如何操作...

执行之前显示的程序。

```py
# image_preserved_aspect_resize_1.py
#>>>>>>>>>>>>>>>>>>>>>>>>>>
import Image
im_1 = Image.open("/constr/pics1/dusi_leo_1.jpg")
im_width = im_1.size[0]
im_height = im_1.size[1]
print im_width, im_height
new_size = 0.2 # New image to be reduced to one fifth of original.
# adjust width and height to desired size
width = int(im_width * new_size)
height = int(im_height * new_size)
# Filter is compulsory to resize the image
im_2 = im_1.resize((width, height), Image.NEAREST)
im_2.save("/constr/picsx/dusi_leo_3.jpg")

```

## 它是如何工作的...

`Image.size()` 函数返回两个整数，- 打开图像的宽度，size[0]，和高度，size[1]。我们使用缩放乘数 `new_size` 以相同的比例缩放宽度和高度。

# 在图像中分离一个颜色带

我们仅隔离图像的绿色部分或颜色带。

## 准备工作

正如我们之前的做法，我们需要将 `dusi_leo.png` 图像放入文件夹 `/constr/pics1`。

## 如何操作...

执行之前显示的程序。

```py
#image_get_green_1.py
#>>>>>>>>>>>>>>>>>>
import ImageEnhance
import Image
red_frac = 1.0
green_frac = 1.0
blue_frac = 1.0
im_1 = Image.open("/a_constr/pics1/dusi_leo_1.jpg")
# split the image into individual bands
source = im_1.split()
R, G, B = 0, 1, 2
# Assign color intensity bands, zero for red and blue.
red_band = source[R].point(lambda i: i * 0.0)
green_band = source[G]
blue_band = source[B].point(lambda i: i * 0.0)
new_source = [red_band, green_band, blue_band]
# Merge (add) the three color bands
im_2 = Image.merge(im_1.mode, new_source)
im_2.show()

```

## 它是如何工作的...

`Image.split()`函数将原始`JPG`图像中的红色、绿色和蓝色三个颜色波段分离出来。红色波段是`source[0]`，绿色波段是`source[1]`，蓝色波段是`[2]`。`JPG`图像没有透明度（alpha）波段。`PNG`图像可以有 alpha 波段。如果这样的`PNG`图像被`split()`，其透明度波段将是`source[3]`。图像中特定像素的颜色量以字节数据记录。您可以通过在分割波段中为每个像素的比例进行类似的调整来改变这个量，例如在`red_band = source[R].point(lambda i: i * proportion)`这一行中，其中比例是一个介于`0.0`和`1.0`之间的数字。

在这个菜谱中，我们通过将比例量设置为`0.0`来消除所有红色和蓝色。

## 还有更多...

在下一个菜谱中，我们以非零比例混合三种颜色。

# 图像中的红、绿、蓝颜色调整

在这个例子中，我们进一步制作了一个重新混合原始图像颜色的图像，以不同的比例。与之前的例子使用相同的代码布局。

## 准备中

如前所述，将`dusi_leo.png`图像放入文件夹`/constr/pics1`中。

## 如何做...

执行以下代码。

```py
#image_color_manip_1.py
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import ImageEnhance, Image
im_1 = Image.open("/constr/pics1/dusi_leo_smlr_1.jpg")
# Split the image into individual bands
source = im_1.split()
R, G, B = 0, 1, 2
# Select regions where red is less than 100
red_band = source[R]
green_band = source[G]
blue_band = source[B]
# Process the red band: intensify red x 2
out_red = source[R].point(lambda i: i * 2.0)
# Process the green band: weaken by 20%
out_green = source[G].point(lambda i: i * 0.8)
# process the blue band: Eliminate all blue
out_blue = source[B].point(lambda i: i * 0.0)
# Make a new source of color band values
new_source = [out_red, out_green, out_blue]
# Add the three altered bands back together
im_2 = Image.merge(im_1.mode, new_source)
im_2.show()

```

## 它是如何工作的...

如前所述，`Image.split()`函数将原始`JPG`图像的红色、绿色和蓝色三个颜色波段分离出来。在这种情况下，红色、绿色和蓝色的比例分别是蓝色 200%，20%，和 0%。

## 还有更多...

在现有图片中调整颜色比例是一种复杂而微妙的艺术，正如我们在前面的章节中所做的那样，在下一个例子中，我们提供了一个使用滑动控制来允许用户通过试错法在波段分离的图像上达到期望颜色混合的菜谱。

# 滑块控制的颜色调整

我们构建了一个工具，用于在波段分离的图像上获得期望的颜色混合。之前我们使用过的滑动控制是一个方便的设备，用于有意识地调整每个主色波段中颜色的相对比例。

## 准备中

使用文件夹`/constr/pics1`中的`dusi_leo.png`图像。

## 如何做...

执行以下程序。

```py
#image_color_adjuster_1.py
#>>>>>>>>>>>>>>>>>>>
import ImageEnhance
import Image
import Tkinter
root =Tkinter.Tk()
root.title("Photo Image Color Adjuster")
red_frac = 1.0
green_frac = 1.0
blue_frac = 1.0
slide_value_red = Tkinter.IntVar()
slide_value_green = Tkinter.IntVar()
slide_value_blue = Tkinter.IntVar()
im_1 = Image.open("/constr/pics1/dusi_leo_smlr_1.jpg")
im_1.show()
source = im_1.split() # split the image into individual bands
R, G, B = 0, 1, 2
# Assign color intensity bands
red_band = source[R]
green_band = source[G]
blue_band = source[B]
#===============================================
# Slider and Button event service functions (callbacks)
def callback_button_1():
toolconstructing, for desirable color mix# Adjust red intensity by slider value.
out_red = source[R].point(lambda i: i * red_frac)
out_green = source[G].point(lambda i: i * green_frac) # Adjust # green
out_blue = source[B].point(lambda i: i * blue_frac) # Adjust # blue
new_source = [out_red, out_green, out_blue]
im_2 = Image.merge(im_1.mode, new_source) # Re-combine bands
im_2.show()
button_1= Tkinter.Button(root,bg= "sky blue", text= "Display adjusted image \
(delete previous one)", command=callback_ \button_1)
button_1.grid(row=1, column=2, columnspan=3)
def callback_red(*args):
global red_frac
red_frac = slide_value_red.get()/100.0
def callback_green(*args):
global green_frac
green_frac = slide_value_green.get()/100.0
def callback_blue(*args):
global blue_frac
blue_frac = slide_value_blue.get()/100.0
slide_value_red.trace_variable("w", callback_red)
slide_value_green.trace_variable("w", callback_green)
slide_value_blue.trace_variable("w", callback_blue)
slider_red = Tkinter.Scale(root,
length = 400,
fg = 'red',
activebackground = "tomato",
background = "grey",
troughcolor = "red",
label = "RED",
from_ = 0,
to = 200,
resolution = 1,
variable = slide_value_red,
orient = 'vertical')
slider_red.grid(row=0, column=2)
slider_green = Tkinter.Scale(root,
length = 400,
fg = 'dark green',
activebackground = "green yellow",
background = "grey",
troughcolor = "green",
label = "GREEN",
from_ = 0,
to = 200,
toolconstructing, for desirable color mixresolution = 1,
variable = slide_value_green,
orient = 'vertical')
slider_green.grid(row=0, column=3)
slider_blue = Tkinter.Scale(root,
length = 400,
fg = 'blue',
activebackground = "turquoise",
background = "grey",
troughcolor = "blue",
label = "BLUE",
from_ = 0,
to = 200,
resolution = 1,
variable = slide_value_blue,
orient = 'vertical')
slider_blue.grid(row=0, column=4)
root.mainloop()
#===============================================

```

## 它是如何工作的...

使用鼠标控制的滑块位置，我们调整每个红色、绿色和蓝色通道中的颜色强度。调整的幅度从零到 200，但在回调函数中缩放到百分比值。

在`source = im_1.split()`中，`split()`方法将图像分割成红色、绿色和蓝色波段。`point(lambda i: i * intensity)`方法将每个波段中每个像素的颜色值乘以一个`intensity`值，而`merge(im_1.mode, new_source)`方法将结果波段重新组合成一个新的图像。

在这个例子中，我们使用了 PIL 和 Tkinter 一起。

如果你使用`from Tkinter import *`，你似乎会得到命名空间混淆：

解释器说：

`" im_1 = Image.open("/a_constr/pics1/redcar.jpg")`

`AttributeError: class Image has no attribute 'open' "`

但如果你只是说`import Tkinter`，这似乎是可行的。

但当然现在你必须将 Tkinter 的所有方法前缀为 Tkinter。

# 通过混合组合图像

混合两个图像的效果就像从两个不同的投影仪将两个透明的幻灯片图像投射到投影仪屏幕上，每个投影仪的光量由比例设置控制。命令的形式是`Image.blend(image_1, image_2, proportion-of-image_1)`。

## 准备工作

使用来自文件夹`/constr/pics1`的两个图像`100_canary.png`和`100_cockcrow.png`。标题中的`100_`是一个提醒，表示这些图像的大小是 100 x 100 像素，我们将在控制台上看到每个图像的格式、大小和类型。

## 如何操作...

在放置好两个大小和模式相同的图像后，执行以下代码。

```py
# image_blend_1.py
# >>>>>>>>>>>>>>>>
import Image
im_1 = Image.open("/constr/pics1/100_canary.png") # mode is RGBA
im_2 = Image.open("/constr/pics1/100_cockcrow.png") # mode is RGB
# Check on mode, size and format first for compatibility.
print "im_1 format:", im_1.format, ";size:", im_1.size, "; mode:",im_1.mode
print "im_2 format:", im_2.format, ";size:", im_2.size, "; mode:",im_2.mode
im_2 = im_2.convert("RGBA") # Make both modes the same
im_4 = Image.blend(im_1, im_2, 0.5)
im_4.show()

```

## 更多...

从格式信息中，我们可以看到第一幅图像的模式是`RGBA`，而第二幅是`RGB`。因此，首先需要将第二幅图像转换为`RGBA`。

在这个特定的例子中，比例控制被设置为`0.5`。这意味着两幅图像以相等的量混合在一起。如果比例设置是`0.2`，那么`im_1`的`20%`将与`im_2`的`80%`结合。

### 更多信息部分 1

另一种组合图像的方法是使用第三幅图像作为遮罩来控制遮罩确定的结果图像中每个图像的形状和比例的位置。

# 通过改变百分比混合图像

我们以不同的透明度混合两个图像。

## 如何操作...

在放置好两个大小和模式相同的图像后，执行以下代码。

```py
# image_blend_2.py
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import Image
im_1 = Image.open("/constr/pics1/lion_ramp_2.png")
im_2 = Image.open("/constr/pics1/fiery_2.png")
# Various degrees of alpha
im_3 = Image.blend(im_1, im_2, 0.05) # 95% im_1, 5% im_2
im_4 = Image.blend(im_1, im_2, 0.2)
im_5 = Image.blend(im_1, im_2, 0.5)
im_6 = Image.blend(im_1, im_2, 0.6)
im_7 = Image.blend(im_1, im_2, 0.8)
im_8 = Image.blend(im_1, im_2, 0.95) # 5% im_1, 95% im_2
im_3.save("/constr/picsx/fiery_lion_1.png")
im_4.save("/constr/picsx/fiery_lion_2.png")
im_5.save("/constr/picsx/fiery_lion_3.png")
im_6.save("/constr/picsx/fiery_lion_4.png")
im_7.save("/constr/picsx/fiery_lion_5.png")
im_8.save("/constr/picsx/fiery_lion_6.png")

```

## 它是如何工作的...

通过改变图像混合时使用的 alpha 值，我们可以控制每个图像在结果中占主导的程度。

## 更多...

这种在每个电影帧上执行的过程是常用于从一个场景淡出到另一个场景的效果。

# 使用遮罩图像制作合成图像

在这里，我们使用函数`Image.composite(image_1, image_2, mask_image)`来控制两个图像的组合。

## 准备工作

使用来自文件夹`/constr/pics1`的图像`100_canary.png, 100_cockcrow.png`和`100_sun_1.png`。标题中的`100_`是一个提醒，表示这些图像的大小是 100 x 100 像素，我们将在控制台上看到每个图像的格式、大小和类型。

## 如何操作...

在放置好三个大小和模式相同的图像后，执行以下代码。

```py
# image_composite_1.py
# >>>>>>>>>>>>>>>>>
import Image
im_1 = Image.open("/constr/pics1/100_canary.png") # mode is RGBA
im_2 = Image.open("/constr/pics1/100_cockcrow.png") # mode is RGB
im_3 = Image.open("/constr/pics1/100_sun_1.png")
# Check on mode, size and format first for compatibility.
print "im_1 format:", im_1.format, ";size:", im_1.size, "; mode:", \im_1.mode
print "im_2 format:", im_2.format, ";size:", im_2.size, "; mode:", \im_2.mode
print "im_3 format:", im_3.format, ";size:", im_3.size, "; mode:", \im_3.mode
im_2 = im_2.convert("RGBA")
im_3 = im_3.convert("L")
im_4 = Image.composite(im_1, im_2, im_3)
im_4.show()

```

## 它是如何工作的...

从格式信息中，我们可以看到第一幅图像的模式是`RGBA`，而第二幅是`RGB`。因此，首先需要将第二幅图像转换为`RGBA`。

遮罩图像必须是`1, L`或`RGBA`格式，并且大小相同。在这个菜谱中，我们将其转换为模式`L`，这是一个 256 级的灰度图像。遮罩中每个像素的值用于乘以源图像。如果某个特定位置的像素值为`56`，那么`image_1`将被乘以`256 * 56 = 200`，而`image_2`将被乘以`56`。

## 还有更多...

还有其他效果，如`Image.eval(function, Image)`，其中每个像素都乘以该函数，并且我们可以将函数转换为一些复杂的代数表达式。如果图像有多个波段，则函数应用于每个波段。

另一种效果是`Image.merge(mode, bandList)`，它从多个相同大小的单波段图像创建多波段图像。我们指定新图像的所需模式。**bandList 指定器**是要组合的单波段图像组件的序列。

## 参见

通过组合之前显示的图像操作，可以实现无数的效果。我们将深入到图像和信号处理的世界，这可能会变得极其复杂和精细。某些效果已经相当标准化，可以在图像处理应用程序（如`GIMP`和`Photoshop`）提供的过滤选项列表中看到。

# 水平和垂直偏移（滚动）图像

我们在这里看到如何滚动图像。也就是说，将其向右或向左移动而不丢失任何东西——图像实际上就像边缘连接一样滚动。同样的过程在垂直方向上也会起作用。

## 准备工作

使用文件夹`/constr/pics1`中的`image_canary_a.jpg`。

## 如何操作...

执行以下程序，注意我们需要导入 PIL 中的一个模块，名为`ImageChops`。**Chops**代表**通道操作**。

```py
# image_offset_1.py
# >>>>>>>>>>>>>>>
import Image
import ImageChops
im_1 = Image.open("/constr/pics1/canary_a.jpg")
# adjust width and height to desired size
dx = 200
dy = 300
im_2 = ImageChops.offset(im_1, dx, dy)
im_2.save("/constr/picsx/canary_2.jpg")

```

## 它是如何工作的...

约翰·希普曼（John Shipman）于 2009 年中旬撰写的优秀指南《Python Imaging Library》没有提及 ImageChops。

# 水平、垂直翻转和旋转

在这里，我们查看一组快速且易于操作的变换，这些操作在照片查看器中很典型。

## 准备工作

使用文件夹`/constr/pics1`中的`image_dusi_leo_1.jpg`。

## 如何操作...

执行以下程序，并在`/constr/picsx`中查看结果。

```py
# image_flips_1.py
#>>>>>>>>>>>>>>
import Image
im_1 = Image.open("/a_constr/pics1/dusi_leo_1.jpg")
im_out_1 = im_1.transpose(Image.FLIP_LEFT_RIGHT)
im_out_2 = im_1.transpose(Image.FLIP_TOP_BOTTOM)
im_out_3 = im_1.transpose(Image.ROTATE_90)
im_out_4 = im_1.transpose(Image.ROTATE_180)
im_out_5 = im_1.transpose(Image.ROTATE_270)
im_out_1.save("/a_constr/picsx/dusi_leo_horizontal.jpg")
im_out_1.save("/a_constr/picsx/dusi_leo_vertical.jpg")
im_out_1.save("/a_constr/picsx/dusi_leo_90.jpg")
im_out_1.save("/a_constr/picsx/dusi_leo_180.jpg")
im_out_1.save("/a_constr/picsx/duzi_leo_270.jpg")

```

## 它是如何工作的...

前面的命令是自我解释的。

# 过滤效果：模糊、锐化、对比度等

PIL 有一个**ImageFilter**模块，它包含了一系列有用的过滤器，可以增强图像的某些特性，例如锐化模糊的特征。以下菜谱中演示了其中的十个过滤器。

## 准备工作

使用文件夹 `/constr/pics1` 中的图像 `russian_doll.png`。为结果过滤图像创建一个文件夹 `/constr/picsx`。使用单独的文件夹来存储结果有助于防止文件夹 `pics1` 因冗余图像和工作中的图像而变得拥挤。每次执行后，我们可以删除 `picsx` 的内容，而不用担心会丢失菜谱的源图像。

## 如何操作...

在执行以下代码之前，请在您的屏幕上打开文件夹 `/constr/picsx`，并在执行完成后观察图像的出现。源图像被选择为模糊的俄罗斯娃娃和清晰的背景，因为这允许我们轻松地区分不同过滤器的影响。

```py
# image_pileof_filters_1.py
# >>>>>>>>>>>>>>>>>>>
import ImageFilter
im_1 = Image.open("/constr/pics1/russian_doll.png")
im_2 = im_1.filter(ImageFilter.BLUR)
im_3 = im_1.filter(ImageFilter.CONTOUR)
im_4 = im_1.filter(ImageFilter.DETAIL)
im_5 = im_1.filter(ImageFilter.EDGE_ENHANCE)
im_6 = im_1.filter(ImageFilter.EDGE_ENHANCE_MORE)
im_7 = im_1.filter(ImageFilter.EMBOSS)
im_8 = im_1.filter(ImageFilter.FIND_EDGES)
im_9 = im_1.filter(ImageFilter.SMOOTH)
im_10 = im_1.filter(ImageFilter.SMOOTH_MORE)
im_11 = im_1.filter(ImageFilter.SHARPEN)
im_2.save("/constr/picsx/russian_doll_BLUR.png")
im_3.save("/constr/picsx/ russian_doll_CONTOUR.png")
im_4.save("/constr/picsx/ russian_doll_DETAIL.png")
im_5.save("/constr/picsx/ russian_doll_EDGE_ENHANCE.png")
im_6.save("/constr/picsx/ russian_doll_EDGE_ENHANCE_MORE.png")
im_7.save("/constr/picsx/ russian_doll_EMBOSS.png")
im_8.save("/constr/picsx/ russian_doll_FIND_EDGES.png")
im_9.save("/constr/picsx/ russian_doll_SMOOTH.png")
im_10.save("/constr/picsx/ russian_doll_SMOOTH_MORE.png")
im_11.save("/constr/picsx/ russian_doll_SHARPEN.png")

```

## 它是如何工作的...

这个菜谱表明，最佳的过滤结果高度依赖于我们希望增强或抑制的图像特征，以及正在处理的单个图像的一些微妙特征。在下面这个例子中，`EDGE_ENHANCE` 过滤器特别有效于对抗娃娃的模糊焦点。与 `SHARPEN` 过滤器相比，它提高了色彩对比度。调整大小并粘贴以进行合成旋转

我们想要创建一个动画，使图像看起来在图片中间围绕一个垂直轴旋转。在这个菜谱中，我们看到动画的基本图像序列是如何准备的。

我们想要制作一系列图像，这些图像逐渐变窄，就像它们是一张逐渐旋转的板上的海报。然后，我们想要将这些窄图像粘贴在标准尺寸黑色背景的中间。如果这个序列以时间控制的帧序列显示，我们会看到图像似乎围绕一个中心垂直轴旋转。

## 准备工作

我们在目录 `/constr/pics1` 中使用图像 `100_canary.png`，并将结果放置在 `/constr/picsx` 中，以避免我们的源文件夹 `/constr/pics1` 过于杂乱。

## 如何操作...

再次在执行以下代码之前，在您的屏幕上打开文件夹 `/constr/picsx`，并在执行完成后观察图像的出现。这不是必需的，但观看结果在眼前形成是非常有趣的。

```py
# image_rotate_resize_1.py
# >>>>>>>>>>>>>>>>>>>>
import Image
import math
THETADEG = 5.0 # degrees
THETARAD = math.radians(THETADEG)
im_1 = Image.open("/constr/pics1/blank.png")
im_seed = Image.open("/constr/pics1/100_canary.png") # THE SEED IMAGE
im_seq_name = "canary"
#GET IMAGE WIDTH AND HEIGHT - not done here
# For the time being assume the image is 100 x 100
width = 100
height = 100
num_images = int(math.pi/(2*THETARAD))
Q = []
for j in range(0,2*num_images + 1):
Q.append(j)
for i in range(0, num_images):
new_size = width * math.cos(i*THETARAD) # Width for reduced # image
im_temp = im_seed.resize((new_size, height), Image.NEAREST)
im_width = im_temp.size[0] # Get the width of the reduced image
x_start = 50 -im_width/2 # Centralize new image in a 100x100 # square.
im_1.paste(im_temp,( x_start,10)) # Paste: This creates the # annoying ghosting.
stri = str(i)
# Save the reduced image
Q[i] = "/constr/picsx/" + im_seq_name + stri + ".gif"
im_1.save(Q[i])
# Flip horizontally and save the reduced image.
im_transpose = im_temp.transpose(Image.FLIP_LEFT_RIGHT)
im_1.paste(im_transpose,( x_start,10))
strj = str(2 * num_images - i)
Q[ 2 * num_images - i ] = "/constr/picsx/" + im_seq_name + strj \ + ".gif"
im_1.save(Q[ 2 * num_images - i ])

```

## 它是如何工作的...

为了模仿旋转的效果，我们将每个图像的宽度减少到 `cosine(new_angle)`，其中 `new_angle` 对于每个图像增加 5 度的旋转。然后我们把这个缩小的图像粘贴到一个空白黑色正方形上。最后，我们以系统的方式给序列中的每张图片命名，例如 `canary0.gif, canary1.gif`，依此类推，直到最后一张图片命名为 `canary36.gif`。

## 更多内容...

这个例子展示了 Python 图像库非常适合的任务类型——当你需要反复对一个图像或一组图像执行受控的转换时。这些图像可能是电影胶片的帧。如淡入淡出、缩放、颜色变换、锐化、模糊等效果是显而易见的，可以用到的，但你的程序员的想象力还能想出许多其他的效果。
