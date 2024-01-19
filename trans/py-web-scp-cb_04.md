# 第四章：处理图像、音频和其他资产

在本章中，我们将涵盖：

+   在网上下载媒体内容

+   使用 urllib 解析 URL 以获取文件名

+   确定 URL 的内容类型

+   从内容类型确定文件扩展名

+   下载并将图像保存到本地文件系统

+   下载并将图像保存到 S3

+   为图像生成缩略图

+   使用 Selenium 进行网站截图

+   使用外部服务对网站进行截图

+   使用 pytessaract 对图像执行 OCR

+   创建视频缩略图

+   将 MP4 视频转换为 MP3

# 介绍

在抓取中的一个常见做法是下载、存储和进一步处理媒体内容（非网页或数据文件）。这些媒体可以包括图像、音频和视频。为了正确地将内容存储在本地（或在 S3 等服务中），我们需要知道媒体类型，并且仅仅信任 URL 中的文件扩展名是不够的。我们将学习如何根据来自 Web 服务器的信息下载和正确表示媒体类型。

另一个常见的任务是生成图像、视频甚至网站页面的缩略图。我们将研究如何生成缩略图并制作网站页面截图的几种技术。这些缩略图经常用作新网站上缩略图链接，以链接到现在存储在本地的抓取媒体。

最后，通常需要能够转码媒体，例如将非 MP4 视频转换为 MP4，或更改视频的比特率或分辨率。另一个场景是从视频文件中提取音频。我们不会讨论视频转码，但我们将使用`ffmpeg`从 MP4 文件中提取 MP3 音频。从那里开始，还可以使用`ffmpeg`转码视频。

# 从网上下载媒体内容

从网上下载媒体内容是一个简单的过程：使用 Requests 或其他库，就像下载 HTML 内容一样。

# 准备工作

解决方案的`util`文件夹中的`urls.py`模块中有一个名为`URLUtility`的类。该类处理本章中的几种场景，包括下载和解析 URL。我们将在这个配方和其他一些配方中使用这个类。确保`modules`文件夹在您的 Python 路径中。此外，此配方的示例位于`04/01_download_image.py`文件中。

# 如何做到这一点

以下是我们如何进行的步骤：

1.  `URLUtility`类可以从 URL 下载内容。配方文件中的代码如下：

```py
import const
from util.urls import URLUtility

util = URLUtility(const.ApodEclipseImage())
print(len(util.data))
```

1.  运行时，您将看到以下输出：

```py
Reading URL: https://apod.nasa.gov/apod/image/1709/BT5643s.jpg
Read 171014 bytes
171014
```

示例读取了`171014`字节的数据。

# 它是如何工作的

URL 被定义为`const`模块中的常量`const.ApodEclipseImage()`：

```py
def ApodEclipseImage():
    return "https://apod.nasa.gov/apod/image/1709/BT5643s.jpg" 
```

`URLUtility`类的构造函数具有以下实现：

```py
def __init__(self, url, readNow=True):
    """ Construct the object, parse the URL, and download now if specified"""
  self._url = url
    self._response = None
  self._parsed = urlparse(url)
    if readNow:
        self.read()
```

构造函数存储 URL，解析它，并使用`read()`方法下载文件。以下是`read()`方法的代码：

```py
def read(self):
    self._response = urllib.request.urlopen(self._url)
    self._data = self._response.read()
```

该函数使用`urlopen`获取响应对象，然后读取流并将其存储为对象的属性。然后可以使用数据属性检索该数据：

```py
@property def data(self):
    self.ensure_response()
    return self._data
```

然后，该代码简单地报告了该数据的长度，值为`171014`。

# 还有更多...

这个类将用于其他任务，比如确定文件的内容类型、文件名和扩展名。接下来我们将研究解析 URL 以获取文件名。

# 使用 urllib 解析 URL 以获取文件名

从 URL 下载内容时，我们经常希望将其保存在文件中。通常情况下，将文件保存在 URL 中找到的文件名中就足够了。但是 URL 由许多片段组成，那么我们如何从 URL 中找到实际的文件名，特别是在文件名后经常有许多参数的情况下？

# 准备工作

我们将再次使用`URLUtility`类来完成这个任务。该配方的代码文件是`04/02_parse_url.py`。

# 如何做到这一点

使用您的 Python 解释器执行配方文件。它将运行以下代码：

```py
util = URLUtility(const.ApodEclipseImage())
print(util.filename_without_ext)
```

这导致以下输出：

```py
Reading URL: https://apod.nasa.gov/apod/image/1709/BT5643s.jpg
Read 171014 bytes
The filename is: BT5643s
```

# 它是如何工作的

在`URLUtility`的构造函数中，调用了`urlib.parse.urlparse`。 以下演示了交互式使用该函数：

```py
>>> parsed = urlparse(const.ApodEclipseImage())
>>> parsed
ParseResult(scheme='https', netloc='apod.nasa.gov', path='/apod/image/1709/BT5643s.jpg', params='', query='', fragment='')
```

`ParseResult`对象包含 URL 的各个组件。 路径元素包含路径和文件名。 对`.filename_without_ext`属性的调用仅返回没有扩展名的文件名：

```py
@property def filename_without_ext(self):
    filename = os.path.splitext(os.path.basename(self._parsed.path))[0]
    return filename
```

对`os.path.basename`的调用仅返回路径的文件名部分（包括扩展名）。 `os.path.splittext()`然后分隔文件名和扩展名，并且该函数返回该元组/列表的第一个元素（文件名）。

# 还有更多...

这似乎有点奇怪，它没有将扩展名作为文件名的一部分返回。 这是因为我们不能假设我们收到的内容实际上与扩展名所暗示的类型匹配。 更准确的是使用 Web 服务器返回的标题来确定这一点。 这是我们下一个配方。

# 确定 URL 的内容类型

当从 Web 服务器获取内容的`GET`请求时，Web 服务器将返回许多标题，其中一个标识了内容的类型，从 Web 服务器的角度来看。 在这个配方中，我们学习如何使用它来确定 Web 服务器认为的内容类型。

# 做好准备

我们再次使用`URLUtility`类。 配方的代码在`04/03_determine_content_type_from_response.py`中。

# 如何做到这一点

我们按以下步骤进行：

1.  执行配方的脚本。 它包含以下代码：

```py
util = URLUtility(const.ApodEclipseImage())
print("The content type is: " + util.contenttype)
```

1.  得到以下结果：

```py
Reading URL: https://apod.nasa.gov/apod/image/1709/BT5643s.jpg
Read 171014 bytes
The content type is: image/jpeg
```

# 它是如何工作的

`.contentype`属性的实现如下：

```py
@property def contenttype(self):
    self.ensure_response()
    return self._response.headers['content-type']
```

`_response`对象的`.headers`属性是一个类似字典的标题类。 `content-type`键将检索服务器指定的`content-type`。 对`ensure_response()`方法的调用只是确保已执行`.read()`函数。

# 还有更多...

响应中的标题包含大量信息。 如果我们更仔细地查看响应的`headers`属性，我们可以看到返回以下标题：

```py
>>> response = urllib.request.urlopen(const.ApodEclipseImage())
>>> for header in response.headers: print(header)
Date
Server
Last-Modified
ETag
Accept-Ranges
Content-Length
Connection
Content-Type
Strict-Transport-Security
```

我们可以看到每个标题的值。

```py
>>> for header in response.headers: print(header + " ==> " + response.headers[header])
Date ==> Tue, 26 Sep 2017 19:31:41 GMT
Server ==> WebServer/1.0
Last-Modified ==> Thu, 31 Aug 2017 20:26:32 GMT
ETag ==> "547bb44-29c06-5581275ce2b86"
Accept-Ranges ==> bytes
Content-Length ==> 171014
Connection ==> close
Content-Type ==> image/jpeg
Strict-Transport-Security ==> max-age=31536000; includeSubDomains
```

这本书中有许多我们不会讨论的内容，但对于不熟悉的人来说，知道它们存在是很好的。

# 从内容类型确定文件扩展名

使用`content-type`标题来确定内容的类型，并确定用于存储内容的扩展名是一个很好的做法。

# 做好准备

我们再次使用了我们创建的`URLUtility`对象。 配方的脚本是`04/04_determine_file_extension_from_contenttype.py`。

# 如何做到这一点

通过运行配方的脚本来进行。

可以使用`.extension`属性找到媒体类型的扩展名：

```py
util = URLUtility(const.ApodEclipseImage())
print("Filename from content-type: " + util.extension_from_contenttype)
print("Filename from url: " + util.extension_from_url)
```

这导致以下输出：

```py
Reading URL: https://apod.nasa.gov/apod/image/1709/BT5643s.jpg
Read 171014 bytes
Filename from content-type: .jpg
Filename from url: .jpg
```

这报告了从文件类型和 URL 确定的扩展名。 这些可能不同，但在这种情况下它们是相同的。

# 它是如何工作的

以下是`.extension_from_contenttype`属性的实现：

```py
@property def extension_from_contenttype(self):
    self.ensure_response()

    map = const.ContentTypeToExtensions()
    if self.contenttype in map:
        return map[self.contenttype]
    return None 
```

第一行确保我们已从 URL 读取响应。 然后，该函数使用在`const`模块中定义的 Python 字典，其中包含内容类型到扩展名的字典：

```py
def ContentTypeToExtensions():
    return {
        "image/jpeg": ".jpg",
  "image/jpg": ".jpg",
  "image/png": ".png"
  }
```

如果内容类型在字典中，则将返回相应的值。 否则，将返回`None`。

注意相应的属性`.extension_from_url`：

```py
@property def extension_from_url(self):
    ext = os.path.splitext(os.path.basename(self._parsed.path))[1]
    return ext
```

这使用与`.filename`属性相同的技术来解析 URL，但是返回代表扩展名而不是基本文件名的`[1]`元素。

# 还有更多...

如前所述，最好使用`content-type`标题来确定用于本地存储文件的扩展名。 除了这里提供的技术之外，还有其他技术，但这是最简单的。

# 下载并将图像保存到本地文件系统

有时在爬取时，我们只是下载和解析数据，比如 HTML，提取一些数据，然后丢弃我们读取的内容。其他时候，我们希望通过将其存储为文件来保留已下载的内容。

# 如何做

这个配方的代码示例在`04/05_save_image_as_file.py`文件中。文件中重要的部分是：

```py
# download the image item = URLUtility(const.ApodEclipseImage())

# create a file writer to write the data FileBlobWriter(expanduser("~")).write(item.filename, item.data)
```

用你的 Python 解释器运行脚本，你将得到以下输出：

```py
Reading URL: https://apod.nasa.gov/apod/image/1709/BT5643s.jpg
Read 171014 bytes
Attempting to write 171014 bytes to BT5643s.jpg:
The write was successful
```

# 工作原理

这个示例只是使用标准的 Python 文件访问函数将数据写入文件。它通过使用标准的写入数据接口以面向对象的方式来实现，使用了`FileBlobWriter`类的基于文件的实现：

```py
""" Implements the IBlobWriter interface to write the blob to a file """   from interface import implements
from core.i_blob_writer import IBlobWriter

class FileBlobWriter(implements(IBlobWriter)):
    def __init__(self, location):
        self._location = location

    def write(self, filename, contents):
        full_filename = self._location + "/" + filename
        print ("Attempting to write {0} bytes to {1}:".format(len(contents), filename))

        with open(full_filename, 'wb') as outfile:
            outfile.write(contents)

        print("The write was successful")
```

该类传递一个表示文件应该放置的目录的字符串。实际上，数据是在稍后调用`.write()`方法时写入的。这个方法合并了文件名和`directory (_location)`，然后打开/创建文件并写入字节。`with`语句确保文件被关闭。

# 还有更多...

这篇文章可以简单地使用一个包装代码的函数来处理。这个对象将在本章中被重复使用。我们可以使用 Python 的鸭子类型，或者只是一个函数，但是接口的清晰度更容易。说到这一点，以下是这个接口的定义：

```py
""" Defines the interface for writing a blob of data to storage """   from interface import Interface

class IBlobWriter(Interface):
   def write(self, filename, contents):
      pass
```

我们还将看到另一个实现这个接口的方法，让我们可以将文件存储在 S3 中。通过这种类型的实现，通过接口继承，我们可以很容易地替换实现。

# 下载并保存图像到 S3

我们已经看到了如何在第三章中将内容写入 S3，*处理数据*。在这里，我们将把这个过程扩展到 IBlobWriter 的接口实现，以便写入 S3。

# 准备工作

这个配方的代码示例在`04/06_save_image_in_s3.py`文件中。还要确保你已经将 AWS 密钥设置为环境变量，这样 Boto 才能验证脚本。

# 如何做

我们按照以下步骤进行：

1.  运行配方的脚本。它将执行以下操作：

```py
# download the image item = URLUtility(const.ApodEclipseImage())

# store it in S3 S3BlobWriter(bucket_name="scraping-apod").write(item.filename, item.data)
```

1.  在 S3 中检查，我们可以看到存储桶已经创建，并且图像已放置在存储桶中：

![](img/5abb4f94-3072-4d9a-b868-60ac32c2d295.png)S3 中的图像

# 工作原理

以下是`S3BlobWriter`的实现：

```py
class S3BlobWriter(implements(IBlobWriter)):
    def __init__(self, bucket_name, boto_client=None):
        self._bucket_name = bucket_name

        if self._bucket_name is None:
            self.bucket_name = "/"    # caller can specify a boto client (can reuse and save auth times)
  self._boto_client = boto_client

        # or create a boto client if user did not, use secrets from environment variables
  if self._boto_client is None:
            self._boto_client = boto3.client('s3')

    def write(self, filename, contents):
        # create bucket, and put the object
  self._boto_client.create_bucket(Bucket=self._bucket_name, ACL='public-read')
        self._boto_client.put_object(Bucket=self._bucket_name,
  Key=filename,
  Body=contents,
  ACL="public-read")
```

我们之前在写入 S3 的配方中看到了这段代码。这个类将它整齐地包装成一个可重用的接口实现。创建一个实例时，指定存储桶名称。然后每次调用`.write()`都会保存在同一个存储桶中。

# 还有更多...

S3 在存储桶上提供了一个称为启用网站的功能。基本上，如果你设置了这个选项，存储桶中的内容将通过 HTTP 提供。我们可以将许多图像写入这个目录，然后直接从 S3 中提供它们，而不需要实现一个 Web 服务器！

# 为图像生成缩略图

许多时候，在下载图像时，你不想保存完整的图像，而只想保存缩略图。或者你也可以同时保存完整尺寸的图像和缩略图。在 Python 中，使用 Pillow 库可以很容易地创建缩略图。Pillow 是 Python 图像库的一个分支，包含许多有用的图像处理函数。你可以在[Pillow 官网](https://python-pillow.org)找到更多关于 Pillow 的信息。在这个配方中，我们使用 Pillow 来创建图像缩略图。

# 准备工作

这个配方的脚本是`04/07_create_image_thumbnail.py`。它使用了 Pillow 库，所以确保你已经用 pip 或其他包管理工具将 Pillow 安装到你的环境中。

```py
pip install pillow
```

# 如何做

以下是如何进行配方：

运行配方的脚本。它将执行以下代码：

```py
from os.path import expanduser
import const
from core.file_blob_writer import FileBlobWriter
from core.image_thumbnail_generator import ImageThumbnailGenerator
from util.urls import URLUtility

# download the image and get the bytes img_data = URLUtility(const.ApodEclipseImage()).data

# we will store this in our home folder fw = FileBlobWriter(expanduser("~"))

# Create a thumbnail generator and scale the image tg = ImageThumbnailGenerator(img_data).scale(200, 200)

# write the image to a file fw.write("eclipse_thumbnail.png", tg.bytes)
```

结果将是一个名为`eclipse_thumbnail.png`的文件写入你的主目录。

![](img/bc8c1992-366f-43c9-bcb4-281c5644df69.png)我们创建的缩略图

Pillow 保持宽度和高度的比例一致。

# 工作原理

`ImageThumbnailGenerator`类封装了对 Pillow 的调用，为创建图像缩略图提供了一个非常简单的 API：

```py
import io
from PIL import Image

class ImageThumbnailGenerator():
    def __init__(self, bytes):
        # Create a pillow image with the data provided
  self._image = Image.open(io.BytesIO(bytes))

    def scale(self, width, height):
        # call the thumbnail method to create the thumbnail
  self._image.thumbnail((width, height))
        return self    @property
  def bytes(self):
        # returns the bytes of the pillow image   # save the image to an in memory objects  bytesio = io.BytesIO()
        self._image.save(bytesio, format="png")

```

```py
        # set the position on the stream to 0 and return the underlying data
  bytesio.seek(0)
        return bytesio.getvalue()

```

构造函数传递图像数据并从该数据创建 Pillow 图像对象。通过调用`.thumbnail()`创建缩略图，参数是表示缩略图所需大小的元组。这将调整现有图像的大小，并且 Pillow 会保留纵横比。它将确定图像的较长边并将其缩放到元组中表示该轴的值。此图像的高度大于宽度，因此缩略图的高度为 200 像素，并且宽度相应地缩放（在本例中为 160 像素）。

# 对网站进行截图

一个常见的爬取任务是对网站进行截图。在 Python 中，我们可以使用 selenium 和 webdriver 来创建缩略图。

# 准备就绪

此示例的脚本是`04/08_create_website_screenshot.py`。还要确保您的路径中有 selenium，并且已安装 Python 库。

# 操作步骤

运行该示例的脚本。脚本中的代码如下：

```py
from core.website_screenshot_generator import WebsiteScreenshotGenerator
from core.file_blob_writer import FileBlobWriter
from os.path import expanduser

# get the screenshot image_bytes = WebsiteScreenshotGenerator().capture("http://espn.go.com", 500, 500).image_bytes

# save it to a file FileBlobWriter(expanduser("~")).write("website_screenshot.png", image_bytes)
```

创建一个`WebsiteScreenshotGenerator`对象，然后调用其 capture 方法，传递要捕获的网站的 URL 和图像的所需宽度（以像素为单位）。

这将创建一个 Pillow 图像，可以使用`.image`属性访问，并且可以直接使用`.image_bytes`访问图像的字节。此脚本获取这些字节并将它们写入到您的主目录中的`website_screenshot.png`文件中。

您将从此脚本中看到以下输出：

```py
Connected to pydev debugger (build 162.1967.10)
Capturing website screenshot of: http://espn.go.com
Got a screenshot with the following dimensions: (500, 7416)
Cropped the image to: 500 500
Attempting to write 217054 bytes to website_screenshot.png:
The write was successful
```

我们的结果图像如下（图像的内容会有所不同）：

![](img/b9c8c756-e789-43ae-a20b-d90e7b146181.png)网页截图

# 工作原理

以下是`WebsiteScreenshotGenerator`类的代码：

```py
class WebsiteScreenshotGenerator():
    def __init__(self):
        self._screenshot = None   def capture(self, url, width, height, crop=True):
        print ("Capturing website screenshot of: " + url)
        driver = webdriver.PhantomJS()

        if width and height:
            driver.set_window_size(width, height)

        # go and get the content at the url
  driver.get(url)

        # get the screenshot and make it into a Pillow Image
  self._screenshot = Image.open(io.BytesIO(driver.get_screenshot_as_png()))
        print("Got a screenshot with the following dimensions: {0}".format(self._screenshot.size))

        if crop:
            # crop the image
  self._screenshot = self._screenshot.crop((0,0, width, height))
            print("Cropped the image to: {0} {1}".format(width, height))

        return self    @property
  def image(self):
        return self._screenshot

    @property
  def image_bytes(self):
        bytesio = io.BytesIO()
        self._screenshot.save(bytesio, "PNG")
        bytesio.seek(0)
        return bytesio.getvalue()
```

调用`driver.get_screenshot_as_png()`完成了大部分工作。它将页面呈现为 PNG 格式的图像并返回图像的字节。然后将这些数据转换为 Pillow 图像对象。

请注意输出中来自 webdriver 的图像高度为 7416 像素，而不是我们指定的 500 像素。PhantomJS 渲染器将尝试处理无限滚动的网站，并且通常不会将截图限制在窗口给定的高度上。

要实际使截图达到指定的高度，请将裁剪参数设置为`True`（默认值）。然后，此代码将使用 Pillow Image 的裁剪方法设置所需的高度。如果使用`crop=False`运行此代码，则结果将是高度为 7416 像素的图像。

# 使用外部服务对网站进行截图

前一个示例使用了 selenium、webdriver 和 PhantomJS 来创建截图。这显然需要安装这些软件包。如果您不想安装这些软件包，但仍想制作网站截图，则可以使用许多可以截图的网络服务之一。在此示例中，我们将使用[www.screenshotapi.io](http://www.screenshotapi.io)上的服务来创建截图。

# 准备就绪

首先，前往`www.screenshotapi.io`注册一个免费账户：

![](img/ec4f1644-7736-4e42-ad3f-6f1b12bf1cc0.png)免费账户注册的截图

创建账户后，继续获取 API 密钥。这将需要用于对其服务进行身份验证：

![](img/4834f589-e457-4f33-aac9-c809451b33c5.png)API 密钥

# 操作步骤

此示例的脚本是`04/09_screenshotapi.py`。运行此脚本将生成一个截图。以下是代码，结构与前一个示例非常相似：

```py
from core.website_screenshot_with_screenshotapi import WebsiteScreenshotGenerator
from core.file_blob_writer import FileBlobWriter
from os.path import expanduser

# get the screenshot image_bytes = WebsiteScreenshotGenerator("bd17a1e1-db43-4686-9f9b-b72b67a5535e")\
    .capture("http://espn.go.com", 500, 500).image_bytes

# save it to a file FileBlobWriter(expanduser("~")).write("website_screenshot.png", image_bytes)
```

与前一个示例的功能区别在于，我们使用了不同的`WebsiteScreenshotGenerator`实现。这个来自`core.website_screenshot_with_screenshotapi`模块。

运行时，以下内容将输出到控制台：

```py
Sending request: http://espn.go.com
{"status":"ready","key":"2e9a40b86c95f50ad3f70613798828a8","apiCreditsCost":1}
The image key is: 2e9a40b86c95f50ad3f70613798828a8
Trying to retrieve: https://api.screenshotapi.io/retrieve
Downloading image: https://screenshotapi.s3.amazonaws.com/captures/2e9a40b86c95f50ad3f70613798828a8.png
Saving screenshot to: downloaded_screenshot.png2e9a40b86c95f50ad3f70613798828a8
Cropped the image to: 500 500
Attempting to write 209197 bytes to website_screenshot.png:
The write was successful
```

并给我们以下图像：

![](img/6e8e5801-80c3-4d34-a228-f6633af08c75.png)`screenshotapi.io`的网站截图

# 它是如何工作的

以下是此`WebsiteScreenshotGenerator`的代码：

```py
class WebsiteScreenshotGenerator:
    def __init__(self, apikey):
        self._screenshot = None
  self._apikey = apikey

    def capture(self, url, width, height, crop=True):
        key = self.beginCapture(url, "{0}x{1}".format(width, height), "true", "firefox", "true")

        print("The image key is: " + key)

        timeout = 30
  tCounter = 0
  tCountIncr = 3    while True:
            result = self.tryRetrieve(key)
            if result["success"]:
                print("Saving screenshot to: downloaded_screenshot.png" + key)

                bytes=result["bytes"]
                self._screenshot = Image.open(io.BytesIO(bytes))

                if crop:
                    # crop the image
  self._screenshot = self._screenshot.crop((0, 0, width, height))
                    print("Cropped the image to: {0} {1}".format(width, height))
                break    tCounter += tCountIncr
            print("Screenshot not yet ready.. waiting for: " + str(tCountIncr) + " seconds.")
            time.sleep(tCountIncr)
            if tCounter > timeout:
                print("Timed out while downloading: " + key)
                break
 return self    def beginCapture(self, url, viewport, fullpage, webdriver, javascript):
        serverUrl = "https://api.screenshotapi.io/capture"
  print('Sending request: ' + url)
        headers = {'apikey': self._apikey}
        params = {'url': urllib.parse.unquote(url).encode('utf8'), 'viewport': viewport, 'fullpage': fullpage,
  'webdriver': webdriver, 'javascript': javascript}
        result = requests.post(serverUrl, data=params, headers=headers)
        print(result.text)
        json_results = json.loads(result.text)
        return json_results['key']

    def tryRetrieve(self, key):
        url = 'https://api.screenshotapi.io/retrieve'
  headers = {'apikey': self._apikey}
        params = {'key': key}
        print('Trying to retrieve: ' + url)
        result = requests.get(url, params=params, headers=headers)

        json_results = json.loads(result.text)
        if json_results["status"] == "ready":
            print('Downloading image: ' + json_results["imageUrl"])
            image_result = requests.get(json_results["imageUrl"])
            return {'success': True, 'bytes': image_result.content}
        else:
            return {'success': False}

    @property
  def image(self):
        return self._screenshot

    @property
  def image_bytes(self):
        bytesio = io.BytesIO()
        self._screenshot.save(bytesio, "PNG")
        bytesio.seek(0)
        return bytesio.getvalue()
```

`screenshotapi.io` API 是一个 REST API。有两个不同的端点：

+   [`api.screenshotapi.io/capture`](https://api.screenshotapi.io/capture)

+   [`api.screenshotapi.io/retrieve`](https://api.screenshotapi.io/retrieve)

首先调用第一个端点，并将 URL 和其他参数传递给其服务。成功执行后，此 API 将返回一个密钥，可用于在另一个端点上检索图像。截图是异步执行的，我们需要不断调用使用从捕获端点返回的密钥的“检索”API。当截图完成时，此端点将返回`ready`状态值。代码简单地循环，直到设置为此状态，发生错误或代码超时。

当快照可用时，API 会在“检索”响应中返回图像的 URL。然后，代码会检索此图像，并从接收到的数据构造一个 Pillow 图像对象。

# 还有更多...

`screenshotapi.io` API 有许多有用的参数。其中几个允许您调整要使用的浏览器引擎（Firefox、Chrome 或 PhantomJS）、设备仿真以及是否在网页中执行 JavaScript。有关这些选项和 API 的更多详细信息，请访问[`docs.screenshotapi.io/rest-api/`](http://docs.screenshotapi.io/rest-api)。

# 使用 pytesseract 对图像执行 OCR

可以使用 pytesseract 库从图像中提取文本。在本示例中，我们将使用 pytesseract 从图像中提取文本。Tesseract 是由 Google 赞助的开源 OCR 库。源代码在这里可用：[`github.com/tesseract-ocr/tesseract`](https://github.com/tesseract-ocr/tesseract)，您还可以在那里找到有关该库的更多信息。pytesseract 是一个提供了 Python API 的薄包装器，为可执行文件提供了 Python API。

# 准备工作

确保您已安装 pytesseract：

```py
pip install pytesseract
```

您还需要安装 tesseract-ocr。在 Windows 上，有一个可执行安装程序，您可以在此处获取：`https://github.com/tesseract-ocr/tesseract/wiki/4.0-with-LSTM#400-alpha-for-windows`。在 Linux 系统上，您可以使用`apt-get`：

```py
sudo apt-get tesseract-ocr
```

在 Mac 上安装最简单的方法是使用 brew：

```py
brew install tesseract
```

此配方的代码位于`04/10_perform_ocr.py`中。

# 如何做

执行该配方的脚本。脚本非常简单：

```py
import pytesseract as pt
from PIL import Image

img = Image.open("textinimage.png")
text = pt.image_to_string(img)
print(text)
```

将要处理的图像是以下图像：

![](img/1ede956f-a997-4723-8f79-39bdd0d1d30f.png)我们将进行 OCR 的图像

脚本给出以下输出：

```py
This is an image containing text.
And some numbers 123456789

And also special characters: !@#$%"&*(_+
```

# 它是如何工作的

首先将图像加载为 Pillow 图像对象。我们可以直接将此对象传递给 pytesseract 的`image_to_string()`函数。该函数在图像上运行 tesseract 并返回它找到的文本。

# 还有更多...

在爬取应用程序中使用 OCR 的主要目的之一是解决基于文本的验证码。我们不会涉及验证码解决方案，因为它们可能很麻烦，而且也在其他 Packt 标题中有记录。

# 创建视频缩略图

您可能希望为从网站下载的视频创建缩略图。这些可以用于显示多个视频缩略图的页面，并允许您单击它们观看特定视频。

# 准备工作

此示例将使用一个名为 ffmpeg 的工具。ffmpeg 可以在 www.ffmpeg.org 上找到。根据您的操作系统的说明进行下载和安装。

# 如何做

示例脚本位于`04/11_create_video_thumbnail.py`中。它包括以下代码：

```py
import subprocess
video_file = 'BigBuckBunny.mp4' thumbnail_file = 'thumbnail.jpg' subprocess.call(['ffmpeg', '-i', video_file, '-ss', '00:01:03.000', '-vframes', '1', thumbnail_file, "-y"])
```

运行时，您将看到来自 ffmpeg 的输出：

```py
 built with Apple LLVM version 8.1.0 (clang-802.0.42)
 configuration: --prefix=/usr/local/Cellar/ffmpeg/3.3.4 --enable-shared --enable-pthreads --enable-gpl --enable-version3 --enable-hardcoded-tables --enable-avresample --cc=clang --host-cflags= --host-ldflags= --enable-libmp3lame --enable-libx264 --enable-libxvid --enable-opencl --enable-videotoolbox --disable-lzma --enable-vda
 libavutil 55\. 58.100 / 55\. 58.100
 libavcodec 57\. 89.100 / 57\. 89.100
 libavformat 57\. 71.100 / 57\. 71.100
 libavdevice 57\. 6.100 / 57\. 6.100
 libavfilter 6\. 82.100 / 6\. 82.100
 libavresample 3\. 5\. 0 / 3\. 5\. 0
 libswscale 4\. 6.100 / 4\. 6.100
 libswresample 2\. 7.100 / 2\. 7.100
 libpostproc 54\. 5.100 / 54\. 5.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'BigBuckBunny.mp4':
 Metadata:
 major_brand : isom
 minor_version : 512
 compatible_brands: mp41
 creation_time : 1970-01-01T00:00:00.000000Z
 title : Big Buck Bunny
 artist : Blender Foundation
 composer : Blender Foundation
 date : 2008
 encoder : Lavf52.14.0
 Duration: 00:09:56.46, start: 0.000000, bitrate: 867 kb/s
 Stream #0:0(und): Video: h264 (Constrained Baseline) (avc1 / 0x31637661), yuv420p, 320x180 [SAR 1:1 DAR 16:9], 702 kb/s, 24 fps, 24 tbr, 24 tbn, 48 tbc (default)
 Metadata:
 creation_time : 1970-01-01T00:00:00.000000Z
 handler_name : VideoHandler
 Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 159 kb/s (default)
 Metadata:
 creation_time : 1970-01-01T00:00:00.000000Z
 handler_name : SoundHandler
Stream mapping:
 Stream #0:0 -> #0:0 (h264 (native) -> mjpeg (native))
Press [q] to stop, [?] for help
[swscaler @ 0x7fb50b103000] deprecated pixel format used, make sure you did set range correctly
Output #0, image2, to 'thumbnail.jpg':
 Metadata:
 major_brand : isom
 minor_version : 512
 compatible_brands: mp41
 date : 2008
 title : Big Buck Bunny
 artist : Blender Foundation
 composer : Blender Foundation
 encoder : Lavf57.71.100
 Stream #0:0(und): Video: mjpeg, yuvj420p(pc), 320x180 [SAR 1:1 DAR 16:9], q=2-31, 200 kb/s, 24 fps, 24 tbn, 24 tbc (default)
 Metadata:
 creation_time : 1970-01-01T00:00:00.000000Z
 handler_name : VideoHandler
 encoder : Lavc57.89.100 mjpeg
 Side data:
 cpb: bitrate max/min/avg: 0/0/200000 buffer size: 0 vbv_delay: -1
frame= 1 fps=0.0 q=4.0 Lsize=N/A time=00:00:00.04 bitrate=N/A speed=0.151x 
video:8kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
```

输出的 JPG 文件将是以下 JPG 图像：

![](img/c9568a21-3200-4c2b-ad42-12ad7f7b92e4.jpg)从视频创建的缩略图

# 它是如何工作的

`.ffmpeg`文件实际上是一个可执行文件。代码将以下 ffmpeg 命令作为子进程执行：

```py
ffmpeg -i BigBuckBunny.mp4 -ss 00:01:03.000 -frames:v 1 thumbnail.jpg -y
```

输入文件是`BigBuckBunny.mp4`。`-ss`选项告诉我们要检查视频的位置。`-frames:v`表示我们要提取一个帧。最后，我们告诉`ffmpeg`将该帧写入`thumbnail.jpg`（`-y`确认覆盖现有文件）。

# 还有更多...

ffmpeg 是一个非常多才多艺和强大的工具。我曾经创建过一个爬虫，它会爬取并找到媒体（实际上是在网站上播放的商业广告），并将它们存储在数字档案中。然后，爬虫会通过消息队列发送消息，这些消息会被一组服务器接收，它们的唯一工作就是运行 ffmpeg 将视频转换为许多不同的格式、比特率，并创建缩略图。从那时起，更多的消息将被发送给审计员，使用一个前端应用程序来检查内容是否符合广告合同条款。了解 ffmeg，它是一个很棒的工具。

# 将 MP4 视频转换为 MP3

现在让我们来看看如何将 MP4 视频中的音频提取为 MP3 文件。你可能想这样做的原因包括想要携带视频的音频（也许是音乐视频），或者你正在构建一个爬虫/媒体收集系统，它还需要音频与视频分开。

这个任务可以使用`moviepy`库来完成。`moviepy`是一个很棒的库，可以让你对视频进行各种有趣的处理。其中一个功能就是提取音频为 MP3。

# 准备工作

确保你的环境中安装了 moviepy：

```py
pip install moviepy
```

我们还需要安装 ffmpeg，这是我们在上一个示例中使用过的，所以你应该已经满足了这个要求。

# 如何操作

演示将视频转换为 MP3 的代码在`04/12_rip_mp3_from_mp4.py`中。`moviepy`使这个过程变得非常容易。

1.  以下是在上一个示例中下载的 MP4 文件的提取：

```py
import moviepy.editor as mp
clip = mp.VideoFileClip("BigBuckBunny.mp4")
clip.audio.write_audiofile("movie_audio.mp3")
```

1.  当运行时，你会看到输出，比如下面的内容，因为文件正在被提取。这只花了几秒钟：

```py
[MoviePy] Writing audio in movie_audio.mp3
100%|██████████| 17820/17820 [00:16<00:00, 1081.67it/s]
[MoviePy] Done.
```

1.  完成后，你将得到一个 MP3 文件：

```py
# ls -l *.mp3 -rw-r--r--@ 1 michaelheydt  staff  12931074 Sep 27 21:44 movie_audio.mp3
```

# 还有更多...

有关 moviepy 的更多信息，请查看项目网站[`zulko.github.io/moviepy/`](http://zulko.github.io/moviepy/)。
