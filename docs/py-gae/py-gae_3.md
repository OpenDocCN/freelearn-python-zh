# 第三章：存储和处理用户数据

有一些数据需要持久化，并且不太适合 Datastore 或类似存储系统，例如图像和媒体文件；这些通常很大，其大小会影响应用程序成本以及它们应该如何上传、存储和当请求时提供。此外，有时我们需要在服务器端修改这些内容，并且操作可能需要很长时间。

我们将在 Notes 应用程序中添加一些将引发这些问题的功能，我们将看到 App Engine 如何提供我们面对这些问题的所有所需。

在本章中，我们将涵盖以下主题：

+   在我们的应用程序中添加表单以允许用户上传图像

+   将上传的文件返回给客户端

+   使用 Images 服务转换图像

+   使用任务队列执行长时间作业

+   调度任务

+   处理来自我们应用程序的电子邮件消息

# 将文件上传到 Google Cloud Storage

对于一个网络应用程序来说，处理图像文件或 PDF 文档是非常常见的，Notes 也不例外。对于用户来说，除了标题和描述文本外，将图像或文档附加到一个或多个笔记中可能非常有用。

在 Datastore 中存储大量二进制数据将是不高效的，并且相当昂贵，因此我们需要使用不同的、专门的系统：Google Cloud Storage。Cloud Storage 允许我们在称为**桶**的位置存储大文件。一个应用程序可以从多个桶中读取和写入，并且我们可以设置**访问控制列表**（**ACL**）来决定谁可以访问特定的桶以及具有什么权限。每个 App Engine 应用程序都有一个默认的桶与之关联，但我们可以通过开发者控制台创建、管理和浏览任意数量的桶。

## 安装 Cloud Storage 客户端库

为了更好地与 Cloud Storage 交互，我们需要一个外部软件，它不包括在 App Engine 运行时环境中，这就是**GCS 客户端库**。这个 Python 库实现了读取和写入桶内文件的功能，并处理错误和重试。以下这些函数的详细列表：

+   **open()**方法：这允许我们在桶内容上操作类似文件缓冲区

+   **listbucket()**方法：这检索桶的内容

+   **stat()**方法：这为桶中的文件获取元数据

+   **delete()**方法：这将从桶中删除文件

要安装 GCS 客户端库，我们可以使用 pip：

```py
pip install GoogleAppEngineCloudStorageClient -t <app_root>

```

使用`-t`选项指定包的目标目录非常重要，因为这是在生产服务器上安装由 App Engine 不提供的第三方包的唯一方式。当我们部署应用程序时，应用程序根目录中的所有内容都将复制到远程服务器上，包括`cloudstorage`包。

如果我们系统上安装了`svn`仓库，我们还可以克隆**Subversion**（**SVN**）可执行文件并检出源代码的最新版本：

```py
svn checkout http://appengine-gcs- client.googlecode.com/svn/trunk/python gcs-client

```

要检查库是否正常工作，我们可以在命令行中发出以下命令，并验证没有错误打印出来：

```py
python -c"import cloudstorage"

```

### 注意

与 Google Cloud Storage 交互的另一种方式是**Blobstore API**，它是 App Engine 环境捆绑的一部分。Blobstore 是第一个提供大文件便宜且有效存储的 App Engine 服务，尽管云存储更新且更活跃地开发，但它仍然可用。即使我们不在 Blobstore 中存储任何数据，我们也会在本章的后面使用 Blobstore API 与云存储一起使用。

## 添加表单上传图片

我们开始在用于创建笔记的 HTML 表单中添加一个字段，以便用户可以指定要上传的文件。在提交按钮之前，我们插入一个输入标签：

```py
<div class="form-group">
  <label for="uploaded_file">Attached file:</label>
  <input type="file" id="uploaded_file" name="uploaded_file">
</div>
```

我们将把每个用户的所有文件存储在默认存储桶下的一个以用户 ID 命名的文件夹中；如果未更改默认访问控制列表，我们的应用程序是访问该文件的唯一方式，因此我们可以在应用程序级别强制执行安全和隐私。为了从`webapp2`请求对象中访问上传的文件，我们需要重写`MainHandler`类的`post`方法，但首先，我们需要在`main.py`模块的顶部添加以下导入语句：

```py
from google.appengine.api import app_identity
import cloudstorage
import mimetypes
```

我们很快就会看到这些模块的用途；这是将被添加到`MainHandler`类中的代码：

```py
def post(self):
    user = users.get_current_user()
    if user is None:
        self.error(401)

    bucket_name = app_identity.get_default_gcs_bucket_name()
    uploaded_file = self.request.POST.get('uploaded_file')
    file_name = getattr(uploaded_file, 'filename', None)
    file_content = getattr(uploaded_file, 'file', None)
    real_path = ''
    if file_name and file_content:
        content_t = mimetypes.guess_type(file_name)[0]
        real_path = os.path.join('/', bucket_name, user.user_id(),
                                 file_name)

        with cloudstorage.open(real_path, 'w',
                               content_type=content_t) as f:
            f.write(file_content.read())

    self._create_note(user, file_name)

    logout_url = users.create_logout_url(self.request.uri)
    template_context = {
        'user': user.nickname(),
        'logout_url': logout_url,
    }
    self.response.out.write(
        self._render_template('main.html', template_context))
```

我们首先通过调用`app_identity`服务的`get_default_gcs_bucket_name()`方法来检索我们应用程序的默认存储桶名称。然后，我们访问`request`对象以获取`uploaded_file`字段的值。当用户指定要上传的文件时，`self.request.POST.get('uploaded_file')`返回 Python 标准库中`cgi`模块定义的`FileStorage`类的实例。`FieldStorage`对象有两个字段，`filename`和`file`，分别包含上传文件的名称和内容。如果用户没有指定要上传的文件，则`uploaded_file`字段的值变为空字符串。

在处理上传的文件时，我们尝试使用 Python 标准库中的`mimetypes`模块来猜测其类型，然后根据`/<bucket_name>/<user_id>/<filename>`方案构建文件的完整路径。最后一部分涉及到 GCS 客户端库；实际上，它允许我们像在常规文件系统上一样在云存储上打开一个文件进行写入。我们通过在`file_name`对象上调用`read`方法来写入上传文件的正文。最后，我们调用`_create_note`方法，同时传递文件名，这样它就会被存储在`Note`实例内部。

### 注意

如果用户上传的文件与云存储中已存在的文件同名，后者将被新数据覆盖。如果我们想处理这个问题，需要添加一些逻辑，例如重命名新文件或询问用户如何操作。

在重构`_create_note()`方法以接受和处理附加到笔记的文件名之前，我们需要向我们的`Note`模型类添加一个属性来存储附加文件的名称。模型变为以下内容：

```py
class Note(ndb.Model):
    title = ndb.StringProperty()
    content = ndb.TextProperty(required=True)
    date_created = ndb.DateTimeProperty(auto_now_add=True)
    checklist_items = ndb.KeyProperty("CheckListItem",
                                      repeated=True)
    files = ndb.StringProperty(repeated=True)

    @classmethod
    def owner_query(cls, parent_key):
        return cls.query(ancestor=parent_key).order(
            -cls.date_created)
```

即使我们在创建笔记时只支持添加单个文件，我们也存储一个文件名列表，以便我们已经在单个笔记中提供了对多个附件的支持。

在`main.py`模块中，我们将`_create_note()`方法重构如下：

```py
@ndb.transactional
def _create_note(self, user, file_name):
    note = Note(parent=ndb.Key("User", user.nickname()),
                title=self.request.get('title'),
                content=self.request.get('content'))
    note.put()

    item_titles = self.request.get('checklist_items').split(',')
    for item_title in item_titles:
        item = CheckListItem(parent=note.key, title=item_title)
        item.put()
        note.checklist_items.append(item.key)

    if file_name:
       note.files.append(file_name)

   note.put()
```

当`file_name`参数未设置为`None`值时，我们添加文件名并更新`Note`实体。现在我们可以运行代码，在创建笔记时尝试上传文件。我们编写的代码到目前为止只存储上传的文件而没有任何反馈，所以为了检查一切是否正常工作，我们需要在本地开发控制台中使用 Blobstore 查看器。如果我们正在生产服务器上运行应用程序，我们可以使用 Google 开发者控制台上的云存储界面来列出默认存储桶的内容。

### 注意

在编写此内容时，本地开发服务器以与模拟 Blobstore 相同的方式模拟云存储，这就是为什么我们只会在开发控制台中找到 Blobstore 查看器。

## 从云存储中提供文件

由于我们没有为默认存储桶指定访问控制列表，因此未经身份验证或通过笔记应用程序，它只能从开发者控制台访问。只要我们希望将文件保留给执行上传的用户，这就可以了，但我们需要提供一个应用程序的 URL，以便可以检索这些文件。例如，如果用户想要检索名为`example.png`的文件，URL 可以是`/media/example.png`。我们需要为这样的 URL 提供请求处理器，检查当前登录用户是否上传了请求的文件，并相应地提供响应。在`main.py`模块中，我们添加以下类：

```py
class MediaHandler(webapp2.RequestHandler):
    def get(self, file_name):
        user = users.get_current_user()
        bucket_name = app_identity.get_default_gcs_bucket_name()
        content_t = mimetypes.guess_type(file_name)[0]
        real_path = os.path.join('/', bucket_name, user.user_id(),
                                 file_name)

        try:
            with cloudstorage.open(real_path, 'r') as f:
                self.response.headers.add_header('Content-Type',
                                                 content_t)
                self.response.out.write(f.read())
        except cloudstorage.errors.NotFoundError:
            self.abort(404)
```

确定当前登录用户后，我们使用与存储 `<bucket_name>/<user_id>/<filename>` 文件相同的方案构建请求文件的完整路径。如果文件不存在，GCS 客户端库会引发 `NotFoundError` 错误，我们使用请求处理器的 `abort()` 方法提供 **404：未找到** 的礼貌页面。如果文件实际上在云存储中，我们使用 GCS 客户端库提供的常规文件接口打开它进行读取，并在设置正确的 `Content-Type` HTTP 头部后将其内容写入响应体。这样，即使我们知道文件名，我们也无法访问其他用户上传的任何文件，因为我们的用户 ID 将用于确定文件的完整路径。

要使用 `MediaHandler` 类，我们在 `WSGIApplication` 构造函数中添加一个元组：

```py
app = webapp2.WSGIApplication([
    (r'/', MainHandler),
    (r'/media/(?P<file_name>[\w.]{0,256})', MediaHandler),
], debug=True)
```

正则表达式试图匹配以 `/media/` 路径开始的任何 URL，后面跟一个文件名。在匹配时，名为 `file_name` 的正则表达式组被传递到 `MediaHandler` 类的 `get()` 方法作为参数。

最后一步是在主页中为每个附加到笔记的文件添加一个链接，以便用户可以下载它们。我们只需在 `main.html` 模板的清单项迭代之前添加一个 `for` 迭代即可：

```py
{% if note.files %}
<ul>
  {% for file in note.files %}
  <li class="file"><a href="/media/{{ file }}">{{ file }}</a></li>
  {% endfor %}
</ul>
{% endif %}
```

我们最终将 CSS 的 `file` 类添加到 `li` 元素上，以区分文件和清单项；我们将相应的样式添加到 `note.css` 文件中：

```py
div.note > ul > li.file {
    border: 0;
    background: #0070B3;
}

li.file > a {
    color: white;
    text-decoration: none;
}
```

使用这个更新的样式表，文件项的背景颜色与清单项不同，链接文本颜色为白色。

## 通过 Google 的内容分发网络提供文件服务

我们目前通过 `MediaHandler` 请求处理类使用我们的 WSGI 应用程序提供附加到笔记的文件服务，这非常方便，因为我们可以在执行安全检查的同时确保用户只能获取他们之前更新过的文件。尽管如此，这种方法有几个缺点：与常规 Web 服务器相比，应用程序效率较低，我们消耗了诸如内存和带宽之类的资源，这可能会给我们带来大量的费用。

然而，有一个替代方案；如果我们放宽 Notes 应用程序的要求，允许内容公开访问，我们可以从高度优化且无 cookie 的基础设施中低延迟地提供此类文件：**Google 内容分发网络 (CDN)**。如何实现这取决于我们必须提供哪种类型的文件：图片或其他任何 **MIME** 类型。

### 服务器端图片

如果我们处理的是图像文件，我们可以使用图像服务生成一个 URL，该 URL 是公开的但不可猜测的，可以访问存储在云存储中的内容。首先，我们需要计算一个编码密钥，代表我们想要在云存储中提供的服务文件；为此，我们使用 Blobstore API 提供的`create_gs_key()`方法。然后，我们使用图像服务提供的`get_serving_url()`方法为编码密钥生成一个托管 URL。如果我们需要以不同的尺寸提供相同的图像——例如，提供缩略图——就没有必要多次存储相同的文件；实际上，我们可以指定我们想要提供的图像的大小，CDN 将负责处理。我们需要在`main.py`模块的顶部导入所需的包：

```py
from google.appengine.api import images
from google.appengine.ext import blobstore
```

为了方便，我们在`MainHandler`类中添加了一个`_get_urls_for()`方法，我们可以在需要获取云存储中文件的托管 URL 时调用此方法：

```py
def _get_urls_for(self, file_name):
    user = users.get_current_user()
    if user is None:
        return

    bucket_name = app_identity.get_default_gcs_bucket_name()
    path = os.path.join('/', bucket_name, user.user_id(),
                        file_name)
    real_path = '/gs' + path
    key = blobstore.create_gs_key(real_path)
    url = images.get_serving_url(key, size=0)
    thumbnail_url = images.get_serving_url(key, size=150,
                                           crop=True)
    return url, thumbnail_url
```

此方法接受文件名作为参数，并使用略有不同的`/gs/<bucket_name>/<user_id>/<filename>`方案（注意只有在生成编码密钥时需要添加前缀的`/gs`字符串）构建云存储的完整路径。然后，将文件的真正路径传递给`create_gs_key()`函数，该函数生成一个编码密钥，然后我们调用两次`get_serving_url()`方法：一次生成全尺寸图像的 URL，然后生成一个 150 像素大小的裁剪缩略图的 URL。最后，返回这两个 URL。除非我们从图像服务中调用`delete_serving_url()`方法并传递相同的密钥，否则这些 URL 将永久可用。如果我们没有指定`size`参数，CDN 将默认提供优化后的图像版本，其大小更小；通过将`size=0`参数显式传递给`get_serving_url()`函数的第一次调用，将使 CDN 提供原始图像。

我们可以通过提供一个描述附加到笔记的文件的新类型来改进数据模型。在`models.py`模块中，我们添加以下内容：

```py
class NoteFile(ndb.Model):
    name = ndb.StringProperty()
    url = ndb.StringProperty()
    thumbnail_url = ndb.StringProperty()
    full_path = ndb.StringProperty()
```

我们为每个文件存储名称、两个 URL 和云存储中的完整路径。然后，我们从`Note`模型中引用`NoteFile`实例而不是纯文件名：

```py
class Note(ndb.Model):
    title = ndb.StringProperty()
    content = ndb.TextProperty(required=True)
    date_created = ndb.DateTimeProperty(auto_now_add=True)
    checklist_items = ndb.KeyProperty("CheckListItem",
                                      repeated=True)
    files = ndb.KeyProperty("NoteFile",
                            repeated=True)

    @classmethod
    def owner_query(cls, parent_key):
        return cls.query(ancestor=parent_key).order(
            -cls.date_created)
```

为了根据新的模型存储数据，我们重构了`_create_note()`方法：

```py
@ndb.transactional
def _create_note(self, user, file_name, file_path):
    note = Note(parent=ndb.Key("User", user.nickname()),
                title=self.request.get('title'),
                content=self.request.get('content'))
    note.put()

    item_titles = self.request.get('checklist_items').split(',')
    for item_title in item_titles:
        item = CheckListItem(parent=note.key, title=item_title)
        item.put()
        note.checklist_items.append(item.key)

    if file_name and file_path:
        url, thumbnail_url = self._get_urls_for(file_name)

        f = NoteFile(parent=note.key, name=file_name,
                     url=url, thumbnail_url=thumbnail_url,
                     full_path=file_path)
        f.put()
        note.files.append(f.key)

        note.put()
```

我们生成 URL 并创建`NoteFile`实例，将其添加到`Note`实体组中。在`MainHandler`类的`post()`方法中，我们现在按照以下方式调用`_create_note()`方法：

```py
self._create_note(user, file_name, real_path)
```

在 HTML 模板中，我们添加以下代码：

```py
{% if note.files %}
<ul>
  {% for file in note.files %}
  <li class="file">
    <a href="{{ file.get().url }}">
      <img src="img/{{ file.get().thumbnail_url }}">
    </a>
  </li>
  {% endfor %}
</ul>
{% endif %}
```

我们不是显示文件的名称，而是在指向图像全尺寸版本的链接中显示缩略图。

### 提供其他类型的文件服务

我们不能在非图像文件上使用图像服务，因此在这种情况下我们需要遵循不同的策略。公开可访问的存储在云存储中的文件可以通过组合 Google CDN 的 URL 和它们的完整路径来访问。

那么，首先要做的事情是，在 `MainHandler` 类的 `post()` 方法中保存文件时更改默认的 ACL：

```py
with cloudstorage.open(real_path, 'w', content_type=content_t,
                      options={'x-goog-acl': 'public-read'}) as f:
    f.write(file_content.read())
```

GCS 客户端库的 `open()` 方法的 `options` 参数让我们可以指定一个包含要传递给 Cloud Storage 服务的额外头部的字符串字典：在这种情况下，我们将 `x-goog-acl` 头设置为 `public-read` 值，以便文件将公开可用。从现在起，我们可以通过 `http://storage.googleapis.com/<bucket_name>/<file_path>` 类型的 URL 来访问该文件，因此让我们添加代码来组合和存储此类 URL，用于不是图像的文件。

在 `_get_urls_for()` 方法中，我们捕获 `TransformationError` 或 `NotImageError` 类型的错误，假设如果 Images 服务未能处理某个文件，那么该文件不是图像：

```py
def _get_urls_for(self, file_name):
    user = users.get_current_user()
    if user is None:
        return

    bucket_name = app_identity.get_default_gcs_bucket_name()
    path = os.path.join('/', bucket_name, user.user_id(),
                        file_name)
    real_path = '/gs' + path
    key = blobstore.create_gs_key(real_path)
    try:
        url = images.get_serving_url(key, size=0)
        thumbnail_url = images.get_serving_url(key, size=150,
                                               crop=True)
    except images.TransformationError, images.NotImageError:
        url = "http://storage.googleapis.com{}".format(path)
        thumbnail_url = None

    return url, thumbnail_url
```

如果文件类型不受 Images 服务的支持，我们将按照之前所述的方式组合 `url` 参数，并将 `thumbnail_url` 变量设置为 `None` 值。

在 HTML 模板中，对于不是图像的文件，我们将显示文件名而不是缩略图：

```py
{% if note.files %}
<ul>
  {% for file in note.files %}
  {% if file.get().thumbnail_url %}
  <li class="file">
    <a href="{{ file.get().url }}">
      <img src="img/{{ file.get().thumbnail_url }}">
    </a>
  </li>
  {% else %}
  <li class="file">
    <a href="{{ file.get().url }}">{{ file.get().name }}</a>
  </li>
  {% endif %}
  {% endfor %}
</ul>
{% endif %}
```

# 使用 Images 服务转换图像

我们已经使用了 App Engine Images 服务通过 Google 的 CDN 来提供图像服务，但它可以做更多的事情。它可以调整图像大小、旋转、翻转、裁剪图像，并将多个图像组合成一个文件。它可以使用预定义的算法增强图片。它可以转换图像的格式。该服务还可以提供有关图像的信息，例如其格式、宽度、高度以及颜色值的直方图。

### 注意

要在本地开发服务器上使用 Images 服务，我们需要下载并安装 **Python Imaging Library** (**PIL**) 包，或者，作为替代方案，安装 `pillow` 包。

我们可以直接从我们的应用程序传递图像数据到服务，或者指定存储在 Cloud Storage 中的资源。为了了解这是如何工作的，我们在 Notes 应用程序中添加了一个函数，用户可以触发该函数以缩小任何笔记中附加的所有图像，以便在 Cloud Storage 中节省空间。为此，我们在 `main.py` 模块中添加了一个专用的请求处理器，当用户点击 `/shrink` URL 时将被调用：

```py
class ShrinkHandler(webapp2.RequestHandler):
    def _shrink_note(self, note):
        for file_key in note.files:
            file = file_key.get()
            try:
                with cloudstorage.open(file.full_path) as f:
                    image = images.Image(f.read())
                    image.resize(640)
                    new_image_data = image.execute_transforms()

                content_t = images_formats.get(str(image.format))
                with cloudstorage.open(file.full_path, 'w',
                                     content_type=content_t) as f:
                    f.write(new_image_data)

            except images.NotImageError:
                pass

    def get(self):
        user = users.get_current_user()
        if user is None:
            login_url = users.create_login_url(self.request.uri)
            return self.redirect(login_url)

        ancestor_key = ndb.Key("User", user.nickname())
        notes = Note.owner_query(ancestor_key).fetch()

        for note in notes:
            self._shrink_note(note)

        self.response.write('Done.')
```

在`get()`方法中，我们从 Datastore 加载属于当前登录用户的全部笔记，然后对每个笔记调用`_shrink_note()`方法。对于每个附加到笔记的文件，我们检查它是否是图片；如果不是，我们捕获错误并传递给下一个。如果文件实际上是图片，我们使用 GCS 客户端库打开文件并将图像数据传递给`Image`类构造函数。图像对象封装图像数据并提供了一个用于操作和获取封装图像信息的接口。变换不会立即应用；它们被添加到一个队列中，当我们对`Image`实例调用`execute_transforms()`方法时进行处理。在我们的情况下，我们只应用一个变换，即将图像宽度调整为 640 像素。`execute_transforms()`方法返回我们用来覆盖原始文件的变换后的图像数据。在将新的图像数据写入云存储时，我们需要再次指定文件的内容类型：我们从`image`对象的`format`属性中推导出正确的内容类型。这个值是一个整数，必须映射到一个内容类型字符串；我们通过在`main.py`模块顶部添加此字典来完成此操作：

```py
images_formats = {
    '0': 'image/png',
    '1': 'image/jpeg',
    '2': 'image/webp',
    '-1': 'image/bmp',
    '-2': 'image/gif',
    '-3': 'image/ico',
    '-4': 'image/tiff',
}
```

我们将`image.format`值转换为字符串并访问正确的字符串，将其传递给 GCS 客户端库的`open()`方法。

我们在`main.py`模块中添加了`/shrink` URL 的映射：

```py
app = webapp2.WSGIApplication([
    (r'/', MainHandler),
    (r'/media/(?P<file_name>[\w.]{0,256})', MediaHandler),
    (r'/shrink', ShrinkHandler),
], debug=True)
```

为了让用户访问此功能，我们在主页上添加了一个超链接。我们借此机会为我们的应用程序提供一个主菜单，如下修改`main.html`模板：

```py
<h1>Welcome to Notes!</h1>

<ul class="menu">
  <li>Hello, <b>{{ user }}</b></li>
  <li><a href="{{ logout_url }}">Logout</a></li>
  <li><a href="/shrink">Shrink images</a></li>
</ul>

<form action="" method="post" enctype="multipart/form-data">
```

为了使菜单水平布局，我们在`notes.css`文件中添加了以下行：

```py
ul.menu > li {
    display: inline;
    padding: 5px;
    border-left: 1px solid;
}

ul.menu > li > a {
    text-decoration: none;
}
```

用户现在可以通过点击主页菜单中的相应操作来缩小其笔记中附加的图片所占的空间。

# 使用任务队列处理长时间作业

App Engine 提供了一种称为**请求计时器**的机制，以确保客户端请求有一个有限的生命周期，避免无限循环并防止应用程序过度使用资源。特别是，当请求完成超过 60 秒时，请求计时器会引发一个`DeadlineExceededError`错误。如果我们的应用程序提供涉及复杂查询、I/O 操作或图像处理的功能，我们必须考虑这一点。上一段中的`ShrinkHandler`类就是这样一种情况：要加载的笔记数量和要处理的附加图片可能足够多，使得请求持续超过 60 秒。在这种情况下，我们可以使用**任务队列**，这是 App Engine 提供的一项服务，允许我们在请求/响应周期之外执行操作，具有更宽的时间限制，即 10 分钟。

有两种类型的任务队列：**推送队列**，用于由 App Engine 基础设施自动处理的任务，以及**拉取队列**，允许开发者使用另一个 App Engine 应用程序或从另一个基础设施外部构建自己的任务消费策略。我们将使用推送队列，以便我们有来自 App Engine 的现成解决方案，无需担心外部组件的设置和可伸缩性。

我们将在任务队列内部运行缩放图像功能，为此，我们需要重构 `ShrinkHandler` 类：在 `get()` 方法中，我们将启动任务，将查询执行和图像处理移动到 `post()` 方法。`post()` 方法将由任务队列消费者基础设施调用以处理任务。

我们首先需要导入 `taskqueue` 包以使用任务队列 Python API：

```py
from google.appengine.api import taskqueue
```

然后，我们将 `post()` 方法添加到 `ShrinkHandler` 类：

```py
def post(self):
    if not 'X-AppEngine-TaskName' in self.request.headers:
        self.error(403)

    user_email = self.request.get('user_email')
    user = users.User(user_email)

    ancestor_key = ndb.Key("User", user.nickname())
    notes = Note.owner_query(ancestor_key).fetch()

    for note in notes:
        self._shrink_note(note)
```

为了确保我们已经收到任务队列请求，我们检查 `X-AppEngine-TaskName` HTTP 头是否已设置；如果请求来自平台外部，App Engine 会删除这些类型的头，因此我们可以信任客户端。如果此头缺失，我们设置 `HTTP 403: Forbidden` 响应代码。

请求包含一个 `user_email` 参数，该参数包含添加此任务到队列的用户电子邮件地址（我们将在稍后看到这个参数需要在何处设置）；我们通过传递电子邮件地址来实例化一个 `User` 对象，以匹配有效用户，并继续进行图像处理。

`ShrinkHandler` 类的 `get()` 方法需要按照以下方式进行重构：

```py
def get(self):
    user = users.get_current_user()
    if user is None:
        login_url = users.create_login_url(self.request.uri)
        return self.redirect(login_url)

    taskqueue.add(url='/shrink',
                  params={'user_email': user.email()})
    self.response.write('Task successfully added to the queue.')
```

在检查用户是否登录后，我们使用任务队列 API 向队列中添加一个任务。我们将映射到执行作业的处理器的 URL 作为参数传递，以及包含我们想要传递给处理器的参数的字典。在这种情况下，我们将 `post()` 方法中使用的 `user_email` 参数设置为加载有效的 `User` 实例。任务添加到队列后，将立即返回响应，实际缩放操作可能持续长达 10 分钟。

# 使用 Cron 调度任务

我们将缩放操作设计为用户触发的可选功能，但我们可以为每个用户在确定的时间间隔内运行它，以降低云存储的成本。App Engine 支持使用 Cron 服务调度作业的执行；每个应用程序都有一定数量的 Cron 作业可用，这取决于我们的计费计划。Cron 作业具有与任务队列中的任务相同的限制，因此请求可以持续长达 10 分钟。

我们首先准备一个实现作业的请求处理器：

```py
class ShrinkCronJob(ShrinkHandler):
    def post(self):
        self.abort(405, headers=[('Allow', 'GET')])

    def get(self):
        if 'X-AppEngine-Cron' not in self.request.headers:
            self.error(403)

        notes = Note.query().fetch()
        for note in notes:
            self._shrink_note(note)
```

我们从`ShrinkHandler`类派生出`ShrinkCronJob`类以继承`_shrink_note()`方法。Cron 服务执行一个类型为`GET`的 HTTP 请求，因此我们应该重写`post()`方法，简单地返回一个**HTTP 405: 方法不允许**错误，从而避免有人用 HTTP `POST`请求击中我们的处理程序。所有逻辑都在处理程序类的`get()`方法中实现。为了确保处理程序是由 Cron 服务触发的，而不是由外部客户端触发的，我们首先检查请求是否包含通常由 App Engine 移除的`X-AppEngine-Cron`头；如果不是这种情况，我们返回一个**HTTP 403: 未授权**错误。然后，我们加载所有笔记实体并调用每个实体上的`_shrink_note()`方法。

然后，我们将`ShrinkCronJob`处理程序映射到`/shrink_all` URL：

```py
app = webapp2.WSGIApplication([
    (r'/', MainHandler),
    (r'/media/(?P<file_name>[\w.]{0,256})', MediaHandler),
    (r'/shrink', ShrinkHandler),
    (r'/shrink_all', ShrinkCronJob),
], debug=True)
```

Cron 作业以`YAML`文件的形式列在应用程序根目录中，因此我们创建了一个包含以下内容的`cron.yaml`文件：

```py
cron:
- description: shrink images in the GCS
  url: /shrink_all
  schedule: every day 00:00
```

该文件包含一系列带有一些属性的作业定义：对于每个作业，我们必须指定 URL 和`schedule`属性，分别包含映射到实现作业的处理程序的 URL 和作业执行的时时间间隔，即每天午夜。我们还添加了一个可选的`description`属性，其中包含一个字符串来详细说明作业。

每次我们部署应用程序时，都会更新计划中的 Cron 作业列表；我们可以通过访问开发者控制台或本地开发控制台来检查作业的详细信息和工作状态。

# 发送通知电子邮件

对于 Web 应用程序来说，向用户发送通知是非常常见的，电子邮件是一种既便宜又有效的传递渠道。笔记应用程序也可以从通知系统中受益：在本章早期，我们修改了`shrink`图像函数，使其在任务队列中运行。用户会立即收到响应，但实际上作业被放入队列，他们不知道缩放操作何时成功完成。

由于我们可以代表管理员或具有 Google 账户的用户从 App Engine 应用程序发送电子邮件消息，因此当缩放操作完成后，我们立即向用户发送消息。

我们首先在`main.py`模块中导入邮件包：

```py
from google.appengine.api import mail
```

然后，我们将以下代码追加到`ShrinkHandler`类中的`post()`方法末尾：

```py
sender_address = "Notes Team <notes@example.com>"
subject = "Shrink complete!"
body = "We shrunk all the images attached to your notes!"
mail.send_mail(sender_address, user_email, subject, body)
```

我们只需调用`send_mail()`方法，传入发件人地址、目的地地址、电子邮件主题和消息正文。

如果我们在生产服务器上运行应用程序，则`sender_address`参数必须包含 App Engine 上一位管理员的注册地址，否则消息将无法送达。

如果应用程序在本地开发服务器上运行，App Engine 将不会发送真实的电子邮件，而是在控制台上显示一条详细的消息。

# 接收用户数据作为电子邮件消息

对于一个 Web 应用程序来说，一个不太常见但很有用的功能是能够接收来自其用户的电子邮件消息：例如，一个**客户关系管理**（**CRM**）应用程序在收到用户发送到特定地址（例如，`support@example.com`）的电子邮件后，可以打开一个支持工单。

为了展示这在 App Engine 上的工作原理，我们添加了用户通过向笔记应用程序发送电子邮件消息来创建笔记的能力：电子邮件的主题将被用作标题，消息正文用作笔记内容，并且每封电子邮件消息中附加的每个文件都将存储在云存储上，并将其附加到笔记中。

App Engine 应用程序可以在任何`<string>@<appid>.appspotmail.com`格式的地址接收电子邮件消息；然后，消息被转换成对`/_ah/mail/<address>` URL 的 HTTP 请求，在那里一个请求处理器将处理数据。

在我们开始之前，我们需要启用默认禁用的入站电子邮件服务，因此我们将在我们的`app.yaml`文件中添加以下内容：

```py
inbound_services:
- mail
```

然后，我们需要实现一个用于电子邮件消息的处理程序，它从 App Engine 提供的专门`InboundMailHandler`请求处理程序类中派生。我们的子类必须重写接受一个包含`InboundEmailMessage`类实例的参数的`receive()`方法，我们可以使用这个实例来访问我们从收到的电子邮件中获取的所有详细信息。我们将这个新的处理程序添加到`main.py`模块中，但在继续之前，我们需要导入所需的模块和包：

```py
from google.appengine.ext.webapp import mail_handlers
import re
```

然后，我们开始实现我们的`CreateNoteHandler`类；这是代码的第一部分：

```py
class CreateNoteHandler(mail_handlers.InboundMailHandler):
    def receive(self, mail_message):
        email_pattern = re.compile(
            r'([\w\-\.]+@(\w[\w\-]+\.)+[\w\-]+)')
        match = email_pattern.findall(mail_message.sender)
        email_addr = match[0][0] if match else ''

        try:
            user = users.User(email_addr)
            user = self._reload_user(user)
        except users.UserNotFoundError:
            return self.error(403)

        title = mail_message.subject
        content = ''
        for content_t, body in mail_message.bodies('text/plain'):
            content += body.decode()

        attachments = getattr(mail_message, 'attachments', None)

        self._create_note(user, title, content, attachments)
```

代码的第一部分实现了一个简单的安全检查：我们实际上只为来自用户注册相同地址的电子邮件消息创建特定用户的笔记。我们首先使用正则表达式从包含在`mail_message`参数中的`InboundEmailMessage`实例的`sender`字段中提取电子邮件地址。然后，我们实例化一个代表发送该消息的电子邮件地址所有者的`User`对象。如果发送者不对应于已注册用户，App Engine 将引发`UserNotFoundError`错误，我们返回一个`403: Forbidden` HTTP 响应代码，否则我们调用`_reload_user()`方法。

如果用户想要将文件附加到他们的笔记中，笔记应用程序需要知道笔记所有者的用户 ID 来构建在云存储上存储文件时的路径；问题是当我们没有从`users` API 调用`get_current_user()`方法来实例化`User`类时，该实例的`user_id()`方法总是返回`None`值。在撰写本文时，App Engine 没有提供一种干净的方法来确定`User`类实例的用户 ID，因此我们通过以下步骤实现了一个解决方案：

1.  将`User`实例分配给 Datastore 实体的一个字段，该字段称为`UserLoader`实体。

1.  将`UserLoader`实体存储在 Datastore 中。

1.  立即再次加载实体。

这样，我们强制`Users`服务填写所有用户数据；通过访问`UserLoader`实体中包含`User`实例的字段，我们将获得所有用户属性，包括`id`属性。我们在处理器类的实用方法中执行此操作：

```py
def _reload_user(self, user_instance):
    key = UserLoader(user=user_instance).put()
    key.delete(use_datastore=False)
    u_loader = UserLoader.query(
        UserLoader.user == user_instance).get()
    return UserLoader.user
```

要强制从 Datastore 重新加载实体，我们首先需要清除 NDB 缓存，这是通过在传递`use_datastore=False`参数的键上调用`delete()`方法来实现的。然后我们从 Datastore 重新加载实体并返回`user`属性，现在它包含我们所需的所有数据。我们将`UserLoader`模型类添加到我们的`models.py`模块中：

```py
class UserLoader(ndb.Model):
    user = ndb.UserProperty()
```

在`receive()`方法中，我们在重新加载`User`实例后继续从电子邮件消息中提取所需的所有数据；为了提取所有数据，我们需要创建一个笔记：消息主题是一个简单的字符串，我们将用它作为笔记标题。访问正文稍微复杂一些，因为电子邮件消息可能有多个正文，内容类型不同，通常是纯文本或 HTML；在这种情况下，我们只提取纯文本正文并将其用作笔记内容。

在这种情况下，电子邮件消息有附件，并且`mail_message`实例提供了`attachments`属性：我们将其作为参数传递给专门用于创建笔记的方法，即`_create_note()`方法。`_create_note()`方法在事务中运行并封装了创建`Note`实体所需的所有逻辑：

```py
@ndb.transactional
def _create_note(self, user, title, content, attachments):

    note = Note(parent=ndb.Key("User", user.nickname()),
                title=title,
                content=content)
    note.put()

    if attachments:
        bucket_name = app_identity.get_default_gcs_bucket_name()
        for file_name, file_content in attachments:
            content_t = mimetypes.guess_type(file_name)[0]
            real_path = os.path.join('/', bucket_name,
                                     user.user_id(), file_name)

            with cloudstorage.open(real_path, 'w',
                    content_type=content_t,
                    options={'x-goog-acl': 'public-read'}) as f:
                f.write(file_content.decode())

            key = blobstore.create_gs_key('/gs' + real_path)
            try:
                url = images.get_serving_url(key, size=0)
                thumbnail_url = images.get_serving_url(key,
                    size=150, crop=True)
            except images.TransformationError,
                   images.NotImageError:
                url = "http://storage.googleapis.com{}".format(
                    real_path)
                thumbnail_url = None

            f = NoteFile(parent=note.key, name=file_name,
                         url=url, thumbnail_url=thumbnail_url,
                         full_path=real_path)
            f.put()
            note.files.append(f.key)

        note.put()
```

该方法与`MainHandler`类中同名的方法非常相似；主要区别在于我们从电子邮件消息中附加的文件中访问数据的方式。`attachments`参数是一个包含两个元素的元组列表：一个是包含文件名的字符串，另一个是包含消息有效载荷的**包装器**类的实例。我们使用文件名来构建云存储中文件的完整路径，并使用`decode()`方法来访问有效载荷数据并将其存储在文件中。

最后，我们将 URL 映射到处理器：

```py
app = webapp2.WSGIApplication([
    (r'/', MainHandler),
    (r'/media/(?P<file_name>[\w.]{0,256})', MediaHandler),
    (r'/shrink', ShrinkHandler),
    (r'/shrink_all', ShrinkCronJob),
    (r'/_ah/mail/<appid>\.appspotmail\.com', CreateNoteHandler),
], debug=True)
```

在本地开发服务器上测试应用程序时，我们可以使用开发控制台从 Web 界面模拟发送电子邮件；此功能可通过点击左侧栏上的**入站邮件**菜单项获得。

# 摘要

在本章中，我们在 Notes 应用程序中添加了许多功能，现在我们应该能够利用云存储并将其用于存储和从我们的应用程序中提供静态内容。我们看到了 Images API 的实际应用，现在我们应该知道如何处理耗时较长的请求，我们还学习了如何安排重复任务。在最后一部分，我们深入探讨了 Mail API 的功能，并了解了 App Engine 如何提供发送和接收电子邮件消息的现成解决方案。

在下一章中，我们将审视我们应用程序的性能，并探讨我们可以在哪些方面以及如何进行改进，这包括利用我们已使用的组件的高级功能，以及 App Engine 提供的更多服务。
