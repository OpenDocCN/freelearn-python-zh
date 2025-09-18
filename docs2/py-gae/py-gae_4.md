# 第四章：提高应用程序性能

即使我们的笔记应用程序缺少许多细节，到目前为止，我们已经在使用云平台的一些关键组件，因此它可以被认为是一个完整的网络应用程序。这是一个很好的机会停止添加主要功能，并尝试深入了解涉及 Datastore、Memcache 和模块服务的实现细节，以优化应用程序性能。

在阅读本章内容时，我们必须考虑到如何优化在按使用付费的服务（如 App Engine）上运行的网络应用程序，这对于最大化性能和降低成本至关重要。

在本章中，我们将涵盖以下主题：

+   深入了解 Datastore：属性、查询、缓存、索引和管理

+   如何将临时数据存储到 Memcache 中

+   如何在模块服务的帮助下构建我们的应用程序

# Datastore 的高级使用

到目前为止，我们已经对 Datastore 学习了很多，包括如何使用模型类定义实体类型、属性概念以及如何进行简单查询。

我们可以使用 NDB Python API 做很多事情来优化应用程序，正如我们很快就会看到的。

## 更多关于属性的内容——使用 StructuredProperty 安排复合数据

在我们的笔记应用程序中，我们定义了 `CheckListItem` 模型类来表示可勾选项，然后我们在 `Note` 模型中添加了一个名为 `checklist_items` 的属性，该属性引用了该种实体的列表。这通常被称为笔记和清单项之间的一对多关系，这是构建应用程序数据的一种常见方式。然而，按照这种策略，每次我们向笔记添加一个项目时，我们都必须在 Datastore 中创建并存储一个新的实体。这根本不是什么坏习惯，但我们必须考虑到我们根据所进行的操作数量来支付 Datastore 的使用费用；因此，如果我们有大量数据，保持低写入操作率可以潜在地节省很多钱。

Python NDB API 提供了一种名为 `StructuredProperty` 的属性类型，我们可以使用它将一种模型包含在另一种模型中；而不是在 `Note` 模型的 `KeyProperty` 类型的属性中引用 `CheckListItem` 模型，我们将其存储在 `StructuredProperty` 类型的属性中。在我们的 `models.py` 模块中，我们按照以下方式更改 Note 模型：

```py
class Note(ndb.Model):
    title = ndb.StringProperty()
    content = ndb.TextProperty(required=True)
    date_created = ndb.DateTimeProperty(auto_now_add=True)
    checklist_items = ndb.StructuredProperty(CheckListItem,
                                             repeated=True)
    files = ndb.KeyProperty("NoteFile", repeated=True)
```

在 `main.py` 模块中，当我们创建新笔记时需要调整代码以存储清单项，因此我们这样重构了 `create_note` 方法：

```py
@ndb.transactional
def _create_note(self, user, file_name, file_path):
    note = Note(parent=ndb.Key("User", user.nickname()),
                title=self.request.get('title'),
                content=self.request.get('content'))

    item_titles = self.request.get('checklist_items').split(',')
    for item_title in item_titles:
        if not item_title:
            continue
        item = CheckListItem(title=item_title)
        note.checklist_items.append(item)
    note.put()

    if file_name and file_path:
        url, thumbnail_url = self._get_urls_for(file_name)

        f = NoteFile(parent=note.key, name=file_name,
                     url=url, thumbnail_url=thumbnail_url,
                     full_path=file_path)
        f.put()
        note.files.append(f.key)
        note.put()
```

首先，我们将对 `note.put()` 方法的调用移至笔记创建下方；我们不需要在 `CheckListItem` 构造函数的 `parent` 参数中提供一个有效的键，因此我们可以在方法末尾持久化 `Note` 实例。然后，我们为每个想要添加到笔记中的项目实例化一个 `CheckListItem` 对象，就像之前一样，但实际上并不在 Datastore 中创建任何实体；这些对象将由 NDB API 在 `Note` 实体内部透明地序列化。

我们还需要调整 HTML 模板，因为笔记实体中的 `checklist_items` 属性不再包含键的列表；它包含 `CheckListItem` 对象的列表。在 `main.html` 文件中，我们相应地更改代码，删除了 `get()` 方法的调用：

```py
{% if note.checklist_items %}
<ul>
  {% for item in note.checklist_items %}
  <li class="{%if item.checked%}checked{%endif%}">
    {{item.title}}
  </li>
  {% endfor %}
</ul>
{% endif %}
```

为了看到与结构化属性一起工作是多么容易，我们在应用程序中添加了一个非常小的功能：一个切换清单中项目选中状态的链接。为了切换项目的状态，我们必须向请求处理程序提供包含项目的笔记的键以及 `checklist_items` 列表中项目的索引，因此我们构建了一个具有方案 `/toggle/<note_key>/<item_index>` 的 URL。在 `main.html` 文件中，我们添加了以下内容：

```py
{% if note.checklist_items %}
<ul>
  {% for item in note.checklist_items %}
  <li class="{%if item.checked%}checked{%endif%}">
    <a href="/toggle/{{note.key.urlsafe()}}/{{ loop.index }}">
      {{item.title}}
    </a>
  </li>
  {% endfor %}
</ul>
{% endif %}
```

`Key` 类的实例有一个 `urlsafe()` 方法，可以将键对象序列化为一个字符串，该字符串可以安全地用作 URL 的一部分。为了在循环中检索当前索引，我们使用 Jinja2 提供的 `loop.index` 表达式。我们还可以向 `notes.css` 文件中添加一个简单的 CSS 规则，使项目看起来更好一些：

```py
div.note > ul > li > a {
    text-decoration: none;
    color: inherit;
}
```

为了实现切换逻辑，我们在 `main.py` 模块中添加了 `ToggleHandler` 类：

```py
class ToggleHandler(webapp2.RequestHandler):
    def get(self, note_key, item_index):
        item_index = int(item_index) - 1
        note = ndb.Key(urlsafe=note_key).get()
        item = note.checklist_items[item_index]
        item.checked = not item.checked
        note.put()
        self.redirect('/')
```

我们将项目索引标准化，使其从零开始，然后使用其键从 Datastore 加载一个笔记实体。我们通过将使用 `urlsafe()` 方法生成的字符串传递给构造函数并使用 `urlsafe` 关键字参数来实例化一个 `Key` 对象，然后使用 `get()` 方法检索实体。在切换请求索引处的项目状态后，我们通过调用 `put()` 方法更新 Datastore 中的笔记内容。最后，我们将用户重定向到应用程序的主页。

最终，我们将 URL 映射添加到应用程序构造函数中，使用正则表达式匹配我们的 URL 方案，`/toggle/<note_key>/<item_index>`：

```py
app = webapp2.WSGIApplication([
    (r'/', MainHandler),
    (r'/media/(?P<file_name>[\w.]{0,256})', MediaHandler),
    (r'/shrink', ShrinkHandler),
    (r'/shrink_all', ShrinkCronJob),
    (r'/toggle/(?P<note_key>[\w\-]+)/(?P<item_index>\d+)', ToggleHandler),
    (r'/_ah/mail/create@book-123456\.appspotmail\.com', CreateNoteHandler),
], debug=True)
```

与结构化属性一起工作很简单；我们只需像访问实际实体一样访问 `checklist_items` 属性中包含的对象的属性和字段。

这种方法的唯一缺点是`CheckListItem`实体实际上并没有存储在 Datastore 中；它们没有键，我们不能独立于它们所属的`Note`实体来加载它们，但这对我们的用例来说完全没问题。我们不是加载我们想要更新的`CheckListItem`实体，而是加载`Note`实体，并使用索引来访问项目。作为交换，在笔记创建期间，我们为笔记保存一个`put()`方法调用，并为清单中的每个项目保存一个`put()`方法调用；在检索笔记时，我们为清单中的每个项目保存一个`get()`方法调用。不言而喻，这种优化可以有利于应用程序的成本。

## 更多关于查询的内容 - 使用投影来节省空间，并使用映射来优化迭代

查询在应用程序中用于搜索 Datastore 中的实体，这些实体符合我们可以通过过滤器定义的搜索标准。我们已经使用 Datastore 查询通过过滤器检索实体；例如，每次我们执行祖先查询时，我们实际上是在过滤掉那些具有与提供给 NDB API `query()`函数不同的父实体的这些实体。

尽管如此，我们还可以使用查询过滤器做更多的事情，在本节中，我们将详细探讨 NDB API 提供的两个可以用来优化应用程序性能的功能：投影查询和映射。

### 投影查询

当我们使用查询检索一个实体时，我们会得到该实体的所有属性和数据，这是预期的；但有时，在检索一个实体之后，我们只使用其数据的一个小子集。例如，看看我们`ShrinkHandler`类中的`post()`方法；我们执行一个祖先查询来检索属于当前登录用户的笔记，然后对每个笔记调用`_shrink_note()`方法。`_shrink_note()`方法只访问笔记实体的`files`属性，所以即使我们只需要它的一小部分，我们仍然在内存中保留并传递一个相当大的对象。

使用 NDB API，我们可以向`fetch()`方法传递一个投影参数，该参数包含我们想要为检索到的实体设置的属性列表。例如，在`ShrinkHandler`类的`post()`方法中，我们可以这样修改代码：

```py
notes = Note.owner_query(ancestor_key).fetch(
    projection=[Note.files])
```

这是一种所谓的投影查询，以这种方式检索到的实体将只设置`files`属性。这种检索方式效率更高，因为它检索和序列化的数据更少，实体在内存中占用的空间也更少。如果我们尝试访问这些实体上的任何其他属性而不是`files`，将会抛出`UnprojectedPropertyError`错误。

投影有一些我们必须注意的限制。首先，正如我们所预期的，使用投影获取的实体不能在 Datastore 上保存，因为它们只是部分填充的。另一个限制是关于索引的；实际上，我们只能在投影中指定索引属性，这使得无法投影具有未索引类型（如 `TextProperty` 类型）的属性。

### 映射

有时，我们需要在查询返回的一组实体上调用相同的函数。例如，在 `ShrinkHandler` 类的 `post()` 方法中，我们需要对当前用户的每个笔记实体调用 `_shrink_note()` 方法：

```py
ancestor_key = ndb.Key("User", user.nickname())
notes = Note.owner_query(ancestor_key).fetch()
for note in notes:
    self._shrink_note(note)
```

我们首先获取与笔记列表中查询匹配的所有实体，然后对列表中的每个项目调用相同的函数。我们可以通过用 NDB API 提供的 `map()` 方法的一个单独调用替换 `for` 迭代来重写那段代码：

```py
ancestor_key = ndb.Key("User", user.nickname())
Note.owner_query(ancestor_key).map(self._shrink_note)
```

我们通过传递我们希望在查询的每个结果上调用的回调函数来调用 `map()` 方法；回调函数接收一个类型为 Note 的实体对象作为其唯一参数，除非我们使用 `keys_only=True` 参数调用 `map()` 方法。在这种情况下，当被调用时，回调将接收一个 `Key` 实例。

由于 `map()` 方法接受标准查询选项集（这就是为什么我们可以传递 `keys_only` 参数），我们也可以对投影查询执行映射：

```py
Note.owner_query(ancestor_key).map(
    self._shrink_note, projection=[Note.files])
```

除了投影之外，这个版本的代码稍微更有效率，因为 Datastore 可以在加载实体时应用一些并发性，并且结果是以批量的形式检索的，而不是在内存中检索整个数据集。如果我们想在回调函数内部获取有关当前批次的详细信息，我们需要在调用 `map()` 方法时传递 `pass_batch_into_callback=True` 参数。在这种情况下，回调将接收三个参数：一个由 App Engine 提供的 `Batch` 对象，它封装了大量有关当前批次的详细信息，当前批次中当前项目的索引，以及从 Datastore 获取的实体对象（或如果使用了 `keys_only` 参数，则为实体键）。

## NDB 异步操作

正如我们所预期的，Datastore 是考虑应用程序性能时的一个关键组件；调整查询和使用正确的惯用表达式可以显著提高效率并降低成本，但还有更多。多亏了 NDB API，我们可以在与其他作业并行执行 Datastore 操作或同时执行多个 Datastore 操作来加速我们的应用程序。

NDB API 提供的几个函数都有一个 `_async` 对应版本，它们接受完全相同的参数，例如 `put` 和 `put_async` 函数。每个异步函数都返回一个 **future**，这是一个表示已启动但可能尚未完成的操作的对象。我们可以通过调用 `get_result()` 方法从 future 本身获取异步操作的结果。

在我们的笔记应用程序中，我们可以在`MainHandler`类的`_render_template()`方法中使用异步操作：

```py
def _render_template(self, template_name, context=None):
    if context is None:
        context = {}

    user = users.get_current_user()
    ancestor_key = ndb.Key("User", user.nickname())
    qry = Note.owner_query(ancestor_key)
    context['notes'] = qry.fetch()

    template = jinja_env.get_template(template_name)
    return template.render(context)
```

目前，我们在加载模板之前等待获取笔记，但我们可以同时在数据存储工作时加载模板：

```py
def _render_template(self, template_name, context=None):
    if context is None:
        context = {}

    user = users.get_current_user()
    ancestor_key = ndb.Key("User", user.nickname())
    qry = Note.owner_query(ancestor_key)
    future = qry.fetch_async()

    template = jinja_env.get_template(template_name)

    context['notes'] = future.get_result()
    return template.render(context)
```

以这种方式，应用程序在获取数据时不会阻塞，因为`fetch_async()`方法会立即返回；我们随后继续加载模板，同时数据存储正在工作。当需要填充上下文变量时，我们在未来对象上调用`get_result()`方法。在这个时候，要么结果可用，我们继续进行渲染操作，要么`get_result()`方法会阻塞，等待数据存储准备好。在这两种情况下，我们都成功地并行执行了两个任务，从而提高了性能。

使用 NDB API，我们还可以实现称为**任务函数**的异步任务，在执行其他工作的同时返回一个未来。例如，在本章的早期，我们在`ShrinkHandler`类中使用`map()`方法对从数据存储检索的一组实体调用相同的函数。我们知道这段代码比显式`for`循环版本稍微高效一些，但实际上并没有快多少；回调函数在同步的`get()`方法上阻塞，因此映射的每一步都要等待前一步完成。

如果我们将回调函数转换为任务函数，App Engine 可以并行运行映射，从而显著提高应用程序性能。由于 NDB API，编写任务函数很简单；例如，`ShrinkHandler`类的`_shrink_note()`方法只需两行代码就可以转换为任务函数，如下所示：

```py
@ndb.tasklet
def _shrink_note(self, note):
    for file_key in note.files:
        file = yield file_key.get_async()
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
```

我们首先将`ndb.tasklet`装饰器应用于我们想要转换为任务函数的函数；装饰器提供了所有逻辑来支持带有`get_result()`方法的前置机制。然后我们使用`yield`语句告诉 App Engine，我们将在执行的那个点暂停，等待`get_async()`方法的结果。在我们暂停期间，`map()`方法可以执行另一个具有不同实体的任务函数，而不是等待我们完成。

## 缓存

缓存是像 App Engine 这样的系统中的关键组件，因为它影响应用程序性能和数据存储往返，从而影响应用程序成本。NDB API 自动为我们管理缓存，并提供了一套配置缓存系统的工具。如果我们想利用这些功能，理解 NDB 缓存的工作方式非常重要。

NDB 使用两个缓存级别：运行在进程内存中的 **in-context** 缓存和连接到 App Engine Memcache 服务的网关。in-context 缓存仅在单个 HTTP 请求的持续时间内存储数据，并且仅对处理请求的代码本地，因此它非常快。当我们使用 NDB 函数在 Datastore 上写入数据时，它首先填充 in-context 缓存。对称地，当我们使用 NDB 函数从 Datastore 获取实体时，它首先在 in-context 缓存中搜索，甚至在最佳情况下都不需要访问 Datastore。

Memcache 的速度比 in-context cache 慢，但仍然比 Datastore 快得多。默认情况下，所有在事务外执行的 Datastore 操作都会在 Memcache 上进行缓存，并且 App Engine 确保数据位于同一服务器上以最大化性能。当 NDB 在事务内操作时忽略 Memcache，但在提交事务时，它会尝试从 Memcache 中删除所有涉及的实体，我们必须考虑到其中一些删除可能会失败。

这两个缓存都由一个所谓的上下文管理，该上下文由 App Engine 提供的 `Context` 类的实例表示。每个传入的 HTTP 请求和每个事务都在一个新的上下文中执行，我们可以使用 NDB API 提供的 `get_context()` 方法访问当前上下文。

在我们的 Notes 应用程序中，我们已经经历过这些罕见的情况之一，即 NDB 自动缓存实际上是一个问题；在 `CreateNoteHandler` 类的 `_reload_user()` 方法中，我们必须强制从 Datastore 重新加载 `UserLoader` 实体，作为填充 `User` 对象的解决方案。在 `UserLoader` 实体的 `put()` 方法和 `get()` 方法之间，我们编写了这个指令来从除 Datastore 之外的所有位置删除实体：

```py
UserLoader(user=user_instance).put()
key.delete(use_datastore=False)
u_loader = UserLoader.query(
    UserLoader.user == user_instance).get()
```

没有这个指令，NDB 缓存系统就不会从头开始从 Datastore 获取实体，正如我们所需要的。现在我们知道了 NDB 缓存的工作方式，我们可以以等效的方式重写那个方法，从而更明确地管理缓存，使用 `Context` 实例：

```py
ctx = ndb.get_context()
ctx.set_cache_policy(lambda key: key.kind() != 'UserLoader')
UserLoader(user=user_instance).put()
u_loader = UserLoader.query(
    UserLoader.user == user_instance).get()
```

由上下文对象公开的 `set_cache_policy()` 方法接受一个键对象并返回一个布尔结果。当该方法返回 `False` 参数时，由该键标识的实体不会被保存到任何缓存中；在我们的情况下，我们仅在实体为 `UserLoader` 类型时返回 `False` 参数。

## 备份和恢复功能

为了使用 App Engine 为 Datastore 提供的备份和恢复功能，我们首先需要启用 **Datastore Admin**，默认情况下它是禁用的。Datastore Admin 是一个提供一组非常有助于管理任务的 Web 应用程序。在撰写本文时，启用和访问 Datastore Admin 的唯一方法是使用位于 [`appengine.google.com`](https://appengine.google.com) 的旧版 Admin Console。

我们访问我们的项目控制台，然后我们必须执行以下步骤：

1.  点击页面左侧**数据**部分下的**数据存储管理员**菜单。

1.  点击按钮以启用管理员。

1.  选择一个或多个我们想要备份或恢复的实体类型。

要执行完整备份，我们首先必须将我们的应用程序置于只读模式。从控制台，我们需要执行以下步骤：

1.  点击左侧**管理**菜单下的**应用程序设置**。

1.  在页面底部，点击**禁用写入...**按钮，位于**禁用数据存储写入**选项下。

1.  返回到**数据存储管理员**部分，并选择我们想要备份的所有实体类型。

1.  点击**备份实体**按钮。

1.  选择备份的目标位置，并在**blobstore**和**云存储**之间进行选择。指定备份文件的名称。

1.  点击**备份实体**按钮。

1.  备份在后台运行；一旦完成，它就会在**数据存储管理员**中列出。

1.  重新启用我们应用程序的写入权限。

从数据存储管理员界面，我们可以选择一个备份并执行恢复操作。在开始恢复操作后，数据存储管理员会询问我们想要恢复哪些实体类型，然后它将在后台进行。

## 索引

索引是按索引的某些属性和可选的实体祖先顺序排列的数据存储实体的表。每次我们对数据存储进行写入时，索引都会更新以反映它们各自实体的变化；当我们从数据存储读取时，结果通过访问索引来获取。这基本上是为什么从数据存储读取比写入快得多的原因。

我们的 Notes 应用程序执行多个查询，这意味着必须存在某些索引，但我们从未直接管理或创建索引。这是因为两个原因。第一个原因是当我们运行本地开发服务器时，它会扫描我们的源代码，寻找查询并自动生成创建所有所需索引的代码。另一个原因是数据存储为每个类型的每个属性自动生成基本索引，称为预定义索引，这对于简单的查询是有用的。

索引在应用程序根目录的`index.yaml`文件中声明，其语法如下：

```py
- kind: Note
  ancestor: yes
  properties:
  - name: date_created
    direction: desc
  - name: NoteFile
```

这些是需要定义和创建索引的属性，以便我们对属于当前登录用户的 Note 实体进行查询，并按日期逆序排序。当我们部署应用程序时，`index.yaml`文件会被上传，并且 App Engine 开始构建索引。

如果我们的应用程序执行了每一种可能的查询，包括每一种排序组合，那么开发服务器生成的条目将代表一个完整的索引集。这就是为什么在绝大多数情况下，我们不需要声明索引或自定义现有的索引，除非我们有一个非常特殊的案例需要处理。无论如何，为了优化我们的应用程序，我们可以禁用那些我们知道永远不会进行查询的属性上的索引。预定义的索引没有列在`index.yaml`文件中，但我们可以使用`models.py`模块中的属性构造函数来禁用它们。例如，如果我们事先知道我们永远不会直接通过查询搜索`NoteFile`实体，我们可以禁用所有其属性的索引：

```py
class NoteFile(ndb.Model):
    name = ndb.StringProperty(indexed=False)
    url = ndb.StringProperty(indexed=False)
    thumbnail_url = ndb.StringProperty(indexed=False)
    full_path = ndb.StringProperty(indexed=False)
```

通过将`indexed=False`参数传递给构造函数，我们避免了 App Engine 为这些属性创建索引，这样每次我们存储一个`NoteFile`实体时，将会有更少的索引需要更新，从而加快写入操作。`NoteFile`实体仍然可以通过`Note`实体中的`files`属性检索，因为 App Engine 会继续创建预定义的索引来按类型和键检索实体。

# 使用 Memcache

我们已经知道 Memcache 是 App Engine 提供的分布式内存数据缓存。一个典型的用法是将其用作从持久存储（如 Datastore）快速检索数据的缓存，但我们已经知道 NDB API 会为我们做这件事，所以没有必要显式地缓存实体。

存储在 Memcache 中的数据可以随时被清除，因此我们只应该缓存那些即使丢失也不会影响完整性的数据。例如，在我们的笔记应用中，我们可以缓存每个用户全局存储的笔记总数，并在主页上显示这种类型的指标。每次用户访问主页时，我们都可以执行一个查询来计算`Note`实体，但这会很麻烦，可能会抵消我们迄今为止所做的所有优化。更好的策略是在 Memcache 中保持一个计数器，并在应用中创建笔记时增加该计数器；如果 Memcache 数据过期，我们将再次执行计数查询而不会丢失任何数据，并重新开始增加内存中的计数器。

我们实现两个函数来封装 Memcache 操作：一个用于获取计数器的值，另一个用于增加它。我们首先在`utils.py`文件中创建一个新的 Python 模块，该模块包含以下代码：

```py
from google.appengine.api import memcache
from models import Note
def get_note_counter():
    data = memcache.get('note_count')
    if data is None:
        data = Note.query().count()
        memcache.set('note_count', data)

    return data
```

我们首先尝试从 Memcache 中调用`get()`方法访问计数器的值，请求`note_count`键。如果返回值是`None`，我们假设键不在缓存中，然后我们继续查询 Datastore。然后我们将查询的结果存储在 Memcache 中，并返回该值。

我们想在主页上显示计数器，所以我们在`MainHandler`类的`_render_template()`方法中将它添加到模板上下文中：

```py
def _render_template(self, template_name, context=None):
    if context is None:
        context = {}
    user = users.get_current_user()
    ancestor_key = ndb.Key("User", user.nickname())
    qry = Note.owner_query(ancestor_key)
    future = qry.fetch_async()

    template = jinja_env.get_template(template_name)

    context['notes'] = future.get_result()
    context['note_count'] = get_note_counter()

    return template.render(context)
```

在使用函数获取计数器之前，我们需要从 `main` 模块中导入它：

```py
from utils import get_note_counter
```

我们还需要修改 HTML 模板：

```py
<body>
  <div class="container">

    <h1>Welcome to Notes!</h1>
    <h5>{{ note_count }} notes stored so far!</h5>
```

然后，我们可以刷新笔记应用程序的主页，以查看计数器的实际效果。现在，我们需要编写增加计数器的代码，但在继续之前，有一些事情我们应该注意。

多个请求可能会并发尝试在 Memcache 中增加值，这可能导致竞争条件。为了避免这种情况，Memcache 提供了两个函数，`incr()` 和 `decr()`，它们可以原子性地增加和减少 64 位整数值。这些将非常适合我们的计数器，但我们可以提供一个更通用的解决方案，该解决方案也适用于非整数的缓存值，使用 App Engine Python API 的 **比较** 和 **设置** 功能。

在 `utils.py` 模块中，我们添加了以下函数：

```py
def inc_note_counter():
    client = memcache.Client()
    retry = 0
    while retry < 10:
        data = client.gets('note_count')
        if client.cas('note_count', data+1):
            break
        retry += 1
```

我们使用 `Client` 类的一个实例，因为 `memcache` 模块中没有提供比较和设置功能作为函数。在获取到 `Client` 实例后，我们进入所谓的 `retry` 循环，如果检测到罕见条件，我们将重复迭代最多 10 次。然后我们尝试使用客户端的 `gets` 方法获取 `note_count` 键的值。此方法会改变客户端的内部状态，存储由 Memcache 服务提供的戳记值。然后我们尝试通过在客户端对象上调用 `cas()` 方法来增加与同一键对应的值；该方法将键的新值传输到 Memcache，以及之前提到的戳记。如果戳记匹配，则值被更新，`cas()` 方法返回 `True` 参数，导致 `retry` 循环退出；否则，它返回 `False` 参数，我们再次尝试。

在 `main` 模块中导入 `inc_note_counter()` 函数后，我们可以在创建新笔记的地方调用它来增加计数器：在 `MainHandler` 类的 `_create_note` 中，以及在 `CreateNoteHandler` 类的 `_create_note` 方法中。

# 将我们的应用程序拆分为模块

目前，我们的笔记应用程序提供了一些前端功能，例如服务主页，以及一些后端功能，例如处理定时任务。这对于大多数用例来说是可以的，但如果应用程序架构复杂且流量很大，那么有多个后端任务占用资源，这并不总是可以接受的。为了应对这类问题，App Engine 提供了一种极其灵活的方式来使用 **模块** 来布局 Web 应用程序。

每个 App Engine 应用程序至少由一个模块组成；即使我们之前不知道，到目前为止，我们已经处理了 Notes 应用程序的默认模块。模块通过名称标识，由源代码和配置文件组成，可以位于应用程序根目录或子目录中。每个模块都有一个版本，我们可以部署同一模块的多个版本；每个版本将根据我们如何配置其扩展性而生成一个或多个 App Engine 实例。能够部署同一模块的多个版本，特别是对于测试新组件或部署渐进式升级非常有用。属于同一应用程序的模块共享服务，如 Memcache、Datastore 和任务队列，并且可以使用 Python API 模块以安全的方式通信。

要深入了解一些其他细节，我们可以通过添加一个专门处理 cron 作业的新模块来重构我们的 Notes 应用程序。我们不需要添加任何功能；我们只是分解和重构现有代码。由于我们的应用程序架构非常简单，我们可以直接在应用程序根目录中添加该模块。首先，我们需要配置这个新模块，我们将在一个新文件`backend.yaml`中将其命名为`backend`，该文件包含以下内容：

```py
application: notes
module: backend
version: 1
runtime: python27
api_version: 1
threadsafe: yes

handlers:
- url: .*
  script: backend.app
```

这与任何应用程序配置文件非常相似，但主要区别是包含模块名称的`module`属性。当此属性不在配置文件中，或其值为`default`字符串时，App Engine 假定这是应用程序的默认模块。然后我们告诉 App Engine，我们希望`backend_main`文件中的 Python 模块处理该模块将接收到的每个请求。当我们不在配置文件中指定任何扩展选项时，将假定**自动扩展**。

我们在`backend_main.py`文件中编写了一个全新的 Python 模块，其中包含一个专用的 WSGI 兼容应用程序：

```py
app = webapp2.WSGIApplication([
    (r'/shrink_all', ShrinkCronJob),
], debug=True)
```

如我们从映射中看到的那样，此应用程序将仅处理 shrink cron 作业的请求。我们从主模块中获取处理程序代码，为了避免依赖它，我们重写了`ShrinkCronJob`类，使其不再需要从`ShrinkHandler`类派生。同样，在`backend_main.py`模块中，我们添加以下内容：

```py
class ShrinkCronJob(webapp2.RequestHandler):
    @ndb.tasklet
    def _shrink_note(self, note):
        for file_key in note.files:
            file = yield file_key.get_async()
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
        if 'X-AppEngine-Cron' not in self.request.headers:
            self.error(403)

        notes = Note.query().fetch()
        for note in notes:
            self._shrink_note(note)
```

为了方便起见，我们可以将`image_formats`字典移动到`utils.py`模块中，这样我们就可以在这里以及`main.py`模块中重用它。

现在我们有两个模块，我们需要将进入我们应用程序的请求路由到正确的模块，我们可以通过在应用程序根目录中创建一个名为`dispatch.yaml`的文件来实现，该文件包含以下内容：

```py
dispatch:

  - url: "*/shrink_all"
    module: backend

  - url: "*/*"
    module: default
```

基本上，这是我们可以在 App Engine 上拥有的最高级别的 URL 映射。我们可以使用通配符而不是正则表达式来将传入请求的 URL 路由到正确的模块；在这种情况下，我们将请求路由到`/shrink_all` URL，将所有其他请求留给默认模块。

### 注意

理想情况下，我们可以将实现通过电子邮件创建笔记的代码也移动到后端模块，但不幸的是，App Engine 只允许在默认模块上使用入站服务。

在本地开发环境和生产环境中使用模块会增加一些复杂性，因为我们不能使用 App Engine Launcher 图形界面来启动和停止开发服务器或部署应用程序；我们必须使用命令行工具。

例如，我们可以检查模块在本地环境中的工作方式，但我们必须通过传递每个模块的`YAML`文件以及`dispatch.yaml`文件作为参数来启动开发服务器。在我们的情况下，我们在命令行上执行以下操作：

```py
dev_appserver.py app.yaml backend.yaml dispatch.yaml

```

要在 App Engine 上部署应用程序，我们使用`appcfg`命令行工具传递我们想要部署的模块的`YAML`文件，确保在第一次部署时，默认模块的配置文件是列表中的第一个，例如，我们可以使用以下`YAML`文件：

```py
appcfg.py update app.yaml backend.yaml

```

当应用程序重新启动时，我们应该能够在开发控制台或管理控制台中看到为额外的后端模块运行的实例。

### 注意

由于在像笔记这样的小型应用程序上使用模块不太实用，并且对本书的目的没有提供任何好处，我们可以切换回只有一个模块的布局。

# 摘要

在本章中，我们深入探讨了迄今为止我们使用的云平台组件的许多细节。正如之前提到的，当使用按使用付费的服务，如云平台时，掌握细节和最佳实践对性能和成本都有益。本章的大部分内容都致力于云数据存储，确认这是几乎所有 Web 应用程序的关键组件；了解如何布局数据或执行查询可以决定我们应用程序的成功。

我们还学习了如何从 Python 应用程序安全地使用 Memcache，避免竞争条件和难以调试的奇怪行为。在章节的最后部分，我们涵盖了 App Engine 的模块功能；即使我们必须处理复杂的应用程序以完全欣赏模块化架构的好处，了解模块是什么以及它们能为我们做什么也是重要信息，如果我们想在 App Engine 上部署我们的应用程序。

下一章完全致力于 Google Cloud SQL 服务。我们将学习如何创建和管理数据库实例，以及如何建立连接和执行查询。
