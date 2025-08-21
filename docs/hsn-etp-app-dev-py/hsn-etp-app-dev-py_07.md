# 第七章：构建优化的前端

在本书中，我们已经在尝试了解如何在 Python 中为企业构建应用程序时走得很远。到目前为止，我们已经涵盖了如何为我们的企业应用程序构建一个可扩展和响应迅速的后端，以满足大量并发用户，以便我们的企业应用程序能够成功地为其用户提供服务。然而，在构建企业级应用程序时，有一个我们一直忽视的话题，通常在构建企业级应用程序时很少受到关注：应用程序的前端。

当用户与我们的应用程序交互时，他们很少关心后端发生了什么。用户的体验直接与应用程序的前端如何响应他们的输入相关。这使得应用程序的前端不仅是应用程序最重要的方面之一，也使其成为应用程序在用户中成功的主要决定因素之一。

在本章中，我们将看看如何构建应用程序前端，不仅提供易于使用的体验，还能快速响应他们的输入。

在阅读本章时，我们将学习以下主题：

+   优化应用前端的需求

+   优化前端所依赖的资源

+   利用客户端缓存来简化页面加载

+   利用 Web 存储持久化用户数据

# 技术要求

本书中的代码清单可以在`chapter06`中的`bugzot`应用程序目录下找到[`github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python`](https://github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python)。

可以通过运行以下命令克隆代码示例：

```py
git clone https://github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python
```

代码的执行不需要任何特定的特殊工具或框架，这是一个非常简单的过程。`README.md`文件指向了如何运行本章的代码示例。

# 优化前端的需求

应用的用户界面是最重要的用户界面组件之一。它决定了用户如何感知应用程序。一个流畅的前端在定义用户体验方面起着很大的作用。

这种对流畅用户体验的需求带来了优化应用前端的需求，它提供了一个易于使用的界面，快速的响应时间和操作的流畅性。如果我们继续向 Web 2.0 公司（如 Google、Facebook、LinkedIn 等）看齐，他们会花费大量资源来优化他们的前端，以减少几毫秒的渲染时间。这就是优化前端的重要性。

# 优化前端的组件

我们正在讨论优化前端。但是优化的前端包括什么？我们如何决定一个前端是否被优化了？让我们来看一下。

优化的前端有几个组件，不是每个组件都需要从前端反映出来。这些组件如下：

+   **快速渲染时间**：前端优化的首要重点之一是减少页面的渲染时间。虽然没有预定义的渲染时间可以被认为是好还是坏，但你可以认为一个好的渲染时间是用户在一个体面的互联网连接上不必等待太长时间页面加载的时间。另外，...

# 导致前端问题的原因

前端问题是一类问题，用户很容易察觉到，因为它们影响用户与应用程序的交互方式。在这里，为了清楚起见，当我们说企业 Web 应用的前端时，我们不仅谈论其用户界面，还谈论代码和模板，这些都是用来呈现所需用户界面的。现在，让我们继续了解前端特定问题的可能原因：

+   **过多的对象**：在大多数负责呈现前端的动态填充模板中，第一个问题出现在呈现过多对象时。当大量对象传递给需要呈现的模板时，页面响应时间往往会增加，导致过程明显减慢。

+   **过多的包含**：软件工程中关注的一个主要问题是如何增加代码库的模块化。模块化的增加有助于增加组件的可重用性。然而，过度的模块化可能是可能出现重大问题的信号。当前端模板被模块化到超出所需程度时，模板的呈现性能会降低。原因在于每个包含都需要从磁盘加载一个新文件，这是一个异常缓慢的操作。这里的一个反驳观点可能是，一旦模板加载了所有包含的内容，呈现引擎就可以缓存模板，并从缓存中提供后续请求。然而，大多数缓存引擎对它们可以缓存的包含深度有一个限制，超出这个限制，性能损失将是明显的。

+   **不必要的资源集**：一些前端可能加载了大量不在特定页面上使用的资源。这包括包含仅在少数页面上执行的函数的 JavaScript 文件。每个额外加载的文件不仅增加了带宽的消耗，还影响了前端的加载性能。

+   **强制串行加载代码**：现代大多数浏览器都经过优化，可以并行加载大量资源，以有效利用网络带宽并减少页面加载时间。然而，有时，我们用来减少代码量的一些技巧可能会强制页面按顺序加载，而不是并行加载。可能导致页面资源按顺序加载的最常见示例之一是使用 CSS 导入。尽管 CSS 导入提供了直接在另一个样式表中加载第三方 CSS 文件的灵活性，但它也减少了浏览器加载 CSS 文件内容的能力，因此增加了呈现页面所需的时间。

这一系列原因构成了可能导致页面呈现时间减慢的问题的非穷尽列表，因此给用户带来不愉快的体验。

现在，让我们看看如何优化我们的前端，使其具有响应性，并提供最佳的用户体验。

# 优化前端

到目前为止，我们了解了可能影响前端性能的各种问题。现在，是时候看看我们如何减少前端的性能影响，并使它们在企业级环境中快速响应。

# 优化资源

我们首先要看的优化是在请求特定页面时加载的资源。为此，请考虑管理面板中用户数据显示页面的以下代码片段，该页面负责显示数据库中的用户表：

```py
<table>
{% for user in users %}
  <tr>
    <td class="user-data-column">{{ user.username }}</td>
    <td class="user-data-column">{{ user.email }}</td>
    <td class="user-data-column">{{ user.status }}</td>
  </tr>
{% endfor %}
</table>
```

到目前为止，一切顺利。正如我们所看到的，代码片段只是循环遍历用户对象，并根据用户表中存储的记录数量来渲染表格。这对于大多数情况下用户记录只有少量（例如 100 条左右）的情况来说是很好的。但随着应用程序中用户数量的增长，这段代码将开始出现问题。想象一下尝试从应用程序数据库中加载 100 万条记录并在 UI 上显示它们。这会带来一些问题：

+   **数据库查询缓慢：**尝试同时从数据库加载 100 万条记录将会非常缓慢，并且可能需要相当长的时间，因此会阻塞视图很长时间。

+   **解码前端对象：**在前端，为了渲染页面，模板引擎必须解码所有对象的数据，以便能够在页面上显示数据。这种操作不仅消耗 CPU，而且速度慢。

+   **页面大小过大：**想象一下从服务器到客户端通过网络传输数百万条记录的页面。这个过程耗时且使页面不适合在慢速连接上加载。

那么，我们可以在这里做些什么呢？答案很简单：让我们优化将要加载的资源量。为了实现这一点，我们将利用一种称为分页的概念。

为了实现分页，我们需要对负责渲染前端模板的视图以及前端模板进行一些更改。以下代码描述了如果视图需要支持分页，它将会是什么样子：

```py
From bugzot.application import app, db
from bugzot.models import User
from flask.views import MethodView
from flask import render_template, session, request

class UserListView(MethodView):
    """User list view for displaying user data in admin panel.

      The user list view is responsible for rendering the table of users that are registered
      in the application.
    """

    def get(self):
        """HTTP GET handler."""

        page = request.args.get('next_page', 1) # get the page number to be displayed
        users = User.query.paginate(page, 20, False)
        total_records = users.total
        user_records = users.items

        return render_template('admin/user_list.html', users=user_records, next_page=page+1)
```

我们现在已经完成了对视图的修改，它现在支持分页。通过使用 SQLAlchemy 提供的设施，实现这种分页是一项相当容易的任务，使用`paginate()`方法从数据库表中分页结果。这个`paginate()`方法需要三个参数，即页面编号（应从 1 开始），每页记录数，以及`error_out`，它负责设置该方法的错误报告。在这里设置为`False`会禁用在`stdout`上显示错误。

开发支持分页的视图后，下一步是定义模板，以便它可以利用分页。以下代码显示了修改后的模板代码，以利用分页：

```py
<table>
{% for user in users %}
  <tr>
    <td class="user-data-column">{{ user.username }}</td>
    <td class="user-data-column">{{ user.email }}</td>
    <td class="user-data-column">{{ user.status }}</td>
  </tr>
{% endfor %}
</table>
<a href="{{ url_for('admin_user_list', next_page) }}">Next Page</a>

```

有了这个视图代码，我们的视图代码已经准备好了。这个视图代码非常简单，因为我们只是通过添加一个`href`来扩展之前的模板，该`href`加载下一页的数据。

现在我们已经优化了发送到页面的资源，接下来我们需要关注的是如何使我们的前端更快地加载更多资源。

# 通过避免 CSS 导入并行获取 CSS

CSS 是任何前端的重要组成部分，它帮助为浏览器提供样式信息，告诉浏览器如何对从服务器接收到的页面进行样式设置。通常，前端可能会有许多与之关联的 CSS 文件。我们可以通过使这些 CSS 文件并行获取来实现一些可能的优化。

所以，让我们想象一下我们有以下一组 CSS 文件，即`main.css`、`reset.css`、`responsive.css`和`grid.css`，我们的前端需要加载。我们允许浏览器并行加载所有这些文件的方式是通过使用 HTML 链接标签将它们链接到前端，而不是使用 CSS 导入，这会导致加载 CSS 文件...

# 打包 JavaScript

在当前时间和希望的未来，我们将不断看到网络带宽的增加，无论是宽带网络还是移动网络，都可以实现资源的并行更快下载。但是对于每个需要从远程服务器获取的资源，由于每个单独的资源都需要向服务器发出单独的请求，仍然涉及一些网络延迟。当需要加载大量资源并且用户在高延迟网络上时，这种延迟可能会影响。

通常，大多数现代 Web 应用程序广泛利用 JavaScript 来实现各种目的，包括输入验证、动态生成内容等。所有这些功能都分成多个文件，其中可能包括一些库、自定义代码等。虽然将所有这些拆分成不同的文件可以帮助并行加载，但有时 JavaScript 文件包含用于在网页上生成动态内容的代码，这可能会阻止网页的呈现，直到成功加载网页呈现所需的所有必要文件。

我们可以减少浏览器加载这些脚本资源所需的时间的一种可能的方法是将它们全部捆绑到一个单一文件中。这允许所有脚本组合成一个单一的大文件，浏览器可以在一个请求中获取。虽然这可能会导致用户在首次访问网站时体验有点慢，但一旦资源被获取和缓存，用户对网页的后续加载将会显著更快。

今天，有很多第三方库可用，可以让我们捆绑这些 JavaScript。让我们以一个名为 Browserify 的简单工具为例，它允许我们捆绑我们的 JavaScript 文件。例如，如果我们有多个 JavaScript 文件，如`jquery.js`、`image-loader.js`、`slideshow.js`和`input-validator.js`，并且我们想要使用 Browserify 将这些文件捆绑在一起，我们只需要运行以下命令：

```py
browserify jquery.js image-loader.js slideshow.js input-validator.js > bundle.js
```

这个命令将把这些 JavaScript 文件创建成一个称为`bundle.js`的公共文件包，现在可以通过简单的脚本标签包含在我们的 Web 应用程序中，如下所示：

```py
<script type="text/javascript" src="js/bundle.js"></script>
```

将 JavaScript 捆绑到一个请求中加载，我们可能会开始看到一些改进，以便页面在浏览器中快速获取和显示给用户。现在，让我们来看看另一个可能有用的有趣主题，它可能会在网站重复访问时对我们的 Web 应用程序加载速度产生真正的影响。

我们讨论的 JavaScript 捆绑技术也可以用于包含 CSS 文件的优化。

# 利用客户端缓存

缓存长期以来一直被用来加快频繁使用的资源的加载速度。例如，大多数现代操作系统利用缓存来提供对最常用应用程序的更快访问。Web 浏览器也利用缓存，在用户再次访问同一网站时，提供对资源的更快访问。这样做是为了避免如果文件没有更改就一遍又一遍地从远程服务器获取它们，从而减少可能需要的数据传输量，同时提高页面的呈现时间。

现在，在企业应用程序的世界中，像客户端缓存这样的东西可能会非常有用。这是因为...

# 设置应用程序范围的缓存控制

由于我们的应用程序基于 Flask，我们可以利用几种简单的机制来为我们的应用程序设置缓存控制。例如，将以下代码添加到我们的`bugzot/application.py`文件的末尾可以启用站点范围的缓存控制，如下所示：

```py
@app.after_request
def cache_control(response):
  """Implement side wide cache control."""
  response.cache_control.max_age = 300
  response.cache_control.public = True
  return response
```

在这个例子中，我们利用 Flask 内置的`after_request`装饰器钩子来设置 HTTP 响应头，一旦请求到达 Flask 应用程序，装饰的函数需要一个参数，该参数接收一个响应类的对象，并返回一个修改后的响应对象。

对于我们的用例，在`after_request`钩子的方法代码中，我们设置了`cache_control.max_age`头，该头指定了内容在再次从服务器获取之前从缓存中提供的时间的上限，以及`cache_control.public`头，该头定义了缓存响应是否可以与多个请求共享。

现在，可能会有时候我们想为特定类型的请求设置不同的缓存控制。例如，我们可能不希望为用户个人资料页面设置`cache_control.public`，以避免向不同的用户显示相同的个人资料数据。我们的应用程序允许我们相当快速地实现这些类型的场景。让我们来看一下。

# 设置请求级别的缓存控制

在 Flask 中，我们可以在将响应发送回客户端之前修改响应头。这可以相当容易地完成。以下示例显示了一个实现响应特定头控制的简单视图：

```py
from bugzot.application import app, dbfrom bugzot.models import Userfrom flask.views import MethodViewfrom flask import render_template, session, request, make_responseclass UserListView(MethodView):  """User list view for displaying user data in admin panel.  The user list view is responsible for rendering the table of users that are registered  in the application.  """  def get(self):    """HTTP GET handler."""        page = request.args.get('next_page', 1) # get the page number to be displayed users = User.query.paginate(page, ...
```

# 利用 Web 存储

任何曾经处理过即使是一点点用户管理的应用程序的 Web 应用程序开发人员肯定都听说过 Web cookies，它本质上提供了一种在客户端存储一些信息的机制。

利用 cookies 提供了一种简单的方式，通过它我们可以在客户端维护小量用户数据，并且可以多次读取，直到 cookies 过期。但是，尽管处理 cookies 很容易，但有一些限制限制了 cookies 的实用性，除了在客户端维护少量应用程序状态之外。其中一些限制如下：

+   cookies 随每个请求传输，因此增加了每个请求传输的数据量

+   Cookies 允许存储少量数据，最大限制为 4 KB

现在，出现的问题是，如果我们想存储更多的数据，或者我们想避免在每个请求中一遍又一遍地获取相同的存储数据，我们该怎么办？

为了处理这种情况，HTML 的最新版本 HTML 5 提供了各种功能，允许处理客户端 Web 存储。这种 Web 存储相对于基于 cookies 的机制提供了许多优点，例如：

+   由于 Web 存储直接在客户端上可用，因此不需要服务器一遍又一遍地将信息发送到客户端

+   Web 存储 API 提供了最多 10 MB 的存储空间，这是比使用 cookies 存储的多次更大的存储空间

+   Web 存储提供了在本地存储中存储数据的灵活性，例如，即使用户关闭并重新打开浏览器，数据也是可访问的，或者基于每个会话的基础上存储数据，其中存储在 Web 存储中的数据将在会话失效时被清除，无论是当用户会话被应用程序处理用户注销的处理程序销毁，还是浏览器关闭

这使得 Web 存储成为一个吸引人的地方，可以存放数据，避免一遍又一遍地加载

对于我们的企业应用程序，这可以通过仅在用户浏览器中存储中间步骤的结果，然后仅在填写完所有必需的输入字段时将它们提交回服务器，从而提供很大的灵活性。

另一个可能更适用于 Bugzot 的用例是，我们可以将用户提交的错误报告存储到 Web 存储中，并在完成错误报告时将其发送到服务器。在这种情况下，用户可以灵活地随时回到处理其错误报告，而不必担心再次从头开始。

现在我们知道了 Web 存储提供的好处，让我们看看如何利用 Web 存储的使用。

# 使用本地 Web 存储

使用 HTML 5 的本地 Web 存储非常容易，因为它提供了许多 API 来与 Web 存储交互。因此，让我们不浪费时间，看一下我们如何使用本地 Web 存储的一个简单例子。为此，我们将创建一个名为`localstore.js`的简单 JavaScript 文件，内容如下：

```py
// check if the localStorage is supported by the browser or notif(localStorage) {  // Put some contents inside the local storagelocalStorage.setItem("username", "joe_henry");  localStorage.setItem("uid", "28372");    // Retrieve some contents from the local storage  var user_email = localStorage.getItem("user_email");} else {  alert("The browser does not support local web storage");}
```

这是...

# 使用会话存储

使用本地存储同样简单，会话存储也不会增加任何复杂性。例如，让我们看看将我们的`localStorage`示例轻松转换为`sessionStorage`有多容易：

```py
// check if the sessionStorage is supported by the browser or not
if(sessionStorage) {
  // Put some contents inside the local storage
sessionStorage.setItem("username", "joe_henry");
  sessionStorage.setItem("uid", "28372");

  // Retrieve some contents from the session storage
  var user_email = sessionStorage.getItem("user_email");
} else {
  alert("The browser does not support session web storage");
}
```

从这个例子可以明显看出，从本地存储转移到会话存储非常容易，因为这两种存储选项都提供了类似的存储 API，唯一的区别在于存储中的数据保留时间有多长。

通过了解如何优化前端以提供完全可扩展和响应的企业 Web 应用程序，现在是时候我们访问一些确保我们构建的内容安全并符合预期的企业应用程序开发方面的内容，而不会带来意外惊喜。

# 摘要

在本章的过程中，我们了解了为企业应用程序拥有优化的前端的重要性，以及前端如何影响我们在企业内部使用应用程序。然后，我们继续了解通常会影响 Web 前端性能的问题类型，以及我们可以采取哪些可能的解决方案来改进应用程序前端。这包括减少前端加载的资源量，允许 CSS 并行加载，捆绑 JavaScript 等。然后，我们继续了解缓存如何在考虑企业 Web 应用程序的使用情况下证明是有用的。一旦我们了解了缓存的概念，我们就进入了领域...

# 问题

1.  CDN 的使用如何提高前端性能？

1.  我们能做些什么让浏览器利用现有的连接从服务器加载资源吗？

1.  我们如何从 Web 存储中删除特定键或清除 Web 存储的内容？
