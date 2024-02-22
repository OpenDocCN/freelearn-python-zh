# 第十六章：使用 QtWebEngine 进行 Web 浏览

在第八章中，*使用 QtNetwork 进行网络操作*，您学习了如何使用套接字和 HTTP 与网络系统进行交互。然而，现代网络远不止于网络协议；它是建立在 HTML、JavaScript 和 CSS 组合之上的编程平台，有效地使用它需要一个完整的 Web 浏览器。幸运的是，Qt 为我们提供了`QtWebEngineWidgets`库，为我们的应用程序提供了一个完整的 Web 浏览器小部件。

在本章中，我们将学习如何在以下部分中使用 Qt 访问 Web：

+   使用`QWebEngineView`构建基本浏览器

+   高级`QtWebEngine`用法

# 技术要求

除了本书中使用的基本 PyQt5 设置之外，您还需要确保已从 PyPI 安装了`PyQtWebEngine`软件包。您可以使用以下命令执行此操作：

```py
$ pip install --user PyQtWebEngine
```

您可能还想要本章的示例代码，可以从[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter16`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter16)获取。

查看以下视频，了解代码的运行情况：[`bit.ly/2M5xFtD`](http://bit.ly/2M5xFtD)

# 使用`QWebEngineView`构建基本浏览器

从`QtWebEngineWidgets`中使用的主要类是`QWebEngineView`类；这个类在`QWidget`对象中提供了一个几乎完整的基于 Chromium 的浏览器。Chromium 是支持许多 Google Chrome、最新版本的 Microsoft Edge 和许多其他浏览器的开源项目。

Qt 还有一个基于**Webkit**渲染引擎的已弃用的`QtWebKit`模块，用于 Safari、Opera 和一些旧版浏览器。`QtWebKit`和`QtWebEngineWidgets`之间的 API 和渲染行为存在一些显着差异，后者更适合新项目。

在本节中，我们将看到使用`QtWebEngineWidgets`构建一个简单的 Web 浏览器，将 Web 内容包含在 Qt 应用程序中是多么容易。

# 使用 QWebEngineView 小部件

我们需要从第四章中复制我们的 Qt 应用程序模板，*使用 QMainWindow 构建应用程序*，并将其命名为`simple_browser.py`；我们将开发一个带有选项卡和历史记录显示的基本浏览器。

我们首先导入`QtWebEngineWidgets`库，如下所示：

```py
from PyQt5 import QtWebEngineWidgets as qtwe
```

请注意，还有一个`QtWebEngine`模块，但它是用于与**Qt 建模语言**（**QML**）声明性框架一起使用的，而不是本书涵盖的 Qt 小部件框架。`QtWebEngineWidgets`包含基于小部件的浏览器。

在我们的`MainWindow`类构造函数中，我们将通过定义导航工具栏来启动 GUI：

```py
        navigation = self.addToolBar('Navigation')
        style = self.style()
        self.back = navigation.addAction('Back')
        self.back.setIcon(style.standardIcon(style.SP_ArrowBack))
        self.forward = navigation.addAction('Forward')
        self.forward.setIcon(style.standardIcon(style.SP_ArrowForward))
        self.reload = navigation.addAction('Reload')
        self.reload.setIcon(style.standardIcon(style.SP_BrowserReload))
        self.stop = navigation.addAction('Stop')
        self.stop.setIcon(style.standardIcon(style.SP_BrowserStop))
        self.urlbar = qtw.QLineEdit()
        navigation.addWidget(self.urlbar)
        self.go = navigation.addAction('Go')
        self.go.setIcon(style.standardIcon(style.SP_DialogOkButton))
```

在这里，我们为标准浏览器操作定义了工具栏按钮，以及用于 URL 栏的`QLineEdit`对象。我们还从默认样式中提取了这些操作的图标，就像我们在第四章的*添加工具栏*部分中所做的那样，*使用 QMainWindow 构建应用程序*。

现在我们将创建一个`QWebEngineView`对象：

```py
        webview = qtwe.QWebEngineView()
        self.setCentralWidget(webview)
```

`QWebEngineView`对象是一个（大多数情况下，正如您将看到的那样）功能齐全且交互式的 Web 小部件，能够检索和呈现 HTML、CSS、JavaScript、图像和其他标准 Web 内容。

要在视图中加载 URL，我们将`QUrl`传递给其`load()`方法：

```py
        webview.load(qtc.QUrl('http://www.alandmoore.com'))
```

这将提示 Web 视图下载并呈现页面，就像普通的 Web 浏览器一样。

当然，尽管该网站很好，我们希望能够浏览其他网站，因此我们将添加以下连接：

```py
        self.go.triggered.connect(lambda: webview.load(
            qtc.QUrl(self.urlbar.text())))
```

在这里，我们将我们的`go`操作连接到一个`lambda`函数，该函数检索 URL 栏的文本，将其包装在`QUrl`对象中，并将其发送到 Web 视图。如果此时运行脚本，您应该能够在栏中输入 URL，点击 Go，然后像任何其他浏览器一样浏览 Web。

`QWebView`具有所有常见浏览器导航操作的插槽，我们可以将其连接到我们的导航栏：

```py
        self.back.triggered.connect(webview.back)
        self.forward.triggered.connect(webview.forward)
        self.reload.triggered.connect(webview.reload)
        self.stop.triggered.connect(webview.stop)
```

通过连接这些信号，我们的脚本已经在成为一个完全功能的网络浏览体验的路上。但是，我们目前仅限于单个浏览器窗口；我们想要选项卡，因此让我们在以下部分实现它。

# 允许多个窗口和选项卡

在`MainWindow.__init__()`中，删除或注释掉刚刚添加的 Web 视图代码（返回到创建`QWebEngineView`对象）。我们将将该功能移动到一个方法中，以便我们可以在选项卡界面中创建多个 Web 视图。我们将按照以下方式进行：

1.  首先，我们将用`QTabWidget`对象替换我们的`QWebEngineView`对象作为我们的中央小部件：

```py
        self.tabs = qtw.QTabWidget(
            tabsClosable=True, movable=True)
        self.tabs.tabCloseRequested.connect(self.tabs.removeTab)
        self.new = qtw.QPushButton('New')
        self.tabs.setCornerWidget(self.new)
        self.setCentralWidget(self.tabs)
```

此选项卡小部件将具有可移动和可关闭的选项卡，并在左上角有一个新按钮用于添加新选项卡。

1.  要添加一个带有 Web 视图的新选项卡，我们将创建一个`add_tab()`方法：

```py
    def add_tab(self, *args):
        webview = qtwe.QWebEngineView()
        tab_index = self.tabs.addTab(webview, 'New Tab')
```

该方法首先创建一个 Web 视图小部件，并将其添加到选项卡小部件的新选项卡中。

1.  现在我们有了我们的 Web 视图对象，我们需要连接一些信号：

```py
        webview.urlChanged.connect(
            lambda x: self.tabs.setTabText(tab_index, x.toString()))
        webview.urlChanged.connect(
            lambda x: self.urlbar.setText(x.toString()))
```

`QWebEngineView`对象的`urlChanged`信号在将新 URL 加载到视图中时发出，并将新 URL 作为`QUrl`对象发送。我们将此信号连接到一个`lambda`函数，该函数将选项卡标题文本设置为 URL，以及另一个函数，该函数设置 URL 栏的内容。这将使 URL 栏与用户在网页中使用超链接导航时与浏览器保持同步，而不是直接使用 URL 栏。

1.  然后，我们可以使用其`setHtml()`方法向我们的 Web 视图对象添加默认内容：

```py
        webview.setHtml(
            '<h1>Blank Tab</h1><p>It is a blank tab!</p>',
            qtc.QUrl('about:blank'))
```

这将使浏览器窗口的内容成为我们提供给它的任何 HTML 字符串。如果我们还传递一个`QUrl`对象，它将被用作当前 URL（例如发布到`urlChanged`信号）。

1.  为了启用导航，我们需要将我们的工具栏操作连接到浏览器小部件。由于我们的浏览器有一个全局工具栏，我们不能直接将这些连接到 Web 视图小部件。我们需要将它们连接到将信号传递到当前活动 Web 视图的插槽的方法。首先创建回调方法如下：

```py
    def on_back(self):
        self.tabs.currentWidget().back()

    def on_forward(self):
        self.tabs.currentWidget().forward()

    def on_reload(self):
        self.tabs.currentWidget().reload()

    def on_stop(self):
        self.tabs.currentWidget().stop()

    def on_go(self):
        self.tabs.currentWidget().load(
            qtc.QUrl(self.urlbar.text()))
```

这些方法本质上与单窗格浏览器使用的方法相同，但有一个关键变化——它们使用选项卡窗口小部件的`currentWidget()`方法来检索当前可见选项卡的`QWebEngineView`对象，然后在该 Web 视图上调用导航方法。

1.  在`__init__()`中连接以下方法：

```py
        self.back.triggered.connect(self.on_back)
        self.forward.triggered.connect(self.on_forward)
        self.reload.triggered.connect(self.on_reload)
        self.stop.triggered.connect(self.on_stop)
        self.go.triggered.connect(self.on_go)
        self.urlbar.returnPressed.connect(self.on_go)
        self.new.clicked.connect(self.add_tab)
```

为了方便和键盘友好性，我们还将 URL 栏的`returnPressed`信号连接到`on_go()`方法。我们还将我们的新按钮连接到`add_tab()`方法。

现在尝试浏览器，您应该能够添加多个选项卡并在每个选项卡中独立浏览。

# 为弹出窗口添加选项卡

目前，我们的脚本存在问题，即如果您*Ctrl* +单击超链接，或打开配置为打开新窗口的链接，将不会发生任何事情。默认情况下，`QWebEngineView`无法打开新标签页或窗口。为了启用此功能，我们必须使用一个函数覆盖其`createWindow()`方法，该函数创建并返回一个新的`QWebEngineView`对象。

我们可以通过更新我们的`add_tab()`方法来轻松实现这一点：

```py
        webview.createWindow = self.add_tab
        return webview
```

我们不会对`QWebEngineView`进行子类化以覆盖该方法，而是将我们的`MainWindow.add_tab()`方法分配给其`createWindow()`方法。然后，我们只需要确保在方法结束时返回创建的 Web 视图对象。

请注意，我们不需要在`createWindow()`方法中加载 URL；我们只需要适当地创建视图并将其添加到 GUI 中。Qt 将负责在我们返回的 Web 视图对象中执行浏览所需的操作。

现在，当您尝试浏览器时，您应该发现*Ctrl * +单击会打开一个带有请求链接的新选项卡。

# 高级 QtWebEngine 用法

虽然我们已经实现了一个基本的、可用的浏览器，但它还有很多不足之处。在本节中，我们将通过修复用户体验中的一些痛点和实现有用的工具，如历史和文本搜索，来探索`QtWebEngineWidgets`的一些更高级的功能。

# 共享配置文件

虽然我们可以在浏览器中查看多个选项卡，但它们在与经过身份验证的网站一起工作时存在一个小问题。访问任何您拥有登录帐户的网站；登录，然后*Ctrl *+单击站点内的链接以在新选项卡中打开它。您会发现您在新选项卡中没有经过身份验证。对于使用多个窗口或选项卡来实现其用户界面的网站来说，这可能是一个真正的问题。我们希望身份验证和其他会话数据是整个浏览器范围的，所以让我们来解决这个问题。

会话信息存储在一个由`QWebEngineProfile`对象表示的**配置文件**中。这个对象是为每个`QWebEngineWidget`对象自动生成的，但我们可以用自己的对象来覆盖它。

首先在`MainWindow.__init__()`中创建一个：

```py
        self.profile = qtwe.QWebEngineProfile()
```

当我们在`add_tab()`中创建新的 web 视图时，我们需要将这个配置文件对象与每个新的 web 视图关联起来。然而，配置文件实际上并不是 web 视图的属性；它们是 web 页面对象的属性。页面由`QWebEnginePage`对象表示，可以被视为 web 视图的*模型*。每个 web 视图都会生成自己的`page`对象，它充当了浏览引擎的接口。

为了覆盖 web 视图的配置文件，我们需要创建一个`page`对象，覆盖它的配置文件，然后用我们的新页面覆盖 web 视图的页面，就像这样：

```py
        page = qtwe.QWebEnginePage(self.profile)
        webview.setPage(page)
```

配置文件*必须*作为参数传递给`QWebEnginePage`构造函数，因为没有访问函数可以在之后设置它。一旦我们有了一个使用我们的配置文件的新的`QWebEnginePage`对象，我们就可以调用`QWebEngineView.setPage()`将其分配给我们的 web 视图。

现在当您测试浏览器时，您的身份验证状态应该在所有选项卡中保持不变。

# 查看历史记录

每个`QWebEngineView`对象都管理着自己的浏览历史，我们可以访问它来允许用户查看和导航已访问的 URL。

为了构建这个功能，让我们创建一个界面，显示当前选项卡的历史记录，并允许用户点击历史记录项进行导航：

1.  首先在`MainView.__init__()`中创建一个历史记录的停靠窗口小部件：

```py
        history_dock = qtw.QDockWidget('History')
        self.addDockWidget(qtc.Qt.RightDockWidgetArea, history_dock)
        self.history_list = qtw.QListWidget()
        history_dock.setWidget(self.history_list)
```

历史记录停靠窗口只包含一个`QListWidget`对象，它将显示当前选定选项卡的历史记录。

1.  由于我们需要在用户切换选项卡时刷新这个列表，将选项卡小部件的`currentChanged`信号连接到一个可以执行此操作的回调函数：

```py
        self.tabs.currentChanged.connect(self.update_history)
```

1.  `update_history()`方法如下：

```py
    def update_history(self, *args):
        self.history_list.clear()
        webview = self.tabs.currentWidget()
        if webview:
            history = webview.history()
            for history_item in reversed(history.items()):
                list_item = qtw.QListWidgetItem()
                list_item.setData(
                    qtc.Qt.DisplayRole, history_item.url())
                self.history_list.addItem(list_item)
```

首先，我们清除列表小部件并检索当前活动选项卡的 web 视图。如果 web 视图存在（如果所有选项卡都关闭了，它可能不存在），我们使用`history()`方法检索 web 视图的历史记录。

这个历史记录是一个`QWebEngineHistory`对象；这个对象是 web 页面对象的属性，用来跟踪浏览历史。当在 web 视图上调用`back()`和`forward()`槽时，会查询这个对象，找到正确的 URL 进行加载。历史对象的`items()`方法返回一个`QWebEngineHistoryItem`对象的列表，详细描述了 web 视图对象的整个浏览历史。

我们的`update_history`方法遍历这个列表，并为历史中的每个项目添加一个新的`QListWidgetItem`对象。请注意，我们使用列表小部件项的`setData()`方法，而不是`setText()`，因为它允许我们直接存储`QUrl`对象，而不必将其转换为字符串（`QListWidget`将自动将 URL 转换为字符串进行显示，使用 URL 的`toString()`方法）。

1.  除了在切换选项卡时调用此方法之外，我们还需要在 web 视图导航到新页面时调用它，以便在用户浏览时保持历史记录的最新状态。为了实现这一点，在`add_tab()`方法中为每个新生成的 web 视图添加一个连接：

```py
        webview.urlChanged.connect(self.update_history)
```

1.  为了完成我们的历史功能，我们希望能够双击历史中的项目并在当前打开的标签中导航到其 URL。我们将首先创建一个`MainWindow`方法来进行导航：

```py
    def navigate_history(self, item):
        qurl = item.data(qtc.Qt.DisplayRole)
        if self.tabs.currentWidget():
            self.tabs.currentWidget().load(qurl)
```

我们将使用`QListWidget`中的`itemDoubleClicked`信号来触发此方法，该方法将`QListItemWidget`对象传递给其回调。我们只需通过调用其`data()`访问器方法从列表项中检索 URL，然后将 URL 传递给当前可见的 web 视图。

1.  现在，回到`__init__()`，我们将连接信号到回调如下：

```py
        self.history_list.itemDoubleClicked.connect(
            self.navigate_history)
```

这完成了我们的历史功能；启动浏览器，您会发现可以使用停靠中的历史列表查看和导航。

# Web 设置

`QtWebEngine`浏览器，就像它所基于的 Chromium 浏览器一样，提供了一个非常可定制的网络体验；我们可以编辑许多设置来实现各种安全、功能或外观的更改。

为此，我们需要访问以下默认的`settings`对象：

```py
        settings = qtwe.QWebEngineSettings.defaultSettings()
```

`defaultSettings()`静态方法返回的`QWebEngineSettings`对象是一个全局对象，由程序中所有的 web 视图引用。我们不必（也不能）在更改后将其显式分配给 web 视图。一旦我们检索到它，我们可以以各种方式配置它，我们的设置将被所有我们创建的 web 视图所尊重。

例如，让我们稍微改变字体：

```py
        # The web needs more drama:
        settings.setFontFamily(
            qtwe.QWebEngineSettings.SansSerifFont, 'Impact')
```

在这种情况下，我们将所有无衬线字体的默认字体系列设置为`Impact`。除了设置字体系列，我们还可以设置默认的`fontSize`对象和`defaultTextEncoding`对象。

`settings`对象还具有许多属性，这些属性是布尔开关，我们可以切换；例如：

```py
        settings.setAttribute(
            qtwe.QWebEngineSettings.PluginsEnabled, True)
```

在这个例子中，我们启用了 Pepper API 插件的使用，例如 Chrome 的 Flash 实现。我们可以切换 29 个属性，以下是其中的一些示例：

| 属性 | 默认 | 描述 |
| --- | --- | --- |
| `JavascriptEnabled` | `True` | 允许运行 JavaScript 代码。 |
| `JavascriptCanOpenWindows` | `True` | 允许 JavaScript 打开新的弹出窗口。 |
| 全屏支持已启用 | 假 | 允许浏览器全屏显示。 |
| `AllowRunningInsecureContent` | `False` | 允许在 HTTPS 页面上运行 HTTP 内容。 |
| `PlaybackRequiresUserGesture` | `False` | 在用户与页面交互之前不要播放媒体。 |

要更改单个 web 视图的设置，请使用`page().settings()`访问其`QWebEnginSettings`对象。

# 构建文本搜索功能

到目前为止，我们已经在我们的 web 视图小部件中加载和显示了内容，但实际内容并没有做太多事情。我们通过`QtWebEngine`获得的强大功能之一是能够通过将我们自己的 JavaScript 代码注入到这些页面中来操纵网页的内容。为了看看这是如何工作的，我们将使用以下说明来开发一个文本搜索功能，该功能将突出显示搜索词的所有实例：

1.  我们将首先在`MainWindow.__init__()`中添加 GUI 组件：

```py
        find_dock = qtw.QDockWidget('Search')
        self.addDockWidget(qtc.Qt.BottomDockWidgetArea, find_dock)
        self.find_text = qtw.QLineEdit()
        find_dock.setWidget(self.find_text)
        self.find_text.textChanged.connect(self.text_search)
```

搜索小部件只是一个嵌入在停靠窗口中的`QLineEdit`对象。我们已经将`textChanged`信号连接到一个回调函数，该函数将执行搜索。

1.  为了实现搜索功能，我们需要编写一些 JavaScript 代码，以便为我们定位和突出显示搜索词的所有实例。我们可以将此代码添加为字符串，但为了清晰起见，让我们将其写在一个单独的文件中；打开一个名为`finder.js`的文件，并添加以下代码：

```py
function highlight_selection(){
    let tag = document.createElement('found');
    tag.style.backgroundColor = 'lightgreen';
    window.getSelection().getRangeAt(0).surroundContents(tag);}

function highlight_term(term){
    let found_tags = document.getElementsByTagName("found");
    while (found_tags.length > 0){
        found_tags[0].outerHTML = found_tags[0].innerHTML;}
    while (window.find(term)){highlight_selection();}
    while (window.find(term, false, true)){highlight_selection();}}
```

这本书不是一本 JavaScript 文本，所以我们不会深入讨论这段代码的工作原理，只是总结一下正在发生的事情：

+   1.  `highlight_term()`函数接受一个字符串作为搜索词。它首先清理任何 HTML`<found>`标签；这不是一个真正的标签——这是我们为了这个功能而发明的，这样它就不会与任何真正的标签冲突。

1.  然后该函数通过文档向前和向后搜索搜索词的实例。

1.  当它找到一个时，它会用背景颜色设置为浅绿色的`<found>`标签包裹它。

1.  回到`MainWindow.__init__()`，我们将读取这个文件并将其保存为一个实例变量：

```py
        with open('finder.js', 'r') as fh:
            self.finder_js = fh.read()
```

1.  现在，让我们在`MainWindow`下实现我们的搜索回调方法：

```py
    def text_search(self, term):
        term = term.replace('"', '')
        page = self.tabs.currentWidget().page()
        page.runJavaScript(self.finder_js)
        js = f'highlight_term("{term}");'
        page.runJavaScript(js)
```

在我们当前的网页视图中运行 JavaScript 代码，我们需要获取它的`QWebEnginePage`对象的引用。然后我们可以调用页面的`runJavaScript()`方法。这个方法简单地接受一个包含 JavaScript 代码的字符串，并在网页上执行它。

1.  在这种情况下，我们首先运行我们的`finder.js`文件的内容来设置函数，然后我们调用`highlight_term()`函数并插入搜索词。作为一个快速而粗糙的安全措施，我们还从搜索词中剥离了所有双引号；因此，它不能用于注入任意的 JavaScript。如果你现在运行应用程序，你应该能够在页面上搜索字符串，就像这样：

![](img/3e7dac13-b284-4ea5-ae91-54413221f4f9.png)

这个方法效果还不错，但是每次更新搜索词时重新定义这些函数并不是很有效，是吗？如果我们只定义这些函数一次，然后在我们导航到的任何页面上都可以访问它们，那就太好了。

1.  这可以使用`QWebEnginePage`对象的`scripts`属性来完成。这个属性存储了一个`QWebEngineScript`对象的集合，其中包含了每次加载新页面时要运行的 JavaScript 片段。通过将我们的脚本添加到这个集合中，我们可以确保我们的函数定义仅在每次页面加载时运行，而不是每次我们尝试搜索时都运行。为了使这个工作，我们将从`MainWindow.__init__()`开始，定义一个`QWebEngineScript`对象：

```py
        self.finder_script = qtwe.QWebEngineScript()
        self.finder_script.setSourceCode(self.finder_js)
```

1.  集合中的每个脚本都在 256 个**worlds**中的一个中运行，这些 worlds 是隔离的 JavaScript 上下文。为了在后续调用中访问我们的函数，我们需要确保我们的`script`对象通过设置它的`worldId`属性在主 world 中执行：

```py
        self.finder_script.setWorldId(qtwe.QWebEngineScript.MainWorld)
```

`QWebEngineScript.MainWorld`是一个常量，指向主 JavaScript 执行上下文。如果我们没有设置这个，我们的脚本会运行，但函数会在它们自己的 world 中运行，并且在网页上下文中不可用于搜索。

1.  现在我们有了我们的`script`对象，我们需要将它添加到网页对象中。这应该在`MainWindow.add_tab()`中完成，当我们创建我们的`page`对象时：

```py
        page.scripts().insert(self.finder_script)
```

1.  最后，我们可以缩短`text_search()`方法：

```py
    def text_search(self, term):
        page = self.tabs.currentWidget().page()
        js = f'highlight_term("{term}");'
        page.runJavaScript(js)
```

除了运行脚本，我们还可以从脚本中检索数据并将其发送到我们的 Python 代码中的回调方法。

例如，我们可以对我们的 JavaScript 进行以下更改，以从我们的函数中返回匹配项的数量：

```py
function highlight_term(term){
    //cleanup
    let found_tags = document.getElementsByTagName("found");
    while (found_tags.length > 0){
        found_tags[0].outerHTML = found_tags[0].innerHTML;}
    let matches = 0
    //search forward and backward
    while (window.find(term)){
        highlight_selection();
        matches++;
    }
    while (window.find(term, false, true)){
        highlight_selection();
        matches++;
    }
    return matches;
}
```

这个值*不*是从`runJavaScript()`返回的，因为 JavaScript 代码是异步执行的。

要访问返回值，我们需要将一个 Python 可调用的引用作为`runJavaScript()`的第二个参数传递；Qt 将调用该方法，并传递被调用代码的返回值：

```py
    def text_search(self, term):
        term = term.replace('"', '')
        page = self.tabs.currentWidget().page()
        js = f'highlight_term("{term}");'
        page.runJavaScript(js, self.match_count)
```

在这里，我们将 JavaScript 调用的输出传递给一个名为`match_count()`的方法，它看起来像下面的代码片段：

```py
    def match_count(self, count):
        if count:
            self.statusBar().showMessage(f'{count} matches ')
        else:
            self.statusBar().clearMessage()
```

在这种情况下，如果找到任何匹配项，我们将显示一个状态栏消息。再次尝试浏览器，你会看到消息应该成功传达。

# 总结

在本章中，我们探讨了`QtWebEngineWidgets`为我们提供的可能性。您实现了一个简单的浏览器，然后学习了如何利用浏览历史、配置文件共享、多个选项卡和常见设置等功能。您还学会了如何向网页注入任意 JavaScript 并检索这些调用的结果。

在下一章中，您将学习如何准备您的代码以进行共享、分发和部署。我们将讨论如何正确地构建项目目录结构，如何使用官方工具分发 Python 代码，以及如何使用 PyInstaller 为各种平台创建独立的可执行文件。

# 问题

尝试这些问题来测试您从本章中学到的知识：

1.  以下代码给出了一个属性错误；出了什么问题？

```py
   from PyQt5 import QtWebEngine as qtwe
   w = qtwe.QWebEngineView()
```

1.  以下代码应该将`UrlBar`类与`QWebEngineView`连接起来，以便在按下*return*/*Enter*键时加载输入的 URL。但是它不起作用；出了什么问题？

```py
   class UrlBar(qtw.QLineEdit):

       url_request = qtc.pyqtSignal(str)

       def __init__(self):
           super().__init__()
           self.returnPressed.connect(self.request)

       def request(self):
           self.url_request.emit(self.text())

   mywebview = qtwe.QWebEngineView()
   myurlbar = UrlBar()
   myurlbar.url_request(mywebview.load)
```

1.  以下代码的结果是什么？

```py
   class WebView(qtwe.QWebEngineView):

       def createWindow(self, _):

           return self
```

1.  查看[`doc.qt.io/qt-5/qwebengineview.html`](https://doc.qt.io/qt-5/qwebengineview.html)中的`QWebEngineView`文档。您将如何在浏览器中实现缩放功能？

1.  正如其名称所示，`QWebEngineView`代表了模型-视图架构中的视图部分。在这个设计中，哪个类代表了模型？

1.  给定一个名为`webview`的`QWebEngineView`对象，编写代码来确定`webview`上是否启用了 JavaScript。

1.  您在我们的浏览器示例中看到`runJavaScript()`可以将整数值传递给回调函数。编写一个简单的演示脚本来测试可以返回哪些其他类型的 JavaScript 对象，以及它们在 Python 代码中的表现方式。

# 进一步阅读

有关更多信息，请参考以下内容：

+   **QuteBrowser**是一个使用`QtWebEngineWidgets`用 Python 编写的开源网络浏览器。您可以在[`github.com/qutebrowser/qutebrowser`](https://github.com/qutebrowser/qutebrowser)找到其源代码。

+   **ADMBrowser**是一个基于`QtWebEngineWidgets`的浏览器，由本书的作者创建，并可用于信息亭系统。您可以在[`github.com/alandmoore/admbrowser`](https://github.com/alandmoore/admbrowser)找到它。

+   `QtWebChannel`是一个功能，允许您的 PyQt 应用程序与 Web 内容之间进行更强大的通信。您可以在[`doc.qt.io/qt-5/qtwebchannel-index.html`](https://doc.qt.io/qt-5/qtwebchannel-index.html)开始探索这一高级功能。
