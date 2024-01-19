# 使用 QTimer 和 QThread 进行多线程处理

尽管计算机硬件的功能不断增强，程序仍然经常需要执行需要几秒甚至几分钟才能完成的任务。虽然这种延迟可能是由于程序员无法控制的因素造成的，但它仍然会影响应用程序的性能，使其在后台任务运行时变得无响应。在本章中，我们将学习一些工具，可以帮助我们通过推迟重型操作或将其移出线程来保持应用程序的响应性。我们还将学习如何使用多线程应用程序设计来加快多核系统上的这些操作。

本章分为以下主题：

+   使用`QTimer`进行延迟操作

+   使用`QThread`进行多线程处理

+   使用`QThreadPool`和`QRunner`实现高并发

# 技术要求

本章只需要您在整本书中一直在使用的基本 Python 和 PyQt5 设置。您还可以参考[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter10`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter10)上的示例代码。

查看以下视频以查看代码的运行情况：[`bit.ly/2M6iSPl`](http://bit.ly/2M6iSPl)

# 使用 QTimer 进行延迟操作

在程序中能够延迟操作在各种情况下都是有用的。例如，假设我们想要一个无模式的**弹出**对话框，在定义的秒数后自动关闭，而不是等待用户点击按钮。

我们将从子类化`QDialog`开始：

```py
class AutoCloseDialog(qtw.QDialog):

    def __init__(self, parent, title, message, timeout):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle(title)
        self.setLayout(qtw.QVBoxLayout())
        self.layout().addWidget(qtw.QLabel(message))
        self.timeout = timeout
```

保存了一个`timeout`值后，我们现在想要重写对话框的`show()`方法，以便在指定的秒数后关闭它。

一个天真的方法可能是：

```py
    def show(self):
        super().show()
        from time import sleep
        sleep(self.timeout)
        self.hide()
```

Python 的`time.sleep()`函数将暂停程序执行我们传入的秒数。乍一看，它似乎应该做我们想要的事情，即显示窗口，暂停`timeout`秒，然后隐藏窗口。

因此，让我们在我们的`MainWindow.__init__()`方法中添加一些代码来测试它：

```py
        self.dialog = AutoCloseDialog(
            self,
            "Self-destructing message",
            "This message will self-destruct in 10 seconds",
            10
        )
        self.dialog.show()
```

如果运行程序，您会发现事情并不如预期。由于这个对话框是无模式的，它应该出现在我们的主窗口旁边，而不会阻塞任何东西。此外，由于我们在调用`sleep()`之前调用了`show()`，它应该在暂停之前显示自己。相反，您很可能得到一个空白和冻结的对话框窗口，它在其存在的整个期间都会暂停整个程序。那么，这里发生了什么？

从第一章 *PyQt 入门*中记得，Qt 程序有一个**事件循环**，当我们调用`QApplication.exec()`时启动。当我们调用`show()`这样的方法时，它涉及许多幕后操作，如绘制小部件和与窗口管理器通信，这些任务不会立即执行。相反，它们被放置在任务队列中。事件循环逐个处理任务队列中的工作，直到它为空。这个过程是**异步**的，因此调用`QWidget.show()`方法不会等待窗口显示后再返回；它只是将显示小部件的任务放在事件队列中并返回。

我们对`time.sleep()`方法的调用在程序中创建了一个立即阻塞的延迟，直到函数退出为止，这将停止所有其他处理。这包括停止 Qt 事件循环，这意味着所有仍在队列中的绘图操作都不会发生。事实上，直到`sleep()`完成，没有事件会被处理。这就是为什么小部件没有完全绘制，程序在`sleep()`执行时为什么没有继续的原因。

为了正确工作，我们需要将`hide()`调用放在事件循环中，这样我们对`AutoCloseDialog.show()`的调用可以立即返回，并让事件循环处理隐藏对话框，就像它处理显示对话框一样。但我们不想立即这样做，我们希望在事件队列上延迟执行一段时间。这就是`QtCore.QTimer`类可以为我们做的事情。

# 单发定时器

`QTimer`是一个简单的`QObject`子类，可以在一定时间后发出`timeout`信号。

使用`QTimer`延迟单个操作的最简单方法是使用`QTimer.singleShot()`静态方法，如下所示：

```py
    def show(self):
        super().show()
        qtc.QTimer.singleShot(self.timeout * 1000, self.hide)
```

`singleShot()`接受两个参数：毫秒为单位的间隔和回调函数。在这种情况下，我们在一定数量的`self.timeout`秒后调用`self.hide()`方法（我们将乘以 1,000 将其转换为毫秒）。

再次运行此脚本，您现在应该看到您的对话框表现如预期。

# 重复定时器

在应用程序中，有时我们需要在指定的间隔重复执行某个操作，比如自动保存文档，轮询网络套接字，或者不断地催促用户在应用商店给应用程序评 5 星（好吧，也许不是这个）。

`QTimer`也可以处理这个问题，您可以从以下代码块中看到：

```py
        interval_seconds = 10
        self.timer = qtc.QTimer()
        self.timer.setInterval(interval_seconds * 1000)
        self.interval_dialog = AutoCloseDialog(
            self, "It's time again",
            f"It has been {interval_seconds} seconds "
            "since this dialog was last shown.", 2000)
        self.timer.timeout.connect(self.interval_dialog.show)
        self.timer.start()
```

在这个例子中，我们明确创建了一个`QTimer`对象，而不是使用静态的`singleShot()`方法。然后，我们使用`setInterval()`方法配置了以毫秒为单位的超时间隔。当间隔过去时，定时器对象将发出`timeout`信号。默认情况下，`QTimer`对象将在达到指定间隔的末尾时重复发出`timeout`信号。您也可以使用`setSingleShot()`方法将其转换为单发，尽管一般来说，使用我们在*单发定时器*部分演示的静态方法更容易。

创建`QTimer`对象并配置间隔后，我们只需将其`timeout`信号连接到另一个`AutoCloseDialog`对象的`show()`方法，然后通过调用`start()`方法启动定时器。

我们也可以停止定时器，然后重新启动：

```py
        toolbar = self.addToolBar('Tools')
        toolbar.addAction('Stop Bugging Me', self.timer.stop)
        toolbar.addAction('Start Bugging Me', self.timer.start)
```

`QTimer.stop()`方法停止定时器，`start()`方法将重新开始。值得注意的是这里没有`pause()`方法；`stop()`方法将清除任何当前的进度，`start()`方法将从配置的间隔重新开始。

# 从定时器获取信息

`QTimer`有一些方法，我们可以用来提取有关定时器状态的信息。例如，让我们通过以下代码让用户了解事情的进展：

```py
        self.timer2 = qtc.QTimer()
        self.timer2.setInterval(1000)
        self.timer2.timeout.connect(self.update_status)
        self.timer2.start()
```

我们设置了另一个定时器，它将每秒调用`self.update_status()`。`update_status()`然后查询信息的第一次如下：

```py
    def update_status(self):
        if self.timer.isActive():
            time_left = (self.timer.remainingTime() // 1000) + 1
            self.statusBar().showMessage(
                f"Next dialog will be shown in {time_left} seconds.")
        else:
            self.statusBar().showMessage('Dialogs are off.')
```

`QTimer.isActive()`方法告诉我们定时器当前是否正在运行，而`remainingTime()`告诉我们距离下一个`timeout`信号还有多少毫秒。

现在运行这个程序，您应该看到关于下一个对话框的状态更新。

# 定时器的限制

虽然定时器允许我们将操作推迟到事件队列，并可以帮助防止程序中的尴尬暂停，但重要的是要理解连接到`timeout`信号的函数仍然在主执行线程中执行，并且因此会阻塞主执行线程。

例如，假设我们有一个长时间阻塞的方法，如下所示：

```py
    def long_blocking_callback(self):
        from time import sleep
        self.statusBar().showMessage('Beginning a long blocking function.')
        sleep(30)
        self.statusBar().showMessage('Ending a long blocking function.')
```

您可能认为从单发定时器调用此方法将阻止其锁定应用程序。让我们通过将此代码添加到`MainView.__init__()`来测试这个理论：

```py
        qtc.QTimer.singleShot(1, self.long_blocking_callback)
```

使用`1`毫秒延迟调用`singleShot()`是安排一个几乎立即发生的事件的简单方法。那么，它有效吗？

好吧，实际上并不是这样；如果你运行程序，你会发现它会锁定 30 秒。尽管我们推迟了操作，但它仍然是一个长时间的阻塞操作，会在运行时冻结程序。也许我们可以调整延迟值，以确保它被推迟到更合适的时刻（比如在应用程序绘制完毕后或者在启动画面显示后），但迟早，应用程序将不得不冻结并在任务运行时变得无响应。

然而，对于这样的问题有一个解决方案；在下一节*使用 QThread 进行多线程处理*中，我们将看看如何将这样的繁重阻塞任务推送到另一个线程，以便我们的程序可以继续运行而不会冻结。

# 使用 QThread 进行多线程处理

等待有时是不可避免的。无论是查询网络、访问文件系统还是运行复杂的计算，有时程序只是需要时间来完成一个过程。然而，在等待的时候，我们的 GUI 没有理由完全变得无响应。具有多个 CPU 核心和线程技术的现代系统允许我们运行并发进程，我们没有理由不利用这一点来制作响应式的 GUI。尽管 Python 有自己的线程库，但 Qt 为我们提供了`QThread`对象，可以轻松构建多线程应用程序。它还有一个额外的优势，就是集成到 Qt 中，并且与信号和槽兼容。

在本节中，我们将构建一个相对缓慢的文件搜索工具，然后使用`QThread`来确保 GUI 保持响应。

# SlowSearcher 文件搜索引擎

为了有效地讨论线程，我们首先需要一个可以在单独线程上运行的缓慢过程。打开一个新的 Qt 应用程序模板副本，并将其命名为`file_searcher.py`。

让我们开始实现一个文件搜索引擎：

```py
class SlowSearcher(qtc.QObject):

    match_found = qtc.pyqtSignal(str)
    directory_changed = qtc.pyqtSignal(str)
    finished = qtc.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.term = None
```

我们将其称为`SlowSearcher`，因为它将是故意非优化的。它首先定义了一些信号，如下所示：

+   当文件名与搜索项匹配时，将发出`match_found`信号，并包含匹配的文件名

+   每当我们开始在一个新目录中搜索时，将发出`directory_changed`信号

+   当整个文件系统树已经被搜索时，将发出`finished`信号

最后，我们重写`__init__()`只是为了定义一个名为`self.term`的实例变量。

接下来，我们将为`term`创建一个 setter 方法：

```py
    def set_term(self, term):
        self.term = term
```

如果你想知道为什么我们要费力实现一个如此简单的 setter 方法，而不是直接设置变量，这个原因很快就会显而易见，当我们讨论`QThread`的一些限制时，这个原因将很快显现出来。

现在，我们将创建搜索方法，如下所示：

```py
    def do_search(self):
        root = qtc.QDir.rootPath()
        self._search(self.term, root)
        self.finished.emit()
```

这个方法将是我们调用来启动搜索过程的槽。它首先将根目录定位为一个`QDir`对象，然后调用`_search()`方法。一旦`_search()`返回，它就会发出`finished`信号。

实际的`_search()`方法如下：

```py
    def _search(self, term, path):
        self.directory_changed.emit(path)
        directory = qtc.QDir(path)
        directory.setFilter(directory.filter() |
            qtc.QDir.NoDotAndDotDot | qtc.QDir.NoSymLinks)
        for entry in directory.entryInfoList():
            if term in entry.filePath():
                print(entry.filePath())
                self.match_found.emit(entry.filePath())
            if entry.isDir():
                self._search(term, entry.filePath())
```

`_search()`是一个递归搜索方法。它首先发出`directory_changed`信号，表示我们正在一个新目录中搜索，然后为当前路径创建一个`QDir`对象。接下来，它设置`filter`属性，以便在查询`entryInfoList()`方法时，不包括符号链接或`.`和`..`快捷方式（这是为了避免搜索中的无限循环）。最后，我们遍历`entryInfoList()`检索到的目录内容，并为每个匹配的项目发出`match_found`信号。对于每个找到的目录，我们在其上运行`_search()`方法。

这样，我们的方法将递归遍历文件系统中的所有目录，寻找与我们的搜索词匹配的内容。这不是最优化的方法，这是故意这样做的。根据您的硬件、平台和驱动器上的文件数量，这个搜索可能需要几秒钟到几分钟的时间才能完成，因此它非常适合查看线程如何帮助必须执行缓慢进程的应用程序。

在多线程术语中，执行实际工作的类被称为`Worker`类。`SlowSearcher`是`Worker`类的一个示例。

# 一个非线程化的搜索器

为了实现一个搜索应用程序，让我们添加一个用于输入搜索词和显示搜索结果的 GUI 表单。

让我们称它为`SearchForm`，如下所示：

```py
class SearchForm(qtw.QWidget):

    textChanged = qtc.pyqtSignal(str)
    returnPressed = qtc.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setLayout(qtw.QVBoxLayout())
        self.search_term_inp = qtw.QLineEdit(
            placeholderText='Search Term',
            textChanged=self.textChanged,
            returnPressed=self.returnPressed)
        self.layout().addWidget(self.search_term_inp)
        self.results = qtw.QListWidget()
        self.layout().addWidget(self.results)
        self.returnPressed.connect(self.results.clear)
```

这个 GUI 只包含一个用于输入搜索词的`QLineEdit`小部件和一个用于显示结果的`QListWidget`小部件。我们将`QLineEdit`小部件的`returnPressed`和`textChanged`信号转发到`SearchForm`对象上的同名信号，以便我们可以更容易地在我们的`MainView`方法中连接它们。我们还将`returnPressed`连接到列表小部件的`clear`槽，以便开始新搜索时清除结果区域。

`SearchForm()`方法还需要一个方法来添加新项目：

```py
    def addResult(self, result):
        self.results.addItem(result)
```

这只是一个方便的方法，这样一来，主应用程序就不必直接操作表单中的小部件。

在我们的`MainWindow.__init__()`方法中，我们可以创建一个搜索器和表单对象，并将它们连接起来，如下所示：

```py
        form = SearchForm()
        self.setCentralWidget(form)
        self.ss = SlowSearcher()
        form.textChanged.connect(self.ss.set_term)
        form.returnPressed.connect(self.ss.do_search)
        self.ss.match_found.connect(form.addResult)
```

创建`SlowSearcher`和`SearchForm`对象并将表单设置为中央部件后，我们将适当的信号连接在一起，如下所示：

+   表单的`textChanged`信号，发出输入的字符串，连接到搜索器的`set_term()`设置方法。

+   表单的`returnPressed`信号连接到搜索器的`do_search()`方法以触发搜索。

+   搜索器的`match_found`信号，携带找到的路径名，连接到表单的`addResult()`方法。

最后，让我们添加两个`MainWindow`方法，以便让用户了解搜索的状态：

```py
    def on_finished(self):
        qtw.QMessageBox.information(self, 'Complete', 'Search complete')

    def on_directory_changed(self, path):
        self.statusBar().showMessage(f'Searching in: {path}')
```

第一个将显示一个指示搜索已完成的状态，而第二个将显示一个指示搜索器正在搜索的当前路径的状态。

回到`__init__()`，这些将连接到搜索器，如下所示：

```py
        self.ss.finished.connect(self.on_finished)
        self.ss.directory_changed.connect(self.on_directory_changed)
```

# 测试我们的非线程化搜索应用程序

我们对这个脚本的期望是，当我们在系统中搜索目录时，我们将在结果区域得到稳定的搜索结果打印输出，同时状态栏中的当前目录也会不断更新。

然而，如果您运行它，您会发现实际发生的并不是这样。相反，一旦搜索开始，GUI 就会冻结。状态栏中什么都没有显示，列表小部件中也没有条目出现，尽管匹配项已经打印到控制台上。只有当搜索最终完成时，结果才会出现，状态才会更新。

为了解决这个问题，我们需要引入线程。

那么，为什么程序会实时打印到控制台，但不会实时更新我们的 GUI 呢？这是因为`print()`是同步的——它在调用时立即执行，并且直到文本被写入控制台后才返回。然而，我们的 GUI 方法是异步的——它们被排队在 Qt 事件队列中，并且直到主事件循环执行`SlowSearcher.search()`方法后才会执行。

# 添加线程

**线程**是独立的代码执行上下文。默认情况下，我们所有的代码都在一个线程中运行，因此我们将其称为**单线程**应用程序。使用`QtCore.QThread`类，我们可以创建新的线程并将代码的部分移动到这些线程中，使其成为**多线程**应用程序。

您可以使用`QThread`对象，如下所示：

```py
        self.searcher_thread = qtc.QThread()
        self.ss.moveToThread(self.searcher_thread)
        self.ss.finished.connect(self.searcher_thread.quit)
        self.searcher_thread.start()
```

我们首先创建一个`QThread`对象，然后使用`SlowSearcher.moveToThread()`方法将我们的`SlowSearcher`对象移动到新线程中。`moveToThread()`是`QObject`的一个方法，由任何子类`QObject`的类继承。

接下来，我们将搜索器的`finished`信号连接到线程的`quit`槽；这将导致线程在搜索完成时停止执行。由于搜索线程不是我们主要的执行线程的一部分，它必须有一种方法来自行退出，否则在搜索结束后它将继续运行。

最后，我们需要调用搜索线程的`start()`方法来开始执行代码，并允许我们的主线程与`SlowSearcher`对象交互。

这段代码需要在创建`SlowSearcher`对象之后插入，但在连接到它的任何信号或槽之前（我们将在*线程提示和注意事项*部分讨论原因）。

由于我们在每次搜索后都要退出线程，所以需要在每次开始新搜索时重新启动线程。我们可以通过以下连接来实现这一点：

```py
        form.returnPressed.connect(self.searcher_thread.start)
```

这就是使用线程所需的一切。再次运行脚本，你会看到随着搜索的进行，GUI 会更新。

让我们总结一下这个过程，如下所示：

1.  创建`Worker`类的实例

1.  创建一个`QThread`对象

1.  使用`Worker`类的`moveToThread()`方法将其移动到新线程

1.  连接任何其他信号和槽

1.  调用线程的`start()`方法

# 另一种方法

虽然`moveToThread()`方法是使用`QThread`的推荐方法，但还有另一种方法可以完全正常地工作，并且在某种程度上简化了我们的代码。这种方法是通过对`QThread`进行子类化并重写`run()`方法来创建我们的`Worker`类，使用我们的工作代码。

例如，创建`SlowSearcher`的副本，并进行如下修改：

```py
class SlowSearcherThread(qtc.QThread):
    # rename "do_search()" to "run()":

    def run (self):
        root = qtc.QDir.rootPath()
        self._search(self.term, root)
        self.finished.emit()

    # The rest of the class is the same
```

在这里，我们只改变了三件事：

+   我们已将类重命名为`SlowSearcherThread`。

+   我们已将父类更改为`QThread`。

+   我们已经将`do_search()`重命名为`run()`。

我们的`MainWindow.__init__()`方法现在会简单得多：

```py
        form = SearchForm()
        self.setCentralWidget(form)
        self.ss = SlowSearcherThread()
        form.textChanged.connect(self.ss.set_term)
        form.returnPressed.connect(self.ss.start)
        self.ss.match_found.connect(form.addResult)
        self.ss.finished.connect(self.on_finished)
        self.ss.directory_changed.connect(self.on_directory_changed)
```

现在，我们只需要将`returnPressed`连接到`SlowSearcher.start()`。`start()`方法创建了新线程，并在新线程中执行对象的`run()`方法。这意味着，通过重写该方法，我们可以有效地将该代码放在一个新线程中。

始终记得实现`run()`，但调用`start()`。不要搞混了，否则你的多线程就无法工作！

虽然这种方法有一些有效的用例，但它可能会在对象数据的线程所有权上产生微妙的问题。即使`QThread`对象为辅助线程提供了控制接口，但对象本身仍然存在于主线程中。当我们在`worker`对象上调用`moveToThread()`时，我们可以确保`worker`对象完全移动到新线程中。然而，当`worker`对象是`QThread`的子类时，`QThread`的部分必须保留在主线程中，即使执行的代码被移动到新线程中。这可能会导致微妙的错误，因为很难搞清楚`worker`对象的哪些部分在哪个线程中。

最终，除非你有清晰的理由来对`QThread5`进行子类化，否则应该使用`moveToThread()`。

# 线程的提示和注意事项

之前的示例可能让多线程编程看起来很简单，但那是因为代码经过精心设计，避免了在处理线程时可能出现的一些问题。实际上，在单线程应用程序上进行多线程改造可能会更加困难。

一个常见的问题是`worker`对象在主线程中被卡住，导致我们失去了多线程的好处。这可能以几种方式发生。

例如，在我们原始的线程脚本（使用`moveToThread()`的脚本）中，我们必须在连接任何信号之前将工作线程移动到线程中。如果您尝试在信号连接之后移动线程代码，您会发现 GUI 会锁定，就好像您没有使用线程一样。

发生这种情况的原因是我们的工作线程方法是 Python 方法，并且连接到它们会在 Python 中创建一个连接，这个连接必须在主线程中持续存在。解决这个问题的一种方法是使用`pyqtSlot()`装饰器将工作线程的方法转换为真正的 Qt 槽，如下所示：

```py
    @qtc.pyqtSlot(str)
    def set_term(self, term):
        self.term = term

    @qtc.pyqtSlot()
    def do_search(self):
        root = qtc.QDir.rootPath()
        self._search(self.term, root)
        self.finished.emit()
```

一旦您这样做了，顺序就不重要了，因为连接将完全存在于 Qt 对象之间，而不是 Python 对象之间。

您还可以通过在主线程中直接调用`worker`对象的一个方法来捕获`worker`对象：

```py
        # in MainView__init__():
        self.ss.set_term('foo')
        self.ss.do_search()
```

将上述行放在`__init__()`中将导致 GUI 保持隐藏，直到对`foo`进行的文件系统搜索完成。有时，这个问题可能会很微妙；例如，以下`lambda`回调表明我们只是将信号直接连接到槽：

```py
        form.returnPressed.connect(lambda: self.ss.do_search())
```

然而，这种连接会破坏线程，因为`lambda`函数本身是主线程的一部分，因此对`search()`的调用将在主线程中执行。

不幸的是，这个限制也意味着您不能将`MainWindow`方法用作调用工作方法的槽；例如，我们不能在`MainWindow`中运行以下代码：

```py
    def on_return_pressed(self):
        self.searcher_thread.start()
        self.ss.do_search()
```

将其作为`returnPressed`的回调，而不是将信号连接到`worker`对象的方法，会导致线程失败和 GUI 锁定。

简而言之，最好将与`worker`对象的交互限制为纯 Qt 信号和槽连接，没有中间函数。

# 使用 QThreadPool 和 QRunner 进行高并发

`QThreads`非常适合将单个长时间的进程放入后台，特别是当我们希望使用信号和槽与该进程进行通信时。然而，有时我们需要做的是使用尽可能多的线程并行运行多个计算密集型操作。这可以通过`QThread`来实现，但更好的选择是在`QThreadPool`和`QRunner`中找到。

`QRunner`代表我们希望工作线程执行的单个可运行任务。与`QThread`不同，它不是从`QObject`派生的，也不能使用信号和槽。然而，它非常高效，并且在需要多个线程时使用起来更简单。

`QThreadPool`对象的工作是管理`QRunner`对象的队列，当计算资源可用时，启动新线程来执行对象。

为了演示如何使用这个，让我们构建一个文件哈希实用程序。

# 文件哈希 GUI

我们的文件哈希工具将接受一个源目录、一个目标文件和要使用的线程数。它将使用线程数来计算目录中每个文件的 MD5 哈希值，然后在执行此操作时将信息写入目标文件。

诸如 MD5 之类的**哈希函数**用于从任意数据计算出唯一的固定长度的二进制值。哈希经常用于确定文件的真实性，因为对文件的任何更改都会导致不同的哈希值。

从第四章中制作一个干净的 Qt 模板的副本，*使用 QMainWindow 构建应用程序*，将其命名为`hasher.py`。

然后，我们将从我们的 GUI 表单类开始，如下所示：

```py
class HashForm(qtw.QWidget):

    submitted = qtc.pyqtSignal(str, str, int)

    def __init__(self):
        super().__init__()
        self.setLayout(qtw.QFormLayout())
        self.source_path = qtw.QPushButton(
            'Click to select…', clicked=self.on_source_click)
        self.layout().addRow('Source Path', self.source_path)
        self.destination_file = qtw.QPushButton(
            'Click to select…', clicked=self.on_dest_click)
        self.layout().addRow('Destination File', self.destination_file)
        self.threads = qtw.QSpinBox(minimum=1, maximum=7, value=2)
        self.layout().addRow('Threads', self.threads)
        submit = qtw.QPushButton('Go', clicked=self.on_submit)
        self.layout().addRow(submit)
```

这种形式与我们在前几章设计的形式非常相似，有一个`submitted`信号来发布数据，`QPushButton`对象来存储选定的文件，一个旋转框来选择线程的数量，以及另一个按钮来提交表单。

文件按钮的回调将如下所示：

```py
    def on_source_click(self):
        dirname = qtw.QFileDialog.getExistingDirectory()
        if dirname:
            self.source_path.setText(dirname)

    def on_dest_click(self):
        filename, _ = qtw.QFileDialog.getSaveFileName()
        if filename:
            self.destination_file.setText(filename)
```

在这里，我们使用`QFileDialog`静态函数（你在第五章中学到的，*使用模型视图类创建数据接口*）来检索要检查的目录名称和我们将用来保存输出的文件名。

最后，我们的`on_submit()`回调如下：

```py
    def on_submit(self):
        self.submitted.emit(
            self.source_path.text(),
            self.destination_file.text(),
            self.threads.value()
        )
```

这个回调只是简单地从我们的小部件中收集数据，并使用`submitted`信号发布它。

在`MainWindow.__init__()`中，创建一个表单并将其设置为中央小部件：

```py
        form = HashForm()
        self.setCentralWidget(form)
```

这样我们的 GUI 就完成了，现在让我们来构建后端。

# 哈希运行器

`HashRunner`类将表示我们要执行的实际任务的单个实例。对于我们需要处理的每个文件，我们将创建一个唯一的`HashRunner`实例，因此它的构造函数将需要接收输入文件名和输出文件名作为参数。它的任务将是计算输入文件的 MD5 哈希，并将其与输入文件名一起追加到输出文件中。

我们将通过子类化`QRunnable`来启动它：

```py
class HashRunner(qtc.QRunnable):

    file_lock = qtc.QMutex()
```

我们首先创建一个`QMutex`对象。在多线程术语中，**互斥锁**是一个在线程之间共享的可以被锁定或解锁的对象。

你可以将互斥锁看作是单用户洗手间的门的方式；假设 Bob 试图进入洗手间并锁上门。如果 Alice 已经在洗手间里，那么门不会打开，Bob 将不得不耐心地等待，直到 Alice 解锁门并离开洗手间。然后，Bob 才能进入并锁上门。

同样，当一个线程尝试锁定另一个线程已经锁定的互斥锁时，它必须等到第一个线程完成并解锁互斥锁，然后才能获取锁。

在`HashRunner`中，我们将使用我们的`file_lock`互斥锁来确保两个线程不会同时尝试写入输出文件。请注意，该对象是在类定义中创建的，因此它将被`HashRunner`的所有实例共享。

现在，让我们创建`__init__()`方法：

```py
    def __init__(self, infile, outfile):
        super().__init__()
        self.infile = infile
        self.outfile = outfile
        self.hasher = qtc.QCryptographicHash(
            qtc.QCryptographicHash.Md5)
        self.setAutoDelete(True)
```

该对象将接收输入文件和输出文件的路径，并将它们存储为实例变量。它还创建了一个`QtCore.QCryptographicHash`的实例。这个对象能够计算数据的各种加密哈希，比如 MD5、SHA-256 或 Keccak-512。这个类支持的哈希的完整列表可以在[`doc.qt.io/qt-5/qcryptographichash.html`](https://doc.qt.io/qt-5/qcryptographichash.html)找到。

最后，我们将类的`autoDelete`属性设置为`True`。`QRunnable`的这个属性将导致对象在`run()`方法返回时被删除，节省我们的内存和资源。

运行器执行的实际工作在`run()`方法中定义：

```py
    def run(self):
        print(f'hashing {self.infile}')
        self.hasher.reset()
        with open(self.infile, 'rb') as fh:
            self.hasher.addData(fh.read())
        hash_string = bytes(self.hasher.result().toHex()).decode('UTF-8')
```

我们的函数首先通过打印一条消息到控制台并重置`QCryptographicHash`对象来开始，清除其中可能存在的任何数据。

然后，我们使用`addData()`方法将文件的二进制内容读入哈希对象中。可以使用`result()`方法从哈希对象中计算和检索哈希值作为`QByteArray`对象。然后，我们使用`toHex()`方法将字节数组转换为十六进制字符串，然后通过`bytes`对象将其转换为 Python Unicode 字符串。

现在，我们只需要将这个哈希字符串写入输出文件。这就是我们的互斥锁对象发挥作用的地方。

传统上，使用互斥锁的方式如下：

```py
        try:
            self.file_lock.lock()
            with open(self.outfile, 'a', encoding='utf-8') as out:
                out.write(f'{self.infile}\t{hash_string}\n')
        finally:
            self.file_lock.unlock()
```

我们在`try`块内调用互斥锁的`lock()`方法，然后执行我们的文件操作。在`finally`块内，我们调用`unlock`方法。之所以在`try`和`finally`块内执行这些操作，是为了确保即使`file`方法出现问题，互斥锁也一定会被释放。

然而，在 Python 中，每当我们有像这样具有初始化和清理代码的操作时，最好使用**上下文管理器**对象与`with`关键字结合使用。PyQt 为我们提供了这样的对象：`QMutexLocker`。

我们可以像下面这样使用这个对象：

```py
        with qtc.QMutexLocker(self.file_lock):
            with open(self.outfile, 'a', encoding='utf-8') as out:
                out.write(f'{self.infile}\t{hash_string}\n')
```

这种方法更加清晰。通过使用互斥上下文管理器，我们确保`with`块内的任何操作只由一个线程执行，其他线程将等待直到对象完成。

# 创建线程池

这个应用程序的最后一部分将是一个`HashManager`对象。这个对象的工作是接收表单输出，找到要进行哈希处理的文件，然后为每个文件启动一个`HashRunner`对象。

它将开始像这样：

```py
class HashManager(qtc.QObject):

    finished = qtc.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.pool = qtc.QThreadPool.globalInstance()
```

我们基于`QObject`类，这样我们就可以定义一个`finished`信号。当所有的运行者完成他们的任务时，这个信号将被发射。

在构造函数中，我们创建了`QThreadPool`对象。但是，我们使用`globalInstance()`静态方法来访问每个 Qt 应用程序中已经存在的全局线程池对象，而不是创建一个新对象。你不必这样做，但对于大多数应用程序来说已经足够了，并且消除了涉及多个线程池的一些复杂性。

这个类的真正工作将在一个我们将称之为`do_hashing`的方法中发生：

```py
    @qtc.pyqtSlot(str, str, int)
    def do_hashing(self, source, destination, threads):
        self.pool.setMaxThreadCount(threads)
        qdir = qtc.QDir(source)
        for filename in qdir.entryList(qtc.QDir.Files):
            filepath = qdir.absoluteFilePath(filename)
            runner = HashRunner(filepath, destination)
            self.pool.start(runner)
```

这个方法被设计为直接连接到`HashForm.submitted`信号，所以我们将它作为一个槽与匹配的信号。它首先通过将线程池的最大线程数（由`maxThreadCount`属性定义）设置为函数调用中接收到的数字。一旦设置了这个值，我们可以在线程池中排队任意数量的`QRunnable`对象，但只有`maxThreadCount`个线程会同时启动。

接下来，我们将使用`QDir`对象的`entryList()`方法来遍历目录中的文件，并为每个文件创建一个`HashRunner`对象。然后将运行对象传递给线程池的`start()`方法，将其添加到池的工作队列中。

在这一点上，我们所有的运行者都在单独的执行线程中运行，但是当它们完成时，我们想发射一个信号。不幸的是，`QThreadPool`中没有内置的信号告诉我们这一点，但`waitForDone()`方法将继续阻塞，直到所有线程都完成。

因此，将以下代码添加到`do_hashing()`中：

```py
        self.pool.waitForDone()
        self.finished.emit()
```

回到`MainWindow.__init__()`，让我们创建我们的管理器对象并添加我们的连接：

```py
        self.manager = HashManager()
        self.manager_thread = qtc.QThread()
        self.manager.moveToThread(self.manager_thread)
        self.manager_thread.start()
        form.submitted.connect(self.manager.do_hashing)
```

创建了我们的`HashManager`之后，我们使用`moveToThread()`将其移动到一个单独的线程中。这是因为我们的`do_hashing()`方法将阻塞，直到所有的运行者都完成，而我们不希望 GUI 在等待时冻结。如果我们省略了`do_hashing()`的最后两行，这是不必要的（但我们也永远不会知道何时完成）。

为了获得发生的反馈，让我们添加两个更多的连接：

```py
        form.submitted.connect(
            lambda x, y, z: self.statusBar().showMessage(
                f'Processing files in {x} into {y} with {z} threads.'))
        self.manager.finished.connect(
            lambda: self.statusBar().showMessage('Finished'))
```

第一个连接将在表单提交时设置状态，指示即将开始的工作的详细信息；第二个连接将在工作完成时通知我们。

# 测试脚本

继续启动这个脚本，让我们看看它是如何工作的。将源目录指向一个充满大文件的文件夹，比如 DVD 镜像、存档文件或视频文件。将线程的旋钮保持在默认设置，并点击`Go`。

从控制台输出中可以看到，文件正在一次处理两个。一旦一个完成，另一个就开始，直到所有文件都被处理完。

再试一次，但这次将线程数增加到四或五。注意到更多的文件正在同时处理。当您调整这个值时，您可能也会注意到有一个收益递减的点，特别是当您接近 CPU 核心数时。这是关于并行化的一个重要教训——有时候，过多会导致性能下降。

# 线程和 Python GIL

在 Python 中，没有讨论多线程是完整的，而不涉及全局解释器锁（GIL）。GIL 是官方 Python 实现（CPython）中内存管理系统的一部分。本质上，它就像我们在`HashRunner`类中使用的互斥锁一样——就像`HashRunner`类必须在写入输出之前获取`file_lock`互斥锁一样，Python 应用程序中的任何线程在执行任何 Python 代码之前必须获取 GIL。换句话说，一次只有一个线程可以执行 Python 代码。

乍一看，这可能会使 Python 中的多线程看起来是徒劳的；毕竟，如果只有一个线程可以一次执行 Python 代码，那么创建多个线程有什么意义呢？

答案涉及 GIL 要求的两个例外情况：

+   长时间运行的代码可以是 CPU 绑定或 I/O 绑定。CPU 绑定意味着大部分处理时间都用于运行繁重的 CPU 操作，比如加密哈希。I/O 绑定操作是指大部分时间都花在等待输入/输出调用上，比如将大文件写入磁盘或从网络套接字读取数据。当线程进行 I/O 调用并开始等待响应时，它会释放 GIL。因此，如果我们的工作代码大部分是 I/O 绑定的，我们可以从多线程中受益，因为在等待 I/O 操作完成时，其他代码可以运行。

+   如果 CPU 绑定的代码在 Python 之外运行，则会释放 GIL。换句话说，如果我们使用 C 或 C++函数或对象执行 CPU 绑定操作，那么 GIL 会被释放，只有在下一个 Python 操作运行时才重新获取。

这就是为什么我们的`HashRunner`起作用的原因；它的两个最重的操作如下：

+   从磁盘读取大文件（这是一个 I/O 绑定操作）

+   对文件内容进行哈希处理（这是在`QCryptographicHash`对象内部处理的——这是一个在 Python 之外运行的 C++对象）

如果我们要在纯 Python 中实现一个哈希算法，那么我们很可能会发现我们的多线程代码实际上比单线程实现还要慢。

最终，多线程并不是 Python 中加速代码的魔法子弹；必须仔细规划，以避免与 GIL 和我们在“线程提示和注意事项”部分讨论的陷阱有关的问题。然而，经过适当的关怀，它可以帮助我们创建快速响应的程序。

# 总结

在本章中，您学会了如何在运行缓慢的代码时保持应用程序的响应性。您学会了如何使用`QTimer`将操作推迟到以后的时间，无论是作为一次性操作还是重复操作。您学会了如何使用`QThread`将代码推送到另一个线程，既可以使用`moveToThread()`也可以通过子类化`QThread`。最后，您学会了如何使用`QThreadPool`和`QRunnable`来构建高度并发的数据处理应用程序。

在第十一章中，“使用 QTextDocument 创建丰富的文本”，我们将看看如何在 PyQt 中处理丰富的文本。您将学会如何使用类似 HTML 的标记定义丰富的文本，以及如何使用`QDocument`API 检查和操作文档。您还将学会如何利用 Qt 的打印支持将文档带入现实世界。

# 问题

尝试回答这些问题，以测试你从本章学到的知识：

1.  创建代码以每 10 秒调用`self.every_ten_seconds()`方法。

1.  以下代码错误地使用了`QTimer`。你能修复它吗？

```py
   timer = qtc.QTimer()
   timer.setSingleShot(True)
   timer.setInterval(1000)
   timer.start()
   while timer.remainingTime():
       sleep(.01)
   run_delayed_command()
```

1.  您已经创建了以下单词计数的`Worker`类，并希望将其移动到另一个线程以防止大型文档减慢 GUI。但它没有起作用——你需要改变这个类的什么？

```py
   class Worker(qtc.QObject):

    counted = qtc.pyqtSignal(int)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def count_words(self):
        content = self.parent.textedit.toPlainText()
        self.counted.emit(len(content.split()))
```

1.  以下代码是阻塞的，而不是在单独的线程中运行。为什么会这样？

```py
   class Worker(qtc.QThread):

       def set_data(data):
           self.data = data

       def run(self):n
           start_complex_calculations(self.data)

    class MainWindow(qtw.QMainWindow):

        def __init__(self):
            super().__init__()
            form = qtw.QWidget()
            self.setCentralWidget(form)
            form.setLayout(qtw.QFormLayout())

            worker = Worker()
            line_edit = qtw.QLineEdit(textChanged=worker.set_data)
            button = qtw.QPushButton('Run', clicked=worker.run)
            form.layout().addRow('Data:', line_edit)
            form.layout().addRow(button)
            self.show()
```

1.  这个`Worker`类会正确运行吗？如果不会，为什么？

```py
   class Worker(qtc.QRunnable):

       finished = qtc.pyqtSignal()

       def run(self):
           calculate_navigation_vectors(30)
           self.finished.emit()
```

1.  以下代码是设计用于处理科学设备输出的大型数据文件的`QRunnable`类的`run()`方法。这些文件包含数百万行以空格分隔的长数字。这段代码可能会受到 Python GIL 的影响吗？您能否减少 GIL 的干扰？

```py
       def run(self):
           with open(self.file, 'r') as fh:
               for row in fh:
                   numbers = [float(x) for x in row.split()]
                   if numbers:
                       mean = sum(numbers) / len(numbers)
                       numbers.append(mean)
                   self.queue.put(numbers)
```

1.  以下是您正在编写的多线程 TCP 服务器应用程序中`QRunnable`类的`run()`方法。所有线程共享通过`self.datastream`访问的服务器套接字实例。然而，这段代码不是线程安全的。您需要做什么来修复它？

```py
       def run(self):
           message = get_http_response_string()
           message_len = len(message)
           self.datastream.writeUInt32(message_len)
           self.datastream.writeQString(message)
```

# 进一步阅读

欲了解更多信息，请参考以下内容：

+   信号量类似于互斥锁，但允许获取任意数量的锁，而不仅仅是单个锁。您可以在[`doc.qt.io/qt-5/qsemaphore.html`](https://doc.qt.io/qt-5/qsemaphore.html)了解更多关于 Qt 实现的`QSemaphore`类的信息。

+   David Beazley 在 PyCon 2010 的演讲提供了更深入的了解 Python GIL 的运作，可在[`www.youtube.com/watch?v=Obt-vMVdM8s`](https://www.youtube.com/watch?v=Obt-vMVdM8s)上观看。
