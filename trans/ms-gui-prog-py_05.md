# 使用QMainWindow构建应用程序

基本的Qt小部件可以在构建简单表单时带我们走很远，但完整的应用程序包括诸如菜单、工具栏、对话框等功能，这些功能可能很繁琐和棘手，从头开始构建。幸运的是，PyQt为这些标准组件提供了现成的类，使构建应用程序相对轻松。

在本章中，我们将探讨以下主题：

+   `QMainWindow`类

+   标准对话框

+   使用`QSettings`保存设置

# 技术要求

本章将需要与[第1章](bce5f3b1-2979-4f78-817b-3986e7974725.xhtml)的设置相同。您可能还希望参考我们在GitHub存储库中找到的代码，网址为[https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter04](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter04)。

查看以下视频以查看代码的实际操作：[http://bit.ly/2M5OGnq](http://bit.ly/2M5OGnq)

# QMainWindow类

到目前为止，我们一直在使用`QWidget`作为顶级窗口的基类。这对于简单的表单效果很好，但它缺少许多我们可能期望从应用程序的主窗口中得到的功能，比如菜单栏或工具栏。Qt提供了`QMainWindow`类来满足这种需求。

从[第1章](bce5f3b1-2979-4f78-817b-3986e7974725.xhtml)的应用程序模板中复制一份，并进行一个小但至关重要的更改：

```py
class MainWindow(qtw.QMainWindow):
```

我们不再继承自`QWidget`，而是继承自`QMainWindow`。正如您将看到的，这将改变我们编写GUI的方式，但也会为我们的主窗口添加许多很好的功能。

为了探索这些新功能，让我们构建一个简单的纯文本编辑器。以下屏幕截图显示了我们完成的编辑器的外观，以及显示`QMainWindow`类的主要组件的标签：

![](assets/67b30b0a-e7d2-42cf-8b9d-3d2f92ee0828.png)

保存您更新的模板，将其复制到一个名为`text_editor.py`的新文件中，并在您的代码编辑器中打开新文件。让我们开始吧！

# 设置中央小部件

`QMainWindow`分为几个部分，其中最重要的是**中央小部件**。这是一个代表界面主要业务部分的单个小部件。

我们通过将任何小部件的引用传递给`QMainWindow.setCentralWidget（）`方法来设置这一点，就像这样：

```py
        self.textedit = qtw.QTextEdit()
        self.setCentralWidget(self.textedit)
```

只能有一个中央小部件，因此在更复杂的应用程序（例如数据输入应用程序）中，它更可能是一个`QWidget`对象，您在其中安排了一个更复杂的GUI；对于我们的简单文本编辑器，一个单独的`QTextEdit`小部件就足够了。请注意，我们没有在`QMainWindow`上设置布局；这样做会破坏组件的预设排列。 

# 添加状态栏

**状态栏**是应用程序窗口底部的一条条纹，用于显示短文本消息和信息小部件。在Qt中，状态栏是一个`QStatusBar`对象，我们可以将其分配给主窗口的`statusBar`属性。

我们可以像这样创建一个：

```py
        status_bar = qtw.QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage('Welcome to text_editor.py')
```

然而，没有必要费这么大的劲；如果没有状态栏，`QMainWindow`对象的`statusBar（）`方法会自动创建一个新的状态栏，如果有状态栏，则返回现有的状态栏。

因此，我们可以将所有的代码简化为这样：

```py
        self.statusBar().showMessage('Welcome to text_editor.py')
```

`showMessage（）`方法确切地做了它所说的，显示状态栏中给定的字符串。这是状态栏最常见的用法；但是，`QStatusBar`对象也可以包含其他小部件。

例如，我们可以添加一个小部件来跟踪我们的字符计数：

```py
        charcount_label = qtw.QLabel("chars: 0")
        self.textedit.textChanged.connect(
            lambda: charcount_label.setText(
                "chars: " +
                str(len(self.textedit.toPlainText()))
                )
            )
        self.statusBar().addPermanentWidget(charcount_label)
```

每当我们的文本更改时，这个`QLabel`就会更新输入的字符数。

请注意，我们直接将其添加到状态栏，而不引用布局对象；`QStatusBar`具有自己的方法来添加或插入小部件，有两种模式：**常规**和**永久**。在常规模式下，如果状态栏发送了一个长消息来显示，小部件可能会被覆盖。在永久模式下，它们将保持可见。在这种情况下，我们使用`addPermanentWidget()`方法以永久模式添加`charcount_label`，这样它就不会被长文本消息覆盖。

在常规模式下添加小部件的方法是`addWidget()`和`insertWidget()`；对于永久模式，请使用`addPermanentWidget()`和`insertPermanentWidget()`。

# 创建应用程序菜单

**应用程序菜单**对于大多数应用程序来说是一个关键功能，它提供了对应用程序所有功能的访问，以分层组织的下拉菜单形式。

我们可以使用`QMainWindow.menuBar()`方法轻松创建一个。

```py
        menubar = self.menuBar()
```

`menuBar()`方法返回一个`QMenuBar`对象，与`statusBar()`一样，如果存在窗口的现有菜单，此方法将返回该菜单，如果不存在，则会创建一个新的菜单。

默认情况下，菜单是空白的，但是我们可以使用菜单栏的`addMenu()`方法添加子菜单，如下所示：

```py
        file_menu = menubar.addMenu('File')
        edit_menu = menubar.addMenu('Edit')
        help_menu = menubar.addMenu('Help')
```

`addMenu()`返回一个`QMenu`对象，表示下拉子菜单。传递给该方法的字符串将用于标记主菜单栏中的菜单。

某些平台，如macOS，不会显示空的子菜单。有关在macOS中构建菜单的更多信息，请参阅*macOS上的菜单*部分。

要向这些菜单填充项目，我们需要创建一些**操作**。操作只是`QAction`类的对象，表示我们的程序可以执行的操作。要有用，`QAction`对象至少需要一个名称和一个回调；它们还可以为操作定义键盘快捷键和图标。

创建操作的一种方法是调用`QMenu`对象的`addAction()`方法，如下所示：

```py
        open_action = file_menu.addAction('Open')
        save_action = file_menu.addAction('Save')
```

我们创建了两个名为`Open`和`Save`的操作。它们实际上什么都没做，因为我们还没有分配回调方法，但是如果运行应用程序脚本，您会看到文件菜单确实列出了两个项目，`Open`和`Save`。

创建实际执行操作的项目，我们可以传入第二个参数，其中包含一个Python可调用对象或Qt槽：

```py
        quit_action = file_menu.addAction('Quit', self.destroy)
        edit_menu.addAction('Undo', self.textedit.undo)
```

对于需要更多控制的情况，可以显式创建`QAction`对象并将其添加到菜单中，如下所示：

```py
        redo_action = qtw.QAction('Redo', self)
        redo_action.triggered.connect(self.textedit.redo)
        edit_menu.addAction(redo_action)
```

`QAction`对象具有`triggered`信号，必须将其连接到可调用对象或槽，以使操作产生任何效果。当我们使用`addAction()`方法创建操作时，这将自动处理，但在显式创建`QAction`对象时，必须手动执行。

虽然在技术上不是必需的，但在显式创建`QAction`对象时传入父窗口小部件非常重要。如果未这样做，即使将其添加到菜单中，该项目也不会显示。

# macOS上的菜单

`QMenuBar`默认包装操作系统的本机菜单系统。在macOS上，本机菜单系统有一些需要注意的特殊之处：

+   macOS使用**全局菜单**，这意味着菜单栏不是应用程序窗口的一部分，而是附加到桌面顶部的栏上。默认情况下，您的主窗口的菜单栏将用作全局菜单。如果您有一个具有多个主窗口的应用程序，并且希望它们都使用相同的菜单栏，请不要使用`QMainWindow.menuBar()`来创建菜单栏。而是显式创建一个`QMenuBar`对象，并使用`setMenuBar()`方法将其分配给您使用的主窗口对象。

+   macOS还有许多默认的子菜单和菜单项。要访问这些项目，只需在添加子菜单时使用相同的方法。有关添加子菜单的更多详细信息，请参阅*进一步阅读*部分中有关macOS菜单的更多详细信息。

+   如前所述，macOS不会在全局菜单上显示空子菜单。

如果您发现这些问题对您的应用程序太具有问题，您可以始终指示 Qt 不使用本机菜单系统，就像这样：

```py
        self.menuBar().setNativeMenuBar(False)
```

这将在应用程序窗口中放置菜单栏，并消除特定于平台的问题。但是，请注意，这种方法会破坏 macOS 软件的典型工作流程，用户可能会感到不适。

有关 macOS 上的 Qt 菜单的更多信息，请访问[https://doc.qt.io/qt-5/macos-issues.html#menu-bar](https://doc.qt.io/qt-5/macos-issues.html#menu-bar)。

# 添加工具栏

**工具栏**是一排长按钮，通常用于编辑命令或类似操作。与主菜单不同，工具栏不是分层的，按钮通常只用图标标记。

`QMainWindow`允许我们使用`addToolBar()`方法向应用程序添加多个工具栏，就像这样：

```py
        toolbar = self.addToolBar('File')
```

`addToolBar()`方法创建并返回一个`QToolBar`对象。传递给该方法的字符串成为工具栏的标题。

我们可以像向`QMenu`对象添加`QAction`对象一样添加到`QToolBar`对象中：

```py
        toolbar.addAction(open_action)
        toolbar.addAction("Save")
```

与菜单一样，我们可以添加`QAction`对象，也可以只添加构建操作所需的信息（标题、回调等）。

运行应用程序；它应该看起来像这样：

![](assets/f019127f-1cea-4671-84cd-3379db227da6.png)

请注意，工具栏的标题不会显示在工具栏上。但是，如果右键单击工具栏区域，您将看到一个弹出菜单，其中包含所有工具栏标题，带有复选框，允许您显示或隐藏应用程序的任何工具栏。

默认情况下，工具栏可以从应用程序中拆下并悬浮，或者停靠到应用程序的四个边缘中的任何一个。可以通过将`movable`和`floatable`属性设置为`False`来禁用此功能：

```py
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
```

您还可以通过将其`allowedAreas`属性设置为来自`QtCore.Qt.QToolBarAreas`枚举的标志组合，限制窗口的哪些边可以停靠该工具栏。

例如，让我们将工具栏限制为仅限于顶部和底部区域：

```py
        toolbar.setAllowedAreas(
            qtc.Qt.TopToolBarArea |
            qtc.Qt.BottomToolBarArea
        )
```

我们的工具栏当前具有带文本标签的按钮，但通常工具栏会有带图标标签的按钮。为了演示它的工作原理，我们需要一些图标。

我们可以从内置样式中提取一些图标，就像这样：

```py
        open_icon = self.style().standardIcon(qtw.QStyle.SP_DirOpenIcon)
        save_icon = self.style().standardIcon(qtw.QStyle.SP_DriveHDIcon)
```

现在不要担心这段代码的工作原理；有关样式和图标的完整讨论将在[第6章](c3eb2567-0e73-4c37-9a9e-a0e2311e106c.xhtml) *Qt 应用程序的样式* 中进行。现在只需了解`open_icon`和`save_icon`是`QIcon`对象，这是 Qt 处理图标的方式。

这些可以附加到我们的`QAction`对象，然后可以将它们附加到工具栏，就像这样：

```py
        open_action.setIcon(open_icon)
        toolbar.addAction(open_action)
```

如您所见，这看起来好多了：

![](assets/7b1be327-2b73-4735-a461-fc53946a9f16.png)

注意，当您运行此代码时，菜单中的文件 | 打开选项现在也有图标。因为两者都使用`open_action`对象，我们对该操作对象所做的任何更改都将传递到对象的所有使用中。

图标对象可以作为第一个参数传递给工具栏的`addAction`方法，就像这样：

```py
        toolbar.addAction(
            save_icon,
            'Save',
            lambda: self.statusBar().showMessage('File Saved!')
        )
```

这将在工具栏中添加一个带有图标和一个相当无用的回调的保存操作。请注意，这一次，菜单中的文件 | 保存操作没有图标；尽管我们使用了相同的标签文本，在两个地方分别调用`addAction()`会导致两个不同且不相关的`QAction`对象。

最后，就像菜单一样，我们可以显式创建`QAction`对象，并将它们添加到工具栏中，就像这样：

```py
        help_action = qtw.QAction(
            self.style().standardIcon(qtw.QStyle.SP_DialogHelpButton),
            'Help',
            self,  # important to pass the parent!
            triggered=lambda: self.statusBar().showMessage(
                'Sorry, no help yet!'
                )
        )
        toolbar.addAction(help_action)
```

要在多个操作容器（工具栏、菜单等）之间同步操作，可以显式创建`QAction`对象，或者保存从`addAction()`返回的引用，以确保在每种情况下都添加相同的操作对象。

我们可以向应用程序添加任意数量的工具栏，并将它们附加到应用程序的任何一侧。要指定一侧，我们必须使用`addToolBar()`的另一种形式，就像这样：

```py
        toolbar2 = qtw.QToolBar('Edit')
        toolbar2.addAction('Copy', self.textedit.copy)
        toolbar2.addAction('Cut', self.textedit.cut)
        toolbar2.addAction('Paste', self.textedit.paste)
        self.addToolBar(qtc.Qt.RightToolBarArea, toolbar2)
```

要使用这种形式的`addToolBar()`，我们必须首先创建工具栏，然后将其与`QtCore.Qt.ToolBarArea`常量一起传递。

# 添加停靠窗口

**停靠窗口**类似于工具栏，但它们位于工具栏区域和中央窗口之间，并且能够包含任何类型的小部件。

添加一个停靠窗口就像显式创建一个工具栏一样：

```py
        dock = qtw.QDockWidget("Replace")
        self.addDockWidget(qtc.Qt.LeftDockWidgetArea, dock)
```

与工具栏一样，默认情况下，停靠窗口可以关闭，浮动或移动到应用程序的另一侧。要更改停靠窗口是否可以关闭，浮动或移动，我们必须将其`features`属性设置为`QDockWidget.DockWidgetFeatures`标志值的组合。

例如，让我们使用户无法关闭我们的停靠窗口，通过添加以下代码：

```py
        dock.setFeatures(
            qtw.QDockWidget.DockWidgetMovable |
            qtw.QDockWidget.DockWidgetFloatable
        )
```

我们已将`features`设置为`DockWidgetMovable`和`DockWidgetFloatable`。由于这里缺少`DockWidgetClosable`，用户将无法关闭小部件。

停靠窗口设计为容纳使用`setWidget()`方法设置的单个小部件。与我们主应用程序的`centralWidget`一样，我们通常会将其设置为包含某种表单或其他GUI的`QWidget`。

让我们构建一个表单放在停靠窗口中，如下所示：

```py
        replace_widget = qtw.QWidget()
        replace_widget.setLayout(qtw.QVBoxLayout())
        dock.setWidget(replace_widget)

        self.search_text_inp = qtw.QLineEdit(placeholderText='search')
        self.replace_text_inp = qtw.QLineEdit(placeholderText='replace')
        search_and_replace_btn = qtw.QPushButton(
            "Search and Replace",
            clicked=self.search_and_replace
            )
        replace_widget.layout().addWidget(self.search_text_inp)
        replace_widget.layout().addWidget(self.replace_text_inp)
        replace_widget.layout().addWidget(search_and_replace_btn)
        replace_widget.layout().addStretch()
```

`addStretch()`方法可以在布局上调用，以添加一个扩展的`QWidget`，将其他小部件推在一起。

这是一个相当简单的表单，包含两个`QLineEdit`小部件和一个按钮。当点击按钮时，它调用主窗口的`search_and_replace()`方法。让我们快速编写代码：

```py
    def search_and_replace(self):
        s_text = self.search_text_inp.text()
        r_text = self.replace_text_inp.text()

        if s_text:
            self.textedit.setText(
                self.textedit.toPlainText().replace(s_text, r_text)
                )
```

这种方法只是检索两行编辑的内容；然后，如果第一个中有内容，它将在文本编辑的内容中用第二个文本替换所有实例。

此时运行程序，您应该在应用程序的左侧看到我们的停靠窗口，如下所示：

![](assets/755c0898-64be-465f-8532-8db0e1916875.png)

请注意停靠窗口右上角的图标。这允许用户将小部件分离并浮动到应用程序窗口之外。

# 其他`QMainWindow`功能

尽管我们已经涵盖了它的主要组件，但`QMainWindow`提供了许多其他功能和配置选项，您可以在其文档中探索这些选项[https://doc.qt.io/qt-5/qmainwindow.html](https://doc.qt.io/qt-5/qmainwindow.html)。我们可能会在未来的章节中涉及其中一些，因为我们将从现在开始广泛使用`QMainWindow`。

# 标准对话框

**对话框**在应用程序中通常是必需的，无论是询问问题，呈现表单还是仅向用户提供一些信息。Qt提供了各种各样的现成对话框，用于常见情况，以及定义自定义对话框的能力。在本节中，我们将看一些常用的对话框类，并尝试设计自己的对话框。

# QMessageBox

`QMessageBox`是一个简单的对话框，主要用于显示短消息或询问是或否的问题。使用`QMessageBox`的最简单方法是利用其方便的静态方法，这些方法可以创建并显示一个对话框，而不需要太多麻烦。

六个静态方法如下：

| 功能 | 类型 | 对话框 |
| --- | --- | --- |
| `about()` | 非模态 | 显示应用程序的**关于**对话框，并提供给定的文本。 |
| `aboutQt()` | 非模态 | 显示Qt的**关于**对话框。 |
| `critical()` | 模态 | 显示带有提供的文本的关键错误消息。 |
| `information()` | 模态 | 显示带有提供的文本的信息消息。 |
| `warning()` | 模态 | 显示带有提供的文本的警告消息。 |
| `question()` | 模态 | 向用户提问。 |

这些对话框之间的主要区别在于默认图标，默认按钮和对话框的模态性。

对话框可以是**模态**的，也可以是**非模态**的。模态对话框阻止用户与程序的任何其他部分进行交互，并在显示时阻止程序执行，并且在完成时可以返回一个值。非模态对话框不会阻止执行，但它们也不会返回值。在模态`QMessageBox`的情况下，返回值是表示按下的按钮的`enum`常量。

让我们使用`about()`方法向我们的应用程序添加一个**关于**消息。首先，我们将创建一个回调来显示对话框：

```py
    def showAboutDialog(self):
        qtw.QMessageBox.about(
            self,
            "About text_editor.py",
```

```py
            "This is a text editor written in PyQt5."
        )
```

**关于**对话框是非模态的，因此它实际上只是一种被动显示信息的方式。参数依次是对话框的父窗口小部件，对话框的窗口标题文本和对话框的主要文本。

回到构造函数，让我们添加一个菜单操作来调用这个方法：

```py
        help_menu.addAction('About', self.showAboutDialog)
```

模态对话框可用于从用户那里检索响应。例如，我们可以警告用户我们的编辑器尚未完成，并查看他们是否真的打算使用它，如下所示：

```py
        response = qtw.QMessageBox.question(
            self,
            'My Text Editor',
            'This is beta software, do you want to continue?'
        )
        if response == qtw.QMessageBox.No:
            self.close()
            sys.exit()
```

所有模态对话框都返回与用户按下的按钮相对应的Qt常量；默认情况下，`question()`创建一个带有`QMessageBox.Yes`和`QMessageBox.No`按钮值的对话框，因此我们可以测试响应并做出相应的反应。还可以通过传入第四个参数来覆盖呈现的按钮，该参数包含使用管道运算符组合的多个按钮。

例如，我们可以将`No`更改为`Abort`，如下所示：

```py
        response = qtw.QMessageBox.question(
            self,
            'My Text Editor',
            'This is beta software, do you want to continue?',
            qtw.QMessageBox.Yes | qtw.QMessageBox.Abort
        )
        if response == qtw.QMessageBox.Abort:
            self.close()
            sys.exit()
```

如果静态的`QMessageBox`方法不提供足够的灵活性，还可以显式创建`QMessageBox`对象，如下所示：

```py
        splash_screen = qtw.QMessageBox()
        splash_screen.setWindowTitle('My Text Editor')
        splash_screen.setText('BETA SOFTWARE WARNING!')
        splash_screen.setInformativeText(
            'This is very, very beta, '
            'are you really sure you want to use it?'
        )
        splash_screen.setDetailedText(
            'This editor was written for pedagogical '
            'purposes, and probably is not fit for real work.'
        )
        splash_screen.setWindowModality(qtc.Qt.WindowModal)
        splash_screen.addButton(qtw.QMessageBox.Yes)
        splash_screen.addButton(qtw.QMessageBox.Abort)
        response = splash_screen.exec()
        if response == qtw.QMessageBox.Abort:
            self.close()
            sys.exit()
```

正如您所看到的，我们可以在消息框上设置相当多的属性；这些在这里描述：

| 属性 | 描述 |
| --- | --- |
| `windowTitle` | 对话框任务栏和标题栏中打印的标题。 |
| `text` | 对话框中显示的文本。 |
| `informativeText` | 在`text`字符串下显示的较长的解释性文本，通常以较小或较轻的字体显示。 |
| `detailedText` | 将隐藏在“显示详细信息”按钮后面并显示在滚动文本框中的文本。用于调试或日志输出。 |
| `windowModality` | 用于设置消息框是模态还是非模态。需要一个`QtCore.Qt.WindowModality`常量。 |

我们还可以使用`addButton()`方法向对话框添加任意数量的按钮，然后通过调用其`exec()`方法显示对话框。如果我们配置对话框为模态，此方法将返回与单击的按钮匹配的常量。

# QFileDialog

应用程序通常需要打开或保存文件，用户需要一种简单的方法来浏览和选择这些文件。 Qt为我们提供了`QFileDialog`类来满足这种需求。

与`QMessageBox`一样，`QFileDialog`类包含几个静态方法，显示适当的模态对话框并返回用户选择的值。

此表显示了静态方法及其预期用途：

| 方法 | 返回 | 描述 |
| --- | --- | --- |
| `getExistingDirectory` | String | 选择现有目录路径。 |
| `getExistingDirectoryUrl` | `QUrl` | 选择现有目录URL。 |
| `getOpenFileName` | String | 选择要打开的现有文件名路径。 |
| `getOpenFileNames` | List | 选择多个现有文件名路径以打开。 |
| `getOpenFileUrl` | `QUrl` | 选择现有文件名URL。 |
| `getSaveFileName` | String | 选择要保存到的新文件名路径或现有文件名路径。 |
| `getSaveFileUrl` | `QUrl` | 选择新的或现有的URL。 |

在支持的平台上，这些方法的URL版本允许选择远程文件和目录。

要了解文件对话框的工作原理，让我们在应用程序中创建打开文件的能力：

```py
    def openFile(self):
        filename, _ = qtw.QFileDialog.getOpenFileName()
        if filename:
            try:
                with open(filename, 'r') as fh:
                    self.textedit.setText(fh.read())
            except Exception as e:
                qtw.QMessageBox.critical(f"Could not load file: {e}")
```

`getOpenFileName()`返回一个包含所选文件名和所选文件类型过滤器的元组。如果用户取消对话框，将返回一个空字符串作为文件名，并且我们的方法将退出。如果我们收到一个文件名，我们尝试打开文件并将`textedit`小部件的内容写入其中。

由于我们不使用方法返回的第二个值，我们将其分配给`_`（下划线）变量。这是命名不打算使用的变量的标准Python约定。

`getOpenFileName()`有许多用于配置对话框的参数，所有这些参数都是可选的。按顺序，它们如下：

1.  父窗口小部件

1.  标题，用于窗口标题

1.  起始目录，作为路径字符串

1.  文件类型过滤器下拉菜单可用的过滤器

1.  默认选择的过滤器

1.  选项标志

例如，让我们配置我们的文件对话框：

```py
        filename, _ = qtw.QFileDialog.getOpenFileName(
            self,
            "Select a text file to open…",
            qtc.QDir.homePath(),
            'Text Files (*.txt) ;;Python Files (*.py) ;;All Files (*)',
            'Python Files (*.py)',
            qtw.QFileDialog.DontUseNativeDialog |
            qtw.QFileDialog.DontResolveSymlinks
        )
```

`QDir.homePath()`是一个返回用户主目录的静态方法。

请注意，过滤器被指定为单个字符串；每个过滤器都是一个描述加上括号内的通配符字符串，并且过滤器之间用双分号分隔。这将导致一个看起来像这样的过滤器下拉菜单：

![](assets/cf80d69c-0bbc-402c-8a6b-ec719c7d9afe.png)

最后，我们可以使用管道运算符组合一系列选项标志。在这种情况下，我们告诉Qt不要使用本机OS文件对话框，也不要解析符号链接（这两者都是默认情况下）。有关选项标志的完整列表，请参阅`QFileDialog`文档[https://doc.qt.io/qt-5/qfiledialog.html#Option-enum](https://doc.qt.io/qt-5/qfiledialog.html#Option-enum)。

保存文件对话框的工作方式基本相同，但提供了更适合保存文件的界面。我们可以实现我们的`saveFile()`方法如下：

```py
    def saveFile(self):
        filename, _ = qtw.QFileDialog.getSaveFileName(
            self,
            "Select the file to save to…",
            qtc.QDir.homePath(),
            'Text Files (*.txt) ;;Python Files (*.py) ;;All Files (*)'
        )
        if filename:
            try:
                with open(filename, 'w') as fh:
                    fh.write(self.textedit.toPlainText())
            except Exception as e:
                qtw.QMessageBox.critical(f"Could not save file: {e}")
```

其他`QFileDialog`便利方法的工作方式相同。与`QMessageBox`一样，也可以显式创建一个`QFileDialog`对象，手动配置其属性，然后使用其`exec()`方法显示它。然而，这很少是必要的，因为内置方法对大多数文件选择情况都是足够的。

在继续之前，不要忘记在`MainWindow`构造函数中添加调用这些方法的操作：

```py
        open_action.triggered.connect(self.openFile)
        save_action.triggered.connect(self.saveFile)
```

# QFontDialog

Qt提供了许多其他方便的选择对话框，类似于`QFileDialog`；其中一个对话框是`QFontDialog`，允许用户选择和配置文本字体的各个方面。

与其他对话框类一样，最简单的方法是调用静态方法显示对话框并返回用户的选择，这种情况下是`getFont()`方法。

让我们在`MainWindow`类中添加一个回调方法来设置编辑器字体：

```py
    def set_font(self):
        current = self.textedit.currentFont()
        font, accepted = qtw.QFontDialog.getFont(current, self)
        if accepted:
            self.textedit.setCurrentFont(font)
```

`getFont`以当前字体作为参数，这使得它将所选字体设置为当前字体（如果您忽略这一点，对话框将默认为列出的第一个字体）。

它返回一个包含所选字体和一个布尔值的元组，指示用户是否点击了确定。字体作为`QFont`对象返回，该对象封装了字体系列、样式、大小、效果和字体的书写系统。我们的方法可以将此对象传回到`QTextEdit`对象的`setCurrentFont()`槽中，以设置其字体。

与`QFileDialog`一样，如果操作系统有原生字体对话框，Qt会尝试使用它；否则，它将使用自己的小部件。您可以通过将`DontUseNativeDialog`选项传递给`options`关键字参数来强制使用对话框的Qt版本，就像我们在这里做的那样：

```py
        font, accepted = qtw.QFontDialog.getFont(
            current,
            self,
            options=(
                qtw.QFontDialog.DontUseNativeDialog |
                qtw.QFontDialog.MonospacedFonts
            )
        )
```

我们还在这里传入了一个选项，以限制对话框为等宽字体。有关可用选项的更多信息，请参阅`QFontDialog`的Qt文档[https://doc.qt.io/qt-5/qfontdialog.html#FontDialogOption-enum](https://doc.qt.io/qt-5/qfontdialog.html#FontDialogOption-enum)。

# 其他对话框

Qt包含其他对话框类，用于选择颜色、请求输入值等。所有这些类似于文件和字体对话框，它们都是`QDialog`类的子类。我们可以自己子类化`QDialog`来创建自定义对话框。

例如，假设我们想要一个对话框来输入我们的设置。我们可以像这样开始构建它：

```py
class SettingsDialog(qtw.QDialog):
    """Dialog for setting the settings"""

    def __init__(self, settings, parent=None):
        super().__init__(parent, modal=True)
        self.setLayout(qtw.QFormLayout())
        self.settings = settings
        self.layout().addRow(
            qtw.QLabel('<h1>Application Settings</h1>'),
        )
        self.show_warnings_cb = qtw.QCheckBox(
            checked=settings.get('show_warnings')
        )
        self.layout().addRow("Show Warnings", self.show_warnings_cb)

        self.accept_btn = qtw.QPushButton('Ok', clicked=self.accept)
        self.cancel_btn = qtw.QPushButton('Cancel', clicked=self.reject)
        self.layout().addRow(self.accept_btn, self.cancel_btn)
```

这段代码与我们在过去章节中使用`QWidget`创建的弹出框并没有太大的区别。然而，通过使用`QDialog`，我们可以免费获得一些东西，特别是这些：

+   我们获得了`accept`和`reject`插槽，可以将适当的按钮连接到这些插槽。默认情况下，这些会导致窗口关闭并分别发出`accepted`或`rejected`信号。

+   我们还可以使用`exec()`方法，该方法返回一个布尔值，指示对话框是被接受还是被拒绝。

+   我们可以通过向`super()`构造函数传递适当的值来轻松设置对话框为模态或非模态。

`QDialog`为我们提供了很多灵活性，可以让我们如何利用用户输入的数据。例如，我们可以使用信号来发射数据，或者重写`exec()`来返回数据。

在这种情况下，由于我们传入了一个可变的`dict`对象，我们将重写`accept()`来修改那个`dict`对象：

```py
    def accept(self):
        self.settings['show_warnings'] = self.show_warnings_cb.isChecked()
        super().accept()
```

回到`MainWindow`类，让我们创建一个属性和方法来使用新的对话框：

```py
class MainWindow(qtw.QMainWindow):

    settings = {'show_warnings': True}

    def show_settings(self):
        settings_dialog = SettingsDialog(self.settings, self)
        settings_dialog.exec()
```

使用`QDialog`类就像创建对话框类的实例并调用`exec()`一样简单。在这种情况下，由于我们直接编辑我们的`settings` dict，所以我们不需要担心连接`accepted`信号或使用`exec()`的输出。

# 使用QSettings保存设置

任何合理大小的应用程序都可能积累需要在会话之间存储的设置。保存这些设置通常涉及大量繁琐的文件操作和数据序列化工作，当我们希望跨平台良好地工作时，这种工作变得更加复杂。Qt的`QtCore.QSettings`类解救了我们。

`QSettings`类是一个简单的键值数据存储，会以平台适当的方式自动持久化。例如，在Windows上，设置存储在注册表数据库中，而在Linux上，它们被放置在`~/.config`下的纯文本配置文件中。

让我们用`QSettings`对象替换我们在文本编辑器中创建的设置`dict`对象。

要创建一个`QSettings`对象，我们需要传入公司名称和应用程序名称，就像这样：

```py
class MainWindow(qtw.QMainWindow):

    settings = qtc.QSettings('Alan D Moore', 'text editor')
```

这些字符串将确定存储设置的注册表键或文件路径。例如，在Linux上，此设置文件将保存在`~/.config/Alan D Moore/text editor.conf`。在Windows上，它将存储在注册表中的`HKEY_CURRENT_USER\Alan D Moore\text editor\`。

我们可以使用对象的`value()`方法查询任何设置的值；例如，我们可以根据`show_warnings`设置使我们的启动警告对话框成为有条件的：

```py
        if self.settings.value('show_warnings', False, type=bool):
            # Warning dialog code follows...
```

`value()`的参数是键字符串、如果未找到键则是默认值，以及`type`关键字参数，告诉`QSettings`如何解释保存的值。`type`参数至关重要；并非所有平台都能以明确的方式充分表示所有数据类型。例如，如果未指定数据类型，则布尔值将作为字符串`true`和`false`返回，这两者在Python中都是`True`。

设置键的值使用`setValue()`方法，就像在`SettingsDialog.accept()`方法中所示的那样：

```py
        self.settings.setValue(
            'show_warnings',
            self.show_warnings_cb.isChecked()
        )
```

请注意，我们不必做任何事情将这些值存储到磁盘上；它们会被Qt事件循环定期自动同步到磁盘上。它们也会在创建`QSettings`对象的时候自动从磁盘上读取。简单地用`QSettings`对象替换我们原来的`settings` dict就足以让我们获得持久的设置，而无需编写一行文件I/O代码！

# QSettings的限制

尽管它们很强大，`QSettings`对象不能存储任何东西。设置对象中的所有值都存储为`QVariant`对象，因此只有可以转换为`QVariant`的对象才能存储。这包括了一个长列表的类型，包括几乎任何Python内置类型和`QtCore`中的大多数数据类。甚至函数引用也可以被存储（尽管不是函数定义）。

不幸的是，如果你尝试存储一个无法正确存储的对象，`QSettings.setValue()`既不会抛出异常也不会返回错误。它会在控制台打印警告并存储一些可能不会有用的东西，例如：

```py
app = qtw.QApplication([])
s = qtc.QSettings('test')
s.setValue('app', app)
# Prints: QVariant::save: unable to save type 'QObject*' (type id: 39).
```

一般来说，如果你正在存储清晰表示数据的对象，你不应该遇到问题。

`QSettings`对象的另一个主要限制是它无法自动识别一些存储对象的数据类型，就像我们在布尔值中看到的那样。因此，在处理任何不是字符串值的东西时，传递`type`参数是至关重要的。

# 总结

在本章中，你学习了有助于构建完整应用程序的PyQt类。你学习了`QMainWindow`类，它的菜单、状态栏、工具栏和停靠窗口。你还学习了从`QDialog`派生的标准对话框和消息框，以及如何使用`QSettings`存储应用程序设置。

在下一章中，我们将学习Qt中的模型-视图类，这将帮助我们分离关注点并创建更健壮的应用程序设计。

# 问题

尝试这些问题来测试你从本章中学到的知识：

1.  你想要使用`QMainWindow`与[第3章](dbb86a9b-0050-490e-94da-1f4661d8bc66.xhtml)中的`calendar_app.py`脚本，*使用信号和槽处理事件*。你会如何进行转换？

1.  你正在开发一个应用程序，并将子菜单名称添加到菜单栏，但没有填充任何子菜单项。你的同事说在他们测试时，他们的桌面上没有出现任何菜单名称。你的代码看起来是正确的；这里可能出了什么问题？

1.  你正在开发一个代码编辑器，并希望为与调试器交互创建一个侧边栏面板。哪个`QMainWindow`特性对这个任务最合适？

1.  以下代码不正确；无论点击什么都会继续进行。为什么它不起作用，你该如何修复它？

```py
    answer = qtw.QMessageBox.question(
        None, 'Continue?', 'Run this program?')
    if not answer:
        sys.exit()
```

1.  你正在通过子类化`QDialog`来构建一个自定义对话框。你需要将输入到对话框中的信息传回主窗口对象。以下哪种方法将不起作用？

+   1.  传入一个可变对象，并使用对话框的`accept()`方法来更改其值。

1.  重写对象的`accept()`方法，并让它返回输入值的字典。

1.  重写对话框的`accepted`信号，使其传递输入值的字典。将此信号连接到主窗口类中的回调函数。

1.  你正在Linux上编写一个名为**SuperPhoto**的照片编辑器。你已经编写了代码并保存了用户设置，但在`~/.config/`中找不到`SuperPhoto.conf`。查看代码并确定出了什么问题：

```py
    settings = qtc.QSettings()
    settings.setValue('config_file', 'SuperPhoto.conf')
    settings.setValue('default_color', QColor('black'))
    settings.sync()
```

1.  你正在从设置对话框保存偏好设置，但由于某种原因，保存的设置回来的时候非常奇怪。这里有什么问题？

```py
    settings = qtc.QSettings('My Company', 'SuperPhoto')
    settings.setValue('Default Name', dialog.default_name_edit.text)
    settings.setValue('Use GPS', dialog.gps_checkbox.isChecked)
    settings.setValue('Default Color', dialog.color_picker.color)
```

# 进一步阅读

有关更多信息，请参考以下内容：

+   Qt的`QMainWindow`文档可以在[https://doc.qt.io/qt-5/qmainwindow.html](https://doc.qt.io/qt-5/qmainwindow.html)找到。

+   使用`QMainWindow`的示例可以在[https://github.com/pyqt/examples/tree/master/mainwindows](https://github.com/pyqt/examples/tree/master/mainwindows)找到。

+   苹果的macOS人机界面指南包括如何构建应用程序菜单的指导。这些可以在[https://developer.apple.com/design/human-interface-guidelines/macos/menus/menu-anatomy/](https://developer.apple.com/design/human-interface-guidelines/macos/menus/menu-anatomy/)找到。

+   微软提供了有关为Windows应用程序设计菜单的指南，网址为[https://docs.microsoft.com/en-us/windows/desktop/uxguide/cmd-menus](https://docs.microsoft.com/en-us/windows/desktop/uxguide/cmd-menus)。

+   PyQt提供了一些关于对话框使用的示例，网址为[https://github.com/pyqt/examples/tree/master/dialogs](https://github.com/pyqt/examples/tree/master/dialogs)。

+   `QMainWindow`也可以用于创建**多文档界面**（**MDIs**）。有关如何构建MDI应用程序的更多信息，请参见[https://www.pythonstudio.us/pyqt-programming/multiple-document-interface-mdi.html](https://www.pythonstudio.us/pyqt-programming/multiple-document-interface-mdi.html)，以及[https://doc.qt.io/qt-5/qtwidgets-mainwindows-mdi-example.html](https://doc.qt.io/qt-5/qtwidgets-mainwindows-mdi-example.html)上的示例代码。
