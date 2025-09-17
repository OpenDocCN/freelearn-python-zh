# 信号、槽和事件处理器

在本书的前几章中，我们介绍了 GUI 应用程序的图形组件，以及一些与使用数据相关的附加功能解释。然而，Qt 库有一个非常重要的通信组件，它通过点击按钮、悬停标签、拖动元素、选择框选项等多种方式实现了用户与应用程序之间的通信。在第十章“图形表示”中，我们介绍了 Qt 库的`QObject`类，它实现了功能，并且是便于对象之间通信的基类之一。这种行为是 GUI 应用程序的主要目标，需要学习。

在本章中，我们将涵盖以下主题：

+   信号/槽机制

+   事件和事件处理器

# 信号和槽

通常，用于开发图形应用程序的工具包和框架使用称为**回调**的通信机制。这种方法将某个函数的回调指针发送到另一个函数，以便在需要时调用回调过程。Qt 库提供了回调的替代方案，即信号和槽机制。这种技术用于对象之间的通信。信号可以由对象发出；例如，按钮被点击（信号）并调用响应特定信号的函数（槽）。

我们可以将 Qt 库的信号/槽机制描述如下：在继承方案中从`QObject`类继承的所有类都可以包含信号和槽。这种机制被称为**类型安全**。信号的定义必须与接收槽的签名相匹配，并且可以接受任何类型和数量的参数，同时仍然保持类型安全。

一些基本的信号/槽特性如下：

+   一个信号可以连接到多个槽。

+   一个槽可以连接到多个信号。

+   一个信号可以连接到另一个信号。

+   信号/槽连接可以是同步（直接）或异步（排队）的。

+   信号/槽连接可以通过线程实现。

现在，让我们更详细地分别描述信号和槽组件。

# 信号

信号可以被定义为对象发出的信号动作，以实现确保结果的过程。例如，当我们点击想要的按钮时，`clicked()`、`pressed()`或`triggered()`信号将在应用程序中产生结果，例如关闭窗口、颜色变化、发送电子邮件等。信号通过与 C++语言表示相关的公共访问函数执行。在这种情况下，当信号被发出时，连接到这些信号的槽通常会被立即作为正常函数调用执行。

信号和槽机制完全独立于 GUI 事件循环，并且 `emit` 语句之后的代码仅在所有槽都必须作为结果返回时才会执行。在这种情况下，当多个槽连接到同一信号时，这些槽将根据它们连接的方式依次执行。要使用这些信号，让我们描述基本知识并创建一个新的信号。正如我们之前在 第四章 “PyQt 和 PySide 入门”中提到的，信号声明在 PyQt5 和 PySide2 绑定之间存在一些差异：

1.  首先，从 `import` 语句开始，并将以下行添加到 `utools.py` 文件中：

+   在 PyQt5 的 `import` 语句的情况下，将以下内容添加到 `utools.py` 文件中：

```py
...
from PyQt5.QtCore import pyqtSignal as app_signal
...
```

+   在 PySide2 的 `import` 语句的情况下，将以下内容添加到 `utools.py` 文件中：

```py
...
from PySide2.QtCore import Signal as app_signal
...
```

在应用程序的上下文中，我们现在有导入的未绑定信号作为类属性。此未绑定信号可以检索以下参数：

```py
app_signal(types, name, revision, arguments)
```

参数定义如下：

+   `types`: 定义此信号签名的类型，例如 `str`、`int`、`list` 或任何 Python 类型对象。

+   `name`: 作为关键字参数的此信号的名称。

+   `revision`: 作为关键字参数导出到 QML 的信号的修订版本。

+   `arguments`: 导出到 QML 的此信号参数名称的序列。

1.  要绑定信号，我们需要将其引用为类实例的属性，Python 绑定将自动将实例绑定到信号以创建绑定信号。我们可以通过向 `u_tools.py` 文件中的 `UTools` 类添加以下行来实现。在 `__init__()` 函数之前将绑定信号添加到类中：

```py
...
class UTools(object):

    sig1 = app_signal(int)

    def __init__(self):
        ...
    ...
...
```

这种方式并非偶然，并且建议在应用程序类中构建信号时使用。现在，我们可以将整数作为类型参数的绑定信号。绑定信号有以下方法：

+   `sig1.connect(object, type, no_receiver_check=bool)`: 这将创建一个连接。此信号连接的方法定义如下：

+   `object`: 绑定的信号或槽，作为连接到它的 Python 可调用对象。

+   `type`: 连接的类型 (`QtCore.Qt.ConnectionType`)。

+   `no_receiver_check`: 这将抑制对底层接收器实例是否仍然存在的检查，并无论如何都传递信号。

+   `sig1.disconnect([object])`: 这将从信号断开一个或多个槽。信号断开的方法定义为 `object`；这是作为 Python 可调用对象连接到的绑定信号或槽。

+   `sig1.emit(*args)`: 这会发出一个信号，其中 `*args` 是可选的参数序列，这些参数将被传递到槽中。

# 槽

槽可以定义为当信号发出时将处理的函数，并且它们需要实现某些功能；例如，使用 `close()` 函数关闭窗口，使用 `setColor(QtGui.QColor("FFFFFF"))` 属性更改颜色，或使用 `smtplib` Python 标准库模块的 `sendmail()` 函数发送消息。

槽与普通函数之间的区别在于，槽在某种意义上具有特殊功能，信号可以连接到它们。换句话说，信号定义了某些函数将是槽。实际上，槽可以是任何 Python 可调用对象，但在某些情况下，我们可以显式定义槽并装饰函数。连接到装饰的 Python 方法在内存使用方面有一些优势。我们还可以创建新的槽和新的信号。为此，我们需要将以下行添加到 `u_tools.py` 文件的 `import` 语句中：

+   在 PyQt5 的 `import` 语句中，将以下内容添加到 `utools.py` 文件中：

```py
...
from PyQt5.QtCore import pyqtSlot as app_slot
...
```

+   在 PySide2 的 `import` 语句中，将以下内容添加到 `utools.py` 文件中：

```py
...
from PySide2.QtCore import Slot as app_slot
...
```

`import` 语句也有所不同。`app_slot(types, name, result, revision)` 参数可以传递给槽。其函数可以定义如下：

+   `types`: 定义此槽签名类型的类型，例如 `str`、`int`、`list` 或任何 Python 类型对象。

+   `name`: 作为关键字参数的此槽的名称。

+   `result`: 定义此槽签名的结果类型，例如 `str`、`int`、`list` 或任何 Python 类型对象。

+   `revision`: 作为关键字参数导出到 QML 的槽的版本。

这些槽通常以以下方式装饰：

```py
...
    ...
    @app_slot(int, name='slot_func')
    def slot_func(self, num):
        # Some processing...
    ...
...
```

# 信号-槽连接

如我们所知，许多 GUI 应用程序都是使用 Qt 库的 PyQt/PySide Python 绑定构建的。如果我们审查这些应用程序的代码，我们将看到信号/槽连接构造的风格有所不同。我们需要考虑应用程序中可用的构造。已绑定的信号具有 `connect()`、`emit()` 和 `disconnect()` 方法，它们提供了与信号和槽的操作。返回信号宏签名的 `signal()` 属性也是可用的。

# 连接

我们可以使用 `triggered()` 信号将选择选项连接到提供这些选项功能的槽。这里使用了非静态连接。然而，出于演示目的，也将使用静态构造。现在，我们需要查看 `u_app.py` 文件中 `UApp` 类的行：

```py
...
class UApp(UWindow, UTools):

    def __init__(self, parent=None):
        ...
        self.combox1.activated.connect(self.txt_table)
        self.mb1.triggered.connect(self.files)
        self.mb3.triggered.connect(self.data)
        self.mb4.triggered.connect(self.options)
        ...
        self.push_but1.clicked.connect()
        self.push_but2.clicked.connect(lambda:
                        self.text_edit.setText("HELLO"),
                        QtCore.Qt.QueuedConnection)

...
```

让我们考虑与信号相关的行。应用程序的组合框用于在`Text`和`Table`之间选择一个选项，并使用`activated()`信号以非静态方式构建，该信号在鼠标悬停和点击时激活选项，并调用槽。顶部面板菜单使用`triggered()`信号连接到相关槽，也是非静态的。`Ok`按钮将使用`clicked()`信号（非静态构建）来调用 lambda 函数。换句话说，非静态方法可以使用以下语法描述：

```py
object.signal.connect(SLOT, type=QtCore.Qt.ConnectionType)
```

在这里，`object`是一个小部件或另一个信号，`signal`是一个可用的信号或构造的信号，`connect()`方法连接到槽，而`SLOT`是一个函数/方法或 Python 可调用对象。`type`参数描述了将要使用的连接类型。以下类型的连接可用：

+   `QtCore.Qt.AutoConnection`—`0`: 这是默认类型。如果信号的接收者在发出信号的线程中，将使用`QtCore.Qt.DirectConnection`。如果接收者和发出信号的线程不同，将使用`QtCore.Qt.QueuedConnection`。

+   `QtCore.Qt.DirectConnection`—`1`: 立即调用此槽，并以与发出信号相同的方式在相同线程中执行。

+   `QtCore.Qt.QueuedConnection`—`2`: 当控制返回接收者线程的事件循环时，将调用此槽，并在接收者线程中执行。

+   `QtCore.Qt.BlockingQueuedConnection`—`3`: 当控制返回接收者线程的事件循环时，将调用此槽，并在接收者线程中执行，同时阻塞信号线程。如果发出信号和接收者的线程相同，则不能使用此方法。

+   `QtCore.Qt.UniqueConnection`—`3`: 将使用唯一连接，如果相同的信号已经连接到相同的小部件对上的相同槽，则另一个连接将失败。

使用`QObject`类构建连接的方法也有以下静态方法：

```py
QtCore.QObject.connect(object, SIGNAL, SLOT, type=QtCore.Qt.ConnectionType)
```

在这里，`object`是一个小部件、信号等；`SIGNAL`，作为`QtCore.SIGNAL ("signal_name()")`，是一个可用的信号或构造的信号；而`SLOT`是一个函数/方法或 Python 可调用对象，也可以是**`QtCore.SLOT ("slot_name()")`**。`type`参数描述了将要使用的连接类型。`connect`方法用作`QtCore`模块中`QObject`类的静态函数。

在静态函数的情况下，我们有以下：

```py
QtCore.QObject.connect(object1, SIGNAL, object2, SLOT, type=QtCore.Qt.ConnectionType)
```

在这里，`object1`（小部件、信号等）是`QtCore.QObject`类型的发送者，`SIGNAL`（`QtCore.SIGNAL ("signal_name()")`）是`QtCore.QMetaMethod`类型的信号，`object2`（小部件、信号等）是`QtCore.QObject`类型的接收者，`SLOT`（`QtCore.SLOT("slot_name()")`）是`QtCore.QMetaMethod`类型的功能/方法，而`type`是连接将要使用的参数。

实际上，使用`QObject`类的静态构造在 PySide 和 PySide2，或 PyQt4 中是可用的。PyQt5 绑定不支持这些构造，并使用之前描述的新式的信号和槽。为了避免混淆并保持材料的复杂性，我们不会这样做。在我们的应用程序中，我们将使用非静态方法在信号和槽之间建立连接。

# emit

在某些情况下，当应用程序正在处理时，需要从应用程序中发出信号。`emit()`方法可以在连接的信号/槽对象之间执行信号的发射，同时（可选地）向接收者发送一些相关数据。这对于使用线程来标准化过程执行的应用程序非常有用，这将在第十六章线程和多进程中详细讨论。

让我们考虑一个与`emit()`方法相关的例子，其中使用这些各种数据处理工具的应用程序开始变慢。我们将更改用于打开 1,000 行的 pandas 工具函数，使用线程来分割这个功能，并操作将要写入文件的 1,000,000 行/5 列的表格。为此，我们需要打开`u_tools.py`文件并添加/更改以下行：

1.  首先，我们需要将这些行添加到这些文件的`import`部分：

+   在 PyQt5 的`u_tools.py`文件的情况下，将以下内容添加到`import`部分：

```py
...
from PyQt5.QtCore import QThread
import time
...
```

+   在 PySide2 的`u_tools.py`文件的情况下，将以下内容添加到`import`部分：

```py
...
from PySide2.QtCore import QThread
import time
...
```

1.  在`UTools`类之前，我们需要添加将使用 pandas 读取/写入 CSV 文件的线程类。

添加带有线程的`WPandas`类以写入 CSV 文件：

```py
...
class WPandas(QThread):

    sig1 = app_signal(object, str)

    def __init__(self, parent=None):
        super(WPandas, self).__init__(parent) 

    def on_source(self, datas):
        self.datas = datas

    def run(self):
        try:
            import pandas
            uindex = [i for i in range(self.datas[2])]
            udata = {"User_Name": range(0, self.datas[2]),
                     "User_email": range(0, self.datas[2]),
                     "User_password": range(0, self.datas[2]),
                     "User_data": range(0, self.datas[2])}
            df = pandas.DataFrame(udata, columns=self.datas[3],
                             index=uindex, dtype=self.datas[5])
            df.index.name = "rows\columns"
            if self.datas[1] == "csv":
                df.to_csv(self.datas[0])
            if self.datas[1] == "excel":
                df.to_excel(self.datas[0])
            if self.datas[1] == "html":
                df.to_html(self.datas[0])
            if self.datas[1] == "json":
                df.to_json(self.datas[0])
            if self.isFinished():
                self.quit()
        except Exception as err:
            self.sig1.emit('', str(err))
            if self.isFinished():
                self.quit()
...
```

这个类包括`sig1`信号，如果发生错误，将发出带有错误信息的字符串。

1.  现在，添加带有线程的`RPandas`类以读取 CSV 文件：

```py
...
class RPandas(QThread):

    sig1 = app_signal(object, str)

    def __init__(self, parent=None):
        super(RPandas, self).__init__(parent)

    def on_source(self, datas):
        self.datas = datas

    def run(self):
        try:
            import pandas
            if self.datas[1] == "csv":
                df = pandas.read_csv(self.datas[0],
                                     chunksize=self.datas[6],
                                     engine=self.datas[4]) 
            if self.datas[1] == "excel":
                df = pandas.read_excel(self.datas[0]) 
            if self.datas[1] == "html":
                df = pandas.read_html(self.datas[0])
            if self.datas[1]== "json":
                df = pandas.read_json(self.datas[0]) 
            pandas.options.display.max_rows = self.datas[5]
            for ch in df:
                self.sig1.emit(ch, '')
                time.sleep(0.1)
            if self.isFinished():
                self.quit()
        except Exception as err:
            self.sig1.emit('', str(err)) 
            if self.isFinished():
                self.quit()
...
```

此类包括`sig1`信号，该信号将发出带有`object`类型的`DataFrame`，如果存在，则带有`string`类型的错误。CSV 文件将分块读取，并在循环中发出，而不是一次性读取。这样做是因为大文件难以读取，当文本添加到文本字段时可能会出现问题。因为我们需要记住文本编辑字段是另一个线程的一部分——即应用程序的主 GUI 线程——每个块包含 10,000 行，但只显示块中的 9 行。这展示了处理大型数据集的可能性，因为将所有这些数据显示在应用程序的文本字段中可能会使应用程序冻结。

实际上，这并不需要，因为用户不想读取文件的所有行；他们只想操作这些数据并可视化它；这是关键。所有前面的线程都在`on_source()`函数外部检索数据。

1.  现在，我们需要继续添加/修改`u_tools.py`文件中的`UTools`类。

在`UTools`类的`__init__()`函数之前添加以下信号：

```py
...
class UTools(object):

    ...
    pandas_sig1 = app_signal(list)
    pandas_sig2 = app_signal(list)

    def __init__(self):
    ...
...
```

这创建了类函数和线程之间的绑定信号，用于通信。这些信号将发送带有参数的列表到线程。

1.  现在，我们需要更改`pandas_write()`函数，如下所示：

```py
...
    ...
    def pandas_write(self, filename=None, writer="csv",
                               data=None, columns=None,
                               index=None, dtype=object):
        data = 1000000
        index = 1000000
        datas = [filename, writer, data, columns, index, dtype]
        self.pandas_thread1 = WPandas()
        self.pandas_sig1.connect(self.pandas_thread1.on_source)
        self.pandas_sig1.emit(datas)
        self.pandas_thread1.start()
    ...
...
```

在这里，我们正在替换函数中的数据和索引变量。这是可选的，并表明数据将在 0-1,000,000 的`data`范围内，行数等于 1,000,000 的`index`。当创建线程实例时，`pandas_sig1`信号连接到线程的`on_source()`函数，然后向线程发出带有数据和参数的列表以进行处理。

1.  按如下方式更改`pandas_read()`函数：

```py
...
    ...
    def pandas_read(self, filename=None, reader="csv", sep=',',
                               delimiter=None, engine='python',
                                        maxrows=9, chunk=10000):
        datas = [filename, reader, sep, delimiter,
                 engine, maxrows, chunk]
        self.pandas_thread2 = RPandas()
        self.pandas_sig2.connect(self.pandas_thread2.on_source)
        self.pandas_sig2.emit(datas)
        self.pandas_thread2.start()
        return self.pandas_thread2
    ...
...
```

此函数创建线程实例，定义读取参数，例如文件名、读取器和块大小，连接到`on_source()`函数，并将参数作为列表发出。启动线程后返回此线程，以便可以自由使用。

现在，我们需要更改`u_app.py`文件中的`UApp`类。让我们开始吧：

1.  在使用 pandas 打开 CSV 文件的`data()`函数中进行更改：

```py
...
    ...
    def data(self, action):
        ...
        if self.actx == "Pandas":
            try:
                pread = self.pandas_read(
                            filename="data/bigtests.csv",
                                   reader="csv", sep=',')
                def to_field(df, er):
                    if er == '':
                        self.text_edit.append(
                                          "\n" + str(df))
                    else:
                        self.stat_bar.showMessage(
                                    self.actx + ' ' + er)
                pread.sig1.connect(
                    to_field, QtCore.Qt.QueuedConnection)
                self.stat_bar.showMessage(
                                   self.actx + " opened")
            except Exception as err:
                self.stat_bar.showMessage(
                              self.actx + ' ' + str(err))
        ...
    ...
...
```

在这里，我们添加了线程的`sig1`信号，该信号连接到槽，以及一个嵌套的`to_field()`函数，用于将读取文件的块追加到应用程序的文本编辑字段，或将错误添加到状态栏。

1.  `user_data4()`函数现在如下所示：

```py
...
    ...
    def user_data4(self, uname, umail, upass, udata):
        try:
            ucolumns = ["User_Name", "User_email",
                        "User_password", "User_data"]
            self.pandas_write(filename="data/bigtests.csv",
                            writer="csv", columns=ucolumns)
        except Exception as err:
            self.error = err
    ...
...
```

这些示例展示了我们如何将信号连接到槽（应用程序函数）以及从/到线程和从/到函数发出各种数据。

# 断开连接

信号断开连接的方法很简单。在某些情况下，我们需要从槽断开信号的连接，这如下所示：

```py
object.signal.disconnect(SLOT)
```

在这里，`object`是一个控件或另一个信号，`signal`是一个使用的信号，具有`disconnect`方法，以及一个`SLOT`，这是一个连接到此信号的功能/方法或 Python 可调用对象。

这个构造函数断开了连接的槽的信号。需要注意的是，构造函数需要与信号连接时使用的相同。

# Qt 模块

Qt 库提供了几个类来实现信号和槽的附加功能。这些类是**`QSignalBlocker`**、`QSignalMapper`和`QSignalTransition`。所有这些都在`QtCore`模块中可用。

# QSignalBlocker

这个类实现了对`blockSignals()`方法的异常安全包装，该方法阻止了项的信号。通常，这个类可以用作`blockSignals()`方法的替代。这个类的声明语法如下：

```py
signal_blocker = QtCore.QSignalBlocker()
```

`QSignalBlocker`类通过以下函数提高了功能。

# functional

这些是与功能变化相关的函数。

`signal_blocker.reblock()`: 这重新阻止了之前取消阻止的信号。

`signal_blocker.unblock()`: 这将`signalsBlocked()`状态恢复到阻塞前的状态。

# QSignalMapper

这个类实现了从可识别的发送者发出的信号包，并提供了一组信号。它使用与发送者对应的`integer`、`string`或`widget`参数重新发出它们。这个类的声明语法如下：

```py
signal_mapper = QtCore.QSignalMapper()
```

`QSignalMapper`类通过以下函数提高了功能。

# set

这些是与信号映射器相关的设置参数/属性的函数。

`signal_mapper.setMapping(QtCore.QObject, int)`: 这设置了当`map()`函数从发送者（第一个参数）发出信号时的映射；映射的信号 ID（第二个参数）将被发出。

`signal_mapper.setMapping(QtCore.QObject, str)`: 这设置了当`map()`函数从发送者（第一个参数）发出信号时的映射；映射的信号文本（第二个参数）将被发出。

`signal_mapper.setMapping(QtCore.QObject, object)`: 这设置了当`map()`函数从发送者（第一个参数）发出信号时的映射；映射的信号对象（第二个参数）将被发出。

`signal_mapper.setMapping(QtCore.QObject, QtWidgets.QWidget)`: 这设置了当`map()`函数从发送者（第一个参数）发出信号时的映射；映射的信号控件（第二个参数）将被发出。

# functional

这些是与信号映射器当前值的返回、功能变化等相关的函数。

`signal_mapper.map()`: 这根据向它发送信号的发送者对象发出信号。

`signal_mapper.map(QtCore.QObject)`: 这根据参数中指定的发送者发出信号。

`signal_mapper.mapping(int)`: 这返回与参数中指定的 ID 关联的 `QtCore.QObject` 类型的发送者。

`signal_mapper.mapping(str)`: 这返回与参数中指定的文本关联的 `QtCore.QObject` 类型的发送者。

`signal_mapper.mapping(object)`: 这返回与参数中指定的对象关联的 `QtCore.QObject` 类型的发送者。

`signal_mapper.mapping(QtWidgets.QWidget)`: 这返回与参数中指定的小部件关联的 `QtCore.QObject` 类型的发送者。

`signal_mapper.removeMappings(QtCore.QObject)`: 这移除参数中指定的发送者的映射。

# 信号

`QSignalMapper` 类的可用信号如下：

`signal_mapper.mapped(int)`: 当从具有 ID 映射设置的对象发出 `map()` 函数信号时，会发出此信号；ID 通过参数传递。

`signal_mapper.mapped(str)`: 当从具有字符串映射设置的对象发出 `map()` 函数信号时，会发出此信号；文本通过参数传递。

`signal_mapper.mapped(object)`: 当从具有对象映射设置的对象发出 `map()` 函数信号时，会发出此信号；对象通过参数传递。

`signal_mapper.mapped(QtWidgets.QWidget)`: 当从具有小部件映射设置的对象发出 `map()` 函数信号时，会发出此信号；小部件通过参数传递。

# QSignalTransition

此类实现信号转换。该类使用参数中定义的源状态构建新的信号转换。此类的声明语法如下：

```py
signal_transit = QtCore.QSignalTransition(QtCore.QState)
```

`QSignalTransition` 类通过以下函数改进了功能。

# 设置

这些是与信号转换相关联的参数/属性设置相关的函数：

`signal_transit.setSenderObject(QtCore.QObject)`: 这设置参数中指定的将与该信号转换关联的发送者。

`signal_transit.setSignal(QtCore.QByteArray)`: 这设置参数中指定的将与该信号转换关联的信号。

# 功能

这些是与信号转换当前值返回、功能变化等相关联的函数：

`signal_transit.senderObject()`: 这返回与该信号转换关联的 `QtCore.QObject` 类型的发送者。

`signal_transit.signal()`: 这返回与该信号转换关联的 `QtCore.QByteArray` 类型的信号。

# 事件和事件处理器

GUI 应用程序的一个重要方面是实现事件和事件处理器。事件通知应用程序有关由于与应用程序进程相关联的内部或外部活动而发生的事情。在 Qt 库中，这种行为通过 Qt 事件系统来表征和实现。通过此系统，事件是派生于`QtCore`模块的抽象`QEvent`类的对象。事件可以通过`QObject`子类的任何实例来处理。通常，事件是通过调用一个虚拟函数——事件处理器——来传递的，这提供了一个方便的方式来处理与应用程序相关的事件。事件处理器通常如下所示：

```py
QEnterEvent 
```

这里，它使用`enterEvent()`方法作为事件处理器。Qt 框架实现了绘画、调整大小、发送、显示、拖放、鼠标、键盘等事件处理器。下一节描述了在 GUI 应用程序中可以实现的常用事件和事件处理器。

# QEvent

这是所有事件类的基类，并提供了具有特殊事件类型功能的事件系统实现。此类的声明语法如下：

```py
event = QtCore.QEvent(QtCore.QEvent.Type)
```

可以在参数中指定的可用类型完整列表可以在 Qt 文档中找到（[`doc.qt.io`](https://doc.qt.io/)）。`QEvent`类通过以下函数提高了功能。

# 设置

此函数与设置与事件相关的参数/属性相关：

`event.setAccepted(bool)`: 如果参数为`True`，则将此事件设置为接受状态。

# 是

此函数返回一个与事件状态相关的布尔值（`bool`）：

`event.isAccepted()`: 如果此事件被接受，则返回`True`。

# 功能

这些是与当前事件值的返回、功能变化等相关联的函数：

`event.accept()`: 此函数设置事件对象的接受标志；此事件将被接受。

`event.ignore()`: 这个函数设置事件对象的忽略标志；此事件将被忽略。

`event.registerEventType(int)`: 此函数注册并返回一个自定义事件类型，参数中指定了提示。

`event.spontaneous()`: 如果这是一个系统事件（在应用程序外部），则返回`True`。

`event.type()`: 这个函数返回事件类型，作为`QtCore.QEvent.Type`对象。

# 事件处理器

此事件的处理程序如下：

```py
def event(self, event):
    return True
```

# QEventLoop

此类提供了进入和离开事件循环的功能。主事件循环在应用程序启动时实现，并进入无限循环。此类的声明语法如下：

```py
event_loop = QtCore.QEventLoop()
```

`QEventLoop`类通过以下函数提高了功能。

# 是

此函数返回一个与事件循环状态相关的布尔值（`bool`）：

`event.isRunning()`: 如果此事件循环正在运行，则返回`True`；否则，返回`false`。

# 功能性

这些是与事件循环当前值的返回、功能变化等相关的函数：

`event_loop.exec_(QtCore.QEventLoop.ProcessEventsFlags)`: 这将进入事件循环，并开始根据参数中指定的标志进行处理。可用的标志如下：

+   `QtCore.QEventLoop.AllEvents`: 将处理所有事件。

+   **`QtCore.QEventLoop.ExcludeUserInputEvents`**: 排除在处理过程中的用户输入事件。

+   `QtCore.QEventLoop.ExcludeSocketNotifiers`: 排除在处理过程中的套接字通知事件。

+   `QtCore.QEventLoop.WaitForMoreEvents`: 如果没有挂起事件，则处理将等待事件。

`event_loop.exit(int)`: 这将使用参数中指定的返回代码退出事件循环。返回代码为 `0` 表示成功；其他非零值表示错误。

`event_loop.processEvents(QtCore.QEventLoop.ProcessEventsFlags)`: 这将处理与参数中指定的标志匹配的挂起事件。

`event_loop.processEvents(QtCore.QEventLoop.ProcessEventsFlags, int)`: 这将在指定的最大时间（以毫秒为单位）内处理与标志（第一个参数）匹配的挂起事件。

`event_loop.quit()`: 这将正常退出事件循环，类似于 `event_loop.exit(0)` 方法。

`event_loop.wakeUp()`: 这将唤醒事件循环。

# Q 子事件

此类实现了与子对象相关的事件。当子对象被添加或删除时，事件被发送到对象。此类的声明语法如下：

```py
child_event = QtCore.QChildEvent(QtCore.QEvent.Type, object)
```

此事件可用的类型如下：

+   `QtCore.QEvent.ChildAdded`: 对象中添加了子对象。

+   `QtCore.QEvent.ChildRemoved`: 从对象中移除了子对象。

+   `QtCore.QEvent.ChildPolished`: 子对象被抛光。

`QChildEvent` 类通过以下函数提高功能：

# 功能性

这些是与当前事件值的返回、功能变化等相关的函数：

`child_event.added()`: 如果此事件的类型是 `ChildAdded`，则返回 `True`。

`child_event.child()`: 这返回被添加或删除的 `QtCore.QObject` 类型的子对象。

`child_event.polished()`: 如果此事件的类型是 `ChildPolished`，则返回 `True`。

`child_event.removed()`: 如果此事件的类型是 `ChildRemoved`，则返回 `True`。

# 事件处理器

此事件的处理器如下：

```py
def childEvent(self, event):
    """Some code lines for processing..."""
```

# QTimerEvent

此类可以实现定期向对象发送事件的计时器事件。此类的声明语法如下：

```py
timer_event = QtCore.QTimerEvent(int)
```

计时器的唯一 ID 在事件的参数中指定。`QTimerEvent` 类通过以下函数提高功能：

# 功能性

此函数与当前事件的值的返回、功能变化等相关：

`timer_event.timerId()`: 这返回计时器的唯一 ID。

# 事件处理器

此事件的处理器如下：

```py
def timerEvent(self, event):
    """Some code lines for processing..."""
```

# QActionEvent

此类提供了当使用 `QAction` 类实现动作添加、移除或更改时出现的事件。它适用于支持操作的项目，如 `QMenu`。此类的声明语法如下：

```py
action_event = QtGui.QActionEvent(QtCore.QEvent.Type,
                           QtWidgets.QAction, QtWidgets.QAction)
```

第一个参数是事件类型，第二个是动作，第三个是指定的先前动作。此事件的可选类型如下：

+   `QtCore.QEvent.ActionChanged`: 动作已更改。

+   `QtCore.QEvent.ActionAdded`: 向对象添加动作。

+   `QtCore.QEvent.ActionRemoved`: 从对象中移除动作。

`QActionEvent` 类通过以下函数改进了功能。

# 功能性

这些是与当前事件值的返回、功能变化等相关联的函数：

`action_event.action()`: 这返回已添加、更改或移除的动作。

`action_event.before()`: 如果动作类型是 `ActionAdded`，则返回之前出现的动作。

# 事件处理器

此事件的处理器如下：

```py
def actionEvent(self, event):
    """Some code lines for processing..."""
```

# QDropEvent

此类提供了当拖放操作完成时出现的事件。它适用于支持拖动操作的项目，如 `QWidget` 和 `QTextEdit`。此类的声明语法如下：

```py
drop_event = QtGui.QDropEvent(QtCore.QPointF, QtCore.Qt.DropActions,
                           QtCore.QMimeData, QtCore.Qt.MouseButtons,
                    QtCore.Qt.KeyboardModifiers, QtCore.QEvent.Drop)
```

构建丢弃事件时使用以下参数：

+   位置（第一个参数）。

+   丢弃操作（第二个参数）。

+   MIME 数据（第三个参数）。

+   按钮状态（第四个参数）。

+   键盘修饰符（第五个参数）。

+   类型（第六个参数）。

`QDropEvent` 类通过以下函数改进了功能。

# set

此函数与设置与丢弃事件相关的参数/属性有关：

`drop_event.setDropAction(QtCore.Qt.DropAction)`: 这将设置参数中指定的动作，该动作将用于此事件。

# 功能性

这些是与丢弃事件当前值的返回、功能变化等相关联的函数：

`drop_event.acceptProposedAction()`: 这将设置丢弃操作为此事件的建议操作。

`drop_event.dropAction()`: 这返回与此事件一起使用的 `QtCore.Qt.DropAction` 类型的动作。

`drop_event.keyboardModifiers()`: 这返回与此事件一起使用的 `QtCore.Qt.KeyboardModifiers` 类型的键盘修饰符。

`drop_event.mimeData()`: 这返回与此事件一起使用的 `QtCore.QMimeData` 类型的 MIME 数据。

`drop_event.mouseButtons()`: 这返回与此事件一起使用的 `QtCore.Qt.MouseButtons` 类型的鼠标按钮。

`drop_event.pos()`: 这返回丢弃操作的 `QtCore.QPoint` 类型的位置。

`drop_event.posF()`: 这返回丢弃操作的 `QtCore.QPointF` 类型的位置。

`drop_event.possibleActions()`: 这返回 `QtCore.Qt.DropActions` 类型的可能丢弃操作。

`drop_event.proposedAction()`: 这返回 `QtCore.Qt.DropAction` 类型的建议动作。

`drop_event.source()`: 此函数返回用于拖放事件操作的`QtCore.QObject`类型源，例如小部件。

# 事件处理程序

此事件的处理程序如下：

```py
def dropEvent(self, event):
    """Some code lines for processing..."""
```

`QDragEnterEvent`、`QDragMoveEvent`和`QDragLeaveEvent`类型的事件处理程序，这些处理程序提高了`QDropEvent`的功能，如下所示：

```py
def dragEnterEvent(self, event):
    """Some code lines for processing..."""

def dragMoveEvent(self, event):
    """Some code lines for processing..."""

def dragLeaveEvent(self, event):
    """Some code lines for processing..."""
```

# QEnterEvent

此类处理鼠标光标进入小部件、窗口或其他 GUI 元素/项时的事件。它几乎适用于所有支持鼠标光标进入操作的项。此类声明的语法如下：

```py
enter_event = QtGui.QEnterEvent(QtCore.QPointF,
                        QtCore.QPointF,QtCore.QPointF)
```

关于进入事件的构建，使用以下参数：

+   本地位置（第一个参数）。

+   窗口位置（第二个参数）。

+   鼠标光标相对于接收项的屏幕位置（第三个参数）。

`QEnterEvent`类通过以下函数提高功能。

# 功能性

这些是与当前进入事件值返回、功能变化等相关的函数：

`enter_event.globalPos()`: 当进入事件发生时，此函数返回项的`QtCore.QPoint`类型全局位置。

`enter_event.globalX()`: 当进入事件发生时，此函数返回鼠标光标在项上的全局 x 轴位置。

`enter_event.globalY()`: 当进入事件发生时，此函数返回鼠标光标在项上的全局 y 轴位置。

`enter_event.localPos()`: 当进入事件发生时，此函数返回鼠标光标在项上的`QtCore.QPointF`类型本地位置。

`enter_event.pos()`: 当进入事件发生时，此函数返回鼠标光标在全局屏幕坐标下的`QtCore.QPoint`类型位置。

`enter_event.screenPos()`: 当进入事件发生时，此函数返回鼠标光标在屏幕上的`QtCore.QPointF`类型位置。

`enter_event.windowPos()`: 当进入事件发生时，此函数返回鼠标光标在窗口上的`QtCore.QPointF`类型位置。

`enter_event.x()`: 当进入事件发生时，此函数返回鼠标光标在项上的*x*位置。

`enter_event.y()`: 当进入事件发生时，此函数返回鼠标光标在项上的*y*位置，

# 事件处理程序

此事件的处理程序如下：

```py
def enterEvent(self, event):
    """Some code lines for processing..."""
```

鼠标光标从项中离开的事件可以如下实现：

```py
def leaveEvent(self, event):
    """Some code lines for processing..."""
```

# QFocusEvent

此类处理项的焦点事件。这些事件出现在键盘输入焦点变化时。它适用于支持键盘焦点操作的`QWidget`等小部件。此类声明的语法如下：

```py
focus_event = QtGui.QFocusEvent(QtCore.QEvent.Type,
                                QtCore.Qt.FocusReason)
```

关于焦点事件的构建，使用事件类型（第一个参数）和焦点原因（第二个参数）。此事件可用的类型如下：

+   `QtCore.QEvent.FocusIn`: 此项获得键盘焦点。

+   `QtCore.QEvent.FocusOut`: 此项失去键盘焦点。

+   `QtCore.QEvent.FocusAboutToChange`: 此项的焦点即将改变。

`QFocusEvent`类通过以下函数增强了功能。

# 功能

这些是与当前焦点事件当前值相关的函数：

`focus_event.gotFocus()`: 如果此事件具有`FocusIn`类型，则返回`True`。

`focus_event.lostFocus()`: 如果此事件具有`FocusOut`类型，则返回`True`。

`focus_event.reason()`: 返回此焦点事件的`QtCore.Qt.FocusReason`类型的理由。

# 事件处理器

此事件的处理程序如下：

```py
def focusInEvent(self, event):
    """Some code lines for processing..."""

def focusOutEvent(self, event):
    """Some code lines for processing..."""
```

# QKeyEvent

此类处理与键盘活动相关的事件。当您使用`QWidget`等小部件按下键盘键时，它变得可用。此类的声明语法如下：

```py
key_event = QtGui.QKeyEvent(QtCore.QEvent.Type, int,
                            QtCore.Qt.KeyboardModifiers)
```

关于按键事件的构建，使用类型（第一个参数）、键（第二个参数）和键盘修饰符（第三个参数）。此事件可用的类型如下：

+   `QtCore.QEvent.KeyPress`: 按下键。

+   `QtCore.QEvent.KeyRelease`: 释放键。

+   `QtCore.QEvent.ShortcutOverride`: 子窗口中的按键。

`QKeyEvent`类通过以下函数增强了功能。

# 是

此函数返回与按键事件状态相关的布尔值（`bool`）：

`key_event.isAutoRepeat()`: 如果按键事件来自自动重复键，则返回`True`。

# 功能

这些是与当前按键事件当前值相关的函数：

`key_event.count()`: 返回此事件可用的键的数量。

`key_event.key()`: 返回键的代码，对应于`QtCore.Qt.Key`。

`key_event.matches(QtGui.QKeySequence.StandardKey)`: 如果按键事件与参数中指定的标准键匹配，则返回`True`。

`key_event.modifiers()`: 返回此按键事件的`QtCore.Qt.KeyboardModifiers`类型的键盘修饰符。

`key_event.nativeModifiers()`: 返回此按键事件的本地修饰符。

`key_event.nativeScanCode()`: 返回此按键事件的扫描码。

`key_event.nativeVirtualKey()`: 返回此按键事件的虚拟键。

`key_event.text()`: 返回由此键生成的文本。

# 事件处理器

此事件的处理程序如下：

```py
def keyPressEvent(self, event):
    """Some code lines for processing..."""

def keyReleaseEvent((self, event):
    """Some code lines for processing..."""
```

# QMouseEvent

此类处理与鼠标活动同时出现的事件。它几乎与所有可以与鼠标交互的图形项都可用。此类的声明语法如下：

```py
mouse_event = QtGui.QMouseEvent(QtCore.QEvent.Type,
                           QtCore.QPointF, QtCore.QPointF,
                           QtCore.QPointF, QtCore.Qt.MouseButton, 
                           QtCore.Qt.MouseButtons,
                           QtCore.Qt.KeyboardModifiers)
```

关于鼠标事件构建，使用以下参数：

+   类型（第一个参数）。

+   光标局部位置（第二个参数）。

+   光标窗口位置（第三个参数）。

+   光标屏幕位置（第四个参数）。

+   导致事件的按钮（第五个参数）。

+   描述鼠标/键盘状态的按钮（第六个参数）。

此事件可用的类型如下：

+   `QtCore.QEvent.MouseButtonPress`: 点击鼠标按钮。

+   `QtCore.QEvent.MouseButtonRelease`: 释放鼠标按钮。

+   `QtCore.QEvent.MouseMove`: 将鼠标移动到项目上。

+   `QtCore.QEvent.MouseButtonDblClick`: 双击鼠标按钮。

通过以下函数，`QMouseEvent` 类通过以下方式提高功能。

# 设置

此函数与设置与鼠标事件相关的参数/属性相关：

`mouse_event.setLocalPos(QtCore.QPointF)`: 此函数设置此鼠标事件的局部位置，位置由参数指定。

# 功能性

这些是与当前鼠标事件当前值返回相关的函数：

`mouse_event.button()`: 此函数返回导致此鼠标事件的 `QtCore.Qt.MouseButton` 类型按钮。

`mouse_event.buttons()`: 此函数返回与此鼠标事件一起生成的 `QtCore.Qt.MouseButtons` 类型按钮状态。

`mouse_event.flags()`: 此函数返回此鼠标事件的 `QtCore.Qt.MouseEventFlags` 类型标志。

`mouse_event.globalPos()`: 当事件发生时，此函数返回鼠标光标的 `QtCore.QPoint` 类型全局位置。

`mouse_event.globalX()`: 当鼠标事件发生时，此函数返回鼠标光标的全局 *x* 轴位置。

`mouse_event.globalY()`: 当鼠标事件发生时，此函数返回鼠标光标的全局 *y* 轴位置。

`mouse_event.localPos()`: 当鼠标事件发生时，此函数返回鼠标光标在项目上的 `QtCore.QPointF` 类型局部位置。

`mouse_event.pos()`: 当鼠标事件发生时，此函数返回鼠标光标在全局屏幕坐标下的 `QtCore.QPoint` 类型位置。

`mouse_event.screenPos()`: 当鼠标事件发生时，此函数返回鼠标光标在屏幕上的 `QtCore.QPointF` 类型位置。

`mouse_event.windowPos()`: 当鼠标事件发生时，此函数返回鼠标光标在窗口上的 `QtCore.QPointF` 类型位置。

`mouse_event.source()`: 此函数返回与鼠标事件源相关的 `QtCore.Qt.MouseEventSource` 类型信息。

`mouse_event.x()`: 当鼠标事件发生时，此函数返回鼠标光标在项目上的 *x* 位置。

`mouse_event.y()`: 当鼠标事件发生时，此函数返回鼠标光标在项目上的 *y* 位置。

# 事件处理器

此事件的处理器如下：

```py
def mousePressEvent(self, event):
    """Some code lines for processing..."""

def mouseReleaseEvent(self, event):
    """Some code lines for processing..."""

def mouseMoveEvent(self, event):
    """Some code lines for processing..."""

def mouseDoubleClickEvent(self, event):
    """Some code lines for processing..."""
```

# QWheelEvent

此类处理当鼠标滚轮被操作时出现的事件。这些事件是为鼠标滚轮和触摸板滚动手势生成的。此类声明的语法如下：

```py
wheel_event = QtGui.QWheelEvent(QtCore.QPointF, QtCore.QPointF,
                                QtCore.QPoint, QtCore.QPoint,
                                int, QtCore.Qt.Orientation,
                                QtCore.Qt.MouseButtons,
                                QtCore.Qt.KeyboardModifiers,
                                QtCore.Qt.ScrollPhase,
                                QtCore.Qt.MouseEventSource, bool)
```

关于滚轮事件的构建，以下参数被使用：

+   鼠标光标位置的位置（第一个参数）。

+   全局位置（第二个参数）。

+   像素增量（第三个参数）或屏幕上的滚动距离。

+   角度增量（第四个参数）或滚轮旋转距离。

+   qt4 delta（第五个参数）单向旋转。

+   qt4 方向（第六个参数）单向方向。

+   鼠标状态（第七个参数）。

+   键盘状态（第八个参数）。

+   滚动阶段（第九个参数）。

+   鼠标滚轮或手势的来源（第十个参数）。

+   反转（第十一个参数）选项。

`QWheelEvent` 类通过以下函数提高功能。

# 功能性

这些是与当前滚轮事件值返回相关的函数：

`wheel_event.angleDelta()`: 这返回滚轮旋转的 `QtCore.QPoint` 类型的距离。

`wheel_event.buttons()`: 这返回由此滚轮事件生成的 `QtCore.Qt.MouseButtons` 类型的按钮状态。

`wheel_event.globalPos()`: 这返回与该滚轮事件相关的指针的 `QtCore.QPoint` 类型的全球位置。

`wheel_event.globalPosF()`: 这返回与该滚轮事件相关的指针的 `QtCore.QPointF` 类型的全球位置。

`wheel_event.globalX()`: 这返回与该滚轮事件相关的指针的全球 *x* 轴位置。

`wheel_event.globalY()`: 这返回与该滚轮事件相关的指针的全球 *y* 轴位置。

`wheel_event.inverted()`: 如果此事件的增量值被反转，则返回 `True`。

`wheel_event.orientation()`: 这返回此滚轮的 `QtCore.Qt.Orientation` 类型的方向。

`wheel_event.phase()`: 这返回此事件的滚动阶段。

`wheel_event.pixelDelta()`: 这返回 `QtCore.QPoint` 类型的像素增量，作为屏幕上的滚动距离。

`wheel_event.pos()`: 这返回与项目相关的指针的 `QtCore.QPoint` 类型的位置。

`wheel_event.posF()`: 这返回与项目相关的指针的 `QtCore.QPointF` 类型的位置。

`wheel_event.source()`: 这返回与滚轮事件源相关的 `QtCore.Qt.MouseEventSource` 类型的信息。

`wheel_event.x()`: 这返回事件发生时与项目相关的指针的 *x* 位置。

`wheel_event.y()`: 这返回事件发生时与项目相关的指针的 *y* 位置。

# 事件处理器

此事件的处理器如下：

```py
def wheelEvent(self, event):
    """Some code lines for processing..."""
```

# QMoveEvent

此类处理与项目移动活动相关的事件。它几乎适用于所有可以实现移动的图形项目。此类的声明语法如下：

```py
move_event = QtGui.QMoveEvent(QtCore.QPoint, QtCore.QPoint)
```

关于移动事件的构建，使用新位置（第一个参数）和旧位置（第二个参数）。`QMoveEvent` 类通过以下函数提高功能。

# 功能性

这些是与当前移动事件值返回相关的函数：

`move_event.oldPos()`: 这返回移动项目的 `QtCore.QPoint` 类型的旧位置。

`move_event.pos()`: 这返回移动项目的 `QtCore.QPoint` 类型的新位置。

# 事件处理器

此事件的处理器如下：

```py
def moveEvent(self, event):
    """Some code lines for processing..."""
```

# QPaintEvent

此类处理与绘制相关的、与项目相关的事件。它几乎适用于所有可以进行绘制的图形项目。正如我们在前面的章节中描述的，Qt 库的所有图形元素都可以进行绘制，因此可以使用 `paintEvent()` 事件处理器来更新项目图形表示。此类的声明语法如下：

```py
paint_event = QtGui.QPaintEvent(QtCore.QRect)
# or
paint_event = QtGui.QPaintEvent(QtGui.QRegion)
```

关于画布事件的构建，使用参数中指定的矩形或区域进行绘制。`QPaintEvent` 类通过以下函数提高功能。

# functional

这些是与画布事件当前值相关的函数：

`paint_event.rect()`: 这返回用于更新的 `QtCore.QRect` 类型的矩形。

`paint_event.region()`: 这返回用于更新的 `QtGui.QRegion` 类型的矩形。

# 事件处理器

此事件的处理器如下：

```py
def paintEvent(self, event):
    """Some code lines for processing..."""
```

# QResizeEvent

此类处理调整项目大小时出现的事件。它几乎适用于所有可以调整项目大小的图形项目。此类的声明语法如下：

```py
resize_event = QtGui.QResizeEvent(QtCore.QSize, QtCore.QSize)
```

关于调整大小事件的构建，使用新大小（第一个参数）和旧大小（第二个参数）。`QResizeEvent` 类通过以下函数提高功能。

# functional

这些是与调整大小事件当前值相关的函数：

`resize_event.oldSize()`: 这返回正在调整大小的项目的 `QtCore.QSize` 类型的旧大小。

`resize_event.size()`: 这返回正在调整大小的项目的 `QtCore.QSize` 类型的新大小。

# 事件处理器

此事件的处理器如下：

```py
def resizeEvent(self, event):
    """Some code lines for processing..."""
```

# QTabletEvent

此类处理与平板设备功能相关的事件。此类的声明语法如下：

```py
tablet_event = QtGui.QTabletEvent(QtCore.QEvent.Type,
                                  QtCore.QPointF,
                                  QtCore.QPointF, int,
                                  int, float, int, int,
                                  float, float, int,
                                  QtCore.Qt.KeyboardModifiers,
                                  int, QtCore.Qt.MouseButton,
                                  QtCore.Qt.MouseButtons)
```

关于平板事件的构建，使用以下参数：

+   类型（第一个参数）。

+   事件发生的位置（第二个参数）。

+   绝对坐标系中的全局位置（第三个参数）。

+   设备（第四个参数）。

+   指针类型（第五个参数）。

+   对设备施加的压力（第六个参数）。

+   此设备的*x*倾斜度（第七个参数）。

+   此设备的*y*倾斜度（第八个参数）。

+   空气刷的切向压力（第九个参数）。

+   此设备的旋转（第十个参数）。

+   设备的*z*（第十一个参数）坐标。

+   键盘状态（第十二个参数）。

+   唯一标识符（第十三个参数）。

+   引起事件的按钮（第十四个参数）。

+   事件发生时的按钮状态（第十五个参数）。

`QTabletEvent` 类通过以下函数增强了功能：

# 函数式

这些是与当前平板事件返回值相关的函数：

`tablet_event.button()`: 此函数返回导致此平板事件的 `QtCore.Qt.MouseButton` 类型的按钮。

`tablet_event.buttons()`: 此函数返回与该平板事件一起生成的 `QtCore.Qt.MouseButtons` 类型的按钮状态。

`tablet_event.device()`: 此函数返回生成此平板事件的设备的类型，作为 `QtGui.QTabletEvent.TabletDevice`。

`tablet_event.globalPos()`: 此函数返回当事件发生时设备的 `QtCore.QPoint` 类型的全局位置。

`tablet_event.globalPosF()`: 此函数返回当事件发生时设备的 **`QtCore.QPointF`** 类型的全局位置。

`tablet_event.globalX()`: 此函数返回当平板事件发生时设备的全局 *x* 轴位置。

`tablet_event.globalY()`: 此函数返回当平板事件发生时设备的全局 *y* 轴位置。

`tablet_event.hiResGlobalX()`: 此函数返回该设备的高精度 *x* 位置。

`tablet_event.hiResGlobalY()`: 此函数返回该设备的**高精度** *y* 位置。

`tablet_event.pointerType()`: 此函数返回生成此事件的指针类型，作为 **`QtGui.QTabletEvent.PointerType`**。

`tablet_event.pos()`: 此函数返回与项目相关的设备的 `QtCore.QPoint` 类型的位置。

`tablet_event.posF()`: 此函数返回与项目相关的设备的 `QtCore.QPointF` 类型的位置。

`tablet_event.pressure()`: 此函数返回设备的压力，范围从 `0.0`（笔不在平板上）到 `1.0`（笔在平板上且施加最大压力）。

`tablet_event.rotation()`: 此函数返回设备的旋转角度，其中 `0` 表示笔尖指向平板顶部，正值表示向右旋转，负值表示向左旋转。

`tablet_event.tangentialPressure()`: 此函数返回由气刷工具上的手指滚轮提供的该设备的切向压力，范围在 `-1.0` 到 `1.0` 之间。

`tablet_event.uniqueId()`: 此函数返回此平板设备的唯一 ID。

`tablet_event.xTilt()`: 此函数返回设备与垂直线之间的 *x* 轴角度。

`tablet_event.yTilt()`: 此函数返回设备与垂直线之间的 *y* 轴角度。

`tablet_event.x()`: 此函数返回当事件发生时与项目相关的设备的 *x* 位置。

`tablet_event.y()`: 此函数返回当事件发生时与项目相关的设备的 *y* 位置。

`tablet_event.z()`: 此函数返回设备沿 z 轴的 *z* 位置；例如，4D 鼠标的滚轮所表示。

# 事件处理器

此事件的处理器如下：

```py
def tabletEvent(self, event):
    """Some code lines for processing..."""
```

# QTouchEvent

此类处理我们在支持触摸操作设备的触摸点上移动一个或多个触摸点时出现的事件。这些设备必须具有触摸屏或轨迹板。为了使此功能可用，控件或图形项需要将`acceptTouchEvents`属性设置为`True`。此类的声明语法如下：

```py
touch_event = QtGui.QTouchEvent(QtCore.QEvent.Type,
                              QtGui.QTouchDevice,
                              QtCore.Qt.KeyboardModifiers,
                              QtCore.Qt.TouchPointStates,
                              [QtGui.QTouchEvent.TouchPoint])
```

关于触摸事件的构建，使用了以下参数：

+   类型（第一个参数）。

+   设备（第二个参数）。

+   键盘修饰符（第三个参数）。

+   触点状态（第四个参数）。

+   触点（第五个参数）。

`QTouchEvent`类通过以下函数提高了功能。

# 设置

这些是与设置与触摸事件相关的参数/属性相关的函数：

`touch_event.setDevice(QtGui.QTouchDevice)`: 这设置参数中指定的设备，该设备将被使用。

`touch_event.setTarget(QtCore.QObject)`: 这为此事件设置参数中指定的目标（如控件）。

`touch_event.setTouchPoints([QtGui.QTouchEvent.TouchPoint])`: 这为此事件设置参数中指定的触摸点。

`touch_event.setTouchPointStates(QtCore.Qt.TouchPointStates)`: 这为此事件设置参数中指定的触摸点状态。

`touch_event.setWindow(QtGui.QWindow)`: 这为此触摸事件设置参数中指定的窗口。

# 功能性

这些是与当前触摸事件当前值返回相关的函数：

`touch_event.device()`: 这返回从发生触摸事件的`QtGui.QTouchDevice`类型的设备。

`touch_event.target()`: 这返回事件发生的`QtCore.QObject`类型的目标对象。

`touch_event.touchPoints()`: 这返回此触摸事件的触摸点列表。

`touch_event.touchPointStates()`: 这返回此`touch`事件的**`QtCore.Qt.TouchPointStates`**类型的触摸点状态。

`touch_event.window()`: 这返回发生触摸事件的**`QtGui.QWindow`**类型的窗口。

# 事件处理程序

此事件的处理程序如下：

```py
def touchEvent(self, event):
    """Some code lines for processing..."""
```

# 其他符号

Qt 库有未在此描述但我们将简要提及的事件。其中一些实现特殊或普通事件的类如下：

`QtCore.QDynamicPropertyChangeEvent`：用于动态属性更改事件。

`QtGui.QCloseEvent`：用于处理通过`closeEvent()`事件处理程序关闭的事件。

`QtGui.QHideEvent`：用于与隐藏控件相关的事件。

`QtGui.QShowEvent`：用于控件的显示事件。

`QtGui.QContextMenuEvent`：用于上下文菜单事件。

`QtGui.QExposeEvent`：用于处理通过`exposeEvent()`事件处理程序暴露的事件。

`QtGui.QFileOpenEvent`：用于与打开文件操作相关的事件。

`QtGui.QHelpEvent`：用于与控件中点的有用信息相关的事件。

`QtGui.QHoverEvent`：用于与`QGraphicsItem`相关联的鼠标事件。

`QtGui.QIconDragEvent`：用于主图标的拖动事件。

`QtGui.QInputEvent`：用于用户的输入事件。

`QtGui.QInputMethodEvent`：用于输入方法事件。

`QtGui.QNativeGestureEvent`：用于手势事件。

`QtGui.QScrollEvent`：用于滚动事件。

`QtGui.QScrollPrepareEvent`：用于滚动准备事件。

`QtGui.QShortcutEvent`：用于键组合事件。

`QtGui.QStatusTipEvent`：用于状态栏事件。

`QtGui.QWhatsThisClickedEvent`：用于处理“这是什么？”文本中的超链接。

`QtGui.QWindowStateChangeEvent`：用于窗口状态更改事件。

`QtWidgets`模块的类，如`QGestureEvent`和`QGraphicsSceneEvent`，也可以使用。

# 发送事件

在应用程序开发中，有时可能需要发送事件。这可以帮助我们创建更灵活的功能。为此，可以使用`QtCore`模块的`QCoreApplication`类的静态方法。静态方法可以按以下方式实现：

`QtCore.QCoreApplication.sendEvent(object, event)`：这是`sendEvent()`方法立即将事件发送到对象的地方。`object`是一个`QtCore.QObject`，例如小部件、按钮或其他项目，而`event`是一个`QtCore.QEvent`，例如进入事件或鼠标事件。

`QtCore.QCoreApplication.postEvent(object, event)`：这是`postEvent()`方法将事件添加到队列的地方。`object`是一个`QtCore.QObject`，例如小部件、按钮或其他项目，而`event`是一个`QtCore.QEvent`，例如进入事件或鼠标事件。此方法可以与应用程序中的线程一起使用。

# 事件示例

为了演示事件和事件处理器，让我们使我们的应用程序现代化。通常，在应用程序中，使用可用于项目的可用事件处理器。它们处理一些事件并提供额外的功能。在这里，我们将介绍向应用程序的小部件添加事件处理器的最佳方式。为此，我们需要添加/更改`u_style.py`文件中`UWid`类的某些行。让我们开始吧：

1.  首先，通过添加额外的参数来改进功能，更改`UWid`类的`__init__()`函数：

```py
...
class UWid(QtWidgets.QWidget):

    def __init__(self, parent=None, bg=color[1],
                 bgh=color[3], minw=0, minh=0,
                 maxw=None, maxh=None, fixw=None,
                 fixh=None, mrg=0, pad=0, bds="solid",
                 bdr=3, bdw=0, bdc=color[3]):
        ...
...
```

这将用于根据小部件在应用程序中的表示更改小部件的参数，类似于`UBut1`类。

1.  现在，我们需要将行添加到`UWid`类的`__init__()`函数中，如下所示：

```py
...
    def __init__(...):
        ...
        self.setMinimumWidth(minw)
        self.setMinimumHeight(minh)
        if maxw is not None:
            self.setMaximumWidth(maxw)
        if maxh is not None:
            self.setMaximumHeight(maxh)
        if fixw is not None:
            self.setFixedWidth(fixw)
        if fixh is not None:
            self.setFixedHeight(fixh)
        self.bg, self.bgh, self.mrg, self.pad = bg, bgh, mrg, pad
        self.bds, self.bdr, self.bdw, self.bdc = bds, bdr, bdw, bdc
        self.setStyleSheet(self.wid_style(self.mrg, self.pad,
                                          self.bg, self.bds,
                                          self.bdr, self.bdw,
                                          self.bgh))
...
```

这里可以选择设置固定宽度/高度、最小/最大宽度/高度、背景颜色等。

1.  现在，我们需要将`wid_style()`样式函数添加到`UWid`类中，该函数将根据发生的事件重新样式化此小部件：

```py
...
    def __init__(...):
        ...
    def wid_style(self, mrg=None, pad=None, bg=None, bds=None,
                                 bdr=None, bdw=None, bdc=None):
        style = """margin: %spx; padding: %spx;
                background-color: %s; border-style: %s;
                border-radius: %spx; border-width: %spx;
                border-color: %s;""" % (mrg, pad, bg, bds, bdr,
                                                      bdw, bdc)
        return style 
...
```

这是一个可选功能，用于减少各种事件行数。现在，我们需要添加将用于处理此小部件事件的事件处理器。

1.  将`enterEvent()`事件处理器添加到`UWid`类中，以处理与鼠标光标进入此小部件相关的事件：

```py
...
    ...
    def enterEvent(self, event):
        self.setStyleSheet(self.wid_style(self.mrg, self.pad,
                           self.bgh, self.bds, self.bdr,
                           self.bdw, self.bdc))
...
```

1.  然后，将`leaveEvent()`事件处理器添加到`UWid`类中，以处理与鼠标光标离开此小部件相关的事件：

```py
...
    ...
    def leaveEvent(self, event):
        self.setStyleSheet(self.wid_style(self.mrg, self.pad,
                           self.bg, self.bds, self.bdr,
                           self.bdw, self.bdc))
...
```

现在，如果我们运行`u_app.py`文件，我们将看到结果。我们也可以通过实验添加其他事件处理器。

# 摘要

本章考虑了信号功能的主要原则和在 GUI 应用程序中可以处理的一些常用事件。理解这些基础知识非常重要，因为这代表了任何现代应用程序的核心功能。许多 GUI 没有这些机制就变得不太有用。本章提供的最后一个示例演示了根据发生的事件对小部件进行样式化。通过这种方式，所有实现过的样式化元素都可以扩展。本章是 Qt 库的信号和事件的一个介绍。官方文档可以巩固你在这一领域的知识。

在下一章中，我们将介绍任何应用程序的另一个重要方面——线程和进程的实现。
