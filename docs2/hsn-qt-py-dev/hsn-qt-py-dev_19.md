# 线程和进程池

正如我们在整本书中看到的那样，如果我们继续向我们的 GUI 应用程序添加新功能，我们将会遇到一些问题，例如应用程序冻结、操作速度下降、同时执行的问题以及许多其他不舒适的情况。这些问题在任何多功能应用程序中都会出现。根据我们自己的经验，当我们使用微软办公软件、设计工具或其他占用大量内存资源的软件时，我们会遇到各种等待、冻结等情况。如果我们向正在创建的应用程序的功能中添加内容，在某个阶段，应用程序将变得缓慢（在最好的情况下），并且可能不会像我们希望的那样功能齐全。为什么会这样呢？在大多数情况下，几乎每个操作系统都详细说明了执行进程和线程。我们可以打开我们操作系统的任务管理器，看到各种程序作为进程（任务）运行。它们有**进程 ID**（**PID**）、名称等等。这些应用程序（进程）使用一些内部任务，并且通常有执行额外操作的线程。它们也可以使用外部任务，即独立进程。这些进程可以包含多个线程，它们可以并发执行任务。当我们的应用程序在一个进程中以一个线程运行，并且有很多任务时，可用的内存资源可能不足以运行它们。解决这个问题的方法是将任务分割成不同的线程，如果需要的话，还可以分割成不同的进程。本章将介绍 GUI 创建的这些方面。我们将探索 Qt 库工具，例如`QProcess`和`QThread`。我们还将演示可以轻松实现于 GUI 中的 Python 标准库工具。我们还将检查这些工具的优点和缺点。

在本章中，我们将涵盖以下主题：

+   进程

+   线程

+   锁

+   Python 标准库工具

# 进程

在计算中，进程是任何环境的主要部分。从广义上讲，进程是程序的实例，它执行应用程序。它可以有多个进程来执行操作任务，通常包括多个线程。如果只使用单个 CPU，则任何时刻只能执行一个进程。然而，如果 CPU 有多个核心，任务的执行可以在这些核心之间分配。但这并不意味着多核架构会并发执行所有操作进程/任务。实际上，系统是多任务的，这意味着当新任务开始时，它们可以中断已经启动的任务，可以被新任务中断，并且可以再次启动。这样，进程的执行被分割成并发操作的段。一个常见的情况是主程序有一个父进程，子进程并行执行。进程之间的通信通常是通过使用输入通道（数据流）`STDIN`（标准输入）和两个预定义的输出通道（分离的数据流）——`STDOUT`（标准输出）和`STDERR`（标准错误）来进行的。进程的通道也被称为**读**（`STDOUT`和`STDERR`）和**写**（`STDIN`）通道。Qt 库运行新进程并在应用程序中实现**进程间通信**（**IPC**）的一种方式是通过使用`QProcess`类。这个类允许我们管理应用程序的外部进程。`QProcess`是一个跨平台类，它在 Qt 库的`QtCore`模块中可用。在应用程序中，它可以用来启动外部程序作为子进程，并根据操作任务与它们通信。这种机制是一个 API，用于控制和监控子进程的状态。`QProcess`类还通过继承**`QtCore.QIODevice`**类提供了对子进程的 I/O（输入/输出）通道的访问。这些操作的简单示例是运行实现附加功能（如打开网页或运行服务器）的第三方应用程序。它也可以用于长期任务，包括循环和估计。然而，`QProcess`类不支持某些平台，例如 iOS。因此，应用程序可以使用 Python 标准库工具，如`subprocess`模块和`multiprocessing`包。

# 应用程序流程

要理解在创建和运行 GUI 或非 GUI 应用程序时发生的应用程序进程，我们需要了解 Qt 模块以及参与应用程序进程的类的继承方案。`QtCore` 模块中的 `QCoreApplication` 类继承自 `QObject` 类，并为非 GUI 应用程序创建事件循环。`QtGui` 模块中的 `QGuiApplication` 类继承自 `QCoreApplication`，并使用 GUI 应用程序的主要设置管理控制流。`QtWidgets` 模块中的 `QApplication` 类继承自 `QGuiApplication`，并基于附加功能，使用 GUI 应用程序相对于 `QWidget` 类的主要设置管理控制流。要访问应用程序对象，您可以使用全局指针：

+   `QtCore.QCoreApplication.instance()`：用于 `QCoreApplication`、`QGuiApplication` 和 `QApplication`。

+   `QtGui.QGuiApplication.qGuiApp`：用于 `QGuiApplication`。

+   `QtWidgets.QApplication.qApp`：用于 `QApplication`。

在应用程序中，可以使用 `QCoreApplication` 类的 `processEvents()` 静态方法与长期任务（循环）一起使用。长期操作通常如下所示：

```py
...
    def long_task():
        while True:
            QtWidgets.qApp.processEvents(QtCore.QEventLoop.AllEvents)
            print("HELLO")
    ...
...
```

此 `while` 指令是无限循环的，并将打印字符串，直到通过按 *Ctrl* + *Z* 停止。

# QProcess

此类提供了在应用程序中处理进程时可以使用的 Qt 库工具。使用此类在单独的进程中启动外部任务（程序），并组织与它们的通信。此类的声明语法如下：

```py
process = QtCore.QProcess()
```

`QProcess` 继承自 `QIODevice`，它是 Qt 库中所有 I/O 设备的基接口类，并通过以下功能增强了功能。

# set

这些函数设置进程的参数/属性：

`process.setArguments([str])`：此操作为在进程启动时调用的程序设置指定的参数。

`process.setEnvironment(["environment"])`：此操作为该进程设置指定的环境，该环境将与该进程一起使用。

`process.setInputChannelMode(QtCore.QProcess.InputChannelMode)`：此操作为该进程的 `STDIN` 设置指定的参数中的通道模式。可用的输入通道模式如下：

+   **`QtCore.QProcess.ManagedInputChannel`**—`0`：此进程管理运行进程的输入。

+   **`QtCore.QProcess.ForwardedInputChannel`**—`1`：此进程将主进程的输入转发到运行进程。

`process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode)`：此操作为该进程的 `STDOUT`（标准输出）设置指定的参数中的通道模式。可用的进程通道模式如下：

+   `QtCore.QProcess.SeparateChannels`—`0`：`STDOUT` 和 `STDERR` 数据在单独的内部缓冲区中。

+   `QtCore.QProcess.MergedChannels`—`1`：此操作将进程的输出合并到 `STDOUT`（标准输出）通道。

+   `QtCore.QProcess.ForwardedChannels`—`2`: 这将进程的输出转发到主进程。

+   `QtCore.QProcess.ForwardedErrorChannel`—`4`: 这将 `STDERR`（标准错误）转发到主进程。

`process.setProcessEnvironment(QtCore.QProcessEnvironment)`: 这将设置由参数指定的环境，该环境将用于此进程。

`process.setProcessState(QtCore.QProcess.ProcessState)`: 这将此进程的状态设置为参数中指定的状态。可用状态如下：

+   `QtCore.QProcess.NotRunning`—`0`: 此进程未运行。

+   `QtCore.QProcess.Starting`—`1`: 此进程正在启动，但该进程所操作的程序尚未被调用。

+   `QtCore.QProcess.Running`—`2`: 此进程正在运行且准备就绪。

+   `process.setProgram("program")`: 这将设置由参数指定的程序，该程序将在此进程中启动。

+   `process.setReadChannel(QtCore.QProcess.ProcessChannel)`: 这将为此进程设置参数中指定的通道。可用的进程通道如下：

    +   `QtCore.QProcess.StandardOutput`—`0`: 进程的 `STDOUT`（标准输出）。

    +   `QtCore.QProcess.StandardError`—`1`: 进程的 `STDERR`（标准错误）。

`process.setStandardErrorFile("path/to/the/filename", QtCore.QIODevice.OpenMode)`: 这将 `STDERR`（标准错误）重定向到文件（第一个参数），相对于文件模式（第二个参数）。

`process.setStandardInputFile("path/to/the/filename")`: 这将 `STDIN`（标准输入）重定向到参数指定的文件。

`process.setStandardOutputFile("path/to/the/filename", QtCore.QIODevice.OpenMode)`: 这将 `STDOUT`（标准输出）重定向到文件（第一个参数），相对于文件模式（第二个参数）。

`process.setStandardOutputProcess(QtCore.QProcess)`: 这将此进程的 `STDOUT`（标准输出）流管道连接到参数指定的进程的 `STDIN`（标准输入）。

`process.setWorkingDirectory("path/to/dir")`: 这将设置由参数指定的工作目录，在此进程中启动此进程。

# is

此函数返回与进程状态相关的布尔值 (`bool`)：

`process.isSequential()`: 如果此进程是顺序的，则返回 `True`。

# functional

这些函数与进程的当前值、功能变化等相关：

`process.arguments()`: 这返回最近启动的进程的命令行参数列表。

`process.atEnd()`: 如果此进程未运行且没有更多数据可读，则返回 `True`。

`process.bytesAvailable()`: 这返回可用于读取的此进程的字节数。

`process.bytesToWrite()`: 这返回可用于写入的此进程的字节数。

`process.canReadLine()`: 如果可以通过此进程读取完整的数据行，则返回 `True`。

`process.close()`: 这将终止此进程并关闭所有通信。

`process.closeReadChannel(QtCore.QProcess.ProcessChannel)`: 这关闭指定的读取通道。

`process.closeWriteChannel()`: 这在所有数据都已写入后关闭通道。

`process.environment()`: 这返回此进程的环境。

`process.error()`: 这返回最后一个发生的错误，类型为 `QtCore.QProcess.ProcessError`。可用的进程错误如下：

+   `QtCore.QProcess.FailedToStart`—`0`: 此进程启动失败。

+   `QtCore.QProcess.Crashed`—`1`: 此进程崩溃。

+   **`QtCore.QProcess.Timedout`**—`2`: 此进程超时。

+   `QtCore.QProcess.ReadError`—`3`: 从此进程读取时出现错误。

+   `QtCore.QProcess.WriteError`—`4`: 向此进程写入时出现错误。

+   `QtCore.QProcess.UnknownError`—`5`: 此进程中出现未知错误。

`process.execute("command")`: 这将在新进程中启动由参数指定的命令。

`process.execute("program", "arguments")`: 这将在新进程中以参数（第二个参数）启动程序（第一个参数）。

`process.exitCode()`: 这返回最后一个进程退出代码。

`process.exitStatus()`: 这返回最后一个进程退出状态，类型为 **`QtCore.QProcess.ExitStatus`**。可用的退出状态如下：

+   `QtCore.QProcess.NormalExit`—`0`: 此进程正常退出。

+   `QtCore.QProcess.CrashExit`—`1`: 此进程崩溃。

`process.inputChannelMode()`: 这返回此进程 `STDIN` (标准输入) 通道的 `QtCore.QProcess.InputChannelMode` 类型的通道模式。

`process.kill()`: 这将立即终止此进程并退出。

`process.nullDevice()`: 这是操作系统中的空设备，用于丢弃进程的输出流或空文件用于输入流。

`process.open(QtCore.QIODevice.OpenMode)`: 这以参数指定的模式打开进程。

`process.processChannelMode()`: 这返回此进程标准输出和标准错误通道的 `QtCore.QProcess.ProcessChannelMode` 类型的通道模式。

`process.processEnvironment()`: 这返回此进程的 `QtCore.QProcessEnvironment` 类型的环境。

`process.processId()`: 这返回运行进程的本地 ID。

`process.program()`: 这返回与此进程一起启动的最后一个程序。

`process.readAllStandardError()`: 这将从此进程的 `STDERR` 返回 `QtCore.QByteArray` 类型的所有错误数据。

`process.readAllStandardOutput()`: 这将从此进程的 `STDOUT` 返回 `QtCore.QByteArray` 类型的所有数据。

`process.readChannel()`: 这返回此进程的 **`QtCore.QProcess.ProcessChannel`** 类型的读取通道。

`process.readData(int)`: 这将读取限制在参数指定的最大大小的字节到数据中。

`process.start(QtCore.QIODevice.OpenMode)`: 这将在新进程中以参数中指定的模式启动程序。

`process.start("command", QtCore.QIODevice.OpenMode)`: 这将在新进程中以模式（第二个参数）启动命令（第一个参数）。

`process.start("program", ["arguments"], QtCore.QIODevice.OpenMode)`: 这将在新进程中以相对于模式（第三个参数）的参数（第二个参数）启动程序（第一个参数）。

`process.startDetached()`: 这将在新进程中启动程序，然后从该进程中分离出来。

`process.startDetached(int)`: 这将在新进程中以参数中指定的进程 ID 启动程序，然后从该进程中分离出来。

`process.startDetached("command")`: 这将在新进程中启动参数中指定的命令，然后从该进程中分离出来。

`process.startDetached("program", ["arguments"])`: 这将在新进程中以参数（第二个参数）启动程序（第一个参数），然后从该进程中分离出来。

`process.startDetached("program", ["arguments"], "path/to/dir")`: 这将在新进程中以参数（第二个参数）和工作目录（第三个参数）启动程序（第一个参数），然后从该进程中分离出来。

`process.state()`: 这返回此进程的 `QtCore.QProcess.ProcessState` 类型的当前状态。

`process.systemEnvironment()`: 这返回此进程的系统环境。

`process.terminate()`: 这将终止进程。

`process.waitForBytesWritten(int)`: 这将等待参数中指定的毫秒数，直到已将缓冲写入的字节数据写入。

`process.waitForFinished(int)`: 这将等待参数中指定的毫秒数，直到此进程完成，阻塞进程。

`process.waitForReadyRead(int)`: 这将等待参数中指定的毫秒数，直到有新数据可供读取，阻塞进程。

`process.waitForStarted(int)`: 这将等待参数中指定的毫秒数，直到此进程开始，阻塞进程。

`process.workingDirectory()`: 这返回用于此进程的工作目录。

# 信号

以下是与 `QProcess` 类一起可用的信号：

`process.errorOccurred(QtCore.QProcess.ProcessError)`: 当此进程发生错误时，会发出此信号，并将错误作为参数传递。

`process.finished(int, QtCore.QProcess.ExitStatus)`: 当此进程完成时，会发出此信号，并将退出代码和退出状态作为参数传递。

`process.readyReadStandardError()`: 当此进程在 `STDERR` 通道上提供新数据时，会发出此信号。

`process.readyReadStandardOutput()`: 当此进程在 `STDOUT` 通道上提供新数据时，会发出此信号。

`process.started()`: 当此进程开始时，会发出此信号。

`process.stateChanged(QtCore.QProcess.ProcessState)`: 当此进程的状态改变时，会发出此信号，并将新的进程状态作为参数传递。

# QProcessEnvironment

此类创建在启动使用进程的应用程序中的程序时可以使用的环境变量。进程的环境变量表示为键/值对的集合，例如 `["PATH=/Path/To/dir", "USER=user"]`。此类的声明语法如下：

```py
process_env = QtCore.QProcessEnvironment()
```

`QProcessEnvironment` 类通过以下函数增强了功能。

# 是

此函数返回一个与进程环境状态相关的布尔值 (`bool`)：

`process_env.isEmpty()`: 这将返回 `True`，如果这个进程环境为空且没有任何环境变量。

# 功能性

这些函数与进程环境的当前值、功能变化等相关：

`process_env.clear()`: 这将清除此进程环境中的所有键/值对。

`process_env.contains(str)`: 如果在参数指定的名称在此进程环境中找到变量，则返回 `True`。

`process_env.insert(QtCore.QProcessEnvironment)`: 这将参数指定的进程环境的内容插入到此进程环境中。

`process_env.insert(str, str)`: 这将在此进程环境中插入环境变量的键（第一个参数）和值（第二个参数）。

`process_env.keys()`: 这将返回一个包含此进程环境所有环境变量键的列表。

`process_env.remove(str)`: 这将删除包含参数指定的名称（键）的环境变量。

`process_env.swap(QtCore.QProcessEnvironment)`: 这将与此参数指定的进程环境交换此进程环境。

`process_env.systemEnvironment()`: 这将返回进程的 `**`QtCore.QProcessEnvironment`**` 类型的系统环境。

`process_env.toStringList()`: 这将此进程环境转换为键/值对的字符串列表。

`process_env.value(str, str)`: 这将返回第一个参数指定的名称（键）的值，或者如果此进程环境中不存在具有该名称的变量，则返回默认值（第二个参数）。

# 进程示例

让我们通过向我们的应用程序添加一些内容来查看使用 `QProcess` 类的示例。我们将使用本书中之前使用的 QML 脚本来可视化标签控件的应用程序部分中的按钮，并通过单击其中的一些按钮来运行这些应用程序。请注意，使用 `QProcess` 类创建的进程可能无法与某些操作系统一起工作，因此 `subprocess` 模块构造仍然是注释的。因此，为了做到这一点，我们需要将我们用于 QML 实现的目录（如 `qmls/` 和 `jscripts/`）复制到我们的工作目录中。我们还需要复制 `u_qml.qml` 文件。现在，我们需要在 `App/App_PySide2/` 和 `App/App_PyQt5/` 目录中创建新的 `apps.qml` 文件，其中将包含应用程序中按钮的 QML 脚本。`apps.qml` 文件如下所示，可以复制并从我们之前创建的 `qmls/UGrid.qml` 文件中进行修改：

1.  将 QML 导入部分、一个基本的 `Rectangle` 以及其属性和按钮的网格布局添加到 `apps.qml` 文件中：

```py
import QtQuick 2.7
import QtQuick.Layouts 1.3
import "qmls" as Uqmls

Rectangle {
    visible: true
    color: Qt.rgba(0, 0.07, 0.14, 1);
    GridLayout {
        id: grid1; anchors.fill: parent; visible: true
        function wsize() {
            if (width < 590) {return 1;} else {return 2;};
        }
        columns: wsize();
        ...
    ...
...
```

这是 QML 中所有矩形的父矩形。请注意，在编写本书时，Qt 的版本发生了变化，`QtQuick` 的版本也发生了变化。因此，在使用 QML 脚本之前，建议您检查可用的版本。

1.  添加第一个发光按钮，该按钮将用于运行视频摄像头，并将其添加到网格布局中：

```py
...
    ...
        ...
        Uqmls.URectGlow {
            id: g5; Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 20
            color: Qt.rgba(0, 0.07, 0.14, 1);
            glowcolor: Qt.rgba(0.007, 1, 1, 1);
            txglow: Qt.rgba(0.007, 0.7, 0.7, 1);
            txtext: "Camera"
            txcolor: Qt.rgba(0.2, 0.2, 0.2, 1)
            signal clicked();
            MouseArea { 
                anchors.fill: parent
                onClicked: {
                    g5.glowcolor == Qt.rgba(0.007, 1, 1, 1) ?
                    g5.glowcolor = Qt.rgba(0, 0.07, 0.14, 1) :
                    g5.glowcolor = Qt.rgba(0.007, 1, 1, 1);
                    g5.txglow == Qt.rgba(0, 0.07, 0.14, 1) ?
                    g5.txglow = Qt.rgba(0.007, 1, 1, 1) :
                    g5.txglow = Qt.rgba(0, 0.07, 0.14, 1);
                    g5.clicked();
                }
            }
        }
        ...
    ...
...
```

这是带有发光效果的第一个矩形，它将在单独的窗口中调用视频摄像头应用程序。

1.  将第二个发光按钮添加到网格布局中，该按钮将用于运行 QML 应用程序示例：

```py
...
    ...
        ...
        Uqmls.URectGlow {
            id: g6; Layout.fillWidth: true;
            Layout.fillHeight: true
            Layout.margins: 20
            color: Qt.rgba(0, 0.07, 0.14, 1);
            glowcolor: Qt.rgba(0.95, 0, 0, 1);
            txglow: Qt.rgba(0.77, 0, 0, 1);
            txtext: "QMLS"
            txcolor: Qt.rgba(0.2, 0.2, 0.2, 1)
            signal clicked();
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    g6.glowcolor == Qt.rgba(0.95, 0, 0, 1) ?
                    g6.glowcolor = Qt.rgba(0, 0.07, 0.14, 1) :
                    g6.glowcolor = Qt.rgba(0.95, 0, 0, 1);
                    g6.txglow == Qt.rgba(0, 0.07, 0.14, 1) ?
                    g6.txglow = Qt.rgba(0.77, 0, 0, 1) :
                    g6.txglow = Qt.rgba(0, 0.07, 0.14, 1);
                    g6.clicked(); 
                }
            }
        }
        ...
    ...
...
```

这是带有发光效果的第二个矩形，它将通过创建的 `clicked()` 信号调用 QML 应用程序。

1.  添加第三个发光按钮，该按钮将用于运行 Jupyter Notebook，并将其添加到网格布局中：

```py
...
    ...
        ...
        Uqmls.URectGlow {
            id: g7; Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 20
            color: Qt.rgba(0, 0.07, 0.14, 1);
            glowcolor: Qt.rgba(0,0.95,0.37,1);
            txglow: Qt.rgba(0,0.47,0.37,1);
            txtext: "JUPYTER"
            txcolor: Qt.rgba(0.2, 0.2, 0.2, 1)
            signal clicked();
            MouseArea {
                anchors.fill: parent 
                onClicked: {
                    g7.glowcolor == Qt.rgba(0, 0.95, 0.37, 1) ?
                    g7.glowcolor = Qt.rgba(0, 0.07, 0.14, 1) :
                    g7.glowcolor = Qt.rgba(0, 0.95, 0.37, 1);
                    g7.txglow == Qt.rgba(0, 0.07, 0.14, 1) ?
                    g7.txglow = Qt.rgba(0, 0.47, 0.37, 1) :
                    g7.txglow = Qt.rgba(0, 0.07, 0.14, 1);
                    g7.clicked(); 
                }
            }
        }
        ...
    ...
...
```

带有发光效果的第三个矩形将在使用 `QProcess` 类创建的分离进程中启动 Jupyter Notebook。

1.  最后，添加第四个发光按钮，该按钮将用于运行默认的网页浏览器，并将其添加到网格布局中，从而完成 QML 文件：

```py
...
    ...
        ...
        Uqmls.URectGlow {
            id: g8; Layout.fillWidth: true; Layout.fillHeight: true
            Layout.margins: 20
            color: Qt.rgba(0, 0.07, 0.14, 1);
            glowcolor: Qt.rgba(1, 1, 1, 1);
            txglow: "grey";
            txtext: "WEB"
            txcolor: Qt.rgba(0.2, 0.2, 0.2, 1)
            signal clicked();
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    g8.glowcolor == Qt.rgba(1, 1, 1, 1) ?
                    g8.glowcolor = Qt.rgba(0, 0.07, 0.14, 1) :
                    g8.glowcolor = Qt.rgba(1, 1, 1, 1);
                    g8.txglow == Qt.rgba(0, 0.07, 0.14, 1) ?
                    g8.txglow = "grey" :
                    g8.txglow = Qt.rgba(0, 0.07, 0.14, 1);
                    g8.clicked();
                }
            }
        }
    }
}
```

带有发光效果的第四个矩形将在单独的进程中打开系统的默认网页浏览器。

现在，我们需要在 `u_app.py` 文件中进行一些更改，以实现 QML 脚本并在运行新进程的应用程序中运行，例如 QML 应用程序和其他第三方程序：

1.  首先，将以下行添加到每个文件的导入部分：

+   将以下内容添加到 PySide2 的 `u_app.py` 文件中：

```py
...
from PySide2 import QtQuickWidgets
...
```

+   将以下内容添加到 PyQt5 的 `u_app.py` 文件中：

```py
...
from PyQt5 import QtQuickWidgets
...
```

1.  然后，根据以下内容修改或添加 `UApp` 类的 `__init__()` 函数中的某些行：

```py
...
class UApp(UWindow, UTools):

    def __init__(self, parent=None):
        ...
        self.apps = QtQuickWidgets.QQuickWidget(self.twid1)
        self.apps.setSource(QtCore.QUrl("apps.qml"))
        self.properties = self.apps.rootObject()
        ...
        self.qmlbut1 = self.properties.childItems()[0].childItems()[0]
        self.qmlbut1.clicked.connect(self.video_camera)
        self.qmlbut2 = self.properties.childItems()[0].childItems()[1]
        self.qmlbut2.clicked.connect(self.qml_apps)
        self.qmlbut3 = self.properties.childItems()[0].childItems()[2]
        self.qmlbut3.clicked.connect(self.jupyter)
        self.qmlbut4 = self.properties.childItems()[0].childItems()[3]
        self.qmlbut4.clicked.connect(self.web_browse)
        ...
        self.qapp1 = 0
        self.qapp2 = 0
        self.qapp3 = 0
        self.qapp4 = 0
    ...
...
```

在 QML 文件中创建的信号将被用于调用与将要运行的新任务相关联的函数。现在，我们需要向 `UApp` 类添加一些函数，以实现 QML 并使用指定的应用程序运行进程。

1.  添加调整大小事件处理程序以调整视图中的 QML 元素大小：

```py
...
    ...
    def resizeEvent(self, event):
        self.properties.setWidth(
             float(self.tabwid.currentWidget().width()))
        self.properties.setHeight(
             float(self.tabwid.currentWidget().height()))
    ...
...
```

此事件处理程序将在使用鼠标或其他方式调整应用程序窗口大小时，调整包含 QML 元素的窗口的标签大小。

1.  添加第一个进程以运行视频摄像头设备：

```py
...
    ...
    def video_camera(self):
        self.qapp1 += 1
        if self.qapp1 == 1:
            # subprocess.Popen(["python", r"u_media.py"])
            self.approc1 = QtCore.QProcess()
            self.approc1.start("python", ["u_media.py"])
        if self.qapp1 == 2:
            self.approc1.kill()
            self.qapp1 = 0
    ...
...
```

这将在新进程中启动摄像头小部件。Qt 库提供的摄像头设备功能可能因版本而异。如果您使用的是需要 root 权限的操作系统，您需要根据这些要求启动此功能。

1.  添加第二个进程，该进程将运行我们之前创建的 QML 应用程序：

```py
...
    ...
    def qml_apps(self):
        self.qapp2 += 1
        if self.qapp2 == 1:
            # subprocess.Popen(["python", r"u_qml.py"])
            self.approc2 = QtCore.QProcess()
            self.approc2.start("python", ["u_qml.py"])
        if self.qapp2 == 2:
            self.approc2.kill()
            self.qapp2 = 0
    ...
...
```

这将在新进程中运行 QML 应用程序。我们还需要在 `App/App_PySide2/` 和 `App/App_PyQt5/` 目录中创建一个 `u_qml.py` 文件，并添加以下行。

在每个 `import` 部分添加以下行：

+   将以下行添加到 PySide2 的 `u_qml.py` 文件中：

```py
...
from PySide2 import QtWidgets, QtCore
from PySide2.QtQml import QQmlApplicationEngine
import sys
...
```

+   将以下行添加到 PyQt5 的 `u_qml.py` 文件中：

```py
...
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtQml import QQmlApplicationEngine
import sys
...
```

+   最后，将以下用于在应用程序中启动 QML 的行添加到两个文件中：

```py
...
app = QtWidgets.QApplication(sys.argv)
qwid = QQmlApplicationEngine()
qwid.load(QtCore.QUrl('u_qml.qml'))
sys.exit(app.exec_())
...
```

我们以类似的方式更改此文件，即不使用类。然而，我们在 第二章，*QML 概述*中创建的 `u_qml.py` 文件也可以通过额外的修改来使用。

1.  为了在指定的浏览器中运行 Jupyter Notebook（可以更改为您喜欢的浏览器；如果没有指定，将使用默认浏览器）：

```py
...
    ...
    def jupyter(self):
        self.qapp3 += 1
        if self.qapp3 == 1:
            # subprocess.Popen(["jupyter", ["notebook", "-- 
            #                             browser=firefox"])
            self.approc3 = QtCore.QProcess()
            self.approc3.start("jupyter", ["notebook", "--
                                         browser=firefox"])
        if self.qapp3 == 2:
            self.approc3.kill()
            self.qapp3 = 0
    ...
...
```

此函数将在新进程中启动 Jupyter Notebook，它将在 Firefox 浏览器中显示。如果我们想在我们的默认浏览器中打开笔记本，可以通过在 `start()` 函数的浏览器参数中不指定任何浏览器来实现。或者，我们可以通过指定浏览器参数来使用我们喜欢的浏览器。

1.  最后，添加第四个进程，该进程将使用 Python 标准库的 `webbrowser` 模块运行默认的网页浏览器，并指定 URL：

```py
...
    ...
    def web_browse(self):
        self.qapp4 += 1
        if self.qapp4 == 1:
            # subprocess.Popen("python", ["-m", "webbrowser", "-n", 
            #                            "https://www.python.org"])
            self.approc4 = QtCore.QProcess()
            self.approc4.start("python", ["-m", "webbrowser", "-n",
                                          "https://www.python.org"])
        if self.qapp4 == 2:
            self.approc4.kill()
            self.qapp4 = 0
    ...
...
```

在这里，我们指定了 `QProcess` 类的 `start()` 函数中的命令，类似于 Python 标准库中 `subprocess` 模块的构建。所有这些用于启动进程的函数都模拟了按钮的切换。当按钮被点击时，进程将启动。当不再被点击时，将调用 `kill()` 函数。这些函数是可选的，可以用来启动另一个程序。

# 线程

与应用程序和程序的环境相关的执行操作的下一个元素是线程。在本质上，线程与进程非常相似，但也有一些区别。线程以并行的方式做事。从广义上讲，执行线程是操作系统操作调度器可以独立管理的指令序列。在目前大多数可用的应用程序中，线程是进程的一个组成部分。在我们周围的世界里，我们通常遇到的多任务操作系统提供了多线程模型，允许在单个进程的上下文中存在多个线程。那么，我们如何描述线程呢？当你使用你喜欢的操作系统，或者运行你喜欢的编辑器，例如，Anaconda 工具集包含的 Spyder 编辑器时，你想要找到最新的新闻并打开一个网页浏览器。假设你打开你的媒体播放器来听音乐或看视频——这些应用程序通常在你做类似的事情时使用。这是一个几个进程在使用执行的多任务模型并行工作的例子。此外，正如我们所看到的，这些进程中的每一个都有其他在单个进程中并行执行的内部进程。当你使用编辑器的文本字段时，你可以同时运行这个编辑器的另一个选项。这种行为描述了应用程序中线程的意义——单个进程内的并发。

我们如何描述并发执行呢？正如我们在本章的*进程*部分所描述的，在单核 CPU 上的并行执行只是一个假象。对于进程来说，这就像当另一个进程开始时对进程的打断，然后当另一个进程开始时，另一个进程又被打断。这种情况可以推广到单个进程的程序段执行。当一个进程开始时，它总是执行一个代码段。这被称为进程有一个线程。我们可以将进程的执行分割成两个代码段，这样看起来就像有两个不同的代码序列同时在运行。当我们有一个单核 CPU 时，并发是通过与进程的并行执行类似的方式实现的。如果我们有一个多核 CPU，这些线程可以被分配到多个核心上。这实际上就是并发。

关于使用 Qt（尽管这不仅仅与 Qt 相关）进行 GUI 应用程序开发，我们需要了解两种类型的线程。第一种类型是作为应用程序主线程使用的 GUI 线程。当我们运行应用程序时，通过调用 `QtCore.QApplication.exec()` 函数启动此线程。所有 GUI 组件，如小部件，以及一些类，如 `QPixmap`，都必须在此线程中运行。第二种类型是辅助线程，也称为工作线程，用于卸载主线程并运行长期任务。本书中展示的所有提供额外功能（如 pandas）的线程都是工作线程。每个线程都有自己的堆栈（调用历史和局部变量），其大小通常由操作系统定义。线程共享相同的地址空间。如果我们想在应用程序中使用线程，我们需要了解以下内容：

+   实际并发执行可用的线程数等于可用的 CPU 核心数

+   多个线程不能同时访问同一对象

+   所有线程对象只有在其他线程与它们无关且对象没有与其他线程的隐式耦合时才能安全使用

+   我们不能从工作线程中更改 GUI 线程中的某个内容

Qt 库提供了几个用于在应用程序中处理线程的类，例如 `QThread` 和 `QThreadPool`。`QThread` 类的优点是我们可以使用信号机制更改 GUI 组件。Qt 库提供了诸如 *可重入* 和 *线程安全* 这样的文档术语，用于标记类和函数，以指示它们如何在多线程应用程序中使用。术语 **线程安全** 意味着即使调用使用共享数据，此方法也可以从多个线程中调用，因为所有对共享数据的引用都是序列化的。

术语 **可重入** 意味着此方法可以从多个线程中调用，但前提是每个调用都使用自己的数据。在本节中，我们将介绍 `QThread` 和 `QThreadPool` 类。我们还建议学习 `QtConcurrent` 模块和 `WorkerScript` QML 类型。

# QThread

此类管理应用程序中可用的线程。它是平台无关的，并提供了一种将任务执行分离到不同事件循环的方法。此类声明的语法如下：

```py
thread = QtCore.QThread()
```

`QThread` 从 `QObject` 类继承，并通过以下函数增强了功能。

# 设置

这些函数将参数/属性设置到线程中：

`thread.setEventDispatcher(QtCore.QAbstractEventDispatcher)`: 这将设置用于此线程的事件分发器，该分发器由参数指定。

`thread.setPriority(QtCore.QThread.Priority)`: 这将设置用于此线程运行的优先级，该优先级由参数指定。可用的优先级如下：

+   `QtCore.QThread.IdlePriority`—`0`: 此线程仅在没有任何其他线程运行时运行。

+   `QtCore.QThread.LowestPriority`—`1`: 此线程具有最低的运行优先级。

+   `QtCore.QThread.LowPriority`—`2`: 此线程具有低运行优先级。

+   `QtCore.QThread.NormalPriority`—`3`: 此线程具有正常（操作系统的默认值）的运行优先级。

+   `QtCore.QThread.HighPriority`—`4`: 此线程具有高运行优先级。

+   `QtCore.QThread.HighestPriority`—`5`: 此线程具有最高的运行优先级。

+   `QtCore.QThread.TimeCriticalPriority`—`6`: 此线程尽可能频繁地与其他线程运行。

+   `QtCore.QThread.InheritPriority`—`7`: 此线程以与创建线程相同的优先级运行。

`thread.setStackSize(int)`: 这将此线程的最大堆栈大小设置为参数中指定的字节数。

`thread.setTerminationEnabled(bool)`: 如果此参数为 `True`，则启用线程的终止。

# is

这些函数返回与线程状态相关的布尔值，`bool`：

`thread.isFinished()`: 如果此线程已完成，则返回 `True`。

`thread.isInterruptionRequested()`: 如果请求中断此线程，则返回 `True`。

`thread.isRunning()`: 如果此线程正在运行，则返回 `True`。

# functional

这些函数与线程的当前值、功能变化等相关：

`thread.currentThread()`: 这返回当前操作线程的 `QtCore.QThread` 类型的对象。

`thread.currentThreadId()`: 这返回当前线程的处理程序。

`thread.eventDispatcher()`: 这返回当前操作线程的 `QtCore.QAbstractEventDispatcher` 类型的事件分发器对象。

`thread.exec_()`: 这通过进入事件循环来执行此线程。

`thread.exit(int)`: 这以参数中指定的返回代码退出线程的事件循环（`0` 表示成功；任何非零值表示错误）。

`thread.idealThreadCount()`: 这返回系统上可以使用的理想线程数。

`thread.loopLevel()`: 这返回此线程的事件循环级别。

`thread.msleep(int)`: 在参数中指定的毫秒数过去后，此线程会进入休眠状态。

`thread.priority()`: 这返回用于此线程的 `QtCore.QThread.Priority` 类型的优先级。

`thread.quit()`: 这以返回代码 `0` 退出此线程的事件循环。

`thread.requestInterruption()`: 这请求中断此线程

`thread.run()`: 这调用创建的线程的 `run()` 函数。

`thread.sleep(int)`: 在参数中指定的秒数过去后，此线程会进入休眠状态。

`thread.stackSize()`: 这返回此线程的最大堆栈大小，以字节数表示。

`thread.start(QtCore.QThread.Priority)`: 这启动此线程并开始执行参数中指定的 `run()` 函数指令，指定优先级。

`thread.terminate()`: 这将终止此线程的执行。

`thread.usleep(int)`: 该线程将在参数指定的微秒数内休眠。

`thread.wait(int)`: 这将阻塞此线程，直到它完成执行或等待参数指定的毫秒数。

`thread.yieldCurrentThread()`: 这将使当前线程的执行权交由另一个线程，操作系统将切换到该线程。

# 信号

这些是 `QThread` 类的可用信号：

`thread.finished()`: 在此线程执行完毕之前发出此信号。

`thread.started()`: 在此线程开始执行之前发出此信号。

# QThreadPool

此类管理可用于应用程序的线程集合。Qt 应用程序有一个全局的 `QThreadPool` 对象，用于管理应用程序中使用的主（**GUI**）和附加（**工作**）线程。可以通过调用 `globalInstance()` 静态函数来访问现有的线程池。此类的声明语法如下：

```py
thread_pool = QtCore.QThreadPool()
```

`QThreadPool` 从 `QObject` 类继承，并通过以下函数增强了其功能。

# 设置

这些函数设置线程池的参数/属性：

`thread_pool.setExpiryTimeout(int)`: 这将为线程池设置过期超时时间（以毫秒为单位），指定在参数中，之后所有未使用的线程都被视为过期并退出。

`thread_pool.setMaxThreadCount(int)`: 这将设置池中将使用的最大线程数，该数由参数指定。

`thread_pool.setStackSize(int)`: 这将为池中的工作线程设置最大堆栈大小，以字节数的形式指定在参数中。

# 函数式

这些函数与线程池的当前值、功能变化等相关：

`thread_pool.activeThreadCount()`: 这返回池中活动线程的数量。

`thread_pool.cancel(QtCore.QRunnable)`: 这将从队列中移除尚未启动或指定在参数中的可运行对象。

`thread_pool.clear()`: 这将从队列中移除尚未启动的可运行对象。

`thread_pool.expiryTimeout()`: 这返回线程池的过期超时时间，或所有未使用的线程被视为过期并退出后的时间（以毫秒为单位）。

`thread_pool.globalInstance()`: 这返回线程池的 **`QtCore.QThreadPool`** 类的全局实例。

`thread_pool.maxThreadCount()`: 这将返回池中使用的最大线程数。

`thread_pool.releaseThread()`: 通过 `reserveThread()` 函数之前保留的线程被释放。

`thread_pool.reserveThread()`: 这将保留线程。

`thread_pool.stackSize()`: 这返回池中工作线程的最大堆栈大小（以字节数表示）。

`thread_pool.start(QtCore.QRunnable, int)`：这会保留这个线程，并使用它来运行一个可运行的（第一个参数）对象，具有优先级（第二个参数），以确定队列的执行顺序。

`thread_pool.tryStart(QtCore.QRunnable)`：这尝试保留线程以运行在参数中指定的可运行对象。

`thread_pool.tryTake(QtCore.QRunnable)`：这尝试从队列中移除在参数中指定的可运行对象。

`thread_pool.waitForDone(int)`：这将等待在参数中指定的毫秒数超时，等待所有线程退出，并从池中移除所有线程（默认超时为`-1`或忽略）。

# 线程示例

为了查看线程的示例，我们将修改与 CouchDB 和 MongoDB 功能相关的先前代码。在`u_tools.py`文件中，让我们添加具有线程和函数的类，这些函数将使线程依赖于应用程序中的任务：

1.  添加用于 MongoDB 服务器的线程类：

```py
...
class MongoThread(QThread):

    sig1 = app_signal(object, str)

    def __init__(self, parent=None):
        super(MongoThread, self).__init__(parent)

    def on_source(self, datas):
        self.datas = datas
    ...
...
```

这是课程的第一个部分，它有一个会发射数据的信号。它还包含`on_source()`函数，该函数将接收数据。

1.  现在，我们需要将实现我们的线程功能的`run()`函数添加到`MongoThread`类中：

```py
...
    ...
    def run(self):
        try:
            import pymongo
            try:
                self.client = pymongo.MongoClient('localhost',
                                                         27017)
                self.db = self.client['umongodb']
            except pymongo.errors as err:
                self.sig1.emit('', str(err))
            if self.datas[1] == "insert":
                posts = self.db.posts
                posts.insert_one(self.datas[0])
            if self.datas[1] == "select":
                dbdata = self.db.posts.find()
                self.sig1.emit(dbdata, '')
            if self.isFinished():
                self.quit()
        except Exception as err:
            self.sig1.emit('', str(err))
            if self.isFinished():
                self.quit()
...
```

线程的`run()`函数将启动线程功能。它将尝试连接到 MongoDB 服务器。当连接成功后，它将尝试向/从数据库插入/选择值。

1.  添加用于 CouchDB 服务器的`CouchThread`类：

```py
...
class CouchThread(QThread):

    sig1 = app_signal(object, str)

    def __init__(self, parent=None):
        super(CouchThread, self).__init__(parent)

    def on_source(self, datas):
        self.datas = datas
    ...
...
```

这是课程的第一个部分，它有一个会发射数据的信号，以及`on_source()`函数，该函数将接收数据。

1.  现在，将实现线程功能的`run()`函数添加到`CouchThread`类中：

```py
...
    ...
    def run(self):
        try:
            import couchdb
            try:
                self.couch = couchdb.Server(
                                "http://127.0.0.1:5984/")
                self.db = self.couch["u_couchdb"]
            except Exception as err:
                self.sig1.emit('', str(err))
            if self.datas[1] == "insert":
                self.db.save(self.datas[0])
            if self.datas[1] == "select":
                self.sig1.emit(self.db, '')
            if self.isFinished():
                self.quit()
        except Exception as err:
            self.sig1.emit('', str(err))
            if self.isFinished():
                self.quit()
...
```

这个线程的`run()`函数将启动线程功能。它将尝试连接到 CouchDB 服务器。当连接成功后，它将尝试向/从数据库插入/选择值。现在，我们需要添加将在应用程序中提供线程功能的功能。首先，我们需要添加在`UTools`类中将使用的信号。

1.  用于通信的信号看起来像这样：

```py
...
class UTools(object):

    pandas_sig1 = app_signal(list)
    pandas_sig2 = app_signal(list)
    mongo_sig1 = app_signal(list)
    mongo_sig2 = app_signal(list)
    couch_sig1 = app_signal(list)
    couch_sig2 = app_signal(list)

    def __init__(self):
        ...
    ...
...
```

这些新信号将在`UTools`类的功能与为 Mongo 和 Couch 数据库实现创建的线程之间进行通信。然后，我们需要将`u_tools.py`文件中的`UTools`类的先前创建的函数修改为写入和读取 Couch 和 Mongo 数据库。

1.  修改`UTools`类的`mongo_insert()`函数：

```py
...
    ...
    def mongo_insert(self, username=None, email=None,
                                    passw=None, data=None):
        datas = [{"User Name": username, "Email": email,
        "Password": passw, "Data": data}, "insert"]
        self.mongo_thread1 = MongoThread()
        self.mongo_sig1.connect(self.mongo_thread1.on_source)
        self.mongo_sig1.emit(datas)
        self.mongo_thread1.start()
    ...
...
```

这个函数将使用线程写入`datas`列表中指定的数据。这个线程的实例将被使用，以及用于向线程发射这些数据的信号的连接。这将启动线程（默认优先级为`InheritPriority`）。

1.  修改`UTools`类的`mongo_select()`函数：

```py
...
    ...
    def mongo_select(self):
        datas = [{}, "select"]
        self.mongo_thread2 = MongoThread()
        self.mongo_sig2.connect(self.mongo_thread2.on_source)
        self.mongo_sig2.emit(datas)
        self.mongo_thread2.start()
        return self.mongo_thread2
    ...
...
```

此函数将使用线程读取 MongoDB 实例的数据。线程实例将与用于向线程发射此数据的信号连接创建。发射的数据将是一个包含空字典（可选）和指示读取操作的字符串的列表。然后，线程开始运行。

1.  现在，修改`UTools`类的`couch_insert()`函数：

```py
...
    ...
    def couch_insert(self, username=None, email=None,
                                passw=None, data=None):
        datas = [{"User Name": username,
          "User email": email,
          "User password": passw,
                  "User Data": data}, "insert"]
        self.couch_thread1 = CouchThread()
        self.couch_sig1.connect(self.couch_thread1.on_source)
        self.couch_sig1.emit(datas)
        self.couch_thread1.start()
    ...
...
```

此函数将使用线程将`datas`列表中指定的数据写入 Couch 数据库。线程实例将被使用，同时连接用于向线程发射此数据的信号。现在，线程将开始运行。

1.  修改`UTools`类的`couch_select()`函数：

```py
...
    ...
    def couch_select(self):
        datas = [{}, "select"]
        self.couch_thread2 = CouchThread()
        self.couch_sig2.connect(self.couch_thread2.on_source)
        self.couch_sig2.emit(datas)
        self.couch_thread2.start()
        return self.couch_thread2
    ...
...
```

此函数将使用线程读取数据。线程实例将被创建，同时连接用于向线程发射此数据的信号。发射的数据将是一个包含空字典（可选）和指示读取操作的字符串的列表。

现在，我们需要修改与从 Couch 和 Mongo 数据库读取相关的`u_app.py`文件中的`UApp`类的`data()`函数。让我们开始吧：

1.  修改`UApp`类的`data()`函数：

```py
...
    ...
        ...
        if self.actx == "MongoDB":
            try:
                mongo_data = self.mongo_select()
                def to_field(dbdata, er):
                    if er == '':
                        for dtx in dbdata:
                            self.text_edit.append(
                                    "%s\n%s\n%s\n%s" % (
                                    dtx["User Name"],
                                    dtx["Email"],
                                    dtx["Password"],
                                    dtx["Data"]))
                    else:
                        self.stat_bar.showMessage(
                                    self.actx + ' ' + er)
                mongo_data.sig1.connect(
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

当在顶部面板中选择`数据`|`MongoDB`选项时，这些行提供 MongoDB 功能。它接收到线程对象时调用`mongo_select()`函数，通过线程的信号连接到嵌套函数，并将数据库数据放入文本字段。如果发生错误，它将在应用程序的状态栏中显示此错误。

1.  修改`UApp`类的`data()`函数：

```py
...
    ...
        ...
        if self.actx == "CouchDB":
            try:
                couch_data = self.couch_select()
                def to_field(dbdata, er):
                    if er == '':
                        for dtx in dbdata.view(
                                        "_all_docs",
                                        include_docs=True):
                            self.text_edit.append(
                                 "%s\n%s\n%s\n%s" % (
                                 dtx["doc"]["User Name"],
                                 dtx["doc"]["User email"],
                                 dtx["doc"]["User password"],
                                 dtx["doc"]["User Data"]))
                    else:
                        self.stat_bar.showMessage(
                                      self.actx + ' ' + er)
                couch_data.sig1.connect(
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

当选择`CouchDB`选项时，这些行提供 CouchDB 功能。它接收到线程对象时调用`couch_select()`函数，通过线程的信号连接到嵌套函数，并将数据库数据放入文本字段。如果发生错误，它将在应用程序的状态栏中显示此错误。

这些示例在使用方面有一些限制。请注意，当我们使用此功能时，我们需要确保 CouchDB 和 MongoDB 服务器已配置并正在运行，并且数据库已在该目录中创建。

# 锁定

在应用程序中使用线程时，可能会出现与多个线程访问类似源相关的问题，或者当您需要锁定代码的一部分执行时。Qt 库提供了一些类来解决此问题，例如`QMutex`、`QMutexLocker`、`QReadWriteLock`、`QSemaphore`和`QWaitCondition`。在这里，我们将描述其中的一些。然而，请注意，锁定某些源可能会创建与阻塞 GUI 线程或另一个线程相关的其他问题。因此，在应用程序中使用锁不是一件简单的事情，在实现之前，我们需要仔细考虑。

# QMutex

这个类允许线程之间的访问序列化。为了强制**互斥**（**mutex**），线程锁定互斥锁以获取对资源的访问。这个类的声明语法如下：

```py
mutex = QtCore.QMutex(QtCore.QMutex.RecursionMode)
```

可以通过参数指定递归模式来构造互斥锁。可用的递归模式如下：

+   `QtCore.QMutex.Recursive`—`0`: 这个线程能够多次锁定相同的互斥锁。这个互斥锁将不会解锁，直到相应的`unlock()`函数调用次数达到相应的数量。

+   `QtCore.QMutex.NonRecursive`—`1`: 这个线程只能锁定一次相同的互斥锁。

`QMutex`从`QBasicMutex`类继承，并通过以下函数增强了其功能。

# 是

这个函数返回一个与互斥锁状态相关的布尔值（`bool`）：

`mutex.isRecursive()`: 如果这个互斥锁具有递归模式，则返回`True`。

# 功能性

这些函数与互斥锁的当前值、功能变化等相关：

`mutex.lock()`: 这将锁定互斥锁。

`mutex.tryLock(int)`: 这将尝试锁定互斥锁。如果另一个线程已经锁定了互斥锁，它将在锁定之前等待指定参数中的毫秒数（默认超时为`0`）。

`mutex.unlock()`: 这将解锁互斥锁。

# QMutex 示例

当使用线程时，通常可以在应用程序中实现互斥锁。让我们用以下代码行来演示这一点：

```py
...
class WPandas(QThread):
    ...
    def __init__(self, parent=None):
        ...
        self.mutex = QMutex()
    ...
    def run(self):
        self.mutex.lock()
        try:
            ...
        except Exception as err:
            ...
        self.mutex.unlock()
...
```

`__init__()`函数创建了从`QtCore`模块导入的`QMutex`类的实例。在线程的`run()`函数内部，我们添加了包含互斥锁的`lock()`和`unlock()`方法的行，它们将分别锁定和解锁它们之间的代码。

# QMutexLocker

这个类提供了锁定和解锁互斥锁的附加方便功能。这个类的声明语法如下：

```py
mutex_locker = QtCore.QMutexLocker(QtCore.QBasicMutex)
```

`QMutexLocker`类通过以下函数提高了功能。

# 功能性

这些函数与互斥锁锁定的当前值、功能变化等相关：

`mutex_locker.mutex()`: 这将返回由这个互斥锁锁定器操作的`QtCore.QMutex`类型的互斥锁。

`mutex_locker.relock()`: 这将重新锁定已解锁的互斥锁锁定器。

`mutex_locker.unlock()`: 这将解锁互斥锁锁定器。

# QMutexLocker 示例

之前关于互斥锁的例子有一些缺陷。如果在`unlock()`方法执行代码之前发生异常，互斥锁可能会永久锁定。为了解决这个问题，我们可以使用`QMutexLocker`类。让我们用之前创建的互斥锁来演示这一点：

```py
...
class WPandas(QThread):
    ...
    def __init__(self, parent=None):
        ...
        self.mutex = QMutex()
    ...
    def run(self):
        mutex_locker = QMutexLocker(self.mutex)
        try:
            ...
        except Exception as err:
            ...
...
```

在 `run()` 函数中，我们创建了一个从 `QtCore` 模块导入的 `QMutexLocker` 类的实例，并将指定的互斥锁作为参数。在作用域结束时，类的析构函数将被调用，并且这个锁将自动释放。这也可以用于构建上下文管理器，例如指令；例如，*with*/*as*。将前面的代码行更改为以下内容：

```py
...
class WPandas(QThread):
    ...
    def __init__(self, parent=None):
        ...
        self.mutex = QMutex()
    ...
    def run(self):
        with QMutexLocker(self.mutex):
            try:
                ...
            except Exception as err:
                ...
...
```

# QSystemSemaphore

此类为在应用程序中与线程和多个进程一起工作提供了一般计数系统信号量。此类的声明语法如下：

```py
system_semaphore = QtCore.QSystemSemaphore(str, int,
                         QtCore.QSystemSemaphore.AccessMode)
```

此系统信号量可以使用键（第一个参数）和系统依赖的初始化资源数量（第二个参数）以及访问模式（第三个参数）进行构造。可用的访问模式如下：

+   `QtCore.QSystemSemaphore.Open`—`0`: 如果系统信号量存在，则其初始资源计数不会被重置；否则，它将被创建。

+   `QtCore.QSystemSemaphore.Create`—`0`: 系统信号量将被创建。

`QSystemSemaphore` 类通过以下函数增强了功能。

# set

此函数设置系统信号量的参数/属性：

`system_semaphore.setKey(str, int, QtCore.QSystemSemaphore.AccessMode)`: 这使用键（第一个参数）和系统依赖的初始化资源数量（第二个参数）以及访问模式（第三个参数）重建系统信号量对象。

# functional

这些函数与系统信号量的当前值、功能变化等相关：

`system_semaphore.acquire()`: 这将获取由该系统信号量守护的资源之一。

`system_semaphore.error(QtCore.QSystemSemaphore.SystemSemaphoreError)`: 如果此系统信号量发生错误，则返回错误类型的值。可能发生的错误如下：

+   `QtCore.QSystemSemaphore.NoError`—`0`: 无错误。

+   `QtCore.QSystemSemaphore.PermissionDenied`—`1`: 调用者权限不足。

+   `QtCore.QSystemSemaphore.KeyError`—`2`: 已指定无效的键。

+   `QtCore.QSystemSemaphore.AlreadyExists`—`3`: 指定的键的系统信号量已存在。

+   `QtCore.QSystemSemaphore.NotFound`—`4`: 指定的系统信号量无法找到。

+   `QtCore.QSystemSemaphore.OutOfResources`—`5`: 内存不足。

+   `QtCore.QSystemSemaphore.UnknownError`—`6`: 错误未知。

`system_semaphore.errorString()`: 这返回错误文本描述。

`system_semaphore.key()`: 这返回此系统信号量的键或从其他进程访问此系统信号量的名称。

`system_semaphore.release(int)`: 这将释放由该系统信号量守护的指定参数数量的资源。

# QSemaphore

这个类为应用程序中与线程一起工作创建了一般计数信号量。这个类的声明语法如下：

```py
semaphore = QtCore.QSemaphore(int)
```

信号量可以通过指定参数中指定的资源数量进行初始化。`QSemaphore`类通过以下函数增强了功能。

# functional

这些函数与信号量的当前值、功能变化等相关：

`semaphore.acquire(int)`: 这获取由该信号量保护的指定数量（参数中指定）的资源。

`semaphore.available()`: 这返回此信号量可用的资源数量。

`semaphore.release(int)`: 这释放由该信号量保护的指定数量的资源。

`semaphore.tryAcquire(int)`: 这尝试获取由该信号量保护的指定数量的资源。

`semaphore.tryAcquire(int, int)`: 这尝试获取由该信号量保护的资源数量（第一个参数）在指定的时间内（第二个参数）。

信号量构造通常用于控制多个线程对资源的访问。在 PyQt5/PySide 绑定和`/examples/threads/`文件夹中的线程示例中，可以找到一个很好的信号量示例，该文件夹位于`site-packages/`，这与使用的绑定相关。

# QWaitCondition

这个类通过提供条件变量来同步线程。这个类的声明语法如下：

```py
wait_condition = QtCore.QWaitCondition()
```

`QWaitCondition`类通过以下函数增强了功能。

# functional

这些函数与信号量的当前值返回、功能变化等相关：

`wait_condition.wait(QtCore.QMutex, int)`: 这释放被锁定的互斥锁（第一个参数）并等待等待条件，例如其他线程的信号，包括`wakeOne()`或`wakeAll()`，或者毫秒数（第二个参数）超时。

`wait_condition.wait(QtCore.QReadWriteLock, int)`: 这释放被锁定的读写锁（第一个参数）并等待等待条件，例如其他线程的信号，包括`wakeOne()`或`wakeAll()`，或者毫秒数（第二个参数）超时。

`wait_condition.wakeAll()`: 这唤醒所有正在等待等待条件的线程。

`wait_condition.wakeOne()`: 这唤醒一个正在等待等待条件的线程。

# Python 标准库工具

Python 标准库的工具可以通过 Qt 库的 PySide2/PyQt5 Python 绑定轻松实现，在我们的 GUI 应用程序中。我们可以用类似我们在本章前面描述的方式描述我们将要使用的工具。

# threading

此 Python 标准库模块使用 PyQt 和 PySide 绑定实现了任何基于 Python 的应用程序的线程功能。此模块可以在比类似 `QThread` 类更广泛的意义上使用，用于应用程序中的专用任务。但我们需要知道，在 PyQt/PySide GUI 中，`QThread` 类允许通过信号进行通信和功能。要在应用程序中使用线程，我们需要导入此模块：

```py
...
import threading
...
```

我们将只描述此模块的常用部分。有关此模块的完整信息可在官方文档中找到：[`docs.python.org/3/`](https://docs.python.org/3/)。此模块使用以下有用的函数：

`threading.active_count()`: 这返回当前正在运行的线程数。

`threading.current_thread()`: 这返回当前正在使用的线程。

`threading.get_ident()`: 这返回当前线程的标识符。

`threading.enumerate()`: 这返回当前所有正在运行的线程的列表。

`threading.main_thread()`: 这返回主线程。在 Python 语境中，这是启动操作 Python 环境的 Python 解释器的线程。

`threading.stack_size(int)`: 这返回创建新线程时将使用的堆栈大小。如果指定了可选参数，则将其用作堆栈大小。请注意，该参数是字节数，并且可以使用 `0` 或至少 32 KiB (`32768`)；例如，512 KiB 以数字形式表示为 `524288`。

`threading` 模块通过几个类实现此功能（提供的参数与 Python 3.x 相关）。

# Thread

`threading` 模块的 `thread` 类如下：

```py
thread1 = threading.Thread(group=None, target=None, name=None,
                             args=(), kwargs={}, *, daemon=None)
```

`group` 保留用于将来扩展，当实现 `ThreadGroup` 类时；`target` 是一个可调用对象；`name` 是线程的名称；`args` 是用于的参数元组；`kwargs` 是关键字参数的字典；`daemon` 将线程设置为守护线程。`Thread` 类具有以下功能：

+   `thread1.start()`: 这将启动线程。

+   `thread1.run()`: 这代表线程活动。

+   `thread1.join(float)`: 这等待直到线程终止。超时参数可选地等待在阻塞操作之前或终止后立即阻塞。

+   `thread1.is_alive()`: 如果此线程正在运行，则返回 `True`，否则返回 `False`。

构建此类通常如下所示：

```py
def func(n):
    ...

thread1 = threading.Thread(target=func, args=(14,))
thread1.start()
thread1.join()
```

# Lock

这些是原始锁对象：

```py
thread1_lock = threading.Lock
```

`Lock` 类具有以下功能：

`thread1_lock.acquire(blocking=True, timeout=-1)`: 这将获取阻塞或非阻塞。

`thread1_lock.release()`: 这将释放锁。

# RLock

这些是可重入锁对象：

```py
thread1_rlock = threading.RLock
```

`RLock` 类具有以下功能：

`thread1_rlock.acquire(blocking=True, timeout=-1)`: 这将获取阻塞或非阻塞。

`thread1_rlock.release()`: 这将释放锁。

# 条件

这些是条件变量对象：

```py
thread1_cond = threading.Condition(lock=None)
```

`Condition`类有以下功能：

`thread1_cond.acquire(*args)`: 这获取底层的锁。

`thread1_cond.release()`: 这释放了底层的锁。

`thread1_cond.wait(timeout=None)`: 这将等待直到被通知或发生超时。

`thread1_cond.wait_for(predicate, timeout=None)`: 这将等待直到条件评估为`True`。条件参数是一个可调用对象，它返回一个布尔值（`True`或`False`）。

`thread1_cond.notify(n=1)`: 这将唤醒`n`个线程。

`thread1_cond.notify_all()`: 这唤醒所有线程。

# 信号量

这些是信号量对象，它们管理一个计数器，该计数器计算释放次数减去获取次数，再加上初始值：

```py
thread1_sema = threading.Semaphore(value=1)
```

`Semaphore`类有以下功能：

`thread1_sema.acquire(*args)`: 这获取信号量。

`thread1_sema.release()`: 这释放信号量。

# 有界信号量

这些是有界信号量对象，用于检查当前值是否不超过其初始值：

```py
thread1_bsema = threading.BoundedSemaphore(value=1)
```

# 事件

这些是用于线程间通信的事件对象。这是通过管理内部标志来完成的：

```py
thread1_event = threading.Event
```

`Event`类有以下功能：

`thread1_event.is_set()`: 如果内部标志为`True`，则返回`True`。

`thread1_event.set()`: 这将内部标志设置为`True`。

`thread1_event.clear()`: 这将内部标志设置为`False`。

`thread1_event.wait(timeout=None)`: 这将阻塞直到标志为`True`。

# 定时器

这些是运行动作计时的定时器对象。这将在指定的时间段之后运行。提供的参数是动作将运行的`interval`；要运行的`function`；作为将使用的参数的`args`；以及作为关键字参数的`kwargs`：

```py
thread1_timer = threading.Timer(interval, function,
                                args=None, kwargs=None)
```

`Timer`类有以下功能：

`thread1_timer.cancel()`: 这通过取消其执行来停止定时器。

# 屏障

这些是使用固定数量的线程实现的屏障对象，这些线程需要相互等待。线程通过调用`wait()`方法来尝试通过屏障。提供的参数是`parties`，即线程的数量；`action`，即将被一个线程调用的可调用对象；以及`timeout`，它是用于`wait()`方法的值：

```py
thread1_barrier = threading.Barrier(parties, action=None,
                                             timeout=None)
```

`Barrier`类有以下功能：

`thread1_barrier.wait(timeout=None)`: 这通过屏障。

`thread1_barrier.reset()`: 这重置屏障并设置空状态。

`thread1_barrier.abort()`: 这中止屏障并设置损坏状态。

`thread1_barrier.parties`: 这返回通过屏障所需的线程数量。

`thread1_barrier.n_waiting`: 这返回在屏障中等待的线程数量。

`thread1_barrier.broken`: 如果屏障处于损坏状态，则返回`True`。

# 队列

在多个线程等待运行一个任务的情况下，使用队列来运行这个任务非常重要。标准库`queue`模块可以在我们的应用程序中使用多生产者和多消费者队列功能。Python 对 Qt 的绑定有相对复杂的队列实现工具，并且这个模块在 GUI 的线程结构中得到了广泛的应用。此模块实现了三种类型的队列：**先进先出**（**FIFO**）、**后进先出**（**LIFO**）和带优先级的队列。要在我们的应用程序中使用它们，我们需要导入以下模块：

```py
...
import queue
..
```

让我们描述一下可以使用的最重要的类和方法。

# Queue

这些是具有`maxsize`上限的 FIFO 队列，可以放置在队列中的项目数量：

```py
queue1 = queue.Queue(maxsize=0)
```

# LifoQueue

这些是具有`maxsize`上限的 LIFO 队列，可以放置在队列中的项目数量：

```py
queue1 = queue.LifoQueue(maxsize=0)
```

# PriorityQueue

这些是具有`maxsize`上限的优先队列，可以放置在队列中的项目数量：

```py
queue1 = queue.PriorityQueue(maxsize=0)
```

# SimpleQueue

这些是无界 FIFO 队列：

```py
queue1 = queue.SimpleQueue
```

# functions

可以使用的`Queue`, `LifoQueue`, `PriorityQueue`, 和 `SimpleQueue`类的方法定义如下：

`queue1.qsize()`: 返回队列的大约大小，因为队列通常会变化（`Queue`, `LifoQueue`, `PriorityQueue`, `SimpleQueue`）。

`queue1.empty()`: 如果队列大约为空，则返回`True`（`Queue`, `LifoQueue`, `PriorityQueue`, `SimpleQueue`）。

`queue1.full()`: 如果队列大约已满（至少有一个项目），则返回`True`（`Queue`, `LifoQueue`, `PriorityQueue`）。

`queue1.put(item, block=True, timeout=None)`: 将指定的项目放入队列。提供了可选的 block 和 timeout 参数（`Queue`, `LifoQueue`, `SimpleQueue`）。

`queue1.put((priority, item), block=True, timeout=None)`: 将指定的项目以整数优先级值放入队列。提供了可选的 block 和 timeout 参数（`PriorityQueue`）。

`queue1.put_nowait(item)`: 将指定的项目放入队列。提供了可选的 block 和 timeout 参数（`Queue`, `LifoQueue`, `SimpleQueue`）。

`queue1.put_nowait((priority, item))`: 将指定的项目以整数优先级值放入队列。提供了可选的 block 和 timeout 参数（`PriorityQueue`）。

`queue1.get(block=True, timeout=None)`: 从队列中返回一个项目。提供了可选的 block 和 timeout 参数（`Queue`, `LifoQueue`, `PriorityQueue`, `SimpleQueue`）。

`queue1.get_nowait()`: 这从队列中返回一个项目（`Queue`, `LifoQueue`, `PriorityQueue`, `SimpleQueue`）。

`queue1.task_done()`: 这表示队列的任务已完成（`Queue`, `LifoQueue`, `PriorityQueue`）。

`queue1.join()`: 这将阻塞，直到所有项目都已被处理并完成（`Queue`, `LifoQueue`, `PriorityQueue`）。

# subprocess

这个 Python 标准库模块以类似于 Qt 库中的`QProcess`类的方式实现了运行进程的功能。此模块以新进程的形式运行任务，连接到进程的输入/输出/错误管道，并获取它们的返回码。之前，我们实现了运行相机设备功能等子进程。让我们更详细地描述这个模块。要在我们的应用程序中使用它，我们需要导入以下模块：

```py
...
import subprocess
...
```

在以下部分中描述了在应用程序中可以使用的最重要的类和方法。

# run()

从 Python 3.5 版本开始可用的`run()`方法，在新的进程中运行任务。其语法如下：

```py
subprocess1 = subprocess.run(["command", "-flags", "args"],
                    *, stdin=None, input=None, stdout=None,
                         stderr=None, capture_output=False,
                       shell=False, cwd=None, timeout=None,
                   check=False, encoding=None, errors=None,
                   text=None, env=None, universal_newlines=None)
```

第一个参数是一个包含命令、参数以及如果有的话标志的列表。`stdin`、`stdout`和`stderr`参数指定了执行程序的`STDIN`（标准输入）、`STDOUT`（标准输出）和`STDERR`（标准错误）。可以使用`PIPE`、`DEVNULL`和`STDOUT`等值。`input`用于`communicate()`方法。如果`capture_output`为`True`，则`stdout`和`stderr`将被捕获。如果`shell`为`True`，则指定通过 shell 执行的命令。如果`cwd`不是`None`，则在执行前会更改`cwd`（当前工作目录）。`timeout`用于`communicate()`方法。当超时到期时，子进程将被杀死并等待。如果`check`为`True`，并且进程以非零退出码退出，则将引发`CalledProcessError`异常。`encoding`指定将使用哪种编码，例如`"utf-8"`或`"cp1252"`。`errors`指定如何使用字符串值（如`"strict"`、`"ignore"`、`"replace"`、`"backslashreplace"`、`"xmlcharrefreplace"`和`"namereplace"`）处理编码和解码错误。如果`text`为`True`，则`stdin`、`stdout`和`stderr`的文件对象将以文本模式打开。`env`定义了新进程的环境变量。`universal_newlines`与`text`等效，并提供向后兼容性。

# Popen

`Popen`类在`subprocess`模块中处理底层进程创建和管理。这个类提供了可以使用的附加可选参数。这个类的语法看起来像这样：

```py
subprocess1 = subprocess.Popen(["command", "-flags", "args"],
                     bufsize=-1, executable=None, stdin=None,
                   stdout=None, stderr=None, preexec_fn=None,
                       close_fds=True, shell=False, cwd=None,
                           env=None, universal_newlines=None,
                           startupinfo=None, creationflags=0,
                                        restore_signals=True,
                     start_new_session=False, pass_fds=(), *,
                       encoding=None, errors=None, text=None)
```

第一个参数是一个包含命令、参数以及（如果提供）标志的列表。`bufsize`将在创建`stdin`/`stdout`/`stderr`管道文件对象时与`open()`函数一起使用。`executable`指定要执行的替换程序。`stdin`、`stdout`和`stderr`参数指定要执行的程序的`STDIN`（标准输入）、`STDOUT`（标准输出）和`STDERR`（标准错误）。有效值包括`PIPE`、`DEVNULL`和`STDOUT`。如果设置了`preexec_fn`，则对象将在子进程执行之前在子进程中调用（仅限 POSIX）。如果`close_fds`为`True`，则在执行之前将关闭所有文件描述符，除了`0`、`1`和`2`。如果`shell`为`True`，则将通过 shell 执行命令。如果`cwd`不是`None`，则在执行之前将更改`cwd`（当前工作目录）。`env`定义了新进程的环境变量。`universal_newlines`等同于文本，并提供向后兼容性。`startupinfo`将是一个`STARTUPINFO`对象，它将带有创建标志传递给`CreateProcess`函数。如果`restore_signals`为`True`，则在执行之前将所有设置为`SIG_IGN`的信号恢复为`SIG_DFL`（仅限 POSIX）。如果`start_new_session`为`True`，则在子进程执行子进程之前将执行`setsid()`系统调用（仅限 POSIX）。`pass_fds`是父进程和子进程之间保持打开的文件描述符序列（仅限 POSIX）。`encoding`指定将使用哪种编码，例如`"utf-8"`或`"cp1252"`。`errors`指定如何处理编码和解码错误，字符串值包括`"strict"`、**`"ignore"`**、`"replace"`、`"backslashreplace"`、`"xmlcharrefreplace"`和`"namereplace"`。如果`text`为`True`，则将`stdin`、`stdout`和`stderr`的文件对象以文本模式打开。

# 函数

可以与该模块一起使用的以下方法：

`subprocess1.poll()`: 这检查子进程是否已终止。

`subprocess1.wait(timeout=None)`: 如果指定了超时，则等待子进程终止。

`subprocess1.communicate(input=None, timeout=None)`: 通过向子进程发送数据并从`STDOUT`/`STDERR`读取数据，同时等待终止来与进程交互。输入可以是发送到子进程的数据。

`subprocess1.send_signal(signal)`: 这向子进程发送信号。它有以下功能：

+   `subprocess1.terminate()`: 使用操作系统参数终止子进程。

+   `subprocess1.kill()`: 使用操作系统参数终止子进程。

+   `subprocess1.args`: 这返回传递给`Popen`的参数。

+   `subprocess1.stdin`: 这返回传递给`Popen`标准输入的参数。

+   `subprocess1.stdout`: 这返回传递给`Popen`标准输出的参数。

+   `subprocess1.stderr`: 这返回传递给`Popen`标准错误的参数。

+   `subprocess1.pid`: 这将返回子进程的进程 ID。

+   `subprocess1.returncode`: 这将返回子进程的返回代码。

# multiprocessing

这个 Python 标准库包管理了在应用程序中可以使用的进程创建。当使用本地和远程并发时，这个工具通过使用子进程而不是线程来绕过 **全局解释器锁** (**GIL**) 的限制，如果设备有多个处理器，则可以有效地利用多个处理器。multiprocessing 包的 API 与 `threading` 模块类似，方法大多复制了 `threading` 模块的 `Thread` 类，类似于前面展示的 `queue` 模块的 `Queue`。为了在我们的应用程序中使用它，我们需要导入以下包：

```py
...
import multiprocessing
..
```

# Process

这个类表示了能够实现多进程和单独进程活动的进程对象。语法如下：

```py
process1 = multiprocessing.Process(group=None, target=None,
                                        name=None, args=(),
                                 kwargs={}, *, daemon=None)
```

参数的含义与 `threading` 模块的 `Thread` 类中的参数类似。在这个类中使用的如下方法：

`process1.run()`: 这代表进程的活动。

`process1.start()`: 这将启动进程。

`process1.join(float)`: 这将等待直到进程终止。超时参数（可选）在阻塞操作之前等待，或者在终止后立即阻塞。

`process1.is_alive()`: 如果此进程正在运行，则返回 `True`，否则返回 `False`。

`process1.close()`: 这将关闭进程并释放所有相关资源。

`process1.kill()`: 这将杀死进程。

`process1.terminate()`: 这将终止进程。

`process1.name`: 这将返回进程的名称。

`process1.daemon`: 这将返回进程的守护进程标志，即 `True` 或 `False`。

`process1.pid`: 这将返回进程的 ID。

`process1.exitcode`: 这将返回子进程的退出代码。

`process1.authkey`: 这将返回进程的认证密钥。

`process1.sentinel`: 这将返回系统对象数字句柄，当进程结束时将准备好。

# Connection

这个类创建了连接对象，允许我们发送和接收可序列化的对象或字符串。语法如下：

```py
conn = multiprocessing.connection.Connection
```

可以使用的方法如下：

`conn.send(obj)`: 这将可序列化的对象发送到连接的另一端。

`conn.recv()`: 这将返回从连接另一端接收到的对象。

`conn.fileno()`: 这将返回连接的文件描述符或句柄。

`conn.close()`: 这将关闭连接。

`conn.poll(timeout)`: 如果有可读数据，则返回 `True`。

`conn.send_bytes(buffer, offset, size)`: 这将从字节对象发送数据。如果指定了 `offset`，则将从 `buffer` 中的位置读取数据，如果指定了 `size`，则以字节数为单位。

`conn.recv_bytes(maxlength)`: 这返回从连接的另一端接收到的字节数据的消息。如果指定了`maxlength`，则限制消息。

`conn.recv_bytes_into(buffer, offset)`: 这读取从连接的另一端接收到的字节数据的消息，并返回消息中的字节数。如果指定了`offset`，则消息将从该位置写入`buffer`。

连接通常使用`Pipe`类创建。构建此类连接的语法如下：

```py
conn1, conn2 = multiprocessing.Pipe(duplex)
```

如果`duplex`参数为`True`，则管道是双向的；如果为`False`，则管道是单向的。这意味着`conn1`用于接收消息，而`conn2`用于发送消息。以下是一个连接的示例：

```py
def func1(msg, conn):
    conn.send(str(msg))
    conn.close()

def func2(msg, conn):
    conn.send(msg)
    conn.close()

if __name__ == "__main__":
    conn1, conn2 = multiprocessing.Pipe()
    process1 = multiprocessing.Process(
           target=func1, args=(
                  "Hello Process # 1", conn2))
    process2 = multiprocessing.Process(
           target=func2, args=(
                  "Hello Process # 2", conn1))
    process1.start()
    process2.start()
    print(conn1.recv())
    print(conn2.recv())
    process1.join()
    process2.join()
```

在这里，我们创建了具有`Connection`类的发送方法的函数；使用了`Pipe`类进行连接，并且也使用了之前描述的方法。这种构建不是随机的。在多进程方面，我们需要在**`if __name__ == "__main__"`**指令内操作进程，或者调用具有多进程功能的功能。使用`multiprocessing`包，类如`Pool`（创建进程池）、`Queue`（创建队列）、`Manager`（控制管理共享对象的服务器进程）以及连接模块的`Listener`和`Client`类也是可用的。建议您了解这些类和模块。此外，多进程包还具有以下方法，可能很有用：

+   `multiprocessing.active_children()`: 这返回当前进程的所有活动子进程。

+   `multiprocessing.cpu_count()`: 这返回设备中使用的 CPU 数量。

+   `multiprocessing.current_process()`: 这返回当前进程。

# 摘要

本章完成了本书材料的基本部分。线程的使用提高了应用程序的生产力。仪器的重性和大小是一个如此庞大的主题，以至于我们需要一本书来涵盖所有内容。这就是为什么我们只看了起点。关于这些有趣且有用的工具的更多信息可以在 PySide2 的官方文档（[`doc.qt.io/qtforpython/index.html`](https://doc.qt.io/qtforpython/index.html)）、PyQt5（[`www.riverbankcomputing.com/static/Docs/PyQt5/`](https://www.riverbankcomputing.com/static/Docs/PyQt5/））和当然还有 Python（[https://docs.python.org/3/](https://docs.python.org/3/)）的官方文档中找到。多线程和多进程的构建以及在我们应用程序中的实现具有核心地位，因为它们使得 GUI 应用程序舒适且用户友好。本章涵盖了 GUI 开发中所需的所有必要内容。

下一章将完成这本书。我们将最终确定我们的图形用户界面应用程序，提供将应用程序嵌入不同平台的方法，并在解释基础知识的同时尝试一些代码。
