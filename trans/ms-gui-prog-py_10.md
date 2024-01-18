# 使用QtNetwork进行网络连接

人类是社会性动物，越来越多的软件系统也是如此。尽管计算机本身很有用，但与其他计算机连接后，它们的用途要大得多。无论是在小型本地交换机还是全球互联网上，通过网络与其他系统进行交互对于大多数现代软件来说都是至关重要的功能。在本章中，我们将探讨Qt提供的网络功能以及如何在PyQt5中使用它们。

特别是，我们将涵盖以下主题：

+   使用套接字进行低级网络连接

+   使用`QNetworkAccessManager`进行HTTP通信

# 技术要求

与其他章节一样，您需要一个基本的Python和PyQt5设置，如[第1章](bce5f3b1-2979-4f78-817b-3986e7974725.xhtml)中所述，并且您将受益于从我们的GitHub存储库下载示例代码[https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter08](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter08)。

此外，您将希望至少有另一台装有Python的计算机连接到同一局域网。

查看以下视频以查看代码的运行情况：[http://bit.ly/2M5xqid](http://bit.ly/2M5xqid)

# 使用套接字进行低级网络连接

几乎每个现代网络都使用**互联网协议套件**，也称为**TCP/IP**，来促进计算机或其他设备之间的连接。TCP/IP是一组管理网络上原始数据传输的协议。直接在代码中使用TCP/IP最常见的方法是使用**套接字API**。

套接字是一个类似文件的对象，代表系统的网络连接点。每个套接字都有一个**主机地址**，**网络端口**和**传输协议**。

主机地址，也称为**IP地址**，是用于在网络上标识单个网络主机的一组数字。尽管骨干系统依赖IPv6协议，但大多数个人计算机仍使用较旧的IPv4地址，该地址由点分隔的四个介于`0`和`255`之间的数字组成。您可以使用GUI工具找到系统的地址，或者通过在命令行终端中键入以下命令之一来找到地址：

| OS | Command |
| --- | --- |
| Windows | `ipconfig` |  |
| macOS | `ifconfig` |
| Linux | `ip address` |  |

端口只是一个从`0`到`65535`的数字。虽然您可以使用任何端口号创建套接字，但某些端口号分配给常见服务；这些被称为**众所周知的端口**。例如，HTTP服务器通常分配到端口`80`，SSH通常在端口`22`上。在许多操作系统上，需要管理或根权限才能在小于`1024`的端口上创建套接字。

可以在[https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml](https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml)找到官方的众所周知的端口列表。

传输协议包括**传输控制协议**（**TCP**）和**用户数据报协议**（**UDP**）。TCP是两个系统之间的有状态连接。您可以将其视为电话呼叫 - 建立连接，交换信息，并在某个明确的点断开连接。由于其有状态性，TCP确保接收所有传输的数据包。另一方面，UDP是一种无状态协议。将其视为使用对讲机 - 用户传输消息，接收者可能完整或部分接收，且不会建立明确的连接。UDP相对轻量级，通常用于广播消息，因为它不需要与特定主机建立连接。

`QtNetwork`模块为我们提供了建立TCP和UDP套接字连接的类。为了理解它们的工作原理，我们将构建两个聊天系统 - 一个使用UDP，另一个使用TCP。

# 构建聊天 GUI

让我们首先创建一个基本的 GUI 表单，我们可以在聊天应用的两个版本中使用。从[第 4 章](9281bd2a-64a1-4128-92b0-e4871b79c040.xhtml)的应用程序模板开始，*使用 QMainWindow 构建应用程序*，然后添加这个类：

```py
class ChatWindow(qtw.QWidget):

    submitted = qtc.pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.setLayout(qtw.QGridLayout())
        self.message_view = qtw.QTextEdit(readOnly=True)
        self.layout().addWidget(self.message_view, 1, 1, 1, 2)
        self.message_entry = qtw.QLineEdit()
        self.layout().addWidget(self.message_entry, 2, 1)
        self.send_btn = qtw.QPushButton('Send', clicked=self.send)
        self.layout().addWidget(self.send_btn, 2, 2)
```

GUI 很简单，只有一个文本编辑器来显示对话，一个行编辑器来输入消息，以及一个发送按钮。我们还实现了一个信号，每当用户提交新消息时就可以发出。

GUI 还将有两个方法：

```py
    def write_message(self, username, message):
        self.message_view.append(f'<b>{username}: </b> {message}<br>')

    def send(self):
        message = self.message_entry.text().strip()
        if message:
            self.submitted.emit(message)
            self.message_entry.clear()
```

`send()` 方法由 `send_btn` 按钮触发，发出包含行编辑中文本的 `submitted` 信号，以及 `write_message()` 方法，该方法接收 `username` 和 `message` 并使用一些简单的格式将其写入文本编辑器。

在 `MainWindow.__init__()` 方法中，添加以下代码：

```py
        self.cw = ChatWindow()
        self.setCentralWidget(self.cw)
```

最后，在我们可以进行任何网络编码之前，我们需要为 `QtNetwork` 添加一个 `import`。像这样将其添加到文件的顶部：

```py
from PyQt5 import QtNetwork as qtn
```

这段代码将是我们的 UDP 和 TCP 聊天应用程序的基础代码，所以将这个文件保存为 `udp_chat.py` 的一个副本，另一个副本保存为 `tcp_chat.py`。我们将通过为表单创建一个后端对象来完成每个应用程序。

# 构建 UDP 聊天客户端

UDP 最常用于本地网络上的广播应用程序，因此为了演示这一点，我们将使我们的 UDP 聊天成为一个仅限本地网络的广播聊天。这意味着在运行此应用程序副本的本地网络上的任何计算机都将能够查看并参与对话。

我们将首先创建我们的后端类，我们将其称为 `UdpChatInterface`：

```py
class UdpChatInterface(qtc.QObject):

    port = 7777
    delimiter = '||'
    received = qtc.pyqtSignal(str, str)
    error = qtc.pyqtSignal(str)
```

我们的后端继承自 `QObject`，以便我们可以使用 Qt 信号，我们定义了两个信号——一个 `received` 信号，当接收到消息时我们将发出它，一个 `error` 信号，当发生错误时我们将发出它。我们还定义了一个要使用的端口号和一个 `delimiter` 字符串。当我们序列化消息进行传输时，`delimiter` 字符串将用于分隔用户名和消息；因此，当用户 `alanm` 发送消息 `Hello World` 时，我们的接口将在网络上发送字符串 `alanm||Hello World`。

一次只能将一个应用程序绑定到一个端口；如果您已经有一个使用端口 `7777` 的应用程序，您应该将这个数字更改为 `1024` 到 `65535` 之间的其他数字。在 Windows、macOS 和旧版 Linux 系统上，可以使用 `netstat` 命令来显示正在使用哪些端口。在较新的 Linux 系统上，可以使用 `ss` 命令。

现在开始一个 `__init__()` 方法：

```py
    def __init__(self, username):
        super().__init__()
        self.username = username

        self.socket = qtn.QUdpSocket()
        self.socket.bind(qtn.QHostAddress.Any, self.port)
```

调用 `super()` 并存储 `username` 变量后，我们的首要任务是创建和配置一个 `QUdpSocket` 对象。在我们可以使用套接字之前，它必须**绑定**到本地主机地址和端口号。`QtNetwork.QHostAddress.Any` 表示本地系统上的所有地址，因此我们的套接字将在所有本地接口上监听和发送端口 `7777` 上的数据。

要使用套接字，我们必须处理它的信号：

```py
        self.socket.readyRead.connect(self.process_datagrams)
        self.socket.error.connect(self.on_error)
```

Socket 对象有两个我们感兴趣的信号。第一个是 `readyRead`，每当套接字接收到数据时就会发出该信号。我们将在一个名为 `process_datagrams()` 的方法中处理该信号，我们马上就会写这个方法。

`error` 信号在发生任何错误时发出，我们将在一个名为 `on_error()` 的实例方法中处理它。

让我们从错误处理程序开始，因为它相对简单：

```py
    def on_error(self, socket_error):
        error_index = (qtn.QAbstractSocket
                       .staticMetaObject
                       .indexOfEnumerator('SocketError'))
        error = (qtn.QAbstractSocket
                 .staticMetaObject
                 .enumerator(error_index)
                 .valueToKey(socket_error))
        message = f"There was a network error: {error}"
        self.error.emit(message)
```

这种方法在其中有一点Qt的魔力。网络错误在`QAbstractSocket`类（`UdpSocket`的父类）的`SocketError`枚举中定义。不幸的是，如果我们只是尝试打印错误，我们会得到常量的整数值。要实际获得有意义的字符串，我们将深入与`QAbstractSocket`关联的`staticMetaObject`。我们首先获取包含错误常量的枚举类的索引，然后使用`valueToKey()`将我们的套接字错误整数转换为其常量名称。这个技巧可以用于任何Qt枚举，以检索有意义的名称而不仅仅是它的整数值。

一旦被检索，我们只需将错误格式化为消息并在我们的`error`信号中发出。

现在让我们来解决`process_datagrams()`：

```py
    def process_datagrams(self):
        while self.socket.hasPendingDatagrams():
            datagram = self.socket.receiveDatagram()
            raw_message = bytes(datagram.data()).decode('utf-8')
```

单个UDP传输被称为**数据报**。当我们的套接字接收到数据报时，它被存储在缓冲区中，并发出`readyRead`信号。只要该缓冲区有等待的数据报，套接字的`hasPendingDatagrams()`将返回`True`。因此，只要有待处理的数据报，我们就会循环调用套接字的`receiveDatagram()`方法，该方法返回并移除缓冲区中等待的下一个数据报，直到检索到所有数据报为止。

`receiveDatagram()`返回的数据报对象是`QByteArray`，相当于Python的`bytes`对象。由于我们的程序传输的是字符串，而不是二进制对象，我们可以将`QByteArray`直接转换为Unicode字符串。这样做的最快方法是首先将其转换为`bytes`对象，然后使用`decode()`方法将其转换为UTF-8 Unicode文本。

现在我们有了原始字符串，我们需要检查它以确保它来自`udp_chat.py`的另一个实例，然后将其拆分成`username`和`message`组件：

```py
            if self.delimiter not in raw_message:
                continue
            username, message = raw_message.split(self.delimiter, 1)
            self.received.emit(username, message)
```

如果套接字接收到的原始文本不包含我们的`delimiter`字符串，那么它很可能来自其他程序或损坏的数据包，我们将跳过它。否则，我们将在第一个`delimiter`的实例处将其拆分为`username`和`message`字符串，然后发出这些字符串与`received`信号。

我们的聊天客户端需要的最后一件事是发送消息的方法，我们将在`send_message()`方法中实现：

```py
   def send_message(self, message):
        msg_bytes = (
            f'{self.username}{self.delimiter}{message}'
        ).encode('utf-8')
        self.socket.writeDatagram(
            qtc.QByteArray(msg_bytes),
            qtn.QHostAddress.Broadcast,
            self.port
        )
```

这种方法首先通过使用`delimiter`字符串格式化传递的消息与我们配置的用户名，然后将格式化的字符串编码为`bytes`对象。

接下来，我们使用`writeDatagram()`方法将数据报写入我们的套接字对象。这个方法接受一个`QByteArray`（我们已经将我们的`bytes`对象转换为它）和一个目标地址和端口。我们的目的地被指定为`QHostAddress.Broadcast`，这表示我们要使用广播地址，端口当然是我们在类变量中定义的端口。

**广播地址**是TCP/IP网络上的保留地址，当使用时，表示传输应该被所有主机接收。

让我们总结一下我们在这个后端中所做的事情：

+   发送消息时，消息将以用户名为前缀，并作为字节数组广播到网络上的所有主机的端口`7777`。

+   当在端口`7777`上接收到消息时，它将从字节数组转换为字符串。消息和用户名被拆分并发出信号。

+   发生错误时，错误号将被转换为错误字符串，并与错误信号一起发出。

现在我们只需要将我们的后端连接到前端表单。

# 连接信号

回到我们的`MainWindow`构造函数，我们需要通过创建一个`UdpChatInterface`对象并连接其信号来完成我们的应用程序：

```py
        username = qtc.QDir.home().dirName()
        self.interface = UdpChatInterface(username)
        self.cw.submitted.connect(self.interface.send_message)
        self.interface.received.connect(self.cw.write_message)
        self.interface.error.connect(
            lambda x: qtw.QMessageBox.critical(None, 'Error', x))
```

在创建界面之前，我们通过获取当前用户的主目录名称来确定`username`。这有点像黑客，但对我们的目的来说足够好了。

接下来，我们创建我们的接口对象，并将聊天窗口的`submitted`信号连接到其`send_message()`槽。

然后，我们将接口的`received`信号连接到聊天窗口的`write_message()`方法，将`error`信号连接到一个lambda函数，用于在`QMessageBox`中显示错误。

一切都连接好了，我们准备好测试了。

# 测试聊天

要测试这个聊天系统，您需要两台安装了Python和PyQt5的计算机，运行在同一个局域网上。在继续之前，您可能需要禁用系统的防火墙或打开UDP端口`7777`。

完成后，将`udp_chat.py`复制到两台计算机上并启动它。在一台计算机上输入一条消息；它应该会显示在两台计算机的聊天窗口中，看起来像这样：

![](assets/152c38db-31b6-4c6a-b19e-beea931a4787.png)

请注意，系统也会接收并对自己的广播消息做出反应，因此我们不需要担心在文本区域中回显自己的消息。

UDP确实很容易使用，但它有许多限制。例如，UDP广播通常无法路由到本地网络之外，而且无状态连接的缺失意味着无法知道传输是否已接收或丢失。在*构建TCP聊天客户端*部分，我们将构建一个没有这些问题的聊天TCP版本。

# 构建TCP聊天客户端

TCP是一种有状态的传输协议，这意味着建立并维护连接直到传输完成。TCP也主要是一对一的主机连接，我们通常使用**客户端-服务器**设计来实现。我们的TCP聊天应用程序将在两个网络主机之间建立直接连接，并包含一个客户端组件，用于连接应用程序的其他实例，以及一个服务器组件，用于处理传入的客户端连接。

在您之前创建的`tcp_chat.py`文件中，像这样启动一个TCP聊天接口类：

```py
class TcpChatInterface(qtc.QObject):

    port = 7777
    delimiter = '||'
    received = qtc.pyqtSignal(str, str)
    error = qtc.pyqtSignal(str)
```

到目前为止，这与UDP接口完全相同，除了名称。现在让我们创建构造函数：

```py
    def __init__(self, username, recipient):
        super().__init__()
        self.username = username
        self.recipient = recipient
```

与以前一样，接口对象需要一个`username`，但我们还添加了一个`recipient`参数。由于TCP需要与另一个主机建立直接连接，我们需要指定要连接的远程主机。

现在我们需要创建服务器组件，用于监听传入的连接：

```py
        self.listener = qtn.QTcpServer()
        self.listener.listen(qtn.QHostAddress.Any, self.port)
        self.listener.acceptError.connect(self.on_error)

        self.listener.newConnection.connect(self.on_connection)
        self.connections = []
```

`listener`是一个`QTcpServer`对象。`QTcpServer`使我们的接口能够在给定接口和端口上接收来自TCP客户端的传入连接，这里我们将其设置为端口`7777`上的任何本地接口。

当有传入连接出现错误时，服务器对象会发出一个`acceptError`信号，我们将其连接到一个`on_error()`方法。这些是`UdpSocket`发出的相同类型的错误，因此我们可以从`udp_chat.py`中复制`on_error()`方法并以相同的方式处理它们。

每当有新连接进入服务器时，都会发出`newConnection`信号；我们将在一个名为`on_connection()`的方法中处理这个信号，它看起来像这样：

```py
    def on_connection(self):
        connection = self.listener.nextPendingConnection()
        connection.readyRead.connect(self.process_datastream)
        self.connections.append(connection)
```

服务器的`nextPendingConnection()`方法返回一个`QTcpSocket`对象作为下一个等待连接。像`QUdpSocket`一样，`QTcpSocket`在接收数据时会发出`readyRead`信号。我们将把这个信号连接到一个`process_datastream()`方法。

最后，我们将在`self.connections`列表中保存对新连接的引用。

# 处理数据流

虽然UDP套接字使用数据报，但TCP套接字使用**数据流**。顾名思义，数据流涉及数据的流动而不是离散的单元。TCP传输被发送为一系列网络数据包，这些数据包可能按照正确的顺序到达，也可能不会，接收方需要正确地重新组装接收到的数据。为了使这个过程更容易，我们可以将套接字包装在一个`QtCore.QDataStream`对象中，它提供了一个从类似文件的源读取和写入数据的通用接口。

让我们像这样开始我们的方法：

```py
    def process_datastream(self):
        for socket in self.connections:
            self.datastream = qtc.QDataStream(socket)
            if not socket.bytesAvailable():
                continue
```

我们正在遍历连接的套接字，并将每个传递给`QDataStream`对象。`socket`对象有一个`bytesAvailable()`方法，告诉我们有多少字节的数据排队等待读取。如果这个数字为零，我们将继续到列表中的下一个连接。

如果没有，我们将从数据流中读取：

```py
            raw_message = self.datastream.readQString()
            if raw_message and self.delimiter in raw_message:
                username, message = raw_message.split(self.delimiter, 1)
                self.received.emit(username, message)
```

`QDataStream.readQString()`尝试从数据流中提取一个字符串并返回它。尽管名称如此，在PyQt5中，这个方法实际上返回一个Python Unicode字符串，而不是`QString`。重要的是要理解，这个方法*只有*在原始数据包中发送了`QString`时才起作用。如果发送了其他对象（原始字节字符串、整数等），`readQString()`将返回`None`。

`QDataStream`有用于写入和读取各种数据类型的方法。请参阅其文档[https://doc.qt.io/qt-5/qdatastream.html](https://doc.qt.io/qt-5/qdatastream.html)。

一旦我们将传输作为字符串，我们将检查原始消息中的`delimiter`字符串，并且如果找到，拆分原始消息并发出`received`信号。

# 通过TCP发送数据

`QTcpServer`已经处理了消息的接收；现在我们需要实现发送消息。为此，我们首先需要创建一个`QTcpSocket`对象作为我们的客户端套接字。

让我们将其添加到`__init__()`的末尾：

```py
        self.client_socket = qtn.QTcpSocket()
        self.client_socket.error.connect(self.on_error)
```

我们创建了一个默认的`QTcpSocket`对象，并将其`error`信号连接到我们的错误处理方法。请注意，我们不需要绑定此套接字，因为它不会监听。

为了使用客户端套接字，我们将创建一个`send_message()`方法；就像我们的UDP聊天一样，这个方法将首先将消息格式化为原始传输字符串：

```py
    def send_message(self, message):
        raw_message = f'{self.username}{self.delimiter}{message}'
```

现在我们需要连接到要通信的远程主机：

```py
    socket_state = self.client_socket.state()
    if socket_state != qtn.QAbstractSocket.ConnectedState:
        self.client_socket.connectToHost(
            self.recipient, self.port)
```

套接字的`state`属性可以告诉我们套接字是否连接到远程主机。`QAbstractSocket.ConnectedState`状态表示我们的客户端已连接到服务器。如果没有，我们调用套接字的`connectToHost()`方法来建立与接收主机的连接。

现在我们可以相当肯定我们已经连接了，让我们发送消息。为了做到这一点，我们再次转向`QDataStream`对象来处理与我们的TCP套接字通信的细节。

首先创建一个附加到客户端套接字的新数据流：

```py
        self.datastream = qtc.QDataStream(self.client_socket)
```

现在我们可以使用`writeQString()`方法向数据流写入字符串：

```py
        self.datastream.writeQString(raw_message)
```

重要的是要理解，对象只能按照我们发送它们的顺序从数据流中提取。例如，如果我们想要在字符串前面加上它的长度，以便接收方可以检查它是否损坏，我们可以这样做：

```py
        self.datastream.writeUInt32(len(raw_message))
        self.datastream.writeQString(raw_message)
```

然后我们的`process_datastream()`方法需要相应地进行调整：

```py
    def process_datastream(self):
        #...
        message_length = self.datastream.readUInt32()
        raw_message = self.datastream.readQString()
```

在`send_message()`中我们需要做的最后一件事是本地发出我们的消息，以便本地显示可以显示它。由于这不是广播消息，我们的本地TCP服务器不会听到发送出去的消息。

在`send_message()`的末尾添加这个：

```py
        self.received.emit(self.username, message)
```

让我们总结一下这个后端的操作方式：

+   我们有一个TCP服务器组件：

+   TCP服务器对象在端口`7777`上监听来自远程主机的连接

+   当接收到连接时，它将连接存储为套接字，并等待来自该套接字的数据

+   当接收到数据时，它将从套接字中读取数据流，解释并发出

+   我们有一个TCP客户端组件：

+   当需要发送消息时，首先对其进行格式化

+   然后检查连接状态，如果需要建立连接

+   一旦确保连接状态，消息将被写入套接字使用数据流

# 连接我们的后端并进行测试

回到`MainWindow.__init__()`，我们需要添加相关的代码来创建我们的接口并连接信号：

```py
        recipient, _ = qtw.QInputDialog.getText(
            None, 'Recipient',
            'Specify of the IP or hostname of the remote host.')
        if not recipient:
            sys.exit()

        self.interface = TcpChatInterface(username, recipient)
        self.cw.submitted.connect(self.interface.send_message)
        self.interface.received.connect(self.cw.write_message)
        self.interface.error.connect(
            lambda x: qtw.QMessageBox.critical(None, 'Error', x))
```

由于我们需要一个接收者，我们将使用`QInputDialog`询问用户。这个对话框类允许您轻松地查询用户的单个值。在这种情况下，我们要求输入另一个系统的IP地址或主机名。这个值我们传递给`TcpChatInterface`构造函数。

代码的其余部分基本上与UDP聊天客户端相同。

要测试这个聊天客户端，您需要在同一网络上的另一台计算机上运行一个副本，或者在您自己的网络中可以访问的地址上运行。当您启动客户端时，请指定另一台计算机的IP或主机名。一旦两个客户端都在运行，您应该能够互发消息。如果您在第三台计算机上启动客户端，请注意您将看不到消息，因为它们只被发送到单台计算机。

# 使用`QNetworkAccessManager`进行HTTP通信

**超文本传输协议**（**HTTP**）是构建万维网的协议，也可以说是我们这个时代最重要的通信协议。我们当然可以在套接字上实现自己的HTTP通信，但Qt已经为我们完成了这项工作。`QNetworkAccessManager`类实现了一个可以传输HTTP请求和接收HTTP回复的对象。我们可以使用这个类来创建与Web服务和API通信的应用程序。

# 简单下载

为了演示`QNetworkAccessManager`的基本用法，我们将构建一个简单的命令行HTTP下载工具。打开一个名为`downloader.py`的空文件，让我们从一些导入开始：

```py
import sys
from os import path
from PyQt5 import QtNetwork as qtn
from PyQt5 import QtCore as qtc
```

由于我们这里不需要`QtWidgets`或`QtGui`，只需要`QtNetwork`和`QtCore`。我们还将使用标准库`path`模块进行一些基于文件系统的操作。

让我们为我们的下载引擎创建一个`QObject`子类：

```py
class Downloader(qtc.QObject):

    def __init__(self, url):
        super().__init__()
        self.manager = qtn.QNetworkAccessManager(
            finished=self.on_finished)
        self.request = qtn.QNetworkRequest(qtc.QUrl(url))
        self.manager.get(self.request)
```

在我们的下载引擎中，我们创建了一个`QNetworkAccessManager`，并将其`finished`信号连接到一个名为`on_finish()`的回调函数。当管理器完成网络事务并准备好处理回复时，它会发出`finished`信号，并将回复包含在信号中。

接下来，我们创建一个`QNetworkRequest`对象。`QNetworkRequest`代表我们发送到远程服务器的HTTP请求，并包含我们要发送的所有信息。在这种情况下，我们只需要构造函数中传入的URL。

最后，我们告诉我们的网络管理器使用`get()`执行请求。`get()`方法使用HTTP `GET`方法发送我们的请求，通常用于请求下载的信息。管理器将发送这个请求并等待回复。

当回复到来时，它将被发送到我们的`on_finished()`回调函数：

```py
    def on_finished(self, reply):
        filename = reply.url().fileName() or 'download'
        if path.exists(filename):
            print('File already exists, not overwriting.')
            sys.exit(1)
        with open(filename, 'wb') as fh:
            fh.write(reply.readAll())
        print(f"{filename} written")
        sys.exit(0)
```

这里的`reply`对象是一个`QNetworkReply`实例，其中包含从远程服务器接收的数据和元数据。

我们首先尝试确定一个文件名，我们将用它来保存文件。回复的`url`属性包含原始请求所发出的URL，我们可以查询URL的`fileName`属性。有时这是空的，所以我们将退而求其次使用`'download'`字符串。

接下来，我们将检查文件名是否已经存在于我们的系统上。出于安全考虑，如果存在，我们将退出，这样您就不会在测试这个演示时破坏重要文件。

最后，我们使用它的`readAll()`方法从回复中提取数据，并将这些数据写入本地文件。请注意，我们以`wb`模式（写入二进制）打开文件，因为`readAll()`以`QByteAarray`对象的形式返回二进制数据。

我们的`Downloader`类的主要执行代码最后出现：

```py
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <download url>')
        sys.exit(1)
    app = qtc.QCoreApplication(sys.argv)
    d = Downloader(sys.argv[1])
    sys.exit(app.exec_())
```

在这里，我们只是从命令行中获取第一个参数，并将其传递给我们的`Downloader`对象。请注意，我们使用的是`QCoreApplication`而不是`QApplication`；当您想要创建一个命令行Qt应用程序时，可以使用这个类。否则，它与`QApplication`是一样的。

简而言之，使用`QNetworkAccessManager`就是这么简单：

+   创建一个`QNetworkAccessManager`对象

+   创建一个`QNetworkRequest`对象

+   将请求传递给管理器的`get()`方法

+   在与管理器的`finished`信号连接的回调中处理回复

# 发布数据和文件

使用`GET`请求检索数据是相当简单的HTTP；为了更深入地探索PyQt5的HTTP通信，我们将构建一个实用程序，允许我们向远程URL发送带有任意键值和文件数据的`POST`请求。例如，这个实用程序可能对测试Web API很有用。

# 构建GUI

从[第4章](9281bd2a-64a1-4128-92b0-e4871b79c040.xhtml)的Qt应用程序模板的副本开始，*使用QMainWindow构建应用程序*，让我们将主要的GUI代码添加到`MainWindow.__init__()`方法中：

```py
        widget = qtw.QWidget(minimumWidth=600)
        self.setCentralWidget(widget)
        widget.setLayout(qtw.QVBoxLayout())
        self.url = qtw.QLineEdit()
        self.table = qtw.QTableWidget(columnCount=2, rowCount=5)
        self.table.horizontalHeader().setSectionResizeMode(
            qtw.QHeaderView.Stretch)
        self.table.setHorizontalHeaderLabels(['key', 'value'])
        self.fname = qtw.QPushButton(
            '(No File)', clicked=self.on_file_btn)
        submit = qtw.QPushButton('Submit Post', clicked=self.submit)
        response = qtw.QTextEdit(readOnly=True)
        for w in (self.url, self.table, self.fname, submit, response):
            widget.layout().addWidget(w)
```

这是一个建立在`QWidget`对象上的简单表单。有一个用于URL的输入行，一个用于输入键值对的表格小部件，以及一个用于触发文件对话框并存储所选文件名的按钮。

之后，我们有一个用于发送请求的`submit`按钮和一个只读文本编辑框，用于显示返回的结果。

`fname`按钮在单击时调用`on_file_btn()`，其代码如下：

```py
    def on_file_btn(self):
        filename, accepted = qtw.QFileDialog.getOpenFileName()
        if accepted:
            self.fname.setText(filename)
```

该方法只是调用`QFileDialog`函数来检索要打开的文件名。为了保持简单，我们采取了略微不正统的方法，将文件名存储为我们的`QPushButton`文本。

最后的`MainWindow`方法是`submit()`，当单击`submit`按钮时将调用该方法。在编写我们的Web后端之后，我们将回到该方法，因为它的操作取决于我们如何定义该后端。

# POST后端

我们的Web发布后端将基于`QObject`，这样我们就可以使用信号和槽。

首先通过子类化`QObject`并创建一个信号：

```py
class Poster(qtc.QObject):

    replyReceived = qtc.pyqtSignal(str)
```

当我们从服务器接收到我们正在发布的回复时，`replyReceived`信号将被发出，并携带回复的主体作为字符串。

现在让我们创建构造函数：

```py
    def __init__(self):
        super().__init__()
        self.nam = qtn.QNetworkAccessManager()
        self.nam.finished.connect(self.on_reply)
```

在这里，我们正在创建我们的`QNetworkAccessManager`对象，并将其`finished`信号连接到名为`on_reply()`的本地方法。

`on_reply()`方法将如下所示：

```py
    def on_reply(self, reply):
        reply_bytes = reply.readAll()
        reply_string = bytes(reply_bytes).decode('utf-8')
        self.replyReceived.emit(reply_string)
```

回想一下，`finished`信号携带一个`QNetworkReply`对象。我们可以调用它的`readAll()`方法来获取回复的主体作为`QByteArray`。就像我们对原始套接字数据所做的那样，我们首先将其转换为`bytes`对象，然后使用`decode()`方法将其转换为UTF-8 Unicode数据。最后，我们将使用来自服务器的字符串发出我们的`replyReceived`信号。

现在我们需要一个方法，实际上会将我们的键值数据和文件发布到URL。我们将其称为`make_request()`，并从以下位置开始：

```py
    def make_request(self, url, data, filename):
        self.request = qtn.QNetworkRequest(url)
```

与`GET`请求一样，我们首先从提供的URL创建一个`QNetworkRequest`对象。但与`GET`请求不同，我们的`POST`请求携带数据负载。为了携带这个负载，我们需要创建一个特殊的对象，可以与请求一起发送。

HTTP请求可以以几种方式格式化数据负载，但通过HTTP传输文件的最常见方式是使用**多部分表单**请求。这种请求包含键值数据和字节编码的文件数据，是通过提交包含输入小部件和文件小部件混合的HTML表单获得的。

要在PyQt中执行这种请求，我们将首先创建一个`QtNetwork.QHttpMultiPart`对象，如下所示：

```py
        self.multipart = qtn.QHttpMultiPart(
            qtn.QHttpMultiPart.FormDataType)
```

有不同类型的多部分HTTP消息，我们通过将`QtNetwork.QHttpMultiPart.ContentType`枚举常量传递给构造函数来定义我们想要的类型。我们在这里使用的是用于一起传输文件和表单数据的`FormDataType`类型。

HTTP多部分对象是一个包含`QHttpPart`对象的容器，每个对象代表我们数据负载的一个组件。我们需要从传入此方法的数据创建这些部分，并将它们添加到我们的多部分对象中。

让我们从我们的键值对开始：

```py
        for key, value in (data or {}).items():
            http_part = qtn.QHttpPart()
            http_part.setHeader(
                qtn.QNetworkRequest.ContentDispositionHeader,
                f'form-data; name="{key}"'
            )
            http_part.setBody(value.encode('utf-8'))
            self.multipart.append(http_part)
```

每个HTTP部分都有一个标头和一个主体。标头包含有关部分的元数据，包括其**Content-Disposition**—也就是它包含的内容。对于表单数据，那将是`form-data`。

因此，对于`data`字典中的每个键值对，我们正在创建一个单独的`QHttpPart`对象，将Content-Disposition标头设置为`form-data`，并将`name`参数设置为键。最后，我们将HTTP部分的主体设置为我们的值（编码为字节字符串），并将HTTP部分添加到我们的多部分对象中。

要包含我们的文件，我们需要做类似的事情：

```py
        if filename:
            file_part = qtn.QHttpPart()
            file_part.setHeader(
                qtn.QNetworkRequest.ContentDispositionHeader,
                f'form-data; name="attachment"; filename="{filename}"'
            )
            filedata = open(filename, 'rb').read()
            file_part.setBody(filedata)
            self.multipart.append(file_part)
```

这一次，我们的Content-Disposition标头仍然设置为`form-data`，但也包括一个`filename`参数，设置为我们文件的名称。HTTP部分的主体设置为文件的内容。请注意，我们以`rb`模式打开文件，这意味着它的二进制内容将被读取为`bytes`对象，而不是将其解释为纯文本。这很重要，因为`setBody()`期望的是bytes而不是Unicode。

现在我们的多部分对象已经构建好了，我们可以调用`QNetworkAccessManager`对象的`post()`方法来发送带有多部分数据的请求：

```py
        self.nam.post(self.request, self.multipart)
```

回到`MainWindow.__init__()`，让我们创建一个`Poster`对象来使用：

```py
        self.poster = Poster()
        self.poster.replyReceived.connect(self.response.setText)
```

由于`replyReceived`将回复主体作为字符串发出，我们可以直接将其连接到响应小部件的`setText`上，以查看服务器的响应。

最后，是时候创建我们的`submit()`回调了：

```py
    def submit(self):
        url = qtc.QUrl(self.url.text())
        filename = self.fname.text()
        if filename == '(No File)':
            filename = None
        data = {}
        for rownum in range(self.table.rowCount()):
            key_item = self.table.item(rownum, 0)
            key = key_item.text() if key_item else None
            if key:
                data[key] = self.table.item(rownum, 1).text()
        self.poster.make_request(url, data, filename)
```

请记住，`make_request()`需要`QUrl`、键值对的`dict`和文件名字符串；因此，这个方法只是遍历每个小部件，提取和格式化数据，然后将其传递给`make_request()`。

# 测试实用程序

如果您可以访问接受POST请求和文件上传的服务器，您可以使用它来测试您的脚本；如果没有，您也可以使用本章示例代码中包含的`sample_http_server.py`脚本。这个脚本只需要Python 3和标准库，它会将您的POST请求回显给您。

在控制台窗口中启动服务器脚本，然后在第二个控制台中运行您的`poster.py`脚本，并执行以下操作：

+   输入URL为`http://localhost:8000`

+   向表中添加一些任意的键值对

+   选择要上传的文件（可能是一个不太大的文本文件，比如您的Python脚本之一）

+   点击提交帖子

您应该在服务器控制台窗口和GUI上的响应文本编辑中看到您请求的打印输出。它应该是这样的：

![](assets/fbc5b22b-9a2f-4e97-8897-c328187ecffd.png)

总之，使用`QNetworkAccessManager`处理`POST`请求涉及以下步骤：

+   创建`QNetworkAccessManager`并将其`finished`信号连接到将处理`QNetworkReply`的方法

+   创建指向目标URL的`QNetworkRequest`

+   创建数据有效负载对象，比如`QHttpMultiPart`对象

+   将请求和数据有效负载传递给`QNetworkAccessManager`对象的`post()`方法

# 总结

在本章中，我们探讨了如何将我们的PyQt应用程序连接到网络。您学会了如何使用套接字进行低级编程，包括UDP广播应用程序和TCP客户端-服务器应用程序。您还学会了如何使用`QNetworkAccessManager`与HTTP服务进行交互，从简单的下载到复杂的多部分表单和文件数据上传。

下一章将探讨使用SQL数据库存储和检索数据。您将学习如何构建和查询SQL数据库，如何使用`QtSQL`模块将SQL命令集成到您的应用程序中，以及如何使用SQL模型视图组件快速构建数据驱动的GUI应用程序。

# 问题

尝试这些问题来测试您从本章中学到的知识：

1.  您正在设计一个应用程序，该应用程序将向本地网络发出状态消息，您将使用管理员工具进行监视。哪种类型的套接字对象是一个不错的选择？

1.  你的GUI类有一个名为`self.socket`的`QTcpSocket`对象。你已经将它的`readyRead`信号连接到以下方法，但它不起作用。发生了什么，你该如何修复它？

```py
       def on_ready_read(self):
           while self.socket.hasPendingDatagrams():
               self.process_data(self.socket.readDatagram())
```

1.  使用`QTcpServer`来实现一个简单的服务，监听端口`8080`，并打印接收到的任何请求。让它用你选择的字节字符串回复客户端。

1.  你正在为你的应用程序创建一个下载函数，用于获取一个大数据文件以导入到你的应用程序中。代码不起作用。阅读代码并决定你做错了什么：

```py
       def download(self, url):
        self.manager = qtn.QNetworkAccessManager(
            finished=self.on_finished)
        self.request = qtn.QNetworkRequest(qtc.QUrl(url))
        reply = self.manager.get(self.request)
        with open('datafile.dat', 'wb') as fh:
            fh.write(reply.readAll())
```

1.  修改你的`poster.py`脚本，以便将键值数据发送为JSON，而不是HTTP表单数据。

# 进一步阅读

欲了解更多信息，请参考以下内容：

+   有关数据报包结构的更多信息，请参阅[https://en.wikipedia.org/wiki/Datagram](https://en.wikipedia.org/wiki/Datagram)。

+   随着对网络通信中安全和隐私的关注不断增加，了解如何使用SSL是很重要的。请参阅[https://doc.qt.io/qt-5/ssl.html](https://doc.qt.io/qt-5/ssl.html) 了解使用SSL的`QtNetwork`工具的概述。

+   **Mozilla开发者网络**在[https://developer.mozilla.org/en-US/docs/Web/HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP)上有大量资源，用于理解HTTP及其各种标准和协议。
