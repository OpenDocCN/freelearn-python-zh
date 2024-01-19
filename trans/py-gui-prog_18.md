# 第十八章：网络和管理大型文档

在本章中，我们将学习如何使用网络概念以及如何以块的形式查看大型文档。我们将涵盖以下主题：

+   创建一个小型浏览器

+   创建一个服务器端应用程序

+   建立客户端-服务器通信

+   创建一个可停靠和可浮动的登录表单

+   多文档界面

+   使用选项卡小部件在部分中显示信息

+   创建自定义菜单栏

# 介绍

设备屏幕上的空间总是有限的，但有时您会遇到这样的情况：您想在屏幕上显示大量信息或服务。在这种情况下，您可以使用可停靠的小部件，这些小部件可以在屏幕的任何位置浮动；MDI 可以根据需要显示多个文档；选项卡小部件框可以显示不同块中的信息；或者菜单可以在单击菜单项时显示所需的信息。此外，为了更好地理解网络概念，您需要了解客户端和服务器如何通信。本章将帮助您了解所有这些。

# 创建一个小型浏览器

现在让我们学习一种显示网页或 HTML 文档内容的技术。我们将简单地使用 LineEdit 和 PushButton 小部件，以便用户可以输入所需站点的 URL，然后点击 PushButton 小部件。单击按钮后，该站点将显示在自定义小部件中。让我们看看。

在这个示例中，我们将学习如何制作一个小型浏览器。因为 Qt Designer 没有包含任何特定的小部件，所以这个示例的重点是让您了解如何将自定义小部件提升为`QWebEngineView`，然后可以用于显示网页。

应用程序将提示输入 URL，当用户输入 URL 并点击“Go”按钮后，指定的网页将在`QWebEngineView`对象中打开。

# 如何做...

在这个示例中，我们只需要三个小部件：一个用于输入 URL，第二个用于点击按钮，第三个用于显示网站。以下是创建一个简单浏览器的步骤：

1.  基于没有按钮的对话框模板创建一个应用程序。

1.  通过拖放 Label、LineEdit、PushButton 和 Widget 将`QLabel`、`QLineEdit`、`QPushButton`和`QWidget`小部件添加到表单中。

1.  将 Label 小部件的文本属性设置为“输入 URL”。

1.  将 PushButton 小部件的文本属性设置为`Go`。

1.  将 LineEdit 小部件的 objectName 属性设置为`lineEditURL`，将 PushButton 小部件的 objectName 属性设置为`pushButtonGo`。

1.  将应用程序保存为`demoBrowser.ui`。

表单现在将显示如下截图所示：

![](img/7edee36a-8e27-42e5-b7b6-de89b3c3fd4c.png)

1.  下一步是将`QWidget`提升为`QWebEngineView`，因为要显示网页，需要`QWebEngineView`。

1.  通过右键单击 QWidget 对象并从弹出菜单中选择“提升为...”选项来提升`QWidget`对象。

1.  在弹出的对话框中，将基类名称选项保留为默认的 QWidget。

1.  在 Promoted 类名框中输入`QWebEngineView`，在头文件框中输入`PyQt5.QtWebEngineWidgets`。

1.  选择 Promote 按钮，将 QWidget 提升为`QWebEngineView`类，如下截图所示：

![](img/03e31411-d4ea-45fe-997b-13b698a4aebe.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。

1.  要进行转换，您需要打开命令提示符窗口并导航到保存文件的文件夹，然后发出以下命令：

```py
C:\Pythonbook\PyQt5>pyuic5 demoBrowser.ui -o demoBrowser.py
```

您可以在本书的源代码包中看到自动生成的 Python 脚本文件`demoBrowser.py`。

1.  将上述代码视为一个头文件，并将其导入到将调用其用户界面设计的文件中。

1.  让我们创建另一个名为`callBrowser.pyw`的 Python 文件，并将`demoBrowser.py`代码导入其中：

```py
import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView
from demoBrowser import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonGo.clicked.connect(self.dispSite)
        self.show()
    def dispSite(self):
        self.ui.widget.load(QUrl(self.ui.lineEditURL.text()))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

在`demoBrowser.py`文件中，创建了一个名为顶级对象的类，前面加上`Ui_`。也就是说，对于顶级对象`Dialog`，创建了`Ui_Dialog`类，并存储了我们小部件的接口元素。该类包括两个方法，`setupUi()`和`retranslateUi()`。`setupUi()`方法创建了在 Qt Designer 中定义用户界面时使用的小部件。此方法还设置了小部件的属性。`setupUi()`方法接受一个参数，即应用程序的顶级小部件，即`QDialog`的实例。`retranslateUi()`方法翻译了界面。

在`callBrowser.pyw`文件中，您会看到推送按钮小部件的 click()事件连接到`dispSite`方法；在行编辑小部件中输入 URL 后，当用户单击推送按钮时，将调用`dispSite`方法。

`dispSite()`方法调用`QWidget`类的`load()`方法。请记住，`QWidget`对象被提升为`QWebEngineView`类，用于查看网页。`QWebEngineView`类的`load()`方法接收`lineEditURL`对象中输入的 URL，因此指定 URL 的网页将在`QWebEngine`小部件中打开或加载。

运行应用程序时，您会得到一个空的行编辑框和一个推送按钮小部件。在行编辑小部件中输入所需的 URL，然后单击“Go”按钮，您会发现网页在`QWebEngineView`小部件中打开，如下屏幕截图所示：

![](img/4f26797c-c0ca-406f-886f-3fb16878c474.png)

# 创建服务器端应用程序

网络在现代生活中扮演着重要角色。我们需要了解两台机器之间的通信是如何建立的。当两台机器通信时，一台通常是服务器，另一台是客户端。客户端向服务器发送请求，服务器通过为客户端提出的请求提供响应。

在本示例中，我们将创建一个客户端-服务器应用程序，在客户端和服务器之间建立连接，并且每个都能够向另一个传输文本消息。也就是说，将创建两个应用程序，并且将同时执行，一个应用程序中编写的文本将出现在另一个应用程序中。

# 如何做...

让我们首先创建一个服务器应用程序，如下所示：

1.  基于无按钮对话框模板创建应用程序。

1.  通过将标签、文本编辑、行编辑和推送按钮小部件拖放到表单上，向表单添加`QLabel`、`QTextEdit`、`QLineEdit`和`QPushButton`。

1.  将标签小部件的文本属性设置为“服务器”，以指示这是服务器应用程序。

1.  将推送按钮小部件的文本属性设置为“发送”。

1.  将文本编辑小部件的对象名称属性设置为`textEditMessages`。

1.  将行编辑小部件的对象名称属性设置为`lineEditMessage`。

1.  将推送按钮小部件设置为`pushButtonSend`。

1.  将应用程序保存为`demoServer.ui`。表单现在将显示如下屏幕截图所示：

![](img/88690655-831b-4b2a-a84e-5c04aabf9418.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。生成文件`demoServer.py`的代码可以在本书的源代码包中看到。

# 工作原理...

`demoServer.py`文件将被视为头文件，并将被导入到另一个 Python 文件中，该文件将使用头文件的 GUI 并在服务器和客户端之间传输数据。但在此之前，让我们为客户端应用程序创建一个 GUI。客户端应用程序的 GUI 与服务器应用程序完全相同，唯一的区别是该应用程序顶部的标签小部件将显示文本“客户端”。

`demoServer.py`文件是我们拖放到表单上的 GUI 小部件的生成 Python 脚本。

要在服务器和客户端之间建立连接，我们需要一个套接字对象。要创建套接字对象，您需要提供以下两个参数：

+   **套接字地址**：套接字地址使用特定的地址系列表示。每个地址系列都需要一些参数来建立连接。在本应用程序中，我们将使用`AF_INET`地址系列。`AF_INET`地址系列需要一对（主机，端口）来建立连接，其中参数`host`是主机名，可以是字符串格式、互联网域表示法或 IPv4 地址格式，参数`port`是用于通信的端口号。

+   **套接字类型**：套接字类型通过几个常量表示：`SOCK_STREAM`、`SOCK_DGRAM`、`SOCK_RAW`、`SOCK_RDM`和`SOCK_SEQPACKET`。在本应用程序中，我们将使用最常用的套接字类型`SOCK_STREAM`。

应用程序中使用`setsockopt()`方法设置给定套接字选项的值。它包括以下两个基本参数：

+   `SOL_SOCKET`：此参数是套接字层本身。它用于协议无关的选项。

+   `SO_REUSEADDR`：此参数允许其他套接字`bind()`到此端口，除非已经有一个活动的监听套接字绑定到该端口。

您可以在先前的代码中看到，创建了一个`ServerThread`类，它继承了 Python 的线程模块的`Thread`类。`run()`函数被重写，其中定义了`TCP_IP`和`TCP_HOST`变量，并且`tcpServer`与这些变量绑定。

此后，服务器等待看是否有任何客户端连接。对于每个新的客户端连接，服务器在`while`循环内创建一个新的`ClientThread`。这是因为为每个客户端创建一个新线程不会阻塞服务器的 GUI 功能。最后，线程被连接。

# 建立客户端-服务器通信

在这个教程中，我们将学习如何制作一个客户端，并看到它如何向服务器发送消息。主要思想是理解消息是如何发送的，服务器如何监听端口，以及两者之间的通信是如何建立的。

# 如何做...

要向服务器发送消息，我们将使用`LineEdit`和`PushButton`小部件。在单击推送按钮时，LineEdit 小部件中编写的消息将传递到服务器。以下是创建客户端应用程序的逐步过程：

1.  基于没有按钮的对话框模板创建另一个应用程序。

1.  通过将 Label、TextEdit、LineEdit 和 PushButton 小部件拖放到表单上，向表单添加`QLabel`、`QTextEdit`、`QLineEdit`和`QPushButton`。

1.  将 Label 小部件的文本属性设置为`Client`。

1.  将 PushButton 小部件的文本属性设置为`Send`。

1.  将 TextEdit 小部件的 objectName 属性设置为`textEditMessages`。

1.  将 LineEdit 小部件的 objectName 属性设置为`lineEditMessage`。

1.  将 PushButton 小部件设置为`pushButtonSend`。

1.  将应用程序保存为`demoClient.ui`。

表单现在将显示如下截图所示：

![](img/f513ff3a-a1bc-4cb3-8baf-e2b9e965d3f0.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。自动生成的文件`demoClient.py`的代码可以在本书的源代码包中找到。要使用`demoClient.py`文件中创建的 GUI，需要将其导入到另一个 Python 文件中，该文件将使用 GUI 并在服务器和客户端之间传输数据。

1.  创建另一个名为`callServer.pyw`的 Python 文件，并将`demoServer.py`代码导入其中。`callServer.pyw`脚本中的代码如下所示：

```py
import sys, time
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtCore import QCoreApplication
import socket
from threading import Thread
from socketserver import ThreadingMixIn
conn=None
from demoServer import *
class Window(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.textEditMessages=self.ui.textEditMessages
        self.ui.pushButtonSend.clicked.connect(self.dispMessage)
        self.show()

    def dispMessage(self):
        text=self.ui.lineEditMessage.text()
        global conn
        conn.send(text.encode("utf-8"))
        self.ui.textEditMessages.append("Server:   
        "+self.ui.lineEditMessage.text())
        self.ui.lineEditMessage.setText("")
class ServerThread(Thread):
    def __init__(self,window):
        Thread.__init__(self)
        self.window=window
    def run(self):
        TCP_IP = '0.0.0.0'
        TCP_PORT = 80
        BUFFER_SIZE = 1024
        tcpServer = socket.socket(socket.AF_INET,  
        socket.SOCK_STREAM)
        tcpServer.setsockopt(socket.SOL_SOCKET,         
        socket.SO_REUSEADDR, 1)
        tcpServer.bind((TCP_IP, TCP_PORT))
        threads = []
        tcpServer.listen(4)
        while True:
            global conn
            (conn, (ip,port)) = tcpServer.accept()
            newthread = ClientThread(ip,port,window)
            newthread.start()
            threads.append(newthread)
        for t in threads:
            t.join()
class ClientThread(Thread):
    def __init__(self,ip,port,window):
        Thread.__init__(self)
        self.window=window
        self.ip = ip
        self.port = port
    def run(self):
        while True :
            global conn
            data = conn.recv(1024)
            window.textEditMessages.append("Client: 
            "+data.decode("utf-8"))

if __name__=="__main__":
    app = QApplication(sys.argv)
    window = Window()
    serverThread=ServerThread(window)
    serverThread.start()
    window.exec()
    sys.exit(app.exec_())
```

# 工作原理...

在`ClientThread`类中，`run`函数被重写。在`run`函数中，每个客户端等待从服务器接收的数据，并在文本编辑小部件中显示该数据。一个`window`类对象被传递给`ServerThread`类，后者将该对象传递给`ClientThread`，后者又使用它来访问在行编辑元素中编写的内容。

接收到的数据被解码，因为接收到的数据是以字节形式，必须使用 UTF-8 编码转换为字符串。

在前面的部分生成的`demoClient.py`文件需要被视为一个头文件，并且需要被导入到另一个 Python 文件中，该文件将使用头文件的 GUI 并在客户端和服务器之间传输数据。因此，让我们创建另一个名为`callClient.pyw`的 Python 文件，并将`demoClient.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QApplication, QDialog
import socket
from threading import Thread
from socketserver import ThreadingMixIn
from demoClient import *
tcpClientA=None
class Window(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.textEditMessages=self.ui.textEditMessages
        self.ui.pushButtonSend.clicked.connect(self.dispMessage)
        self.show()
    def dispMessage(self):
        text=self.ui.lineEditMessage.text()
        self.ui.textEditMessages.append("Client:  
        "+self.ui.lineEditMessage.text())
        tcpClientA.send(text.encode())
        self.ui.lineEditMessage.setText("")
class ClientThread(Thread):
    def __init__(self,window):
        Thread.__init__(self)
        self.window=window
    def run(self):
        host = socket.gethostname()
        port = 80
        BUFFER_SIZE = 1024
        global tcpClientA
        tcpClientA = socket.socket(socket.AF_INET, 
        socket.SOCK_STREAM)
        tcpClientA.connect((host, port))
        while True:
            data = tcpClientA.recv(BUFFER_SIZE)
            window.textEditMessages.append("Server: 
            "+data.decode("utf-8"))
            tcpClientA.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    window = Window()
    clientThread=ClientThread(window)
    clientThread.start()
    window.exec()
    sys.exit(app.exec_())
```

`ClientThread`类是一个继承`Thread`类并重写`run`函数的类。在`run`函数中，通过在`socket`类上调用`hostname`方法来获取服务器的 IP 地址；并且，使用端口`80`，客户端尝试连接到服务器。一旦与服务器建立连接，客户端尝试在 while 循环内从服务器接收数据。

从服务器接收数据后，将数据从字节格式转换为字符串格式，并显示在文本编辑小部件中。

我们需要运行两个应用程序来查看客户端-服务器通信。运行`callServer.pyw`文件，您将在以下截图的左侧看到输出，运行`callClient.pyw`文件，您将在右侧看到输出。两者相同；只有顶部的标签有所区别：

![](img/61308f2b-a3b2-4762-8c4f-ed9a4dfe39fc.png)

用户可以在底部的行编辑框中输入文本，然后按下发送按钮。按下发送按钮后，输入的文本将出现在服务器和客户端应用程序的文本编辑框中。文本以`Server:`为前缀，以指示该文本是从服务器发送的，如下截图所示：

![](img/e91410d8-0334-44a0-8414-ba6f766587fe.png)

同样，如果在客户端应用程序的行编辑小部件中输入文本，然后按下发送按钮，文本将出现在两个应用程序的文本编辑小部件中。文本将以`Client:`为前缀，以指示该文本已从客户端发送，如下截图所示：

![](img/17844150-0e96-41b8-9d78-ff12122757dd.png)

# 创建一个可停靠和可浮动的登录表单

在本教程中，我们将学习创建一个登录表单，该表单将要求用户输入电子邮件地址和密码以进行身份验证。这个登录表单不同于通常的登录表单，因为它是一个可停靠的表单。也就是说，您可以将这个登录表单停靠在窗口的四个边缘之一——顶部、左侧、右侧和底部，甚至可以将其用作可浮动的表单。这个可停靠的登录表单将使用 Dock 小部件创建，所以让我们快速了解一下 Dock 小部件。

# 准备工作

要创建一组可分离的小部件或工具，您需要一个 Dock 小部件。Dock 小部件是使用`QDockWidget`类创建的，它是一个带有标题栏和顶部按钮的容器，用于调整大小。包含一组小部件或工具的 Dock 小部件可以关闭、停靠在停靠区域中，或者浮动并放置在桌面的任何位置。Dock 小部件可以停靠在不同的停靠区域，例如`LeftDockWidgetArea`、`RightDockWidgetArea`、`TopDockWidgetArea`和`BottomDockWidgetArea`。`TopDockWidgetArea`停靠区域位于工具栏下方。您还可以限制 Dock 小部件可以停靠的停靠区域。这样做后，Dock 小部件只能停靠在指定的停靠区域。当将 Dock 窗口拖出停靠区域时，它将成为一个自由浮动的窗口。

以下是控制 Dock 小部件的移动以及其标题栏和其他按钮外观的属性：

| **属性** | **描述** |
| --- | --- |
| `DockWidgetClosable` | 使 Dock 小部件可关闭。 |
| `DockWidgetMovable` | 使 Dock 小部件在停靠区域之间可移动。 |
| `DockWidgetFloatable` | 使 Dock 小部件可浮动，也就是说，Dock 小部件可以从主窗口中分离并在桌面上浮动。 |
| `DockWidgetVerticalTitleBar` | 在 Dock 小部件的左侧显示垂直标题栏。 |
| `AllDockWidgetFeatures` | 它打开属性，如`DockWidgetClosable`，`DockWidgetMovable`和`DockWidgetFloatable`，也就是说，Dock 小部件可以关闭，移动或浮动。 |
| `NoDockWidgetFeatures` | 如果选择，Dock 小部件将无法关闭，移动或浮动。 |

为了制作可停靠的登录表单，我们将使用 Dock 小部件和其他一些小部件。让我们看看逐步的操作步骤。

# 如何做...

让我们在 Dock 小部件中制作一个小的登录表单，提示用户输入其电子邮件地址和密码。由于可停靠，此登录表单可以移动到屏幕上的任何位置，并且可以浮动。以下是创建此应用程序的步骤：

1.  启动 Qt Designer 并创建一个新的主窗口应用程序。

1.  将一个 Dock 小部件拖放到表单上。

1.  拖放您希望在停靠区域或作为浮动窗口在 Dock 小部件中可用的小部件。

1.  在 Dock 小部件上拖放三个 Label 小部件，两个 LineEdit 小部件和一个 PushButton 小部件。

1.  将三个 Label 小部件的文本属性设置为`登录`，`电子邮件地址`和`密码`。

1.  将 Push Button 小部件的文本属性设置为`登录`。

1.  我们将不设置 LineEdit 和 PushButton 小部件的 objectName 属性，并且不会为 PushButton 小部件提供任何代码，因为此应用程序的目的是了解 Dock 小部件的工作原理。

1.  将应用程序保存为`demoDockWidget.ui`。

表单将显示如下屏幕截图所示：

![](img/f23980de-4068-42d2-aca1-6e4cf5dfcd66.png)

1.  要启用 Dock 小部件中的所有功能，请选择它并在属性编辑器窗口的功能部分中检查其 AllDockWidgetFeatures 属性，如下图所示：

![](img/fca7bb78-58e2-446e-a112-ca985a7a692c.png)

在上述屏幕截图中，AllDockWidgetFeatures 属性是使 Dock 小部件可关闭，在停靠时可移动，并且可以在桌面的任何位置浮动。如果选择了 NoDockWidgetFeatures 属性，则功能部分中的所有其他属性将自动取消选中。这意味着所有按钮将从 Dock 小部件中消失，您将无法关闭或移动它。如果希望 Dock 小部件在应用程序启动时显示为可浮动，请在属性编辑器窗口中的功能部分上方检查浮动属性。

查看以下屏幕截图，显示了 Dock 小部件上的各种功能和约束：

![](img/8a6db2f9-92ff-4fb9-9d8c-92b896669a56.png)

执行以下步骤，将所需的功能和约束应用于 Dock 小部件：

1.  在 allowedAreas 部分中检查 AllDockWidgetAreas 选项，以使 Dock 小部件可以停靠在左侧，右侧，顶部和底部的所有 Dock 小部件区域。

1.  此外，通过在属性编辑器窗口中使用 windowTitle 属性，将停靠窗口的标题设置为 Dockable Sign In Form，如上图所示。

1.  检查停靠属性，因为这是使 Dock 小部件可停靠的重要属性。如果未选中停靠属性，则 Dock 小部件无法停靠到任何允许的区域。

1.  将 dockWidgetArea 属性保留其默认值 LeftDockWidgetArea。dockWidgetArea 属性确定您希望停靠窗口小部件在应用程序启动时出现为停靠的位置。dockWidgetArea 属性的 LeftDockWidgetArea 值将使停靠窗口小部件首先出现为停靠在左侧停靠窗口区域。如果在 allowedAreas 部分设置了 NoDockWidgetArea 属性，则 allowedAreas 部分中的所有其他属性将自动取消选择。因此，您可以将停靠窗口移动到桌面的任何位置，但不能将其停靠在主窗口模板的停靠区域中。使用 Qt Designer 创建的用户界面存储在一个`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。在 XML 文件上应用`pyuic5`命令行实用程序后，生成的文件是一个 Python 脚本文件`demoDockWidget.py`。您可以在本书的源代码包中看到生成的`demoDockWidget.py`文件的代码。

1.  将`demoDockWidget.py`文件中的代码视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callDockWidget.pyw`的 Python 文件，并将`demoDockWidget.py`的代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from demoDockWidget import *
class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

如前面的代码所示，导入了必要的模块。创建了一个`AppWindow`类，它继承自基类`QMainWindow`。调用了`QMainWindow`的默认构造函数。

因为每个 PyQt5 应用程序都需要一个应用程序对象，在上面的代码中，通过调用`QApplication()`方法创建了一个名为 app 的应用程序对象。将`sys.argv`参数作为参数传递给`QApplication()`方法，以传递命令行参数和其他外部属性给应用程序。`sys.argv`参数包含命令行参数和其他外部属性（如果有的话）。为了显示界面中定义的小部件，创建了一个名为`w`的`AppWindow`类的实例，并在其上调用了`show()`方法。为了退出应用程序并将代码返回给可能用于错误处理的 Python 解释器，调用了`sys.exit()`方法。

当应用程序执行时，默认情况下会得到一个停靠在左侧可停靠区域的停靠窗口小部件，如下面的屏幕截图所示。这是因为您已经将`dockWidgetArea`属性的值分配给了`LeftDockWidgetArea`：

![](img/81e13139-7926-4a9a-88b9-0681a52e6606.png)

停靠窗口小部件内的小部件不完全可见，因为默认的左侧和可停靠区域比停靠窗口小部件中放置的小部件要窄。因此，您可以拖动停靠窗口小部件的右边框，使所有包含的小部件可见，如下面的屏幕截图所示：

![](img/ede66a10-a27a-4042-9199-7976711f57af.png)

您可以将小部件拖动到任何区域。如果将其拖动到顶部，则会停靠在`TopDockWidgetArea`停靠区域，如下面的屏幕截图所示：

![](img/b2259d77-1078-4302-9c9c-e1f24625a314.png)

同样，当将停靠窗口小部件拖动到右侧时，它将停靠在`RightDockWidgetArea`中

您可以将停靠窗口小部件拖动到主窗口模板之外，使其成为一个独立的浮动窗口。停靠窗口小部件将显示为一个独立的浮动窗口，并可以移动到桌面的任何位置：

![](img/f01528da-0749-4beb-9fb3-8c921e4cf9c5.png)

# 多文档界面

在这个示例中，我们将学习如何创建一个应用程序，可以同时显示多个文档。我们不仅能够管理多个文档，还将学会以不同的格式排列这些文档。我们将能够使用称为多文档界面的概念来管理多个文档，让我们快速了解一下这个概念。

# 准备工作

通常，一个应用程序提供一个主窗口对应一个文档，这样的应用程序被称为**单文档界面**（**SDI**）应用程序。顾名思义，**多文档界面**（**MDI**）应用程序能够显示多个文档。MDI 应用程序由一个主窗口以及一个菜单栏、一个工具栏和一个中心空间组成。多个文档可以显示在中心空间中，每个文档可以通过各自的子窗口小部件进行管理；在 MDI 中，可以显示多个文档，每个文档都显示在自己的窗口中。这些子窗口也被称为子窗口。

MDI 是通过使用`MdiArea`小部件来实现的。`MdiArea`小部件提供了一个区域，用于显示子窗口。子窗口有标题和按钮，用于显示、隐藏和最大化其大小。每个子窗口可以显示一个单独的文档。可以通过设置`MdiArea`小部件的相应属性，将子窗口以级联或平铺方式排列。`MdiArea`小部件是`QMdiArea`类的一个实例，子窗口是`QMdiSubWindow`的实例。

以下是`QMdiArea`提供的方法：

+   `subWindowList()`: 这个方法返回 MDI 区域中所有子窗口的列表。返回的列表按照通过`WindowOrder()`函数设置的顺序排列。

+   `WindowOrder`：这个静态变量设置了对子窗口列表进行排序的标准。以下是可以分配给这个静态变量的有效值：

+   `CreationOrder`：窗口按照它们创建的顺序返回。这是默认顺序。

+   `StackingOrder`：窗口按照它们叠放的顺序返回，最上面的窗口最后出现在列表中。

+   `ActivationHistoryOrder`：窗口按照它们被激活的顺序返回。

+   `activateNextSubWindow()`: 这个方法将焦点设置为子窗口列表中的下一个窗口。当前窗口的顺序决定了要激活的下一个窗口。

+   `activatePreviousSubWindow()`: 这个方法将焦点设置为子窗口列表中的上一个窗口。当前窗口的顺序决定了要激活的上一个窗口。

+   `cascadeSubWindows()`: 这个方法以级联方式排列子窗口。

+   `tileSubWindows()`: 这个方法以平铺方式排列子窗口。

+   `closeAllSubWindows()`: 这个方法关闭所有子窗口。

+   `setViewMode()`: 这个方法设置 MDI 区域的视图模式。子窗口可以以两种模式查看，子窗口视图和选项卡视图：

+   子窗口视图：这个方法显示带有窗口框架的子窗口（默认）。如果以平铺方式排列，可以看到多个子窗口的内容。它还由一个常量值`0`表示。

+   选项卡视图：在选项卡栏中显示带有选项卡的子窗口。一次只能看到一个子窗口的内容。它还由一个常量值`1`表示。

# 如何做...

让我们创建一个应用程序，其中包含两个文档，每个文档将通过其各自的子窗口显示。我们将学习如何按需排列和查看这些子窗口：

1.  启动 Qt Designer 并创建一个新的主窗口应用程序。

1.  将`MdiArea`小部件拖放到表单上。

1.  右键单击小部件，从上下文菜单中选择“添加子窗口”以将子窗口添加到`MdiArea`小部件中。

当子窗口添加到`MdiArea`小部件时，该小部件将显示为深色背景，如下面的屏幕截图所示：

![](img/bfb235c6-0e67-49ea-aad6-142b11a6c7d2.png)

1.  让我们再次右键单击`MdiArea`小部件，并向其添加一个子窗口。

1.  要知道哪一个是第一个，哪一个是第二个子窗口，可以在每个子窗口上拖放一个 Label 小部件。

1.  将放置在第一个子窗口中的 Label 小部件的文本属性设置为`First subwindow`。

1.  将放置在第二个子窗口中的 Label 小部件的文本属性设置为`Second subwindow`，如下面的屏幕截图所示：

![](img/4847a248-744d-47d0-b837-3ddeac8fd0dd.png)

`MdiArea`小部件以以下两种模式显示放置在其子窗口中的文档：

+   子窗口视图：这是默认视图模式。在此视图模式下，子窗口可以以级联或平铺方式排列。当子窗口以平铺方式排列时，可以同时看到多个子窗口的内容。

+   选项卡视图：在此模式下，选项卡栏中会显示多个选项卡。选择选项卡时，将显示与之关联的子窗口。一次只能看到一个子窗口的内容。

1.  通过菜单选项激活子窗口视图和选项卡视图模式，双击菜单栏中的 Type Here 占位符，并向其添加两个条目：子窗口视图和选项卡视图。

此外，为了查看子窗口以级联和平铺方式排列时的外观，将两个菜单项 Cascade View 和 Tile View 添加到菜单栏中，如下面的屏幕截图所示：

![](img/1aed9dce-6c6e-4c3c-9dc7-9aacb6562892.png)

1.  将应用程序保存为`demoMDI.ui`。使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。在应用`pyuic5`命令行实用程序时，`.ui`（XML）文件将被转换为 Python 代码：

```py
 C:\Pythonbook\PyQt5>pyuic5 demoMDI.ui -o demoMDI.py.
```

您可以在本书的源代码包中看到生成的 Python 代码`demoMDI.py`。

1.  将`demoMDI.py`文件中的代码视为头文件，并将其导入到您将调用其用户界面设计的文件中。前面的代码中的用户界面设计包括`MdiArea`，用于显示其中创建的子窗口以及它们各自的小部件。我们将要创建的 Python 脚本将包含用于执行不同任务的菜单选项的代码，例如级联和平铺子窗口，将视图模式从子窗口视图更改为选项卡视图，反之亦然。让我们将该 Python 脚本命名为`callMDI.pyw`，并将`demoMDI.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QFileDialog
from demoMDI import *
class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.mdiArea.addSubWindow(self.ui.subwindow)
        self.ui.mdiArea.addSubWindow(self.ui.subwindow_2)
        self.ui.actionSubWindow_View.triggered.connect
        (self.SubWindow_View)
        self.ui.actionTabbed_View.triggered.connect(self.
        Tabbed_View)
        self.ui.actionCascade_View.triggered.connect(self.
        cascadeArrange)
        self.ui.actionTile_View.triggered.connect(self.tileArrange)
        self.show()
    def SubWindow_View(self):
        self.ui.mdiArea.setViewMode(0)
    def Tabbed_View(self):
        self.ui.mdiArea.setViewMode(1)
    def cascadeArrange(self):
        self.ui.mdiArea.cascadeSubWindows()
    def tileArrange(self):
        self.ui.mdiArea.tileSubWindows()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

在上述代码中，您可以看到具有默认 objectName 属性`subwindow`和`subwindow_2`的两个子窗口被添加到`MdiArea`小部件中。之后，具有 objectName 属性`actionSubWindow_View`、`actionTabbed_View`、`actionCascade_View`和`actionTile_View`的四个菜单选项分别连接到四个方法`SubWindow_View`、`Tabbed_View`、`cascadeArrange`和`tileArrange`。因此，当用户选择子窗口视图菜单选项时，将调用`SubWindow_View`方法。在`SubWindow_View`方法中，通过将`0`常量值传递给`MdiArea`小部件的`setViewMode`方法来激活子窗口视图模式。子窗口视图显示带有窗口框架的子窗口。

类似地，当用户选择选项卡视图菜单选项时，将调用`Tabbed_View`方法。在`Tabbed_View`方法中，通过将`1`常量值传递给`MdiArea`小部件的`setViewMode`方法来激活选项卡视图模式。选项卡视图模式在选项卡栏中显示选项卡，单击选项卡时，将显示关联的子窗口。

选择级联视图菜单选项时，将调用`cascadeArrange`方法，该方法又调用`MdiArea`小部件的`cascadeSubWindows`方法以级联形式排列子窗口。

选择平铺视图菜单选项时，将调用`tileArrange`方法，该方法又调用`MdiArea`小部件的`tileSubWindows`方法以平铺形式排列子窗口。

运行应用程序时，子窗口最初以缩小模式出现在`MdiArea`小部件中，如下面的屏幕截图所示。您可以看到子窗口以及它们的标题和最小化、最大化和关闭按钮：

![](img/1ebe29d9-558d-47d0-b8f1-5a58f3fc8a95.png)

您可以拖动它们的边框到所需的大小。在 Windows 菜单中选择第一个窗口时，子窗口将变为活动状态；选择第二个窗口时，下一个子窗口将变为活动状态。活动子窗口显示为更亮的标题和边界。在下面的截图中，您可以注意到第二个子窗口是活动的。您可以拖动任何子窗口的边界来增加或减少其大小。您还可以最小化一个子窗口，并拖动另一个子窗口的边界以占据整个`MdiArea`小部件的整个宽度。如果在任何子窗口中选择最大化，它将占据`MdiArea`的所有空间，使其他子窗口不可见：

![](img/416296d6-bf06-45ad-a2d8-98efad13ec01.png)

在选择级联时，子窗口以级联模式排列，如下截图所示。如果在级联模式下最大化窗口，则顶部子窗口将占据整个`MdiArea`小部件，将其他子窗口隐藏在其后，如下截图所示：

![](img/0db471f4-3caa-4770-99c6-c1bfa496aadd.png)

在选择平铺按钮时，子窗口会展开并平铺。两个子窗口均等地扩展以覆盖整个工作区，如下截图所示：

![](img/a869236e-f6a6-4acb-acde-4a1c66edab7f.png)

在选择选项卡视图按钮时，`MdiArea`小部件将从子窗口视图更改为选项卡视图。您可以选择任何子窗口的选项卡使其处于活动状态，如下截图所示：

![](img/26183590-38c9-433f-8129-cdb27e2b27b0.png)

# 使用选项卡小部件显示信息的部分

在这个应用程序中，我们将制作一个小型购物车，它将在一个选项卡中显示某些待售产品；在用户从第一个选项卡中选择所需产品后，当用户选择第二个选项卡时，他们将被提示输入首选付款选项。第三个选项卡将要求用户输入交付产品的地址。

我们将使用选项卡小部件使我们能够选择并分块填写所需的信息，所以您一定想知道，选项卡小部件是什么？

当某些信息被分成小节，并且您希望为用户显示所需部分的信息时，您需要使用选项卡小部件。在选项卡小部件容器中，有许多选项卡，当用户选择任何选项卡时，将显示分配给该选项卡的信息。

# 如何做...

以下是逐步创建应用程序以使用选项卡显示信息的过程：

1.  让我们基于没有按钮的对话框模板创建一个新应用程序。

1.  将选项卡小部件拖放到表单上。当您将选项卡小部件拖放到对话框上时，它将显示两个默认选项卡按钮，标有 Tab1 和 Tab2，如下截图所示：

![](img/d0bb4b0d-397e-4ec6-9e3f-a24ae20c57a7.png)

1.  您可以向选项卡小部件添加更多选项卡按钮，并通过添加新的选项卡按钮删除现有按钮；右键单击任一选项卡按钮，然后从弹出的菜单中选择“插入页面”。您将看到两个子选项，当前页面之后和当前页面之前。

1.  选择“当前页面之后”子选项以在当前选项卡之后添加一个新选项卡。新选项卡将具有默认文本“页面”，您可以随时更改。我们将要制作的应用程序包括以下三个选项卡：

+   第一个选项卡显示某些产品以及它们的价格。用户可以从第一个选项卡中选择任意数量的产品，然后单击“添加到购物车”按钮。

+   在选择第二个选项卡时，将显示所有付款选项。用户可以选择通过借记卡、信用卡、网上银行或货到付款进行付款。

+   第三个选项卡在选择时将提示用户输入交付地址：客户的完整地址以及州、国家和联系电话。

我们将首先更改选项卡的默认文本：

1.  使用选项卡小部件的 currentTabText 属性，更改每个选项卡按钮上显示的文本。

1.  将第一个选项卡按钮的文本属性设置为“产品列表”，将第二个选项卡按钮的文本属性设置为“付款方式”。

1.  要添加一个新的选项卡按钮，在“付款方式”选项卡上右键单击，并从出现的上下文菜单中选择“插入页面”。

1.  从出现的两个选项中，选择“当前页之后”和“当前页之前”，选择“当前页之后”以在“付款方式”选项卡之后添加一个新选项卡。新选项卡将具有默认文本“页面”。

1.  使用 currentTabText 属性，将其文本更改为“交付地址”。

1.  通过选择并拖动其节点来展开选项卡窗口，以在选项卡按钮下方提供空白空间，如下面的屏幕截图所示：

![](img/b6126880-8e88-4af4-8962-2e6c6189ab03.png)

1.  选择每个选项卡按钮，并将所需的小部件放入提供的空白空间。例如，将四个复选框小部件放到第一个选项卡按钮“产品列表”上，以显示可供销售的物品。

1.  在表单上放置一个推送按钮小部件。

1.  将四个复选框的文本属性更改为`手机$150`、`笔记本电脑$500`、`相机$250`和`鞋子$200`。

1.  将推送按钮小部件的文本属性更改为“添加到购物车”，如下面的屏幕截图所示：

![](img/b2d67f02-bf7b-429a-ab06-7126574542d3.png)

1.  类似地，要提供不同的付款方式，选择第二个选项卡，并在可用空间中放置四个单选按钮。

1.  将四个单选按钮的文本属性设置为“借记卡”、“信用卡”、“网上银行”和“货到付款”，如下面的屏幕截图所示：

![](img/4364f884-9029-4a33-8108-ea92a68f713a.png)

1.  选择第三个选项卡，然后拖放几个 LineEdit 小部件，提示用户提供交付地址。

1.  将六个 Label 和六个 LineEdit 小部件拖放到表单上。

1.  将 Label 小部件的文本属性设置为`地址 1`、`地址 2`、`州`、`国家`、`邮政编码`和`联系电话`。每个 Label 小部件前面的 LineEdit 小部件将用于获取交付地址，如下面的屏幕截图所示：

![](img/a7e83202-7705-4a75-be08-304e168d0f2a.png)

1.  将应用程序保存为`demoTabWidget.ui`。

1.  使用 Qt Designer 创建的用户界面存储在一个`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。要进行转换，需要打开命令提示符窗口，转到保存文件的文件夹，并发出此命令：

```py
C:PythonbookPyQt5>pyuic5 demoTabWidget.ui -o demoTabWidget.py
```

生成的 Python 脚本文件`demoTabWidget.py`的代码可以在本书的源代码包中找到。通过将其导入到另一个 Python 脚本中，使用自动生成的代码`demoTablWidget.py`创建的用户界面设计。

1.  创建另一个名为`callTabWidget.pyw`的 Python 文件，并将`demoTabWidget.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoTabWidget import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.show()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

如`callTabWidget.pyw`中所示，导入了必要的模块。创建了`MyForm`类，并继承自基类`QDialog`。调用了`QDialog`的默认构造函数。

通过`QApplication()`方法创建名为`app`的应用程序对象。每个 PyQt5 应用程序都必须创建一个应用程序对象。在创建应用程序对象时，将`sys.argv`参数传递给`QApplication()`方法。`sys.argv`参数包含来自命令行的参数列表，并有助于传递和控制脚本的启动属性。之后，使用`MyForm`类的实例创建名为`w`的实例。在实例上调用`show()`方法，将在屏幕上显示小部件。`sys.exit()`方法确保干净的退出，释放内存资源。

当应用程序执行时，您会发现默认情况下选择了第一个选项卡“产品列表”，并且该选项卡中指定的可供销售的产品如下屏幕截图所示：

![](img/e9c425e7-bfb5-4e0f-9260-a0fc9787747a.png)

同样，在选择其他选项卡“付款方式”和“交货地址”时，您将看到小部件提示用户选择所需的付款方式并输入交货地址。

# 创建自定义菜单栏

一个大型应用程序通常被分解为小的、独立的、可管理的模块。这些模块可以通过制作不同的工具栏按钮或菜单项来调用。也就是说，我们可以在单击菜单项时调用一个模块。我们在不同的软件包中看到了文件菜单、编辑菜单等，因此让我们学习如何制作自己的自定义菜单栏。

在本教程中，我们将学习创建显示特定菜单项的菜单栏。我们将学习如何添加菜单项，向菜单项添加子菜单项，在菜单项之间添加分隔符，向菜单项添加快捷键和工具提示，以及更多内容。我们还将学习如何向这些菜单项添加操作，以便单击任何菜单项时会执行某个操作。

我们的菜单栏将包括两个菜单，绘图和编辑。绘图菜单将包括四个菜单项，绘制圆形、绘制矩形、绘制直线和属性。属性菜单项将包括两个子菜单项，页面设置和设置密码。第二个菜单，编辑，将包括三个菜单项，剪切、复制和粘贴。让我们创建一个新应用程序，以了解如何实际创建这个菜单栏。

# 如何做…

我们将按照逐步程序来制作两个菜单，以及每个菜单中的相应菜单项。为了快速访问，每个菜单项也将与快捷键相关联。以下是创建我们自定义菜单栏的步骤：

1.  启动 Qt Designer 并创建一个基于 Main Window 模板的应用程序。

您会得到具有默认菜单栏的新应用程序，因为 Qt Designer 的 Main Window 模板默认提供了一个显示菜单栏的主应用程序窗口。默认菜单栏如下截图所示：

![](img/2e3d83c2-7891-4cdd-894e-b425ca161c42.png)

1.  您可以通过右键单击主窗口并从弹出的上下文菜单中选择“删除菜单栏”选项来删除默认菜单栏。

1.  您还可以通过从上下文菜单中选择“创建菜单栏”选项来稍后添加菜单栏。

默认菜单栏包含“在此处输入”占位符。您可以用菜单项文本替换它们。

1.  单击占位符以突出显示它，并输入以修改其文本。当您添加菜单项时，“在此处输入”将出现在新菜单项下方。

1.  再次，只需单击“在此处输入”占位符以选择它，然后简单地输入下一个菜单项的文本。

1.  您可以通过右键单击任何菜单项并从弹出的上下文菜单中选择“删除操作 action_name”选项来删除任何菜单项。

菜单栏中的菜单和菜单项可以通过拖放在所需位置进行排列。

在编写菜单或菜单项文本时，如果在任何字符之前添加一个`&`字符，菜单中的该字符将显示为下划线，并且将被视为快捷键。我们还将学习如何稍后为菜单项分配快捷键。

1.  当您通过替换“在此处输入”占位符创建新菜单项时，该菜单项将显示为操作编辑框中的单独操作，您可以从那里配置其属性。

回想一下，我们想在这个菜单栏中创建两个菜单，文本为“绘图”和“编辑”。绘图菜单将包含三个菜单项，绘制圆形、绘制矩形和绘制直线。在这三个菜单项之后，将插入一个分隔符，然后是一个名为“属性”的第四个菜单项。属性菜单项将包含两个子菜单项，页面设置和设置密码。编辑菜单将包含三个菜单项，剪切、复制和粘贴。

1.  双击“在此处输入”占位符，输入第一个菜单“绘图”的文本。

在“绘图”菜单上按下箭头键会弹出“在此处输入”和“添加分隔符”选项，如下截图所示：

![](img/ffd03bb2-b7a0-4bfd-b06f-8ff20c3fe644.png)

1.  双击“在此处输入”，并为“绘制”菜单下的第一个菜单项输入“绘制圆形”。在“绘制圆形”菜单上按下箭头键会再次提供“在此处输入”和“添加分隔符”选项。

1.  双击“在此处输入”并输入“绘制矩形”作为菜单项。

1.  按下下箭头键以获取两个选项，“在此处输入”和“添加分隔符”。

1.  双击“在此处输入”，并为第三个菜单项输入“绘制线条”。

1.  按下下箭头键后，再次会出现两个选项，“在此处输入”和“添加分隔符”，如下截图所示：

![](img/3c657aa6-7475-4351-b2ec-cf8ee426cad9.png)

1.  选择“添加分隔符”以在前三个菜单项后添加分隔符。

1.  在分隔符后按下下箭头键，并添加第四个菜单项“属性”。这是因为我们希望“属性”菜单项有两个子菜单项。

1.  选择右箭头以向“属性”菜单添加子菜单项。

1.  在任何菜单项上按下右箭头键，以向其添加子菜单项。在子菜单项中，选择“在此处输入”，并输入第一个子菜单“页面设置”。

1.  选择下箭头，并在页面设置子菜单项下输入“设置密码”，如下截图所示：

![](img/825e7797-8b3e-40b5-a874-80eaa7cb8b7b.png)

1.  第一个菜单“绘制”已完成。现在，我们需要添加另一个菜单“编辑”。选择“绘制”菜单，并按下右箭头键，表示要在菜单栏中添加第二个菜单。

1.  将“在此处输入”替换为“编辑”。

1.  按下下箭头，并添加三个菜单项，剪切、复制和粘贴，如下截图所示：

![](img/7a3b7048-2461-4c44-bc0c-f8fa84ae0c48.png)

所有菜单项的操作将自动显示在操作编辑框中，如下截图所示：

![](img/82d0ef70-87a2-4caf-b144-188354deaddd.png)

您可以看到操作名称是通过在每个菜单文本前缀文本操作并用下划线替换空格而生成的。这些操作可用于配置菜单项。

1.  要添加悬停在任何菜单项上时出现的工具提示消息，可以使用 ToolTip 属性。

1.  要为“绘制”菜单的“绘制圆形”菜单项分配工具提示消息，请在操作编辑框中选择 actionDraw_Circle，并将 ToolTip 属性设置为“绘制圆形”。类似地，您可以为所有菜单项分配工具提示消息。

1.  要为任何菜单项分配快捷键，请从操作编辑框中打开其操作，并单击快捷方式框内。

1.  在快捷方式框中，按下要分配给所选菜单项的键组合。

例如，如果在快捷方式框中按下*Ctrl* + *C*，则如下截图所示，Ctrl+C 将出现在框中：

![](img/6bb09b03-387a-4d7a-8c93-cf97e7cb1f01.png)

您可以使用任何组合的快捷键，例如*Shift* +键，*Alt* +键和*Ctrl* + *Shift* +键，用于任何菜单项。快捷键将自动显示在菜单栏中的菜单项中。您还可以使任何菜单项可选，即可以将其设置为切换菜单项。

1.  为此，选择所需菜单项的操作并勾选可选复选框。每个菜单项的操作，以及其操作名称、菜单文本、快捷键、可选状态和工具提示，都会显示在操作编辑框中。以下截图显示了“设置密码”子菜单项的操作，确认其快捷键为*Shift* + *P*，并且可以选择：

![](img/f59b87e8-e715-4468-955c-e06836bb1e81.png)

1.  对于“绘制圆形”、“绘制矩形”和“绘制线条”菜单项，我们将添加代码来分别绘制圆形、矩形和直线。

1.  对于其余的菜单项，我们希望当用户选择任何一个时，在表单上会出现一个文本消息，指示选择了哪个菜单项。

1.  要显示消息，请将标签小部件拖放到表单上。

1.  我们的菜单栏已完成；使用名称`demoMenuBar.ui`保存应用程序。

1.  我们使用`pyuic5`命令行实用程序将`.ui`（XML）文件转换为 Python 代码。

生成的 Python 代码`demoMenuBar.py`可以在本书的源代码包中找到。

1.  创建一个名为`callMenuBar.pyw`的 Python 脚本，导入之前的代码`demoMenuBar.py`，以调用菜单并在选择菜单项时显示带有 Label 小部件的文本消息。

您希望出现一条消息，指示选择了哪个菜单项。此外，当选择 Draw Circle、Draw Rectangle 和 Draw Line 菜单项时，您希望分别绘制一个圆、矩形和线。Python `callMenuBar.pyw`脚本中的代码将如下屏幕截图所示：

```py
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPainter

from demoMenuBar import *

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.pos1 = [0,0]
        self.pos2 = [0,0]
        self.toDraw=""
        self.ui.actionDraw_Circle.triggered.connect(self.
        drawCircle)
        self.ui.actionDraw_Rectangle.triggered.connect(self.
        drawRectangle)
        self.ui.actionDraw_Line.triggered.connect(self.drawLine)
        self.ui.actionPage_Setup.triggered.connect(self.pageSetup)
        self.ui.actionSet_Password.triggered.connect(self.
        setPassword)
        self.ui.actionCut.triggered.connect(self.cutMethod)
        self.ui.actionCopy.triggered.connect(self.copyMethod)
        self.ui.actionPaste.triggered.connect(self.pasteMethod)      
        self.show()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.toDraw=="rectangle":
            width = self.pos2[0]-self.pos1[0]
            height = self.pos2[1] - self.pos1[1]    
            qp.drawRect(self.pos1[0], self.pos1[1], width, height)
        if self.toDraw=="line":
            qp.drawLine(self.pos1[0], self.pos1[1], self.pos2[0], 
            self.pos2[1])
        if self.toDraw=="circle":
            width = self.pos2[0]-self.pos1[0]
            height = self.pos2[1] - self.pos1[1]           
            rect = QtCore.QRect(self.pos1[0], self.pos1[1], width,
            height)
            startAngle = 0
            arcLength = 360 *16
            qp.drawArc(rect, startAngle, 
            arcLength)     
        qp.end()

    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.pos1[0], self.pos1[1] = event.pos().x(), 
            event.pos().y()

    def mouseReleaseEvent(self, event):
        self.pos2[0], self.pos2[1] = event.pos().x(), 
        event.pos().y()   
        self.update()

    def drawCircle(self):
        self.ui.label.setText("")
        self.toDraw="circle"

    def drawRectangle(self):
        self.ui.label.setText("")
        self.toDraw="rectangle"

    def drawLine(self):
        self.ui.label.setText("")
        self.toDraw="line"

    def pageSetup(self):
        self.ui.label.setText("Page Setup menu item is selected")

    def setPassword(self):
        self.ui.label.setText("Set Password menu item is selected")

    def cutMethod(self):
        self.ui.label.setText("Cut menu item is selected")

    def copyMethod(self):
        self.ui.label.setText("Copy menu item is selected")

    def pasteMethod(self):
        self.ui.label.setText("Paste menu item is selected")

app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())
```

# 工作原理... 

每个菜单项的操作的 triggered()信号都连接到其相应的方法。每个菜单项的 triggered()信号都连接到`drawCircle()`方法，因此每当从菜单栏中选择 Draw Circle 菜单项时，都会调用`drawCircle()`方法。类似地，actionDraw_Rectangle 和 actionDraw_Line 菜单的 triggered()信号分别连接到`drawRectangle()`和`drawLine()`方法。在`drawCircle()`方法中，`toDraw`变量被分配一个字符串`circle`。`toDraw`变量将用于确定在`paintEvent`方法中要绘制的图形。`toDraw`变量可以分配三个字符串中的任何一个，即`line`、`circle`或`rectangle`。对`toDraw`变量中的值应用条件分支，相应地将调用绘制线条、矩形或圆的方法。图形将根据鼠标确定的大小进行绘制，即用户需要单击鼠标并拖动以确定图形的大小。

两种方法，`mousePressEvent()`和`mouseReleaseEvent()`，在按下和释放左鼠标按钮时会自动调用。为了存储按下和释放左鼠标按钮的位置的`x`和`y`坐标，使用了两个数组`pos1`和`pos2`。左鼠标按钮按下和释放的位置的`x`和`y`坐标值通过`mousePressEvent`和`mouseReleaseEvent`方法分配给`pos1`和`pos2`数组。

在`mouseReleaseEvent`方法中，分配鼠标释放位置的`x`和`y`坐标值后，调用`self.update`方法来调用`paintEvent()`方法。在`paintEvent()`方法中，基于分配给`toDraw`变量的字符串进行分支。如果`toDraw`变量分配了`line`字符串，`QPainter`类将通过`drawLine()`方法来绘制两个鼠标位置之间的线。类似地，如果`toDraw`变量分配了`circle`字符串，`QPainter`类将通过`drawArc()`方法来绘制直径由鼠标位置提供的圆。如果`toDraw`变量分配了`rectangle`字符串，`QPainter`类将通过`drawRect()`方法来绘制由鼠标位置提供的宽度和高度的矩形。

除了三个菜单项 Draw Circle、Draw Rectangle 和 Draw Line 之外，如果用户单击任何其他菜单项，将显示一条消息，指示用户单击的菜单项。因此，其余菜单项的 triggered()信号将连接到显示用户通过 Label 小部件选择的菜单项的消息信息的方法。

运行应用程序时，您会发现一个带有两个菜单 Draw 和 Edit 的菜单栏。Draw 菜单将显示四个菜单项 Draw Circle、Draw Rectangle、Draw Line 和 Properties，在 Properties 菜单项之前显示一个分隔符。Properties 菜单项显示两个子菜单项 Page Setup 和 Set Password，以及它们的快捷键，如下面的屏幕截图所示：

![](img/de30b61b-0eec-478b-9087-8742218d3ff2.png)

绘制一个圆，点击“绘制圆”菜单项，在窗体上的某个位置点击鼠标按钮，保持鼠标按钮按住，拖动以定义圆的直径。释放鼠标按钮时，将在鼠标按下和释放的位置之间绘制一个圆，如下截图所示：

![](img/84c641ca-4c5c-4551-adf5-3fda193607ee.png)

选择其他菜单项时，将显示一条消息，指示按下的菜单项。例如，选择“复制”菜单项时，将显示消息“选择了复制菜单项”，如下截图所示：

![](img/a080f906-4107-4d99-bc5b-3206026fb4fa.png)
