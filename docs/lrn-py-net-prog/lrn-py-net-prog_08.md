# 第八章：客户端和服务器应用程序

在上一章中，我们通过使用套接字接口来查看设备之间的数据交换。在本章中，我们将使用套接字来构建网络应用程序。套接字遵循计算机网络的主要模型之一，即**客户端/服务器**模型。我们将重点关注构建服务器应用程序。我们将涵盖以下主题：

+   设计一个简单的协议

+   构建回声服务器和客户端

+   构建聊天服务器和客户端

+   多线程和事件驱动的服务器架构

+   `eventlet`和`asyncio`库

本章的示例最好在 Linux 或 Unix 操作系统上运行。Windows 套接字实现有一些特殊之处，这可能会导致一些错误条件，我们在这里不会涉及。请注意，Windows 不支持我们将在一个示例中使用的`poll`接口。如果您使用 Windows，那么您可能需要使用*ctrl* + *break*来在控制台中终止这些进程，而不是使用*ctrl* - *c*，因为在 Windows 命令提示符中，当 Python 在套接字发送或接收时阻塞时，它不会响应*ctrl* - *c*，而在本章中这种情况会经常发生！（如果像我一样，不幸地尝试在没有*break*键的 Windows 笔记本上测试这些内容，那么请准备好熟悉 Windows 任务管理器的**结束任务**按钮）。

# 客户端和服务器

客户端/服务器模型中的基本设置是一个设备，即运行服务并耐心等待客户端连接并请求服务的服务器。一个 24 小时的杂货店可能是一个现实世界的类比。商店等待顾客进来，当他们进来时，他们请求某些产品，购买它们然后离开。商店可能会进行广告以便人们知道在哪里找到它，但实际的交易发生在顾客访问商店时。

一个典型的计算示例是一个 Web 服务器。服务器在 TCP 端口上监听需要其网页的客户端。例如，当客户端，例如 Web 浏览器，需要服务器托管的网页时，它连接到服务器然后请求该页面。服务器回复页面的内容，然后客户端断开连接。服务器通过具有主机名来进行广告，客户端可以使用该主机名来发现 IP 地址，以便连接到它。

在这两种情况下，都是客户端发起任何交互-服务器纯粹是对该交互的响应。因此，运行在客户端和服务器上的程序的需求是非常不同的。

客户端程序通常面向用户和服务之间的接口。它们检索和显示服务，并允许用户与之交互。服务器程序被编写为长时间运行，保持稳定，高效地向请求服务的客户端提供服务，并可能处理大量同时连接而对任何一个客户端的体验影响最小化。

在本章中，我们将通过编写一个简单的回声服务器和客户端来查看这个模型，然后将其升级为一个可以处理多个客户端会话的聊天服务器。Python 中的`socket`模块非常适合这项任务。

# 回声协议

在编写我们的第一个客户端和服务器程序之前，我们需要决定它们将如何相互交互，也就是说，我们需要为它们的通信设计一个协议。

我们的回声服务器应该保持监听，直到客户端连接并发送一个字节字符串，然后我们希望它将该字符串回显给客户端。我们只需要一些基本规则来做到这一点。这些规则如下：

1.  通信将通过 TCP 进行。

1.  客户端将通过创建套接字连接到服务器来启动回声会话。

1.  服务器将接受连接并监听客户端发送的字节字符串。

1.  客户端将向服务器发送一个字节字符串。

1.  一旦它发送了字节字符串，客户端将等待服务器的回复

1.  当服务器从客户端接收到字节字符串时，它将把字节字符串发送回客户端。

1.  当客户端从服务器接收了字节字符串后，它将关闭其套接字以结束会话。

这些步骤足够简单。这里缺少的元素是服务器和客户端如何知道何时发送了完整的消息。请记住，应用程序将 TCP 连接视为无尽的字节流，因此我们需要决定字节流中的什么将表示消息的结束。

## 框架

这个问题被称为**分帧**，我们可以采取几种方法来处理它。主要方法如下：

1.  将其作为协议规则，每次连接只发送一个消息，一旦发送了消息，发送方将立即关闭套接字。

1.  使用固定长度的消息。接收方将读取字节数，并知道它们有整个消息。

1.  在消息前加上消息的长度。接收方将首先从流中读取消息的长度，然后读取指示的字节数以获取消息的其余部分。

1.  使用特殊字符定界符指示消息的结束。接收方将扫描传入的流以查找定界符，并且消息包括定界符之前的所有内容。

选项 1 是非常简单协议的一个很好选择。它易于实现，不需要对接收到的流进行任何特殊处理。但是，它需要为每条消息建立和拆除套接字，当服务器同时处理多条消息时，这可能会影响性能。

选项 2 再次实现简单，但只有在我们的数据以整齐的固定长度块出现时才能有效利用网络。例如，在聊天服务器中，消息长度是可变的，因此我们将不得不使用特殊字符，例如空字节，来填充消息到块大小。这仅适用于我们确切知道填充字符永远不会出现在实际消息数据中的情况。还有一个额外的问题，即如何处理长于块长度的消息。

选项 3 通常被认为是最佳方法之一。虽然编码可能比其他选项更复杂，但实现仍然相当简单，并且它有效地利用了带宽。包括每条消息的长度所带来的开销通常与消息长度相比是微不足道的。它还避免了对接收到的数据进行任何额外处理的需要，这可能是选项 4 的某些实现所需要的。

选项 4 是最节省带宽的选项，当我们知道消息中只会使用有限的字符集，例如 ASCII 字母数字字符时，这是一个很好的选择。如果是这种情况，那么我们可以选择一个定界字符，例如空字节，它永远不会出现在消息数据中，然后当遇到这个字符时，接收到的数据可以很容易地被分成消息。实现通常比选项 3 简单。虽然可以将此方法用于任意数据，即定界符也可能出现为消息中的有效字符，但这需要使用字符转义，这需要对数据进行额外的处理。因此，在这些情况下，通常更简单的是使用长度前缀。

对于我们的回显和聊天应用程序，我们将使用 UTF-8 字符集发送消息。在 UTF-8 中，除了空字节本身，空字节在任何字符中都不使用，因此它是一个很好的分隔符。因此，我们将使用空字节作为定界符来对我们的消息进行分帧。

因此，我们的规则 8 将变为：

> *消息将使用 UTF-8 字符集进行编码传输，并以空字节终止。*

现在，让我们编写我们的回显程序。

# 一个简单的回显服务器

当我们在本章中工作时，我们会发现自己在重复使用几段代码，因此为了避免重复，我们将设置一个具有有用函数的模块，我们可以在以后重复使用。创建一个名为`tincanchat.py`的文件，并将以下代码保存在其中：

```py
import socket

HOST = ''
PORT = 4040

def create_listen_socket(host, port):
    """ Setup the sockets our server will receive connection requests on """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(100)
    return sock

def recv_msg(sock):
    """ Wait for data to arrive on the socket, then parse into messages using b'\0' as message delimiter """
    data = bytearray()
    msg = ''
    # Repeatedly read 4096 bytes off the socket, storing the bytes
    # in data until we see a delimiter
    while not msg:
        recvd = sock.recv(4096)
        if not recvd:
            # Socket has been closed prematurely
            raise ConnectionError()
        data = data + recvd
        if b'\0' in recvd:
            # we know from our protocol rules that we only send
            # one message per connection, so b'\0' will always be
            # the last character
            msg = data.rstrip(b'\0')
    msg = msg.decode('utf-8')
    return msg

def prep_msg(msg):
    """ Prepare a string to be sent as a message """
    msg += '\0'
    return msg.encode('utf-8')

def send_msg(sock, msg):
    """ Send a string over a socket, preparing it first """
    data = prep_msg(msg)
    sock.sendall(data)
```

首先，我们定义一个默认接口和要侦听的端口号。在`HOST`变量中指定的空的`''`接口告诉`socket.bind（）`侦听所有可用的接口。如果要将访问限制为仅限于您的计算机，则将代码开头的`HOST`变量的值更改为`127.0.0.1`。

我们将使用`create_listen_socket（）`来设置我们的服务器监听连接。这段代码对于我们的几个服务器程序是相同的，因此重复使用它是有意义的。

`recv_msg（）`函数将被我们的回显服务器和客户端用于从套接字接收消息。在我们的回显协议中，我们的程序在等待接收消息时不需要做任何事情，因此此函数只是在循环中调用`socket.recv（）`，直到接收到整个消息为止。根据我们的分帧规则，它将在每次迭代中检查累积的数据，以查看是否收到了空字节，如果是，则将返回接收到的数据，去掉空字节并解码为 UTF-8。

`send_msg（）`和`prep_msg（）`函数一起用于对消息进行分帧和发送。我们将空字节终止和 UTF-8 编码分离到`prep_msg（）`中，因为我们将在以后单独使用它们。

## 处理接收到的数据

请注意，就字符串编码而言，我们在发送和接收函数之间划定了一条谨慎的界限。Python 3 字符串是 Unicode，而我们通过网络接收的数据是字节。我们最不想做的最后一件事就是在程序的其余部分处理这些数据的混合，因此我们将在程序的边界处仔细编码和解码数据，数据进入和离开网络的地方。这将确保我们代码的其余部分可以假定它们将使用 Python 字符串，这将在以后为我们带来很多便利。

当然，并非我们可能想要通过网络发送或接收的所有数据都是文本。例如，图像、压缩文件和音乐无法解码为 Unicode 字符串，因此需要一种不同的处理方式。通常，这将涉及将数据加载到类中，例如**Python Image Library**（**PIL**）图像，如果我们要以某种方式操作对象。

在对接收到的数据进行完整处理之前，可以在此处对接收到的数据进行基本检查，以快速标记数据中的任何问题。此类检查的一些示例如下：

+   检查接收到的数据的长度

+   检查文件的前几个字节是否有魔术数字来确认文件类型

+   检查更高级别协议头的值，例如`HTTP`请求中的`Host`头

这种检查将允许我们的应用程序在出现明显问题时快速失败。

## 服务器本身

现在，让我们编写我们的回显服务器。打开一个名为`1.1-echo-server-uni.py`的新文件，并将以下代码保存在其中：

```py
import tincanchat

HOST = tincanchat.HOST
PORT = tincanchat.PORT

def handle_client(sock, addr):
    """ Receive data from the client via sock and echo it back """
    try:
        msg = tincanchat.recv_msg(sock)  # Blocks until received
                                         # complete message
        print('{}: {}'.format(addr, msg))
        tincanchat.send_msg(sock, msg)  # Blocks until sent
    except (ConnectionError, BrokenPipeError):
        print('Socket error')
    finally:
        print('Closed connection to {}'.format(addr))
        sock.close()

if __name__ == '__main__':
    listen_sock = tincanchat.create_listen_socket(HOST, PORT)
    addr = listen_sock.getsockname()
    print('Listening on {}'.format(addr))

    while True:
        client_sock, addr = listen_sock.accept()
        print('Connection from {}'.format(addr))
        handle_client(client_sock, addr)
```

这是一个服务器可以变得多么简单的例子！首先，我们使用`create_listen_socket（）`调用设置我们的监听套接字。其次，我们进入我们的主循环，在那里我们永远监听来自客户端的传入连接，阻塞在`listen_sock.accept（）`上。当客户端连接进来时，我们调用`handle_client（）`函数，根据我们的协议处理客户端。我们为此代码创建了一个单独的函数，部分原因是为了保持主循环的整洁，部分原因是因为我们将来会想要在后续程序中重用这组操作。

这就是我们的服务器，现在我们只需要创建一个客户端来与它通信。

# 一个简单的回显客户端

创建一个名为`1.2-echo_client-uni.py`的文件，并将以下代码保存在其中：

```py
import sys, socket
import tincanchat

HOST = sys.argv[-1] if len(sys.argv) > 1 else '127.0.0.1'
PORT = tincanchat.PORT

if __name__ == '__main__':
    while True:
        try:
            sock = socket.socket(socket.AF_INET,
                                 socket.SOCK_STREAM)
            sock.connect((HOST, PORT))
            print('\nConnected to {}:{}'.format(HOST, PORT))
            print("Type message, enter to send, 'q' to quit")
            msg = input()
            if msg == 'q': break
            tincanchat.send_msg(sock, msg)  # Blocks until sent
            print('Sent message: {}'.format(msg))
            msg = tincanchat.recv_msg(sock)  # Block until
                                             # received complete
                                             # message
            print('Received echo: ' + msg)
        except ConnectionError:
            print('Socket error')
            break
        finally:
            sock.close()
            print('Closed connection to server\n')
```

如果我们在与运行客户端的计算机不同的计算机上运行服务器，则可以将服务器的 IP 地址或主机名作为命令行参数提供给客户端程序。如果不这样做，它将默认尝试连接到本地主机。

代码的第三和第四行检查服务器地址的命令行参数。一旦确定要连接的服务器，我们进入我们的主循环，该循环将一直循环，直到我们通过输入`q`来终止客户端。在主循环中，我们首先创建与服务器的连接。其次，我们提示用户输入要发送的消息，然后使用`tincanchat.send_msg()`函数发送消息。然后我们等待服务器的回复。一旦收到回复，我们打印它，然后根据我们的协议关闭连接。

尝试运行我们的客户端和服务器。通过使用以下命令在终端中运行服务器：

```py
**$ python 1.1-echo_server-uni.py**
**Listening on ('0.0.0.0', 4040)**

```

在另一个终端中，运行客户端并注意，如果您需要连接到另一台计算机，您将需要指定服务器，如下所示：

```py
**$ python 1.2-echo_client.py 192.168.0.7**
**Type message, enter to send, 'q' to quit**

```

并排运行终端是一个好主意，因为您可以同时看到程序的行为。

在客户端中输入一些消息，看看服务器如何接收并将它们发送回来。与客户端断开连接也应该在服务器上提示通知。

# 并发 I/O

如果您有冒险精神，那么您可能已经尝试过同时使用多个客户端连接到我们的服务器。如果您尝试从它们中的两个发送消息，那么您会发现它并不像我们希望的那样工作。如果您还没有尝试过，请试一试。

客户端上的工作回显会话应该是这样的：

```py
**Type message, enter to send. 'q' to quit**
**hello world**
**Sent message: hello world**
**Received echo: hello world**
**Closed connection to server**

```

然而，当尝试使用第二个连接的客户端发送消息时，我们会看到类似这样的情况：

```py
**Type message, enter to send. 'q' to quit**
**hello world**
**Sent message: hello world**

```

当发送消息时，客户端将挂起，并且不会收到回显回复。您还可能注意到，如果我们使用第一个连接的客户端发送消息，那么第二个客户端将收到其响应。那么，这里发生了什么？

问题在于服务器一次只能监听来自一个客户端的消息。一旦第一个客户端连接，服务器就会在`tincanchat.recv_msg()`中的`socket.recv()`调用处阻塞，等待第一个客户端发送消息。在此期间，服务器无法接收其他客户端的消息，因此当另一个客户端发送消息时，该客户端也会阻塞，等待服务器发送回复。

这是一个稍微牵强的例子。在这种情况下，可以通过在建立与服务器的连接之前要求用户输入来轻松解决客户端端的问题。但是在我们完整的聊天服务中，客户端需要能够同时监听来自服务器的消息，同时等待用户输入。这在我们目前的程序设置中是不可能的。

解决这个问题有两种方法。我们可以使用多个线程或进程，或者使用**非阻塞**套接字以及**事件驱动**架构。我们将研究这两种方法，首先从**多线程**开始。

# 多线程和多进程

Python 具有允许我们编写多线程和多进程应用程序的 API。多线程和多进程背后的原则很简单，即复制我们的代码并在额外的线程或进程中运行它们。操作系统会自动调度可用 CPU 核心上的线程和进程，以提供公平的处理时间分配给所有线程和进程。这有效地允许程序同时运行多个操作。此外，当线程或进程阻塞时，例如等待 IO 时，操作系统可以将线程或进程降低优先级，并将 CPU 核心分配给其他有实际计算任务的线程或进程。

以下是线程和进程之间关系的概述：

![多线程和多进程](img/6008OS_08_01.jpg)

线程存在于进程内。 一个进程可以包含多个线程，但它始终至少包含一个线程，有时称为**主线程**。 同一进程中的线程共享内存，因此线程之间的数据传输只是引用共享对象的情况。 进程不共享内存，因此必须使用其他接口（如文件，套接字或专门分配的共享内存区域）来在进程之间传输数据。

当线程有操作要执行时，它们会请求操作系统线程调度程序为它们分配一些 CPU 时间，调度程序会根据各种参数（从 OS 到 OS 不等）将等待的线程分配给 CPU 核心。 同一进程中的线程可以同时在不同的 CPU 核心上运行。

尽管在前面的图中显示了两个进程，但这里并没有进行多进程处理，因为这些进程属于不同的应用程序。 显示第二个进程是为了说明 Python 线程和大多数其他程序中线程之间的一个关键区别。 这个区别就是 GIL 的存在。

## 线程和 GIL

CPython 解释器（可从[www.python.org](http://www.python.org)下载的 Python 标准版本）包含一个称为**全局解释器锁**（**GIL**）的东西。 GIL 的存在是为了确保在 Python 进程中只能同时运行一个线程，即使存在多个 CPU 核心。 有 GIL 的原因是它使 Python 解释器的底层 C 代码更容易编写和维护。 这样做的缺点是，使用多线程的 Python 程序无法利用多个核心进行并行计算。

这是一个引起很多争议的原因； 但是，对我们来说，这并不是一个大问题。 即使有 GIL 存在，仍然在 I/O 阻塞的线程被 OS 降低优先级并置于后台，因此有计算工作要做的线程可以运行。 以下图是这一点的简化说明：

![线程和全局解释器锁](img/6008OS_08_02.jpg)

**等待 GIL**状态是指线程已发送或接收了一些数据，因此准备退出阻塞状态，但另一个线程拥有 GIL，因此准备好的线程被迫等待。 在许多网络应用程序中，包括我们的回显和聊天服务器，等待 I/O 的时间远远高于处理数据的时间。 只要我们没有非常多的连接（这是我们在后面讨论事件驱动架构时会讨论的情况），由 GIL 引起的线程争用相对较低，因此线程仍然是这些网络服务器应用程序的合适架构。

考虑到这一点，我们将在我们的回显服务器中使用多线程而不是多进程。 共享数据模型将简化我们需要允许聊天客户端彼此交换消息的代码，并且因为我们是 I/O 绑定的，所以我们不需要进程进行并行计算。 在这种情况下不使用进程的另一个原因是，进程在 OS 资源方面更“笨重”，因此创建新进程比创建新线程需要更长的时间。 进程还使用更多内存。

需要注意的一点是，如果您需要在网络服务器应用程序中执行密集计算（也许您需要在将大型文件发送到网络之前对其进行压缩），那么您应该调查在单独的进程中运行此操作的方法。 由于 GIL 的实现中存在一些怪癖，即使在多个 CPU 核心可用时，将单个计算密集型线程放在主要是 I/O 绑定的进程中也会严重影响所有 I/O 绑定线程的性能。 有关更多详细信息，请查看以下信息框中链接到的 David Beazley 演示文稿：

### 注意

进程和线程是不同的东物，如果你对这些区别不清楚，值得阅读一下。一个很好的起点是维基百科关于线程的文章，可以在[`en.wikipedia.org/wiki/Thread_(computing)`](http://en.wikipedia.org/wiki/Thread_(computing))找到。

本主题的一个很好的概述在 Benjamin Erb 的论文*第四章*中给出，可以在[`berb.github.io/diploma-thesis/community/`](http://berb.github.io/diploma-thesis/community/)找到。

关于 GIL 的更多信息，包括保持它在 Python 中的原因，可以在官方 Python 文档中找到，网址为[`wiki.python.org/moin/GlobalInterpreterLock`](https://wiki.python.org/moin/GlobalInterpreterLock)。

您还可以在 Nick Coghlan 的 Python 3 问答中阅读更多关于这个主题的内容，网址为[`python-notes.curiousefficiency.org/en/latest/python3/questions_and_answers.html#but-but-surely-fixing-the-gil-is-more-important-than-fixing-unicode`](http://python-notes.curiousefficiency.org/en/latest/python3/questions_and_answers.html#but-but-surely-fixing-the-gil-is-more-important-than-fixing-unicode)。

最后，David Beazley 对多核系统上 GIL 的性能进行了一些引人入胜的研究。两个重要的演示资料可以在线找到。它们提供了一个与本章相关的很好的技术背景。这些可以在[`pyvideo.org/video/353/pycon-2010--understanding-the-python-gil---82`](http://pyvideo.org/video/353/pycon-2010--understanding-the-python-gil---82)和[`www.youtube.com/watch?v=5jbG7UKT1l4`](https://www.youtube.com/watch?v=5jbG7UKT1l4)找到。

# 多线程回显服务器

多线程方法的一个好处是操作系统为我们处理线程切换，这意味着我们可以继续以过程化的方式编写程序。因此，我们只需要对服务器程序进行小的调整，使其成为多线程，并因此能够同时处理多个客户端。

创建一个名为`1.3-echo_server-multi.py`的新文件，并将以下代码添加到其中：

```py
import threading
import tincanchat

HOST = tincanchat.HOST
PORT = tincanchat.PORT

def handle_client(sock, addr):
    """ Receive one message and echo it back to client, then close
        socket """
    try:
        msg = tincanchat.recv_msg(sock)  # blocks until received
                                         # complete message
        msg = '{}: {}'.format(addr, msg)
        print(msg)
        tincanchat.send_msg(sock, msg)  # blocks until sent
    except (ConnectionError, BrokenPipeError):
        print('Socket error')
    finally:
        print('Closed connection to {}'.format(addr))
        sock.close()

if __name__ == '__main__':
    listen_sock = tincanchat.create_listen_socket(HOST, PORT)
    addr = listen_sock.getsockname()
    print('Listening on {}'.format(addr))

    while True:
        client_sock,addr = listen_sock.accept()
        # Thread will run function handle_client() autonomously
        # and concurrently to this while loop
        thread = threading.Thread(target=handle_client,
                                  args=[client_sock, addr],
                                  daemon=True)
        thread.start()
        print('Connection from {}'.format(addr))
```

您可以看到，我们刚刚导入了一个额外的模块，并修改了我们的主循环，以在单独的线程中运行`handle_client()`函数，而不是在主线程中运行它。对于每个连接的客户端，我们创建一个新的线程，只运行`handle_client()`函数。当线程在接收或发送时阻塞时，操作系统会检查其他线程是否已经退出阻塞状态，如果有任何线程退出了阻塞状态，那么它就会切换到其中一个线程。

请注意，我们在线程构造函数调用中设置了`daemon`参数为`True`。这将允许程序在我们按下*ctrl* - *c*时退出，而无需我们显式关闭所有线程。

如果您尝试使用多个客户端进行此回显服务器，则会发现第二个连接并发送消息的客户端将立即收到响应。

# 设计聊天服务器

我们已经有一个工作的回显服务器，它可以同时处理多个客户端，所以我们离一个功能齐全的聊天客户端很近了。然而，我们的服务器需要将接收到的消息广播给所有连接的客户端。听起来很简单，但我们需要克服两个问题才能实现这一点。

首先，我们的协议需要进行改进。如果我们考虑从客户端的角度来看需要发生什么，那么我们就不能再依赖简单的工作流程：

客户端连接 > 客户端发送 > 服务器发送 > 客户端断开连接。

客户现在可能随时接收消息，而不仅仅是当他们自己向服务器发送消息时。

其次，我们需要修改我们的服务器，以便向所有连接的客户端发送消息。由于我们使用多个线程来处理客户端，这意味着我们需要在线程之间建立通信。通过这样做，我们正在涉足并发编程的世界，这需要谨慎和深思熟虑。虽然线程的共享状态很有用，但在其简单性中也是具有欺骗性的。有多个控制线程异步访问和更改相同资源是竞争条件和微妙死锁错误的理想滋生地。虽然关于并发编程的全面讨论远远超出了本文的范围，但我们将介绍一些简单的原则，这些原则可以帮助保持您的理智。

# 一个聊天协议

我们协议更新的主要目的是规定客户端必须能够接受发送给它们的所有消息，无论何时发送。

理论上，一个解决方案是让我们的客户端自己建立一个监听套接字，这样服务器在有新消息要传递时就可以连接到它。在现实世界中，这个解决方案很少适用。客户端几乎总是受到某种防火墙的保护，防止任何新的入站连接连接到客户端。为了让我们的服务器连接到客户端的端口，我们需要确保任何中间的防火墙都配置为允许我们的服务器连接。这个要求会让我们的软件对大多数用户不那么吸引，因为已经有一些不需要这样做的聊天解决方案了。

如果我们不能假设服务器能够连接到客户端，那么我们需要通过仅使用客户端发起的连接到服务器来满足我们的要求。我们可以通过两种方式来做到这一点。首先，我们可以让我们的客户端默认运行在断开状态，然后定期连接到服务器，下载任何等待的消息，然后再次断开连接。或者，我们可以让我们的客户端连接到服务器，然后保持连接打开。然后他们可以持续监听连接，并在一个线程中处理服务器发送的新消息，同时在另一个线程中接受用户输入并通过相同的连接发送消息。

您可能会认出这些情景，它们是一些电子邮件客户端中可用的**拉**和**推**选项。它们被称为拉和推，是因为操作对客户端的外观。客户端要么从服务器拉取数据，要么服务器向客户端推送数据。

使用这两种方法都有利有弊，决定取决于应用程序的需求。拉取会减少服务器的负载，但会增加客户端接收消息的延迟。虽然这对于许多应用程序来说是可以接受的，比如电子邮件，在聊天服务器中，我们通常希望立即更新。虽然我们可以频繁轮询，但这会给客户端、服务器和网络带来不必要的负载，因为连接会反复建立和拆除。

推送更适合聊天服务器。由于连接保持持续打开，网络流量的量仅限于初始连接设置和消息本身。此外，客户端几乎可以立即从服务器获取新消息。

因此，我们将使用推送方法，现在我们将编写我们的聊天协议如下：

1.  通信将通过 TCP 进行。

1.  客户端将通过创建套接字连接到服务器来启动聊天会话。

1.  服务器将接受连接，监听来自客户端的任何消息，并接受它们。

1.  客户端将在连接上监听来自服务器的任何消息，并接受它们。

1.  服务器将把来自客户端的任何消息发送给所有其他连接的客户端。

1.  消息将以 UTF-8 字符集进行编码传输，并以空字节终止。

# 处理持久连接上的数据

我们持久连接方法引发的一个新问题是，我们不能再假设我们的 `socket.recv()` 调用将只包含来自一个消息的数据。在我们的回显服务器中，由于我们已经定义了协议，我们知道一旦看到空字节，我们收到的消息就是完整的，并且发送者不会再发送任何内容。也就是说，我们在最后一个 `socket.recv()` 调用中读取的所有内容都是该消息的一部分。

在我们的新设置中，我们将重用同一连接来发送无限数量的消息，这些消息不会与我们从每个 `socket.recv()` 中提取的数据块同步。因此，很可能从一个 `recv()` 调用中获取的数据将包含多个消息的数据。例如，如果我们发送以下内容：

```py
caerphilly,
illchester,
brie
```

然后在传输中它们将如下所示：

```py
caerphilly**\0**illchester**\0**brie**\0**

```

然而，由于网络传输的变化，一系列连续的 `recv()` 调用可能会接收到它们：

```py
recv 1: caerphil
recv 2: ly**\0**illches
recv 3: ter**\0**brie**\0**

```

请注意，`recv 1` 和 `recv 2` 一起包含一个完整的消息，但它们也包含下一个消息的开头。显然，我们需要更新我们的解析。一种选择是逐字节从套接字中读取数据，也就是使用 `recv(1)`，并检查每个字节是否为空字节。然而，这是一种非常低效的使用网络套接字的方式。我们希望在调用 `recv()` 时尽可能多地读取数据。相反，当我们遇到不完整的消息时，我们可以缓存多余的字节，并在下次调用 `recv()` 时使用它们。让我们这样做，将这些函数添加到 `tincanchat.py` 文件中：

```py
def parse_recvd_data(data):
    """ Break up raw received data into messages, delimited
        by null byte """
    parts = data.split(b'\0')
    msgs = parts[:-1]
    rest = parts[-1]
    return (msgs, rest)

def recv_msgs(sock, data=bytes()):
    """ Receive data and break into complete messages on null byte
       delimiter. Block until at least one message received, then
       return received messages """
    msgs = []
    while not msgs:
        recvd = sock.recv(4096)
        if not recvd:
            raise ConnectionError()
        data = data + recvd
        (msgs, rest) = parse_recvd_data(data)
    msgs = [msg.decode('utf-8') for msg in msgs]
    return (msgs, rest)
```

从现在开始，我们将在以前使用 `recv_msg()` 的地方使用 `recv_msgs()`。那么，我们在这里做什么呢？通过快速浏览 `recv_msgs()`，您可以看到它与 `recv_msg()` 类似。我们重复调用 `recv()` 并像以前一样累积接收到的数据，但现在我们将使用 `parse_recvd_data()` 进行解析，期望它可能包含多个消息。当 `parse_recvd_data()` 在接收到的数据中找到一个或多个完整的消息时，它将将它们拆分成列表并返回它们，如果在最后一个完整消息之后还有任何剩余内容，则使用 `rest` 变量另外返回这些内容。然后，`recv_msgs()` 函数解码来自 UTF-8 的消息，并返回它们和 `rest` 变量。

`rest` 值很重要，因为我们将在下次调用 `recv_msgs()` 时将其返回，并且它将被添加到 `recv()` 调用的数据前缀。这样，上次 `recv_msgs()` 调用的剩余数据就不会丢失。

因此，在我们之前的例子中，解析消息将按照以下方式进行：

| `recv_msgs` 调用 | `data` 参数 | `recv` 结果 | 累积的 `data` | `msgs` | `rest` |
| --- | --- | --- | --- | --- | --- |
| 1 | - | `'caerphil'` | `'caerphil'` | `[]` | `b''` |
| 1 | - | `'ly\0illches'` | `'caerphilly\0illches'` | `['caerphilly']` | `'illches'` |
| 2 | `'illches'` | `'ter\0brie\0'` | `'illchester\0brie\0'` | `['illchester', 'brie']` | `b''` |

在这里，我们可以看到第一个 `recv_msgs()` 调用在其第一次迭代后没有返回。它循环是因为 `msgs` 仍然为空。这就是为什么 `recv_msgs` 调用编号为 1、1 和 2 的原因。

# 一个多线程聊天服务器

因此，让我们利用这一点并编写我们的聊天服务器。创建一个名为 `2.1-chat_server-multithread.py` 的新文件，并将以下代码放入其中：

```py
import threading, queue
import tincanchat

HOST = tincanchat.HOST
PORT = tincanchat.PORT

send_queues = {}
lock = threading.Lock()

def handle_client_recv(sock, addr):
    """ Receive messages from client and broadcast them to
        other clients until client disconnects """
    rest = bytes()
    while True:
        try:
            (msgs, rest) = tincanchat.recv_msgs(sock, rest)
        except (EOFError, ConnectionError):
            handle_disconnect(sock, addr)
            break
        for msg in msgs:
            msg = '{}: {}'.format(addr, msg)
            print(msg)
            broadcast_msg(msg)

def handle_client_send(sock, q, addr):
    """ Monitor queue for new messages, send them to client as
        they arrive """
    while True:
        msg = q.get()
        if msg == None: break
        try:
            tincanchat.send_msg(sock, msg)
        except (ConnectionError, BrokenPipe):
            handle_disconnect(sock, addr)
            break

def broadcast_msg(msg):
    """ Add message to each connected client's send queue """
    with lock:
        for q in send_queues.values():
            q.put(msg)

def handle_disconnect(sock, addr):
    """ Ensure queue is cleaned up and socket closed when a client
        disconnects """
    fd = sock.fileno()
    with lock:
        # Get send queue for this client
        q = send_queues.get(fd, None)
    # If we find a queue then this disconnect has not yet
    # been handled
    if q:
        q.put(None)
        del send_queues[fd]
        addr = sock.getpeername()
        print('Client {} disconnected'.format(addr))
        sock.close()

if __name__ == '__main__':
    listen_sock = tincanchat.create_listen_socket(HOST, PORT)
    addr = listen_sock.getsockname()
    print('Listening on {}'.format(addr))

    while True:
        client_sock,addr = listen_sock.accept()
        q = queue.Queue()
        with lock:
            send_queues[client_sock.fileno()] = q
        recv_thread = threading.Thread(target=handle_client_recv,
                                       args=[client_sock, addr],
                                       daemon=True)
        send_thread = threading.Thread(target=handle_client_send,
                                       args=[client_sock, q,
                                             addr],
                                       daemon=True)
        recv_thread.start()
        send_thread.start()
        print('Connection from {}'.format(addr))
```

现在我们为每个客户端使用两个线程。一个线程处理接收到的消息，另一个线程处理发送消息的任务。这里的想法是将可能发生阻塞的每个地方都分解成自己的线程。这将为每个客户端提供最低的延迟，但这会以系统资源为代价。我们减少了可能同时处理的客户端数量。我们可以使用其他模型，比如为每个客户端使用单个线程接收消息，然后自己将消息发送给所有连接的客户端，但我选择了优化延迟。

为了方便分开线程，我们将接收代码和发送代码分别放入`handle_client_recv()`函数和`handle_client_send()`函数中。

我们的`handle_client_recv`线程负责从客户端接收消息，而`handle_client_send`线程负责向客户端发送消息，但是接收到的消息如何从接收线程传递到发送线程呢？这就是`queue`、`send_queue`、`dict`和`lock`对象发挥作用的地方。

## 队列

`Queue`是一个**先进先出**（**FIFO**）管道。您可以使用`put()`方法向其中添加项目，并使用`get()`方法将它们取出。`Queue`对象的重要之处在于它们完全是**线程安全**的。在 Python 中，除非在其文档中明确指定，否则对象通常不是线程安全的。线程安全意味着对对象的操作保证是**原子**的，也就是说，它们将始终在没有其他线程可能到达该对象并对其执行意外操作的情况下完成。

等一下，你可能会问，之前，你不是说由于全局解释器锁（GIL），操作系统在任何给定时刻只能运行一个 Python 线程吗？如果是这样，那么两个线程如何能同时对一个对象执行操作呢？嗯，这是一个公平的问题。实际上，Python 中的大多数操作实际上由许多操作组成，这些操作是在操作系统级别进行的，线程是在操作系统级别进行调度的。一个线程可以开始对一个对象进行操作，比如向`list`中添加一个项目，当线程进行到操作系统级别的操作的一半时，操作系统可能会切换到另一个线程，这个线程也开始向同一个`list`中添加。由于`list`对象在被线程滥用时（它们不是线程安全的）没有对其行为提供任何保证，接下来可能发生任何事情，而且不太可能是一个有用的结果。这种情况可以称为**竞争条件**。

线程安全对象消除了这种可能性，因此在线程之间共享状态时，绝对应该优先选择它们。

回到我们的服务器，`Queues`的另一个有用的行为是，如果在空的`Queue`上调用`get()`，那么它将阻塞，直到有东西被添加到`Queue`中。我们利用这一点在我们的发送线程中。注意，我们进入一个无限循环，第一个操作是对`Queue`调用`get()`方法。线程将在那里阻塞并耐心等待，直到有东西被添加到它的`Queue`中。而且，你可能已经猜到了，我们的接收线程将消息添加到队列中。

我们为每个发送线程创建一个`Queue`对象，并将队列存储在`send_queues`字典中。为了广播新消息给我们的接收线程，它们只需要将消息添加到`send_queues`中的每个`Queue`中，这是在`broadcast_msgs()`函数中完成的。我们等待的发送线程将解除阻塞，从它们的`Queue`中取出消息，然后将其发送给它们的客户端。

我们还添加了一个`handle_disconnect()`函数，每当客户端断开连接或套接字发生错误时都会调用该函数。该函数确保与关闭连接相关的队列被清理，并且套接字从服务器端正确关闭。

## 锁

将我们对`Queues`对象的使用与我们对`send_queues`的使用进行对比。`Dict`对象不是线程安全的，不幸的是，在 Python 中没有线程安全的关联数组类型。由于我们需要共享这个`dict`，所以我们在访问它时需要额外小心，这就是`Lock`发挥作用的地方。`Lock`对象是一种**同步原语**。这些是具有功能的特殊对象，可以帮助管理我们的线程，并确保它们不会互相干扰。

`Lock`要么被锁定，要么被解锁。线程可以通过调用`acquire()`来锁定线程，或者像我们的程序一样，将其用作上下文管理器。如果一个线程已经获取了锁，另一个线程也试图获取锁，那么第二个线程将在`acquire()`调用上阻塞，直到第一个线程释放锁或退出上下文。一次可以有无限多个线程尝试获取锁 - 除了第一个之外，所有线程都会被阻塞。通过用锁包装对非线程安全对象的所有访问，我们可以确保没有两个线程同时操作该对象。

因此，每当我们向`send_queues`添加或删除内容时，我们都会将其包装在`Lock`上下文中。请注意，当我们迭代`send_queues`时，我们也在保护它。即使我们没有改变它，我们也希望确保在我们处理它时它不会被修改。

尽管我们很小心地使用锁和线程安全的原语，但我们并没有完全保护自己免受所有可能的与线程相关的问题。由于线程同步机制本身会阻塞，因此仍然很可能会出现死锁，即两个线程同时在由另一个线程锁定的对象上阻塞。管理线程通信的最佳方法是将对共享状态的所有访问限制在代码中尽可能小的区域内。在这个服务器的情况下，这个模块可以重新设计为提供最少数量的公共方法的类。它还可以被记录下来，以阻止任何内部状态的更改。这将使线程的这一部分严格限制在这个类中。

# 多线程聊天客户端

现在我们有了一个新的、全接收和广播的聊天服务器，我们只需要一个客户端。我们之前提到，当尝试同时监听网络数据和用户输入时，我们的过程化客户端会遇到问题。现在我们对如何使用线程有了一些想法，我们可以试着解决这个问题。创建一个名为`2.2-chat_client-multithread.py`的新文本文件，并将以下代码保存在其中：

```py
import sys, socket, threading
import tincanchat

HOST = sys.argv[-1] if len(sys.argv) > 1 else '127.0.0.1'
PORT = tincanchat.PORT

def handle_input(sock):
    """ Prompt user for message and send it to server """    
    print("Type messages, enter to send. 'q' to quit")
    while True:
        msg = input()  # Blocks
        if msg == 'q':
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            break
        try:
            tincanchat.send_msg(sock, msg)  # Blocks until sent
        except (BrokenPipeError, ConnectionError):
            break

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print('Connected to {}:{}'.format(HOST, PORT))

    # Create thread for handling user input and message sending
    thread = threading.Thread(target=handle_input,
                              args=[sock],
                              daemon=True)
    thread.start()
    rest = bytes()
    addr = sock.getsockname()
    # Loop indefinitely to receive messages from server
    while True:
        try:
            # blocks
            (msgs, rest) = tincanchat.recv_msgs(sock, rest)
            for msg in msgs:
                print(msg)
        except ConnectionError:
            print('Connection to server closed')
            sock.close()
            break
```

我们已经更新了我们的客户端，通过创建一个新线程来处理用户输入和发送消息，同时在主线程中处理接收消息，来遵守我们的新聊天协议。这允许客户端同时处理用户输入和接收消息。

请注意，这里没有共享状态，所以我们不必在`Queues`或同步原语上耍花招。

让我们来试试我们的新程序。启动多线程聊天服务器，然后启动至少两个客户端。如果可以的话，在终端中运行它们，这样你就可以同时观看它们。现在，尝试从客户端发送一些消息，看看它们是如何发送到所有其他客户端的。

# 事件驱动服务器

对于许多目的来说，线程是很好的，特别是因为我们仍然可以以熟悉的过程化、阻塞 IO 风格进行编程。但是它们的缺点是在同时管理大量连接时会遇到困难，因为它们需要为每个连接维护一个线程。每个线程都会消耗内存，并且在线程之间切换会产生一种称为**上下文切换**的 CPU 开销。虽然这对于少量线程来说不是问题，但是当需要管理许多线程时，它会影响性能。多进程也面临类似的问题。

使用**事件驱动**模型是线程和多进程的一种替代方法。在这种模型中，我们不是让操作系统自动在活动线程或进程之间切换，而是使用一个单线程，将阻塞对象（如套接字）注册到操作系统中。当这些对象准备好离开阻塞状态时，例如套接字接收到一些数据，操作系统会通知我们的程序；我们的程序可以以非阻塞模式访问这些对象，因为它知道它们处于立即可用的状态。在非阻塞模式下调用对象的调用总是立即返回。我们的应用程序围绕一个循环进行结构化，等待操作系统通知我们阻塞对象上的活动，然后处理该活动，然后回到等待状态。这个循环被称为**事件循环**。

这种方法提供了与线程和多进程相当的性能，但没有内存或上下文切换的开销，因此可以在相同的硬件上实现更大的扩展。工程应用程序能够有效处理大量同时连接的挑战在历史上被称为**c10k 问题**，指的是在单个线程中处理一万个并发连接。借助事件驱动架构，这个问题得到了解决，尽管这个术语在处理许多并发连接时仍经常被使用。

### 注意

在现代硬件上，使用多线程方法实际上可以处理一万个并发连接，也可以参考这个 Stack Overflow 问题来了解一些数字[`stackoverflow.com/questions/17593699/tcp-ip-solving-the-c10k-with-the-thread-per-client-approach`](https://stackoverflow.com/questions/17593699/tcp-ip-solving-the-c10k-with-the-thread-per-client-approach)。

现代挑战是“c10m 问题”，即一千万并发连接。解决这个问题涉及一些激进的软件甚至操作系统架构的变化。尽管这在短期内可能无法通过 Python 来解决，但可以在[`c10m.robertgraham.com/p/blog-page.html`](http://c10m.robertgraham.com/p/blog-page.html)找到有关该主题的有趣（尽管不幸是不完整的）概论。

下图显示了事件驱动服务器中进程和线程的关系：

![事件驱动服务器](img/6008OS_08_03.jpg)

尽管 GIL 和操作系统线程调度器在这里是为了完整性而显示的，但在事件驱动服务器的情况下，它们对性能没有影响，因为服务器只使用一个线程。I/O 处理的调度是由应用程序完成的。

# 低级事件驱动聊天服务器

因此，事件驱动架构有一些很大的好处，但问题在于，对于低级实现，我们需要以完全不同的风格编写我们的代码。让我们编写一个事件驱动的聊天服务器来说明这一点。

请注意，这个例子在 Windows 上根本无法工作，因为 Windows 缺乏我们将在这里使用的`poll`接口。然而，Windows 支持一个名为`select`的旧接口，但它更慢，更复杂。我们稍后讨论的事件驱动框架会自动切换到`select`，如果我们在 Windows 上运行的话。

有一个称为`epoll`的`poll`的高性能替代品，它在 Linux 操作系统上可用，但它也更复杂，所以为了简单起见，我们将在这里坚持使用`poll`。同样，我们稍后讨论的框架会自动利用`epoll`。

最后，令人费解的是，Python 的`poll`接口位于一个名为`select`的模块中，因此我们将在程序中导入`select`。

创建一个名为`3.1-chat_server-poll.py`的文件，并将以下代码保存在其中：

```py
import select
import tincanchat
from types import SimpleNamespace
from collections import deque

HOST = tincanchat.HOST
PORT = tincanchat.PORT
clients = {}

def create_client(sock):
    """ Return an object representing a client """
    return SimpleNamespace(
                    sock=sock,
                    rest=bytes(),
                    send_queue=deque())

def broadcast_msg(msg):
    """ Add message to all connected clients' queues """
    data = tincanchat.prep_msg(msg)
    for client in clients.values():
        client.send_queue.append(data)
        poll.register(client.sock, select.POLLOUT)

if __name__ == '__main__':
    listen_sock = tincanchat.create_listen_socket(HOST, PORT)
    poll = select.poll()
    poll.register(listen_sock, select.POLLIN)
    addr = listen_sock.getsockname()
    print('Listening on {}'.format(addr))

    # This is the event loop. Loop indefinitely, processing events
    # on all sockets when they occur
    while True:
        # Iterate over all sockets with events
        for fd, event in poll.poll():
            # clear-up a closed socket
            if event & (select.POLLHUP | 
                        select.POLLERR |
                        select.POLLNVAL):
                poll.unregister(fd)
                del clients[fd]

            # Accept new connection, add client to clients dict
            elif fd == listen_sock.fileno():
                client_sock,addr = listen_sock.accept()
                client_sock.setblocking(False)
                fd = client_sock.fileno()
                clients[fd] = create_client(client_sock)
                poll.register(fd, select.POLLIN)
                print('Connection from {}'.format(addr))

            # Handle received data on socket
            elif event & select.POLLIN:
                client = clients[fd]
                addr = client.sock.getpeername()
                recvd = client.sock.recv(4096)
                if not recvd:
                    # the client state will get cleaned up in the
                    # next iteration of the event loop, as close()
                    # sets the socket to POLLNVAL
                    client.sock.close()
                    print('Client {} disconnected'.format(addr))
                    continue
                data = client.rest + recvd
                (msgs, client.rest) = \
                                tincanchat.parse_recvd_data(data)
                # If we have any messages, broadcast them to all
                # clients
                for msg in msgs:
                    msg = '{}: {}'.format(addr, msg)
                    print(msg)
                    broadcast_msg(msg)

            # Send message to ready client
            elif event & select.POLLOUT:
                client = clients[fd]
                data = client.send_queue.popleft()
                sent = client.sock.send(data)
                if sent < len(data):
                    client.sends.appendleft(data[sent:])
                if not client.send_queue:
                    poll.modify(client.sock, select.POLLIN)
```

这个程序的关键是我们在执行开始时创建的`poll`对象。这是一个用于内核`poll`服务的接口，它允许我们注册套接字，以便操作系统在它们准备好供我们使用时通知我们。

我们通过调用`poll.register()`方法注册套接字，将套接字作为参数与我们希望内核监视的活动类型一起传递。我们可以通过指定各种`select.POLL*`常量来监视几种条件。在这个程序中，我们使用`POLLIN`和`POLLOUT`来监视套接字何时准备好接收和发送数据。在我们的监听套接字上接受新的传入连接将被视为读取。

一旦套接字被注册到`poll`中，操作系统将监视它，并记录当套接字准备执行我们请求的活动时。当我们调用`poll.poll()`时，它返回一个列表，其中包含所有已准备好供我们使用的套接字。对于每个套接字，它还返回一个`event`标志，指示套接字的状态。我们可以使用此事件标志来判断我们是否可以从套接字读取（`POLLIN`事件）或向套接字写入（`POLLOUT`事件），或者是否发生了错误（`POLLHUP`，`POLLERR`，`POLLNVAL`事件）。

为了利用这一点，我们进入我们的事件循环，重复调用`poll.poll()`，迭代返回的准备好的对象，并根据它们的`event`标志对它们进行操作。

因为我们只在一个线程中运行，所以我们不需要在多线程服务器中使用的任何同步机制。我们只是使用一个常规的`dict`来跟踪我们的客户端。如果你以前没有遇到过，我们在`create_client()`函数中使用的`SimpleNamespace`对象只是一个创建带有`__dict__`的空对象的新习惯用法（这是必需的，因为`Object`实例没有`__dict__`，所以它们不会接受任意属性）。以前，我们可能会使用以下内容来给我们一个可以分配任意属性的对象：

```py
class Client:
  pass
client = Client()
```

Python 版本 3.3 及更高版本为我们提供了新的更明确的`SimpleNamespace`对象。

我们可以运行我们的多线程客户端与这个服务器进行通信。服务器仍然使用相同的网络协议，两个程序的架构不会影响通信。试一试，验证是否按预期工作。

这种编程风格，使用`poll`和非阻塞套接字，通常被称为**非阻塞**和**异步**，因为我们使用非阻塞模式的套接字，并且控制线程根据需要处理 I/O，而不是锁定到单个 I/O 通道直到完成。但是，你应该注意，我们的程序并不完全是非阻塞的，因为它仍然在`poll.poll()`调用上阻塞。在 I/O 绑定系统中，这几乎是不可避免的，因为当没有发生任何事情时，你必须等待 I/O 活动。

# 框架

正如你所看到的，使用这些较低级别的线程和`poll`API 编写服务器可能会相当复杂，特别是考虑到一些在生产系统中预期的事情，比如日志记录和全面的错误处理，由于简洁起见，我们的示例中没有包括。

许多人在我们之前遇到了这些问题，并且有几个库和框架可用于减少编写网络服务器的工作量。

# 基于 eventlet 的聊天服务器

`eventlet`库提供了一个高级 API，用于事件驱动编程，但它的风格模仿了我们在多线程服务器中使用的过程式阻塞 IO 风格。结果是，我们可以有效地采用多线程聊天服务器代码，对其进行一些小的修改，以使用`eventlet`，并立即获得事件驱动模型的好处！

`eventlet`库可在 PyPi 中找到，并且可以使用`pip`进行安装，如下所示：

```py
**$ pip install eventlet**
**Downloading/unpacking eventlet**

```

### 注意

如果`poll`不可用，`eventlet`库会自动退回到`select`，因此它将在 Windows 上正常运行。

安装完成后，创建一个名为`4.1-chat_server-eventlet.py`的新文件，并将以下代码保存在其中：

```py
import eventlet
import eventlet.queue as queue
import tincanchat

HOST = tincanchat.HOST
PORT = tincanchat.PORT
send_queues = {}

def handle_client_recv(sock, addr):
    """ Receive messages from client and broadcast them to
        other clients until client disconnects """
    rest = bytes()
    while True:
        try:
            (msgs, rest) = tincanchat.recv_msgs(sock)
        except (EOFError, ConnectionError):
            handle_disconnect(sock, addr)
            break
        for msg in msgs:
            msg = '{}: {}'.format(addr, msg)
            print(msg)
            broadcast_msg(msg)

def handle_client_send(sock, q, addr):
    """ Monitor queue for new messages, send them to client as
        they arrive """
    while True:
        msg = q.get()
        if msg == None: break
        try:
            tincanchat.send_msg(sock, msg)
        except (ConnectionError, BrokenPipe):
            handle_disconnect(sock, addr)
            break

def broadcast_msg(msg):
    """ Add message to each connected client's send queue """
    for q in send_queues.values():
        q.put(msg)

def handle_disconnect(sock, addr):
    """ Ensure queue is cleaned up and socket closed when a client
        disconnects """
    fd = sock.fileno()
    # Get send queue for this client
    q = send_queues.get(fd, None)
    # If we find a queue then this disconnect has not yet
    # been handled
    if q:
        q.put(None)
        del send_queues[fd]
        addr = sock.getpeername()
        print('Client {} disconnected'.format(addr))
        sock.close()

if __name__ == '__main__':
    server = eventlet.listen((HOST, PORT))
    addr = server.getsockname()
    print('Listening on {}'.format(addr))

    while True:
        client_sock,addr = server.accept()
        q = queue.Queue()
        send_queues[client_sock.fileno()] = q
        eventlet.spawn_n(handle_client_recv,
                         client_sock,
                         addr)
        eventlet.spawn_n(handle_client_send,
                         client_sock,
                         q,
                         addr)
        print('Connection from {}'.format(addr))
```

我们可以使用我们的多线程客户端进行测试，以确保它按预期工作。

正如你所看到的，它与我们的多线程服务器几乎完全相同，只是做了一些更改以使用`eventlet`。请注意，我们已经删除了同步代码和`send_queues`周围的`lock`。我们仍然使用队列，尽管它们是`eventlet`库的队列，因为我们希望保留`Queue.get()`的阻塞行为。

### 注意

在 eventlet 网站上有更多使用 eventlet 进行编程的示例，网址为[`eventlet.net/doc/examples.html`](http://eventlet.net/doc/examples.html)。

# 基于 asyncio 的聊天服务器

`asyncio`标准库模块是 Python 3.4 中的新功能，它是在标准库中围绕异步 I/O 引入一些标准化的努力。`asyncio`库使用基于协程的编程风格。它提供了一个强大的循环类，我们的程序可以将准备好的任务（称为协程）提交给它，以进行异步执行。事件循环处理任务的调度和性能优化，以处理阻塞 I/O 调用。

它内置支持基于套接字的网络，这使得构建基本服务器成为一项简单的任务。让我们看看如何做到这一点。创建一个名为`5.1-chat_server-asyncio.py`的新文件，并将以下代码保存在其中：

```py
import asyncio
import tincanchat

HOST = tincanchat.HOST
PORT = tincanchat.PORT
clients = []

class ChatServerProtocol(asyncio.Protocol):
  """ Each instance of class represents a client and the socket 
       connection to it. """

    def connection_made(self, transport):
        """ Called on instantiation, when new client connects """
           self.transport = transport
        self.addr = transport.get_extra_info('peername')
        self._rest = b''
        clients.append(self)
        print('Connection from {}'.format(self.addr))

    def data_received(self, data):
        """ Handle data as it's received. Broadcast complete
        messages to all other clients """
        data = self._rest + data
        (msgs, rest) = tincanchat.parse_recvd_data(data)
        self._rest = rest
        for msg in msgs:
            msg = msg.decode('utf-8')
            msg = '{}: {}'.format(self.addr, msg)
            print(msg)
            msg = tincanchat.prep_msg(msg)
            for client in clients:
                client.transport.write(msg)  # <-- non-blocking

    def connection_lost(self, ex):
        """ Called on client disconnect. Clean up client state """
        print('Client {} disconnected'.format(self.addr))
        clients.remove(self)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    # Create server and initialize on the event loop
    coroutine = loop.create_server(ChatServerProtocol,
                                  host=HOST,
                                  port=PORT)
    server = loop.run_until_complete(coroutine)
    # print listening socket info
    for socket in server.sockets:
        addr = socket.getsockname()
        print('Listening on {}'.format(addr))
    # Run the loop to process client connections
    loop.run_forever()
```

同样，我们可以使用我们的多线程客户端进行测试，以确保它按我们的预期工作。

让我们逐步了解代码，因为它与我们以前的服务器有很大不同。我们首先定义了服务器行为，它是`asyncio.Protocol`抽象类的子类。我们需要重写三个方法`connection_made()`、`data_received()`和`connection_lost()`。通过使用这个类，我们可以在事件循环上实例化一个新的服务器，它将监听一个套接字，并根据这三种方法的内容进行操作。我们在主要部分中使用`loop.create_server()`调用来执行这个实例化。

当新客户端连接到我们的套接字时，将调用`connection_made()`方法，这相当于`socket.accept()`接收到一个连接。它接收的`transport`参数是一个可写流对象，也就是一个`asyncio.WriteTransport`实例。我们将使用它向套接字写入数据，因此通过将其分配给`self.transport`属性来保留它。我们还通过使用`transport.get_extra_info('peername')`来获取客户端的主机和端口。这是传输的`socket.getpeername()`的等价物。然后我们设置一个`rest`属性来保存从`tincanchat.parse_recvd_data()`调用中剩下的数据，然后我们将我们的实例添加到全局的`clients`列表中，以便其他客户端可以向其进行广播。

`data_received()`方法是发生操作的地方。每次`Protocol`实例的套接字接收到任何数据时，都会调用此函数。这相当于`poll.poll()`返回`POLLIN`事件，然后我们在套接字上执行`recv()`。调用此方法时，将接收到的数据作为`data`参数传递给该方法，然后我们使用`tincanchat.parse_recvd_data()`进行解析，就像以前一样。

然后，我们遍历接收到的任何消息，并对每条消息，通过在客户端的传输对象上调用`write()`方法，将其发送到`clients`列表中的每个客户端。这里需要注意的重要一点是，`Transport.write()`调用是非阻塞的，因此会立即返回。发送只是被提交到事件循环中，以便很快安排完成。 

`connection_lost()`方法在客户端断开连接或连接丢失时被调用，这相当于`socket.recv()`返回一个空结果，或者一个`ConnectionError`。在这里，我们只是从`clients`全局列表中移除客户端。

在主模块代码中，我们获取一个事件循环，然后创建我们的`Protocol`服务器的实例。调用`loop.run_until_complete()`在事件循环上运行我们服务器的初始化阶段，设置监听套接字。然后我们调用`loop.run_forever()`，这将使我们的服务器开始监听传入的连接。

# 更多关于框架

在最后一个示例中，我打破了我们通常的过程形式，采用了面向对象的方法，原因有两个。首先，虽然可以使用`asyncio`编写纯过程风格的服务器，但这需要比我们在这里提供的更深入的理解协程。如果你感兴趣，可以阅读`asyncio`文档中的一个示例协程风格的回显服务器，网址为[`docs.python.org/3/library/asyncio-stream.html#asyncio-tcp-echo-server-streams`](https://docs.python.org/3/library/asyncio-stream.html#asyncio-tcp-echo-server-streams)。

第二个原因是，这种基于类的方法通常是在完整系统中更易管理的模型。

实际上，Python 3.4 中有一个名为`selectors`的新模块，它提供了一个基于`select`模块中 IO 原语快速构建面向对象服务器的 API（包括`poll`）。文档和示例可以在[`docs.python.org/3.4/library/selectors.html`](https://docs.python.org/3.4/library/selectors.html)中找到。

还有其他第三方事件驱动框架可用，流行的有 Tornado（[www.tornadoweb.org](http://www.tornadoweb.org)）和 circuits（[`github.com/circuits/circuits`](https://github.com/circuits/circuits)）。如果你打算为项目选择一个框架，这两个都值得进行比较。

此外，没有讨论 Python 异步 I/O 的内容是完整的，而没有提到 Twisted 框架。直到 Python 3 之前，这一直是任何严肃的异步 I/O 工作的首选解决方案。它是一个事件驱动引擎，支持大量的网络协议，性能良好，并且拥有庞大而活跃的社区。不幸的是，它还没有完全转向 Python 3（迁移进度可以在[`rawgit.com/mythmon/twisted-py3-graph/master/index.html`](https://rawgit.com/mythmon/twisted-py3-graph/master/index.html)中查看）。由于我们在本书中专注于 Python 3，我们决定不对其进行详细处理。然而，一旦它到达那里，Python 3 将拥有另一个非常强大的异步框架，这将值得你为你的项目进行调查。

# 推动我们的服务器前进

有许多事情可以做来改进我们的服务器。对于多线程系统，通常会有一种机制来限制同时使用的线程数量。这可以通过保持活动线程的计数并在超过阈值时立即关闭来自客户端的任何新传入连接来实现。

对于我们所有的服务器，我们还希望添加一个日志记录机制。我强烈推荐使用标准库`logging`模块，它的文档非常完整，包含了很多好的例子。如果你以前没有使用过，基本教程是一个很好的起点，可以在[`docs.python.org/3/howto/logging.html#logging-basic-tutorial`](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial)中找到。

我们还希望更全面地处理错误。由于我们的服务器意图是长时间运行并且最小干预，我们希望确保除了关键异常之外的任何情况都不会导致进程退出。我们还希望确保处理一个客户端时发生的错误不会影响其他已连接的客户端。

最后，聊天程序还有一些基本功能可能会很有趣：让用户输入一个名字，在其他客户端上显示他们的消息旁边；添加聊天室；以及在套接字连接中添加 TLS 加密以提供隐私和安全性。

# 总结

我们研究了如何在考虑诸如连接顺序、数据传输中的数据帧等方面开发网络协议，以及这些选择对客户端和服务器程序架构的影响。

我们通过编写一个简单的回显服务器并将其升级为多客户端聊天服务器，演示了网络服务器和客户端的不同架构，展示了多线程和事件驱动模型之间的差异。我们讨论了围绕线程和事件驱动架构的性能问题。最后，我们看了一下`eventlet`和`asyncio`框架，这些框架在使用事件驱动方法时可以极大地简化服务器编写的过程。

在本书的下一章和最后一章中，我们将探讨如何将本书的几个主题融合起来，用于编写服务器端的 Web 应用程序。
