# 第五章。高级 Cython

在整本书中，我们一直是在混合 C 和 Python。在本章中，我们将深入研究 C++ 和 Cython。随着 Cython C++ 的每一次发布，支持都得到了改善。这并不是说它现在还不能使用。在本章中，我们将涵盖以下主题：

+   使本地的 C++ 类可从 Python 调用。

+   包装 C++ 命名空间和模板

+   异常如何从 C++ 和 Python 中传播

+   C++ 的 new 和 del 关键字

+   运算符重载

+   Cython gil 和 nogil 关键字

我们将通过将一个网络服务器嵌入到一个玩具 C++ 消息服务器中来结束本章。

# Cython 和 C++

在所有绑定生成器中，Cython 与 C++ 的结合最为无缝。C++ 在编写其绑定时有一些复杂性，例如调用约定、模板和类。我发现这种异常处理是 Cython 的一个亮点，我们将查看每个示例。

## 命名空间

我首先介绍命名空间，因为 Cython 使用命名空间作为在模块中引用 C++ 代码的方式。考虑以下具有以下命名空间的 C++ 头文件：

```py
#ifndef __MY_HEADER_H__
#define __MY_HEADER_H__

namespace mynamespace {
….
}

#endif //__MY_HEADER_H__
```

你可以用 `cdef extern` 声明将其包装起来：

```py
cdef extern from "header.h" namespace "mynamespace":
    …
```

你现在可以在 Cython 中像通常对模块那样访问它：

```py
import cythonfile
cythonfile.mynamespace.attribute

```

只需使用命名空间，它就真的感觉像是一个 Python 模块。

## 类

我猜测，你大部分的 C++ 代码都是围绕使用类来编写的。作为一个面向对象的语言，Cython 可以无缝地处理这一点：

```py
#ifndef __MY_HEADER_H__
#define __MY_HEADER_H__

namespace mynamespace {
  void myFunc (void);

  class myClass {
  public:
    int x;
    void printMe (void);
  };
}

#endif //__MY_HEADER_H__
```

我们可以使用 Cython 的 `cppclass` 关键字。这个特殊的关键字允许你声明 C++ 类并直接与之交互，因此你不需要编写包装代码，这在大型项目中可能会非常繁琐且容易出错。使用之前的命名空间示例，我们将包装命名空间，然后是命名空间内的类：

```py
cdef extern from "myheader.h" namespace "mynamespace":
    void myFunc ()
    cppclass myClass:
        int x
        void printMe ()
```

这就像 C 类型一样简单。尽管现在，你有一个本地的 C++ 对象，这可以非常强大。

记住，Cython 只会关心 `public` 属性。由于封装了私有和受保护的方法，这些是调用者可以访问的唯一属性。不可能扩展 C++ 类。现在，你可以像处理 `cdef` 结构体一样处理这些。只需像以前一样使用 `.` 运算符来访问所有必要的属性。

# C++ 的 new 和 del 关键字

Cython 理解 C++ 的 `new` 关键字；所以，考虑你有一个 C++ 类：

```py
 class Car {
    int doors;
    int wheels;
  public:
 Car ();
 ~Car ();
    void printCar (void);
    void setWheels (int x) { wheels = x; };
    void setDoors (int x) { doors = x; };
  };
```

它在 Cython 中如下定义：

```py
cdef extern from "cppcode.h" namespace "mynamespace":
    cppclass Car:
 Car ()
        void printCar ()
        void setWheels (int)
        void setDoors (int)
```

注意，我们没有声明 `~Car` 析构函数，因为我们从未直接调用它。它不是一个显式可调用的公共成员；这就是为什么我们从未直接调用它，但 `delete` 会，编译器将确保在它将离开栈作用域时调用它。要在 Cython 代码中在堆上实例化原始 C++ 类，我们可以简单地运行以下代码：

```py
cdef Car * c = new Car ()
```

然后，你可以使用 Python 的 `del` 关键字在任何时候使用 `del` 删除对象：

```py
del c

```

你会发现析构函数的调用正如你所预期的那样：

```py
$ cd chapter5/cppalloc; make; ./test
Car constructor
Car has 3 doors and 4 wheels
Car destructor

```

我们也可以声明一个栈分配的对象，但它必须只有一个默认构造函数，如下所示：

```py
cdef Car c
```

在 Cython 中，使用此语法无法传递参数。但是，请注意，你不能在这个实例上使用`del`，否则你会得到以下错误：

```py
cpycode.pyx:13:6: Deletion of non-heap C++ object

```

## 异常

使用 C++异常处理，你可以感受到 Cython 在 C++代码中的无缝感。如果抛出了任何异常，例如内存分配，Cython 将处理这些异常并将它们转换为更有用的错误，你仍然会得到有效的 C++异常对象。Python 也会理解这些是否被捕获以及是否按需处理。此表为你提供了 Python 异常在 C++中映射的概览：

| C++ | Python |
| --- | --- |
| `bad_alloc` | `MemoryError` |
| `bad_cast` | `TypeError` |
| `domain_error` | `ValueError` |
| `invalid_argument` | `ValueError` |
| `ios_base::failure` | `IOError` |
| `out_of_range` | `IndexError` |
| `overflow_error` | `OverflowError` |
| `range_error` | `ArithmeticError` |
| `underflow_error` | `ArithmeticError` |
| 所有其他异常 | `RuntimeError` |

例如，考虑以下 C++代码。当调用`myFunc`函数时，它将简单地抛出一个异常。首先，我们使用以下内容定义一个异常：

```py
namespace mynamespace {
  class mycppexcept: public std::exception {
 virtual const char * what () const throw () {
 return "C++ exception happened";
    }
  };

  void myFunc (void) throw (mycppexcept);
}
```

现在，我们编写一个抛出异常的函数：

```py
void mynamespace::myFunc (void) throw (mynamespace::mycppexcept) {
  mynamespace::mycppexcept ex;
  cout << "About to throw an exception!" << endl;
 throw ex;
}
```

我们可以在 Cython 中使用以下方式调用它：

```py
cdef extern from "myheader.h" namespace "mynamespace":
    void myFunc () except +RuntimeError

```

当我们运行函数时，我们得到以下输出：

```py
>>> import cpycode
About to throw an exception!
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "cpycode.pyx", line 3, in init cpycode (cpycode.cpp:763)
 myFunc ()
RuntimeError: C++ exception happened
>>> ^D

```

如果你想在 Python 代码中捕获 C++异常，你可以像平常一样使用它：

```py
try:
...
except RuntimeError:
...
```

注意，我们告诉 Cython 将任何异常转换为`RuntimeError`。这很重要，以确保你理解哪些接口和位置可能会抛出异常。未处理的异常看起来真的很糟糕，并且可能更难调试。在这个阶段，Cython 无法对状态做出太多假设，因为编译器在 C++代码级别上不会对可能未处理的异常抛出错误。如果发生这种情况，你将得到以下内容，因为没有准备好异常处理程序：

```py
$ cd chapter5/cppexceptions; make; python
Python 2.7.2 (default, Oct 11 2012, 20:14:37)
[GCC 4.2.1 Compatible Apple Clang 4.0 (tags/Apple/clang-418.0.60)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cpycode
About to throw an exception!
Segmentation fault: 11

```

## 布尔类型

如前一章所示，要使用 C++的本地`bool`类型，你首先需要导入以下内容：

```py
from libcpp cimport bool

```

然后，你可以使用`bool`作为正常的`cdef`。如果你想使用纯 PyObject `bool`类型，你需要导入以下内容：

```py
from cpython cimport bool

```

你可以使用正常的`true`或`false`值来分配它们。

# 重载

由于 Python 支持重载以包装 C++重载，只需按正常方式列出成员即可：

```py
cdef foobar (int)
cdef foobar (int, int)
…
```

Cython 理解我们处于 C++模式，并且可以像正常一样处理所有类型转换。有趣的是，它还可以轻松处理运算符重载，因为它只是另一个钩子！例如，让我们再次以`Car`类为例，执行一些如下的运算符重载操作：

```py
namespace mynamespace {
  class Car {
    int doors;
    int wheels;
  public:
    Car ();
    ~Car ();
 Car * operator+(Car *);
    void printCar (void);
    void setWheels (int x) { wheels = x; };
    void setDoors (int x) { doors = x; };
  };
};
```

记得将这些运算符重载类成员添加到你的 Cython 化类中；否则，你的 Cython 将抛出以下错误：

```py
Invalid operand types for '+' (Car *; Car *)

```

运算符重载的 Cython 声明看起来正如你所期望的那样：

```py
cdef extern from "cppcode.h" namespace "mynamespace":
    cppclass Car:
        Car ()
 Car * operator+ (Car *)
        void printCar ()
        void setWheels (int)
        void setDoors (int)
```

现在，你可以执行以下操作：

```py
cdef Car * ccc = c[0] + cc
ccc.printCar ()
```

这将在命令行上给我们以下输出：

```py
$ cd chapter5/cppoverloading; make; ./test
Car constructor
Car constructor
Car has 3 doors and 4 wheels
Car has 6 doors and 8 wheels
inside operator +
Car constructor
Car has 9 doors and 12 wheels

```

所有的处理方式都如你所预期。对我来说，这证明了 Guido 设计 Python 类所受到的启发原则。

# 模板

Cython 支持模板。尽管如此，为了完整性，模板元编程模式可能无法正确包装或无法编译。随着每个版本的发布，这都在不断改进，所以请带着一点点盐来接受这个评论。

C++ 类模板工作得非常好；我们可以实现一个名为 `LinkedList` 的模板，如下所示：

```py
cppclass LinkedList[T]:
        LinkedList ()
        void append (T)
        int getLength ()
...
```

现在，你可以使用名为 `T` 的声明来访问模板类型。你可以继续阅读 `chapter5/cpptemplates` 中的其余代码。

# 静态类成员属性

有时，在类中，拥有一个静态属性，如以下内容，是有用的：

```py
namespace mynamespace {
  class myClass {
    public:
 static void myStaticMethod (void);
  };
}
```

在 Cython 中，没有通过 `static` 关键字支持这一点，但你可以将这个函数绑定到一个命名空间上，使其成为以下内容：

```py
cdef extern from "header.h" namespace "mynamespace::myClass":
    void myStaticMethod ()

```

现在，你只需在 Cython 中将其作为全局方法调用即可。

# 调用 C++ 函数 - 注意事项

当你编写代码从 C 调用 C++ 函数时，你需要将原型包装在以下内容中：

```py
extern "C" { … }
```

这允许你调用 C++ 原型，因为 C 无法理解 C++ 类。使用 Cython 时，如果你要让你的 C 输出调用 C++ 函数，你需要小心选择编译器，或者你需要编写一个新的头文件来实现所需的包装函数，以便调用 C++。

## 命名空间 - 注意事项

Cython 似乎通常需要一个命名空间来保持嵌套，你可能已经在你的 C++ 代码中这样做。在非命名空间代码上创建 PXD 似乎会创建新的声明，这意味着你将因为多个符号而得到链接错误。从这些模板来看，C++ 的支持看起来非常好，并且更多的元编程习惯用法在 Cython 中可能难以表达。当多态发挥作用时，跟踪编译错误可能很困难。我强调，你应该尽可能保持你的接口简单，以便进行调试和更动态地操作！

### 小贴士

记住，当使用 Cython 生成 C++ 时，你需要指定 `–cplus`，这样它将默认输出 `cythonfile.cpp`。注意扩展名；我更喜欢使用 `.cc` 作为我的 C++ 代码，所以请确保你的构建系统正确无误。

# Python distutils

与往常一样，我们也可以使用 Python `distutils`，但你需要指定语言，这样所需的辅助 C++ 代码才会由正确的编译器编译：

```py
from distutils.core import setup
from Cython.Build import cythonize

setup (ext_modules = cythonize(
    "mycython.pyx",
    sources = ["mysource.cc"],
 language = "c++",
))
```

现在，你可以将你的 C++ 代码编译成 Python 模块。

# Python 线程和 GIL

**GIL**代表**全局解释器锁**。这意味着当你将程序链接到`libpython.so`并使用它时，你实际上在你的代码中拥有整个 Python 解释器。这个存在的原因是为了使并发应用程序变得非常容易。在 Python 中，你可以有两个线程同时读写同一位置，Python 会自动为你处理所有这些；与 Java 不同，在 Java 中你需要指定 Python 中的所有内容都在 GIL 之下。在讨论 GIL 及其作用时，有两个需要考虑的事情——指令原子性和读写锁。

## 原子指令

记住，Cython 必然会生成 C 代码，使其看起来与任何你可以导入的 Python 模块相似。所以，在底层发生的事情是，它会生成所有代码来获取 GIL 的锁，以便在运行时操作 Python 对象。让我们考虑两种执行类型。首先，你有 C 栈，它以原子方式执行，正如你所期望的那样；它不关心线程之间的同步——这留给程序员来处理。另一种是 Python，它为我们做所有的同步。当你手动使用`Py_Initilize`将 Python 嵌入到你的应用程序中时，这处于 C 执行之下。当涉及到调用某些内容，比如`import sys`和`sys.uname`，在从 C 调用的 Cython 代码中，Python GIL 会调度并阻塞多个线程同时调用以保持安全。这使得编写多线程 Python 代码非常安全。任何同时写入同一位置的错误都可以发生并被正确处理，而不是需要在 C 中的关键部分使用**互斥锁**。

## 读写锁

读写锁很棒，因为在 Python 中，你很少需要关心数据上的信号量或互斥锁，除非你想要同步不同线程对资源的访问。最糟糕的情况是，你的程序可能会进入不一致的状态，但与 C/C++不同，你不会崩溃。任何对全局字典的读写操作都会按照你预期的 Python 方式处理。

# Cython 关键字

好的，那么这如何影响你，更重要的是，你的代码呢？了解你的代码应该如何以及将会以并发方式执行是很重要的。如果没有理解这一点，你的调试将会很困惑。有时候 GIL 会阻碍执行，导致从 Python 到 C 代码或反之的执行出现问题。Cython 允许我们通过`gil`和`nogil`关键字来控制 GIL，这通过为我们封装这个状态而变得简单得多：

| Cython | Python |
| --- | --- |
| 使用 gil | `PyGILState_Ensure ()` |
| 使用 nogil | `PyGILState_Release (state)` |

我发现用 Python 来考虑多线程更容易从阻塞和非阻塞执行的角度来思考。在下一个例子中，我们将检查将 Web 服务器嵌入到玩具消息服务器中所需的步骤。

# 消息服务器

消息服务器是高度并发的示例之一；比如说，我们想在其中嵌入一个 Web 服务器来显示连接到服务器的客户端列表。如果你查看 flask，你可以看到你可以在大约八行代码中轻松地拥有一个完整的 Web 容器。

消息服务器是异步的；因此，它在 C 代码中是基于回调的。然后，这些回调可以通过 Cython 调用 Python 的 roster 对象。然后，我们可以遍历 roster 字典以获取在线客户端，并简单地作为 Web 服务返回一些 JSON，非常容易地重用 Python 代码，无需在 C/C++中编写任何内容。

在嵌入 Web 服务器时，重要的是要注意它们启动了很多线程。调用启动 Web 服务器函数将阻塞，直到它退出，这意味着如果我们首先启动 Web 服务器，消息服务器将不会并发运行。此外，由于 Web 服务器函数阻塞，如果我们在一个单独的线程上启动它，它将永远不会退出。因此，我们被迫在后台线程上运行消息服务器，我们可以从 Python 线程模块中这样做。再次强调，这是 GIL 状态变得重要的地方。如果我们用 GIL 运行消息服务器，当回调开始时，它们将崩溃或阻塞。我们可以将消息服务器包装在名为`MessageServer`的玩具类中：

```py
class MessageServer(threading.Thread):

    _port = None

    def __init__ (self, port):
        threading.Thread.__init__(self)
        # self.daemon = True
        self._port = port

    @property
    def roster(self):
        return _ROSTER

    @property
    def port(self):
        return self._port

    @staticmethod
    def set_callbacks():
        SetConnectCallback(pyconnect_callback)
        SetDisconnectCallback(pydisconnect_callback)
        SetReadCallback(pyread_callback)

    def stop(self):
        with nogil:
            StopServer();

    def run(self):
        logging.info("Starting Server on localhost:%i" % self.port)
        MessageServer.set_callbacks()
        cdef int cport = self.port
        with nogil:
            StartServer(cport)
        logging.info("Message Server Finished")

```

然后，正如你所期望的，我们可以通过运行以下代码来启动线程：

```py
   # start libevent server
    message_server = MessageServer(port)
    message_server.start()
```

注意，我指定了`with nogil`。我们的 C 代码不需要 GIL，因为我们只使用纯 C 类型，并且在回调之前不接触任何 Python 运行时。一旦`libevent`套接字服务器异步运行，我们就可以开始启动我们的 flask Web 服务器：

```py
from flask import Flask
from flask import jsonify

app = Flask("DashboardExample")
dashboard = None

@app.route("/")
def status():
    return jsonify(dashboard.roster.client_list())

class Dashboard:

    _port = None
    _roster = None

    def __init__(self, port, roster):
        global dashboard
        self._port = port
        self._roster = roster
        dashboard = self

    @property
    def port(self):
        return self._port

    @property
    def roster(self):
        return self._roster

    def start(self):
        app.run(port=self.port)
```

Flask 非常适合编写 RESTful Web 服务。它干净、简单，最重要的是，易于使用和阅读。此服务返回客户端 roster 的 JSON 表示。由于我已经封装了 roster 对象，我使用一个简单的全局变量，以便所有 flask 路由都可以查询正确的上下文：

```py
# start webserver
dashboard = Dashboard(port, roster)
dashboard.start()
```

Web 服务器现在会阻塞，直到收到终止信号。然后，它将返回，然后我们可以终止`MessageServer`：

```py
   # stop message server
message_server.stop()
```

现在，我们监听`server.cfg`中指定的端口：

```py
[MessageServer]
port = 8080
webport = 8081
```

此 roster 对象包含一个客户端列表并处理每个回调：

```py
class Roster:

    _clients = { }

    def handle_connect_event(self, client):
        """
        :returns True if client already exists else false
        """
        logging.info("connect: %s" % client)
        if client in self._clients:
            return True
        self._clients[client] = None
        return False;

    def handle_disconnect_event(self, client):
        logging.info("disconnect: %s" % client)
        self._clients.pop(client, None)

    def handle_read_event(self, client, message):
        logging.info("read: %s:[%s]" % (client, message))
        self._clients[client] = message

    def client_list(self):
        return self._clients
```

我们按照以下方式运行服务器：

```py
$ python server --config=config.cfg
```

然后，我们可以使用简单的 telnet 会话连接客户端：

```py
$ telnet localhost 8080
```

我们可以输入消息，在服务器日志中看到它被处理，然后按*Q*退出。然后，我们可以查询 Web 服务以获取客户端列表：

```py
$ curl -X GET localhost:8081
{
  "127.0.0.1": "Hello World"
}
```

# GIL 的注意事项

使用`gil`时有一个需要注意的注意事项。在我们的回调中，在调用任何 Python 代码之前，我们需要在每个回调中获取 GIL；否则，我们将发生段错误并感到非常困惑。所以，如果你查看调用 Cython 函数时的每个`libevent`回调，你会看到以下内容：

```py
 PyGILState_STATE gilstate_save = PyGILState_Ensure();
 readcb (client, (char *)data);
 PyGILState_Release(gilstate_save);
```

注意，这也在其他两个回调上调用——首先是在`discb`回调上：

```py
  PyGILState_STATE gilstate_save = PyGILState_Ensure();
 discb (client, NULL);
 PyGILState_Release(gilstate_save);
```

最后，在连接回调中，我们必须更加小心，并这样调用它：

```py
 PyGILState_STATE gilstate_save = PyGILState_Ensure();
  if (!conncb (NULL, inet_ntoa (client_addr.sin_addr)))
    {
…
    }
 else
    close (client_fd);
 PyGILState_Release(gilstate_save);

```

我们必须这样做，因为我们使用 Cython 的`nogil`执行了它。在我们返回 Python 领域之前，我们需要获取`gil`。你真的需要戴上你的创造力帽子，看看这样一些东西，并想象你能用它做什么。例如，你可以用它作为捕获数据的方式，并使用 Twisted Web 服务器实现嵌入式 RESTful 服务器。也许，你甚至可以使用 Python JSON 将数据包装成漂亮的对象。但是，它展示了如何使用 Python 库真正扩展一个相当复杂的 C 软件组件，使其既好又具有高级性质。这使一切都非常简单且易于维护，而不是从头开始尝试做所有事情。

# 本地代码的单元测试

Cython 的另一个用途是单元测试共享 C 库的核心功能。如果你维护一个`.pxd`文件（这实际上是你需要的全部），你可以编写自己的包装类，并使用 Python 的表达力进行数据结构的可伸缩性测试。例如，我们可以按照以下方式为`std::map`和`std::vector`编写单元测试：

```py
from libcpp.vector cimport vector

PASSED = False

cdef vector[int] vect
cdef int i
for i in range(10):
    vect.push_back(i)
for i in range(10):
    print vect[i]

PASSED = True
```

然后，按照以下方式为`map`编写测试：

```py
from libcpp.map cimport map

PASSED = False

cdef map[int,int] mymap
cdef int i
for i in range (10):
    mymap[i] = (i + 1)

for i in range (10):
    print mymap[i]

PASSED = True
```

然后，如果我们将它们编译成单独的模块，我们可以简单地编写一个测试执行器：

```py
#!/usr/bin/env python
print "Cython C++ Unit test executor"

print "[TEST] std::map"
import testmap
assert testmap.PASSED
print "[PASS]"

print "[TEST] std::vec"
import testvec
assert testvec.PASSED
print "[PASS]"

print "Done..."
```

这实际上是非常简单的代码，但它展示了这个想法。如果你添加了大量的断言和致命错误处理，你就可以对你的 C/C++代码进行一些非常棒的单元测试。我们可以更进一步，使用 Python 的本地单元测试框架来实现这一点。

# 防止子类化

如果你使用 Cython 创建了一个扩展类型，一个你永远不会希望被子类化的类型，它是一个被 Python 类包裹的`cpp`类。为了防止这种情况，你可以这样做：

```py
cimport cython

@cython.final
cdef class A: pass

cdef class B (A): pass
```

当有人尝试子类化时，这个注释将引发错误：

```py
pycode.pyx:7:5: Base class 'A' of type 'B' is final

```

注意，这些注释只适用于`cdef`或`cpdef`函数，而不适用于正常的 Python `def`函数。

# 解析大量数据

我想通过展示解析大量 XML 的差异来尝试证明 C 类型对程序员是多么强大和原生编译的。我们可以将政府的地域数据作为这个实验的测试数据（[`www.epa.gov/enviro/geospatial-data-download-service`](http://www.epa.gov/enviro/geospatial-data-download-service)）。

让我们看看这个 XML 数据的大小：

```py
 ls -liah
total 480184
7849156 drwxr-xr-x   5 redbrain  staff   170B 25 Jul 16:42 ./
5803438 drwxr-xr-x  11 redbrain  staff   374B 25 Jul 16:41 ../
7849208 -rw-r--r--@  1 redbrain  staff   222M  9 Mar 04:27 EPAXMLDownload.xml
7849030 -rw-r--r--@  1 redbrain  staff    12M 25 Jul 16:38 EPAXMLDownload.zip
7849174 -rw-r--r--   1 redbrain  staff    57B 25 Jul 16:42 README

```

它太大了！在我们编写程序之前，我们需要了解一些关于这些数据结构的信息，看看我们想用它做什么。它包含设施站点地址。这似乎是这里数据的大头，所以让我们尝试使用以下纯 Python XML 解析器解析它：

```py
from xml.etree import ElementTree as etree

```

代码使用`etree`通过以下方式解析 XML 文件：

```py
 xmlroot = etree.parse (__xmlFile)
```

然后，我们通过以下方式查找头文件和设施：

```py
headers = xmlroot.findall ('Header')
facs = xmlroot.findall ('FacilitySite')
```

最后，我们将它们输出到文件中：

```py
   try:
        fd = open (__output, "wb")
        for i in facs:
            location = ""
            for y in i:
                if isinstance (y.text, basestring):
                    location += y.tag + ": " + y.text + '\n'
            fd.write (location)
    # There is some dodgy unicode character
    # python doesn't like just ignore it
    except UnicodeEncodeError: pass
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise
    finally:
        if fd: fd.close ()
```

我们随后按照以下方式计时执行：

```py
10-4-5-52:bigData redbrain$ time python pyparse.py
USEPA Geospatial DataEnvironmental Protection AgencyUSEPA Geospatial DataThis XML file was produced by US EPA and contains data specifying the locations of EPA regulated facilities or cleanups that are being provided by EPA for use by commercial mapping services and others with an interest in using this information. Updates to this file are produced on a regular basis by EPA and those updates as well as documentation describing the contents of the file can be found at URL:http://www.epa.gov/enviro
MAR-08-2013
[INFO] Number of Facilties 118421
[INFO] Dumping facilities to xmlout.dat

real    2m21.936s
user    1m58.260s
sys     0m9.5800s

```

这段内容相当长，但让我们使用不同的 XML 实现来比较一下——Python 的 `lxml`。这是一个使用 Cython 实现的库，但它实现了与之前纯 Python XML 解析器相同的库：

```py
10-4-5-52:bigData redbrain$ sudo pip install lxml

```

我们可以简单地将在下面的替换导入：

```py
from lxml import etree

```

代码保持不变，但执行时间显著减少（通过运行 `make` 编译 Cython 版本，`cpyparse` 二进制文件是由相同的代码创建的，只是导入方式不同）：

```py
10-4-5-52:bigData redbrain$ time ./cpyparse
USEPA Geospatial DataEnvironmental Protection AgencyUSEPA Geospatial DataThis XML file was produced by US EPA and contains data specifying the locations of EPA regulated facilities or cleanups that are being provided by EPA for use by commercial mapping services and others with an interest in using this information. Updates to this file are produced on a regular basis by EPA and those updates as well as documentation describing the contents of the file can be found at URL:http://www.epa.gov/enviro
MAR-08-2013
[INFO] Number of Facilties 118421
[INFO] Dumping facilities to xmlout.dat

real    0m7.874s
user    0m5.307s
sys     0m1.839s

```

当你只付出一点努力时，你真的可以看到使用原生代码的强大之处。为了最终确保代码相同，让我们计算我们创建的 `xmlout.dat` 的 `MD5` 校验和：

```py
10-4-5-52:bigData redbrain$ md5 xmlout.dat xmlout.dat.cython
MD5 (xmlout.dat.python) = c2103a2252042f143489216b9c238283
MD5 (xmlout.dat.cython) = c2103a2252042f143489216b9c238283

```

因此，你可以看到输出完全相同，这样我们就知道没有发生任何奇怪的事情。这让人感到害怕，这可以使你的 XML 解析速度有多快；如果我们计算速度增加率，它大约快了 17.75 倍；但不要只听我的话；自己试一试。我的 MacBook 配有固态硬盘，有 4 GB 的 RAM，2 GHz 的 Core 2 Duo 处理器。

# 摘要

到目前为止，你已经看到了使用 Cython 可以实现的核心功能。在本章中，我们介绍了从 Cython 调用 C++ 类。你学习了如何封装模板，甚至查看了一个更复杂的应用，展示了 `gil` 和 `nogil` 的使用。

第六章，*进一步阅读* 是最后一章，将回顾一些关于 Cython 的最终注意事项和用法。我将展示如何使用 Cython 与 Python 3 结合。最后，我们将探讨相关项目和我在它们使用方面的观点。
