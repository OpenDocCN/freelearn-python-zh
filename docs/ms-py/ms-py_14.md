# 第十四章：C/C++扩展，系统调用和 C/C++库

现在我们对性能和多处理有了更多了解，我们将解释另一个至少与性能有关的主题——使用 C 和/或 C++扩展。

有多个原因需要考虑 C/C++扩展。拥有现有库可用是一个重要原因，但实际上，最重要的原因是性能。在第十二章中，*性能-跟踪和减少内存和 CPU 使用情况*，我们看到`cProfile`模块大约比`profile`模块快 10 倍，这表明至少一些 C 扩展比它们的纯 Python 等效快。然而，本章不会太注重性能。这里的目标是与非 Python 库的交互。任何性能改进只会是一个完全无意的副作用。

在本章中，我们将讨论以下选项：

+   用于处理 Python 中的外部（C/C++）函数和数据的 Ctypes

+   **CFFI**（**C Foreign Function Interface**的缩写），类似于`ctypes`但是有稍微不同的方法

+   使用本机 C/C++扩展 Python

# 介绍

在开始本章之前，重要的是要注意，本章将需要一个与你的 Python 解释器良好配合的工作编译器。不幸的是，这些因平台而异。虽然对于大多数 Linux 发行版来说通常很容易，但在 Windows 上可能是一个很大的挑战。对于 OS X 来说，通常很容易，只要你安装了正确的工具。

通用的构建说明始终可以在 Python 手册中找到：

[`docs.python.org/3.5/extending/building.html`](https://docs.python.org/3.5/extending/building.html)

## 你需要 C/C++模块吗？

在几乎所有情况下，我倾向于说你不需要 C/C++模块。如果你真的需要最佳性能，那么几乎总是有高度优化的库可用来满足你的目的。有一些情况下，需要本机 C/C++（或者只是“不是 Python”）。如果你需要直接与具有特定时间的硬件通信，那么 Python 可能对你来说行不通。然而，一般来说，这种通信应该留给负责特定时间的驱动程序。无论如何，即使你永远不会自己编写这些模块之一，当你调试项目时，你可能仍然需要知道它们的工作原理。

## Windows

对于 Windows，一般建议使用 Visual Studio。具体的版本取决于你的 Python 版本：

+   Python 3.2 及更低版本：Microsoft Visual Studio 2008

+   Python 3.3 和 3.4：Microsoft Visual Studio 2010

+   Python 3.5 和 3.6：Microsoft Visual Studio 2015

安装 Visual Studio 和编译 Python 模块的具体细节有点超出了本书的范围。幸运的是，Python 文档中有一些文档可以帮助你入门：

[`docs.python.org/3.5/extending/windows.html`](https://docs.python.org/3.5/extending/windows.html)

## OS X

对于 Mac，这个过程大多是直接的，但是有一些特定于 OS X 的技巧。

首先，通过 Mac App Store 安装 Xcode。一旦你这样做了，你应该能够运行以下命令：

```py
xcode-select --install

```

接下来是有趣的部分。因为 OS X 带有捆绑的 Python 版本（通常已过时），我建议通过 Homebrew 安装一个新的 Python 版本。安装 Homebrew 的最新说明可以在 Homebrew 主页上找到（[`brew.sh/`](http://brew.sh/)），但安装 Homebrew 的要点是这个命令：

```py
# /usr/bin/ruby -e "$(curl -fsSL \
 **https://raw.githubusercontent.com/Homebrew/install/master/install)"

```

之后，确保使用`doctor`命令检查一切是否设置正确：

```py
# brew doctor

```

当所有这些都完成时，只需通过 Homebrew 安装 Python，并确保在执行脚本时使用该 Python 版本：

```py
# brew install python3
# python3 –version
Python 3.5.1
which python3
/usr/local/bin/python3

```

还要确保 Python 进程在`/usr/local/bin`中，也就是自制版本。常规的 OS X 版本将在`/usr/bin/`中。

## Linux/Unix

Linux/Unix 系统的安装在很大程度上取决于发行版，但通常很简单。

对于使用`yum`作为软件包管理器的 Fedora、Red Hat、Centos 和其他系统，请使用以下命令：

```py
# sudo yum install yum-utils
# sudo yum-builddep python3

```

对于使用`apt`作为软件包管理器的 Debian、Ubuntu 和其他系统，请使用以下命令：

```py
# sudo apt-get build-dep python3.5

```

请注意，Python 3.5 并不是随处都可用的，所以您可能需要使用 Python 3.4。

### 提示

对于大多数系统，要获取安装帮助，可以通过类似`<操作系统> python.h`的网页搜索来解决问题。

# 使用 ctypes 调用 C/C++

`ctypes`库使得从 C 库调用函数变得非常容易，但您需要小心内存访问和数据类型。Python 在内存分配和类型转换方面通常非常宽容；C 则绝对不是那么宽容。

## 特定于平台的库

尽管所有平台都将在某个地方提供标准的 C 库，但其位置和调用方法因平台而异。为了拥有一个对大多数人来说易于访问的简单环境，我将假设使用 Ubuntu（虚拟）机器。如果您没有本机 Ubuntu 可用，您可以在 Windows、Linux 和 OS X 上通过 VirtualBox 轻松运行它。

由于您通常希望在本机系统上运行示例，我们将首先展示从标准 C 库中加载`printf`的基础知识。

### Windows

从 Python 调用 C 函数的一个问题是默认库是特定于平台的。虽然以下示例在 Windows 系统上可以正常运行，但在其他平台上则无法运行：

```py
>>> import ctypes
>>> ctypes.cdll
<ctypes.LibraryLoader object at 0x...>
>>> libc = ctypes.cdll.msvcrt
>>> libc
<CDLL 'msvcrt', handle ... at ...>
>>> libc.printf
<_FuncPtr object at 0x...>

```

由于这些限制，不是所有示例都可以在每个 Python 版本和发行版上工作，而不需要手动编译。从外部库调用函数的基本前提是简单地将它们的名称作为`ctypes`导入的属性来访问。然而，有一个区别；在 Windows 上，模块通常会自动加载，而在 Linux/Unix 系统上，您需要手动加载它们。

### Linux/Unix

从 Linux/Unix 调用标准系统库确实需要手动加载，但幸运的是这并不太复杂。从标准 C 库中获取`printf`函数非常简单：

```py
>>> import ctypes
>>> ctypes.cdll
<ctypes.LibraryLoader object at 0x...>
>>> libc = ctypes.cdll.LoadLibrary('libc.so.6')
>>> libc
<CDLL 'libc.so.6', handle ... at ...>
>>> libc.printf
<_FuncPtr object at 0x...>

```

### OS X

对于 OS X，也需要显式加载，但除此之外，它与常规 Linux/Unix 系统上的所有工作方式非常相似：

```py
>>> import ctypes
>>> libc = ctypes.cdll.LoadLibrary('libc.dylib')
>>> libc
<CDLL 'libc.dylib', handle ... at 0x...>
>>> libc.printf
<_FuncPtr object at 0x...>

```

### 使其变得简单

除了加载库的方式不同之外，还有更多的差异，但这些示例至少给出了标准的 C 库。它允许您直接从 C 实现中调用诸如`printf`之类的函数。如果由于某种原因，您在加载正确的库时遇到问题，总是可以使用`ctypes.util.find_library`函数。我始终建议显式声明而不是隐式声明，但使用这个函数可以使事情变得更容易。让我们在 OS X 系统上进行一次运行：

```py
>>> from ctypes import util
>>> from ctypes import cdll
>>> libc = cdll.LoadLibrary(util.find_library('libc'))
>>> libc
<CDLL '/usr/lib/libc.dylib', handle ... at 0x...>

```

## 调用函数和本机类型

通过`ctypes`调用函数几乎和调用本机 Python 函数一样简单。显著的区别在于参数和返回语句。这些应该转换为本机 C 变量：

### 注意

这些示例将假定您在前几段中的一个示例中已经将`libc`纳入了范围。

```py
>>> spam = ctypes.create_string_buffer(b'spam')
>>> ctypes.sizeof(spam)
5
>>> spam.raw
b'spam\x00'
>>> spam.value
b'spam'
>>> libc.printf(spam)
4
spam>>>

```

正如您所看到的，要调用`printf`函数，您*必须*——我无法再次强调这一点——将您的值从 Python 显式转换为 C。虽然最初可能看起来可以工作，但实际上并不行：

```py
>>> libc.printf(123)
segmentation fault (core dumped)  python3

```

### 注意

请记住使用第十一章中的`faulthandler`模块，*调试-解决错误*来调试段错误。

从这个例子中需要注意的另一件事是 `ctypes.sizeof(spam)` 返回 `5` 而不是 `4`。这是由 C 字符串所需的尾随空字符引起的。这在 C 字符串的原始属性中是可见的。如果没有它，`printf` 函数就不知道字符串在哪里结束。

要将其他类型（如整数）传递给 `libc` 函数，我们也必须进行一些转换。在某些情况下，这是可选的：

```py
>>> format_string = ctypes.create_string_buffer(b'Number: %d\n')
>>> libc.printf(format_string, 123)
Number: 123
12
>>> x = ctypes.c_int(123)
>>> libc.printf(format_string, x)
Number: 123
12

```

但并非所有情况都是如此，因此强烈建议您在所有情况下明确转换您的值：

```py
>>> format_string = ctypes.create_string_buffer(b'Number: %.3f\n')
>>> libc.printf(format_string, 123.45)
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
ctypes.ArgumentError: argument 2: <class 'TypeError'>: Don't know how to convert parameter 2
>>> x = ctypes.c_double(123.45)
>>> libc.printf(format_string, x)
Number: 123.450
16

```

重要的是要注意，即使这些值可以用作本机 C 类型，它们仍然可以通过 `value` 属性进行更改：

```py
>>> x = ctypes.c_double(123.45)
>>> x.value
123.45
>>> x.value = 456
>>> x
c_double(456.0)

```

然而，如果原始对象是不可变的，情况就不同了，这是一个非常重要的区别。`create_string_buffer` 对象创建一个可变的字符串对象，而 `c_wchar_p`、`c_char_p` 和 `c_void_p` 创建对实际 Python 字符串的引用。由于字符串在 Python 中是不可变的，这些值也是不可变的。你仍然可以更改 `value` 属性，但它只会分配一个新的字符串。实际上，将其中一个传递给会改变内部值的 C 函数会导致问题。

应该毫无问题地转换为 C 的唯一值是整数、字符串和字节，但我个人建议你始终转换所有的值，这样你就可以确定你将得到哪种类型以及如何处理它。

## 复杂的数据结构

我们已经看到，我们不能简单地将 Python 值传递给 C，但如果我们需要更复杂的对象呢？也就是说，不仅仅是直接可转换为 C 的裸值，而是包含多个值的复杂对象。幸运的是，我们可以很容易地使用 `ctypes` 创建（和访问）C 结构：

```py
>>> class Spam(ctypes.Structure):
...     _fields_ = [
...         ('spam', ctypes.c_int),
...         ('eggs', ctypes.c_double),
...     ]
...>>> spam = Spam(123, 456.789)
>>> spam.spam
123
>>> spam.eggs
456.789

```

## 数组

在 Python 中，我们通常使用列表来表示对象的集合。这些非常方便，因为你可以很容易地添加和删除值。在 C 中，默认的集合对象是数组，它只是一个具有固定大小的内存块。

以字节为单位的块的大小是通过将项数乘以类型的大小来决定的。在 `char` 的情况下，这是 `8` 位，所以如果你想存储 `100` 个字符，你将有 `100 * 8 位 = 800 位 = 100 字节`。

这实际上就是一个内存块，C 给你的唯一引用是指向内存块起始地址的指针。由于指针有类型，在这种情况下是 `char*`，C 就知道在尝试访问不同项时需要跳过多少字节。实际上，在尝试访问 `char` 数组中的第 25 项时，你只需要执行 `array_pointer + 25 * sizeof(char)`。这有一个方便的快捷方式：`array_pointer[25]`。

请注意，C 不会存储数组中的项数，因此即使我们的数组只有 100 项，我们也可以执行 `array_pointer[1000]` 并读取其他（随机）内存。

如果你考虑了所有这些，它绝对是可用的，但错误很快就会发生，而且 C 是不可原谅的。没有警告，只有崩溃和奇怪的行为代码。除此之外，让我们看看我们如何使用 `ctypes` 轻松地声明一个数组：

```py
>>> TenNumbers = 10 * ctypes.c_double
>>> numbers = TenNumbers()
>>> numbers[0]
0.0

```

正如你所看到的，由于固定的大小和在使用之前声明类型的要求，它的使用略显笨拙。然而，它确实像你期望的那样运行，并且这些值默认初始化为零。显然，这也可以与先前讨论的结构相结合：

```py
>>> Spams = 5 * Spam
>>> spams = Spams()
>>> spams[0].eggs = 123.456
>>> spams
<__main__.Spam_Array_5 object at 0x...>
>>> spams[0]
<__main__.Spam object at 0x...>
>>> spams[0].eggs
123.456
>>> spams[0].spam
0

```

尽管你不能简单地追加这些数组来调整它们的大小，但它们实际上是可调整大小的，有一些限制。首先，新数组的大小需要大于原始数组。其次，大小需要以字节为单位指定，而不是项数。举个例子，我们有这个例子：

```py
>>> TenNumbers = 10 * ctypes.c_double
>>> numbers = TenNumbers()
>>> ctypes.resize(numbers, 11 * ctypes.sizeof(ctypes.c_double))
>>> ctypes.resize(numbers, 10 * ctypes.sizeof(ctypes.c_double))
>>> ctypes.resize(numbers, 9 * ctypes.sizeof(ctypes.c_double))
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
ValueError: minimum size is 80
>>> numbers[:5] = range(5)
>>> numbers[:]
[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]

```

## 内存管理的注意事项

除了明显的内存分配问题和混合可变和不可变对象之外，还有一个奇怪的内存可变性问题：

```py
>>> class Point(ctypes.Structure):
...     _fields_ = ('x', ctypes.c_int), ('y', ctypes.c_int)
...
>>> class Vertex(ctypes.Structure):
...     _fields_ = ('a', Point), ('b', Point), ('c', Point)
...
>>> v = Vertex()
>>> v.a = Point(0, 1)
>>> v.b = Point(2, 3)
>>> v.c = Point(4, 5)
>>> v.a.x, v.a.y, v.b.x, v.b.y, v.c.x, v.c.y
(0, 1, 2, 3, 4, 5)
>>> v.a, v.b, v.c = v.b, v.c, v.a
>>> v.a.x, v.a.y, v.b.x, v.b.y, v.c.x, v.c.y
(2, 3, 4, 5, 2, 3)
>>> v.a.x = 123
>>> v.a.x, v.a.y, v.b.x, v.b.y, v.c.x, v.c.y
(123, 3, 4, 5, 2, 3)

```

为什么我们没有得到`2, 3, 4, 5, 0, 1`？问题在于这些对象被复制到一个临时缓冲变量中。与此同时，该对象的值正在发生变化，因为它在内部包含了单独的对象。之后，对象被传回，但值已经改变，导致了不正确的结果。

# CFFI

`CFFI`库提供了与`ctypes`非常相似的选项，但它更直接一些。与`ctypes`库不同，C 编译器对于`CFFI`来说确实是必需的。它带来了直接以非常简单的方式调用你的 C 编译器的机会：

```py
>>> import cffi
>>> ffi = cffi.FFI()
>>> ffi.cdef('int printf(const char* format, ...);')
>>> libc = ffi.dlopen(None)
>>> arg = ffi.new('char[]', b'spam')
>>> libc.printf(arg)
4
spam>>>

```

好吧...看起来有点奇怪对吧？我们不得不定义`printf`函数的外观，并用有效的 C 类型声明指定`printf`的参数。然而，回到声明，而不是`None`到`ffi.dlopen`，你也可以指定你希望加载的库。如果你记得`ctypes.util.find_library`函数，你可以在这种情况下再次使用它：

```py
>>> from ctypes import util
>>> import cffi
>>> libc = ffi.dlopen(util.find_library('libc'))
>>> ffi.printf
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
AttributeError: 'FFI' object has no attribute 'printf'

```

但它仍然不会为你提供其定义。函数定义仍然是必需的，以确保一切都按照你希望的方式工作。

## 复杂的数据结构

`CFFI`的定义与`ctypes`的定义有些相似，但不是让 Python 模拟 C，而是直接从 Python 访问纯 C。实际上，这只是一个小的语法差异。而`ctypes`是一个用于从 Python 访问 C 的库，同时尽可能接近 Python 语法，`CFFI`使用纯 C 语法来访问 C 系统，这实际上消除了一些对于熟悉 C 的人的困惑。我个人发现`CFFI`更容易使用，因为我知道实际发生了什么，而对于`ctypes`，我并不总是 100%确定。让我们用 CFFI 重复`Vertex`和`Point`的例子：

```py
>>> import cffi
>>> ffi = cffi.FFI()
>>> ffi.cdef('''
... typedef struct {
...     int x;
...     int y;
... } point;
...
... typedef struct {
...     point a;
...     point b;
...     point c;
... } vertex;
... ''')
>>> vertices = ffi.new('vertex[]', 5)
>>> v = vertices[0]
>>> v.a.x = 1
>>> v.a.y = 2
>>> v.b.x = 3
>>> v.b.y = 4
>>> v.c.x = 5
>>> v.c.y = 6
>>> v.a.x, v.a.y, v.b.x, v.b.y, v.c.x, v.c.y
(1, 2, 3, 4, 5, 6)
v.a, v.b, v.c = v.b, v.c, v.a
v.a.x, v.a.y, v.b.x, v.b.y, v.c.x, v.c.y
>>> v.a, v.b, v.c = v.b, v.c, v.a
>>> v.a.x, v.a.y, v.b.x, v.b.y, v.c.x, v.c.y
(3, 4, 5, 6, 3, 4)

```

你可以看到，可变变量问题仍然存在，但代码仍然是可以使用的。

## 数组

使用`CFFI`为新变量分配内存几乎是微不足道的。前面的段落向你展示了数组分配的一个例子；现在让我们看看数组定义的可能性：

```py
>>> import cffi
>>> ffi = cffi.FFI()
>>> x = ffi.new('int[10]')
>>> y = ffi.new('int[]', 10)
>>> x[0:10] = range(10)
>>> y[0:10] = range(10, 0, -1)
>>> list(x)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> list(y)
[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

```

在这种情况下，你可能会想知道为什么切片包括起始和结束。这实际上是`CFFI`的要求。并不总是有问题，但仍然有点烦人。然而，目前，这是不可避免的。

## ABI 还是 API？

像往常一样，还有一些注意事项——不幸的是。到目前为止的例子部分使用了 ABI，它从库中加载二进制结构。对于标准 C 库，这通常是安全的；对于其他库，通常不是。API 和 ABI 之间的区别在于后者在二进制级别调用函数，直接寻址内存，直接调用内存位置，并期望它们是函数。实际上，这是`ffi.dlopen`和`ffi.cdef`之间的区别。在这里，`dlopen`并不总是安全的，但`cdef`是安全的，因为它传递了一个编译器，而不仅仅是猜测如何调用一个方法。

## CFFI 还是 ctypes？

这实际上取决于你在寻找什么。如果你有一个 C 库，只需要调用而且不需要任何特殊的东西，那么`ctypes`很可能是更好的选择。如果你实际上正在编写自己的 C 库并尝试链接它，那么`CFFI`可能是一个更方便的选择。如果你不熟悉 C 编程语言，那么我肯定会推荐`ctypes`。或者，你会发现`CFFI`是一个更方便的选择。

# 本地 C/C++扩展

到目前为止，我们使用的库只是向我们展示了如何在我们的 Python 代码中访问 C/C++库。现在我们将看看故事的另一面——实际上是如何编写 Python 中的 C/C++函数/模块以及如何创建`cPickle`和`cProfile`等模块。

## 一个基本的例子

在我们实际开始编写和使用本地 C/C++扩展之前，我们有一些先决条件。首先，我们需要编译器和 Python 头文件；本章开头的说明应该已经为我们处理了这些。之后，我们需要告诉 Python 要编译什么。`setuptools`包大部分会处理这个问题，但我们确实需要创建一个`setup.py`文件：

```py
import setuptools

spam = setuptools.Extension('spam', sources=['spam.c'])

setuptools.setup(
    name='Spam',
    version='1.0',
    ext_modules=[spam],
)
```

这告诉 Python 我们有一个名为`Spam`的`Extension`对象，它将基于`spam.c`。

现在，让我们在 C 中编写一个函数，它将对给定数字之前的所有完全平方数（`2*2`，`3*3`等）进行求和。Python 代码将如下所示：

```py
def sum_of_squares(n):
    sum = 0

    for i in range(n):
        if i * i < n:
            sum += i * i
        else:
            break

    return sum
```

这段代码的原始 C 版本看起来像这样：

```py
long sum_of_squares(long n){
    long sum = 0;

    /* The actual summing code */
    for(int i=0; i<n; i++){
        if((i * i) < n){
            sum += i * i;
        }else{
            break;
        }
    }

    return sum;
}
```

Python C 版本看起来像这样：

```py
#include <Python.h>

static PyObject* spam_sum_of_squares(PyObject *self, PyObject
        *args){
    /* Declare the variables */
    int n;
    int sum = 0;

    /* Parse the arguments */
    if(!PyArg_ParseTuple(args, "i", &n)){
        return NULL;
    }

    /* The actual summing code */
    for(int i=0; i<n; i++){
        if((i * i) < n){
            sum += i * i;
        }else{
            break;
        }
    }

    /* Return the number but convert it to a Python object first
     */
    return PyLong_FromLong(sum);
}

static PyMethodDef spam_methods[] = {
    /* Register the function */
    {"sum_of_squares", spam_sum_of_squares, METH_VARARGS,
     "Sum the perfect squares below n"},
    /* Indicate the end of the list */
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef spam_module = {
    PyModuleDef_HEAD_INIT,
    "spam", /* Module name */
    NULL, /* Module documentation */
    -1, /* Module state, -1 means global. This parameter is
           for sub-interpreters */
    spam_methods,
};

/* Initialize the module */
PyMODINIT_FUNC PyInit_spam(void){
    return PyModule_Create(&spam_module);
}
```

看起来很复杂，但实际上并不难。在这种情况下，只是有很多额外的开销，因为我们只有一个函数。通常情况下，你会有几个函数，这种情况下你只需要扩展`spam_methods`数组并创建函数。下一段将更详细地解释代码，但首先让我们看一下如何运行我们的第一个示例。我们需要构建并安装模块：

```py
# python setup.py build install
running build
running build_ext
running install
running install_lib
running install_egg_info
Removing lib/python3.5/site-packages/Spam-1.0-py3.5.egg-info
Writing lib/python3.5/site-packages/Spam-1.0-py3.5.egg-info

```

现在，让我们创建一个小的测试脚本来测试 Python 版本和 C 版本之间的差异：

```py
import sys
import spam
import timeit

def sum_of_squares(n):
    sum = 0

    for i in range(n):
        if i * i < n:
            sum += i * i
        else:
            break

    return sum

if __name__ == '__main__':
    c = int(sys.argv[1])
    n = int(sys.argv[2])
    print('%d executions with n: %d' % (c, n))
    print('C sum of squares: %d took %.3f seconds' % (
        spam.sum_of_squares(n),
        timeit.timeit('spam.sum_of_squares(n)', number=c,
                      globals=globals()),
    ))
    print('Python sum of squares: %d took %.3f seconds' % (
        sum_of_squares(n),
        timeit.timeit('sum_of_squares(n)', number=c,
                      globals=globals()),
    ))
```

现在让我们执行它：

```py
# python3 test_spam.py 10000 1000000
10000 executions with n: 1000000
C sum of squares: 332833500 took 0.008 seconds
Python sum of squares: 332833500 took 1.778 seconds

```

太棒了！完全相同的结果，但速度快了 200 多倍！

## C 不是 Python-大小很重要

Python 语言使编程变得如此简单，以至于你有时可能会忘记底层数据结构；而在 C 中，你不能这样做。只需拿我们上一章的示例，但使用不同的参数：

```py
# python3 test_spam.py 1000 10000000
1000 executions with n: 10000000
C sum of squares: 1953214233 took 0.002 seconds
Python sum of squares: 10543148825 took 0.558 seconds

```

它仍然非常快，但数字发生了什么？Python 和 C 版本给出了不同的结果，`1953214233`与`10543148825`。这是由 C 中的整数溢出引起的。而 Python 数字基本上可以有任何大小，而 C 中，常规数字有固定的大小。你得到多少取决于你使用的类型（`int`，`long`等）和你的架构（32 位，64 位等），但这绝对是需要小心的事情。在某些情况下，它可能快上数百倍，但如果结果不正确，那就毫无意义了。

当然，我们可以稍微增加一点大小。这样会更好：

```py
static PyObject* spam_sum_of_squares(PyObject *self, PyObject *args){
    /* Declare the variables */
    unsigned long long int n;
    unsigned long long int sum = 0;

    /* Parse the arguments */
    if(!PyArg_ParseTuple(args, "K", &n)){
        return NULL;
    }

    /* The actual summing code */
    for(unsigned long long int i=0; i<n; i++){
        if((i * i) < n){
            sum += i * i;
        }else{
            break;
        }
    }

    /* Return the number but convert it to a Python object first */
    return PyLong_FromUnsignedLongLong(sum);
}
```

如果我们现在测试它，我们会发现它运行得很好：

```py
# python3 test_spam.py 1000 100000001000 executions with n: 10000000
C sum of squares: 10543148825 took 0.002 seconds
Python sum of squares: 10543148825 took 0.635 seconds

```

除非我们使数字更大：

```py
# python3 test_spam.py 1 100000000000000 ~/Dropbox/Mastering Python/code/h14
1 executions with n: 100000000000000
C sum of squares: 1291890006563070912 took 0.006 seconds
Python sum of squares: 333333283333335000000 took 2.081 seconds

```

那么你该如何解决这个问题呢？简单的答案是你不能。复杂的答案是，如果你使用不同的数据类型来存储你的数据，你是可以的。C 语言本身并没有 Python 所具有的“大数支持”。Python 通过在实际内存中组合几个常规数字来支持无限大的数字。在 C 中，没有常见的这种支持，因此没有简单的方法来使其工作。但我们可以检查错误：

```py
static unsigned long long int get_number_from_object(int* overflow, PyObject* some_very_large_number){
    return PyLong_AsLongLongAndOverflow(sum, overflow);
}
```

请注意，这仅适用于`PyObject*`，这意味着它不适用于内部 C 溢出。但你当然可以保留原始的 Python 长整型并对其执行操作。因此，你可以在 C 中轻松获得大数支持。

## 示例解释

我们已经看到了我们示例的结果，但如果你不熟悉 Python C API，你可能会对为什么函数参数看起来像这样感到困惑。`spam_sum_of_squares`中的基本计算与常规 C`sum_of_squares`函数是相同的，但有一些小的不同。首先，使用 Python C API 定义函数的类型应该看起来像这样：

```py
static PyObject* spam_sum_of_squares(PyObject *self, PyObject
 ***args)

```

### 静态

这意味着函数是`static`。静态函数只能从编译器内的同一翻译单元中调用。这实际上导致了一个函数，不能从其他模块链接，这允许编译器进一步优化。由于 C 中的函数默认是全局的，这可以非常有用地防止冲突。但为了确保，我们已经在函数名前加上了`spam_`前缀，以表明这个函数来自`spam`模块。

要小心，不要将此处的`static`与变量前面的`static`混淆。它们是完全不同的东西。`static`变量意味着该变量将存在于整个程序的运行时间，而不仅仅是函数的运行时间。

### PyObject*

`PyObject`类型是 Python 数据类型的基本类型，这意味着所有 Python 对象都可以转换为`PyObject*`（`PyObject`指针）。实际上，它只告诉编译器期望的属性类型，这些属性可以在以后用于类型识别和内存管理。而不是直接访问`PyObject*`，通常最好使用可用的宏，例如`Py_TYPE(some_object)`。在内部，这会扩展为`(((PyObject*)(o))->ob_type)`，这就是为什么宏通常是一个更好的主意。除了难以阅读之外，很容易出现拼写错误。

属性列表很长，且在很大程度上取决于对象的类型。对于这些，我想参考 Python 文档：

[`docs.python.org/3/c-api/typeobj.html`](https://docs.python.org/3/c-api/typeobj.html)

整个 Python C API 可以填满一本书，但幸运的是在 Python 手册中有很好的文档。然而，使用可能不太明显。

### 解析参数

使用常规的 C 和 Python，您需要明确指定参数，因为使用 C 处理可变大小的参数有点棘手。这是因为它们需要被单独解析。`PyObject* args`是包含实际值的对象的引用。要解析这些，您需要知道期望的变量数量和类型。在示例中，我们使用了`PyArg_ParseTuple`函数，它只解析位置参数，但很容易使用`PyArg_ParseTupleAndKeywords`或`PyArg_VaParseTupleAndKeywords`解析命名参数。最后两者之间的区别在于第一个使用可变数量的参数来指定目的地，而后者使用`va_list`来设置值。但首先，让我们分析一下实际示例中的代码：

```py
if(!PyArg_ParseTuple(args, "i", &n)){
    return NULL;
}
```

我们知道`args`是包含对实际参数的引用的对象。`"i"`是一个格式字符串，在这种情况下将尝试解析一个整数。`&n`告诉函数将值存储在`n`变量的内存地址。

格式字符串在这里是重要的部分。根据字符的不同，您会得到不同的数据类型，但有很多；`i`指定一个常规整数，`s`将您的变量转换为 c 字符串（实际上是一个`char*`，它是一个以空字符结尾的字符数组）。值得注意的是，这个函数很幸运地足够聪明，可以考虑到溢出。

解析多个参数非常类似；您只需要向格式字符串添加多个字符和多个目标变量：

```py
PyObject* callback;
int n;

/* Parse the arguments */
if(!PyArg_ParseTuple(args, "Oi", &callback, &n)){
    return NULL;
}
```

带有关键字参数的版本类似，但需要进行一些代码更改，因为方法列表需要被告知函数接受关键字参数。否则，`kwargs`参数将永远不会到达：

```py
static PyObject* function(
        PyObject *self,
        PyObject *args,
        PyObject *kwargs){
    /* Declare the variables */
    int sum = 0;

    PyObject* callback;
    int n;

    static char* keywords[] = {"callback", "n", NULL};

    /* Parse the arguments */
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keywords,
                &callback, &n)){
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    /* Register the function with kwargs */
    {"function", function, METH_VARARGS | METH_KEYWORDS,
     "Some kwargs function"},
    /* Indicate the end of the list */
    {NULL, NULL, 0, NULL},
};
```

请注意，这仍然支持普通参数，但现在也支持关键字参数。

## C 不是 Python-错误是沉默的或致命的

正如我们在前面的例子中看到的，整数溢出通常不容易注意到，而且不幸的是，没有很好的跨平台方法来捕获它们。然而，这些通常是更容易处理的错误；最糟糕的错误通常是内存管理。使用 Python，如果出现错误，您将得到一个可以捕获的异常。但是在 C 中，您实际上无法优雅地处理它。例如，以零除：

```py
# python3 -c '1/0'
Traceback (most recent call last):
 **File "<string>", line 1, in <module>
ZeroDivisionError: division by zero

```

这很容易通过`try: ... except ZeroDivisionError: ...`捕获。另一方面，对于 C 来说，如果出现严重错误，它将终止整个进程。但是，调试 C 代码是 C 编译器具有调试器的功能，为了找到错误的原因，您可以使用第十一章中讨论的`faulthandler`模块，*调试-解决错误*。现在，让我们看看如何可以正确地从 C 中抛出错误。让我们使用之前的`spam`模块，但为了简洁起见，我们将省略其余的 C 代码：

```py
static PyObject* spam_eggs(PyObject *self, PyObject *args){
    PyErr_SetString(PyExc_RuntimeError, "Too many eggs!");
    return NULL;
}

static PyMethodDef spam_methods[] = {
    /* Register the function */
    {"eggs", spam_eggs, METH_VARARGS,
     "Count the eggs"},
    /* Indicate the end of the list */
    {NULL, NULL, 0, NULL},
};
```

这是执行过程：

```py
# python3 setup.py clean build install
...
# python3 -c 'import spam; spam.eggs()'
Traceback (most recent call last):
 **File "<string>", line 1, in <module>
RuntimeError: Too many eggs!

```

语法略有不同——`PyErr_SetString`而不是`raise`——但基本原理是相同的，幸运的是。

## 从 C 调用 Python-处理复杂类型

我们已经看到如何从 Python 调用 C 函数，但现在让我们尝试从 C 返回 Python。我们将构建一个自己的回调函数，并处理任何类型的可迭代对象，而不是使用现成的`sum`函数。虽然这听起来足够简单，但实际上确实需要一些类型干涉，因为你只能期望`PyObject*`作为参数。这与简单类型相反，例如整数、字符和字符串，它们会立即转换为本机 Python 版本：

```py
static PyObject* spam_sum(PyObject* self, PyObject* args){
    /* Declare all variables, note that the values for sum and
     * callback are defaults in the case these arguments are not
     * specified */
    long long int sum = 0;
    int overflow = 0;
    PyObject* iterator;
    PyObject* iterable;
    PyObject* callback = NULL;
    PyObject* value;
    PyObject* item;

    /* Now we parse a PyObject* followed by, optionally
     * (the | character), a PyObject* and a long long int */
    if(!PyArg_ParseTuple(args, "O|OL", &iterable, &callback,
                &sum)){
        return NULL;
    }

    /* See if we can create an iterator from the iterable. This is
     * effectively the same as doing iter(iterable) in Python */
    iterator = PyObject_GetIter(iterable);
    if(iterator == NULL){
        PyErr_SetString(PyExc_TypeError,
                "Argument is not iterable");
        return NULL;
    }

    /* Check if the callback exists or wasn't specified. If it was
     * specified check whether it's callable or not */
    if(callback != NULL && !PyCallable_Check(callback)){
        PyErr_SetString(PyExc_TypeError,
                "Callback is not callable");
        return NULL;
    }

    /* Loop through all items of the iterable */
    while((item = PyIter_Next(iterator))){
        /* If we have a callback available, call it. Otherwise
         * just return the item as the value */
        if(callback == NULL){
            value = item;
        }else{
            value = PyObject_CallFunction(callback, "O", item);
        }

        /* Add the value to sum and check for overflows */
        sum += PyLong_AsLongLongAndOverflow(value, &overflow);
        if(overflow > 0){
            PyErr_SetString(PyExc_RuntimeError,
                    "Integer overflow");
            return NULL;
        }else if(overflow < 0){
            PyErr_SetString(PyExc_RuntimeError,
                    "Integer underflow");
            return NULL;
        }

        /* If we were indeed using the callback, decrease the
         * reference count to the value because it is a separate
         * object now */
        if(callback != NULL){
            Py_DECREF(value);
        }
        Py_DECREF(item);
    }
    Py_DECREF(iterator);

    return PyLong_FromLongLong(sum);
}
```

确保您注意`PyDECREF`调用，这样可以确保您不会泄漏这些对象。如果没有它们，对象将继续使用，Python 解释器将无法清除它们。

这个函数可以以三种不同的方式调用：

```py
>>> import spam
>>> x = range(10)
>>> spam.sum(x)
45
>>> spam.sum(x, lambda y: y + 5)
95
>>> spam.sum(x, lambda y: y + 5, 5)
100

```

另一个重要问题是，即使我们在转换为`long long int`时捕获了溢出错误，这段代码仍然不安全。如果我们甚至对两个非常大的数字求和（接近`long long int`限制），我们仍然会发生溢出：

```py
>>> import spam
>>> n = (2 ** 63) - 1
>>> x = n,
>>> spam.sum(x)
9223372036854775807
>>> x = n, n
>>> spam.sum(x)
-2

```

# 总结

在本章中，您学习了使用`ctypes`、`CFFI`编写代码以及如何使用本机 C 扩展 Python 功能的最重要方面。这些主题本身就足够广泛，可以填满一本书，但是现在您应该掌握了最重要的主题。即使您现在能够创建 C/C++扩展，我仍然建议您尽量避免这样做。这是因为不够小心很容易出现错误。实际上，至少本章中的一些示例在内存管理方面可能存在错误，并且在给出错误输入时可能会使您的 Python 解释器崩溃。不幸的是，这是 C 的副作用。一个小错误可能会产生巨大的影响。

在构建本章中的示例时，您可能已经注意到我们使用了一个`setup.py`文件，并从`setuptools`库导入。下一章将涵盖这一点——将您的代码打包成可安装的 Python 库，并在 Python 软件包索引上进行分发。
