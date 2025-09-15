# 17

# C/C++ 扩展、系统调用和 C/C++ 库

最后几章向我们展示了众多机器学习和科学计算库。许多这些库并非纯 Python 编写，因为它们从现有库中复用代码，或者出于性能考虑。在本章中，我们将学习如何通过创建 C/C++ 扩展来实现其中的一些功能。

在 *第十二章*，*性能 – 跟踪和减少你的内存和 CPU 使用* 中，我们了解到 `cProfile` 模块比 `profile` 模块快约 10 倍，这表明至少一些 C 扩展比它们的纯 Python 等效物更快。然而，本章不会过多关注性能。这里的目的是与非 Python 库的交互。用 Linus Torvalds 的话来说，任何性能提升都将是完全无意的结果。

如果性能是你的主要目标，你真的不应该考虑手动编写 C/C++ 扩展。对于 Python 核心模块，当然已经这样做了，但在大多数实际应用中，使用 `numba` 或 `cython` 会更好。或者，如果用例允许，可以使用预存的库，如 `numpy` 或 `jax`。使用本章中工具的主要原因应该是复用现有库，这样你就不必重新发明轮子。

在本章中，我们将讨论以下主题：

+   `ctypes` 用于处理来自 Python 的外部（C/C++）函数和数据

+   **C 外部函数接口**（**CFFI**），类似于 `ctypes`，但采用略有不同的方法

+   编写原生 C/C++ 以扩展 Python

# 设置工具

在我们开始之前，重要的是要注意，本章将需要一个与你的 Python 解释器兼容的编译器。不幸的是，这些编译器因平台而异。对于 Linux 发行版，通常可以通过一两个命令轻松实现，而无需太多麻烦。

对于 OS X，体验通常非常相似，主要是因为繁重的工作可以委托给包管理系统，如 Homebrew。对于 Windows，可能会稍微复杂一些，但这个过程在过去几年中已经简化了。

获取所需工具的一个良好且最新的起点是 Python 开发者指南：[`devguide.python.org/setup/`](https://devguide.python.org/setup/).

对于构建实际的扩展，Python 手册可能会有所帮助：[`docs.python.org/3/extending/building.html`](https://docs.python.org/3/extending/building.html).

## 你需要 C/C++ 模块吗？

在几乎所有情况下，我倾向于认为你不需要 C/C++模块。如果你真的需要最佳性能，那么几乎总是有高度优化的 Python 库可用，它们内部使用 C/C++/Fortran 等，并且适合你的需求。有些情况下，原生 C/C++（或只是“非 Python”）是必需的。如果你需要直接与具有特定时序的硬件通信，那么 Python 可能不起作用。然而，通常这类通信应留给操作系统内核级驱动程序来处理特定的时序。无论如何，即使你永远不会自己编写这些模块，当你调试项目时，你可能仍然需要了解它们是如何工作的。

## Windows

对于 Windows，一般推荐使用 Visual Studio。具体版本取决于你的 Python 版本：

+   **Python 3.4**: Microsoft Visual Studio 2010

+   **Python 3.5 和 3.6**: Microsoft Visual Studio 2015 或 Visual Studio 2017

+   **Python 3.7–3.10**: Microsoft Visual Studio 2017

Visual Studio 2019 也受到支持，但 Python 3.7 到 Python 3.10 的官方构建仍然使用 Visual Studio 2017，因此这是推荐解决方案。

安装 Visual Studio 和编译 Python 模块的具体细节超出了本书的范围。幸运的是，Python 文档提供了一些文档来帮助你入门：[`devguide.python.org/setup/#windows`](https://devguide.python.org/setup/#windows)。

如果你正在寻找一个更类似 Linux/Unix 的解决方案，你也可以选择通过 MinGW 使用 GCC 编译器。

## OS X

对于 Mac，这个过程主要是直接的，但也有一些针对 OS X 的特定提示。首先，通过 Mac App Store 安装 Xcode。

一旦完成这些，你应该能够运行以下命令：

```py
$ xcode-select --install 
```

接下来是更有趣的部分。因为 OS X 自带了一个捆绑的 Python 版本（通常已经过时），我建议通过 Homebrew 安装一个新的 Python 版本。安装 Homebrew 的最新说明可以在 Homebrew 主页上找到（[`brew.sh/`](http://brew.sh/)），但安装 Homebrew 的基本命令如下：

```py
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" 
```

之后，请确保使用`doctor`命令检查一切是否设置正确：

```py
$ brew doctor 
```

当所有这些完成后，只需通过 Homebrew 安装 Python，并确保在执行脚本时使用该 Python 版本：

```py
$ brew install python3
$ python3 --version
Python 3.9.7
which python3
/usr/local/bin/python3 
```

还要确保 Python 进程在`/usr/local/bin`中，即 Homebrew 版本。常规 OS X 版本将在`/usr/bin/`中。

## Linux/Unix

对于 Linux/Unix 系统的安装，很大程度上取决于发行版，但通常很简单。

对于使用`yum`作为包管理器的 Fedora、Red Hat、CentOS 和其他系统，使用以下命令：

```py
$ sudo yum install yum-utils
$ sudo yum-builddep python3 
```

对于使用`apt`作为包管理器的 Debian、Ubuntu 和其他系统，使用以下命令：

```py
$ sudo apt-get build-dep python3.10 
```

注意，Python 3.10 目前并非在所有地方都可用，因此你可能需要使用 Python 3.9 或甚至 Python 3.8。

对于大多数系统，要获取安装帮助，进行类似 `<操作系统> python.h` 的网络搜索应该可以解决问题。

# 使用 ctypes 调用 C/C++

`ctypes` 库使得从 C 库调用函数变得非常容易，但你确实需要小心内存访问和数据类型。Python 通常在内存分配和类型转换方面非常宽容；而 C 则绝对不是那么宽容。

## 平台特定库

尽管所有平台都会在某个地方提供标准 C 库，但它的位置和调用方式因平台而异。为了拥有一个简单且易于大多数人访问的环境，我将假设使用 Ubuntu（虚拟）机器。如果你没有可用的原生 Ubuntu 机器，你可以在 Windows、Linux 和 OS X 上通过 VirtualBox 运行它。

由于你通常会希望在本地系统上运行示例，我们将首先展示从标准 C 库加载 `printf` 的基础知识。

### Windows

从 Python 调用 C 函数的一个问题是默认库是平台特定的。虽然以下示例在 Windows 系统上可以正常运行，但在其他平台上则无法运行：

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

`c` `types` 库将 C/C++ 库（在本例中为 `MSVCRT.DLL`）的函数和属性暴露给 Python 安装。由于 `ms` 部分代表 Microsoft，这是一个你通常在非 Windows 系统上找不到的库。

在加载方面，Linux/Unix 和 Windows 之间也存在差异；在 Windows 上，模块通常会被自动加载，而在 Linux/Unix 系统上，你需要手动加载它们，因为这些系统通常会提供同一库的多个版本。

### Linux/Unix

从 Linux/Unix 调用标准系统库确实需要手动加载，但幸运的是，这并不复杂。从标准 C 库获取 `printf` 函数相当简单：

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

对于 OS X，也需要显式加载，但除此之外，它与常规的 Linux/Unix 系统上的工作方式相当相似：

```py
>>> import ctypes
>>> libc = ctypes.cdll.LoadLibrary('libc.dylib')
>>> libc
<CDLL 'libc.dylib', handle ... at 0x...>
>>> libc.printf
<_FuncPtr object at 0x...> 
```

### 使其变得简单

除了库的加载方式之外，不幸的是还有更多差异，但至少早期的示例为你提供了标准 C 库，这允许你直接从 C 实现中调用如 `printf` 这样的函数。如果你由于某种原因加载正确的库有困难，`ctypes.util.find_library` 函数总是可用。

如往常一样，我推荐使用显式声明而不是隐式声明，但在某些情况下，使用此函数可以使事情变得更容易。以下是在 OS X 系统上运行的示例：

```py
# OS X
>>> from ctypes import util
>>> from ctypes import cdll

>>> library = util.find_library('libc')
>>> library
'/usr/lib/libc.dylib'

# Load the library
>>> libc = cdll.LoadLibrary(library)
>>> libc
<CDLL '/usr/lib/libc.dylib', handle ... at 0x...> 
```

## 调用函数和本地类型

通过`ctypes`调用函数几乎和调用原生 Python 函数一样简单。值得注意的是参数和`return`语句。这些应该转换为原生 C 变量。

这些示例假设你从上一段中的某个示例中已经有了`libc`。

我们现在将创建一个 C 字符串，它实际上是一个内存块，字符为 ASCII 字符，并以空字符结尾。在创建 C 字符串后，我们将对字符串运行`printf`：

```py
>>> c_string = ctypes.create_string_buffer(b'some bytes')
>>> ctypes.sizeof(c_string)
11
>>> c_string.raw
b'some bytes\x00'
>>> c_string.value
b'some bytes'
>>> libc.printf(c_string)
10
some bytes>>> 
```

这个输出一开始可能看起来有点混乱，所以让我们分析一下。当我们对`c_string`调用`libc.printf`时，它将直接将字符串写入`stdout`。正因为如此，你可以看到输出是交织的（`some bytes>>>`）与 Python 输出，因为这绕过了 Python 输出缓冲区，Python 并不知道这件事正在发生。此外，你可以看到`libc.printf`返回了`10`，这是写入`stdout`的字节数。

要调用`printf`函数，你*必须*——我无法强调这一点——明确地将你的值从 Python 转换为 C。虽然一开始可能看起来不需要这样做也能工作，但实际上并不是这样：

```py
>>> libc.printf(123)
segmentation fault (core dumped)  python3 
```

记得使用第十一章的`faulthandler`模块来调试`segfaults`。

从示例中还可以注意到，`ctypes.sizeof(c_string)`返回`11`而不是`10`。这是由于 C 字符串所需的尾随空字符造成的，这在 C 字符串的原始属性中是可见的。

没有它，C 语言中的字符串函数（如`printf`）将不知道字符串在哪里结束，因为 C 字符串只是内存中的一块字节，C 只知道字符串的起始内存地址；结束由空字符指示。这就是为什么 C 语言中的内存管理需要非常注意。

如果你分配了一个大小为 5 的字符串并写入 10 个字节，你将写入你的变量之外的内存，这可能是另一个函数、另一个变量，或者程序内存之外。这将导致段错误。

Python 通常会保护你免受愚蠢的错误；C 和 C++则绝对不会。引用 Bjarne Stroustrup（C++的创造者）的话：

*“C 语言让你容易踩到自己的脚；C++语言让你更难，但当你这么做时，它会让你失去整条腿。”*

与 C 语言不同，C++确实有字符串类型来保护你在这些情况下的安全。然而，它仍然是一种你可以轻松访问内存地址的语言，错误很容易发生。

要将其他类型（如整数）传递给`libc`函数，我们不得不使用一些转换。在某些情况下，这是可选的：

```py
>>> format_string = b'Number: %d\n'
>>> libc.printf(format_string, 123)
Number: 123
12

>>> x = ctypes.c_int(123)
>>> libc.printf(format_string, x)
Number: 123
12 
```

但并非所有情况都是这样，所以建议谨慎行事，并明确转换是更安全的选项：

```py
>>> format_string = b'Number: %.3f\n'
>>> libc.printf(format_string, 123.45)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ctypes.ArgumentError: argument 2: <class 'TypeError'>: Don't know how to convert parameter 2

>>> x = ctypes.c_double(123.45)
>>> libc.printf(format_string, x)
Number: 123.450
16 
```

重要的是要注意，尽管这些值可以作为原生 C 类型使用，但它们仍然可以通过`value`属性进行修改：

```py
>>> x = ctypes.c_double(123.45)
>>> x.value
123.45
>>> x.value = 456
>>> x
c_double(456.0) 
```

除非原始对象是不可变的，否则这种情况才会发生。这是一个非常重要的区别。`create_string_buffer` 对象创建了一个可变字符串对象，而 `c_wchar_p`、`c_char_p` 和 `c_void_p` 创建了对实际 Python 字符串的引用。由于 Python 中的字符串是不可变的，因此这些值也是不可变的。你仍然可以更改 `value` 属性，但它只会分配一个新的字符串。将其中一个不可变变量传递给一个会修改内部值的 C 函数会导致不可预测的行为和/或崩溃。

只有整数、字符串和字节应该能够无问题地转换为 C，但我个人建议你始终转换所有值，这样你就可以确定你会得到哪种类型以及如何处理它。

## 复杂数据结构

我们已经看到，我们不能直接将 Python 值传递给 C，但如果我们需要更复杂的对象，如类或元组怎么办？幸运的是，我们可以轻松地使用 `ctypes` 创建（并访问）C 结构：

```py
>>> from _libc import libc
>>> import ctypes

>>> class ComplexStructure(ctypes.Structure):
...     _fields_ = [
...         ('some_int', ctypes.c_int),
...         ('some_double', ctypes.c_double),
...         ('some_char', ctypes.c_char),
...         ('some_string', ctypes.c_char_p),
...     ]
... 
>>> structure = ComplexStructure(123, 456.789, b'x', b'abc')
>>> structure.some_int
123
>>> structure.some_double
456.789
>>> structure.some_char
b'x'
>>> structure.some_string
b'abc' 
```

这支持任何基本数据类型，如整数、浮点数和字符串。嵌套也是支持的；例如，在这个例子中，其他结构可以使用 `ComplexStructure` 而不是 `ctypes.c_int`。

## 数组

在 Python 中，我们通常使用 `list` 来表示对象的集合。这些非常方便，因为你可以轻松地添加和删除值。在 C 中，默认的集合对象是 **array**，它只是一个具有固定大小的内存块。

块的大小（以字节为单位）是通过将元素数量乘以类型大小来确定的。对于 `char` 类型，这是 `8` 位，所以如果你想要存储 `100` 个字符，你将需要 `100 * 8 bits = 800 bits = 100 bytes`。

这实际上就是它——一个内存块——而你从 C 收到的唯一引用是内存块开始地址的指针。由于指针确实有类型，在这个例子中是 `char*`，C 将知道在尝试访问不同项时需要跳过多少字节。实际上，当尝试访问 `char` 数组中的第 25 项时，你只需执行 `array_pointer + 24 * sizeof(char)`。这有一个方便的快捷方式：`array_pointer[24]`。请注意，我们需要访问索引 24，因为我们从 0 开始计数，就像 Python 的集合如列表和字符串一样。

注意，C 语言不存储数组中元素的数量，因此尽管我们的数组只有 100 个元素，但这不会阻止我们执行 `array_pointer[1000]` 并读取其他（随机）内存。然而，在某个时刻，你将超出应用程序预留的内存，操作系统将用段错误来惩罚你。

如果考虑到所有这些限制，C 数组确实可以使用，但错误会很快出现，C 语言不会宽容。没有警告；只有崩溃和奇怪的行为代码。除此之外，让我们看看我们如何容易地使用 `ctypes` 声明一个数组：

```py
>>> TenNumbers = 10 * ctypes.c_double
>>> numbers = TenNumbers()
>>> numbers[0]
0.0 
```

如您所见，由于固定大小和在使用前必须声明类型的要求，其使用略显笨拙。然而，它确实按您预期的那样工作。此外，与常规 C 语言不同，默认情况下值会被初始化为零，这可以在从 Python 访问时保护您免受越界错误。当然，这可以与之前创建的定制结构相结合：

```py
>>> GrossComplexStructures = 144 * ComplexStructure 
>>> complex_structures = GrossComplexStructures()

>>> complex_structures[10].some_double = 123
>>> complex_structures[10]
<__main__.ComplexStructure object at ...>
>>> complex_structures
<__main__.ComplexStructure_Array_144 object at ...> 
```

尽管您不能简单地向这些数组追加内容以调整大小，但它们实际上在几个约束条件下是可以调整大小的。首先，新数组需要比原始数组大。其次，大小需要以字节为单位指定，而不是以项目为单位。为了说明这一点，我们有这个示例：

```py
>>> TenNumbers = 10 * ctypes.c_double
>>> numbers = TenNumbers()

>>> ctypes.resize(numbers, 11 * ctypes.sizeof(ctypes.c_double))
>>> ctypes.resize(numbers, 10 * ctypes.sizeof(ctypes.c_double))
>>> ctypes.resize(numbers, 9 * ctypes.sizeof(ctypes.c_double))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: minimum size is 80

>>> numbers[:5] = range(5)
>>> numbers[:]
[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
```

作为起点，`TenNumbers` 数组有 10 个项目。接下来，我们尝试将数组大小调整为 11，这是可行的，因为它比原始的 10 大。将大小调整回 10 也是允许的，但将大小调整为 9 个项目是不允许的，因为这将少于我们最初拥有的 10 个项目。

最后，我们同时修改一个项目切片，这正如您所预期的那样工作。

## 内存管理的陷阱

除了明显的内存分配问题和混合可变和不可变对象之外，还有一个不那么明显的内存可变性问题。

在常规 Python 中，我们可以做一些像 `a, b = b, a` 的事情，并且它会按您预期的那样工作，因为 Python 使用内部临时变量。不幸的是，在常规 C 中，您没有这样的便利；在 `ctypes` 中，您确实有 Python 为您处理临时变量的好处，但有时这仍然可能出错：

```py
>>> import ctypes

>>> class Point(ctypes.Structure):
...     _fields_ = ('x', ctypes.c_int), ('y', ctypes.c_int)

>>> class Vertex(ctypes.Structure):
...     _fields_ = ('c', Point), ('d', Point)

>>> a = Point(0, 1)
>>> b = Point(2, 3)
>>> a.x, a.y, b.x, b.y
(0, 1, 2, 3)

# Swap points a and b
>>> a, b = b, a
>>> a.x, a.y, b.x, b.y
(2, 3, 0, 1)

>>> v = Vertex()
>>> v.c = Point(0, 1)
>>> v.d = Point(2, 3)
>>> v.c.x, v.c.y, v.d.x, v.d.y
(0, 1, 2, 3)

# Swap points c and d
>>> v.c, v.d = v.d, v.c
>>> v.c.x, v.c.y, v.d.x, v.d.y
(2, 3, 2, 3) 
```

在第一个示例中，当我们交换 `a` 和 `b` 时，我们得到预期的 `2, 3, 0, 1`。在第二个示例中，我们得到 `2, 3, 2, 3`。问题在于这些对象被复制到一个临时缓冲变量中，但在此期间对象本身正在被改变。

让我们进一步阐述以增加清晰度。在 Python 中，当你执行 `a, b = b, a` 时，它实际上会运行 `temp = a; a = b; b = temp`。这样，替换就会按预期工作，你将在 `a` 和 `b` 中收到正确的值。

当你在 C 语言中执行 `a, b = b, a` 时，实际上得到的是 `a = b; b = a`。在执行 `b = a` 语句时，`a` 的值已经被 `a = b` 语句所改变，因此此时 `a` 和 `b` 都将具有 `b` 在那个点的原始值。

# CFFI

`CFFI`（C Foreign Function Interface）库提供了与 `ctypes` 非常相似的选择，但它更为直接。与 `ctypes` 库不同，CFFI 真正需要 C 编译器。有了它，您就有机会以简单的方式从 Python 直接调用您的 C 编译器。我们通过调用 `printf` 来说明：

```py
>>> import cffi

>>> ffi = cffi.FFI()
>>> ffi.cdef('int printf(const char* format, ...);')
>>> libc = ffi.dlopen(None)
>>> arg = ffi.new('char[]', b'Printing using CFFI\n')
>>> libc.printf(arg)
20
Printing using CFFI 
```

好吧…这看起来有点奇怪，对吧？我们不得不定义`printf`函数的形态，并使用有效的 C 函数头指定`printf`的参数。此外，我们还需要手动指定 C 字符串为`char[]`数组。使用`ctypes`的话，这就不需要了，但与`ctypes`相比，`CFFI`有几个优点。

使用 CFFI，我们可以直接控制发送给 C 编译器的信息，这使得我们比使用`ctypes`有更多的内部控制权。这意味着你可以精确控制你提供给函数的类型以及你返回的类型，并且你可以使用 C 宏。

此外，CFFI 允许轻松重用现有的 C 代码。如果你使用的 C 代码有几个`struct`定义，你不需要手动将它们映射到`ctypes.Structure`类；你可以直接使用`struct`定义。你甚至可以直接在你的 Python 代码中编写 C 代码，CFFI 会为你调用编译器和构建库。

回到声明部分，你可能注意到我们使用`ffi.dlopen`时传入了`None`参数。当你向这个函数传递`None`时，它将自动加载整个 C 命名空间；至少在非 Windows 系统上是这样。在 Windows 系统上，你需要明确告诉 CFFI 要加载哪个库。

如果你记得`ctypes.util.find_library`函数，你可以在这种情况下再次使用它，具体取决于你的操作系统：

```py
>>> from ctypes import util
>>> import cffi

# Initialize the FFI builder
>>> ffi = cffi.FFI()

# Find the libc library on OS X. Look back at the ctypes examples
# for other platforms.
>>> library = util.find_library('libc.dylib')
>>> library
'/usr/lib/libc.dylib'

# Load the library
>>> libc = ffi.dlopen(library)
>>> libc
<cffi.api._make_ffi_library.<locals>.FFILibrary object at ...>

# We do have printf available, but CFFI requires a signature
>>> libc.printf
Traceback (most recent call last):
  ...
AttributeError: printf

# Define the printf signature and call printf
>>> ffi.cdef('int printf(const char* format, ...);')
>>> libc.printf
<cdata 'int(*)(char*, ...)' ...> 
```

我们可以看到，最初的工作方式与`ctypes`相当，加载库也很简单。真正不同的是在调用函数和使用库属性时；那些需要明确定义。

幸运的是，函数签名几乎总是可以在 C 头文件中找到，以便你不必自己编写。这就是 CFFI 的一个优点：它允许你重用现有的 C 代码。

## 复杂数据结构

`CFFI`的定义与`ctypes`的定义有些相似，但与 Python 模拟 C 不同，它只是从 Python 可访问的纯 C。实际上，这只是一个小的语法差异。虽然`ctypes`是一个用于从 Python 访问 C 的库，同时尽可能接近 Python 语法，但 CFFI 使用纯 C 语法来访问 C 系统，这实际上消除了对熟悉 C 的人来说的一些困惑。我个人觉得 CFFI 更容易使用，因为我有 C 的经验，知道实际上发生了什么，而我对`ctypes`则不是总是 100%确定。

让我们用 CFFI 重复`Vertex`和`Point`的例子：

```py
>>> import cffi

>>> ffi = cffi.FFI()

# Create the structures as C structs
>>> ffi.cdef('''
... typedef struct {
...     int x;
...     int y;
... } point;
...
... typedef struct {
...     point a;
...     point b;
... } vertex;
... ''')

# Create a vertex and return the pointer
>>> v = ffi.new('vertex*')

# Set the data
>>> v.a.x, v.a.y, v.b.x, v.b.y = (0, 1, 2, 3)

# Print before change
>>> v.a.x, v.a.y, v.b.x, v.b.y
(0, 1, 2, 3)

>>> v.a, v.b = v.b, v.a

# Print after change
>>> v.a.x, v.a.y, v.b.x, v.b.y
(2, 3, 2, 3) 
```

如你所见，可变变量的问题仍然存在，但代码仍然可用。由于结构可以从你的 C 头文件中复制，你唯一需要做的是为顶点分配内存。

在 C 语言中，一个普通的`int`类型变量`x`看起来像`int x;`。一个指向具有`int`大小的内存地址的指针看起来像这样：`int *x;`。指针中的`int`部分告诉编译器在使用变量时需要获取多少内存。为了说明：

```py
int a = 123; // Variable a contains integer 123
int* b = &a; // Variable b contains the memory address of a
int c = *b;  // Variable c contains 123, the value at memory address c 
```

`&`运算符返回变量的内存地址，而`*`运算符返回指针地址处的值。

CFFI 的特殊工作方式允许你简化这些操作。在 C 语言中，通常使用`vertex*`只会分配指针的内存，而不是`vertex`本身。在 CFFI 的情况下，这一点会自动处理。

## 数组

使用 CFFI 分配新变量的内存几乎是微不足道的。上一节向你展示了单个`struct`分配的例子。现在让我们看看我们如何分配结构体数组：

```py
>>> import cffi

>>> ffi = cffi.FFI()

# Create arrays of size 10:
>>> x = ffi.new('int[10]')
>>> y = ffi.new('int[]', 10)

>>> x
<cdata 'int[10]' owning 40 bytes>
>>> y
<cdata 'int[]' owning 40 bytes>

>>> x[0:10] = range(10)
>>> list(x)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>> x[:] = range(10)
Traceback (most recent call last):
    ...
IndexError: slice start must be specified

>>> x[0:100] = range(100)
Traceback (most recent call last):
    ...
IndexError: index too large (expected 100 <= 10) 
```

在这种情况下，你可能会想知道为什么切片包括了起始和结束位置。这是 CFFI 的要求。虽然不是问题，但多少有点令人烦恼。幸运的是，正如你在上面的例子中可以看到的，CFFI 确实保护我们不会超出数组边界进行分配。

## ABI 还是 API？

总是有些注意事项。到目前为止的例子部分使用了**ABI**（**应用程序二进制接口**），它从库中加载二进制结构。使用标准 C 库通常很安全；使用其他库通常则不是。**API**（**应用程序编程接口**）和 ABI 之间的区别在于后者在二进制级别调用函数，直接访问内存，直接调用内存位置，并期望它们是函数。

要能够这样做，所有的大小都需要保持一致。当作为 32 位二进制文件编译时，指针将是 32 位；当作为 64 位二进制文件编译时，指针将是 64 位。这意味着偏移量不一定是一致的，你可能会错误地将内存块作为函数调用。

在 CFFI 中，这是`ffi.dlopen`和`ffi.set_source`之间的区别。在这里，`dlopen`并不总是安全的，但`set_source`是安全的，因为它传递了一个编译器而不是仅仅猜测如何调用方法。使用`set_source`的缺点是你需要你打算使用的库的实际源代码。让我们看看使用`ffi.set_source`调用我们定义的函数的快速示例：

```py
>>> import cffi

>>> ffi = cffi.FFI()

# In API mode, we can in-line the actual C code
>>> ffi.set_source('_sum', '''
... int sum(int* input, int n){
...     int result = 0;
...     while(n--)result += input[n];
...     return result;
... }
... ''')
>>> ffi.cdef('int sum(int*, int);')

>>> library = ffi.compile() 
```

CFFI 的初始化和往常一样正常，但不是使用`ffi.dlopen()`，我们现在使用`ffi.set_source()`直接将 C 代码传递给 CFFI。通过这样做，CFFI 可以为我们自己的系统编译特定的库，因此我们知道我们不会遇到 ABI 问题，因为我们是通过调用`ffi.compile()`自己创建 ABI 的。

在`ffi.compile()`步骤完成后，CFFI 已经创建了一个`_sum.dll`、`sum.so`或`_sum.cpython-...-os.so`文件，它可以作为一个普通的 Python 库导入。现在我们将使用生成的库：

```py
# Now we can import the library
>>> import _sum

# Or use 'ffi.dlopen()' with the results from the compile step
>>> _sum_lib = ffi.dlopen(library)

# Create an array with 5 items
>>> N = 5
>>> array = ffi.new('int[]', N)
>>> array[0:N] = range(N)

# Call our C function from either the import or the dlopen
>>> _sum.lib.sum(array, N)
10

>>> _sum_lib.sum(array, N)
10 
```

如你所见，`import _sum` 和 `ffi.dlopen(library)` 在这个情况下都有效。对于生产环境的应用，我推荐使用 `import _sum` 方法，但 `ffi.dlopen()` 方法对于像 Jupyter Notebooks 这样的长时间运行的应用来说可能非常方便。如果你使用 `import _sum` 并在库中做了更改，除非你首先调用 `reload(_sum)`，否则它不会显示你的更改。

由于这是一个 C 函数，我们需要传递一个 C 数组来处理复杂数据类型，这就是为什么我们在这里使用 `ffi.new()`。之后，函数调用就很简单了，但由于 C 数组没有大小概念，我们需要传递数组大小才能使其工作。

你可以很容易地越界，用一些任意数字代替 `N`，函数很可能会正常运行而不会崩溃，但它会返回非常奇怪的结果，因为它会在你的内存中随机求和数据。

## CFFI 或 ctypes？

这完全取决于你正在寻找什么。如果你有一个只需要调用的 C 库，而且你不需要任何特殊功能，那么 `ctypes` 很可能是更简单的选择。如果你实际上正在编写自己的 C 库并尝试从 Python 链接库，CFFI 可能是一个更方便的选项。

在 C/C++ 中，链接一个库意味着使用一个外部预编译的库，而不需要源代码。你确实需要拥有头文件，这些头文件包含诸如函数参数和返回类型等详细信息。这正是我们使用 CFFI 在 ABI 模式下所做的事情。

如果你不太熟悉 C 编程语言，我肯定会推荐 `ctypes` 或许是 `cython`。

# 本地 C/C++ 扩展

我们迄今为止所使用的库只展示了如何在 Python 代码中访问 C/C++ 库。现在我们将来看故事的另一面：Python 中的 C/C++ 函数/模块是如何实际编写的，以及像 `cPickle` 和 `cProfile` 这样的模块是如何创建的。

## 一个基本示例

在我们真正开始编写和使用本地的 C/C++ 扩展之前，我们有一些先决条件。首先，我们需要编译器和 Python 头文件；本章开头提供的说明应该已经为我们处理了这一点。之后，我们需要告诉 Python 要编译什么。`setuptools` 包主要处理这一点，但我们需要创建一个 `setup.py` 文件：

```py
import pathlib
import setuptools

# Get the current directory
PROJECT_PATH = pathlib.Path(__file__).parent

sum_of_squares = setuptools.Extension('sum_of_squares', sources=[
    # Get the relative path to sum_of_squares.c
    str(PROJECT_PATH / 'sum_of_squares.c'),
])

if __name__ == '__main__':
    setuptools.setup(
        name='SumOfSquares',
        version='1.0',
        ext_modules=[sum_of_squares],
    ) 
```

这告诉 Python 我们有一个名为 `sum_of_squares` 的 `Extension` 对象，它将基于 `sum_of_squares.c`。

现在，让我们编写一个 C 函数，该函数计算给定数字的所有完全平方数（`2*2`、`3*3` 等等）。Python 代码将存储在 `sum_of_squares_python.py` 中，看起来像这样：

```py
def sum_of_squares(n):
    total = 0
    for i in range(n):
        if i * i < n:
            total += i * i
        else:
            break

    return total 
```

这段代码的原始 C 版本可能看起来像这样：

```py
long sum_of_squares(long n){
    long total = 0;
    /* The actual summing code */
    for(int i=0; i<n; i++){
        if((i * i) < n){
            total += i * i;
        }else{
            break;
        }
    }

    return total;
} 
```

既然我们已经知道了 C 代码的样子，我们将创建我们将要使用的实际的 C Python 版本。

正如我们从`ctypes`和`CFFI`中看到的那样，Python 和 C 有不同的数据类型，需要进行一些转换。由于 CPython 解释器是用 C 编写的，因此它有特定的定义来处理这个转换步骤。

要加载这些定义，我们需要包含`Python.h`，这是 CPython 头文件，应该包含你需要的一切。

如果你仔细观察，你会看到实际的求和代码与 C 版本相同，但我们需要相当多的转换步骤来让 Python 理解我们在做什么：

```py
#include <Python.h>

static PyObject* sum_of_squares(PyObject *self, PyObject
        *args){
    /* Declare the variables */
    int n;
    int total = 0;

    /* Parse the arguments */
    if(!PyArg_ParseTuple(args, "i", &n)){
        return NULL;
    }

    /* The actual summing code */
    for(int i=0; i<n; i++){
        if((i * i) < n){
            total += i * i;
        }else{
            break;
        }
    }

    /* Return the number but convert it to a Python object first */
    return PyLong_FromLong(total);
}

static PyMethodDef methods[] = {
    /* Register the function */
    {"sum_of_squares", sum_of_squares, METH_VARARGS,
     "Sum the perfect squares below n"},
    /* Indicate the end of the list */
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "sum_of_squares", /* Module name */
    NULL, /* Module documentation */
    -1, /* Module state, -1 means global. This parameter is
           for sub-interpreters */
    methods,
};

/* Initialize the module */
PyMODINIT_FUNC PyInit_sum_of_squares(void){
    return PyModule_Create(&module);
} 
```

它看起来相当复杂，但实际上并不难。在这种情况下，只是有很多开销，因为我们只有一个函数。通常，你会有几个函数，在这种情况下，你只需要扩展`methods`数组并创建函数。我们将在稍后详细解释代码，但首先，让我们看看如何运行我们的第一个示例。我们需要构建和安装模块：

```py
$ python3 T_09_native/setup.py build install
running build
running build_ext
building 'sum_of_squares' extension ...
...
Processing dependencies for SumOfSquares==1.0
Finished processing dependencies for SumOfSquares==1.0 
```

现在，让我们创建一个小型测试脚本，以测量 Python 版本和 C 版本之间的差异。首先，一些导入和设置：

```py
import sys
import timeit
import argparse
import functools

from sum_of_squares_py import sum_of_squares as sum_py

try:
    from sum_of_squares import sum_of_squares as sum_c
except ImportError:
    print('Please run "python setup.py build install" first')
    sys.exit(1) 
```

现在我们已经导入了模块（或者如果你还没有运行构建步骤，你会得到一个错误），我们可以开始基准测试：

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('repetitions', type=int)
    parser.add_argument('maximum', type=int)
    args = parser.parse_args()

    timer = functools.partial(
        timeit.timeit, number=args.repetitions, globals=globals())

    print(f'Testing {args.repetitions} repetitions with maximum: '
          f'{args.maximum}')

    result = sum_c(args.maximum)
    duration_c = timer('sum_c(args.maximum)')
    print(f'C: {result} took {duration_c:.3f} seconds')

    result = sum_py(args.maximum)
    duration_py = timer('sum_py(args.maximum)')
    print(f'Py: {result} took {duration_py:.3f} seconds')

    print(f'C was {duration_py / duration_c:.1f} times faster') 
```

从本质上讲，我们有一个基本的基准测试脚本，其中我们将 C 版本与这里的 Python 版本进行比较，具有可配置的重复次数和最大测试数。现在，让我们执行它：

```py
$ python3 T_09_native/test.py 10000 1000000
Testing 10000 repetitions with maximum: 1000000
C: 332833500 took 0.009 seconds
Py: 332833500 took 1.264 seconds
C was 148.2 times faster 
```

完美！结果完全相同，但速度要快得多。

如果你的目标是追求速度，那么你应该尝试使用`numba`。将`@numba.njit`装饰器添加到`sum_of_squares_python`中要简单得多，而且可能甚至更快。

写 C 模块的主要优势是重用现有的 C 代码，然而。对于速度提升，你通常使用`cython`、`numba`或者将你的代码转换为使用`numpy`或`jax`等库会更好。

## C 不是 Python – 尺寸很重要

Python 语言使得编程变得如此简单，有时你可能会忘记底层数据结构；在 C 和 C++中，你不能承担这样的风险。只需参考我们上一节中的示例，但使用不同的参数：

```py
$ python3 T_09_native/test.py 1000 10000000
Testing 1000 repetitions with maximum: 10000000
C sum of squares: 1953214233 took 0.003 seconds
Python sum of squares: 10543148825 took 0.407 seconds
C was 145.6 times faster 
```

它仍然非常快，但数字怎么了？Python 和 C 版本给出了不同的结果，`1953214233`与`10543148825`。这是由 C 中的整数溢出引起的。虽然 Python 数字本质上可以有任意大小，但 C 中一个常规数字有一个固定的大小。你得到多少取决于你使用的类型（`int`、`long`等）和你的架构（32 位、64 位等），但这绝对是一件需要小心的事情。在某些情况下，它可能要快数百倍，但如果结果不正确，那就毫无意义。

我们当然可以增加一点大小。这使得它更好：

```py
typedef unsigned long long int bigint;

static PyObject* sum_of_large_squares(PyObject *self, PyObject *args){
    /* Declare the variables */
    bigint n;
    bigint total = 0;

    /* Parse the arguments */
    if(!PyArg_ParseTuple(args, "K", &n)){
        return NULL;
    }

    /* The actual summing code */
    for(bigint i=0; i<n; i++){
        if((i * i) < n){
            total += i * i;
        }else{
            break;
        }
    }

    /* Return the number but convert it to a Python object first */
    return PyLong_FromUnsignedLongLong(total);
} 
```

我们使用`typedef`创建了一个`bigint`别名，用于`unsigned long long int`。

如果我们现在测试它，我们会发现它运行得很好：

```py
$ python3 T_10_size_matters/test.py 1000 10000000
Testing 1000 repetitions with maximum: 10000000
C: 10543148825 took 0.001 seconds
Py: 10543148825 took 0.405 seconds
C was 270.3 times faster 
```

随着规模的增加，性能差异也随之增大。

将数字变得更大又会导致问题再次出现，因为即使是`unsigned long long int`也有其极限：

```py
$ python3 T_10_size_matters/test.py 1 100000000000000
Testing 1 repetitions with maximum: 100000000000000
C: 1291890006563070912 took 0.004 seconds
Py: 333333283333335000000 took 1.270 seconds
C was 293.7 times faster 
```

那么，如何解决这个问题呢？简单的答案是您无法解决，Python 也没有真正解决这个问题。复杂的答案是，如果您使用不同的数据类型来存储数据，您就可以解决这个问题。C 语言本身并没有 Python 所具有的“大数支持”。

Python 通过在内存中组合几个常规数字来支持无限大的数字。当需要时，它会自动切换到这些类型的数字。在 Python 2 中，`int`和`long`类型之间的区别更为明显。在 Python 3 中，`long`和`int`类型已经被合并到`int`类型中。您将不会注意到`long`类型的切换；它将在后台自动发生。

在 C 语言中，没有常见的提供这种功能的条款，因此没有简单的方法来实现这一点。但我们可以检查错误：

```py
static unsigned long long int get_number_from_object(int* overflow, 
        PyObject* some_very_large_number){
    return PyLong_AsLongLongAndOverflow(sum, overflow);
} 
```

注意，这仅适用于`PyObject*`，这意味着它不适用于内部 C 溢出。然而，当然，您只需保留原始的 Python long 并在其上执行操作即可。因此，您在 C 语言中也可以不费太多力气地实现大数支持。

## 以下是对示例的解释

我们已经看到了示例的结果，但如果您不熟悉 Python C API，您可能会对函数参数看起来为什么是这个样子感到困惑。

`sum_of_squares`中的基本计算与常规 C 语言的`sum_of_squares`函数相同，但有一些细微的差别。首先，使用 Python C API 的函数的类型定义应该看起来像这样：

```py
static PyObject* sum_of_squares(PyObject *self, PyObject *args); 
```

让我们分解这个问题。

### 静态

这意味着该函数是**静态的**。静态函数只能从同一编译单元中调用。这实际上导致了一个无法从其他模块链接（导入/使用）的函数，这使得编译器可以进一步优化。由于 C 语言中的函数默认是全局的，这可以非常有用，可以防止命名冲突。不过，为了确保安全，如果您使用一个不太可能唯一的名称，您可以在函数名称前加上模块的名称。

注意不要将这里的单词`static`与变量前的`static`混淆。它们是完全不同的概念。一个`static`变量意味着该变量将在整个程序运行期间存在，而不是仅在函数运行期间。

### PyObject*

`PyObject` 类型是 Python 数据类型的基本类型，这意味着所有 Python 对象都可以转换为 `PyObject*`（`PyObject` 指针）。实际上，它只告诉编译器期望哪种属性，这可以在以后的类型识别和内存管理中使用。通常，使用可用的宏，如 `Py_TYPE(some_object)`，而不是直接访问 `PyObject*` 是更好的选择。内部，这个宏扩展为 `(((PyObject*)(o))->ob_type)`，这就是为什么宏通常是一个更好的选择。除了难以阅读外，打字错误也容易发生。

属性列表很长，并且很大程度上取决于对象的类型。对于这些，你可以参考 Python 文档：[`docs.python.org/3/c-api/typeobj.html`](https://docs.python.org/3/c-api/typeobj.html)。

整个 Python C API 可能能填满一本书，但幸运的是，它在 Python 手册中有很好的文档。另一方面，它的使用可能不那么明显。

### 解析参数

使用常规的 C 和 Python，你需要明确指定参数，因为 C 中处理可变大小的参数有点棘手。这是因为它们需要单独解析。`PyObject* args` 是指向包含实际值的对象的引用。为了解析这些，你需要知道期望多少种类型的变量。在示例中，我们使用了 `PyArg_ParseTuple` 函数，它仅将参数解析为位置参数，但使用 `PyArg_ParseTupleAndKeywords` 或 `PyArg_VaParseTupleAndKeywords` 也可以轻松地解析命名参数。这些函数之间的区别在于，前者使用可变数量的参数来指定目标，而后者使用 `va_list` 来设置值。

让我们分析实际示例中的代码：

```py
if(!PyArg_ParseTuple(args, "i", &n)){
    return NULL;
} 
```

我们知道 `args` 是包含实际参数引用的对象。`"i"` 是一个格式字符串，在这种情况下将尝试解析一个整数。`&n` 告诉函数将值存储在 `n` 变量的内存地址中。

格式字符串在这里是重要的部分。根据字符的不同，你可以得到不同的数据类型，但有很多种；`i` 指定一个常规整数，而 `s` 将你的变量转换为 C 字符串（实际上是一个 `char*`，它是一个以空字符终止的字符数组）。应该注意的是，幸运的是，这个函数足够智能，能够考虑溢出。

解析多个参数相当类似；你需要在格式字符串中添加多个字符，以及多个目标变量：

```py
PyObject* callback;
int n;

/* Parse the arguments */
if(!PyArg_ParseTuple(args, "Oi", &callback, &n)){
    return NULL;
} 
```

关键字参数的版本类似，但需要更多的代码更改，因为方法列表需要被告知该函数接受关键字参数。否则，`kwargs` 参数永远不会到达：

```py
static PyObject* function(
        PyObject *self,
        PyObject *args,
        PyObject *kwargs){
    /* Declare the variables */
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

让我们看看与仅支持 `*args` 版本的区别：

1.  与纯 Python 类似，我们的函数头现在包括 `PyObject *kwargs`。

1.  因为我们需要在 C 中预先分配字符串，所以我们有一个名为 `keywords` 的单词数组，其中包含我们计划解析的所有 `kwargs`。

1.  我们现在必须使用 `PyArg_ParseTupleAndKeywords` 而不是 `PyArg_ParseTuple`。这个函数与 `PyArg_ParseTuple` 函数重叠，并通过遍历先前定义的 `keywords` 数组添加了关键字解析。

1.  在函数注册表中，我们需要指定该函数支持关键字参数，除了 `METH_VARARGS` 标志外，还要添加 `METH_KEYWORDS` 标志。

注意，这仍然支持正常参数，但现在也支持关键字参数。

## C 不是 Python – 错误是沉默的或致命的

正如我们在前面的示例中所看到的，整数溢出通常不会引起你的注意，而且不幸的是，没有好的跨平台方法来捕获它们。然而，这些实际上是更容易处理的错误；最糟糕的是通常内存管理。在 Python 中，如果你得到一个错误，你会得到一个异常，你可以捕获它。在 C 中，你实际上无法优雅地处理它。以除以零为例：

```py
$ python3 -c '1/0'
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ZeroDivisionError: division by zero 
```

这足够简单，可以用 `try: ... except ZeroDivisionError: ...` 来捕获。另一方面，如果 C 中出现错误，它将杀死整个进程。但调试 C 代码是 C 编译器调试器的作用，要找到错误的原因，可以使用在第十一章“调试 – 解决错误”中讨论的 `faulthandler` 模块。现在，让我们看看我们如何从 C 中正确地抛出错误：

```py
static PyObject* count_eggs(PyObject *self, PyObject *args){
    PyErr_SetString(PyExc_RuntimeError, "Too many eggs!");
    return NULL;
} 
```

当执行此操作时，它实际上会运行 `raise RuntimeError('Too many eggs!')`。语法略有不同——使用 `PyErr_SetString` 而不是 `raise`——但基本原理相同。

## 从 C 调用 Python – 处理复杂类型

我们已经看到了如何从 Python 调用 C 函数，但现在让我们尝试从 C 调用 Python 并返回。我们不会使用现成的 `sum` 函数，而是将构建一个自己的函数，它使用回调并处理任何类型的可迭代对象。虽然这听起来足够简单，但实际上确实需要一点类型处理，因为您只能期望作为参数的 `PyObject*`。这与简单的类型（如整数、字符和字符串）相反，这些类型会立即转换为原生 Python 版本。

为了清晰起见，这只是一个被拆分成多个部分的单个函数。

首先，我们从 `include` 函数签名开始，并声明我们需要的变量。请注意，`total` 和 `callback` 的值是默认值，以防这些参数未指定：

```py
#include <Python.h>

static PyObject* custom_sum(PyObject* self, PyObject* args){
    long long int total = 0;
    int overflow = 0;
    PyObject* iterator;
    PyObject* iterable;
    PyObject* callback = NULL;
    PyObject* value;
    PyObject* item; 
```

现在我们解析一个 `PyObject*`，后面可以跟一个可选的 `PyObject*` 和一个 `long long int`。这由 `O|OL` 参数指定。结果将存储在 `iterable`、`callback` 和 `total` 的内存地址中（`&` 发送变量的内存地址）：

```py
 if(!PyArg_ParseTuple(args, "O|OL", &iterable, &callback, &total)){
        return NULL;
    } 
```

我们检查是否可以从可迭代对象中创建一个迭代器。这在 Python 中相当于 `iter(iterable)`：

```py
 iterator = PyObject_GetIter(iterable);
    if(iterator == NULL){
        PyErr_SetString(PyExc_TypeError,
                "Argument is not iterable");
        return NULL;
    } 
```

接下来，我们检查回调是否存在或未指定。如果已指定，检查它是否可调用：

```py
 if(callback != NULL && !PyCallable_Check(callback)){
        PyErr_SetString(PyExc_TypeError, "Callback is not callable");
        return NULL;
    } 
```

在遍历可迭代对象时，如果我们有一个可用的回调，我们将调用它。否则，我们只需使用`item`作为`value`：

```py
 while((item = PyIter_Next(iterator))){
        if(callback == NULL){
            value = item;
        }else{
            value = PyObject_CallFunction(callback, "O", item);
        } 
```

我们将值添加到`total`中并检查溢出：

```py
 total += PyLong_AsLongLongAndOverflow(value, &overflow);
        if(overflow > 0){
            PyErr_SetString(PyExc_RuntimeError, "Integer overflow");
            return NULL;
        }else if(overflow < 0){
            PyErr_SetString(PyExc_RuntimeError, "Integer underflow");
            return NULL;
        } 
```

如果我们确实使用了回调，由于它现在是一个单独的对象，我们将减少该值的引用计数。

我们还需要取消引用`item`和迭代器。忘记这样做会导致内存泄漏，因为它会减少 Python 垃圾回收器的引用计数。

因此，始终确保在使用`PyObject*`类型后调用`PyDECREF`函数：

```py
 if(callback != NULL){
            Py_DECREF(value);
        }
        Py_DECREF(item);
    }
    Py_DECREF(iterator); 
```

最后，我们需要将`total`转换为正确的返回类型并返回它：

```py
 return PyLong_FromLongLong(total);
} 
```

这个函数可以通过三种不同的方式调用。当只提供一个可迭代对象时，它将求和可迭代对象并返回值。可选地，我们可以传递一个回调函数，该函数将在求和之前应用于可迭代对象中的每个值。作为第二个可选参数，我们可以指定初始值：

```py
>>> x = range(10)
>>> custom_sum(x)
45
>>> custom_sum(x, lambda y: y + 5)
95
>>> custom_sum(x, lambda y: y + 5, 5)
100 
```

另一个重要的问题是，尽管我们在将值转换为`long long int`时捕获了溢出错误，但此代码仍然不安全。如果我们对两个非常大的数字（接近`long long int`限制）求和，我们仍然会溢出：

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

在这种情况下，您可以通过执行类似`if(value > INT_MAX - total)`的操作来测试这一点，但这个解决方案并不总是适用，因此在使用 C 时，最重要的是要意识到溢出和下溢。

# 练习

外部库的可能性是无限的，所以你可能已经有了关于要实现什么的一些想法。如果没有，这里有一些灵感：

+   尝试使用`ctypes`、`CFFI`和本地扩展对数字列表进行排序。您可以使用`stdlib`中的`qsort`函数。

+   尝试通过添加适当的错误处理来提高我们创建的`custom_sum`函数的安全性，以处理溢出/下溢问题。此外，当对多个数字求和时，如果只有溢出或下溢，需要捕获这些错误。

这些练习应该是利用您新获得的知识做一些有用事情的一个很好的起点。如果您正在寻找更多本地的 C/C++示例，我建议查看 CPython 源代码。有许多示例可供参考：[`github.com/python/cpython/tree/main/Modules`](https://github.com/python/cpython/tree/main/Modules)。我建议从一个相对简单的模块开始，例如`bisect`模块。

这些练习的示例答案可以在 GitHub 上找到：`github.com/mastering-python/exercises`。鼓励您提交自己的解决方案，并从他人的替代方案中学习。

# 摘要

在本章中，您学习了在 C/C++中编写和使用扩展。作为一个快速回顾，我们涵盖了：

+   使用`ctypes`加载外部（系统）库，例如`stdlib`

+   使用`ctypes`和`CFFI`创建和处理复杂的数据结构

+   使用 `ctypes` 和 `CFFI` 处理数组

+   结合 C 和 Python 函数

+   关于数字类型、数组、溢出和其他错误处理的注意事项

尽管你现在可以创建 C/C++ 扩展，但我仍然建议如果可能的话避免使用它们，因为很容易出现错误。即使是本章中的代码示例也没有处理许多可能的错误场景，而且与 Python 中的错误不同，如果这些错误发生在 C 中，它们可能会完全杀死你的解释器或应用程序。

如果你的目标是更好的性能，那么我建议尝试使用 `numba` 或 `cython`。然而，如果你确实需要与非 Python 库进行互操作，这些库是不错的选择。这些通用库的几个例子包括 TensorFlow 和 OpenCV，它们在许多语言中都有可用，并且提供了 Python 封装以方便使用。

在构建本章的示例时，你可能已经注意到我们使用了 `setup.py` 文件并从 `setuptools` 库中导入。这就是下一章将要介绍的内容：将你的代码打包成可安装的 Python 库并在 Python 包索引中分发。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：[`discord.gg/QMzJenHuJf`](https://discord.gg/QMzJenHuJf)

![二维码](img/QR_Code156081100001293319171.png)
