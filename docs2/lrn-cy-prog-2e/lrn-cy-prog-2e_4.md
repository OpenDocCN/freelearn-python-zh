# 第四章：调试 Cython

由于 Cython 程序编译成原生代码，我们无法使用 Python 调试器逐步执行代码。然而，我们可以使用 GDB。**GNU 项目调试器**（**GDB**）是一个跨平台调试器。Python 插件支持从版本 7.0 开始添加，这被用来将 Cython 支持添加到`gdb`中作为一个简单的脚本；这意味着你可以无缝地在 C/C++代码和 Cython 之间逐步执行。

当涉及到语言绑定时，保持接口尽可能简单是一个好的做法。这将使调试变得简单，直到你对资源管理或稳定性方面的绑定满意。我将迭代一些 GDB 和注意事项的例子。

在本章中，我们将涵盖以下主题：

+   使用 GFB 与 Cython

+   Cython 注意事项

# 使用 GDB 与 Cython

要调试 Cython，你需要 GDB >= 7.0。在 Mac OS X Xcode 中，构建工具已移动到 LLVM 和 lldb 作为相应的调试器。你可以使用 homebrew 安装`gdb`：

```py
$ brew install gdb
```

由于 Cython 代码编译成 C/C++，我们无法使用 Python 调试器。因此，在没有 Cython 插件的情况下调试时，你将逐步执行生成的 C/C++代码，这不会很有帮助，因为它不会理解 Cython 程序的环境。

## 运行 cygdb

Cygdb 作为 Cython 的一部分安装，并且是 GDB 的包装器（它通过传递参数调用 GDB 以设置 Cython 插件）。在您能够调试 Cython 代码之前，我们需要生成调试信息。就像 C/C++一样，我们需要指定编译器选项以生成可调试的代码，我们可以在调用 Cython 编译器时传递`–gdb`：

```py
$ cython --gdb cycode.pyx

```

### 注意

在您开始在 Debian 上调试之前，您需要安装 Python 调试信息包和 GDB，因为它们不是与`build-essential`一起安装的。要安装这些，请运行以下命令：

```py
$ sudo apt-get install gdb build-essential cython python-dbg
```

现在你已经安装了 GDB 和生成的调试信息，你可以使用以下命令启动 Cython 调试器：

```py
$ cygdb . --args python-dbg main.py

```

一旦你熟悉了 GDB，你就可以简单地使用所有的正常`gdb`命令。然而，`cygdb`的全部目的在于我们可以使用 Cython 命令，我们将在下面使用并解释：

```py
(gdb) cy break
__init__                cycode.foobar.__init__  cycode.foobar.print_me  cycode.func             func                    print_me

```

如果你使用 Tab 键自动完成`cy break`，你会看到一个可以设置 Cython 断点的符号列表。接下来，我们需要运行程序并继续到我们的断点，如下所示：

**(gdb) cy break func**

```py
Function "__pyx_pw_6cycode_1func" not defined.
Breakpoint 1 (__pyx_pw_6cycode_1func) pending.

```

现在我们已经设置了断点，我们需要运行程序：

```py
(gdb) cy run
1    def func (int x):

```

现在我们已经到达了`func`函数的声明，我们可以继续并做一些内省，如下所示：

```py
(gdb) cy globals
Python globals:
 __builtins__ = <module at remote 0x7ffff7fabb08>
 __doc__      = None
 __file__     = '$HOME/chapter4/gdb1/cycode.so'
 __name__     = 'cycode'
 __package__  = None
 __test__     = {}
 foobar       = <classobj at remote 0x7ffff7ee50b8>
 func         = <built-in function func>
C globals:

```

`globals`命令将显示当前帧作用域中的任何全局标识符，因此我们可以看到`func`函数和`classobj foobar`。我们可以通过列出代码和逐步执行代码来进一步检查：

```py
(gdb) cy list
 1    def func (int x):
 2        print x
 3        return x + 1
 4

```

我们也可以按照以下方式逐步执行代码：

```py
(gdb) cy step
1
4    cycode.func (1)

(gdb) cy list
 1    #!/usr/bin/python
 2    import cycode
 3
 4    cycode.func (1)
>    5    object = cycode.foobar ()
 6    object.print_me ()

(gdb) cy step
3        return x + 1

(gdb) cy list
 1    def func (int x):
 2        print x
>    3        return x + 1
 4
 5    class foobar:
 6        x = 0
 7        def __init__ (self):

```

你甚至可以从类中获得相当整洁的列表：

```py
(gdb) cy list
 3        return x + 1
 4
 5    class foobar:
 6        x = 0
 7        def __init__ (self):
>    8            self.x = 1
 9
 10        def print_me (self):
 11            print self.x

```

我们甚至可以看到当前 Python 状态的后退跟踪：

```py
(gdb) cy bt
#9  0x000000000047b6a0 in <module>() at main.py:6
 6    object.print_me ()
#13 0x00007ffff6a05ea0 in print_me() at /home/redbrain/cython-book/chapter4/gdb1/cycode.pyx:8
 8            self.x = 1

```

帮助信息可以通过运行以下命令找到：

```py
(gdb) help cy

```

我想你已经明白了！尝试一下，查看帮助文档，并亲自尝试这些，以获得使用 cygdb 进行调试的感觉。为了获得良好的感觉，你真的需要通过 GDB 练习并熟悉它。

# Cython 注意事项

当混合 C 和 Python 代码时，Cython 有一些需要注意的注意事项。在构建生产就绪的产品时，参考这些注意事项是个好主意。

## 类型检查

你可能已经注意到，在前面的代码示例中，我们能够使用 `malloc` 将 `void *` 指针转换为我们的扩展类型，使用 `malloc`。Cython 支持一些更高级的类型检查，如下所示：

```py
char * buf = <char *> malloc (sizeof (...))
```

在基本类型转换中，Cython 支持 `<type?>` 用于类型检查：

```py
char * buf  = <char *?> malloc (...)
```

这将进行类型检查，如果被转换的类型不是 `char *` 的子类，则会抛出错误。所以，在这种情况下，它会通过；然而，如果你要这样做：

```py
cdef class A:
     pass
cdef class B (A):
     pass

def myfunc ():
    cdef A class1 = A ()
    cdef B class2 = B ()
 cdef B x = <B?> class1

```

这将在运行时返回一个错误：

```py
Traceback (most recent call last):
  File "main.py", line 2, in <module>
    myfunc ()
  File "cycode.pyx", line 12, in cycode.myfunc (cycode.c:714)
 cdef B x = <B?> class1
TypeError: Cannot convert cycode.A to cycode.B

```

因此，这可以为你的 Cython API 增加更多的类型安全性。

## 取引用运算符 (*)

在 Cython 中，我们没有取引用运算符。例如，如果你要将 C 数组和长度传递给一个函数，你可以使用指针算术来迭代和访问数组的元素：

```py
  int * ptr = array;
  int i;
  for (i = 0; i < len; ++i)
    printf ("%i\n", *ptr++);
```

在 Cython 中，我们必须通过访问元素零来稍微明确一些。然后，我们增加指针：

```py
    cdef int i
    cdef int * ptr = array
    for i in range (len):
        print ptr [0]
        ptr = ptr + 1
```

这里并没有什么特别之处。如果你想取消引用 `int *x`，你只需使用 `x[0]`。

## Python 异常

另一个需要关注的话题是，如果你的 Cython 代码将异常传播到 C 代码中会发生什么。在下一章中，我们将介绍 C++ 原生异常如何与 Python 交互，但在 C 中我们没有这个。考虑以下代码：

```py
cdef public void myfunc ():
    raise Exception ("Raising an exception!")
```

这只是将异常返回到 C，并给出以下内容：

```py
$ ./test
Exception: Exception('Raising an exception!',) in 'cycode.myfunc' ignored
Away doing something else now...

```

如你所见，打印了一个警告，但没有发生异常处理，所以程序继续执行其他操作。这是因为不返回 Python 对象的简单 `cdef` 函数没有处理异常的方法；因此，打印了一个简单的警告消息。如果我们想控制 C 程序的行为，我们需要在 Cython 函数原型中声明异常。

有三种形式可以做到这一点。首先，我们可以这样做：

```py
cdef int myfunc () except -1:
    cdef int retval = -1
     ….
     return retval
```

这使得函数在返回 `-1` 时抛出异常。这也导致异常被传播到调用者；因此，在 Cython 中，我们可以这样做：

```py
cdef public void run ():
    try:
 myfunc ()
        somethingElse ()
    except Exception:
        print "Something wrong"
```

你还可以使用 *maybe* 异常（正如我希望称呼它），其形式如下：

```py
cdef int myfunc () except ? -1:
    cdef int retval = -1
     ….
     return retval
```

这意味着它可能是一个错误，也可能不是。Cython 从 C API 生成对 `PyErr_Occurred` 的调用以执行验证。最后，我们可以使用通配符：

```py
cdef int myfunc () except *:

```

这使得它总是调用 `PyErr_Occurred`，你可以通过 `PyErr_PrintEx` 或其他方式在 [`docs.python.org/2/c-api/exceptions.html`](http://docs.python.org/2/c-api/exceptions.html) 检查。

注意，函数指针声明也可以在它们的原型中处理这个问题。只需确保返回类型与异常类型匹配，该类型必须是枚举、浮点、指针类型或常量表达式；如果不是这种情况，你将得到一个令人困惑的编译错误。

## C/C++ 迭代器

Cython 对 C 风格的 `for` 循环有更多支持，并且它还可以根据迭代器的声明方式对 `range` 函数进行进一步的优化。通常，在 Python 中，你只需做以下操作：

```py
for i in iterable_type: …
```

这在 PyObjects 上是可行的，因为它们理解迭代器，但 C 类型没有这些抽象。你需要对你的数组类型进行指针运算以访问索引。例如，我们首先可以使用 `range` 函数做以下操作：

```py
cdef void myfunc (int length, int * array)
 cdef int i
 for i in range (length):
        print array [i]
```

当在 C 类型上使用范围函数时，例如以下使用 `cdef int i` 的示例，它针对实际的 C 数组访问进行了优化。我们还可以使用其他几种形式。我们可以将循环转换为以下形式：

```py
cdef int i
for i in array [:length]: print i
```

这看起来更像是一个正常的 Python `for` 循环执行迭代，分配 `i`，索引数据。还有一个 Cython 引入的最后一个形式，使用 `for .. from` 语法。这看起来像真正的 C `for` 循环，我们现在可以写：

```py
def myfunc (int length, int * array):
 cdef int i
 for i from 0 <= i < length;
        print array [i]
```

我们还可以引入步长：

```py
for i from 0 <= i < length by 2:
     print array [i]
```

这些额外的 `for` 循环结构在处理大量 C 类型时特别有用，因为它们不理解额外的 Python 结构。

## 布尔错误

当你尝试在 Cython 中使用 `bool` 时，你会得到以下结果：

```py
cycode.pyx:2:9: 'bool' is not a type identifier

```

因此，你需要使用这个：

```py
from libcpp cimport bool
```

当你编译它时，你会得到以下结果：

```py
cycode.c: In function '__pyx_pf_6cycode_run':
cycode.c:642: error: 'bool' undeclared (first use in this function)
cycode.c:642: error: (Each undeclared identifier is reported only once
cycode.c:642: error: for each function it appears in.)
cycode.c:642: error: expected ';' before '__pyx_v_mybool'
cycode.c:657: error: '__pyx_v_mybool' undeclared (first use in this function)

```

你需要确保你使用的是 C++ 编译器进行编译，因为 `bool` 是一个原生类型。

## Const 关键字

Cython 在 0.18 之前不理解 `const` 关键字，但我们可以通过以下 typedefs 来解决这个问题：

```py
cdef extern from *:
 ctypedef char* const_char_ptr "const char*"

```

现在，我们可以像以下这样使用 `const` 关键字：

```py
cdef public void foo_c(const_char_ptr s):
   ...
```

如果你使用的是 Cython 0.18 或更高版本，你可以像从 C 那样使用 `const`。

## 多个 Cython 输入

Cython 不处理多个 `.pyx` 文件。因此，Cython 有另一个关键字和约定——`.pxi`。这是一个额外的包含文件，它就像 C 包含文件一样工作。所有其他 Cython 文件都会被拉入一个文件，以创建一个 Cython 编译。为此，你需要做以下操作：

```py
include "myothercythonfile.pxi"
```

重要的是要记住，这作为一个 C 包含文件工作，并将从文件中包含代码到包含点的代码放入。

## 结构体初始化

当声明 `struct` 时，你不能像以下这样进行正常的 C 初始化：

```py
struct myStruct {
  int x;
  char * y;
}
struct myStruct x = { 2, "bla" };
```

你需要做以下操作：

```py
cdef myStruct x:
x.x = 2
x.y = "bla"
```

因此，你手动更详细地指定字段。所以，当使用结构体时，你应该确保在使用之前使用 memset 或显式设置每个元素。

## 调用纯 Python 模块

你总是可以调用一些纯 Python 代码（非 Cython），但你应该始终保持警惕，并使用 Python `disutils` 确保模块在开发环境之外正确安装。

# 摘要

总体来说，我们已经看到了使用 cygdb 包装器进行的一些基本调试。更重要的是，我们检查了一些 Cython 的注意事项和特性。在下一章中，我们将看到如何从 Cython 直接绑定 C++代码以及如何与 C++构造工作，例如模板和 STL 库，特别是。我们还将看到 GIL 如何影响在 Cython 和 C/C++中与代码的工作。
