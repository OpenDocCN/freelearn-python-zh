# 第七章：其他语言中的 Python 扩展

在编写基于 Python 的应用程序时，您不仅限于 Python 语言。还有一些工具，比如 Hy，在第三章中简要提到，*语法最佳实践-类级别以上*。它允许您使用其他语言（Lisp 的方言）编写模块、包，甚至整个应用程序，这些应用程序将在 Python 虚拟机中运行。尽管它使您能够用完全不同的语法表达程序逻辑，但它仍然是相同的语言，因为它编译成相同的字节码。这意味着它具有与普通 Python 代码相同的限制：

+   由于 GIL 的存在，线程的可用性大大降低

+   它没有被编译

+   它不提供静态类型和可能的优化

帮助克服这些核心限制的解决方案是完全用不同的语言编写的扩展，并通过 Python 扩展 API 公开它们的接口。

本章将讨论使用其他语言编写自己的扩展的主要原因，并向您介绍帮助创建它们的流行工具。您将学到：

+   如何使用 Python/C API 编写简单的 C 扩展

+   如何使用 Cython 做同样的事情

+   扩展引入的主要挑战和问题是什么

+   如何与编译的动态库进行接口，而不创建专用扩展，仅使用 Python 代码

# 不同的语言意味着-C 或 C++

当我们谈论不同语言的扩展时，我们几乎只考虑 C 和 C++。甚至像 Cython 或 Pyrex 这样的工具，它们提供 Python 语言的超集，仅用于扩展的目的，实际上是源到源编译器，从扩展的 Python-like 语法生成 C 代码。

如果只有这样的编译是可能的，那么确实可以在 Python 中使用任何语言编写的动态/共享库，因此它远远超出了 C 和 C++。但共享库本质上是通用的。它们可以在支持它们加载的任何语言中使用。因此，即使您用完全不同的语言（比如 Delphi 或 Prolog）编写这样的库，很难称这样的库为 Python 扩展，如果它不使用 Python/C API。

不幸的是，仅使用裸的 Python/C API 在 C 或 C++中编写自己的扩展是相当苛刻的。这不仅因为它需要对这两种相对难以掌握的语言之一有很好的理解，而且还因为它需要大量的样板文件。有很多重复的代码必须编写，只是为了提供一个接口，将您实现的逻辑与 Python 及其数据类型粘合在一起。无论如何，了解纯 C 扩展是如何构建的是很好的，因为：

+   您将更好地了解 Python 的工作原理

+   有一天，您可能需要调试或维护本机 C/C++扩展

+   它有助于理解构建扩展的高级工具的工作原理

## C 或 C++中的扩展是如何工作的

如果 Python 解释器能够使用 Python/C API 提供适当的接口，它就能从动态/共享库中加载扩展。这个 API 必须被合并到扩展的源代码中，使用与 Python 源代码一起分发的`Python.h` C 头文件。在许多 Linux 发行版中，这个头文件包含在一个单独的软件包中（例如，在 Debian/Ubuntu 中是`python-dev`），但在 Windows 下，默认情况下分发，并且可以在 Python 安装的`includes/`目录中找到。

Python/C API 通常会随着 Python 的每个版本发布而改变。在大多数情况下，这些只是对 API 的新功能的添加，因此通常是源代码兼容的。无论如何，在大多数情况下，它们不是二进制兼容的，因为**应用程序二进制接口**（**ABI**）发生了变化。这意味着扩展必须为每个 Python 版本单独构建。还要注意，不同的操作系统具有不兼容的 ABI，因此这几乎不可能为每种可能的环境创建二进制分发。这就是为什么大多数 Python 扩展以源代码形式分发的原因。

自 Python 3.2 以来，已经定义了 Python/C API 的一个子集，具有稳定的 ABI。因此可以使用这个有限的 API（具有稳定的 ABI）构建扩展，因此扩展只需构建一次，就可以在任何高于或等于 3.2 的 Python 版本上工作，无需重新编译。无论如何，这限制了 API 功能的数量，并且不能解决旧版本 Python 或以二进制形式分发扩展到使用不同操作系统的环境的问题。因此这是一个权衡，稳定 ABI 的代价似乎有点高而收益很低。

你需要知道的一件事是，Python/C API 是限于 CPython 实现的功能。一些努力已经为 PyPI、Jython 或 IronPython 等替代实现带来了扩展支持，但目前似乎没有可行的解决方案。唯一一个应该轻松处理扩展的替代 Python 实现是 Stackless Python，因为它实际上只是 CPython 的修改版本。

Python 的 C 扩展需要在可用之前编译成共享/动态库，因为显然没有本地的方法可以直接从源代码将 C/C++代码导入 Python。幸运的是，`distutils`和`setuptools`提供了帮助，将编译的扩展定义为模块，因此可以使用`setup.py`脚本处理编译和分发，就像它们是普通的 Python 包一样。这是官方文档中处理带有构建扩展的简单包的`setup.py`脚本的一个示例：

```py
from distutils.core import setup, Extension

module1 = Extension(
    'demo',
    sources=['demo.c']
)

setup(
    name='PackageName',
    version='1.0',
    description='This is a demo package',
    ext_modules=[module1]
)
```

准备好之后，你的分发流程还需要一个额外的步骤：

```py
python setup.py build

```

这将根据`ext_modules`参数编译所有你的扩展，根据`Extension()`调用提供的所有额外编译器设置。将使用的编译器是你的环境的默认编译器。如果要分发源代码分发包，则不需要进行这个编译步骤。在这种情况下，你需要确保目标环境具有所有编译的先决条件，例如编译器、头文件和将链接到二进制文件的其他库（如果你的扩展需要）。有关打包 Python 扩展的更多细节将在*挑战*部分中解释。

# 为什么你可能想使用扩展

写 C/C++扩展是否明智的决定并不容易。一般的经验法则可能是，“除非别无选择，否则永远不要”。但这是一个非常主观的说法，留下了很多解释空间，关于在 Python 中做不到的事情。事实上，很难找到一件事情，纯 Python 代码做不到，但有一些问题，扩展可能特别有用：

+   绕过 Python 线程模型中的**全局解释器锁**（**GIL**）

+   改进关键代码部分的性能

+   集成第三方动态库

+   集成用不同语言编写的源代码

+   创建自定义数据类型

例如，核心语言约束，如 GIL，可以通过不同的并发方法轻松克服，例如绿色线程或多进程，而不是线程模型。

## 改进关键代码部分的性能

让我们诚实一点。开发人员选择 Python 并不是因为性能。它执行速度不快，但可以让你快速开发。尽管我们作为程序员有多么高效，多亏了这种语言，有时我们可能会发现一些问题，这些问题可能无法使用纯 Python 有效解决。

在大多数情况下，解决性能问题实际上只是选择合适的算法和数据结构，而不是限制语言开销的常数因子。如果代码已经编写得很差或者没有使用适当的算法，依赖扩展来节省一些 CPU 周期实际上并不是一个好的解决方案。通常情况下，性能可以在不需要通过在堆栈中循环另一种语言来增加项目复杂性的情况下提高到可接受的水平。如果可能的话，应该首先这样做。无论如何，即使使用*最先进*的算法方法和最适合的数据结构，我们也很可能无法仅仅使用 Python 就满足一些任意的技术约束。

将一些对应用程序性能施加了明确定义限制的示例领域是**实时竞价**（**RTB**）业务。简而言之，整个 RTB 都是关于以类似于真实拍卖或证券交易的方式购买和销售广告库存（广告位置）。交易通常通过一些广告交换服务进行，该服务向有兴趣购买它们的**需求方平台**（**DSP**）发送有关可用库存的信息。这就是事情变得令人兴奋的地方。大多数广告交换使用基于 HTTP 的 OpenRTB 协议与潜在竞标者进行通信，其中 DSP 是负责对其 HTTP 请求提供响应的站点。广告交换总是对整个过程施加非常有限的时间限制（通常在 50 到 100 毫秒之间）——从接收到第一个 TPC 数据包到服务器写入的最后一个字节。为了增加趣味，DSP 平台通常每秒处理成千上万个请求并不罕见。能够将请求处理时间推迟几毫秒甚至是这个行业的生死攸关。这意味着即使是将微不足道的代码移植到 C 语言在这种情况下也是合理的，但前提是它是性能瓶颈的一部分，并且在算法上不能进一步改进。正如有人曾经说过的：

> *“你无法击败用 C 语言编写的循环。”*

## 整合不同语言编写的现有代码

在计算机科学的短暂历史中，已经编写了许多有用的库。每次出现新的编程语言时忘记所有这些遗产将是一个巨大的损失，但也不可能可靠地将曾经编写的任何软件完全移植到任何可用的语言。

C 和 C++语言似乎是提供了许多库和实现的最重要的语言，你可能希望在应用程序代码中集成它们，而无需完全将它们移植到 Python。幸运的是，CPython 已经是用 C 编写的，因此通过自定义扩展是集成这样的代码的最自然的方式。

## 集成第三方动态库

使用不同技术编写的代码的集成并不仅限于 C/C++。许多库，特别是具有闭源的第三方软件，都是以编译后的二进制形式分发的。在 C 中，加载这样的共享/动态库并调用它们的函数非常容易。这意味着只要使用 Python/C API 包装它，就可以使用任何 C 库。

当然，这并不是唯一的解决方案，还有诸如`ctypes`或 CFFI 之类的工具，允许您使用纯 Python 与动态库进行交互，而无需编写 C 扩展。通常情况下，Python/C API 可能仍然是更好的选择，因为它在集成层（用 C 编写）和应用程序的其余部分之间提供了更好的分离。

## 创建自定义数据类型

Python 提供了非常多样化的内置数据类型。其中一些真正使用了最先进的内部实现（至少在 CPython 中），专门为在 Python 语言中使用而量身定制。基本类型和可用的集合数量对于新手来说可能看起来令人印象深刻，但显然它并不能涵盖我们所有可能的需求。

当然，您可以通过完全基于一些内置类型或从头开始构建全新类来在 Python 中创建许多自定义数据结构。不幸的是，对于一些可能严重依赖这些自定义数据结构的应用程序来说，性能可能不够。像`dict`或`set`这样的复杂集合的全部功能来自它们的底层 C 实现。为什么不做同样的事情，也在 C 中实现一些自定义数据结构呢？

# 编写扩展

如前所述，编写扩展并不是一项简单的任务，但作为您辛勤工作的回报，它可以给您带来许多优势。编写自己扩展的最简单和推荐的方法是使用诸如 Cython 或 Pyrex 的工具，或者简单地使用`ctypes`或`cffi`集成现有的动态库。这些项目将提高您的生产力，还会使代码更易于开发、阅读和维护。

无论如何，如果您对这个主题还不熟悉，了解一点是好的，即您可以通过仅使用裸 C 代码和 Python/C API 编写一个扩展来开始您的扩展之旅。这将提高您对扩展工作原理的理解，并帮助您欣赏替代解决方案的优势。为了简单起见，我们将以一个简单的算法问题作为示例，并尝试使用三种不同的方法来实现它：

+   编写纯 C 扩展

+   使用 Cython

+   使用 Pyrex

我们的问题将是找到斐波那契数列的第*n*个数字。很少有人会仅为了这个问题创建编译扩展，但它非常简单，因此它将作为将任何 C 函数连接到 Python/C API 的非常好的示例。我们的唯一目标是清晰和简单，因此我们不会试图提供最有效的解决方案。一旦我们知道这一点，我们在 Python 中实现的斐波那契函数的参考实现如下：

```py
"""Python module that provides fibonacci sequence function"""

def fibonacci(n):
    """Return nth Fibonacci sequence number computed recursively.
    """
    if n < 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
```

请注意，这是`fibonnaci()`函数的最简单实现之一，可以对其进行许多改进。尽管如此，我们拒绝改进我们的实现（例如使用记忆化模式），因为这不是我们示例的目的。同样地，即使编译后的代码提供了更多的优化可能性，我们在讨论 C 或 Cython 中的实现时也不会优化我们的代码。

## 纯 C 扩展

在我们完全深入 C 编写的 Python 扩展的代码示例之前，这里有一个重要的警告。如果您想用 C 扩展 Python，您需要已经对这两种语言非常了解。这对于 C 尤其如此。对它的熟练程度不足可能会导致真正的灾难，因为它很容易被误用。

如果您已经决定需要为 Python 编写 C 扩展，我假设您已经对 C 语言有了足够的了解，可以完全理解所呈现的示例。这里将不会解释除 Python/C API 细节之外的任何内容。本书是关于 Python 而不是其他任何语言。如果您根本不懂 C，那么在获得足够的经验和技能之前，绝对不应该尝试用 C 编写自己的 Python 扩展。把它留给其他人，坚持使用 Cython 或 Pyrex，因为从初学者的角度来看，它们更安全得多。这主要是因为 Python/C API，尽管经过精心设计，但绝对不是 C 的良好入门。

如前所述，我们将尝试将`fibonacci()`函数移植到 C 并将其作为扩展暴露给 Python 代码。没有与 Python/C API 连接的裸实现，类似于前面的 Python 示例，大致如下：

```py
long long fibonacci(unsigned int n) {
    if (n < 2) {
        return 1;
    } else {
        return fibonacci(n - 2) + fibonacci(n - 1);
    }
}
```

以下是一个完整、完全功能的扩展的示例，它在编译模块中公开了这个单一函数：

```py
#include <Python.h>

long long fibonacci(unsigned int n) {
    if (n < 2) {
        return 1;
    } else {
        return fibonacci(n-2) + fibonacci(n-1);
    }
}

static PyObject* fibonacci_py(PyObject* self, PyObject* args) {
    PyObject *result = NULL;
    long n;

    if (PyArg_ParseTuple(args, "l", &n)) {
        result = Py_BuildValue("L", fibonacci((unsigned int)n));
    }

    return result;
}

static char fibonacci_docs[] =
    "fibonacci(n): Return nth Fibonacci sequence number "
    "computed recursively\n";

static PyMethodDef fibonacci_module_methods[] = {
    {"fibonacci", (PyCFunction)fibonacci_py,
     METH_VARARGS, fibonacci_docs},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fibonacci_module_definition = {
    PyModuleDef_HEAD_INIT,
    "fibonacci",
    "Extension module that provides fibonacci sequence function",
    -1,
    fibonacci_module_methods
};

PyMODINIT_FUNC PyInit_fibonacci(void) {
    Py_Initialize();

    return PyModule_Create(&fibonacci_module_definition);
}
```

前面的例子乍一看可能有点令人不知所措，因为我们不得不添加四倍的代码才能让`fibonacci()` C 函数可以从 Python 中访问。我们稍后会讨论代码的每一部分，所以不用担心。但在我们讨论之前，让我们看看如何将其打包并在 Python 中执行。我们模块的最小`setuptools`配置需要使用`setuptools.Extension`类来指示解释器如何编译我们的扩展：

```py
from setuptools import setup, Extension

setup(
    name='fibonacci',
    ext_modules=[
        Extension('fibonacci', ['fibonacci.c']),
    ]
)
```

扩展的构建过程可以通过 Python 的`setup.py`构建命令来初始化，但也会在包安装时自动执行。以下是在开发模式下安装的结果以及一个简单的交互会话，我们在其中检查和执行我们编译的`fibonacci()`函数：

```py
$ ls -1a
fibonacci.c
setup.py

$ pip install -e .
Obtaining file:///Users/swistakm/dev/book/chapter7
Installing collected packages: fibonacci
 **Running setup.py develop for fibonacci
Successfully installed Fibonacci

$ ls -1ap
build/
fibonacci.c
fibonacci.cpython-35m-darwin.so
fibonacci.egg-info/
setup.py

$ python
Python 3.5.1 (v3.5.1:37a07cee5969, Dec  5 2015, 21:12:44)** 
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import fibonacci
>>> help(fibonacci.fibonacci)

Help on built-in function fibonacci in fibonacci:

fibonacci.fibonacci = fibonacci(...)
 **fibonacci(n): Return nth Fibonacci sequence number computed recursively

>>> [fibonacci.fibonacci(n) for n in range(10)]
[1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
>>>** 

```

### 对 Python/C API 的更详细了解

由于我们知道如何正确地打包、编译和安装自定义 C 扩展，并且确信它按预期工作，现在是讨论我们的代码的正确时间。

扩展模块以一个包含`Python.h`头文件的单个 C 预处理指令开始：

```py
#include <Python.h>
```

这将引入整个 Python/C API，并且是您需要包含的一切，以便能够编写您的扩展。在更现实的情况下，您的代码将需要更多的预处理指令，以从 C 标准库函数中获益或集成其他源文件。我们的示例很简单，因此不需要更多的指令。

接下来是我们模块的核心：

```py
long long fibonacci(unsigned int n) {
    if (n < 2) {
        return 1;
    } else {
        return fibonacci(n - 2) + fibonacci(n - 1);
    }
}
```

前面的`fibonacci()`函数是我们代码中唯一有用的部分。它是纯 C 实现，Python 默认情况下无法理解。我们的示例的其余部分将创建接口层，通过 Python/C API 将其暴露出来。

将此代码暴露给 Python 的第一步是创建与 CPython 解释器兼容的 C 函数。在 Python 中，一切都是对象。这意味着在 Python 中调用的 C 函数也需要返回真正的 Python 对象。Python/C API 提供了`PyObject`类型，每个可调用函数都必须返回指向它的指针。我们函数的签名是：

```py
static PyObject* fibonacci_py(PyObject* self, PyObject* args)s
```

请注意，前面的签名并未指定确切的参数列表，而只是`PyObject* args`，它将保存指向包含提供的值元组的结构的指针。参数列表的实际验证必须在函数体内执行，这正是`fibonacci_py()`所做的。它解析`args`参数列表，假设它是单个`unsigned int`类型，并将该值用作`fibonacci()`函数的参数来检索斐波那契数列元素：

```py
static PyObject* fibonacci_py(PyObject* self, PyObject* args) {
    PyObject *result = NULL;
    long n;

    if (PyArg_ParseTuple(args, "l", &n)) {
        result = Py_BuildValue("L", fibonacci((unsigned int)n));
    }

    return result;
}
```

### 注意

前面的示例函数有一些严重的错误，有经验的开发人员的眼睛应该很容易发现。尝试找到它，作为使用 C 扩展的练习。现在，为了简洁起见，我们将它保留下来。在*异常处理*部分讨论处理错误的细节时，我们将尝试稍后修复它。

`"l"`字符串在`PyArg_ParseTuple(args, "l", &n)`调用中意味着我们希望`args`只包含一个`long`值。如果失败，它将返回`NULL`并在每个线程的解释器状态中存储有关异常的信息。关于异常处理的详细信息将在*异常处理*部分稍后描述。

解析函数的实际签名是`int PyArg_ParseTuple(PyObject *args, const char *format, ...)`，在`format`字符串之后的是一个可变长度的参数列表，表示解析值输出（作为指针）。这类似于 C 标准库中的`scanf()`函数的工作方式。如果我们的假设失败，用户提供了不兼容的参数列表，那么`PyArg_ParseTuple()`将引发适当的异常。一旦你习惯了这种方式，这是一种非常方便的编码函数签名的方式，但与纯 Python 代码相比，它有一个巨大的缺点。由`PyArg_ParseTuple()`调用隐式定义的这种 Python 调用签名在 Python 解释器内部不能轻松地检查。在使用作为扩展提供的代码时，您需要记住这一点。

如前所述，Python 期望从可调用对象返回对象。这意味着我们不能将从`fibonacci()`函数获得的`long long`值作为`fibonacci_py()`的结果返回。这样的尝试甚至不会编译，基本 C 类型不会自动转换为 Python 对象。必须使用`Py_BuildValue(*format, ...)`函数。它是`PyArg_ParseTuple()`的对应物，并接受类似的格式字符串集。主要区别在于参数列表不是函数输出而是输入，因此必须提供实际值而不是指针。

在定义了`fibonacci_py()`之后，大部分繁重的工作都已完成。最后一步是执行模块初始化并向我们的函数添加元数据，这将使用户的使用变得更简单一些。这是我们扩展代码的样板部分，对于一些简单的例子，比如这个例子，可能会占用比我们想要公开的实际函数更多的空间。在大多数情况下，它只是由一些静态结构和一个初始化函数组成，该函数将由解释器在模块导入时执行。

首先，我们创建一个静态字符串，它将成为`fibonacci_py()`函数的 Python 文档字符串的内容：

```py
static char fibonacci_docs[] =
    "fibonacci(n): Return nth Fibonacci sequence number "
    "computed recursively\n";
```

请注意，这可能会*内联*在`fibonacci_module_methods`的某个地方，但将文档字符串分开并存储在与其引用的实际函数定义的附近是一个很好的做法。

我们定义的下一部分是`PyMethodDef`结构的数组，该数组定义了将在我们的模块中可用的方法（函数）。该结构包含四个字段：

+   `char* ml_name`: 这是方法的名称。

+   `PyCFunction ml_meth`: 这是指向函数的 C 实现的指针。

+   `int ml_flags`: 这包括指示调用约定或绑定约定的标志。后者仅适用于定义类方法。

+   `char* ml_doc`: 这是指向方法/函数文档字符串内容的指针。

这样的数组必须始终以`{NULL, NULL, 0, NULL}`的哨兵值结束，表示其结束。在我们的简单情况下，我们创建了`static PyMethodDef fibonacci_module_methods[]`数组，其中只包含两个元素（包括哨兵值）：

```py
static PyMethodDef fibonacci_module_methods[] = {
    {"fibonacci", (PyCFunction)fibonacci_py,
     METH_VARARGS, fibonacci_docs},
    {NULL, NULL, 0, NULL}
};
```

这就是第一个条目如何映射到`PyMethodDef`结构：

+   `ml_name = "fibonacci"`: 在这里，`fibonacci_py()` C 函数将以`fibonacci`名称作为 Python 函数公开

+   `ml_meth = (PyCFunction)fibonacci_py`: 在这里，将`PyCFunction`转换仅仅是 Python/C API 所需的，并且由`ml_flags`中定义的调用约定决定

+   `ml_flags = METH_VARARGS`: 在这里，`METH_VARARGS`标志表示我们的函数的调用约定接受可变参数列表，不接受关键字参数

+   `ml_doc = fibonacci_docs`: 在这里，Python 函数将使用`fibonacci_docs`字符串的内容进行文档化

当函数定义数组完成时，我们可以创建另一个结构，其中包含整个模块的定义。它使用`PyModuleDef`类型进行描述，并包含多个字段。其中一些仅适用于需要对模块初始化过程进行细粒度控制的更复杂的情况。在这里，我们只对其中的前五个感兴趣：

+   `PyModuleDef_Base m_base`: 这应该始终用`PyModuleDef_HEAD_INIT`进行初始化。

+   `char* m_name`: 这是新创建模块的名称。在我们的例子中是`fibonacci`。

+   `char* m_doc`: 这是模块的文档字符串内容的指针。通常在一个 C 源文件中只定义一个模块，因此将我们的文档字符串内联在整个结构中是可以的。

+   `Py_ssize_t m_size`: 这是分配给保持模块状态的内存的大小。只有在需要支持多个子解释器或多阶段初始化时才会使用。在大多数情况下，您不需要它，它的值为`-1`。

+   `PyMethodDef* m_methods`: 这是指向包含由`PyMethodDef`值描述的模块级函数的数组的指针。如果模块不公开任何函数，则可以为`NULL`。在我们的情况下，它是`fibonacci_module_methods`。

其他字段在官方 Python 文档中有详细解释（参考[`docs.python.org/3/c-api/module.html`](https://docs.python.org/3/c-api/module.html)），但在我们的示例扩展中不需要。如果不需要，它们应该设置为`NULL`，当未指定时，它们将隐式地初始化为该值。这就是为什么我们的模块描述包含在`fibonacci_module_definition`变量中可以采用这种简单的五元素形式的原因：

```py
static struct PyModuleDef fibonacci_module_definition = {
    PyModuleDef_HEAD_INIT,
    "fibonacci",
    "Extension module that provides fibonacci sequence function",
    -1,
    fibonacci_module_methods
};
```

最后一段代码是我们工作的巅峰，即模块初始化函数。这必须遵循非常特定的命名约定，以便 Python 解释器在加载动态/共享库时可以轻松地选择它。它应该被命名为`PyInit_name`，其中*name*是您的模块名称。因此，它与在`PyModuleDef`定义中用作`m_base`字段和`setuptools.Extension()`调用的第一个参数的字符串完全相同。如果您不需要对模块进行复杂的初始化过程，它将采用与我们示例中完全相同的非常简单的形式：

```py
PyMODINIT_FUNC PyInit_fibonacci(void) {
    return PyModule_Create(&fibonacci_module_definition);
}
```

`PyMODINIT_FUNC`宏是一个预处理宏，它将声明此初始化函数的返回类型为`PyObject*`，并根据平台需要添加任何特殊的链接声明。

### 调用和绑定约定

如*深入了解 Python/C API*部分所述，`PyMethodDef`结构的`ml_flags`位字段包含调用和绑定约定的标志。**调用约定标志**包括：

+   `METH_VARARGS`: 这是 Python 函数或方法的典型约定，只接受参数作为其参数。对于这样的函数，`ml_meth`字段提供的类型应该是`PyCFunction`。该函数将提供两个`PyObject*`类型的参数。第一个要么是`self`对象（对于方法），要么是`module`对象（对于模块函数）。具有该调用约定的 C 函数的典型签名是`PyObject* function(PyObject* self, PyObject* args)`。

+   `METH_KEYWORDS`：这是 Python 函数在调用时接受关键字参数的约定。其关联的 C 类型是`PyCFunctionWithKeywords`。C 函数必须接受三个`PyObject*`类型的参数：`self`，`args`和关键字参数的字典。如果与`METH_VARARGS`组合，前两个参数的含义与前一个调用约定相同，否则`args`将为`NULL`。典型的 C 函数签名是：`PyObject* function(PyObject* self, PyObject* args, PyObject* keywds)`。

+   `METH_NOARGS`：这是 Python 函数不接受任何其他参数的约定。C 函数应该是`PyCFunction`类型，因此签名与`METH_VARARGS`约定相同（两个`self`和`args`参数）。唯一的区别是`args`将始终为`NULL`，因此不需要调用`PyArg_ParseTuple()`。这不能与任何其他调用约定标志组合。

+   `METH_O`：这是接受单个对象参数的函数和方法的简写。C 函数的类型再次是`PyCFunction`，因此它接受两个`PyObject*`参数：`self`和`args`。它与`METH_VARARGS`的区别在于不需要调用`PyArg_ParseTuple()`，因为作为`args`提供的`PyObject*`将已经表示在 Python 调用该函数时提供的单个参数。这也不能与任何其他调用约定标志组合。

接受关键字的函数可以用`METH_KEYWORDS`或者`METH_VARARGS |` `METH_KEYWORDS`的形式来描述。如果是这样，它应该使用`PyArg_ParseTupleAndKeywords()`来解析它的参数，而不是`PyArg_ParseTuple()`或者`PyArg_UnpackTuple()`。下面是一个示例模块，其中有一个返回`None`的函数，接受两个命名关键字参数，并将它们打印到标准输出：

```py
#include <Python.h>

static PyObject* print_args(PyObject *self, PyObject *args, PyObject *keywds)
{
    char *first;
    char *second;

    static char *kwlist[] = {"first", "second", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss", kwlist,
                                     &first, &second))
        return NULL;

    printf("%s %s\n", first, second);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef module_methods[] = {
    {"print_args", (PyCFunction)print_args,
     METH_VARARGS | METH_KEYWORDS,
     "print provided arguments"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "kwargs",
    "Keyword argument processing example",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_kwargs(void) {
    return PyModule_Create(&module_definition);
}
```

Python/C API 中的参数解析非常灵活，并且在官方文档中有详细描述。`PyArg_ParseTuple()`和`PyArg_ParseTupleAndKeywords()`中的格式参数允许对参数数量和类型进行精细的控制。Python 中已知的每个高级调用约定都可以使用此 API 在 C 中编码，包括：

+   带有默认参数值的函数

+   指定为关键字参数的函数

+   带有可变数量参数的函数

**绑定约定标志**是`METH_CLASS`，`METH_STATIC`和`METH_COEXIST`，它们保留给方法，并且不能用于描述模块函数。前两个相当不言自明。它们是`classmethod`和`staticmethod`装饰器的 C 对应物，并且改变了传递给 C 函数的`self`参数的含义。

`METH_COEXIST`允许在现有定义的位置加载一个方法。这很少有用。这主要是当您想要提供一个从已定义的类型的其他特性自动生成的 C 方法的实现时。Python 文档给出了`__contains__()`包装器方法的示例，如果类型定义了`sq_contains`槽，它将自动生成。不幸的是，使用 Python/C API 定义自己的类和类型超出了本入门章节的范围。在讨论 Cython 时，我们将在以后讨论创建自己的类型，因为在纯 C 中这样做需要太多样板代码，并且容易出错。

### 异常处理

与 Python 甚至 C++不同，C 没有语法来引发和捕获异常。所有错误处理通常通过函数返回值和可选的全局状态来处理，用于存储可以解释最后一次失败原因的细节。

Python/C API 中的异常处理建立在这个简单原则的基础上。有一个全局的每个线程指示器，用于描述 C API 中发生的最后一个错误。它被设置为描述问题的原因。还有一种标准化的方法，用于在调用期间通知函数的调用者是否更改了此状态：

+   如果函数应返回指针，则返回`NULL`

+   如果函数应返回`int`类型，则返回`-1`

在 Python/C API 中，前述规则的唯一例外是返回`1`表示成功，返回`0`表示失败的`PyArg_*（）`函数。

为了了解这在实践中是如何工作的，让我们回顾一下前几节中示例中的`fibonacci_py（）`函数：

```py
static PyObject* fibonacci_py(PyObject* self, PyObject* args) {
 **PyObject *result = NULL;
    long n;

 **if (PyArg_ParseTuple(args, "l", &n)) {
 **result = Py_BuildValue("L", fibonacci((unsigned int) n));
    }

 **return result;
}
```

以某种方式参与我们的错误处理的行已经被突出显示。它从初始化`result`变量开始，该变量应存储我们函数的返回值。它被初始化为`NULL`，正如我们已经知道的那样，这是一个错误指示器。这通常是您编写扩展的方式，假设错误是代码的默认状态。

稍后，我们有`PyArg_ParseTuple（）`调用，如果发生异常，将设置错误信息并返回`0`。这是`if`语句的一部分，在这种情况下，我们不做任何其他操作并返回`NULL`。调用我们的函数的人将收到有关错误的通知。

`Py_BuildValue（）`也可能引发异常。它应返回`PyObject*`（指针），因此在失败的情况下会返回`NULL`。我们可以简单地将其存储为我们的结果变量，并将其作为返回值传递。

但我们的工作并不仅仅是关心 Python/C API 调用引发的异常。很可能您需要通知扩展用户发生了其他类型的错误或失败。Python/C API 有多个函数可帮助您引发异常，但最常见的是`PyErr_SetString（）`。它使用提供的附加字符串设置错误指示器和给定的异常类型作为错误原因的解释。此函数的完整签名是：

```py
void PyErr_SetString(PyObject* type, const char* message)
```

我已经说过我们的`fibonacci_py（）`函数的实现存在严重错误。现在是修复它的正确时机。幸运的是，我们有适当的工具来做到这一点。问题在于在以下行中将`long`类型不安全地转换为`unsigned int`：

```py
    if (PyArg_ParseTuple(args, "l", &n)) {
      result = Py_BuildValue("L", fibonacci((unsigned int) n));
    }
```

感谢`PyArg_ParseTuple（）`调用，第一个且唯一的参数将被解释为`long`类型（`"l"`指定符），并存储在本地`n`变量中。然后将其转换为`unsigned int`，因此如果用户使用负值从 Python 调用`fibonacci（）`函数，则会出现问题。例如，作为有符号 32 位整数的`-1`在转换为无符号 32 位整数时将被解释为`4294967295`。这样的值将导致深度递归，并导致堆栈溢出和分段错误。请注意，如果用户提供任意大的正参数，也可能会发生相同的情况。我们无法在没有完全重新设计 C `fibonacci（）`函数的情况下解决这个问题，但至少我们可以尝试确保传递的参数满足一些先决条件。在这里，我们检查`n`参数的值是否大于或等于零，如果不是，则引发`ValueError`异常：

```py
static PyObject* fibonacci_py(PyObject* self, PyObject* args) {
    PyObject *result = NULL;
    long n;
    long long fib;

    if (PyArg_ParseTuple(args, "l", &n)) {
        if (n<0) {
            PyErr_SetString(PyExc_ValueError,
                            "n must not be less than 0");
        } else {
            result = Py_BuildValue("L", fibonacci((unsigned int)n));
        }
    }

    return result;
}
```

最后一点是全局错误状态不会自行清除。您的 C 函数中可能会优雅地处理一些错误（就像在 Python 中使用`try ... except`子句一样），如果错误指示器不再有效，则需要能够清除错误指示器。用于此目的的函数是`PyErr_Clear（）`。

### 释放 GIL

我已经提到扩展可以是绕过 Python GIL 的一种方法。CPython 实现有一个著名的限制，即一次只能有一个线程执行 Python 代码。虽然多进程是绕过这个问题的建议方法，但对于一些高度可并行化的算法来说，由于运行额外进程的资源开销，这可能不是一个好的解决方案。

因为扩展主要用于在纯 C 中执行大部分工作而没有调用 Python/C API 的情况下，所以在一些应用程序部分释放 GIL 是可能的（甚至是建议的）。由于这一点，您仍然可以从拥有多个 CPU 核心和多线程应用程序设计中受益。您唯一需要做的就是使用 Python/C API 提供的特定宏将已知不使用任何 Python/C API 调用或 Python 结构的代码块进行包装。这两个预处理器宏旨在简化释放和重新获取全局解释器锁的整个过程：

+   `Py_BEGIN_ALLOW_THREADS`：这声明了隐藏的本地变量，保存了当前线程状态并释放了 GIL

+   `Py_END_ALLOW_THREADS`：这重新获取 GIL 并从使用前一个宏声明的本地变量恢复线程状态

当我们仔细观察我们的`fibonacci`扩展示例时，我们可以清楚地看到`fibonacci()`函数不执行任何 Python 代码，也不触及任何 Python 结构。这意味着简单包装`fibonacci(n)`执行的`fibonacci_py()`函数可以更新以在调用周围释放 GIL：

```py
static PyObject* fibonacci_py(PyObject* self, PyObject* args) {
    PyObject *result = NULL;
    long n;
    long long fib;

    if (PyArg_ParseTuple(args, "l", &n)) {
        if (n<0) {
            PyErr_SetString(PyExc_ValueError,
                            "n must not be less than 0");
        } else {
            Py_BEGIN_ALLOW_THREADS;
            fib = fibonacci(n);
            Py_END_ALLOW_THREADS;

            result = Py_BuildValue("L", fib);
        }}

    return result;
}
```

### 引用计数

最后，我们来到了 Python 中内存管理的重要主题。Python 有自己的垃圾回收器，但它只设计用来解决**引用计数**算法中的循环引用问题。引用计数是管理不再需要的对象的释放的主要方法。

Python/C API 文档引入了*引用的所有权*来解释它如何处理对象的释放。Python 中的对象从不被拥有，它们总是被共享。对象的实际创建由 Python 的内存管理器管理。这是 CPython 解释器的一个组件，负责为存储在私有堆中的对象分配和释放内存。可以拥有的是对对象的引用。

Python 中的每个对象，由一个引用（`PyObject*`指针）表示，都有一个关联的引用计数。当引用计数为零时，意味着没有人持有对象的有效引用，可以调用与其类型相关联的解分配器。Python/C API 提供了两个宏来增加和减少引用计数：`Py_INCREF()`和`Py_DECREF()`。但在讨论它们的细节之前，我们需要了解与引用所有权相关的一些术语：

+   **所有权的传递**：每当我们说函数*传递了对引用的所有权*时，这意味着它已经增加了引用计数，调用者有责任在不再需要对象的引用时减少计数。大多数返回新创建对象的函数，比如`Py_BuildValue`，都会这样做。如果该对象将从我们的函数返回给另一个调用者，那么所有权会再次传递。在这种情况下，我们不会减少引用计数，因为这不再是我们的责任。这就是为什么`fibonacci_py()`函数不在`result`变量上调用`Py_DECREF()`的原因。

+   **借用引用**：*借用*引用发生在函数将某个 Python 对象的引用作为参数接收时。在该函数中，除非在其范围内明确增加了引用计数，否则不应该减少此类引用的引用计数。在我们的`fibonacci_py()`函数中，`self`和`args`参数就是这样的借用引用，因此我们不对它们调用`PyDECREF()`。Python/C API 的一些函数也可能返回借用引用。值得注意的例子是`PyTuple_GetItem()`和`PyList_GetItem()`。通常说这样的引用是*不受保护*的。除非它将作为函数的返回值返回，否则不需要释放其所有权。在大多数情况下，如果我们将这样的借用引用用作其他 Python/C API 调用的参数，就需要额外小心。在某些情况下，可能需要在将其用作其他函数的参数之前，额外使用`Py_INCREF()`来保护这样的引用，然后在不再需要时调用`Py_DECREF()`。

+   **窃取引用**：Python/C API 函数还可以在提供为调用参数时*窃取*引用，而不是*借用*引用。这是确切的两个函数的情况：`PyTuple_SetItem()`和`PyList_SetItem()`。它们完全承担了传递给它们的引用的责任。它们本身不增加引用计数，但在不再需要引用时会调用`Py_DECREF()`。

在编写复杂的扩展时，监视引用计数是最困难的事情之一。一些不那么明显的问题可能直到在多线程设置中运行代码时才会被注意到。

另一个常见的问题是由 Python 对象模型的本质和一些函数返回借用引用的事实引起的。当引用计数变为零时，将执行解分配函数。对于用户定义的类，可以定义一个`__del__()`方法，在那时将被调用。这可以是任何 Python 代码，可能会影响其他对象及其引用计数。官方 Python 文档给出了以下可能受到此问题影响的代码示例：

```py
void bug(PyObject *list) {
    PyObject *item = PyList_GetItem(list, 0);

    PyList_SetItem(list, 1, PyLong_FromLong(0L));
    PyObject_Print(item, stdout, 0); /* BUG! */
}
```

看起来完全无害，但问题实际上是我们无法知道`list`对象包含哪些元素。当`PyList_SetItem()`在`list[1]`索引上设置一个新值时，之前存储在该索引处的对象的所有权被处理。如果它是唯一存在的引用，引用计数将变为 0，并且对象将被解分配。可能是某个用户定义的类，具有`__del__()`方法的自定义实现。如果在这样的`__del__()`执行的结果中，`item[0]`将从列表中移除，将会出现严重问题。请注意，`PyList_GetItem()`返回一个*借用*引用！在返回引用之前，它不会调用`Py_INCREF()`。因此，在该代码中，可能会调用`PyObject_Print()`，并且会使用一个不再存在的对象的引用。这将导致分段错误并使 Python 解释器崩溃。

正确的方法是在我们需要它们的整个时间内保护借用引用，因为有可能在其中的任何调用可能导致任何其他对象的解分配，即使它们看似无关：

```py
void no_bug(PyObject *list) {
    PyObject *item = PyList_GetItem(list, 0);

    Py_INCREF(item);
    PyList_SetItem(list, 1, PyLong_FromLong(0L));
    PyObject_Print(item, stdout, 0);
    Py_DECREF(item);
}
```

## Cython

Cython 既是一个优化的静态编译器，也是 Python 的超集编程语言的名称。作为编译器，它可以对本地 Python 代码和其 Cython 方言进行*源到源*编译，使用 Python/C API 将其转换为 Python C 扩展。它允许您结合 Python 和 C 的强大功能，而无需手动处理 Python/C API。

### Cython 作为源到源编译器

使用 Cython 创建的扩展的主要优势是可以使用它提供的超集语言。无论如何，也可以使用*源到源*编译从纯 Python 代码创建扩展。这是 Cython 的最简单方法，因为它几乎不需要对代码进行任何更改，并且可以在非常低的开发成本下获得一些显著的性能改进。

Cython 提供了一个简单的`cythonize`实用函数，允许您轻松地将编译过程与`distutils`或`setuptools`集成。假设我们想将`fibonacci()`函数的纯 Python 实现编译为 C 扩展。如果它位于`fibonacci`模块中，最小的`setup.py`脚本可能如下所示：

```py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='fibonacci',
    ext_modules=cythonize(['fibonacci.py'])
)
```

Cython 作为 Python 语言的源编译工具还有另一个好处。源到源编译到扩展可以是源分发安装过程的完全可选部分。如果需要安装包的环境没有 Cython 或任何其他构建先决条件，它可以像普通的*纯 Python*包一样安装。用户不应该注意到以这种方式分发的代码行为上的任何功能差异。

使用 Cython 构建的扩展的常见方法是包括 Python/Cython 源代码和从这些源文件生成的 C 代码。这样，该包可以根据构建先决条件的存在以三种不同的方式安装：

+   如果安装环境中有 Cython 可用，则会从提供的 Python/Cython 源代码生成扩展 C 代码。

+   如果 Cython 不可用，但存在构建先决条件（C 编译器，Python/C API 头文件），则从分发的预生成 C 文件构建扩展。

+   如果前述的先决条件都不可用，但扩展是从纯 Python 源创建的，则模块将像普通的 Python 代码一样安装，并且跳过编译步骤。

请注意，Cython 文档表示，包括生成的 C 文件以及 Cython 源是分发 Cython 扩展的推荐方式。同样的文档表示，Cython 编译应该默认禁用，因为用户可能在他的环境中没有所需版本的 Cython，这可能导致意外的编译问题。无论如何，随着环境隔离的出现，这似乎是一个今天不太令人担忧的问题。此外，Cython 是一个有效的 Python 包，可以在 PyPI 上获得，因此可以很容易地在特定版本中定义为您项目的要求。当然，包括这样的先决条件是一个具有严重影响的决定，应该非常谨慎地考虑。更安全的解决方案是利用`setuptools`包中的`extras_require`功能的强大功能，并允许用户决定是否要使用特定环境变量来使用 Cython：

```py
import os

from distutils.core import setup
from distutils.extension import Extension

try:
    # cython source to source compilation available
    # only when Cython is available
    import Cython
    # and specific environment variable says
    # explicitely that Cython should be used
    # to generate C sources
    USE_CYTHON = bool(os.environ.get("USE_CYTHON"))

except ImportError:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("fibonacci", ["fibonacci"+ext])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='fibonacci',
    ext_modules=extensions,
    extras_require={
        # Cython will be set in that specific version
        # as a requirement if package will be intalled
        # with '[with-cython]' extra feature
        'cython': ['cython==0.23.4']
    }
)
```

`pip`安装工具支持通过在包名后添加`[extra-name]`后缀来使用*extras*选项安装包。对于前面的示例，可以使用以下命令启用从本地源安装时的可选 Cython 要求和编译：

```py
$ USE_CYTHON=1 pip install .[with-cython]

```

### Cython 作为一种语言

Cython 不仅是一个编译器，还是 Python 语言的超集。超集意味着任何有效的 Python 代码都是允许的，并且可以进一步更新为具有额外功能的代码，例如支持调用 C 函数或在变量和类属性上声明 C 类型。因此，任何用 Python 编写的代码也是用 Cython 编写的。这解释了为什么普通的 Python 模块可以如此轻松地使用 Cython 编译为 C。

但我们不会停留在这个简单的事实上。我们将尝试对我们的参考`fibonacci()`函数进行一些改进，而不是说它也是 Python 的超集中有效扩展的代码。这不会对我们的函数设计进行任何真正的优化，而是一些小的更新，使它能够从在 Cython 中编写的好处中受益。

Cython 源文件使用不同的文件扩展名。它是`.pyx`而不是`.py`。假设我们仍然想要实现我们的 Fibbonacci 序列。`fibonacci.pyx`的内容可能如下所示：

```py
"""Cython module that provides fibonacci sequence function."""

def fibonacci(unsigned int n):
    """Return nth Fibonacci sequence number computed recursively."""
    if n < 2:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
```

正如您所看到的，真正改变的只是`fibonacci()`函数的签名。由于 Cython 中的可选静态类型，我们可以将`n`参数声明为`unsigned int`，这应该稍微改进了我们函数的工作方式。此外，它比我们以前手工编写扩展时做的事情要多得多。如果 Cython 函数的参数声明为静态类型，则扩展将自动处理转换和溢出错误，引发适当的异常：

```py
>>> from fibonacci import fibonacci
>>> fibonacci(5)
5
>>> fibonacci(-1)
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "fibonacci.pyx", line 21, in fibonacci.fibonacci (fibonacci.c:704)
OverflowError: can't convert negative value to unsigned int
>>> fibonacci(10 ** 10)
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "fibonacci.pyx", line 21, in fibonacci.fibonacci (fibonacci.c:704)
OverflowError: value too large to convert to unsigned int

```

我们已经知道 Cython 只编译*源到源*，生成的代码使用与我们手工编写 C 代码扩展时相同的 Python/C API。请注意，`fibonacci()`是一个递归函数，因此它经常调用自身。这意味着尽管我们为输入参数声明了静态类型，在递归调用期间，它将像任何其他 Python 函数一样对待自己。因此，`n-1`和`n-2`将被打包回 Python 对象，然后传递给内部`fibonacci()`实现的隐藏包装层，再次将其转换为`unsigned int`类型。这将一次又一次地发生，直到我们达到递归的最终深度。这不一定是一个问题，但涉及到比实际需要的更多的参数处理。

我们可以通过将更多的工作委托给一个纯 C 函数来削减 Python 函数调用和参数处理的开销。我们以前在使用纯 C 创建 C 扩展时就这样做过，我们在 Cython 中也可以这样做。我们可以使用`cdef`关键字声明只接受和返回 C 类型的 C 风格函数：

```py
cdef long long fibonacci_cc(unsigned int n):
    if n < 2:
        return n
    else:
        return fibonacci_cc(n - 1) + fibonacci_cc(n - 2)

def fibonacci(unsigned int n):
    """ Return nth Fibonacci sequence number computed recursively
    """
    return fibonacci_cc(n)
```

我们甚至可以走得更远。通过一个简单的 C 示例，我们最终展示了如何在调用我们的纯 C 函数时释放 GIL，因此扩展对多线程应用程序来说更加友好。在以前的示例中，我们使用了 Python/C API 头文件中的`Py_BEGIN_ALLOW_THREADS`和`Py_END_ALLOW_THREADS`预处理器宏来标记代码段为无需 Python 调用。Cython 语法要简短得多，更容易记住。可以使用简单的`with nogil`语句在代码段周围释放 GIL：

```py
def fibonacci(unsigned int n):
    """ Return nth Fibonacci sequence number computed recursively
    """
 **with nogil:
        result = fibonacci_cc(n)

    return fibonacci_cc(n)
```

您还可以将整个 C 风格函数标记为无需 GIL 即可调用：

```py
cdef long long fibonacci_cc(unsigned int n) nogil:
    if n < 2:
        return n
    else:
        return fibonacci_cc(n - 1) + fibonacci_cc(n - 2)
```

重要的是要知道，这样的函数不能将 Python 对象作为参数或返回类型。每当标记为`nogil`的函数需要执行任何 Python/C API 调用时，它必须使用`with gil`语句获取 GIL。

# 挑战

老实说，我之所以开始接触 Python，只是因为我厌倦了用 C 和 C++编写软件的所有困难。事实上，程序员们意识到其他语言无法满足用户需求时，很常见的是开始学习 Python。与 C、C++或 Java 相比，用 Python 编程是一件轻而易举的事情。一切似乎都很简单而且设计良好。你可能会认为没有地方会让你绊倒，也不再需要其他编程语言了。

当然，这种想法是错误的。是的，Python 是一种令人惊叹的语言，具有许多很酷的功能，并且在许多领域中被使用。但这并不意味着它是完美的，也没有任何缺点。它易于理解和编写，但这种简单性是有代价的。它并不像许多人认为的那样慢，但永远不会像 C 那样快。它高度可移植，但它的解释器并不像其他语言的编译器那样在许多架构上都可用。我们可以永远列出这样的列表。

解决这个问题的一个方法是编写扩展，这样我们就可以将*好老的 C*的一些优点带回 Python。在大多数情况下，这样做效果很好。问题是：我们真的是因为想用 C 来扩展 Python 吗？答案是否定的。这只是在我们没有更好选择的情况下的一种不便的必要性。

## 额外的复杂性

毫无秘密，用许多不同的语言开发应用程序并不是一件容易的事情。Python 和 C 是完全不同的技术，很难找到它们共同之处。同样真实的是没有一个应用程序是没有 bug 的。如果在你的代码库中扩展变得很常见，调试可能会变得痛苦。不仅因为调试 C 代码需要完全不同的工作流程和工具，而且因为你需要经常在两种不同的语言之间切换上下文。

我们都是人类，都有有限的认知能力。当然，有些人可以有效地处理多层抽象和技术堆栈，但他们似乎是非常罕见的。无论你有多么有技巧，对于维护这样的混合解决方案，总是需要额外付出代价。这要么涉及额外的努力和时间来在 C 和 Python 之间切换，要么涉及额外的压力，最终会使你效率降低。

根据 TIOBE 指数，C 仍然是最流行的编程语言之一。尽管事实如此，Python 程序员很常见地对它知之甚少，甚至几乎一无所知。就我个人而言，我认为 C 应该是编程世界的*通用语言*，但我的观点在这个问题上很不可能改变任何事情。Python 也是如此诱人和易学，以至于许多程序员忘记了他们以前的所有经验，完全转向了新技术。而编程不像骑自行车。如果不经常使用和充分磨练，这种特定的技能会更快地消失。即使是具有扎实 C 背景的程序员，如果决定长时间深入 Python，也会逐渐失去他们以前的知识。以上所有情况都导致一个简单的结论——很难找到能够理解和扩展你的代码的人。对于开源软件包，这意味着更少的自愿贡献者。对于闭源软件，这意味着并非所有的队友都能够在不破坏东西的情况下开发和维护扩展。

## 调试

当涉及到失败时，扩展可能会出现严重故障。静态类型给你比 Python 更多的优势，并允许你在编译步骤中捕获很多问题，这些问题在 Python 中很难注意到，除非进行严格的测试例程和全面的测试覆盖。另一方面，所有内存管理必须手动执行。错误的内存管理是 C 中大多数编程错误的主要原因。在最好的情况下，这样的错误只会导致一些内存泄漏，逐渐消耗所有环境资源。最好的情况并不意味着容易处理。内存泄漏真的很难在不使用适当的外部工具（如 Valgrind）的情况下找到。无论如何，在大多数情况下，扩展代码中的内存管理问题将导致分段错误，在 Python 中无法恢复，并且会导致解释器崩溃而不引发任何异常。这意味着最终您将需要额外的工具，大多数 Python 程序员不需要使用。这给您的开发环境和工作流程增加了复杂性。

# 无需扩展即可与动态库进行接口

由于`ctypes`（标准库中的一个模块）或`cffi`（一个外部包），您可以在 Python 中集成几乎所有编译的动态/共享库，无论它是用什么语言编写的。而且您可以在纯 Python 中进行，无需任何编译步骤，因此这是编写 C 扩展的有趣替代方案。

这并不意味着您不需要了解 C。这两种解决方案都需要您对 C 有合理的理解，以及对动态库的工作原理有所了解。另一方面，它们消除了处理 Python 引用计数的负担，并大大减少了犯错误的风险。通过`ctypes`或`cffi`与 C 代码进行接口，比编写和编译 C 扩展模块更具可移植性。

## ctypes

`ctypes` 是调用动态或共享库函数最流行的模块，无需编写自定义的 C 扩展。其原因是显而易见的。它是标准库的一部分，因此始终可用，不需要任何外部依赖。它是一个**外部函数接口**（**FFI**）库，并提供了一个用于创建兼容 C 数据类型的 API。

### 加载库

`ctypes`中有四种类型的动态库加载器，以及两种使用它们的约定。表示动态和共享库的类有`ctypes.CDLL`、`ctypes.PyDLL`、`ctypes.OleDLL`和`ctypes.WinDLL`。最后两个仅在 Windows 上可用，因此我们不会在这里讨论它们。`CDLL`和`PyDLL`之间的区别如下：

+   `ctypes.CDLL`：此类表示已加载的共享库。这些库中的函数使用标准调用约定，并假定返回`int`。在调用期间释放 GIL。

+   `ctypes.PyDLL`：此类与`CDLL`类似，但在调用期间不会释放 GIL。执行后，将检查 Python 错误标志，并在设置时引发异常。仅在直接从 Python/C API 调用函数时才有用。

要加载库，您可以使用前述类之一实例化，并使用适当的参数，或者调用与特定类相关联的子模块的`LoadLibrary()`函数：

+   `ctypes.cdll.LoadLibrary()` 用于 `ctypes.CDLL`

+   `ctypes.pydll.LoadLibrary()` 用于 `ctypes.PyDLL`

+   `ctypes.windll.LoadLibrary()` 用于 `ctypes.WinDLL`

+   `ctypes.oledll.LoadLibrary()` 用于 `ctypes.OleDLL`

在加载共享库时的主要挑战是如何以便携方式找到它们。不同的系统对共享库使用不同的后缀（Windows 上为`.dll`，OS X 上为`.dylib`，Linux 上为`.so`）并在不同的位置搜索它们。在这方面的主要问题是 Windows，它没有预定义的库命名方案。因此，我们不会讨论在这个系统上使用`ctypes`加载库的细节，而主要集中在处理这个问题的一致和类似方式的 Linux 和 Mac OS X 上。如果您对 Windows 平台感兴趣，可以参考官方的`ctypes`文档，其中有大量关于支持该系统的信息（参见[`docs.python.org/3.5/library/ctypes.html`](https://docs.python.org/3.5/library/ctypes.html)）。

加载库的两种约定（`LoadLibrary()`函数和特定的库类型类）都要求您使用完整的库名称。这意味着需要包括所有预定义的库前缀和后缀。例如，在 Linux 上加载 C 标准库，您需要编写以下内容：

```py
>>> import ctypes
>>> ctypes.cdll.LoadLibrary('libc.so.6')
<CDLL 'libc.so.6', handle 7f0603e5f000 at 7f0603d4cbd0>

```

在这里，对于 Mac OS X，这将是：

```py
>>> import ctypes
>>> ctypes.cdll.LoadLibrary('libc.dylib')

```

幸运的是，`ctypes.util`子模块提供了一个`find_library()`函数，允许使用其名称加载库，而无需任何前缀或后缀，并且将在具有预定义共享库命名方案的任何系统上工作：

```py
>>> import ctypes
>>> from ctypes.util import find_library
>>> ctypes.cdll.LoadLibrary(find_library('c'))
<CDLL '/usr/lib/libc.dylib', handle 7fff69b97c98 at 0x101b73ac8>
>>> ctypes.cdll.LoadLibrary(find_library('bz2'))
<CDLL '/usr/lib/libbz2.dylib', handle 10042d170 at 0x101b6ee80>
>>> ctypes.cdll.LoadLibrary(find_library('AGL'))
<CDLL '/System/Library/Frameworks/AGL.framework/AGL', handle 101811610 at 0x101b73a58>

```

### 使用 ctypes 调用 C 函数

当成功加载库时，通常的模式是将其存储为与库同名的模块级变量。函数可以作为对象属性访问，因此调用它们就像调用来自任何其他已导入模块的 Python 函数一样：

```py
>>> import ctypes
>>> from ctypes.util import find_library
>>>** 
>>> libc = ctypes.cdll.LoadLibrary(find_library('c'))
>>>** 
>>> libc.printf(b"Hello world!\n")
Hello world!
13

```

不幸的是，除了整数、字符串和字节之外，所有内置的 Python 类型都与 C 数据类型不兼容，因此必须包装在`ctypes`模块提供的相应类中。以下是来自`ctypes`文档的完整兼容数据类型列表：

| ctypes 类型 | C 类型 | Python 类型 |
| --- | --- | --- |
| --- | --- | --- |
| `c_bool` | `_Bool` | `bool`（1） |
| `c_char` | `char` | 1 个字符的`bytes`对象 |
| `c_wchar` | `wchar_t` | 1 个字符的`string` |
| `c_byte` | `char` | `int` |
| `c_ubyte` | `unsigned char` | `int` |
| `c_short` | `short` | `int` |
| `c_ushort` | `unsigned short` | `int` |
| `c_int` | `int` | `int` |
| `c_uint` | `unsigned int` | `int` |
| `c_long` | `long` | `int` |
| `c_ulong` | `unsigned long` | `int` |
| `c_longlong` | `__int64 或 long long` | `int` |
| `c_ulonglong` | `unsigned __int64 或 unsigned long long` | `int` |
| `c_size_t` | `size_t` | `int` |
| `c_ssize_t` | `ssize_t 或 Py_ssize_t` | `int` |
| `c_float` | `float` | `float` |
| `c_double` | `double` | `float` |
| `c_longdouble` | `long double` | `float` |
| `c_char_p` | `char *（NUL 终止）` | `bytes`对象或`None` |
| `c_wchar_p` | `wchar_t *（NUL 终止）` | `string`或`None` |
| `c_void_p` | `void *` | `int`或`None` |

正如您所看到的，上表中没有专门的类型来反映任何 Python 集合作为 C 数组。创建 C 数组类型的推荐方法是简单地使用所需的基本`ctypes`类型与乘法运算符：

```py
>>> import ctypes
>>> IntArray5 = ctypes.c_int * 5
>>> c_int_array = IntArray5(1, 2, 3, 4, 5)
>>> FloatArray2 = ctypes.c_float * 2
>>> c_float_array = FloatArray2(0, 3.14)
>>> c_float_array[1]
3.140000104904175

```

### 将 Python 函数作为 C 回调传递

将函数实现的一部分委托给用户提供的自定义回调是一种非常流行的设计模式。C 标准库中接受此类回调的最知名函数是提供了**Quicksort**算法的`qsort()`函数。您可能不太可能使用此算法而不是更适合对 Python 集合进行排序的默认 Python **Timsort**。无论如何，`qsort()`似乎是一个高效排序算法和使用回调机制的 C API 的典型示例，在许多编程书籍中都可以找到。这就是为什么我们将尝试将其用作将 Python 函数作为 C 回调传递的示例。

普通的 Python 函数类型将不兼容`qsort()`规范所需的回调函数类型。以下是来自 BSD `man`页面的`qsort()`签名，其中还包含了接受的回调类型（`compar`参数）的类型：

```py
void qsort(void *base, size_t nel, size_t width,
           int (*compar)(const void *, const void *));
```

因此，为了执行`libc`中的`qsort()`，您需要传递：

+   `base`：这是需要作为`void*`指针排序的数组。

+   `nel`：这是`size_t`类型的元素数量。

+   `width`：这是`size_t`类型的数组中单个元素的大小。

+   `compar`：这是指向应该返回`int`并接受两个`void*`指针的函数的指针。它指向比较正在排序的两个元素大小的函数。

我们已经从*使用 ctypes 调用 C 函数*部分知道了如何使用乘法运算符从其他`ctypes`类型构造 C 数组。`nel`应该是`size_t`，它映射到 Python `int`，因此不需要任何额外的包装，可以作为`len(iterable)`传递。一旦我们知道了`base`数组的类型，就可以使用`ctypes.sizeof()`函数获取`width`值。我们需要知道的最后一件事是如何创建与`compar`参数兼容的 Python 函数指针。

`ctypes`模块包含一个`CFUNTYPE()`工厂函数，允许我们将 Python 函数包装并表示为 C 可调用函数指针。第一个参数是包装函数应该返回的 C 返回类型。它后面是作为其参数接受的 C 类型的可变列表。与`qsort()`的`compar`参数兼容的函数类型将是：

```py
CMPFUNC = ctypes.CFUNCTYPE(
    # return type
    ctypes.c_int,
    # first argument type
    ctypes.POINTER(ctypes.c_int),
    # second argument type
    ctypes.POINTER(ctypes.c_int),
)
```

### 注意

`CFUNTYPE()`使用`cdecl`调用约定，因此只与`CDLL`和`PyDLL`共享库兼容。在 Windows 上使用`WinDLL`或`OleDLL`加载的动态库使用`stdcall`调用约定。这意味着必须使用其他工厂将 Python 函数包装为 C 可调用函数指针。在`ctypes`中，它是`WINFUNCTYPE()`。

总结一切，假设我们想要使用标准 C 库中的`qsort()`函数对随机洗牌的整数列表进行排序。以下是一个示例脚本，展示了如何使用到目前为止我们学到的关于`ctypes`的一切来实现这一点：

```py
from random import shuffle

import ctypes
from ctypes.util import find_library

libc = ctypes.cdll.LoadLibrary(find_library('c'))

CMPFUNC = ctypes.CFUNCTYPE(
    # return type
    ctypes.c_int,
    # first argument type
    ctypes.POINTER(ctypes.c_int),
    # second argument type
    ctypes.POINTER(ctypes.c_int),
)

def ctypes_int_compare(a, b):
    # arguments are pointers so we access using [0] index
    print(" %s cmp %s" % (a[0], b[0]))

    # according to qsort specification this should return:
    # * less than zero if a < b
    # * zero if a == b
    # * more than zero if a > b
    return a[0] - b[0]

def main():
    numbers = list(range(5))
    shuffle(numbers)
    print("shuffled: ", numbers)

    # create new type representing array with length
    # same as the length of numbers list
    NumbersArray = ctypes.c_int * len(numbers)
    # create new C array using a new type
    c_array = NumbersArray(*numbers)

    libc.qsort(
        # pointer to the sorted array
        c_array,
        # length of the array
        len(c_array),
        # size of single array element
        ctypes.sizeof(ctypes.c_int),
        # callback (pointer to the C comparison function)
        CMPFUNC(ctypes_int_compare)
    )
    print("sorted:   ", list(c_array))

if __name__ == "__main__":
    main()
```

作为回调提供的比较函数有一个额外的`print`语句，因此我们可以看到它在排序过程中是如何执行的：

```py
$ python ctypes_qsort.py** 
shuffled:  [4, 3, 0, 1, 2]
 **4 cmp 3
 **4 cmp 0
 **3 cmp 0
 **4 cmp 1
 **3 cmp 1
 **0 cmp 1
 **4 cmp 2
 **3 cmp 2
 **1 cmp 2
sorted:    [0, 1, 2, 3, 4]

```

## CFFI

CFFI 是 Python 的外部函数接口，是`ctypes`的一个有趣的替代方案。它不是标准库的一部分，但在 PyPI 上很容易获得作为`cffi`软件包。它与`ctypes`不同，因为它更注重重用纯 C 声明，而不是在单个模块中提供广泛的 Python API。它更加复杂，还具有一个功能，允许您自动将集成层的某些部分编译成扩展，使用 C 编译器。因此，它可以用作填补 C 扩展和`ctypes`之间差距的混合解决方案。

因为这是一个非常庞大的项目，不可能在几段话中简要介绍它。另一方面，不多说一些关于它的东西会很遗憾。我们已经讨论了使用`ctypes`集成标准库中的`qsort()`函数的一个例子。因此，展示这两种解决方案之间的主要区别的最佳方式将是使用`cffi`重新实现相同的例子。我希望一段代码能比几段文字更有价值：

```py
from random import shuffle

from cffi import FFI

ffi = FFI()

ffi.cdef("""
void qsort(void *base, size_t nel, size_t width,
           int (*compar)(const void *, const void *));
""")
C = ffi.dlopen(None)

@ffi.callback("int(void*, void*)")
def cffi_int_compare(a, b):
    # Callback signature requires exact matching of types.
    # This involves less more magic than in ctypes
    # but also makes you more specific and requires
    # explicit casting
    int_a = ffi.cast('int*', a)[0]
    int_b = ffi.cast('int*', b)[0]
    print(" %s cmp %s" % (int_a, int_b))

    # according to qsort specification this should return:
    # * less than zero if a < b
    # * zero if a == b
    # * more than zero if a > b
    return int_a - int_b

def main():
    numbers = list(range(5))
    shuffle(numbers)
    print("shuffled: ", numbers)

    c_array = ffi.new("int[]", numbers)

    C.qsort(
        # pointer to the sorted array
        c_array,
        # length of the array
        len(c_array),
        # size of single array element
        ffi.sizeof('int'),
        # callback (pointer to the C comparison function)
        cffi_int_compare,
    )
    print("sorted:   ", list(c_array))

if __name__ == "__main__":
    main()
```

# 总结

本章解释了本书中最高级的主题之一。我们讨论了构建 Python 扩展的原因和工具。我们从编写纯 C 扩展开始，这些扩展仅依赖于 Python/C API，然后用 Cython 重新实现它们，以展示如果你选择合适的工具，它可以是多么容易。

仍然有一些理由可以*以困难的方式*做事，并且仅使用纯 C 编译器和`Python.h`头文件。无论如何，最好的建议是使用诸如 Cython 或 Pyrex（这里没有介绍）这样的工具，因为它将使您的代码库更易读和可维护。它还将使您免受由粗心的引用计数和内存管理引起的大部分问题的困扰。

我们对扩展的讨论以`ctypes`和 CFFI 作为集成共享库的替代方法的介绍结束。因为它们不需要编写自定义扩展来调用编译后的二进制文件中的函数，所以它们应该是你在这方面的首选工具，特别是如果你不需要使用自定义的 C 代码。

在下一章中，我们将从低级编程技术中短暂休息，并深入探讨同样重要的主题——代码管理和版本控制系统。
