# 第六章. 进一步阅读

到目前为止，在这本书中，我们已经探讨了使用 Cython 的基本和高级主题。但，这并没有结束；还有更多您可以探索的主题。

# 概述

本章我们将讨论的其他主题包括 OpenMP 支持、Cython 预处理器以及其他相关项目。考虑其他 Python 实现，如 PyPy 或使其与 Python 3 兼容。不仅如此，还有哪些 Cython 替代方案和相关 Cython 工具可供使用。我们将探讨 numba 和 Parakeet，并查看 numpy 作为 Cython 的旗舰用法。

# OpenMP 支持

OpenMP 是一种用于共享内存并行计算的语言标准 API；它在多个开源项目中使用，例如 ImageMagick ([`www.imagemagick.org/`](http://www.imagemagick.org/))，旨在加快大型图像处理的速度。Cython 对此编译器扩展提供了一些支持。但是，您必须意识到您需要使用支持 OpenMP 的编译器，如 GCC 或 MSVC。Clang/LLVM 目前还没有 OpenMP 支持。这并不是解释何时以及为什么使用 OpenMP 的地方，因为它是一个庞大的主题，但您应该查看以下网站：[`docs.cython.org/src/userguide/parallelism.html`](http://docs.cython.org/src/userguide/parallelism.html)。

# 编译时预处理器

在编译时，类似于 C/C++，我们有 C 预处理器来决定编译什么，这主要基于条件、定义和两者的混合。在 Cython 中，我们可以使用 `IF`、`ELIF`、`ELSE` 和 `DEF` 来复制其中的一些行为。以下代码行展示了这一示例：

```py
DEF myConstant = "hello cython"
```

我们还可以从 Cython 编译器访问预定义的常量 `os.uname`：

+   `UNAME_SYSNAME`

+   `UNAME_NODENAME`

+   `UNAME_RELEASE`

+   `UNAME_VERSION`

+   `UNAME_MACHINE`

我们也可以对这些内容进行条件表达式，如下所示：

```py
IF UNAME_SYSNAME == "Windows":
    include "windows.pyx"
ELSE:
    include "unix.pyx"
```

您还可以在条件表达式中使用 `ELIF`。如果您将某些内容与 C 程序中的头文件进行比较，您将看到如何在 Cython 中复制基本的 C 预处理器行为。这为您快速了解如何在头文件中复制 C 预处理器使用提供了思路。

# Python 3

将代码迁移到 Python 3 可能很痛苦，但围绕这个主题的阅读表明，人们通过仅用 Cython 编译他们的模块而不是实际迁移代码，已经成功地将他们的代码迁移到 3.*x*。使用 Cython，您可以通过以下方式指定输出以符合 Python 3 API：

```py
$ cython -3 <options>

```

这将确保您输出的是 Python 3 内容，而不是默认的 `-2` 参数，该参数为 2.*x* 标准生成。

# PyPy

PyPy 已成为标准 Python 实现的流行替代品。更重要的是，现在许多公司（从小到大）正在将其用于生产环境以提升性能和可扩展性。PyPy 与正常的 CPython 有何不同？虽然后者是一个传统的解释器，但前者是一个完整的虚拟机。它在大多数相关架构上维护了一个即时编译器后端，以进行运行时优化。

要在 PyPy 上运行 Cython 化的模块，取决于它们的 **cpyext** 模拟层。这还不完整，有许多不一致之处。但是，如果你勇敢并愿意尝试，它将随着每个版本的发布而变得越来越好。

# AutoPXD

当涉及到编写 Cython 模块时，你大部分的工作将包括正确获取你的 pxd 声明，以便正确操作原生代码。有几个项目试图创建一个编译器，读取 C/C++ 头文件并生成你的 pxd 声明作为输出。主要问题是维护一个完全符合 C 和 C++ 解析器的编译器。我的 Google Summer of Code 项目的一部分是使用 Python 插件系统作为 GCC 的一部分，以重用 GCC 的代码来解析 C/C++ 代码。该插件可以拦截声明、类型和原型。它还没有完全准备好使用，还有其他类似的项目试图解决同样的问题。更多信息可以在[`github.com/cython/cython/wiki/AutoPxd`](https://github.com/cython/cython/wiki/AutoPxd)找到。

# Pyrex 和 Cython

Cython 是 Pyrex 的衍生产品。然而，Pyrex 更加原始，Cython 为我们提供了更强大的类型和功能，以及优化和异常处理的信心。

# SWIG 和 Cython

总体来说，如果你将 SWIG ([`swig.org/`](http://swig.org/)) 视为编写原生 Python 模块的方法，你可能会被误导，认为 Cython 和 SWIG 是相似的。SWIG 主要用于编写语言绑定的包装器。例如，如果你有一些如下所示的 C 代码：

```py
int myFunction (int, const char *){ … }
```

你可以按照以下方式编写 SWIG 接口文件：

```py
/* example.i */
%module example
%{
  extern int myFunction (int, const char *);
...
%}
```

使用以下命令编译：

```py
$ swig -python example.i

```

你可以像编译 Cython 输出一样编译和链接模块，因为这将生成必要的 C 代码。如果你只想创建一个基本的模块，从 Python 调用 C，这是可以的。但 Cython 为用户提供得更多。

Cython 发展得更加完善和优化，它真正理解如何与 C 类型和工作内存管理协同工作，以及如何处理异常。使用 SWIG，你无法操作数据；你只能从 Python 调用 C 端的函数。在 Cython 中，我们可以从 Python 调用 C，反之亦然。类型转换功能非常强大；不仅如此，我们还可以将 C 类型封装成真正的 Python 类，使 C 数据感觉更像是 Pythonic。

来自第五章 高级 Cython 的 XML 示例，我们能够插入 `import` 替换？这是由于 Cython 的类型转换，API 非常 Pythonic。我们不仅可以把 C 类型包装成 Pythonic 对象，而且还让 Cython 生成 Python 执行此操作所需的样板代码，而无需将事物包装成类。更重要的是，Cython 为用户生成了更多优化的代码。

# Cython 和 NumPy

NumPy 是一个科学库，旨在提供类似于 MATLAB 的功能，MATLAB 是一个付费的专有数学包。由于你可以使用 C 类型从高度计算密集型的代码中获得更多性能，NumPy 在 Cython 用户中非常受欢迎。在 Cython 中，你可以如下导入这个库：

```py
import numpy as np
cimport numpy as np

np.import_array()
```

你可以如下访问完整的 Python API：

```py
np.PyArray_ITER_NOTDONE
```

因此，你可以在 API 的一个非常本地区域与迭代器集成。这允许 NumPy 用户在通过以下方式使用本地类型时获得很多速度：

```py
cdef double * val = (<double*>np.PyArray_MultiIter_DATA(it, 0))[0]
```

我们可以将数组中的数据转换为 `double`，在 Cython 中它是一个 `cdef` 类型，现在可以与之一起工作。有关更多信息以及 NumPy 教程，请访问 [`github.com/cython/cython/wiki/tutorials-numpy`](https://github.com/cython/cython/wiki/tutorials-numpy)。

# Numba 与 Cython 的比较

Numba 是另一种让你的 Python 代码几乎成为宿主系统的本地代码的方法，通过无缝输出要在 LLVM 上运行的代码。Numba 使用以下装饰器等：

```py
@autojit
def myFunction (): ...
```

Numba 还与 NumPy 集成。总的来说，这听起来很棒。与 Cython 不同，你只需将装饰器应用于纯 Python 代码，它为你做所有事情，但你可能会发现优化会更少，也不那么强大。

Numba 并没有像 Cython 那样与 C/C++ 集成。如果你想让它集成，你需要使用 **外部函数接口**（**FFI**）来包装调用。你还需要在 Python 代码中以非常抽象的方式定义结构体并与 C 类型一起工作，以至于与 Cython 相比，你实际上几乎没有多少控制权。

Numba 主要由装饰器组成，例如来自 Cython 的 `@locals`。但最终，所有这些创建的只是即时编译的函数，具有适当的本地函数签名。由于你可以指定函数调用的类型，这应该会在调用和从函数返回数据时提供更本地的速度。我认为，与 Cython 相比，你将获得的优化将非常有限，因为你可能需要很多抽象来与本地代码通信；尽管如此，调用很多函数可能是一种更快的技术。

仅作参考，LLVM 是一个低级虚拟机；它是一个编译器开发基础设施，项目可以使用它作为即时编译器。该基础设施可以扩展以运行各种事物，例如纯 Java 字节码，甚至通过 Numba 运行 Python。它几乎可以用于任何目的，并提供了一个良好的 API 用于开发。与 GCC（一个编译时编译器基础设施）相反，GCC 在代码运行之前会提前执行大量的静态分析，LLVM 允许代码在运行时进行更改。

### 小贴士

如需了解更多关于 Numba 和 LLVM 的信息，您可以参考以下链接中的任何一个：

[`numba.pydata.org/`](http://numba.pydata.org/)

[`llvm.org/`](http://llvm.org/)

# Parakeet 和 Numba

Parakeet 是另一个与 Numba 一起工作的项目，它为使用大量嵌套循环和并行性的 Python 代码添加了非常具体的优化。与 OpenMP 类似，它真的很酷，Numba 也需要您在代码上使用注解来完成所有这些工作。缺点是您不会神奇地优化任何 Python 代码，Parakeet 所做的优化是针对非常具体的代码集。

# 相关链接

一些有用的参考链接：

+   [`github.com/cython/cython/wiki/FAQ`](https://github.com/cython/cython/wiki/FAQ)

+   [`github.com/cython/cython/wiki`](https://github.com/cython/cython/wiki)

+   [`cython.org/`](http://cython.org/)

+   [`www.cosc.canterbury.ac.nz/greg.ewing/python/Pyrex/`](http://www.cosc.canterbury.ac.nz/greg.ewing/python/Pyrex/)

+   [`swig.org/`](http://swig.org/)

+   [`www.numpy.org/`](http://www.numpy.org/)

+   [`wiki.cython.org/tutorials/numpy`](http://wiki.cython.org/tutorials/numpy)

+   [`en.wikipedia.org/wiki/NumPy`](http://en.wikipedia.org/wiki/NumPy)

+   [`llvm.org/`](http://llvm.org/)

+   [`numba.pydata.org/`](http://numba.pydata.org/)

+   [`numba.pydata.org/numba-doc/0.9/interface_c.html`](http://numba.pydata.org/numba-doc/0.9/interface_c.html)

+   [`gcc.gnu.org/`](http://gcc.gnu.org/)

# 摘要

如果您已经阅读到这里，那么您现在应该对 Cython 非常熟悉，以至于您可以使用 C 绑定将其嵌入，甚至可以使一些纯 Python 代码更加高效。我已经向您展示了如何将 Cython 应用于实际的开源项目，甚至如何使用 Twisted Web 服务器扩展原生软件！正如我在整本书中一直说的那样，这使得 C 感觉到似乎有无穷无尽的可能性来控制逻辑，或者您可以使用大量的 Python 模块来扩展系统。感谢您的阅读。
