# *第四章*：使用 Cython 的 C 性能

Cython 是一种扩展 Python 的语言，它通过支持函数、变量和类的类型声明来扩展 Python。这些类型声明使 Cython 能够将 Python 脚本编译成高效的 C 代码。Cython 还可以作为 Python 和 C 之间的桥梁，因为它提供了易于使用的结构来编写对外部 C 和 C++ 例程的接口。

在本章中，我们将学习以下主题：

+   编译 Cython 扩展

+   添加静态类型

+   分享声明

+   与数组一起工作

+   在 Cython 中使用粒子模拟器

+   分析 Cython

+   在 Jupyter 中使用 Cython

通过本章，我们将学习如何利用 Cython 提高我们程序的效率。虽然对 C 的基本了解有帮助，但本章仅关注 Python 优化背景下的 Cython。因此，它不需要任何 C 背景。

# 技术要求

你可以在 GitHub 上访问本章使用的代码，地址为 [`github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter04`](https://github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter04)。

# 编译 Cython 扩展

按照设计，Cython 语法是 Python 的超集。Cython 可以编译大多数 Python 模块，只有少数例外，而无需任何更改。Cython 源文件具有 `.pyx` 扩展名，可以使用 `cython` 命令编译成 C 文件。

将 Python 代码转换为 Cython 所需的只是对语法的一些修改，我们将在本章中看到这些修改（例如在声明变量和函数时），以及编译。虽然这个程序一开始可能看起来令人畏惧，但 Cython 通过它提供的计算优势将远远弥补这一点。

首先，要安装 Cython，我们可以简单地运行 `pip` 命令，如下所示：

```py
$pip install cython
```

有关更多详细信息，请参阅 [`pypi.org/project/Cython/`](https://pypi.org/project/Cython/) 上的文档。我们的第一个 Cython 脚本将包含一个简单的函数，该函数将输出 `Hello, World!`。按照以下步骤操作：

1.  创建一个包含以下代码的新 `hello.pyx` 文件：

    ```py
        def hello(): 
          print('Hello, World!') 
    ```

1.  `cython` 命令将读取 `hello.pyx` 并生成一个 `hello.c` 文件，如下所示：

    ```py
    $ cython hello.pyx
    ```

1.  要将 `hello.c` 编译成 Python 扩展模块，我们将使用 `/usr/include/python3.5/`：

    ```py
    $ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-
    strict-aliasing -lm -I/usr/include/python3.5/ -o 
    hello.so hello.c
    ```

    注意

    要找到你的 Python `include` 目录，你可以使用 `distutils` 工具并运行 `sysconfig.get_python_inc`。要执行它，你可以简单地发出以下命令：`python -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"`。

1.  这将生成一个名为 `hello.so` 的文件，这是一个可以直接导入 Python 会话的 C 扩展模块。代码在下面的代码片段中展示：

    ```py
        >>> import hello 
        >>> hello.hello() 
        Hello, World!
    ```

1.  Cython 既可以接受 Python 2，也可以接受 Python 3 作为 `hello.pyx` 文件，使用 `-3` 选项，如下面的代码片段所示：

    ```py
    $ cython -3 hello.pyx
    ```

1.  生成的`hello.c`文件可以通过包含相应的头文件（使用`-I`选项）来编译，无需对 Python 2 和 Python 3 进行任何更改，如下所示：

    ```py
    $ gcc -I/usr/include/python3.5 # ... other options
    $ gcc -I/usr/include/python2.7 # ... other options
    ```

1.  使用`distutils`，Python 的标准打包工具，可以更直接地编译 Cython 程序。通过编写`setup.py`脚本，我们可以直接将`.pyx`文件编译为扩展模块。要编译我们的`hello.pyx`示例，我们可以编写一个包含以下代码的最小`setup.py`脚本：

    ```py
        from distutils.core import setup 
        from Cython.Build import cythonize 
        setup( 
          name='Hello',
          ext_modules = cythonize('hello.pyx')
        ) 
    ```

在前面的代码片段的前两行中，我们导入了`setup`函数和`cythonize`辅助函数。`setup`函数包含一些键值对，指定了应用程序的名称和需要构建的扩展。

1.  `cythonize`辅助函数接受一个字符串或字符串列表，其中包含我们想要编译的 Cython 模块。你也可以通过运行以下代码使用 glob 模式：

    ```py
        cythonize(['hello.pyx', 'world.pyx', '*.pyx']) 
    ```

1.  要使用`distutils`编译我们的扩展模块，你可以使用以下代码执行`setup.py`脚本：

    ```py
    $ python setup.py build_ext --inplace
    ```

`build_ext`选项告诉脚本构建`ext_modules`中指示的扩展模块，而`--inplace`选项告诉脚本将`hello.so`输出文件放置在源文件相同的目录中（而不是构建目录）。

1.  Cython 模块也可以使用`pyximport`自动编译。你所需要做的只是在脚本开头调用`pyximport.install()`（或者你需要在解释器中发出该命令），如下面的代码片段所示。完成之后，你可以直接导入`.pyx`文件，`pyximport`将透明地编译相应的 Cython 模块：

    ```py
        >>> import pyximport 
        >>> pyximport.install() 
        >>> import hello # This will compile hello.pyx 
    ```

不幸的是，`pyximport`并不适用于所有类型的配置（例如，当它们涉及 C 和 Cython 文件的组合时），但它对于测试简单的脚本来说很有用。

1.  自从版本 0.13 起，IPython 包含了`cythonmagic`扩展，可以交互式地编写和测试一系列 Cython 语句。你可以使用`load_ext`在 IPython shell 中加载扩展，如下所示：

    ```py
        %load_ext Cython
    ```

一旦扩展被加载，你可以使用`%%cython` *cell magic*来编写多行 Cython 代码片段。在下面的示例中，我们定义了一个`hello_snippet`函数，该函数将被编译并添加到 IPython 会话命名空间中：

```py
    %%cython 
    def hello_snippet(): 
        print("Hello, Cython!") 
    hello_snippet()
    Hello,  Cython! 
```

与 Cython 源代码一起工作非常简单。在下一节中，我们将看到如何将静态类型添加到我们的程序中，使其更接近 C 代码。

# 添加静态类型

在 Python 中，一个变量可以在程序执行期间与不同类型的对象关联。虽然这个特性使得语言更加灵活和动态，但它也给解释器带来了显著的开销，因为它需要在运行时查找变量的类型和方法，这使得执行各种优化变得困难。Cython 通过编译扩展 Python 语言，以显式类型声明，从而可以生成高效的 C 扩展。

在 Cython 中声明数据类型的主要方式是通过 `cdef` 语句。`cdef` 关键字可以在多个上下文中使用，例如变量、函数和扩展类型（静态类型类）。我们将在以下小节中看到如何做到这一点。

## 声明变量

在 Cython 中，可以通过在变量前加上 `cdef` 和相应的类型来声明变量的类型。例如，我们可以以下这种方式声明 `i` 变量为 16 位整数：

```py
    cdef int i 
```

`cdef` 语句支持在同一行上声明多个变量名，以及可选的初始化，如下面的代码行所示：

```py
    cdef double a, b = 2.0, c = 3.0 
```

带类型的变量与普通变量处理方式不同。在 Python 中，变量通常被描述为 *标签*，它们指向内存中的对象。例如，我们可以在程序的任何位置将值 `'hello'` 赋给 `a` 变量，而不受限制，如下所示：

```py
    a = 'hello' 
```

`a` 变量持有 `'hello'` 字符串的引用。我们还可以在代码的后面自由地将另一个值（例如整数 `1`）赋给同一个变量，如下所示：

```py
    a = 1 
```

Python 会将整数 `1` 无任何问题地赋值给 `a` 变量。

带类型的变量表现得很不同，通常被描述为 *数据容器*：我们只能存储适合容器中值的类型。例如，如果我们将 `a` 变量声明为 `int`，然后我们尝试将其赋值为 `double` 数据类型，Cython 将会触发错误，如下面的代码片段所示：

```py
    %%cython 
    cdef int i 
    i = 3.0 
    # Output has been cut 
    ...cf4b.pyx:2:4 Cannot assign type 'double' to 'int' 
```

静态类型使得编译器能够执行有用的优化。例如，如果我们将循环索引声明为 `int`，Cython 将会重写循环为纯 C 代码，而无需进入 Python 解释器。类型声明保证了索引的类型始终是 `int`，并且在运行时不能被覆盖，这样编译器就可以自由地进行优化，而不会损害程序的正确性。

我们可以通过一个小测试用例来评估这种情况下的速度提升。在以下示例中，我们实现了一个简单的循环，该循环将变量 `100` 次自增。使用 Cython，`example` 函数可以编写如下：

```py
    %%cython 
    def example(): 
       cdef int i, j=0 
       for i in range(100):
           j += 1 
       return j 
    example() 
    # Result:
    # 100 
```

我们可以比较类似的无类型、纯 Python 循环的速度，如下所示：

```py
    def example_python(): 
        j=0 
        for i in range(100):
            j += 1 
        return j 
    %timeit example() 
    10000000 loops, best of 3: 25 ns per loop 
    %timeit example_python() 
    100000 loops, best of 3: 2.74 us per loop 
```

通过实现这种简单的类型声明获得的加速效果是惊人的 100 倍！这是因为 Cython 循环首先被转换为纯 C 代码，然后转换为高效的机器代码，而 Python 循环仍然依赖于缓慢的解释器。

在 Cython 中，可以声明变量为任何标准 C 类型，也可以使用经典的 C 构造定义自定义类型，例如 `struct`、`enum` 和 `typedef`。

一个有趣的例子是，如果我们声明一个变量为 `object` 类型，该变量将接受任何类型的 Python 对象，如下面的代码片段所示：

```py
    cdef object a_py 
    # both 'hello' and 1 are Python objects 
    a_py = 'hello' 
    a_py = 1 
```

注意，将变量声明为`object`没有性能优势，因为访问和操作该对象仍然需要解释器查找变量的底层类型及其属性和方法。

有时，某些数据类型（如`float`和`int`数字）在某种意义上是兼容的，即它们可以相互转换。在 Cython 中，可以通过在目标类型周围放置尖括号来在类型之间进行转换（*cast*），如下面的代码片段所示：

```py
    cdef int a = 0 
    cdef double b 
    b = <double> a 
```

除了变量的静态类型外，我们还可以提供有关函数的信息，我们将在下一节学习如何做到这一点。

## 声明函数

您可以通过在每个参数名称前指定类型来向 Python 函数的参数添加类型信息。以这种方式指定的函数将像常规 Python 函数一样工作并执行，但它们的参数将进行类型检查。我们可以编写一个`max_python`函数，返回两个整数之间的较大值，如下所示：

```py
    def max_python(int a, int b):
        return a if a > b else b 
```

以这种方式指定的函数将执行类型检查并将参数视为类型化变量，就像在`cdef`定义中一样。然而，该函数仍然是一个 Python 函数，多次调用它仍然需要切换回解释器。为了允许 Cython 对函数调用进行优化，我们应该使用`cdef`语句声明返回类型的类型，如下所示：

```py
    cdef int max_cython(int a, int b): 
        return a if a > b else b 
```

以这种方式声明的函数被转换为本地 C 函数，与 Python 函数相比，开销要小得多。一个显著的缺点是它们不能从 Python 中使用，而只能从 Cython 中使用，并且它们的范围限制在相同的 Cython 文件中，除非它们在定义文件中公开（参考*共享声明*部分）。

幸运的是，Cython 允许您定义既可以从 Python 调用又可转换为高性能 C 函数的函数。如果您使用`cpdef`语句声明函数，Cython 将生成函数的两个版本：一个可供解释器使用的 Python 版本，以及一个从 Cython 可用的快速 C 函数。`cpdef`语法与`cdef`等效，如下所示：

```py
    cpdef int max_hybrid(int a, int b): 
        return a if a > b else b 
```

有时，即使使用 C 函数，调用开销也可能成为性能问题，尤其是在关键循环中多次调用同一函数时。当函数体较小时，在函数定义前添加`inline`关键字是很方便的；函数调用将被函数体本身替换。我们的`max`函数是进行内联的好候选，如下面的代码片段所示：

```py
    cdef inline int max_inline(int a, int b): 
        return a if a > b else b 
```

最后，我们将看到如何处理类类型。

## 声明类

我们可以使用`cdef class`语句定义扩展类型，并在类体中声明其属性。例如，我们可以创建一个`Point`扩展类型，如下面的代码片段所示，它存储两个`double`类型的坐标（`x`，`y`）：

```py
    cdef class Point:
        cdef double x 
        cdef double y
        def __init__(self, double x, double y): 
            self.x = x 
            self.y = y 
```

在类方法中访问声明的属性允许 Cython 通过直接访问底层 C 结构中的给定字段来绕过昂贵的 Python 属性查找。因此，类型化类中的属性访问是一个极快的操作。

要在代码中使用`cdef class`语句，您需要在编译时显式声明您打算使用的变量的类型。您可以在任何您将使用标准类型（如`double`、`float`和`int`）的上下文中使用扩展类型名称（如`Point`）。例如，如果我们想要一个 Cython 函数来计算从原点（在示例中，该函数称为`norm`）到`Point`的距离，我们必须将输入变量声明为`Point`，如下面的代码片段所示：

```py
    cdef double norm(Point p): 
        return (p.x**2 + p.y**2)**0.5 
```

正如类型化函数一样，类型化类也有一些限制。如果您尝试从 Python 访问扩展类型属性，您将收到一个`AttributeError`警告，如下所示：

```py
    >>> a = Point(0.0, 0.0) 
    >>> a.x 
    AttributeError: 'Point' object has no attribute 'x' 
```

为了从 Python 代码中访问属性，您必须在属性声明中使用`public`（用于读写访问）或`readonly`指定符，如下面的代码片段所示：

```py
    cdef class Point: 
        cdef public double x 
```

此外，可以使用`cpdef`语句声明方法，就像常规函数一样。

扩展类型不支持在运行时添加额外的属性。为了实现这一点，可以通过定义一个 Python 类，使其成为类型化类的子类，并在纯 Python 中扩展其属性和方法来解决这个问题。

通过这种方式，我们已经看到了如何在 Cython 中向各种对象添加静态类型。在下一节中，我们将开始讨论声明。

# 声明共享

当编写您的 Cython 模块时，您可能希望将最常用的函数和类声明重新组织到一个单独的文件中，以便它们可以在不同的模块中重用。Cython 允许您将这些组件放入一个*定义文件*中，并通过`cimport`语句访问它们。

假设我们有一个包含`max`和`min`函数的模块，并且我们想在多个 Cython 程序中重用这些函数。如果我们简单地在`.pyx`文件中编写一些函数，声明将仅限于同一文件。

注意

定义文件也用于将 Cython 与外部 C 代码接口。想法是将定义文件中的类型和函数原型复制（或更准确地说，翻译）到外部 C 代码中，该代码将在单独的步骤中编译和链接。 

为了共享`max`和`min`函数，我们需要编写一个具有`.pxd`扩展名的定义文件。这样的文件只包含我们想要与其他模块共享的类型和函数原型——一个*公共*接口。我们可以在名为`mathlib.pxd`的文件中声明我们的`max`和`min`函数的原型，如下所示：

```py
    cdef int max(int a, int b) 
    cdef int min(int a, int b) 
```

如您所见，我们只编写了函数名和参数，而没有实现函数体。

函数实现将放入与基本名称相同但扩展名为 `.pyx` 的实现文件 `mathlib.pyx` 中，如下所示：

```py
    cdef int max(int a, int b): 
      return a if a > b else b 
    cdef int min(int a, int b): 
      return a if a < b else b 
```

`mathlib` 模块现在可以从另一个 Cython 模块导入。

为了测试我们新的 Cython 模块，我们将创建一个名为 `distance.pyx` 的文件，其中包含一个名为 `chebyshev` 的函数。该函数将计算两点之间的 Chebyshev 距离，如下所示。两点坐标——`(x1, y1)` 和 `(x2, y2)` 之间的 Chebyshev 距离定义为每个坐标之间差异的最大值：

```py
    max(abs(x1 - x2), abs(y1 - y2)) 
```

为了实现 `chebyshev` 函数，我们将使用在 `mathlib.pxd` 中声明的 `max` 函数，通过 `cimport` 语句导入，如下所示：

```py
    from mathlib cimport max 
    def chebyshev(int x1, int y1, int x2, int y2): 
        return max(abs(x1 - x2), abs(y1 - y2)) 
```

`cimport` 语句将读取 `mathlib.pxd`，并将 `max` 定义用于生成 `distance.c` 文件。

除了静态类型和声明之外，C 通常比 Python 快的一个因素是其高度优化的数组操作，我们将在下一节中探讨。

# 与数组一起工作

数值和高性能计算通常使用数组。Cython 提供了一种简单的方法来与不同类型的数组交互，可以直接使用低级 C 数组，或者更通用的 *类型内存视图*。我们将在以下小节中看到如何做到这一点。

## C 数组和指针

C 数组是相同类型项的集合，在内存中连续存储。在深入细节之前，了解（或复习）C 中内存的管理方式是有帮助的。

C 中的变量就像容器。在创建变量时，会在内存中预留空间以存储其值。例如，如果我们创建一个包含 64 位浮点数（`double`）的变量，程序将分配 64 位（16 字节）的内存。可以通过访问该内存位置的地址来访问这部分内存。

要获取变量的地址，我们可以使用表示为 `&` 符号的 *地址运算符*。我们还可以使用 `printf` 函数，如下所示，这在 `libc.stdio` Cython 模块中可用，用于打印该变量的地址：

```py
    %%cython 
    cdef double a 
    from libc.stdio cimport printf 
    printf("%p", &a)
    # Output:
    # 0x7fc8bb611210 
```

注意

只有当代码从标准 Python 终端运行时，才会生成输出。IPython 的这个限制在 [`github.com/ipython/ipython/issues/1230`](https://github.com/ipython/ipython/issues/1230) 中有详细说明。

内存地址可以存储在特殊的变量中，即 *指针*，可以通过在变量名前放置一个 `*` 前缀来声明，如下所示：

```py
    from libc.stdio cimport printf 
    cdef double a 
    cdef double *a_pointer 
    a_pointer = &a # a_pointer and &a are of the same type 
```

如果我们有一个指针，并且想要获取它指向的地址中包含的值，我们可以使用这里显示的零索引表示法：

```py
    cdef double a 
    cdef double *a_pointer 
    a_pointer = &a 
    a = 3.0 
    print(a_pointer[0]) # prints 3.0 
```

在声明 C 数组时，程序会分配足够的空间来容纳请求的所有元素。例如，为了创建一个包含 10 个 `double` 值（每个 16 字节）的数组，程序将在内存中预留 *16* * *10* = *160* 字节的连续空间。在 Cython 中，我们可以使用以下语法声明此类数组：

```py
    cdef double arr[10]
```

我们也可以使用以下语法声明一个多维数组，例如具有`5`行和`2`列的数组：

```py
    cdef double arr[5][2] 
```

内存将在单个内存块中分配，行后行。这种顺序通常被称为*行主序*，如下面的截图所示。数组也可以按*列主序*排序，这是 Fortran 编程语言的情况：

![图 4.1 – 行主序](img/B17499_Figure_4.1.jpg)

图 4.1 – 行主序

数组排序有重要的后果。当我们按最后一个维度迭代 C 数组时，我们访问连续的内存块（在我们的例子中，0, 1, 2, 3 ...），而当我们按第一个维度迭代时，我们会跳过一些位置（0, 2, 4, 6, 8, 1 ...）。你应该始终尝试按顺序访问内存，因为这优化了缓存和内存使用。

我们可以使用标准索引来存储和检索数组中的元素，如下所示；C 数组不支持复杂索引或切片：

```py
    arr[0] = 1.0 
```

C 数组具有与指针许多相同的行为。实际上，`arr`变量指向数组的第一个元素的内存位置。我们可以使用解引用操作符来验证数组第一个元素的地址与`arr`变量中包含的地址相同，如下所示：

```py
%%cython 
from libc.stdio cimport printf 
cdef double arr[10] 
printf("%p\n", arr) 
printf("%p\n", &arr[0])
# Output
# 0x7ff6de204220 
# 0x7ff6de204220
```

注意

只有当从标准 Python 终端运行上述代码时，才会生成输出。IPython 的这个限制在[`github.com/ipython/ipython/issues/1230`](https://github.com/ipython/ipython/issues/1230)中详细说明。

在与现有的 C 库接口或需要精细控制内存时（此外，它们性能也非常出色），你应该使用 C 数组和指针。这种精细控制水平也容易出错，因为它不能阻止你访问错误的内存位置。对于更常见的用例和改进的安全性，你可以使用 NumPy 数组或类型化内存视图。

## 使用 NumPy 数组

NumPy 数组可以在 Cython 中用作普通 Python 对象，使用它们已经优化的广播操作。然而，Cython 提供了一个`numpy`模块，它提供了更好的直接迭代支持。

当我们通常访问 NumPy 数组的一个元素时，解释器级别会执行一些其他操作，这导致大量开销。Cython 可以通过直接作用于 NumPy 数组使用的底层内存区域来绕过这些操作和检查，从而获得令人印象深刻的性能提升。

NumPy 数组可以声明为`ndarray`数据类型。要在我们的代码中使用该数据类型，我们首先需要`cimport` `numpy` Cython 模块（它与 Python 的`numpy`模块不同）。我们将该模块绑定到`c_np`变量，以使与 Python `numpy`模块的差异更加明确，如下所示：

```py
    cimport numpy as c_np
    import numpy as np
```

我们现在可以通过指定其类型和方括号之间的维度数来声明 NumPy 数组（这称为*缓冲区语法*）。要声明`double`，我们可以使用以下代码：

```py
    cdef c_np.ndarray[double, ndim=2] arr 
```

对这个数组的访问将通过直接操作底层内存区域来完成；这种操作将避免进入解释器，给我们带来巨大的速度提升。

在下一个示例中，我们将展示类型化 NumPy 数组的用法，并将其与常规 Python 版本进行比较。

我们首先编写一个`numpy_bench_py`函数，该函数递增`py_arr`中的每个元素。我们将`i`索引声明为整数，这样我们就可以避免`for`循环的开销，如下所示：

```py
    %%cython 
    import numpy as np 
    def numpy_bench_py(): 
        py_arr = np.random.rand(1000) 
        cdef int i 
        for i in range(1000): 
            py_arr[i] += 1 
```

然后，我们使用`ndarray`类型编写相同的函数。请注意，在定义`c_arr`变量使用`c_np.ndarray`之后，我们可以从`numpy`Python 模块分配一个数组给它。代码如下所示：

```py
    %%cython 
    import numpy as np 
    cimport numpy as c_np 
    def numpy_bench_c(): 
        cdef c_np.ndarray[double, ndim=1] c_arr 
        c_arr = np.random.rand(1000) 
        cdef int i
        for i in range(1000): 
           c_arr[i] += 1 
```

我们可以使用`timeit`来计时结果，并且我们可以看到这里，类型化的版本要快 50 倍：

```py
    %timeit numpy_bench_c() 
    100000 loops, best of 3: 11.5 us per loop 
    %timeit numpy_bench_py() 
    1000 loops, best of 3: 603 us per loop 
```

这使我们从 Python 代码中获得了显著的加速效果！

## 使用类型化的内存视图

C 和 NumPy 数组，以及内置的`bytes`、`bytearray`和`array.array`对象，在某种意义上是相似的，因为它们都在连续的内存区域（也称为内存*缓冲区*）上操作。Cython 提供了一个通用接口——*类型化的内存视图*——它统一并简化了对所有这些数据类型的访问。

以下是以这种方式声明一个`int`和一个`double`的二维内存视图：

```py
    cdef int[:] a 
    cdef double[:, :] b 
```

同样的语法也适用于变量、函数定义、类属性等的类型声明。任何暴露缓冲区接口的对象（例如，NumPy 数组、`bytes`和`array.array`对象）都将自动绑定到内存视图。例如，我们可以通过简单的变量赋值将内存视图绑定到 NumPy 数组，如下所示：

```py
    import numpy as np 
    cdef int[:] arr 
    arr_np = np.zeros(10, dtype='int32') 
    arr = arr_np # We bind the array to the memoryview 
```

重要的是要注意，内存视图并不*拥有*数据，它只提供了一种*访问*和*更改*它所绑定数据的途径；在这种情况下，所有权留给了 NumPy 数组。正如你在下面的示例中可以看到的，通过内存视图所做的更改将作用于底层内存区域，并将反映在原始 NumPy 结构中（反之亦然）：

```py
    arr[2] = 1 # Changing memoryview 
    print(arr_np) 
    # [0 0 1 0 0 0 0 0 0 0] 
```

在某种意义上，内存视图背后的机制与 NumPy 在切片数组时产生的机制相似。正如我们在*第三章*中看到的，“使用 NumPy、Pandas 和 Xarray 进行快速数组操作”，切片 NumPy 数组不会复制数据，而是返回对同一内存区域的视图，对视图的更改将反映在原始数组中。

内存视图也支持使用标准的 NumPy 语法进行数组切片，如下面的代码片段所示：

```py
    cdef int[:, :, :] a 
    arr[0, :, :] # Is a 2-dimensional memoryview 
    arr[0, 0, :] # Is a 1-dimensional memoryview 
    arr[0, 0, 0] # Is an int 
```

要在两个内存视图之间复制数据，你可以使用类似于切片赋值的语法，如下面的代码片段所示：

```py
    import numpy as np 
    cdef double[:, :] b 
    cdef double[:] r 
    b = np.random.rand(10, 3) 
    r = np.zeros(3, dtype='float64') 
    b[0, :] = r # Copy the value of r in the first row of b 
```

在下一节中，我们将使用类型化的内存视图来为粒子模拟器中的数组声明类型。

# 在 Cython 中使用粒子模拟器

现在我们已经对 Cython 的工作原理有了基本的了解，我们可以重写`ParticleSimulator.evolve`方法。多亏了 Cython，我们可以将我们的循环转换为 C 语言，从而消除 Python 解释器引入的开销。

在*第三章*《使用 NumPy、Pandas 和 Xarray 进行快速数组操作》中，我们使用 NumPy 编写了一个相当高效的`evolve`方法版本。我们可以将旧版本重命名为`evolve_numpy`以区分新旧版本。代码如下所示：

```py
    def evolve_numpy(self, dt): 
        timestep = 0.00001 
        nsteps = int(dt/timestep) 
        r_i = np.array([[p.x, p.y] for p in \
            self.particles])     
        ang_speed_i = np.array([p.ang_speed for p in \
          self.particles]) 
        v_i = np.empty_like(r_i) 
        for i in range(nsteps): 
            norm_i = np.sqrt((r_i ** 2).sum(axis=1)) 
            v_i = r_i[:, [1, 0]] 
            v_i[:, 0] *= -1 
            v_i /= norm_i[:, np.newaxis]         
            d_i = timestep * ang_speed_i[:, np.newaxis] * \
                v_i 
            r_i += d_i 
        for i, p in enumerate(self.particles): 
            p.x, p.y = r_i[i] 
```

我们希望将此代码转换为 Cython。我们的策略将是利用快速的索引操作，通过移除 NumPy 数组广播，从而回到基于索引的算法。由于 Cython 生成高效的 C 代码，我们可以自由地使用尽可能多的循环而不会产生任何性能惩罚。

作为设计选择，我们可以决定将循环封装在一个函数中，我们将用 Cython 模块`cevolve.pyx`重写这个函数。该模块将包含一个单一的 Python 函数`c_evolve`，它将接受粒子位置、角速度、时间步长和步数作为输入。

起初，我们并没有添加类型信息；我们只想隔离函数并确保我们可以无错误地编译我们的模块。代码如下所示：

```py
    # file: simul.py 
    def evolve_cython(self, dt): 
        timestep = 0.00001 
        nsteps = int(dt/timestep) 

        r_i = np.array([[p.x, p.y] for p in \
            self.particles])     
        ang_speed_i = np.array([p.ang_speed for p in \
            self.particles]) 

        c_evolve(r_i, ang_speed_i, timestep, nsteps) 

        for i, p in enumerate(self.particles): 
            p.x, p.y = r_i[i] 

    # file: cevolve.pyx 
    import numpy as np 

    def c_evolve(r_i, ang_speed_i, timestep, nsteps): 
        v_i = np.empty_like(r_i) 

        for i in range(nsteps): 
            norm_i = np.sqrt((r_i ** 2).sum(axis=1)) 

            v_i = r_i[:, [1, 0]] 
            v_i[:, 0] *= -1 
            v_i /= norm_i[:, np.newaxis]         

            d_i = timestep * ang_speed_i[:, np.newaxis] * 
                v_i 

            r_i += d_i 
```

注意，我们不需要为`c_evolve`返回值，因为值是在`r_i`数组中就地更新的。我们可以通过稍微修改我们的`benchmark`函数来将无类型 Cython 版本与旧的 NumPy 版本进行基准测试，如下所示：

```py
    def benchmark(npart=100, method='python'): 
        particles = [
                     Particle(uniform(-1.0, 1.0),
                              uniform(-1.0, 1.0),
                              uniform(-1.0, 1.0)) 
                              for i in range(npart)
            ] 
        simulator = ParticleSimulator(particles) 
        if method=='python': 
            simulator.evolve_python(0.1)
        elif method == 'cython': 
            simulator.evolve_cython(0.1) 
        elif method == 'numpy': 
            simulator.evolve_numpy(0.1) 
```

我们可以在 IPython shell 中计时不同的版本，如下所示：

```py
    %timeit benchmark(100, 'cython') 
    1 loops, best of 3: 401 ms per loop 
    %timeit benchmark(100, 'numpy') 
    1 loops, best of 3: 413 ms per loop 
```

这两个版本的速度相同。在纯 Python 版本中，不使用静态类型编译 Cython 模块没有任何优势。下一步是声明所有重要变量的类型，以便 Cython 可以执行其优化。

我们可以开始向函数参数添加类型，并查看性能如何变化。我们可以声明数组为包含`double`值的类型化内存视图。值得一提的是，如果我们传递`int`或`float32`类型的数组，类型转换不会自动发生，我们将得到一个错误。代码如下所示：

```py
    def c_evolve(double[:, :] r_i,
                 double[:] ang_speed_i,
                 double timestep,
                 int nsteps): 
```

到目前为止，我们可以重写粒子和时间步长的循环。我们可以声明`i`和`j`迭代索引以及`nparticles`粒子数为`int`，如下所示：

```py
    cdef int i, j 
    cdef int nparticles = r_i.shape[0] 
```

算法与纯 Python 版本非常相似；我们遍历粒子和时间步长，并使用以下代码计算每个粒子坐标的速度和位移向量：

```py
      for i in range(nsteps): 
          for j in range(nparticles): 
              x = r_i[j, 0] 
              y = r_i[j, 1] 
              ang_speed = ang_speed_i[j] 

              norm = sqrt(x ** 2 + y ** 2) 

              vx = (-y)/norm 
              vy = x/norm 

              dx = timestep * ang_speed * vx 
              dy = timestep * ang_speed * vy 

              r_i[j, 0] += dx 
              r_i[j, 1] += dy 
```

在前面的代码片段中，我们添加了`x`、`y`、`ang_speed`、`norm`、`vx`、`vy`、`dx`和`dy`变量。为了避免 Python 解释器的开销，我们不得不在函数开始处声明它们对应的类型，如下所示：

```py
    cdef double norm, x, y, vx, vy, dx, dy, ang_speed 
```

我们还使用了一个名为`sqrt`的函数来计算范数。如果我们使用`math`模块或`numpy`中的`sqrt`函数，我们将在关键循环中再次包含一个慢速的 Python 函数，从而降低我们的性能。标准 C 库中有一个快速的`sqrt`函数，已经包装在`libc.math` Cython 模块中。运行以下代码来导入它：

```py
    from libc.math cimport sqrt 
```

我们可以重新运行基准测试来评估我们的改进，如下所示：

```py
    In [4]: %timeit benchmark(100, 'cython') 
    100 loops, best of 3: 13.4 ms per loop 
    In [5]: %timeit benchmark(100, 'numpy') 
    1 loops, best of 3: 429 ms per loop 
```

对于小粒子数量，速度提升非常显著，因为我们获得了比上一个版本快 40 倍的性能。然而，我们也应该尝试用更多的粒子数量来测试性能缩放，如下所示：

```py
    In [2]: %timeit benchmark(1000, 'cython') 
    10 loops, best of 3: 134 ms per loop 
    In [3]: %timeit benchmark(1000, 'numpy') 
    1 loops, best of 3: 877 ms per loop
```

随着粒子数量的增加，两个版本的速度越来越接近。通过将粒子大小增加到`1000`，我们已经将我们的速度提升降低到更适度的 6 倍。这可能是由于随着粒子数量的增加，Python `for`循环的开销与其他操作的速度相比变得越来越不显著。

基准测试的主题自然地过渡到我们下一个部分：分析。

# 分析 Cython

Cython 提供了一个名为*注释视图*的功能，有助于找到在 Python 解释器中执行的哪些行，以及哪些是后续优化的良好候选者。我们可以通过使用带有`-a`选项编译 Cython 文件来启用此功能。这样，Cython 将生成以下所示的内容：

```py
$ cython -a cevolve.pyx
$ firefox cevolve.html
```

下面的截图显示的 HTML 文件显示了我们的 Cython 文件逐行：

![图 4.2 – 包含注释代码的生成 HTML](img/B17499_Figure_4.2.jpg)

图 4.2 – 包含注释代码的生成 HTML

源代码中的每一行都可以显示为不同的黄色阴影。颜色越深，与解释器相关的调用就越多，而白色行则被转换为常规的 C 代码。由于解释器调用会显著减慢执行速度，目标是将函数体尽可能变为白色。通过点击任何一行，我们可以检查 Cython 编译器生成的代码。例如，`v_y = x/norm`这一行检查范数是否不为`0`，如果条件未验证，则抛出`ZeroDivisionError`错误。`x = r_i[j, 0]`这一行显示 Cython 检查索引是否在数组的范围内。你可能已经注意到最后一行颜色非常深；通过检查代码，我们可以看到这实际上是一个错误——代码引用了与函数结束相关的样板代码。

Cython 可以关闭如除以零等检查，以便它可以移除那些额外的解释器相关调用；这通常通过编译器指令完成。有几种不同的方法可以添加编译器指令，如下所述：

+   使用装饰器或上下文管理器

+   在文件开头使用注释

+   使用 Cython 命令行选项

对于 Cython 编译器指令的完整列表，你可以参考官方文档在 [`docs.cython.org/src/reference/compilation.html#compiler-directives`](http://docs.cython.org/src/reference/compilation.html#compiler-directives)。

例如，要禁用数组的边界检查，只需用 `cython.boundscheck` 装饰一个函数，如下所示：

```py
    cimport cython 
    @cython.boundscheck(False) 
    def myfunction(): 
        # Code here 
```

或者，我们可以使用 `cython.boundscheck` 将代码块包装成上下文管理器，如下所示：

```py
    with cython.boundscheck(False): 
        # Code here 
```

如果我们想禁用整个模块的边界检查，我们可以在文件开头添加以下代码行：

```py
    # cython: boundscheck=False 
```

要使用命令行选项更改指令，你可以使用 `-X` 选项，如下所示：

```py
$ cython -X boundscheck=True
```

要禁用 `c_evolve` 函数中的额外检查，我们可以禁用 `boundscheck` 指令并启用 `cdivision`（这防止了 `ZeroDivisionError` 的检查），如下面的代码片段所示：

```py
    cimport cython 
    @cython.boundscheck(False) 
    @cython.cdivision(True) 
    def c_evolve(double[:, :] r_i,double[:] ang_speed_i, \
                 double timestep,int nsteps): 
```

如果我们再次查看注释视图，循环体已经完全变为白色——我们从内部循环中移除了所有解释器的痕迹。为了重新编译，只需再次输入 `python setup.py build_ext --inplace`。然而，通过运行基准测试，我们注意到我们没有获得性能提升，这表明这些检查不是瓶颈的一部分，正如我们在这里可以看到的：

```py
    In [3]: %timeit benchmark(100, 'cython') 
    100 loops, best of 3: 13.4 ms per loop 
```

分析 Cython 代码的另一种方法是使用 `cProfile` 模块。例如，我们可以编写一个简单的函数来计算坐标数组之间的 Chebyshev 距离。创建一个 `cheb.py` 文件，如下所示：

```py
    import numpy as np 
    from distance import chebyshev 
    def benchmark(): 
        a = np.random.rand(100, 2) 
        b = np.random.rand(100, 2) 
        for x1, y1 in a: 
            for x2, y2 in b: 
                chebyshev(x1, x2, y1, y2) 
```

如果我们尝试以当前状态分析这个脚本，我们将不会得到关于我们在 Cython 中实现的函数的任何统计信息。如果我们想收集 `max` 和 `min` 函数的配置文件信息，我们需要在 `mathlib.pyx` 文件中添加 `profile=True` 选项，如下面的代码片段所示：

```py
    # cython: profile=True 
    cdef int max(int a, int b): 
        # Code here 
```

我们现在可以使用 IPython 的 `%prun` 来对脚本进行性能分析，如下所示：

```py
    import cheb 
    %prun cheb.benchmark() 
# Output:
2000005 function calls in 2.066 seconds 
  Ordered by: internal time 
  ncalls tottime percall cumtime percall 
  filename:lineno(function) 
       1   1.664   1.664   2.066   2.066 
 cheb.py:4(benchmark) 
 1000000   0.351   0.000   0.401   0.000
 {distance.chebyshev} 
 1000000   0.050   0.000   0.050   0.000 mathlib.pyx:2(max) 
       2   0.000   0.000   0.000   0.000 {method 'rand' of 
'mtrand.RandomState' objects} 
       1   0.000   0.000   2.066   2.066 
 <string>:1(<module>) 
       1   0.000   0.000   0.000   0.000 {method 'disable' 
 of        '_lsprof.Profiler' objects} 
```

从输出中，我们可以看到 `max` 函数存在并且不是瓶颈。大部分时间似乎都花在了 `benchmark` 函数上，这意味着瓶颈很可能是纯 Python 的 `for` 循环。在这种情况下，最佳策略将是将循环重写为 NumPy 或将代码移植到 Cython。

Python 中许多用户都喜欢的特性之一是能够与 Jupyter 笔记本一起工作。当使用 Cython 时，你不必放弃这个特性。在本章的下一节和最后一节中，我们将看到如何使用 Cython 与 Jupyter 一起使用。

# 使用 Cython 与 Jupyter

优化 Cython 代码需要大量的尝试和错误。幸运的是，Cython 工具可以通过 Jupyter 笔记本方便地访问，以获得更流畅和集成的体验。

你可以通过在命令行中输入 `jupyter notebook` 来启动笔记本会话，你可以在一个单元中输入 `%load_ext cython` 来加载 Cython 魔法。

如前所述，`%%cython` 魔法可以用于在当前会话中编译和加载 Cython 代码。例如，我们可以将 `cheb.py` 文件的内容复制到一个笔记本单元格中，如下所示：

```py
    %%cython
    import numpy as np
    cdef int max(int a, int b):
        return a if a > b else b
    cdef int chebyshev(int x1, int y1, int x2, int y2):
        return max(abs(x1 - x2), abs(y1 - y2))
    def c_benchmark():
        a = np.random.rand(1000, 2)
        b = np.random.rand(1000, 2)
        for x1, y1 in a:
           for x2, y2 in b:
               chebyshev(x1, x2, y1, y2)
```

`%%cython` 魔法的有用特性是 `-a` 选项，它将编译代码并直接在笔记本中生成源代码的注释视图（就像 `-a` 命令行选项一样），如下面的截图所示：

![图 4.3 – 生成的注释代码](img/B17499_Figure_4.3.jpg)

图 4.3 – 生成的注释代码

这允许您快速测试代码的不同版本，并使用 Jupyter 中可用的其他集成工具。例如，我们可以使用 `%prun` 和 `%timeit` 等工具在同一会话中计时和性能分析代码（前提是我们激活了单元格中的性能分析指令）。我们还可以通过利用 `%prun` 魔法来检查性能分析结果，如下面的截图所示：

![图 4.4 – 性能分析输出](img/B17499_Figure_4.4.jpg)

图 4.4 – 性能分析输出

还可以直接在笔记本中使用我们之前在 *第一章* 中讨论的 `line_profiler` 工具，即 *基准测试和性能分析*。为了支持行注释，需要执行以下操作：

+   启用 `linetrace=True` 和 `binding=True` 编译器指令。

+   在编译时启用 `CYTHON_TRACE=1` 标志。

这可以通过向 `%%cython` 魔法添加相应的参数以及设置编译器指令来实现，如下面的代码片段所示：

```py
    %%cython -a -f -c=-DCYTHON_TRACE=1
    # cython: linetrace=True
    # cython: binding=True
    import numpy as np
    cdef int max(int a, int b):
        return a if a > b else b
    def chebyshev(int x1, int y1, int x2, int y2):
        return max(abs(x1 - x2), abs(y1 - y2))
    def c_benchmark():
        a = np.random.rand(1000, 2)
        b = np.random.rand(1000, 2)

        for x1, y1 in a:
            for x2, y2 in b:
                chebyshev(x1, x2, y1, y2)
```

一旦代码被调试，我们就可以通过 `pip install line_profiler` 命令安装 `line_profiler` 包，并使用 `%lprun` 魔法进行性能分析，如下所示：

```py
%load_ext line_profiler
%lprun -f c_benchmark c_benchmark()
# Output:
Timer unit: 1e-06 s
Total time: 2.322 s
File: /home/gabriele/.cache/ipython/cython/_cython_
magic_18ad8204e9d29650f3b09feb48ab0f44.pyx
Function: c_benchmark at line 11
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    11                                           def 
c_benchmark():
    12         1          226    226.0      0.0      a = 
np.random.rand...
    13         1           67     67.0      0.0      b = 
np.random.rand...    
    14                                               
    15      1001         1715      1.7      0.1      for 
x1, y1 in a:
    16   1001000      1299792      1.3     56.0          
for x2, y2 in b:
    17   1000000      1020203      1.0     43.9              
chebyshev...
```

如您所见，大部分时间都花在了 *第 16 行*，这是一个纯 Python 循环，是进一步优化的良好候选。

Jupyter Notebook 中的工具允许快速编辑-编译-测试周期，以便您可以快速原型设计并在测试不同解决方案时节省时间。

# 摘要

Cython 是一个工具，它将 Python 的便利性与 C 的速度相结合。与 C 绑定相比，Cython 程序更容易维护和调试，这得益于与 Python 的紧密集成和兼容性，以及优秀工具的可用性。

在本章中，我们介绍了 Cython 语言的基础知识以及如何通过给变量和函数添加静态类型来使我们的程序运行更快。我们还学习了如何与 C 数组、NumPy 数组和 memoryviews 一起工作。

我们通过重写关键的 `evolve` 函数来优化我们的粒子模拟器，获得了巨大的速度提升。最后，我们学习了如何使用注释视图来查找难以发现的与解释器相关的调用，以及如何在 Cython 中启用 `cProfile` 支持。此外，我们还学习了如何利用 Jupyter Notebook 对 Cython 代码进行集成分析和性能分析。

所有这些任务都为我们提供了与使用 Python 时相同的高级别灵活性，同时允许我们的程序通过低级 C 代码进行更优化的操作。

在下一章中，我们将探讨其他可以在不要求将我们的代码编译为 C 的情况下即时生成快速机器代码的工具（**AOT**）。

# 问题

1.  实现静态类型有什么好处？

1.  memoryview 有什么好处？

1.  本章介绍了哪些用于分析 Cython 的工具？
