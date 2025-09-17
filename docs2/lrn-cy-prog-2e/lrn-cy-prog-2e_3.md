# 第三章：扩展应用程序

如前几章所述，我想向你展示如何使用 Cython 与现有代码交互或扩展。所以，让我们直接开始做吧。Cython 最初是为了使原始 Python 计算更快而设计的。因此，Cython 的初始概念验证是允许程序员使用 Cython 的 `cdef` 关键字将现有 Python 代码转换为原生类型，以绕过 Python 运行时进行重计算。这一成果是计算时间的性能提升和内存使用降低。甚至可以编写类型安全的包装器来为完全类型化的 Python 代码扩展现有 Python 库。

在本章中，我们将首先看到一个编写 Python 代码的示例。接下来，我将演示 Cython 的 `cdef` 类，它允许我们将原生 C/C++ 类型包装成垃圾回收的 Python 类。我们还将看到如何通过创建纯 Python 命令对象来扩展原生应用程序 **Tmux**，该对象直接嵌入到原生代码中。

在本章中，我们将涵盖以下主题：

+   Cython 纯 Python 代码

+   编译纯 Python 代码

+   Python 垃圾回收器

+   扩展 Tmux

+   嵌入 Python

+   Cython 化 struct cmd_entry

+   实现一个 Tmux 命令

# Cython 纯 Python 代码

让我们查看一个实际上来自 Cython 文档的数学应用。我将其等效地用纯 Python 编写，以便我们可以比较速度。如果你打开本章的 `primes` 示例，你会看到两个程序——Cython 的 `primes.pyx` 示例和我的纯 Python 版本。它们看起来几乎相同：

```py
def primes(kmax):
    n = 0
    k = 0
    i = 0
    if kmax > 1000:
        kmax = 1000
    p = [0] * kmax
    result = []
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result
primes (10000)
```

这实际上是将 Cython 代码直接转换为 Python。两者都调用 `primes (10000)`，但它们在性能方面的评估时间差异很大：

```py
$ make
cython --embed primes.pyx
gcc -g -O2 -c primes.c -o primes.o `python-config --includes`
gcc -g -O2 -o primes primes.o `python-config –libs`
$ time python pyprimes.py
        0.18 real         0.17 user         0.01 sys
$ time ./primes
        0.04 real         0.03 user         0.01 sys
```

你可以看到，纯 Python 版本在执行相同任务时几乎慢了五倍。此外，几乎每一行代码都是相同的。Cython 可以做到这一点，因为我们已经明确地表达了 C 类型，因此没有类型转换或折叠，我们甚至不需要使用 Python 运行时。我想强调的是，仅通过简单的代码而不调用其他原生库就能获得的速度提升。这就是为什么 Cython 在 SAGE 中如此普遍。

# 编译纯 Python 代码

Cython 的另一个用途是编译 Python 代码。例如，如果我们回到 `primes` 示例，我们可以做以下操作：

```py
$ cython pyprimes.py –embed
$ gcc -g -O2 pyprimes.c -o pyprimes `python-config --includes –libs`
```

然后，我们可以比较同一程序的三个不同版本：使用 `cdef` 为原生类型编写的 Cython 版本，作为 Python 脚本运行的纯 Python 版本，以及最终，由 Cython 编译的纯 Python 版本，它生成 Python 代码的可执行二进制文件：

+   首先，使用原生类型的 Cython 版本：

    ```py
    $ time ./primes
    real    0m0.050s
    user    0m0.035s
    sys     0m0.013s

    ```

+   接下来，可执行的纯 Python 版本：

    ```py
    $ time ./pyprimes
    real    0m0.139s
    user    0m0.122s
    sys     0m0.013s

    ```

+   最后，Python 脚本版本：

    ```py
    philips-macbook:primes redbrain$ time python pyprimes.py
    real    0m0.184s
    user    0m0.165s
    sys     0m0.016s

    ```

纯 Python 版本运行速度最慢，编译后的 Python 版本运行速度略快，最后，原生类型化的 Cython 版本运行速度最快。我认为这仅仅突出了 Cython 以几种不同的方式为你提供一些动态语言优化的能力。

注意，当将 Python 版本编译成二进制文件时，我在调用 Cython 编译器时指定了 `–embed`。这告诉编译器为我们嵌入一个主方法，并按预期运行正常的 Python 脚本。

## 避免使用 Makefile – pyximport

从前面的例子中，你可以看到这是不依赖于任何外部库的代码。要使这样的代码有用，如果我们可以绕过 Makefile 和编译器的调用，那岂不是很好？实际上，在不需要链接其他原生库的情况下，我们可以直接将 `.pyx` 文件导入到 Python 程序中。然而，你仍然需要将 Cython 作为依赖项安装。

回到第一章，“Cython 不会咬人”，我们可以通过首先导入 `pyximport` 来简单地导入我们的 `helloworld.pyx`：

```py
>>> import pyximport
>>> pyximport.install()
(None, <pyximport.pyximport.PyxImporter object at 0x102fba4d0>)
>>> import helloworld
Hello World from cython!
```

在幕后，Cython 会为你调用所有的编译工作，这样你就不必亲自做了。但这引发了一些有趣的想法，比如你只需将 Cython 代码添加到任何 Python 项目中，只要 Cython 是一个依赖项即可。

# Python 垃圾回收器

当包装原生 `struct` 时，例如，可能会非常诱人遵循标准的 C/C++ 习惯用法，并要求 Python 程序员手动调用、分配和释放不同的对象。这非常繁琐，而且不太符合 Python 风格。Cython 允许我们创建 `cdef` 类，这些类有额外的初始化和析构钩子，我们可以使用这些钩子来控制 `struct` 的所有内存管理。这些钩子会自动由 Python 垃圾回收器触发，使一切变得简单。考虑以下简单的 `struct`：

```py
typedef struct data {
 int value;
} data_t;
```

我们可以将 C `struct` 的 Cython 声明写入 `PyData.pxd`，如下所示：

```py
cdef extern from "Data.h":
    struct data:
        int value
    ctypedef data data_t
```

现在我们已经定义了 `struct`，我们可以将 `struct` 包装成一个类：

```py
cimport PyData

cdef class Data(object):
    cdef PyData.data_t * _nativeData
    …
```

将数据包装成这样的类将需要我们在正确的时间分配和释放内存。幸运的是，Cython 几乎暴露了所有的 `libc` 作为导入：

```py
from libc.stdlib cimport malloc, free
```

现在我们能够分配和释放内存，剩下要做的只是理解类的生命周期以及在哪里进行钩子。Cython 类提供了两个特殊方法：`__cinit__` 和 `__dealloc__`。`__cinit__` 提供了一种实例化原生代码的方式，因此在我们的例子中，我们将内存分配给原生 C `struct`，正如你可以猜到的，在析构时这是垃圾回收器的销毁钩子，并给我们一个机会来释放任何已分配的资源：

```py
def __cinit__(self):
        self._nativeData = <data_t*>malloc(sizeof(data_t))
        if not self._nativeData:
            self._nativeData = NULL
            raise MemoryError()

def __dealloc__(self):
        if self._nativeData is not NULL:
            free(self._nativeData)
            self._nativeData = NULL
```

需要注意的是，`__cinit__`不会覆盖 Python 的`__init__`，更重要的是，`__cinit__`在此点并不设计为调用任何 Python 代码，因为它不能保证类的完全初始化。一个初始化方法可能如下所示：

```py
def __init__(self, int value):
        self.SetValue(value)

def SetValue(self, int value):
        self.SetNativeValue(value)

cdef SetNativeValue(self, int value):
        self._nativeData.value = value
```

注意，我们能够在这类函数上输入参数以确保我们不会尝试将 Python 对象放入`struct`中，这将会失败。这里令人印象深刻的是，这个类表现得就像是一个普通的 Python 类：

```py
from PyData import Data

def TestPythonData():
    # Looks and feels like normal python objects
    objectList = [Data(1), Data(2), Data(3)]

    # Print them out
    for dataObject in objectList:
        print dataObject

    # Show the Mutability
    objectList[1].SetValue(1234)
    print objectList[1]
```

如果你将一个简单的`print`语句放在`__dealloc__`钩子上并运行程序，你会看到所有析构函数都按预期执行。这意味着我们刚刚在原生代码上利用了 Python 垃圾回收器。

# 扩展 Tmux

**Tmux**是一个受 GNU Screen ([`tmux.github.io/`](http://tmux.github.io/))启发的终端多路复用器，但它支持更简单、更好的配置。更重要的是，它的实现更干净、更容易维护，并且它还使用了`libevent`和非常优秀的 C 代码。

我想向你展示如何通过编写 Python 代码而不是 C 代码来扩展 Tmux 的新内置命令。总的来说，这个项目有几个部分，如下所示：

+   修改 autotool 的构建系统以编译 Cython

+   为相关声明，如`struct cmd_entry`创建 PXD 声明

+   将 Python 嵌入到 Tmux 中

+   将 Python 命令添加到全局 Tmux `cmd_table`

让我们快速查看 Tmux 的源代码，特别是包含命令声明和实现的任何`cmd-*.c`文件。例如，`cmd-kill-window.c`是命令入口。这告诉 Tmux 命令的名称、别名以及它是否接受参数；最后，它接受一个指向实际命令代码的函数指针：

```py
const struct cmd_entry cmd_kill_window_entry = {
  "kill-window", "killw",
  "at:", 0, 0,
  "[-a] " CMD_TARGET_WINDOW_USAGE,
  0,
  NULL,
  NULL,
 cmd_kill_window_exec
};
```

因此，如果我们能够实现并初始化包含这些信息的自己的`struct`，我们就可以运行我们的`cdef`代码。接下来，我们需要查看 Tmux 如何获取这个命令定义以及它是如何执行的。

如果我们查看`tmux.h`，我们会找到我们需要操作的所有内容的原型：

```py
extern const struct cmd_entry *cmd_table[];
extern const struct cmd_entry cmd_attach_session_entry;
extern const struct cmd_entry cmd_bind_key_entry;
….
```

因此，我们需要为我们的`cmd_entry`定义在这里添加一个原型。接下来，我们需要查看`cmd.c`；这是命令表初始化的地方，以便稍后可以查找以执行命令：

```py
const struct cmd_entry *cmd_table[] = {
  &cmd_attach_session_entry,
  &cmd_bind_key_entry,
…
```

现在命令表已经初始化，代码在哪里执行呢？如果我们查看`tmux.h`头文件中的`cmd_entry`定义，我们可以看到以下内容：

```py
/* Command definition. */
struct cmd_entry {
 const char  *name;
 const char  *alias;

  const char  *args_template;
  int     args_lower;
  int     args_upper;

 const char  *usage;

#define CMD_STARTSERVER 0x1
#define CMD_CANTNEST 0x2
#define CMD_SENDENVIRON 0x4
#define CMD_READONLY 0x8
  int     flags;

  void     (*key_binding)(struct cmd *, int);
  int     (*check)(struct args *);
 enum cmd_retval   (*execc)(struct cmd *, struct cmd_q *);
};
```

`execc`钩子是我们真正关心的函数指针，所以如果你`grep`源代码，你应该会找到以下内容：

```py
Philips-MacBook:tmux-project redbrain$ ack-5.12 execc
tmux-1.8/cmd-queue.c
229:               retval = cmdq->cmd->entry->execc(cmdq->cmd, cmdq);

```

你可能会注意到在官方的 Tmux Git 中，这个钩子简单地命名为`exec`。我将其重命名为`execc`，因为`exec`是 Python 中的保留字——我们需要避免这种情况。首先，让我们编译一些代码。首先，我们需要让构建系统发挥作用。

## Tmux 构建系统

Tmux 使用 autotools，因此我们可以重用第二章，*理解 Cython*中的片段，以添加 Python 支持。我们可以在`configure.ac`中添加`–enable-python`开关，如下所示：

```py
# want python support for pytmux scripting
found_python=no
AC_ARG_ENABLE(
  python,
  AC_HELP_STRING(--enable-python, create python support),
  found_python=yes
)
AM_CONDITIONAL(IS_PYTHON, test "x$found_python" = xyes)

PYLIBS=""
PYINCS=""
if test "x$found_python" = xyes; then
 AC_CHECK_PROG(CYTHON_CHECK,cython,yes)
   if test x"$CYTHON_CHECK" != x"yes" ; then
      AC_MSG_ERROR([Please install cython])
   fi
 AC_CHECK_PROG(PYTHON_CONF_CHECK,python-config,yes)
 PYLIBS=`python-config --libs`
 PYINCS=`python-config --includes`
   if test "x$PYLIBS" == x; then
      AC_MSG_ERROR("python-dev not found")
   fi
 AC_DEFINE(HAVE_PYTHON)
fi
AC_SUBST(PYLIBS)
AC_SUBST(PYINCS)
```

这给我们提供了`./configure –-enable-python`选项。接下来，我们需要查看`Makefile.am`文件。让我们将我们的 Cython 文件命名为`cmdpython.pyx`。请注意，Cython 不喜欢文件名中的尴尬字符，如`-`，如第二章中所述。如果我们想在构建时使 Python 支持条件选项，我们应该将以下内容添加到`Makefile.am`中：

```py
if IS_PYTHON
PYTHON_SOURCES = cmdpython.pyx
else
PYTHON_SOURCES =
endif

# List of sources.
dist_tmux_SOURCES = \
 $(PYTHON_SOURCES) \
...
```

我们必须确保它首先被需要并编译。记住，如果我们创建`public`声明，Cython 会为我们生成一个头文件。我们只需将我们的公共头文件添加到`tmux.h`中，以保持头文件非常简单。然后，为了确保 Cython 文件在构建时被 automake 识别并正确编译，根据正确的依赖管理，我们需要添加以下内容：

```py
SUFFIXES = .pyx
.pyx.c:
  @echo "  CPY   " $<
 @cython -2 -o $@ $<

```

这添加了后缀规则，以确保`*.pyx`文件被 Cython 化，然后像任何正常的 C 文件一样编译生成的`.c`文件。如果你在 autotools 项目中恰好使用了`AM_SILENT_RULES([yes])`，这个片段运行得很好，因为它正确地格式化了 echo 消息。最后，我们需要确保在配置脚本中的`AC_SUBST`中添加必要的`CFLAGS`和`LIBS`选项到编译器：

```py
CFLAGS += $(PYINCS)
tmux_LDADD = \
 $(PYLIBS)

```

现在你的构建系统应该已经准备好了，但由于所做的更改，我们必须现在重新生成 autotools 的内容。只需运行`./autogen.sh`。

# 嵌入 Python

现在我们有文件正在编译，我们需要初始化 Python。我们的模块。Tmux 是一个分叉的服务器，客户端连接到，所以尽量不要把它看作一个单线程系统。它是一个客户端*和*服务器，所以所有命令都在服务器上执行。现在，让我们找到服务器中事件循环开始的地方，并在服务器这里初始化和最终化，以确保正确完成。查看`int server_start(int lockfd, char *lockfile)`，我们可以添加以下内容：

```py
#ifdef HAVE_PYTHON
 Py_InitializeEx (0);
#endif
  server_loop();
#ifdef HAVE_PYTHON
  Py_Finalize ();
#endif
```

Python 现在被嵌入到 Tmux 服务器中。注意，我使用的是`Py_InitializeEx (0)`而不是简单地使用`Py_Initialize`。这复制了相同的行为，但不会启动正常的 Python 信号处理器。Tmux 有自己的信号处理器，所以我不想覆盖它们。当扩展像这样的现有应用程序时，使用`Py_InitializeEx (0)`可能是一个好主意，因为它们通常实现自己的信号处理。使用这个选项可以阻止 Python 尝试处理会冲突的信号。

# Cython 化 struct cmd_entry

接下来，让我们考虑创建一个`cythonfile.pxd`文件，用于 Tmux 必要的`cdef`声明，我们需要了解的。我们需要查看`struct cmd_entry`声明，并从这个声明反向工作：

```py
struct cmd_entry {
  const char  *name;
  const char  *alias;

  const char  *args_template;
  int     args_lower;
  int     args_upper;

  const char  *usage;
  int     flags;

  void     (*key_binding)(struct cmd *, int);
  int     (*check)(struct args *);
 enum cmd_retval   (*execc)(struct cmd *, struct cmd_q *);
};
```

如您所见，`cmd_entry`依赖于几个其他类型，因此我们需要稍微回溯一下。如果您想偷懒并冒险，如果您不关心通过类型转换任何指针（如`void *`）来正确访问数据，有时您可以侥幸逃脱。但如果你是一个经验丰富的 C 程序员，你知道这是相当危险的，应该避免。您可以看到这个类型依赖于`struct cmd *`、`struct cmd_q *`和`struct args *`。我们理想情况下想在某个时刻访问这些，所以逐个实现它们是个好主意，因为其余的都是原生 C 类型，Cython 可以理解。

实现`enum`应该是迄今为止最简单的：

```py
/* Command return values. */
enum cmd_retval {
  CMD_RETURN_ERROR = -1,
  CMD_RETURN_NORMAL = 0,
  CMD_RETURN_WAIT,
  CMD_RETURN_STOP
};
```

然后，将其转换为以下形式：

```py
cdef enum cmd_retval:
        CMD_RETURN_ERROR = -1
        CMD_RETURN_NORMAL = 0
        CMD_RETURN_WAIT = 1
        CMD_RETURN_STOP = 2
```

现在我们有了`exec`钩子的返回值，接下来我们需要查看`struct cmd`并实现它：

```py
struct cmd {
  const struct cmd_entry  *entry;
  struct args    *args;

  char      *file;
  u_int       line;

  TAILQ_ENTRY(cmd)   qentry;
};
```

看一下`TAILQ_ENTRY`。这是一个简单的预处理宏，是**BSD libc**的扩展，可以将任何类型转换为它自己的链表。我们可以忽略它：

```py
 cdef struct cmd:
 cmd_entry * entry
 args * aargs
        char * file
        int line
```

注意，这个`struct`依赖于`struct cmd_entry`和`struct args`的定义，我们还没有实现。现在不用担心这个问题；暂时先放它们在这里。接下来，让我们实现`struct args`，因为它很简单：

```py
/* Parsed arguments. */
struct args {
 bitstr_t  *flags;
 char    *values[SCHAR_MAX];

  int     argc;
  char         **argv;
};
```

注意，它使用了`bitstr_t`和可变长度的数组列表。我选择忽略`bitstr_t`，因为它是一个系统依赖的头文件，实现起来相当棘手。让我们简单地将它们转换为`char *`和`char **`以使事情运行起来：

```py
 cdef struct args:
 char * flags
 char **values
        int argc
        char **argv
```

现在已经将`args`结构 Cython 化，让我们实现`struct cmd_q`，这稍微有点棘手：

```py
/* Command queue. */
struct cmd_q {
  int       references;
  int       dead;

 struct client    *client;
  int       client_exit;

 struct cmd_q_items   queue;
 struct cmd_q_item  *item;
 struct cmd    *cmd;

  time_t       time;
  u_int       number;

  void       (*emptyfn)(struct cmd_q *);
  void      *data;

 struct msg_command_data  *msgdata;

 TAILQ_ENTRY(cmd_q)       waitentry;
};
```

还有许多其他结构依赖于它，但在这里我们不会看到它们。让我们现在尝试将这些转换为`void *`；例如，`struct client *`。我们可以将其转换为`void *`，然后`struct cmd_q_items`简单地转换为`int`，即使这不正确。只要我们不打算尝试访问这些字段，我们就会没事。但请记住，如果我们使用 Cython 的`sizeof`，我们可能会遇到由 C 和 Cython 分配的不同大小的内存损坏。我们可以继续处理其他类型，如`struct cmd_q_item *`，并将它们再次转换为`void *`。最后，我们来到`time_t`，我们可以重用 Cython 的`libc.stdlib cimport` time。这是一个很好的练习，为 C 应用程序实现 Cython 声明；它真正锻炼了你的代码分析能力。在处理非常长的结构时，请记住，我们可以通过将它们转换为`void`来使事情运转起来。如果你关心你的 Cython API 中的数据类型，请小心处理`struct`的对齐和类型：

```py
 cdef struct cmd_q:
        int references
        int dead
 void * client
        int client_exit
        int queue
 void * item
 cmd * cmd
        int time
        int number
        void (*emptyfn)(cmd_q *)
 void * msgdata

```

这是对许多项目特定内部结构的深入探讨，但我希望您能理解——我们实际上并没有做什么特别可怕的事情。我们甚至作弊了，将我们实际上并不关心的东西进行了类型转换。在实现了所有这些辅助类型之后，我们最终可以实施我们关心的类型，即`struct cmd_entry`：

```py
cdef struct cmd_entry:
        char * name
        char * alias
        char * args_template
        int args_lower
        int args_upper
        char * usage
        int flags
        void (*keybinding)(cmd *, int)
        int (*check)(args *)
        cmd_retval (*execc)(cmd *, cmd_q *)
```

通过这个`cmdpython.pxd`文件，我们现在可以实现我们的 Tmux 命令！

# 实现 Tmux 命令

Cython 有一个注意事项是我们不能像在 C 中那样静态初始化结构体，因此我们需要创建一个钩子，以便在 Python 启动时初始化`cmd_entry`：

```py
cimport cmdpython

cdef public cmd_entry cmd_entry_python
```

通过这种方式，我们现在有了`cmd_entry_python`的公共声明，我们将在启动钩子中初始化它，如下所示：

```py
cdef public void tmux_init_cython () with gil:
 cmd_entry_python.name = "python"
    cmd_entry_python.alias = "py"
    cmd_entry_python.args_template = ""
    cmd_entry_python.args_lower = 0
    cmd_entry_python.args_upper = 0
    cmd_entry_python.usage = "python usage..."
    cmd_entry_python.flags = 0
    #cmd_entry_python.key_binding = NULL
    #cmd_entry_python.check = NULL
 cmd_entry_python.execc = python_exec

```

记住，因为我们是在顶层声明的，所以我们知道它位于堆上，不需要向结构体声明任何内存，这对我们来说非常方便。你之前已经见过结构体的访问方式；函数套件应该看起来很熟悉。但让我在这里强调几点：

+   我们声明`public`是为了确保我们可以调用它。

+   执行钩子只是一个`cdef` Cython 函数。

+   最后，你可能注意到了`gil`。我将在第五章 *高级 Cython* 中解释这个用于什么。

现在，让我们看看一个简单的执行钩子：

```py
cdef cmd_retval python_exec (cmd * cmd, cmd_q * cmdq) with gil:
    cdef char * message = "Inside your python command inside tmux!!!"
    log_debug (message)
    return CMD_RETURN_NORMAL;
```

现在将这个钩子连接到 Tmux 没有太多剩余的工作要做。只需将其添加到`cmd_table`，并将启动钩子添加到服务器初始化中。

### 注意

注意，我在`log_debug`函数中向 PXD 添加了一些内容；如果你查看 Tmux，这是一个`VA_ARGS`函数。Cython 目前还不理解这些，但我们可以通过将其转换为接受字符串的函数来简单地“黑客”它，让它运行起来。只要我们不尝试像使用任何`printf`一样使用它，我们应该就没事了。

# 将一切连接起来

现在，我们还需要对 Tmux 进行一点小小的调整，但这并不痛苦，一旦完成，我们就可以自由地发挥创意。从根本上说，我们应该在忘记之前在`server.c`中调用`cmd_entry`初始化钩子：

```py
#ifdef HAVE_PYTHON
  Py_InitializeEx (0);
 tmux_init_cython ();
#endif

  server_loop();

#ifdef HAVE_PYTHON
  Py_Finalize ();
#endif
```

现在已经完成，我们需要确保将`cmd_entry_python`外部声明添加到`tmux.h`中：

```py
extern const struct cmd_entry cmd_wait_for_entry;
#ifdef HAVE_PYTHON
# include "cmdpython.h"
#endif
```

最后，将其添加到`cmd_table`：

```py
const struct cmd_entry *cmd_table[] = {
  &cmd_attach_session_entry,
  &cmd_bind_key_entry,
  &cmd_break_pane_entry,
…
  &cmd_wait_for_entry,
 &cmd_entry_python,
  NULL
};
```

现在已经完成，我认为我们可以开始了——让我们测试一下这个小家伙。用以下方式编译 Tmux：

```py
$ ./configure –enable-python
$ make
$ ./tmux -vvv
$ tmux: C-b :python
$ tmux: exit

```

我们可以查看`tmux-server-*.log`来查看我们的调试信息：

```py
complete key ^M 0xd
cmdq 0xbb38f0: python (client 8)
Inside your python command inside tmux!!!
keys are 1 (e)

```

我希望你现在可以看到，你可以很容易地将其扩展到做你自己的选择，比如使用 Python 库直接调用你的音乐播放器，并且所有这些都将与 Tmux 集成在一起。

# 概述

本章展示了许多不同的技术和想法，但它应该作为常见技术的强大参考。我们看到了使用本地类型绕过运行时的加速，并将编译的 Python 代码编译成自己的二进制文件。`pyximport`语句显示我们可以绕过编译，简单地导入`.pyx`文件，就像它是普通的 Python 文件一样。最后，我在本章的结尾通过逐步演示我的过程，展示了如何将 Python 嵌入到 Tmux 中。在下一章中，我们将看到使用`gdb`进行调试的实际操作，以及使用 Cython 的一些注意事项。
