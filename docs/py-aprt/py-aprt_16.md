## 第十二章：使用 PDB 进行调试

即使有全面的自动化测试套件，我们仍然可能遇到需要调试器来弄清楚发生了什么的情况。幸运的是，Python 包含了一个强大的调试器，即标准库中的 PDB。PDB 是一个命令行调试器，如果您熟悉像 GDB 这样的工具，那么您已经对如何使用 PDB 有了一个很好的了解。

PDB 相对于其他 Python 调试器的主要优势在于，作为 Python 本身的一部分，PDB 几乎可以在 Python 存在的任何地方使用，包括将 Python 语言嵌入到较大系统中的专用环境，例如 ESRI 的*ArcGIS*地理信息系统。也就是说，使用所谓的*图形*调试器可能会更加舒适，例如*Jetbrains*的*PyCharm*或*Microsoft*的*Python Tools for Visual Studio*中包含的调试器。您应该随时跳过本章，直到熟悉 PDB 变得更加紧迫；您不会错过我们在本书中或在*Python 学徒*或*Python 大师*中依赖的任何内容。

PDB 与许多调试工具不同，它实际上并不是一个单独的程序，而是像任何其他 Python 模块一样的模块。您可以将`pdb`导入任何程序，并使用`set_trace()`函数调用启动调试器。此函数在程序执行的任何点开始调试器。

对于我们对 PDB 的第一次尝试，让我们使用 REPL 并使用`set_trace()`启动调试器：

```py
>>> import pdb
>>> pdb.set_trace()
--Return--
> <stdin>(1)<module>()->None
(Pdb)

```

您会看到在执行`set_trace()`后，您的提示从三个尖括号变为`(Pdb)`-这是您知道自己在调试器中的方式。

### 调试命令

我们要做的第一件事是查看调试器中有哪些命令，方法是键入`help`：

```py
(Pdb) help

Documented commands (type help <topic>):
========================================
EOF    cl         disable  interact  next     return  u          where
a      clear      display  j         p        retval  unalias
alias  commands   down     jump      pp       run     undisplay
args   condition  enable   l         print    rv      unt
b      cont       exit     list      q        s       until
break  continue   h        ll        quit     source  up
bt     d          help     longlist  r        step    w
c      debug      ignore   n         restart  tbreak  whatis

Miscellaneous help topics:
==========================
pdb  exec

```

这列出了几十个命令，其中一些你几乎在每个调试会话中都会使用，而另一些你可能根本不会使用。

您可以通过键入`help`后跟命令名称来获取有关命令的具体帮助。例如，要查看`continue`的功能，请键入`help continue`：

```py
    (Pdb) help continue
    c(ont(inue))
            Continue execution, only stop when a breakpoint is encountered.

```

命令名称中的奇怪括号告诉您，`continue`可以通过键入`c`、`cont`或完整单词`continue`来激活。了解常见 PDB 命令的快捷方式可以极大地提高您在调试时的舒适度和速度。

### 回文调试

我们将不列出所有常用的 PDB 命令，而是调试一个简单的函数。我们的函数`is_palindrome()`接受一个整数，并确定整数的数字是否是回文。回文是一个正向和反向都相同的序列。

我们要做的第一件事是创建一个新文件`palindrome.py`，其中包含以下代码：

```py
import unittest

def digits(x):
    """Convert an integer into a list of digits.

 Args:
 x: The number whose digits we want.

 Returns: A list of the digits, in order, of ``x``.

 >>> digits(4586378)
 [4, 5, 8, 6, 3, 7, 8]
 """

    digs = []
    while x != 0:
        div, mod = divmod(x, 10)
        digs.append(mod)
        x = mod
    digs.reverse()
    return digs

def is_palindrome(x):
    """Determine if an integer is a palindrome.

 Args:
 x: The number to check for palindromicity.

 Returns: True if the digits of ``x`` are a palindrome,
 False otherwise.

 >>> is_palindrome(1234)
 False
 >>> is_palindrome(2468642)
 True
 """
    digs = digits(x)
    for f, r in zip(digs, reversed(digs)):
        if f != r:
            return False
    return True

class Tests(unittest.TestCase):
    """Tests for the ``is_palindrome()`` function."""
    def test_negative(self):
        "Check that it returns False correctly."
        self.assertFalse(is_palindrome(1234))

    def test_positive(self):
        "Check that it returns True correctly."
        self.assertTrue(is_palindrome(1234321))

    def test_single_digit(self):
        "Check that it works for single digit numbers."
        for i in range(10):
            self.assertTrue(is_palindrome(i))

if __name__ == '__main__':
    unittest.main()

```

正如您所看到的，我们的代码有三个主要部分。第一个是`digits()`函数，它将整数转换为数字列表。

第二个是`is_palindrome()`函数，它首先调用`digits()`，然后检查结果列表是否是回文。

第三部分是一组单元测试。我们将使用这些测试来驱动程序。

正如您可能期望的，由于这是一个关于调试的部分，这段代码中有一个错误。我们将首先运行程序并注意到错误，然后我们将看看如何使用 PDB 来找到错误。

#### 使用 PDB 进行错误调试

因此，让我们运行程序。我们有三个测试希望运行，由于这是一个相对简单的程序，我们期望它运行得非常快：

```py
$ python palindrome.py

```

我们看到这个程序似乎运行了很长时间！如果您查看其内存使用情况，还会看到它随着运行时间的增加而增加。显然出现了问题，所以让我们使用 Ctrl-C 来终止程序。

让我们使用 PDB 来尝试理解这里发生了什么。由于我们不知道问题可能出在哪里，也不知道在哪里放置`set_trace()`调用，所以我们将使用命令行调用来在 PDB 的控制下启动程序：

```py
$ python -m pdb palindrome.py
> /Users/sixty_north/examples/palindrome.py(1)<module>()
-> import unittest
(Pdb)

```

在这里，我们使用了`-m`参数，告诉 Python 执行特定的模块 - 在这种情况下是 PDB - 作为脚本。其余的参数传递给该脚本。所以在这里，我们告诉 Python 执行 PDB 模块作为脚本，并将我们的错误文件的名称传递给它。

我们看到的是，我们立即进入了 PDB 提示符。指向`import unittest`的箭头告诉我们，这是我们继续执行时将执行的下一条语句。但是那条语句在哪里？

让我们使用`where`命令来找出：

```py
(Pdb) where
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/bdb.py(387)run()
-> exec cmd in globals, locals
  <string>(1)<module>()
> /Users/sixty_north/examples/palindrome.py(1)<module>()
-> import unittest

```

`where`命令报告我们当前的调用堆栈，最近的帧在底部，我们可以看到 PDB 已经在`palindrome.py`的第一行暂停了执行。这强调了 Python 执行的一个重要方面，我们之前已经讨论过：一切都在运行时评估。在这种情况下，我们在`import`语句之前暂停了执行。

我们可以通过使用`next`命令执行此导入到下一条语句：

```py
(Pdb) next
> /Users/sixty_north/examples/palindrome.py(3)<module>()
-> def digits(x):
(Pdb)

```

我们看到这将我们带到`digits()`函数的`def`调用。当我们执行另一个`next`时，我们移动到`is_palindrome()`函数的定义：

```py
(Pdb) next
> /Users/sixty_north/examples/palindrome.py(12)<module>()
-> def is_palindrome(x):
(Pdb)

```

#### 使用采样查找无限循环

我们可以继续使用`next`来移动程序的执行，但由于我们不知道错误出在哪里，这可能不是一个非常有用的技术。相反，记住我们程序的问题是似乎一直在运行。这听起来很像一个无限循环！

因此，我们不是逐步执行我们的代码，而是让它执行，然后当我们认为我们可能在那个循环中时，我们将使用 Ctrl-C 中断回到调试器：

```py
(Pdb) cont
^C
Program interrupted. (Use 'cont' to resume).
> /Users/sixty_north/examples/palindrome.py(9)digits()
-> x = mod
(Pdb)

```

让程序运行几秒钟后，我们按下 Ctrl-C，这将停止程序并显示我们在`palindrome.py`的`digits()`函数中。如果我们想在那一行看到源代码，我们可以使用 PDB 命令`list`：

```py
(Pdb) list
  4       "Convert an integer into a list of digits."
  5       digs = []
  6       while x != 0:
  7           div, mod = divmod(x, 10)
  8           digs.append(mod)
  9  ->       x = mod
 10       return digs
 11
 12   def is_palindrome(x):
 13       "Determine if an integer is a palindrome."
 14       digs = digits(x)
(Pdb)

```

我们看到这确实是在一个循环内部，这证实了我们的怀疑可能涉及无限循环。

我们可以使用`return`命令尝试运行到当前函数的末尾。如果这不返回，我们将有非常强有力的证据表明这是一个无限循环：

```py
(Pdb) r

```

我们让它运行几秒钟，以确认我们从未退出该函数，然后我们按下 Ctrl-C。一旦我们回到 PDB 提示符，让我们使用`quit`命令退出 PDB：

```py
(Pdb) quit
%

```

#### 设置显式断点

由于我们知道问题出在`digits()`中，让我们使用之前提到的`pdb.set_trace()`函数在那里设置一个显式断点：

```py
def digits(x):
    """Convert an integer into a list of digits.

 Args:
 x: The number whose digits we want.

 Returns: A list of the digits, in order, of ``x``.

 >>> digits(4586378)
 [4, 5, 8, 6, 3, 7, 8]
 """

    import pdb; pdb.set_trace()

    digs = []
    while x != 0:
        div, mod = divmod(x, 10)
        digs.append(mod)
        x = mod
    digs.reverse()
    return digs

```

记住，`set_trace()`函数将停止执行并进入调试器。

所以现在我们可以执行我们的脚本，而不指定 PDB 模块：

```py
% python palindrome.py
> /Users/sixty_north/examples/palindrome.py(8)digits()
-> digs = []
(Pdb)

```

我们看到我们几乎立即进入 PDB 提示符，执行在我们的`digits()`函数的开始处暂停。

为了验证我们知道我们在哪里，让我们使用`where`来查看我们的调用堆栈：

```py
(Pdb) where
  /Users/sixty_north/examples/palindrome.py(35)<module>()
-> unittest.main()
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/main.py(95\
)__init__()
-> self.runTests()
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/main.py(22\
9)runTests()
-> self.result = testRunner.run(self.test)
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/runner.py(\
151)run()
-> test(result)
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/suite.py(7\
0)__call__()
-> return self.run(*args, **kwds)
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/suite.py(1\
08)run()
-> test(result)
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/suite.py(7\
0)__call__()
-> return self.run(*args, **kwds)
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/suite.py(1\
08)run()
-> test(result)
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/case.py(39\
1)__call__()
-> return self.run(*args, **kwds)
  /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/unittest/case.py(32\
7)run()
-> testMethod()
  /Users/sixty_north/examples/palindrome.py(25)test_negative()
-> self.assertFalse(is_palindrome(1234))
  /Users/sixty_north/examples/palindrome.py(17)is_palindrome()
-> digs = digits(x)
> /Users/sixty_north/examples/palindrome.py(8)digits()
-> digs = []

```

记住，最近的帧在此列表的末尾。经过很多`unittest`函数后，我们看到我们确实在`digits()`函数中，并且它是由`is_palindrome()`调用的，正如我们所预期的那样。

#### 逐步执行

现在我们要做的是观察执行，并看看为什么我们从未退出这个函数的循环。让我们使用`next`移动到循环体的第一行：

```py
(Pdb) next
> /Users/sixty_north/examples/palindrome.py(9)digits()
-> while x != 0:
(Pdb) next
> /Users/sixty_north/examples/palindrome.py(10)digits()
-> div, mod = divmod(x, 10)
(Pdb)

```

现在让我们看一下一些变量的值，并尝试决定我们期望发生什么。我们可以使用`print`命令来检查值^(34)：

```py
(Pdb) print(digs)
[]
(Pdb) print x
1234

```

这看起来是正确的。`digs`列表 - 最终将包含数字序列 - 是空的，`x`是我们传入的。我们期望`divmod()`函数返回`123`和`4`，所以让我们试试看：

```py
(Pdb) next
> /Users/sixty_north/examples/palindrome.py(11)digits()
-> digs.append(mod)
(Pdb) print div,mod
123 4

```

这看起来正确：`divmod()`已经从我们的数字中剪掉了最低有效位数字，下一行将该数字放入我们的结果列表中：

```py
(Pdb) next
> /Users/sixty_north/examples/palindrome.py(12)digits()
-> x = mod

```

如果我们查看`digs`，我们会看到它现在包含`mod`：

```py
(Pdb) print digs
[4]

```

下一行现在将更新`x`，以便我们可以继续从中剪切数字：

```py
(Pdb) next
> /Users/sixty_north/examples/palindrome.py(9)digits()
-> while x != 0:

```

我们看到执行回到了 while 循环，正如我们所预期的那样。让我们查看`x`，确保它有正确的值：

```py
(Pdb) print x
4

```

等一下！我们期望`x`保存的是不在结果列表中的数字。相反，它只包含结果列表中的数字。显然我们在更新`x`时犯了一个错误！

如果我们查看我们的代码，很快就会发现我们应该将`div`而不是`mod`分配给`x`。让我们退出 PDB：

```py
(Pdb) quit

```

请注意，由于 PDB 和`unittest`的交互方式，您可能需要运行几次`quit`。

#### 修复错误

当您退出 PDB 后，让我们删除`set_trace()`调用并修改`digits()`来解决我们发现的问题：

```py
def digits(x):
    """Convert an integer into a list of digits.

    Args:
      x: The number whose digits we want.

    Returns: A list of the digits, in order, of ``x``.

    >>> digits(4586378)
    [4, 5, 8, 6, 3, 7, 8]
    """

    digs = []
    while x != 0:
        div, mod = divmod(x, 10)
        digs.append(mod)
        x = div
    digs.reverse()
    return digs

```

如果我们现在运行我们的程序，我们会看到我们通过了所有的测试，并且运行非常快：

```py
$ python palindrome.py
...
----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK

```

这就是一个基本的 PDB 会话，并展示了 PDB 的一些核心特性。然而，PDB 还有许多其他命令和特性，学习它们的最佳方法是开始使用 PDB 并尝试这些命令。这个回文程序可以作为学习 PDB 大多数特性的一个很好的例子。

### 总结

+   Python 的标准调试器称为 PDB。

+   PDB 是一个标准的命令行调试器。

+   `pdb.set_trace()`方法可用于停止程序执行并进入调试器。

+   当您处于调试器中时，您的 REPL 提示将更改为（Pdb）。

+   您可以通过输入“help”来访问 PDB 的内置帮助系统。

+   您可以使用`python -m pdb`后跟脚本名称来从头开始在 PDB 下运行程序。

+   PDB 的`where`命令显示当前的调用堆栈。

+   PDB 的`next`命令让执行继续到下一行代码。

+   PDB 的`continue`命令让程序执行无限期地继续，或者直到您使用 control-c 停止它。

+   PDB 的`list`命令显示您当前位置的源代码。

+   PDB 的`return`命令恢复执行，直到当前函数的末尾。

+   PDB 的`print`命令让您在调试器中查看对象的值。

+   使用`quit`退出 PDB。

在这个过程中，我们发现：

+   `divmod()`可以一次计算除法运算的商和余数。

+   `reversed()`函数可以反转一个序列。

+   您可以通过在 Python 命令后传递`-m`来使其作为脚本运行一个模块。

+   调试使得清楚 Python 在运行时评估一切。
