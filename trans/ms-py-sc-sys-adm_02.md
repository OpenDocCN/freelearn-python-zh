# 调试和分析 Python 脚本

调试和分析在 Python 开发中扮演重要角色。调试器帮助程序员分析完整的代码。调试器设置断点，而分析器运行我们的代码并提供执行时间的详细信息。分析器将识别程序中的瓶颈。在本章中，我们将学习`pdb` Python 调试器、`cProfile`模块和`timeit`模块来计算 Python 代码的执行时间。

在本章中，您将学习以下内容：

+   Python 调试技术

+   错误处理（异常处理）

+   调试器工具

+   调试基本程序崩溃

+   分析和计时程序

+   使程序运行更快

# 什么是调试？

调试是解决代码中出现的问题并防止软件正常运行的过程。在 Python 中，调试非常容易。Python 调试器设置条件断点，并逐行调试源代码。我们将使用 Python 标准库中的`pdb`模块来调试我们的 Python 脚本。

# Python 调试技术

为了更好地调试 Python 程序，有各种技术可用。我们将看一下 Python 调试的四种技术：

+   `print()`语句：这是了解发生了什么的最简单方法，因此您可以检查已执行了什么。

+   **`logging`**：这类似于`print`语句，但提供更多上下文信息，以便您可以完全理解。

+   `pdb`调试器：这是一种常用的调试技术。使用`pdb`的优势是可以从命令行、在解释器内和在程序内部使用`pdb`。

+   IDE 调试器：IDE 具有集成的调试器。它允许开发人员执行其代码，然后开发人员可以在程序执行时进行检查。

# 错误处理（异常处理）

在本节中，我们将学习 Python 如何处理异常。但首先，什么是异常？异常是程序执行过程中发生的错误。每当发生任何错误时，Python 都会生成一个异常，该异常将使用`try…except`块进行处理。有些异常无法由程序处理，因此会导致错误消息。现在，我们将看一些异常示例。

在您的终端中，启动`python3`交互式控制台，我们将看到一些异常示例：

```py
student@ubuntu:~$ python3 Python 3.5.2 (default, Nov 23 2017, 16:37:01) [GCC 5.4.0 20160609] on linux Type "help", "copyright", "credits" or "license" for more information. >>> >>> 50 / 0 Traceback (most recent call last):
 File "<stdin>", line 1, in <module> ZeroDivisionError: division by zero >>> >>> 6 + abc*5 Traceback (most recent call last):
 File "<stdin>", line 1, in <module> NameError: name 'abc' is not defined >>> >>> 'abc' + 2 Traceback (most recent call last):
 File "<stdin>", line 1, in <module> TypeError: Can't convert 'int' object to str implicitly >>> >>> import abcd Traceback (most recent call last):
 File "<stdin>", line 1, in <module> ImportError: No module named 'abcd' >>> 
```

这些是一些异常示例。现在，我们将看看如何处理这些异常。

每当您的 Python 程序发生错误时，都会引发异常。我们还可以使用`raise`关键字强制引发异常。

现在我们将看到一个处理异常的`try…except`块。在`try`块中，我们将编写可能生成异常的代码。在`except`块中，我们将为该异常编写解决方案。

`try…except`的语法如下：

```py
try:
 statement(s)
except:
 statement(s)
```

`try`块可以有多个 except 语句。我们还可以在`except`关键字后输入异常名称来处理特定异常。处理特定异常的语法如下：

```py
try:
 statement(s)
except exception_name:
 statement(s)
```

我们将创建一个`exception_example.py`脚本来捕获`ZeroDivisionError`。在您的脚本中编写以下代码：

```py
a = 35 b = 57 try:
 c = a + b print("The value of c is: ", c) d = b / 0 print("The value of d is: ", d)except:
 print("Division by zero is not possible")print("Out of try...except block")
```

按以下方式运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 exception_example.py The value of c is:  92 Division by zero is not possible Out of try...except block
```

# 调试工具

Python 支持许多调试工具：

+   `winpdb`

+   `pydev`

+   `pydb`

+   `pdb`

+   `gdb`

+   `pyDebug`

在本节中，我们将学习`pdb` Python 调试器。`pdb`模块是 Python 标准库的一部分，始终可供使用。

# pdb 调试器

`pdb`模块用于调试 Python 程序。Python 程序使用`pdb`交互式源代码调试器来调试程序。`pdb`设置断点并检查堆栈帧，并列出源代码。

现在我们将学习如何使用`pdb`调试器。有三种使用此调试器的方法：

+   在解释器内

+   从命令行

+   在 Python 脚本内

我们将创建一个`pdb_example.py`脚本，并在该脚本中添加以下内容：

```py
class Student:
 def __init__(self, std): self.count = std            def print_std(self):
 for i in range(self.count): print(i) return if __name__ == '__main__':
 Student(5).print_std()
```

使用此脚本作为学习 Python 调试的示例，我们将详细了解如何启动调试器。

# 在解释器中

要从 Python 交互式控制台启动调试器，我们使用`run()`或`runeval()`。

启动您的`python3`交互式控制台。运行以下命令启动控制台：

```py
 $ python3
```

导入我们的`pdb_example`脚本名称和`pdb`模块。现在，我们将使用`run()`，并将一个字符串表达式作为参数传递给`run()`，该表达式将由 Python 解释器自身进行评估：

```py
student@ubuntu:~$ python3 Python 3.5.2 (default, Nov 23 2017, 16:37:01) [GCC 5.4.0 20160609] on linux Type "help", "copyright", "credits" or "license" for more information. >>> >>> import pdb_example >>> import pdb >>> pdb.run('pdb_example.Student(5).print_std()') > <string>(1)<module>() (Pdb)
```

要继续调试，在（`Pdb`）提示后输入`continue`，然后按*Enter*。如果想要了解我们可以在其中使用的选项，那么在（`Pdb`）提示后按两次*Tab*键。

现在，在输入`continue`后，我们将得到以下输出：

```py
student@ubuntu:~$ python3 Python 3.5.2 (default, Nov 23 2017, 16:37:01) [GCC 5.4.0 20160609] on linux Type "help", "copyright", "credits" or "license" for more information. >>> >>> import pdb_example >>> import pdb >>> pdb.run('pdb_example.Student(5).print_std()') > <string>(1)<module>() (Pdb) continue 0 1 2 3 4 >>> 
```

# 从命令行

从命令行运行调试器的最简单和最直接的方法。我们的程序将作为调试器的输入。您可以按以下方式从命令行使用调试器：

```py
$ python3 -m pdb pdb_example.py
```

当您从命令行运行调试器时，将加载源代码，并且它将在找到的第一行上停止执行。输入`continue`以继续调试。以下是输出：

```py
student@ubuntu:~$ python3 -m pdb pdb_example.py > /home/student/pdb_example.py(1)<module>() -> class Student: (Pdb) continue 0 1 2 3 4 The program finished and will be restarted > /home/student/pdb_example.py(1)<module>() -> class Student: (Pdb)
```

# 在 Python 脚本中

前两种技术将在 Python 程序的开头启动调试器。但是这第三种技术最适合长时间运行的进程。要在脚本中启动调试器，请使用`set_trace()`。

现在，按以下方式修改您的`pdb_example.py`文件：

```py
import pdb class Student:
 def __init__(self, std): self.count = std            def print_std(self):
 for i in range(self.count): pdb.set_trace() print(i) returnif __name__ == '__main__':
 Student(5).print_std()
```

现在，按以下方式运行程序：

```py
student@ubuntu:~$ python3 pdb_example.py > /home/student/pdb_example.py(10)print_std() -> print(i) (Pdb) continue 0 > /home/student/pdb_example.py(9)print_std() -> pdb.set_trace() (Pdb)
```

`set_trace()`是一个 Python 函数，因此您可以在程序的任何地方调用它。

因此，这些是您可以启动调试器的三种方式。

# 调试基本程序崩溃

在本节中，我们将看到跟踪模块。跟踪模块有助于跟踪程序的执行。因此，每当您的 Python 程序崩溃时，我们都可以了解它崩溃的位置。我们可以通过将其导入到脚本中以及从命令行中使用跟踪模块。

现在，我们将创建一个名为`trace_example.py`的脚本，并在脚本中写入以下内容：

```py
class Student:
 def __init__(self, std): self.count = std            def go(self):
 for i in range(self.count): print(i) return if __name__ == '__main__':
 Student(5).go()
```

输出将如下所示：

```py
student@ubuntu:~$ python3 -m trace --trace trace_example.py
 --- modulename: trace_example, funcname: <module> trace_example.py(1): class Student:
 --- modulename: trace_example, funcname: Student trace_example.py(1): class Student: trace_example.py(2):   def __init__(self, std): trace_example.py(5):   def go(self): trace_example.py(10): if __name__ == '__main__': trace_example.py(11):             Student(5).go()
 --- modulename: trace_example, funcname: init trace_example.py(3):               self.count = std
 --- modulename: trace_example, funcname: go trace_example.py(6):               for i in range(self.count): trace_example.py(7):                           print(i) 0 trace_example.py(6):               for i in range(self.count): trace_example.py(7):                           print(i) 1 trace_example.py(6):               for i in range(self.count): trace_example.py(7):                           print(i) 2 trace_example.py(6):               for i in range(self.count): trace_example.py(7):                           print(i) 3 trace_example.py(6):               for i in range(self.count): trace_example.py(7):                           print(i) 4
```

因此，通过在命令行中使用`trace --trace`，开发人员可以逐行跟踪程序。因此，每当程序崩溃时，开发人员都将知道它崩溃的位置。

# 对程序进行分析和计时

对 Python 程序进行分析意味着测量程序的执行时间。它测量了每个函数中花费的时间。Python 的`cProfile`模块用于对 Python 程序进行分析。

# cProfile 模块

如前所述，分析意味着测量程序的执行时间。我们将使用`cProfile` Python 模块对程序进行分析。

现在，我们将编写一个`cprof_example.py`脚本，并在其中写入以下代码：

```py
mul_value = 0 def mul_numbers( num1, num2 ):
 mul_value = num1 * num2; print ("Local Value: ", mul_value) return mul_value mul_numbers( 58, 77 ) print ("Global Value: ", mul_value)
```

运行程序，您将看到以下输出：

```py
student@ubuntu:~$ python3 -m cProfile cprof_example.py Local Value:  4466 Global Value:  0
 6 function calls in 0.000 seconds Ordered by: standard name   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 1    0.000    0.000    0.000    0.000 cprof_example.py:1(<module>) 1    0.000    0.000    0.000    0.000 cprof_example.py:2(mul_numbers) 1    0.000    0.000    0.000    0.000 {built-in method builtins.exec} 2    0.000    0.000    0.000    0.000 {built-in method builtins.print} 1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

因此，使用`cProfile`，所有调用的函数都将打印出每个函数所花费的时间。现在，我们将看看这些列标题的含义：

+   `ncalls`：调用次数

+   **`tottime`**: 在给定函数中花费的总时间

+   `percall`：`tottime`除以`ncalls`的商

+   `cumtime`：在此及所有`子函数`中花费的累计时间

+   `percall`：`cumtime`除以原始调用的商

+   `filename:lineno(function)`: 提供每个函数的相应数据

# timeit

`timeit`是一个用于计时 Python 脚本的 Python 模块。您可以从命令行调用`timeit`，也可以将`timeit`模块导入到您的脚本中。我们将编写一个脚本来计时一段代码。创建一个`timeit_example.py`脚本，并将以下内容写入其中：

```py
import timeit prg_setup = "from math import sqrt" prg_code = ''' def timeit_example():
 list1 = [] for x in range(50): list1.append(sqrt(x)) ''' # timeit statement print(timeit.timeit(setup = prg_setup, stmt = prg_code, number = 10000)) 
```

使用`timeit`，我们可以决定要测量性能的代码片段。因此，我们可以轻松地分别定义设置代码以及要执行测试的代码片段。主要代码运行 100 万次，这是默认时间，而设置代码只运行一次。

# 使程序运行更快

有各种方法可以使您的 Python 程序运行更快，例如以下方法：

+   对代码进行分析，以便识别瓶颈

+   使用内置函数和库，这样解释器就不需要执行循环。

+   避免使用全局变量，因为 Python 在访问全局变量时非常慢

+   使用现有包

# 总结

在本章中，我们学习了调试和分析程序的重要性。我们了解了调试的不同技术。我们学习了`pdb` Python 调试器以及如何处理异常。我们学习了如何在分析和计时我们的脚本时使用 Python 的`cProfile`和`timeit`模块。我们还学习了如何使您的脚本运行更快。

在下一章中，我们将学习 Python 中的单元测试。我们将学习如何创建和使用单元测试。

# 问题

1.  调试程序时，使用哪个模块？

1.  查看如何使用`ipython`以及所有别名和魔术函数。

1.  什么是**全局解释器锁**（**GIL**）？

1.  `PYTHONSTARTUP`，`PYTHONCASEOK`，`PYTHONHOME`和`PYTHONSTARTUP`环境变量的目的是什么？

1.  以下代码的输出是什么？ a) `[0]`，b) `[1]`，c) `[1, 0]`，d) `[0, 1]`。

```py
def foo(k):
    k = [1]
q = [0]
foo(q)
print(q)
```

1.  以下哪个是无效的变量？

a) `my_string_1`

b) `1st_string`

c) `foo`

d) `_`

# 进一步阅读

+   如何解决 Python 中的 GIL 问题：[`realpython.com/python-gil/`](https://realpython.com/python-gil/)

+   查看如何在命令行中使用`pdb`模块：[`fedoramagazine.org/getting-started-python-debugger/`](https://fedoramagazine.org/getting-started-python-debugger/)
