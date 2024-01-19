# 单元测试-单元测试框架简介

测试您的项目是软件开发的重要部分。在本章中，我们将学习 Python 中的单元测试。Python 有一个名为`unittest`的模块，这是一个单元测试框架。我们将在本章学习`unittest`框架。

在本章中，您将学习以下主题：

+   单元测试框架简介

+   创建单元测试任务

# 什么是 unittest？

`unittest`是 Python 中的一个单元测试框架。它支持多个任务，如测试固件、编写测试用例、将测试用例聚合到测试套件中以及运行测试。

`unittest`支持四个主要概念，列在这里：

+   `test fixture`: 这包括为执行一个或多个测试准备和清理活动

+   `test case`: 这包括您的单个单元测试。通过使用`unittest`的`TestCase`基类，我们可以创建新的测试用例

+   `test suite`: 这包括一组测试用例、测试套件或两者。这是为了一起执行测试用例

+   `test runner`: 这包括安排测试执行并向用户提供输出

Python 有一个`unittest`模块，我们将在脚本中导入它。`unittest`模块有`TestCase`类用于创建测试用例。

可以将单独的测试用例创建为方法。这些方法名称以单词*test*开头。因此，测试运行器将知道哪些方法代表测试用例。

# 创建单元测试

在本节中，我们将创建单元测试。为此，我们将创建两个脚本。一个将是您的常规脚本，另一个将包含用于测试的代码。

首先，创建一个名为`arithmetic.py`的脚本，并在其中编写以下代码：

```py
# In this script, we are going to create a 4 functions: add_numbers, sub_numbers, mul_numbers, div_numbers. def add_numbers(x, y):
 return x + y def sub_numbers(x, y):
 return x - y def mul_numbers(x, y):
 return x * y def div_numbers(x, y):
 return (x / y)
```

在上面的脚本中，我们创建了四个函数：`add_numbers`、`sub_numbers`、`mul_numbers`和`div_numbers`。现在，我们将为这些函数编写测试用例。首先，我们将学习如何为`add_numbers`函数编写测试用例。创建一个名为`test_addition.py`的脚本，并在其中编写以下代码：

```py
import arithmetic import unittest # Testing add_numbers function from arithmetic. class Test_addition(unittest.TestCase): # Testing Integers def test_add_numbers_int(self): sum = arithmetic.add_numbers(50, 50) self.assertEqual(sum, 100) # Testing Floats def test_add_numbers_float(self): sum = arithmetic.add_numbers(50.55, 78) self.assertEqual(sum, 128.55) # Testing Strings def test_add_numbers_strings(self): sum = arithmetic.add_numbers('hello','python') self.assertEqual(sum, 'hellopython')  if __name__ == '__main__': unittest.main()
```

在上面的脚本中，我们为`add_numbers`函数编写了三个测试用例。第一个是测试整数，第二个是测试浮点数，第三个是测试字符串。在字符串中，添加意味着连接两个字符串。类似地，您可以为减法、乘法和除法编写测试用例。

现在，我们将运行我们的`test_addition.py`测试脚本，并看看运行此脚本后我们得到什么结果。

按照以下方式运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 test_addition.py ... ---------------------------------------------------------------------- Ran 3 tests in 0.000s
OK
```

在这里，我们得到了`OK`，这意味着我们的测试成功了。

每当运行测试脚本时，您有三种可能的测试结果：

| **结果** | **描述** |
| --- | --- |
| `OK` | 成功 |
| `FAIL` | 测试失败-引发`AssertionError`异常 |
| `ERROR` | 引发除`AssertionError`之外的异常 |

# 单元测试中使用的方法

每当我们使用`unittest`时，我们在脚本中使用一些方法。这些方法如下：

+   `assertEqual()`和`assertNotEqual()`: 这检查预期结果

+   `assertTrue()`和`assertFalse()`: 这验证条件

+   `assertRaises()`: 这验证特定异常是否被引发

+   `setUp()`和`tearDown()`: 这定义了在每个测试方法之前和之后执行的指令

您也可以从命令行使用`unittest`模块。因此，您可以按照以下方式运行先前的测试脚本：

```py
student@ubuntu:~$ python3 -m unittest test_addition.py ... ---------------------------------------------------------------------- Ran 3 tests in 0.000s
OK
```

现在，我们将看另一个例子。我们将创建两个脚本：`if_example.py`和`test_if.py`。`if_example.py`将是我们的常规脚本，`test_if.py`将包含测试用例。在此测试中，我们正在检查输入的数字是否等于`100`。如果等于`100`，则我们的测试将是`成功`的。如果不是，它必须显示一个`FAILED`结果。

创建一个名为`if_example.py`的脚本，并在其中编写以下代码：

```py
def check_if():
 a = int(input("Enter a number \n")) if (a == 100): print("a is equal to 100") else: print("a is not equal to 100") return a
```

现在，创建一个名为`test_if.py`的测试脚本，并在其中编写以下代码：

```py
import if_example import unittest  class Test_if(unittest.TestCase): def test_if(self): result = if_example.check_if() self.assertEqual(result, 100) if __name__ == '__main__':
 unittest.main()
```

按以下方式运行测试脚本：

```py
student@ubuntu:~/Desktop$ python3 -m unittest test_if.py Enter a number 100 a is equal to 100 . ---------------------------------------------------------------------- Ran 1 test in 1.912s OK 
```

我们运行脚本以获得成功的测试结果。现在，我们将输入除`100`之外的一些值，我们必须得到一个`FAILED`的结果。按以下方式运行脚本：

```py
student@ubuntu:~/Desktop$ python3 -m unittest test_if.py Enter a number 50 a is not equal to 100 F ====================================================================== FAIL: test_if (test_if.Test_if) ---------------------------------------------------------------------- Traceback (most recent call last):
 File "/home/student/Desktop/test_if.py", line 7, in test_if self.assertEqual(result, 100) AssertionError: 50 != 100
---------------------------------------------------------------------- Ran 1 test in 1.521s
FAILED (failures=1)
```

# 摘要

在本章中，我们学习了 Python 的单元测试框架`unittest`。我们还学习了如何创建测试用例以及单元测试中使用的方法。

在下一章中，我们将学习如何自动化系统管理员的常规管理活动。您将学习如何接受输入，处理密码，执行外部命令，读取配置文件，向脚本添加警告代码，设置 CPU 限制，启动 web 浏览器，使用`os`模块并进行备份。

# 问题

1.  什么是单元测试、自动化测试和手动测试？

1.  除了`unittest`之外还有哪些替代模块？

1.  编写测试用例有什么用？

1.  什么是 PEP8 标准？

# 进一步阅读

+   单元测试文档：[`docs.python.org/3/library/unittest.html `](https://docs.python.org/3/library/unittest.html)

+   Python 中的 PEP8 编码标准：[`www.python.org/dev/peps/pep-0008/ `](https://www.python.org/dev/peps/pep-0008/)
