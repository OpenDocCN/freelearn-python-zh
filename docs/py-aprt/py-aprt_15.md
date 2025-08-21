## 第十一章：使用 Python 标准库进行单元测试

当我们构建甚至是轻微复杂的程序时，代码中会有无数种缺陷的方式。这可能发生在我们最初编写代码时，但当我们对其进行修改时，我们同样有可能引入缺陷。为了帮助掌握缺陷并保持代码质量高，拥有一组可以运行的测试通常非常有用，这些测试可以告诉您代码是否按照您的期望行事。

为了帮助进行这样的测试，Python 标准库包括[`unittest`模块](https://docs.python.org/3/library/unittest.html)。尽管其名称暗示了它只有单元测试，但实际上，这个模块不仅仅用于单元测试。事实上，它是一个灵活的框架，可以自动化各种测试，从验收测试到集成测试再到单元测试。它的关键特性，就像许多语言中的许多测试框架一样，是帮助您进行*自动化*和*可重复*的测试。有了这样的测试，您可以在任何时候廉价且轻松地验证代码的行为。

### 测试用例

`unittest`模块围绕着一些关键概念构建，其中心是*测试用例*的概念。测试用例 - 体现在[`unittest.TestCase`类](https://docs.python.org/3/library/unittest.html#unittest.TestCase)中 - 将一组相关的测试方法组合在一起，它是`unittest`框架中的测试组织的基本单元。正如我们稍后将看到的，单个测试方法是作为`unittest.TestCase`子类上的方法实现的。

### 固定装置

下一个重要概念是*固定装置*。固定装置是在每个测试方法之前和/或之后运行的代码片段。固定装置有两个主要目的：

1.  *设置*固定装置确保测试环境在运行测试之前处于预期状态。

1.  *清理*固定装置在测试运行后清理环境，通常是通过释放资源。

例如，设置固定装置可能在运行测试之前在数据库中创建特定条目。类似地，拆卸固定装置可能会删除测试创建的数据库条目。测试不需要固定装置，但它们非常常见，通常对于使测试可重复至关重要。

### 断言

最终的关键概念是*断言*。断言是测试方法中的特定检查，最终决定测试是否通过或失败。除其他事项外，断言可以：

+   进行简单的布尔检查

+   执行对象相等性测试

+   验证是否抛出了适当的异常

如果断言失败，那么测试方法也会失败，因此断言代表了您可以执行的最低级别的测试。您可以在`unittest`文档中找到[断言的完整列表](https://docs.python.org/3/library/unittest.html#assert-methods)。

### 单元测试示例：文本分析

有了这些概念，让我们看看如何实际在实践中使用`unittest`模块。在这个例子中，我们将使用*测试驱动开发*^(29)来编写一个简单的文本分析函数。这个函数将以文件名作为唯一参数。然后它将读取该文件并计算：

+   文件中的行数

+   文件中的字符数

TDD 是一个迭代的开发过程，因此我们不会在 REPL 上工作，而是将我们的测试代码放在一个名为`text_analyzer.py`的文件中。首先，我们将创建我们的第一个测试^(30)，并提供足够的支持代码来实际运行它。

```py
# text_analyzer.py

import unittest

class TextAnalysisTests(unittest.TestCase):
    """Tests for the ``analyze_text()`` function."""

    def test_function_runs(self):
        """Basic smoke test: does the function run."""
        analyze_text()

if __name__ == '__main__':
    unittest.main()

```

我们首先导入`unittest`模块。然后，我们通过定义一个从`unittest.TestCase`派生的类`TextAnalysisTests`来创建我们的测试用例。这是您使用`unittest`框架创建测试用例的方法。

要在测试用例中定义单独的测试方法，只需在`TestCase`子类上创建以“`test_`”开头的方法。`unittest`框架在执行时会自动发现这样的方法，因此您不需要显式注册您的测试方法。

在这种情况下，我们定义了最简单的测试：我们检查`analyze_text()`函数是否运行！我们的测试没有进行任何明确的检查，而是依赖于测试方法如果抛出任何异常则会失败的事实。在这种情况下，如果`analyze_text()`没有被定义，我们的测试将失败。

最后，我们定义了惯用的“main”块，当这个模块被执行时调用`unittest.main()`。`unittest.main()`将在模块中搜索所有的`TestCase`子类，并执行它们所有的测试方法。

#### 运行初始测试

由于我们正在使用测试驱动设计，我们期望我们的测试一开始会失败。事实上，我们的测试失败了，原因很简单，我们还没有定义`analyze_text()`：

```py
$ python text_analyzer.py
E
======================================================================
ERROR: test_function_runs (__main__.TextAnalysisTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "text_analyzer.py", line 5, in test_function_runs
    analyze_text()
NameError: global name 'analyze_text' is not defined

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (errors=1)

```

正如你所看到的，`unittest.main()`生成了一个简单的报告，告诉我们运行了多少个测试，有多少个失败了。它还向我们展示了测试是如何失败的，比如在我们尝试运行不存在的函数`analyze_text()`时，它告诉我们我们得到了一个`NameError`。

#### 使测试通过

通过定义`analyze_text()`来修复我们失败的测试。请记住，在测试驱动开发中，我们只编写足够满足测试的代码，所以现在我们只是创建一个空函数。为了简单起见，我们将把这个函数放在`text_analyzer.py`中，尽管通常你的测试代码和实现代码会在不同的模块中：

```py
# text_analyzer.py

def analyze_text():
    """Calculate the number of lines and characters in a file.
 """
    pass

```

将这个函数放在模块范围。再次运行测试，我们发现它们现在通过了：

```py
% python text_analyzer.py
.
----------------------------------------------------------------------
Ran 1 test in 0.001s

OK

```

我们已经完成了一个 TDD 周期，但当然我们的代码还没有真正做任何事情。我们将迭代地改进我们的测试和实现，以得到一个真正的解决方案。

### 使用固定装置创建临时文件

接下来要做的事情是能够向`analyze_text()`传递一个文件名，以便它知道要处理什么。当然，为了让`analyze_text()`工作，这个文件名应该指的是一个实际存在的文件！为了确保我们的测试中存在一个文件，我们将定义一些固定装置。

我们可以定义的第一个固定装置是`TestCase.setUp()`方法。如果定义了，这个方法会在`TestCase`中的每个测试方法之前运行。在这种情况下，我们将使用`setUp()`为我们创建一个文件，并将文件名记住为`TestCase`的成员：

```py
# text_analyzer.py

class TextAnalysisTests(unittest.TestCase):
    . . .
    def setUp(self):
        "Fixture that creates a file for the text methods to use."
        self.filename = 'text_analysis_test_file.txt'
        with open(self.filename, 'w') as f:
            f.write('Now we are engaged in a great civil war,\n'
                    'testing whether that nation,\n'
                    'or any nation so conceived and so dedicated,\n'
                    'can long endure.')

```

我们可以使用的第二个固定装置是`TestCase.tearDown()`。`tearDown()`方法在`TestCase`中的每个测试方法之后运行，在这种情况下，我们将使用它来删除在`setUp()`中创建的文件：

```py
# text_analyzer.py

import os
. . .
class TextAnalysisTests(unittest.TestCase):
    . . .
    def tearDown(self):
        "Fixture that deletes the files used by the test methods."
        try:
            os.remove(self.filename)
        except OSError:
            pass

```

请注意，由于我们在`tearDown()`中使用了`os`模块，我们需要在文件顶部导入它。

还要注意`tearDown()`如何吞没了`os.remove()`抛出的任何异常。我们这样做是因为`tearDown()`实际上不能确定文件是否存在，所以它尝试删除文件，并假设任何异常都可以安全地被忽略。

### 使用新的固定装置

有了我们的两个固定装置，我们现在每个测试方法之前都有一个文件被创建，并且在每个测试方法之后都被删除。这意味着每个测试方法都是从一个稳定的、已知的状态开始的。这对于制作可重复的测试是至关重要的。让我们通过修改现有的测试将这个文件名传递给`analyze_text()`：

```py
# text_analyzer.py

class TextAnalysisTests(unittest.TestCase):
    . . .
    def test_function_runs(self):
        "Basic smoke test: does the function run."
        analyze_text(self.filename)

```

记住我们的`setUp()`将文件名存储在`self.filename`上。由于传递给固定装置的`self`参数与传递给测试方法的实例相同，我们的测试可以使用该属性访问文件名。

当我们运行我们的测试时，我们发现这个测试失败了，因为`analyze_text()`还没有接受任何参数：

```py
% python text_analyzer.py
E
======================================================================
ERROR: test_function_runs (__main__.TextAnalysisTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "text_analyzer.py", line 25, in test_function_runs
    analyze_text(self.filename)
TypeError: analyze_text() takes no arguments (1 given)

----------------------------------------------------------------------
Ran 1 test in 0.003s

FAILED (errors=1)

```

我们可以通过向`analyze_text()`添加一个参数来修复这个问题：

```py
# text_analyzer.py

def analyze_text(filename):
    pass

```

如果我们再次运行我们的测试，我们会再次通过：

```py
% python text_analyzer.py
.
----------------------------------------------------------------------
Ran 1 test in 0.003s

OK

```

我们仍然没有一个做任何有用事情的实现，但你可以开始看到测试如何驱动实现。

### 使用断言来测试行为

现在我们满意`analyze_text()`存在并接受正确数量的参数，让我们看看是否可以让它做真正的工作。我们首先想要的是函数返回文件中的行数，所以让我们定义那个测试：

```py
# text_analyzer.py

class TextAnalysisTests(unittest.TestCase):
    . . .
    def test_line_count(self):
        "Check that the line count is correct."
        self.assertEqual(analyze_text(self.filename), 4)

```

这里我们看到了我们的第一个断言示例。`TestCase`类有[许多断言方法](https://docs.python.org/3/library/unittest.html#assert-methods)，在这种情况下，我们使用`assertEqual()`来检查我们的函数计算的行数是否等于四。如果`analyze_text()`返回的值不等于四，这个断言将导致测试方法失败。如果我们运行我们的新测试，我们会看到这正是发生的：

```py
% python text_analyzer.py
.F
======================================================================
FAIL: test_line_count (__main__.TextAnalysisTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "text_analyzer.py", line 28, in test_line_count
    self.assertEqual(analyze_text(self.filename), 4)
AssertionError: None != 4

----------------------------------------------------------------------
Ran 2 tests in 0.003s

FAILED (failures=1)

```

在这里我们看到我们现在运行了两个测试，其中一个通过了，而新的一个失败了，出现了`AssertionError`。

#### 计算行数

现在让我们暂时违反 TDD 规则，加快一点速度。首先我们将更新函数以返回文件中的行数：

```py
# text_analyzer.py

def analyze_text(filename):
    """Calculate the number of lines and characters in a file.

 Args:
 filename: The name of the file to analyze.

 Raises:
 IOError: If ``filename`` does not exist or can't be read.

 Returns: The number of lines in the file.
 """
    with open(filename, 'r') as f:
        return sum(1 for _ in f)

```

这个改变确实给了我们想要的结果^(33)：

```py
% python text_analyzer.py
..
----------------------------------------------------------------------
Ran 2 tests in 0.003s

OK

```

#### 计算字符

所以让我们添加一个我们想要的另一个功能的测试，即计算文件中字符的数量。由于`analyze_text()`现在应该返回两个值，我们将它返回一个元组，第一个位置是行数，第二个位置是字符数。我们的新测试看起来像这样：

```py
# text_analyzer.py

class TextAnalysisTests(unittest.TestCase):
    . . .
    def test_character_count(self):
        "Check that the character count is correct."
        self.assertEqual(analyze_text(self.filename)[1], 131)

```

并且如预期的那样失败了：

```py
% python text_analyzer.py
E..
======================================================================
ERROR: test_character_count (__main__.TextAnalysisTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "text_analyzer.py", line 32, in test_character_count
    self.assertEqual(analyze_text(self.filename)[1], 131)
TypeError: 'int' object has no attribute '__getitem__'

----------------------------------------------------------------------
Ran 3 tests in 0.004s

FAILED (errors=1)

```

这个结果告诉我们它无法索引`analyze_text()`返回的整数。所以让我们修复`analyze_text()`以返回正确的元组：

```py
# text_analyzer.py

def analyze_text(filename):
    """Calculate the number of lines and characters in a file.

 Args:
 filename: The name of the file to analyze.

 Raises:
 IOError: If ``filename`` does not exist or can't be read.

 Returns: A tuple where the first element is the number of lines in
 the files and the second element is the number of characters.

 """
    lines = 0
    chars = 0
    with open(filename, 'r') as f:
        for line in f:
            lines += 1
            chars += len(line)
    return (lines, chars)

```

这修复了我们的新测试，但我们发现我们破坏了旧的测试：

```py
% python text_analyzer.py
..F
======================================================================
FAIL: test_line_count (__main__.TextAnalysisTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "text_analyzer.py", line 34, in test_line_count
    self.assertEqual(analyze_text(self.filename), 4)
AssertionError: (4, 131) != 4

----------------------------------------------------------------------
Ran 3 tests in 0.004s

FAILED (failures=1)

```

幸运的是，这很容易修复，因为我们只需要在早期的测试中考虑新的返回类型：

```py
# text_analyzer.py

class TextAnalysisTests(unittest.TestCase):
    . . .
    def test_line_count(self):
        "Check that the line count is correct."
        self.assertEqual(analyze_text(self.filename)[0], 4)

```

现在一切又通过了：

```py
% python text_analyzer.py
...
----------------------------------------------------------------------
Ran 3 tests in 0.004s

OK

```

### 测试异常

我们还想测试的另一件事是，当`analyze_text()`传递一个不存在的文件名时，它会引发正确的异常，我们可以这样测试：

```py
# text_analyzer.py

class TextAnalysisTests(unittest.TestCase):
    . . .
    def test_no_such_file(self):
        "Check the proper exception is thrown for a missing file."
        with self.assertRaises(IOError):
            analyze_text('foobar')

```

在这里，我们使用了`TestCase.assertRaises()`断言。这个断言检查指定的异常类型——在这种情况下是`IOError`——是否从 with 块的主体中抛出。

由于`open()`对于不存在的文件引发`IOError`，我们的测试已经通过，无需进一步实现：

```py
% python text_analyzer.py
....
----------------------------------------------------------------------
Ran 4 tests in 0.004s

OK

```

### 测试文件是否存在

最后，我们可以通过编写一个测试来验证`analyze_text()`不会删除文件——这是对函数的合理要求！：

```py
# text_analyzer.py

class TextAnalysisTests(unittest.TestCase):
    . . .
    def test_no_deletion(self):
        "Check that the function doesn't delete the input file."
        analyze_text(self.filename)
        self.assertTrue(os.path.exists(self.filename))

```

`TestCase.assertTrue()` 检查传递给它的值是否评估为`True`。还有一个等效的`assertFalse()`，它对 false 值进行相同的测试。

正如你可能期望的那样，这个测试已经通过了：

```py
% python text_analyzer.py
.....
----------------------------------------------------------------------
Ran 5 tests in 0.002s

OK

```

所以现在我们有了一个有用的、通过的测试集！这个例子很小，但它演示了`unittest`模块的许多重要部分。`unittest`模块还有[更多的部分](https://docs.python.org/3/library/unittest.html)，但是你可以通过我们在这里看到的技术走得很远。

* * *

### 禅宗时刻

![](img/zen-in-the-face-of-ambiguity-refuse-the-temptation-to-guess.png)

猜测的诱惑，或者用一厢情愿的想法忽略模棱两可，可能会带来短期收益。但它往往会导致未来的混乱，以及难以理解和修复的错误。在进行下一个快速修复之前，问问自己需要什么信息才能正确地进行操作。

* * *

### 总结

+   `unittest`模块是一个开发可靠自动化测试的框架。

+   通过从`unittest.TestCase`继承来定义*测试用例*。

+   `unittest.main()`函数对于运行模块中的所有测试非常有用。

+   `setUp()`和`tearDown()`装置用于在每个测试方法之前和之后运行代码。

+   测试方法是通过在测试用例对象上创建以`test_`开头的方法名称来定义的。

+   各种`TestCase.assert...`方法可用于在不满足正确条件时使测试方法失败。

+   使用`TestCase.assertRaises()`在 with 语句中检查测试中是否抛出了正确的异常。
