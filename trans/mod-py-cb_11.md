# 第11章。测试

在本章中，我们将看以下配方：

+   使用文档字符串进行测试

+   测试引发异常的函数

+   处理常见的doctest问题

+   创建单独的测试模块和包

+   结合unittest和doctest测试

+   测试涉及日期或时间的事物

+   测试涉及随机性的事物

+   模拟外部资源

# 介绍

测试是创建可工作软件的核心。这是关于测试重要性的经典陈述：

> *任何没有自动化测试的程序功能都不存在。*

这是肯特·贝克的书《极限编程解释：拥抱变化》中的内容。

我们可以区分几种测试：

+   **单元测试**：这适用于独立的软件*单元*：函数、类或模块。该单元被孤立测试以确认它是否正确工作。

+   **集成测试**：这将单元组合以确保它们正确集成。

+   **系统测试**：这测试整个应用程序或一组相互关联的应用程序，以确保软件组件的集合正常工作。这经常用于整体接受软件的使用。

+   **性能测试**：这确保一个单元满足性能目标。在某些情况下，性能测试包括对内存、线程或文件描述符等资源的研究。目标是确保软件适当地利用系统资源。

Python有两个内置的测试框架。其中一个检查文档字符串中包含`>>>`提示的示例。这就是`doctest`工具。虽然这被广泛用于单元测试，但也可以用于简单的集成测试。

另一个测试框架使用了从`unittest`模块定义的类构建的定义。这个模块定义了一个`TestCase`类。这也主要用于单元测试，但也可以应用于集成和性能测试。

当然，我们希望结合这些工具。这两个模块都有特性允许共存。我们经常利用`unittest`包的测试加载协议来合并所有测试。

此外，我们可能会使用工具`nose2`或`py.test`来进一步自动化测试发现，并添加额外的功能，如测试用例覆盖率。这些项目通常对特别复杂的应用程序很有帮助。

有时使用GIVEN-WHEN-THEN测试用例命名风格来总结一个测试是有帮助的：

+   **GIVEN**一些初始状态或上下文

+   **WHEN**请求行为

+   **THEN**被测试的组件有一些预期的结果或状态变化

# 使用文档字符串进行测试

良好的Python包括每个模块、类、函数和方法内部的文档字符串。许多工具可以从文档字符串创建有用的、信息丰富的文档。

文档字符串的一个重要元素是示例。示例成为一种单元测试用例。一个示例通常符合GIVEN-WHEN-THEN测试模型，因为它显示了一个单元、一个请求和一个响应。

我们如何将示例转化为适当的测试用例？

## 准备就绪

我们将看一个简单的函数定义以及一个简单的类定义。每个都将包括包含示例的文档字符串，这些示例可以用作正式测试。

这是一个计算两个数字的二项式系数的简单函数。它显示了*n*个事物以*k*个大小的组合的数量。例如，一副52张的牌可以被分成5张牌的方式可以这样计算：

![准备就绪](Image00054.jpg)

这定义了一个小的Python函数，我们可以这样写：

```py
    from math import factorial 
    def binom(n: int, k: int) -> int: 
        return factorial(n) // (factorial(k) * factorial(n-k)) 

```

这个函数进行了一个简单的计算并返回一个值。由于它没有内部状态，所以相对容易测试。这将是用于展示可用的单元测试工具的示例之一。

我们还将看一个简单的类，它具有均值和中位数的延迟计算。它使用一个内部的`Counter`对象，可以被询问以确定模式：

```py
    from statistics import median 
    from collections import Counter 

    class Summary: 

        def __init__(self): 
            self.counts = Counter() 

        def __str__(self): 
            return "mean = {:.2f}\nmedian = {:d}".format( 
            self.mean, self.median) 

        def add(self, value): 
            self.counts[value] += 1 

        @property 
        def mean(self): 
            s0 = sum(f for v,f in self.counts.items()) 
            s1 = sum(v*f for v,f in self.counts.items()) 
            return s1/s0 

        @property 
        def median(self): 
            return median(self.counts.elements()) 

```

`add()`方法改变了这个对象的状态。由于这种状态改变，我们需要提供更复杂的示例，展示`Summary`类的实例的行为方式。

## 如何做...

我们将在这个示例中展示两种变化。第一种是用于大部分无状态操作，比如计算`binom()`函数。第二种是用于有状态操作，比如`Summary`类。

1.  将示例放入文档字符串中。

1.  将doctest模块作为程序运行。有两种方法：

+   在命令提示符下：

```py
         **$ python3.5 -m doctest code/ch11_r01.py** 

        ```

如果所有示例都通过，就不会有输出。使用`-v`选项会产生总结测试的详细输出。

+   通过包含一个`__name__ == '__main__'`部分。这可以导入doctest模块并执行`testmod()`函数：

```py
                        if __name__ == '__main__': 
                            import doctest 
                            doctest.testmod() 

        ```

如果所有示例都通过，就不会有输出。要查看一些输出，可以使用`testmod()`函数的`verbose=1`参数创建更详细的输出。

### 为无状态函数编写示例

1.  用摘要开始文档字符串：

```py
            '''Computes the binomial coefficient. 
            This shows how many combinations of 
            *n* things taken in groups of size *k*. 

    ```

1.  包括参数定义：

```py
            :param n: size of the universe 
            :param k: size of each subset 

    ```

1.  包括返回值定义：

```py
            :returns: the number of combinations 

    ```

1.  模拟一个在Python的`>>>`提示下使用该函数的示例：

```py
     **>>> binom(52, 5) 
          2598960** 

    ```

1.  用适当的引号关闭长文档字符串：

```py
            ''' 

    ```

### 为有状态对象编写示例

1.  用摘要编写类级别的文档字符串：

```py
            '''Computes summary statistics. 

            ''' 

    ```

我们留下了填写示例的空间。

1.  使用摘要编写方法级别的文档字符串。这是`add()`方法：

```py
            def add(self, value): 
                '''Adds a value to be summarized. 

                :param value: Adds a new value to the collection. 
                ''' 
                self.counts[value] += 1 

    ```

1.  这是`mean()`方法：

```py
            @property 
            def mean(self): 
                '''Computes the mean of the collection. 
                :return: mean value as a float 
                ''' 
                s0 = sum(f for v,f in self.counts.items()) 
                s1 = sum(v*f for v,f in self.counts.items()) 
                return s1/s0 

    ```

`median()`方法和其他写入的方法也需要类似的字符串。

1.  扩展类级别的文档字符串具体示例。在这种情况下，我们将写两个。第一个示例显示`add()`方法没有返回值，但改变了对象的状态。`mean()`方法显示了这个状态：

```py
          **>>> s = Summary() 
          >>> s.add(8) 
          >>> s.add(9) 
          >>> s.add(9) 
          >>> round(s.mean, 2) 
          8.67 
          >>> s.median 
          9** 

    ```

我们将平均值的结果四舍五入，以避免显示一个长的浮点值，在所有平台上可能没有完全相同的文本表示。当我们运行doctest时，通常会得到一个静默的响应，因为测试通过了。

第二个示例显示了`__str__()`方法的多行结果：

```py
 **>>> print(str(s)) 
mean = 8.67 
median = 9** 

```

当某些事情不起作用时会发生什么？想象一下，我们将期望的输出更改为错误答案。当我们运行doctest时，我们将看到如下输出：

```py
 ************************************************************************* 

File "__main__", line ?, in __main__.Summary 
 **Failed example:** 

 **s.median** 

 **Expected:** 

    10 
 **Got:** 

    9 
 ************************************************************************* 

 **1 items had failures:** 

   1 of   6 in __main__.Summary 
 *****Test Failed*** 1 failures.** 

 **TestResults(failed=1, attempted=9)** 

```

这显示了错误的位置。它显示了测试示例的预期值和实际答案。

## 它是如何工作的...

`doctest`模块包括一个主程序，以及几个函数，它将扫描Python文件中的`>>>`示例。我们可以利用模块扫描函数`testmod()`来扫描当前模块。我们可以使用这个来扫描任何导入的模块。

扫描操作寻找具有`>>>`行特征模式的文本块，后面是显示命令响应的行。

doctest解析器从提示行和响应文本块创建一个小的测试用例对象。有三种常见情况：

+   没有预期的响应文本：当我们为`Summary`类的`add()`方法定义测试时，我们看到了这种模式。

+   单行响应文本：这在`binom()`函数和`mean()`方法中得到了体现。

+   多行响应：响应由下一个`>>>`提示或空行限定。这在`Summary`类的`str()`示例中得到了体现。

doctest模块将执行每个带有`>>>`提示的代码行。它将实际结果与期望结果进行比较。比较是非常简单的文本匹配。除非使用特殊注释，否则输出必须精确匹配期望。

这种测试协议的简单性对软件设计提出了一些要求。函数和类必须设计为从`>>>`提示中工作。因为在文档字符串示例中创建非常复杂的对象可能会变得尴尬，所以设计必须保持足够简单，以便可以进行交互演示。保持软件足够简单，以便在`>>>`提示处进行演示通常是有益的。

结果的比较简单性可能会对显示的输出造成一些复杂性。例如，请注意，我们将平均值的值四舍五入到两位小数。这是因为浮点值的显示可能会因平台而异。

Python 3.5.1（在Mac OS X上）显示`8.666666666666666`，而Python 2.6.9（同样在Mac OS X上）显示`8.6666666666666661`。这些值在小数点后16位相等。这大约是48位数据，这是浮点值的实际限制。

我们将在*处理常见的doctest问题*配方中详细讨论精确比较问题。

## 还有更多...

一个重要的测试考虑因素是边界情况。**边界情况**通常关注计算设计的极限。例如，二项式函数有两个边界：

![还有更多...](Image00055.jpg)

我们可以很容易地将这些添加到示例中，以确保我们的实现是正确的；这将导致一个看起来像下面这样的函数：

```py
    def binom(n: int, k: int) -> int: 
        '''Computes the binomial coefficient. 
        This shows how many combinations of 
        *n* things taken in groups of size *k*. 

        :param n: size of the universe 
        :param k: size of each subset 

        :returns: the number of combinations 

        >>> binom(52, 5) 
        2598960 
        >>> binom(52, 0) 
        1 
        >>> binom(52, 52) 
        1 
        ''' 
        return factorial(n) // (factorial(k) * factorial(n-k)) 

```

在某些情况下，我们可能需要测试超出有效值范围的值。这些情况并不适合放入文档字符串，因为它们会使本来应该发生的事情的解释变得混乱。

我们可以在一个名为`__test__`的全局变量中包含额外的文档字符串测试用例。这个变量必须是一个映射。映射的键是测试用例的名称，映射的值是doctest示例。这些示例需要是三引号字符串。

因为这些示例不在文档字符串内，所以在使用内置的`help()`函数时不会显示出来。当使用其他工具从源代码创建文档时，它们也不会显示出来。

我们可能会添加类似这样的内容：

```py
    __test__ = { 
    'GIVEN_binom_WHEN_0_0_THEN_1':  
    ''' 
    >>> binom(0, 0) 
    1 
    ''', 

    } 

```

我们已经用没有缩进的键编写了映射。值已经缩进了四个空格，这样它们就会从键中脱颖而出，并且稍微容易发现。

Doctest程序会找到这些测试用例，并将其包含在整体测试套件中。我们可以用这个来进行重要的测试，但并不真正有助于文档编制。

## 另请参阅

+   在*测试引发异常的函数*和*处理常见的doctest问题*配方中，我们将看到另外两种doctest技术。这是重要的，因为异常通常会包括一个回溯，其中可能包括每次运行程序时都会有所不同的对象ID。

# 测试引发异常的函数

良好的Python在每个模块、类、函数和方法内部都包含文档字符串。许多工具可以从这些文档字符串中创建有用的、信息丰富的文档。

文档字符串的一个重要元素是示例。示例成为一种单元测试用例。Doctest对期望输出与实际输出进行简单的、字面的匹配。

然而，当示例引发异常时，Python的回溯消息并不总是相同的。它可能包括会改变的对象ID值或模块行号，这取决于执行测试的上下文。当涉及异常时，doctest的字面匹配规则并不适用。

我们如何将异常处理和由此产生的回溯消息转化为正确的测试用例？

## 准备就绪

我们将看一个简单的函数定义以及一个简单的类定义。其中每一个都将包括包含示例的文档字符串，这些示例可以用作正式测试。

这是一个简单的函数，用于计算两个数字的二项式系数。它显示了*n*个东西在*k*组中取的组合数。例如，一个52张牌的牌组可以被分成5张牌的手的方式有多少种：

![准备就绪](Image00056.jpg)

这定义了一个小的Python函数，我们可以这样写：

```py
    from math import factorial 
    def binom(n: int, k: int) -> int: 
        ''' 
        Computes the binomial coefficient. 
        This shows how many combinations of 
        *n* things taken in groups of size *k*. 

        :param n: size of the universe 
        :param k: size of each subset 

        :returns: the number of combinations 

        >>> binom(52, 5) 
        2598960 
        ''' 
        return factorial(n) // (factorial(k) * factorial(n-k)) 

```

这个函数进行简单的计算并返回一个值。我们想在`__test__`变量中包含一些额外的测试用例，以展示在给定超出预期范围的值时会发生什么。

## 如何做...

1.  在模块中创建一个全局的`__test__`变量：

```py
            __test__ = { 

            } 

    ```

我们留下了空间来插入一个或多个测试用例。

1.  对于每个测试用例，提供一个名称和一个示例的占位符：

```py
            __test__ = { 
            'GIVEN_binom_WHEN_wrong_relationship_THEN_error':  
            ''' 
                example goes here. 
            ''', 
            } 

    ```

1.  包括一个带有`doctest`指令注释的调用，`IGNORE_EXCEPTION_DETAIL`。这将替换“示例在这里”：

```py
     **>>> binom(5, 52)  # doctest: +IGNORE_EXCEPTION_DETAIL** 

    ```

该指令以`# doctest:`开头。指令通过`+`启用，通过`-`禁用。

1.  包括一个实际的回溯消息。这是*示例在这里*的一部分；它在`>>>`语句之后显示预期的响应：

```py
            Traceback (most recent call last):
              File "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/doctest.py", line 1320, in __run 
                compileflags, 1), test.globs) 
              File "<doctest __main__.__test__.GIVEN_binom_WHEN_wrong_relationship_THEN_error[0]>", line 1, in <module> 
                binom(5, 52) 
              File "/Users/slott/Documents/Writing/Python Cookbook/code/ch11_r01.py", line 24, in binom 
                return factorial(n) // (factorial(k) * factorial(n-k)) 
            ValueError: factorial() not defined for negative values 

    ```

1.  以`File...`开头的三行将被忽略。`ValueError:`行将被检查以确保测试产生了预期的异常。

总体语句看起来像这样：

```py
    __test__ = { 
    'GIVEN_binom_WHEN_wrong_relationship_THEN_error': ''' 
        >>> binom(5, 52)  # doctest: +IGNORE_EXCEPTION_DETAIL 
        Traceback (most recent call last): 
          File "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/doctest.py", line 1320, in __run 
            compileflags, 1), test.globs) 
          File "<doctest __main__.__test__.GIVEN_binom_WHEN_wrong_relationship_THEN_error[0]>", line 1, in <module> 
            binom(5, 52) 
          File "/Users/slott/Documents/Writing/Python Cookbook/code/ch11_r01.py", line 24, in binom 
            return factorial(n) // (factorial(k) * factorial(n-k)) 
        ValueError: factorial() not defined for negative values 
    ''' 
    } 

```

现在我们可以使用这样的命令来测试整个模块的功能：

```py
 **python3.5 -R -m doctest ch11_r01.py** 

```

## 它是如何工作的...

doctest解析器有几个指令，可以用来修改测试行为。这些指令被包含为特殊注释，与执行测试操作的代码行一起。

我们有两种处理包含异常的测试的方法：

+   我们可以使用`# doctest: +IGNORE_EXCEPTION_DETAIL`并提供完整的回溯错误消息。回溯的细节将被忽略，只有最终的异常行与预期值匹配。这使得很容易复制实际错误并将其粘贴到文档中。

+   我们可以使用`# doctest: +ELLIPSIS`并用`...`替换回溯消息的部分。这也允许预期输出省略细节并专注于实际错误的最后一行。

对于这种第二种异常示例，我们可以包括一个像这样的测试用例：

```py
    'GIVEN_binom_WHEN_negative_THEN_exception':  
    ''' 
        >>> binom(52, -5)  # doctest: +ELLIPSIS 
        Traceback (most recent call last): 
        ... 
        ValueError: factorial() not defined for negative values 
    ''', 

```

测试用例使用了`+ELLIPSIS`指令。错误回溯的细节已被替换为`...`。相关材料已被保留完整，以便实际异常消息与预期异常消息精确匹配。

Doctest将忽略第一个`Traceback...`行和最后一个`ValueError:...`行之间的所有内容。通常，最后一行是测试的正确执行所关心的。中间文本取决于测试运行的上下文。

## 还有更多...

还有几个比较指令可以提供给单个测试。

+   `+ELLIPSIS`：这允许预期结果通过用`...`替换细节来概括。

+   `+IGNORE_EXCEPTION_DETAIL`：这允许预期值包括完整的回溯消息。大部分回溯将被忽略，只有最终的异常行会被检查。

+   `+NORMALIZE_WHITESPACE`：在某些情况下，预期值可能会被包裹到多行上以便于阅读。或者，它的间距可能与标准Python值略有不同。使用此标志允许预期值的空格有一定的灵活性。

+   +SKIP：测试被跳过。有时会为设计用于未来版本的测试而这样做。在功能完成之前可能会包括测试。测试可以保留在原位以供未来开发工作使用，但为了按时发布版本而被跳过。

+   `+DONT_ACCEPT_TRUE_FOR_1`：这涵盖了Python 2中常见的一种特殊情况。在`True`和`False`被添加到语言之前，值`1`和`0`被用来代替。与实际结果进行比较的doctest算法将通过匹配`True`和`1`来尊重这种较旧的方案。可以在命令行上使用`-o DONT_ACCEPT_TRUE_FOR_1`提供此指令。然后，这个改变将对所有测试全局有效。

+   `+DONT_ACCEPT_BLANKLINE`：通常，空行会结束一个示例。在示例输出包括空行的情况下，预期结果必须使用特殊语法`<blankline>`。使用这个语法可以显示预期的空行位置，并且示例不会在这个空行结束。在非常罕见的情况下，预期输出实际上会包括字符串`<blankline>`。这个指令确保`<blankline>`不是用来表示空行，而是代表它自己。在为文档测试模块本身编写测试时，这是有意义的。

在评估`testmod()`或`testfile()`函数时，这些也可以作为`optionsflags`参数提供。

## 另请参阅

+   查看*使用文档字符串进行测试*配方，了解文档测试的基础知识

+   查看*处理常见的文档测试问题*配方，了解其他需要文档测试指令的特殊情况

# 处理常见的文档测试问题

良好的Python包括每个模块、类、函数和方法内部的文档字符串。许多工具可以从完整的文档字符串中创建有用的、信息丰富的文档。

文档字符串的一个重要元素是示例。示例成为一种单元测试用例。文档测试对预期输出进行简单、字面的匹配。然而，有一些Python对象在每次引用它们时并不一致。

例如，所有对象哈希值都是随机的。这意味着集合中元素的顺序或字典中键的顺序可能会有所不同。我们有几种选择来创建测试用例示例输出：

+   编写可以容忍随机化的测试。通常通过转换为排序结构。

+   规定`PYTHONHASHSEED`环境变量的值。

+   要求使用`-R`选项运行Python以完全禁用哈希随机化。

除了集合中键或项的位置的简单变化之外，还有一些其他考虑因素。以下是一些其他问题：

+   `id()`和`repr()`函数可能会暴露内部对象ID。对于这些值无法做出任何保证。

+   浮点值可能会因平台而异。

+   当前日期和时间在测试用例中没有实际意义。

+   使用默认种子的随机数很难预测。

+   操作系统资源可能不存在，或者可能不处于适当的状态。

在这个配方中，我们将使用一些文档测试技术来解决前两个问题。我们将在*涉及日期或时间的测试*和*涉及随机性的测试*配方中研究`datetime`和`random`。我们将在*模拟外部资源*配方中研究如何处理外部资源。

文档测试示例需要与文本完全匹配。我们如何编写处理哈希随机化或浮点实现细节的文档测试示例？

## 准备工作

在*使用CSV模块读取分隔文件*配方中，我们看到`csv`模块将读取数据，为每一行输入创建一个映射。在那个配方中，我们看到了一个`CSV`文件，其中记录了一艘帆船日志中的一些实时数据。这是`waypoints.csv`文件。

`DictReader`类生成的行如下所示：

```py
    {'date': '2012-11-27', 
     'lat': '32.8321666666667', 
     'lon': '-79.9338333333333', 
     'time': '09:15:00'} 

```

这是一个文档测试的噩梦，因为哈希随机化确保这个字典中键的顺序很可能是不同的。

当我们尝试编写涉及字典的文档测试示例时，我们经常会遇到这样的问题：

```py
    Failed example: 
        next(row_iter) 
    Expected: 
        {'date': '2012-11-27', 'lat': '32.8321666666667', 
        'lon': '-79.9338333333333', 'time': '09:15:00'} 
    Got: 
        {'lon': '-79.9338333333333', 'time': '09:15:00', 
        'date': '2012-11-27', 'lat': '32.8321666666667'} 

```

预期和实际行中的数据明显匹配。然而，字典值的字符串显示并不完全相同。键的顺序不一致。

我们还将研究一个小型的实值函数，以便我们可以处理浮点值：

![准备工作](Image00057.jpg)

这个函数是标准z分数的累积概率密度函数。对于标准化变量，该变量的Z分数值的平均值将为零，标准差将为一。有关标准化分数概念的更多信息，请参见[第8章](text00088.html#page "第8章.功能和响应式编程特性")中的*创建部分函数*配方，*功能和响应式编程特性*。

这个函数Φ(*n*)告诉我们人口中有多少比例在给定的z分数下。例如，Φ(0) = 0.5：一半的人口的z分数低于零。

这个函数涉及一些相当复杂的处理。单元测试必须反映浮点精度问题。

## 如何操作...

我们将在一个配方中查看映射（和集合）排序。我们将单独查看浮点数。

### 为映射或集合值编写doctest示例

1.  导入必要的库并定义函数：

```py
            import csv 
            def raw_reader(data_file): 
                """ 
                Read from a given, open file. 

                :param data_file: Open file, ready to be processed. 
                :returns: iterator over individual rows as dictionaries. 

                Example: 

                """ 
                data_reader = csv.DictReader(data_file) 
                for row in data_reader: 
                    yield row 

    ```

我们在文档字符串中包含了示例标题。

1.  我们可以用`io`包中的`StringIO`类的实例替换实际数据文件。这可以在示例内部使用，以提供固定的样本数据：

```py
     **>>> from io import StringIO 
          >>> mock_file = StringIO('''lat,lon,date,time 
          ... 32.8321,-79.9338,2012-11-27,09:15:00 
          ... ''') 
          >>> row_iter = iter(raw_reader(mock_file))** 

    ```

1.  从概念上讲，测试用例是这样的。这段代码将无法正常工作，因为键将被打乱。但是，可以很容易地重构它：

```py
     **>>> row = next(row_iter) 
          >>> row 
          {'time': '09:15:00', 'lat': '32.8321', etc. }** 

    ```

我们省略了其余的输出，因为每次运行测试时都会有所不同：

代码必须这样编写，以强制将键按固定顺序排列：

```py
     **>>> sorted(row.items())  # doctest: +NORMALIZE_WHITESPACE 
          [('date', '2012-11-27'), ('lat', '32.8321'), 
          ('lon', '-79.9338'), ('time', '09:15:00')]** 

    ```

排序后的项目是按一致的顺序排列的。

### 为浮点值编写doctest示例

1.  导入必要的库并定义函数：

```py
            from math import * 
            def phi(n): 
                """ 
                The cumulative distribution function for the standard normal 
                distribution. 

                :param n: number of standard deviations 
                :returns: cumulative fraction of values below n. 

                Examples: 
                """ 
                return (1+erf(n/sqrt(2)))/2 

    ```

我们在文档字符串中留下了示例的空间。

1.  对于每个示例，包括显式使用`round()`：

```py
     **>>> round(phi(0), 3) 
          0.399 
          >>> round(phi(-1), 3) 
          0.242 
          >>> round(phi(+1), 3) 
          0.242** 

    ```

浮点值四舍五入，以便浮点实现细节的差异不会导致看似不正确的结果。

## 它是如何工作的...

由于哈希随机化，用于字典的哈希键是不可预测的。这是一个重要的安全特性，可以防止微妙的拒绝服务攻击。有关详细信息，请参见[http://www.ocert.org/advisories/ocert-2011-003.html](http://www.ocert.org/advisories/ocert-2011-003.html)。

我们有两种方法可以处理没有定义顺序的字典键：

+   我们可以编写针对每个键具体的测试用例：

```py
     **>>> row['date'] 
          '2012-11-27' 
          >>> row['lat'] 
          '32.8321' 
          >>> row['lon'] 
          '-79.9338' 
          >>> row['time'] 
          '09:15:00'** 

    ```

+   我们可以将其转换为一个具有固定顺序的数据结构。`row.items()`的值是一个可迭代的键值对序列。顺序不是提前设置的，但我们可以使用以下方法来强制排序：

```py
     **>>> sorted(row.items())** 

    ```

这将返回一个按顺序排列的键列表。这使我们能够创建一个一致的文字值，每次评估测试时都将是相同的。

大多数浮点实现都是相当一致的。然而，对于任何给定的浮点数的最后几位，很少有正式的保证。与其相信所有的53位都有完全正确的值，往往更容易将值四舍五入为与问题域相匹配的值。

对于大多数现代处理器，浮点值通常是32位或64位值。32位值大约有七位小数。将值四舍五入，使值中不超过六位数字通常是最简单的方法。

将数字四舍五入到六位并不意味着使用`round(x, 6)`。`round()`函数不会保留数字的位数。这个函数四舍五入到小数点右边的位数；它不考虑小数点左边的位数。将一个数量级为10^(12)的数字四舍五入到小数点右边的六个位置会得到18位数字，对于32位值来说太多了。将一个数量级为10^(-7)的数字四舍五入到小数点右边的六个位置会得到零。

## 还有更多...

在处理`set`对象时，我们还必须注意项目的顺序。我们通常可以使用`sorted()`将`set`转换为`list`并强加特定的顺序。

Python `dict`对象出现在令人惊讶的许多地方：

+   当我们编写一个使用`**`来收集参数值字典的函数时。没有保证参数的顺序。

+   当我们使用诸如 `vars()` 这样的函数从局部变量或对象的属性创建字典时，字典没有保证的顺序。

+   当我们编写依赖于类定义内省的程序时，方法是在类级别的字典对象中定义的。我们无法预测它们的顺序。

当存在不可靠的测试用例时，这一点变得明显。一个似乎随机通过或失败的测试用例可能是基于哈希随机化的结果。提取键并对其进行排序以克服这个问题。

我们也可以使用这个命令行选项来运行测试：

```py
 **python3.5 -R -m doctest ch11_r03.py** 

```

这将关闭哈希随机化，同时在特定文件 `ch11_r03.py` 上运行 doctest。

## 另请参阅

+   *涉及日期或时间的测试* 配方，特别是 datetime 的 `now()` 方法需要一些小心。

+   *涉及随机性的测试* 配方将展示如何测试涉及 `random` 处理的过程。

# 创建单独的测试模块和包

我们可以在文档字符串示例中进行任何类型的单元测试。然而，有些事情如果用这种方式做会变得极其乏味。

`unittest` 模块允许我们超越简单的示例。这些测试依赖于测试用例类定义。`TestCase` 的子类可以用来编写非常复杂和复杂的测试；这些测试可以比作为 doctest 示例进行的相同测试更简单。

`unittest` 模块还允许我们在文档字符串之外打包测试。这对于特别复杂的边界情况的测试非常有帮助，当放在文档中时并不那么有用。理想情况下，doctest 用例说明了 **happy path –** 最常见的用例。通常使用 `unittest` 来进行不在 happy path 上的测试用例。

我们如何创建更复杂的测试？

## 准备工作

一个测试通常可以用一个三部分的 *Given-When-Then* 故事来总结：

+   **GIVEN**：处于初始状态或上下文中的某个单元

+   **WHEN**：请求一种行为

+   **THEN**：被测试的组件有一些预期的结果或状态变化

`TestCase` 类并不完全遵循这种三部分结构。它有两部分；必须做出一些设计选择，关于测试的三个部分应该分配到哪里：

+   一个实现测试用例的 *Given* 部分的 `setUp()` 方法。它也可以处理 *When* 部分。

+   一个必须处理 *Then* 部分的 `runTest()` 方法。这也可以处理 *When* 部分。 *Then* 条件通过一系列断言来确认。这些通常使用 `TestCase` 类的复杂断言方法。

在哪里实现 *When* 部分的选择与重用的问题有关。在大多数情况下，有许多替代的 *When* 条件，每个条件都有一个独特的 *Then* 来确认正确的操作。*Given* 可能是 `setUp()` 方法的共同部分，并被一些 `TestCase` 子类共享。每个子类都有一个独特的 `runTest()` 方法来实现 *When* 和 *Then* 部分。

在某些情况下，*When* 部分被分成一些常见部分和一些特定于测试用例的部分。在这种情况下，*When* 部分可能在 `setUp()` 方法中部分定义，部分在 `runTest()` 方法中定义。

我们将为一个设计用于计算一些基本描述性统计的类创建一些测试。我们希望提供的样本数据远远大于我们作为 doctest 示例输入的任何内容。我们希望使用成千上万的数据点而不是两三个。

这是我们想要测试的类定义的概要。我们只提供了方法和一些摘要。代码的大部分在*使用文档字符串进行测试*中显示。我们省略了所有的实现细节。这只是类的概要，提醒了方法的名称是什么：

```py
    from statistics import median 
    from collections import Counter 

    class Summary: 
        def __init__(self): 
           pass 

        def __str__(self): 
            '''Returns a multi-line text summary.''' 

        def add(self, value): 
            '''Adds a value to be summarized.''' 

        @property 
        def count(self): 
            '''Number of samples.''' 

        @property 
        def mean(self): 
            '''Mean of the collection.''' 

        @property 
        def median(self): 
            '''Median of the collection.''' 
            return median(self.counts.elements()) 

        @property 
        def mode(self): 
            '''Returns the items in the collection in decreasing 
            order by frequency. 
            ''' 

```

因为我们没有关注实现细节，这是一种黑盒测试。代码是一个黑盒——内部是不透明的。为了强调这一点，我们从前面的代码中省略了实现细节。

我们希望确保当我们使用成千上万的样本时，这个类能够正确执行。我们也希望确保它能够快速工作；我们将把它作为整体性能测试的一部分，以及单元测试。

## 如何做...

1.  我们将测试代码包含在与工作代码相同的模块中。这将遵循将测试和代码捆绑在一起的doctest模式。我们将使用`unittest`模块来创建测试类：

```py
            import unittest 
            import random 

    ```

我们还将使用`random`来打乱输入数据。

1.  创建一个`unittest.TestCase`的子类。为这个类提供一个显示测试意图的名称：

```py
            class GIVEN_Summary_WHEN_1k_samples_THEN_mean(unittest.TestCase): 

    ```

*GIVEN-WHEN-THEN*的名称非常长。我们将依赖`unittest`来发现`TestCase`的所有子类，这样我们就不必多次输入这个类名。

1.  在这个类中定义一个`setUp()`方法，处理测试的*Given*方面。这将为测试处理创建一个上下文：

```py
            def setUp(self): 
                self.summary = Summary() 
                self.data = list(range(1001)) 
                random.shuffle(self.data) 

    ```

我们创建了一个包含`1,001`个样本的集合，值范围从`0`到`1,000`。平均值恰好是500，中位数也是。我们将数据随机排序。

1.  定义一个`runTest()`方法，处理测试的*When*方面。这将执行状态变化：

```py
            def runTest(self): 
                for sample in self.data: 
                    self.summary.add(sample) 

    ```

1.  添加断言来实现测试的*Then*方面。这将确认状态变化是否正常工作：

```py
            self.assertEqual(500, self.summary.mean) 
            self.assertEqual(500, self.summary.median) 

    ```

1.  为了使运行变得非常容易，添加一个主程序部分：

```py
            if __name__ == "__main__": 
                unittest.main() 

    ```

有了这个，测试可以在命令提示符下运行。也可以从命令行运行。

## 它是如何工作的...

我们使用了`unittest`模块的几个部分：

+   `TestCase`类用于定义一个测试用例。这可以有一个`setUp()`方法来创建单元和可能的请求。这必须至少有一个`runTest()`来发出请求并检查响应。

我们可以在一个文件中有多个这样的类定义，以便构建一个适当的测试集。对于简单的类，可能只有几个测试用例。对于复杂的模块，可能有几十甚至几百个用例。

+   `unittest.main()`函数做了几件事：

+   它创建一个空的`TestSuite`，其中包含所有的`TestCase`对象。

+   它使用默认加载器来检查一个模块并找到所有的`TestCase`实例。这些被加载到`TestSuite`中。这个过程是我们可能想要修改或扩展的。

+   然后运行`TestSuite`并显示结果的摘要。

当我们运行这个模块时，我们会看到以下输出：

```py
 **.---------------------------------------------------------------------- 
Ran 1 test in 0.005s 

OK** 

```

每次通过一个测试，都会显示一个`。`。这表明测试套件正在取得进展。在`-`行之后是测试运行的摘要和时间。如果有失败或异常，计数将反映这一点。

最后，有一个`OK`的总结，显示所有测试是否都通过或者有任何测试失败。

如果我们稍微改变测试以确保它失败，我们会看到以下输出：

```py
 **F** 

 **======================================================================** 

 **FAIL: runTest (__main__.GIVEN_Summary_WHEN_1k_samples_THEN_mean)** 

 **----------------------------------------------------------------------** 

 **Traceback (most recent call last):** 

 **File "/Users/slott/Documents/Writing/Python Cookbook/code/ch11_r04.py", line 24, in runTest** 

 **self.assertEqual(501, self.summary.mean)** 

 **AssertionError: 501 != 500.0** 

 **----------------------------------------------------------------------** 

 **Ran 1 test in 0.004s** 

 **FAILED (failures=1)** 

```

对于通过的测试，显示一个`.`，对于失败的测试，显示一个`F`。然后是断言失败的回溯。为了强制测试失败，我们将期望的平均值改为`501`，而不是计算出的平均值`500.0`。

最后有一个`FAILED`的总结。这包括套件作为一个整体失败的原因：`(failures=1)`。

## 还有更多...

在这个例子中，我们在`runTest()`方法中有两个*Then*条件。如果一个失败，测试就会停止作为一个失败，另一个条件就不会被执行。

这是这个测试设计的一个弱点。如果第一个测试失败，我们将得不到所有可能想要的诊断信息。我们应该避免在 `runTest()` 方法中独立收集断言。在许多情况下，一个测试用例可能涉及多个依赖断言；单个失败提供了所有所需的诊断信息。断言的聚类是简单性和诊断细节之间的设计权衡。

当我们需要更多的诊断细节时，我们有两个一般选择：

+   使用多个测试方法而不是 `runTest()`。编写多个以 `test_` 开头的方法。删除任何名为 `runTest()` 的方法。默认的测试加载器将在重新运行公共的 `setUp()` 方法后，分别执行每个 `test_` 方法。

+   使用 `GIVEN_Summary_WHEN_1k_samples_THEN_mean` 类的多个子类，每个子类都有一个单独的条件。由于 `setUp()` 是公共的，这可以被继承。

按照第一种选择，测试类将如下所示：

```py
    class GIVEN_Summary_WHEN_1k_samples_THEN_mean_median(unittest.TestCase): 

        def setUp(self): 
            self.summary = Summary() 
            self.data = list(range(1001)) 
            random.shuffle(self.data) 
            for sample in self.data: 
                self.summary.add(sample) 

        def test_mean(self): 
            self.assertEqual(500, self.summary.mean) 

        def test_median(self): 
            self.assertEqual(500, self.summary.median) 

```

我们已经重构了 `setUp()` 方法，包括测试的 *Given* 和 *When* 条件。两个独立的 *Then* 条件被重构为它们自己单独的 `test_mean()` 和 `test_median()` 方法。没有 `runTest()` 方法。

由于每个测试是单独运行的，我们将看到计算均值或计算中位数的问题的单独错误报告。

### 一些其他断言

`TestCase` 类定义了许多断言，可以作为 *Then* 条件的一部分使用；以下是一些最常用的：

+   `assertEqual()` 和 `assertNotEqual()` 使用默认的 `==` 运算符比较实际值和期望值。

+   `assertTrue()` 和 `assertFalse()` 需要一个布尔表达式。

+   `assertIs()` 和 `assertIsNot()` 使用 `is` 比较来确定两个参数是否是对同一个对象的引用。

+   `assertIsNone()` 和 `assertIsNotNone()` 使用 `is` 来将给定值与 `None` 进行比较。

+   `assertIsInstance()` 和 `assertNotIsInstance()` 使用 `isinstance()` 函数来确定给定值是否是给定类（或类元组）的成员。

+   `assertAlmostEquals()` 和 `assertNotAlmostEquals()` 将给定值四舍五入到七位小数，以查看大部分数字是否相等。

+   `assertRegex()` 和 `assertNotRegex()` 使用正则表达式比较给定的字符串。这使用正则表达式的 `search()` 方法来匹配字符串。

+   `assertCountEqual()` 比较两个序列，看它们是否具有相同的元素，不考虑顺序。这对比较字典键和集合也很方便。

还有更多的断言方法。其中一些提供了检测异常、警告和日志消息的方法。另一组提供了更多类型特定的比较能力。

例如，`Summary` 类的模式特性产生一个列表。我们可以使用特定的 `assertListEqual()` 断言来比较结果：

```py
    class GIVEN_Summary_WHEN_1k_samples_THEN_mode(unittest.TestCase): 

        def setUp(self): 
            self.summary = Summary() 
            self.data = [500]*97 
            # Build 993 more elements each item n occurs n times. 
            for i in range(1,43): 
                self.data += [i]*i 
            random.shuffle(self.data) 
            for sample in self.data: 
                self.summary.add(sample) 

        def test_mode(self): 
            top_3 = self.summary.mode[:3] 
            self.assertListEqual([(500,97), (42,42), (41,41)], top_3) 

```

首先，我们构建了一个包含 1000 个值的集合。其中，有 97 个是数字 500 的副本。剩下的 903 个元素是介于 1 和 42 之间的数字的副本。这些数字有一个简单的规则——频率就是值。这个规则使得确认结果更容易。

`setUp()` 方法将数据随机排序。然后使用 `add()` 方法构建 `Summary` 对象。

我们使用了一个 `test_mode()` 方法。这允许扩展到包括这个测试的其他 *Then* 条件。在这种情况下，我们检查了模式的前三个值，以确保它具有预期的值分布。`assertListEqual()` 比较两个 `list` 对象；如果任一参数不是列表，我们将得到一个更具体的错误消息，显示参数不是预期类型。

### 单独的测试目录

我们已经在被测试的代码的同一模块中显示了 `TestCase` 类的定义。对于小类来说，这可能是有帮助的。与类相关的一切都可以在一个模块文件中找到。

在较大的项目中，将测试文件隔离到一个单独的目录是常见做法。测试可能（而且通常）非常庞大。测试代码的数量可能比应用程序代码还要多，这并不是不合理的。

完成后，我们可以依赖`unittest`框架中的发现应用程序。该应用程序可以搜索给定目录的所有文件以寻找测试文件。通常，这些文件将是名称与模式`test*.py`匹配的文件。如果我们对所有测试模块使用简单、一致的名称，那么它们可以通过简单的命令定位并运行。

`unittest`加载器将在目录中搜索所有从`TestCase`类派生的类。这些类的集合在更大的模块集合中成为完整的`TestSuite`。我们可以使用`os`命令来做到这一点：

```py
 **$ python3 -m unittest discover -s tests** 

```

这将在项目的`tests`目录中找到所有的测试。

## 另请参阅

+   我们将在*结合unittest和doctest测试*的示例中结合`unittest`和`doctest`。我们将在*模拟外部资源*的示例中查看模拟外部对象。

# 结合unittest和doctest测试

在大多数情况下，我们将结合使用`unittest`和`doctest`测试用例。有关doctest的示例，请参阅*使用文档字符串进行测试*的示例。有关unittest的示例，请参阅*创建单独的测试模块和包*的示例。

`doctest`示例是模块、类、方法和函数的文档字符串的重要组成部分。`unittest`案例通常会在一个单独的`tests`目录中，文件的名称与模式`test_*.py`匹配。

我们如何将所有这些不同的测试组合成一个整洁的包呢？

## 准备工作

我们将回顾*使用文档字符串进行测试*的示例。这个示例为一个名为`Summary`的类创建了测试，该类执行一些统计计算。在那个示例中，我们在文档字符串中包含了示例。

该类开始如下：

```py
    class Summary: 
        '''Computes summary statistics. 

        >>> s = Summary() 
        >>> s.add(8) 
        >>> s.add(9) 
        >>> s.add(9) 
        >>> round(s.mean, 2) 
        8.67 
        >>> s.median 
        9 
        >>> print(str(s)) 
        mean = 8.67 
        median = 9 
        '''
```

这里省略了方法，以便我们可以专注于文档字符串中提供的示例。

在*创建单独的测试模块和包*的示例中，我们编写了一些`unittest.TestCase`类来为这个类提供额外的测试。我们创建了类定义如下：

```py
    class GIVEN_Summary_WHEN_1k_samples_THEN_mean_median(unittest.TestCase): 

        def setUp(self): 
            self.summary = Summary() 
            self.data = list(range(1001)) 
            random.shuffle(self.data) 
            for sample in self.data: 
                    self.summary.add(sample) 

        def test_mean(self): 
            self.assertEqual(500, self.summary.mean) 

        def test_median(self): 
            self.assertEqual(500, self.summary.median) 

```

这个测试创建了一个`Summary`对象；这是*给定*方面。然后向该`Summary`对象添加了许多值。这是测试的*当*方面。这两个`test_`方法实现了这个测试的两个*然后*方面。

通常可以看到一个项目文件夹结构，看起来像这样：

```py
    git-project-name/ 
        statstools/ 
            summary.py 
        tests/ 
            test_summary.py 

```

我们有一个顶层文件夹`git-project-name`，与源代码库中的项目名称匹配。我们假设正在使用Git，但也可能使用其他工具。

在顶层目录中，我们将有一些对大型Python项目通用的开销。这将包括文件，如包含项目描述的`README.rst`，可以与`pip`一起使用的`requirements.txt`来安装额外的包，以及可能的`setup.py`来将包安装到标准库中。

目录`statstools`包含一个模块文件`summary.py`。这是我们提供有趣和有用功能的模块。该模块在代码中散布了文档字符串注释。

目录`tests`包含另一个模块文件`test_summary.py`。其中包含了`unittest`测试用例。我们选择了名称`tests`和`test_*.py`，以便它们与自动化测试发现很好地匹配。

我们需要将所有的测试组合成一个单一的、全面的测试套件。

我们将展示的示例使用`ch11_r01`而不是一些更酷的名称，比如`summary`。一个真实的项目通常有巧妙、有意义的名称。书籍内容非常庞大，名称设计得与整体章节和配方大纲相匹配。

## 如何做...

1.  在本例中，我们假设unittest测试用例在与被测试代码分开的文件中。我们将有`ch11_r01`和`test_ch11_r01`。

要使用doctest测试，导入`doctest`模块。我们将把doctest示例与`TestCase`类结合起来，创建一个全面的测试套件：

```py
            import unittest 
            import doctest 

    ```

我们假设`unittest`的`TestCase`类已经就位，我们正在向测试套件中添加更多的测试。

1.  导入正在测试的模块。这个模块将包含一些doctests的字符串：

```py
            import ch11_r01 

    ```

1.  要实现`load_tests`协议，请在测试模块中包含以下函数：

```py
            def load_tests(loader, standard_tests, pattern): 
                return standard_tests 

    ```

这个函数必须有这个名字才能被测试加载器找到。

1.  要包含doctest测试，需要一个额外的加载器。我们将使用`doctest.DocTestSuite`类来创建一个测试套件。这些测试将被添加到作为`standard_tests`参数值提供的测试套件中：

```py
            def load_tests(loader, standard_tests, pattern): 
                dt = doctest.DocTestSuite(ch11_r01) 
                standard_tests.addTests(dt) 
                return standard_tests 

    ```

`loader`参数是当前正在使用的测试用例加载器。`standard_tests`值将是默认加载的所有测试。通常，这是所有`TestCase`的子类的测试套件。模式值是提供给加载器的值。

现在我们可以添加`TestCase`类和整体的`unittest.main()`函数，以创建一个包括unittest `TestCase`和所有doctest示例的全面测试模块。

这可以通过包括以下代码来完成：

```py
    if __name__ == "__main__": 
        unittest.main() 

```

这使我们能够运行模块并执行测试。

## 它是如何工作的...

当我们在这个模块中评估`unittest.main()`时，测试加载器的过程将被限制在当前模块中。加载器将找到所有扩展`TestCase`的类。这些是提供给`load_tests()`函数的标准测试。

我们将用`doctest`模块创建的测试来补充标准测试。通常，我们将能够导入被测试的模块，并使用`DocTestSuite`从导入的模块构建一个测试套件。

`load_tests()`函数会被`unittest`模块自动使用。这个函数可以对给定的测试套件执行各种操作。在这个例子中，我们用额外的测试补充了测试套件。

## 还有更多...

在某些情况下，一个模块可能非常复杂；这可能导致多个测试模块。可能会有几个测试模块，名称类似于`tests/test_module_feature.py`，或者类似的名称，以显示对一个复杂模块的多个功能进行了多次测试。

在其他情况下，我们可能有一个测试模块，其中包含对几个不同但密切相关的模块的测试。一个包可能被分解成多个模块。然而，一个单独的测试模块可能涵盖了被测试包中的所有模块。

当组合许多较小的模块时，可能会在`load_tests()`函数中构建多个测试套件。函数体可能如下所示：

```py
    def load_tests(loader, standard_tests, pattern): 
        for module in ch11_r01, ch11_r02, ch11_r03: 
            dt = doctest.DocTestSuite(module) 
            standard_tests.addTests(dt) 
        return standard_tests 

```

这将包含来自多个模块的`doctests`。

## 另请参阅

+   有关doctest的示例，请参阅*使用文档字符串进行测试*配方。有关unittest的示例，请参阅*创建单独的测试模块和包*配方。

![](image/614271.jpg)

# 测试涉及日期或时间的事物

许多应用程序依赖于`datetime.datetime.now()`来创建时间戳。当我们在单元测试中使用它时，结果基本上是不可能预测的。我们在这里有一个依赖注入的问题，我们的应用程序依赖于一个我们希望只在测试时替换的类。

一个选择是避免使用`now()`和`utcnow()`。我们可以创建一个发出时间戳的工厂函数来代替直接使用这些函数。在测试目的中，这个函数可以被替换为产生已知结果的函数。在一个复杂的应用程序中避免使用`now()`方法似乎有些尴尬。

另一个选择是完全避免直接使用`datetime`类。这需要设计包装`datetime`类的类和模块。然后可以使用一个产生`now()`已知值的包装类进行测试。这也似乎是不必要的复杂。

我们如何处理`datetime`时间戳？

## 准备工作

我们将使用一个创建`CSV`文件的小函数。这个文件的名称将包括日期和时间。我们将创建类似于这样的名称的文件：

```py
    extract_20160704010203.json 

```

这种文件命名约定可能会被长时间运行的服务器应用程序使用。该名称有助于匹配文件和相关的日志事件。它可以帮助跟踪服务器正在执行的工作。

我们将使用这样的函数来创建这些文件：

```py
    import datetime 
    import json 
    from pathlib import Path 

    def save_data(some_payload): 
        now_date = datetime.datetime.utcnow() 
        now_text = now_date.strftime('extract_%Y%m%d%H%M%S') 
        file_path = Path(now_text).with_suffix('.json') 
        with file_path.open('w') as target_file: 
            json.dump(some_payload, target_file, indent=2) 

```

这个函数使用了`utcnow()`。从技术上讲，可以重新设计函数并将时间戳作为参数提供。在某些情况下，这种重新设计可能会有所帮助。还有一个方便的替代重新设计的方法。

我们将创建`datetime`模块的模拟版本，并修补测试上下文以使用模拟版本而不是实际版本。这个测试将包含`datetime`类的模拟类定义。在该类中，我们将提供一个模拟的`utcnow()`方法，该方法将提供预期的响应。

由于被测试的函数创建了一个文件，我们需要考虑这个操作系统的后果。当同名文件已经存在时应该发生什么？应该引发异常吗？文件名应该添加后缀吗？根据我们的设计决定，我们可能需要有两个额外的测试用例：

+   给出一个没有冲突的目录。在这种情况下，一个`setUp()`方法来删除任何先前的测试输出。我们可能还想创建一个`tearDown()`方法来在测试后删除文件。

+   给出一个具有冲突名称的目录。在这种情况下，一个`setUp()`方法将创建一个冲突的文件。我们可能还想创建一个`tearDown()`方法来在测试后删除文件。

对于这个示例，我们将假设重复的文件名并不重要。新文件应该简单地覆盖任何先前的文件，而不会发出警告或通知。这很容易实现，并且通常适用于现实世界的情况，即在不到1秒的时间内创建多个文件没有理由。

## 如何做...

1.  对于这个示例，我们将假设`unittest`测试用例与被测试的代码是同一个模块。导入`unittest`和`unittest.mock`模块：

```py
            import unittest 
            from unittest.mock import * 

    ```

`unittest`模块只是被导入。要使用这个模块的特性，我们必须用`unittest.`来限定名称。从`unittest.mock`导入了所有名称，因此可以在没有任何限定符的情况下使用这些名称。我们将使用模拟模块的许多特性，而且长的限定名称很笨拙。

1.  包括要测试的代码。这是之前显示的。

1.  为测试创建以下骨架。我们提供了一个类定义，以及一个可以用来执行测试的主脚本：

```py
            class GIVEN_data_WHEN_save_data_THEN_file(unittest.TestCase): 
                def setUp(self): 
                    '''GIVEN conditions for the test.''' 

                def runTest(self): 
                    '''WHEN and THEN conditions for this test.'''' 

            if __name__ == "__main__": 
                unittest.main() 

    ```

我们没有定义`load_tests()`函数，因为我们没有任何文档字符串测试要包含。

1.  `setUp()`方法将有几个部分：

+   要处理的示例数据：

```py
                    self.data = {'primes': [2, 3, 5, 7, 11, 13, 17, 19]} 

        ```

+   `datetime`模块的模拟对象。这个对象提供了被测试单元使用的精确特性。`Mock`模块包含了`datetime`类的一个单一`Mock`类定义。在该类中，它提供了一个单一的模拟方法`utcnow()`，它总是提供相同的响应：

```py
                    self.mock_datetime = Mock( 
                        datetime = Mock( 
                            utcnow = Mock( 
                                return_value = datetime.datetime(2017, 7, 4, 1, 2, 3) 
                            ) 
                        ) 
                    ) 

        ```

+   给出上面显示的`datetime`对象的预期文件名：

```py
                    self.expected_name = 'extract_20170704010203.json' 

        ```

+   需要进行一些额外的配置处理来建立*Given*条件。我们将删除要完全确保测试断言不使用来自先前测试运行的文件的任何先前版本：

```py
                    self.expected_path = Path(self.expected_name) 
                    if self.expected_path.exists(): 
                        self.expected_path.unlink() 

        ```

1.  `runTest()`方法将有两个部分：

+   *When*处理。这将修补当前模块`__main__`，以便将对`datetime`的引用替换为`self.mock_datetime`对象。然后在修补的上下文中执行请求：

```py
                    with patch('__main__.datetime', self.mock_datetime): 
                        save_data(self.data) 

        ```

+   *Then*处理。在这种情况下，我们将打开预期的文件，加载内容，并确认结果与源数据匹配。这将以必要的断言结束。如果文件不存在，这将引发`IOError`异常：

```py
                with self.expected_path.open() as result_file: 
                    result_data = json.load(result_file) 
                self.assertDictEqual(self.data, result_data) 

    ```

## 它是如何工作的...

`unittest.mock`模块在这里有两个有价值的组件——`Mock`对象定义和`patch()`函数。

当我们创建`Mock`类的实例时，必须提供结果对象的方法和属性。当我们提供一个命名参数值时，这将被保存为结果对象的属性。简单的值成为对象的属性。基于`Mock`对象的值成为方法函数。

当我们创建一个提供`return_value`（或`side_effect`）命名参数值的`Mock`实例时，我们正在创建一个可调用的对象。这是一个行为像一个非常愚蠢的函数的模拟对象的例子：

```py
 **>>> from unittest.mock import * 
>>> dumb_function = Mock(return_value=12) 
>>> dumb_function(9) 
12 
>>> dumb_function(18) 
12** 

```

我们创建了一个模拟对象`dumb_function`，它将表现得像一个可调用的函数，只返回值`12`。对于单元测试来说，这可能非常方便，因为结果是简单和可预测的。

更重要的是`Mock`对象的这个特性：

```py
 **>>> dumb_function.mock_calls 
[call(9), call(18)]** 

```

`dumb_function()`跟踪了每次调用。然后我们可以对这些调用进行断言。例如，`assert_called_with()`方法检查历史记录中的最后一次调用：

```py
 **>>> dumb_function.assert_called_with(18)** 

```

如果最后一次调用确实是`dumb_function(18)`，那么这将悄无声息地成功。如果最后一次调用不符合断言，那么会引发一个`AssertionError`异常，`unittest`模块将捕获并注册为测试失败。

我们可以像这样看到更多细节：

```py
 **>>> dumb_function.assert_has_calls( [call(9), call(18)] )** 

```

这个断言检查整个调用历史。它使用`Mock`模块的`call()`函数来描述函数调用中提供的参数。

`patch()`函数可以进入模块的上下文并更改该上下文中的任何引用。在这个例子中，我们使用`patch()`来调整`__main__`模块中的定义——当前正在运行的模块。在许多情况下，我们会导入另一个模块，并且需要对导入的模块进行修补。重要的是要到达对被测试模块有效的上下文并修补该引用。

## 还有更多...

在这个例子中，我们为`datetime`模块创建了一个模拟，它具有非常狭窄的功能集。

该模块只有一个元素，即`Mock`类的一个实例，名为`datetime`。对于单元测试，模拟的类通常表现得像一个返回对象的函数。在这种情况下，该类返回了一个`Mock`对象。

代替`datetime`类的`Mock`对象有一个属性`utcnow()`。我们在定义这个属性时使用了特殊的`return_value`关键字，以便它返回一个固定的`datetime`实例。我们可以扩展这种模式，并模拟多个属性以表现得像一个函数。这是一个模拟`utcnow()`和`now()`的例子：

```py
    self.mock_datetime = Mock( 
       datetime = Mock( 
            utcnow = Mock( 
                return_value = datetime.datetime(2017, 7, 4, 1, 2, 3) 
            ), 
            now = Mock( 
                return_value = datetime.datetime(2017, 7, 4, 4, 2, 3) 
            ) 
        ) 
    ) 

```

两个模拟的方法，`utcnow()`和`now()`，分别创建了不同的`datetime`对象。这使我们能够区分这些值。我们可以更容易地确认单元测试的正确操作。

请注意，所有这些`Mock`对象的构造都是在`setUp()`方法中执行的。这是在`patch()`函数进行修补之前很久。在`setUp()`期间，`datetime`类是可用的。在`with`语句的上下文中，`datetime`类不可用，并且被`Mock`对象替换。

我们可以添加以下断言来确认`utcnow()`函数被单元测试正确使用：

```py
    self.mock_datetime.datetime.utcnow.assert_called_once_with() 

```

这将检查`self.mock_datetime`模拟对象。它在这个对象中查看`datetime`属性，我们已经定义了一个`utcnow`属性。我们期望这个属性被调用一次，没有参数值。

如果`save_data()`函数没有正确调用`utcnow()`，这个断言将检测到失败。测试接口的两侧是至关重要的。这导致了测试的两个部分：

+   模拟的`datetime`的结果被被测试的单元适当地使用

+   被测试的单元对模拟的`datetime`对象发出了适当的请求

在某些情况下，我们可能需要确认一个已过时或不推荐使用的方法从未被调用。我们可能会有类似这样的内容来确认另一个方法没有被使用：

```py
    self.assertFalse( self.mock_datetime.datetime.called ) 

```

这种类型的测试在重构软件时使用。在这个例子中，之前的版本可能使用了`now()`方法。更改后，函数需要使用`utcnow()`方法。我们已经包含了一个测试，以确保不再使用`now()`方法。

## 另请参阅

+   创建单独的测试模块和包的配方中有关`unittest`模块的基本使用的更多信息

# 测试涉及随机性的事物

许多应用程序依赖于`random`模块来创建随机值或将值随机排序。在许多统计测试中，会进行重复的随机洗牌或随机子集计算。当我们想要测试其中一个算法时，结果基本上是不可能预测的。

我们有两种选择来尝试使`random`模块足够可预测，以编写有意义的单元测试：

+   设置一个已知的种子值，这是常见的，在许多其他配方中我们已经大量使用了这个。

+   使用`unittest.mock`来用一些不太随机的东西替换`random`模块。

如何对涉及随机性的算法进行单元测试？

## 准备工作

给定一个样本数据集，我们可以计算统计量，如均值或中位数。一个常见的下一步是确定这些统计量对于一些整体人口的可能值。这可以通过一种称为**自助法**的技术来完成。

这个想法是反复对初始数据集进行重采样。每个重采样提供了统计量的不同估计。这个整体的重采样指标集显示了整体人口的测量可能方差。

为了确保重采样算法有效，有助于从处理中消除随机性。我们可以使用`random.choice()`函数的非随机版本对精心策划的数据集进行重采样。如果这样可以正常工作，那么我们有理由相信真正的随机版本也会正常工作。

这是我们的候选重采样函数。我们需要验证这一点，以确保它正确地进行了带替换的抽样：

```py
    def resample(population, N): 
        for i in range(N): 
            sample = random.choice(population) 
            yield sample 

```

我们通常会应用`resample()`函数来填充一个`Counter`对象，用于跟踪特定测量值的每个不同值，例如均值。整体的重采样过程如下：

```py
    mean_distribution = Counter() 
    for n in range(1000): 
        subset = list(resample(population, N)) 
        measure = round(statistics.mean(subset), 1) 
        mean_distribution[measure] += 1 

```

这评估了`resample()`函数`1,000`次。这将导致许多子集，每个子集可能具有不同的均值。这些值用于填充`mean_distribution`对象。

`mean_distribution`的直方图将为人口方差提供有意义的估计。这个方差的估计将有助于显示人口最可能的实际均值。

## 如何做...

1.  定义整体测试类的大纲：

```py
            class GIVEN_resample_WHEN_evaluated_THEN_fair(unittest.TestCase): 
                def setUp(self): 

                def runTest(self): 

            if __name__ == "__main__": 
                unittest.main() 

    ```

我们已经包含了一个主程序，这样我们就可以简单地运行模块来测试它。在使用诸如IDLE之类的工具时，这很方便；我们可以在进行更改后使用*F5*键来测试模块。

1.  定义`random.choice()`函数的模拟版本。我们将提供一个模拟数据集`self.data`，以及对`choice()`函数的模拟响应：

```py
            self.expected_resample_data.self.data = [2, 3, 5, 7, 11, 13, 17, 19] 
            self.expected_resample_data = [23, 29, 31, 37, 41, 43, 47, 53] 
            self.mock_random = Mock( 
                choice = Mock( 
                    side_effect = self.expected_resample_data 
                ) 
            ) 

    ```

我们使用`side_effect`属性定义了`choice()`函数。这将从给定序列中一次返回一个值。我们提供了八个模拟值，这些值与源序列不同，因此我们可以很容易地识别`choice()`函数的输出。

1.  定义测试的*When*和*Then*方面。在这种情况下，我们将修补`__main__`模块，以替换对`random`模块的引用。然后测试可以建立结果是否具有预期的值，并且`choice()`函数是否被多次调用：

```py
            with patch('__main__.random', self.mock_random): 
                resample_data = list(resample(self.data, 8)) 

            self.assertListEqual(self.expected_resample_data, resample_data) 
            self.mock_random.choice.assert_has_calls( 8*[call(self.data)] ) 

    ```

## 工作原理...

当我们创建`Mock`类的实例时，必须提供生成对象的方法和属性。当`Mock`对象包括一个命名参数值时，这将被保存为生成对象的属性。

当我们创建一个提供`side_effect`命名参数值的`Mock`实例时，我们正在创建一个可调用对象。可调用对象将从`side_effect`列表中返回一个值，每次调用`Mock`对象时。

这是一个行为像一个非常愚蠢的函数的模拟对象的例子：

```py
 **>>> from unittest.mock import * 
>>> dumb_function = Mock(side_effect=[11,13]) 
>>> dumb_function(23) 
11 
>>> dumb_function(29) 
13 
>>> dumb_function(31)   
Traceback (most recent call last): 
  ... (traceback details omitted) 
StopIteration** 

```

首先，我们创建了一个`Mock`对象，并将其分配给名称`dumb_function`。这个`Mock`对象的`side_effect`属性提供了一个将返回的两个不同值的短列表。

然后的例子使用两个不同的参数值两次评估`dumb_function()`。每次，下一个值从`side_effect`列表中返回。第三次尝试引发了一个`StopIteration`异常，导致了测试失败。

这种行为使我们能够编写一个测试，检测函数或方法的某些不当使用。如果函数被调用太多次，将引发异常。其他不当使用必须使用各种断言来检测可以用于`Mock`对象的各种类型。

## 还有更多...

我们可以轻松地用提供适当行为的模拟对象替换`random`模块的其他特性，而不实际上是随机的。例如，我们可以用一个提供已知顺序的函数替换`shuffle()`函数。我们可以像这样遵循上面的测试设计模式：

```py
    self.mock_random = Mock( 
        choice = Mock( 
            side_effect = self.expected_resample_data 
        ), 
        shuffle = Mock( 
            return_value = self.expected_resample_data 
        ) 
    ) 

```

这个模拟的`shuffle()`函数返回一组不同的值，可以用来确认某个过程是否正确使用了`random`模块。

## 另请参阅

+   在[第4章](text00048.html#page "第4章。内置数据结构-列表、集合、字典")中，*内置数据结构-列表、集合、字典*，*使用集合方法和运算符*，*创建字典-插入和更新*配方，以及[第5章](text00063.html#page "第5章。用户输入和输出")中的*用户输入和输出*，*使用cmd创建命令行应用程序*配方，展示了如何种子随机数生成器以创建可预测的值序列。

+   在[第6章](text00070.html#page "第6章。类和对象的基础")中，*类和对象的基础*，还有其他几个配方展示了另一种方法，例如*使用类封装数据+处理*，*设计具有大量处理的类*，*使用__slots__优化小对象*和*使用惰性属性*。

+   此外，在[第7章](text00079.html#page "第7章。更高级的类设计")中，*更高级的类设计*，请参阅*选择继承和扩展之间的选择-是一个问题*，*通过多重继承分离关注*，*利用Python的鸭子类型*，*创建一个具有可排序对象的类*和*定义一个有序集合*配方。

# 模拟外部资源

*涉及日期或时间的测试*和*涉及随机性的测试*配方展示了模拟相对简单对象的技术。在*涉及日期或时间的测试*配方中，被模拟的对象基本上是无状态的，一个返回值就可以很好地工作。在*涉及随机性的测试*配方中，对象有一个状态变化，但状态变化不依赖于任何输入参数。

在这些更简单的情况下，测试提供了一系列请求给一个对象。可以构建基于已知和精心计划的状态变化序列的模拟对象。测试用例精确地遵循对象的内部状态变化。这有时被称为白盒测试，因为需要定义测试序列和模拟对象的实现细节。

然而，在某些情况下，测试场景可能不涉及明确定义的状态更改序列。被测试的单元可能以难以预测的顺序发出请求。这有时是黑盒测试的结果，其中实现细节是未知的。

我们如何创建更复杂的模拟对象，这些对象具有内部状态并进行自己的内部状态更改？

## 准备工作

我们将研究如何模拟有状态的RESTful Web服务请求。在这种情况下，我们将使用弹性数据库的数据库API。有关此数据库的更多信息，请参见[https://www.elastic.co/](https://www.elastic.co/)。该数据库具有使用简单的RESTful Web服务的优势。这些可以很容易地模拟为简单、快速的单元测试。

对于这个配方，我们将测试一个使用RESTful API创建记录的函数。**表述性状态转移**（**REST**）是一种使用**超文本传输协议**（**HTTP**）在进程之间传输对象状态表示的技术。例如，要创建一个数据库记录，客户端将使用HTTP `POST`请求将对象状态的表示传输到数据库服务器。在许多情况下，JSON表示法用于表示对象状态。

测试这个函数将涉及模拟`urllib.request`模块的一部分。替换`urlopen()`函数将允许测试用例模拟数据库活动。这将允许我们测试依赖于Web服务的函数，而不实际进行可能昂贵或缓慢的外部请求。

在我们的应用软件中，有两种总体方法可以使用弹性搜索API：

+   我们可以在我们的笔记本电脑或一些我们可以访问的服务器上安装弹性数据库。安装是一个两部分的过程，首先安装适当的**Java开发工具包**（**JDK**），然后安装ElasticSearch软件。我们不会在这里详细介绍，因为我们有一个似乎更简单的替代方案。

在本地计算机上创建和访问对象的URL将如下所示：

```py
            http://localhost:9200/eventlog/event/ 

    ```

请求将在请求的正文中使用多个数据项。这些请求不需要任何HTTP头部用于安全或认证目的。

+   我们可以使用诸如[http://orchestrate.io](http://orchestrate.io)之类的托管服务。这需要注册该服务以获取API密钥，而不是安装软件。API密钥授予对定义应用程序的访问权限。在应用程序中，可以创建多个集合。由于我们不必安装额外的软件，这似乎是一个方便的方法。

在远程服务器上处理对象的URL将如下所示：

```py
            https://api.orchestrate.io/v0/eventlog/ 

    ```

请求还将使用多个HTTP头部向主机提供信息。接下来，我们将详细了解这项服务。

要创建的文档的数据有效载荷将如下所示：

```py
    { 
        "timestamp": "2016-06-15T17:57:54.715", 
        "levelname": "INFO", 
        "module": "ch09_r10", 
        "message": "Sample Message One" 
    } 

```

这个JSON文档代表了一个日志条目。这是在之前的示例中使用的`sample.log`文件中提取的。这个文档可以被理解为将保存在数据库的`eventlog`索引中的事件类型的特定实例。该对象有四个属性，其值为字符串。

在[第9章](text00099.html#page "第9章。输入/输出、物理格式和逻辑布局")的*使用正则表达式读取复杂格式*配方中，*输入/输出、物理格式和逻辑布局*，展示了如何解析复杂的日志文件。在*使用多个上下文读写文件*的配方中，复杂的日志记录被写入了`CSV`文件。在这个例子中，我们将展示如何将日志记录放入使用弹性等数据库的基于云的存储中。

### 在entrylog集合中创建一个条目文档

我们将在数据库的`entrylog`集合中创建条目文档。使用HTTP `POST`请求创建新项目。`201 Created`的响应将表明数据库创建了新事件。

要使用 `orchestrate.io` 数据库服务，每个请求都有一个基本 URL。我们可以用这样的字符串来定义它：

```py
    service = "https://api.orchestrate.io" 

```

使用 `https` 方案是为了确保数据在客户端和服务器之间是私密的，使用 **SSL** 。主机名是 `api.orchestrate.io`。每个请求将基于这个基本服务定义的 URL。

每个请求的 HTTP 头将如下所示：

```py
    headers = { 
        'Accept': 'application/json', 
        'Content-Type': 'application/json', 
        'Authorization': basic_header(api_key, '') 
    } 

```

`Accept` 头显示期望的响应类型。`Content-Type` 头显示内容所使用的文档表示类型。这两个头指示数据库使用 JSON 表示对象状态。

`Authorization` 头是 API 密钥的发送方式。这个头的值是一个相当复杂的字符串。最容易的方法是构建编码的 API 密钥字符串代码如下：

```py
    import base64 
    def basic_header(username, password): 
        combined_bytes = (username + ':' + password).encode('utf-8') 
        encoded_bytes = base64.b64encode(combined_bytes) 
        return 'Basic ' + encoded_bytes.decode('ascii') 

```

这段代码将把用户名和密码组合成一个字符流，然后使用 `UTF-8` 编码方案将这些字符编码为字节流。`base64` 模块创建了第二个字节流。在这个输出流中，四个字节将包含构成三个输入字节的位。这些字节是从一个简化的字母表中选择的。然后将这个值与关键字 `'Basic '` 转换回 Unicode 字符。这个值可以与 `Authorization` 头一起使用。

通过创建一个 `Request` 对象来使用 RESTful API 是最容易的。该类在 `urllib.request` 模块中定义。`Request` 对象结合了数据、URL 和头，并命名了特定的 HTTP 方法。以下是创建 `Request` 实例的代码：

```py
    data_document = { 
        "timestamp": "2016-06-15T17:57:54.715", 
        "levelname": "INFO", 
        "module": "ch09_r10", 
        "message": "Sample Message One" 
    } 

    headers={ 
        'Accept': 'application/json', 
        'Content-Type': 'application/json', 
        'Authorization': basic_header(api_key, '') 
    }     

    request = urllib.request.Request( 
        url=service + '/v0/eventlog', 
        headers=headers, 
        method='POST', 
        data=json.dumps(data_document).encode('utf-8') 
    ) 

```

请求对象包括四个元素：

+   `url` 参数的值是基本服务 URL 加上集合名称，`/v0/eventlog`。路径中的 `v0` 是必须在每个请求中提供的版本信息。

+   `headers` 参数包括具有授权访问应用程序的 API 密钥的 `Authorization` 头。

+   `POST` 方法将在数据库中创建一个新对象。

+   `data` 参数是要保存的文档。我们已经将一个 Python 对象转换为 JSON 表示的字符串。然后使用 `UTF-8` 编码将 Unicode 字符编码为字节。

### 查看典型的响应

处理涉及发送请求和接收响应。`urlopen()` 函数接受 `Request` 对象作为参数；这构建了发送到数据库服务器的请求。来自数据库服务器的响应将包括三个元素：

+   状态。这包括一个数字代码和一个原因字符串。创建文档时，预期的响应代码是 `201`，字符串是 `CREATED`。对于许多其他请求，代码是 `200`，字符串是 `OK`。

+   响应还将包括头信息。对于创建请求，这些将包括以下内容：

```py
            [ 
             ('Content-Type', 'application/json'), 
             ('Location', '/v0/eventlog/12950a87ef024e43/refs/8e50b6bfc50b2dfa'), 
             ('ETag', '"8e50b6bfc50b2dfa"'), 
             ... 
             ] 

    ```

`Content-Type` 头告诉我们内容是以 JSON 编码的。`Location` 头提供了一个 URL，可以用来检索创建的对象。它还提供了一个 `ETag` 头，这是对象当前状态的哈希摘要；这有助于支持缓存对象的本地副本。其他头可能存在；我们在示例中只显示了 `...` 。

+   响应可能有一个主体。如果存在，这将是从数据库检索到的一个 JSON 编码的文档（或文档）。必须使用响应的 `read()` 方法来读取主体。主体可能非常大；`Content-Length` 头提供了确切的字节数。

### 数据库访问的客户端类

我们将为数据库访问定义一个简单的类。一个类可以为多个相关操作提供上下文和状态信息。在使用 Elastic 数据库时，访问类可以只创建一次请求头字典，并在多个请求中重复使用。

这是数据库客户端类的本质。我们将在几个部分中展示这一点。首先是整个类的定义：

```py
    class ElasticClient: 
        service = "https://api.orchestrate.io" 

```

这定义了一个类级别的变量`service`，带有方案和主机名。初始化方法`__init__()`可以构建各种数据库操作中使用的标头：

```py
    def __init__(self, api_key, password=''): 
        self.headers = { 
            'Accept': 'application/json', 
            'Content-Type': 'application/json', 
            'Authorization': ElasticClient.basic_header(api_key, password), 
        } 

```

这个方法接受API密钥并创建一组依赖于HTTP基本授权的标头。密码不会被编排服务使用。但我们已经包含了它，因为用户名和密码用于示例单元测试用例。

这是方法：

```py
    @staticmethod 
    def basic_header(username, password=''): 
        """ 
        >>> ElasticClient.basic_header('Aladdin', 'OpenSesame') 
        'Basic QWxhZGRpbjpPcGVuU2VzYW1l' 
        """ 
        combined_bytes = (username + ':' + password).encode('utf-8') 
        encoded_bytes = base64.b64encode(combined_bytes) 
        return 'Basic ' + encoded_bytes.decode('ascii') 

```

这个函数可以将用户名和密码组合起来，创建HTTP`Authorization`标头的值。`orchestrate.io` API使用分配的API密钥作为用户名；密码是一个零长度的字符串`''`。当有人注册他们的服务时，API密钥就被分配了。免费级别的服务允许合理数量的交易和一个舒适小的数据库。

我们已经包含了一个以文档字符串形式的单元测试用例。这提供了结果正确的证据。测试用例来自维基百科关于HTTP基本认证的页面。

最后一部分是一个将一个数据项加载到数据库的`eventlog`集合中的方法：

```py
    def load_eventlog(self, data_document): 
        request = urllib.request.Request( 
            url=self.service + '/v0/eventlog', 
            headers=self.headers, 
            method='POST', 
            data=json.dumps(data_document).encode('utf-8') 
        ) 

        with urllib.request.urlopen(request) as response: 
            assert response.status == 201, "Insertion Error" 
            response_headers = dict(response.getheaders()) 
            return response_headers['Location'] 

```

这个函数使用四个必需的信息构建一个`Request`对象——完整的URL、HTTP标头、方法字符串和编码数据。在这种情况下，数据被编码为JSON字符串，并使用`UTF-8`编码方案将JSON字符串编码为字节。

评估`urlopen()`函数会发送请求并检索一个响应对象。这个对象被用作上下文管理器。`with`语句确保即使在响应处理过程中引发异常，资源也会被正确释放。

`POST`方法应该以`201`状态响应。任何其他状态都是问题。在这段代码中，状态是通过`assert`语句进行检查的。最好提供一条消息，比如`Expected 201 status, got {}.format(response.status)`。

然后检查标头以获取`Location`标头。这提供了一个用于定位已创建对象的URL片段。

## 如何做...

1.  创建数据库访问模块。这个模块将包含`ElasticClient`类定义。它还将包含这个类需要的任何其他定义。

1.  这个示例将使用`unittest`和`doctest`来创建一个统一的测试套件。它将使用`unittest.mock`中的`Mock`类，以及`json`。由于这个模块是与被测试的单元分开的，它需要导入`ch11_r08_load`，该模块包含将被测试的类定义：

```py
            import unittest 
            from unittest.mock import * 
            import doctest 
            import json 
            import ch11_r08_load 

    ```

1.  这是一个测试用例的整体框架。我们将在下面填写这个测试的`setUp()`和`runTest()`方法。名称显示了当我们调用`load_eventlog()`时，我们得到了一个`ElasticClient`实例，然后进行了一个正确的RESTful API请求：

```py
            class GIVEN_ElasticClient_WHEN_load_eventlog_THEN_request(unittest.TestCase): 

                def setUp(self): 

                def runTest(self): 

    ```

1.  `setUp()`方法的第一部分是一个模拟上下文管理器，提供类似于`urlopen()`函数的响应：

```py
            def setUp(self): 
                # The context manager object itself. 
                self.mock_context = Mock( 
                    __exit__ = Mock(return_value=None), 
                    __enter__ = Mock( 
                        side_effect = self.create_response 
                    ),      
                ) 

                # The urlopen() function that returns a context. 
                self.mock_urlopen = Mock( 
                    return_value = self.mock_context, 
                ) 

    ```

当调用`urlopen()`时，返回值是一个行为像上下文管理器的响应对象。模拟这个的最佳方法是返回一个模拟上下文管理器。模拟上下文管理器的`__enter__()`方法执行真正的工作来创建响应对象。在这种情况下，`side_effect`属性标识了一个辅助函数，该函数将被调用来准备从调用`__enter__()`方法的结果。`self.create_response`还没有被定义。我们将使用一个函数，定义如下。

1.  `setUp()`方法的第二部分是一些要加载的模拟数据：

```py
            # The test document. 
            self.document = { 
                "timestamp": "2016-06-15T17:57:54.715", 
                "levelname": "INFO", 
                "module": "ch09_r10", 
                "message": "Sample Message One" 
            } 

    ```

在一个更复杂的测试中，我们可能想要模拟一个大型的可迭代文档集合。

1.  这是一个`create_response()`辅助方法，用于构建类似响应的对象。响应对象可能很复杂，因此我们定义了一个函数来创建它们：

```py
            def create_response(self): 
                self.database_id = hex(hash(self.mock_urlopen.call_args[0][0].data))[2:] 
                self.location = '/v0/eventlog/{id}'.format(id=self.database_id) 
                response_headers = [ 
                    ('Location', self.location), 
                    ('ETag', self.database_id), 
                    ('Content-Type', 'application/json'), 
                ] 
                return Mock( 
                    status = 201, 
                    getheaders = Mock(return_value=response_headers) 
                ) 

    ```

这个方法使用`self.mock_urlopen.call_args`来检查对这个`Mock`对象的最后一次调用。这个调用的参数是一个包含位置参数值和关键字参数的元组。第一个`[0]`索引从元组中选择位置参数值。第二个`[0]`索引选择第一个位置参数值。这将是要加载到数据库中的对象。`hex()`函数的值是一个包含`0x`前缀的字符串，我们将其丢弃。

在更复杂的测试中，可能需要这个方法来保持一个加载到数据库中的对象的缓存，以便更准确地模拟类似数据库的响应。

1.  `runTest()`方法对被测试的模块进行了补丁。它定位了从`ch11_r08_load`到`urllib.request`和`urlopen()`函数的引用。这些引用被替换为`mock_urlopen`替代品：

```py
            def runTest(self): 
                with patch('ch11_r08_load.urllib.request.urlopen', self.mock_urlopen): 
                    client = ch11_r08_load.ElasticClient('Aladdin', 'OpenSesame') 
                    response = client.load_eventlog(self.document) 

                self.assertEqual(self.location, response) 

                call_request = self.mock_urlopen.call_args[0][0] 
                self.assertEqual( 
                    'https://api.orchestrate.io/v0/eventlog', call_request.full_url) 
                self.assertDictEqual( 
                    {'Accept': 'application/json', 
                     'Authorization': 'Basic QWxhZGRpbjpPcGVuU2VzYW1l', 
                     'Content-type': 'application/json' 
                    }, 
                     call_request.headers) 
                self.assertEqual('POST', call_request.method) 
                self.assertEqual( 
                    json.dumps(self.document).encode('utf-8'), call_request.data) 

                self.mock_context.__enter__.assert_called_once_with() 
                self.mock_context.__exit__.assert_called_once_with(None, None, None) 

    ```

这个测试遵循`ElasticClient`首先创建一个客户端对象的要求。它不使用实际的API密钥，而是使用用户名和密码，这将为`Authorization`头创建一个已知的值。`load_eventlog()`的结果是一个类似响应的对象，可以检查它是否具有正确的值。

所有这些交互都将通过模拟对象完成。我们可以使用各种断言来确认是否创建了一个正确的请求对象。测试检查请求对象的四个属性，并确保上下文的使用是否正确。

1.  我们还将定义一个`load_tests()`函数，将这个`unittest`套件与`ch11_r08_load`的文档字符串中找到的任何测试示例结合起来：

```py
            def load_tests(loader, standard_tests, pattern): 
                dt = doctest.DocTestSuite(ch11_r08_load) 
                standard_tests.addTests(dt) 
                return standard_tests 

    ```

1.  最后，我们将提供一个整体的主程序来运行完整的测试套件。这样可以很容易地将测试模块作为独立的脚本运行：

```py
            if __name__ == "__main__": 
                unittest.main() 

    ```

## 工作原理...

这个示例结合了许多`unittest`和`doctest`特性，创建了一个复杂的测试用例。这些特性包括：

+   创建上下文管理器

+   使用side-effect功能创建动态、有状态的测试

+   模拟复杂对象

+   使用加载测试协议来结合doctest和unittest案例

我们将分别查看这些特性。

### 创建上下文管理器

上下文管理器协议在对象外部包装了一个额外的间接层。有关此内容的更多信息，请参阅*使用上下文管理器读写文件*和*使用多个上下文读写文件*的示例。必须模拟的核心特性是`__enter__()`和`__exit__()`方法。

模拟上下文管理器的模式如下：

```py
    self.mock_context = Mock( 
        __exit__ = Mock(return_value=None), 
        __enter__ = Mock( 
            side_effect = self.create_response 
            # or 
            # return_value = some_value 
        ), 
    ) 

```

上下文管理器对象有两个属性。`__exit__()`将被调用一次。`True`的返回值将使任何异常静音。`None`或`False`的返回值将允许异常传播。

`__enter__()`方法返回在`with`语句中分配的对象。在这个例子中，我们使用了`side_effect`属性并提供了一个函数，以便可以计算动态结果。

`__enter__()`方法的一个常见替代方法是使用固定的`return_value`属性，并每次提供相同的管理器对象。还可以使用`side_effect`提供一个序列；在这种情况下，每次调用该方法时，都会返回序列中的另一个对象。

### 创建动态、有状态的测试

在许多情况下，测试可以使用静态的、固定的对象集。模拟响应可以在`setUp()`方法中定义。然而，在某些情况下，对象的状态可能需要在复杂测试的操作过程中发生变化。在这种情况下，可以使用`Mock`对象的`side_effect`属性来跟踪状态变化。

在这个例子中，`side_effect`属性使用`create_response()`方法来构建动态响应。`side_effect`引用的函数可以做任何事情；这可以用来更新动态状态信息，用于计算复杂的响应。

这里有一个微妙的界限。一个复杂的测试用例可能会引入自己的错误。通常最好尽可能简单地保持测试用例，以避免不得不编写`元测试`来测试测试用例。

对于非平凡的测试，确保测试实际上可以失败很重要。有些测试涉及无意的同义反复。可能会创建一个人为的测试，其意义与`self.assertEqual(4, 2+2)`一样。为了确保测试实际上使用了被测试的单元，当代码缺失或注入了错误时，它应该失败。

### 模拟一个复杂对象

`urlopen()`的响应对象具有大量的属性和方法。对于我们的单元测试，我们只需要设置其中的一些特性。

我们使用了以下内容：

```py
    return Mock( 
        status = 201, 
       getheaders = Mock(return_value=response_headers) 
    ) 

```

这创建了一个具有两个属性的`Mock`对象：

+   `status`属性有一个简单的数值。

+   `getheaders`属性使用了一个`Mock`对象，具有`return_value`属性来创建一个方法函数。这个方法函数返回了动态的`response_headers`值。

`response_headers`的值是一个包含*(key, value)*对的两元组序列。这种响应头的表示可以很容易地转换成字典。

对象是这样构建的：

```py
    response_headers = [ 
        ('Location', self.location), 
        ('ETag', self.database_id), 
        ('Content-Type', 'application/json'), 
    ] 

```

这设置了三个头：`Location`，`ETag`和`Content-Type`。根据测试用例可能需要其他头。重要的是不要在测试用例中添加未使用的头部。这种混乱可能导致测试本身的错误。

数据库id和位置是基于以下计算：

```py
    hex(hash(self.mock_urlopen.call_args[0][0].data))[2:] 

```

这使用了`self.mock_urlopen.call_args`来检查提供给测试用例的参数。`call_args`属性的值是一个包含位置参数和关键字参数值的二元组。位置参数也是一个元组。这意味着`call_args[0]`是位置参数，`call_args[0][0]`是第一个位置参数。这将是加载到数据库的文档。

许多Python对象都有哈希值。在这种情况下，预期对象是由`json.dumps()`函数创建的字符串。这个字符串的哈希值是一个大数。该数字的十六进制值将是一个带有`0x`前缀的字符串。我们将使用`[2:]`切片来忽略前缀。有关此信息，请参见[第1章](text00014.html#page "第1章. 数字、字符串和元组")中的*重写不可变字符串*一节，*数字、字符串和元组*。

### 使用load_tests协议

一个复杂的模块将包括类和函数定义。整个模块需要一个描述性的文档字符串。每个类和函数都需要一个文档字符串。类中的每个方法也需要一个文档字符串。这将提供关于模块、类、函数和方法的基本信息。

此外，每个文档字符串都可以包含一个示例。这些示例可以通过`doctest`模块进行测试。有关示例的信息，请参见*使用文档字符串进行测试*一节。我们可以将文档字符串示例测试与更复杂的单元测试结合起来。有关如何执行此操作的更多信息，请参见*结合unittest和doctest测试*一节。

## 还有更多...

`unittest`模块也可以用于构建集成测试。集成测试的想法是避免模拟，实际上在测试模式下使用真实的外部服务。这可能会很慢或很昂贵；通常要避免集成测试，直到所有单元测试提供了软件可能正常工作的信心。

例如，我们可以使用`orchestrate.io`创建两个应用程序——真实应用程序和测试应用程序。这将为我们提供两个API密钥。测试密钥将被用于将数据库重置为初始状态，而不会为真实数据的实际用户创建问题。

我们可以使用`unittest`、`setUpModule()`和`tearDownModule()`函数来控制这一切。`setUpModule()`函数在给定模块文件中的所有测试之前执行。这是设置数据库为已知状态的一种方便方式。

我们还可以使用`tearDownModule()`函数来删除数据库。这对于删除测试创建的不必要的资源非常方便。有时为了调试目的，保留资源可能更有帮助。因此，`tearDownModule()`函数可能不像`setUpModule()`函数那样有用。

## 另请参阅

+   *涉及日期或时间的测试*和*涉及随机性的测试*配方展示了技巧。

+   在[第9章](text00099.html#page "第9章。输入/输出、物理格式和逻辑布局")的*输入/输出、物理格式和逻辑布局*中，*使用正则表达式读取复杂格式*配方展示了如何解析复杂的日志文件。在*使用多个上下文读写文件*配方中，复杂的日志记录被写入了一个`CSV`文件。

+   有关如何切割字符串以替换部分内容的信息，请参阅*重写不可变字符串*配方。

+   这些内容的一部分可以通过`doctest`模块进行测试。请参阅*使用文档字符串进行测试*配方以获取示例。将这些测试与任何doctests结合起来也很重要。有关如何执行此操作的更多信息，请参阅*结合unittest和doctest测试*配方。
