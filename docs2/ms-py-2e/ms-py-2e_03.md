

# 第三章：Pythonic 语法和常见陷阱

在本章中，你将学习如何编写 Python 风格的代码，同时了解一些 Python 的常见陷阱以及如何规避它们。这些陷阱从将列表或字典（可变对象）作为参数传递到更高级的陷阱，如闭包中的后期绑定。你还将了解如何以干净的方式解决循环导入问题。本章示例中使用的某些技术可能对于如此早期的章节来说显得有些高级。不过，不要担心，因为其内部工作原理将在后面的内容中介绍。

本章我们将探讨以下主题：

+   代码风格（PEP 8、`pyflakes`、`flake8`等）

+   常见陷阱（将列表或字典（可变对象）作为函数参数，值传递与引用传递，以及继承行为）

本章中使用的 Pythonic 代码的定义基于普遍接受的编码指南和我的主观意见。在项目工作中，保持与项目编码风格的一致性是最重要的。

# Python 简史

Python 项目始于 1989 年 12 月，是 Guido van Rossum 在圣诞节休假期间的一个业余项目。他的目标是编写一个易于使用的 ABC 编程语言的继任者，并修复限制其适用性的问题。Python 的主要设计目标之一，并且始终如此，就是可读性。这就是本章第一部分的内容：可读性。

为了便于添加新功能和保持可读性，开发了**Python 增强提案**（**PEP**）流程。此流程允许**任何人**提交一个 PEP 以添加新功能、库或其他内容。经过在 Python 邮件列表上的讨论和一些改进后，将决定接受或拒绝该提案。

Python 风格指南（PEP 8：[`peps.python.org/pep-0008/`](https://peps.python.org/pep-0008/））最初作为那些 PEP 之一被提交，被接受，并且自那时起一直得到定期改进。它包含许多广受好评的约定，以及一些有争议的约定。特别是，79 个字符的最大行长度是讨论的焦点。将行限制在 79 个字符确实有一些优点。最初，这个选择是因为终端宽度为 80 个字符，但如今，更大的显示器允许你将多个文件并排放置。对于文档字符串和注释，建议使用 72 个字符的限制以提高可读性。此外，这是 Linux/Unix man（手册）页面的常见约定。

虽然仅仅风格指南本身并不能使代码具有 Python 风格，正如*Python 的禅宗*（PEP 20：[https://peps.python.org/pep-0020/](https://peps.python.org/pep-0020/））优雅地所说：“优美胜于丑陋。”PEP 8 以精确的方式定义了代码的格式，而 PEP 20 则更多地是一种哲学和心态。

几乎 30 年来，Python 项目的所有重大决策都是由 Guido van Rossum 做出的，他被亲切地称为**BDFL**（**终身仁慈独裁者**）。不幸的是，BDFL 中的“终身”部分在关于 PEP 572 的激烈辩论后并未实现。PEP 572（本章后面将详细介绍）是一个关于赋值运算符的提案，它允许在`if`语句内设置变量，这是 C、C++、C#等语言中的常见做法。Guido van Rossum 并不喜欢这种语法，并反对了 PEP。这引发了一场巨大的辩论，他遇到了如此大的阻力，以至于他不得不辞去 BDFL 的职务。许多人对此感到悲伤，因为 Guido van Rossum，这位社区普遍喜爱的决策者，觉得他不得不这样做。至少对我来说，我将非常怀念他作为决策者的洞察力。我希望我们还能看到他的“时间机器”发挥作用几次。Guido van Rossum 被认为拥有时间机器，因为他反复用“我昨晚刚刚实现了那个功能”来回答功能请求。

没有 BDFL 做出最终决定，Python 社区不得不想出一种新的决策方式，为此已经撰写了一系列提案来解决这一问题：

+   PEP 8010：维持现状（ish）：[https://peps.python.org/pep-8010/](https://peps.python.org/pep-8010/)

+   PEP 8011：类似于现状但有三位共同领导者：[`peps.python.org/pep-8011/`](https://peps.python.org/pep-8011/)

+   PEP 8012：无中央权威：[`peps.python.org/pep-8012/`](https://peps.python.org/pep-8012/)

+   PEP 8013：非核心监督：[`peps.python.org/pep-8013/`](https://peps.python.org/pep-8013/)

+   PEP 8014：核心监督：[`peps.python.org/pep-8014/`](https://peps.python.org/pep-8014/)

+   PEP 8015：Python 社区组织：[`peps.python.org/pep-8015/`](https://peps.python.org/pep-8015/)

+   PEP 8016：指导委员会模型：[`peps.python.org/pep-8016/`](https://peps.python.org/pep-8016/)

经过一番小辩论后，PEP 8016——指导委员会模型——被接受为解决方案。PEP 81XX 已被预留用于未来指导委员会的选举，其中 PEP 8100 用于 2019 年的选举，PEP 8101 用于 2020 年的选举，以此类推。

# 代码风格 - 什么是 Pythonic 代码？

当你第一次听说 Pythonic 代码时，你可能会认为它是一种编程范式，类似于面向对象或函数式编程。实际上，它更多的是一种设计哲学。Python 让你自由选择以面向对象、过程式、函数式、面向方面或甚至逻辑导向的方式编程。这些自由使得 Python 成为了一种极佳的编程语言，但它们也有缺点，即需要更多的纪律来保持代码的整洁和可读性。PEP 8 告诉我们如何格式化代码，而 PEP 20 则是关于风格以及如何编写 Pythonic 代码。PEP 20，Pythonic 哲学，是关于以下方面的代码：

+   整洁

+   简单

+   美观

+   明确性

+   可读性

其中大部分听起来像是常识，我认为它们应该是。然而，有些情况下，编写代码并没有一个明显的方法（除非你是荷兰人，当然，你将在本章后面读到这一点）。这就是本章的目标——帮助你学习如何编写漂亮的 Python 代码，并理解 Python 风格指南中某些决策的原因。

让我们开始吧。

## 使用空白符而不是花括号

对于非 Python 程序员来说，Python 最常见的一个抱怨是使用空白符而不是花括号。两种情况都有可说的，最终，这并不那么重要。由于几乎每种编程语言默认都使用类似的缩进规则，即使有花括号，为什么不尽可能省略花括号，使代码更易读呢？这就是 Guido van Rossum 在设计 Python 语言时可能想到的。

在某个时候，一些程序员问 Guido van Rossum Python 是否将支持花括号。从那天起，通过`__future__`导入已经可以使用花括号了。试一试：

```py
>>> from __future__ import braces 
```

接下来，让我们谈谈字符串的格式化。

## 字符串格式化——printf、str.format 还是 f-string？

Python 长期以来一直支持 printf 风格（`%`）和`str.format`，因此你很可能已经熟悉这两种方法。随着 Python 3.6 的引入，又增加了一个选项，即 f-string（PEP 498）。f-string 是`str.format`的便捷简写，有助于简洁（因此，我会争辩说，可读性）。

PEP 498 – 字面字符串插值：[`peps.python.org/pep-0498/`](https://peps.python.org/pep-0498/)

本书的前一版主要使用了 printf 风格，因为在代码示例中简洁性很重要。虽然按照 PEP 8 的规定最大行长度为 79 个字符，但本书在换行前限制为 66 个字符。有了 f-string，我们终于有了 printf 风格的简洁替代方案。

**本书运行代码的小贴士**

由于大部分代码包含`>>>`前缀，只需将其复制/粘贴到 IPython 中，它就会像常规 Python 代码一样执行。

或者，本书的 GitHub 仓库有一个脚本来自动将 doctest 风格示例转换为常规 Python：[`github.com/mastering-python/code_2/blob/master/doctest_to_python.py`](https://github.com/mastering-python/code_2/blob/master/doctest_to_python.py)

为了展示 f-string 的力量，让我们看看`str.format`和 printf 风格并排的几个示例。

本章中的示例显示了 Python 控制台返回的输出。对于常规 Python 文件，您需要添加`print()`才能看到输出。

### 简单格式化

格式化一个简单的字符串：

```py
# Simple formatting
>>> name = 'Rick'

>>> 'Hi %s' % name
'Hi Rick'

>>> 'Hi {}'.format(name)
'Hi Rick' 
```

使用两位小数格式化浮点数：

```py
>>> value = 1 / 3

>>> '%.2f' % value
'0.33'

>>> '{:.2f}'.format(value)
'0.33' 
```

第一个真正的优势在于多次使用变量时。在不使用命名值的情况下，printf 风格无法做到这一点：

```py
>>> name = 'Rick'
>>> value = 1 / 3

>>> 'Hi {0}, value: {1:.3f}. Bye {0}'.format(name, value)
'Hi Rick, value: 0.333\. Bye Rick' 
```

如您所见，我们通过使用引用`{0}`两次来使用`name`。

### 命名变量

使用命名变量相当类似，这也是我们接触到 f 字符串魔力的地方：

```py
>>> name = 'Rick'

>>> 'Hi %(name)s' % dict(name=name)
'Hi Rick'

>>> 'Hi {name}'.format(name=name)
'Hi Rick'

>>> f'Hi {name}'
'Hi Rick' 
```

如您所见，使用 f 字符串，变量会自动从作用域中获取。这基本上是一个简写形式：

```py
>>> 'Hi {name}'.format(**globals())
'Hi Rick' 
```

### 任意表达式

任意表达式是 f 字符串真正强大之处。f 字符串的功能远超 printf 风格的字符串插值。f 字符串还支持完整的 Python 表达式，这意味着它们支持复杂对象、调用方法、`if`语句，甚至循环：

```py
## Accessing dict items
>>> username = 'wolph'
>>> a = 123
>>> b = 456
>>> some_dict = dict(a=a, b=b)

>>> f'''a: {some_dict['a']}'''
'a: 123'

>>> f'''sum: {some_dict['a'] + some_dict['b']}'''
'sum: 579'

## Python expressions, specifically an inline if statement
>>> f'if statement: {a if a > b else b}'
'if statement: 456'

## Function calls
>>> f'min: {min(a, b)}'
'min: 123'

>>> f'Hi {username}. And in uppercase: {username.upper()}'
'Hi wolph. And in uppercase: WOLPH'

## Loops
>>> f'Squares: {[x ** 2 for x in range(5)]}'
'Squares: [0, 1, 4, 9, 16]' 
```

## PEP 20，Python 的禅意

*Python 的禅意*，如前文*Python 简史*部分所述，是关于不仅能够工作，而且具有 Python 风格的代码。Python 风格的代码是可读的、简洁的且易于维护。PEP 20 说得最好：

> “长期 Python 程序员 Tim Peters 简洁地传达了 BDFL（Python 之父）为 Python 设计所制定的指导原则，仅用 20 条格言，其中只有 19 条被记录下来。”

接下来的几段将用一些示例代码解释这 19 条格言的意图。

为了清晰起见，让我们在开始之前看看这些格言：

```py
>>> import this
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those! 
```

### 美比丑更好

美是主观的，当然，但仍然有一些风格规则是值得遵守的。例如（来自 PEP 8）的规则：

+   使用空格而不是制表符缩进

+   行长度限制

+   每个语句都在单独的一行

+   每个导入都在单独的一行

当有疑问时，始终牢记一致性比固定规则更重要。如果一个项目更喜欢使用制表符而不是空格，或者反之，最好是保持这种制表符/空格，而不是通过替换制表符/空格来可能破坏现有的代码（和版本控制历史）。

简而言之，与其使用这种难以阅读的代码，它显示了 10 以下的奇数：

```py
>>> filter_modulo = lambda i, m: (i[j] for j in \
...                               range(len(i)) if i[j] % m)
>>> list(filter_modulo(range(10), 2))
[1, 3, 5, 7, 9] 
```

我更愿意：

```py
>>> def filter_modulo(items, modulo):
...     for item in items:
...         if item % modulo:
...             yield item
...

>>> list(filter_modulo(range(10), 2))
[1, 3, 5, 7, 9] 
```

它更简单，更容易阅读，而且稍微更美观！

这些例子是生成器的早期介绍。生成器将在*第七章*，*生成器和协程——一次一步的无限*中更详细地讨论。

### 明确优于隐晦

导入、参数和变量名只是许多情况下显式代码更容易阅读的例子，代价是编写代码时需要更多的努力和/或冗长。

这里有一个例子说明这可能会出错：

```py
>>> from os import *
>>> from asyncio import *

>>> assert wait 
```

在这种情况下，`wait`从哪里来？你可能会说这很明显——它来自`os`。但有时你会犯错。在 Windows 上，`os`模块没有`wait`函数，所以应该是`asyncio.wait`。

情况可能更糟：许多编辑器和代码清理工具都有排序导入功能。如果你的导入顺序发生变化，你的项目行为也会发生变化。

立即的修复方法很简单：

```py
>>> from os import path
>>> from asyncio import wait

>>> assert wait 
```

使用这种方法，我们至少有了一种找出`wait`从何而来途径。但我建议更进一步，通过模块导入，这样执行代码立即显示哪个函数被调用：

```py
>>> import os
>>> import asyncio

>>> assert asyncio.wait
>>> assert os.path 
```

对于 `*args` 和 `**kwargs` 也是如此。虽然它们非常有用，但它们可能会使你的函数和类的使用变得不那么明显：

```py
>>> def spam(eggs, *args, **kwargs):
...     for arg in args:
...         eggs += arg
...     for extra_egg in kwargs.get('extra_eggs', []):
...         eggs += extra_egg
...     return eggs

>>> spam(1, 2, 3, extra_eggs=[4, 5])
15 
```

不看函数内部的代码，你无法知道应该传递什么作为 `**kwargs` 或 `*args` 做了什么。当然，一个合理的函数名在这里能有所帮助：

```py
>>> def sum_ints(*args):
...     total = 0
...     for arg in args:
...         total += arg
...     return total

>>> sum_ints(1, 2, 3, 4, 5)
15 
```

对于这些情况，文档显然是有帮助的，我经常使用 `*args` 和 `**kwargs`，但确实是一个好主意，至少让最常见的参数明确。即使这要求你为父类重复参数，这也使得代码更加清晰。在将来重构父类时，你会知道是否有子类仍在使用某些参数。

### 简单优于复杂

> “简单优于复杂。复杂优于复杂化。”

保持事物简单往往比你想象的要困难得多。复杂性有逐渐渗透的趋势。你从一个美丽的脚本开始，然后不知不觉中，特性膨胀使其变得复杂（或者更糟，复杂化）。

```py
>>> import math
>>> import itertools

>>> def primes_complicated():
...     sieved = dict()
...     i = 2
...     
...     while True:
...         if i not in sieved:
...             yield i
...             sieved[i * i] = [i]
...         else:
...             for j in sieved[i]:
...                 sieved.setdefault(i + j, []).append(j)
...             del sieved[i]
...         
...         i += 1

>>> list(itertools.islice(primes_complicated(), 10))
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29] 
```

初看，这段代码可能看起来有些困难。然而，如果你熟悉欧几里得筛法，你会很快意识到正在发生什么。只需一点努力，你就会发现这个算法并不复杂，但使用了一些技巧来减少必要的计算。

我们可以做得更好；让我们看看一个不同的例子，它展示了 Python 3.8 的赋值运算符：

```py
>>> def primes_complex():
...     numbers = itertools.count(2)
...     while True:
...         yield (prime := next(numbers))
...         numbers = filter(prime.__rmod__, numbers)

>>> list(itertools.islice(primes_complex(), 10))
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29] 
```

这个算法看起来不那么令人畏惧，但我不认为它一开始就那么明显。`prime := next(numbers)` 是 Python 3.8 中在同一个语句中设置变量并立即返回它的版本。`prime.__rmod__` 使用给定的数字进行取模操作，类似于之前的例子。

然而，可能令人困惑的是，`numbers` 变量在每个迭代中都被重新分配，并添加了过滤器。让我们看看更好的解决方案：

```py
>>> def is_prime(number):
...     if number == 0 or number == 1:
...         return False
...     for modulo in range(2, number):
...         if not number % modulo:
...             return False
...     else:
...         return True

>>> def primes_simple():
...     for i in itertools.count():
...         if is_prime(i):
...             yield i

>>> list(itertools.islice(primes_simple(), 10))
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29] 
```

现在我们来到了生成素数的最明显的方法之一。`is_prime` 函数非常简单，立即显示了 `is_prime` 正在做什么。而 `primes_simple` 函数不过是一个带有过滤器的循环。

除非你真的有充分的理由选择复杂的方法，否则尽量让你的代码尽可能简单。当你未来阅读代码时，你（也许还有其他人）会为此感到感激。

### 平铺优于嵌套

嵌套的代码很快就会变得难以阅读和理解。这里没有严格的规则，但一般来说，当你有多个嵌套循环级别时，就是时候重构了。

只需看看以下示例，它打印了一个二维矩阵列表。虽然这里没有什么具体的问题，但将其拆分成更多函数可能会更容易理解其目的，也更容易进行测试：

```py
>>> def between_and_modulo(value, a, b, modulo):
...     if value >= a:
...         if value <= b:
...             if value % modulo:
...                 return True
...     return False

>>> for i in range(10):
...     if between_and_modulo(i, 2, 9, 2):
...         print(i, end=' ')
3 5 7 9 
```

这是更简洁的版本：

```py
>>> def between_and_modulo(value, a, b, modulo):
...     if value < a:
...         return False
...     elif value > b:
...         return False
...     elif not value % modulo:
...         return False
...     else:
...         return True

>>> for i in range(10):
...     if between_and_modulo(i, 2, 9, 2):
...         print(i, end=' ')
3 5 7 9 
```

这个例子可能有点牵强，但想法是合理的。深层嵌套的代码很容易变得难以阅读，将代码拆分成多行甚至函数可以大大提高可读性。

### 稀疏比密集好

空白通常是一件好事。是的，它会使你的文件更长，你的代码会占用更多空间，但如果你的代码逻辑上拆分得好，它可以帮助提高可读性。让我们举一个例子：

```py
>>> f=lambda x:0**x or x*f(x-1)
>>> f(40)
815915283247897734345611269596115894272000000000 
```

通过查看输出和代码，你可能能够猜出这是一个阶乘函数。但它的运作原理可能并不立即明显。让我们尝试重新编写：

```py
>>> def factorial(x):
...     if 0 ** x:
...         return 1
...     else:
...         return x * factorial(x - 1)

>>> factorial(40)
815915283247897734345611269596115894272000000000 
```

通过使用合适的名称，扩展`if`语句，并明确返回`1`，它突然变得非常明显发生了什么。

### 可读性很重要

短不一定意味着更容易阅读。让我们以斐波那契数为例。有很多人写这段代码的方法，其中许多很难阅读：

```py
>>> from functools import reduce

>>> fib=lambda n:n if n<2 else fib(n-1)+fib(n-2)
>>> fib(10)
55

>>> fib=lambda n:reduce(lambda x,y:(x[0]+x[1],x[0]),[(1,1)]*(n-1))[0]
>>> fib(10)
55 
```

尽管解决方案中存在一种美和优雅，但它们并不易读。只需进行一些小的修改，我们就可以将这些函数改为更易读的函数，其功能相似：

```py
>>> def fib(n):
...     if n < 2:
...         return n
...     else:
...         return fib(n - 1) + fib(n - 2)

>>> fib(10)
55

>>> def fib(n):
...     a = 0
...     b = 1
...     for _ in range(n):
...         a, b = b, a + b
...
...     return a

>>> fib(10)
55 
```

### 实用性胜过纯粹性

> “特殊情况并不足以打破规则。尽管实用性胜过纯粹性。”

有时打破规则可能很有吸引力，但这是一个滑稽的斜坡。如果你的快速修复会打破规则，你真的应该立即尝试重构它。很可能你以后没有时间修复它，并且会后悔。

虽然没有必要做得太过分。如果解决方案足够好，重构会花费更多的工作，那么选择工作方法可能更好。尽管所有这些例子都涉及导入，但这个指南几乎适用于所有情况。

为了防止长行，可以通过使用几种方法来缩短导入，添加反斜杠，添加括号，或者只是缩短导入。我将在下面展示一些选项：

```py
>>> from concurrent.futures import ProcessPoolExecutor, \
...     CancelledError, TimeoutError 
```

这个情况可以通过使用括号轻松避免：

```py
>>> from concurrent.futures import (
...     ProcessPoolExecutor, CancelledError, TimeoutError) 
```

或者我个人的偏好，导入模块而不是单独的对象：

```py
>>> from concurrent import futures 
```

但关于真正长的导入呢？

```py
>>> from concurrent.futures.process import \
...     ProcessPoolExecutor 
```

在那种情况下，我建议使用括号。如果你需要将导入拆分到多行，我建议每行一个导入以提高可读性：

```py
>>> from concurrent.futures.process import (
...     ProcessPoolExecutor
... )

>>> from concurrent.futures import (
...     ProcessPoolExecutor,
...     CancelledError,
...     TimeoutError,
... ) 
```

### 错误绝不应该默默通过

> “错误绝不应该默默通过。除非明确地被压制。”

正确处理错误真的很困难，没有一种方法适用于所有情况。然而，有一些方法比其他方法更好或更差来捕获错误。

裸露或过于宽泛的异常捕获可能会在出现错误时使你的生活变得有些困难。完全不传递异常信息可能会让你（或正在编写代码的其他人）长时间对发生的事情感到困惑。

为了说明一个裸露的异常，最糟糕的选择如下：

```py
>>> some_user_input = '123abc'

>>> try:
...     value = int(some_user_input)
... except:
...     pass 
```

一个更好的解决方案是明确捕获你需要的错误：

```py
>>> some_user_input = '123abc'

>>> try:
...     value = int(some_user_input)
... except ValueError:
...     pass 
```

或者，如果你真的需要捕获所有异常，请确保正确地记录它们：

```py
>>> import logging

>>> some_user_input = '123abc'

>>> try:
...     value = int(some_user_input)
... except Exception as exception:
...     logging.exception('Uncaught: {exception!r}') 
```

当在`try`块中使用多行时，由于有更多的代码可能负责隐藏的异常，跟踪错误的难题进一步加剧。当`except`意外地捕获了几层深处的函数的异常时，跟踪错误也变得更加困难。例如，考虑以下代码块：

```py
>>> some_user_input_a = '123'
>>> some_user_input_b = 'abc'

>>> try:
...     value = int(some_user_input_a)
...     value += int(some_user_input_b)
... except:
...     value = 0 
```

如果抛出了异常，是哪一行引起的？在没有运行调试器的情况下，通过静默捕获错误，你无法知道。如果，而不是使用`int()`，你使用了一个更复杂的函数，异常甚至可能是在代码的几层深的地方引起的。

如果你在一个特定的代码块中测试特定的异常，更安全的方法是在`try`/`except`中使用`else`。`else`只有在没有异常的情况下才会执行。

为了说明`try`/`except:`的全部威力，这里是一个包括`else`、`finally`和`BaseException`的所有变体的例子：

```py
>>> try:
...     1 / 0  # Raises ZeroDivisionError
... except ZeroDivisionError:
...     print('Got zero division error')
... except Exception as exception:
...     print(f'Got unexpected exception: {exception}')
... except BaseException as exception:
...     # Base exceptions are a special case for keyboard
...     # interrupts and a few other exceptions that are not
...     # technically errors.
...     print(f'Got base exception: {exception}')
... else:
...     print('No exceptions happened, we can continue')
... finally:
...     # Useful cleanup functions such as closing a file
...     print('This code is _always_ executed')
Got zero division error
This code is _always_ executed 
```

### 面对歧义，拒绝猜测的诱惑

尽管猜测在很多情况下都会有效，但如果你不小心，它们可能会给你带来麻烦。正如在*明确优于隐含*部分所展示的，当你有少量`from ... import *`时，你无法总是确定哪个模块为你提供了你期望的变量。

清晰且无歧义的代码会产生更少的错误，因此始终考虑当别人阅读你的代码时会发生什么总是一个好主意。一个歧义性的主要例子是函数调用。例如，以下两个函数调用：

```py
>>> fh_a = open('spam', 'w', -1, None, None, '\n')
>>> fh_b = open(file='spam', mode='w', buffering=-1, newline='\n') 
```

这两个调用具有完全相同的结果。然而，在第二个调用中，很明显`-1`正在配置缓冲区。你可能对`open()`的前两个参数了如指掌，但其他参数则不太常见。

无论怎样，没有看到`help(open)`或以其他方式查看文档，你无法说这两个是否相同。

注意，我认为你不必在所有情况下都使用关键字参数，但如果涉及许多参数和/或难以识别的参数（例如数字），这可能是一个好主意。一个好的替代方案是使用好的变量名，这可以使函数调用更加明显：

```py
>>> filename = 'spam'
>>> mode = 'w'
>>> buffers = -1

>>> fh_b = open(filename, mode, buffers, newline='\n') 
```

### 做这件事的一个明显方法

> “应该有一个——最好是只有一个——明显的做法。虽然这个方法可能一开始并不明显，除非你是荷兰人。”

通常情况下，思考一段时间困难的问题后，你会发现有一个解决方案明显优于其他替代方案。然而，有时情况并非如此，在这种情况下，如果你是荷兰人，这可能是有用的。这里的笑话是，Python 的原始作者 Guido van Rossum 是荷兰人（我也是），而且在某些情况下，只有 Guido 知道明显的做法。

另一个笑话是 Perl 编程语言的口号正好相反：“有多种方法可以做到。”

现在比永远不做好

> “现在比永远不做好。尽管永远通常比 *现在* 做好。”

立即解决问题比将其推迟到未来更好。然而，在某些情况下，立即解决问题并不是一个选择。在这种情况下，一个好的替代方案是将函数标记为已弃用，这样就没有忘记问题的风险：

```py
>>> import warnings

>>> warnings.warn('Something deprecated', DeprecationWarning) 
```

### 难以解释，容易解释

> “如果实现难以解释，那是个坏主意。如果实现容易解释，那可能是个好主意。”

总是保持尽可能简单。虽然复杂的代码可以很好地进行测试，但它更容易出现错误。你越能保持简单，就越好。

### 命名空间是一个非常好的想法

> “命名空间是一个非常好的想法——让我们做更多这样的！”

命名空间可以使代码更易于使用。正确命名它们会使它变得更好。例如，假设在更大的文件中 `import` 没有显示在你的屏幕上。`loads` 行做什么？

```py
>>> from json import loads

>>> loads('{}')
{} 
```

现在让我们看看带有命名空间版本的示例：

```py
>>> import json

>>> json.loads('{}')
{} 
```

现在很明显 `loads()` 是 `json` 加载器，而不是任何其他类型的加载器。

命名空间快捷方式仍然很有用。让我们看看 Django 中的 `User` 类，它在几乎每个 Django 项目中都被使用。默认情况下，`User` 类存储在 `django.contrib.auth.models.User` 中（可以被覆盖）。许多项目以以下方式使用该对象：

```py
from django.contrib.auth.models import User
# Use it as: User 
```

虽然这相当清晰，但项目可能会使用多个名为 `User` 的类，这会模糊导入。此外，这也可能让人误以为 `User` 类是当前类的本地类。通过以下方式做可以让人知道它位于不同的模块中：

```py
from django.contrib.auth import models
# Use it as: models.User 
```

然而，这很快就会与其他模型的导入发生冲突，所以我个人使用以下方法代替：

```py
from django.contrib.auth import models as auth_models
# Use it as auth_models.User 
```

或者更简短的说法：

```py
import django.contrib.auth.models as auth_models
# Use it as auth_models.User 
```

现在你应该对 Python 主义有所了解——创建以下代码：

+   美观

+   易读

+   清晰无误

+   足够明确

+   不是完全没有空格

那么，让我们继续看看如何使用 Python 风格指南创建美观、易读和简单的代码的一些更多示例。

## 解释 PEP 8

前面的部分已经展示了使用 PEP 20 作为参考的许多示例，但还有一些其他重要的指南需要注意。PEP 8 风格指南指定了标准的 Python 编码约定。

仅遵循 PEP 8 标准并不能使你的代码具有 Python 风格，但它确实是一个很好的开始。你使用哪种风格并不是那么重要，只要你保持一致。最糟糕的事情不是使用合适的风格指南，而是对其不一致。

### Duck typing

Duck typing 是一种通过行为处理变量的方法。引用 Alex Martelli（我的 Python 英雄之一，也被许多人昵称为 MartelliBot）的话：

> “不要检查它是否是一只鸭子：检查它是否像鸭子一样嘎嘎叫，像鸭子一样走路，等等，具体取决于你需要用鸭子行为的一个子集来玩你的语言游戏。如果这个参数没有通过这个特定的鸭子属性子集测试，那么你可以耸耸肩，问‘为什么是一只鸭子？’”

在许多情况下，当人们进行`if spam != ''`这样的比较时，他们实际上只是在寻找任何被认为是真值的对象。虽然你可以将值与字符串值`''`进行比较，但你通常不必做得如此具体。在许多情况下，简单地做`if spam:`就足够了，而且实际上效果更好。

例如，以下代码行使用`timestamp`的值来生成一个文件名：

```py
>>> timestamp = 12345

>>> filename = f'{timestamp}.csv' 
```

因为变量命名为`timestamp`，你可能会想检查它实际上是否是一个`date`或`datetime`对象，如下所示：

```py
>>> import datetime

>>> timestamp = 12345

>>> if isinstance(timestamp, datetime.datetime):
...     filename = f'{timestamp}.csv'
... else:
...     raise TypeError(f'{timestamp} is not a valid datetime')
Traceback (most recent call last):
...
TypeError: 12345 is not a valid datetime 
```

虽然这本身并没有错，但在 Python 中，比较类型被认为是一种不好的做法，因为通常没有必要这么做。

在 Python 中，常用的风格是**EAFP**（**求原谅比求许可更容易**：[`docs.python.org/3/glossary.html#term-eafp`](https://docs.python.org/3/glossary.html#term-eafp)），它假设不会出错，但在需要时可以捕获错误。在 Python 解释器中，如果没有抛出异常，`try`/`except`块非常高效。然而，实际捕获异常是昂贵的，因此这种方法主要推荐在你不期望`try`经常失败的情况下使用。

EAFP（**先做后检查**：[`docs.python.org/3/glossary.html#term-lbyl`](https://docs.python.org/3/glossary.html#term-lbyl)）的相反做法是**LBYL**（**跳之前先看**），在执行其他调用或查找之前检查先决条件。这种方法的一个显著缺点是在多线程环境中可能存在竞争条件。当你正在检查字典中键的存在时，另一个线程可能已经将其移除了。

这就是为什么在 Python 中，鸭子类型通常更受欢迎。只需测试变量是否具有你需要的特性，而不用担心实际的类型。为了说明这可能会对最终结果产生多小的差异，请看以下代码：

```py
>>> import datetime

>>> timestamp = datetime.date(2000, 10, 5)
>>> filename = f'{timestamp}.csv'
>>> print(f'Filename from date: {filename}')
Filename from date: 2000-10-05.csv 
```

与字符串而不是日期进行比较：

```py
>>> timestamp = '2000-10-05'
>>> filename = f'{timestamp}.csv'
>>> print(f'Filename from str: {filename}')
Filename from str: 2000-10-05.csv 
```

正如你所见，结果是相同的。

同样，将数字转换为浮点数或整数也是如此；而不是强制执行某种类型，只需要求某些特性。需要能通过作为数字的？只需尝试将其转换为`int`或`float`。需要一个`file`对象？为什么不检查是否有`read`方法呢，使用`hasattr`？

### 值比较和身份比较之间的差异

Python 中有许多比较对象的方法：大于、位运算符、等于等，但有一个比较器是特殊的：身份比较操作符。你不会使用`if spam == eggs`，而是使用`if spam is eggs`。第一个比较值，第二个比较身份或**内存地址**。因为它只比较内存地址，所以它是你可以得到的轻量级和严格的查找之一。而值检查需要确保类型是可比较的，可能还需要检查子值，而身份检查只是检查唯一标识符是否相同。

如果你曾经编写过 Java，你应该熟悉这个原则。在 Java 中，常规的字符串比较（`spam == eggs`）将使用身份而不是值。要比较值，你需要使用`spam.equals(eggs)`来获得正确的结果。

这些比较建议在期望对象身份保持不变时使用。一个明显的例子是与`True`、`False`或`None`的比较。为了演示这种行为，让我们看看在按值比较时评估为`True`或`False`的值，但实际上是不同的：

```py
>>> a = 1
>>> a == True
True
>>> a is True
False

>>> b = 0
>>> b == False
True
>>> b is False
False 
```

类似地，你需要小心处理`if`语句和`None`值，这是默认函数参数的一个常见模式：

```py
>>> def some_unsafe_function(arg=None):
...     if not arg:
...         arg = 123
...
...     return arg

>>> some_unsafe_function(0)
123
>>> some_unsafe_function(None)
123 
```

第二个确实需要默认参数，但第一个有一个实际应该使用的值：

```py
>>> def some_safe_function(arg=None):
...     if arg is None:
...         arg = 123
...
...     return arg

>>> some_safe_function(0)
0
>>> some_safe_function(None)
123 
```

现在我们实际上得到了我们传递的值，因为我们使用了身份而不是值检查`arg`。

尽管身份有一些陷阱。让我们看看一个没有意义的例子：

```py
>>> a = 200 + 56
>>> b = 256
>>> c = 200 + 57
>>> d = 257

>>> a == b
True
>>> a is b
True
>>> c == d
True
>>> c is d
False 
```

虽然值相同，但身份不同。问题是 Python 保留了一个整数对象的内部数组，用于所有介于`-5`和`256`之间的整数；这就是为什么它对`256`有效，但对`257`无效。

要查看 Python 实际上使用`is`操作符内部做了什么，你可以使用`id`函数。当执行`if spam is eggs`时，Python 将执行相当于`if id(spam) == id(eggs)`的操作，而`id()`（至少对于 CPython）返回内存地址。

### 循环

来自其他语言的人可能会倾向于使用带有计数器的`for`循环或`while`循环来处理`list`、`tuple`、`str`等项。虽然这是有效的，但它比所需的更复杂。例如，考虑以下代码：

```py
>>> my_range = range(5)
>>> i = 0
>>> while i < len(my_range ):
...     item = my_range [i]
...     print(i, item, end=', ')
...     i += 1
0 0, 1 1, 2 2, 3 3, 4 4, 
```

在 Python 中，没有必要构建自定义循环：你可以简单地迭代可迭代对象。尽管包括计数器的枚举也是容易实现的：

```py
>>> my_range  = range(5)
>>> for item in my_range :
...     print(item, end=', ')
0, 1, 2, 3, 4,

>>> for i, item in enumerate(my_range ):
...     print(i, item, end=', ')
0 0, 1 1, 2 2, 3 3, 4 4, 
```

当然，这可以写得更短（尽管不是 100%相同，因为我们没有使用`print`），但我不建议在大多数情况下这样做，因为这会影响可读性：

```py
>>> my_range  = range(5)
>>> [(i, item) for i, item in enumerate(my_range)]
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)] 
```

最后一个选项可能对一些人来说很清楚，但对所有人来说可能不是。一个常见的建议是将`list`/`dict`/`set`推导式和`map`/`filter`语句的使用限制在整行可以容纳整个语句的情况下。

### 最大行长度

许多 Python 程序员认为 79 个字符的限制太严格，因此他们只是让行更长。虽然我不会为 79 个字符的具体数字辩护，但设定一个低限是一个好主意，这样你可以轻松地将多个编辑器并排打开。我经常有四个 Python 文件并排打开。如果行宽超过 79 个字符，那就无法适应了。

PEP 8 告诉我们，当行过长时应该使用反斜杠。虽然我同意反斜杠比长行更可取，但我仍然认为如果可能的话应该避免使用，因为它们在通过复制/粘贴和重新排列代码时很容易产生语法错误。以下是一个来自 PEP 8 的例子：

```py
with open('/path/to/some/file/you/want/to/read') as file_1, \
        open('/path/to/some/file/being/written', 'w') as file_2:
    file_2.write(file_1.read()) 
```

而不是使用反斜杠，我会通过引入额外的变量来重新格式化代码，这样所有行都容易阅读：

```py
filename_1 = '/path/to/some/file/you/want/to/read'
filename_2 = '/path/to/some/file/being/written'
with open(filename_1) as file_1, open(filename_2, 'w') as file_2:
    file_2.write(file_1.read()) 
```

或者在这个特定的文件名案例中，通过使用 `pathlib`：

```py
import pathlib
filename_1 = pathlib.Path('/path/to/some/file/you/want/to/read')
filename_2 = pathlib.Path('/path/to/some/file/being/written')
with filename_1.open() as file_1, filename_2.open('w') as file_2:
    file_2.write(file_1.read()) 
```

当然，这并不总是可行的选择，但保持代码简短和可读性是一个很好的考虑。实际上，这实际上为代码添加了更多信息。如果你使用的是 `filename_1` 而不是传达文件名目标的名称，那么你立即就能清楚地知道你试图做什么。

## 验证代码质量，pep8，pyflakes 等

在 Python 中有很多检查代码质量和风格的工具。选项从检查与 PEP 8 相关的规则的 `pycodestyle`（之前命名为 `pep8`）到捆绑了许多工具的 `flake8`，这些工具可以帮助重构代码并追踪看似正常工作的代码中的错误。

让我们更详细地探讨一下。

#### pycodestyle/pep8

`pycodestyle` 包（之前命名为 `pep8`）是开始时的默认代码风格检查器。`pycodestyle` 检查器试图验证许多在 PEP 8 中提出的规则，这些规则被认为是社区的标准。它并不检查 PEP 8 标准中的所有内容，但它已经走了很长的路，并且仍然定期更新以添加新的检查。`pycodestyle` 检查的一些最重要的内容如下：

+   缩进：虽然 Python 不会检查你使用多少空格来缩进，但这并不有助于提高代码的可读性

+   缺少空白，例如 `spam=123`

+   过多的空白，例如 `def eggs(spam = 123):`

+   空行过多或过少

+   行过长

+   语法和缩进错误

+   不正确和/或多余的比较（`not in`，`is not`，`if spam is True`，以及没有 `isinstance` 的类型比较）

如果某些特定规则不符合你的喜好，你可以轻松地调整它们以适应你的目的。除此之外，该工具并不太具有主观性，这使得它成为任何 Python 项目的理想起点。

值得一提的是 `black` 项目，这是一个 Python 格式化工具，可以自动将你的代码格式化为大量遵循 PEP 8 风格的代码。`black` 这个名字来源于亨利·福特的名言：“任何顾客都可以得到一辆任何颜色的车，只要它是黑色的。”

这立即显示了`black`的缺点：它在定制方面提供的非常有限。如果你不喜欢其中的某条规则，你很可能运气不佳。

#### pyflakes

`pyflakes`检查器通过解析（而不是导入）代码来检测你代码中的错误和潜在的错误。这使得它非常适合与编辑器集成，但它也可以用来警告你代码中可能存在的问题。它将警告你关于以下内容：

+   未使用的导入

+   通配符导入（`from module import *`）

+   不正确的`__future__`导入（在其他导入之后）

更重要的是，它警告你关于潜在的错误，例如以下内容：

+   重新定义已导入的名称

+   使用未定义的变量

+   在赋值之前引用变量

+   重复的参数名称

+   未使用的局部变量

#### pep8-naming

PEP 8 的最后一点由`pep8-naming`包处理。它确保你的命名接近 PEP 8 指定的标准：

+   类名使用`CapWord`

+   函数、变量和参数名称全部小写

+   常量使用全大写并被视为常量

+   实例方法和类方法的第一参数分别为`self`和`cls`

#### McCabe

最后，是 McCabe 复杂度。它通过查看 Python 从源代码内部构建的**抽象语法树（AST**）来检查代码的复杂度。它找出有多少行、级别和语句，并在你的代码复杂度超过预配置的阈值时发出警告。通常，你会通过`flake8`使用 McCabe，但也可以手动调用。使用以下代码：

```py
def noop():
    pass

def yield_cube_points(matrix):
    for x in matrix:
        for y in x:
            for z in y:
                yield (x, y, z)

def print_cube(matrix):
    for x in matrix:
        for y in x:
            for z in y:
                print(z, end='')
            print()
        print() 
```

McCabe 将给出以下输出：

```py
$ pip3 install mccabe
...
$ python3 -m mccabe T_16_mccabe.py
1:0: 'noop' 1
5:0: 'yield_cube_points' 4
12:0: 'print_cube' 4 
```

起初，当你看到`noop`生成的`1`时，你可能会认为`mccabe`计算的是代码行数。经过进一步检查，你可以看到这并不是事实。有多个`noop`操作符不会增加计数，`print_cube`函数中的`print`语句也不会增加计数。

`mccabe`工具检查代码的循环复杂度。简而言之，这意味着它计算可能的执行路径数量。没有任何控制流语句（如`if`/`for`/`while`）的代码计为 1，正如你在`noop`函数中看到的那样。一个简单的`if`或`if`/`else`会产生两个选项：一个`if`语句为`True`的情况和一个`if`语句为`False`的情况。如果有嵌套的`if`或`elif`，这将进一步增加。循环计为 2，因为有项目时进入循环的流程，没有项目时不进入循环。

`mccabe`的默认警告阈值设置为 10，但可以配置。如果你的代码实际得分超过 10，那么是时候进行一些重构了。记住 PEP 20 的建议。

#### Mypy

Mypy 是一种用于检查代码中变量类型的工具。虽然指定固定类型与鸭子类型相矛盾，但确实有一些情况下这很有用，并且可以保护你免受错误的影响。

以以下代码为例：

```py
some_number: int
some_number = 'test' 
```

`mypy`命令会告诉我们我们犯了一个错误：

```py
$ mypy T_17_mypy.py
T_17_mypy.py:2: error: Incompatible types in assignment (expression has type "str", variable has type "int")
Found 1 error in 1 file (checked 1 source file) 
```

注意，这个语法依赖于 Python 3.5 中引入的类型提示。对于较旧的 Python 版本，你可以使用注释来代替类型提示：

```py
some_number = 'test'  # type: int 
```

即使你不在自己的代码中使用代码提示，这也可以用来检查你的外部库调用是否正确。如果一个外部库的函数参数在更新中发生了变化，这可以快速告诉你错误位置有问题，而不是需要追踪整个代码中的错误。

#### flake8

要运行所有这些测试的组合，你可以使用默认运行`pycodestyle`、`pyflakes`和`mccabe`的`flake8`工具。运行这些命令后，`flake8`将它们的输出合并成单个报告。`flake8`生成的某些警告可能不符合你的口味，所以每个检查都可以禁用，无论是按文件还是按整个项目（如果需要的话）。例如，我个人为所有项目禁用了`W391`，它会警告你文件末尾有空白行。

这是我工作时发现很有用的一点，这样我就可以轻松地跳到文件末尾并开始编写代码，而不是先添加几行。

还有许多插件可以使`flake8`更加强大。

一些示例插件包括：

+   `pep8-naming`：测试 PEP 命名约定

+   `flake8-docstrings`：测试 docstrings 是否遵循 PEP 257、NumPy 或 Google 约定。关于这些约定的更多内容将在关于文档的章节中介绍。

+   `flake8-bugbear`：在代码中查找可能的错误和设计问题，例如裸露的 except。

+   `flake8-mypy`：测试值类型是否与声明的类型一致。

通常，在提交代码和/或将代码上线之前，只需从你的源目录中运行`flake8`来递归地检查一切。

这里有一些格式不佳的代码的演示：

```py
def spam(a,b,c):print(a,b+c)
def eggs():pass 
```

这会导致以下结果：

```py
$ pip3 install flake8
...
$ flake8 T_18_flake8.py
T_18_flake8.py:1:11: E231 missing whitespace after ','
T_18_flake8.py:1:13: E231 missing whitespace after ','
T_18_flake8.py:1:16: E231 missing whitespace after ':'
T_18_flake8.py:1:24: E231 missing whitespace after ','
T_18_flake8.py:2:11: E231 missing whitespace after ':' 
```

## Python 语法的最近添加

在过去十年中，Python 语法在很大程度上保持不变，但我们已经看到了一些添加，比如 f-strings、类型提示和异步函数，当然。我们已经在本章开头介绍了 f-strings，其他两个分别在*第九章*和*第十三章*中介绍，但还有一些其他最近添加到 Python 语法中的内容，你可能错过了。此外，在*第四章*中，你将看到 Python 3.9 中添加的字典合并操作符。

### PEP 572：赋值表达式/海象操作符

我们在本章前面已经简要地介绍过这一点，但自从 Python 3.8 版本以来，我们有了赋值表达式。如果你有 C 或 C++的经验，你很可能之前见过类似的东西：

```py
if((fh = fopen("filename.txt", "w")) == NULL) 
```

在 C 中，使用 `fopen()` 打开文件，将 `fopen()` 的结果存储在 `fh` 中，并检查 `fopen()` 调用的结果是否为 `NULL`。直到 Python 3.8，我们总是必须将这些两个操作分成一个赋值和一个 `if` 语句，假设我们的 Python 代码中也有 `fopen()` 和 `NULL`：

```py
fh = fopen("filename.txt", "w")
if fh == NULL: 
```

自从 Python 3.8 以来，我们可以使用赋值表达式在一行中完成这个操作，类似于 C：

```py
if (fh := fopen("filename.txt", "w")) == NULL: 
```

使用 := 运算符，您可以在一个操作中分配和检查结果。这在读取用户输入时非常有用，例如：

```py
 while (line := input('Please enter a line: ')) != '':
        # Process the line here
    # The last line was empty, continue the script 
```

这个运算符通常被称为海象运算符，因为它看起来有点像海象的眼睛和獠牙（:=）。

### PEP 634：结构化模式匹配，switch 语句

许多刚开始接触 Python 的程序员想知道为什么它不像大多数常见编程语言那样有 switch 语句。通常，switch 语句的缺失是通过字典查找或简单地使用一系列 `if`/`elif`/`elif`/`elif`/`else` 语句来解决的。虽然这些解决方案可以正常工作，但我个人觉得有时我的代码如果使用 switch 语句可能会更美观、更易读。

自从 Python 3.10 以来，我们终于拥有了一个与 switch 语句非常相似但功能更强大的特性。正如 Python 的三元运算符（即 `true_value if condition else false_value`）的情况一样，其语法与其它语言的直接复制相去甚远。在这种情况下，这反而更好。在大多数编程语言中，很容易忘记 switch 中的 `break` 语句，这可能会导致意外的副作用。

从表面上看，Python 的实现语法和功能似乎更简单。没有 `break` 语句，你可能会想知道如何一次性匹配多个模式。请耐心等待，我们将揭晓！模式匹配功能非常强大，并且提供了比您预期的更多功能。

#### 基本的匹配语句

首先，让我们看一个基本示例。这个例子提供的帮助不大，但仍然比常规的 `if`/`elif`/`else` 语句更容易阅读：

```py
>>> some_variable = 123

>>> match some_variable:
...     case 1:
...         print('Got 1')
...     case 2:
...         print('Got 2')
...     case _:
...         print('Got something else')
Got something else

>>> if some_variable == 1:
...     print('Got 1')
... elif some_variable == 1:
...     print('Got 2')
... else:
...     print('Got something else')
Got something else 
```

由于我们这里既有 `if` 语句也有 `match` 语句，您可以轻松地进行比较。在这种情况下，我会选择 `if` 语句，但不需要重复 `some_variable ==` 部分的主要优势仍然很有用。

`_` 是 match 语句的特殊通配符情况。它匹配任何值，因此它可以看作是 `else` 语句的等价物。

#### 将后备存储为变量

一个稍微更有用的例子是在不匹配时自动存储结果。前面的例子使用了下划线（`_`），实际上并没有存储在 `_` 中，因为它是一个特殊的情况，但如果我们给变量起不同的名字，我们就可以存储结果：

```py
>>> some_variable = 123

>>> match some_variable:
...     case 1:
...         print('Got 1')
...     case other:
...         print('Got something else:', other)
Got something else: 123 
```

在这种情况下，我们将 `else` 情况存储在 `other` 变量中。请注意，您不能同时使用 `_` 和变量名，因为它们做的是同一件事，这将是没有用的。

#### 从变量中进行匹配

你看到，例如 `case other:` 这样的情况会将结果存储在 `other` 中，而不是与 `other` 的值进行比较，所以你可能想知道我们是否可以做等效的操作：

```py
if some_variable == some_value: 
```

答案是我们可以，但有一个前提。由于任何裸露的 `case variable:` 都会导致将值存储到变量中，我们需要有某种不匹配该模式的东西。常见的绕过这种限制的方法是通过引入一个点：

```py
>>> class Direction:
...     LEFT = -1
...     RIGHT = 1

>>> some_variable = Direction.LEFT

>>> match some_variable:
...     case Direction.LEFT:
...         print('Going left')
...     case Direction.RIGHT:
...         print('Going right')
Going left 
```

只要它不能被解释为变量名，这对你来说就会起作用。当然，在比较局部变量时，也可以使用 `if` 语句。

#### 在单个情况中匹配多个值

如果你熟悉许多其他编程语言中的 `switch` 语句，你可能想知道在你 `break` 之前是否可以有多个 `case` 语句，例如（C++）：

```py
switch(variable){
    case Direction::LEFT:
    case Direction::RIGHT:
        cout << "Going horizontal" << endl;
        break;
    case Direction::UP:
    case Direction::DOWN:
        cout << "Going vertical" << endl;
} 
```

这大致意味着如果 `variable` 等于 `LEFT` 或 `RIGHT`，则打印 `"Going horizontal"` 行并 `break`。由于 Python 的 `match` 语句没有 `break`，我们如何匹配这样的内容？嗯，为了这个目的，引入了一些特定的语法：

```py
>>> class Direction:
...     LEFT = -1
...     UP = 0
...     RIGHT = 1
...     DOWN = 2

>>> some_variable = Direction.LEFT

>>> match some_variable:
...     case Direction.LEFT | Direction.RIGHT:
...         print('Going horizontal')
...     case Direction.UP | Direction.DOWN:
...         print('Going vertical')
Going horizontal 
```

正如你所见，使用 `|` 操作符（它也用于位运算），你可以同时测试多个值。

#### 使用 guards 或额外条件匹配值

有时候你想要更高级的比较，比如 `if variable > value:`。幸运的是，即使这样也可以通过使用带有称为 guards 的 `match` 语句来实现。

```py
>>> values = -1, 0, 1

>>> for value in values:
...     print('matching', value, end=': ')
...     match value:
...         case negative if negative < 0:
...             print(f'{negative} is smaller than 0')
...         case positive if positive > 0:
...             print(f'{positive} is greater than 0')
...         case _:
...             print('no match')
matching -1: -1 is smaller than 0
matching 0: no match
matching 1: 1 is greater than 0 
```

注意这使用了刚刚引入的变量名，但它是一个常规的 Python 正则表达式，所以你也可以比较其他内容。然而，你总是需要在 `if` 前面有变量名。这不会起作用：`case if ...`。

#### 匹配列表、元组和其它序列

如果你熟悉 `tuple` 解包，你可能可以猜出序列匹配是如何工作的：

```py
>>> values = (0, 1), (0, 2), (1, 2)

>>> for value in values:
...     print('matching', value, end=': ')
...     match value:
...         case 0, 1:
...             print('exactly matched 0, 1')
...         case 0, y:
...             print(f'matched 0, y with y: {y}')
...         case x, y:
...             print(f'matched x, y with x, y: {x}, {y}')
matching (0, 1): exactly matched 0, 1
matching (0, 2): matched 0, y with y: 2
matching (1, 2): matched x, y with x, y: 1, 2 
```

第一个情况明确匹配了给定的两个值，这等同于 `if value == (0, 1):`。

第二个情况明确匹配第一个值为 `0`，但将第二个值作为一个变量，并存储在 `y` 中。实际上这相当于 `if value[0] == 0: y = value[1]`。

最后一个情况为 `x` 和 `y` 值存储一个变量，并将匹配任何恰好有两个元素的序列。

#### 匹配序列模式

如果你认为之前的变量解包示例很有用，你将喜欢这一部分。`match` 语句的一个真正强大的功能是基于模式进行匹配。

假设我们有一个函数，它接受最多三个参数，`host`、`port` 和 `protocol`。对于 `port` 和 `protocol`，我们可以假设 `443` 和 `https`，这样只剩下 `hostname` 作为必需的参数。我们如何匹配这样，使得一个、两个、三个或更多的参数都得到支持并正确工作？让我们来看看：

```py
>>> def get_uri(*args):
...     # Set defaults so we only have to store changed variables
...     protocol, port, paths = 'https', 443, ()
...     match args:
...         case (hostname,):
...             pass
...         case (hostname, port):
...             pass
...         case (hostname, port, protocol, *paths):
...             pass
...         case _:
...             raise RuntimeError(f'Invalid arguments {args}')
...
...     path = '/'.join(paths)
...     return f'{protocol}://{hostname}:{port}/{path}'

>>> get_uri('localhost')
'https://localhost:443/'
>>> get_uri('localhost', 12345)
'https://localhost:12345/'
>>> get_uri('localhost', 80, 'http')
'http://localhost:80/'
>>> get_uri('localhost', 80, 'http', 'some', 'paths')
'http://localhost:80/some/paths' 
```

如你所见，`match` 语句还处理不同长度的序列，这是一个非常有用的工具。你当然也可以用 `if` 语句来做这件事，但我从未找到一种真正漂亮的方式来处理它。当然，你仍然可以将其与前面的示例结合起来，所以如果你想要调用特定的行为，你可以有一个 `case`，例如：`case (hostname, port, 'http'):`。你还可以使用 `*variable` 来捕获所有额外的变量。`*` 匹配序列中的 0 个或多个额外项。

#### 捕获子模式

除了指定一个变量名来保存所有值之外，你还可以存储显式的值匹配：

```py
>>> values = (0, 1), (0, 2), (1, 2)

>>> for value in values:
...     print('matching', value, end=': ')
...     match value:
...         case 0 as x, (1 | 2) as y:
...             print(f'matched x, y with x, y: {x}, {y}')
...         case _:
...             print('no match')
matching (0, 1): matched x, y with x, y: 0, 1
matching (0, 2): matched x, y with x, y: 0, 2
matching (1, 2): no match 
```

在这种情况下，我们明确地将 `0` 作为 `value` 的第一部分进行匹配，将 `1` 或 `2` 作为 `value` 的第二部分进行匹配。并将这些分别存储在变量 `x` 和 `y` 中。

这里需要注意的是，在 `case` 语句的上下文中，`|` 运算符始终按或操作符对 `case` 起作用，而不是按位或操作符对变量/值。通常 `1 | 2` 会得到 `3`，因为在二进制中 `1 = 0001`，`2 = 0010`，这两个数的组合是 `3 = 0011`。

#### 匹配字典和其他映射

自然地，也可以通过键来匹配映射（如 `dict`）：

```py
>>> values = dict(a=0, b=0), dict(a=0, b=1), dict(a=1, b=1)

>>> for value in values:
...     print('matching', value, end=': ')
...     match value:
...         case {'a': 0}:
...             print('matched a=0:', value)
...         case {'a': 0, 'b': 0}:
...             print('matched a=0, b=0:', value)
...         case _:
...             print('no match')
matching {'a': 0, 'b': 0}: matched a=0: {'a': 0, 'b': 0}
matching {'a': 0, 'b': 1}: matched a=0: {'a': 0, 'b': 1}
matching {'a': 1, 'b': 1}: no match 
```

注意，`match` 只检查给定的键和值，并不关心映射中的额外键。这就是为什么第一个案例匹配前两个项目。

正如前一个示例所示，匹配是按顺序发生的，并且它会在第一个匹配项处停止，而不是在最佳匹配项处停止。在这种情况下，第二个案例永远不会被触及。

#### 使用 isinstance 和属性进行匹配

如果你认为之前的 `match` 语句示例很令人印象深刻，那么你准备好完全惊讶吧。`match` 语句可以匹配包括属性在内的实例的方式非常强大，并且可以非常实用。只需看看以下示例，并尝试理解正在发生的事情：

```py
>>> class Person:
...     def __init__(self, name):
...         self.name = name

>>> values = Person('Rick'), Person('Guido')

>>> for value in values:
...     match value:
...         case Person(name='Rick'):
...             print('I found Rick')
...         case Person(occupation='Programmer'):
...             print('I found a programmer')
...         case Person() as person:
...             print('I found a person:', person.name)
I found Rick
I found a person: Guido 
```

虽然我必须承认语法有点令人困惑，甚至可以说不够 Pythonic，但它非常实用，所以仍然有意义。

首先，我们将查看 `case Person() as person:`。我们首先讨论这个，因为在我们继续其他示例之前，理解这里发生的事情非常重要。这一行实际上与 `if isinstance(value, Person):` 相同，在这个点上它并没有真正实例化 `Person` 类，这有点令人困惑。

其次，`case Person(name='Rick')` 匹配实例类型 `Person`，并且要求实例具有名为 `name` 的属性，其值为 `Rick`。

最后，`case Person(occupation='Programmer')` 匹配 `value` 是一个 `Person` 实例，并且有一个名为 `occupation` 的属性，其值为 `Programmer`。由于该属性不存在，它默默地忽略了这个问题。

注意，这也适用于内置类型，并支持嵌套：

```py
>>> class Person:
...     def __init__(self, name):
...         self.name = name

>>> value = Person(123)
>>> match value:
...     case Person(name=str() as name):
...         print('Found person with str name:', name)
...     case Person(name=int() as name):
...         print('Found person with int name:', name)
Found person with int name: 123 
```

我们已经介绍了几个新模式匹配功能的工作示例，但你可能还会想到更多。由于所有部分都可以嵌套，可能性真的是无限的。这可能不是解决所有问题的完美方案，语法可能感觉有点奇怪，但它是一个非常强大的解决方案，我建议任何 Python 程序员都应牢记于心。

# 常见陷阱

Python 是一种旨在清晰和易于阅读的语言，没有任何歧义和意外行为。不幸的是，这些目标并不是在所有情况下都能实现的，这就是为什么 Python 确实有一些边缘情况，它可能会做与你预期不同的事情。

本节将向你展示在编写 Python 代码时可能会遇到的一些问题。

## 范围很重要！

在 Python 中，有些情况下你可能会没有使用你实际期望的作用域。一些例子是在声明类和函数参数时，但最令人烦恼的是意外尝试覆盖一个`global`变量。

### 全局变量

从全局作用域访问变量时，一个常见问题是设置变量使其成为局部变量，即使是在访问全局变量时。

这一点是有效的：

```py
>>> g = 1

>>> def print_global():
...     print(f'Value: {g}')

>>> print_global()
Value: 1 
```

但以下是不正确的：

```py
>>> g = 1

>>> def print_global():
...     g += 1
...     print(f'Value: {g}')

>>> print_global()
Traceback (most recent call last):
    ...
UnboundLocalError: local variable 'g' referenced before assignment 
```

问题在于`g += 1`实际上翻译为`g = g + 1`，任何包含`g =`的操作都会使变量成为你作用域内的局部变量。由于在那个点正在分配局部变量，它还没有值，而你却在尝试使用它。

对于这些情况，有`global`语句，尽管通常建议完全避免写入`global`变量，因为这可能会在调试时使你的生活变得非常困难。现代编辑器可以大量帮助跟踪谁或什么正在写入你的`global`变量，但重构你的代码，使其明确地通过清晰路径传递和修改值，可以帮助你避免许多错误。

### 可变变量的引用传递

在 Python 中，变量是通过引用传递的。这意味着当你做类似`x = y`的操作时，`x`和`y`都将指向同一个变量。当你更改任一`x`或`y`的值（不是对象）时，另一个也会相应改变。

由于大多数变量类型，如字符串、整数、浮点数和元组是不可变的，所以这不是问题。执行`x = 123`不会影响`y`，因为我们没有改变`x`的值，而是用一个新的具有值`123`的对象替换了`x`。

然而，对于可变变量，我们可以改变对象的值。让我们说明这种行为以及如何绕过它：

```py
>>> x = []
>>> y = x
>>> z = x.copy()

>>> x.append('x')
>>> y.append('y')
>>> z.append('z')

>>> x
['x', 'y']
>>> y
['x', 'y']
>>> z
['z'] 
```

除非你明确地复制变量，就像我们用`z`做的那样，否则你的新变量将指向同一个对象。

现在，你可能想知道`copy()`是否总是有效。正如你可能猜到的，它并不总是有效。`copy()`函数只复制对象本身，而不是对象内的值。为此，我们有`deepcopy()`，它可以安全地处理递归：

```py
>>> import copy

>>> x = [[1], [2, 3]]
>>> y = x.copy()
>>> z = copy.deepcopy(x)

>>> x.append('a')
>>> x[0].append(x)

>>> x
[[1, [...]], [2, 3], 'a']
>>> y
[[1, [...]], [2, 3]]
>>> z
[[1], [2, 3]] 
```

### 可变函数默认参数

虽然可以轻松避免可变参数的问题，并在大多数情况下看到这些问题，但函数默认参数的情况就明显不那么明显了：

```py
>>> def append(list_=[], value='value'):
...    list_.append(value)
...    return list_

>>> append(value='a')
['a']
>>> append(value='b')
['a', 'b'] 
```

注意，这对于`dict`、`list`、`set`以及`collections`中的几种类型都适用。此外，你自己定义的类默认是可变的。

为了解决这个问题，你可以考虑将函数改为以下形式：

```py
>>> def append(list_=None, value='value'):
...    if list_ is None:
...        list_ = []
...    list_.append(value)
...    return list_

>>> append(value='a')
['a']
>>> append(value='b')
['b'] 
```

注意，我们在这里必须使用`if list_ is None`。如果我们使用`if not list_`，那么如果传递了一个空的`list`，它将忽略给定的`list_`。

### 类属性

可变变量的问题在定义类时也会出现。很容易混淆类属性和实例属性。这可能会让人感到困惑，尤其是当你来自像 C#这样的其他语言时。让我们通过以下示例来说明：

```py
>>> class SomeClass:
...     class_list = []
...
...     def __init__(self):
...         self.instance_list = []

>>> SomeClass.class_list.append('from class')
>>> instance = SomeClass()
>>> instance.class_list.append('from instance')
>>> instance.instance_list.append('from instance')

>>> SomeClass.class_list
['from class', 'from instance']
>>> SomeClass.instance_list
Traceback (most recent call last):
...
AttributeError: ... 'SomeClass' has no attribute 'instance_list'

>>> instance.class_list
['from class', 'from instance']
>>> instance.instance_list
['from instance'] 
```

就像函数参数一样，列表和字典是共享的。所以，如果你想为类定义一个不共享于所有实例的可变属性，你需要在`__init__`或其他任何实例方法中定义它。

在处理类时，还有另一个需要注意的重要事项，那就是类的属性将会被继承，这可能会让人感到困惑。在继承过程中，原始属性将保持对原始值的引用（除非被覆盖），即使在子类中也是如此：

```py
>>> class Parent:
...     pass

>>> class Child(Parent):
...     pass

>>> Parent.parent_property = 'parent'
>>> Child.parent_property
'parent'

>>> Child.parent_property = 'child'
>>> Parent.parent_property
'parent'
>>> Child.parent_property
'child'

>>> Child.child_property = 'child'
>>> Parent.child_property
Traceback (most recent call last):
...
AttributeError: ... 'Parent' has no attribute 'child_property' 
```

虽然由于继承这是可以预料的，但其他人使用这个类时可能不会预料到变量会在同时改变。毕竟，我们修改的是`Parent`，而不是`Child`。

有两种简单的方法可以防止这种情况。显然，你可以简单地为每个类分别设置属性。但更好的解决方案是从不修改类属性，除非在类定义之外。很容易忘记属性将在多个位置改变，而且如果它必须可修改，通常最好将其放在实例变量中。

## 覆盖和/或创建额外的内建函数

虽然在某些情况下可能很有用，但通常你想要避免覆盖全局函数。PEP 8 的函数命名约定——类似于内建语句、函数和变量——是使用尾随下划线。

所以，不要这样做：

```py
list = [1, 2, 3] 
```

相反，使用以下方法：

```py
list_ = [1, 2, 3] 
```

对于列表等，这只是一个好的约定。对于`from`、`import`和`with`等语句，这是必需的。忘记这一点可能会导致非常令人困惑的错误：

```py
>>> list = list((1, 2, 3))
>>> list
[1, 2, 3]

>>> list((4, 5, 6))
Traceback (most recent call last):
    ...
TypeError: 'list' object is not callable

>>> import = 'Some import'
Traceback (most recent call last):
    ...
SyntaxError: invalid syntax 
```

如果你确实想定义一个在任何地方都可以使用而不需要导入的内建函数，这是可能的。为了调试目的，我在开发过程中曾将此代码添加到项目中：

```py
import builtins
import inspect
import pprint
import re

def pp(*args, **kwargs):
    '''PrettyPrint function that prints the variable name when
    available and pprints the data'''
    name = None
    # Fetch the current frame from the stack
    frame = inspect.currentframe().f_back
    # Prepare the frame info
    frame_info = inspect.getframeinfo(frame)

    # Walk through the lines of the function
    for line in frame_info[3]:
        # Search for the pp() function call with a fancy regexp
        m = re.search(r'\bpp\s*\(\s*([^)]*)\s*\)', line)
        if m:
            print('# %s:' % m.group(1), end=' ')
            break

    pprint.pprint(*args, **kwargs)

builtins.pf = pprint.pformat
builtins.pp = pp 
```

这段代码对于生产环境来说过于简陋，但在处理大型项目且需要打印语句进行调试时仍然很有用。替代（且更好的）调试解决方案可以在*第十一章*，*调试 – 解决错误*中找到。

使用方法相当简单：

```py
x = 10
pp(x) 
```

这里是输出：

```py
# x: 10 
```

## 在迭代时修改

在某个时刻，你将遇到这个问题：在迭代一些可变对象，如`dict`和`set`时，你不能修改它们。所有这些都会导致一个`RuntimeError`，告诉你不能在迭代过程中修改对象：

```py
>>> dict_ = dict(a=123)
>>> set_ = set((456,))

>>> for key in dict_:
...     del dict_[key]
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: dictionary changed size during iteration

>>> for item in set_:
...     set_.remove(item)
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Set changed size during iteration 
```

对于列表来说，这确实可行，但可能会导致非常奇怪的结果，因此绝对应该避免：

```py
>>> list_ = list(range(10))
>>> list_
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>> for item in list_:
...     print(list_.pop(0), end=', ')
0, 1, 2, 3, 4,

>>> list_
[5, 6, 7, 8, 9] 
```

虽然这些问题可以通过在使用前复制集合来避免，但在许多情况下，如果你遇到这个问题，那么你做的是错误的。如果确实需要操作，构建一个新的集合通常是更简单的方法，因为代码看起来会更明显。当未来有人查看这样的代码时，他们可能会尝试通过移除`list()`来重构它，因为乍一看这似乎是徒劳的：

```py
>>> list_ = list(range(10))

>>> for item in list(list_):
...     print(list_.pop(0), end=', ')
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
```

## 捕获和存储异常

在 Python 中捕获和存储异常时，你必须记住，出于性能原因，存储的异常是`except`块本地的。结果是，你需要显式地将异常存储在*不同的*变量中。在`try`/`except`块之前简单地声明变量是不起作用的，并且会使你的变量消失：

```py
>>> exception = None

>>> try:
...     1 / 0
... except ZeroDivisionError as exception:
...     pass

>>> exception
Traceback (most recent call last):
    ...
NameError: name 'exception' is not defined 
```

将结果存储在新变量中是有效的：

```py
>>> try:
...     1 / 0
... except ZeroDivisionError as exception:
...     new_exception = exception

>>> new_exception
ZeroDivisionError('division by zero') 
```

如你或许已经看到的，这段代码现在确实有一个错误。如果我们没有遇到异常，`new_exception`还没有被定义。我们可能需要给`try`/`except`添加一个`else`，或者，更好的做法是在`try`/`except`之前预先声明变量。

我们确实需要显式地保存它，因为 Python 3 在`except`语句结束时自动删除使用`as variable`保存的任何内容。原因在于 Python 3 中的异常包含一个`__traceback__`属性。拥有这个属性使得垃圾收集器更难检测到哪些内存应该被释放，因为它引入了递归自引用循环。

具体来说，这是 `exception -> traceback -> exception -> traceback ...` 的过程。

这确实意味着你应该记住，存储这些异常可能会将内存泄漏引入到你的程序中。

Python 的垃圾收集器足够智能，能够理解变量不再可见，并最终删除变量，但这可能需要更多的时间，因为这是一个更复杂的垃圾收集过程。垃圾收集是如何实际工作的，在*第十二章*，*性能 – 跟踪和减少你的内存和 CPU 使用*中有详细说明。

## 动态绑定和闭包

闭包是实现代码中局部作用域的一种方法。它们使得可以在局部定义变量而不覆盖父（或全局）作用域中的变量，并在之后将变量隐藏在外部作用域中。Python 中闭包的问题在于，Python 为了性能原因尽可能晚地绑定其变量。虽然通常很有用，但它确实有一些意外的副作用：

```py
>>> functions = [lambda: i for i in range(3)]

>>> for function in functions:
...     print(function(), end=', ')
2, 2, 2, 
```

你可能期望的是 `0, 1, 2`，但由于延迟绑定，所有函数都获取 `i` 的最后一个值，即 `2`。

我们应该怎么做呢？与前面段落中的情况一样，变量需要被本地化。一个选项是通过使用 `partial` 强制立即绑定函数：

```py
>>> from functools import partial

>>> functions = [partial(lambda x: x, i) for i in range(3)]

>>> for function in functions:
...     print(function(), end=', ')
0, 1, 2, 
```

更好的解决方案是避免绑定问题，不引入额外的作用域（如 `lambda`）使用外部变量。如果将 `i` 指定为 `lambda` 的参数，这就不会是问题。

## 循环导入

尽管 Python 对循环导入相当宽容，但仍然有一些情况下你会遇到错误。

假设我们有两个文件：

`T_28_circular_imports_a.py`:

```py
import T_28_circular_imports_b

class FileA:
    pass

class FileC(T_28_circular_imports_b.FileB):
    pass 
```

`T_28_circular_imports_b.py`:

```py
import T_28_circular_imports_a

class FileB(T_28_circular_imports_a.FileA):
    pass 
```

运行这些文件中的任何一个都会导致循环导入错误：

```py
Traceback (most recent call last):
  File "T_28_circular_imports_a.py", line 1, in <module>
    import T_28_circular_imports_b
  File "T_28_circular_imports_b.py", line 1, in <module>
    import T_28_circular_imports_a
  File "T_28_circular_imports_a.py", line 8, in <module>
    class FileC(T_28_circular_imports_b.FileB):
AttributeError: partially initialized module 'T_28_circular_imports_b' has no attribute 'FileB' (most likely due to a circular import) 
```

解决这个问题的方法有几个。最简单的解决方案是将 `import` 语句移动，使得循环导入不再发生。在这种情况下，`import T_28_circular_imports_a.py` 需要在 `FileA` 和 `FileB` 之间移动。

在大多数情况下，更好的解决方案是重构代码。将公共基类移动到单独的文件中，这样就不需要再进行循环导入了。对于上面的例子，它看起来可能像这样：

`T_29_circular_imports_a.py`:

```py
class FileA:
    pass 
```

`T_29_circular_imports_b.py`:

```py
import T_29_circular_imports_a

class FileB(T_29_circular_imports_a.FileA):
    pass 
```

`T_29_circular_imports_c.py`:

```py
import T_29_circular_imports_b

class FileC(T_29_circular_imports_b.FileB):
    pass 
```

如果这也行不通，可以在运行时而不是导入时从函数中导入，这可能很有用。当然，这对于类继承来说不是一个容易的选择，但如果你只需要在运行时导入，你可以推迟导入。

最后，还有动态导入的选项，例如 Django 框架用于 `ForeignKey` 字段的选项。除了实际的类之外，`ForeignKey` 字段还支持字符串，这些字符串在需要时将自动导入。

虽然这是一个非常有效的解决方法，但它确实意味着你的编辑器、linting 工具和其他工具不会理解你正在处理的对象。对这些工具来说，它看起来就像一个字符串，所以除非为这些工具添加特定的黑客技巧，否则它们不会假设值除了字符串之外的其他任何内容。

此外，由于 `import` 只在运行时发生，你只有在执行函数时才会注意到导入问题。这意味着那些通常会在你运行脚本或应用程序时立即出现的错误现在只有在调用函数时才会显示出来。这是一个很好的难以追踪的 bug 配方，它不会发生在你身上，但会发生在其他代码使用者身上。

这种模式对于插件系统等场景仍然很有用，但只要小心避免提到的注意事项。这里有一个简单的例子来动态导入：

```py
>>> import importlib

>>> module_name = 'sys'
>>> attribute = 'version_info'

>>> module = importlib.import_module(module_name)
>>> module
<module 'sys' (built-in)>
>>> getattr(module, attribute).major
3 
```

使用 `importlib`，动态导入模块相当容易，通过使用 `getattr`，你可以从模块中获取特定的对象。

## 导入冲突

一个可能非常令人困惑的问题是存在冲突的导入——多个包/模块具有相同的名称。我收到了不少关于这类情况的错误报告。

以我的`numpy-stl`项目为例，代码被放在一个名为`stl`的包中。许多人创建了一个名为`stl.py`的测试文件。当从`stl.py`导入`stl`时，它会导入自身而不是`stl`包。

此外，还存在包之间不兼容的问题。几个包可能会使用相同的名称，因此在安装一系列类似包时要小心，因为它们可能正在共享相同的名称。如果有疑问，只需创建一个新的虚拟环境并再次尝试。这样做可以节省你大量的调试时间。

# 摘要

本章向您展示了 Python 哲学是什么以及背后的某些推理。此外，您还了解了 Python 的 Zen 以及 Python 社区中认为美丽和丑陋的东西。虽然代码风格非常个人化，但 Python 有一些非常有用的指南，至少能让人保持大致相同的页面和风格。

最后，我们都是同意的成年人；每个人都有权按照自己的方式编写代码。但我确实请求您请阅读风格指南，并尽量遵守它们，除非您有非常好的理由不这样做。

权力越大，责任越大，尽管陷阱并不多。有些陷阱足够复杂，以至于我经常被它们愚弄，而且我已经写 Python 很长时间了！尽管如此，Python 一直在改进。自 Python 2 以来，已经解决了许多陷阱，但一些将始终存在。例如，循环导入和定义在大多数支持它们的语言中很容易让你上当，但这并不意味着我们会停止努力改进 Python。

Python 多年来改进的一个很好的例子是`collections`模块。它包含了许多用户添加的有用集合，因为存在这种需求。其中大部分实际上是用纯 Python 实现的，因此它们足够简单，任何人都可以阅读。理解它们可能需要更多的努力，但我真心相信，如果你能读到这本书的结尾，你将不会对集合的功能有任何问题。然而，我无法保证完全理解其内部工作方式；其中一些部分更多地涉及通用计算机科学而不是 Python 精通。

下一章将向您展示一些 Python 中可用的集合以及它们是如何在内部构建的。尽管你无疑熟悉列表和字典等集合，但你可能并不了解某些操作的性能特性。如果本章的一些示例不够清晰，你不必担心。下一章至少会回顾其中的一些，更多内容将在后面的章节中介绍。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：[`discord.gg/QMzJenHuJf`](https://discord.gg/QMzJenHuJf)

![](img/QR_Code156081100001293319171.png)
