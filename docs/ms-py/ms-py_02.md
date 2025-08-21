# 第二章：Pythonic 语法，常见陷阱和风格指南

Python 编程语言的设计和开发一直由其原始作者 Guido van Rossum 掌握，他常常被亲切地称为**终身仁慈独裁者**（**BDFL**）。尽管 van Rossum 被认为拥有一台时光机（他曾多次回答功能请求说“我昨晚刚实现了这个”：[`www.catb.org/jargon/html/G/Guido.html`](http://www.catb.org/jargon/html/G/Guido.html)），但他仍然只是一个人，需要帮助来维护和发展 Python。为了方便这一点，**Python Enhancement Proposal**（**PEP**）流程已经被开发出来。这个流程允许任何人提交一个带有功能技术规范和为其有用性辩护的理由的 PEP。在 Python 邮件列表上进行讨论并可能进行一些改进后，BDFL 将做出接受或拒绝提案的决定。

Python 风格指南（`PEP 8`：[`www.python.org/dev/peps/pep-0008/`](https://www.python.org/dev/peps/pep-0008/)）曾经作为其中一个 PEP 提交，自那以后它一直被接受和不断改进。它有很多伟大和广泛接受的惯例，也有一些有争议的。特别是，79 个字符的最大行长度是许多讨论的话题。然而，将一行限制在 79 个字符确实有一些优点。除此之外，虽然风格指南本身并不能使代码成为 Pythonic，正如“Python 之禅”（`PEP 20`：[`www.python.org/dev/peps/pep-0020/`](https://www.python.org/dev/peps/pep-0020/)）所说的那样：“美丽胜过丑陋。” `PEP 8`定义了代码应该以确切的方式进行格式化，而`PEP 20`更多的是一种哲学和心态。

常见的陷阱是一系列常见的错误，从初学者的错误到高级错误不等。它们范围广泛，从将列表或字典（可变的）作为参数传递到闭包中的延迟绑定问题。更重要的问题是如何以一种清晰的方式解决循环导入的问题。

本章中使用的一些技术可能对于这样一个早期的章节来说有点过于先进，但请不要担心。本章是关于风格和常见陷阱的。使用的技术的内部工作将在后面的章节中介绍。

我们将在本章中涵盖以下主题：

+   代码风格（`PEP 8`，`pyflakes`，`flake8`等）

+   常见陷阱（列表作为函数参数，按值传递与按引用传递，以及继承行为）

### 注意

Pythonic 代码的定义是非常主观的，主要反映了本作者的观点。在项目中工作时，与该项目的编码风格保持一致比遵循 Python 或本书给出的编码指南更重要。

# 代码风格 - 或者什么是 Pythonic 代码？

Pythonic code - 当你第一次听到它时，你可能会认为它是一种编程范式，类似于面向对象或函数式编程。虽然有些地方可以被认为是这样，但实际上它更多的是一种设计哲学。Python 让你可以自由选择以面向对象，过程式，函数式，面向方面甚至逻辑导向的方式进行编程。这些自由使 Python 成为一个很好的编程语言，但是，自由总是需要很多纪律来保持代码的清晰和可读性。`PEP8`标准告诉我们如何格式化代码，但 Pythonic 代码不仅仅是语法。这就是 Pythonic 哲学（`PEP20`）的全部内容，即代码应该是：

+   清晰

+   简单

+   美丽

+   显式

+   可读性

大多数听起来都像是常识，我认为它们应该是。然而，也有一些情况，没有一个明显的方法来做（除非你是荷兰人，当然，你将在本章后面读到）。这就是本章的目标 - 学习什么样的代码是美丽的，以及为什么在 Python 风格指南中做出了某些决定。

### 注意

有些程序员曾经问过 Guido van Rossum，Python 是否会支持大括号。从那天起，大括号就可以通过`__future__`导入使用了：

```py
>>> from __future__ import braces
 **File "<stdin>", line 1
SyntaxError: not a chance

```

## 格式化字符串 - `printf-style`还是`str.format`？

Python 长期以来一直支持`printf-style`（`%`）和`str.format`，所以你很可能已经对两者都很熟悉了。

在本书中，`printf-style`格式将被用于一些原因：

+   最重要的原因是这对我来说很自然。我已经在许多不同的编程语言中使用`printf`大约 20 年了。

+   大多数编程语言都支持`printf`语法，这使得它对很多人来说很熟悉。

+   尽管这只与本书中的示例有关，但它占用的空间稍微少一些，需要较少的格式更改。与显示器相反，书籍多年来并没有变得更宽。

一般来说，大多数人现在推荐使用`str.format`，但这主要取决于个人偏好。`printf-style`更简单，而`str.format`方法更强大。

如果你想了解更多关于如何用`str.format`替换`printf-style`格式（或者反过来，当然也可以），我推荐访问 PyFormat 网站[`pyformat.info/`](https://pyformat.info/)。

## PEP20，Python 之禅

大部分 Python 哲学可以通过 PEP20 来解释。Python 有一个小彩蛋，可以始终提醒你`PEP20`。只需在 Python 控制台中键入`import this`，就会得到`PEP20`的内容。引用`PEP20`：

> *"长期的 Python 程序员 Tim Peters 简洁地表达了 BDFL 对 Python 设计的指导原则，总共有 20 条格言，其中只有 19 条被记录下来。"*

接下来的几段将解释这 19 行的意图。

### 注意

PEP20 部分的示例在工作上并不完全相同，但它们确实有相同的目的。这里的许多示例都是虚构的，除了解释段落的理由外，没有其他目的。

为了清晰起见，在我们开始之前，让我们看一下`import this`的输出：

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

### 美丽胜过丑陋

尽管美是相当主观的，但有一些 Python 风格规则需要遵守：限制行长度，保持语句在单独的行上，将导入拆分为单独的行等等。

简而言之，与这样一个相当复杂的函数相比：

```py
 def filter_modulo(items, modulo):
    output_items = []
    for i in range(len(items)):
        if items[i] % modulo:
            output_items.append(items[i])
    return output_items
```

或者这样：

```py
filter_modulo = lambda i, m: [i[j] for i in range(len(i))
                              if i[j] % m]
```

只需执行以下操作：

```py
def filter_modulo(items, modulo):
    for item in items:
        if item % modulo:
            yield item
```

更简单，更易读，更美丽一些！

### 注意

这些示例的结果并不相同。前两个返回列表，而最后一个返回生成器。生成器将在第六章中更详细地讨论，*生成器和协程-无限，一步一次*。

### 显式胜过隐式

导入、参数和变量名只是许多情况中的一些，显式代码更容易阅读，但编写代码时需要付出更多的努力和/或冗长。

这是一个例子：

```py
from spam import *
from eggs import *

some_function()
```

虽然这样可以节省一些输入，但很难看出`some_function`是在哪里定义的。它是在`foo`中定义的吗？在`bar`中定义的吗？也许在两个模块中都定义了？有一些具有高级内省功能的编辑器可以帮助你，但为什么不明确地保持，这样每个人（即使只是在线查看代码）都能看到它在做什么呢？

```py
import spam
import eggs

spam.some_function()
eggs.some_function()
```

额外的好处是我们可以明确地从`spam`或`eggs`中调用函数，每个人都会更清楚代码的作用。

对于具有`*args`和`**kwargs`的函数也是一样。它们有时可能非常有用，但它们的缺点是很难确定哪些参数对于函数是有效的：

```py
def spam(egg, *args, **kwargs):
    processed_egg = process_egg(egg, *args, **kwargs)
    return Spam(processed_egg)
```

文档显然对这样的情况有所帮助，我并不反对一般情况下使用`*args`和`**kwargs`，但至少保留最常见的参数是个好主意。即使这需要你重复父类的参数，这样代码会更清晰。在未来重构父类时，你会知道是否还有子类使用了一些参数。

### 简单胜于复杂

> *"简单胜于复杂。复杂胜于混乱。"*

在开始一个新项目时，问自己最重要的问题是：它需要多复杂？

例如，假设我们已经编写了一个小程序，现在我们需要存储一些数据。我们有哪些选择？

+   完整的数据库服务器，比如 PostgreSQL 或 MySQL

+   简单的文件系统数据库，比如 SQLite 或 AnyDBM

+   平面文件存储，比如 CSV 和 TSV

+   结构化存储，比如 JSON、YAML 或 XML

+   序列化的 Python，比如 Pickle 或 Marshal

所有这些选项都有自己的用例以及根据用例的优势和劣势：

+   你存储了大量数据吗？那么完整的数据库服务器和平面文件存储通常是最方便的选择。

+   它是否能够轻松地在不需要任何包安装的不同系统上移植？这使得除了完整的数据库服务器之外的任何选项都很方便。

+   我们需要搜索数据吗？这在使用其中一个数据库系统时要容易得多，无论是文件系统还是完整的服务器。

+   是否有其他应用需要能够编辑数据？这使得像平面文件存储和结构化存储这样的通用格式成为方便的选择，但排除了序列化的 Python。

很多问题！但最重要的一个是：它需要多复杂？在`pickle`文件中存储数据是可以在三行内完成的，而连接到数据库（即使是 SQLite）将会更复杂，并且在许多情况下是不需要的：

```py
import pickle  # Or json/yaml
With open('data.pickle', 'wb') as fh:
    pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
```

对比：

```py
import sqlite3
connection = sqlite3.connect('database.sqlite')
cursor = connection.cursor()
cursor.execute('CREATE TABLE data (key text, value text)')
cursor.execute('''INSERT INTO data VALUES ('key', 'value')''')
connection.commit()
connection.close()
```

当然，这些例子远非相同，一个存储了完整的数据对象，而另一个只是在 SQLite 数据库中存储了一些键值对。然而，重点不在于此。重点是，尽管使用适当的库可以简化这个过程，但在许多情况下，代码更加复杂，而实际上却不够灵活。简单胜于复杂，如果不需要复杂性，最好避免它。

### 扁平胜于嵌套

嵌套的代码很快变得难以阅读和理解。这里没有严格的规则，但通常当你有三层嵌套循环时，就是重构的时候了。

只需看下面的例子，它打印了一个二维矩阵的列表。虽然这里没有明显的错误，但将其拆分为更多的函数可能会使目的更容易理解，也更容易测试：

```py
def print_matrices():
    for matrix in matrices:
        print('Matrix:')
        for row in matrix:
            for col in row:
                print(col, end='')
            print()
        print()
```

稍微扁平化的版本如下：

```py
def print_row(row):
    for col in row:
        print(col, end='')

def print_matrix(matrix):
    for row in matrix:
        print_row(row)
        print()

def print_matrices(matrices):
    for matrix in matrices:
        print('Matrix:')
        print_matrix(matrix)
        print()
```

这个例子可能有点复杂，但思路是正确的。深度嵌套的代码很容易变得难以阅读。

### 稀疏胜于密集

空白通常是件好事。是的，它会使你的文件变得更长，你的代码会占用更多的空间，但如果你按逻辑拆分你的代码，它可以帮助很多可读性：

```py
>>> def make_eggs(a,b):'while',['technically'];print('correct');\
...     {'this':'is','highly':'unreadable'};print(1-a+b**4/2**2)
...
>>> make_eggs(1,2)
correct
4.0
```

虽然从技术上讲是正确的，但这并不是所有人都能读懂的。我相信这需要一些努力才能找出代码实际在做什么，以及它会打印出什么数字，而不是尝试它。

```py
>>> def make_eggs(a, b):
...     'while', ['technically']
...     print('correct')
...     {'this': 'is', 'highly': 'unreadable'}
...     print(1 - a + ((b ** 4) / (2 ** 2)))
...
>>> make_eggs(1, 2)
correct
4.0
```

不过，这还不是最佳代码，但至少在代码中发生了什么更加明显了一些。

### 可读性很重要

更短并不总是意味着更容易阅读：

```py
fib=lambda n:reduce(lambda x,y:(x[0]+x[1],x[0]),[(1,1)]*(n-2))[0]
```

虽然简短的版本在简洁上有一定的美感，但我个人觉得下面的更美观：

```py
def fib(n):
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

### 实用性胜过纯粹

> *"特殊情况并不足以打破规则。尽管实用性胜过纯粹。"*

违反规则有时会很诱人，但往往会导致一连串的问题。当然，这适用于所有规则。如果你的快速修复会违反规则，你应该立即尝试重构它。很可能你以后没有时间来修复它，并会后悔。

不需要过分。如果解决方案已经足够好，而重构会更费力，那么选择有效的方法可能更好。尽管所有这些例子都涉及导入，但这个指导原则几乎适用于所有情况。

为了防止行过长，可以通过几种方法使导入变得更短，比如添加反斜杠、添加括号，或者只是缩短导入：

```py
from spam.eggs.foo.bar import spam, eggs, extra_spam, extra_eggs, extra_stuff  from spam.eggs.foo.bar import spam, eggs, extra_spam, extra_eggs
```

这种情况可以很容易地避免，只需遵循`PEP8`（每行一个导入）：

```py
from spam.eggs.foo.bar import spam from spam.eggs.foo.bar import eggs from spam.eggs.foo.bar import extra_spam from spam.eggs.foo.bar import extra_eggs from spam.eggs.foo.bar import extra_stuff  from spam.eggs.foo.bar import spam
from spam.eggs.foo.bar import eggs
from spam.eggs.foo.bar import extra_spam
from spam.eggs.foo.bar import extra_eggs
```

但是长导入怎么办？

```py
from spam_eggs_and_some_extra_spam_stuff import my_spam_and_eggs_stuff_which_is_too_long_for_a_line
```

是的…即使通常不建议为导入添加反斜杠，但在某些情况下这仍然是最佳选择：

```py
from spam_eggs_and_some_extra_spam_stuff \
    import my_spam_and_eggs_stuff_which_is_too_long_for_a_line
```

### 错误不应该悄悄地传递

> *“错误不应该悄悄地传递。除非明确地被压制。”

用 Jamie Zawinsky 的话来说：有些人在遇到错误时，会想“我知道了，我会使用`try`/`except`/`pass`块。”现在他们有了两个问题。

裸露或过于宽泛的异常捕获已经是一个坏主意了。不传递它们会让你（或者其他人在处理代码时）长时间猜测发生了什么：

```py
try:
    value = int(user_input)
except:
    pass
```

如果你真的需要捕获所有错误，就要非常明确地表达出来：

```py
try:
    value = int(user_input)
except Exception as e:
    logging.warn('Uncaught exception %r', e)
```

或者更好的是，明确捕获并添加一个合理的默认值：

```py
try:
    value = int(user_input)
except ValueError:
    value = 0
```

问题实际上更加复杂。对于依赖异常内部发生情况的代码块怎么办？例如，考虑以下代码块：

```py
try:
    value = int(user_input)
    value = do_some_processing(value)
    value = do_some_other_processing(value)
except ValueError:
    value = 0
```

如果引发了`ValueError`，是哪一行导致的？是`int(user_input)`，`do_some_processing(value)`，还是`do_some_other_processing(value)`？如果错误被悄悄地捕获，那么在正常执行代码时就无法知道，这可能非常危险。如果由于某种原因其他函数的处理发生了变化，那么以这种方式处理异常就会成为一个问题。所以，除非确实打算这样做，否则请使用这种方式：

```py
try:
    value = int(user_input)
except ValueError:
    value = 0
else:
    value = do_some_processing(value)
    value = do_some_other_processing(value)
```

### 面对模棱两可，拒绝猜测

虽然猜测在许多情况下都有效，但如果不小心就会出问题。正如在“明确胜于含糊”一段中已经展示的，当有一些`from ... import *`时，你并不能总是确定哪个模块提供了你期望的变量。

通常应该避免模棱两可，以避免猜测。清晰明了的代码会产生更少的错误。模棱两可可能出现的一个有用情况是函数调用。比如，以下两个函数调用：

```py
spam(1, 2, 3, 4, 5)
spam(spam=1, eggs=2, a=3, b=4, c=5)
```

它们可能是相同的，但也可能不是。没有看到函数的情况下是无法说的。如果函数是以以下方式实现的，那么两者之间的结果将会大不相同：

```py
def spam(a=0, b=0, c=0, d=0, e=0, spam=1, eggs=2):
    pass
```

我并不是说你应该在所有情况下使用关键字参数，但如果涉及许多参数和/或难以识别的参数（比如数字），那么这是个好主意。你可以选择逻辑变量名来传递参数，只要从代码中清楚地传达了含义。

举个例子，以下是一个类似的调用，使用自定义变量名来传达意图：

```py
a = 3
b = 4
c = 5
spam(a, b, c)
```

### 一种明显的方法

> *“应该有一种——最好只有一种——明显的方法来做。虽然一开始可能不明显，除非你是荷兰人。”

一般来说，经过一段时间思考一个困难的问题后，你会发现有一种解决方案明显优于其他选择。当然也有例外情况，这时如果你是荷兰人就会很有用。这里的笑话是指 Python 的 BDFL 和原始作者 Guido van Rossum 是荷兰人（就像我一样）。

### 现在总比永远好

> *“现在比不做要好。尽管不做通常比*立刻*做要好。”*

最好立即解决问题，而不是将问题推到未来。然而，有些情况下，立即解决问题并不是一个选择。在这些情况下，一个很好的选择可能是将一个函数标记为已弃用，这样就不会有意外忘记问题的机会：

```py
import warnings
warnings.warn('Something deprecated', DeprecationWarning)
```

### 难以解释，易于解释

> *“如果实现很难解释，那就是一个坏主意。如果实现很容易解释，那可能是一个好主意。”*

一如既往，尽量保持简单。虽然复杂的代码可能很好测试，但更容易出现错误。你能保持事情简单，就越好。

### 命名空间是一个非常棒的想法

> *“命名空间是一个非常棒的想法——让我们做更多这样的事情！”*

命名空间可以使代码更加清晰易用。正确命名它们会让它们变得更好。例如，下面这行代码是做什么的？

```py
load(fh)
```

不太清楚，对吧？

带有命名空间的版本怎么样？

```py
pickle.load(fh)
```

现在我们明白了。

举一个命名空间的例子，其完整长度使其难以使用，我们将看一下 Django 中的`User`类。在 Django 框架中，`User`类存储在`django.contrib.auth.models.User`中。许多项目以以下方式使用该对象：

```py
from django.contrib.auth.models import User
# Use it as: User
```

虽然这相当清晰，但可能会让人认为`User`类是当前类的本地类。而以下做法让人们知道它在另一个模块中：

```py
from django.contrib.auth import models
# Use it as: models.User
```

然而，这很快就会与其他模块的导入发生冲突，所以我个人建议改用以下方式：

```py
from django.contrib.auth import models as auth_models
# Use it as auth_models.User
```

这里有另一种选择：

```py
import django.contrib.auth as auth_models
# Use it as auth_models.User
```

### 结论

现在我们应该对 Python 的思想有了一些了解。创建代码：

+   美观

+   可读

+   明确的

+   足够明确

+   并非完全没有空格

所以让我们继续看一些使用 Python 风格指南创建美观、可读和简单代码的更多例子。

## 解释 PEP8

前面的段落已经展示了很多使用`PEP20`作为参考的例子，但还有一些其他重要的指南需要注意。PEP8 风格指南规定了标准的 Python 编码约定。简单地遵循 PEP8 标准并不能使你的代码变得 Pythonic，但这绝对是一个很好的开始。你使用哪种风格并不是那么重要，只要你保持一致。没有比不使用适当的风格指南更糟糕的事情了，不一致地使用更糟糕。

### 鸭子类型

鸭子类型是一种通过行为处理变量的方法。引用 Alex Martelli（我的 Python 英雄之一，也被许多人称为 MartelliBot）的话：

> *“不要检查它是否是一只鸭子：检查它是否像一只鸭子一样嘎嘎叫，像一只鸭子一样走路，等等，根据你需要玩语言游戏的鸭子行为子集。如果参数未通过这个特定的鸭子测试，那么你可以耸耸肩，问一句‘为什么是一只鸭子？’”*

在许多情况下，当人们进行比较，比如`if spam != '':`，他们实际上只是在寻找任何被认为是真值的东西。虽然你可以将值与字符串值`''`进行比较，但你通常不必这么具体。在许多情况下，只需使用`if spam:`就足够了，而且实际上功能更好。

例如，以下代码行使用`timestamp`的值生成文件名：

```py
filename = '%s.csv' % timestamp

```

因为它被命名为`timestamp`，有人可能会想要检查它实际上是一个`date`或`datetime`对象，像这样：

```py
import datetime
if isinstance(timestamp, (datetime.date, datetime.datetime)):
    filename = '%s.csv' % timestamp
else:
    raise TypeError(
        'Timestamp %r should be date(time) object, got %s'
        % (timestamp, type(timestamp))) 
```

虽然这并不是本质上错误的，但在 Python 中，比较类型被认为是一种不好的做法，因为通常情况下并不需要。在 Python 中，更倾向于鸭子类型。只需尝试将其转换为字符串，不必在乎它实际上是什么。为了说明这对最终结果几乎没有什么影响，看下面的代码：

```py
import datetime
timestamp = datetime.date(2000, 10, 5)
filename = '%s.csv' % timestamp
print('Filename from date: %s' % filename)

timestamp = '2000-10-05'
filename = '%s.csv' % timestamp
print('Filename from str: %s' % filename)
```

正如你所期望的那样，结果是相同的：

```py
Filename from date: 2000-10-05.csv
Filename from str: 2000-10-05.csv
```

同样适用于将数字转换为浮点数或整数；而不是强制执行某种类型，只需要求某些特性。需要一个可以作为数字传递的东西？只需尝试转换为`int`或`float`。需要一个`file`对象？为什么不只是检查是否有一个带有`hasattr`的`read`方法呢？

所以，不要这样做：

```py
if isinstance(value, int):
```

相反，只需使用以下内容：

```py
value = int(value)
```

而不是这样：

```py
import io

if isinstance(fh, io.IOBase):
```

只需使用以下行：

```py
if hasattr(fh, 'read'):
```

### 值和身份比较之间的差异

在 Python 中有几种比较对象的方法，标准的大于和小于，等于和不等于。但实际上还有一些其他方法，其中一个有点特殊。那就是身份比较运算符：不是使用`if spam == eggs`，而是使用`if spam is eggs`。最大的区别在于一个比较值，另一个比较身份。这听起来有点模糊，但实际上相当简单。至少在 CPython 实现中，比较的是内存地址，这意味着这是你可以得到的最轻量级的查找之一。而值需要确保类型是可比较的，也许需要检查子值，身份检查只是检查唯一标识符是否相同。

### 注意

如果你曾经写过 Java，你应该对这个原则很熟悉。在 Java 中，普通的字符串比较（`spam == eggs`）将使用身份而不是值。要比较值，你需要使用`spam.equals(eggs)`来获得正确的结果。

看看这个例子：

```py
a = 200 + 56
b = 256
c = 200 + 57
d = 257

print('%r == %r: %r' % (a, b, a == b))
print('%r is %r: %r' % (a, b, a is b))
print('%r == %r: %r' % (c, d, c == d))
print('%r is %r: %r' % (c, d, c is d))
```

虽然值是相同的，但身份是不同的。这段代码的实际结果如下：

```py
256 == 256: True
256 is 256: True
257 == 257: True
257 is 257: False
```

问题在于 Python 为所有介于`-5`和`256`之间的整数保留了一个内部整数对象数组；这就是为什么对`256`有效但对`257`无效的原因。

你可能会想知道为什么有人会想要使用`is`而不是`==`。有多个有效的答案；取决于情况，一个是正确的，另一个不是。但性能也可以是一个非常重要的考虑因素。基本准则是，当比较 Python 的单例对象，如`True`、`False`和`None`时，总是使用`is`进行比较。

至于性能考虑，考虑以下例子：

```py
spam = range(1000000)
eggs = range(1000000)
```

当执行`spam == eggs`时，这将比较两个列表中的每个项目，因此在内部实际上进行了 100 万次比较。将其与使用`spam is eggs`时的简单身份检查进行比较。

要查看 Python 在内部实际上使用`is`运算符时的操作，可以使用`id`函数。当执行`if spam is eggs`时，Python 实际上会在内部执行`if id(spam) == id(eggs)`。

### 循环

对于来自其他语言的人来说，可能会倾向于使用`for`循环或甚至`while`循环来处理`list`、`tuple`、`str`等的项目。虽然有效，但比必要的复杂。例如，考虑这段代码：

```py
i = 0
while i < len(my_list):
    item = my_list[i]
    i += 1
    do_something(i, item)
```

而不是你可以这样做：

```py
for i, item in enumerate(my_list):
    do_something(i, item)
```

虽然这可以写得更短，但通常不建议这样做，因为它不会提高可读性：

```py
[do_something(i, item) for i, item in enumerate(my_list)]
```

最后一个选项对一些人可能是清晰的，但对一些人可能不是。我个人更倾向于在实际存储结果时才使用列表推导、字典推导和 map 和 filter 语句。

例如：

```py
spam_items = [x for x in items if x.startswith('spam_')]
```

但前提是不会影响代码的可读性。

考虑一下这段代码：

```py
eggs = [is_egg(item) or create_egg(item) for item in list_of_items if egg and hasattr(egg, 'egg_property') and isinstance(egg, Egg)]  eggs = [is_egg(item) or create_egg(item) for item in list_of_items
        if egg and hasattr(egg, 'egg_property')
        and isinstance(egg, Egg)]
```

不要把所有东西都放在列表推导中，为什么不把它分成几个函数呢？

```py
def to_egg(item):
    return is_egg(item) or create_egg(item)

def can_be_egg(item):
    has_egg_property = hasattr(egg, 'egg_property')
    is_egg_instance = isinstance(egg, Egg)
    return egg and has_egg_property and is_egg_instance

eggs = [to_egg(item) for item in list_of_items if can_be_egg(item)]  eggs = [to_egg(item) for item in list_of_items if
        can_be_egg(item)]
```

虽然这段代码有点长，但我个人认为这样更易读。

### 最大行长度

许多 Python 程序员认为 79 个字符太过约束，只是保持行长。虽然我不会特别为 79 个字符辩论，但设置一个低且固定的限制，比如 79 或 99 是一个好主意。虽然显示器变得越来越宽，限制你的行仍然可以帮助你提高可读性，并且允许你将多个文件放在一起。我经常会打开四个 Python 文件并排放在一起。如果行宽超过 79 个字符，那就根本放不下了。

PEP8 指南告诉我们在行变得太长的情况下使用反斜杠。虽然我同意反斜杠比长行更可取，但我仍然认为应尽量避免使用。以下是 PEP8 的一个例子：

```py
with open('/path/to/some/file/you/want/to/read') as file_1, \
        open('/path/to/some/file/being/written', 'w') as file_2:
    file_2.write(file_1.read())
```

我会重新格式化它，而不是使用反斜杠：

```py
filename_1 = '/path/to/some/file/you/want/to/read'
filename_2 = '/path/to/some/file/being/written'
with open(filename_1) as file_1, open(filename_2, 'w') as file_2:
    file_2.write(file_1.read())
```

或者可能是以下内容：

```py
filename_1 = '/path/to/some/file/you/want/to/read'
filename_2 = '/path/to/some/file/being/written'
with open(filename_1) as file_1:
    with open(filename_2, 'w') as file_2:
        file_2.write(file_1.read())
```

当然并非总是一个选择，但保持代码简洁和可读是一个很好的考虑。它实际上为代码添加了更多信息的奖励。如果您使用传达文件名目标的名称，而不是`filename_1`，那么您正在尝试做什么就立即变得更清晰。

## 验证代码质量，pep8，pyflakes 等

有许多用于检查 Python 代码质量的工具。最简单的工具，比如`pep8`，只验证一些简单的`PEP8`错误。更先进的工具，比如`pylint`，进行高级内省，以检测潜在的错误在其他情况下工作的代码。`pylint`提供的大部分内容对许多项目来说有点过头，但仍然值得一看。

### flake8

`flake8`工具将 pep8、pyflakes 和 McCabe 结合起来，为代码设置了一个质量标准。`flake8`工具是我维护代码质量中最重要的包之一。我维护的所有包都要求 100%的`flake8`兼容性。它并不承诺可读的代码，但至少要求一定程度的一致性，这在与多个程序员一起编写项目时非常重要。

#### Pep8

用于检查 Python 代码质量的最简单的工具之一是`pep8`包。它并不检查 PEP8 标准中的所有内容，但它走了很长一段路，并且仍然定期更新以添加新的检查。`pep8`检查的一些最重要的事情如下：

+   缩进，虽然 Python 不会检查你用多少空格缩进，但这并不有助于你的代码可读性

+   缺少空格，比如`spam=123`

+   太多的空格，比如`def eggs(spam = 123):`

+   太多或太少的空行

+   行太长

+   语法和缩进错误

+   不正确和/或多余的比较（`not in`，`is not`，`if spam is True`，以及没有`isinstance`的类型比较）

结论是，`pep8`工具在测试空格和一些常见的样式问题方面帮助很大，但仍然相当有限。

#### pyflakes

这就是 pyflakes 的用武之地。pyflakes 比`pep8`更智能，它会警告你一些风格问题，比如：

+   未使用的导入

+   通配符导入（`from module import *`）

+   不正确的`__future__`导入（在其他导入之后）

但更重要的是，它会警告潜在的错误，比如以下内容：

+   重新定义已导入的名称

+   使用未定义的变量

+   在赋值之前引用变量

+   重复的参数名称

+   未使用的局部变量

PEP8 的最后一部分由 pep8-naming 包涵盖。它确保您的命名接近 PEP8 规定的标准：

+   类名为*CapWord*

+   函数、变量和参数名称全部小写

+   常量全大写并被视为常量

+   实例方法和类方法的第一个参数分别为*self*和*cls*

#### McCabe

最后，还有 McCabe 复杂性。它通过查看**抽象语法树**（**AST**）来检查代码的复杂性。它会找出有多少行、级别和语句，并在您的代码比预先配置的阈值更复杂时警告您。通常，您将通过`flake8`使用 McCabe，但也可以手动调用。使用以下代码：

```py
def spam():
    pass

def eggs(matrix):
    for x in matrix:
        for y in x:
            for z in y:
                print(z, end='')
            print()
        print()
```

McCabe 将给我们以下输出：

```py
# pip install mccabe
...
# python -m mccabe cabe_test.py 1:1: 'spam' 1
5:1: 'eggs' 4

```

当然，您的最大阈值是可配置的，但默认值为 10。 McCabe 测试返回一个受函数大小、嵌套深度和其他一些参数影响的数字。如果您的函数达到 10，可能是时候重构代码了。

#### flake8

所有这些组合在一起就是`flake8`，这是一个将这些工具结合起来并输出单个报告的工具。`flake8`生成的一些警告可能不符合您的口味，因此如果需要，每一项检查都可以在文件级别和整个项目级别上禁用。例如，我个人在所有项目中都禁用`W391`，它会警告文件末尾的空行。这是我在编写代码时发现很有用的，这样我就可以轻松地跳到文件末尾并开始编写代码，而不必先添加几行。

一般来说，在提交代码和/或将其放在网上之前，只需从源目录运行`flake8`以递归检查所有内容。

以下是一些格式不佳的代码演示：

```py
def spam(a,b,c):
    print(a,b+c)

def eggs():
    pass
```

它的结果如下：

```py
# pip install flake8
...
# flake8 flake8_test.py
flake8_test.py:1:11: E231 missing whitespace after ','
flake8_test.py:1:13: E231 missing whitespace after ','
flake8_test.py:2:12: E231 missing whitespace after ','
flake8_test.py:2:14: E226 missing whitespace around arithmetic operator
flake8_test.py:4:1: E302 expected 2 blank lines, found 1

```

### Pylint

`pylint`是一个更先进的——在某些情况下更好的——代码质量检查器。然而，`pylint`的强大功能也带来了一些缺点。而`flake8`是一个非常快速、轻量级和安全的质量检查工具，`pylint`具有更先进的内省，因此速度要慢得多。此外，`pylint`很可能会给出大量无关或甚至错误的警告。这可能被视为`pylint`的缺陷，但实际上更多的是被动代码分析的限制。诸如`pychecker`之类的工具实际上会加载和执行您的代码。在许多情况下，这是安全的，但也有一些情况是不安全的。想象一下执行一个删除文件的命令可能会发生什么。

虽然我对`pylint`没有意见，但一般来说，我发现大多数重要的问题都可以通过`flake8`来处理，其他问题也可以通过一些适当的编码标准轻松避免。如果配置正确，它可能是一个非常有用的工具，但如果没有配置，它会非常冗长。

# 常见陷阱

Python 是一种旨在清晰可读且没有任何歧义和意外行为的语言。不幸的是，这些目标并非在所有情况下都能实现，这就是为什么 Python 确实有一些特殊情况，它可能会做一些与您期望的不同的事情。

本节将向您展示编写 Python 代码时可能遇到的一些问题。

## 范围很重要！

在 Python 中有一些情况，您可能没有使用您实际期望的范围。一些例子是在声明类和使用函数参数时。

### 函数参数

以下示例显示了由于默认参数的粗心选择而导致的一个案例：

```py
def spam(key, value, list_=[], dict_={}):
    list_.append(value)
    dict_[key] = value

    print('List: %r' % list_)
    print('Dict: %r' % dict_)

spam('key 1', 'value 1')
spam('key 2', 'value 2')
```

您可能会期望以下输出：

```py
List: ['value 1']
Dict: {'key 1': 'value 1'}
List: ['value 2']
Dict: {'key 2': 'value 2'}
```

但实际上是这样的：

```py
List: ['value 1']
Dict: {'key 1': 'value 1'}
List: ['value 1', 'value 2']
Dict: {'key 1': 'value 1', 'key 2': 'value 2'}
```

原因是`list_`和`dict_`实际上是在多次调用之间共享的。唯一有用的情况是在做一些巧妙的事情时，所以请避免在函数中使用可变对象作为默认参数。

相同示例的安全替代如下：

```py
def spam(key, value, list_=None, dict_=None):
    if list_ is None:
        list_ = []

    if dict_ is None:
        dict_ {}

    list_.append(value)
    dict_[key] = value
```

### 类属性

在定义类时也会出现问题。很容易混淆类属性和实例属性。特别是对于从其他语言（如 C#）转过来的人来说，这可能会令人困惑。让我们来举个例子：

```py
class Spam(object):
    list_ = []
    dict_ = {}

    def __init__(self, key, value):
        self.list_.append(value)
        self.dict_[key] = value

        print('List: %r' % self.list_)
        print('Dict: %r' % self.dict_)

Spam('key 1', 'value 1')
Spam('key 2', 'value 2')
```

与函数参数一样，列表和字典是共享的。因此，输出如下：

```py
List: ['value 1']
Dict: {'key 1': 'value 1'}
List: ['value 1', 'value 2']
Dict: {'key 1': 'value 1', 'key 2': 'value 2'}
```

更好的选择是在类的`__init__`方法中初始化可变对象。这样，它们不会在实例之间共享：

```py
class Spam(object):
    def __init__(self, key, value):
        self.list_ = [key]
        self.dict_ = {key: value}

        print('List: %r' % self.list_)
        print('Dict: %r' % self.dict_)
```

处理类时需要注意的另一件重要事情是，类属性将被继承，这可能会让事情变得混乱。在继承时，原始属性将保留（除非被覆盖），即使在子类中也是如此：

```py
 **>>> class A(object):
...     spam = 1

>>> class B(A):
...     pass

Regular inheritance, the spam attribute of both A and B are 1 as
you would expect.
>>> A.spam
1
>>> B.spam
1

Assigning 2 to A.spam now modifies B.spam as well
>>> A.spam = 2

>>> A.spam
2
>>> B.spam
2

```

虽然由于继承而可以预料到这一点，但使用类的其他人可能不会怀疑变量在此期间发生变化。毕竟，我们修改了`A.spam`，而不是`B.spam`。

有两种简单的方法可以避免这种情况。显然，可以简单地为每个类单独设置`spam`。但更好的解决方案是永远不要修改类属性。很容易忘记属性将在多个位置更改，如果它必须是可修改的，通常最好将其放在实例变量中。

### 修改全局范围的变量

从全局范围访问变量时的一个常见问题是，设置变量会使其成为局部变量，即使访问全局变量也是如此。

这样可以工作：

```py
 **>>> def eggs():
...     print('Spam: %r' % spam)

>>> eggs()
Spam: 1

```

但以下内容不是：

```py
 **>>> spam = 1

>>> def eggs():
...     spam += 1
...     print('Spam: %r' % spam)

>>> eggs()
Traceback (most recent call last):
 **...
UnboundLocalError: local variable 'spam' referenced before assignment

```

问题在于`spam += 1`实际上转换为`spam = spam + 1`，而包含`spam =`的任何内容都会使变量成为您的范围内的局部变量。由于在那一点上正在分配局部变量，它还没有值，您正在尝试使用它。对于这些情况，有`global`语句，尽管我真的建议您完全避免使用全局变量。

## 覆盖和/或创建额外的内置函数

虽然在某些情况下可能有用，但通常您会希望避免覆盖全局函数。命名函数的`PEP8`约定-类似于内置语句、函数和变量-是使用尾随下划线。

因此，不要使用这个：

```py
list = [1, 2, 3]
```

而是使用以下方法：

```py
list_ = [1, 2, 3]
```

对于列表等，这只是一个很好的约定。对于`from`、`import`和`with`等语句，这是一个要求。忘记这一点可能会导致非常令人困惑的错误：

```py
>>> list = list((1, 2, 3))
>>> list
[1, 2, 3]

>>> list((4, 5, 6))
Traceback (most recent call last):
 **...
TypeError: 'list' object is not callable

>>> import = 'Some import'
Traceback (most recent call last):
 **...
SyntaxError: invalid syntax

```

如果您确实想要定义一个在任何地方都可用的内置函数，是可能的。出于调试目的，我已经知道在开发项目时向项目中添加此代码：

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

对于生产代码来说太过狡猾，但在需要打印语句进行调试的大型项目中仍然很有用。替代（更好的）调试解决方案可以在第十一章“调试-解决错误”中找到。

使用起来非常简单：

```py
x = 10
pp(x)
```

以下是输出：

```py
# x: 10
```

## 在迭代时修改

在某个时候，您将遇到这个问题：在迭代可变对象（如列表、字典或集合）时，您不能修改它们。所有这些都会导致`RuntimeError`告诉您在迭代期间不能修改对象：

```py
dict_ = {'spam': 'eggs'}
list_ = ['spam']
set_ = {'spam', 'eggs'}

for key in dict_:
    del dict_[key]

for item in list_:
    list_.remove(item)

for item in set_:
    set_.remove(item)
```

这可以通过复制对象来避免。最方便的选项是使用`list`函数：

```py
dict_ = {'spam': 'eggs'}
list_ = ['spam']
set_ = {'spam', 'eggs'}

for key in list(dict_):
    del dict_[key]

for item in list(list_):
    list_.remove(item)

for item in list(set_):
    set_.remove(item)
```

## 捕获异常- Python 2 和 3 之间的区别

使用 Python 3，捕获异常并存储它已经变得更加明显，使用`as`语句。问题在于许多人仍然习惯于`except Exception, variable`语法，这种语法已经不再起作用。幸运的是，Python 3 的语法已经回溯到 Python 2，所以现在您可以在任何地方使用以下语法：

```py
try:
    ... # do something here
except (ValueError, TypeError) as e:
    print('Exception: %r' % e)
```

另一个重要的区别是，Python 3 使这个变量局限于异常范围。结果是，如果您想要在`try`/`except`块之后使用它，您需要在之前声明异常变量：

```py
def spam(value):
    try:
        value = int(value)
    except ValueError as exception:
        print('We caught an exception: %r' % exception)

    return exception

spam('a')
```

您可能期望由于我们在这里得到一个异常，这样可以工作；但实际上，它不起作用，因为在`return`语句的那一点上`exception`不存在。

实际输出如下：

```py
We caught an exception: ValueError("invalid literal for int() with base 10: 'a'",)
Traceback (most recent call last):
  File "test.py", line 14, in <module>
    spam('a')
  File "test.py", line 11, in spam
    return exception
UnboundLocalError: local variable 'exception' referenced before assignment
```

就个人而言，我会认为前面的代码在任何情况下都是错误的：如果没有异常怎么办？它会引发相同的错误。幸运的是，修复很简单；只需将值写入到作用域之外的变量中。这里需要注意的一点是，你需要明确保存变量到父作用域。这段代码也不起作用：

```py
def spam(value):
    exception = None
    try:
        value = int(value)
    except ValueError as exception:
        print('We caught an exception: %r' % exception)

    return exception
```

我们真的需要明确保存它，因为 Python 3 会自动删除在`except`语句结束时使用`as variable`保存的任何内容。这样做的原因是 Python 3 的异常包含一个`__traceback__`属性。拥有这个属性会让垃圾收集器更难处理，因为它引入了一个递归的自引用循环（*exception -> traceback -> exception -> traceback… ad nauseum*）。为了解决这个问题，Python 基本上执行以下操作：

```py
exception = None
try:
    value = int(value)
except ValueError as exception:
    try:
        print('We caught an exception: %r' % exception)
    finally:
        del exception
```

解决方案非常简单 - 幸运的是 - 但你应该记住，这可能会在程序中引入内存泄漏。Python 的垃圾收集器足够聪明，可以理解这些变量不再可见，并最终会删除它，但这可能需要更长的时间。垃圾收集实际上是如何工作的在第十二章中有介绍，*性能 - 跟踪和减少内存和 CPU 使用情况*。这是代码的工作版本：

```py
def spam(value):
    exception = None
    try:
        value = int(value)
    except ValueError as e:
        exception = e
        print('We caught an exception: %r' % exception)

    return exception
```

## 延迟绑定 - 要小心闭包

闭包是在代码中实现局部作用域的一种方法。它使得可以在本地定义变量，而不会覆盖父（或全局）作用域中的变量，并且稍后将变量隐藏在外部作用域中。Python 中闭包的问题在于出于性能原因，Python 尝试尽可能晚地绑定其变量。虽然通常很有用，但它确实具有一些意想不到的副作用：

```py
eggs = [lambda a: i * a for i in range(3)]

for egg in eggs:
    print(egg(5))
```

预期结果？应该是这样的，对吧？

```py
0
5
10
```

不，不幸的是。这类似于类继承与属性的工作方式。由于延迟绑定，变量`i`在调用时从周围的作用域中调用，而不是在实际定义时调用。

实际结果如下：

```py
10
10
10
```

那么应该怎么做呢？与前面提到的情况一样，需要将变量设为局部变量。一种替代方法是通过使用`partial`对函数进行柯里化来强制立即绑定。

```py
import functools

eggs = [functools.partial(lambda i, a: i * a, i) for i in range(3)]

for egg in eggs:
    print(egg(5))
```

更好的解决方案是通过不引入额外的作用域（`lambda`）来避免绑定问题，这些作用域使用外部变量。如果`i`和`a`都被指定为`lambda`的参数，这将不是一个问题。

## 循环导入

尽管 Python 对循环导入相当宽容，但也有一些情况会出现错误。

假设我们有两个文件。

`eggs.py`：

```py
from spam import spam

def eggs():
    print('This is eggs')
    spam()
```

`spam.py`：

```py
from eggs import eggs

def spam():
    print('This is spam')

if __name__ == '__main__':
    eggs()
```

运行`spam.py`将导致循环`import`错误：

```py
Traceback (most recent call last):
  File "spam.py", line 1, in <module>
    from eggs import eggs
  File "eggs.py", line 1, in <module>
    from spam import spam
  File "spam.py", line 1, in <module>
    from eggs import eggs
ImportError: cannot import name 'eggs'
```

有几种方法可以解决这个问题。重新构造代码通常是最好的方法，但最佳解决方案取决于问题。在前面的情况下，可以很容易地解决。只需使用模块导入而不是函数导入（无论是否存在循环导入，我都建议这样做）。

`eggs.py`：

```py
import spam

def eggs():
    print('This is eggs')
    spam.spam()
```

`spam.py`：

```py
import eggs

def spam():
    print('This is spam')

if __name__ == '__main__':
    eggs.eggs()
```

另一种解决方案是将导入语句移到函数内部，以便在运行时发生。这不是最漂亮的解决方案，但在许多情况下都能解决问题。

`eggs.py`：

```py
def eggs():
    from spam import spam
    print('This is eggs')
    spam()
```

`spam.py`：

```py
def spam():
    from eggs import eggs
    print('This is spam')

if __name__ == '__main__':
    eggs()
```

最后还有一种解决方案，即将导入移到实际使用它们的代码下面。这通常不被推荐，因为它可能会使导入的位置不明显，但我仍然认为这比在函数调用中使用`import`更可取。

`eggs.py`：

```py
def eggs():
    print('This is eggs')
    spam()

from spam import spam
```

`spam.py`：

```py
def spam():
    print('This is spam')

from eggs import eggs

if __name__ == '__main__':
    eggs()
```

是的，还有其他解决方案，比如动态导入。其中一个例子是 Django 的`ForeignKey`字段支持字符串而不是实际类。但这些通常是一个非常糟糕的想法，因为它们只会在运行时进行检查。因此，错误只会在执行使用它的任何代码时引入，而不是在修改代码时引入。因此，请尽量避免这些情况，或者确保添加适当的自动化测试以防止意外错误。特别是当它们在内部引起循环导入时，它们将成为一个巨大的调试痛点。

## 导入冲突

一个极其令人困惑的问题是导入冲突——多个具有相同名称的包/模块。我在我的包上收到了不少 bug 报告，例如，有人试图使用我的`numpy-stl`项目，它位于名为`stl`的包中的一个名为`stl.py`的测试文件。结果是：它导入了自身而不是`stl`包。虽然这种情况很难避免，至少在包内部，相对导入通常是一个更好的选择。这是因为它还告诉其他程序员，导入来自本地范围而不是另一个包。因此，不要写`import spam`，而是写`from . import spam`。这样，代码将始终从当前包加载，而不是任何偶然具有相同名称的全局包。

除此之外，还存在包之间不兼容的问题。常见名称可能被几个包使用，因此在安装这些包时要小心。如果有疑问，只需创建一个新的虚拟环境，然后再试一次。这样做可以节省大量的调试时间。

# 摘要

本章向我们展示了 Python 哲学的全部内容，并向我们解释了 Python 之禅的含义。虽然代码风格是非常个人化的，但 Python 至少有一些非常有帮助的指导方针，至少能让人们大致保持在同一页面和风格上。最后，我们都是成年人；每个人都有权利按照自己的意愿编写代码。但我请求您。请阅读风格指南，并尽量遵守它们，除非您有一个真正充分的理由不这样做。

随着这种力量而来的是巨大的责任，也有一些陷阱，尽管并不多。有些陷阱足够棘手，以至于经常会让我困惑，而我已经写了很长时间的 Python 了！Python 不断改进。自 Python 2 以来，许多陷阱已经得到解决，但有些将永远存在。例如，递归导入和定义在大多数支持它们的语言中很容易让你掉进陷阱，但这并不意味着我们会停止努力改进 Python。

Python 多年来的改进的一个很好的例子是 collections 模块。它包含了许多有用的集合，这些集合是由用户添加的，因为有这样的需求。其中大多数实际上是用纯 Python 实现的，因此它们很容易被任何人阅读。理解可能需要更多的努力，但我真的相信，如果你能读完这本书，你将没有问题理解这些集合的作用。但我不能保证完全理解内部工作；其中一些部分更多地涉及通用计算机科学而不是 Python 掌握。

下一章将向您展示 Python 中可用的一些集合以及它们的内部构造。尽管您无疑熟悉列表和字典等集合，但您可能不清楚某些操作涉及的性能特征。如果本章中的一些示例不够清晰，您不必担心。下一章将至少重新讨论其中一些，并且更多内容将在后续章节中介绍。
