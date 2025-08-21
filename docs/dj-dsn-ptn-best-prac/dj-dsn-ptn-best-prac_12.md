# 附录 A. Python 2 与 Python 3

这本书中的所有代码示例都是为 Python 3.4 编写的。除了非常小的更改，它们也可以在 Python 2.7 中运行。作者认为 Python 3 已经成为新的 Django 项目的首选选择。

Python 2.7 的开发原计划在 2015 年结束，但通过 2020 年延长了五年。不会有 Python 2.8。很快，所有主要的 Linux 发行版都将完全转换为使用 Python 3 作为默认版本。许多 PaaS 提供商，如 Heroku，已经支持 Python 3.4。

Python Wall of Superpowers 中列出的大多数软件包已经变成了绿色（表示它们支持 Python 3）。几乎所有红色的软件包都有一个正在积极开发的 Python 3 版本。

Django 从 1.5 版本开始支持 Python 3。事实上，策略是用 Python 3 重写代码，并将 Python 2 作为向后兼容的要求。这主要是使用 Six 这个 Python 2 和 3 兼容性库的实用函数实现的。

正如你很快会看到的，Python 3 在许多方面都是一种更优越的语言，因为它有许多改进，主要是为了一致性。然而，如果你正在用 Django 构建 Web 应用程序，那么在转向 Python 3 时可能会遇到的差异是相当微不足道的。

# 但我仍在使用 Python 2.7！

如果你被困在 Python 2.7 的环境中，那么示例项目可以很容易地回溯。项目根目录下有一个名为`backport3to2.py`的自定义脚本，可以执行一次性转换为 Python 2.x。请注意，它不适用于其他项目。

然而，如果你想知道为什么 Python 3 更好，那就继续阅读。

# Python 3

Python 3 的诞生是出于必要性。Python 2 的一个主要问题是其对非英语字符的处理不一致（通常表现为臭名昭著的`UnicodeDecodeError`异常）。Guido 启动了 Python 3 项目，清理了许多这样的语言问题，同时打破了向后兼容性。

Python 3.0 的第一个 alpha 版本是在 2007 年 8 月发布的。从那时起，Python 2 和 Python 3 一直在并行开发，由核心开发团队开发了多年。最终，Python 3 有望成为该语言的未来。

## Python 3 for Djangonauts

本节涵盖了从 Django 开发者的角度看 Python 3 的最重要的变化。有关所有变化的完整列表，请参考本章末尾的推荐阅读部分。

示例分别以 Python 3 和 Python 2 给出。根据你的安装，所有 Python 3 命令可能需要从`python`更改为`python3`或`python3.4`。

## 将所有的 __unicode__ 方法更改为 __str__

在 Python 3 中，用于模型的字符串表示调用`__str__()`方法，而不是尴尬的`__unicode__()`方法。这是识别 Python 3 移植代码最明显的方法之一：

| Python 2 | Python 3 |
| --- | --- |

|

```py
class Person(models.Model):
    name = models.TextField()

    def __unicode__(self):
        return self.name
```

|

```py
class Person(models.Model):
    name = models.TextField()

    def __str__(self):
        return self.name
```

|

前面的表反映了 Python 3 处理字符串的方式的不同。在 Python 2 中，类的可读表示可以通过`__str__()`（字节）或`__unicode__()`（文本）返回。然而，在 Python 3 中，可读表示只是通过`__str__()`（文本）返回。

## 所有的类都继承自 object 类

Python 2 有两种类：旧式（经典）和新式。新式类是直接或间接继承自`object`的类。只有新式类才能使用 Python 的高级特性，如 slots、描述符和属性。其中许多被 Django 使用。然而，出于兼容性原因，类仍然默认为旧式。

在 Python 3 中，旧式类不再存在。如下表所示，即使你没有明确地提到任何父类，`object`类也会作为基类存在。因此，所有的类都是新式的。

| Python 2 | Python 3 |
| --- | --- |

|

```py
>>> class CoolMixin:
...     pass
>>> CoolMixin.__bases__
() 
```

|

```py
>>> class CoolMixin:
...     pass
>>> CoolMixin.__bases__
(<class 'object'>,) 
```

|

## 调用 super()更容易

在 Python 3 中，更简单的调用`super()`，不带任何参数，将为你节省一些输入。

| Python 2 | Python 3 |
| --- | --- |

|

```py
class CoolMixin(object):

    def do_it(self):
        return super(CoolMixin, 
                  self).do_it()
```

|

```py
class CoolMixin:

    def do_it(self):
        return super().do_it()
```

|

指定类名和实例是可选的，从而使你的代码更加干燥，减少了重构时出错的可能性。

## 相对导入必须是显式的

想象一个名为`app1`的包的以下目录结构：

```py
/app1
  /__init__.py
  /models.py
  /tests.py 
```

现在，在 Python 3 中，让我们在`app1`的父目录中运行以下代码：

```py
$ echo "import models" > app1/tests.py
$ python -m app1.tests
Traceback (most recent call last):
   ... omitted ...
ImportError: No module named 'models'
$ echo "from . import models" > app1/tests.py
$ python -m app1.tests
# Successfully imported
```

在一个包内，当引用一个兄弟模块时，你应该使用显式相对导入。在 Python 3 中，你可以省略`__init__.py`，尽管它通常用于标识一个包。

在 Python 2 中，你可以使用`import models`成功导入`models.py`模块。然而，这是模棱两可的，可能会意外地导入 Python 路径中的任何其他`models.py`。因此，在 Python 3 中是被禁止的，在 Python 2 中也是不鼓励的。

## HttpRequest 和 HttpResponse 有 str 和 bytes 类型

在 Python 3 中，根据 PEP 3333（WSGI 标准的修正），我们要小心不要混合通过 HTTP 进入或离开的数据，这些数据将是字节，而不是框架内的文本，这些文本将是本地（Unicode）字符串。

基本上，对于`HttpRequest`和`HttpResponse`对象：

+   标头将始终是`str`对象

+   输入和输出流将始终是`byte`对象

与 Python 2 不同，字符串和字节在执行彼此的比较或连接时不会被隐式转换。字符串只意味着 Unicode 字符串。

## 异常语法的变化和改进

在 Python 3 中，异常处理的语法和功能得到了显著改进。

在 Python 3 中，你不能使用逗号分隔的语法来处理`except`子句。而是使用`as`关键字：

| Python 2 | Python 3 and 2 |
| --- | --- |

|

```py
try:
  pass
except e, BaseException:
  pass
```

|

```py
try:
  pass
except e as BaseException:
  pass
```

|

新的语法也建议在 Python 2 中使用。

在 Python 3 中，所有的异常都必须派生（直接或间接）自`BaseException`。在实践中，你会通过从`Exception`类派生来创建你自己的自定义异常。

作为错误报告的一个重大改进，如果在处理异常时发生了异常，那么整个异常链都会被报告：

| Python 2 | Python 3 |
| --- | --- |

|

```py
>>> try:
...   print(undefined)
... except Exception:
...   print(oops)
... 
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  NameError: name 'oops' is not defined
```

|

```py
>>> try:
...   print(undefined)
... except Exception:
...   print(oops)
... 
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  NameError: name 'undefined' is not defined
```

在处理前面的异常时，发生了另一个异常：

```py
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  NameError: name 'oops' is not defined
```

|

一旦你习惯了这个特性，你肯定会在 Python 2 中想念它。

## 标准库重新组织

核心开发人员已经清理和组织了 Python 标准库。例如，`SimpleHTTPServer`现在位于`http.server`模块中：

| Python 2 | Python 3 |
| --- | --- |

|

```py
$ python -m SimpleHTTP

ServerServing HTTP on 0.0.0.0 port 8000 ...
```

|

```py
$python -m http.server

Serving HTTP on 0.0.0.0 port 8000 ...
```

|

## 新的好东西

Python 3 不仅仅是关于语言修复。这也是 Python 最前沿的开发发生的地方。这意味着语言在语法、性能和内置功能方面的改进。

一些值得注意的新模块添加到 Python 3 中如下：

+   `asyncio`：这包含异步 I/O、事件循环、协程和任务

+   `unittest.mock`：这包含用于测试的模拟对象库

+   `pathlib`：这包含面向对象的文件系统路径

+   `statistics`：这包含数学统计函数

即使其中一些模块已经回溯到 Python 2，迁移到 Python 3 并利用它们作为内置模块更具吸引力。

### 使用 Pyvenv 和 Pip

大多数严肃的 Python 开发者更喜欢使用虚拟环境。`virtualenv`非常流行，可以将项目设置与系统范围的 Python 安装隔离开来。值得庆幸的是，Python 3.3 集成了类似的功能，使用`venv`模块。

自 Python 3.4 开始，一个新的虚拟环境将预先安装 pip，一个流行的安装程序：

```py
$ python -m venv djenv

[djenv] $ source djenv/bin/activate

[djenv] $ pip install django
```

请注意，命令提示符会更改以指示你的虚拟环境已被激活。

## 其他变化

我们不可能在这个附录中涵盖所有 Python 3 的变化和改进。然而，其他常见的变化如下：

1.  `print()` **现在是一个函数**：以前它是一个语句，也就是说，参数不需要括号。

1.  **整数不会溢出**：`sys.maxint`已经过时，整数将具有无限精度。

1.  **不等运算符** `<>` **已被移除**：请使用`!=`代替。

1.  **真正的整数除法**：在 Python 2 中，`3 / 2`会计算为`1`。在 Python 3 中将正确计算为`1.5`。

1.  **使用** `range` **而不是** `xrange()`：`range()`现在将返回迭代器，就像`xrange()`以前的工作方式一样。

1.  **字典键是视图**：`dict`和`dict`-like 类（如`QueryDict`）将返回迭代器而不是`keys()`、`items()`和`values()`方法调用的列表。

# 更多信息

+   阅读 Guido 的*Python 3.0 的新功能*，网址为[`docs.python.org/3/whatsnew/3.0.html`](https://docs.python.org/3/whatsnew/3.0.html)

+   要查找 Python 每个版本的新功能，请阅读[`docs.python.org/3/whatsnew/`](https://docs.python.org/3/whatsnew/)上的*Python 的新功能*。

+   有关 Python 3 的详细答案，请阅读 Nick Coghlan 的*Python 3 问答*，网址为[`python-notes.curiousefficiency.org/en/latest/python3/questions_and_answers.html`](http://python-notes.curiousefficiency.org/en/latest/python3/questions_and_answers.html)
