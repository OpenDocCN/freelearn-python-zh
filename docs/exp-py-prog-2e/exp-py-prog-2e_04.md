# 第四章：选择好的名称

大部分标准库都是在考虑可用性的基础上构建的。例如，使用内置类型是自然而然的，并且旨在易于使用。在这种情况下，Python 可以与您在编写程序时可能考虑的伪代码进行比较。大部分代码都可以大声朗读出来。例如，任何人都应该能理解这段代码：

```py
my_list = []

if 'd' not in my_list:
    my_list.append('d')
```

这就是为什么与其他语言相比，编写 Python 如此容易的原因之一。当您编写程序时，您的思维流很快就会转化为代码行。

本章重点介绍了编写易于理解和使用的代码的最佳实践，包括：

+   使用在 PEP 8 中描述的命名约定

+   一套命名最佳实践

+   检查遵循样式指南的流行工具的简短摘要

# PEP 8 和命名最佳实践

PEP 8（[`www.python.org/dev/peps/pep-0008`](http://www.python.org/dev/peps/pep-0008)）提供了编写 Python 代码的风格指南。除了一些基本规则，如空格缩进、最大行长度和其他有关代码布局的细节，PEP 8 还提供了一个关于命名约定的部分，大多数代码库都遵循这些约定。

本节提供了对本 PEP 的快速摘要，并为每种元素添加了命名最佳实践指南。您仍应考虑阅读 PEP 8 文件作为强制性要求。

## 何时以及为什么要遵循 PEP 8？

如果您正在创建一个打算开源的新软件包，那么答案很简单：始终如此。对于大多数 Python 开源软件来说，PEP 8 实际上是标准的代码风格。如果您希望接受其他程序员的任何合作，那么您绝对应该坚持 PEP 8，即使您对最佳代码风格指南有不同的看法。这样做的好处是使其他开发人员更容易直接参与到您的项目中。对于新手来说，代码将更容易阅读，因为它在风格上与大多数其他 Python 开源软件包保持一致。

此外，完全遵循 PEP 8 可以节省您未来的时间和麻烦。如果您想将代码发布给公众，最终您将不可避免地面临来自其他程序员的建议，要求您切换到 PEP 8。关于是否真的有必要为特定项目这样做的争论往往是无休止的争论，无法取胜。这是一个悲伤的事实，但您最终可能会被迫遵循这个风格指南，以保持一致性，以免失去贡献者。

此外，如果项目的整个代码库处于成熟的开发状态，重新设计可能需要大量的工作。在某些情况下，这种重新设计可能需要改变几乎每一行代码。虽然大部分更改可以自动化（缩进、换行和尾随空格），但这种大规模的代码改造通常会在基于分支的每个版本控制工作流程中引入许多冲突。同时一次性审查如此多的更改也非常困难。这就是为什么许多开源项目有一个规则，即样式修复更改应始终包含在不影响任何功能或错误的单独拉取/合并请求或补丁中的原因。

## 超越 PEP 8 - 面向团队的特定风格指南

尽管 PEP 8 提供了一套全面的风格指南，但仍然为开发人员留下了一些自由。特别是在嵌套数据文字和需要长参数列表的多行函数调用方面。一些团队可能决定他们需要额外的样式规则，最佳选择是将它们正式化为每个团队成员都可以访问的某种文档。

此外，在某些情况下，严格遵守 PEP 8 对于一些旧项目可能是不可能的，或者在经济上是不可行的，因为这些项目没有定义风格指南。即使这些项目不符合官方的 PEP 8 规则，它们仍将受益于实际编码约定的形式化。请记住，与 PEP 8 一致性更重要的是项目内的一致性。如果规则被形式化并且对每个程序员都可用作参考，那么在项目和组织内保持一致性就会更容易。

# 命名风格

Python 中使用的不同命名风格有：

+   驼峰命名法

+   混合大小写

+   大写字母和下划线

+   小写和下划线小写

+   前导和尾随下划线，有时是双下划线

小写和大写元素通常是一个单词，有时是几个单词连接在一起。使用下划线时，它们通常是缩写短语。使用一个单词更好。前导和尾随下划线用于标记隐私和特殊元素。

这些风格适用于：

+   变量

+   函数和方法

+   属性

+   类

+   模块

+   包

## 变量

Python 中有两种类型的变量：

+   常量

+   公共和私有变量

### 常量

对于常量全局变量，使用大写字母和下划线。它告诉开发人员给定的变量表示一个常量值。

### 注意

Python 中没有像 C++ 中的 `const` 那样的真正的常量。您可以更改任何变量的值。这就是为什么 Python 使用命名约定来标记变量为常量的原因。

例如，`doctest` 模块提供了一系列选项标志和指令（[`docs.python.org/2/library/doctest.html`](https://docs.python.org/2/library/doctest.html)），它们是简短的句子，清楚地定义了每个选项的用途：

```py
from doctest import IGNORE_EXCEPTION_DETAIL
from doctest import REPORT_ONLY_FIRST_FAILURE
```

这些变量名似乎相当长，但清楚地描述它们是很重要的。它们的使用大多位于初始化代码中，而不是代码本身的主体，因此这种冗长并不令人讨厌。

### 注意

缩写名称大多数时候会使代码难以理解。当缩写不清晰时，不要害怕使用完整的单词。

有些常量的名称也受到底层技术的驱动。例如，`os` 模块使用一些在 C 侧定义的常量，例如 `EX_XXX` 系列，它定义了 Unix 退出代码数字。例如，相同的名称代码可以在系统的 `sysexits.h` C 头文件中找到：

```py
import os
import sys

sys.exit(os.EX_SOFTWARE)
```

在使用常量时的另一个好的做法是将它们收集在使用它们的模块的顶部，并在它们用于此类操作时将它们组合在新变量下：

```py
import doctest
TEST_OPTIONS = (doctest.ELLIPSIS |
                doctest.NORMALIZE_WHITESPACE | 
                doctest.REPORT_ONLY_FIRST_FAILURE)
```

### 命名和用法

常量用于定义程序依赖的一组值，例如默认配置文件名。

一个很好的做法是将所有常量收集到包中的单个文件中。例如 Django 就是这样工作的。一个名为 `settings.py` 的模块提供了所有常量：

```py
# config.py
SQL_USER = 'tarek'
SQL_PASSWORD = 'secret'
SQL_URI = 'postgres://%s:%s@localhost/db' % (
    SQL_USER, SQL_PASSWORD
)
MAX_THREADS = 4
```

另一种方法是使用配置文件，可以使用 `ConfigParser` 模块解析，或者像 `ZConfig` 这样的高级工具，它是 Zope 中用于描述其配置文件的解析器。但有些人认为在 Python 这样的语言中使用另一种文件格式是一种过度使用，因为文件可以像文本文件一样容易地被编辑和更改。

对于像标志一样的选项，一种常见的做法是将它们与布尔运算结合使用，就像 `doctest` 和 `re` 模块所做的那样。从 `doctest` 中采用的模式非常简单：

```py
OPTIONS = {}

def register_option(name):
    return OPTIONS.setdefault(name, 1 << len(OPTIONS))

def has_option(options, name):
    return bool(options & name)

# now defining options
BLUE = register_option('BLUE')
RED = register_option('RED')
WHITE = register_option('WHITE')
```

您将获得：

```py
>>> # let's try them
>>> SET = BLUE | RED
>>> has_option(SET, BLUE)
True
>>> has_option(SET, WHITE)
False

```

当创建这样一组新的常量时，避免为它们使用共同的前缀，除非模块有几组。模块名称本身就是一个共同的前缀。另一个解决方案是使用内置的`enum`模块中的`Enum`类，并简单地依赖于`set`集合而不是二进制运算符。不幸的是，`Enum`类在针对旧版本的 Python 的代码中应用有限，因为`enum`模块是在 Python 3.4 版本中提供的。

### 注意

在 Python 中，使用二进制位操作来组合选项是很常见的。包含 OR（`|`）运算符将允许您将多个选项组合成单个整数，而 AND（`&`）运算符将允许您检查该选项是否存在于整数中（参考`has_option`函数）。

### 公共和私有变量

对于可变且可以通过导入自由使用的全局变量，在需要保护时应使用带下划线的小写字母。但是这种类型的变量并不经常使用，因为模块通常提供 getter 和 setter 来处理它们在需要保护时。在这种情况下，前导下划线可以将变量标记为包的私有元素：

```py
_observers = []

def add_observer(observer):
    _observers.append(observer)

def get_observers():
    """Makes sure _observers cannot be modified."""
    return tuple(_observers)
```

位于函数和方法中的变量遵循相同的规则，并且从不标记为私有，因为它们是局部的。

对于类或实例变量，只有在使变量成为公共签名的一部分不带来任何有用信息或是多余的情况下，才需要使用私有标记（前导下划线）。

换句话说，如果变量在方法中用于提供公共功能，并且专门用于此角色，则最好将其设置为私有。

例如，为属性提供动力的属性是良好的私有成员：

```py
class Citizen(object):
    def __init__(self):
        self._message = 'Rosebud...'

    def _get_message(self):
        return self._message

    kane = property(_get_message)
```

另一个例子是保持内部状态的变量。这个值对于代码的其余部分并不有用，但参与类的行为：

```py
class UnforgivingElephant(object):
    def __init__(self, name):
        self.name = name
        self._people_to_stomp_on = []

    def get_slapped_by(self, name):
        self._people_to_stomp_on.append(name)
        print('Ouch!')

    def revenge(self):
        print('10 years later...')
        for person in self._people_to_stomp_on:
            print('%s stomps on %s' % (self.name, person))
```

在交互式会话中，您将看到以下内容：

```py
>>> joe = UnforgivingElephant('Joe')
>>> joe.get_slapped_by('Tarek')
Ouch!
>>> joe.get_slapped_by('Bill')
Ouch!
>>> joe.revenge()
10 years later...
Joe stomps on Tarek
Joe stomps on Bill

```

### 函数和方法

函数和方法应该使用小写和下划线。在旧标准库模块中，这条规则并不总是成立。Python 3 对标准库进行了大量重组，因此大多数函数和方法都具有一致的大小写。但是，对于一些模块，如`threading`，您可以访问使用*mixedCase*的旧函数名称（例如`currentThread`）。这是为了更容易地向后兼容，但如果您不需要在旧版本的 Python 中运行代码，那么您应该避免使用这些旧名称。

在小写规范成为标准之前，编写方法的方式是很常见的，并且一些框架，如 Zope 和 Twisted，也在使用*mixedCase*来命名方法。与他们一起工作的开发人员社区仍然相当庞大。因此，使用*mixedCase*和小写加下划线之间的选择绝对受到您使用的库的驱动。

作为 Zope 开发人员，要保持一致并不容易，因为构建一个同时混合纯 Python 模块和导入 Zope 代码的应用程序是困难的。在 Zope 中，一些类混合了这两种约定，因为代码库仍在不断发展，Zope 开发人员试图采纳被许多人接受的常见约定。

在这种库环境中的一个不错的做法是，仅对在框架中公开的元素使用*mixedCase*，并将其余代码保持在 PEP 8 风格中。

值得注意的是，Twisted 项目的开发人员对这个问题采取了完全不同的方法。Twisted 项目和 Zope 一样，早于 PEP 8 文档。它是在没有代码风格的官方指南时开始的，因此它有自己的指南。关于缩进、文档字符串、行长度等的风格规则可以很容易地采用。另一方面，将所有代码更新以匹配 PEP 8 的命名约定将导致完全破坏的向后兼容性。对于如此庞大的 Twisted 项目来说，这是不可行的。因此，Twisted 尽可能地采用了 PEP 8 的规范，并将*mixedCase*作为其自己的编码标准的一部分。这与 PEP 8 的建议完全兼容，因为它明确指出项目内的一致性比与 PEP 8 风格指南的一致性更重要。

### 私有争议

对于私有方法和函数，通常会添加一个前导下划线。这条规则曾经引起了很大的争议，因为 Python 中有名称修饰的特性。当一个方法有两个前导下划线时，解释器会即时将其重命名，以防止与任何子类的方法发生名称冲突。

因此，一些人倾向于在子类中使用双下划线来避免名称冲突：

```py
class Base(object):
    def __secret(self):
        print("don't tell")

    def public(self):
        self.__secret()

class Derived(Base):
    def __secret(self):
        print("never ever")
```

你会看到：

```py
>>> Base.__secret
Traceback (most recent call last):
 **File "<input>", line 1, in <module>
AttributeError: type object 'Base' has no attribute '__secret'
>>> dir(Base)
['_Base__secret', ..., 'public']
>>> Derived().public()
don't tell

```

Python 中名称修饰的最初动机不是为了提供类似 C++中的私有花招，而是为了确保一些基类在子类中隐式避免冲突，特别是在多重继承的情况下。但是，将其用于每个属性会使代码在私有部分变得模糊，这完全不符合 Python 的风格。

因此，一些人认为应始终使用显式名称修饰：

```py
class Base:
    def _Base_secret(self):  # don't do this !!!
        print("you told it ?")
```

这在整个代码中重复了类名，因此应该优先使用`__`。

但是，最佳实践是避免使用名称修饰，而是在编写子类方法之前查看类的`__mro__`（方法解析顺序）值。必须小心更改基类的私有方法。

有关此主题的更多信息，请参阅多年前在 Python-Dev 邮件列表中发生的有趣讨论，人们在那里讨论了名称修饰的实用性及其在语言中的命运。可以在[`mail.python.org/pipermail/python-dev/2005-December/058555.html`](http://mail.python.org/pipermail/python-dev/2005-December/058555.html)找到。

### 特殊方法

特殊方法（[`docs.python.org/3/reference/datamodel.html#special-method-names`](https://docs.python.org/3/reference/datamodel.html#special-method-names)）以双下划线开头和结尾，普通方法不应使用此约定。一些开发人员习惯称它们为*dunder*方法，这是双下划线的混成词。它们用于操作符重载、容器定义等。为了可读性，它们应该在类定义的开头收集起来：

```py
class WeirdInt(int):
    def __add__(self, other):
        return int.__add__(self, other) + 1

    def __repr__(self):
        return '<weirdo %d>' % self

    # public API
    def do_this(self):
        print('this')

    def do_that(self):
        print('that')
```

对于普通方法，你不应该使用这种名称。因此，不要为方法发明这样的名称：

```py
class BadHabits:
    def __my_method__(self):
        print('ok')
```

### 参数

参数使用小写，如果需要则使用下划线。它们遵循与变量相同的命名规则。

### 属性

属性的名称为小写，或者小写加下划线。大多数情况下，它们表示对象的状态，可以是名词、形容词，或者在需要时是一个小短语：

```py
class Connection:
    _connected = []

    def connect(self, user):
        self._connected.append(user)

    @property

    def connected_people(self):
        return ', '.join(self._connected)

```

在交互式会话中运行时：

```py
>>> connection = Connection()
>>> connection.connect('Tarek')
>>> connection.connect('Shannon')
>>> print(connection.connected_people)
Tarek, Shannon

```

### 类

类的名称始终为 CamelCase，并且如果它们对模块是私有的，则可能有一个前导下划线。

类和实例变量通常是名词短语，并且与动词短语的方法名称形成使用逻辑：

```py
class Database:
    def open(self):
        pass

class User:
    pass
```

以下是交互式会话中的一个示例用法：

```py
>>> user = User()
>>> db = Database()
>>> db.open()

```

### 模块和包

除了特殊模块`__init__`之外，模块名称都是小写的，没有下划线。

以下是标准库中的一些示例：

+   操作系统

+   `sys`

+   `shutil`

当模块对包是私有的时，会添加一个前导下划线。编译的 C 或 C++模块通常以下划线命名，并在纯 Python 模块中导入。

包名称遵循相同的规则，因为它们的行为类似于更结构化的模块。

# 命名指南

一组通用的命名规则可以应用于变量、方法、函数和属性。类和模块的名称也对命名空间的构建以及代码的可读性起着重要作用。这个迷你指南提供了选择它们的名称的常见模式和反模式。

## 对布尔元素使用`has`或`is`前缀

当一个元素持有布尔值时，`is`和`has`前缀提供了一种使其在其命名空间中更易读的自然方式：

```py
class DB:
    is_connected = False
    has_cache = False
```

## 对于持有集合的变量使用复数

当一个元素持有一个集合时，最好使用复数形式。当它们像序列一样被暴露时，一些映射也可以从中受益：

```py
class DB:
    connected_users = ['Tarek']
    tables = {
        'Customer': ['id', 'first_name', 'last_name']
    }
```

## 对于字典使用显式名称

当一个变量持有一个映射时，尽可能使用显式名称。例如，如果一个`dict`持有一个人的地址，它可以被命名为`persons_addresses`：

```py
persons_addresses = {'Bill': '6565 Monty Road', 
                     'Pamela': '45 Python street'}
persons_addresses['Pamela']
'45 Python street'
```

## 避免使用通用名称

如果你的代码没有构建新的抽象数据类型，即使是对于局部变量，使用`list`、`dict`、`sequence`或`elements`等术语也是有害的。这使得代码难以阅读、理解和使用。还必须避免使用内置名称，以避免在当前命名空间中遮蔽它。通用动词也应该避免使用，除非它们在命名空间中有意义。

相反，应该使用特定于域的术语：

```py
def compute(data):  # too generic
    for element in data:
        yield element ** 2

def squares(numbers):  # better
    for number in numbers:
        yield number ** 2
```

还有一些前缀和后缀的列表，尽管它们在编程中非常常见，但实际上在函数和类名称中应该避免使用：

+   管理器

+   对象

+   做、处理或执行

原因是它们模糊、含糊不清，并且对实际名称没有任何价值。Discourse 和 Stack Overflow 的联合创始人 Jeff Atwood 在这个主题上有一篇非常好的文章，可以在他的博客上找到[`blog.codinghorror.com/i-shall-call-it-somethingmanager/`](http://blog.codinghorror.com/i-shall-call-it-somethingmanager/)。

还有一些应该避免的包名称。一切不能给出关于其内容的线索的东西在长期内都会对项目造成很大的伤害。诸如`misc`、`tools`、`utils`、`common`或`core`这样的名称有很强的倾向成为各种无关代码片段的无尽袋，质量非常差，似乎呈指数增长。在大多数情况下，这样一个模块的存在是懒惰或者没有足够的设计努力的标志。这些模块名称的爱好者可以简单地预见未来，并将它们重命名为`trash`或`dumpster`，因为这正是他们的队友最终会对待这样的模块的方式。

在大多数情况下，拥有更多小模块几乎总是更好的，即使内容很少，但名称很好地反映了内部内容。老实说，像`utils`和`common`这样的名称并没有本质上的错误，而且可以负责任地使用它们。但现实表明，在许多情况下，它们反而成为危险的结构反模式的替代品，这些反模式扩散得非常快。如果你不够快地采取行动，你可能永远无法摆脱它们。因此，最好的方法就是简单地避免这种风险的组织模式，并在项目中由其他人引入时及时制止它们。

## 避免现有名称

在上下文中使用已经存在的名称是不好的做法，因为它会使阅读，特别是调试非常混乱：

```py
>>> def bad_citizen():
...     os = 1
...     import pdb; pdb.set_trace()
...     return os
...
>>> bad_citizen()
> <stdin>(4)bad_citizen()
(Pdb) os
1
(Pdb) import os
(Pdb) c
<module 'os' from '/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/os.pyc'>

```

在这个例子中，`os`名称被代码遮蔽了。应该避免使用内置的和标准库中的模块名称。

尝试创建原始名称，即使它们只在上下文中使用。对于关键字，使用尾随下划线是避免冲突的一种方法：

```py
def xapian_query(terms, or_=True):
    """if or_ is true, terms are combined with the OR clause"""
    ...
```

注意，`class`通常被`klass`或`cls`替换：

```py
def factory(klass, *args, **kwargs):
    return klass(*args, **kwargs)
```

# 参数的最佳实践

函数和方法的签名是代码完整性的守护者。它们驱动其使用并构建其 API。除了我们之前看到的命名规则之外，对参数还必须特别小心。这可以通过三条简单的规则来实现：

+   通过迭代设计构建参数

+   相信参数和你的测试

+   谨慎使用`*args`和`**kwargs`魔术参数

## 通过迭代设计构建参数

为每个函数拥有一个固定和明确定义的参数列表使代码更加健壮。但这在第一个版本中无法完成，因此参数必须通过迭代设计构建。它们应反映元素创建的精确用例，并相应地发展。

例如，当添加一些参数时，应尽可能使用默认值，以避免任何回归：

```py
class Service:  # version 1
    def _query(self, query, type):
        print('done')

    def execute(self, query):
        self._query(query, 'EXECUTE')

>>> Service().execute('my query')
done

import logging

class Service(object):  # version 2
    def _query(self, query, type, logger):
        logger('done')

    def execute(self, query, logger=logging.info):
        self._query(query, 'EXECUTE', logger)

>>> Service().execute('my query')    # old-style call
>>> Service().execute('my query', logging.warning)
WARNING:root:done

```

当需要更改公共元素的参数时，应使用逐步废弃的过程，该过程将在本节后面介绍。

## 相信参数和你的测试

鉴于 Python 的动态类型特性，一些开发人员在其函数和方法顶部使用断言来确保参数具有适当的内容：

```py
def division(dividend, divisor):
    assert isinstance(dividend, (int, float))
    assert isinstance(divisor, (int, float))
    return dividend / divisor

>>> division(2, 4)
0.5
>>> division(2, None)
Traceback (most recent call last):
 **File "<input>", line 1, in <module>
 **File "<input>", line 3, in division
AssertionError

```

这通常是由习惯于静态类型的开发人员完成的，他们觉得 Python 中缺少了一些东西。

检查参数的这种方式是**契约设计**（**DbC**，参见[`en.wikipedia.org/wiki/Design_By_Contract`](http://en.wikipedia.org/wiki/Design_By_Contract)）编程风格的一部分，在实际运行代码之前检查前置条件。

这种方法的两个主要问题是：

+   DbC 的代码解释了它应该如何使用，使其不太可读

+   这可能会使其变慢，因为每次调用都会进行断言

后者可以通过解释器的`"-O"`选项避免。在这种情况下，所有断言都会在生成字节码之前从代码中删除，因此检查会丢失。

无论如何，断言都必须小心使用，并且不应该用来将 Python 弯曲成静态类型的语言。唯一的用例是保护代码免受无意义的调用。

健康的测试驱动开发风格在大多数情况下提供了健壮的基础代码。在这里，功能和单元测试验证了代码创建的所有用例。

当库中的代码被外部元素使用时，进行断言可能是有用的，因为传入的数据可能会破坏事物，甚至造成损害。这种情况发生在处理数据库或文件系统的代码中。

另一种方法是**模糊测试**（`http://en.wikipedia.org/wiki/Fuzz_testing`），其中随机数据片段被发送到程序以检测其弱点。当发现新的缺陷时，可以修复代码以解决该问题，并编写新的测试。

让我们确保遵循 TDD 方法的代码库朝着正确的方向发展，并且在每次出现新的失败时进行调整，从而变得越来越健壮。当以正确的方式完成时，测试中的断言列表在某种程度上类似于前置条件的列表。

## 谨慎使用`*args`和`**kwargs`魔术参数

`*args`和`**kwargs`参数可能会破坏函数或方法的健壮性。它们使签名变得模糊，代码往往开始构建一个小的参数解析器，而不应该这样做：

```py
def fuzzy_thing(**kwargs):

    if 'do_this' in kwargs:
        print('ok i did')

    if 'do_that' in kwargs:
        print('that is done')

    print('errr... ok')

>>> fuzzy_thing(do_this=1)
ok i did
errr... ok
>>> fuzzy_thing(do_that=1)
that is done
errr... ok
>>> fuzzy_thing(hahaha=1)
errr... ok

```

如果参数列表变得又长又复杂，很容易添加魔术参数。但这更多地表明了一个弱函数或方法，应该被拆分或重构。

当`*args`用于处理在函数中以相同方式处理的元素序列时，最好要求一个唯一的容器参数，比如`iterator`：

```py
def sum(*args):  # okay
    total = 0
    for arg in args:
        total += arg
    return total

def sum(sequence):  # better!
    total = 0
    for arg in sequence:
        total += arg
    return total
```

对于`**kwargs`，同样适用相同的规则。最好修复命名参数，使方法的签名有意义：

```py
def make_sentence(**kwargs):
    noun = kwargs.get('noun', 'Bill')
    verb = kwargs.get('verb', 'is')
    adj = kwargs.get('adjective', 'happy')
    return '%s %s %s' % (noun, verb, adj)

def make_sentence(noun='Bill', verb='is', adjective='happy'):
    return '%s %s %s' % (noun, verb, adjective)
```

另一种有趣的方法是创建一个容器类，将几个相关参数分组以提供执行上下文。这种结构不同于`*args`或`**kwargs`，因为它可以提供在值上工作的内部，并且可以独立地发展。使用它作为参数的代码将不必处理其内部。

例如，传递给函数的 Web 请求通常由类的实例表示。这个类负责保存 Web 服务器传递的数据：

```py
def log_request(request):  # version 1
    print(request.get('HTTP_REFERER', 'No referer'))

def log_request(request):  # version 2
    print(request.get('HTTP_REFERER', 'No referer'))
    print(request.get('HTTP_HOST', 'No host'))
```

有时无法避免使用魔术参数，特别是在元编程中。例如，在创建适用于任何类型签名的函数的装饰器时，它们是不可或缺的。更全局地，在任何处理未知数据只是遍历函数的地方，魔术参数都很棒：

```py
import logging

def log(**context):
    logging.info('Context is:\n%s\n' % str(context))
```

# 类名

类的名称必须简洁，准确，以便从中了解类的作用。一个常见的做法是使用一个后缀来通知其类型或性质，例如：

+   **SQL**Engine

+   **Mime**Types

+   **String**Widget

+   **Test**Case

对于基类或抽象类，可以使用**Base**或**Abstract**前缀，如下所示：

+   **Base**Cookie

+   **Abstract**Formatter

最重要的是要与类属性保持一致。例如，尽量避免类和其属性名称之间的冗余：

```py
>>> SMTP.smtp_send()  # redundant information in the namespace
>>> SMTP.send()       # more readable and mnemonic** 

```

# 模块和包名称

模块和包名称告知其内容的目的。名称要简短，小写，不带下划线：

+   `sqlite`

+   `postgres`

+   `sha1`

如果它们实现了一个协议，通常会在后缀加上`lib`：

```py
import smtp**lib
import url**lib
import telnet**lib

```

它们还需要在命名空间内保持一致，以便更容易使用：

```py
from widgets.stringwidgets import TextWidget  # bad
from widgets.strings import TextWidget        # better
```

同样，始终避免使用与标准库中模块之一的相同名称。

当一个模块变得复杂，并包含许多类时，最好的做法是创建一个包，并将模块的元素拆分到其他模块中。

`__init__`模块也可以用于将一些 API 放回顶层，因为它不会影响其使用，但会帮助重新组织代码为更小的部分。例如，考虑一个`foo`包中的`__init__`模块，其内容如下：

```py
from .module1 import feature1, feature2
from .module2 import feature3
```

这将允许用户直接导入功能，如下面的代码所示：

```py
from foo import feature1, feature2, feature3

```

但要注意，这可能会增加循环依赖的机会，并且添加到`__init__`模块中的代码将被实例化。因此要小心使用。

# 有用的工具

以前的惯例和做法的一部分可以通过以下工具来控制和解决：

+   **Pylint**：这是一个非常灵活的源代码分析器

+   **pep8**和**flake8**：这是一个小的代码风格检查器，以及一个添加了一些更有用功能的包装器，如静态分析和复杂度测量

## Pylint

除了一些质量保证指标外，Pylint 还允许您检查给定源代码是否遵循命名约定。其默认设置对应于 PEP 8，Pylint 脚本提供了一个 shell 报告输出。

要安装 Pylint，可以使用`pip`：

```py
$ pip install pylint

```

完成此步骤后，该命令可用，并且可以针对一个模块或多个模块使用通配符运行。让我们在 Buildout 的`bootstrap.py`脚本上尝试一下：

```py
$ wget -O bootstrap.py https://bootstrap.pypa.io/bootstrap-buildout.py -q
$ pylint bootstrap.py
No config file found, using default configuration
************* Module bootstrap
C: 76, 0: Unnecessary parens after 'print' keyword (superfluous-parens)
C: 31, 0: Invalid constant name "tmpeggs" (invalid-name)
C: 33, 0: Invalid constant name "usage" (invalid-name)
C: 45, 0: Invalid constant name "parser" (invalid-name)
C: 74, 0: Invalid constant name "options" (invalid-name)
C: 74, 9: Invalid constant name "args" (invalid-name)
C: 84, 4: Import "from urllib.request import urlopen" should be placed at the top of the module (wrong-import-position)

...

Global evaluation
-----------------
Your code has been rated at 6.12/10

```

真正的 Pylint 输出有点长，在这里被截断了。

请注意，Pylint 可能会给出不良评价或投诉。例如，在某些情况下，模块本身的代码没有使用的导入语句是完全可以的（在命名空间中可用）。

调用使用 mixedCase 方法的库也可能降低您的评级。无论如何，全局评估并不重要。Pylint 只是一个指出可能改进的工具。

调整 Pylint 的第一件事是在项目目录中创建一个`.pylinrc`配置文件，使用`–generate-rcfile`选项：

```py
$ pylint --generate-rcfile > .pylintrc

```

这个配置文件是自我记录的（每个可能的选项都有注释描述），并且应该已经包含了每个可用的配置选项。

除了检查是否符合某些武断的编码标准外，Pylint 还可以提供有关整体代码质量的其他信息，例如：

+   代码重复度量

+   未使用的变量和导入

+   缺少函数、方法或类的文档字符串

+   函数签名太长

默认启用的可用检查列表非常长。重要的是要知道，其中一些规则是武断的，并不容易适用于每个代码库。记住，一致性总是比遵守某些武断标准更有价值。幸运的是，Pylint 非常可调，所以如果您的团队使用的一些命名和编码约定与默认情况下假定的不同，您可以轻松配置它以检查这些约定的一致性。

## pep8 和 flake8

`pep8`是一个只有一个目的的工具：它只提供对 PEP 8 标准的代码约定的样式检查。这是与 Pylint 的主要区别，后者具有许多其他功能。这是对于那些只对 PEP 8 标准的自动化代码样式检查感兴趣的程序员来说是最佳选择，而无需进行任何额外的工具配置，就像 Pylint 的情况一样。

`pep8`可以通过`pip`安装：

```py
$ pip install pep8

```

当在 Buildout 的`bootstrap.py`脚本上运行时，它将给出一份代码样式违规的简短列表：

```py
$ wget -O bootstrap.py https://bootstrap.pypa.io/bootstrap-buildout.py -q
$ pep8 bootstrap.py
bootstrap.py:118:1: E402 module level import not at top of file
bootstrap.py:119:1: E402 module level import not at top of file
bootstrap.py:190:1: E402 module level import not at top of file
bootstrap.py:200:1: E402 module level import not at top of file

```

与 Pylint 输出的主要区别在于其长度。`pep8`只集中在样式上，因此它不提供任何其他警告，例如未使用的变量、函数名太长或缺少文档字符串。它也不提供任何评级。这确实是有道理的，因为没有部分一致性这样的事情。任何，甚至是最轻微的，违反样式指南的行为都会立即使代码不一致。

`pep8`的输出比 Pylint 更简单，更容易解析，因此如果您想将其与一些持续集成解决方案（如 Jenkins）集成，它可能是更好的选择。如果您缺少一些静态分析功能，还有`flake8`包，它是`pep8`和其他几个工具的包装器，易于扩展，并提供更广泛的功能套件：

+   McCabe 复杂度测量

+   通过`pyflakes`进行静态分析

+   使用注释禁用整个文件或单行

# 摘要

本章通过指向官方 Python 样式指南（PEP 8 文档）来解释了最受欢迎的编码约定。官方样式指南还补充了一些命名建议，这些建议将使您未来的代码更加明确，还有一些在保持代码风格一致方面不可或缺的有用工具。

所有这些都为我们准备了本书的第一个实际主题——编写和分发软件包。在下一章中，我们将学习如何在公共 PyPI 存储库上发布我们自己的软件包，以及如何在您的私人组织中利用包装生态系统的力量。
