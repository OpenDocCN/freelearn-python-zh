# 第八章：元类-使类（而不是实例）更智能

前几章已经向我们展示了如何使用装饰器修改类和函数。但这并不是修改或扩展类的唯一选项。在创建类之前修改你的类的更高级的技术是使用**元类**。这个名字已经暗示了它可能是什么；元类是一个包含有关类的元信息的类。

元类的基本前提是在定义时为你生成另一个类的类，因此通常你不会用它来改变类实例，而只会用它来改变类定义。通过改变类定义，可以自动向类添加一些属性，验证是否设置了某些属性，改变继承关系，自动将类注册到管理器，并做许多其他事情。

尽管元类通常被认为是比（类）装饰器更强大的技术，但实际上它们在可能性上并没有太大的区别。选择通常取决于方便性或个人偏好。

本章涵盖了以下主题：

+   基本的动态类创建

+   带参数的元类

+   类创建的内部工作原理，操作顺序

+   抽象基类、示例和内部工作原理

+   使用元类的自动插件系统

+   存储类属性的定义顺序

# 动态创建类

元类是在 Python 中创建新类的工厂。实际上，即使你可能不知道，Python 在你创建一个类时总是会执行`type`元类。

在以程序方式创建类时，`type`元类被用作一个函数。这个函数接受三个参数：`name`，`bases`和`dict`。`name`将成为`__name__`属性，`bases`是继承的基类列表，将存储在`__bases__`中，`dict`是包含所有变量的命名空间字典，将存储在`__dict__`中。

应该注意`type()`函数还有另一个用途。根据之前记录的参数，它会根据这些规格创建一个类。给定一个类实例的单个参数，它也会返回该类，但是从实例中返回。你下一个问题可能是，“如果我在类定义而不是类实例上调用`type()`会发生什么？”嗯，这会返回类的元类，默认为`type`。

让我们用几个例子来澄清这一点：

```py
>>> class Spam(object):
>>>     eggs = 'my eggs'

>>> Spam = type('Spam', (object,), dict(eggs='my eggs'))

```

前两个`Spam`的定义完全相同；它们都创建了一个具有`eggs`和`object`作为基类的类。让我们测试一下这是否像你期望的那样工作：

```py
>>> class Spam(object):
...     eggs = 'my eggs'

>>> spam = Spam()
>>> spam.eggs
'my eggs'
>>> type(spam)
<class '…Spam'>
>>> type(Spam)
<class 'type'>

>>> Spam = type('Spam', (object,), dict(eggs='my eggs'))

>>> spam = Spam()
>>> spam.eggs
'my eggs'
>>> type(spam)
<class '...Spam'>
>>> type(Spam)
<class 'type'>

```

如预期的那样，这两个结果是相同的。在创建类时，Python 会悄悄地添加`type`元类，而`custom`元类只是继承`type`的类。一个简单的类定义有一个隐式的元类，使得一个简单的定义如下：

```py
class Spam(object):
 **pass

```

本质上与：

```py
class Spam(object, metaclass=type):
 **pass

```

这引发了一个问题，即如果每个类都是由一个（隐式的）元类创建的，那么`type`的元类是什么？这实际上是一个递归定义；`type`的元类是`type`。这就是自定义元类的本质：一个继承了 type 的类，允许在不需要修改类定义本身的情况下修改类。

## 一个基本的元类

由于元类可以修改任何类属性，你可以做任何你想做的事情。在我们继续讨论更高级的元类之前，让我们看一个基本的例子：

```py
# The metaclass definition, note the inheritance of type instead
# of object
>>> class MetaSpam(type):
...
...     # Notice how the __new__ method has the same arguments
...     # as the type function we used earlier?
...     def __new__(metaclass, name, bases, namespace):
...         name = 'SpamCreatedByMeta'
...         bases = (int,) + bases
...         namespace['eggs'] = 1
...         return type.__new__(metaclass, name, bases, namespace)

# First, the regular Spam:
>>> class Spam(object):
...     pass

>>> Spam.__name__
'Spam'
>>> issubclass(Spam, int)
False
>>> Spam.eggs
Traceback (most recent call last):
 **...
AttributeError: type object 'Spam' has no attribute 'eggs'

# Now the meta-Spam
>>> class Spam(object, metaclass=MetaSpam):
...     pass

>>> Spam.__name__
'SpamCreatedByMeta'
>>> issubclass(Spam, int)
True
>>> Spam.eggs
1

```

正如你所看到的，使用元类可以轻松修改类定义的所有内容。这使得它既是一个非常强大又是一个非常危险的工具，因为你可以很容易地引起非常意外的行为。

## 元类的参数

向元类添加参数的可能性是一个鲜为人知但非常有用的特性。在许多情况下，简单地向类定义添加属性或方法就足以检测要做什么，但也有一些情况下更具体的指定是有用的。

```py
>>> class MetaWithArguments(type):
...     def __init__(metaclass, name, bases, namespace, **kwargs):
...         # The kwargs should not be passed on to the
...         # type.__init__
...         type.__init__(metaclass, name, bases, namespace)
...
...     def __new__(metaclass, name, bases, namespace, **kwargs):
...         for k, v in kwargs.items():
...             namespace.setdefault(k, v)
...
...         return type.__new__(metaclass, name, bases, namespace)

>>> class WithArgument(metaclass=MetaWithArguments, spam='eggs'):
...     pass

>>> with_argument = WithArgument()
>>> with_argument.spam
'eggs'

```

这个简单的例子可能没有用，但可能性是存在的。你需要记住的唯一一件事是，为了使其工作，`__new__` 和 `__init__` 方法都需要被扩展。

## 通过类访问元类属性

在使用元类时，可能会感到困惑，注意到类实际上不仅仅是构造类，它实际上在创建时继承了类。举个例子：

```py
>>> class Meta(type):
...
...     @property
...     def spam(cls):
...         return 'Spam property of %r' % cls
...
...     def eggs(self):
...         return 'Eggs method of %r' % self

>>> class SomeClass(metaclass=Meta):
...     pass

>>> SomeClass.spam
"Spam property of <class '...SomeClass'>"
>>> SomeClass().spam
Traceback (most recent call last):
 **...
AttributeError: 'SomeClass' object has no attribute 'spam'

>>> SomeClass.eggs()
"Eggs method of <class '...SomeClass'>"
>>> SomeClass().eggs()
Traceback (most recent call last):
 **...
AttributeError: 'SomeClass' object has no attribute 'eggs'

```

正如前面的例子中所示，这些方法仅适用于 `class` 对象，而不适用于实例。`spam` 属性和 `eggs` 方法无法通过实例访问，但可以通过类访问。我个人认为这种行为没有任何有用的情况，但它确实值得注意。

# 使用 collections.abc 的抽象类

抽象基类模块是 Python 中最有用和最常用的元类示例之一，因为它可以轻松确保类遵循特定接口，而无需进行大量手动检查。我们已经在前几章中看到了一些抽象基类的示例，但现在我们将看看这些抽象基类的内部工作原理和更高级的特性，比如自定义 ABC。

## 抽象类的内部工作原理

首先，让我们演示常规抽象基类的用法：

```py
>>> import abc

>>> class Spam(metaclass=abc.ABCMeta):
...
...     @abc.abstractmethod
...     def some_method(self):
...         raise NotImplemented()

>>> class Eggs(Spam):
...     def some_new_method(self):
...         pass

>>> eggs = Eggs()
Traceback (most recent call last):
 **...
TypeError: Can't instantiate abstract class Eggs with abstract
methods some_method

>>> class Bacon(Spam):
...     def some_method():
...         pass

>>> bacon = Bacon()

```

正如你所看到的，抽象基类阻止我们在继承所有抽象方法之前实例化类。除了常规方法外，还支持 `property`、`staticmethod` 和 `classmethod`。

```py
>>> import abc

>>> class Spam(object, metaclass=abc.ABCMeta):
...     @property
...     @abc.abstractmethod
...     def some_property(self):
...         raise NotImplemented()
...
...     @classmethod
...     @abc.abstractmethod
...     def some_classmethod(cls):
...         raise NotImplemented()
...
...     @staticmethod
...     @abc.abstractmethod
...     def some_staticmethod():
...         raise NotImplemented()
...
...     @abc.abstractmethod
...     def some_method():
...         raise NotImplemented()

```

那么 Python 在内部做了什么呢？当然，你可以阅读 `abc.py` 源代码，但我认为简单的解释会更好。

首先，`abc.abstractmethod` 将 `__isabstractmethod__` 属性设置为 `True`。因此，如果你不想使用装饰器，你可以简单地模拟这种行为，做一些类似的事情：

```py
some_method.__isabstractmethod__ = True

```

在那之后，`abc.ABCMeta` 元类遍历命名空间中的所有项目，并查找 `__isabstractmethod__` 属性评估为 `True` 的对象。除此之外，它还遍历所有基类，并检查每个基类的 `__abstractmethods__` 集合，以防类继承了一个 `abstract` 类。所有 `__isabstractmethod__` 仍然评估为 `True` 的项目都被添加到 `__abstractmethods__` 集合中，该集合存储在类中作为 `frozenset`。

### 注意

请注意，我们不使用 `abc.abstractproperty`、`abc.abstractclassmethod` 和 `abc.abstractstaticmethod`。自 Python 3.3 起，这些已被弃用，因为 `classmethod`、`staticmethod` 和 `property` 装饰器被 `abc.abstractmethod` 所识别，因此简单的 `property` 装饰器后跟 `abc.abstractmethod` 也被识别。在对装饰器进行排序时要小心；`abc.abstractmethod` 需要是最内层的装饰器才能正常工作。

现在的问题是实际的检查在哪里进行；检查类是否完全实现。这实际上是通过一些 Python 内部功能实现的：

```py
>>> class AbstractMeta(type):
...     def __new__(metaclass, name, bases, namespace):
...         cls = super().__new__(metaclass, name, bases, namespace)
...         cls.__abstractmethods__ = frozenset(('something',))
...         return cls

>>> class Spam(metaclass=AbstractMeta):
...     pass

>>> eggs = Spam()
Traceback (most recent call last):
 **...
TypeError: Can't instantiate abstract class Spam with ...

```

我们可以很容易地自己使用 `metaclass` 模拟相同的行为，但应该注意 `abc.ABCMeta` 实际上做了更多，我们将在下一节中进行演示。为了模仿内置抽象基类支持的行为，看看下面的例子：

```py
>>> import functools

>>> class AbstractMeta(type):
...     def __new__(metaclass, name, bases, namespace):
...         # Create the class instance
...         cls = super().__new__(metaclass, name, bases, namespace)
...
...         # Collect all local methods marked as abstract
...         abstracts = set()
...         for k, v in namespace.items():
...             if getattr(v, '__abstract__', False):
...                 abstracts.add(k)
...
...         # Look for abstract methods in the base classes and add
...         # them to the list of abstracts
...         for base in bases:
...             for k in getattr(base, '__abstracts__', ()):
...                 v = getattr(cls, k, None)
...                 if getattr(v, '__abstract__', False):
...                     abstracts.add(k)
...
...         # store the abstracts in a frozenset so they cannot be
...         # modified
...         cls.__abstracts__ = frozenset(abstracts)
...
...         # Decorate the __new__ function to check if all abstract
...         # functions were implemented
...         original_new = cls.__new__
...         @functools.wraps(original_new)
...         def new(self, *args, **kwargs):
...             for k in self.__abstracts__:
...                 v = getattr(self, k)
...                 if getattr(v, '__abstract__', False):
...                     raise RuntimeError(
...                         '%r is not implemented' % k)
...
...             return original_new(self, *args, **kwargs)
...
...         cls.__new__ = new
...         return cls

>>> def abstractmethod(function):
...     function.__abstract__ = True
...     return function

>>> class Spam(metaclass=AbstractMeta):
...     @abstractmethod
...     def some_method(self):
...         pass

# Instantiating the function, we can see that it functions as the
# regular ABCMeta does
>>> eggs = Spam()
Traceback (most recent call last):
 **...
RuntimeError: 'some_method' is not implemented

```

实际的实现要复杂一些，因为它仍然需要处理旧式类和`property`、`classmethod` 和 `staticmethod` 类型的方法。此外，它还具有缓存功能，但这段代码涵盖了实现的最有用部分。这里最重要的技巧之一是实际的检查是通过装饰实际类的 `__new__` 函数来执行的。这个方法在类中只执行一次，所以我们可以避免为多个实例化添加这些检查的开销。

### 注意

抽象方法的实际实现可以通过在 Python 源代码中查找 `Objects/descrobject.c`、`Objects/funcobject.c` 和 `Objects/object.c` 文件中的 `__isabstractmethod__` 来找到。实现的 Python 部分可以在 `Lib/abc.py` 中找到。

## 自定义类型检查

当然，使用抽象基类来定义自己的接口是很好的。但是告诉 Python 你的类实际上类似于什么样的类型也是非常方便的。为此，`abc.ABCMeta` 提供了一个注册函数，允许你指定哪些类型是相似的。例如，一个自定义的列表将列表类型视为相似的：

```py
>>> import abc

>>> class CustomList(abc.ABC):
...     'This class implements a list-like interface'
...     pass

>>> CustomList.register(list)
<class 'list'>

>>> issubclass(list, CustomList)
True
>>> isinstance([], CustomList)
True
>>> issubclass(CustomList, list)
False
>>> isinstance(CustomList(), list)
False

```

正如最后四行所示，这是一个单向关系。反过来通常很容易通过继承列表来实现，但在这种情况下不起作用。`abc.ABCMeta` 拒绝创建继承循环。

```py
>>> import abc

>>> class CustomList(abc.ABC, list):
...     'This class implements a list-like interface'
...     pass

>>> CustomList.register(list)
Traceback (most recent call last):
 **...
RuntimeError: Refusing to create an inheritance cycle

```

为了能够处理这样的情况，`abc.ABCMeta` 中还有另一个有用的特性。在子类化 `abc.ABCMeta` 时，可以扩展 `__subclasshook__` 方法来定制 `issubclass` 和 `isinstance` 的行为。

```py
>>> import abc

>>> class UniversalClass(abc.ABC):
...    @classmethod
...    def __subclasshook__(cls, subclass):
...        return True

>>> issubclass(list, UniversalClass)
True
>>> issubclass(bool, UniversalClass)
True
>>> isinstance(True, UniversalClass)
True
>>> issubclass(UniversalClass, bool)
False

```

`__subclasshook__` 应该返回 `True`、`False` 或 `NotImplemented`，这将导致 `issubclass` 返回 `True`、`False` 或在引发 `NotImplemented` 时的通常行为。

## 在 Python 3.4 之前使用 abc.ABC

我们在本段中使用的 `abc.ABC` 类仅在 Python 3.4 及更高版本中可用，但在旧版本中实现它是微不足道的。它只是 `metaclass=abc.ABCMeta` 的语法糖。要自己实现它，你可以简单地使用以下代码片段：

```py
import abc

class ABC(metaclass=abc.ABCMeta):
    pass
```

# 自动注册插件系统

元类最常见的用途之一是让类自动注册为插件/处理程序。这些示例可以在许多项目中看到，比如 Web 框架。这些代码库太庞大了，在这里无法有用地解释。因此，我们将展示一个更简单的例子，展示元类作为自注册的 `plugin` 系统的强大功能：

```py
>>> import abc

>>> class Plugins(abc.ABCMeta):
...     plugins = dict()
...
...     def __new__(metaclass, name, bases, namespace):
...         cls = abc.ABCMeta.__new__(metaclass, name, bases,
...                                   namespace)
...         if isinstance(cls.name, str):
...             metaclass.plugins[cls.name] = cls
...         return cls
...
...     @classmethod
...     def get(cls, name):
...         return cls.plugins[name]

>>> class PluginBase(metaclass=Plugins):
...     @property
...     @abc.abstractmethod
...     def name(self):
...         raise NotImplemented()

>>> class SpamPlugin(PluginBase):
...     name = 'spam'

>>> class EggsPlugin(PluginBase):
...     name = 'eggs'

>>> Plugins.get('spam')
<class '...SpamPlugin'>
>>> Plugins.plugins
{'spam': <class '...SpamPlugin'>,
 **'eggs': <class '...EggsPlugin'>}

```

当然，这个例子有点简单，但它是许多插件系统的基础。这是在实现这样的系统时需要注意的一个非常重要的事情；然而，尽管元类在定义时运行，模块仍然需要被导入才能工作。有几种选项可以做到这一点；通过 `get` 方法进行按需加载是我的选择，因为这样即使插件没有被使用也不会增加加载时间。

以下示例将使用以下文件结构以获得可重现的结果。所有文件将包含在一个名为 plugins 的目录中。

`__init__.py` 文件用于创建快捷方式，因此简单的导入 plugins 将导致 `plugins.Plugins` 可用，而不需要显式导入 `plugins.base`。

```py
# plugins/__init__.py
from .base import Plugin
from .base import Plugins

__all__ = ['Plugin', 'Plugins']
```

包含 `Plugins` 集合和 `Plugin` 基类的 `base.py` 文件：

```py
# plugins/base.py
import abc

class Plugins(abc.ABCMeta):
    plugins = dict()

    def __new__(metaclass, name, bases, namespace):
        cls = abc.ABCMeta.__new__(
            metaclass, name, bases, namespace)
        if isinstance(cls.name, str):
            metaclass.plugins[cls.name] = cls
        return cls

    @classmethod
    def get(cls, name):
        return cls.plugins[name]

class Plugin(metaclass=Plugins):
    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplemented()
```

和两个简单的插件，`spam.py`：

```py
from . import base

class Spam(base.Plugin):
    name = 'spam'
```

和 `eggs.py`：

```py
from . import base

class Eggs(base.Plugin):
    name = 'eggs'
```

## 按需导入插件

解决导入问题的第一个解决方案是在 `Plugins` 元类的 `get` 方法中处理它。每当在注册表中找不到插件时，它应该自动从 `plugins` 目录加载模块。

这种方法的优势在于，不仅插件不需要显式预加载，而且只有在需要时才加载插件。未使用的插件不会被触及，因此这种方法有助于减少应用程序的加载时间。

缺点是代码不会被运行或测试，所以它可能完全失效，直到最终加载时你才会知道。这个问题的解决方案将在测试章节中介绍，第十章，*测试和日志 - 为错误做准备*。另一个问题是，如果代码自注册到应用程序的其他部分，那么该代码也不会被执行。

修改`Plugins.get`方法，我们得到以下结果：

```py
import abc
import importlib

class Plugins(abc.ABCMeta):
    plugins = dict()

    def __new__(metaclass, name, bases, namespace):
        cls = abc.ABCMeta.__new__(
            metaclass, name, bases, namespace)
        if isinstance(cls.name, str):
            metaclass.plugins[cls.name] = cls
        return cls

    @classmethod
    def get(cls, name):
        if name not in cls.plugins:
            print('Loading plugins from plugins.%s' % name)
            importlib.import_module('plugins.%s' % name)
        return cls.plugins[name]
```

执行时会得到以下结果：

```py
>>> import plugins
>>> plugins.Plugins.get('spam')
Loading plugins from plugins.spam
<class 'plugins.spam.Spam'>

>>> plugins.Plugins.get('spam')
<class 'plugins.spam.Spam'>

```

正如你所看到的，这种方法只会导入一次`import`。第二次，插件将在插件字典中可用，因此不需要加载。

## 通过配置导入插件

通常只加载所需的插件是一个更好的主意，但预加载可能需要的插件也有其优点。显式比隐式更好，显式加载插件列表通常是一个很好的解决方案。这种方法的附加优势是，首先你可以使注册更加先进，因为你保证它被运行，其次你可以从多个包中加载插件。

在`get`方法中，我们将这次添加一个`load`方法；一个导入所有给定模块名称的`load`方法：

```py
import abc
import importlib

class Plugins(abc.ABCMeta):
    plugins = dict()

    def __new__(metaclass, name, bases, namespace):
        cls = abc.ABCMeta.__new__(
            metaclass, name, bases, namespace)
        if isinstance(cls.name, str):
            metaclass.plugins[cls.name] = cls
        return cls

    @classmethod
    def get(cls, name):
        return cls.plugins[name]

    @classmethod
    def load(cls, *plugin_modules):
        for plugin_module in plugin_modules:
            plugin = importlib.import_module(plugin_module)
```

可以使用以下代码调用：

```py
>>> import plugins

>>> plugins.Plugins.load(
...     'plugins.spam',
...     'plugins.eggs',
... )

>>> plugins.Plugins.get('spam')
<class 'plugins.spam.Spam'>

```

一个相当简单和直接的系统，根据设置加载插件，这可以很容易地与任何类型的设置系统结合使用来填充`load`方法。

## 通过文件系统导入插件

在可能的情况下，最好避免让系统依赖于文件系统上模块的自动检测，因为这直接违反了`PEP8`。特别是，“显式比隐式更好”。虽然这些系统在特定情况下可以正常工作，但它们经常会使调试变得更加困难。在 Django 中类似的自动导入系统给我带来了不少头疼，因为它们往往会混淆错误。话虽如此，基于插件目录中所有文件的自动插件加载仍然是一个值得演示的可能性。

```py
import os
import re
import abc
import importlib

MODULE_NAME_RE = re.compile('[a-z][a-z0-9_]*', re.IGNORECASE)

class Plugins(abc.ABCMeta):
    plugins = dict()

    def __new__(metaclass, name, bases, namespace):
        cls = abc.ABCMeta.__new__(
            metaclass, name, bases, namespace)
        if isinstance(cls.name, str):
            metaclass.plugins[cls.name] = cls
        return cls

    @classmethod
    def get(cls, name):
        return cls.plugins[name]

    @classmethod
    def load_directory(cls, module, directory):
        for file_ in os.listdir(directory):
            name, ext = os.path.splitext(file_)
            full_path = os.path.join(directory, file_)
            import_path = [module]
            if os.path.isdir(full_path):
                import_path.append(file_)
            elif ext == '.py' and MODULE_NAME_RE.match(name):
                import_path.append(name)
            else:
                # Ignoring non-matching files/directories
                continue

            plugin = importlib.import_module('.'.join(import_path))

    @classmethod
    def load(cls, **plugin_directories):
        for module, directory in plugin_directories.items():
            cls.load_directory(module, directory)
```

如果可能的话，我会尽量避免使用完全自动的导入系统，因为它很容易出现意外错误，并且会使调试变得更加困难，更不用说导入顺序无法轻松地通过这种方式进行控制。为了使这个系统变得更加智能（甚至导入 Python 路径之外的包），你可以使用`importlib.abc`中的抽象基类创建一个插件加载器。请注意，你很可能仍然需要通过`os.listdir`或`os.walk`列出目录。

# 实例化类时的操作顺序

在调试动态创建和/或修改的类时，类实例化的操作顺序非常重要。类的实例化按以下顺序进行。

## 查找元类

元类来自于类的显式给定的元类或`bases`，或者使用默认的`type`元类。

对于每个类，类本身和 bases，将使用以下匹配的第一个：

+   显式给定的元类

+   从 bases 中显式元类

+   `type()`

### 注意

请注意，如果找不到是所有候选元类的子类型的元类，将引发`TypeError`。这种情况发生的可能性不太大，但在使用多重继承/混入元类时肯定是可能的。

## 准备命名空间

通过之前选择的元类准备类命名空间。如果元类有一个`__prepare__`方法，它将被调用`namespace = metaclass.__prepare__(names, bases, **kwargs)`，其中`**kwargs`来自类定义。如果没有`__prepare__`方法可用，结果将是`namespace = dict()`。

请注意，有多种实现自定义命名空间的方法，正如我们在前一段中看到的，`type()`函数调用还接受一个`dict`参数，也可以用于修改命名空间。

## 执行类主体

类的主体执行方式与普通代码执行非常相似，但有一个关键区别，即单独的命名空间。由于类有一个单独的命名空间，不应该污染`globals()/locals()`命名空间，因此在该上下文中执行。结果调用看起来像这样：`exec(body, globals(), namespace)`，其中`namespace`是先前生成的命名空间。

## 创建类对象（而不是实例）

现在我们已经准备好所有组件，实际的类对象可以被生成。这是通过`class_ = metaclass(name, bases, namespace, **kwargs)`调用完成的。正如您所看到的，这实际上与之前讨论的`type()`调用完全相同。这里的`**kwargs`与之前传递给`__prepare__`方法的参数相同。

值得注意的是，这也是在`super()`调用中不带参数时将被引用的对象。

## 执行类装饰器

现在类对象实际上已经完成，类装饰器将被执行。由于这仅在类对象中的所有其他内容已经构建完成后执行，因此变得更难修改类属性，例如继承哪些类以及类的名称。通过修改`__class__`对象，您仍然可以修改或覆盖这些内容，但至少更加困难。

## 创建类实例

从先前生成的类对象中，现在我们可以像通常一样创建实际的实例。应该注意的是，与之前的步骤不同，这两个步骤和类装饰器步骤是唯一在每次实例化类时执行的步骤。在这两个步骤之前的步骤只在每个类定义时执行一次。

## 示例

足够的理论！让我们说明创建和实例化类对象的过程，以便检查操作顺序：

```py
>>> import functools

>>> def decorator(name):
...     def _decorator(cls):
...         @functools.wraps(cls)
...         def __decorator(*args, **kwargs):
...             print('decorator(%s)' % name)
...             return cls(*args, **kwargs)
...         return __decorator
...     return _decorator

>>> class SpamMeta(type):
...
...     @decorator('SpamMeta.__init__')
...     def __init__(self, name, bases, namespace, **kwargs):
...         print('SpamMeta.__init__()')
...         return type.__init__(self, name, bases, namespace)
...
...     @staticmethod
...     @decorator('SpamMeta.__new__')
...     def __new__(cls, name, bases, namespace, **kwargs):
...         print('SpamMeta.__new__()')
...         return type.__new__(cls, name, bases, namespace)
...
...     @classmethod
...     @decorator('SpamMeta.__prepare__')
...     def __prepare__(cls, names, bases, **kwargs):
...         print('SpamMeta.__prepare__()')
...         namespace = dict(spam=5)
...         return namespace

>>> @decorator('Spam')
... class Spam(metaclass=SpamMeta):
...
...     @decorator('Spam.__init__')
...     def __init__(self, eggs=10):
...         print('Spam.__init__()')
...         self.eggs = eggs
decorator(SpamMeta.__prepare__)
SpamMeta.__prepare__()
decorator(SpamMeta.__new__)
SpamMeta.__new__()
decorator(SpamMeta.__init__)
SpamMeta.__init__()

# Testing with the class object
>>> spam = Spam
>>> spam.spam
5
>>> spam.eggs
Traceback (most recent call last):
 **...
AttributeError: ... object has no attribute 'eggs'

# Testing with a class instance
>>> spam = Spam()
decorator(Spam)
decorator(Spam.__init__)
Spam.__init__()
>>> spam.spam
5
>>> spam.eggs
10

```

该示例清楚地显示了类的创建顺序：

1.  通过`__prepare__`准备命名空间。

1.  使用`__new__`创建类主体。

1.  使用`__init__`初始化元类（请注意，这不是类`__init__`）。

1.  通过类装饰器初始化类。

1.  通过类`__init__`函数初始化类。

我们可以从中注意到的一点是，类装饰器在实际实例化类时每次都会执行，而不是在此之前。当然，这既是优点也是缺点，但如果您希望构建所有子类的注册表，那么使用元类肯定更方便，因为装饰器在实例化类之前不会注册。

除此之外，在实际创建类对象（而不是实例）之前修改命名空间的能力也是非常强大的。例如，可以方便地在几个类对象之间共享特定范围，或者轻松确保某些项目始终在范围内可用。

# 按定义顺序存储类属性

有些情况下，定义顺序是有影响的。例如，假设我们正在创建一个表示 CSV（逗号分隔值）格式的类。CSV 格式期望字段有特定的顺序。在某些情况下，这将由标题指示，但保持一致的字段顺序仍然很有用。类似的系统在 ORM 系统（如 SQLAlchemy）中使用，用于存储表定义的列顺序以及在 Django 中的表单中的输入字段顺序。

## 没有元类的经典解决方案

一种简单的存储字段顺序的方法是给字段实例一个特殊的`__init__`方法，每次定义都会增加，因此字段具有递增的索引属性。这种解决方案可以被认为是经典解决方案，因为它在 Python 2 中也适用。

```py
>>> import itertools

>>> class Field(object):
...     counter = itertools.count()
...
...     def __init__(self, name=None):
...         self.name = name
...         self.index = next(Field.counter)
...
...     def __repr__(self):
...         return '<%s[%d] %s>' % (
...             self.__class__.__name__,
...             self.index,
...             self.name,
...         )

>>> class FieldsMeta(type):
...     def __new__(metaclass, name, bases, namespace):
...         cls = type.__new__(metaclass, name, bases, namespace)
...         fields = []
...         for k, v in namespace.items():
...             if isinstance(v, Field):
...                 fields.append(v)
...                 v.name = v.name or k
...
...         cls.fields = sorted(fields, key=lambda f: f.index)
...         return cls

>>> class Fields(metaclass=FieldsMeta):
...     spam = Field()
...     eggs = Field()

>>> Fields.fields
[<Field[0] spam>, <Field[1] eggs>]

>>> fields = Fields()
>>> fields.eggs.index
1
>>> fields.spam.index
0
>>> fields.fields
[<Field[0] spam>, <Field[1] eggs>]

```

为了方便起见，也为了使事情更美观，我们添加了`FieldsMeta`类。这里并不严格需要它，但它会自动填写名称（如果需要的话），并添加包含字段排序列表的`fields`列表。

## 使用元类获取排序的命名空间

前面的解决方案更加直接，并且也支持 Python 2，但是在 Python 3 中我们有更多的选择。正如你在前面的段落中看到的，自从 Python 3 以来，我们有了`__prepare__`方法，它返回命名空间。从前面的章节中，你可能还记得`collections.OrderedDict`，所以让我们看看当我们将它们结合起来会发生什么。

```py
>>> import collections

>>> class Field(object):
...     def __init__(self, name=None):
...         self.name = name
...
...     def __repr__(self):
...         return '<%s %s>' % (
...             self.__class__.__name__,
...             self.name,
...         )

>>> class FieldsMeta(type):
...     @classmethod
...     def __prepare__(metaclass, name, bases):
...         return collections.OrderedDict()
...
...     def __new__(metaclass, name, bases, namespace):
...         cls = type.__new__(metaclass, name, bases, namespace)
...         cls.fields = []
...         for k, v in namespace.items():
...             if isinstance(v, Field):
...                 cls.fields.append(v)
...                 v.name = v.name or k
...
...         return cls

>>> class Fields(metaclass=FieldsMeta):
...     spam = Field()
...     eggs = Field()

>>> Fields.fields
[<Field spam>, <Field eggs>]
>>> fields = Fields()
>>> fields.fields
[<Field spam>, <Field eggs>]
```

正如你所看到的，字段确实按照我们定义的顺序排列。`Spam`在前，`eggs`在后。由于类命名空间现在是`collections.OrderedDict`实例，我们知道顺序是有保证的。而不是 Python `dict`的常规非确定性顺序。这展示了元类在以通用方式扩展类时可以多么方便。元类的另一个重要优势是，与自定义的`__init__`方法不同，如果用户忘记调用父类的`__init__`方法，他们也不会失去功能。元类总是会被执行，除非添加了不同的元类。

# 总结

Python 元类系统是每个 Python 程序员一直在使用的东西，也许甚至不知道。每个类都应该通过某个（子类）`type`来创建，这允许无限的定制和魔法。现在，你可以像平常一样创建类，并在定义期间动态添加、修改或删除类的属性；非常神奇但非常有用。然而，魔法组件也是它应该谨慎使用的原因。虽然元类可以让你的生活变得更轻松，但它们也是产生完全难以理解的代码的最简单方式之一。

尽管如此，元类有一些很好的用例，许多库如`SQLAlchemy`和`Django`都使用元类来使你的代码工作更加轻松，而且可以说更好。实际上，理解内部使用的魔法通常对于使用这些库并不是必需的，这使得这些情况是可以辩护的。问题在于，对于初学者来说，是否值得使用更好的体验来换取一些内部的黑魔法，从这些库的成功来看，我会说在这种情况下是值得的。

总之，当考虑使用元类时，请记住蒂姆·彼得斯曾经说过的话：“元类比 99%的用户应该担心的更深奥。如果你想知道自己是否需要它们，那就不需要。”

现在我们将继续解决一些元类产生的魔法：文档。下一章将向我们展示如何为代码编写文档，如何测试文档，并且最重要的是，如何通过在文档中注释类型来使文档更加智能。
