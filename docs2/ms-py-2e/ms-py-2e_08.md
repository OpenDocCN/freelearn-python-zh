# 8

# 元类 – 使类（非实例）更智能

前面的章节已经向我们展示了如何使用装饰器修改类和函数。但这并不是修改或扩展类的唯一选项。在创建之前修改你的类的一个更高级的技术是使用元类。名称已经给你一个提示，它可能是什么；元类是一个包含有关类的元信息的类。

元类的基本前提是一个在定义时为你生成另一个类的类，所以通常你不会用它来改变类实例，而只会改变类定义。通过改变类定义，可以自动向类添加一些属性，验证某些属性是否已设置，改变继承，自动将类注册到管理器，以及许多其他事情。

虽然元类通常被认为比（类）装饰器更强大的一种技术，但实际上它们在可能性上并没有太大的区别。选择通常取决于便利性或个人偏好。

在本章中，我们将涵盖以下主题：

+   基本动态类创建

+   带参数的元类

+   抽象基类、示例和内部工作原理

+   使用元类的自动插件系统

+   类创建的内部和操作顺序

+   存储类属性的定义顺序

# 动态创建类

元类是 Python 中创建新类的工厂。实际上，即使你可能没有意识到，Python 在创建类时总是会执行`type`元类。

一些元类内部使用的常见示例包括`abc`（抽象基类）、`dataclasses`和 Django 框架，该框架严重依赖于元类来创建`Model`类。

以过程式创建类时，使用`type`元类作为一个接受三个参数的函数：`name`、`bases`和`dict`。`name`将变成`__name__`属性，`bases`是继承的基类列表，并将存储在`__bases__`中，`dict`是包含所有变量的命名空间字典，并将存储在`__dict__`中。

应该注意的是，`type()`函数还有另一个用途。给定上述文档化的参数，它将创建一个具有那些规格的类。给定一个类实例的单个参数（例如，`type(spam)`），它将返回类对象/定义。

你的下一个问题可能是，如果我调用`type()`一个类定义而不是类实例会发生什么？嗯，那返回的是类的元类，默认情况下是`type`。

让我们通过几个示例来澄清这一点：

```py
>>> class Spam(object):
...     eggs = 'my eggs'

>>> Spam = type('Spam', (object,), dict(eggs='my eggs')) 
```

上面的两个`Spam`定义完全相同；它们都创建了一个具有实例属性`eggs`和以`object`为基类的类。让我们测试一下它是否真的像你预期的那样工作：

```py
>>> class Spam(object):
...     eggs = 'my eggs'

>>> spam = Spam()
>>> spam.eggs
'my eggs'
>>> type(spam)
<class ' ...Spam'>
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

如预期的那样，两个的结果是相同的。当创建一个类时，Python 会默默地添加`type`元类，而自定义元类是继承自`type`的类。一个简单的类定义有一个无声的元类，使得一个简单的定义如下：

```py
class Spam(object):
    pass 
```

实质上等同于：

```py
class Spam(object, metaclass=type):
    pass 
```

这引发了以下问题：如果每个类都是由一个（无声的）元类创建的，那么`type`的元类是什么？这是一个递归定义；`type`的元类是`type`。这就是自定义元类的本质：一个继承自`type`的类，允许在不修改类定义本身的情况下修改类。

# 一个基本的元类

由于元类可以修改任何类属性，你可以做任何你想做的事情。在我们继续讨论更高级的元类之前，让我们创建一个执行以下操作的元类：

1.  使类继承`int`

1.  向类添加`lettuce`属性

1.  改变类的名称

首先，我们创建元类。然后，我们创建一个带有和不带有元类的类：

```py
# The metaclass definition, note the inheritance of type instead
# of object
>>> class MetaSandwich(type):
...     # Notice how the __new__ method has the same arguments
...     # as the type function we used earlier?
...     def __new__(metaclass, name, bases, namespace):
...         name = 'SandwichCreatedByMeta'
...         bases = (int,) + bases
...         namespace['lettuce'] = 1
...         return type.__new__(metaclass, name, bases, namespace) 
```

首先，普通的三明治：

```py
>>> class Sandwich(object):
...     pass

>>> Sandwich.__name__
'Sandwich'
>>> issubclass(Sandwich, int)
False
>>> Sandwich.lettuce
Traceback (most recent call last):
    ...
AttributeError: type object 'Sandwich' has no attribute 'lettuce' 
```

现在，元-Sandwich：

```py
>>> class Sandwich(object, metaclass=MetaSandwich):
...     pass

>>> Sandwich.__name__
'SandwichCreatedByMeta'
>>> issubclass(Sandwich, int)
True
>>> Sandwich.lettuce
1 
```

如你所见，现在具有自定义元类的类继承了`int`，具有`lettuce`属性，并且具有不同的名称。

使用元类，你可以修改类定义的任何方面。这使得它们成为一个既非常强大又可能非常令人困惑的工具。通过一些小的修改，你可以在你的（或他人的）代码中引起最奇怪的 bug。

## 元类的参数

向元类添加参数的可能性是一个鲜为人知的功能，但仍然非常有用。在许多情况下，仅仅向类定义添加属性或方法就足以检测要做什么，但有些情况下，更具体一些是有用的：

```py
>>> class AddClassAttributeMeta(type):
...     def __init__(metaclass, name, bases, namespace, **kwargs):
...         # The kwargs should not be passed on to the
...         # type.__init__
...         type.__init__(metaclass, name, bases, namespace)
...
...     def __new__(metaclass, name, bases, namespace, **kwargs):
...         for k, v in kwargs.items():
...             # setdefault so we don't overwrite attributes
...             namespace.setdefault(k, v)
...
...         return type.__new__(metaclass, name, bases, namespace)

>>> class WithArgument(metaclass=AddClassAttributeMeta, a=1234):
...     pass

>>> WithArgument.a
1234
>>> with_argument = WithArgument()
>>> with_argument.a
1234 
```

这个简单的例子可能没有太大用处，但可能性是存在的。例如，一个自动在插件注册表中注册插件的元类可以使用这个特性来指定插件名称别名。

使用这个特性，你不需要将所有创建类的参数作为属性和方法包含在类中，你可以传递这些参数而不污染你的类。你需要记住的唯一一点是，为了使这个功能正常工作，`__new__` 和 `__init__` 方法都需要被扩展，因为参数是传递给元类构造函数（`__init__`）的。

然而，从 Python 3.6 开始，我们已经有了这个效果的更简单替代方案。Python 3.6 引入了`__init_subclass__`魔法方法，它以稍微简单一些的方式允许进行类似的修改：

```py
>>> class AddClassAttribute:
...     def __init_subclass__(cls, **kwargs):
...         super().__init_subclass__()
...
...         for k, v in kwargs.items():
...             setattr(cls, k, v)

>>> class WithAttribute(metaclass=AddClassAttributeMeta, a=1234):
...     pass

>>> WithAttribute.a
1234
>>> with_attribute = WithAttribute()
>>> with_attribute.a
1234 
```

本章中的一些元类可以用`__init_subclass__`方法替换，这对于小的修改来说是一个非常实用的选项。对于更大的更改，我建议使用完整的元类，以便使普通类和元类之间的区别更加明显。

## 通过类访问元类属性

当使用元类时，可能会让人困惑的是，类实际上做的不仅仅是简单地构造类；它实际上在创建过程中继承了类。为了说明：

```py
>>> class Meta(type):
...     @property
...     def some_property(cls):
...         return 'property of %r' % cls
...
...     def some_method(self):
...         return 'method of %r' % self

>>> class SomeClass(metaclass=Meta):
...     pass

# Accessing through the class definition
>>> SomeClass.some_property
"property of <class '...SomeClass'>"
>>> SomeClass.some_method
<bound method Meta.some_method of <class '__main__.SomeClass'>>
>>> SomeClass.some_method()
"method of <class '__main__.SomeClass'>"

# Accessing through an instance
>>> some_class = SomeClass()
>>> some_class.some_property
Traceback (most recent call last):
    ...
AttributeError: 'SomeClass' object has no attribute 'some_property'
>>> some_class.some_method
Traceback (most recent call last):
    ...
AttributeError: 'SomeClass' object has no attribute 'some_method' 
```

如前例所示，这些方法仅对类对象可用，而不是实例。`some_property`和`some_method`不能通过实例访问，而可以通过类访问。这可以用于使某些函数仅对类（而不是实例）可用，并使您的类命名空间更干净。

然而，在一般情况下，我怀疑这只会增加混淆，所以我通常会建议反对这样做。

# 使用 collections.abc 的抽象类

抽象基类（也称为接口类）模块是 Python 中元类最有用和最广泛使用的例子之一，因为它使得确保类遵循特定接口而无需大量手动检查变得容易。我们已经在之前的章节中看到了一些抽象基类的例子，但现在我们还将探讨它们的内部工作原理和一些更高级的功能，例如自定义抽象基类（ABC）。

## 抽象类的内部工作原理

首先，让我们演示常规抽象基类的用法：

```py
>>> import abc

>>> class AbstractClass(metaclass=abc.ABCMeta):
...     @abc.abstractmethod
...     def some_method(self):
...         raise NotImplemented()

>>> class ConcreteClass(AbstractClass):
...     pass

>>> ConcreteClass()
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class ConcreteClass with
abstract methods some_method

>>> class ImplementedConcreteClass(ConcreteClass):
...     def some_method():
...         pass

>>> instance = ImplementedConcreteClass() 
```

如您所见，抽象基类阻止我们实例化类，直到所有抽象方法都被继承。这在您的代码期望某些属性或方法可用，但没有合理的默认值时非常有用。一个常见的例子是与插件和数据模型的基类。

除了常规方法外，还支持`property`、`staticmethod`和`classmethod`：

```py
>>> import abc

>>> class AbstractClass(object, metaclass=abc.ABCMeta):
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

那么 Python 内部是如何做的呢？当然，您可以阅读`abc.py`源代码，但我认为一个简单的解释会更好。

首先，`abc.abstractmethod`将函数的`__isabstractmethod__`属性设置为`True`。所以如果您不想使用装饰器，您可以通过以下类似的方式简单地模拟行为：

```py
some_method.__isabstractmethod__ = True 
```

之后，`abc.ABCMeta`元类遍历`namespace`中的所有项，并查找`__isabstractmethod__`属性评估为`True`的对象。除此之外，它还会遍历所有`bases`，并检查每个基类的`__abstractmethods__`集合，以防类继承自抽象类。所有`__isabstractmethod__`仍然评估为`True`的项都将被添加到存储在类中的`__abstractmethods__`集合中，作为一个`frozenset`。

注意，我们不使用 `abc.abstractproperty`、`abc.abstractclassmethod` 和 `abc.abstractstaticmethod`。从 Python 3.3 开始，这些已经被弃用，因为 `classmethod`、`staticmethod` 和 `property` 装饰器被 `abc.abstractmethod` 识别，所以简单的 `property` 装饰器后面跟着 `abc.abstractmethod` 也会被识别。在排序装饰器时要小心；`abc.abstractmethod` 需要是最内层的装饰器，这样才能正常工作。

接下来的问题是实际检查在哪里，即检查类是否完全实现。这实际上是通过几个 Python 内部机制来实现的：

```py
>>> class AbstractMeta(type):
...     def __new__(metaclass, name, bases, namespace):
...         cls = super().__new__(metaclass, name, bases,
...                               namespace)
...         cls.__abstractmethods__ = frozenset(('something',))
...         return cls

>>> class ConcreteClass(metaclass=AbstractMeta):
...     pass

>>> ConcreteClass()
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class ConcreteClass with 
abstract methods something 
```

我们可以很容易地使用元类自己模拟相同的行为，但应该注意的是，`abc.ABCMeta` 实际上做得更多，我们将在下一节中演示。为了说明上述行为，让我们创建一个模拟 `abc.ABCMeta` 的抽象基类元类：

```py
>>> import functools

>>> class AbstractMeta(type):
...     def __new__(metaclass, name, bases, namespace):
...         # Create the class instance
...         cls = super().__new__(metaclass, name, bases,
...                               namespace)
...
...         # Collect all local methods marked as abstract
...         abstracts = set()
...         for k, v in namespace.items():
...             if getattr(v, '__abstract__', False):
...                 abstracts.add(k)
...
...         # Look for abstract methods in the base classes and
...         # add them to the list of abstracts
...         for base in bases:
...             for k in getattr(base, '__abstracts__', ()):
...                 v = getattr(cls, k, None)
...                 if getattr(v, '__abstract__', False):
...                     abstracts.add(k)
...
...         # Store the abstracts in a frozenset so they cannot be
...         # modified
...         cls.__abstracts__ = frozenset(abstracts)
...
...         # Decorate the __new__ function to check if all
...         # abstract functions were implemented
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

# Create a decorator that sets the '__abstract__' attribute
>>> def abstractmethod(function):
...     function.__abstract__ = True
...     return function 
```

现在我们有了创建抽象类的元类和装饰器，让我们看看它是否按预期工作：

```py
>>> class ConcreteClass(metaclass=AbstractMeta):
...     @abstractmethod
...     def some_method(self):
...         pass

# Instantiating the function, we can see that it functions as the
# regular ABCMeta does
>>> ConcreteClass()
Traceback (most recent call last):
    ...
RuntimeError: 'some_method' is not implemented 
```

实际实现要复杂得多，因为它需要处理如 `property`、`classmethod` 和 `staticmethod` 这样的装饰器。它还有一些缓存特性，但此代码涵盖了实现中最有用的部分。这里需要注意的一个最重要的技巧是，实际的检查是通过装饰实际类的 `__new__` 函数来执行的。此方法在类中只执行一次，因此我们可以避免多次实例化时的检查开销。

抽象方法的实际实现可以通过在以下文件中查找 `__isabstractmethod__` 来找到 Python 源代码：`Objects/descrobject.c`、`Objects/funcobject.c` 和 `Objects/object.c`。实现中的 Python 部分可以在 `Lib/abc.py` 中找到。

## 自定义类型检查

当然，使用抽象基类定义自己的接口是很好的。但也可以很方便地告诉 Python 你的类实际上类似于什么，以及哪些类型是相似的。为此，`abc.ABCMeta` 提供了一个注册函数，允许你指定哪些类型是相似的。例如，一个将 `list` 类型视为相似的定制 `list`：

```py
>>> import abc

>>> class CustomList(abc.ABC):
...     '''This class implements a list-like interface'''

>>> class CustomInheritingList(list, abc.ABC):
...     '''This class implements a list-like interface'''

>>> issubclass(list, CustomList)
False
>>> issubclass(list, CustomInheritingList)
False

>>> CustomList.register(list)
<class 'list'>

# We can't make it go both ways, however
>>> CustomInheritingList.register(list)
Traceback (most recent call last):
    ...
RuntimeError: Refusing to create an inheritance cycle

>>> issubclass(list, CustomList)
True
>>> issubclass(list, CustomInheritingList)
False

# We need to inherit list to make it work the other way around
>>> issubclass(CustomList, list)
False
>>> isinstance(CustomList(), list)
False
>>> issubclass(CustomInheritingList, list)
True
>>> isinstance(CustomInheritingList(), list)
True 
```

如最后八行所示，这是一个单向关系。反过来则需要继承 `list`，但由于继承循环，不能双向进行。否则，`CustomInheritingList` 将继承 `list`，而 `list` 将继承 `CustomInheritingList`，这可能导致在 `issubclass()` 调用期间无限递归。

为了能够处理这些情况，`abc.ABCMeta` 中还有一个有用的特性。当子类化 `abc.ABCMeta` 时，可以扩展 `__subclasshook__` 方法来自定义 `issubclass` 和 `isinstance` 的行为：

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

`__subclasshook__` 应返回 `True`、`False` 或 `NotImplemented`，这将导致 `issubclass` 返回 `True`、`False` 或在返回 `NotImplemented` 时的常规行为。

# 自动注册插件系统

使用元类的一个非常有用方法是将类自动注册为插件/处理器。

而不是在创建类后手动添加注册调用或通过添加装饰器来添加，你可以让它对用户来说完全自动。这意味着你的库或插件系统的用户不会意外忘记添加注册调用。

注意区分注册和导入的区别。虽然这个第一个例子展示了自动注册，但自动导入将在后面的章节中介绍。

这些示例可以在许多项目中看到，例如网络框架。例如，Django 网络框架使用元类来处理其数据库模型（实际上是表），根据类和属性名称自动生成表和列名称。

尽管这些项目的实际代码库过于庞大，无法在此有用地解释，因此我们将展示一个更简单的示例，以展示元类作为自注册插件系统的强大功能：

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

>>> class PluginA(PluginBase):
...     name = 'a'

>>> class PluginB(PluginBase):
...     name = 'b'

>>> Plugins.get('a')
<class '...PluginA'>

>>> Plugins.plugins
{'a': <class '...PluginA'>,
 'b': <class '...PluginB'>} 
```

当然，这个例子有点简单，但它是许多插件系统的基础。

虽然元类在定义时运行，但模块仍然需要被 **导入** 才能工作。有几种方法可以做到这一点；如果可能的话，通过 `get` 方法按需加载将是我投票的选择，因为这样也不会增加插件未使用时的加载时间。

以下示例将使用以下文件结构来获得可重复的结果。所有文件都将包含在 `plugins` 目录中。请注意，本书的所有代码，包括此示例，都可以在 GitHub 上找到：[`github.com/mastering-python/code_2`](https://github.com/mastering-python/code_2)。

`__init__.py` 文件用于创建快捷方式，所以简单的 `import plugins` 将导致 `plugins.Plugins` 可用，而不是需要显式导入 `plugins.base`：

```py
# plugins/__init__.py
from .base import Plugin
from .base import Plugins

__all__ = ['Plugin', 'Plugins'] 
```

下面是包含 `Plugins` 集合和 `Plugin` 基类 的 `base.py` 文件：

```py
# plugins/base.py
import abc

class Plugins(abc.ABCMeta):
    plugins = dict()

    def __new__(metaclass, name, bases, namespace):
        cls = abc.ABCMeta.__new__(
            metaclass, name, bases, namespace)
        metaclass.plugins[name.lower()] = cls
        return cls

    @classmethod
    def get(cls, name):
        return cls.plugins[name]

class Plugin(metaclass=Plugins):
    pass 
```

以及两个简单的插件，`a.py` 和 `b.py`（由于它与 `a.py` 功能上相同，所以被省略）：

```py
from . import base

class A(base.Plugin):
    pass 
```

现在我们已经设置了插件和自动注册，我们需要注意 `a.py` 和 `b.py` 的加载。虽然 `A` 和 `B` 将在 `Plugins` 中自动注册，但如果忘记导入它们，它们将不会注册。为了解决这个问题，我们有几种选择；首先我们将看看按需加载。

## 按需导入插件

解决导入问题的第一个方案是在 `Plugins` 元类的 `get` 方法中处理它。每当插件在注册表中找不到时，`get` 方法应自动从 `plugins` 目录中 `import` 模块。

这种方法的优势在于插件不需要显式预加载，同时插件也只有在需要时才会加载。未使用的插件不会被触及，因此这种方法可以帮助减少应用程序的加载时间。

缺点是代码将不会运行或测试，因此它可能完全损坏，而你直到它最终加载时才知道。关于这个问题的解决方案将在第十章“测试”中介绍。另一个问题是，如果代码在应用程序的其他部分自我注册，那么该代码也不会被执行，除非你在代码的其他部分添加所需的`import`，也就是说。

修改`Plugins.get`方法，我们得到以下结果：

```py
import importlib

# Plugins class omitted for brevity
class PluginsOnDemand(Plugins):
    @classmethod
    def get(cls, name):
        if name not in cls.plugins:
            print('Loading plugins from plugins.%s' % name)
            importlib.import_module('plugins.%s' % name)
        return cls.plugins[name] 
```

现在我们从这个 Python 文件中运行它：

```py
import plugins

print(plugins.PluginsOnDemand.get('a'))
print(plugins.PluginsOnDemand.get('a')) 
```

这会导致以下结果：

```py
Loading plugins from plugins.a
<class 'plugins.a.A'>
<class 'plugins.a.A'> 
```

正如你所见，这种方法只会运行一次`import`；第二次，插件将在插件字典中可用，因此不需要再次加载。

## 通过配置导入插件

虽然只加载所需的插件是有用的，因为它可以减少你的初始加载时间和内存开销，但关于预先加载你可能会需要的插件也有一些话要说。根据 Python 的禅意，明确优于隐晦，所以一个明确的插件加载列表通常是一个好的解决方案。这种方法的优势在于，你可以确保注册更加高级，因为你可以保证它会运行，并且你可以从多个包中加载插件。当然，缺点是你需要明确定义要加载哪些插件，这可能会被视为违反 DRY（不要重复自己）原则。

这次我们不再在`get`方法中导入，而是添加一个`load`方法，该方法导入所有给定的模块名称：

```py
# PluginsOnDemand class omitted for brevity
class PluginsThroughConfiguration(PluginsOnDemand):
    @classmethod
    def load(cls, *plugin_names):
        for plugin_name in plugin_names:
            cls.get(plugin_name) 
```

可以使用以下代码调用：

```py
import plugins

plugins.PluginsThroughConfiguration.load(
    'a',
    'b',
)

print('After load')
print(plugins.PluginsThroughConfiguration.get('a'))
print(plugins.PluginsThroughConfiguration.get('a')) 
```

这会导致以下输出：

```py
Loading plugins from plugins.a
Loading plugins from plugins.b
After load
<class 'plugins.a.A'>
<class 'plugins.a.A'> 
```

一个相当简单直接的系统，根据设置加载插件，这可以很容易地与任何类型的设置系统结合使用，以填充`load`方法。这种方法的一个例子是 Django 中的`INSTALLED_APPS`。

## 通过文件系统导入插件

加载插件最方便的方法是无需思考的方法，因为它会自动发生。虽然这非常方便，但应该考虑一些非常重要的注意事项。

首先，它们往往会使调试变得更加困难。Django 中类似的自动导入系统给我带来了不少麻烦，因为它们往往会模糊化错误，甚至完全隐藏它们，让你花费数小时进行调试。

其次，这可能会带来安全风险。如果有人有权写入你的插件目录之一，他们可以有效地在你的应用程序中执行代码。

话虽如此，特别是对于初学者和/或你的框架的新用户，自动插件加载可以非常方便，并且确实值得演示。

这次，我们继承了在前面示例中创建的`PluginsThroughConfiguration`类，并添加了一个`autoload`方法来检测可用的插件。

```py
import re
import pathlib
import importlib

CURRENT_FILE = pathlib.Path(__file__)
PLUGINS_DIR = CURRENT_FILE.parent
MODULE_NAME_RE = re.compile('[a-z][a-z0-9_]*', re.IGNORECASE)

class PluginsThroughFilesystem(PluginsThroughConfiguration):
    @classmethod
    def autoload(cls):
        for filename in PLUGINS_DIR.glob('*.py'):
            # Skip __init__.py and other non-plugin files
            if not MODULE_NAME_RE.match(filename.stem):
                continue
                cls.get(filename.stem)

            # Skip this file
            if filename == CURRENT_FILE:
                continue

            # Load the plugin
            cls.get(filename.stem) 
```

现在，让我们尝试运行这段代码：

```py
import pprint
import plugins

plugins.PluginsThroughFilesystem.autoload()

print('After load')
pprint.pprint(plugins.PluginsThroughFilesystem.plugins) 
```

这会导致：

```py
Loading plugins from plugins.a
Loading plugins from plugins.b
After load
{'a': <class 'plugins.a.A'>,
 'b': <class 'plugins.b.B'>,
 'plugin': <class 'plugins.base.Plugin'>} 
```

现在，`plugins`目录中的每个文件都将自动加载。但请注意，这可能会掩盖某些错误。例如，如果你的某个插件导入了一个你没有安装的库，你将不会从实际库中收到`ImportError`，而是从插件中收到。

要使这个系统更智能一些（甚至导入 Python 路径之外的包），你可以使用`importlib.abc`中的抽象基类创建一个插件加载器；请注意，你很可能仍然需要以某种方式列出文件和/或目录。为了改进这一点，你还可以查看`importlib`中的加载器。使用这些加载器，你可以从 ZIP 文件和其他来源加载插件。

现在我们已经完成了插件系统，是时候看看如何使用元类而不是装饰器来实现`dataclasses`了。

# Dataclasses

在*第四章*，*Pythonic 设计模式*中，我们已经看到了`dataclasses`模块，它使得在类中实现简单的类型提示甚至强制某些结构成为可能。

现在，让我们看看我们如何使用元类实现自己的版本。实际的`dataclasses`模块主要依赖于类装饰器，但这不是问题。元类可以被视为类装饰器的更强大版本，所以它们将正常工作。使用元类，你可以使用继承来重用它们，或者使类继承其他类，但最重要的是，它们允许你修改类对象，而不是使用装饰器修改实例。

`dataclasses`模块中有几个非平凡的技巧，难以复制。除了添加文档和一些实用方法之外，它还生成一个与`dataclass`字段匹配的`__init__`方法。由于整个`dataclasses`模块大约有 1,300 行，我们的实现将无法接近。因此，我们将实现`__init__()`方法，包括为类型提示生成的`signature`和`__annotations__`，以及一个`__repr__`方法来显示结果：

```py
import inspect

class Dataclass(type):
    def _get_signature(namespace):
        # Get the annotations from the class
        annotations = namespace.get('__annotations__', dict())

        # Signatures are immutable so we need to build the
        # parameter list before creating the signature
        parameters = []
        for name, annotation in annotations.items():

            # Create Parameter shortcut for readability
            Parameter = inspect.Parameter
            # Create the parameter with the correct type
            # annotation and default. You could also choose to
            # make the arguments keyword/positional only here
            parameters.append(Parameter(
                name=name,
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=namespace.get(name, Parameter.empty),
                annotation=annotation,
            ))

        return inspect.Signature(parameters)

    def _create_init(namespace, signature):
        # If init exists we don't need to do anything
        if '__init__' in namespace:
            return

        # Create the __init__ method and use the signature to
        # process the arguments
        def __init__(self, *args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            for key, value in bound.arguments.items():
                # Convert to the annotation to enforce types
                parameter = signature.parameters[key]
                # Set the casted value
                setattr(self, key, parameter.annotation(value))

        # Override the signature for __init__ so help() works
        __init__.__signature__ = signature

        namespace['__init__'] = __init__

    def _create_repr(namespace, signature):
        def __repr__(self):
            arguments = []
            for key, value in vars(self).items():
                arguments.append(f'{key}={value!r}')
            arguments = ', '.join(arguments)
            return f'{self.__class__.__name__}({arguments})'

        namespace['__repr__'] = __repr__

    def __new__(metaclass, name, bases, namespace):
        signature = metaclass._get_signature(namespace)
        metaclass._create_init(namespace, signature)
        metaclass._create_repr(namespace, signature)

        cls = super().__new__(metaclass, name, bases, namespace)

        return cls 
```

乍一看，这可能会看起来很复杂，但实际上整个过程相当简单：

1.  我们从类的`__annotations__`和默认值中生成一个签名。

1.  我们根据签名生成一个`__init__`方法。

1.  我们让`__init__`方法使用签名来自动绑定传递给函数的参数并将它们应用到实例上。

1.  我们生成一个`__repr__`方法，该方法简单地打印出类的名称和存储在实例中的值。请注意，这个方法相当有限，会显示你添加到类中的任何内容。

注意，作为额外的小细节，我们有一个转换为注解类型的强制转换，以确保类型的正确性。

让我们通过使用*第四章*中的`dataclass`示例并添加一些小的修改来测试类型转换，看看它是否按预期工作：

```py
>>> from T_10_dataclasses import Dataclass

>>> class Sandwich(metaclass=Dataclass):
...     spam: int
...     eggs: int = 3

>>> Sandwich(1, 2)
Sandwich(spam=1, eggs=2)

>>> sandwich = Sandwich(4)
>>> sandwich
Sandwich(spam=4, eggs=3)
>>> sandwich.eggs
3

>>> help(Sandwich.__init__)
Help on function __init__ in ...
<BLANKLINE>
__init__(spam: int, eggs: int = 3)
<BLANKLINE>

>>> Sandwich('a')
Traceback (most recent call last):
    ...
ValueError: invalid literal for int() with base 10: 'a'
>>> Sandwich('1234', 56.78)
Sandwich(spam=1234, eggs=56) 
```

所有这些按预期工作，输出与原始`dataclass`相似。当然，它的功能要有限得多，但它展示了如何动态生成自己的类和函数，以及如何轻松地将基于自动注解的类型转换添加到代码中。

接下来，我们将深入探讨类的创建和实例化。

# 实例化类时的操作顺序

在调试动态创建和/或修改的类的问题时，操作顺序非常重要。假设一个错误的顺序可能会导致难以追踪的 bug。类的实例化按照以下顺序进行：

1.  查找元类

1.  准备命名空间

1.  执行类体

1.  创建类对象

1.  执行类装饰器

1.  创建类实例

我们现在将逐一介绍这些内容。

## 查找元类

元类来自类或`bases`中显式给出的元类，或者使用默认的`type`元类。

对于每个类，包括类本身和基类，将使用以下匹配中的第一个：

+   显式给出的元类

+   从`bases`显式地定义元类

+   `type()`

注意，如果没有找到所有候选元类的子类型元类，将引发`TypeError`。这种情况不太可能发生，但在使用元类和多重继承/混入时确实有可能发生。

## 准备命名空间

通过上述选择的元类准备类命名空间。如果元类有一个`__prepare__`方法，它将被调用为`namespace = metaclass.__prepare__(names, bases, **kwargs)`，其中`**kwargs`来自类定义。如果没有`__prepare__`方法可用，结果将是`namespace = dict()`。

注意，有多种方法可以实现自定义命名空间。正如我们在上一节中看到的，`type()`函数调用也接受一个`dict`参数，可以用来更改命名空间。

## 执行类体

类体的执行与正常代码执行非常相似，只有一个关键区别：独立的命名空间。由于类有一个独立的命名空间，不应污染`globals()`/`locals()`命名空间，因此它在那个上下文中执行。生成的调用看起来像这样：

```py
exec(body, globals(), namespace) 
```

其中`namespace`是之前生成的命名空间。

## 创建类对象（不是实例）

现在我们已经准备好了所有组件，可以实际生成类对象。这是通过`class_ = metaclass(name, bases, namespace, **kwargs)`调用完成的，如您所见，这实际上与之前讨论的`type()`调用相同。这里的`**kwargs`与之前传递给`__prepare__`方法的相同。

可能需要注意，这也是`super()`不带参数引用的对象。

## 执行类装饰器

现在类对象实际上已经完成，类装饰器将被执行。由于这仅在类对象中的所有其他内容都已构建之后执行，因此修改类属性（如正在继承的类和类的名称）变得困难。通过修改`__class__`对象，你仍然可以修改或覆盖这些属性，但这至少是更困难的。

## 创建类实例

从上面产生的类对象，我们现在可以最终创建实际的实例，就像通常使用类一样。需要注意的是，与上面的步骤不同，这一步和类装饰器步骤是每次实例化类时唯一执行的步骤。这两个步骤之前的步骤每个类定义只执行一次。

## 示例

理论已经足够了——让我们通过展示类对象的创建和实例化来检查操作顺序：

```py
>>> import functools

>>> def decorator(name):
...     def _decorator(cls):
...         @functools.wraps(cls)
...         def __decorator(*args, **kwargs):
...             print('decorator(%s)' % name)
...             return cls(*args, **kwargs)
...
...         return __decorator
...
...     return _decorator

>>> class SpamMeta(type):
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
```

使用创建的类和装饰器，我们现在可以说明`__prepare__`和`__new__`等方法的调用时机：

```py
>>> @decorator('Spam')
... class Spam(metaclass=SpamMeta):
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
  ...
  File "<doctest T_11_order_of_operations.rst[6]>", line 1, in ...
AttributeError: 'function' object has no attribute 'eggs'

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

示例清楚地显示了类的创建顺序：

1.  通过`__prepare__`准备命名空间

1.  使用`__new__`创建类体

1.  使用`__init__`初始化元类（注意：这不是类的`__init__`）

1.  通过类装饰器初始化类

1.  通过类`__init__`函数初始化类

从这一点我们可以注意到，类装饰器是在类实际实例化时执行，而不是在此之前。这当然既有优点也有缺点，但如果你希望构建所有子类的注册表，使用元类肯定更方便，因为装饰器只有在实例化类之后才会注册。

此外，在实际上创建类对象（而不是实例）之前修改命名空间的能力也非常强大。这可以方便地在几个类对象之间共享一定的作用域，例如，或者确保某些项目始终在作用域中可用。

# 按定义顺序存储类属性

有时候定义顺序会起作用。例如，假设我们正在创建一个表示 CSV（逗号分隔值）格式的类。CSV 格式期望字段有特定的顺序。在某些情况下，这可能会由标题指示，但保持一致的字段顺序仍然很有用。类似系统在 ORM 系统如 SQLAlchemy 中用于存储表定义的列顺序，以及在 Django 表单中的输入字段顺序。

## 没有元类的经典解决方案

存储字段顺序的一个简单方法是为字段实例提供一个特殊的`__init__`方法，该方法在每次定义时递增，因此字段具有递增的索引属性。这种解决方案可以被认为是经典解决方案，因为它在 Python 2 中也能工作：

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

为了方便和使事物看起来更美观，我们添加了`FieldsMeta`类。

这里不是严格必需的，但它会自动处理在需要时填充`name`，并添加包含字段排序列表的`fields`列表。

## 使用元类获取排序命名空间

之前的方法更直接一些，也支持 Python 2，但使用 Python 3 我们有更多的选择。正如您在前一节中看到的，Python 3 给了我们`__prepare__`方法，它返回命名空间。从*第四章*，您可能还记得`collections.OrderedDict`，那么让我们看看当我们结合它们会发生什么：

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

如您所见，字段确实按照我们定义的顺序排列。首先是 Spam，然后是鸡蛋。由于类命名空间现在是一个`collections.OrderedDict`实例，我们知道顺序是有保证的。需要注意的是，自 Python 3.6 以来，普通`dict`的顺序也是一致的，但`__prepare__`的使用示例仍然有用。它展示了元类如何方便地以通用方式扩展你的类。与自定义`__init__`方法相比，元类的一个重大优势是，如果用户忘记调用父`__init__`方法，他们不会丢失功能。除非添加了不同的元类，否则元类总是会执行。

# 练习

本章最重要的要点是教您如何内部工作元类：元类只是一个创建类的类，而这个类反过来又由另一个元类创建（最终递归到`type`）。然而，如果您想挑战自己，您还可以用元类做更多的事情：

+   验证是元类可以非常有用的一个最突出的例子。你可以验证属性/方法是否存在，你可以检查是否继承了所需的类，等等。可能性是无限的。

+   创建一个元类，它将每个方法包装在装饰器中（可能对日志记录/调试目的有用），具有如下签名：

    ```py
    class SomeClass(metaclass=WrappingMeta, wrapper=some_wrapper): 
    ```

这些练习的示例答案可以在 GitHub 上找到：`github.com/mastering-python/exercises`。我们鼓励您提交自己的解决方案，并从他人的替代方案中学习。

# 摘要

Python 的元类系统是每个 Python 程序员都在使用的，也许他们甚至都不知道。每个类都是通过某种（子类）`type`创建的，这允许进行无限定制的魔法。

你现在可以像平时一样创建类，并在定义过程中动态地添加、修改或从你的类中删除属性；非常神奇但非常有用。然而，魔法成分也是为什么应该非常谨慎地使用元类的原因。虽然它们可以使你的生活变得更加容易，但它们也是产生完全无法理解的代码的最简单方法之一。

不论如何，元类有一些非常好的用例，许多库如 SQLAlchemy 和 Django 都使用元类来使你的代码工作得更加容易，并且可以说是更好。实际上理解这些库内部使用的魔法通常不是使用这些库所必需的，这使得这些用例有可辩护性。

问题变成了是否一个对初学者来说更好的体验值得一些内部的黑暗魔法，并且从这些库的成功来看，我认为在这种情况下答案是肯定的。

总结来说，当考虑使用元类时，请记住蒂姆·彼得斯曾经说过的话：

> “元类比 99%的用户应该关心的任何东西都要深奥。如果你想知道你是否需要它们，那么你就不需要。”

随着类装饰器和`__init_subclass__`、`__set_name__`等方法的引入，对元类的需求进一步减少。所以当你犹豫不决时，你可能真的不需要它们。

现在我们将继续介绍一种解决方案来移除元类生成的一些魔法——文档。下一章将展示你的代码如何进行文档化，如何测试这些文档，以及最重要的是，如何通过注释类型使文档变得更加智能。

# 加入我们的 Discord 社区

加入我们的 Discord 空间，与作者和其他读者进行讨论：[`discord.gg/QMzJenHuJf`](https://discord.gg/QMzJenHuJf)

![二维码](img/QR_Code156081100001293319171.png)
