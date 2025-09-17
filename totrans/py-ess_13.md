# 第十三章。元编程和装饰器

我们所涵盖的大部分内容都是编程——编写 Python 语句来处理数据。我们也可以使用 Python 来处理 Python，而不是处理数据。我们将这称为元编程。我们将探讨两个方面：装饰器和元类。

装饰器是一个接受函数作为参数并返回函数的函数。我们可以使用它来向函数添加功能，而无需在多个不同的函数定义中重复该功能。装饰器防止了复制粘贴编程。我们经常用于日志记录、审计或安全目的；这些都是将跨越多个类或函数定义的事情。

元类定义将扩展当我们创建类的实例时发生的本质对象创建。隐式地，使用特殊的`__new__()`方法名称来创建一个裸对象，随后由类的`__init__()`方法进行初始化。元类允许我们改变对象创建的一些基本特性。

# 使用装饰器的简单元编程

Python 有一些内置的装饰器可以修改函数或类的成员方法。例如，在第十一章类定义中，我们看到了`@staticmethod`和`@property`，它们用于改变类中方法的行怍。`@staticmethod`装饰器将函数改为在类上工作，而不是在类的实例上工作。`@property`装饰器使得评估无参数方法可以通过与属性相同的语法进行。

`functools`模块中可用的函数装饰器是`@lru_cache`。这修改了一个函数以添加备忘录。缓存结果可以显著提高速度。它看起来是这样的：

```py
from functools import lru_cache
from glob import glob
import os

@lru_cache(100)
def find_source(directory):
    return glob(os.path.join(directory,"*.py"))
```

在这个例子中，我们导入了`@lru_cache`装饰器。我们还导入了`glob.glob()`函数和`os`模块，这样我们就可以使用`os.path.join()`来创建文件名，而不管操作系统特定的标点符号。

我们向`@lru_cache()`装饰器提供了一个大小参数。参数化装饰器通过添加一个将保存 100 个先前结果的缓存来修改`find_source()`函数。这可以加快大量使用本地文件系统的程序。**最近最少使用**（**LRU**）算法确保保留最近的请求，并悄悄地忘记较旧的请求，以限制缓存的大小。

`@lru_cache`装饰器体现了一种可重用的优化，可以应用于各种函数。我们将函数实现的备忘录方面与其他方面分离。

*Python 标准库*定义了一些装饰器。有关装饰器元编程的更多示例，请参阅 Python 装饰器库页面，[`wiki.python.org/moin/PythonDecoratorLibrary`](https://wiki.python.org/moin/PythonDecoratorLibrary)。

# 定义我们自己的装饰器

在某些情况下，我们可以从多个函数中提取一个共同方面。像安全、审计或日志记录这样的关注点是我们在许多函数或类中一致实现的一些常见示例。

让我们看看一种支持增强调试的方法。我们的目标是有一个简单的注解，我们可以用它从几个无关的函数中提供一致、详细的输出。我们希望创建一个具有如下定义的模块：

```py
@debug_log
def some_function(ksloc):
    return 2.4*ksloc**1.05
@debug_log
def another_function(ksloc, a=3.6, b=1.20):
    return a*ksloc**b
```

我们定义了两个简单的函数，这些函数将被装饰器包装以提供一致的调试输出。

装饰器是一个接受函数作为参数并返回函数作为结果的函数。前面代码块中展示的内容评估如下：

```py
>>> def some_function(ksloc):
...    return 2.4*ksloc**1.05
>>>  some_function = debug_log(debug_log)
```

当我们将装饰器应用于一个函数时，我们隐式地用原始函数作为参数评估装饰器函数。这将创建一个装饰后的函数作为结果。使用装饰器创建的结果与原始函数具有相同的名称——装饰后的版本替换了原始版本。

为了使这生效，我们需要编写一个装饰器来创建调试日志条目。这必须是通用的，以便它适用于任何函数。正如我们在第七章中提到的，*基本函数定义*，我们可以使用`*`和`**`修饰符将“所有其他”位置参数和所有其他关键字参数收集到一个单一的序列或一个单一的字典中。这允许我们编写完全通用的装饰器。

这里是`@debug_log`装饰器函数：

```py
import logging
from functools import wraps
def debug_log(func):
    log= logging.getLogger(func.__name__)
    @wraps(func)
    def decorated(*args, **kw):
        log.debug(">>> call(*{0}, **{1})".format(args, kw))
        try:
            result= func(*args, **kw)
            log.debug("<<< return {}".format(result))
            return result
        except Exception as ex:
            log.exception( "*** {}".format(ex) )
            raise
    return decorated
```

装饰器定义的主体做了三件事。首先，它根据原始函数的名称`func.__name__`创建了一个日志记录器。其次，它定义了一个全新的函数，命名为`decorated()`，这个函数基于原始函数。最后，它返回这个新函数。

注意，我们使用了`functools`库中的一个装饰器`@wraps`来显示新装饰器函数包装了原始函数。这将确保名称和文档字符串正确地从原始函数复制到装饰后的函数。装饰后的版本将无法与原始版本区分。

我们可以像使用普通函数一样使用这些函数：

```py
>>> round(some_function(25),3)
70.477
```

装饰对函数的值没有影响。它有轻微的性能影响。

如果我们启用了日志记录，并将日志级别设置为`DEBUG`，我们将在日志中看到额外的输出。前面的示例会导致以下内容出现在日志记录器的输出中：

```py
DEBUG:some_function:>>> call(*(25,), **{})
DEBUG:some_function:<<< return 70.47713658528114
```

这显示了由这个装饰器产生的调试细节。日志显示了参数值和结果值。如果有异常，我们还会看到参数值以及异常信息，这比只显示异常信息更有用。

启用日志记录的一个简单方法是在应用程序中包含以下内容：

```py
import sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
```

这将把日志输出定向到标准错误流。它还将包括所有严重级别高于调试级别的消息。我们可以将此级别设置更改为类似 `logging.INFO` 的值，以静默调试消息，同时保留信息性消息。

一个也接受参数值的装饰器——类似于 `@lru_cache` 装饰器——更复杂。首先将参数值应用于创建装饰器。然后使用这个初始绑定产生的装饰器来构建装饰函数。

# 使用元类的更复杂元编程

在某些情况下，类中内置的默认功能可能不适合我们的特定应用。我们可以看到一些常见的情况，我们可能想要扩展对象构造的默认行为。

+   我们可以使用元类来保留定义类的原始源代码的一部分。默认情况下，每个类对象使用 `dict` 来存储各种方法和类级属性。我们可能想使用有序字典来保留类级属性的原始源代码顺序。一个例子可以在 *Python 标准库* 的 3.3.3.5 节中找到。

+   **抽象基类**（**ABC**）依赖于一个 `metaclass __new__()` 方法来确认当我们尝试创建类的实例时，具体的子类是否完整。如果我们未能在一个 ABC 的子类中提供所有必需的方法，我们就无法创建该子类的实例。

+   元类可用于简化对象序列化。元类可以包含用于 XML 或 JSON 表示实例所需的信息。

+   我们可以使用元类向对象注入额外的属性。因为元类提供了创建空对象的 `__new__()` 方法的实现，它能够在 `__init__()` 方法评估之前注入属性。对于一些不可变类，如元组，没有 `__init__()` 方法，元组的子类必须使用 `__new__()` 方法来设置值。

默认元类是 `type`。这是由应用程序类在调用 `__init__()` 方法之前创建新裸对象时使用的。内置的 `type.__new__()` 方法需要四个参数值——元类、应用程序类的名称、应用程序类的基础类，以及系统定义的初始值命名空间。

当我们创建元类时，我们将覆盖 `__new__()` 方法。我们仍然会使用 `type.__new__()` 方法来创建裸对象。然后我们可以在返回对象之前扩展或修改这个裸对象。

这里有一个在 `__init__()` 之前插入日志记录器的元类：

```py
import logging
class Logged(type):
    def __new__(cls, name, bases, namespace, **kwds):
        result = type.__new__(cls, name, bases, dict(namespace))
        result.logger= logging.getLogger(name)
        return result
```

我们定义了一个扩展内置 `type` 类的类。我们定义了一个重写的特殊方法，`__new__()`。这个特殊方法使用超类 `type.__new__()` 方法来创建裸对象，并将其分配给 `result` 变量。

一旦我们有了裸对象，我们就可以创建一个记录器并将这个记录器注入到裸对象中。这个`self.logger`属性将在使用这个元类创建的每个类的`__init__()`方法的第 一行就可用。

我们可以创建利用这个元类的应用程序类，如下所示：

```py
class Machine(metaclass=Logged):
    def __init__(self, machine_id, base, cost_each):
        self.logger.info("creating {0} at {1}+{2}".format(
            machine_id, base, cost_each))
        self.machine_id= machine_id
        self.base= base
        self.cost_each= cost_each
    def application(self, units):
        total= self.base + self.cost_each*units
        self.logger.debug("Applied {units} ==> {total}".format(
            total=total, units=units, **self.__dict__))
        return total
```

我们定义了一个显式依赖于`Logged`元类的类。如果我们不包括`metaclass`关键字参数，将使用默认的`type`元类。在这个类中，`logger`属性是在调用`__init__()`方法之前创建的。这允许我们在`__init__()`方法中使用记录器而不需要任何额外的开销。

# 摘要

在本章中，我们探讨了两种常见的元编程技术。第一种是编写装饰器函数——这些可以用来转换原始函数以添加新特性。第二种是使用元类来扩展类定义的默认行为。

我们可以使用这些技术来开发跨越许多功能和类的应用程序特性。一次编写一个特性并将其应用于多个类，可以确保一致性，并在调试、升级或重构期间提供帮助。

在第十四章“完善 - 单元测试、打包和文档”中，我们将探讨一系列特征，这些特征定义了一个完整的 Python 项目。我们不会处理技术语言特性，而是会探讨我们可以如何使用 Python 特性来创建精致、完整的解决方案。
