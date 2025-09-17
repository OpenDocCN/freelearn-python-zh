# 第二十章：*第二十章*：装饰器模式

正如我们在上一章中看到的，使用**适配器**，第一个结构化设计模式，你可以将实现给定接口的对象适配到实现另一个接口。这被称为**接口****适配**，包括那些鼓励组合而非继承的模式的种类，这在你需要维护大型代码库时可能会带来好处。

另一个值得学习的有趣的结构模式是**装饰器**模式，它允许我们动态且透明地（不影响其他对象）地向对象添加职责；这将是本章的主题。在我们讨论的过程中，我们将了解更多关于这个设计模式的具体用法：**记忆化**。

我们将讨论以下主题：

+   介绍装饰器模式

+   现实世界示例

+   用例

+   实现

# 技术要求

本章的代码文件可以通过此链接访问：[`github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter20`](https://github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter20)。

# 介绍装饰器模式

作为 Python 开发者，我们可以在`func_in`中编写装饰器，作为输入并返回另一个函数对象`func_out`。这是一种常用的技术，用于扩展函数、方法或类的行为。

但这个特性不应该对你来说完全陌生。我们已经在*第十六章*，“工厂模式”，和*第十七章*，“建造者模式”中看到了如何使用内置的`property`装饰器，它使得一个方法在两种模式中都可以像变量一样出现。Python 中还有几个其他有用的内置装饰器。在本章的*实现*部分，我们将学习如何实现和使用我们自己的装饰器。

注意，装饰器模式与 Python 的装饰器特性之间没有一对一的关系。Python 装饰器实际上可以做比装饰器模式更多的事情。它们可以用作实现装饰器模式的事情之一([j.mp/moinpydec](http://j.mp/moinpydec))。

现在，让我们讨论一些适用装饰器模式的示例。

# 现实世界示例

装饰器模式通常用于扩展对象的功能。在日常生活中，功能扩展的例子包括给手机添加支架或使用不同的相机镜头。

在大量使用装饰器的 Django 框架中，我们有`View`装饰器，它可以用于以下操作([j.mp/djangodec](http://j.mp/djangodec))：

+   根据 HTTP 请求限制对视图的访问

+   控制特定视图的缓存行为

+   基于视图控制压缩

+   根据特定的 HTTP 请求头控制缓存

Pyramid 框架和 Zope 应用程序服务器也使用装饰器来实现各种目标，如下所示：

+   将函数注册为事件订阅者

+   使用特定权限保护方法

+   实现适配器模式

为了更具体，我们将在下一节中迭代设计模式的具体用例。

# 用例

当用于实现横切关注点时，装饰器模式特别出色（[j.mp/wikicrosscut](http://j.mp/wikicrosscut)）。横切关注点的例子如下：

+   数据验证

+   缓存

+   记录

+   监控

+   调试

+   业务规则

+   加密

通常，所有通用的应用程序部分，可以应用于它的许多不同部分，都被认为是横切关注点。

使用装饰器模式的另一个流行例子是**图形用户界面**（**GUI**）工具包。在 GUI 工具包中，我们希望能够为单个组件/小部件添加诸如边框、阴影、颜色和滚动等特性。

现在，让我们继续本章的实现部分，我们将看到装饰器模式如何帮助实现缓存。

# 实现

Python 装饰器是通用的且非常强大。你可以在`python.org`的装饰器库中找到许多它们如何被使用的例子（[j.mp/pydeclib](http://j.mp/pydeclib)）。在本节中，我们将看到我们如何实现一个缓存装饰器（[j.mp/memoi](http://j.mp/memoi)）。所有递归函数都可以从缓存中受益，所以让我们尝试一个函数，`number_sum()`，它返回前`n`个数的和。注意，这个函数已经在`math`模块中作为`fsum()`可用，但让我们假装它不可用。

首先，让我们看看朴素实现（`number_sum_naive.py` 文件）：

```py
def number_sum(n): 
    '''Returns the sum of the first n numbers''' 
    assert(n >= 0), 'n must be >= 0' 

    if n == 0:
        return 0
    else:
        return n + number_sum(n-1)  

if __name__ == '__main__': 
    from timeit import Timer 
    t = Timer('number_sum(30)', 'from __main__ import \
       number_sum')
    print('Time: ', t.timeit())
```

此示例的样本执行显示了这种实现有多慢。在 MacBook 上计算前 30 个数的和大约需要 3 秒钟，这可以通过执行`python number_sum_naive.py`命令来看到：

```py
Time:  3.023907012
```

让我们看看使用缓存是否可以帮助我们提高性能。在下面的代码中，我们使用`dict`来缓存已计算的和。我们还更改了传递给`number_sum()`函数的参数。我们想要计算前 300 个数的和，而不仅仅是前 30 个。

这里是使用缓存的新代码版本：

```py
sum_cache = {0:0}

def number_sum(n): 
    '''Returns the sum of the first n numbers''' 
    assert(n >= 0), 'n must be >= 0'

    if n in sum_cache:
        return sum_cache[n]
    res = n + number_sum(n-1)
    # Add the value to the cache
    sum_cache[n] = res
    return res

if __name__ == '__main__': 
    from timeit import Timer 
    t = Timer('number_sum(300)', 'from __main__ import \
      number_sum')
    print('Time: ', t.timeit())
```

执行基于缓存的代码显示性能显著提高，即使是计算大数值也是可接受的。

使用`python number_sum.py`执行的样本执行如下：

```py
Time:  0.12304591899999999
```

但这种方法已经存在一些问题。虽然性能不再是问题，但代码的整洁性不如不使用记忆化时。如果我们决定扩展代码以包含更多的数学函数并将其转变为一个模块，会发生什么呢？我们可以想到几个对我们模块有用的函数，例如帕斯卡三角形或斐波那契数列算法问题。

因此，如果我们想在同一个模块中创建一个与`number_sum()`相同的函数，用于斐波那契数列，并使用相同的记忆化技术，我们会添加如下代码：

```py
cache_fib = {0:0, 1:1} 

def fibonacci(n): 
    '''Returns the suite of Fibonacci numbers''' 
    assert(n >= 0), 'n must be >= 0' 

    if n in cache_fib: 
        return cache_fib[n] 
    res = fibonacci(n-1) + fibonacci(n-2) 
    cache_fib[n] = res 
    return res
```

你已经注意到这个问题了吗？我们最终得到了一个新的`dict`，名为`cache_fib`，它作为我们的`fibonacci()`函数的缓存，以及一个比不使用记忆化更复杂的函数。我们的模块正变得不必要地复杂。是否有可能编写这些函数，使它们尽可能简单，但又能达到使用记忆化函数的性能？

幸运的是，确实如此，解决方案是使用装饰器模式。

首先，我们创建一个`memoize()`装饰器，如下面的示例所示。我们的装饰器接受需要记忆化的`fn`函数作为输入。它使用名为`cache`的`dict`作为缓存数据容器。`functools.wraps()`函数用于在创建装饰器时方便使用。虽然不是强制性的，但使用它是良好的实践，因为它确保了装饰的函数的文档和签名被保留（[j.mp/funcwraps](http://j.mp/funcwraps)）。在这种情况下，`*args`参数列表是必需的，因为我们想要装饰的函数接受输入参数（例如我们两个函数的`n`参数）：

```py
import functools 

def memoize(fn): 
    cache = dict() 

    @functools.wraps(fn) 
    def memoizer(*args): 
        if args not in cache: 
            cache[args] = fn(*args) 
        return cache[args] 

    return memoizer
```

现在，我们可以使用我们的`memoize()`装饰器与函数的原始版本一起使用。这的好处是代码可读性高，而不会影响性能。我们使用所谓的装饰（或装饰行）来应用装饰器。装饰使用`@name`语法，其中`name`是我们想要使用的装饰器的名称。这不过是一种语法糖，用于简化装饰器的使用。我们甚至可以绕过这种语法并手动执行装饰器，但这留给你作为练习。

因此，`memoize()`装饰器可以按照以下方式与我们的递归函数一起使用：

```py
@memoize 
def number_sum(n): 
    '''Returns the sum of the first n numbers''' 
    assert(n >= 0), 'n must be >= 0' 
    if n == 0:
        return 0
    else:
        return n + number_sum(n-1)

@memoize 
def fibonacci(n): 
    '''Returns the suite of Fibonacci numbers''' 
    assert(n >= 0), 'n must be >= 0'
    if n in (0, 1):
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

在代码的最后部分，通过`main()`函数，我们展示了如何使用装饰过的函数并测量它们的性能。`to_execute`变量用于保存一个包含每个函数引用和相应的`timeit.Timer()`调用的元组列表（在测量时间的同时执行），从而避免代码重复。注意`__name__`和`__doc__`方法属性分别显示了正确的函数名称和文档值。尝试从`memoize()`中移除`@functools.wraps(fn)`装饰，看看是否仍然如此。

这是代码的最后部分：

```py
def main():
    from timeit import Timer
    to_execute = [
        (number_sum, 
         Timer('number_sum(300)', 'from __main__ import \
           number_sum')),
        (fibonacci, 
         Timer('fibonacci(100)', 'from __main__ import \
           fibonacci'))    
    ]

    for item in to_execute:
        fn = item[0]
        print(f'Function "{fn.__name__}": {fn.__doc__}')
        t = item[1]
        print(f'Time: {t.timeit()}')
        print()
if __name__ == '__main__': 
    main()
```

让我们回顾一下如何编写我们数学模块的完整代码（`mymath.py`文件）：

1.  在导入我们将要使用的 Python 的`functools`模块之后，我们定义了`memoize()`装饰器函数。

1.  然后，我们定义了`number_sum()`函数，并使用`memoize()`进行了装饰。

1.  我们还定义了`fibonacci()`函数，并对其进行了装饰。

1.  最后，我们添加了前面展示的`main()`函数，并使用通常的技巧来调用它。

在执行`python mymath.py`命令时，这是一个示例输出：

```py
Function "number_sum": Returns the sum of the first n 
numbers
Time: 0.152614356
Function "fibonacci": Returns the suite of Fibonacci 
numbers
Time: 0.142395913
```

（执行时间可能因情况而异。）

到这一点，我们得到了可读的代码和可接受的性能。现在，你可能会争辩说这并不是装饰器模式，因为我们没有在运行时应用它。事实是，装饰过的函数不能被去装饰，但你仍然可以在运行时决定装饰器是否执行。这是一个有趣的练习，留给你。

注意

使用一个充当包装器的装饰器，它根据某些条件决定是否执行真正的装饰器。

本章未涵盖的装饰器的另一个有趣特性是，你可以用多个装饰器装饰一个函数。所以，这里有一个练习：创建一个帮助调试递归函数的装饰器，将其应用于`number_sum()`和`fibonacci()`，并最终确定多个装饰器执行的顺序。

# 概述

本章介绍了装饰器模式及其与 Python 编程语言的关系。我们方便地使用了装饰器模式来扩展对象的行为，而不使用继承。Python 通过其内置的装饰器功能，甚至进一步扩展了装饰器概念，允许我们扩展任何可调用对象（函数、方法或类）的行为，而不使用继承或组合。

我们已经看到了一些装饰过的现实世界对象的例子，例如相机。从软件的角度来看，Django 和 Pyramid 都使用装饰器来实现不同的目标，例如控制 HTTP 压缩和缓存。

装饰器模式是实现横切关注点的一个很好的解决方案，因为它们是通用的，并且不适合很好地融入面向对象范式。我们在*用例*部分提到了几个横切关注点的类别。实际上，在*实现*部分，我们演示了一个横切关注点：记忆化。我们看到了装饰器如何帮助我们保持函数的整洁，同时不牺牲性能。

下一章将介绍桥接模式。

# 问题

1.  装饰器模式的主要动机是什么？

1.  为什么装饰器模式在 Python 中特别相关？

1.  装饰器模式如何帮助实现记忆化？
