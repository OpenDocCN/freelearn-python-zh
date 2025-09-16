# 4

# 结构设计模式

在上一章中，我们介绍了创建型模式和面向对象编程模式，这些模式帮助我们处理对象创建过程。接下来，我们想要介绍的模式类别是**结构设计模式**。结构设计模式提出了一种组合对象以提供新功能的方法。

本章我们将涵盖以下主要内容：

+   适配器模式

+   装饰器模式

+   桥接模式

+   门面模式

+   享元模式

+   代理模式

在本章结束时，你将掌握使用结构设计模式高效且优雅地构建代码的技能。

# 技术要求

请参阅第一章中提出的需求。

# 适配器模式

**适配器模式**是一种结构设计模式，它帮助我们使两个不兼容的接口变得兼容。这究竟意味着什么？如果我们有一个旧组件，我们想在新的系统中使用它，或者我们想在旧系统中使用的新组件，这两个组件在没有进行任何代码更改的情况下很少能够相互通信。但是，更改代码并不总是可能的，要么是因为我们没有访问权限，要么是因为这样做不切实际。在这种情况下，我们可以编写一个额外的层，它会对所有必要的修改进行操作，以使两个接口之间能够通信。这个层被称为**适配器**。

通常，如果你想要使用一个期望`function_a()`的接口，但你只有`function_b()`，你可以使用适配器将（适配）`function_b()`转换为`function_a()`。

## 现实世界案例

当你从大多数欧洲国家前往英国或美国，或者相反，你需要使用一个插头适配器来给你的笔记本电脑充电。连接某些设备到你的电脑也需要同种类型的适配器：USB 适配器。

在软件类别中，`zope.interface`包（[`pypi.org/project/zope.interface/`](https://pypi.org/project/zope.interface/)），是**Zope 工具包**（**ZTK**）的一部分，提供了帮助定义接口和执行接口适配的工具。这些工具被用于几个 Python 网络框架项目的核心（包括 Pyramid 和 Plone）。

注意

`zope.interface`是 Python 中处理接口的解决方案，由 Zope 应用程序服务器和 ZTK 背后的团队提出，在 Python 引入内置机制之前，首先提出了**抽象基类**（也称为**ABCs**），后来又提出了协议。

## 适配器模式的使用案例

通常，两个不兼容的接口中有一个是外来的，或者是旧的/遗留的。如果接口是外来的，这意味着我们没有访问源代码。如果是旧的，通常重构它是不切实际的。

在实现之后使用适配器使事物工作是一种好方法，因为它不需要访问外部接口的源代码。如果我们必须重用一些遗留代码，这通常也是一个实用的解决方案。但要注意，它可能会引入难以调试的副作用。因此，请谨慎使用。

## 实现适配器模式——适配遗留类

让我们考虑一个例子，其中我们有一个遗留的支付系统和一个新的支付网关。适配器模式可以使它们在不更改现有代码的情况下一起工作，正如我们将要看到的。

遗留支付系统使用一个类实现，包含一个`make_payment()`方法，用于执行支付的核心工作，如下所示：

```py
class OldPaymentSystem:
    def __init__(self, currency):
        self.currency = currency
    def make_payment(self, amount):
        print(
            f"[OLD] Pay {amount} {self.currency}"
        )
```

新的支付系统使用以下类实现，提供了一个`execute_payment()`方法：

```py
class NewPaymentGateway:
    def __init__(self, currency):
        self.currency = currency
    def execute_payment(self, amount):
        print(
            f"Execute payment of {amount} {self.currency}"
        )
```

现在，我们将添加一个类，它将提供`make_payment()`方法，在这个方法中，我们在适配对象上调用`execute_payment()`方法来完成支付。代码如下：

```py
class PaymentAdapter:
    def __init__(self, system):
        self.system = system
    def make_payment(self, amount):
        self.system.execute_payment(amount)
```

这就是`PaymentAdapter`类如何适配`NewPaymentGateway`的接口以匹配`OldPaymentSystem`的接口。

让我们通过添加一个`main()`函数并包含测试代码来查看这种适配的结果，如下所示：

```py
def main():
    old_system = OldPaymentSystem("euro")
    print(old_system)
    new_system = NewPaymentGateway("euro")
    print(new_system)
    adapter = PaymentAdapter(new_system)
    adapter.make_payment(100)
```

让我们回顾一下实现的全代码（见`ch04/adapter/adapt_legacy.py`文件）：

1.  我们有一些遗留支付系统的代码，由`OldPaymentSystem`类表示，它提供了一个`make_payment()`方法，用于执行支付的核心工作，如下所示：

1.  我们引入了新的支付系统，使用`NewPaymentGateway`类，它提供了一个`execute_payment()`方法。

1.  我们添加了一个适配器类`PaymentAdapter`，它有一个属性用于存储支付系统对象，以及一个`make_payment()`方法；在该方法中，我们在支付系统对象上调用`execute_payment()`方法（通过`self.system.execute_payment(amount)`）。

1.  我们添加了测试我们接口适配设计的代码（并在常规的`if __name__ == "__main__"`块中调用它）。

执行代码，使用`python ch04/adapter/adapt_legacy.py`，应该得到以下输出：

```py
<__main__.OldPaymentSystem object at 0x10ee58fd0>
<__main__.NewPaymentGateway object at 0x10ee58f70>
Execute payment of 100 euro
```

你现在明白了。这种适配技术使我们能够使用新的支付网关，同时使用期望旧接口的现有代码。

## 实现适配器模式——将几个类适配到统一接口

让我们看看另一个应用来展示适配的例子：一个俱乐部的活动。我们俱乐部有两个主要活动：

+   聘请有才华的艺术家在俱乐部表演

+   组织表演和活动以娱乐其客户

在核心，我们有一个`Club`类，它代表俱乐部，聘请的艺术家在某个晚上表演。`organize_performance()`方法是俱乐部可以执行的主要动作。代码如下：

```py
class Club:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f"the club {self.name}"
    def organize_event(self):
        return "hires an artist to perform"
```

大多数时候，我们的俱乐部聘请 DJ 表演，但我们的应用程序应该能够组织多样化的表演：由音乐家或音乐乐队、舞者、单人或双人表演等。

通过我们的研究尝试重用现有代码，我们发现了一个开源贡献的库，它为我们带来了两个有趣的类：`Musician`和`Dancer`。在`Musician`类中，主要动作由`play()`方法执行。在`Dancer`类中，由`dance()`方法执行。

在我们的例子中，为了表明这两个类是外部的，我们将它们放在一个单独的模块中（在`ch04/adapter/external.py`文件中）。代码包括两个类，`Musician`和`Dancer`，如下所示：

```py
class Musician:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f"the musician {self.name}"
    def play(self):
        return "plays music"
class Dancer:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f"the dancer {self.name}"
    def dance(self):
        return "does a dance performance"
```

我们编写的代码，用于从外部库使用这两个类，只知道如何调用`organize_performance()`方法（在`Club`类上）；它对`play()`或`dance()`方法（在相应类上）一无所知。

我们如何在不修改`Musician`和`Dancer`类的情况下让代码工作？

适配器来拯救！我们创建了一个通用的`Adapter`类，它允许我们将具有不同接口的多个对象适配到一个统一的接口。`__init__()`方法的`obj`参数是我们想要适配的对象，`adapted_methods`是一个包含键/值对的字典，匹配客户端调用的方法和应该调用的方法。该类的代码如下：

```py
class Adapter:
    def __init__(self, obj, adapted_methods):
        self.obj = obj
        self.__dict__.update(adapted_methods)
    def __str__(self):
        return str(self.obj)
```

当处理不同类的实例时，我们有两种情况：

+   属于`Club`类的兼容对象不需要适配。我们可以将其视为原样。

+   不兼容的对象需要首先使用`Adapter`类进行适配。

结果是，客户端代码可以在所有对象上继续使用已知的`organize_performance()`方法，而无需意识到任何接口差异。考虑以下`main()`函数代码以证明设计按预期工作：

```py
def main():
    objects = [
        Club("Jazz Cafe"),
        Musician("Roy Ayers"),
        Dancer("Shane Sparks"),
    ]
    for obj in objects:
        if hasattr(obj, "play") or hasattr(
            obj, "dance"
        ):
            if hasattr(obj, "play"):
                adapted_methods = dict(
                    organize_event=obj.play
                )
            elif hasattr(obj, "dance"):
                adapted_methods = dict(
                    organize_event=obj.dance
                )
            obj = Adapter(obj, adapted_methods)
        print(f"{obj} {obj.organize_event()}")
```

让我们回顾一下我们适配器模式实现的完整代码（在`ch04/adapter/adapt_to_unified_interface.py`文件中）：

1.  我们从`external`模块中导入`Musician`和`Dancer`类。

1.  我们有`Club`类。

1.  我们定义了`Adapter`类。

1.  我们添加了`main()`函数，我们在通常的`if __name__ == "__main__"`块中调用它。

执行`python ch04/adapter/adapt_to_unified_interface.py`命令时的输出如下：

```py
the club Jazz Cafe hires an artist to perform
the musician Roy Ayers plays music
the dancer Shane Sparks does a dance performance
```

如你所见，我们成功地使`Musician`和`Dancer`类与客户端代码期望的接口兼容，而没有改变这些外部类的源代码。

# 装饰器模式

另一个值得学习的有趣结构模式是**装饰器**模式，它允许程序员以动态和透明的方式（不影响其他对象）向对象添加职责。

这个模式对我们来说还有一个有趣的原因，你将在下一分钟看到。

作为 Python 开发者，我们可以以**Pythonic**的方式（意味着使用语言的功能）编写装饰器，多亏了内置的装饰器功能。

注意

Python 装饰器是一个可调用对象（函数、方法或类），它接收一个`func_in`函数对象作为输入，并返回另一个函数对象`func_out`。这是一种常用的技术，用于扩展函数、方法或类的行为。

有关 Python 装饰器功能的更多详细信息，请参阅官方文档：[`docs.python.org/3/reference/compound_stmts.html#function`](https://docs.python.org/3/reference/compound_stmts.html#function)

但这个特性不应该对你来说完全陌生。我们在前面的章节中已经遇到了常用的装饰器（`@abstractmethod`，`@property`），Python 中还有几个其他有用的内置装饰器。现在，我们将学习如何实现和使用我们自己的装饰器。

注意，装饰器模式和 Python 的装饰器功能之间没有一对一的关系。Python 装饰器实际上可以做很多装饰器模式做不到的事情。它们可以用作实现装饰器模式的事情之一。

## 现实世界的例子

装饰器模式通常用于扩展对象的功能。在日常生活中，此类扩展的例子包括给枪支添加消声器、使用不同的相机镜头等。

在像 Django 这样的 Web 框架中，它大量使用装饰器，我们有以下用途的装饰器：

+   基于请求限制对视图（或 HTTP 请求处理函数）的访问

+   在特定视图中控制缓存行为

+   基于视图的压缩控制

+   基于特定 HTTP 请求头控制缓存

+   将函数注册为事件订阅者

+   使用特定权限保护函数

## 装饰器模式的用例

当用于实现横切关注点时，装饰器模式特别有用，例如以下内容：

+   数据验证

+   缓存

+   记录日志

+   监控

+   调试

+   业务规则

+   加密

通常，一个应用程序中所有通用且可以应用于其许多其他部分的组成部分都被认为是横切关注点。

使用装饰器模式的一个流行例子是在**图形用户界面**（**GUI**）工具包中。在 GUI 工具包中，我们希望能够向单个组件/小部件添加诸如边框、阴影、颜色和滚动等功能。

## 实现装饰器模式

Python 装饰器是通用的且非常强大。在本节中，我们将看到如何实现一个`number_sum()`函数，该函数返回前*n*个数字的总和。请注意，此函数已经在`math`模块中作为`fsum()`提供，但让我们假装它不存在。

首先，让我们看看这个简单的实现（在`ch04/decorator/number_sum_naive.py`文件中）：

```py
def number_sum(n):
    if n == 0:
        return 0
    else:
        return n + number_sum(n - 1)
if __name__ == "__main__":
    from timeit import Timer
    t = Timer(
        "number_sum(50)",
        "from __main__ import number_sum",
    )
    print("Time: ", t.timeit())
```

此示例的执行样本显示了这种实现的缓慢程度。在我的电脑上，计算前 50 个数字之和需要超过*7*秒。执行`python ch04/decorator/number_sum_naive.py`命令时，我们得到以下输出：

```py
dict for caching the already computed sums. We also change the parameter passed to the number_sum() function. We want to calculate the sum of the first 300 numbers instead of only the first 50.
			Here is the new version of the code (in the `ch04``/decorator/number_sum.py` file), using memoization:

```

sum_cache = {0: 0}

def number_sum(n):

if n in sum_cache:

return sum_cache[n]

res = n + number_sum(n - 1)

# Add the value to the cache

sum_cache[n] = res

return res

if __name__ == "__main__":

from timeit import Timer

t = Timer(

"number_sum(300)",

"from __main__ import number_sum",

)

print("Time: ", t.timeit())

```py

			Executing the memoization-based code shows that performance improves dramatically, and is acceptable even for computing large values.
			A sample execution, using `python ch04/decorator/number_sum.py`, is as follows:

```

Time:  0.1288748119986849

```py

			But there are a few problems with this approach. First, while the performance is not an issue any longer, the code is not as clean as it is when not using memoization. And what happens if we decide to extend the code with more math functions and turn it into a module? We can think of several functions that would be useful for our module, for problems such as Pascal’s triangle or the Fibonacci numbers suite algorithm.
			So, if we wanted a function in the same module as `number_sum()` for the Fibonacci numbers suite, using the same memoization technique, we would add code as follows (see the version in the `ch04/decorator/number_sum_and_fibonacci.py` file):

```

fib_cache = {0: 0, 1: 1}

def fibonacci(n):

if n in fib_cache:

return fib_cache[n]

res = fibonacci(n - 1) + fibonacci(n - 2)

fib_cache[n] = res

return res

```py

			Do you notice the problem? We ended up with a new dictionary called `fib_cache` that acts as our cache for the `fibonacci()` function, and a function that is more complex than it would be without using memoization. Our module is becoming unnecessarily complex.
			Is it possible to write these functions while keeping them as simple as the naive versions, but achieving a performance similar to the performance of the functions that use memoization?
			Fortunately, it is, and the solution is to use the decorator pattern.
			First, we create a `memoize()` decorator as shown in the following example. Our decorator accepts the `func` function, which needs to be memoized, as an input. It uses `dict` named `cache` as the cached data container. The `functools.wraps()` function is used for convenience when creating decorators. It is not mandatory but it’s a good practice to use it, since it makes sure that the documentation and the signature of the function that is decorated are preserved. The `*args` argument list is required in this case because the functions that we want to decorate accept input arguments (such as the `n` argument for our two functions):

```

import functools

def memoize(func):

cache = {}

@functools.wraps(func)

def memoizer(*args):

if args not in cache:

cache[args] = func(*args)

return cache[args]

return memoizer

```py

			Now we can use our `memoize()` decorator with the naive version of our functions. This has the benefit of readable code without performance impact. We apply a decorator using what is known as `@name` syntax, where `name` is the name of the decorator that we want to use. It is nothing more than syntactic sugar for simplifying the usage of decorators. We can even bypass this syntax and execute our decorator manually, but that is left as an exercise for you.
			So, the `memoize()` decorator can be used with our recursive functions as follows:

```

@memoize

def number_sum(n):

if n == 0:

return 0

else:

return n + number_sum(n - 1)

@memoize

def fibonacci(n):

if n in (0, 1):

return n

else:

return fibonacci(n - 1) + fibonacci(n - 2)

```py

			In the last part of the code, via the `main()` function, we show how to use the decorated functions and measure their performance. The `to_execute` variable is used to hold a list of tuples containing the reference to each function and the corresponding `timeit.Timer()` call (to execute it while measuring the time spent), thus avoiding code repetition. Note how the `__name__` and `__doc__` method attributes show the proper function names and documentation values, respectively. Try removing the `@functools.wraps(func)` decoration from `memoize()` and see whether this is still the case.
			Here is the last part of the code:

```

def main():

from timeit import Timer

to_execute = [

(

number_sum,

Timer(

"number_sum(300)",

"from __main__ import number_sum",

),

),

(

fibonacci,

Timer(

"fibonacci(100)",

"from __main__ import fibonacci",

),

),

]

for item in to_execute:

func = item[0]

print(

f'Function "{func.__name__}": {func.__doc__}'

)

t = item[1]

print(f"Time: {t.timeit()}")

print()

```py

			Let’s recapitulate how we write the complete code of our math module (the `ch04/decorator/decorate_math.py` file):

				1.  After the import of Python’s `functools` module that we will be using, we define the `memoize()` decorator function.
				2.  Then, we define the `number_sum()` function, decorated using `memoize()`.
				3.  Next, we define the `fibonacci()` function, decorated the same way.
				4.  Finally, we add the `main()` function, as shown earlier, and the usual trick to call it.

			Here is a sample output when executing the `python` `ch04/decorator/decorate_math.py` command:

```

Function "number_sum": Returns the sum of the first n numbers

Time: 0.2148694

Function "fibonacci": Returns the suite of Fibonacci numbers

Time: 0.202763251

```py

			Note
			The execution times might differ in your case. Also, regardless of the time spent, we can see that the decorator-based implementation is a win because the code is more maintainable.
			Nice! We ended up with readable code and acceptable performance. Now, you might argue that this is not the decorator pattern, since we don’t apply it at runtime. The truth is that a decorated function cannot be undecorated, but you can still decide at runtime whether the decorator will be executed or not. That’s an interesting exercise left for you. *Hint for the exercise:* use a decorator that acts as a wrapper, which decides whether or not the real decorator is executed based on some condition.
			The bridge pattern
			A third structural pattern to look at is the **bridge** pattern. We can actually compare the bridge and the adapter patterns, looking at the way both work. While the adapter pattern is used *later* to make unrelated classes work together, as we saw in the implementation example we discussed earlier in the section on *The adapter pattern*, the bridge pattern is designed *up-front* to decouple an implementation from its abstraction, as we are going to see.
			Real-world examples
			In our modern, everyday lives, an example of the bridge pattern I can think of is from the *digital economy*: information products. Nowadays, the information product or *infoproduct* is part of the resources one can find online for training, self-improvement, or one’s ideas and business development. The purpose of an information product that you find on certain marketplaces, or the website of the provider, is to deliver information on a given topic in such a way that it is easy to access and consume. The provided material can be a PDF document or ebook, an ebook series, a video, a video series, an online course, a subscription-based newsletter, or a combination of all those formats.
			In the software realm, we can find two examples:

				*   **Device drivers**: Developers of an OS define the interface for device (such as printers) vendors to implement it
				*   **Payment gateways**: Different payment gateways can have different implementations, but the checkout process remains consistent

			Use cases for the bridge pattern
			Using the bridge pattern is a good idea when you want to share an implementation among multiple objects. Basically, instead of implementing several specialized classes, and defining all that is required within each class, you can define the following special components:

				*   An abstraction that applies to all the classes
				*   A separate interface for the different objects involved

			An implementation example we are about to see will illustrate this approach.
			Implementing the bridge pattern
			Let’s assume we are building an application where the user is going to manage and deliver content after fetching it from diverse sources, which could be the following:

				*   A web page (based on its URL)
				*   A resource accessed on an FTP server
				*   A file on the local filesystem
				*   A database server

			So, here is the idea: instead of implementing several content classes, each holding the methods responsible for getting the content pieces, assembling them, and showing them inside the application, we can define an abstraction for the *Resource Content* and a separate interface for the objects that are responsible for fetching the content. Let’s try it!
			We begin with the interface for the implementation classes that help fetch content – that is, the `ResourceContentFetcher` class. This concept is called the `protocols` feature, as follows:

```

class ResourceContentFetcher(Protocol):

def fetch(self, path: str) -> str:

...

```py

			Then, we define the class for our Resource Content abstraction, called `ResourceContent`. The first trick we use here is that, via an attribute (`_imp`) on the `ResourceContent` class, we maintain a reference to the object that represents the Implementor (fulfilling the `ResourceContentFetcher` interface). The code is as follows:

```

class ResourceContent:

def __init__(self, imp: ResourceContentFetcher):

self._imp = imp

def get_content(self, path):

return self._imp.fetch(path)

```py

			Now we can add an `implementation` class to fetch content from a web page or resource:

```

class URLFetcher:

def fetch(self, path):

res = ""

req = urllib.request.Request(path)

with urllib.request.urlopen(

req

) as response:

if response.code == 200:

res = response.read()

return res

```py

			We can also add an `implementation` class to fetch content from a file on the local filesystem:

```

class LocalFileFetcher:

def fetch(self, path):

with open(path) as f:

res = f.read()

return res

```py

			Based on that, a `main` function with some testing code to show content using both *content fetchers* could look like the following:

```

def main():

url_fetcher = URLFetcher()

rc = ResourceContent(url_fetcher)

res = rc.get_content("http://python.org")

print(

f"Fetched content with {len(res)} characters"

)

localfs_fetcher = LocalFileFetcher()

rc = ResourceContent(localfs_fetcher)

pathname = os.path.abspath(__file__)

dir_path = os.path.split(pathname)[0]

path = os.path.join(dir_path, "file.txt")

res = rc.get_content(path)

print(

f"Fetched content with {len(res)} characters"

)

```py

			Let’s see a summary of the complete code of our example (the `ch04/bridge/bridge.py` file):

				1.  We import the modules we need for the program (`os`, `urllib.request`, and `typing.Protocol`).
				2.  We define the `ResourceContentFetcher` interface, using *protocols*, for the *Implementor*.
				3.  We define the `ResourceContent` class for the interface of the abstraction.
				4.  We define two implementation classes:
    *   `URLFetcher` for fetching content from a URL
    *   `LocalFileFetcher` for fetching content from the local filesystem
				5.  Finally, we add the `main()` function, as shown earlier, and the usual trick to call it.

			Here is a sample output when executing the `python` `ch04/bridge/bridge.py` command:

```

Fetched content with 51265 characters

Fetched content with 1327 characters

```py

			This is a basic illustration of how using the bridge pattern in your design, you can extract content from different sources and integrate the results in the same data manipulation system or user interface.
			The facade pattern
			As systems evolve, they can get very complex. It is not unusual to end up with a very large (and sometimes confusing) collection of classes and interactions. In many cases, we don’t want to expose this complexity to the client. This is where our next structural pattern comes to the rescue: **facade**.
			The facade design pattern helps us hide the internal complexity of our systems and expose only what is necessary to the client through a simplified interface. In essence, facade is an abstraction layer implemented over an existing complex system.
			Let’s take the example of the computer to illustrate things. A computer is a complex machine that depends on several parts to be fully functional. To keep things simple, the word “computer,” in this case, refers to an IBM derivative that uses a von Neumann architecture. Booting a computer is a particularly complex procedure. The CPU, main memory, and hard disk need to be up and running, the boot loader must be loaded from the hard disk to the main memory, the CPU must boot the operating system kernel, and so forth. Instead of exposing all this complexity to the client, we create a facade that encapsulates the whole procedure, making sure that all steps are executed in the right order.
			In terms of object design and programming, we should have several classes, but only the `Computer` class needs to be exposed to the client code. The client will only have to execute the `start()` method of the `Computer` class, for example, and all the other complex parts are taken care of by the facade `Computer` class.
			Real-world examples
			The facade pattern is quite common in life. When you call a bank or a company, you are usually first connected to the customer service department. The customer service employee acts as a facade between you and the actual department (billing, technical support, general assistance, and so on), where an employee will help you with your specific problem.
			As another example, a key used to turn on a car or motorcycle can also be considered a facade. It is a simple way of activating a system that is very complex internally. And, of course, the same is true for other complex electronic devices that we can activate with a single button, such as computers.
			In software, the `django-oscar-datacash` module is a Django third-party module that integrates with the **DataCash** payment gateway. The module has a gateway class that provides fine-grained access to the various DataCash APIs. On top of that, it also offers a facade class that provides a less granular API (for those who don’t want to mess with the details), and the ability to save transactions for auditing purposes.
			The `Requests` library is another great example of the facade pattern. It simplifies sending HTTP requests and handling responses, abstracting the complexities of the HTTP protocol. Developers can easily make HTTP requests without dealing with the intricacies of sockets or the underlying HTTP methods.
			Use cases for the facade pattern
			The most usual reason to use the facade pattern is to provide a single, simple entry point to a complex system. By introducing facade, the client code can use a system by simply calling a single method/function. At the same time, the internal system does not lose any functionality, it just encapsulates it.
			Not exposing the internal functionality of a system to the client code gives us an extra benefit: we can introduce changes to the system, but the client code remains unaware of and unaffected by the changes. No modifications are required to the client code.
			Facade is also useful if you have more than one layer in your system. You can introduce one facade entry point per layer and let all layers communicate with each other through their facades. That promotes **loose coupling** and keeps the layers as independent as possible.
			Implementing the facade pattern
			Assume that we want to create an operating system using a multi-server approach, similar to how it is done in MINIX 3 or GNU Hurd. A multi-server operating system has a minimal kernel, called the **microkernel**, which runs in privileged mode. All the other services of the system are following a server architecture (driver server, process server, file server, and so forth). Each server belongs to a different memory address space and runs on top of the microkernel in user mode. The pros of this approach are that the operating system can become more fault-tolerant, reliable, and secure. For example, since all drivers are running in user mode on a driver server, a bug in a driver cannot crash the whole system, nor can it affect the other servers. The cons of this approach are the performance overhead and the complexity of system programming, because the communication between a server and the microkernel, as well as between the independent servers, happens using message passing. Message passing is more complex than the shared memory model used in monolithic kernels such as Linux.
			We begin with a `Server` interface. Also, an `Enum` parameter describes the different possible states of a server. We use the `ABC` technique to forbid direct instantiation of the `Server` interface and make the fundamental `boot()` and `kill()` methods mandatory, assuming that different actions are needed to be taken for booting, killing, and restarting each server. Here is the code for these elements, the first important bits to support our implementation:

```

State = Enum(

"State",

"NEW RUNNING SLEEPING RESTART ZOMBIE",

)

# ...

class Server(ABC):

@abstractmethod

def __init__(self):

pass

def __str__(self):

return self.name

@abstractmethod

def boot(self):

pass

@abstractmethod

def kill(self, restart=True):

pass

```py

			A modular operating system can have a great number of interesting servers: a file server, a process server, an authentication server, a network server, a graphical/window server, and so forth. The following example includes two stub servers: `FileServer` and `ProcessServer`. Apart from the `boot()` and `kill()` methods all servers have, `FileServer` has a `create_file()` method for creating files, and `ProcessServer` has a `create_process()` method for creating processes.
			The `FileServer` class is as follows:

```

class FileServer(Server):

def __init__(self):

self.name = "FileServer"

self.state = State.NEW

def boot(self):

print(f"booting the {self}")

self.state = State.RUNNING

def kill(self, restart=True):

print(f"Killing {self}")

self.state = (

State.RESTART if restart else State.ZOMBIE

)

def create_file(self, user, name, perms):

msg = (

f"尝试创建文件 '{name}' "

f"for user '{user}' "

f"权限为 {perms}"

)

print(msg)

```py

			The `ProcessServer` class is as follows:

```

class ProcessServer(Server):

def __init__(self):

self.name = "ProcessServer"

self.state = State.NEW

def boot(self):

print(f"启动 {self}")

self.state = State.RUNNING

def kill(self, restart=True):

print(f"杀死 {self}")

self.state = (

State.RESTART if restart else State.ZOMBIE

)

def create_process(self, user, name):

msg = (

f"尝试创建进程 '{name}' "

f"for user '{user}'"

)

print(msg)

```py

			The `OperatingSystem` class is a facade. In its `__init__()`, all the necessary server instances are created. The `start()` method, used by the client code, is the entry point to the system. More wrapper methods can be added, if necessary, as access points to the services of the servers, such as the wrappers, `create_file()` and `create_process()`. From the client’s point of view, all those services are provided by the `OperatingSystem` class. The client should not be confused by unnecessary details such as the existence of servers and the responsibility of each server.
			The code for the `OperatingSystem` class is as follows:

```

class OperatingSystem:

"""门面模式”

def __init__(self):

self.fs = FileServer()

self.ps = ProcessServer()

def start(self):

[i.boot() for i in (self.fs, self.ps)]

def create_file(self, user, name, perms):

return self.fs.create_file(user, name, perms)

def create_process(self, user, name):

return self.ps.create_process(user, name)

```py

			As you are going to see in a minute, when we present a summary of the example, there are many dummy classes and servers. They are there to give you an idea about the required abstractions (`User`, `Process`, `File`, and so forth) and servers (`WindowServer`, `NetworkServer`, and so forth) for making the system functional.
			Finally, we add our main code for testing the design, as follows:

```

def main():

os = OperatingSystem()

os.start()

os.create_file("foo", "hello.txt", "-rw-r-r")

os.create_process("bar", "ls /tmp")

```py

			We are going to recapitulate the details of our implementation example; the full code is in the `ch04/facade.py` file:

				1.  We start with the imports we need.
				2.  We define the `State` constant using `Enum`, as shown earlier.
				3.  We then add the `User`, `Process`, and `File` classes, which do nothing in this minimal but functional example.
				4.  We define the abstract `Server` class, as shown earlier.
				5.  We then define the `FileServer` class and the `ProcessServer` class, which are both subclasses of `Server`.
				6.  We add two other dummy classes, `WindowServer` and `NetworkServer`.
				7.  Then we define our facade class, `OperatingSystem`, as shown earlier.
				8.  Finally, we add the main part of the code, where we use the facade we have defined.

			As you can see, executing the `python ch04/facade.py` command shows the messages produced by our two stub servers:

```

启动文件服务器

启动 ProcessServer

尝试为用户 'foo' 创建文件 'hello.txt'，权限为 -rw-r-r

操作系统类做得很好。客户端代码可以创建文件和进程，而无需了解操作系统内部细节，例如多个服务器的存在。更准确地说，客户端代码可以调用创建文件和进程的方法，但它们目前是虚拟的。作为一个有趣的练习，你可以实现这两种方法中的一种，甚至两种都可以。

            轻量级模式

            每当我们创建一个新的对象时，都需要额外分配内存。虽然虚拟内存从理论上为我们提供了无限的内存，但现实情况并非如此。如果一个系统的所有物理内存都用完了，它将开始与辅助存储（通常是**硬盘驱动器**（**HDD**））交换页面，由于主内存和 HDD 之间的性能差异，这在大多数情况下是不可接受的。**固态驱动器**（**SSD**）通常比 HDD 有更好的性能，但并不是每个人都期望使用 SSD。因此，SSD 不太可能在不久的将来完全取代 HDD。

            除了内存使用外，性能也是一个考虑因素。图形软件，包括计算机游戏，应该能够非常快速地渲染 3-D 信息（例如，有成千上万树木的森林，满是士兵的村庄，或者有很多汽车的城区）。如果 3-D 地形中的每个对象都是单独创建的，并且没有使用数据共享，性能将会非常低。

            作为软件工程师，我们应该通过编写更好的软件来解决软件问题，而不是强迫客户购买额外的或更好的硬件。**轻量级**设计模式是一种技术，通过在相似对象之间引入数据共享来最小化内存使用并提高性能。轻量级对象是一个包含状态无关、不可变（也称为**内在**）数据的共享对象。状态相关、可变（也称为**外在**）数据不应成为轻量级对象的一部分，因为这是无法共享的信息，因为它在每个对象中都是不同的。如果轻量级对象需要外在数据，它应该由客户端代码显式提供。

            以下是一个例子，可以帮助阐明如何实际使用轻量级模式。假设我们正在创建一个性能关键的游戏——例如，一个**第一人称射击游戏**（**FPS**）。在 FPS 游戏中，玩家（士兵）共享一些状态，例如表示和行为。例如，在《反恐精英》中，同一队的所有士兵（反恐分子与恐怖分子）看起来都一样（表示）。在同一个游戏中，所有士兵（两队）都有一些共同的动作，如跳跃、蹲下等（行为）。这意味着我们可以创建一个包含所有共同数据的轻量级对象。当然，士兵们也有许多不同的数据，这些数据对于每个士兵来说都是独特的，并且不会成为轻量级对象的一部分，例如武器、健康、位置等。

            现实世界中的例子

            轻量级模式是一种优化设计模式；因此，在非计算领域中很难找到一个好的例子。我们可以将轻量级模式视为现实生活中的缓存。例如，许多书店都有专门的书架，用于存放最新和最受欢迎的出版物。这是一个缓存。首先，你可以查看你正在寻找的书的专门书架，如果你找不到，你可以请书店老板帮忙。

            Exaile 音乐播放器使用轻量级模式来重用对象（在这种情况下，是音乐曲目），这些对象通过相同的 URL 进行标识。如果对象与现有对象具有相同的 URL，就没有必要创建新的对象，因此可以重用相同的对象以节省资源。

            轻量级模式的使用场景

            轻量级模式（Flyweight）主要关注性能和内存使用的提升。所有嵌入式系统（如手机、平板电脑、游戏机、微控制器等）以及性能关键的应用程序（如游戏、3-D 图形处理、实时系统等）都可以从中受益。

            《设计模式：可复用面向对象软件的基础》（*Gang of Four*，*GoF*）一书中列出了以下需要满足的要求，才能有效地使用轻量级模式：

                +   应用程序需要使用大量的对象。

                +   有如此多的对象，存储/渲染它们会非常昂贵。一旦移除了可变状态（因为如果需要，应该由客户端代码显式传递给轻量级对象），许多不同的对象组可以被相对较少的共享对象所替代。

                +   对象标识对于应用来说并不重要。我们不能依赖于对象标识，因为对象共享会导致标识比较失败（对客户端代码看起来不同的对象最终会有相同的标识）。

            实现享元模式

            让我们看看我们如何实现一个包含区域的汽车示例。我们将创建一个小型停车场来展示这个想法，确保整个输出在单个终端页面上可读。然而，无论停车场有多大，内存分配保持不变。

            缓存与享元模式的比较

            缓存是一种优化技术，它使用缓存来避免重新计算在早期执行步骤中已经计算过的结果。缓存并不专注于特定的编程范式，如**面向对象编程**（**OOP**）。在 Python 中，缓存可以应用于类方法和简单函数。

            享元是一种特定于面向对象编程的优化设计模式，它专注于共享对象数据。

            让我们开始编写这个示例的代码。

            首先，我们需要一个`Enum`参数来描述停车场中存在的三种不同类型的汽车：

```py
CarType = Enum(
    "CarType", "SUBCOMPACT COMPACT SUV"
)
```

            然后，我们将定义我们实现的核心类：`Car`。`pool`变量是对象池（换句话说，我们的缓存）。请注意，`pool`是一个类属性（一个所有实例共享的变量）。

            使用在`__init__()`之前被调用的特殊方法`__new__()`，我们将`Car`类转换为一个支持自引用的元类。这意味着`cls`引用了`Car`类。当客户端代码创建`Car`实例时，它们会传递汽车的类型作为`car_type`。汽车的类型用于检查是否已经创建了相同类型的汽车。如果是这样，则返回先前创建的对象；否则，将新的汽车类型添加到池中并返回：

```py
class Car:
    pool = dict()
    def __new__(cls, car_type):
        obj = cls.pool.get(car_type, None)
        if not obj:
            obj = object.__new__(cls)
            cls.pool[car_type] = obj
            obj.car_type = car_type
        return obj
```

            `render()`方法将用于在屏幕上渲染汽车。注意，所有未知于享元的信息都需要客户端代码显式传递。在这种情况下，每个汽车使用随机的`color`和位置的坐标（形式为`x`，`y`）。

            此外，请注意，为了使`render()`更有用，必须确保没有汽车渲染在彼此之上。把这当作一个练习。如果你想使渲染更有趣，可以使用图形工具包，如 Tkinter、Pygame 或 Kivy。

            `render()`方法定义如下：

```py
    def render(self, color, x, y):
        type = self.car_type
        msg = f"render a {color} {type.name} car at ({x}, {y})"
        print(msg)
```

            `main()`函数展示了如何使用轻量级模式。汽车的颜色是从预定义颜色列表中随机选择的值。坐标使用 1 到 100 之间的随机值。尽管渲染了 18 辆车，但只分配了 3 个内存。输出中的最后一行证明，在使用轻量级模式时，我们不能依赖于对象身份。`id()`函数返回对象的内存地址。这不是 Python 的默认行为，因为默认情况下，`id()`为每个对象返回一个唯一的 ID（实际上是对象的内存地址的整数）。在我们的情况下，即使两个对象看起来不同，如果它们属于同一个`car_type`，它们实际上具有相同的身份。当然，仍然可以使用不同的身份比较来比较不同家族的对象，但这只有在客户端知道实现细节的情况下才可能。

            我们的示例`main()`函数的代码如下：

```py
def main():
    rnd = random.Random()
    colors = [
        "white",
        "black",
        "silver",
        "gray",
        "red",
        "blue",
        "brown",
        "beige",
        "yellow",
        "green",
    ]
    min_point, max_point = 0, 100
    car_counter = 0
    for _ in range(10):
        c1 = Car(CarType.SUBCOMPACT)
        c1.render(
            random.choice(colors),
            rnd.randint(min_point, max_point),
            rnd.randint(min_point, max_point),
        )
        car_counter += 1
    for _ in range(3):
        c2 = Car(CarType.COMPACT)
        c2.render(
            random.choice(colors),
            rnd.randint(min_point, max_point),
            rnd.randint(min_point, max_point),
        )
        car_counter += 1
    for _ in range(5):
        c3 = Car(CarType.SUV)
        c3.render(
            random.choice(colors),
            rnd.randint(min_point, max_point),
            rnd.randint(min_point, max_point),
        )
        car_counter += 1
    print(f"cars rendered: {car_counter}")
    print(
        f"cars actually created: {len(Car.pool)}"
    )
    c4 = Car(CarType.SUBCOMPACT)
    c5 = Car(CarType.SUBCOMPACT)
    c6 = Car(CarType.SUV)
    print(
        f"{id(c4)} == {id(c5)}? {id(c4) == id(c5)}"
    )
    print(
        f"{id(c5)} == {id(c6)}? {id(c5) == id(c6)}"
    )
```

            下面是完整代码列表（`ch04/flyweight.py`文件）的回顾，以展示如何实现和使用轻量级模式：

                1.  我们需要导入几个模块：`random`和`Enum`（来自`enum`模块）。

                1.  我们为汽车类型定义了`Enum`。

                1.  然后我们有`Car`类，它具有`pool`属性以及`__new__()`和`render()`方法。

                1.  在`main`函数的第一部分，我们定义了一些变量并渲染了一组小型车。

                1.  `main`函数的第二部分。

                1.  `main`函数的第三部分。

                1.  最后，我们添加`main`函数的第四部分。

            执行`python ch04/flyweight.py`命令的输出显示了渲染对象的类型、随机颜色和坐标，以及相同/不同家族的轻量级对象之间的身份比较结果：

```py
render a gray SUBCOMPACT car at (25, 79)
render a black SUBCOMPACT car at (31, 99)
render a brown SUBCOMPACT car at (16, 74)
render a green SUBCOMPACT car at (10, 1)
render a gray SUBCOMPACT car at (55, 38)
render a red SUBCOMPACT car at (30, 45)
render a brown SUBCOMPACT car at (17, 78)
render a gray SUBCOMPACT car at (14, 21)
render a gray SUBCOMPACT car at (7, 28)
render a gray SUBCOMPACT car at (22, 50)
render a brown COMPACT car at (75, 26)
render a red COMPACT car at (22, 61)
render a white COMPACT car at (67, 87)
render a beige SUV car at (23, 93)
render a white SUV car at (37, 100)
render a red SUV car at (33, 98)
render a black SUV car at (77, 22)
render a green SUV car at (16, 51)
cars rendered: 18
cars actually created: 3
4493672400 == 4493672400? True
4493672400 == 4493457488? False
```

            由于颜色和坐标是随机的，并且对象身份取决于内存映射，因此不要期望看到相同的输出。

            代理模式

            **代理**设计模式的名称来源于用于在访问实际对象之前执行重要操作的**代理**对象（也称为**代表**）。有四种著名的代理类型。具体如下：

                1.  一个**虚拟代理**，它使用**延迟初始化**来推迟在真正需要时创建计算密集型对象。

                1.  一个**保护/防护代理**，用于控制对敏感对象的访问。

                1.  一个**远程代理**，作为实际存在于不同地址空间中的对象的本地表示（例如，网络服务器）。

                1.  一个**智能（引用）代理**，在访问对象时执行额外操作。此类操作的例子包括引用计数和线程安全检查。

            现实世界例子

            **芯片**卡是保护代理在现实生活中应用的一个好例子。借记/信用卡包含一个芯片，首先需要由 ATM 或读卡器读取。芯片验证后，需要输入密码（PIN）才能完成交易。这意味着，如果没有物理出示卡片并知道 PIN，就无法进行任何交易。

            使用现金购买和交易的银行支票是远程代理的一个例子。支票可以访问银行账户。

            在软件中，Python 的`weakref`模块包含一个`proxy()`方法，它接受一个输入对象并返回一个智能代理。弱引用是向对象添加引用计数支持的推荐方式。

            代理模式的用例

            由于至少有四种常见的代理类型，因此代理设计模式有许多用例。

            当使用私有网络或云来创建分布式系统时，这种模式被使用。在分布式系统中，一些对象存在于本地内存中，而一些对象存在于远程计算机的内存中。如果我们不希望客户端代码意识到这些差异，我们可以创建一个远程代理来隐藏/封装它们，使应用程序的分布式特性变得透明。

            当我们的应用程序由于昂贵对象的早期创建而遭受性能问题时，代理模式也很有用。通过使用虚拟代理进行延迟初始化，只在需要时创建对象，可以给我们带来显著的性能提升。

            作为第三个案例，这种模式用于检查用户是否有足够的权限访问某些信息。如果我们的应用程序处理敏感信息（例如，医疗数据），我们希望确保尝试访问/修改它的用户能够这样做。保护/防护代理可以处理所有与安全相关的操作。

            这种模式适用于我们的应用程序（或库、工具包、框架等）使用多个线程，并且我们希望将线程安全的问题从客户端代码转移到应用程序上。在这种情况下，我们可以创建一个智能代理来隐藏线程安全的复杂性，不让客户端知道。

            **对象关系映射**（ORM）API 也是如何使用远程代理的一个例子。许多流行的 Web 框架（Django、Flask、FastAPI...）使用 ORM 来提供面向对象的数据库访问。ORM 充当一个代理，可以位于任何地方，无论是本地服务器还是远程服务器。

            实现代理模式——虚拟代理

            在 Python 中创建虚拟代理有许多方法，但我总是喜欢关注惯用/Pythonic 的实现。这里展示的代码基于[stackoverflow.com](http://stackoverflow.com)网站的用户 Cyclone 给出的一个关于“Python memoising/deferred lookup property decorator”问题的优秀答案。

            注意

            在本节中，术语*属性*、*变量*和*属性*可以互换使用。

            首先，我们创建了一个`LazyProperty`类，它可以作为装饰器使用。当它装饰一个属性时，`LazyProperty`会在第一次使用时延迟加载该属性，而不是立即加载。`__init__()`方法创建了两个变量，用作初始化属性的方法的别名：`method`是实际方法的别名，`method_name`是方法名的别名。为了更好地理解这两个别名是如何使用的，将它们的值打印到输出中（取消注释代码中该部分的两个注释行）：

```py
class LazyProperty:
    def __init__(self, method):
        self.method = method
        self.method_name = method.__name__
        # print(f"function overriden: {self.method}")
        # print(f"function's name: {self.method_name}")
```

            `LazyProperty`类实际上是一个描述符。描述符是在 Python 中用来覆盖其属性访问方法（`__get__()`、`__set__()`和`__delete__()`）默认行为的推荐机制。`LazyProperty`类仅覆盖`__set__()`，因为这是它需要覆盖的唯一访问方法。换句话说，我们不需要覆盖所有访问方法。`__get__()`方法访问底层方法想要分配的属性值，并使用`setattr()`手动进行分配。`__get()__`实际上执行的操作非常巧妙：它用值替换了方法！这意味着属性不仅被延迟加载，而且只能设置一次。我们稍后会看到这意味着什么。

```py
    def __get__(self, obj, cls):
        if not obj:
            return None
        value = self.method(obj)
        # print(f'value {value}')
        setattr(obj, self.method_name, value)
        return value
```

            再次，取消注释代码中该部分的注释行以获取一些额外信息。

            然后，`Test`类展示了我们如何使用`LazyProperty`类。有三个属性：`x`、`y`和`_resource`。我们希望`_resource`变量能够延迟加载；因此，我们将其初始化为`None`，如下所示：

```py
class Test:
    def __init__(self):
        self.x = "foo"
        self.y = "bar"
        self._resource = None
```

            `resource()`方法被`LazyProperty`类装饰。为了演示目的，`LazyProperty`类将`_resource`属性初始化为一个元组，如下所示。通常，这会是一个缓慢/昂贵的初始化（数据库、图形等）。

```py
    @LazyProperty
    def resource(self):
        print("initializing self._resource...")
        print(f"... which is: {self._resource}")
        self._resource = tuple(range(5))
        return self._resource
```

            如下所示的`main()`函数展示了延迟初始化的行为：

```py
def main():
    t = Test()
    print(t.x)
    print(t.y)
    # do more work...
    print(t.resource)
    print(t.resource)
```

            注意，覆盖`__get()__`访问方法使得将`resource()`方法视为一个简单属性成为可能（我们可以使用`t.resource`而不是`t.resource()`）。

            让我们回顾一下示例代码（在`ch04/proxy/proxy_lazy.py`中）：

                1.  我们定义了`LazyProperty`类。

                1.  我们定义了带有`resource()`方法的`Test`类，并使用`LazyProperty`对其进行装饰。

                1.  我们添加了主函数来测试我们的设计示例。

            如果你能够执行示例的原始版本（其中为了更好地理解而添加的行被注释），使用`python ch04/proxy/proxy_lazy.py`命令，你将得到以下输出：

```py
foo
bar
initializing self._resource...
... which is: None
(0, 1, 2, 3, 4)
(0, 1, 2, 3, 4)
```

            根据这个输出，我们可以看到以下内容：

                +   `_resource`变量确实是在我们使用`t.resource`时初始化的，而不是在`t`实例创建时。

                +   第二次使用 `t.resource` 时，变量不再重新初始化。这就是为什么初始化字符串只初始化 `self._resource` 一次的原因。

            其他信息

            在面向对象编程（OOP）中，存在两种基本的懒加载初始化方式。具体如下：

            - **在实例级别**：这意味着对象的属性是懒加载初始化的，但属性具有对象作用域。同一类的每个实例（对象）都有自己的（不同的）属性副本。

            - **在类或模块级别**：在这种情况下，我们不希望每个实例有不同的副本，而是所有实例共享相同的属性，该属性是懒加载初始化的。这种情况在本章中没有涉及。如果你对此感兴趣，可以考虑将其作为练习。

            由于使用代理模式的可能性有很多，让我们看看另一个例子。

            实现代理模式 – 保护代理

            作为第二个例子，让我们实现一个简单的保护代理来查看和添加用户。服务提供了两种选项：

                +   **查看用户列表**：此操作不需要特殊权限

                +   **添加新用户**：此操作要求客户端提供特殊秘密消息

            `SensitiveInfo` 类包含我们想要保护的信息。`users` 变量是现有用户列表。`read()` 方法打印用户列表。`add()` 方法将新用户添加到列表中。该类的代码如下：

```py
class SensitiveInfo:
    def __init__(self):
        self.users = ["nick", "tom", "ben", "mike"]
    def read(self):
        nb = len(self.users)
        print(f"There are {nb} users: {' '.join(self.users)}")
    def add(self, user):
        self.users.append(user)
        print(f"Added user {user}")
```

            `Info` 类是 `SensitiveInfo` 的保护代理。秘密变量是客户端代码添加新用户所需知道/提供的消息。

            注意，这只是一个例子。在现实中，你永远不应该做以下事情：

                +   在源代码中存储密码

                +   以明文形式存储密码

                +   使用弱（例如，MD5）或自定义形式的加密

            在 `Info` 类中，如我们接下来看到的，`read()` 方法是对 `SensitiveInfo.read()` 的包装，而 `add()` 方法确保只有当客户端代码知道秘密消息时，才能添加新用户：

```py
class Info:
    def __init__(self):
        self.protected = SensitiveInfo()
        self.secret = "0xdeadbeef"
    def read(self):
        self.protected.read()
    def add(self, user):
        sec = input("what is the secret? ")
        if sec == self.secret:
            self.protected.add(user)
        else:
            print("That's wrong!")
```

            `main()` 函数展示了客户端代码如何使用代理模式。客户端代码创建 `Info` 类的实例，并使用显示的菜单读取列表、添加新用户或退出应用程序。让我们考虑以下代码：

```py
def main():
    info = Info()
    while True:
        print("1\. read list |==| 2\. add user |==| 3\. quit")
        key = input("choose option: ")
        if key == "1":
            info.read()
        elif key == "2":
            name = input("choose username: ")
            info.add(name)
        elif key == "3":
            exit()
        else:
            print(f"unknown option: {key}")
```

            让我们回顾一下完整的代码（`ch04/proxy/proxy_protection.py`）：

                1.  首先，我们定义 `SensitiveInfo` 类。

                1.  然后，我们有 `Info` 类的代码。

                1.  最后，我们添加主函数以及我们的测试代码。

            我们可以在以下示例中看到程序执行 `python ch04/proxy/proxy_protection.py` 命令时的输出样本：

```py
1\. read list |==| 2\. add user |==| 3\. quit
choose option: 1
There are 4 users: nick tom ben mike
1\. read list |==| 2\. add user |==| 3\. quit
choose option: 2
choose username: tom
what is the secret? 0xdeadbeef
Added user tom
1\. read list |==| 2\. add user |==| 3\. quit
choose option: 3
```

            你已经发现了可以解决以改进我们的保护代理示例的缺陷或缺失功能吗？以下是一些建议：

                +   这个示例有一个非常大的安全漏洞。没有任何东西阻止客户端代码通过直接创建`SensitiveInfo`的实例来绕过应用程序的安全。改进这个示例以防止这种情况。一种方法是通过使用`abc`模块禁止直接实例化`SensitiveInfo`。在这种情况下还需要进行哪些代码更改？

                +   一个基本的安全规则是，我们永远不应该存储明文密码。只要我们知道使用哪些库，安全地存储密码并不难。如果你对安全感兴趣，尝试实现一种安全的方式来外部存储秘密消息（例如，在文件或数据库中）。

                +   应用程序仅支持添加新用户，但关于删除现有用户怎么办？添加一个`remove()`方法。

            实现代理模式 – 远程代理

            想象我们正在构建一个文件管理系统，客户端可以在远程服务器上执行文件操作。这些操作可能包括读取文件、写入文件和删除文件。远程代理隐藏了网络请求的复杂性，对客户端来说。

            我们首先创建一个接口，定义可以在远程服务器上执行的操作，`RemoteServiceInterface`，以及实现该接口的类`RemoteService`以提供实际服务。

            接口定义如下：

```py
from abc import ABC, abstractmethod
class RemoteServiceInterface(ABC):
    @abstractmethod
    def read_file(self, file_name):
        pass
    @abstractmethod
    def write_file(self, file_name, contents):
        pass
    @abstractmethod
    def delete_file(self, file_name):
        pass
```

            `RemoteService`类定义如下（为了简单起见，方法仅返回一个字符串，但通常，你会在远程服务上进行特定的文件处理代码）：

```py
class RemoteService(RemoteServiceInterface):
    def read_file(self, file_name):
        # Implementation for reading a file from the server
        return "Reading file from remote server"
    def write_file(self, file_name, contents):
        # Implementation for writing to a file on the server
        return "Writing to file on remote server"
    def delete_file(self, file_name):
        # Implementation for deleting a file from the server
        return "Deleting file from remote server"
```

            然后，我们为代理定义了`ProxyService`。它实现了`RemoteServiceInterface`接口，并作为`RemoteService`的代理，处理与后者的通信：

```py
class ProxyService(RemoteServiceInterface):
    def __init__(self):
        self.remote_service = RemoteService()
    def read_file(self, file_name):
        print("Proxy: Forwarding read request to RemoteService")
        return self.remote_service.read_file(file_name)
    def write_file(self, file_name, contents):
        print("Proxy: Forwarding write request to RemoteService")
        return self.remote_service.write_file(file_name, contents)
    def delete_file(self, file_name):
        print("Proxy: Forwarding delete request to RemoteService")
        return self.remote_service.delete_file(file_name)
```

            客户端与`ProxyService`组件交互，就像它是`RemoteService`一样，并不知道实际服务的远程性质。代理处理与远程服务的通信，可能包括添加日志、访问控制或缓存。为了测试，我们可以添加以下代码，基于创建`ProxyService`的实例：

```py
if __name__ == "__main__":
    proxy = ProxyService()
    print(proxy.read_file("example.txt"))
```

            让我们回顾一下实现过程（完整代码位于`ch04/proxy/proxy_remote.py`）：

                1.  我们首先定义接口，`RemoteServiceInterface`，以及一个实现该接口的类，`RemoteService`。

                1.  然后，我们定义了`ProxyService`类，它也实现了`RemoteService`接口。

                1.  最后，我们添加一些代码来测试代理对象。

            通过运行`python ch04/proxy/proxy_remote.py`，让我们看看示例的结果：

```py
Proxy: Forwarding read request to RemoteService
Reading file from remote server
```

            这成功了。这个轻量级的示例有效地展示了如何实现远程代理用例。

            实现代理模式 – 智能代理

            让我们考虑一个场景，在你的应用程序中有一个共享资源，例如数据库连接。每次对象访问这个资源时，你都想跟踪资源引用的数量。一旦没有更多引用，资源就可以安全地释放或关闭。智能代理将帮助管理这个数据库连接的引用计数，确保它只在所有引用释放后关闭。

            在上一个示例中，我们需要一个定义访问数据库操作的接口，`DBConnectionInterface`，以及一个代表实际数据库连接的类，`DBConnection`。

            对于接口，我们使用`Protocol`（从`ABC`方式更改）：

```py
from typing import Protocol
class DBConnectionInterface(Protocol):
    def exec_query(self, query):
        ...
```

            数据库连接的类如下：

```py
class DBConnection:
    def __init__(self):
        print("DB connection created")
    def exec_query(self, query):
        return f"Executing query: {query}"
    def close(self):
        print("DB connection closed")
```

            然后，我们定义了`SmartProxy`类；它也实现了`DBConnectionInterface`接口（请参阅`exec_query()`方法）。我们使用这个类来管理引用计数和`DBConnection`对象的访问。它确保在首次执行查询时按需创建`DBConnection`对象，并且只有当没有更多引用时才关闭。代码如下：

```py
class SmartProxy:
    def __init__(self):
        self.cnx = None
        self.ref_count = 0
    def access_resource(self):
        if self.cnx is None:
            self.cnx = DBConnection()
        self.ref_count += 1
        print(f"DB connection now has {self.ref_count} references.")
    def exec_query(self, query):
        if self.cnx is None:
            # Ensure the connection is created
            # if not already
            self.access_resource()
        result = self.cnx.exec_query(query)
        print(result)
        # Decrement reference count after
        # executing query
        self.release_resource()
        return result
    def release_resource(self):
        if self.ref_count > 0:
            self.ref_count -= 1
            print("Reference released...")
            print(f"{self.ref_count} remaining refs.")
        if self.ref_count == 0 and self.cnx is not None:
            self.cnx.close()
            self.cnx = None
```

            现在，我们可以添加一些代码来测试实现：

```py
if __name__ == "__main__":
    proxy = SmartProxy()
    proxy.exec_query("SELECT * FROM users")
    proxy.exec_query("UPDATE users SET name = 'John Doe' WHERE id = 1")
```

            让我们回顾一下实现（完整代码在`ch04/proxy/proxy_smart.py`中）：

                1.  我们首先定义接口，`DBConnectionInterface`，以及一个实现它的类，代表数据库连接，`DBConnection`。

                1.  然后，我们定义了`SmartProxy`类，它也实现了`DBConnectionInterface`。

                1.  最后，我们添加一些代码来测试代理对象。

            让我们通过运行`python ch04/proxy/proxy_smart.py`来查看示例的结果：

```py
DB connection created
DB connection now has 1 references.
Executing query: SELECT * FROM users
Reference released...
0 remaining refs.
DB connection closed
DB connection created
DB connection now has 1 references.
Executing query: UPDATE users SET name = 'John Doe' WHERE id = 1
Reference released...
0 remaining refs.
DB connection closed
```

            这是对代理模式的一次另类演示。在这里，它帮助我们实现了一种改进的解决方案，适用于数据库连接在不同应用程序部分之间共享且需要谨慎管理以避免耗尽数据库资源或泄露连接的场景。

            摘要

            结构模式对于创建干净、可维护和可扩展的代码至关重要。它们为你在日常编码中遇到的许多挑战提供了解决方案。

            首先，适配器模式作为一种灵活的解决方案，用于协调不匹配的接口。我们可以使用这种模式来弥合过时遗留系统与现代接口之间的差距，从而促进更加紧密和易于管理的软件系统。

            然后，我们讨论了装饰器模式，这是一种方便地扩展对象行为而不使用继承的方法。Python 通过其内置的装饰器功能，甚至进一步扩展了装饰器概念，允许我们扩展任何可调用的行为而不使用继承或组合。装饰器模式是实现横切关注点的绝佳解决方案，因为它们是通用的，并且不适合 OOP 范式。我们在“装饰器模式的使用案例”部分提到了几个横切关注点的类别。我们看到了装饰器如何帮助我们保持函数的整洁，同时不牺牲性能。

            与适配器模式相似，桥接模式在定义抽象及其实现时有所不同，它是在一开始就以一种解耦的方式定义抽象和实现，以便两者可以独立变化。桥接模式在编写操作系统和设备驱动程序、GUI 和网站构建器等问题的软件时非常有用，在这些领域中我们有多套主题，并且需要根据某些属性更改网站的主题。我们在内容提取和管理领域讨论了一个例子，其中我们定义了一个抽象的接口、一个实现者的接口和两个实现。

            外观模式非常适合为希望使用复杂系统但不需要了解系统复杂性的客户端代码提供一个简单的接口。计算机就是一个外观，因为我们只需要按下一个按钮就可以打开它。所有其他硬件复杂性都由 BIOS、引导加载程序和系统软件的其他组件透明地处理。还有更多现实生活中的外观例子，比如当我们连接到银行或公司的客户服务部门时，以及我们用来启动车辆的钥匙。我们介绍了一个多服务器操作系统使用的接口实现。

            通常，当应用程序需要创建大量计算成本高昂的对象，而这些对象共享许多属性时，我们会使用享元模式。关键点是区分不可变（共享）属性和可变属性。我们看到了如何实现一个支持三个不同汽车家族的汽车渲染器。通过显式地将可变的颜色和 x、y 属性提供给`render()`方法，我们只创建了 3 个不同的对象，而不是 18 个。虽然这可能看起来不是很大的胜利，但想象一下如果汽车有 2,000 辆而不是 18 辆会怎样。

            我们以代理模式结束。我们讨论了代理模式的几个用例，包括性能、安全性和如何向用户提供简单的 API。我们为通常需要的四种代理类型中的每一种都看到了实现示例：虚拟代理、保护代理、远程服务代理和智能代理。

            在下一章中，我们将探讨行为设计模式，这些模式涉及对象交互和算法。

```py

```
