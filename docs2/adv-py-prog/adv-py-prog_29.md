# 第二十六章：*第二十六章* 观察者模式

当我们想要在对象状态发生变化时通知/通知所有利益相关者（一个对象或一组对象）时，我们使用观察者模式。观察者模式的一个重要特性是，订阅者/观察者的数量以及订阅者是谁可能会变化，并且可以在运行时更改。

在本章中，我们将学习这种设计模式，并将其与我们过去看到的一个类似模式进行比较，即 MVC 模式，并使用它来实现数据格式化。

具体来说，我们将涵盖以下主题：

+   理解观察者模式

+   现实世界例子

+   用例

+   实现方式

# 技术要求

本章的代码文件可以在[`github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter26`](https://github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter26)找到。

# 理解观察者模式

当我们需要在另一个对象的状态发生变化时更新一组对象时，**模型-视图-控制器**（**MVC**）模式提供了一个流行的解决方案。假设我们在两个**视图**中使用相同的*模型*数据；例如，在一个饼图和一个电子表格中。每当模型被修改时，两个视图都需要更新。这就是观察者模式的作用。

观察者模式描述了单个对象（发布者，也称为**主题**或**可观察者**）与一个或多个对象（订阅者，也称为**观察者**）之间的发布-订阅关系。

在 MVC 的情况下，发布者是模型，而订阅者是视图。本章中我们将讨论其他一些例子。

观察者模式背后的思想与关注点分离原则背后的思想相同；也就是说，为了增加发布者和订阅者之间的解耦，并使在运行时添加/删除订阅者变得容易。让我们看看几个这样的例子。

# 现实世界例子

在现实中，拍卖类似于观察者模式。每个拍卖竞标者都有一个编号的 paddle，每当他们想要出价时就会举起。每当竞标者举起 paddle 时，拍卖师作为主题通过更新出价价格并向所有竞标者（订阅者）广播新价格来行动。

在软件中，我们可以至少举出两个例子：

+   Kivy，一个用于开发用户界面的 Python 框架，有一个名为**Properties**的模块，它实现了观察者模式。使用这种技术，您可以指定当属性值发生变化时应该发生什么。

+   RabbitMQ 库可以用于向应用程序添加异步消息支持。它支持多种消息协议，如 HTTP 和 AMQP。RabbitMQ 可以在 Python 应用程序中用于实现发布-订阅模式，这实际上就是观察者设计模式（[j.mp/rabbitmqobs](http://j.mp/rabbitmqobs)）。

在下一节中，我们将讨论何时可以使用并应该使用这种设计模式。

# 用例

我们通常在想要通知/更新关于在某个对象（主题/发布者/可观察对象）上发生变化的**一个或多个对象**（观察者/订阅者）时使用观察者模式。观察者的数量以及观察者是谁可能会变化，并且可以动态更改。

我们可以想到许多情况下观察者模式是有用的。其中一个用例是**新闻源**。通过 RSS、Atom 或其他相关格式，你可以订阅一个源，每次它更新时，你都会收到关于更新的通知。

在社交网络中，也存在同样的概念。如果你通过社交网络服务与另一个人建立了联系，并且你的联系更新了某些内容，你会收到通知。无论这个联系是你关注的 Twitter 用户、Facebook 上的真实朋友，还是在 LinkedIn 上的商业同事，都无关紧要。

**事件驱动系统**是另一个通常使用观察者模式的例子。在这样的系统中，你有**监听器**，它们**监听**特定的事件。当它们监听的事件被创建时，监听器会被触发。这可以是按特定的键（在键盘上）、移动鼠标等等。事件扮演发布者的角色，而监听器扮演观察者的角色。在这种情况下，关键点是多个监听器（观察者）可以附加到单个事件（发布者）上。

最后，在下一节中，我们将实现一个数据格式化器。

# 实现

这里描述的思想基于 ActiveState Python Observer 代码配方([`code.activestate.com/`](https://code.activestate.com/))。有一个默认的格式化器，以十进制格式显示值。然而，我们可以添加/注册更多的格式化器。在这个例子中，我们将添加十六进制格式化器和二进制格式化器。每当默认格式化器的值更新时，注册的格式化器将被通知并采取行动。在这种情况下，操作是在相关格式中显示新值。

观察者模式是其中继承有意义的模式之一。我们可以有一个包含添加、删除和通知观察者公共功能的基类`Publisher`。我们的`DefaultFormatter`类从`Publisher`派生，并添加了格式化器特定的功能。我们还可以根据需要动态添加和删除观察者。

我们将从`Publisher`类开始。观察者被保存在观察者列表中。`add()`方法注册一个新的观察者，如果它已经存在则抛出错误。`remove()`方法注销现有的观察者，如果不存在则抛出异常。最后，`notify()`方法通知所有观察者关于变化的信息。这将在下面的代码块中展示：

```py
class Publisher:  
    def __init__(self):  
        self.observers = []  

    def add(self, observer):  
        if observer not in self.observers:  
            self.observers.append(observer)  
        else:  
            print(f'Failed to add: {observer}')  

    def remove(self, observer):  
        try:  
            self.observers.remove(observer)  
        except ValueError:  
            print(f'Failed to remove: {observer}')  

    def notify(self):  
        [o.notify(self) for o in self.observers]  
```

让我们继续 `DefaultFormatter` 类。`__init__()` 方法首先做的事情是调用基类的 `__init__()` 方法，因为在 Python 中这不会自动完成。

`DefaultFormatter` 实例有一个名称，这使得我们更容易跟踪其状态。我们使用 `_data` 变量来表示它不应该被直接访问。请注意，在 Python 中这始终是可能的，但其他开发者没有理由这样做，因为代码已经声明了他们不应该这样做。`DefaultFormatter` 将 `_data` 变量视为整数，默认值是 `0`：

```py
class DefaultFormatter(Publisher):  
     def __init__(self, name):  
         Publisher.__init__(self)  
         self.name = name  
         self._data = 0
```

`__str__()` 方法返回有关发布者名称和 `_data` 属性值的信息。`type(self).__name__` 是一个方便的技巧，可以获取类的名称而不需要硬编码。这是那些使你的代码更容易维护的技巧之一：

```py
def __str__(self):
     return f"{type(self).__name__}: '{self.name}' \
       has data = 
     {self._data}"
```

有两个 `data()` 方法。第一个方法使用 `@property` 装饰器来提供对 `_data` 变量的读取访问。使用这种方式，我们可以直接执行 `object.data` 而不是 `object.data()`：

```py
@property  
def data(self):  
    return self._data
```

第二个 `data()` 方法更有趣。它使用 `@setter` 装饰器，每次使用赋值运算符（`=`）为 `_data` 变量分配新值时都会调用它。此方法还尝试将新值转换为整数，并在操作失败时进行异常处理：

```py
@data.setter  
def data(self, new_value):  
    try:  
       self._data = int(new_value)  
    except ValueError as e:  
       print(f'Error: {e}')  
    else:
       self.notify()
```

下一步是添加观察者。`HexFormatter` 和 `BinaryFormatter` 的功能非常相似。它们之间的唯一区别是它们如何格式化发布者接收到的数据值——即分别以十六进制和二进制格式：

```py
class HexFormatterObs:  
    def notify(self, publisher):  
        value = hex(publisher.data)
        print(f"{type(self).__name__}: '{publisher.name}' \
           has now hex data = {value}")  

class BinaryFormatterObs:  
    def notify(self, publisher):  
        value = bin(publisher.data)
        print(f"{type(self).__name__}: '{publisher.name}' \
          has now bin data = {value}")
```

为了帮助我们使用这些类，`main()` 函数最初创建了一个名为 `test1` 的 `DefaultFormatter` 实例，之后附加（并断开）了两个可用的观察者。我们还添加了一些异常处理，以确保当用户传递错误数据时应用程序不会崩溃。

代码如下：

```py
def main():  
    df = DefaultFormatter('test1')  
    print(df)  

    print()  
    hf = HexFormatterObs()  
    df.add(hf)  
    df.data = 3  
    print(df)  

    print()  
    bf = BinaryFormatterObs()  
    df.add(bf)  
    df.data = 21  
    print(df)
```

此外，尝试添加相同的观察者两次或删除不存在的观察者等任务不应导致崩溃：

```py
print()  
df.remove(hf)  
df.data = 40  
print(df)  

print()  
df.remove(hf)  
df.add(bf)  

df.data = 'hello'  
print(df)  

print()  
df.data = 15.8  
print(df)
```

在我们运行此代码并观察输出之前，让我们回顾一下完整的代码（`observer.py` 文件）：

1.  首先，我们定义了 `Publisher` 类。

1.  然后，我们定义了 `DefaultFormatter` 类，以及它的 `special __init__` 和 `__str__` 方法。

1.  我们将 `data` 属性的获取器和设置器方法添加到 `DefaultFormatter` 类中。

1.  我们定义了我们的两个观察者类。

1.  最后，我们处理程序的主要部分。

执行 `python observer.py` 命令会给我们以下输出：

```py
DefaultFormatter: 'test1' has data = 0
HexFormatterObs: 'test1' has now hex data = 0x3
DefaultFormatter: 'test1' has data = 3
HexFormatterObs: 'test1' has now hex data = 0x15
BinaryFormatterObs: 'test1' has now bin data = 0b10101
DefaultFormatter: 'test1' has data = 21
BinaryFormatterObs: 'test1' has now bin data = 0b101000
DefaultFormatter: 'test1' has data = 40
Failed to remove: <__main__.HexFormatterObs object at 
0x7fe6e4c9d670>
Failed to add: <__main__.BinaryFormatterObs object at 
0x7fe6e4c9d5b0>
Error: invalid literal for int() with base 10: 'hello'
DefaultFormatter: 'test1' has data = 40
BinaryFormatterObs: 'test1' has now bin data = 0b1111
DefaultFormatter: 'test1' has data = 15
```

在这里，我们可以看到，随着额外观察者的添加，显示的输出（更多且更相关）也越来越多，当移除观察者时，它将不再被通知。这正是我们想要的：可以按需启用/禁用的运行时通知。

应用程序的防御性编程部分看起来也运行良好。尝试做一些有趣的事情，比如移除一个不存在的观察者或者重复添加同一个观察者，是不被允许的。显示的消息并不十分友好，但我将这项任务留给你作为练习，使其更加友好。当 API 期望一个数字时尝试传递一个字符串等运行时错误也被妥善处理，不会导致应用程序崩溃/终止。

如果这个示例是交互式的，将会更有趣。甚至一个简单的菜单，允许用户在运行时附加/移除观察者并修改`DefaultFormatter`的值，也会很棒，因为运行时特性会变得更加明显。请随意这样做。

另一个不错的练习是添加更多的观察者。例如，你可以添加一个八进制格式器、一个罗马数字格式器或任何其他使用你喜欢的表示方法的观察者。发挥创意！通过这一点，我们完成了对观察者模式的讨论。

# 摘要

在本章中，我们介绍了观察者设计模式，包括许多示例，例如 Kivy，一个用于开发创新用户界面的框架，以及其**属性**概念和模块，以及 RabbitMQ 的 Python 绑定（我们提到了一个用于实现发布-订阅或观察者模式的 RabbitMQ 的具体示例）。

我们还学习了如何使用观察者模式创建可以在运行时附加和移除的数据格式器，以丰富对象的行为。希望你会觉得推荐的练习很有趣。

这也标志着本书的结束。恭喜你坚持到最后，希望所涵盖的材料对你提升 Python 技能有所帮助！

# 问题

回答以下问题以测试你对本章内容的了解：

1.  观察者模式的主要动机是什么？

1.  当目标组件发生变化时，观察者模式与 MVC 模式在更新应用程序的其他组件方面有何不同？

1.  在 Python 示例中，值格式器的观察者模式是如何实现的？
