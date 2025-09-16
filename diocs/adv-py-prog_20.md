# 第十七章：*第十七章*：构建器模式

在上一章中，我们介绍了前两个创建型模式——工厂方法和抽象工厂，它们都提供了在非平凡情况下改进对象创建方式的方法。另一方面，构建器设计模式，正如我们将在本章中讨论的，对于管理由多个部分组成且需要按顺序实现的对象非常有用。通过解耦对象的构建和其表示，构建器模式允许我们多次重用构建过程。

正如上一章一样，我们将讨论使用此设计模式的实际应用，以及我们自己动手实现一个实例。

在本章中，我们将讨论以下主题：

+   理解构建器模式

+   现实世界例子

+   用例

+   实现一个订单应用

到本章结束时，我们将了解如何使用构建器模式及其实际益处。

# 技术要求

本章的代码文件可以通过此链接访问：

[`github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter17`](https://github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter17)

# 理解构建器模式

假设我们想要创建一个由多个部分组成且需要逐步构建的对象。除非所有部分都完全创建，否则对象是不完整的。这就是**构建器设计模式**能帮到我们的地方。构建器模式将复杂对象的构建与其表示分离。通过将构建与表示分离，相同的构建过程可以用来创建多个不同的表示([j.mp/builderpat](http://j.mp/builderpat))。

一个实际例子可以帮助我们理解构建器模式的目的。假设我们想要创建一个由`<html>`开始并以`</html>`结束的 HTML 文档；在 HTML 部分包含`<head>`和`</head>`元素；在头部部分包含`<title>`和`</title>`元素；等等。但页面的表示可能不同。每个页面都有自己的标题、自己的标题和不同的`<body>`内容。此外，页面通常分步骤构建：一个函数添加标题，另一个添加主要标题，另一个添加页脚，等等。只有当整个页面结构完成后，才能使用最终的渲染函数将其展示给客户端。我们可以更进一步，扩展 HTML 生成器，使其能够生成完全不同的 HTML 页面。一个页面可能包含表格，另一个页面可能包含图片画廊，另一个页面可能包含联系表单，等等。

使用构建器模式可以解决 HTML 页面生成问题。在这个模式中，有两个主要参与者，如下所述：

+   **构建者**：负责创建复杂对象各个部分的组件。在这个例子中，这些部分是页面的标题、标题、正文和页脚。

+   `builder` 实例。它调用构建器的函数来设置标题、标题等，使用不同的 `builder` 实例允许我们创建不同的 HTML 页面，而不需要触及导演的任何代码。

首先，让我们在下一节中讨论一些真实生活中应用此模式的情况。

# 真实世界的例子

在我们的日常生活中，*构建设计模式*在快餐店中被使用。制作汉堡和包装（盒子和平装袋）的相同程序总是被使用，即使有许多不同种类的汉堡（经典、芝士汉堡等）和不同的包装（小号盒子、中号盒子等等）。经典汉堡和芝士汉堡之间的区别在于表示，而不是构建程序。在这种情况下，**导演**是收银员，他向工作人员下达需要准备的指示，而**构建者**是负责特定订单的工作人员。

我们还可以找到软件示例，如下所示：

+   在本章开头提到的 HTML 示例实际上被 `django-widgy`（[`wid.gy/`](https://wid.gy/)）使用，这是一个第三方 Django 树编辑器，可以用作 `django-widgy` 编辑器，其中包含一个页面构建器，可以用来创建具有不同布局的 HTML 页面。

+   `django-query-builder` 库（[`github.com/ambitioninc/django-query-builder`](https://github.com/ambitioninc/django-query-builder)）是另一个依赖构建模式的第三方 Django 库。这个库可以用来动态构建 **结构化查询语言**（**SQL**）查询，允许你控制查询的所有方面，并创建从简单到非常复杂的各种查询。

在下一节中，我们将看到这个设计模式是如何实际工作的。

# 用例

当我们知道一个对象必须通过多个步骤创建，并且需要相同构建的不同表示时，我们使用构建模式。这些需求存在于许多应用程序中，例如页面生成器（例如，本章中提到的 HTML 页面生成器）、文档转换器和**用户界面**（**UI**）表单创建器（[j.mp/pipbuild](http://j.mp/pipbuild)）。

一些在线资源提到，建造者模式也可以作为解决望远镜构造器问题（telescopic constructor problem）的解决方案。望远镜构造器问题发生在我们被迫为支持不同的对象创建方式而创建新的构造器时。问题在于我们最终会有很多构造器，以及难以管理的长参数列表。Stack Overflow 网站上列出了一个望远镜构造器的例子 ([j.mp/sobuilder](http://j.mp/sobuilder))。幸运的是，这个问题在 Python 中不存在，因为它可以通过至少两种方式解决，如下所述：

+   使用在类构造函数中定义不同行为的命名参数 ([j.mp/sobuipython](http://j.mp/sobuipython))

+   使用参数列表解包，这在精神上类似于命名参数 ([j.mp/arglistpy](http://j.mp/arglistpy))

这些特定于 Python 的特性帮助我们轻松控制其代码的行为，从而避免我们之前描述的问题。

在这一点上，建造者模式和工厂模式之间的区别可能不是很清楚。主要区别在于，工厂模式在单步中创建一个对象，而建造者模式在多个步骤中创建一个对象，通常通过使用一个导演（director）来实现。一些针对建造者模式的特定实现，如 Java 的`StringBuilder`，绕过了导演的使用，但这只是例外。

另一个区别是，虽然工厂模式立即返回创建的对象，但在建造者模式中，客户端代码明确要求导演在需要时返回最终对象 ([j.mp/builderpat](http://j.mp/builderpat))。

新的计算机类比可能有助于您区分建造者模式（builder pattern）和工厂模式（factory pattern）。假设您想购买一台新电脑。如果您决定购买一款特定的、预配置的电脑型号——例如，最新的苹果 1.4 `apple_factory.py`）：

```py
MINI14 = '1.4GHz Mac mini'
class AppleFactory:
    class MacMini14:
        def __init__(self):
            self.memory = 4 # in gigabytes
            self.hdd = 500 # in gigabytes
            self.gpu = 'Intel HD Graphics 5000'
        def __str__(self):
            info = (f'Model: {MINI14}',
                    f'Memory: {self.memory}GB',
                    f'Hard Disk: {self.hdd}GB',
                    f'Graphics Card: {self.gpu}')
            return '\n'.join(info)
    def build_computer(self, model):
        if model == MINI14:
            return self.MacMini14()
        else:
            msg = f"I don't know how to build {model}"
            print(msg)
```

现在，我们添加程序的主要部分——使用`AppleFactory`类的代码片段。代码如下所示：

```py
if __name__ == '__main__':
    afac = AppleFactory()
    mac_mini = afac.build_computer(MINI14)
    print(mac_mini)
```

注意

注意嵌套的`MacMini14`类。这是一种禁止直接实例化类的巧妙方法。

另一个选择是购买一台定制 PC。在这种情况下，您使用建造者模式。您是导演，向制造商（`builder`）下达关于理想电脑规格的命令。代码上，看起来是这样的（`computer_builder.py`）：

1.  我们定义一个`Computer`类，如下所示：

    ```py
    class Computer:
        def __init__(self, serial_number):
            self.serial = serial_number
            self.memory = None # in gigabytes
            self.hdd = None # in gigabytes
            self.gpu = None
        def __str__(self):
            info = (f'Memory: {self.memory}GB',
                    f'Hard Disk: {self.hdd}GB',
                    f'Graphics Card: {self.gpu}')
            return '\n'.join(info)
    ```

1.  我们定义一个`ComputerBuilder`类，如下所示：

    ```py
    class ComputerBuilder:
        def __init__(self):
            self.computer = Computer('AG23385193')
        def configure_memory(self, amount):
            self.computer.memory = amount
        def configure_hdd(self, amount):
            self.computer.hdd = amount
        def configure_gpu(self, gpu_model):
            self.computer.gpu = gpu_model
    ```

1.  我们定义一个`HardwareEngineer`类，如下所示：

    ```py
    class HardwareEngineer:
        def __init__(self):
            self.builder = None
        def construct_computer(self, memory, hdd, gpu):
            self.builder = ComputerBuilder()
            steps = (self.builder.configure_memory(memory),
                     self.builder.configure_hdd(hdd),
                     self.builder.configure_gpu(gpu))
            [step for step in steps]
        @property
        def computer(self):
            return self.builder.computer
    ```

1.  我们以`main()`函数结束我们的代码，然后通过以下代码片段中的技巧在从命令行调用文件时调用它：

    ```py
    def main():
        engineer = HardwareEngineer()
        engineer.construct_computer(hdd=500, 
                                    memory=8, 
                                    gpu='GeForce GTX 650 Ti')
        computer = engineer.computer
        print(computer)
    if __name__ == '__main__':
        main()
    ```

基本的变化是引入了一个建造者（`ComputerBuilder`）、一个导演（`HardwareEngineer`）以及逐步构建一台电脑的过程，现在它支持不同的配置（注意`memory`、`hdd`和`gpu`是参数，而不是预先配置的）。如果我们想支持平板电脑的构建，我们应该如何操作？将此作为练习来实现。

你可能还想将电脑的`serial_number`值改为不同的值，因为现在，这意味着所有电脑都将有相同的序列号（这是不切实际的）。

# 实现订购应用程序

让我们看看如何使用建造者设计模式来制作一个披萨订购应用程序。披萨的例子特别有趣，因为披萨的制备需要遵循特定的步骤顺序。要加酱料，你首先需要准备面团。要加配料，你首先需要加酱料。而且，除非酱料和配料都放在面团上，否则你不能开始烤披萨。此外，每种披萨通常需要不同的烘烤时间，这取决于面团厚度和使用的配料。

我们首先导入所需的模块，声明一些`Enum`参数（[j.mp/pytenum](http://j.mp/pytenum)）以及应用程序中多次使用的常量。`STEP_DELAY`常量用于在准备披萨的不同步骤之间添加时间延迟（如准备面团、加酱料等），如下所示：

```py
from enum import Enum
import time
PizzaProgress = Enum('PizzaProgress', 'queued preparation \
  baking ready')
PizzaDough = Enum('PizzaDough', 'thin thick')
PizzaSauce = Enum('PizzaSauce', 'tomato creme_fraiche')
PizzaTopping = Enum('PizzaTopping', 'mozzarella \
  double_mozzarella bacon ham mushrooms red_onion oregano')
STEP_DELAY = 3 # in seconds for the sake of the example
```

我们的产品最终是披萨，它由`Pizza`类来描述。在使用建造者模式时，最终产品不需要承担太多责任，因为它不应该被直接实例化。建造者创建最终产品的实例并确保它被正确准备。这就是为什么`Pizza`类如此简洁。它基本上将所有数据初始化为合理的默认值。一个例外是`prepare_dough()`方法。

`prepare_dough()`方法定义在`Pizza`类中而不是建造者中，有两个原因——首先，为了阐明最终产品通常是简洁的，这并不意味着你永远不应该给它分配任何责任；其次，为了通过组合来促进代码重用。

因此，我们定义我们的`Pizza`类如下：

```py
class Pizza:
    def __init__(self, name):
        self.name = name
        self.dough = None
        self.sauce = None
        self.topping = []
    def __str__(self):
        return self.name
    def prepare_dough(self, dough):
        self.dough = dough
        print(f'preparing the {self.dough.name} dough of your \
          {self}...')
        time.sleep(STEP_DELAY)
        print(f'done with the {self.dough.name} dough')
```

有两个建造者：一个用于创建玛格丽塔披萨（`MargaritaBuilder`）和另一个用于创建奶油培根披萨（`CreamyBaconBuilder`）。每个建造者创建一个`Pizza`实例，并包含遵循披萨制作程序的各个方法：`prepare_dough()`、`add_sauce()`、`add_topping()`和`bake()`。更准确地说，`prepare_dough()`只是对`Pizza`类的`prepare_dough()`方法的包装。

注意每个建造者如何处理所有与披萨相关的细节。例如，玛格丽塔披萨的配料是双份马苏里拉奶酪和牛至，而奶油培根披萨的配料是马苏里拉奶酪、培根、火腿、蘑菇、红洋葱和牛至。

我们代码的这一部分布局如下：

1.  我们定义了一个`MargaritaBuilder`类，如下所示：

    ```py
    class MargaritaBuilder:
        def __init__(self):
            self.pizza = Pizza('margarita')
            self.progress = PizzaProgress.queued
            self.baking_time = 5 # in seconds for the sake of 
            the example
        def prepare_dough(self):
            self.progress = PizzaProgress.preparation
            self.pizza.prepare_dough(PizzaDough.thin)
        def add_sauce(self):
            print('adding the tomato sauce to your \
              margarita...')
            self.pizza.sauce = PizzaSauce.tomato
            time.sleep(STEP_DELAY)
            print('done with the tomato sauce')
        def add_topping(self):
            topping_desc = 'double mozzarella, oregano'
            topping_items = (PizzaTopping.double_mozzarella,
            PizzaTopping.oregano)
            print(f'adding the topping ({topping_desc}) to \
              your margarita')
            self.pizza.topping.append([t for t in \
              topping_items])
            time.sleep(STEP_DELAY)
            print(f'done with the topping ({topping_desc})')
        def bake(self):
            self.progress = PizzaProgress.baking
            print(f'baking your margarita for \
              {self.baking_time} seconds')
            time.sleep(self.baking_time)
            self.progress = PizzaProgress.ready
            print('your margarita is ready')
    ```

1.  我们定义了一个`CreamyBaconBuilder`类，如下所示：

    ```py
    class CreamyBaconBuilder:
        def __init__(self):
            self.pizza = Pizza('creamy bacon')
            self.progress = PizzaProgress.queued
            self.baking_time = 7 # in seconds for the sake of 
            the example
        def prepare_dough(self):
            self.progress = PizzaProgress.preparation
            self.pizza.prepare_dough(PizzaDough.thick)
        def add_sauce(self):
            print('adding the crème fraîche sauce to your \
              creamy bacon')
            self.pizza.sauce = PizzaSauce.creme_fraiche
            time.sleep(STEP_DELAY)
            print('done with the crème fraîche sauce')
        def add_topping(self):
            topping_desc = 'mozzarella, bacon, ham, \
              mushrooms, red onion, oregano'
            topping_items =  (PizzaTopping.mozzarella,
                              PizzaTopping.bacon,
                              PizzaTopping.ham,
                              PizzaTopping.mushrooms,
                              PizzaTopping.red_onion, 
                              PizzaTopping.oregano)
            print(f'adding the topping ({topping_desc}) to \
              your creamy bacon')
            self.pizza.topping.append([t for t in \
              topping_items])
            time.sleep(STEP_DELAY)
            print(f'done with the topping ({topping_desc})')
        def bake(self):
            self.progress = PizzaProgress.baking
            print(f'baking your creamy bacon for \
              {self.baking_time} seconds')
            time.sleep(self.baking_time)
            self.progress = PizzaProgress.ready
            print('your creamy bacon is ready')
    ```

在这个例子中，导演是服务员。`Waiter`类的核心是`construct_pizza()`方法，它接受一个`builder`作为参数，并按正确的顺序执行所有披萨准备步骤。选择合适的构建器，甚至可以在运行时完成，这使我们能够创建不同的披萨风格，而无需修改导演（`Waiter`）的任何代码。`Waiter`类还包含一个`pizza()`方法，该方法将最终产品（准备好的披萨）作为变量返回给调用者，如下所示：

```py
class Waiter:
    def __init__(self):
        self.builder = None
    def construct_pizza(self, builder):
        self.builder = builder
        steps = (builder.prepare_dough, 
                 builder.add_sauce, 
                 builder.add_topping, 
                 builder.bake)
        [step() for step in steps]
    @property
    def pizza(self):
        return self.builder.pizza
```

`validate_style()`函数与在*第十六章*“工厂模式”中描述的`validate_age()`函数类似。它用于确保用户给出有效的输入，在这种情况下是一个映射到披萨构建器的字符。`m`字符使用`MargaritaBuilder`类，而`c`字符使用`CreamyBaconBuilder`类。这些映射在构建器参数中。返回一个元组，第一个元素如果输入有效则设置为`True`，如果无效则设置为`False`，如下所示：

```py
def validate_style(builders):
    try:
        input_msg = 'What pizza would you like, [m]argarita or \
        [c]reamy bacon? '
        pizza_style = input(input_msg)
        builder = builders[pizza_style]()
        valid_input = True
    except KeyError:
        error_msg = 'Sorry, only margarita (key m) and creamy \
        bacon (key c) are available'
        print(error_msg)
        return (False, None)
    return (True, builder)
```

最后的部分是`main()`函数。`main()`函数包含实例化披萨构建器的代码。然后，`Waiter`导演使用披萨构建器来准备披萨。制作的披萨可以在任何后续时间点交付给客户。代码在以下片段中展示：

```py
def main():
    builders = dict(m=MargaritaBuilder, c=CreamyBaconBuilder)
    valid_input = False
    while not valid_input:
        valid_input, builder = validate_style(builders)
    print()
    waiter = Waiter()
    waiter.construct_pizza(builder)
    pizza = waiter.pizza
    print()
    print(f'Enjoy your {pizza}!')
```

这里是实现总结（请参阅`builder.py`文件中的完整代码）：

1.  我们开始于需要的一些导入，对于标准的`Enum`类和`time`模块。

1.  我们声明了一些常量的变量：`PizzaProgress`、`PizzaDough`、`PizzaSauce`、`PizzaTopping`和`STEP_DELAY`。

1.  我们定义了我们的`Pizza`类。

1.  我们为两个构建器定义了类，`MargaritaBuilder`和`CreamyBaconBuilder`。

1.  我们定义了我们的`Waiter`类。

1.  我们添加了`validate_style()`函数来改进关于异常处理的事情。

1.  最后，我们有`main()`函数，随后是程序运行时调用它的片段。在`main`函数中，以下操作会发生：

    +   我们通过`validate_style()`函数的验证后，使根据用户的输入选择披萨构建器成为可能。

    +   服务员使用披萨构建器来准备披萨。

    +   然后送出制作的披萨。

调用`python builder.py`命令执行此示例程序时，产生了以下输出：

```py
What pizza would you like, [m]argarita or [c]reamy bacon? r
Sorry, only margarita (key m) and creamy bacon (key c) are 
available
What pizza would you like, [m]argarita or [c]reamy bacon? m
preparing the thin dough of your margarita...
done with the thin dough
adding the tomato sauce to your margarita...
done with the tomato sauce
adding the topping (double mozzarella, oregano) to your 
margarita
done with the topping (double mozzarella, oregano)
baking your margarita for 5 seconds
your margarita is ready
Enjoy your margarita!
```

但是……只支持两种披萨类型是件遗憾的事。想要一个夏威夷披萨构建器吗？考虑在考虑了优势和劣势之后使用继承。检查一下典型夏威夷披萨的配料，并决定你需要扩展哪个类：`MargaritaBuilder`还是`CreamyBaconBuilder`？也许两者都需要([j.mp/pymulti](http://j.mp/pymulti))？

在他的书《有效 Java（第二版）》中，Joshua Bloch 描述了一种有趣的构建器模式变体，其中构建器方法的调用是链式的。这是通过将构建器本身定义为内部类，并从其上的每个 setter-like 方法返回自身来实现的。`build()`方法返回最终对象。这种模式被称为**流畅构建器**。以下是一个 Python 实现，这是由本书的一位审稿人友好地提供的：

```py
class Pizza: 
    def __init__(self, builder): 
        self.garlic = builder.garlic 
        self.extra_cheese  = builder.extra_cheese 

    def __str__(self): 
        garlic = 'yes' if self.garlic else 'no' 
        cheese = 'yes' if self.extra_cheese else 'no' 
        info = (f'Garlic: {garlic}', f'Extra cheese: {cheese}') 
        return '\n'.join(info) 

    class PizzaBuilder: 
        def __init__(self): 
            self.extra_cheese = False 
            self.garlic = False 

        def add_garlic(self): 
            self.garlic = True 
            return self 

        def add_extra_cheese(self): 
            self.extra_cheese = True 
            return self 

        def build(self): 
            return Pizza(self) 

if __name__ == '__main__': 
    pizza = Pizza.PizzaBuilder().add_garlic().add_extra_ \
      cheese().build() 
    print(pizza)
```

使用这种流畅的构建器模式，我们可以看到，通过在一行代码中链式调用`add_garlic()`、`add_extra_cheese()`和`build()`方法，可以快速构建最终的`Pizza`对象，这在许多情况下都很有用。

# 摘要

在本章中，我们看到了如何使用构建器设计模式。当使用工厂模式（无论是工厂方法还是抽象工厂）不是一个好选择时，我们使用构建器模式来创建对象。当我们想要创建一个复杂对象，需要对象的不同表示，或者我们想要在某个时间点创建对象但在稍后访问它时，构建器模式通常比工厂模式是一个更好的选择。

我们看到了构建器模式如何在快餐店准备餐点时使用，以及两个第三方 Django 包`django-widgy`和`django-query-builder`如何分别用于生成 HTML 页面和动态 SQL 查询。我们关注了构建器模式和工厂模式之间的区别，并提供了预配置（工厂）和客户（构建器）计算机订单类比来澄清它们。我们还探讨了如何创建一个具有准备依赖关系的比萨订购应用程序。在这些示例中，我们清楚地看到了构建器模式的好处和灵活性，这将帮助你更好地处理未来需要设计模式的应用程序。

在下一章中，你将学习其他有用的创建型模式。

# 问题

1.  构建器模式有哪些高级应用？

1.  有哪些常见的计算机应用程序需要或从构建器模式中受益？

1.  构建器模式是如何创建对象的，以及这个过程与工厂模式有何不同？
