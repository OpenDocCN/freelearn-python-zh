## 第七章：7

类和对象的基础

计算的目的在于处理数据。我们通常将处理和数据封装到单个定义中。我们可以将具有共同属性集的对象组织成类，以定义它们的内部状态和共同行为。类的每个实例都是一个具有独特内部状态和行为的独立对象。

这种状态和行为的概念特别适用于游戏的工作方式。当构建类似交互式游戏的东西时，用户的操作会更新游戏状态。玩家的每个可能动作都是一个改变游戏状态的方法。在许多游戏中，这会导致大量的动画来展示状态之间的转换。在单人街机风格的游戏中，敌人或对手通常会是独立的对象，每个对象都有一个基于其他敌人动作和玩家动作而变化的内部状态。

另一方面，如果我们考虑一副牌或掷骰子游戏，可能的状态可能非常少。像 Zonk 这样的游戏涉及玩家掷（并重新掷）骰子，只要他们的分数提高。如果随后的掷骰子未能改善他们的骰子组合，他们的回合就结束了。手牌的状态是由构成得分子集的骰子池，通常推到桌子的一个边上。在一个六骰子的游戏中，将有从一到六个得分骰子作为不同的状态。此外，当所有骰子都是得分骰子时，玩家可以通过重新掷所有骰子来再次开始掷骰子的过程。这导致了一个额外的“超常”状态，玩家也必须记住。

面向对象设计的目的是使用对象的属性来定义当前状态。每个对象都被定义为类似对象的类的一个实例。我们用 Python 编写类定义，并使用这些定义来创建对象。类中定义的方法会在对象上引起状态变化。

在本章中，我们将探讨以下食谱：

+   使用类来封装数据和处理

+   类定义的基本类型提示

+   设计具有大量处理的类

+   使用 typing.NamedTuple 来表示不可变对象

+   使用数据类来表示可变对象

+   使用冻结的数据类来表示不可变对象

+   使用 __slots__ 优化小对象

+   使用更复杂的集合

+   扩展内置集合 – 一个可以进行统计的列表

+   使用属性来表示延迟属性

+   创建上下文和上下文管理器

+   使用多个资源管理多个上下文

面向对象设计的主旨相当广泛。在本章中，我们将介绍一些基本概念。我们将从一些基础概念开始，例如类定义如何封装类的所有实例的状态和处理细节。

# 7.1 使用类封装数据和处理

类设计受到 SOLID 设计原则的影响。单一责任和接口隔离原则提供了有用的建议。综合考虑，这些原则建议我们，一个类应该有方法，这些方法专注于单一、明确的责任。

考虑类的一种另一种方式是作为一个紧密相关的函数组，这些函数使用共同的数据。我们称这些为处理数据的函数。类定义应该包含处理对象数据的最小方法集合。

我们希望基于狭窄的责任分配创建类定义。我们如何有效地定义责任？设计一个类的好方法是什么？

## 7.1.1 准备工作

让我们看看一个简单的、有状态的对象——一对骰子。这个背景是一个模拟简单游戏如 Craps 的应用程序。

软件对象可以看作是类似事物——名词。类的行为可以看作是动词。这种与名词和动词的认同给我们提供了如何有效地设计类以有效工作的线索。

这引导我们进入几个准备步骤。我们将通过使用一对骰子进行游戏模拟来提供这些步骤的具体示例。我们按以下步骤进行：

1.  写下描述类实例所做事情的简单句子。我们可以称之为问题陈述。专注于单动词句子，只关注名词和动词是至关重要的。以下是一些例子：

    +   Craps 游戏有两个标准的骰子。

    +   每个骰子有六个面，点数从一到六。

    +   玩家掷骰子。虽然作者和编辑更喜欢主动语态版本，“玩家掷骰子”，但骰子通常被其他对象所作用，使得被动语态句子稍微更有用。

    +   骰子的总和改变了 Craps 游戏的状态。这些规则与骰子是分开的。

    +   如果两个骰子匹配，这个数字被描述为“硬掷”。如果两个骰子不匹配，掷骰子被描述为“易掷”。

1.  识别句子中的所有名词。在这个例子中，名词包括骰子、面、点数和玩家。名词识别不同类别的对象，可能是合作者，如玩家和游戏。名词也可能识别对象的属性，如面和点数。

1.  识别句子中的所有动词。动词通常成为所讨论的类的成员方法。在这个例子中，动词包括 roll 和 match。

这些信息有助于定义对象的状态和行为。拥有这些背景信息将帮助我们编写类定义。

## 7.1.2 如何做...

由于我们编写的模拟涉及骰子的随机投掷，我们将依赖于 from random import randint 提供有用的 randint() 函数。定义类的步骤如下：

1.  使用类声明开始编写类：

    ```py
     class Dice:
    ```

1.  在 __init__() 方法的主体中初始化对象的属性。我们将使用 faces 属性来模拟骰子的内部状态。需要一个 self 变量来确保我们引用的是类的给定实例的属性。我们将在每个属性上提供类型提示，以确保在整个类定义中正确使用：

    ```py
     def __init__(self) -> None: 

            self.faces: tuple[int, int] = (0, 0)
    ```

1.  根据描述中的动词定义对象的方法。当玩家掷骰子时，roll() 方法可以设置两个骰子面上的值。我们通过设置 self 对象的 faces 属性来实现这一点：

    ```py
     def roll(self) -> None: 

            self.faces = (randint(1,6), randint(1,6))
    ```

    此方法会改变对象的内部状态。我们选择不返回任何值。

1.  玩家掷骰子后，total() 方法有助于计算骰子的总和：

    ```py
     def total(self) -> int: 

            return sum(self.faces)
    ```

1.  可以提供额外的方法来回答有关骰子状态的问题。在这种情况下，当两个骰子都匹配时，总和是通过“困难的方式”得到的：

    ```py
     def hardway(self) -> bool: 

            return self.faces[0] == self.faces[1] 

        def easyway(self) -> bool: 

            return self.faces[0] != self.faces[1] 
    ```

## 7.1.3 它是如何工作的...

核心思想是使用普通的语法规则——名词、动词和形容词——作为识别类基本特征的一种方式。在我们的例子中，骰子是真实的事物。我们尽量避免使用抽象术语，如随机化器或事件生成器。描述真实事物的可触摸特征更容易，然后定义一个实现来匹配这些可触摸特征。

掷骰子的想法是一个我们可以通过方法定义来模拟的物理动作。这个掷骰子的动作会改变对象的状态。在极少数情况下——36 次中的 1 次——下一个状态会恰好与之前的状态相同。

下面是使用 Dice 类的一个示例：

1.  首先，我们将使用一个固定值来初始化随机数生成器，以便我们可以得到一个固定的结果序列：

    ```py
     >>> import random 

    >>> random.seed(1)
    ```

1.  我们将创建一个 Dice 对象，并将其分配给变量 d1。然后我们可以使用 roll() 方法设置其状态。然后我们将查看 total() 方法以查看掷出了什么。我们将通过查看 faces 属性来检查状态：

    ```py
     >>> d1 = Dice() 

    >>> d1.roll() 

    >>> d1.total() 

    7 

    >>> d1.faces 

    (2, 5)
    ```

## 7.1.4 更多...

捕获导致状态变化的必要内部状态和方法是良好类设计的第一步。我们可以使用缩写 SOLID 总结一些有用的设计原则：

+   单一职责原则：一个类应该有一个明确定义的责任。

+   开放/封闭原则：一个类应该对扩展开放——通常通过继承——但对修改封闭。我们应该设计我们的类，以便我们不需要调整代码来添加或更改功能。

+   李斯克夫替换原则：我们需要设计继承，使得子类可以替代父类使用。

+   接口隔离原则：在编写问题陈述时，我们希望确保协作类尽可能少地依赖。在许多情况下，这一原则将引导我们将大问题分解成许多小的类定义。

+   依赖倒置原则：一个类直接依赖于其他类并不理想。如果类依赖于抽象，并且用具体实现类替换抽象类，则更好。

目标是创建具有必要行为并遵循设计原则的类，以便它们可以被扩展和重用。

## 7.1.5 参考内容

+   请参阅使用属性实现懒属性的配方，我们将探讨选择积极属性和懒属性之间的选择。

+   在第八章中，我们将更深入地探讨类设计技术。

+   请参阅第十五章，了解如何为类编写适当的单元测试配方。

# 7.2 类定义的必要类型提示

类名也是一个类型提示，允许变量直接引用应该定义与变量相关联的对象的类。这种关系使工具如 mypy 能够推理我们的程序，以确保对象引用和方法引用似乎与代码中的类型提示相匹配。

除了类名之外，我们将在类定义中的三个常见位置使用类型提示：

+   在方法定义中，我们将使用类型提示来注释参数和返回类型。

+   在`__init__()`方法中，我们可能需要为定义对象状态的实例变量提供提示。

+   在类的整体属性中。这些不是常见的，这里的类型提示也很少。

## 7.2.1 准备工作

我们将检查一个具有各种类型提示的类。在这个例子中，我们的类将模拟一把骰子。我们将允许重新掷选定的骰子，使类的实例具有状态。

骰子集合可以通过第一次掷骰子来设置，其中所有骰子都被掷出。该类允许掷出骰子子集的后续掷骰。同时也会计算掷骰子的次数。

类型提示将反映骰子集合的性质、整数计数、浮点平均值以及整个手牌的字符串表示。这将展示一系列类型提示及其编写方式。

## 7.2.2 如何实现...

1.  此定义将涉及随机数以及集合和列表的类型提示。我们导入 random 模块：

    ```py
     import random 
    ```

1.  定义类。这创建了一个新类型：

    ```py
     class Dice:
    ```

1.  类级别的变量很少需要类型提示。它们几乎总是通过赋值语句创建的，这些语句使类型信息对人类或像 mypy 这样的工具来说很清晰。在这种情况下，我们希望我们的骰子类的所有实例共享一个共同的随机数生成器对象：

    ```py
     RNG = random.Random()
    ```

1.  __init__()方法创建了定义对象状态的实例变量。在这种情况下，我们将保存一些配置细节和一些内部状态。__init__()方法还有初始化参数。通常，我们会在这些参数上放置类型提示。其他内部状态变量可能需要类型提示来显示其他类方法将分配哪些类型的值。在这个例子中，faces 属性没有初始值；我们声明当它被设置时，它将是一个 List[int]对象：

    ```py
     def __init__(self, n: int, sides: int = 6) -> None: 

            self.n_dice = n 

            self.sides = sides 

            self.faces: list[int] 

            self.roll_number = 0
    ```

1.  计算新导出值的方法可以用它们的返回类型信息进行注解。这里有三个例子，用于返回字符串表示、计算总和以及计算骰子的平均值。这些函数的返回类型分别是 str、int 和 float，如下所示：

    ```py
     def __str__(self) -> str: 

            return ", ".join( 

                f"{i}: {f}" 

                for i, f in enumerate(self.faces) 

            ) 

        def total(self) -> int: 

            return sum(self.faces) 

        def average(self) -> float: 

            return sum(self.faces) / self.n_dice
    ```

1.  对于有参数的方法，我们在参数上以及返回类型上包含类型提示。在这种情况下，改变内部状态的方法也会返回值。两个方法的返回值都是骰子面的列表，描述为 list[int]。reroll()方法的参数是要重新掷的骰子集合。这表示为 set[int]，需要一组整数。Python 比这要灵活一些，我们将探讨一些替代方案：

    ```py
     def first_roll(self) -> list[int]: 

            self.roll_number = 0 

            self.faces = [ 

                self.RNG.randint(1, self.sides) 

                for _ in range(self.n_dice) 

            ] 

            return self.faces 

        def reroll(self, positions: set[int]) -> list[int]: 

            self.roll_number += 1 

            for p in positions: 

                self.faces[p] = self.RNG.randint(1, self.sides) 

            return self.faces
    ```

## 7.2.3 工作原理...

类型提示信息被程序如 mypy 使用，以确保在整个应用程序中正确使用类的实例。

如果我们尝试编写如下函数：

```py
 def example_mypy_failure() -> None: 

    d = Dice(2.5) 

    d.first_roll() 

    print(d)
```

使用浮点值作为 n 参数创建 Dice 类实例的尝试与类型提示发生冲突。Dice 类的 __init__()方法的提示称参数值应该是整数。mypy 程序报告如下：

```py
src/ch07/recipe_02_bad.py:9: error: Argument 1 to "Dice" has incompatible type "float"; expected "int"  [arg-type]
```

如果我们尝试执行应用程序，它将在另一个地方引发 TypeError 异常。错误将在评估 d.first_roll()方法时显现。异常在这里被引发，因为 __init__()方法的主体可以很好地处理任何类型的值。提示称期望特定的类型，但在运行时，可以提供任何对象。在执行期间不会检查提示。

类似地，当我们使用其他方法时，mypy 程序会检查我们的方法使用是否与类型提示定义的期望相匹配。这里有一个另一个例子：

```py
 r1: list[str] = d.first_roll()
```

这个赋值语句中，r1 变量的类型提示与 first_roll()方法返回值的类型提示不匹配。这种冲突是由 mypy 检测到的，并报告为“赋值中不兼容的类型”错误。

## 7.2.4 更多内容...

在这个例子中，有一个类型提示过于具体。用于重新掷骰子的 reroll()函数有一个 positions 参数。positions 参数在 for 语句中使用，这意味着对象必须是某种可迭代对象。

错误在于提供了一个类型提示 set[int]，这仅仅是许多可迭代对象中的一种。我们可以通过将类型提示从非常具体的 set[int]切换到更通用的 Iterable[int]来泛化这个定义。

放宽提示意味着任何集合、列表或元组对象都可以作为此参数的有效参数值。所需的唯一其他代码更改是导入 collections.abc 模块中的 Iterable。

for 语句有一个从可迭代集合获取迭代器对象、将值赋给变量和执行缩进体的特定协议。这个协议由 Iterable 类型提示定义。有许多这样的基于协议的类型，它们允许我们提供与 Python 固有的类型灵活性相匹配的类型提示。

## 7.2.5 参考信息

+   在第三章的函数参数和类型提示配方中，展示了多个类似的概念。

+   在第四章中，编写与列表相关的类型提示和编写与集合相关的类型提示配方解决了额外的详细类型提示。

+   在第五章中，编写与字典相关的类型提示配方也解决了类型提示。

# 7.3 设计具有大量处理的类

有时，一个对象将包含定义其内部状态的所有数据。然而，有些情况下，一个类不持有数据，而是设计用来对存储在单独容器中的数据进行处理进行整合。

这种设计的典型例子是统计算法，这些算法通常位于被分析的数据之外。数据可能在一个内置的列表或 Counter 对象中；处理定义在数据容器之外的一个类中。

## 7.3.1 准备工作

在对已经总结成组或箱的数据进行分析是很常见的。例如，我们可能有一个包含大量工业过程测量的巨大数据文件。

为了了解背景，请参阅 NIST 气溶胶粒子尺寸案例研究：[`www.itl.nist.gov/div898/handbook/pmc/section6/pmc62.htm`](https://www.itl.nist.gov/div898/handbook/pmc/section6/pmc62.htm)

与分析大量的原始数据相比，首先总结重要的变量，然后分析总结后的数据通常要快得多。总结数据可以保存在一个 Counter 对象中。数据看起来是这样的：

```py
 data = Counter({7: 80, 

         6: 67, 

         8: 62, 

         9: 50,
```

```py
... Details omitted ...
```

```py
 2: 3, 

         3: 2, 

         1: 1})
```

键（7、6、8、9 等等）是反映粒子大小的代码。实际尺寸从 109 到 119 不等。从 s 代码计算实际尺寸 c，公式为 c = ⌊2(s − 109)⌋。

（NIST 背景信息中没有提供单位。由于大量数据反映了电子芯片晶圆和制造过程，单位可能非常小。）

我们希望在不需要强制与原始大量数据集一起工作的前提下，计算这个 Counter 对象上的某些统计数据。一般来说，设计用于存储和处理数据的类有两种一般的设计策略：

+   扩展存储类定义，在这个例子中是 Counter，以添加统计处理。我们将在扩展内置集合 – 具有统计功能的列表食谱中详细说明。

+   在一个提供所需额外功能的类中包装 Counter 对象。当我们这样做时，我们还有两个选择：

    +   暴露底层 Counter 对象。我们将关注这一点。

    +   编写特殊方法以使包装器看起来也是一个集合，封装 Counter 对象。我们将在第八章中探讨这一点。

对于这个食谱，我们将专注于包装变体，其中我们定义一个统计计算类，该类公开一个 Counter 对象。我们有两种设计这种计算密集型处理的方法：

+   积极实现会在尽可能早的时候计算统计数据。这些值成为简单的属性。我们将关注这个选择。

+   懒惰方法不会在需要通过方法函数或属性获取值之前进行任何计算。我们将在使用属性进行懒惰属性食谱中探讨这一点。

两种设计的基本算法是相同的。唯一的问题是计算工作何时完成。

## 7.3.2 如何实现...

1.  从 collections 模块导入适当的类。计算使用 math.sqrt()。务必添加所需的导入 math：

    ```py
     from collections import Counter 

    import math
    ```

1.  使用描述性的名称定义类：

    ```py
     class CounterStatistics:
    ```

1.  编写 __init__()方法以包含数据所在的对象。在这种情况下，类型提示是 Counter[int]，因为 Counter 对象中使用的键将是整数：

    ```py
     def __init__(self, raw_counter: Counter[int]) ->  None: 

            self.raw_counter = raw_counter
    ```

1.  在 __init__()方法中初始化任何其他可能有用的局部变量。由于我们将积极计算值，最积极的时间是在对象创建时。我们将编写对一些尚未定义的函数的引用：

    ```py
     self.mean = self.compute_mean() 

            self.stddev = self.compute_stddev()
    ```

1.  定义所需的方法以计算各种值。以下是计算平均值的示例：

    ```py
     def compute_mean(self) -> float: 

            total, count = 0.0, 0 

            for value, frequency in self.raw_counter.items(): 

                total += value * frequency 

                count += frequency 

            return total / count
    ```

1.  这是我们如何计算标准差的方法：

    ```py
     def compute_stddev(self) -> float: 

            total, count = 0.0, 0 

            for value, frequency in self.raw_counter.items(): 

                total += frequency * (value - self.mean) ** 2 

                count += frequency 

            return math.sqrt(total / (count - 1))
    ```

注意，这个计算需要首先计算平均值，并创建 self.mean 实例变量。从无已知平均值到已知平均值再到已知标准差这种内部状态变化是一个潜在的复杂性，需要清晰的文档说明。

本例的原始数据位于[`www.itl.nist.gov/div898/handbook//datasets/NEGIZ4.DAT`](https://www.itl.nist.gov/div898/handbook//datasets/NEGIZ4.DAT)。由于数据前有 50 行标题文本，这个文件的结构显得很复杂。此外，文件不是常见的 CSV 格式。因此，处理汇总数据更容易。

本书代码库中包含一个名为 data/binned.csv 的文件，其中包含分箱的摘要数据。该数据有三个列：size_code、size 和 frequency。我们只对 size_code 和 frequency 感兴趣。

这是我们可以从该文件构建合适的 Counter 对象的方法：

```py
 >>> from pathlib import Path 

>>> import csv 

>>> from collections import Counter 

>>> data_path = Path.cwd() / "data" / "binned.csv" 

>>> with data_path.open() as data_file: 

...     reader = csv.DictReader(data_file) 

...     extract = { 

...         int(row[’size_code’]): int(row[’frequency’]) 

...         for row in reader 

...     } 

>>> data = Counter(extract)
```

我们使用字典推导来创建从 size_code 到该代码值频率的映射。然后将其提供给 Counter 类，从现有的摘要构建 Counter 对象 data。我们可以将此数据提供给 CounterStatistics 类，从分箱数据中获得有用的摘要统计信息。这看起来像以下示例：

```py
 >>> stats = CounterStatistics(data) 

>>> print(f"Mean: {stats.mean:.1f}") 

Mean: 10.4 

>>> print(f"Standard Deviation: {stats.stddev:.2f}") 

Standard Deviation: 4.17
```

我们提供了数据对象来创建 CounterStatistics 类的实例。创建这个实例也将立即计算摘要统计信息。不需要额外的显式方法评估。这些值作为 stats.mean 和 stats.stddev 属性可用。

计算统计信息的处理成本最初就支付了。正如我们下面将看到的，任何对底层数据的更改都可以关联一个非常小的增量成本。

## 7.3.3 它是如何工作的...

这个类封装了两个复杂算法，但不包括这些算法的任何数据。数据被单独保存在 Counter 对象中。我们编写了一个高级规范来处理，并将其放置在 __init__()方法中。然后我们编写了实现指定处理步骤的方法。我们可以设置所需的所有属性，这使得这是一种非常灵活的方法。

这种设计的优点是，属性值可以重复使用。计算平均值和标准差的成本只支付一次；每次使用属性值时，不需要进一步处理。

这种设计的缺点是，对底层 Counter 对象状态的任何更改都将使 CounterStatistics 对象的状态过时且不正确。例如，如果我们添加了数百个更多数据值，平均值和标准差就需要重新计算。当底层 Counter 对象不会改变时，急切计算值的设计是合适的。

## 7.3.4 更多内容...

如果我们需要对有状态、可变对象进行计算，我们有几种选择：

+   封装 Counter 对象并通过 CounterStatistics 类进行更改。这需要小心地暴露数据收集足够多的方法。我们将把这种设计推迟到第八章（ch012.xhtml#x1-4520008）。

+   使用延迟计算。参见本章中的使用属性进行延迟属性配方。

+   添加一个方法来实现计算平均值和标准差，这样在更改底层 Counter 对象后可以重新计算这些值。这导致重构 __init__()方法以使用这种新的计算方法。我们将把这个留作读者的练习。

+   编写文档说明每次底层 Counter 对象发生变化时创建新的 CounterStatistics 实例的要求。这不需要代码，只需明确说明对象状态上的约束即可。

## 7.3.5 参见

+   在扩展内置集合 – 统计列表的菜谱中，我们将探讨一种不同的设计方法，其中这些新的汇总函数被用来扩展类定义。

+   我们将在使用属性创建懒属性的菜谱中探讨不同的方法。这个替代菜谱将使用属性按需计算属性。

+   在第八章中，也探讨了 wrap=extend 设计选择。

# 7.4 使用 typing.NamedTuple 创建不可变对象

在某些情况下，一个对象是一个相对复杂数据的容器，但实际上并没有对数据进行很多处理。实际上，在许多情况下，我们将定义一个不需要任何独特方法函数的类。这些类是相对被动的数据项容器，没有太多的处理。

在许多情况下，Python 的内置容器类 – 列表、集合或字典 – 可以覆盖你的用例。小问题是，访问字典或列表中项的语法并不像访问对象属性那样优雅。

我们如何创建一个类，使我们能够使用 object.attribute 语法而不是更复杂的 object[‘attribute’] 语法？

## 7.4.1 准备工作

任何类型的类设计都有两种情况：

+   它是无状态的（或不可变的）吗？它是否包含永远不会改变的值的属性？这是一个 NamedTuple 的好例子。

+   它是有状态的（或可变的）吗？是否会有一个或多个属性的状态变化？这是 Python 类定义的默认情况。一个普通类是有状态的。我们可以使用使用 dataclasses 创建可变对象的菜谱来简化创建有状态对象的过程。

我们将定义一个类来描述具有点数和花色的简单扑克牌。由于牌的点数和花色不会改变，我们将为这个创建一个小型的无状态类。typing.NamedTuple 类是这类类定义的便捷基类。

## 7.4.2 如何做...

1.  我们将定义无状态对象为 typing.NamedTuple 的子类：

    ```py
     from typing import NamedTuple 
    ```

1.  将类名称定义为 NamedTuple 的扩展。包括具有各自类型提示的属性：

    ```py
     class Card(NamedTuple): 

        rank: int 

        suit: str
    ```

这是我们如何使用这个类定义来创建 Card 对象的方法：

```py
 >>> eight_hearts = Card(rank=8, suit=’\N{White Heart Suit}’) 

>>> eight_hearts 

Card(rank=8, suit=’’) 

>>> eight_hearts.rank 

8 

>> eight_hearts.suit 

’’ 

>>> eight_hearts[0]
```

我们创建了一个名为 Card 的新类，它有两个属性名称：rank 和 suit。在定义了类之后，我们可以创建类的实例。我们构建了一个单张 Card 对象，eight_hearts，其点数为八，花色为 ♡。

我们可以用其名称或其元组内的位置来引用这个对象的属性。当我们使用 eight_hearts.rank 或 eight_hearts[0]时，我们会看到 rank 属性的值，因为该属性在属性名称序列中定义在第一位。

这种类型的对象是不可变的。以下是一个尝试更改实例属性的示例：

```py
 >>> eight_hearts.suit = ’\N{Black Spade Suit}’ 

Traceback (most recent call last): 

... 

AttributeError: can’t set attribute
```

我们尝试更改 eight_hearts 对象的 suit 属性。这引发了一个 AttributeError 异常，表明命名元组的实例是不可变的。

元组可以包含任何类型的对象。

当元组包含可变项，如列表、集合或字典时，这些对象保持可变。

只有顶层包含的元组是不可变的。元组内的列表、集合或字典是可变的。

## 7.4.3 它是如何工作的...

typing.NamedTuple 类让我们定义一个具有明确定义属性列表的新子类。自动创建了一些方法，以提供最小级别的 Python 行为。我们可以看到一个实例将显示一个可读的文本表示，显示各种属性的值。

对于命名元组子类，其行为基于内置元组实例的工作方式。属性的顺序定义了元组之间的比较。例如，我们的 Card 定义首先列出 rank 属性。这意味着我们可以很容易地按等级排序牌。对于等级相同的两张牌，花色将按顺序排序。因为命名元组也是元组，所以它很好地作为集合的成员或字典的键。

在这个例子中，rank 和 suit 这两个属性作为类定义的一部分命名，但作为实例变量实现。为我们创建了一个元组的 __new__()方法的变体。该方法有两个参数与实例变量名称匹配。自动创建的方法将在对象创建时将参数值分配给实例变量。

## 7.4.4 更多...

我们可以向这个类定义添加方法。例如，如果每张牌都有一个点数，我们可能希望扩展类，使其看起来像以下示例：

```py
 class CardPoints(NamedTuple): 

    rank: int 

    suit: str 

    def points(self) -> int: 

        if 1 <= self.rank < 10: 

            return self.rank 

        else: 

            return 10
```

我们编写了一个 CardsPoints 类，它有一个 points()方法，该方法返回分配给每个等级的点数。这种点规则适用于像克里比奇这样的游戏，而不适用于像黑杰克这样的游戏。

因为这是一个元组，所以方法不能添加新的属性或更改属性。在某些情况下，我们通过其他元组构建复杂的元组。

## 7.4.5 参见

+   在设计大量处理的类的配方中，我们查看了一个完全处理且几乎没有数据的类。它作为这个类的完全对立面。

# 7.5 使用 dataclasses 处理可变对象

我们已经记录了 Python 中的两种一般类型的对象：

+   不可变：在设计过程中，我们会询问是否有属性具有永远不会改变的值。如果答案是肯定的，请参阅使用 typing.NamedTuple 为不可变对象菜谱，它提供了一种为不可变对象构建类定义的方法。

+   可变：一个或多个属性会有状态变化吗？在这种情况下，我们可以从头开始构建一个类，或者我们可以利用 @dataclass 装饰器从一些属性和类型提示中创建一个类定义。这个案例是这个菜谱的重点。

我们如何利用 dataclasses 库来帮助设计可变对象？

## 7.5.1 准备工作

我们将仔细研究一个具有内部状态的可变对象，以表示一副牌。虽然单个卡片是不可变的，但它们可以被插入到一副牌中并从一副牌中移除。在像克里比（Cribbage）这样的游戏中，手牌会有许多状态变化。最初，六张牌被分给两位玩家。玩家将各自放下一对牌来创建克里比。然后，剩下的四张牌交替出牌，以创造得分机会。然后，手牌在隔离状态下计数，得分机会的混合略有所不同。庄家从计数克里比中的牌中获得额外的手牌得分。（是的，最初是不公平的，但发牌轮流进行，所以最终是公平的。）

我们将研究一个简单的集合来存放卡片，并丢弃形成克里比的两张卡片。

## 7.5.2 如何操作...

1.  为了定义数据类，我们将导入 @dataclass 装饰器：

    ```py
     from dataclasses import dataclass
    ```

1.  使用 @dataclass 装饰器定义新的类：

    ```py
     @dataclass 

    class CribbageHand:
    ```

1.  使用适当的类型提示定义各种属性。在这个例子中，我们期望玩家拥有一组由 list[CardPoints] 表示的卡片集合。因为每张卡片都是唯一的，我们也可以使用 set[CardPoints] 类型提示：

    ```py
     cards: list[CardPoints]
    ```

1.  定义任何会改变对象状态的函数：

    ```py
     def to_crib(self, card1: CardPoints, card2: CardPoints) -> None: 

            self.cards.remove(card1)
    ```

这是完整的类定义，正确缩进：

```py
 @dataclass 

class CribbageHand: 

    cards: list[CardPoints] 

    def to_crib(self, card1: CardPoints, card2: CardPoints) -> None: 

        self.cards.remove(card1) 

        self.cards.remove(card2)
```

这个定义提供了一个单一的实例变量 self.cards，它可以被任何编写的函数使用。因为我们提供了类型提示，所以 mypy 程序可以检查类以确保它被正确使用。

这是创建这个 CribbageHand 类实例时的样子：

```py
 >>> cards = [ 

... CardPoints(rank=3, suit=’\N{WHITE DIAMOND SUIT}’), 

... CardPoints(rank=6, suit=’\N{BLACK SPADE SUIT}’), 

... CardPoints(rank=7, suit=’\N{WHITE DIAMOND SUIT}’), 

... CardPoints(rank=1, suit=’\N{BLACK SPADE SUIT}’), 

... CardPoints(rank=6, suit=’\N{WHITE DIAMOND SUIT}’), 

... CardPoints(rank=10, suit=’\N{WHITE HEART SUIT}’)] 

>>> ch1 = CribbageHand(cards) 

>>> from pprint import pprint 

>>> pprint(ch1) 

CribbageHand(cards=[CardPoints(rank=3, suit=’’), 

                    CardPoints(rank=6, suit=’’), 

                    CardPoints(rank=7, suit=’’), 

                    CardPoints(rank=1, suit=’’), 

                    CardPoints(rank=6, suit=’’), 

                    CardPoints(rank=10, suit=’’)]) 

>>> [c.points() for c in ch1.cards] 

[3, 6, 7, 1, 6, 10] 
```

在以下示例中，玩家决定（可能不明智）将 3♢ 和 A♠ 卡片放进行克里比：

```py
 >>> ch1.to_crib( 

...     CardPoints(rank=3, suit=’\N{WHITE DIAMOND SUIT}’), 

...     CardPoints(rank=1, suit=’\N{BLACK SPADE SUIT}’)) 

>>> pprint(ch1) 

CribbageHand(cards=[CardPoints(rank=6, suit=’’), 

                    CardPoints(rank=7, suit=’’), 

                    CardPoints(rank=6, suit=’’), 

                    CardPoints(rank=10, suit=’’)]) 

>>> [c.points() for c in ch1.cards] 

[6, 7, 6, 10]
```

在 to_crib() 方法从手中移除两张卡片后，剩余的四张卡片被显示出来。然后创建了一个新的列表推导式，包含剩余四张卡片的点数。

## 7.5.3 它是如何工作的...

@dataclass 装饰器帮助我们定义一个具有几个有用方法以及从命名变量及其类型提示中抽取的属性列表的类。我们可以看到，一个实例显示了一个可读的文本表示，显示了各种属性的值。

属性作为类定义的一部分命名，但实际上作为实例变量实现。在这个例子中，只有一个属性，cards。为我们创建了一个非常复杂的 __init__()方法。在这个例子中，它将有一个与每个实例变量名称匹配的参数，并将参数值分配给匹配的实例变量。

@dataclass 装饰器有几个选项可以帮助我们选择我们想要的类特性。以下是我们可以选择的选项和默认设置：

+   init=True：默认情况下，将创建一个 __init__()方法，其参数与实例变量相匹配。

+   repr=True：默认情况下，将创建一个 __repr__()方法来返回显示对象状态的字符串。

+   eq=True：默认情况下，提供了 __eq__()和 __ne__()方法。这些方法实现了==和!=运算符。

+   order=False：不会自动创建 __lt__(), __le__(), __gt__(), 和 __ge__()方法。这些方法实现了<, <=, >, 和 >=运算符。

+   unsafe_hash=False：通常，可变对象没有哈希值，不能用作字典的键或集合的元素。可以自动添加 __hash__()方法，但这很少是可变对象的一个明智选择，这就是为什么这个选项被称为“不安全”的哈希。

+   frozen=False：这创建了一个不可变对象。有关更多详细信息，请参阅本章中的使用冻结数据类创建不可变对象配方。

由于为我们编写了大量的代码，我们可以专注于类定义的属性。我们可以编写真正有特色的函数，避免编写具有明显定义的“样板”方法。

## 7.5.4 更多...

一副牌需要一种初始化方法来提供 Card 对象的集合。一个默认的 __init__()方法可以填充这个集合。

考虑创建一副牌，而不是一手牌。初始牌组是一个不需要初始化方法来设置实例变量的数据类的例子。相反，牌组需要一个没有参数的自定义 __init__()方法；它总是创建相同的 52 个 Card 对象集合。这意味着我们将使用 init=False 在@dataclass 装饰器中定义这个方法，用于 Deck 类定义。

@dataclass 定义的一般模式是提供类级别的名称，这些名称既用于定义实例变量，也用于创建初始化方法 __init__()。这涵盖了状态对象的一个常见用例。

然而，在某些情况下，我们想要定义一个不用于创建实例变量但将保留为类级别变量的类级别变量。这可以通过 ClassVar 类型提示来完成。ClassVar 类型表示一个不是实例变量或 __init__()方法部分的类级别变量。

在以下示例中，我们将创建一个具有花色字符串序列的类变量：

```py
 import random 

from typing import ClassVar 

@dataclass(init=False) 

class Deck: 

    SUITS: ClassVar[tuple[str, ...]] = ( 

    ’\N{Black Club Suit}’, 

    ’\N{White Diamond Suit}’, 

    ’\N{White Heart Suit}’, 

    ’\N{Black Spade Suit}’ 

    ) 

    cards: list[CardPoints] 

    def __init__(self) -> None: 

        self.cards = [ 

            CardPoints(rank=r, suit=s) 

            for r in range(1, 14)
```

此示例类定义提供了一个类级变量 `SUITS`，它是 `Deck` 类的一部分。此变量是用于定义花色的字符的元组。

`cards` 变量有一个提示表明它将具有 `list[CardPoints]` 类型。此信息被 `mypy` 程序用于确认 `__init__()` 方法的主体正确初始化了此属性。它还确认此属性被其他类适当地使用。

## 7.5.5 参考信息

+   查看使用 `typing.NamedTuple` 构建无状态对象 菜单了解如何为无状态对象构建类定义。

+   使用类封装数据和处理 菜单涵盖了构建不使用 `@dataclass` 装饰器创建的额外方法的类技术。

# 7.6 使用冻结数据类实现不可变对象

在 使用 `typing.NamedTuple` 构建无状态对象 菜单中，我们看到了如何定义具有固定属性集的类。这些属性可以通过 `mypy` 程序进行检查，以确保它们被正确使用。在某些情况下，我们可能想使用稍微更灵活的数据类来创建不可变对象。

使用数据类的一个潜在原因是因为它比 `NamedTuple` 子类具有更复杂的字段定义。另一个潜在原因是能够自定义初始化和创建的哈希函数。由于 `NamedTuple` 实质上是一个元组，因此在此类中调整实例的行为的能力有限。

## 7.6.1 准备工作

我们将重新审视定义具有等级和花色的简单扑克牌的想法。等级可以通过介于 1（A）和 13（K）之间的整数来表示。花色可以通过集合 {‘♠’，‘♡’，‘♢’，‘♣’} 中的单个 Unicode 字符来表示。由于牌的等级和花色不会改变，我们将创建一个小的、冻结的数据类来表示这一点。

## 7.6.2 如何实现...

1.  从 `dataclasses` 模块导入 `dataclass` 装饰器：

    ```py
     from dataclasses import dataclass
    ```

1.  使用 `@dataclass` 装饰器开始类定义，使用 `frozen=True` 选项确保对象是不可变的。我们还包含了 `order=True` 以定义比较运算符，允许将此类实例按顺序排序：

    ```py
     @dataclass(frozen=True, order=True) 

    class Card:
    ```

1.  为此类每个实例的属性提供属性名称和类型提示：

    ```py
     rank: int 

        suit: str
    ```

我们可以在代码中如下使用这些对象：

```py
 >>> eight_hearts = Card(rank=8, suit=’\N{White Heart Suit}’) 

>>> eight_hearts 

Card(rank=8, suit=’’) 

>>> eight_hearts.rank 

8 

>>> eight_hearts.suit 

’’
```

我们已经创建了一个具有特定等级和花色属性的 `Card` 类实例。由于该对象是不可变的，任何尝试更改状态的操作都将导致一个异常，如下面的示例所示：

```py
 >>> eight_hearts.suit = ’\N{Black Spade Suit}’ 

Traceback (most recent call last): 

... 

dataclasses.FrozenInstanceError: cannot assign to field ’suit’
```

这显示了尝试更改冻结数据类实例的属性。`dataclasses.FrozenInstanceError` 异常被抛出以表示此类操作是不允许的。

## 7.6.3 它是如何工作的...

这个 @dataclass 装饰器向类定义中添加了多个内置方法。正如我们在使用 dataclasses 处理可变对象配方中提到的，有一些特性可以被启用或禁用。每个特性可能会让我们在类定义中包含一个或多个单独的方法。

## 7.6.4 更多内容...

@dataclass 初始化方法相当复杂。我们将探讨一个有时很有用的特性，用于定义可选属性。

考虑一个可以持有牌手的类。虽然常见用例提供了牌集来初始化手，但我们也可以有在游戏中逐步构建的手，从空集合开始，并在游戏过程中添加牌。

我们可以使用 dataclasses 模块中的 field() 函数定义这种可选属性。field() 函数允许我们提供一个函数来构建默认值，称为 default_factory。我们将在以下示例中这样使用它：

```py
 from dataclasses import dataclass, field 

@dataclass(frozen=True, order=True) 

class Hand: 

    cards: list[CardPoints] = field(default_factory=list)
```

Hand dataclass 有一个单一属性，cards，它是一个 CardPoints 对象的列表。field() 函数提供了一个默认工厂：如果没有提供初始值，将执行 list() 函数来创建一个新的空列表。

我们可以使用这个 dataclass 创建两种类型的手。以下是一个传统示例，其中我们处理六张牌：

```py
 >>> cards = [ 

... CardPoints(rank=3, suit=’\N{WHITE DIAMOND SUIT}’), 

... CardPoints(rank=6, suit=’\N{BLACK SPADE SUIT}’), 

... CardPoints(rank=7, suit=’\N{WHITE DIAMOND SUIT}’), 

... CardPoints(rank=1, suit=’\N{BLACK SPADE SUIT}’), 

... CardPoints(rank=6, suit=’\N{WHITE DIAMOND SUIT}’), 

... CardPoints(rank=10, suit=’\N{WHITE HEART SUIT}’)] 

>>> 

>>> h = Hand(cards)
```

The Hands() 类型期望一个单一属性，与该类中属性的定义相匹配。这是可选的，我们可以像以下示例中那样构建一个空手：

```py
 >>> crib = Hand() 

>>> d3 = CardPoints(rank=3, suit=’\N{WHITE DIAMOND SUIT}’) 

>>> h.cards.remove(d3) 

>>> crib.cards.append(d3) 

>>> from pprint import pprint 

>>> pprint(crib) 

Hand(cards=[CardPoints(rank=3, suit=’’)])
```

在这个例子中，我们创建了一个没有参数值的 Hand() 实例，并将其分配给 crib 变量。由于 cards 属性是用提供了一个 default_factory 的字段定义的，因此将使用 list() 函数为 cards 属性创建一个空列表。

## 7.6.5 另请参阅

+   使用 dataclasses 处理可变对象配方涵盖了使用 dataclasses 避免编写类定义复杂性的额外主题。

# 7.7 使用 __slots__ 优化小对象

对象的一般情况允许动态属性集合。对于基于元组类的固定属性集合的对象有一个特殊情况。我们在使用 typing.NamedTuple 处理不可变对象配方中探讨了这两个。

存在一个折衷方案。我们也可以定义一个具有固定数量属性的对象，但属性的值可以更改。通过将类从无限属性集合转换为固定属性集，我们发现我们还可以节省内存和处理时间。

我们如何创建具有固定属性集的优化类？

## 7.7.1 准备工作

通常，Python 允许向对象添加属性。这可能是不可取的，尤其是在处理大量对象时。大多数类定义使用字典的方式的灵活性在内存使用上是有代价的。使用特定的 __slots__ 名称将类限制在命名属性上，从而节省内存。

例如，Cribbage 纸牌游戏有几个组成部分：

+   一副牌。

+   两名玩家，他们将轮流担任庄家和对手的角色。

这个小领域的事物似乎适合作为类定义的候选。每位玩家都有一手牌和一个分数。玩家的角色是一个有趣的复杂因素。两个角色之间有一些重要差异。

+   作为庄家的玩家将获得 crib 牌。

+   如果起始牌是 JACK，庄家角色将为此获得分数。

+   对手先出第一张牌。

+   对手先计算他们的手牌。

+   庄家从他们的手中出牌，但计算他们的手牌和 crib。

比赛的特定顺序和计分方式很重要，因为第一个通过 120 分的玩家就是赢家，无论游戏处于何种状态。

看起来 Cribbage 游戏包括一副牌和两名玩家。属于庄家的 crib（底牌）可以被视为游戏整体的一个特性。当新一轮游戏开始时，我们将探讨如何切换庄家和对手的角色。

## 7.7.2 如何实现...

在创建类时，我们将利用 __slots__ 特殊名称：

1.  定义一个具有描述性名称的类：

    ```py
     class Cribbage:
    ```

1.  定义属性名称列表。这标识了允许此类实例的唯一两个属性。任何尝试添加另一个属性都将引发 AttributeError 异常：

    ```py
     __slots__ = (’deck’, ’players’, ’crib’, ’dealer’, ’opponent’)
    ```

1.  添加一个初始化方法。这必须为命名槽位创建实例变量：

    ```py
     def __init__( 

                self, 

                deck: Deck, 

                player1: Player, 

                player2: Player 

        ) -> None: 

            self.deck = deck 

            self.players = [player1, player2] 

            random.shuffle(self.players) 

            self.dealer, self.opponent = self.players 

            self.crib = Hand()
    ```

    Deck 类的定义在本章的使用 dataclasses 创建可变对象配方中展示。

1.  添加更新集合的方法。在这个例子中，我们定义了一个切换角色的方法。

    ```py
     def new_deal(self) -> None: 

            self.deck.shuffle() 

            self.players = list(reversed(self.players)) 

            self.dealer, self.opponent = self.players 

            self.crib = Hand() 
    ```

以下是我们可以使用此类构建一手牌的方法。我们需要 Card 类的定义，基于使用 typing.NamedTuple 创建不可变对象配方中的示例：

```py
 >>> deck = Deck() 

>>> c = Cribbage(deck, Player("1"), Player("2")) 

>>> c.dealer 

Player(name=’2’) 

>>> c.opponent 

Player(name=’1’) 

>>> c.new_deal() 

>>> c.dealer 

Player(name=’1’) 

>>> c.opponent 

Player(name=’2’)
```

初始的 Cribbage 对象是用 Deck 和两个 Player 实例创建的。这三个对象填充了牌和玩家槽位。然后 __init__()方法随机化玩家，使其中一名成为庄家，另一名成为对手。crib 被初始化为一个空的 Hand 实例。

new_deal()方法会对 Cribbage 实例的状态进行多项更改。这可以通过检查庄家和对手属性来揭示。

如果我们尝试创建一个新属性，会发生以下情况：

```py
 >>> c.some_other_attribute = True 

Traceback (most recent call last): 

... 

AttributeError: ’Cribbage’ object has no attribute ’some_other_attribute’ 
```

我们尝试在 Cribbage 对象 c 上创建一个名为 some_other_attribute 的属性。这引发了一个 AttributeError 异常。使用 __slots__ 意味着不能向类的实例添加新属性。

## 7.7.3 它是如何工作的...

当我们创建一个对象实例时，该过程中的步骤部分由对象的类和内置的 type() 函数定义。隐式地，一个类有一个特殊的 __new__() 方法，用于处理创建新、空对象所需的内部管理。之后，__init__() 方法创建并初始化属性。

Python 有三个创建类实例的基本路径：

+   当我们定义一个类而没有做任何不寻常的事情时，默认行为是由内置的 object 和 type() 函数定义的。每个实例都包含一个 __dict__ 属性，用于存储所有其他属性。因为对象的属性保存在字典中，所以我们可以自由地添加、更改和删除属性。这种灵活性需要为每个实例内部的字典对象使用额外的内存。

+   __slots__ 行为避免了创建 __dict__ 属性。因为对象只有 __slots__ 序列中命名的属性，所以我们不能添加或删除属性。我们可以更改定义的属性值。这种缺乏灵活性意味着每个对象使用的内存更少。

+   元组子类的行为定义了不可变对象。创建这些类的一个简单方法是以 typing.NamedTuple 作为父类。一旦构建，实例就是不可变的，不能被更改。虽然可以直接从元组中派生，但 NamedTuple 的额外功能似乎使这成为理想的选择。

一个大型应用程序可能会受到内存使用的限制，将具有最大实例数的类切换到 __slots__ 可以提高性能。

## 7.7.4 更多...

可以调整 __new__() 方法的工作方式，用不同类型的字典替换默认的 __dict__ 属性。这是一个高级技术，因为它暴露了类和对象的内部工作原理。

Python 依赖于元类来创建类的实例。默认的元类是 type 类。其思想是元类提供了一些用于创建每个对象的功能。一旦创建了空对象，类的 __init__() 方法将初始化这个空对象。

通常，元类会提供一个 __new__() 方法的定义，如果需要定制对象，可能还会提供 __prepare__()。Python 语言参考文档中有一个广泛使用的例子，它调整了用于创建类的命名空间。

更多详情，请参阅 [`docs.python.org/3/reference/datamodel.html#metaclass-example`](https://docs.python.org/3/reference/datamodel.html#metaclass-example)。

## 7.7.5 参见

+   不可变对象或完全灵活对象的更常见情况在 使用 typing.NamedTuple 创建不可变对象 章节中进行了介绍。

# 7.8 使用更复杂的集合

Python 拥有丰富的内置集合。在第四章中，我们对其进行了详细探讨。在选择数据结构的配方中，我们提供了一个决策树，以帮助从可用的选择中定位适当的数据结构。

当我们考虑标准库中的内置类型和其他数据结构时，我们有更多的选择，需要做出的决定也更多。我们如何为我们的问题选择正确的数据结构？

## 7.8.1 准备工作

在我们将数据放入集合之前，我们需要考虑我们将如何收集数据，以及一旦我们拥有它，我们将如何处理这个集合。始终存在的一个大问题是我们在集合中如何识别特定的项目。我们将探讨一些关键问题，这些问题需要我们回答，以帮助我们选择适合我们需求的适当集合。

这里是一些替代集合的概述。collections 模块包含许多内置集合的变体。以下是一些包括的内容：

+   deque：一个双端队列。这是一个可变序列，对从两端推入和弹出进行了优化。请注意，类名以小写字母开头；这在 Python 中是不典型的。

+   defaultdict：一种可以为一个缺失的键提供默认值的映射。请注意，类名以小写字母开头；这在 Python 中是不典型的。

+   Counter：一种设计用来计算不同键出现次数的映射。这有时被称为多重集或包。

+   ChainMap：一种将多个字典组合成一个单一映射的映射。

heapq 模块包含一个优先队列实现。这是一个专门化的库，利用内置的列表序列来保持项目排序。

bisect 模块包含搜索排序列表的方法。这在大字典功能和列表功能之间产生了一些重叠。

此外，collections 模块中还有一个 OrderedDict 类。从 Python 3.7 开始，普通字典的键按创建顺序保留，这使得 OrderedDict 类变得冗余。

## 7.8.2 如何实现...

我们需要回答一些问题来决定是否需要一个库数据集合而不是内置集合：

1.  这种结构是生产者和消费者之间的缓冲吗？算法的某个部分产生数据项，而另一个部分消费数据项吗？

    +   队列用于先入先出（FIFO）处理。项目在一端插入，从另一端消费。我们可以使用 list.append()和 list.pop(0)来模拟这个过程，尽管 collections.deque 将更高效；我们可以使用 deque.append()和 deque.popleft()。

    +   栈用于后进先出（LIFO）处理。项目从同一端插入和消耗。我们可以使用 list.append()和 list.pop()来模拟这一点，尽管 collections.deque 将更高效；我们可以使用 deque.append()和 deque.pop()。

    +   优先队列（或堆队列）按某种顺序保持队列排序，这种顺序与到达顺序不同。我们可以通过使用 list.append()、list.sort(key=lambda x:x.priority)和 list.pop(-1)操作来模拟这一点，以保持项目按优先级排序。每次插入后进行排序可能会使其效率低下。使用 heapq 模块可能更高效。heapq 模块有用于创建和更新堆的函数。

1.  我们应该如何处理字典中的缺失键？

    +   抛出异常。这是内置的 dict 类的工作方式。

    +   创建一个默认项。这是 collections.defaultdict 的工作方式。我们必须提供一个返回默认值的函数。常见的例子包括 defaultdict(int)和 defaultdict(float)来使用默认值 0 或 0.0。我们还可以使用 defauldict(list)和 defauldict(set)来创建字典-of-list 或字典-of-set 结构。

    +   用于创建计数字典的 defaultdict(int)非常常见，以至于 collections.Counter 类正是这样做的。

1.  我们希望如何处理字典中键的顺序？通常，Python 3.6 以上版本会保持键的插入顺序。如果我们想有不同的顺序，我们将不得不手动排序它们。

1.  我们将如何构建字典？

    +   我们有一个简单的算法来创建项。在这种情况下，一个内置的 dict 对象可能就足够了。

    +   我们有多个需要合并的字典。这可能在读取配置文件时发生。我们可能有一个单独的配置、系统范围的配置以及默认的应用程序配置，所有这些都需要使用 ChainMap 集合合并成一个单一的字典。

## 7.8.3 它是如何工作的...

数据处理有两个主要资源约束：

+   存储

+   时间

我们的所有编程都必须遵守这些约束。在大多数情况下，这两者是相反的：我们为了减少存储使用而做的事情往往会增加处理时间，而我们为了减少处理时间而做的事情会增加存储使用。算法和数据结构设计寻求在约束之间找到一个最佳平衡。

时间方面通过复杂度指标形式化。描述算法复杂性的方法有很多：

+   复杂度 O(1)不随数据量的大小而改变。对于某些集合，实际的长期平均整体几乎接近 O(1)，但有少数例外。许多字典操作是 O(1)。向列表中添加元素，以及从列表末尾弹出元素非常快，使得 LIFO 栈非常高效。从列表前面弹出元素是 O(n)，这使得由简单列表构建的 FIFO 队列相当昂贵；deque 类和 heapq 模块通过更好的设计来解决这个问题。

+   被描述为 O(log n)的复杂度意味着成本的增长速度低于数据量 n 的增长速度。二分查找模块允许我们通过将列表分成两半来更有效地搜索排序后的列表。请注意，首先对列表进行排序是 O(nlog n)，因此需要大量的搜索来分摊排序的成本。

+   被描述为 O(n)的复杂度意味着成本随着数据量 n 的增长而增长。在列表中查找一个项目具有这种复杂度。如果项目在列表的末尾，必须检查所有 n 个项目。集合和映射没有这个问题，并且具有接近 O(1)的复杂度。

+   被描述为 O(nlog n)的复杂度比数据量增长得更快。排序列表通常具有这种复杂度。因此，最小化或消除大量数据的排序是有帮助的。

+   有些情况甚至更糟。一些算法的复杂度为 O(n²)、O(2^n)，甚至 O(n!)。我们希望通过巧妙的设计和良好的数据结构选择来避免这些非常昂贵的算法。在实践中，这些算法可能会具有欺骗性。我们可能能够设计出一个 O(2^n)的算法，在 n 为 3 或 4 的小测试用例中似乎表现良好。在这些情况下，组合数只有 8 或 16 种。如果实际数据涉及 70 个项目，组合数将达到 10²²的数量级，一个有 22 位数的数字。

标准库中可用的各种数据结构反映了时间和存储之间的许多权衡。

## 7.8.4 更多内容...

作为具体和极端的例子，让我们看看搜索一个特定事件序列的 Web 日志文件。我们有两个总体设计策略：

+   使用类似 file.read().splitlines()的方法将所有事件读入一个列表结构中。然后我们可以使用 for 语句遍历列表，寻找事件的组合。虽然初始读取可能需要一些时间，但由于日志全部在内存中，搜索将会非常快。

+   从日志文件中逐个读取和处理事件。当一个日志条目是搜索到的模式的一部分时，只保存这个事件在日志的子集中是有意义的。我们可能使用一个以会话 ID 或客户端 IP 地址作为键，事件列表作为值的 defaultdict。这将花费更长的时间来读取日志，但内存中的结果结构将比所有日志条目的列表小得多。

第一个算法，将所有内容读入内存，可能非常不切实际。在一个大型网络服务器上，日志可能涉及数百 GB 的数据。日志可能太大，无法放入任何计算机的内存中。

第二种方法有几个替代实现：

+   单进程：这里大多数 Python 食谱的一般方法假设我们正在创建一个作为单个进程运行的应用程序。

+   多进程：我们可能会将逐行搜索扩展为多进程应用程序，使用 multiprocessing 或 concurrent.futures 包。这些包让我们创建一组工作进程，每个进程可以处理可用数据的一个子集，并将结果返回给一个消费者，该消费者将结果组合起来。在现代多处理器、多核计算机上，这可以是非常有效的资源利用方式。

+   多主机：极端情况需要多个服务器，每个服务器处理数据的一个子集。这需要在主机之间进行更复杂的协调以共享结果集。通常，使用 Dask 或 Spark 这样的框架来处理这类工作效果很好。虽然 multiprocessing 模块相当复杂，但像 Dask 这样的工具更适合大规模计算。

我们经常将大搜索分解为映射和归约处理。映射阶段对集合中的每个项目应用一些处理或过滤。归约阶段将映射结果组合成摘要或聚合对象。在许多情况下，有一个复杂的 Map-Reduce 阶段层次结构应用于先前 Map-Reduce 操作的结果。

## 7.8.5 相关阅读

+   请参阅第四章中的选择数据结构配方，以了解选择数据结构的基础决策集。

# 7.9 扩展内置集合 – 一个能进行统计的列表

在设计具有大量处理的类的配方中，我们查看了一种区分复杂算法和集合的方法。我们展示了如何将算法和数据封装到不同的类中。另一种设计策略是扩展集合以包含有用的算法。

我们如何扩展 Python 的内置集合？我们如何向内置列表添加功能？

## 7.9.1 准备工作

我们将创建一个复杂的列表类，其中每个实例都可以计算列表中项目的总和和平均值。这需要应用仅将数字放入列表中；否则，将引发 ValueError 异常。

我们将展示一些方法，这些方法明确使用生成器表达式作为可以包含额外处理的地方。我们不会使用 sum(self)，而是强调 sum(v for v in self)，因为有两个常见的未来扩展：sum(m(v) for v in self) 和 sum(v for v in self if f(v))。这些是映射和过滤的替代方案，其中映射函数 m(v) 应用于每个项目；或者过滤函数 f(v) 应用于通过或拒绝每个项目。例如，计算平方和将映射应用于计算每个值的平方，然后再求和。

## 7.9.2 如何实现...

1.  为列表选择一个同时也能进行简单统计的名字。将类定义为内置列表类的扩展：

    ```py
     class StatsList(list[float]):
    ```

    我们可以坚持使用通用的类型提示 list。这通常太宽泛了。由于结构将包含数字，使用更窄的提示 list[float]更合理。

    当处理数值数据时，mypy 将 float 类型视为 float 和 int 的超类，从而节省了我们定义显式 Union[float, int]的需要。

1.  将额外的处理定义为方法。self 变量将是一个继承了超类所有属性和方法的对象。在这种情况下，超类是 list[float]。我们在这里使用生成器表达式作为一个可能包含未来更改的地方。以下是一个 sum()方法：

    ```py
     def sum(self) -> float: 

            return sum(v for v in self)
    ```

1.  这里是另一个我们经常应用于列表的方法。它计算项目数量并返回大小。我们使用生成器表达式使其易于添加映射或过滤条件，如果需要的话：

    ```py
     def size(self) -> float: 

            return sum(1 for v in self)
    ```

1.  这里是平均数方法：

    ```py
     def mean(self) -> float: 

            return self.sum() / self.size()
    ```

1.  这里有一些额外的函数。sum2()方法计算列表中值的平方和。这用于计算方差。然后使用方差来计算列表中值的标准差。与之前的 sum()和 count()方法不同，那里没有映射，在这种情况下，生成器表达式包括一个映射转换：

    ```py
     def sum2(self) -> float: 

            return sum(v ** 2 for v in self) 

        def variance(self) -> float: 

            return ( 

              (self.sum2() - self.sum() ** 2 / self.size()) 

              / (self.size() - 1) 

            ) 

        def stddev(self) -> float: 

            return math.sqrt(self.variance())
    ```

StatsList 类的定义继承了内置列表对象的所有特性。它通过我们添加的方法进行了扩展。以下是创建此集合中实例的示例：

```py
 >>> subset1 = StatsList([10, 8, 13, 9, 11]) 

>>> data = StatsList([14, 6, 4, 12, 7, 5]) 

>>> data.extend(subset1)
```

我们从字面列表对象中创建了两个 StatsList 对象，分别是 subset1 和 data。我们使用了从列表超类继承的 extend()方法来合并这两个对象。以下是结果对象：

```py
 >>> data 

[14, 6, 4, 12, 7, 5, 10, 8, 13, 9, 11]
```

这是我们可以使用这个对象上定义的额外方法的方式：

```py
 >>> data.mean() 

9.0 

>>> data.variance() 

11.0
```

我们已经展示了 mean()和 variance()方法的结果。内置列表类的所有特性也存在于我们的扩展中。

## 7.9.3 它是如何工作的...

类定义的一个基本特征是继承的概念。当我们创建一个超类-子类关系时，子类继承超类的所有特性。这有时被称为泛化-特殊化关系。超类是一个更通用的类；子类更特殊，因为它添加或修改了特性。

所有内置类都可以扩展以添加功能。在这个例子中，我们添加了一些统计处理，创建了一个特殊的数字列表子类。

两种设计策略之间存在重要的紧张关系：

+   扩展：在这种情况下，我们扩展了一个类以添加功能。这些特性与这个单一的数据结构紧密相连，我们无法轻易地将其用于不同类型的序列。

+   包装：在设计具有大量处理功能的类时，我们保持了处理与集合的分离。这导致在处理两个对象时出现一些复杂性。

很难说其中哪一种本质上优于另一种。在许多情况下，我们会发现包装可能具有优势，因为它似乎更适合 SOLID 设计原则。然而，通常会有一些情况，扩展内置集合是合适的。

## 7.9.4 更多内容...

概化的想法可能导致抽象超类。由于抽象类是不完整的，它需要一个子类来扩展它并提供缺失的实现细节。我们不能创建一个抽象类的实例，因为它会缺少使其有用的功能。

正如我们在第四章的选择数据结构配方中提到的，所有内置集合都有抽象超类。我们不仅可以从一个具体类开始，还可以从一个抽象基类开始我们的设计。

例如，我们可以这样开始一个类定义：

```py
 from collections.abc import MutableMapping 

class MyFancyMapping(MutableMapping[int, int]): 

    ... # etc.
```

为了完成这门课程，我们需要为一些特殊方法提供实现：

+   `__getitem__`

+   `__setitem__`

+   `__delitem__`

+   `__iter__`

+   `__len__`

这些方法在抽象类中都不存在；在 Mapping 类中没有具体的实现。一旦我们为每个方法提供了可行的实现，我们就可以创建新子类实例。

## 7.9.5 参考信息

+   在设计大量处理类的配方中，我们采取了不同的方法。在那个配方中，我们将复杂的算法留在了另一个类中。

# 7.10 使用属性懒加载

在设计大量处理类的配方中，我们定义了一个类，它会急切地计算集合中数据的许多属性。那里的想法是尽可能早地计算值，这样属性就不会有进一步的计算成本。

我们将其描述为急切处理，因为工作尽可能早地完成。另一种方法是懒处理，其中工作尽可能晚地完成。

如果我们有一些很少使用且计算成本很高的值，我们该怎么办？我们如何最小化初始计算，并且只在真正需要时计算值？

## 7.10.1 准备工作...

对于背景信息，请参阅 NIST 气溶胶粒子尺寸案例研究：[`www.itl.nist.gov/div898/handbook/pmc/section6/pmc62.htm`](https://www.itl.nist.gov/div898/handbook/pmc/section6/pmc62.htm)

在本章中，请参阅设计大量处理类的配方以获取更多关于此数据集的详细信息。与处理原始数据相比，处理包含在 Counter 对象中的摘要信息可能会有所帮助。配方展示了从粒子大小到数量的映射，以及特定大小被测量的次数。

我们想计算这个 Counter 的一些统计数据。我们有两个总体策略来完成这项工作：

+   扩展：我们在扩展内置集合 – 执行统计的列表的菜谱中详细介绍了这一点，我们将在第八章中查看其他扩展类的例子。

+   包装：我们可以将 Counter 对象包装在另一个只提供所需功能的类中。我们将在第八章中探讨这一点。

在包装时的一种常见变体是创建一个与数据收集对象分开的统计计算对象。这种包装的变体通常会导致一个优雅的解决方案。

无论我们选择哪种类架构，我们还有两种设计处理的方式：

+   贪婪：这意味着我们将尽快计算统计值。这是设计具有大量处理的类菜谱中采用的方法。

+   懒惰：这意味着我们不会在需要之前通过方法函数或属性进行任何计算。在扩展内置集合 – 执行统计的列表的菜谱中，我们向集合类添加了方法。这些额外的方法是惰性计算的例子。统计值仅在需要时才进行计算。

两种设计的基本数学是相同的。唯一的问题是何时进行计算。

## 7.10.2 如何实现...

1.  使用描述性的名称定义类：

    ```py
     class LazyCounterStatistics:
    ```

1.  编写初始化方法以包含此对象将要连接的对象。我们定义了一个方法函数，它接受一个 Counter 对象作为参数值。这个 Counter 对象被保存为 Counter_Statistics 实例的一部分：

    ```py
     def __init__(self, raw_counter: Counter[int]) -> None: 

            self.raw_counter = raw_counter
    ```

1.  定义一些有用的辅助方法。这些方法中的每一个都用@property 装饰，使其表现得像简单的属性：

    ```py
     @property 

        def sum(self) -> float: 

            return sum( 

                f * v 

                for v, f in self.raw_counter.items() 

            ) 

        @property 

        def count(self) -> float: 

            return sum( 

                f 

                for v, f in self.raw_counter.items() 

            )
    ```

1.  定义所需的各种值的必要方法。以下是对平均值的计算。这也用@property 装饰。其他方法可以像属性一样引用，尽管它们是正确的方法函数：

    ```py
     @property 

        def mean(self) -> float: 

            return self.sum / self.count
    ```

1.  这是我们如何计算标准差的方法。注意，我们一直在使用 math.sqrt()。务必在 Python 模块中添加所需的 import math 语句：

    ```py
     @property 

        def sum2(self) -> float: 

            return sum( 

                f * v ** 2 

                for v, f in self.raw_counter.items() 

            ) 

        @property 

        def variance(self) -> float: 

          return ( 

              (self.sum2 - self.sum ** 2 / self.count) / 

              (self.count - 1) 

          ) 

        @property 

        def stddev(self) -> float: 

            return math.sqrt(self.variance)
    ```

为了展示这是如何工作的，我们将这个类的实例应用于一些汇总数据。这本书的代码库包括一个 data/binned.csv 文件，它包含汇总数据的分箱。这些数据有三个列：size_code，size 和 frequency。我们只对 size_code 和 frequency 感兴趣。

这是我们如何从这个文件中构建一个合适的 Counter 对象的方法：

```py
 >>> from pathlib import Path 

>>> import csv 

>>> from collections import Counter 

>>> data_path = Path.cwd() / "data" / "binned.csv" 

>>> with data_path.open() as data_file: 

...     reader = csv.DictReader(data_file) 

...     extract = { 

...         int(row[’size_code’]): int(row[’frequency’]) 

...         for row in reader 

...     } 

>>> data = Counter(extract)
```

我们使用字典推导来创建从 size_code 到该代码值频率的映射。然后，我们将它提供给 Counter 类，从现有的汇总中构建一个名为 data 的 Counter 对象。

这是我们如何分析 Counter 对象的方法：

```py
 >>> stats = LazyCounterStatistics(data) 

>>> print(f"Mean: {stats.mean:.1f}") 

Mean: 10.4 

>>> print(f"Standard Deviation: {stats.stddev:.2f}") 

Standard Deviation: 4.17
```

我们提供了数据对象来创建 LazyCounterStatistics 类的实例，即 stats 变量。当我们打印 stats.mean 属性和 stats.stddev 属性的值时，将调用方法来进行适当的值计算。

计算成本只有在客户端对象请求 stats.mean 或 stats.stddev 属性值时才会支付。这将引发一系列计算来计算这些值。

当底层数据发生变化时，整个计算将再次执行。在高度动态数据的罕见情况下，这可能很昂贵。在更常见的情况下，分析先前汇总的数据，这相当高效。

## 7.10.3 它是如何工作的...

当值很少使用时，懒计算的思路效果很好。在这个例子中，计数在计算方差和标准差时被计算了两次。

在值频繁重新计算的情况下，一个简单的懒设计在某些情况下可能不是最优的。这通常是一个容易解决的问题。我们总是可以创建额外的局部变量来缓存中间结果，而不是重新计算。我们将在本食谱的后面讨论这个问题。

为了使这个类看起来像它已经执行了急切计算，我们使用了@property 装饰器。这使得一个方法看起来像是一个属性。这只能适用于没有参数值的方法。

在所有情况下，一个被急切计算的属性可以被一个懒属性所替代。创建急切属性变量的主要原因是优化计算成本。在计算结果可能不会被总是使用的情况下，懒属性可以避免昂贵的计算。

## 7.10.4 更多内容...

有一些情况下，我们可以进一步优化一个属性，以限制在值变化时所做的额外计算量。这需要仔细分析使用案例，以便理解底层数据的更新模式。

在数据被加载到集合中并执行分析的情况下，我们可以缓存结果以避免第二次计算。我们可能做如下操作：

```py
 from typing import cast 

class CachingLazyCounterStatistics: 

    def __init__(self, raw_counter: Counter[int]) -> None: 

        self.raw_counter = raw_counter 

        self._sum: float | None = None 

        self._count: float | None = None 

    @property 

    def sum(self) -> float: 

        if self._sum is None: 

            self._sum = sum( 

                f * v 

                for v, f in self.raw_counter.items() 

            ) 

        return self._sum
```

这种技术使用两个属性来保存求和和计数计算的值，self._sum 和 self._count。这些值将只计算一次，并在需要时返回，无需额外的计算成本。

类型提示显示这些属性是可选的。一旦 self._sum 和 self._count 的值被计算出来，这些值就不再是可选的，但将始终存在。我们用 cast()类型提示向像 mypy 这样的工具描述这一点。这个提示告诉类型检查工具将 self._sum 视为一个 float 对象，而不是 float | None 对象。这个函数没有成本，因为它什么也不做；它的目的是注释处理过程，以显示设计意图。

如果 raw_counter 对象的状态从不改变，这种缓存优化是有帮助的。在一个更新底层 Counter 的应用程序中，这个缓存的值会过时。这种类型的应用程序需要在底层 Counter 更新时重置 self._sum 和 self._count 的内部缓存值。

## 7.10.5 参见...

+   在设计具有大量处理的类的配方中，我们定义了一个类，它急切地计算了许多属性。这代表了一种管理计算成本的不同策略。

# 7.11 创建上下文和上下文管理器

许多 Python 对象表现得像上下文管理器。其中一些最明显的例子是文件对象。我们通常使用 with path.open() as file:来在一个可以保证资源释放的上下文中处理文件。在第二章中，使用 with 语句管理上下文的配方涵盖了使用基于文件的上下文管理器的基础知识。

我们如何创建自己的类，使其充当上下文管理器？

## 7.11.1 准备工作

我们将在第三章的基于部分函数选择参数顺序配方中查看一个函数。这个配方介绍了一个函数，haversine()，它有一个上下文类似的参数，用于将答案从无量纲弧度调整到有用的单位，如公里、海里或美国英里。在许多方面，这个距离因子是一种上下文，用于定义所进行的计算类型。

我们希望能够使用 with 语句来描述一个变化不快的对象；实际上，这种变化充当了一种边界，定义了计算的范围。我们可能想要使用如下代码：

```py
 >>> with Distance(r=NM) as nm_dist: 

...     print(f"{nm_dist(p1, p2)=:.2f}") 

...     print(f"{nm_dist(p2, p3)=:.2f}") 

nm_dist(p1, p2)=39.72 

nm_dist(p2, p3)=30.74
```

Distance(r=NM)构造函数提供了上下文的定义，创建了一个新的对象 nm_dist，该对象已配置为以海里为单位执行所需的计算。这只能在 with 语句体中使用。

这个 Distance 类定义可以看作是创建了一个部分函数，nm_dist()。这个函数为使用 haversine()函数的多个后续计算提供了一个固定的单位参数，r。

有许多其他方法可以创建部分函数，包括 lambda 对象、functools.partial()函数和可调用对象。我们在第三章的基于部分函数选择参数顺序配方中探讨了部分函数的替代方案。

## 7.11.2 如何做...

上下文管理器类有两个特殊方法，我们需要定义：

1.  从一个有意义的类名开始：

    ```py
     class Distance:
    ```

1.  定义一个初始化器，创建上下文的任何独特功能。在这种情况下，我们想要设置使用的距离单位：

    ```py
     def __init__(self, r: float) -> None: 

            self.r = r
    ```

1.  定义 `__enter__()` 方法。当 `with` 语句块开始时，会调用此方法。语句 `with Distance(r=NM) as nm_dist` 执行了两件事。首先，它创建了 `Distance` 类的实例，然后调用该对象的 `__enter__()` 方法以启动上下文。`__enter__()` 方法的返回值通过 `as` 子句分配给一个局部变量。这并不总是必需的。对于简单情况，上下文管理器通常返回自身。如果此方法需要返回同一类别的实例，请注意类尚未完全定义，必须提供类名类型提示作为字符串。对于这个配方，我们将返回一个函数，类型提示基于 `Callable`：

    ```py
     def __enter__(self) -> Callable[[Point, Point], float]: 

            return self.distance
    ```

1.  定义 `__exit__()` 方法。当上下文结束时，将调用此方法。这是释放资源和进行清理的地方。在这个例子中，不需要做更多的事情。任何异常的详细信息都提供给此方法；方法可以静默异常或允许其传播。如果 `__exit__()` 方法的返回值为 `True`，则异常将被静默。返回值 `False` 或 `None` 将允许异常在 `with` 语句外部可见：

    ```py
     def __exit__( 

            self, 

            exc_type: type[Exception] | None, 

            exc_val: Exception | None, 

            exc_tb: TracebackType | None 

        ) -> bool | None: 

            return None
    ```

1.  创建一个类（或定义此类的函数）以在上下文中工作。在这种情况下，该方法将使用第三章中单独定义的 `haversine()` 函数：

    ```py
     def distance(self, p1: Point, p2: Point) -> float: 

            return haversine( 

                p1.lat, p1.lon, p2.lat, p2.lon, R=self.r 

            )
    ```

大多数上下文管理器类需要相当多的导入：

```py
 from collections.abc import Callable 

from types import TracebackType 

from typing import NamedTuple
```

此类已定义为与 `Point` 类的对象一起工作。这可以是 `NamedTuple`、`@dataclass` 或其他提供所需两个属性的类。以下是 `NamedTuple` 的定义：

```py
 class Point(NamedTuple): 

    lat: float 

    lon: float
```

此类定义提供了一个名为 `Point` 的类，具有所需的属性名称。

## 7.11.3 它是如何工作的...

上下文管理器依赖于 `with` 语句执行大量操作。

我们将把以下结构放在显微镜下观察：

```py
 >>> p1 = Point(38.9784, -76.4922) 

>>> p2 = Point(36.8443, -76.2922) 

>>> nm_distance = Distance(r=NM) 

>>> with nm_distance as nm_calc: 

...     print(f"{nm_calc(p1, p2)=:.2f}") 

nm_calc(p1, p2)=128.48
```

第一行创建了 `Distance` 类的一个实例。该实例的 `r` 参数值等于常数 `NM`，这使得我们能够在海里进行计算。`Distance` 实例被分配给 `nm_distance` 变量。

当 `with` 语句开始执行时，通过执行 `__enter__()` 方法来通知上下文管理器对象。在这种情况下，`__enter__()` 方法返回的值是一个函数，类型为 `Callable[[Point, Point], float]`。该函数接受两个 `Point` 对象并返回一个浮点结果。`as` 子句将此函数对象分配给 `nm_calc` 名称。

`print()` 函数使用 `nm_calc` 对象来完成其工作。该对象是一个函数，将从两个 `Point` 实例计算距离。

当 with 语句结束时，__exit__() 方法将被执行。对于更复杂的上下文管理器，这可能涉及关闭文件或释放网络连接。可能需要许多种类的上下文清理。在这种情况下，不需要做任何清理上下文的事情。

这有一个优点，就是定义了一个固定边界，在这个边界内使用部分函数。在某些情况下，上下文管理器内部的计算可能涉及数据库或复杂的网络服务，从而导致更复杂的 __exit__() 方法。

## 7.11.4 更多内容…

__exit__() 方法的操作对于充分利用上下文管理器至关重要。在先前的例子中，我们使用了以下“什么都不做”的 __exit__() 方法：

```py
 def __exit__( 

        self, 

        exc_type: type[Exception] | None, 

        exc_val: Exception | None, 

        exc_tb: TracebackType | None 

    ) -> bool | None: 

        # Cleanup goes here. 

        return None 
```

这里的问题是允许任何异常正常传播。我们经常看到任何清理处理替换了 # Cleanup goes here. 注释。这就是缓冲区被刷新、文件被关闭和错误日志消息被写入的地方。

有时，我们需要处理特定的异常细节。考虑以下交互会话的片段：

```py
 >>> p1 = Point(38.9784, -76.4922) 

>>> p2 = Point(36.8443, -76.2922) 

>>> with Distance(None) as nm_dist: 

...     print(f"{nm_dist(p1, p2)=:.2f}") 

Traceback (most recent call last): 

... 

TypeError: unsupported operand type(s) for *: ’NoneType’ and ’int’
```

Distance 对象使用 r 参数值设置为 None 进行初始化。虽然这段代码会导致像 mypy 这样的工具发出警告，但从语法上是有效的。然而，TypeError 的 traceback 并不指向 Distance；它指向 haversine() 函数中的一行代码。

我们可能想报告一个 ValueError 而不是这个 TypeError。下面是 Distance 类的一个变体，它隐藏了 TypeError，用 ValueError 替换它：

```py
 class Distance_2: 

    def __init__(self, r: float) -> None: 

        self.r = r 

    def __enter__(self) -> Callable[[Point, Point], float]: 

        return self.distance 

    def __exit__( 

        self, 

        exc_type: type[Exception] | None, 

        exc_val: Exception | None, 

        exc_tb: TracebackType | None 

    ) -> bool | None: 

        if exc_type is TypeError: 

            raise ValueError(f"Invalid r={self.r!r}") 

        return None 

    def distance(self, p1: Point, p2: Point) -> float: 

        return haversine(p1.lat, p1.lon, p2.lat, p2.lon, R=self.r)
```

这显示了如何在 __exit__() 方法中检查异常的详细信息。提供的信息与 sys.exc_info() 函数类似，包括异常的类型、异常对象以及一个具有 types.TracebackType 类型的 traceback 对象。

## 7.11.5 参见

+   在第二章的使用 with 语句管理上下文 节中，我们介绍了使用基于文件的上下文管理器的基础知识。

# 7.12 使用多个资源管理多个上下文

我们经常使用上下文管理器与打开的文件一起使用。因为上下文管理器可以保证操作系统资源被释放，这样做可以防止资源泄漏。它可以用来防止在没有将所有缓冲区刷新到持久存储的情况下关闭文件。

当处理多个资源时，通常意味着需要多个上下文管理器。例如，如果我们有三个打开的文件，我们可能需要三个嵌套的 with 语句？我们如何优化或简化多个 with 语句？

## 7.12.1 准备工作

我们将查看创建包含多个腿的行程计划。我们的起始数据收集是一个定义我们路线的点列表。例如，穿越切萨皮克湾可能涉及从马里兰州安纳波利斯出发，航行到索尔 omon 岛、弗吉尼亚州的 Deltaville，然后到弗吉尼亚州的诺福克。为了规划目的，我们希望将其视为三条腿，而不是四个点。一条腿有距离，需要时间穿越：计算时间、速度和距离是规划问题的本质。

在运行配方之前，我们将先进行一些基础定义。首先是单个点的定义，具有纬度和经度属性：

```py
 @dataclass(frozen=True) 

class Point: 

    lat: float 

    lon: float
```

可以使用如下语句构建一个点：p = Point(38.9784, -76.4922)。这让我们可以在后续计算中引用 p.lat 和 p.lon。使用属性名称使代码更容易阅读。

一条腿是一对点。我们可以如下定义它：

```py
 @dataclass 

class Leg: 

    start: Point 

    end: Point 

    distance: float = field(init=False)
```

我们已将其创建为可变对象。距离属性具有由 dataclasses.field()函数定义的初始值。使用 init=False 表示在初始化对象时不会提供该属性；它必须在初始化后提供。

这是一个上下文管理器，用于从点实例创建 Leg 对象。这与创建上下文和上下文管理器配方中显示的上下文管理器类似。这里有一个微小但重要的区别。__init__()保存一个值给 self.r 来设置距离单位上下文。默认值是海里：

```py
 from types import TracebackType 

class LegMaker: 

    def __init__(self, r: float=NM) -> None: 

        self.last_point: Point | None = None 

        self.last_leg: Leg | None = None 

        self.r = r 

    def __enter__(self) -> "LegMaker": 

        return self 

    def __exit__( 

        self, 

        exc_type: type[Exception] | None, 

        exc_val: Exception | None, 

        exc_tb: TracebackType | None 

    ) -> bool | None: 

        return None
```

重要的方法 waypoint()接受一个航点并创建一个 Leg 对象。第一个航点，即航行的起点，将返回 None。所有后续的点将返回一个 Leg 对象：

```py
 def waypoint(self, next_point: Point) -> Leg | None: 

        leg: Leg | None 

        if self.last_point is None: 

            # Special case for the first leg 

            self.last_point = next_point 

            leg = None 

        else: 

            leg = Leg(self.last_point, next_point) 

            d = haversine( 

                leg.start.lat, leg.start.lon, 

                leg.end.lat, leg.end.lon, 

                R=self.r 

            ) 

            leg.distance = round(d) 

            self.last_point = next_point 

        return leg
```

此方法使用缓存的点对象 self.last_point 和下一个点 next_point 来创建一个 Leg 实例，然后更新该实例。

如果我们想在 CSV 格式中创建输出文件，我们需要使用两个上下文管理器：一个用于创建 Leg 对象，另一个用于管理打开的文件。我们将把这个复杂的多上下文处理放入一个单独的函数中。

## 7.12.2 如何做到这一点...

1.  我们将使用 csv 和 pathlib 模块。此外，此配方还将使用 Iterable 类型提示和 dataclasses 模块中的 asdict 函数：

    ```py
     from collections.abc import Iterable 

    import csv 

    from dataclasses import asdict 

    from pathlib import Path
    ```

1.  由于我们将创建 CSV 文件，我们需要定义用于 CSV 输出的标题：

    ```py
     HEADERS = ["start_lat", "start_lon", "end_lat", "end_lon", "distance"]
    ```

1.  定义一个函数，将复杂对象转换为适合写入每一行数据的字典。输入是一个 Leg 对象；输出是一个具有与 HEADERS 列表中列名匹配的键的字典：

    ```py
     def flat_dict(leg: Leg) -> dict[str, float]: 

        struct = asdict(leg) 

        return dict( 

            start_lat=struct["start"]["lat"], 

            start_lon=struct["start"]["lon"], 

            end_lat=struct["end"]["lat"], 

            end_lon=struct["end"]["lon"], 

            distance=struct["distance"], 

        )
    ```

1.  定义一个具有意义名称的函数。我们将提供两个参数：一个点对象列表和一个 Path 对象，显示 CSV 文件应该创建的位置。我们已使用 Iterable[Point]作为类型提示，因此此函数可以接受任何可迭代的点实例集合：

    ```py
     def make_route_file( 

        points: Iterable[Point], target: Path 

    ) -> None:
    ```

1.  使用单个 with 语句启动两个上下文。这将调用两个 __enter__()方法来为工作准备两个上下文。这一行可能会很长：

    ```py
     with ( 

            LegMaker(r=NM) as legger, 

            target.open(’w’, newline=’’) as csv_file 

        ):
    ```

1.  一旦上下文准备好工作，我们可以创建一个 CSV 写入器并开始写入行：

    ```py
     writer = csv.DictWriter(csv_file, HEADERS) 

            writer.writeheader() 

            for point in points: 

                leg = legger.waypoint(point) 

                if leg is not None: 

                    writer.writerow(flat_dict(leg))
    ```

1.  在上下文结束时，进行任何最终的汇总处理。这不是在 with 语句体的缩进内；它与 with 关键字本身的缩进级别相同：

    ```py
     print(f"Finished creating {target}")
    ```

    通过将此消息放在 with 上下文之外，它提供了重要的证据，表明文件已正确关闭，所有的计算都已完成。

## 7.12.3 它是如何工作的...

复合的 with 语句为我们创建了一系列上下文管理器。所有的管理器都将使用它们的 __enter__()方法来开始处理，并且可选地返回一个可以在上下文中使用的对象。LegMaker 类定义了一个返回 LegMaker 实例的 __enter__()方法。Path.open()方法返回一个 TextIO 对象；这些也是上下文管理器。

当 with 语句结束时上下文退出，将调用所有上下文管理器的 __exit__()方法。这允许每个上下文管理器执行任何最终的清理。在 TextIO 对象的情况下，这将关闭外部文件，释放正在使用的任何 OS 资源。

在 LegMaker 对象的情况下，上下文退出时没有进行最终的清理处理。创建了一个 LegMaker 对象；从 __enter__()方法返回的值是这个对象方法的引用。legger 可调用对象将继续在上下文外部正确运行。这是一个特殊的情况，在没有在 __exit__()方法中进行清理的情况下发生。如果需要防止进一步使用 legger 可调用对象，那么 __exit__()方法需要在 LegMaker 对象内部进行显式的状态改变，以便抛出异常。一种方法是在 __exit__()方法中将 self.r 值设置为 None，这将防止进一步使用 waypoint()方法。

## 7.12.4 更多内容...

上下文管理器的任务是隔离资源管理的细节。最常见的情况是文件和网络连接。我们已经展示了在算法周围使用上下文管理器来帮助管理带有单个 Point 对象的缓存。

当处理非常大的数据集时，使用压缩通常很有帮助。这可以在处理周围创建不同的上下文。内置的 open()方法通常在 io 模块中分配给 io.open()函数。这意味着我们通常可以用 bz2.open()这样的函数替换 io.open()来处理压缩文件。

我们可以用类似这样的事物替换一个未压缩的文件上下文管理器：

```py
 import bz2 

def make_route_bz2(points: Iterable[Point], target: Path) -> None: 

    with ( 

        LegMaker(r=NM) as legger, 

        bz2.open(target, "wt") as archive 

    ): 

        writer = csv.DictWriter(archive, HEADERS) 

        writer.writeheader() 

        for point in points: 

            leg = legger.waypoint(point) 

            if leg is not None: 

                writer.writerow(flat_dict(leg)) 

    print(f"Finished creating {target}")
```

我们已经用 bz2.open(path)替换了原始的 path.open()方法。其余的上下文处理保持不变。这种灵活性允许我们在数据量增长时，最初处理文本文件，然后将其转换为压缩文件。

## 7.12.5 参见

+   在第二章的使用 with 语句管理上下文 菜单中，我们介绍了基于文件上下文管理器的使用基础。

+   创建上下文和上下文管理器 菜单涵盖了创建一个上下文管理器类的核心内容。

# 加入我们的社区 Discord 空间

加入我们的 Python Discord 工作空间，讨论并了解更多关于这本书的信息：[`packt.link/dHrHU`](https://packt.link/dHrHU)

![PIC](img/file1.png)
