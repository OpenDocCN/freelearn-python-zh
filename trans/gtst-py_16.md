# Python 中的对象

因此，我们现在手头上有一个设计，并且准备将该设计转化为一个可工作的程序！当然，通常情况下不会这样。我们将在整本书中看到好的软件设计示例和提示，但我们的重点是面向对象的编程。因此，让我们来看一下 Python 语法，它允许我们创建面向对象的软件。

完成本章后，我们将了解以下内容：

+   如何在 Python 中创建类和实例化对象

+   如何向 Python 对象添加属性和行为

+   如何将类组织成包和模块

+   如何建议人们不要破坏我们的数据

# 创建 Python 类

我们不必写太多 Python 代码就能意识到 Python 是一种非常*干净*的语言。当我们想做某事时，我们可以直接做，而不必设置一堆先决条件代码。Python 中无处不在的*hello world*，正如你可能已经看到的，只有一行。

同样，Python 3 中最简单的类如下所示：

```py
class MyFirstClass: 
    pass 
```

这是我们的第一个面向对象的程序！类定义以`class`关键字开头。然后是一个名称（我们选择的）来标识类，并以冒号结束。

类名必须遵循标准的 Python 变量命名规则（必须以字母或下划线开头，只能由字母、下划线或数字组成）。此外，Python 风格指南（在网上搜索*PEP 8*）建议使用**CapWords**表示法来命名类（以大写字母开头；任何后续的单词也应以大写字母开头）。

类定义行后面是类内容，缩进。与其他 Python 结构一样，缩进用于界定类，而不是大括号、关键字或括号，就像许多其他语言使用的那样。同样符合风格指南，除非有充分的理由不这样做（比如适应使用制表符缩进的其他人的代码），否则使用四个空格进行缩进。

由于我们的第一个类实际上并没有添加任何数据或行为，我们只需在第二行使用`pass`关键字表示不需要采取进一步的行动。

我们可能会认为这个最基本的类没有太多可以做的，但它确实允许我们实例化该类的对象。我们可以将该类加载到 Python 3 解释器中，这样我们就可以交互式地使用它。为了做到这一点，将前面提到的类定义保存在一个名为`first_class.py`的文件中，然后运行`python -i first_class.py`命令。`-i`参数告诉 Python*运行代码然后转到交互式解释器*。以下解释器会话演示了与这个类的基本交互：

```py
>>> a = MyFirstClass()
>>> b = MyFirstClass()
>>> print(a)
<__main__.MyFirstClass object at 0xb7b7faec>
>>> print(b)
<__main__.MyFirstClass object at 0xb7b7fbac>
>>>  
```

这段代码从新类实例化了两个对象，命名为`a`和`b`。创建一个类的实例只需要输入类名，后面跟着一对括号。它看起来很像一个普通的函数调用，但 Python 知道我们*调用*的是一个类而不是一个函数，所以它知道它的工作是创建一个新对象。当打印时，这两个对象告诉我们它们属于哪个类以及它们所在的内存地址。在 Python 代码中很少使用内存地址，但在这里，它们表明有两个不同的对象参与其中。

# 添加属性

现在，我们有一个基本的类，但它相当无用。它不包含任何数据，也不做任何事情。我们需要做什么来为给定的对象分配属性？

实际上，在类定义中我们不必做任何特殊的事情。我们可以使用点符号在实例化的对象上设置任意属性：

```py
class Point: 
    pass 

p1 = Point() 
p2 = Point() 

p1.x = 5 
p1.y = 4 

p2.x = 3 
p2.y = 6 

print(p1.x, p1.y) 
print(p2.x, p2.y) 
```

如果我们运行这段代码，结尾的两个`print`语句会告诉我们两个对象上的新属性值：

```py
5 4
3 6
```

这段代码创建了一个没有数据或行为的空`Point`类。然后，它创建了该类的两个实例，并分别为这些实例分配`x`和`y`坐标，以标识二维空间中的一个点。我们只需要使用`<object>.<attribute> = <value>`语法为对象的属性分配一个值。这有时被称为**点符号表示法**。在阅读标准库或第三方库提供的对象属性时，你可能已经遇到过这种表示法。值可以是任何东西：Python 原语、内置数据类型或另一个对象。甚至可以是一个函数或另一个类！

# 让它做点什么

现在，拥有属性的对象很棒，但面向对象编程实际上是关于对象之间的交互。我们感兴趣的是调用会影响这些属性的动作。我们有数据；现在是时候为我们的类添加行为了。

让我们在我们的`Point`类上建模一些动作。我们可以从一个名为`reset`的**方法**开始，它将点移动到原点（原点是`x`和`y`都为零的地方）。这是一个很好的介绍性动作，因为它不需要任何参数：

```py
class Point: 
 def reset(self): 
        self.x = 0 
        self.y = 0 

p = Point() 
p.reset() 
print(p.x, p.y) 
```

这个`print`语句显示了属性上的两个零：

```py
0 0  
```

在 Python 中，方法的格式与函数完全相同。它以`def`关键字开头，后面跟着一个空格，然后是方法的名称。然后是一组包含参数列表的括号（我们将在接下来讨论`self`参数），并以冒号结束。下一行缩进包含方法内部的语句。这些语句可以是任意的 Python 代码，对对象本身和传入的任何参数进行操作，方法会自行决定。

# 自言自语

在方法和普通函数之间的一个语法上的区别是，所有方法都有一个必需的参数。这个参数通常被命名为`self`；我从未见过 Python 程序员使用其他名称来命名这个变量（约定是一件非常有力的事情）。但是没有什么能阻止你将其命名为`this`甚至`Martha`。

方法中的`self`参数是对调用该方法的对象的引用。我们可以访问该对象的属性和方法，就好像它是另一个对象一样。这正是我们在`reset`方法中所做的，当我们设置`self`对象的`x`和`y`属性时。

在这个讨论中，注意**类**和**对象**之间的区别。我们可以将**方法**视为附加到类的函数。**self**参数是该类的特定实例。当你在两个不同的对象上调用方法时，你调用了相同的方法两次，但是将两个不同的**对象**作为**self**参数传递。

请注意，当我们调用`p.reset()`方法时，我们不必将`self`参数传递给它。Python 会自动为我们处理这部分。它知道我们在调用`p`对象上的方法，所以会自动将该对象传递给方法。

然而，方法实际上只是一个恰好在类上的函数。我们可以不在对象上调用方法，而是显式地在类上调用函数，将我们的对象作为`self`参数传递：

```py
>>> p = Point() 
>>> Point.reset(p) 
>>> print(p.x, p.y) 
```

输出与前面的例子相同，因为在内部发生了完全相同的过程。

如果我们在类定义中忘记包括`self`参数会发生什么？Python 会报错，如下所示：

```py
>>> class Point:
... def reset():
... pass
...
>>> p = Point()
>>> p.reset()
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
TypeError: reset() takes 0 positional arguments but 1 was given
```

错误消息并不像它本应该的那样清晰（嘿，傻瓜，你忘了`self`参数会更有信息量）。只要记住，当你看到指示缺少参数的错误消息时，首先要检查的是你是否在方法定义中忘记了`self`。

# 更多参数

那么，我们如何将多个参数传递给一个方法呢？让我们添加一个新的方法，允许我们将一个点移动到任意位置，而不仅仅是原点。我们还可以包括一个接受另一个`Point`对象作为输入并返回它们之间距离的方法：

```py
import math

class Point:
 def move(self, x, y):
        self.x = x
        self.y = y

    def reset(self):
        self.move(0, 0)

 def calculate_distance(self, other_point):
        return math.sqrt(
            (self.x - other_point.x) ** 2
            + (self.y - other_point.y) ** 2
        )

# how to use it:
point1 = Point()
point2 = Point()

point1.reset()
point2.move(5, 0)
print(point2.calculate_distance(point1))
assert point2.calculate_distance(point1) == point1.calculate_distance(
    point2
)
point1.move(3, 4)
print(point1.calculate_distance(point2))
print(point1.calculate_distance(point1))
```

结尾处的`print`语句给出了以下输出：

```py
5.0
4.47213595499958
0.0  
```

这里发生了很多事情。这个类现在有三个方法。`move`方法接受两个参数`x`和`y`，并在`self`对象上设置值，就像前面示例中的旧`reset`方法一样。旧的`reset`方法现在调用`move`，因为重置只是移动到一个特定的已知位置。

`calculate_distance`方法使用不太复杂的勾股定理来计算两点之间的距离。我希望你能理解这个数学（`**2`表示平方，`math.sqrt`计算平方根），但这并不是我们当前重点的要求，我们的当前重点是学习如何编写方法。

前面示例的结尾处的示例代码显示了如何调用带有参数的方法：只需将参数包含在括号内，并使用相同的点表示法来访问方法。我只是随机选择了一些位置来测试这些方法。测试代码调用每个方法并在控制台上打印结果。`assert`函数是一个简单的测试工具；如果`assert`后面的语句评估为`False`（或零、空或`None`），程序将退出。在这种情况下，我们使用它来确保无论哪个点调用另一个点的`calculate_distance`方法，距离都是相同的。

# 初始化对象

如果我们不显式设置`Point`对象上的`x`和`y`位置，要么使用`move`，要么直接访问它们，我们就会得到一个没有真实位置的破碎点。当我们尝试访问它时会发生什么呢？

好吧，让我们试试看。*试一试*是 Python 学习中非常有用的工具。打开你的交互式解释器，然后开始输入。以下交互式会话显示了如果我们尝试访问一个缺失属性会发生什么。如果你将前面的示例保存为文件，或者正在使用本书分发的示例，你可以使用`python -i more_arguments.py`命令将其加载到 Python 解释器中：

```py
>>> point = Point()
>>> point.x = 5
>>> print(point.x)
5
>>> print(point.y)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
AttributeError: 'Point' object has no attribute 'y' 
```

好吧，至少它抛出了一个有用的异常。我们将在第十八章中详细介绍异常，*预料之外的情况*。你可能以前见过它们（特别是无处不在的 SyntaxError，它意味着你输入了错误的东西！）。在这一点上，只需意识到它意味着出了问题。

输出对于调试是有用的。在交互式解释器中，它告诉我们错误发生在第 1 行，这只是部分正确的（在交互式会话中，一次只执行一行）。如果我们在文件中运行脚本，它会告诉我们确切的行号，这样很容易找到错误的代码。此外，它告诉我们错误是`AttributeError`，并给出一个有用的消息告诉我们这个错误是什么意思。

我们可以捕获并从这个错误中恢复，但在这种情况下，感觉我们应该指定某种默认值。也许每个新对象默认应该被`reset()`，或者也许当用户创建对象时，我们可以强制用户告诉我们这些位置应该是什么。

大多数面向对象的编程语言都有**构造函数**的概念，这是一个特殊的方法，用于在创建对象时创建和初始化对象。Python 有点不同；它有一个构造函数*和*一个初始化器。构造函数很少使用，除非你在做一些非常奇特的事情。所以，我们将从更常见的初始化方法开始讨论。

Python 的初始化方法与任何其他方法相同，只是它有一个特殊的名称`__init__`。前导和尾随的双下划线意味着这是一个特殊的方法，Python 解释器将把它视为一个特殊情况。

永远不要以双下划线开头和结尾命名自己的方法。它可能对 Python 今天无关紧要，但总有可能 Python 的设计者将来会添加一个具有该名称特殊目的的函数，当他们这样做时，你的代码将会出错。

让我们在我们的`Point`类上添加一个初始化函数，当实例化`Point`对象时需要用户提供`x`和`y`坐标：

```py
class Point: 
 def __init__(self, x, y): 
        self.move(x, y) 

    def move(self, x, y): 
        self.x = x 
        self.y = y 

    def reset(self): 
        self.move(0, 0) 

# Constructing a Point 
point = Point(3, 5) 
print(point.x, point.y) 
```

现在，我们的点永远不会没有`y`坐标！如果我们尝试构造一个点而没有包括正确的初始化参数，它将失败，并显示一个类似于我们之前忘记`self`参数时收到的`参数不足`错误。

如果我们不想使这两个参数成为必需的，我们可以使用与 Python 函数使用的相同语法来提供默认参数。关键字参数语法在每个变量名称后附加一个等号。如果调用对象没有提供此参数，则将使用默认参数。变量仍然可用于函数，但它们将具有参数列表中指定的值。这是一个例子：

```py
class Point: 
    def __init__(self, x=0, y=0): 
        self.move(x, y) 
```

大多数情况下，我们将初始化语句放在`__init__`函数中。但正如前面提到的，Python 除了初始化函数外还有一个构造函数。你可能永远不需要使用另一个 Python 构造函数（在十多年的专业 Python 编码中，我只想到了两种情况，在其中一种情况下，我可能不应该使用它！），但知道它的存在是有帮助的，所以我们将简要介绍一下。

构造函数被称为`__new__`，而不是`__init__`，并且只接受一个参数；正在构造的**类**（在构造对象之前调用，因此没有`self`参数）。它还必须返回新创建的对象。在涉及复杂的元编程时，这具有有趣的可能性，但在日常 Python 中并不是非常有用。实际上，你几乎永远不需要使用`__new__`。`__init__`方法几乎总是足够的。

# 自我解释

Python 是一种非常易于阅读的编程语言；有些人可能会说它是自我记录的。然而，在进行面向对象编程时，编写清楚总结每个对象和方法功能的 API 文档是很重要的。保持文档的最新状态是困难的；最好的方法是将其直接写入我们的代码中。

Python 通过使用**文档字符串**来支持这一点。每个类、函数或方法头部都可以有一个标准的 Python 字符串作为定义后面的第一行（以冒号结尾的行）。这一行应与随后的代码缩进相同。

文档字符串只是用撇号（`'`）或引号（`"`）括起来的 Python 字符串。通常，文档字符串非常长，跨越多行（风格指南建议行长不超过 80 个字符），可以格式化为多行字符串，用匹配的三个撇号（`'''`）或三引号（`"""`）字符括起来。

文档字符串应清楚而简洁地总结所描述的类或方法的目的。它应解释任何使用不明显的参数，并且还是包含如何使用 API 的简短示例的好地方。还应注意任何使用 API 的不知情用户应该注意的注意事项或问题。

为了说明文档字符串的用法，我们将以完全记录的`Point`类结束本节：

```py
import math

class Point:
    "Represents a point in two-dimensional geometric coordinates"

    def __init__(self, x=0, y=0):
        """Initialize the position of a new point. The x and y
           coordinates can be specified. If they are not, the
           point defaults to the origin."""
        self.move(x, y)

    def move(self, x, y):
        "Move the point to a new location in 2D space."
        self.x = x
        self.y = y

    def reset(self):
        "Reset the point back to the geometric origin: 0, 0"
        self.move(0, 0)

    def calculate_distance(self, other_point):
        """Calculate the distance from this point to a second
        point passed as a parameter.

        This function uses the Pythagorean Theorem to calculate
        the distance between the two points. The distance is
        returned as a float."""

        return math.sqrt(
            (self.x - other_point.x) ** 2
            + (self.y - other_point.y) ** 2
        )
```

尝试在交互式解释器中键入或加载（记住，是`python -i point.py`）这个文件。然后，在 Python 提示符下输入`help(Point)<enter>`。

你应该看到类的格式良好的文档，如下面的屏幕截图所示：

![](img/42cfc96e-6a55-47a9-aad7-b896b9f4fe59.png)

# 模块和包

现在我们知道如何创建类和实例化对象了。在开始失去追踪之前，你不需要写太多的类（或者非面向对象的代码）。对于小程序，我们可以把所有的类放在一个文件中，并在文件末尾添加一个小脚本来启动它们的交互。然而，随着项目的增长，要在我们定义的许多类中找到需要编辑的类可能会变得困难。这就是**模块**的用武之地。模块只是 Python 文件，没有别的。我们小程序中的单个文件就是一个模块。两个 Python 文件就是两个模块。如果我们有两个文件在同一个文件夹中，我们可以从一个模块中加载一个类以在另一个模块中使用。

例如，如果我们正在构建一个电子商务系统，我们可能会在数据库中存储大量数据。我们可以把所有与数据库访问相关的类和函数放在一个单独的文件中（我们将其称为一个合理的名字：`database.py`）。然后，我们的其他模块（例如，客户模型、产品信息和库存）可以导入该模块中的类以访问数据库。

`import`语句用于导入模块或特定类或函数。我们在前一节的`Point`类中已经看到了一个例子。我们使用`import`语句获取 Python 的内置`math`模块，并在`distance`计算中使用它的`sqrt`函数。

这里有一个具体的例子。假设我们有一个名为`database.py`的模块，其中包含一个名为`Database`的类。第二个名为`products.py`的模块负责与产品相关的查询。在这一点上，我们不需要太多考虑这些文件的内容。我们知道的是`products.py`需要从`database.py`中实例化`Database`类，以便它可以在数据库中的产品表上执行查询。

有几种`import`语句的变体语法可以用来访问这个类：

```py
import database 
db = database.Database() 
# Do queries on db 
```

这个版本将`database`模块导入到`products`命名空间（模块或函数中当前可访问的名称列表），因此可以使用`database.<something>`的表示法访问`database`模块中的任何类或函数。或者，我们可以使用`from...import`语法只导入我们需要的一个类：

```py
from database import Database 
db = Database() 
# Do queries on db 
```

如果由于某种原因，`products`已经有一个名为`Database`的类，我们不希望这两个名称混淆，我们可以在`products`模块中使用时重命名该类：

```py
from database import Database as DB 
db = DB() 
# Do queries on db 
```

我们也可以在一个语句中导入多个项目。如果我们的`database`模块还包含一个`Query`类，我们可以使用以下代码导入两个类：

```py
from database import Database, Query 
```

一些来源称我们可以使用以下语法从`database`模块中导入所有类和函数：

```py
from database import * 
```

**不要这样做。** 大多数有经验的 Python 程序员会告诉你，你不应该使用这种语法（有些人会告诉你有一些非常具体的情况下它是有用的，但我不同意）。他们会使用模糊的理由，比如*它会使命名空间混乱*，这对初学者来说并不太有意义。避免使用这种语法的一个方法是使用它并在两年后尝试理解你的代码。但我们可以通过一个简单的解释来节省一些时间和两年的糟糕代码！

当我们在文件顶部明确导入`database`类时，使用`from database import Database`，我们可以很容易地看到`Database`类来自哪里。我们可能会在文件的后面 400 行使用`db = Database()`，我们可以快速查看导入来看`Database`类来自哪里。然后，如果我们需要澄清如何使用`Database`类，我们可以访问原始文件（或者在交互式解释器中导入模块并使用`help(database.Database)`命令）。然而，如果我们使用`from database import *`语法，要找到该类的位置就要花费更多的时间。代码维护变成了一场噩梦。

此外，大多数代码编辑器能够提供额外的功能，比如可靠的代码补全、跳转到类的定义或内联文档，如果使用普通的导入。`import *`语法通常会完全破坏它们可靠地执行这些功能的能力。

最后，使用`import *`语法可能会将意外的对象带入我们的本地命名空间。当然，它会导入从被导入的模块中定义的所有类和函数，但它也会导入任何被导入到该文件中的类或模块！

模块中使用的每个名称都应该来自一个明确定义的地方，无论它是在该模块中定义的，还是从另一个模块中明确导入的。不应该有看起来像是凭空出现的魔术变量。我们应该*总是*能够立即确定我们当前命名空间中的名称来自哪里。我保证，如果你使用这种邪恶的语法，总有一天你会非常沮丧地发现*这个类到底是从哪里来的？*

玩一下，尝试在交互式解释器中输入`import this`。它会打印一首很好的诗（其中有一些你可以忽略的笑话），总结了一些 Python 程序员倾向于实践的习惯用法。特别是在这次讨论中，注意到了*明确胜于隐式*这一句。将名称明确导入到你的命名空间中，比隐式的`import *`语法使你的代码更容易浏览。

# 模块组织

随着项目逐渐发展成为越来越多模块的集合，我们可能会发现我们想要在模块的层次上添加另一层抽象，一种嵌套的层次结构。然而，我们不能将模块放在模块内；毕竟，一个文件只能包含一个文件，而模块只是文件。

然而，文件可以放在文件夹中，模块也可以。**包**是文件夹中模块的集合。包的名称就是文件夹的名称。我们需要告诉 Python 一个文件夹是一个包，以区别于目录中的其他文件夹。为此，在文件夹中放置一个（通常是空的）名为`__init__.py`的文件。如果我们忘记了这个文件，我们将无法从该文件夹导入模块。

让我们将我们的模块放在一个名为`ecommerce`的包中，该包还将包含一个`main.py`文件来启动程序。此外，让我们在`ecommerce`包内添加另一个用于各种支付选项的包。文件夹层次结构将如下所示：

```py
parent_directory/ 
    main.py 
    ecommerce/ 
        __init__.py 
        database.py 
        products.py 
        payments/ 
            __init__.py 
            square.py 
            stripe.py 
```

在包之间导入模块或类时，我们必须注意语法。在 Python 3 中，有两种导入模块的方式：绝对导入和相对导入。

# 绝对导入

绝对导入指定要导入的模块、函数或类的完整路径。如果我们需要访问`products`模块内的`Product`类，我们可以使用以下任何一种语法来执行绝对导入：

```py
import ecommerce.products 
product = ecommerce.products.Product() 

//or

from ecommerce.products import Product 
product = Product() 

//or

from ecommerce import products 
product = products.Product() 
```

`import`语句使用句点运算符来分隔包或模块。

这些语句将从任何模块中起作用。我们可以在`main.py`、`database`模块中或两个支付模块中的任何一个中使用这种语法实例化`Product`类。确实，假设包对 Python 可用，它将能够导入它们。例如，这些包也可以安装在 Python 站点包文件夹中，或者`PYTHONPATH`环境变量可以被定制为动态地告诉 Python 要搜索哪些文件夹以及它要导入的模块。

那么，在这些选择中，我们选择哪种语法呢？这取决于你的个人喜好和手头的应用。如果`products`模块中有数十个类和函数我想要使用，我通常使用`from ecommerce import products`语法导入模块名称，然后使用`products.Product`访问单个类。如果我只需要`products`模块中的一个或两个类，我可以直接使用`from ecommerce.products import Product`语法导入它们。我个人不经常使用第一种语法，除非我有某种名称冲突（例如，我需要访问两个完全不同的名为`products`的模块并且需要将它们分开）。做任何你认为使你的代码看起来更优雅的事情。

# 相对导入

在包内使用相关模块时，指定完整路径似乎有些多余；我们知道父模块的名称。这就是**相对导入**的用武之地。相对导入基本上是一种说法，即按照当前模块的位置来查找类、函数或模块。例如，如果我们在`products`模块中工作，并且想要从旁边的`database`模块导入`Database`类，我们可以使用相对导入：

```py
from .database import Database 
```

`database`前面的句点表示*使用当前包内的数据库模块*。在这种情况下，当前包是包含我们当前正在编辑的`products.py`文件的包，也就是`ecommerce`包。

如果我们正在编辑`ecommerce.payments`包内的`paypal`模块，我们可能会希望*使用父包内的数据库包*。这很容易通过两个句点来实现，如下所示：

```py
from ..database import Database 
```

我们可以使用更多句点来进一步上溯层次。当然，我们也可以沿着一边下去，然后沿着另一边上来。我们没有足够深的示例层次结构来正确说明这一点，但是如果我们有一个包含`email`模块并且想要将`send_mail`函数导入到我们的`paypal`模块的`ecommerce.contact`包，以下将是一个有效的导入：

```py
from ..contact.email import send_mail 
```

这个导入使用两个句点，表示*父级支付包*，然后使用正常的`package.module`语法返回到联系包。

最后，我们可以直接从包中导入代码，而不仅仅是包内的模块。在这个例子中，我们有一个名为`ecommerce`的包，其中包含两个名为`database.py`和`products.py`的模块。数据库模块包含一个`db`变量，可以从许多地方访问。如果可以像`import ecommerce.db`而不是`import ecommerce.database.db`这样导入，那不是很方便吗？

还记得`__init__.py`文件定义目录为包吗？这个文件可以包含我们喜欢的任何变量或类声明，并且它们将作为包的一部分可用。在我们的例子中，如果`ecommerce/__init__.py`文件包含以下行：

```py
from .database import db 
```

然后我们可以从`main.py`或任何其他文件中使用以下导入访问`db`属性：

```py
from ecommerce import db 
```

将`__init__.py`文件视为一个`ecommerce.py`文件可能有所帮助，如果该文件是一个模块而不是一个包。如果您将所有代码放在一个单独的模块中，然后决定将其拆分为多个模块的包，这也可能很有用。新包的`__init__.py`文件仍然可以是其他模块与其交流的主要联系点，但代码可以在几个不同的模块或子包中进行内部组织。

我建议不要在`__init__.py`文件中放太多代码。程序员不希望在这个文件中发生实际逻辑，就像`from x import *`一样，如果他们正在寻找特定代码的声明并且找不到直到他们检查`__init__.py`，它可能会让他们困惑。

# 组织模块内容

在任何一个模块内，我们可以指定变量、类或函数。它们可以是一种方便的方式来存储全局状态，而不会发生命名空间冲突。例如，我们一直在将`Database`类导入各种模块，然后实例化它，但也许更合理的是只有一个`database`对象全局可用于`database`模块。`database`模块可能是这样的：

```py
class Database: 
    # the database implementation 
    pass 

database = Database() 
```

然后我们可以使用我们讨论过的任何导入方法来访问`database`对象，例如：

```py
from ecommerce.database import database 
```

前面的类的一个问题是，`database`对象在模块第一次被导入时就被立即创建，通常是在程序启动时。这并不总是理想的，因为连接到数据库可能需要一些时间，会减慢启动速度，或者数据库连接信息可能尚未可用。我们可以通过调用`initialize_database`函数来延迟创建数据库，以创建一个模块级变量：

```py
class Database: 
    # the database implementation 
    pass 

database = None 

def initialize_database(): 
    global database 
    database = Database() 
```

`global`关键字告诉 Python，`initialize_database`内部的数据库变量是我们刚刚定义的模块级变量。如果我们没有将变量指定为全局的，Python 会创建一个新的局部变量，当方法退出时会被丢弃，从而保持模块级别的值不变。

正如这两个例子所说明的，所有模块级代码都会在导入时立即执行。但是，如果它在方法或函数内部，函数会被创建，但其内部代码直到调用函数时才会被执行。对于执行脚本（比如我们电子商务示例中的主要脚本）来说，这可能是一个棘手的问题。有时，我们编写一个执行有用操作的程序，然后后来发现我们想要从该模块导入一个函数或类到另一个程序中。然而，一旦我们导入它，模块级别的任何代码都会立即执行。如果我们不小心，我们可能会在真正只想访问该模块中的一些函数时运行第一个程序。

为了解决这个问题，我们应该总是将启动代码放在一个函数中（通常称为`main`），并且只有在知道我们正在作为脚本运行模块时才执行该函数，而不是在我们的代码被从另一个脚本导入时执行。我们可以通过在条件语句中**保护**对`main`的调用来实现这一点，如下所示：

```py
class UsefulClass:
    """This class might be useful to other modules."""

    pass

def main():
    """Creates a useful class and does something with it for our module."""
    useful = UsefulClass()
    print(useful)

if __name__ == "__main__":
    main()
```

每个模块都有一个`__name__`特殊变量（记住，Python 使用双下划线表示特殊变量，比如类的`__init__`方法），它指定了模块在导入时的名称。当模块直接用`python module.py`执行时，它不会被导入，所以`__name__`会被任意设置为`"__main__"`字符串。制定一个规则，将所有脚本都包裹在`if __name__ == "__main__":`测试中，以防万一你写了一个以后可能想被其他代码导入的函数。

那么，方法放在类中，类放在模块中，模块放在包中。这就是全部吗？

实际上，不是。这是 Python 程序中的典型顺序，但不是唯一可能的布局。类可以在任何地方定义。它们通常在模块级别定义，但也可以在函数或方法内部定义，就像这样：

```py
def format_string(string, formatter=None):
    """Format a string using the formatter object, which 
    is expected to have a format() method that accepts 
    a string."""

    class DefaultFormatter:
        """Format a string in title case."""

        def format(self, string):
            return str(string).title()

    if not formatter:
        formatter = DefaultFormatter()

    return formatter.format(string)

hello_string = "hello world, how are you today?"
print(" input: " + hello_string)
print("output: " + format_string(hello_string))
```

输出如下：

```py
 input: hello world, how are you today?
output: Hello World, How Are You Today?
```

`format_string`函数接受一个字符串和可选的格式化器对象，然后将格式化器应用于该字符串。如果没有提供格式化器，它会创建一个自己的格式化器作为本地类并实例化它。由于它是在函数范围内创建的，这个类不能从函数外部访问。同样，函数也可以在其他函数内部定义；一般来说，任何 Python 语句都可以在任何时候执行。

这些内部类和函数偶尔对于不需要或不值得在模块级别拥有自己的作用域的一次性项目是有用的，或者只在单个方法内部有意义。然而，通常不会看到频繁使用这种技术的 Python 代码。

# 谁可以访问我的数据？

大多数面向对象的编程语言都有**访问控制**的概念。这与抽象有关。对象上的一些属性和方法被标记为私有，意味着只有该对象可以访问它们。其他的被标记为受保护，意味着只有该类和任何子类才能访问。其余的是公共的，意味着任何其他对象都可以访问它们。

Python 不这样做。Python 实际上不相信强制执行可能在某一天妨碍你的法律。相反，它提供了未强制执行的指南和最佳实践。从技术上讲，类上的所有方法和属性都是公开可用的。如果我们想表明一个方法不应该公开使用，我们可以在文档字符串中放置一个注释，指出该方法仅用于内部使用（最好还要解释公共 API 的工作原理！）。

按照惯例，我们还应该使用下划线字符`_`作为内部属性或方法的前缀。Python 程序员会将其解释为*这是一个内部变量，在直接访问之前要三思*。但是，如果他们认为这样做符合他们的最佳利益，解释器内部没有任何东西可以阻止他们访问它。因为，如果他们这样认为，我们为什么要阻止他们呢？我们可能不知道我们的类将来可能被用于什么用途。

还有另一件事可以强烈建议外部对象不要访问属性或方法：用双下划线`__`作为前缀。这将对属性进行**名称混淆**。实质上，名称混淆意味着如果外部对象真的想这样做，仍然可以调用该方法，但这需要额外的工作，并且强烈表明您要求您的属性保持**私有**。以下是一个示例代码片段：

```py
class SecretString:
    """A not-at-all secure way to store a secret string."""

    def __init__(self, plain_string, pass_phrase):
 self.__plain_string = plain_string
 self.__pass_phrase = pass_phrase

    def decrypt(self, pass_phrase):
        """Only show the string if the pass_phrase is correct."""
 if pass_phrase == self.__pass_phrase:
 return self.__plain_string
        else:
            return ""
```

如果我们在交互式解释器中加载这个类并测试它，我们可以看到它将明文字符串隐藏在外部世界之外：

```py
>>> secret_string = SecretString("ACME: Top Secret", "antwerp")
>>> print(secret_string.decrypt("antwerp"))
ACME: Top Secret
>>> print(secret_string.__plain_string)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
AttributeError: 'SecretString' object has no attribute
'__plain_string'  
```

看起来好像可以了；没有人可以在没有口令的情况下访问我们的`plain_string`属性，所以应该是安全的。然而，在我们过于兴奋之前，让我们看看有多容易破解我们的安全性：

```py
>>> print(secret_string._SecretString__plain_string)
ACME: Top Secret  
```

哦不！有人发现了我们的秘密字符串。好在我们检查了。

这就是 Python 名称混淆的工作原理。当我们使用双下划线时，属性前缀为`_<classname>`。当类中的方法内部访问变量时，它们会自动取消混淆。当外部类希望访问它时，它们必须自己进行名称混淆。因此，名称混淆并不保证隐私；它只是强烈建议。除非有极其充分的理由，大多数 Python 程序员不会触碰另一个对象上的双下划线变量。

然而，大多数 Python 程序员不会在没有充分理由的情况下触碰单个下划线变量。因此，在 Python 中使用名称混淆的变量的很少有很好的理由，这样做可能会引起麻烦。例如，名称混淆的变量可能对尚未知道的子类有用，它必须自己进行混淆。如果其他对象想要访问您的隐藏信息，就让它们知道，使用单下划线前缀或一些清晰的文档字符串，表明您认为这不是一个好主意。

# 第三方库

Python 附带了一个可爱的标准库，这是一个包和模块的集合，可以在运行 Python 的每台机器上使用。然而，您很快会发现它并不包含您所需的一切。当这种情况发生时，您有两个选择：

+   自己编写一个支持包

+   使用别人的代码

我们不会详细介绍如何将您的软件包转换为库，但是如果您有需要解决的问题，而且不想编写代码（最好的程序员非常懒惰，更喜欢重用现有的经过验证的代码，而不是编写自己的代码），您可能可以在**Python 软件包索引**（**PyPI**）[`pypi.python.org/`](http://pypi.python.org/)上找到您想要的库。确定要安装的软件包后，您可以使用一个名为`pip`的工具来安装它。但是，`pip`不随 Python 一起提供，但 Python 3.4 及更高版本包含一个称为`ensurepip`的有用工具。您可以使用此命令来安装它：

```py
$python -m ensurepip  
```

这可能在 Linux、macOS 或其他 Unix 系统上失败，这种情况下，您需要成为 root 用户才能使其工作。在大多数现代 Unix 系统上，可以使用`sudo python -m ensurepip`来完成此操作。

如果您使用的 Python 版本早于 Python 3.4，您需要自己下载并安装`pip`，因为`ensurepip`不可用。您可以按照以下网址的说明进行操作：[`pip.readthedocs.org/`](http://pip.readthedocs.org/)。

一旦安装了`pip`并且知道要安装的软件包的名称，您可以使用以下语法来安装它：

```py
$pip install requests  
```

然而，如果这样做，您要么会直接将第三方库安装到系统 Python 目录中，要么更有可能会收到您没有权限这样做的错误。您可以以管理员身份强制安装，但 Python 社区的共识是，您应该只使用系统安装程序将第三方库安装到系统 Python 目录中。

相反，Python 3.4（及更高版本）提供了`venv`工具。该实用程序基本上为您的工作目录提供了一个名为*虚拟环境*的迷你 Python 安装。当您激活迷你 Python 时，与 Python 相关的命令将在该目录上运行，而不是在系统目录上运行。因此，当您运行`pip`或`python`时，它根本不会触及系统 Python。以下是如何使用它：

```py
cd project_directory
python -m venv env
source env/bin/activate  # on Linux or macOS
env/bin/activate.bat     # on Windows  
```

通常，您会为您工作的每个 Python 项目创建一个不同的虚拟环境。您可以将虚拟环境存储在任何地方，但我传统上将它们保存在与项目文件相同的目录中（但在版本控制中被忽略），因此我们首先`cd`进入该目录。然后，我们运行`venv`实用程序来创建名为`env`的虚拟环境。最后，我们使用最后两行中的一行（取决于操作系统，如注释中所示）来激活环境。每次想要使用特定的虚拟环境时，我们都需要执行此行，然后在完成该项目的工作时使用`deactivate`命令。

虚拟环境是保持第三方依赖项分开的绝佳方式。通常会有不同的项目依赖于特定库的不同版本（例如，旧网站可能在 Django 1.8 上运行，而更新的版本则在 Django 2.1 上运行）。将每个项目放在单独的虚拟环境中可以轻松地在 Django 的任一版本中工作。此外，如果您尝试使用不同的工具安装相同的软件包，它还可以防止系统安装的软件包和`pip`安装的软件包之间发生冲突。

有几种有效管理虚拟环境的第三方工具。其中一些包括`pyenv`、`virtualenvwrapper`和`conda`。我个人在撰写本文时更偏好`pyenv`，但这里没有明显的赢家。快速搜索一下，看看哪种适合您。

# 案例研究

为了将所有这些联系在一起，让我们构建一个简单的命令行笔记本应用程序。这是一个相当简单的任务，所以我们不会尝试使用多个软件包。但是，我们将看到类、函数、方法和文档字符串的常见用法。

让我们先进行快速分析：笔记是存储在笔记本中的简短备忘录。每个笔记应记录写入的日期，并可以添加标签以便轻松查询。应该可以修改笔记。我们还需要能够搜索笔记。所有这些事情都应该从命令行完成。

一个明显的对象是`Note`对象；一个不太明显的对象是`Notebook`容器对象。标签和日期似乎也是对象，但我们可以使用 Python 标准库中的日期和逗号分隔的字符串来表示标签。为了避免复杂性，在原型中，我们不需要为这些对象定义单独的类。

`Note`对象具有`memo`本身，`tags`和`creation_date`的属性。每个笔记还需要一个唯一的整数`id`，以便用户可以在菜单界面中选择它们。笔记可以有一个修改笔记内容的方法和另一个标签的方法，或者我们可以让笔记本直接访问这些属性。为了使搜索更容易，我们应该在`Note`对象上放置一个`match`方法。这个方法将接受一个字符串，并且可以告诉我们一个笔记是否与字符串匹配，而不直接访问属性。这样，如果我们想修改搜索参数（例如，搜索标签而不是笔记内容，或者使搜索不区分大小写），我们只需要在一个地方做就可以了。

`Notebook`对象显然具有笔记列表作为属性。它还需要一个搜索方法，返回一个经过筛选的笔记列表。

但是我们如何与这些对象交互？我们已经指定了一个命令行应用程序，这可能意味着我们以不同的选项运行程序来添加或编辑命令，或者我们有某种菜单，允许我们选择对笔记本做不同的事情。我们应该尽量设计它，以便支持任一接口，并且未来的接口，比如 GUI 工具包或基于 Web 的接口，可以在未来添加。

作为一个设计决策，我们现在将实现菜单界面，但会牢记命令行选项版本，以确保我们设计`Notebook`类时考虑到可扩展性。

如果我们有两个命令行界面，每个界面都与`Notebook`对象交互，那么`Notebook`将需要一些方法供这些界面与之交互。我们需要能够`add`一个新的笔记，并且通过`id`来`modify`一个现有的笔记，除了我们已经讨论过的`search`方法。界面还需要能够列出所有笔记，但它们可以通过直接访问`notes`列表属性来实现。

我们可能会错过一些细节，但我们对需要编写的代码有一个很好的概述。我们可以用一个简单的类图总结所有这些分析：

![](img/ade40d12-754a-4428-80a5-64690676d0c8.png)

在编写任何代码之前，让我们为这个项目定义文件夹结构。菜单界面应该明确地放在自己的模块中，因为它将是一个可执行脚本，并且我们将来可能会有其他可执行脚本访问笔记本。`Notebook`和`Note`对象可以放在一个模块中。这些模块可以都存在于同一个顶级目录中，而不必将它们放在一个包中。一个空的`command_option.py`模块可以帮助我们在未来提醒自己，我们计划添加新的用户界面：

```py
parent_directory/ 
    notebook.py 
    menu.py 
    command_option.py 
```

现在让我们看一些代码。我们首先定义`Note`类，因为它似乎最简单。以下示例完整呈现了`Note`。示例中的文档字符串解释了它们如何组合在一起，如下所示：

```py
import datetime

# Store the next available id for all new notes
last_id = 0

class Note:
    """Represent a note in the notebook. Match against a
    string in searches and store tags for each note."""

    def __init__(self, memo, tags=""):
        """initialize a note with memo and optional
        space-separated tags. Automatically set the note's
        creation date and a unique id."""
        self.memo = memo
        self.tags = tags
        self.creation_date = datetime.date.today()
        global last_id
        last_id += 1
        self.id = last_id

    def match(self, filter):
        """Determine if this note matches the filter
        text. Return True if it matches, False otherwise.

        Search is case sensitive and matches both text and
        tags."""
        return filter in self.memo or filter in self.tags
```

在继续之前，我们应该快速启动交互式解释器并测试我们到目前为止的代码。经常测试，因为事情从来不按照你的期望工作。事实上，当我测试这个例子的第一个版本时，我发现我在`match`函数中忘记了`self`参数！我们将在第二十四章中讨论自动化测试，*测试面向对象的程序*。目前，只需使用解释器检查一些东西就足够了：

```py
>>> from notebook import Note
>>> n1 = Note("hello first")
>>> n2 = Note("hello again")
>>> n1.id
1
>>> n2.id
2
>>> n1.match('hello')
True
>>> n2.match('second')
False  
```

看起来一切都表现如预期。让我们接下来创建我们的笔记本：

```py
class Notebook:
    """Represent a collection of notes that can be tagged,
    modified, and searched."""

    def __init__(self):
        """Initialize a notebook with an empty list."""
        self.notes = []

    def new_note(self, memo, tags=""):
        """Create a new note and add it to the list."""
        self.notes.append(Note(memo, tags))

    def modify_memo(self, note_id, memo):
        """Find the note with the given id and change its
        memo to the given value."""
        for note in self.notes:
            if note.id == note_id:
                note.memo = memo
                break

    def modify_tags(self, note_id, tags):
        """Find the note with the given id and change its
        tags to the given value."""
        for note in self.notes:
            if note.id == note_id:
                note.tags = tags
                break

    def search(self, filter):
        """Find all notes that match the given filter
        string."""
        return [note for note in self.notes if note.match(filter)]
```

我们将很快整理一下。首先，让我们测试一下以确保它能正常工作：

```py
>>> from notebook import Note, Notebook
>>> n = Notebook()
>>> n.new_note("hello world")
>>> n.new_note("hello again")
>>> n.notes
[<notebook.Note object at 0xb730a78c>, <notebook.Note object at 0xb73103ac>]
>>> n.notes[0].id
1
>>> n.notes[1].id
2
>>> n.notes[0].memo
'hello world'
>>> n.search("hello")
[<notebook.Note object at 0xb730a78c>, <notebook.Note object at 0xb73103ac>]
>>> n.search("world")
[<notebook.Note object at 0xb730a78c>]
>>> n.modify_memo(1, "hi world")
>>> n.notes[0].memo
'hi world'  
```

它确实有效。但是代码有点混乱；我们的`modify_tags`和`modify_memo`方法几乎是相同的。这不是良好的编码实践。让我们看看如何改进它。

两种方法都试图在对笔记做某事之前识别具有给定 ID 的笔记。因此，让我们添加一个方法来定位具有特定 ID 的笔记。我们将在方法名称前加下划线以表明该方法仅供内部使用，但是，当然，我们的菜单界面可以访问该方法，如果它想要的话：

```py
    def _find_note(self, note_id):
        """Locate the note with the given id."""
        for note in self.notes:
            if note.id == note_id:
                return note
        return None

    def modify_memo(self, note_id, memo):
        """Find the note with the given id and change its
        memo to the given value."""
        self._find_note(note_id).memo = memo

    def modify_tags(self, note_id, tags):
        """Find the note with the given id and change its
        tags to the given value."""
        self._find_note(note_id).tags = tags
```

现在应该可以工作了。让我们看看菜单界面。界面需要呈现菜单并允许用户输入选择。这是我们的第一次尝试：

```py
import sys
from notebook import Notebook

class Menu:
    """Display a menu and respond to choices when run."""

    def __init__(self):
        self.notebook = Notebook()
        self.choices = {
            "1": self.show_notes,
            "2": self.search_notes,
            "3": self.add_note,
            "4": self.modify_note,
            "5": self.quit,
        }

    def display_menu(self):
        print(
            """
Notebook Menu

1\. Show all Notes
2\. Search Notes
3\. Add Note
4\. Modify Note
5\. Quit
"""
        )

    def run(self):
        """Display the menu and respond to choices."""
        while True:
            self.display_menu()
            choice = input("Enter an option: ")
            action = self.choices.get(choice)
            if action:
                action()
            else:
                print("{0} is not a valid choice".format(choice))

    def show_notes(self, notes=None):
        if not notes:
            notes = self.notebook.notes
        for note in notes:
            print("{0}: {1}\n{2}".format(note.id, note.tags, note.memo))

    def search_notes(self):
        filter = input("Search for: ")
        notes = self.notebook.search(filter)
        self.show_notes(notes)

    def add_note(self):
        memo = input("Enter a memo: ")
        self.notebook.new_note(memo)
        print("Your note has been added.")

    def modify_note(self):
        id = input("Enter a note id: ")
        memo = input("Enter a memo: ")
        tags = input("Enter tags: ")
        if memo:
            self.notebook.modify_memo(id, memo)
        if tags:
            self.notebook.modify_tags(id, tags)

    def quit(self):
        print("Thank you for using your notebook today.")
        sys.exit(0)

if __name__ == "__main__":
    Menu().run()
```

这段代码首先使用绝对导入导入笔记本对象。相对导入不起作用，因为我们还没有将我们的代码放在一个包内。`Menu`类的`run`方法重复显示菜单，并通过调用笔记本上的函数来响应选择。这是使用 Python 特有的一种习惯用法；它是命令模式的一个轻量级版本，我们将在第二十二章中讨论，*Python 设计模式 I*。用户输入的选择是字符串。在菜单的`__init__`方法中，我们创建一个将字符串映射到菜单对象本身的函数的字典。然后，当用户做出选择时，我们从字典中检索对象。`action`变量实际上是指特定的方法，并且通过在变量后附加空括号（因为没有一个方法需要参数）来调用它。当然，用户可能输入了不合适的选择，所以我们在调用之前检查动作是否真的存在。

各种方法中的每一个都请求用户输入，并调用与之关联的`Notebook`对象上的适当方法。对于`search`实现，我们注意到在过滤了笔记之后，我们需要向用户显示它们，因此我们让`show_notes`函数充当双重职责；它接受一个可选的`notes`参数。如果提供了，它只显示过滤后的笔记，但如果没有提供，它会显示所有笔记。由于`notes`参数是可选的，`show_notes`仍然可以被调用而不带参数作为空菜单项。

如果我们测试这段代码，我们会发现如果我们尝试修改一个笔记，它会失败。有两个错误，即：

+   当我们输入一个不存在的笔记 ID 时，笔记本会崩溃。我们永远不应该相信用户输入正确的数据！

+   即使我们输入了正确的 ID，它也会崩溃，因为笔记 ID 是整数，但我们的菜单传递的是字符串。

后一个错误可以通过修改`Notebook`类的`_find_note`方法，使用字符串而不是存储在笔记中的整数来比较值来解决，如下所示：

```py
    def _find_note(self, note_id):
        """Locate the note with the given id."""
        for note in self.notes:
            if str(note.id) == str(note_id):
                return note
        return None
```

在比较它们之前，我们只需将输入（`note_id`）和笔记的 ID 都转换为字符串。我们也可以将输入转换为整数，但是如果用户输入字母`a`而不是数字`1`，那么我们会遇到麻烦。

用户输入不存在的笔记 ID 的问题可以通过更改笔记本上的两个`modify`方法来解决，检查`_find_note`是否返回了一个笔记，如下所示：

```py
    def modify_memo(self, note_id, memo):
        """Find the note with the given id and change its
        memo to the given value."""
        note = self._find_note(note_id)
        if note:
            note.memo = memo
            return True
        return False
```

这个方法已更新为返回`True`或`False`，取决于是否找到了一个笔记。菜单可以使用这个返回值来显示错误，如果用户输入了一个无效的笔记。

这段代码有点笨拙。如果它引发异常会好一些。我们将在第十八章中介绍这些，*预料之外*。

# 练习

编写一些面向对象的代码。目标是使用本章学到的原则和语法，确保你理解我们所涵盖的主题。如果你一直在做一个 Python 项目，回过头来看看，是否有一些对象可以创建，并添加属性或方法。如果项目很大，尝试将其分成几个模块，甚至包，并玩弄语法。

如果你没有这样的项目，尝试开始一个新的项目。它不一定要是你打算完成的东西；只需勾勒出一些基本的设计部分。你不需要完全实现所有内容；通常，只需要`print("这个方法将做一些事情")`就足以让整体设计就位。这被称为**自顶向下设计**，在这种设计中，你先解决不同的交互，并描述它们应该如何工作，然后再实际实现它们所做的事情。相反，**自底向上设计**首先实现细节，然后将它们全部联系在一起。这两种模式在不同的时候都很有用，但对于理解面向对象的原则，自顶向下的工作流更合适。

如果你在想法上遇到困难，可以尝试编写一个待办事项应用程序。（提示：它将类似于笔记本应用程序的设计，但具有额外的日期管理方法。）它可以跟踪你每天想做的事情，并允许你标记它们为已完成。

现在尝试设计一个更大的项目。与之前一样，它不一定要真正做任何事情，但确保你尝试使用包和模块导入语法。在各个模块中添加一些函数，并尝试从其他模块和包中导入它们。使用相对和绝对导入。看看它们之间的区别，并尝试想象你想要使用每种导入方式的场景。

# 总结

在本章中，我们学习了在 Python 中创建类并分配属性和方法是多么简单。与许多语言不同，Python 区分构造函数和初始化程序。它对访问控制有一种放松的态度。有许多不同级别的作用域，包括包、模块、类和函数。我们理解了相对导入和绝对导入之间的区别，以及如何管理不随 Python 一起提供的第三方包。

在下一章中，我们将学习如何使用继承来共享实现。
