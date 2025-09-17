

# 第四章：使用类组织我们的代码

你的数据输入表单进展顺利！你的老板和同事对你的进步感到兴奋，并且已经开始提出一些可以添加的其他功能的想法。说实话，这让你有点紧张！虽然他们看到一个看起来专业的表单，但你清楚底下的代码正变得越来越庞大和重复。你还有一些瑕疵，比如全局变量和一个非常杂乱的全球命名空间。在你开始添加更多功能之前，你希望掌握这段代码，并开始将其分解成一些可管理的块。为此，你需要创建**类**。

在本章中，我们将涵盖以下主题：

+   在 *Python 类的入门指南* 中，我们将回顾如何创建 Python 类和子类。

+   在 *使用 Tkinter 的类* 中，我们将发现如何有效地在 Tkinter 代码中使用类。

+   在 *使用类重写我们的应用程序* 中，我们将将这些技术应用到 ABQ 数据输入应用程序中。

# Python 类的入门指南

虽然类的基本概念表面上看起来很简单，但类带来了一系列术语和概念，这些概念常常让许多初学者感到困惑。在本节中，我们将讨论使用类的优点，探讨类的不同特性，并回顾在 Python 中创建类的语法。

## 使用类的优点

许多初学者甚至中级 Python 程序员避免或忽视 Python 中类的使用；与函数或变量不同，类在简短的简单脚本中没有明显的用途。然而，随着我们的应用程序代码的增长，类成为组织我们的代码成可管理单元的不可或缺的工具。让我们看看类如何帮助我们构建更干净的代码。

### 类是 Python 的组成部分

类本质上是一个创建**对象**的蓝图。什么是对象？在 Python 中，*一切*都是对象：整数、字符串、浮点数、列表、字典、Tkinter 小部件，甚至函数都是对象。这些对象类型都是由类定义的。如果你在 Python 提示符下使用`type`命令，就可以很容易地看到这一点，如下所示：

```py
>>> type('hello world')
<class 'str'>
>>> type(1)
<class 'int'>
>>> type(print)
<class 'builtin_function_or_method'> 
```

`type`函数显示了你用来构建特定对象的类。当一个对象由特定的类构建时，我们称它为该类的**实例**。

*实例*和*对象*经常可以互换使用，因为每个对象都是某个类的实例。

因为 Python 中的所有东西都是类，所以创建我们自己的类允许我们使用与内置对象相同的语法来处理自定义对象。

### 类使数据与函数之间的关系明确

通常，在代码中，我们有一组所有相关联的数据。例如，在一个多人游戏中，你可能会有每个玩家的分数、健康或进度等变量。操作这些变量的函数需要确保操作的是指向同一玩家的变量。类允许我们创建这些变量和操作它们的函数之间的显式关系，这样我们就可以更容易地将它们作为一个单元组织起来。

### 类有助于创建可重用的代码

类是减少代码冗余的强大工具。假设我们有一组在提交时具有相似行为但输入字段不同的表单。使用类继承，我们可以创建一个具有所需共同行为的基表单；然后，我们可以从这个基表单派生出单个表单类，只需要实现每个表单中独特的内容。

## 类创建的语法

创建一个类与创建一个函数非常相似，只是我们使用`class`关键字，如下所示：

```py
class Banana:
  """A tasty tropical fruit"""
  pass 
```

注意，我们还包含了一个**文档字符串**，它被 Python 工具（如内置的`help`函数）用于生成有关类的文档。在 Python 中，类名传统上使用**帕斯卡大小写**，即每个单词的首字母大写；有时，第三方库会使用其他约定，但通常不会。

一旦我们定义了一个类，我们就可以通过调用它来创建类的实例，就像调用一个函数一样：

```py
my_banana = Banana() 
```

在这种情况下，`my_banana`是一个`Banana`类的实例对象。当然，更有用的类将在类体内部定义一些内容；具体来说，我们可以定义**属性**和**方法**，这些统称为**成员**。

### 属性和方法

属性仅仅是变量，它们可以是**类属性**或**实例属性**。类属性是在类体顶部的范围内定义的，如下所示：

```py
class Banana:
  """A tasty tropical fruit"""
  food_group = 'fruit'
  colors = [
    'green', 'green-yellow', 'yellow',
    'brown spotted', 'black'
  ] 
```

类属性被类的所有实例共享，通常用于设置默认值、常量和其他只读值。

注意，与类名不同，成员名称按照惯例使用**蛇形命名法**，即单词之间用下划线分隔。

实例属性存储特定于类单个实例的值；要创建一个实例属性，我们需要访问一个实例。我们可以这样做：

```py
my_banana = Banana()
my_banana.color = 'yellow' 
```

然而，如果我们能在类定义内部定义一些实例属性，而不是像那样外部定义，那将更加理想。为了做到这一点，我们需要在类定义内部对类的实例有一个引用。这可以通过一个**实例方法**来实现。

方法只是附加到类上的函数。实例方法是一种自动接收实例引用作为其第一个参数的方法。我们可以这样定义一个：

```py
class Banana:
  def peel(self):
    self.peeled = True 
```

正如你所见，定义一个实例方法就是简单地在类体内定义一个函数。这个函数将接收的第一个参数是类的实例的引用；你可以称它为你喜欢的任何名字，但根据长期以来的 Python 约定，我们称它为`self`。在函数内部，`self`可以用来对实例进行操作，例如分配实例属性。

注意，实例（`self`）也可以访问类属性（例如，`self.colors`），如下所示：

```py
 def set_color(self, color):
    """Set the color of the banana"""
    if color in self.colors:
      self.color = color
    else:
      raise ValueError(f'A banana cannot be {color}!') 
```

当我们使用实例方法时，我们不显式传递`self`；它是隐式传递的，如下所示：

```py
my_banana = Banana()
my_banana.set_color('green')
my_banana.peel() 
```

`self`的隐式传递通常会在传递错误数量的参数时导致令人困惑的错误信息。例如，如果你调用`my_banana.peel(True)`，你会得到一个异常，表明期望一个参数但传递了两个。从你的角度来看，你只传递了一个参数，但方法接收到了两个，因为实例引用被自动添加了。

除了实例方法之外，类还可以有**类方法**和**静态方法**。与实例方法不同，这些方法没有访问类的实例，并且不能读取或写入实例属性。

类方法是在方法定义之前使用**装饰器**创建的，如下所示：

```py
 **@classmethod**
  def check_color(cls, color):
    """Test a color string to see if it is valid."""
    return color in cls.colors
 **@classmethod**
  def make_greenie(cls):
    """Create a green banana object"""
    banana = cls()
    banana.set_color('green')
    return banana 
```

正如实例方法隐式地传递了实例的引用一样，类方法也隐式地传递了类的引用作为第一个参数。同样，你可以称那个参数为你喜欢的任何名字，但传统上我们称之为`cls`。类方法通常用于与类变量交互。例如，在上面的`check_color()`方法中，该方法需要引用类变量`colors`。类方法也用作生成特定配置的类实例的便利函数；例如，上面的`make_greenie()`方法使用其类引用创建颜色预设置为`green`的`Banana`实例。

**静态方法**也是一个附加到类上的函数，但它不接收任何隐式参数，方法内的代码也无法访问类或实例。就像类方法一样，我们使用装饰器来定义静态方法，如下所示：

```py
 **@staticmethod**
  def estimate_calories(num_bananas):
    """Given `num_bananas`, estimate the number of calories"""
    return num_bananas * 105 
```

静态方法通常用于定义类内部使用的算法或实用函数。

类和静态方法可以直接在类本身上调用；例如，我们可以调用`Banana.estimate_calories()`或`Banana.check_color()`而不实际创建`Banana`的实例。然而，实例方法*必须*在类的实例上调用。调用`Banana.set_color()`或`Banana.peel()`是没有意义的，因为这些方法旨在操作实例。相反，我们应该创建一个实例，并在其上调用那些方法（例如，`my_banana.peel()`）。

### 魔法属性和方法

所有 Python 对象都自动获得一组称为**魔法属性**的属性和一组称为**魔法方法**的方法，也称为特殊方法或*dunder 方法*，因为它们由属性或方法名周围的两个下划线指示（“dunder”是“double under”的混合词）。

魔法属性通常存储关于对象的元数据。例如，任何对象的`__class__`属性存储了对对象类的引用：

```py
>>> print(my_banana.__class__)
<class '__main__.Banana'> 
```

魔法方法定义了 Python 对象如何响应运算符（如`+`、`%`或`[]`）或内置函数（如`dir()`或`setattr()`）。例如，`__str__()`方法定义了当对象传递给`str()`函数（无论是显式还是隐式，例如通过传递给`print()`）时返回的内容：

```py
class Banana:
  # ....
  def __str__(self):
    # "Magic Attributes" contain metadata about the object
    return f'A {self.color} {self.__class__.__name__}' 
```

在这里，我们不仅访问实例的`color`属性，还使用`__class__`属性来检索其类，然后使用类对象的`__name__`属性来获取类名。

尽管它很令人困惑，但**类**也是一个**对象**。它是`type`类的一个实例。记住，Python 中的所有东西都是对象，所有对象都是某个类的实例。

因此，当打印`Banana`对象时，它看起来是这样的：

```py
>>> my_banana = Banana()
>>> my_banana.set_color('yellow')
>>> print(my_banana)
A yellow Banana 
```

到目前为止，最重要的魔法方法是**初始化器**方法，`__init__()`。每当调用类对象以创建实例时，该方法都会执行，我们为其定义的参数成为创建实例时可以传递的参数。例如：

```py
 def __init__(self, color='green'):
    if not self.check_color(color):
      raise ValueError(
        f'A {self.__class__.__name__} cannot be {color}'
      )
    self.color = color 
```

在这里，我们创建了一个带有可选参数`color`的初始化器，允许我们在创建对象时设置`Banana`对象的颜色值。因此，我们可以这样创建一个新的`Banana`：

```py
>>> my_new_banana = Banana('green')
>>> print(my_new_banana)
A green Banana 
```

理想情况下，任何在类中使用的实例属性都应该在`__init__()`中创建，这样我们就可以确保它们对于类的所有实例都存在。例如，我们应该这样创建我们的`peeled`属性：

```py
 def __init__(self, color='green'):
    # ...
    **self.peeled =** **False** 
```

如果我们没有在这里定义这个属性，它将不会存在，直到调用`peel()`方法。在调用该方法之前寻找`my_banana.peel`值的代码将引发异常。

最终，初始化器应该使对象处于一个程序可以使用它的状态。

在其他面向对象的语言中，设置类对象的那个方法被称为**构造函数**，它不仅初始化新对象，还返回它。有时，Python 开发者会随意将`__init__()`称为构造函数。然而，Python 对象的实际构造方法为`__new__()`，我们通常在 Python 类中保持其不变。

### 公共、私有和受保护的成员

类是用于**抽象**的强大工具——也就是说，将复杂对象或过程简化为一个简单、高级的接口，供应用程序的其他部分使用。为了帮助它们做到这一点，Python 程序员使用一些命名约定来区分公共、私有和受保护的成员：

+   **公共成员**是那些打算由类外部的代码读取或调用的成员。它们使用普通的成员名称。

+   **受保护成员**仅用于类内部或其子类。它们以单个下划线为前缀。

+   **私有成员**仅用于类内部。它们以双下划线为前缀。

Python 实际上并不强制区分公共、受保护和私有成员；这些只是其他程序员理解并用来指示哪些部分可以外部访问，哪些是类内部实现的一部分且不打算在类外使用的规定。

Python **会**通过自动将它们的名称更改为 `_classname__member_name` 来协助强制私有成员。

例如，让我们将以下代码添加到 `Banana` 类中：

```py
 __ripe_colors = ['yellow', 'brown spotted']
  def _is_ripe(self):
    """Protected method to see if the banana is ripe."""
    return self.color in self.__ripe_colors
  def can_eat(self, must_be_ripe=False):
    """Check if I can eat the banana."""
    if must_be_ripe and not self._is_ripe():
      return False
    return True 
```

在这里，`__ripe_colors` 是一个私有属性。如果你尝试访问 `my_banana.__ripe_colors`，Python 会抛出一个 `AttributeError` 异常，因为它隐式地将这个属性重命名为 `my_banana._Banana__ripe_colors`。方法 `_is_ripe()` 是一个受保护成员，但与私有成员不同，Python 不会更改它的名称。它可以作为 `my_banana._is_ripe()` 执行，但使用你的类的程序员会理解这个方法是为了内部使用，而不是在外部代码中依赖。相反，应该调用公共的 `can_eat()` 方法。

有许多原因会让你想要将成员标记为私有或受保护，但通常是因为该成员是某些内部过程的一部分，对于外部代码的使用可能是无意义、不可靠或缺乏上下文。

虽然单词“私有”和“受保护”似乎表明了一种安全特性，但这并不是它们的意图，使用它们也不会为类提供任何安全性。意图仅仅是区分类的公共接口（外部代码应该使用）和类的内部机制（应该保持不变）。

## 继承和子类

构建我们自己的类确实是一个强大的工具，但鉴于 Python 中的一切都是从一个类构建的对象，如果我们能够简单地修改其中一个现有类以适应我们的需求，那岂不是很好？这样我们就不必每次都从头开始了。

幸运的是，我们可以！当我们创建一个类时，Python 允许我们从现有类中派生它，如下所示：

```py
class RedBanana(Banana):
  """Bananas of the red variety"""
  pass 
```

我们创建了 `RedBanana` 类作为 `Banana` 的 **子类** 或 **派生类**。在这种情况下，`Banana` 被称为 **父类** 或 **超类**。最初，`RedBanana` 是 `Banana` 的一个精确副本，将表现得完全相同，但我们可以通过简单地定义成员来修改它，如下所示：

```py
class RedBanana(Banana):
  colors = ['green', 'orange', 'red', 'brown', 'black']
  botanical_name = 'red dacca'
  def set_color(self, color):
    if color not in self.colors:
      raise ValueError(f'A Red Banana cannot be {color}!') 
```

指定现有成员，如`colors`和`set_color`，将掩盖超类中这些成员的版本。因此，在`RedBanana`实例上调用`set_color()`将调用`RedBanana`版本的方法，然后，当引用`self.colors`时，它将咨询`RedBanana`版本的`colors`。我们还可以添加新成员，如`botanical_name`属性，它将仅在子类中存在。

在某些情况下，我们可能希望子类方法添加到超类方法中，但仍然执行超类方法版本中的代码。我们可以将超类代码复制到子类代码中，但有一个更好的方法：使用`super()`。

在实例方法内部，`super()`给我们提供了对超类版本实例的引用，如下所示：

```py
 def peel(self):
    super().peel()
    print('It looks like a regular banana inside!') 
```

在这种情况下，调用`super().peel()`会导致在`RedBanana`实例上执行`Banana.peel()`中的代码。然后，我们可以在子类版本的`peel()`中添加额外的代码。

正如你将在下一节中看到的，`super()`通常在`__init__()`方法中使用，以运行超类初始化器。这对于 Tkinter GUI 类来说尤其正确，因为它们在初始化方法中执行了很多关键的外部设置。

关于 Python 类，我们在这里讨论的还有很多，包括**多重继承**的概念，我们将在第五章“通过验证和自动化减少用户错误”中学习到。然而，到目前为止我们所学的已经足够我们应用到 Tkinter 代码中。让我们看看类如何在 GUI 环境中帮助我们。

# 使用 Tkinter 中的类

GUI 框架和面向对象代码是相辅相成的。虽然 Tkinter 比大多数框架都更允许你使用过程式编程创建 GUI，但这样做我们会失去很多组织上的优势。尽管在这本书的整个过程中，我们会找到许多在 Tkinter 代码中使用类的方法，但在这里我们将探讨三种主要的类使用方法：

+   提升或扩展 Tkinter 类以获得更多功能

+   创建**复合小部件**以节省重复输入

+   将我们的应用程序组织成自包含的**组件**

## 提升 Tkinter 类

让我们面对现实：一些 Tkinter 对象在功能上有些不足。我们可以通过子类化 Tkinter 类并创建自己的改进版本来解决这个问题。例如，虽然我们已经看到 Tkinter 控制变量类很有用，但它们仅限于字符串、整数、双精度和布尔类型。如果我们想要这些变量的功能，但针对更复杂的对象，如字典或列表呢？我们可以通过子类化和一些 JSON 的帮助来实现。

**JavaScript 对象表示法**（**JSON**）是一种标准化的格式，用于将列表、字典和其他复合对象表示为字符串。Python 标准库自带了一个`json`库，它允许我们将这些对象转换为字符串格式，然后再转换回来。我们将在第七章“使用菜单和 Tkinter 对话框创建菜单”中更多地使用 JSON。

打开一个名为 `tkinter_classes_demo.py` 的新脚本，让我们从一些导入开始，如下所示：

```py
# tkinter_classes_demo.py
import tkinter as tk
import json 
```

除了 Tkinter，我们还导入了标准库中的 `json` 模块。此模块包含两个函数，我们将使用它们来实现我们的变量：

+   `json.dumps()` 接收一个 Python 对象，如列表、字典、字符串、整数或浮点数，并返回一个 JSON 格式的字符串。

+   `json.loads()` 接收一个 JSON 字符串并返回一个 Python 对象，如列表、字典或字符串，具体取决于 JSON 字符串中存储的内容。

通过创建一个名为 `JSONVar` 的 `tk.StringVar` 子类来开始新的变量类：

```py
class JSONVar(tk.StringVar):
  """A Tk variable that can hold dicts and lists""" 
```

为了使我们的 `JSONVar` 正常工作，我们需要拦截任何传递给对象的 `value` 参数，并使用 `json.dumps()` 方法将其转换为 JSON 字符串。第一个需要拦截 `value` 参数的地方是在 `__init__()` 方法中，我们将像这样覆盖它：

```py
 def __init__(self, *args, **kwargs):
    kwargs['value'] = json.dumps(kwargs.get('value')
    super().__init__(*args, **kwargs) 
```

在这里，我们只是从关键字参数中检索 `value` 参数，并使用 `json.dumps()` 将其转换为字符串。转换后的字符串将覆盖 `value` 参数，然后将其传递给超类初始化器。如果未提供 `value` 参数（记住，它是一个可选参数），`kwargs.get()` 将返回 `None`，这将被转换为 JSON `null` 值。

在覆盖你未编写的类中的方法时，始终包含 `*args` 和 `**kwargs` 是一个好主意，以捕获任何未明确列出的参数。这样，该方法将继续允许所有与超类版本相同的参数，但你不必明确列出它们。

我们需要拦截 `value` 参数的下一个地方是在 `set()` 方法中，如下所示：

```py
 def set(self, value, *args, **kwargs):
    string = json.dumps(value)
    super().set(string, *args, **kwargs) 
```

我们再次拦截了 `value` 参数，并在将其传递给超类版本的 `set()` 方法之前将其转换为 JSON 字符串。

最后，让我们修复 `get()` 方法：

```py
 def get(self, *args, **kwargs):
    string = super().get(*args, **kwargs)
    return json.loads(string) 
```

在这里，我们做了与前两种方法相反的操作：首先，我们从超类中获取了字符串，然后使用 `json.loads()` 将其转换回对象。完成这些操作后，我们就准备好了！我们现在有一个可以存储和检索列表或字典的变量，就像任何其他 Tkinter 变量一样。

让我们测试一下：

```py
root = tk.Tk()
var1 = JSONVar(root)
var1.set([1, 2, 3])
var2 = JSONVar(root, value={'a': 10, 'b': 15})
print("Var1: ", var1.get()[1])
# Should print 2
print("Var2: ", var2.get()['b'])
# Should print 15 
```

如您所见，子类化 Tkinter 对象为我们代码打开了全新的可能性。我们将在本章后面以及在第五章 *通过验证和自动化减少用户错误* 中更广泛地应用这一概念。不过，首先，让我们看看我们还可以用类与 Tkinter 代码结合的两种方法。

## 创建复合小部件

许多 GUI（尤其是数据输入表单）包含需要大量重复模板代码的模式。例如，输入小部件通常有一个伴随的标签来告诉用户他们需要输入什么。这通常需要几行代码来创建和配置每个对象并将它们添加到表单中。我们不仅可以节省时间，而且通过创建一个可重用的**复合小部件**将两者结合成一个单一类，还可以确保输出的一致性。

通过创建一个`LabelInput`类来组合输入小部件和标签，开始如下：

```py
# tkinter_classes_demo.py
class LabelInput(tk.Frame):
  """A label and input combined together""" 
```

`tk.Frame`小部件，一个没有任何内容的基础小部件，是一个理想的类，可以用来创建复合小部件。在开始我们的类定义之后，接下来我们需要做的是思考我们的小部件将需要哪些数据，并确保这些数据可以通过`__init__()`方法传递。

对于一个基本小部件，可能的最小参数集看起来可能如下所示：

+   父小部件

+   标签的文本

+   要使用的输入小部件类型

+   要传递给输入小部件的参数字典

让我们在`LabelInput`类中实现这一点：

```py
 def __init__(
    self, parent, label, inp_cls, 
    inp_args, *args, **kwargs
  ):
    super().__init__(parent, *args, **kwargs)
    self.label = tk.Label(self, text=label, anchor='w')
    self.input = inp_cls(self, **inp_args) 
```

在这里我们首先调用超类初始化器，以便构建`Frame`小部件。请注意，我们传递了`parent`参数，因为这将成为`Frame`本身的父小部件；`Label`和输入小部件的父小部件是`self`——即`LabelInput`对象本身。

不要混淆“父类”和“父小部件”。“父类”指的是我们的子类从中继承其成员的超类。“父小部件”指的是我们的小部件（可能属于一个无关的类）所附加的小部件。为了避免混淆，当我们在本书中讨论类继承时，我们将坚持使用超/子类术语。

在创建我们的`label`和`input`小部件之后，我们可以根据需要将它们排列在`Frame`上；例如，我们可能希望标签位于输入旁边，如下所示：

```py
 self.columnconfigure(1, weight=1)
    self.label.grid(sticky=tk.E + tk.W)
    self.input.grid(row=0, column=1, sticky=tk.E + tk.W) 
```

或者，我们可能更喜欢在输入小部件上方使用标签，如下所示：

```py
 self.columnconfigure(0, weight=1)
    self.label.grid(sticky=tk.E + tk.W)
    self.input.grid(sticky=tk.E + tk.W) 
```

在任何情况下，如果我们使用`LabelInput`创建表单上的所有输入，我们就有能力仅用三行代码来改变整个表单的布局。我们还可以考虑添加一个初始化器参数，以便为每个实例单独配置布局。

让我们看看这个类在实际中的应用。由于我们的`inp_args`参数将被直接扩展到对`inp_cls`初始化器的调用中，我们可以填充任何我们希望输入小部件接收到的参数，如下所示：

```py
# tkinter_classes_demo.py
li1 = LabelInput(root, 'Name', tk.Entry, {'bg': 'red'})
li1.grid() 
```

我们甚至可以将一个变量传递给绑定到小部件：

```py
age_var = tk.IntVar(root, value=21)
li2 = LabelInput(
  root, 'Age', tk.Spinbox,
  {'textvariable': age_var, 'from_': 10, 'to': 150}
)
li2.grid() 
```

复合小部件为我们节省了一些代码，但更重要的是，它将我们的输入表单代码提升到对正在发生的事情的更高层次的描述。我们不必关注每个标签相对于每个小部件的放置细节，我们可以从这些更大的组件的角度来考虑表单。

## 构建封装的组件

创建复合小部件对于我们在应用程序中计划重用的结构很有用，但同样的概念可以有益地应用于我们应用程序的更大块，即使它们只出现一次。

这样做使我们能够将方法附加到应用程序的组件上，以构建功能自包含的单元，这些单元更容易管理。

例如，让我们创建一个`MyForm`类来保存一个简单的表单：

```py
# tkinter_classes_demo.py
class MyForm(tk.Frame):
  def __init__(self, parent, data_var, *args, **kwargs):
    super().__init__(parent, *args, **kwargs)
    self.data_var = data_var 
```

正如我们处理复合小部件一样，我们已经从`tk.Frame`派生子类并定义了一个新的初始化方法。`parent`、`*args`和`**kwargs`参数将被传递给超类初始化器，但我们还将接受一个`data_var`参数，它将是我们新`JSONVar`类型的实例。我们将使用此参数将表单数据传回表单。

接下来，我们将创建一些内部控制变量以绑定到我们的表单小部件：

```py
 self._vars = {
      'name': tk.StringVar(self),
      'age': tk.IntVar(self, value=2)
    } 
```

正如我们在数据输入应用程序中已经看到的，将表单数据变量保存在字典中将使以后从它们中提取数据变得简单。然而，我们不是使用全局变量，而是通过将其添加到`self`并使用下划线作为前缀来创建字典作为受保护的实例变量。这是因为这个字典仅用于我们表单的内部使用。

现在，让我们使用我们的`LabelInput`类来创建表单的实际小部件：

```py
 LabelInput(
      self, 'Name', tk.Entry,
      {'textvariable': self._vars['name']}
    ).grid(sticky=tk.E + tk.W)
    LabelInput(
      self, 'Age', tk.Spinbox,
      {'textvariable': self._vars['age'], 'from_': 10, 'to': 150}
    ).grid(sticky=tk.E + tk.W) 
```

你可以看到`LabelInput`大大减少了我们的 GUI 构建代码！现在，让我们为我们的表单添加一个提交按钮：

```py
 tk.Button(self, text='Submit', command=self._on_submit).grid() 
```

提交按钮被配置为调用名为`_on_submit`的保护实例方法。这展示了使用类为我们 GUI 组件提供的一个强大功能：通过将按钮绑定到实例方法，该方法将能够访问所有其他实例成员。例如，它可以访问我们的`_vars`字典：

```py
 def _on_submit(self):
    data = { key: var.get() for key, var in self._vars.items() }
    self.data_var.set(data) 
```

如果不使用类，我们就必须依赖于全局变量，就像我们在*第三章*中编写的`data_entry_app.py`应用程序中所做的那样。相反，我们的回调方法只需要隐式传递的`self`对象来访问它需要的所有对象。在这种情况下，我们使用**字典推导**从我们的小部件中提取所有数据，然后将结果字典存储在我们的`JSONVar`对象中。

字典推导类似于列表推导，但它创建一个字典；语法是`{ key: value for expression in iterator }`。例如，如果您想创建一个包含数字及其平方的字典，您可以编写`{ n: n**2 for n in range(100) }`。

因此，每当点击提交按钮时，`data_var`对象将使用当前输入小部件的内容进行更新。

### 派生 Tk

我们可以将组件构建的概念扩展到顶级窗口`Tk`对象。通过从`Tk`派生子类并在它们自己的类中构建其他应用程序组件，我们可以以高级方式组合应用程序的布局和行为。

让我们用我们当前的演示脚本试一试：

```py
# tkinter_classes_demo.py
class Application(tk.Tk):
  """A simple form application"""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs) 
```

记住，`Tk`对象不仅是我们顶级窗口，还代表了我们应用程序的核心。因此，我们将其子类命名为`Application`，以表明它代表了我们整个应用程序的基础。我们的初始化方法以必要的调用`super().__init__()`开始，并将任何参数传递给`Application.__init__()`方法。

接下来，我们将创建一些变量来跟踪我们应用程序中的数据：

```py
 self.jsonvar = JSONVar(self)
    self.output_var = tk.StringVar(self) 
```

如你所预期，`JSONVar`将被传递到我们的`MyForm`对象中，以处理其数据。`output_var`只是一个`StringVar`，我们将用它来显示一些输出。接下来，让我们向我们的窗口添加一些小部件：

```py
 tk.Label(self, text='Please fill the form').grid(sticky='ew')
    MyForm(self, self.jsonvar).grid(sticky='nsew')
    tk.Label(self, textvariable=self.output_var).grid(sticky='ew')
    self.columnconfigure(0, weight=1)
    self.rowconfigure(1, weight=1) 
```

在这里，我们为表单添加了一个简单的标题标签，一个`MyForm`对象，以及另一个用于显示输出的标签。我们还配置了框架，使得第一列（也是唯一的一列）可以扩展到额外空间，第二行（包含表单的那一行）可以扩展到额外的垂直空间。

由于`MyForm`的提交会更新我们传递给它的`JSONVar`对象，我们需要一种方法在变量内容更改时执行提交处理回调。我们可以通过在`jsonvar`上设置一个**跟踪**来实现这一点：

```py
 self.jsonvar.trace_add('write', self._on_data_change) 
```

`trace_add()`方法可以用在任何一个 Tkinter 变量（或变量子类）上，在变量相关事件发生时执行回调函数。让我们花点时间更详细地考察它。

`trace_add()`的第一个参数指定了触发跟踪的事件；它可以是以下之一：

+   `read`：变量的值被读取（例如通过`get()`调用）。

+   `write`：变量的值被修改（例如通过`set()`调用）。

+   `unset`：删除变量。

+   `array`：这是 Tcl/Tk 的一个遗迹，在 Python 中并不真正有意义，但仍然是有效的语法。你很可能永远不会用到它。

第二个参数指定了事件的回调函数，在这种情况下，是实例方法`_on_data_change()`，它将在`jsonvar`更新时被触发。我们将这样处理它：

```py
 def _on_data_change(self, *args, **kwargs):
    data = self.jsonvar.get()
    output = ''.join([
    f'{key} = {value}\n'
    for key, value in data.items()
    ])
    self.output_var.set(output) 
```

这个方法简单地遍历从`jsonvar`检索到的字典中的值，然后将它们组合成一个格式化的字符串。最后，将格式化的字符串传递给`output_var`，这将更新主窗口底部的标签以显示我们的表单值。在实际应用程序中，你可能会将检索到的数据保存到文件中，或者将它们用作批量操作的参数，例如。

在实例方法中，何时应该使用实例变量（例如，`self.jsonvar`），何时应该使用常规变量（例如，`data`）？方法中的常规变量在其作用域内是 **局部** 的，这意味着一旦方法返回，它们就会被销毁。此外，它们不能被类中的其他方法引用。实例变量在其实例本身的整个生命周期内保持作用域，并且可供任何其他实例方法读取或写入。在 `Application` 类的情况下，`data` 变量仅在 `_on_data_change()` 方法内部需要，而 `jsonvar` 需要在 `__init__()` 和 `_on_datachange()` 中访问。

由于我们已经从 `Tk` 继承了子类，我们不应再以 `root = tk.Tk()` 这行开始我们的脚本。请确保删除该行，以及删除引用 `root` 的代码的上一行。相反，我们将这样执行我们的应用程序：

```py
if __name__ == "__main__":
  app = Application()
  app.mainloop() 
```

注意，这些行、我们的类定义和我们的导入是我们唯一执行的最高级代码。这大大清理了我们的全局作用域，将代码的更详细细节限制在一个更小的范围内。

在 Python 中，`if __name__ == "__main__":` 是一个常见的惯用语，用于检查脚本是否被直接运行，例如，当我们在一个命令提示符中键入 `python3 tkinter_classes_demo.py` 时。如果我们将此文件作为模块导入到另一个 Python 脚本中，此检查将为假，并且该块内的代码将不会运行。将程序的主要执行代码放在此检查下面是一个好习惯，这样你就可以安全地在更大的应用程序中重用你的类和函数。

# 使用类重写我们的应用程序

现在我们已经学会了在代码中使用类的方法，让我们将其应用到我们的 ABQ 数据录入应用程序中。我们将从一个名为 `data_entry_app.py` 的新文件开始，并添加我们的导入语句，如下所示：

```py
# data_entry_app.py
from datetime import datetime
from pathlib import Path
import csv
import tkinter as tk
from tkinter import ttk 
```

现在，让我们看看我们如何应用一些基于类的技术来重写我们应用程序代码的更简洁版本。

## 向 Text 小部件添加 StringVar

在创建我们的应用程序时，我们发现的一个烦恼是 `Text` 小部件不允许使用 `StringVar` 来存储其内容，这迫使我们不得不与其他所有小部件不同地处理它。这确实有一个很好的原因：Tkinter 的 `Text` 小部件远不止是一个多行 `Entry` 小部件，它可以包含富文本、图像和其他低级 `StringVar` 无法存储的东西。话虽如此，我们并没有使用这些功能，因此对我们来说，有一个更有限的 `Text` 小部件，它可以绑定到一个变量上会更好。

让我们创建一个名为 `BoundText` 的子类来解决这个问题；从以下代码开始：

```py
class BoundText(tk.Text):
  """A Text widget with a bound variable.""" 
```

我们的类需要向 `Text` 类添加三件事情：

+   它需要允许我们传入一个 `StringVar`，它将被绑定到。

+   它需要在变量更新时更新小部件内容；例如，如果它从文件中加载或被另一个小部件更改。

+   它需要在小部件更新时更新变量内容；例如，当用户在控件中键入或粘贴内容时。

### 传递一个变量

我们将首先覆盖初始化器，以便传递一个控制变量：

```py
 def __init__(self, *args, textvariable=None, **kwargs):
    super().__init__(*args, **kwargs)
    self._variable = textvariable 
```

按照 Tkinter 的惯例，我们将使用`textvariable`参数传递`StringVar`对象。在将剩余参数传递给`super().__init__()`之后，我们将变量存储为类的保护成员。

接下来，如果用户提供了变量，我们将将其内容插入到小部件中（这解决了分配给变量的任何默认值）：

```py
 if self._variable:
      self.insert('1.0', self._variable.get()) 
```

注意，如果没有传递变量，`textvariable`（以及因此`self._variable`）将是`None`。

### 将小部件与变量同步

下一步，我们需要将控制变量的修改绑定到一个将更新小部件的实例方法。

仍然在`__init__()`方法中工作，让我们在刚刚创建的`if`块内添加一个跟踪，如下所示：

```py
 if self._variable:
      self.insert('1.0', self._variable.get())
      **self._variable.trace_add('write', self._set_content)** 
```

我们跟踪的回调是一个名为`_set_content()`的保护成员函数，它将更新小部件的内容，以变量的内容为准。让我们继续创建这个回调：

```py
 def _set_content(self, *_):
    """Set the text contents to the variable"""
    self.delete('1.0', tk.END)
    self.insert('1.0', self._variable.get()) 
```

首先，请注意我们的回调函数的参数列表中包含`* _`。这种表示法简单地将传递给函数的任何位置参数包装在一个名为`_`（下划线）的变量中。单个下划线或一系列下划线是我们用来命名 Python 变量的一种传统方式，我们提供这些变量但不打算使用它们。在这种情况下，我们使用它来消耗 Tkinter 在响应事件调用此函数时传递给此函数的任何额外参数。您将在我们打算将它们绑定到 Tkinter 事件的其他回调方法中看到这种技术被使用。

在方法内部，我们将简单地使用其`delete()`和`insert()`方法修改小部件的内容。

### 将变量与小部件同步

当小部件被修改时更新变量稍微复杂一些。我们需要找到一个事件，每当`Text`小部件被编辑时就会触发，以便绑定到我们的回调。我们可以使用`<Key>`事件，它在按键时触发，但它不会捕获基于鼠标的编辑，如粘贴操作。然而，`Text`小部件确实有一个`<<Modified>>`事件，它在第一次修改时发出。

我们可以从这里开始；在`__init__()`方法中`if`语句的末尾添加另一行，如下所示：

```py
 if self._variable:
      self.insert('1.0', self._variable.get())
      self._variable.trace_add('write', self._set_content)
      `self.bind('<<Modified>>', self._set_var)` 
```

然而，`<<Modified>>`事件仅在第一次修改小部件时触发。之后，我们需要通过更改小部件的修改标志来重置事件。我们可以使用`Text`小部件的`edit_modified()`方法来完成此操作，该方法还允许我们检索修改标志的状态。

为了了解这将如何工作，让我们编写`_set_var()`回调：

```py
 def _set_var(self, *_):
    """Set the variable to the text contents"""
    if self.edit_modified():
      content = self.get('1.0', 'end-1chars')
      self._variable.set(content)
      self.edit_modified(False) 
```

在这个方法中，我们首先通过调用 `edit_modified()` 来检查小部件是否已被修改。如果是，我们将使用小部件的 `get()` 方法检索内容。请注意，`get` 的结束索引为 `end-1chars`。这意味着“内容结束前的一个字符。”回想一下，`Text` 小部件的 `get()` 方法会自动将换行符追加到内容的末尾，因此通过使用此索引，我们可以消除额外的换行符。

在检索小部件的内容后，我们需要通过将 `False` 传递给 `edit_modified()` 方法来重置已修改标志。这样，它就准备好在用户下次与小部件交互时触发 `<<Modified>>` 事件。

## 创建一个更高级的 LabelInput()

我们在 *创建复合小部件* 下创建的 `LabelInput` 类似乎很有用，但如果我们想在程序中使用它，它将需要更多的完善。

让我们再次从类定义和初始化方法开始：

```py
# data_entry_app.py
class LabelInput(tk.Frame):
  """A widget containing a label and input together."""
  def __init__(
    self, parent, label, var, input_class=ttk.Entry,
    input_args=None, label_args=None, **kwargs
  ):
    super().__init__(parent, **kwargs)
    input_args = input_args or {}
    label_args = label_args or {}
    self.variable = var
    self.variable.label_widget = self 
```

和以前一样，我们有父小部件、标签文本、输入类和输入参数的参数。由于我们现在想要使用的每个小部件都可以绑定一个变量，我们将接受它作为必需参数，并且如果需要，我们将添加一个可选参数来传递给标签小部件的参数字典。我们将 `input_class` 默认设置为 `ttk.Entry`，因为我们有几个这样的小部件。

注意，`input_args` 和 `label_args` 参数的默认值是 `None`，并且如果它们是 `None`，我们在方法内部将它们作为字典。为什么不直接使用空字典作为默认参数呢？在 Python 中，默认参数是在函数定义首次运行时评估的。这意味着在函数签名中创建的字典对象将在每次函数运行时都是相同的对象，而不是每次都是一个新的空字典。由于我们希望每次都是一个新的空字典，所以我们将在函数体内部而不是在参数列表中创建字典。对于列表和其他可变对象也是如此。

在方法内部，我们像往常一样调用 `super().__init__()`，然后确保 `input_args` 和 `label_args` 是字典。最后，我们将 `input_var` 保存到实例变量中，并将标签小部件本身保存为变量对象的属性。这样做意味着我们不需要存储 `LabelInput` 对象的引用；如果我们需要，我们可以通过变量对象来访问它们。

接下来，是设置标签的时候了：

```py
 if input_class in (ttk.Checkbutton, ttk.Button):
      input_args["text"] = label
    else:
      self.label = ttk.Label(self, text=label, **label_args)
      self.label.grid(row=0, column=0, sticky=(tk.W + tk.E)) 
```

`Checkbutton` 和 `Button` 小部件内部已经集成了标签，所以我们不希望有一个单独的标签悬挂在那里。相反，我们将只设置小部件的 `text` 参数为传入的任何内容。（`Radiobutton` 对象也内置了标签，但我们会稍作不同处理，你将在下一刻看到）。对于所有其他小部件，我们将在 `LabelInput` 的第一行和第一列中添加一个 `Label` 小部件。

接下来，我们需要设置输入参数，以便输入的控制变量将以正确的参数名称传递：

```py
 if input_class in (
      ttk.Checkbutton, ttk.Button, ttk.Radiobutton
    ):
      input_args["variable"] = self.variable
    else:
      input_args["textvariable"] = self.variable 
```

记得按钮类使用 `variable` 作为参数名称，而所有其他类使用 `textvariable`。通过在类内部处理这个问题，我们就不必在构建我们的表单时担心这种区别。

现在，让我们设置输入小部件。大多数小部件的设置都很简单，但对于 `Radiobutton`，我们需要做些不同的事情。我们需要为传递的每个可能的值创建一个 `Radiobutton` 小部件（使用 `input_args` 中的 `values` 键）。记住，我们通过让按钮共享相同的变量来链接它们，我们在这里也会这样做。

我们可以这样添加：

```py
 if input_class == ttk.Radiobutton:
      self.input = tk.Frame(self)
      for v in input_args.pop('values', []):
        button = ttk.Radiobutton(
          self.input, value=v, text=v, **input_args
        )
        button.pack(
          side=tk.LEFT, ipadx=10, ipady=2, expand=True, fill='x'
        ) 
```

首先，我们创建一个 `Frame` 对象来容纳按钮；然后，对于传递给 `values` 的每个值，我们向 `Frame` 布局中添加一个 `Radiobutton` 小部件。注意，我们调用 `pop()` 方法从 `input_args` 字典中获取 `values` 项。`dict.pop()` 几乎与 `dict.get()` 相同，如果给定键存在，则返回该键的值，如果不存在，则返回第二个参数。区别在于 `pop()` 还会从字典中删除检索到的项。我们这样做是因为 `values` 不是 `Radiobutton` 的有效参数，所以在将 `input_args` 传递给 `Radiobutton` 初始化器之前，我们需要将其删除。`input_args` 中剩余的项应该是小部件的有效关键字参数。

对于非 `Radiobutton` 小部件的情况，操作相当直接：

```py
 else:
      self.input = input_class(self, **input_args) 
```

我们只需调用带有 `input_args` 参数的任何 `input_class` 类。现在我们已经创建了 `self.input`，我们只需将其添加到 `LabelInput` 布局中：

```py
 self.input.grid(row=1, column=0, sticky=(tk.W + tk.E))
    self.columnconfigure(0, weight=1) 
```

最后一次调用 `columnconfigure` 命令是告诉 `LabelWidget` 小部件填充其整个宽度至列 `0`。

当我们创建自己的小部件时（无论是自定义子类还是复合小部件），我们可以为几何布局设置一些合理的默认值。例如，我们希望所有 `LabelInput` 小部件都紧贴其容器的左右两侧，以便它们填充最大可用宽度。而不是每次定位 `LabelInput` 小部件时都必须传递 `sticky=(tk.E + tk.W)`，让我们将其设置为默认值，如下所示：

```py
 def grid(self, sticky=(tk.E + tk.W), **kwargs):
    """Override grid to add default sticky values"""
    super().grid(sticky=sticky, **kwargs) 
```

我们已经覆盖了 `grid` 并将参数传递给了超类版本，但为 `sticky` 添加了一个默认值。如果需要，我们仍然可以覆盖它，但这将节省我们很多麻烦。

我们的 `LabelInput` 现在相当稳健了；是时候让它派上用场了！

## 创建表单类

现在我们已经准备好了构建块，是时候构建我们应用程序的主要组件了。将应用程序分解成合理的组件需要考虑什么可能构成合理的职责划分。最初，我们的应用程序似乎可以分解成两个组件：数据输入表单和根应用程序本身。但哪些功能放在哪里呢？

一种合理的评估可能如下：

+   数据输入表本身当然应该包含所有的小部件。它还应该包含保存和重置按钮，因为这些按钮与表分开没有意义。

+   应用程序标题和状态栏属于通用级别，因为它们将适用于应用程序的所有部分。文件保存可以与表一起进行，但它还必须与一些应用程序级别的项目（如状态栏或`records_saved`变量）交互。这是一个棘手的选择，但我们将它暂时放在应用程序对象中。

让我们从构建我们的数据输入表类`DataRecordForm`开始：

```py
# data_entry_app.py
class DataRecordForm(ttk.Frame):
  """The input form for our widgets"""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs) 
```

像往常一样，我们首先通过继承`Frame`并调用超类初始化方法来开始。在这个阶段，我们实际上不需要添加任何自定义参数。

现在，让我们创建一个字典来保存所有我们的变量对象：

```py
 self._vars = {
      'Date': tk.StringVar(),
      'Time': tk.StringVar(),
      'Technician': tk.StringVar(),
      'Lab': tk.StringVar(),
      'Plot': tk.IntVar(),
      'Seed Sample': tk.StringVar(),
      'Humidity': tk.DoubleVar(),
      'Light': tk.DoubleVar(),
      'Temperature': tk.DoubleVar(),
      'Equipment Fault': tk.BooleanVar(),
      'Plants': tk.IntVar(),
      'Blossoms': tk.IntVar(),
      'Fruit': tk.IntVar(),
      'Min Height': tk.DoubleVar(),
      'Max Height': tk.DoubleVar(),
      'Med Height': tk.DoubleVar(),
      'Notes': tk.StringVar()
    } 
```

这只是直接从我们的数据字典中提取的。请注意，多亏了我们的`BoundText`类，我们可以将`StringVar`对象分配给注释。现在，我们准备开始向我们的 GUI 添加小部件。在我们的应用程序当前版本中，我们使用类似这样的代码块为应用程序的每个部分添加了一个`LabelFrame`小部件：

```py
r_info = ttk.LabelFrame(drf, text='Record Information')
r_info.grid(sticky=(tk.W + tk.E))
for i in range(3):
  r_info.columnconfigure(i, weight=1 ) 
```

这段代码为每个框架重复了一次，只是变量名和标签文本有所变化。为了避免这种重复，我们可以将这个过程抽象成一个实例方法。让我们创建一个可以为我们添加新的标签框架的方法；将此代码添加到`__init__()`定义之上：

```py
 def _add_frame(self, label, cols=3):
    """Add a LabelFrame to the form"""
    frame = ttk.LabelFrame(self, text=label)
    frame.grid(sticky=tk.W + tk.E)
    for i in range(cols):
      frame.columnconfigure(i, weight=1)
    return frame 
```

这个方法只是以前面的代码以通用方式重新定义，这样我们就可以传入标签文本，以及可选的列数。回滚到我们在`DataRecordForm.__init__()`方法中的位置，让我们使用这个方法创建一个记录信息部分，如下所示：

```py
 r_info = self._add_frame("Record Information") 
```

现在我们有了框架，让我们尝试使用`LabelInput`并开始构建表的第一部分，如下所示：

```py
 LabelInput(
      r_info, "Date", var=self._vars['Date']
    ).grid(row=0, column=0)
    LabelInput(
      r_info, "Time", input_class=ttk.Combobox,
      var=self._vars['Time'],
      input_args={"values": ["8:00", "12:00", "16:00", "20:00"]}
    ).grid(row=0, column=1)
    LabelInput(
      r_info, "Technician",  var=self._vars['Technician']
    ).grid(row=0, column=2) 
```

如您所见，`LabelInput`已经为我们节省了很多冗余的杂乱。

让我们继续第二行：

```py
 LabelInput(
      r_info, "Lab", input_class=ttk.Radiobutton,
      var=self._vars['Lab'],
      input_args={"values": ["A", "B", "C"]}
    ).grid(row=1, column=0)
    LabelInput(
      r_info, "Plot", input_class=ttk.Combobox,
      var=self._vars['Plot'],
      input_args={"values": list(range(1, 21))}
    ).grid(row=1, column=1)
    LabelInput(
      r_info, "Seed Sample",  var=self._vars['Seed Sample']
    ).grid(row=1, column=2) 
```

记住，为了使用与`LabelInput`一起的`RadioButton`小部件，我们需要将值列表传递给输入参数，就像我们对`Combobox`做的那样。完成`记录信息`部分后，让我们继续下一个部分，`环境数据`：

```py
 e_info = self._add_frame("Environment Data")
    LabelInput(
      e_info, "Humidity (g/m³)",
      input_class=ttk.Spinbox,  var=self._vars['Humidity'],
      input_args={"from_": 0.5, "to": 52.0, "increment": .01}
    ).grid(row=0, column=0)
    LabelInput(
      e_info, "Light (klx)", input_class=ttk.Spinbox,
      var=self._vars['Light'],
      input_args={"from_": 0, "to": 100, "increment": .01}
    ).grid(row=0, column=1)
    LabelInput(
      e_info, "Temperature (°C)",
      input_class=ttk.Spinbox,  var=self._vars['Temperature'],
      input_args={"from_": 4, "to": 40, "increment": .01}
    ).grid(row=0, column=2)
    LabelInput(
      e_info, "Equipment Fault",
      input_class=ttk.Checkbutton,  
      var=self._vars['Equipment Fault']
    ).grid(row=1, column=0, columnspan=3) 
```

同样，我们使用我们的`_add_frame()`方法添加并配置了一个`LabelFrame`，并用四个`LabelInput`小部件填充它。

现在，让我们添加`植物数据`部分：

```py
 p_info = self._add_frame("Plant Data")
    LabelInput(
      p_info, "Plants", input_class=ttk.Spinbox,
      var=self._vars['Plants'],
      input_args={"from_": 0, "to": 20}
    ).grid(row=0, column=0)
    LabelInput(
      p_info, "Blossoms", input_class=ttk.Spinbox,
      var=self._vars['Blossoms'],
      input_args={"from_": 0, "to": 1000}
    ).grid(row=0, column=1)
    LabelInput(
      p_info, "Fruit", input_class=ttk.Spinbox,
      var=self._vars['Fruit'],
      input_args={"from_": 0, "to": 1000}
    ).grid(row=0, column=2)
    LabelInput(
      p_info, "Min Height (cm)",
      input_class=ttk.Spinbox,  var=self._vars['Min Height'],
      input_args={"from_": 0, "to": 1000, "increment": .01}
    ).grid(row=1, column=0)
    LabelInput(
      p_info, "Max Height (cm)",
      input_class=ttk.Spinbox,  var=self._vars['Max Height'],
      input_args={"from_": 0, "to": 1000, "increment": .01}
    ).grid(row=1, column=1)
    LabelInput(
      p_info, "Median Height (cm)",
      input_class=ttk.Spinbox,  var=self._vars['Med Height'],
      input_args={"from_": 0, "to": 1000, "increment": .01}
    ).grid(row=1, column=2) 
```

我们几乎完成了；让我们接下来添加我们的`注释`部分：

```py
 LabelInput(
      self, "Notes",
      input_class=BoundText,  var=self._vars['Notes'],
      input_args={"width": 75, "height": 10}
    ).grid(sticky=tk.W, row=3, column=0) 
```

在这里，我们利用我们的`BoundText`对象来附加一个变量。否则，这看起来就像对`LabelInput`的所有其他调用一样。

现在，是时候添加按钮了：

```py
 buttons = tk.Frame(self)
    buttons.grid(sticky=tk.W + tk.E, row=4)
    self.savebutton = ttk.Button(
      buttons, text="Save", command=self.master._on_save)
    self.savebutton.pack(side=tk.RIGHT)
    self.resetbutton = ttk.Button(
      buttons, text="Reset", command=self.reset)
    self.resetbutton.pack(side=tk.RIGHT) 
```

与之前一样，我们在`Frame`上添加了我们的按钮小部件。不过，这一次，我们将传递一些实例方法作为按钮的回调命令。`重置`按钮将获得我们在本类中定义的实例方法，但由于我们决定保存文件是应用程序对象的责任，我们将`保存`按钮绑定到父对象的实例方法（通过这个对象的`master`属性访问）。

将 GUI 对象直接绑定到其他对象的命令上不是解决对象间通信问题的好方法，但就目前而言，它将完成这项工作。在*第六章*，*为应用程序的扩展做准备*，我们将学习一种更优雅的方法来完成这项工作。

这样就结束了我们的`__init__()`方法，但在我们完成之前，这个类还需要几个更多的方法。首先，我们需要实现`reset()`方法来处理表单重置；它看起来是这样的：

```py
 def reset(self):
    """Resets the form entries"""
    for var in self._vars.values():
      if isinstance(var, tk.BooleanVar):
        var.set(False)
      else:
        var.set('') 
```

实际上，我们只需要将所有变量设置为空字符串。然而，对于`BooleanVar`对象，这样做将会引发异常，因此我们需要将其设置为`False`来取消选中复选框。

最后，我们需要一个方法，使得应用程序对象能够从表单中检索数据，以便它可以保存数据。按照 Tkinter 的约定，我们将这个方法命名为`get()`：

```py
 def get(self):
    data = dict()
    fault = self._vars['Equipment Fault'].get()
    for key, variable in self._vars.items():
      if fault and key in ('Light', 'Humidity', 'Temperature'):
        data[key] = ''
      else:
        try:
          data[key] = variable.get()
        except tk.TclError:
          message = f'Error in field: {key}.  Data was not saved!'
          raise ValueError(message)
    return data 
```

这里的代码与我们之前版本应用程序中`on_save()`函数的数据检索代码非常相似，但有几点不同。首先，我们从`self._vars`而不是全局变量字典中检索数据。其次，在发生错误的情况下，我们创建一个错误消息并重新抛出一个`ValueError`，而不是直接更新 GUI。我们必须确保调用此方法的代码能够处理`ValueError`异常。最后，与之前版本的应用程序中保存数据的方式不同，我们只是返回数据。

这样就完成了表单类的创建！现在剩下的只是编写一个应用程序来保持它。

## 创建应用程序类

我们的应用程序类将处理应用程序级别的功能，同时也是我们的顶级窗口。在 GUI 方面，它需要包含：

+   标题标签

+   我们`DataRecordForm`类的一个实例

+   状态栏

它还需要一个方法将表单中的数据保存到 CSV 文件中。

让我们开始我们的类：

```py
class Application(tk.Tk):
  """Application root window"""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs) 
```

这里没有什么新的内容，只是现在我们正在继承`Tk`而不是`Frame`。

让我们设置一些窗口参数：

```py
 self.title("ABQ Data Entry Application")
    self.columnconfigure(0, weight=1) 
```

与程序的程序性版本一样，我们已经设置了窗口标题并配置了网格的第一列以扩展。现在，我们将创建标题标签：

```py
 ttk.Label(
      self, text="ABQ Data Entry Application",
      font=("TkDefaultFont", 16)
    ).grid(row=0) 
```

这里没有什么真正不同的地方，只是要注意，父对象现在是`self`——将不再有`root`对象；`self`是这个类内部的`Tk`实例。

让我们创建一个记录表单：

```py
 self.recordform = DataRecordForm(self)
    self.recordform.grid(row=1, padx=10, sticky=(tk.W + tk.E)) 
```

尽管`DataRecordForm`的大小和复杂性，将其添加到应用程序中就像添加任何其他小部件一样简单。

现在，让我们看看状态栏：

```py
 self.status = tk.StringVar()
    ttk.Label(
      self, textvariable=self.status
    ).grid(sticky=(tk.W + tk.E), row=2, padx=10) 
```

再次强调，这和过程式版本非常相似，只是我们的`status`变量是一个实例变量。这意味着它将可以访问我们类中的任何方法。

最后，让我们创建一个受保护的实例变量来保存已保存的记录数：

```py
 self._records_saved = 0 
```

完成了`__init__()`方法后，我们现在可以编写最后一个方法：`_on_save()`。这个方法将非常接近我们之前编写的过程式函数：

```py
 def _on_save(self):
    """Handles save button clicks"""
    datestring = datetime.today().strftime("%Y-%m-%d")
    filename = "abq_data_record_{}.csv".format(datestring)
    newfile = not Path(filename).exists()
    try:
      data = self.recordform.get()
    except ValueError as e:
      self.status.set(str(e))
      return
    with open(filename, 'a', newline='') as fh:
      csvwriter = csv.DictWriter(fh, fieldnames=data.keys())
      if newfile:
        csvwriter.writeheader()
      csvwriter.writerow(data)
    self._records_saved += 1
    self.status.set(
      "{} records saved this session".format(self._records_saved))
    self.recordform.reset() 
```

再次强调，这个函数使用当前日期生成文件名，然后以追加模式打开文件。不过，这次我们可以通过简单地调用`self.recordform.get()`来获取我们的数据，这抽象了从变量获取数据的过程。记住，我们确实必须处理`ValueError`异常，以防表中有不良数据，我们在这里已经做到了。如果数据不良，我们只需在方法尝试保存数据之前在状态栏中显示错误并退出。如果没有异常，数据将被保存，因此我们增加`_records_saved`属性并更新状态。

使这个应用程序运行的最后一件事情是创建我们的`Application`对象实例并启动其`mainloop`：

```py
if __name__ == "__main__":
  app = Application()
  app.mainloop() 
```

注意，除了我们的类定义和模块导入之外，这两行是唯一在顶层作用域中执行的。另外，因为`Application`负责构建 GUI 和其他对象，所以我们可以使用`if __name__ == "__main__"`保护来一起在应用程序的末尾执行它和`mainloop()`调用。

# 摘要

在本章中，你学习了如何利用 Python 类的强大功能。你学习了如何创建自己的类，定义属性和方法，以及魔术方法的功能。你还学习了如何通过子类化扩展现有类的功能。

我们探讨了如何将这些技术有力地应用于 Tkinter 类，以扩展其功能，构建复合小部件，并将我们的应用程序组织成组件。

在下一章中，我们将学习 Tkinter 的验证功能，并进一步使用子类化使我们的小部件更加直观和健壮。我们还将学习如何自动化输入以节省用户时间并确保数据输入的一致性。
