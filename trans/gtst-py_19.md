# 何时使用面向对象编程

在之前的章节中，我们已经涵盖了面向对象编程的许多定义特性。我们现在知道面向对象设计的原则和范例，并且我们已经涵盖了Python中面向对象编程的语法。

然而，我们并不确切知道如何，尤其是何时在实践中利用这些原则和语法。在本章中，我们将讨论我们所获得的知识的一些有用应用，同时查看一些新的主题：

+   如何识别对象

+   数据和行为，再次

+   使用属性封装数据行为

+   使用行为限制数据

+   不要重复自己的原则

+   识别重复的代码

# 将对象视为对象

这可能看起来很明显；你通常应该在代码中为问题域中的单独对象给予特殊的类。我们在之前章节的案例研究中已经看到了这样的例子：首先，我们确定问题中的对象，然后对其数据和行为进行建模。

在面向对象分析和编程中，识别对象是一项非常重要的任务。但这并不总是像计算短段落中的名词那样容易，坦率地说，我明确为此目的构建了。记住，对象是既有数据又有行为的东西。如果我们只处理数据，通常最好将其存储在列表、集合、字典或其他Python数据结构中。另一方面，如果我们只处理行为，但没有存储的数据，一个简单的函数更合适。

然而，对象既有数据又有行为。熟练的Python程序员使用内置数据结构，除非（或直到）明显需要定义一个类。如果这并没有帮助组织我们的代码，那么没有理由添加额外的抽象级别。另一方面，*明显的*需要并不总是不言自明的。

我们通常可以通过将数据存储在几个变量中来启动我们的Python程序。随着程序的扩展，我们将会发现我们正在将相同的一组相关变量传递给一组函数。这是思考将变量和函数组合成一个类的时候了。如果我们正在设计一个在二维空间中模拟多边形的程序，我们可能会从将每个多边形表示为点列表开始。这些点将被建模为两个元组（*x*，*y*），描述该点的位置。这是所有的数据，存储在一组嵌套的数据结构中（具体来说，是一个元组列表）：

```py
square = [(1,1), (1,2), (2,2), (2,1)] 
```

现在，如果我们想要计算多边形周长的距离，我们需要计算每个点之间的距离。为此，我们需要一个函数来计算两点之间的距离。以下是两个这样的函数：

```py
import math

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def perimeter(polygon):
    perimeter = 0
    points = polygon + [polygon[0]]
    for i in range(len(polygon)):
        perimeter += distance(points[i], points[i+1])
    return perimeter

```

现在，作为面向对象的程序员，我们清楚地认识到`polygon`类可以封装点的列表（数据）和`perimeter`函数（行为）。此外，`point`类，就像我们在[第16章](9179e7f0-fea3-4c8a-b134-a190544099b4.xhtml)中定义的那样，*Python中的对象*，可能封装`x`和`y`坐标以及`distance`方法。问题是：这样做有价值吗？

对于以前的代码，也许是，也许不是。有了我们最近在面向对象原则方面的经验，我们可以以创纪录的速度编写面向对象的版本。让我们进行比较：

```py
class Point:
 def __init__(self, x, y):
 self.x = x
 self.y = y

    def distance(self, p2):
        return math.sqrt((self.x-p2.x)**2 + (self.y-p2.y)**2)

class Polygon:
 def __init__(self):
 self.vertices = []

 def add_point(self, point):
 self.vertices.append((point))

    def perimeter(self):
        perimeter = 0
        points = self.vertices + [self.vertices[0]]
        for i in range(len(self.vertices)):
            perimeter += points[i].distance(points[i+1])
        return perimeter
```

正如我们从突出显示的部分所看到的，这里的代码量是我们之前版本的两倍，尽管我们可以争辩说`add_point`方法并不是严格必要的。

现在，为了更好地理解这两种API之间的差异，让我们比较一下两种使用情况。以下是使用面向对象的代码计算正方形的周长：

```py
>>> square = Polygon()
>>> square.add_point(Point(1,1))
>>> square.add_point(Point(1,2))
>>> square.add_point(Point(2,2))
>>> square.add_point(Point(2,1))
>>> square.perimeter()
4.0  
```

这可能看起来相当简洁和易读，但让我们将其与基于函数的代码进行比较：

```py
>>> square = [(1,1), (1,2), (2,2), (2,1)]
>>> perimeter(square)
4.0  
```

嗯，也许面向对象的API并不那么紧凑！也就是说，我认为它比函数示例更容易*阅读*。我们怎么知道第二个版本中的元组列表应该表示什么？我们怎么记得我们应该传递到`perimeter`函数的对象是什么？（两个元组的列表？这不直观！）我们需要大量的文档来解释这些函数应该如何使用。

相比之下，面向对象的代码相对自我说明。我们只需要查看方法列表及其参数，就可以知道对象的功能和如何使用它。当我们为函数版本编写所有文档时，它可能会比面向对象的代码还要长。

最后，代码长度并不是代码复杂性的良好指标。一些程序员会陷入复杂的*一行代码*中，这一行代码可以完成大量工作。这可能是一个有趣的练习，但结果通常是令人难以阅读的，即使对于原始作者来说，第二天也是如此。最小化代码量通常可以使程序更易于阅读，但不要盲目地假设这是正确的。

幸运的是，这种权衡是不必要的。我们可以使面向对象的`Polygon` API与函数实现一样易于使用。我们只需要修改我们的`Polygon`类，使其可以用多个点构造。让我们给它一个接受`Point`对象列表的初始化器。事实上，让我们也允许它接受元组，如果需要，我们可以自己构造`Point`对象：

```py
def __init__(self, points=None): 
    points = points if points else [] 
    self.vertices = [] 
    for point in points: 
        if isinstance(point, tuple): 
            point = Point(*point) 
        self.vertices.append(point) 
```

这个初始化器遍历列表，并确保任何元组都转换为点。如果对象不是元组，我们将其保留，假设它已经是`Point`对象，或者是一个未知的鸭子类型对象，可以像`Point`对象一样工作。

如果您正在尝试上述代码，您可以对`Polygon`进行子类化，并覆盖`__init__`函数，而不是替换初始化器或复制`add_point`和`perimeter`方法。

然而，在面向对象和更注重数据的版本之间没有明显的赢家。它们都做同样的事情。如果我们有新的函数接受多边形参数，比如`area(polygon)`或`point_in_polygon(polygon, x, y)`，面向对象代码的好处变得越来越明显。同样，如果我们为多边形添加其他属性，比如`color`或`texture`，将这些数据封装到一个类中就变得更有意义。

区别是一个设计决策，但一般来说，数据集越重要，就越有可能具有针对该数据的多个特定功能，使用具有属性和方法的类会更有用。

在做出这个决定时，考虑类将如何使用也是很重要的。如果我们只是试图在更大的问题的背景下计算一个多边形的周长，使用函数可能会是编码最快且最容易*仅一次*使用。另一方面，如果我们的程序需要以各种方式操作大量多边形（计算周长、面积和与其他多边形的交集、移动或缩放它们等），我们几乎肯定已经确定了一个对象；一个需要非常灵活的对象。

此外，要注意对象之间的交互。寻找继承关系；继承无法在没有类的情况下优雅地建模，因此一定要使用它们。寻找我们在[第15章](c232a9c5-f0ac-46d8-8c31-41aa4d225a71.xhtml)中讨论的其他类型的关系，*面向对象设计*，关联和组合。组合在技术上可以使用只有数据结构来建模；例如，我们可以有一个包含元组值的字典列表，但有时创建几个对象类会更不复杂，特别是如果与数据相关联的行为。

不要急于使用对象，只是因为你可以使用对象，但是当你需要使用类时，不要忽视创建一个类。

# 使用属性为类数据添加行为

在整本书中，我们一直专注于行为和数据的分离。这在面向对象编程中非常重要，但是我们将看到，在Python中，这种区别是模糊的。Python非常擅长模糊界限；它并不完全帮助我们*打破思维定势*。相反，它教会我们停止思考盒子。

在我们深入细节之前，让我们讨论一些糟糕的面向对象理论。许多面向对象的语言教导我们永远不要直接访问属性（Java是最臭名昭著的）。他们坚持我们应该像这样写属性访问：

```py
class Color: 
    def __init__(self, rgb_value, name): 
        self._rgb_value = rgb_value 
        self._name = name 

 def set_name(self, name): 
        self._name = name 

 def get_name(self): 
        return self._name 
```

变量以下划线开头，表示它们是私有的（其他语言实际上会强制它们为私有）。然后，`get`和`set`方法提供对每个变量的访问。这个类将在实践中使用如下：

```py
>>> c = Color("#ff0000", "bright red")
>>> c.get_name()
'bright red'
>>> c.set_name("red")
>>> c.get_name()
'red'  
```

这不像Python青睐的直接访问版本那样易读：

```py
class Color: 
    def __init__(self, rgb_value, name): 
        self.rgb_value = rgb_value 
        self.name = name 

c = Color("#ff0000", "bright red") 
print(c.name) c.name = "red"
print(c.name)
```

那么，为什么有人坚持使用基于方法的语法呢？他们的理由是，有一天，我们可能希望在设置或检索值时添加额外的代码。例如，我们可以决定缓存一个值以避免复杂的计算，或者我们可能希望验证给定的值是否是合适的输入。

例如，在代码中，我们可以决定将`set_name()`方法更改如下：

```py
def set_name(self, name): 
    if not name: 
        raise Exception("Invalid Name") 
    self._name = name 
```

现在，在Java和类似的语言中，如果我们最初为直接属性访问编写了原始代码，然后稍后将其更改为像前面的方法，我们会有问题：任何访问属性的代码现在都必须访问一个方法。如果他们没有将访问样式从属性访问更改为函数调用，他们的代码将会出错。

这些语言中的口头禅是我们永远不应该将公共成员变为私有成员。这在Python中并没有太多意义，因为Python没有真正的私有成员的概念！

Python给了我们`property`关键字，可以使方法看起来像属性。因此，我们可以编写代码来直接访问成员，如果我们需要在获取或设置属性值时进行一些计算，我们可以在不改变接口的情况下进行修改。让我们看看它是什么样子：

```py
class Color: 
    def __init__(self, rgb_value, name): 
        self.rgb_value = rgb_value 
        self._name = name 

    def _set_name(self, name): 
        if not name: 
            raise Exception("Invalid Name") 
        self._name = name 

    def _get_name(self): 
        return self._name 

 name = property(_get_name, _set_name) 
```

与之前的类相比，我们首先将`name`属性更改为(半)私有的`_name`属性。然后，我们添加了两个更多的(半)私有方法来获取和设置该变量，在设置时执行验证。

最后，我们在底部有`property`声明。这就是Python的魔力。它在`Color`类上创建了一个名为`name`的新属性，以替换直接的`name`属性。它将此属性设置为**property**。在幕后，`property`在访问或更改值时调用我们刚刚创建的两个方法。这个新版本的`Color`类可以像以前的版本一样使用，但是现在在设置`name`属性时执行验证：

```py
>>> c = Color("#0000ff", "bright red")
>>> print(c.name)
bright red
>>> c.name = "red"
>>> print(c.name)
red
>>> c.name = ""
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "setting_name_property.py", line 8, in _set_name
 raise Exception("Invalid Name")
Exception: Invalid Name  
```

因此，如果我们以前编写了访问`name`属性的代码，然后更改为使用基于`property`的对象，以前的代码仍然可以工作，除非它发送了一个空的`property`值，这正是我们想要在第一次禁止的行为。成功！

请记住，即使有了`name`属性，以前的代码也不是100%安全的。人们仍然可以直接访问`_name`属性，并将其设置为空字符串。但是，如果他们访问了我们明确标记为下划线的变量，暗示它是私有的，他们就必须处理后果，而不是我们。

# 属性详解

将`property`函数视为返回一个对象，通过我们指定的方法代理对设置或访问属性值的任何请求。内置的`property`就像这样的对象的构造函数，并且该对象被设置为给定属性的公共成员。

这个`property`构造函数实际上可以接受两个额外的参数，一个`delete`函数和一个属性的文档字符串。在实践中很少提供`delete`函数，但如果我们有理由这样做，它可能对记录已删除的值或可能否决删除很有用。文档字符串只是描述属性功能的字符串，与我们在[第16章](9179e7f0-fea3-4c8a-b134-a190544099b4.xhtml)中讨论的文档字符串没有什么不同，*Python中的对象*。如果我们不提供此参数，文档字符串将从第一个参数的文档字符串复制：`getter`方法。这是一个愚蠢的例子，说明每当调用任何方法时：

```py
class Silly:
    def _get_silly(self):
        print("You are getting silly")
        return self._silly

    def _set_silly(self, value):
        print("You are making silly {}".format(value))
        self._silly = value

    def _del_silly(self):
        print("Whoah, you killed silly!")
        del self._silly

 silly = property(_get_silly, _set_silly, _del_silly, "This is a silly property")
```

如果我们实际使用这个类，当我们要求它时，它确实会打印出正确的字符串：

```py
>>> s = Silly()
>>> s.silly = "funny"
You are making silly funny
>>> s.silly
You are getting silly
'funny'
>>> del s.silly
Whoah, you killed silly!  
```

此外，如果我们查看`Silly`类的帮助文件（通过在解释器提示符处发出`help(Silly)`），它会显示我们的`silly`属性的自定义文档字符串：

```py
Help on class Silly in module __main__: 

class Silly(builtins.object) 
 |  Data descriptors defined here: 
 |   
 |  __dict__ 
 |      dictionary for instance variables (if defined) 
 |   
 |  __weakref__ 
 |      list of weak references to the object (if defined) 
 |   
 |  silly 
 |      This is a silly property 
```

再次，一切都按我们计划的那样运行。在实践中，属性通常只使用前两个参数进行定义：`getter`和`setter`函数。如果我们想为属性提供文档字符串，我们可以在`getter`函数上定义它；属性代理将把它复制到自己的文档字符串中。`delete`函数通常为空，因为对象属性很少被删除。如果程序员尝试删除没有指定`delete`函数的属性，它将引发异常。因此，如果有正当理由删除我们的属性，我们应该提供该函数。

# 装饰器-创建属性的另一种方法

如果您以前从未使用过Python装饰器，您可能希望跳过本节，在我们讨论[第22章](f09b264b-a4c5-4a1e-9911-8f00ca74144c.xhtml)中的装饰器模式之后再回来，*Python设计模式I*。但是，您不需要理解正在发生的事情，以使用装饰器语法来使属性方法更易读。

`property`函数可以与装饰器语法一起使用，将`get`函数转换为`property`函数，如下所示：

```py
class Foo: 
 @property 
    def foo(self): 
        return "bar" 
```

这将`property`函数应用为装饰器，并且等同于以前的`foo = property(foo)`语法。从可读性的角度来看，主要区别在于我们可以在方法的顶部将`foo`函数标记为属性，而不是在定义之后，这样很容易被忽视。这也意味着我们不必创建带有下划线前缀的私有方法来定义属性。

更进一步，我们可以为新属性指定一个`setter`函数，如下所示：

```py
class Foo: 
 @property 
    def foo(self): 
        return self._foo 

 @foo.setter 
    def foo(self, value): 
        self._foo = value 
```

这个语法看起来很奇怪，尽管意图是明显的。首先，我们将`foo`方法装饰为getter。然后，我们通过应用最初装饰的`foo`方法的`setter`属性，装饰了第二个同名方法！`property`函数返回一个对象；这个对象总是带有自己的`setter`属性，然后可以将其应用为其他函数的装饰器。使用相同的名称来命名获取和设置方法并不是必需的，但它确实有助于将访问一个属性的多个方法分组在一起。

我们还可以使用`@foo.deleter`指定一个`delete`函数。我们不能使用`property`装饰器来指定文档字符串，因此我们需要依赖于属性从初始getter方法复制文档字符串。下面是我们之前的`Silly`类重写，以使用`property`作为装饰器：

```py
class Silly: 
 @property 
    def silly(self): 
        "This is a silly property" 
        print("You are getting silly") 
        return self._silly 

 @silly.setter 
    def silly(self, value): 
        print("You are making silly {}".format(value)) 
        self._silly = value 

 @silly.deleter 
    def silly(self): 
        print("Whoah, you killed silly!") 
        del self._silly 
```

这个类的操作*完全*与我们之前的版本相同，包括帮助文本。您可以使用您认为更可读和优雅的任何语法。

# 决定何时使用属性

由于内置的属性模糊了行为和数据之间的区分，很难知道何时选择属性、方法或属性。我们之前看到的用例示例是属性的最常见用法之一；我们在类上有一些数据，然后希望添加行为。在决定使用属性时，还有其他因素需要考虑。

在Python中，数据、属性和方法在类上都是属性。方法可调用的事实并不能将其与其他类型的属性区分开；事实上，我们将在[第20章](72a5d45b-2ade-4c5d-a00c-1c1a36e1a510.xhtml)中看到，*Python面向对象的快捷方式*，可以创建可以像函数一样调用的普通对象。我们还将发现函数和方法本身也是普通对象。

方法只是可调用的属性，属性只是可定制的属性，这可以帮助我们做出这个决定。方法通常应该表示动作；可以对对象执行的操作。当你调用一个方法时，即使只有一个参数，它也应该*做*一些事情。方法名称通常是动词。

确认属性不是一个动作后，我们需要在标准数据属性和属性之间做出选择。通常情况下，始终使用标准属性，直到需要以某种方式控制对该属性的访问。无论哪种情况，您的属性通常是一个名词。属性和属性之间唯一的区别是，当检索、设置或删除属性时，我们可以自动调用自定义操作。

让我们看一个更现实的例子。自定义行为的常见需求是缓存难以计算或昂贵的查找值（例如，需要网络请求或数据库查询）。目标是将值存储在本地，以避免重复调用昂贵的计算。

我们可以通过属性的自定义getter来实现这一点。第一次检索值时，我们执行查找或计算。然后，我们可以将值作为对象的私有属性（或专用缓存软件中）进行本地缓存，下次请求值时，我们返回存储的数据。以下是我们可能缓存网页的方法：

```py
from urllib.request import urlopen

class WebPage:
    def __init__(self, url):
        self.url = url
        self._content = None

    @property
 def content(self):
 if not self._content:
 print("Retrieving New Page...")
 self._content = urlopen(self.url).read()
 return self._content
```

我们可以测试这段代码，以查看页面只被检索一次：

```py
>>> import time
>>> webpage = WebPage("http://ccphillips.net/")
>>> now = time.time()
>>> content1 = webpage.content
Retrieving New Page...
>>> time.time() - now
22.43316888809204
>>> now = time.time()
>>> content2 = webpage.content
>>> time.time() - now
1.9266459941864014
>>> content2 == content1
True  
```

我在2010年首次测试这段代码时使用的是糟糕的卫星连接，第一次加载内容花了20秒。第二次，我在2秒内得到了结果（实际上只是在解释器中输入这些行所花费的时间）。在我更现代的连接上，情况如下：

```py
>>> webpage = WebPage("https://dusty.phillips.codes")
>>> import time
>>> now = time.time() ; content1 = webpage.content ; print(time.time() - now)
Retrieving New Page...
0.6236202716827393
>>> now = time.time() ; content2 = webpage.content ; print(time.time() - now)
1.7881393432617188e-05M
```

从我的网络主机检索页面大约需要620毫秒。从我的笔记本电脑的RAM中，只需要0.018毫秒！

自定义getter也适用于需要根据其他对象属性动态计算的属性。例如，我们可能想要计算整数列表的平均值：

```py
class AverageList(list): 
    @property 
    def average(self): 
        return sum(self) / len(self) 
```

这个非常简单的类继承自`list`，所以我们可以免费获得类似列表的行为。我们只需向类添加一个属性，就可以得到列表的平均值。

```py
>>> a = AverageList([1,2,3,4])
>>> a.average
2.5  
```

当然，我们也可以将其制作成一个方法，但那样我们应该将其命名为`calculate_average()`，因为方法代表动作。但名为`average`的属性更合适，而且更容易输入和阅读。

自定义setter对于验证是有用的，正如我们已经看到的，但它们也可以用于将值代理到另一个位置。例如，我们可以为`WebPage`类添加一个内容setter，以便在设置值时自动登录到我们的Web服务器并上传新页面。

# 管理对象

我们一直专注于对象及其属性和方法。现在，我们将看看如何设计更高级的对象；管理其他对象的对象 - 将所有东西联系在一起的对象。

这些对象与大多数先前的示例之间的区别在于，后者通常代表具体的想法。管理对象更像办公室经理；他们不会在现场进行实际的*可见*工作，但没有他们，部门之间就不会有沟通，也没有人知道他们应该做什么（尽管如果组织管理不善，这也可能是真的！）。类似地，管理类上的属性倾向于引用做*可见*工作的其他对象；这样的类上的行为在适当的时候委托给这些其他类，并在它们之间传递消息。

例如，我们将编写一个程序，对存储在压缩的ZIP文件中的文本文件执行查找和替换操作。我们需要对象来表示ZIP文件和每个单独的文本文件（幸运的是，我们不必编写这些类，因为它们在Python标准库中可用）。管理对象将负责确保以下三个步骤按顺序发生：

1.  解压缩压缩文件

1.  执行查找和替换操作

1.  压缩新文件

该类使用`.zip`文件名、搜索和替换字符串进行初始化。我们创建一个临时目录来存储解压后的文件，以便文件夹保持干净。`pathlib`库在文件和目录操作中提供帮助。接口在以下示例中应该很清楚：

```py
import sys 
import shutil 
import zipfile 
from pathlib import Path 

class ZipReplace: 
    def __init__(self, filename, search_string, replace_string): 
        self.filename = filename 
        self.search_string = search_string 
        self.replace_string = replace_string 
        self.temp_directory = Path(f"unzipped-{filename}")
```

然后，我们为三个步骤创建一个整体*管理*方法。该方法将责任委托给其他对象：

```py
def zip_find_replace(self): 
    self.unzip_files() 
    self.find_replace() 
    self.zip_files() 
```

显然，我们可以在一个方法中完成所有三个步骤，或者在一个脚本中完成，而不必创建对象。将三个步骤分开有几个优点：

+   可读性：每个步骤的代码都在一个易于阅读和理解的自包含单元中。方法名称描述了方法的作用，需要更少的额外文档来理解正在发生的事情。

+   可扩展性：如果子类想要使用压缩的TAR文件而不是ZIP文件，它可以重写`zip`和`unzip`方法，而无需复制`find_replace`方法。

+   分区：外部类可以创建此类的实例，并在不必`zip`内容的情况下直接在某个文件夹上调用`find_replace`方法。

委托方法是以下代码中的第一个；其余方法包括在内是为了完整性：

```py
    def unzip_files(self):
        self.temp_directory.mkdir()
        with zipfile.ZipFile(self.filename) as zip:
            zip.extractall(self.temp_directory)

    def find_replace(self):
        for filename in self.temp_directory.iterdir():
            with filename.open() as file:
                contents = file.read()
            contents = contents.replace(self.search_string, self.replace_string)
            with filename.open("w") as file:
                file.write(contents)

    def zip_files(self):
        with zipfile.ZipFile(self.filename, "w") as file:
            for filename in self.temp_directory.iterdir():
                file.write(filename, filename.name)
        shutil.rmtree(self.temp_directory)

if __name__ == "__main__":
    ZipReplace(*sys.argv[1:4]).zip_find_replace()
```

为了简洁起见，对于压缩和解压缩文件的代码文档很少。我们目前关注的是面向对象的设计；如果您对`zipfile`模块的内部细节感兴趣，请参考标准库中的文档，可以在线查看，也可以在交互式解释器中输入`import zipfile ; help(zipfile)`。请注意，此玩具示例仅搜索ZIP文件中的顶层文件；如果解压后的内容中有任何文件夹，它们将不会被扫描，也不会扫描这些文件夹中的任何文件。

如果您使用的是早于3.6的Python版本，则需要在调用`ZipFile`对象上的`extractall`、`rmtree`和`file.write`之前将路径对象转换为字符串。

示例中的最后两行允许我们通过传递`zip`文件名、搜索字符串和替换字符串作为参数来从命令行运行程序，如下所示：

```py
$python zipsearch.py hello.zip hello hi  
```

当然，这个对象不一定要从命令行创建；它可以从另一个模块导入（执行批量ZIP文件处理），或者作为GUI界面的一部分访问，甚至作为一个更高级别的管理对象的一部分，该对象知道从哪里获取ZIP文件（例如，从FTP服务器检索它们或将它们备份到外部磁盘）。

随着程序变得越来越复杂，被建模的对象变得越来越不像物理对象。属性是其他抽象对象，方法是改变这些抽象对象状态的行为。但无论多么复杂，每个对象的核心都是一组具体数据和明确定义的行为。

# 删除重复的代码

通常，诸如`ZipReplace`之类的管理样式类中的代码非常通用，可以以各种方式应用。可以使用组合或继承来帮助将此代码放在一个地方，从而消除重复代码。在我们查看任何此类示例之前，让我们讨论一点理论。具体来说，为什么重复代码是一件坏事？

有几个原因，但归根结底都是可读性和可维护性。当我们编写类似于早期代码的新代码时，最容易的方法是复制旧代码并更改需要更改的内容（变量名称、逻辑、注释），使其在新位置上运行。或者，如果我们正在编写似乎类似但不完全相同的新代码，与项目中的其他代码相比，通常更容易编写具有类似行为的新代码，而不是弄清楚如何提取重叠功能。

但是，一旦有人阅读和理解代码，并且遇到重复的代码块，他们就面临着两难境地。可能看起来有意义的代码突然必须被理解。一个部分与另一个部分有何不同？它们如何相同？在什么条件下调用一个部分？我们什么时候调用另一个部分？你可能会争辩说你是唯一阅读你的代码的人，但是如果你八个月不碰那段代码，它对你来说将和对一个新手编程人员一样难以理解。当我们试图阅读两个相似的代码部分时，我们必须理解它们为何不同，以及它们如何不同。这浪费了读者的时间；代码应始终被编写为首要可读性。

我曾经不得不尝试理解某人的代码，其中有三个完全相同的300行非常糟糕的代码副本。在我最终理解这三个*相同*版本实际上执行略有不同的税收计算之前，我已经与这段代码一起工作了一个月。一些微妙的差异是有意的，但也有明显的地方，某人在一个函数中更新了一个计算，而没有更新其他两个。代码中难以理解的微妙错误数量不计其数。最终，我用一个大约20行的易于阅读的函数替换了所有900行。

阅读这样的重复代码可能很烦人，但代码维护更加痛苦。正如前面的故事所示，保持两个相似的代码部分最新可能是一场噩梦。每当我们更新其中一个部分时，我们必须记住更新两个部分，并且我们必须记住多个部分的不同之处，以便在编辑每个部分时修改我们的更改。如果我们忘记更新所有部分，我们最终会遇到非常恼人的错误，通常表现为“但我已经修复了，为什么还在发生*？”

结果是，阅读或维护我们的代码的人们必须花费天文数字的时间来理解和测试它，而不是在第一次编写时以非重复的方式编写它所需的时间。当我们自己进行维护时，这更加令人沮丧；我们会发现自己说，“为什么我第一次就没做对呢？”通过复制和粘贴现有代码节省的时间在第一次进行维护时就丢失了。代码被阅读和修改的次数比编写的次数多得多，而且频率也更高。可理解的代码应始终是优先考虑的。

这就是为什么程序员，尤其是Python程序员（他们倾向于比普通开发人员更重视优雅的代码），遵循所谓的**不要重复自己**（**DRY**）原则。DRY代码是可维护的代码。我给初学者的建议是永远不要使用编辑器的复制粘贴功能。对于中级程序员，我建议他们在按下*Ctrl* + *C*之前三思。

但是，我们应该怎么做才能避免代码重复呢？最简单的解决方案通常是将代码移到一个函数中，该函数接受参数以解决不同的部分。这不是一个非常面向对象的解决方案，但通常是最佳的解决方案。

例如，如果我们有两段代码，它们将ZIP文件解压缩到两个不同的目录中，我们可以很容易地用一个接受目录参数的函数来替换它。这可能会使函数本身稍微难以阅读，但一个好的函数名称和文档字符串很容易弥补这一点，任何调用该函数的代码都会更容易阅读。

这就足够的理论了！故事的寓意是：始终努力重构代码，使其更易读，而不是编写可能看起来更容易的糟糕代码。

# 在实践中

让我们探讨两种重用现有代码的方法。在编写代码以替换ZIP文件中的文本文件中的字符串后，我们后来受托将ZIP文件中的所有图像缩放到640 x 480。看起来我们可以使用与我们在`ZipReplace`中使用的非常相似的范例。我们的第一反应可能是保存该文件的副本，并将`find_replace`方法更改为`scale_image`或类似的内容。

但是，这是次优的。如果有一天我们想要更改`unzip`和`zip`方法以打开TAR文件呢？或者也许我们想要为临时文件使用一个保证唯一的目录名称。在任何一种情况下，我们都必须在两个不同的地方进行更改！

我们将从展示基于继承的解决方案开始解决这个问题。首先，我们将修改我们原始的`ZipReplace`类，将其变成一个用于处理通用ZIP文件的超类：

```py
import sys
import shutil
import zipfile
from pathlib import Path

class ZipProcessor:
    def __init__(self, zipname):
        self.zipname = zipname
        self.temp_directory = Path(f"unzipped-{zipname[:-4]}")

    def process_zip(self):
        self.unzip_files()
        self.process_files()
        self.zip_files()

    def unzip_files(self):
        self.temp_directory.mkdir()
        with zipfile.ZipFile(self.zipname) as zip:
            zip.extractall(self.temp_directory)

    def zip_files(self):
        with zipfile.ZipFile(self.zipname, "w") as file:
            for filename in self.temp_directory.iterdir():
                file.write(filename, filename.name)
        shutil.rmtree(self.temp_directory)
```

我们将`filename`属性更改为`zipname`，以避免与各种方法内部的`filename`本地变量混淆。这有助于使代码更易读，尽管实际上并没有改变设计。

我们还删除了`__init__`中的两个参数（`search_string`和`replace_string`），这些参数是特定于`ZipReplace`的。然后，我们将`zip_find_replace`方法重命名为`process_zip`，并让它调用一个（尚未定义的）`process_files`方法，而不是`find_replace`；这些名称更改有助于展示我们新类的更一般化特性。请注意，我们已经完全删除了`find_replace`方法；该代码是特定于`ZipReplace`，在这里没有业务。

这个新的`ZipProcessor`类实际上并没有定义`process_files`方法。如果我们直接运行它，它会引发异常。因为它不是用来直接运行的，我们删除了原始脚本底部的主要调用。我们可以将其作为抽象基类，以便传达这个方法需要在子类中定义，但出于简洁起见，我将其省略了。

现在，在我们转向图像处理应用程序之前，让我们修复我们原始的`zipsearch`类，以利用这个父类，如下所示：

```py
class ZipReplace(ZipProcessor):
    def __init__(self, filename, search_string, replace_string):
        super().__init__(filename)
        self.search_string = search_string
        self.replace_string = replace_string

    def process_files(self):
        """perform a search and replace on all files in the
        temporary directory"""
        for filename in self.temp_directory.iterdir():
            with filename.open() as file:
                contents = file.read()
            contents = contents.replace(self.search_string, self.replace_string)
            with filename.open("w") as file:
                file.write(contents)
```

这段代码比原始版本要短，因为它继承了父类的ZIP处理能力。我们首先导入我们刚刚编写的基类，并使`ZipReplace`扩展该类。然后，我们使用`super()`来初始化父类。`find_replace`方法仍然存在，但我们将其重命名为`process_files`，以便父类可以从其管理界面调用它。因为这个名称不像旧名称那样描述性强，我们添加了一个文档字符串来描述它正在做什么。

现在，考虑到我们现在所做的工作量相当大，而我们现在的程序在功能上与我们开始的程序并无不同！但是经过这样的工作，我们现在可以更容易地编写其他操作ZIP存档文件的类，比如（假设请求的）照片缩放器。此外，如果我们想要改进或修复ZIP功能，我们只需更改一个`ZipProcessor`基类，就可以同时为所有子类进行操作。因此维护工作将更加有效。

看看现在创建一个利用`ZipProcessor`功能的照片缩放类有多简单：

```py
from PIL import Image 

class ScaleZip(ZipProcessor): 

    def process_files(self): 
        '''Scale each image in the directory to 640x480''' 
        for filename in self.temp_directory.iterdir(): 
            im = Image.open(str(filename)) 
            scaled = im.resize((640, 480)) 
            scaled.save(filename)

if __name__ == "__main__": 
    ScaleZip(*sys.argv[1:4]).process_zip() 
```

看看这个类有多简单！我们之前所做的所有工作都得到了回报。我们所做的就是打开每个文件（假设它是一个图像；如果文件无法打开或不是图像，程序将崩溃），对其进行缩放，然后保存。`ZipProcessor`类负责压缩和解压，而我们无需额外工作。

# 案例研究

对于这个案例研究，我们将尝试进一步探讨一个问题，即何时应该选择对象而不是内置类型？我们将建模一个可能在文本编辑器或文字处理器中使用的`Document`类。它应该具有哪些对象、函数或属性？

我们可能会从`Document`内容开始使用`str`，但在Python中，字符串是不可变的。一旦定义了一个`str`，它就永远存在。我们无法在其中插入字符或删除字符，而不创建全新的字符串对象。这将导致大量的`str`对象占用内存，直到Python的垃圾收集器决定清理它们。

因此，我们将使用字符列表而不是字符串，这样我们可以随意修改。此外，我们需要知道列表中的当前光标位置，并且可能还需要存储文档的文件名。

真正的文本编辑器使用一种名为`rope`的基于二叉树的数据结构来模拟其文档内容。本书的标题不是*高级数据结构*，所以如果你对这个迷人的主题感兴趣，你可能想在网上搜索*rope数据结构*了解更多信息。

我们可能想对文本文档进行许多操作，包括插入、删除和选择字符；剪切、复制和粘贴所选内容；以及保存或关闭文档。看起来有大量的数据和行为，因此将所有这些内容放入自己的`Document`类是有道理的。

一个相关的问题是：这个类应该由一堆基本的Python对象组成，比如`str`文件名、`int`光标位置和字符的`list`？还是应该将其中一些或全部内容定义为自己的特定对象？单独的行和字符呢？它们需要有自己的类吗？

我们将在进行过程中回答这些问题，但让我们先从最简单的`Document`类开始，看看它能做什么：

```py
class Document: 
    def __init__(self): 
        self.characters = [] 
        self.cursor = 0 
        self.filename = '' 

    def insert(self, character): 
        self.characters.insert(self.cursor, character) 
        self.cursor += 1 

    def delete(self): 
        del self.characters[self.cursor] 

    def save(self): 
        with open(self.filename, 'w') as f: 
            f.write(''.join(self.characters)) 

    def forward(self): 
        self.cursor += 1 

    def back(self): 
        self.cursor -= 1 
```

这个基本类允许我们完全控制编辑基本文档。看看它的运行情况：

```py
>>> doc = Document()
>>> doc.filename = "test_document"
>>> doc.insert('h')
>>> doc.insert('e')
>>> doc.insert('l')
>>> doc.insert('l')
>>> doc.insert('o')
>>> "".join(doc.characters)
'hello'
>>> doc.back()
>>> doc.delete()
>>> doc.insert('p')
>>> "".join(doc.characters)
'hellp'  
```

看起来它正在工作。我们可以将键盘的字母和箭头键连接到这些方法，文档将正常跟踪一切。

但是，如果我们想要连接的不仅仅是箭头键。如果我们还想连接*Home*和*End*键怎么办？我们可以向`Document`类添加更多方法，用于在字符串中向前或向后搜索换行符（换行符，转义为`\n`，表示一行的结束和新行的开始），并跳转到它们，但如果我们为每种可能的移动操作（按单词移动，按句子移动，*Page Up*，*Page Down*，行尾，空格开头等）都这样做，那么这个类将会很庞大。也许把这些方法放在一个单独的对象上会更好。因此，让我们将`Cursor`属性转换为一个对象，该对象知道自己的位置并可以操纵该位置。我们可以将向前和向后的方法移到该类中，并为*Home*和*End*键添加另外两个方法，如下所示：

```py
class Cursor:
    def __init__(self, document):
        self.document = document
        self.position = 0

    def forward(self):
        self.position += 1

    def back(self):
        self.position -= 1

    def home(self):
        while self.document.characters[self.position - 1].character != "\n":
            self.position -= 1
            if self.position == 0:
                # Got to beginning of file before newline
                break

    def end(self):
        while (
            self.position < len(self.document.characters)
            and self.document.characters[self.position] != "\n"
        ):
            self.position += 1
```

这个类将文档作为初始化参数，以便方法可以访问文档字符列表的内容。然后提供了向后和向前移动的简单方法，以及移动到`home`和`end`位置的方法。

这段代码并不是很安全。你很容易越过结束位置，如果你试图在空文件上回到开头，它会崩溃。这些示例被保持简短以便阅读，但这并不意味着它们是防御性的！你可以通过练习来改进这段代码的错误检查；这可能是扩展你的异常处理技能的绝佳机会。

`Document`类本身几乎没有改变，只是删除了移动到`Cursor`类的两个方法：

```py
class Document: 
    def __init__(self): 
        self.characters = [] 
        self.cursor = Cursor(self) 
        self.filename = '' 

       def insert(self, character): 
        self.characters.insert(self.cursor.position, 
                character) 
        self.cursor.forward() 

    def delete(self): 
        del self.characters[self.cursor.position] 

    def save(self):
        with open(self.filename, "w") as f:
            f.write("".join(self.characters))
```

我们刚刚更新了访问旧光标整数的任何内容，以使用新对象代替。我们现在可以测试`home`方法是否真的移动到换行符，如下所示：

```py
>>> d = Document()
>>> d.insert('h')
>>> d.insert('e')
>>> d.insert('l')
>>> d.insert('l')
>>> d.insert('o')
>>> d.insert('\n')
>>> d.insert('w')
>>> d.insert('o')
>>> d.insert('r')
>>> d.insert('l')
>>> d.insert('d')
>>> d.cursor.home()
>>> d.insert("*")
>>> print("".join(d.characters))
hello
*world  
```

现在，由于我们一直在大量使用字符串`join`函数（将字符连接起来，以便查看实际文档内容），我们可以向`Document`类添加一个属性，以便得到完整的字符串，如下所示：

```py
@property 
def string(self): 
    return "".join(self.characters) 
```

这使得我们的测试变得更简单：

```py
>>> print(d.string)
hello
world  
```

这个框架很容易扩展，创建和编辑完整的纯文本文档（尽管可能会有点耗时！）现在，让我们将其扩展到适用于富文本的工作；可以具有**粗体**、下划线或*斜体*字符的文本。

我们可以以两种方式处理这个问题。第一种是在字符列表中插入*虚假*字符，这些字符就像指令一样，比如*粗体字符，直到找到停止粗体字符*。第二种是向每个字符添加信息，指示它应该具有什么格式。虽然前一种方法在真实编辑器中更常见，但我们将实现后一种解决方案。为此，我们显然需要一个字符类。这个类将具有表示字符的属性，以及三个布尔属性，表示它是否*粗体、斜体或下划线*。

嗯，等等！这个`Character`类会有任何方法吗？如果没有，也许我们应该使用许多Python数据结构之一；元组或命名元组可能就足够了。有没有任何操作我们想要在字符上执行或调用？

嗯，显然，我们可能想对字符进行一些操作，比如删除或复制它们，但这些是需要在`Document`级别处理的事情，因为它们实际上是在修改字符列表。是否有需要对单个字符进行处理的事情？

实际上，现在我们在思考`Character`类实际上**是**什么……它是什么？可以肯定地说`Character`类是一个字符串吗？也许我们应该在这里使用继承关系？然后我们可以利用`str`实例带来的众多方法。

我们在谈论什么样的方法？有`startswith`、`strip`、`find`、`lower`等等。这些方法中的大多数都希望在包含多个字符的字符串上工作。相比之下，如果`Character`是`str`的子类，我们可能最好重写`__init__`，以便在提供多字符字符串时引发异常。由于我们将免费获得的所有这些方法实际上并不适用于我们的`Character`类，因此似乎我们不应该使用继承。

这让我们回到了最初的问题；`Character`甚至应该是一个类吗？`object`类上有一个非常重要的特殊方法，我们可以利用它来表示我们的字符。这个方法叫做`__str__`（两端都有两个下划线，就像`__init__`一样），它在字符串操作函数中被使用，比如`print`和`str`构造函数，将任何类转换为字符串。默认实现做了一些无聊的事情，比如打印模块和类的名称，以及它在内存中的地址。但如果我们重写它，我们可以让它打印任何我们喜欢的东西。

对于我们的实现，我们可以使用特殊字符作为前缀来表示字符是否为粗体、斜体或下划线。因此，我们将创建一个表示字符的类，如下所示：

```py
class Character: 
    def __init__(self, character, 
            bold=False, italic=False, underline=False): 
        assert len(character) == 1 
        self.character = character 
        self.bold = bold 
        self.italic = italic 
        self.underline = underline 

    def __str__(self): 
        bold = "*" if self.bold else '' 
        italic = "/" if self.italic else '' 
        underline = "_" if self.underline else '' 
        return bold + italic + underline + self.character 
```

这个类允许我们创建字符，并在应用`str()`函数时用特殊字符作为前缀。没有太多激动人心的地方。我们只需要对`Document`和`Cursor`类进行一些小修改，以便与这个类一起工作。在`Document`类中，我们在`insert`方法的开头添加以下两行：

```py
def insert(self, character): 
    if not hasattr(character, 'character'): 
        character = Character(character) 
```

这是一段相当奇怪的代码。它的基本目的是检查传入的字符是`Character`还是`str`。如果是字符串，它就会被包装在`Character`类中，以便列表中的所有对象都是`Character`对象。然而，完全有可能有人使用我们的代码想要使用既不是`Character`也不是字符串的类，使用鸭子类型。如果对象有一个字符属性，我们假设它是类似`Character`的对象。但如果没有，我们假设它是类似`str`的对象，并将其包装在`Character`中。这有助于程序利用鸭子类型和多态性；只要对象具有字符属性，它就可以在`Document`类中使用。

这种通用检查可能非常有用。例如，如果我们想要制作一个带有语法高亮的程序员编辑器，我们需要字符的额外数据，比如字符属于哪种类型的语法标记。请注意，如果我们要做很多这种比较，最好实现`Character`作为一个带有适当`__subclasshook__`的抽象基类，如[第17章](63f74e80-a17c-456d-969e-2a36ffb0067b.xhtml)中讨论的那样，*当对象相似*。

此外，我们需要修改`Document`上的字符串属性，以接受新的`Character`值。我们只需要在连接之前对每个字符调用`str()`，如下所示：

```py
    @property 
    def string(self): 
        return "".join((str(c) for c in self.characters)) 
```

这段代码使用了一个生成器表达式，我们将在[第21章](b9232138-1747-4f88-b7ac-002c40332e92.xhtml)中讨论，*迭代器模式*。这是一个在序列中对所有对象执行特定操作的快捷方式。

最后，我们还需要检查`home`和`end`函数中的`Character.character`，而不仅仅是我们之前存储的字符串字符，看它是否匹配换行符，如下所示：

```py
    def home(self): 
        while self.document.characters[ 
                self.position-1].character != '\n': 
            self.position -= 1 
            if self.position == 0: 
                # Got to beginning of file before newline 
                break 

    def end(self): 
        while self.position < len( 
                self.document.characters) and \ 
                self.document.characters[ 
                        self.position 
                        ].character != '\n': 
            self.position += 1
```

这完成了字符的格式化。我们可以测试它，看它是否像下面这样工作：

```py
>>> d = Document()
>>> d.insert('h')
>>> d.insert('e')
>>> d.insert(Character('l', bold=True))
>>> d.insert(Character('l', bold=True))
>>> d.insert('o')
>>> d.insert('\n')
>>> d.insert(Character('w', italic=True))
>>> d.insert(Character('o', italic=True))
>>> d.insert(Character('r', underline=True))
>>> d.insert('l')
>>> d.insert('d')
>>> print(d.string)
he*l*lo
/w/o_rld
>>> d.cursor.home()
>>> d.delete()
>>> d.insert('W')
>>> print(d.string)
he*l*lo
W/o_rld
>>> d.characters[0].underline = True
>>> print(d.string)
_he*l*lo
W/o_rld  
```

正如预期的那样，每当我们打印字符串时，每个粗体字符前面都有一个`*`字符，每个斜体字符前面都有一个`/`字符，每个下划线字符前面都有一个`_`字符。我们所有的函数似乎都能工作，并且我们可以在事后修改列表中的字符。我们有一个可以插入到适当的图形用户界面中并与键盘进行输入和屏幕进行输出的工作的富文本文档对象。当然，我们希望在UI中显示真正的*粗体、斜体和下划线*字体，而不是使用我们的`__str__`方法，但它对我们要求的基本测试是足够的。

# 练习

我们已经看过了在面向对象的Python程序中对象、数据和方法可以相互交互的各种方式。和往常一样，您的第一个想法应该是如何将这些原则应用到您自己的工作中。您是否有一些混乱的脚本横七竖八地散落在那里，可以使用面向对象的管理器进行重写？浏览一下您的旧代码，寻找一些不是动作的方法。如果名称不是动词，尝试将其重写为属性。

思考您用任何语言编写的代码。它是否违反了DRY原则？是否有任何重复的代码？您是否复制和粘贴了代码？您是否编写了两个类似代码的版本，因为您不想理解原始代码？现在回顾一下您最近的一些代码，看看是否可以使用继承或组合重构重复的代码。尝试选择一个您仍然有兴趣维护的项目；不要选择那些您永远不想再碰的代码。这将有助于在您进行改进时保持您的兴趣！

现在，回顾一下本章中我们看过的一些例子。从使用属性缓存检索数据的缓存网页示例开始。这个示例的一个明显问题是缓存从未被刷新。在属性的getter中添加一个超时，并且只有在页面在超时过期之前被请求时才返回缓存的页面。您可以使用`time`模块（`time.time() - an_old_time`返回自`an_old_time`以来经过的秒数）来确定缓存是否已过期。

还要看看基于继承的`ZipProcessor`。在这里使用组合而不是继承可能是合理的。您可以在`ZipProcessor`构造函数中传递这些类的实例，并调用它们来执行处理部分。实现这一点。

您觉得哪个版本更容易使用？哪个更优雅？哪个更容易阅读？这些都是主观问题；答案因人而异。然而，了解答案是重要的。如果您发现自己更喜欢继承而不是组合，那么您需要注意不要在日常编码中过度使用继承。如果您更喜欢组合，请确保不要错过创建优雅的基于继承的解决方案的机会。

最后，在案例研究中为各种类添加一些错误处理程序。它们应确保输入单个字符，不要尝试将光标移动到文件的末尾或开头，不要删除不存在的字符，也不要保存没有文件名的文件。尽量考虑尽可能多的边缘情况，并对其进行考虑（考虑边缘情况大约占专业程序员工作的90%！）。考虑不同的处理方式；当用户尝试移动到文件末尾时，您应该引发异常，还是只停留在最后一个字符？

在您的日常编码中，注意复制和粘贴命令。每次在编辑器中使用它们时，考虑是否改进程序的组织结构，以便您只有要复制的代码的一个版本。

# 总结

在这一章中，我们专注于识别对象，特别是那些不太明显的对象；管理和控制对象。对象应该既有数据又有行为，但属性可以用来模糊两者之间的区别。 DRY 原则是代码质量的重要指标，继承和组合可以用来减少代码重复。

在下一章中，我们将讨论如何整合 Python 的面向对象和非面向对象的方面。在这个过程中，我们会发现它比起初看起来更加面向对象！
