# 何时使用面向对象编程

在前几章中，我们介绍了面向对象编程的许多定义特征。我们现在知道了面向对象设计的原理和范式，并且我们已经介绍了 Python 中面向对象编程的语法。

然而，我们并不知道确切如何以及特别是在实践中何时利用这些原则和语法。在本章中，我们将讨论一些有用知识的应用，沿途探讨一些新主题：

+   如何识别对象

+   数据和行为，再次强调

+   使用属性包装数据行为

+   使用行为限制数据

+   不要重复原则

+   识别重复的代码

# 将对象视为对象

这可能看起来很明显；你应该通常给你的问题域中的单独对象在代码中一个特殊的类。我们在前几章的案例研究中看到了这样的例子：首先，我们识别问题中的对象，然后建模它们的数据和行为。

识别对象是面向对象分析和编程中一个非常重要的任务。但这并不总是像在简短的段落中数名词那么简单，坦白说，我为此目的明确地构建了这些段落。记住，对象是既有数据又有行为的事物。如果我们只处理数据，我们通常最好将其存储在列表、集合、字典或其他 Python 数据结构中（我们将在第六章，*Python 数据结构*中彻底介绍）。另一方面，如果我们只处理行为，但没有存储的数据，一个简单的函数就更为合适。

然而，一个对象既有数据也有行为。熟练的 Python 程序员会使用内置的数据结构，除非（或直到）有明显的需要定义一个类。如果没有帮助组织我们的代码，就没有理由添加额外的抽象层次。另一方面，*明显*的需要并不总是显而易见的。

我们经常可以通过在几个变量中存储数据来开始我们的 Python 程序。随着程序的扩展，我们稍后会发现我们将同一组相关的变量传递给一组函数。这时，我们应该考虑将变量和函数分组到一个类中。如果我们正在设计一个用于模拟二维空间中多边形的程序，我们可能从每个多边形表示为点的列表开始。这些点将被建模为描述该点位置的(*x*, *y*)两个元组。这全是数据，存储在一系列嵌套的数据结构中（具体来说，是一个元组的列表）：

```py
square = [(1,1), (1,2), (2,2), (2,1)] 
```

现在，如果我们想要计算多边形周长的距离，我们需要求出每个点之间的距离之和。为此，我们需要一个函数来计算两点之间的距离。这里有两个这样的函数：

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

现在，作为面向对象的程序员，我们明显认识到`polygon`类可以封装点列表（数据）和`perimeter`函数（行为）。此外，一个`point`类，例如我们在第二章“Python 中的对象”中定义的，可以封装`x`和`y`坐标以及`distance`方法。问题是：这样做有价值吗？

对于之前的代码，也许是的，也许不是。凭借我们最近在面向对象原则方面的经验，我们可以快速编写面向对象的版本。让我们如下进行比较：

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

如我们从突出显示的部分中可以看到，这里的代码量是我们早期版本的两倍，尽管我们可以争论`add_point`方法并不是严格必要的。

现在，为了更好地理解两者之间的差异，让我们比较一下两个 API 的使用情况。以下是使用面向对象代码计算正方形周长的方法：

```py
>>> square = Polygon()
>>> square.add_point(Point(1,1))
>>> square.add_point(Point(1,2))
>>> square.add_point(Point(2,2))
>>> square.add_point(Point(2,1))
>>> square.perimeter()
4.0  
```

这看起来相当简洁且易于阅读，你可能会这样认为，但让我们将其与基于函数的代码进行比较：

```py
>>> square = [(1,1), (1,2), (2,2), (2,1)]
>>> perimeter(square)
4.0  
```

嗯，也许面向对象的 API 并不那么紧凑！话虽如此，我认为它比函数示例更容易*阅读*。我们如何知道在第二个版本中元组的列表应该代表什么？我们如何记住我们应该传递给`perimeter`函数的对象类型？（一个包含两个元组的列表？这并不直观！）我们需要大量的文档来解释这些函数应该如何使用。

相比之下，面向对象的代码相对自文档化。我们只需查看方法列表及其参数，就可以知道对象的功能以及如何使用它。等到我们为函数版本编写完所有文档，它可能比面向对象的代码还要长。

最后，代码长度并不是代码复杂性的良好指标。一些程序员会陷入复杂的*一行代码*中，一行代码就能完成大量的工作。这可能是一项有趣的练习，但结果往往是难以阅读的，即使是原作者第二天再看也会如此。尽量减少代码量通常可以使程序更容易阅读，但不要盲目地假设这是正确的。

幸运的是，这种权衡是不必要的。我们可以使面向对象的`Polygon` API 与函数实现一样易于使用。我们只需要修改我们的`Polygon`类，使其可以用多个点来构造。让我们给它一个接受`Point`对象列表的初始化器。实际上，让我们允许它接受元组，如果需要，我们还可以自己构造`Point`对象：

```py
def __init__(self, points=None): 
    points = points if points else [] 
    self.vertices = [] 
    for point in points: 
        if isinstance(point, tuple): 
            point = Point(*point) 
        self.vertices.append(point) 
```

这个初始化器会遍历列表，并确保将任何元组转换为点。如果对象不是元组，我们就保持原样，假设它要么是一个已经存在的`Point`对象，要么是一个可以像`Point`对象一样操作的未知鸭子类型对象。

如果你正在尝试上述代码，你可以通过子类化`Polygon`并覆盖`__init__`函数来代替替换初始化器或复制`add_point`和`perimeter`方法。

然而，在面向对象和更多数据导向的代码版本之间，并没有明显的胜者。它们都做了同样的事情。如果我们有接受多边形参数的新函数，如`area(polygon)`或`point_in_polygon(polygon, x, y)`，面向对象代码的好处将越来越明显。同样，如果我们给多边形添加其他属性，如`color`或`texture`，将数据封装到单个类中会更有意义。

这种区别是一个设计决策，但一般来说，一组数据越重要，就越有可能有针对该数据的多个特定函数，使用具有属性和方法类的用途就越大。

在做出这个决定时，考虑类将如何被使用也是有帮助的。如果我们只是在更大的问题背景下尝试计算一个多边形的周长，使用一个函数可能编写起来最快，也更容易一次性使用。另一方面，如果我们的程序需要以多种方式操纵大量多边形（计算周长、面积、与其他多边形的交集、移动或缩放等），我们几乎肯定已经识别出一个对象；一个需要极其灵活的对象。

此外，请注意对象之间的交互。寻找继承关系；没有类，继承是无法优雅建模的，所以请确保使用它们。寻找我们在第一章“面向对象设计”中讨论的其他类型的关系，如关联和组合。在技术上，组合可以使用仅数据结构来建模；例如，我们可以有一个包含元组值的字典列表，但有时创建几个具有与数据相关行为的对象类会更简单。

不要因为可以使用对象就急于使用对象，但当你需要使用类时，也不要忽视创建一个类。

# 将行为添加到具有属性的类数据中

在整本书中，我们一直关注行为和数据分离。这在面向对象编程中非常重要，但我们将看到，在 Python 中，这种区别非常模糊。Python 非常擅长模糊化区别；它并不真正帮助我们跳出思维定式。相反，它教会我们停止考虑思维定式。

在我们深入细节之前，让我们讨论一些不好的面向对象理论。许多面向对象的语言教导我们永远不要直接访问属性（Java 是最臭名昭著的）。他们坚持认为我们应该这样编写属性访问：

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

变量以下划线为前缀，以表明它们是私有的（其他语言实际上会强制它们成为私有）。然后，`get`和`set`方法提供了对每个变量的访问。这个类在实际应用中的使用如下：

```py
>>> c = Color("#ff0000", "bright red")
>>> c.get_name()
'bright red'
>>> c.set_name("red")
>>> c.get_name()
'red'  
```

这在可读性上远不如 Python 所青睐的直接访问版本：

```py
class Color: 
    def __init__(self, rgb_value, name): 
        self.rgb_value = rgb_value 
        self.name = name 

c = Color("#ff0000", "bright red") 
print(c.name) c.name = "red"
print(c.name)
```

那么，为什么有人会坚持使用基于方法的语法呢？他们的理由是，总有一天，我们可能想在设置或检索值时添加额外的代码。例如，我们可以决定缓存一个值以避免复杂的计算，或者我们可能想验证给定的值是否是一个合适的输入。

例如，在代码中，我们可以决定将`set_name()`方法更改为以下内容：

```py
def set_name(self, name): 
    if not name: 
        raise Exception("Invalid Name") 
    self._name = name 
```

现在，在 Java 和类似的语言中，如果我们最初为直接属性访问编写了原始代码，然后后来将其更改为前面提到的方法，我们会遇到问题：任何编写了直接访问属性代码的人现在都必须访问一个方法。如果他们没有将访问样式从属性访问更改为函数调用，他们的代码就会出错。

这些语言中的格言是，我们永远不应该将公共成员设置为私有。在 Python 中这并没有太多意义，因为 Python 中并没有真正的私有成员概念！

Python 为我们提供了`property`关键字来创建看起来像属性的函数。因此，我们可以编写代码以使用直接成员访问，如果我们需要意外地更改实现以在获取或设置该属性的值时进行一些计算，我们可以这样做而不必更改接口。让我们看看它看起来如何：

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

与早期版本相比，我们首先将`name`属性更改为（半）私有属性`_name`。然后，我们添加了两个更多（半）私有方法来获取和设置该变量，在我们设置它时执行验证。

最后，我们在底部有`property`声明。这是 Python 的魔法。它为`Color`类创建了一个名为`name`的新属性，以替换直接的`name`属性。它将此属性设置为**属性**。在底层，`property`在访问或更改值时调用我们刚刚创建的两个方法。这个`Color`类的新版本可以像早期版本一样使用，但现在在设置`name`属性时它将执行验证：

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

因此，如果我们之前编写了访问`name`属性的代码，然后将其更改为使用我们的基于`property`的对象，之前的代码仍然会工作，除非它发送了一个空的`property`值，这正是我们最初想要禁止的行为。成功了！

请记住，即使有了`name`属性，之前的代码也不是 100%安全的。人们仍然可以直接访问`_name`属性并将其设置为空字符串，如果他们想这么做的话。但如果他们访问了我们明确标记为下划线以表明它是私有的变量，那么他们必须承担后果，而不是我们。

# 属性的详细说明

将 `property` 函数想象成返回一个对象，该对象通过我们指定的方法代理对属性值的设置或访问请求。`property` 内置函数就像这样一个对象的构造函数，并且该对象被设置为给定属性的公共成员。

实际上，这个 `property` 构造函数可以接受两个额外的参数，一个 `delete` 函数和属性的文档字符串。在实践中，很少提供 `delete` 函数，但它可以用于记录值已被删除的事实，或者如果我们有理由这样做，可以拒绝删除。文档字符串只是一个描述属性做什么的字符串，与我们在第二章，《Python 中的对象》中讨论的文档字符串没有区别。如果我们不提供此参数，则文档字符串将复制自第一个参数的文档字符串：`getter` 方法。以下是一个愚蠢的例子，它声明了每次调用任何方法时：

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

如果我们实际使用这个类，当我们要求它打印正确的字符串时，它确实会这样做：

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

此外，如果我们查看 `Silly` 类的帮助文件（通过在解释器提示符中输入 `help(Silly)`），它将显示我们 `silly` 属性的自定义文档字符串：

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

再次强调，一切都在我们的计划之中。在实践中，属性通常只使用前两个参数定义：`getter` 和 `setter` 函数。如果我们想为属性提供一个文档字符串，我们可以在 `getter` 函数上定义它；属性代理将把它复制到自己的文档字符串中。`delete` 函数通常留空，因为对象属性很少被删除。如果一个编码者尝试删除没有指定 `delete` 函数的属性，它将引发异常。因此，如果有合法的理由删除我们的属性，我们应该提供该函数。

# 装饰器 – 创建属性的另一种方式

如果你之前从未使用过 Python 装饰器，你可能想要跳过这一节，在我们讨论了第十章，《Python 设计模式 I》中的装饰器模式后再回来。然而，你不需要理解正在发生的事情，就可以使用装饰器语法来使属性方法更易读。

`property` 函数可以用装饰器语法使用，将 `get` 函数转换为 `property` 函数，如下所示：

```py
class Foo: 
 @property 
    def foo(self): 
        return "bar" 
```

这将 `property` 函数用作装饰器，与之前的 `foo = property(foo)` 语法等效。从可读性的角度来看，主要区别在于我们可以在方法顶部标记 `foo` 函数为属性，而不是在定义之后，这样它就不容易被忽视了。这也意味着我们不需要创建带有下划线前缀的私有方法来定义属性。

再进一步，我们可以为新的属性指定一个 `setter` 函数，如下所示：

```py
class Foo: 
 @property 
    def foo(self): 
        return self._foo 

 @foo.setter 
    def foo(self, value): 
        self._foo = value 
```

这个语法看起来相当奇怪，尽管意图是明显的。首先，我们将`foo`方法装饰为一个 getter。然后，我们通过应用原始装饰的`foo`方法的`setter`属性来装饰一个具有完全相同名称的第二个方法！`property`函数返回一个对象；这个对象总是带有自己的`setter`属性，然后可以将它作为装饰器应用于其他函数。get 和 set 方法使用相同的名称不是必需的，但它有助于将访问同一属性的多个方法分组在一起。

我们还可以使用`@foo.deleter`指定一个`delete`函数。我们不能使用`property`装饰器来指定文档字符串，因此我们需要依赖于属性从初始的 getter 方法复制文档字符串。以下是我们之前重写的`Silly`类，使用`property`作为装饰器：

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

这个类与我们的早期版本操作**完全相同**，包括帮助文本。你可以使用你认为更易读和优雅的语法。

# 决定何时使用属性

由于内置的`property`模糊了行为和数据之间的界限，知道何时选择属性、方法或属性可能会令人困惑。我们之前看到的用例示例是属性最常见的使用之一；我们有一个类上的数据，我们稍后想添加行为。在决定使用属性时，还需要考虑其他因素。

从技术上讲，在 Python 中，数据、属性和方法都是类上的属性。一个方法是可调用的这一事实并不能将其与其他类型的属性区分开来；实际上，我们将在第七章，《Python 面向对象快捷方式》中看到，可以创建出可以像函数一样调用的普通对象。我们还将发现函数和方法本身也是普通对象。

方法只是可调用的属性，属性只是可定制的属性这一事实可以帮助我们做出这个决定。方法通常表示动作；可以对对象执行或由对象执行的事情。当你调用一个方法时，即使只有一个参数，它也应该**做**些事情。方法名通常是动词。

一旦确认一个属性不是动作，我们需要在标准数据属性和属性之间做出选择。一般来说，直到你需要以某种方式控制对该属性的访问时，才使用标准属性。在两种情况下，你的属性通常是名词。属性和属性之间的唯一区别是我们可以在属性被检索、设置或删除时自动调用自定义操作。

让我们看看一个更实际的例子。自定义行为的常见需求是缓存一个难以计算或查找代价高昂的值（例如，需要网络请求或数据库查询）。目标是存储该值以避免重复调用昂贵的计算。

我们可以通过在属性上使用自定义获取器来实现这一点。第一次检索值时，我们执行查找或计算。然后，我们可以在我们的对象（或专门的缓存软件）上本地缓存该值作为私有属性，下次请求该值时，我们返回存储的数据。以下是我们可能缓存网页的方式：

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

我们可以测试这段代码，看看页面是否只检索一次：

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

当我在 2010 年测试这本书的第一版时，我原本在一个糟糕的卫星连接上，第一次加载内容时花费了 20 秒。第二次，我用了 2 秒（这实际上只是将行输入解释器所需的时间）。在我的更现代的连接上，它看起来如下所示：

```py
>>> webpage = WebPage("https://dusty.phillips.codes")
>>> import time
>>> now = time.time() ; content1 = webpage.content ; print(time.time() - now)
Retrieving New Page...
0.6236202716827393
>>> now = time.time() ; content2 = webpage.content ; print(time.time() - now)
1.7881393432617188e-05M
```

从我的网络主机检索一个页面大约需要 620 毫秒。从我的笔记本电脑的 RAM 中，它只需要 0.018 毫秒！

自定义获取器对于需要根据其他对象属性动态计算属性的情况也非常有用。例如，我们可能想要计算一系列整数的平均值：

```py
class AverageList(list): 
    @property 
    def average(self): 
        return sum(self) / len(self) 
```

这个非常简单的类从`list`继承，因此我们免费获得了类似列表的行为。我们只需向类中添加一个属性，嘿，我们的列表就可以有平均值了，如下所示：

```py
>>> a = AverageList([1,2,3,4])
>>> a.average
2.5  
```

当然，我们本可以将这做成一个方法，但那时我们应该将其命名为`calculate_average()`，因为方法代表动作。但名为`average`的属性更合适，它既容易输入也容易阅读。

自定义设置器在验证方面很有用，正如我们之前所看到的，但它们也可以用来代理一个值到另一个位置。例如，我们可以在`WebPage`类中添加一个内容设置器，每当值被设置时，它会自动登录我们的网络服务器并上传一个新页面。

# 管理对象

我们一直关注对象及其属性和方法。现在，我们将探讨设计更高级的对象；那些管理其他对象的对象——那些将一切联系在一起的对象。

这些对象与之前的大多数示例之间的区别在于，后者通常代表具体的概念。管理对象更像是办公室经理；他们不在现场做实际的*可见*工作，但没有他们，部门之间就没有沟通，没有人知道他们应该做什么（尽管，如果组织管理不善，这也可能是真的！）。类似地，管理类上的属性往往指的是做*可见*工作的其他对象；此类上的行为在适当的时候委托给其他类，并在它们之间传递消息。

例如，我们将编写一个程序，对存储在压缩 ZIP 文件中的文本文件执行查找和替换操作。我们需要对象来表示 ZIP 文件和每个单独的文本文件（幸运的是，我们不需要编写这些类，因为它们在 Python 标准库中可用）。管理对象将负责确保以下三个步骤按顺序发生：

1.  解压压缩文件

1.  执行查找和替换操作

1.  压缩新文件

类使用`.zip`文件名、搜索和替换字符串进行初始化。我们创建一个临时目录来存储解压的文件，这样文件夹就可以保持干净。`pathlib`库帮助处理文件和目录操作。我们将在第八章中了解更多关于它的信息，*字符串和序列化*，但以下示例中的接口应该相当清晰：

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

然后，我们为每个步骤创建一个总的*管理*方法。这个方法将责任委托给其他对象：

```py
def zip_find_replace(self): 
    self.unzip_files() 
    self.find_replace() 
    self.zip_files() 
```

显然，我们可以在一个方法中完成这三个步骤，或者在一个脚本中完成，而不需要创建对象。分离这三个步骤有几个优点：

+   **可读性**：每个步骤的代码都是一个自包含的单元，易于阅读和理解。方法名描述了方法的作用，不需要额外的文档就可以理解正在发生的事情。

+   **可扩展性**：如果子类想使用压缩 TAR 文件而不是 ZIP 文件，它可以覆盖`zip`和`unzip`方法，而无需重复`find_replace`方法。

+   **分区**：外部类可以创建这个类的实例，并直接在某个文件夹上调用`find_replace`方法，而无需压缩内容。

委派方法是以下代码中的第一个；其余的方法包括为了完整性：

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

为了简洁，对文件压缩和解压缩的代码只有很少的文档说明。我们目前的重点是面向对象设计；如果你对`zipfile`模块的内部细节感兴趣，可以参考标准库中的文档，无论是在线还是通过在你的交互式解释器中输入`import zipfile ; help(zipfile)`来查看。请注意，这个玩具示例只搜索 ZIP 文件中的顶级文件；如果解压内容中包含任何文件夹，它们将不会被扫描，文件夹内的任何文件也不会被扫描。

如果你使用的是低于 3.6 版本的 Python，在调用`extractall`、`rmtree`和`file.write`在`ZipFile`对象上之前，你需要将路径对象转换为字符串。

示例中的最后两行允许我们通过传递`zip`文件名、搜索字符串和替换字符串作为参数，从命令行运行程序，如下所示：

```py
$python zipsearch.py hello.zip hello hi  
```

当然，这个对象不必从命令行创建；它可以从另一个模块导入（以执行批量 ZIP 文件处理），或作为 GUI 界面的一部分访问，甚至是一个更高级的管理对象，该对象知道在哪里获取 ZIP 文件（例如，从 FTP 服务器检索或将其备份到外部磁盘）。

随着程序变得越来越复杂，被建模的对象越来越不像物理对象。属性是其他抽象对象，方法则是改变这些抽象对象状态的动作。但无论对象多么复杂，其核心总是一组具体数据和定义良好的行为。

# 移除重复代码

经常，管理风格课程中的代码，如`ZipReplace`，相当通用，可以以多种方式应用。可以使用组合或继承来帮助将此代码保留在一个地方，从而消除重复代码。在我们查看任何此类示例之前，让我们先讨论一点理论。具体来说，为什么重复代码是件坏事？

有几个原因，但它们都归结为可读性和可维护性。当我们编写一个与早期代码相似的新代码时，最简单的事情就是复制旧代码，并根据需要更改（变量名、逻辑、注释）以使其在新位置工作。或者，如果我们编写的新代码似乎与项目中的其他代码相似，但又不完全相同，那么编写具有相似行为的全新代码通常比找出如何提取重叠功能要容易得多。

但一旦有人需要阅读和理解代码，并遇到重复的代码块，他们就会面临一个困境。原本可能看起来有意义的代码突然需要被理解。这一部分与另一部分有何不同？它们有何相同之处？在什么条件下调用一个部分？何时调用另一个？你可能认为只有你一个人会阅读你的代码，但如果你八个月不接触那段代码，它对你来说将像对一个新手程序员一样难以理解。当我们试图阅读两段相似的代码时，我们必须理解它们为什么不同，以及它们是如何不同的。这浪费了读者的时间；代码应该首先易于阅读。

我曾经不得不尝试理解某人的代码，这段代码中有三份完全相同的 300 行糟糕的代码。我在这个代码上工作了整整一个月，才最终明白这三份**相同**的版本实际上执行的是略微不同的税务计算。其中一些细微的差异是有意为之，但也有明显的地方，有人在更新了一个函数中的计算时，没有更新其他两个函数。代码中细微、难以理解的错误数量无法计数。我最终用大约 20 行的易读函数替换了所有 900 行。

阅读这样的重复代码可能会感到疲倦，但代码维护更是折磨。正如前面的故事所暗示的，保持两段相似代码的更新状态可能是一场噩梦。我们必须记住在更新其中一个时更新两个部分，并且我们必须记住多个部分之间的差异，以便在编辑每个部分时修改我们的更改。如果我们忘记更新所有部分，我们最终会得到极其烦人的错误，这些错误通常表现为，“我已经修复了那个，为什么它还在发生”？

结果是，阅读或维护我们代码的人必须花费天文数字般的时间来理解和测试它，与最初以非重复方式编写它所需的时间相比。当我们自己进行维护时，这甚至更加令人沮丧；我们会发现自己说，为什么我没有一开始就做得正确？通过复制和粘贴现有代码节省的时间，在第一次维护它时就已经丢失了。代码被阅读和修改的次数比它被编写的次数多得多，也更为频繁。可理解的代码始终应该是优先考虑的。

这就是为什么程序员，尤其是 Python 程序员（他们往往比普通开发者更重视优雅的代码），遵循所谓的**不要重复自己**（**DRY**）原则。DRY 代码是可维护的代码。我对初学者的建议是永远不要使用编辑器的复制粘贴功能。对于中级程序员，我建议他们在按下*Ctrl* + *C*之前三思。

但我们除了代码重复之外还能做什么呢？最简单的解决方案通常是将其移动到函数中，该函数接受参数以处理任何不同的部分。这不是一个特别面向对象的解决方案，但它通常是最佳选择。

例如，如果我们有两个将 ZIP 文件解压到两个不同目录的代码片段，我们可以轻松地用一个接受参数的函数来替换它，该参数指定了它应该解压到的目录。这可能会使函数本身稍微难以阅读，但一个好的函数名和文档字符串可以轻松弥补这一点，并且任何调用该函数的代码都将更容易阅读。

理论已经足够多了！这个故事的意义是：总是努力重构你的代码，使其更易于阅读，而不是编写可能看起来更容易编写但质量较差的代码。

# 在实践中

让我们探索两种我们可以重用现有代码的方法。在编写了替换文本文件 ZIP 文件中的字符串的代码之后，我们后来被委托去将 ZIP 文件中的所有图片缩放到 640 x 480。看起来我们可以使用与`ZipReplace`中使用的非常相似的模式。我们的第一个冲动可能是保存该文件的副本，并将`find_replace`方法更改为`scale_image`或类似的方法。

但是，这并不理想。如果我们有一天想将`unzip`和`zip`方法也改为打开 TAR 文件怎么办？或者我们可能希望为临时文件使用一个保证唯一的目录名称。在任何情况下，我们都必须在不同地方进行更改！

我们将首先演示一个基于继承的解决方案。首先，我们将修改原始的`ZipReplace`类，使其成为处理通用 ZIP 文件的超类：

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

我们将`filename`属性更改为`zipname`，以避免与各种方法内部的`filename`局部变量混淆。这有助于使代码更易于阅读，尽管这实际上并不是设计上的改变。

我们还删除了`__init__`方法中的两个特定于`ZipReplace`的参数（`search_string`和`replace_string`）。然后，我们将`zip_find_replace`方法重命名为`process_zip`，并让它调用一个尚未定义的`process_files`方法而不是`find_replace`；这些名称更改有助于展示我们新类更通用的特性。请注意，我们已经完全删除了`find_replace`方法；这段代码是特定于`ZipReplace`的，并且在这里没有存在的必要。

这个新的`ZipProcessor`类实际上并没有定义`process_files`方法。如果我们直接运行它，它会引发异常。因为它不是用来直接运行的，所以我们删除了原始脚本底部的 main 调用。我们可以将其制作成一个抽象基类，以表明这个方法需要在子类中定义，但我为了简洁起见省略了它。

现在，在我们继续到图像处理应用程序之前，让我们修复原始的`zipsearch`类，使其能够使用这个父类，如下所示：

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

这段代码比原始版本更短，因为它从父类继承了 ZIP 处理能力。我们首先导入我们刚刚编写的基类，并让`ZipReplace`扩展这个类。然后，我们使用`super()`来初始化父类。`find_replace`方法仍然存在，但我们将其重命名为`process_files`，这样父类就可以从其管理界面调用它。因为这个名称不如原来的名称描述性强，所以我们添加了一个文档字符串来描述它的功能。

现在的工作量相当大，考虑到我们现在拥有的程序在功能上与我们开始时使用的程序没有区别！但是，完成了这项工作后，我们现在写其他操作 ZIP 存档中文件的类（例如，假设请求的）照片缩放器就变得容易多了。此外，如果我们想改进或修复 zip 功能的 bug，我们只需更改一个 `ZipProcessor` 基类，就可以一次性为所有子类进行操作。因此，维护将更加有效。

看看现在创建一个利用 `ZipProcessor` 功能的照片缩放类是多么简单：

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

看看这个类是多么简单！我们之前所做的所有工作都得到了回报。我们只是打开每个文件（假设它是图像；如果文件无法打开或不是图像，它将无礼地崩溃），将其缩放，并保存回去。`ZipProcessor` 类负责压缩和解压缩，而无需我们做任何额外的工作。

# 案例研究

对于这个案例研究，我们将尝试进一步探讨这个问题：我应该何时选择对象而不是内置类型？我们将模拟一个可能在文本编辑器或文字处理器中使用的 `Document` 类。它应该有什么对象、函数或属性？

我们可能从 `Document` 内容的 `str` 开始，但在 Python 中，字符串是不可变的（不能被更改）。一旦定义了一个 `str`，它就永远不变。我们无法在不创建一个新的字符串对象的情况下向其中插入字符或删除一个字符。这样就会留下很多占用内存的 `str` 对象，直到 Python 的垃圾回收器决定清理。

因此，我们不会使用字符串，而会使用字符列表，我们可以随意修改它。此外，我们还需要知道列表中的当前光标位置，并且可能还需要存储文档的文件名。

真实的文本编辑器使用基于二叉树的数据结构，称为 `rope` 来模拟它们的文档内容。这本书的标题不是《高级数据结构》，所以如果你对了解更多关于这个有趣的主题感兴趣，你可能想在网上搜索 `rope 数据结构`。

我们可能想要对文本文档做很多事情，包括插入、删除和选择字符；剪切、复制和粘贴选择；以及保存或关闭文档。看起来有大量的数据和操作，所以将这些所有东西放入一个单独的 `Document` 类中是有意义的。

一个相关的问题是：这个类应该由一些基本的 Python 对象组成，如 `str` 文件名、`int` 光标位置和一个字符的 `list`？或者，这些中的某些或所有东西应该被特别定义的对象？关于单独的行和字符呢？它们需要有自己的类吗？

我们将在进行过程中回答这些问题，但让我们首先从最简单的类开始——`Document`，看看它能做什么：

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

这个基本类允许我们完全控制编辑一个基本文档。看看它在实际操作中的表现：

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

它看起来是正常工作的。我们可以将这些方法连接到键盘的字母键和箭头键上，这样文档就可以跟踪一切了。

但如果我们想连接的不仅仅是箭头键呢？如果我们还想连接 `Home` 和 `End` 键呢？我们可以在 `Document` 类中添加更多方法，这些方法在字符串中向前或向后搜索换行字符（换行字符，转义为 `\n`，代表一行结束和下一行的开始），并跳转到它们，但如果为每个可能的移动动作（按单词移动、按句子移动、*Page Up*、*Page Down*、行尾、空白开始等）都这样做，类将会变得非常大。也许将这些方法放在一个单独的对象上会更好。所以，让我们将 `Cursor` 属性转换成一个知道其位置并能操作该位置的对象。我们可以将向前和向后方法移动到那个类中，并为 `Home` 和 `End` 键添加几个更多的方法，如下所示：

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

这个类将文档作为初始化参数，因此方法可以访问文档字符列表的内容。然后它提供了简单的向前和向后移动方法，就像之前一样，以及移动到 `home` 和 `end` 位置的方法。

这段代码并不十分安全。你很容易就会移动到结束位置，如果你尝试在一个空文件上回到起始位置，它将会崩溃。这些示例被保持得比较短，以便于阅读，但这并不意味着它们是防御性的！你可以作为一个练习来改进这段代码的错误检查；这可能是一个扩展你的异常处理技能的绝佳机会。

`Document` 类本身几乎没有变化，除了移除了移动到 `Cursor` 类的两个方法：

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

我们刚刚更新了所有访问旧光标整数的代码，以使用新对象。现在我们可以测试 `home` 方法是否真的移动到了换行字符，如下所示：

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

现在，因为我们已经大量使用了那个字符串 `join` 函数（用于连接字符，以便我们可以看到实际的文档内容），我们可以在 `Document` 类中添加一个属性，以提供完整的字符串，如下所示：

```py
@property 
def string(self): 
    return "".join(self.characters) 
```

这使得我们的测试变得稍微简单一些：

```py
>>> print(d.string)
hello
world  
```

这个框架简单易扩展，可以创建和编辑一个完整的纯文本文档（尽管可能需要一点时间！）现在，让我们扩展它以支持富文本；可以包含**粗体**、下划线或*斜体*字符的文本。

我们可以有两种处理方式。第一种是在我们的字符列表中插入*假的*字符，它们像指令一样工作，例如*直到找到停止粗体的字符为止，粗体字符*。第二种是在每个字符上添加信息，指示它应该有什么格式。虽然前一种方法在实际编辑器中更为常见，但我们将实现后一种解决方案。为了做到这一点，显然我们需要一个字符类。这个类将有一个表示字符的属性，以及三个布尔属性，表示它是否是*粗体、斜体或下划线*。

嗯，等等！这个`Character`类会有任何方法吗？如果没有，也许我们应该使用许多 Python 数据结构中的任何一个；一个元组或命名元组可能就足够了。我们会对字符执行或调用哪些操作？

很明显，我们可能想要对字符做一些事情，比如删除或复制它们，但这些事情需要在`Document`级别处理，因为它们实际上是在修改字符列表。对单个字符需要做些什么？

实际上，现在我们正在思考一个`Character`类实际上**是什么**...它是什么？我们是否可以说`Character`类就是一个字符串？也许我们应该在这里使用继承关系？这样我们就可以利用`str`实例所带的众多方法。

我们在谈论哪些方法呢？有`startswith`、`strip`、`find`、`lower`等等。大多数这些方法都期望在包含多个字符的字符串上工作。相比之下，如果`Character`类继承自`str`，我们可能明智地重写`__init__`方法，在提供多字符字符串时抛出异常。由于那些我们免费获得的方法实际上并不适用于我们的`Character`类，所以最终看来我们不应该使用继承。

这又带我们回到了最初的问题；`Character`甚至应该是一个类吗？`object`类中有一个非常重要的特殊方法，我们可以利用它来表示我们的字符。这个方法叫做`__str__`（两端各两个下划线，就像`__init__`一样），在字符串操作函数如`print`和`str`构造函数中使用，用于将任何类转换为字符串。默认实现做了一些无聊的事情，比如打印模块和类的名称，以及它在内存中的地址。但如果我们重写它，我们可以让它打印我们想要的任何内容。对于我们的实现，我们可以让它在字符前加上特殊字符来表示它们是粗体、斜体还是下划线。因此，我们将创建一个表示字符的类，这就是它：

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

这个类允许我们在应用`str()`函数时为字符创建前缀特殊字符。这并没有什么激动人心的地方。我们只需要对`Document`和`Cursor`类进行一些小的修改，以便与这个类一起工作。在`Document`类中，我们在`insert`方法的开始处添加这两行，如下所示：

```py
def insert(self, character): 
    if not hasattr(character, 'character'): 
        character = Character(character) 
```

这段代码相当奇怪。它的基本目的是检查传入的字符是否是`Character`或`str`。如果是字符串，它会被包裹在`Character`类中，这样列表中的所有对象都是`Character`对象。然而，完全有可能有人使用我们的代码，想要使用既不是`Character`也不是字符串的类，使用鸭子类型。如果对象有一个字符属性，我们假设它是一个类似`Character`的对象。但如果它没有，我们假设它是一个类似`str`的对象，并将其包裹在`Character`中。这有助于程序利用鸭子类型和多态；只要一个对象有一个字符属性，它就可以在`Document`类中使用。

这种通用的检查可能非常有用。例如，如果我们想制作一个具有语法高亮的程序员编辑器，我们需要关于字符的额外数据，例如字符属于哪种语法标记类型。请注意，如果我们进行很多这种类型的比较，可能最好将`Character`实现为一个具有适当`__subclasshook__`的抽象基类，正如在第三章中讨论的，*当对象相似时*。

此外，我们还需要修改`Document`上的字符串属性，以便接受新的`Character`值。我们只需要在连接之前对每个字符调用`str()`，如下所示：

```py
    @property 
    def string(self): 
        return "".join((str(c) for c in self.characters)) 
```

这段代码使用生成器表达式，我们将在第九章中讨论，*迭代器模式*。这是在序列中的所有对象上执行特定操作的快捷方式。

最后，我们还需要检查`Character.character`，而不仅仅是之前存储的字符串字符，在`home`和`end`函数中查看它是否匹配换行符时，如下所示：

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

这完成了字符的格式化。我们可以按照以下方式测试它是否正常工作：

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

如预期的那样，每次我们打印字符串时，每个粗体字符前面都有一个`*`字符，每个斜体字符前面都有一个`/`字符，每个下划线字符前面都有一个`_`字符。我们的所有函数似乎都正常工作，我们可以在事后修改列表中的字符。我们有一个可以连接到适当的图形用户界面并与键盘输入和屏幕输出连接的富文本文档对象。当然，我们希望在 UI 中显示真正的*粗体、斜体和下划线*字体，而不是使用我们的`__str__`方法，但对于我们要求的基本测试来说，这是足够的。

# 练习

我们已经探讨了在面向对象的 Python 程序中对象、数据和方法如何相互交互的各种方式。像往常一样，你的第一个想法应该是如何将这些原则应用到自己的工作中。你是否有任何混乱的脚本可以使用面向对象的经理重写？查看一些你的旧代码，寻找不是动作的方法。如果名称不是一个动词，尝试将其重写为一个属性。

思考一下你在任何语言中编写的代码。它是否违反了 DRY 原则？是否有任何重复的代码？你是否复制和粘贴了代码？你是否因为不想理解原始代码而编写了两个类似代码的版本？现在回顾一下你最近的一些代码，看看你是否可以使用继承或组合重构重复的代码。尽量选择一个你仍然感兴趣维护的项目；不是那种你永远不会再次触碰的旧代码。这将有助于你在进行改进时保持兴趣！

现在，回顾一下我们在本章中查看的一些示例。从一个使用属性来缓存检索数据的缓存网页示例开始。这个示例的一个明显问题是缓存永远不会刷新。给属性的 getter 添加一个超时，并且只有当页面在超时之前被请求时才返回缓存的页面。你可以使用`time`模块（`time.time() - an_old_time`返回自`an_old_time`以来经过的秒数）来确定缓存是否已过期。

还要看看基于继承的`ZipProcessor`。在这里使用组合而不是继承可能是合理的。你可以在`ZipReplace`和`ScaleZip`类中扩展类，而不是将这些类的实例传递给`ZipProcessor`构造函数并调用它们来执行处理部分。实现这一点。

你觉得哪个版本更容易使用？哪个更优雅？哪个更容易阅读？这些问题都是主观的；每个人的答案都不尽相同。然而，知道答案是很重要的。如果你发现你更喜欢继承而不是组合，你需要注意在日常编码中不要过度使用继承。如果你更喜欢组合，确保不要错过创建基于继承的优雅解决方案的机会。

最后，为我们在案例研究中创建的各种类添加一些错误处理器。它们应该确保只输入单个字符，不要尝试将光标移动到文件的末尾或开头，不要删除不存在的字符，并且不要在没有文件名的情况下保存文件。尽量思考尽可能多的边缘情况，并考虑如何处理它们（考虑边缘情况大约是专业程序员工作的 90%）！考虑不同的处理方式；当用户尝试移动到文件末尾时，你应该抛出一个异常，还是仅仅停留在最后一个字符上？

在你的日常编码中，请注意复制和粘贴命令。每次你在编辑器中使用它们时，考虑一下是否是一个改进程序组织的好主意，这样你就可以只保留你即将复制的代码的一个版本。

# 摘要

在本章中，我们专注于识别对象，特别是那些不是立即显而易见的对象；那些管理和控制的对象。对象应该既有数据也有行为，但属性可以被用来模糊两者之间的区别。DRY 原则是代码质量的重要指标，继承和组合可以用来减少代码重复。

在下一章中，我们将介绍几个内置的 Python 数据结构和对象，重点关注它们的面向对象属性以及它们如何被扩展或适应。
