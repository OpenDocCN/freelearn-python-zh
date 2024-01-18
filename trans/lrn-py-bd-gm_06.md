# 面向对象编程

编程不仅仅是编写程序，同样重要的是理解它们，这样我们就可以修复其中的错误和漏洞。因此，我们说程序员天生就是用来阅读和理解代码的。然而，随着程序变得越来越复杂，编写易读的程序变得更加困难。在本书中，我们既写了美观的代码，也写了混乱的代码。我们用顺序编程制作了一个井字棋游戏，其可读性较低。我们可以将这些程序视为不优雅的，因为我们很难阅读和理解它们的代码和顺序流程。在编写这些程序之后，我们使用函数对其进行了修改，从而使我们混乱的代码更加优雅。然而，如果你正在处理包含数千行代码的程序，很难在同一个文件中编写程序并理解你正在使用的每个函数的行为。因此，发现和修复以过程方式编写的程序的错误也是困难的。因此，我们需要一种方法，可以将多行程序轻松地分解成更小的模块或部分，以便更容易地发现和修复这些错误。有许多实现这一目标的方法，但最有效和流行的方法是使用**面向对象编程**（**OOP**）方法。

事实证明，我们从本书的开头就一直在使用对象，但并没有准确地了解它们是如何制作和使用的。本章将帮助您通过一些简单的示例了解面向对象编程的术语和概念。本章末尾，我们还将根据 OOP 方法修改我们在前几章中使用函数编写的贪吃蛇游戏代码。

本章将涵盖以下主题：

+   面向对象编程概述

+   Python 类

+   封装

+   继承

+   多态

+   使用 OOP 实现的贪吃蛇游戏

+   可能的错误和修改

# 技术要求

为了充分利用本章，您将需要以下内容：

+   Python 3.5 或更新版本

+   Python IDLE（Python 内置 IDE）

+   文本编辑器

+   网络浏览器

本章的文件可以在本书的 GitHub 存储库中找到：[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter06`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter06)

查看以下视频，以查看代码的实际运行情况：

[`bit.ly/2oKD6D2`](http://bit.ly/2oKD6D2)

# 面向对象编程概述

*Python 中的一切都是对象*。我们从本书的开头就一直在雄辩地陈述这一观点，并且在每一章中都在证明这个说法。一切都是对象。对象可以是元素、属性或函数的集合。数据结构、变量、数字和函数都是对象。面向对象编程是一种编程范式，它通过对象的帮助提供了一种优雅的程序结构方式。对象的行为和属性被捆绑到模板中，我们称之为类。这些行为和属性可以从该类的不同对象中调用。不要被行为和属性这些术语搞混了。它们只是方法和变量的不同名称。在某些类中定义的函数被称为方法。

我们将在本章后面深入探讨类和方法的概念，但现在，让我们在实际为它们制作模板之前更多地了解对象。

我们从这本书的开始就在不知不觉地使用对象。我们以前使用过不同类的方法，比如`randint()`方法。这个方法是通过导入一个名为`random`的模块来使用的。这个方法也是 Python 的内置类。类是一个模板，我们可以在其中编写对象的函数。例如，一个人可以被表示为一个对象。一个人有不同的特征，比如`name`、`age`和`hair_color`，这些是唯一的属性。然而，人所执行的行为，比如吃饭、走路和睡觉，是行为或方法。我们可以从这些模板中创建任意多的对象。但现在，让我们想象两个对象：

```py
Object 1: Stephen : name = "Stephen Hawking", age= 56, hair_color= brown, eating, walking, sleeping

Object 2: Albert: name = "Albert Einstein", age = 77, hair_color= black, eating, walking, sleeping
```

在前面的两个对象中，`name`、`age`和`hair_color`是唯一的。所有的对象都有唯一的属性，但它们执行的行为或方法是相同的，比如吃饭、走路和睡觉。因此，我们可以得出结论，与输入和输出交互的数据模型是一个属性，因为它将被输入到方法中。根据每个对象的唯一属性，类的方法将产生不同的结果。

因此，我们可以说，面向对象编程是将现实世界的实体建模为具有唯一数据关联的对象，并且可以执行某些函数的方法。在类内部定义的函数称为方法，因此我们只需要从函数切换到方法。然而，请注意，方法的工作方式与函数的工作方式类似。就像函数是通过其名称或标志调用的一样，方法也需要通过其名称调用。但是，这个调用应该由对象发起。让我们看一个简单的例子来澄清这一点：

```py
>>> [1,2,3].pop()
3
```

我们在前面的章节中看过这些方法。但是如果你仔细看这段代码，你会发现我们是从对象中调用方法。我们使用了`pop`方法，并在列表对象上调用它。这是面向对象编程的一个简单原型。面向对象编程的一个优点是它隐藏了方法调用的内部复杂性。正如你可能记得的，我们用随机模块调用了`randint`方法。我们甚至没有查看随机库的内容。因此，我们避免了库的工作复杂性。面向对象编程的这个特性将使我们只关注程序的重要部分，而不是方法的内部工作。

面向对象编程的两个主要实体是对象和类。我们可以通过使用模板来模拟类，其中方法和属性被映射。方法是函数的同义词，而属性是将每个对象与另一个对象区分开的属性。让我们通过创建一个简单的类和对象来对这个术语有一个很好的理解。

# Python 类

正如我们在前一节中讨论的，对象继承了类内部编写的所有代码。因此，我们可以使用在类主体内映射的方法和属性。类是一个模板，可以从中创建实例。看一下下面的例子：

![](img/f96c5006-9dd8-4944-b0c4-3c0a5dfd2257.png)

在前面，`Bike`类可以被认为是一个模板，可以从中实例化对象。在`Bike`类中，有一些属性，这些属性唯一地代表了从这个类创建的对象。每个创建的对象都会有不同的属性，比如名称、颜色和价格，但它们会调用相同的方法。这个方法应该与类的实例一起调用。让我们看看如何在 Python 中创建类：

```py
>>> class Bike:
        pass
```

我们用 class 关键字在 Python 中创建一个类，后面跟着类的名称。通常，类名的第一个字母是大写的；在这里，我们写了`Bike`，B 是大写的。现在，在全局范围内，我们已经创建了`Bike`类。在类的主体内，我们写了一个 pass，而不是方法和属性。现在，让我们创建这个类的对象：

```py
>>> suzuki = Bike()
>>> type(suzuki)
<class '__main__.Bike'>
```

在上面的代码中，我们从`Bike`类中创建了一个名为`Suzuki`的实例。实例化表达式看起来类似于函数调用。现在，如果您检查`Suzuki`对象的类型，它是`Bike`类的类型。因此，任何对象的类型都将是类类型，因为对象是类的实例。

现在是时候向这个`Bike`类添加一些方法了。这类似于函数的声明。`def`关键字，后面跟着方法的名称，是声明类的方法的最佳方式。让我们看一下以下代码：

```py
#class_ex_1.py
class Bike:
    def ride_Left(self):
        print("Bike is turning to left")

    def ride_Right(self):
        print("Bike is turning to right")

    def Brake(self):
        print("Breaking......")

suzuki = Bike()
suzuki.ride_Left()
suzuki.Brake()

>>> 
Bike is turning to left
Breaking......
```

我们向`Bike`类添加了三个方法。在声明这些方法时使用的参数是`self`变量。这个`self`变量或关键字也是类的一个实例。您可以将这个`self`变量与指针进行比较，该指针指向当前对象。在每次实例化时，`self`变量表示指向当前类的指针对象。我们将很快澄清`self`关键字的用法和重要性，但在此之前，看一下上面的代码，我们创建了一个`Suzuki`对象，并用它调用了类的方法。

上面的代码类似于我们从 random 模块调用`randint`方法的代码。这是因为我们正在使用 random 库的方法。

当定义任何类时，只定义了对象的表示，这最终减少了内存损失。在上面的例子中，我们用名为`Bike`的原型制作了一个原型。可以从中制作不同的实例，如下所示：

```py
>>> honda = Bike() #first instance
>>> honda.ride_Right()
 Bike is turning to right

 >>> bmw = Bike() #second instance
 >>> bmw.Brake()
 Breaking......
```

现在我们已经看过如何创建对象并使用类内定义的方法，我们将向类添加属性。属性或属性定义了每个对象的独特特征。让我们向我们的类添加一些属性，比如`name`，`color`和`price`：

```py
class Bike:
     name = ''
     color= ' '
     price = 0

     def info(self, name, color, price):
         self.name = name
         self.color = color
         self.price = price
         print("{}: {} and {}".format(self.name,self.color,self.price))

 >>> suzuki = Bike()
 >>> suzuki.info("Suzuki", "Black", 100000)
 Suzuki: Black and 100000
```

在上面的代码中有很多行话。在幕后，这个程序是关于类和对象的创建。我们添加了三个属性：`name`，`color`和`price`。要使用类的这些属性，我们必须用`self`关键字引用它们。`name`，`color`和`price`参数被传递到`info`函数中，并分配给`Bike`类的相应`name`，`color`和`price`属性。`self.name, self.color, self.price = name,color,price`语句将初始化类变量。这个过程称为初始化。我们也可以使用构造函数进行初始化，就像这样：

```py
class Bike:
    def __init__(self,name,color,price):
        self.name = name
        self.color = color
        self.price = price

    def info(self):
        print("{}: {} and {}".format(self.name,self.color,self.price))

>>> honda = Bike("Honda", "Blue", 30000)
>>> honda.info()
Honda: Blue and 30000
```

在 Python 中，特殊的`init`方法将模拟构造函数。构造函数是用于初始化类的属性的方法或函数。构造函数的定义在我们创建类的实例时执行。根据`init`的定义，我们可以在创建类的对象时提供任意数量的参数。类的第一个方法应该是构造函数，并且必须初始化类的成员。类的基本格式应该在开始时有属性声明，然后是方法。

现在我们已经创建了自己的类并声明了一些方法，让我们探索面向对象范式的一些基本特性。我们将从**封装**开始，它用于嵌入在类内声明的方法和变量的访问权限。

# 封装

封装是将数据与代码绑定为一个称为胶囊的单元的一种方式。这样，它提供了安全性，以防止对代码进行不必要的修改。使用面向对象范式编写的代码将以属性的形式具有关键数据。因此，我们必须防止数据被损坏或变得脆弱。这就是所谓的数据隐藏，是封装的主要特性。为了防止数据被意外修改，封装起着至关重要的作用。我们可以将类的成员设为私有成员以实现封装。私有成员，无论是方法还是属性，都可以在其签名的开头使用双下划线来创建。在下面的例子中，`__updateTech`是一个私有方法：

```py
class Bike:
    def __init__(self):
        self.__updateTech()
    def Ride(self):
        print("Riding...")
    def __updateTech(self):
        print("Updating your Bike..")

>>> honda = Bike()
Updating your Bike..
>>> honda.Ride()
Riding...
>>> honda.__updateTech()
AttributeError: 'Bike' object has no attribute '__updateTech'
```

在前面的例子中，我们无法从类的对象中调用`updateTech`方法。这是由于封装。我们使用双下划线将此方法设为私有。但有时我们可能需要修改这些属性或行为的值。我们可以使用 getter 和 setter 来修改。这些方法将获取类的属性的值并设置值。因此，我们可以得出结论，封装是面向对象编程的一个特性，它将防止我们意外修改和访问数据，但不是有意的。类的私有成员实际上并不是隐藏的；相反，它们只是与其他成员区分开来，以便 Python 解析器能够唯一解释它们。`updateTech`方法是通过在其名称开头使用双下划线(`__`)来使其成为私有和唯一的。类的属性也可以使用相同的技术来私有化。现在让我们来看一下：

```py
class Bike:
    __name = " "
    __color = " "
    def __init__(self,name,color):
        self.__name = name
        self.__color = color
    def info(self):
        print("{} is of {} color".format(self.__name,self.__color))

>>> honda = Bike("Honda", "Black")
>>> honda.info()
Honda is of Black color
```

我们可以清楚地看到`name`和`color`属性是私有的，因为它们以双下划线开头。现在，让我们尝试使用对象来修改这些值：

```py
>>> honda.__color = "Blue"
>>> honda.info()
Honda is of Black color
```

我们尝试修改`Bike`类的颜色属性，但什么也没发生。这表明封装将防止意外更改。但如果我们需要有意地进行更改呢？这可以通过 getter 和 setter 来实现。看看以下例子以了解更多关于 getter 和 setter 的信息：

```py
class Bike:
    __name = " "
    __color = " "
    def __init__(self,name,color):
        self.__name = name
        self.__color = color
    def setNewColor(self, color):
        self.__color = color
    def info(self):
        print("{} is of {} color".format(self.__name,self.__color))

>>> honda = Bike("Honda", "Blue")
>>> honda.info()
Honda is of Blue color
>>> honda.setNewColor("Orange")
>>> honda.info()
Honda is of Orange color
```

在前面的程序中，我们定义了一个`Bike`类，其中包含一些私有成员，如`name`和`color`。我们使用`init`构造函数在创建类的实例时初始化属性的值。我们尝试修改它的值。然而，我们无法更改其值，因为 Python 解析器将这些属性视为私有。因此，我们使用`setNewColor` setter 为该私有成员设置新值。通过提供这些 getter 和 setter 方法，我们可以使类成为只读或只写，从而防止意外数据修改和有意的窃取。

现在，是时候来看一下面向对象范式的另一个重要特性——继承。继承帮助我们编写将从其父类继承每个成员并允许我们修改它们的类。

# 继承

继承是面向对象编程范式中最重要和最知名的特性。你还记得函数的可重用特性吗？继承也提供了可重用性，但伴随着大量的代码。要使用继承，我们必须有一个包含一些代码的现有类。这必须由一个新类继承。这样的现有类称为**父**类或**基**类。我们可以创建一个新类作为`Child`类，它将获取并访问父类的所有属性和方法，这样我们就不必从头开始编写代码。我们还可以修改子类继承的方法的定义和规范。

在下面的示例中，我们可以看到**Child**类或**Derived**类指向**Base**或**Parent**类，这意味着存在单一继承：

![](img/49a3b5cc-20a5-4a22-935d-770befb3da2a.png)

在 Python 中，使用继承很容易。通过在`Child`类后面的括号中提及`Parent`类的名称，`Child`类可以从`Parent`类继承。以下代码显示了如何实现单一继承：

```py
class Child_class(Parent_class):
    <child-class-members>
```

单个类也可以继承多个类。我们可以通过在括号内写入所有这些类的名称来实现这一点：

```py
class Child_class(Base_class1, Base_class2, Base_class3 .....):
    <child-class-members>
```

让我们写一个简单的例子，以便更好地理解继承。在下面的例子中，`Bike`将是`Parent`类，`Suzuki`将是`Child`类：

```py
class Bike:
    def __init__(self):
        print("Bike is starting..")
    def Ride(self):
        print("Riding...")

class Suzuki(Bike):
    def __init__(self,name,color):
        self.name = name
        self.color = color
    def info(self):
        print("You are riding {0} and it's color is 
          {1}".format(self.name,self.color))

#Save above code in python file and Run it

>>> suzuki = Suzuki("Suzuki", "Blue")
>>> suzuki.Ride()
Riding...
>>> suzuki.info()
You are riding Suzuki and it's color is Blue
```

让我们看一下前面的代码，并对继承感到惊讶。首先，我们创建了一个`Base`类，并在其中添加了两个方法。之后，我们创建了另一个类，即子类或派生类，称为`Suzuki`。它是一个子类，因为它使用`class Suzuki(Bike)`语法继承了其父类`Bike`的成员。我们还向子类添加了一些方法。创建这两个类后，我们创建了子类的对象。我们知道，当创建对象时，将自动调用要调用的方法是构造函数或`init`。因此，在创建该类的对象时，我们传递了构造函数要求的值。之后，我们从`Suzuki`类的对象中调用`Ride`方法。您可以在`Suzuki`类的主体内检查`Ride`方法。它不在那里——相反，它在`Bike`类的套件中。由于继承，我们能够调用`Base`类的方法，就好像它们在`Child`类中一样。我们还可以在`Child`类中使用在`Base`类中定义的每个属性。

然而，并非所有特性都在子类中继承。当我们创建子类的实例时，子类的`init`方法被调用，但`Parent`的方法没有被调用。然而，有一种方法可以调用该构造函数：使用`super`方法。这在下面的代码中显示：

```py
class Bike:
    def __init__(self):
        print("Bike is starting..")
    def Ride(self):
        print("Riding...")

class Suzuki(Bike):
    def __init__(self,name,color):
        self.name = name
        self.color = color
        super().__init__()

>>> suzuki = Suzuki("Suzuki", "Blue")
Bike is starting..
```

`super()`方法指的是超类或`Parent`类。因此，在实例化超类之后，我们调用了该超类的`init`方法。

这类似于`Bike().__init__()`，但在这种情况下，Bike is starting..将被打印两次，因为`Bike()`语句将创建一个`Bike`类的对象。这意味着`init`方法将被自动调用。第二次调用是使用`Bike`类的对象进行的。

在 Python 中，多级继承是可用的。当任何子类从另一个子类继承时，将创建一个链接的序列。关于如何创建多级继承链，没有限制。以下图表描述了多个类从其父类继承特性：

![](img/4dfded1e-55cf-43fa-8e23-5512ec11b5ff.png)

以下代码显示了多级继承的特点。我们创建了三个类，每个类都继承了前一个类的特点：

```py
class Mobile:
    def __init__(self):
        print("Mobile features: Camera, Phone, Applications")
class Samsung(Mobile):
    def __init__(self):
        print("Samsung Company")
        super().__init__()
class Samsung_Prime(Samsung):
    def __init__(self):
        print("Samsung latest Mobile")
        super().__init__()

>>> mobile = Samsung_Prime()
Samsung latest Mobile
Samsung Company
Mobile features: Camera, Phone, Applications
```

现在我们已经看过继承，是时候看看另一个特性，即多态。从字面上看，**多态**是适应不同形式的能力。因此，这个特性将帮助我们以不同的形式使用相同的代码，以便可以用它执行多个任务。让我们来看一下。

# 多态

在面向对象的范式中，多态性允许我们在`Child`类中定义与`Parent`类中定义的相同签名的方法。正如我们所知，继承允许我们使用`Parent`类的每个方法，就好像它们是在`Child`类中的子类对象的帮助下。然而，我们可能会遇到这样的情况，我们必须修改在父类中定义的方法的规格，以便它独立于`Parent`类执行。这种技术称为方法重写。顾名思义，我们正在用`Child`类内部的新规格覆盖`Base`类的已有方法。使用方法重写，我们可以独立调用这两个方法。如果你在子类中重写了父类的方法，那么该方法的任何版本（无论是子类的新版本还是父类的旧版本）都将根据使用它的对象的类型来调用。例如，如果你想调用方法的新版本，你应该使用`Child`类对象来调用它。谈到父类方法，我们必须使用`Parent`类对象来调用它。因此，我们可以想象到这两组方法已经开发出来，但是具有相同的名称和签名，这意味着基本的多态性。在编程中，多态性是指相同的函数或方法以不同的形式或类型使用。

我们可以从到目前为止学到的知识中开始思考多态性的例子。你还记得`len()`函数吗？这是一个内置的 Python 函数，以对象作为参数。这里，对象可以是任何东西；它可以是字符串、列表、元组等。即使它有相同的名称，它也不限于执行单一任务——它可以以不同的形式使用，如下面的代码所示：

```py
>>> len(1,2,3) #works with tuples
3
>>> len([1,2,3]) #works with lists
3
>>> len("abc") #works with strings
3
```

让我们看一个例子来演示继承的多态性。我们将编写一个程序，创建三个类；一个是`Base`类，另外两个是`Child`类。这两个`Child`类将继承`Parent`类的每一个成员，但它们每个都会独立实现一个方法。这将是方法重写的应用。让我们看一个使用继承的多态性概念的例子：

```py
class Bird:
    def about(self):
        print("Species: Bird")
    def Dance(self):
        print("Not all but some birds can dance")

class Peacock(Bird):
    def Dance(self):
        print("Peacock can dance")
class Sparrow(Bird):
    def Dance(self):
        print("Sparrow can't dance")

>>> peacock = Peacock()
>>> peacock.Dance()
Peacock can dance
>>> sparrow = Sparrow()
>>> sparrow.Dance()
Sparrow can't dance
>>> sparrow.about() #inheritance
Species: Bird
```

你看到的第一件事是`Dance`方法在所有三个类中都是共同的。但在这些类的每一个中，我们对`Dance`方法有不同的规格。这个特性特别有用，因为在某些情况下，我们可能想要定制从`Parent`类继承的方法，在`Child`类中可能没有任何意义。在这种情况下，我们使用与`Child`类内部相同签名的方法重新定义这个方法。这种重新实现方法的技术称为方法重写，通过这个过程创建的不同方法实现了多态性。

现在我们已经学习了面向对象编程的重要概念及其主要特性，比如封装、继承和多态性，是时候利用这些知识来修改我们在上一章中使用 curses 制作的蛇游戏了。由于我们无法使用这些面向对象的原则来使上一章的代码变得不那么混乱和晦涩，我们将使我们的代码更具可重用性和可读性。我们将在下一节开始使用 OOP 修改我们的游戏。

# 蛇游戏实现

在本章中，我们探讨了面向对象编程的各种特性，包括继承、多态性、数据隐藏和封装。我们没有涉及的一个特性，称为方法重载，将在第九章“数据模型实现”中介绍。我们已经学到了足够多关于 OOP 的知识，使我们的代码更易读和可重用。让我们按照传统模式开始这一部分，即头脑风暴和信息收集。

# 头脑风暴和信息收集

正如我们已经讨论过的，面向对象编程与游戏界面编程无关；相反，它是一种使代码更加稳健和更加清晰的范式。因此，我们的界面将类似于由 curses 模块制作的程序——基于文本的终端。然而，我们将使用面向对象的范式来完善我们的代码，并且我们将专注于对象而不是动作和逻辑。我们知道面向对象编程是一种数据驱动的方法。因此，我们的程序必须容纳游戏屏幕和用户事件数据。

我们在游戏中使用面向对象的原则的主要目标如下：

+   将程序分成较小的部分，称为对象。这将使程序更易读，并允许我们轻松跟踪错误和错误。

+   能够通过函数在对象之间进行通信。

+   数据是安全的，因为它不能被外部函数使用。这就是封装。

+   我们将更加注重数据而不是方法或程序。

+   对程序进行修改，如添加属性和方法，可以很容易地完成。

现在，让我们开始头脑风暴并收集一些关于游戏模型的信息。显然，我们必须使用上一章的相同代码来布局游戏和其角色，即`Snake`和`Food`。因此，我们必须为它们各自取两个类。`Snake`和`Food`类将在其中定义控制游戏布局和用户事件的方法。

我们必须使用诸如`KEY_DOWN`、`KEY_UP`、`KEY_LEFT`和`KEY_RIGHT`等 curses 事件来处理蛇角色的移动。让我们来可视化一下基本的类和方法：

1.  首先，我们必须导入 curses 来初始化游戏屏幕并处理用户按键移动。

1.  然后，我们必须导入随机模块，因为一旦蛇吃了食物，我们就必须在随机位置生成食物。

1.  之后，我们初始化常量，如屏幕高度、宽度、默认蛇长度和超时时间。

1.  然后，我们用构造函数声明了`Snake`类，它将初始化蛇的默认位置、窗口、头部位置和蛇的身体。

1.  在`Snake`类内部，我们将添加一些方法，如下：

+   `eat_food`将检查蛇是否吃了食物。如果吃了，蛇的长度将增加。

+   `collision`将检查蛇是否与自身发生了碰撞。

+   `update`将在用户移动并改变`Snake`角色的位置时被调用。

1.  最后，我们声明`Food`类并定义渲染和重置方法来在随机位置生成和删除食物。

现在，让我们通过声明常量和导入必要的模块来开始编写程序。这与上一章没有什么不同——我们将使用 curses 来初始化游戏屏幕并处理用户事件。我们将使用随机模块在游戏控制台上生成一个随机位置，以便我们可以在该位置生成新的食物。

# 声明常量并初始化屏幕

与前一章类似，我们将导入 curses 模块，以便我们可以初始化游戏屏幕并通过指定高度和宽度来自定义它。我们必须声明默认蛇长度和其位置作为常量。以下代码对你来说将是熟悉的，除了`name == "__main__"`模式：

```py
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP
from random import randint

WIDTH = 35
HEIGHT = 20
MAX_X = WIDTH - 2
MAX_Y = HEIGHT - 2
SNAKE_LENGTH = 5
SNAKE_X = SNAKE_LENGTH + 1
SNAKE_Y = 3
TIMEOUT = 100

if __name__ == '__main__':
    curses.initscr()
    curses.beep()
    curses.beep()
    window = curses.newwin(HEIGHT, WIDTH, 0, 0)
    window.timeout(TIMEOUT)
    window.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    window.border(0)
```

在前面的代码中，我们声明了一堆常量来指定高度、宽度、默认蛇长度和超时时间。我们对所有这些术语都很熟悉，除了`__name__ == "__main__"`模式。让我们详细讨论一下：

通过查看这个模式，我们可以得出结论，将`"main"`字符串赋值给 name 变量。就像`__init__()`是一个特殊方法一样，`__name__`是一个特殊变量。每当我们执行脚本文件时，Python 解释器将执行写在零缩进级别的代码。但是在 Python 中，没有像 C/C++中那样自动调用的`main()`函数。因此，Python 解释器将使用特殊的`__name__`变量设置为`__main__`字符串。每当 Python 脚本作为主程序执行时，解释器将使用该字符串设置特殊变量。但是当文件从另一个模块导入时，name 变量的值将设置为该模块的名称。因此，我们可以得出结论，name 变量将确定当前的工作模块。我们可以评估这个模式的工作方式如下：

+   **当前源代码文件是主程序时**：当我们将当前源文件作为主程序运行，即`C:/> python example.py`，解释器将把`"__main__"`字符串赋给特殊的 name 变量，即`name == "__main__"`。

+   **当另一个程序导入您的模块时**：假设任何其他程序是主程序，并且它正在导入我们的模块。`>>> import example`语句将 example 模块导入主程序。现在，Python 解释器将通过删除`.py`扩展名来细化脚本文件的名称，并将该模块名称设置为 name 变量，即`name == "example"`。由于这个原因，写在 example 模块中的代码将对主程序可用。特殊变量设置完成后，Python 解释器将逐行执行语句。

因此，`__name__ == "__main__"`模式可用于执行其中写入的代码，如果源文件直接执行，而不是导入。我们可以得出结论，写在此模式内的代码将被执行。在 Python 中，没有`main()`函数，它是在低级编程语言中自动调用的。

在这种情况下，顶层代码以一个`if`块开始，后面跟着模式的**name**，评估当前的工作模块。如果当前程序是`main`，我们将执行写在`if`块内的代码，通过 curses 初始化游戏屏幕并在游戏中创建一个新窗口。

现在我们已经开始编写一个程序，初始化了游戏屏幕并声明了一些常量，是时候创建一些类了。游戏中有两个角色：`Snake`和`Food`。我们将从现在开始创建两个类，并根据需要对它们进行修改。让我们从创建`Snake`类开始。

# 创建蛇类

在为游戏创建屏幕后，我们的下一个重点将是在屏幕上渲染游戏角色。我们将首先创建`Snake`类。我们知道类将有不同的成员，即属性和方法。正如我们在上一章中提到的，创建`Snake`角色时，我们必须跟踪蛇在游戏窗口中的*x*和*y*位置。为了跟踪蛇的身体位置，我们必须提取蛇的*x*和*y*坐标。我们应该使用字母字符来构成蛇的身体，因为 curses 只支持基于文本的终端。让我们开始创建`Body`类，它将为我们提供蛇的位置并提供蛇身体的字符：

```py
class Body(object):
    def __init__(self, x, y, char='#'):
        self.x = x
        self.y = y
        self.char = char

    def coor(self):
        return self.x, self.y
```

在前面的程序中，`#`用于构成蛇的身体结构。我们在`Body`类内定义了两个成员：构造函数和`coor`方法。`coor`方法用于提取蛇身体的当前坐标。

现在，让我们为游戏角色创建一个类。我们将从`Snake`类开始。我们应该维护一个列出的数据结构，以便我们可以存储蛇的身体位置。应该使用构造函数来初始化这些属性。让我们开始编写`Snake`类的构造函数：

```py
class Snake:
    REV_DIR_MAP = {
        KEY_UP: KEY_DOWN, KEY_DOWN: KEY_UP,
        KEY_LEFT: KEY_RIGHT, KEY_RIGHT: KEY_LEFT,
    }

    def __init__(self, x, y, window):
        self.body_list= [] 
        self.timeout = TIMEOUT
        for i in range(SNAKE_LENGTH, 0, -1):
            self.body_list.append(Body(x - i, y))

        self.body_list.append(Body(x, y, '0'))
        self.window = window
        self.direction = KEY_RIGHT
        self.last_head_coor = (x, y)
        self.direction_map = {
            KEY_UP: self.move_up,
            KEY_DOWN: self.move_down,
            KEY_LEFT: self.move_left,
            KEY_RIGHT: self.move_right
        }

```

在`Snake`类中，我们创建了一个字典。每个键和值表示一个相反的方向。如果您对屏幕上的方向表示感到困惑，请返回到上一章。字符的位置用坐标表示。我们声明了构造函数，它允许我们初始化类的属性。我们创建了`body_list`来保存蛇的身体；一个代表蛇游戏屏幕的窗口对象；蛇的默认方向，即右方向；和一个方向映射，其中包含使用 curses 常量如`KEY_UP`、`KEY_DOWN`、`KEY_LEFT`和`KEY_RIGHT`来容纳角色的移动。

对于每个方向映射，我们调用`move_up`、`move_down`、`move_left`和`move_right`函数。我们将很快创建这些方法。

下面的代码行声明在`Snake`类中，并将蛇身体的坐标添加到`body_list`中。`Body(x-i,y)`语句是`Body`类的实例，它将指定蛇身体的坐标。在`Body`类的构造函数中，`#`用于指定蛇身体的布局：

```py
for i in range(SNAKE_LENGTH, 0, -1):
            self.body_list.append(Body(x - i, y))
```

让我们看一下前面的代码并探索一下。这段代码将扩展`Snake`类的特性：

1.  首先，我们必须通过在`Snake`类中添加一些新成员来开始。我们首先添加一个简单的方法，它将扩展蛇的身体：

```py
      def add_body(self, body_list):
              self.body_list.extend(body_list)
```

1.  现在，我们必须创建另一个方法，将游戏对象渲染到屏幕上。这个程序的一个重要步骤是将蛇的身体渲染到游戏屏幕上。由于我们必须用`#`表示蛇，我们可以使用 curses，并使用`addstr`方法。在下面的渲染方法中，我们循环遍历了蛇的整个`body_list`，并为每个实例添加了`'#'`：

```py
        def render(self):
                    for body in self.body_list:
                        self.window.addstr(body.y, body.x, body.char)
```

1.  现在，让我们创建`Snake`类的对象。我们可以在`name == '__main__'`模式中创建它：

```py
      if __name__ == '__main__':
       #code from preceding topic
       snake = Snake(SNAKE_X, SNAKE_Y, window)

       while True:
       window.clear()
       window.border(0)
       snake.render()
```

在上述程序中，我们创建了一个蛇对象。由于在创建对象时`Snake`类的构造函数将自动调用，我们传入了`SNAKE_X`和`SNAKE_Y`参数，这提供了蛇和窗口的默认位置。窗口对象屏幕是通过 curses 的`newwin`方法创建的。在 while 循环中，我们使用蛇对象调用渲染方法，这将在游戏屏幕中添加一个蛇。

尽管我们已经成功将蛇渲染到游戏控制台中，但我们的游戏还没有准备好测试，因为程序无法处理某些操作，例如用户按键盘上的左、右、上和下键来移动`Snake`角色。我们知道 curses 模块提供了一个方法，让我们可以从用户那里获取输入，并相应地处理它。

# 处理用户事件

在上一章中，我们看到使用 curses 模块从用户那里获取输入并处理输入是非常容易的。在本节中，我们将把这些方法添加到`Snake`类中，因为与用户操作相关的方法与`Snake`角色的移动相关。让我们在`Snake`类中添加一些方法：

```py
def change_direction(self, direction):
        if direction != Snake.REV_DIR_MAP[self.direction]:
            self.direction = direction
```

上述方法将改变蛇的方向。在这里，我们初始化了`REV_DIR_MAP`字典，其中包含表示相反方向的键和值。因此，我们将当前方向传递给这个方法，根据用户按下的事件来改变它。方向参数是从用户那里输入的。

现在，是时候提取蛇的头部和头部的坐标了。我们知道蛇的头部位置在蛇移动时会改变。即使穿过蛇的边界，我们也必须使蛇从另一侧出现。因此，蛇的头部位置将根据用户的移动而改变。我们需要创建一个可以适应这些变化的方法。我们可以使用属性装饰器来实现这一点，它将把`Snake`类的头部属性的更改视为方法。这就像一个 getter。不要被这些术语所压倒，因为我们将在以后的章节中介绍这些内容（列表推导和属性）。话虽如此，让我们来看一下以下示例。这个例子将帮助你理解`@property`装饰器：

```py
class Person:
    def __init__(self,first,last):
        self.first = first
        self.last = last
        self.email = '{0}.{1}@gmail.com'.format(self.first, self.last)

per1 = Person('Ross', 'Geller')
print(per1.first)
print(per1.last)
print(per1.email)

#output
Ross
Geller
Ross.Geller@gmail.com
```

现在，让我们改变`first`属性的值并打印所有这些值：

```py
per1.first = "Rachel"
print(per1.first)
print(per1.email)

#output
Rachel
Ross.Geller@gmail.com
```

你可以清楚地看到，更改没有反映在电子邮件中。电子邮件的名称已经保留了之前的`Ross`值。因此，为了使程序自动适应变化，我们需要将属性设置为装饰器。让我们将电子邮件设置为属性并观察结果：

```py
class Person:
    def __init__(self,first,last):
        self.first = first
        self.last = last

    @property
    def email(self):
        return '{0}.{1}@gmail.com'.format(self.first,self.last)
```

以下代码在 Python shell 中执行：

```py
>>> per1 = Person('Ross', 'Geller')
>>> per1.first = "Racheal"
>>> per1.email()
Racheal.Geller@gmail.com
```

我们对属性所做的更改已经在类的属性中得到了自发的反映，这得益于装饰器属性的帮助。我们将在下一章中详细了解这一点。这只是一个快速的介绍。

我们只涵盖了它，因为这是使蛇的头属性成为属性装饰器的重要部分：

```py
   @property
    def head(self):
        return self.body_list[-1]

    @property
    def coor(self):
        return self.head.x, self.head.y
```

`head`方法将提取列表的最后一个元素，表示蛇的头部。`coor`方法将返回一个包含（*x*，*y*）坐标的元组，表示蛇的头部。

让我们再添加一个函数，用于更新蛇的方向：

```py
 def update(self):
        last_body = self.body_list.pop(0)
        last_body.x = self.body_list[-1].x
        last_body.y = self.body_list[-1].y
        self.body_list.insert(-1, last_body)
        self.last_head_coor = (self.head.x, self.head.y)
        self.direction_map[self.direction]()
```

前面的`update`方法将弹出身体的最后一部分，并将其插入到更新新头部位置之前。

现在，让我们使用 curses 模块处理用户事件：

```py
if __name__ == '__main__':
    #code from preceding topic
    #snake is object of Snake class
    while True:
        event = window.getch()
         if event == 27:
            break

        if event in [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT]:
            snake.change_direction(event)

          if event == 32:
            key = -1
            while key != 32:
                key = window.getch()

        snake.update()
```

我们在上一章的前面代码中学习了工作机制，所以你不应该有任何问题理解它。现在，让我们让蛇朝着某个方向移动。在`Snake`类中，我们之前添加了`direction_map`属性，其中包含了映射到不同函数的字典，如`move_up`、`move_down`、`move_left`和`move_right`。这些函数将根据用户的操作改变蛇的位置：

```py
#These functions are added inside the Snake class
 def move_up(self):
        self.head.y -= 1
        if self.head.y < 1:
            self.head.y = MAX_Y

    def move_down(self):
        self.head.y += 1
        if self.head.y > MAX_Y:
            self.head.y = 1

    def move_left(self):
        self.head.x -= 1
        if self.head.x < 1:
            self.head.x = MAX_X

    def move_right(self):
        self.head.x += 1
        if self.head.x > MAX_X:
            self.head.x = 1
```

我们在上一章中制定了这个逻辑，并将使蛇向上、向下、向左或向右移动。我们可以将屏幕想象成一个包含行和列的矩阵。通过向上的动作，蛇将在 Y 轴上移动，因此 y 位置应该减小；同样，通过向下的动作，蛇将向下移动 Y 轴，因此我们需要增加 y 坐标。对于蛇的左右移动，我们将分别减小和增加 X 轴。

现在，我们已经处理了用户事件，这结束了`Snake`类。如果有碰撞，现在是处理碰撞的时候了。我们还必须向游戏添加另一个角色，即`Food`，这将通过创建一个新类来实现。

# 处理碰撞装饰器属性的帮助。

在这一部分不会创建高尚的逻辑。我们必须检查蛇的头部是否与蛇的身体部分发生了碰撞。这应该通过检查头部的坐标（*y*，*x*）是否与蛇的身体的任何坐标相匹配来完成。因此，让我们制作一个新的`@property`方法，用于检查碰撞：

```py
    @property
    def collided(self):
        return any([body.coor == self.head.coor
                    for body in self.body_list[:-1]])
```

在上面的例子中，如果可迭代对象中的任何项为`True`，则任何函数将返回`True`；否则，它将返回`False`。`any`函数中的语句是一个列表推导语句，用于检查蛇头的坐标是否与蛇身的任何部分的坐标相同。

现在，让我们在主循环中使用`snake`对象调用这个方法：

```py
if __name__ == "__main__": while True:
        #code from preceding topics
        #snake is Snake class object
        if snake.collided:
            break
```

# 添加食物类

我们需要添加到游戏中的下一个角色是`Food`。正如我们已经说过的，我们必须为每个角色创建一个不同的类，因为它们应该具有不同的行为和属性。让我们为`Food`角色创建另一个类。我们将称之为`Food`类。

```py
class Food:
    def __init__(self, window, char='&'):
        self.x = randint(1, MAX_X)
        self.y = randint(1, MAX_Y)
        self.char = char
        self.window = window

    def render(self):
        self.window.addstr(self.y, self.x, self.char)

    def reset(self):
        self.x = randint(1, MAX_X)
        self.y = randint(1, MAX_Y)
```

如果你仔细阅读了本章的*Python 类*部分，这一节不应该让你感到困惑。在 Python 中创建一个类，我们使用`class`关键字，后面跟着类名。然而，我们必须使用括号来表示继承。如果你将括号留空，它们将抛出一个错误。因此，我们在括号内添加了一个对象，这是可选的。你可以简单地移除括号，它们将完美地工作。我们使用了 random 模块中的`randint`方法来在随机位置创建食物。`render`方法将在指定的(*y*,*x*)位置添加`X`字符。

现在，让我们创建`Food`类的对象，并通过调用`render`方法在屏幕上渲染食物：

```py
if __name__ == '__main__':
    food = Food(window, '*')
    while True:
        food.render()
```

你可能还记得，我们创建的逻辑使蛇吃食物的方式与蛇头坐标与食物坐标发生碰撞的逻辑相同。在实际制作这个逻辑之前，我们将为`Snake`类制作另一个方法，用于处理吃食物后的后续逻辑：

```py
def eat_food(self, food):
    food.reset()
    body = Body(self.last_head_coor[0], self.last_head_coor[1])
    self.body_list.insert(-1, body)
```

在蛇吃了食物之后，上述逻辑将被调用。吃了食物之后，我们将重置它，这意味着食物将在下一个随机位置生成。然后，我们将通过将食物的最后一个坐标添加到蛇的身体上来增加身体的位置。

现在，让我们添加一些逻辑，确保我们调用这个方法。正如我们已经讨论过的，逻辑将会很简单：每当蛇头与食物的位置发生碰撞时，我们将调用`eat_food`方法。

```py
if __name__ == '__main__':
#snake is object of Snake class
#food is object of Food class
    while True:
        if snake.head.x == food.x and snake.head.y == food.y:
            snake.eat_food(food)

curses.endwin()
```

让我们运行游戏并观察输出：

![](img/2e913c55-45cb-4b3c-ae48-3d794eb98a42.png)

最后，我们已经用面向对象的范式修改了游戏。你可能觉得使用类和对象更复杂和冗长，但通过更多的练习，你会变得更加熟悉。话虽如此，面向对象编程为我们的程序提供了更多的可读性和可重用性特性。举个例子，如果你在`Snake`角色中发现了一个 bug，你可以通过检查食物的不必要代码来追踪它。现在，让我们跳到下一节，测试游戏并对其进行必要的修改。

# 游戏测试和可能的修改

无法通过按下*F5*直接从 Python 脚本运行 curses 应用程序。因此，我们必须通过命令提示符外部运行它，使用`filename.py`命令。

现在，让我们在游戏中添加分数：

1.  首先，在`Snake`类中将分数值初始化为 0。我们还将在`Snake`类中添加一个`score`方法：

```py
      class Snake:
          self.score = 0
          @property
          def score(self):
              return 'Score : {0}'.format(self.score)
```

1.  现在，我们必须在蛇吃食物时每次增加这个分数。蛇吃食物后将调用的方法是`eat_food`方法。因此，我们将在这个方法中增加分数：

```py
      def eat_food(self, food):
          food.reset()
          body = Body(self.last_head_coor[0], self.last_head_coor[1])
          self.body_list.insert(-1, body)
          self.score += 1
```

1.  现在，让我们使用 curses 窗口对象的`addstr`方法渲染分数：

```py
      while True:
          window.addstr(0, 5, snake.score)
```

1.  上述语句将从蛇对象中调用`score`方法，并在(0,5)位置添加分数。请记住，在 curses 中，第一个位置是 y，第二个位置是 x。

让我们再次运行游戏：

![](img/ffdafeea-9cb8-407a-a07b-35439d08ce74.png)

# 总结

在本章中，我们学习了编程中最重要的范式之一——面向对象编程。我们涵盖了类和对象的所有概念，以使您更容易阅读和编写自己的代码。我们还探讨了如何定义类的成员并访问它们。通过实际示例，我们熟悉了面向对象方法的特性。我们还学习了继承、封装、多态和方法重写。这些特性也将在接下来的章节中使用，所以确保您对这些主题每个都有很好的掌握。

在下一章中，我们将学习列表推导和属性。下一章的目的是找到一种优化代码的方法，使程序在执行方面更短、更快。我们将学习如何处理条件和逻辑，以实现更易读和更易调试的单行代码。我们还将利用这个概念来修改我们的贪吃蛇游戏。
