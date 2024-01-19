# 当对象相似时

在编程世界中，重复的代码被认为是邪恶的。我们不应该在不同的地方有相同或相似的代码的多个副本。

有许多方法可以合并具有类似功能的代码或对象。在本章中，我们将介绍最著名的面向对象原则：继承。正如在第十五章中讨论的那样，*面向对象设计*，继承允许我们在两个或多个类之间创建 is a 关系，将通用逻辑抽象到超类中，并在子类中管理特定细节。特别是，我们将介绍以下内容的 Python 语法和原则：

+   基本继承

+   从内置类型继承

+   多重继承

+   多态和鸭子类型

# 基本继承

从技术上讲，我们创建的每个类都使用继承。所有 Python 类都是名为`object`的特殊内置类的子类。这个类在数据和行为方面提供的很少（它提供的行为都是为了内部使用的双下划线方法），但它确实允许 Python 以相同的方式对待所有对象。

如果我们不明确从不同的类继承，我们的类将自动从`object`继承。然而，我们可以明确声明我们的类从`object`派生，使用以下语法：

```py
class MySubClass(object): 
    pass 
```

这就是继承！从技术上讲，这个例子与我们在第十六章中的第一个例子没有什么不同，*Python 中的对象*，因为如果我们不明确提供不同的**超类**，Python 3 会自动从`object`继承。超类或父类是被继承的类。子类是从超类继承的类。在这种情况下，超类是`object`，而`MySubClass`是子类。子类也被称为从其父类派生，或者说子类扩展了父类。

从示例中你可能已经发现，继承需要比基本类定义多出一点额外的语法。只需在类名和后面的冒号之间的括号内包含父类的名称。这就是我们告诉 Python 新类应该从给定的超类派生的所有内容。

我们如何在实践中应用继承？继承最简单和最明显的用途是向现有类添加功能。让我们从一个简单的联系人管理器开始，跟踪几个人的姓名和电子邮件地址。`Contact`类负责在类变量中维护所有联系人的列表，并为单个联系人初始化姓名和地址：

```py
class Contact:
    all_contacts = []

    def __init__(self, name, email):
        self.name = name
        self.email = email
        Contact.all_contacts.append(self)
```

这个例子向我们介绍了**类变量**。`all_contacts`列表，因为它是类定义的一部分，被这个类的所有实例共享。这意味着只有一个`Contact.all_contacts`列表。我们也可以在`Contact`类的任何实例方法中作为`self.all_contacts`访问它。如果在对象（通过`self`）上找不到字段，那么它将在类上找到，并且因此将引用相同的单个列表。

对于这个语法要小心，因为如果你使用`self.all_contacts`来*设置*变量，你实际上会创建一个**新的**与该对象关联的实例变量。类变量仍然不变，并且可以作为`Contact.all_contacts`访问。

这是一个简单的类，允许我们跟踪每个联系人的一些数据。但是如果我们的一些联系人也是我们需要从中订购物品的供应商呢？我们可以在`Contact`类中添加一个`order`方法，但这将允许人们意外地从客户或家庭朋友的联系人那里订购东西。相反，让我们创建一个新的`Supplier`类，它的行为类似于我们的`Contact`类，但有一个额外的`order`方法：

```py
class Supplier(Contact):
    def order(self, order):
        print(
            "If this were a real system we would send "
            f"'{order}' order to '{self.name}'"
        )
```

现在，如果我们在我们可靠的解释器中测试这个类，我们会发现所有联系人，包括供应商，在它们的`__init__`中都接受名称和电子邮件地址，但只有供应商有一个功能性的订单方法：

```py
>>> c = Contact("Some Body", "somebody@example.net")
>>> s = Supplier("Sup Plier", "supplier@example.net")
>>> print(c.name, c.email, s.name, s.email)
Some Body somebody@example.net Sup Plier supplier@example.net
>>> c.all_contacts
[<__main__.Contact object at 0xb7375ecc>,
 <__main__.Supplier object at 0xb7375f8c>]
>>> c.order("I need pliers")
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
AttributeError: 'Contact' object has no attribute 'order'
>>> s.order("I need pliers")
If this were a real system we would send 'I need pliers' order to
'Sup Plier '  
```

所以，现在我们的`Supplier`类可以做所有联系人可以做的事情（包括将自己添加到`all_contacts`列表中）以及作为供应商需要处理的所有特殊事情。这就是继承的美妙之处。

# 扩展内置类

这种继承的一个有趣用途是向内置类添加功能。在前面看到的`Contact`类中，我们正在将联系人添加到所有联系人的列表中。如果我们还想按名称搜索该列表怎么办？嗯，我们可以在`Contact`类上添加一个搜索方法，但感觉这个方法实际上属于列表本身。我们可以使用继承来实现这一点：

```py
class ContactList(list):
    def search(self, name):
        """Return all contacts that contain the search value
        in their name."""
        matching_contacts = []
        for contact in self:
            if name in contact.name:
                matching_contacts.append(contact)
        return matching_contacts

class Contact:
    all_contacts = ContactList()

    def __init__(self, name, email):
        self.name = name
        self.email = email
        Contact.all_contacts.append(self)
```

我们不是实例化一个普通列表作为我们的类变量，而是创建一个扩展内置`list`数据类型的新`ContactList`类。然后，我们将这个子类实例化为我们的`all_contacts`列表。我们可以测试新的搜索功能如下：

```py
>>> c1 = Contact("John A", "johna@example.net")
>>> c2 = Contact("John B", "johnb@example.net")
>>> c3 = Contact("Jenna C", "jennac@example.net")
>>> [c.name for c in Contact.all_contacts.search('John')]
['John A', 'John B']  
```

你是否想知道我们如何将内置语法`[]`改变成我们可以继承的东西？使用`[]`创建一个空列表实际上是使用`list()`创建一个空列表的快捷方式；这两种语法的行为是相同的：

```py
>>> [] == list()
True  
```

实际上，`[]`语法实际上是所谓的**语法糖**，在幕后调用`list()`构造函数。`list`数据类型是一个我们可以扩展的类。事实上，列表本身扩展了`object`类：

```py
>>> isinstance([], object)
True  
```

作为第二个例子，我们可以扩展`dict`类，它与列表类似，是在使用`{}`语法缩写时构造的类：

```py
class LongNameDict(dict): 
    def longest_key(self): 
        longest = None 
        for key in self: 
            if not longest or len(key) > len(longest): 
                longest = key 
        return longest 
```

这在交互式解释器中很容易测试：

```py
>>> longkeys = LongNameDict()
>>> longkeys['hello'] = 1
>>> longkeys['longest yet'] = 5
>>> longkeys['hello2'] = 'world'
>>> longkeys.longest_key()
'longest yet'  
```

大多数内置类型都可以类似地扩展。常见的扩展内置类包括`object`、`list`、`set`、`dict`、`file`和`str`。数值类型如`int`和`float`有时也会被继承。

# 重写和 super

因此，继承非常适合*向*现有类添加新行为，但是*改变*行为呢？我们的`Contact`类只允许名称和电子邮件地址。这对大多数联系人可能已经足够了，但是如果我们想为我们的亲密朋友添加电话号码呢？

正如我们在第十六章中看到的，*Python 中的对象*，我们可以很容易地在构造后在联系人上设置`phone`属性。但是，如果我们想在初始化时使这个第三个变量可用，我们必须重写`__init__`。重写意味着用子类中的新方法（具有相同名称）更改或替换超类的方法。不需要特殊的语法来做到这一点；子类的新创建的方法会自动被调用，而不是超类的方法。如下面的代码所示：

```py
class Friend(Contact): 
 def __init__(self, name, email, phone):         self.name = name 
        self.email = email 
        self.phone = phone 
```

任何方法都可以被重写，不仅仅是`__init__`。然而，在继续之前，我们需要解决这个例子中的一些问题。我们的`Contact`和`Friend`类有重复的代码来设置`name`和`email`属性；这可能会使代码维护复杂化，因为我们必须在两个或更多地方更新代码。更令人担忧的是，我们的`Friend`类忽略了将自己添加到我们在`Contact`类上创建的`all_contacts`列表中。

我们真正需要的是一种方法，可以从我们的新类内部执行`Contact`类上的原始`__init__`方法。这就是`super`函数的作用；它将对象作为父类的实例返回，允许我们直接调用父类方法：

```py
class Friend(Contact): 
    def __init__(self, name, email, phone): 
 super().__init__(name, email) 
        self.phone = phone 
```

这个例子首先使用`super`获取父对象的实例，并在该对象上调用`__init__`，传入预期的参数。然后进行自己的初始化，即设置`phone`属性。

`super()`调用可以在任何方法内部进行。因此，所有方法都可以通过覆盖和调用`super`进行修改。`super`的调用也可以在方法的任何地方进行；我们不必将调用作为第一行。例如，我们可能需要在将传入参数转发给超类之前操纵或验证传入参数。

# 多重继承

多重继承是一个敏感的话题。原则上，它很简单：从多个父类继承的子类能够访问它们两者的功能。实际上，这并没有听起来那么有用，许多专家程序员建议不要使用它。

作为一个幽默的经验法则，如果你认为你需要多重继承，你可能是错的，但如果你知道你需要它，你可能是对的。

最简单和最有用的多重继承形式被称为**mixin**。mixin 是一个不打算独立存在的超类，而是打算被其他类继承以提供额外的功能。例如，假设我们想要为我们的`Contact`类添加功能，允许向`self.email`发送电子邮件。发送电子邮件是一个常见的任务，我们可能希望在许多其他类上使用它。因此，我们可以编写一个简单的 mixin 类来为我们发送电子邮件：

```py
class MailSender: 
    def send_mail(self, message): 
        print("Sending mail to " + self.email) 
        # Add e-mail logic here 
```

为了简洁起见，我们不会在这里包含实际的电子邮件逻辑；如果你有兴趣学习如何做到这一点，请参阅 Python 标准库中的`smtplib`模块。

这个类并没有做任何特别的事情（实际上，它几乎不能作为一个独立的类运行），但它确实允许我们定义一个新的类，描述了`Contact`和`MailSender`，使用多重继承：

```py
class EmailableContact(Contact, MailSender): 
    pass 
```

多重继承的语法看起来像类定义中的参数列表。在括号内不是包含一个基类，而是包含两个（或更多），用逗号分隔。我们可以测试这个新的混合体，看看 mixin 的工作情况：

```py
>>> e = EmailableContact("John Smith", "jsmith@example.net")
>>> Contact.all_contacts
[<__main__.EmailableContact object at 0xb7205fac>]
>>> e.send_mail("Hello, test e-mail here")
Sending mail to jsmith@example.net  
```

`Contact`初始化器仍然将新联系人添加到`all_contacts`列表中，mixin 能够向`self.email`发送邮件，所以我们知道一切都在运行。

这并不难，你可能想知道关于多重继承的严重警告是什么。我们将在一分钟内讨论复杂性，但让我们考虑一下我们在这个例子中的其他选择，而不是使用 mixin：

+   我们本可以使用单一继承，并将`send_mail`函数添加到子类中。这里的缺点是，邮件功能必须为任何其他需要邮件的类重复。

+   我们可以创建一个独立的 Python 函数来发送电子邮件，并在需要发送电子邮件时以参数的形式调用该函数并提供正确的电子邮件地址（这将是我的选择）。

+   我们本可以探索一些使用组合而不是继承的方法。例如，`EmailableContact`可以将`MailSender`对象作为属性，而不是继承它。

+   我们可以在创建类之后对`Contact`类进行 monkey patching（我们将在第二十章中简要介绍 monkey patching，*Python 面向对象的快捷方式*）。这是通过定义一个接受`self`参数的函数，并将其设置为现有类的属性来完成的。

当混合来自不同类的方法时，多重继承效果还不错，但当我们必须在超类上调用方法时，情况就变得非常混乱。有多个超类。我们怎么知道该调用哪一个？我们怎么知道以什么顺序调用它们？

让我们通过向我们的`Friend`类添加家庭地址来探讨这些问题。我们可能会采取一些方法。地址是一组表示联系人的街道、城市、国家和其他相关细节的字符串。我们可以将这些字符串中的每一个作为参数传递给`Friend`类的`__init__`方法。我们也可以将这些字符串存储在元组、字典或数据类中，并将它们作为单个参数传递给`__init__`。如果没有需要添加到地址的方法，这可能是最好的做法。

另一个选择是创建一个新的`Address`类来保存这些字符串，然后将这个类的实例传递给我们的`Friend`类的`__init__`方法。这种解决方案的优势在于，我们可以为数据添加行为（比如，一个给出方向或打印地图的方法），而不仅仅是静态存储。这是组合的一个例子，正如我们在第十五章中讨论的那样，*面向对象设计*。组合是这个问题的一个完全可行的解决方案，它允许我们在其他实体中重用`Address`类，比如建筑物、企业或组织。

然而，继承也是一个可行的解决方案，这就是我们想要探讨的。让我们添加一个新的类来保存地址。我们将这个新类称为`AddressHolder`，而不是`Address`，因为继承定义了一种是一个关系。说`Friend`类是`Address`类是不正确的，但由于朋友可以有一个`Address`类，我们可以说`Friend`类是`AddressHolder`类。稍后，我们可以创建其他实体（公司，建筑物）也持有地址。然而，这种复杂的命名是一个很好的指示，我们应该坚持组合，而不是继承。但出于教学目的，我们将坚持使用继承。这是我们的`AddressHolder`类：

```py
class AddressHolder: 
    def __init__(self, street, city, state, code): 
        self.street = street 
        self.city = city 
        self.state = state 
        self.code = code 
```

我们只需在初始化时将所有数据放入实例变量中。

# 菱形问题

我们可以使用多重继承将这个新类添加为现有`Friend`类的父类。棘手的部分是现在我们有两个父`__init__`方法，它们都需要被初始化。而且它们需要用不同的参数进行初始化。我们该怎么做呢？嗯，我们可以从一个天真的方法开始：

```py
class Friend(Contact, AddressHolder): 
    def __init__( 
        self, name, email, phone, street, city, state, code): 
 Contact.__init__(self, name, email) 
        AddressHolder.__init__(self, street, city, state, code) 
        self.phone = phone 
```

在这个例子中，我们直接调用每个超类的`__init__`函数，并显式传递`self`参数。这个例子在技术上是有效的；我们可以直接在类上访问不同的变量。但是有一些问题。

首先，如果我们忽略显式调用初始化程序，超类可能会未初始化。这不会破坏这个例子，但在常见情况下可能会导致难以调试的程序崩溃。例如，想象一下尝试将数据插入未连接的数据库。

一个更隐匿的可能性是由于类层次结构的组织而多次调用超类。看看这个继承图：

![](img/aa756ecd-f4b1-4ece-b1ec-50fc35c748fa.png)

`Friend`类的`__init__`方法首先调用`Contact`的`__init__`，这隐式地初始化了`object`超类（记住，所有类都派生自`object`）。然后`Friend`调用`AddressHolder`的`__init__`，这又隐式地初始化了`object`超类。这意味着父类已经被设置了两次。对于`object`类来说，这相对无害，但在某些情况下，这可能会带来灾难。想象一下，每次请求都要尝试两次连接到数据库！

基类应该只被调用一次。是的，但是何时呢？我们先调用`Friend`，然后`Contact`，然后`Object`，然后`AddressHolder`？还是`Friend`，然后`Contact`，然后`AddressHolder`，然后`Object`？

方法的调用顺序可以通过修改类的`__mro__`（**方法解析顺序**）属性来动态调整。这超出了本书的范围。如果您认为您需要了解它，我们建议阅读*Expert Python Programming*，*Tarek Ziadé*，*Packt Publishing*，或者阅读有关该主题的原始文档（注意，它很深！）[`www.python.org/download/releases/2.3/mro/`](http://www.python.org/download/releases/2.3/mro/)。

让我们看一个更清楚地说明这个问题的第二个刻意的例子。在这里，我们有一个基类，它有一个名为`call_me`的方法。两个子类重写了该方法，然后另一个子类使用多重继承扩展了这两个子类。这被称为菱形继承，因为类图的形状是菱形：

![](img/ad8de812-f1cd-43b8-86d2-0c1b13a40b49.png)

让我们将这个图转换成代码；这个例子展示了方法何时被调用：

```py
class BaseClass:
    num_base_calls = 0

    def call_me(self):
        print("Calling method on Base Class")
        self.num_base_calls += 1

class LeftSubclass(BaseClass):
    num_left_calls = 0

    def call_me(self):
        BaseClass.call_me(self)
        print("Calling method on Left Subclass")
        self.num_left_calls += 1

class RightSubclass(BaseClass):
    num_right_calls = 0

    def call_me(self):
        BaseClass.call_me(self)
        print("Calling method on Right Subclass")
        self.num_right_calls += 1

class Subclass(LeftSubclass, RightSubclass):
    num_sub_calls = 0

    def call_me(self):
 LeftSubclass.call_me(self)
 RightSubclass.call_me(self)
        print("Calling method on Subclass")
        self.num_sub_calls += 1
```

这个例子确保每个重写的`call_me`方法直接调用具有相同名称的父方法。它通过将信息打印到屏幕上来告诉我们每次调用方法。它还更新了类的静态变量，以显示它被调用的次数。如果我们实例化一个`Subclass`对象并调用它的方法一次，我们会得到输出：

```py
>>> s = Subclass()
>>> s.call_me()
Calling method on Base Class
Calling method on Left Subclass
Calling method on Base Class
Calling method on Right Subclass
Calling method on Subclass
>>> print(
... s.num_sub_calls,
... s.num_left_calls,
... s.num_right_calls,
... s.num_base_calls)
1 1 1 2  
```

因此，我们可以清楚地看到基类的`call_me`方法被调用了两次。如果该方法正在执行实际工作，比如两次存入银行账户，这可能会导致一些隐匿的错误。

多重继承要记住的一件事是，我们只想调用类层次结构中的`next`方法，而不是`parent`方法。实际上，下一个方法可能不在当前类的父类或祖先上。`super`关键字再次拯救了我们。事实上，`super`最初是为了使复杂的多重继承形式成为可能。以下是使用`super`编写的相同代码：

```py
class BaseClass:
    num_base_calls = 0

    def call_me(self):
        print("Calling method on Base Class")
        self.num_base_calls += 1

class LeftSubclass(BaseClass):
    num_left_calls = 0

    def call_me(self):
 super().call_me()
        print("Calling method on Left Subclass")
        self.num_left_calls += 1

class RightSubclass(BaseClass):
    num_right_calls = 0

    def call_me(self):
 super().call_me()
        print("Calling method on Right Subclass")
        self.num_right_calls += 1

class Subclass(LeftSubclass, RightSubclass):
    num_sub_calls = 0

    def call_me(self):
 super().call_me()
        print("Calling method on Subclass")
        self.num_sub_calls += 1
```

更改非常小；我们只用`super()`调用替换了天真的直接调用，尽管底部子类只调用了一次`super`，而不是必须为左侧和右侧都进行调用。更改足够简单，但是当我们执行它时，看看差异：

```py
>>> s = Subclass()
>>> s.call_me()
Calling method on Base Class
Calling method on Right Subclass
Calling method on Left Subclass
Calling method on Subclass
>>> print(s.num_sub_calls, s.num_left_calls, s.num_right_calls,
s.num_base_calls)
1 1 1 1  
```

看起来不错；我们的基本方法只被调用了一次。但是`super()`在这里实际上是在做什么呢？由于`print`语句是在`super`调用之后执行的，打印输出的顺序是每个方法实际执行的顺序。让我们从后往前看输出，看看是谁在调用什么。

首先，`Subclass`的`call_me`调用了`super().call_me()`，这恰好是在引用

到`LeftSubclass.call_me()`。然后`LeftSubclass.call_me()`方法调用`super().call_me()`，但在这种情况下，`super()`指的是`RightSubclass.call_me()`。

**特别注意**：`super`调用*不*调用`LeftSubclass`的超类（即`BaseClass`）上的方法。相反，它调用`RightSubclass`，即使它不是`LeftSubclass`的直接父类！这是*next*方法，而不是父方法。然后`RightSubclass`调用`BaseClass`，并且`super`调用确保了类层次结构中的每个方法都被执行一次。

# 不同的参数集

当我们返回到我们的`Friend`多重继承示例时，这将使事情变得复杂。在`Friend`的`__init__`方法中，我们最初调用了两个父类的`__init__`，*使用不同的参数集*：

```py
Contact.__init__(self, name, email) 
AddressHolder.__init__(self, street, city, state, code) 
```

在使用`super`时如何管理不同的参数集？我们不一定知道`super`将尝试首先初始化哪个类。即使我们知道，我们也需要一种方法来传递`extra`参数，以便后续对其他子类的`super`调用接收正确的参数。

具体来说，如果对`super`的第一个调用将`name`和`email`参数传递给`Contact.__init__`，然后`Contact.__init__`调用`super`，它需要能够将与地址相关的参数传递给`next`方法，即`AddressHolder.__init__`。

每当我们想要调用具有相同名称但不同参数集的超类方法时，就会出现这个问题。通常情况下，您只会在`__init__`中想要使用完全不同的参数集，就像我们在这里做的那样。即使在常规方法中，我们可能也想要添加仅对一个子类或一组子类有意义的可选参数。

遗憾的是，解决这个问题的唯一方法是从一开始就计划好。我们必须设计基类参数列表，以接受任何不是每个子类实现所需的参数的关键字参数。最后，我们必须确保该方法自由接受意外的参数并将它们传递给其`super`调用，以防它们对继承顺序中的后续方法是必要的。

Python 的函数参数语法提供了我们需要做到这一点的所有工具，但它使整体代码看起来笨重。请看下面`Friend`多重继承代码的正确版本：

```py
class Contact:
    all_contacts = []

 def __init__(self, name="", email="", **kwargs):
 super().__init__(**kwargs)
        self.name = name
        self.email = email
        self.all_contacts.append(self)

class AddressHolder:
 def __init__(self, street="", city="", state="", code="", **kwargs):
 super().__init__(**kwargs)
        self.street = street
        self.city = city
        self.state = state
        self.code = code

class Friend(Contact, AddressHolder):
 def __init__(self, phone="", **kwargs):
 super().__init__(**kwargs)
        self.phone = phone
```

我们通过给它们一个空字符串作为默认值，将所有参数都更改为关键字参数。我们还确保包含一个`**kwargs`参数来捕获我们特定方法不知道如何处理的任何额外参数。它将这些参数传递给`super`调用的下一个类。

如果您不熟悉`**kwargs`语法，它基本上会收集传递给方法的任何未在参数列表中明确列出的关键字参数。这些参数存储在一个名为`kwargs`的字典中（我们可以随意命名变量，但约定建议使用`kw`或`kwargs`）。当我们使用`**kwargs`语法调用不同的方法（例如`super().__init__`）时，它会解包字典并将结果作为普通关键字参数传递给方法。我们将在第二十章中详细介绍这一点，*Python 面向对象的快捷方式*。

前面的例子做了它应该做的事情。但是它开始看起来凌乱，很难回答问题，“我们需要传递什么参数到`Friend.__init__`中？”这是任何计划使用该类的人首要考虑的问题，因此应该在方法中添加一个文档字符串来解释发生了什么。

此外，即使使用这种实现方式，如果我们想要在父类中*重用*变量，它仍然是不够的。当我们将`**kwargs`变量传递给`super`时，字典不包括任何作为显式关键字参数包含的变量。例如，在`Friend.__init__`中，对`super`的调用在`kwargs`字典中没有`phone`。如果其他类中需要`phone`参数，我们需要确保它包含在传递的字典中。更糟糕的是，如果我们忘记这样做，调试将变得非常令人沮丧，因为超类不会抱怨，而只会简单地将默认值（在这种情况下为空字符串）分配给变量。

有几种方法可以确保变量向上传递。假设`Contact`类出于某种原因需要使用`phone`参数进行初始化，并且`Friend`类也需要访问它。我们可以采取以下任一方法：

+   不要将`phone`作为显式关键字参数包含在内。相反，将其留在`kwargs`字典中。`Friend`可以使用`kwargs['phone']`语法查找它。当它将`**kwargs`传递给`super`调用时，`phone`仍将存在于字典中。

+   将`phone`作为显式关键字参数，但在将其传递给`super`之前更新`kwargs`字典，使用标准字典`kwargs['phone'] = phone`语法。

+   将`phone`作为一个显式关键字参数，但使用`kwargs.update`方法更新`kwargs`字典。如果有多个参数需要更新，这是很有用的。您可以使用`dict(phone=phone)`构造函数或`{'phone': phone}`语法创建传递给`update`的字典。

+   将`phone`作为一个显式关键字参数，但使用`super().__init__(phone=phone, **kwargs)`语法将其明确传递给 super 调用。

我们已经涵盖了 Python 中多重继承的许多注意事项。当我们需要考虑所有可能的情况时，我们必须为它们做计划，我们的代码会变得混乱。基本的多重继承可能很方便，但在许多情况下，我们可能希望选择一种更透明的方式来组合两个不同的类，通常使用组合或我们将在第二十二章和第二十三章中介绍的设计模式之一。

我已经浪费了我生命中的整整一天，搜索复杂的多重继承层次结构，试图弄清楚我需要传递到其中一个深度嵌套的子类的参数。代码的作者倾向于不记录他的类，并经常传递 kwargs——以防万一将来可能会需要。这是一个特别糟糕的例子，使用了不需要的多重继承。多重继承是一个新编码者喜欢炫耀的大而复杂的术语，但我建议避免使用它，即使你认为它是一个好选择。当他们以后不得不阅读代码时，你未来的自己和其他编码者会很高兴他们理解你的代码。

# 多态性

我们在《面向对象设计》的第十五章中介绍了多态性。这是一个华丽的名字，描述了一个简单的概念：不同的行为发生取决于使用哪个子类，而不必明确知道子类实际上是什么。举个例子，想象一个播放音频文件的程序。媒体播放器可能需要加载一个`AudioFile`对象，然后`play`它。我们可以在对象上放一个`play()`方法，负责解压或提取音频并将其路由到声卡和扬声器。播放`AudioFile`的行为可能是非常简单的：

```py
audio_file.play() 
```

然而，解压和提取音频文件的过程对不同类型的文件来说是非常不同的。虽然`.wav`文件是未压缩存储的，`.mp3`、`.wma`和`.ogg`文件都使用完全不同的压缩算法。

我们可以使用多态性的继承来简化设计。每种类型的文件可以由`AudioFile`的不同子类表示，例如`WavFile`和`MP3File`。每个子类都会有一个`play()`方法，为了确保正确的提取过程，每个文件的实现方式都会有所不同。媒体播放器对象永远不需要知道它正在引用哪个`AudioFile`的子类；它只是调用`play()`，并以多态的方式让对象处理实际的播放细节。让我们看一个快速的骨架，展示这可能是什么样子：

```py
class AudioFile:
    def __init__(self, filename):
        if not filename.endswith(self.ext):
            raise Exception("Invalid file format")

        self.filename = filename

class MP3File(AudioFile):
    ext = "mp3"

    def play(self):
        print("playing {} as mp3".format(self.filename))

class WavFile(AudioFile):
    ext = "wav"

    def play(self):
        print("playing {} as wav".format(self.filename))

class OggFile(AudioFile):
    ext = "ogg"

    def play(self):
        print("playing {} as ogg".format(self.filename))
```

所有音频文件都会检查初始化时是否给出了有效的扩展名。但你是否注意到父类中的`__init__`方法如何能够从不同的子类访问`ext`类变量？这就是多态性的工作原理。如果文件名不以正确的名称结尾，它会引发异常（异常将在下一章中详细介绍）。`AudioFile`父类实际上并没有存储对`ext`变量的引用，但这并不妨碍它能够在子类上访问它。

此外，`AudioFile`的每个子类以不同的方式实现`play()`（这个例子实际上并不播放音乐；音频压缩算法确实值得单独一本书！）。这也是多态的实现。媒体播放器可以使用完全相同的代码来播放文件，无论它是什么类型；它不关心它正在查看的`AudioFile`的子类是什么。解压音频文件的细节被*封装*。如果我们测试这个例子，它会按照我们的期望工作。

```py
>>> ogg = OggFile("myfile.ogg")
>>> ogg.play()
playing myfile.ogg as ogg
>>> mp3 = MP3File("myfile.mp3")
>>> mp3.play()
playing myfile.mp3 as mp3
>>> not_an_mp3 = MP3File("myfile.ogg")
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "polymorphic_audio.py", line 4, in __init__
 raise Exception("Invalid file format")
Exception: Invalid file format  
```

看看`AudioFile.__init__`如何能够检查文件类型，而不实际知道它指的是哪个子类？

多态实际上是面向对象编程中最酷的东西之一，它使一些在早期范式中不可能的编程设计变得显而易见。然而，由于鸭子类型，Python 使多态看起来不那么令人敬畏。Python 中的鸭子类型允许我们使用*任何*提供所需行为的对象，而无需强制它成为子类。Python 的动态性使这变得微不足道。下面的例子不扩展`AudioFile`，但可以使用完全相同的接口在 Python 中与之交互：

```py
class FlacFile: 
    def __init__(self, filename): 
        if not filename.endswith(".flac"): 
            raise Exception("Invalid file format") 

        self.filename = filename 

    def play(self): 
        print("playing {} as flac".format(self.filename)) 
```

我们的媒体播放器可以像扩展`AudioFile`的对象一样轻松地播放这个对象。

在许多面向对象的上下文中，多态是使用继承的最重要原因之一。因为在 Python 中可以互换使用任何提供正确接口的对象，所以减少了对多态公共超类的需求。继承仍然可以用于共享代码，但如果所有被共享的只是公共接口，那么只需要鸭子类型。这种对继承的需求减少也减少了对多重继承的需求；通常，当多重继承似乎是一个有效的解决方案时，我们可以使用鸭子类型来模仿多个超类中的一个。

当然，只因为一个对象满足特定接口（通过提供所需的方法或属性）并不意味着它在所有情况下都能简单地工作。它必须以在整个系统中有意义的方式满足该接口。仅仅因为一个对象提供了`play()`方法并不意味着它会自动与媒体播放器一起工作。例如，我们在第十五章中的国际象棋 AI 对象，*面向对象设计*，可能有一个`play()`方法来移动国际象棋棋子。即使它满足了接口，这个类在我们试图将它插入媒体播放器时可能会以惊人的方式崩溃！

鸭子类型的另一个有用特性是，鸭子类型的对象只需要提供实际被访问的方法和属性。例如，如果我们需要创建一个假的文件对象来读取数据，我们可以创建一个具有`read()`方法的新对象；如果将与假对象交互的代码不会调用`write`方法，那么我们就不必覆盖`write`方法。简而言之，鸭子类型不需要提供可用对象的整个接口；它只需要满足实际被访问的接口。

# 抽象基类

虽然鸭子类型很有用，但事先很难判断一个类是否能够满足你所需的协议。因此，Python 引入了**抽象基类**（**ABC**）的概念。抽象基类定义了一组类必须实现的方法和属性，以便被视为该类的鸭子类型实例。该类可以扩展抽象基类本身，以便用作该类的实例，但必须提供所有适当的方法。

实际上，很少需要创建新的抽象基类，但我们可能会发现需要实现现有 ABC 的实例的情况。我们将首先介绍实现 ABC，然后简要介绍如何创建自己的 ABC，如果你有需要的话。

# 使用抽象基类

Python 标准库中存在的大多数抽象基类都位于`collections`模块中。其中最简单的之一是`Container`类。让我们在 Python 解释器中检查一下这个类需要哪些方法：

```py
>>> from collections import Container 
>>> Container.__abstractmethods__ 
frozenset(['__contains__']) 
```

因此，`Container`类确切地有一个需要被实现的抽象方法，`__contains__`。你可以发出`help(Container.__contains__)`来查看这个函数签名应该是什么样子的：

```py
Help on method __contains__ in module _abcoll:
 __contains__(self, x) unbound _abcoll.Container method
```

我们可以看到`__contains__`需要接受一个参数。不幸的是，帮助文件并没有告诉我们这个参数应该是什么，但从 ABC 的名称和它实现的单个方法来看，很明显这个参数是用户要检查的容器是否包含的值。

这个方法由`list`、`str`和`dict`实现，用于指示给定的值是否*在*该数据结构中。然而，我们也可以定义一个愚蠢的容器，告诉我们给定的值是否在奇数集合中：

```py
class OddContainer: 
    def __contains__(self, x): 
        if not isinstance(x, int) or not x % 2: 
            return False 
        return True 
```

有趣的是：我们可以实例化一个`OddContainer`对象，并确定，即使我们没有扩展`Container`，该类也是一个`Container`对象。

```py
>>> from collections import Container 
>>> odd_container = OddContainer() 
>>> isinstance(odd_container, Container) 
True 
>>> issubclass(OddContainer, Container) 
True 
```

这就是为什么鸭子类型比经典多态更棒的原因。我们可以创建关系而不需要编写设置继承（或更糟的是多重继承）的代码的开销。

`Container` ABC 的一个很酷的地方是，任何实现它的类都可以免费使用`in`关键字。实际上，`in`只是语法糖，委托给`__contains__`方法。任何具有`__contains__`方法的类都是`Container`，因此可以通过`in`关键字查询，例如：

```py
>>> 1 in odd_container 
True 
>>> 2 in odd_container 
False 
>>> 3 in odd_container 
True 
>>> "a string" in odd_container 
False 
```

# 创建一个抽象基类

正如我们之前看到的，要启用鸭子类型并不需要有一个抽象基类。然而，想象一下我们正在创建一个带有第三方插件的媒体播放器。在这种情况下，最好创建一个抽象基类来记录第三方插件应该提供的 API（文档是 ABC 的一个更强大的用例）。`abc`模块提供了你需要做到这一点的工具，但我提前警告你，这利用了 Python 中一些最深奥的概念，就像下面的代码块中所演示的那样：

```py
import abc 

class MediaLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def play(self):
        pass

    @abc.abstractproperty
    def ext(self):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is MediaLoader:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True

        return NotImplemented
```

这是一个复杂的例子，包括了几个 Python 特性，这些特性在本书的后面才会被解释。它被包含在这里是为了完整性，但你不需要理解所有这些来了解如何创建你自己的 ABC。

第一件奇怪的事情是`metaclass`关键字参数被传递到类中，而在通常情况下你会看到父类列表。这是来自元类编程的神秘艺术中很少使用的构造。我们不会在本书中涵盖元类，所以你需要知道的是，通过分配`ABCMeta`元类，你为你的类赋予了超级英雄（或至少是超类）的能力。

接下来，我们看到了`@abc.abstractmethod`和`@abc.abstractproperty`构造。这些是 Python 装饰器。我们将在第二十二章中讨论这些。现在，只需要知道通过将方法或属性标记为抽象，你声明了这个类的任何子类必须实现该方法或提供该属性，才能被视为该类的合格成员。

看看如果你实现了提供或不提供这些属性的子类会发生什么：

```py
>>> class Wav(MediaLoader): 
...     pass 
... 
>>> x = Wav() 
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module> 
TypeError: Can't instantiate abstract class Wav with abstract methods ext, play 
>>> class Ogg(MediaLoader): 
...     ext = '.ogg' 
...     def play(self): 
...         pass 
... 
>>> o = Ogg() 
```

由于`Wav`类未实现抽象属性，因此无法实例化该类。该类仍然是一个合法的抽象类，但你必须对其进行子类化才能实际执行任何操作。`Ogg`类提供了这两个属性，因此可以干净地实例化。

回到`MediaLoader` ABC，让我们解剖一下`__subclasshook__`方法。它基本上是说，任何提供了这个 ABC 所有抽象属性的具体实现的类都应该被认为是`MediaLoader`的子类，即使它实际上并没有继承自`MediaLoader`类。

更常见的面向对象语言在接口和类的实现之间有明确的分离。例如，一些语言提供了一个明确的`interface`关键字，允许我们定义一个类必须具有的方法，而不需要任何实现。在这样的环境中，抽象类是提供了接口和一些但不是所有方法的具体实现的类。任何类都可以明确声明它实现了给定的接口。

Python 的 ABCs 有助于提供接口的功能，而不会影响鸭子类型的好处。

# 解密魔术

如果你想要创建满足这个特定契约的抽象类，你可以复制并粘贴子类代码而不必理解它。我们将在本书中涵盖大部分不寻常的语法，但让我们逐行地概述一下：

```py
    @classmethod 
```

这个装饰器标记方法为类方法。它基本上表示该方法可以在类上调用，而不是在实例化的对象上调用：

```py
    def __subclasshook__(cls, C): 
```

这定义了`__subclasshook__`类方法。这个特殊的方法是由 Python 解释器调用来回答这个问题：类`C`是这个类的子类吗？

```py
        if cls is MediaLoader: 
```

我们检查方法是否是在这个类上专门调用的，而不是在这个类的子类上调用。例如，这可以防止`Wav`类被认为是`Ogg`类的父类：

```py
            attrs = set(dir(C)) 
```

这一行所做的只是获取类的方法和属性集，包括其类层次结构中的任何父类：

```py
            if set(cls.__abstractmethods__) <= attrs: 
```

这一行使用集合符号来查看候选类中是否提供了这个类中的抽象方法。请注意，它不检查方法是否已经被实现；只是检查它们是否存在。因此，一个类可能是一个子类，但仍然是一个抽象类本身。

```py
                return True 
```

如果所有的抽象方法都已经提供，那么候选类是这个类的子类，我们返回`True`。该方法可以合法地返回三个值之一：`True`，`False`或`NotImplemented`。`True`和`False`表示该类是否明确是这个类的子类：

```py
return NotImplemented 
```

如果任何条件都没有被满足（也就是说，这个类不是`MediaLoader`，或者没有提供所有的抽象方法），那么返回`NotImplemented`。这告诉 Python 机制使用默认机制（候选类是否明确扩展了这个类？）来检测子类。

简而言之，我们现在可以将`Ogg`类定义为`MediaLoader`类的子类，而不实际扩展`MediaLoader`类：

```py
>>> class Ogg(): ... ext = '.ogg' ... def play(self): ... print("this will play an ogg file") ... >>> issubclass(Ogg, MediaLoader) True >>> isinstance(Ogg(), MediaLoader) True
```

# 案例研究

让我们尝试用一个更大的例子把我们学到的东西联系起来。我们将为编程作业开发一个自动评分系统，类似于 Dataquest 或 Coursera 使用的系统。该系统需要为课程作者提供一个简单的基于类的接口，以便创建他们的作业，并且如果不满足该接口，应该提供有用的错误消息。作者需要能够提供他们的课程内容，并编写自定义答案检查代码，以确保他们的学生得到正确的答案。他们还可以访问学生的姓名，使内容看起来更友好一些。

评分系统本身需要跟踪学生当前正在进行的作业。学生可能在得到正确答案之前尝试几次作业。我们希望跟踪尝试次数，以便课程作者可以改进更难的课程内容。

让我们首先定义课程作者需要使用的接口。理想情况下，除了课程内容和答案检查代码之外，它将要求课程作者写入最少量的额外代码。以下是我能想到的最简单的类：

```py
class IntroToPython:
    def lesson(self):
        return f"""
            Hello {self.student}. define two variables,
            an integer named a with value 1
            and a string named b with value 'hello'

        """

```

```py
    def check(self, code):
        return code == "a = 1\nb = 'hello'"
```

诚然，该课程作者可能对他们的答案检查方式有些天真。

我们可以从定义这个接口的抽象基类开始，如下所示：

```py
class Assignment(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def lesson(self, student):
        pass

    @abc.abstractmethod
    def check(self, code):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Assignment:
            attrs = set(dir(C))
            if set(cls.__abstractmethods__) <= attrs:
                return True

        return NotImplemented
```

这个 ABC 定义了两个必需的抽象方法，并提供了魔术`__subclasshook__`方法，允许一个类被视为子类，而无需明确扩展它（我通常只是复制并粘贴这段代码。不值得记忆。）

我们可以使用`issubclass(IntroToPython, Assignment)`来确认`IntroToPython`类是否满足这个接口，这应该返回`True`。当然，如果愿意，我们也可以明确扩展`Assignment`类，就像在第二个作业中所看到的那样：

```py
class Statistics(Assignment):
    def lesson(self):
        return (
            "Good work so far, "
            + self.student
            + ". Now calculate the average of the numbers "
            + " 1, 5, 18, -3 and assign to a variable named 'avg'"
        )

    def check(self, code):
        import statistics

        code = "import statistics\n" + code

        local_vars = {}
        global_vars = {}
        exec(code, global_vars, local_vars)

        return local_vars.get("avg") == statistics.mean([1, 5, 18, -3])
```

不幸的是，这位课程作者也相当天真。`exec`调用将在评分系统内部执行学生的代码，使他们可以访问整个系统。显然，他们将首先对系统进行黑客攻击，使他们的成绩达到 100%。他们可能认为这比正确完成作业更容易！

接下来，我们将创建一个类，用于管理学生在特定作业上尝试的次数：

```py
class AssignmentGrader:
    def __init__(self, student, AssignmentClass):
        self.assignment = AssignmentClass()
        self.assignment.student = student
        self.attempts = 0
        self.correct_attempts = 0

    def check(self, code):
        self.attempts += 1
        result = self.assignment.check(code)
        if result:
            self.correct_attempts += 1

        return result

    def lesson(self):
        return self.assignment.lesson()
```

这个类使用组合而不是继承。乍一看，这些方法存在于`Assignment`超类似乎是有道理的。这将消除令人讨厌的`lesson`方法，它只是代理到作业对象上的相同方法。当然，可以直接在`Assignment`抽象基类上放置所有这些逻辑，甚至可以让 ABC 从这个`AssignmentGrader`类继承。事实上，我通常会推荐这样做，但在这种情况下，这将强制所有课程作者明确扩展该类，这违反了我们尽可能简单地请求内容创作的要求。

最后，我们可以开始组建`Grader`类，该类负责管理哪些作业是可用的，每个学生当前正在进行哪个作业。最有趣的部分是注册方法：

```py
import uuid

class Grader:
    def __init__(self):
        self.student_graders = {}
        self.assignment_classes = {}

    def register(self, assignment_class):
        if not issubclass(assignment_class, Assignment):
            raise RuntimeError(
                "Your class does not have the right methods"
            )

        id = uuid.uuid4()
        self.assignment_classes[id] = assignment_class
        return id
```

这个代码块包括初始化器，其中包括我们将在一分钟内讨论的两个字典。`register`方法有点复杂，所以我们将彻底剖析它。

第一件奇怪的事是这个方法接受的参数：`assignment_class`。这个参数意味着是一个实际的类，而不是类的实例。记住，类也是对象，可以像其他类一样传递。鉴于我们之前定义的`IntroToPython`类，我们可以在不实例化的情况下注册它，如下所示：

```py
from grader import Grader
from lessons import IntroToPython, Statistics

grader = Grader()
itp_id = grader.register(IntroToPython)
```

该方法首先检查该类是否是`Assignment`类的子类。当然，我们实现了一个自定义的`__subclasshook__`方法，因此这包括了不明确地作为`Assignment`子类的类。命名可能有点欺骗性！如果它没有这两个必需的方法，它会引发一个异常。异常是我们将在下一章详细讨论的一个主题；现在，只需假设它会使程序生气并退出。

然后，我们生成一个随机标识符来表示特定的作业。我们将`assignment_class`存储在一个由该 ID 索引的字典中，并返回该 ID，以便调用代码将来可以查找该作业。据推测，另一个对象将在某种课程大纲中放置该 ID，以便学生按顺序完成作业，但在项目的这一部分我们不会这样做。

`uuid`函数返回一个称为通用唯一标识符的特殊格式字符串，也称为全局唯一标识符。它基本上代表一个几乎不可能与另一个类似生成的标识符冲突的极大随机数。这是创建用于跟踪项目的任意 ID 的一种很好、快速和干净的方法。

接下来，我们有`start_assignment`函数，它允许学生开始做一项作业，给定该作业的 ID。它所做的就是构造我们之前定义的`AssignmentGrader`类的一个实例，并将其放入存储在`Grader`类上的字典中，如下所示：

```py
    def start_assignment(self, student, id):
        self.student_graders[student] = AssignmentGrader(
            student, self.assignment_classes[id]
        )
```

之后，我们编写了一些代理方法，用于获取学生当前正在进行的课程或检查作业的代码：

```py
    def get_lesson(self, student):
        assignment = self.student_graders[student]
        return assignment.lesson()

    def check_assignment(self, student, code):
        assignment = self.student_graders[student]
        return assignment.check(code)
```

最后，我们创建了一个方法，用于总结学生当前作业的进展情况。它查找作业对象，并创建一个格式化的字符串，其中包含我们对该学生的所有信息：

```py

    def assignment_summary(self, student):
        grader = self.student_graders[student]
        return f"""
        {student}'s attempts at {grader.assignment.__class__.__name__}:

        attempts: {grader.attempts}
        correct: {grader.correct_attempts}

        passed: {grader.correct_attempts > 0}
        """
```

就是这样。您会注意到，这个案例研究并没有使用大量的继承，这可能看起来有点奇怪，因为这一章的主题，但鸭子类型非常普遍。Python 程序通常被设计为使用继承，随着迭代的进行，它会简化为更多功能的构造。举个例子，我最初将`AssignmentGrader`定义为继承关系，但中途意识到最好使用组合，原因如前所述。

以下是一些测试代码，展示了所有这些对象是如何连接在一起的：

```py
grader = Grader()
itp_id = grader.register(IntroToPython)
stat_id = grader.register(Statistics)

grader.start_assignment("Tammy", itp_id)
print("Tammy's Lesson:", grader.get_lesson("Tammy"))
print(
    "Tammy's check:",
    grader.check_assignment("Tammy", "a = 1 ; b = 'hello'"),
)
print(
    "Tammy's other check:",
    grader.check_assignment("Tammy", "a = 1\nb = 'hello'"),
)

print(grader.assignment_summary("Tammy"))

grader.start_assignment("Tammy", stat_id)
print("Tammy's Lesson:", grader.get_lesson("Tammy"))
print("Tammy's check:", grader.check_assignment("Tammy", "avg=5.25"))
print(
    "Tammy's other check:",
    grader.check_assignment(
        "Tammy", "avg = statistics.mean([1, 5, 18, -3])"
    ),
)

print(grader.assignment_summary("Tammy"))
```

# 练习

看看你的工作空间中的一些物理物体，看看你能否用继承层次结构描述它们。人类几个世纪以来一直在将世界划分为这样的分类法，所以这应该不难。在对象类之间是否存在一些非明显的继承关系？如果你要在计算机应用程序中对这些对象进行建模，它们会共享哪些属性和方法？哪些属性需要多态地重写？它们之间有哪些完全不同的属性？

现在写一些代码。不是为了物理层次结构；那很无聊。物理物品比方法更多。只是想想你过去一年想要解决的宠物编程项目。无论你想解决什么问题，都试着想出一些基本的继承关系，然后实现它们。确保你也注意到了实际上不需要使用继承的关系。有哪些地方你可能想要使用多重继承？你确定吗？你能看到任何你想使用混入的地方吗？试着拼凑一个快速的原型。它不必有用，甚至不必部分工作。你已经看到了如何使用`python -i`测试代码；只需编写一些代码并在交互式解释器中测试它。如果它有效，再写一些。如果不行，修复它！

现在，看看案例研究中的学生评分系统。它缺少很多东西，不仅仅是良好的课程内容！学生如何进入系统？是否有一个课程大纲规定他们应该按照什么顺序学习课程？如果你将`AssignmentGrader`更改为在`Assignment`对象上使用继承而不是组合，会发生什么？

最后，尝试想出一些使用混入的好用例，然后尝试使用混入，直到意识到可能有更好的设计使用组合！

# 总结

我们已经从简单的继承，这是面向对象程序员工具箱中最有用的工具之一，一直到多重继承——最复杂的之一。继承可以用来通过继承向现有类和内置类添加功能。将类似的代码抽象成父类可以帮助增加可维护性。父类上的方法可以使用`super`进行调用，并且在使用多重继承时，参数列表必须安全地格式化以使这些调用起作用。抽象基类允许您记录一个类必须具有哪些方法和属性才能满足特定接口，并且甚至允许您更改*子类*的定义。

在下一章中，我们将介绍处理特殊情况的微妙艺术。
