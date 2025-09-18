# 第五章。设计模式之美

在本章中，我们将学习一些设计模式，这些模式将帮助我们编写更好的软件，使软件可重用且整洁。但是，最大的帮助是它们让开发者能够在架构层面进行思考。它们是针对重复问题的解决方案。虽然学习它们对于 C 和 C++等编译语言非常有帮助，因为它们实际上是解决问题的方案，但在 Python 中，由于语言的动态性和代码的简洁性，开发者通常“只是编写代码”，不需要任何设计模式。这对于以 Python 为第一语言的开发者来说尤其如此。我的建议是学习设计模式，以便能够在架构层面而不是函数和类层面处理信息和设计。

在本章中，我们将涵盖以下主题：

+   观察者模式

+   策略模式

+   单例模式

+   模板模式

+   适配器模式

+   门面模式

+   享元模式

+   命令模式

+   抽象工厂

+   注册模式

+   状态模式

# 观察者模式

**关键 1：向所有听众传播信息。**

这是基本模式，其中一个对象告诉其他对象一些有趣的事情。它在 GUI 应用程序、pub/sub 应用程序以及需要通知大量松散耦合的应用程序组件关于一个源节点发生变化的那些应用程序中非常有用。在以下代码中，`Subject`是其他对象通过`register_observer`注册自己的事件的对象。`observer`对象是监听对象。`observers`开始观察将`observers`对象注册到`Subject`对象的函数。每当`Subject`有事件发生时，它将事件级联到所有`observers`：

```py
import weakref

class Subject(object):
    """Provider of notifications to other objects
    """

    def __init__(self, name):
        self.name = name
        self._observers = weakref.WeakSet()

    def register_observer(self, observer):
        """attach the observing object for this subject
        """
        self._observers.add(observer)
        print("observer {0} now listening on {1}".format(
            observer.name, self.name))

    def notify_observers(self, msg):
        """transmit event to all interested observers
        """
        print("subject notifying observers about {}".format(msg,))
        for observer in self._observers:
            observer.notify(self, msg)

class Observer(object):

    def __init__(self, name):
        self.name = name

    def start_observing(self, subject):
        """register for getting event for a subject
        """
        subject.register_observer(self)

    def notify(self, subject, msg):
        """notify all observers 
        """
        print("{0} got msg from {1} that {2}".format(
            self.name, subject.name, msg))

class_homework = Subject("class homework")
student1 = Observer("student 1")
student2 = Observer("student 2")

student1.start_observing(class_homework)
student2.start_observing(class_homework)

class_homework.notify_observers("result is out")

del student2

class_homework.notify_observers("20/20 passed this sem")
```

前面代码的输出如下：

```py
(tag)[ ch5 ] $ python codes/B04885_05_code_01.py
observer student 1 now listening on class homework
observer student 2 now listening on class homework
subject notifying observers about result is out
student 1 got msg from class homework that result is out
student 2 got msg from class homework that result is out
subject notifying observers about 20/20 passed this sem
student 1 got msg from class homework that 20/20 passed this sem
```

# 策略模式

**关键 2：改变算法的行为。**

有时，同一块代码必须对不同客户端的不同调用有不同的行为。例如，所有国家的时区转换必须处理某些国家的夏令时，并在此类情况下更改其策略。主要用途是切换实现。在这个模式中，算法的行为是在运行时选择的。由于 Python 是一种动态语言，将函数分配给变量并在运行时更改它们是微不足道的。类似于以下代码段，有两种实现来计算税，即`tax_simple`和`tax_actual`。对于以下代码片段，`tax_cal`引用了使用的客户端。可以通过更改对实现函数的引用来更改实现：

```py
TAX_PERCENT = .12

def tax_simple(billamount):
    return billamount * TAX_PERCENT

def tax_actual(billamount):
    if billamount < 500:
        return billamount * (TAX_PERCENT//2)
    else:
        return billamount * TAX_PERCENT

tax_cal = tax_simple
print(tax_cal(400),tax_cal(700))

tax_cal = tax_actual
print(tax_cal(400),tax_cal(700))
```

前面代码片段的输出如下：

```py
48.0 84.0
0.0 84.0
```

但前面实现的问题是，在某一时刻，所有客户端都将看到相同的税务计算策略。我们可以通过一个根据请求参数选择实现的类来改进这一点。在下面的示例中，在 `TaxCalculator` 类的实例中，策略是在运行时对它的每次调用确定的。如果请求是印度 `IN`，则按照印度标准计算税务，如果请求是 `US`，则按照美国标准计算：

```py
TAX_PERCENT = .12

class TaxIN(object):
    def __init__(self,):
        self.country_code = "IN"

    def __call__(self, billamount):
        return billamount * TAX_PERCENT

class TaxUS(object):
    def __init__(self,):
        self.country_code = "US"

    def __call__(self,billamount):
        if billamount < 500:
            return billamount * (TAX_PERCENT//2)
        else:
            return billamount * TAX_PERCENT

class TaxCalculator(object):

    def __init__(self):
        self._impls = [TaxIN(),TaxUS()]

    def __call__(self, country, billamount):
    """select the strategy based on country parameter
    """
        for impl in self._impls:
            if impl.country_code == country:
                return impl(billamount)
        else:
            return None

tax_cal = TaxCalculator()
print(tax_cal("IN", 400), tax_cal("IN", 700))
print(tax_cal("US", 400), tax_cal("US", 700))
```

前面代码的输出如下：

```py
48.0 84.0
0.0 84.0 
```

# 单例模式

**关键 3：为所有人提供相同的视图。**

单例模式保持类所有实例的相同状态。当我们在一个程序中的某个地方更改一个属性时，它将反映在所有对这个实例的引用中。由于模块是全局共享的，我们可以将它们用作单例方法，并且它们中定义的变量在所有地方都是相同的。但是，也存在类似的问题，即当模块被重新加载时，可能需要更多的单例类。我们还可以以下方式使用元类创建单例模式。`six` 是一个第三方库，用于帮助编写在 Python 2 和 Python 3 上可运行的相同代码。

在下面的代码中，`Singleton` 元类有一个注册字典，其中存储了每个新类对应的实例。当任何类请求一个新的实例时，这个类在注册表中搜索，如果找到，则传递旧实例。否则，创建一个新的实例，存储在注册表中，并返回。这可以在下面的代码中看到：

```py
from six import with_metaclass

class Singleton(type):
    _registry = {}

    def __call__(cls, *args, **kwargs):
        print(cls, args, kwargs)
        if cls not in Singleton._registry:
            Singleton._registry[cls] = type.__call__(cls, *args, **kwargs)
        return Singleton._registry[cls]

class Me(with_metaclass(Singleton, object)):

    def __init__(self, data):
        print("init ran", data)
        self.data = data

m = Me(2)
n = Me(3)
print(m.data, n.data)
```

下面的输出是前面代码的结果：

```py
<class '__main__.Me'> (2,) {}
init ran 2
<class '__main__.Me'> (3,) {}
2 2
```

# 模板模式

**关键 4：细化算法以适应用例。**

在这个模式中，我们使用名为“模板方法”的方法来定义算法的骨架，其中将一些步骤推迟到子类中。我们这样做的方式如下，我们分析程序，将其分解为逻辑步骤，这些步骤对于不同的用例是不同的。现在，我们可以在主类中实现这些步骤的默认实现，也可能不实现。主类的子类将实现主类中没有实现的步骤，并且它们可能跳过一些通用步骤的实现。在下面的示例中，`AlooDish` 是具有 `cook` 模板方法的基本类。它适用于正常的土豆炒菜，这些菜有共同的烹饪程序。每个食谱在原料、烹饪时间等方面都有所不同。两种变体 `AlooMatar` 和 `AlooPyaz` 定义了与其他不同的步骤集：

```py
import six

class AlooDish(object):

    def get_ingredients(self,):
        self.ingredients = {}

    def prepare_vegetables(self,):
        for item in six.iteritems(self.ingredients):
            print("take {0} {1} and cut into smaller pieces".format(item[0],item[1]))
        print("cut all vegetables in small pieces")

    def fry(self,):
        print("fry for 5 minutes")

    def serve(self,):
        print("Dish is ready to be served")

    def cook(self,):
        self.get_ingredients()
        self.prepare_vegetables()
        self.fry()
        self.serve()

class AlooMatar(AlooDish):

    def get_ingredients(self,):
        self.ingredients = {'aloo':"1 Kg",'matar':"1/2 kg"}

    def fry(self,):
        print("wait 10 min")

class AlooPyaz(AlooDish):

    def get_ingredients(self):
        self.ingredients = {'aloo':"1 Kg",'pyaz':"1/2 kg"}

aloomatar = AlooMatar()
aloopyaz = AlooPyaz()
print("*******************  aloomatar cook")
aloomatar.cook()
print("******************* aloopyaz cook")
aloopyaz.cook()
```

下面的输出是前面示例代码的结果：

```py
*******************  aloomatar cook
take matar 1/2 kg and cut into smaller pieces
take aloo 1 Kg and cut into smaller pieces
cut all vegetables in small pieces
wait 10 min
Dish is ready to be served
******************* aloopyaz cook
take pyaz 1/2 kg and cut into smaller pieces
take aloo 1 Kg and cut into smaller pieces
cut all vegetables in small pieces
fry for 5 minutes
Dish is ready to be served
```

# 适配器模式

**关键 5：桥接类接口。**

这个模式用于将给定的类适配到新的接口。它解决了接口不匹配的问题。为了演示这一点，让我们假设我们有一个 API 函数用于创建比赛以运行不同的动物。动物应该有一个`running_speed`函数，它告诉它们的速度以便进行比较。`Cat`是这样一个类。现在，如果我们有一个位于不同库中的`Fish`类，它也想参加这个函数，它必须能够知道它的`running_speed`函数。由于改变`Fish`的实现不是一个好的选择，我们可以创建一个`适配器`类，它可以通过提供必要的桥梁来适配`Fish`类以运行：

```py
def running_competition(*list_of_animals):
    if len(list_of_animals)<1:
        print("No one Running")
        return
    fastest_animal = list_of_animals[0]
    maxspeed = fastest_animal.running_speed()
    for animal in list_of_animals[1:]:
        runspeed =  animal.running_speed()
        if runspeed > maxspeed:
            fastest_animal = animal
            maxspeed = runspeed
    print("winner is {0} with {1} Km/h".format(fastest_animal.name,maxspeed))

class Cat(object):

    def __init__(self, name, legs):
        self.name = name
        self.legs = legs

    def running_speed(self,):
        if self.legs>4 :
            return 20
        else:
            return 40

running_competition(Cat('cat_a',4),Cat('cat_b',3))

class Fish(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def swim_speed(self):
        if self.age < 2:
            return 40
        else:
            return 60

# to let our fish to participate in tournament it should have similar interface as
# cat, we can also do this by using an adaptor class RunningFish

class RunningFish(object):
    def __init__(self, fish):
        self.legs = 4 # dummy
        self.fish = fish

    def running_speed(self):
        return self.fish.swim_speed()

    def __getattr__(self, attr):
        return getattr(self.fish,attr)

running_competition(Cat('cat_a',4),
                    Cat('cat_b',3),
                    RunningFish(Fish('nemo',3)),
                    RunningFish(Fish('dollar',1)))
```

上一段代码的输出如下：

```py
winner is cat_a with 40 Km/h
winner is nemo with 60 Km/h
```

# 外观模式

**关键点 6：隐藏系统复杂性以实现更简单的接口。**

在这个模式中，一个称为外观的主要类向客户端类导出一个更简单的接口，并封装了与系统许多其他类的交互复杂性。它就像一个通往复杂功能集的门户，如下例所示，`WalkingDrone`类隐藏了`Leg`类的同步复杂性，并为客户端类提供了一个更简单的接口：

```py
class Leg(object):
    def __init__(self,name):
        self.name = name

    def forward(self):
        print("{0},".format(self.name), end="")

class WalkingDrone(object):

    def __init__(self, name):
        self.name = name
        self.frontrightleg = Leg('Front Right Leg')
        self.frontleftleg = Leg('Front Left Leg')
        self.backrightleg = Leg('Back Right Leg')
        self.backleftleg = Leg('Back Left Leg')

    def walk(self):
        print("\nmoving ",end="")
        self.frontrightleg.forward()
        self.backleftleg.forward()
        print("\nmoving ",end="")
        self.frontleftleg.forward()
        self.backrightleg.forward()

    def run(self):
        print("\nmoving ",end="")
        self.frontrightleg.forward()
        self.frontleftleg.forward()
        print("\nmoving ",end="")
        self.backrightleg.forward()
        self.backleftleg.forward()

wd = WalkingDrone("RoboDrone" )
print("\nwalking")
wd.walk()
print("\nrunning")
wd.run()
```

这段代码将给出以下输出：

```py
walking

moving Front Right Leg,Back Left Leg,
moving Front Left Leg,Back Right Leg,
running

moving Front Right Leg,Front Left Leg,
moving Back Right Leg,Back Left Leg,Summary
```

# 享元模式

**关键点 7：使用共享对象减少内存消耗。**

享元设计模式有助于节省内存。当我们有很多对象计数时，我们存储对先前相似对象的引用，并提供它们而不是创建新对象。在以下示例中，我们有一个浏览器使用的`Link`类，它存储链接数据。

浏览器使用这些数据，并且可能与链接引用的图片关联大量数据，例如图片内容、大小等，图片可以在页面上重复使用。因此，使用它的节点仅存储一个轻量级的`BrowserImage`对象以减少内存占用。当链接类尝试创建一个新的`BrowserImage`实例时，`BrowserImage`类会检查其`_resources`映射中是否有该资源路径的实例。如果有，它将只传递旧实例：

```py
import weakref

class Link(object):

    def __init__(self, ref, text, image_path=None):
        self.ref = ref
        if image_path:
            self.image = BrowserImage(image_path)
        else:
            self.image = None
        self.text = text

    def __str__(self):
        if not self.image:
            return "<Link (%s)>" % self.text
        else:
            return "<Link (%s,%s)>" % (self.text, str(self.image))

class BrowserImage(object):
    _resources = weakref.WeakValueDictionary()

    def __new__(cls, location):
        image = BrowserImage._resources.get(location, None)
        if not image:
            image = object.__new__(cls)
            BrowserImage._resources[location] = image
            image.__init(location)
        return image

    def __init(self, location):
        self.location = location
        # self.content = load picture into memory

    def __str__(self,):
        return "<BrowserImage(%s)>" % self.location

icon = Link("www.pythonunlocked.com",
            "python unlocked book",
            "http://pythonunlocked.com/media/logo.png")
footer_icon = Link("www.pythonunlocked.com/#bottom",
                   "unlocked series python book",
                   "http://pythonunlocked.com/media/logo.png")
twitter_top_header_icon = Link("www.twitter.com/pythonunlocked",
                               "python unlocked twitter link",
                               "http://pythonunlocked.com/media/logo.png")

print(icon,)
print(footer_icon,)
print(twitter_top_header_icon,)
```

上一段代码的输出如下：

```py
<Link (python unlocked book,<BrowserImage(http://pythonunlocked.com/media/logo.png)>)>
<Link (unlocked series python book,<BrowserImage(http://pythonunlocked.com/media/logo.png)>)>
<Link (python unlocked twitter link,<BrowserImage(http://pythonunlocked.com/media/logo.png)>)>
```

# 命令模式

**关键点 8：命令的简单执行管理。**

在这个模式中，我们封装了执行命令所需的信息，以便命令本身可以具有进一步的能力，例如撤销、取消和后续时间点所需的元数据。例如，让我们在一家餐厅中创建一个简单的`Chef`，用户可以发出订单（命令），这里的命令具有用于取消它们的元数据。这类似于记事本应用，其中每个用户操作都会记录一个撤销方法。这使得调用者和执行者之间的耦合变得松散，如下所示：

```py
import time
import threading

class Chef(threading.Thread):

    def __init__(self,name):
        self.q = []
        self.doneq = []
        self.do_orders = True
        threading.Thread.__init__(self,)
        self.name = name
        self.start()

    def makeorder(self, order):
        print("%s Preparing Menu :"%self.name )
        for item in order.items:
            print("cooking ",item)
            time.sleep(1)
        order.completed = True
        self.doneq.append(order)

    def run(self,):
        while self.do_orders:
            if len(self.q) > 0:
                order = self.q.pop(0)
                self.makeorder(order)
                time.sleep(1)

    def work_on_order(self,order):
        self.q.append(order)

    def cancel(self, order):
        if order in self.q:
            if order.completed == True:
                print("cannot cancel, order completed")
                return
            else:
                index = self.q.index(order)
                del self.q[index]
                print(" order canceled %s"%str(order))
                return
        if order in self.doneq:
            print("order completed, cannot be canceled")
            return
        print("Order not given to me")

class Check(object):

    def execute(self,):
        raise NotImplementedError()

    def cancel(self,):
        raise NotImplementedError()

class MenuOrder(Check):

    def __init__(self,*items):
        self.items = items
        self.completed = False

    def execute(self,chef):
        self.chef = chef
        chef.work_on_order(self)

    def cancel(self,):
        if self.chef.cancel(self):
            print("order cancelled")

    def __str__(self,):
        return ''.join(self.items)

c = Chef("Arun")
order1 = MenuOrder("Omellette", "Dosa", "Idli")
order2 = MenuOrder("Mohito", "Pizza")
order3 = MenuOrder("Rajma", )
order1.execute(c)
order2.execute(c)
order3.execute(c)

time.sleep(1)
order3.cancel()
time.sleep(9)
c.do_orders = False
c.join()
```

上一段代码的输出如下：

```py
Arun Preparing Menu :
cooking  Omellette
 order canceled Rajma
cooking  Dosa
cooking  Idli
Arun Preparing Menu :
cooking  Mohito
cooking  Pizza
```

# 抽象工厂

这种设计模式创建了一个接口，用于创建一系列相互关联的对象，而不指定它们的具体类。它类似于一个超级工厂。它的优点是我们可以添加更多的变体，并且客户端无需进一步担心接口或新变体的实际类。它在支持各种平台、窗口系统、数据类型等方面非常有帮助。在以下示例中，`Animal`类是客户端将了解的任何动物实例的接口。`AnimalFactory`是`DogFactory`和`CatFactory`实现的抽象工厂。现在，在运行时通过用户输入、配置文件或运行时环境检查，我们可以决定是否将所有实例都设置为`Dog`或`Cat`。添加新的类实现非常方便，如下所示：

```py
import os
import abc
import six

class Animal(six.with_metaclass(abc.ABCMeta, object)):
    """ clients only need to know this interface for animals"""
    @abc.abstractmethod
    def sound(self, ):
        pass

class AnimalFactory(six.with_metaclass(abc.ABCMeta, object)):
    """clients only need to know this interface for creating animals"""
    @abc.abstractmethod
    def create_animal(self,name):
        pass

class Dog(Animal):
    def __init__(self, name):
        self.name = name

    def sound(self, ):
        print("bark bark")

class DogFactory(AnimalFactory):
    def create_animal(self,name):
        return Dog(name)

class Cat(Animal):
    def __init__(self, name):
        self.name = name
    def sound(self, ):
        print("meow meow")

class CatFactory(AnimalFactory):
    def create_animal(self,name):
        return Cat(name)

class Animals(object):
    def __init__(self,factory):
        self.factory = factory

    def create_animal(self, name):
        return self.factory.create_animal(name)

if __name__ == '__main__':
    atype = input("what animal (cat/dog) ?").lower()
    if atype == 'cat':
        animals = Animals(CatFactory())
    elif atype == 'dog':
        animals = Animals(DogFactory())
    a = animals.create_animal('bulli')
    a.sound()
```

前面的代码将给出以下输出：

```py
1st run:

what animal (cat/dog) ?dog
bark bark

2nd run:
what animal (cat/dog) ?cat
meow meow
```

# 注册模式

**关键 9：从代码的任何位置向类添加功能。**

这是我的最爱之一，并且非常有帮助。在这个模式中，我们将类注册到注册表中，该注册表跟踪命名到功能。因此，我们可以从代码的任何位置向主类添加功能。在以下代码中，`Convertor`跟踪所有从字典到 Python 对象的转换器。我们可以很容易地从代码的任何位置使用`convertor.register`装饰器向系统添加更多功能，如下所示：

```py
class ConvertError(Exception):

    """Error raised on errors on conversion"""
    pass

class Convertor(object):

    def __init__(self,):
        """create registry for storing method mapping """
        self.__registry = {}

    def to_object(self, data_dict):
        """convert to python object based on type of dictionary"""
        dtype = data_dict.get('type', None)
        if not dtype:
            raise ConvertError("cannot create object, type not defined")
        elif dtype not in self.__registry:
            raise ConvertError("cannot convert type not registered")
        else:
            convertor = self.__registry[dtype]
            return convertor.to_python(data_dict['data'])

    def register(self, convertor):
        iconvertor = convertor()
        self.__registry[iconvertor.dtype] = iconvertor

convertor = Convertor()

class Person():

    """ a class in application """

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self,):
        return "<Person (%s, %s)>" % (self.name, self.age)

@convertor.register
class PersonConvertor(object):

    def __init__(self,):
        self.dtype = 'person'

    def to_python(self, data):
        # not checking for errors in dictionary to instance creation
        p = Person(data['name'], data['age'])
        return p

print(convertor.to_object(
    {'type': 'person', 'data': {'name': 'arun', 'age': 12}}))
```

以下是对前面代码的输出：

```py
<Person (arun, 12)>
```

# 状态模式

**关键 10：根据状态改变执行。**

状态机对于控制流的向量依赖于应用程序状态的算法非常有用。类似于在解析带有部分的日志输出时，你可能希望在下一个部分更改解析器逻辑。对于编写允许在特定范围内执行某些命令的网络服务器/客户端代码也非常有用：

```py
def outputparser(loglines):
    state = 'header'
    program,end_time,send_failure= None,None,False
    for line in loglines:
        if state == 'header':
            program = line.split(',')[0]
            state = 'body'
        elif state == 'body':
            if 'send_failure' in line:
                send_failure = True
            if '======' in line:
                state = 'footer'
        elif state == 'footer':
            end_time = line.split(',')[0]
    return program, end_time, send_failure

print(outputparser(['sampleapp,only a sampleapp',
              'logline1  sadfsfdf',
              'logline2 send_failure',
              '=====================',
              '30th Jul 2016,END']))
```

这将给出以下输出：

```py
 ('sampleapp', '30th Jul 2016', True)
```

# 概述

在本章中，我们看到了各种可以帮助我们更好地组织代码的设计模式，在某些情况下，还可以提高性能。模式的好处是它们让你能够超越类去思考，并为你的应用程序架构提供策略。作为本章的结束语，不要为了使用设计模式而编码；当你编码并看到良好的匹配时，再使用设计模式。

现在，我们将进行测试，这对于任何严肃的开发工作都是必须的。
