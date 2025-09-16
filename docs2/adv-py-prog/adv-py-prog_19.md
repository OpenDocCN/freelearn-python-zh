# 第十六章：*第十六章*: 工厂模式

**设计模式**是可重用的编程解决方案，已在各种实际应用场景中使用，并已被证明能够产生预期的结果。在本章中，我们将学习最常见的设计模式之一：工厂设计模式。正如我们稍后将看到的，这种模式使得跟踪程序中创建的对象变得更加容易，从而将创建对象的代码与使用它的代码分离。我们将研究工厂设计模式的两种形式：**工厂方法**和**抽象方法**。

在本章中，我们将涵盖以下主题：

+   理解设计模式

+   实现工厂方法

+   应用抽象工厂

在本章结束时，我们将通过一个实际案例，深入理解工厂设计模式及其益处。

# 技术要求

本章的代码文件可以在[`github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter16`](https://github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter16)找到。

# 理解设计模式

设计模式在程序员之间共享，并且随着时间的推移不断得到改进。这个主题之所以受欢迎，得益于 Erich Gamma、Richard Helm、Ralph Johnson 和 John Vlissides 合著的书籍，书名为《设计模式：可重用面向对象软件元素》。

四人帮

Erich Gamma、Richard Helm、Ralph Johnson 和 John Vlissides 合著的书籍也简称为*四人帮*书籍（或简称为*GOF*书籍）。

通常，设计模式帮助程序员创建常用的实现模式，尤其是在**面向对象编程**（**OOP**）中。从设计模式的角度看待应用程序的好处很多。首先，它缩小了构建特定应用程序的最有效方法和必要的步骤。其次，你可以参考相同设计模式的现有示例来改进你的应用程序。总的来说，设计模式是软件工程中非常有用的指导方针。

根据它们解决的问题类型和/或帮助构建的解决方案类型，在面向对象编程（OOP）中使用了多种设计模式类别。在他们的书中，“四人帮”提出了 23 种设计模式，分为三个类别：**创建型**、**结构型**和**行为型**。

`__init__()`函数，不太方便。

重要提示

欲了解对象类和 Python 用于初始化新类实例的特殊`__init__()`方法的快速概述，请参阅[`docs.python.org/3/tutorial/classes.html`](https://docs.python.org/3/tutorial/classes.html)。

我们将从 *Gang of Four* 书中的第一个创建型设计模式开始：**工厂设计模式**。在工厂设计模式中，**客户端**（意味着客户端代码）请求一个对象，而不知道这个对象是从哪里来的（也就是说，使用了哪个类来生成它）。工厂背后的想法是简化对象创建过程。如果通过一个中央函数来完成，那么跟踪哪些对象被创建比让客户端使用直接类实例化来创建对象要容易得多。工厂通过解耦创建对象的代码和使用它的代码来降低维护应用程序的复杂性。

工厂通常有两种形式：**工厂方法**，这是一个方法（对于 Python 开发者来说，简单地说就是一个函数），它根据输入参数返回不同的对象，以及**抽象工厂**，它是一组用于创建相关对象族的工厂方法。

这就是我们开始所需的所有理论。在下一节中，我们将讨论工厂方法。

# 实现工厂方法

工厂方法基于一个单一的功能，该功能被编写来处理我们的对象创建任务。我们执行它，传递一个参数，提供我们想要的信息。结果，我们想要的那个对象就被创建了。

有趣的是，当我们使用工厂方法时，我们不需要了解关于结果对象实现细节和来源的任何信息。首先，我们将讨论一些使用工厂方法的现实生活应用，然后实现一个示例应用程序，该应用程序处理 XML 和 JSON 文件。

## 真实世界的例子

真实生活中使用的工厂方法模式的例子是在塑料玩具构建套件的上下文中。用于构建塑料玩具的成型材料是相同的，但可以使用正确的塑料模具生产不同的玩具（不同的形状或形状）。这就像有一个工厂方法，其中输入是我们想要的玩具名称（例如，`duck` 或 `car`），输出（成型后）是我们请求的塑料玩具。

在软件世界中，*Django* 网络框架使用工厂方法模式来创建网页表单的字段。Django 包含的 `forms` 模块支持创建不同类型的字段（例如，`CharField`、`EmailField` 等）。它们的行为部分可以通过 `max_length` 或 `required` 等属性进行自定义 ([j.mp/djangofac](http://j.mp/djangofac))。

考虑以下示例：

```py
from django import forms
class PersonForm(forms.Form):
    name = forms.CharField(max_length=100)
    birth_date = forms.DateField(required=False)
```

上述代码可以由一个开发者编写，作为 Django 应用程序 UI 代码的一部分，用于表单（包含 `name` 和 `birth_date` 字段的 `PersonForm` 表单）。

## 用例

如果你意识到你无法追踪应用程序创建的对象，因为创建它们的代码分布在许多不同的地方而不是一个单独的函数/方法中，你应该考虑使用工厂方法模式。工厂方法集中化对象创建，跟踪对象变得容易得多。请注意，创建多个工厂方法是绝对可以的，这也是实践中通常的做法。每个工厂方法逻辑上分组具有相似性的对象。例如，一个工厂方法可能负责连接到不同的数据库（MySQL 和 SQLite），另一个工厂方法可能负责创建你请求的几何对象（圆形和三角形），等等。

当你想要将对象创建与对象使用解耦时，工厂方法也很有用。在创建对象时，我们并不绑定到特定的类；我们只是通过调用一个函数来提供关于我们想要什么的部分信息。这意味着引入对函数的更改很容易，并且不需要对其使用的代码进行任何更改。

另一个值得提到的用例是与提高应用程序的性能和内存使用相关。工厂方法可以通过仅在必要时创建新对象来提高性能和内存使用。当我们使用直接类实例化创建对象时，每次创建新对象都会分配额外的内存（除非类内部使用缓存，这通常不是情况）。我们可以在以下代码（在`id.py`文件中）中看到这一点，它创建了两个相同类`A`的实例，并使用`id()`函数来比较它们的**内存地址**。这些地址也打印在输出中，以便我们可以检查它们。内存地址不同的事实意味着创建了两个不同的对象，如下所示：

```py
class A:
    pass
if __name__ == '__main__':
    a = A()
    b = A()    
    print(id(a) == id(b))
    print(a, b)
```

在我的电脑上执行`python id.py`命令会产生以下输出：

```py
False
<__main__.A object at 0x7f5771de8f60> <__main__.A object at 
0x7f5771df2208>
```

注意，当你执行文件时看到的地址与我看到的地址不同，因为它们取决于当前的内存布局和分配。但结果必须相同：这两个地址应该是不同的。有一个例外，如果你在 Python **读取-评估-打印循环**（**REPL**）中编写和执行代码——简单来说，就是交互式提示符——但这是一种 REPL 特定的优化，通常不会发生。

## 实现工厂方法

数据以多种形式存在。存储/检索数据的主要文件类别有两种：可读文件和二进制文件。可读文件的例子包括 XML、RSS/Atom、YAML 和 JSON。二进制文件的例子包括 SQLite 使用的`.sq3`文件格式和用于听音乐的`.mp3`音频文件格式。

在这个例子中，我们将关注两种流行的人可读格式：**XML** 和 **JSON**。尽管人可读文件通常比二进制文件解析速度慢，但它们使数据交换、检查和修改变得容易得多。因此，建议除非有其他限制（主要是不可接受的性能和专有二进制格式），否则您应使用人可读文件。

在这种情况下，我们有一些输入数据存储在 XML 文件和 JSON 文件中，我们想要解析它们并检索一些信息。同时，我们想要集中管理客户端对这些（以及所有未来的）外部服务的连接。我们将使用工厂方法来解决这个问题。此示例仅关注 XML 和 JSON，但添加对更多服务的支持应该是直接的。

首先，让我们看一下数据文件。

JSON 文件 `movies.json` 可以在本章代码文件夹的 `data` 子文件夹中找到，是一个包含有关美国电影（标题、年份、导演姓名、类型等）信息的示例数据集。这是一个大文件，但这里展示其内容的一部分以说明其组织方式：

```py
[
 {"title":"After Dark in Central Park",
  "year":1900, 
  "director":null, "cast":null, "genre":null},
 {"title":"Boarding School Girls' Pajama Parade",
  "year":1900, 
  "director":null, "cast":null, "genre":null},
 {"title":"Buffalo Bill's Wild West Parad",
  "year":1900, 
  "director":null, "cast":null, "genre":null},
 {"title":"Caught",
  "year":1900, 
  "director":null, "cast":null, "genre":null},
 {"title":"Clowns Spinning Hats",
  "year":1900, 
  "director":null, "cast":null, "genre":null},
 ...
  ]
```

XML 文件 `person.xml` 基于维基百科示例 ([j.mp/wikijson](http://j.mp/wikijson))，包含有关个人（`firstName`、`lastName`、`gender` 等）的信息，如下所示：

1.  我们从 `persons` XML 容器的封装标签开始：

    ```py
    <persons> 
    ```

1.  然后，展示了一个代表个人数据代码的 XML 元素，如下所示：

    ```py
    <person> 
      <firstName>John</firstName> 
      <lastName>Smith</lastName> 
      <age>25</age> 
      <address> 
        <streetAddress>21 2nd Street</streetAddress> 
        <city>New York</city> 
        <state>NY</state> 
        <postalCode>10021</postalCode> 
      </address> 
      <phoneNumbers> 
        <phoneNumber type="home">
          212 555-1234</phoneNumber> 
        <phoneNumber type="fax">646 555-4567</phoneNumber> 
      </phoneNumbers> 
      <gender> 
        <type>male</type> 
      </gender> 
    </person> 
    ```

1.  必须提供另一个代表另一个人数据的 XML 元素：

    ```py
    <person> 
      <firstName>Jimy</firstName> 
      <lastName>Liar</lastName> 
      <age>19</age> 
      <address> 
        <streetAddress>18 2nd Street</streetAddress> 
        <city>New York</city> 
        <state>NY</state> 
        <postalCode>10021</postalCode> 
      </address> 
      <phoneNumbers> 
      <phoneNumber type="home">212 555-1234</phoneNumber> 
      </phoneNumbers> 
      <gender> 
        <type>male</type> 
      </gender> 
    </person> 
    ```

1.  然后展示了一个代表第三个人数据的 XML 元素：

    ```py
    <person> 
      <firstName>Patty</firstName> 
      <lastName>Liar</lastName> 
      <age>20</age> 
      <address> 
        <streetAddress>18 2nd Street</streetAddress> 
        <city>New York</city> 
        <state>NY</state> 
        <postalCode>10021</postalCode> 
      </address> 
      <phoneNumbers> 
        <phoneNumber type="home">
          212 555-1234</phoneNumber> 
        <phoneNumber type="mobile">
          001 452-8819</phoneNumber> 
      </phoneNumbers> 
      <gender> 
        <type>female</type> 
      </gender> 
    </person> 
    ```

1.  最后，我们必须关闭 XML 容器：

    ```py
    </persons>
    ```

1.  我们将使用 Python 分发的一部分库来处理 JSON 和 XML，即 `json` 和 `xml.etree.ElementTree`，如下所示：

    ```py
    import json
    import xml.etree.ElementTree as etree
    ```

1.  `JSONDataExtractor` 类解析 JSON 文件，并有一个 `parsed_data()` 方法，该方法返回所有数据作为一个字典（`dict`）。属性装饰器用于使 `parsed_data()` 看起来像一个普通属性而不是一个方法，如下所示：

    ```py
    class JSONDataExtractor:
      def __init__(self, filepath):
        self.data = dict()
        with open(filepath, mode='r', encoding='utf-8') as 
        f:self.data = json.load(f)
        @property
        def parsed_data(self):
            return self.data
    ```

1.  `XMLDataExtractor` 类解析 XML 文件，并有一个 `parsed_data()` 方法，该方法返回所有数据作为一个 `xml.etree.Element` 列表，如下所示：

    ```py
    class XMLDataExtractor:
      def __init__(self, filepath):
        self.tree =  etree.parse(filepath)
      @property
      def parsed_data(self):
      return self.tree
    ```

1.  `dataextraction_factory()` 函数是一个工厂方法。它根据输入文件路径的扩展名返回 `JSONDataExtractor` 或 `XMLDataExtractor` 的实例，如下所示：

    ```py
    def dataextraction_factory(filepath):
        if filepath.endswith('json'):
            extractor = JSONDataExtractor
        elif filepath.endswith('xml'):
            extractor = XMLDataExtractor
        else:
            raise ValueError('Cannot extract data from 
              {}'.format(filepath))
        return extractor(filepath)
    ```

1.  `extract_data_from()` 函数是 `dataextraction_factory()` 的包装器。它添加了异常处理，如下所示：

    ```py
    def extract_data_from(filepath):
        factory_obj = None
        try:
            factory_obj = dataextraction_factory(filepath)
        except ValueError as e:
            print(e)
        return factory_obj
    ```

1.  `main()` 函数演示了如何使用工厂方法设计模式。第一部分确保异常处理有效，如下所示：

    ```py
    def main():
        sqlite_factory = 
          extract_data_from('data/person.sq3')
        print()
    ```

1.  接下来部分展示了如何使用工厂方法处理 JSON 文件。基于解析，可以显示电影标题、年份、导演姓名和类型（当值为非空时），如下所示：

    ```py
    json_factory = extract_data_from('data/movies.json')
    json_data = json_factory.parsed_data
    print(f'Found: {len(json_data)} movies')
    for movie in json_data:
      print(f"Title: {movie['title']}")
      year = movie['year']
      if year:
      print(f"Year: {year}")
      director = movie['director']
      if director:
      print(f"Director: {director}")
      genre = movie['genre']
      if genre:
      print(f"Genre: {genre}")
      print()
    ```

1.  最后的部分展示了如何使用工厂方法来处理 XML 文件。XPath 用于找到所有姓氏为`Liar`的人元素（使用`liars = xml_data.findall(f".//person[lastName='Liar']")`）。对于每个匹配的人，他们的基本姓名和电话号码信息如下所示：

    ```py
    xml_factory = extract_data_from('data/person.xml')
    xml_data = xml_factory.parsed_data
    liars = 
      xml_data.findall(f".//person[lastName='Liar']")
    print(f'found: {len(liars)} persons')
    for liar in liars:
        firstname = liar.find('firstName').text
        print(f'first name: {firstname}')
        lastname = liar.find('lastName').text
        print(f'last name: {lastname}')
        [print(f"phone number ({p.attrib['type']}):", 
          p.text) 
        for p in liar.find('phoneNumbers')]
        print()
    ```

这里是实现总结（你可以在`factory_method.py`文件中找到代码）：

1.  我们首先导入所需的模块（`json`和`ElementTree`）。

1.  我们定义了 JSON 数据提取器类（`JSONDataExtractor`）。

1.  我们定义了 XML 数据提取器类（`XMLDataExtractor`）。

1.  我们添加了工厂函数`dataextraction_factory()`，以获取正确的数据提取器类。

1.  我们还添加了我们的异常处理包装器——`extract_data_from()`函数。

1.  最后，我们有`main()`函数，随后是 Python 从命令行调用此文件时的传统技巧。以下是`main`函数的方面：

    +   我们尝试从 SQL 文件（`data/person.sq3`）中提取数据以展示如何处理异常。

    +   我们从 JSON 文件中提取数据并解析结果。

    +   我们从 XML 文件中提取数据并解析结果。

以下是通过调用`python factory_method.py`命令获得的输出类型（对于不同的情况）：

首先，当你尝试访问 SQLite（`.sq3`）文件时，你会看到一个异常消息：

```py
Cannot extract data from data/person.sq3
```

然后，我们从处理`movies`文件（JSON）中得到了以下结果：

```py
Found: 9 movies
Title: After Dark in Central Park
Year: 1900
Title: Boarding School Girls' Pajama Parade
Year: 1900
Title: Buffalo Bill's Wild West Parad
Year: 1900
Title: Caught
Year: 1900
Title: Clowns Spinning Hats
Year: 1900
Title: Capture of Boer Battery by British
Year: 1900
Director: James H. White
Genre: Short documentary
Title: The Enchanted Drawing
Year: 1900
Director: J. Stuart Blackton
Title: Family Troubles
Year: 1900
Title: Feeding Sea Lions
Year: 1900
```

最后，我们从处理`person` XML 文件以找到姓氏为`Liar`的人的过程中得到了这个结果：

```py
found: 2 persons
first name: Jimy
last name: Liar
phone number (home): 212 555-1234
first name: Patty
last name: Liar
phone number (home): 212 555-1234
phone number (mobile): 001 452-8819
```

注意，尽管`JSONDataExtractor`和`XMLDataExtractor`具有相同的接口，但`parsed_data()`返回的内容处理并不统一。必须使用不同的 Python 代码来处理每个**数据提取器**。虽然能够使用相同的代码处理所有提取器听起来很理想，但在大多数情况下这是不现实的，除非我们使用某种类型的数据通用映射，这通常由外部数据提供商提供。一个有用的练习是假设你可以使用相同的代码来处理 XML 和 JSON 文件，并查看需要哪些更改来支持第三种格式，例如 SQLite。找到一个 SQLite 文件或创建自己的，并尝试它。

到目前为止，我们已经了解了工厂方法，它再次是工厂设计模式的第一种形式。在下一节中，我们将讨论第二种：抽象工厂设计模式。

# 应用抽象工厂

抽象工厂设计模式是工厂方法的泛化。抽象工厂是一组（逻辑）工厂方法，其中每个工厂方法负责生成不同类型的对象。

在本节中，我们将讨论一些示例、用例以及该模式的可能实现。

## 现实世界示例

抽象工厂在汽车制造中得到了应用。相同的机器用于不同汽车模型的零件（车门、面板、引擎盖、挡泥板和镜子）的冲压。由机器组装的模型是可配置的，并且可以随时轻松更改。

在软件类别中，`factory_boy` ([`github.com/FactoryBoy/factory_boy`](https://github.com/FactoryBoy/factory_boy)) 包为测试中创建 Django 模型提供了一个抽象工厂实现。它用于创建支持**特定于测试的属性**的模型实例。这很重要，因为这样，你的测试就会变得可读，并且你可以避免共享不必要的代码。

重要提示

Django 模型是特殊的类，框架使用这些类来帮助存储和与数据库（表）中的数据进行交互。有关更多详细信息，请参阅 Django 文档（[`docs.djangoproject.com`](https://docs.djangoproject.com)）。

## 用例

由于抽象工厂模式是工厂方法模式的泛化，它提供了相同的优势，它使得跟踪对象创建变得更容易，它将对象创建与对象使用解耦，并且它为我们提供了改进应用程序内存使用和性能的潜力。

但是，一个问题被提了出来：*我们如何知道何时使用工厂方法，而不是使用抽象工厂？* 答案是，我们通常从工厂方法开始，因为它更简单。如果我们发现我们的应用程序需要许多工厂方法，而这些方法合在一起创建一个对象家族是有意义的，那么我们最终会得到一个抽象工厂。

抽象工厂的一个好处通常在用户使用工厂方法时并不明显，那就是我们可以通过更改活动工厂方法来动态（在运行时）修改应用程序的行为。一个经典的例子是在应用程序使用期间，无需终止并重新启动应用程序，就可以更改应用程序的外观和感觉（例如，类似苹果、类似 Windows 等）。

## 实现抽象工厂模式

为了演示抽象工厂模式，我将重复使用我最喜欢的例子之一，这个例子包含在 Bruce Eckel 所著的《Python 3 Patterns, Recipes, and Idioms》一书中。想象一下，我们正在创建一个游戏，或者我们想在应用程序中包含一个迷你游戏来娱乐用户。我们希望包含至少两个游戏——一个针对儿童和一个针对成人。我们将根据用户输入在运行时决定创建和启动哪个游戏。抽象工厂负责游戏创建部分。

让我们从儿童游戏开始。它被称为*FrogWorld*。主要英雄是一只喜欢吃虫子的青蛙。每个英雄都需要一个好的名字，在我们的例子中，这个名字是在运行时由用户提供的。`interact_with()` 方法用于描述青蛙如何与障碍物（例如，虫子、谜题和其他青蛙）互动，如下所示：

```py
class Frog:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def interact_with(self, obstacle):
        act = obstacle.action()
        msg = f'{self} the Frog encounters {obstacle} 
          and {act}!'
        print(msg)
```

可能存在许多不同种类的障碍，但在我们的例子中，障碍只能是一个虫子。当青蛙遇到虫子时，只支持一个动作。它会吃掉它：

```py
class Bug:
    def __str__(self):
        return 'a bug'
    def action(self):
        return 'eats it'
```

`FrogWorld`类是一个抽象工厂。其主要职责是创建游戏中的主要角色和障碍物。将创建方法分开并使用通用的名称（例如，`make_character()`和`make_obstacle()`）允许我们动态地更改活动工厂（因此是活动游戏），而无需对代码进行任何更改。在静态类型语言中，抽象工厂将是一个抽象类/接口，具有空方法，但在 Python 中，这不是必需的，因为类型是在运行时检查的（[j.mp/ginstromdp](http://j.mp/ginstromdp)）。代码如下：

```py
class FrogWorld:
    def __init__(self, name):
        print(self)
        self.player_name = name
    def __str__(self):
        return '\n\n\t------ Frog World -------'
    def make_character(self):
        return Frog(self.player_name)
    def make_obstacle(self):
        return Bug()
```

*WizardWorld*游戏类似。唯一的区别是巫师与怪物（如兽人）战斗，而不是吃虫子！

下面是`Wizard`类的定义，它与`Frog`类类似：

```py
class Wizard:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def interact_with(self, obstacle):
        act = obstacle.action()
        msg = f'{self} the Wizard battles against 
          {obstacle} and {act}!'
        print(msg)
```

然后，下面是`Ork`类的定义：

```py
class Ork: 
    def __str__(self): 
        return 'an evil ork' 

    def action(self): 
        return 'kills it'
```

我们还需要定义与`FrogWorld`类似的`WizardWorld`类；在这种情况下，障碍是一个`Ork`实例：

```py
class WizardWorld: 
    def __init__(self, name): 
        print(self) 
        self.player_name = name 

    def __str__(self): 
        return '\n\n\t------ Wizard World -------' 

    def make_character(self): 
        return Wizard(self.player_name) 

    def make_obstacle(self): 
        return Ork()
```

`GameEnvironment`类是我们游戏的入口点。它接受工厂作为输入，并使用它来创建游戏的世界。`play()`方法启动创建的英雄与障碍之间的交互，如下所示：

```py
class GameEnvironment:
    def __init__(self, factory):
        self.hero = factory.make_character()
        self.obstacle = factory.make_obstacle()
    def play(self):
        self.hero.interact_with(self.obstacle)
```

`validate_age()`函数提示用户输入一个有效的年龄。如果年龄无效，它返回一个元组，第一个元素设置为`False`。如果年龄有效，元组的第一个元素设置为`True`。这就是我们关注元组的第二个元素的地方，即用户给出的年龄，如下所示：

```py
def validate_age(name):
    try:
        age = input(f'Welcome {name}. How old are you? ')
        age = int(age)
    except ValueError as err:
        print(f"Age {age} is invalid, please try again...")
        return (False, age)
    return (True, age)
```

最后但同样重要的是`main()`函数。它会询问用户的姓名和年龄，并根据用户的年龄决定应该玩哪个游戏，如下所示：

```py
def main():
    name = input("Hello. What's your name? ")
    valid_input = False
    while not valid_input:
        valid_input, age = validate_age(name)
    game = FrogWorld if age < 18 else WizardWorld
    environment = GameEnvironment(game(name))
    environment.play()
```

以下是我们刚刚讨论的实现总结（请参阅`abstract_factory.py`文件中的完整代码）：

1.  首先，我们为 FrogWorld 游戏定义`Frog`和`Bug`类。

1.  我们添加了`FrogWorld`类，其中我们使用了我们的`Frog`和`Bug`类。

1.  我们为 WizardWorld 游戏定义`Wizard`和`Ork`类。

1.  我们添加了`WizardWorld`类，其中我们使用了我们的`Wizard`和`Ork`类。

1.  我们定义`GameEnvironment`类。

1.  我们添加了`validate_age()`函数。

1.  最后，我们有`main()`函数，然后是调用它的传统技巧。以下是这个函数的各个方面：

    +   我们获取用户输入的姓名和年龄。

    +   我们根据用户的年龄决定使用哪个游戏类。

    +   我们实例化正确的游戏类，然后是`GameEnvironment`类。

    +   我们在环境对象上调用`play()`来玩游戏。

让我们使用`python abstract_factory.py`命令运行这个程序，并查看一些示例输出。

青少年的示例输出如下：

```py
Hello. What's your name? Billy
Welcome Billy. How old are you? 12
     ------ Frog World -------
Billy the Frog encounters a bug and eats it!
```

成年人的示例输出如下：

```py
Hello. What's your name? Charles
Welcome Charles. How old are you? 25
     ------ Wizard World -------
Charles the Wizard battles against an evil ork and kills 
it!
```

尝试扩展游戏使其更加完整。你可以做到你想做的程度；创建许多障碍、许多敌人以及你喜欢的任何其他东西。

# 摘要

在本章中，我们学习了如何使用工厂方法和抽象工厂设计模式。这两种模式在我们想要跟踪对象创建、将对象创建与对象使用解耦，甚至提高应用程序的性能和资源使用时都会用到。本章没有演示性能改进。你可以考虑将其作为一个好的练习尝试一下。

工厂方法设计模式作为一个不属于任何类的单一函数实现，负责创建单一类型的对象（如形状、连接点等）。我们看到了工厂方法与玩具建造的关系，提到了它如何被 Django 用于创建不同的表单字段，并讨论了它的其他可能用例。作为一个例子，我们实现了一个工厂方法，它提供了访问 XML 和 JSON 文件的功能。

抽象工厂设计模式通过几个属于单个类的工厂方法实现，用于创建一系列相关对象（如汽车的部件、游戏环境等）。我们提到了抽象工厂与汽车制造的关系，说明了 Django 的`django_factory`包如何利用它来创建干净的测试，然后我们讨论了它的常见用例。我们的抽象工厂实现示例是一个小型游戏，展示了我们如何在单个类中使用许多相关工厂。

在下一章中，我们将讨论建造者模式，这是另一种可以用于微调复杂对象创建的创建型模式。

# 问题

1.  使用工厂模式有哪些高级好处？

1.  工厂模式有两种形式，它们的主要区别是什么？

1.  在构建应用程序时，我们应该如何决定使用工厂模式的哪种形式？
