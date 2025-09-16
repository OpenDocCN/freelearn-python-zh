# 3

# 创建型设计模式

设计模式是可重用的编程解决方案，已在各种实际应用场景中使用，并已被证明能产生预期的结果。它们在程序员之间共享，并且随着时间的推移而不断改进。这个主题之所以受欢迎，多亏了 Erich Gamma、Richard Helm、Ralph Johnson 和 John Vlissides 合著的书籍，书名为《设计模式：可重用面向对象软件的元素》。

这里是来自“四人帮”书中关于设计模式的一句话：

*设计模式系统地命名、阐述并解释了一种通用的设计，该设计解决了面向对象系统中反复出现的设计问题。它描述了问题、解决方案、何时应用解决方案及其后果。它还提供了实现提示和示例。解决方案是一组通用的对象和类安排，用于解决问题。解决方案是根据特定环境进行定制和实现的，以解决该问题。*

根据它们解决的问题类型和/或它们帮助我们构建的解决方案类型，**面向对象编程**（**OOP**）中使用了几个设计模式类别。在他们的书中，“四人帮”提出了 23 个设计模式，分为三个类别：*创建型*、*结构型*和*行为型*。

`__init__()` 函数，不方便。

在本章中，我们将涵盖以下主要内容：

+   工厂模式

+   构建者模式

+   原型模式

+   单例模式

+   对象池模式

到本章结束时，你将有一个稳固的理解，无论是它们在 Python 中是否有用，以及如何在它们有用时使用它们。

# 技术要求

请参阅第一章中提出的需求。

# 工厂模式

我们将从“四人帮”书中的第一个创建型设计模式开始：工厂设计模式。在工厂设计模式中，客户端（即客户端代码）请求一个对象，而不知道这个对象是从哪里来的（即，使用哪个类来生成它）。工厂背后的想法是简化对象创建过程。如果通过一个中心函数来完成，那么跟踪哪些对象被创建比让客户端使用直接类实例化来创建对象要容易得多。工厂通过将创建对象的代码与使用它的代码解耦，减少了维护应用程序的复杂性。

工厂通常有两种形式——工厂方法，它是一个方法（对于 Python 开发者来说，也可以是一个简单的函数），根据输入参数返回不同的对象，以及抽象工厂，它是一组用于创建相关对象族的工厂方法。

让我们讨论两种工厂模式的形式，从工厂方法开始。

## 工厂方法

工厂方法基于一个编写来处理我们的对象创建任务的单一函数。我们执行它，传递一个参数，提供关于我们想要什么的信息，然后，作为结果，所需的对象被创建。

有趣的是，在使用工厂方法时，我们不需要了解任何关于结果对象是如何实现以及从哪里来的细节。

### 现实世界中的例子

我们可以在塑料玩具构建套件的背景下找到工厂方法模式在现实生活中的应用。用于构建塑料玩具的模具材料是相同的，但通过使用正确的塑料模具，可以生产出不同的玩具（不同的形状或图案）。这就像有一个工厂方法，其中输入是我们想要的玩具名称（例如，鸭子或汽车），输出（成型后）是我们请求的塑料玩具。

在软件世界中，Django Web 框架使用工厂方法模式来创建网页表单的字段。Django 包含的 `forms` 模块（[`github.com/django/django/blob/main/django/forms/forms.py`](https://github.com/django/django/blob/main/django/forms/forms.py)）支持创建不同类型的字段（例如，`CharField`、`EmailField` 等）。它们的行为的一部分可以通过 `max_length` 和 `required` 等属性进行自定义。

### 工厂方法模式的用例

如果你意识到你无法追踪应用程序创建的对象，因为创建它们的代码分布在许多不同的地方而不是一个单独的函数/方法中，你应该考虑使用工厂方法模式。工厂方法集中了对象的创建，跟踪对象变得容易得多。请注意，创建多个工厂方法是完全可以接受的，这也是实践中通常的做法。每个工厂方法逻辑上分组创建具有相似性的对象。例如，一个工厂方法可能负责连接到不同的数据库（MySQL、SQLite）；另一个工厂方法可能负责创建你请求的几何对象（圆形、三角形）；等等。

当你想要将对象创建与对象使用解耦时，工厂方法也非常有用。在创建对象时，我们并不依赖于特定的类；我们只是通过调用一个函数来提供关于我们想要什么的部分信息。这意味着修改函数是容易的，并且不需要对其使用的代码进行任何更改。

另一个值得提及的使用案例与提高应用程序的性能和内存使用有关。工厂方法可以通过仅在必要时创建新对象来提高性能和内存使用。当我们使用直接类实例化创建对象时，每次创建新对象时都会分配额外的内存（除非类内部使用缓存，这通常不是情况）。我们可以在以下代码（`ch03/factory/id.py`）中看到这一点，该代码创建了`MyClass`类的两个实例，并使用`id()`函数比较它们的内存地址。地址也打印在输出中，以便我们可以检查它们。内存地址不同的事实意味着创建了两个不同的对象。代码如下：

```py
class MyClass:
    pass
if __name__ == "__main__":
    a = MyClass()
    b = MyClass()
    print(id(a) == id(b))
    print(id(a))
    print(id(b))
```

在我的计算机上执行代码（`ch03/factory/id.py`）的结果如下：

```py
False
4330224656
4331646704
```

注意

当你执行文件时看到的地址，其中调用了`id()`函数，与我看到的地址不同，因为它们依赖于当前的内存布局和分配。但结果必须相同——这两个地址应该是不同的。有一个例外，如果你在 Python **读-求值-打印循环**（**REPL**）中编写和执行代码——或者简单地说，交互式提示符——那么那是一个 REPL 特定的优化，通常不会发生。

### 实现工厂方法模式

数据以多种形式存在。存储/检索数据主要有两种文件类别：可读文件和二进制文件。可读文件的例子有 XML、RSS/Atom、YAML 和 JSON。二进制文件的例子有 SQLite 使用的`.sq3`文件格式和用于听音乐的`.mp3`音频文件格式。

在这个例子中，我们将关注两种流行的可读格式——XML 和 JSON。尽管可读文件通常比二进制文件解析速度慢，但它们使数据交换、检查和修改变得容易得多。因此，建议你在没有其他限制不允许的情况下（主要是不可接受的性能或专有二进制格式）与可读文件一起工作。

在这种情况下，我们有一些输入数据存储在 XML 和 JSON 文件中，我们想要解析它们并检索一些信息。同时，我们想要集中管理客户端对这些（以及所有未来的）外部服务的连接。我们将使用工厂方法来解决这个问题。示例仅关注 XML 和 JSON，但添加对更多服务的支持应该是简单的。

首先，让我们看看数据文件。

JSON 文件，`movies.json`，是一个包含关于美国电影（标题、年份、导演姓名、类型等）信息的示例数据集：

```py
[
  {
    "title": "After Dark in Central Park",
    "year": 1900,
    "director": null,
    "cast": null,
    "genre": null
  },
  {
    "title": "Boarding School Girls' Pajama Parade",
    "year": 1900,
    "director": null,
    "cast": null,
    "genre": null
  },
  {
    "title": "Buffalo Bill's Wild West Parad",
    "year": 1900,
    "director": null,
    "cast": null,
    "genre": null
  },
  {
    "title": "Caught",
    "year": 1900,
    "director": null,
    "cast": null,
    "genre": null
  },
  {
    "title": "Clowns Spinning Hats",
    "year": 1900,
    "director": null,
    "cast": null,
    "genre": null
  },
  {
    "title": "Capture of Boer Battery by British",
    "year": 1900,
    "director": "James H. White",
    "cast": null,
    "genre": "Short documentary"
  },
  {
    "title": "The Enchanted Drawing",
    "year": 1900,
    "director": "J. Stuart Blackton",
    "cast": null,
    "genre": null
  },
  {
    "title": "Family Troubles",
    "year": 1900,
    "director": null,
    "cast": null,
    "genre": null
  },
  {
    "title": "Feeding Sea Lions",
    "year": 1900,
    "director": null,
    "cast": "Paul Boyton",
    "genre": null
  }
]
```

XML 文件，`person.xml`，包含有关个人（`firstName`、`lastName`、`gender`等）的信息，如下所示：

1.  我们从`persons` XML 容器的封装标签开始：

    ```py
    <persons>
    ```

1.  然后，一个表示个人数据代码的 XML 元素如下所示：

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
        <number type="home">212 555-1234</number>
        <number type="fax">646 555-4567</number>
      </phoneNumbers>
      <gender>
        <type>male</type>
      </gender>
    </person>
    ```

1.  一个表示另一个人数据的 XML 元素如下所示：

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
        <number type="home">212 555-1234</number>
      </phoneNumbers>
      <gender>
        <type>male</type>
      </gender>
    </person>
    ```

1.  一个表示第三个人数据的 XML 元素如下所示：

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
        <number type="home">212 555-1234</number>
        <number type="mobile">001 452-8819</number>
      </phoneNumbers>
      <gender>
        <type>female</type>
      </gender>
    </person>
    ```

1.  最后，我们关闭 XML 容器：

    ```py
    </persons>
    ```

我们将使用 Python 发行版中用于处理 JSON 和 XML 的两个库：`json`和`xml.etree.ElementTree`。

我们首先导入所需的模块以进行各种操作（`json`、`ElementTree`和`pathlib`），并定义一个`JSONDataExtractor`类，从文件中加载数据，并使用`parsed_data`属性来获取它。这部分代码如下：

```py
import json
import xml.etree.ElementTree as ET
from pathlib import Path
class JSONDataExtractor:
    def __init__(self, filepath: Path):
        self.data = {}
        with open(filepath) as f:
            self.data = json.load(f)
    @property
    def parsed_data(self):
        return self.data
```

我们还定义了一个`XMLDataExtractor`类，通过`ElementTree`的解析器加载文件中的数据，并使用`parsed_data`属性来获取结果，如下所示：

```py
class XMLDataExtractor:
    def __init__(self, filepath: Path):
        self.tree = ET.parse(filepath)
    @property
    def parsed_data(self):
        return self.tree
```

现在，我们提供工厂函数，该函数根据目标文件的扩展名选择正确的数据提取器类（如果不受支持，则抛出异常），如下所示：

```py
def extract_factory(filepath: Path):
    ext = filepath.name.split(".")[-1]
    if ext == "json":
        return JSONDataExtractor(filepath)
    elif ext == "xml":
        return XMLDataExtractor(filepath)
    else:
        raise ValueError("Cannot extract data")
```

接下来，我们定义我们程序的主要函数，`extract()`；在函数的第一部分，代码处理 JSON 情况，如下所示：

```py
def extract(case: str):
    dir_path = Path(__file__).parent
    if case == "json":
        path = dir_path / Path("movies.json")
        factory = extract_factory(path)
        data = factory.parsed_data
        for movie in data:
            print(f"- {movie['title']}")
            director = movie["director"]
            if director:
                print(f"   Director: {director}")
            genre = movie["genre"]
            if genre:
                print(f"   Genre: {genre}")
```

我们添加了`extract()`函数的最后部分，使用工厂方法处理 XML 文件。使用 XPath 查找所有姓氏为`Liar`的个人元素。对于每个匹配的个人，显示基本姓名和电话号码信息。代码如下：

```py
    elif case == "xml":
        path = dir_path / Path("person.xml")
        factory = extract_factory(path)
        data = factory.parsed_data
        search_xpath = ".//person[lastName='Liar']"
        items = data.findall(search_xpath)
        for item in items:
            first = item.find("firstName").text
            last = item.find("lastName").text
            print(f"- {first} {last}")
            for pn in item.find("phoneNumbers"):
                pn_type = pn.attrib["type"]
                pn_val = pn.text
                phone = f"{pn_type}: {pn_val}"
                print(f"   {phone}")
```

最后，我们添加一些测试代码：

```py
if __name__ == "__main__":
    print("* JSON case *")
    extract(case="json")
    print("* XML case *")
    extract(case="xml")
```

这里是实现总结（在`ch03/factory/factory_method.py`文件中）：

1.  在导入所需的模块后，我们首先定义一个 JSON 数据提取类（`JSONDataExtractor`）和一个 XML 数据提取类（`XMLDataExtractor`）。

1.  我们添加了一个工厂函数，`extract_factory()`，以获取正确的数据提取器类进行实例化。

1.  我们还添加了我们的包装器和主函数，`extract()`。

1.  最后，我们添加测试代码，从 JSON 文件和 XML 文件中提取数据并解析结果文本。

要测试示例，请运行以下命令：

```py
python ch03/factory/factory_method.py
```

你应该得到以下输出：

```py
* JSON case *
- After Dark in Central Park
- Boarding School Girls' Pajama Parade
- Buffalo Bill's Wild West Parad
- Caught
- Clowns Spinning Hats
- Capture of Boer Battery by British
   Director: James H. White
   Genre: Short documentary
- The Enchanted Drawing
   Director: J. Stuart Blackton
- Family Troubles
- Feeding Sea Lions
* XML case *
- Jimy Liar
   home: 212 555-1234
- Patty Liar
   home: 212 555-1234
   mobile: 001 452-8819
```

注意，尽管`JSONDataExtractor`和`XMLDataExtractor`具有相同的接口，但`parsed_data()`返回的内容处理方式并不统一；在一种情况下我们有一个列表，在另一种情况下我们有一个树。必须使用不同的 Python 代码来处理每个数据提取器。虽然能够为所有提取器使用相同的代码会很理想，但在大多数情况下这是不现实的，除非我们使用某种类型的数据通用映射，这通常由外部数据提供者提供。假设你可以使用相同的代码来处理 XML 和 JSON 文件，那么为了支持第三种格式——例如 SQLite，需要进行哪些更改？找到一个 SQLite 文件或创建自己的文件并尝试它。

### 你应该使用工厂方法模式吗？

经验丰富的 Python 开发者经常对工厂方法模式提出的主要批评是，它对于许多用例来说可能被认为是过度设计或过于复杂。Python 的动态类型和一等函数通常允许对工厂方法旨在解决的问题有更简单、更直接的方法。在 Python 中，你通常可以直接使用简单函数或类方法来创建对象，而无需创建单独的工厂类或函数。这使代码更具可读性和 Python 风格，遵循语言“简单比复杂好”的哲学。

此外，Python 对默认参数、关键字参数和其他语言特性的支持通常使得向后兼容地扩展构造函数变得更加容易，从而减少了单独工厂方法的需求。因此，虽然工厂方法模式是在静态类型语言（如 Java 或 C++）中建立起来的一个良好的设计模式，但它通常被认为对于 Python 更灵活和动态的本质来说过于繁琐或冗长。

为了展示在没有工厂方法模式的情况下如何处理简单用例，已在 `ch03/factory/factory_method_not_needed.py` 文件中提供了一个替代实现。正如你所看到的，不再有工厂。以下代码摘录显示了当我们说在 Python 中，你只需在需要的地方创建对象，而不需要一个中间函数或类，这使得你的代码更具 Python 风格的含义：

```py
if case == "json":
    path = dir_path / Path("movies.json")
    data = JSONDataExtractor(path).parsed_data
```

## 抽象工厂模式

抽象工厂模式是工厂方法思想的泛化。基本上，抽象工厂是一组（逻辑）工厂方法，其中每个工厂方法负责生成不同类型的对象。

我们将讨论一些示例、用例和可能的实现。

### 现实世界示例

抽象工厂在汽车制造中得到了应用。相同的机器用于不同车型（车门、面板、引擎盖、挡泥板和镜子）的部件冲压。由机器组装的模型是可配置的，并且可以随时更改。

在软件类别中，`factory_boy` 包([`github.com/FactoryBoy/factory_boy`](https://github.com/FactoryBoy/factory_boy))为测试中创建 Django 模型提供了一个抽象工厂实现。另一个工具是 `model_bakery` ([`github.com/model-bakers/model_bakery`](https://github.com/model-bakers/model_bakery))。这两个包都用于创建支持特定测试属性的模式实例。这很重要，因为这样，可以提高测试的可读性，并避免共享不必要的代码。

注意

Django 模型是框架用来帮助存储和与数据库（表）中的数据交互的特殊类。有关更多详细信息，请参阅 Django 文档([`docs.djangoproject.com`](https://docs.djangoproject.com))。

### 抽象工厂模式的使用案例

由于抽象工厂模式是工厂方法模式的泛化，它提供了相同的优点：它使跟踪对象创建变得更容易，它将对象创建与对象使用解耦，并且它为我们提供了改进应用程序内存使用和性能的潜力。

### 实现抽象工厂模式

为了演示抽象工厂模式，我将重用我最喜欢的例子之一，它包含在 Bruce Eckel 所著的《Python 3 Patterns, Recipes and Idioms》一书中。想象一下，我们正在创建一个游戏，或者我们想在应用程序中包含一个迷你游戏来娱乐用户。我们希望包含至少两个游戏，一个供儿童玩，一个供成人玩。我们将根据用户输入在运行时决定创建和启动哪个游戏。抽象工厂负责游戏创建部分。

让我们从儿童游戏开始。它被称为`interact_with()`方法，用于描述青蛙与障碍（例如，虫子、谜题和其他青蛙）的交互，如下所示：

```py
class Frog:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def interact_with(self, obstacle):
        act = obstacle.action()
        msg = f"{self} the Frog encounters {obstacle} and {act}!"
        print(msg)
```

可能会有许多种障碍，但就我们的例子而言，障碍只能是一个虫子。当青蛙遇到虫子时，只支持一个动作。它会吃掉它：

```py
class Bug:
    def __str__(self):
        return "a bug"
    def action(self):
        return "eats it"
```

`FrogWorld`类是一个抽象工厂。其主要职责是创建游戏中的主要角色和障碍。将创建方法分开并使用通用的名称（例如，`make_character()`和`make_obstacle()`）允许我们动态地更改活动工厂（因此，活动游戏）而无需任何代码更改。代码如下：

```py
class FrogWorld:
    def __init__(self, name):
        print(self)
        self.player_name = name
    def __str__(self):
        return "\n\n\t------ Frog World -------"
    def make_character(self):
        return Frog(self.player_name)
    def make_obstacle(self):
        return Bug()
```

**WizardWorld**游戏类似。唯一的区别是法师与食虫虫的怪物如兽人战斗，而不是吃虫子！

这里是`Wizard`类的定义，它与`Frog`类类似：

```py
class Wizard:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def interact_with(self, obstacle):
        act = obstacle.action()
        msg = f"{self} the Wizard battles against {obstacle} and {act}!"
        print(msg)
```

然后，`Ork`类的定义如下：

```py
class Ork:
    def __str__(self):
        return "an evil ork"
    def action(self):
        return "kills it"
```

我们还需要定义一个`WizardWorld`类，类似于我们讨论过的`FrogWorld`类；在这种情况下，障碍是一个`Ork`实例：

```py
class WizardWorld:
    def __init__(self, name):
        print(self)
        self.player_name = name
    def __str__(self):
        return "\n\n\t------ Wizard World -------"
    def make_character(self):
        return Wizard(self.player_name)
    def make_obstacle(self):
        return Ork()
```

`GameEnvironment`类是我们游戏的主要入口点。它接受工厂作为输入，并使用它来创建游戏的世界。`play()`方法启动创建的英雄与障碍之间的交互，如下所示：

```py
class GameEnvironment:
    def __init__(self, factory):
        self.hero = factory.make_character()
        self.obstacle = factory.make_obstacle()
    def play(self):
        self.hero.interact_with(self.obstacle)
```

`validate_age()`函数提示用户输入有效的年龄。如果年龄无效，它返回一个元组，第一个元素设置为`False`。如果年龄有效，元组的第一个元素设置为`True`，这就是我们关注元组的第二个元素的情况，即用户输入的年龄，如下所示：

```py
def validate_age(name):
    age = None
    try:
        age_input = input(
            f"Welcome {name}. How old are you? "
        )
        age = int(age_input)
    except ValueError:
        print(
            f"Age {age} is invalid, please try again..."
        )
        return False, age
    return True, age
```

最后是`main()`函数的定义，然后调用它。它询问用户的姓名和年龄，并根据用户的年龄决定应该玩哪个游戏，如下所示：

```py
def main():
    name = input("Hello. What's your name? ")
    valid_input = False
    while not valid_input:
        valid_input, age = validate_age(name)
    game = FrogWorld if age < 18 else WizardWorld
    environment = GameEnvironment(game(name))
    environment.play()
if __name__ == "__main__":
    main()
```

我们刚才讨论的实现总结（请参阅`ch03/factory/abstract_factory.py`文件中的完整代码）如下：

1.  我们为**FrogWorld**游戏定义了`Frog`和`Bug`类。

1.  我们添加了一个`FrogWorld`类，其中我们使用了我们的`Frog`和`Bug`类。

1.  我们为**WizardWorld**游戏定义了`Wizard`和`Ork`类。

1.  我们添加了一个`WizardWorld`类，其中我们使用了我们的`Wizard`和`Ork`类。

1.  我们定义了一个`GameEnvironment`类。

1.  我们添加了一个`validate_age()`函数。

1.  最后，我们有`main()`函数，接着是调用它的传统技巧。以下是这个函数的几个方面：

    +   我们获取用户的姓名和年龄输入。

    +   我们根据用户的年龄决定使用哪个游戏类。

    +   我们实例化正确的游戏类，然后是`GameEnvironment`类。

    +   我们在`environment`对象上调用`.play()`来玩游戏。

让我们使用`python ch03/factory/abstract_factory.py`命令调用这个程序，并查看一些示例输出。

青少年的示例输出如下：

```py
Hello. What's your name? Arthur
Welcome Arthur. How old are you? 13
------ Frog World -------
Arthur the Frog encounters a bug and eats it!
```

成人的示例输出如下：

```py
Hello. What's your name? Tom
Welcome Tom. How old are you? 34
------ Wizard World -------
Tom the Wizard battles against an evil ork and kills it!
```

尝试扩展游戏使其更加完整。你可以做到你想做的程度；创建许多障碍、许多敌人，以及你喜欢的任何其他东西。

# 构建器模式

我们刚刚介绍了前两种创建型模式，即工厂方法和抽象工厂方法，它们都提供了在非平凡情况下改进我们创建对象的方法。

现在，假设我们想要创建一个由多个部分组成的对象，并且组合需要逐步完成。除非所有部分都完全创建，否则对象不是完整的。这就是构建器设计模式能帮助我们的地方。构建器设计模式将复杂对象的构建与其表示分离。通过将构建与表示分离，相同的构建可以用来创建几个不同的表示。

## 现实世界示例

在我们的日常生活中，构建器设计模式在快餐店中被使用。制作汉堡和包装（盒子和平装袋）的相同程序总是被使用，即使有各种各样的汉堡（经典汉堡、芝士汉堡等等）和不同的包装（小号盒子、中号盒子等等）。经典汉堡和芝士汉堡之间的区别在于表示，而不是构建过程。在这种情况下，导演是收银员，他向工作人员下达需要准备的指令，而构建者是负责特定订单的工作人员。

在[软件](https://github.com/ambitioninc/django-query-builder)中，我们可以考虑`django-query-builder`库([`github.com/ambitioninc/django-query-builder`](https://github.com/ambitioninc/django-query-builder))，这是一个依赖构建器模式的第三方 Django 库。这个库可以用来动态构建 SQL 查询，允许你控制查询的所有方面，并创建从简单到非常复杂的各种查询。

## 与工厂模式的比较

到这一点，构建器模式和工厂模式之间的区别可能不是很清楚。主要区别是，工厂模式在单步中创建对象，而构建器模式在多步中创建对象，并且几乎总是使用一个*导演*。

另一个区别是，虽然工厂模式立即返回创建的对象，但在构建器模式中，客户端代码明确要求导演在需要时返回最终对象。

## 构建器模式的用例

当一个对象需要用许多可能的配置构建时，构建器模式特别有用。一个典型的情况是，一个类有多个构造函数，参数数量不同，这往往会导致混淆或容易出错的代码。

当对象的构建过程比简单地设置初始值更复杂时，该模式也有益。例如，如果一个对象的完整创建涉及多个步骤，如参数验证、设置数据结构或甚至调用外部服务，构建器模式可以封装这种复杂性。

## 实现构建器模式

让我们看看我们如何使用构建器设计模式来制作一个点餐应用程序。这个例子特别有趣，因为披萨的制备需要遵循特定的顺序。要加酱料，你首先需要准备面团。要加配料，你首先需要加酱料。除非酱料和配料都放在面团上，否则你不能开始烤披萨。此外，每块披萨通常需要不同的烘烤时间，这取决于面团的厚度和使用的配料。

我们首先导入所需的模块，并声明一些`Enum`参数以及一个在应用程序中多次使用的常量。`STEP_DELAY`常量用于在准备披萨的不同步骤之间添加时间延迟，如下所示：

```py
import time
from enum import Enum
PizzaProgress = Enum(
    "PizzaProgress", "queued preparation baking ready"
)
PizzaDough = Enum("PizzaDough", "thin thick")
PizzaSauce = Enum("PizzaSauce", "tomato creme_fraiche")
PizzaTopping = Enum(
    "PizzaTopping",
    "mozzarella double_mozzarella bacon ham mushrooms red_onion oregano",
)
# Delay in seconds
STEP_DELAY = 3
```

我们最终的产品是披萨，由`Pizza`类描述。当使用构建器模式时，最终产品没有很多责任，因为它不应该直接实例化。构建器创建最终产品的实例，并确保它被正确准备。这就是为什么`Pizza`类如此极简。它基本上将所有数据初始化为合理的默认值。一个例外是`prepare_dough()`方法。

`prepare_dough()`方法定义在`Pizza`类中而不是构建器中，有两个原因。首先，为了阐明最终产品通常是极简的，这并不意味着你永远不应该给它分配任何责任。其次，为了通过组合来促进代码重用。

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
        print(
            f"preparing the {self.dough.name} dough of your {self}..."
        )
        time.sleep(STEP_DELAY)
        print(f"done with the {self.dough.name} dough")
```

有两个构建器：一个用于创建玛格丽塔披萨（`MargaritaBuilder`）和另一个用于创建奶油培根披萨（`CreamyBaconBuilder`）。每个构建器创建一个 `Pizza` 实例，并包含遵循披萨制作程序的各个方法：`prepare_dough()`、`add_sauce()`、`add_topping()` 和 `bake()`。更准确地说，`prepare_dough()` 只是 `Pizza` 类中 `prepare_dough()` 方法的包装。

注意每个构建器如何处理所有与披萨相关的细节。例如，玛格丽塔披萨的配料是双份马苏里拉奶酪和牛至，而奶油培根披萨的配料是马苏里拉奶酪、培根、火腿、蘑菇、红洋葱和牛至。

`MargaritaBuilder` 类的代码片段如下（完整的代码请参阅 `ch03/builder.py` 文件）：

```py
class MargaritaBuilder:
    def __init__(self):
        self.pizza = Pizza("margarita")
        self.progress = PizzaProgress.queued
        self.baking_time = 5
    def prepare_dough(self):
        self.progress = PizzaProgress.preparation
        self.pizza.prepare_dough(PizzaDough.thin)
    ...
```

`CreamyBaconBuilder` 类的代码片段如下：

```py
class CreamyBaconBuilder:
    def __init__(self):
        self.pizza = Pizza("creamy bacon")
        self.progress = PizzaProgress.queued
        self.baking_time = 7
    def prepare_dough(self):
        self.progress = PizzaProgress.preparation
        self.pizza.prepare_dough(PizzaDough.thick)
    ...
```

在此示例中，*导演* 是服务员。`Waiter` 类的核心是 `construct_pizza()` 方法，它接受一个构建器作为参数并按正确顺序执行所有披萨准备步骤。选择合适的构建器，甚至可以在运行时完成，这使我们能够创建不同的披萨风格，而无需修改导演（`Waiter`）的任何代码。`Waiter` 类还包含 `pizza()` 方法，该方法将最终产品（准备好的披萨）作为变量返回给调用者。该类的代码如下：

```py
class Waiter:
    def __init__(self):
        self.builder = None
    def construct_pizza(self, builder):
        self.builder = builder
        steps = (
            builder.prepare_dough,
            builder.add_sauce,
            builder.add_topping,
            builder.bake,
        )
        [step() for step in steps]
    @property
    def pizza(self):
        return self.builder.pizza
```

`validate_style()` 方法与本章前面标题为 *工厂模式* 的部分中描述的 `validate_age()` 函数类似。它用于确保用户输入有效，在这种情况下是一个映射到披萨构建器的字符。`m` 字符使用 `MargaritaBuilder` 类，而 `c` 字符使用 `CreamyBaconBuilder` 类。这些映射在 `builder` 参数中。返回一个元组，第一个元素设置为 `True` 如果输入有效或 `False` 如果无效，如下所示：

```py
def validate_style(builders):
    try:
        input_msg = "What pizza would you like, [m]argarita or [c]reamy bacon? "
        pizza_style = input(input_msg)
        builder = builders[pizza_style]()
        valid_input = True
    except KeyError:
        error_msg = "Sorry, only margarita (key m) and creamy bacon (key c) are available"
        print(error_msg)
        return (False, None)
    return (True, builder)
```

最后的部分是 `main()` 函数。`main()` 函数包含实例化披萨构建器的代码。然后，`Waiter` 导演使用披萨构建器准备披萨。创建的披萨可以在任何后续时间点交付给客户：

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
    print(f"Enjoy your {pizza}!")
```

这里是实现总结（在 `ch03/builder.py` 文件中）：

1.  我们开始于需要的一些导入，对于标准的 `Enum` 类和 `time` 模块。

1.  我们声明了一些常量的变量：`PizzaProgress`、`PizzaDough`、`PizzaSauce`、`PizzaTopping` 和 `STEP_DELAY`。

1.  我们定义了我们的 `Pizza` 类。

1.  我们为两个构建器定义了类，`MargaritaBuilder` 和 `CreamyBaconBuilder`。

1.  我们定义了我们的 `Waiter` 类。

1.  我们添加了一个 `validate_style()` 函数来改进异常处理。

1.  最后，我们有 `main()` 函数，随后是程序运行时调用它的代码片段。在 `main()` 函数中，以下操作发生：

    +   我们通过`validate_style()`函数进行验证后，使它能够根据用户的输入选择披萨构建器。

    +   服务员使用披萨构建器来准备披萨。

    +   然后将制作的披萨送出。

下面是调用`python ch03/builder.py`命令执行此示例程序产生的输出：

```py
What pizza would you like, [m]argarita or [c]reamy bacon? c
preparing the thick dough of your creamy bacon...
done with the thick dough
adding the crème fraîche sauce to your creamy bacon
done with the crème fraîche sauce
adding the topping (mozzarella, bacon, ham, mushrooms, red onion, oregano) to your creamy bacon
done with the topping (mozzarella, bacon, ham, mushrooms, red onion, oregano)
baking your creamy bacon for 7 seconds
your creamy bacon is ready
Enjoy your creamy bacon!
```

这是一个很好的结果。

但是...只支持两种披萨类型是件遗憾的事。想要一个夏威夷披萨构建器吗？在考虑了优势和劣势之后，考虑使用*继承*。或者*组合*，正如我们在*第一章*中看到的，它有其优势。

# 原型模式

原型模式允许您通过复制现有对象来创建新对象，而不是从头开始创建。当初始化对象的成本比复制现有对象更昂贵或更复杂时，此模式特别有用。本质上，原型模式通过复制现有实例来创建类的新的实例，从而避免了初始化新对象的额外开销。

在其最简单的版本中，这个模式只是一个接受对象作为输入参数并返回其副本的`clone()`函数。在 Python 中，可以使用`copy.deepcopy()`函数来实现这一点。

## 真实世界的例子

通过剪枝繁殖植物是原型模式的一个真实世界的例子。使用这种方法，你不是从种子中生长植物；而是创建一个新的植物，它是现有植物的副本。

许多 Python 应用程序都使用原型模式，但很少将其称为*原型*，因为克隆对象是 Python 语言的一个内置功能。

## 原型模式的用例

当我们有一个需要保持不变且我们想要创建其精确副本的现有对象时，原型模式非常有用，允许在副本的某些部分进行更改。

除了从数据库中复制并具有对其他基于数据库的对象引用的对象外，还需要频繁地复制对象。克隆这样一个复杂的对象成本很高（对数据库进行多次查询），因此原型是一个方便解决问题的方法。

## 实现原型模式

现在，一些组织，甚至规模较小的组织，通过其基础设施/DevOps 团队、托管提供商或**云服务提供商**（**CSPs**）处理许多网站和应用。

当你必须管理多个网站时，有一个点变得难以跟踪。你需要快速访问信息，比如涉及的 IP 地址、域名及其到期日期，以及 DNS 参数的详细信息。因此，你需要一种库存工具。

让我们想象一下这些团队如何处理这种类型的数据以进行日常活动，并简要讨论实现一个帮助整合和维护数据的软件（除了 Excel 表格之外）。

首先，我们需要导入 Python 的标准`copy`模块，如下所示：

```py
import copy
```

在这个系统的核心，我们将有一个`Website`类来存储所有有用的信息，例如名称、域名、描述、我们管理的网站的作者等。

在类的`__init__()`方法中，只有一些参数是固定的：`name`、`domain`和`description`。但我们还希望有灵活性，客户端代码可以使用`kwargs`变量长度集合（每个对成为`kwargs` Python 字典的一项）以`name=value`的形式传递更多参数。

其他信息

Python 有一个帮助在`obj`对象上设置任意属性名为`attr`、值为`val`的惯用语，使用内置的`setattr()`函数：`setattr(obj, attr, val)`。

因此，我们定义了一个`Website`类并初始化其对象，使用`setattr`技术为可选属性，如下所示：

```py
class Website:
    def __init__(
        self,
        name: str,
        domain: str,
        description: str,
        **kwargs,
    ):
        self.name = name
        self.domain = domain
        self.description = description
        for key in kwargs:
            setattr(self, key, kwargs[key])
```

这还不算完。为了提高类的可用性，我们还添加了其字符串表示方法（`__str__()`）。我们使用`vars()`技巧提取所有实例属性的值，并将这些值注入方法返回的字符串中。此外，由于我们计划克隆对象，我们还使用`id()`函数包含对象的内存地址。代码如下：

```py
def __str__(self) -> str:
    summary = [
        f"- {self.name} (ID: {id(self)})\n",
    ]
    infos = vars(self).items()
    ordered_infos = sorted(infos)
    for attr, val in ordered_infos:
        if attr == "name":
            continue
        summary.append(f"{attr}: {val}\n")
    return "".join(summary)
```

其他信息

Python 中的`vars()`函数返回对象的`__dict__`属性。`__dict__`属性是一个包含对象属性（数据属性和方法）的字典。这个函数对于调试很有用，因为它允许你检查对象或函数内的局部变量的属性和方法。但请注意，并非所有对象都有`__dict__`属性。例如，列表和字典等内置类型没有这个属性。

接下来，我们添加一个实现原型设计模式的`Prototype`类。在这个类的核心，我们有`clone()`方法，它负责使用`copy.deepcopy()`函数克隆对象。

注意

当我们使用`copy.deepcopy()`克隆对象时，克隆对象的内存地址必须与原始对象的内存地址不同。

由于克隆意味着我们允许为可选属性设置值，请注意我们在这里如何使用`setattr`技术与`attrs`字典。此外，为了方便起见，`Prototype`类包含`register()`和`unregister()`方法，这些方法可以用来跟踪注册表（字典）中的克隆对象。该类的代码如下：

```py
class Prototype:
    def __init__(self):
        self.registry = {}
    def register(self, identifier: int, obj: object):
        self.registry[identifier] = obj
    def unregister(self, identifier: int):
        del self.registry[identifier]
    def clone(self, identifier: int, **attrs) -> object:
        found = self.registry.get(identifier)
        if not found:
            raise ValueError(
              f"Incorrect object identifier: {identifier}"
            )
        obj = copy.deepcopy(found)
        for key in attrs:
            setattr(obj, key, attrs[key])
        return obj
```

在我们接下来定义的`main()`函数中，我们完成程序：我们克隆一个`Website`实例，命名为`site1`，以获取第二个对象`site2`。基本上，我们实例化`Prototype`类，并使用其`.clone()`方法。然后，我们显示结果。该函数的代码如下：

```py
def main():
    keywords = (
        "python",
        "programming",
        "scripting",
        "data",
        "automation",
    )
    site1 = Website(
        "Python",
        domain="python.org",
        description="Programming language and ecosystem",
        category="Open Source Software",
        keywords=keywords,
    )
    proto = Prototype()
    proto.register("python-001", site1)
    site2 = proto.clone(
        "python-001",
        name="Python Package Index",
        domain="pypi.org",
        description="Repository for published packages",
        category="Open Source Software",
    )
    for site in (site1, site2):
        print(site)
```

最后，我们调用`main()`函数，如下所示：

```py
if __name__ == "__main__":
    main()
```

这里是对我们在代码中执行的操作的总结（`ch03/prototype.py`）：

1.  我们首先导入`copy`模块。

1.  我们定义了一个`Website`类，它具有初始化方法（`__init__()`）和字符串表示方法（`__str__()`）。

1.  我们定义了前面展示的`Prototype`类。

1.  然后，我们有`main()`函数，其中我们执行以下操作：

    +   我们定义了我们需要的`keywords`列表。

    +   我们创建`Website`类的实例，称为`site1`（这里我们使用`keywords`列表）。

    +   我们创建一个`Prototype`对象，并使用其`register()`方法将`site1`及其标识符注册（这有助于我们跟踪字典中的克隆对象）。

    +   我们克隆`site1`对象以获得`site2`。

    +   我们显示结果（两个`Website`对象）。

当我在电脑上执行`python ch03/prototype.py`命令时的一个示例输出如下：

```py
- Python (ID: 4369628560)
category: Open Source Software
description: Programming language and ecosystem
domain: python.org
keywords: ('python', 'programming', 'scripting', 'data', 'automation')
- Python Package Index (ID: 4369627552)
category: Open Source Software
description: Repository site for Python's published packages
domain: pypi.org
keywords: ('python', 'programming', 'scripting', 'data', 'automation')
```

事实上，`Prototype`按预期工作。我们可以看到原始`Website`对象及其克隆的信息。

通过查看每个`Website`对象的 ID 值，我们可以看到两个地址是不同的。

# 单例模式

单例模式是面向对象编程的一个原始设计模式，它限制了一个类的实例化只能有一个对象，这在需要单个对象来协调系统动作时非常有用。

基本思想是，为了满足程序的需求，只为特定类创建一个执行特定工作的实例。为了确保这一点，我们需要防止类被多次实例化和克隆的机制。

在 Python 程序员社区中，单例模式实际上被认为是一种反模式。让我们首先探讨这个模式，然后我们将讨论我们被鼓励在 Python 中使用的替代方法。

## 现实世界示例

在现实生活中，我们可以想到一艘船或船的船长。在船上，他们是负责人。他们负责重要的决策，并且由于这个责任，许多请求都指向他们。

另一个例子是办公室环境中的打印机打印队列，它确保打印作业通过一个单一点协调，避免冲突并确保有序打印。

## 单例模式的用例

单例设计模式在你需要创建单个对象或需要某种能够维护程序全局状态的对象时非常有用。

其他可能的用例如下：

+   控制对共享资源的并发访问——例如，管理数据库连接的类

+   一种跨越应用程序不同部分或不同用户访问的服务或资源，并执行其工作——例如，日志系统或实用程序的核心类

## 实现单例模式

如前所述，单例模式确保一个类只有一个实例，并提供了一个全局点来访问它。在这个例子中，我们将创建一个 `URLFetcher` 类，用于从网页获取内容。我们希望确保只有一个此类实例存在，以跟踪所有获取的 URL。

假设你在程序的多个部分有多个获取器，但你希望跟踪所有已获取的 URL。这是一个单例模式的典型用例。通过确保程序的所有部分都使用相同的获取器实例，你可以轻松地在同一位置跟踪所有已获取的 URL。

初始时，我们创建了一个简单的 `URLFetcher` 类。这个类有一个 `fetch()` 方法，用于获取网页内容并将 URL 存储在列表中：

```py
import urllib.request
class URLFetcher:
    def __init__(self):
        self.urls = []
    def fetch(self, url):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            if response.code == 200:
                page_content = response.read()
             with open("content.html", "a") as f:
                 f.write(page_content + "\n")
             self.urls.append(url)
```

为了检查我们的类是否是 `is` 操作符。如果它们相同，那么它就是一个单例：

```py
if __name__ == "__main__":
    print(URLFetcher() is URLFetcher())
```

如果你运行此代码 (`ch03/singleton/before_singleton.py`)，你会看到以下输出：

```py
False
```

这个输出显示，在这个版本中，类还没有遵循单例模式。为了使其成为单例，我们将使用 **元类** 技术。

其他信息

Python 中的元类是一个定义了类如何行为的类。

我们将创建一个 `SingletonType` 元类，确保只有一个 `URLFetcher` 实例存在，如下所示：

```py
import urllib.request
class SingletonType(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            obj = super(SingletonType, cls).__call__(*args, **kwargs)
            cls._instances[cls] = obj
        return cls._instances[cls]
```

现在，我们修改我们的 `URLFetcher` 类以使用这个元类，如下所示：

```py
class URLFetcher(metaclass=SingletonType):
    def __init__(self):
        self.urls = []
    def fetch(self, url):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            if response.code == 200:
                page_content = response.read()
                with open("content.html", "a") as f:
                    f.write(str(page_content))
                self.urls.append(url)
```

最后，我们创建了一个 `main()` 函数并调用它来测试我们的单例，代码如下：

```py
def main():
    my_urls = [
            "http://python.org",
            "https://planetpython.org/",
            "https://www.djangoproject.com/",
    ]
    print(URLFetcher() is URLFetcher())
    fetcher = URLFetcher()
    for url in my_urls:
        fetcher.fetch(url)
    print(f"Done URLs: {fetcher.urls}")
if __name__ == "__main__":
    main()
```

下面是我们在代码中执行的操作的总结 (`ch03``/singleton/singleton.py`)：

1.  我们从所需的模块导入开始 (`urllib.request`)。

1.  我们定义了一个 `SingletonType` 类，它有一个特殊的 `__call__()` 方法。

1.  我们定义了 `URLFetcher` 类，该类实现了网页的获取器，并通过 `urls` 属性初始化它；如前所述，我们添加了它的 `fetch()` 方法。

1.  最后，我们添加了我们的 `main()` 函数，并添加了 Python 中用于调用它的传统代码片段。

为了测试实现，运行 `python ch03/singleton/singleton.py` 命令。你应该得到以下输出：

```py
True
Done URLs: ['http://python.org', 'https://planetpython.org/', 'https://www.djangoproject.com/']
```

此外，你将发现已创建了一个名为 `content.html` 的文件，其中包含了来自不同 URL 的 HTML 文本。

因此，程序按预期完成了任务。这是一个演示如何使用单例模式的例子。

## 你应该使用单例模式吗？

虽然单例模式有其优点，但它可能并不总是管理全局状态或资源的最 Pythonic 方法。我们的实现示例是有效的，但如果我们停下来再次分析代码，我们会注意到以下情况：

+   实现所用的技术相当高级，不易向初学者解释。

+   通过阅读 `SingletonType` 类的定义，如果不看名字，你可能不会立即看出它提供了一个单例的元类。

在 Python 中，开发者通常更喜欢单例的简单替代方案：使用模块级全局对象。

注意

Python 模块充当自然命名空间，可以包含变量、函数和类，这使得它们非常适合组织和共享全局资源。

通过采用全局对象技术，正如布兰登·罗德斯在其所谓的*全局对象模式*（[`python-patterns.guide/python/module-globals/`](https://python-patterns.guide/python/module-globals/））中解释的那样，你可以在不需要复杂实例化过程或迫使一个类只有一个实例的情况下，达到单例模式相同的结果。

作为练习，你可以使用全局对象重写我们示例的实现。为了参考，定义全局对象的等效代码在`ch03/singleton/instead_of_singleton/example.py`文件中；有关其使用，请查看`ch03/singleton/instead_of_singleton/use_example.py`文件。

# 对象池模式

对象池模式是一种创建型设计模式，它允许你在需要时重用现有对象，而不是创建新的对象。这种模式在初始化新对象在系统资源、时间等方面的成本较高时特别有用。

## 现实世界中的例子

考虑一个汽车租赁服务。当客户租车时，服务不会为他们制造一辆新车。相反，它从可用的汽车池中提供一辆。一旦客户归还了汽车，它就会回到池中，准备好供下一个客户使用。

另一个例子是一个公共游泳池。而不是每次有人想要游泳时都往游泳池里加水，而是对水进行处理并重复使用，供多个游泳者使用。这既节省了时间又节省了资源。

## 对象池模式的用例

对象池模式在资源初始化成本高昂或耗时的情况下特别有用。这可能涉及 CPU 周期、内存使用，甚至网络带宽。例如，在一个射击视频游戏中，你可能会使用这种模式来管理子弹对象。每次开枪时创建一个新的子弹可能会消耗大量资源。相反，你可以有一个子弹对象池，这些对象可以重复使用。

## 实现对象池模式

让我们实现一个可重用`car`对象池，用于汽车租赁应用程序，以避免重复创建和销毁它们。

首先，我们需要定义一个`Car`类，如下所示：

```py
class Car:
    def __init__(self, make: str, model: str):
        self.make = make
        self.model = model
        self.in_use = False
```

然后，我们开始定义一个`CarPool`类及其初始化，如下所示：

```py
class CarPool:
    def __init__(self):
        self._available = []
        self._in_use = []
```

我们需要表达当客户端获取一辆车时会发生什么。为此，我们在类上定义了一个方法，执行以下操作：如果没有可用的汽车，我们实例化一辆并添加到池中可用的汽车列表中；否则，我们返回一个可用的`car`对象，同时执行以下操作：

+   将`car`对象中的`_in_use`属性设置为`True`

+   将`car`对象添加到“正在使用”的车辆列表中（存储在`pool`对象的`_in_use`属性中）

我们将那个方法的代码添加到类中，如下所示：

```py
    def acquire_car(self) -> Car:
        if len(self._available) == 0:
            new_car = Car("BMW", "M3")
            self._available.append(new_car)
        car = self._available.pop()
        self._in_use.append(car)
        car.in_use = True
        return car
```

然后我们添加了一个处理客户释放车辆时的方法，如下所示：

```py
    def release_car(self, car: Car) -> None:
        car.in_use = False
        self._in_use.remove(car)
        self._available.append(car)
```

最后，我们添加了一些测试实现结果的代码，如下所示：

```py
if __name__ == "__main__":
    pool = CarPool()
    car_name = "Car 1"
    print(f"Acquire {car_name}")
    car1 = pool.acquire_car()
    print(f"{car_name} in use: {car1.in_use}")
    print(f"Now release {car_name}")
    pool.release_car(car1)
    print(f"{car_name} in use: {car1.in_use}")
```

下面是我们在代码中执行的操作的摘要（在文件`ch03/object_pool.py`中）：

1.  我们定义了一个`Car`类。

1.  我们定义了一个带有`acquire_car()`和`release_car()`方法的`CarPool`类，如前所述。

1.  我们添加了测试实现结果的代码，如前所述。

要测试程序，请运行以下命令：

```py
python ch03/object_pool.py
```

你应该得到以下输出：

```py
Acquire Car 1
Car 1 in use: True
Now release Car 1
Car 1 in use: False
```

干得好！这个输出表明我们的对象池模式实现按预期工作。

# 摘要

在本章中，我们看到了*创建型设计模式*，这对于构建灵活、可维护和模块化的代码至关重要。我们通过检查工厂模式的两种变体开始了本章，每种变体都为对象创建提供了独特的优势。接下来，我们探讨了构建者模式，它提供了一种更易读、更易于维护的方式来构建复杂对象。随后，原型模式引入了一种高效克隆对象的方法。最后，我们通过讨论单例和对象池模式结束了本章，这两种模式都旨在优化资源管理并确保应用程序中状态的一致性。

现在，我们拥有了这些对象创建的基础模式，我们为下一章做好了准备，我们将发现*结构性*设计模式。
