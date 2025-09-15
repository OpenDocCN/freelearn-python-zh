## 第十章：10

使用类型匹配和注释进行工作

本章将探讨我们如何与具有各种数据类型的数据结构一起工作。这通常意味着检查属性的类型、元组的一个元素或字典中的一个值。

在前面的章节中，我们避免过多地关注数据验证的考虑。在本章中，我们将仔细检查输入值，以确保它们符合预期的数据类型和值范围。

这种数据验证是一种类型检查。它验证的值域比整数或字符串等非常广泛的类要窄。应用程序必须检查对象的值，以确保它们适用于预期的目的。

一些数据结构，如 JSON 或 XML 文档，可以包含各种数据类型的对象。一个常见的情况可以总结为第一范式（1NF），其中集合中的每个项目都是同一类型。然而，这并不普遍。当解析复杂的文件，如编程语言语句时，我们会看到一系列不同的数据类型。不同类型的存在意味着应用程序软件不能简单地假设一个单一、一致的类型，而必须处理可用的数据。

在本章中，我们将探讨与类型和类型匹配相关的多个配方：

+   使用类型提示进行设计

+   使用内置的类型匹配函数

+   使用 match 语句

+   处理类型转换

+   使用 Pydantic 实现更严格的类型检查

+   包含运行时有效值检查

# 10.1 使用类型提示进行设计

函数定义中的注释在 2006 年引入到语言语法中，没有正式的语义。注释的想法附带了一系列潜在的使用案例，其中之一是类型检查。在 2014 年，类型提示的想法得到了巩固和正式化，成为了一个类型模块和一些相关工具，包括 mypy 工具。

几年前，注释是一种通用的语法，类型提示是注释的一个特定用例。到 2017 年，注释的其他用途已被弃用，并且注释语法明确专注于类型提示。虽然注释和类型提示之间曾经存在细微的差别，但这种区别已经消失，留下了两个同义词。

使用类型提示有三个重要方面：

+   类型提示是可选的。我们可以不使用类型提示来编写 Python 代码。

+   类型提示可以逐步应用。应用程序的一部分可以有提示，而另一部分则没有。像 mypy 这样的工具可以容忍带有和不带有提示的代码的混合。

+   类型提示在运行时不会被使用，并且没有性能开销。

在整本书中，我们将提示视为良好软件设计的必要条件。它们与单元测试和连贯的文档一样重要，这两者也是技术上的可选项，但对于可信赖的软件来说是必不可少的。我们发现它们通过强制执行一定程度的严谨性和正式性来帮助防止问题。

Python 的处理依赖于鸭子类型规则。有关更多背景信息，请参阅第八章，特别是 Leveraging Python’s duck typing 食谱。我们有两种广泛的设计模式可供选择：

+   具有共同超类的一严格层次结构。

+   利用鸭子类型，一组类可以具有共同特性，通常定义为指定相关特性的协议。

在这个食谱中，我们将探讨两种设计包含类型提示且可由 mypy 等工具检查的代码的方法。

## 10.1.1 准备工作

我们将研究一个涉及处理在源文件中混合在一起的两个不同类型数据的问题。在这种情况下，我们将使用具有大量数据文件的数据目录的内容进行分类。此外，我们还有一个 src 目录，其中包含大量包含应用程序程序和脚本的子目录。我们想要创建一组数据结构来表示两种不同的数据文件类：

+   未被任何应用程序程序或脚本命名的数据文件

+   被一个或多个应用程序程序引用的数据文件

## 10.1.2 如何做...

设计此类程序有两种广泛的方法：

+   首先概述数据类型和转换，然后编写代码以适应这些类型。

+   首先编写代码，然后为工作代码添加类型提示。

都不能说是最好的。在许多情况下，两者是并行发展的。

我们将在本食谱中分别查看这些内容的各种变体。

### 首先类型提示设计

我们将处理各种类的对象。在这个变体中，我们将首先定义类型提示，然后填写所需的处理。以下是从类定义开始定义分类器和相关类的方法：

1.  定义两个子类。在这个例子中，我们将它们称为未引用文件和引用文件。对于每个类，写一句话来描述每个类实例的独特用途。这些将作为类定义的起点。

1.  选择合适的可用类。这可能是一个具有可变属性的普通类、一个 NamedTuple 或一个@dataclass。通常从@dataclass 开始可以提供最大的灵活性。在命名元组、dataclasses 和冻结数据 classes 之间切换涉及最小的语法变化：

    ```py
    from pathlib import Path 

    from dataclasses import dataclass 

    @dataclass 

    class Referenced: 

        """Defines a data file and applications that reference it."""
    ```

    未引用类定义将与适当的文档字符串相似。

1.  添加定义每个实例状态的属性和值。对于 Referenced 类，这是 Path 以及每个有引用的源文件的 Path 对象集合。这两个属性定义看起来是这样的：

    ```py
    datafile: Path 

        recipes: list[Path] 
    ```

    对于 Unreferenced 类，实际上并没有很多其他属性，除了路径。这提出了一个有趣的问题：这值得一个单独的类声明，还是它可以简单地是一个 Path 对象？

    由于 Python 允许类型别名和类型联合，实际上不需要 Unreferenced 类；现有的 Path 就足够了。提供这个类型别名是有帮助的：

    ```py
    from typing import TypeAlias 

    Unreferenced: TypeAlias = Path
    ```

1.  正式化这些不同类的联合。

    ```py
    ContentType: TypeAlias = Unreferenced | Referenced
    ```

现在我们有了类型定义，我们可以编写一个函数，该函数是 ContentType 类联合的迭代器。这个函数将产生一系列 Unreferenced 和 Referenced 对象，每个数据文件一个。

函数可能看起来是这样的：

```py
def datafile_iter(base: Path) -> Iterator[ContentType]: 

    data = (base / "data") 

    code = (base / "src") 

    for path in sorted(data.glob("*.*")): 

        if not path.is_file(): 

            continue 

        used_by = [ 

            chap_recipe.relative_to(code) 

            for chap_recipe in code.glob("**/*.py") 

            if ( 

                chap_recipe.is_file() 

                and "__pycache__" not in chap_recipe.parts 

                and ".venv" not in chap_recipe.parts 

                and "ch10" not in chap_recipe.parts 

                and path.name in chap_recipe.read_text() 

            ) 

        ] 

        if used_by: 

            yield Referenced(path.relative_to(data), used_by) 

        else: 

            yield path.relative_to(data)
```

datafile_iter()函数会跳过数据目录中的任何非文件名。它还会跳过一些源代码目录、__pycache__ 和.venv。此外，我们必须忽略第十章中的一些文件，因为这些文件将包含数据文件名称的测试用例，从而产生令人困惑的结果。

如果数据文件名出现在源文件中，引用将被保存在 used_by 集合中。具有非空 used_by 集合的文件将创建一个 Referenced 实例。其余的文件是 Path 对象；由于 TypeAlias，这些也被识别为 Unreferenced 实例。我们不需要正式地将 Path 对象转换为 Unreferenced 类型。像 mypy 这样的工具将使用 TypeAlias 来查看等价性，而无需任何额外的代码。

结果迭代器提供了一系列不同类型的对象。在使用 match 语句的配方中，我们将探讨处理不同类型对象的便捷方法。

### 首先进行代码设计

我们将处理多种类的对象。在这个变体中，我们将首先定义处理程序，然后加入类型提示以阐明我们的意图。以下是从函数定义开始定义分类器和相关类的方法：

1.  从提供所需参数的函数定义开始：

    ```py
    def datafile_iter(base): 

        data = (base / "data") 

        code = (base / "src")
    ```

1.  编写处理程序以累积所需的数据值。在这种情况下，我们需要遍历数据文件名称。对于每个数据文件，我们需要在所有源文件中查找引用。

    ```py
        for path in sorted(data.glob("*.*")): 

            if not path.is_file(): 

                continue 

            used_by = [ 

                chap_recipe.relative_to(code) 

                for chap_recipe in code.glob("**/*.py") 

                if ( 

                        chap_recipe.is_file() 

                        and "__pycache__" not in chap_recipe.parts 

                        and ".venv" not in chap_recipe.parts 

                        and "ch10" not in chap_recipe.parts 

                        and path.name in chap_recipe.read_text() 

                ) 

            ]
    ```

1.  决定函数的各种输出需要什么。在某些情况下，我们可以产生包含各种可用值的元组对象。

    ```py
            if used_by: 

                yield (path.relative_to(data), used_by) 

            else: 

                yield path.relative_to(data)
    ```

    对于源中没有引用的路径，我们产生 Path 对象。对于源中有引用的路径，我们可以产生数据 Path 和源 Path 实例的列表。

1.  对于具有更复杂内部状态的对象，考虑引入类定义以正确封装状态。对于这个例子，引入一个具有引用的数据文件类型是有意义的。这将导致用以下类似的 NamedTuple 替换一个简单、匿名的元组：

    ```py
    from typing import NamedTuple 

    class Referenced(NamedTuple): 

        datafile: Path 

        recipes: list[Path]
    ```

    这反过来又导致对 Referenced 实例的 yield 语句进行修订。

    ```py
    yield Referenced(path.relative_to(data), used_by) 
    ```

1.  回顾函数定义以添加类型提示。

    ```py
    def datafile_iter_2(base: Path) -> Iterator[Path | Referenced]:
    ```

调方程序的两个变体中的处理几乎相同。差异在于如何最好地呈现结果的选择。在前一个例子中，创建了一个显式的联合类型 Content_Type。对于这个版本，联合类型是隐式的。

## 10.1.3 它是如何工作的...

Python 的鸭子类型允许在设计中有很大的灵活性。我们可以从类型定义开始，也可以从代码开始并添加类型提示。最终的代码往往会相似，因为它对相同的数据执行相同的处理。

代码优先或类型优先的选择可能会导致对性能或优化的洞察。每个选择都强调最终代码的特定属性。代码优先的方法可能强调简单的处理，而类型优先可能强调正在处理的对象的统一性。选择方法也可能源于作者对 Python 类型的舒适度。

在某些情况下，编写类型提示的过程可能会暗示算法或优化。这可能导致对已编写代码的有益重构。

需要注意的是，类型提示的存在与否对性能没有影响。任何性能的提升（或损失）都是普通的设计问题，使用类型提示可能会使这些问题更加明显。

## 10.1.4 更多内容...

当将一个大问题分解成更小的部分时，小部分之间的接口是设计过程中必须早期做出的关键设计决策。对数据结构的早期决策通常会导致整体上采用类型优先的设计过程。面向外部的组件必须具有明确定义接口。支持这些外部组件的函数或方法可以设计得更加自由，约束更少。

这导致在复杂软件的整体架构中首先考虑类型，当在更详细的层次上工作时，保留类型优先或代码优先的设计选择。当我们考虑分布式应用程序——如网络服务——其中服务器和客户端位于不同的机器上时，我们发现类型优先是必不可少的。

随着代码量的增加，类型提示的重要性也在增加。很难将大量细节记住在脑海中。有一个类型提示来总结更复杂的数据结构可以减少代码周围的细节杂乱。

在分布式计算环境中，我们经常需要考虑某些组件可能不是 Python 程序。在这些情况下，我们无法共享 Python 类型提示。这意味着我们被迫使用存在于 Python 之外的模式定义，但它提供了对 Python 类型的所需映射。

跨越语言的这类正式定义的例子包括 JSON Schema、Protocol Buffers、AVRO 以及许多其他。JSON Schema 方法是典型的，并且被许多 Python 工具支持。在本章后面，我们将探讨使用 Pydantic，它支持使用 JSON Schema 定义数据。

## 10.1.5 参见

+   在第十一章的阅读 JSON 和 YAML 文档配方中，我们将重新使用 JSON 文档来处理复杂数据。

+   在本章后面的使用 match 语句配方中，我们将探讨如何使用 match 语句处理各种类型的数据。这使得处理类型联合相对容易。

+   在本章后面的使用 Pydantic 实现更严格的类型检查配方中，我们将探讨使用 pydantic 包进行更强的类型定义。

# 10.2 使用内置类型匹配函数

当我们有一个混合类型的对象集合时，我们通常需要区分这些类型。当我们使用自己定义的类时，我们可以定义正确多态的类。这通常不是使用 Python 的内部对象或处理涉及我们定义的类和 Python 内置类的数据集合的情况。 

当我们完全使用自己的类时，我们可以设计它们具有共同的方法和属性，但根据涉及哪个子类提供不同的行为。这种设计符合 S.O.L.I.D 设计原则中的“L”原则：Liskov 替换原则。任何子类都可以替换基类使用，因为它们都有一个共同的方法定义集。有关更多信息，请参阅第八章。

这种以抽象驱动的设 计在 Python 中并不总是需要的。由于 Python 的鸭子类型，设计不需要一个共同的基类。在某些情况下，这甚至不切实际：我们可能有多种类型而没有统一的抽象。与内置类以及我们自己的类定义中的对象混合工作是非常常见的。我们不能对内置类施加多态性。

我们如何利用内置函数来编写对类型灵活的函数和方法？对于这个配方，我们将重用本章前面使用类型提示进行设计配方中的处理。

## 10.2.1 准备工作

在使用类型提示进行设计的配方中，我们定义了一个名为 datafile_iter()的函数，该函数发出两种不同的对象：Path 对象和 Referenced 对象。一个 Referenced 对象是一组 Path 实例的集合，显示一个被一个或多个应用程序使用的数据文件。一个独立的 Path 对象是一个未被任何应用程序使用的数据文件。这些未引用的路径是移除以减少杂乱的对象。

我们需要以不同的方式处理这两类对象。它们由单个生成函数 datafile_iter()创建。此函数发出一系列未引用和引用实例。这种混合意味着应用程序必须通过类型过滤对象。

应用程序将与一系列对象一起工作。这些对象将由以下定义的函数创建：

```py
from collections.abc import Iterator 

DataFileIter: TypeAlias = Iterator[Unreferenced | Referenced] 

def datafile_iter(base: Path) -> DataFileIter:
```

datafile_iter()函数将生成一系列未引用和引用的对象。这将反映给定目录中文件的状态。一些将在源代码中有引用；其他则没有任何引用。请参阅使用类型提示进行设计配方中的此函数。

## 10.2.2 如何操作...

执行分析的应用程序函数将消费各种类型的对象。该函数设计如下：

1.  从以下定义开始，该定义显示了消耗的类型：

    ```py
    from collections.abc import Iterable 

    def analysis(source: Iterable[Unreferenced | Referenced]) -> None:
    ```

1.  创建一个空列表，该列表将保存具有引用的数据文件。编写 for 语句以从源可迭代对象中消费对象，并填充该列表：

    ```py
        good_files: list[Referenced] = [] 

        for file in source:
    ```

1.  为了通过类型区分对象，我们可以使用 isinstance()函数来查看一个对象是否是给定类型的类。

    要区分类，请使用 isinstance()函数：

    ```py
            if isinstance(file, Unreferenced): 

                print(f"delete {file}") 

            elif isinstance(file, Referenced): 

                good_files.append(file)
    ```

1.  虽然技术上是不必要的，但似乎总是明智地包括一个 else 条件，在不太可能的情况下，如果 datafile_iter 函数以某种惊人的方式更改，则引发异常：

    ```py
            else: 

                raise ValueError(f"unexpected type {type(file)}")
    ```

    关于此设计模式的更多信息，请参阅第二章中的设计复杂的 if...elif 链配方。

1.  编写最终的总结：

    ```py
        print(f"Keep {len(good_files)} files")
    ```

## 10.2.3 它是如何工作的...

isinstance()函数检查一个对象属于哪些类。第二个参数可以是单个类或替代类的元组。

重要的是要注意，一个对象通常有许多父类，形成一个从类对象起源的晶格。如果使用多重继承，可以通过超级类定义有大量的路径。isinstance()函数检查所有替代父类。

isinstance()函数不仅了解在应用程序中导入和定义的类，还了解 TypeAlias 名称。这为我们提供了很大的灵活性，可以在类型提示中使用有意义的名称。

在 Python 3.12 中，TypeAlias 构造可以替换为新的 type 语句：

```py
type Unreferenced = Path
```

有关 mypy 工具对类型语句的支持的更多信息，请参阅[Mypy Issue #15238](https://github.com/python/mypy/issues/15238)。

在这个问题得到解决之前，我们选择在这本书中使用 TypeAlias。

## 10.2.4 更多内容...

isinstance()函数是与 filter()高阶函数配合得很好的布尔函数。有关更多信息，请参阅第九章（ch013_split_000.xhtml#x1-5270004）中的选择子集 – 三种过滤方式配方。

除了内置的 isinstance()函数用于查询对象外，还有一个 iscsubclass()函数允许应用程序检查类型定义。区分类的实例和类对象很重要；iscsubclass()函数用于检查类型定义。iscsubclass()函数通常用于元编程：关注软件本身而不是应用程序数据的软件。当设计处理对象类型的函数而不是对象时，iscsubclass()函数是必要的。

在检查对象类型时，match 语句通常比 isinstance()函数更好。原因是 match 语句的 case 子句具有非常复杂的类型模式匹配，而 isinstance()函数仅限于确保对象在其父类中具有给定的类（或类元组中的类）。

## 10.2.5 另请参阅

+   有关替代方案，请参阅使用 match 语句配方。

+   有关与玩牌和它们所涉及的有趣类层次结构相关的多个配方，请参阅第七章（ch011_split_000.xhtml#x1-3760007）和第八章（ch012.xhtml#x1-4520008）。

# 10.3 使用 match 语句

定义一组紧密相关的类型的一个重要原因是为了区分应用于对象的处理方式。提供不同行为的一种技术是通过使用多态设计：多个子类提供了对公共函数的不同实现。当我们完全使用自己的类时，我们可以设计它们具有共同的方法和属性，但根据涉及哪个子类提供不同的行为。这一点在第八章（ch012.xhtml#x1-4520008）中有详细说明。

当与 Python 的内部对象一起工作，或者与涉及我们定义的类和 Python 内部部分内置类的数据集合一起工作时，通常不可能实现这一点。在这些情况下，简单地依赖类型匹配来实现不同的行为会更简单。本章中使用内置类型匹配函数配方中展示了一种方法。

我们还可以使用 match 语句来编写灵活的函数和方法，这些函数和方法可以处理各种类型的参数值。对于这个配方，我们将重用本章前面提到的使用类型提示进行设计和使用内置类型匹配函数配方中的处理过程。

## 10.3.1 准备工作

在使用类型提示进行设计配方中，我们定义了一个 datafile_iter()函数，该函数发出两种不同的对象类型：Path 对象和 Referenced 对象。

我们需要以不同的方式处理这两种类型的对象。这种混合意味着应用程序必须通过它们的类型来过滤它们。

## 10.3.2 如何做...

应用程序将处理一系列不同类型的对象。函数设计如下：

1.  从以下定义开始，显示消耗的类型：

    ```py
    from collections.abc import Iterable 

    def analysis(source: Iterable[Unreferenced | Referenced]) -> None:
    ```

    这个函数将消耗一个对象的可迭代序列。这个函数将计算具有引用的对象数量。它将建议删除没有引用的文件。

1.  创建一个空列表，用于存储具有引用的数据文件。编写 for 语句以从源可迭代中消耗对象：

    ```py
        good_files: list[Referenced] = [] 

        for file in source:
    ```

1.  使用文件变量开始编写 match 语句：

    ```py
            match file:
    ```

1.  要处理各种类别的文件，创建显示必须匹配的对象类型的 case 语句。这些 case 语句在 match 语句内缩进：

    ```py
    case Unreferenced() as unref: 

                    print(f"delete {unref}") 

                case Referenced() as ref: 

                    good_files.append(file) 
    ```

1.  虽然技术上不是必需的，但似乎总是明智地包括一个 case _: condition。_ 将匹配任何内容。在这个子句的主体中，如果 datafile_iter 函数以某种惊人的方式被更改，可能会抛出异常：

    ```py
                case _: 

                    raise ValueError(f"unexpected type {type(file)}")
    ```

    更多关于这种设计模式的信息，请参阅第二章中的设计复杂的 if...elif 链配方。

1.  编写最终的总结：

    ```py
        print(f"Keep {len(good_files)} files")
    ```

## 10.3.3 它是如何工作的...

match 语句使用一系列 case 子句来建立与给定对象匹配的类。虽然存在许多不同的 case 子句，但一个常见的 case 是 case class() as name: variant，称为类模式。在括号内，我们可以提供子模式来匹配具有特定参数类型的对象。

对于这个例子，我们不需要更复杂的匹配模式。我们可以提供一个看起来像实例的东西——由类名和()组成——以表明 case 子句将匹配类的实例。不需要有关实例结构的任何额外细节。

case Unreferenced()的使用几乎看起来像表达式 Unreferenced()将创建 Unreferenced 类的一个实例。这里的意图不是创建一个对象，而是编写一个看起来非常像对象创建的表达式。这种语法有助于阐明使用 case 来匹配命名类中任何对象的意图。

其他模式允许匹配简单的字面值、序列和映射，以及类。此外，还有方法提供替代方案组，甚至可以通过与模式匹配一起使用的守卫条件应用额外的过滤。

case _ 子句是一个通配符子句。它将匹配在匹配语句中提供的任何内容。_ 变量名在这里有特殊意义，并且只能使用这个变量。

这个设计的关键是 case 定义的清晰性。这些比一系列 elif 子句中的 isinstance()函数评估更易于阅读。

## 10.3.4 更多内容...

我们将扩展这个配方，展示这些 case 子句中一些复杂的类型匹配。考虑我们想要从只包含一个引用它的应用程序列表中的引用文件中分离出来的情况。

我们正在寻找看起来像这个具体示例的对象：

```py
single use: Referenced(datafile=PosixPath(’race_result.json’), recipes=[PosixPath(’ch11/recipe_06.py’)])
```

这种情况可以总结为 Referenced(_, [Path()])。我们想要匹配一个 Referenced 类的实例，其中第二个参数是一个包含单个 Path 实例的列表。

这变成了一个新的 case 子句。以下是新的、更具体的 case 子句，后面跟着更一般的 case 子句：

```py
 case Referenced(_, [Path()]) as single: 

                print(f"single use: {single}") 

                good_files.append(single) 

            case Referenced() as multiple: 

                good_files.append(multiple)
```

匹配语句按顺序处理情况。更具体的情况必须先于更不具体的情况。如果我们颠倒这两个情况的顺序，case Referenced()将先匹配，而 case Referenced(_, [Path()])甚至不会被检查。最一般的情况，case _:，必须是最后的。

## 10.3.5 参考信息

+   请参阅使用内置类型匹配函数的配方，了解使用内置 isinstance()函数的替代方法。

+   请参阅第八章，了解与多态类设计相关的几个配方。有时，这可以减少对类型匹配的需求。

# 10.4 处理类型转换

Python 的一个有用特性是“数值塔”概念。请参阅 Python 标准库文档中的数值塔。这个概念是指数值可以从整数移动到有理数，再到实数，最后到复数，沿着塔“向上”移动。

数值转换基于这样的想法，即存在几个重叠的数值域。这些包括ℤ整数、ℚ有理数、ℙ无理数、ℝ实数和ℂ复数。这些形成了一个嵌套的集合系列：ℤ ⊂ℚ ⊂ℝ ⊂ℂ。此外，ℚ ∪ℙ = ℝ：实数包括有理数和无理数。

这些内置的数值类型遵循抽象概念：

+   ℂ通过复数类型实现。任何低于此类型的类型都可以转换为复数值。

+   ℝ由 float 类型支持。需要注意的是，float 涉及近似，并不完全符合实数的数学理想。当这个类中的运算符遇到 int 或分数值时，它将创建等效的 float 值。

+   ℚ使用 fractions 模块中的 Fraction 类。当 Fraction 类中的算术运算符遇到 int 时，它将静默地创建一个与整数具有相同值的 Fraction。![z 1](img/file73.png) = z。

+   ℤ是 int 类。

通常，Python 语言避免过多地转换为其他类型。例如，字符串不会自动解析以创建数值。需要使用显式的内置函数如 int()或 float()来处理包含数字的字符串。

我们经常希望自己的类型共享这种行为。我们希望我们的函数是灵活的，并在需要时将对象转换为其他类型。例如，我们可能希望允许纬度-经度点的多种表示。这些替代方案可能包括：

+   一个包含两个浮点数值的元组

+   一对字符串，每个字符串代表一个浮点值

+   一个包含两个由逗号字符分隔的数值的单个字符串

与数值塔一样，我们自己的类定义需要将其他类型转换为所需的目标类型。

## 10.4.1 准备工作

我们将考虑一个计算地球上表面两点之间距离的函数。这涉及到一些巧妙的球面三角学。更多信息，请参阅第三章，特别是基于部分函数选择参数顺序配方。还可以参阅第七章中的创建上下文和上下文管理器配方。

函数定义如下：

```py
def haversine( 

    lat_1: float, lon_1: float, 

    lat_2: float, lon_2: float, *, R: float) -> float: 

    ...  # etc.
```

这个定义需要将源数据转换为单个浮点值。在集成来自多个源的数据的应用中，这些转换非常常见，因此最好将它们集中到一个封装基本 haversine()计算的函数中。

我们需要一个这样的函数：

```py
def distance( 

    *args: str | float | tuple[float, float], 

    R: float = NM 

) -> float:
```

此函数将计算定义为各种数据类型的点之间的距离。*args 参数意味着所有位置参数值将组合成一个单一的元组。必须应用一系列验证规则来理解这个元组。以下是我们将开始的规则：四个浮点值：直接使用这些值。例如：distance(36.12, -86.67, 33.94, -118.40, R=6372.8)。四个字符串：将这些字符串转换为浮点值。例如：distance("36.12", "-86.67", "33.94", "-118.40", R=6372.8)。两个字符串：解析每个字符串，以逗号分隔。每个字符串应包含两个浮点值。例如：distance("36.12,-86.67", "33.94,-118.40", R=6372.8)。两个元组：解包每个元组以确保它包含两个浮点值。例如：distance((36.12, -86.67), (33.94, -118.40), R=6372.8)。

理想情况下，也许支持这些组合也不错。我们将设计一个执行所需类型转换的函数。

## 10.4.2 如何实现...

包含类型转换的函数通常与底层处理分开构建。如果将这些处理方面的两个部分——转换和计算——分开，这有助于测试和调试：

1.  导入所需的 `literal_eval()` 函数以转换预期为 Python 字面量的字符串：

    ```py
    from ast import literal_eval
    ```

    使用这个函数，我们可以评估 `literal_eval("2,3")` 来得到一个正确的元组结果，(2, 3)。我们不需要使用正则表达式来分解字符串以查看文本的模式。

1.  定义执行转换的距离函数：

    ```py
    def distance( 

        *args: str | float | tuple[float, float], 

        R: float = NM 

    ) -> float:
    ```

1.  开始匹配各种参数模式的匹配语句。

    ```py
        match args:
    ```

1.  编写单独的情况，从更具体到更不具体。从四个不同的浮点值开始，因为不需要进行转换。浮点值的元组具有更复杂的类型结构，但不需要任何转换。

    ```py
            case [float(lat_1), float(lon_1), float(lat_2), float(lon_2)]: 

                pass 

            case ( 

                [[float(lat_1), float(lon_1)], 

                 [float(lat_2), float(lon_2)]] 

            ): 

                pass
    ```

    我们提供了 `lat_1`、`lon_1`、`lat_2` 和 `lon_2` 变量，以便将 `args` 结构中的值绑定到变量名。这使我们免去了编写解包参数元组的赋值语句。使用 `pass` 语句占位符是因为不需要进行除解包数据结构之外的其他处理。

1.  编写涉及提供的值转换的情况：

    ```py
            case [str(s1), str(s2), str(s3), str(s4)]: 

                lat_1, lon_1, lat_2, lon_2 = ( 

                    float(s1), float(s2), float(s3), float(s4) 

                ) 

            case [str(ll1), str(ll2)]: 

                lat_1, lon_1 = literal_eval(ll1) 

                lat_2, lon_2 = literal_eval(ll2)
    ```

    当参数值是四个字符串时，我们提供了四个变量来解包这四个字符串。

    当参数模式是两个字符串时，我们提供了两个变量，ll1 和 ll2，每个变量都需要被转换为两个数字元组然后解包。

1.  编写一个默认情况，它会匹配任何其他情况并引发异常：

    ```py
    case _: 

                raise ValueError(f"unexpected types in {args!r}") 
    ```

1.  现在参数已经被正确解包并且应用了任何转换，使用 `haversine()` 函数来计算所需的结果：

    ```py
        return haversine(lat_1, lon_1, lat_2, lon_2, R=R)
    ```

## 10.4.3 它是如何工作的...

类型转换的基本功能是使用匹配语句为支持的类型提供适当的转换。在这个例子中，我们容忍了可以转换和解包的字符串和元组的混合，以定位所需的四个参数值。匹配语句有许多聪明的类型匹配规则。例如，表达式 `((float(f1), float(f2)), (float(f3), float(f4)))` 将匹配两个元组，每个元组包含两个浮点值。此外，它从元组中解包值并将它们分配给四个变量。

转换值的机制也基于内置功能。`float()` 函数将数字字符串转换为浮点值或引发 `ValueError` 异常。

`ast.literal_eval()` 函数对于评估字符串形式的 Python 字面量非常方便。该函数由于仅限于字面值和一些简单数据结构（由字面值构建的元组、列表、字典和集合）而安全，因此不会评估危险的表达式。它允许我们直接将字符串 "36.12,-86.67" 解析为 (36.12, -86.67)。

## 10.4.4 更多...

使用独立的 case 子句使得添加额外的类型转换相对容易。例如，我们可能想要处理看起来像 {"lat": 36.12, "lon": -86.67} 的两个字典结构的元组。这可以与以下 case 匹配：

```py
        case ( 

             {"lat": float(lat_1), "lon": float(lon_1)}, 

             {"lat": float(lat_2), "lon": float(lon_2)} 

        ): 

            pass
```

参数元组模式周围有括号（()），这使得它很容易被拆分成多行。从字典中提取的四个值将被绑定到四个目标变量上。

如果我们想要允许更多的灵活性，我们可以考虑这种情况：我们有两个类型模式的混合参数值。例如，distance("36.12,-86.67", (33.94, -118.40), R=6372.8)。这有两种不同的格式：一个字符串和一个包含一对浮点值的元组。

而不是列举所有可能的组合，我们可以将一对值的解析分解成一个单独的函数，parse()，它将对两个参数值应用相同的转换：

```py
        case [p_1, p_2]: 

            lat_1, lon_1 = parse(p_1) 

            lat_2, lon_2 = parse(p_2)
```

这个新的 parse() 函数必须处理提供经纬度值的所有情况。这包括字符串、元组和映射。它看起来是这样的：

```py
    def parse(item: Point | float) -> tuple[float, float]: 

        match item: 

            case [float(lat), float(lon)]: 

                pass 

            case {"lat": float(lat), "lon": float(lon)}: 

                pass 

            case str(sll): 

                lat, lon = literal_eval(sll) 

            case _: 

                raise ValueError(f"unexpected types in {item!r}") 

        return lat, lon
```

这将稍微简化 distance 函数中的 match 语句。重构后的语句只处理四种情况：

```py
    match args: 

        case [float(lat_1), float(lon_1), float(lat_2), float(lon_2)]: 

            pass 

        case [str(s1), str(s2), str(s3), str(s4)]: 

            lat_1, lon_1, lat_2, lon_2 = float(s1), float(s2), float(s3), float(s4) 

        case [p_1, p_2]: 

            lat_1, lon_1 = parse(p_1) 

            lat_2, lon_2 = parse(p_2) 

        case _: 

            raise ValueError(f"unexpected types in {args!r}")
```

前两种情况处理提供了四个参数值的情况。第三种情况查看一对值，这些值可以是任何一对格式。

我们明确避免提供三个参数值的情况。这需要更多的注意来解释，因为三个参数值中的一个必须是经纬度对。其他两个值必须是分开的经纬度值。逻辑并不特别复杂，但细节偏离了这个配方的核心思想。

虽然这个配方侧重于内置类型，包括 str 和 float，但任何类型都可以使用。例如，可以很容易地在 case 子句中添加一个自定义的 Leg 类型，它具有起始和结束位置。

## 10.4.5 参见

+   关于数字和转换的更多信息，请参阅第一章的 选择 float、decimal 和 fraction 之间的区别 配方。这提供了一些关于 float 近似限制的更多信息。

+   关于 haversine() 函数的更多信息，请参阅第三章的 基于部分函数选择参数顺序 配方。还可以参阅第七章的 创建上下文和上下文管理器 配方。

# 10.5 使用 Pydantic 实现更严格的类型检查

对于大多数情况，Python 的内部处理将正确地处理许多简单的有效性检查。如果我们编写了一个将字符串转换为浮点数的函数，该函数将处理浮点值和字符串值。如果我们尝试将 float() 函数应用于 Path 对象，它将引发 ValueError 异常。

为了使类型提示可选，运行时类型检查是确保某些处理可以继续的最小检查级别。这与 mypy 等工具执行的严格检查截然不同。

类型提示不执行运行时处理。

Python（不使用任何附加包）在运行时不会进行数据类型检查或值范围检查。当操作符遇到它无法处理的类型时，会引发异常，而不考虑类型提示。

这意味着 Python 可能能够处理被提示排除的类型。可以编写一个窄提示，如 list[str]。具有给定函数体的 set[str] 对象也可能与 Pydantic 包一起工作。

在某些应用中，我们希望在运行时进行更强的检查。这些通常在需要使用扩展或插件的程序中很有帮助，我们希望确保额外的插件代码表现良好。

提供运行时类型检查的一种方法是通过使用 Pydantic 包。此模块允许我们定义带有运行时类型检查的复杂对象，以及可以广泛共享的模式定义管理。

在第五章，在 创建字典 – 插入和更新 食谱中，我们查看了一个需要解析成更有用结构的日志文件。在第九章，在 使用 yield 语句编写生成器函数 食谱中，我们查看了一个编写生成器函数的例子，该函数将解析并生成解析后的对象。我们将这些生成的对象称为 RawLog，没有类型检查或类型转换。我们应用了一个简单的转换来创建一个带有日期时间戳从文本转换为 datetime.datetime 对象的 DatedLog 实例。

Pydantic 包可以处理一些转换为 DatedLog 实例的转换，从而节省我们一些编程工作。此外，由于模式可以自动生成，我们可以构建 JSON Schema 定义并执行 JSON 序列化，而无需进行大量复杂的工作。

必须下载并安装 Pydantic 包。通常，这可以通过以下终端命令完成：

```py
(cookbook3) % python -m pip install pydantic
```

使用 python -m pip 命令确保我们将使用与当前活动虚拟环境一起的 pip 命令，在示例中显示为 cookbook3。

## 10.5.1 准备工作

日志数据中的日期时间戳以字符串值表示。我们需要解析这些值以创建适当的 datetime 对象。为了使本食谱中的内容更集中，我们将使用 Flask 编写的 Web 服务器生成的简化日志。

条目最初是类似以下的文本行：

```py
[2016-06-15 17:57:54,715] INFO in ch10_r10: Sample Message One 

[2016-06-15 17:57:54,716] DEBUG in ch10_r10: Debugging 

[2016-06-15 17:57:54,720] WARNING in ch10_r10: Something might have gone wrong
```

我们在第八章的使用更复杂结构 - 列表映射配方中看到了其他处理这种日志的例子。使用第一章中的使用正则表达式进行字符串解析配方中的 RE，我们可以将每一行分解成更有用的结构。

观察其他配方，用于解析的正则表达式具有一个重要的特性。在 (?P<name>...) 组中使用的名称被特别设计为普通的 Python 属性名称。这将很好地与我们将要构建的类定义相匹配。

我们需要定义一个类，以有用的形式捕获每条日志行的基本内容。我们将使用 Pydantic 包来定义和填充这个类。

## 10.5.2 如何操作...

1.  为了创建这个类定义，我们需要以下导入：

    ```py
    import datetime 

    from enum import StrEnum 

    from typing import Annotated 

    from pydantic import BaseModel, Field
    ```

1.  为了正确验证具有多个值的字符串，需要一个 Enum 类。我们将定义 StrEnum 的子类来列出有效的字符串值。每个类级别变量提供了一个名称和用于名称序列化的字符串字面量：

    ```py
    class LevelClass(StrEnum): 

        DEBUG = "DEBUG" 

        INFO = "INFO" 

        WARNING = "WARNING" 

        ERROR = "ERROR"
    ```

    在这个类中，Python 属性名称和字符串字面量相匹配。这不是一个要求。对于这个枚举字符串值的集合来说，这碰巧很方便。

1.  这个类将是 pydantic 包中的 BaseModel 类的子类：

    ```py
    class LogData(BaseModel):
    ```

    BaseModel 类必须是任何使用 pydantic 特性的模型的超类。

1.  我们将定义每个字段，字段名称与解析字段使用的正则表达式中的组名称相匹配。这不是一个要求，但它使得从正则表达式匹配对象的部分组字典构建 LogData 类的实例变得非常容易：

    ```py
        date: datetime.datetime 

        level: LevelClass 

        module: Annotated[str, Field(pattern=r’^\w+$’)] 

        message: str
    ```

    日期被定义为 datetime.datetime 实例。从 BaseModel 类继承的方法将处理这个转换。级别是 LevelClass 的实例。同样，BaseModel 的特性将为我们处理这个转换。我们使用了 Annotated 类型来提供类型，str，以及一个注解参数，Field(...)。这将由 BaseModel 的方法用于验证字段的内容。

这是一个生成器函数，用于读取和解析日志记录：

```py
from typing import Iterable, Iterator 

def logdata_iter(source: Iterable[str]) -> Iterator[LogData]: 

    for row in source: 

        if match := pattern.match(row): 

            l = LogData.model_validate(match.groupdict()) 

            yield l
```

这将使用正则表达式模式，pattern 来解析每条记录。组字典，match.groupdict() 将包含组名称和解析后的文本。BaseModel 的 model_validate() 方法将从编译的正则表达式创建的字典构建 LogData 类的实例。

当我们使用这个 logdata_iter 函数来创建 LogData 类的实例时，它看起来像以下示例：

```py
>>> from pprint import pprint 

>>> pprint(list(logdata_iter(data.splitlines()))) 

[LogData(date=datetime.datetime(2016, 6, 15, 17, 57, 54, 715000), level=<LevelClass.INFO: ’INFO’>, module=’ch10_r10’, message=’Sample Message One’), 

 LogData(date=datetime.datetime(2016, 6, 15, 17, 57, 54, 716000), level=<LevelClass.DEBUG: ’DEBUG’>, module=’ch10_r10’, message=’Debugging’), 

 LogData(date=datetime.datetime(2016, 6, 15, 17, 57, 54, 720000), level=<LevelClass.WARNING: ’WARNING’>, module=’ch10_r10’, message=’Something might have gone wrong’)]
```

此函数已将文本行转换为填充了适当的 Python 对象（datetime.datetime 实例和来自 LevelClass 的枚举值）的 LogData 对象：进一步，它验证了模块名称以确保它们匹配特定的正则表达式模式。

## 10.5.3 它是如何工作的...

Pydantic 包包括许多用于数据验证和类定义的工具。Python 的类型使用，以及更详细的 Annotated 类型，提供了帮助我们定义类成员的语法，包括数据转换和数据验证。在这个例子中，转换是隐含的；类提供了目标类型，从 BaseModel 类继承的方法确保源数据被正确转换为所需的目标类型。

这个小的类定义有三个不同类型的类型提示：

+   日期和级别字段涉及转换为目标类型。

+   模块字段使用注解类型为属性提供了 Pydantic Field 定义。正则表达式模式将检查每个字符串值以确保它匹配所需的模式。

+   提供的消息字段提供了一个简单的类型，该类型将与源数据类型匹配。对于此字段不会执行任何额外的验证。

@dataclass 和 BaseModel 子类的工作方式之间有一些相似之处。Pydantic 包提供了比 dataclass 定义更为复杂的定义。例如，@dataclass 不执行类型检查或任何自动数据转换。在定义 dataclass 时提供的类型信息主要对像 mypy 这样的工具感兴趣。相比之下，BaseModel 的子类执行了更多的自动化转换和运行时类型检查。

DataModel 的子类附带了许多方法。

`model_dump_json()` 和 `model_validate_json()` 方法对于网络服务特别有帮助，在这些服务中，应用程序通常与以 JSON 表示法表示的对象状态的 RESTful 传输一起工作。这些可以序列化为换行符分隔的文件，以便将多个复杂对象收集到以标准化物理格式存储的文件中。

Pydantic 包通常非常快。当前版本涉及编译为提供非常高性能的 Python 扩展。显然，缺少许多 Pydantic 功能的数据类将更快，但功能更少。然而，额外的数据验证通常值得额外的开销。

## 10.5.4 更多...

与 Pydantic 一起工作的一个好处是自动支持 JSON Schema 定义和 JSON 序列化。

这显示了我们可以如何获取模型的 JSON Schema：

```py
>>> import json 

>>> print(json.dumps(LogData.model_json_schema(), indent=2))
```

JSON Schema 的细节很长，与 Python 类的定义相匹配。我们省略了输出。

我们可以将这些 LogData 实例以 JSON 表示法进行序列化。以下是它的样子：

```py
>>> for record in logdata_iter(data.splitlines()): 

...     print(record.model_dump_json()) 

{"date":"2016-06-15T17:57:54.715000","level":"INFO","module":"ch10_r10","message":"Sample Message One"} 

{"date":"2016-06-15T17:57:54.716000","level":"DEBUG","module":"ch10_r10","message":"Debugging"} 

{"date":"2016-06-15T17:57:54.720000","level":"WARNING","module":"ch10_r10","message":"Something might have gone wrong"}
```

我们已经使用 model_dump_json()方法将对象序列化为 JSON 文档。这使我们能够将来自各种来源的文档转换为通用格式。这使得围绕通用格式创建分析处理变得容易，将解析、合并和验证与分析和分析处理的有趣结果分开。

## 10.5.5 参考信息

+   有关一些可能的附加验证规则，请参阅包含运行时有效值检查配方。

+   有关数据类的更多信息，请参阅第七章中的使用数据类处理可变对象配方。Pydantic 对数据类的变体通常比数据类模块更有用。

+   有关以 JSON 格式读取数据的更多信息，请参阅第十一章中的读取 JSON 和 YAML 文档配方。

# 10.6 包含运行时有效值检查

数据分析通常涉及大量的“数据处理”：处理无效数据或异常数据。源应用程序软件的变化很常见，导致数据文件的新格式，当解析这些文件时，可能会给下游分析应用带来问题。企业流程或政策的变更可能会导致新的数据类型或新的编码值，这可能会干扰分析处理。

类似地，当与机器和机器人（有时称为物联网）一起工作时，设备在启动时或无法正常工作时提供无效数据是很常见的。在某些情况下，当不良数据到达时，可能需要发出警报。在其他情况下，超出范围的数据需要被悄悄忽略。

Pydantic 包提供了非常复杂的验证函数，允许我们有两个选择：

+   将数据从非标准格式转换为 Python 对象。

+   对于无法转换或未能通过更具体领域检查的数据，抛出异常。

在某些情况下，我们还需要验证生成的对象在内部是否一致。这通常意味着必须检查几个字段是否相互一致。这被称为模型验证，它与单独字段验证是不同的。

验证的概念可以扩展。它可以包括拒绝无效数据，以及过滤掉对于特定应用来说有效但无趣的数据。

## 10.6.1 准备工作

我们正在查看美国国家海洋和大气管理局（NOAA）关于海岸潮汐的数据。移动一艘大型帆船意味着确保有足够的水让它浮起来。这个约束条件要求检查已知浅且难以通过的地区的潮汐高度预测。

特别是，一个名为 El Jobean 的地方，位于 Myakka 河上，有一个浅滩，在穿越时需要小心。我们可以从 NOAA [潮汐和流速](https://tidesandcurrents.noaa.gov/noaatidepredictions.html?id=8725769)网站获取潮汐预测。这个网页允许输入日期范围并下载给定日期范围的潮汐预测文本文件。

生成的文本文件看起来如下所示：

```py
NOAA/NOS/CO-OPS 

Disclaimer: These data are based upon the latest information available as of the date of your request, and may differ from the published tide tables. 

Daily Tide Predictions 

StationName: EL JOBEAN, MYAKKA RIVER 

State: FL 

Stationid: 8725769 

... 

Date           Day    Time    Pred    High/Low 

2024/04/01      Mon    04:30  -0.19  L 

2024/04/01      Mon    20:07  1.91    H 

...
```

这些数据几乎符合 CSV 格式，但一些怪癖使其难以处理。以下是一些复杂因素：

+   在有用的列标题行之前，文件有 19 行数据。

+   列使用制表符（\t）作为分隔符，而不是逗号。

+   相关数据的标题行中隐藏了一些多余的空格。

以下函数将提供干净的 CSV 行以供进一步处理：

```py
import csv 

from collections.abc import Iterator 

from typing import TextIO 

def tide_table_reader(source: TextIO) -> Iterator[dict[str, str]]: 

    line_iter = iter(source) 

    for line in line_iter: 

        if len(line.rstrip()) == 0: 

            break 

    header = next(line_iter).rstrip().split(’\t’) 

    del header[1]  # Extra tab in the header 

    reader = csv.DictReader(line_iter, fieldnames=header, delimiter=’\t’) 

    yield from reader
```

标题注释中的额外制表符用于处理标题，其中包含一个额外的空格字符。这个标题行在日期和日期列名称之间有两个制表符（\t）：

```py
 ’Date \t\tDay\tTime\tPred\tHigh/Low\n’
```

有关从列表中删除项的技术，请参阅第四章中的切片和切块列表配方。

这个列名称列表可以用来构建一个 DictReader 实例以消费其余的数据。（有关 CSV 文件，请参阅第十一章中的使用 CSV 模块读取分隔文件配方。）

我们可以使用 Pydantic 验证功能将每个字典转换为类实例。

## 10.6.2 如何做到这一点...

核心数据模型将验证数据行，创建一个类的实例。我们可以向这个类添加功能以处理特定于应用程序的处理。以下是构建这个类的方法：

1.  从每行中的数据类型导入开始，加上 BaseModel 类和一些相关类：

    ```py
    import datetime 

    from enum import StrEnum 

    from typing import Annotated 

    from pydantic import BaseModel, Field, PlainValidator
    ```

1.  定义高/低列的值域。这两个代码作为 Enum 子类的枚举：

    ```py
    class HighLow(StrEnum): 

        high = "H" 

        low = "L"
    ```

1.  由于日期文本不是 Pydantic 使用的默认格式，我们需要定义一个验证函数，该函数将从给定的字符串生成日期对象：

    ```py
    def validate_date(v: str | datetime.date) -> datetime.date: 

        match v: 

            case datetime.date(): 

                return v 

            case str(): 

                return datetime.datetime.strptime(v, "%Y/%m/%d").date() 

            case _: 

                raise TypeError("can’t validate {v!r} of type {type(v)}")
    ```

    Pydantic 验证器可以用于内部 Python 对象以及来自源 CSV 文件或 JSON 文档的字符串。当应用于 datetime.date 对象时，不需要额外的转换。

1.  定义模型。字段定义的 validation_alias 参数将从字典中的源字段中提取数据，该字段与类中目标属性名称不完全相同：

    ```py
    class TideTable(BaseModel): 

        date: Annotated[ 

            datetime.date, 

            Field(validation_alias=’Date ’), 

            PlainValidator(validate_date)] 

        day: Annotated[ 

            str, Field(validation_alias=’Day’)] 

        time: Annotated[ 

            datetime.time, Field(validation_alias=’Time’)] 

        prediction: Annotated[ 

            float, Field(validation_alias=’Pred’)] 

        high_low: Annotated[ 

            HighLow, Field(validation_alias=’High/Low’)]
    ```

    每个字段使用 Annotated 类型来定义基本类型，以及验证字符串和将它们转换为该类型所需的其他详细信息。

    天字段（包含星期几）实际上并不有用。它是从日期派生出来的数据。出于调试目的，这个数据被保留。

给定这个类，我们可以用它来验证来自一系列字典实例的模型实例。它看起来是这样的：

```py
>>> tides = [TideTable.model_validate(row) for row in dict_rows] 

>>> tides[0] 

TideTable(date=datetime.date(2024, 4, 1), day=’Mon’, time=datetime.time(4, 30), prediction=-0.19, high_low=<HighLow.low: ’L’>) 

>>> tides[-1] 

TideTable(date=datetime.date(2024, 4, 30), day=’Tue’, time=datetime.time(19, 57), prediction=1.98, high_low=<HighLow.high: ’H’>)
```

这个对象序列包含太多数据。我们可以使用 Pydantic 来过滤数据，并只传递有用的行。我们将通过修改这个类定义并创建一个包含要传递的数据规则的替代方案来实现这一点。

## 10.6.3 它是如何工作的...

BaseModel 类包括一些与类属性注释类型提示一起工作的操作。考虑这个类型提示：

```py
    date: Annotated[ 

        datetime.date, 

        Field(validation_alias=’Date ’), 

        PlainValidator(validate_date)]
```

这提供了一个基础类型 datetime.date。它提供了一个 Field 对象，可以从字典中提取名为 'Date' 的字段，并对其应用验证规则。最后，PlainValidator 对象提供了一个一步验证规则，应用于源数据。validate_date() 函数被编写为接受已验证的日期对象，并将字符串对象转换为日期对象。这允许验证用于原始数据以及 Python 对象。

在这个例子中，我们的应用程序涉及对数据域的某些缩小。有三个重要标准：

+   我们只对高潮预测感兴趣。

+   我们希望潮高至少比基准线高 1.5 英尺（45 厘米）。

+   我们需要这个操作在 10:00 之后和 17:00 之前发生。

我们可以利用 Pydantic 来执行额外的验证，以缩小数据域。这些额外的验证可以拒绝低于 1.5 英尺的最小潮高。

## 10.6.4 更多内容...

我们可以将此模型扩展以添加验证规则，将有效行的域缩小到符合我们基于一天中的时间和潮高选择标准的那些行。我们将在任何数据转换之后应用这些更窄的数据验证规则。这些规则将引发 ValidationError 异常。这扩展了从 pydantic 包中的导入。

我们将定义一系列额外的验证函数。以下是一个为低潮数据引发异常的验证器：

```py
    BaseModel, Field, PlainValidator, AfterValidator, ValidationError 

) 

def pass_high_tide(hl: HighLow) -> HighLow: 

    assert hl == HighLow.high, f"rejected low tide" 

    return hl
```

assert 语句对于这个任务来说非常简洁。这也可以用 if 和 raise 来完成。

类似的验证器可以引发异常，对于超出可接受时间窗口的数据：

```py
def pass_daylight(time: datetime.time) -> datetime.time: 

    assert datetime.time(10, 0) <= time <= datetime.time(17, 0) 

    return time
```

最后，我们可以将这些额外的验证器组合到注释类型定义中：

```py
class HighTideTable(BaseModel): 

    date: Annotated[ 

        datetime.date, 

        Field(validation_alias=’Date ’), 

        PlainValidator(validate_date)] 

    time: Annotated[ 

        datetime.time, 

        Field(validation_alias=’Time’), 

        AfterValidator(pass_daylight)]  # Range check 

    prediction: Annotated[ 

        float, 

        Field(validation_alias=’Pred’, ge=1.5)]  # Minimum check 

    high_low: Annotated[ 

        HighLow, 

        Field(validation_alias=’High/Low’), 

        AfterValidator(pass_high_tide)]  # Required value check
```

额外的验证器将拒绝不符合我们狭窄要求的标准的数据。输出将只包含高潮，潮高大于 1.5 英尺，并且在白天。

这些数据形成了一个高潮表实例的序列，如下所示：

```py
>>> from pathlib import Path 

>>> data = Path("data") / "tide-table-2024.txt" 

>>> with open(data) as tide_file: 

...     for ht in high_tide_iter(tide_table_reader(tide_file)): 

...         print(repr(ht)) 

HighTideTable(date=datetime.date(2024, 4, 7), time=datetime.time(15, 42), prediction=1.55, high_low=<HighLow.high: ’H’>) 

... 

HighTideTable(date=datetime.date(2024, 4, 10), time=datetime.time(16, 42), prediction=2.1, high_low=<HighLow.high: ’H’>) 

... 

HighTideTable(date=datetime.date(2024, 4, 26), time=datetime.time(16, 41), prediction=2.19, high_low=<HighLow.high: ’H’>)
```

我们省略了一些行，只显示第一行、中间的一行和最后一行。这些是具有 Python 对象属性的高潮表对象，适合进一步分析和处理。

Pydantic 设计的一般方法意味着组合原始数据字段、转换数据和过滤数据的规则都是分开的。我们可以放心地更改这些规则之一，而不用担心破坏应用程序的其他部分。

这个配方包括三种检查方法：

+   范围检查以确保连续值在允许的范围内。AfterValidator 用于确保字符串被转换为时间。

+   确保连续值超过限制的最小检查。对于数字，这可以通过字段定义直接完成。

+   必要值检查以确保离散值具有所需的其中一个值。AfterValidator 用于确保字符串被转换为枚举类型。

这些类型的检查在基本类型匹配之后执行，并用于应用更窄的验证规则。

## 10.6.5 参见

+   在第十一章中，我们将更深入地探讨读取数据文件。

+   参见 使用 Pydantic 实现更严格的类型检查 菜单以获取使用 Pydantic 的更多示例。Pydantic 使用编译的 Python 扩展来应用验证规则，几乎没有开销。

# 加入我们的社区 Discord 空间

加入我们的 Python Discord 工作空间，讨论并了解更多关于这本书的信息：[`packt.link/dHrHU`](https://packt.link/dHrHU)

![PIC](img/file1.png)
