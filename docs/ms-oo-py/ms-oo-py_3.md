# 第二部分：持久性和序列化

*序列化和保存-JSON、YAML、Pickle、CSV 和 XML*

*通过 Shelve 存储和检索对象*

*通过 SQLite 存储和检索对象*

*传输和共享对象*

*配置文件和持久性*

# 持久性和序列化

持久对象是已经写入某种存储介质的对象。可以从存储中检索对象并在 Python 应用程序中使用。也许对象以 JSON 形式表示并写入文件系统。也许一个**对象关系映射**（**ORM**）层已经将对象表示为 SQL 表中的行，以将对象存储在数据库中。

序列化对象有两个目的。我们对对象进行序列化是为了使它们在本地文件系统中持久化。我们还对对象进行序列化，以在进程或应用程序之间交换对象。虽然重点不同，但持久性通常包括序列化；因此，一个良好的持久性技术也将适用于数据交换。我们将看看 Python 处理序列化和持久性的几种方式。本部分的章节组织如下：

+   第九章，“序列化和保存-JSON、YAML、Pickle、CSV 和 XML”，涵盖了使用专注于各种数据表示的库进行简单持久化：JSON、YAML、pickle、XML 和 CSV。这些是 Python 数据的常见、广泛使用的格式。它们适用于持久性以及数据交换。它们更多地关注单个对象，而不是大量对象的持久性。

+   第十章，“通过 Shelve 存储和检索对象”，涵盖了使用 Python 模块（如 Shelve 和 dBm）进行基本数据库操作。这些提供了 Python 对象的简单存储，并专注于多个对象的持久性。

+   第十一章，“通过 SQLite 存储和检索对象”，转向更复杂的 SQL 和关系数据库世界。由于 SQL 特性与面向对象编程特性不匹配，我们面临阻抗不匹配问题。一个常见的解决方案是使用对象关系映射来允许我们持久化大量对象。

+   对于 Web 应用程序，我们经常使用**表述状态转移**（**REST**）。第十二章，“传输和共享对象”，将研究 HTTP 协议，JSON，YAML 和 XML 表示传输对象。

+   最后，第十三章，“配置文件和持久性”，将涵盖 Python 应用程序可以使用配置文件的各种方式。有许多格式，每种格式都有一些优点和缺点。配置文件只是一组可以轻松被人类用户修改的持久对象。

在本部分中经常出现的重要主题是在更高级别的抽象中使用的设计模式。我们将这些称为架构模式，因为它们描述了应用程序的整体架构，将其分成层或层。我们被迫将应用程序分解成片段，以便我们可以实践通常被表述为**关注点分离**的原则。我们需要将持久性与其他功能（如应用程序的核心处理和向用户呈现数据）分开。精通面向对象的设计意味着要查看更高级别的架构设计模式。

# 第九章：序列化和保存-JSON、YAML、Pickle、CSV 和 XML

要使 Python 对象持久，我们必须将其转换为字节并将字节写入文件。我们将其称为**序列化**；它也被称为编组、压缩或编码。我们将研究几种将 Python 对象转换为字符串或字节流的方法。

这些序列化方案中的每一个也可以称为**物理数据格式**。每种格式都有一些优点和缺点。没有*最佳*格式来表示对象。我们必须区分**逻辑数据格式**，它可能是简单的重新排序或更改空格使用方式，而不改变对象的值，但改变字节序列。

重要的是要注意（除了 CSV），这些表示法偏向于表示单个 Python 对象。虽然单个对象可以是对象列表，但它仍然是固定大小的列表。为了处理其中一个对象，整个列表必须被反序列化。有方法可以执行增量序列化，但这需要额外的工作。与摆弄这些格式以处理多个对象相比，有更好的方法来处理第十章中的许多不同对象的方法，*通过 Shelve 存储和检索对象*，第十一章，*通过 SQLite 存储和检索对象*，以及第十二章，*传输和共享对象*。

由于每个方案都专注于单个对象，我们受限于适合内存的对象。当我们需要处理大量不同的项目，而不是所有项目一次性放入内存时，我们无法直接使用这些技术；我们需要转移到更大的数据库、服务器或消息队列。我们将研究以下序列化表示：

+   JavaScript 对象表示法（JSON）：这是一种广泛使用的表示法。有关更多信息，请参见[`www.json.org`](http://www.json.org)。`json`模块提供了在此格式中加载和转储数据所需的类和函数。在*Python 标准库*中，查看第十九部分*Internet Data Handling*，而不是第十二部分*Persistence*。`json`模块专注于 JSON 表示，而不是 Python 对象持久性的更一般问题。

+   YAML 不是标记语言（YAML）：这是对 JSON 的扩展，可以简化序列化输出。有关更多信息，请参见[`yaml.org`](http://yaml.org)。这不是 Python 库的标准部分；我们必须添加一个模块来处理这个问题。具体来说，`PyYaml`包具有许多 Python 持久性特性。

+   **pickle**：`pickle`模块具有其自己的 Python 特定的数据表示形式。由于这是 Python 库的一部分，我们将仔细研究如何以这种方式序列化对象。这的缺点是它不适合与非 Python 程序交换数据。这是第十章，“通过 Shelve 存储和检索对象”的`shelve`模块以及第十二章，“传输和共享对象”中的消息队列的基础。

+   逗号分隔值（CSV）模块：这对于表示复杂的 Python 对象来说可能不方便。由于它被广泛使用，我们需要想办法以 CSV 表示法序列化 Python 对象。有关参考，请查看《Python 标准库》第十四部分“文件格式”，而不是第十二部分“持久性”，因为它只是一个文件格式，没有更多内容。CSV 允许我们对无法放入内存的 Python 对象集合进行递增表示。

+   **XML**：尽管存在一些缺点，但这是非常广泛使用的，因此能够将对象转换为 XML 表示法并从 XML 文档中恢复对象非常重要。XML 解析是一个庞大的主题。参考资料在《Python 标准库》第二十部分“结构化标记处理工具”中。有许多模块用于解析 XML，每个都有不同的优点和缺点。我们将重点关注`ElementTree`。

除了这些简单的类别，我们还可能遇到混合问题。一个例子是用 XML 编码的电子表格。这意味着我们有一个包裹在 XML 解析问题中的行列数据表示问题。这导致了更复杂的软件，以解开被扁平化为类似 CSV 的行的各种数据，以便我们可以恢复有用的 Python 对象。在第十二章，“传输和共享对象”，以及第十三章，“配置文件和持久性”中，我们将重新讨论这些主题，因为我们使用 RESTful web 服务与序列化对象以及可编辑的序列化对象用于配置文件。

## 理解持久性、类、状态和表示形式

主要地，我们的 Python 对象存在于易失性计算机内存中。它们只能存在于 Python 进程运行的时间。它们甚至可能活不了那么久；它们可能只能活到它们在命名空间中有引用的时间。如果我们想要一个超出 Python 进程或命名空间寿命的对象，我们需要使其持久化。

大多数操作系统以文件系统的形式提供持久存储。这通常包括磁盘驱动器、闪存驱动器或其他形式的非易失性存储。这似乎只是将字节从内存传输到磁盘文件的问题。

复杂性的原因在于我们的内存中的 Python 对象引用其他对象。一个对象引用它的类。类引用它的元类和任何基类。对象可能是一个容器，并引用其他对象。对象的内存版本是一系列引用和关系。由于内存位置不固定，尝试简单地转储和恢复内存字节而不将地址重写为某种位置无关的键将会破坏这些关系。

引用网络中的许多对象在很大程度上是静态的——例如类定义变化非常缓慢，与变量相比。理想情况下，类定义根本不会改变。但是，我们可能有类级实例变量。更重要的是，我们需要升级我们的应用软件，改变类定义，从而改变对象特性。我们将这称为**模式迁移问题**，管理数据模式（或类）的变化。

Python 为对象的实例变量和类的其他属性之间给出了正式的区别。我们的设计决策利用了这一区别。我们定义对象的实例变量来正确显示对象的动态状态。我们使用类级属性来存储该类的对象将共享的信息。如果我们只能持久化对象的动态状态——与类和类定义的引用网络分开——那将是一种可行的序列化和持久化解决方案。

实际上，我们不必做任何事情来持久化我们的类定义；我们已经有一个完全独立且非常简单的方法来做到这一点。类定义主要存在于源代码中。易失性内存中的类定义是从源代码（或源代码的字节码版本）中每次需要时重新构建的。如果我们需要交换类定义，我们交换 Python 模块或包。

### 常见的 Python 术语

Python 术语往往侧重于*转储*和*加载*这两个词。我们将要使用的大多数各种类都将定义以下方法：

+   `dump(object, file)`: 这将把给定的对象转储到给定的文件中

+   `dumps(object)`: 这将转储一个对象，并返回一个字符串表示

+   `load(file)`: 这将从给定的文件加载一个对象，并返回构造的对象

+   `loads(string)`: 这将从一个字符串表示中加载一个对象，并返回构造的对象

没有标准；方法名并不是由任何正式的 ABC 继承或混合类定义*保证*的。然而，它们被广泛使用。通常，用于转储或加载的文件可以是任何*类似文件*的对象。加载需要一些方法，如`read()`和`readline()`，但我们不需要更多。因此，我们可以使用`io.StringIO`对象以及`urllib.request`对象作为加载的来源。同样，转储对数据源的要求很少。我们将在下一节中深入探讨这些文件对象的考虑。

## 文件系统和网络考虑因素

由于操作系统文件系统（和网络）以字节为单位工作，我们需要将对象的实例变量的值表示为序列化的字节流。通常，我们会使用两步转换为字节；我们将对象的状态表示为一个字符串，并依赖于 Python 字符串提供标准编码的字节。Python 内置的将字符串编码为字节的功能很好地解决了问题的这一部分。

当我们查看操作系统文件系统时，我们会看到两类广泛的设备：块模式设备和字符模式设备。块模式设备也可以称为*可寻址*，因为操作系统支持可以以任意顺序访问文件中的任何字节的寻址操作。字符模式设备不可寻址；它们是以串行方式传输字节的接口。寻址将涉及向后移动时间。

字符和块模式之间的这种区别可能会影响我们如何表示复杂对象或对象集合的状态。本章中我们将要讨论的序列化重点是最简单的常见特性集：有序的字节流；这些格式不使用可寻址设备；它们将将字节流保存到字符模式或块模式文件中。

然而，在第十章和第十一章中我们将要讨论的格式，*通过 Shelve 存储和检索对象*和*通过 SQLite 存储和检索对象*，将需要块模式存储以便编码更多的对象，而不是可能适合内存的对象。`shelve`模块和`SQLite`数据库广泛使用可寻址文件。

一个小的令人困惑的因素是操作系统将块和字符模式设备统一到一个单一的文件系统隐喻中的方式。Python 标准库的一些部分实现了块和字符设备之间的最低公共特性集。当我们使用 Python 的 `urllib.request` 时，我们可以访问网络资源，以及本地文件的数据。当我们打开一个本地文件时，这个模块必须对一个本来是可寻址的文件施加有限的字符模式接口。

## 定义支持持久性的类

在我们可以处理持久性之前，我们需要一些我们想要保存的对象。与持久性相关的有几个设计考虑，所以我们将从一些简单的类定义开始。我们将看一个简单的微博和该博客上的帖子。这里是 `Post` 的一个类定义：

```py
import datetime
class Post:
    def __init__( self, date, title, rst_text, tags ):
        self.date= date
        self.title= title
        self.rst_text= rst_text
        self.tags= tags
    def as_dict( self ):
        return dict(
            date= str(self.date),
            title= self.title,
            underline= "-"*len(self.title),
            rst_text= self.rst_text,
            tag_text= " ".join(self.tags),
        )
```

实例变量是每个微博帖子的属性：日期、标题、一些文本和一些标签。我们的属性名称为我们提供了一个提示，即文本应该是 RST 标记，尽管这对于数据模型的其余部分来说并不重要。

为了支持简单的替换到模板中，`as_dict()` 方法返回一个值的字典，这些值已经转换为字符串格式。我们稍后会看一下使用 `string.Template` 进行模板处理。

此外，我们添加了一些值来帮助创建 RST 输出。`tag_text` 属性是标签值元组的扁平文本版本。`underline` 属性生成一个与标题字符串长度相匹配的下划线字符串；这有助于 RST 格式化工作得很好。我们还将创建一个博客作为帖子的集合。我们将通过包括标题的附加属性使这个集合不仅仅是一个简单的列表。我们有三种选择用于集合设计：包装、扩展或发明一个新的类。我们将通过提供这个警告来避免一些混淆：如果你打算使它持久化，不要扩展一个 `list`。

### 提示

**扩展可迭代对象可能会令人困惑**

当我们扩展一个序列时，可能会混淆一些内置的序列化算法。内置算法可能会绕过我们在序列的子类中放入的扩展特性。包装序列通常比扩展序列更好。

这迫使我们考虑包装或发明。这是一个简单的序列，为什么要发明新的东西呢？包装是我们将强调的设计策略。这里有一系列微博帖子。我们已经包装了一个列表，因为扩展列表并不总是有效的：

```py
from collections import defaultdict
class Blog:
    def __init__( self, title, posts=None ):
        self.title= title
        self.entries= posts if posts is not None else []
    def append( self, post ):
        self.entries.append(post)
    def by_tag(self):
        tag_index= defaultdict(list)
        for post in self.entries:
            for tag in post.tags:
                tag_index[tag].append( post.as_dict() )
        return tag_index
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= "="*len(self.title),
            entries= [p.as_dict() for p in self.entries],
        )
```

除了包装列表，我们还包括了一个微博的标题属性。初始化程序使用了一种常见的技术，以避免提供可变对象作为默认值。我们为 `posts` 提供了 `None` 作为默认值。如果 `posts` 是 `None`，我们使用一个新创建的空列表 `[]`。否则，我们使用给定的 `posts` 值。

此外，我们定义了一个按标签索引帖子的方法。在生成的 `defaultdict` 中，每个键都是一个标签的文本。每个值都是共享给定标签的帖子的列表。

为了简化使用 `string.Template`，我们添加了另一个 `as_dict()` 方法，将整个博客简化为一个简单的字符串和字典的字典。这里的想法是只产生具有简单字符串表示的内置类型。接下来我们将展示模板渲染过程。这里是一些示例数据：

```py
travel = Blog( "Travel" )
travel.append(
    Post( date=datetime.datetime(2013,11,14,17,25),
        title="Hard Aground",
        rst_text="""Some embarrassing revelation. Including ☹ and ⎕""",
        tags=("#RedRanger", "#Whitby42", "#ICW"),
        )
)
travel.append(
    Post( date=datetime.datetime(2013,11,18,15,30),
        title="Anchor Follies",
        rst_text="""Some witty epigram. Including < & > characters.""",,
        tags=("#RedRanger", "#Whitby42", "#Mistakes"),
        )
)
```

我们已将 `Blog` 和 `Post` 序列化为 Python 代码。这并不是一个完全糟糕的表示博客的方式。有一些用例中，Python 代码是对象的一个完全合适的表示。在第十三章 *配置文件和持久性* 中，我们将更仔细地看一下简单地使用 Python 编码数据。

### 渲染博客和帖子

为了完整起见，这里有一种将博客呈现为 RST 的方法。从这个输出文件中，docutils 的`rst2html.py`工具可以将 RST 输出转换为最终的 HTML 文件。这样我们就不必深入研究 HTML 和 CSS 了。此外，我们将使用 RST 来编写第十八章中的文档，*质量和文档*有关 docutils 的更多信息，请参见*一些准备工作*。

我们可以使用`string.Template`类来做到这一点。然而，这很笨拙和复杂。有许多附加的模板工具可以在模板本身内执行更复杂的替换，包括循环和条件处理。这里有一些替代方案：[`wiki.python.org/moin/Templating`](https://wiki.python.org/moin/Templating)。我们将向您展示一个使用 Jinja2 模板工具的示例。请参阅[`pypi.python.org/pypi/Jinja2`](https://pypi.python.org/pypi/Jinja2)。这是一个使用模板在 RST 中呈现这些数据的脚本：

```py
from jinja2 import Template
blog_template= Template( """
{{title}}
{{underline}}

{% for e in entries %}
{{e.title}}
{{e.underline}}

{{e.rst_text}}

:date: {{e.date}}

:tags: {{e.tag_text}}
{% endfor %}

Tag Index
=========
{% for t in tags %}

*   {{t}}
    {% for post in tags[t] %}

    -   `{{post.title}}`_
    {% endfor %}
{% endfor %}
""")
print( blog_template.render( tags=travel.by_tag(), **travel.as_dict() ) )
```

`{{title}}`和`{{underline}}`元素（以及所有类似的元素）向我们展示了如何将值替换为模板文本。使用`render()`方法调用`**travel.as_dict()`，以确保属性（如`title`和`underline`）将成为关键字参数。

`{%for%}`和`{%endfor%}`构造向我们展示了 Jinja 如何遍历`Blog`中`Post`条目的序列。在此循环的主体中，变量`e`将是从每个`Post`创建的字典。我们从字典中为每个帖子挑选了特定的键：`{{e.title}}`、`{{e.rst_text}}`等。

我们还遍历了`Blog`的`tags`集合。这是一个字典，其中包含每个标签的键和该标签的帖子。循环将访问每个键，分配给`t`。循环的主体将遍历字典值中的帖子，即`tags[t]`。

``{{post.title}}`_`构造是一个 RST 标记，它生成一个链接到文档中具有该标题的部分。这种非常简单的标记是 RST 的优势之一。我们已经将博客标题用作索引中的部分和链接。这意味着标题*必须*是唯一的，否则我们将获得 RST 呈现错误。

因为这个模板遍历给定的博客，它将以一种平稳的动作呈现所有的帖子。内置于 Python 的`string.Template`不能进行迭代。这使得呈现`Blog`的所有`Posts`变得更加复杂。

## 使用 JSON 进行转储和加载

JSON 是什么？来自[www.json.org](http://www.json.org)网页的一节指出：

> JSON（JavaScript 对象表示）是一种轻量级的数据交换格式。人类很容易阅读和书写。机器很容易解析和生成。它基于 JavaScript 编程语言的一个子集，标准 ECMA-262 第 3 版-1999 年 12 月。JSON 是一种完全与语言无关的文本格式，但使用了熟悉 C 系列语言的程序员的约定，包括 C、C++、C#、Java、JavaScript、Perl、Python 等。这些特性使 JSON 成为一种理想的数据交换语言。

这种格式被广泛用于各种语言和框架。诸如 CouchDB 之类的数据库将其数据表示为 JSON 对象，简化了应用程序之间的数据传输。JSON 文档具有类似 Python `list`和`dict`文字值的优势。它们易于阅读和手动编辑。

`json`模块与内置的 Python 类型一起使用。它不适用于我们定义的类，直到我们采取一些额外的步骤。接下来我们将看看这些扩展技术。对于以下 Python 类型，有一个映射到 JSON 使用的 JavaScript 类型：

| Python 类型 | JSON |
| --- | --- |
| `dict` | `object` |
| `list, tuple` | `array` |
| `str` | `string` |
| `int, float` | `number` |
| `True` | `true` |
| `False` | `false` |
| `None` | `null` |

其他类型不受支持，必须通过我们可以插入到 dump 和 load 函数中的扩展函数来强制转换为这些类型之一。我们可以通过将我们的微博对象转换为更简单的 Python`lists`和`dicts`来探索这些内置类型。当我们查看我们的`Post`和`Blog`类定义时，我们已经定义了`as_dict()`方法，将我们的自定义类对象减少为内置的 Python 对象。以下是生成我们博客数据的 JSON 版本所需的代码：

```py
import json
print( json.dumps(travel.as_dict(), indent=4) )
```

以下是输出：

```py
{
    "entries": [
        {
            "title": "Hard Aground",
            "underline": "------------",
            "tag_text": "#RedRanger #Whitby42 #ICW",
            "rst_text": "Some embarrassing revelation. Including \u2639 and \u2693",
            "date": "2013-11-14 17:25:00"
        },
        {
            "title": "Anchor Follies",
            "underline": "--------------",
            "tag_text": "#RedRanger #Whitby42 #Mistakes",
            "rst_text": "Some witty epigram. Including < & > characters.",
            "date": "2013-11-18 15:30:00"
        }
    ],
    "title": "Travel"
}
```

前面的输出向我们展示了各种对象是如何从 Python 转换为 JSON 表示的。这种方法的优雅之处在于我们的 Python 对象已经被写入了一个标准化的表示法。我们可以与其他应用程序共享它们。我们可以将它们写入磁盘文件并保存它们。JSON 表示的一些不愉快特性有：

+   我们不得不将我们的 Python 对象重写为字典。更好的方法是以更简单的方式转换 Python 对象，而不需要显式创建额外的字典。

+   当我们加载这个 JSON 表示时，我们无法轻松地重建我们原来的`Blog`和`Post`对象。当我们使用`json.load()`时，我们得到的不是`Blog`或`Post`对象，而是`dict`和列表对象。我们需要提供一些额外的提示来重建`Blog`和`Post`对象。

+   对象的`__dict__`中有一些值我们不想持久化，比如`Post`的下划线文本。

我们需要比内置的 JSON 编码更复杂的东西。

### 在我们的类中支持 JSON

为了正确支持 JSON，我们需要通知 JSON 编码器和解码器关于我们的类。为了将我们的对象编码为 JSON，我们需要提供一个函数，将我们的对象减少为 Python 原始类型。这被称为*默认*函数；它为未知类的对象提供默认编码。

为了从 JSON 中解码我们的对象，我们需要提供一个函数，将 Python 原始类型的字典转换回适当类的对象。这被称为*对象钩子*函数；它用于将`dict`转换为自定义类的对象。

`json`模块文档建议我们可能希望使用类提示。Python 文档包括对 JSON-RPC 版本 1 规范的引用。参见[`json-rpc.org/wiki/specification`](http://json-rpc.org/wiki/specification)。这个建议是将自定义类的实例编码为以下的字典：

```py
{"__jsonclass__": ["class name", [param1,...]] }
```

与`"__jsonclass__"`键关联的建议值是一个包含两个项目的列表：类名和创建该类实例所需的参数列表。规范允许更多的特性，但它们与 Python 无关。

从 JSON 字典中解码对象时，我们可以查找`"__jsonclass__"`键作为提示，表明我们需要构建一个类，而不是一个内置的 Python 对象。类名可以映射到一个类对象，并且参数序列可以用来构建实例。

当我们查看其他复杂的 JSON 编码器（比如 Django Web 框架自带的编码器）时，我们可以看到它们提供了更复杂的自定义类编码。它们包括类、数据库主键和属性值。我们将看看如何实现自定义编码和解码。规则被表示为简单的函数，这些函数被插入到 JSON 编码和解码函数中。

### 自定义 JSON 编码

对于类提示，我们将提供三个信息。我们将包括一个`__class__`键，命名目标类。`__args__`键将提供一个位置参数值的序列。`__kw__`键将提供一个关键字参数值的字典。这将涵盖`__init__()`的所有选项。以下是遵循这种设计的编码器：

```py
def blog_encode( object ):
    if isinstance(object, datetime.datetime):
        return dict(
            __class__= "datetime.datetime",
            __args__= [],
            __kw__= dict(
                year= object.year,
                month= object.month,
                day= object.day,
                hour= object.hour,
                minute= object.minute,
                second= object.second,
            )
        )
    elif isinstance(object, Post):
        return dict(
            __class__= "Post",
            __args__= [],
            __kw__= dict(
                date= object.date,
                title= object.title,
                rst_text= object.rst_text,
                tags= object.tags,
            )
        )
    elif isinstance(object, Blog):
        return dict(
            __class__= "Blog",
            __args__= [
                object.title,
                object.entries,
            ],
            __kw__= {}
        )
    else:
        return json.JSONEncoder.default(o)
```

这个函数展示了三个类的两种不同风格的对象编码：

+   我们将一个`datetime.datetime`对象编码为一个单独字段的字典

+   我们还将一个`Post`实例编码为一个单独字段的字典

+   我们将一个`Blog`实例编码为标题和文章条目的序列

如果我们无法处理这个类，我们会调用现有编码器的默认编码。这将处理内置类。我们可以使用这个函数进行编码，如下所示：

```py
text= json.dumps(travel, indent=4, default=blog_encode)
```

我们将我们的函数`blog_encode()`作为`json.dumps()`函数的`default=`关键字参数提供。这个函数被 JSON 编码器用来确定对象的编码。这个编码器导致的 JSON 对象看起来像下面的代码：

```py
{
    "__args__": [
        "Travel",
        [
            {
                "__args__": [],
                "__kw__": {
                    "tags": [
                        "#RedRanger",
                        "#Whitby42",
                        "#ICW"
                    ],
                    "rst_text": "Some embarrassing revelation. Including \u2639 and \u2693",
                    "date": {
                        "__args__": [],
                        "__kw__": {
                            "minute": 25,
                            "hour": 17,
                            "day": 14,
                            "month": 11,
                            "year": 2013,
                            "second": 0
                        },
                        "__class__": "datetime.datetime"
                    },
                    "title": "Hard Aground"
                },
                "__class__": "Post"
            },
.
.
.
    "__kw__": {},
    "__class__": "Blog"
}
```

我们删除了第二个博客条目，因为输出太长了。现在，`Blog`对象用一个提供类和两个位置参数值的`dict`包装起来。同样，`Post`和`datetime`对象也用类名和关键字参数值包装起来。

### 自定义 JSON 解码

为了解码一个 JSON 对象，我们需要在 JSON 解析的结构内工作。我们定制的类定义的对象被编码为简单的`dicts`。这意味着每个被 JSON 解码的`dict` *可能* 是我们定制的类之一。或者，`dict`可能只是一个`dict`。

JSON 解码器的“对象钩子”是一个函数，它会为每个`dict`调用，以查看它是否表示一个定制对象。如果`dict`不被`hook`函数识别，那么它只是一个字典，应该原样返回。这是我们的对象钩子函数：

```py
def blog_decode( some_dict ):
    if set(some_dict.keys()) == set( ["__class__", "__args__", "__kw__"] ):
        class_= eval(some_dict['__class__'])
        return class_( *some_dict['__args__'], **some_dict['__kw__'] )
    else:
        return some_dict
```

每次调用此函数时，它都会检查定义对象编码的键。如果存在这三个键，那么将使用给定的参数和关键字调用该函数。我们可以使用这个对象钩子来解析 JSON 对象，如下所示：

```py
blog_data= json.loads(text, object_hook= blog_decode)
```

这将解码一块以 JSON 表示的文本，使用我们的`blog_decode()`函数将`dict`转换为正确的`Blog`和`Post`对象。

### 安全和 eval()问题

一些程序员会反对在我们的`blog_decode()`函数中使用`eval()`函数，声称这是一个普遍存在的安全问题。可笑的是声称`eval()`是一个普遍存在的问题。如果恶意代码被写入 JSON 对象的表示中，那么它就是一个*潜在*的安全问题，这是一个本地的 EGP 可以访问 Python 源代码。为什么要去微调 JSON 文件？为什么不直接编辑 Python 源代码呢？

作为一个实际问题，我们必须考虑通过互联网传输 JSON 文档；这是一个实际的安全问题。然而，这并不是一般情况下对`eval()`的控诉。

必须考虑一种情况，即一个不可信的文档被**中间人**攻击篡改。在这种情况下，一个 JSON 文档在通过包括一个不可信的服务器作为代理的网络接口时被篡改。SSL 通常是防止这个问题的首选方法。

如果有必要，我们可以用一个从名称到类的映射字典来替换`eval()`。我们可以将`eval(some_dict['__class__'])`改为`{"Post":Post, "Blog":Blog, "datetime.datetime":datetime.datetime`：

```py
}[some_dict['__class__']]

```

这将防止在通过非 SSL 编码连接传递 JSON 文档时出现问题。这也导致了一个维护要求，即每当应用程序设计发生变化时，都需要微调这个映射。

### 重构编码函数

理想情况下，我们希望重构我们的编码函数，专注于每个定义类的正确编码的责任。我们不想把所有的编码规则堆积到一个单独的函数中。

要使用诸如`datetime`之类的库类来做到这一点，我们需要为我们的应用程序扩展`datetime.datetime`。如果我们这样做了，我们需要确保我们的应用程序使用我们扩展的`datetime`而不是`datetime`库。这可能会变得有点头疼，以避免使用内置的`datetime`类。通常，我们必须在我们定制的类和库类之间取得平衡。以下是将创建 JSON 可编码类定义的两个类扩展。我们可以向`Blog`添加一个属性：

```py
    @property
    def _json( self ):
        return dict( __class__= self.__class__.__name__,
            __kw__= {},
            __args__= [ self.title, self.entries ]
        )
```

这个属性将提供初始化参数，可供我们的解码函数使用。我们可以将这两个属性添加到`Post`中：

```py
    @property
    def _json( self ):
        return dict(
            __class__= self.__class__.__name__,
            __kw__= dict(
                date= self.date,
                title= self.title,
                rst_text= self.rst_text,
                tags= self.tags,
            ),
            __args__= []
        )
```

与`Blog`一样，这个属性将提供初始化参数，可供我们的解码函数使用。我们可以修改编码器，使其变得更简单一些。以下是修订后的版本：

```py
def blog_encode_2( object ):
    if isinstance(object, datetime.datetime):
        return dict(
            __class__= "datetime.datetime",
            __args__= [],
            __kw__= dict(
                year= object.year,
                month= object.month,
                day= object.day,
                hour= object.hour,
                minute= object.minute,
                second= object.second,
            )
        )
    else:
        try:
            encoding= object._json()
        except AttributeError:
            encoding= json.JSONEncoder.default(o)
        return encoding
```

我们仍然受到使用库`datetime`模块的选择的限制。在这个例子中，我们选择不引入子类，而是将编码处理为特殊情况。

### 标准化日期字符串

我们对日期的格式化没有使用广泛使用的 ISO 标准文本日期格式。为了与其他语言更兼容，我们应该正确地对`datetime`对象进行标准字符串编码和解析标准字符串。

由于我们已经将日期视为特殊情况，这似乎是对该特殊情况处理的合理扩展。这可以在不太改变我们的编码和解码的情况下完成。考虑对编码进行的这个小改变：

```py
    if isinstance(object, datetime.datetime):
        fmt= "%Y-%m-%dT%H:%M:%S"
        return dict(
            __class__= "datetime.datetime.strptime",
            __args__= [ object.strftime(fmt), fmt ],
            __kw__= {}
        )
```

编码输出命名了静态方法`datetime.datetime.strptime()`，并提供了编码的参数`datetime`以及要用于解码的格式。现在，帖子的输出看起来像以下代码片段：

```py
            {
                "__args__": [],
                "__class__": "Post_J",
                "__kw__": {
                    "title": "Anchor Follies",
                    "tags": [
                        "#RedRanger",
                        "#Whitby42",
                        "#Mistakes"
                    ],
                    "rst_text": "Some witty epigram.",
                    "date": {
                        "__args__": [
                            "2013-11-18T15:30:00",
                            "%Y-%m-%dT%H:%M:%S"
                        ],
                        "__class__": "datetime.datetime.strptime",
                        "__kw__": {}
                    }
                }
            }
```

这向我们表明，现在我们有一个 ISO 格式的日期，而不是单独的字段。我们还摆脱了使用类名创建对象的方式。`__class__`值扩展为类名或静态方法名。

### 将 JSON 写入文件

当我们写 JSON 文件时，我们通常会这样做：

```py
with open("temp.json", "w", encoding="UTF-8") as target:
    json.dump( travel3, target, separators=(',', ':'), default=blog_j2_encode )
```

我们使用所需的编码打开文件。我们将文件对象提供给`json.dump()`方法。当我们读取 JSON 文件时，我们将使用类似的技术：

```py
with open("some_source.json", "r", encoding="UTF-8") as source:objects= json.load( source, object_hook= blog_decode)
```

这个想法是将 JSON 表示作为文本与生成文件上的字节转换分开。JSON 中有一些可用的格式选项。我们展示了缩进四个空格，因为这似乎产生了漂亮的 JSON。作为替代，我们可以通过留下缩进选项使输出更紧凑。通过使分隔符更简洁，我们甚至可以使其更加紧凑。以下是在`temp.json`中创建的输出：

```py
{"__class__":"Blog_J","__args__":["Travel",[{"__class__":"Post_J","__args__":[],"__kw__":{"rst_text":"Some embarrassing revelation.","tags":["#RedRanger","#Whitby42","#ICW"],"title":"Hard Aground","date":{"__class__":"datetime.datetime.strptime","__args__":["2013-11-14T17:25:00","%Y-%m-%dT%H:%M:%S"],"__kw__":{}}}},{"__class__":"Post_J","__args__":[],"__kw__":{"rst_text":"Some witty epigram.","tags":["#RedRanger","#Whitby42","#Mistakes"],"title":"Anchor Follies","date":{"__class__":"datetime.datetime.strptime","__args__":["2013-11-18T15:30:00","%Y-%m-%dT%H:%M:%S"],"__kw__":{}}}}]],"__kw__":{}}
```

## 使用 YAML 进行转储和加载

[yaml.org](http://yaml.org)网页指出：

> YAML™（与“骆驼”押韵）是一种人性化的、跨语言的、基于 Unicode 的数据序列化语言，旨在围绕敏捷编程语言的常见本机数据类型设计。

`json`模块的 Python 标准库文档指出：

> JSON 是 YAML 1.2 的子集。此模块的默认设置（特别是默认分隔符值）生成的 JSON 也是 YAML 1.0 和 1.1 的子集。因此，该模块也可以用作 YAML 序列化器。

从技术上讲，我们可以使用`json`模块准备 YAML 数据。但是，`json`模块无法用于反序列化更复杂的 YAML 数据。YAML 的两个好处。首先，它是一种更复杂的表示法，允许我们对我们的对象编码更多的细节。其次，PyYAML 实现与 Python 有深度集成，使我们能够非常简单地创建 Python 对象的 YAML 编码。YAML 的缺点是它没有像 JSON 那样被广泛使用。我们需要下载和安装一个 YAML 模块。可以在[`pyyaml.org/wiki/PyYAML`](http://pyyaml.org/wiki/PyYAML)找到一个好的模块。安装了包之后，我们可以以 YAML 表示法转储我们的对象：

```py
import yaml
text= yaml.dump(travel2)
print( text )
```

这是我们微博的 YAML 编码：

```py
!!python/object:__main__.Blog
entries:
- !!python/object:__main__.Post
  date: 2013-11-14 17:25:00
  rst_text: Some embarrassing revelation. Including ☹ and ⎕
  tags: !!python/tuple ['#RedRanger', '#Whitby42', '#ICW']
  title: Hard Aground
- !!python/object:__main__.Post
  date: 2013-11-18 15:30:00
  rst_text: Some witty epigram. Including < & > characters.
  tags: !!python/tuple ['#RedRanger', '#Whitby42', '#Mistakes']
  title: Anchor Follies
```

输出相对简洁，但也非常完整。此外，我们可以轻松编辑 YAML 文件以进行更新。类名使用 YAML `!!`标记进行编码。YAML 包含 11 个标准标记。`yaml`模块包括十几个特定于 Python 的标记，以及五个*复杂*的 Python 标记。

Python 类名由定义模块限定。在我们的情况下，该模块碰巧是一个简单的脚本，因此类名是`__main__.Blog`和`__main__.Post`。如果我们从另一个模块导入这些类，类名将反映定义类的模块。

列表中的项目以块序列形式显示。每个项目以`-`序列开头；其余项目缩进两个空格。当`list`或`tuple`足够小，它可以流到一行。如果它变得更长，它将换行到多行。要从 YAML 文档加载 Python 对象，我们可以使用以下代码：

```py
copy= yaml.load(text)
```

这将使用标记信息来定位类定义，并将在 YAML 文档中找到的值提供给类构造函数。我们的微博对象将被完全重建。

### 在文件上格式化 YAML 数据

当我们写 YAML 文件时，我们通常会做这样的事情：

```py
with open("some_destination.yaml", "w", encoding="UTF-8") as target:
    yaml.dump( some_collection, target )
```

我们以所需的编码打开文件。我们将文件对象提供给`yaml.dump()`方法；输出将写入那里。当我们读取 YAML 文件时，我们将使用类似的技术：

```py
with open("some_source.yaml", "r", encoding="UTF-8") as source:objects= yaml.load( source )
```

将 YAML 表示作为文本与结果文件上的字节转换分开的想法。我们有几种格式选项来创建更漂亮的 YAML 表示我们的数据。以下表格显示了一些选项：

| `explicit_start` 如果为`true`，在每个对象之前写入一个`---`标记。 |
| --- |
| `explicit_end` 如果为`true`，在每个对象之后写入一个`...`标记。如果我们将一系列 YAML 文档转储到单个文件并且需要知道一个结束和下一个开始时，我们可能会使用这个或`explicit_start`。 |
| `version` 给定一对整数(x,y)，在开头写入`%YAML x.y`指令。这应该是`version=(1,2)`。 |
| `tags` 给定一个映射，它会发出一个带有不同标记缩写的 YAML `%TAG`指令。 |
| `canonical` 如果为`true`，则在每个数据片段上包括一个标记。如果为 false，则假定一些标记。 |
| `indent` 如果设置为一个数字，改变用于块的缩进。 |
| `width` 如果设置为一个数字，改变长项换行到多个缩进行的宽度。 |
| `allow_unicode` 如果设置为`true`，允许完全使用 Unicode 而无需转义。否则，ASCII 子集之外的字符将被应用转义。 |
| `line_break` 使用不同的换行符；默认为换行符。 |

在这些选项中，`explicit_end`和`allow_unicode`可能是最有用的。

### 扩展 YAML 表示

有时，我们的类之一具有整洁的表示，比默认的 YAML 转储属性值更好。例如，我们的 Blackjack `Card`类定义的默认 YAML 将包括一些我们不需要保留的派生值。

`yaml`模块包括为类定义添加**representer**和**constructor**的规定。representer 用于创建 YAML 表示，包括标记和值。构造函数用于从给定值构建 Python 对象。这是另一个`Card`类层次结构：

```py
class Card:
    def __init__( self, rank, suit, hard=None, soft=None ):
        self.rank= rank
        self.suit= suit
        self.hard= hard or int(rank)
        self.soft= soft or int(rank)
    def __str__( self ):
        return "{0.rank!s}{0.suit!s}".format(self)

class AceCard( Card ):
    def __init__( self, rank, suit ):
        super().__init__( rank, suit, 1, 11 )

class FaceCard( Card ):
    def __init__( self, rank, suit ):
        super().__init__( rank, suit, 10, 10 )
```

我们使用了数字卡的超类，并为 A 和面值卡定义了两个子类。在先前的示例中，我们广泛使用了工厂函数来简化构建。工厂处理了从 1 到`AceCar`类的等级的映射，以及从 11、12 和 13 等级到`FaceCard`类的映射。这是必不可少的，这样我们就可以轻松地使用简单的`range(1,14)`来构建一副牌。

从 YAML 加载时，类将通过 YAML`!!`标记完全拼写出来。唯一缺少的信息将是与卡片的每个子类关联的硬值和软值。硬点和软点有三种相对简单的情况，可以通过可选的初始化参数来处理。当我们将这些对象转储到 YAML 格式时，它看起来是这样的：

```py
- !!python/object:__main__.AceCard {hard: 1, rank: A, soft: 11, suit: ♣}
- !!python/object:__main__.Card {hard: 2, rank: '2', soft: 2, suit: ♥}
- !!python/object:__main__.FaceCard {hard: 10, rank: K, soft: 10, suit: ♦}
```

这些是正确的，但对于像扑克牌这样简单的东西来说可能有点啰嗦。我们可以扩展`yaml`模块，以便为这些简单对象生成更小、更专注的输出。我们将为`Card`子类定义表示和构造函数。以下是三个函数和注册：

```py
def card_representer(dumper, card):
    return dumper.represent_scalar('!Card',
    "{0.rank!s}{0.suit!s}".format(card) )
def acecard_representer(dumper, card):
    return dumper.represent_scalar('!AceCard',
    "{0.rank!s}{0.suit!s}".format(card) )
def facecard_representer(dumper, card):
    return dumper.represent_scalar('!FaceCard',
    "{0.rank!s}{0.suit!s}".format(card) )

yaml.add_representer(Card, card_representer)
yaml.add_representer(AceCard, acecard_representer)
yaml.add_representer(FaceCard, facecard_representer)
```

我们已将每个`Card`实例表示为一个简短的字符串。YAML 包括一个标记，显示应从字符串构建哪个类。所有三个类使用相同的格式字符串。这恰好与`__str__()`方法匹配，从而导致潜在的优化。

我们需要解决的另一个问题是从解析的 YAML 文档构造`Card`实例。为此，我们需要构造函数。以下是三个构造函数和注册：

```py
def card_constructor(loader, node):
    value = loader.construct_scalar(node)
    rank, suit= value[:-1], value[-1]
    return Card( rank, suit )

def acecard_constructor(loader, node):
    value = loader.construct_scalar(node)
    rank, suit= value[:-1], value[-1]
    return AceCard( rank, suit )

def facecard_constructor(loader, node):
    value = loader.construct_scalar(node)
    rank, suit= value[:-1], value[-1]
    return FaceCard( rank, suit )

yaml.add_constructor('!Card', card_constructor)
yaml.add_constructor('!AceCard', acecard_constructor)
yaml.add_constructor('!FaceCard', facecard_constructor)
```

当解析标量值时，标记将用于定位特定的构造函数。然后构造函数可以分解字符串并构建`Card`实例的适当子类。这是一个快速演示，演示了每个类的一张卡片：

```py
deck = [ AceCard('A','♣',1,11), Card('2','♥',2,2), FaceCard('K','♦',10,10) ]
text= yaml.dump( deck, allow_unicode=True )
```

以下是输出：

```py
[!AceCard 'A♣', !Card '2♥', !FaceCard 'K♦']
```

这给我们提供了可以用来重建 Python 对象的卡片的简短而优雅的 YAML 表示。

我们可以使用以下简单语句重新构建我们的 3 张牌组：

```py
cards= yaml.load( text )
```

这将解析表示，使用构造函数，并构建预期的对象。因为构造函数确保适当的初始化完成，硬值和软值的内部属性将被正确重建。

### 安全和安全加载

原则上，YAML 可以构建任何类型的对象。这允许对通过互联网传输 YAML 文件的应用程序进行攻击，而不需要适当的 SSL 控制。

YAML 模块提供了一个`safe_load()`方法，拒绝执行任意 Python 代码作为构建对象的一部分。这严重限制了可以加载的内容。对于不安全的数据交换，我们可以使用`yaml.safe_load()`来创建仅包含内置类型的 Python`dict`和`list`对象。然后我们可以从`dict`和`list`实例构建我们的应用程序类。这与我们使用 JSON 或 CSV 交换必须用于创建正确对象的`dict`的方式有些相似。

更好的方法是为我们自己的对象使用`yaml.YAMLObject`混合类。我们使用这个类来设置一些类级别的属性，为`yaml`提供提示，并确保对象的安全构建。以下是我们如何定义用于安全传输的超类：

```py
class Card2( yaml.YAMLObject ):
    yaml_tag = '!Card2'
    yaml_loader= yaml.SafeLoader
```

这两个属性将警告`yaml`，这些对象可以安全加载，而不会执行任意和意外的 Python 代码。`Card2`的每个子类只需设置将要使用的唯一 YAML 标记：

```py
class AceCard2( Card2 ):
    yaml_tag = '!AceCard2'
```

我们添加了一个属性，警告`yaml`，这些对象仅使用此类定义。这些对象可以安全加载；它们不执行任意不可信代码。

通过对类定义进行这些修改，我们现在可以在 YAML 流上使用`yaml.safe_load()`，而不必担心文档在不安全的互联网连接上插入恶意代码。对我们自己的对象使用`yaml.YAMLObject`混合类以及设置`yaml_tag`属性具有几个优点。它导致文件稍微更紧凑。它还导致更美观的 YAML 文件——长而通用的`!!python/object:__main__.AceCard`标记被更短的`!AceCard2`标记替换。

## 使用 pickle 进行转储和加载

`pickle`模块是 Python 的本机格式，用于使对象持久化。

Python 标准库对`pickle`的描述如下：

> pickle 模块可以将复杂对象转换为字节流，并且可以将字节流转换为具有相同内部结构的对象。对这些字节流最明显的用途可能是将它们写入文件，但也可以想象将它们发送到网络或存储在数据库中。

`pickle`的重点是 Python，仅限于 Python。这不是诸如 JSON、YAML、CSV 或 XML 之类的数据交换格式，可以与其他语言编写的应用程序一起使用。

`pickle`模块与 Python 紧密集成在各种方式。例如，类的`__reduce__()`和`__reduce_ex__()`方法存在以支持`pickle`处理。

我们可以轻松地将我们的微博 pickle 如下：

```py
import pickle
with open("travel_blog.p","wb") as target:
    pickle.dump( travel, target )
```

将整个`travel`对象导出到给定文件。该文件以原始字节形式写入，因此`open()`函数使用`"wb"`模式。

我们可以通过以下方式轻松恢复一个 picked 对象：

```py
with open("travel_blog.p","rb") as source:
    copy= pickle.load( source )
```

由于 pickled 数据是以字节形式写入的，因此文件必须以`"rb"`模式打开。pickled 对象将正确绑定到适当的类定义。底层的字节流不是为人类消费而设计的。它在某种程度上是可读的，但它不像 YAML 那样设计用于可读性。

### 设计一个可靠的 pickle 处理类

类的`__init__()`方法实际上并不用于取消封存对象。通过使用`__new__()`并将 pickled 值直接设置到对象的`__dict__`中，`__init__()`方法被绕过。当我们的类定义包括`__init__()`中的一些处理时，这一区别很重要。例如，如果`__init__()`打开外部文件，创建 GUI 界面的某个部分，或者对数据库执行某些外部更新，则在取消封存时不会执行这些操作。

如果我们在`__init__()`处理期间计算一个新的实例变量，就没有真正的问题。例如，考虑一个 Blackjack`Hand`对象，在创建`Hand`时计算`Card`实例的总数。普通的`pickle`处理将保留这个计算出的实例变量。在取消封存对象时，不会重新计算它。先前计算出的值将被简单地取消封存。

依赖于`__init__()`期间处理的类必须特别安排以确保此初始处理将正确进行。我们可以做两件事：

+   避免在`__init__()`中进行急切的启动处理。相反，进行一次性的初始化处理。例如，如果有外部文件操作，必须推迟到需要时才执行。

+   定义`__getstate__()`和`__setstate__()`方法，这些方法可以被 pickle 用来保存状态和恢复状态。然后，`__setstate__()`方法可以调用与`__init__()`在普通 Python 代码中执行一次性初始化处理的相同方法。

我们将看一个例子，其中由`__init__()`方法记录为审计目的加载到`Hand`中的初始`Card`实例。以下是在取消封存时无法正常工作的`Hand`版本：

```py
class Hand_x:
    def __init__( self, dealer_card, *cards ):
        self.dealer_card= dealer_card
        self.cards= list(cards)
        **for c in self.cards:
 **audit_log.info( "Initial %s", c )
    def append( self, card ):
        self.cards.append( card )
        **audit_log.info( "Hit %s", card )
    def __str__( self ):
        cards= ", ".join( map(str,self.cards) )
        return "{self.dealer_card} | {cards}".format( self=self, cards=cards )
```

这有两个记录位置：在`__init__()`和`append()`期间。`__init__()`处理在初始对象创建和取消封存以重新创建对象之间不能一致工作。以下是用于查看此问题的日志设置：

```py
import logging,sys
audit_log= logging.getLogger( "audit" )
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
```

此设置创建日志并确保日志级别适合查看审计信息。以下是一个快速脚本，用于构建、pickle 和 unpickle`Hand`：

```py
h = Hand_x( FaceCard('K','♦'), AceCard('A','♣'), Card('9','♥') )
data = pickle.dumps( h )
h2 = pickle.loads( data )
```

当我们执行这个时，我们发现在处理`__init__()`时写入的日志条目在反拾取`Hand`时没有被写入。为了正确地为反拾取编写审计日志，我们可以在这个类中放置延迟日志测试。例如，我们可以扩展`__getattribute__()`以在从这个类请求任何属性时写入初始日志条目。这导致了有状态的日志记录和每次手对象执行操作时执行的`if`语句。一个更好的解决方案是利用`pickle`保存和恢复状态的方式。

```py
class Hand2:
    def __init__( self, dealer_card, *cards ):
        self.dealer_card= dealer_card
        self.cards= list(cards)
        for c in self.cards:
            audit_log.info( "Initial %s", c )
    def append( self, card ):
        self.cards.append( card )
        audit_log.info( "Hit %s", card )
    def __str__( self ):
        cards= ", ".join( map(str,self.cards) )
        return "{self.dealer_card} | {cards}".format( self=self, cards=cards )
    def __getstate__( self ):
        return self.__dict__
    def __setstate__( self, state ):
        self.__dict__.update(state)
        for c in self.cards:
            audit_log.info( "Initial (unpickle) %s", c )
```

`__getstate__()` 方法在拾取时用于收集对象的当前状态。这个方法可以返回任何东西。例如，对于具有内部记忆缓存的对象，缓存可能不会被拾取以节省时间和空间。这个实现使用内部的`__dict__`而没有任何修改。

`__setstate__()` 方法在反拾取时用于重置对象的值。这个版本将状态合并到内部的`__dict__`中，然后写入适当的日志条目。

### 安全和全局问题

在反拾取期间，pickle 流中的全局名称可能导致任意代码的评估。一般来说，全局名称是类名或函数名。然而，可能包括一个函数名是`os`或`subprocess`等模块中的全局名称。这允许对试图通过互联网传输拾取对象的应用程序进行攻击，而没有强大的 SSL 控制。这对于完全本地文件来说并不是问题。

为了防止执行任意代码，我们必须扩展`pickle.Unpickler`类。我们将覆盖`find_class()`方法以替换为更安全的内容。我们必须考虑几个反拾取问题，例如：

+   我们必须防止使用内置的`exec()`和`eval()`函数。

+   我们必须防止使用可能被认为是不安全的模块和包。例如，应该禁止使用`sys`和`os`。

+   我们必须允许使用我们的应用程序模块。

以下是一个施加一些限制的示例：

```py
import builtins
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins":
            if name not in ("exec", "eval"):
                 return getattr(builtins, name)
        elif module == "__main__":
            return globals()[name]
        # elif module in any of our application modules...
        raise pickle.UnpicklingError(
        "global '{module}.{name}' is forbidden".format(module=module, name=name))
```

这个`Unpickler`类的版本将帮助我们避免由篡改的 pickle 流可能引起的大量潜在问题。它允许使用除了`exec()`和`eval()`之外的任何内置函数。它允许仅在`__main__`中定义的类的使用。在所有其他情况下，它会引发异常。

## 使用 CSV 进行转储和加载

`csv`模块将简单的`list`或`dict`实例编码和解码为 CSV 符号。与之前讨论的`json`模块一样，这并不是一个非常完整的持久性解决方案。然而，由于 CSV 文件的广泛采用，通常需要在 Python 对象和 CSV 之间进行转换。

处理 CSV 文件涉及我们的对象和 CSV 结构之间的手动映射。我们需要仔细设计映射，注意 CSV 符号的限制。这可能很困难，因为对象的表达能力与 CSV 文件的表格结构之间存在不匹配。

CSV 文件的每一列的内容—根据定义—都是纯文本。从 CSV 文件加载数据时，我们需要将这些值转换为更有用的类型。这种转换可能会受到电子表格执行意外类型强制转换的影响。例如，我们可能有一个电子表格，其中美国邮政编码已被电子表格应用程序更改为浮点数。当电子表格保存为 CSV 时，邮政编码可能会变成看起来奇怪的数值。

因此，我们可能需要使用转换，比如`('00000'+row['zip'])[-5:]`来恢复前导零。另一种情况是必须使用类似`"{0:05.0f}".format(float(row['zip']))`来恢复前导零。另外，不要忘记文件可能包含 ZIP 和 ZIP+4 邮政编码的混合，这使得这个过程更具挑战性。

为了更复杂地处理 CSV 文件，我们必须意识到它们经常被手动操作，并且由于人为调整，它们经常不兼容。软件在面对现实世界中出现的不规则性时保持灵活是很重要的。

当我们有相对简单的类定义时，我们通常可以将每个实例转换为简单的扁平数据值行。通常情况下，`namedtuple`是 CSV 源文件和 Python 对象之间的良好匹配。反过来，如果我们的应用程序将数据保存在 CSV 符号中，我们可能需要围绕`namedtuples`设计我们的 Python 类。

当我们有容器类时，通常很难确定如何在扁平的 CSV 行中表示结构化容器。这是对象模型和用于 CSV 文件或关系数据库的扁平规范化表结构之间的**阻抗不匹配**。阻抗不匹配没有好的解决方案；它需要仔细设计。我们将从简单的扁平对象开始，向您展示一些 CSV 映射。

### 将简单序列转储到 CSV

理想的映射是`namedtuple`实例和 CSV 文件中的行之间的映射。每一行代表一个不同的`namedtuple`。考虑以下 Python 类：

```py
from collections import namedtuple
GameStat = namedtuple( "GameStat", "player,bet,rounds,final" )
```

我们已经定义了对象为简单的扁平属性序列。数据库架构师称之为**第一范式**。没有重复的组，每个项目都是原子数据。我们可能会从一个看起来像以下代码的模拟中产生这些对象：

```py
def gamestat_iter( player, betting, limit=100 ):
    for sample in range(30):
        b = Blackjack( player(), betting() )
        b.until_broke_or_rounds(limit)
        yield GameStat( player.__name__, betting.__name__, b.rounds, b.betting.stake )
```

这个迭代器将创建具有给定玩家和投注策略的二十一点模拟。它将执行游戏，直到玩家破产或者在 100 个独立的游戏回合中坐在桌子旁。在每个会话结束时，它将产生一个带有玩家策略、投注策略、回合数和最终赌注的`GameStat`对象。这将允许我们为每个玩法或投注策略或组合计算统计数据。以下是我们如何将其写入文件以供以后分析：

```py
import csv
with open("blackjack.stats","w",newline="") as target:
    writer= csv.DictWriter( target, GameStat._fields )
    writer.writeheader()
    for gamestat in gamestat_iter( Player_Strategy_1, Martingale_Bet ):
        writer.writerow( gamestat._asdict() )
```

创建 CSV 写入器有三个步骤：

1.  打开一个带有 newline 选项设置为`""`的文件。这将支持 CSV 文件的（可能）非标准行结束。

1.  创建 CSV `writer`对象。在这个例子中，我们创建了`DictWriter`实例，因为它允许我们轻松地从字典对象创建行。

1.  在文件的第一行放一个标题。这样做可以通过提供一些关于 CSV 文件中内容的提示，使数据交换稍微简单一些。

一旦`writer`对象准备好了，我们可以使用 writer 的`writerow()`方法将每个字典写入 CSV 文件。我们可以在一定程度上通过使用`writerows()`方法稍微简化这个过程。这个方法期望一个迭代器而不是一个单独的行。以下是我们如何使用`writerows()`与一个迭代器：

```py
data = gamestat_iter( Player_Strategy_1, Martingale_Bet )
with open("blackjack.stats","w",newline="") as target:
    writer= csv.DictWriter( target, GameStat._fields )
    writer.writeheader()
    writer.writerows( g._asdict() for g in data )
```

我们将迭代器分配给一个变量`data`。对于`writerows()`方法，我们从迭代器产生的每一行得到一个字典。

### 从 CSV 加载简单序列

我们可以使用类似以下代码的循环从 CSV 文件中加载简单的顺序对象：

```py
with open("blackjack.stats","r",newline="") as source:
    reader= csv.DictReader( source )
    for gs in **( GameStat(**r) for r in reader )**:
        print( gs )
```

我们为文件定义了一个`reader`对象。由于我们知道文件有一个适当的标题，我们可以使用`DictReader`。这将使用第一行来定义属性名称。现在我们可以从 CSV 文件中的行构造`GameStat`对象。我们使用了一个生成器表达式来构建行。

在这种情况下，我们假设列名与我们的`GameStat`类定义的属性名匹配。如果必要，我们可以通过比较`reader.fieldnames`和`GameStat._fields`来确认文件是否与预期格式匹配。由于顺序不必匹配，我们需要将每个字段名称列表转换为集合。以下是我们如何检查列名：

```py
assert set(reader.fieldnames) == set(GameStat._fields)
```

我们忽略了从文件中读取的值的数据类型。当我们从 CSV 文件中读取时，两个数值列将最终成为字符串值。因此，我们需要进行更复杂的逐行转换，以创建正确的数据值。以下是执行所需转换的典型工厂函数：

```py
def gamestat_iter(iterator):
    for row in iterator:
        yield GameStat( row['player'], row['bet'], int(row['rounds']), int(row['final']) )
```

我们已经将`int`函数应用于应该具有数值的列。在文件具有正确的标题但数据不正确的罕见情况下，我们将从失败的“int（）”函数中获得普通的`ValueError`。我们可以使用这个生成器函数如下：

```py
with open("blackjack.stats","r",newline="") as source:
    reader= csv.DictReader( source )
    assert set(reader.fieldnames) == set(GameStat._fields)
    for gs in gamestat_iter(reader):
        print( gs )
```

这个版本的读取器通过对数值进行转换，正确重建了`GameStat`对象。

### 处理容器和复杂类

当我们回顾我们的微博示例时，我们有一个包含许多`Post`实例的`Blog`对象。我们设计`Blog`作为`list`的包装器，以便`Blog`包含一个集合。在处理 CSV 表示时，我们必须设计从复杂结构到表格表示的映射。我们有三种常见的解决方案：

+   我们可以创建两个文件：一个博客文件和一个帖子文件。博客文件只包含`Blog`实例。在我们的示例中，每个`Blog`都有一个标题。然后，每个`Post`行可以引用帖子所属的`Blog`行。我们需要为每个`Blog`添加一个键。然后，每个`Post`将具有对`Blog`键的外键引用。

+   我们可以在单个文件中创建两种类型的行。我们将有`Blog`行和`Post`行。我们的写入器纠缠了各种类型的数据；我们的读取器必须解开数据类型。

+   我们可以在各种行之间执行关系数据库连接，重复在每个`Post`子行上的`Blog`父信息。

在这些选择中没有*最佳*解决方案。我们必须设计一个解决扁平 CSV 行和更结构化的 Python 对象之间的阻抗不匹配的解决方案。数据的用例将定义一些优点和缺点。

创建两个文件需要我们为每个`Blog`创建某种唯一标识符，以便`Post`可以正确地引用`Blog`。我们不能轻易使用 Python 内部 ID，因为这些 ID 不能保证在每次 Python 运行时保持一致。

一个常见的假设是`Blog`标题是一个唯一的键；由于这是`Blog`的属性，它被称为自然主键。这很少能奏效；我们不能更改`Blog`标题而不更新所有引用`Blog`的`Posts`。一个更好的计划是发明一个唯一标识符，并更新类设计以包括该标识符。这被称为**代理键**。Python 的`uuid`模块可以为此目的提供唯一标识符。

使用多个文件的代码几乎与先前的示例相同。唯一的变化是为`Blog`类添加适当的主键。一旦定义了键，我们就可以像以前一样创建写入器和读取器来处理`Blog`和`Post`实例到它们各自的文件中。

### 在 CSV 文件中转储和加载多种行类型

在单个文件中创建多种类型的行使格式变得更加复杂。列标题必须成为所有可用列标题的并集。由于各种行类型之间可能存在名称冲突的可能性，我们可以通过位置访问行，防止我们简单地使用`csv.DictReader`，或者我们必须发明一个更复杂的列标题，结合类和属性名称。

如果我们为每一行提供一个额外的列作为类别鉴别器，那么这个过程就会更简单。这个额外的列告诉我们行代表的是什么类型的对象。对象的类名会很好地起作用。以下是我们可能使用两种不同的行格式将博客和帖子写入单个 CSV 文件的方法：

```py
with open("blog.csv","w",newline="") as target:
    wtr.writerow(['__class__','title','date','title','rst_text','tags'])
    wtr= csv.writer( target )
    for b in blogs:
        wtr.writerow(['Blog',b.title,None,None,None,None])
        for p in b.entries:
            wtr.writerow(['Post',None,p.date,p.title,p.rst_text,p.tags])
```

我们在文件中创建了两种行的变体。一些行在第一列中有`'Blog'`，只包含`Blog`对象的属性。其他行在第一列中有`'Post'`，只包含`Post`对象的属性。

我们没有使标题唯一，因此无法使用字典读取器。像这样按位置分配列时，每行都会根据它必须共存的其他类型的行来分配未使用的列。这些额外的列填充为`None`。随着不同行类型的数量增加，跟踪各个位置列的分配可能变得具有挑战性。

此外，单独的数据类型转换可能有些令人困惑。特别是，我们忽略了时间戳和标签的数据类型。我们可以尝试通过检查行鉴别器来重新组装我们的`Blogs`和`Posts`：

```py
with open("blog.csv","r",newline="") as source:
    rdr= csv.reader( source )
    header= next(rdr)
    assert header == ['__class__','title','date','title','rst_text','tags']
    blogs = []
    for r in rdr:
        if r[0] == 'Blog':
 **blog= Blog( *r[1:2] )
            blogs.append( blog )
        if r[0] == 'Post':
 **post= post_builder( r )
            blogs[-1].append( post )
```

这段代码将构建一个`Blog`对象列表。每个`'Blog'`行使用`slice(1,2)`中的列来定义`Blog`对象。每个`'Post'`行使用`slice(2,6)`中的列来定义`Post`对象。这要求每个`Blog`后面都跟着相关的`Post`实例。外键不用于将这两个对象联系在一起。

我们对 CSV 文件中的列做了两个假设，即它们的顺序和类型与类构造函数的参数相同。对于`Blog`对象，我们使用了`blog= Blog( *r[1:2] )`，因为唯一的列是文本，这与类构造函数匹配。在处理外部提供的数据时，这个假设可能是无效的。

为了构建`Post`实例，我们使用了一个单独的函数来从列映射到类构造函数。以下是映射函数：

```py
import ast
def builder( row ):
    return Post(
        date=datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S"),
        title=row[3],
        rst_text=row[4],
        tags=ast.literal_eval(row[5]) )
```

这将从文本行正确构建一个`Post`实例。它将`datetime`的文本和标签的文本转换为它们正确的 Python 类型。这有一个使映射明确的优点。

在这个例子中，我们使用`ast.literal_eval()`来解码更复杂的 Python 文字值。这允许 CSV 数据包括一个字符串值的元组："('#RedRanger', '#Whitby42', '#ICW')"。

### 使用迭代器过滤 CSV 行

我们可以重构先前的加载示例，通过迭代`Blog`对象而不是构建`Blog`对象的列表。这使我们能够浏览大型 CSV 文件并定位只有相关的`Blog`和`Post`行。这个函数是一个生成器，分别产生每个单独的`Blog`实例：

```py
def blog_iter(source):
    rdr= csv.reader( source )
    header= next(rdr)
    assert header == ['__class__','title','date','title','rst_text','tags']
    blog= None
    for r in rdr:
        if r[0] == 'Blog':
            if blog:
                **yield blog
            blog= Blog( *r[1:2] )
        if r[0] == 'Post':
            post= post_builder( r )
            blog.append( post )
    if blog:
        **yield blog

```

这个`blog_iter()`函数创建`Blog`对象并附加`Post`对象。每当出现一个`Blog`标题时，前一个`Blog`就完成了并且可以被产出。最后，最终的`Blog`对象也必须被产出。如果我们想要大量的`Blog`实例列表，我们可以使用以下代码：

```py
with open("blog.csv","r",newline="") as source:
    blogs= list( blog_iter(source) )
```

这将使用迭代器在极少数情况下构建一个`Blogs`列表，实际上我们确实希望整个序列保存在内存中。我们可以使用以下方法逐个处理每个`Blog`，将其呈现为创建 RST 文件：

```py
with open("blog.csv","r",newline="") as source:
    for b in blog_iter(source):
        with open(blog.title+'.rst','w') as rst_file:
            render( blog, rst_file )
```

我们使用`blog_iter()`函数来读取每个博客。读取后，它可以呈现为一个 RST 格式文件。一个单独的进程可以运行`rst2html.py`将每个博客转换为 HTML。

我们可以轻松地添加一个过滤器来处理只选择的`Blog`实例。我们可以添加一个`if`语句来决定应该呈现哪些`Blogs`，而不仅仅是呈现所有的`Blog`实例。

### 在 CSV 文件中转储和加载连接的行

将对象连接在一起意味着每一行都是一个子对象，与所有父对象连接在一起。这会导致每个子对象重复父对象的属性。当存在多层容器时，这可能导致大量重复的数据。

这种重复的优势在于每行都是独立的，不属于由其上面的行定义的上下文。我们不需要类鉴别器，因为父值为每个子对象重复。

这对于形成简单层次结构的数据效果很好；每个子对象都添加了一些父属性。当数据涉及更复杂的关系时，简单的父子模式就会崩溃。在这些例子中，我们将`Post`标签合并到一个文本列中。如果我们尝试将标签分成单独的列，它们将成为每个`Post`的子对象，这意味着`Post`的文本可能会重复出现。显然，这不是一个好主意！

列标题必须成为所有可用列标题的并集。由于各种行类型之间可能存在名称冲突的可能性，我们将用类名限定每个列名。这将导致列标题，如`'Blog.title'`和`'Post.title'`，从而避免名称冲突。这允许使用`DictReader`和`DictWriter`而不是列的位置赋值。然而，这些有资格的名称并不会简单地匹配类定义的属性名称；这会导致更多的文本处理来解析列标题。以下是我们如何编写一个包含父属性和子属性的联合行：

```py
with open("blog.csv","w",newline="") as target:
    wtr= csv.writer( target )
    wtr.writerow(['Blog.title','Post.date','Post.title', 'Post.tags','Post.rst_text'])
    for b in blogs:
        for p in b.entries:
            wtr.writerow([b.title,p.date,p.title,p.tags,p.rst_text])
```

我们看到了有资格的列标题。在这种格式中，每一行现在包含了`Blog`属性和`Post`属性的并集。这样更容易准备，因为不需要用`None`填充未使用的列。由于每个列名都是唯一的，我们也可以很容易地切换到`DictWriter`。以下是从 CSV 行重构原始容器的方法：

```py
def blog_iter2( source ):
    rdr= csv.DictReader( source )
    assert set(rdr.fieldnames) == set(['Blog.title','Post.date','Post.title', 'Post.tags','Post.rst_text'])
    row= next(rdr)
    blog= Blog(row['Blog.title'])
    post= post_builder5( row )
    blog.append( post )
    for row in rdr:
        if row['Blog.title'] != blog.title:
            yield blog
            blog= Blog( row['Blog.title'] )
        post= post_builder5( row )
        blog.append( post )
    yield blog
```

第一行数据用于构建`Blog`实例和该`Blog`中的第一个`Post`。随后的循环不变条件假设存在一个合适的`Blog`对象。拥有一个有效的`Blog`实例使得处理逻辑变得简单得多。`Post`实例是用以下函数构建的：

```py
import ast
def post_builder5( row ):
    return Post(
        date=datetime.datetime.strptime(
            row['Post.date'], "%Y-%m-%d %H:%M:%S"),
        title=row['Post.title'],
        rst_text=row['Post.rst_text'],
        tags=ast.literal_eval(row['Post.tags']) )
```

我们通过将每行中的单独列映射到类构造函数的参数来映射。这使得所有的转换都是显式的。它正确处理了从 CSV 文本到 Python 对象的所有类型转换。

我们可能想要将`Blog`构建器重构为一个单独的函数。但是，它非常小，遵循 DRY 原则似乎有点麻烦。因为列标题与参数名称匹配，我们可以尝试使用以下代码构建每个对象：

```py
    def make_obj( row, class_=Post, prefix="Post" ):
        column_split = ( (k,)+tuple(k.split('.')) for k in row )
        kw_args = dict( (attr,row[key])
            for key,classname,attr in column_split if classname==prefix )
        return class( **kw_args )
```

我们在这里使用了两个生成器表达式。第一个生成器表达式将列名拆分为类和属性，并构建一个包含完整键、类名和属性名的 3 元组。第二个生成器表达式过滤了所需目标类的类；它构建了一个包含属性和值对的 2 元组序列，可以用来构建字典。

这并不处理`Posts`的数据转换。单个列映射并不通用。当与`post_builder5()`函数相比时，向此添加大量处理逻辑并不是很有帮助。

如果我们有一个空文件，即有标题行但没有`Blog`条目的文件，初始的`row=next(rdr)`函数将引发`StopIteration`异常。由于这个生成器函数没有处理异常，它将传播到评估`blog_iter2()`的循环；这个循环将被正确终止。

## 使用 XML 进行转储和加载

Python 的`xml`包包括许多解析 XML 文件的模块。还有一个**文档对象模型**（**DOM**）实现，可以生成 XML 文档。与之前的`json`模块一样，这对于 Python 对象来说并不是一个非常完整的持久性解决方案。然而，由于广泛采用 XML 文件，通常需要在 Python 对象和 XML 文档之间进行转换。

处理 XML 文件涉及我们的对象和 XML 结构之间的手动映射。我们需要仔细设计映射，同时要意识到 XML 符号的约束。这可能很困难，因为对象的表达能力与 XML 文档的严格分层性质之间存在不匹配。

XML 属性或标记的内容是纯文本。在加载 XML 文档时，我们需要将这些值转换为我们应用程序内部更有用的类型。在某些情况下，XML 文档可能包括属性或标记以指示预期的类型。

如果我们愿意忍受一些限制，我们可以使用`plistlib`模块将一些内置的 Python 结构发出为 XML 文档。我们将在第十三章中详细介绍这个模块，*配置文件和持久性*，在那里我们将使用它来加载配置文件。

### 注意

`json`模块提供了将 JSON 编码扩展到包括我们自定义类的方法；`plistlib`模块没有提供此额外的钩子。

当我们考虑将 Python 对象转储为 XML 文档时，有三种常见的构建文本的方法：

+   在我们的类设计中包含 XML 输出方法。在这种情况下，我们的类发出可以组装成 XML 文档的字符串。

+   使用`xml.etree.ElementTree`构建`ElementTree`节点并返回此结构。这可以呈现为文本。

+   使用外部模板并将属性填充到该模板中。除非我们有一个复杂的模板工具，否则这样做效果不佳。标准库中的`string.Template`类仅适用于非常简单的对象。

有一些通用的 Python XML 序列化器示例。尝试创建通用序列化器的问题在于 XML 非常灵活；每个 XML 应用似乎都有独特的**XML 模式定义**（**XSD**）或**文档类型定义**（**DTD**）要求。

一个开放的设计问题是如何编码原子值。有很多选择。我们可以在标记的属性中使用特定类型的标记：`<int name="the_answer">42</int>`。另一种可能性是在标记的属性中使用特定类型的标记：`<the_answer type="int">42</the_answer>`。我们还可以使用嵌套标记：`<the_answer><int>42</int></the_answer>`。或者，我们可以依赖于单独的模式定义，建议`the_answer`应该是一个整数，并仅将值编码为文本：`<the_answer>42</the_answer>`。我们还可以使用相邻的标记：`<key>the_answer</key><int>42</int>`。这并不是一个详尽的列表；XML 为我们提供了很多选择。

当从 XML 文档中恢复 Python 对象时，我们受到解析器 API 的限制。通常，我们必须解析文档，然后检查 XML 标记结构，从可用数据中组装 Python 对象。

一些 Web 框架，如 Django，包括 Django 定义类的 XML 序列化。这不是任意 Python 对象的通用序列化。序列化由 Django 的数据建模组件严格定义。此外，还有诸如`dexml`、`lxml`和`pyxser`等软件包，作为 Python 对象和 XML 之间的替代绑定。请参阅[`pythonhosted.org/dexml/api/dexml.html`](http://pythonhosted.org/dexml/api/dexml.html)、[`lxml.de`](http://lxml.de)和[`coder.cl/products/pyxser/`](http://coder.cl/products/pyxser/)。以下是候选软件包的更长列表：[`wiki.python.org/moin/PythonXml`](https://wiki.python.org/moin/PythonXml)。

### 使用字符串模板转储对象

将 Python 对象序列化为 XML 的一种方法是创建 XML 文本。这是一种手动映射，通常实现为一个方法函数，该函数发出与 Python 对象对应的 XML 片段。对于复杂对象，容器必须获取容器内每个项目的 XML。以下是我们的微博类结构的两个简单扩展，添加了文本的 XML 输出功能：

```py
class Blog_X( Blog ):
    def xml( self ):
        children= "\n".join( c.xml() for c in self.entries )
        return """\
<blog><title>{0.title}</title>
<entries>
{1}
<entries></blog>""".format(self,children)

class Post_X( Post ):
    def xml( self ):
        tags= "".join( "<tag>{0}</tag>".format(t) for t in self.tags )
        return """\
<entry>
    <title>{0.title}</title>
    <date>{0.date}</date>
    <tags>{1}</tags>
    <text>{0.rst_text}</text>
</entry>""".format(self,tags)
```

我们编写了一些高度特定于类的 XML 输出方法。这些方法将发出包装在 XML 语法中的相关属性。这种方法不太通用。`Blog_X.xml()`方法发出带有标题和条目的`<blog>`标记。`Post_X.xml()`方法发出带有各种属性的`<post>`标记。在这两种方法中，使用`"".join()`或`"\n".join()`创建了较短字符串元素的较长字符串。当我们将`Blog`对象转换为 XML 时，结果如下：

```py
<blog><title>Travel</title>
<entries>
<entry>
    <title>Hard Aground</title>
    <date>2013-11-14 17:25:00</date>
    <tags><tag>#RedRanger</tag><tag>#Whitby42</tag><tag>#ICW</tag></tags>
    <text>Some embarrassing revelation. Including ☹ and ⚓</text>
</entry>
<entry>
    <title>Anchor Follies</title>
    <date>2013-11-18 15:30:00</date>
    <tags><tag>#RedRanger</tag><tag>#Whitby42</tag><tag>#Mistakes</tag></tags>
    <text>Some witty epigram.</text>
</entry>
<entries></blog>
```

这种方法有两个缺点：

+   我们忽略了 XML 命名空间。这是发出标记的文字的一个小改变。

+   每个类还需要正确转义`<`、`&`、`>`和`"`字符为 XML 实体`&lt;`、`&gt;`、`&amp;`和`&quot;`。`html`模块包括`html.escape()`函数来执行此操作。

这确实发出了正确的 XML；可以依赖它工作；它不太优雅，也不太通用。

### 使用 xml.etree.ElementTree 转储对象

我们可以使用`xml.etree.ElementTree`模块构建可以作为 XML 发出的`Element`结构。使用`xml.dom`和`xml.minidom`进行这项工作是具有挑战性的。DOM API 需要一个顶级文档，然后构建单独的元素。当尝试序列化具有多个属性的简单类时，必要的上下文对象的存在会导致混乱。我们必须首先创建文档，然后序列化文档的所有元素，并将文档上下文作为参数提供。

通常，我们希望设计中的每个类都构建一个顶级元素并返回。大多数顶级元素将具有一系列子元素。我们可以为构建的每个元素分配文本以及属性。我们还可以分配一个*tail*，即跟在封闭标记后面的多余文本。在某些内容模型中，这只是空白。由于名称很长，可能有助于以以下方式导入`ElementTree`：

```py
import xml.etree.ElementTree as XML
```

以下是我们的微博类结构的两个扩展，将 XML 输出功能添加为`Element`实例。我们向`Blog`类添加了以下方法：

```py
    def xml( self ):
        blog= XML.Element( "blog" )
        title= XML.SubElement( blog, "title" )
        title.text= self.title
        title.tail= "\n"
        entities= XML.SubElement( blog, "entities" )
        entities.extend( c.xml() for c in self.entries )
        blog.tail= "\n"
        return blog
```

我们向`Post`类添加了以下方法：

```py
    def xml( self ):
        post= XML.Element( "entry" )
        title= XML.SubElement( post, "title" )
        title.text= self.title
        date= XML.SubElement( post, "date" )
        date.text= str(self.date)
        tags= XML.SubElement( post, "tags" )
        for t in self.tags:
            tag= XML.SubElement( tags, "tag" )
            tag.text= t
        text= XML.SubElement( post, "rst_text" )
        text.text= self.rst_text
        post.tail= "\n"
        return post
```

我们编写了高度特定于类的 XML 输出方法。这些方法将构建具有适当文本值的`Element`对象。

### 注意

没有用于构建子元素的流畅快捷方式。我们必须逐个插入每个文本项。

在`blog`方法中，我们能够执行`Element.extend()`将所有单独的帖子条目放在`<entry>`元素内。这使我们能够灵活而简单地构建 XML 结构。这种方法可以优雅地处理 XML 命名空间。我们可以使用`QName`类为 XML 命名空间构建合格的名称。`ElementTree`模块正确地将命名空间限定符应用于 XML 标记。这种方法还可以将`<`、`&`、`>`和`"`字符正确转义为 XML 实体`&lt;`、`&gt;`、`&amp;`和`&quot;`。这些方法生成的 XML 输出大部分将与上一节相匹配。空格将不同。

### 加载 XML 文档

从 XML 文档加载 Python 对象是一个两步过程。首先，我们需要解析 XML 文本以创建文档对象。然后，我们需要检查文档对象以生成 Python 对象。正如前面所述，XML 符号的巨大灵活性意味着没有单一的 XML 到 Python 序列化。

遍历 XML 文档的一种方法涉及进行类似 XPath 的查询，以定位解析的各种元素。以下是一个遍历 XML 文档的函数，从可用的 XML 中发出`Blog`和`Post`对象：

```py
    import ast
    doc= XML.parse( io.StringIO(text.decode('utf-8')) )
    xml_blog= doc.getroot()
    blog= Blog( xml_blog.findtext('title') )
    for xml_post in xml_blog.findall('entries/entry'):
        tags= [t.text for t in xml_post.findall( 'tags/tag' )]
        post= Post(
            date= datetime.datetime.strptime(
                xml_post.findtext('date'), "%Y-%m-%d %H:%M:%S"),
            title=xml_post.findtext('title'),
            tags=tags,
            rst_text= xml_post.findtext('rst_text')
         )
        blog.append( post )
    render( blog )
```

这段代码遍历了一个`<blog>` XML 文档。它定位了`<title>`标记，并收集该元素内的所有文本，以创建顶层的`Blog`实例。然后，它定位了`<entries>`元素内找到的所有`<entry>`子元素。这些用于构建每个`Post`对象。`Post`对象的各种属性被单独转换。`<tags>`元素内每个单独的`<tag>`元素的文本被转换为文本值列表。日期从其文本表示中解析出来。每个`Post`对象都被追加到整体的`Blog`对象中。这种从 XML 文本到 Python 对象的*手动*映射是解析 XML 文档的一个重要特性。

## 摘要

我们已经看过了多种序列化 Python 对象的方法。我们可以在各种符号中对我们的类定义进行编码，包括 JSON、YAML、pickle、XML 和 CSV。每种符号都有各种优点和缺点。

这些不同的库模块通常围绕着从外部文件加载对象或将对象转储到文件的想法。这些模块并不完全一致，但它们非常相似，允许我们应用一些常见的设计模式。

使用 CSV 和 XML 往往会暴露出最困难的设计问题。我们在 Python 中的类定义可以包括在 CSV 或 XML 符号中没有很好表示的对象引用。

### 设计考虑和权衡

有许多方法可以序列化和持久化 Python 对象。我们还没有看到它们的全部。本节中的格式侧重于两个基本用例：

+   **与其他应用程序的数据交换**：我们可能会为其他应用程序发布数据或接受其他应用程序的数据。在这种情况下，我们通常受到其他应用程序接口的限制。通常，其他应用程序和框架使用 JSON 和 XML 作为其首选的数据交换形式。在某些情况下，我们将使用 CSV 来交换数据。

+   **我们自己应用程序的持久数据**：在这种情况下，我们通常会选择`pickle`，因为它是完整的，并且已经是 Python 标准库的一部分。然而，YAML 的一个重要优势是它的可读性；我们可以查看、编辑甚至修改文件。

在处理这些格式时，我们有许多设计考虑。首先，这些格式偏向于序列化单个 Python 对象。它可能是其他对象的列表，但本质上是单个对象。例如，JSON 和 XML 具有在序列化对象之后编写的结束分隔符。对于从较大域中持久化单个对象，我们可以查看第十章中的`shelve`和`sqlite3`，*通过 Shelve 存储和检索对象*和第十一章中的`shelve`和`sqlite3`，*通过 SQLite 存储和检索对象*。

JSON 是一个广泛使用的标准。它不方便表示复杂的 Python 类。在使用 JSON 时，我们需要意识到我们的对象如何被简化为与 JSON 兼容的表示形式。JSON 文档是人类可读的。JSON 的限制使其在通过互联网传输对象时可能更安全。

YAML 并不像 JSON 那样广泛使用，但它解决了序列化和持久性中的许多问题。YAML 文档是人类可读的。对于可编辑的配置文件，YAML 是理想的。我们可以使用 safe-load 选项使 YAML 安全。

Pickle 非常适合于 Python 对象的简单，快速的本地持久性。它是从 Python 到 Python 的传输的紧凑表示。CSV 是一个广泛使用的标准。在 CSV 表示中为 Python 对象制定表示形式是具有挑战性的。在 CSV 表示中共享数据时，我们经常在应用程序中使用`namedtuples`。我们必须设计一个从 Python 到 CSV 和从 CSV 到 Python 的映射。

XML 是另一种广泛使用的序列化数据的表示形式。XML 非常灵活，导致了多种在 XML 表示中编码 Python 对象的方式。由于 XML 用例，我们经常有外部规范，如 XSD 或 DTD。解析 XML 以创建 Python 对象的过程总是相当复杂的。

因为每个 CSV 行在很大程度上独立于其他行，CSV 允许我们编码或解码极大的对象集合。因此，CSV 通常用于编码和解码无法放入内存的巨大集合。

在某些情况下，我们面临混合设计问题。在阅读大多数现代电子表格文件时，我们遇到了 CSV 行列问题和 XML 解析问题。例如，[OpenOffice.org](http://OpenOffice.org)。ODS 文件是压缩存档。存档中的一个文件是`content.xml`文件。使用 XPath 搜索`body/spreadsheet/table`元素将定位电子表格文档的各个选项卡。在每个表格中，我们会找到通常映射到 Python 对象的`table-row`元素。在每行中，我们会找到包含构建对象属性的单个值的`table-cell`元素。

### 模式演变

在处理持久对象时，我们必须解决模式演变的问题。我们的对象具有动态状态和静态类定义。我们可以轻松地保存动态状态。我们的类定义是持久数据的模式。然而，类并非*绝对*静态。当类发生变化时，我们需要提供加载由应用程序的先前版本转储的数据的方法。

最好考虑外部文件兼容性，以区分主要和次要发布版本号。主要发布应意味着文件不再兼容，必须进行转换。次要发布应意味着文件格式兼容，升级不涉及数据转换。

一种常见的方法是在文件扩展名中包含主版本号。我们可能会有以`.json2`或`.json3`结尾的文件名，以指示涉及哪种数据格式。支持持久文件格式的多个版本通常变得相当复杂。为了提供无缝升级路径，应用程序应能够解码先前的文件格式。通常，最好将数据持久化在最新和最好的文件格式中，即使其他格式也支持输入。

在接下来的章节中，我们将讨论不专注于单个对象的序列化。`shelve`和`sqlite3`模块为我们提供了序列化一系列不同对象的方法。之后，我们将再次使用这些技术来进行**表述状态转移**（**REST**）以将对象从一个进程传输到另一个进程。此外，我们还将再次使用这些技术来处理配置文件。

### 展望未来

在第十章和第十一章中，我们将看到两种常见的方法来创建更大的持久对象集合。这两章向我们展示了创建 Python 对象数据库的不同方法。

在第十二章中，*传输和共享对象*，我们将把这些序列化技术应用到使对象在另一个进程中可用的问题上。我们将专注于 RESTful web 服务作为在进程之间传输对象的简单和流行的方式。

在第十三章中，*配置文件和持久化*，我们将再次应用这些序列化技术。在这种情况下，我们将使用 JSON 和 YAML 等表示形式来编码应用程序的配置信息。

# 第十章：通过 Shelve 存储和检索对象

有许多应用程序需要单独持久化对象。我们在第九章中看到的技术，*序列化和保存-JSON、YAML、Pickle、CSV 和 XML*，偏向于处理单个对象。有时，我们需要持久化来自更大领域的单独对象。

具有持久对象的应用程序可能展示四种用例，总结为**CRUD 操作**：创建、检索、更新和删除。在一般情况下，这些操作中的任何一个都可以应用于域中的任何对象；这导致需要比单一的加载或转储到文件更复杂的持久化机制。除了浪费内存外，简单的加载和转储通常比精细的、逐个对象的存储效率低。

使用更复杂的存储将使我们更加关注责任的分配。各种关注点为我们提供了应用软件架构的整体设计模式。这些更高级别的设计模式之一是**三层架构**：

+   **表示层**：这可能是 Web 浏览器或移动应用程序，有时两者都有。

+   **应用层**：这通常部署在应用服务器上。应用层应该被细分为应用层和数据模型层。处理层涉及体现应用行为的类。数据模型层定义了问题域的对象模型。

+   **数据层**：这包括访问层和持久化层。访问层提供对持久对象的统一访问。持久化层将对象序列化并将其写入持久存储。

这个模型可以应用于单个 GUI 应用程序。表示层是 GUI；应用层是相关的处理器和数据模型；访问层是持久性模块。它甚至适用于命令行应用程序，其中表示层仅仅是一个选项解析器以及`print()`函数。

`shelve`模块定义了一个类似映射的容器，我们可以在其中存储对象。每个存储的对象都被 pickled 并写入文件。我们还可以从文件中 unpickle 并检索任何对象。`shelve`模块依赖于`dbm`模块来保存和检索对象。

本节将重点关注从应用程序层获取的数据模型以及从数据层获取的访问和持久性。这两个层之间的接口可以简单地是单个应用程序内的类接口。或者，它可以是一个更复杂的网络接口。在本章中，我们将重点关注简单的类与类接口。我们将在第十二章中，使用 REST，*传输和共享对象*，关注基于网络的接口。

## 分析持久对象的用例

我们在第九章中看到的持久性机制，*序列化和保存 - JSON、YAML、Pickle、CSV 和 XML*，侧重于读取和写入一个序列化对象的紧凑文件。如果我们想要更新文件的任何部分，我们被迫替换整个文件。这是使用紧凑表示法的后果；很难到达文件中对象的位置，如果大小发生变化，替换对象也很困难。我们并没有用巧妙、复杂的算法来解决这些困难，而是简单地对对象进行了序列化和写入。当我们有一个更大的领域，有许多持久的、可变的对象时，我们引入了一些额外的深度到用例中。以下是一些额外的考虑：

+   我们可能不想一次将所有对象加载到内存中。对于许多*大数据*应用程序，一次性加载所有对象可能是不可能的。

+   我们可能只更新来自对象领域的小子集或单个实例。加载然后转储所有对象以更新一个对象是相对低效的处理。

+   我们可能不会一次性转储所有对象；我们可能会逐渐累积对象。一些格式，如 YAML 和 CSV，允许我们以很少的复杂性将自己附加到文件上。其他格式，如 JSON 和 XML，有终止符，使得简单地附加到文件变得困难。

我们可能还想要更多的功能。将序列化、持久性以及并发更新或写访问混为一谈，统称为*数据库*是很常见的。`shelve`模块本身并不是一个全面的数据库解决方案。`shelve`使用的底层`dbm`模块并不直接处理并发写。它也不处理多操作事务。可以使用低级别的操作系统文件锁定来容忍并发更新，但这往往是高度依赖操作系统的。对于并发写访问，最好使用适当的数据库或 RESTful 数据服务器。参见第十一章，*通过 SQLite 存储和检索对象*，以及第十二章，*传输和共享对象*。

### ACID 属性

我们的设计必须考虑**ACID 属性**如何适用于我们的`shelve`数据库。我们的应用程序通常会对相关操作进行捆绑更改，这些操作应该将数据库从一个一致的状态更改到下一个一致的状态。改变数据库的一系列操作可以称为事务。

多操作事务的一个例子可能涉及更新两个对象，以保持总和不变。我们可能会从一个财务账户中扣除并存入另一个账户。整体余额必须保持恒定，以使数据库处于一致的有效状态。ACID 属性表征了我们希望数据库事务作为一个整体的行为。有四条规则定义了我们的期望：

+   **原子性**：事务必须是原子的。如果事务中有多个操作，要么所有操作都完成，要么都不完成。不应该可能查看一个部分完成的事务的架子。

+   **一致性**：事务必须保证一致性。它将把数据库从一个有效状态改变为另一个有效状态。事务不应该损坏数据库或在并发用户之间创建不一致的视图。所有用户看到已完成事务的相同净效果。

+   **隔离性**：每个事务应该像完全隔离一样正常运行。我们不能让两个并发用户干扰彼此的尝试更新。我们必须能够将并发访问转换为（可能更慢的）串行访问，并且数据库更新将产生相同的结果。

+   **持久性**：对数据库的更改是**持久的**；它们在文件系统中正确地持久存在。

当我们使用内存中的 Python 对象时，显然，我们得到了**ACI**，但没有得到**D**。内存中的对象根据定义是不持久的。如果我们尝试在几个并发进程中使用`shelve`模块而没有锁定或版本控制，我们可能只得到 D，但失去 ACI 属性。

`shelve`模块不直接支持原子性；它没有处理由多个操作组成的事务的方法。如果我们有多个操作的事务并且需要原子性，我们必须确保它们全部成功或全部失败。这可能涉及到相当复杂的`try:`语句，必须在失败的情况下恢复数据库的先前状态。

`shelve`模块不保证对所有种类的更改都是持久的。如果我们将一个可变对象放到架子上，然后在内存中更改对象，架子文件上的持久版本将不会*自动*更改。如果我们要改变架子上的对象，我们的应用程序必须明确地更新架子。我们可以要求架子对象通过*写回模式*跟踪更改，但使用这个特性可能会导致性能不佳。

## 创建一个架子

创建架子的第一部分是使用模块级函数`shelve.open()`来创建一个持久的架子结构。第二部分是正确关闭文件，以便所有更改都被写入底层文件系统。我们稍后会在一个更完整的例子中看到这一点。

在幕后，`shelve`模块使用`dbm`模块来进行真正的工作，打开文件并从键到值的映射。`dbm`模块本身是一个围绕底层 DBM 兼容库的包装器。因此，`shelve`功能有许多潜在的实现。好消息是，`dbm`实现之间的差异在很大程度上是无关紧要的。

`shelve.open()`模块函数需要两个参数：文件名和文件访问模式。通常，我们希望使用`'c'`的默认模式来打开一个现有的架子，如果不存在则创建一个。专门情况下的替代方案有：

+   `'r'`是一个只读的架子

+   `'w'`是一个读写的架子，*必须*存在，否则将引发异常

+   `'n'`是一个新的、空的架子；任何以前的版本都将被覆盖

关闭架子以确保它被正确地持久化到磁盘是绝对必要的。架子本身不是上下文管理器，但`contextlib.closing()`函数可以用来确保架子被关闭。有关上下文管理器的更多信息，请参见第五章，“使用可调用和上下文”。

在某些情况下，我们可能还希望显式地将架子与磁盘同步，而不关闭文件。`shelve.sync()`方法将在关闭之前持久化更改。理想的生命周期看起来像以下代码：

```py
import shelve
from contextlib import closing
with closing( shelve.open('some_file') ) as shelf:
    process( shelf )
```

我们打开了一个架子，并将打开的架子提供给一些执行我们应用程序真正工作的函数。当这个过程完成时，上下文将确保架子被关闭。如果`process()`函数引发异常，架子仍将被正确关闭。

## 设计可架架对象

如果我们的对象相对简单，那么将它们放在架子上将是微不足道的。对于不是复杂容器或大型集合的对象，我们只需要解决键到值的映射。对于更复杂的对象——通常包含其他对象的对象——我们必须就访问的粒度和对象之间的引用做出一些额外的设计决策。

我们将首先看一个简单的情况，我们只需要设计用于访问我们对象的键。然后，我们将看一些更复杂的情况，其中粒度和对象引用起作用。

### 为我们的对象设计键

`shelve`（和`dbm`）的重要特性是可以立即访问任意大的对象宇宙中的任何对象。`shelve`模块与类似字典的映射一起工作。架子映射存在于持久存储上，因此我们放在架子上的任何对象都将被序列化和保存。`pickle`模块用于执行实际的序列化。

我们必须用某种键来标识我们的架子对象，这个键将映射到对象。与字典一样，键是经过哈希处理的，这是一个非常快速的计算。这很快是因为键被限制为字节字符串；哈希是这些字节的模和。由于 Python 字符串可以轻松编码为字节，这意味着字符串值是键的常见选择。这与内置的`dict`不同，其中任何不可变对象都可以用作键。

由于键定位值，这意味着键必须是唯一的。这对我们的类施加了一些设计考虑，以提供适当的唯一键。在某些情况下，问题域将具有一个明显的唯一键属性。在这种情况下，我们可以简单地使用该属性来构造这个键：`shelf[object.key_attribute]= object`。这是最简单的情况，但不太通用。

在其他情况下，我们的应用问题没有提供适当的唯一键。例如，当对象的每个属性都可能是可变的或潜在的非唯一时，就会经常出现这个问题。例如，在处理美国公民时，社会安全号码并不是唯一的；它们可以被社会安全管理局重新使用。此外，一个人可能会错误报告社会安全号码，应用程序可能需要更改它；因为它可以更改，这是它不适合作为主键的第二个原因。

我们的应用程序可能有候选或主键的非字符串值。例如，我们可能有一个`datetime`对象、一个数字，甚至一个元组作为唯一标识符。在所有这些情况下，我们可能希望将值编码为字节或字符串。

在没有明显主键的情况下，我们可以尝试找到一组值的组合，创建一个唯一的**复合键**。这并不总是一个非常好的主意，因为现在键不是原子的，对键的任何部分的更改都会创建数据更新问题。

遵循一种称为**代理键**的设计模式通常是最简单的。这个键不依赖于对象内部的数据；它是对象的代理。这意味着对象的任何属性都可以更改而不会导致复杂或限制。Python 的内部对象 ID 就是一种代理键的例子。架子键的字符串表示可以遵循这种模式：`class:oid`。

键字符串包括与对象实例的唯一标识符配对的对象类。我们可以使用这种形式的键轻松地将各种类的对象存储在单个架子中。即使我们认为架子中只会有一种类型的对象，这种格式仍然有助于为索引、管理元数据和未来扩展保存命名空间。

当我们有一个合适的自然键时，我们可以这样做来将对象持久化到架子中：`self[object.__class__.__name__+":"+object.key_attribute]= object`

这为我们提供了一个独特的类名，以及一个简单的标识符作为每个对象的唯一键值。对于代理键，我们需要为键定义某种生成器。

### 为对象生成代理键

我们将使用整数计数器生成唯一的代理键。为了确保我们正确更新这个计数器，我们将把它与我们的其他数据一起存储在架子中。尽管 Python 有一个内部对象 ID，但我们不应该使用 Python 的内部标识符作为代理键。Python 的内部 ID 号没有任何保证。

由于我们将向我们的架子中添加一些管理对象，我们必须给这些对象分配具有独特前缀的唯一键。我们将使用`_DB`。这将是我们架子中对象的一个虚假类。这些管理对象的设计决策与应用程序对象的设计类似。我们需要选择存储的粒度。我们有两种选择：

+   **粗粒度**：我们可以创建一个带有所有代理键生成的管理开销的单个`dict`对象。一个单一的键，比如`_DB:max`可以标识这个对象。在这个`dict`中，我们可以将类名映射到使用的最大标识符值。每次创建一个新对象，我们都会从这个映射中分配 ID，然后在架子中替换映射。我们将在下一节展示粗粒度解决方案。

+   **细粒度**：我们可以向数据库添加许多项目，每个项目都具有不同类的对象的最大键值。这些额外的键项中的每一个都具有形式`_DB:max:class`。每个键的值只是一个整数，迄今为止为给定类分配的最大顺序标识符。

这里的一个重要考虑因素是，我们已经将应用程序类的键设计与类设计分开。我们可以（也应该）尽可能简单地设计我们的应用程序对象。我们应该添加足够的开销，使`shelve`正常工作，但不要过多。

### 设计一个带有简单键的类

将`shelve`键存储为存储对象的属性是有帮助的。将键保留在对象中使得删除或替换对象更容易。显然，在创建对象时，我们将从不带键的对象开始，直到它存储在架子上。一旦存储，Python 对象需要设置一个键属性，以便内存中的每个对象都包含正确的键。

在检索对象时，有两种用例。我们可能需要一个已知键的特定对象。在这种情况下，架子将键映射到对象。我们可能还需要一组相关对象，不是通过它们的键而是通过其他属性的值来识别。在这种情况下，我们将通过某种搜索或查询来发现对象的键。我们将在下一节中讨论搜索算法。

为了支持在对象中保存架子键，我们将为每个对象添加一个`_id`属性。它将在每个放入架子或从架子中检索的对象中保留架子键。这将简化需要在架子中替换或移除的对象的管理。我们有以下选择来将其添加到类中：

+   不：这对于课程并不重要；这只是持久性机制的开销

+   是的：这是重要的数据，我们应该在`__init__()`中正确初始化它

我们建议不要在`__init__()`方法中定义代理键；它们并不重要，只是持久性实现的一部分。例如，代理键不会有任何方法函数，它永远不会成为应用程序层或表示层的处理层的一部分。这是一个整体`Blog`的定义：

```py
class Blog:
    def __init__( self, title, *posts ):
        self.title= title
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= "="*len(self.title),
        )
```

我们只提供了一个`title`属性和一点点更多。`Blog.as_dict()`方法可以与模板一起使用，以 RST 表示法提供字符串值。我们将把博客中的个别帖子的考虑留给下一节。

我们可以以以下方式创建一个`Blog`对象：

```py
>>> b1= Blog( title="Travel Blog" )
```

当我们将这个简单对象存储在架子上时，我们可以做这样的事情：

```py
>>> import shelve
>>> shelf= shelve.open("blog")
>>> b1._id= 'Blog:1'
>>> shelf[b1._id]= b1
```

我们首先打开了一个新的架子。文件名为“`blog`”。我们在我们的`Blog`实例`b1`中放入了一个键“'Blog:1'”。我们使用`_id`属性中给定的键将该`Blog`实例存储在架子中。

我们可以这样从架子上取回物品：

```py
>>> shelf['Blog:1']
<__main__.Blog object at 0x1007bccd0>
>>> shelf['Blog:1'].title
'Travel Blog'
>>> shelf['Blog:1']._id
'Blog:1'
>>> list(shelf.keys())
['Blog:1']
>>> shelf.close()
```

当我们引用`shelf['Blog:1']`时，它将从架子中获取我们原始的`Blog`实例。我们只在架子上放了一个对象，正如我们从键列表中看到的那样。因为我们关闭了架子，对象是持久的。我们可以退出 Python，重新启动，打开架子，看到对象仍然在架子上，使用分配的键。之前，我们提到了检索的第二个用例：在不知道键的情况下定位项目。这是一个查找，找到所有标题为给定标题的博客：

```py
>>> shelf= shelve.open('blog')
>>> results = ( shelf[k] for k in shelf.keys() if k.startswith('Blog:') and shelf[k].title == 'Travel Blog' )
>>> list(results)                                                               [<__main__.Blog object at 0x1007bcc50>]
>>> r0= _[0]
>>> r0.title
'Travel Blog'
>>> r0._id
'Blog:1'
```

我们打开了架子以访问对象。`results`生成器表达式检查架子中的每个项目，以找到那些键以`'Blog:'`开头，并且对象的标题属性是字符串`'Travel Blog'`的项目。

重要的是，键`'Blog:1'`存储在对象本身内。`_id`属性确保我们对应用程序正在处理的任何项目都有正确的键。现在我们可以改变对象并使用其原始键将其替换到架子中。

### 为容器或集合设计类

当我们有更复杂的容器或集合时，我们需要做出更复杂的设计决策。第一个问题是关于包含范围。我们必须决定我们架子上的对象的粒度。

当我们有一个容器时，我们可以将整个容器作为单个复杂对象持久化到我们的架子上。在某种程度上，这可能会破坏首先在架子上有多个对象的目的。存储一个大容器给我们粗粒度的存储。如果我们更改一个包含的对象，整个容器必须被序列化和存储。如果我们最终在单个容器中有效地将整个对象宇宙进行 pickle，为什么要使用`shelve`？我们必须找到一个适合应用需求的平衡点。

另一种选择是将集合分解为单独的个体项目。在这种情况下，我们的顶级`Blog`对象将不再是一个适当的 Python 容器。父对象可能使用键的集合引用每个子对象。每个子对象可以通过键引用父对象。这种使用键的方式在面向对象设计中是不寻常的。通常，对象只包含对其他对象的引用。在使用`shelve`（或其他数据库）时，我们必须使用键的间接引用。

每个子对象现在将有两个键：它自己的主键，加上一个**外键**，这个外键是父对象的主键。这导致了一个关于表示父对象和子对象的键字符串的第二个设计问题。

### 通过外键引用对象

我们用来唯一标识一个对象的键是它的**主键**。当子对象引用父对象时，我们需要做出额外的设计决策。我们如何构造子对象的主键？基于对象类之间的依赖关系的类型，有两种常见的子键设计策略：

+   `"Child:cid"`: 当我们有子对象可以独立于拥有父对象存在时，我们将使用这个。例如，发票上的项目指的是一个产品；即使没有产品的发票项目，产品也可以存在。

+   `"Parent:pid:Child:cid"`: 当子对象不能没有父对象存在时，我们将使用这个。例如，顾客地址没有顾客的话就不存在。当子对象完全依赖于父对象时，子对象的键可以包含拥有父对象的 ID 以反映这种依赖关系。

与父类设计一样，如果我们保留主键和与每个子对象关联的所有外键，那么最容易。我们建议不要在`__init__()`方法中初始化它们，因为它们只是持久性的特征。这是`Blog`中`Post`的一般定义：

```py
import datetime
class Post:
    def __init__( self, date, title, rst_text, tags ):
        self.date= date
        self.title= title
        self.rst_text= rst_text
        self.tags= tags
    def as_dict( self ):
        return dict(
            date= str(self.date),
            title= self.title,
            underline= "-"*len(self.title),
            rst_text= self.rst_text,
            tag_text= " ".join(self.tags),
        )
```

我们为每个微博帖子提供了几个属性。`Post.as_dict()`方法可以与模板一起使用，以 RST 格式提供字符串值。我们避免提及`Post`的主键或任何外键。以下是两个`Post`实例的示例：

```py
p2= Post( date=datetime.datetime(2013,11,14,17,25),
        title="Hard Aground",
        rst_text="""Some embarrassing revelation. Including ☹ and ⎕""",
        tags=("#RedRanger", "#Whitby42", "#ICW"),
        )

p3= Post( date=datetime.datetime(2013,11,18,15,30),
        title="Anchor Follies",
        rst_text="""Some witty epigram. Including < & > characters.""",
        tags=("#RedRanger", "#Whitby42", "#Mistakes"),
        )
```

我们现在可以通过设置属性和分配键来将这些与它们拥有的博客关联起来。我们将通过几个步骤来做到这一点：

1.  我们将打开架子并取出一个父`Blog`对象。我们将称之为`owner`：

```py
>>> import shelve
>>> shelf= shelve.open("blog")
>>> owner= shelf['Blog:1']
```

我们使用主键来定位拥有者项目。实际应用可能会使用搜索来通过标题定位这个项目。我们可能还创建了一个索引来优化搜索。我们将在下面看一下索引和搜索。

1.  现在，我们可以将这个拥有者的键分配给每个`Post`对象并持久化这些对象：

```py
>>> p2._parent= owner._id
>>> p2._id= p2._parent + ':Post:2'
>>> shelf[p2._id]= p2

>>> p3._parent= owner._id
>>> p3._id= p3._parent + ':Post:3'
>>> shelf[p3._id]= p3
```

我们将父信息放入每个`Post`中。我们使用父信息来构建主键。对于这种依赖类型的键，`_parent`属性值是多余的；它可以从键中推断出来。然而，如果我们对`Posts`使用独立键设计，`_parent`就不会在键中重复。当我们查看键时，我们可以看到`Blog`加上两个`Post`实例：

```py
>>> list(shelf.keys())
['Blog:1:Post:3', 'Blog:1', 'Blog:1:Post:2']
```

当我们获取任何子`Post`时，我们将知道每个帖子的正确父`Blog`：

```py
>>> p2._parent
'Blog:1'
>>> p2._id
'Blog:1:Post:2'
```

从父`Blog`到子`Post`的键的反向跟踪会更加复杂。我们将单独讨论这个，因为我们经常希望通过索引优化从父对象到子对象的路径。

### 设计复杂对象的 CRUD 操作

当我们将一个更大的集合分解为多个独立的细粒度对象时，我们将在架子上有多个类别的对象。因为它们是独立的对象，它们将导致每个类别的对象有独立的 CRUD 操作集合。在某些情况下，这些对象是独立的，对一个类别的对象的操作不会影响到其他对象。

然而，在我们的例子中，`Blog`和`Post`对象存在依赖关系。`Post`对象是父`Blog`的子对象；子对象不能没有父对象存在。当存在这些依赖关系时，我们需要设计更加复杂的操作集合。以下是一些考虑因素：

+   独立（或父）对象上的 CRUD 操作：

+   我们可以创建一个新的空父对象，为这个对象分配一个新的主键。我们以后可以将子对象分配给这个父对象。例如，`shelf['parent:'+object._id]= object`这样的代码将创建父对象。

+   我们可以更新或检索此父级，而不会对子级产生任何影响。我们可以在赋值的右侧执行`shelf['parent:'+some_id]`来检索父级。一旦我们有了对象，我们可以执行`shelf['parent:'+object._id]= object`来保存更改。

+   删除父级可能导致两种行为之一。一种选择是级联删除以包括所有引用父级的子级。或者，我们可以编写代码来禁止删除仍具有子级引用的父级。这两种选择都是合理的，选择取决于问题域所施加的要求。

+   对依赖（或子级）对象进行 CRUD 操作：

+   我们可以创建一个引用现有父级的新子级。我们必须解决键设计问题，以决定我们想要为子级使用什么样的键。

+   我们可以在父级之外更新、检索或删除子级。这甚至可以包括将子级分配给不同的父级。

由于替换对象的代码与更新对象的代码相同，因此 CRUD 处理的一半通过简单的赋值语句处理。删除使用`del`语句完成。删除与父级关联的子级可能涉及检索以定位子级。然后剩下的是检索处理的检查，这可能会更复杂一些。

## 搜索、扫描和查询

*不要惊慌；这些只是同义词。我们将交替使用这些词*。

在查看数据库搜索时，我们有两种设计选择。我们可以返回键序列，也可以返回对象序列。由于我们的设计强调在每个对象中存储键，因此从数据库获取对象序列就足够了，因此我们将专注于这种设计。 

搜索本质上是低效的。我们更希望有更有针对性的索引。我们将在下一节中看看如何创建更有用的索引。然而，蛮力扫描的备用计划总是有效的。

当子类具有独立风格的键时，我们可以轻松地使用简单的迭代器扫描所有某个`Child`类的实例的架子。以下是一个定位所有子级的生成器表达式：

```py
children = ( shelf[k] for k in shelf.keys() if key.startswith("Child:") )
```

这会查看架子中的每个键，以选择以`"Child:"`开头的子集。我们可以在此基础上应用更多条件，使用更复杂的生成器表达式：

```py
children_by_title = ( c for c in children if c.title == "some title" )
```

我们使用了嵌套的生成器表达式来扩展初始的`children`查询，添加条件。这样的嵌套生成器表达式在 Python 中非常高效。这不会使数据库进行两次扫描。这是一个带有两个条件的单次扫描。内部生成器的每个结果都会传递给外部生成器以构建结果。

当子类具有依赖风格的键时，我们可以使用更复杂的匹配规则的迭代器在架子中搜索特定父级的子级。以下是一个定位给定父级所有子级的生成器表达式：

```py
children_of = ( shelf[k] for k in shelf.keys() if key.startswith(parent+":Child:") )
```

这种依赖风格的键结构使得在简单循环中特别容易删除父级和所有子级：

```py
for obj in (shelf[k] for k in shelf.keys() if key.startswith(parent)):
    del obj
```

在使用分层"`Parent:` *pid* `:Child:` *cid* "键时，我们在将父级与子级分开时必须小心。使用这种多部分键，我们会看到许多以"Parent:*pid*"开头的对象键。其中一个键将是正确的父级，简单地"`Parent:` *pid*"。其他键将是带有"`Parent:` *pid* `:Child:` *cid*"的子级。我们经常使用这三种条件进行蛮力搜索：

+   `key.startswith("Parent:pid")` 找到父级和子级的并集；这不是常见的要求。

+   `key.startswith("Parent:pid:Child:")` 找到给定父级的子级。我们可以使用正则表达式，如`r"^(Parent:\d+):(Child:\d+)$"`来匹配键。

+   `key.startswith("Parent:pid")` 和 `":Child:"` 键仅找到父级，不包括子级。我们可以使用正则表达式，如`r"^Parent:\d+$"`来匹配键。

所有这些查询都可以通过构建索引来优化。

## 为架子设计访问层

这是应用程序如何使用`shelve`的方式。我们将查看编辑和保存微博帖子的应用程序的各个部分。我们将应用程序分为两个层：应用程序层和数据层。在应用程序层中，我们将区分两个层：

+   **应用程序处理**：这些对象不是持久的。这些类将体现整个应用程序的行为。这些类响应用户选择的命令、菜单项、按钮和其他处理元素。

+   **问题域数据模型**：这些对象将被写入架子。这些对象体现了整个应用程序的状态。

先前显示的博客和帖子的定义之间没有正式的关联。这些类是独立的，因此我们可以在架子上分别处理它们。我们不想通过将`Blog`转换为集合类来创建一个单一的大容器对象。

在数据层中，可能会有许多功能，这取决于数据存储的复杂性。我们将专注于两个功能：

+   **访问**：这些组件提供对问题域对象的统一访问。我们将定义一个`Access`类，它提供对`Blog`和`Post`实例的访问。它还将管理定位架子中的`Blog`和`Post`对象的键。

+   **持久性**：这些组件将问题域对象序列化并写入持久存储。这是`shelve`模块。

我们将`Access`类分成三个独立的部分。这是第一部分，包括文件打开和关闭的各个部分：

```py
import shelve
class Access:
    def new( self, filename ):
        self.database= shelve.open(filename,'n')
        self.max= { 'Post': 0, 'Blog': 0 }
        self.sync()
    def open( self, filename ):
        self.database= shelve.open(filename,'w')
        self.max= self.database['_DB:max']
    def close( self ):
        if self.database:
            self.database['_DB:max']= self.max
            self.database.close()
        self.database= None
    def sync( self ):
        self.database['_DB:max']= self.max
        self.database.sync()
    def quit( self ):
        self.close()
```

对于`Access.new()`，我们将创建一个新的空架子。对于`Access.open()`，我们将打开一个现有的架子。在关闭和同步时，我们确保将当前最大键值的小词典发布到架子中。

我们还没有解决诸如实现“另存为...”方法以复制文件的事情。我们也没有解决不保存退出以恢复到数据库文件的上一个版本的选项。这些附加功能涉及使用`os`模块来管理文件副本。我们为您提供了`close()`和`quit()`方法。这可以使设计 GUI 应用程序稍微简单一些。以下是更新架子中的`Blog`和`Post`对象的各种方法：

```py
def add_blog( self, blog ):
        self.max['Blog'] += 1
        key= "Blog:{id}".format(id=self.max['Blog'])
        blog._id= key
        **self.database[blog._id]= blog
return blog
    def get_blog( self, id ):
        return self.database[id]
    def add_post( self, blog, post ):
        self.max['Post'] += 1
        try:
            key= "{blog}:Post:{id}".format(blog=blog._id,id=self.max['Post'])
        except AttributeError:
            raise OperationError( "Blog not added" )
        post._id= key
        post._blog= blog._id
        **self.database[post._id]= post
return post
    def get_post( self, id ):
        return self.database[id]
    def replace_post( self, post ):
        **self.database[post._id]= post
return post
    def delete_post( self, post ):
        del self.database[post._id]
```

我们提供了一组最小的方法，将`Blog`与其关联的`Post`实例放入架子中。当我们添加`Blog`时，`add_blog()`方法首先计算一个新的键，然后更新`Blog`对象的键，最后将`Blog`对象持久化在架子中。我们已经突出显示了改变架子内容的行。简单地在架子中设置一个项目，类似于在字典中设置一个项目，将使对象持久化。

当我们添加一个帖子时，我们必须提供父`Blog`，以便两者在架子上正确关联。在这种情况下，我们获取`Blog`键，创建一个新的`Post`键，然后更新`Post`的键值。这个更新的`Post`可以持久化在架子上。`add_post()`中的突出行使对象在架子中持久化。

在极少数情况下，如果我们尝试添加`Post`而没有先前添加父`Blog`，我们将会出现属性错误，因为`Blog._id`属性将不可用。

我们提供了代表性的方法来替换`Post`和删除`Post`。还有一些其他可能的操作；我们没有包括替换`Blog`或删除`Blog`的方法。当我们编写删除`Blog`的方法时，我们必须解决防止在仍然有`Posts`时删除或级联删除以包括`Posts`的问题。最后，还有一些搜索方法，作为迭代器来查询`Blog`和`Post`实例：

```py
    def __iter__( self ):
        for k in self.database:
            if k[0] == "_": continue
            yield self.database[k]
    def blog_iter( self ):
        for k in self.database:
            if not k.startswith("Blog:"): continue
            if ":Post:" in k: continue # Skip children
            yield self.database[k]
    def post_iter( self, blog ):
        key= "{blog}:Post:".format(blog=blog._id)
        for k in self.database:
            if not k.startswith(key): continue
            yield self.database[k]
    def title_iter( self, blog, title ):
        return ( p for p in self.post_iter(blog) if p.title == title )
```

我们已经定义了默认迭代器 `__iter__()`，它过滤掉了以 `_` 开头的内部对象。到目前为止，我们只定义了一个这样的键 `_DB:max`，但这个设计给我们留下了发明其他键的空间。

`blog_iter()` 方法遍历 `Blog` 条目。由于 `Blog` 和 `Post` 条目都以 `"Blog:"` 开头，我们必须明确丢弃 `Blog` 的子级 `Post` 条目。一个专门构建的索引对象通常是一个更好的方法。我们将在下一节中讨论这个问题。

`post_iter()` 方法遍历属于特定博客的帖子。`title_iter()` 方法检查与特定标题匹配的帖子。这会检查架子中的每个键，这可能是一个低效的操作。

我们还定义了一个迭代器，它定位在给定博客中具有请求标题的帖子。这是一个简单的生成器函数，它使用 `post_iter()` 方法函数，并且只返回匹配的标题。

### 编写演示脚本

我们将使用技术尖峰来向您展示一个应用程序如何使用这个 `Access` 类来处理微博对象。尖峰脚本将保存一些 `Blog` 和 `Post` 对象到数据库中，以展示应用程序可能使用的一系列操作。这个演示脚本可以扩展为单元测试用例。更完整的单元测试将向我们展示所有功能是否存在并且是否正确工作。这个小的尖峰脚本向我们展示了 `Access` 的工作方式：

```py
from contextlib import closing
with closing( Access() ) as access:
    access.new( 'blog' )
    access.add_blog( b1 )
    # b1._id is set.
    for post in p2, p3:
        access.add_post( b1, post )
        # post._id is set
    b = access.get_blog( b1._id )
    print( b._id, b )
    for p in access.post_iter( b ):
        print( p._id, p )
    access.quit()
```

我们已经在访问层上创建了 `Access` 类，以便它被包装在上下文管理器中。目标是确保访问层被正确关闭，无论可能引发的任何异常。

通过 `Access.new()`，我们创建了一个名为 `'blog'` 的新架子。这可能是通过导航到**文件** **|** **新建**来完成的。我们将新的博客 `b1` 添加到了架子中。`Access.add_blog()` 方法将更新 `Blog` 对象及其架子键。也许有人在页面上填写了一些空白，并在他们的 GUI 应用程序上点击了**新建博客**。

一旦我们添加了 `Blog`，我们可以向其添加两篇帖子。父 `Blog` 条目的键将用于构建每个子 `Post` 条目的键。同样，这个想法是用户填写了一些字段，并在他们的 GUI 上点击了**新建帖子**。

还有一组最终的查询，从架子中转储键和对象。这向我们展示了这个脚本的最终结果。我们可以执行 `Access.get_blog()` 来检索创建的博客条目。我们可以使用 `Access.post_iter()` 遍历属于该博客的帖子。最后的 `Access.quit()` 确保了用于生成唯一键的最大值被记录下来，并且架子被正确关闭。

## 创建索引以提高效率

效率的一个规则是避免搜索。我们之前使用架子中键的迭代器的例子是低效的。更明确地说，搜索*定义了*低效。我们将强调这一点。

### 提示

Brute-force search 可能是处理数据的最糟糕的方式。我们必须始终设计基于子集或映射的索引来提高性能。

为了避免搜索，我们需要创建列出我们想要的项目的索引。这样可以避免通过整个架子来查找项目或子集。架子索引不能引用 Python 对象，因为那样会改变对象存储的粒度。架子索引只能列出键值。这使得对象之间的导航间接，但仍然比在架子中搜索所有项目要快得多。

作为索引的一个例子，我们可以在架子中为每个 `Blog` 关联的 `Post` 键保留一个列表。我们可以很容易地修改 `add_blog()`、`add_post()` 和 `delete_post()` 方法来更新相关的 `Blog` 条目。以下是这些博客更新方法的修订版本：

```py
class Access2( Access ):
    def add_blog( self, blog ):
        self.max['Blog'] += 1
        key= "Blog:{id}".format(id=self.max['Blog'])
        blog._id= key
        **blog._post_list= []
        self.database[blog._id]= blog
        return blog

    def add_post( self, blog, post ):
        self.max['Post'] += 1
        try:
            key= "{blog}:Post:{id}".format(blog=blog._id,id=self.max['Post'])
        except AttributeError:
            raise OperationError( "Blog not added" )
        post._id= key
        post._blog= blog._id
        self.database[post._id]= post
        **blog._post_list.append( post._id )
        **self.database[blog._id]= blog
        return post
    def delete_post( self, post ):
        del self.database[post._id]
        blog= self.database[blog._id]
        **blog._post_list.remove( post._id )
 **self.database[blog._id]= blog

```

`add_blog()`方法确保每个`Blog`都有一个额外的属性`_post_list`。其他方法将更新这个属性，以维护属于`Blog`的每个`Post`的键列表。请注意，我们没有添加`Posts`本身。如果这样做，我们将整个`Blog`合并为一个 shelf 中的单个条目。通过只添加键信息，我们保持了`Blog`和`Post`对象的分离。

`add_post()`方法将`Post`添加到 shelf。它还将`Post._id`附加到`Blog`级别维护的键列表中。这意味着任何`Blog`对象都将具有提供子帖子键序列的`_post_list`。

这个方法对 shelf 进行了两次更新。第一次只是保存了`Post`对象。第二次更新很重要。我们没有试图简单地改变 shelf 中存在的`Blog`对象。我们有意将对象存储到 shelf 中，以确保对象以其更新后的形式持久化。

同样，`delete_post()`方法通过从所属博客的`_post_list`中移除一个未使用的帖子来保持索引的最新状态。与`add_post()`一样，对 shelf 进行了两次更新：`del`语句删除了`Post`，然后更新了`Blog`对象以反映索引的变化。

这个改变深刻地改变了我们对`Post`对象的查询方式。这是搜索方法的修订版本：

```py
    def __iter__( self ):
        for k in self.database:
            if k[0] == "_": continue
            yield self.database[k]
    def blog_iter( self ):
        for k in self.database:
            if not k.startswith("Blog:"): continue
            if ":Post:" in k: continue # Skip children
            yield self.database[k]
    **def post_iter( self, blog ):
 **for k in blog._post_list:
 **yield self.database[k]
    def title_iter( self, blog, title ):
        return ( p for p in self.post_iter(blog) if p.title == title )
```

我们能够用更高效的操作替换`post_iter()`中的扫描。这个循环将根据在`Blog`的`_post_list`属性中保存的键快速产生`Post`对象。我们可以考虑用生成器表达式替换这个`for`语句：

```py
return (self.database[k] for k in blog._post_list)
```

对`post_iter()`方法的这种优化的重点是消除对匹配键的*所有*键的搜索。我们用适当的相关键序列的简单迭代替换了搜索所有键。一个简单的时间测试，交替更新`Blog`和`Post`并将`Blog`呈现为 RST，向我们展示了以下结果：

```py
Access2: 14.9
Access: 19.3
```

如预期的那样，消除搜索减少了处理`Blog`及其各个`Posts`所需的时间。这个变化是巨大的；几乎 25%的处理时间都浪费在搜索上。

### 创建顶层索引

我们为每个`Blog`添加了一个定位属于该`Blog`的`Posts`的索引。我们还可以为 shelf 添加一个顶层索引，以定位所有`Blog`实例。基本设计与之前展示的类似。对于要添加或删除的每个博客，我们必须更新一个索引结构。我们还必须更新迭代器以正确使用索引。这是另一个用于调解访问我们对象的类设计：

```py
class Access3( Access2 ):
    def new( self, *args, **kw ):
        super().new( *args, **kw )
        **self.database['_DB:Blog']= list()

    def add_blog( self, blog ):
        self.max['Blog'] += 1
        key= "Blog:{id}".format(id=self.max['Blog'])
        blog._id= key
        blog._post_list= []
        self.database[blog._id]= blog
        **self.database['_DB:Blog'].append( blog._id )
        return blog

    **def blog_iter( self ):
 **return ( self.database[k] for k in self.database['_DB:Blog']** )
```

在创建新数据库时，我们添加了一个管理对象和一个索引，键为`"_DB:Blog"`。这个索引将是一个列表，我们将在其中存储每个`Blog`条目的键。当我们添加一个新的`Blog`对象时，我们还将使用修订后的键列表更新这个`"_DB:Blog"`对象。我们没有展示删除的实现。这应该是不言自明的。

当我们遍历`Blog`的帖子时，我们使用索引列表，而不是在数据库中对键进行蛮力搜索。以下是性能结果：

```py
Access3: 4.0
Access2: 15.1
Access: 19.4
```

从中我们可以得出结论，*大部分*的处理时间都浪费在对数据库中键的蛮力搜索上。这应该加强这样一个观念，即我们尽可能地避免搜索，将极大地提高程序的性能。

## 添加更多的索引维护

显然，shelf 的索引维护方面可能会增长。对于我们简单的数据模型，我们可以很容易地为`Posts`的标签、日期和标题添加更多的顶层索引。这里是另一个访问层实现，为`Blogs`定义了两个索引。一个索引简单地列出了`Blog`条目的键。另一个索引根据`Blog`的标题提供键。我们假设标题不是唯一的。我们将分三部分介绍这个访问层。这是 CRUD 处理的*创建*部分：

```py
class Access4( Access2 ):
    def new( self, *args, **kw ):
        super().new( *args, **kw )
        self.database['_DB:Blog']= list()
        self.database['_DB:Blog_Title']= defaultdict(list)

    def add_blog( self, blog ):
        self.max['Blog'] += 1
        key= "Blog:{id}".format(id=self.max['Blog'])
        blog._id= key
        blog._post_list= []
        self.database[blog._id]= blog
        self.database['_DB:Blog'].append( blog._id )
        **blog_title= self.database['_DB:Blog_Title']
 **blog_title[blog.title].append( blog._id )
 **self.database['_DB:Blog_Title']= blog_title
        return blog
```

我们添加了两个索引：`Blog`键的简单列表加上`defaultdict`，它为给定标题字符串提供了一个键列表。如果每个标题都是唯一的，那么列表都将是单例的。如果标题不唯一，那么每个标题将有一个`Blog`键列表。

当我们添加一个`Blog`实例时，我们还会更新两个索引。通过追加新键并将其保存到架子上来更新`Blog`键的简单列表。标题索引要求我们从架子上获取现有的`defaultdict`，将其追加到映射到`Blog`标题的键列表中，然后将`defaultdict`放回架子上。下一节向我们展示了 CRUD 处理的*更新*部分：

```py
    def update_blog( self, blog ):
        """Replace this Blog; update index."""
        self.database[blog._id]= blog
        blog_title= self.database['_DB:Blog_Title']
        # Remove key from index in old spot.
        empties= []
        for k in blog_title:
            if blog._id in blog_title[k]:
                blog_title[k].remove( blog._id )
                if len(blog_title[k]) == 0: empties.append( k )
        # Cleanup zero-length lists from defaultdict.
        for k in empties:
            del blog_title[k]
        # Put key into index in new spot.
        **blog_title[blog.title].append( blog._id )
 **self.database['_DB:Blog_Title']= blog_title

```

当我们更新`Blog`对象时，可能会更改`Blog`属性的标题。如果我们的模型有更多属性和更多索引，我们可能希望将修订后的值与架子上的值进行比较，以确定哪些属性已更改。对于这个简单的模型——只有一个属性——不需要比较来确定哪些属性已更改。

操作的第一部分是从索引中删除`Blog`的键。由于我们没有缓存`Blog.title`属性的先前值，所以不能简单地根据旧标题删除键。相反，我们被迫搜索与`Blog`关联的键，并从任何与其关联的标题中删除键。

### 注意

`博客`具有唯一标题将使标题的键列表为空。我们也应该清理未使用的标题。

一旦与旧标题关联的键从索引中删除，我们就可以使用新标题将键追加到索引中。这最后两行与创建`Blog`时使用的代码相同。以下是一些检索处理的示例：

```py
    def blog_iter( self ):
        return ( self.database[k] for k in self.database['_DB:Blog'] )

    def blog_title_iter( self, title ):
        blog_title= self.database['_DB:Blog_Title']
        return ( self.database[k] for k in blog_title[title] )
```

`blog_iter()`方法函数通过从架子上获取索引对象来迭代所有的博客。`blog_title_iter()`方法函数使用索引来获取所有具有给定标题的博客。当有许多单独的博客时，这应该可以很快地按标题找到一个博客。

## 索引更新的写回替代方案

我们可以要求使用`writeback=True`打开一个架子。这将通过保持每个对象的缓存版本来跟踪可变对象的更改。与负担`shelve`模块跟踪所有访问的对象以检测和保留更改不同，这里显示的设计将更新可变对象，并明确强制架子更新对象的持久版本。

这是运行时性能的一个小变化。例如，`add_post()`操作变得稍微更昂贵，因为它还涉及更新`Blog`条目。如果添加了多个`Posts`，这些额外的`Blog`更新将成为一种开销。然而，通过避免对架子键进行漫长的搜索来跟踪给定博客的帖子，这种成本可能会得到平衡。这里显示的设计避免了创建一个在应用程序运行期间可能无限增长的`writeback`缓存。

### 模式演变

在使用`shelve`时，我们必须解决模式演变的问题。我们的对象具有动态状态和静态类定义。我们可以很容易地持久化动态状态。我们的类定义是持久化数据的模式。然而，类并不是*绝对*静态的。如果我们更改类定义，我们将如何从架子上获取对象？一个好的设计通常涉及以下技术的某种组合。

方法函数和属性的更改不会改变持久化对象的状态。我们可以将这些分类为次要更改，因为架子上的数据仍然与更改后的类定义兼容。新软件发布可以有一个新的次要版本号，用户应该有信心它将可以正常工作。

属性的更改将改变持久化的对象。我们可以称这些为重大变化，而存储的数据将不再与新的类定义兼容。这种改变不应该通过*修改*类定义来进行。这种改变应该通过定义一个新的子类，并提供一个更新的工厂函数来创建任何版本的类的实例。

我们可以灵活地支持多个版本，或者我们可以使用一次性转换。为了灵活，我们必须依赖于工厂函数来创建对象的实例。一个灵活的应用程序将避免直接创建对象。通过使用工厂函数，我们可以确保应用程序的所有部分可以一致地工作。我们可能会这样做来支持灵活的模式更改：

```py
def make_blog( *args, **kw ):
    version= kw.pop('_version',1)
    if version == 1: return Blog( *args, **kw )
    elif version == 2: return Blog2( *args, **kw )
    else: raise Exception( "Unknown Version {0}".format(version) )
```

这种工厂函数需要一个`_version`关键字参数来指定使用哪个`Blog`类定义。这允许我们升级模式以使用不同的类，而不会破坏我们的应用程序。`Access`层可以依赖这种函数来实例化正确版本的对象。我们还可以创建一个类似这样的流畅工厂：

```py
class Blog:
    @staticmethod
    def version( self, version ):
        self.version= version
    @staticmethod
    def blog( self, *args, **kw ):
        if self.version == 1: return Blog1( *args, **kw )
        elif self.version == 2: return Blog2( *args, **kw )
        else: raise Exception( "Unknown Version {0}".format(self.version) )
```

我们可以如下使用这个工厂：

```py
blog= Blog.version(2).blog( title=this, other_attribute=that )
```

一个架子应该包括模式版本信息，可能作为一个特殊的`__version__`键。这将为访问层提供信息，以确定应该使用哪个类的版本。应用程序在打开架子后应该首先获取这个对象，并在模式版本错误时快速失败。

对于这种灵活性的替代方案是一次性转换。应用程序的这个特性将使用旧的类定义获取所有存储的对象，转换为新的类定义，并以新格式存储回架子。对于 GUI 应用程序，这可能是打开文件或保存文件的一部分。对于 Web 服务器，这可能是由管理员作为应用程序发布的一部分运行的脚本。

## 摘要

我们已经了解了如何使用`shelve`模块的基础知识。这包括创建一个架子，并设计键来访问我们放在架子上的对象。我们还看到了需要一个访问层来执行架子上的低级 CRUD 操作的需求。这个想法是我们需要区分专注于我们应用程序的类定义和支持持久性的其他管理细节。

### 设计考虑和权衡

`shelve`模块的一个优点是允许我们持久化不同的项目。这给我们带来了一个设计负担，即识别项目的适当粒度。粒度太细，我们会浪费时间从它们的部分组装容器。粒度太粗，我们会浪费时间获取和存储不相关的项目。

由于架子需要一个键，我们必须为我们的对象设计适当的键。我们还必须管理我们各种对象的键。这意味着使用额外的属性来存储键，可能创建额外的键集合来充当架子上项目的索引。

用于访问`shelve`数据库中项目的键就像`weakref`；它是一个间接引用。这意味着需要额外的处理来跟踪和访问引用的项目。有关`weakref`的更多信息，请参见第二章，“与 Python 基本特殊方法无缝集成”。

一个键的选择是定位一个属性或属性组合，这些属性是适当的主键，不能被更改。另一个选择是生成不能被更改的代理键；这允许所有其他属性被更改。由于`shelve`依赖于`pickle`来表示架子上的项目，我们有一个高性能的 Python 对象的本机表示。这减少了设计将放置到架子上的类的复杂性。任何 Python 对象都可以被持久化。

### 应用软件层

由于使用`shelve`时可用的相对复杂性，我们的应用软件必须更加合理地分层。通常，我们将研究具有以下层次结构的软件架构：

+   **表示层**：顶层用户界面，可以是 Web 演示或桌面 GUI。

+   **应用层**：使应用程序工作的内部服务或控制器。这可以称为处理模型，与逻辑数据模型不同。

+   **业务层或** **问题域模型层**：定义业务领域或问题空间的对象。有时被称为逻辑数据模型。我们已经看过如何对这些对象建模，使用微博`Blog`和`Post`的示例。

+   **基础设施**：通常包括几个层次以及其他横切关注点，如日志记录、安全性和网络访问。

+   **数据访问层**。这些是访问数据对象的协议或方法。我们已经研究了设计类来从`shelve`存储中访问我们的应用对象。

+   **持久层**。这是文件存储中看到的物理数据模型。`shelve`模块实现了持久性。

当查看本章和第十一章*通过 SQLite 存储和检索对象*时，清楚地看到，掌握面向对象编程涉及一些更高级的设计模式。我们不能简单地孤立设计类，而是需要考虑类将如何组织成更大的结构。最后，最重要的是，蛮力搜索是一件可怕的事情。必须避免。

### 展望未来

下一章将与本章大致平行。我们将研究使用 SQLite 而不是 shelve 来持久保存我们的对象。复杂之处在于 SQL 数据库没有提供存储复杂 Python 对象的方法，导致阻抗不匹配问题。我们将研究在使用关系数据库（如 SQLite）时解决这个问题的两种方法。

第十二章*传输和共享对象*将把焦点从简单的持久性转移到传输和共享对象。这将依赖于我们在本部分看到的持久性；它将在网络协议中添加。

# 第十一章：通过 SQLite 存储和检索对象

有许多应用程序需要单独持久化对象。我们在第九章*序列化和保存 - JSON、YAML、Pickle、CSV 和 XML*中所研究的技术偏向于处理单个的、整体的对象。有时，我们需要持久化来自更大领域的单独的对象。我们可能会在单个文件结构中保存博客条目、博客帖子、作者和广告。

在第十章*通过 Shelve 存储和检索对象*中，我们研究了在`shelve`数据存储中存储不同的 Python 对象。这使我们能够对大量对象实现 CRUD 处理。任何单个对象都可以在不加载和转储整个文件的情况下进行创建、检索、更新或删除。

在本章中，我们将研究将 Python 对象映射到关系数据库；具体来说，是与 Python 捆绑在一起的`sqlite3`数据库。这将是**三层架构**设计模式的另一个示例**。**

在这种情况下，SQLite 数据层比 Shelve 更复杂。SQLite 可以通过锁定允许并发更新。SQLite 提供了基于 SQL 语言的访问层。它通过将 SQL 表保存到文件系统来实现持久性。Web 应用程序是数据库用于处理对单个数据池的并发更新而不是简单文件持久性的一个例子。RESTful 数据服务器也经常使用关系数据库来提供对持久对象的访问。

为了可扩展性，可以使用独立的数据库服务器进程来隔离所有数据库事务。这意味着它们可以分配给一个相对安全的主机计算机，与 Web 应用服务器分开，并位于适当的防火墙后面。例如，MySQL 可以作为独立的服务器进程实现。SQLite 不是独立的数据库服务器；它必须作为主机应用程序的一部分存在；对于我们的目的，Python 是主机。

## SQL 数据库，持久性和对象

在使用 SQLite 时，我们将使用基于 SQL 语言的关系数据库访问层。SQL 语言是来自对象导向编程稀有时代的遗留。SQL 语言在很大程度上偏向于过程式编程，从而创建了所谓的关系数据模型和对象数据模型之间的阻抗不匹配。在 SQL 数据库中，我们通常关注三个数据建模层，如下所示：

+   **概念模型**：这些是由 SQL 模型隐含的实体和关系。在大多数情况下，这些可以映射到 Python 对象，并应该与应用程序层的数据模型层对应。这是**对象关系映射**层有用的地方。

+   **逻辑模型**：这些是似乎存在于 SQL 数据库中的表、行和列。我们将在我们的 SQL 数据操作语句中处理这些实体。我们说这些似乎存在是因为它们由一个物理模型实现，这个物理模型可能与数据库模式中定义的表、行和列有些不同。例如，SQL 查询的结果看起来像表，但可能不涉及与任何定义的表的存储相平行的存储。

+   **物理模型**：这些是持久物理存储的文件、块、页、位和字节。这些实体由管理 SQL 语句定义。在一些更复杂的数据库产品中，我们可以对数据的物理模型行使一定的控制，以进一步调整性能。然而，在 SQLite 中，我们几乎无法控制这一点。

在使用 SQL 数据库时，我们面临许多设计决策。也许最重要的一个是决定如何处理阻抗不匹配。我们如何处理 SQL 的传统数据模型与 Python 对象模型之间的映射？有三种常见的策略：

+   **不映射到 Python**：这意味着我们不从数据库中获取复杂的 Python 对象，而是完全在独立的原子数据元素和处理函数的 SQL 框架内工作。这种方法将避免对持久数据库对象的面向对象编程的深度强调。这将限制我们使用四种基本的 SQLite 类型 NULL、INTEGER、REAL 和 TEXT，以及 Python 的`datetime.date`和`datetime.datetime`的添加。

+   **手动映射**：我们定义一个访问层，用于在我们的类定义和 SQL 逻辑模型之间进行映射，包括表、列、行和键。

+   **ORM 层**：我们下载并安装一个 ORM 层来处理类和 SQL 逻辑模型之间的映射。

我们将在以下示例中查看所有三种选择。在我们可以查看从 SQL 到对象的映射之前，我们将详细查看 SQL 逻辑模型，并在此过程中涵盖无映射选项。

### SQL 数据模型 - 行和表

SQL 数据模型基于具有命名列的命名表。表包含多行数据。每一行都有点像可变的`namedtuple`。整个表就像`list`。

当我们定义一个 SQL 数据库时，我们定义表及其列。当我们使用 SQL 数据库时，我们操作表中的数据行。在 SQLite 的情况下，我们有一个狭窄的数据类型领域，SQL 将处理这些数据类型。SQLite 处理`NULL`，`INTEGER`，`REAL`，`TEXT`和`BLOB`数据。Python 类型`None`，`int`，`float`，`str`和`bytes`被映射到这些 SQL 类型。同样，当从 SQLite 数据库中获取这些类型的数据时，这些项目将被转换为 Python 对象。

我们可以通过向 SQLite 添加更多的转换函数来调解这种转换。`sqlite3`模块以这种方式添加了`datetime.date`和`datetime.datetime`扩展。我们将在下一节中介绍手动映射。

SQL 语言可以分为三个子语言：**数据定义语言**（**DDL**），**数据操作语言**（**DML**）和**数据控制语言**（**DCL**）。DDL 用于定义表、它们的列和索引。例如，我们可能以以下方式定义一些表：

```py
CREATE TABLE BLOG(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    TITLE TEXT );
CREATE TABLE POST(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    DATE TIMESTAMP,
    TITLE TEXT,
    RST_TEXT TEXT,
    BLOG_ID INTEGER REFERENCES BLOG(ID)  );
CREATE TABLE TAG(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    PHRASE TEXT UNIQUE ON CONFLICT FAIL );
CREATE TABLE ASSOC_POST_TAG(
  POST_ID INTEGER REFERENCES POST(ID),
  TAG_ID INTEGER REFERENCES TAG(ID) );
```

我们创建了四个表来表示微博应用程序的`Blog`和`Post`对象。有关 SQLite 处理的 SQL 语言的更多信息，请参阅[`www.sqlite.org/lang.html`](http://www.sqlite.org/lang.html)。对于 SQL 的更广泛背景，像*Creating your MySQL Database: Practical Design Tips and Techniques*这样的书籍将介绍 MySQL 数据库上下文中的 SQL 语言。SQL 语言是不区分大小写的。出于没有好的理由，我们更喜欢看到 SQL 全部大写，以区别于周围的 Python 代码。

`BLOG`表定义了一个带有`AUTOINCREMENT`选项的主键；这将允许 SQLite 分配键值，使我们不必在代码中生成键。`TITLE`列是博客的标题。我们将其定义为`TEXT`。在一些数据库产品中，我们必须提供最大大小；在 SQLite 中，这是不需要的，所以我们将避免混乱。

`POST`表定义了一个主键，以及日期，标题和 RST 文本作为帖子正文。请注意，在此表定义中我们没有引用标签。我们将在后续 SQL 表所需的设计模式中返回。然而，`POST`表包括一个正式的`REFERENCES`子句，以向我们显示这是对拥有`BLOG`的外键引用。`TAG`表定义了单个标签文本项，没有其他内容。

最后，我们有一个`POST`和`TAG`之间的关联表。这个表只有两个外键。它关联标签和帖子，允许每个帖子有无限数量的标签，以及无限数量的帖子共享一个公共标签。这种关联表是处理这种关系的常见 SQL 设计模式。我们将在下一节中看一些其他 SQL 设计模式。我们可以执行上述定义来创建我们的数据库：

```py
import sqlite3
database = sqlite3.connect('p2_c11_blog.db')
database.executescript( sql_ddl )
```

所有数据库访问都需要一个连接，使用模块函数`sqlite3.connect()`创建。我们提供了要分配给我们的数据库的文件名。我们将在单独的部分中查看此函数的其他参数。

DB-API 假设我们的应用程序进程连接到一个单独的数据库服务器进程。在 SQLite 的情况下，实际上并没有单独的进程。但是，为了符合标准，我们使用`connect()`函数。

`sql_ddl`变量只是一个长字符串变量，其中包含四个`CREATE TABLE`语句。如果没有错误消息，那么表结构已经定义。

`Connection.executescript()`方法在 Python 标准库中被描述为*非标准快捷方式*。从技术上讲，数据库操作涉及`cursor`。以下是一种标准化的方法：

```py
crsr = database.cursor()
for stmt in sql_ddl.split(";"):
    crsr.execute(stmt)
```

由于我们专注于 SQLite，我们将大量使用非标准快捷方式。如果我们关心对其他数据库的可移植性，我们将把重点转移到更严格地遵守 DB-API。在下一节中，当查看查询时，我们将回到游标对象的性质。

### 通过 SQL DML 语句进行 CRUD 处理

以下四个经典的 CRUD 操作直接映射到 SQL 语句：

+   创建是通过`INSERT`语句完成的

+   检索是通过`SELECT`语句完成的

+   更新是通过`UPDATE`语句以及`REPLACE`语句（如果支持）来完成的

+   删除是通过`DELETE`语句完成的。

我们必须注意，有一种字面的 SQL 语法，以及带有绑定变量占位符而不是字面值的语法。字面的 SQL 语法适用于脚本；然而，因为值始终是字面的，它对应用程序编程来说非常糟糕。在应用程序中构建字面的 SQL 语句涉及无休止的字符串操作和著名的安全问题。请参阅[`xkcd.com/327/`](http://xkcd.com/327/)，了解组装字面 SQL 的特定安全问题。我们将专注于带有绑定变量的 SQL。

字面 SQL 被广泛使用，这是一个错误。

### 注意

永远不要使用字符串操作构建字面的 SQL DML 语句。

Python DB-API 接口，**Python Enhancement Proposal**（**PEP**）249，[`www.python.org/dev/peps/pep-0249/`](http://www.python.org/dev/peps/pep-0249/)，定义了将应用程序变量绑定到 SQL 语句中的几种方法。SQLite 可以使用带有`?`的位置绑定或带有`:name`的命名绑定。我们将向您展示这两种绑定变量的样式。

我们使用`INSERT`语句来创建一个新的`BLOG`行，如下面的代码片段所示：

```py
create_blog= """
INSERT INTO BLOG(TITLE) VALUES(?)
"""
database.execute(create_blog, ("Travel Blog",))
```

我们创建了一个带有位置绑定变量`?`的 SQL 语句，用于`BLOG`表的`TITLE`列。然后，在将一组值绑定到绑定变量后，执行该语句。只有一个绑定变量，所以元组中只有一个值。执行完语句后，数据库中就有一行数据。

我们清楚地将 SQL 语句与周围的 Python 代码分开，使用三引号的长字符串文字。在一些应用程序中，SQL 被存储为单独的配置项。将 SQL 保持分开最好是作为从语句名称到 SQL 文本的映射来处理。例如，我们可以将 SQL 保存在 JSON 文件中。这意味着我们可以使用`SQL=json.load("sql_config.json")`来获取所有 SQL 语句。然后，我们可以使用`SQL["some statement name"]`来引用特定 SQL 语句的文本。这可以通过将 SQL 从 Python 编程中分离出来，简化应用程序的维护。

`DELETE`和`UPDATE`语句需要`WHERE`子句来指定将更改或删除哪些行。要更改博客的标题，我们可以这样做：

```py
update_blog="""
UPDATE BLOG SET TITLE=:new_title WHERE TITLE=:old_title
"""
database.execute( "BEGIN" )
database.execute( update_blog,
    dict(new_title="2013-2014 Travel", old_title="Travel Blog") )
database.commit()
```

`UPDATE`语句有两个命名绑定变量：`:new_title`和`:old_title`。此事务将更新`BLOG`表中具有给定旧标题的所有行，将标题设置为新标题。理想情况下，标题是唯一的，只有一个行受到影响。SQL 操作被定义为对一组行进行操作。确保所需行是集合的内容是数据库设计的问题。因此，建议为每个表设置唯一的主键。

在实现删除操作时，我们总是有两种选择。我们可以在子项仍然存在时禁止删除父项，或者我们可以级联删除父项以同时删除相关的子项。我们将看一下`Blog`，`Post`和标签关联的级联删除。以下是`DELETE`语句的序列：

```py
delete_post_tag_by_blog_title= """
DELETE FROM ASSOC_POST_TAG
WHERE POST_ID IN (
    SELECT DISTINCT POST_ID
    FROM BLOG JOIN POST ON BLOG.ID = POST.BLOG_ID
    WHERE BLOG.TITLE=:old_title)
"""
delete_post_by_blog_title= """
DELETE FROM POST WHERE BLOG_ID IN (
    SELECT ID FROM BLOG WHERE TITLE=:old_title)
"""
delete_blog_by_title="""
DELETE FROM BLOG WHERE TITLE=:old_title
"""
try:
    with database:
        title= dict(old_title="2013-2014 Travel")
        database.execute( delete_post_tag_by_blog_title, title )
        database.execute( delete_post_by_blog_title, title )
        database.execute( delete_blog_by_title, title )
    print( "Delete finished normally." )
except Exception as e:
    print( "Rolled Back due to {0}".format(e) )
```

我们进行了一个三步删除操作。首先，我们根据标题从给定的`Blog`中删除了`ASSOC_POST_TAG`的所有行。注意嵌套查询；我们将在下一节中讨论查询。在 SQL 构造中，表之间的导航是一个常见的问题。在这种情况下，我们必须查询`BLOG-POST`关系以定位将被移除的`POST` ID；然后，我们可以删除与将被移除的博客相关联的帖子的`ASSOC_POST_TAG`行。接下来，我们删除了属于特定博客的所有帖子。这也涉及到一个嵌套查询，以定位基于标题的博客的 ID。最后，我们可以删除博客本身。

这是一个显式级联删除设计的示例，我们需要将操作从`BLOG`表级联到另外两个表。我们将所有删除操作包装在`with`上下文中，以便它作为一个单独的事务提交。在失败的情况下，它将回滚部分更改，使数据库保持原样。

### 使用 SQL SELECT 语句查询行

单单关于`SELECT`语句就可以写一本大部头的书。我们将跳过除了`SELECT`最基本的特性之外的所有内容。我们的目的是只涵盖足够的 SQL 来存储和检索数据库中的对象。

之前，我们提到，从技术上讲，在执行 SQL 语句时，我们应该使用游标。对于 DDL 和其他 DML 语句，游标的存在与否并不太重要。我们将使用显式创建游标，因为它极大地简化了 SQL 编程。

然而，对于查询来说，游标对于从数据库中检索行是必不可少的。要通过标题查找博客，我们可以从以下简单的代码开始：

```py
"SELECT * FROM BLOG WHERE TITLE=?"
```

我们需要获取结果行对象的集合。即使我们期望作为响应的是一行，但在 SQL 世界中，一切都是一个集合。通常，从`SELECT`查询的每个结果集看起来都像是由`SELECT`语句定义的行和列的表，而不是任何`CREATE TABLE` DDL。

在这种情况下，使用`SELECT *`意味着我们避免了枚举预期结果列。这可能导致检索到大量列。以下是使用 SQLite 快捷方式进行此操作的常见优化：

```py
query_blog_by_title= """
SELECT * FROM BLOG WHERE TITLE=?
"""
for blog in database.execute( query_blog_by_title, ("2013-2014 Travel",) ):
    print( blog[0], blog[1] )
```

在`SELECT`语句中，`*`是所有可用列的简写。它只对涉及单个表的简单查询真正有用。

我们将请求的博客标题绑定到`SELECT`语句中的"`?`"参数。`execute()`函数的结果是一个游标对象。游标是可迭代的；它将产生结果集中的所有行和匹配`WHERE`子句中选择条件的所有行。

为了完全符合 Python DB-API 标准，我们可以将其分解为以下步骤：

```py
crsr= database.cursor()
crsr.execute( query_blog_by_title, ("2013-2014 Travel",) )
for blog in crsr.fetchall():
    print( blog[0], blog[1] )
```

这向我们展示了如何使用连接来创建一个游标对象。然后我们可以使用游标对象执行查询语句。一旦我们执行了查询，我们就可以获取结果集中的所有行。每一行都将是来自`SELECT`子句的值的元组。在这种情况下，由于`SELECT`子句是`*`，这意味着将使用原始`CREATE TABLE`语句中的所有列。

### SQL 事务和 ACID 属性

正如我们所见，SQL DML 语句映射到 CRUD 操作。在讨论 SQL 事务的特性时，我们将看到`INSERT`、`SELECT`、`UPDATE`和`DELETE`语句的序列。

SQL DML 语句都在 SQL 事务的上下文中工作。在事务中执行的 SQL 语句是一个逻辑工作单元。整个事务可以作为一个整体提交，或者作为一个整体回滚。这支持原子性属性。

SQL DDL 语句（即`CREATE`，`DROP`）不能在事务中工作。它们隐式结束了任何先前的正在进行的事务。毕竟，它们正在改变数据库的结构；它们是一种不同类型的语句，事务概念不适用。

ACID 属性是原子性、一致性、隔离性和持久性。这些是由多个数据库操作组成的事务的基本特性。有关更多信息，请参见第十章*通过 Shelve 存储和检索对象*。

除非在特殊的**读取未提交**模式下工作，否则对数据库的每个连接都只能看到包含已提交事务结果的一致版本的数据。未提交的事务通常对其他数据库客户端进程不可见，支持一致性属性。

SQL 事务还支持隔离属性。SQLite 支持几种不同的**隔离级别**设置。隔离级别定义了 SQL DML 语句在多个并发进程中的交互方式。这是基于锁的使用方式以及进程的 SQL 请求等待锁的方式。从 Python 中，隔离级别在连接到数据库时设置。

每个 SQL 数据库产品对隔离级别和锁定采取不同的方法。没有单一的模型。

在 SQLite 的情况下，有四个隔离级别定义了锁定和事务的性质。有关详细信息，请参见[`www.sqlite.org/isolation.html`](http://www.sqlite.org/isolation.html)。以下是隔离级别：

+   `isolation_level=None`：这是默认值，也称为**自动提交**模式。在这种模式下，每个单独的 SQL 语句在执行时都会提交到数据库。这会破坏原子性，除非出现一些奇怪的巧合，所有事务都只涉及单个 SQL 语句。

+   `isolation_level='DEFERRED'`：在这种模式下，锁在事务中尽可能晚地获取。例如，`BEGIN`语句不会立即获取任何锁。其他读操作（即`SELECT`语句）将获取共享锁。写操作将获取保留锁。虽然这可以最大程度地提高并发性，但也可能导致竞争进程之间的死锁。

+   `isolation_level='IMMEDIATE'`：在这种模式下，事务`BEGIN`语句获取一个阻止所有写入的锁。但读取将继续进行。

+   `isolation_level='EXCLUSIVE'`：在这种模式下，事务`BEGIN`语句获取一个阻止几乎所有访问的锁。对于处于特殊读取未提交模式的连接，它们忽略锁定有一个例外。

对于所有已提交的事务，持久性属性是得到保证的。数据被写入数据库文件。

SQL 规则要求我们执行`BEGIN TRANSACTION`和`COMMIT TRANSACTION`语句来框定一系列步骤。在出现错误的情况下，需要执行`ROLLBACK TRANSACTION`语句来撤销潜在的更改。Python 接口简化了这一过程。我们可以执行`BEGIN`语句。其他语句作为`sqlite3.Connection`对象的函数提供；我们不执行 SQL 语句来结束事务。我们可能会编写诸如以下代码来明确表示：

```py
database = sqlite3.connect('p2_c11_blog.db', isolation_level='DEFERRED')
try:
    database.execute( 'BEGIN' )
    database.execute( "some statement" )
    database.execute( "another statement" )
    database.commit()
except Exception as e:
    database.rollback()
    raise e
```

我们在建立数据库连接时选择了`DEFERRED`的隔离级别。这导致我们需要明确开始和结束每个事务。一个典型的场景是将相关的 DML 包装在`try`块中，如果事情顺利，则提交事务，或者在出现问题的情况下回滚事务。我们可以通过使用`sqlite3.Connection`对象作为上下文管理器来简化这个过程：

```py
database = sqlite3.connect('p2_c11_blog.db', isolation_level='DEFERRED')
with database:
    database.execute( "some statement" )
    database.execute( "another statement" )
```

这与先前的例子类似。我们以相同的方式打开了数据库。我们没有执行显式的`BEGIN`语句，而是进入了一个上下文；上下文为我们处理了`Begin`。

在`with`上下文的末尾，`database.commit()`将自动完成。在发生异常时，将执行`database.rollback()`，并且异常将由`with`语句引发。

### 设计主键和外键

SQL 表不需要特定的主键。然而，对于给定表的行，省略主键是相当糟糕的设计。正如我们在第十章中所指出的，*通过 Shelve 存储和检索对象*，可能有一个属性（或属性的组合）可以成为适当的主键。也完全有可能没有属性适合作为主键，我们必须定义代理键。

之前的例子使用了 SQLite 创建的代理键。这可能是最简单的设计，因为它对数据施加了最少的约束。一个约束是主键不能被更新；这成为应用程序编程必须强制执行的规则。在某些情况下，例如在主键值中纠正错误时，我们需要以某种方式更新主键。做到这一点的一种方法是删除并重新创建约束。另一种方法是删除有错误的行，并重新插入具有更正键的行。当存在级联删除时，用于纠正主键的事务可能变得非常复杂。使用代理键可以防止这类问题。

所有表之间的关系都是通过主键和外键引用来完成的。关系有两种极为常见的设计模式。前面的表向我们展示了这两种主要的设计模式。关系有三种设计模式，如下符号列表所示：

+   一对多：这种关系是一个父博客和许多子帖子之间的关系。`REFERENCES`子句向我们展示了`POST`表中的许多行将引用`BLOG`表中的一行。如果从子到父的方向来看，它将被称为多对一关系。

+   多对多：这种关系是许多帖子和许多标签之间的关系。这需要在`POST`和`TAG`表之间有一个中间关联表；中间表有两个（或更多）外键。多对多关联表也可以有自己的属性。

+   一对一：这种关系是一种较少见的设计模式。与一对多关系没有技术上的区别；零行或一行的基数是应用程序必须管理的约束。

在数据库设计中，关系可能会有约束：关系可能被描述为可选或强制性；关系可能有基数限制。有时，这些可选性和基数约束会用简短的描述来总结，比如“0:m”表示“零到多个”或“可选的一对多”。可选性和基数约束是应用程序编程逻辑的一部分；在 SQLite 数据库中没有正式的方法来说明这些约束。基本表关系可以以以下一种或两种方式在数据库中实现：

+   显式：我们可以称这些为声明，因为它们是数据库的 DDL 声明的一部分。理想情况下，它们由数据库服务器强制执行，不遵守关系约束可能会导致某种错误。这些关系也将在查询中重复。

+   隐式：这些关系仅在查询中说明；它们不是 DDL 的正式部分。

请注意，我们的表定义实现了博客和该博客中各个条目之间的一对多关系。我们在编写的各种查询中使用了这些关系。

## 使用 SQL 处理应用程序数据

前几节中的示例向我们展示了我们可以称之为**过程式**SQL 处理。我们避免了从问题域对象中使用任何面向对象的设计。我们不是使用`Blog`和`Post`对象，而是使用 SQLite 可以处理的数据元素：字符串、日期、浮点数和整数值。我们主要使用了过程式风格的编程。

我们可以看到一系列查询可以用来定位一个博客，所有属于该博客的帖子，以及与与博客相关联的帖子相关联的所有标签。处理看起来像下面的代码：

```py
query_blog_by_title= """
SELECT * FROM BLOG WHERE TITLE=?
"""
query_post_by_blog_id= """
SELECT * FROM POST WHERE BLOG_ID=?
"""
query_tag_by_post_id= """
SELECT TAG.*
FROM TAG JOIN ASSOC_POST_TAG ON TAG.ID = ASSOC_POST_TAG.TAG_ID
WHERE ASSOC_POST_TAG.POST_ID=?
"""
for blog in database.execute( query_blog_by_title, ("2013-2014 Travel",) ):
    print( "Blog", blog )
    for post in database.execute( query_post_by_blog_id, (blog[0],) ):
        print( "Post", post )
        for tag in database.execute( query_tag_by_post_id, (post[0],) ):
            print( "Tag", tag )
```

我们定义了三个 SQL 查询。第一个将按标题获取博客。对于每个博客，我们获取属于该博客的所有帖子。最后，我们获取与给定帖子相关联的所有标签。

第二个查询隐含地重复了`POST`表和`BLOG`表之间的`REFERENCES`定义。我们正在查找特定博客父级的子帖子；我们需要在查询过程中重复一些表定义。

第三个查询涉及`ASSOC_POST_TAG`表的行和`TAG`表之间的关系连接。`JOIN`子句重述了表定义中的外键引用。`WHERE`子句也重复了表定义中的`REFERENCES`子句。

因为第三个查询中连接了多个表，使用`SELECT *`将产生所有表的列。我们实际上只对`TAG`表的属性感兴趣，所以我们使用`SELECT TAG.*`只产生所需的列。

这些查询为我们提供了数据的所有单独的部分。然而，这些查询并没有为我们重建 Python 对象。如果我们有更复杂的类定义，我们必须从检索到的单个数据片段构建对象。特别是，如果我们的 Python 类定义有重要的方法函数，我们需要更好的 SQL 到 Python 映射来利用更完整的 Python 类定义。

### 在纯 SQL 中实现类似类的处理

让我们看一个更复杂的`Blog`类的定义。这个定义是从第九章中重复的，我们突出显示了一个感兴趣的方法函数：

```py
from collections import defaultdict
class Blog:
    def __init__( self, title, *posts ):
        self.title= title
        self.entries= list(posts)
    def append( self, post ):
        self.entries.append(post)
    **def by_tag(self):
 **tag_index= defaultdict(list)
 **for post in self.entries:
 **for tag in post.tags:
 **tag_index[tag].append( post )
 **return tag_index
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= "="*len(self.title),
            entries= [p.as_dict() for p in self.entries],
        )
```

博客的`Blog.by_tag()`功能将成为一个相当复杂的 SQL 查询。作为面向对象的编程，它只是遍历`Post`实例的集合，创建`defaultdict`，将每个标签映射到共享该标签的`Posts`序列。以下是一个产生类似结果的 SQL 查询：

```py
query_by_tag="""
SELECT TAG.PHRASE, POST.TITLE, POST.ID
FROM TAG JOIN ASSOC_POST_TAG ON TAG.ID = ASSOC_POST_TAG.TAG_ID
JOIN POST ON POST.ID = ASSOC_POST_TAG.POST_ID
JOIN BLOG ON POST.BLOG_ID = BLOG.ID
WHERE BLOG.TITLE=?
"""
```

这个查询的结果集是一个类似表的行序列，有三个属性：`TAG.PHRASE`、`POST.TITLE`和`POST.ID`。每个`POST`标题和`POST` ID 都将与所有相关的`TAG`短语重复。为了将其转换为一个简单的、HTML 友好的索引，我们需要将所有具有相同`TAG.PHRASE`的行分组到一个辅助列表中，如下面的代码所示：

```py
tag_index= defaultdict(list)
for tag, post_title, post_id in database.execute( query_by_tag, ("2013-2014 Travel",) ):
    tag_index[tag].append( (post_title, post_id) )
print( tag_index )
```

这个额外的处理将`POST`标题和`POST` ID 的两元组分组成一个有用的结构，可以用来生成 RST 和 HTML 输出。SQL 查询加上相关的 Python 处理非常长 - 比本地面向对象的 Python 更长。

更重要的是，SQL 查询与表定义是分离的。SQL 不是一种面向对象的编程语言。没有整洁的类来捆绑数据和处理在一起。像这样使用 SQL 的过程式编程有效地关闭了面向对象的编程。从严格的面向对象编程的角度来看，我们可以将其标记为“失败”。

有一种观点认为，这种 SQL-heavy、无对象编程对于某些问题比 Python 更合适。通常，这些问题涉及 SQL 的`GROUP BY`子句。虽然在 SQL 中很方便，但 Python 的`defaultdict`和`Counter`也实现得非常有效。Python 版本通常如此有效，以至于使用`defaultdict`查询大量行的小程序可能比使用`GROUP BY`的数据库服务器更快。如果有疑问，请测量。当数据库管理员力主 SQL 魔法般更快时，请测量。

## 将 Python 对象映射到 SQLite BLOB 列

我们可以将 SQL 列映射到类定义，以便我们可以从数据库中的数据创建适当的 Python 对象实例。SQLite 包括一个**二进制大对象**（**BLOB**）数据类型。我们可以将我们的 Python 对象进行 pickle 并将其存储在 BLOB 列中。我们可以计算出我们的 Python 对象的字符串表示（例如，使用 JSON 或 YAML 表示法）并使用 SQLite 文本列。

这种技术必须谨慎使用，因为它实际上破坏了 SQL 处理。BLOB 列不能用于 SQL DML 操作。我们不能对其进行索引或在 DML 语句的搜索条件中使用它。

SQLite BLOB 映射应该保留给那些可以对周围 SQL 处理不透明的对象。最常见的例子是媒体对象，如视频、静态图像或声音片段。SQL 偏向于文本和数字字段。它通常不处理更复杂的对象。

如果我们处理财务数据，我们的应用程序应该使用`decimal.Decimal`值。我们可能希望使用这种数据在 SQL 中进行查询或计算。由于`decimal.Decimal`不受 SQLite 直接支持，我们需要扩展 SQLite 以处理这种类型的值。

这有两个方向：转换和适应。我们需要**适应**Python 数据到 SQLite，我们需要**转换**SQLite 数据回到 Python。以下是两个函数和注册它们的请求：

```py
import decimal

def adapt_currency(value):
    return str(value)
sqlite3.register_adapter(decimal.Decimal, adapt_currency)

def convert_currency(bytes):
    return decimal.Decimal(bytes.decode())
sqlite3.register_converter("DECIMAL", convert_currency)
```

我们编写了一个`adapt_currency()`函数，它将`decimal.Decimal`对象调整为适合数据库的形式。在这种情况下，我们只是简单地将其转换为字符串。我们注册了适配器函数，以便 SQLite 的接口可以使用注册的适配器函数转换`decimal.Decimal`类的对象。

我们还编写了一个`convert_currency()`函数，它将 SQLite 字节对象转换为 Python 的`decimal.Decimal`对象。我们注册了`converter`函数，以便`DECIMAL`类型的列将被正确转换为 Python 对象。

一旦我们定义了适配器和转换器，我们就可以将`DECIMAL`作为一个完全支持的列类型。为了使其正常工作，我们必须通过在建立数据库连接时设置`detect_types=sqlite3.PARSE_DECLTYPES`来通知 SQLite。以下是使用我们的新列数据类型的表定义：

```py
CREATE TABLE BUDGET(
    year INTEGER,
    month INTEGER,
    category TEXT,
    amount DECIMAL
)
```

我们可以像这样使用我们的新列定义：

```py
database= sqlite3.connect( 'p2_c11_blog.db', detect_types=sqlite3.PARSE_DECLTYPES )
database.execute( decimal_ddl )

insert_budget= """
INSERT INTO BUDGET(year, month, category, amount) VALUES(:year, :month, :category, :amount)
"""
database.execute( insert_budget,
    dict(year=2013, month=1, category="fuel", amount=decimal.Decimal('256.78')) )
database.execute( insert_budget,
    dict(year=2013, month=2, category="fuel", amount=decimal.Decimal('287.65')) )

query_budget= """
SELECT * FROM BUDGET
"""
for row in database.execute( query_budget ):
    print( row )
```

我们创建了一个需要通过转换器函数映射声明类型的数据库连接。一旦我们有了连接，我们可以使用新的`DECIMAL`列类型创建我们的表。

当我们向表中插入行时，我们使用适当的`decimal.Decimal`对象。当我们从表中获取行时，我们会发现我们从数据库中得到了适当的`decimal.Decimal`对象。以下是输出：

```py
(2013, 1, 'fuel', Decimal('256.78'))
(2013, 2, 'fuel', Decimal('287.65'))
```

这向我们表明我们的`decimal.Decimal`对象已经被正确存储和从数据库中恢复。我们可以为任何 Python 类编写适配器和转换器。我们需要发明适当的字节表示。由于字符串很容易转换为字节，创建字符串通常是最简单的方法。

## 手动将 Python 对象映射到数据库行

我们可以将 SQL 行映射到类定义，以便我们可以从数据库中的数据创建适当的 Python 对象实例。如果我们对数据库和类定义小心，这并不是不可能的复杂。然而，如果我们粗心大意，我们可能会创建 SQL 表示非常复杂的 Python 对象。复杂性的一个后果是在对象和数据库行之间的映射中涉及大量查询。挑战在于在面向对象设计和 SQL 数据库施加的约束之间取得平衡。

我们将不得不修改我们的类定义，使其更加了解 SQL 实现。我们将对第十章中显示的`Blog`和`Post`类设计进行几处修改，*通过 Shelve 存储和检索对象*。

以下是`Blog`类的定义：

```py
from collections import defaultdict
class Blog:
    def __init__( self, **kw ):
        """Requires title"""
        self.id= kw.pop('id', None)
        self.title= kw.pop('title', None)
        if kw: raise TooManyValues( kw )
        **self.entries= list() # ???
    def append( self, post ):
        self.entries.append(post)
    def by_tag(self):
        tag_index= defaultdict(list)
        **for post in self.entries: # ???
            for tag in post.tags:
                tag_index[tag].append( post )
        return tag_index
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= "="*len(self.title),
            entries= [p.as_dict() for p in self.entries],
        )
```

我们允许数据库 ID 作为对象的一部分。此外，我们已经修改了初始化，使其完全基于关键字。每个关键字值都从`kw`参数中弹出。任何额外的值都会引发`TooManyValues`异常。

我们有两个之前未回答的问题。我们如何处理与博客相关联的帖子列表？我们将修改以下类以添加此功能。以下是`Post`类定义：

```py
import datetime
class Post:
    def __init__( self, **kw ):
        """Requires date, title, rst_text."""
        self.id= kw.pop('id', None)
        self.date= kw.pop('date', None)
        self.title= kw.pop('title', None)
        self.rst_text= kw.pop('rst_text', None)
        self.tags= list()
        if kw: raise TooManyValues( kw )
    def append( self, tag ):
        self.tags.append( tag )
    def as_dict( self ):
        return dict(
            date= str(self.date),
            title= self.title,
            underline= "-"*len(self.title),
            rst_text= self.rst_text,
            tag_text= " ".join(self.tags),
        )
```

与`Blog`一样，我们允许数据库 ID 作为对象的一部分。此外，我们已经修改了初始化，使其完全基于关键字。以下是异常类定义：

```py
class TooManyValues( Exception ):
    pass
```

一旦我们有了这些类定义，我们就可以编写一个访问层，将这些类的对象和数据库之间的数据移动。访问层实现了将 Python 类转换和适应为数据库表中的行的更复杂版本。

### 为 SQLite 设计访问层

对于这个小的对象模型，我们可以在一个类中实现整个访问层。这个类将包括对每个持久类执行 CRUD 操作的方法。在更大的应用程序中，我们可能需要将访问层分解为每个持久类的单独**策略**类。然后，我们将统一所有这些类在一个单一的访问层**Facade**或**Wrapper**下。

这个例子不会痛苦地包括完整访问层的所有方法。我们将向您展示重要的方法。我们将把这分解成几个部分来处理`Blogs`，`Posts`和迭代器。这是我们访问层的第一部分：

```py
class Access:
    get_last_id= """
    SELECT last_insert_rowid()
    """
    def open( self, filename ):
        self.database= sqlite3.connect( filename )
        self.database.row_factory = sqlite3.Row
    def get_blog( self, id ):
        query_blog= """
        SELECT * FROM BLOG WHERE ID=?
        """
        row= self.database.execute( query_blog, (id,) ).fetchone()
        blog= Blog( id= row['ID'], title= row['TITLE'] )
        return blog
    def add_blog( self, blog ):
        insert_blog= """
        INSERT INTO BLOG(TITLE) VALUES(:title)
        """
        self.database.execute( insert_blog, dict(title=blog.title) )
        row = self.database.execute( get_last_id ).fetchone()
        blog.id= row[0]
        return blog
```

这个类将`Connection.row_factory`设置为使用`sqlite3.Row`类，而不是简单的元组。`Row`类允许通过数字索引和列名访问。

`get_blog()`方法从获取的数据库行构造一个`Blog`对象。因为我们使用`sqlite3.Row`对象，我们可以通过名称引用列。这澄清了 SQL 和 Python 类之间的映射。

`add_blog()`方法根据`Blog`对象向`BLOG`表中插入一行。这是一个两步操作。首先，我们创建新行。然后，我们执行 SQL 查询以获取分配给该行的行 ID。

请注意，我们的表定义使用`INTEGER PRIMARY KEY AUTOINCREMENT`。因此，表的主键将匹配行 ID，并且分配的行 ID 将通过`last_insert_rowid()`函数可用。这允许我们检索分配的行 ID；然后我们可以将其放入 Python 对象以供将来参考。以下是我们如何从数据库中检索单个`Post`对象：

```py
    def get_post( self, id ):
        query_post= """
        SELECT * FROM POST WHERE ID=?
        """
        row= self.database.execute( query_post, (id,) ).fetchone()
        post= Post( id= row['ID'], title= row['TITLE'],
            date= row['DATE'], rst_text= row['RST_TEXT'] )
        query_tags= """
        SELECT TAG.*
        FROM TAG JOIN ASSOC_POST_TAG ON TAG.ID = ASSOC_POST_TAG.TAG_ID
        WHERE ASSOC_POST_TAG.POST_ID=?
        """
        results= self.database.execute( query_tags, (id,) )
        for id, tag in results:
            post.append( tag )
        return post
```

为了构建`Post`，我们有两个查询：首先，我们从`POST`表中获取一行，以构建`Post`对象的一部分。然后，我们获取与`TAG`表中的行连接的关联行。这用于构建`Post`对象的标签列表。

当我们保存`Post`对象时，它将有几个部分。必须向`POST`表添加一行。此外，还需要向`ASSOC_POST_TAG`表添加行。如果标签是新的，则可能需要向`TAG`表添加行。如果标签存在，则我们只是将帖子与现有标签的 ID 关联。这是`add_post()`方法函数：

```py
    def add_post( self, blog, post ):
        insert_post="""
        INSERT INTO POST(TITLE, DATE, RST_TEXT, BLOG_ID)
            VALUES(:title, :date, :rst_text, :blog_id)
        """
        query_tag="""
        SELECT * FROM TAG WHERE PHRASE=?
        """
        insert_tag= """
        INSERT INTO TAG(PHRASE) VALUES(?)
        """
        insert_association= """
        INSERT INTO ASSOC_POST_TAG(POST_ID, TAG_ID) VALUES(:post_id, :tag_id)
        """
        with self.database:
            self.database.execute( **insert_post**,
                dict(title=post.title, date=post.date,
                    rst_text=post.rst_text, blog_id=blog.id) )
            row = self.database.execute( **get_last_id** ).fetchone()
            post.id= row[0]
            for tag in post.tags:
                tag_row= self.database.execute( **query_tag**, (tag,) ).fetchone()
                if tag_row is not None:
                    tag_id= tag_row['ID']
                else:
                    self.database.execute(**insert_tag**, (tag,))
                    row = self.database.execute( **get_last_id** ).fetchone()
                    tag_id= row[0]
                self.database.execute(**insert_association**,
                    dict(tag_id=tag_id,post_id=post.id))
        return post
```

在数据库中创建完整帖子的过程涉及几个 SQL 步骤。我们使用`insert_post`语句在`POST`表中创建行。我们还将使用通用的`get_last_id`查询返回新`POST`行的分配的主键。

`query_tag`语句用于确定数据库中是否存在标签。如果查询的结果不是`None`，则意味着找到了`TAG`行，我们有该行的 ID。否则，必须使用`insert_tag`语句创建一行；必须使用`get_last_id`查询确定分配的 ID。

每个`POST`都通过向`ASSOC_POST_TAG`表插入行与相关标签相关联。`insert_association`语句创建必要的行。这里有两种迭代器样式查询来定位`Blogs`和`Posts`：

```py
    def blog_iter( self ):
        query= """
        SELECT * FROM BLOG
        """
        results= self.database.execute( query )
        for row in results:
            blog= Blog( id= row['ID'], title= row['TITLE'] )
            yield blog
    def post_iter( self, blog ):
        query= """
        SELECT ID FROM POST WHERE BLOG_ID=?
        """
        results= self.database.execute( query, (blog.id,) )
        for row in results:
            yield self.get_post( row['ID'] )
```

`blog_iter()`方法函数定位所有`BLOG`行并从这些行构建`Blog`实例。

`post_iter()`方法函数定位与`BLOG` ID 相关联的`POST` ID。`POST` ID 与`get_post()`方法一起用于构建`Post`实例。由于`get_post()`将对`POST`表执行另一个查询，因此在这两种方法之间可能存在优化。

### 实现容器关系

我们对`Blog`类的定义包括两个需要访问该博客中包含的所有帖子的特性。`Blog.entries`属性和`Blog.by_tag()`方法函数都假定博客包含`Post`实例的完整集合。

为了使其工作，`Blog`类必须知道`Access`对象，以便它可以使用`Access.post_iter()`方法来实现`Blog.entries`。我们对此有两种整体设计模式：

+   全局`Access`对象简单且工作得很好。我们必须确保全局数据库连接适当打开，这可能是全局`Access`对象的一个挑战。

+   将`Access`对象注入到每个要持久化的`Blog`对象中。这有点复杂，因为我们必须调整与数据库关联的每个对象。

由于每个与数据库相关的对象都应该由`Access`类创建，因此`Access`类将适合**工厂**模式。我们可以对这个工厂进行三种改变。这些将确保博客或帖子知道活动的`Access`对象：

+   每个`return blog`都需要扩展为`blog._access= self; return blog`。这发生在`get_blog()`、`add_blog()`和`blog_iter()`中。

+   每个`return post`都需要扩展为`post._access= self; return post`。这发生在`get_post()`、`add_post()`和`post_iter()`中。

+   修改`add_blog()`方法以接受构建`Blog`对象的参数，而不是接受在`Access`工厂之外构建的`Blog`或`Post`对象。定义看起来会像下面这样：`def add_blog( self, title ):`

+   修改`add_post()`方法以接受一个博客和构建`Post`对象的参数。定义看起来会像这样：`def add_post( self, blog, title, date, rst_text, tags ):`

一旦我们将`_access`属性注入到每个`Blog`实例中，我们就可以这样做：

```py
@property
def entries( self ):return self._access.post_iter( self )
```

这将返回属于博客对象的一系列帖子对象。这使我们能够定义类定义中的方法，这些方法将处理子对象或父对象，就好像它们包含在对象中一样。

## 通过索引提高性能

改善 SQLite 等关系数据库性能的一种方法是加快连接操作。这样做的理想方式是包含足够的索引信息，以便不需要进行缓慢的搜索操作来查找匹配的行。没有索引，必须读取整个表才能找到引用的行。有了索引，只需读取相关的行子集。

当我们定义一个可能在查询中使用的列时，我们应该考虑为该列构建一个索引。这意味着在我们的表定义中添加更多的 SQL DDL 语句。

索引是一个单独的存储，但与特定的表和列相关联。SQL 看起来像以下代码：

```py
CREATE INDEX IX_BLOG_TITLE ON BLOG( TITLE );
```

这将在`Blog`表的`title`列上创建一个索引。不需要做其他任何事情。SQL 数据库在执行基于索引列的查询时将使用该索引。当数据被创建、更新或删除时，索引将自动调整。

索引涉及存储和计算开销。很少使用的索引可能会因为创建和维护成本而成为性能障碍，而不是帮助。另一方面，一些索引非常重要，可以带来显著的性能改进。在所有情况下，我们无法直接控制正在使用的数据库算法；我们所能做的就是创建索引并测量性能的影响。

在某些情况下，将列定义为键可能会自动包括添加索引。这方面的规则通常在数据库的 DDL 部分中清楚地说明。例如，SQLite 表示：

> 在大多数情况下，唯一和主键约束是通过在数据库中创建唯一索引来实现的。

它接着列出了两个例外。其中一个是整数主键例外，这是我们一直在使用的设计模式，用于强制数据库为我们创建代理键。因此，我们的整数主键设计不会创建任何额外的索引。

## 添加 ORM 层

有相当多的 Python ORM 项目。这些项目的列表可以在这里找到：[`wiki.python.org/moin/HigherLevelDatabaseProgramming`](https://wiki.python.org/moin/HigherLevelDatabaseProgramming)。

我们将选择其中一个作为示例。我们将使用 SQLAlchemy，因为它为我们提供了许多功能，并且相当受欢迎。与许多事物一样，并没有*最佳*；其他 ORM 层具有不同的优势和劣势。

由于使用关系数据库支持 Web 开发的流行，Web 框架通常包括 ORM 层。Django 有自己的 ORM 层，web.py 也有。在某些情况下，我们可以从更大的框架中分离出 ORM。但是，与独立的 ORM 一起工作似乎更简单。

SQLAlchemy 的文档、安装指南和代码可在[`www.sqlalchemy.org`](http://www.sqlalchemy.org)找到。在安装时，如果不需要高性能优化，使用`--without-cextensions`可以简化流程。

重要的是要注意，SQLAlchemy 可以完全用一流的 Python 构造替换应用程序的所有 SQL 语句。这具有深远的优势，可以让我们使用单一语言 Python 编写应用程序，即使在数据访问层中使用了第二种语言 SQL。这可以节省一些开发和调试的复杂性。

然而，这并不消除理解底层 SQL 数据库约束以及我们的设计如何适应这些约束的义务。ORM 层并不能神奇地消除设计考虑。它只是将实现语言从 SQL 更改为 Python。

### 设计 ORM 友好的类

使用 ORM 时，我们将根本改变设计和实现持久类的方式。我们将扩展类定义的语义，具有三个不同的层次含义：

+   该类将是一个 Python 类，可以用来创建 Python 对象。方法函数被这些对象使用。

+   该类还将描述一个 SQL 表，并可以被 ORM 用来创建构建和维护数据库结构的 SQL DDL。

+   该类还将定义 SQL 表和 Python 类之间的映射。它将成为将 Python 操作转换为 SQL DML 并从 SQL 查询构建 Python 对象的工具。

大多数 ORM 都是设计成我们将使用描述符来正式定义类的属性。我们不只是在`__init__()`方法中定义属性。有关描述符的更多信息，请参见第三章，*属性访问、属性和描述符*。

SQLAlchemy 要求我们构建一个**声明基类**。这个基类为我们应用程序的类定义提供了一个元类。它还作为我们为数据库定义的元数据的存储库。如果我们遵循默认设置，很容易将这个类称为`Base`。

以下是可能有用的导入列表：

```py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Table
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Enum, \
    Float, Integer, Interval, LargeBinary, Numeric, PickleType, \
    SmallInteger, String, Text, Time, Unicode, UnicodeText ForeignKey
from sqlalchemy.orm import relationship, backref
```

我们导入了一些必要的定义来创建表的列，列和创建不特定地映射到 Python 类的稀有表，`Table`。我们导入了所有通用列类型定义。我们只会使用其中的一些列类型。SQLAlchemy 不仅定义了这些通用类型，还定义了 SQL 标准类型，还为各种支持的 SQL 方言定义了特定于供应商的类型。似乎很容易坚持使用通用类型，并允许 SQLAlchemy 在通用、标准和供应商类型之间进行映射。

我们还导入了两个助手来定义表之间的关系，`relationship`和`backref`。SQLAlchemy 的元类是由`declarative_base()`函数构建的：

```py
Base = declarative_base()
```

创建的`Base`对象必须是我们要定义的任何持久类的元类。我们将定义三个映射到 Python 类的表。我们还将定义第四个表，这个表仅仅是 SQL 实现多对多关系所需的。

这是`Blog`类：

```py
class Blog(Base):
    __tablename__ = "BLOG"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= '='*len(self.title),
            entries= [ e.as_dict() for e in self.entries ]
        )
```

我们的`Blog`类映射到一个名为`"BLOG"`的表。我们在这个表中包含了两个列的描述符。`id`列被定义为`Integer`主键。隐式地，这将是一个自动增量字段，以便为我们生成代理键。

标题列被定义为通用字符串。我们可以使用`Text`、`Unicode`甚至`UnicodeText`。底层引擎可能对这些不同类型有不同的实现。在我们的情况下，SQLite 将几乎相同地处理所有这些。还要注意，SQLite 不需要对列的长度设置上限；其他数据库引擎可能需要对`String`的大小设置上限。

`as_dict()`方法函数指的是一个`entries`集合，在这个类中显然没有定义。当我们查看`Post`类的定义时，我们将看到`entries`属性是如何构建的。这是`Post`类的定义：

```py
class Post(Base):
    __tablename__ = "POST"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    date = Column(DateTime)
    rst_text = Column(UnicodeText)
    blog_id = Column(Integer, ForeignKey('BLOG.id'))
    blog = relationship( 'Blog', backref='entries' )
    tags = relationship('Tag', secondary=assoc_post_tag, backref='posts')
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= '-'*len(self.title),
            date= self.date,
            rst_text= self.rst_text,
            tags= [ t.phrase for t in self.tags],
        )
```

这个类有五个属性，两个关系和一个方法函数。`id`属性是一个整数主键；这将是一个默认的自动增量值。`title`属性是一个简单的字符串。`date`属性将是一个`DateTime`列；`rst_text`被定义为`UnicodeText`，以强调我们对该字段中任何 Unicode 字符的期望。

`blog_id`是一个外键引用，指向包含此帖子的父博客。除了外键列的定义，我们还包括了帖子和父博客之间的显式`relationship`定义。这个`relationship`定义成为我们可以用于从帖子导航到父博客的属性。

`backref`选项包括一个将被添加到`Blog`类中的反向引用。`Blog`类中的这个引用将是包含在`Blog`中的`Posts`的集合。`backref`选项将在`Blog`类中命名新属性，以引用子`Posts`。

`tags`属性使用`relationship`定义；这个属性将通过一个关联表导航，以定位与帖子相关联的所有`Tag`实例。我们将看看下面的关联表。这也使用`backref`来在`Tag`类中包含一个属性，引用`Post`实例的相关集合。

`as_dict()`方法利用`tags`属性来定位与此`Post`相关联的所有`Tags`。以下是`Tag`类的定义：

```py
class Tag(Base):
    __tablename__ = "TAG"
    id = Column(Integer, primary_key=True)
    phrase = Column(String, unique=True)
```

我们定义了一个主键和一个`String`属性。我们包括了一个约束，以确保每个标签都是明确唯一的。尝试插入重复的标签将导致数据库异常。`Post`类定义中的关系意味着将在这个类中创建额外的属性。

根据 SQL 的要求，我们需要一个关联表来处理标签和帖子之间的多对多关系。这个表纯粹是 SQL 中的技术要求，不需要映射到 Python 类：

```py
assoc_post_tag = Table('ASSOC_POST_TAG', Base.metadata,
    Column('POST_ID', Integer, ForeignKey('POST.id') ),
    Column('TAG_ID', Integer, ForeignKey('TAG.id') )
)
```

我们必须显式地将其绑定到`Base.metadata`集合。这种绑定自动成为使用`Base`作为元类的类的一部分。我们定义了一个包含两个`Column`实例的表。每个列都是我们模型中另一个表的外键。

### 使用 ORM 层构建模式

为了连接到数据库，我们需要创建一个引擎。引擎的一个用途是使用我们的表声明构建数据库实例。引擎的另一个用途是管理会话中的数据，这是我们稍后会看到的。以下是一个我们可以用来构建数据库的脚本：

```py
from sqlalchemy import create_engine
engine = create_engine('sqlite:///./p2_c11_blog2.db', echo=True)
Base.metadata.create_all(engine)
```

当我们创建一个`Engine`实例时，我们使用类似 URL 的字符串，其中包含了命名供应商产品的名称以及创建与该数据库的连接所需的所有附加参数。在 SQLite 的情况下，连接是一个文件名。在其他数据库产品的情况下，可能会有服务器主机名和身份验证凭据。

一旦我们有了引擎，我们就完成了一些基本的元数据操作。我们已经执行了`create_all()`，它会构建所有的表。我们也可以执行`drop_all()`，它会删除所有的表，丢失所有的数据。当然，我们也可以创建或删除单个模式项。

如果我们在软件开发过程中更改表定义，它不会自动改变 SQL 表定义。我们需要显式地删除并重建表。在某些情况下，我们可能希望保留一些操作数据，从旧表中创建和填充新表可能会导致潜在的复杂手术。

`echo=True`选项会写入生成的 SQL 语句的日志条目。这有助于确定声明是否完整并创建了预期的数据库设计。以下是生成的输出片段：

```py
CREATE TABLE "BLOG" (
  id INTEGER NOT NULL,
  title VARCHAR,
  PRIMARY KEY (id)
)
CREATE TABLE "TAG" (
  id INTEGER NOT NULL,
  phrase VARCHAR,
  PRIMARY KEY (id),
  UNIQUE (phrase)
)

CREATE TABLE "POST" (
  id INTEGER NOT NULL,
  title VARCHAR,
  date DATETIME,
  rst_text TEXT,
  blog_id INTEGER,
  PRIMARY KEY (id),
  FOREIGN KEY(blog_id) REFERENCES "BLOG" (id)
)

CREATE TABLE "ASSOC_POST_TAG" (
  "POST_ID" INTEGER,
  "TAG_ID" INTEGER,
  FOREIGN KEY("POST_ID") REFERENCES "POST" (id),
  FOREIGN KEY("TAG_ID") REFERENCES "TAG" (id)
)
```

这显示了基于我们的类定义创建的`CREATE TABLE`语句。

数据库建立后，我们可以创建、检索、更新和删除对象。为了处理数据库对象，我们需要创建一个作为 ORM 管理对象缓存的会话。

### 使用 ORM 层操作对象

为了使用对象，我们需要一个会话缓存。这与一个引擎绑定在一起。我们将新对象添加到会话缓存中。我们还将使用会话缓存来查询数据库中的对象。这确保了所有需要持久存在的对象都在缓存中。以下是创建一个工作会话的方法：

```py
from sqlalchemy.orm import sessionmaker
Session= sessionmaker(bind=engine)
session= Session()
```

我们使用 SQLAlchemy 的`sessionmaker()`函数来创建一个`Session`类。这个类绑定到我们之前创建的数据库引擎。然后我们使用`Session`类来构建一个`session`对象，我们可以用它来执行数据操作。通常需要一个会话来处理对象。

通常，我们会创建一个`sessionmaker`类以及引擎。然后我们可以使用那个`sessionmaker`类来为我们的应用程序处理构建多个会话。

对于简单的对象，我们创建它们并将它们加载到会话中，如下所示的代码：

```py
blog= Blog( title="Travel 2013" )
session.add( blog )
```

这将一个新的`Blog`对象放入名为`session`的会话中。`Blog`对象不一定会被写入数据库。在执行数据库写入之前，我们需要提交会话。为了满足原子性要求，我们将在提交会话之前完成构建一个帖子。

首先，我们将在数据库中查找`Tag`实例。如果它们不存在，我们将创建它们。如果它们存在，我们将使用在数据库中找到的标签：

```py
tags = [ ]
for phrase in "#RedRanger", "#Whitby42", "#ICW":
    try:
        tag= session.query(Tag).filter(Tag.phrase == phrase).one()
    except sqlalchemy.orm.exc.NoResultFound:
        tag= Tag(phrase=phrase)
        session.add(tag)
    tags.append(tag)
```

我们使用`session.query()`函数来检查给定类的实例。每个`filter()`函数都会向查询中添加一个条件。`one()`函数确保我们找到了一行。如果引发异常，那么意味着`Tag`不存在。我们需要构建一个新的`Tag`并将其添加到会话中。

一旦我们找到或创建了`Tag`实例，我们可以将其附加到一个名为`tags`的本地列表中；我们将使用这个`Tag`实例列表来创建`Post`对象。以下是我们如何构建一个`Post`：

```py
p2= Post( date=datetime.datetime(2013,11,14,17,25),
    title="Hard Aground",
    rst_text="""Some embarrassing revelation. Including ☹ and ⎕""",
    blog=blog,
    tags=tags
    )
session.add(p2)
blog.posts= [ p2 ]
```

这包括对父博客的引用。它还包括我们构建的（或在数据库中找到的）`Tag`实例的列表。

`Post.blog`属性在类定义中被定义为一个关系。当我们分配一个对象时，SQLAlchemy 会提取出正确的 ID 值，以创建 SQL 数据库用来实现关系的外键引用。

`Post.tags`属性也被定义为一个关系。`Tag`对象通过关联表引用。SQLAlchemy 正确跟踪 ID 值，以为我们构建 SQL 关联表中必要的行。

为了将`Post`与`Blog`关联起来，我们将利用`Blog.posts`属性。这也被定义为一个关系。当我们将`Post`对象列表分配给这个关系属性时，ORM 将在每个`Post`对象中构建适当的外键引用。这是因为我们在定义关系时提供了`backref`属性。最后，我们提交会话：

```py
session.commit()
```

数据库插入都是在自动生成的 SQL 中处理的。对象仍然保留在会话中的缓存中。如果我们的应用程序继续使用这个会话实例，那么对象池将保持可用，而不一定执行任何针对数据库的实际查询。

另一方面，如果我们希望确保其他并发进程写入的任何更新都包含在查询中，我们可以为该查询创建一个新的空会话。当我们丢弃一个会话并使用一个空会话时，对象必须从数据库中获取以刷新会话。

我们可以编写一个简单的查询来检查并打印所有的`Blog`对象：

```py
session= Session()
for blog in session.query(Blog):
    print( "{title}\n{underline}\n".format(**blog.as_dict()) )
    for p in blog.entries:
        print( p.as_dict() )
```

这将检索所有的`Blog`实例。`Blog.as_dict()`方法将检索博客中的所有帖子。`Post.as_dict()`方法将检索所有标签。SQL 查询将由 SQLAlchemy 自动生成并自动执行。

我们没有包括来自第九章的基于模板的格式的其余部分。它没有改变。我们能够从`Blog`对象通过`entries`列表导航到`Post`对象，而不需要编写复杂的 SQL 查询。将导航转换为查询是 SQLAlchemy 的工作。对于 SQLAlchemy 来说，使用 Python 迭代器就足以生成正确的查询来刷新缓存并返回预期的对象。

如果我们为`Engine`实例定义了`echo=True`，那么我们将能够看到执行检索`Blog`、`Post`和`Tag`实例的 SQL 查询序列。这些信息可以帮助我们了解应用程序对数据库服务器进程的工作负载。

## 给定一个标签字符串查询帖子对象

关系数据库的一个重要好处是我们能够遵循对象之间的关系。使用 SQLAlchemy 的查询功能，我们可以从`Tag`到`Post`的关系，并找到所有共享给定`Tag`字符串的`Posts`。

查询是会话的一个特性。这意味着已经在会话中的对象不需要从数据库中获取，这可能节省时间。不在会话中的对象被缓存在会话中，以便在提交时处理更新或删除。

为了收集所有具有特定标签的帖子，我们需要使用中间关联表以及`Post`和`Tag`表。我们将使用会话的查询方法来指定我们希望得到的对象类型。我们将使用流畅接口来加入各种中间表和我们希望的最终表以及选择条件。看起来是这样的：

```py
for post in session.query(Post).join(assoc_post_tag).join(Tag).filter(
    Tag.phrase == "#Whitby42" ):
    print( post.blog.title, post.date, post.title, [t.phrase for t in post.tags] )
```

`session.query()`方法指定了我们想要查看的表。如果我们只是这样做，我们会看到每一行。`join()`方法标识必须匹配的附加表。因为我们在类定义中提供了关系信息，SQLAlchemy 可以计算出使用主键和外键匹配行所需的 SQL 细节。最终的`filter()`方法为所需子集的行提供了选择条件。这是生成的 SQL：

```py
SELECT "POST".id AS "POST_id", "POST".title AS "POST_title", "POST".date AS "POST_date", "POST".rst_text AS "POST_rst_text", "POST".blog_id AS "POST_blog_id"
FROM "POST" JOIN "ASSOC_POST_TAG" ON "POST".id = "ASSOC_POST_TAG"."POST_ID"
JOIN "TAG" ON "TAG".id = "ASSOC_POST_TAG"."TAG_ID"
WHERE "TAG".phrase = ?
```

Python 版本稍微更容易理解，因为关键匹配的细节可以被省略。`print()`函数使用`post.blog.title`从`Post`实例导航到相关的博客并显示`title`属性。如果博客在会话缓存中，这种导航会很快完成。如果博客不在会话缓存中，它将从数据库中获取。

这种导航行为也适用于`[t.phrase for t in post.tags]`。如果对象在会话缓存中，它就会被简单地使用。在这种情况下，与帖子相关的`Tag`对象的集合可能会导致复杂的 SQL 查询：

```py
SELECT "TAG".id AS "TAG_id", "TAG".phrase AS "TAG_phrase"
FROM "TAG", "ASSOC_POST_TAG"
WHERE ? = "ASSOC_POST_TAG"."POST_ID"
AND "TAG".id = "ASSOC_POST_TAG"."TAG_ID"
```

在 Python 中，我们只需通过`post.tags`进行导航。SQLAlchemy 为我们生成并执行了 SQL。

## 通过索引提高性能

改善关系数据库（如 SQLite）性能的一种方法是加快连接操作。我们不希望 SQLite 读取整个表来查找匹配的行。通过在特定列上建立索引，SQLite 可以检查索引并仅从表中读取相关行。

当我们定义可能在查询中使用的列时，我们应该考虑为该列建立索引。这是一个简单的过程，使用 SQLAlchemy。我们只需用`index=True`注释类的属性。

我们可以对我们的`Post`表进行相当小的更改，例如添加索引：

```py
class Post(Base):
    __tablename__ = "POST"
    id = Column(Integer, primary_key=True)
    title = Column(String, index=True)
    date = Column(DateTime, index=True)
    blog_id = Column(Integer, ForeignKey('BLOG.id'), index=True)
```

为标题和日期添加两个索引通常会加快按标题或日期查询帖子的速度。并不一定保证性能会有所改善。关系数据库的性能涉及许多因素。重要的是要在有索引和没有索引的情况下测量现实工作负载的性能。

通过`blog_id`添加索引，同样，可能会加快在`Blog`和`Post`表中行之间的连接操作。数据库引擎也可能使用一种不受此索引影响的算法。

索引涉及存储和计算开销。很少使用的索引可能创建和维护的成本如此之高，以至于它成为一个问题，而不是解决方案。另一方面，一些索引非常重要，可以带来显著的性能改进。在所有情况下，我们无法直接控制正在使用的数据库算法；我们能做的就是创建索引并测量性能影响。

### 模式演变

在处理 SQL 数据库时，我们必须解决模式演变的问题。我们的对象具有动态状态和静态类定义。我们可以轻松地持久化动态状态。我们的类定义是持久数据的模式的一部分；我们还有对正式 SQL 模式的映射。无论是类还是 SQL 模式都不是*绝对*静态的。

如果我们更改了类定义，我们如何从数据库中获取对象？如果数据库必须更改，我们如何升级 Python 映射并仍然映射数据？一个好的设计通常涉及几种技术的组合。

Python 类的方法函数和属性的更改不会改变与 SQL 行的映射。这些可以称为次要更改，因为数据库中的表仍与更改后的类定义兼容。新软件发布可以有一个新的次要版本号。

Python 类属性的更改不一定会改变持久化对象的状态。在将数据类型从数据库转换为 Python 对象时，SQL 可能会有些灵活。ORM 层可以增加灵活性。在某些情况下，我们可以进行一些类或数据库更改，并称其为次要版本更新，因为现有的 SQL 模式仍将与新的类定义一起工作。例如，我们可以将 SQL 表从整数更改为字符串，而不会因为 SQL 和 ORM 转换而出现重大破坏。

对 SQL 表定义的更改将明显修改持久化对象。当现有数据库行不再与新类定义兼容时，这些可以称为重大更改。这些类型的更改不应该通过*修改*Python 类定义来进行。这些类型的更改应该通过定义一个新的子类，并提供一个更新的工厂函数来创建旧类或新类的实例。

在处理持久的 SQL 数据时，可以通过以下两种方式之一进行模式更改：

+   使用 SQL 的`ALTER`语句对现有模式进行更改。某些类型的更改可以逐步对 SQL 模式进行。对所允许的更改有许多约束和限制。这并不具有很好的泛化性；应该将其视为一种可能适用于较小更改的特殊情况。

+   创建新表和删除旧表。一般来说，SQL 模式更改将足够重要，以至于我们需要从旧表创建新版本的表，对数据结构进行深刻的更改。

SQL 数据库模式更改通常涉及运行一次性转换脚本。此脚本将使用旧模式查询现有数据，将其转换为新数据，并使用新模式将新数据插入数据库。当然，这必须在用户首选的实时操作数据库之前在备份数据库上进行测试。一旦完成模式更改，就可以安全地忽略旧模式，并稍后删除以释放存储空间。

这种转换可以在单个数据库中使用不同的表名或不同的模式名（对于支持命名模式的数据库）。如果我们将旧数据和新数据并排放置，我们就可以从旧应用程序灵活地升级到新应用程序。这对于试图提供全天候可用性的网站尤为重要。

在某些情况下，有必要向模式添加表，其中仅包含纯粹的管理细节，例如模式版本的标识。应用程序可以在建立数据库连接后首先查询此表，并在模式版本错误时快速失败。

## 总结

我们以三种方式查看了使用 SQLite 的基础知识：直接使用、通过访问层、以及通过 SQLAlchemy ORM。我们必须创建 SQL DDL 语句；我们可以直接在我们的应用程序中或在访问层中进行此操作。我们还可以通过 SQLAlchemy 类定义来构建 DDL。为了操作数据，我们将使用 SQL DML 语句；我们可以以过程化风格直接进行此操作，或者我们可以使用我们自己的访问层或 SQLAlchemy 来创建 SQL。

### 设计考虑和权衡

`sqlite3`模块的一个优点是它允许我们持久化不同的项目。由于我们使用支持并发写入的数据库，我们可以有多个进程更新数据，依靠 SQLite 通过其内部锁定处理并发。

使用关系数据库会施加许多限制。我们必须考虑如何将我们的对象映射到数据库表的行：

+   我们可以直接使用 SQL，仅使用支持的 SQL 列类型，并在很大程度上避免面向对象的类

+   我们可以使用手动映射来扩展 SQLite 以处理我们的对象作为 SQLite BLOB 列

+   我们可以编写自己的访问层来适应和转换我们的对象和 SQL 行

+   我们可以使用 ORM 层来实现行到对象的映射。

### 映射替代方案

混合 Python 和 SQL 的问题在于可能会产生一种我们可以称之为“全能 SQL”解决方案的冲动。这里的想法是关系数据库在某种程度上是理想的平台，而 Python 通过注入不必要的面向对象特性来破坏这一点。

有时，全 SQL、无对象的设计策略被证明更适合某些类型的问题。具体来说，支持者会指出使用 SQL 的`GROUP BY`子句对大量数据进行汇总是 SQL 的理想用途。

这是由 Python 的`defaultdict`和`Counter`非常有效地实现的。Python 版本通常如此有效，以至于一个小型的 Python 程序查询大量行并使用`defaultdict`累积摘要可能比使用`GROUP BY`执行 SQL 的数据库服务器更快。

如果有疑问，就进行测量。SQL 数据库支持者会说一些无稽之谈。当面对 SQL 应该神奇地比 Python 更快的说法时，收集证据。这种数据收集不仅限于一次性的初始技术尖峰情况。随着使用量的增长和变化，SQL 数据库与 Python 的相对优点也会发生变化。

自制的访问层往往会对问题域高度特定。这可能具有高性能和相对透明的从行到对象的映射的优势。但每当类发生变化或数据库实现发生变化时，维护可能会很烦人。

一个成熟的 ORM 项目可能需要一些初始努力来学习 ORM 的特性，但长期的简化是重要的好处。学习 ORM 层的特性可能既涉及初始工作，也涉及重新工作，因为经验教训。首次尝试设计具有良好对象特性并仍适合 SQL 框架的设计将不得不重新进行，因为应用程序的权衡和考虑变得更加清晰。

### 键和关键设计

因为 SQL 依赖于键，我们必须小心设计和管理各种对象的键。我们必须设计从对象到将用于标识该对象的键的映射。一种选择是找到适当的主键属性（或属性组合），并且不能更改。另一种选择是生成不能更改的代理键；这允许所有其他属性被更改。

大多数关系数据库可以为我们生成代理键。这通常是最好的方法。对于其他唯一属性或候选键属性，我们可以定义 SQL 索引以提高处理性能。

我们还必须考虑对象之间的外键关系。有几种常见的设计模式：一对多，多对一，多对多和可选的一对一。我们需要知道 SQL 如何使用键来实现这些关系，以及 SQL 查询将用于填充 Python 集合。

### 应用软件层

由于使用`sqlite3`时相对复杂，我们的应用软件必须更加合理地分层。通常，我们将查看具有类似以下层的软件架构：

+   表示层：这是顶层用户界面，可以是 Web 演示或桌面 GUI。

+   应用层：这是使应用程序工作的内部服务或控制器。这可以称为处理模型，与逻辑数据模型不同。

+   业务层或问题域模型层：这些是定义业务领域或问题空间的对象。有时被称为逻辑数据模型。我们看了如何使用微博博客和帖子示例来对这些对象进行建模。

+   基础设施：这通常包括几个层，以及其他横切关注点，如日志记录、安全性和网络访问：

+   数据访问层：这些是访问数据对象的协议或方法。通常是 ORM 层。我们已经看过 SQLAlchemy。还有许多其他选择。

+   持久性层：这是在文件存储中看到的物理数据模型。`sqlite3`模块实现了持久性。当使用诸如 SQLAlchemy 之类的 ORM 层时，我们只在创建引擎时引用 SQLite。

在本章中查看`sqlite3`和第十章中的`shelve`，*通过 Shelve 存储和检索对象*，清楚地表明掌握面向对象编程涉及一些更高级别的设计模式。我们不能简单地孤立设计类，而是需要考虑如何将类组织成更大的结构。

### 展望未来

在下一章中，我们将研究如何使用 REST 传输和共享对象。这种设计模式向我们展示了如何管理状态的表示以及如何将对象状态从一个进程传输到另一个进程。我们将利用许多持久性模块来表示正在传输的对象的状态。

在第十三章中，*配置文件和持久性*，我们将研究配置文件。我们将研究利用持久性数据的几种方法，以控制应用程序。

# 第十二章：传输和共享对象

我们将扩展我们在第九章中展示的对象表示的序列化技术，*序列化和保存 - JSON、YAML、Pickle、CSV 和 XML*。当我们需要传输一个对象时，我们执行某种**表述性状态转移**（**REST**）。当我们序列化一个对象时，我们正在创建对象状态的表示。这种表示可以传输到另一个进程（通常在另一台主机上）；然后，另一个进程可以根据状态的表示和本地类的定义构建原始对象的版本。

我们可以以多种方式执行 REST 处理。其中之一是我们可以使用的状态表示。另一个方面是控制传输的协议。我们不会涵盖所有这些方面的组合。相反，我们将专注于两种组合。

对于互联网传输，我们将利用 HTTP 协议来实现**创建-检索-更新-删除**（**CRUD**）处理操作。这通常被称为 REST Web 服务器。我们还将研究提供 RESTful Web 服务。这将基于 Python 的**Web 服务网关接口**（**WSGI**）参考实现，即`wsgiref`包。

对于在同一主机上的进程之间的本地传输，我们将研究`multiprocessing`模块提供的本地消息队列。有许多复杂的队列管理产品。我们将专注于标准库提供的内容。

这种处理建立在使用 JSON 或 XML 来表示对象的基础上。对于 WSGI，我们将添加 HTTP 协议和一组设计模式来定义 Web 服务器中的事务。对于多处理，我们将添加一个处理池。

在处理 REST 传输时，还有一个额外的考虑因素：源或数据可能不可信。我们必须实施一些安全措施。在使用常用表示形式 JSON 和 XML 时，几乎没有安全考虑。YAML 引入了一个安全问题，并支持安全加载操作；有关更多信息，请参见第九章中的内容。由于安全问题，`pickle`模块还提供了一个受限制的反序列化器，可以信任不导入异常模块并执行有害代码。

## 类、状态和表示

在某些情况下，我们可能正在创建一个将向远程客户端提供数据的服务器。在其他情况下，我们可能希望从远程计算机消耗数据。我们可能有一个混合情况，即我们的应用既是远程计算机的客户端，又是移动应用程序的服务器。有许多情况下，我们的应用程序与远程持久化的对象一起工作。

我们需要一种方法来从一个进程传输对象到另一个进程。我们可以将更大的问题分解为两个较小的问题。互联网协议可以帮助我们将字节从一个主机上的一个进程传输到另一个主机上的一个进程。序列化可以将我们的对象转换为字节。

与对象状态不同，我们通过一个完全独立且非常简单的方法传输类定义。我们通过源代码交换类定义。如果我们需要向远程主机提供类定义，我们将向该主机发送 Python 源代码。代码必须被正确安装才能有用；这通常是由管理员手动执行的操作。

我们的网络传输字节。因此，我们需要将对象实例变量的值表示为字节流。通常，我们将使用两步转换为字节；我们将对象的状态表示为字符串，并依赖于字符串以标准编码之一提供字节。

## 使用 HTTP 和 REST 传输对象

**超文本传输协议**（**HTTP**）是通过一系列**请求评论**（**RFC**）文档定义的。我们不会审查所有细节，但我们将触及三个重点。

HTTP 协议包括请求和响应。请求包括方法、**统一资源标识符**（**URI**）、一些标头和可选附件。标准中定义了许多可用的方法。大多数浏览器专注于进行`GET`和`POST`请求。标准浏览器包括`GET`、`POST`、`PUT`和`DELETE`请求，这些是我们将利用的，因为它们对应于 CRUD 操作。我们将忽略大部分标头，关注 URI 的路径部分。

响应包括状态码数字和原因、标头和一些数据。有各种各样的状态码数字。其中，我们只对其中的一些感兴趣。`200`状态码是服务器的通用`OK`响应。`201`状态码是`已创建`响应，可能适合显示我们的帖子已经成功并且数据已经发布。`204`状态码是`无内容`响应，可能适合`DELETE`。`400`状态码是`错误请求`，`401`状态码是`未经授权`，`404`状态码是`未找到`。这些状态码通常用于反映无法执行或无效的操作。

大多数`2xx`成功的响应将包括一个编码的对象或对象序列。`4xx`错误响应可能包括更详细的错误消息。

HTTP 被定义为无状态的。服务器不应该记得先前与客户端的交互。我们有许多候选的解决方法来解决这个限制。对于交互式网站，使用 cookie 来跟踪事务状态并改善应用程序行为。然而，对于 Web 服务，客户端不会是一个人；每个请求都可以包括认证凭据。这进一步要求保护连接。对于我们的目的，我们将假设服务器将使用**安全套接字层**（**SSL**）并在端口 443 上使用 HTTPS 连接，而不是在端口 80 上使用 HTTP。

### 通过 REST 实现 CRUD 操作

我们将讨论 REST 协议背后的三个基本理念。第一个理念是使用任何方便的文本序列化对象状态。其次，我们可以使用 HTTP 请求 URI 来命名一个对象；URI 可以包括任何级别的细节，包括模式、模块、类和统一格式的对象标识。最后，我们可以使用 HTTP 方法来映射到 CRUD 规则，以定义对命名对象执行的操作。

将 HTTP 用于 RESTful 服务推动了 HTTP 请求和响应的原始定义的边界。这意味着一些请求和响应语义是开放的，正在进行讨论。我们不会呈现所有的替代方案，每个替代方案都有独特的优点，我们将建议一个单一的方法。我们的重点是 Python 语言，而不是设计 RESTful Web 服务的更一般的问题。REST 服务器通常通过以下五个基本用例支持 CRUD 操作：

+   **创建**：我们将使用`HTTP POST`请求来创建一个新对象，并提供仅提供类信息的 URI。例如`//host/app/blog/`这样的路径可能命名类。响应可能是一个包含对象副本的 201 消息，该对象最终被保存。返回的对象信息可能包括 RESTful 服务器为新创建的对象分配的 URI，或者构建 URI 的相关键。`POST`请求预期通过创建新的东西来改变 RESTful 资源。

+   检索-搜索：这是一个可以检索多个对象的请求。我们将使用`HTTP GET`请求和提供搜索条件的 URI，通常是在`?`字符之后的查询字符串的形式。URI 可能是`//host/app/blog/?title="Travel 2012-2013"`。请注意，`GET`永远不会改变任何 RESTful 资源的状态。

+   检索-实例：这是一个请求单个对象的请求。我们将使用`HTTP GET`请求和在 URI 路径中命名特定对象的 URI。URI 可能是`//host/app/blog/id/`。虽然预期的响应是一个单一对象，但它可能仍然被包装在列表中，以使其与搜索响应兼容。由于此响应是`GET`，因此状态没有变化。

+   更新：我们将使用`HTTP PUT`请求和标识要替换的对象的 URI。URI 可能是`//host/app/blog/id/`。响应可能是一个包含修订对象副本的 200 消息。显然，这预计会对 RESTful 资源进行更改。使用 200 以外的其他状态响应是有充分理由的。我们将在这里的示例中坚持使用 200。

+   删除：我们将使用`HTTP DELETE`请求和类似`//host/app/blog/id/`的 URI。响应可能是一个简单的`204 NO CONTENT`，在响应中不提供任何对象细节。

由于 HTTP 协议是无状态的，没有提供登录和注销的功能。每个请求必须单独进行身份验证。我们经常使用 HTTP `Authorization`头来提供用户名和密码凭据。在这样做时，我们绝对必须使用 SSL 来保护`Authorization`头的内容。还有更复杂的替代方案，利用单独的身份管理服务器提供身份验证令牌而不是凭据。

### 实施非 CRUD 操作

一些应用程序将具有无法轻松归类为 CRUD 的操作。例如，我们可能有一个**远程过程调用**（**RPC**）风格的应用程序，执行复杂的计算。计算的参数通过 URI 提供，因此在服务器状态中没有 RESTful 的变化。

大多数情况下，这些以计算为重点的操作可以实现为`GET`请求，因为状态没有变化。然而，如果我们要保留请求和回复的日志作为不可否认方案的一部分，我们可能会考虑将它们作为`POST`请求。这在收费网站中尤为重要。

### REST 协议和 ACID

ACID 属性在第十章中定义，*通过 Shelve 存储和检索对象*。这些属性是原子性、一致性、隔离性和持久性。这些是由多个数据库操作组成的事务的基本特征。这些属性不会自动成为 REST 协议的一部分。我们必须考虑当我们确保满足 ACID 属性时 HTTP 是如何工作的。

每个 HTTP 请求都是原子的；因此，我们应该避免设计一个应用程序，该应用程序进行一系列相关的`POST`请求，希望这些请求变得原子。相反，我们应该寻找一种将所有信息捆绑成一个单一请求的方法。此外，我们必须意识到请求通常会从各种客户端交错进行；因此，我们没有一种干净的方法来处理交错请求序列之间的隔离。如果我们有一个适当的多层设计，我们应该将持久性委托给一个单独的持久性模块。

为了实现 ACID 属性，一个常见的技术是定义包含*所有*相关信息的`POST`、`PUT`或`DELETE`请求。通过提供单个复合对象，应用程序可以在单个 REST 请求中执行所有操作。这些更大的对象成为*文档*，可能包含更复杂交易的几个部分。

当查看我们的博客和帖子关系时，我们发现我们可能希望处理两种`HTTP POST`请求来创建一个新的`Blog`实例。这两个请求如下：

+   **只有标题没有额外帖子条目的博客**：对于这个，我们可以很容易地实现 ACID 属性，因为它只是一个单一的对象。

+   **一个复合对象，即博客加上一系列帖子条目**：我们需要序列化博客和所有相关的`Post`实例。这需要作为一个单独的`POST`请求发送。然后，我们可以通过创建博客、相关帖子，并在整个对象集合变得持久时返回单个`201 Created`状态来实现 ACID 属性。这可能涉及支持 RESTful web 服务器的数据库中的复杂多语句事务。

### 选择一种表示形式 - JSON、XML 或 YAML

没有一个很好的理由来选择单一的表示；支持多种表示相对容易。客户端应该被允许要求一种表示。客户端可以在几个地方指定表示：

+   我们可以使用查询字符串的一部分，`https://host/app/class/id/?form=XML`。

+   我们可以使用 URI 的一部分：`https://host/app;XML/class/id/`。在这个例子中，我们使用了一个子分隔符来标识所需的表示。`app;XML`语法命名了应用程序`app`和格式`XML`。

+   我们可以使用片段标识符，`https://host/app/class/id/#XML`。

+   我们可以在头部提供它。例如，`Accept`头可以用来指定表示形式。

这些都没有*明显*的优势。与现有的 RESTful web 服务的兼容性可能会建议特定的格式。框架解析 URI 模式的相对容易可能会建议一种格式。

JSON 被许多 JavaScript 表示层所偏爱。其他表示形式，如 XML 或 YAML，对其他表示层或其他类型的客户端也可能有帮助。在某些情况下，可能会有另一种表示形式。例如，特定客户端应用程序可能需要 MXML 或 XAML。

## 实现 REST 服务器 - WSGI 和 mod_wsgi

由于 REST 是建立在 HTTP 之上的，因此 REST 服务器是对 HTTP 服务器的扩展。为了进行强大、高性能、安全的操作，通常的做法是在诸如**Apache httpd**或**nginx**之类的服务器上构建。这些服务器默认不支持 Python；它们需要一个扩展模块来与 Python 应用程序进行接口。

在 Web 服务器和 Python 之间广泛使用的接口是 WSGI。有关更多信息，请参见[`www.wsgi.org`](http://www.wsgi.org)。Python 标准库包括一个 WSGI 参考实现。请参阅 PEP 3333，[`www.python.org/dev/peps/pep-3333/`](http://www.python.org/dev/peps/pep-3333/)，了解这个参考实现在 Python 3 中的工作方式。

WSGI 背后的理念是围绕一个相对简单和可扩展的 Python API 标准化 HTTP 请求-响应处理。这使我们能够从相对独立的组件中构建复杂的 Python 解决方案。目标是创建一个嵌套的应用程序系列，对请求进行增量处理。这创建了一种管道，其中每个阶段都向请求环境添加信息。

每个 WSGI 应用程序必须具有此 API：

```py
result = application(environ, start_response)
```

`environ`变量必须是包含环境信息的`dict`。必须使用`start_response`函数来开始准备向客户端发送响应；这是发送响应状态码和标头的方式。返回值必须是一个字符串的可迭代对象；也就是说，响应的正文。

在 WSGI 标准中，术语*应用程序*被灵活地使用。一个单一的服务器可能有许多 WSGI 应用程序。WSGI 的目的不是鼓励或要求在符合 WSGI 的应用程序的低级别进行编程。其目的是使用更大、更复杂的 Web 框架。所有的 Web 框架都会使用 WSGI API 定义来确保兼容性。

WSGI 参考实现不打算成为公共面向的 Web 服务器。此服务器不直接处理 SSL；需要一些工作来使用适当的 SSL 加密包装套接字。为了访问端口 80（或端口 443），进程必须以`setuid`模式执行，使用特权用户 ID。一种常见的做法是在 Web 服务器中安装 WSGI 扩展模块或使用支持 WSGI API 的 Web 服务器。这意味着 Web 请求通过标准 WSGI 接口从 Web 服务器路由到 Python。这允许 Web 服务器提供静态内容。通过 WSGI 接口可用的 Python 应用程序将提供动态内容。

以下是一些要么用 Python 编写，要么具有 Python 插件的 Web 服务器的列表，[`wiki.python.org/moin/WebServers`](https://wiki.python.org/moin/WebServers)。这些服务器（或插件）旨在提供强大、安全的、面向公众的 Web 服务器。

另一种选择是构建一个独立的 Python 服务器，并使用重定向将请求从面向公众的服务器转移到单独的 Python 守护程序。在使用 Apache httpd 时，可以通过`mod_wsgi`模块创建一个单独的 Python 守护程序。由于我们专注于 Python，我们将避免 nginx 或 Apache httpd 的细节。

### 创建一个简单的 REST 应用程序和服务器

我们将编写一个非常简单的 REST 服务器，提供轮盘赌的旋转。这是一个对简单请求做出响应的服务的示例。我们将专注于 Python 中的 RESTful web 服务器编程。还需要一些额外的细节来将此软件插入到较大的 Web 服务器中，例如 Apache httpd 或 nginx。

首先，我们将定义一个简化的轮盘赌轮：

```py
class Wheel:
    """Abstract, zero bins omitted."""
    def __init__( self ):
        self.rng= random.Random()
        self.bins= [
            {str(n): (35,1),
            self.redblack(n): (1,1),
            self.hilo(n): (1,1),
            self.evenodd(n): (1,1),
            } for n in range(1,37)
        ]
    @staticmethod
    def redblack(n):
        return "Red" if n in (1, 3, 5, 7, 9,  12, 14, 16, 18,
            19, 21, 23, 25, 27,  30, 32, 34, 36) else "Black"
    @staticmethod
    def hilo(n):
        return "Hi" if n >= 19 else "Lo"
    @staticmethod
    def evenodd(n):
        return "Even" if n % 2 == 0 else "Odd"
    def spin( self ):
        return self.rng.choice( self.bins )
```

`Wheel`类是一个箱子的列表。每个箱子都是`dict`；键是如果球落在该箱子中将获胜的赌注。箱子中的值是支付比例。我们只向您展示了一个简短的赌注列表。可用的轮盘赌赌注的完整列表相当庞大。

此外，我们省略了零或双零箱。有两种不同类型的常用轮子。以下是定义常用轮子不同类型的两个混合类：

```py
class Zero:
    def __init__( self ):
        super().__init__()
        self.bins += [ {'0': (35,1)} ]

class DoubleZero:
    def __init__( self ):
        super().__init__()
        self.bins += [ {'00': (35,1)} ]
```

`Zero` mixin 包括单个零的初始化。`DoubleZero` mixin 包括双零。这些是相对简单的箱子；只有在对数字本身下注时才会有回报。

我们在这里使用混合类，因为我们将在以下一些示例中调整`Wheel`的定义。通过使用混合类，我们可以确保对基类`Wheel`的每个扩展都能保持一致。有关混合样式设计的更多信息，请参见第八章，*装饰器和混合类-横切面*。

以下是定义常用轮子不同类型的两个子类：

```py
class American( Zero, DoubleZero, Wheel ):
    pass

class European( Zero, Wheel ):
    pass
```

这两个定义使用混合类扩展了基本的`Wheel`类，这些混合类将为每种类型的轮子正确初始化箱子。`Wheel`的这些具体子类可以如下使用：

```py
american = American()
european = European()
print( "SPIN", american.spin() )
```

`spin()`的每次评估都会产生一个简单的字典，如下所示：

```py
{'Even': (1, 1), 'Lo': (1, 1), 'Red': (1,   1), '12': (35, 1)}
```

这个`dict`中的键是赌注名称。值是一个包含支付比例的两元组。前面的例子向我们展示了红色 12 作为赢家；它也是低和偶数。如果我们在 12 上下注，我们的赢利将是我们的赌注的 35 倍，支付比例为 35 比 1。其他赌注的支付比例为 1 比 1：我们会翻倍赢钱。

我们将定义一个 WSGI 应用程序，使用简单的路径来确定使用哪种类型的轮子。例如`http://localhost:8080/european/`这样的 URI 将使用欧洲轮盘。任何其他路径将使用美式轮盘。

以下是使用`Wheel`实例的 WSGI 应用程序：

```py
import sys
import wsgiref.util
import json
def wheel(environ, start_response):
    request= wsgiref.util.shift_path_info(environ) # 1\. Parse.
    print( "wheel", request, file=sys.stderr ) # 2\. Logging.
    if request.lower().startswith('eu'): # 3\. Evaluate.
        winner= european.spin()
    else:
        winner= american.spin()
    status = '200 OK' # 4\. Respond.
    headers = [('Content-type', 'application/json; charset=utf-8')]
    start_response(status, headers)
    return [ json.dumps(winner).encode('UTF-8') ]
```

这向我们展示了 WSGI 应用程序中的一些基本要素。

首先，我们使用`wsgiref.util.shift_path_info()`函数来检查`environ['PATH_INFO']`的值。这将解析请求中路径信息的一个级别；它将返回找到的字符串值，或者在没有提供路径的情况下返回`None`。

其次，日志行告诉我们，如果我们想生成日志，*必须*写入`sys.stderr`。写入`sys.stdout`的任何内容都将被用作 WSGI 应用程序的响应的一部分。在调用`start_response()`之前打印的任何内容都将导致异常，因为状态和标头尚未发送。

第三，我们评估请求以计算响应。我们使用两个全局变量`european`和`american`，以提供一致随机化的响应序列。如果我们尝试为每个请求创建一个唯一的`Wheel`实例，那么我们就不恰当地使用了随机数生成器。

第四，我们用适当的状态码和 HTTP 标头制定了一个响应。响应的主体是一个 JSON 文档，我们使用 UTF-8 进行编码，以生成符合 HTTP 要求的适当字节流。

我们可以使用以下函数启动此服务器的演示版本：

```py
from wsgiref.simple_server import make_server
def roulette_server(count=1):
    httpd = make_server('', 8080, wheel)
    if count is None:
        httpd.serve_forever()
    else:
        for c in range(count):
            httpd.handle_request()
```

`wsgiref.simple_server.make_server()`函数创建服务器对象。该对象将调用可调用的`wheel()`来处理每个请求。我们使用本地主机名`''`和非特权端口`8080`。使用特权端口`80`需要`setuid`权限，并且最好由**Apache httpd**服务器处理。

构建服务器后，它可以自行运行；这是`httpd.serve_forever()`方法。然而，对于单元测试，通常最好处理有限数量的请求，然后停止服务器。

我们可以在终端窗口的命令行中运行此函数。一旦我们运行该函数，我们可以使用浏览器查看我们向`http://localhost:8080/`发出请求时的响应。这在创建技术性的突发情况或调试时非常有帮助。

### 实现 REST 客户端

在查看更智能的 REST 服务器应用程序之前，我们将看一下编写 REST 客户端。以下是一个将向 REST 服务器发出简单的`GET`请求的函数：

```py
import http.client
import json
def json_get(path="/"):
    rest= http.client.HTTPConnection('localhost', 8080)
    rest.request("GET", path)
    response= rest.getresponse()
    print( response.status, response.reason )
    print( response.getheaders() )
    raw= response.read().decode("utf-8")
    if response.status == 200:
        document= json.loads(raw)
        print( document )
    else:
        print( raw )
```

这向我们展示了使用 RESTful API 的本质。`http.client`模块有一个四步过程：

+   通过`HTTPConnection()`建立连接

+   发送带有命令和路径的请求

+   获取响应

+   要读取响应中的数据

请求可以包括附加的文档（用于 POST）以及其他标头。在此函数中，我们打印了响应的几个部分。在此示例中，我们读取了状态码和原因文本，并将其打印出来。大多数情况下，我们期望状态码为 200，原因为`OK`。我们还读取并打印了所有标头。

最后，我们将整个响应读入临时字符串`raw`。如果状态码为 200，我们使用`json`模块从响应字符串中加载对象。这将恢复从服务器发送的任何 JSON 编码对象。

如果状态码不是 200，我们只需打印可用文本。这可能是一个错误消息或其他有用于调试的信息。

### 演示和单元测试 RESTful 服务

进行 RESTful 服务器的突发演示相对较容易。我们可以导入服务器类和函数定义，并从终端窗口运行服务器函数。我们可以连接到`http://localhost:8080`来查看响应。

为了进行适当的单元测试，我们希望客户端和服务器之间进行更正式的交换。对于受控的单元测试，我们希望启动然后停止服务器进程。然后我们可以对服务器进行测试，并检查客户端的响应。

我们可以使用`concurrent.futures`模块创建一个单独的子进程来运行服务器。以下是一个代码片段，展示了可以成为单元测试用例一部分的处理方式：

```py
    import concurrent.futures
    import time
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit( roulette_server, 4 )
        time.sleep(2) # Wait for the server to start
        json_get()
        json_get()
        json_get("/european/")
        json_get("/european/")
```

我们通过创建`concurrent.futures.ProcessPoolExecutor`的实例来创建一个单独的进程。然后，我们可以提交一个函数到这个服务器，带有适当的参数值。

在这种情况下，我们执行了我们的`json_get()`客户端函数来读取默认路径`/`两次。然后我们在"`/european/`"路径上执行了两次`GET`操作。

`executor.submit()`函数使进程池评估`roulette_server(4)`函数。这将处理四个请求，然后终止。因为`ProcessPoolExecutor`是一个上下文管理器，我们可以确保所有资源都会被正确清理。单元测试的输出日志以以下方式分组：

```py
wheel 'european'
127.0.0.1 - - [08/Dec/2013 09:32:08] "GET /european/ HTTP/1.1" 200 62
200 OK
[('Date', 'Sun, 08 Dec 2013 14:32:08 GMT'), ('Server', 'WSGIServer/0.2 CPython/3.3.3'), ('Content-type', 'application/json; charset=utf-8'), ('Content-Length', '62')]
{'20': [35, 1], 'Even': [1, 1], 'Black': [1, 1], 'Hi': [1, 1]}
```

`wheel 'european'`行是我们的`wheel()`WSGI 应用程序的日志输出。`127.0.0.1 - - [08/Dec/2013 09:32:08] "GET /european/ HTTP/1.1" 200 62`日志行是默认从 WSGI 服务器写入的，它告诉我们请求已经完全处理，没有错误。

客户端`json_get()`函数编写了接下来的三行。`200 OK`行是第一个`print()`函数。这些行是作为服务器响应的一部分发送的标头。最后，我们向您展示了从服务器发送到客户端的解码字典对象。在这种情况下，赢家是 20 黑。

另外，请注意，我们的原始元组在 JSON 编码和解码过程中被转换为列表。我们原始的字典是`'20': (35, 1)`。在编码和解码后的结果是`'20': [35, 1]`。

请注意，正在测试的模块将由`ProcessPool`服务器导入。这个导入将找到命名函数`roulette_server()`。因为服务器将导入被测试的模块，被测试的模块必须正确使用`__name__ == "__main__"`保护，以确保在导入期间不会执行任何额外的处理；它只能提供定义。我们必须确保在定义服务器的脚本中使用这种构造：

```py
if __name__ == "__main__":
    roulette_server()
```

## 使用 Callable 类来实现 WSGI 应用程序

我们可以将 WSGI 应用程序实现为`Callable`对象，而不是独立的函数。这允许我们在 WSGI 服务器中进行有状态的处理，而不会造成全局变量的混乱。在我们之前的例子中，`get_spin()`WSGI 应用程序依赖于两个全局变量，`american`和`european`。应用程序和全局变量之间的绑定可能是神秘的。

定义类的目的是将处理和数据封装到一个单一的包中。我们可以使用`Callable`对象以更好的方式封装我们的应用程序。这可以使有状态的`Wheel`和 WSGI 应用程序之间的绑定更清晰。这是对`Wheel`类的扩展，使其成为一个可调用的 WSGI 应用程序：

```py
from collections.abc import Callable
class Wheel2( Wheel, Callable ):
    def __call__(self, environ, start_response):
        winner= self.spin() # 3\. Evaluate.
        status = '200 OK' # 4\. Respond.
        headers = [('Content-type', 'application/json; charset=utf-8')]
        start_response(status, headers)
        return [ json.dumps(winner).encode('UTF-8') ]
```

我们扩展了基本的`Wheel`类，以包括 WSGI 接口。这不会对请求进行任何解析；WSGI 处理已经被简化为只有两个步骤：评估和响应。我们将在更高级别的包装应用程序中处理解析和日志记录。这个`Wheel2`应用程序只是选择一个结果并将其编码为结果。

请注意，我们已经为`Wheel2`类添加了一个独特的设计特性。这是一个不属于`Wheel`的*is-a*定义的关注点的例子。这更像是一个*acts-as*特性。这可能应该被定义为一个 mixin 或装饰器，而不是类定义的一流特性。

这里有两个子类，实现了轮盘的美式和欧式变体：

```py
class American2( Zero, DoubleZero, Wheel2 ):
    pass

class European2( Zero, Wheel2 ):
    pass
```

这两个子类依赖于超类中的`__call__()`方法函数。与前面的例子一样，我们使用 mixin 来向轮盘添加适当的零箱。

我们已经将轮子从一个简单的对象变成了一个 WSGI 应用程序。这意味着我们的高级包装应用程序可以更简单一些。高级应用程序不是评估其他对象，而是简单地将请求委托给对象。下面是一个修改后的包装应用程序，它选择要旋转的轮子并委托请求：

```py
class Wheel3( Callable ):
    def __init__( self ):
        self.am = American2()
        self.eu = European2()
    def __call__(self, environ, start_response):
        request= wsgiref.util.shift_path_info(environ) # 1\. Parse
        print( "Wheel3", request, file=sys.stderr ) # 2\. Logging
        if request.lower().startswith('eu'): # 3\. Evaluate
            response= self.eu(environ,start_response)
        else:
            response= self.am(environ,start_response)
        return response # 4\. Respond
```

创建这个`Wheel3`类的实例时，它将创建两个轮子。每个轮子都是一个 WSGI 应用程序。

当处理请求时，`Wheel3` WSGI 应用程序将解析请求。然后将这两个参数（`environ`和`start_response`函数）传递给另一个应用程序来执行实际的评估并计算响应。在许多情况下，这种委托还包括从请求路径或标头解析的参数和参数更新`environ`变量。最后，这个`Wheel3.__call__()`函数将返回被调用的另一个应用程序的响应。

这种委托方式是 WSGI 应用程序的特点。这就是 WSGI 应用程序如此优雅地嵌套在一起的原因。请注意，包装应用程序有两个地方可以注入处理：

+   在调用另一个应用程序之前，它将调整环境以添加信息。

+   调用另一个应用程序后，它可以调整响应文档

通常，我们喜欢在包装应用程序中调整环境。然而，在这种情况下，没有真正需要使用额外信息更新环境，因为请求是如此微不足道。

### 设计 RESTful 对象标识符

对象序列化涉及为每个对象定义某种标识符。对于`shelve`或`sqlite`，我们需要为每个对象定义一个字符串键。RESTful web 服务器也提出了相同的要求，以定义一个可用于明确跟踪对象的可行键。

一个简单的替代键也可以用于 RESTful web 服务标识符。它可以轻松地与`shelve`或`sqlite`使用的键并行。

重要的是要明白“酷的 URI 不会改变”的概念。参见[`www.w3.org/Provider/Style/URI.html`](http://www.w3.org/Provider/Style/URI.html)。

对我们来说，定义一个永远不会改变的 URI 是很重要的。重要的是对象的有状态方面永远不要作为 URI 的一部分。例如，微博应用程序可能支持多个作者。如果我们按作者将博客帖子组织成文件夹，就会为共享作者身份创建问题，当一个作者接管另一个作者的内容时，就会产生更大的问题。我们不希望 URI 在纯粹的管理功能（如*所有权*）发生变化时切换。

RESTful 应用程序可能提供许多索引或搜索条件。然而，资源或对象的基本标识不应随索引的更改或重新组织而改变。

对于相对简单的对象，我们通常可以找到某种标识符，通常是数据库替代键。对于博客帖子，通常使用发布日期（因为它不会改变）和标题的版本，标点和空格用`_`字符替换。其目的是创建一个标识符，无论网站如何重新组织，都不会改变。添加或更改索引不会改变微博帖子的基本标识。

对于更复杂的容器对象，我们必须决定可以引用这些更复杂对象的粒度。继续微博示例，我们有整个博客，其中包含许多个别的帖子。

博客的 URI 可以是这样简单的：

```py
/microblog/blog/bid/
```

最顶层的名称（`微博`）是整个应用程序。然后，我们有资源类型（`博客`），最后是特定实例的 ID。

然而，帖子的 URI 名称有几种选择：

```py
/microblog/post/title_string/
/microblog/post/bid/title_string/
/microblog/blog/bid/post/title_string/
```

当不同的博客有相同标题的帖子时，第一个 URI 效果不佳。在这种情况下，作者可能会看到他们的标题被添加了额外的`_2`或其他装饰，以强制标题变得唯一。这通常是不可取的。

第二个 URI 使用博客 ID（`bid`）作为上下文或命名空间，以确保在博客的上下文中将`Post`标题视为唯一的。这种技术通常被扩展以包括额外的细分，比如日期，以进一步缩小搜索空间。

第三个示例在两个级别上使用了显式的类/对象命名：`blog/bid`和`post/title_string`。这样做的缺点是路径更长，但它的优点是允许一个复杂的容器在不同的内部集合中有多个项目。

请注意，REST 服务的效果是定义持久存储的 API。实际上，URI 类似于接口方法的名称。它们必须选择得清晰、有意义和耐用。

### 多层 REST 服务

这是一个更智能、多层次的 REST 服务器应用程序。我们将分段展示给你。首先，我们需要用一个 Roulette 桌子来补充我们的`Wheel`类：

```py
from collections import defaultdict
class Table:
    def __init__( self, stake=100 ):
        self.bets= defaultdict(int)
        self.stake= stake
    def place_bet( self, name, amount ):
        self.bets[name] += amount
    def clear_bets( self, name ):
        self.bets= defaultdict(int)
    def resolve( self, spin ):
        """spin is a dict with bet:(x:y)."""
        details= []
        while self.bets:
            bet, amount= self.bets.popitem()
            if bet in spin:
                x, y = spin[bet]
                self.stake += amount*x/y
                details.append( (bet, amount, 'win') )
            else:
                self.stake -= amount
                details.append( (bet, amount, 'lose') )
        return details
```

`Table`类跟踪来自单个匿名玩家的赌注。每个赌注都是轮盘桌上一个空间的字符串名称和一个整数金额。在解决赌注时，`Wheel`类提供了一个单次旋转给`resolve()`方法。下注与旋转中的获胜赌注进行比较，并且随着赌注的赢得或失去，玩家的赌注会进行调整。

我们将定义一个 RESTful 的 Roulette 服务器，它展示了通过`HTTP POST`方法实现的有状态事务。我们将把 Roulette 游戏分成三个 URI：

+   `/player/`

+   向这个 URI 发送`GET`请求将检索一个 JSON 编码的`dict`，其中包含有关玩家的信息，包括他们的赌注和迄今为止玩的轮数。未来的扩展将是定义一个适当的`Player`对象并返回一个序列化的实例。

+   未来的扩展将处理`POST`以创建额外的下注玩家。

+   `/bet/`

+   向这个 URI 发送`POST`请求将包括一个 JSON 编码的`dict`或一个创建赌注的字典列表。每个赌注字典将有两个键：`bet`和`amount`。

+   `GET`将返回一个 JSON 编码的`dict`，显示迄今为止下注和金额。

+   `/wheel/`

+   向这个 URI 发送没有数据的`POST`请求将旋转并计算支付。这是作为`POST`实现的，以加强它正在对可用的赌注和玩家进行有状态的更改的感觉。

+   `GET`可能会重复之前的结果，显示上次旋转，上次支付和玩家的赌注。这可能是非否认方案的一部分；它返回旋转收据的额外副本。

以下是我们 WSGI 应用程序系列的两个有用的类定义：

```py
class WSGI( Callable ):
    def __call__( self, environ, start_response ):
        raise NotImplementedError

class RESTException( Exception ):
    pass
```

我们对`Callable`进行了简单的扩展，以明确表示我们将定义一个 WSGI 应用程序类。我们还定义了一个异常，我们可以在 WSGI 应用程序中使用它来发送与`wsgiref`实现提供的 Python 错误不同的错误状态代码。这是 Roulette 服务器的顶层：

```py
class Roulette( WSGI ):
    def __init__( self, wheel ):
        self.table= Table(100)
        self.rounds= 0
        self.wheel= wheel
    def __call__( self, environ, start_response ):
        #print( environ, file=sys.stderr )
        app= wsgiref.util.shift_path_info(environ)
        try:
            if app.lower() == "player":
                return self.player_app( environ, start_response )
            elif app.lower() == "bet":
                return self.bet_app( environ, start_response )
            elif app.lower() == "wheel":
                return self.wheel_app( environ, start_response )
            else:
                raise RESTException("404 NOT_FOUND",
                    "Unknown app in {SCRIPT_NAME}/{PATH_INFO}".format_map(environ))
        except RESTException as e:
            status= e.args[0]
            headers = [('Content-type', 'text/plain; charset=utf-8')]
            start_response( status, headers, sys.exc_info() )
            return [ repr(e.args).encode("UTF-8") ]
```

我们定义了一个 WSGI 应用程序，它包装了其他应用程序。`wsgiref.util.shift_path_info()`函数将解析路径，在`/`上断开以获取第一个单词。基于此，我们将调用另外三个 WSGI 应用程序中的一个。在这种情况下，每个应用程序将是类定义内的一个方法函数。

我们提供了一个总体异常处理程序，它将把任何`RESTException`实例转换为适当的 RESTful 响应。我们没有捕获的异常将转换为`wsgiref`提供的通用状态码 500 错误。这是`player_app`方法函数：

```py
    def player_app( self, environ, start_response ):
        if environ['REQUEST_METHOD'] == 'GET':
            details= dict( stake= self.table.stake, rounds= self.rounds )
            status = '200 OK'
            headers = [('Content-type', 'application/json; charset=utf-8')]
            start_response(status, headers)
            return [ json.dumps( details ).encode('UTF-8') ]
        else:
            raise RESTException("405 METHOD_NOT_ALLOWED",
                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))
```

我们创建了一个响应对象`details`。然后我们将这个对象序列化为一个 JSON 字符串，并进一步使用 UTF-8 编码该字符串为字节。

在极少数情况下，尝试对`/player/`路径进行 Post（或 Put 或 Delete）将引发异常。这将在顶层`__call__()`方法中捕获，并转换为错误响应。

这是`bet_app()`函数：

```py
    def bet_app( self, environ, start_response ):
        if environ['REQUEST_METHOD'] == 'GET':
            details = dict( self.table.bets )
        elif environ['REQUEST_METHOD'] == 'POST':
            size= int(environ['CONTENT_LENGTH'])
            raw= environ['wsgi.input'].read(size).decode("UTF-8")
            try:
                data = json.loads( raw )
                if isinstance(data,dict): data= [data]
                for detail in data:
                    self.table.place_bet( detail['bet'], int(detail['amount']) )
            except Exception as e:
                raise RESTException("403 FORBIDDEN",
                 Bet {raw!r}".format(raw=raw))
            details = dict( self.table.bets )
        else:
            raise RESTException("405 METHOD_NOT_ALLOWED",
                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))
        status = '200 OK'
        headers = [('Content-type', 'application/json; charset=utf-8')]
        start_response(status, headers)
        return [ json.dumps(details).encode('UTF-8') ]
```

这做了两件事，取决于请求方法。当使用`GET`请求时，结果是当前下注的字典。当使用`POST`请求时，必须有一些数据来定义下注。当尝试任何其他方法时，将返回错误。

在`POST`情况下，下注信息作为附加到请求的数据流提供。我们必须执行几个步骤来读取和处理这些数据。第一步是使用`environ['CONTENT_LENGTH']`的值来确定要读取多少字节。第二步是解码字节以获得发送的字符串值。

我们使用了请求的 JSON 编码。这绝对不是浏览器或 Web 应用程序服务器处理来自 HTML 表单的`POST`数据的方式。当使用浏览器从 HTML 表单发布数据时，编码是`urllib.parse`模块实现的一组简单的转义。`urllib.parse.parse_qs()`模块函数将解析带有 HTML 数据的编码查询字符串。

对于 RESTful Web 服务，有时会使用`POST`兼容数据，以便基于表单的处理与 RESTful 处理非常相似。在其他情况下，会使用单独的编码，如 JSON，以创建比 Web 表单产生的引号数据更容易处理的数据结构。

一旦我们有了字符串`raw`，我们使用`json.loads()`来获取该字符串表示的对象。我们期望两类对象中的一个。一个简单的`dict`对象将定义一个单独的下注。一系列`dict`对象将定义多个下注。作为一个简单的概括，我们将单个`dict`转换为单例序列。然后，我们可以使用一般的`dict`实例序列来放置所需的下注。

请注意，我们的异常处理将保留一些下注，但会发送一个总体的`403 Forbidden`消息。更好的设计是遵循**Memento**设计模式。下注时，我们还会创建一个可以撤销任何下注的备忘录对象。备忘录的一个实现是使用**Before Image**设计模式。备忘录可以包括在应用更改之前的所有下注的副本。在发生异常时，我们可以删除损坏的版本并恢复以前的版本。当处理可变对象的嵌套容器时，这可能会很复杂，因为我们必须确保复制任何可变对象。由于此应用程序仅使用不可变的字符串和整数，因此`table.bets`的浅复制将非常有效。

对于`POST`和`GET`方法，响应是相同的。我们将`table.bets`字典序列化为 JSON 并发送回 REST 客户端。这将确认已下注的预期下注。

这节课的最后一部分是`wheel_app()`方法：

```py
    def wheel_app( self, environ, start_response ):
        if environ['REQUEST_METHOD'] == 'POST':
            size= environ['CONTENT_LENGTH']
            if size != '':
                raw= environ['wsgi.input'].read(int(size))
                raise RESTException("403 FORBIDDEN",
                    "Data '{raw!r}' not allowed".format(raw=raw))
            spin= self.wheel.spin()
            payout = self.table.resolve( spin )
            self.rounds += 1
            details = dict( spin=spin, payout=payout,
                stake= self.table.stake, rounds= self.rounds )
            status = '200 OK'
            headers = [('Content-type', 'application/json; charset=utf-8')]
            start_response(status, headers)
            return [ json.dumps( details ).encode('UTF-8') ]
        else:
            raise RESTException("405 METHOD_NOT_ALLOWED",
                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))
```

该方法首先检查它是否被调用以提供没有数据的`post`。为了确保套接字被正确关闭，所有数据都被读取并忽略。这可以防止一个编写不良的客户端在套接字关闭时崩溃。

一旦这些琐事处理完毕，剩下的处理就是执行新的轮盘旋转，解决各种下注，并生成包括旋转、支付、玩家赌注和回合数的响应。这份报告被构建为一个`dict`对象。然后将其序列化为 JSON，编码为 UTF-8，并发送回客户端。

请注意，我们已经避免处理多个玩家。这将添加一个类和另一个`/player/`路径下的`POST`方法。这将增加一些定义和簿记。创建新玩家的`POST`处理将类似于下注处理。这是一个有趣的练习，但它并没有引入任何新的编程技术。

### 创建轮盘服务器

一旦我们有了可调用的`Roulette`类，我们可以按照以下方式创建一个 WSGI 服务器：

```py
def roulette_server_3(count=1):
    from wsgiref.simple_server import make_server
    from wsgiref.validate import validator
    wheel= American()
    roulette= Roulette(wheel)
    debug= validator(roulette)
    httpd = make_server('', 8080, debug)
    if count is None:
        httpd.serve_forever()
    else:
        for c in range(count):
            httpd.handle_request()
```

此函数创建我们的 Roulette WSGI 应用程序`roulette`。它使用`wsgiref.simple_server.make_server()`创建一个服务器，该服务器将对每个请求使用`roulette`可调用。

在这种情况下，我们还包括了`wsgiref.validate.validator()` WSGI 应用程序。该应用程序验证了轮盘应用程序使用的接口；它使用 assert 语句装饰各种 API 以提供一些诊断信息。它还在 WSGI 应用程序出现更严重的编程问题时生成稍微更易读的错误消息。

### 创建轮盘客户端

定义一个具有 RESTful 客户端 API 的模块是常见做法。通常，客户端 API 将具有专门针对所请求服务的函数。

我们将定义一个通用的客户端函数，而不是定义一个专门的客户端，该函数将与各种 RESTful 服务器一起工作。这可能成为一个特定于 Roulette 的客户端的基础。以下是一个通用的客户端函数，它将与我们的`Roulette`服务器一起工作：

```py
def roulette_client(method="GET", path="/", data=None):
    rest= http.client.HTTPConnection('localhost', 8080)
    if data:
        header= {"Content-type": "application/json; charset=utf-8'"}
        params= json.dumps( data ).encode('UTF-8')
        rest.request(method, path, params, header)
    else:
        rest.request(method, path)
    response= rest.getresponse()
    raw= response.read().decode("utf-8")
    if 200 <= response.status < 300:
        document= json.loads(raw)
        return document
    else:
        print( response.status, response.reason )
        print( response.getheaders() )
        print( raw )
```

此客户端进行`GET`或`POST`请求，并将`POST`请求的数据编码为 JSON 文档。请注意，请求数据的 JSON 编码绝对不是浏览器处理 HTML 表单的`POST`数据的方式。浏览器使用`urllib.parse.urlencode()`模块函数实现的编码。

我们的客户端函数在半开范围内解码 JSON 文档并返回它，这些是成功的状态代码。我们可以按以下方式操作我们的客户端和服务器：

```py
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit( roulette_server_3, 4 )
        time.sleep(3) # Wait for the server to start
        print( roulette_client("GET", "/player/" ) )
        print( roulette_client("POST", "/bet/", {'bet':'Black', 'amount':2}) )
        print( roulette_client("GET", "/bet/" ) )
        print( roulette_client("POST", "/wheel/" ) )
```

首先，我们创建`ProcessPool`作为练习的上下文。我们向该服务器提交一个请求；实际上，请求是`roulette_server_3(4)`。一旦服务器启动，我们就可以操作该服务器。

在这种情况下，我们进行了四次请求。我们检查玩家的状态。我们下注然后检查下注。最后，我们转动轮盘。在每个步骤中，我们打印 JSON 响应文档。

日志如下：

```py
127.0.0.1 - - [09/Dec/2013 08:21:34] "GET /player/ HTTP/1.1" 200 27
{'stake': 100, 'rounds': 0}
127.0.0.1 - - [09/Dec/2013 08:21:34] "POST /bet/ HTTP/1.1" 200 12
{'Black': 2}
127.0.0.1 - - [09/Dec/2013 08:21:34] "GET /bet/ HTTP/1.1" 200 12
{'Black': 2}
127.0.0.1 - - [09/Dec/2013 08:21:34] "POST /wheel/ HTTP/1.1" 200 129
{'stake': 98, 'payout': [['Black', 2, 'lose']], 'rounds': 1, 'spin': {'27': [35, 1], 'Odd': [1, 1], 'Red': [1, 1], 'Hi': [1, 1]}}
```

这向我们展示了我们的服务器如何响应请求，如何在桌子上下注，如何随机旋转轮盘，并如何正确地更新玩家的结果。

## 创建安全的 REST 服务

我们可以将应用程序安全性分解为两个考虑因素：身份验证和授权。我们需要知道用户是谁，并且我们需要确保用户被授权执行特定的 WSGI 应用程序。这是相对简单地使用 HTTP `Authorization`头来处理，以确保这些凭据的加密传输。

如果我们使用 SSL，我们可以简单地使用 HTTP 基本授权模式。`Authorization`头的这个版本可以在每个请求中包含用户名和密码。对于更复杂的措施，我们可以使用 HTTP 摘要授权，它需要与服务器交换以获取一个称为**nonce**的数据片段，用于以更安全的方式创建摘要。

通常，我们会尽早在流程中处理身份验证。这意味着一个前端 WSGI 应用程序会检查`Authorization`头并更新环境或返回错误。理想情况下，我们将使用一个提供此功能的复杂 Web 框架。有关这些 Web 框架考虑的更多信息，请参见下一节。

关于安全性的最重要的建议可能是以下内容：

### 注意

**永远不要存储密码**

唯一可以存储的是密码加盐的重复加密哈希。密码本身必须是不可恢复的；完全研究*加盐密码哈希*或下载一个可信的库。永远不要存储明文密码或加密密码。

这是一个示例类，向我们展示了加盐密码哈希的工作原理：

```py
from hashlib import sha256
import os
class Authentication:
    iterations= 1000
    def __init__( self, username, password ):
        """Works with bytes. Not Unicode strings."""
        self.username= username
        self.salt= os.urandom(24)
        self.hash= self._iter_hash( self.iterations, self.salt, username, password )
    @staticmethod
    def _iter_hash( iterations, salt, username, password ):
        seed= salt+b":"+username+b":"+password
        for i in range(iterations):
            seed= sha256( seed ).digest()
        return seed
    def __eq__( self, other ):
        return self.username == other.username and self.hash == other.hash
    def __hash__( self, other ):
        return hash(self.hash)
    def __repr__( self ):
        salt_x= "".join( "{0:x}".format(b) for b in self.salt )
        hash_x= "".join( "{0:x}".format(b) for b in self.hash )
        return "{username} {iterations:d}:{salt}:{hash}".format(
            username=self.username, iterations=self.iterations,
            salt=salt_x, hash=hash_x)
    def match( self, password ):
        test= self._iter_hash( self.iterations, self.salt, self.username, password )
        return self.hash == test # **Constant Time is Best

```

这个类为给定的用户名定义了一个`Authentication`对象。该对象包含用户名、每次设置或重置密码时创建的唯一随机盐，以及盐加上密码的最终哈希。这个类还定义了一个`match()`方法，确定给定的密码是否会产生与原始密码相同的哈希。

请注意，密码没有被存储。只有密码的哈希值被保留。我们在比较函数上提供了一个注释（“`# Constant Time is Best`”）。一个在恒定时间内运行的算法——并且不是特别快——对于这种比较是理想的。我们还没有实现它。

我们还包括了一个相等测试和一个哈希测试，以强调这个对象是不可变的。我们不能调整任何值。当用户更改密码时，我们只能丢弃并重建整个`Authentication`对象。另一个设计特性是使用`__slots__`来保存存储空间。

请注意，这些算法使用的是字节字符串，而不是 Unicode 字符串。我们要么使用字节，要么使用 Unicode 用户名或密码的 ASCII 编码。下面是我们可能创建一个用户集合的方法：

```py
class Users( dict ):
    def __init__( self, *args, **kw ):
        super().__init__( *args, **kw )
        # Can never match -- keys are the same.
        self[""]= Authentication( b"__dummy__", b"Doesn't Matter" )
    def add( self, authentication ):
        if authentication.username == "":
            raise KeyError( "Invalid Authentication" )
        self[authentication.username]= authentication
    def match( self, username, password ):
        if username in self and username != "":
            return self[username].match(password)
        else:
            return self[""].match(b"Something which doesn't match")
```

我们创建了一个`dict`的扩展，引入了一个`add()`方法来保存一个`Authentication`实例和一个匹配方法，确定用户是否在这个字典中，以及他们的凭证是否匹配。

请注意，我们的匹配需要是一个恒定时间的比较。我们为一个未知的用户名提供了一个额外的虚拟用户。通过对虚拟用户进行匹配，执行时间不会提供太多关于凭证错误的提示。如果我们简单地返回`False`，那么不匹配的用户名会比不匹配的密码响应更快。

我们明确禁止设置用户名为`""`的身份验证，或匹配用户名为`""`。这将确保虚拟用户名永远不会被更改为可能匹配的有效条目，任何尝试匹配它都将失败。下面是我们构建的一个示例用户：

```py
users = Users()
users.add( Authentication(b"Aladdin", b"open sesame") )
```

只是为了看看这个类里面发生了什么，我们可以手动创建一个用户：

```py
>>> al= Authentication(b"Aladdin", b"open sesame")
>>> al
b'Aladdin' 1000:16f56285edd9326282da8c6aff8d602a682bbf83619c7f:9b86a2ad1ae0345029ae11de402ba661ade577df876d89b8a3e182d887a9f7
```

盐是一个由 24 个字节组成的字符串，在用户的密码被创建或更改时被重置。哈希是用户名、密码和盐的重复哈希。

### WSGI 身份验证应用程序

一旦我们有了存储用户和凭证的方法，我们就可以检查请求中的`Authentication`头部。下面是一个检查头部并更新验证用户环境的 WSGI 应用程序：

```py
import base64
class Authenticate( WSGI ):
    def __init__( self, users, target_app ):
        self.users= users
        self.target_app= target_app
    def __call__( self, environ, start_response ):
        if 'HTTP_AUTHORIZATION' in environ:
            scheme, credentials = environ['HTTP_AUTHORIZATION'].split()
            if scheme == "Basic":
                username, password= base64.b64decode( credentials ).split(b":")
                if self.users.match(username, password):
                    environ['Authenticate.username']= username
                    return self.target_app(environ, start_response)
        status = '401 UNAUTHORIZED'
        headers = [('Content-type', 'text/plain; charset=utf-8'),
            ('WWW-Authenticate', 'Basic realm="roulette@localhost"')]
        start_response(status, headers)
        return [ "Not authorized".encode('utf-8') ]
```

这个 WSGI 应用程序包含一个用户池，还有一个目标应用程序。当我们创建这个`Authenticate`类的实例时，我们将提供另一个 WSGI 应用程序作为`target_app`；这个包装应用程序只会看到经过身份验证的用户的请求。当调用`Authenticate`应用程序时，它会执行几个测试，以确保请求来自经过身份验证的用户：

+   必须有一个 HTTP`Authorization`头。这个头部保存在`environ`字典中的`HTTP_AUTHORIZATION`键中

+   头部必须使用`Basic`作为认证方案

+   基本方案中的凭证必须是`username+b":"+password`的 base 64 编码；这必须与定义的用户的凭证匹配

如果所有这些测试都通过了，我们可以使用经过身份验证的用户名更新`environ`字典。然后，目标应用程序可以被调用。

然后，包装应用程序可以处理授权细节，知道用户已经通过身份验证。这种关注点的分离是 WSGI 应用程序的一个优雅特性。我们把身份验证放在了一个地方。

## 使用 Web 应用程序框架实现 REST

由于 REST web 服务器是一个 Web 应用程序，我们可以利用任何流行的 Python Web 应用程序框架。从头开始编写 RESTful 服务器是在证明框架提供的问题不可接受之后可以采取的一步。在许多情况下，使用框架进行技术性的尝试可以帮助澄清任何问题，并允许与不使用框架编写的 REST 应用程序进行详细比较。

一些 Python Web 框架包括一个或多个 REST 组件。在某些情况下，RESTful 功能几乎完全内置。在其他情况下，附加项目可以帮助以最少的编程定义 RESTful Web 服务。

这是 Python Web 框架的列表：[`wiki.python.org/moin/WebFrameworks`](https://wiki.python.org/moin/WebFrameworks)。这些项目的目的是提供一个相对完整的环境来构建 Web 应用程序。

这是 Python Web 组件软件包的列表：[`wiki.python.org/moin/WebComponents`](https://wiki.python.org/moin/WebComponents)。这些都是可以用来支持 Web 应用程序开发的部分和片段。

在 PyPI，[`pypi.python.org`](https://pypi.python.org)，搜索 REST 将会找到大量的软件包。显然，已经有许多可用的解决方案。

花时间搜索、下载和学习一些现有的框架可以减少一些开发工作。特别是安全性方面是具有挑战性的。自制的安全算法通常存在严重的缺陷。使用他人验证过的安全工具可能有一些优势。

## 使用消息队列传输对象

`multiprocessing`模块也使用对象的序列化和传输。我们可以使用队列和管道对对象进行序列化，然后将其传输到其他进程。有许多外部项目可以提供复杂的消息队列处理。我们将专注于`multiprocessing`队列，因为它内置于 Python 并且运行良好。

对于高性能应用程序，可能需要更快的消息队列。可能还需要使用比 pickling 更快的序列化技术。在本章中，我们只关注 Python 设计问题。`multiprocessing`模块依赖于`pickle`来编码对象。有关更多信息，请参见第九章，“序列化和保存 - JSON、YAML、Pickle、CSV 和 XML”。我们无法轻松地提供受限制的 unpickler；因此，该模块为我们提供了一些相对简单的安全措施，以防止 unpickle 问题。

在使用`multiprocessing`时有一个重要的设计考虑：通常最好避免多个进程（或多个线程）尝试更新共享对象。同步和锁定问题是如此深刻（并且容易出错），以至于标准笑话是，

> 当程序员面对问题时，他会想：“我会使用多个线程。”

通过 RESTful Web 服务或`multiprocessing`使用进程级同步可以防止同步问题，因为没有共享对象。基本的设计原则是将处理视为离散步骤的管道。每个处理步骤都将有一个输入队列和一个输出队列；该步骤将获取一个对象，执行一些处理，并写入该对象。

`multiprocessing`的哲学与将 POSIX 概念写成`process1 | process2 | process3`的 shell 管道相匹配。这种 shell 管道涉及三个相互连接的并发进程。重要的区别在于，我们不需要使用 STDIN、STDOUT 和对象的显式序列化。我们可以相信`multiprocessing`模块来处理操作系统级的基础设施。

POSIX shell 管道有限，每个管道只有一个生产者和一个消费者。Python 的`multiprocessing`模块允许我们创建包括多个消费者的消息队列。这使我们能够创建一个从一个源进程到多个目标进程的扇出流水线。一个队列也可以有多个消费者，这使我们能够构建一个流水线，其中多个源进程的结果可以由单个目标进程组合。

为了最大化计算机系统的吞吐量，我们需要有足够的待处理工作，以便没有处理器或核心会闲置。当任何给定的操作系统进程正在等待资源时，至少应该有另一个进程准备好运行。

例如，当我们观察我们的赌场游戏模拟时，我们需要通过多次执行玩家策略或投注策略（或两者）来收集具有统计学意义的模拟数据。我们的目标是创建一个处理请求队列，以便我们计算机的处理器（和核心）完全参与处理我们的模拟。

每个处理请求可以是一个 Python 对象。`multiprocessing`模块将对该对象进行 pickle 处理，以便通过队列传输到另一个进程。

我们将在第十四章中重新讨论这个问题，当我们看看`logging`模块如何使用`multiprocessing`队列为单独的生产者进程提供一个集中的日志时。在这些示例中，从一个进程传输到另一个进程的对象将是`logging.LogRecord`实例。

### 定义进程

我们必须将每个处理步骤设计为一个简单的循环，从队列中获取请求，处理该请求，并将结果放入另一个队列。这将大问题分解为多个形成流水线的阶段。由于每个阶段都将同时运行，系统资源使用将被最大化。此外，由于这些阶段涉及简单的从独立队列获取和放置，所以没有复杂的锁定或共享资源问题。一个进程可以是一个简单的函数或可调用对象。我们将专注于将进程定义为`multiprocessing.Process`的子类。这给了我们最大的灵活性。

对于我们赌场游戏的模拟，我们可以将模拟分解为三个步骤的流水线：

1.  一个总体驱动程序将模拟请求放入处理队列。

1.  一组模拟器将从处理队列获取请求，执行模拟，并将统计数据放入结果队列。

1.  汇总器将从结果队列获取结果，并创建最终的结果汇总。

使用进程池允许我们同时运行尽可能多的模拟，以便我们的 CPU 可以处理。模拟器池可以配置，以确保模拟尽快运行。

以下是模拟器进程的定义：

```py
import multiprocessing
class Simulation( multiprocessing.Process ):
    def __init__( self, setup_queue, result_queue ):
        self.setup_queue= setup_queue
        self.result_queue= result_queue
        super().__init__()
    def run( self ):
        """Waits for a termination"""
        print( self.__class__.__name__, "start" )
        item= self.setup_queue.get()
        while item != (None,None):
            table, player = item
            self.sim= Simulate( table, player, samples=1 )
            results= list( self.sim )
            self.result_queue.put( (table, player, results[0]) )
            item= self.setup_queue.get()
        print( self.__class__.__name__, "finish" )
```

我们已经扩展了`multiprocessing.Process`。这意味着我们必须做两件事才能正确地使用多进程：我们必须确保执行`super().__init__()`，并且我们必须重写`run()`。

在`run()`的主体内，我们使用了两个队列。`setup_queue`队列实例将包含`Table`和`Player`对象的两元组。进程将使用这两个对象来运行模拟。它将把结果放入`result_queue`队列实例中。`Simulate`类的 API 如下：

```py
class Simulate:
    def __init__( self, table, player, samples ):
    def __iter__( self ): yields summaries
```

迭代器将产生请求的数量`samples`的统计摘要。我们已经包括了一个**sentinel 对象**通过`setup_queue`到达。这个对象将被用来优雅地关闭处理。如果我们不使用一个 sentinel 对象，我们将被迫终止进程，这可能会破坏锁定和其他系统资源。以下是摘要过程：

```py
class Summarize( multiprocessing.Process ):
    def __init__( self, queue ):
        self.queue= queue
        super().__init__()
    def run( self ):
        """Waits for a termination"""
        print( self.__class__.__name__, "start" )
        count= 0
        item= self.queue.get()
        while item != (None, None, None):
            print( item )
            count += 1
            item= self.queue.get()
        print( self.__class__.__name__, "finish", count )
```

这也扩展了`multiprocessing.Process`。在这种情况下，我们从队列中获取项目并简单地对其进行计数。一个更有用的进程可能会使用多个`collection.Counter`对象来累积更有趣的统计数据。

与`Simulation`类一样，我们还将检测到一个标记并优雅地关闭处理。使用标记对象可以让我们在进程完成工作后立即关闭处理。在一些应用中，子进程可以无限期地运行。

### 构建队列和提供数据

构建队列涉及创建`multiprocessing.Queue`的实例或其子类的实例。对于这个例子，我们可以使用以下内容：

```py
setup_q= multiprocessing.SimpleQueue()
results_q= multiprocessing.SimpleQueue()
```

我们创建了两个定义处理流水线的队列。当我们将模拟请求放入`setup_q`时，我们期望`Simulation`进程会接收请求对并运行模拟。这应该在`results_q`队列中生成一个包含表、玩家和结果的三元组。这个结果三元组应该进一步导致`Summarize`进程进行工作。以下是如何启动单个`Summarize`进程的方法：

```py
result= Summarize( results_q )
result.start()
```

以下是如何创建四个并发模拟进程的方法：

```py
    simulators= []
    for i in range(4):
        sim= Simulation( setup_q, results_q )
        sim.start()
        simulators.append( sim )
```

四个并发模拟器将竞争工作。每个模拟器都将尝试从待处理请求的队列中获取下一个请求。一旦所有四个模拟器都忙于工作，队列将开始填充未处理的请求。一旦队列和进程都在等待，驱动函数就可以开始将请求放入`setup_q`队列。以下是一个将生成大量请求的循环：

```py
table= Table( decks= 6, limit= 50, dealer=Hit17(),
    split= ReSplit(), payout=(3,2) )
for bet in Flat, Martingale, OneThreeTwoSix:
    player= Player( SomeStrategy, bet(), 100, 25 )
    for sample in range(5):
        setup_q.put( (table, player) )
```

我们创建了一个`Table`对象。对于三种投注策略，我们创建了一个`Player`对象，然后排队一个模拟请求。`Simulation`对象将从队列中获取 pickled 的两元组，然后对其进行处理。为了有序终止，我们需要为每个模拟器排队标记对象：

```py
    for sim in simulators:
        setup_q.put( (None,None) )

    for sim in simulators:
        sim.join()
```

对于每个模拟器，我们将一个标记对象放入队列中以供消耗。一旦所有模拟器都消耗了标记对象，我们就可以等待进程完成执行并重新加入到父进程中。

一旦`Process.join()`操作完成，将不会再创建模拟数据。我们也可以将一个标记对象放入模拟结果队列中：

```py
results_q.put( (None,None,None) )
result.join()
```

一旦结果标记对象被处理，`Summarize`进程将停止接受输入，我们也可以`join()`它。

我们使用多进程将对象从一个进程传输到另一个进程。这为我们提供了一个相对简单的方法来创建高性能的多处理数据流水线。`multiprocessing`模块使用`pickle`，因此对可以通过流水线推送的对象的性质几乎没有限制。

## 总结

我们研究了使用 RESTful web 服务和`wsgiref`模块以及`multiprocessing`模块来传输和共享对象，这两种架构都提供了通信对象状态表示的方式。在`multiprocessing`的情况下，使用 pickle 来表示状态。在构建 RESTful web 服务的情况下，我们必须选择要使用的表示形式。在这里使用的示例中，我们专注于 JSON，因为它被广泛使用并且具有简单的实现。许多框架也会提供 XML 的简单实现。

使用 WSGI 应用程序框架执行 RESTful web 服务规范化了接收 HTTP 请求、反序列化任何对象、执行请求的处理、序列化任何结果和提供响应的过程。由于 WSGI 应用程序具有简单、标准化的 API，我们可以轻松地创建复合应用程序和编写包装应用程序。我们通常可以利用包装应用程序以简单、一致的方式处理安全性的身份验证元素。

我们还研究了使用`multiprocessing`来对共享队列中的消息进行入队和出队操作。使用消息队列的美妙之处在于我们可以避免与共享对象的并发更新相关的锁定问题。

### 设计考虑和权衡

我们还必须决定要提供什么级别的对象以及如何使用明智的 URI 标识这些对象。对于较大的对象，我们可以轻松实现 ACID 属性。然而，我们可能也会上传和下载过多的数据以满足我们应用程序的用例。在某些情况下，我们需要提供替代级别的访问：大对象以支持 ACID 属性，小对象以在客户端应用程序需要数据子集时快速响应。

为了实现更加本地化的处理，我们可以利用`multiprocessing`模块。这更侧重于在受信任的主机或主机网络中构建高性能处理管道。

在某些情况下，这两种设计模式结合在一起，以便一个 RESTful 请求由多进程管道处理。传统的 Web 服务器（如 Apache HTTPD）通过`mod_wsgi`扩展可以使用多进程技术，通过命名管道将请求从 Apache 前端传递到 WSGI 应用程序后端。

### 模式演变

在处理面向公众的 RESTful 服务的 API 时，我们必须解决模式演变问题。如果我们更改类定义，我们将如何更改响应消息？如果外部 RESTful API 必须更改以与其他程序兼容，我们如何升级 Python Web 服务以支持不断变化的 API？

通常，我们必须在我们的 API 中提供一个主要的发布版本号。这可能是作为路径的一部分明确提供，或者隐含地通过包括在`POST`，`PUT`和`DELETE`请求中的数据字段提供。

我们需要区分不会改变 URI 路径或响应的更改和将改变 URI 或响应的更改。对功能的较小更改不会改变 URI 或响应的结构。

对 URI 或响应结构的更改可能会破坏现有的应用程序。这些是重大变化。使应用程序通过模式升级优雅地工作的一种方法是在 URI 路径中包含版本号。例如，`/roulette_2/wheel/`明确指定了轮盘服务器的第二个版本。

### 应用软件层

由于使用`sqlite3`时相对复杂，我们的应用软件必须更加合理地分层。对于 REST 客户端，我们可能会考虑具有层的软件架构。

当我们构建一个 RESTful 服务器时，表示层变得大大简化。它被简化为基本的请求-响应处理。它解析 URI 并以 JSON 或 XML（或其他表示形式）的文档进行响应。这一层应该被简化为对较低级别功能的薄 RESTful 外观。

在一些复杂情况下，人类用户所看到的最前端应用涉及来自几个不同来源的数据。整合来自不同来源的数据的一种简单方法是将每个来源包装在 RESTful API 中。这为我们提供了对数据不同来源的统一接口。它允许我们编写应用程序以统一的方式收集这些不同类型的数据。

### 展望未来

在下一章中，我们将使用持久化技术来处理配置文件。可由人类编辑的文件是配置数据的主要要求。如果我们使用一个知名的持久化模块，那么我们的应用程序可以在较少的编程下解析和验证配置数据。

# 第十三章：配置文件和持久性

配置文件是一种对象持久化的形式。它包含了应用程序或服务器的一些默认状态的序列化、可编辑表示。我们将扩展我们在第九章中展示的对象表示的序列化技术，*序列化和保存 - JSON、YAML、Pickle、CSV 和 XML* 来创建配置文件。

除了拥有一个纯文本可编辑的配置文件，我们还必须设计我们的应用程序是可配置的。此外，我们必须定义一种应用程序可以使用的配置对象（或集合）。在许多情况下，我们将有一系列包括系统范围默认值和用户特定覆盖的默认值。我们将研究配置数据的六种表示：

+   INI 文件使用的格式是 Windows 的一部分。它之所以受欢迎，部分原因是它是一种现有的格式，许多其他配置文件可能使用这种表示法。

+   PY 文件是普通的 Python 代码。这有很多优势，因为人们熟悉并且简单地使用它。

+   JSON 或 YAML 都设计成人性化和易于编辑。

+   属性文件经常在 Java 环境中使用。它们相对容易使用，也设计成人性化。

+   XML 文件很受欢迎，但有时很啰嗦，有时很难正确编辑。Mac OS 使用一种基于 XML 的格式，称为属性列表或`.plist`文件。

每种形式都为我们提供了一些优势和一些劣势。没有一种技术是最好的。在许多情况下，选择是基于与其他软件的兼容性或用户社区中对其他格式的熟悉程度。

## 配置文件的用例

有两种配置文件的用例。有时，我们可以稍微扩展定义，添加第三种用例。前两种应该是相当清楚的：

+   一个人需要编辑一个配置文件

+   软件将读取配置文件并利用选项和参数来调整其行为

配置文件很少是应用程序的*主要*输入。一个大的例外是模拟，其中配置可能是主要输入。在大多数其他情况下，配置不是主要输入。例如，Web 服务器的配置文件可能调整服务器的行为，但 Web 请求是一个主要输入，数据库或文件系统是另一个主要输入。在 GUI 应用程序的情况下，用户的交互事件是一个输入，文件或数据库可能是另一个输入；配置文件可以微调应用程序。

在主要输入和配置输入之间存在模糊的边界。理想情况下，一个应用程序的行为应该与配置细节无关。然而，从实用的角度来看，配置可能会引入额外的策略或状态到现有的应用程序中，从而改变其行为。在这种情况下，配置可以跨越界限，成为代码的一部分，而不仅仅是固定代码库的配置。

可能的第三种用例是在应用程序更新后将配置保存回文件。这种使用持久状态对象的方式是不典型的，因为配置文件已经变成了主要输入，程序正在保存其操作状态。这种用例可能表明两件事已经融合成一个文件：配置参数和持久操作状态。最好将其设计为使用人类可读格式的持久状态。

配置文件可以为应用程序提供多种参数和参数值。我们需要更深入地研究一些这些不同类型的数据，以决定如何最好地表示它们。

+   默认值

+   设备名称，可能与文件系统的位置重叠

+   文件系统位置和搜索路径

+   限制和边界

+   消息模板和数据格式规范

+   消息文本，可能已经翻译成国际化

+   网络名称、地址和端口号

+   可选行为

+   安全密钥、令牌、用户名、密码

+   值域：

这些值是相对常见类型的值：字符串、整数和浮点数。所有这些值都有一个整洁的文本表示，对于人来说相对容易编辑。它们对我们的 Python 应用程序来说也很容易解析人类输入。

在某些情况下，我们可能会有值的列表。例如，值域或路径可能是更简单类型的集合。通常，这是一个简单的序列或元组序列。类似字典的映射通常用于消息文本，以便将应用程序的软件密钥映射到定制的自然语言措辞。

还有一个不是简单类型的额外配置值，它没有整洁的文本表示。我们可以将这个项目添加到前面的列表中：

+   代码的附加功能、插件和扩展：

这是具有挑战性的，因为我们不一定向应用程序提供一个简单的字符串值。配置提供了一个应用程序将使用的对象。当插件有更多的 Python 代码时，我们可以提供已安装的 Python 模块的路径，就像在`import`语句中使用这个点名一样：'`package.module.object`'。然后应用程序可以执行预期的'`from package.module import object`'代码并使用给定的类或函数。

对于非 Python 代码，我们有另外两种技术来导入代码，以便可以使用它：

+   对于不是适当的可执行程序的二进制文件，我们可以尝试使用`ctypes`模块调用定义的 API 方法

+   对于可执行程序的二进制文件，`subprocess`模块为我们提供了执行它们的方法

这两种技术都不是特定于 Python 的，并且推动了本章的边界。我们将专注于获取参数或参数值的核心问题。这些值的使用是一个非常大的话题。

## 表示、持久性、状态和可用性

查看配置文件时，我们正在查看一个或多个对象状态的人性化版本。当我们编辑配置文件时，我们正在更改对象的持久状态，当应用程序启动（或重新启动）时将重新加载。我们有两种常见的查看配置文件的方式：

+   从参数名称到值的映射或一组映射

+   一个序列化的对象，不仅仅是一个简单的映射

当我们试图将配置文件简化为映射时，我们可能会限制配置中可能存在的关系范围。在简单映射中，一切都必须通过名称引用，并且我们必须解决与第十章中讨论的`shelve`和第十一章中讨论的`sqlite`的键设计问题相同的键设计问题。我们在配置的一部分提供一个唯一的名称，以便其他部分可以正确引用它。

查看`logging`配置的示例有助于理解如何配置复杂系统可能非常具有挑战性。Python 日志对象之间的关系——记录器、格式化程序、过滤器和处理程序——必须全部绑定在一起才能创建可用的记录器。*标准库参考*的第 16.8 节向我们展示了日志配置文件的两种不同语法。我们将在第十四章中查看日志，*日志和警告模块*。

在某些情况下，将复杂的 Python 对象序列化或者使用 Python 代码直接作为配置文件可能更简单。如果配置文件增加了太多的复杂性，那么它可能并没有真正的价值。

### 应用程序配置设计模式

应用程序配置有两种核心设计模式：

+   **全局属性映射**：一个全局对象将包含所有的配置参数。这可以是一个 `name:value` 对的映射，也可以是一个属性值的大型命名空间对象。这可能遵循**单例**设计模式，以确保只有一个实例存在。

+   **对象构造**：我们将定义一种**工厂**或**工厂**集合，使用配置数据来构建应用程序的对象。在这种情况下，配置信息在程序启动时使用一次，以后再也不使用。配置信息不会作为全局对象保留。

全局属性映射设计非常受欢迎，因为它简单且可扩展。我们可能会有一个如下代码简单的对象：

```py
class Configuration:
    some_attribute= "default_value"
```

我们可以使用前面的类定义作为属性的全局容器。在初始化过程中，我们可能会在解析配置文件的一部分中有类似以下的内容：

```py
Configuration.some_attribute= "user-supplied value"
```

在程序的其他地方，我们可以使用 `Configuration.some_attribute` 的值。这个主题的一个变体是制作一个更正式的**单例**对象设计模式。这通常是通过全局模块来完成的，因为这样可以很容易地导入，从而为我们提供一个可访问的全局定义。

我们可能有一个名为 `configuration.py` 的模块。在那个文件中，我们可以有以下定义：

```py
settings= dict()
```

现在，应用程序可以使用 `configuration.settings` 作为应用程序所有设置的全局存储库。一个函数或类可以解析配置文件，加载这个字典与应用程序将使用的配置值。

在一个二十一点模拟中，我们可能会看到类似以下的代码：

```py
shoe= Deck( configuration.settings['decks'] )
```

或者，我们可能会看到类似以下的代码：

```py
If bet > configuration.settings['limit']: raise InvalidBet()
```

通常，我们会尽量避免使用全局变量。因为全局变量隐式地存在于任何地方，所以它可能会被忽视。我们可以通过对象构造来更整洁地处理配置，而不是使用全局变量。

### 通过对象构造进行配置

在通过对象构造配置应用程序时，目标是构建所需的对象。实际上，配置文件定义了将要构建的对象的各种初始化参数。

我们经常可以将这种初始对象构造的大部分集中在一个单一的 `main()` 函数中。这将创建应用程序的真正工作的对象。我们将在第十六章 *处理命令行*中重新讨论并扩展这些设计问题。

考虑一下二十一点玩法和投注策略的模拟。当我们运行模拟时，我们想要收集特定组合的独立变量的性能。这些变量可能包括一些赌场政策，包括牌组数量、桌面限制和庄家规则。这些变量可能包括玩家的游戏策略，例如何时要牌、停牌、分牌和加倍。它还将包括玩家的投注策略，如平注、马丁尼投注或更复杂的拜占庭投注系统。我们的基线代码开始如下所示：

```py
import csv
def simulate_blackjack():
    dealer_rule= Hit17()
    split_rule= NoReSplitAces()
    table= Table( decks=6, limit=50, dealer=dealer_rule,
        split=split_rule, payout=(3,2) )
    player_rule= SomeStrategy()
    betting_rule= Flat()
    player= Player( play=player_rule, betting=betting_rule, rounds=100, stake=50 )

    simulator= Simulate( table, player, 100 )
    with open("p2_c13_simulation.dat","w",newline="") as results:
        wtr= csv.writer( results )
        for gamestats in simulator:
            wtr.writerow( gamestats )
```

这是一种技术飞跃，它已经硬编码了所有的对象类和初始值。我们需要添加配置参数来确定对象的类和它们的初始值。

`Simulate` 类有一个 API，看起来像以下代码：

```py
class Simulate:
    def __init__( self, table, player, samples ):
        """Define table, player and number of samples."""
        self.table= table
        self.player= player
        self.samples= samples
    def __iter__( self ):
        """Yield statistical samples."""
```

这使我们能够使用一些适当的初始化参数构建`Simulate()`对象。一旦我们建立了`Simulate()`的实例，我们可以通过该对象进行迭代，以获得一系列统计摘要对象。

有趣的部分是使用配置参数而不是类名。例如，某些参数应该用于决定`dealer_rule`值是创建`Hit17`还是`Stand17`实例。同样，`split_rule`值应该是在几个类中选择，这些类体现了赌场中使用的几种不同的分牌规则。

在其他情况下，应该使用参数来为类的`__init__()`方法提供参数。例如，牌组数量、庄家下注限制和二十一点赔付值是用于创建`Table`实例的配置值。

一旦对象建立，它们通过`Simulate.run()`方法正常交互以产生统计输出。不再需要全局参数池：参数值通过它们的实例变量绑定到对象中。

对象构造设计并不像全局属性映射那样简单。它避免了全局变量的优势，也使参数处理在一些主要工厂函数中变得集中和明显。

在使用对象构造时添加新参数可能会导致重构应用程序以公开参数或关系。这可能会使其看起来比从名称到值的全局映射更复杂。

这种技术的一个重要优势是消除了应用程序深处的复杂`if`语句。使用`Strategy`设计模式倾向于将决策推进到对象构造中。除了简化处理外，消除`if`语句还可以提高性能。

### 实施配置层次结构

我们通常有几种选择来放置配置文件。有五种常见选择，我们可以使用所有五种来创建参数的一种继承层次结构：

+   **应用程序的安装目录**：实际上，这类似于基类定义。这里有两个子选择。较小的应用程序可以安装在 Python 的库结构中；初始化文件也可以安装在那里。较大的应用程序通常会有自己的用户名，拥有一个或多个安装目录树。

+   **Python 安装目录**：我们可以使用模块的`__file__`属性找到模块的安装位置。从这里，我们可以使用`os.path.split()`来定位配置文件：

```py
	>>> import this
	>>> this.__file__
	'/Library/Frameworks/Python.framework/Versions/3.3/lib/python3.3/this.py'
```

+   **应用程序安装目录**：这将基于拥有的用户名，因此我们可以使用`~theapp/`和`os.path.expanduser()`来跟踪配置默认值。

+   **系统范围的配置目录**：这通常存在于`/etc`中。在 Windows 上，这可以转换为`C:\etc`。其他选择包括`os.environ['WINDIR']`或`os.environ['ALLUSERSPROFILE']`的值。

+   **当前用户的主目录**：通常可以使用`os.path.expanduser()`将`~/`转换为用户的主目录。对于 Windows，Python 将正确使用`%HOMEDRIVE%`和`%HOMEPATH%`环境变量。

+   **当前工作目录**：该目录通常称为`./`，尽管`os.path.curdir`更具可移植性。

+   **在命令行参数中命名的文件**：这是一个明确命名的文件，不应进一步处理名称。

应用程序可以从基类（首先列出）到命令行选项中集成所有这些来源的配置选项。通过这种方式，安装默认值是最通用且最不特定于用户的；这些值可以被更具体和不那么通用的值覆盖。

这意味着我们经常会有一系列文件，如以下代码所示：

```py
import os
config_name= "someapp.config"
config_locations = (
  os.path.expanduser("~thisapp/"), # or thisapp.__file__,
  "/etc",
  os.path.expanduser("~/"),
  os.path.curdir,
)
candidates = ( os.path.join(dir,config_name)
    for dir in config_locations )
config_names = [ name for name in candidates if os.path.exists(name) ]
```

我们取了一个备用文件目录的元组，并通过将目录与配置文件名连接起来创建了一个候选文件名列表。

一旦我们有了这个配置文件名列表，我们可以使用以下代码将通过命令行参数提供的任何文件名附加到列表的末尾：

```py
config_names.append(command_line_option)
```

这给了我们一个可以检查以定位配置文件或配置默认值的位置列表。

## 将配置存储在 INI 文件中

INI 文件格式起源于早期的 Windows 操作系统。解析这些文件的模块是`configparser`。

有关 INI 文件的更多细节，请参阅维基百科文章：[`en.wikipedia.org/wiki/INI_file`](http://en.wikipedia.org/wiki/INI_file)。

INI 文件有各个部分和每个部分内的属性。我们的示例主程序有三个部分：表配置，玩家配置和整体模拟数据收集。

我们可以想象一个看起来像以下代码的 INI 文件：

```py
; Default casino rules
[table]
    dealer= Hit17
    split= NoResplitAces
    decks= 6
    limit= 50
    payout= (3,2)

; Player with SomeStrategy
; Need to compare with OtherStrategy
[player]
    play= SomeStrategy
    betting= Flat
    rounds= 100
    stake= 50

[simulator]
    samples= 100
    outputfile= p2_c13_simulation.dat
```

我们将参数分为三个部分。在每个部分中，我们提供了一些命名参数，这些参数对应于我们前面模型应用初始化中显示的类名和初始化值。

一个文件可以非常简单地解析：

```py
import configparser
config = configparser.ConfigParser()
config.read('blackjack.ini')
```

我们创建了一个解析器的实例，并将目标配置文件名提供给该解析器。解析器将读取文件，定位各个部分，并定位每个部分内的各个属性。

如果我们想要支持文件的多个位置，我们可以使用`config.read(config_names)`。当我们将文件名列表提供给`ConfigParser.read()`时，它将按顺序读取文件。我们希望从最通用的文件到最具体的文件提供文件。软件安装中的通用配置文件将首先被解析以提供默认值。用户特定的配置将稍后被解析以覆盖这些默认值。

一旦我们解析了文件，我们需要利用各种参数和设置。这是一个根据解析配置文件创建的配置对象构建我们对象的函数。我们将其分为三部分。这是构建`Table`实例的部分：

```py
def main_ini( config ):
    dealer_nm= config.get('table','dealer', fallback='Hit17')
    dealer_rule= {'Hit17': Hit17(),
        'Stand17': Stand17()}.get(dealer_nm, Hit17())
    split_nm= config.get('table','split', fallback='ReSplit')
    split_rule= {'ReSplit': ReSplit(),
        'NoReSplit': NoReSplit(),
        'NoReSplitAces': NoReSplitAces()}.get(split_nm, ReSplit())
    decks= config.getint('table','decks', fallback=6)
    limit= config.getint('table','limit', fallback=100)
    payout= eval( config.get('table','payout', fallback='(3,2)') )
    table= Table( decks=decks, limit=limit, dealer=dealer_rule,
        split=split_rule, payout=payout )
```

我们使用了 INI 文件中`[table]`部分的属性来选择类名并提供初始化值。这里有三种广泛的情况：

+   **将字符串映射到类名**：我们使用映射来根据字符串类名查找对象。这是为了创建`dealer_rule`和`split_rule`。如果这是一个需要大量更改的地方，我们可能能够将这个映射提取到一个单独的工厂函数中。

+   **获取 ConfigParser 可以为我们解析的值**：该类可以直接处理`str`、`int`、`float`和`bool`。该类具有从字符串到布尔值的复杂映射，使用各种常见代码和`True`和`False`的同义词。

+   **评估非内置内容**：在`payout`的情况下，我们有一个字符串值，`'(3,2)'`，这不是`ConfigParser`的直接支持的数据类型。我们有两种选择来处理这个问题。我们可以尝试自己解析它，或者坚持该值是有效的 Python 表达式，并让 Python 来处理。在这种情况下，我们使用了`eval()`。一些程序员称这是一个*安全问题*。下一节将处理这个问题。

这是这个示例的第二部分，它使用了 INI 文件中`[player]`部分的属性来选择类和参数值：

```py
    player_nm= config.get('player','play', fallback='SomeStrategy')
    player_rule= {'SomeStrategy': SomeStrategy(),
        'AnotherStrategy': AnotherStrategy()}.get(player_nm,SomeStrategy())
    bet_nm= config.get('player','betting', fallback='Flat')
    betting_rule= {'Flat': Flat(),
        'Martingale': Martingale(),
        'OneThreeTwoSix': OneThreeTwoSix()}.get(bet_nm,Flat())
    rounds= config.getint('player','rounds', fallback=100)
    stake= config.getint('player','stake', fallback=50)
    player= Player( play=player_rule, betting=betting_rule,
        rounds=rounds, stake=stake )
```

这使用了字符串到类的映射以及内置数据类型。它初始化了两个策略对象，然后从这两个策略加上两个整数配置值创建了`Player`。

这是最后一部分；这创建了整体模拟器：

```py
    outputfile= config.get('simulator', 'outputfile', fallback='blackjack.csv')
    samples= config.getint('simulator', 'samples', fallback=100)
    simulator= Simulate( table, player, samples )
    with open(outputfile,"w",newline="") as results:
        wtr= csv.writer( results )
        for gamestats in simulator:
            wtr.writerow( gamestats )
```

我们从`[simulator]`部分使用了两个参数，这些参数超出了对象创建的狭窄范围。`outputfile`属性用于命名文件；`samples`属性作为方法函数的参数提供。

## 通过 eval()变体处理更多文字

配置文件可能具有没有简单字符串表示的类型的值。例如，一个集合可以作为`tuple`或`list`文字提供；一个映射可以作为`dict`文字提供。我们有几种选择来处理这些更复杂的值。

选择解决了一个问题，即转换能够容忍多少 Python 语法。对于一些类型（`int`、`float`、`bool`、`complex`、`decimal.Decimal`、`fractions.Fraction`），我们可以安全地将字符串转换为文字值，因为这些类型的对象`__init__()`处理字符串值而不容忍任何额外的 Python 语法。

然而，对于其他类型，我们不能简单地进行字符串转换。我们有几种选择来继续进行：

+   禁止这些数据类型，并依赖于配置文件语法加上处理规则，从非常简单的部分组装复杂的 Python 值。这很繁琐，但可以做到。

+   使用`ast.literal_eval()`，因为它处理许多 Python 文字值的情况。这通常是理想的解决方案。

+   使用`eval()`来简单评估字符串并创建预期的 Python 对象。这将解析比`ast.literal_eval()`更多种类的对象。这种广泛性真的有必要吗？

使用`ast`模块来编译和审查结果代码对象。这个审查过程可以检查`import`语句以及使用一些允许的模块。这非常复杂；如果我们有效地允许代码，也许我们应该设计一个框架，而不是一个带有配置文件的应用程序。

在我们通过网络执行 RESTful 传输 Python 对象的情况下，绝对不能信任对结果文本的`eval()`。参见第九章 - *序列化和保存 - JSON、YAML、Pickle、CSV 和 XML*。

然而，在读取本地配置文件的情况下，`eval()`可能是可用的。在某些情况下，Python 代码和配置文件一样容易修改。当基本代码可以被调整时，担心`eval()`可能并不有用。

以下是我们如何使用`ast.literal_eval()`而不是`eval()`：

```py
>>> import ast
>>> ast.literal_eval('(3,2)')
(3, 2)
```

这扩大了配置文件中可能值的领域。它不允许任意对象，但允许广泛的文字值。

## 将配置存储在 PY 文件中

PY 文件格式意味着使用 Python 代码作为配置文件以及实现应用程序的语言。我们将有一个配置文件，它只是一个模块；配置是用 Python 语法编写的。这消除了解析模块的需要。

使用 Python 给我们带来了许多设计考虑。我们有两种策略来使用 Python 作为配置文件：

+   **顶层脚本**：在这种情况下，配置文件只是最顶层的主程序

+   **exec()导入**：在这种情况下，我们的配置文件提供参数值，这些值被收集到模块全局变量中

我们可以设计一个顶层脚本文件，看起来像以下代码：

```py
from simulator import *
def simulate_SomeStrategy_Flat():
    dealer_rule= Hit17()
    split_rule= NoReSplitAces()
    table= Table( decks=6, limit=50, dealer=dealer_rule,
        split=split_rule, payout=(3,2) )
    player_rule= SomeStrategy()
    betting_rule= Flat()
    player= Player( play=player_rule, betting=betting_rule, rounds=100, stake=50 )
    simulate( table, player, "p2_c13_simulation3.dat", 100 )

if __name__ == "__main__":
    simulate_SomeStrategy_Flat()
```

这显示了我们用来创建和初始化对象的各种配置参数。我们只是直接将配置参数写入代码中。我们将处理过程分解到一个单独的函数`simulate()`中。

使用 Python 作为配置语言的一个潜在缺点是 Python 语法的复杂性。出于两个原因，这通常是一个无关紧要的问题。首先，通过一些精心设计，配置的语法应该是简单的赋值语句，带有一些`()`和`,`。其次，更重要的是，其他配置文件有其自己的复杂语法，与 Python 语法不同。使用单一语言和单一语法是减少复杂性的一种方式。

`simulate()`函数是从整个`simulator`应用程序中导入的。这个`simulate()`函数可能看起来像以下代码：

```py
import csv
def simulate( table, player, outputfile, samples ):
    simulator= Simulate( table, player, samples )
    with open(outputfile,"w",newline="") as results:
        wtr= csv.writer( results )
        for gamestats in simulator:
            wtr.writerow( gamestats )
```

这个函数是关于表、玩家、文件名和样本数量的通用函数。

这种配置技术的困难在于缺乏方便的默认值。顶层脚本必须完整：*所有*配置参数必须存在。提供所有值可能会很烦人；为什么要提供很少更改的默认值呢？

在某些情况下，这并不是一个限制。在需要默认值的情况下，我们将看看如何解决这个限制。

### 通过类定义进行配置

有时我们在顶层脚本配置中遇到的困难是缺乏方便的默认值。为了提供默认值，我们可以使用普通的类继承。以下是我们如何使用类定义来构建一个具有配置值的对象：

```py
import simulation
class Example4( simulation.Default_App ):
    dealer_rule= Hit17()
    split_rule= NoReSplitAces()
    table= Table( decks=6, limit=50, dealer=dealer_rule,
        split=split_rule, payout=(3,2) )
    player_rule= SomeStrategy()
    betting_rule= Flat()
    player= Player( play=player_rule, betting=betting_rule, rounds=100, stake=50 )
    outputfile= "p2_c13_simulation4.dat"
    samples= 100
```

这允许我们使用默认配置定义`Default_App`。我们在这里定义的类可以简化为仅提供来自`Default_App`版本的覆盖值。

我们还可以使用 mixin 来将定义分解为可重用的部分。我们可以将我们的类分解为表、玩家和模拟组件，并通过 mixin 组合它们。有关 mixin 类设计的更多信息，请参见第八章，*装饰器和 Mixin-横切面*。

在两个小方面，这种类定义的使用推动了边界。没有方法定义；我们只会使用这个类来定义一个实例。然而，这是一种非常整洁的方式，可以将一小块代码打包起来，以便赋值语句填充一个小的命名空间。

我们可以修改我们的`simulate()`函数来接受这个类定义作为参数：

```py
def simulate_c( config ):
    simulator= Simulate( config.table, config.player, config.samples )
    with open(config.outputfile,"w",newline="") as results:
        wtr= csv.writer( results )
        for gamestats in simulator:
            wtr.writerow( gamestats )
```

这个函数从整体配置对象中挑选出相关的值，并用它们构建一个`Simulate`实例并执行该实例。结果与之前的`simulate()`函数相同，但参数结构不同。以下是我们如何将这个类的单个实例提供给这个函数：

```py
if __name__ == "__main__":
    simulation.simulate_c(Example4())
```

这种方法的一个小缺点是它与`argparse`不兼容，无法收集命令行参数。我们可以通过使用`types.SimpleNamespace`对象来解决这个问题。

### 通过 SimpleNamespace 进行配置

使用`types.SimpleNamespace`对象允许我们根据需要简单地添加属性。这类似于使用类定义。在定义类时，所有赋值语句都局限于类。在创建`SimpleNamespace`对象时，我们需要明确地使用`NameSpace`对象来限定每个名称，我们正在填充的`NameSpace`对象。理想情况下，我们可以创建类似以下代码的`SimpleNamespace`：

```py
>>> import types
>>> config= types.SimpleNamespace( 
...     param1= "some value",
...     param2= 3.14,
... )
>>> config
namespace(param1='some value', param2=3.14)
```

如果所有配置值彼此独立，则这种方法非常有效。然而，在我们的情况下，配置值之间存在一些复杂的依赖关系。我们可以通过以下两种方式之一来处理这个问题：

+   我们可以只提供独立的值，让应用程序构建依赖的值

+   我们可以逐步构建命名空间中的值

只创建独立值，我们可以做如下操作：

```py
import types
config5a= types.SimpleNamespace(
  dealer_rule= Hit17(),
  split_rule= NoReSplitAces(),
  player_rule= SomeStrategy(),
  betting_rule= Flat(),
  outputfile= "p2_c13_simulation5a.dat",
  samples= 100,
  )

config5a.table= Table( decks=6, limit=50, dealer=config5a.dealer_rule,
        split=config5a.split_rule, payout=(3,2) )
config5a.player= Player( play=config5a.player_rule, betting=config5a.betting_rule,
        rounds=100, stake=50 )
```

在这里，我们使用六个独立值创建了`SimpleNamespace`的配置。然后，我们更新配置以添加另外两个值，这些值依赖于四个独立值。

`config5a`对象几乎与前面示例中通过评估`Example4()`创建的对象相同。基类不同，但属性及其值的集合是相同的。以下是另一种方法，在顶层脚本中逐步构建配置：

```py
import types
config5= types.SimpleNamespace()
config5.dealer_rule= Hit17()
config5.split_rule= NoReSplitAces()
config5.table= Table( decks=6, limit=50, dealer=config5.dealer_rule,
        split=config5.split_rule, payout=(3,2) )
config5.player_rule= SomeStrategy()
config5.betting_rule= Flat()
config5.player= Player( play=config5.player_rule, betting=config5.betting_rule,
        rounds=100, stake=50 )
config5.outputfile= "p2_c13_simulation5.dat"
config5.samples= 100
```

与之前显示的`simulate_c()`函数相同，可以用于这种类型的配置。

遗憾的是，这与通过顶层脚本进行配置的问题相同。没有方便的方法为配置对象提供默认值。我们可能希望有一个可以导入的工厂函数，它使用适当的默认值创建`SimpleNamespace`：

```py
From simulation import  make_config
config5= make_config()
```

如果我们使用类似上面的代码，那么默认值可以由工厂函数`make_config()`分配。然后每个用户提供的配置只需提供对默认值的必要覆盖。

我们的默认提供`make_config()`函数将具有以下类型的代码：

```py
def make_config( ):
    config= types.SimpleNamespace()
    # set the default values
    config.some_option = default_value
    return config
```

`make_config()`函数将通过一系列赋值语句构建默认配置。然后应用程序只能设置有趣的*覆盖*值：

```py
config= make_config()
config.some_option = another_value
simulate_c( config )
```

这使应用程序能够构建配置，然后以相对简单的方式使用它。主脚本非常简短且简洁。如果使用关键字参数，我们可以很容易地使其更加灵活：

```py
 def make_config( **kw ):
    config= types.SimpleNamespace()
    # set the default values
    config.some_option = kw.get("some_option", default_value)
    return config
```

这使我们能够创建包括覆盖的配置，如下所示：

```py
config= make_config( some_option= another_value )
simulate_c( config )
```

这略短一些，似乎保留了前面示例的清晰度。

所有来自第一章方法")的技术，*__init__()方法*，都适用于定义这种类型的配置工厂函数。如果需要，我们可以构建出很大的灵活性。这有一个优点，它很好地符合`argparse`模块解析命令行参数的方式。我们将在第十六章中扩展这一点，*处理命令行*

### 使用 Python 和 exec()进行配置

当我们决定使用 Python 作为配置的表示时，我们可以使用`exec()`函数在受限制的命名空间中评估一块代码。我们可以想象编写看起来像以下代码的配置文件：

```py
# SomeStrategy setup

# Table
dealer_rule= Hit17()
split_rule= NoReSplitAces()
table= Table( decks=6, limit=50, dealer=dealer_rule,
        split=split_rule, payout=(3,2) )

# Player
player_rule= SomeStrategy()
betting_rule= Flat()
player= Player( play=player_rule, betting=betting_rule,
        rounds=100, stake=50 )

# Simulation
outputfile= "p2_c13_simulation6.dat"
samples= 100
```

这是一组愉快、易于阅读的配置参数。它类似于 INI 文件和属性文件，我们将在下一节中进行讨论。我们可以评估此文件，使用`exec()`函数创建一种命名空间：

```py
with open("config.py") as py_file:
    code= compile(py_file.read(), 'config.py', 'exec')
config= {}
exec( code, globals(), config  )
simulate( config['table'], config['player'],
    config['outputfile'], config['samples'])
```

在这个例子中，我们决定使用`compile()`函数显式构建代码对象。这不是必需的；我们可以简单地将文件的文本提供给`exec()`函数，它将编译代码。

对`exec()`的调用提供了三个参数：代码、应该用于解析任何全局名称的字典，以及将用于创建任何局部变量的字典。当代码块完成时，赋值语句将用于在局部字典中构建值；在这种情况下，是`config`变量。键将是变量名。

然后我们可以使用这个在程序初始化期间构建对象。我们将必要的对象传递给`simulate()`函数来执行模拟。`config`变量将获得所有局部赋值，并将具有类似以下代码的值：

```py
{'betting_rule': <__main__.Flat object at 0x101828510>,
 'dealer_rule': <__main__.Hit17 object at 0x101828410>,
 'outputfile': 'p2_c13_simulation6.dat',
 'player': <__main__.Player object at 0x101828550>,
 'player_rule': <__main__.SomeStrategy object at 0x1018284d0>,
 'samples': 100,
 'split_rule': <__main__.NoReSplitAces object at 0x101828450>,
 'table': <__main__.Table object at 0x101828490>}
```

但是，初始化必须是一个书面的字典表示法：`config['table']`，`config['player']`。

由于字典表示法不方便，我们将使用基于第三章，“属性访问、属性和描述符”中的想法的设计模式。这是一个根据字典键提供命名属性的类：

```py
class AttrDict( dict ):
    def __getattr__( self, name ):
        return self.get(name,None)
    def __setattr__( self, name, value ):
        self[name]= value
    def __dir__( self ):
        return list(self.keys())
```

这个类只有在键是合适的 Python 变量名时才能工作。有趣的是，这是`exec()`函数初始化`config`变量的方式：

```py
config= AttrDict()
```

然后，我们可以使用更简单的属性表示法，`config.table`，`config.player`，来进行初始对象构建和初始化。在复杂的应用程序中，这种少量的语法糖可能会有所帮助。另一种方法是定义这个类：

```py
class Configuration:
    def __init__( self, **kw ):
        self.__dict__.update(kw)
```

然后我们可以将简单的`dict`转换为具有愉快的命名属性的对象：

```py
config= Configuration( **config )
```

这将把`dict`转换为一个具有易于使用的属性名称的对象。当然，这只适用于字典键已经是 Python 变量名的情况。它也仅限于结构是平面的情况。对于我们在其他格式中看到的嵌套字典结构，这种方法是行不通的。

## 为什么`exec()`不是问题？

前一节讨论了`eval()`。相同的考虑也适用于`exec()`。

通常，可用的`globals()`集合是受严格控制的。通过从提供给`exec()`的全局变量中删除它们来消除对`os`模块或`__import__()`函数的访问。

如果你有一个邪恶的程序员，他会巧妙地破坏配置文件，那么请记住，他们可以完全访问所有的 Python 源代码。当他们可以直接改变应用程序代码本身时，为什么要浪费时间巧妙地调整配置文件呢？

一个常见的问题是：“如果有人认为他们可以通过强制新代码进入配置文件来猴子补丁一个损坏的应用程序怎么办？”这个人很可能以同样聪明/疯狂的方式破坏应用程序。避免 Python 配置文件不会阻止不道德的程序员通过做一些不明智的事情来破坏事物。有无数潜在的弱点；不必要地担心`exec()`可能不会有益。

在某些情况下，可能需要改变整体理念。一个高度可定制的应用程序实际上可能是一个通用框架，而不是一个整洁的成品应用程序。

## 使用 ChainMap 进行默认值和覆盖

我们经常会有一个配置文件层次结构。之前，我们列出了可以安装配置文件的几个位置。例如，`configparser`模块旨在按顺序读取多个文件，并通过后续文件覆盖先前文件的值来集成设置。

我们可以使用`collections.ChainMap`类实现优雅的默认值处理。有关此类的一些背景，请参阅第六章，“创建容器和集合”。我们需要将配置参数保留为`dict`实例，这在使用`exec()`来评估 Python 语言初始化文件时非常有效。

使用这种方法需要我们将配置参数设计为一组平面值的字典。对于从多个来源集成的大量复杂配置值的应用程序来说，这可能有点麻烦。我们将向您展示一种合理的方式来展平名称。

首先，我们将根据标准位置构建一个文件列表：

```py
from collections import ChainMap
import os
config_name= "config.py"
config_locations = (
  os.path.expanduser("~thisapp/"), # or thisapp.__file__,
  "/etc",
  os.path.expanduser("~/"),
  os.path.curdir,
)
candidates = ( os.path.join(dir,config_name)
    for dir in config_locations )
config_names = ( name for name in candidates if os.path.exists(name) )
```

我们从一个目录列表开始：安装目录、系统全局目录、用户的主目录和当前工作目录。我们将配置文件名放入每个目录，然后确认文件实际存在。

一旦我们有了候选文件的名称，我们就可以通过折叠每个文件来构建`ChainMap`：

```py
config = ChainMap()
for name in config_names:
    config= config.new_child()
    exec(name, globals(), config)
simulate( config.table, config.player, config.outputfile, config.samples)
```

每个文件都涉及创建一个新的空映射，可以使用本地变量进行更新。`exec()`函数将文件的本地变量添加到`new_child()`创建的空映射中。每个新子代都更加本地化，覆盖先前加载的配置。

在`ChainMap`中，通过搜索映射序列来解析每个名称以查找值。当我们将两个配置文件加载到`ChainMap`中时，我们将得到以下结构的代码：

```py
ChainMap(
    {'player': <__main__.Player object at 0x10101a710>, 'outputfile': 'p2_c13_simulation7a.dat', 'player_rule': <__main__.AnotherStrategy object at 0x10101aa90>},
    {'dealer_rule': <__main__.Hit17 object at 0x10102a9d0>, 'betting_rule': <__main__.Flat object at 0x10101a090>, 'split_rule': <__main__.NoReSplitAces object at 0x10102a910>, 'samples': 100, 'player_rule': <__main__.SomeStrategy object at 0x10102a8d0>, 'table': <__main__.Table object at 0x10102a890>, 'outputfile': 'p2_c13_simulation7.dat', 'player': <__main__.Player object at 0x10101a210>},
    {})
```

我们有一系列映射；第一个映射是最后定义的最本地变量。这些是覆盖。第二个映射具有应用程序默认值。还有第三个空映射，因为`ChainMap`始终至少有一个映射；当我们构建`config`的初始值时，必须创建一个空映射。

唯一的缺点是初始化将使用字典表示法，`config['table']`，`config['player']`。我们可以扩展`ChainMap()`以实现属性访问以及字典项访问。

这是`ChainMap`的一个子类，如果我们发现`getitem()`字典表示法太麻烦，我们可以使用它：

```py
class AttrChainMap( ChainMap ):
    def __getattr__( self, name ):
        if name == "maps":
            return self.__dict__['maps']
        return super().get(name,None)
    def __setattr__( self, name, value ):
        if name == "maps":
            self.__dict__['maps']= value
            return
        self[name]= value
```

现在我们可以使用`config.table`而不是`config['table']`。这揭示了我们对`ChainMap`的扩展的一个重要限制：我们不能将`maps`用作属性。`maps`键是父`ChainMap`类的一级属性。

## 将配置存储在 JSON 或 YAML 文件中

我们可以相对轻松地将配置值存储在 JSON 或 YAML 文件中。语法设计得用户友好。我们可以在 YAML 中表示各种各样的事物。在 JSON 中，我们受到更窄的对象类别的限制。我们可以使用类似以下代码的 JSON 配置文件：

```py
{
    "table":{
        "dealer":"Hit17",
        "split":"NoResplitAces",
        "decks":6,
        "limit":50,
        "payout":[3,2]
    },
    "player":{
        "play":"SomeStrategy",
        "betting":"Flat",
        "rounds":100,
        "stake":50
    },
    "simulator":{
        "samples":100,
        "outputfile":"p2_c13_simulation.dat"
    }
}
```

JSON 文档看起来像字典的字典。这正是在加载此文件时将构建的对象。我们可以使用以下代码加载单个配置文件：

```py
import json
config= json.load( "config.json" )
```

这使我们可以使用`config['table']['dealer']`来查找用于荷官规则的特定类。我们可以使用`config['player']['betting']`来定位玩家特定的投注策略类名。

与 INI 文件不同，我们可以轻松地将`tuple`编码为值序列。因此，`config['table']['payout']`值将是一个正确的两元素序列。严格来说，它不会是`tuple`，但它足够接近，我们可以在不必使用`ast.literal_eval()`的情况下使用它。

以下是我们将如何使用此嵌套结构。我们只会向您展示`main_nested_dict()`函数的第一部分：

```py
def main_nested_dict( config ):
    dealer_nm= config.get('table',{}).get('dealer', 'Hit17')
    dealer_rule= {'Hit17':Hit17(),
        'Stand17':Stand17()}.get(dealer_nm, Hit17())
    split_nm= config.get('table',{}).get('split', 'ReSplit')
    split_rule= {'ReSplit':ReSplit(),
        'NoReSplit':NoReSplit(),
        'NoReSplitAces':NoReSplitAces()}.get(split_nm, ReSplit())
    decks= config.get('table',{}).get('decks', 6)
    limit= config.get('table',{}).get('limit', 100)
 **payout= config.get('table',{}).get('payout', (3,2))
    table= Table( decks=decks, limit=limit, dealer=dealer_rule,
        split=split_rule, payout=payout )
```

这与之前显示的`main_ini()`函数非常相似。当我们将其与之前的使用`configparser`的版本进行比较时，很明显复杂性几乎相同。命名略微简单。我们使用`config.get('table',{}).get('decks')`代替`config.getint('table','decks')`。

最大的区别显示在突出显示的行中。JSON 格式为我们提供了正确解码的整数值和正确的值序列。我们不需要使用`eval()`或`ast.literal_eval()`来解码元组。其他部分，构建`Player`和配置`Simulate`对象，与`main_ini()`版本类似。

### 使用展平的 JSON 配置

如果我们想通过集成多个配置文件来提供默认值，我们不能同时使用`ChainMap`和类似这样的嵌套字典。我们必须要么展平程序的参数，要么寻找合并来自不同来源的参数的替代方法。

我们可以通过在名称之间使用简单的`.`分隔符来轻松地展平名称。我们的 JSON 文件可能看起来像以下代码：

```py
{
"player.betting": "Flat",
"player.play": "SomeStrategy",
"player.rounds": 100,
"player.stake": 50,
"table.dealer": "Hit17",
"table.decks": 6,
"table.limit": 50,
"table.payout": [3, 2],
"table.split": "NoResplitAces",
"simulator.outputfile": "p2_c13_simulation.dat",
"simulator.samples": 100
}
```

这有利于我们使用`ChainMap`从各种来源累积配置值。它还略微简化了定位特定参数值的语法。给定配置文件名列表`config_names`，我们可能会这样做：

```py
config = ChainMap( *[json.load(file) for file in reversed(config_names)] )
```

从*反向*配置文件名列表构建一个适当的`ChainMap`。为什么是反向的？我们必须反转列表，因为我们希望列表从最具体的开始到最一般的结束。这与`configparser`使用列表的方式相反，也与我们通过将子项添加到映射列表的前面来逐步构建`ChainMap`的方式相反。在这里，我们只是将一系列`dict`加载到`ChainMap`中，第一个`dict`将是被键首先搜索的。

我们可以使用类似这样的方法来利用`ChainMap`。我们只会展示第一部分，构建`Table`实例：

```py
def main_cm( config ):
    dealer_nm= config.get('table.dealer', 'Hit17')
    dealer_rule= {'Hit17':Hit17(),
        'Stand17':Stand17()}.get(dealer_nm, Hit17())
    split_nm= config.get('table.split', 'ReSplit')
    split_rule= {'ReSplit':ReSplit(),
        'NoReSplit':NoReSplit(),
        'NoReSplitAces':NoReSplitAces()}.get(split_nm, ReSplit())
    decks= int(config.get('table.decks', 6))
    limit= int(config.get('table.limit', 100))
    **payout= config.get('table.payout', (3,2))
    table= Table( decks=decks, limit=limit, dealer=dealer_rule,
        split=split_rule, payout=payout )
```

其他部分，构建`Player`和配置`Simulate`对象，与`main_ini()`版本类似。

当我们将其与使用`configparser`的先前版本进行比较时，很明显复杂性几乎相同。命名稍微简单。在这里，我们使用`int(config.get('table.decks'))`而不是`config.getint('table','decks')`。

### 加载 YAML 配置

由于 YAML 语法包含 JSON 语法，前面的例子也可以用 YAML 和 JSON 加载。这是从 JSON 文件中的嵌套字典技术的版本：

```py
player:
  betting: Flat
  play: SomeStrategy
  rounds: 100
  stake: 50
table:
  dealer: Hit17
  decks: 6
  limit: 50
  payout: [3, 2]
  split: NoResplitAces
simulator: {outputfile: p2_c13_simulation.dat, samples: 100}
```

这是比纯 JSON 更好的文件语法；更容易编辑。对于配置主要由字符串和整数控制的应用程序，这有很多优势。加载此文件的过程与加载 JSON 文件的过程相同：

```py
import yaml
config= yaml.load( "config.yaml" )
```

这与嵌套字典具有相同的限制。除非我们展平名称，否则我们没有处理默认值的简单方法。

然而，当我们超越简单的字符串和整数时，我们可以尝试利用 YAML 编码类名和创建我们定制类的实例的能力。这是一个 YAML 文件，将直接构建我们模拟所需的配置对象：

```py
# Complete Simulation Settings
table: !!python/object:__main__.Table
  dealer: !!python/object:__main__.Hit17 {}
  decks: 6
  limit: 50
  payout: !!python/tuple [3, 2]
  split: !!python/object:__main__.NoReSplitAces {}
player: !!python/object:__main__.Player
  betting:  !!python/object:__main__.Flat {}
  init_stake: 50
  max_rounds: 100
  play: !!python/object:__main__.SomeStrategy {}
  rounds: 0
  stake: 63.0
samples: 100
outputfile: p2_c13_simulation9.dat
```

我们已经在 YAML 中编码了类名和实例构造，允许我们定义`Table`和`Player`的完整初始化。我们可以像这样使用这个初始化文件：

```py
import yaml
if __name__ == "__main__":
    config= yaml.load( yaml1_file )
    simulate( config['table'], config['player'],
        config['outputfile'], config['samples'] )
```

这向我们展示了 YAML 配置文件可以用于人工编辑。YAML 为我们提供了与 Python 相同的功能，但具有不同的语法。对于这种类型的示例，Python 配置脚本可能比 YAML 更好。

## 将配置存储在属性文件中

属性文件通常与 Java 程序一起使用。我们没有理由不使用它们与 Python 一起使用。它们相对容易解析，并允许我们以方便、易于使用的格式编码配置参数。有关格式的更多信息，请参阅：[`en.wikipedia.org/wiki/.properties`](http://en.wikipedia.org/wiki/.properties)。属性文件可能如下所示：

```py
# Example Simulation Setup

player.betting: Flat
player.play: SomeStrategy
player.rounds: 100
player.stake: 50

table.dealer: Hit17
table.decks: 6
table.limit: 50
table.payout: (3,2)
table.split: NoResplitAces

simulator.outputfile = p2_c13_simulation8.dat
simulator.samples = 100
```

这在简单性方面有一些优势。`section.property`限定名称通常被使用。这些在非常复杂的配置文件中可能会变得很长。

### 解析属性文件

Python 标准库中没有内置的属性解析器。我们可以从 Python 包索引（[`pypi.python.org/pypi`](https://pypi.python.org/pypi)）下载属性文件解析器。然而，这不是一个复杂的类，这是一个很好的高级面向对象编程练习。

我们将类分解为顶层 API 函数和较低级别的解析函数。以下是一些整体 API 方法：

```py
import re
class PropertyParser:
    def read_string( self, data ):
        return self._parse(data)
    def read_file( self, file ):
        data= file.read()
        return self.read_string( data )
    def read( self, filename ):
        with open(filename) as file:
            return self.read_file( file )
```

这里的基本特性是它将解析文件名、文件或一块文本。这遵循了`configparser`的设计模式。一个常见的替代方法是减少方法的数量，并使用`isinstance()`来确定参数的类型，还确定要对其执行什么处理。

文件名是字符串。文件本身通常是`io.TextIOBase`的实例。一块文本也是一个字符串。因此，许多库使用`load()`来处理文件或文件名，使用`loads()`来处理简单的字符串。类似这样的东西会回显`json`的设计模式：

```py
    def load( self, file_or_name ):
        if isinstance(file_or_name, io.TextIOBase):
            self.loads(file_or_name.read())
        else:
            with open(filename) as file:
                self.loads(file.read())
    def loads( self, string ):
        return self._parse(data)
```

这些方法也可以处理文件、文件名或文本块。这些额外的方法名称为我们提供了一个可能更容易使用的替代 API。决定因素是在各种库、包和模块之间实现一致的设计。这是`_parse()`方法：

```py
    key_element_pat= re.compile(r"(.*?)\s*(?<!\\)[:=\s]\s*(.*)")
    def _parse( self, data ):
        logical_lines = (line.strip()
            for line in re.sub(r"\\\n\s*", "", data).splitlines())
        non_empty= (line for line in logical_lines
            if len(line) != 0)
        non_comment= (line for line in non_empty
            if not( line.startswith("#") or line.startswith("!") ) )
        for line in non_comment:
            ke_match= self.key_element_pat.match(line)
            if ke_match:
                key, element = ke_match.group(1), ke_match.group(2)
            else:
                key, element = line, ""
            key= self._escape(key)
            element= self._escape(element)
            yield key, element
```

这个方法从三个生成器表达式开始，处理属性文件中物理行和逻辑行的一些整体特性。生成器表达式的优势在于它们被惰性执行；直到它们被`for line in non_comment`语句评估时，这些表达式才会创建中间结果。

第一个表达式赋给`logical_lines`，合并以`\`结尾的物理行，以创建更长的逻辑行。前导（和尾随）空格被去除，只留下行内容。**正则表达式**（**RE**）`r"\\\n\s*"`旨在匹配行尾的`\`和下一行的所有前导空格。

第二个表达式赋给`non_empty`，只会迭代长度非零的行。空行将被这个过滤器拒绝。

第三，`non_comment`表达式只会迭代不以`#`或`!`开头的行。以`#`或`!`开头的行将被这个过滤器拒绝。

由于这三个生成器表达式，`for line in non_comment`循环只会迭代非注释、非空白、逻辑行，这些行已经合并并去除了空格。循环的主体将剩下的每一行分开，以分隔键和元素，然后应用`self._escape()`函数来扩展任何转义序列。

键-元素模式`key_element_pat`寻找非转义的显式分隔符`:`, `=`或由空白包围的空格。这个模式使用否定的后行断言，一个`(?<!\\)`的 RE，表示接下来的 RE 必须是非转义的；接下来的模式前面不能有`\`。这意味着`(?<!\\)[:=\s]`是非转义的`:`，或`=`, 或空格。

如果找不到键-元素模式，就没有分隔符。我们解释这种缺乏匹配模式表示该行是一个只有键的退化情况；没有提供值。

由于键和元素形成了一个 2 元组的序列，这个序列可以很容易地转换成一个字典，提供一个配置映射，就像我们看到的其他配置表示方案一样。它们也可以保留为一个序列，以显示文件的原始内容。最后一部分是一个小的方法函数，将转义转换为它们的最终字符：

```py
    def _escape( self, data ):
        d1= re.sub( r"\\([:#!=\s])", lambda x:x.group(1), data )
        d2= re.sub( r"\\u([0-9A-Fa-f]+)", lambda x:chr(int(x.group(1),16)), d1 )
        return d2
```

这个`_escape()`方法函数执行两次替换。第一次替换将转义的标点符号替换为它们的纯文本版本：`\:`, `\#`, `\!`, `\=`, 和 `\`都去掉了`\`。对于 Unicode 转义，使用数字字符串创建一个适当的 Unicode 字符，替换`\uxxxx`序列。十六进制数字被转换为整数，然后转换为替换的字符。

这两个替换可以合并成一个单独的操作，以节省创建一个只会被丢弃的中间字符串。这将提高性能。可能看起来像以下代码：

```py
        d2= re.sub( r"\\([:#!=\s])|\\u([0-9A-Fa-f]+)",
            lambda x:x.group(1) if x.group(1) else chr(int(x.group(2),16)), data )
```

更好性能的好处可能会被正则表达式和替换函数的复杂性所抵消。

### 使用属性文件

我们在如何使用属性文件上有两种选择。我们可以遵循`configparser`的设计模式，解析多个文件以创建一个从各种值的并集中得到的单一映射。或者，我们可以遵循`ChainMap`模式，为每个配置文件创建一个属性映射序列。

`ChainMap`处理相当简单，并为我们提供了所有必需的功能：

```py
config= ChainMap(
    *[dict( pp.read(file) )
        for file in reversed(candidate_list)] )
```

我们按照相反的顺序列出了列表：最具体的设置将首先出现在内部列表中；最一般的设置将是最后一个。一旦`ChainMap`被加载，我们就可以使用这些属性来初始化和构建我们的`Player`、`Table`和`Simulate`实例。

这似乎比从几个来源更新单个映射更简单。此外，这遵循了处理 JSON 或 YAML 配置文件的模式。

我们可以使用类似这样的方法来利用`ChainMap`。这与之前显示的`main_cm()`函数非常相似。我们只会向您展示构建`Table`实例的第一部分：

```py
import ast
def main_cm_str( config ):
    dealer_nm= config.get('table.dealer', 'Hit17')
    dealer_rule= {'Hit17':Hit17(),
        'Stand17':Stand17()}.get(dealer_nm, Hit17())
    split_nm= config.get('table.split', 'ReSplit')
    split_rule= {'ReSplit':ReSplit(),
        'NoReSplit':NoReSplit(),
        'NoReSplitAces':NoReSplitAces()}.get(split_nm, ReSplit())
    decks= int(config.get('table.decks', 6))
    limit= int(config.get('table.limit', 100))
    **payout= ast.literal_eval(config.get('table.payout', '(3,2)'))
    table= Table( decks=decks, limit=limit, dealer=dealer_rule,
        split=split_rule, payout=payout )
```

这个版本与`main_cm()`函数的区别在于处理支付元组的方式。在以前的版本中，JSON（和 YAML）可以解析元组。当使用属性文件时，所有值都是简单的字符串。我们必须使用`eval()`或`ast.literal_eval()`来评估给定的值。这个`main_cm_str()`函数的其他部分与`main_cm()`是相同的。

## 将配置存储在 XML 文件中 - PLIST 和其他文件

正如我们在第九章中所指出的，*序列化和保存 - JSON、YAML、Pickle、CSV 和 XML*，Python 的`xml`包包括许多解析 XML 文件的模块。由于 XML 文件的广泛采用，通常需要在 XML 文档和 Python 对象之间进行转换。与 JSON 或 YAML 不同，从 XML 的映射并不简单。

在 XML 中表示配置数据的一种常见方式是`.plist`文件。有关`.plist`格式的更多信息，请参阅：[`developer.apple.com/documentation/Darwin/Reference/ManPages/man5/plist.5.html`](http://developer.apple.com/documentation/Darwin/Reference/ManPages/man5/plist.5.html)

Macintosh 用户可以执行`man plist`来查看这个 man 页面。`.plist`格式的优点是它使用了少量非常通用的标签。这使得创建和解析`.plist`文件变得容易。这是我们配置参数的示例`.plist`文件。

```py
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>player</key>
  <dict>
    <key>betting</key>
    <string>Flat</string>
    <key>play</key>
    <string>SomeStrategy</string>
    <key>rounds</key>
    <integer>100</integer>
    <key>stake</key>
    <integer>50</integer>
  </dict>
  <key>simulator</key>
  <dict>
    <key>outputfile</key>
    <string>p2_c13_simulation8.dat</string>
    <key>samples</key>
    <integer>100</integer>
  </dict>
  <key>table</key>
  <dict>
    <key>dealer</key>
    <string>Hit17</string>
    <key>decks</key>
    <integer>6</integer>
    <key>limit</key>
    <integer>50</integer>
    <key>payout</key>
    <array>
      <integer>3</integer>
      <integer>2</integer>
    </array>
    <key>split</key>
    <string>NoResplitAces</string>
  </dict>
</dict>
</plist>
```

在这个例子中，我们展示了嵌套的字典结构。有许多与 XML 标签编码兼容的 Python 类型。

| Python 类型 | Plist 标签 |
| --- | --- |
| `str` | `<string>` |
| `float` | `<real>` |
| `int` | `<integer>` |
| `datetime` | `<date>` |
| `boolean` | `<true/> 或 <false/>` |
| `bytes` | `<data>` |
| `list` | `<array>` |
| `dict` | `<dict>` |

正如前面的例子所示，字典`<key>`的值是字符串。这使得 plist 成为我们模拟应用程序参数的非常愉快的编码方式。我们可以相对容易地加载`.plist`：

```py
import plistlib
print( plistlib.readPlist(plist_file) )
```

这将重建我们的配置参数。然后，我们可以在 JSON 配置文件的前一节中使用`main_nested_dict()`函数显示的嵌套字典结构。

使用单个模块函数来解析文件使`.plist`格式非常吸引人。对于任何自定义的 Python 类定义的支持不足，这使得它等同于 JSON 或属性文件。

### 自定义 XML 配置文件

对于更复杂的 XML 配置文件，请参阅[`wiki.metawerx.net/wiki/Web.xml`](http://wiki.metawerx.net/wiki/Web.xml)。这些文件包含特定用途的标签和通用标签的混合。这些文档可能很难解析。有两种一般的方法：

+   编写一个文档处理类，使用 XPath 查询来定位文档中包含有趣数据的标签。在这种情况下，我们将编写能够在 XML 文档结构中定位请求信息的属性（或方法）。

+   将 XML 文档解开成 Python 数据结构。这是之前展示的`plist`模块所采用的方法。

根据`web.xml`文件的示例，我们将设计我们自己的定制 XML 文档来配置我们的模拟应用程序：

```py
<?xml version="1.0" encoding="UTF-8"?>
<simulation>
    <table>
        <dealer>Hit17</dealer>
        <split>NoResplitAces</split>
        <decks>6</decks>
        <limit>50</limit>
        <payout>(3,2)</payout>
    </table>
    <player>
        <betting>Flat</betting>
        <play>SomeStrategy</play>
        <rounds>100</rounds>
        <stake>50</stake>
    </player>
    <simulator>
        <outputfile>p2_c13_simulation11.dat</outputfile>
        <samples>100</samples>
    </simulator>
</simulation>
```

这是一个专门的 XML 文件。我们没有提供 DTD 或 XSD，因此没有正式的方法来根据模式验证 XML。但是，这个文件很小，易于调试，并且与其他示例初始化文件相似。这里有一个`Configuration`类，可以使用 XPath 查询从这个文件中检索信息：

```py
import xml.etree.ElementTree as XML
class Configuration:
    def read_file( self, file ):
        self.config= XML.parse( file )
    def read( self, filename ):
        self.config= XML.parse( filename )
    def read_string( self, text ):
        self.config= XML.fromstring( text )
    def get( self, qual_name, default ):
        section, _, item = qual_name.partition(".")
        query= "./{0}/{1}".format( section, item )
        node= self.config.find(query)
        if node is None: return default
        return node.text
    def __getitem__( self, section ):
        query= "./{0}".format(section)
        parent= self.config.find(query)
        return dict( (item.tag, item.text) for item in parent )
```

我们实现了三种方法来加载 XML 文档：`read()`，`read_file()`和`read_string()`。这些方法只是将自己委托给`xml.etree.ElementTree`类的现有方法函数。这与`configparser`API 相似。我们也可以使用`load()`和`loads()`，因为它们会将自己委托给`parse()`和`fromstring()`。

为了访问配置数据，我们实现了两种方法：`get()`和`__getitem__()`。`get()`方法允许我们使用这样的代码：`stake= int(config.get('player.stake', 50))`。`__getitem__()`方法允许我们使用这样的代码：`stake= config['player']['stake']`。

解析比`.plist`文件稍微复杂一些。但是，XML 文档比等效的`.plist`文档简单得多。

我们可以使用前一节中显示的`main_cm_str()`函数来处理属性文件上的配置。

## 总结

我们研究了许多表示配置参数的方法。其中大多数是基于我们在第九章中看到的更一般的序列化技术，*序列化和保存-JSON、YAML、Pickle、CSV 和 XML*。`configparser`模块提供了一个额外的格式，对一些用户来说更舒适。

配置文件的关键特征是内容可以轻松地由人类编辑。因此，pickle 文件不建议作为良好的表示。

### 设计考虑和权衡

配置文件可以简化运行应用程序或启动服务器。这可以将所有相关参数放在一个易于阅读和易于修改的文件中。我们可以将这些文件放在配置控制下，跟踪更改历史，并通常使用它们来提高软件的质量。

对于这些文件，我们有几种替代格式，所有这些格式都相对友好，易于编辑。它们在解析的难易程度以及可以编码的 Python 数据的任何限制方面有所不同：

+   **INI 文件**：这些文件易于解析，仅限于字符串和数字。

+   **Python 代码（PY 文件）**：这些文件使用主脚本进行配置。没有解析，没有限制。它们使用`exec()`文件。易于解析，没有限制。

+   **JSON 或 YAML 文件**：这些文件易于解析。它们支持字符串，数字，字典和列表。YAML 可以编码 Python，但为什么不直接使用 Python 呢？

+   **属性文件**：这些文件需要一个特殊的解析器。它们仅限于字符串。

+   **XML 文件**：

+   `.plist` **文件**：这些文件易于解析。它们支持字符串，数字，字典和列表。

+   **定制 XML**：这些文件需要一个特殊的解析器。它们仅限于字符串。

与其他应用程序或服务器的共存通常会确定配置文件的首选格式。如果我们有其他应用程序使用`.plist`或 INI 文件，那么我们的 Python 应用程序应该做出更符合用户使用习惯的选择。

从可以表示的对象的广度来看，我们有四个广泛的配置文件类别：

+   **只包含字符串的简单文件**：定制 XML，属性文件。

+   **简单文件，只包含简单的 Python 文字**：INI 文件。

+   **更复杂的文件，包含 Python 文字，列表和字典**：JSON，YAML，`.plist`和 XML。

+   **任何东西。Python**：我们可以使用 YAML，但当 Python 有更清晰的语法时，这似乎有些愚蠢。

### 创建共享配置

当我们在第十七章中查看模块设计考虑时，*模块和包设计*，我们将看到模块如何符合**单例**设计模式。这意味着我们只能导入一个模块，并且单个实例是共享的。

因此，通常需要在一个独立的模块中定义配置并导入它。这允许单独的模块共享一个公共配置。每个模块都将导入共享配置模块；配置模块将定位配置文件并创建实际的配置对象。

### 模式演变

配置文件是公共 API 的一部分。作为应用程序设计者，我们必须解决模式演变的问题。如果我们改变一个类的定义，我们将如何改变配置？

因为配置文件通常具有有用的默认值，它们通常非常灵活。原则上，内容是完全可选的。

当软件经历主要版本更改时，改变 API 或数据库模式的更改，配置文件也可能经历重大更改。配置文件的版本号可能必须包含以消除旧版配置参数和当前发布参数之间的歧义。

对于次要版本更改，配置文件，如数据库、输入和输出文件以及 API，应保持兼容。任何配置参数处理都应具有适当的默认值，以应对次要版本更改。

配置文件是应用程序的一流输入。它不是事后想法或变通方法。它必须像其他输入和输出一样经过精心设计。当我们在第十四章中查看更大的应用程序架构设计时，*日志和警告模块*和第十六章，*处理命令行*，我们将扩展解析配置文件的基础知识。

### 展望未来

在接下来的章节中，我们将看到更大规模的设计考虑。第十四章，*日志和警告模块*，将介绍如何使用`logging`和`warnings`模块来创建审计信息以及调试。我们将探讨为可测试性设计以及在第十五章中如何使用`unittest`和`doctest`。第十六章，*处理命令行*，将介绍如何使用`argparse`模块来解析选项和参数。我们将进一步使用**命令**设计模式来创建可以组合和扩展的程序组件，而不必编写 shell 脚本。在第十七章，*模块和包设计*，我们将探讨模块和包的设计。在第十八章，*质量和文档*，我们将探讨如何记录我们的设计以创建对我们的软件正确性和正确实现的信任。
