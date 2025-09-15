## 5

内置数据结构第二部分：字典

从第四章开始，内置数据结构第一部分：列表和集合，我们开始探讨 Python 丰富的内置数据结构集合。这些数据结构有时被称为“容器”或“集合”，因为它们包含一系列单独的项目。

在本章中，我们将介绍字典结构。字典是从键到值的映射，有时也称为关联数组。将映射与两个序列（列表和集合）分开似乎是有道理的。

本章还将探讨一些与 Python 处理可变集合对象引用相关的更高级主题。这会影响函数的定义方式。

在本章中，我们将探讨以下菜谱，所有这些都与 Python 的内置数据结构相关：

+   创建字典 – 插入和更新

+   缩小字典 – pop() 方法和 del 语句

+   编写与字典相关的类型提示

+   理解变量、引用和赋值

+   创建对象的浅拷贝和深拷贝

+   避免为函数参数使用可变默认值

我们将从如何创建一个字典开始。

# 5.1 创建字典 – 插入和更新

字典是 Python 的一种映射类型。内置类型 dict 提供了许多基础特性。在 collections 模块中定义了一些这些特性的常见变体。

正如我们在第四章开头的 选择数据结构 菜谱中提到的，当我们有一个需要将键映射到给定值的键时，我们将使用字典。例如，我们可能想要将一个单词映射到该单词的复杂定义，或者将某个值映射到该值在数据集中出现的次数。

## 5.1.1 准备工作

我们将探讨一个用于定位事务处理各个阶段的算法。这依赖于为每个请求分配一个唯一的 ID，并在事务期间包含每个日志记录中该 ID。由于多线程服务器可能同时处理多个请求，每个请求的事务阶段将不可预测地交错。按请求 ID 重新组织日志有助于隔离每个事务。

这里是一个模拟的三个并发请求的日志条目序列：

```py
[2019/11/12:08:09:10,123] INFO #PJQXB^{}eRwnEGG?2%32U path="/openapi.yaml" method=GET 

[2019/11/12:08:09:10,234] INFO 9DiC!B^{}nXxnEGG?2%32U path="/items?limit=x" method=GET 

[2019/11/12:08:09:10,235] INFO 9DiC!B^{}nXxnEGG?2%32U error="invalid query" 

[2019/11/12:08:09:10,345] INFO #PJQXB^{}eRwnEGG?2%32U status="200" bytes="11234" 

[2019/11/12:08:09:10,456] INFO 9DiC!B^{}nXxnEGG?2%32U status="404" bytes="987" 

[2019/11/12:08:09:10,567] INFO >~UL>~PB_R>&nEGG?2%32U path="/category/42" method=GET
```

行很长，可能随意换行以适应书籍的边距。每一行都有一个时间戳。示例中显示的每个记录的严重级别都是 INFO。接下来的 20 个字符是一个事务 ID。之后是针对事务中特定步骤的日志信息。

以下正则表达式定义了日志记录：

```py
import re 

log_parser = re.compile(r"\[(.*?)\] (\w+) (\S+) (.*)")
```

此模式捕获每个日志条目的四个字段。有关正则表达式的更多信息，请参阅第一章的 使用正则表达式进行字符串解析 菜谱。

解析这些行将产生一个四元组序列。结果对象看起来像这样：

```py
[(’2019/11/12:08:09:10,123’, 

  ’INFO’, 

  ’#PJQXB^{}eRwnEGG?2%32U’, 

  ’path="/openapi.yaml" method=GET’), 

 (’2019/11/12:08:09:10,234’, 

  ’INFO’, 

  ’9DiC!B^{}nXxnEGG?2%32U’, 

  ’path="/items?limit=x" method=GET’),
```

```py
... details omitted ...
```

```py
 (’2019/11/12:08:09:10,567’, 

  ’INFO’, 

  ’>~UL>~PB_R>&nEGG?2%32U’, 

  ’path="/category/42" method=GET’)]
```

我们需要知道每个唯一路径被请求的频率。这意味着忽略一些日志记录并从其他记录中收集数据。从路径字符串到计数的映射是一种优雅地收集这些数据的方法。我们将详细讨论如何实现。稍后，我们将查看 collections 模块中的一些替代实现。

## 5.1.2 如何实现...

我们有多种构建字典对象的方法：

+   文字：我们可以通过使用由 {} 字符包围的键值对序列来创建字典的显示。我们在键和关联的值之间使用冒号。文字看起来像这样：{"num": 355, "den": 113}。

+   转换函数：一个由两个元组组成的序列可以转换成这样的字典：dict([(‘num’，355)，（‘den’，113）]). 每个元组变成一个键值对。键必须是不可变对象，如字符串、数字或不可变对象的元组。我们也可以这样构建字典：dict(num=355, den=113). 每个参数名称都变成一个键。这限制了字典键为字符串，这些字符串也是有效的 Python 变量名。

+   插入：我们可以使用字典 [key] = value 语法在字典中设置或替换一个值。我们将在本菜谱的后面讨论这个问题。

+   理解：与列表和集合类似，我们可以编写一个字典理解来从某些数据源构建字典。

### 通过设置项构建字典

我们通过创建一个空字典然后向其中设置项来构建字典：

1.  创建一个空字典以映射路径到计数。我们也可以使用 dict() 来创建一个空字典。由于我们将创建一个计数路径使用次数的直方图，我们将它命名为 histogram：

    ```py
    >>> histogram = {}
    ```

    我们也可以使用函数 dict() 而不是字面量值 {} 来创建一个空字典。

1.  对于每条日志行，过滤掉那些在索引为 3 的项中没有以 path 开头的值的那些行：

    ```py
    >>> for line in log_lines: 

    ...     path_method = line[3]  # group(4) of the original match 

    ...     if path_method.startswith("path"):
    ```

1.  如果路径不在字典中，我们需要添加它。一旦 path_method 字符串的值在字典中，我们就可以根据数据中的键在字典中增加值。

    ```py
     ...         if path_method not in histogram: 

    ...             histogram[path_method] = 0 

    ...         histogram[path_method] += 1
    ```

这种技术将每个新的 path_method 值添加到字典中。一旦确定 path_method 键在字典中，我们就可以增加与该键关联的值。

### 通过理解构建字典

每条日志行的最后一个字段有一个或两个字段。可能有一个像 path="/openapi.yaml" method=GET 这样的值，包含两个属性 path 和 method，或者一个像 error="invalid query" 这样的值，只有一个属性 error。

我们可以使用以下正则表达式来分解每行的最终字段：

```py
param_parser = re.compile( 

    r’(\w+)=(".*?"|\w+)’ 

)
```

这个正则表达式的 findall()方法将基于匹配文本提供一个包含两个元组的序列。然后我们可以从匹配组的序列构建一个字典：

1.  对于每条日志行，应用正则表达式以创建一对序列：

    ```py
    >>> for line in log_lines: 

    ...     name_value_pairs = param_parser.findall(line[3])
    ```

1.  使用字典推导式，将第一个匹配组作为键，第二个匹配组作为值：

    ```py
     ...     params = {match[0]: match[1] for match in name_value_pairs} 
    ```

我们可以打印出参数值，我们会看到如下示例中的字典：

```py
{’path’: ’"/openapi.yaml"’, ’method’: ’GET’} 

{’path’: ’"/items?limit=x"’, ’method’: ’GET’} 

{’error’: ’"invalid query"’}
```

使用字典作为每个日志记录的最终字段，使得分离重要信息变得更容易。

## 5.1.3 工作原理...

字典的核心功能是从不可变键到任何类型值对象的映射。在第一个示例中，我们使用了不可变的字符串作为键，整数作为值。我们在类型提示中描述它为 dict[str, int]。

理解+=赋值语句的工作方式很重要。+=的实现基本上是这样的：

```py
histogram[customer] = histogram[customer] + 1
```

从字典中获取的直方图[客户]值被计算出一个新值，并将结果用于更新字典。

确保字典键对象是不可变的。我们不能使用列表、集合或字典作为字典映射的键。然而，我们可以将列表转换为不可变的元组，或者将集合转换为 frozenset，这样我们就可以使用这些更复杂对象中的一个作为键。在本食谱中显示的示例中，我们使用了不可变的字符串作为每个字典的键。

## 5.1.4 更多...

我们不必使用 if 语句来添加缺失的键。我们可以使用字典的 setdefault()方法。使用 collections 模块中的类甚至更容易。

这是使用 collections 模块中的 defaultdict 类的版本：

```py
>>> from collections import defaultdict

>>> histogram = defaultdict(int) 

>>> for line in log_lines: 

...     path_method = line[3]  # group(4) of the match 

...     if path_method.startswith("path"): 

...         histogram[path_method] += 1
```

我们创建了一个 defaultdict 实例，它将使用 int()函数初始化任何未知的键值。我们将函数对象 int 提供给了 defaultdict 构造函数。defaultdict 实例将评估给定的函数以创建默认值。

这允许我们使用 histogram[path_method] += 1。如果与 path_method 键关联的值之前在字典中，该值将增加并放回字典中。如果 path_method 键不在字典中，则调用 int()函数不带任何参数；这个默认值将被增加并放入字典中。

我们还可以通过创建 Counter 对象来累积频率计数。我们可以如下从原始数据构建 Counter 对象：

```py
>>> from collections import Counter 

>>> filtered_paths = ( 

...     line[3] 

...     for line in log_lines 

...     if line[3].startswith("path") 

... ) 

>>> histogram = Counter(filtered_paths) 

>>> histogram 

Counter({’path="/openapi.yaml" method=GET’: 1, ’path="/items?limit=x" method=GET’: 1, ’path="/category/42" method=GET’: 1})
```

首先，我们使用生成表达式创建了一个过滤路径数据的迭代器；这被分配给了 filtered_paths。然后我们从数据源创建了一个 Counter；Counter 类将扫描数据并计算不同出现的次数。

## 5.1.5 参见

+   在收缩字典 – pop() 方法和 del 语句的菜谱中，我们将探讨如何通过删除项来修改字典。

# 5.2 收缩字典 – pop() 方法和 del 语句

字典的一个常见用途是作为关联存储：它保持键和值对象之间的关联。这意味着我们可能对字典中的项执行任何 CRUD 操作：

+   创建一个新的键值对。

+   获取与键关联的值。

+   更新与键关联的值。

+   从字典中删除键（及其对应的值）。

## 5.2.1 准备工作

大量的处理支持围绕一个（或多个）不同的共同值对项目进行分组的需求。我们将回到本章中创建字典 – 插入和更新菜谱中显示的日志数据。

我们将使用一个迭代器算法，该算法使用交易 ID 作为字典中的键。这个键的值将是交易的步骤序列。在非常长的日志中，我们通常不想在巨大的字典中保存每一笔交易。当我们达到交易序列的终止时，我们可以产生交易日志条目的列表。一个函数可以消费这个迭代器，独立地处理每一批交易。

## 5.2.2 如何做...

这个菜谱的上下文将需要一个条件为 match := log_parser.match(line) 的 if 语句。这将应用正则表达式，并将结果收集在 match 变量中。给定这个上下文，更新或从字典中删除的处理如下：

1.  这个函数使用 defaultdict 类和两个额外的类型提示，可迭代和迭代器：

    ```py
    from collections import defaultdict 

    from collections.abc import Iterable, Iterator
    ```

1.  定义一个 defaultdict 对象来保存交易步骤。键是 20 个字符的字符串。值是日志记录的列表。在这种情况下，每个日志记录都将从源文本解析成单个字符串的元组：

    ```py
    LogRec = tuple[str, ...] 

    def request_iter_t(source: Iterable[str]) -> Iterator[list[LogRec]]: 

        requests: defaultdict[str, list[LogRec]] = defaultdict(list)
    ```

1.  定义每个日志条目组的键：

    ```py
     for line in source: 

            if match := log_parser.match(line): 

                id = match.group(3)
    ```

1.  使用日志记录更新字典项：

    ```py
     requests[id].append(tuple(match.groups()))
    ```

1.  如果这个日志记录完成了一笔交易，作为生成器函数的一部分产生这个组。然后从字典中删除交易，因为它已经完成：

    ```py
     if match.group(4).startswith(’status’): 

                    yield requests[id] 

                    del requests[id]
    ```

1.  最后，可能会有一个非空的请求字典。这反映了在日志文件切换时正在进行的交易。

## 5.2.3 它是如何工作的...

因为字典是一个可变对象，所以我们可以从字典中删除键。del 语句将删除与键关联的键和值对象。在这个例子中，当数据表明交易完成时，键被删除。一个处理平均每秒 10 笔交易的繁忙的 Web 服务器在 24 小时内将看到 864,000 笔交易。如果每笔交易平均有 2.5 条日志条目，文件中至少将有 2,160,000 行。

如果我们只想知道每个资源的耗时，我们不想在内存中保留包含 864,000 个事务的整个字典。我们更愿意将日志转换成一个中间摘要文件以供进一步分析。

这种临时数据的概念使我们把解析的日志行累积到一个列表实例中。每行新内容都追加到属于该事务的适当列表中。当找到最后一行时，这些行可以从字典中清除。

## 5.2.4 更多内容...

在示例中，我们使用了 del 语句。pop()方法也可以使用。如果给定的项目在字典中找不到，del 语句将引发 KeyError 异常。

pop()方法看起来是这样的：

```py
        requests.pop(id)
```

这将就地修改字典，如果存在则删除项目，或者引发 KeyError 异常。

当 pop()方法提供一个默认值时，如果找不到键，它可以返回给定的默认值而不是引发异常。在任何情况下，键将不再存在于字典中。请注意，此方法既修改了集合又返回了一个值。

popitem()方法将从字典中删除一个键值对。这些对以最后进入，最先出来（LIFO）的顺序返回。这意味着字典也是一种栈。

## 5.2.5 参见

+   在创建字典 – 插入和更新菜谱中，我们探讨了如何创建字典并将它们填充键和值。

# 5.3 编写与字典相关的类型提示

当我们查看集合和列表时，我们通常期望列表（或集合）中的每个项目都是相同类型。当我们查看面向对象的类设计时，在第七章中，我们将看到如何一个公共超类可以成为紧密相关的对象类型家族的共同类型。虽然在一个列表或集合集合中可以有异构类型，但通常处理起来相当复杂，需要匹配语句来进行适当的类型匹配。然而，字典可以用来创建类型的区分联合。特定的键值可以用来定义字典中存在哪些其他键。这意味着一个简单的 if 语句可以区分异构类型。

## 5.3.1 准备工作

我们将查看两种类型的字典类型提示，一种用于同质值类型，另一种用于异构值类型。我们将查看最初是这些类型之一的数据字典，但后来被转换成更复杂的类型定义。

我们将从一个以下 CSV 文件开始：

```py
date,engine on,fuel height on,engine off,fuel height off 

10/25/13,08:24:00,29,13:15:00,27 

10/26/13,09:12:00,27,18:25:00,22 

10/28/13,13:21:00,22,06:25:00,14
```

这描述了乘帆船进行的多日旅行中的三个独立阶段。燃料是通过油箱中的高度来测量的，而不是使用浮子或其他仪表的间接方法。因为油箱大约是矩形的，31 英寸的深度大约是 75 加仑的燃料。

## 5.3.2 如何做...

csv.DictReader 的初始使用将导致具有同质类型定义的字典：

1.  定位字典中键的类型。当读取 CSV 文件时，键是字符串，类型为 str。

1.  定位字典中值的类型。当读取 CSV 文件时，值是字符串，类型为 str。

1.  使用 dict 类型提示组合类型。这产生 dict[str, str]。

这里是一个从 CSV 文件读取数据的示例函数：

```py
import csv 

from pathlib import Path 

def get_fuel_use(source_path: Path) -> list[dict[str, str]]: 

    with source_path.open() as source_file: 

        rdr = csv.DictReader(source_file) 

        data: list[dict[str, str]] = list(rdr) 

    return data
```

get_fuel_use() 函数产生与源数据匹配的值。在这种情况下，它是一个将字符串列名映射到字符串单元格值的字典。

这份数据本身难以处理。常见的第二步是对源行应用转换以创建更有用的数据类型。我们可以用类型提示来描述结果：

1.  确定所需的各种值类型。在这个例子中，有五个字段，三种不同类型，如下所示：

    +   日期字段是一个 datetime.date 对象。

    +   引擎字段是一个 datetime.time 对象。

    +   燃料高度字段是一个整数，但我们知道它将在浮点上下文中使用，因此我们将直接创建一个浮点数。

    +   引擎关闭字段是一个 datetime.time 对象。

    +   燃料高度字段也是一个浮点值。

1.  从 typing 模块导入 TypedDict 类型定义。

1.  定义具有新异构字典类型的 TypedDict 子类。

    ```py
    import datetime 

    from typing import TypedDict 

    class History(TypedDict): 

        date: datetime.date 

        start_time: datetime.time 

        start_fuel: float 

        end_time: datetime.time 

        end_fuel: float
    ```

    这部分是第七章的预告 7。它展示了一种非常简单的类定义。在这种情况下，类是具有五个特定键的字典，所有这些键都是必需的，并且必须具有给定类型的值。

在这个例子中，我们还重命名了字段，使它们成为有效的 Python 名称。用 _ 替换标点是明显的第一步。我们还更改了一些，因为 CSV 文件中的列名看起来很别扭。

执行转换的函数可能看起来像以下示例：

```py
from collections.abc import Iterable, Iterator 

def make_history(source: Iterable[dict[str, str]]) -> Iterator[History]: 

    for row in source: 

        yield dict( 

            date=datetime.datetime.strptime( 

                row[’date’], "%m/%d/%y").date(), 

            start_time=datetime.datetime.strptime( 

                row[’engine on’], ’%H:%M:%S’).time(), 

            start_fuel=float(row[’fuel height on’]), 

            end_time=datetime.datetime.strptime( 

                row[’engine off’], ’%H:%M:%S’).time(), 

            end_fuel=float(row[’fuel height off’]), 

        )
```

这个函数消耗初始 dict[str, str] 字典的实例，并创建由 History 类描述的字典的实例。以下是这两个函数如何一起工作的：

```py
>>> from pprint import pprint 

>>> source_path = Path("data/fuel2.csv") 

>>> fuel_use = make_history(get_fuel_use(source_path)) 

>>> for row in fuel_use: 

...     pprint(row) 

{’date’: datetime.date(2013, 10, 25), 

 ’end_fuel’: 27.0, 

 ’end_time’: datetime.time(13, 15), 

 ’start_fuel’: 29.0, 

 ’start_time’: datetime.time(8, 24)} 

{’date’: datetime.date(2013, 10, 26), 

 ’end_fuel’: 22.0, 

 ’end_time’: datetime.time(18, 25), 

 ’start_fuel’: 27.0, 

 ’start_time’: datetime.time(9, 12)} 

{’date’: datetime.date(2013, 10, 28), 

 ’end_fuel’: 14.0, 

 ’end_time’: datetime.time(6, 25), 

 ’start_fuel’: 22.0, 

 ’start_time’: datetime.time(13, 21)}
```

这展示了如何通过 make_history() 函数处理 get_fuel_use() 函数的输出，以创建一个字典的可迭代序列。每个结果字典都将源数据转换为更实用的类型。

## 5.3.3 它是如何工作的...

字典的核心类型提示命名了键类型和值类型，形式为 dict[key, value]。TypedDict 类允许我们更具体地描述字典键与广泛值域之间的绑定。

重要的是要注意，类型提示仅由 mypy 等程序检查。这些提示对运行时没有影响。例如，我们可以编写如下语句：

```py
result: History = {’date’: 42}
```

这个语句声称结果字典将匹配 History 类型定义中的类型提示。然而，字典字面量在 'date' 字段类型不正确，并且许多其他字段缺失。虽然这会执行，但会从 mypy 引发错误。

运行 mypy 程序会显示如下列表中的错误：

```py
(cookbook3) % python -m mypy src/ch05/recipe_04_bad.py 

src/ch05/recipe_04_bad.py:18: error: Missing keys ("start_time", "start_fuel", "end_time", "end_fuel") for TypedDict "History"  [typeddict-item] 

Found 1 error in 1 file (checked 1 source file)
```

对于运行时数据验证，像 Pydantic 这样的项目可以非常有帮助。

## 5.3.4 更多...

字典键异质性的常见情况之一是可选项。类型提示 Optional[str]或 str | None 描述了这一点。在字典中很少需要这样做，因为它可以更简单地省略整个键值对。

假设我们需要 History 类型的两个变体：

+   在此配方中较早展示的变体，其中所有字段都存在。

+   两个“不完整”的记录，一个没有关机时间或结束燃油高度，另一个变体没有开机时间或起始燃油高度。这两个记录可能用于有动力过夜航行。

在这种情况下，我们可能需要使用 NotRequired 注解这些字段。生成的类定义将如下所示：

```py
from typing import TypedDict, NotRequired 

class History2(TypedDict): 

   date: datetime.date 

   start_time: NotRequired[datetime.time] 

   start_fuel: NotRequired[float] 

   end_time: NotRequired[datetime.time] 

   end_fuel: NotRequired[float] 
```

此记录允许字典值有很大的可变性。它需要使用 if 语句来确定数据中存在的字段组合。此外，它还需要在 make_history()函数中进行一些更复杂的处理，以根据 CSV 文件中的空列创建这些变体记录。

TypedDict 和 NamedTuple 类型定义之间存在一些相似之处。将 TypedDict 更改为 NamedTuple 将创建一个命名元组类而不是类型字典类。

由于 NamedTuple 类有一个 _asdict()方法，因此可以从命名元组生成与 TypedDict 结构匹配的字典。

与 TypedDict 提示匹配的字典是可变的。然而，NamedTuple 的子类是不可变的。这是这两个类型提示之间的一个主要区别。更重要的是，字典使用 row[‘date’]语法通过键‘date’来引用一个项目。命名元组使用 row.date 语法通过名称来引用一个项目。

## 5.3.5 参见

+   使用 NamedTuples 简化元组中的项目访问的配方提供了关于 NamedTuple 类型提示的更多详细信息。

+   关于列表的类型提示，请参阅第四章中的编写与列表相关的类型提示配方。

+   第四章中的编写与集合相关的类型提示配方从集合类型的角度涵盖了这一点。

+   对于运行时数据验证，像 Pydantic 这样的项目可以非常有帮助。请参阅[`docs.pydantic.dev/latest/`](https://docs.pydantic.dev/latest/)。

# 5.4 理解变量、引用和赋值

变量实际上是如何工作的？当我们将可变对象分配给两个变量时会发生什么？当两个变量共享对公共可变对象的引用时，行为可能会令人困惑。

这是核心原则：Python 共享引用；它不会复制数据。

为了了解这个关于引用共享的规则意味着什么，我们将创建两个数据结构：一个是可变的，另一个是不可变的。

## 5.4.1 准备工作

我们将查看两种类型的序列，尽管我们也可以用两种类型的集合做类似的事情：

```py
>>> mutable = [1, 1, 2, 3, 5, 8] 

>>> immutable = (5, 8, 13, 21)
```

我们将查看当这些对象的引用被共享时会发生什么。

我们可以用类似的方式与集合和 frozenset 进行比较。我们无法轻松地这样做，因为 Python 没有提供方便的不可变映射。

## 5.4.2 如何实现...

这个配方将展示如何观察当有两个引用到底层可变对象时的“超距作用”。我们将在制作浅拷贝和深拷贝对象配方中查看防止这种情况的方法。以下是查看可变和不可变集合之间差异的步骤：

1.  将每个集合分配给一个额外的变量。这将创建对该结构的两个引用：

    ```py
    >>> mutable_b = mutable 

    >>> immutable_b = immutable 
    ```

    现在我们有两个引用到列表 [1, 1, 2, 3, 5, 8] 和两个引用到元组 (5, 8, 13, 21)。

1.  我们可以使用 is 运算符来确认这一点。这确定两个变量是否引用了同一个底层对象：

    ```py
    >>> mutable_b is mutable 

    True 

    >>> immutable_b is immutable 

    True
    ```

1.  对集合的两个引用之一进行更改。对于列表类型，我们有像 extend() 或 append() 这样的方法。在这个例子中，我们将使用 + 运算符：

    ```py
    >>> mutable += [mutable[-2] + mutable[-1]]
    ```

    我们可以用类似的方法对不可变结构进行操作：

    ```py
    >>> immutable += (immutable[-2] + immutable[-1],)
    ```

1.  看看引用可变结构的另外两个变量。因为这两个变量是同一个底层列表对象的引用，每个变量都显示了当前状态：

    ```py
    >>> mutable_b 

    [1, 1, 2, 3, 5, 8, 13] 

    >>> mutable is mutable_b 

    True
    ```

1.  看看引用不可变结构的两个变量。最初，两个变量共享一个公共对象。当执行赋值语句时，创建了一个新的元组，只有一个变量更改以引用新的元组：

    ```py
     >>> immutable_b 

    (5, 8, 13, 21) 

    >>> immutable 

    (5, 8, 13, 21, 34)
    ```

## 5.4.3 它是如何工作的...

两个变量 mutable 和 mutable_b 仍然引用同一个底层对象。正因为如此，我们可以使用任何一个变量来更改对象，并看到更改反映在另一个变量的值上。

两个变量 immutable_b 和 immutable 最初引用的是同一个对象。因为该对象不能就地修改，对其中一个变量的更改意味着将新对象分配给该变量。另一个变量仍然牢固地附加到原始对象上。

在 Python 中，变量是一个附加到对象的标签。我们可以把它们想象成我们暂时贴在对象上的鲜艳颜色的粘性便签。可以对对象附加多个标签。是赋值语句将变量名放置在对象上。

考虑以下声明：

```py
immutable += (immutable[-2] + immutable[-1],)
```

这与以下语句具有相同的效果：

```py
immutable = immutable + (immutable[-2] + immutable[-1],)
```

等号右侧的表达式从不可变元组的上一个值创建一个新的元组。然后赋值语句将标签 immutable 分配给新铸造的对象。

将值赋给变量有两种可能的行为：

+   对于提供适当原地赋值操作符定义的可变对象，如`+=`，赋值会被转换为特殊方法；在这种情况下，`__iadd__()`。特殊方法将修改对象的内部状态。

+   对于不提供如`+=`等赋值定义的不可变对象，赋值会被转换为`=`和`+`。`+`操作符将构建一个新的对象，并将变量名附加到这个新对象上。之前引用被替换对象的变量不受影响；它们将继续引用旧对象。

Python 计算一个对象被引用的次数。当引用计数变为零时，该对象在任何地方都不再被使用，可以从内存中移除。

## 5.4.4 更多...

一些语言除了对象外还有原始类型。在这些语言中，`+=`语句可以利用硬件指令的特性来调整原始类型的值。

Python 没有这种优化。数字是不可变对象；没有特殊的指令来调整它们的值。考虑以下赋值语句：

```py
a = 355 

a += 113
```

处理过程不会调整对象 355 的内部状态。`int`类不提供`__iadd__()`特殊方法。将创建一个新的不可变整数对象。这个新对象被标记为 a。之前分配给 a 的旧值不再需要，存储可以被回收。

## 5.4.5 参见

+   在制作浅拷贝和深拷贝对象的菜谱中，我们将探讨如何复制可变结构以防止共享引用。

+   此外，请参阅避免为函数参数使用可变默认值以了解 Python 中引用共享方式的另一种后果。

+   对于 CPython 实现，一些对象可以是永生的。有关此实现细节的更多信息，请参阅[PEP 683](https://peps.python.org/pep-0683/)。

# 5.5 制作浅拷贝和深拷贝对象

在本章中，我们讨论了赋值语句如何共享对象引用。对象通常不会被复制。

考虑以下赋值语句：

```py
a = b
```

这创建了两个指向相同底层对象的引用。如果 b 变量的值具有可变类型，如列表、集合或字典类型，那么使用 a 或 b 进行更改将更新底层可变对象。更多背景信息，请参阅理解变量、引用和赋值菜谱。

大多数情况下，我们希望这种行为。这对于向函数提供可变对象并在函数中有一个局部变量来修改函数外部创建的对象是理想的。在罕见的情况下，我们希望实际上从单个原始对象创建两个独立的对象。

当两个变量引用相同的底层对象时，有两种方法可以断开这种连接：

+   制作结构的浅拷贝

+   创建结构的深拷贝

## 5.5.1 准备工作

Python 不会自动复制一个对象。我们已经看到了几种创建副本的语法：

+   序列 - 列表，以及 str、bytes 和 tuple 类型：我们可以使用序列[:] 通过使用空切片表达式来复制序列。这是序列的一个特例。

+   几乎所有集合都有一个 copy() 方法。

+   调用一个类型，以该类型的实例作为唯一参数，返回一个副本。例如，如果 d 是一个字典，dict(d) 将创建 d 的浅拷贝。

重要的是，这些都是浅拷贝。当两个集合是浅拷贝时，它们各自包含对相同底层对象的引用。如果底层对象是不可变的，如元组、数字或字符串，这种区别并不重要。

例如，如果我们有 a = [1, 1, 2, 3]，我们就无法对 a[0] 进行任何修改。a[0] 中的数字 1 没有内部状态。我们只能替换对象。

然而，当我们有一个涉及可变对象的集合时，问题就出现了。首先，我们会创建一个对象，然后创建一个副本：

```py
>>> some_dict = {’a’: [1, 1, 2, 3]} 

>>> another_dict = some_dict.copy()
```

这个例子创建了一个字典的浅拷贝。这两个副本看起来很相似，因为它们都包含对相同对象的引用。有一个共享的引用到不可变的字符串 'a' 和一个共享的引用到可变的列表 [1, 1, 2, 3]。我们可以显示 another_dict 的值来看到它看起来像我们开始时的 some_dict 对象：

```py
>>> another_dict 

{’a’: [1, 1, 2, 3]}
```

当我们更新字典副本中的共享列表时，会发生以下情况。我们将更改 some_dict 的值，并看到结果也出现在 another_dict 中：

```py
>>> some_dict[’a’].append(5) 

>>> another_dict 

{’a’: [1, 1, 2, 3, 5]}
```

我们可以使用 id() 函数看到项是共享的：

```py
>>> id(some_dict[’a’]) == id(another_dict[’a’]) 

True
```

因为两个 id() 值相同，所以它们是相同的底层对象。与键 'a' 关联的值在 some_dict 和 another_dict 中都是相同的可变列表。我们也可以使用 is 操作符来看到它们是相同的对象。

这种对浅拷贝的修改适用于包含所有其他可变对象类型作为项的列表集合：

因为我们不能创建一个包含可变对象的集合，所以我们实际上不必考虑创建共享项的集合的浅拷贝。

一个元组可以包含可变对象。虽然元组是不可变的，但元组内的对象是可变的。

元组的不可变性不会神奇地传播到元组内的项。

如果我们想要完全断开两个副本的连接？我们如何创建深拷贝而不是浅拷贝？

## 5.5.2 如何实现...

Python 通常通过共享引用来工作。它不情愿地复制对象。默认行为是创建浅拷贝，共享集合中项的引用。以下是创建深拷贝的方法：

1.  导入 copy 模块：

    ```py
    >>> import copy
    ```

1.  使用 copy.deepcopy() 函数复制一个对象及其包含的所有可变项：

    ```py
    >>> some_dict = {’a’: [1, 1, 2, 3]} 

    >>> another_dict = copy.deepcopy(some_dict)
    ```

这将创建没有共享引用的副本。对一个副本的可变内部项的更改不会对其他任何地方产生任何影响：

```py
>>> some_dict[’a’].append(5) 

>>> some_dict 

{’a’: [1, 1, 2, 3, 5]} 

>>> another_dict 

{’a’: [1, 1, 2, 3]}
```

我们更新了 some_dict 中的某个项，但它对另一个 _dict 中的副本没有影响。我们可以使用 id()函数看到这些对象是不同的：

```py
>>> id(some_dict[’a’]) == id(another_dict[’a’]) 

False
```

由于 id 值不同，这些是不同的对象。我们还可以使用 is 运算符来查看它们是不同的对象。

## 5.5.3 它是如何工作的...

创建浅复制相对简单。我们甚至可以用自己的版本编写算法，使用推导式（包含生成器表达式）：

```py
>>> copy_of_list = [item for item in some_list] 

>>> copy_of_dict = {key:value for key, value in some_dict.items()}
```

在列表的情况下，新列表的项是源列表中项的引用。同样，在字典的情况下，键和值是源字典中键和值的引用。

deepcopy()函数使用递归算法来查看每个可变集合中的每个项。

对于具有列表类型的对象，概念算法类似于以下内容：

```py
from typing import Any 

def deepcopy_json(some_obj: Any) -> Any: 

    match some_obj: 

        case int() | float() | tuple() | str() | bytes() | None: 

            return some_obj 

        case list() as some_list: 

            list_copy: list[Any] = [] 

            for item in some_list: 

                list_copy.append(deepcopy_json(item)) 

            return list_copy 

        case dict() as some_dict: 

            dict_copy: dict[Any, Any] = {} 

            for key in some_dict: 

                dict_copy[key] = deepcopy_json(some_dict[key]) 

            return dict_copy 

        case _: 

            raise ValueError(f"can’t copy {type(some_obj)}") 
```

这可以用于 JSON 文档中使用的类型集合。对于第一种情况子句中的不可变类型，没有必要进行复制；这些类型之一的对象不能被修改。对于 JSON 文档中使用的两种可变类型，构建空结构，然后插入每个项的副本。处理涉及递归以确保——无论嵌套多深——所有可变项都被复制。

deepcopy()函数的实际实现处理了 JSON 规范之外的额外类型。这个例子旨在展示深度复制函数的一般思想。

## 5.5.4 参见

+   在理解变量、引用和赋值的配方中，我们探讨了 Python 如何倾向于创建对象的引用。

# 5.6 避免为函数参数使用可变默认值

在第 3 章中，我们探讨了 Python 函数定义的许多方面。在设计具有可选参数的函数的配方中，我们展示了处理可选参数的配方。当时，我们没有深入探讨提供可变结构引用作为默认值的问题。我们将仔细研究函数参数可变默认值的后果。

## 5.6.1 准备工作

让我们想象一个函数，它要么创建，要么更新一个可变的 Counter 对象。我们将称之为 gather_stats()。

理想情况下，一个小型数据收集函数可能看起来像这样：

```py
from collections import Counter 

from random import randint, seed 

def gather_stats_bad( 

    n: int, 

    samples: int = 1000, 

    summary: Counter[int] = Counter() 

) -> Counter[int]: 

    summary.update( 

      sum(randint(1, 6) 

      for d in range(n)) for _ in range(samples) 

    ) 

    return summary
```

这展示了函数的一个糟糕设计。它有两个场景：

1.  第一个场景为摘要参数没有提供任何参数值。当省略此参数时，函数创建并返回一个统计集合。以下是这个故事的例子：

    ```py
    >>> seed(1) 

    >>> s1 = gather_stats_bad(2) 

    >>> s1 

    Counter({7: 168, 6: 147, 8: 136, 9: 114, 5: 110, 10: 77, 11: 71, 4: 70, 3: 52, 12: 29, 2: 26})
    ```

1.  第二个场景允许我们为摘要参数提供一个显式的参数值。当提供此参数时，此函数更新给定的对象。以下是这个故事的例子：

    ```py
    >>> seed(1) 

    >>> mc = Counter() 

    >>> gather_stats_bad(2, summary=mc) 

    Counter... 

    >>> mc 

    Counter({7: 168, 6: 147, 8: 136, 9: 114, 5: 110, 10: 77, 11: 71, 4: 70, 3: 52, 12: 29, 2: 26})
    ```

    我们已经设置了随机数种子以确保两个随机值序列是相同的。我们提供了一个 Counter 对象以确认结果是一致的。

问题出现在我们执行上述第一个场景之后的以下操作：

```py
>>> seed(1) 

>>> s3b = gather_stats_bad(2) 

>>> s3b 

Counter({7: 336, 6: 294, 8: 272, 9: 228, 5: 220, 10: 154, 11: 142, 4: 140, 3: 104, 12: 58, 2: 52})
```

这个例子中的值是不正确的。它们被加倍了。出了点问题。这只有在多次使用默认场景时才会发生。这段代码可以通过一个简单的单元测试套件并看起来是正确的。

正如我们在制作浅拷贝和深拷贝对象的配方中看到的，Python 更喜欢共享引用。共享的一个后果是，由 s1 变量引用的对象和由 s3b 变量引用的对象是同一个对象：

```py
>>> s1 is s3b 

True
```

这意味着当为 s3b 变量的对象创建时，s1 变量引用的对象的值发生了变化。从这个例子中，应该很明显，该函数正在更新一个单一、共享的集合对象，并返回共享集合的引用。

这个 gather_stats_bad() 函数的摘要参数使用的默认值导致结果值由一个单一、共享的对象构建。我们如何避免这种情况？

## 5.6.2 如何操作...

解决可变默认参数问题的有两种方法：

+   提供一个不可变的默认值。

+   改变设计。

我们首先看看不可变默认值。改变设计通常是一个更好的主意。为了看到为什么改变设计更好，我们将展示一个纯粹的技术解决方案。

当我们为函数提供默认值时，默认对象只创建一次，并且之后永远共享。这里有一个替代方案：

1.  将任何可变的默认参数值替换为 None：

    ```py
    def gather_stats_good( 

        n: int, 

        samples: int = 1000, 

        summary: Counter[int] | None = None 

    ) -> Counter[int]:
    ```

    ```py
     def gather_stats_good( 

        n: int, 

        summary: Counter[int] | None = None, 

        samples: int = 1000, 

    ) -> Counter[int]:
    ```

1.  添加一个 if 语句来检查 None 参数值，并用一个全新的、正确的可变对象替换它：

    ```py
     if summary is None: 

            summary = Counter()
    ```

    这将确保每次函数在没有参数值的情况下评估时，我们都会创建一个全新的、可变的对象。我们将避免反复共享单个可变对象。

## 5.6.3 它是如何工作的...

如我们之前所提到的，Python 更喜欢共享引用。它很少在没有显式使用 copy 模块或对象的 copy() 方法的情况下创建对象的副本。因此，函数参数值的默认值将是共享对象。Python 不会为默认参数值创建全新的对象。

永远不要为函数参数的默认值使用可变默认值。

而不是使用可变对象（例如，集合、列表或字典）作为默认值，使用 None。

在大多数情况下，我们应该考虑改变设计，根本不提供默认值。相反，定义两个单独的函数。一个函数更新参数值，另一个函数使用这个函数，但提供一个全新的、空的、可变的对象。

对于这个例子，它们可能被称为 create_stats() 和 update_stats()，具有明确的参数：

```py
def update_stats( 

    n: int, 

    summary: Counter[int], 

    samples: int = 1000, 

) -> Counter[int]: 

    summary.update( 

        sum(randint(1, 6) 

        for d in range(n)) for _ in range(samples)) 

    return summary 

def create_stats(n: int, samples: int = 1000) -> Counter[int]: 

    return update_stats(n, Counter(), samples)
```

注意，update_stats()函数的 summary 参数不是可选的。同样，create_stats()函数也没有定义 summary 对象参数。

可选可变参数的想法并不好，因为作为参数默认值的可变对象被重复使用。

## 5.6.4 更多内容...

在标准库中，有一些示例展示了如何创建新的默认对象的一个酷技术。许多地方使用工厂函数作为参数。这个函数可以用来创建一个全新的可变对象。

为了利用这种设计模式，我们需要修改 update_stats()函数的设计。我们不再在函数中更新现有的 Counter 对象。我们将始终创建一个全新的对象。

这里有一个调用工厂函数来创建有用默认值的函数：

```py
from collections import Counter 

from collections.abc import Callable, Iterable, Hashable 

from typing import TypeVar, TypeAlias 

T = TypeVar(’T’, bound=Hashable) 

Summarizer: TypeAlias = Callable[[Iterable[T]], Counter[T]] 

def gather_stats_flex( 

    n: int, 

    samples: int = 1000, 

    summary_func: Summarizer[int] = Counter 

) -> Counter[int]: 

    summary = summary_func( 

        sum(randint(1, 6) 

        for d in range(n)) for _ in range(samples)) 

    return summary
```

对于这个版本，我们定义了 Summarizer 类型为一个接受一个参数的函数，该函数将创建一个 Counter 对象。默认值使用 Counter 类作为单参数函数。我们可以用任何单参数函数来覆盖 summary_func 函数，该函数将收集细节而不是总结。

这里是一个使用 list 而不是 collections.Counter 的例子：

```py
>>> seed(1) 

>>> gather_stats_flex(2, 12, summary_func=list) 

[7, 4, 5, 8, 10, 3, 5, 8, 6, 10, 9, 7]
```

在这个例子中，我们提供了 list 函数来创建一个包含单个随机样本的列表。

这里是一个不带参数值的例子。每次使用时，它都会创建一个新的 collections.Counter 对象：

```py
>>> seed(1) 

>>> gather_stats_flex(2, 12) 

Counter({7: 2, 5: 2, 8: 2, 10: 2, 4: 1, 3: 1, 6: 1, 9: 1})
```

在这个例子中，我们使用默认的 summary_func 值来评估函数，它从随机样本中创建一个 collections.Counter 对象。

## 5.6.5 参见

+   参见创建字典 – 插入和更新食谱，它展示了 defaultdict 集合的工作方式。

# 加入我们的社区 Discord 空间

加入我们的 Python Discord 工作空间，讨论并了解更多关于这本书的信息：[`packt.link/dHrHU`](https://packt.link/dHrHU)

![图片](img/file1.png)
