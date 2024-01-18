# 容器和数据结构

在本章中，我们将涵盖以下食谱：

+   计数频率-计算任何可散列值的出现次数

+   带有回退的字典-为任何丢失的键设置回退值

+   解包多个-关键字参数-如何多次使用`**`

+   有序字典-保持字典中键的顺序

+   MultiDict-每个键具有多个值的字典

+   优先处理条目-高效获取排序条目的顶部

+   Bunch-表现得像对象的字典

+   枚举-处理已知状态集

# 介绍

Python具有一组非常简单和灵活的内置容器。作为Python开发人员，您几乎可以用`dict`或`list`实现任何功能。Python字典和列表的便利性是如此之大，以至于开发人员经常忘记它们的限制。与任何数据结构一样，它们都经过了优化，并且设计用于特定用例，可能在某些情况下效率低下，甚至无法处理它们。

曾经试图在字典中两次放入一个键吗？好吧，你不能，因为Python字典被设计为具有唯一键的哈希表，但*MultiDict*食谱将向您展示如何做到这一点。曾经试图在不遍历整个列表的情况下从列表中获取最低/最高值吗？列表本身不能，但在*优先处理条目*食谱中，我们将看到如何实现这一点。

标准Python容器的限制对Python专家来说是众所周知的。因此，多年来，标准库已经发展出了克服这些限制的方法，经常有一些模式是如此常见，以至于它们的名称被广泛认可，即使它们没有正式定义。

# 计数频率

在许多类型的程序中，一个非常常见的需求是计算值或事件的出现次数，这意味着计数频率。无论是需要计算文本中的单词，博客文章上的点赞次数，还是跟踪视频游戏玩家的得分，最终计数频率意味着计算特定值的数量。

对于这种需求，最明显的解决方案是保留我们需要计数的计数器。如果有两个、三个或四个，也许我们可以在一些专用变量中跟踪它们，但如果有数百个，保留这么多变量显然是不可行的，我们很快就会得到一个基于容器的解决方案来收集所有这些计数器。

# 如何做到...

以下是此食谱的步骤：

1.  假设我们想要跟踪文本中单词的频率；标准库来拯救我们，并为我们提供了一种非常好的跟踪计数和频率的方法，即通过专用的`collections.Counter`对象。

1.  `collections.Counter`对象不仅跟踪频率，还提供了一些专用方法来检索最常见的条目，至少出现一次的条目，并快速计算任何可迭代对象。

1.  您提供给`Counter`的任何可迭代对象都将被“计数”其值的频率：

```py
>>> txt = "This is a vast world you can't traverse world in a day"
>>>
>>> from collections import Counter
>>> counts = Counter(txt.split())
```

1.  结果将会正是我们所期望的，即我们短语中单词的频率字典：

```py
Counter({'a': 2, 'world': 2, "can't": 1, 'day': 1, 'traverse': 1, 
         'is': 1, 'vast': 1, 'in': 1, 'you': 1, 'This': 1})
```

1.  然后，我们可以轻松查询最常见的单词：

```py
>>> counts.most_common(2)
[('world', 2), ('a', 2)]
```

1.  获取特定单词的频率：

```py
>>> counts['world']
2
```

或者，获取总出现次数：

```py
>>> sum(counts.values())
12
```

1.  我们甚至可以对计数器应用一些集合操作，例如合并它们，减去它们，或检查它们的交集：

```py
>>> Counter(["hello", "world"]) + Counter(["hello", "you"])
Counter({'hello': 2, 'you': 1, 'world': 1})
>>> Counter(["hello", "world"]) & Counter(["hello", "you"])
Counter({'hello': 1})
```

# 它是如何工作的...

我们的计数代码依赖于`Counter`只是一种特殊类型的字典，字典可以通过提供一个可迭代对象来构建。可迭代对象中的每个条目都将添加到字典中。

在计数器的情况下，添加一个元素意味着增加其计数；对于我们列表中的每个“单词”，我们会多次添加该单词（每次它在列表中出现一次），因此它在`Counter`中的值每次遇到该单词时都会继续增加。

# 还有更多...

依赖`Counter`实际上并不是跟踪频率的唯一方法；我们已经知道`Counter`是一种特殊类型的字典，因此复制`Counter`的行为应该是非常简单的。

我们每个人可能都会得到这种形式的字典：

```py
counts = dict(hello=0, world=0, nice=0, day=0)
```

每当我们遇到`hello`、`world`、`nice`或`day`的新出现时，我们就会增加字典中关联的值，并称之为一天：

```py
for word in 'hello world this is a very nice day'.split():
    if word in counts:
        counts[word] += 1
```

通过依赖`dict.get`，我们也可以很容易地使其适应计算任何单词，而不仅仅是我们可以预见的那些：

```py
for word in 'hello world this is a very nice day'.split():
    counts[word] = counts.get(word, 0) + 1
```

但标准库实际上提供了一个非常灵活的工具，我们可以使用它来进一步改进这段代码，那就是`collections.defaultdict`。

`defaultdict`是一个普通的字典，对于任何缺失的值都不会抛出`KeyError`，而是调用我们可以提供的函数来生成缺失的值。

因此，诸如`defaultdict(int)`这样的东西将创建一个字典，为任何它没有的键提供`0`，这对我们的计数目的非常方便：

```py
from collections import defaultdict

counts = defaultdict(int)
for word in 'hello world this is a very nice day'.split():
    counts[word] += 1
```

结果将会完全符合我们的期望：

```py
defaultdict(<class 'int'>, {'day': 1, 'is': 1, 'a': 1, 'very': 1, 'world': 1, 'this': 1, 'nice': 1, 'hello': 1})
```

对于每个单词，第一次遇到它时，我们将调用`int`来获得起始值，然后加`1`。由于`int`在没有任何参数的情况下调用时会返回`0`，这就实现了我们想要的效果。

虽然这大致解决了我们的问题，但对于计数来说远非完整解决方案——我们跟踪频率，但在其他方面，我们是自己的。如果我们想知道我们的词袋中最常见的词是什么呢？

`Counter`的便利性基于其提供的一组专门用于计数的附加功能；它不仅仅是一个具有默认数值的字典，它是一个专门用于跟踪频率并提供方便的访问方式的类。

# 带有回退的字典

在处理配置值时，通常会在多个地方查找它们——也许我们从配置文件中加载它们——但我们可以用环境变量或命令行选项覆盖它们，如果没有提供选项，我们可以有一个默认值。

这很容易导致像这样的长链的`if`语句：

```py
value = command_line_options.get('optname')
if value is None:
    value = os.environ.get('optname')
if value is None:
    value = config_file_options.get('optname')
if value is None:
    value = 'default-value'
```

这很烦人，而对于单个值来说可能只是烦人，但随着添加更多选项，它将变成一个庞大、令人困惑的条件列表。

命令行选项是一个非常常见的用例，但问题与链式作用域解析有关。在Python中，变量是通过查看`locals()`来解析的；如果找不到它们，解释器会查看`globals()`，如果还找不到，它会查找内置变量。

# 如何做到...

对于这一步，您需要按照以下步骤进行：

1.  与使用多个`if`实例相比，`dict.get`的默认值链的替代方案可能并不会改进代码太多，如果我们想要添加一个额外的作用域，我们将不得不在每个查找值的地方都添加它。

1.  `collections.ChainMap`是这个问题的一个非常方便的解决方案；我们可以提供一个映射容器的列表，它将在它们所有中查找一个键。

1.  我们之前的涉及多个不同`if`实例的示例可以转换为这样的形式：

```py
import os
from collections import ChainMap

options = ChainMap(command_line_options, os.environ, config_file_options)
value = options.get('optname', 'default-value')
```

1.  我们还可以通过将`ChainMap`与`defaultdict`结合来摆脱最后的`.get`调用。在这种情况下，我们可以使用`defaultdict`为每个键提供一个默认值：

```py
import os
from collections import ChainMap, defaultdict

options = ChainMap(command_line_options, os.environ, config_file_options,
                   defaultdict(lambda: 'default-value'))
value = options['optname']
value2 = options['other-option']
```

1.  打印`value`和`value2`将会得到以下结果：

```py
optvalue
default-value
```

`optname`将从包含它的`command_line_options`中检索，而`other-option`最终将由`defaultdict`解析。

# 它是如何工作的...

`ChainMap`类接收多个字典作为参数；每当向`ChainMap`请求一个键时，它实际上会逐个查看提供的字典，以检查该键是否在其中任何一个中可用。一旦找到键，它就会返回，就好像它是`ChainMap`自己拥有的键一样。

未提供的选项的默认值是通过将`defaultdict`作为提供给`ChainMap`的最后一个字典来实现的。每当在之前的任何字典中找不到键时，它会在`defaultdict`中查找，`defaultdict`使用提供的工厂函数为所有键返回默认值。

# 还有更多...

`ChainMap`的另一个很棒的功能是它也允许更新，但是它总是更新第一个字典，而不是更新找到键的字典。结果是一样的，因为在下一次查找该键时，我们会发现第一个字典覆盖了该键的任何其他值（因为它是检查该键的第一个地方）。优点是，如果我们将空字典作为提供给`ChainMap`的第一个映射，我们可以更改这些值而不触及原始容器：

```py
>>> population=dict(italy=60, japan=127, uk=65) >>> changes = dict()
>>> editablepop = ChainMap(changes, population)

>>> print(editablepop['japan'])
127
>>> editablepop['japan'] += 1
>>> print(editablepop['japan'])
128
```

但即使我们将日本的人口更改为1.28亿，原始人口也没有改变：

```py
>>> print(population['japan'])
127
```

我们甚至可以使用`changes`来找出哪些值被更改了，哪些值没有被更改：

```py
>>> print(changes.keys()) 
dict_keys(['japan']) 
>>> print(population.keys() - changes.keys()) 
{'italy', 'uk'}
```

顺便说一句，如果字典中包含的对象是可变的，并且我们直接对其进行改变，`ChainMap`无法避免改变原始对象。因此，如果我们在字典中存储的不是数字，而是列表，每当我们向字典追加值时，我们将改变原始字典：

```py
>>> citizens = dict(torino=['Alessandro'], amsterdam=['Bert'], raleigh=['Joseph']) >>> changes = dict() 
>>> editablecits = ChainMap(changes, citizens) 
>>> editablecits['torino'].append('Simone') 
>>> print(editablecits['torino']) ['Alessandro', 'Simone']
>>> print(changes)
{}
>>> print(citizens)
{'amsterdam': ['Bert'], 
 'torino': ['Alessandro', 'Simone'], 
 'raleigh': ['Joseph']} 
```

# 解包多个关键字参数

经常情况下，你会发现自己需要从字典中向函数提供参数。如果你曾经面临过这种需求，你可能也会发现自己需要从多个字典中获取参数。

通常，Python函数通过解包（`**`语法）从字典中接受参数，但到目前为止，在同一次调用中两次解包还不可能，也没有简单的方法来合并两个字典。

# 如何做...

这个食谱的步骤是：

1.  给定一个函数`f`，我们希望按以下方式从两个字典`d1`和`d2`传递参数：

```py
>>> def f(a, b, c, d):
...     print (a, b, c, d)
...
>>> d1 = dict(a=5, b=6)
>>> d2 = dict(b=7, c=8, d=9)
```

1.  `collections.ChainMap`可以帮助我们实现我们想要的；它可以处理重复的条目，并且适用于任何Python版本：

```py
>>> f(**ChainMap(d1, d2))
5 6 8 9
```

1.  在Python 3.5及更新版本中，你还可以通过字面语法组合多个字典来创建一个新字典，然后将结果字典作为函数的参数传递：

```py
>>> f(**{**d1, **d2})
5 7 8 9
```

1.  在这种情况下，重复的条目也被接受，但按照`ChainMap`的优先级的相反顺序处理（从右到左）。请注意，`b`的值为`7`，而不是`ChainMap`中的`6`，这是由于优先级的反向顺序造成的。

由于涉及到大量的解包运算符，这种语法可能更难阅读，而使用`ChainMap`对于读者来说可能更加明确发生了什么。

# 它是如何工作的...

正如我们已经从之前的示例中知道的那样，`ChainMap`在所有提供的字典中查找键，因此它就像所有字典的总和。解包运算符（`**`）通过将所有键放入容器，然后为每个键提供一个参数来工作。

由于`ChainMap`具有所有提供的字典键的总和，它将提供包含在所有字典中的键给解包运算符，从而允许我们从多个字典中提供关键字参数。

# 还有更多...

自Python 3.5通过PEP 448，现在可以解包多个映射以提供关键字参数：

```py
>>> def f(a, b, c, d):
...     print (a, b, c, d)
...
>>> d1 = dict(a=5, b=6)
>>> d2 = dict(c=7, d=8)
>>> f(**d1, **d2)
5 6 7 8
```

这种解决方案非常方便，但有两个限制：

+   仅适用于Python 3.5+

+   它无法处理重复的参数

如果你不知道你要解包的映射/字典来自哪里，很容易出现重复参数的问题：

```py
>>> d1 = dict(a=5, b=6)
>>> d2 = dict(b=7, c=8, d=9)
>>> f(**d1, **d2)
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: f() got multiple values for keyword argument 'b'
```

在前面的示例中，`b`键在`d1`和`d2`中都有声明，这导致函数抱怨它收到了重复的参数。

# 有序字典

对于新用户来说，Python字典最令人惊讶的一个方面是，它们的顺序是不可预测的，而且在不同的环境中可能会发生变化。因此，您在自己的系统上期望的键的顺序可能在朋友的计算机上完全不同。

这经常会在测试期间导致意外的失败；如果涉及到持续集成系统，则运行测试的系统上的字典键的排序可能与您的系统上的排序不同，这可能导致随机失败。

假设您有一小段代码，它生成了一个带有一些属性的HTML标签：

```py
>>> attrs = dict(style="background-color:red", id="header")
>>> '<span {}>'.format(' '.join('%s="%s"' % a for a in attrs.items()))
'<span id="header" style="background-color:red">'
```

也许会让你感到惊讶的是，在某些系统上，你最终会得到这样的结果：

```py
'<span id="header" style="background-color:red">'
```

而在其他情况下，结果可能是这样的：

```py
'<span style="background-color:red" id="header">'
```

因此，如果您期望能够比较生成的字符串，以检查您的函数在生成此标签时是否做对了，您可能会感到失望。

# 如何做到这一点...

键的排序是一个非常方便的功能，在某些情况下，它实际上是必需的，因此Python标准库提供了`collections.OrderedDict`容器。

在`collections.OrderedDict`的情况下，键始终按插入的顺序排列：

```py
>>> attrs = OrderedDict([('id', 'header'), ('style', 'background-color:red')])
>>> '<span {}>'.format(' '.join('%s="%s"' % a for a in attrs.items()))
'<span id="header" style="background-color:red">'
```

# 它是如何工作的...

`OrderedDict`同时存储键到值的映射和一个用于保留它们顺序的键列表。

因此，每当您查找键时，查找都会通过映射进行，但每当您想要列出键或对容器进行迭代时，您都会通过键列表来确保它们按照插入的顺序进行处理。

使用`OrderedDict`的主要问题是，Python在3.6之前的版本中没有保证关键字参数的任何特定顺序：

```py
>>> attrs = OrderedDict(id="header", style="background-color:red")
```

即使使用了`OrderedDict`，这将再次引入完全随机的键顺序。这不是因为`OrderedDict`没有保留这些键的顺序，而是因为它们可能以随机顺序接收到。

由于PEP 468的原因，现在在Python 3.6和更新版本中保证了参数的顺序（字典的顺序仍然不确定；请记住，它们是有序的只是偶然的）。因此，如果您使用的是Python 3.6或更新版本，我们之前的示例将按预期工作，但如果您使用的是较旧版本的Python，您将得到一个随机的顺序。

幸运的是，这是一个很容易解决的问题。与标准字典一样，`OrderedDict`支持任何可迭代的内容作为其内容的来源。只要可迭代对象提供了一个键和一个值，就可以用它来构建`OrderedDict`。

因此，通过在元组中提供键和值，我们可以在任何Python版本中在构建时提供它们并保留顺序：

```py
>>> OrderedDict((('id', 'header'), ('style', 'background-color:red')))
OrderedDict([('id', 'header'), ('style', 'background-color:red')])
```

# 还有更多...

Python 3.6引入了保留字典键顺序的保证，作为对字典的一些更改的副作用，但它被认为是一个内部实现细节，而不是语言保证。自Python 3.7以来，它成为语言的一个官方特性，因此如果您使用的是Python 3.6或更新版本，可以放心地依赖于字典的顺序。

# MultiDict

如果您曾经需要提供一个反向映射，您可能已经发现Python缺乏一种方法来为字典中的每个键存储多个值。这是一个非常常见的需求，大多数语言都提供了某种形式的多映射容器。

Python倾向于有一种单一的做事方式，因为为键存储多个值意味着只是为键存储一个值列表，所以它不提供专门的容器。

存储值列表的问题在于，为了能够将值附加到我们的字典中，列表必须已经存在。

# 如何做到这一点...

按照以下步骤进行此操作：

1.  正如我们已经知道的，`defaultdict`将通过调用提供的可调用函数为每个缺失的键创建一个默认值。我们可以将`list`构造函数作为可调用函数提供：

```py
>>> from collections import defaultdict
>>> rd = defaultdict(list)
```

1.  因此，我们通过使用`rd[k].append(v)`而不是通常的`rd[k] = v`来将键插入到我们的多映射中：

```py
>>> for name, num in [('ichi', 1), ('one', 1), ('uno', 1), ('un', 1)]:
...   rd[num].append(name)
...
>>> rd
defaultdict(<class 'list'>, {1: ['ichi', 'one', 'uno', 'un']})
```

# 它是如何工作的...

`MultiDict`通过为每个键存储一个列表来工作。每当访问一个键时，都会检索包含该键所有值的列表。

在缺少键的情况下，将提供一个空列表，以便为该键添加值。

这是因为每次`defaultdict`遇到缺少的键时，它将插入一个由调用`list`生成的值。调用`list`实际上会提供一个空列表。因此，执行`rd[v]`将始终提供一个列表，取决于`v`是否是已经存在的键。一旦我们有了列表，添加新值只是追加它的问题。

# 还有更多...

Python中的字典是关联容器，其中键是唯一的。一个键只能出现一次，且只有一个值。

如果我们想要支持每个键多个值，实际上可以通过将`list`保存为键的值来满足需求。然后，该列表可以包含我们想要保留的所有值：

```py
>>> rd = {1: ['one', 'uno', 'un', 'ichi'],
...       2: ['two', 'due', 'deux', 'ni'],
...       3: ['three', 'tre', 'trois', 'san']}
>>> rd[2]
['two', 'due', 'deux', 'ni']
```

如果我们想要为`2`（例如西班牙语）添加新的翻译，我们只需追加该条目：

```py
>>> rd[2].append('dos')
>>> rd[2]
['two', 'due', 'deux', 'ni', 'dos']
```

当我们想要引入一个新的键时，问题就出现了：

```py
>>> rd[4].append('four')
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
KeyError: 4
```

对于键`4`，没有列表存在，因此我们无法追加它。因此，我们的自动反向映射片段无法轻松适应处理多个值，因为它在尝试插入值时会出现键错误：

```py
>>> rd = {}
>>> for k,v in d.items():
...     rd[v].append(k)
Traceback (most recent call last):
    File "<stdin>", line 2, in <module>
KeyError: 1
```

检查每个条目是否已经在字典中，然后根据情况采取行动并不是非常方便。虽然我们可以依赖字典的`setdefault`方法来隐藏该检查，但是通过使用`collections.defaultdict`可以获得更加优雅的解决方案。

# 优先处理条目

选择一组值的第一个/顶部条目是一个非常频繁的需求；这通常意味着定义一个优先于其他值的值，并涉及排序。

但是排序可能很昂贵，并且每次添加条目到您的值时重新排序肯定不是一种非常方便的方式来从一组具有某种优先级的值中选择第一个条目。

# 如何做...

堆是一切具有优先级的完美匹配，例如优先级队列：

```py
import time
import heapq

class PriorityQueue:
    def __init__(self):
        self._q = []

    def add(self, value, priority=0):
        heapq.heappush(self._q, (priority, time.time(), value))

    def pop(self):
        return heapq.heappop(self._q)[-1]
```

然后，我们的`PriorityQueue`可以用于检索给定优先级的条目：

```py
>>> def f1(): print('hello')
>>> def f2(): print('world')
>>>
>>> pq = PriorityQueue()
>>> pq.add(f2, priority=1)
>>> pq.add(f1, priority=0)
>>> pq.pop()()
hello
>>> pq.pop()()
world
```

# 它是如何工作的...

`PriorityQueue`通过在堆中存储所有内容来工作。堆在检索排序集的顶部/第一个元素时特别高效，而无需实际对整个集进行排序。

我们的优先级队列将所有值存储在一个三元组中：`priority`，`time.time()`和`value`。

我们元组的第一个条目是`priority`（较低的优先级更好）。在示例中，我们记录了`f1`的优先级比`f2`更好，这确保了当我们使用`heap.heappop`获取要处理的任务时，我们首先得到`f1`，然后是`f2`，这样我们最终得到的是`hello world`消息而不是`world hello`。

第二个条目`timestamp`用于确保具有相同优先级的任务按其插入顺序进行处理。最旧的任务将首先被处理，因为它将具有最小的时间戳。

然后，我们有值本身，这是我们要为任务调用的函数。

# 还有更多...

对于排序的一个非常常见的方法是将条目列表保存在一个元组中，其中第一个元素是我们正在排序的`key`，第二个元素是值本身。

对于记分牌，我们可以保留每个玩家的姓名和他们得到的分数：

```py
scores = [(123, 'Alessandro'),
          (143, 'Chris'),
          (192, 'Mark']
```

将这些值存储在元组中有效，因为比较两个元组是通过将第一个元组的每个元素与另一个元组中相同索引位置的元素进行比较来执行的：

```py
>>> (10, 'B') > (10, 'A')
True
>>> (11, 'A') > (10, 'B')
True
```

如果您考虑字符串，就可以很容易地理解发生了什么。`'BB' > 'BB'`与`('B', 'B') > ('B', 'A')`相同；最终，字符串只是字符列表。

我们可以利用这个属性对我们的`scores`进行排序，并检索比赛的获胜者：

```py
>>> scores = sorted(scores)
>>> scores[-1]
(192, 'Mark')
```

这种方法的主要问题是，每次我们向列表添加条目时，我们都必须重新对其进行排序，否则我们的计分板将变得毫无意义：

```py
>>> scores.append((137, 'Rick'))
>>> scores[-1]
(137, 'Rick')
>>> scores = sorted(scores)
>>> scores[-1]
(192, 'Mark')
```

这很不方便，因为如果我们有多个地方向列表添加元素，很容易错过重新排序的地方，而且每次对整个列表进行排序可能会很昂贵。

Python标准库提供了一种数据结构，当我们想要找出比赛的获胜者时，它是完美的匹配。

在`heapq`模块中，我们有一个完全工作的堆数据结构的实现，这是一种特殊类型的树，其中每个父节点都小于其子节点。这为我们提供了一个具有非常有趣属性的树：根元素始终是最小的。

并且它是建立在列表之上的，这意味着`l[0]`始终是`heap`中最小的元素：

```py
>>> import heapq
>>> l = []
>>> heapq.heappush(l, (192, 'Mark'))
>>> heapq.heappush(l, (123, 'Alessandro'))
>>> heapq.heappush(l, (137, 'Rick'))
>>> heapq.heappush(l, (143, 'Chris'))
>>> l[0]
(123, 'Alessandro')
```

顺便说一句，您可能已经注意到，堆找到了我们比赛的失败者，而不是获胜者，而我们对找到最好的玩家，即最高价值的玩家感兴趣。

这是一个我们可以通过将所有分数存储为负数来轻松解决的小问题。如果我们将每个分数存储为`* -1`，那么堆的头部将始终是获胜者：

```py
>>> l = []
>>> heapq.heappush(l, (-143, 'Chris'))
>>> heapq.heappush(l, (-137, 'Rick'))
>>> heapq.heappush(l, (-123, 'Alessandro'))
>>> heapq.heappush(l, (-192, 'Mark'))
>>> l[0]
(-192, 'Mark')
```

# Bunch

Python非常擅长变形对象。每个实例都可以有自己的属性，并且在运行时添加/删除对象的属性是完全合法的。

偶尔，我们的代码需要处理未知形状的数据。例如，在用户提交的数据的情况下，我们可能不知道用户提供了哪些字段；也许我们的一些用户有名字，一些有姓氏，一些有一个或多个中间名字段。

如果我们不是自己处理这些数据，而只是将其提供给其他函数，我们实际上并不关心数据的形状；只要我们的对象具有这些属性，我们就没问题。

一个非常常见的情况是在处理协议时，如果您是一个HTTP服务器，您可能希望向您后面运行的应用程序提供一个`request`对象。这个对象有一些已知的属性，比如`host`和`path`，还可能有一些可选的属性，比如`query`字符串或`content`类型。但是，它也可以有客户端提供的任何属性，因为HTTP在头部方面非常灵活，我们的客户端可能提供了一个`x-totally-custom-header`，我们可能需要将其暴露给我们的代码。

在表示这种类型的数据时，Python开发人员通常倾向于查看字典。最终，Python对象本身是建立在字典之上的，并且它们符合将任意值映射到名称的需求。

因此，我们可能最终会得到以下内容：

```py
>>> request = dict(host='www.example.org', path='/index.html')
```

这种方法的一个副作用在于，一旦我们不得不将这个对象传递给其他代码，特别是第三方代码时，就变得非常明显。函数通常使用对象工作，虽然它们不需要特定类型的对象，因为鸭子类型是Python中的标准，但它们会期望某些属性存在。

另一个非常常见的例子是在编写测试时，Python作为一种鸭子类型的语言，希望提供一个假对象而不是提供对象的真实实例是绝对合理的，特别是当我们需要模拟一些属性的值（如使用`@property`声明），因此我们不希望或无法创建对象的真实实例。

在这种情况下，使用字典是不可行的，因为它只能通过`request['path']`语法访问其值，而不能通过`request.path`访问，这可能是我们提供对象给函数时所期望的。

此外，我们访问这个值的次数越多，就越清楚使用点符号表示法传达了代码意图的实体协作的感觉，而字典传达了纯粹数据的感觉。

一旦我们记住Python对象可以随时改变形状，我们可能会尝试创建一个对象而不是字典。不幸的是，我们无法在初始化时提供属性：

```py
>>> request = object(host='www.example.org', path='/index.html')
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
TypeError: object() takes no parameters
```

如果我们尝试在构建对象后分配这些属性，情况也不会有所改善：

```py
>>> request = object()
>>> request.host = 'www.example.org'
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
AttributeError: 'object' object has no attribute 'host'
```

# 如何做...

通过一点努力，我们可以创建一个利用字典来包含我们想要的任何属性并允许通过属性和字典访问的类：

```py
>>> class Bunch(dict):
...    def __getattribute__(self, key):
...        try: 
...            return self[key]
...        except KeyError:
...            raise AttributeError(key)
...    
...    def __setattr__(self, key, value): 
...        self[key] = value
...
>>> b = Bunch(a=5)
>>> b.a
5
>>> b['a']
5
```

# 它是如何工作的...

`Bunch`类继承自`dict`，主要是为了提供一个值可以被存储的上下文，然后大部分工作由`__getattribute__`和`__setattr__`完成。因此，对于在对象上检索或设置的任何属性，它们只会检索或设置`self`中的一个键（记住我们继承自`dict`，所以`self`实际上是一个字典）。

这使得`Bunch`类能够将任何值存储和检索为对象的属性。方便的特性是它在大多数情况下既可以作为对象又可以作为`dict`来使用。

例如，可以找出它包含的所有值，就像任何其他字典一样：

```py
>>> b.items()
dict_items([('a', 5)])
```

它还能够将它们作为属性访问：

```py
>>> b.c = 7
>>> b.c
7
>>> b.items()
dict_items([('a', 5), ('c', 7)])
```

# 还有更多...

我们的`bunch`实现还不完整，因为它将无法通过任何类名称测试（它总是被命名为`Bunch`），也无法通过任何继承测试，因此无法伪造其他对象。

第一步是使`Bunch`能够改变其属性，还能改变其名称。这可以通过每次创建`Bunch`时动态创建一个新类来实现。该类将继承自`Bunch`，除了提供一个新名称外不会做任何其他事情：

```py
>>> class BunchBase(dict):
...    def __getattribute__(self, key):
...        try: 
...            return self[key]
...        except KeyError:
...            raise AttributeError(key)
...    
...    def __setattr__(self, key, value): 
...        self[key] = value
...
>>> def Bunch(_classname="Bunch", **attrs):
...     return type(_classname, (BunchBase, ), {})(**attrs)
>>>
```

`Bunch`函数从原来的类本身变成了一个工厂，将创建所有作为`Bunch`的对象，但可以有不同的类。每个`Bunch`将是`BunchBase`的子类，其中在创建`Bunch`时可以提供`_classname`名称：

```py
>>> b = Bunch("Request", path="/index.html", host="www.example.org")
>>> print(b)
{'path': '/index.html', 'host': 'www.example.org'}
>>> print(b.path)
/index.html
>>> print(b.host)
www.example.org
```

这将允许我们创建任意类型的`Bunch`对象，并且每个对象都将有自己的自定义类型：

```py
>>> print(b.__class__)
<class '__main__.Request'>
```

下一步是使我们的`Bunch`实际上看起来像它必须模仿的任何其他类型。这对于我们想要在另一个对象的位置使用`Bunch`的情况是必要的。由于`Bunch`可以具有任何类型的属性，因此它可以代替任何类型的对象，但为了能够这样做，它必须通过自定义类型的类型检查。

我们需要回到我们的`Bunch`工厂，并使`Bunch`对象不仅具有自定义类名，还要看起来是从自定义父类继承而来。

为了更好地理解发生了什么，我们将声明一个示例`Person`类型；这个类型将是我们的`Bunch`对象尝试伪造的类型：

```py
class Person(object):
    def __init__(name, surname):
        self.name = name
        self.surname = surname

    @property
    def fullname(self):
        return '{} {}'.format(self.name, self.surname)
```

具体来说，我们将通过一个自定义的`print`函数打印`Hello Your Name`，该函数仅适用于`Person`：

```py
def hello(p):
    if not isinstance(p, Person):
        raise ValueError("Sorry, can only greet people")
    print("Hello {}".format(p.fullname))
```

我们希望改变我们的`Bunch`工厂，接受该类并创建一个新类型：

```py
def Bunch(_classname="Bunch", _parent=None, **attrs):
    parents = (_parent, ) if parent else tuple()
    return type(_classname, (BunchBase, ) + parents, {})(**attrs)
```

现在，我们的`Bunch`对象将显示为我们想要的类的实例，并且始终显示为`_parent`的子类：

```py
>>> p = Bunch("Person", Person, fullname='Alessandro Molina')
>>> hello(p)
Hello Alessandro Molina
```

`Bunch`可以是一种非常方便的模式；在其完整和简化版本中，它被广泛用于许多框架中，具有各种实现，但都可以实现几乎相同的结果。

展示的实现很有趣，因为它让我们清楚地知道发生了什么。有一些非常聪明的方法可以实现`Bunch`，但可能会让人难以猜测发生了什么并进行自定义。

实现`Bunch`模式的另一种可能的方法是通过修补包含类的所有属性的`__dict__`类：

```py
class Bunch(dict):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.__dict__ = self
```

在这种形式下，每当创建`Bunch`时，它将以`dict`的形式填充其值（通过调用`super().__init__`，这是`dict`的初始化），然后，一旦所有提供的属性都存储在`dict`中，它就会用`self`交换`__dict__`对象，这是包含所有对象属性的字典。这使得刚刚填充了所有值的`dict`也成为了包含对象所有属性的`dict`。

我们之前的实现是通过替换我们查找属性的方式来工作的，而这个实现是替换我们查找属性的地方。

# 枚举

枚举是存储只能表示几种状态的值的常见方式。每个符号名称都绑定到一个特定的值，通常是数字，表示枚举可以具有的状态。

枚举在其他编程语言中非常常见，但直到最近，Python才没有对枚举提供明确的支持。

# 如何做到...

通常，枚举是通过将符号名称映射到数值来实现的；在Python中，通过`enum.IntEnum`是允许的：

```py
>>> from enum import IntEnum
>>> 
>>> class RequestType(IntEnum):
...     POST = 1
...     GET = 2
>>>
>>> request_type = RequestType.POST
>>> print(request_type)
RequestType.POST
```

# 它是如何工作的...

`IntEnum`是一个整数，除了在类定义时创建所有可能的值。`IntEnum`继承自`int`，因此它的值是真正的整数。

在`RequestType`的定义过程中，所有`enum`的可能值都在类体内声明，并且这些值通过元类进行重复验证。

此外，`enum`提供了对特殊值`auto`的支持，它的意思是*只是放一个值进去，我不在乎*。通常你只关心它是`POST`还是`GET`，你通常不关心`POST`是`1`还是`2`。

最后但并非最不重要的是，如果枚举定义了至少一个可能的值，那么枚举就不能被子类化。

# 还有更多...

`IntEnum`的值在大多数情况下表现得像`int`，这通常很方便，但如果开发人员不注意类型，它们可能会引起问题。

例如，如果提供了另一个枚举或整数值，而不是正确的枚举值，函数可能会意外执行错误的操作：

```py
>>> def do_request(kind):
...    if kind == RequestType.POST:
...        print('POST')
...    else:
...        print('OTHER')
```

例如，使用`RequestType.POST`或`1`调用`do_request`将做完全相同的事情：

```py
>>> do_request(RequestType.POST)
POST
>>> do_request(1)
POST
```

当我们不想将枚举视为数字时，可以使用`enum.Enum`，它提供了不被视为普通数字的枚举值：

```py
>>> from enum import Enum
>>> 
>>> class RequestType(Enum):
...     POST = 1
...     GET = 2
>>>
>>> do_request(RequestType.POST)
POST
>>> do_request(1)
OTHER
```

因此，一般来说，如果你需要一个简单的枚举值集合或依赖于`enum`的可能状态，`Enum`更安全，但如果你需要依赖于`enum`的一组数值，`IntEnum`将确保它们表现得像数字。
