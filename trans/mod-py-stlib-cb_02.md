# 文本管理

在本章中，我们将涵盖以下配方：

+   模式匹配-正则表达式不是解析模式的唯一方法；Python提供了更简单且同样强大的工具来解析模式

+   文本相似性-检测两个相似字符串的性能可能很困难，但Python有一些易于使用的内置工具

+   文本建议-Python寻找最相似的一个建议给用户正确的拼写

+   模板化-在生成文本时，模板化是定义规则的最简单方法

+   保留空格拆分字符串-在空格上拆分可能很容易，但当您想保留一些空格时会变得更加困难

+   清理文本-从文本中删除任何标点符号或奇怪的字符

+   文本标准化-在处理国际文本时，通常方便避免处理特殊字符和单词拼写错误

+   对齐文本-在输出文本时，正确对齐文本大大增加了可读性

# 介绍

Python是为系统工程而生的，当与shell脚本和基于shell的软件一起工作时，经常需要创建和解析文本。这就是为什么Python有非常强大的工具来处理文本。

# 模式匹配

在文本中寻找模式时，正则表达式通常是解决这类问题的最常见方式。它们非常灵活和强大，尽管它们不能表达所有种类的语法，但它们通常可以处理大多数常见情况。

正则表达式的强大之处在于它们可以生成的广泛符号和表达式集。问题在于，对于不习惯正则表达式的开发人员来说，它们可能看起来就像纯噪音，即使有经验的人也经常需要花一点时间才能理解下面的表达式：

```py
"^(*d{3})*( |-)*d{3}( |-)*d{4}$"
```

这个表达式实际上试图检测电话号码。

对于大多数常见情况，开发人员需要寻找非常简单的模式：例如，文件扩展名（它是否以`.txt`结尾？），分隔文本等等。

# 如何做...

`fnmatch`模块提供了一个简化的模式匹配语言，对于大多数开发人员来说，语法非常快速和易于理解。

很少有字符具有特殊含义：

+   `*`表示任何文本

+   `?`表示任何字符

+   `[...]`表示方括号内包含的字符

+   `[!...]`表示除了方括号内包含的字符之外的所有内容

您可能会从系统shell中认出这个语法，所以很容易看出`*.txt`意味着*每个具有.txt扩展名的名称*：

```py
>>> fnmatch.fnmatch('hello.txt', '*.txt')
True
>>> fnmatch.fnmatch('hello.zip', '*.txt')
False
```

# 还有更多...

实际上，`fnmatch`可以用于识别由某种常量值分隔的文本片段。

例如，如果我有一个模式，定义了变量的`类型`，`名称`和`值`，通过`:`分隔，我们可以通过`fnmatch`识别它，然后声明所描述的变量：

```py
>>> def declare(decl):
...   if not fnmatch.fnmatch(decl, '*:*:*'):
...     return False
...   t, n, v = decl.split(':', 2)
...   globals()[n] = getattr(__builtins__, t)(v)
...   return True
... 
>>> declare('int:somenum:3')
True
>>> somenum
3
>>> declare('bool:somebool:True')
True
>>> somebool
True
>>> declare('int:a')
False
```

显然，`fnmatch`在文件名方面表现出色。如果您有一个文件列表，很容易提取只匹配特定模式的文件：

```py
>>> os.listdir()
['.git', '.gitignore', '.vscode', 'algorithms.rst', 'concurrency.rst', 
 'conf.py', 'crypto.rst', 'datastructures.rst', 'datetimes.rst', 
 'devtools.rst', 'filesdirs.rst', 'gui.rst', 'index.rst', 'io.rst', 
 'make.bat', 'Makefile', 'multimedia.rst', 'networking.rst', 
 'requirements.txt', 'terminal.rst', 'text.rst', 'venv', 'web.rst']
>>> fnmatch.filter(os.listdir(), '*.git*')
['.git', '.gitignore']
```

虽然非常方便，`fnmatch`显然是有限的，但当一个工具达到其极限时，最好的事情之一就是提供与可以克服这些限制的替代工具兼容的兼容性。

例如，如果我想找到所有包含单词`git`或`vs`的文件，我不能在一个`fnmatch`模式中做到这一点。我必须声明两种不同的模式，然后将结果连接起来。但是，如果我可以使用正则表达式，那是绝对可能的。

`fnmatch.translate`在`fnmatch`模式和正则表达式之间建立桥梁，提供描述`fnmatch`模式的正则表达式，以便可以根据需要进行扩展。

例如，我们可以创建一个匹配这两种模式的正则表达式：

```py
>>> reg = '({})|({})'.format(fnmatch.translate('*.git*'), 
                             fnmatch.translate('*vs*'))
>>> reg
'(.*\.git.*\Z(?ms))|(.*vs.*\Z(?ms))'
>>> import re
>>> [s for s in os.listdir() if re.match(reg, s)]
['.git', '.gitignore', '.vscode']
```

`fnmatch`的真正优势在于它是一种足够简单和安全的语言，可以向用户公开。假设您正在编写一个电子邮件客户端，并且希望提供搜索功能，如果您有来自Jane Smith和Smith Lincoln的电子邮件，您如何让用户搜索名为Smith或姓为Smith的人？

使用`fnmatch`很容易，因为您可以将其提供给用户，让他们编写`*Smith`或`Smith*`，具体取决于他们是在寻找名为Smith的人还是姓氏为Smith的人：

```py
>>> senders = ['Jane Smith', 'Smith Lincoln']
>>> fnmatch.filter(senders, 'Smith*')
['Smith Lincoln']
>>> fnmatch.filter(senders, '*Smith')
['Jane Smith']
```

# 文本相似性

在许多情况下，当处理文本时，我们可能需要识别与其他文本相似的文本，即使这两者并不相等。这在记录链接、查找重复条目或更正打字错误时非常常见。

查找文本相似性并不是一项简单的任务。如果您尝试自己去做，您很快就会意识到它很快变得复杂和缓慢。

Python库提供了在`difflib`模块中检测两个序列之间差异的工具。由于文本本身是一个序列（字符序列），我们可以应用提供的函数来检测字符串的相似性。

# 如何做...

执行此食谱的以下步骤：

1.  给定一个字符串，我们想要比较：

```py
>>> s = 'Today the weather is nice'
```

1.  此外，我们想将一组字符串与第一个字符串进行比较：

```py
>>> s2 = 'Today the weater is nice'
>>> s3 = 'Yesterday the weather was nice'
>>> s4 = 'Today my dog ate steak'
```

1.  我们可以使用`difflib.SequenceMatcher`来计算字符串之间的相似度（从0到1）。

```py
>>> import difflib
>>> difflib.SequenceMatcher(None, s, s2, False).ratio()
0.9795918367346939
>>> difflib.SequenceMatcher(None, s, s3, False).ratio()
0.8
>>> difflib.SequenceMatcher(None, s, s4, False).ratio()
0.46808510638297873
```

因此，`SequenceMatcher`能够检测到`s`和`s2`非常相似（98%），除了`weather`中的拼写错误之外，它们实际上是完全相同的短语。然后它指出`Today the weather is nice`与`Yesterday the weather was nice`相似度为80%，最后指出`Today the weather is nice`和`Today my dog ate steak`几乎没有共同之处。

# 还有更多...

`SequenceMatcher`提供了对一些值标记为*junk*的支持。您可能期望这意味着这些值被忽略，但实际上并非如此。

使用和不使用垃圾计算比率在大多数情况下将返回相同的值：

```py
>>> a = 'aaaaaaaaaaaaaXaaaaaaaaaa'
>>> b = 'X'
>>> difflib.SequenceMatcher(lambda c: c=='a', a, b, False).ratio()
0.08
>>> difflib.SequenceMatcher(None, a, b, False).ratio()
0.08    
```

即使我们提供了一个报告所有`a`结果为垃圾的`isjunk`函数（`SequenceMatcher`的第一个参数），`a`的结果也没有被忽略。

您可以通过使用`.get_matching_blocks()`来看到，在这两种情况下，字符串匹配的唯一部分是`X`在位置`13`和`0`处的`a`和`b`：

```py
>>> difflib.SequenceMatcher(None, a, b, False).get_matching_blocks()
[Match(a=13, b=0, size=1), Match(a=24, b=1, size=0)]
>>> difflib.SequenceMatcher(lambda c: c=='a', a, b, False).get_matching_blocks()
[Match(a=13, b=0, size=1), Match(a=24, b=1, size=0)]
```

如果您想在计算差异时忽略一些字符，您将需要在运行`SequenceMatcher`之前剥离它们，也许使用一个丢弃它们的翻译映射：

```py
>>> discardmap = str.maketrans({"a": None})
>>> difflib.SequenceMatcher(None, a.translate(discardmap), b.translate(discardmap), False).ratio()
1.0
```

# 文本建议

在我们之前的食谱中，我们看到`difflib`如何计算两个字符串之间的相似度。这意味着我们可以计算两个单词之间的相似度，并向我们的用户提供建议更正。

如果已知*正确*单词的集合（通常对于任何语言都是如此），我们可以首先检查单词是否在这个集合中，如果不在，我们可以寻找最相似的单词建议给用户正确的拼写。

# 如何做...

遵循此食谱的步骤是：

1.  首先，我们需要一组有效的单词。为了避免引入整个英语词典，我们只会抽样一些单词：

```py
dictionary = {'ability', 'able', 'about', 'above', 'accept',    
              'according', 
              'account', 'across', 'act', 'action', 'activity', 
              'actually', 
              'add', 'address', 'administration', 'admit', 'adult', 
              'affect', 
              'after', 'again', 'against', 'age', 'agency', 
              'agent', 'ago', 
              'agree', 'agreement', 'ahead', 'air', 'all', 'allow',  
              'almost', 
              'alone', 'along', 'already', 'also', 'although', 
              'always', 
              'American', 'among', 'amount', 'analysis', 'and', 
              'animal', 
              'another', 'answer', 'any', 'anyone', 'anything', 
              'appear', 
              'apply', 'approach', 'area', 'argue', 
              'arm', 'around', 'arrive', 
              'art', 'article', 'artist', 'as', 'ask', 'assume', 
              'at', 'attack', 
              'attention', 'attorney', 'audience', 'author',  
              'authority', 
              'available', 'avoid', 'away', 'baby', 'back', 'bad', 
              'bag', 
              'ball', 'bank', 'bar', 'base', 'be', 'beat', 
              'beautiful', 
              'because', 'become'}
```

1.  然后我们可以编写一个函数，对于提供的任何短语，都会在我们的字典中查找单词，如果找不到，就通过`difflib`提供最相似的候选词：

```py
import difflib

def suggest(phrase):
    changes = 0
    words = phrase.split()
    for idx, w in enumerate(words):
        if w not in dictionary:
            changes += 1
            matches = difflib.get_close_matches(w, dictionary)
            if matches:
                words[idx] = matches[0]
    return changes, ' '.join(words)
```

1.  我们的`suggest`函数将能够检测拼写错误并建议更正的短语：

```py
>>> suggest('assume ani answer')
(1, 'assume any answer')
>>> suggest('anoter agrement ahead')
(2, 'another agreement ahead')
```

第一个返回的参数是检测到的错误单词数，第二个是具有最合理更正的字符串。

1.  如果我们的短语没有错误，我们将得到原始短语的`0`：

```py
>>> suggest('beautiful art')
(0, 'beautiful art')
```

# 模板

向用户显示文本时，经常需要根据软件状态动态生成文本。

通常，这会导致这样的代码：

```py
name = 'Alessandro'
messages = ['Message 1', 'Message 2']

txt = 'Hello %s, You have %s message' % (name, len(messages))
if len(messages) > 1:
    txt += 's'
txt += ':n'
for msg in messages:
    txt += msg + 'n'
print(txt)
```

这使得很难预见消息的即将到来的结构，而且在长期内也很难维护。生成文本时，通常更方便的是反转这种方法，而不是将文本放入代码中，我们应该将代码放入文本中。这正是模板引擎所做的，虽然标准库提供了非常完整的格式化解决方案，但缺少一个开箱即用的模板引擎，但可以很容易地扩展为一个模板引擎。

# 如何做...

本教程的步骤如下：

1.  `string.Formatter`对象允许您扩展其语法，因此我们可以将其专门化以支持将代码注入到它将要接受的表达式中：

```py
import string

class TemplateFormatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
        if field_name.startswith("$"):
            code = field_name[1:]
            val = eval(code, {}, dict(kwargs))
            return val, field_name
        else:
            return super(TemplateFormatter, self).get_field(field_name, args, kwargs)
```

1.  然后，我们的`TemplateFormatter`可以用来以更简洁的方式生成类似于我们示例的文本：

```py
messages = ['Message 1', 'Message 2']

tmpl = TemplateFormatter()
txt = tmpl.format("Hello {name}, "
                  "You have {$len(messages)} message{$len(messages) and 's'}:n{$'\n'.join(messages)}", 
                  name='Alessandro', messages=messages)
print(txt)
```

结果应该是：

```py
Hello Alessandro, You have 2 messages:
Message 1
Message 2
```

# 它是如何工作的...

`string.Formatter`支持与`str.format`方法支持的相同语言。实际上，它根据Python称为*格式化字符串语法*的内容解析包含在`{}`中的表达式。`{}`之外的所有内容保持不变，而`{}`中的任何内容都会被解析为`field_name!conversion:format_spec`规范。因此，由于我们的`field_name`不包含`!`或`:`，它可以是任何其他内容。

然后提取的`field_name`被提供给`Formatter.get_field`，以查找`format`方法提供的参数中该字段的值。

因此，例如，采用这样的表达式：

```py
string.Formatter().format("Hello {name}", name='Alessandro')
```

这导致：

```py
Hello Alessandro
```

因为`{name}`被识别为要解析的块，所以会在`.format`参数中查找名称，并保留其余部分不变。

这非常方便，可以解决大多数字符串格式化需求，但缺乏像循环和条件语句这样的真正模板引擎的功能。

我们所做的是扩展`Formatter`，不仅解析`field_name`中指定的变量，还评估Python表达式。

由于我们知道所有的`field_name`解析都要经过`Formatter.get_field`，在我们自己的自定义类中覆盖该方法将允许我们更改每当评估像`{name}`这样的`field_name`时发生的情况：

```py
class TemplateFormatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
```

为了区分普通变量和表达式，我们使用了`$`符号。由于Python变量永远不会以`$`开头，因此我们不会与提供给格式化的参数发生冲突（因为`str.format($something=5`实际上是Python中的语法错误）。因此，像`{$something}`这样的`field_name`不意味着查找`''$something`的值，而是评估`something`表达式：

```py
if field_name.startswith("$"):
    code = field_name[1:]
    val = eval(code, {}, dict(kwargs))
```

`eval`函数运行在字符串中编写的任何代码，并将执行限制为表达式（Python中的表达式总是导致一个值，与不导致值的语句不同），因此我们还进行了语法检查，以防止模板用户编写`if something: x='hi'`，这将不会提供任何值来显示在渲染模板后的文本中。

然后，由于我们希望用户能够查找到他们提供的表达式引用的任何变量（如`{$len(messages)}`），我们将`kwargs`提供给`eval`作为`locals`变量，以便任何引用变量的表达式都能正确解析。我们还提供一个空的全局上下文`{}`，以便我们不会无意中触及软件的任何全局变量。

剩下的最后一部分就是将`eval`提供的表达式执行结果作为`field_name`解析的结果返回：

```py
return val, field_name
```

真正有趣的部分是所有处理都发生在`get_field`阶段。转换和格式规范仍然受支持，因为它们是应用于`get_field`返回的值。

这使我们可以写出这样的东西：

```py
{$3/2.0:.2f}
```

我们得到的输出是`1.50`，而不是`1.5`。这是因为我们在我们专门的`TemplateFormatter.get_field`方法中首先评估了`3/2.0`，然后解析器继续应用格式规范（`.2f`）到结果值。

# 还有更多...

我们的简单模板引擎很方便，但仅限于我们可以将生成文本的代码表示为一组表达式和静态文本的情况。

问题在于更高级的模板并不总是可以表示。我们受限于简单的表达式，因此实际上任何不能用`lambda`表示的东西都不能由我们的模板引擎执行。

虽然有人会认为通过组合多个`lambda`可以编写非常复杂的软件，但大多数人会认为语句会导致更可读的代码。

因此，如果你需要处理非常复杂的文本，你应该使用功能齐全的模板引擎，并寻找像Jinja、Kajiki或Mako这样的解决方案。特别是对于生成HTML，像Kajiki这样的解决方案，它还能够验证你的HTML，非常方便，可以比我们的`TemplateFormatter`做得更多。

# 拆分字符串并保留空格

通常在按空格拆分字符串时，开发人员倾向于依赖`str.split`，它能够很好地完成这个目的。但是当需要*拆分一些空格并保留其他空格*时，事情很快变得更加困难，实现一个自定义解决方案可能需要投入时间来进行适当的转义。

# 如何做...

只需依赖`shlex.split`而不是`str.split`：

```py
>>> import shlex
>>>
>>> text = 'I was sleeping at the "Windsdale Hotel"'
>>> print(shlex.split(text))
['I', 'was', 'sleeping', 'at', 'the', 'Windsdale Hotel']
```

# 工作原理...

`shlex`是最初用于解析Unix shell代码的模块。因此，它支持通过引号保留短语。通常在Unix命令行中，由空格分隔的单词被提供为调用命令的参数，但如果你想将多个单词作为单个参数提供，可以使用引号将它们分组。

这正是`shlex`所复制的，为我们提供了一个可靠的驱动拆分的方法。我们只需要用双引号或单引号包裹我们想要保留的所有内容。

# 清理文本

在分析用户提供的文本时，我们通常只对有意义的单词感兴趣；标点、空格和连词可能很容易妨碍我们。假设你想要统计一本书中单词的频率，你不希望最后得到"world"和"world"被计为两个不同的单词。

# 如何做...

你需要执行以下步骤：

1.  提供要清理的文本：

```py
txt = """And he looked over at the alarm clock,
ticking on the chest of drawers. "God in Heaven!" he thought.
It was half past six and the hands were quietly moving forwards,
it was even later than half past, more like quarter to seven.
Had the alarm clock not rung? He could see from the bed that it
had been set for four o'clock as it should have been; it certainly must have rung.
Yes, but was it possible to quietly sleep through that furniture-rattling noise?
True, he had not slept peacefully, but probably all the more deeply because of that."""
```

1.  我们可以依赖`string.punctuation`来知道我们想要丢弃的字符，并制作一个转换表来丢弃它们全部：

```py
>>> import string
>>> trans = str.maketrans('', '', string.punctuation)
>>> txt = txt.lower().translate(trans)
```

结果将是我们文本的清理版本：

```py
"""and he looked over at the alarm clock
ticking on the chest of drawers god in heaven he thought
it was half past six and the hands were quietly moving forwards
it was even later than half past more like quarter to seven
had the alarm clock not rung he could see from the bed that it
had been set for four oclock as it should have been it certainly must have rung
yes but was it possible to quietly sleep through that furniturerattling noise
true he had not slept peacefully but probably all the more deeply because of that"""
```

# 工作原理...

这个示例的核心是使用转换表。转换表是将字符链接到其替换的映射。像`{'c': 'A'}`这样的转换表意味着任何`'c'`都必须替换为`'A'`。

`str.maketrans`是用于构建转换表的函数。第一个参数中的每个字符将映射到第二个参数中相同位置的字符。然后最后一个参数中的所有字符将映射到`None`：

```py
>>> str.maketrans('a', 'b', 'c')
{97: 98, 99: None}
```

`97`，`98`和`99`是`'a'`，`'b'`和`'c'`的Unicode值：

```py
>>> print(ord('a'), ord('b'), ord('c'))
97 98 99
```

然后我们的映射可以传递给`str.translate`来应用到目标字符串上。有趣的是，任何映射到`None`的字符都将被删除：

```py
>>> 'ciao'.translate(str.maketrans('a', 'b', 'c'))
'ibo'
```

在我们之前的示例中，我们将`string.punctuation`作为`str.maketrans`的第三个参数。

`string.punctuation`是一个包含最常见标点字符的字符串：

```py
>>> string.punctuation
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
```

通过这样做，我们建立了一个事务映射，将每个标点字符映射到`None`，并没有指定任何其他映射：

```py
>>> str.maketrans('', '', string.punctuation)
{64: None, 124: None, 125: None, 91: None, 92: None, 93: None,
 94: None, 95: None, 96: None, 33: None, 34: None, 35: None,
 36: None, 37: None, 38: None, 39: None, 40: None, 41: None,
 42: None, 43: None, 44: None, 45: None, 46: None, 47: None,
 123: None, 126: None, 58: None, 59: None, 60: None, 61: None,
 62: None, 63: None}
```

这样一来，一旦应用了`str.translate`，标点字符就都被丢弃了，保留了所有其他字符：

```py
>>> 'This, is. A test!'.translate(str.maketrans('', '', string.punctuation))
'This is A test'
```

# 文本规范化

在许多情况下，一个单词可以用多种方式书写。例如，写"Über"和"Uber"的用户可能意思相同。如果你正在为博客实现标记等功能，你肯定不希望最后得到两个不同的标记。

因此，在保存标签之前，您可能希望将它们标准化为普通的ASCII字符，以便它们最终被视为相同的标签。

# 如何做...

我们需要的是一个翻译映射，将所有带重音的字符转换为它们的普通表示：

```py
import unicodedata, sys

class unaccented_map(dict):
    def __missing__(self, key):
        ch = self.get(key)
        if ch is not None:
            return ch
        de = unicodedata.decomposition(chr(key))
        if de:
            try:
                ch = int(de.split(None, 1)[0], 16)
            except (IndexError, ValueError):
                ch = key
        else:
            ch = key
        self[key] = ch
        return ch

unaccented_map = unaccented_map()
```

然后我们可以将其应用于任何单词来进行规范化：

```py
>>> 'Über'.translate(unaccented_map) Uber >>> 'garçon'.translate(unaccented_map) garcon
```

# 它是如何工作的...

我们已经知道如何解释*清理文本*食谱中解释的那样，`str.translate`是如何工作的：每个字符都在翻译表中查找，并且用表中指定的替换进行替换。

因此，我们需要的是一个翻译表，将`"Ü"`映射到`"U"`，将`"ç"`映射到`"c"`，依此类推。

但是我们如何知道所有这些映射呢？这些字符的一个有趣特性是它们可以被认为是带有附加符号的普通字符。就像`à`可以被认为是带有重音的`a`。

Unicode等价性知道这一点，并提供了多种写入被认为是相同字符的方法。我们真正感兴趣的是分解形式，这意味着将字符写成定义它的多个分隔符。例如，`é`将被分解为`0065`和`0301`，这是`e`和重音的代码点。

Python提供了一种通过`unicodedata.decompostion`函数知道字符分解版本的方法：

```py
>>> import unicodedata
>>> unicodedata.decomposition('é')
'0065 0301'
```

第一个代码点是基本字符的代码点，而第二个是添加的符号。因此，要规范化我们的`è`，我们将选择第一个代码点`0065`并丢弃符号：

```py
>>> unicodedata.decomposition('é').split()[0]
'0065'
```

现在我们不能单独使用代码点，但我们想要它表示的字符。幸运的是，`chr`函数提供了一种从其代码点的整数表示中获取字符的方法。

`unicodedata.decomposition`函数提供的代码点是表示十六进制数字的字符串，因此首先我们需要将它们转换为整数：

```py
>>> int('0065', 16)
101
```

然后我们可以应用`chr`来知道实际的字符：

```py
>>> chr(101)
'e'
```

现在我们知道如何分解这些字符并获得我们想要将它们全部标准化为的基本字符，但是我们如何为它们构建一个翻译映射呢？

答案是我们不需要。事先为所有字符构建翻译映射并不是很方便，因此我们可以使用字典提供的功能，在需要时动态地为字符构建翻译。

翻译映射是字典，每当字典需要查找它不知道的键时，它可以依靠`__missing__`方法为该键生成一个值。因此，我们的`__missing__`方法必须做我们刚才做的事情，并使用`unicodedata.decomposition`来获取字符的规范化版本，每当`str.translate`尝试在我们的翻译映射中查找它时。

一旦我们计算出所请求字符的翻译，我们只需将其存储在字典本身中，这样下次再被请求时，我们就不必再计算它。

因此，我们的食谱的`unaccented_map`只是一个提供`__missing__`方法的字典，该方法依赖于`unicodedata.decompostion`来检索每个提供的字符的规范化版本。

如果它无法找到字符的非规范化版本，它将只返回原始版本一次，以免字符串被损坏。

# 对齐文本

在打印表格数据时，通常非常重要的是确保文本正确对齐到固定长度，既不长也不短于我们为表格单元保留的空间。

如果文本太短，下一列可能会开始得太早；如果太长，它可能会开始得太晚。这会导致像这样的结果：

```py
col1 | col2-1
col1-2 | col2-2
```

或者这样：

```py
col1-000001 | col2-1
col1-2 | col2-2
```

这两者都很难阅读，并且远非显示正确表格的样子。

给定固定的列宽（20个字符），我们希望我们的文本始终具有确切的长度，以便它不会导致错位的表格。

# 如何做...

以下是此食谱的步骤：

1.  一旦将`textwrap`模块与`str`对象的特性结合起来，就可以帮助我们实现预期的结果。首先，我们需要打印的列的内容：

```py
cols = ['hello world', 
        'this is a long text, maybe longer than expected, surely long enough', 
        'one more column']
```

1.  然后我们需要修复列的大小：

```py
COLSIZE = 20
```

1.  一旦这些准备好了，我们就可以实际实现我们的缩进函数：

```py
import textwrap, itertools

def maketable(cols):
    return 'n'.join(map(' | '.join, itertools.zip_longest(*[
        [s.ljust(COLSIZE) for s in textwrap.wrap(col, COLSIZE)] for col in cols
    ], fillvalue=' '*COLSIZE)))
```

1.  然后我们可以正确地打印任何表格：

```py
>>> print(maketable(cols))
hello world          | this is a long text, | one more column     
                     | maybe longer than    |                     
                     | expected, surely     |                     
                     | long enough          |                     
```

# 它是如何工作的...

我们必须解决三个问题来实现我们的`maketable`函数：

+   长度小于20个字符的文本

+   将长度超过20个字符的文本拆分为多行

+   填充列中缺少的行

如果我们分解我们的`maketable`函数，它的第一件事就是将长度超过20个字符的文本拆分为多行：

```py
[textwrap.wrap(col, COLSIZE) for col in cols]
```

将其应用于每一列，我们得到了一个包含列的列表，每个列包含一列行：

```py
[['hello world'], 
 ['this is a long text,', 'maybe longer than', 'expected, surely', 'long enough'],
 ['one more column']]
```

然后我们需要确保每行长度小于20个字符的文本都扩展到恰好20个字符，以便我们的表保持形状，这是通过对每行应用`ljust`方法来实现的：

```py
[[s.ljust(COLSIZE) for s in textwrap.wrap(col, COLSIZE)] for col in cols]
```

将`ljust`与`textwrap`结合起来，就得到了我们想要的结果：包含每个20个字符的行的列的列表：

```py
[['hello world         '], 
 ['this is a long text,', 'maybe longer than   ', 'expected, surely    ', 'long enough         '],
 ['one more column     ']]
```

现在我们需要找到一种方法来翻转行和列，因为在打印时，由于`print`函数一次打印一行，我们需要按行打印。此外，我们需要确保每列具有相同数量的行，因为按行打印时需要打印所有行。

这两个需求都可以通过`itertools.zip_longest`函数解决，它将生成一个新列表，通过交错提供的每个列表中包含的值，直到最长的列表用尽。由于`zip_longest`会一直进行，直到最长的可迭代对象用尽，它支持一个`fillvalue`参数，该参数可用于指定用于填充较短列表的值：

```py
list(itertools.zip_longest(*[
    [s.ljust(COLSIZE) for s in textwrap.wrap(col, COLSIZE)] for col in cols
], fillvalue=' '*COLSIZE))
```

结果将是一列包含一列的行的列表，对于没有值的行，将有空列：

```py
[('hello world         ', 'this is a long text,', 'one more column     '), 
 ('                    ', 'maybe longer than   ', '                    '), 
 ('                    ', 'expected, surely    ', '                    '), 
 ('                    ', 'long enough         ', '                    ')]
```

文本的表格形式现在清晰可见。我们函数中的最后两个步骤涉及在列之间添加`|`分隔符，并通过`' | '.join`将列合并成单个字符串：

```py
map(' | '.join, itertools.zip_longest(*[
    [s.ljust(COLSIZE) for s in textwrap.wrap(col, COLSIZE)] for col in cols
], fillvalue=' '*COLSIZE))
```

这将导致一个包含所有三列文本的字符串列表：

```py
['hello world          | this is a long text, | one more column     ', 
 '                     | maybe longer than    |                     ', 
 '                     | expected, surely     |                     ', 
 '                     | long enough          |                     ']
```

最后，行可以被打印。为了返回单个字符串，我们的函数应用了最后一步，并通过应用最终的`'n'.join()`将所有行连接成一个由换行符分隔的单个字符串，从而返回一个包含整个文本的单个字符串，准备打印：

```py
'''hello world          | this is a long text, | one more column     
                        | maybe longer than    |                     
                        | expected, surely     |                     
                        | long enough          |                     '''
```
