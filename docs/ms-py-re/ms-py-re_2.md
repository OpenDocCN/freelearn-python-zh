# 第二章：Python 正则表达式

在上一章中，我们已经看到了通用正则表达式的工作原理。在本章中，我们将带您了解 Python 提供给我们的所有操作来处理正则表达式以及 Python 如何处理它们。

为此，我们将看到处理正则表达式时语言的怪癖，不同类型的字符串，通过`RegexObject`和`MatchObject`类提供的 API，我们可以深入地了解它们的所有操作，并提供许多示例，以及一些通常由用户面临的问题。最后，我们将看到 Python 和其他正则表达式引擎以及 Python 2 和 Python 3 之间的小细微差别。

# 简要介绍

自 v1.5 以来，Python 提供了一种类似 Perl 的正则表达式，其中有一些微妙的例外情况，我们稍后会看到。要搜索的模式和字符串都可以是**Unicode**字符串，也可以是 8 位字符串（**ASCII**）。

### 提示

Unicode 是一种通用编码，有超过 110,000 个字符和 100 种文字，可以表示世界上所有的活字和历史文字。您可以将它视为数字之间的映射，或者称为代码点，和字符。因此，我们可以用一个单一的数字表示每个字符，无论是什么语言。例如，字符![A brief introduction](img/inlinemedia1.jpg)是数字 26159，它在 Python 中表示为\u662f（十六进制）。

正则表达式由`re`模块支持。因此，与 Python 中的所有模块一样，我们只需要导入它就可以开始使用它们。为此，我们需要使用以下代码行启动 Python 交互式 shell：

```py
>>> import re
```

一旦我们导入了模块，我们就可以开始尝试匹配模式。为此，我们需要编译一个模式，将其转换为**字节码**，如下面的代码行所示。这个字节码稍后将由用 C 编写的引擎执行。

```py
>>> pattern = re.compile(r'\bfoo\b')
```

### 提示

字节码是一种中间语言。它是由语言生成的输出，稍后将由解释器解释。由 JVM 解释的 Java 字节码可能是最著名的例子。

一旦我们有了编译后的模式，我们可以尝试将其与字符串匹配，就像以下代码中所示的那样：

```py
>>> pattern.match("foo bar")
<_sre.SRE_Match at 0x108acac60>
```

正如我们在前面的例子中提到的，我们编译了一个模式，然后搜索这个模式是否与文本*foo bar*匹配。

在命令行中使用 Python 和正则表达式很容易进行快速测试。您只需要启动 Python 解释器并像之前提到的那样导入`re`模块。但是，如果您更喜欢使用 GUI 来测试您的正则表达式，您可以在以下链接下载一个用 Python 编写的 GUI：

[`svn.python.org/view/*checkout*/python/trunk/Tools/scripts/redemo.py?content-type=text%2Fplain`](http://svn.python.org/view/*checkout*/python/trunk/Tools/scripts/redemo.py?content-type=text%2Fplain)

有许多在线工具，比如[`pythex.org/`](https://pythex.org/)，以及桌面程序，比如我们将在第五章中介绍的 RegexBuddy，*正则表达式的性能*。

在这一点上，最好使用解释器来熟练掌握它们并获得直接的反馈。

# 字符串文字中的反斜杠

正则表达式不是 Python 核心语言的一部分。因此，它们没有特殊的语法，因此它们被处理为任何其他字符串。正如我们在第一章中看到的，*介绍正则表达式*，反斜杠字符`\`用于指示正则表达式中的元字符或特殊形式。反斜杠也用于字符串中转义特殊字符。换句话说，在 Python 中它有特殊含义。因此，如果我们需要使用`\`字符，我们将不得不对其进行转义：`\\`。这将给反斜杠赋予字符串字面意义。然而，为了在正则表达式中匹配，我们应该转义反斜杠，实际上写四个反斜杠：`\\\\`。

举个例子，让我们写一个正则表达式来匹配`\`：

```py
>>> pattern = re.compile("\\\\")
>>> pattern.match("\\author")
<_sre.SRE_Match at 0x104a88e68>
```

正如你所看到的，当模式很长时，这是繁琐且难以理解的。

Python 提供了**原始字符串表示法** `r`，其中反斜杠被视为普通字符。因此，`r"\b"`不再是退格键；它只是字符`\`和字符`b`，对于`r"\n"`也是一样。

Python 2.x 和 Python 3.x 对字符串的处理方式不同。在 Python 2 中，有两种类型的字符串，8 位字符串和 Unicode 字符串；而在 Python 3 中，我们有文本和二进制数据。文本始终是 Unicode，并且编码后的 Unicode 表示为二进制数据（[`docs.python.org/3.0/whatsnew/3.0.html#text-vs-data-instead-of-unicode-vs-8-bit`](http://docs.python.org/3.0/whatsnew/3.0.html#text-vs-data-instead-of-unicode-vs-8-bit)）。

字符串有特殊的表示法来指示我们使用的类型。

## 字符串 Python 2.x

| 类型 | 前缀 | 描述 |
| --- | --- | --- |

| 字符串 |   | 字符串字面值。它们通过使用默认编码（在我们的情况下是 UTF-8）进行自动编码。反斜杠是必要的，以转义有意义的字符。

```py
>>>"España \n"
'Espa\xc3\xb1a \n'
```

|

| 原始字符串 | `r` 或 `R` | 它们与字面字符串相同，除了反斜杠被视为普通字符。

```py
>>>r"España \n"
'Espa\xc3\xb1a \\n'
```

|

Unicode 字符串 | `u` 或 `U` | 这些字符串使用 Unicode 字符集（ISO 10646）。

```py
>>>u"España \n"
u'Espa\xf1a \n'
```

|

Unicode 原始字符串 | `ur` 或 `UR` | 它们是 Unicode 字符串，但将反斜杠视为普通的原始字符串。

```py
>>>ur"España \n"
u'Espa\xf1a \\n'
```

|

转到*Python 3 中的新内容*部分，了解 Python 3 中的表示法是什么

根据 Python 官方文档，使用原始字符串是推荐的选项，这也是我们在整本书中将要使用的 Python 2.7。因此，考虑到这一点，我们可以将正则表达式重写如下：

```py
>>> pattern = re.compile(r"\\")
>>> pattern.match(r"\author")
<_sre.SRE_Match at 0x104a88f38>
```

# Python 正则表达式的构建块

在 Python 中，有两种不同的对象处理正则表达式：

+   `RegexObject`：它也被称为*Pattern Object*。它表示编译后的正则表达式

+   `MatchObject`：它表示匹配的模式

## RegexObject

为了开始匹配模式，我们将不得不编译正则表达式。Python 给了我们一个接口来做到这一点，就像我们之前看到的那样。结果将是一个模式对象或`RegexObject`。这个对象有几种用于正则表达式的典型操作的方法。正如我们将在后面看到的，`re`模块提供了每个操作的简写，以便我们可以避免首先编译它。

```py
>>> pattern = re.compile(r'fo+')
```

正则表达式的编译产生一个可重用的模式对象，提供了所有可以进行的操作，比如匹配模式和找到所有匹配特定正则表达式的子字符串。因此，例如，如果我们想知道一个字符串是否以`<HTML>`开头，我们可以使用以下代码：

```py
>>> pattern = re.compile(r'<HTML>')
>>> pattern.match("<HTML>")
   <_sre.SRE_Match at 0x108076578>
```

有两种匹配模式和执行与正则表达式相关的操作的方法。我们可以编译一个模式，这给了我们一个`RegexObject`，或者我们可以使用模块操作。让我们在以下示例中比较这两种不同的机制。

如果我们想要重复使用正则表达式，我们可以使用以下代码：

```py
>>> pattern = re.compile(r'<HTML>')
>>> pattern.match("<HTML>")
```

另一方面，我们可以直接在模块上执行操作，使用以下代码行：

```py
>>> re.match(r'<HTML>', "<HTML>")
```

`re`模块为`RegexObject`中的每个操作提供了一个包装器。您可以将它们视为快捷方式。

在内部，这些包装器创建了`RegexObject`，然后调用相应的方法。您可能想知道每次调用这些包装器时是否都会先编译正则表达式。答案是否定的。`re`模块会缓存已编译的模式，以便在将来的调用中不必再次编译它。

注意您的程序的内存需求。当您使用模块操作时，您无法控制缓存，因此可能会导致大量内存使用。您可以随时使用`re.purge`来清除缓存，但这会影响性能。使用编译后的模式允许您对内存消耗进行精细控制，因为您可以决定何时清除它们。

这两种方式之间有一些区别。使用`RegexObject`，可以限制模式将在其中搜索的区域，例如限制在索引 2 和 20 之间的模式搜索。除此之外，您可以通过在模块中使用操作来在每次调用中设置`flags`。但是要小心；每次更改标志时，都会编译并缓存一个新模式。

让我们深入了解可以使用模式对象执行的最重要操作。

### 搜索

让我们看看我们必须在字符串中查找模式的操作。请注意，Python 有两种操作，match 和 search；而许多其他语言只有一种操作，match。

#### match(string[, pos[, endpos]])

这种方法尝试仅在字符串的开头匹配编译后的模式。如果匹配成功，则返回一个`MatchObject`。因此，例如，让我们尝试匹配一个字符串是否以`<HTML>`开头：

```py
>>> pattern = re.compile(r'<HTML>')
>>> pattern.match("<HTML><head>")
<_sre.SRE_Match at 0x108076578>
```

在上面的示例中，首先我们编译了模式，然后在`<HTML><head>`字符串中找到了一个匹配。

让我们看看当字符串不以`<HTML>`开头时会发生什么，如下面的代码行所示：

```py
>>> pattern.match("**⇢**<HTML>")
    None
```

如您所见，没有匹配。请记住我们之前说过的，`match`尝试在字符串的开头进行匹配。字符串以空格开头，与模式不同。请注意与以下示例中的`search`的区别：

```py
>>> pattern.search("⇢<HTML>")
<_sre.SRE_Match at 0x108076578>
```

正如预期的那样，我们有一个匹配。

可选的**pos**参数指定从哪里开始搜索，如下面的代码所示：

```py
>>> pattern = re.compile(r'<HTML>')
>>> pattern.match("⇢ ⇢ <HTML>")
    None
>>> pattern.match("**⇢ ⇢ **<HTML>", 2)
 **<_sre.SRE_Match at 0x1043bc850>

```

在上面的代码中，我们可以看到即使字符串中有两个空格，模式也能匹配。这是可能的，因为我们将**pos**设置为`2`，所以匹配操作从该位置开始搜索。

请注意，**pos**大于 0 并不意味着字符串从该索引开始，例如：

```py
>>> pattern = re.compile(**r'^<HTML>'**)
>>> pattern.match("<HTML>")
   <_sre.SRE_Match at 0x1043bc8b8>
>>> pattern.match("⇢ ⇢ <HTML>",  2)
    None
```

在上面的代码中，我们创建了一个模式，用于匹配字符串，其中“start”后的第一个字符后面跟着`<HTML>`。然后，我们尝试从第二个字符`<`开始匹配字符串`<HTML>`。由于模式试图首先在位置`2`匹配`^`元字符，因此没有匹配。

### 提示

**锚字符提示**

字符`^`和`$`分别表示字符串的开头和结尾。您既看不到它们在字符串中，也不能写它们，但它们总是存在的，并且是正则表达式引擎的有效字符。

请注意，如果我们将字符串切片 2 个位置，结果会有所不同，如下面的代码所示：

```py
>>> pattern.match("⇢ ⇢ <HTML>"[2:])
   <_sre.SRE_Match at 0x1043bca58>
```

切片给我们一个新的字符串；因此，它里面有一个`^`元字符。相反，**pos**只是将索引移动到字符串中搜索的起始点。

第二个参数**endpos**设置模式在字符串中尝试匹配的距离。在下面的情况中，它相当于切片：

```py
>>> pattern = re.compile(r'<HTML>')
>>> pattern.match("<HTML>"[:2]) 
    None
>>> pattern.match("<HTML>", 0, 2) 
    None
```

因此，在下面的情况中，我们不会遇到**pos**中提到的问题。即使使用了`$`元字符，也会有匹配：

```py
>>> pattern = re.compile(r'<HTML>$')
>>> pattern.match("<HTML>⇢", 0,6)
<_sre.SRE_Match object at 0x1007033d8>
>>> pattern.match("<HTML>⇢"[:6])
<_sre.SRE_Match object at 0x100703370>
```

如您所见，切片和**endpos**之间没有区别。

#### search(string[, pos[, endpos]])

这个操作就像许多语言中的**match**，例如 Perl。它尝试在字符串的任何位置匹配模式，而不仅仅是在开头。如果有匹配，它会返回一个`MatchObject`。

```py
>>> pattern = re.compile(r"world")
>>> pattern.search("hello⇢world")
   <_sre.SRE_Match at 0x1080901d0>
>>> pattern.search("hola⇢mundo ")
    None
```

**pos**和**endpos**参数的含义与`match`操作中的相同。

请注意，使用`MULTILINE`标志，`^`符号在字符串的开头和每行的开头匹配（我们稍后会更多地了解这个标志）。因此，它改变了`search`的行为。

在下面的例子中，第一个`search`匹配`<HTML>`，因为它在字符串的开头，但第二个`search`不匹配，因为字符串以空格开头。最后，在第三个`search`中，我们有一个匹配，因为我们在新行后找到了`<HTML>`，这要归功于`re.MULTILINE`。

```py
>>> pattern = re.compile(r'^<HTML>', re.MULTILINE)
>>> pattern.search("<HTML>")
   <_sre.SRE_Match at 0x1043d3100>
>>> pattern.search("⇢<HTML>")
   None
>>> pattern.search("**⇢ ⇢**\n<HTML>")
   <_sre.SRE_Match at 0x1043bce68>
```

因此，只要**pos**参数小于或等于新行，就会有一个匹配。

```py
>>> pattern.search("⇢ ⇢\n<HTML>",  3)
  <_sre.SRE_Match at 0x1043bced0>
>>> pattern.search('</div></body>\n<HTML>', 4)
  <_sre.SRE_Match at 0x1036d77e8>
>>> pattern.search("**  **\n<HTML>", 4)
   None
```

#### findall(string[, pos[, endpos]])

以前的操作一次只能匹配一个。相反，在这种情况下，它返回一个列表，其中包含模式的所有不重叠的出现，而不是像`search`和`match`那样返回`MatchObject`。

在下面的例子中，我们正在寻找字符串中的每个单词。因此，我们得到一个列表，其中每个项目都是找到的模式，这里是一个单词。

```py
>>> pattern = re.compile(r"\w+")
>>> pattern.findall("hello⇢world")
    ['hello', 'world']
```

请记住，空匹配是结果的一部分：

```py
>>> pattern = re.compile(r'a*')
>>> pattern.findall("aba")
    ['a', '', 'a', '']
```

我敢打赌你想知道这里发生了什么？这个技巧来自`*`量词，它允许前面的正则表达式重复 0 次或更多次；与`?`量词发生的情况相同。

```py
>>> pattern = re.compile(r'a?')
>>> pattern.findall("aba")
    ['a', '', 'a', '']
```

基本上，它们两个都匹配表达式，即使前面的正则表达式没有找到：

![findall(string[, pos[, endpos]])](graphics/3156OS_02_01.jpg)

findall 匹配过程

首先，正则表达式匹配字符`a`，然后跟着`b`。由于`*`量词，空字符串也会匹配。之后，它匹配另一个`a`，最后尝试匹配`$`。正如我们之前提到的，即使你看不到`$`，它对于正则表达式引擎来说也是一个有效的字符。就像`b`一样，由于`*`量词，它会匹配。

我们在第一章*介绍正则表达式*中深入了解了量词。

如果模式中有组，它们将作为元组返回。字符串从左到右扫描，因此组将按照它们被找到的顺序返回。

以下示例尝试匹配由两个单词组成的模式，并为每个单词创建一个组。这就是为什么我们有一个元组列表，其中每个元组有两个组。

```py
>>> pattern = re.compile(r"(\w+) (\w+)")
>>> pattern.findall("Hello⇢world⇢hola⇢mundo")
    [('Hello', 'world'), ('hola', 'mundo')]
```

`findall`操作以及`groups`似乎是许多人困惑的另一件事情。在第三章*分组*中，我们专门有一个完整的部分来解释这个复杂的主题。

#### finditer(string[, pos[, endpos]])

它的工作原理与`findall`基本相同，但它返回一个迭代器，其中每个元素都是一个`MatchObject`，因此我们可以使用这个对象提供的操作。因此，当您需要每个匹配的信息时，例如匹配子字符串的位置时，它非常有用。有好几次，我发现自己使用它来理解`findall`中发生了什么。

让我们回到我们最初的一个例子。匹配每两个单词并捕获它们：

```py
>>> pattern = re.compile(r"(\w+) (\w+)")
>>> it = pattern.finditer("Hello⇢world⇢hola⇢mundo")
>>> match = it.next()
>>> match.groups()
    ('Hello', 'world')
>>> match.span()
    (0, 11)
```

在前面的例子中，我们可以看到我们得到了一个包含所有匹配的迭代器。对于迭代器中的每个元素，我们得到一个`MatchObject`，因此我们可以看到模式中捕获的组，在这种情况下是两个。我们还将得到匹配的位置。

```py
>>> match = it.next()
>>> match.groups()
    ('hola', 'mundo')
>>> match.span()
    (12, 22)
```

现在，我们从迭代器中消耗另一个元素，并执行与之前相同的操作。因此，我们得到下一个匹配，它的组和匹配的位置。我们与第一个匹配所做的事情一样：

```py
>>> match = it.next()
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
StopIteration
```

最后，我们尝试消耗另一个匹配，但在这种情况下会抛出`StopIteration`异常。这是指示没有更多元素的正常行为。

### 修改字符串

在本节中，我们将看到修改字符串的操作，比如将字符串分割的操作和替换其中某些部分的操作。

#### split(string, maxsplit=0)

在几乎每种语言中，你都可以在字符串中找到`split`操作。最大的区别在于`re`模块中的`split`更加强大，因为你可以使用正则表达式。因此，在这种情况下，字符串是基于模式的匹配进行分割的。和往常一样，最好的理解方法是通过一个例子，所以让我们将一个字符串分割成行：

```py
>>> re.split(r"\n", "Beautiful⇢is better⇢than⇢ugly.\nExplicit⇢is⇢better⇢than⇢implicit.")

['Beautiful⇢is⇢better⇢than⇢ugly.', 'Explicit⇢is⇢better⇢than⇢implicit.']
```

在前面的例子中，匹配是`\n`；因此，字符串是使用它作为分隔符进行分割的。让我们看一个更复杂的例子，如何获取字符串中的单词：

```py
>>> pattern = re.compile(**r"\W")
>>> pattern.split("hello⇢world")
['Hello', 'world']
```

在前面的例子中，我们定义了一个匹配任何非字母数字字符的模式。因此，在这种情况下，匹配发生在空格中。这就是为什么字符串被分割成单词。让我们看另一个例子来更好地理解它：

```py
>>> pattern = re.compile(**r"\W")
>>> pattern.findall("hello⇢world")
['⇢']
```

请注意，匹配的是空格。

**maxsplit**参数指定最多可以进行多少次分割，并将剩余部分返回为结果：

```py
>>> pattern = re.compile(r"\W")
>>> pattern.split("Beautiful is better than ugly", 2)
['Beautiful', 'is', 'better than ugly']
```

正如你所看到的，只有两个单词被分割，其他单词是结果的一部分。

你是否意识到匹配的模式没有被包括？看一下本节中的每个例子。如果我们想要捕获模式，我们该怎么办？

答案是使用组：

```py
>>> pattern = re.compile(r"(-)")
>>> pattern.split("hello-word")
['hello', '-', 'word']
```

这是因为分割操作总是返回捕获的组。

请注意，当一个组匹配字符串的开头时，结果将包含空字符串作为第一个结果：

```py
>>> pattern = re.compile(r"(\W)")
>>> pattern.split("⇢hello⇢word")
['', '⇢', 'hello', '⇢', 'word']
```

#### sub(repl, string, count=0)

此操作返回原始字符串中替换匹配模式后的结果字符串。如果未找到模式，则返回原始字符串。例如，我们将用`-`（破折号）替换字符串中的数字：

```py
>>> pattern = re.compile(r"[0-9]+")
>>> pattern.sub("-", "order0⇢order1⇢order13")
  'order-⇢order-⇢order-'
```

基本上，正则表达式匹配 1 个或多个数字，并用匹配的模式`0`、`1`和`13`替换为`-`（破折号）。

请注意，它替换了模式的最左边的非重叠出现。让我们看另一个例子：

```py
 >>> re.sub('00', '-', 'order00000')
   'order--0'
```

在前面的例子中，我们是两两替换零。因此，首先匹配并替换前两个，然后接下来的两个零也被匹配并替换，最后剩下最后一个零。

`repl`参数也可以是一个函数，这种情况下，它接收一个 MatchObject 作为参数，并返回的字符串是替换后的结果。例如，想象一下你有一个旧系统，其中有两种类型的订单。一些以破折号开头，另一些以字母开头：

+   -1234

+   A193, B123, C124

你必须将其更改为以下内容：

+   A1234

+   B193, B123, B124

简而言之，以破折号开头的应该以 A 开头，其余的应该以 B 开头。

```py
>>>def normalize_orders(matchobj):
       if matchobj.group(1) == '-': return "A"
       else: return "B"

>>> re.sub('([-|A-Z])', normalize_orders, '-1234⇢A193⇢ B123')
'A1234⇢B193⇢B123'
```

如前所述，对于每个匹配的模式，都会调用`normalize_orders`函数。因此，如果第一个匹配的组是`–`，那么我们返回`A`；在任何其他情况下，我们返回`B`。

请注意，在代码中，我们使用索引 1 获取第一个组；看一下`group`操作，以了解原因。

反向引用，也是`sub`提供的一个强大功能。我们将在下一章中深入了解它们。基本上，它的作用是用相应的组替换反向引用。例如，假设你想要将 markdown 转换为 HTML，为了简化示例，只需将文本加粗：

```py
>>> text = "imagine⇢a⇢new⇢*world*,⇢a⇢magic⇢*world*"
>>> pattern = re.compile(r'\*(.*?)\*')
>>> pattern.sub(r"<b>\g<1><\\b>", text)
'imagine⇢a⇢new⇢<b>world<\\b>,⇢a⇢magic⇢<b>world<\\b>'
```

和往常一样，前面的例子首先编译了模式，它匹配两个`*`之间的每个单词，并且捕获了这个单词。请注意，由于`?`元字符，模式是非贪婪的。

请注意，`\g<number>`是为了避免与字面数字产生歧义，例如，想象一下，你需要在一个组后面添加"1"：

```py
>>> pattern = re.compile(r'\*(.*?)\*')
>>> pattern.sub(r"<b>\g<1>1<\\b>", text)
   'imagine⇢a⇢new⇢<b>world1<\\b>,⇢a⇢magic⇢<b>world1<\\b>'
```

正如你所看到的，行为是符合预期的。让我们看看在使用没有`<`和`>`的符号时会发生什么：

```py
>>> text = "imagine⇢a⇢new⇢*world*,⇢a⇢magic⇢*world*"
>>> pattern = re.compile(r'\*(.*?)\*')
>>> pattern.sub(r"<b>**\g1
1**<\\b>", text)
 error: bad group name
```

在前面的示例中，突出显示了该组以消除歧义并帮助我们看到它，这正是正则表达式引擎所面临的问题。在这里，正则表达式引擎尝试使用不存在的第 11 组。因此，有`\g<group>`表示法。

`sub`的另一点需要记住的是，替换字符串中的每个反斜杠都将被处理。正如你在`<\\b>`中看到的，如果你想避免它，你需要对它们进行转义。

您可以使用可选的**count**参数限制替换的次数。

#### subn(repl, string, count=0)

它基本上与`sub`相同，你可以将它视为`sub`的一个实用程序。它返回一个包含新字符串和替换次数的元组。让我们通过使用与之前相同的示例来看一下它的工作：

```py
>>> text = "imagine⇢a⇢new⇢*world*,⇢a⇢magic⇢*world*"
>>> pattern = re.compile(r'\*(.*?)\*')
>>> pattern.subn(r"<b>\g<1><\\b>", text)
('imagine⇢a⇢new⇢<b>world<\\b>,⇢a⇢magic⇢<b>world<\\b>', 2)
```

这是一个很长的部分。我们探讨了我们可以使用`re`模块和`RegexObject`类进行的主要操作以及示例。让我们继续讨论匹配后得到的对象。

## MatchObject

这个对象代表了匹配的模式；每次执行这些操作时都会得到一个：

+   match

+   search

+   finditer

这个对象为我们提供了一组操作，用于处理捕获的组，获取有关匹配位置的信息等。让我们看看最重要的操作。

### group([group1, …])

`group`操作给出了匹配的子组。如果没有参数或零调用它，它将返回整个匹配；而如果传递一个或多个组标识符，则将返回相应组的匹配。

让我们用一个例子来看看：

```py
>>> pattern = re.compile(r"(\w+) (\w+)")
>>> match = pattern.search("Hello⇢world")
```

模式匹配整个字符串并捕获两个组，`Hello`和`world`。一旦我们有了匹配，我们可以看到以下具体情况：

+   没有参数或零，它返回整个匹配。

```py
>>> match.group()
'Hello⇢world'

>>> match.group(0)
'Hello⇢world'
```

+   使用`group1`大于 0，它返回相应的组。

```py
>>> match.group(1)
'Hello'

>>> match.group(2)
'world'
```

+   如果该组不存在，将抛出`IndexError`。

```py
>>> match.group(3)
…
IndexError: no such group
```

+   使用多个参数，它返回相应的组。

```py
>>> match.group(0, 2)
   ('Hello⇢world', 'world')
```

在这种情况下，我们想要整个模式和第二组，这就是为什么我们传递`0`和`2`。

组可以被命名，我们将在下一章中深入讨论；有一个特殊的表示方法。如果模式有命名组，可以使用名称或索引来访问它们：

```py
>>> pattern = re.compile(r"(?P<first>\w+) (?P<second>\w+)")
```

在前面的示例中，我们编译了一个模式来捕获两个组：第一个命名为`first`，第二个命名为`second`。

```py
>>> match = pattern.search("Hello⇢world")
>>> match.group('first')
'Hello'
```

通过这种方式，我们可以通过名称获取组。请注意，使用命名组，我们仍然可以通过它们的索引获取组，就像下面的代码中所示：

```py
>>> match.group(1)
'Hello'
```

我们甚至可以同时使用两种类型：

```py
>>> match.group(0, 'first', 2)
('Hello⇢world', 'Hello', 'world')
```

### groups([default])

`groups`操作类似于前面的操作。但是，在这种情况下，它返回一个包含匹配中所有子组的元组，而不是给出一个或一些组。让我们用前一节中使用的例子来看一下：

```py
>>> pattern = re.compile("(\w+) (\w+)")
>>> match = pattern.search("Hello⇢World")
>>> match.groups()
   ('Hello', 'World')
```

就像我们在前一节中看到的那样，我们有两个组`Hello`和`World`，这正是`groups`给我们的。在这种情况下，您可以将`groups`视为`group(1, lastGroup)`。

如果有不匹配的组，将返回默认参数。如果未指定默认参数，则使用`None`，例如：

```py
>>> pattern = re.compile("(\w+) (\w+)**?**")
>>> match = pattern.search("Hello⇢")
>>> match.groups("**mundo**")
   ('Hello', 'mundo')
>>> match.groups()
   ('Hello', **None**)
```

前面的示例中的模式试图匹配由一个或多个字母数字字符组成的两个组。第二个是可选的；所以我们只得到一个包含字符串`Hello`的组。在获得匹配后，我们调用`groups`，将`default`设置为`mundo`，这样它就返回`mundo`作为第二组。请注意，在下面的调用中，我们没有设置默认值，因此返回`None`。

### groupdict([default])

`groupdict`方法用于已使用命名组的情况。它将返回一个包含所有找到的组的字典：

```py
>>> pattern = re.compile(r"(?P<first>\w+) (?P<second>\w+)")
>>> pattern.search("Hello⇢world").groupdict()
{'first': 'Hello', 'second': 'world'}
```

在前面的示例中，我们使用了与前几节中看到的类似的模式。它使用名称为`first`和`second`的两个组进行捕获。因此，`groupdict`以字典形式返回它们。请注意，如果没有命名组，则它将返回一个空字典。

如果您不太明白这里发生了什么，不要担心。正如我们之前提到的，我们将在第三章中看到与分组相关的所有内容，*分组*。

### start（[组]）

有时，知道模式匹配的索引位置是有用的。与所有与组相关的操作一样，如果参数组为零，则该操作将使用匹配的整个字符串：

```py
>>> pattern = re.compile(r"(?P<first>\w+) (?P<second>\w+)?")
>>> match = pattern.search("Hello⇢")
>>> match.start(1)
0
```

如果有不匹配的组，则返回`-1`：

```py
>>> math = pattern.search("Hello⇢")
>>> match..start(2)
-1
```

### end（[组]）

`end`操作的行为与`start`完全相同，只是它返回与组匹配的子字符串的结尾：

```py
>>> pattern = re.compile(r"(?P<first>\w+) (?P<second>\w+)?")
>>> match = pattern.search("Hello⇢")
>>> match.end (1)
5
```

### span（[组]）

这是一个操作，它给出一个包含`start`和`end`的元组。这个操作经常用于文本编辑器中来定位和突出显示搜索。以下代码是这个操作的一个示例：

```py
>>> pattern = re.compile(r"(?P<first>\w+) (?P<second>\w+)?")
>>> match = pattern.search("Hello⇢")
>>> match.span(1)
(0, 5)
```

### 扩展（模板）

此操作返回替换模板字符串后的字符串。它类似于`sub`。

继续上一节的示例：

```py
>>> text = "imagine⇢a⇢new⇢*world*,⇢a⇢magic⇢*world*"
>>> match = re.search(r'\*(.*?)\*', text)
>>> match.expand(r"<b>\g<1><\\b>")
  '<b>world<\\b>'
```

## 模块操作

让我们看看模块中的两个有用的操作。

### 转义（）

它转义可能出现在表达式中的文字。

```py
>>> re.findall(re.escape("^"), "^like^")
['^', '^']
```

### 清除（）

它清除正则表达式缓存。我们已经谈论过这一点；当您通过模块使用操作时，您需要使用它以释放内存。请记住，这会影响性能；一旦释放缓存，每个模式都必须重新编译和缓存。

干得好，你已经知道了你可以用`re`模块做的主要操作。之后，你可以在项目中开始使用正则表达式而不会遇到太多问题。

现在，我们将看到如何更改模式的默认行为。

# 编译标志

在将模式字符串编译为模式对象时，可以修改模式的标准行为。为了做到这一点，我们必须使用编译标志。这些可以使用按位或`|`组合。

| 标志 | Python | 描述 |
| --- | --- | --- |
| `re.IGNORECASE`或`re.I` | 2.x3.x | 该模式将匹配小写和大写。 |

| `re.MULTILINE`或`re.M` | 2.x3.x | 此标志更改了两个元字符的行为：

+   `^`：现在匹配字符串的开头和每一行的开头。

+   `$`：在这种情况下，它匹配字符串的结尾和每一行的结尾。具体来说，它匹配换行符之前的位置。

|

| `re.DOTALL`或`re.S` | 2.x3.x | 元字符“。”将匹配任何字符，甚至包括换行符。 |
| --- | --- | --- |
| `re.LOCALE`或`re.L` | 2.x3.x | 此标志使\w、\W、\b、\B、\s 和\S 依赖于当前区域设置。“re.LOCALE 只是将字符传递给底层的 C 库。它实际上只适用于每个字符有 1 个字节的字节串。UTF-8 将 ASCII 范围之外的码点编码为每个码点多个字节，re 模块将把这些字节中的每一个都视为单独的字符。”（在[`www.gossamer-threads.com/lists/python/python/850772`](http://www.gossamer-threads.com/lists/python/python/850772)）请注意，当使用`re.L`和`re.U`一起时（re.L&#124;re.U，只使用区域设置）。另请注意，在 Python 3 中，不鼓励使用此标志；请查看文档以获取更多信息。 |

| `re.VERBOSE`或`re.X` | 2.x3.x | 它允许编写更易于阅读和理解的正则表达式。为此，它以一种特殊的方式处理一些字符：

+   忽略空格，除非它在字符类中或者在反斜杠之前

+   #右侧的所有字符都被忽略，就像是注释一样，除非#之前有反斜杠或者它在字符类中。

|

| `re.DEBUG` | 2.x3.x | 它为您提供有关编译模式的信息。 |
| --- | --- | --- |
| `re.UNICODE`或`re.U` | 2.x | 它使\w、\W、\b、\B、\d、\D、\s 和\S 依赖于 Unicode 字符属性数据库。 |
| `re.ASCII`或`re.A`（仅 Python 3） | 3.x | 它使\w、\W、\b、\B、\d、\D、\s 和\S 执行仅 ASCII 匹配。这是有道理的，因为在 Python 3 中，默认情况下匹配是 Unicode 的。您可以在“Python 3 的新功能”部分中找到更多信息。 |

让我们看一些最重要的标志示例。

## re.IGNORECASE 或 re.I

正如您所看到的，以下模式匹配，即使字符串以 A 开头而不是 a 开头。

```py
>>> pattern = re.compile(r"[a-z]+", re.I)
>>> pattern.search("Felix")
<_sre.SRE_Match at 0x10e27a238>
>>> pattern.search("felix")
<_sre.SRE_Match at 0x10e27a510>
```

## re.MULTILINE 或 re.M

在下面的示例中，模式不匹配换行符后的日期，因为我们没有使用标志：

```py
>>> pattern = re.compile("^\w+\: (\w+/\w+/\w+)")
>>> pattern.findall("date: ⇢12/01/2013 \ndate: 11/01/2013")
['12/01/2013']
```

但是，使用“多行”标志时，它匹配了两个日期：

```py
>>> pattern = re.compile("^\w+\: (\w+/\w+/\w+)", re.M)
>>> pattern.findall("date: ⇢12/01/2013⇢\ndate: ⇢11/01/2013")
  ['12/01/2013', '12/01/2013']
```

### 注意

这不是捕获日期的最佳方法。

## re.DOTALL 或 re.S

让我们尝试匹配数字后的任何内容：

```py
>>> re.findall("^\d(.)", "1\ne")
   []
```

我们可以在前面的例子中看到，具有默认行为的字符类“。”不匹配换行符。让我们看看使用标志会发生什么：

```py
>>> re.findall("^\d(.)", "1\ne", re.S)
['\n']

```

预期的是，使用`DOTALL`标志后，它完美地匹配了换行符。

## re.LOCALE 或 re.L

在下面的示例中，我们首先获取了前 256 个字符，然后尝试在字符串中找到每个字母数字字符，因此我们得到了预期的字符，如下所示：

```py
>>> chars = ''.join(chr(i) for i in xrange(256))
>>> " ".join(re.findall(r"\w", chars))
'0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z _ a b c d e f g h i j k l m n o p q r s t u v w x y z'   
```

在将区域设置为我们的系统区域设置后，我们可以再次尝试获取每个字母数字字符：

```py
>>> locale.setlocale(locale.LC_ALL, '')
'ru_RU.KOI8-R'  
```

在这种情况下，根据新的区域设置，我们得到了更多的字符：

```py
>>> " ".join(re.findall(r"\w", chars, re.LOCALE))
'0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z _ a b c d e f g h i j k l m n o p q r s t u v w x y z \xa3 \xb3 \xc0 \xc1 \xc2 \xc3 \xc4 \xc5 \xc6 \xc7 \xc8 \xc9 \xca \xcb \xcc \xcd \xce \xcf \xd0 \xd1 \xd2 \xd3 \xd4 \xd5 \xd6 \xd7 \xd8 \xd9 \xda \xdb \xdc \xdd \xde \xdf \xe0 \xe1 \xe2 \xe3 \xe4 \xe5 \xe6 \xe7 \xe8 \xe9 \xea \xeb \xec \xed \xee \xef \xf0 \xf1 \xf2 \xf3 \xf4 \xf5 \xf6 \xf7 \xf8 \xf9 \xfa \xfb \xfc \xfd \xfe \xff'
```

## re.UNICODE 或 re.U

让我们尝试在字符串中找到所有字母数字字符：

```py
>>> re.findall("\w+", "this⇢is⇢an⇢example")
['this', 'is', 'an', 'example']
```

但是，如果我们想要在其他语言中执行相同的操作会发生什么呢？字母数字字符取决于语言，因此我们需要将其指示给正则表达式引擎：

```py
>>> re.findall(ur"\w+", u"这是一个例子", re.UNICODE)
  [u'\u8fd9\u662f\u4e00\u4e2a\u4f8b\u5b50']
>>> re.findall(ur"\w+", u"هذا مثال", re.UNICODE)
   [u'\u0647\u0630\u0627', u'\u0645\u062b\u0627\u0644']
```

## re.VERBOSE 或 re.X

在下面的模式中，我们使用了几个⇢；第一个被忽略，因为它不在字符类中，也没有在反斜杠之前，第二个是模式的一部分。我们还使用了#三次，第一个和第三个被忽略，因为它们没有在反斜杠之前，第二个是模式的一部分。

```py
>>> pattern = re.compile(r"""[#|_] + #comment
              \ \# #comment
              \d+""", re.VERBOSE)
>>> pattern.findall("#⇢#2")
['#⇢#2']
```

## re.DEBUG

```py
>>>re.compile(r"[a-f|3-8]", re.DEBUG)
  in
    range (97, 102)
    literal 124
    range (51, 56)
```

# Python 和正则表达式的特殊考虑

在本节中，我们将回顾与其他版本的差异，如何处理 Unicode，以及 Python 2.x 和 Python 3 之间的`re`模块的差异。

## Python 和其他版本之间的差异

正如我们在本书开头提到的，re 模块具有 Perl 风格的正则表达式。但是，这并不意味着 Python 支持 Perl 引擎具有的每个功能。

有太多的差异无法在这样一本简短的书中涵盖，如果您想深入了解它们，这里有两个很好的起点：

+   [`en.wikipedia.org/wiki/Comparison_of_regular_expression_engines`](http://en.wikipedia.org/wiki/Comparison_of_regular_expression_engines)

+   [`www.regular-expressions.info/reference.html`](http://www.regular-expressions.info/reference.html)

## Unicode

当您使用 Python 2.x 并且要匹配 Unicode 时，正则表达式必须是 Unicode 转义。例如：

```py
>>> re.findall(r"\u03a9", u"adeΩa")
[]
>>> re.findall(ur"\u03a9", u"adeΩa")
[u'\u03a9']
```

请注意，如果您使用 Unicode 字符，但您使用的字符串类型不是 Unicode，则 Python 会自动使用默认编码对其进行编码。例如，在我的情况下，我有 UTF-8：

```py
>>> u"Ω".encode("utf-8")
'\xce\xa9'
>>> "Ω"
'\xce\xa9'
```

因此，在混合类型时，您必须小心：

```py
>>> re.findall(r'Ω', "adeΩa")
['\xce\xa9']
```

在这里，您不是匹配 Unicode，而是默认编码中的字符：

```py
>>> re.findall(r'\xce\xa9', "adeΩa")
['\xce\xa9']
```

因此，如果您在其中任何一个中使用 Unicode，则您的模式将不匹配任何内容：

```py
>>> re.findall(r'Ω', u"adeΩa")
[]
```

另一方面，您可以在两侧使用 Unicode，并且它将按预期进行匹配：

```py
>>> re.findall(ur'Ω', u"adeΩa")
   [u'\u03a9']
```

`re`模块不执行 Unicode 大小写折叠，因此在 Unicode 上不起作用：

```py
>>> re.findall(ur"ñ" ,ur"Ñ", re.I)
[]
```

## Python 3 中的新功能

Python 3 中有一些影响正则表达式行为的变化，并且已经向`re`模块添加了新功能。首先，让我们回顾一下字符串表示法如何发生变化。

| 类型 | 前缀 | 描述 |
| --- | --- | --- |

| 字符串 |   | 它们是字符串文字。它们是 Unicode。反斜杠是必要的，用于转义有意义的字符。

```py
>>>"España \n"
'España \n'
```

|

| 原始字符串 | `r` 或 `R` | 它们与文字字符串相同，只是反斜杠被视为普通字符。

```py
>>>r"España \n"
'España \\n'
```

|

| 字节字符串 | `b` 或 `B` | 以字节表示的字符串。它们只能包含 ASCII 字符；如果字节大于 128，必须进行转义。

```py
>>> b"Espa\xc3\xb1a \n"
b'Espa\xc3\xb1a \n'
```

我们可以这样转换为 Unicode：

```py
>>> str(b"Espa\xc3\xb1a \n", "utf-8")
'España \n'
```

反斜杠是必要的，用于转义有意义的字符。

| 字节原始字符串 | `r` 或 `R` | 它们类似于字节字符串，但反斜杠被转义。

```py
>>> br"Espa\xc3\xb1a \n"
b'Espa\\xc3\\xb1a \\n'
```

因此，用于转义字节的反斜杠再次被转义，这使得它们转换为 Unicode 变得更加复杂：

```py
>>> str(br"Espa\xc3\xb1a \n", "utf-8")
'Espa\\xc3\\xb1a \\n'
```

|

| Unicode | `r` 或 `U` | `u`前缀在 Python 3 的早期版本中被移除，但在 3.3 版本中又被接受。它们与字符串相同。 |
| --- | --- | --- |

在 Python 3 中，文字字符串默认为 Unicode，这意味着不再需要使用 Unicode 标志。

```py
>>> re.findall(r"\w+", "这是一个例子")
  ['这是一个例子']
```

Python 3.3 ([`docs.python.org/dev/whatsnew/3.3.html`](http://docs.python.org/dev/whatsnew/3.3.html)) 添加了更多与 Unicode 相关的功能以及语言中对其处理的方式。例如，它增加了对完整代码点范围的支持，包括非 BMP ([`en.wikipedia.org/wiki/Plane_(Unicode)`](http://en.wikipedia.org/wiki/Plane_(Unicode)))。因此，例如：

+   在 Python 2.7 中：

```py
>>> re.findall(r".", u'\U0010FFFF')
[u'\udbff', u'\udfff'] 
```

+   在 Python 3.3.2 中：

```py
>>> re.findall(r".", u'\U0010FFFF')
['\U0010ffff']
```

正如我们在*编译标志*部分中看到的，已添加了 ASCII 标志。

在使用 Python 3 时需要注意的另一个重要方面与元字符有关。由于字符串默认为 Unicode，元字符也是如此，除非您使用 8 位模式或使用 ASCII 标志。

```py
>>> re.findall(r"\w+", "هذا⇢مثال")
['هذا', 'مثال'] 
>>> re.findall(r"\w+", "هذا⇢مثال word", re.ASCII)
['word']
```

在前面的例子中，不是 ASCII 的字符被忽略了。

请注意，Unicode 模式和 8 位模式不能混合使用。

在下面的例子中，我们试图将一个 8 位模式与 Unicode 字符串匹配，这就是为什么会抛出异常（请记住，在 Python 2.x 中可以工作）：

```py
>>> re.findall(b"\w+", b"hello⇢world")
[b'hello', b'world']
>>> re.findall(b"\w+", "hello world")
….
TypeError: can't use a bytes pattern on a string-like object
```

# 总结

这是一个很长的章节！我们在其中涵盖了很多内容。我们从 Python 中字符串的工作方式及其在 Python 2.x 和 Python 3.x 中的不同表示开始。之后，我们看了如何构建正则表达式，`re`模块提供给我们处理它们的对象和接口，以及搜索和修改字符串的最重要操作。我们还学习了如何通过`MatchObject`从模式中提取信息，例如匹配的位置或组。我们还学习了如何使用编译标志修改一些字符类和元字符的默认行为。最后，我们看到了如何处理 Unicode 以及在 Python 3.x 中可以找到的新功能。

在本章中，我们看到组是正则表达式的重要部分，`re`模块的许多操作都是为了与组一起使用。这就是为什么我们在下一章中深入讨论组。
