# 文本处理和正则表达式

在本章中，我们将学习有关文本处理和正则表达式的知识。文本处理是创建或修改文本的过程。Python 有一个非常强大的名为正则表达式的库，可以执行搜索和提取数据等任务。您将学习如何在文件中执行此操作，还将学习读取和写入文件。

我们将学习有关 Python 正则表达式和处理文本的`re`模块。我们将学习`re`模块的`match()`、`search()`、`findall()`和`sub()`函数。我们还将学习使用`textwrap`模块在 Python 中进行文本包装。最后，我们将学习有关 Unicode 字符。

在本章中，我们将涵盖以下主题：

+   文本包装

+   正则表达式

+   Unicode 字符串

# 文本包装

在本节中，我们将学习有关`textwrap` Python 模块。该模块提供了执行所有工作的`TextWrapper`类。`textwrap`模块用于格式化和包装纯文本。该模块提供了五个主要函数：`wrap()`、`fill()`、`dedent()`、`indent()`和`shorten()`。我们现在将逐一学习这些函数。

# wrap()函数

`wrap()`函数用于将整个段落包装成单个字符串。输出将是输出行的列表。

语法是`textwrap.wrap(text, width)`：

+   `text`：要包装的文本。

+   `width`：包装行的最大长度。默认值为`70`。

现在，我们将看到`wrap()`的一个示例。创建一个`wrap_example.py`脚本，并在其中写入以下内容：

```py
import textwrap

sample_string = '''Python is an interpreted high-level programming language for general-purpose programming. Created by Guido van Rossum and first released in 1991, Python has a design philosophy that emphasizes code readability, notably using significant whitespace.'''

w = textwrap.wrap(text=sample_string, width=30)
print(w)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 wrap_example.py
['Python is an interpreted high-', 'level programming language for', 'general-purpose programming.', 'Created by Guido van Rossum', 'and first released in', '1991, Python has a design', 'philosophy that emphasizes', 'code readability,  notably', 'using significant whitespace.']
```

在前面的示例中，我们使用了 Python 的`textwrap`模块。首先，我们创建了一个名为`sample_string`的字符串。接下来，使用`TextWrapper`类指定了宽度。然后，使用`wrap`函数将字符串包装到宽度为`30`。然后，我们打印了这些行。

# fill()函数

`fill()`函数与`textwrap.wrap`类似，只是它返回连接成单个以换行符分隔的字符串的数据。此函数将文本包装并返回包含包装文本的单个字符串。

此函数的语法是：

```py
textwrap.fill(text, width)
```

+   `text`：要包装的文本。

+   `width`：包装行的最大长度。默认值为`70`。

现在，我们将看到`fill()`的一个示例。创建一个`fill_example.py`脚本，并在其中写入以下内容：

```py
import textwrap  sample_string = '''Python is an interpreted high-level programming language.'''  w = textwrap.fill(text=sample_string, width=50) print(w)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 fill_example.py
Python is an interpreted high-level programming
language.
```

在前面的示例中，我们使用了`fill()`函数。过程与我们在`wrap()`中所做的相同。首先，我们创建了一个字符串变量。接下来，我们创建了`textwrap`对象。然后，我们应用了`fill()`函数。最后，我们打印了输出。

# dedent()函数

`dedent()`是`textwrap`模块的另一个函数。此函数从文本的每一行中删除常见的前导`空格`。

此函数的语法如下：

```py
 textwrap.dedent(text)
```

`text`是要`dedent`的文本。

现在，我们将看到`dedent()`的一个示例。创建一个`dedent_example.py`脚本，并在其中写入以下内容：

```py
import textwrap  str1 = ''' Hello Python World \tThis is Python 101 Scripting language\n Python is an interpreted high-level programming language for general-purpose programming. ''' print("Original: \n", str1) print()  t = textwrap.dedent(str1) print("Dedented: \n", t)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 dedent_example.py 
Hello Python World   This is Python 101
Scripting language

Python is an interpreted high-level programming language for general-purpose programming.
```

在前面的示例中，我们创建了一个`str1`字符串变量。然后我们使用`textwrap.dedent()`来删除常见的前导空格。制表符和空格被视为空格，但它们不相等。因此，唯一的常见空格，在我们的情况下是`tab`，被移除。

# indent()函数

`indent()`函数用于在文本的选定行开头添加指定的前缀。

此函数的语法是：

```py
 textwrap.indent(text, prefix)
```

+   `text`：主字符串

+   `prefix`：要添加的前缀

创建一个`indent_example.py`脚本，并在其中写入以下内容：

```py
import textwrap  str1 = "Python is an interpreted high-level programming language for general-purpose programming. Created by Guido van Rossum and first released in 1991, \n\nPython has a design philosophy that emphasizes code readability, notably using significant whitespace."  w = textwrap.fill(str1, width=30) i = textwrap.indent(w, '*') print(i)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 indent_example.py *Python is an interpreted high- *level programming language for *general-purpose programming. *Created by Guido van Rossum *and first released in 1991, *Python has a design philosophy *that emphasizes code *readability, notably using *significant whitespace.
```

在上面的示例中，我们使用了`textwrap`模块的`fill()`和`indent()`函数。首先，我们使用`fill`方法将数据存储到变量`w`中。接下来，我们使用了`indent`方法。使用`indent()`，输出中的每一行都将有一个`*`前缀。然后，我们打印了输出。

# shorten()函数

`textwrap`模块的这个函数用于将文本截断以适应指定的宽度。例如，如果您想要创建摘要或预览，请使用`shorten()`函数。使用`shorten()`，文本中的所有空格将被标准化为单个空格。

此函数的语法是：

```py
            textwrap.shorten(text, width)
```

现在我们将看一个`shorten()`的例子。创建一个`shorten_example.py`脚本，并在其中写入以下内容：

```py
import textwrap str1 = "Python is an interpreted high-level programming language for general-purpose programming. Created by Guido van Rossum and first released in 1991, \n\nPython has a design philosophy that emphasizes code readability, notably using significant whitespace." s = textwrap.shorten(str1, width=50) print(s)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 shorten_example.py Python is an interpreted high-level [...]
```

在上面的示例中，我们使用了`shorten()`函数来截断我们的文本，并将该文本适应指定的宽度。首先，所有空格都被截断为单个空格。如果结果适合指定的宽度，则结果将显示在屏幕上。如果不适合，则指定宽度的单词将显示在屏幕上，其余部分将放在占位符中。

# 正则表达式

在本节中，我们将学习 Python 中的正则表达式。正则表达式是一种专门的编程语言，它嵌入在 Python 中，并通过`re`模块提供给用户使用。我们可以定义要匹配的字符串集的规则。使用正则表达式，我们可以从文件、代码、文档、电子表格等中提取特定信息。

在 Python 中，正则表达式表示为`re`，可以通过`re`模块导入。正则表达式支持四种功能：

+   标识符

+   修饰符

+   空白字符

+   标志

下表列出了标识符，并对每个标识符进行了描述：

| **标识符** | **描述** |
| --- | --- |
| `\w` | 匹配字母数字字符，包括下划线(`_`) |
| `\W` | 匹配非字母数字字符，不包括下划线(`_`) |
| `\d` | 匹配数字 |
| `\D` | 匹配非数字 |
| `\s` | 匹配空格 |
| `\S` | 匹配除空格之外的任何字符 |
| `.` | 匹配句号(`.`) |
| `\b` | 匹配除换行符之外的任何字符 |

下表列出了修饰符，并对每个修饰符进行了描述：

| **修饰符** | **描述** |
| --- | --- |
| `^` | 匹配字符串的开头 |
| `$` | 匹配字符串的结尾 |
| `?` | 匹配`0`或`1` |
| `*` | 匹配`0`或更多 |
| `+` | 匹配`1`或更多 |
| `&#124;` | 匹配`x/y`中的任意一个 |
| `[ ]` | 匹配范围 |
| `{x}` | 前置代码的数量 |

下表列出了空白字符，并对每个字符进行了描述：

| **字符** | **描述** |
| --- | --- |
| `\s` | 空格 |
| `\t` | 制表符 |
| `\n` | 换行 |
| `\e` | 转义 |
| `\f` | 换页符 |
| `\r` | 回车 |

下表列出了标志，并对每个标志进行了描述：

| **标志** | **描述** |
| --- | --- |
| `re.IGNORECASE` | 不区分大小写匹配 |
| `re.DOTALL` | 匹配包括换行符在内的任何字符 |
| `re.MULTILINE` | 多行匹配 |
| `Re.ASCII` | 仅使转义匹配 ASCII 字符 |

现在我们将看一些正则表达式的示例。我们将学习`match()`、`search()`、`findall()`和`sub()`函数。

要在 Python 中使用正则表达式，必须在脚本中导入`re`模块，以便能够使用正则表达式的所有函数和方法。

现在我们将逐一学习这些功能。

# match()函数

`match()`函数是`re`模块的一个函数。此函数将使用指定的`re`模式与字符串匹配。如果找到匹配项，将返回一个`match`对象。`match`对象将包含有关匹配的信息。如果找不到匹配项，我们将得到结果为`None`。`match`对象有两种方法：

+   `group(num)`: 返回整个匹配

+   `groups()`: 返回一个元组中的所有匹配子组

这个函数的语法如下：

```py
re.match(pattern, string)
```

现在，我们要看一个`re.match()`的例子。创建一个`re_match.py`脚本，并在其中写入以下内容：

```py
import re  str_line = "This is python tutorial. Do you enjoy learning python ?" obj = re.match(r'(.*) enjoy (.*?) .*', str_line) if obj:
 print(obj.groups())
```

运行脚本，你会得到以下输出：

```py
student@ubuntu:~/work$ python3 re_match.py
('This is python tutorial. Do you', 'learning')
```

在前面的脚本中，我们导入了`re`模块以在 Python 中使用正则表达式。然后我们创建了一个`str_line`字符串。接下来，我们创建了一个`obj`匹配对象，并将匹配模式的结果存储在其中。在这个例子中，`(.*) enjoy (.*?) .*`模式将打印`enjoy`关键字之前的所有内容，并且只会打印`enjoy`关键字之后的一个单词。接下来，我们使用了`match`对象的`groups()`方法。它将以元组的形式打印所有匹配的子字符串。因此，你将得到的输出是，`('This is python tutorial. Do you', 'learning')`。

# search()函数

`re`模块的`search()`函数将在字符串中搜索。它将寻找指定的`re`模式的任何位置。`search()`将接受一个模式和文本，并在我们指定的字符串中搜索匹配项。当找到匹配项时，它将返回一个`match`对象。如果找不到匹配项，它将返回`None`。`match`对象有两个方法：

+   `group(num)`: 返回整个匹配

+   `groups()`: 返回一个元组中的所有匹配子组

这个函数的语法如下：

```py
re.search(pattern, string)
```

创建一个`re_search.py`脚本，并在其中写入以下内容：

```py
import re pattern = ['programming', 'hello'] str_line = 'Python programming is fun' for p in pattern:
 print("Searching for %s in %s" % (p, str_line)) if re.search(p, str_line): print("Match found") else: print("No match found")
```

运行脚本，你会得到以下输出：

```py
student@ubuntu:~/work$ python3 re_search.py Searching for programming in Python programming is fun Match found Searching for hello in Python programming is fun No match found
```

在前面的例子中，我们使用了`match`对象的`search()`方法来查找`re`模式。在导入 re 模块之后，我们在列表中指定了模式。在那个列表中，我们写了两个字符串：`programming`和`hello`。接下来，我们创建了一个字符串：`Python programming is fun`。我们写了一个 for 循环，它将逐个检查指定的模式。如果找到匹配项，将执行`if`块。如果找不到匹配项，将执行`else`块。

# findall()函数

这是`match`对象的方法之一。`findall()`方法找到所有匹配项，然后将它们作为字符串列表返回。列表的每个元素表示一个匹配项。此方法搜索模式而不重叠。

创建一个`re_findall_example.py`脚本，并在其中写入以下内容：

```py
import re pattern = 'Red' colors = 'Red, Blue, Black, Red, Green' p = re.findall(pattern, colors) print(p) str_line = 'Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?' pt = re.findall('pe\w+', str_line) pt1 = re.findall('pic\w+', str_line) print(pt) print(pt1) line = 'Hello hello HELLO bye' p = re.findall('he\w+', line, re.IGNORECASE) print(p)
```

运行脚本，你会得到以下输出：

```py
student@ubuntu:~/work$ python3 re_findall_example.py
['Red', 'Red']
['per', 'peck', 'peppers', 'peppers', 'per']
['picked', 'pickled', 'pickled', 'pick']
['Hello', 'hello', 'HELLO']
```

在前面的脚本中，我们写了`findall()`方法的三个例子。在第一个例子中，我们定义了一个模式和一个字符串。我们使用`findall()`方法从字符串中找到该模式，然后打印它。在第二个例子中，我们创建了一个字符串，然后使用`findall()`找到前两个字母是`pe`的单词，并打印它们。我们将得到前两个字母是`pe`的单词列表。

此外，我们找到了前三个字母是`pic`的单词，然后打印它们。在这里，我们也会得到字符串列表。在第三个例子中，我们创建了一个字符串，在其中我们指定了大写和小写的`hello`，还有一个单词：`bye`。使用`findall()`，我们找到了前两个字母是`he`的单词。同样在`findall()`中，我们使用了一个`re.IGNORECASE`标志，它会忽略单词的大小写并打印它们。

# sub()函数

这是 re 模块中最重要的函数之一。`sub()`用于用指定的替换字符串替换`re`模式。它将用替换字符串替换`re`模式的所有出现。语法如下：

```py
   re.sub(pattern, repl_str, string, count=0)
```

+   `pattern`: `re`模式。

+   `repl_str`: 替换字符串。

+   `string`: 主字符串。

+   `count`: 要替换的出现次数。默认值为`0`，表示替换所有出现。

现在我们要创建一个`re_sub.py`脚本，并在其中写入以下内容：

```py
import re

str_line = 'Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?'

print("Original: ", str_line)
p = re.sub('Peter', 'Mary', str_line)
print("Replaced: ", p)

p = re.sub('Peter', 'Mary', str_line, count=1)
print("Replacing only one occurrence of Peter… ")
print("Replaced: ", p)
```

运行脚本，你会得到以下输出：

```py
student@ubuntu:~/work$ python3 re_sub.py
Original:  Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?
Replaced:  Mary Piper picked a peck of pickled peppers. How many pickled peppers did Mary Piper pick?
Replacing only one occurrence of Peter...
Replaced:  Mary Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?
```

在上面的例子中，我们使用`sub()`来用指定的替换字符串替换`re`模式。我们用 Mary 替换了 Peter。所以，所有的 Peter 都将被替换为 Mary。接下来，我们还包括了`count`参数。我们提到了`count=1`：这意味着只有一个 Peter 的出现将被替换，其他的 Peter 的出现将保持不变。

现在，我们将学习 re 模块的`subn()`函数。`subn()`函数与`sub()`的功能相同，但还有额外的功能。`subn()`函数将返回一个包含新字符串和执行的替换次数的元组。让我们看一个`subn()`的例子。创建一个`re_subn.py`脚本，并在其中写入以下内容：

```py
import re

print("str1:- ")
str1 = "Sky is blue. Sky is beautiful."

print("Original: ", str1)
p = re.subn('beautiful', 'stunning', str1)
print("Replaced: ", p)
print()

print("str_line:- ")
str_line = 'Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?'

print("Original: ", str_line)
p = re.subn('Peter', 'Mary', str_line)
print("Replaced: ", p)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 re_subn.py
str1:-
Original:  Sky is blue. Sky is beautiful.
Replaced:  ('Sky is blue. Sky is stunning.', 1)

str_line:-
Original:  Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?
Replaced:  ('Mary Piper picked a peck of pickled peppers. How many pickled peppers did Mary Piper pick?', 2)
```

在上面的例子中，我们使用了`subn()`函数来替换 RE 模式。结果，我们得到了一个包含替换后的字符串和替换次数的元组。

# Unicode 字符串

在本节中，我们将学习如何在 Python 中打印 Unicode 字符串。Python 以一种非常简单的方式处理 Unicode 字符串。字符串类型实际上保存的是 Unicode 字符串，而不是字节序列。

在您的系统中启动`python3`控制台，并开始编写以下内容：

```py
student@ubuntu:~/work$ python3
Python 3.6.6 (default, Sep 12 2018, 18:26:19)
[GCC 8.0.1 20180414 (experimental) [trunk revision 259383]] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
>>> print ('\u2713')

>>> print ('\u2724')

>>> print ('\u2750')

>>> print ('\u2780')

>>> chinese = '\u4e16\u754c\u60a8\u597d!
>>> chinese
![](img/5088de25-a7d1-4cde-8821-03151178533d.png) ----- (Meaning “Hello world!”)
>>>
>>> s = '\u092E\u0941\u0902\u092C\u0908'
>>> s
'मुंबई'                            ------(Unicode translated in Marathi)
>>>
>>> s = '\u10d2\u10d0\u10db\u10d0\u10e0\u10ef\u10dd\u10d1\u10d0'
>>> s
'გამარჯობა'                 ------(Meaning “Hello” in Georgian)
>>>
>>> s = '\u03b3\u03b5\u03b9\u03b1\u03c3\u03b1\u03c2'
>>> s
'γειασας'                     ------(Meaning “Hello” in Greek)
>>> 
```

# Unicode 代码点

在本节中，我们将学习 Unicode 代码点。Python 有一个强大的内置函数`ord()`，用于从给定字符获取 Unicode 代码点。因此，让我们看一个从字符获取 Unicode 代码点的例子，如下所示：

```py
>>> str1 = u'Office'
>>> for char in str1:
... print('U+%04x' % ord(char))
...
U+004f
U+0066
U+0066
U+0069
U+0063
U+0065
>>> str2 = ![](img/89322582-8eec-4421-a610-31da6e1876bf.png)
>>> for char in str2:
... print('U+%04x' % ord(char))
...
U+4e2d
U+6587

```

# 编码

从 Unicode 代码点到字节字符串的转换称为编码。因此，让我们看一个将 Unicode 代码点编码的例子，如下所示：

```py
>>> str = u'Office'
>>> enc_str = type(str.encode('utf-8'))
>>> enc_str
<class 'bytes'>
```

# 解码

从字节字符串到 Unicode 代码点的转换称为解码。因此，让我们看一个将字节字符串解码为 Unicode 代码点的例子，如下所示：

```py
>>> str = bytes('Office', encoding='utf-8')
>>> dec_str = str.decode('utf-8')
>>> dec_str
'Office'
```

# 避免 UnicodeDecodeError

每当字节字符串无法解码为 Unicode 代码点时，就会发生`UnicodeDecodeError`。为了避免这种异常，我们可以在`decode`的`error`参数中传递`replace`、`backslashreplace`或`ignore`，如下所示：

```py
>>> str = b"\xaf"
>>> str.decode('utf-8', 'strict')
 Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xaf in position 0: invalid start byte

>>> str.decode('utf-8', "replace")
'\ufffd'
>>> str.decode('utf-8', "backslashreplace")
'\\xaf'
>>> str.decode('utf-8', "ignore")
' '
```

# 摘要

在本章中，我们学习了正则表达式，使用它可以定义一组我们想要匹配的字符串的规则。我们学习了`re`模块的四个函数：`match()`、`search()`、`findall()`和`sub()`。

我们学习了`textwrap`模块，它用于格式化和包装纯文本。我们还学习了`textwrap`模块的`wrap()`、`fill()`、`dedent()`、`indent()`和`shorten()`函数。最后，我们学习了 Unicode 字符以及如何在 Python 中打印 Unicode 字符串。

在下一章中，我们将学习如何使用 Python 对信息进行标准文档化和报告。

# 问题

1.  Python 中的正则表达式是什么？

1.  编写一个 Python 程序来检查一个字符串是否只包含某个字符集（在本例中为`a–z`、`A–Z`和`0–9`）。

1.  Python 中的哪个模块支持正则表达式？

a) `re`

b) `regex`

c) `pyregex`

d) 以上都不是

1.  `re.match`函数的作用是什么？

a) 在字符串的开头匹配模式

b) 在字符串的任何位置匹配模式

c) 这样的函数不存在

d) 以上都不是

1.  以下的输出是什么？

句子："we are humans"

匹配：`re.match(r'(.*) (.*?) (.*)'`, `sentence)`

`print(matched.group())`

a) `('we', 'are', 'humans')`

b) `(we, are, humans)`

c) `('we', 'humans')`

d) `'we are humans'`

# 进一步阅读

+   正则表达式：[`docs.python.org/3.2/library/re.html`](https://docs.python.org/3.2/library/re.html)

+   Textwrap 文档：[`docs.python.org/3/library/textwrap.html`](https://docs.python.org/3/library/textwrap.html)

+   Unicode 文档：[`docs.python.org/3/howto/unicode.html`](https://docs.python.org/3/howto/unicode.html)
