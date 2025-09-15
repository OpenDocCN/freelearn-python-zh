# 字符串与序列化

在我们深入研究高级设计模式之前，让我们深入探讨 Python 中最常见的对象之一：字符串。我们会看到字符串远不止表面看起来那么简单，还会涵盖在字符串中搜索模式以及序列化数据以进行存储或传输。

尤其是我们将探讨以下主题：

+   字符串、字节和字节数组的复杂性

+   字符串格式化的来龙去脉

+   几种序列化数据的方法

+   神秘的正则表达式

# 字符串

字符串是 Python 中的一个基本原始类型；我们已经在迄今为止的几乎所有示例中都使用了它们。它们所做的只是表示一个不可变的字符序列。然而，尽管你可能之前没有考虑过，*字符*这个词有点模糊；Python 字符串能否表示带重音的字符序列？中文字符？那么希腊文、西里尔文或波斯文呢？

在 Python 3 中，答案是肯定的。Python 字符串全部以 Unicode 表示，这是一个字符定义标准，可以表示地球上任何语言的几乎所有字符（以及一些虚构语言和随机字符）。这是无缝完成的。因此，让我们将 Python 3 字符串视为一个不可变的 Unicode 字符序列。我们在之前的示例中已经触及了许多字符串操作的方法，但让我们在这里快速总结一下：字符串理论的快速入门！

# 字符串操作

如你所知，在 Python 中，可以通过将字符序列用单引号或双引号括起来来创建字符串。使用三个引号字符可以轻松创建多行字符串，并且可以通过将它们并排放置来连接多个硬编码的字符串。以下是一些示例：

```py
a = "hello" 
b = 'world' 
c = '''a multiple 
line string''' 
d = """More 
multiple""" 
e = ("Three " "Strings " 
        "Together") 
```

那最后一个字符串会被解释器自动组合成一个单一的字符串。也可以使用`+`运算符来连接字符串（如`"hello " + "world"`）。当然，字符串不必是硬编码的。它们也可以来自各种外部来源，例如文本文件、用户输入，或者可以在网络上编码。

当遗漏逗号时，相邻字符串的自动连接可能会产生一些令人捧腹的错误。然而，当需要将长字符串放入函数调用中而不超过 Python 风格指南建议的 79 个字符行长度限制时，这却非常有用。

与其他序列一样，字符串可以逐个字符迭代（按字符索引），切片或连接。语法与列表相同。

`str`类上有许多方法来简化字符串操作。Python 解释器中的`dir`和`help`命令可以告诉我们如何使用它们的所有方法；我们将直接考虑一些更常见的方法。

几个布尔便利方法帮助我们确定字符串中的字符是否匹配某种模式。以下是这些方法的总结。其中大多数，如 `isalpha`、`isupper`/`islower`、`startswith`/`endswith`，都有明显的解释。`isspace` 方法也很明显，但请记住，所有空白字符（包括制表符和换行符）都被考虑在内，而不仅仅是空格字符。

`istitle` 方法返回 `True`，如果每个单词的首字母都大写且所有其他字母都小写。请注意，它并不严格遵循英语语法对标题格式的定义。例如，利·亨特的诗作《手套与狮子》应该是一个有效的标题，即使不是所有单词都大写。罗伯特·塞尔的《萨姆·麦基的火化》也应该是一个有效的标题，即使最后一个单词的中间有一个大写字母。

在使用 `isdigit`、`isdecimal` 和 `isnumeric` 方法时要小心，因为它们比我们预期的要复杂。除了我们习惯的 10 个数字之外，许多 Unicode 字符也被认为是数字。更糟糕的是，我们用来从字符串构造浮点数的点字符不被认为是十进制字符，所以 `'45.2'.isdecimal()` 返回 `False`。真正的十进制字符由 Unicode 值 0660 表示，如 45.2（或 `45\u06602`）。此外，这些方法不验证字符串是否是有效的数字；`127.0.0.1` 对所有三种方法都返回 `True`。我们可能会认为我们应该用那个十进制字符而不是点来表示所有的数值，但将那个字符传递给 `float()` 或 `int()` 构造函数会将那个十进制字符转换为零：

```py
>>> float('45\u06602')
4502.0  
```

所有这些不一致的结果是，布尔数值检查几乎没有任何用处。我们通常使用正则表达式（本章后面将讨论）来确认字符串是否匹配特定的数值模式会更好。

其他用于模式匹配的方法不返回布尔值。`count` 方法告诉我们给定子字符串在字符串中出现的次数，而 `find`、`index`、`rfind` 和 `rindex` 告诉我们在原始字符串中给定子字符串的位置。两个 `r`（代表 *right* 或 *reverse*）方法从字符串的末尾开始搜索。`find` 方法在找不到子字符串时返回 `-1`，而 `index` 在这种情况下会引发 `ValueError`。看看这些方法在实际中的应用：

```py
>>> s = "hello world"
>>> s.count('l')
3
>>> s.find('l')
2
>>> s.rindex('m')
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
ValueError: substring not found  
```

大多数剩余的字符串方法返回字符串的转换。`upper`、`lower`、`capitalize` 和 `title` 方法创建具有给定格式的新字符串，其中包含所有字母字符。`translate` 方法可以使用字典将任意输入字符映射到指定的输出字符。

对于所有这些方法，请注意输入字符串保持不变；而是返回一个新的`str`实例。如果我们需要操作结果字符串，我们应该将其分配给一个新的变量，例如`new_value = value.capitalize()`。通常，一旦我们完成了转换，我们就不再需要旧值了，所以一个常见的习惯是将它分配给同一个变量，例如`value = value.title()`。

最后，有一些字符串方法返回或操作列表。`split`方法接受一个子字符串，并将字符串分割成一个字符串列表，其中该子字符串出现的位置。你可以传递一个数字作为第二个参数来限制结果字符串的数量。`rsplit`方法在没有限制字符串数量时与`split`行为相同，但如果你提供了限制，它将从字符串的末尾开始分割。`partition`和`rpartition`方法仅在子字符串的第一个或最后一个出现处分割字符串，并返回一个包含三个值的元组：子字符串之前的字符、子字符串本身以及子字符串之后的字符。

作为`split`的逆操作，`join`方法接受一个字符串列表，并返回所有这些字符串通过在它们之间放置原始字符串组合在一起。`replace`方法接受两个参数，并返回一个字符串，其中每个第一个参数的实例都被第二个参数替换。以下是一些这些方法在实际中的应用：

```py
>>> s = "hello world, how are you"
>>> s2 = s.split(' ')
>>> s2
['hello', 'world,', 'how', 'are', 'you']
>>> '#'.join(s2)
'hello#world,#how#are#you'
>>> s.replace(' ', '**')
'hello**world,**how**are**you'
>>> s.partition(' ')
('hello', ' ', 'world, how are you')  
```

就这样，我们快速浏览了`str`类中最常见的方法！现在，让我们看看 Python 3 的字符串和变量组合方法来创建新的字符串。

# 字符串格式化

Python 3 拥有强大的字符串格式化和模板机制，允许我们构建由硬编码文本和穿插变量组成的字符串。我们已经在许多之前的例子中使用过它，但它比我们使用的简单格式化说明符要灵活得多。

一个字符串可以通过在开引号前加上 f 来转换成一个格式字符串（也称为**f-string**），例如`f"hello world"`。如果这样的字符串包含特殊字符`{`和`}`，可以使用周围作用域中的变量来替换它们，如下例所示：

```py
name = "Dusty"
activity = "writing"
formatted = f"Hello {name}, you are currently {activity}."
print(formatted)
```

如果我们运行这些语句，它会按照以下顺序替换花括号中的变量：

```py
Hello Dusty, you are currently writing. 
```

# 转义花括号

花括号字符在字符串中除了格式化之外通常很有用。我们需要一种方法来在它们需要作为自身显示而不是被替换的情况下进行转义。这可以通过加倍花括号来实现。例如，我们可以使用 Python 格式化一个基本的 Java 程序：

```py
classname = "MyClass"
python_code = "print('hello world')"
template = f"""
public class {classname} {{
    public static void main(String[] args) {{
        System.out.println("{python_code}");
    }}
}}"""

print(template)
```

在模板中我们看到`{{`或`}}`序列——即包围 Java 类和方法定义的花括号——我们知道 f-string 将用单个花括号替换它们，而不是周围方法中的某个参数。以下是输出：

```py
public class MyClass {
 public static void main(String[] args) {
 System.out.println("print('hello world')");
 }
}  
```

输出类的名称和内容已被两个参数替换，而双花括号已被单花括号替换，从而生成一个有效的 Java 文件。结果证明，这是打印最简单的 Java 程序（该程序可以打印最简单的 Python 程序）的 Python 程序中最简单的一种。

# f-string 可以包含 Python 代码

我们不仅可以将简单的字符串变量传递给 f-string 方法。任何原始数据类型，如整数或浮点数，都可以进行格式化。更有趣的是，包括列表、元组、字典和任意对象在内的复杂对象也可以使用，并且我们可以在 `format` 字符串中访问这些对象的索引和变量或调用这些对象上的函数。

例如，如果我们的电子邮件消息将 `From` 和 `To` 电子邮件地址组合成一个元组，并将主题和消息放入一个字典中，出于某种原因（可能是因为我们需要使用现有的 `send_mail` 函数作为输入），我们可以这样格式化：

```py
emails = ("a@example.com", "b@example.com")
message = {
    "subject": "You Have Mail!",
    "message": "Here's some mail for you!",
}

formatted = f"""
From: <{emails[0]}>
To: <{emails[1]}>
Subject: {message['subject']}
{message['message']}"""
print(formatted)
```

模板字符串中花括号内的变量看起来有点奇怪，让我们看看它们在做什么。两个电子邮件地址是通过 `emails[x]` 查找的，其中 `x` 要么是 `0` 要么是 `1`。方括号内带有数字的索引查找与我们在常规 Python 代码中看到的是同一类型的，所以 `emails[0]` 指的是 `emails` 元组中的第一个元素。索引语法适用于任何可索引的对象，因此当我们访问 `message[subject]` 时，我们看到类似的行为，只不过这次我们在字典中查找一个字符串键。请注意，与 Python 代码不同，在字典查找中我们不需要在字符串周围加上引号。

如果我们有嵌套的数据结构，我们甚至可以进行多级查找。如果我们修改上面的代码，将 `emails` 元组放入 `message` 字典中，我们可以使用以下索引查找：

```py
message["emails"] = emails

formatted = f"""
From: <{message['emails'][0]}>
To: <{message['emails'][1]}>
Subject: {message['subject']}
{message['message']}"""
print(formatted)
```

我不建议经常这样做，因为模板字符串很快就会变得难以理解。

或者，如果您有一个对象或类，您可以在 f-string 中执行对象查找或甚至调用方法。让我们再次更改我们的电子邮件消息数据，这次到一个类：

```py
class EMail:
    def __init__(self, from_addr, to_addr, subject, message):
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.subject = subject
        self._message = message

    def message(self):
        return self._message

email = EMail(
    "a@example.com",
    "b@example.com",
    "You Have Mail!",
    "Here's some mail for you!",
)

formatted = f"""
From: <{email.from_addr}>
To: <{email.to_addr}>
Subject: {email.subject}

{email.message()}"""
print(formatted)
```

与之前的示例相比，这个示例中的模板可能更容易阅读，但创建一个 `email` 类的开销增加了 Python 代码的复杂性。为了将对象包含在模板中而创建一个类是愚蠢的。通常，我们会使用这种查找，如果我们试图格式化的对象已经存在。

几乎任何你期望返回字符串（或可以由 `str()` 函数转换为字符串的值）的 Python 代码都可以在 f-string 中执行。作为一个例子，看看它有多强大，你甚至可以在格式字符串参数中使用列表推导式或三元运算符：

```py
>>> f"['a' for a in range(5)]"
"['a' for a in range(5)]"
>>> f"{'yes' if True else 'no'}"
'yes'
```

# 使其看起来正确

能够在模板字符串中包含变量是件好事，但有时变量需要一点强制才能在输出中看起来像我们想要的那样。例如，如果我们正在执行货币计算，我们可能会得到一个我们不希望在模板中显示的长小数：

```py
subtotal = 12.32
tax = subtotal * 0.07
total = subtotal + tax

print(
    "Sub: ${0} Tax: ${1} Total: ${total}".format(
        subtotal, tax, total=total
    )
)
```

如果我们运行这段格式化代码，输出结果并不完全像正确的货币格式：

```py
Sub: $12.32 Tax: $0.8624 Total: $13.182400000000001
```

技术上，我们永远不应该在这种货币计算中使用浮点数；我们应该构建`decimal.Decimal()`对象。浮点数是危险的，因为它们的计算在特定精度水平以上是不准确的。但我们正在看字符串，而不是浮点数，货币是一个很好的格式化例子！

为了修复前面的`format`字符串，我们可以在大括号内包含一些额外的信息来调整参数的格式。我们可以自定义很多东西，但大括号内的基本语法是相同的。在提供模板值后，我们包括一个冒号，然后是一些特定的格式化语法。这是一个改进的版本：

```py

print(
    "Sub: ${0:0.2f} Tax: ${1:0.2f} "
    "Total: ${total:0.2f}".format(subtotal, tax, total=total)
)

```

冒号后面的`0.2f`格式说明符基本上是这样说的，从左到右：

+   `0`：对于小于一的值，确保在十进制点的左侧显示零

+   `.`：显示小数点

+   `2`：显示两位小数

+   `f`：将输入值格式化为浮点数

我们还可以指定每个数字应该在屏幕上占用特定数量的字符，通过在点号之前放置一个值。这可以用于输出表格数据，例如：

```py
orders = [("burger", 2, 5), ("fries", 3.5, 1), ("cola", 1.75, 3)]

print("PRODUCT QUANTITY PRICE SUBTOTAL")
for product, price, quantity in orders:
    subtotal = price * quantity
    print(
 f"{product:10s}{quantity: ⁹d} "
 f"${price: <8.2f}${subtotal: >7.2f}"
    )
```

好吧，这个格式字符串看起来相当吓人，所以在我们将其分解成可理解的部分之前，让我们先看看它是如何工作的：

```py
PRODUCT    QUANTITY    PRICE    SUBTOTAL
burger        5        $2.00    $  10.00
fries         1        $3.50    $   3.50
cola          3        $1.75    $   5.25  
```

真棒！那么，这实际上是如何发生的呢？我们有四个变量需要格式化，在`for`循环的每一行中。第一个变量是一个使用`{product:10s}`格式化的字符串。从右到左读起来更容易：

+   `s` 表示这是一个字符串变量。

+   `10` 表示它应该占用 10 个字符。默认情况下，对于字符串，如果字符串的长度小于指定的字符数，它会在字符串的右侧添加空格，使其足够长（但是请注意：如果原始字符串太长，它不会被截断！）。

+   `product:`, 当然，是正在格式化的变量或 Python 表达式的名称。

`quantity`值的格式化器是`{quantity: ⁹d}`。你可以从右到左这样解释这个格式：

+   `d` 代表一个整数值。

+   `9` 告诉我们值应该在屏幕上占用九个字符。

+   `^` 告诉我们数字应该在这个可用填充区的中心对齐；这使得列看起来更专业。

+   （空格）告诉格式化器使用空格作为填充字符。对于整数，默认情况下，额外的字符是零。

+   `quantity:` 是正在格式化的变量。

所有这些指定符都必须按照正确的顺序排列，尽管它们都是可选的：首先填充，然后对齐，接着是大小，最后是类型。

我们对`price`和`subtotal`的指定符做类似处理。对于`price`，我们使用`{2:<8.2f}`；对于`subtotal`，我们使用`{3:>7.2f}`。在这两种情况下，我们指定空格作为填充字符，但分别使用`<`和`>`符号来表示数字应在至少八或七个字符的最小空间内左对齐或右对齐。此外，每个浮点数应格式化为两位小数。

对于不同类型的*类型*字符可以影响格式化输出。我们已经看到了`s`、`d`和`f`类型，分别用于字符串、整数和浮点数。大多数其他格式指定符都是这些类型的替代版本；例如，`o`代表八进制格式，`X`代表十六进制格式，如果格式化整数。`n`类型指定符可以用于在当前区域设置的格式中格式化整数分隔符。对于浮点数，`%`类型将乘以 100 并将浮点数格式化为百分比。

# 自定义格式化程序

虽然这些标准格式化程序适用于大多数内置对象，但其他对象也可以定义非标准指定符。例如，如果我们将一个`datetime`对象传递给`format`，我们可以使用`datetime.strftime`函数中使用的指定符，如下所示：

```py
import datetime 
print("{the_date:%Y-%m-%d %I:%M%p }".format( 
    datetime.datetime.now())) 
```

甚至可以为我们自己创建的对象编写自定义格式化程序，但这超出了本书的范围。如果你需要在代码中这样做，请查看是否需要重写`__format__`特殊方法。

Python 的格式化语法非常灵活，但它是一个难以记住的小型语言。我每天都在使用它，仍然偶尔需要查阅文档中忘记的概念。它也不够强大，无法满足严肃的模板需求，例如生成网页。如果你需要做更多基本的字符串格式化之外的事情，可以查看几个第三方模板库。

# 格式化方法

有一些情况下你将无法使用 f-string。首先，你不能用不同的变量重用单个模板字符串。其次，f-string 是在 Python 3.6 中引入的。如果你卡在 Python 的旧版本上或需要重用模板字符串，可以使用较旧的`str.format`方法。它使用与 f-string 相同的格式指定符，但可以在一个字符串上多次调用。以下是一个示例：

```py
>>> template = "abc {number:*¹⁰d}"
>>> template.format(number=32)
'abc ****32****'
>>> template.format(number=84)
'abc ****84****'
```

`format`方法的行为与 f-string 类似，但有一些区别：

+   它在可以查找的内容上有限制。你可以访问对象上的属性或在列表或字典中查找索引，但不能在模板字符串内部调用函数。

+   你可以使用整数来访问传递给格式化方法的定位参数：`"{0} world".format('bonjour')`。如果你按顺序指定变量，索引是可选的：`"{} {}".format('hello', 'world')`。

# 字符串是 Unicode 编码

在本节的开始，我们将字符串定义为不可变 Unicode 字符的集合。这实际上在某些时候使事情变得非常复杂，因为 Unicode 并不是真正的存储格式。例如，如果你从一个文件或套接字中获取一个字节字符串，它们不会是 Unicode。实际上，它们将是内置类型`bytes`。字节是...字节的不可变序列。字节是计算中的基本存储格式。它们代表 8 位，通常描述为介于 0 和 255 之间的整数，或介于 0 和 FF 之间的十六进制等效值。字节不表示任何特定内容；字节序列可能存储编码字符串的字符，或图像中的像素。

如果我们打印一个字节对象，任何映射到 ASCII 表示的字节将被打印为其原始字符，而非 ASCII 字节（无论是二进制数据还是其他字符）将被打印为`\x`转义序列逃逸的十六进制代码。你可能觉得奇怪，一个表示为整数的字节可以映射到一个 ASCII 字符。但 ASCII 实际上是一种代码，其中每个字母都由不同的字节模式表示，因此，不同的整数。字符*a*由与整数 97 相同的字节表示，这是十六进制数 0x61。具体来说，这些都是对二进制模式 01100001 的解释。

许多 I/O 操作只知道如何处理`字节`，即使`字节`对象引用的是文本数据。因此，了解如何在`字节`和 Unicode 之间进行转换至关重要。

问题在于有许多方法可以将`字节`映射到 Unicode 文本。字节是机器可读的值，而文本是供人类阅读的格式。介于两者之间的是一种编码，它将给定的字节序列映射到给定的文本字符序列。

然而，存在多种这样的编码（ASCII 只是其中之一）。当使用不同的编码映射时，相同的字节序列会表示完全不同的文本字符！因此，`字节`必须使用与它们编码时相同的字符集进行解码。如果不了解字节应该如何解码，就无法从字节中获取文本。如果我们收到未指定编码的未知字节，我们最好的做法是猜测它们编码的格式，我们可能会出错。

# 将字节转换为文本

如果我们从某个地方有一个`字节`数组，我们可以使用`bytes`类的`.decode`方法将其转换为 Unicode。此方法接受一个字符串作为字符编码的名称。有许多这样的名称；用于西方语言的常见名称包括 ASCII、UTF-8 和 latin-1。

字节序列（以十六进制表示），63 6c 69 63 68 e9，实际上代表了拉丁-1 编码中单词 cliché的字符。以下示例将使用 latin-1 编码对这个字节序列进行编码，并将其转换为 Unicode 字符串：

```py
characters = b'\x63\x6c\x69\x63\x68\xe9' 
print(characters) 
print(characters.decode("latin-1")) 
```

第一行创建了一个`bytes`对象。类似于 f-string，字符串前面的`b`字符告诉我们我们正在定义一个`bytes`对象，而不是普通的 Unicode 字符串。在字符串中，每个字节都使用——在这种情况下——十六进制数指定。`\x`字符在字节字符串中转义，每个表示——在这种情况下——使用十六进制数字表示一个字节。

只要我们使用的 shell 支持 latin-1 编码，两个`print`调用将输出以下字符串：

```py
b'clich\xe9'
cliché  
```

第一个`print`语句将 ASCII 字符的字节渲染为其自身。未知（对 ASCII 而言）的字符保持其转义十六进制格式。输出包括行首的`b`字符，以提醒我们这是一个`bytes`表示，而不是字符串。

下一个调用使用 latin-1 编码解码字符串。`decode`方法返回一个带有正确字符的正常（Unicode）字符串。然而，如果我们使用西里尔文`iso8859-5`编码解码这个相同的字符串，我们最终会得到`'clichщ'`字符串！这是因为`\xe9`字节在这两种编码中映射到不同的字符。 

# 将文本转换为字节

如果我们需要将传入的字节转换为 Unicode，我们显然也会遇到将输出的 Unicode 转换为字节序列的情况。这是通过`str`类的`encode`方法完成的，它，就像`decode`方法一样，需要一个字符集。以下代码创建了一个 Unicode 字符串，并在不同的字符集中对其进行编码：

```py
characters = "cliché" 
print(characters.encode("UTF-8")) 
print(characters.encode("latin-1")) 
print(characters.encode("CP437")) 
print(characters.encode("ascii")) 
```

前三种编码为带音标的字符创建了一组不同的字节。第四种甚至无法处理该字节：

```py
    b'clich\xc3\xa9'
    b'clich\xe9'
    b'clich\x82'
    Traceback (most recent call last):
      File "1261_10_16_decode_unicode.py", line 5, in <module>
        print(characters.encode("ascii"))
    UnicodeEncodeError: 'ascii' codec can't encode character '\xe9' in position 5: ordinal not in range(128)  
```

现在你应该理解编码的重要性了！带音标的字符在每种编码中表示为不同的字节；如果我们解码字节到文本时使用错误的编码，我们会得到错误的字符。

最后一种情况下的异常并不总是我们期望的行为；可能存在我们希望以不同方式处理未知字符的情况。`encode`方法接受一个可选的字符串参数`errors`，可以定义如何处理此类字符。此字符串可以是以下之一：

+   `strict`

+   `replace`

+   `ignore`

+   `xmlcharrefreplace`

`strict`替换策略是我们刚刚看到的默认策略。当遇到一个在请求的编码中没有有效表示的字节序列时，会引发异常。当使用`replace`策略时，字符会被替换为不同的字符；在 ASCII 中，它是一个问号；其他编码可能使用不同的符号，例如一个空框。`ignore`策略简单地丢弃它不理解的任何字节，而`xmlcharrefreplace`策略创建一个表示 Unicode 字符的`xml`实体。这在将未知字符串转换为用于 XML 文档时可能很有用。以下是每种策略如何影响我们的示例单词：

| **策略** | **应用 `"cliché".encode("ascii", strategy)` 的结果** |
| --- | --- |
| `replace` | `b'clich?'` |
| `ignore` | `b'clich'` |
| `xmlcharrefreplace` | `b'cliché'` |

可以调用 `str.encode` 和 `bytes.decode` 方法而不传递编码名称。编码将被设置为当前平台的默认编码。这取决于当前的操作系统和区域设置；你可以使用 `sys.getdefaultencoding()` 函数来查找它。尽管如此，通常最好明确指定编码，因为平台的默认编码可能会更改，或者程序可能有一天会被扩展以处理来自更广泛来源的文本。

如果你正在编码文本，但不知道要使用哪种编码，最好使用 UTF-8 编码。UTF-8 能够表示任何 Unicode 字符。在现代软件中，它是事实上的标准编码，以确保任何语言（甚至多种语言）的文档可以交换。其他可能的编码对于旧文档或在默认使用不同字符集的区域是有用的。

UTF-8 编码使用一个字节来表示 ASCII 和其他常见字符，对于更复杂的字符则使用最多四个字节。UTF-8 是特殊的，因为它与 ASCII 兼容；任何使用 UTF-8 编码的 ASCII 文档都将与原始 ASCII 文档相同。

我总是记不清是使用 `encode` 还是 `decode` 将二进制字节转换为 Unicode。我总是希望这些方法被命名为 `to_binary` 和 `from_binary`。如果你也有同样的问题，试着在心中将单词 *code* 替换为 *binary*；*enbinary* 和 *debinary* 与 *to_binary* 和 *from_binary* 非常接近。自从想出这个助记符以来，我没有查阅方法帮助文件就节省了很多时间。

# 可变字节字符串

`bytes` 类型，像 `str` 一样，是不可变的。我们可以在 `bytes` 对象上使用索引和切片表示法来搜索特定的字节序列，但我们不能扩展或修改它们。在处理 I/O 时，这可能会非常不方便，因为通常需要缓冲传入或传出的字节，直到它们准备好发送。例如，如果我们从套接字接收数据，可能需要多次 `recv` 调用才能接收到整个消息。

这就是内置的 `bytearray` 类型发挥作用的地方。这种类型的行为类似于列表，但它只包含字节。该类的构造函数可以接受一个 `bytes` 对象来初始化它。可以使用 `extend` 方法将另一个 `bytes` 对象追加到现有数组中（例如，当从套接字或其他 I/O 通道接收更多数据时）。

可以在 `bytearray` 上使用切片表示法来修改项。例如，此代码从一个 `bytes` 对象构建一个 `bytearray`，然后替换两个字节：

```py
b = bytearray(b"abcdefgh") 
b[4:6] = b"\x15\xa3" 
print(b) 
```

输出看起来像这样：

```py
bytearray(b'abcd\x15\xa3gh')  
```

如果我们要在 `bytearray` 中操作单个元素，我们必须传递一个介于 0 和 255（包含）之间的整数作为值。这个整数代表一个特定的 `bytes` 模式。如果我们尝试传递一个字符或 `bytes` 对象，它将引发异常。

可以使用 `ord`（意为序数）函数将单个字节字符转换为整数。此函数返回单个字符的整数表示：

```py
b = bytearray(b"abcdef")
b[3] = ord(b"g")
b[4] = 68
print(b)
```

输出看起来像这样：

```py
bytearray(b'abcgDf')  
```

在构建数组后，我们将索引 `3`（第四个字符，因为索引从 `0` 开始，就像列表一样）处的字符替换为字节 `103`。这个整数是由 `ord` 函数返回的，是小写 `g` 的 ASCII 字符。为了说明，我们还用字节编号 `68` 替换了下一个字符，它映射到大写 `D` 的 ASCII 字符。

`bytearray` 类型有允许其表现得像列表（例如，我们可以向其中追加整数字节）的方法，但也可以像 `bytes` 对象一样使用；我们可以使用 `count` 和 `find` 等方法，就像它们在 `bytes` 或 `str` 对象上表现一样。区别在于 `bytearray` 是一个可变类型，这对于从特定输入源构建复杂的字节序列非常有用。

# 正则表达式

你知道使用面向对象原则真正难以做什么吗？解析字符串以匹配任意模式，就是这样。已经有许多学术论文被撰写，其中使用面向对象设计来设置字符串解析，但结果总是非常冗长且难以阅读，并且在实践中并不广泛使用。

在现实世界中，大多数编程语言的字符串解析都是由正则表达式处理的。这些表达式并不冗长，但哇，它们确实很难读，至少在你学会语法之前是这样。尽管正则表达式不是面向对象的，但 Python 正则表达式库提供了一些类和对象，你可以使用它们来构建和运行正则表达式。

正则表达式用于解决一个常见问题：给定一个字符串，确定该字符串是否与给定的模式匹配，并且可选地收集包含相关信息子串。它们可以用来回答以下问题：

+   这个字符串是否是一个有效的 URL？

+   日志文件中所有警告消息的日期和时间是什么？

+   `/etc/passwd` 中的哪些用户属于给定的组？

+   访问者输入的 URL 请求了哪个用户名和文档？

有许多类似的场景，其中正则表达式是正确的答案。许多程序员因为不知道或不学习正则表达式而错误地实现了复杂且脆弱的字符串解析库。在本节中，我们将获得足够的正则表达式知识，以避免犯这样的错误。

# 匹配模式

正则表达式是一个复杂的迷你语言。它们依赖于特殊字符来匹配未知字符串，但让我们从字面字符开始，例如字母、数字和空格字符，这些字符总是匹配自身。让我们看一个基本示例：

```py
import re 

search_string = "hello world" 
pattern = "hello world" 

match = re.match(pattern, search_string) 

if match: 
    print("regex matches") 
```

Python 标准库中的正则表达式模块被称为 `re`。我们导入它并设置一个搜索字符串和搜索模式；在这种情况下，它们是相同的字符串。由于搜索字符串与给定的模式匹配，条件通过并且执行了 `print` 语句。

请记住，`match` 函数会将模式与字符串的开始部分进行匹配。因此，如果模式是 `"ello world"`，则不会找到匹配项。由于令人困惑的不对称性，解析器一旦找到匹配项就会停止搜索，所以模式 `"hello wo"` 可以成功匹配。让我们构建一个小型示例程序来展示这些差异，并帮助我们学习其他正则表达式语法：

```py
import sys 
import re 

pattern = sys.argv[1] 
search_string = sys.argv[2] 
match = re.match(pattern, search_string) 

if match: 
    template = "'{}' matches pattern '{}'" 
else: 
    template = "'{}' does not match pattern '{}'" 

print(template.format(search_string, pattern)) 
```

这只是之前示例的通用版本，它从命令行接受模式和搜索字符串。我们可以看到模式的开始必须匹配，但一旦在以下命令行交互中找到匹配项，就会返回一个值：

```py
$ python regex_generic.py "hello worl" "hello world"
'hello world' matches pattern 'hello worl'
$ python regex_generic.py "ello world" "hello world"
'hello world' does not match pattern 'ello world'  
```

我们将在接下来的几节中使用这个脚本。虽然脚本总是使用 `python regex_generic.py "<pattern>" "<string>"` 命令来调用，但我们将只看到以下示例中的输出，以节省空间。

如果你需要控制项目是否出现在行首或行尾（或者字符串中没有换行符，或者在字符串的开始和结束处），你可以使用 `^` 和 `$` 字符分别表示字符串的开始和结束。如果你想使模式匹配整个字符串，包含这两个字符是个好主意：

```py
'hello world' matches pattern '^hello world$'
'hello worl' does not match pattern '^hello world$'  
```

# 匹配一组字符

让我们从匹配任意字符开始。在正则表达式模式中使用点字符可以匹配任何单个字符。在字符串中使用点意味着你不在乎字符是什么，只要那里有一个字符即可。以下是一些示例：

```py
'hello world' matches pattern 'hel.o world'
'helpo world' matches pattern 'hel.o world'
'hel o world' matches pattern 'hel.o world'
'helo world' does not match pattern 'hel.o world'  
```

注意到最后一个示例没有匹配，因为在模式的点位置上没有字符。

那么一切都很好，但如果我们只想匹配几个特定的字符怎么办？我们可以在方括号内放置一组字符来匹配这些字符中的任何一个。所以，如果我们在一个正则表达式模式中遇到字符串 `[abc]`，我们知道这五个字符（包括两个方括号）将只匹配正在搜索的字符串中的一个字符，并且进一步地，这个字符将是 `a`、`b` 或 `c` 中的一个。让我们看几个示例：

```py
'hello world' matches pattern 'hel[lp]o world'
'helpo world' matches pattern 'hel[lp]o world'
'helPo world' does not match pattern 'hel[lp]o world'  
```

这些方括号集合应该被称为字符集，但它们更常被称为**字符类**。通常，我们希望在集合内包含大量字符，输入它们可能会很单调且容易出错。幸运的是，正则表达式的设计者想到了这一点，并给了我们一个快捷方式。在字符集中，破折号字符将创建一个范围。如果你想要匹配所有小写字母、所有字母或所有数字，这特别有用，如下所示：

```py
 'hello   world' does not match pattern 'hello [a-z] world'
 'hello b world' matches pattern 'hello [a-z] world'
 'hello B world' matches pattern 'hello [a-zA-Z] world'
 'hello 2 world' matches pattern 'hello [a-zA-Z0-9] world'  
```

还有其他方法可以匹配或排除单个字符，但如果你想知道它们是什么，你需要通过网络搜索找到更全面的教程！

# 转义字符

如果在模式中放置一个点字符可以匹配任何任意字符，那么我们如何匹配字符串中的单个点呢？一种可能的方法是将点放在方括号内以创建一个字符类，但一个更通用的方法是使用反斜杠来转义它。以下是一个匹配 0.00 到 0.99 之间两位小数的正则表达式：

```py
'0.05' matches pattern '0\.[0-9][0-9]'
'005' does not match pattern '0\.[0-9][0-9]'
'0,05' does not match pattern '0\.[0-9][0-9]'  
```

对于这个模式，两个字符`\.`匹配单个`.`字符。如果点字符缺失或是一个不同的字符，它将不会匹配。

这个反斜杠转义序列用于正则表达式中的各种特殊字符。你可以使用`\[`来插入一个方括号而不开始一个字符类，并且使用`\(`来插入一个括号，我们稍后会看到它也是一个特殊字符。

更有趣的是，我们还可以使用转义符号后跟一个字符来表示特殊字符，如换行符（`\n`）和制表符（`\t`）。此外，一些字符类可以使用转义字符串更简洁地表示：`\s`表示空白字符；`\w`表示字母、数字和下划线；`\d`表示数字：

```py
'(abc]' matches pattern '\(abc\]'
' 1a' matches pattern '\s\d\w'
'\t5n' does not match pattern '\s\d\w'
'5n' matches pattern '\s\d\w'  
```

# 匹配多个字符

使用这些信息，我们可以匹配大多数已知长度的字符串，但大多数时候，我们不知道在模式内要匹配多少个字符。正则表达式也可以处理这个问题。我们可以通过在模式后附加几个难以记住的标点符号之一来修改模式，以匹配多个字符。

星号（`*`）字符表示前面的模式可以匹配零次或多次。这听起来可能有些荒谬，但它是最有用的重复字符之一。在我们探索为什么之前，考虑一些荒谬的例子以确保我们理解它的作用：

```py
'hello' matches pattern 'hel*o'
'heo' matches pattern 'hel*o'
'helllllo' matches pattern 'hel*o'  
```

因此，模式中的`*`字符表示前面的模式（`l`字符）是可选的，如果存在，可以尽可能多地重复以匹配模式。其余的字符（`h`、`e`和`o`）必须恰好出现一次。

通常情况下，我们并不需要多次匹配单个字母，但如果我们将星号与匹配多个字符的模式结合起来，情况就变得更有趣了。例如，`.*` 将匹配任何字符串，而 `[a-z]*` 则匹配任何由小写字母组成的单词集合，包括空字符串。以下是一些示例：

```py
'A string.' matches pattern '[A-Z][a-z]* [a-z]*\.'
'No .' matches pattern '[A-Z][a-z]* [a-z]*\.'
'' matches pattern '[a-z]*.*'  
```

模式中的加号（`+`）与星号的行为类似；它表示前面的模式可以重复一次或多次，但与星号不同的是，它不是可选的。问号（`?`）确保模式恰好出现零次或一次，但不超过一次。让我们通过玩数字来探索一些这些模式（记住 `\d` 匹配与 `[0-9]` 相同的字符类）：

```py
'0.4' matches pattern '\d+\.\d+'
'1.002' matches pattern '\d+\.\d+'
'1.' does not match pattern '\d+\.\d+'
'1%' matches pattern '\d?\d%'
'99%' matches pattern '\d?\d%'
'999%' does not match pattern '\d?\d%'  
```

# 将模式分组

到目前为止，我们已经看到我们可以多次重复一个模式，但我们受到可以重复的模式类型的限制。如果我们想重复单个字符，我们没问题，但如果我们想重复字符序列呢？将任何一组模式括起来，在应用重复操作时，可以将它们视为一个单独的模式。比较以下模式：

```py
'abccc' matches pattern 'abc{3}'
'abccc' does not match pattern '(abc){3}'
'abcabcabc' matches pattern '(abc){3}'  
```

结合复杂的模式，这种分组功能极大地扩展了我们的模式匹配能力。以下是一个匹配简单英语句子的正则表达式：

```py
'Eat.' matches pattern '[A-Z][a-z]*( [a-z]+)*\.$'
'Eat more good food.' matches pattern '[A-Z][a-z]*( [a-z]+)*\.$'
'A good meal.' matches pattern '[A-Z][a-z]*( [a-z]+)*\.$'  
```

第一个单词以大写字母开头，后面跟着零个或多个小写字母。然后，我们进入一个匹配单个空格后跟一个由一个或多个小写字母组成的单词的括号表达式。这个括号表达式可以重复零次或多次，并且模式以句号结束。句号之后不能有其他字符，正如 `$` 匹配字符串的结尾所示。

我们已经看到了许多最基本模式，但正则表达式语言支持许多其他模式。我在使用正则表达式的最初几年里，每次需要做某事时都会查找语法。值得将 Python 的 `re` 模块文档添加到书签并经常查阅。正则表达式几乎可以匹配任何内容，它们应该是解析字符串时首先考虑的工具。

# 从正则表达式获取信息

现在，让我们关注 Python 方面的事情。正则表达式语法与面向对象编程相差甚远。然而，Python 的 `re` 模块提供了一个面向对象的接口来访问正则表达式引擎。

我们一直在检查 `re.match` 函数是否返回一个有效的对象。如果模式不匹配，该函数返回 `None`。如果它匹配，则返回一个有用的对象，我们可以用它来获取有关模式的信息。

到目前为止，我们的正则表达式已经回答了诸如“这个字符串是否与这个模式匹配？”等问题。匹配模式很有用，但在许多情况下，一个更有趣的问题是，“如果这个字符串与这个模式匹配，相关子串的值是什么？”如果你使用组来标识稍后想要引用的模式部分，你可以从匹配返回值中获取它们，如下一个示例所示：

```py
pattern = "^[a-zA-Z.]+@([a-z.]*\.[a-z]+)$" 
search_string = "some.user@example.com" 
match = re.match(pattern, search_string) 

if match: 
    domain = match.groups()[0] 
    print(domain) 
```

描述有效电子邮件地址的规范极其复杂，能够准确匹配所有可能性的正则表达式非常长。因此，我们采取了欺骗手段，创建了一个简单的正则表达式来匹配一些常见的电子邮件地址；目的是我们想要访问域名（在`@`符号之后），以便我们可以连接到该地址。通过将模式中这部分内容用括号括起来，并在`match`方法返回的对象上调用`groups()`方法，可以轻松实现这一点。

`groups`方法返回一个包含模式内部所有匹配组的元组，你可以通过索引来访问特定的值。组是从左到右排序的。然而，请注意，组可以嵌套，这意味着你可以在另一个组内部有一个或多个组。在这种情况下，组是按照它们的左括号顺序返回的，所以最外层的组将先于其内部匹配组返回。

除了`match`函数外，`re`模块还提供了一些其他有用的函数，`search`和`findall`。`search`函数找到匹配模式的第一个实例，放宽了模式应该从字符串的第一个字母开始的限制。请注意，你可以通过使用`match`并在模式前面加上`^.*`字符来达到类似的效果，以匹配字符串开始和你要查找的模式之间的任何字符。

`findall`函数的行为与搜索类似，但它找到的是匹配模式的全部非重叠实例，而不仅仅是第一个。基本上，它会找到第一个匹配项，然后重置搜索到匹配字符串的末尾，并找到下一个匹配项。

与你预期的返回匹配对象列表不同，它返回一个匹配字符串或元组的列表。有时是字符串，有时是元组。这根本不是一个很好的 API！与所有糟糕的 API 一样，你必须记住差异，不要依赖直觉。返回值的类型取决于正则表达式内部括号组的数量：

+   如果模式中没有组，`re.findall`将返回一个字符串列表，其中每个值都是从源字符串中匹配模式的完整子串。

+   如果模式中恰好有一个组，`re.findall`将返回一个字符串列表，其中每个值是那个组的全部内容。

+   如果模式中有多个组，`re.findall`将返回一个元组列表，其中每个元组包含匹配组中的一个值，按顺序排列。

当你在自己的 Python 库中设计函数调用时，尽量让函数总是返回一致的数据结构。设计能够接受任意输入并处理它们的函数通常是个好主意，但返回值不应该根据输入从单个值切换到列表，或者从值列表切换到元组列表。`re.findall` 就是一个教训！

下面的交互式会话中的示例有望阐明这些差异：

```py
>>> import re
>>> re.findall('a.', 'abacadefagah')
['ab', 'ac', 'ad', 'ag', 'ah']
>>> re.findall('a(.)', 'abacadefagah')
['b', 'c', 'd', 'g', 'h']
>>> re.findall('(a)(.)', 'abacadefagah')
[('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'g'), ('a', 'h')]
>>> re.findall('((a)(.))', 'abacadefagah')
[('ab', 'a', 'b'), ('ac', 'a', 'c'), ('ad', 'a', 'd'), ('ag', 'a', 'g'), ('ah', 'a', 
'h')]  
```

# 使重复的正则表达式更高效

每次调用正则表达式的一个方法时，引擎都必须将模式字符串转换为一种内部结构，这使得字符串搜索变得快速。这种转换需要相当多的时间。如果一个正则表达式模式将被多次使用（例如，在 `for` 或 `while` 循环中），那么这个转换步骤只做一次会更好。

这可以通过 `re.compile` 方法实现。它返回一个面向对象的正则表达式版本，该版本已被编译并具有我们已探索的方法（如 `match`、`search` 和 `findall` 等）。我们将在案例研究中看到这方面的例子。

这肯定是对正则表达式的一个浓缩介绍。到目前为止，我们对基础知识有了很好的感觉，并且会在需要进一步研究时识别出来。如果我们有一个字符串模式匹配问题，正则表达式几乎肯定能够为我们解决这些问题。然而，我们可能需要在一个更全面的正则表达式主题覆盖中查找新的语法。但现在我们知道该寻找什么了！让我们继续到一个完全不同的主题：文件系统路径。

# 文件系统路径

所有操作系统都提供了一个 *文件系统*，一种将 *文件夹*（或 *目录*）和 *文件* 的逻辑抽象映射到硬盘或其它存储设备上存储的位和字节的方法。作为人类，我们通常通过文件夹和不同类型的文件的拖放界面，或者通过 `cp`、`mv` 和 `mkdir` 等命令行程序与文件系统交互。

作为程序员，我们必须通过一系列系统调用来与文件系统交互。你可以把它们看作是操作系统提供的库函数，以便程序可以调用它们。它们有一个笨拙的接口，包括整数文件句柄和缓冲读取和写入，而且这个接口取决于你使用的操作系统。Python 在 `os.path` 模块中提供了对这些系统调用的操作系统无关的抽象。与直接访问操作系统相比，这要容易一些，但并不直观。它需要大量的字符串连接，并且你必须意识到在目录之间使用正斜杠还是反斜杠，这取决于操作系统。有一个 `os.sep` 文件表示路径分隔符，但使用它需要像这样的代码：

```py
>>> path = os.path.abspath(os.sep.join(['.', 'subdir', 'subsubdir', 'file.ext']))
>>> print(path)
/home/dusty/subdir/subsubdir/file.ext
```

在整个标准库中，与文件系统路径一起工作可能是最令人烦恼的字符串使用之一。在命令行上容易输入的路径在 Python 代码中变得难以辨认。当你必须操作和访问多个路径时（例如，在处理机器学习计算机视觉问题的数据管道中的图像时），仅仅管理这些目录就变成了一项艰巨的任务。

因此，Python 语言设计者将一个名为 `pathlib` 的模块包含在标准库中。它是对路径和文件的面向对象表示，与它一起工作要愉快得多。使用 `pathlib` 的前一个路径看起来像这样：

```py
>>> path = (pathlib.Path(".") / "subdir" / " subsubdir" / "file.ext").absolute()
>>> print(path)
/home/dusty/subdir/subsubdir/file.ext
```

如您所见，它要容易得多，可以清楚地看到发生了什么。注意除法运算符作为路径分隔符的独特使用，这样你就不需要做任何与 `os.sep` 相关的事情。

在一个更实际的例子中，考虑一些代码，它计算当前目录及其子目录中所有 Python 文件的代码行数（不包括空白和注释）：

```py
import pathlib

def count_sloc(dir_path):
    sloc = 0
    for path in dir_path.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_dir():
            sloc += count_sloc(path)
            continue
        if path.suffix != ".py":
            continue
        with path.open() as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    sloc += 1
    return sloc

root_path = pathlib.Path(".")

print(f"{count_sloc(root_path)} lines of python code")

```

在典型的 `pathlib` 使用中，我们很少需要构建超过一个或两个路径。通常，其他文件或目录相对于一个通用路径。这个例子演示了这一点。我们只构建一个路径，即使用 `pathlib.Path(".")` 从当前目录开始。然后，其他路径基于这个路径创建。

`count_sloc` 函数首先将 **sloc**（**源代码行数**）计数器初始化为零。然后，它使用 `dir_path.iterdir` 生成器遍历函数传入路径中的所有文件和目录（我们将在下一章详细讨论生成器；现在，可以将其视为一种动态列表）。`iterdir` 返回给 `for` 循环的每个路径本身也是一个路径。我们首先测试这个路径是否以 `.` 开头，这在大多数操作系统上代表一个隐藏目录（如果你使用版本控制，这将防止它计算 `.git` 目录中的任何文件）。然后，我们使用 `isdir()` 方法检查它是否是目录。如果是，我们递归调用 `count_sloc` 来计算子包中模块的代码行数。

如果它不是一个目录，我们假设它是一个普通文件，并使用 `suffix` 属性跳过任何不以 `.py` 扩展名结尾的文件。现在，我们知道我们有一个指向 Python 文件的路径，我们使用 `open()` 方法打开文件，该方法返回一个上下文管理器。我们将其包裹在一个 `with` 块中，这样当我们完成时文件会自动关闭。

`Path.open` 方法与内置的 `open` 函数具有类似的参数，但它使用更面向对象的语法。如果你更喜欢函数版本，你可以将一个 `Path` 对象作为第一个参数传递给它（换句话说，`with open(Path('./README.md')):`），就像传递一个字符串一样。但我觉得如果路径已经存在，`Path('./README.md').open()` 的可读性更好。

我们然后遍历文件中的每一行，并将其添加到计数中。我们跳过空白行和注释行，因为这些并不代表实际的源代码。总计数返回给调用函数，这可能是最初的调用或递归的父调用。

`pathlib` 模块中的 `Path` 类有一个方法或属性来覆盖你可能会对路径做的几乎所有操作。除了我们在示例中提到的那些之外，这里还有一些我最喜欢的：

+   `.absolute()` 返回从文件系统根目录的完整路径。我通常在构建每个路径时都调用这个方法，因为我对可能会忘记相对路径的来源有一点点偏执。

+   `.parent` 返回父目录的路径。

+   `.exists()` 检查文件或目录是否存在。

+   `.mkdir()` 在当前路径创建一个目录。它接受布尔参数 `parents` 和 `exist_ok` 来指示如果需要，它应该递归地创建目录，并且如果目录已存在，它不应该引发异常。

更多关于 `pathlib` 的特殊用法，请参阅标准库文档：[`docs.python.org/3/library/pathlib.html`](https://docs.python.org/3/library/pathlib.html)。

大多数接受字符串路径的标准库模块也可以接受 `pathlib.Path` 对象。例如，你可以通过传递一个路径到它来打开一个 ZIP 文件：

```py
>>> zipfile.ZipFile(Path('nothing.zip'), 'w').writestr('filename', 'contents')
```

这并不总是有效，特别是如果你正在使用作为 C 扩展实现的第三方库。在这些情况下，你必须使用 `str(pathname)` 将路径转换为字符串。

# 对象序列化

现在，我们把将数据写入文件并在任意晚些时候检索它的能力视为理所当然。尽管这样做很方便（想象一下如果我们不能存储任何东西的计算状态！），我们经常发现自己将存储在内存中一个漂亮的对象或设计模式中的数据转换为某种笨拙的文本或二进制格式以进行存储、通过网络传输或在远程服务器上远程调用。

Python 的 `pickle` 模块是一种面向对象的方式来直接以特殊存储格式存储对象。它本质上将一个对象（以及它作为属性持有的所有对象）转换为一个字节序列，我们可以按我们的需要存储或传输。

对于基本任务，`pickle` 模块有一个极其简单的接口。它包括四个基本函数用于存储和加载数据：两个用于操作文件对象，两个用于操作 `bytes` 对象（后者只是文件对象接口的快捷方式，因此我们不必自己创建 `BytesIO` 文件对象）。

`dump` 方法接受要写入的对象和一个文件对象，将序列化的字节写入该对象。此对象必须有一个 `write` 方法（否则它就不是文件对象），该方法必须知道如何处理 `bytes` 参数（因此，用于文本输出的打开的文件不会工作）。

`load` 方法正好相反；它从一个文件类似的对象中读取一个序列化的对象。此对象必须具有适当的文件类似 `read` 和 `readline` 参数，每个参数当然必须返回 `bytes`。`pickle` 模块将从这些字节中加载对象，`load` 方法将返回完全重建的对象。以下是一个示例，它在一个列表对象中存储和加载一些数据：

```py
import pickle 

some_data = ["a list", "containing", 5, 
        "values including another list", 
        ["inner", "list"]] 

with open("pickled_list", 'wb') as file: 
    pickle.dump(some_data, file) 

with open("pickled_list", 'rb') as file: 
    loaded_data = pickle.load(file) 

print(loaded_data) 
assert loaded_data == some_data 
```

这段代码按预期工作：对象被存储在文件中，然后从同一文件中加载。在这种情况下，我们使用 `with` 语句打开文件，以便它自动关闭。文件首先用于写入，然后再次用于读取，具体取决于我们是存储还是加载数据。

在末尾的 `assert` 语句会在新加载的对象不等于原始对象时引发错误。相等性并不意味着它们是同一个对象。实际上，如果我们打印两个对象的 `id()`，我们会发现它们是不同的。然而，由于它们都是内容相等的列表，这两个列表也被认为是相等的。

`dumps` 和 `loads` 函数的行为与它们的文件类似，除了它们返回或接受 `bytes` 而不是文件类似的对象。`dumps` 函数只需要一个参数，即要存储的对象，并返回一个序列化的 `bytes` 对象。`loads` 函数需要一个 `bytes` 对象，并返回恢复的对象。方法名中的 `'s'` 字符代表字符串；这是 Python 早期版本中的一个遗留名称，当时使用 `str` 对象而不是 `bytes`。

可以多次在单个打开的文件上调用 `dump` 或 `load`。每次调用 `dump` 都会存储一个对象（以及它由或包含的任何对象），而调用 `load` 将加载并返回一个对象。因此，对于单个文件，在存储对象时对 `dump` 的每次单独调用都应该在稍后日期恢复时有一个相关的 `load` 调用。

# 自定义 pickles

对于大多数常见的 Python 对象，pickling 只需“正常工作”。基本原语，如整数、浮点数和字符串可以 picklable，任何容器对象，如列表或字典也可以，只要这些容器的内容也是 picklable。更重要的是，任何对象都可以被 picklable，只要它的所有属性也是 picklable。

那么，是什么让一个属性不可 picklable？通常，这与未来加载时没有意义的时敏属性有关。例如，如果我们有一个打开的网络套接字、打开的文件、正在运行的线程或数据库连接作为对象的属性存储，那么将这些对象 picklable 就没有意义；当我们尝试稍后重新加载它们时，大量的操作系统状态将简单地消失。我们不能假装线程或套接字连接存在并使其出现！不，我们需要以某种方式自定义此类瞬态数据的存储和恢复方式。

这是一个每小时加载网页内容的类，以确保它们保持最新。它使用`threading.Timer`类来安排下一次更新：

```py
from threading import Timer 
import datetime 
from urllib.request import urlopen 

class UpdatedURL: 
    def __init__(self, url): 
        self.url = url 
        self.contents = '' 
        self.last_updated = None 
        self.update() 

    def update(self): 
        self.contents = urlopen(self.url).read() 
        self.last_updated = datetime.datetime.now() 
        self.schedule() 

    def schedule(self): 
        self.timer = Timer(3600, self.update) 
        self.timer.setDaemon(True) 
        self.timer.start() 
```

`url`、`contents`和`last_updated`都是可 pickle 的，但如果我们尝试 pickle 这个类的实例，`self.timer`实例会变得有些疯狂：

```py
>>> u = UpdatedURL("http://dusty.phillips.codes")
^[[Apickle.dumps(u)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
TypeError: can't pickle _thread.lock objects
```

这不是一个非常有用的错误，但看起来我们正在尝试对不应该进行 pickle 操作的东西进行操作。那将是`Timer`实例；我们在 schedule 方法中存储了对`self.timer`的引用，而这个属性不能被序列化。

当`pickle`尝试序列化一个对象时，它只是尝试存储对象的`__dict__`属性；`__dict__`是一个字典，将对象上的所有属性名映射到它们的值。幸运的是，在检查`__dict__`之前，`pickle`会检查是否存在`__getstate__`方法。如果存在，它将存储该方法的返回值而不是`__dict__`。

让我们在`UpdatedURL`类中添加一个`__getstate__`方法，该方法简单地返回一个没有计时器的`__dict__`的副本：

```py
    def __getstate__(self): 
        new_state = self.__dict__.copy() 
        if 'timer' in new_state: 
            del new_state['timer'] 
        return new_state 
```

如果我们现在 pickle 对象，它将不再失败。我们甚至可以使用`loads`成功恢复该对象。然而，恢复的对象没有计时器属性，所以它不会像设计的那样刷新内容。我们需要在对象反序列化时以某种方式创建一个新的计时器（以替换缺失的计时器）。

如我们所预期，有一个互补的`__setstate__`方法可以实现来自定义反序列化。该方法接受一个参数，即`__getstate__`返回的对象。如果我们实现了这两个方法，`__getstate__`不需要返回一个字典，因为`__setstate__`将知道如何处理`__getstate__`选择返回的任何对象。在我们的情况下，我们只想恢复`__dict__`，然后创建一个新的计时器：

```py
 def __setstate__(self, data): self.__dict__ = data self.schedule() 
```

`pickle`模块非常灵活，并提供其他工具来进一步自定义序列化过程，如果你需要的话。然而，这些超出了本书的范围。我们介绍的工具对于许多基本的序列化任务已经足够了。要序列化的对象通常是相对简单的数据对象；我们可能不会 pickle 整个运行中的程序或复杂的设计模式，例如。

# 序列化网络对象

从未知或不可信的来源加载 pickle 对象不是一个好主意。通过 pickle，可以注入任意代码到 pickle 文件中，恶意攻击计算机。pickle 的另一个缺点是它们只能被其他 Python 程序加载，不能轻易与其他语言编写的服务共享。

这些年来，已经使用了许多用于此目的的格式。**可扩展标记语言**（**XML**）曾经非常流行，尤其是在 Java 开发者中。**另一种标记语言**（**YAML**）是另一种偶尔会提到的格式。表格数据通常以 **逗号分隔值**（**CSV**）格式交换。其中许多正在逐渐消失，你将在未来遇到更多。Python 为所有这些格式都提供了坚实的标准或第三方库。

在使用这些库处理不可信数据之前，请确保调查每个库的安全问题。例如，XML 和 YAML 都有一些不为人知的特性，如果被恶意使用，可以在主机机器上执行任意命令。这些特性可能默认并未关闭。请做好研究。

**JavaScript 对象表示法**（**JSON**）是一种用于交换原始数据的人可读格式。JSON 是一种标准格式，可以被各种异构客户端系统解释。因此，JSON 对于在完全解耦的系统之间传输数据非常有用。此外，JSON 不支持可执行代码，只能序列化数据；因此，将其注入恶意语句更加困难。

由于 JSON 可以很容易地被 JavaScript 引擎解释，它通常用于在 Web 服务器和具有 JavaScript 功能的 Web 浏览器之间传输数据。如果提供数据的服务器端应用程序是用 Python 编写的，它需要一种方法将内部数据转换为 JSON 格式。

有一个模块可以完成这个任务，其名称很自然地被命名为 `json`。这个模块提供了一个与 `pickle` 模块类似的接口，包括 `dump`、`load`、`dumps` 和 `loads` 函数。这些函数的默认调用几乎与 `pickle` 中的调用完全相同，所以这里就不重复细节了。存在一些差异；显然，这些调用的输出是有效的 JSON 语法，而不是被序列化的对象。此外，`json` 函数操作的是 `str` 对象，而不是 `bytes`。因此，在向文件写入或从文件读取时，我们需要创建文本文件而不是二进制文件。

JSON 序列化器不如 `pickle` 模块健壮；它只能序列化基本类型，如整数、浮点数和字符串，以及简单的容器，如字典和列表。这些都有直接映射到 JSON 表示的直接映射，但 JSON 无法表示类、方法或函数。无法以这种格式传输完整的对象。因为我们将对象序列化到 JSON 格式后，接收者通常不是 Python 对象，它无论如何也无法像 Python 那样理解类或方法。尽管其名称中有 O 代表对象，但 JSON 是一种 **数据** 语法；如你所知，对象由数据和行为组成。

如果我们有只想序列化数据的对象，我们始终可以序列化对象的`__dict__`属性。或者，我们可以通过提供自定义代码来半自动化此任务，从某些类型的对象创建或解析一个可序列化的 JSON 字典。

在`json`模块中，存储和加载对象的功能都接受可选参数以自定义行为。`dump`和`dumps`方法接受一个命名不佳的`cls`（简称 class，是一个保留关键字）关键字参数。如果传递了，这应该是一个`JSONEncoder`类的子类，并且重写了`default`方法。此方法接受任意对象并将其转换为`json`可以消化的字典。如果它不知道如何处理该对象，我们应该调用`super()`方法，这样它就可以以正常方式处理基本类型。

`load`和`loads`方法也接受这样的`cls`参数，它可以是一个逆类`JSONDecoder`的子类。然而，通常只需使用`object_hook`关键字参数将这些方法传递给一个函数就足够了。此函数接受一个字典并返回一个对象；如果它不知道如何处理输入字典，它可以返回未修改的字典。

让我们看看一个例子。假设我们有一个以下简单的联系人类，我们想要序列化：

```py
class Contact: 
    def __init__(self, first, last): 
        self.first = first 
        self.last = last 

    @property 
    def full_name(self): 
        return("{} {}".format(self.first, self.last)) 
```

我们可以直接序列化`__dict__`属性：

```py
    >>> c = Contact("John", "Smith")
    >>> json.dumps(c.__dict__)
    '{"last": "Smith", "first": "John"}'  
```

但是以这种方式访问特殊（双下划线）属性有点粗糙。此外，如果接收到的代码（可能是一些网页上的 JavaScript）需要`full_name`属性，该怎么办？当然，我们可以手动构造字典，但让我们创建一个自定义编码器：

```py
import json

class ContactEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Contact):
            return {
                "is_contact": True,
                "first": obj.first,
                "last": obj.last,
                "full": obj.full_name,
            }
        return super().default(obj)
```

`default`方法基本上检查我们正在尝试序列化的对象类型。如果是联系人，我们手动将其转换为字典。否则，我们让父类处理序列化（假设它是一个基本类型，`json`知道如何处理）。请注意，我们传递一个额外的属性来识别这个对象是一个联系人，因为在加载时无法区分。这只是一个约定；对于更通用的序列化机制，可能更有意义在字典中存储字符串类型，甚至可能是包括包和模块的全类名。记住，字典的格式取决于接收端的代码；必须就如何指定数据达成一致。

我们可以通过将类（而不是实例化对象）传递给`dump`或`dumps`函数来使用此类来编码联系人：

```py
    >>> c = Contact("John", "Smith")
    >>> json.dumps(c, cls=ContactEncoder)
    '{"is_contact": true, "last": "Smith", "full": "John Smith",
    "first": "John"}'  
```

对于解码，我们可以编写一个函数，该函数接受一个字典并检查`is_contact`变量的存在以决定是否将其转换为联系人：

```py
def decode_contact(dic):
    if dic.get("is_contact"):
        return Contact(dic["first"], dic["last"])
    else:
        return dic
```

我们可以通过使用`object_hook`关键字参数将此函数传递给`load`或`loads`函数：

```py
    >>> data = ('{"is_contact": true, "last": "smith",'
         '"full": "john smith", "first": "john"}')

    >>> c = json.loads(data, object_hook=decode_contact)
    >>> c
    <__main__.Contact object at 0xa02918c>
    >>> c.full_name
    'john smith'  
```

# 案例研究

让我们在 Python 中构建一个基于正则表达式的模板引擎。这个引擎将解析一个文本文件（例如 HTML 页面），并将某些指令替换为从这些指令的输入计算出的文本。这是我们可能想要用正则表达式完成的最为复杂的任务；实际上，这个功能的完整版本可能会利用适当的语言解析机制。

考虑以下输入文件：

```py
/** include header.html **/ 
<h1>This is the title of the front page</h1> 
/** include menu.html **/ 
<p>My name is /** variable name **/. 
This is the content of my front page. It goes below the menu.</p> 
<table> 
<tr><th>Favourite Books</th></tr> 
/** loopover book_list **/ 
<tr><td>/** loopvar **/</td></tr> 

/** endloop **/ 
</table> 
/** include footer.html **/ 
Copyright &copy; Today 
```

此文件包含形式为`/** <directive> <data> **/`的*标签*，其中数据是可选的单个单词，指令如下：

+   `include`: 在此处复制另一个文件的内容

+   `variable`: 在此处插入变量的内容

+   `loopover`: 对于列表变量重复循环的内容

+   `endloop`: 信号循环文本的结束

+   `loopvar`: 从正在循环的列表中插入单个值

这个模板将根据传入的变量渲染不同的页面。这些变量将从所谓的上下文文件传入。这将编码为一个`json`对象，其中的键代表所涉及的变量。我的上下文文件可能看起来像这样，但你会根据自己的需求进行修改：

```py
{ 
    "name": "Dusty", 
    "book_list": [ 
        "Thief Of Time", 
        "The Thief", 
        "Snow Crash", 
        "Lathe Of Heaven" 
    ] 
} 
```

在我们进行实际的字符串处理之前，让我们为处理文件和从命令行获取数据编写一些面向对象的基础代码：

```py
import re 
import sys 
import json 
from pathlib import Path 

DIRECTIVE_RE = re.compile( 
    r'/\*\*\s*(include|variable|loopover|endloop|loopvar)' 
    r'\s*([^ *]*)\s*\*\*/') 

class TemplateEngine: 
    def __init__(self, infilename, outfilename, contextfilename): 
        self.template = open(infilename).read() 
        self.working_dir = Path(infilename).absolute().parent 
        self.pos = 0 
        self.outfile = open(outfilename, 'w') 
        with open(contextfilename) as contextfile: 
            self.context = json.load(contextfile) 

    def process(self): 
        print("PROCESSING...") 

if __name__ == '__main__': 
    infilename, outfilename, contextfilename = sys.argv[1:] 
    engine = TemplateEngine(infilename, outfilename, contextfilename) 
    engine.process() 
```

这一切都很基础，我们创建一个类，并通过命令行传入一些变量来初始化它。

注意我们如何尝试通过将正则表达式拆分为两行来使其更易于阅读？我们使用原始字符串（r 前缀），因此我们不需要对所有的反斜杠进行双重转义。这在正则表达式中很常见，但仍然很混乱。（正则表达式总是这样，但它们通常值得这么做。）

`pos`表示我们正在处理的内容中的当前字符；我们很快就会看到更多关于它的内容。

现在剩下的只是实现`process`方法。有几种方法可以做到这一点。让我们以一种相当明确的方式来做。

`process`方法必须找到与正则表达式匹配的每个指令，并对其进行适当处理。然而，它还必须注意将每个指令之前、之后和之间的正常文本输出到输出文件，且不进行修改。

正则表达式的编译版本的一个好特点是，我们可以通过传递`pos`关键字参数来告诉`search`方法从特定位置开始搜索。如果我们暂时将使用指令执行适当工作定义为*忽略指令并从输出文件中删除它*，那么我们的过程循环看起来相当简单：

```py
def process(self): 
    match = DIRECTIVE_RE.search(self.template, pos=self.pos) 
    while match: 
        self.outfile.write(self.template[self.pos:match.start()]) 
        self.pos = match.end() 
        match = DIRECTIVE_RE.search(self.template, pos=self.pos) 
    self.outfile.write(self.template[self.pos:]) 
```

在英语中，这个函数在文本中找到第一个与正则表达式匹配的字符串，输出从当前位置到该匹配开始的所有内容，然后将位置移动到上述匹配的末尾。一旦没有更多匹配项，它将输出从上次位置以来的所有内容。

当然，在模板引擎中忽略指令是非常无用的，所以让我们用委托到类上不同方法的代码替换那个位置推进行：

```py
def process(self): 
    match = DIRECTIVE_RE.search(self.template, pos=self.pos) 
    while match: 
        self.outfile.write(self.template[self.pos:match.start()]) 
        directive, argument = match.groups() 
        method_name = 'process_{}'.format(directive) 
        getattr(self, method_name)(match, argument) 
        match = DIRECTIVE_RE.search(self.template, pos=self.pos) 
    self.outfile.write(self.template[self.pos:]) 
```

因此，我们从正则表达式中提取指令和单个参数。指令变成了方法名，我们动态地在`self`对象上查找该方法名（如果模板编写者提供了无效的指令，这里会有一些错误处理，会更好）。我们将`match`对象和参数传递给该方法，并假设该方法将适当地处理所有事情，包括移动`pos`指针。

现在我们已经实现了面向对象架构的这一步，实现被委托的方法实际上相当简单。`include`和`variable`指令完全直接：

```py
def process_include(self, match, argument): 
    with (self.working_dir / argument).open() as includefile: 
        self.outfile.write(includefile.read()) 
        self.pos = match.end() 

def process_variable(self, match, argument): 
    self.outfile.write(self.context.get(argument, '')) 
    self.pos = match.end() 
```

第一个指令简单地查找包含的文件并插入文件内容，而第二个指令在上下文字典（在`__init__`方法中从`json`加载）中查找变量名，如果不存在则默认为空字符串。

处理循环的三个方法稍微复杂一些，因为它们必须在三者之间共享状态。为了简单起见（我敢肯定你急于看到这个漫长章节的结尾——我们几乎到了！），我们将使用类本身的实例变量来处理这种情况。作为一个练习，你可能想要考虑更好的架构方式，尤其是在阅读接下来的三个章节之后：

```py
    def process_loopover(self, match, argument): 
        self.loop_index = 0 
        self.loop_list = self.context.get(argument, []) 
        self.pos = self.loop_pos = match.end() 

    def process_loopvar(self, match, argument): 
        self.outfile.write(self.loop_list[self.loop_index]) 
        self.pos = match.end() 

    def process_endloop(self, match, argument): 
        self.loop_index += 1 
        if self.loop_index >= len(self.loop_list): 
            self.pos = match.end() 
            del self.loop_index 
            del self.loop_list 
            del self.loop_pos 
        else: 
            self.pos = self.loop_pos 
```

当我们遇到`loopover`指令时，我们不需要输出任何内容，但我们必须设置三个变量的初始状态。`loop_list`变量假设是从上下文字典中拉取的列表。`loop_index`变量指示在这个循环迭代中应该输出列表中的哪个位置，而`loop_pos`被存储起来，以便我们知道在到达循环末尾时跳回的位置。

`loopvar`指令输出`loop_list`变量当前位置的价值，并跳转到指令的末尾。请注意，它不会增加循环索引，因为`loopvar`指令可以在循环内部多次调用。

`endloop`指令更复杂。它确定`loop_list`中是否有更多元素；如果有，它就跳回循环的开始，增加索引。否则，它重置用于处理循环的所有变量，并跳转到指令的末尾，以便引擎可以继续进行下一个匹配。

注意，这个特定的循环机制非常脆弱；如果模板设计者尝试嵌套循环或忘记调用 `endloop`，结果可能会很糟糕。我们需要更多的错误检查，并且可能需要存储更多的循环状态，以便将其作为一个生产平台。但我承诺本章的结尾即将到来，所以让我们在看到我们的示例模板及其上下文如何渲染后，直接进入练习：

```py
<html>

<body>

<h1>This is the title of the front page</h1>
<a href="link1.html">First Link</a>
<a href="link2.html">Second Link</a>

<p>My name is Dusty. This is the content of my front page. It goes below the menu.</p>
<table>
    <tr>
        <th>Favourite Books</th>
    </tr>

    <tr>
        <td>Thief Of Time</td>
    </tr>

    <tr>
        <td>The Thief</td>
    </tr>

    <tr>
        <td>Snow Crash</td>
    </tr>

    <tr>
        <td>Lathe Of Heaven</td>
    </tr>

</table>
</body>

</html>
 Copyright &copy; Today
```

由于我们计划模板的方式，有一些奇怪的换行效果，但它们按预期工作。

# 练习

在本章中，我们涵盖了广泛的主题，从字符串到正则表达式，再到对象序列化，现在考虑一下如何将这些想法应用到你的代码中。

Python 字符串非常灵活，Python 是一个用于字符串操作的极其强大的工具。如果你在日常工作中不经常进行字符串处理，尝试设计一个专门用于字符串操作的工具。尝试提出一些创新的想法，如果你卡住了，可以考虑编写一个网络日志分析器（每小时有多少请求？有多少人访问了五个以上的页面？）或一个模板工具，该工具用其他文件的内容替换某些变量名。

投入大量时间去玩转字符串格式化运算符，直到你记住了它们的语法。编写一些模板字符串和对象，将它们传递给格式化函数，看看能得到什么样的输出。尝试一些异国风情的格式化运算符，比如百分比或十六进制表示法。尝试使用填充和对齐运算符，看看它们在整数、字符串和浮点数上的表现有何不同。考虑编写一个包含 `__format__` 方法的类；我们之前没有详细讨论这一点，但探索一下你可以如何自定义格式化。

确保你理解 `bytes` 和 `str` 对象之间的区别。在 Python 的旧版本中，这种区别非常复杂（那时没有 `bytes`，`str` 既可以作为 `bytes` 也可以作为 `str`，除非我们需要非 ASCII 字符，在这种情况下有一个单独的 `unicode` 对象，它类似于 Python 3 的 `str` 类。这比听起来还要复杂！）。现在它更清晰了；`bytes` 用于二进制数据，而 `str` 用于字符数据。唯一棘手的部分是知道何时以及如何在这两者之间进行转换。为了练习，尝试将文本数据写入一个以 `bytes` 写入模式打开的文件（你将不得不自己编码文本），然后从同一个文件中读取。

对 `bytearray` 进行一些实验。看看它如何同时像 `bytes` 对象、列表或容器对象一样工作。尝试写入一个缓冲区，该缓冲区在数据达到一定长度之前保持数据在字节数组中。你可以通过使用 `time.sleep` 调用来模拟将数据放入缓冲区的代码，以确保数据不会太快到达。

在线学习正则表达式。再深入一些。特别是学习关于命名组、贪婪匹配与懒惰匹配以及正则表达式标志，这三个我们在本章中没有涉及到的特性。有意识地决定何时不使用它们。很多人对正则表达式有非常强烈的看法，要么过度使用，要么完全拒绝使用。试着让自己只在适当的时候使用它们，并找出何时是适当的时候。

如果你曾经编写过适配器，从文件或数据库中加载少量数据并将其转换为对象，考虑使用 pickle。pickle 不适合存储大量数据，但它们可以用于加载配置或其他简单对象。尝试用多种方式编码：使用 pickle、文本文件或小型数据库。你发现哪种最容易使用？

尝试对数据进行序列化实验，然后修改包含数据的类，并将 pickle 加载到新类中。哪些可行？哪些不可行？是否有方法可以对类进行重大更改，例如重命名属性或将它拆分为两个新属性，同时还能从旧的 pickle 中提取数据？（提示：在每个对象上放置一个私有的 pickle 版本号，并在更改类时更新它；你可以在`__setstate__`中添加迁移路径。）

如果你进行过任何网络开发，尝试使用 JSON 序列化器做一些实验。我个人更喜欢只序列化标准 JSON 可序列化对象，而不是编写自定义编码器或`object_hooks`，但期望的效果实际上取决于前端（通常是 JavaScript）和后端代码之间的交互。

在模板引擎中创建一些接受多个或任意数量参数的新指令。你可能需要修改正则表达式或添加新的。查看 Django 项目的在线文档，看看是否有其他你想要与之合作的模板标签。尝试模仿它们的过滤器语法，而不是使用`variable`标签。

当你学习了迭代和协程后，回顾本章内容，看看你是否能想出一个更紧凑的方式来表示相关指令之间的状态，例如循环。

# 摘要

我们在本章中介绍了字符串操作、正则表达式和对象序列化。可以使用强大的字符串格式化系统将硬编码的字符串和程序变量组合成可输出的字符串。区分二进制和文本数据非常重要，`bytes`和`str`有特定的用途，必须理解。两者都是不可变的，但在操作字节时可以使用`bytearray`类型。

正则表达式是一个复杂的话题，我们只是触及了表面。有许多方法可以序列化 Python 数据；pickle 和 JSON 是最受欢迎的两种。

在下一章中，我们将探讨一个对 Python 编程至关重要的设计模式，它甚至得到了特殊的语法支持：迭代器模式。
