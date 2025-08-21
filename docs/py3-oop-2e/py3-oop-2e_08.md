# 第八章：字符串和序列化

在我们涉及更高级别的设计模式之前，让我们深入研究 Python 中最常见的对象之一：字符串。我们会发现字符串比看上去更复杂，还会涵盖搜索字符串的模式和序列化数据以便存储或传输。

特别是，我们将讨论：

+   字符串、字节和字节数组的复杂性

+   字符串格式化的内在和外在

+   几种序列化数据的方法

+   神秘的正则表达式

# 字符串

字符串是 Python 中的基本原语；我们几乎在我们迄今讨论的每个例子中都使用了它们。它们所做的就是表示一个不可变的字符序列。然而，虽然你以前可能没有考虑过，"字符"是一个有点模糊的词；Python 字符串能表示重音字符的序列吗？中文字符？希腊、西里尔或波斯字符呢？

在 Python 3 中，答案是肯定的。Python 字符串都以 Unicode 表示，这是一个可以表示地球上任何语言中的几乎任何字符的字符定义标准（还包括一些虚构的语言和随机字符）。这在很大程度上是无缝的。因此，让我们把 Python 3 字符串看作是不可变的 Unicode 字符序列。那么我们可以用这个不可变序列做什么呢？我们在之前的例子中已经提到了许多字符串可以被操作的方式，但让我们快速地在一个地方概括一下：字符串理论的速成课程！

## 字符串操作

如你所知，可以通过用单引号或双引号包裹一系列字符来在 Python 中创建字符串。可以使用三个引号字符轻松创建多行字符串，并且可以通过将它们并排放置来连接多个硬编码字符串。以下是一些例子：

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

解释器会自动将最后一个字符串组合成一个字符串。也可以使用`+`运算符连接字符串（如`"hello " + "world"`）。当然，字符串不一定是硬编码的。它们也可以来自各种外部来源，如文本文件、用户输入，或者在网络上编码。

### 提示

相邻字符串的自动连接可能会导致一些滑稽的错误，当逗号丢失时。然而，当需要将长字符串放置在函数调用中而不超过 Python 风格指南建议的 79 个字符行长度限制时，这是非常有用的。

与其他序列一样，字符串可以被迭代（逐个字符），索引，切片或连接。语法与列表相同。

`str`类上有许多方法，可以使操作字符串更容易。Python 解释器中的`dir`和`help`命令可以告诉我们如何使用它们；我们将直接考虑一些更常见的方法。

几种布尔方便方法帮助我们确定字符串中的字符是否与某种模式匹配。以下是这些方法的摘要。其中大多数方法，如`isalpha`，`isupper`/`islower`，`startswith`/`endswith`都有明显的解释。`isspace`方法也相当明显，但请记住，所有空白字符（包括制表符、换行符）都被考虑在内，而不仅仅是空格字符。

`istitle`方法返回`True`，如果每个单词的第一个字符都是大写，其他字符都是小写。请注意，它并不严格执行英语的标题格式定义。例如，Leigh Hunt 的诗歌"The Glove and the Lions"应该是一个有效的标题，即使并非所有单词都是大写。Robert Service 的"The Cremation of Sam McGee"也应该是一个有效的标题，即使最后一个单词中间有一个大写字母。

对于`isdigit`，`isdecimal`和`isnumeric`方法要小心，因为它们比您期望的更微妙。许多 Unicode 字符被认为是数字，除了我们习惯的十个数字之外。更糟糕的是，我们用来从字符串构造浮点数的句点字符不被视为十进制字符，因此`'45.2'.isdecimal()`返回`False`。真正的十进制字符由 Unicode 值 0660 表示，如 45.2 中的 0660（或`45\u06602`）。此外，这些方法不验证字符串是否为有效数字；"127.0.0.1"对所有三种方法都返回`True`。我们可能认为应该使用该十进制字符而不是句点来表示所有数字数量，但将该字符传递给`float()`或`int()`构造函数会将该十进制字符转换为零：

```py
>>> float('45\u06602')
4502.0

```

用于模式匹配的其他有用方法不返回布尔值。`count`方法告诉我们给定子字符串在字符串中出现了多少次，而`find`，`index`，`rfind`和`rindex`告诉我们给定子字符串在原始字符串中的位置。两个`r`（表示“右”或“反向”）方法从字符串的末尾开始搜索。如果找不到子字符串，`find`方法返回`-1`，而`index`在这种情况下会引发`ValueError`。看看其中一些方法的实际应用：

```py
>>> s = "hello world"
>>> s.count('l')
3
>>> s.find('l')
2
>>> s.rindex('m')
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
ValueError: substring not found

```

其余大多数字符串方法返回字符串的转换。`upper`，`lower`，`capitalize`和`title`方法创建具有给定格式的所有字母字符的新字符串。`translate`方法可以使用字典将任意输入字符映射到指定的输出字符。

对于所有这些方法，请注意输入字符串保持不变；而是返回一个全新的`str`实例。如果我们需要操作结果字符串，我们应该将其赋值给一个新变量，如`new_value = value.capitalize()`。通常，一旦我们执行了转换，我们就不再需要旧值了，因此一个常见的习惯是将其赋值给相同的变量，如`value = value.title()`。

最后，一些字符串方法返回或操作列表。`split`方法接受一个子字符串，并在该子字符串出现的地方将字符串拆分为字符串列表。您可以将数字作为第二个参数传递以限制结果字符串的数量。`rsplit`如果不限制字符串的数量，则行为与`split`相同，但如果您提供了限制，它将从字符串的末尾开始拆分。`partition`和`rpartition`方法仅在子字符串的第一次或最后一次出现时拆分字符串，并返回一个包含三个值的元组：子字符串之前的字符，子字符串本身和子字符串之后的字符。

作为`split`的反向操作，`join`方法接受一个字符串列表，并通过将原始字符串放在它们之间来返回所有这些字符串组合在一起的字符串。`replace`方法接受两个参数，并返回一个字符串，其中第一个参数的每个实例都已被第二个参数替换。以下是其中一些方法的实际应用：

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

这就是最常见的`str`类上的方法的快速浏览！现在，让我们看看 Python 3 的方法，用于组合字符串和变量以创建新字符串。

## 字符串格式化

Python 3 具有强大的字符串格式化和模板机制，允许我们构造由硬编码文本和插入的变量组成的字符串。我们在许多先前的示例中使用过它，但它比我们使用的简单格式化说明符要灵活得多。

任何字符串都可以通过在其上调用`format()`方法将其转换为格式化字符串。此方法返回一个新字符串，其中输入字符串中的特定字符已被替换为作为参数和关键字参数传递给函数的值。`format`方法不需要固定的参数集；在内部，它使用了我们在第七章中讨论的`*args`和`**kwargs`语法，*Python 面向对象的快捷方式*。

在格式化字符串中替换的特殊字符是开放和关闭的大括号字符：`{`和`}`。我们可以在字符串中插入这些对，并且它们将按顺序被任何传递给`str.format`方法的位置参数替换：

```py
template = "Hello {}, you are currently {}."
print(template.format('Dusty', 'writing'))
```

如果我们运行这些语句，它将按顺序用变量替换大括号：

```py
Hello Dusty, you are currently writing.

```

如果我们想要在一个字符串中重用变量或者决定在不同位置使用它们，这种基本语法就不是特别有用。我们可以在花括号中放置从零开始的整数，以告诉格式化程序在字符串的特定位置插入哪个位置变量。让我们重复一下名字：

```py
template = "Hello {0}, you are {1}. Your name is {0}."
print(template.format('Dusty', 'writing'))
```

如果我们使用这些整数索引，我们必须在所有变量中使用它们。我们不能将空大括号与位置索引混合使用。例如，这段代码会因为适当的`ValueError`异常而失败：

```py
template = "Hello {}, you are {}. Your name is {0}."
print(template.format('Dusty', 'writing'))
```

### 转义大括号

大括号字符在字符串中通常很有用，除了格式化之外。我们需要一种方法来在我们希望它们以它们自己的形式显示而不是被替换的情况下对它们进行转义。这可以通过加倍大括号来实现。例如，我们可以使用 Python 来格式化一个基本的 Java 程序：

```py
template = """
public class {0} {{
    public static void main(String[] args) {{
        System.out.println("{1}");
    }}
}}"""

print(template.format("MyClass", "print('hello world')"));
```

在模板中，无论我们看到`{{`或`}}`序列，也就是包围 Java 类和方法定义的大括号，我们知道`format`方法将用单个大括号替换它们，而不是一些传递给`format`方法的参数。以下是输出：

```py
public class MyClass {
 **public static void main(String[] args) {
 **System.out.println("print('hello world')");
 **}
}

```

输出的类名和内容已被替换为两个参数，而双大括号已被替换为单大括号，从而给我们一个有效的 Java 文件。结果是，这是一个打印最简单的 Java 程序的最简单的可能的 Python 程序，可以打印最简单的可能的 Python 程序！

### 关键字参数

如果我们要格式化复杂的字符串，要记住参数的顺序或者更新模板如果我们选择插入一个新的参数可能会变得很繁琐。因此，`format`方法允许我们在大括号内指定名称而不是数字。然后将命名变量作为关键字参数传递给`format`方法：

```py
template = """
From: <{from_email}>
To: <{to_email}>
Subject: {subject}

{message}"""
print(template.format(
    from_email = "a@example.com",
    to_email = "b@example.com",
 **message = "Here's some mail for you. "
 **" Hope you enjoy the message!",
    subject = "You have mail!"
    ))
```

我们还可以混合使用索引和关键字参数（与所有 Python 函数调用一样，关键字参数必须跟在位置参数后面）。我们甚至可以将未标记的位置大括号与关键字参数混合使用：

```py
print("{} {label} {}".format("x", "y", label="z"))
```

如预期的那样，这段代码输出：

```py
x z y

```

### 容器查找

我们不仅限于将简单的字符串变量传递给`format`方法。任何原始类型，如整数或浮点数都可以打印。更有趣的是，可以使用复杂对象，包括列表、元组、字典和任意对象，并且可以从`format`字符串中访问这些对象的索引和变量（但不能访问方法）。

例如，如果我们的电子邮件消息将发件人和收件人的电子邮件地址分组到一个元组中，并将主题和消息放在一个字典中，出于某种原因（也许是因为这是现有`send_mail`函数所需的输入），我们可以这样格式化它：

```py
emails = ("a@example.com", "b@example.com")
message = {
        'subject': "You Have Mail!",
        'message': "Here's some mail for you!"
        }
template = """
From: <{0[0]}>
To: <{0[1]}>
Subject: {message[subject]}
{message[message]}"""
print(template.format(emails, message=message))
```

模板字符串中大括号内的变量看起来有点奇怪，所以让我们看看它们在做什么。我们已经将一个参数作为基于位置的参数传递，另一个作为关键字参数。两个电子邮件地址通过`0[x]`查找，其中`x`可以是`0`或`1`。初始的零表示，与其他基于位置的参数一样，传递给`format`的第一个位置参数（在这种情况下是`emails`元组）。

带有数字的方括号是我们在常规 Python 代码中看到的相同类型的索引查找，所以`0[0]`映射到`emails[0]`，在`emails`元组中。索引语法适用于任何可索引的对象，所以当我们访问`message[subject]`时，我们看到类似的行为，除了这次我们在字典中查找一个字符串键。请注意，与 Python 代码不同的是，在字典查找中我们不需要在字符串周围加上引号。

如果我们有嵌套的数据结构，甚至可以进行多层查找。我建议不要经常这样做，因为模板字符串很快就变得难以理解。如果我们有一个包含元组的字典，我们可以这样做：

```py
emails = ("a@example.com", "b@example.com")
message = {
        'emails': emails,
        'subject': "You Have Mail!",
        'message': "Here's some mail for you!"
        }
template = """
From: <{0[emails][0]}>
To: <{0[emails][1]}>
Subject: {0[subject]}
{0[message]}"""
print(template.format(message))
```

### 对象查找

索引使`format`查找功能强大，但我们还没有完成！我们还可以将任意对象作为参数传递，并使用点符号来查找这些对象的属性。让我们再次更改我们的电子邮件消息数据，这次是一个类：

```py
class EMail:
    def __init__(self, from_addr, to_addr, subject, message):
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.subject = subject
        self.message = message

email = EMail("a@example.com", "b@example.com",
        "You Have Mail!",
         "Here's some mail for you!")

template = """
From: <{0.from_addr}>
To: <{0.to_addr}>
Subject: {0.subject}

{0.message}"""
print(template.format(email))
```

在这个例子中，模板可能比之前的例子更易读，但创建一个电子邮件类的开销会给 Python 代码增加复杂性。为了将对象包含在模板中而创建一个类是愚蠢的。通常，如果我们要格式化的对象已经存在，我们会使用这种查找。所有的例子都是如此；如果我们有一个元组、列表或字典，我们会直接将其传递到模板中。否则，我们只需创建一组简单的位置参数和关键字参数。

### 使其看起来正确

在模板字符串中包含变量是很好的，但有时变量需要一点强制转换才能使它们在输出中看起来正确。例如，如果我们在货币计算中，可能会得到一个我们不想在模板中显示的长小数：

```py
subtotal = 12.32
tax = subtotal * 0.07
total = subtotal + tax

print("Sub: ${0} Tax: ${1} Total: ${total}".format(
    subtotal, tax, total=total))
```

如果我们运行这个格式化代码，输出看起来并不像正确的货币：

```py
Sub: $12.32 Tax: $0.8624 Total: $13.182400000000001

```

### 注意

从技术上讲，我们不应该在货币计算中使用浮点数；我们应该使用`decimal.Decimal()`对象来构造。浮点数是危险的，因为它们的计算在特定精度水平之后本质上是不准确的。但我们正在看字符串，而不是浮点数，货币是格式化的一个很好的例子！

为了修复前面的`format`字符串，我们可以在花括号内包含一些额外的信息，以调整参数的格式。我们可以定制很多东西，但花括号内的基本语法是相同的；首先，我们使用早期的布局（位置、关键字、索引、属性访问）中适合的布局来指定我们想要放入模板字符串中的变量。然后我们跟着一个冒号，然后是特定的格式语法。这是一个改进版：

```py
print("Sub: ${0:0.2f} Tax: ${1:0.2f} "
        "Total: ${total:0.2f}".format(
            subtotal, tax, total=total))
```

冒号后面的`0.2f`格式说明符基本上是这样说的，从左到右：对于小于一的值，确保小数点左侧显示一个零；显示小数点后两位；将输入值格式化为浮点数。

我们还可以指定每个数字在屏幕上占据特定数量的字符，方法是在精度的句点之前放置一个值。这对于输出表格数据非常有用，例如：

```py
orders = [('burger', 2, 5),
        ('fries', 3.5, 1),
        ('cola', 1.75, 3)]

print("PRODUCT    QUANTITY    PRICE    SUBTOTAL")
for product, price, quantity in orders:
    subtotal = price * quantity
 **print("{0:10s}{1: ⁹d}    ${2: <8.2f}${3: >7.2f}".format(
 **product, quantity, price, subtotal))

```

好的，这是一个看起来相当可怕的格式字符串，让我们看看它是如何工作的，然后再将其分解成可理解的部分：

```py
PRODUCT    QUANTITY    PRICE    SUBTOTAL
burger        5        $2.00    $  10.00
fries         1        $3.50    $   3.50
cola          3        $1.75    $   5.25

```

厉害！那么，这实际上是如何发生的呢？在`for`循环中的每一行中，我们正在格式化四个变量。第一个变量是一个字符串，并且使用`{0:10s}`进行格式化。`s`表示它是一个字符串变量，`10`表示它应该占用十个字符。默认情况下，对于字符串，如果字符串的长度小于指定的字符数，它会在字符串的右侧附加空格，使其足够长（但要注意，如果原始字符串太长，它不会被截断！）。我们可以更改这种行为（在格式字符串中填充其他字符或更改对齐方式），就像我们对下一个值`quantity`所做的那样。

`quantity`值的格式化程序是`{1: ⁹d}`。`d`表示整数值。`9`告诉我们该值应该占用九个字符。但是对于整数，额外的字符默认情况下是零，而不是空格。这看起来有点奇怪。因此，我们明确指定一个空格（在冒号后面）作为填充字符。插入符`^`告诉我们数字应该对齐在这个可用填充的中心；这使得列看起来更专业一些。说明符必须按正确的顺序，尽管所有都是可选的：首先填充，然后对齐，然后大小，最后类型。

我们对价格和小计的说明符做了类似的处理。对于`price`，我们使用`{2: <8.2f}`，对于`subtotal`，我们使用`{3: >7.2f}`。在这两种情况下，我们指定空格作为填充字符，但是我们分别使用`<`和`>`符号，表示数字应该在八个或七个字符的最小空间内左对齐或右对齐。此外，每个浮点数应该格式化为两位小数。

不同类型的“类型”字符也会影响格式化输出。我们已经看到了`s`、`d`和`f`类型，分别代表字符串、整数和浮点数。大多数其他格式说明符都是这些类型的替代版本；例如，`o`代表八进制格式，`X`代表十六进制格式。`n`类型说明符可以用于在当前区域设置的格式中格式化整数分隔符。对于浮点数，`%`类型将乘以 100 并将浮点数格式化为百分比。

虽然这些标准格式适用于大多数内置对象，但其他对象也可以定义非标准的说明符。例如，如果我们将`datetime`对象传递给`format`，我们可以使用`datetime.strftime`函数中使用的说明符，如下所示：

```py
import datetime
print("{0:%Y-%m-%d %I:%M%p }".format(
    datetime.datetime.now()))
```

甚至可以为我们自己创建的对象编写自定义格式化程序，但这超出了本书的范围。如果您需要在代码中执行此操作，请查看如何覆盖`__format__`特殊方法。最全面的说明可以在 PEP 3101 中找到[`www.python.org/dev/peps/pep-3101/`](http://www.python.org/dev/peps/pep-3101/)，尽管细节有点枯燥。您可以通过网络搜索找到更易理解的教程。

Python 的格式化语法非常灵活，但是很难记住。我每天都在使用它，但偶尔还是不得不查阅文档中忘记的概念。它也不足以满足严肃的模板需求，比如生成网页。如果您需要做更多的字符串基本格式化，可以查看几个第三方模板库。

## 字符串是 Unicode

在本节的开头，我们将字符串定义为不可变的 Unicode 字符集合。这实际上有时会使事情变得非常复杂，因为 Unicode 实际上并不是一种存储格式。例如，如果从文件或套接字中获取字节字符串，它们实际上不会是 Unicode。它们实际上是内置类型`bytes`。字节是不可变的序列...嗯，字节。字节是计算机中最低级别的存储格式。它们代表 8 位，通常描述为介于 0 和 255 之间的整数，或者介于 0 和 FF 之间的十六进制等价物。字节不代表任何特定的内容；一系列字节可以存储编码字符串的字符，或者图像中的像素。

如果我们打印一个字节对象，任何映射到 ASCII 表示的字节都将打印为它们原始的字符，而非 ASCII 字节（无论它们是二进制数据还是其他字符）都将以`\x`转义序列转义的十六进制代码打印出来。你可能会觉得奇怪，一个字节，表示为一个整数，可以映射到一个 ASCII 字符。但 ASCII 实际上只是一个代码，其中每个字母都由不同的字节模式表示，因此，不同的整数。字符“a”由与整数 97 相同的字节表示，这是十六进制数 0x61。具体来说，所有这些都是对二进制模式 01100001 的解释。

许多 I/O 操作只知道如何处理`bytes`，即使字节对象引用文本数据。因此，了解如何在`bytes`和 Unicode 之间转换至关重要。

问题在于有许多种方法可以将`bytes`映射到 Unicode 文本。字节是机器可读的值，而文本是一种人类可读的格式。它们之间是一种编码，它将给定的字节序列映射到给定的文本字符序列。

然而，有多种这样的编码（ASCII 只是其中之一）。当使用不同的编码进行映射时，相同的字节序列代表完全不同的文本字符！因此，`bytes`必须使用与它们编码时相同的字符集进行解码。如果我们收到未知编码的字节而没有指定编码，我们能做的最好的事情就是猜测它们的编码格式，而我们可能会猜错。

### 将字节转换为文本

如果我们从某个地方有一个`bytes`数组，我们可以使用`bytes`类的`.decode`方法将其转换为 Unicode。这个方法接受一个字符串作为字符编码的名称。有许多这样的名称；西方语言的常见名称包括 ASCII、UTF-8 和拉丁-1。

字节序列（十六进制）63 6c 69 63 68 e9，实际上代表了拉丁-1 编码中单词 cliché的字符。以下示例将对这个字节序列进行编码，并使用拉丁-1 编码将其转换为 Unicode 字符串：

```py
characters = b'\x63\x6c\x69\x63\x68\xe9'
print(characters)
print(characters.decode("latin-1"))

```

第一行创建了一个`bytes`对象；字符串前面的`b`字符告诉我们，我们正在定义一个`bytes`对象，而不是一个普通的 Unicode 字符串。在字符串中，每个字节都使用十六进制数字指定。在这种情况下，`\x`字符在字节字符串中转义，并且每个都表示“下面的两个字符使用十六进制数字表示一个字节”。

只要我们使用了理解拉丁-1 编码的 shell，两个`print`调用将输出以下字符串：

```py
b'clich\xe9'
cliché

```

第一个`print`语句将 ASCII 字符的字节呈现为它们自己。未知的（对 ASCII 来说是未知的）字符保持在其转义的十六进制格式中。输出包括一行开头的`b`字符，提醒我们这是一个`bytes`表示，而不是一个字符串。

下一个调用使用 latin-1 编码解码字符串。`decode`方法返回一个带有正确字符的普通（Unicode）字符串。然而，如果我们使用西里尔文“iso8859-5”编码解码相同的字符串，我们最终会得到字符串'clichщ'！这是因为`\xe9`字节在这两种编码中映射到不同的字符。

### 将文本转换为字节

如果我们需要将传入的字节转换为 Unicode，显然我们也会遇到将传出的 Unicode 转换为字节序列的情况。这是通过`str`类上的`encode`方法完成的，就像`decode`方法一样，需要一个字符集。以下代码创建一个 Unicode 字符串，并以不同的字符集对其进行编码：

```py
characters = "cliché"
print(characters.encode("UTF-8"))
print(characters.encode("latin-1"))
print(characters.encode("CP437"))
print(characters.encode("ascii"))
```

前三种编码为重音字符创建了不同的字节集。第四种甚至无法处理该字节：

```py
b'clich\xc3\xa9'
b'clich\xe9'
b'clich\x82'
Traceback (most recent call last):
 **File "1261_10_16_decode_unicode.py", line 5, in <module>
 **print(characters.encode("ascii"))
UnicodeEncodeError: 'ascii' codec can't encode character '\xe9' in position 5: ordinal not in range(128)

```

现在你明白编码的重要性了吗？重音字符对于每种编码都表示为不同的字节；如果我们在解码字节为文本时使用错误的编码，我们会得到错误的字符。

在最后一种情况下，异常并不总是期望的行为；可能有些情况下我们希望以不同的方式处理未知字符。`encode`方法接受一个名为`errors`的可选字符串参数，可以定义如何处理这些字符。这个字符串可以是以下之一：

+   `strict`

+   `replace`

+   `ignore`

+   `xmlcharrefreplace`

`strict`替换策略是我们刚刚看到的默认值。当遇到一个字节序列在请求的编码中没有有效表示时，会引发异常。当使用`replace`策略时，字符将被替换为不同的字符；在 ASCII 中，它是一个问号；其他编码可能使用不同的符号，比如一个空盒子。`ignore`策略简单地丢弃它不理解的任何字节，而`xmlcharrefreplace`策略创建一个代表 Unicode 字符的`xml`实体。这在将未知字符串转换为 XML 文档中使用时非常有用。以下是每种策略对我们示例单词的影响：

| 策略 | "cliché".encode("ascii", strategy) |
| --- | --- |
| `replace` | `b'clich?'` |
| `ignore` | `b'clich'` |
| `xmlcharrefreplace` | `b'cliché'` |

可以调用`str.encode`和`bytes.decode`方法而不传递编码字符串。编码将设置为当前平台的默认编码。这将取决于当前操作系统和区域设置；您可以使用`sys.getdefaultencoding()`函数查找它。不过，通常最好明确指定编码，因为平台的默认编码可能会更改，或者程序可能有一天会扩展到处理更多来源的文本。

如果您要对文本进行编码，但不知道要使用哪种编码，最好使用 UTF-8 编码。UTF-8 能够表示任何 Unicode 字符。在现代软件中，它是确保以任何语言甚至多种语言交换文档的事实标准编码。其他各种可能的编码对于传统文档或仍然默认使用不同字符集的地区非常有用。

UTF-8 编码使用一个字节来表示 ASCII 和其他常见字符，对于更复杂的字符最多使用四个字节。UTF-8 很特殊，因为它向后兼容 ASCII；使用 UTF-8 编码的任何 ASCII 文档将与原始 ASCII 文档相同。

### 提示

我永远记不住是使用`encode`还是`decode`来将二进制字节转换为 Unicode。我总是希望这些方法的名称改为"to_binary"和"from_binary"。如果您有同样的问题，请尝试在脑海中用"binary"替换"code"；"enbinary"和"debinary"与"to_binary"和"from_binary"非常接近。自从想出这个记忆方法以来，我已经节省了很多时间，因为不用再查找方法帮助文件。

## 可变字节字符串

`bytes`类型和`str`一样是不可变的。我们可以在`bytes`对象上使用索引和切片表示法，并搜索特定的字节序列，但我们不能扩展或修改它们。当处理 I/O 时，这可能非常不方便，因为通常需要缓冲传入或传出的字节，直到它们准备好发送。例如，如果我们从套接字接收数据，可能需要多次`recv`调用才能接收到整个消息。

这就是`bytearray`内置的作用。这种类型的行为有点像列表，只是它只包含字节。该类的构造函数可以接受一个`bytes`对象来初始化它。`extend`方法可以用来附加另一个`bytes`对象到现有的数组中（例如，当更多的数据来自套接字或其他 I/O 通道时）。

切片表示法可以在`bytearray`上使用，以内联修改项目。例如，这段代码从`bytes`对象构造了一个`bytearray`，然后替换了两个字节：

```py
b = bytearray(b"abcdefgh")
b[4:6] = b"\x15\xa3"
print(b)
```

输出如下：

```py
bytearray(b'abcd\x15\xa3gh')

```

要小心；如果我们想要操作`bytearray`中的单个元素，它将期望我们传递一个介于 0 和 255 之间的整数作为值。这个整数代表一个特定的`bytes`模式。如果我们尝试传递一个字符或`bytes`对象，它将引发异常。

单字节字符可以使用`ord`（ordinal 的缩写）函数转换为整数。这个函数返回单个字符的整数表示：

```py
b = bytearray(b'abcdef')
b[3] = ord(b'g')
b[4] = 68
print(b)
```

输出如下：

```py
bytearray(b'abcgDf')

```

在构造数组之后，我们用字节 103 替换索引为`3`（第四个字符，因为索引从`0`开始，就像列表一样）。这个整数是由`ord`函数返回的，是小写`g`的 ASCII 字符。为了说明，我们还用字节号`68`替换了上一个字符，它映射到大写`D`的 ASCII 字符。

`bytearray`类型有一些方法，使它可以像列表一样行为（例如，我们可以向其附加整数字节），但也像`bytes`对象；我们可以使用`count`和`find`方法，就像它们在`bytes`或`str`对象上的行为一样。不同之处在于`bytearray`是一种可变类型，这对于从特定输入源构建复杂的字节序列是有用的。

# 正则表达式

你知道使用面向对象的原则真的很难做的事情是什么吗？解析字符串以匹配任意模式，就是这样。已经有相当多的学术论文使用面向对象的设计来设置字符串解析，但结果总是非常冗长和难以阅读，并且在实践中并不广泛使用。

在现实世界中，大多数编程语言中的字符串解析都是由正则表达式处理的。这些表达式并不冗长，但是，哦，它们真的很难阅读，至少在你学会语法之前是这样。尽管正则表达式不是面向对象的，但 Python 正则表达式库提供了一些类和对象，可以用来构建和运行正则表达式。

正则表达式用于解决一个常见问题：给定一个字符串，确定该字符串是否与给定的模式匹配，并且可选地收集包含相关信息的子字符串。它们可以用来回答类似的问题：

+   这个字符串是一个有效的 URL 吗？

+   日志文件中所有警告消息的日期和时间是什么？

+   `/etc/passwd`中的哪些用户属于给定的组？

+   访客输入的 URL 请求了哪个用户名和文档？

有许多类似的情况，正则表达式是正确的答案。许多程序员犯了一个错误，实现了复杂而脆弱的字符串解析库，因为他们不知道或不愿意学习正则表达式。在本节中，我们将获得足够的正则表达式知识，以避免犯这样的错误！

## 匹配模式

正则表达式是一种复杂的迷你语言。它们依赖于特殊字符来匹配未知的字符串，但让我们从字面字符开始，比如字母、数字和空格字符，它们总是匹配它们自己。让我们看一个基本的例子：

```py
import re

search_string = "hello world"
pattern = "hello world"

match = re.match(pattern, search_string)

if match:
    print("regex matches")
```

Python 标准库模块用于正则表达式的称为`re`。我们导入它并设置一个搜索字符串和要搜索的模式；在这种情况下，它们是相同的字符串。由于搜索字符串与给定模式匹配，条件通过并且`print`语句执行。

请记住，`match`函数将模式与字符串的开头匹配。因此，如果模式是`"ello world"`，将找不到匹配。令人困惑的是，解析器一旦找到匹配就停止搜索，因此模式`"hello wo"`可以成功匹配。让我们构建一个小的示例程序来演示这些差异，并帮助我们学习其他正则表达式语法：

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

这只是一个通用版本的早期示例，它从命令行接受模式和搜索字符串。我们可以看到模式的开头必须匹配，但是一旦在以下命令行交互中找到匹配，就会返回一个值：

```py
$ python regex_generic.py "hello worl" "hello world"
'hello world' matches pattern 'hello worl'
$ python regex_generic.py "ello world" "hello world"
'hello world' does not match pattern 'ello world'

```

我们将在接下来的几个部分中使用这个脚本。虽然脚本总是通过命令行`python regex_generic.py "<pattern>" "<string>"`调用，但我们只会在以下示例中看到输出，以节省空间。

如果您需要控制项目是否发生在行的开头或结尾（或者字符串中没有换行符，发生在字符串的开头和结尾），可以使用`^`和`$`字符分别表示字符串的开头和结尾。如果要匹配整个字符串的模式，最好包括这两个：

```py
'hello world' matches pattern '^hello world$'
'hello worl' does not match pattern '^hello world$'

```

### 匹配一组字符

让我们从匹配任意字符开始。句号字符在正则表达式模式中使用时，可以匹配任何单个字符。在字符串中使用句号意味着您不在乎字符是什么，只是有一个字符在那里。例如：

```py
'hello world' matches pattern 'hel.o world'
'helpo world' matches pattern 'hel.o world'
'hel o world' matches pattern 'hel.o world'
'helo world' does not match pattern 'hel.o world'

```

请注意，最后一个示例不匹配，因为在模式中句号的位置上没有字符。

这样做很好，但是如果我们只想匹配几个特定的字符怎么办？我们可以将一组字符放在方括号中，以匹配其中任何一个字符。因此，如果我们在正则表达式模式中遇到字符串`[abc]`，我们知道这五个（包括两个方括号）字符只会匹配字符串中的一个字符，并且进一步地，这一个字符将是`a`、`b`或`c`中的一个。看几个例子：

```py
'hello world' matches pattern 'hel[lp]o world'
'helpo world' matches pattern 'hel[lp]o world'
'helPo world' does not match pattern 'hel[lp]o world'

```

这些方括号集应该被称为字符集，但更常见的是被称为**字符类**。通常，我们希望在这些集合中包含大量的字符，并且将它们全部打出来可能会很单调和容易出错。幸运的是，正则表达式设计者考虑到了这一点，并给了我们一个快捷方式。在字符集中，短横线字符将创建一个范围。如果您想匹配"所有小写字母"、"所有字母"或"所有数字"，可以使用如下方法：

```py
'hello   world' does not match pattern 'hello [a-z] world'
'hello b world' matches pattern 'hello [a-z] world'
'hello B world' matches pattern 'hello [a-zA-Z] world'
'hello 2 world' matches pattern 'hello [a-zA-Z0-9] world'

```

还有其他匹配或排除单个字符的方法，但如果您想找出它们是什么，您需要通过网络搜索找到更全面的教程！

### 转义字符

如果在模式中放置句号字符可以匹配任意字符，那么如何在字符串中匹配一个句号呢？一种方法是将句号放在方括号中以创建一个字符类，但更通用的方法是使用反斜杠进行转义。下面是一个正则表达式，用于匹配 0.00 到 0.99 之间的两位小数：

```py
'0.05' matches pattern '0\.[0-9][0-9]'
'005' does not match pattern '0\.[0-9][0-9]'
'0,05' does not match pattern '0\.[0-9][0-9]'

```

对于这个模式，两个字符`\.`匹配单个`.`字符。如果句号字符缺失或是另一个字符，它就不匹配。

这个反斜杠转义序列用于正则表达式中的各种特殊字符。您可以使用`\[`来插入一个方括号而不开始一个字符类，`\(`来插入一个括号，我们稍后会看到它也是一个特殊字符。

更有趣的是，我们还可以使用转义符号后跟一个字符来表示特殊字符，例如换行符（`\n`）和制表符（`\t`）。此外，一些字符类可以更简洁地用转义字符串表示；`\s`表示空白字符，`\w`表示字母、数字和下划线，`\d`表示数字：

```py
'(abc]' matches pattern '\(abc\]'
' 1a' matches pattern '\s\d\w'
'\t5n' does not match pattern '\s\d\w'
'5n' matches pattern '\s\d\w'

```

### 匹配多个字符

有了这些信息，我们可以匹配大多数已知长度的字符串，但大多数情况下，我们不知道模式内要匹配多少个字符。正则表达式也可以处理这个问题。我们可以通过附加几个难以记住的标点符号来修改模式以匹配多个字符。

星号（`*`）字符表示前面的模式可以匹配零次或多次。这可能听起来很愚蠢，但它是最有用的重复字符之一。在我们探索原因之前，考虑一些愚蠢的例子，以确保我们理解它的作用：

```py
'hello' matches pattern 'hel*o'
'heo' matches pattern 'hel*o'
'helllllo' matches pattern 'hel*o'

```

因此，模式中的`*`字符表示前面的模式（`l`字符）是可选的，如果存在，可以重复多次以匹配模式。其余的字符（`h`，`e`和`o`）必须出现一次。

匹配单个字母多次可能是非常罕见的，但如果我们将星号与匹配多个字符的模式结合起来，就会变得更有趣。例如，`.*`将匹配任何字符串，而`[a-z]*`将匹配任何小写单词的集合，包括空字符串。

例如：

```py
'A string.' matches pattern '[A-Z][a-z]* [a-z]*\.'
'No .' matches pattern '[A-Z][a-z]* [a-z]*\.'
'' matches pattern '[a-z]*.*'

```

模式中的加号（`+`）与星号类似；它表示前面的模式可以重复一次或多次，但与星号不同的是，它不是可选的。问号（`?`）确保模式出现零次或一次，但不会更多。让我们通过玩数字来探索一些例子（记住`\d`与`[0-9]`匹配相同的字符类）：

```py
'0.4' matches pattern '\d+\.\d+'
'1.002' matches pattern '\d+\.\d+'
'1.' does not match pattern '\d+\.\d+'
'1%' matches pattern '\d?\d%'
'99%' matches pattern '\d?\d%'
'999%' does not match pattern '\d?\d%'

```

### 将模式分组在一起

到目前为止，我们已经看到了如何可以多次重复一个模式，但我们在可以重复的模式上受到了限制。如果我们想重复单个字符，那么我们已经覆盖了，但如果我们想要重复一系列字符呢？将任何一组模式括在括号中允许它们在应用重复操作时被视为单个模式。比较这些模式：

```py
'abccc' matches pattern 'abc{3}'
'abccc' does not match pattern '(abc){3}'
'abcabcabc' matches pattern '(abc){3}'

```

与复杂模式结合使用，这种分组功能极大地扩展了我们的模式匹配能力。这是一个匹配简单英语句子的正则表达式：

```py
'Eat.' matches pattern '[A-Z][a-z]*( [a-z]+)*\.$'
'Eat more good food.' matches pattern '[A-Z][a-z]*( [a-z]+)*\.$'
'A good meal.' matches pattern '[A-Z][a-z]*( [a-z]+)*\.$'

```

第一个单词以大写字母开头，后面跟着零个或多个小写字母。然后，我们进入一个匹配一个空格后跟一个或多个小写字母的单词的括号。整个括号部分重复零次或多次，模式以句号结束。句号后不能有任何其他字符，这由`$`匹配字符串结束来表示。

我们已经看到了许多最基本的模式，但正则表达式语言支持更多。我在使用正则表达式的头几年里，每次需要做一些事情时都会查找语法。值得将 Python 的`re`模块文档加入书签，并经常复习。几乎没有什么是正则表达式无法匹配的，当解析字符串时，它们应该是你首选的工具。

## 从正则表达式获取信息

现在让我们专注于 Python 方面。正则表达式语法与面向对象编程完全不同。然而，Python 的`re`模块提供了一个面向对象的接口来进入正则表达式引擎。

我们一直在检查`re.match`函数是否返回有效对象。如果模式不匹配，该函数将返回`None`。但是，如果匹配，它将返回一个有用的对象，我们可以内省有关模式的信息。

到目前为止，我们的正则表达式已经回答了诸如“这个字符串是否与此模式匹配？”的问题。匹配模式是有用的，但在许多情况下，一个更有趣的问题是，“如果这个字符串匹配这个模式，相关子字符串的值是多少？”如果您使用组来标识您想要稍后引用的模式的部分，您可以从匹配返回值中获取它们，如下一个示例所示：

```py
pattern = "^[a-zA-Z.]+@([a-z.]*\.[a-z]+)$"
search_string = "some.user@example.com"
match = re.match(pattern, search_string)

if match:
 **domain = match.groups()[0]
    print(domain)
```

描述有效电子邮件地址的规范非常复杂，准确匹配所有可能性的正则表达式非常长。因此，我们作弊并制作了一个简单的正则表达式，用于匹配一些常见的电子邮件地址；重点是我们想要访问域名（在`@`符号之后），以便我们可以连接到该地址。通过将模式的该部分包装在括号中，并在匹配返回的对象上调用`groups()`方法，可以轻松实现这一点。

`groups`方法返回模式内匹配的所有组的元组，您可以对其进行索引以访问特定值。组从左到右排序。但是，请记住，组可以是嵌套的，这意味着您可以在另一个组内部有一个或多个组。在这种情况下，组按其最左边的括号顺序返回，因此外部组将在其内部匹配组之前返回。

除了匹配函数之外，`re`模块还提供了另外两个有用的函数，`search`和`findall`。`search`函数找到匹配模式的第一个实例，放宽了模式从字符串的第一个字母开始的限制。请注意，您可以通过使用匹配并在模式的前面放置`^.*`字符来获得类似的效果，以匹配字符串的开头和您要查找的模式之间的任何字符。

`findall`函数的行为类似于 search，只是它找到匹配模式的所有非重叠实例，而不仅仅是第一个。基本上，它找到第一个匹配，然后将搜索重置为该匹配字符串的末尾，并找到下一个匹配。

与其返回预期的匹配对象列表，它返回一个匹配字符串的列表。或元组。有时是字符串，有时是元组。这根本不是一个很好的 API！与所有糟糕的 API 一样，您将不得不记住差异并不依赖直觉。返回值的类型取决于正则表达式内括号组的数量：

+   如果模式中没有组，则`re.findall`将返回一个字符串列表，其中每个值都是与模式匹配的源字符串的完整子字符串

+   如果模式中恰好有一个组，则`re.findall`将返回一个字符串列表，其中每个值都是该组的内容

+   如果模式中有多个组，则`re.findall`将返回一个元组列表，其中每个元组包含匹配组的值，按顺序排列

### 注意

当您在设计自己的 Python 库中的函数调用时，请尝试使函数始终返回一致的数据结构。通常设计函数可以接受任意输入并处理它们是很好的，但返回值不应该从单个值切换到列表，或者从值列表切换到元组列表，具体取决于输入。让`re.findall`成为一个教训！

以下交互式会话中的示例将有望澄清差异：

```py
>>> import re
>>> re.findall('a.', 'abacadefagah')
['ab', 'ac', 'ad', 'ag', 'ah']
>>> re.findall('a(.)', 'abacadefagah')
['b', 'c', 'd', 'g', 'h']
>>> re.findall('(a)(.)', 'abacadefagah')
[('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'g'), ('a', 'h')]
>>> re.findall('((a)(.))', 'abacadefagah')
[('ab', 'a', 'b'), ('ac', 'a', 'c'), ('ad', 'a', 'd'), ('ag', 'a', 'g'), ('ah', 'a', 'h')]

```

### 使重复的正则表达式高效

每当调用正则表达式方法之一时，引擎都必须将模式字符串转换为内部结构，以便快速搜索字符串。这种转换需要相当长的时间。如果一个正则表达式模式将被多次重复使用（例如，在`for`或`while`循环内），最好只进行一次这种转换。

这是使用`re.compile`方法实现的。它返回一个已经编译过的正则表达式的面向对象版本，并且具有我们已经探索过的方法（`match`、`search`、`findall`）等。我们将在案例研究中看到这方面的例子。

这绝对是一个简短的正则表达式介绍。到目前为止，我们对基础知识有了很好的了解，并且会意识到何时需要进行进一步的研究。如果我们遇到字符串模式匹配问题，正则表达式几乎肯定能够解决。但是，我们可能需要在更全面地涵盖该主题的情况下查找新的语法。但现在我们知道该找什么了！让我们继续进行一个完全不同的主题：为存储序列化数据。

# 序列化对象

如今，我们认为能够将数据写入文件并在任意以后的日期检索出来是理所当然的。尽管这很方便（想象一下，如果我们不能存储任何东西，计算机的状态会是什么样子！），但我们经常发现自己需要将我们在内存中存储的数据转换为某种笨拙的文本或二进制格式，以便进行存储、在网络上传输或在远程服务器上进行远程调用。

Python 的`pickle`模块是一种以面向对象的方式直接存储对象的特殊存储格式。它基本上将一个对象（以及它作为属性持有的所有对象）转换为一系列字节，可以根据需要进行存储或传输。

对于基本工作，`pickle`模块有一个非常简单的接口。它由四个基本函数组成，用于存储和加载数据；两个用于操作类似文件的对象，两个用于操作`bytes`对象（后者只是文件类似接口的快捷方式，因此我们不必自己创建`BytesIO`文件类似对象）。

`dump`方法接受一个要写入的对象和一个类似文件的对象，用于将序列化的字节写入其中。这个对象必须有一个`write`方法（否则它就不会像文件一样），并且该方法必须知道如何处理`bytes`参数（因此，对于文本输出打开的文件将无法工作）。

`load`方法恰恰相反；它从类似文件的对象中读取序列化的对象。这个对象必须具有适当的类似文件的`read`和`readline`参数，每个参数当然都必须返回`bytes`。`pickle`模块将从这些字节中加载对象，并且`load`方法将返回完全重建的对象。以下是一个存储然后加载列表对象中的一些数据的示例：

```py
import pickle

some_data = ["a list", "containing", 5,
        "values including another list",
        ["inner", "list"]]

with open("pickled_list", 'wb') as file:
 **pickle.dump(some_data, file)

with open("pickled_list", 'rb') as file:
 **loaded_data = pickle.load(file)

print(loaded_data)
assert loaded_data == some_data
```

这段代码按照预期工作：对象被存储在文件中，然后从同一个文件中加载。在每种情况下，我们使用`with`语句打开文件，以便它会自动关闭。文件首先被打开以进行写入，然后第二次以进行读取，具体取决于我们是存储还是加载数据。

最后的`assert`语句会在新加载的对象不等于原始对象时引发错误。相等并不意味着它们是相同的对象。事实上，如果我们打印两个对象的`id()`，我们会发现它们是不同的。但是，因为它们都是内容相等的列表，所以这两个列表也被认为是相等的。

`dumps`和`loads`函数的行为与它们的类似文件的对应函数类似，只是它们返回或接受`bytes`而不是类似文件的对象。`dumps`函数只需要一个参数，即要存储的对象，并返回一个序列化的`bytes`对象。`loads`函数需要一个`bytes`对象，并返回还原的对象。方法名称中的`'s'`字符代表字符串；这是 Python 古老版本的一个遗留名称，那时使用的是`str`对象而不是`bytes`。

两个`dump`方法都接受一个可选的`protocol`参数。如果我们正在保存和加载只会在 Python 3 程序中使用的拾取对象，我们不需要提供此参数。不幸的是，如果我们正在存储可能会被旧版本的 Python 加载的对象，我们必须使用一个更旧且效率低下的协议。这通常不是问题。通常，加载拾取对象的唯一程序将是存储它的程序。拾取是一种不安全的格式，因此我们不希望将其不安全地发送到未知的解释器。

提供的参数是一个整数版本号。默认版本是 3，代表 Python 3 拾取使用的当前高效存储系统。数字 2 是旧版本，将存储一个可以在所有解释器上加载回 Python 2.3 的对象。由于 2.6 是仍然广泛使用的 Python 中最古老的版本，因此通常版本 2 的拾取就足够了。版本 0 和 1 在旧解释器上受支持；0 是 ASCII 格式，而 1 是二进制格式。还有一个优化的版本 4，可能有一天会成为默认版本。

因此，作为一个经验法则，如果您知道您要拾取的对象只会被 Python 3 程序加载（例如，只有您的程序会加载它们），请使用默认的拾取协议。如果它们可能会被未知的解释器加载，传递一个值为 2 的协议值，除非您真的相信它们可能需要被古老版本的 Python 加载。

如果我们向`dump`或`dumps`传递一个协议，我们应该使用关键字参数来指定它：`pickle.dumps(my_object, protocol=2)`。这并不是严格必要的，因为该方法只接受两个参数，但是写出完整的关键字参数会提醒我们代码的读者数字的目的。在方法调用中有一个随机整数会很难阅读。两个是什么？存储对象的两个副本，也许？记住，代码应该始终可读。在 Python 中，较少的代码通常比较长的代码更易读，但并非总是如此。要明确。

可以在单个打开的文件上多次调用`dump`或`load`。每次调用`dump`都会存储一个对象（以及它所组成或包含的任何对象），而调用`load`将加载并返回一个对象。因此，对于单个文件，存储对象时的每个单独的`dump`调用应该在以后的某个日期还原时有一个关联的`load`调用。

## 自定义拾取

对于大多数常见的 Python 对象，拾取“只是起作用”。基本的原始类型，如整数、浮点数和字符串可以被拾取，任何容器对象，如列表或字典，只要这些容器的内容也是可拾取的。此外，任何对象都可以被拾取，只要它的所有属性也是可拾取的。

那么，什么使属性无法被拾取？通常，这与时间敏感的属性有关，这些属性在将来加载时是没有意义的。例如，如果我们在对象的属性上存储了一个打开的网络套接字、打开的文件、运行中的线程或数据库连接，那么将这些对象拾取是没有意义的；当我们尝试以后重新加载它们时，很多操作系统状态将会消失。我们不能假装一个线程或套接字连接存在并使其出现！不，我们需要以某种方式自定义如何存储和还原这样的瞬态数据。

这是一个每小时加载网页内容以确保其保持最新的类。它使用`threading.Timer`类来安排下一次更新：

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

`url`、`contents`和`last_updated`都是可 pickle 的，但如果我们尝试 pickle 这个类的一个实例，事情在`self.timer`实例上会有点混乱：

```py
>>> u = UpdatedURL("http://news.yahoo.com/")
>>> import pickle
>>> serialized = pickle.dumps(u)
Traceback (most recent call last):
 **File "<pyshell#3>", line 1, in <module>
 **serialized = pickle.dumps(u)
_pickle.PicklingError: Can't pickle <class '_thread.lock'>: attribute lookup lock on _thread failed

```

这不是一个非常有用的错误，但看起来我们正在尝试 pickle 我们不应该 pickle 的东西。那将是`Timer`实例；我们在 schedule 方法中存储了对`self.timer`的引用，而该属性无法被序列化。

当`pickle`尝试序列化一个对象时，它只是尝试存储对象的`__dict__`属性；`__dict__`是一个字典，将对象上的所有属性名称映射到它们的值。幸运的是，在检查`__dict__`之前，`pickle`会检查是否存在`__getstate__`方法。如果存在，它将存储该方法的返回值，而不是`__dict__`。

让我们为我们的`UpdatedURL`类添加一个`__getstate__`方法，它简单地返回`__dict__`的副本，而不包括计时器：

```py
    def __getstate__(self):
        new_state = self.__dict__.copy()
        if 'timer' in new_state:
            del new_state['timer']
        return new_state
```

如果我们现在 pickle 对象，它将不再失败。我们甚至可以使用`loads`成功地恢复该对象。然而，恢复的对象没有计时器属性，因此它将无法像设计时那样刷新内容。我们需要在对象被反 pickle 时以某种方式创建一个新的计时器（以替换丢失的计时器）。

正如我们所期望的那样，有一个互补的`__setstate__`方法，可以实现以自定义反 pickle。这个方法接受一个参数，即`__getstate__`返回的对象。如果我们实现了这两个方法，`__getstate__`不需要返回一个字典，因为`__setstate__`将知道如何处理`__getstate__`选择返回的任何对象。在我们的情况下，我们只想恢复`__dict__`，然后创建一个新的计时器：

```py
    def __setstate__(self, data):
        self.__dict__ = data
        self.schedule()
```

`pickle`模块非常灵活，并提供其他工具来进一步自定义 pickling 过程，如果您需要的话。然而，这些超出了本书的范围。我们已经涵盖的工具对于许多基本的 pickling 任务已经足够了。通常被 pickle 的对象是相对简单的数据对象；例如，我们不太可能 pickle 整个运行中的程序或复杂的设计模式。

## 序列化网络对象

从未知或不受信任的来源加载 pickled 对象并不是一个好主意。可以向 pickled 文件中注入任意代码，以恶意攻击计算机。pickles 的另一个缺点是它们只能被其他 Python 程序加载，并且不能轻松地与其他语言编写的服务共享。

多年来已经使用了许多用于此目的的格式。XML（可扩展标记语言）曾经非常流行，特别是在 Java 开发人员中。YAML（另一种标记语言）是另一种格式，偶尔也会看到它被引用。表格数据经常以 CSV（逗号分隔值）格式交换。其中许多已经逐渐被遗忘，而且您将随着时间的推移遇到更多。Python 对所有这些都有坚实的标准或第三方库。

在对不受信任的数据使用这样的库之前，请确保调查每个库的安全性问题。例如，XML 和 YAML 都有模糊的特性，如果恶意使用，可以允许在主机机器上执行任意命令。这些特性可能不会默认关闭。做好你的研究。

**JavaScript 对象表示法**（**JSON**）是一种用于交换基本数据的人类可读格式。JSON 是一种标准格式，可以被各种异构客户端系统解释。因此，JSON 非常适用于在完全解耦的系统之间传输数据。此外，JSON 没有任何对可执行代码的支持，只能序列化数据；因此，很难向其中注入恶意语句。

因为 JSON 可以被 JavaScript 引擎轻松解释，所以经常用于从 Web 服务器传输数据到支持 JavaScript 的 Web 浏览器。如果提供数据的 Web 应用程序是用 Python 编写的，它需要一种将内部数据转换为 JSON 格式的方法。

有一个模块可以做到这一点，它的名称可预测地叫做`json`。该模块提供了与`pickle`模块类似的接口，具有`dump`、`load`、`dumps`和`loads`函数。对这些函数的默认调用几乎与`pickle`中的调用相同，因此我们不再重复细节。有一些区别；显然，这些调用的输出是有效的 JSON 表示，而不是一个被 pickled 的对象。此外，`json`函数操作`str`对象，而不是`bytes`。因此，在转储到文件或从文件加载时，我们需要创建文本文件而不是二进制文件。

JSON 序列化器不像`pickle`模块那样健壮；它只能序列化诸如整数、浮点数和字符串之类的基本类型，以及诸如字典和列表之类的简单容器。每种类型都有直接映射到 JSON 表示，但 JSON 无法表示类、方法或函数。无法以这种格式传输完整的对象。因为我们将对象转储为 JSON 格式的接收者通常不是 Python 对象，所以它无法以与 Python 相同的方式理解类或方法。尽管其名称中有“对象”一词，但 JSON 是一种**数据**表示法；对象，你会记得，由数据和行为组成。

如果我们有要序列化仅包含数据的对象，我们总是可以序列化对象的`__dict__`属性。或者我们可以通过提供自定义代码来从某些类型的对象创建或解析 JSON 可序列化字典来半自动化这个任务。

在`json`模块中，存储和加载函数都接受可选参数来自定义行为。`dump`和`dumps`方法接受一个名为`cls`（缩写为类，这是一个保留关键字）的关键字参数。如果传递了这个参数，它应该是`JSONEncoder`类的子类，并且应该重写`default`方法。此方法接受任意对象并将其转换为`json`可以解析的字典。如果它不知道如何处理对象，我们应该调用`super()`方法，以便它可以以正常方式处理序列化基本类型。

`load`和`loads`方法也接受`cls`参数，该参数可以是`JSONDecoder`的子类。但是，通常只需使用`object_hook`关键字参数将函数传递给这些方法。此函数接受一个字典并返回一个对象；如果它不知道如何处理输入字典，它可以原样返回。

让我们来看一个例子。假设我们有以下简单的联系人类，我们想要序列化：

```py
class Contact:
    def __init__(self, first, last):
        self.first = first
        self.last = last

    @property
    def full_name(self):
        return("{} {}".format(self.first, self.last))
```

我们可以只序列化`__dict__`属性：

```py
>>> c = Contact("John", "Smith")
>>> json.dumps(c.__dict__)
'{"last": "Smith", "first": "John"}'

```

但是，以这种方式访问特殊（双下划线）属性有点粗糙。另外，如果接收代码（也许是网页上的一些 JavaScript）希望提供`full_name`属性呢？当然，我们可以手工构建字典，但让我们创建一个自定义编码器：

```py
import json
class ContactEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Contact):
            return {'is_contact': True,
                    'first': obj.first,
                    'last': obj.last,
                    'full': obj.full_name}
        return super().default(obj)
```

`default`方法基本上是检查我们试图序列化的对象是什么类型；如果是联系人，我们手动将其转换为字典；否则，我们让父类处理序列化（假设它是一个基本类型，`json`知道如何处理）。请注意，我们传递了一个额外的属性来标识这个对象是一个联系人，因为在加载时没有办法知道。这只是一个约定；对于更通用的序列化机制，可能更合理的是在字典中存储一个字符串类型，或者甚至包括包和模块在内的完整类名。请记住，字典的格式取决于接收端的代码；必须就数据的规范方式达成一致。

我们可以使用这个类来通过将类（而不是实例化对象）传递给`dump`或`dumps`函数来编码一个联系人：

```py
>>> c = Contact("John", "Smith")
>>> json.dumps(c, cls=ContactEncoder)
'{"is_contact": true, "last": "Smith", "full": "John Smith",
"first": "John"}'

```

对于解码，我们可以编写一个接受字典并检查`is_contact`变量存在性的函数，以决定是否将其转换为联系人：

```py
def decode_contact(dic):
        if dic.get('is_contact'):
            return Contact(dic['first'], dic['last'])
        else:
            return dic
```

我们可以使用`object_hook`关键字参数将这个函数传递给`load`或`loads`函数：

```py
>>> data = ('{"is_contact": true, "last": "smith",'
 **'"full": "john smith", "first": "john"}')

>>> c = json.loads(data, object_hook=decode_contact)
>>> c
<__main__.Contact object at 0xa02918c>
>>> c.full_name
'john smith'

```

# 案例研究

让我们在 Python 中构建一个基本的基于正则表达式的模板引擎。这个引擎将解析一个文本文件（比如一个 HTML 页面），并用从这些指令输入的文本替换某些指令。这是我们希望用正则表达式做的最复杂的任务；事实上，一个完整的版本可能会利用适当的语言解析机制。

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

这个文件包含形式为`/** <directive> <data> **/`的“标签”，其中数据是可选的单词，指令是：

+   `include`：在这里复制另一个文件的内容

+   `variable`：在这里插入变量的内容

+   `loopover`：重复循环的内容，对应一个列表变量

+   `endloop`：标志循环文本的结束

+   `loopvar`：插入循环变量中的单个值

这个模板将根据传递给它的变量呈现不同的页面。这些变量将从所谓的上下文文件中传递进来。这将被编码为一个表示相关变量的键的`json`对象。我的上下文文件可能看起来像这样，但你可以自己推导出你自己的：

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

在我们进入实际的字符串处理之前，让我们为处理文件和从命令行获取数据编写一些面向对象的样板代码：

```py
import re
import sys
import json
from pathlib import Path

DIRECTIVE_RE = re.compile(
 **r'/\*\*\s*(include|variable|loopover|endloop|loopvar)'
 **r'\s*([^ *]*)\s*\*\*/')

class TemplateEngine:
    def __init__(self, infilename, outfilename, contextfilename):
        self.template = open(infilename).read()
        self.working_dir = Path(infilename).absolute().parent
 **self.pos = 0
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

这都是相当基础的，我们创建一个类，并用从命令行传入的一些变量对其进行初始化。

注意我们如何通过跨两行来使正则表达式变得更可读？我们使用原始字符串（r 前缀），这样我们就不必对所有反斜杠进行双重转义。这在正则表达式中很常见，但仍然很混乱。（正则表达式总是如此，但通常是值得的。）

`pos`表示我们正在处理的内容中的当前字符；我们马上会看到更多。

现在“剩下的就是”实现那个 process 方法。有几种方法可以做到这一点。让我们以一种相当明确的方式来做。

process 方法必须找到与正则表达式匹配的每个指令，并对其进行适当的处理。但是，它还必须负责将每个指令之前、之后和之间的普通文本输出到输出文件中，不经修改。

正则表达式的编译版本的一个很好的特性是，我们可以通过传递`pos`关键字参数告诉`search`方法从特定位置开始搜索。如果我们临时定义对指令进行适当处理为“忽略指令并从输出文件中删除它”，我们的处理循环看起来非常简单：

```py
def process(self):
    match = DIRECTIVE_RE.search(self.template, pos=self.pos)
    while match:
        self.outfile.write(self.template[self.pos:match.start()])
 **self.pos = match.end()
        match = DIRECTIVE_RE.search(self.template, pos=self.pos)
    self.outfile.write(self.template[self.pos:])
```

这个函数在英语中找到文本中与正则表达式匹配的第一个字符串，输出从当前位置到该匹配的开始的所有内容，然后将位置前进到上述匹配的结束。一旦匹配完毕，它就会输出自上次位置以来的所有内容。

当然，在模板引擎中忽略指令是相当无用的，所以让我们设置用不同的方法委托到类上的不同方法的代码来替换那个位置前进的行：

```py
def process(self):
    match = DIRECTIVE_RE.search(self.template, pos=self.pos)
    while match:
        self.outfile.write(self.template[self.pos:match.start()])
 **directive, argument = match.groups()
 **method_name = 'process_{}'.format(directive)
 **getattr(self, method_name)(match, argument)
        match = DIRECTIVE_RE.search(self.template, pos=self.pos)
    self.outfile.write(self.template[self.pos:])
```

所以我们从正则表达式中获取指令和单个参数。指令变成一个方法名，我们动态地在`self`对象上查找该方法名（在模板编写者提供无效指令的情况下，这里可能需要一些错误处理更好）。我们将匹配对象和参数传递给该方法，并假设该方法将适当地处理一切，包括移动`pos`指针。

现在我们的面向对象的架构已经到了这一步，实际上实现委托的方法是非常简单的。`include`和`variable`指令是完全直接的。

```py
def process_include(self, match, argument):
    with (self.working_dir / argument).open() as includefile:
        self.outfile.write(includefile.read())
 **self.pos = match.end()

def process_variable(self, match, argument):
    self.outfile.write(self.context.get(argument, ''))
 **self.pos = match.end()

```

第一个方法简单地查找包含的文件并插入文件内容，而第二个方法在上下文字典中查找变量名称（这些变量是在`__init__`方法中从`json`中加载的），如果不存在则默认为空字符串。

处理循环的三种方法要复杂一些，因为它们必须在它们之间共享状态。为了简单起见（我相信你迫不及待地想看到这一漫长章节的结束，我们快到了！），我们将把这些方法作为类本身的实例变量来处理。作为练习，你可能会考虑更好的架构方式，特别是在阅读完接下来的三章之后。

```py
    def process_loopover(self, match, argument):
        self.loop_index = 0
 **self.loop_list = self.context.get(argument, [])
        self.pos = self.loop_pos = match.end()

    def process_loopvar(self, match, argument):
 **self.outfile.write(self.loop_list[self.loop_index])
        self.pos = match.end()

    def process_endloop(self, match, argument):
 **self.loop_index += 1
        if self.loop_index >= len(self.loop_list):
            self.pos = match.end()
            del self.loop_index
            del self.loop_list
            del self.loop_pos
        else:
 **self.pos = self.loop_pos

```

当我们遇到`loopover`指令时，我们不必输出任何内容，但我们必须在三个变量上设置初始状态。假定`loop_list`变量是从上下文字典中提取的列表。`loop_index`变量指示在循环的这一次迭代中应该输出列表中的哪个位置，而`loop_pos`被存储，这样当我们到达循环的结尾时就知道要跳回到哪里。

`loopvar`指令输出`loop_list`变量中当前位置的值，并跳到指令的结尾。请注意，它不会增加循环索引，因为`loopvar`指令可以在循环内多次调用。

`endloop`指令更复杂。它确定`loop_list`中是否还有更多的元素；如果有，它就跳回到循环的开始，增加索引。否则，它重置了用于处理循环的所有变量，并跳到指令的结尾，这样引擎就可以继续处理下一个匹配。

请注意，这种特定的循环机制非常脆弱；如果模板设计者尝试嵌套循环或忘记调用`endloop`，那对他们来说会很糟糕。我们需要进行更多的错误检查，可能还要存储更多的循环状态，以使其成为一个生产平台。但我承诺这一章快要结束了，所以让我们在查看我们的示例模板如何与其上下文一起呈现后，直接转到练习：

```py
<html>
    <body>

<h1>This is the title of the front page</h1>
<a href="link1.html">First Link</a>
<a href="link2.html">Second Link</a>

<p>My name is Dusty.
This is the content of my front page. It goes below the menu.</p>
<table>
<tr><th>Favourite Books</th></tr>

<tr><td>Thief Of Time</td></tr>

<tr><td>The Thief</td></tr>

<tr><td>Snow Crash</td></tr>

<tr><td>Lathe Of Heaven</td></tr>

</table>
    </body>
</html>

Copyright &copy; Today
```

由于我们规划模板的方式，会产生一些奇怪的换行效果，但它的工作效果如预期。

# 练习

在本章中，我们涵盖了各种主题，从字符串到正则表达式，再到对象序列化，然后再回来。现在是时候考虑这些想法如何应用到你自己的代码中了。

Python 字符串非常灵活，而 Python 是一个非常强大的基于字符串的操作工具。如果您在日常工作中没有进行大量的字符串处理，请尝试设计一个专门用于操作字符串的工具。尝试想出一些创新的东西，但如果遇到困难，可以考虑编写一个网络日志分析器（每小时有多少请求？有多少人访问了五个以上的页面？）或一个模板工具，用其他文件的内容替换某些变量名。

花费大量时间玩弄字符串格式化运算符，直到您记住了语法。编写一堆模板字符串和对象传递给格式化函数，并查看您得到了什么样的输出。尝试一些奇特的格式化运算符，比如百分比或十六进制表示法。尝试填充和对齐运算符，并查看它们在整数、字符串和浮点数上的不同行为。考虑编写一个自己的类，其中有一个`__format__`方法；我们没有详细讨论这一点，但探索一下您可以自定义格式化的程度。

确保您理解`bytes`和`str`对象之间的区别。在旧版本的 Python 中，这个区别非常复杂（没有`bytes`，`str`同时充当`bytes`和`str`，除非我们需要非 ASCII 字符，此时有一个单独的`unicode`对象，类似于 Python 3 的`str`类。这甚至比听起来的更令人困惑！）。现在更清晰了；`bytes`用于二进制数据，`str`用于字符数据。唯一棘手的部分是知道如何以及何时在两者之间转换。练习时，尝试将文本数据写入以`bytes`方式打开的文件（您将不得不自己对文本进行编码），然后从同一文件中读取。

尝试使用`bytearray`进行一些实验；看看它如何同时像一个字节对象和一个列表或容器对象。尝试向一个缓冲区写入数据，直到达到一定长度之前将其返回。您可以通过使用`time.sleep`调用来模拟将数据放入缓冲区的代码，以确保数据不会到达得太快。

在网上学习正则表达式。再多学习一些。特别是要了解有名分组、贪婪匹配与懒惰匹配以及正则表达式标志，这些是我们在本章中没有涵盖的三个特性。要有意识地决定何时不使用它们。许多人对正则表达式有非常强烈的意见，要么过度使用它们，要么根本不使用它们。试着说服自己只在适当的时候使用它们，并找出何时是适当的时候。

如果您曾经编写过一个适配器，用于从文件或数据库中加载少量数据并将其转换为对象，请考虑改用 pickle。Pickles 不适合存储大量数据，但对于加载配置或其他简单对象可能会有用。尝试多种编码方式：使用 pickle、文本文件或小型数据库。哪种方式对您来说最容易使用？

尝试对数据进行 pickling 实验，然后修改保存数据的类，并将 pickle 加载到新类中。什么有效？什么无效？有没有办法对一个类进行重大更改，比如重命名属性或将其拆分为两个新属性，但仍然可以从旧的 pickle 中获取数据？（提示：尝试在每个对象上放置一个私有的 pickle 版本号，并在更改类时更新它；然后可以在`__setstate__`中放置一个迁移路径。）

如果您从事任何网络开发工作，请尝试使用 JSON 序列化器进行一些实验。就个人而言，我更喜欢只序列化标准的 JSON 可序列化对象，而不是编写自定义编码器或`object_hooks`，但期望的效果实际上取决于前端（通常是 JavaScript）和后端代码之间的交互。

在模板引擎中创建一些新的指令，这些指令需要多个或任意数量的参数。您可能需要修改正则表达式或添加新的正则表达式。查看 Django 项目的在线文档，看看是否有任何其他模板标签您想要使用。尝试模仿它们的过滤器语法，而不是使用变量标签。当您学习了迭代和协程时，重新阅读本章，看看是否能找到一种更

# 总结

在本章中，我们涵盖了字符串操作、正则表达式和对象序列化。硬编码的字符串和程序变量可以使用强大的字符串格式化系统组合成可输出的字符串。区分二进制和文本数据很重要，`bytes`和`str`有特定的用途必须要理解。它们都是不可变的，但在操作字节时可以使用`bytearray`类型。

正则表达式是一个复杂的主题，但我们只是触及了表面。有许多种方法可以序列化 Python 数据；pickle 和 JSON 是最流行的两种方法之一。

在下一章中，我们将看一种设计模式，这种模式对于 Python 编程非常基础，以至于它已经被赋予了特殊的语法支持：迭代器模式。
