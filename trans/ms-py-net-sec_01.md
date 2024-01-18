# 使用 Python 脚本

在本章中，我们将介绍 Python 脚本、集合、函数、异常处理和面向对象编程。我们将回顾如何创建类、对象以及 Python 初始化对象的特点，包括使用特殊属性和方法。还将介绍一种方法、工具和开发环境。

本章将涵盖以下主题：

+   编程和安装 Python

+   数据结构和 Python 集合

+   Python 函数和异常处理

+   Python 中的面向对象编程

+   包括如何管理模块、包、依赖项、传递参数、使用虚拟环境以及 Python 脚本的`STB`模块的 OMSTD 方法论

+   Python 脚本开发的主要开发环境

+   与 Python IDE 交互和调试

# 技术要求

在开始阅读本书之前，您应该了解 Python 编程的基础知识，如基本语法、变量类型、数据类型元组、列表字典、函数、字符串和方法。在[python.org/downloads/](http://python.org/downloads/)上提供了两个版本，3.6.5 和 2.7.14。

本章的示例和源代码可在 GitHub 存储库的`chapter 1`文件夹中找到：[`github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security`](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)。

# 编程和安装 Python

Python 是一种易于阅读和编写的字节编译的面向对象编程语言。这种语言非常适合安全专业人员，因为它允许快速创建测试以及可重用的项目以供将来使用。由于许多安全工具都是用 Python 编写的，它为对已经编写的工具进行扩展和添加功能提供了许多机会。

# 介绍 Python 脚本

在本书中，我们将使用两个版本。如果您使用 Debian 或 Kali 等 Linux 发行版，那么不会有问题，因为 Python 是多平台的，并且在大多数 Linux 发行版中默认安装了 2.7 版本。

# 为什么选择 Python？

有很多选择 Python 作为主要编程语言的原因：

+   多平台和开源语言。

+   简单、快速、强大的语言。

+   许多关于计算机安全的库、模块和项目都是用 Python 编写的。

+   有很多文档和一个非常庞大的用户社区。

+   这是一种设计用于用几行代码创建强大程序的语言，而在其他语言中，只有在包含每种语言的许多特性之后才有可能实现。

+   适用于原型和快速概念测试（PoC）。

# 多平台

Python 解释器可在许多平台上使用（Linux、DOS、Windows 和 macOS X）。我们在 Python 中创建的代码在第一次执行时会被翻译成字节码。因此，在我们要执行 Python 中开发的程序或脚本的系统中，我们需要安装解释器。

# 面向对象编程

面向对象编程是一种范式，程序是通过“对象类”来定义的，它们通过发送消息来相互通信。它是程序化、结构化和模块化编程范式的演变，并在 Java、Python 或 C++等语言中实现。

类定义了对象中指定的行为和可用状态，并允许更直接地表示建模问题所需的概念，允许用户定义新类型。

对象的特点是：

+   区分它们之间的身份

+   通过方法定义它们的行为

+   通过属性和属性定义它们的状态

类允许在新类型的数据和与对象相关的功能之间进行分组，有利于在实现的细节和其使用的基本属性之间进行分离。这样，目标是不显示更多的相关信息，隐藏类的状态和内部方法，这被称为“封装”，它是继承自模块化编程的原则。

在使用类的一个重要方面是它们不是直接操作的，而是用来定义新类型。类为对象（类的实例）定义属性和行为。类充当一组对象的模板，这些对象被认为属于该类。

面向对象编程中使用的最重要的技术包括：

+   **抽象**：对象可以执行任务，与其他对象交互，或者修改和报告它们的状态，而无需沟通这些操作是如何执行的。

+   **封装**：对象通过清晰的接口阻止其他对象修改其内部状态或调用内部方法，并且只通过这个接口与其他对象相关联。

+   **多态性**：不同的行为可以与相同的名称相关联。

+   **继承**：对象通过建立层次结构与其他对象相关联，有可能一些对象继承其他对象的属性和方法，扩展它们的行为和/或专业化。对象以这种方式分组在形成层次结构的类中。

# 获取和安装 Python

在 Linux 和 Windows 平台上，Python 的安装速度很快。Windows 用户可以使用一个简单的安装程序，使配置工作变得容易。在 Linux 上，您可以选择从源代码构建安装，但这并不是强制的，您可以使用经典的包管理依赖，如 apt-get。

许多 Linux 发行版预装了 Python 2。在这样的系统上安装 Python 3 时，重要的是要记住我们并没有替换 Python 2 的安装。这样，当我们安装 Python 3 时，它可以与同一台机器上的 Python 2 并行安装。安装 Python 3 后，可以使用 Python3 可执行文件调用 python 解释器。

# 在 Windows 上安装 Python

Windows 用户可以从主 Python 网站获取安装程序：[`www.python.org/ftp/python/2.7.15/python-2.7.15.msi`](https://www.python.org/ftp/python/2.7.15/python-2.7.15.msi)。只需双击安装程序，然后按照安装步骤进行安装。它应该在`C:/Python27/`创建一个目录；这个目录将有`Python.exe`解释器以及所有默认安装的库。

Python 安装允许您自定义环境的安装位置。Python 2.7.14 的默认位置是`C:\Python27`，尽管您可以指定其他位置。当寻找特定模块和工具时，这个路径将是相关的。

如果要包括文档或安装一系列实用程序（如`pip`软件包管理器或 IDLE 开发环境，用于编辑和执行脚本），则可以自定义安装。建议您保留已标记的选项，以便安装它们，使我们拥有尽可能完整的环境：

![](img/5d69cf39-2c64-4588-8b80-95fb8f541609.png)

重要的是要检查“将 python.exe 添加到路径”框。这将允许您从任何路径直接从命令提示符运行 Python，而无需转到安装目录。

在安装 Python 版本的 Windows 时，您还可以看到 IDLE 可用，这是 Python 的编辑器或 IDE（集成开发环境），它将允许我们编写和测试代码。安装完成后，我们可以验证一切是否正确：

1.  打开安装的文件夹

1.  输入`C:\Python27\Lib\idlelib`

1.  双击运行`**idle.bat**`文件

Windows 用户的另一个选择是 WinPython，可以在**[`winpython.github.io`](http://winpython.github.io)**上找到。

WinPython 是一个 Python 发行版；您可以在 Windows 7/8/10 操作系统上安装它进行科学和教育用途。

这个发行版与其他发行版不同，因为它：

+   **无需安装**：WinPython 完全存在于自己的目录中，无需任何操作系统安装

+   **便携式**：您可以轻松地压缩您的 Python 项目并在其他机器上进行安装

# 在 Linux 上安装 Python

Python 默认安装在大多数 Gnu/Linux 发行版中。如果我们想要在 Ubuntu 或基于 Debian 的发行版中安装它，我们可以通过`apt-get`软件包管理器来实现：

```py
sudo apt-get install python2.7
```

# Python 集合

在本节中，我们将回顾不同类型的数据集合，如列表、元组和字典。我们将看到用于管理这些数据结构的方法和操作，以及一个实际示例，我们将在其中回顾主要用例。

# 列表

Python 中的列表相当于 C 等编程语言中的动态向量结构。我们可以通过在一对方括号之间封装它们的元素并用逗号分隔来表示文字。列表的第一个元素的索引为 0。索引运算符允许访问元素，并通过在方括号中添加其索引来在列表中表达语法上：

考虑以下示例：程序员可以通过使用`append()`方法添加项目来构建列表，打印项目，然后在再次打印之前对它们进行排序。在以下示例中，我们定义了一个协议列表，并使用 Python 列表的主要方法，如 append、index 和 remove：

```py
>>> protocolList = []
>>> protocolList.append("ftp")
>>> protocolList.append("ssh")
>>> protocolList.append("smtp")
>>> protocolList.append("http")
>>> print protocolList
```

```py
['ftp','ssh','smtp','http']
```

```py
>>> protocolList.sort()
>>> print protocolList
```

```py
['ftp','http','smtp','ssh']
```

```py
>>> type(protocolList)
<type 'list'>
>>> len(protocolList)
```

```py
4
```

要访问特定位置，我们使用`index`方法，要删除一个元素，我们使用 remove 方法：

```py
>>> position = protocolList.index("ssh")
>>> print "ssh position"+str(position)
```

```py
ssh position 3
```

```py
>>> protocolList.remove("ssh")
>>> print protocolList
```

```py
['ftp','http','smtp']
```

```py
>>> count = len(protocolList)
>>> print "Protocol elements "+str(count)
```

```py
Protocol elements 3
```

要打印整个协议列表，请使用以下代码。这将循环遍历所有元素并将它们打印出来：

```py
>>> for protocol in protocolList:
>>      print (protocol)
```

```py
ftp
http
smtp
```

列表还有一些方法，可以帮助我们操纵其中的值，并允许我们在其中存储多个变量，并为 Python 中的对象数组提供更好的排序方法。这些是最常用的用于操纵列表的方法：

+   **.append(value):** 在列表末尾添加一个元素

+   **.count('x'):** 获取列表中'x'的数量

+   **.index('x'):** 返回列表中'x'的索引

+   **.insert('y','x'):** 在位置'y'插入'x'

+   **.pop():** 返回最后一个元素并从列表中删除它

+   **.remove('x'):** 从列表中删除第一个'x'

+   **.reverse():** 反转列表中的元素

+   **.sort():** 按字母顺序升序或按数字顺序升序对列表进行排序

# 反转列表

我们在列表中拥有的另一个有趣的操作是通过`reverse()`方法返回列表的可能性：

```py
>>> protocolList.reverse()
>>> print protocolList
```

```py
['smtp','http','ftp']
```

执行相同操作的另一种方法是使用`-1`索引。这种快速简便的技术显示了如何以相反的顺序访问列表的所有元素：

```py
>>> protocolList[::-1]
>>> print protocolList
```

```py
['smtp','http','ftp']
```

# 理解列表

理解列表允许您创建一个可迭代对象的新列表。基本上，它们包含必须为迭代每个元素的循环内的表达式。

基本语法是：

```py
new_list = [expression for_loop_one_or_more conditions]
```

列表理解也可以用于迭代字符串：

```py
>>> protocolList = ["FTP", "HTTP", "SNMP", "SSH"]
>>> protocolList_lower= [protocol.lower() for protocol in protocolList]
>>> print(protocolList_lower) # Output: ['ftp', 'http', 'snmp', 'ssh']
```

# 元组

元组类似于列表，但其大小和元素是不可变的，也就是说，其值不能被更改，也不能添加比最初定义的更多的元素。元组由括号括起来。如果我们尝试修改元组的元素，我们会收到一个错误，指示元组对象不支持元素的赋值：

![](img/a3849051-44d7-4322-833b-2bceec82c9a3.png)

# 字典

Python 字典数据结构允许我们将值与键关联起来。键是任何不可变对象。与键关联的值可以通过索引运算符访问。在 Python 中，使用哈希表实现字典。

Python 字典是一种存储键值对的方法。Python 字典用大括号`{}`括起来。字典，也称为关联矩阵，得名于将键和值相关联的集合。例如，让我们看一个具有名称和数字的协议字典：

```py
>>> services = {"ftp":21, "ssh":22, "smtp":25, "http":80}
```

字典的限制在于我们不能使用相同的键创建多个值。这将覆盖重复键的先前值。字典的操作是唯一的。我们可以使用`update`方法将两个不同的字典合并为一个。此外，`update`方法将在元素冲突时合并现有元素：

```py
>>> services = {"ftp":21, "ssh":22, "smtp":25, "http":80}
>>> services2 = {"ftp":21, "ssh":22, "snmp":161, "ldap":389}
>>> services.update(services2)
>>> print services
```

这将返回以下字典：

```py
{"ftp":21, "ssh":22, "smtp":25, "http":80,"snmp":161, "ldap":389}
```

第一个值是键，第二个是与键关联的值。作为键，我们可以使用任何不可变的值：我们可以使用数字、字符串、布尔值或元组，但不能使用列表或字典，因为它们是可变的。

字典与列表或元组的主要区别在于，存储在字典中的值不是通过它们的索引访问的，因为它们没有顺序，而是通过它们的键，再次使用`[]`运算符。

与列表和元组一样，您也可以使用此运算符重新分配值：

```py
>>> services["http"]= 8080
```

构建字典时，每个键都用冒号与其值分隔，我们用逗号分隔项。`.keys()`方法将返回字典的所有键的列表，`.items()`方法将返回字典中所有元素的完整列表。

以下是使用这些方法的示例：

+   `services.keys()`是一个方法，将返回字典中的所有键。

+   `services.items()`是一个方法，将返回字典中所有项目的完整列表。

![](img/81e31826-beac-4f5c-aaca-e6bdafed7960.png)

从性能的角度来看，字典中的键在存储时被转换为哈希值，以节省空间并在搜索或索引字典时提高性能。还可以打印字典并按特定顺序浏览键。以下代码提取字典元素，然后对其进行排序：

```py
>>> items = services.items()
>>> print items
```

```py
[('ftp', 21), ('smtp',25), ('ssh', 22), ('http', 80), ('snmp', 161)]
```

```py
>>> items.sort()
>>> print items
```

```py
[('ftp', 21), ('http', 80), ('smtp', 25), ('snmp', 161), ('ssh', 22)]
```

我们可以提取字典中每个元素的键和值：

```py
>>> keys = services.keys()
>>> print keys
```

```py
['ftp', 'smtp', 'ssh', 'http', 'snmp']
```

```py
>>> keys.sort()
>>> print keys
```

```py
['ftp', 'http', 'smtp', 'snmp', 'ssh']
```

```py
>>> values = services.values()
>>> print values
```

```py
[21, 25, 22, 80, 161]
```

```py
>>> values.sort()
>>> print values
```

```py
[21, 22, 25, 80, 161]
```

```py
>>> services.has_key('http')
```

```py
True
```

```py
>>> services['http']
```

```py
80
```

最后，您可能希望遍历字典并提取和显示所有的“键:值”对：

```py
>>> for key,value in services.items():
        print key,value
ftp 21
smtp 25
ssh 22
http 80
snmp 161
```

# Python 函数和异常管理

在本节中，我们将回顾 Python 函数和异常管理。我们将看到一些声明和在脚本代码中使用它们的示例。我们还将回顾我们可以在 Python 中找到的主要异常，以便在我们的脚本中包含。

# Python 函数

在 Python 中，函数提供了有组织的可重用代码块。通常，这允许程序员编写一块代码来执行单个相关操作。虽然 Python 提供了许多内置函数，程序员可以创建用户定义的函数。除了通过将程序分成部分来帮助我们编程和调试外，函数还允许我们重用代码。

Python 函数是使用 def 关键字定义的，后面跟着函数名和函数参数。函数的主体由要执行的 Python 语句组成。在函数的末尾，您可以选择向函数调用者返回一个值，或者默认情况下，如果您没有指定返回值，它将返回 None 对象。

例如，我们可以定义一个函数，给定一个数字序列和一个通过参数传递的项目，如果元素在序列中，则返回 True，否则返回 False：

```py
>>> def contains(sequence,item):
        for element in sequence:
                if element == item:
                        return True
        return False
>>> print contains([100,200,300,400],200)
```

```py
True
```

```py
>>> print contains([100,200,300,400],300)
```

```py
True
```

```py
>>> print contains([100,200,300,400],350)
```

```py
False
```

# 异常管理

异常是 Python 在程序执行期间检测到的错误。当解释器遇到异常情况时，例如尝试将数字除以 0 或尝试访问不存在的文件时，它会生成或抛出异常，通知用户存在问题。

如果未捕获异常，执行流程将被中断，并在控制台中显示与异常相关的信息，以便程序员解决问题。

让我们看一个小程序，当尝试将 1 除以 0 时会引发异常。如果我们执行它，将会得到以下错误消息：

![](img/13968d79-9053-41f1-bfb8-a73c3e2f14fc.png)

首先显示的是回溯，它由导致异常的调用列表组成。正如我们在堆栈跟踪中看到的那样，错误是由第 7 行的 calculate()调用引起的，该调用又在第 5 行调用 division(1, 0)，最终在 division 的第 2 行执行 a/b 语句。

Python 语言提供了异常处理能力来做到这一点。我们使用 try/except 语句来提供异常处理。现在，程序尝试执行除以零的操作。当错误发生时，我们的异常处理捕获错误并在屏幕上打印消息：

![](img/994fd36f-cd81-417b-b530-651d019d1475.png)

在下面的示例中，我们尝试创建一个文件类型的 f 对象。如果未将文件作为参数传递，则会抛出 IOError 类型的异常，我们通过 try-except 捕获到这个异常：

![](img/867ef05a-d3b5-41e1-9ce7-d4d4dbaa17b6.png)

默认情况下提供的一些异常列在此处（它们派生自的类在括号中）：

+   **BaseException**：所有异常继承的类。

+   异常（BaseException）：所有不输出的异常的超类。

+   **ZeroDivisionError**（ArithmeticError）：当除法或模块运算的第二个参数为`0`时引发。

+   **EnvironmentError**（StandardError）：与输入/输出相关的错误的父类。

+   **IOError**（EnvironmentError）：输入/输出操作中的错误。

+   **OSError**（EnvironmentError）：系统调用中的错误。

+   **ImportError**（StandardError）：未找到要导入的模块或模块元素。

# Python 作为面向对象的语言

在本节中，我们将回顾 Python 中的面向对象编程和继承。

面向对象编程是当今最常用的范例之一。虽然它适用于我们在日常生活中可以找到的许多情况，在 Python 中，我们可以将其与其他范例结合起来，以充分利用语言并在保持最佳代码设计的同时提高我们的生产力。

Python 是一种面向对象的语言，允许您定义类并从这些定义实例化对象。由 class 语句开头的块是类定义。在块中定义的函数是其方法，也称为成员函数。

Python 创建对象的方式是使用 class 关键字。Python 对象是方法、变量和属性的集合。您可以使用相同的类定义创建许多对象。以下是协议对象定义的简单示例：

您可以在`protocol.py`文件中找到以下代码。

```py
class protocol(object):

 def __init__(self, name, number,description):
 self.name = name
 self.number = number
 self.description = description

 def getProtocolInfo(self):
 return self.name+ " "+str(self.number)+ " "+self.description
```

`__init__`方法是一个特殊的方法，正如其名称所示，它充当构造方法来执行任何必要的初始化过程。

该方法的第一个参数是一个特殊的关键字，我们使用 self 标识符来引用当前对象。它是对对象本身的引用，并提供了一种访问其属性和方法的方式。

self 参数相当于在 C++或 Java 等语言中找到的指针。在 Python 中，self 是语言的保留字，是强制性的，它是常规方法的第一个参数，并且通过它可以访问类的属性和方法。

要创建对象，请在类名后面写上任何必要的参数，这些参数将传递给`__init__`方法，这是在实例化类时调用的方法：

```py
>>> protocol_http= protocol("HTTP", 80, "Hypertext transfer protocol")
```

现在我们已经创建了我们的对象，我们可以通过 object.attribute 和`object.method()`语法访问其属性和方法：

```py
>>> protocol_http.name
>>> protocol_http.number
>>> protocol_http.description
>>> protocol_http.getProtocolInfo()
```

# 继承

面向对象编程语言的主要概念是：封装、继承和多态。在面向对象语言中，对象通过建立层次关系与其他对象相关联，有可能一些对象继承其他对象的属性和方法，扩展它们的行为和/或特化。

继承允许我们从另一个类生成一个新类，继承其属性和方法，根据需要进行调整或扩展。要指示一个类从另一个类继承，我们需要将被继承的类的名称放在括号中。

在面向对象编程术语中，有人说“B 继承自 A”，“B 是从 A 派生出来的类”，“A 是 B 的基类”，或者“A 是 B 的超类”。

这有助于代码的重用，因为你可以在基类中实现基本行为和数据，并在派生类中对其进行特化：

![](img/6b5697ad-3499-43a3-8f4a-278cea025cb2.png)

# OMSTD 方法和 Python 脚本的 STB 模块

OMSTD 代表安全工具开发的开放方法论，它是 Python 开发安全工具的方法和一套良好实践。本指南适用于 Python 开发，尽管实际上你可以将相同的想法扩展到其他语言。在这一点上，我将讨论方法和一些技巧，我们可以遵循使代码更易读和可重用。

# Python 包和模块

Python 编程语言是一种高级通用语言，具有清晰的语法和完整的标准库。通常被称为脚本语言，安全专家们已经将 Python 作为开发信息安全工具包的语言。模块化设计、易读的代码和完全开发的库集为安全研究人员和专家构建工具提供了一个起点。

Python 自带了一个全面的标准库，提供了从提供简单 I/O 访问的集成模块到特定平台 API 调用的一切。Python 的美妙之处在于用户贡献的模块、包和个体框架。项目越大，不同方面之间的顺序和分离就越重要。在 Python 中，我们可以使用模块的概念来实现这种分离。

# Python 中的模块是什么？

模块是一个我们可以从程序中使用的函数、类和变量的集合。标准 Python 发行版中有大量的模块可用。

导入语句后面跟着模块的名称，使我们能够访问其中定义的对象。导入的对象通过模块的标识符、点运算符和所需对象的标识符，可以从导入它的程序或模块中访问。

模块可以被定义为包含 Python 定义和声明的文件。文件的名称是附加了`.py`后缀的模块的名称。我们可以从定义一个简单的模块开始，该模块将存在于与我们将要编写的`main.py`脚本相同的目录中：

+   `main.py`

+   `my_module.py`

在`my_module.py`文件中，我们将定义一个简单的`test()`函数，它将打印“This is my first module”：

```py
 # my_module.py
 def test():
    print("This is my first module")
```

在我们的`main.py`文件中，我们可以将这个文件作为一个模块导入，并使用我们新定义的 test()方法，就像这样：

```py
# main.py
 import my_module

 def main():
    my_module.test()

 if __name__ == '__main__':
    main()
```

这就是我们需要在 Python 程序中定义一个非常简单的`python`模块的全部内容。

# Python 模块和 Python 包之间的区别

当我们使用 Python 时，了解 Python 模块和`Python`包之间的区别很重要。重要的是要区分它们；包是包含一个或多个模块的模块。

软件开发的一部分是基于编程语言中的模块添加功能。随着新的方法和创新的出现，开发人员提供这些功能构建块作为模块或包。在 Python 网络中，其中大多数模块和包都是免费的，其中包括完整的源代码，允许您增强提供的模块的行为并独立验证代码。

# 在 Python 中传递参数

为了完成这个任务，最好使用默认安装 Python 时自带的`argparse`模块。

有关更多信息，您可以查看官方网站：[`docs.python.org/3/library/argparse.html`](https://docs.python.org/3/library/argparse.html)。

以下是如何在我们的脚本中使用它的示例：

您可以在文件名`testing_parameters.py`中找到以下代码

```py
import argparse

parser = argparse.ArgumentParser(description='Testing parameters')
parser.add_argument("-p1", dest="param1", help="parameter1")
parser.add_argument("-p2", dest="param2", help="parameter2")
params = parser.parse_args()
print params.param1
print params.param2
```

在 params 变量中，我们有用户从命令行输入的参数。要访问它们，您必须使用以下内容：

```py
params.<Name_dest>
```

其中一个有趣的选项是可以使用 type 属性指示参数的类型。例如，如果我们希望某个参数被视为整数，我们可以这样做：

```py
parser.add_argument("-param", dest="param", type="int")
```

另一件有助于使我们的代码更易读的事情是声明一个充当参数全局对象的类：

```py
class Parameters:
 """Global parameters"""
    def __init__(self, **kwargs):
        self.param1 = kwargs.get("param1")
        self.param2 = kwargs.get("param2")
```

例如，如果我们想要同时向函数传递多个参数，我们可以使用这个全局对象，其中包含全局执行参数。例如，如果我们有两个参数，我们可以这样构建对象：

您可以在文件名`params_global.py`中找到以下代码

```py
import argparse

class Parameters:
 """Global parameters"""

    def __init__(self, **kwargs):
        self.param1 = kwargs.get("param1")
        self.param2 = kwargs.get("param2")

def view_parameters(input_parameters):
    print input_parameters.param1
    print input_parameters.param2

parser = argparse.ArgumentParser(description='Passing parameters in an object')
parser.add_argument("-p1", dest="param1", help="parameter1")
parser.add_argument("-p2", dest="param2", help="parameter2")
params = parser.parse_args()
input_parameters = Parameters(param1=params.param1,param2=params.param2)
view_parameters(input_parameters)
```

在上一个脚本中，我们可以看到我们使用`argparse`模块获取参数，并将这些参数封装在 Parameters 类的对象中。通过这种做法，我们可以在对象中封装参数，以便从脚本的不同点轻松检索这些参数。

# 在 Python 项目中管理依赖项

如果我们的项目依赖于其他库，理想情况是有一个文件，其中包含这些依赖项，以便我们的模块的安装和分发尽可能简单。为此任务，我们可以创建一个名为`requirements.txt`的文件，如果我们使用 pip 实用程序调用它，将降低所讨论模块需要的所有依赖项。

使用 pip 安装所有依赖项：

```py
pip -r requirements.txt
```

在这里，`pip`是`Python`包和依赖项管理器，而`requirements.txt`是详细列出项目所有依赖项的文件。

# 生成 requirements.txt 文件

我们还有可能从项目源代码创建`requirements.txt`文件。

为此任务，我们可以使用`pipreqs`模块，其代码可以从 GitHub 存储库下载：[`github.com/bndr/pipreqs`](https://github.com/bndr/pipreqs)

这样，该模块可以使用`pip install pipreqs`命令或通过 GitHub 代码存储库使用`python setup.py install`命令进行安装。

有关该模块的更多信息，您可以查询官方 pypi 页面：

[`pypi.python.org/pypi/pipreqs`](https://pypi.python.org/pypi/pipreqs)。

要生成`requirements.txt`文件，您必须执行以下命令：

```py
 pipreqs <path_project>
```

# 使用虚拟环境

在使用 Python 时，强烈建议您使用 Python 虚拟环境。虚拟环境有助于分离项目所需的依赖项，并保持我们的全局目录清洁，不受`project`包的影响。虚拟环境为安装 Python 模块提供了一个单独的环境，以及 Python 可执行文件和相关文件的隔离副本。您可以拥有尽可能多的虚拟环境，这意味着您可以配置多个模块配置，并且可以轻松地在它们之间切换。

从版本 3 开始，Python 包括一个`venv`模块，提供了这个功能。文档和示例可在[`docs.python.org/3/using/windows.html#virtual-environments`](https://docs.python.org/3/using/windows.html#virtual-environments)找到

还有一个独立的工具可用于早期版本，可以在以下位置找到：

[`virtualenv.pypa.io/en/latest`](https://virtualenv.pypa.io/en/latest)

# 使用 virtualenv 和 virtualwrapper

当您在本地计算机上安装`Python`模块而不使用虚拟环境时，您正在全局在操作系统中安装它。此安装通常需要用户根管理员，并且该`Python`模块为每个用户和每个项目安装。

在这一点上，最佳实践是如果您需要在多个 Python 项目上工作，或者您需要一种在许多项目中使用所有关联库的方法，那么最好安装 Python 虚拟环境。

Virtualenv 是一个允许您创建虚拟和隔离环境的`Python`模块。基本上，您创建一个包含项目所需的所有可执行文件和模块的文件夹。您可以使用以下命令安装 virtualenv：

```py
$ sudo pip install virtualenv
```

要创建一个新的虚拟环境，请创建一个文件夹，并从命令行进入该文件夹：

```py
$ cd your_new_folder $ virtualenv name-of-virtual-environment
```

例如，这将创建一个名为 myVirtualEnv 的新环境，您必须激活它才能使用它：

```py
$ cd myVirtualEnv/ $ virtualenv myVirtualEnv $ source bin/activate
```

执行此命令将在您当前的工作目录中启动一个名为指示的文件夹，其中包含 Python 的所有可执行文件和允许您在虚拟环境中安装不同包的`pip`模块。

Virtualenv 就像一个沙盒，当您工作时，项目的所有依赖项都将被安装，所有模块和依赖项都是分开保存的。如果用户在他们的计算机上安装了相同版本的 Python，那么相同的代码将在虚拟环境中运行，而不需要任何更改。

`Virtualenvwrapper`允许您更好地组织在您的计算机上所有虚拟管理的环境，并提供更优化的方式来使用`virtualenv`。

我们可以使用 pip 命令安装`virtualwrapper`，因为它在官方 Python 存储库中可用。安装它的唯一要求是先前安装了`virtualenv`：

```py
$ pip install virtualenvwrapper
```

要在 Windows 中创建一个虚拟环境，您可以使用`virtualenv`命令：

```py
virtualenv venv
```

当我们执行前面的命令时，我们会看到这个结果：![](img/9c138124-d264-4e0d-8fec-dff36f65f947.png)

在 Windows 中执行`virtualenv`命令会生成四个文件夹：

![](img/bb6e3654-f1bc-4599-a2a5-d9df33dd48b3.png)

在 scripts 文件夹中，有一个名为`activate.bat`的脚本，用于激活虚拟环境。一旦激活，我们将拥有一个干净的模块和库环境，并且我们将不得不下载我们项目的依赖项，以便将它们复制到这个目录中，使用以下代码：

```py
cd venv\Scripts\activate (venv) > pip install -r requirements.txt
```

这是活动文件夹，当我们可以找到 active.bat 脚本时：![](img/fdaf0e17-8d1d-447b-9900-c131a7463a14.png)

# STB（Security Tools Builder）模块

这个工具将允许我们创建一个基础项目，我们可以在其上开始开发我们自己的工具。

该工具的官方存储库是[`github.com/abirtone/STB`](https://github.com/abirtone/STB)。

对于安装，我们可以通过下载源代码并执行`setup.py`文件来完成，这将下载`requirements.txt`文件中的依赖项。

我们也可以使用`**pip install stb**`命令来完成。

执行`**stb**`命令时，我们会得到以下屏幕，要求我们提供信息来创建我们的项目：

![](img/3701b9aa-a4ca-4a4c-98e2-1bd182c5e566.png)

使用此命令，我们将获得一个带有`setup.py`文件的应用程序骨架，如果我们想要将该工具安装为系统中的命令，则可以执行：

```py
python setup.py install
```

当我们执行前面的命令时，我们会得到下一个文件夹结构：

![](img/de04da7e-a815-4716-8c4c-7efe0105b0d9.png)

这也创建了一个包含允许我们执行它的文件的`port_scanning_lib`文件夹：

```py
python port_scanning.py –h
```

如果我们使用帮助选项（-h）执行脚本，我们会看到一系列可以使用的参数：

![](img/7b7a9cde-99c8-4a5d-ad17-130bb441799e.png)

我们可以看到在`port_scanning.py`文件中生成的代码：

```py
parser = argparse.ArgumentParser(description='%s security tool' % "port_scanning".capitalize(), epilog = examples, formatter_class = argparse.RawTextHelpFormatter)

# Main options
parser.add_argument("target", metavar="TARGET", nargs="*")
parser.add_argument("-v", "--verbosity", dest="verbose", action="count", help="verbosity level: -v, -vv, -vvv.", default=1)
parsed_args = parser.parse_args()

# Configure global log
log.setLevel(abs(5 - parsed_args.verbose) % 5)

# Set Global Config
config = GlobalParameters(parsed_args)
```

在这里，我们可以看到定义的参数，并且使用`GlobalParameters`对象传递`parsed_args`变量中的参数。要执行的方法在`**api.py**`文件中找到。

例如，在这一点上，我们可以从命令行中检索输入的参数：

```py
# ----------------------------------------------------------------------
#
# API call
#
# ----------------------------------------------------------------------
def run(config):
    """
    :param config: GlobalParameters option instance
    :type config: `GlobalParameters`

    :raises: TypeError
     """
     if not isinstance(config, GlobalParameters):
         raise TypeError("Expected GlobalParameters, got '%s' instead" % type(config))

# --------------------------------------------------------------------------
# INSERT YOUR CODE HERE # TODO
# --------------------------------------------------------------------------
print config
print config.target
```

我们可以从命令行执行脚本，将我们的 ip 目标作为参数传递：

```py
python port_scanning.py 127.0.0.1
```

如果我们现在执行，我们可以看到如何在输出中获得首次引入的参数：

![](img/46600ebf-0ca5-46dd-94bd-a8a7ef1f3af1.png)

# 脚本开发的主要开发环境

在本节中，我们将审查 Pycharm 和 WingIDE 作为 Python 脚本的开发环境。

# 设置开发环境

为了快速开发和调试 Python 应用程序，绝对必须使用稳固的 IDE。如果您想尝试不同的选项，我们建议您查看 Python 官方网站上的列表，那里可以根据操作系统和需求查看工具：[`wiki.python.org/moin/IntegratedDevelopmentEnvironments`](https://wiki.python.org/moin/IntegratedDevelopmentEnvironments)。

在所有环境中，我们将强调以下内容：

+   **Pycharm: **[`www.jetbrains.com/pycharm`](http://www.jetbrains.com/pycharm)

+   **Wing IDE**: [`wingware.com`](https://wingware.com)

# Pycharm

PyCharm 是由 Jetbrains 公司开发的 IDE，基于同一公司的 IDE IntelliJ IDEA，但专注于 Java，并且是 Android Studio 的基础。

PyCharm 是多平台的，我们可以在 Windows，Linux 和 macOS X 上找到二进制文件。 PyCharm 有两个版本：社区和专业，其特性与 Web 框架集成和数据库支持相关。

在此网址上，我们可以看到社区版和专业版之间的比较：[`www.jetbrains.com/pycharm`](http://www.jetbrains.com/pycharm)

这个开发环境的主要优势是：

+   自动完成，语法高亮，分析工具和重构。

+   与 Django，Flask，Pyramid，Web2Py，jQuery 和 AngularJS 等 Web 框架集成。

+   高级调试器。

+   与 SQLAlchemy（ORM），Google App Engine，Cython 兼容。

+   与版本控制系统的连接：Git，CVS，Mercurial。

# WingIDE

WingIDE 是一个多平台环境，可在 Windows，Mac 和 Linux 上使用，并提供了与调试和变量探索相关的所有功能。

WingIDE 具有丰富的功能集，可以轻松支持复杂 Python 应用程序的开发。使用 WingIDE，您可以检查变量，堆栈参数和内存位置，而不会在记录它们之前更改任何值。断点是调试过程中最常用的功能。 Wing Personal 是这个 Python IDE 的免费版本，可以在[`wingware.com/downloads/wingide-personal`](https://wingware.com/downloads/wingide-personal)找到

WingIDE 使用您系统中安装的 Python 配置：

![](img/7f4752ae-8d20-42c4-9fb5-5b518dfa22b7.png)

# 使用 WingIDE 进行调试

在这个例子中，我们正在调试一个接受两个输入参数的 Python 脚本：

![](img/609a7394-7c59-4fe9-b2ca-ec68e47e0d8b.png)

一个有趣的话题是在我们的程序中添加断点的可能性，使用`Add Breakpoint`选项，这样，我们可以调试并查看变量的内容，就在我们设置断点的地方：

![](img/d8fa3334-2f5d-4c58-b545-51588db32702.png)

我们可以在调用`view_parameters`方法时设置断点。

要以调试模式执行带参数的脚本，您必须编辑脚本的属性，并在调试标记中添加脚本需要的参数：

![](img/43824730-490c-44bd-8424-18ca965c8fe3.png)

如果我们在函数内部执行调试模式并设置断点，我们可以看到本地**字符串变量**中参数的内容：

![](img/c044a72c-4ac9-4fc8-af31-9f3f516ae7f3.png)

在下面的截图中，我们可以可视化 params 变量的值，该变量包含我们正在调试的值：

![](img/53fc3235-2be1-4166-aae1-ef20f6f7ea24.png)

# 摘要

在本章中，我们学习了如何在 Windows 和 Linux 操作系统上安装 Python。我们回顾了主要的数据结构和集合，如列表、元组和字典。我们还回顾了函数、异常处理的管理，以及如何创建类和对象，以及属性和特殊方法的使用。然后我们看了开发环境和一种介绍 Python 编程的方法论。OMSTD 是 Python 开发安全工具的一种方法论和最佳实践。最后，我们回顾了主要的开发环境，PyCharm 和 WingIDE，用于 Python 脚本开发。

在下一个章节中，我们将探讨用于处理操作系统和文件系统、线程和并发的编程系统包。

# 问题

1.  Python 2.x 和 3.x 之间有什么区别？

1.  Python 开发人员使用的编程范式是什么，这个范式背后的主要概念是什么？

1.  Python 中的哪种数据结构允许我们将值与键关联起来？

1.  Python 脚本的主要开发环境是什么？

1.  作为 Python 开发安全工具的一套良好实践方法，我们可以遵循什么方法论？

1.  有助于创建隔离的 Python 环境的`Python`模块是什么？

1.  哪个工具允许我们创建一个基础项目，我们可以在其上开始开发我们自己的工具？

1.  我们如何在 Python 开发环境中调试变量？

1.  我们如何在`pycharm`中添加断点？

1.  我们如何在 Wing IDE 中添加断点？

# 进一步阅读

在这些链接中，您将找到有关提到的工具和官方 Python 文档的更多信息，以便查找其中一些被评论模块的信息：

+   [`winpython.github.io`](http://winpython.github.io)

+   [`docs.python.org/2.7/library/`](https://docs.python.org/2.7/library/)

+   [`docs.python.org/3.6/library/`](https://docs.python.org/3.6/library/)

+   [`virtualenv.pypa.io/en/latest`](https://virtualenv.pypa.io/en/latest)

+   [`wiki.python.org/moin/IntegratedDevelopmentEnvironments`](https://wiki.python.org/moin/IntegratedDevelopmentEnvironments)
