# Python脚本概述

Python是一种脚本语言，由Guido van Rossum于1991年创建，用于各种应用，如游戏开发，GIS编程，软件开发，Web开发，数据分析，机器学习和系统脚本。

Python是一种面向对象的高级编程语言，具有动态语义。主要是Python是一种解释性语言。Python用于快速应用程序开发，因为它具有所有的高级开发功能。

Python简单易学，因为其语法使程序更易读。因此，程序的维护成本较低。

Python还有一个重要的特性，即导入模块和包。这个特性允许代码重用。Python解释器易于理解。我们可以在其中逐行编写完整的代码，并且由于Python是一种解释性语言，代码会逐行执行。Python还有广泛的库，用于高级功能。

本章将涵盖以下主题：

+   Python脚本

+   安装和使用Python以及各种工具

+   变量，数字和字符串

+   Python支持的数据结构以及如何在脚本中使用所有这些概念

+   决策制定；也就是`if`语句

+   循环语句；也就是`for`和`while`循环

+   函数

+   模块

# 技术要求

在阅读本书之前，您应该了解Python编程的基础知识，比如基本语法，变量类型，元组数据类型，列表字典，函数，字符串和方法。在[python.org/downloads/](https://www.python.org/downloads/)上有两个版本，3.7.2和2.7.15。在本书中，我们将使用版本3.7进行代码示例和包安装。

本章的示例和源代码可在GitHub存储库中找到：[https://github.com/PacktPublishing/Mastering-Python-Scripting-for-System-Administrators-](https://github.com/PacktPublishing/Mastering-Python-Scripting-for-System-Administrators-)。

# 为什么选择Python？

Python有广泛的库，用于开源数据分析工具，Web框架，测试等。Python是一种可以在不同平台（Windows，Mac，Linux和嵌入式Linux硬件平台，如树莓派）上使用的编程语言。它用于开发桌面应用程序和Web应用程序。

如果使用Python，开发人员可以用更少的行数编写程序。原型设计非常快速，因为Python运行在解释器系统上。Python可以以面向对象，过程式或函数式的方式处理。

Python可以执行各种任务，比如创建Web应用程序。它与软件一起用于创建工作流程；连接到数据库系统，处理文件，处理大数据，并执行复杂的数学运算。

# Python语法与其他编程语言的比较

Python中编写的代码非常易读，因为它类似于英语。Python使用新行来完成命令。

Python有一个很棒的特性：缩进。使用缩进，我们可以定义决策语句，循环（如`for`和`while`循环），函数和类的范围。

# Python安装

在这一部分，我们将学习Python在不同平台上的安装，比如Linux和Windows。

# 在Linux平台上安装

大多数Linux发行版在默认安装中都有Python 2。其中一些还包括Python 3。

要在基于Debian的Linux上安装`python3`，请在终端中运行以下命令：

```py
sudo apt install python3
```

要在`centos`上安装`python3`，请在终端中运行以下命令：

```py
sudo yum install python3
```

如果无法使用上述命令安装Python，请从[https://www.python.org/downloads/](https://www.python.org/downloads/)下载Python并按照说明进行操作。

# 在Windows平台上安装

要在Microsoft Windows中安装Python，您需要从`python.org`下载可执行文件并安装。从[https://www.python.org/downloads/](https://www.python.org/downloads/)下载`python.exe`，选择要在您的PC上安装的Python版本。然后，双击下载的`exe`并安装Python。在安装向导中，有一个复选框，上面写着**将Python添加到路径**。选中此复选框，然后按照说明安装`python3`。

# 使用pip安装软件包的安装和使用

在Linux中，安装`pip`如下：

```py
sudo apt install python-pip --- This will install pip for python 2.
sudo apt install python3-pip --- This will install pip for python 3.
```

在Windows中，安装`pip`如下：

```py
python -m pip install pip
```

# 在Mac上安装

要安装`python3`，首先必须在系统上安装`brew`。要在系统上安装`brew`，运行以下命令：

```py
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

通过运行上述命令，`brew`将被安装。现在我们将使用`brew`安装`python3`：

```py
brew install python3
```

# 安装Jupyter笔记本

要安装Jupyter笔记本，请下载Anaconda。

安装已下载的Anaconda并按照向导上的说明操作。

使用`pip`安装Jupyter：

```py
pip install jupyter
```

在Linux中，`pip install jupyter`将为`python 2`安装Jupyter。如果要为`python 3`安装`jupyter`，请运行以下命令：

```py
pip3 install jupyter
```

# 安装和使用虚拟环境

现在我们将看看如何安装虚拟环境以及如何激活它。

要在Linux上安装虚拟环境，请执行以下步骤：

1.  首先检查`pip`是否已安装。我们将为`python3`安装`pip`：

```py
sudo apt install python3-pip
```

1.  使用`pip3`安装虚拟环境：

```py
sudo pip3 install virtualenv
```

1.  现在我们将创建虚拟环境。您可以给它任何名称；我称其为`pythonenv`：

```py
virtualenv pythonenv
```

1.  激活您的虚拟环境：

```py
source venv/bin/activate
```

1.  工作完成后，您可以使用以下命令停用`virtualenv`：

```py
deactivate
```

在Windows中，运行`pip install virtualenv`命令安装虚拟环境。安装`virtualenv`的步骤与Linux相同。

# 安装Geany和PyCharm

从[https://www.geany.org/download/releases](https://www.geany.org/download/releases)下载Geany，并下载所需的二进制文件。在安装时按照说明操作。

从[https://www.jetbrains.com/pycharm/download/#section=windows](https://www.jetbrains.com/pycharm/download/#section=windows)下载PyCharm并按照说明操作。

# Python解释器

Python是一种解释性语言。它有一个名为Python解释器或Python shell的交互式控制台。这个控制台提供了一种逐行执行程序而不创建脚本的方法。

您可以在Python交互式控制台中访问所有的内置函数和库、安装的模块和命令历史。这个控制台让您有机会探索Python。当您准备好时，可以将代码粘贴到脚本中。

# Python和Bash脚本之间的区别

在本节中，我们将学习Python和Bash脚本之间的区别。区别如下：

+   Python是一种脚本语言，而Bash是用于输入和执行命令的Shell

+   使用Python更容易处理更大的程序

+   在Python中，您可以通过调用导入模块的一行函数来完成大多数事情

# 启动交互式控制台

我们可以从已安装Python的任何计算机上访问Python的交互式控制台。运行以下命令启动Python的交互式控制台：

```py
$ python
```

这将启动默认的Python交互式控制台。

在Linux中，如果我们在终端中输入`Python`，则会启动`python2.7`控制台。如果要启动`python3`控制台，则在终端中输入`python3`并按*Enter*。

在Windows中，当您在命令提示符中输入`Python`时，它将启动已下载Python版本的控制台。

# 使用Python交互式控制台编写脚本

Python交互式控制台从`>>>前缀`开始。这个控制台将接受您在`>>>前缀`后面写的Python命令。参考以下截图：

![](assets/d9a71f44-f130-456b-bbaa-3dbbe4b47bed.png)

现在，我们将看如何给变量赋值，就像下面的例子：

```py
>>> name = John
```

在这里，我们给`name`变量赋了一个字符值`John`。我们按下*Enter*键，得到了一个带有`>>>前缀`的新行：

```py
>>> name = John
```

现在，我们将看一个给变量赋值的例子，然后我们将执行一个数学运算来得到这些值：

```py
>>> num1 = 5000
>>> num2 = 3500
>>> num3 = num1 + num2
>>> print (num3)
8500
>>> num4 = num3 - 2575
>>> print (num4)
5925
>>>
```

在这里，我们给变量赋值，添加了两个变量，将结果存储在第三个变量中，并将结果打印到终端上。接下来，我们从结果变量中减去一个变量，输出将存储在第四个变量中。然后，我们将结果打印到终端上。这告诉我们，我们也可以将Python解释器用作计算器：

```py
>>> 509 / 22
23.136363636363637
>>>
```

在这里，我们进行了除法运算。我们将`509`除以`22`，得到的结果是`23.136363636363637`。

# 多行

当我们在Python解释器中编写多行代码（例如`if`语句和`for`和`while`循环函数），解释器会使用三个点(`...`)作为第二个提示符进行行继续。要退出这些行，你必须按两次*Enter*键。现在我们来看下面的例子：

```py
>>> val1 = 2500
>>> val2 = 2400
>>> if val1 > val2:
... print("val1 is greater than val2")
... else:
... print("val2 is greater than val1")
...
val1 is greater than val2
>>>
```

在这个例子中，我们给两个变量`val1`和`val2`赋了整数值，并且检查`val1`是否大于`val2`。在这种情况下，`val1`大于`val2`，所以`if`块中的语句被打印出来。记住，`if`和`else`块中的语句是缩进的。如果你不使用缩进，你会得到以下错误：

```py
>>> if val1 > val2:
... print("val1 is greater than val2")
File "<stdin>", line 2
print("val1 is greater than val2")
^
IndentationError: expected an indented block
>>>
```

# 通过Python解释器导入模块

如果你导入任何模块，那么Python解释器会检查该模块是否可用。你可以使用`import`语句来做到这一点。如果该模块可用，那么在按下*Enter*键后你会看到`>>>`前缀。这表示执行成功。如果该模块不存在，Python解释器会显示一个错误：

```py
>>> import time
>>>
```

导入`time`模块后，我们得到`>>>`前缀。这意味着模块存在，并且这个命令被成功执行了：

```py
>>> import matplotlib
```

如果模块不存在，你将得到`Traceback`错误：

```py
File "<stdin>", line 1, in <module>
ImportError: No module named 'matplotlib'
```

所以在这里，`matplotlib`不可用，所以会出现错误：`ImportError: No module named 'matplotlib'`。

要解决这个错误，我们需要安装`matplotlib`，然后再尝试导入`matplotlib`。安装`matplotlib`后，你应该能够导入模块，如下所示：

```py
>>> import matplotlib
>>>
```

# 退出Python控制台

我们可以通过两种方式退出Python控制台：

+   键盘快捷键：*Ctrl + D*

+   使用`quit()`或`exit()`函数

# 键盘快捷键

键盘快捷键，*Ctrl + D*，将给出以下代码：

```py
>>> val1 = 5000
>>> val2 = 2500
>>>
>>> val3 = val1 - val2
>>> print (val3)
2500
>>>
student@ubuntu:~$
```

# 使用`quit()`或`exit()`函数

`quit()`会让你退出Python的交互式控制台。它还会把你带回到之前所在的原始终端：

```py
>>> Lion = 'Simba'
>>> quit()
student@ubuntu$
```

# 缩进和制表符

在Python中编写代码块时，缩进是必须的。当你编写函数、决策语句、循环语句和类时，缩进是有用的。这样可以方便阅读你的Python程序。

我们在Python程序中使用缩进来表示代码块。要缩进一个代码块，你可以使用空格或制表符。参考以下例子：

```py
if val1 > val2:
 print ("val1 is greater than val2")
print("This part is not indented")
```

在上面的例子中，我们缩进了`print`语句，因为它属于`if`块。下一个打印语句不属于`if`块，所以我们没有对它进行缩进。

# 变量

和其他编程语言一样，不需要先声明变量。在Python中，只需想一个名字给你的变量并给它赋一个值。你可以在程序中使用该变量。所以在Python中，你可以在需要时声明变量。

在Python中，变量的值和类型在程序执行过程中可能会发生变化。在下面的代码行中，我们将值`100`赋给一个变量：

```py
n = 100
Here are assigning 100 to the variable n. Now, we are going to increase the value of n by 1:
>>> n = n + 1
>>> print(n)
101
>>>
```

以下是一个在执行过程中可以改变的变量类型的例子：

```py
a = 50 # data type is implicitly set to integer
a = 50 + 9.50 # data type is changed to float
a = "Seventy" # and now it will be a string
```

Python会处理不同数据类型的表示；也就是说，每种类型的值都存储在不同的内存位置。变量将是一个我们将要为其分配值的名称：

```py
>>> msg = 'And now for something completely different'
>>> a = 20
>>> pi = 3.1415926535897932
```

这个例子做了三个赋值。第一个赋值是将字符串赋给名为`msg`的变量。第二个赋值是将整数赋给名为`a`的变量，最后一个赋值是`pi`值的赋值。

变量的类型是它所引用的值的类型。看看下面的代码：

```py
>>> type(msg)
<type 'str'>
>>> type(a)
<type 'int'>
>>> type(pi)
<type 'float'>
```

# 创建并为变量赋值

在Python中，变量不需要显式声明以保留内存空间。因此，只要将值赋给变量，声明就会自动完成。在Python中，等号`=`用于为变量赋值。

考虑以下例子：

```py
#!/usr/bin/python3
name = 'John'
age = 25
address = 'USA'
percentage = 85.5
print(name)
print(age)
print(address)
print(percentage)

Output:
John
25
USA
85.5
```

在上面的例子中，我们将`John`赋给`name`变量，将`25`赋给`age`变量，将`USA`赋给`address`变量，将`85.5`赋给`percentage`变量。

我们不必像其他语言那样首先声明它们。因此，查看值时，解释器将获取该变量的类型。在上面的例子中，`name`和`address`是`字符串`，age是整数，percentage是浮点类型。

可以如下进行相同值的多重赋值：

```py
x = y = z = 1
```

在上面的例子中，我们创建了三个变量，并将整数值`1`分配给它们，所有这三个变量都将分配到相同的内存位置。

在Python中，我们可以在一行中为多个变量分配多个值：

```py
x, y, z = 10, 'John', 80
```

在这里，我们声明了一个字符串变量`y`，并将值`John`赋给它，还声明了两个整数变量`x`和`z`，并分别将值`10`和`80`赋给它们。

# 数字

Python解释器也可以充当计算器。您只需输入一个表达式，它就会返回值。括号`( )`用于进行分组，如下例所示：

```py
>>> 5 + 5
10
>>> 100 - 5*5
75
>>> (100 - 5*5) / 15
5.0
>>> 8 / 5
1.6
```

整数是`int`类型的，小数部分是`float`类型的。

在Python中，除法(`/`)操作始终返回一个浮点值。`floor`除法(`//`)得到一个整数结果。`%`运算符用于计算余数。

考虑以下例子：

```py
>>> 14/3
4.666666666666667
>>>
>>> 14//3
4
>>>
>>> 14%3
2
>>> 4*3+2
14
>>>
```

要计算幂，Python有`**`运算符，如下例所示：

```py
>>> 8**3
512
>>> 5**7
78125
>>>
```

等号(`=`)用于为变量赋值：

```py
>>> m = 50
>>> n = 8 * 8
>>> m * n
3200
```

如果一个变量没有任何值，但我们仍然尝试使用它，那么解释器将显示错误：

```py
>>> k
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
NameError: name 'k' is not defined
>>>
```

如果运算符具有混合类型的操作数，则得到的值将是浮点数：

```py
>>> 5 * 4.75 - 1
22.75
```

在Python交互式控制台中，`_`包含了最后一个打印的表达式值，如下例所示：

```py
>>> a = 18.5/100
>>> b = 150.50
>>> a * b
27.8425
>>> b + _
178.3425
>>> round(_, 2)
178.34
>>>
```

数字数据类型存储不可变的数值。如果我们这样做，Python将为更改后的数据类型分配一个新对象。

我们可以通过为它们分配一个值来创建数字对象，如下例所示：

```py
num1 = 50
num2 = 25
```

`del`语句用于删除单个或多个变量。参考以下例子：

```py
del num
del num_a, num_b
```

# 数字类型转换

在某些情况下，您需要显式地将一个类型的数字转换为另一个类型以满足某些要求。Python在表达式中内部执行此操作

+   输入`int(a)`将`a`转换为整数

+   输入`float(a)`将`a`转换为浮点数

+   输入`complex(a)`将`a`转换为具有实部`x`和虚部`零`的复数

+   使用`complex(a, b)`将`a`和`b`转换为具有实部`a`和虚部`b`的复数。`a`和`b`是数值表达式

# 字符串

与数字一样，字符串也是Python中的数据结构之一。Python可以操作字符串。字符串可以表示如下：

+   用单引号(`'...'`)括起来

+   用双引号(`"..."`)括起来

看下面的例子：

```py
>>> 'Hello Python'
'Hello Python'
>>> "Hello Python"
'Hello Python'
```

字符串是一组字符。我们可以按顺序访问字符，如下所示：

```py
>>> city = 'delhi'
>>> letter = city[1]
>>> letter = city[-3]
```

在第二个语句中，我们从`city`中选择字符编号`1`并将其分配给`letter`。方括号中的数字是索引。索引表示您要访问的字符。它从`0`开始。因此，在上面的例子中，当您执行`letter = city[1]`时，您将得到以下输出：

```py
city d e l h i
index 0 1 2 3 4
-5 -4 -3 -2 -1

Output:
e
l
```

# 连接（+）和重复（*）

接下来是连接和重复。参考以下代码：

```py
>>> 3 * 'hi' + 'hello'
'hihihihello'
```

在上面的例子中，我们正在进行字符串连接和重复。`3 * 'hi'`意味着`hi`被打印`3`次，并且使用`+`号，我们将`hello`字符串连接到`hi`旁边。

我们可以通过将它们写在一起来自动连接两个字符串。这两个字符串必须用引号括起来，如下所示：

```py
>>> 'he' 'llo'
'hello'
```

当您有很长的字符串并且想要打破它们时，这个功能真的很有帮助。这里有一个例子：

```py
>>> str = ('Several strings'
... 'joining them together.')
>>> str
'Several strings joining them together.'
```

# 字符串切片

字符串支持切片，这意味着从字符串中按指定范围获取字符。让我们看看以下例子。请注意，起始索引值始终包括在内，结束值始终不包括在内。

考虑一个字符串，`str = "Programming":`

```py
>>> str[0:2]
'Pr'
>>> str[2:5]
'ogr'
```

现在，省略的第一个索引的默认值是零，就像例子中一样：

```py
>>> str[:2] + str[2:]
'Python'
>>> str[:4] + str[4:]
'Python' >>> str[:2]
'Py'
>>> str[4:]
'on'
>>> str[-2:]
'on'
```

# 在字符串中访问值

我们可以使用方括号切片从字符串中访问字符。我们还可以在指定范围内从字符串中访问字符。参考以下示例： 

```py
#!/usr/bin/python3
str1 = 'Hello Python!'
str2 = "Object Oriented Programming"
print ("str1[0]: ", str1[0])
print ("str2[1:5]: ", str2[1:5])

Output:
str1[0]: H
str2[1:5]: bjec
```

# 更新字符串

我们可以通过将新值重新分配给指定的索引来更新字符串。参考以下示例：

```py
#!/usr/bin/python3
str1 = 'Hello Python!'
print ("Updated String: - ", str1 [:6] + 'John')

Output:
Updated String: - Hello John
```

# 转义字符

Python支持转义字符，这些字符是不可打印的，可以用反斜杠表示。转义字符在单引号和双引号字符串中都会被解释：

| **符号** | **十六进制字符** | **描述** |
| `a` | `0x07` | 响铃或警报 |
| `b` | `0x08` | 退格 |
| `cx` |  | 控制-`x` |
| `n` | `0x0a` | 换行符 |
| `C-x` |  | 控制-`x` |
| `e` | `0x1b` | 转义 |
| `f` | `0x0c` | 换页符 |
| `s` | `0x20` | 空格 |
| `M-C-x` |  | 元控制-`x` |
| `x` |  | 字符`x` |
| `nnn` |  | 八进制表示法，其中`n`在范围0.7内 |
| `r` | `0x0d` | 回车 |
| `xnn` |  | 十六进制表示法，其中`n`在范围`0.9`，`a.f`或`A.F`内 |
| `t` | `0x09` | 制表符 |
| `v` | `0x0b` | 垂直制表符 |

# 特殊字符串运算符

以下表格显示了字符串的特殊运算符。考虑`a`是`Hello`，`b`是`World`：

| **运算符** | **描述** | **例子** |
| `+` | 连接：在运算符的两侧添加值 | `a + b`会得到`HelloWorld` |
| `[]` | 切片：从给定索引中获取字符 | `a[7]`会得到`r` |
| `[ : ]` | 范围切片：给出给定范围内的字符 | `a[1:4]`会得到`ell` |
| `*` | 重复：创建新字符串，连接多个相同字符串的副本 | `a*2`会得到`HelloHello` |
| `not in` | 成员资格：如果字符不存在于给定字符串中，则返回`true` | `Z`不在`will`中会得到`1` |
| `in` | 成员资格：如果字符存在于给定字符串中，则返回`true` | `H`在`a`中会得到`1` |
| `%` | 格式：执行字符串格式化 |  |

# % 字符串格式化运算符

`%`是Python中的字符串格式化运算符。参考以下示例：

```py
#!/usr/bin/python3
print ("Hello this is %s and my age is %d !" % ('John', 25))

Output:
Hello this is John and my age is 25 !
```

以下表格显示了与`%`一起使用的符号列表：

| **序号** | **格式符号和转换** |
| 1 | `%c` – 字符 |
| 2 | `%s` – 格式化之前通过`str()`进行字符串转换 |
| 3 | `%i` – 有符号十进制整数 |
| 4 | `%d` – 有符号十进制整数 |
| 5 | `%u` – 无符号十进制整数 |
| 6 | `%o` – 八进制整数 |
| 7 | `%x` – 十六进制整数（小写字母） |
| 8 | `%X` – 十六进制整数（大写字母） |
| 9 | `%e` – 指数表示法（小写`e`） |
| 10 | `%E` – 指数表示法（大写`E`） |
| 11 | `%f` – 浮点实数 |

# Python中的三重引号

Python的三重引号功能用于跨越多行，包括换行符和制表符。三重引号的语法由三个连续的单引号或双引号组成。参考以下代码：

```py
#!/usr/bin/python3

para_str = """ Python is a scripting language which was created by
Guido van Rossum in 1991, t which is used in various sectors such as Game Development, GIS Programming, Software Development, web development,
Data Analytics and Machine learning, System Scripting etc.
"""
print (para_str)
```

它产生以下输出。请注意制表符和换行符：

```py
Output:
Python is a scripting language which was created by
Guido van Rossum in 1991, which is used in various sectors such as
Game Development, GIS Programming, Software Development, web development,
Data Analytics and Machine learning, System Scripting etc.
```

# 字符串是不可变的

字符串是不可变的，意味着我们不能改变值。参考给定的示例：

```py
>>> welcome = 'Hello, John!'
>>> welcome[0] = 'Y'
TypeError: 'str' object does not support item assignment
```

由于字符串是不可变的；我们不能改变现有的字符串。但我们可以创建一个与原始字符串不同的新字符串：

```py
>>> str1 = 'Hello John'
>>> new_str = 'Welcome' + str1[5:]
>>> print(str1)
Hello John
>>> print(new_str)
Welcome John
>>>
```

# 理解列表

Python支持一种称为`list`的数据结构，它是一个可变的有序元素序列。列表中的每个元素称为项。列表是通过在方括号`[]`之间插入值来定义的。`list`的每个元素都被赋予一个数字，我们称之为位置或索引。索引从零开始；即第一个索引为零，第二个索引为1，依此类推。我们可以对列表执行以下操作：索引、切片、添加、乘法和检查成员资格。

Python的内置`length`函数返回列表的长度。Python还有用于查找`list`的最大和最小项的函数。列表可以是编号列表、字符串列表或混合列表。

以下是创建列表的代码：

```py
l = list()
numbers = [10, 20, 30, 40]
animals = ['Dog', 'Tiger', 'Lion']
list1 = ['John', 5.5, 500, [110, 450]]
```

在这里，我们创建了三个列表：第一个是`numbers`，第二个是`animals`，第三个是`list1`。列表中的另一个列表称为嵌套列表。我们的`list1`是一个嵌套列表。不包含任何元素的列表称为空列表；可以使用空括号`[]`创建一个空列表。

正如您所期望的，您可以将列表值分配给变量：

```py
>>> cities = ['Mumbai', 'Pune', 'Chennai']
>>> numbers_list = [75, 857]
>>> empty_list = []
>>> print (cities, numbers_list, empty_list)
['Mumbai', 'Pune', 'Chennai'] [75, 857] []
```

# 访问列表中的值

我们可以通过使用索引值从列表中访问值。我们将在`[和]`中指定索引号。索引从`0`开始。参考给定的示例：

```py
#!/usr/bin/python3
cities = ['Mumbai', 'Bangalore', 'Chennai', 'Pune']
numbers = [1, 2, 3, 4, 5, 6, 7 ]
print (cities[0])
print (numbers[1:5])

Output:
Mumbai
[2, 3, 4, 5]
```

# 更新列表

您可以更新列表的元素，如下面的代码所示：

```py
#!/usr/bin/python3
cities = ['Mumbai', 'Bangalore', 'Chennai', 'Pune']
print ("Original Value: ", cities[3])
cities[3] = 'Delhi'
print ("New value: ", cities[3])

Output:
Original Value: Pune
New value: Delhi
```

# 删除列表元素

要删除列表元素，可以使用`del`语句（如果知道要删除的确切元素），也可以使用`remove()`方法（如果不知道要删除哪些项目）。参考以下示例：

```py
#!/usr/bin/python3
cities = ['Mumbai', 'Bangalore', 'Chennai', 'Pune']
print ("Before deleting: ", cities)
del cities[2]
print ("After deleting: ", cities)

Output:
Before deleting: ['Mumbai', 'Bangalore', 'Chennai', 'Pune']
After deleting: ['Mumbai', 'Bangalore', 'Pune']
```

# 基本列表操作

有五种基本的列表操作：

+   连接

+   重复

+   长度

+   成员资格

+   迭代

| **描述** | **表达式** | **结果** |
| 连接 | ` [30, 50, 60] + ['Hello', 75, 66]` | ` [30,50,60,'Hello',75,66]` |
| 成员资格 | ` 45 in [45,58,99,65]` | ` True` |
| 迭代 | ` for x in [45,58,99] : print (x,end = ' ')` | ` 45 58 99` |
| 重复 | ` ['Python'] * 3` | ` ['python', 'python', 'python']` |
| 长度 | ` len([45, 58, 99, 65])` | ` 4` |

# 列表操作

在本节中，我们将学习基本的列表操作：连接和重复。

`+`运算符连接列表：

```py
>>> a = [30, 50, 60]
>>> b = ['Hello', 75, 66 ]
>>> c = a + b
>>> print c
[30,50,60,'Hello',75,66]
```

类似地，`*`运算符重复给定次数的列表：

```py
>>> [0] * 4
[0, 0, 0, 0]
>>> ['Python'] * 3
['python', 'python', 'python']
```

# 索引、切片和矩阵

列表索引的工作方式与字符串索引相同。可以使用`index`访问值。如果尝试读取或写入不存在的元素，则会收到`IndexError`。如果索引具有负值，则从列表的末尾开始向后计数。

现在，我们将创建一个名为`cities`的列表，并查看索引操作：

`cities = ['Mumbai', 'Bangalore', 'Chennai', 'Pune']`

| **描述** | **表达式** | **结果** |
| 索引从零开始 | `cities[2]` | `'Chennai'` |
| 切片：获取部分 | `cities[1:]` | `['Bangalore', 'Chennai', 'Pune']` |
| 负数：从右边计数 | `cities[-3]` | `'Bangalore'` |

# 元组

Python的元组数据结构是不可变的，意味着我们不能改变元组的元素。基本上，元组是由逗号分隔并括在括号`( )`中的值序列。与列表一样，元组是有序的元素序列：

```py
>>> t1 = 'h', 'e', 'l', 'l', 'o'
```

元组用括号`( )`括起来：

```py
>>> t1 = ('h', 'e', 'l', 'l', 'o')
```

您还可以创建一个只有一个元素的元组。您只需在元组中放置最后一个逗号：

```py
>>> t1 = 'h',
>>> type(t1)
<type 'tuple'>
```

括号中的值不是元组：

```py
>>> t1 = ('a')
>>> type(t1)
<type 'str'>
```

我们可以使用`tuple()`函数创建一个空元组：

```py
>>> t1 = tuple()
>>> print (t1)
()
```

如果参数是一个序列（字符串、列表或元组），则结果是具有序列元素的元组：

```py
>>> t = tuple('mumbai')
>>> print t
('m', 'u', 'm', 'b', 'a', 'i')
```

元组的值在括号`（）`之间用逗号分隔：

```py
>>> t = ('a', 'b', 'c', 'd', 'e')
>>> print t[0]
'a'
```

切片运算符选择一系列元素。

```py
>>> print t[1:3]
('b', 'c')
```

# 访问元组中的值

要访问元组中的值，请使用方括号进行切片，并使用索引或索引来获取该索引或索引处的值，如下例所示：

```py
#!/usr/bin/python3
cities = ('Mumbai', 'Bangalore', 'Chennai', 'Pune')
numbers = (1, 2, 3, 4, 5, 6, 7)
print (cities[3])
print (numbers[1:6])

Output:
Pune
(2, 3, 4, 5)
```

# 更新元组

在Python中无法更新元组，因为元组是不可变的。但是可以使用现有元组创建一个新元组，如下例所示：

```py
#!/usr/bin/python3
cities = ('Mumbai', 'Bangalore', 'Chennai', 'Pune')
numbers = (1,2,3,4,5,6,7)
tuple1 = cities + numbers
print(tuple1)

Output:
('Mumbai', 'Bangalore', 'Chennai', 'Pune', 1, 2, 3, 4, 5, 6, 7)
```

# 删除元组元素

我们无法删除单个元组元素。因此，要显式删除整个元组，请使用`del`语句。请参阅以下示例：

```py
#!/usr/bin/python3
cities = ('Mumbai', 'Bangalore', 'Chennai', 'Pune')
print ("Before deleting: ", cities)
del cities
print ("After deleting: ", cities)

Output:
Before deleting: ('Mumbai', 'Bangalore', 'Chennai', 'Pune')
Traceback (most recent call last):
File "01.py", line 5, in <module>
print ("After deleting: ", cities)
NameError: name 'cities' is not defined
```

# 基本元组操作

与列表一样，元组有五种基本操作：

+   连接

+   重复

+   长度

+   成员资格

+   迭代

| **描述** | **表达式** | **结果** |
| 迭代 | ` for x in (45,58,99) : print (x,end = ' ')` | ` 45 58 99` |
| 重复 | ` ('Python') * 3` | ` ('python', 'python', 'python')` |
| 长度 | ` len(45, 58, 99, 65)` | ` 4` |
| 连接 | ` (30, 50, 60) + ('Hello', 75, 66)` | ` (30,50,60,'Hello',75,66)` |
| 成员资格 | ` 45 in (45,58,99,65)` | ` True` |

# 索引、切片和矩阵

元组索引的工作方式与列表索引相同。可以使用索引访问值。如果尝试读取或写入不存在的元素，则会收到`IndexError`。如果索引具有负值，则从列表末尾向后计数。

现在，我们将创建一个名为`cities`的元组并执行一些索引操作：

`cities = ('Mumbai', 'Bangalore', 'Chennai', 'Pune')`

| **描述** | **表达式** | **结果** |
| 索引从零开始 | `cities[2]` | `'Chennai'` |
| 切片：获取部分 | `cities[1:]` | `('Bangalore', 'Chennai', 'Pune')` |
| 负数：从右边计数 | `cities[-3]` | `'Bangalore'` |

# max()和min()

使用`max（）`和`min（）`函数，我们可以从元组中找到最高和最低的值。这些函数允许我们获取有关定量数据的信息。让我们看一个例子：

```py
>>> numbers = (50, 80,98, 110.5, 75, 150.58)
>>> print(max(numbers))
150.58
>>>
```

使用`max（）`，我们将获得元组中的最大值。类似地，我们可以使用`min（）`函数：

```py
>>> numbers = (50, 80,98, 110.5, 75, 150.58)
>>> print(min(numbers))
50
>>>
```

因此，在这里我们得到了最小值。

# 集合

集合是一个无序的元素集合，没有重复项。集合的基本用途是检查成员资格测试和消除重复条目。这些集合对象支持数学运算，如并集、交集、差集和对称差。我们可以使用大括号或`set()`函数创建一个集合。如果要创建一个空集合，则使用`set()`而不是`{}`。

以下是一个简要演示：

```py
>>> fruits = {'Mango', 'Apple', 'Mango', 'Watermelon', 'Apple', 'Orange'}
>>> print (fruits)
{'Orange', 'Mango', 'Apple', 'Watermelon'}
>>> 'Orange' in fruits
True
>>> 'Onion' in fruits
False
>>>
>>> a = set('abracadabra')
>>> b = set('alacazam')
>>> a
{'d', 'c', 'r', 'b', 'a'}
>>> a - b
{'r', 'd', 'b'}
>>> a | b
{'d', 'c', 'r', 'b', 'm', 'a', 'z', 'l'}
>>> a & b
{'a', 'c'}
>>> a ^ b
{'r', 'd', 'b', 'm', 'z', 'l'}
```

Python还支持集合解析。请参阅以下代码：

```py
>>> a = {x for x in 'abracadabra' if x not in 'abc'}
>>> a
{'r', 'd'}
```

# 字典

字典是Python中的一种数据类型，由键值对组成，括在大括号`{}`中。字典是无序的，并由键索引，其中每个键必须是唯一的。这些键必须是不可变类型。如果元组只包含字符串、数字或元组，则可以将元组用作键。

只有一对大括号会创建一个空字典：`{}`。字典的主要操作是使用某个键存储值并提取给定键的值。还可以使用`del`删除键值对。如果使用已经使用的键进行存储，则与该键关联的旧值将被遗忘。使用不存在的键提取值是错误的。以下是使用字典的一个小例子：

```py
>>> student = {'Name':'John', 'Age':25}
>>> student['Address'] = 'Mumbai'
>>> student
student = {'Name':'John', 'Age':25, 'Address':'Mumbai'}
>>> student['Age']
25
>>> del student['Address']
>>> student
student = {'Name':'John', 'Age':25}
>>> list(student.keys())
['Name', 'Age']
>>> sorted(student.keys())
['Age', 'Name']
>>> 'Name' in student
True
>>> 'Age' not in student
False
```

使用字典解析还支持任意键和值表达式来创建字典：

```py
>>> {x: x**2 for x in (4, 6, 8)}
{4: 16, 6: 36, 8: 64}
```

当键是简单字符串时，有时使用关键字参数指定对更容易：

```py
>>> dict(John=25, Nick=27, Jack=28)
{'Nick': 27, 'John': 25, 'Jack': 28}
```

# 解析命令行参数

在本节中，我们将学习解析参数和用于解析参数的模块。

# Python中的命令行参数

我们可以在命令行中使用额外的参数来启动程序。Python程序可以使用命令行参数启动。让我们看一个例子：

```py
$ python program_name.py img.jpg
```

在这里，`program_name.py`和`img.jpg`是参数。

现在，我们将使用模块来获取参数：

| **模块** | **用途** | **Python版本** |
| `optparse` | 已弃用 | `< 2.7` |
| `sys` | `sys.argv`中的所有参数（基本） | 所有 |
| `argparse` | 构建命令行界面 | `>= 2.3` |
| `fire` | 自动生成**命令行界面**（**CLIs**） | 所有 |
| `docopt` | 创建CLIs界面 | `>= 2.5` |

# Sys.argv

`sys`模块用于访问命令行参数。`len(sys.argv)`函数包含参数的数量。要打印所有参数，只需执行`str(sys.argv)`。让我们看一个例子：

```py
01.py
import sys
print('Number of arguments:', len(sys.argv))
print('Argument list:', str(sys.argv))

Output:
Python3 01.py img
Number of arguments 2
Arguments list: ['01.py', 'img']
```

# 决策制定

当条件为`true`时，决策制定就派上用场了。`if...elif...else`语句用于在Python中进行决策制定。

# Python if语句语法

以下是`if`语句的语法：

```py
if test_expression:
 statement(s)
```

在这里，程序评估测试表达式，并且只有在测试表达式为`true`时才执行`语句`。如果测试表达式为`false`，则不执行`语句`。

在Python中，`if`语句的主体由缩进表示。主体以缩进开始，第一行不缩进的行标志着结束。让我们看一个例子：

```py
a = 10
if a > 0:
 print(a, "is a positive number.")
print("This statement is always printed.")

a = -10
if a > 0:
 print(a, "is a positive number.")

Output:
10 is a positive number.
This statement is always printed.
```

# Python if...else语句语法

在本节中，我们将学习`if..else`语句。只有当`if`条件为`false`时，`else`块才会被执行。请参考以下语法：

```py
if test expression:
 if block
else:
 else block
```

`if..else`语句评估测试表达式，并且只有在测试条件为`true`时才执行`if`的主体。如果条件为`false`，则执行`else`的主体。缩进用于分隔块。请参考以下示例：

```py
a = 10
if a > 0:
 print("Positive number")
else:
 print("Negative number")

Output:
Positive number
```

# Python if...elif...else语句

`elif`语句检查多个语句是否为`true`值。每当值评估为`true`时，该代码块就会被执行。请参考以下语法：

```py
if test expression:
 if block statements
elif test expression:
 elif block statements
else:
 else block statements
```

`elif`是`else if`的缩写。它允许我们检查多个表达式。如果`if`语句中的条件为`false`，那么它将检查下一个`elif`块的条件，依此类推。如果所有条件都为`false`，则执行`else`的主体。

根据条件，`if...elif...else`块中的多个块中只有一个块被执行。`if`块只能有一个`else`块。但它可以有多个`elif`块。让我们看一个例子：

```py
a = 10
if a > 50:
 print("a is greater than 50")
elif a == 10:
 print("a is equal to 10")
else:
 print("a is negative")

Output:
a is equal to 10
```

# 循环

为了处理脚本中的所有循环需求，Python支持两种循环：

+   `for循环`

+   `while循环`

现在，我们将学习`for循环`和`while循环`。

# for循环

`for循环`遍历序列或任何其他可迭代对象的每个项目，并且每次都会执行for块中的语句。请参考以下语法：

```py
for i in sequence:
 for loop body
```

在这里，`i`是变量，它在每次迭代时取序列内的项目的值。这个循环会一直持续，直到我们到达序列中的最后一个项目。这在下面的图表中有所说明：

![](assets/822a75d6-353a-4464-9de9-1cfbff43b9e9.png)

请参考以下示例：

```py
numbers = [6, 5, 3, 8, 4, 2, 5, 4, 11]
sum = 0
for i in numbers:
 sum = sum + i
 print("The sum is", sum)

Output:
The sum is 6
The sum is 11
The sum is 14
The sum is 22
The sum is 26
The sum is 28
The sum is 33
The sum is 37
The sum is 48
```

# range()函数

Python的`range()`函数将生成一个数字序列。例如，`range(10)`将生成从`0`到`9`的数字（10个数字）。

我们还可以将起始、停止和步长大小定义为参数，`range()`将如下所示：

```py
range(start, stop, step size).
Step size defaults to 1 if not provided.
For loop example using range() function:
```

让我们看一个例子：

```py
for i in range(5):
 print("The number is", i)

Output:
The number is 0
The number is 1
The number is 2
The number is 3
The number is 4
```

# while循环

`while`是一个循环语句，它将在输入的测试表达式为`true`时迭代一段代码块。当我们不知道迭代将进行多少次时，我们使用这个循环。请参考以下语法：

```py
while test_expression:
 while body statements
```

在while循环中，首先我们将检查测试表达式。只有在测试表达式为`true`时，`while`块才会被执行。经过一次迭代后，表达式将再次被检查，这个过程将继续，直到`test_expression`评估为`false`。这在下图中有所说明：

![](assets/9ce57a7e-fd01-43c8-87d0-61c72da1a24f.png)

以下是`while`循环的示例：

```py
a = 10
sum = 0
i = 1
while i <= a:
 sum = sum + i
 i = i + 1
 print("The sum is", sum)

Output:
The sum is 1
The sum is 3
The sum is 6
The sum is 10
The sum is 15
The sum is 21
The sum is 28
The sum is 36
The sum is 45
The sum is 55
```

# 迭代器

在Python中，迭代器是可以被迭代的对象。它是一个对象，每次返回一个元素的数据。Python的迭代器对象实现了两个方法，`__iter__()`和`__next__()`。大多数情况下，迭代器在循环、生成器和推导式中实现。

在以下示例中，我们使用`next()`函数，它将遍历所有项目。在到达末尾并且没有更多数据需要返回时，它将引发`StopIteration`，如下例所示：

```py
numbers = [10, 20, 30, 40]

numbers_iter = iter(numbers)

print(next(numbers_iter))
print(next(numbers_iter))
print(numbers_iter.__next__())
print(numbers_iter.__next__())

next(numbers_iter)

Output:
10
20
30
40
Traceback (most recent call last):
 File "sample.py", line 10, in <module>
 next(numbers_iter)
StopIteration

```

# 生成器

我们可以使用Python生成器创建迭代器。在Python中，生成器是一个返回可以迭代的对象的函数。

# 如何在Python中创建一个生成器？

在Python中创建生成器很容易。您可以通过定义一个带有`yield`语句而不是`return`语句的函数来创建生成器。如果一个函数包含至少一个`yield`语句，它就成为一个生成器函数。`yield`和`return`语句将从函数返回一些值。以下是一个例子：

```py
def my_gen():
 n = 1
 print('This is printed first')
 yield n
 n += 1
 print('This is printed second')
 yield n
 n += 1
 print('This is printed at last')
 yield n
for item in my_gen():
 print(item)

Output:
This is printed first
1
This is printed second
2
This is printed at last
3
```

# 函数

函数是执行特定任务的一组语句。使用函数有助于将程序分解为更小的部分。如果使用函数，程序将更有组织性，因为它避免了重复，并使代码可重用。看一下以下语法：

```py
def function_name(parameters):
 statement(s)
```

参考以下示例：

```py
def welcome(name):
 print("Hello " + name + ", Welcome to Python Programming !")
 welcome("John")

Output:
Hello John, Welcome to Python Programming !
```

# 返回语句

`return`语句用于退出函数。参考以下语法：

```py
return [expression_list]
```

此语句可能包含一个表达式，其中必须返回一个值。如果没有表达式，那么函数将返回一个None对象，如下例所示：

```py
def return_value(a):
 if a >= 0:
 return a
 else:
 return -a
print(return_value(2))
print(return_value(-4))

Output:
2
4
```

# Lambda函数

在Python中，匿名函数是没有名称定义的函数，称为`lambda`函数，因为它是使用关键字`lambda`定义的。我们在需要短时间内使用函数时使用这些函数。

Lambda函数与内置函数一起使用，例如`filter()`和`map()`。

filter()函数返回一个元素列表，并且只有一个可迭代的输入。以下是使用`filter()`的示例：

```py
numbers = [10, 25, 54, 86, 89, 11, 33, 22]
new_numbers = list(filter(lambda x: (x%2 == 0) , numbers))
print(new_numbers)

Output:
[10, 54, 86, 22]
```

在这个例子中，`filter()`函数接受一个`lambda`函数和一个列表作为参数。

`map()`函数在应用指定函数后返回结果列表。现在，让我们看一个使用`map()`的示例：

```py
my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list = list(map(lambda x: x * 2 , my_list))
print(new_list)

Output:
[2, 10, 8, 12, 16, 22, 6, 24]
```

在这里，`map()`函数接受一个`lambda`函数和一个列表。

# 模块

模块只是包含Python语句和定义的文件。包含Python代码的文件（例如，`sample.py`）被称为模块，其模块名称将是`sample`。使用模块，我们可以将较大的程序分解为小的有组织的部分。模块的一个重要特性是可重用性。您可以在模块中定义最常用的函数的定义，而不是在不同的程序中复制它们，只需在需要时导入它们。

让我们创建一个模块并导入它。我们将创建两个脚本：`sample.py`和`add.py`。我们将在`add.py`中导入一个示例模块。现在，将以下代码保存为`sample.py`。让我们看看以下示例：

```py
sample.py
def addition(num1, num2):
 result = num1 + num2
 return result
```

在这里，我们在名为`sample`的模块中定义了一个`addition()`函数。该函数接受两个数字并返回它们的和。现在我们已经创建了一个模块。您可以在任何Python程序中导入它。

# 导入模块

现在，在创建模块之后，我们将学习如何导入该模块。在前面的示例中，我们创建了一个示例模块。现在我们将在`add.py`脚本中导入示例模块：

```py
add.py
import sample
sum = sample.addition(10, 20)
print(sum)

Output:
30
```

# 总结

在本章中，我们概述了Python脚本语言。我们学习了如何安装Python和各种工具。我们还学习了Python解释器以及如何使用它。我们了解了Python支持的数据类型、变量、数字和字符串、决策语句以及循环语句。我们还学习了函数以及如何在脚本和模块中使用它们以及如何创建和导入它们。

在下一章《调试和分析Python脚本》中，您将学习Python调试技术、错误处理（异常处理）、调试工具、调试基本程序崩溃、程序分析和计时、以及使程序运行更快的方法。

# 问题

1.  迭代器和生成器是什么？

1.  列表是可变的还是不可变的？

1.  Python中的数据结构是什么？

1.  如何访问列表中的值？

1.  模块是什么？

# 进一步阅读

所有Python文档都可以在以下网站上找到：[www.python.org](http://www.python.org)。

您还可以查阅以下书籍，《学习Python的艰难方式》和《Python之字节》，以了解Python的基础知识。
