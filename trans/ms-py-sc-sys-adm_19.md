# 第十九章：评估

# 第一章，Python 脚本概述

1.  迭代器是可以被迭代的对象。它是一个会返回数据的对象，每次返回一个元素。生成器是一个可以迭代的函数，它返回一个对象。

1.  列表是可变的。

1.  Python 中的数据结构是可以一起保存一些数据的结构。换句话说，它们用于存储相关数据的集合。

1.  我们可以通过使用索引值来访问列表中的值。

1.  模块只是包含 Python 语句和定义的文件。

# 第二章，调试和分析 Python 脚本

1.  要调试程序，使用`pdb`模块。

1.  a) 在运行`ipython3`之前，使用`sudo apt-get install ipython3`进行安装。

b) `%lsmagic`。

1.  全局解释器锁是计算机语言解释器中使用的一种机制，用于同步线程的执行，以便一次只有一个本机线程可以执行

1.  以下是答案：

a) `PYTHONPATH`：它的作用类似于 PATH。此变量告诉 Python 解释器在程序中导入的模块文件的位置。它应该包括 Python 源库目录和包含 Python 源代码的目录。`PYTHONPATH`有时会被 Python 安装程序预设。

b) `PYTHONSTARTUP`：它包含包含 Python 源代码的初始化文件的路径。每次启动解释器时都会执行它。在 Unix 中，它被命名为`.pythonrc.py`，它包含加载实用程序或修改`PYTHONPATH`的命令。

c) `PYTHONCASEOK`：在 Windows 中用于指示 Python 在导入语句中找到第一个不区分大小写的匹配项。将此变量设置为任何值以激活它。

d) `PYTHONHOME`：这是一个替代的模块搜索路径。通常嵌入在`PYTHONSTARTUP`或`PYTHONPATH`目录中，以便轻松切换模块库。

1.  答案：`[0]`。

在函数中创建了一个新的列表对象，并且引用丢失了。可以通过比较`k`在`k = [1]`之前和之后的 ID 来检查这一点。

1.  答案：b. 变量名不应以数字开头。

# 第三章，单元测试 - 单元测试框架简介

1.  单元测试是软件测试的一种级别，其中测试软件的各个单元/组件。目的是验证软件的每个单元是否按设计执行。

自动化测试是一种自动化技术，其中测试人员自己编写脚本并使用适当的软件来测试软件。基本上是手动流程的自动化过程。

手动测试是发现软件程序中的缺陷或错误的过程。在这种方法中，测试人员扮演端用户的重要角色，并验证应用程序的所有功能是否正常工作。

1.  Unittest，mock，nose，`pytest`。

1.  测试用例是为验证软件应用程序的特定功能或功能而执行的一组操作。本教程描述了测试用例的设计以及其各个组件的重要性。

1.  PEP 8 是 Python 的风格指南。这是一组规则，用于格式化您的 Python 代码，以最大限度地提高其可读性。按照规范编写代码有助于使具有许多编写者的大型代码库更加统一和可预测。

# 第四章，自动化常规管理活动

1.  `readline()`方法从文件中读取整行。字符串中保留了尾随的换行符。如果存在大小参数并且为非负，则它是包括尾随换行符在内的最大字节计数，并且可能返回不完整的行。

1.  读取：`cat`。

创建新文件：`touch`。

删除文件：`rm`。

列出当前目录中的文件：`ls`。

1.  以下是答案：

```py
os.system(“shell_command”)
subprocess.getstatusoutput(“shell_command”)
```

1.  以下是答案：

```py
import configparser as config
config.set(section, option, value)
```

1.  以下是答案：

```py
 psutil, fabric, salt, asnible, buildbot, shinken
```

1.  以下是答案：

```py
input() 
sys.stdin.readline()
```

1.  当您想要改变列表时使用`list.sort()`，当您想要一个新的排序对象时使用`sorted()`。对于尚未是列表的可迭代对象，使用`sorted()`进行排序更快。对于列表，`list.sort()`比`sorted()`更快，因为它不必创建副本。

# 第五章，处理文件、目录和数据

1.  通过使用`pathlib`库。

1.  以下是答案：

```py
print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
```

1.  如果没有参数调用，则返回当前范围内的名称。否则，返回给定对象的属性（部分）组成的名称的按字母顺序排列的列表，以及可从该对象到达的属性。

1.  DataFrame 是一个二维大小、可变且可能异构的带标签轴的表格数据结构。

Series 是 DataFrame 的单列数据结构，不仅在概念上是如此，而且在内存中实际上是作为一系列存储的。

1.  列表推导提供了一种简洁的方法来创建新列表。

1.  是的：

```py
Set comprehension {s**2 for s in range(10)}
Dict comprehension {n: n**2 for n in range(5)}
```

1.  以下是答案：

```py
df.head(number of lines) default blank 
df.tail(number of lines) default blank
```

1.  以下是答案：

```py
[i for i in range(10) if i%2]
```

1.  答案：b。这是一个元素列表。

# 第六章，文件归档、加密和解密

1.  是的，使用 Python 的`pyminizip`库。

1.  上下文管理器是一种在需要时精确分配和释放某种资源的方法。最简单的例子是文件访问：

```py
with open ("foo", 'w+') as foo:
foo.write("Hello!")
is similar to
foo = open ("foo", 'w+'):
 foo.write("Hello!")
foo.close()
```

1.  在 Python 中，pickling 指的是将对象序列化为二进制流的过程，而 unpickling 是其相反过程。

1.  无参数且无返回值的函数

无参数且有返回值的函数

带参数且无返回值的函数

带参数和返回值的函数

# 第七章，文本处理和正则表达式

1.  正则表达式是编程中用于模式匹配的方法。正则表达式提供了一种灵活而简洁的方法来匹配文本字符串。

1.  以下是答案：

```py
import redef is_allowed_specific_char(string):
 charRe = re.compile(r'[^a-zA-Z0-9.]')
 string = charRe.search(string)
 return not bool(string)
 print(is_allowed_specific_char("ABCDEFabcdef123450"))
 print(is_allowed_specific_char("*&%@#!}{"))
```

1.  答案：a。

`re`是标准库的一部分，可以使用`import re`导入。

1.  答案：a。

它将在开头查找模式，如果找不到则返回`None`。

1.  答案：d。

此函数返回整个匹配。

# 第八章，文档和报告

1.  主要区别在于当您使用`input`和`print`函数时，所有的输出格式化工作都是在幕后完成的。stdin 用于所有交互式输入，包括对`input()`的调用；stdout 用于`print()`和表达式语句的输出以及`input()`的提示。

1.  **简单邮件传输协议**（**SMTP**）是用于电子邮件传输的互联网标准。最初由 RFC 821 在 1982 年定义，2008 年通过 RFC 5321 进行了扩展 SMTP 的更新，这是今天广泛使用的协议。

1.  以下是答案：

```py
Hi Eric. You are a comedian. You were in Monty Python.
```

1.  以下是答案：

```py
str1 + str2 = HelloWorld!
str1 * 3 = HelloHelloHello
```

# 第九章，处理各种文件

1.  `f.readline()`从文件中读取一行；一个换行符（\n）留在字符串的末尾，并且只有在文件的最后一行没有换行符时才会被省略。如果要将文件的所有行读入列表中，还可以使用`list(f)`或`f.readlines()`。

1.  基本上，使用`with open()`只是确保您不会忘记`close()`文件，使其更安全/防止内存问题。

1.  `r`表示该字符串将被视为原始字符串。

1.  生成器简化了迭代器的创建。生成器是一个产生一系列结果而不是单个值的函数。

1.  在 Python 中，pass 语句用于在语法上需要语句但您不希望执行任何命令或代码时使用。pass 语句是一个空操作；执行时什么也不会发生。

1.  在 Python 中，匿名函数是在没有名称的情况下定义的函数。而普通函数是使用`def`关键字定义的，Python 中的匿名函数是使用`lambda`关键字定义的。因此，匿名函数也称为 lambda 函数。

# 第十章，基本网络 - 套接字编程

1.  套接字编程涉及编写计算机程序，使进程能够在计算机网络上相互通信。

1.  在分布式计算中，远程过程调用是指计算机程序导致在不同地址空间中执行过程，这是编码为正常过程调用，程序员不需要显式编码远程交互的细节。

1.  以下是答案：

```py
import filename (import file)
from filename import function1 (import specific function)
from filename import function1, function2(import multiple functions)
from filename import * (import all the functions)
```

1.  列表和元组之间的主要区别在于列表是可变的，而元组是不可变的。可变数据类型意味着可以修改此类型的 Python 对象。不可变意味着不能修改此类型的 Python 对象。

1.  你不能有一个带有重复键的字典，因为在后台它使用了一个哈希机制。

1.  `urllib`和`urllib2`都是执行 URL 请求相关操作的 Python 模块，但提供不同的功能。

`urllib2`可以接受一个请求对象来设置 URL 请求的标头，`urllib`只接受一个 URL。Python 请求会自动编码参数，因此您只需将它们作为简单参数传递。

# 第十一章，使用 Python 脚本处理电子邮件

1.  在计算中，邮局协议是一种应用层互联网标准协议，用于电子邮件客户端从邮件服务器检索电子邮件。 POP 版本 3 是常用的版本。**Internet Message Access Protocol**（**IMAP**）是一种互联网标准协议，用于电子邮件客户端通过 TCP/IP 连接从邮件服务器检索电子邮件消息。 IMAP 由 RFC 3501 定义。

1.  break 语句终止包含它的循环。程序的控制流流向循环体之后的语句。如果 break 语句在嵌套循环（一个循环内部的循环）中，break 将终止最内层的循环。以下是一个例子：

```py
for val in "string":
 if val == "i":
 break
 print(val)
print("The end")
```

1.  continue 语句用于仅跳过当前迭代中循环内部的其余代码。循环不会终止，而是继续下一个迭代：

```py
for val in "string":
 if val == "i":
 continue
 print(val)
print("The end")
```

1.  `pprint`模块提供了一种能够以可用作解释器输入的形式漂亮打印任意 Python 数据结构的功能。如果格式化的结构包括不是基本 Python 类型的对象，则表示可能无法加载。如果包括文件、套接字、类或实例等对象，以及许多其他无法表示为 Python 常量的内置对象，可能会出现这种情况。

1.  在 Python 中，负索引用于从列表、元组或支持索引的任何其他容器类的最后一个元素开始索引。`-1`指的是*最后一个索引*，`-2`指的是*倒数第二个索引*，依此类推。

1.  Python 编译`.py`文件并将其保存为`.pyc`文件，以便在后续调用中引用它们。`.pyc`包含 Python 源文件的已编译字节码。`.pyc`包含 Python 源文件的已编译字节码，这是 Python 解释器将源代码编译为的内容。然后，Python 的虚拟机执行此代码。删除`.pyc`不会造成任何损害，但如果要进行大量处理，它们将节省编译时间。

1.  以下是答案：

```py
num = 7
for index in range(num,0,-1):
if index % 2 != 0:
for row in range(0,num-index):
print(end=" ")
for row in range(0,index):
if row % 2== 0:
print("1",end=" ")
else:
print("0",end=" ")
print()
```

# 第十二章，通过 Telnet 和 SSH 远程监视主机

1.  客户端-服务器模型是一种分布式应用程序结构，它在资源或服务的提供者（称为服务器）和服务请求者（称为客户端）之间分配任务或工作负载。

1.  通过使用以下内容：

```py
os.commands(command_name)
subprocess.getstatusoutput(command_name)
```

1.  虚拟局域网是在数据链路层上分区和隔离的任何广播域，局域网是本地区域网络的缩写，在这种情况下，虚拟指的是通过附加逻辑重新创建和改变的物理对象。

1.  答案：`[]`。

它打印一个空列表，因为列表的大小小于 10。

1.  以下是答案：

```py
import calender
calendar.month(1,1)
```

1.  以下是答案：

```py
def file_lengthy(fname):
 with open(fname) as f:
 for i, l in enumerate(f):
 pass
 return i + 1
print("Number of lines in the file: ",file_lengthy("test.txt"))
```

# 第十三章，构建图形用户界面

1.  图形用户界面，允许用户与电子设备进行交互。

1.  构造函数是一种特殊类型的方法（函数），用于初始化类的实例成员。`__init__ 方法`的实现。析构函数是在对象销毁期间自动调用的特殊方法。`__del__ 方法`的实现。

1.  Self 是对对象本身的对象引用；因此，它们是相同的。

1.  Tkinter 是 Python 绑定到 Tk GUI 工具包的工具。它是 Tk GUI 工具包的标准 Python 接口，也是 Python 的事实标准 GUI。Tkinter 包含在标准的 Linux、Microsoft Windows 和 macOS X 的 Python 安装中。Tkinter 的名称来自 Tk 界面。PyQt 是跨平台 GUI 工具包 Qt 的 Python 绑定，实现为 Python 插件。PyQt 是由英国公司 Riverbank Computing 开发的免费软件。wxPython 是 Python 编程语言的跨平台 GUI API wxWidgets 的包装器。它是 Tkinter 的替代品之一，与 Python 捆绑在一起。它实现为 Python 扩展模块。其他流行的替代品是 PyGTK，它的后继者 PyGObject 和 PyQt。

1.  以下是答案：

```py
def copy(source, destination):
 with open(source, "w") as fw, open(destination,"r") as fr:
 fw.writelines(fr)
copy(source_file_name1, file_name2)
```

1.  以下是答案：

```py
fname = input("Enter file name: ")
l=input("Enter letter to be searched:")
k = 0
with open(fname, 'r') as f:
 for line in f:
 words = line.split()
 for i in words:
 for letter in i:
 if(letter==l):
 k=k+1
print("Occurrences of the letter:")
print(k)
```

# 第十四章，使用 Apache 和其他日志文件

1.  运行时异常发生在程序执行期间，它们会在中途突然退出。编译时异常是在程序执行开始之前发现的异常。

1.  正则表达式、regex 或 regexp 是定义搜索模式的字符序列。通常，这种模式由字符串搜索算法用于字符串的查找或查找和替换操作，或用于输入验证。

1.  以下是 Linux 命令的描述：

+   `head`：用于查看普通文件的前 N 行。

+   `tail`：用于查看普通文件的最后 N 行。

+   `cat`：用于查看普通文件的内容。

+   `awk`：AWK 是一种专为文本处理而设计的编程语言，通常用作数据提取和报告工具。它是大多数类 Unix 操作系统的标准功能。

1.  以下是答案：

```py
def append(source, destination):
 with open(source, "a") as fw, open(destination,"r") as fr:
 fw.writelines(fr)
append(source_file_name1, file_name2)
```

1.  以下是答案：

```py
filename=input("Enter file name: ")
for line in reversed(list(open(filename))):
 print(line.rstrip())
```

1.  表达式的输出如下：

1.  `C@ke`

1.  `Cooookie`

1.  `<h1>`

# 第十五章，SOAP 和 REST API 通信

1.  REST 基本上是 Web 服务的一种架构风格，它作为不同计算机或系统之间的通信渠道在互联网上工作。SOAP 是一种标准的通信协议系统，允许使用不同操作系统（如 Linux 和 Windows）的进程通过 HTTP 及其 XML 进行通信。基于 SOAP 的 API 旨在创建、恢复、更新和删除记录，如账户、密码、线索和自定义对象。

1.  `json.load`可以反序列化文件本身；也就是说，它接受文件对象。

1.  是的。JSON 是平台无关的。

1.  答案：false。

1.  答案：`{'x': 3}`。

# 第十六章，网络抓取-从网站提取有用数据

1.  Web 抓取、网络收集或网络数据提取是用于从网站提取数据的数据抓取。Web 抓取软件可以直接使用超文本传输协议访问万维网，也可以通过 Web 浏览器访问。

1.  Web 爬虫（也称为网络蜘蛛或网络机器人）是以一种有条理、自动化的方式浏览万维网的程序或自动化脚本。这个过程称为网络爬行或蜘蛛。

1.  是的。

1.  是的，使用 Tweepy。

1.  是的，通过使用 Selenium-Python 网络驱动程序。还有其他库可用，如 PhantomJS 和 dryscrape。

# 第十七章，统计收集和报告

1.  NumPy 的主要对象是同质多维数组。它是一张元素表（通常是数字），都是相同类型的，由正整数元组索引。在 NumPy 中，维度被称为轴。

1.  以下是输出：

```py
1st Input array : 
 [[ 1 2 3]
 [-1 -2 -3]]
2nd Input array : 
 [[ 4 5 6]
 [-4 -5 -6]]
Output stacked array :
 [[ 1 2 3 4 5 6]
 [-1 -2 -3 -4 -5 -6]]
```

1.  以下是答案：

```py
Z = np.arange(10)
np.add.reduce(Z)
```

1.  以下是答案：

```py
# Delete the rows with labels 0,1,5
data = data.drop([0,1,2], axis=0)
# Delete the first five rows using iloc selector
data = data.iloc[5:,]
#to delete the column
del df.column_name
```

1.  以下是答案：

```py
df.to_csv(“file_name.csv”,index=False, sep=”,”)
```

1.  **不是** **数字**（NaN），比如空值。在 pandas 中，缺失值用 NaN 表示。

1.  以下是答案：

```py
df.drop_duplicates()
```

1.  以下是答案：

```py
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
```

1.  Matplotlib、Plotly 和 Seaborn。

# 第十八章，MySQL 和 SQLite 数据库管理

1.  将数据存储在行和列中，并且可以轻松快速地执行不同的操作。

1.  在数据库中，CRUD 代表（创建，读取，更新，删除）。

1.  是的，这里有一个例子：

```py
MySQLdb.connect('remote_ip', 'username', 'password', 'databasename')
```

1.  是的。

1.  **DDL**代表**数据定义语言**。它用于定义数据结构。例如，使用 SQL，它将是创建表，修改表等指令。**DML**代表**数据操作语言**。它用于操作数据本身。例如，使用 SQL，它将是插入，更新和删除等指令。
