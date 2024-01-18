# 算术运算、循环和闪烁灯

现在让我们来看看这一章，我们将回顾 Python 中的算术运算和变量。我们还将讨论 Python 中的字符串和接受用户输入。您将了解树莓派的 GPIO 及其特性，并使用 Python 编写代码，使 LED 使用树莓派 Zero 的 GPIO 闪烁。我们还将讨论控制树莓派的 GPIO 的实际应用。

在本章中，我们将涵盖以下主题：

+   Python 中的算术运算

+   Python 中的位运算符

+   Python 中的逻辑运算符

+   Python 中的数据类型和变量

+   Python 中的循环

+   树莓派 Zero 的 GPIO 接口。

# 本章所需的硬件

在本章中，我们将讨论一些例子，我们将控制树莓派的 GPIO。我们需要一个面包板，跳线，LED 和一些电阻（330 或 470 欧姆）来讨论这些例子。

我们还需要一些可选的硬件，我们将在本章的最后一节中讨论。

# 算术运算

Python 可以执行所有标准的算术运算。让我们启动 Python 解释器，了解更多：

+   **加法**：可以使用`+`操作符对两个数字进行相加。结果将打印在屏幕上。使用 Python 解释器尝试以下示例：

```py
       >>>123+456 
       579
```

+   **减法**：可以使用`-`操作符对两个数字进行相加：

```py
       >>>456-123 
       333 
       >>>123-456 
       -333
```

+   **乘法**：可以将两个数字相乘如下：

```py
       >>>123*456 
       56088
```

+   **除法**：可以将两个数字相除如下：

```py
       >>>456/22 
 20.727272727272727 
       >>>456/2.0 
       228.0 
       >>>int(456/228) 
       2
```

+   **模运算符**：在 Python 中，模运算符（`%`）返回除法运算的余数：

```py
       >>>4%2 
       0 
       >>>3%2 
       1
```

+   **floor 运算符**（`//`）是模运算符的相反。此运算符返回商的地板，即整数结果，并丢弃小数部分：

```py
       >>>9//7 
       1 
       >>>7//3 
       2 
       >>>79//25 
       3
```

# Python 中的位运算符

在 Python 中，可以对数字执行位级操作。这在从某些传感器解析信息时特别有帮助。例如，一些传感器以一定频率共享它们的输出。当新的数据点可用时，设置某个特定的位，表示数据可用。可以使用位运算符来检查在从传感器检索数据点之前是否设置了特定的位。

如果您对位运算符有兴趣，我们建议从[`en.wikipedia.org/wiki/Bitwise_operation`](https://en.wikipedia.org/wiki/Bitwise_operation)开始。

考虑数字`3`和`2`，它们的二进制等价物分别是`011`和`010`。让我们看看执行每个数字位操作的不同运算符：

+   **AND 运算符**：AND 运算符用于对两个数字执行 AND 操作。使用 Python 解释器尝试一下：

```py
       >>>3&2 
       2
```

这相当于以下 AND 操作：

```py
   0 1 1 &
   0 1 0
   --------
   0 1 0 (the binary representation of the number 2)
```

+   **OR 运算符**：OR 运算符用于对两个数字执行 OR 操作，如下所示：

```py
       >>>3|2 
       3
```

这相当于以下 OR 操作：

```py
   0 1 1 OR
   0 1 0
   --------
   0 1 1 (the binary representation of the number 3)
```

+   **NOT 运算符**：NOT 运算符翻转数字的位。看下面的例子：

```py
       >>>~1 
       -2
```

在前面的例子中，位被翻转，即`1`变为`0`，`0`变为`1`。因此，`1`的二进制表示是`0001`，当执行按位 NOT 操作时，结果是`1110`。解释器返回结果为`-2`，因为负数存储为它们的*二进制补码*。`1`的二进制补码是`-2`。

为了更好地理解二进制补码等内容，我们建议阅读以下文章，[`wiki.python.org/moin/BitwiseOperators`](https://wiki.python.org/moin/BitwiseOperators)和[`en.wikipedia.org/wiki/Two's_complement`](https://en.wikipedia.org/wiki/Two's_complement)。

+   **XOR 运算符**：可以执行异或操作如下：

```py
       >>>3² 
       1
```

+   **左移运算符**：左移运算符可以将给定值的位向左移动所需的位数。例如，将数字`3`向左移动一位会得到数字`6`。数字`3`的二进制表示是`0011`。将位左移一位将得到`0110`，即数字`6`：

```py
       >>>3<<1 
       6
```

+   **右移运算符**：右移运算符可以将给定值的位向右移动所需的位数。启动命令行解释器并自己尝试一下。当你将数字`6`向右移动一个位置时会发生什么？

# 逻辑运算符

**逻辑运算符**用于检查不同的条件并相应地执行代码。例如，检测与树莓派 GPIO 接口连接的按钮是否被按下，并执行特定任务作为结果。让我们讨论基本的逻辑运算符：

+   **等于**：等于（`==`）运算符用于比较两个值是否相等：

```py
       >>>3==3 
       True 
       >>>3==2 
       False
```

+   **不等于**：不等于（`!=`）运算符比较两个值，如果它们不相等，则返回`True`：

```py
       >>>3!=2 
       True 
       >>>2!=2 
       False
```

+   **大于**：此运算符（`>`）如果一个值大于另一个值，则返回`True`：

```py
       >>>3>2 
       True 
       >>>2>3 
       False
```

+   **小于**：此运算符比较两个值，如果一个值小于另一个值，则返回`True`：

```py
       >>>2<3 
       True 
       >>>3<2 
       False
```

+   **大于或等于（>=）**：此运算符比较两个值，如果一个值大于或等于另一个值，则返回`True`：

```py
       >>>4>=3 
       True 
       >>>3>=3 
       True 
       >>>2>=3 
       False
```

+   **小于或等于（<=）**：此运算符比较两个值，如果一个值小于或等于另一个值，则返回`True`：

```py
       >>>2<=2 
       True 
       >>>2<=3 
       True 
       >>>3<=2 
       False
```

# Python 中的数据类型和变量

在 Python 中，**变量**用于在程序执行期间存储结果或值在计算机的内存中。变量使得可以轻松访问计算机内存中的特定位置，并且使得编写用户可读的代码成为可能。

例如，让我们考虑这样一个情景，一个人想要从办公室或大学获得一张新的身份证。这个人将被要求填写一个包括他们的姓名、部门和紧急联系信息在内的相关信息的申请表。表格将有必需的字段。这将使办公室经理在创建新的身份证时参考表格。

同样，变量通过提供存储信息在计算机内存中的方式来简化代码开发。如果必须考虑存储器映射，编写代码将会非常困难。例如，使用名为 name 的变量比使用特定的内存地址如`0x3745092`更容易。

Python 中有不同种类的数据类型。让我们来回顾一下不同的数据类型：

+   一般来说，姓名、街道地址等都是由字母数字字符组成。在 Python 中，它们被存储为*字符串*。Python 中的字符串表示和存储在变量中如下：

```py
       >>>name = 'John Smith' 
       >>>address = '123 Main Street'
```

+   在 Python 中，*数字*可以存储如下：

```py
       >>>age = 29 
       >>>employee_id = 123456 
       >>>height = 179.5 
       >>>zip_code = 94560
```

+   Python 还可以存储*布尔*变量。例如，一个人的器官捐赠者状态可以是`True`或`False`：

```py
       >>>organ_donor = True
```

+   可以同时*赋值*多个变量的值：

```py
       >>>a = c= 1 
       >>>b = a
```

+   可以*删除*变量如下：

```py
       >>>del(a)
```

Python 中还有其他数据类型，包括列表、元组和字典。我们将在下一章中详细讨论这一点。

# 从用户读取输入

现在，我们将讨论一个简单的程序，要求用户输入两个数字，程序返回两个数字的和。现在，我们假设用户总是提供有效的输入。

在 Python 中，用户可以使用`input()`函数（[`docs.python.org/3/library/functions.html#input`](https://docs.python.org/3/library/functions.html#input)）提供输入给 Python 程序：

```py
    var = input("Enter the first number: ")
```

在前面的例子中，我们使用`input()`函数来获取用户输入的数字。`input()`函数将提示`("Enter the first number: ")`作为参数，并返回用户输入。在这个例子中，用户输入存储在变量`var`中。为了添加两个数字，我们使用`input()`函数请求用户提供两个数字作为输入：

```py
    var1 = input("Enter the first number: ") 
    var2 = input("Enter the second number: ") 
    total = int(var1) + int(var2) 
    print("The sum is %d" % total)
```

我们正在使用`input()`函数来获取两个数字的用户输入。在这种情况下，用户数字分别存储在`var1`和`var2`中。

用户输入是一个字符串。我们需要在将它们相加之前将它们转换为整数。我们可以使用`int()`函数将字符串转换为整数（[`docs.python.org/3/library/functions.html#int`](https://docs.python.org/3/library/functions.html#int)）。

`int()`函数将字符串作为参数，并返回转换后的整数。转换后的整数相加并存储在变量`total`中。前面的例子可与本章一起下载，名称为`input_function.py`。

如果用户输入无效，`int()`函数将抛出异常，表示发生了错误。因此，在本例中，我们假设用户输入是有效的。在后面的章节中，我们将讨论由无效输入引起的异常捕获。

以下快照显示了程序输出：

![](img/257520f6-52ab-41e9-9660-8f6d16cec262.png)input_function.py 的输出

# 格式化的字符串输出

让我们重新讨论前一节中讨论的例子。我们打印了结果如下：

```py
    print("The sum is %d" % total)
```

在 Python 中，可以格式化字符串以显示结果。在前面的例子中，我们使用`%d`来指示它是整数变量的占位符。这使得可以打印带有整数的字符串。除了作为`print()`函数的参数传递的字符串外，还传递需要打印的变量作为参数。在前面的例子中，变量是使用`%`运算符传递的。还可以传递多个变量：

```py
    print("The sum of %d and %d is %d" % (var1, var2, total))
```

也可以按以下方式格式化字符串：

```py
    print("The sum of 3 and 2 is {total}".format(total=5))
```

# str.format()方法

`format()`方法使用大括号（`{}`）作为占位符来格式化字符串。在前面的例子中，我们使用`total`作为占位符，并使用字符串类的格式化方法填充每个占位符。

# 读者的另一个练习

使用`format()`方法格式化一个带有多个变量的字符串。

让我们构建一个从用户那里获取输入并在屏幕上打印的控制台/命令行应用程序。让我们创建一个名为`input_test.py`的新文件（可与本章的下载一起使用），获取一些用户输入并在屏幕上打印它们：

```py
    name = input("What is your name? ") 
    address = input("What is your address? ") 
    age = input("How old are you? ") 

    print("My name is " + name) 
    print("I am " + age + " years old") 
    print("My address is " + address)
```

执行程序并查看发生了什么：

![](img/7adf2c27-707e-4c25-9fba-77779ac12dc4.png)input_test.py 的输出

前面的例子可与本章一起下载，名称为`input_test.py`。

# 读者的另一个练习

使用字符串格式化技术重复前面的例子。

# 连接字符串

在前面的例子中，我们将用户输入与另一个字符串组合打印出来。例如，我们获取用户输入`name`并打印句子`My name is Sai`。将一个字符串附加到另一个字符串的过程称为**连接**。

在 Python 中，可以通过在两个字符串之间添加`+`来连接字符串：

```py
    name = input("What is your name? ") 
    print("My name is " + name)
```

可以连接两个字符串，但不能连接整数。让我们考虑以下例子：

```py
    id = 5 
    print("My id is " + id)
```

它将抛出一个错误，暗示整数和字符串不能结合使用：

![](img/77a4564b-3376-4e22-aa7d-3ec13fb3ae7d.png)一个异常

可以将整数转换为字符串并将其连接到另一个字符串：

```py
    print("My id is " + str(id))
```

这将产生以下结果：

![](img/58856394-d5d4-4de2-bf76-2ecba2e749d5.png)

# Python 中的循环

有时，特定任务必须重复多次。在这种情况下，我们可以使用**循环**。在 Python 中，有两种类型的循环，即`for`循环和`while`循环。让我们通过具体的例子来回顾它们。

# 一个 for 循环

在 Python 中，`for`循环用于执行*n*次任务。`for`循环会迭代序列的每个元素。这个序列可以是字典、列表或任何其他迭代器。例如，让我们讨论一个执行循环的例子：

```py
    for i in range(0, 10): 
       print("Loop execution no: ", i)
```

在前面的例子中，`print`语句被执行了 10 次：

![](img/41fa73db-9be7-4df4-9435-1c1f94d118e8.png)

为了执行`print`任务 10 次，使用了`range()`函数（[`docs.python.org/2/library/functions.html#range`](https://docs.python.org/2/library/functions.html#range)）。`range`函数会为传递给函数的起始和停止值生成一个数字列表。在这种情况下，`0`和`10`被作为参数传递给`range()`函数。这将返回一个包含从`0`到`9`的数字的列表。`for`循环会按照步长为 1 的步骤迭代每个元素的代码块。`range`函数也可以按照步长为 2 生成一个数字列表。这是通过将起始值、停止值和步长值作为参数传递给`range()`函数来实现的：

```py
    for i in range(0, 20, 2): 
       print("Loop execution no: ", i)
```

在这个例子中，`0`是起始值，`20`是停止值，`2`是步长值。这会生成一个 10 个数字的列表，步长为 2：

![](img/78b7d6da-e498-4a0c-a913-847895e24fef.png)

`range`函数可以用来从给定的数字倒数。比如，我们想要从`10`倒数到`1`：

```py
    for i in range(10, 0, -1): 
       print("Count down no: ", i)
```

输出将会是这样的：

![](img/1f72610e-b6f2-4180-b0d6-ffab6e2316e9.png)

`range`函数的一般语法是`range(start, stop, step_count)`。它会生成一个从`start`到`n-1`的数字序列，其中`n`是停止值。

# 缩进

注意`for`循环块中的*缩进*：

```py
    for i in range(10, 1, -1): 
       print("Count down no: ", i)
```

Python 执行`for`循环语句下的代码块。这是 Python 编程语言的一个特性。只要缩进级别相同，它就会执行`for`循环下的任何代码块：

```py
    for i in range(0,10): 
       #start of block 
       print("Hello") 
       #end of block
```

缩进有以下两个用途：

+   它使代码可读性更强

+   它帮助我们识别要在循环中执行的代码块

在 Python 中，要注意缩进，因为它直接影响代码的执行方式。

# 嵌套循环

在 Python 中，可以实现*循环内的循环*。例如，假设我们需要打印地图的`x`和`y`坐标。我们可以使用嵌套循环来实现这个：

```py
for x in range(0,3): 
   for y in range(0,3): 
         print(x,y)
```

预期输出是：

![](img/19048a40-2d7b-4ed3-886b-87572e8d01bc.png)

在嵌套循环中要小心代码缩进，因为它可能会引发错误。考虑以下例子：

```py
for x in range(0,10): 
   for y in range(0,10): 
   print(x,y)
```

Python 解释器会抛出以下错误：

```py
    SyntaxError: expected an indented block
```

这在以下截图中可见：

![](img/da5977ea-4fc5-4952-b462-2b596dc2ba32.png)

因此，在 Python 中要注意缩进是很重要的（特别是嵌套循环），以成功执行代码。IDLE 的文本编辑器会在你编写代码时自动缩进。这应该有助于理解 Python 中的缩进。

# 一个 while 循环

当特定任务需要执行直到满足特定条件时，会使用`while`循环。`while`循环通常用于执行无限循环中的代码。让我们看一个具体的例子，我们想要打印`i`的值从`0`到`9`：

```py
i=0 
while i<10: 
  print("The value of i is ",i) 
  i+=1
```

在`while`循环内，我们每次迭代都会将`i`增加`1`。`i`的值增加如下：

```py
i += 1
```

这等同于`i = i+1`。

这个例子会执行代码，直到`i`的值小于 10。也可以执行无限循环中的某些操作：

```py
i=0 
while True: 
  print("The value of i is ",i) 
  i+=1
```

可以通过在键盘上按下*Ctrl* + *C*来停止这个无限循环的执行。

也可以有嵌套的`while`循环：

```py
i=0 
j=0 
while i<10: 
  while j<10: 
    print("The value of i,j is ",i,",",j) 
    i+=1 
    j+=1
```

与`for`循环类似，`while`循环也依赖于缩进的代码块来执行一段代码。

Python 可以打印字符串和整数的组合，只要它们作为`print`函数的参数呈现，并用逗号分隔。在前面提到的示例中，`i，j 的值是`，`i`是`print`函数的参数。您将在下一章中了解更多关于函数和参数的内容。此功能使得格式化输出字符串以满足我们的需求成为可能。

# 树莓派的 GPIO

树莓派 Zero 配备了一个 40 针的 GPIO 引脚标头。在这 40 个引脚中，我们可以使用 26 个引脚来读取输入（来自传感器）或控制输出。其他引脚是电源引脚（**5V**，**3.3V**和**Ground**引脚）：

![](img/5412df99-2745-4e1b-8af8-6508c1d944b1.png)树莓派 Zero GPIO 映射（来源：https://www.raspberrypi.org/documentation/usage/gpio-plus-and-raspi2/README.md）

我们可以使用树莓派的 GPIO 最多 26 个引脚来接口设备并控制它们。但是，有一些引脚具有替代功能。

较早的图像显示了树莓派的 GPIO 引脚的映射。圆圈中的数字对应于树莓派处理器上的引脚编号。例如，GPIO 引脚**2**（底部行左侧的第二个引脚）对应于树莓派处理器上的 GPIO 引脚**2**，而不是 GPIO 引脚标头上的物理引脚位置。

一开始，尝试理解引脚映射可能会令人困惑。保留 GPIO 引脚手册（可与本章一起下载）以供参考。需要一些时间来适应树莓派 Zero 的 GPIO 引脚映射。

树莓派 Zero 的 GPIO 引脚是 3.3V 兼容的，也就是说，如果将大于 3.3V 的电压应用到引脚上，可能会永久损坏引脚。当设置为*高*时，引脚被设置为 3.3V，当引脚被设置为低时，电压为 0V。

# 闪烁灯

让我们讨论一个例子，我们将使用树莓派 Zero 的 GPIO。我们将把 LED 接口到树莓派 Zero，并使其以 1 秒的间隔闪烁*开*和*关*。

让我们接线树莓派 Zero 开始：

![](img/40d426e0-5599-48c5-afe6-4b4d4ee64ff6.png)使用 Fritzing 生成的 Blinky 原理图

在前面的原理图中，GPIO 引脚 2 连接到 LED 的阳极（最长的腿）。LED 的阴极连接到树莓派 Zero 的地引脚。还使用了 330 欧姆的限流电阻来限制电流的流动。

![](img/)）。**Raspbian Jessie**操作系统映像带有预安装的库。这是一个非常简单易用的库，对于初学者来说是最好的选择。它支持一套标准设备，帮助我们轻松入门。

例如，为了接口 LED，我们需要从`gpiozero`库中导入`LED`类：

```py
from gpiozero import LED
```

我们将在 1 秒的间隔内打开和关闭 LED。为了做到这一点，我们将*导入*`time`库。在 Python 中，我们需要导入一个库来使用它。由于我们将 LED 接口到 GPIO 引脚 2，让我们在我们的代码中提到这一点：

```py
import time 

led = LED(2)
```

我们刚刚创建了一个名为`led`的变量，并定义我们将在`LED`类中使用 GPIO 引脚 2。让我们使用`while`循环来打开和关闭 LED，间隔为 1 秒。

`gpiozero`库的 LED 类带有名为`on()`和`off()`的函数，分别将 GPIO 引脚 2 设置为高电平和低电平：

```py
while True: 
    led.on() 
    time.sleep(1) 
    led.off() 
    time.sleep(1)
```

在 Python 的时间库中，有一个`sleep`函数，可以在打开/关闭 LED 之间引入 1 秒的延迟。这在一个无限循环中执行！我们刚刚使用树莓派 Zero 构建了一个实际的例子。

将所有代码放在名为`blinky.py`的文件中（可与本书一起下载），从命令行终端运行代码（或者，您也可以使用 IDLE3）：

```py
    python3 blinky.py
```

# GPIO 控制的应用

现在我们已经实施了我们的第一个示例，让我们讨论一些能够控制 GPIO 的可能应用。我们可以使用树莓派的 GPIO 来控制家中的灯光。我们将使用相同的示例来控制台灯！

有一个名为**PowerSwitch Tail II**的产品（[`www.powerswitchtail.com/Pages/default.aspx`](http://www.powerswitchtail.com/Pages/default.aspx)），可以将交流家电（如台灯）与树莓派连接起来。PowerSwitch Tail 配有控制引脚（可以接收 3.3V 高电平信号），可用于打开/关闭灯。开关配有必要的电路/保护，可直接与树莓派 Zero 接口：

树莓派 Zero 与 PowerSwitch Tail II 接口

让我们从上一节中使用相同的示例，将 GPIO 引脚 2 连接到 PowerSwitch Tail 的**+in**引脚。让我们将树莓派 Zero 的 GPIO 引脚的地线连接到 PowerSwitch Tail 的**-in**引脚。PowerSwitch Tail 应连接到交流电源。灯应连接到开关的交流输出。如果我们使用相同的代码并将灯连接到 PowerSwitch Tail，我们应该能够以 1 秒的间隔打开/关闭。

![](img/d0909ca2-bc35-4e48-b8a4-70ac53ca9b2f.png)连接到树莓派 Zero 的 PowerSwitch Tail II 使用 LED 闪烁代码进行家电控制只是一个例子。不建议在如此短的时间间隔内打开/关闭台灯。

# 总结

在本章中，我们回顾了 Python 中的整数、布尔和字符串数据类型，以及算术运算和逻辑运算符。我们还讨论了接受用户输入和循环。我们介绍了树莓派 Zero 的 GPIO，并讨论了 LED 闪烁示例。我们使用相同的示例来控制台灯！

您听说过名为*Slack*的聊天应用程序吗？您是否尝试过在工作时从笔记本电脑控制家里的台灯？如果这引起了您的兴趣，请在接下来的几章中与我们一起工作。
