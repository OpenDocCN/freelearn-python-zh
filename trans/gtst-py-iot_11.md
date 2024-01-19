# 第十一章：条件语句、函数和列表

在本章中，我们将在前一章学到的基础上进行扩展。您将学习有关条件语句以及如何使用逻辑运算符来检查条件的使用。接下来，您将学习如何在 Python 中编写简单的函数，并讨论如何使用触摸开关（瞬时按键）将输入接口到树莓派的 GPIO 引脚。我们还将讨论使用树莓派 Zero 进行电机控制（这是最终项目的预演），并使用开关输入来控制电机。让我们开始吧！

在本章中，我们将讨论以下主题：

+   Python 中的条件语句

+   使用条件输入根据 GPIO 引脚状态采取行动

+   使用条件语句跳出循环

+   Python 中的函数

+   GPIO 回调函数

+   Python 中的电机控制

# 条件语句

在 Python 中，条件语句用于确定特定条件是否满足，通过测试条件是`true`还是`false`。条件语句用于确定程序的执行方式。例如，条件语句可以用于确定是否是开灯的时间。语法如下：

```py
if condition_is_true:

  do_something()
```

通常使用逻辑运算符来测试条件，并执行缩进块下的任务集。让我们考虑一个例子，`check_address_if_statement.py`（可在本章下载）中，程序需要使用`yes`或`no`问题来验证用户输入：

```py
check_address = input("Is your address correct(yes/no)? ") 
if check_address == "yes": 
  print("Thanks. Your address has been saved") 
if check_address == "no": 
  del(address) 
  print("Your address has been deleted. Try again")
```

在这个例子中，程序期望输入`yes`或`no`。如果用户提供了输入`yes`，条件`if check_address == "yes"`为`true`，则在屏幕上打印消息`Your address has been saved`。

同样，如果用户输入是`no`，程序将执行在逻辑测试条件`if check_address == "no"`下的缩进代码块，并删除变量`address`。

# if-else 语句

在前面的例子中，我们使用`if`语句测试每个条件。在 Python 中，还有一种名为`if-else`语句的替代选项。`if-else`语句使得在主条件不为`true`时测试替代条件成为可能：

```py
check_address = input("Is your address correct(yes/no)? ") 
if check_address == "yes": 
  print("Thanks. Your address has been saved") 
else: 
  del(address) 
  print("Your address has been deleted. Try again")
```

在这个例子中，如果用户输入是`yes`，则在`if`下的缩进代码块将被执行。否则，将执行`else`下的代码块。

# if-elif-else 语句

在前面的例子中，对于除`yes`之外的任何用户输入，程序执行`else`块下的任何代码。也就是说，如果用户按下回车键而没有提供任何输入，或者提供了`no`而不是`no`，则`if-elif-else`语句的工作如下：

```py
check_address = input("Is your address correct(yes/no)? ") 
if check_address == "yes": 
  print("Thanks. Your address has been saved") 
elif check_address == "no": 
  del(address) 
  print("Your address has been deleted. Try again") 
else: 
  print("Invalid input. Try again")
```

如果用户输入是`yes`，则在`if`语句下的缩进代码块将被执行。如果用户输入是`no`，则在`elif`（*else-if*）下的缩进代码块将被执行。如果用户输入是其他内容，则程序打印消息：`Invalid input. Try again`。

重要的是要注意，代码块的缩进决定了在满足特定条件时需要执行的代码块。我们建议修改条件语句块的缩进，并找出程序执行的结果。这将有助于理解 Python 中缩进的重要性。

到目前为止，我们讨论的三个例子中，可以注意到`if`语句不需要由`else`语句补充。`else`和`elif`语句需要有一个前置的`if`语句，否则程序执行将导致错误。

# 跳出循环

条件语句可以用于跳出循环执行（`for`循环和`while`循环）。当满足特定条件时，可以使用`if`语句来跳出循环：

```py
i = 0 
while True: 
  print("The value of i is ", i) 
  i += 1 
  if i > 100: 
    break
```

在前面的例子中，`while`循环在一个无限循环中执行。`i`的值递增并打印在屏幕上。当`i`的值大于`100`时，程序会跳出`while`循环，并且`i`的值从 1 打印到 100。

# 条件语句的应用：使用 GPIO 执行任务

在上一章中，我们讨论了将输出接口到树莓派的 GPIO。让我们讨论一个简单的按键按下的例子。通过读取 GPIO 引脚状态来检测按钮按下。我们将使用条件语句来根据 GPIO 引脚状态执行任务。

让我们将一个按钮连接到树莓派的 GPIO。你需要准备一个按钮、上拉电阻和几根跳线。稍后给出的图示展示了如何将按键连接到树莓派 Zero。按键的一个端子连接到树莓派 Zero 的 GPIO 引脚的地线。

按键接口的原理图如下：

![](img/9ce1761f-6261-40b0-b7c4-bb387a5e106a.png)树莓派 GPIO 原理图

按键的另一个端子通过 10K 电阻上拉到 3.3V。按键端子和 10K 电阻的交点连接到 GPIO 引脚 2（参考前一章中分享的 BCM GPIO 引脚图）。

![](img/ab6a4c5d-c0e8-4956-a703-5ade988dbee4.png)将按键接口到树莓派 Zero 的 GPIO - 使用 Fritzing 生成的图像

让我们回顾一下需要查看按钮状态的代码。我们利用循环和条件语句来使用树莓派 Zero 读取按钮输入。

我们将使用在上一章介绍的`gpiozero`库。本节的代码示例是`GPIO_button_test.py`，可与本章一起下载。

在后面的章节中，我们将讨论**面向对象编程**（**OOP**）。现在，让我们简要讨论类的概念。在 Python 中，**类**是一个包含定义对象的所有属性的蓝图。例如，`gpiozero`库的`Button`类包含了将按钮接口到树莓派 Zero 的 GPIO 接口所需的所有属性。这些属性包括按钮状态和检查按钮状态所需的函数等。为了接口一个按钮并读取其状态，我们需要使用这个蓝图。创建这个蓝图的副本的过程称为实例化。

让我们开始导入`gpiozero`库，并实例化`gpiozero`库的`Button`类（我们将在后面的章节中讨论 Python 的类、对象及其属性）。按钮接口到 GPIO 引脚 2。我们需要在实例化时传递引脚号作为参数：

```py
from gpiozero import Button 

#button is interfaced to GPIO 2 
button = Button(2)
```

`gpiozero`库的文档可在[`gpiozero.readthedocs.io/en/v1.2.0/api_input.html`](http://gpiozero.readthedocs.io/en/v1.2.0/api_input.html)找到。根据文档，`Button`类中有一个名为`is_pressed`的变量，可以使用条件语句进行测试，以确定按钮是否被按下：

```py
if button.is_pressed: 
    print("Button pressed")
```

每当按下按钮时，屏幕上会打印出消息`Button pressed`。让我们将这段代码片段放在一个无限循环中：

```py
from gpiozero import Button 

#button is interfaced to GPIO 2 
button = Button(2)

while True: 
  if button.is_pressed: 
    print("Button pressed")
```

在无限的`while`循环中，程序不断检查按钮是否被按下，并在按钮被按下时打印消息。一旦按钮被释放，它就会回到检查按钮是否被按下的状态。

# 通过计算按钮按下次数来中断循环

让我们再看一个例子，我们想要计算按钮按下的次数，并在按钮接收到预定数量的按下时中断无限循环：

```py
i = 0 
while True: 
  if button.is_pressed: 
    button.wait_for_release() 
    i += 1 
    print("Button pressed") 

  if i >= 10: 
    break
```

前面的例子可与本章一起下载，文件名为`GPIO_button_loop_break.py`。

在这个例子中，程序检查`is_pressed`变量的状态。在接收到按钮按下时，程序可以使用`wait_for_release`方法暂停，直到按钮被释放。当按钮被释放时，用于存储按下次数的变量会增加一次。

当按钮接收到 10 次按下时，程序会跳出无限循环。

![](img/add8b086-b74e-44a0-909a-a8a30662357b.png)连接到树莓派 Zero GPIO 引脚 2 的红色瞬时按钮

# Python 中的函数

我们简要讨论了 Python 中的函数。函数执行一组预定义的任务。`print`是 Python 中函数的一个例子。它可以将一些东西打印到屏幕上。让我们讨论在 Python 中编写我们自己的函数。

可以使用`def`关键字在 Python 中声明函数。函数可以定义如下：

```py
def my_func(): 
   print("This is a simple function")
```

在这个函数`my_func`中，`print`语句是在一个缩进的代码块下编写的。在函数定义下缩进的任何代码块在代码执行期间调用函数时执行。函数可以被执行为`my_func()`。

# 向函数传递参数：

函数总是用括号定义的。括号用于向函数传递任何必要的参数。参数是执行函数所需的参数。在前面的例子中，没有向函数传递参数。

让我们回顾一个例子，我们向函数传递一个参数：

```py
def add_function(a, b): 
  c = a + b 
  print("The sum of a and b is ", c)
```

在这个例子中，`a`和`b`是函数的参数。函数将`a`和`b`相加，并在屏幕上打印总和。当通过传递参数`3`和`2`调用函数`add_function`时，`add_function(3,2)`，其中`a`为`3`，`b`为`2`。

因此，执行函数需要参数`a`和`b`，或者在没有参数的情况下调用函数会导致错误。可以通过为参数设置默认值来避免与缺少参数相关的错误：

```py
def add_function(a=0, b=0): 
  c = a + b 
  print("The sum of a and b is ", c)
```

前面的函数需要两个参数。如果我们只向这个函数传递一个参数，另一个参数默认为零。例如，`add_function(a=3)`，`b`默认为`0`，或者`add_function(b=2)`，`a`默认为`0`。当在调用函数时未提供参数时，它默认为零（在函数中声明）。

同样，`print`函数打印传递的任何变量。如果调用`print`函数时没有传递任何参数，则会打印一个空行。

# 从函数返回值

函数可以执行一组定义的操作，并最终在结束时返回一个值。让我们考虑以下例子：

```py
def square(a): 
   return a**2
```

在这个例子中，函数返回参数的平方。在 Python 中，`return`关键字用于在执行完成后返回请求的值。

# 函数中变量的作用域

Python 程序中有两种类型的变量：局部变量和全局变量。**局部变量**是函数内部的变量，即在函数内部声明的变量只能在该函数内部访问。例子如下：

```py
def add_function(): 
  a = 3 
  b = 2 
  c = a + b 
  print("The sum of a and b is ", c)
```

在这个例子中，变量`a`和`b`是函数`add_function`的局部变量。让我们考虑一个**全局变量**的例子：

```py
a = 3 
b = 2 
def add_function(): 
  c = a + b 
  print("The sum of a and b is ", c) 

add_function()
```

在这种情况下，变量`a`和`b`在 Python 脚本的主体中声明。它们可以在整个程序中访问。现在，让我们考虑这个例子：

```py
a = 3 
def my_function(): 
  a = 5 
  print("The value of a is ", a)

my_function() 
print("The value of a is ", a)
```

程序输出为：

```py
      The value of a is

      5

      The value of a is

      3
```

在这种情况下，当调用`my_function`时，`a`的值为`5`，在脚本主体的`print`语句中`a`的值为`3`。在 Python 中，不可能在函数内部显式修改全局变量的值。为了修改全局变量的值，我们需要使用`global`关键字：

```py
a = 3 
def my_function(): 
  global a 
  a = 5 
  print("The value of a is ", a)

my_function() 
print("The value of a is ", a)
```

一般来说，不建议在函数内修改变量，因为这不是一个很安全的修改变量的做法。最佳做法是将变量作为参数传递并返回修改后的值。考虑以下例子：

```py
a = 3 
def my_function(a): 
  a = 5 
  print("The value of a is ", a) 
  return a 

a = my_function(a) 
print("The value of a is ", a)
```

在上述程序中，`a`的值为`3`。它作为参数传递给`my_function`。函数返回`5`，保存到`a`中。我们能够安全地修改`a`的值。

# GPIO 回调函数

让我们回顾一下在 GPIO 示例中使用函数的一些用途。函数可以用来处理与树莓派的 GPIO 引脚相关的特定事件。例如，`gpiozero`库提供了在按钮按下或释放时调用函数的能力：

```py
from gpiozero import Button 

def button_pressed(): 
  print("button pressed")

def button_released(): 
  print("button released")

#button is interfaced to GPIO 2 
button = Button(2) 
button.when_pressed = button_pressed 
button.when_released = button_released

while True: 
  pass
```

在这个例子中，我们使用库的 GPIO 类的`when_pressed`和`when_released`属性。当按钮被按下时，执行函数`button_pressed`。同样，当按钮被释放时，执行函数`button_released`。我们使用`while`循环来避免退出程序并继续监听按钮事件。使用`pass`关键字来避免错误，当执行`pass`关键字时什么也不会发生。

能够为不同事件执行不同函数的能力在*家庭自动化*等应用中非常有用。例如，可以用来在天黑时打开灯，反之亦然。

# Python 中的直流电机控制

在本节中，我们将讨论使用树莓派 Zero 进行电机控制。为什么要讨论电机控制？随着我们在本书中不同主题的进展，我们将最终构建一个移动机器人。因此，我们需要讨论使用 Python 编写代码来控制树莓派上的电机。

为了控制电机，我们需要一个**H 桥电机驱动器**（讨论 H 桥超出了我们的范围。有几种资源可供 H 桥电机驱动器使用：[`www.mcmanis.com/chuck/robotics/tutorial/h-bridge/`](http://www.mcmanis.com/chuck/robotics/tutorial/h-bridge/)）。有几种专为树莓派设计的电机驱动器套件。在本节中，我们将使用以下套件：[`www.pololu.com/product/2753`](https://www.pololu.com/product/2753)。

**Pololu**产品页面还提供了如何连接电机的说明。让我们开始编写一些 Python 代码来操作电机：

```py
from gpiozero import Motor 
from gpiozero import OutputDevice 
import time

motor_1_direction = OutputDevice(13) 
motor_2_direction = OutputDevice(12)

motor = Motor(5, 6)

motor_1_direction.on() 
motor_2_direction.on()

motor.forward()

time.sleep(10)

motor.stop()

motor_1_direction.off() 
motor_2_direction.off()
```

树莓派基于电机控制

为了控制电机，让我们声明引脚、电机的速度引脚和方向引脚。根据电机驱动器的文档，电机分别由 GPIO 引脚 12、13 和 5、6 控制。

```py
from gpiozero import Motor 
from gpiozero import OutputDevice 
import time 

motor_1_direction = OutputDevice(13) 
motor_2_direction = OutputDevice(12) 

motor = Motor(5, 6)
```

控制电机就像使用`on()`方法打开电机，使用`forward()`方法向前移动电机一样简单：

```py
motor.forward()
```

同样，通过调用`reverse()`方法可以改变电机方向。通过以下方式可以停止电机：

```py
motor.stop()
```

# 读者的一些迷你项目挑战

以下是一些迷你项目挑战给我们的读者：

+   在本章中，我们讨论了树莓派的输入接口和电机控制。想象一个项目，我们可以驱动一个移动机器人，该机器人从触须开关读取输入并操作移动机器人。结合限位开关和电机，是否可能构建一个沿墙行驶的机器人？

+   在本章中，我们讨论了如何控制直流电机。我们如何使用树莓派控制步进电机？

+   如何使用树莓派 Zero 接口运动传感器来控制家里的灯？

# 总结

在本章中，我们讨论了条件语句以及条件语句在 Python 中的应用。我们还讨论了 Python 中的函数，将参数传递给函数，从函数返回值以及 Python 程序中变量的作用域。我们讨论了回调函数和 Python 中的电机控制。
