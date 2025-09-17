# *附录 A*：如何将伪代码转换为真实代码

本书中的代码示例大部分都是使用伪代码编写的，因为本书的目的是让你了解什么是编程，而不是专注于任何特定的语言。

要能够编写代码，你需要使用一种真正的语言，在这里我们将探讨一些更流行的语言，并看看这本书中使用的代码如何翻译成这些语言。

我们将查看的语言如下：

+   C++

+   C#

+   Java

+   JavaScript

+   PHP

+   Python

对于每种语言，我们将从一个简短的介绍开始。

你不能仅仅从这些简短的示例中开始编写你自己的程序，但你将感受到这些语言，也许以这种方式看到它们将帮助你决定你想先学习哪种语言。

在我们查看不同的语言之前，我们将有几个伪代码示例。然后，这些示例将被翻译成前面的六种语言。所以，让我们开始吧！

# 伪代码示例

在本节中，我们将查看一些伪代码的代码示例。

## 伪代码中的 Hello World

第一个示例将是一个简短的程序，它只是将**Hello, World!**打印到屏幕上。

在我们的伪代码中，它将看起来像这样：

```py
print "Hello, World!"
```

## 伪代码中的变量声明

在这个例子中，我们将创建几个变量。第一个将存储一个整数。第二个将存储第一个变量的值，但将其转换为字符串：

```py
my_int_value = 10
my_string_value = string(my_int_value)
```

## 伪代码中的 for 循环

在这个例子中，我们将有一个`for`循环，它迭代 10 次并打印值`0`到`9`：

```py
for i = 0 to 10
    print i
end_for
```

## 伪代码中的函数

在这个例子中，我们将创建一个小函数，该函数将接受三个整数作为参数。然后，该函数应该返回它们中的最大值。我们还将调用该函数并显示结果。

在函数中，我们首先检查第一个参数是否大于另外两个参数。如果是，我们就找到了最大值，并返回它。

由于我们一旦找到最大值就立即返回，因此在这个程序中我们不需要使用任何`else`语句，因为返回会立即退出函数。

因此，我们只需要将第二个参数与第三个参数进行比较。如果第二个参数大于第三个参数，我们就返回它；否则，我们将返回第三个参数，因为它必须是最大的值。这可以通过以下代码展示：

```py
function max_of_three(first, second, third)
   if first > second and first > third then
        return first
    end_if
    if second > third then
         return second
    end_if
    return third
end_function
maximum = max_of_three(34, 56, 14)
print maximum
```

## while 循环、伪代码中的用户输入、if 语句和 for 循环

在这个例子中，我们将同时说明几个概念。

此程序将要求用户输入数字，数量不限。他们可以通过输入一个负数来停止输入新值。所有值（除了最后的负数）都将存储在一个动态数组中。

在程序退出之前，我们将使用以下代码块打印出我们存储的所有值：

```py
values = [] 
inputValue = 0 
while inputValue >= 0
    print "Enter a number: "
    input inputValue 
    if inputValue >= 0
        values.add(inputValue)
    end_if
end_while
```

从前面的代码中，我们可以看到：

1.  首先，我们创建一个动态数组。记住，这是一个在程序执行期间可以添加和删除值的列表；也就是说，它不是一个固定大小的数组，我们需要定义要存储其中的项目数量：

1.  然后，我们将进入一个`while`循环，并在其中要求用户输入一个数字。

1.  我们将把输入的数字添加到动态数组中，并且会一直这样做，直到用户输入一个负数。这个负数不应该添加到数组中，而应该作为用户完成输入数字的指示，这样我们就可以退出循环。

# C++

C++是由丹麦计算机科学家 Bjarne Stroustrup 开发的，他最初将其称为 C with Classes。这项工作始于 1979 年，他希望创建一种语言，它具有 C 编程语言的力量以及他在为博士论文编程时接触到的面向对象特性。

1982 年，他将语言重命名为 C++，其中两个加号运算符是对 C 中的++运算符的引用，该运算符将变量增加一。这种想法是 C++是 C 加上一个特性，而这个特性就是面向对象。

该语言的第一版商业发布是在 1985 年。

C++是一种通用编译型编程语言，常用于需要高执行速度的情况，程序员可以控制数据在计算机内存中的存储和管理。

下面是一些关于它的快速事实：

+   **名称**：C++

+   **设计者**：Bjarne Stroustrup

+   **首次公开发布**：1985

+   **范式**：多范式、过程式、函数式、面向对象、泛型

+   **类型**：静态

+   `.cpp`, `.h`

## C++中的“Hello world”

所有用 C++编写的应用程序都需要有一个名为`main`的函数，该函数将作为程序执行的开始点。

输出是通过使用所谓的输出流显示到控制台窗口的。该语言提供了一个来自`ostream`类的现成对象用于此目的，称为`cout`。该语言还提供了一个函数（这种类型的函数在 C++中被称为操纵函数），称为`endl`，它将在输出流中添加一个换行符。数据是通过使用`<<`运算符发送到输出流的。

`cout`和`endl`前面的`std::`部分表示这两个是在语言的标准命名空间中定义的。

由于 C++中的`main`函数应该返回一个整数值，表示执行的结果，所以我们返回`0`，这是表示成功的值。

注意，C++中所有非复合语句都以分号结尾，如下所示：

```py
#include <iostream>
int main()
{
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

## C++中的变量声明

由于 C++是一种静态类型语言，我们必须指定变量可以使用的数据类型。之后，这将是这个变量唯一可以处理的数据类型。

C++中的字符串是在一个类中定义的，为了能够使用这个类，我们必须包含`string`，就像我们在第一行所做的那样。

在`main`函数内部，我们首先声明我们的整数变量。我们指定类型为整数，使用`int`。

然后，我们希望将我们的整数转换为字符串。我们可以通过一个名为`to_string`的函数来完成这个任务。它定义在标准命名空间中，并且必须用`std::`前缀。

当声明`string`变量的类型时，我们必须同时声明`string`类位于标准命名空间中：

```py
#include <string>
int main()
{
    int my_int_value = 10;
    std::string my_string_value = std::to_string(my_int_value);
    return 0;
}
```

如果我们想简化这个程序并让编译器确定变量的类型，我们可以这样做。`auto`关键字将帮助我们完成这个任务。由于我们在创建变量时为其赋值，所以它们的类型将与我们分配给它们的数据相同。请参考以下代码：

```py
#include <string>
int main()
{
    auto my_int_value = 10;
    auto my_string_value = std::to_string(my_int_value);
    return 0;
}
```

## C++中的`for`循环

C++使用 C 风格的`for`循环。它有三个部分，由分号分隔，如下所示：

```py
#include <iostream>
int main()
{
    for(int i = 0; i < 10; i++) {
        std::cout << i << std::endl;
    }
}
```

从前面的代码中，我们可以看到以下内容：

+   第一部分将初始化循环变量为其起始值；在我们的例子中，那将是`0`。

+   下一节将告诉我们`for`循环将运行多长时间的条件；在我们的例子中，这意味着只要变量小于 10。

+   最后的部分是变量在每次迭代中如何变化。我们在这里使用`++`运算符，以便变量每次迭代增加一。

在循环内部，我们将打印循环变量的值。

## C++中的函数

C++中的函数必须首先声明其返回类型——也就是说，函数返回什么数据类型。我们还必须指定每个参数的类型。在我们的例子中，我们将传递三个整数，因为函数将返回其中的一个，所以返回类型也将是整数。

注意，在 C++中，`&&`符号表示`and`：

```py
#include <iostream>
int max_of_three(int first, int second, int third) 
{
    if (first > second && first > third) {
        return first;
    }
    if (second > third) {
        return second;
    }
    return third;
}
int main()
{
    int maximum = max_of_three(34, 56, 14);
    std::cout << maximum << std::endl;
}
```

## C++中的 while 循环、用户输入、if 语句和 foreach 循环

我们需要使用动态数据结构，这样我们就可以在程序运行时添加尽可能多的值。在 C++中，我们有这样一个选项，就是使用一个名为`vector`的类。这个类被创建成可以存储任何类型的数据，这就是为什么我们在声明中在`<`和`>`之间有`int`。让我们看看它是如何工作的：

1.  就像许多其他事情一样，`vector`类需要用`std::`指定为属于标准命名空间。

1.  接下来，我们声明一个整数变量，它将接受输入。我们目前将其设置为`0`。当我们进入`while`循环时，我们需要这个值在下一行。当循环迭代时，只要`input_value`等于或大于`0`，我们必须将其设置在该范围内的一个值。

1.  在循环内部，我们向用户打印一条消息，说明我们需要一个值。要从用户那里获取输入，我们使用`cin`，它的工作方式有点像`cout`，但方向相反。它不是将事物发送到屏幕，而是从键盘接受事物。通常，当我们谈论`cout`和`cin`时，我们不会说输出会显示在屏幕上，输入来自键盘，因为这些可以重新映射为其他事物，如文件。相反，我们说`cout`发送到标准输出，通常是屏幕，而`cin`从标准输入读取，通常是键盘。

1.  当我们获得输入时，我们会检查它是否为`0`或正值。这是我们想要存储在我们向量中的唯一值。如果是的话，我们就在我们的向量上使用一个名为`push_back`的方法，它将当前值插入到向量的末尾。

1.  这将继续，直到用户输入一个负值。然后，我们退出`while`循环，进入 C++中称为`for`循环的东西。它类似于`foreach`循环，因为它将遍历我们在向量中的所有项目。当前项将被存储在变量 value 中，并在循环内部打印它。它的代码如下：

    ```py
    #include <iostream>
    #include <vector>
    int main()
    {
        std::vector<int> values;
        int input_value = 0;
        while (input_value >= 0) {
            std::cout << "Enter a number: ";
            std::cin >> input_value;
            if (input_value >= 0) {
                values.push_back(input_value);
            }
        }

        for (auto value : values) {
            std::cout << value << std::endl;
        }
    } 
    ```

# C#

C#，发音类似于同名音乐符号，是由微软开发的一种语言，并于 2000 年作为公司.NET 计划的一部分首次发布。该语言由丹麦软件工程师 Anders Hejlsberg 设计，最初将其命名为**Cool**（代表**C-like Object-Oriented Language**）。由于版权原因，微软在首次正式发布之前将其更名为 C#。

该语言被设计成一种简单、现代且面向对象的编程语言。该语言主要用于微软的.NET 框架中。

注意，C#中所有非复合语句都以分号结尾。

这里有一些快速事实：

+   **名称**：C#

+   **设计者**：Anders Hejlsberg，微软

+   **首次公开发布**：2000

+   **范式**：面向对象、泛型、命令式、结构化、函数式

+   **类型**：静态

+   `.cs`

## C#中的“Hello world”

所有用 C#编写的程序都必须存在于一个类中，并且我们项目中的一个类必须有一个名为`Main`的方法，它将是程序执行的开始点。还应注意的是，所有 C#应用程序都应该存在于一个项目中。

我们首先应该注意的是，在包含`Main`方法头部的行上，我们看到的是`static`关键字。将方法声明为`static`意味着它可以不创建定义在其内的类的对象而执行。简单来说，这意味着`Main`方法可以作为函数执行；这一点我们现在需要知道的就是这些。

`Console`是一个类，它处理 C#控制台应用程序的所有输入和输出。控制台应用程序是一个没有图形用户界面的程序。所有输入和输出都通过控制台或终端窗口进行，仅使用文本。

在`Console`类内部，还有一个名为`WriteLine`的静态方法。在这里我们可以看到，一个`static`方法可以通过类名来调用。这个`WriteLine`方法将输出我们发送到控制台窗口的任何内容。参考以下代码：

```py
using System;
class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Hello World!");
    }
}
```

## C#中的变量声明

由于 C#是一种静态类型语言，我们必须指定一个变量可以使用的数据类型。在那之后，这个变量就只能处理这种数据类型。

我们使用`int`声明`myIntValue`变量为一个整数。

在 C#中，`int`不仅仅是一个基本数据类型，就像在许多其他语言中一样。它是一种称为`struct`的东西。在某种程度上，`struct`与类是相同的东西。这个`struct`将从名为`Object`的类中继承一些东西，这个类定义了一个名为`ToString`的方法，我们可以使用这个方法将整数转换为字符串：

```py
using System;
class Program
{
    static void Main(string[] args)
    {
        int myIntValue = 10;
        string myStringValue = myIntValue.ToString();
    }
}
```

我们可以通过让编译器确定变量的数据类型来简化这个程序。因为我们是在声明它们的同时给它们赋值，编译器将根据这个数据类型创建它们。我们通过`var`关键字来完成这个操作：

```py
using System;
class Program
{
    static void Main(string[] args)
    {
        var myIntValue = 10;
        var myStringValue = myIntValue.ToString();
    }
}
```

## C#中的 for 循环

C#使用 C 风格的`for`循环。它有三个部分，由分号分隔：

+   第一个部分将初始化循环变量为其起始值；在我们的例子中，那将是`0`。

+   下一个部分是告诉`for`循环将运行多长时间的条件；在我们的例子中，那就是变量小于 10\。

+   最后一个部分是变量在每次迭代中如何变化。我们在这里使用`++`运算符，使得变量在每次迭代中增加一。

在循环内部，我们将打印循环变量的值：

```py
using System;
class Program
{
    static void Main(string[] args)
    {
        for(int i = 0; i < 10; i++) 
        {
            System.Console.WriteLine(i);
        }
    }
}
```

## C#中的函数

我们首先应该注意的是，在 C#中，没有函数，因为所有代码都必须定义在一个类中，并且定义在类内部的函数被称为**方法**。尽管如此，它们的行为与普通函数相似。

正如我们在前面的例子中所看到的，如果我们想调用一个方法而不需要这个类的对象，那么这个方法必须被声明为`static`，这是我们声明函数时看到的第一个东西。

在 C#中，我们还必须指定一个方法将返回什么数据类型。这就是为什么在方法名前面有`int`的原因。当我们传入三个整数时，它将返回一个整数，并且它将返回这三个数中的最大值。正如我们所看到的，我们还必须为每个参数指定数据类型。

注意，在 C#中，`&&`符号表示`and`。参考以下代码：

```py
using System;
class Program
{
    static int MaxOfThree(int first, int second, int third)
    {
        if (first > second && first > third) {
            return first;
        }
        if (second > third) {
            return second;
        }
        return third;
    }
    static void Main(string[] args)
    {
        int maximum = MaxOfThree(34, 56, 14);
        System.Console.WriteLine(maximum);
    }
}
```

## C#中的 while 循环、用户输入、if 语句和 foreach 循环

我们需要一个动态数据结构，这样我们就可以在程序运行时添加尽可能多的值。在 C#中，我们有这样一个选项，就是使用一个名为`List`的类：

+   这个类被创建出来，以便一个列表可以持有任何类型的数据，这就是为什么我们在声明中在`<`和`>`之间有`int`。

+   接下来，我们声明一个整数变量，它将接受输入。我们目前将其设置为`0`。当我们进入`while`循环时，我们需要这个值在下一行。由于循环在`inputValue`等于或大于`0`时迭代，我们必须将其设置在该范围内的一个值。

+   在循环内部，我们向用户打印一条消息，表示我们想要一个值。要从用户那里获取输入，我们使用位于`Console`类中的`ReadLine`方法。我们从`ReadLine`获得的是一个字符串。这就是为什么我们使用`Int32.Parse`方法的原因。它将用户输入的任何内容转换为整数。

+   当我们获得输入时，我们检查它是否为`0`或正值。我们只想在我们的列表中存储`0`值。如果是，我们就在我们的列表上使用名为`Add`的方法调用，它将当前值插入列表的末尾。

+   这将继续，直到用户输入一个负值。然后，我们退出`while`循环，进入一个`foreach`循环，该循环将遍历列表中的所有项。

当前项将被存储在名为`value`的变量中，并在循环内部打印它：

```py
using System;
using System.Collections.Generic;
class Program
{
   static void Main(string[] args)
   {
      List<int> values = new List<int>();
      int inputValue = 0;
      while (inputValue >= 0) {
          System.Console.Write("Enter a number: ");
          inputValue = Int32.Parse(System.Console.ReadLine());
          if (inputValue >= 0) {
              values.Add(inputValue);
           }
      }
      foreach(var value in values) {
          System.Console.WriteLine(value);
      }
    }
}
```

# Java

Java 编程语言的工作始于 1991 年，设计目标是创建一个简单、面向对象的、语法对现有程序员熟悉的语言。

James Gosling 是该语言的主要设计者，最初将其命名为 Oak，因为一棵橡树正在他窗户外生长。由于版权原因，后来将其更名为 Java，以纪念 Java 咖啡。

语言设计中的一个基本概念是让程序员一次编写，到处运行，简称*WORA*。这个想法是，用 Java 编写的应用程序可以在大多数平台上运行，无需任何修改或重新编译。

通过让 Java 源代码编译成一个中间表示形式，称为*Java 字节码*，而不是特定平台的机器码，实现了可移植性。然后，由为托管应用程序的硬件编写的虚拟机执行这些字节码。

这里有一些关于它的快速事实：

+   **名称**：Java

+   **设计者**：James Gosling，Sun Microsystems

+   **首次公开发布**：1995 年

+   **范式**：多范式、面向对象、泛型、命令式

+   **类型**：静态

+   `.java`，`.jar`

## Java 中的“Hello World”

Java 要求所有代码都必须在类内部编写，并且所有应用程序都需要一个名为`main`的方法的类。

Java 的一个特点是每个类都必须在一个与类同名的源代码文件中编写。由于这个例子中的类名为`Hello`，它必须保存在一个名为`Hello.java`的文件中。

要将内容打印到控制台窗口，我们将使用`System.out.println`。现在，`System`是一个类，它处理输入和输出等操作。在`System`类内部，定义了一个输出流，称为`out`，这个流有一个名为`println`的方法，它将打印传递给它的数据，并在流的末尾插入一个换行符。

注意，Java 中所有非复合语句都以分号结束：

```py
class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, World!");        
    }
}
```

## Java 中的变量声明

由于 Java 是一种静态类型语言，我们必须指定一个变量可以使用的数据类型。之后，这将是这个变量唯一可以处理的数据类型。

我们首先使用`int`声明我们的整数变量。

所有原始数据类型在 Java 中都有一个类表示。我们可以使用`Integer`类将我们的整数转换为字符串。我们通过调用`Integer`类中的一个静态方法并传递我们想要转换的整数值来实现这一点：

```py
class Variable {
    public static void main(String[] args) {
        int myIntValue = 10;
        String myStringValue = Integer.toString(myIntValue);       
    }
}
```

Java 没有像 C++和 C#中的`auto`和`var`关键字那样的自动类型推断功能。

## Java 中的 for 循环

Java 使用 C 风格的`for`循环。它有三个部分，由分号分隔。第一个部分将循环变量初始化为其起始值；在我们的例子中，那将是 0。下一个部分是条件，它将告诉我们`for`循环将运行多长时间；在我们的例子中，只要变量小于 10。最后一个部分是变量在每次迭代中如何变化。我们在这里使用`++`运算符，所以变量在每次迭代中都会增加 1。

在循环内部，我们将打印循环变量的值：

```py
class For {
    public static void main(String[] args) {
        for(int i = 0; i < 10; i++) {
            System.out.println(i);
        }  
    }
}
```

## Java 中的函数

我们首先应该注意的是，在 Java 中，没有函数，因为所有代码都必须在类中定义，类内声明的函数被称为方法。尽管如此，它们的行为与普通函数类似。

如我们在前面的示例中看到的，如果我们想在没有这个类的对象的情况下调用一个方法，那么这个方法必须被声明为`static`，这是我们声明函数时看到的第一个东西。

在 Java 中，我们还必须指定一个方法将返回什么数据类型。这就是为什么在方法名前面有`int`。它将返回一个整数，因为我们传递了三个整数，并且它将返回这三个数中的最大值。正如我们所看到的，我们必须为每个参数指定数据类型。

注意，在 Java 中，`&&`符号表示`and`：

```py
class Function {
    static int maxOfThree(int first, int second, int third) {
        if (first > second && first > third) {
            return first;
        }
        if (second > third) {
            return second;
        }
        return third;
    }
    public static void main(String[] args) {
        int maximum = maxOfThree(34, 56, 14);
        System.out.println(maximum);
    }
}
```

## Java 中的 while 循环、用户输入、if 语句和 foreach 循环

我们需要使用一个动态数据结构，这样我们就可以在程序运行时添加尽可能多的值。在 Java 中，我们有这样一个选项，就是使用一个名为`ArrayList`的类：

1.  这个类被创建出来，以便列表可以存储任何类型的数据，这就是为什么我们在声明中在`<`和`>`之间有`Integer`。在 Java 中，我们不能使用原始数据类型作为存储在列表中的类型。相反，我们使用`int`的类表示形式，即`Integer`。

1.  接下来，我们声明一个整数变量，它将接受输入。我们目前将其设置为`0`。当我们进入`while`循环时，我们需要这个值。当循环迭代时，只要`inputValue`等于或大于 0，我们必须将其设置在该范围内的一个值。

1.  Java 没有内置的用户输入方法，因此我们需要从名为`BufferedReader`的类中创建一个对象来处理输入。我们称这个对象为`reader`。

1.  在循环内部，我们向用户打印一条消息，表示我们想要一个值。为了从用户那里获取输入，我们使用我们的`reader`对象及其`readLine`方法。我们从`readLine`获取的值是一个字符串。这就是为什么我们使用`Integer.parseInt`方法。它将用户输入的任何内容转换为整数。

1.  当我们获得输入时，我们会检查它是否为`0`或正值。我们只想在我们的列表中存储`0`值。如果是，我们将在我们的列表上使用一个名为`add`的方法，该方法将当前值插入列表的末尾。

1.  Java 将强制我们处理用户输入非数字的情况。如果他们这样做，当我们尝试将字符串转换为数字时，我们会得到一个异常。这就是为什么我们需要带有`catch`语句的`try`块。如果用户输入的不是数字，我们将进入`catch`语句。

1.  这将继续，直到用户输入一个负值。然后，我们退出`while`循环，进入一个`for`循环，该循环将遍历列表中的所有项目。当前的项目将被存储在`value`变量中，在循环内部，我们打印它：

    ```py
    import java.io.BufferedReader;
    import java.io.IOException;
    import java.io.InputStreamReader;
    import java.util.ArrayList;
    class For {
      public static void main(String[] args) {
        ArrayList<Integer> values = new 
          ArrayList<Integer>();
        int inputValue = 0;
        BufferedReader reader = new BufferedReader(new 
                         InputStreamReader(System.in));
        while(inputValue >= 0) {
          System.out.print("Enter a value: ");
          try {
            inputValue = Integer.parseInt(reader.readLine());
            if (inputValue >= 0) {
              values.add(inputValue);
            }
          } catch (NumberFormatException | IOException e) {
            e.printStackTrace();
          }
        }
        for (int value : values) {
          System.out.println(value);
        }
      }
    }
    ```

# JavaScript

在万维网的早期几年，只有一个支持图形用户界面的网络浏览器，即 1993 年发布的 Mosaic。Mosaic 的主要开发者很快成立了 Netscape 公司，并在 1994 年发布了一个更精致的浏览器，名为 Netscape Navigator。

在这些早期年份，网络是一个完全不同的地方，网页只能显示静态内容。Netscape 希望改变这一点，并决定在其 Navigator 中添加一种脚本语言。最初，他们考虑了两种实现这一目标的方法。一种是与 Sun Microsystems 合作并使用 Java 编程语言。另一种选择是让新聘用的布伦丹·艾奇将 Scheme 编程语言嵌入到浏览器中。

这个决定是在两者之间做出的妥协。布伦丹·艾奇被委以创建一种新语言的任务，但其语法应与 Java 紧密相关，而不太像 Scheme。这种语言最初被命名为 LiveScript，这也是它在 1995 年发布的名称。

由于 Java 在当时是一种全新的语言，因此将其名称更改为 JavaScript，以便它能得到更多的关注。这两个语言名称之间的相似性导致了多年来许多混淆，尤其是在不太熟悉编程的人中。

以下是关于 JavaScript 的一些快速事实：

+   **名称**: JavaScript

+   **设计者**: 布伦丹·艾奇

+   **首次公开发布**: 1995 年

+   **范式**: 事件驱动、函数式、命令式

+   **类型**: 动态

+   `.js`

## JavaScript 中的“Hello World”

我们应该注意的第一件事是，JavaScript 被设计为在其程序在网页浏览器中执行。你可以在控制台窗口中运行 JavaScript 应用程序，但要能够做到这一点，我们需要一个可以为我们执行代码的 JavaScript 引擎。其中一个这样的引擎是 Node.js，可以从[`nodejs.org`](https://nodejs.org)免费下载。

JavaScript 是一种脚本语言，因此我们不需要将代码放在任何特定的函数或类中。

在 JavaScript 中，我们可以使用`console`对象来输出数据。它通常用于将数据打印到网页浏览器的调试控制台，但如果我们使用 Node.js 来执行该应用程序，输出将打印到控制台窗口。`console`对象有一个名为`log`的方法，可以输出我们传递给它的任何内容。

注意，JavaScript 中所有非复合语句都以分号结尾：

```py
console.log("Hello, World!");
```

## JavaScript 中的变量声明

JavaScript 没有为整数指定特定的数据类型。相反，它有一个名为`Number`的数据类型，可以处理整数和浮点数。

我们可以通过使用较旧的`var`关键字或较新的`let`来声明变量。

由于 JavaScript 是动态类型的，我们不需要指定变量将使用什么类型。当我们给它赋值时，类型会被自动推断。

可以通过`Number`类中的`toString`方法将数字转换为字符串。由于我们的变量`myIntValue`是`Number`类的一个对象，它具有这样的方法。注意，我们将值`10`传递给`toString`方法。这是我们希望数字所在的基数。我们想要一个十进制数，所以传递`10`。操作如下：

```py
let myIntValue = 10;
let myStringValue = myIntValue.toString(10);
```

## JavaScript 中的 for 循环

JavaScript 使用 C 风格的`for`循环。它有三个部分，由分号分隔：

+   第一个部分将初始化循环变量为其起始值；在我们的例子中，那将是`0`。

+   下一个部分是条件，它将告诉我们`for`循环将运行多长时间；在我们的例子中，只要变量小于`10`。

+   最后的部分是变量在每次迭代中如何变化。在这里我们使用`++`运算符，这样变量在每次迭代中都会增加一。

在循环内部，我们将打印循环变量的值：

```py
for (let i = 0; i < 10; i++) {
  console.log(i);
}
```

## JavaScript 中的函数

由于 JavaScript 是动态类型的，我们不需要指定函数的返回值或参数的数据类型，就像在 C++、C#和 Java 中需要做的那样。

我们使用`function`关键字来定义这是一个函数。

注意，在 JavaScript 中，`&&`符号表示“和”：

```py
function maxOfThree(first, second, third) {
  if (first > second && first > third) {
    return first;
  }
  if (second > third) {
    return second;
  }
  return third;
}
let maximum = maxOfThree(34, 56, 14);
console.log(maximum);
```

## Java 中的 while 循环、用户输入、if 语句和 foreach 循环

首先，我们必须注意，这个例子并不能真正体现 JavaScript 的优势，因为 JavaScript 并不是为了编写这样的应用程序而设计的。这与 JavaScript 被设计为在网页浏览器中运行，而不是作为控制台应用程序有关。

在 JavaScript 中，事情通常是以异步方式完成的。也就是说，程序代码不会像我们在大多数其他语言和情况下所习惯的那样按顺序执行。如果我们尝试以伪代码版本和为所有其他语言编写的版本相同的方式实现这个程序，我们会看到它进入了一个无休止的循环，不断地要求我们输入一个值，一次又一次。

这个程序有些复杂，所以我们就不深入细节了。前几行是为了创建一些处理输入的东西。其核心是一个名为`question`的函数，它将返回一个`promise`对象。`promise`对象是承诺在未来的某个时刻给我们一个值的东西。为了能够使用这个`promise`，它必须从一个函数中调用，并且这个函数必须声明为`async`。这意味着这个函数可以使用`promise`（为了简化事情）。

这个函数没有名字，但正如你所看到的，它被括号包围，并且在最后有两个空括号。这个结构将使这个函数立即执行：

1.  在这个函数内部，我们创建了一个名为`values`的动态数组。我们将它初始化为空，因为我们还没有任何值要存储在其中。

1.  接下来，我们找到我们将用于输入的变量。我们将这个值设置为`0`，这样当我们来到下一行的`while`循环时，我们会进入循环。

1.  在下一行，我们将使用程序顶部看到的所有代码，这些代码处理用户输入。我们说我们`await``question`函数。`await`关键字将允许应用程序去做其他事情，如果需要的话，但当我们得到用户输入的值时，我们会回到这里并继续执行。这是异步调用工作原理的简要描述。这是一个高级话题，所以如果这段代码让你感到困惑，没问题。

1.  如果输入的值大于或等于`0`，我们将这个值推送到数组的末尾。

1.  当用户输入一个负数时，我们将退出`while`循环，并进入一个`for`循环，该循环将迭代数组中的所有项目。`pos`变量将有一个索引值，第一次是`0`，第二次是`1`，以此类推。当我们想要在循环中打印值时，我们可以使用这个值作为数组的索引，这样我们就能在第一次得到第一个值，第二次得到第二个值，依此类推。请参考以下代码：

    ```py
    const readline = require("readline");
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    const question = (q) => {
      return new Promise((res, rej) => {
        rl.question(q, (answer) => {
          res(answer);
        });
      });
    };
    (async () => {
      let values = [];
      let inputValue = 0;
      while (inputValue >= 0) {
        inputValue = await question("Enter a number: ");
        inputValue = parseInt(inputValue);
        if (inputValue >= 0) {
          values.push(inputValue);
        }
      }
      for (let pos in values) {
        console.log(values[pos]);
      }
    })();
    ```

# PHP

在 1994 年，丹麦-加拿大程序员拉斯马斯·勒尔多夫（Rasmus Lerdorf）用 C 语言编写了几个**通用网关接口**（**CGI**）程序。CGI 是一个接口规范，它将允许 Web 服务器执行可以生成动态 Web 内容的程序。勒尔多夫为他的私人网页创建了它，并扩展并添加了处理 Web 表单和数据库通信的功能。他将这个项目命名为**个人主页/表单解释器**，简称**PHP/FI**。

Lerdorf 后来承认他从未打算创建一种新的编程语言，但这个项目获得了自己的生命力，并组建了一个开发团队，1997 年发布了 PHP/FI 2。

该语言主要用于在 Web 服务器上创建动态网页内容。

关于它的快速事实如下：

+   **名称**：PHP

+   **设计者**：Rasmus Lerdorf

+   **首次公开发布**：1995

+   **范式**：命令式、函数式、面向对象、过程式

+   **类型**：动态

+   `.php`

## PHP 中的“Hello World”

PHP 的主要用途是与 Web 服务器一起运行，用 PHP 编写的应用程序通常用于生成动态网页内容。但如果我们从 [`php.net`](https://php.net) 下载 PHP 可执行文件，我们也可以作为独立的控制台应用程序运行 PHP 应用程序：

+   由于 PHP 代码可以与 HTML 代码在同一文档中编写，因此我们编写的所有 PHP 源代码都必须在 `php` 标签内。起始标签是 `<?php`，结束标签是 `?>`。

+   我们使用 `echo` 在控制台窗口中显示我们的消息。您不需要在 `echo` 中使用任何括号，因为它不是一个函数，而是一种语言结构。

    注意，PHP 中所有非复合语句都以分号结束：

    ```py
    <?php
     echo "Hello, World!";
    ?>
    ```

## PHP 中的变量声明

由于 PHP 是动态类型语言，当我们声明变量时，我们不需要提供任何关于使用哪种数据类型的隐式信息。变量类型将自动为我们推导出来，最终类型取决于我们分配给变量的内容。

PHP 从语言 Perl 继承的一个奇特之处在于，所有变量名都必须以美元符号开头。在 Perl 中，不同的符号有不同的意义，但 PHP 只有一个美元符号用于所有类型。

让我们试试这个。我们首先将值 `10` 赋给我们的 `$myIintValue` 变量。

要将这个整数转换为字符串，我们将使用 `strval` 函数并将整数传递给它。这将把此值转换为字符串，如下所示：

```py
<?php
 $myIntValue = 10;
 $myStringValue = strval($myIntValue);
?>
```

## PHP 中的 for 循环

PHP 使用 C 风格的 `for` 循环。它有三个部分，由分号分隔。第一个部分将循环变量初始化为其起始值；在我们的例子中，那将是 `0`。下一个部分是条件，它将告诉我们 `for` 循环将运行多长时间；在我们的例子中，只要变量小于 10。最后一个部分是变量在每次迭代中如何变化。我们在这里使用 `++` 运算符，以便变量在每次迭代中增加一。

在循环内部，我们将打印循环变量的值。

由于 PHP 中的 `echo` 不会提供任何换行符，我们将在每次迭代后在我们的循环变量后附加它。我们可以通过在两个值之间插入一个点来连接循环变量的值和换行符 (`\n`)：

```py
<?php
 for($i = 0; $i < 10; $i++) {
     echo $i . "\n";
 }
?>
```

## PHP 中的函数

由于 PHP 是动态类型，我们不需要为函数的返回值或参数指定任何数据类型，就像在 C++、C# 和 Java 中需要做的那样。

我们使用`function`关键字来定义这是一个函数。

注意，在 PHP 中，`&&`符号表示`and`：

```py
<?php
function maxOfThree($first, $second, $third) {
    if ($first > $second && $first > $third) {
      return $first;
    }
    if ($second > $third) {
      return $second;
    }
    return $third;
}

$maximum = maxOfThree(34, 56, 14);
echo $maximum;

?>
```

## PHP 中的 while 循环、用户输入、if 语句和 foreach 循环

在 PHP 中，我们可以通过使用`array()`来创建动态数组。在 PHP 中，数组不是一个数组，而是一个有序映射，在其他语言中称为字典或关联数组。但在这个应用中，这并不重要：

1.  在创建数组之后，我们声明一个输入变量，它将保存用户输入的值。我们将其设置为`0`，这样当我们来到下一行的`while`循环时，我们将进入循环。

1.  接下来，我们将使用`readline`从用户那里获取一个值。我们可以向`readline`传递一个字符串，该字符串将作为提示打印到屏幕上。这样，我们就不需要单独一行来打印这个消息。

1.  从`readline`获取的值将是一个字符串，因此我们使用`intval`将其转换为整数。

1.  接下来，我们检查该值是否大于或等于`0`。如果是，我们将使用`array_push`函数。这个函数接受两个参数。第一个参数是我们想要推送值的数组，第二个参数是我们想要推送的值。

1.  当用户输入一个负数时，我们将退出`while`循环并进入一个`foreach`循环，该循环将打印用户输入的所有值。如果您将此程序与其他语言编写的程序进行比较，您会发现与 PHP 相比，数组和变量在位置上有所交换。

    在`foreach`循环内部，我们将值打印到控制台：

    ```py
    <?php
     $values = array();
     $inputValue = 0;
     while($inputValue >= 0) {
         $inputValue = intval(readline("Enter a value: "));
         if($inputValue >=  0) {
             array_push($values, $inputValue);
         }
     }
     foreach($values as $value) {
         echo $value . "\n";
     }
    ?>
    ```

# Python

Python 是在 20 世纪 80 年代末由荷兰程序员吉多·范罗苏姆设计的，作为 ABC 语言的继承者。该语言背后的主要设计理念是代码可读性。

在开发语言的过程中，范罗苏姆喜欢英国喜剧团体蒙提·派森，并决定以他们的名字来命名他的新语言。

在过去几年中，该语言的普及率呈指数级增长，现在它被列为最受欢迎的语言之一。

它是一种通用语言，可用于大多数类型的应用。该语言的常见用途包括开发 Web 应用程序和在数据科学中的应用。由于它被认为是最容易学习的编程语言之一，因此它经常被用作入门语言。

关于它的几个快速事实：

+   **名称**：Python

+   **设计者**：吉多·范罗苏姆

+   **首次公开发布**：1990 年

+   **范式**：多范式、函数式、命令式、面向对象、结构化

+   **类型**：动态

+   `.py`

## Python 中的“Hello world”

由于 Python 是一种脚本语言，我们不需要将代码放在任何特殊函数或类中。要向控制台窗口打印消息，我们只需使用`print`函数并将我们想要打印的内容传递给它：

```py
print("Hello, World!")
```

## 在 Python 中声明变量

由于 Python 是一种动态类型语言，我们不需要提供任何关于我们的变量将使用什么类型的信息。当我们给变量赋值时，类型会自动为我们推导出来。

要声明一个整数变量，我们只需给它赋一个整数。

将这个整数转换为字符串，我们可以使用一个名为 `str` 的类，并将整数传递给它。由于 Python 中的一切都是对象，这将返回一个新的字符串对象给我们：

```py
my_int_value = 10
my_string_value = str(my_int_value)
```

## Python 中的 for 循环

当涉及到 `for` 循环时，Python 将与其他所有我们在这里考虑的语言不同。它不实现使用 C 样式格式的 `for` 循环。Python 中的 `for` 循环将遍历某种类型的序列。由于我们没有序列，我们可以使用一个叫做 `range` 的东西。现在，`range` 看起来像是一个函数，但实际上它是一个叫做 `10` 的东西，在第一次迭代时，它将生成值 `0`。在下一个迭代中，生成的值将是 `1`，以此类推，直到 `9`。

还要注意，Python 不使用大括号来表示复合语句，就像我们在 `for` 语句中看到的那样。相反，`for` 循环的内容用四个空格缩进。还要注意，冒号是第一行的最后一个字符。它是表示下一行应该缩进的指示：

```py
for i in range(10):
    print(i)
```

## Python 中的函数

由于 Python 是动态类型的，我们不需要为函数的返回值或参数指定任何数据类型，就像在 C++、C#和 Java 中需要做的那样。

我们使用 `def` 关键字来定义这是一个函数：

```py
def max_of_three(first, second, third):
    if first > second and first > third:
        return first
    if second > third:
        return second
    return third
maximum = max_of_three(34, 56, 14)
print(maximum)
```

## Python 中的 while 循环、用户输入、if 语句和 foreach 循环

在 Python 中，我们可以使用列表来存储用户输入的值。Python 中的列表是动态的，因此它可以随着用户输入新值而增长：

1.  我们声明列表并将其初始化为空，以便开始。

1.  接下来，我们声明用于用户输入的变量并将其设置为 `0`。我们使用零的原因是，当我们到达 `while` 循环的行时，我们希望条件为真。如果 `input_value` 变量是 `0` 或更大，则条件为真。

1.  在 `while` 循环内部，我们将使用 `input` 函数让用户输入值。`input` 函数允许我们向它传递一个字符串，该字符串将被显示给用户。这消除了在某些其他语言中我们需要先打印这条消息然后再获取用户输入的需求。

1.  从 `input` 函数获取的值是一个字符串，因此我们需要将其转换为 `int`。我们通过将输入的字符串传递给 `int()` 来做到这一点。这将创建一个具有输入值的整数。

1.  接下来，我们检查输入的值是否大于或等于 `0`。如果是，我们将它追加到我们的列表中。

    当用户输入一个负数时，我们将退出`while`循环并继续到`for`循环。Python 中的`for`循环总是像`foreach`循环一样工作。`for`循环需要一个序列，并且它会遍历该序列的所有值。我们的列表就是这样一种序列，因此我们每次迭代时都会得到一个项目，现在我们可以打印出这个项目的值，如下所示：

    ```py
    values = []
    input_value = 0
    while input_value >= 0:
        input_value = int(input("Enter a number: "))
        if input_value >= 0:
            values.append(input_value)
    for value in values:
        print(value)
    ```
