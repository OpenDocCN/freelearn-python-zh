## 第六章：6

用户输入和输出

软件的关键目的是产生有用的输出。在许多可能的输出中，一种简单的输出类型是显示有用结果的文本。Python 通过 print()函数支持这一点。

input()函数与 print()函数类似。input()函数从控制台读取文本，使我们能够向程序提供数据。使用 print()和 input()在应用程序的输入和输出之间创建了一种优雅的对称性。

向程序提供输入有许多其他常见方式。解析命令行对许多应用程序很有帮助。我们有时需要使用配置文件来提供有用的输入。数据文件和网络连接是提供输入的更多方式。这些方法各不相同，需要单独考虑。在本章中，我们将关注 input()和 print()的基础知识。

在本章中，我们将探讨以下配方：

+   使用 print()函数的特性

+   使用 input()和 getpass()获取用户输入

+   使用 f”{value=}”字符串进行调试

+   使用 argparse 获取命令行输入

+   使用 invoke 获取命令行输入

+   使用 cmd 创建命令行应用程序

+   使用 OS 环境设置

似乎最好从 print()函数开始，并展示它可以做的一些事情。毕竟，应用程序的输出通常最有用。

# 6.1 使用 print()函数的特性

在许多情况下，print()函数是我们首先了解的函数。第一个脚本通常是以下内容的变体：

```py
 >>> print("Hello, world.") 

Hello, world.
```

print()函数可以显示多个值，项目之间有有帮助的空格。

当我们这样写时：

```py
 >>> count = 9973 

>>> print("Final count", count) 

Final count 9973
```

我们可以看到，为我们包含了空格分隔符。此外，在函数中提供的值之后，通常会打印一个换行符，通常表示为\n 字符。

我们能否控制这种格式？我们能否更改提供的额外字符？

## 6.1.1 准备工作

考虑这个用于记录大型帆船燃油消耗的电子表格。CSV 文件中的行看起来像这样：

```py
 date,engine on,fuel height on,engine off,fuel height off 

10/25/13,08:24:00,29,13:15:00,27 

10/26/13,09:12:00,27,18:25:00,22 

10/28/13,13:21:00,22,06:25:00,14
```

关于此数据的更多信息，请参阅第四章中的缩小集合 – remove()、pop()和 difference 和切片和切块列表配方。由于油箱内部没有传感器，燃油的深度是通过油箱侧面的玻璃面板观察到的。知道油箱大约是矩形的，深度约为 31 英寸，容量约为 72 加仑，可以将深度转换为体积。

这里是一个使用此 CSV 数据的示例。此函数读取文件并返回由每一行构建的字段列表：

```py
 from pathlib import Path 

import csv 

def get_fuel_use(source_path: Path) -> list[dict[str, str]]: 

    with source_path.open() as source_file: 

        rdr = csv.DictReader(source_file) 

        return list(rdr)
```

这里是一个从 CSV 文件中读取和打印行的示例：

```py
 >>> source_path = Path("data/fuel2.csv") 

>>> fuel_use = get_fuel_use(source_path) 

>>> for row in fuel_use: 

...     print(row) 

{’date’: ’10/25/13’, ’engine on’: ’08:24:00’, ’fuel height on’: ’29’, ’engine off’: ’13:15:00’, ’fuel height off’: ’27’} 

{’date’: ’10/26/13’, ’engine on’: ’09:12:00’, ’fuel height on’: ’27’, ’engine off’: ’18:25:00’, ’fuel height off’: ’22’} 

{’date’: ’10/28/13’, ’engine on’: ’13:21:00’, ’fuel height on’: ’22’, ’engine off’: ’06:25:00’, ’fuel height off’: ’14’}
```

print()函数的输出，如这里所示的长行，使用起来具有挑战性。让我们看看如何使用 print()函数的附加功能来改进这个输出。

## 6.1.2 如何做...

我们有两种方式来控制 print()函数的输出格式：

+   设置字段分隔符字符串，sep。默认值是一个空格字符。

+   设置行结束字符串，end。默认值是\n 字符。

这个配方将展示几个变体：

1.  读取数据：

    ```py
     >>> fuel_use = get_fuel_use(Path("data/fuel2.csv"))
    ```

1.  对于数据中的每一项，进行任何有用的数据转换：

    ```py
     >>> for leg in fuel_use: 

    ...     start = float(leg["fuel height on"]) 

    ...     finish = float(leg["fuel height off"])
    ```

1.  以下替代方案展示了不同的包含分隔符的方法：

    +   使用 sep 和 end 的默认值打印标签和字段：

        ```py
         ...     print("On", leg["date"], "from", leg["engine on"], 

        ...         "to", leg["engine off"], 

        ...         "change", start-finish, "in.") 

        On 10/25/13 from 08:24:00 to 13:15:00 change 2.0 in. 

        On 10/26/13 from 09:12:00 to 18:25:00 change 5.0 in. 

        On 10/28/13 from 13:21:00 to 06:25:00 change 8.0 in. 
        ```

        当我们查看输出时，我们可以看到在每一项之间插入了一个空格。

    +   在准备数据时，我们可能希望使用类似于 CSV 的格式，可能使用非简单逗号的列分隔符。我们可以使用" | "作为 sep 参数的字符串值来打印标签和字段：

        ```py
         ...     print(leg["date"], leg["engine on"], 

        ...         leg["engine off"], start-finish, sep=" | ") 

        10/25/13 | 08:24:00 | 13:15:00 | 2.0 

        10/26/13 | 09:12:00 | 18:25:00 | 5.0 

        10/28/13 | 13:21:00 | 06:25:00 | 8.0 
        ```

        在这种情况下，我们可以看到每一列都有给定的分隔符字符串。由于没有更改结束设置，每个 print()函数都产生了一条独特的输出行。

    +   这是我们如何更改默认标点符号以强调字段名称和值的示例。我们可以使用"="作为 sep 参数的字符串值和", "作为 end 参数的值来打印标签和字段：

        ```py
         ...     print("date", leg["date"], sep="=", end=", ") 

        ...     print("on", leg["engine on"], sep="=", end=", ") 

        ...     print("off", leg["engine off"], sep="=", end=", ") 

        ...     print("change", start-finish, sep="=") 

        date=10/25/13, on=08:24:00, off=13:15:00, change=2.0 

        date=10/26/13, on=09:12:00, off=18:25:00, change=5.0 

        date=10/28/13, on=13:21:00, off=06:25:00, change=8.0 
        ```

        由于行尾使用的字符串已更改为", "，因此 print()函数的每次使用不再产生单独的行。为了看到正确的行尾，最后的 print()函数有一个默认的 end 值。我们也可以使用 end="\n"的参数值来明确地表示换行符的存在。

## 6.1.3 它是如何工作的...

print()函数的定义包括几个必须以关键字形式提供的参数。其中两个是 sep 和 end 关键字参数，分别具有空格和换行符的默认值。

使用 print()函数的 sep 和 end 参数对于比这些简单示例更复杂的情况可能会变得相当复杂。而不是处理一系列复杂的 print()函数请求，我们可以使用字符串的 format()方法，或者使用 f-string。

## 6.1.4 更多...

sys 模块定义了两个始终可用的标准输出文件：sys.stdout 和 sys.stderr。通常，print()函数可以被视为 sys.stdout.write()的一个便捷包装器。

我们可以使用 file=关键字参数将内容写入标准错误文件，而不是写入标准输出文件：

```py
 >>> import sys 

>>> print("Red Alert!", file=sys.stderr) 
```

我们导入了 sys 模块，以便我们可以访问标准错误文件。我们使用它来写入一条不会成为标准输出流一部分的消息。

由于这两个文件始终可用，使用 OS 文件重定向技术通常效果很好。当我们的程序的主要输出写入 sys.stdout 时，它可以在 OS 级别进行重定向。用户可能会输入一个类似这样的 shell 命令行：

```py
 % python myapp.py < input.dat > output.dat
```

这将为 sys.stdin 提供 input.dat 文件作为输入。当这个 Python 程序写入 sys.stdout 时，输出将由 OS 重定向到 output.dat 文件。

在某些情况下，我们需要打开额外的文件。在这种情况下，我们可能会看到这样的编程：

```py
 >>> from pathlib import Path 

>>> target_path = Path("data")/"extra_detail.log" 

>>> with target_path.open(’w’) as target_file: 

...     print("Some detailed output", file=target_file) 

...     print("Ordinary log") 

Ordinary log
```

在本例中，我们已为输出打开了一个特定的路径，并使用 with 语句将打开的文件分配给 target_file。然后我们可以将此作为 print()函数中的 file=值来写入此文件。因为文件是一个上下文管理器，所以离开 with 语句意味着文件将被正确关闭；所有 OS 资源都将从应用程序中释放。所有文件操作都应该用 with 语句的上下文包装，以确保资源得到适当的释放。

## 6.1.5 相关阅读

+   对于更多格式化选项，请参阅使用 f”{value=}"字符串进行调试食谱。

+   关于本例中输入数据的更多信息，请参阅第四章中的收缩集合 – remove(), pop(), 和 difference 和切片和切块列表食谱。

+   关于文件操作的一般信息，请参阅第八章。

# 6.2 使用 input()和 getpass()获取用户输入

一些 Python 脚本依赖于从用户那里收集输入。有几种方法可以做到这一点。一种流行的技术是使用控制台以交互方式提示用户输入。

有两种相对常见的情况：

+   普通输入：这将提供输入字符的有帮助的回显。

+   安全、无回显输入：这通常用于密码。输入的字符不会显示，提供了一定程度的隐私。我们使用 getpass 模块中的 getpass()函数来完成这项工作。

作为交互式输入的替代方案，我们将在本章后面的使用 argparse 获取命令行输入食谱中探讨一些其他方法。

input()和 getpass()函数只是从控制台读取的两种实现选择。结果是，获取字符字符串只是收集有用数据的第一步。输入还需要进行验证。

## 6.2.1 准备工作

我们将探讨一种从人那里读取复杂结构的技术。在这种情况下，我们将使用年、月和日作为单独的项目。然后这些项目被组合起来创建一个完整的日期。

这里有一个快速的用户输入示例，省略了所有验证考虑。这是糟糕的设计：

```py
 from datetime import date 

def get_date1() -> date: 

    year = int(input("year: ")) 

    month = int(input("month [1-12]: ")) 

    day = int(input("day [1-31]: ")) 

    result = date(year, month, day) 

    return result
```

虽然使用 input()函数非常容易，但它缺少许多有用的功能。当用户输入无效日期时，这可能会引发一个可能令人困惑的异常。

我们经常需要将 input()函数与数据验证处理包装起来，使其更有用。日历很复杂，我们不愿意在未警告用户的情况下接受 2 月 31 日，这不是一个正确的日期。

## 6.2.2 如何实现...

1.  如果输入是密码或类似需要编辑的内容，则 input()函数不是最佳选择。如果涉及密码或其他秘密，则使用 getpass.getpass()函数。这意味着当涉及秘密时，我们需要以下导入：

    ```py
     from getpass import getpass
    ```

    否则，当不需要秘密输入时，我们将使用内置的 input()函数，不需要额外的导入。

1.  确定将使用哪个提示。在我们的例子中，我们提供了一个字段名称和有关预期数据类型的提示作为 input()或 getpass()函数的提示字符串参数。这有助于将输入与文本到整数的转换分开。这个配方不遵循之前显示的片段；它将操作分解为两个独立的步骤。首先，获取文本值：

    ```py
     year_text = input("year: ")
    ```

1.  确定如何单独验证每个项目。最简单的情况是一个具有涵盖所有内容的单个规则的单个值。在更复杂的情况下——就像这个例子——每个单独的元素都是一个具有范围约束的数字。在后续步骤中，我们将查看验证组合项目：

    ```py
     year = int(year_text)
    ```

    将输入和验证包装成如下所示的 while-try 块：

    ```py
     year = None 

        while year is None: 

            year_text = input("year: ") 

            try: 

                year = int(year_text) 

            except ValueError as ex: 

                print(ex)
    ```

此处应用单个验证规则，即 int(year_txt)表达式，以确保输入是整数。while 语句导致输入和转换步骤的重复，直到 year 变量的值为 None。

对于错误输入抛出异常为我们提供了一些灵活性。我们可以通过扩展额外的异常类来满足输入必须满足的其他条件。

此处理过程仅涵盖年份字段。我们还需要获取月份和日期字段的值。这意味着我们需要为复杂日期对象的这三个字段分别编写三个几乎相同的循环。为了避免复制和粘贴几乎相同的代码，我们需要重构此处理过程。

我们将定义一个新的函数 get_integer()，用于通用数字值输入。以下是完整的函数定义：

```py
 def get_integer(prompt: str) -> int: 

    while True: 

        value_text = input(prompt) 

        try: 

            value = int(value_text) 

            return value 

        except ValueError as ex: 

            print(ex) 
```

我们可以将这些组合成一个整体过程，以获取日期的三个整数。这将涉及类似 while-try 设计模式，但应用于组合对象。它看起来像这样：

```py
 def get_date2() -> date: 

    while True: 

        year = get_integer("year: ") 

        month = get_integer("month [1-12]: ") 

        day = get_integer("day [1-31]: ") 

        try: 

            result = date(year, month, day) 

            return result 

        except ValueError as ex: 

            print(f"invalid, {ex}")
```

这使用围绕 get_integer()函数序列的单独 while-try 处理序列来获取构成日期的各个值。然后，它使用 date()构造函数从单个字段创建日期对象。如果由于组件无效，无法构建日期对象——作为一个整体——则必须重新输入年、月和日以创建一个有效的日期。

## 6.2.3 它是如何工作的...

我们需要将输入问题分解为几个相互独立但密切相关的子问题。为此，想象一个转换步骤的塔。在最底层是与用户的初始交互。我们确定了两种处理这种交互的常见方法：

+   input()：此函数提示并从用户那里读取

+   getpass.getpass()：此函数提示并读取输入（如密码）而不显示回显

这两个函数提供了基本的控制台交互。如果需要更复杂的交互，还有其他库可以提供。例如，Click 项目有一些有用的提示功能。参见[`click.palletsprojects.com/en/7.x/`](https://click.palletsprojects.com/en/7.x/)。

Rich 项目具有极其复杂的终端交互。参见[`rich.readthedocs.io/en/latest/`](https://rich.readthedocs.io/en/latest/)。

在基础之上，我们构建了几个验证处理的层级。层级如下：

+   数据类型验证：这使用内置的转换函数，如 int()或 float()。这些函数对无效文本引发 ValueError 异常。

+   域验证：这使用 if 语句来确定值是否符合任何特定应用程序的约束。为了保持一致性，如果数据无效，也应引发 ValueError 异常。

+   组合对象验证：这是特定于应用程序的检查。在我们的例子中，组合对象是 datetime.date 的一个实例。这也倾向于对无效的日期引发 ValueError 异常。

可能对值施加的约束类型有很多。我们使用了有效的日期约束，因为它特别复杂。

## 6.2.4 更多内容...

我们有几种涉及略微不同方法的用户输入替代方案。我们将详细探讨这两个主题：

+   复杂文本：这将涉及简单使用 input()和更复杂的源文本解析。而不是提示单个字段，可能更好的是接受 yyyy-mm-dd 格式的字符串，并使用 strptime()解析器提取日期。这不会改变设计模式；它用稍微复杂一些的东西替换了 int()或 float()。

+   通过 cmd 模块进行交互：这涉及一个更复杂的类来控制交互。我们将在使用 cmd 创建命令行应用程序的配方中详细探讨这一点。

可以从 JSON 模式定义中提取潜在输入验证规则列表。此类型列表包括布尔值、整数、浮点数和字符串。在 JSON 模式中定义的许多常见字符串格式包括日期时间、时间、日期、电子邮件、主机名、IPv4 和 IPv6 格式的 IP 地址以及 URI。

用户输入验证规则的另一个来源可以在 HTML5 <input>标签的定义中找到。此列表包括颜色、日期、datetime-local、电子邮件、文件、月份、数字、密码、电话号码、时间、URL 和周年的格式。

## 6.2.5 参见

+   在本章中查看使用 cmd 创建命令行应用程序的配方以了解复杂交互。

+   查看使用使用 argparse 获取命令行输入的配方以从命令行收集用户输入。

+   在 SunOS 操作系统的参考资料中，该系统现在由 Oracle 拥有，其中包含一组提示不同类型用户输入的命令：[`docs.oracle.com/cd/E19683-01/816-0210/6m6nb7m5d/index.html`](https://docs.oracle.com/cd/E19683-01/816-0210/6m6nb7m5d/index.html)

# 6.3 使用 f”{value=}”字符串进行调试

Python 中可用的重要调试和设计工具之一是 print()函数。在使用 print()函数的功能食谱中显示的两个格式选项提供的灵活性不多。我们有更多的灵活性使用 f"string"格式。我们将基于第一章、数字、字符串和元组中显示的一些食谱。

## 6.3.1 准备工作

让我们看看一个涉及一些中等复杂计算的多步骤过程。我们将计算一些样本数据的平均值和标准差。给定这些值，我们将定位所有高于平均值一个标准差的项：

```py
 >>> import statistics 

>>> size = [2353, 2889, 2195, 3094, 

...     725, 1099, 690, 1207, 926, 

...     758, 615, 521, 1320] 

>>> mean_size = statistics.mean(size) 

>>> std_size = statistics.stdev(size) 

>>> sig1 = round(mean_size + std_size, 1) 

>>> [x for x in size if x > sig1] 

[2353, 2889, 3094]
```

这个计算有几个工作变量。最终的列表推导式涉及其他三个变量，mean_size、std_size 和 sig1。使用这么多值来过滤大小列表，很难可视化正在发生的事情。了解计算的步骤通常很有帮助；显示中间变量的值可能非常有帮助。

## 6.3.2 如何做...

f"{name=}"字符串将同时包含字面字符串 name=和 name 表达式的值。这通常是一个变量，但可以使用任何表达式。使用这个与 print()函数结合的例子如下：

```py
 >>> print( 

...     f"{mean_size=:.2f}, {std_size=:.2f}" 

... ) 

mean_size=1414.77, std_size=901.10
```

我们可以使用{name=}将任何变量放入 f-string 中并查看其值。上述代码中的这些例子包括格式说明符:.2f 作为后缀，以显示四舍五入到两位小数的值。另一个常见的后缀是!r，用于显示对象的内部表示；我们可能会使用 f"{name=!r}"。

## 6.3.3 它是如何工作的...

关于格式选项的更多背景信息，请参阅第一章中构建复杂的 f-string 字符串食谱。

这个功能有一个非常实用的扩展。我们可以在 f-string 中的=左侧使用任何表达式。这将显示表达式及其计算出的值，为我们提供更多的调试信息。

## 6.3.4 更多...

我们可以使用 f-string 的扩展表达式功能包括额外的计算，这些计算不仅仅是局部变量的值：

```py
 >>> print( 

...     f"{mean_size=:.2f}, {std_size=:.2f}," 

...     f" {mean_size + 2 * std_size=:.2f}" 

... ) 

mean_size=1414.77, std_size=901.10, mean_size + 2 * std_size=3216.97
```

我们已经计算了一个新值，mean_size+2*std_size，它只出现在格式化输出中。这使得我们可以在不创建额外变量的情况下显示中间计算结果。

## 6.3.5 参见

+   参考第一章中的“使用 f-strings 构建复杂的字符串”食谱，了解更多可以使用 f-strings 和字符串的 format()方法完成的事情。

+   参考本章前面的“使用 print()函数的特性”食谱，了解其他格式化选项。

# 6.4 使用 argparse 获取命令行输入

对于某些应用，在没有太多人工交互的情况下从操作系统命令行获取用户输入可能更好。我们更愿意解析命令行参数值，然后执行处理或报告错误。

例如，在操作系统级别，我们可能想运行这样的程序：

```py
 % python ch06/distance_app.py -u KM 36.12,-86.67 33.94,-118.40 

From 36.12,-86.67 to 33.94,-118.4 in KM = 2886.90
```

在%的操作系统提示符下，我们输入了一个命令，python ch06/distance_app.py。此命令有一个可选参数，-u KM，以及两个位置参数 36.12,-86.67 和 33.94,-118.40。

如果用户输入错误，交互可能看起来像这样：

```py
 % python ch06/distance_app.py -u KM 36.12,-86.67 33.94,-118asd 

usage: distance_app.py [-h] [-u {NM,MI,KM}] p1 p2 

distance_app.py: error: argument p2: could not convert string to float: ’-118asd’
```

-118asd 的无效参数值会导致错误消息。用户可以按上箭头键获取之前的命令行，进行更改，然后再次运行程序。交互式用户体验委托给操作系统命令行处理。

## 6.4.1 准备工作

我们需要做的第一件事是重构我们的代码，创建三个单独的函数：

+   一个从命令行获取参数的函数。

+   一个执行实际工作的函数。目的是定义一个可以在各种环境中重用的函数，其中之一是使用命令行参数。

+   一个主函数，它收集参数并使用适当的参数值调用实际工作函数。

下面是我们的实际工作函数，display()：

```py
 from ch06.distance_computation import haversine, MI, NM, KM 

def display( 

        lat1: float, lon1: float, lat2: float, lon2: float, r: str 

) -> None: 

    r_float = {"NM": NM, "KM": KM, "MI": MI}[r] 

    d = haversine(lat1, lon1, lat2, lon2, R=r_float) 

    print(f"From {lat1},{lon1} to {lat2},{lon2} in {r} = {d:.2f}")
```

我们已从 ch06.distance_computation 模块导入了核心计算函数 haversine()。这是基于第三章中“基于部分函数选择参数顺序”食谱中显示的计算：

下面是函数在 Python 内部使用时的样子：

```py
 >>> display(36.12, -86.67, 33.94, -118.4, ’NM’) 

From 36.12,-86.67 to 33.94,-118.4 in NM = 1558.53
```

此函数有两个重要的设计特点。第一个特点是它避免了引用由参数解析创建的 argparse.Namespace 对象的功能。我们的目标是拥有一个可以在多个不同环境中重用的函数。我们需要将用户界面的输入和输出元素分开。

第二个设计特点是该功能显示另一个函数计算出的值。这是将一个较大的问题分解为两个较小问题的分解。我们将打印输出的用户体验与基本计算分离。（这两个方面都相当小，但分离这两个方面的原则很重要。）

## 6.4.2 如何实现...

1.  定义整体参数解析函数：

    ```py
     def get_options(argv: list[str]) -> argparse.Namespace:
    ```

1.  创建解析器对象：

    ```py
     parser = argparse.ArgumentParser()
    ```

1.  将各种类型的参数添加到解析器对象中。有时，这很困难，因为我们仍在改进用户体验。很难想象人们会如何使用程序以及他们可能提出的所有问题。在我们的例子中，我们有两个必需的位置参数和一个可选参数：

    +   第一点：纬度和经度

    +   第二点：纬度和经度

    +   可选的距离单位；我们将提供海里作为默认值：

    ```py
     parser.add_argument("-u", "--units", 

            action="store", choices=("NM", "MI", "KM"), default="NM") 

        parser.add_argument("p1", action="store", type=point_type) 

        parser.add_argument("p2", action="store", type=point_type)
    ```

    我们添加了可选和必需参数的混合。-u 参数以短横线开头，表示它是可选的。支持较长的双短横线版本 --units 作为替代。

    必需的位置参数不带前缀命名。

1.  评估步骤 2 中创建的解析器对象的 parse_args() 方法：

    ```py
     options = parser.parse_args(argv) 
    ```

默认情况下，解析器使用 sys.argv 的值，即用户输入的命令行参数值。当我们能够提供明确的参数值时，测试会更加容易。

下面是最终的函数：

```py
 def get_options(argv: list[str]) -> argparse.Namespace: 

    parser = argparse.ArgumentParser() 

    parser.add_argument("-u", "--units", 

        action="store", choices=("NM", "MI", "KM"), default="NM") 

    parser.add_argument("p1", action="store", type=point_type) 

    parser.add_argument("p2", action="store", type=point_type) 

    options = parser.parse_args(argv) 

    return options
```

这依赖于一个 point_type() 函数，该函数既验证字符串，又把字符串转换成（纬度，经度）两个元素的元组。下面是这个函数的定义：

```py
 def point_type(text: str) -> tuple[float, float]: 

    try: 

        lat_str, lon_str = text.split(",") 

        lat = float(lat_str) 

        lon = float(lon_str) 

        return lat, lon 

    except ValueError as ex: 

        raise argparse.ArgumentTypeError(ex)
```

如果出现任何问题，将引发异常。从这个异常中，我们将引发 ArgumentTypeError 异常。这个异常被 argparse 模块捕获，并导致它向用户报告错误。

这里是结合选项解析器和输出显示功能的主体脚本：

```py
 def main(argv: list[str] = sys.argv[1:]) -> None: 

    options = get_options(argv) 

    lat_1, lon_1 = options.p1 

    lat_2, lon_2 = options.p2 

    display(lat_1, lon_1, lat_2, lon_2, r=options.r) 

if __name__ == "__main__": 

    main()
```

此主体脚本将用户输入连接到显示的输出。错误消息的详细信息和处理帮助被委托给 argparse 模块。

## 6.4.3 它是如何工作的...

参数解析器分为三个阶段：

1.  通过创建一个 ArgumentParser 类的实例来创建一个解析器对象，从而定义整体上下文。

1.  使用 add_argument() 方法添加单个参数。这些参数可以包括可选参数以及必需参数。

1.  解析实际的命令行输入，通常基于 sys.argv。

一些简单的程序可能只有几个可选参数。一个更复杂的程序可能有更多可选参数。

通常，文件名作为位置参数。当程序读取一个或多个文件时，文件名可以按如下方式在命令行中提供：

```py
 % python some_program.py *.rst
```

我们使用了 Linux shell 的通配符功能：*.rst 字符串被扩展成匹配命名规则的文件列表。这是 Linux shell 的一个特性，发生在 Python 解释器开始之前。这个文件列表可以使用以下定义的参数进行处理：

```py
 parser.add_argument(’file’, type=Path, nargs=’*’)
```

命令行上所有不以 - 字符开头的参数都是位置参数，并且它们被收集到由解析器构建的对象的 file 值中。

然后，我们可以使用以下方式处理每个给定的文件：

```py
 for filename in options.file: 

        process(filename)
```

对于 Windows 程序，shell 不会从通配符模式中获取文件名。这意味着应用程序必须处理包含通配符字符（如 "*" 和 "?”）的文件名。Python 的 glob 模块可以帮助处理这个问题。此外，pathlib 模块可以创建 Path 对象，这些对象包括用于在目录中定位匹配文件名的通配符功能。

## 6.4.4 更多内容...

我们可以处理哪些类型的参数？在常见使用中有很多参数风格。所有这些变体都是使用解析器的 add_argument() 方法定义的：

+   简单选项：形式为 -o 或 --option 的参数通常定义可选功能。这些使用的是 ‘store_true’ 或 ‘store_false’ 动作。

+   带值的选项：我们展示了 -r unit 作为带值的选项。‘store’ 动作是保存值的方式。

+   增加计数的选项：动作 ‘count’ 和默认值 =0 允许重复的选项。例如，详细和非常详细的日志选项 -v 和 -vv。

+   累积列表的选项：动作 ‘append’ 和默认值 [] 可以累积多个选项值。

+   显示版本号：可以使用特殊动作 ‘version’ 创建一个将显示版本号并退出的参数。

+   位置参数在其名称中不带有前导 ‘-’。它们必须按照将使用的顺序定义。

argparse 模块使用 -h 和 --help 将显示帮助信息并退出。除非使用具有 ‘help’ 动作的参数更改，否则这些选项都是可用的。

这涵盖了命令行参数处理的常见情况。通常，当我们编写自己的应用程序时，我们会尝试利用这些常见的参数风格。如果我们努力遵循广泛使用的参数风格，我们的用户更有可能理解我们的应用程序是如何工作的。

## 6.4.5 参见

+   我们在 使用 input() 和 getpass() 获取用户输入 的菜谱中探讨了如何获取交互式用户输入。

+   我们将在 使用 OS 环境设置 的菜谱中查看如何添加更多灵活性。

# 6.5 使用 invoke 获取命令行输入

[invoke](https://www.pyinvoke.org) 包不是标准库的一部分。它需要单独安装。通常，这是通过以下终端命令完成的：

```py
(cookbook3) % python -m pip install invoke
```

使用 python -m pip 命令确保我们将使用与当前活动虚拟环境一起的 pip 命令，显示为 cookbook3。

请参阅本章中的 使用 argparse 获取命令行输入 菜谱。它描述了一个类似以下的工作命令行应用程序：

```py
 % RECIPE=7  # invoke
```

命令始终是 invoke。Python 路径信息用于定位名为 tasks.py 的模块文件，以提供可以调用的命令的定义。剩余的命令行值提供给 tasks 模块中定义的函数。

## 6.5.1 准备工作

使用 invoke 时，我们通常会创建一个双层设计。这两层是：

+   一个从命令行获取参数、执行所需的验证或转换，并调用函数执行真实工作的函数。这个函数将被装饰为@task。

+   执行真实工作的函数。如果这个函数被设计成不直接引用命令行选项，那么这会有所帮助。目的是定义一个可以在各种环境中重用的函数，其中之一就是使用来自命令行的参数。

在某些情况下，这两个函数可以合并成一个。这种情况通常发生在 Python 被用作包装器，为底层提供简单接口，而底层应用却异常复杂时。在这种应用中，Python 包装器可能只做很少的处理，参数值验证与应用的“真实工作”之间没有有用的区别。

在本章的使用 argparse 获取命令行输入配方中，定义了 display()函数。这个函数执行应用的“真实工作”。当与 invoke 一起工作时，这种设计将继续使用。

## 6.5.2 如何做...

1.  定义一个描述可以调用的任务的函数。通常，提供一些关于各种参数的帮助信息是至关重要的，这可以通过向@task 装饰器提供参数名称和帮助文本的字典来完成：

    ```py
     import sys 

    from invoke.tasks import task 

    from invoke.context import Context
    ```

    ```py
     @task( 

        help={ 

            ’p1’: ’Lat,Lon’, 

            ’p2’: ’Lat,Lon’, 

            ’u’: ’Unit: KM, MI, NM’}) 

    def distance( 

            context: Context, p1: str, p2: str, u: str = "KM" 

    ) -> None: 

        """Compute distance between two points. 

        """
    ```

    函数的文档字符串成为 invoke distance --help 命令提供的帮助文本。提供一些有助于用户理解各种命令将做什么以及如何使用它们的内容非常重要。

    Context 参数是必需的，但在这个例子中不会使用。该对象在调用多个单独的任务时提供一致的环境。它还提供了运行外部应用程序的方法。

1.  对各种参数值进行所需的转换。使用清洗后的值评估“真实工作”函数：

    ```py
     try: 

            lat_1, lon_1 = point_type(p1) 

            lat_2, lon_2 = point_type(p2) 

            display(lat_1, lon_1, lat_2, lon_2, r=u) 

        except (ValueError, KeyError) as ex: 

            sys.exit(f"{ex}\nFor help use invoke --help distance") 
    ```

    我们已经使用 sys.exit()来生成错误信息。也可以抛出异常，但这会显示长的跟踪信息，可能并不有用。

## 6.5.3 它是如何工作的...

invoke 包检查给定 Python 函数的参数，并构建必要的命令行解析选项。参数名称成为选项的名称。在示例 distance()函数中，p1、p2 和 u 的参数分别成为命令行选项--p1、--p2 和-u。这使得我们可以在运行应用时灵活地提供参数。值可以是按位置提供，也可以通过使用选项标志提供。

## 6.5.4 更多...

invoke 最重要的特性是它能够作为其他二进制应用程序的包装器。提供给每个任务的 Context 对象提供了更改当前工作目录和运行任意 OS 命令的方法。这包括更新子进程环境、捕获输出和错误流、提供输入流以及许多其他功能。

我们可以使用 invoke 在单个包装器下组合多个应用程序。这可以通过提供一个统一的接口来简化复杂的应用程序集合，该接口通过单个任务定义模块实现。

例如，我们可以组合一个计算两点之间距离的应用程序，以及一个处理连接一系列点的完整路线的 CSV 文件的应用程序。

整体设计可能看起来像这样：

```py
@task 

def distance(context: Context, p1: str, p2: str, u: str) -> None: 

    ...  # Shown earlier 

@task 

def route(context: Context, filename: str) -> None: 

    if not path(filename).exists(): 

        sys.exit(f"File not found {filename}") 

    context.run("python some_app.py {filename}", env={"APP_UNITS": "NM"})
```

context.run()方法将调用任意的 OS 级命令。env 参数值提供了更新环境变量的命令。

## 6.5.5 参见

+   应用程序集成的附加配方在第十四章应用程序集成：组合中展示。

+   [`www.pyinvoke.org`](https://www.pyinvoke.org)网页包含了关于 invoke 的所有文档。

# 6.6 使用 cmd 创建命令行应用程序

有几种方法可以创建交互式应用程序。使用 input()和 getpass()获取用户输入配方探讨了 input()和 getpass.getpass()等函数。使用 argparse 获取命令行输入配方展示了如何使用 argparse 模块创建用户可以从 OS 命令行与之交互的应用程序。

我们还有另一种创建交互式应用程序的方法：使用 cmd 模块。此模块将提示用户输入，然后调用我们提供的类的一个特定方法。

这里是一个交互示例：

```py
 ] dice 5 

Rolling 5 dice 

] roll 

[5, 6, 6, 1, 5] 

]
```

我们输入了 dice 5 命令来设置骰子的数量。之后，roll 命令显示了掷出五个骰子的结果。help 命令将显示可用的命令。

## 6.6.1 准备工作

cmd.Cmd 应用程序的核心特性是一个读取-评估-打印循环（REPL）。当存在多个单独的状态变化和许多密切相关用于执行这些状态变化的命令时，这种应用程序运行良好。

我们将使用一个简单的、有状态的骰子游戏。想法是有一把骰子，其中一些可以掷出，而另一些是冻结的。这意味着我们的 Cmd 类定义必须有一些属性来描述一把骰子的当前状态。

命令将包括以下内容：

+   dice 设置骰子的数量

+   roll 掷骰子

+   reroll 重新掷选定的骰子，其他骰子保持不变

## 6.6.2 如何实现...

1.  导入 cmd 模块以使 cmd.Cmd 类定义可用。由于这是一个游戏，还需要随机模块：

    ```py
     import cmd 

    import random
    ```

1.  定义一个扩展到 cmd.Cmd：

    ```py
     class DiceCLI(cmd.Cmd):
    ```

1.  在 preloop() 方法中定义任何所需的初始化：

    ```py
     def preloop(self) -> None: 

            self.n_dice = 6 

            self.dice: list[int] | None = None  # no roll has been made. 

            self.reroll_count = 0
    ```

    此方法在处理开始时评估一次。

    初始化也可以在 __init__() 方法中完成。然而，这样做稍微复杂一些，因为它必须与 Cmd 类的初始化协作。

1.  对于每个命令，创建一个 do_command() 方法。方法名将是命令，前面加上 do_ 字符。任何命令之后的用户输入文本将作为方法的参数值提供。方法定义中的文档字符串注释是命令的帮助文本。以下是由 do_roll() 方法定义的 roll 命令：

    ```py
     def do_roll(self, arg: str) -> bool: 

            """Roll the dice. Use the dice command to set the number of dice.""" 

            self.dice = [random.randint(1, 6) for _ in range(self.n_dice)] 

            print(f"{self.dice}") 

            return False
    ```

1.  解析和验证使用它们的命令的参数。用户在命令之后的输入将作为方法第一个位置参数的值提供。以下是由 do_dice() 方法定义的 dice 命令：

    ```py
     def do_dice(self, arg: str) -> bool: 

            """Sets the number of dice to roll.""" 

            try: 

                self.n_dice = int(arg) 

            except ValueError: 

                print(f"{arg!r} is invalid") 

                return False 

            self.dice = None 

            print(f"Rolling {self.n_dice} dice") 

            return False
    ```

1.  编写主脚本。这将创建此类的实例并执行 cmdloop() 方法：

    ```py
     if __name__ == "__main__": 

        game = DiceCLI() 

        game.cmdloop()
    ```

    cmdloop() 方法处理提示、收集输入和根据用户的输入执行适当方法的细节。

## 6.6.3 它是如何工作的...

Cmd 类包含大量内置功能来显示提示、从用户那里读取输入，然后根据用户的输入定位适当的方法。

例如，当我们输入 dice 5 这样的命令时，Cmd 超类的内置方法将从输入中删除第一个单词 dice，并将其前缀为 do_。然后它将尝试使用行剩余部分的参数值，即 5，来评估该方法。

如果我们输入了一个没有匹配 do_*() 方法的命令，命令处理器将写入一个错误信息。这是自动完成的；我们不需要编写任何代码来处理无效的命令输入。

一些方法，如 do_help()，已经是应用程序的一部分。这些方法将总结其他 do_* 方法。当我们的方法有一个文档字符串时，这将通过内置的帮助功能显示。

Cmd 类依赖于 Python 的内省功能。类的实例可以检查方法名以定位所有以 do_ 开头的方法。内省是一个高级主题，将在第八章中简要介绍。

## 6.6.4 更多...

Cmd 类有多个可以添加交互功能的地方：

+   我们可以定义特定的 help_*() 方法，使其成为帮助主题的一部分。

+   当任何 do_*() 方法返回非 False 值时，循环将结束。我们可能想要添加一个 do_quit() 方法来返回 True。

+   如果输入流被关闭，将提供一个 EOF 命令。在 Linux 中，使用 ctrl-d 将关闭输入文件。这导致 do_EOF() 方法，它应该使用 return True。

+   我们可能提供一个名为 emptyline() 的方法来响应空白行。

+   当用户的输入与任何 do_*() 方法都不匹配时，将评估 default() 方法。

+   postloop() 方法可以在循环结束后进行一些处理。这是一个写总结的好地方。

此外，我们还可以设置一些属性。这些是与方法定义平级的类级变量：

+   提示属性是要写入的提示字符串。介绍属性是在第一个提示之前要写入的介绍性文本。对于我们的示例，我们可以这样做：

    ```py
     class DiceCLI2(cmd.Cmd): 

        prompt = "] " 

        intro = "A dice rolling tool. ? for help."
    ```

+   我们可以通过设置 doc_header、undoc_header、misc_header 和 ruler 属性来定制帮助输出。

目标是能够创建一个尽可能直接处理用户交互的整洁类。

## 6.6.5 相关内容

+   我们将在第七章和第八章中查看类定义。

# 6.7 使用操作系统环境设置

有几种方式来看待我们软件用户提供的输入：

+   交互式输入：这是根据应用程序的要求由用户提供的。请参阅 使用 input() 和 getpass() 获取用户输入 的配方。

+   命令行参数：这些是在程序启动时提供的。请参阅 使用 argparse 获取命令行输入 和 使用 invoke 获取命令行输入 的配方。

+   环境变量：这些是操作系统级别的设置。有几种方式可以设置它们：

    +   在命令行中，当运行应用程序时。

    +   在用户选择的 shell 的配置文件中设置。例如，如果使用 zsh，这些文件是 ~/.zshrc 文件和 ~/.profile 文件。也可以有系统范围的文件，如 /etc/zshrc 文件。

    +   在 Windows 中，有环境变量的高级设置选项。

+   配置文件：这些是特定于应用程序的。它们是第十三章的主题。

环境变量可以通过 os 模块获得。

## 6.7.1 准备工作

在 使用 argparse 获取命令行输入 的配方中，我们将 haversine() 函数包装在一个简单的应用程序中，该应用程序解析命令行参数。我们创建了一个这样工作的程序：

```py
 % python ch06/distance_app.py -u KM 36.12,-86.67 33.94,-118.40 

From 36.12,-86.67 to 33.94,-118.4 in KM = 2886.90 

"""
```

在使用这个版本的应用程序一段时间后，我们可能会发现我们经常使用海里来计算从我们船锚定的地方的距离。我们真的希望有一个输入点的默认值以及 -r 参数的默认值。

由于一艘船可以在多个地方锚定，我们需要在不修改实际代码的情况下更改默认设置。一个“缓慢变化”的参数值的概念与操作系统环境变量很好地吻合。它们可以持久存在，但相对容易更改。

我们将使用两个操作系统环境变量：

+   UNITS 将具有默认的距离单位。

+   HOME_PORT 可以有一个锚点。

我们希望能够做到以下几点：

```py
 % UNITS=NM 

% HOME_PORT=36.842952,-76.300171 

% python ch06/distance_app.py 36.12,-86.67 

From 36.12,-86.67 to 36.842952,-76.300171 in NM = 502.23
```

## 6.7.2 如何实现...

1.  导入 os 模块。要解析的默认命令行参数集来自 sys.argv，因此还需要导入 sys 模块。应用程序还将依赖于 argparse 模块：

    ```py
     import os 

    import sys 

    import argparse
    ```

1.  导入应用程序需要的任何其他类或对象：

    ```py
     from ch03.recipe_11 import haversine, MI, NM, KM 

    from ch06.recipe_04 import point_type, display
    ```

1.  定义一个函数，该函数将使用环境值作为可选命令行参数的默认值：

    ```py
     def get_options(argv: list[str] = sys.argv[1:]) -> argparse.Namespace:
    ```

1.  从操作系统环境设置中收集默认值。这包括所需的任何验证：

    ```py
     default_units = os.environ.get("UNITS", "KM") 

        if default_units not in ("KM", "NM", "MI"): 

            sys.exit(f"Invalid UNITS, {default_units!r} not KM, NM, or MI") 

        default_home_port = os.environ.get("HOME_PORT")
    ```

    注意，使用 os.environ.get()允许应用程序在环境变量未设置的情况下包含一个默认值。

1.  创建解析器对象。为从环境变量中提取的相关参数提供默认值：

    ```py
     parser = argparse.ArgumentParser() 

        parser.add_argument("-u", "--units", 

            action="store", choices=("NM", "MI", "KM"), 

            default=default_units 

        ) 

        parser.add_argument("p1", action="store", type=point_type) 

        parser.add_argument( 

            "p2", nargs="?", action="store", type=point_type, 

            default=default_home_port 

        )
    ```

1.  进行任何额外的验证以确保参数设置正确。在这个例子中，可能没有为 HOME_PORT 设置值，也没有为第二个命令行参数提供值。

    这需要使用 if 语句和调用 sys.exit()：

    ```py
     options = parser.parse_args(argv) 

        if options.p2 is None: 

            sys.exit("Neither HOME_PORT nor p2 argument provided.")
    ```

1.  返回包含有效参数集的最终选项对象：

    ```py
     return options
    ```

这将使-u 参数和第二个点成为可选的。如果这些参数从命令行中省略，参数解析器将使用配置信息提供默认值。

sys.exit()提供的错误代码有细微的区别。当应用程序因命令行问题失败时，通常返回状态码 2，但 sys.exit()会将值设置为 1。一个稍微更好的方法是使用 parser.error()方法。这样做需要重构，在获取和验证环境变量值之前创建 ArgumentParser 实例。

## 6.7.3 它是如何工作的...

我们已经使用操作系统环境变量来创建默认值，这些值可以被命令行参数覆盖。如果环境变量已设置，则该字符串将作为默认值提供给参数定义。如果没有设置环境变量，则应用程序将使用默认值。在 UNITS 变量的情况下，在这个例子中，如果操作系统环境变量未设置，应用程序将使用公里作为默认值。

我们已经使用操作系统环境来设置默认值，这些值可以被命令行参数值覆盖。这支持环境提供可能由多个命令共享的一般上下文的概念。

## 6.7.4 更多...

使用 argparse 获取命令行输入的配方展示了处理从 sys.argv 中可用的默认命令行参数的略微不同的方法。第一个参数是正在执行的 Python 应用程序的名称，通常与参数解析不相关。

sys.argv 的值将是一个字符串列表：

```py
 [’ch06/distance_app.py’, ’-u’, ’NM’, ’36.12,-86.67’]
```

在处理过程中，我们不得不在某些时候跳过 sys.argv[0] 的初始值。通常，应用程序需要将 sys.argv[1:] 提供给解析器。这可以在 get_options() 函数内部完成。这也可以在 main() 函数评估 get_options() 函数时完成。正如本例所示，这也可以在为 get_options() 函数创建默认参数值时完成。

argparse 模块允许我们为参数定义提供类型信息。提供类型信息可以用于验证参数值。在许多情况下，一个值可能有一组有限的选项，这组允许的选项可以作为参数定义的一部分提供。这样做可以创建更好的错误和帮助信息，提高应用程序运行时的用户体验。

## 6.7.5 参见

+   我们将在第十三章节中探讨处理配置文件的多种方法。

# 加入我们的社区 Discord 空间

加入我们的 Python Discord 工作空间，讨论并了解更多关于这本书的信息：[`packt.link/dHrHU`](https://packt.link/dHrHU)

![PIC](img/file1.png)
