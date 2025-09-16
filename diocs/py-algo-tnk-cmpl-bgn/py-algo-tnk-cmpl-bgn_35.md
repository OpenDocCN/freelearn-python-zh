## 第二十三章

循环控制结构简介

### 23.1 什么是循环控制结构？

循环控制结构是一种控制结构，它允许在满足指定条件之前多次执行语句或语句块。

### 23.2 从顺序控制到循环控制结构

下一个示例允许用户输入四个数字，然后计算并显示它们的总和。正如你所看到的，还没有使用循环控制结构，只有熟悉的顺序控制结构。

x = float(input())

y = float(input())

z = float(input())

w = float(input())

total = x + y + z + w

print(total)

虽然这段代码很短，但考虑一个类似的代码，它允许用户输入 1000 个数字而不是仅仅四个。你能想象不得不写 float(input())一千次输入语句吗？如果你能只写一次这个语句并指示计算机执行一千次，那会方便得多，不是吗？这就是循环控制结构发挥作用的地方！

在你深入研究循环控制结构之前，先尝试解决一个谜题！在不使用循环控制结构的情况下，尝试重写之前的示例，只使用两个变量 x 和 total。是的，你听对了！这段代码必须计算并显示四个用户提供的数字的总和，但必须只使用两个变量！你能找到方法吗？

嗯嗯……你现在在想什么很明显：“我可以用两个变量做的唯一一件事就是读取变量 x 中的一个值，然后将该值赋给变量 total”。你的想法相当正确，并且在这里进行了展示。

x = float(input())  # 读取第一个数字

total = x

这可以等价地写成

total = 0

x = float(input())  # 读取第一个数字

total = total + x

那接下来呢？现在，你可以做三件事，那就是：思考，思考，当然，还是思考！

第一个用户提供的数字已经被存储在变量 total 中，因此变量 x 现在可以用于进一步使用！因此，你可以重用变量 x 来读取第二个值，该值也将累积在变量 total 中，如下所示。

total = 0

x = float(input())  # 读取第一个数字

total = total + x

x = float(input())  # 读取第二个数字

total = total + x

![](img/notice.jpg)语句 total = total + x 将 x 的值累加到 total 中，这意味着它将 x 的值加到 total 上，包括 total 中之前存在的任何值。例如，如果变量 total 包含值 5，而变量 x 包含值 3，则语句 total = total + x 将值 8 赋给变量 total。

由于第二个用户提供的数字已经累积在变量 total 中，变量 x 可以被重用！这个过程可以重复，直到所有四个数字都被读取并累积到变量 total 中。最终的代码如下。请注意，它还没有使用任何循环控制结构！

total = 0

x = float(input())

total = total + x

x = float(input())

total = total + x

x = float(input())

total = total + x

x = float(input())

total = total + x

print(total)

![](img/notice.jpg)这两个代码和本节开头的一个初始代码被认为是等效的。然而，它们之间的主要区别在于，这个代码包含四对相同的语句。

显然，你可以使用这个例子来读取并找到超过四个数字的总和。然而，多次编写这些语句可能会相当繁琐，并且如果任何一对语句意外遗漏，可能会导致错误。

你真正需要的是保留一对语句，但使用循环控制结构执行四次（或者如果你愿意，甚至 1000 次）。你可以使用以下代码片段。

total = 0

execute_these_statements_4_times:

x = float(input())

total = total + x

print(total)

显然，Python 中没有 execute_these_statements_4_times 这样的语句。这只是为了演示目的，但很快你将了解 Python 支持的所有循环控制结构！

### 23.3 复习问题：正确/错误

对于以下每个陈述，选择正确或错误。

1)循环控制结构是一种允许在满足指定条件之前多次执行语句或语句块的构造。

2)可以使用一个序列控制结构提示用户输入 1000 个数字，然后计算它们的总和。

3)以下代码片段将值 10 累加到变量 total 中。

total = 10

a = 0

total = total + a

4)以下 Python 程序（不是代码片段）满足有效性的属性。

a = 5

total = total + a

print(total)

5)以下两个代码片段都将值 5 赋给变量 total。

a = 5

total = a

total = 0

a = 5

total = total + a
