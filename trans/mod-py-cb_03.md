# 第三章。函数定义

在本章中，我们将看一下以下配方：

+   设计带有可选参数的函数

+   使用超灵活的关键字参数

+   使用*分隔符强制关键字参数

+   在函数参数上写明确的类型

+   基于部分函数选择参数顺序

+   使用 RST 标记编写清晰的文档字符串

+   围绕 Python 的堆栈限制设计递归函数

+   使用脚本库开关编写可重用脚本

# 介绍

函数定义是将一个大问题分解为较小问题的一种方式。数学家们已经做了几个世纪了。这也是将我们的 Python 编程打包成智力可管理的块的一种方式。

在这些配方中，我们将看一些函数定义技术。这将包括处理灵活参数的方法以及根据一些更高级别的设计原则组织参数的方法。

我们还将看一下 Python 3.5 的 typing 模块以及如何为我们的函数创建更正式的注释。我们可以开始使用`mypy`项目，以对数据类型的使用进行更正式的断言。

# 设计带有可选参数的函数

当我们定义一个函数时，通常需要可选参数。这使我们能够编写更灵活的函数，并且可以在更多情况下使用。

我们也可以将这看作是创建一系列密切相关函数的一种方式，每个函数具有略有不同的参数集合 - 称为**签名** - 但都共享相同的简单名称。许多函数共享相同的名称的想法可能有点令人困惑。因此，我们将更多地关注可选参数的概念。

可选参数的一个示例是`int()`函数。它有两种形式：

+   `int(str)`: 例如，`int('355')`的值为`355`。在这种情况下，我们没有为可选的`base`参数提供值；使用了默认值`10`。

+   `int(str, base)`：例如，`int('0x163', 16)`的值是`355`。在这种情况下，我们为`base`参数提供了一个值。

## 准备工作

许多游戏依赖于骰子的集合。赌场游戏*Craps*使用两个骰子。像*Zilch*（或*Greed*或*Ten Thousand*）这样的游戏使用六个骰子。游戏的变体可能使用更多。

拥有一个可以处理所有这些变化的掷骰子函数非常方便。我们如何编写一个骰子模拟器，可以处理任意数量的骰子，但是将使用两个作为方便的默认值？

## 如何做...

我们有两种方法来设计带有可选参数的函数：

+   **一般到特定**：我们首先设计最一般的解决方案，并为最常见的情况提供方便的默认值。

+   **特定到一般**：我们首先设计几个相关的函数。然后将它们合并为一个涵盖所有情况的一般函数，将原始函数中的一个单独出来作为默认行为。

### 从特定到一般的设计

在遵循特定到一般策略时，我们将设计几个单独的函数，并寻找共同的特征：

1.  编写函数的一个版本。我们将从*Craps*游戏开始，因为它似乎最简单：

```py
     **>>> import random** 

     **>>> def die():** 

     **...    return random.randint(1,6)** 

     **>>> def craps():** 

     **...    return (die(), die())** 

    ```

我们定义了一个方便的辅助函数`die()`，它封装了有时被称为标准骰子的基本事实。有五个可以使用的立体几何体，可以产生四面体、六面体、八面体、十二面体和二十面体骰子。六面骰子有着悠久的历史，最初是作为*骰子*骨头，很容易修剪成六面立方体。

这是底层`die()`函数的一个示例：

```py
     **>>> random.seed(113)** 

     **>>> die(), die()** 

     **(1, 6)** 

    ```

我们掷了两个骰子，以展示值如何组合以掷更大堆的骰子。

我们的*Craps*游戏函数看起来是这样的：

```py
     **>>> craps()** 

     **(6, 3)** 

     **>>> craps()** 

     **(1, 4)** 

    ```

这显示了*Craps*游戏的一些两个骰子投掷。

1.  编写函数的另一个版本：

```py
     **>>> def zonk():** 

     **...    return tuple(die() for x in range(6))** 

    ```

我们使用了一个生成器表达式来创建一个有六个骰子的元组对象。我们将在第八章中深入研究生成器表达式，*函数式和反应式编程特性*。

我们的生成器表达式有一个变量`x`，它被忽略了。通常也可以看到这样写成`tuple(die() for _ in range(6))`。变量`_`是一个有效的 Python 变量名；这个名字可以作为一个提示，表明我们永远不想看到这个变量的值。

这是使用`zonk()`函数的一个例子：

```py
     **>>> zonk()** 

     **(5, 3, 2, 4, 1, 1)** 

    ```

这显示了六个单独骰子的结果。有一个短顺（1-5）以及一对一。在游戏的某些版本中，这是一个很好的得分手。

1.  找出两个函数中的共同特征。这可能需要对各种函数进行一些重写，以找到一个共同的设计。在许多情况下，我们最终会引入额外的变量来替换常数或其他假设。

在这种情况下，我们可以将两元组的创建概括化。我们可以引入一个基于`range(2)`的生成器表达式，它将两次评估`die()`函数：

```py
     **>>> def craps():** 

     **...     return tuple(die() for x in range(2))** 

    ```

这似乎比解决特定的两个骰子问题需要更多的代码。从长远来看，使用一个通用函数意味着我们可以消除许多特定的函数。

1.  合并这两个函数。这通常涉及到暴露一个之前是常数或其他硬编码假设的变量：

```py
     **>>> def dice(n):** 

     **...     return tuple(die() for x in range(n))** 

    ```

这提供了一个通用函数，涵盖了*Craps*和*Zonk*的需求：

```py
     **>>> dice(2)** 

     **(3, 2)** 

     **>>> dice(6)** 

     **(5, 3, 4, 3, 3, 4)** 

    ```

1.  确定最常见的用例，并将其作为引入的任何参数的默认值。如果我们最常见的模拟是*Craps*，我们可能会这样做：

```py
     **>>> def dice(n=2):** 

     **...     return tuple(die() for x in range(n))** 

    ```

现在我们可以简单地在*Craps*中使用`dice()`。我们需要在*Zonk*中使用`dice(6)`。

### 从一般到特殊的设计

在遵循从一般到特殊的策略时，我们会首先确定所有的需求。我们通常会通过在需求中引入变量来做到这一点：

1.  总结掷骰子的需求。我们可能有一个像这样的列表：

+   *Craps*：两个骰子。

+   *Zonk*中的第一次掷骰子：六个骰子。

+   *Zonk*中的后续掷骰子：一到六个骰子。

这个需求列表显示了掷*n*个骰子的一个共同主题。

1.  用一个显式参数重写需求，代替任何字面值。我们将用参数*n*替换所有的数字，并展示我们引入的这个新参数的值：

+   *Craps*：*n*个骰子，其中*n=2*。

+   *Zonk*中的第一次掷骰子：*n*个骰子，其中*n=6*。

+   *Zonk*中的后续掷骰子：*n*个骰子，其中*1≤n≤6*。

这里的目标是确保所有的变化确实有一个共同的抽象。在更复杂的问题中，看似相似的东西可能没有一个共同的规范。

我们还希望确保我们已经正确地对各种函数进行了参数化。在更复杂的情况下，我们可能有一些不需要被参数化的值；它们可以保持为常数。

1.  编写符合一般模式的函数：

```py
     **>>> def dice(n):** 

     **...    return (die() for x in range(n))** 

    ```

在第三种情况下——*Zonk*中的后续掷骰子——我们确定了一个*1≤n≤6*的约束。我们需要确定这是否是我们`dice()`函数的约束，还是这个约束是由使用`dice`函数的模拟应用所施加的。

在这种情况下，约束是不完整的。*Zonk*的规则要求没有被掷动的骰子形成某种得分模式。约束不仅仅是骰子的数量在一到六之间；约束与游戏状态有关。似乎没有充分的理由将`dice()`函数与游戏状态联系起来。

1.  为最常见的用例提供一个默认值。如果我们最常见的模拟是*Craps*，我们可能会这样做：

```py
     **>>> def dice(n=2):** 

     **...     return tuple(die() for x in range(n))** 

    ```

现在我们可以简单地在*Craps*中使用`dice()`。我们需要在*Zonk*中使用`dice(6)`。

## 工作原理...

Python 提供参数值的规则非常灵活。有几种方法可以确保每个参数都有一个值。我们可以将其视为以下方式工作：

1.  将每个参数设置为任何提供的默认值。

1.  对于没有名称的参数，参数值是按位置分配给参数的。

1.  对于具有名称的参数，例如`dice(n=2)`，参数值是使用名称分配的。通过位置和名称同时分配参数是错误的。

1.  如果任何参数没有值，这是一个错误。

这些规则允许我们根据需要提供默认值。它们还允许我们混合位置值和命名值。默认值的存在是使参数可选的原因。

可选参数的使用源于两个考虑因素：

+   我们可以对处理进行参数化吗？

+   该参数的最常见参数值是什么？

在流程定义中引入参数可能是具有挑战性的。在某些情况下，有代码可以帮助我们用参数替换文字值（例如 2 或 6）。

然而，在某些情况下，文字值不需要被参数替换。它可以保留为文字值。我们并不总是想用参数替换每个文字值。例如，我们的`die()`函数有一个文字值为 6，因为我们只对标准的立方骰子感兴趣。这不是一个参数，因为我们不认为有必要制作更一般的骰子。

## 还有更多...

如果我们想非常彻底，我们可以编写专门的版本函数，这些函数是我们更通用的函数的专门版本。这些函数可以简化应用程序：

```py
 **>>> def craps():** 

 **...     return dice(2)** 

 **>>> def zonk():** 

 **...     return dice(6)** 

```

我们的应用程序功能-`craps()`和`zonk()`-依赖于一个通用函数`dice()`。这又依赖于另一个函数`die()`。我们将在*基于部分函数选择参数顺序*食谱中重新讨论这个想法。

这个依赖堆栈中的每一层都引入了一个方便的抽象，使我们不必理解太多细节。这种分层抽象的想法有时被称为**chunking**。这是一种通过隔离细节来管理复杂性的方法。

这种设计模式的常见扩展是在这个函数层次结构中的多个级别提供参数。如果我们想要对`die()`函数进行参数化，我们将为`dice()`和`die()`提供参数。

对于这种更复杂的参数化，我们需要在我们的层次结构中引入更多具有默认值的参数。我们将从`die()`中添加一个参数开始。这个参数必须有一个默认值，这样我们就不会破坏我们现有的测试用例：

```py
 **>>> def die(sides=6):** 

 **...     return random.randint(1,6)** 

```

在引入这个参数到抽象堆栈的底部之后，我们需要将这个参数提供给更高级别的函数：

```py
 **>>> def dice(n=2, sides=6):** 

 **... return tuple(die(sides) for x in range(n))** 

```

我们现在有很多种使用`dice()`函数的方法：

+   所有默认值：`dice()`很好地覆盖了*Craps*。

+   所有位置参数：`dice(6, 6)`将覆盖*Zonk*。

+   位置和命名参数的混合：位置值必须首先提供，因为顺序很重要。例如，`dice(2, sides=8)`将覆盖使用两个八面体骰子的游戏。

+   所有命名参数：`dice(sides=4, n=4)`这将处理我们需要模拟掷四个四面体骰子的情况。在使用所有命名参数时，顺序并不重要。

在这个例子中，我们的函数堆栈只有两层。在更复杂的应用程序中，我们可能需要在层次结构的许多层中引入参数。

## 另请参阅

+   我们将在*基于部分函数选择参数顺序*食谱中扩展一些这些想法。

+   我们使用了涉及不可变对象的可选参数。在这个配方中，我们专注于数字。在第四章中，*内置数据结构-列表、集合、字典*，我们将研究可变对象，它们具有可以更改的内部状态。在*避免函数参数的可变默认值*配方中，我们将研究一些重要的额外考虑因素，这些因素对于设计具有可变对象的可选值的函数非常重要。

# 使用超级灵活的关键字参数

一些设计问题涉及解决一个未知的简单方程，给定足够的已知值。例如，速率、时间和距离之间有一个简单的线性关系。我们可以解决任何一个，只要知道另外两个。以下是我们可以用作示例的三条规则：

+   *d = r* × *t*

+   *r = d / t*

+   *t = d / r*

在设计电路时，例如，基于欧姆定律使用了一组类似的方程。在这种情况下，方程将电阻、电流和电压联系在一起。

在某些情况下，我们希望提供一个简单、高性能的软件实现，可以根据已知和未知的情况执行三种不同的计算中的任何一种。我们不想使用通用的代数框架；我们想将三个解决方案捆绑到一个简单、高效的函数中。

## 准备工作

我们将构建一个单一函数，可以通过体现任意两个已知值的三个解来解决**速率-时间-距离**（**RTD**）计算。通过微小的变量名称更改，这适用于令人惊讶的许多现实世界问题。

这里有一个技巧。我们不一定想要一个单一的值答案。我们可以通过创建一个包含三个值的小 Python 字典来稍微概括这一点。我们将在第四章中更多地了解字典。

当出现问题时，我们将使用`warnings`模块而不是引发异常：

```py
 **>>> import warnings** 

```

有时，产生一个有疑问的结果比停止处理更有帮助。

## 如何做...

解出每个未知数的方程。我们先前已经展示了这一点，例如*d = r * t*，RTD 计算：

1.  这导致了三个单独的表达式：

+   距离=速率*时间

+   速率=距离/时间

+   时间=距离/速率

1.  根据一个值为`None`时未知的情况，将每个表达式包装在一个`if`语句中：

```py
            if distance is None:
                distance = rate * time
            elif rate is None:
                rate = distance / time
            elif time is None:
                time = distance / rate

    ```

1.  参考第二章中的*设计复杂的 if...elif 链*，*语句和语法*，以指导设计这些复杂的`if...elif`链。包括`else`崩溃选项的变体：

```py
            else:
                warnings.warning( "Nothing to solve for" )

    ```

1.  构建生成的字典对象。在简单情况下，我们可以使用`vars()`函数简单地将所有本地变量作为生成的字典发出。在某些情况下，我们可能有一些本地变量不想包括；在这种情况下，我们需要显式构建字典：

```py
            return dict(distance=distance, rate=rate, time=time)

    ```

1.  使用关键字参数将所有这些包装为一个函数：

```py
            def rtd(distance=None, rate=None, time=None):
                if distance is None:
                    distance = rate * time
                elif rate is None:
                    rate = distance / time
                elif time is None:
                    time = distance / rate
                else:
                    warnings.warning( "Nothing to solve for" )
                return dict(distance=distance, rate=rate, time=time)

    ```

我们可以像这样使用生成的函数：

```py
 **>>> def rtd(distance=None, rate=None, time=None):
...     if distance is None:
...         distance = rate * time
...     elif rate is None:
...         rate = distance / time
...     elif time is None:
...         time = distance / rate
...     else:
...         warnings.warning( "Nothing to solve for" )
...     return dict(distance=distance, rate=rate, time=time)
>>> rtd(distance=31.2, rate=6) 
{'distance': 31.2, 'time': 5.2, 'rate': 6}** 

```

这告诉我们，以 6 节的速率行驶 31.2 海里将需要 5.2 小时。

为了得到格式良好的输出，我们可以这样做：

```py
 **>>> result= rtd(distance=31.2, rate=6)** 

 **>>> ('At {rate}kt, it takes '** 

 **... '{time}hrs to cover {distance}nm').format_map(result)** 

 **'At 6kt, it takes 5.2hrs to cover 31.2nm'** 

```

为了打破长字符串，我们使用了第二章中的*设计复杂的 if...elif 链*。

## 工作原理...

因为我们为所有参数提供了默认值，所以我们可以为三个参数中的两个提供参数值，然后函数就可以解决第三个参数。这样可以避免我们编写三个单独的函数。

将字典作为最终结果返回并不是必要的。这只是方便。它允许我们无论提供了哪些参数值，都有一个统一的结果。

## 还有更多...

我们有另一种表述，涉及更多的灵活性。Python 函数有一个*所有其他关键字*参数，前缀为`**`。通常显示如下：

```py
    def rtd2(distance, rate, time, **keywords): 
        print(keywords) 

```

任何额外的关键字参数都会被收集到提供给`**keywords`参数的字典中。然后我们可以用额外的参数调用这个函数。像这样评估这个函数：

```py
    rtd2(rate=6, time=6.75, something_else=60) 

```

然后我们会看到`keywords`参数的值是一个带有`{'something_else': 60}`值的字典对象。然后我们可以对这个结构使用普通的字典处理技术。这个字典中的键和值是在函数被评估时提供的名称和值。

我们可以利用这一点，并坚持要求所有参数都提供关键字：

```py
    def rtd2(**keywords): 
        rate= keywords.get('rate', None) 
        time= keywords.get('time', None) 
        distance= keywords.get('distance', None) 
        etc. 

```

这个版本使用字典`get()`方法在字典中查找给定的键。如果键不存在，则提供`None`的默认值。

（返回`None`的默认值是`get()`方法的默认行为。我们的示例包含一些冗余，以阐明处理过程。对于一些非常复杂的情况，我们可能有除`None`之外的默认值。）

这有可能具有稍微更灵活的优势。它可能的缺点是使实际参数名称非常难以辨别。

我们可以遵循*使用 RST 标记编写清晰文档字符串*的配方，并提供一个良好的文档字符串。然而，通过文档隐式地提供参数名称似乎更好一些。

## 另请参阅

+   我们将查看*使用 RST 标记编写清晰文档字符串*配方中函数的文档

![](img/614271.jpg)

# 使用*分隔符强制使用关键字参数

有些情况下，我们需要将大量的位置参数传递给函数。也许我们遵循了*设计具有可选参数的函数*的配方，这导致我们设计了一个参数如此之多的函数，以至于变得令人困惑。

从实用的角度来看，一个具有超过三个参数的函数可能会令人困惑。大量的传统数学似乎集中在一个和两个参数函数上。似乎没有太多常见的数学运算符涉及三个或更多的操作数。

当难以记住参数的所需顺序时，参数太多了。

## 准备工作

我们将查看一个具有大量参数的函数。我们将使用一个准备风冷表并将数据写入 CSV 格式输出文件的函数。

我们需要提供一系列温度、一系列风速以及我们想要创建的文件的信息。这是很多参数。

基本公式是这样的：

*T[wc]* ( *T[a]*, V* ) = 13.12 + 0.6215 *T[a]* - 11.37 *V* ^(0.16) + 0.3965 *T[a] V* ^(0.16)

风冷温度，*T[wc]*，基于空气温度，*T[a]*，以摄氏度为单位，以及风速，*V*，以 KPH 为单位。

对于美国人来说，这需要一些转换：

+   从°F 转换为°C：*C* = 5( *F* -32) / 9

+   将风速从 MPH，*V[m]*，转换为 KPH，*V[k]*：*V[k] = V[m]* × 1.609344

+   结果需要从°C 转换回°F：*F* = 32 + *C* (9/5)

我们不会将这些纳入这个解决方案。我们将把这留给读者作为一个练习。

创建风冷表的一种方法是创建类似于这样的东西：

```py
    import pathlib 

    def Twc(T, V): 
        return 13.12 + 0.6215*T - 11.37*V**0.16 + 0.3965*T*V**0.16 

    def wind_chill(start_T, stop_T, step_T, 
        start_V, stop_V, step_V, path): 
        """Wind Chill Table.""" 
        with path.open('w', newline='') as target: 
            writer= csv.writer(target) 
            heading = [None]+list(range(start_T, stop_T, step_T)) 
            writer.writerow(heading) 
            for V in range(start_V, stop_V, step_V): 
                row = [V] + [Twc(T, V) 
                    for T in range(start_T, stop_T, step_T)] 
                writer.writerow(row) 

```

我们使用`with`上下文打开了一个输出文件。这遵循了第二章中的*使用 with 语句管理上下文*配方，*语句和语法*。在这个上下文中，我们为 CSV 输出文件创建了一个写入。我们将在第九章中更深入地研究这个问题，*输入/输出、物理格式、逻辑布局*。

我们使用表达式`[None]+list(range(start_T, stop_T, step_T)`，创建了一个标题行。这个表达式包括一个列表文字和一个生成器表达式，用于构建一个列表。我们将在第四章中查看列表，*内置数据结构-列表、集合、字典*。我们将在第八章中查看生成器表达式，*函数式和响应式编程特性*。

同样，表格的每个单元格都是由一个生成器表达式构建的，`[Twc(T, V) for T in range(start_T, stop_T, step_T)]`。这是一个构建列表对象的理解。列表由风冷函数`Twc()`计算的值组成。我们根据表中的行提供风速。我们根据表中的列提供温度。

虽然细节涉及前瞻性部分，`def`行提出了一个问题。这个`def`行非常复杂。

这种设计的问题在于`wind_chill()`函数有七个位置参数。当我们尝试使用这个函数时，我们得到以下代码：

```py
    import pathlib 
    p=pathlib.Path('code/wc.csv') 
    wind_chill(0,-45,-5,0,20,2,p) 

```

所有这些数字是什么？有没有什么可以帮助解释这行代码的意思？

## 如何做到...

当我们有大量参数时，使用关键字参数而不是位置参数会有所帮助。

在 Python 3 中，我们有一种强制使用关键字参数的技术。我们可以使用`*`作为两组参数之间的分隔符：

1.  在`*`之前，我们列出可以*或*按关键字命名的参数值。在这个例子中，我们没有这些参数。

1.  在`*`之后，我们列出必须使用关键字给出的参数值。对于我们的示例，这是所有的参数。

对于我们的示例，生成的函数如下：

```py
    def wind_chill(*, start_T, stop_T, step_T, start_V, stop_V, step_V, path): 

```

当我们尝试使用令人困惑的位置参数时，我们会看到这个：

```py
 **>>> wind_chill(0,-45,-5,0,20,2,p) 
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module> 
TypeError: wind_chill() takes 0 positional arguments but 7 were given** 

```

我们必须按以下方式使用该函数：

```py
    wind_chill(start_T=0, stop_T=-45, step_T=-5, 
        start_V=0, stop_V=20, step_V=2, 
        path=p) 

```

强制使用必填关键字参数的用法迫使我们每次使用这个复杂函数时都写出清晰的语句。

## 它是如何工作的...

`*`字符在函数定义中有两个含义：

+   它作为一个特殊参数的前缀，接收所有未匹配的位置参数。我们经常使用`*args`将所有位置参数收集到一个名为`args`的单个参数中。

+   它被单独使用，作为可以按位置应用的参数和必须通过关键字提供的参数之间的分隔符。

`print()`函数就是一个例子。它有三个仅限关键字参数，用于输出文件、字段分隔符字符串和行结束字符串。

## 还有更多...

当然，我们可以将此技术与各种参数的默认值结合使用。例如，我们可以对此进行更改：

```py
    import sys 
    def wind_chill(*, start_T, stop_T, step_T, start_V, stop_V, step_V, output=sys.stdout): 

```

现在我们可以以两种方式使用这个函数：

+   这是在控制台上打印表的方法：

```py
                wind_chill( 
                    start_T=0, stop_T=-45, step_T=-5, 
                    start_V=0, stop_V=20, step_V=2) 

    ```

+   这是写入文件的方法：

```py
                path = pathlib.Path("code/wc.csv") 
                with path.open('w', newline='') as target: 
                    wind_chill(output=target, 
                        start_T=0, stop_T=-45, step_T=-5, 
                        start_V=0, stop_V=20, step_V=2) 

    ```

我们在这里改变了方法，稍微更加通用。这遵循了*设计具有可选参数的函数*配方。

## 另请参阅

+   查看*基于部分函数选择参数顺序*配方，了解此技术的另一个应用

# 在函数参数上写明确的类型

Python 语言允许我们编写完全与数据类型相关的函数（和类）。以这个函数为例：

```py
    def temperature(*, f_temp=None, c_temp=None): 
        if c_temp is None: 
            return {'f_temp': f_temp, 'c_temp': 5*(f_temp-32)/9} 
        elif f_temp is None: 
            return {'f_temp': 32+9*c_temp/5, 'c_temp': c_temp} 
        else: 
            raise Exception("Logic Design Problem") 

```

这遵循了之前展示的三个配方：*使用超灵活的关键字参数*，*使用本章的*分隔符强制关键字参数*，以及*设计复杂的 if...elif 链*来自第二章，*语句和语法*。

这个函数将适用于任何数值类型的参数值。实际上，它将适用于任何实现`+`、`-`、`*`和`/`运算符的数据结构。

有时我们不希望我们的函数完全通用。在某些情况下，我们希望对数据类型做出更强的断言。虽然我们有时关心数据类型，但我们不想编写大量看起来像这样的代码：

```py
    from numbers import Number 
    def c_temp(f_temp): 
        assert isinstance(F, Number) 
        return 5*(f_temp-32)/9 

```

这引入了额外的`assert`语句的性能开销。它还会用一个通常应该重申显而易见的语句来使我们的程序混乱。

此外，我们不能依赖文档字符串进行测试。这是推荐的风格：

```py
    def temperature(*, f_temp=None, c_temp=None): 
        """Convert between Fahrenheit temperature and 
        Celsius temperature. 

        :key f_temp: Temperature in °F. 
        :key c_temp: Temperature in °C. 
        :returns: dictionary with two keys: 
            :f_temp: Temperature in °F. 
            :c_temp: Temperature in °C. 
        """

```

文档字符串不允许进行任何自动化测试来确认文档实际上是否与代码匹配。两者可能不一致。

我们想要的是关于涉及的数据类型的提示，可以用于测试和确认，但不会影响性能。我们如何提供有意义的类型提示？

## 准备工作

我们将实现`temperature()`函数的一个版本。我们将需要两个模块，这些模块将帮助我们提供关于参数和返回值的数据类型的提示：

```py
    from typing import * 

```

我们选择从`typing`模块导入所有名称。如果我们要提供类型提示，我们希望它们简洁。写`typing.List[str]`很尴尬。我们更喜欢省略模块名称。

我们还需要安装最新版本的`mypy`。这个项目正在快速发展。与其使用`pip`程序从 PyPI 获取副本，最好直接从 GitHub 存储库[`github.com/JukkaL/mypy`](https://github.com/JukkaL/mypy)下载最新版本。

说明中说，*目前，PyPI 上的 mypy 版本与 Python 3.5 不兼容。如果你使用 Python 3.5，请直接从 git 安装*。

```py
 **$ pip3 install git+git://github.com/JukkaL/mypy.git** 

```

`mypy`工具可用于分析我们的 Python 程序，以确定类型提示是否与实际代码匹配。

## 如何做...

Python 3.5 引入了语言类型提示。我们可以在三个地方使用它们：函数参数、函数返回和类型提示注释：

1.  为各种数字定义一个方便的类型：

```py
            from decimal import Decimal 
            from typing import * 
            Number = Union[int, float, complex, Decimal] 

    ```

理想情况下，我们希望在 numbers 模块中使用抽象的`Number`类。目前，该模块没有可用的正式类型规范，因此我们将为`Number`定义自己的期望。这个定义是几种数字类型的联合。理想情况下，`mypy`或 Python 的未来版本将包括所需的定义。

1.  像这样注释函数的参数：

```py
            def temperature(*, 
                f_temp: Optional[Number]=None, 
                c_temp: Optional[Number]=None): 

    ```

我们在参数的一部分添加了`:`和类型提示。在这种情况下，我们使用我们自己的`Number`类型定义来声明任何数字都可以在这里。我们将其包装在`Optional[]`类型操作中，以声明参数值可以是`Number`或`None`。

1.  函数的返回值可以这样注释：

```py
            def temperature(*, 
                f_temp: Optional[Number]=None, 
                c_temp: Optional[Number]=None) -> Dict[str, Number]: 

    ```

我们为此函数的返回值添加了`->`和类型提示。在这种情况下，我们声明结果将是一个具有字符串键`str`和使用我们的`Number`类型定义的数字值的字典对象。

`typing`模块引入了类型提示名称，例如`Dict`，我们用它来解释函数的结果。这与实际构建对象的`dict`类不同。`typing.Dict`只是一个提示。

1.  如果需要的话，我们可以在赋值和`with`语句中添加类型提示作为注释。这些很少需要，但可能会澄清一长串复杂的语句。如果我们想要添加它们，注释可能看起来像这样：

```py
            result = {'c_temp': c_temp, 
                'f_temp': f_temp} # type: Dict[str, Number] 

    ```

我们在构建最终字典对象的语句上添加了`# type: Dict[str, Number]`。

## 工作原理...

我们添加的类型信息称为**提示**。它们不是 Python 编译器以某种方式检查的要求。它们在运行时也不会被检查。

类型提示由一个名为`mypy`的独立程序使用。有关更多信息，请参见[`mypy-lang.org`](http://mypy-lang.org)。

`mypy`程序检查 Python 代码，包括类型提示。它应用一些形式推理和推断技术，以确定各种类型提示是否对 Python 程序可以处理的任何数据为“真”。

对于更大更复杂的程序，`mypy`的输出将包括描述代码本身或装饰代码的类型提示可能存在问题的警告和错误。

例如，这是一个容易犯的错误。我们假设我们的函数返回一个单一的数字。然而，我们的返回语句与我们的期望不匹配：

```py
    def temperature_bad(*, 
        f_temp: Optional[Number]=None, 
        c_temp: Optional[Number]=None) -> Number: 

        if c_temp is None: 
            c_temp = 5*(f_temp-32)/9 
        elif f_temp is None: 
            f_temp = 32+9*c_temp/5 
        else: 
            raise Exception( "Logic Design Problem" ) 
        result = {'c_temp': c_temp, 
            'f_temp': f_temp} # type: Dict[str, Number] 
        return result 

```

当我们运行`mypy`时，我们会看到这个：

```py
    ch03_r04.py: note: In function "temperature_bad": 
    ch03_r04.py:37: error: Incompatible return value type: 
        expected Union[builtins.int, builtins.float, builtins.complex, decimal.Decimal], 
        got builtins.dict[builtins.str, 
        Union[builtins.int, builtins.float, builtins.complex, decimal.Decimal]] 

```

我们可以看到我们的`Number`类型名称在错误消息中被扩展为`Union[builtins.int, builtins.float, builtins.complex, decimal.Decimal]`。更重要的是，我们可以看到在第 37 行，`return`语句与函数定义不匹配。

考虑到这个错误，我们需要修复返回值或定义，以确保期望的类型和实际类型匹配。目前不清楚哪个是“正确”的。以下任一种可能是意图：

+   计算并返回单个值：这意味着需要有两个`return`语句，取决于计算了哪个值。在这种情况下，没有理由构建`result`字典对象。

+   返回字典对象：这意味着我们需要更正`def`语句以具有正确的返回类型。更改这可能会对其他期望`temperature`返回`Number`实例的函数产生连锁变化。

参数和返回值的额外语法对运行时没有真正影响，只有在源代码首次编译成字节码时才会有很小的成本。它们毕竟只是提示。

## 还有更多...

在使用内置类型时，我们经常可以创建复杂的结构。例如，我们可能有一个字典，将三个整数的元组映射到字符串列表：

```py
    a = {(1, 2, 3): ['Poe', 'E'], 
         (3, 4, 5): ['Near', 'a', 'Raven'], 
        } 

```

如果这是函数的结果，我们如何描述这个？

我们将创建一个相当复杂的类型表达式，总结每个结构层次：

```py
Dict[Tuple[int, int, int], List[str]] 

```

我们总结了一个将一个类型`Tuple[int, int, int]`映射为另一个类型`List[str]`的字典。这捕捉了几种内置类型如何组合以构建复杂的数据结构。

在这种情况下，我们将三个整数的元组视为一个匿名元组。在许多情况下，它不仅仅是一个通用元组，它实际上是一个被建模为元组的 RGB 颜色。也许字符串列表实际上是来自更长文档的一行文本，已经根据空格拆分成单词。

在这种情况下，我们应该做如下操作：

```py
Color = Tuple[int, int, int] 
Line = List[str] 
Dict[Color, Line] 

```

创建我们自己的应用程序特定类型名称可以极大地澄清使用内置集合类型执行的处理。

## 另请参阅

+   有关类型提示的更多信息，请参见[`www.python.org/dev/peps/pep-0484/`](https://www.python.org/dev/peps/pep-0484/)。

+   有关当前`mypy`项目，请参见[`github.com/JukkaL/mypy`](https://github.com/JukkaL/mypy)。

+   有关`mypy`如何与 Python 3 一起工作的文档，请参见[`www.mypy-lang.org`](http://www.mypy-lang.org)。

# 基于部分函数选择参数顺序

当我们查看复杂的函数时，有时我们会看到我们使用函数的方式有一个模式。例如，我们可能多次评估一个函数，其中一些参数值由上下文固定，而其他参数值随着处理的细节而变化。

如果我们的设计反映了这一点，它可以简化我们的编程。我们希望提供一种使常见参数比不常见参数更容易处理的方法。我们也希望避免重复大上下文中的参数。

## 准备就绪

我们将看一个 haversine 公式的版本。这计算地球表面上点之间的距离，使用该点的纬度和经度坐标：

![准备就绪](img/Image00005.jpg)

*c* = 2 *arc sin(√a)*

基本的计算得出了两点之间的中心角*c*。角度以弧度表示。我们通过将其乘以地球的平均半径来将其转换为距离。如果我们将角度*c*乘以半径为 3959 英里，距离，我们将角度转换为英里。

这是这个函数的一个实现。我们包括了类型提示：

```py
    from math import radians, sin, cos, sqrt, asin 

    MI= 3959 
    NM= 3440 
    KM= 6372 

    def haversine(lat_1: float, lon_1: float, 
        lat_2: float, lon_2: float, R: float) -> float: 
        """Distance between points. 

        R is Earth's radius. 
        R=MI computes in miles. Default is nautical miles. 

    >>> round(haversine(36.12, -86.67, 33.94, -118.40, R=6372.8), 5) 
    2887.25995 
    """ 
    Δ_lat = radians(lat_2) - radians(lat_1) 
    Δ_lon = radians(lon_2) - radians(lon_1) 
    lat_1 = radians(lat_1) 
    lat_2 = radians(lat_2) 

    a = sin(Δ_lat/2)**2 + cos(lat_1)*cos(lat_2)*sin(Δ_lon/2)**2 
    c = 2*asin(sqrt(a)) 

    return R * c 

```

### 注意

关于 doctest 示例的说明：

示例中的 doctest 使用了一个额外的小数点，这在其他地方没有使用。这样做是为了使这个示例与在线上的其他示例匹配。

地球不是球形的。在赤道附近，更精确的半径是 6378.1370 公里。在极地附近，半径是 6356.7523 公里。我们在常数中使用常见的近似值。

我们经常遇到的问题是，我们通常在一个单一的上下文中工作，并且我们将始终为`R`提供相同的值。例如，如果我们在海洋环境中工作，我们将始终使用`R = NM`来获得海里。

提供参数的一致值有两种常见的方法。我们将看看两种方法。

## 如何做...

在某些情况下，一个整体的上下文将为参数建立一个变量。这个值很少改变。提供参数的一致值有几种常见的方法。这涉及将函数包装在另一个函数中。有几种方法：

+   在一个新函数中包装函数。

+   创建一个偏函数。这有两个进一步的改进：

+   我们可以提供关键字参数

+   或者我们可以提供位置参数

我们将在这个配方中分别看看这些不同的变化。

### 包装一个函数

我们可以通过将一个通用函数包装在一个特定上下文的包装函数中来提供上下文值：

1.  使一些参数成为位置参数，一些参数成为关键字参数。我们希望上下文特征——很少改变的特征——成为关键字。更频繁更改的参数应该保持为位置参数。我们可以遵循*使用*分隔符强制关键字参数*的方法。

我们可能会将基本的 haversine 函数更改为这样：

```py
            def haversine(lat_1: float, lon_1: float, 
                lat_2: float, lon_2: float, *, R: float) -> float: 

    ```

我们插入了`*`来将参数分成两组。第一组可以通过位置或关键字提供参数。第二组，- 在这种情况下是`R` - 必须通过关键字给出。

1.  然后，我们可以编写一个包装函数，它将应用所有的位置参数而不加修改。它将作为长期上下文的一部分提供额外的关键字参数：

```py
            def nm_haversine(*args): 
                return haversine(*args, R=NM) 

    ```

我们在函数声明中使用了`*args`构造来接受一个单独的元组`args`中的所有位置参数值。当评估`haversine()`函数时，我们还使用了`*args`来将元组扩展为该函数的所有位置参数值。

### 使用关键字参数创建一个偏函数

偏函数是一个有一些参数值被提供的函数。当我们评估一个偏函数时，我们将之前提供的参数与额外的参数混合在一起。一种方法是使用关键字参数，类似于包装一个函数：

1.  我们可以遵循*使用*分隔符强制关键字参数*的方法。我们可能会将基本的 haversine 函数更改为这样：

```py
            def haversine(lat_1: float, lon_1: float, 
                lat_2: float, lon_2: float, *, R: float) -> float: 

    ```

1.  使用关键字参数创建一个偏函数：

```py
            from functools import partial 
            nm_haversine = partial(haversine, R=NM) 

    ```

`partial()`函数从现有函数和一组具体的参数值中构建一个新函数。`nm_haversine()`函数在构建偏函数时提供了`R`的特定值。

我们可以像使用任何其他函数一样使用它：

```py
 **>>> round(nm_haversine(36.12, -86.67, 33.94, -118.40), 2) 
1558.53** 

```

我们得到了一个海里的答案，这样我们就可以进行与船只相关的计算，而不必每次使用`haversine()`函数时都要耐心地检查它是否有`R=NM`作为参数。

### 使用位置参数创建一个偏函数

部分函数是一个具有一些参数值的函数。当我们评估部分函数时，我们正在提供额外的参数。另一种方法是使用位置参数。

如果我们尝试使用带有位置参数的`partial()`，我们只能在部分定义中提供最左边的参数值。这让我们想到函数的前几个参数可能被部分函数或包装器隐藏。

1.  我们可能会将基本的`haversine`函数更改为这样：

```py
            def haversine(R: float, lat_1: float, lon_1: float, 
                lat_2: float, lon_2: float) -> float: 

    ```

1.  使用位置参数创建一个部分函数：

```py
            from functools import partial 
            nm_haversine = partial(haversine, NM) 

    ```

`partial()`函数从现有函数和具体的参数值集构建一个新的函数。`nm_haversine()`函数在构建部分时为第一个参数`R`提供了一个特定的值。

我们可以像使用其他函数一样使用这个：

```py
 **>>> round(nm_haversine(36.12, -86.67, 33.94, -118.40), 2) 
1558.53** 

```

我们得到了一个海里的答案，这样我们就可以进行与航海有关的计算，而不必耐心地检查每次使用`haversine()`函数时是否有`R=NM`作为参数。

## 它是如何工作的...

部分函数本质上与包装函数相同。虽然它为我们节省了一行代码，但它有一个更重要的目的。我们可以在程序的其他更复杂的部分中自由构建部分函数。我们不需要使用`def`语句。

请注意，在查看位置参数的顺序时，创建部分函数会引起一些额外的考虑：

+   当我们使用`*args`时，它必须是最后一个。这是语言要求。这意味着在它前面的参数可以被具体识别，其余的都变成了匿名的，并且可以被一次性传递给包装函数。

+   在创建部分函数时，最左边的位置参数最容易提供一个值。

这两个考虑让我们将最左边的参数视为更多的上下文：这些预计很少改变。最右边的参数提供细节并经常改变。

## 还有更多...

还有第三种包装函数的方法——我们也可以构建一个`lambda`对象。这也可以工作：

```py
    nm_haversine = lambda *args: haversine(*args, R=NM) 

```

注意，`lambda`对象是一个被剥离了名称和主体的函数。它被简化为只有两个要素：

+   参数列表

+   一个单一的表达式是结果

`lambda`不能有任何语句。如果我们需要语句，我们需要使用`def`语句来创建一个包含名称和多个语句的定义。

## 另请参阅

+   我们还将在*使用脚本库开关编写可重用脚本*的配方中进一步扩展这个设计

# 使用 RST 标记编写清晰文档字符串

我们如何清楚地记录函数的作用？我们可以提供例子吗？当然可以，而且我们真的应该。在第二章中的*包括描述和文档*，*语句和语法*和*使用 RST 标记编写清晰文档字符串*的配方中，我们看到了一些基本的文档技术。这些配方介绍了**ReStructuredText**（**RST**）用于模块文档字符串。

我们将扩展这些技术，为函数文档字符串编写 RST。当我们使用 Sphinx 等工具时，我们函数的文档字符串将成为描述函数作用的优雅文档。

## 准备工作

在*使用*分隔符强制关键字参数*的配方中，我们看到了一个具有大量参数的函数和另一个只有两个参数的函数。

这是一个稍微不同版本的`Twc()`函数：

```py
 **>>> def Twc(T, V): 
...     """Wind Chill Temperature.""" 
...     if V < 4.8 or T > 10.0: 
...         raise ValueError("V must be over 4.8 kph, T must be below 10°C") 
...     return 13.12 + 0.6215*T - 11.37*V**0.16 + 0.3965*T*V**0.16** 

```

我们需要用更完整的文档来注释这个函数。

理想情况下，我们已经安装了 Sphinx 来看我们的劳动成果。请参阅[`www.sphinx-doc.org`](http://www.sphinx-doc.org)。

## 如何做...

通常我们会为函数描述写以下内容：

+   概要

+   描述

+   参数

+   返回

+   异常

+   测试案例

+   任何其他看起来有意义的东西

这是我们如何为一个函数创建良好文档的方法。我们可以应用类似的方法来为一个函数，甚至一个模块创建文档：

1.  写概要：不需要一个适当的主题——我们不写 *这个函数计算...* ；我们从 *计算...* 开始。没有理由过分强调上下文：

```py
            def Twc(T, V): 
                """Computes the wind chill temperature.""" 

    ```

1.  用详细描述写：

```py
            def Twc(T, V): 
                """Computes the wind chill temperature 

                The wind-chill, :math:`T_{wc}`, is based on 
                air temperature, T, and wind speed, V. 
                """ 

    ```

在这种情况下，我们在描述中使用了一小块排版数学。`:math:` 解释文本角色使用 LaTeX 数学排版。如果你安装了 LaTeX，Sphinx 将使用它来准备一个带有数学的小`.png`文件。如果你愿意，Sphinx 可以使用 MathJax 或 JSMath 来进行 JavaScript 数学排版，而不是创建一个`.png`文件。

1.  描述参数：对于位置参数，通常使用 `:param name: description` 。Sphinx 将容忍许多变化，但这是常见的。

对于必须是关键字的参数，通常使用 `:key name: description` 。使用 `key` 而不是 `param` 显示它是一个仅限关键字的参数：

```py
            def Twc(T: float, V: float): 
                """Computes the wind chill temperature 

                The wind-chill, :math:`T_{wc}`, is based on 
                air temperature, T, and wind speed, V. 

                :param T: Temperature in °C 
                :param V: Wind Speed in kph 
                """ 

    ```

有两种包含类型信息的方法：

+   使用 Python 3 类型提示

+   使用 RST `:type name:` 标记

我们通常不会同时使用这两种技术。类型提示比 RST `:type:` 标记更好。

1.  使用 `:returns:` 描述返回值：

```py
            def Twc(T: float, V: float) -> float: 
                """Computes the wind chill temperature 

                The wind-chill, :math:`T_{wc}`, is based on 
                air temperature, T, and wind speed, V. 

                :param T: Temperature in °C 
                :param V: Wind Speed in kph 
                :returns: Wind-Chill temperature in °C 
                """ 

    ```

有两种包含返回类型信息的方法：

+   使用 Python 3 类型提示

+   使用 RST `:rtype:` 标记

我们通常不会同时使用这两种技术。RST `:rtype:` 标记已被类型提示取代。

1.  确定可能引发的重要异常。使用 `:raises exception:` 原因标记。有几种可能的变化，但 `:raises exception:` 似乎最受欢迎：

```py
            def Twc(T: float, V: float) -> float: 
                """Computes the wind chill temperature 

                The wind-chill, :math:`T_{wc}`, is based on 
                air temperature, T, and wind speed, V. 

                :param T: Temperature in °C 
                :param V: Wind Speed in kph 
                :returns: Wind-Chill temperature in °C 
                :raises ValueError: for wind speeds under over 4.8 kph or T above 10°C 
                """ 

    ```

1.  如果可能的话，包括一个 doctest 测试用例：

```py
            def Twc(T: float, V: float) -> float: 
                """Computes the wind chill temperature 

                The wind-chill, :math:`T_{wc}`, is based on 
                air temperature, T, and wind speed, V. 

                :param T: Temperature in °C 
                :param V: Wind Speed in kph 
                :returns: Wind-Chill temperature in °C 
                :raises ValueError: for wind speeds under over 4.8 kph or T above 10°C 

                >>> round(Twc(-10, 25), 1) 
                -18.8 

                """ 

    ```

1.  写任何其他附加说明和有用信息。我们可以将以下内容添加到文档字符串中：

```py
                See https://en.wikipedia.org/wiki/Wind_chill 

                ..  math:: 

                    T_{wc}(T_a, V) = 13.12 + 0.6215 T_a - 11.37 V^{0.16} + 0.3965 T_a V^{0.16} 

    ```

我们已经包含了一个维基百科页面的参考，该页面总结了风冷计算并链接到更详细的信息。

我们还包括了一个带有函数中使用的 LaTeX 公式的 `.. math::` 指令。这将排版得很好，提供了代码的一个非常可读的版本。

## 它是如何工作的...

有关文档字符串的更多信息，请参见第二章中的*包括描述和文档* 配方，*语句和语法*。虽然 Sphinx 很受欢迎，但它并不是唯一可以从文档字符串注释中创建文档的工具。Python 标准库中的 pydoc 实用程序也可以从文档字符串注释中生成漂亮的文档。

Sphinx 工具依赖于`docutils`包中 RST 处理的核心功能。有关更多信息，请参见[`pypi.python.org/pypi/docutils`](https://pypi.python.org/pypi/docutils)。

RST 规则相对简单。这个配方中的大多数附加功能都利用了 RST 的*解释文本角色*。我们的每个 `:param T:` 、 `:returns:` 和 `:raises ValueError:` 结构都是一个文本角色。RST 处理器可以使用这些信息来决定内容的样式和结构。样式通常包括一个独特的字体。上下文可能是 HTML **定义列表**格式。

## 还有更多...

在许多情况下，我们还需要在函数和类之间包含交叉引用。例如，我们可能有一个准备风冷表的函数。这个函数可能有包含对 `Twc()` 函数的引用的文档。

Sphinx 将使用特殊的 `:func:` 文本角色生成这些交叉引用：

```py
    def wind_chill_table(): 
        """Uses :func:`Twc` to produce a wind-chill 
        table for temperatures from -30°C to 10°C and 
        wind speeds from 5kph to 50kph. 
        """ 

```

我们在 RST 文档中使用了 `:func:`Twc`` 来交叉引用一个函数。Sphinx 将把这些转换为适当的超链接。

## 另请参阅

+   有关 RST 工作的其他配方，请参见第二章中的*包括描述和文档* 和*在文档字符串中编写更好的 RST 标记* 配方。

# 围绕 Python 的堆栈限制设计递归函数

一些函数可以使用递归公式清晰而简洁地定义。有两个常见的例子：

阶乘函数：

![围绕 Python 的堆栈限制设计递归函数](img/Image00006.jpg)

计算斐波那契数的规则：

![围绕 Python 的堆栈限制设计递归函数](img/Image00007.jpg)

其中每个都涉及一个具有简单定义值的情况，以及涉及根据同一函数的其他值计算函数值的情况。

我们面临的问题是，Python 对这种递归函数定义的上限施加了限制。虽然 Python 的整数可以轻松表示*1000!*，但堆栈限制阻止我们随意这样做。

计算*F[n]*斐波那契数涉及一个额外的问题。如果我们不小心，我们会计算很多值超过一次：

*F[5] = F[4] + F[3]*

*F[5] = (F[3] + F[2] ) + (F[2] + F[1] )*

等等。

要计算*F[5]*，我们将计算*F[3]*两次，*F[2]*三次。这是非常昂贵的。

## 准备工作

许多递归函数定义遵循阶乘函数设定的模式。这有时被称为**尾递归**，因为递归情况可以写在函数体的尾部：

```py
def fact(n: int) -> int: 
    if n == 0: 
        return 1 
    return n*fact(n-1) 

```

函数中的最后一个表达式引用了具有不同参数值的函数。

我们可以重新陈述这一点，避免 Python 中的递归限制。

## 如何做...

尾递归也可以被描述为**归约**。我们将从一组值开始，然后将它们减少到一个单一的值：

1.  扩展规则以显示所有细节：

*n! = n* x *(n-* 1 *)* × *(n-* 2 *)* × *(n-* 3 *)...* × 1

1.  编写一个循环，枚举所有的值：

*N =* { *n, n-* 1 *, n-* 2 *, ...,* 1}在 Python 中，它就是这样的：`range(1, n+1)`。然而，在某些情况下，我们可能需要对基本值应用一些转换函数：

*N =* { *f(i):* 1 *≤ i < n* +1}如果我们必须执行某种转换，它在 Python 中可能看起来像这样：

```py
            N = (f(i) for i in range(1,n+1)) 

    ```

1.  整合归约函数。在这种情况下，我们正在计算一个大的乘积，使用乘法。我们可以使用 ![如何做...](img/Image00008.jpg)   *x*  表示这一点。对于这个例子，我们只对产品中计算的值施加了一个简单的边界:![如何做...](img/Image00009.jpg)

以下是 Python 中的实现：

```py
            def prod(int_iter): 
                p = 1 
                for x in int_iter: 
                    p *= x 
                return p 

    ```

我们可以将这个重新陈述为这样的解决方案。这使用了更高级的函数：

```py
    def fact(n): 
        return prod(range(1, n+1)) 

```

这很好地起作用。我们已经优化了将`prod()`和`fact()`函数合并为一个函数的第一个解决方案。事实证明，进行这种优化实际上并没有减少操作的时间。

这里是使用`timeit`模块运行的比较：

| **简单** | **4.7766** |
| 优化 | 4.6901 |

这是一个 2%的性能改进。并不是一个显著的改变。

请注意，Python 3 的`range`对象是惰性的——它不创建一个大的`list`对象，它会在`prod()`函数请求时返回值。这与 Python 2 不同，Python 2 中的`range()`函数急切地创建一个包含所有值的大的`list`对象，而`xrange()`函数是惰性的。

## 它是如何工作的...

尾递归定义很方便，因为它既简短又容易记忆。数学家喜欢这个，因为它可以帮助澄清函数的含义。

许多静态的编译语言都以类似于我们展示的技术进行了优化。这种优化有两个部分：

+   使用相对简单的代数规则重新排列语句，使递归子句实际上是最后一个。`if`子句可以重新组织成不同的物理顺序，以便`return fact(n-1) * n`是最后一个。这种重新排列对于这样组织的代码是必要的：

```py
            def ugly_fact(n): 
                if n > 0: 
                    return fact(n-1) * n 
                elif n == 0: 
                    return 1 
                else: 
                    raise Exception("Logic Error") 

    ```

+   将一个特殊指令注入到虚拟机的字节码中 - 或者实际的机器码中 - 重新评估函数，而不创建新的堆栈帧。Python 没有这个特性。实际上，这个特殊指令将递归转换成一种`while`语句：

```py
            p = n 
            while n != 1: 
                n = n-1 
                p *= n 

    ```

这种纯机械的转换会导致相当丑陋的代码。在 Python 中，它也可能非常慢。在其他语言中，特殊的字节码指令的存在将导致代码运行速度快。

我们不喜欢做这种机械优化。首先，它会导致丑陋的代码。更重要的是 - 在 Python 中 - 它往往会创建比上面开发的替代方案更慢的代码。

## 还有更多...

斐波那契问题涉及两个递归。如果我们将其简单地写成递归，可能会像这样：

```py
    def fibo(n): 
        if n <= 1: 
            return 1 
        else: 
            return fibo(n-1)+fibo(n-2) 

```

将一个简单的机械转换成尾递归是困难的。像这样具有多个递归的问题需要更加仔细的设计。

我们有两种方法来减少这个计算复杂度：

+   使用记忆化

+   重新阐述问题

**记忆化**技术在 Python 中很容易应用。我们可以使用`functools.lru_cache()`作为装饰器。这个函数将缓存先前计算过的值。这意味着我们只计算一次值；每一次，`lru_cache`都会返回先前计算过的值。

它看起来像这样：

```py
    from functools import lru_cache 

    @lru_cache(128) 
    def fibo(n): 
        if n <= 1: 
            return 1 
        else: 
            return fibo(n-1)+fibo(n-2) 

```

添加一个装饰器是优化更复杂的多路递归的简单方法。

重新阐述问题意味着从新的角度来看待它。在这种情况下，我们可以考虑计算所有斐波那契数，直到*F[n]*。我们只想要这个序列中的最后一个值。我们计算所有的中间值，因为这样做更有效。这是一个执行此操作的生成器函数：

```py
    def fibo_iter(): 
        a = 1 
        b = 1 
        yield a 
        while True: 
            yield b 
            a, b = b, a+b 

```

这个函数是斐波那契数的无限迭代。它使用 Python 的`yield`，以便以懒惰的方式发出值。当客户函数使用这个迭代器时，每个数字被消耗时，序列中的下一个数字被计算。

这是一个函数，它消耗值，并对否则无限的迭代器施加一个上限：

```py
    def fibo(n): 
        """ 
        >>> fibo(7) 
        21 
        """ 
        for i, f_i in enumerate(fibo_iter()): 
            if i == n: break 
        return f_i 

```

这个函数从`fibo_iter()`迭代器中消耗每个值。当达到所需的数字时，`break`语句结束`for`语句。

当我们回顾第二章中的*设计一个正确终止的 while 语句*配方时，我们注意到一个带有`break`的`while`语句可能有多个终止的原因。在这个例子中，结束`for`语句只有一种方法。

我们可以始终断言在循环结束时`i == n`。这简化了函数的设计。

## 另请参阅

+   请参阅第二章中的*设计一个正确终止的 while 语句*配方，*语句和语法*

# 使用脚本库开关编写可重用脚本

通常会创建一些小脚本，我们希望将它们组合成一个更大的脚本。我们不想复制和粘贴代码。我们希望将工作代码留在一个文件中，并在多个地方使用它。通常，我们希望从多个文件中组合元素，以创建更复杂的脚本。

我们遇到的问题是，当我们导入一个脚本时，它实际上开始运行。这通常不是我们导入一个脚本以便重用它时的预期行为。

我们如何导入文件中的函数（或类），而不让脚本开始执行某些操作？

## 准备好

假设我们有一个方便的 haversine 距离函数的实现，名为`haversine()`，并且它在一个名为`ch03_r08.py`的文件中。

最初，文件可能是这样的：

```py
    import csv 
    import pathlib 
    from math import radians, sin, cos, sqrt, asin 
    from functools import partial 

    MI= 3959 
    NM= 3440 
    KM= 6373 

    def haversine( lat_1: float, lon_1: float, 
        lat_2: float, lon_2: float, *, R: float ) -> float: 
        ... and more ... 

    nm_haversine = partial(haversine, R=NM) 

    source_path = pathlib.Path("waypoints.csv") 
    with source_path.open() as source_file: 
        reader= csv.DictReader(source_file) 
        start = next(reader) 
        for point in reader: 
            d = nm_haversine( 
                float(start['lat']), float(start['lon']), 
                float(point['lat']), float(point['lon']) 
                ) 
            print(start, point, d) 
            start= point 

```

我们省略了`haversine()`函数的主体，只显示了`...和更多...`，因为它在*基于部分函数选择参数顺序*的配方中有所展示。我们专注于函数在 Python 脚本中的上下文，该脚本还打开一个名为`wapypoints.csv`的文件，并对该文件进行一些处理。

我们如何导入这个模块，而不让它打印出`waypoints.csv`文件中航点之间的距离？

## 如何做...

Python 脚本可以很容易编写。事实上，创建一个可工作的脚本通常太简单了。以下是我们如何将一个简单的脚本转换为可重用的库：

1.  识别脚本的工作语句：我们将区分*定义*和*动作*。例如`import`，`def`和`class`等语句显然是定义性的——它们支持工作但并不执行工作。几乎所有其他语句都是执行动作的。

在我们的例子中，有四个赋值语句更多地是定义而不是动作。区别完全是出于意图。所有语句，根据定义，都会执行一个动作。不过，这些动作更像是`def`语句的动作，而不像脚本后面的`with`语句的动作。

以下是通常的定义性语句：

```py
            MI= 3959 
            NM= 3440 
            KM= 6373 

            def haversine( lat_1: float, lon_1: float, 
                lat_2: float, lon_2: float, *, R: float ) -> float: 
                ... and more ... 

            nm_haversine = partial(haversine, R=NM) 

    ```

其余的语句明显是朝着产生打印结果的动作。

1.  将动作封装成一个函数：

```py
            def analyze(): 
                source_path = pathlib.Path("waypoints.csv") 
                with source_path.open() as source_file: 
                    reader= csv.DictReader(source_file) 
                    start = next(reader) 
                    for point in reader: 
                        d = nm_haversine( 
                            float(start['lat']), float(start['lon']), 
                            float(point['lat']), float(point['lon']) 
                            ) 
                        print(start, point, d) 
                        start= point 

    ```

1.  在可能的情况下，提取文字并将其转换为参数。这通常是将文字移到具有默认值的参数中。

从这里开始：

```py
            def analyze(): 
                source_path = pathlib.Path("waypoints.csv") 

    ```

到这里：

```py
            def analyze(source_name="waypoints.csv"): 
                source_path = pathlib.Path(source_name) 

    ```

这使得脚本可重用，因为路径现在是一个参数而不是一个假设。

1.  将以下内容作为脚本文件中唯一的高级动作语句包括：

```py
            if __name__ == "__main__": 
                analyze() 

    ```

我们已经将脚本的动作封装为一个函数。顶层动作脚本现在被包裹在一个`if`语句中，以便在导入时不被执行。

## 它是如何工作的...

Python 的最重要规则是，导入模块实质上与运行模块作为脚本是一样的。文件中的语句按顺序从上到下执行。

当我们导入一个文件时，通常我们对执行`def`和`class`语句感兴趣。我们可能对一些赋值语句感兴趣。

当 Python 运行一个脚本时，它设置了一些内置的特殊变量。其中之一是`__name__`。这个变量有两个不同的值，取决于文件被执行的上下文：

+   从命令行执行的顶层脚本：在这种情况下，内置特殊名称`__name__`的值设置为`__main__`。

+   由于导入语句而执行的文件：在这种情况下，`__name__`的值是正在创建的模块的名称。

`__main__`的标准名称一开始可能有点奇怪。为什么不在所有情况下使用文件名？这个特殊名称是被分配的，因为 Python 脚本可以从多个来源之一读取。它可以是一个文件。Python 也可以从`stdin`管道中读取，或者可以在 Python 命令行中使用`-c`选项提供。

然而，当一个文件被导入时，`__name__`的值被设置为模块的名称。它不会是`__main__`。在我们的例子中，`import`处理期间`__name__`的值将是`ch03_r08`。

## 还有更多...

现在我们可以围绕一个可重用的库构建有用的工作。我们可能会创建几个看起来像这样的文件：

文件`trip_1.py`：

```py
    from ch03_r08 import analyze 
    analyze('trip_1.csv') 

```

或者甚至更复杂一些：

文件`all_trips.py`：

```py
    from ch03_r08 import analyze 
    for trip in 'trip_1.csv', 'trip_2.csv': 
        analyze(trip) 

```

目标是将实际解决方案分解为两个特性集合：

+   类和函数的定义

+   一个非常小的面向行动的脚本，使用定义来进行有用的工作

为了达到这个目标，我们经常会从一个混合了两组特性的脚本开始。这种脚本可以被视为一个**尖峰解决方案**。我们的尖峰解决方案应该在我们确信它有效之后逐渐演变成一个更精细的解决方案。

尖峰或者悬崖钉是一种可移动的登山装备，它并不能让我们在路线上爬得更高，但它能让我们安全地攀登。

## 另请参阅

+   在第六章中，*类和对象的基础*，我们将看一下类定义。这是另一种广泛使用的定义性语句。
