# 开发令人惊叹的图表

本章将涵盖以下示例：

+   绘制简单的销售图表

+   绘制堆叠条形图

+   绘制饼图

+   显示多条线。

+   绘制散点图

+   可视化地图

+   添加图例和注释

+   组合图表

+   保存图表

# 介绍

图表和图像是呈现复杂数据的绝妙方式，易于理解。在本章中，我们将利用强大的`matplotlib`库来学习如何创建各种图表。`matplotlib`是一个旨在以多种方式显示数据的库，它可以创建绝对令人惊叹的图表，有助于以最佳方式传输和显示信息。

我们将涵盖的图表将从简单的条形图到线图或饼图，并结合多个图表在同一图表中，注释它们，甚至绘制地理地图。

# 绘制简单的销售图表

在这个示例中，我们将看到如何通过绘制与不同时期销售成比例的条形来绘制销售图表。

# 准备工作

我们可以使用以下命令在我们的虚拟环境中安装`matplotlib`：

```py
$ echo "matplotlib==2.2.2" >> requirements.txt
$ pip install -r requirements.txt
```

在某些操作系统中，这可能需要我们安装额外的软件包；例如，在Ubuntu中可能需要我们运行`apt-get install python3-tk`。查看`matplolib`文档以获取详细信息。

如果您使用的是macOS，可能会出现这样的错误—`RuntimeError: Python is not installed as a framework`。请参阅`matplolib`文档以了解如何解决：[https://matplotlib.org/faq/osx_framework.html](https://matplotlib.org/faq/osx_framework.html)。

# 如何做...

1.  导入`matplotlib`：

```py
>>> import matplotlib.pyplot as plt
```

1.  准备要在图表上显示的数据：

```py
>>> DATA = (
...    ('Q1 2017', 100),
...    ('Q2 2017', 150),
...    ('Q3 2017', 125),
...    ('Q4 2017', 175),
... )
```

1.  将数据拆分为图表可用的格式。这是一个准备步骤：

```py
>>> POS = list(range(len(DATA)))
>>> VALUES = [value for label, value in DATA]
>>> LABELS = [label for label, value in DATA]
```

1.  创建一个带有数据的条形图：

```py
>>> plt.bar(POS, VALUES)
>>> plt.xticks(POS, LABELS)
>>> plt.ylabel('Sales')
```

1.  显示图表：

```py
>>> plt.show()
```

1.  结果将在新窗口中显示如下：

![](assets/bd9e3f18-0570-48cd-8817-7a9d8dca9732.png)

# 它是如何工作的...

导入模块后，数据将以方便的方式呈现在第2步的*如何做*部分中，这很可能类似于数据最初的存储方式。

由于`matplotlib`的工作方式，它需要*X*组件以及*Y*组件。在这种情况下，我们的*X*组件只是一系列整数，与数据点一样多。我们在`POS`中创建了这个。在`VALUES`中，我们将销售的数值存储为一个序列，在`LABELS`中存储了每个数据点的相关标签。所有这些准备工作都在第3步完成。

第4步创建了条形图，使用了*X*（`POS`）和*Y*（`VALUES`）的序列。这些定义了我们的条形。为了指定它所指的时期，我们使用`.xticks`在*x*轴上为每个值放置标签。为了澄清含义，我们使用`.ylabel`添加标签。

要显示结果图表，第5步调用`.show`，它会打开一个新窗口显示结果。

调用`.show`会阻止程序的执行。当窗口关闭时，程序将恢复。

# 还有更多...

您可能希望更改值的呈现格式。在我们的示例中，也许数字代表数百万美元。为此，您可以向*y*轴添加格式化程序，以便在那里表示的值将应用于它们：

```py
>>> from matplotlib.ticker import FuncFormatter

>>> def value_format(value, position):
...    return '$ {}M'.format(int(value))

>>> axes = plt.gca()
>>> axes.yaxis.set_major_formatter(FuncFormatter(value_format))
```

`value_format`是一个根据数据的值和位置返回值的函数。在这里，它将返回值100作为`$ 100 M`。

值将以浮点数形式检索，需要将它们转换为整数进行显示。

要应用格式化程序，我们需要使用`.gca`（获取当前轴）检索`axis`对象。然后，`.yaxis`获取格式化程序。

条的颜色也可以使用`color`参数确定。颜色可以以多种格式指定，如[https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html)中所述，但我最喜欢的是遵循XKCD颜色调查，使用`xkcd:`前缀（冒号后没有空格）：

```py
>>> plt.bar(POS, VALUES, color='xkcd:moss green')
```

完整的调查可以在这里找到：[https://xkcd.com/color/rgb/](https://xkcd.com/color/rgb/)。

大多数常见的颜色，如蓝色或红色，也可以用于快速测试。但它们往往有点亮，不能用于漂亮的报告。

将颜色与格式化轴结合起来，得到以下结果：

![](assets/2487d3fa-1b81-4594-97ba-1b5dfafbd44e.png)

条形图不一定需要以时间顺序显示信息。正如我们所见，`matplotlib`要求我们指定每个条的*X*参数。这是一个生成各种图表的强大工具。

例如，可以安排条形以显示直方图，比如显示特定身高的人。条形将从较低的高度开始增加到平均大小，然后再降低。不要局限于电子表格图表！

完整的`matplotlib`文档可以在这里找到：[https://matplotlib.org/](https://matplotlib.org/)。

# 另请参阅

+   *绘制堆叠条形图*的方法

+   *添加图例和注释*的方法

+   *组合图表*的方法

# 绘制堆叠条形图

一种强大的显示不同类别的方法是将它们呈现为堆叠条形图，因此每个类别和总数都会显示出来。我们将在这个方法中看到如何做到这一点。

# 准备就绪

我们需要在虚拟环境中安装`matplotlib`：

```py
$ echo "matplotlib==2.2.2" >> requirements.txt
$ pip install -r requirements.txt
```

如果您使用的是macOS，可能会出现这样的错误：`RuntimeError: Python is not installed as a framework`。请参阅`matplolib`文档以了解如何解决：[https://matplotlib.org/faq/osx_framework.html](https://matplotlib.org/faq/osx_framework.html)。

# 如何做...

1.  导入`matplotlib`：

```py
>>> import matplotlib.pyplot as plt
```

1.  准备数据。这代表了两种产品的销售，一个是已建立的，另一个是新产品：

```py
>>> DATA = (
...     ('Q1 2017', 100, 0),
...     ('Q2 2017', 105, 15),
...     ('Q3 2017', 125, 40),
...     ('Q4 2017', 115, 80),
... )
```

1.  处理数据以准备期望的格式：

```py
>>> POS = list(range(len(DATA)))
>>> VALUESA = [valueA for label, valueA, valueB in DATA]
>>> VALUESB = [valueB for label, valueA, valueB in DATA]
>>> LABELS = [label for label, value1, value2 in DATA]
```

1.  创建条形图。需要两个图：

```py
>>> plt.bar(POS, VALUESB)
>>> plt.bar(POS, VALUESA, bottom=VALUESB)
>>> plt.ylabel('Sales')
>>> plt.xticks(POS, LABELS)
```

1.  显示图表：

```py
>>> plt.show()
```

1.  结果将显示在一个新窗口中，如下所示：

![](assets/6d97abab-342a-4cee-b175-3786b4ae9dd5.png)

# 它是如何工作的...

导入模块后，在第2步以一种方便的方式呈现数据，这可能与数据最初存储的方式类似。

在第3步中，数据准备为三个序列，`VALUESA`，`VALUEB`和`LABELS`。添加了一个`POS`序列以正确定位条形。

第4步创建了条形图，使用了序列*X*（`POS`）和*Y*（`VALUESB`）。第二个条形序列`VALUESA`添加到前一个上面，使用`bottom`参数。这样就堆叠了条形。

请注意，我们首先堆叠第二个值`VALUESB`。第二个值代表市场上推出的新产品，而`VALUESA`更加稳定。这更好地显示了新产品的增长。

每个期间都在*X*轴上用`.xticks`标记。为了澄清含义，我们使用`.ylabel`添加标签。

要显示生成的图表，第5步调用`.show`，这将打开一个新窗口显示结果。

调用`.show`会阻止程序的执行。当窗口关闭时，程序将恢复。

# 还有更多...

呈现堆叠条形的另一种方法是将它们添加为百分比，这样总数不会改变，只是相对大小相互比较。

为了做到这一点，需要根据百分比计算`VALUESA`和`VALUEB`：

```py
>>> VALUESA = [100 * valueA / (valueA + valueB) for label, valueA, valueB in DATA]
>>> VALUESB = [100 * valueB / (valueA + valueB) for label, valueA, valueB in DATA]
```

这使得每个值都等于总数的百分比，总数始终加起来为`100`。这产生了以下图形：

![](assets/f1270a74-2f03-49e2-b731-c2a539e6a774.png)

条形不一定需要堆叠。有时，将条形相互对比呈现可能会更有趣。

为了做到这一点，我们需要移动第二个条形序列的位置。我们还需要设置更细的条形以留出空间：

```py
>>> WIDTH = 0.3
>>> plt.bar([p - WIDTH / 2 for p in POS], VALUESA, width=WIDTH)
>>> plt.bar([p + WIDTH / 2 for p in POS], VALUESB, width=WIDTH)
```

注意条的宽度设置为空间的三分之一，因为我们的参考空间在条之间是`1`。第一根条移到左边，第二根移到右边以使它们居中。已删除`bottom`参数，以不堆叠条形：

![](assets/7cd52074-9e8e-4407-ae9f-e751c8e64e3f.png)

完整的`matplotlib`文档可以在这里找到：[https://matplotlib.org/](https://matplotlib.org/)。

# 另请参阅

+   *绘制简单销售图表*食谱

+   *添加图例和注释*食谱

+   *组合图表*食谱

# 绘制饼图

饼图！商业101最喜欢的图表，也是呈现百分比的常见方式。在这个食谱中，我们将看到如何绘制一个饼图，不同的切片代表不同的比例。

# 准备工作

我们需要使用以下命令在虚拟环境中安装`matplotlib`：

```py
$ echo "matplotlib==2.2.2" >> requirements.txt
$ pip install -r requirements.txt
```

如果您使用的是macOS，可能会出现这样的错误——`RuntimeError: Python is not installed as a framework`。请参阅`matplotlib`文档以了解如何解决此问题：[https://matplotlib.org/faq/osx_framework.html](https://matplotlib.org/faq/osx_framework.html)。

# 如何做...

1.  导入`matplotlib`：

```py
>>> import matplotlib.pyplot as plt
```

1.  准备数据。这代表了几条产品线：

```py
>>> DATA = (
...     ('Common', 100),
...     ('Premium', 75),
...     ('Luxurious', 50),
...     ('Extravagant', 20),
... )
```

1.  处理数据以准备预期格式：

```py
>>> VALUES = [value for label, value in DATA]
>>> LABELS = [label for label, value in DATA]
```

1.  创建饼图：

```py
>>> plt.pie(VALUES, labels=LABELS, autopct='%1.1f%%')
>>> plt.gca().axis('equal')
```

1.  显示图表：

```py
>>> plt.show()
```

1.  结果将显示在新窗口中，如下所示：

![](assets/d11a4172-49d1-4df0-a0a9-df33e6bb7c10.png)

# 工作原理...

在*如何做...*部分的第1步中导入了该模块，并在第2步中导入了要呈现的数据。在第3步中，数据被分成两个部分，一个是`VALUES`的列表，另一个是`LABELS`的列表。

图表的创建发生在第4步。饼图是通过添加`VALUES`和`LABELS`来创建的。`autopct`参数格式化值，以便将其显示为百分比到小数点后一位。

对`axis`的调用确保饼图看起来是圆形的，而不是有一点透视并呈现为椭圆。

要显示生成的图表，第5步调用`.show`，它会打开一个新窗口显示结果。

调用`.show`会阻塞程序的执行。当窗口关闭时，程序将恢复。

# 还有更多...

饼图在商业图表中有点过度使用。大多数情况下，使用带百分比或值的条形图会更好地可视化数据，特别是当显示两个或三个以上的选项时。尽量限制在报告和数据演示中使用饼图。

通过`startangle`参数可以旋转楔形的起始位置，使用`counterclock`来设置楔形的方向（默认为`True`）：

```py
>>> plt.pie(VALUES, labels=LABELS, startangle=90, counterclock=False)
```

标签内的格式可以通过函数设置。由于饼图内的值被定义为百分比，找到原始值可能有点棘手。以下代码片段创建了一个按整数百分比索引的字典，因此我们可以检索引用的值。请注意，这假设没有重复的百分比。如果有这种情况，标签可能会略有不正确。在这种情况下，我们可能需要使用更好的精度，最多使用小数点后一位：

```py
>>> from matplotlib.ticker import FuncFormatter

>>> total = sum(value for label, value in DATA)
>>> BY_VALUE = {int(100 * value / total): value for label, value in DATA}

>>> def value_format(percent, **kwargs):
...     value = BY_VALUE[int(percent)]
...     return '{}'.format(value)
```

一个或多个楔形也可以通过使用explode参数分开。这指定了楔形与中心的分离程度：

```py
>>> explode = (0, 0, 0.1, 0)
>>> plt.pie(VALUES, labels=LABELS, explode=explode, autopct=value_format,
            startangle=90, counterclock=False)
```

结合所有这些选项，我们得到以下结果：

![](assets/907bd5d6-c5cb-49cb-8f51-29aebf936ed1.png)

完整的`matplotlib`文档可以在这里找到：[https://matplotlib.org/](https://matplotlib.org/)。

# 另请参阅

+   *绘制简单销售图表*食谱

+   *绘制堆叠条形图*食谱

# 显示多条线

这个食谱将展示如何在图表中显示多条线。

# 准备工作

我们需要在虚拟环境中安装`matplotlib`：

```py
$ echo "matplotlib==2.2.2" >> requirements.txt
$ pip install -r requirements.txt
```

如果您使用的是macOS，可能会出现这样的错误——`RuntimeError: Python is not installed as a framework`。请参阅`matplolib`文档以了解如何解决此问题：[https://matplotlib.org/faq/osx_framework.html](https://matplotlib.org/faq/osx_framework.html)。

# 如何做...

1.  导入`matplotlib`：

```py
>>> import matplotlib.pyplot as plt
```

1.  准备数据。这代表了两种产品的销售：

```py
>>> DATA = (
...     ('Q1 2017', 100, 5),
...     ('Q2 2017', 105, 15),
...     ('Q3 2017', 125, 40),
...     ('Q4 2017', 115, 80),
... )
```

1.  处理数据以准备预期格式：

```py
>>> POS = list(range(len(DATA)))
>>> VALUESA = [valueA for label, valueA, valueB in DATA]
>>> VALUESB = [valueB for label, valueA, valueB in DATA]
>>> LABELS = [label for label, value1, value2 in DATA]
```

1.  创建线图。需要两条线：

```py
>>> plt.plot(POS, VALUESA, 'o-')
>>> plt.plot(POS, VALUESB, 'o-')
>>> plt.ylabel('Sales')
>>> plt.xticks(POS, LABELS)
```

1.  显示图表：

```py
>>> plt.show()
```

1.  结果将显示在一个新窗口中：

![](assets/defcbfbe-a08d-4ed8-b19d-b03b82ef8552.png)

# 工作原理…

在*如何做…*部分，第1步导入模块，第2步以格式化的方式显示要绘制的数据。

在第3步中，数据准备好了三个序列`VALUESA`，`VALUEB`和`LABELS`。添加了一个`POS`序列来正确定位每个点。

第4步创建了图表，使用了序列*X*（`POS`）和*Y*（`VALUESA`），然后是`POS`和`VALUESB`。添加了值为`'o-'`，以在每个数据点上绘制一个圆圈，并在它们之间绘制一条实线。

默认情况下，图表将显示一条实线，每个点上没有标记。如果只使用标记（即`'o'`），就不会有线。

*X*轴上的每个周期都带有`.xticks`标签。为了澄清含义，我们使用`.ylabel`添加了一个标签。

要显示结果图表，第5步调用`.show`，它会打开一个新窗口显示结果。

调用`.show`会阻塞程序的执行。当窗口关闭时，程序将恢复。

# 还有更多…

带有线条的图表看起来简单，能够创建许多有趣的表示。在显示数学图表时，这可能是最方便的。例如，我们可以用几行代码显示Moore定律的图表。

摩尔定律是戈登·摩尔观察到的一个现象，即集成电路中的元件数量每两年翻一番。它首次在1965年被描述，然后在1975年得到修正。它似乎与过去40年的技术进步历史速度非常接近。

我们首先创建了一条描述理论线的线，数据点从1970年到2013年。从1000个晶体管开始，每两年翻一番，直到2013年：

```py
>>> POS = [year for year in range(1970, 2013)]
>>> MOORES = [1000 * (2 ** (i * 0.5)) for i in range(len(POS))]
>>> plt.plot(POS, MOORES)
```

根据一些文档，我们从这里提取了一些商用CPU的例子，它们的发布年份以及集成元件的数量：[http://www.wagnercg.com/Portals/0/FunStuff/AHistoryofMicroprocessorTransistorCount.pdf](http://www.wagnercg.com/Portals/0/FunStuff/AHistoryofMicroprocessorTransistorCount.pdf)。由于数字很大，我们将使用Python 3中的`1_000_000`表示一百万：

```py
>>> DATA = (
...    ('Intel 4004', 2_300, 1971),
...    ('Motorola 68000', 68_000, 1979),
...    ('Pentium', 3_100_000, 1993),
...    ('Core i7', 731_000_000, 2008),
... )
```

绘制一条带有标记的线，以在正确的位置显示这些点。`'v'`标记将显示一个三角形：

```py
>>> data_x = [x for label, y, x in DATA]
>>> data_y = [y for label, y, x in DATA]
>>> plt.plot(data_x, data_y, 'v')
```

对于每个数据点，将一个标签附加在正确的位置，标有CPU的名称：

```py
>>> for label, y, x in DATA:
>>>    plt.text(x, y, label)
```

最后，成长在线性图表中没有意义，因此我们将比例改为对数，这样指数增长看起来像一条直线。但为了保持尺度的意义，添加一个网格。调用`.show`显示图表：

```py
>>> plt.gca().grid()
>>> plt.yscale('log')
```

结果图将显示：

![](assets/bc851b15-ef70-481c-b08e-51b3879cfb60.png)

完整的`matplotlib`文档可以在这里找到：[https://matplotlib.org/](https://matplotlib.org/)。特别是，可以在这里检查线条（实线、虚线、点线等）和标记（点、圆圈、三角形、星形等）的可用格式：[https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)。

# 另请参阅

+   *添加图例和注释*配方

+   *组合图表*配方

# 绘制散点图

散点图是一种只显示为点的信息，具有*X*和*Y*值。当呈现样本并查看两个变量之间是否存在关系时，它们非常有用。在这个配方中，我们将显示一个图表，绘制在网站上花费的时间与花费的金钱，以查看是否可以看到一个模式。

# 准备就绪

我们需要在虚拟环境中安装`matplotlib`：

```py
$ echo "matplotlib==2.2.2" >> requirements.txt
$ pip install -r requirements.txt
```

如果您使用的是macOS，可能会出现这样的错误——`RuntimeError: Python is not installed as a framework`。请参阅`matplolib`文档，了解如何解决此问题：[https://matplotlib.org/faq/osx_framework.html](https://matplotlib.org/faq/osx_framework.html)。

作为数据点，我们将使用`scatter.csv`文件来读取数据。此文件可在GitHub上找到：[https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter07/scatter.csv](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter07/scatter.csv)。

# 如何做...

1.  导入`matplotlib`和`csv`。还导入`FuncFormatter`以稍后格式化轴：

```py
>>> import csv
>>> import matplotlib.pyplot as plt
>>> from matplotlib.ticker import FuncFormatter
```

1.  准备数据，使用`csv`模块从文件中读取：

```py
>>> with open('scatter.csv') as fp:
...    reader = csv.reader(fp)
...    data = list(reader)
```

1.  准备绘图数据，然后绘制：

```py
>>> data_x = [float(x) for x, y in data]
>>> data_y = [float(y) for x, y in data]
>>> plt.scatter(data_x, data_y)
```

1.  通过格式化轴来改善上下文：

```py
>>> def format_minutes(value, pos):
...     return '{}m'.format(int(value))
>>> def format_dollars(value, pos):
...     return '${}'.format(value)
>>> plt.gca().xaxis.set_major_formatter(FuncFormatter(format_minutes))
>>> plt.xlabel('Time in website')
>>> plt.gca().yaxis.set_major_formatter(FuncFormatter(format_dollars))
>>> plt.ylabel('Spending')
```

1.  显示图表：

```py
>>> plt.show()
```

1.  结果将显示在新窗口中：

![](assets/ced40fbf-6f2b-4eb1-b8b9-f5483b9f4989.png)

# 工作原理...

*如何做…*部分的步骤1和2导入了我们稍后将使用的模块并从CSV文件中读取数据。数据被转换为列表，以允许我们多次迭代，这在第3步中是必要的。

第3步将数据准备为两个数组，然后使用`.scatter`来绘制它们。与`matplotlib`的其他方法一样，`.scatter`的参数需要*X*和*Y*值的数组。它们都需要具有相同的大小。数据从文件格式转换为`float`，以确保数字格式。

第4步改进了数据在每个轴上的呈现方式。相同的操作被呈现两次——创建一个函数来定义该轴上的值应该如何显示（以美元或分钟）。该函数接受要显示的值和位置作为输入。通常，位置将被忽略。轴格式化程序将被覆盖为`.set_major_formatter`。请注意，两个轴都将使用`.gca`（获取当前轴）返回。

使用`.xlabel`和`.ylabel`向轴添加标签。

最后，第5步在新窗口中显示图表。分析结果，我们可以说似乎有两种用户，一些用户花费不到10分钟，从不花费超过10美元，还有一些用户花费更多时间，也更有可能花费高达100美元。

请注意，所呈现的数据是合成的，并且已经根据结果生成。现实生活中的数据可能看起来更分散。

# 还有更多...

散点图不仅可以显示二维空间中的点，还可以添加第三个（面积）甚至第四个维度（颜色）。

要添加这些元素，使用参数`s`表示*大小*，`c`表示*颜色*。

大小被定义为点的直径的平方。因此，对于直径为10的球，将使用100。颜色可以使用`matplotlib`中颜色的任何常规定义，例如十六进制颜色、RGB等。有关更多详细信息，请参阅文档：[https://matplotlib.org/users/colors.html](https://matplotlib.org/users/colors.html)。例如，我们可以使用以下方式生成一个随机图表的四个维度：

```py
>>> import matplotlib.pyplot as plt
>>> import random
>>> NUM_POINTS = 100
>>> COLOR_SCALE = ['#FF0000', '#FFFF00', '#FFFF00', '#7FFF00', '#00FF00']
>>> data_x = [random.random() for _ in range(NUM_POINTS)]
>>> data_y = [random.random() for _ in range(NUM_POINTS)]
>>> size = [(50 * random.random()) ** 2 for _ in range(NUM_POINTS)]
>>> color = [random.choice(COLOR_SCALE) for _ in range(NUM_POINTS)]
>>> plt.scatter(data_x, data_y, s=size, c=color, alpha=0.5)
>>> plt.show()
```

`COLOR_SCALE`从绿色到红色，每个点的大小将在`0`到`50`之间。结果应该是这样的：

![](assets/ab8e4b1e-6dfa-4367-8e57-47e7557c5f45.png)

请注意，这是随机的，因此每次都会生成不同的图表。

`alpha`值使每个点半透明，使我们能够看到它们重叠的位置。该值越高，点的透明度越低。此参数将影响显示的颜色，因为它将点与背景混合。

尽管可以在大小和颜色中显示两个独立的值，但它们也可以与任何其他值相关联。例如，使颜色依赖于大小将使所有相同大小的点具有相同的颜色，这可能有助于区分数据。请记住，图表的最终目标是使数据易于理解。尝试不同的方法来改进这一点。

完整的`matplotlib`文档可以在这里找到：[https://matplotlib.org/](https://matplotlib.org/)。

# 另请参阅

+   *显示多行*的方法

+   *添加图例和注释*的方法

# 可视化地图

要显示从区域到区域变化的信息，最好的方法是显示一张呈现信息的地图，同时为数据提供区域位置和位置的感觉。

在此示例中，我们将利用`fiona`模块导入GIS信息，以及`matplotlib`来显示信息。 我们将显示西欧的地图，并显示每个国家的人口与颜色等级。 颜色越深，人口越多。

# 准备工作

我们需要在虚拟环境中安装`matplotlib`和`fiona`：

```py
$ echo "matplotlib==2.2.2" >> requirements.txt
$ echo "Fiona==1.7.13" >> requirements.txt
$ pip install -r requirements.txt
```

如果您使用的是macOS，可能会出现这样的错误-`RuntimeError: Python is not installed as a framework`。 请参阅`matplolib`文档以了解如何解决此问题：[https://matplotlib.org/faq/osx_framework.html](https://matplotlib.org/faq/osx_framework.html)。

需要下载地图数据。 幸运的是，有很多免费提供的地理信息数据。 在Google上搜索应该很快返回几乎您需要的所有内容，包括有关地区，县，河流或任何其他类型数据的详细信息。

来自许多公共组织的GIS信息以不同格式可用。 `fiona`能够理解大多数常见格式并以等效方式处理它们，但存在细微差异。 请阅读`fiona`文档以获取更多详细信息。

我们将在此示例中使用的数据，涵盖所有欧洲国家，可在GitHub的以下网址找到：[https://github.com/leakyMirror/map-of-europe/blob/master/GeoJSON/europe.geojson](https://github.com/leakyMirror/map-of-europe/blob/master/GeoJSON/europe.geojson)。 请注意，它是GeoJSON格式，这是一种易于使用的标准。

# 如何操作...

1.  导入稍后要使用的模块：

```py
>>> import matplotlib.pyplot as plt
>>> import matplotlib.cm as cm
>>> import fiona
```

1.  加载要显示的国家的人口。 人口已经是：

```py
>>> COUNTRIES_POPULATION = {
...     'Spain': 47.2,
...     'Portugal': 10.6,
...     'United Kingdom': 63.8,
...     'Ireland': 4.7,
...     'France': 64.9,
...     'Italy': 61.1,
...     'Germany': 82.6,
...     'Netherlands': 16.8,
...     'Belgium': 11.1,
...     'Denmark': 5.6,
...     'Slovenia': 2,
...     'Austria': 8.5,
...     'Luxembourg': 0.5,
...     'Andorra': 0.077,
...     'Switzerland': 8.2,
...     'Liechtenstein': 0.038,
... }
>>> MAX_POPULATION = max(COUNTRIES_POPULATION.values())
>>> MIN_POPULATION = min(COUNTRIES_POPULATION.values())
```

1.  准备`colormap`，它将确定每个国家显示在绿色阴影中的颜色。 计算每个国家对应的颜色：

```py
>>> colormap = cm.get_cmap('Greens')
>>> COUNTRY_COLOUR = {
...     country_name: colormap(
...         (population - MIN_POPULATION) / (MAX_POPULATION - MIN_POPULATION)
...     )
...     for country_name, population in COUNTRIES_POPULATION.items()
... }
```

1.  打开文件并读取数据，按照我们在第1步中定义的国家进行过滤：

```py
>>> with fiona.open('europe.geojson') as fd:
>>>     full_data = [data for data in full_data
...                  if data['properties']['NAME'] in COUNTRIES_POPULATION]
```

1.  以正确的颜色绘制每个国家：

```py
>>> for data in full_data:
...     country_name = data['properties']['NAME']
...     color = COUNTRY_COLOUR[country_name]
...     geo_type = data['geometry']['type']
...     if geo_type == 'Polygon':
...         data_x = [x for x, y in data['geometry']['coordinates'][0]]
...         data_y = [y for x, y in data['geometry']['coordinates'][0]]
...         plt.fill(data_x, data_y, c=color)
...     elif geo_type == 'MultiPolygon':
...         for coordinates in data['geometry']['coordinates']:
...             data_x = [x for x, y in coordinates[0]]
...             data_y = [y for x, y in coordinates[0]]
...             plt.fill(data_x, data_y, c=color)
```

1.  显示结果：

```py
>>> plt.show()
```

1.  结果将显示在新窗口中：

![](assets/d4a76e46-e98f-4763-8e14-5658dd95e0e6.png)

# 它是如何工作的...

在*如何操作...*部分的第1步中导入模块后，将在第2步中定义要显示的数据。 请注意，名称需要与GEO文件中的格式相同。 最小和最大人口将被计算以正确平衡范围。

人口已经舍入到一个显著数字，并以百万定义。 仅为此示例的目的定义了一些国家，但在GIS文件中还有更多可用的国家，并且地图可以向东扩展。

在第3步中描述了定义绿色阴影（`Greens`）范围的`colormap`。 这是`matplotlib`中的一个标准`colormap`，但可以使用文档中描述的其他`colormap`（[https://matplotlib.org/examples/color/colormaps_reference.html](https://matplotlib.org/examples/color/colormaps_reference.html)），例如橙色，红色或等离子体，以获得更冷到热的方法。

`COUNTRY_COLOUR`字典存储了由`colormap`为每个国家定义的颜色。 人口减少到从0.0（最少人口）到1.0（最多）的数字，并传递给`colormap`以检索其对应的比例的颜色。

然后在第4步中检索GIS信息。 使用`fiona`读取`europe.geojson`文件，并复制数据，以便在接下来的步骤中使用。 它还会过滤，只处理我们定义了人口的国家，因此不会绘制额外的国家。

步骤5中的循环逐个国家进行，然后我们使用`.fill`来绘制它，它绘制一个多边形。每个不同国家的几何形状都是一个单一的多边形（`Polygon`）或多个多边形（`MultiPolygon`）。在每种情况下，适当的多边形都以相同的颜色绘制。这意味着`MultiPolygon`会被绘制多次。

GIS信息以描述纬度和经度的坐标点的形式存储。区域，如国家，有一系列坐标来描述其中的区域。一些地图更精确，有更多的点来定义区域。可能需要多个多边形来定义一个国家，因为一些部分可能相互分离，岛屿是最明显的情况，但也有飞地。

最后，通过调用`.show`来显示数据。

# 还有更多...

利用GIS文件中包含的信息，我们可以向地图添加额外的信息。`properties`对象包含有关国家名称的信息，还有ISO名称、FID代码和中心位置的`LON`和`LAT`。我们可以使用这些信息来使用`.text`显示国家的名称：

```py
    long, lat = data['properties']['LON'], data['properties']['LAT']
    iso3 = data['properties']['ISO3']
    plt.text(long, lat, iso3, horizontalalignment='center')
```

这段代码将存在于*如何做*部分的步骤6中的循环中。

如果你分析这个文件，你会发现`properties`对象包含有关人口的信息，存储为POP2005，所以你可以直接从地图上绘制人口信息。这留作练习。不同的地图文件将包含不同的信息，所以一定要尝试一下，释放所有可能性。

此外，你可能会注意到在某些情况下地图可能会变形。`matplotlib`会尝试将其呈现为一个正方形的框，如果地图不是大致正方形，这将是明显的。例如，尝试只显示西班牙、葡萄牙、爱尔兰和英国。我们可以强制图表以1点纬度与1点经度相同的空间来呈现，这是一个很好的方法，如果我们不是在靠近极地的地方绘制东西。这是通过在轴上调用`.set_aspect`来实现的。当前轴可以通过`.gca`（**获取当前轴**）获得。

```py
>>> axes = plt.gca()
>>> axes.set_aspect('equal', adjustable='box')
```

此外，为了改善地图的外观，我们可以设置一个背景颜色，以帮助区分背景和前景，并删除轴上的标签，因为打印纬度和经度可能会分散注意力。通过使用`.xticks`和`.yticks`设置空标签来实现在轴上删除标签。背景颜色由轴的前景颜色规定：

```py
>>> plt.xticks([])
>>> plt.yticks([])
>>> axes = plt.gca()
>>> axes.set_facecolor('xkcd:light blue')
```

最后，为了更好地区分不同的区域，可以添加一个包围每个区域的线。这可以通过在`.fill`之后用相同的数据绘制一条细线来实现。请注意，这段代码在步骤2中重复了两次。

```py
 plt.fill(data_x, data_y, c=color)
 plt.plot(data_x, data_y, c='black', linewidth=0.2)
```

将所有这些元素应用到地图上，现在看起来是这样的：

![](assets/a57d8513-6d08-4ba4-a29c-9f03641ec34f.png)

生成的代码可以在GitHub上找到：[https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter07/visualising_maps.py](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter07/visualising_maps.py)。

正如我们所见，地图是以一般的多边形绘制的。不要害怕包括其他几何形状。你可以定义自己的多边形，并用`.fill`或一些额外的标签打印它们。例如，远离的地区可能需要被运输，以避免地图太大。或者，可以使用矩形在地图的部分上打印额外的信息。

完整的`fiona`文档可以在这里找到：[http://toblerity.org/fiona/](http://toblerity.org/fiona/)。完整的`matplotlib`文档可以在这里找到：[https://matplotlib.org/](https://matplotlib.org/)。

# 另请参阅

+   *添加图例和注释*配方

+   *组合图表*配方

# 添加图例和注释

在绘制具有密集信息的图表时，可能需要图例来确定特定颜色或更好地理解所呈现的数据。在`matplotlib`中，图例可以非常丰富，并且有多种呈现方式。注释也是吸引观众注意力的好方法，以便更好地传达信息。

在本示例中，我们将创建一个具有三个不同组件的图表，并显示一个包含信息的图例，以更好地理解它，并在图表上注释最有趣的点。

# 准备工作

我们需要在虚拟环境中安装`matplotlib`：

```py
$ echo "matplotlib==2.2.2" >> requirements.txt
$ pip install -r requirements.txt
```

如果您正在使用macOS，可能会出现这样的错误——`RuntimeError: Python is not installed as a framework`。请参阅`matplolib`文档以了解如何解决：[https://matplotlib.org/faq/osx_framework.html](https://matplotlib.org/faq/osx_framework.html)。

# 操作步骤...

1.  导入`matplotlib`：

```py
>>> import matplotlib.pyplot as plt
```

1.  准备要在图表上显示的数据，以及应该显示的图例。每行由时间标签、`ProductA`的销售额、`ProductB`的销售额和`ProductC`的销售额组成：

```py
>>> LEGEND = ('ProductA', 'ProductB', 'ProductC')
>>> DATA = (
...     ('Q1 2017', 100, 30, 3),
...     ('Q2 2017', 105, 32, 15),
...     ('Q3 2017', 125, 29, 40),
...     ('Q4 2017', 115, 31, 80),
... )
```

1.  将数据拆分为图表可用的格式。这是一个准备步骤：

```py
>>> POS = list(range(len(DATA)))
>>> VALUESA = [valueA for label, valueA, valueB, valueC in DATA]
>>> VALUESB = [valueB for label, valueA, valueB, valueC in DATA]
>>> VALUESC = [valueC for label, valueA, valueB, valueC in DATA]
>>> LABELS = [label for label, valueA, valueB, valueC in DATA]
```

1.  创建带有数据的条形图：

```py
>>> WIDTH = 0.2
>>> plt.bar([p - WIDTH for p in POS], VALUESA, width=WIDTH)
>>> plt.bar([p for p in POS], VALUESB, width=WIDTH)
>>> plt.bar([p + WIDTH for p in POS], VALUESC, width=WIDTH)
>>> plt.ylabel('Sales')
>>> plt.xticks(POS, LABELS)
```

1.  添加一个注释，显示图表中的最大增长：

```py
>>> plt.annotate('400% growth', xy=(1.2, 18), xytext=(1.3, 40),
                 horizontalalignment='center',
                 arrowprops=dict(facecolor='black', shrink=0.05))
```

1.  添加`legend`：

```py
>>> plt.legend(LEGEND)
```

1.  显示图表：

```py
>>> plt.show()
```

1.  结果将显示在新窗口中：

![](assets/7284a61f-70fe-43e5-b151-152dd0086307.png)

# 它是如何工作的...

*操作步骤*的第1步和第2步准备了导入和将在条形图中显示的数据，格式类似于良好结构化的输入数据。在第3步中，数据被拆分成不同的数组，以准备在`matplotlib`中输入。基本上，每个数据序列都存储在不同的数组中。

第4步绘制数据。每个数据序列都会调用`.bar`，指定其位置和值。标签与`.xticks`相同。为了在标签周围分隔每个条形图，第一个条形图向左偏移，第三个向右偏移。

在第二季度的`ProductC`条形图上方添加了一个注释。请注意，注释包括`xy`中的点和`xytext`中的文本位置。

在第6步中，添加了图例。请注意，标签需要按照输入数据的顺序添加。图例会自动放置在不覆盖任何数据的区域。`arroprops`详细说明了指向数据的箭头。

最后，在第7步通过调用`.show`绘制图表。

调用`.show`会阻止程序的执行。当窗口关闭时，程序将恢复执行。

# 还有更多...

图例通常会自动显示，只需调用`.legend`即可。如果需要自定义它们的显示顺序，可以将每个标签指定给特定元素。例如，这种方式（注意它将`ProductA`称为`valueC`系列）

```py
>>> valueA = plt.bar([p - WIDTH for p in POS], VALUESA, width=WIDTH)
>>> valueB = plt.bar([p for p in POS], VALUESB, width=WIDTH)
>>> valueC = plt.bar([p + WIDTH for p in POS], VALUESC, width=WIDTH)
>>> plt.legend((valueC, valueB, valueA), LEGEND)
```

图例的位置也可以通过`loc`参数手动更改。默认情况下，它是`best`，它会在数据最少重叠的区域绘制图例（理想情况下没有）。但是可以使用诸如`right`、`upper left`等值，或者特定的`(X, Y)`元组。

另一种选择是在图表之外绘制图例，使用`bbox_to_anchor`选项。在这种情况下，图例附加到边界框的（*X*，*Y*）位置，其中`0`是图表的左下角，`1`是右上角。这可能导致图例被外部边框剪切，因此您可能需要通过`.subplots_adjust`调整图表的起始和结束位置：

```py
>>> plt.legend(LEGEND, title='Products', bbox_to_anchor=(1, 0.8))
>>> plt.subplots_adjust(right=0.80)
```

调整`bbox_to_anchor`参数和`.subplots_adjust`将需要一些试错，直到产生预期的结果。

`.subplots_adjust`引用了位置，作为将显示的轴的位置。这意味着`right=0.80`将在绘图的右侧留下20%的屏幕空间，而左侧的默认值为0.125，这意味着在绘图的左侧留下12.5%的空间。有关更多详细信息，请参阅文档：[https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html)。

注释可以以不同的样式进行，并可以通过不同的选项进行调整，例如连接方式等。例如，这段代码将创建一个箭头，使用“fancy”样式连接一个曲线。结果显示在这里：

```py
plt.annotate('400% growth', xy=(1.2, 18), xytext=(1.3, 40),
             horizontalalignment='center',
             arrowprops={'facecolor': 'black',
                         'arrowstyle': "fancy",
                         'connectionstyle': "angle3",
                         })
```

在我们的方法中，我们没有精确地注释到条的末端（点（`1.2`，`15`）），而是略高于它，以留出一点空间。

调整注释的确切位置和文本的位置将需要进行一些测试。文本的位置也是通过寻找最佳位置来避免与条形图重叠而定位的。字体大小和颜色可以使用`.legend`和`.annotate`调用中的`fontsize`和`color`参数进行更改。

应用所有这些元素，图表可能看起来类似于这样。可以通过调用GitHub上的`legend_and_annotation.py`脚本来复制此图表：[https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter07/adding_legend_and_annotations.py](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter07/adding_legend_and_annotations.py)：

![](assets/cbe93454-92bf-4133-a125-e94c0269ba18.png)

完整的`matplotlib`文档可以在这里找到：[https://matplotlib.org/](https://matplotlib.org/)。特别是图例的指南在这里：[https://matplotlib.org/users/legend_guide.html#plotting-guide-legend](https://matplotlib.org/users/legend_guide.html#plotting-guide-legend)，注释的指南在这里：[https://matplotlib.org/users/annotations.html](https://matplotlib.org/users/annotations.html)。

# 另请参阅

+   *绘制堆叠条形图*的方法

+   *组合图表*的方法

# 组合图表

可以在同一图表中组合多个图表。在这个方法中，我们将看到如何在同一图表上以两个不同的轴呈现数据，并如何在同一图表上添加更多的图表。

# 准备工作

我们需要在虚拟环境中安装`matplotlib`：

```py
$ echo "matplotlib==2.2.2" >> requirements.txt
$ pip install -r requirements.txt
```

如果您使用的是macOS，可能会出现这样的错误——`RuntimeError: Python is not installed as a framework`。请参阅`matplolib`文档，了解如何解决此问题：[https://matplotlib.org/faq/osx_framework.html](https://matplotlib.org/faq/osx_framework.html)。

# 如何做…

1.  导入`matplotlib`：

```py
>>> import matplotlib.pyplot as plt
```

1.  准备数据以在图表上显示，并显示应该显示的图例。每条线由时间标签、`ProductA`的销售额和`ProductB`的销售额组成。请注意，`ProductB`的值远高于`A`：

```py
>>> DATA = (
...  ('Q1 2017', 100, 3000, 3),
...  ('Q2 2017', 105, 3200, 5),
...  ('Q3 2017', 125, 2900, 7),
...  ('Q4 2017', 115, 3100, 3),
... )
```

1.  准备独立数组中的数据：

```py
>>> POS = list(range(len(DATA)))
>>> VALUESA = [valueA for label, valueA, valueB, valueC in DATA]
>>> VALUESB = [valueB for label, valueA, valueB, valueC in DATA]
>>> VALUESC = [valueC for label, valueA, valueB, valueC in DATA]
>>> LABELS = [label for label, valueA, valueB, valueC in DATA]
```

请注意，这将扩展并为每个值创建一个列表。

这些值也可以通过`LABELS`、`VALUESA`、`VALUESB`、`VALUESC = ZIP(*DATA)`进行扩展。

1.  创建第一个子图：

```py
>>> plt.subplot(2, 1, 1)
```

1.  创建一个关于`VALUESA`的条形图：

```py
>>> valueA = plt.bar(POS, VALUESA)
>>> plt.ylabel('Sales A')
```

1.  创建一个不同的*Y*轴，并将`VALUESB`的信息添加为线图：

```py
>>> plt.twinx()
>>> valueB = plt.plot(POS, VALUESB, 'o-', color='red')
>>> plt.ylabel('Sales B')
>>> plt.xticks(POS, LABELS)
```

1.  创建另一个子图，并用`VALUESC`填充它：

```py
>>> plt.subplot(2, 1, 2)
>>> plt.plot(POS, VALUESC)
>>> plt.gca().set_ylim(ymin=0)
>>> plt.xticks(POS, LABELS)
```

1.  显示图表：

```py
>>> plt.show()
```

1.  结果将显示在一个新窗口中：

![](assets/ba4d0d65-86cb-462a-a017-9b5bc32a8d21.png)

# 它是如何工作的…

导入模块后，数据以一种方便的方式呈现在“如何做…”部分的第2步中，这很可能类似于数据最初的存储方式。第3步是一个准备步骤，将数据分割成不同的数组，以便进行下一步。

第4步创建一个新的`.subplot`。这将把整个图形分成两个元素。参数是行数、列数和所选的子图。因此，我们在一列中创建了两个子图，并在第一个子图中绘制。

第5步使用`VALUESA`数据在此子图中打印了一个`.bar`图，并使用`.ylabel`标记了*Y*轴为`Sales A`。

第6步使用`.twinx`创建一个新的*Y*轴，通过`.plot`绘制`VALUESB`为线图。标签使用`.ylabel`标记为`Sales B`。使用`.xticks`标记*X*轴。

`VALUESB`图形设置为红色，以避免两个图形具有相同的颜色。默认情况下，两种情况下的第一种颜色是相同的，这将导致混淆。数据点使用`'o'`选项标记。

在第7步中，我们使用`.subplot`切换到第二个子图。图形以线条形式打印`VALUESC`，然后使用`.xticker`在*X*轴上放置标签，并将*Y*轴的最小值设置为`0`。然后在第8步显示图形。

# 还有更多...

通常情况下，具有多个轴的图形很难阅读。只有在有充分理由这样做并且数据高度相关时才使用它们。

默认情况下，线图中的*Y*轴将尝试呈现*Y*值的最小值和最大值之间的信息。通常截断轴不是呈现信息的最佳方式，因为它可能扭曲感知的差异。例如，如果图形从10到11，那么在10到11之间的值的变化可能看起来很重要，但这不到10%。将*Y*轴最小值设置为`0`，使用`plt.gca().set_ylim(ymin=0)`是一个好主意，特别是在有两个不同的轴时。

选择子图的调用将首先按行，然后按列进行，因此`.subplot(2, 2, 3)`将选择第一列，第二行的子图。

分割的子图网格可以更改。首先调用`.subplot(2, 2, 1)`和`.subplot(2, 2, 2)`，然后调用`.subplot(2, 1, 2)`，将在第一行创建两个小图和第二行一个较宽的图的结构。返回将覆盖先前绘制的子图。

完整的`matplotlib`文档可以在这里找到：[https://matplotlib.org/](https://matplotlib.org/)。特别是，图例指南在这里：[https://matplotlib.org/users/legend_guide.html#plotting-guide-legend](https://matplotlib.org/users/legend_guide.html#plotting-guide-legend)。有关注释的信息在这里：[https://matplotlib.org/users/annotations.html](https://matplotlib.org/users/annotations.html)。

# 另请参阅

+   *绘制多条线*教程

+   *可视化地图*教程

# 保存图表

一旦图表准备好，我们可以将其存储在硬盘上，以便在其他文档中引用。在本教程中，我们将看到如何以不同的格式保存图表。

# 准备工作

我们需要在虚拟环境中安装`matplotlib`：

```py
$ echo "matplotlib==2.2.2" >> requirements.txt
$ pip install -r requirements.txt
```

如果您使用的是macOS，可能会出现这样的错误——`RuntimeError: Python is not installed as a framework`。请参阅`matplolib`文档以了解如何解决此问题：[https://matplotlib.org/faq/osx_framework.html](https://matplotlib.org/faq/osx_framework.html)。

# 如何做…

1.  导入`matplotlib`：

```py
>>> import matplotlib.pyplot as plt
```

1.  准备要显示在图表上的数据，并将其拆分为不同的数组：

```py
>>> DATA = (
...    ('Q1 2017', 100),
...    ('Q2 2017', 150),
...    ('Q3 2017', 125),
...    ('Q4 2017', 175),
... )
>>> POS = list(range(len(DATA)))
>>> VALUES = [value for label, value in DATA]
>>> LABELS = [label for label, value in DATA]
```

1.  使用数据创建条形图：

```py
>>> plt.bar(POS, VALUES)
>>> plt.xticks(POS, LABELS)
>>> plt.ylabel('Sales')
```

1.  将图表保存到硬盘：

```py
>>> plt.savefig('data.png')
```

# 工作原理...

在*如何做…*部分的第1和第2步中导入和准备数据后，通过调用`.bar`在第3步生成图表。添加了一个`.ylabel`，并通过`.xticks`标记了*X*轴的适当时间描述。

第4步将文件保存到硬盘上，文件名为`data.png`。

# 还有更多...

图像的分辨率可以通过`dpi`参数确定。这将影响文件的大小。使用`72`到`300`之间的分辨率。较低的分辨率将难以阅读，较高的分辨率除非图形的大小巨大，否则没有意义：

```py
>>> plt.savefig('data.png', dpi=72)
```

`matplotlib`了解如何存储最常见的文件格式，如JPEG、PDF和PNG。当文件名具有适当的扩展名时，它将自动使用。

除非您有特定要求，否则请使用PNG。与其他格式相比，它在存储具有有限颜色的图形时非常高效。如果您需要找到所有支持的文件，可以调用`plt.gcf().canvas.get_supported_filetypes()`。

完整的`matplotlib`文档可以在这里找到：[https://matplotlib.org/](https://matplotlib.org/)。特别是图例指南在这里：[https://matplotlib.org/users/legend_guide.html#plotting-guide-legend](https://matplotlib.org/users/legend_guide.html#plotting-guide-legend)。有关注释的信息在这里：[https://matplotlib.org/users/annotations.html](https://matplotlib.org/users/annotations.html)。

# 另请参阅

+   *绘制简单销售图*配方

+   *添加图例和注释*配方
