处理数据和统计

对于需要分析数据的人来说，Python 最吸引人的特点之一是数据操作和分析软件包的庞大生态系统，以及与 Python 合作的数据科学家活跃的社区。Python 使用起来很容易，同时还提供非常强大、快速的库，使得即使是相对新手的程序员也能够快速、轻松地处理大量数据。许多数据科学软件包和工具的核心是 pandas 库。Pandas 提供了两种数据容器类型，它们建立在 NumPy 数组的基础上，并且对于标签（除了简单的整数）有很好的支持。它们还使得处理大量数据变得非常容易。

统计学是使用数学—特别是概率—理论对数据进行系统研究。统计学有两个方面。第一个是找到描述一组数据的数值，包括数据的中心（均值或中位数）和离散程度（标准差或方差）等特征。统计学的第二个方面是推断，使用相对较小的样本数据集来描述一个更大的数据集（总体）。

在本章中，我们将看到如何利用 Python 和 pandas 处理大量数据并进行统计测试。

本章包含以下示例：

+   创建 Series 和 DataFrame 对象

+   从 DataFrame 中加载和存储数据

+   在数据框中操作数据

+   从 DataFrame 绘制数据

+   从 DataFrame 获取描述性统计信息

+   使用抽样了解总体

+   使用 t 检验来测试假设

+   使用方差分析进行假设检验

+   对非参数数据进行假设检验

+   使用 Bokeh 创建交互式图表

# 技术要求

在本章中，我们将主要使用 pandas 库进行数据操作，该库提供了类似于 R 的数据结构，如 `Series` 和 `DataFrame` 对象，用于存储、组织和操作数据。我们还将在本章的最后一个示例中使用 Bokeh 数据可视化库。这些库可以使用您喜欢的软件包管理器（如 pip）进行安装：

```py
          python3.8 -m pip install pandas bokeh

```

我们还将使用 NumPy 和 SciPy 软件包。

本章的代码可以在 GitHub 代码库的 `Chapter 06` 文件夹中找到：[`github.com/PacktPublishing/Applying-Math-with-Python/tree/master/Chapter%2006`](https://github.com/PacktPublishing/Applying-Math-with-Python/tree/master/Chapter%2006)。

查看以下视频以查看代码示例：[`bit.ly/2OQs6NX`](https://bit.ly/2OQs6NX)。

# 创建 Series 和 DataFrame 对象

Python 中的大多数数据处理都是使用 pandas 库完成的，它构建在 NumPy 的基础上，提供了类似于 R 的数据结构来保存数据。这些结构允许使用字符串或其他 Python 对象而不仅仅是整数来轻松索引行和列。一旦数据加载到 pandas 的 `DataFrame` 或 `Series` 中，就可以轻松地进行操作，就像在电子表格中一样。这使得 Python 结合 pandas 成为处理和分析数据的强大工具。

在本示例中，我们将看到如何创建新的 pandas `Series` 和 `DataFrame` 对象，并访问 `Series` 或 `DataFrame` 中的项目。

## 准备工作

对于这个示例，我们将使用以下命令导入 pandas 库：

```py
import pandas as pd
```

NumPy 软件包是 `np`。我们还可以从 NumPy 创建一个（种子）随机数生成器，如下所示：

```py
from numpy.random import default_rng
rng = default_rng(12345)
```

## 如何做...

以下步骤概述了如何创建包含数据的 `Series` 和 `DataFrame` 对象：

1.  首先，创建我们将存储在 `Series` 和 `DataFrame` 对象中的随机数据：

```py
diff_data = rng.normal(0, 1, size=100)
cumulative = np.add.accumulate(diff_data)
```

1.  接下来，创建一个包含 `diff_data` 的 `Series` 对象。我们将打印 `Series` 以查看数据的视图：

```py
data_series = pd.Series(diff_data)
print(data_series)
```

1.  现在，创建一个具有两列的 `DataFrame` 对象：

```py
data_frame = pd.DataFrame({
   "diffs": data_series,
    "cumulative": cumulative
}) 
```

1.  打印 `DataFrame` 对象以查看其包含的数据：

```py
print(data_frame)
```

## 它是如何工作的...

pandas 包提供了`Series`和`DataFrame`类，它们反映了它们的 R 对应物的功能和能力。`Series`用于存储一维数据，如时间序列数据，`DataFrame`用于存储多维数据；您可以将`DataFrame`对象视为"电子表格"。

`Series`与简单的 NumPy `ndarray`的区别在于`Series`索引其项的方式。NumPy 数组由整数索引，这也是`Series`对象的默认索引。但是，`Series`可以由任何可散列的 Python 对象索引，包括字符串和`datetime`对象。这使得`Series`对于存储时间序列数据非常有用。`Series`可以以多种方式创建。在这个示例中，我们使用了 NumPy 数组，但是任何 Python 可迭代对象，如列表，都可以替代。

DataFrame 对象中的每一列都是包含行的系列，就像传统数据库或电子表格中一样。在这个示例中，当通过字典的键构造 DataFrame 对象时，列被赋予标签。

`DataFrame`和`Series`对象在打印时会创建它们包含的数据的摘要。这包括列名、行数和列数，以及框架（系列）的前五行和最后五行。这对于快速获取对象和包含的数据的概述非常有用。

## 还有更多...

`Series`对象的单个行（记录）可以使用通常的索引符号通过提供相应的索引来访问。我们还可以使用特殊的`iloc`属性对象按其数值位置访问行。这允许我们按照它们的数值（整数）索引访问行，就像 Python 列表或 NumPy 数组一样。

可以使用通常的索引符号访问`DataFrame`对象中的列，提供列的名称。这样做的结果是一个包含所选列数据的`Series`对象。DataFrames 还提供了两个属性，可以用来访问数据。`loc`属性提供对个别行的访问，无论这个对象是什么。`iloc`属性提供按数值索引访问行，就像`Series`对象一样。

您可以向`loc`（或只使用对象的索引符号）提供选择条件来选择数据。这包括单个标签、标签列表、标签切片或布尔数组（适当大小的数组）。`iloc`选择方法接受类似的条件。

除了我们在这里描述的简单方法之外，还有其他从 Series 或 DataFrame 对象中选择数据的方法。例如，我们可以使用`at`属性来访问对象中指定行（和列）的单个值。

## 另请参阅

pandas 文档包含了创建和索引 DataFrame 或 Series 对象的不同方法的详细描述，网址为[`pandas.pydata.org/docs/user_guide/indexing.html`](https://pandas.pydata.org/docs/user_guide/indexing.html)。

# 从 DataFrame 加载和存储数据

在 Python 会话中从原始数据创建 DataFrame 对象是相当不寻常的。实际上，数据通常来自外部来源，如现有的电子表格或 CSV 文件、数据库或 API 端点。因此，pandas 提供了许多用于加载和存储数据到文件的实用程序。pandas 支持从 CSV、Excel（xls 或 xlsx）、JSON、SQL、Parquet 和 Google BigQuery 加载和存储数据。这使得将数据导入 pandas 然后使用 Python 操纵和分析这些数据变得非常容易。

在这个示例中，我们将看到如何将数据加载和存储到 CSV 文件中。加载和存储数据到其他文件格式的指令将类似。

## 做好准备

对于这个示例，我们需要导入 pandas 包作为`pd`别名和 NumPy 库作为`np`，并使用以下命令创建一个默认的随机数生成器：

```py
from numpy.random import default_rng
rng = default_rng(12345) # seed for example
```

## 如何做...

按照以下步骤将数据存储到文件，然后将数据加载回 Python：

1.  首先，我们将使用随机数据创建一个样本`DataFrame`对象。然后打印这个`DataFrame`对象，以便我们可以将其与稍后将要读取的数据进行比较：

```py
diffs = rng.normal(0, 1, size=100)
cumulative = np.add.accumulate(diffs)

data_frame = pd.DataFrame({
    "diffs": diffs, 
    "cumulative": cumulative
})
print(data_frame)
```

1.  我们将使用`DataFrame`对象中的数据将数据存储到`sample.csv`文件中，使用`DataFrame`对象上的`to_csv`方法。我们将使用`index=False`关键字参数，以便索引不存储在 CSV 文件中：

```py
data_frame.to_csv("sample.csv", index=False)
```

1.  现在，我们可以使用 pandas 中的`read_csv`例程将`sample.csv`文件读入一个新的`DataFrame`对象。我们将打印这个对象以显示结果：

```py
df = pd.read_csv("sample.csv", index_col=False)
print(df)
```

## 它是如何工作的...

这个示例的核心是 pandas 中的`read_csv`例程。这个例程以路径或类文件对象作为参数，并将文件的内容读取为 CSV 数据。我们可以使用`sep`关键字参数自定义分隔符，默认为逗号（`,`）。还有一些选项可以自定义列标题和自定义每列的类型。

`DataFrame`或`Series`中的`to_csv`方法将内容存储到 CSV 文件中。我们在这里使用了`index`关键字参数，以便索引不会打印到文件中。这意味着 pandas 将从 CSV 文件中的行号推断索引。如果数据是由整数索引的，这种行为是可取的，但如果数据是由时间或日期索引的，情况可能不同。我们还可以使用这个关键字参数来指定 CSV 文件中的哪一列是索引列。

## 另请参阅

请参阅 pandas 文档，了解支持的文件格式列表[`pandas.pydata.org/docs/reference/io.html`](https://pandas.pydata.org/docs/reference/io.html)。

# 在 DataFrames 中操作数据

一旦我们在`DataFrame`中有了数据，我们经常需要对数据应用一些简单的转换或过滤，然后才能进行任何分析。例如，这可能包括过滤缺少数据的行或对单独的列应用函数。

在这个示例中，我们将看到如何对`DataFrame`对象执行一些基本操作，以准备数据进行分析。

## 准备工作

对于这个示例，我们需要导入`pandas`包并使用`pd`别名，导入 NumPy 包并使用`np`别名，并使用以下命令从 NumPy 创建一个默认随机数生成器对象：

```py
from numpy.random import default_rng
rng = default_rng(12345)
```

## 如何做...

以下步骤说明了如何对 pandas 的`DataFrame`执行一些基本的过滤和操作：

1.  我们将首先使用随机数据创建一个样本`DataFrame`：

```py
three = rng.uniform(-0.2, 1.0, size=100)
three[three < 0] = np.nan

data_frame = pd.DataFrame({
    "one": rng.random(size=100),
    "two": np.add.accumulate(rng.normal(0, 1, size=100)),
    "three": three
})
```

1.  接下来，我们必须从现有列生成一个新列。这个新列将在相应的列“one”的条目大于`0.5`时保持`True`，否则为`False`：

```py
data_frame["four"] = data_frame["one"] > 0.5
```

1.  现在，我们必须创建一个新的函数，我们将应用到我们的`DataFrame`上。这个函数将把行“two”的值乘以行“one”和`0.5`的最大值（有更简洁的编写这个函数的方法）：

```py
def transform_function(row):
    if row["four"]:
        return 0.5*row["two"]
    return row["one"]*row["two"]
```

1.  现在，我们将对 DataFrame 中的每一行应用先前定义的函数以生成一个新列。我们还将打印更新后的 DataFrame，以便稍后进行比较：

```py
data_frame["five"] = data_frame.apply(transform_function, axis=1)
print(data_frame)
```

1.  最后，我们必须过滤掉 DataFrame 中包含**NaN**值的行。我们将打印结果 DataFrame：

```py
df = data_frame.dropna()
print(df)
```

## 它是如何工作的...

可以通过简单地将它们分配给新的列索引来向现有的`DataFrame`添加新列。但是，在这里需要注意一些问题。在某些情况下，pandas 会创建一个“视图”到`DataFrame`对象，而不是复制，这种情况下，分配给新列可能不会产生预期的效果。这在 pandas 文档中有所讨论（[`pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy`](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy)）。

Pandas `Series`对象（`DataFrame`中的列）支持丰富的比较运算符，如等于、小于或大于（在本示例中，我们使用了大于运算符）。这些比较运算符返回一个包含布尔值的`Series`，对应于比较为真和假的位置。这可以用来索引原始`Series`，并只获取比较为真的行。在本示例中，我们简单地将这个布尔值的`Series`添加到原始的`DataFrame`中。

`apply`方法接受一个函数（或其他可调用函数）并将其应用于 DataFrame 中的每一列。在本示例中，我们希望将函数应用于每一行，因此我们使用了`axis=1`关键字参数将函数应用于 DataFrame 中的每一行。无论哪种情况，函数都提供了一个由行（列）索引的`Series`对象。我们还将函数应用于每一行，返回使用每一行数据计算的值。实际上，如果 DataFrame 包含大量行，这种应用会相当慢。如果可能的话，应该整体操作列，使用设计用于操作 NumPy 数组的函数，以获得更好的效率。这对于在 DataFrame 的列中执行简单的算术运算尤其如此。就像 NumPy 数组一样，`Series`对象实现了标准的算术运算，这可以极大地提高大型 DataFrame 的操作时间。

在本示例的最后一步中，我们使用了`dropna`方法，快速选择了不包含 NaN 值的 DataFrame 行。Pandas 使用 NaN 来表示 DataFrame 中的缺失数据，因此这个方法选择了不包含缺失值的行。这个方法返回原始`DataFrame`对象的视图，但也可以通过传递`inplace=True`关键字参数来修改原始 DataFrame。在本示例中，这大致相当于使用索引表示法，使用包含布尔值的索引数组来选择行。

当直接修改原始数据时，应该始终谨慎，因为可能无法返回到原始数据以重复分析。如果确实需要直接修改数据，应确保数据已备份，或者修改不会删除以后可能需要的数据。

## 还有更多...

大多数 pandas 例程以明智的方式处理缺失数据（NaN）。然而，如果确实需要在 DataFrame 中删除或替换缺失数据，则有几种方法可以做到这一点。在本示例中，我们使用了`dropna`方法，简单地删除了缺失数据的行。我们也可以使用`fillna`方法填充所有缺失值，或者使用`interpolate`方法插值缺失值使用周围的值。

更一般地，我们可以使用`replace`方法来用其他值替换特定（非 NaN）值。这种方法可以处理数字值或字符串值，包括与正则表达式匹配。

`DataFrame`类有许多有用的方法。我们在这里只涵盖了非常基本的方法，但还有两个方法我们也应该提到。这些是`agg`方法和`merge`方法。

`agg`方法在 DataFrame 的给定轴上聚合一个或多个操作的结果。这允许我们通过应用聚合函数快速为每列（或行）生成摘要信息。输出是一个 DataFrame，其中应用的函数的名称作为行，所选轴的标签（例如列标签）作为列。

`merge`方法在两个 DataFrame 上执行类似 SQL 的连接。这将产生一个包含连接结果的新 DataFrame。可以传递各种参数给`how`关键字参数，以指定要执行的合并类型，默认为`inner`。应该将要执行连接的列或索引的名称传递给`on`关键字参数 - 如果两个`DataFrame`对象包含相同的键 - 或者传递给`left_on`和`right_on`。

# 从 DataFrame 绘制数据

与许多数学问题一样，找到可视化问题和所有信息的一种方法是制定策略。对于基于数据的问题，这通常意味着生成数据的图表，并在视觉上检查趋势、模式和基本结构。由于这是一个常见的操作，pandas 提供了一个快速简单的接口，可以直接从`Series`或`DataFrame`中以各种形式使用 Matplotlib 默认情况下的底层绘制数据。

在本教程中，我们将看到如何直接从`DataFrame`或`Series`绘制数据，以了解其中的趋势和结构。

## 准备工作

对于本教程，我们将需要导入 pandas 库为`pd`，导入 NumPy 库为`np`，导入 matplotlib 的`pyplot`模块为`plt`，并使用以下命令创建一个默认的随机数生成器实例：

```py
from numpy.random import default_rng
rng = default_rng(12345)
```

## 操作步骤...

按照以下步骤使用随机数据创建一个简单的 DataFrame，并绘制其包含的数据的图表：

1.  使用随机数据创建一个示例 DataFrame：

```py
diffs = rng.standard_normal(size=100)
walk = np.add.accumulate(diffs)
df = pd.DataFrame({
    "diffs": diffs,
    "walk": walk
})
```

1.  接下来，我们必须创建一个空白图，准备好绘图的两个子图：

```py
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
```

1.  我们必须将`walk`列绘制为标准折线图。这是通过在`Series`（列）对象上使用`plot`方法而不使用其他参数来完成的。我们将通过传递`ax=ax1`关键字参数来强制在`ax1`上绘图：

```py
df["walk"].plot(ax=ax1, title="Random walk")
ax1.set_xlabel("Index")
ax1.set_ylabel("Value")
```

1.  现在，我们必须通过将`kind="hist"`关键字参数传递给`plot`方法来绘制`diffs`列的直方图：

```py
df["diffs"].plot(kind="hist", ax=ax2, title="Histogram of diffs")
ax2.set_xlabel("Difference")
```

生成的图表如下所示：

![](img/32597d04-7669-4b87-a2cc-8ecb88435534.png)图 6.1 - DataFrame 中行走值和差异直方图的图表

## 工作原理...

`Series`（或`DataFrame`）上的`plot`方法是绘制其包含的数据与行索引的快速方法。`kind`关键字参数用于控制生成的图表类型，默认情况下是线图。有许多选项可用于绘图类型，包括`bar`用于垂直条形图，`barh`用于水平条形图，`hist`用于直方图（也在本教程中看到），`box`用于箱线图，`scatter`用于散点图。还有其他几个关键字参数可用于自定义生成的图表。在本教程中，我们还提供了`title`关键字参数，以向每个子图添加标题。

由于我们想要将两个图形放在同一图上，我们使用了`ax`关键字参数将各自的轴句柄传递给绘图例程。即使您让`plot`方法构建自己的图形，您可能仍然需要使用`plt.show`例程来显示具有某些设置的图形。

## 还有更多...

我们可以使用 pandas 接口生成几种常见类型的图表。除了本教程中提到的图表类型之外，还包括散点图、条形图（水平条形图和垂直条形图）、面积图、饼图和箱线图。`plot`方法还接受各种关键字参数来自定义图表的外观。

# 从 DataFrame 获取描述性统计信息

描述统计或汇总统计是与一组数据相关的简单值，例如平均值、中位数、标准差、最小值、最大值和四分位数。这些值以各种方式描述了数据集的位置和分布。平均值和中位数是数据的中心（位置）的度量，其他值则度量了数据相对于平均值和中位数的分布。这些统计数据对于理解数据集至关重要，并为许多分析技术奠定了基础。

在这个示例中，我们将看到如何为 DataFrame 中的每列生成描述性统计。

## 准备工作

为了这个示例，我们需要导入 pandas 包为 `pd`，导入 NumPy 包为 `np`，导入 matplotlib 的 `pyplot` 模块为 `plt`，并使用以下命令创建一个默认的随机数生成器：

```py
from numpy.random import default_rng
rng = default_rng(12345)
```

## 如何做到...

以下步骤展示了如何为 DataFrame 中的每一列生成描述性统计：

1.  我们首先创建一些样本数据，以便进行分析：

```py
uniform = rng.uniform(1, 5, size=100)
normal = rng.normal(1, 2.5, size=100)
bimodal = np.concatenate([rng.normal(0, 1, size=50), 
    rng.normal(6, 1, size=50)])
df = pd.DataFrame({
    "uniform": uniform, 
    "normal": normal, 
    "bimodal": bimodal
})
```

1.  接下来，我们绘制数据的直方图，以便了解 DataFrame 中数据的分布：

```py
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)

df["uniform"].plot(kind="hist", title="Uniform", ax=ax1)
df["normal"].plot(kind="hist", title="Normal", ax=ax2)
df["bimodal"].plot(kind="hist", title="Bimodal", ax=ax3, bins=20)
```

1.  Pandas `DataFrame` 对象有一个方法，可以为每列获取几个常见的描述性统计。`describe` 方法创建一个新的 DataFrame，其中列标题与原始对象相同，每行包含不同的描述性统计：

```py
descriptive = df.describe()
```

1.  我们还计算了*峰度*并将其添加到我们刚刚获得的新 DataFrame 中。我们还将描述性统计打印到控制台上，以查看这些值是什么：

```py
descriptive.loc["kurtosis"] = df.kurtosis()
print(descriptive)
#             uniform     normal    bimodal
# count    100.000000 100.000000 100.000000
# mean       2.813878   1.087146   2.977682
# std        1.093795   2.435806   3.102760
# min        1.020089  -5.806040  -2.298388
# 25%        1.966120  -0.498995   0.069838
# 50%        2.599687   1.162897   3.100215
# 75%        3.674468   2.904759   5.877905
# max        4.891319   6.375775   8.471313
# kurtosis  -1.055983   0.061679  -1.604305
```

1.  最后，我们在直方图上添加了垂直线，以说明每种情况下的平均值：

```py
uniform_mean = descriptive.loc["mean", "uniform"]
normal_mean = descriptive.loc["mean", "normal"]
bimodal_mean = descriptive.loc["mean", "bimodal"]
ax1.vlines(uniform_mean, 0, 20)
ax2.vlines(uniform_mean, 0, 25)
ax3.vlines(uniform_mean, 0, 20)
```

结果直方图如下所示：

![](img/46b543ec-8e94-4bb2-8246-3bb9db0e213c.png)图 6.2 – 三组数据的直方图及其平均值

## 工作原理...

`describe` 方法返回一个包含以下数据描述统计的 DataFrame：计数、平均值、标准差、最小值、25% 四分位数、中位数（50% 四分位数）、75% 四分位数和最大值。计数相当直观，最小值和最大值也是如此。平均值和中位数是数据的两种不同的“平均值”，大致代表了数据的中心值。平均值的定义是所有值的总和除以值的数量。我们可以用以下公式表示这个数量：

![](img/c536ac7d-473a-45ef-9c61-437e89e568c5.png)

这里，*x[i]* 值代表数据值，*N* 是值的数量。在这里，我们也采用了用条形表示平均值的常见符号。中位数是当所有数据排序时的“中间值”（如果值的数量是奇数，则取两个中间值的平均值）。25% 和 75% 的四分位数同样定义，但是取排序后数值的 25% 或 75% 处的值。你也可以将最小值看作是 0% 四分位数，最大值看作是 100% 四分位数。

**标准差**是数据相对于平均值的分布的度量，与统计学中经常提到的另一个量**方差**有关。方差是标准差的平方，定义如下：

![](img/f014bad4-8c1c-4559-bd40-9607e5653105.png)

你可能还会看到这里的分数中出现了 *N –* 1，这是从样本中估计总体参数时的**偏差**校正。我们将在下一个示例中讨论总体参数及其估计。标准差、方差、四分位数、最大值和最小值描述了数据的分布。例如，如果最大值是 5，最小值是 0，25% 四分位数是 2，75% 四分位数是 4，那么这表明大部分（实际上至少有 50% 的值）数据集中在 2 和 4 之间。

*kurtosis*是衡量数据在分布的“尾部”（远离平均值）集中程度的指标。这不像我们在本教程中讨论的其他数量那样常见，但在一些分析中确实会出现。我们在这里包括它主要是为了演示如何计算不出现在`describe`方法返回的 DataFrame 中的摘要统计值，使用适当命名的方法——在这里是`kurtosis`。当然，还有单独的方法来计算平均值（`mean`）、标准差（`std`）和`describe`方法中的其他数量。

当 pandas 计算本教程中描述的数量时，它将自动忽略由 NaN 表示的任何“缺失值”。这也将反映在描述性统计中报告的计数中。

## 还有更多...

我们在统计中包含的第三个数据集说明了查看数据的重要性，以确保我们计算的值是合理的。事实上，我们计算的平均值约为`2.9`，但通过查看直方图，很明显大部分数据与这个值相差甚远。我们应该始终检查我们计算的摘要统计数据是否准确地总结了样本中的数据。仅仅引用平均值可能会给出样本的不准确表示。

# 使用抽样理解人口

统计学中的一个核心问题是对整个人口的分布进行估计，并量化这些估计的准确程度，只给出一个小（随机）样本。一个经典的例子是，在测量随机选择的人群的身高时，估计一个国家所有人的平均身高。当通常意味着整个人口的平均值的真实人口分布无法被测量时，这种问题尤其有趣。在这种情况下，我们必须依靠我们对统计学的知识和一个（通常要小得多的）随机选择的样本来估计真实的人口平均值和标准差，并量化我们估计的准确程度。后者是导致广泛世界中统计学的混淆、误解和错误表述的根源。

在本教程中，我们将看到如何估计总体均值，并为这些估计提供**置信区间**。

## 准备工作

对于本教程，我们需要导入 pandas 包作为`pd`，从 Python 标准库导入`math`模块，以及使用以下命令导入 SciPy 的`stats`模块：

```py
from scipy import stats
```

## 操作步骤...

在接下来的步骤中，我们将根据随机选择的 20 个人的样本，对英国男性的平均身高进行估计：

1.  我们必须将我们的样本数据加载到 pandas 的`Series`中：

```py
sample_data = pd.Series(
    [172.3, 171.3, 164.7, 162.9, 172.5, 176.3, 174.8, 171.9, 
     176.8, 167.8, 164.5, 179.7, 157.8, 170.6, 189.9, 185\. , 
     172.7, 165.5, 174.5, 171.5]
)
```

1.  接下来，我们将计算样本均值和标准差：

```py
sample_mean = sample_data.mean()
sample_std = sample_data.std()
print(f"Mean {sample_mean}, st. dev {sample_std}")
# Mean 172.15, st. dev 7.473778724383846
```

1.  然后，我们将计算**标准误差**，如下所示：

```py
N = sample_data.count()
std_err = sample_std/math.sqrt(N)
```

1.  我们将计算我们从学生*t*分布中所需的置信值的**临界值**：

```py
cv_95, cv_99 = stats.t.ppf([0.975, 0.995], df=N-1)
```

1.  现在，我们可以使用以下代码计算真实总体均值的 95%和 99%置信区间：

```py
pm_95 = cv_95*std_err
conf_interval_95 = [sample_mean - pm_95, sample_mean + pm_95]
pm_99 = cv_99*std_err
conf_interval_99 = [sample_mean - pm_99, sample_mean + pm_99]

print("95% confidence", conf_interval_95)
# 95% confidence [168.65216388659374, 175.64783611340627]
print("99% confidence", conf_interval_99)
# 99% confidence [167.36884119608774, 176.93115880391227]
```

## 它是如何工作的...

参数估计的关键是正态分布，我们在第四章中讨论过。如果我们找到*z*的临界值，使得标准正态分布随机数小于这个值*z*的概率为 97.5%，那么这样的数值在*-z*和*z*之间的概率为 95%（每个尾部为 2.5%）。这个*z*的临界值结果为 1.96，四舍五入到 2 位小数。也就是说，我们可以有 95%的把握，标准正态分布随机数的值在*-z*和*z*之间。类似地，99%置信的临界值为 2.58（四舍五入到 2 位小数）。

如果我们的样本是“大”的，我们可以引用**中心极限定理**，它告诉我们，即使总体本身不服从正态分布，从这个总体中抽取的随机样本的均值将服从与整个总体相同均值的正态分布。然而，这仅在我们的样本足够大的情况下才有效。在这个配方中，样本并不大——它只有 20 个值，与英国男性总体相比显然不大。这意味着，我们不得不使用具有*N-*1 自由度的学生*t*分布来找到我们的临界值，而不是正态分布，其中*N*是我们样本的大小。为此，我们使用 SciPy `stats`模块中的`stats.t.ppf`例程。

学生*t*分布与正态分布有关，但有一个参数——自由度——它改变了分布的形状。随着自由度的增加，学生*t*分布将越来越像正态分布。你认为分布足够相似的点取决于你的应用和你的数据。一个经验法则说，样本大小为 30 足以引用中心极限定理，并简单使用正态分布，但这绝不是一个好的规则。在基于样本进行推断时，你应该非常小心，特别是如果样本与总体相比非常小。（显然，如果总体由 30 人组成，使用 20 个样本量将是相当描述性的，但如果总体由 3000 万人组成，情况就不同了。）

一旦我们有了临界值，真实总体均值的置信区间可以通过将临界值乘以样本的标准误差，并从样本均值中加减得出。标准误差是对给定样本大小的样本均值分布与真实总体均值之间的差距的近似。这就是为什么我们使用标准误差来给出我们对总体均值的估计的置信区间。当我们将标准误差乘以从学生*t*分布中取得的临界值（在这种情况下）时，我们得到了在给定置信水平下观察到的样本均值与真实总体均值之间的最大差异的估计。

在这个配方中，这意味着我们有 95%的把握，英国男性的平均身高在 168.7 厘米和 175.6 厘米之间，我们有 99%的把握，英国男性的平均身高在 167.4 厘米和 176.9 厘米之间。事实上，我们的样本是从一个平均身高为 175.3 厘米，标准偏差为 7.2 厘米的人群中抽取的。这个真实的平均值（175.3 厘米）确实位于我们两个置信区间内，但仅仅是刚好。

## 参见

有一个有用的包叫做`uncertainties`，用于进行涉及一定不确定性的值的计算。请参阅第十章中的*计算中的不确定性*配方，*其他主题*。

# 使用 t 检验进行假设检验

统计学中最常见的任务之一是在从总体中收集样本数据的情况下，测试关于正态分布总体均值的假设的有效性。例如，在质量控制中，我们可能希望测试在工厂生产的一张纸的厚度是否为 2 毫米。为了测试这一点，我们将随机选择样本纸张并测量厚度以获得我们的样本数据。然后，我们可以使用**t 检验**来测试我们的零假设*H[0]*，即纸张的平均厚度为 2 毫米，对抗备择假设*H[1]*，即纸张的平均厚度不是 2 毫米。我们使用 SciPy 的`stats`模块来计算*t*统计量和*p*值。如果*p*值小于 0.05，则我们接受零假设，显著性为 5%（置信度 95%）。如果*p*值大于 0.05，则我们必须拒绝零假设，支持备择假设。

*在这个步骤中，我们将看到如何使用 t 检验来测试给定样本的假设总体均值是否有效。

## 准备工作

对于这个步骤，我们需要导入 pandas 包作为`pd`，并使用以下命令导入 SciPy 的`stats`模块：

```py
from scipy import stats
```

## 如何做...

按照以下步骤使用 t 检验来测试给定一些样本数据的假设总体均值的有效性：

1.  我们首先将数据加载到 pandas 的`Series`中：

```py
sample = pd.Series([
    2.4, 2.4, 2.9, 2.6, 1.8, 2.7, 2.6, 2.4, 2.8, 2.4, 2.4,
    2.4, 2.7, 2.7, 2.3, 2.4, 2.4, 3.2, 2.2, 2.5, 2.1, 1.8,
    2.9, 2.5, 2.5, 3.2, 2\. , 2.3, 3\. , 1.5, 3.1, 2.5, 3.1,
    2.4, 3\. , 2.5, 2.7, 2.1, 2.3, 2.2, 2.5, 2.6, 2.5, 2.8,
    2.5, 2.9, 2.1, 2.8, 2.1, 2.3
])
```

1.  现在，设置我们将进行测试的假设总体均值和显著性水平：

```py
mu0 = 2.0
significance = 0.05
```

1.  接下来，使用 SciPy 的`stats`模块中的`ttest_1samp`例程生成*t*统计量和*p*值：

```py
t_statistic, p_value = stats.ttest_1samp(sample, mu0)
print(f"t stat: {t_statistic}, p value: {p_value}")
# t stat: 9.752368720068665, p value: 4.596949515944238e-13
```

1.  最后，测试*p*值是否小于我们选择的显著性水平：

```py
if p_value <= significance:
    print("Reject H0 in favour of H1: mu != 2.0")
else:
    print("Accept H0: mu = 2.0")
# Reject H0 in favour of H1: mu != 2.0
```

## 它是如何工作的...

*t*统计量是使用以下公式计算的：

![](img/e9dc48d2-75d6-4d0c-a377-3b9a6ecbd9d1.png)

在这里，*μ[0]*是假设均值（来自零假设），*x* bar 是样本均值，*s*是样本标准差，*N*是样本大小。*t*统计量是观察到的样本均值与假设总体均值*μ[0]*之间差异的估计，通过标准误差进行归一化。假设总体呈正态分布，*t*统计量将遵循*N*-1 自由度的*t*分布。查看 t 统计量在相应的学生*t*分布中的位置，可以让我们了解我们观察到的样本均值来自具有假设均值的总体的可能性。这以*p*值的形式给出。

*p*值是观察到比我们观察到的样本均值更极端值的概率，假设总体均值等于*μ[0]*。如果*p*值小于我们选择的显著性值，那么我们不能期望真实的总体均值是我们假设的值*μ[0]*。在这种情况下，我们必须接受备择假设，即真实的总体均值不等于*μ[0]*。

## 还有更多...

在这个步骤中我们演示的测试是 t 检验的最基本用法。在这里，我们比较了样本均值和假设的总体均值，以决定整个总体的均值是否合理为假设值。更一般地，我们可以使用 t 检验来比较从每个样本中取出的两个独立总体的**2 样本 t 检验**，或者使用**配对 t 检验**来比较数据成对（某种方式）的总体。这使得 t 检验成为统计学家的重要工具。

在统计学中，显著性和置信度是两个经常出现的概念。统计上显著的结果是指具有高正确概率的结果。在许多情境中，我们认为任何具有低于一定阈值（通常为 5%或 1%）的错误概率的结果都是统计上显著的。置信度是对结果的确定程度的量化。结果的置信度是 1 减去显著性。

不幸的是，结果的显著性经常被误用或误解。说一个结果在 5%的显著水平上是统计显著的，意味着我们有 5%的机会错误地接受了零假设。也就是说，如果我们从总体中另外抽取 20 个样本进行相同的测试，我们至少期望其中一个会给出相反的结果。然而，这并不意味着其中一个一定会这样做。

高显著性表明我们更加确信我们得出的结论是正确的，但这并不意味着这确实是情况。事实上，这个配方中找到的结果就是证据；我们使用的样本实际上是从均值为`2.5`，标准差为`0.35`的总体中抽取的。（在创建后对样本进行了一些四舍五入，这会稍微改变分布。）这并不意味着我们的分析是错误的，或者我们从样本得出的结论不正确。

重要的是要记住，t 检验只有在基础总体遵循正态分布，或者至少近似遵循正态分布时才有效。如果不是这种情况，那么您可能需要使用非参数检验。我们将在*测试非参数数据的假设*配方中讨论这一点。

# 使用 ANOVA 进行假设检验

假设我们设计了一个实验，测试两个新的过程与当前过程，并且我们想测试这些新过程的结果是否与当前过程不同。在这种情况下，我们可以使用**方差分析**（**ANOVA**）来帮助我们确定这三组结果的均值之间是否有任何差异（为此，我们需要假设每个样本都是从具有共同方差的正态分布中抽取的）。

*在这个配方中，我们将看到如何使用 ANOVA 来比较多个样本。

## 做好准备

对于这个配方，我们需要 SciPy 的`stats`模块。我们还需要使用以下命令创建一个默认的随机数生成器实例：

```py
from numpy.random import default_rng
rng = default_rng(12345)
```

## 如何做...

按照以下步骤执行（单向）ANOVA 测试，以测试三个不同过程之间的差异：

1.  首先，我们将创建一些样本数据，然后对其进行分析：

```py
current = rng.normal(4.0, 2.0, size=40)
process_a = rng.normal(6.2, 2.0, size=25)
process_b = rng.normal(4.5, 2.0, size=64)
```

1.  接下来，我们将为我们的测试设置显著性水平：

```py
significance = 0.05
```

1.  然后，我们将使用 SciPy 的`stats`模块中的`f_oneway`例程来生成 F 统计量和*p*值：

```py
F_stat, p_value = stats.f_oneway(current, process_a, process_b)
print(f"F stat: {F_stat}, p value: {p_value}")
# F stat: 9.949052026027028, p value: 9.732322721019206e-05
```

1.  现在，我们必须测试*p*值是否足够小，以确定我们是否应该接受或拒绝所有均值相等的零假设：

```py
if p_value <= significance:
    print("Reject H0: there is a difference between means")
else:
    print("Accept H0: all means equal")
# Reject H0: there is a difference between means
```

## 工作原理...

ANOVA 是一种强大的技术，可以同时比较多个样本。它通过比较样本的变化与总体变化的相对变化来工作。ANOVA 在比较三个或更多样本时特别强大，因为不会因运行多个测试而产生累积误差。不幸的是，如果 ANOVA 检测到不是所有的均值都相等，那么从测试信息中无法确定哪个样本与其他样本有显著差异。为此，您需要使用额外的测试来找出差异。

`f_oneway` SciPy `stats`包例程执行单向 ANOVA 测试——ANOVA 生成的检验统计量遵循 F 分布。同样，*p*值是来自测试的关键信息。如果*p*值小于我们预先设定的显著性水平（在这个配方中为 5%），我们接受零假设，否则拒绝零假设。

## 还有更多...

ANOVA 方法非常灵活。我们在这里介绍的单因素方差分析检验是最简单的情况，因为只有一个因素需要测试。双因素方差分析检验可用于测试两个不同因素之间的差异。例如，在药物临床试验中，我们测试对照组，同时也测量性别（例如）对结果的影响。不幸的是，SciPy 在`stats`模块中没有执行双因素方差分析的例程。您需要使用其他包，比如`statsmodels`包。我们将在第七章 *回归和预测* 中使用这个包。

如前所述，ANOVA 只能检测是否存在差异。如果存在显著差异，它无法检测这些差异发生在哪里。例如，我们可以使用 Durnett's 检验来测试其他样本均值是否与对照样本不同，或者使用 Tukey's 范围检验来测试每个组均值与其他每个组均值之间的差异。

# 非参数数据的假设检验

t 检验和 ANOVA 都有一个主要缺点：被抽样的总体必须遵循正态分布。在许多应用中，这并不太严格，因为许多真实世界的总体值遵循正态分布，或者一些规则，如中心极限定理，允许我们分析一些相关数据。然而，事实并非所有可能的总体值以任何合理的方式都遵循正态分布。对于这些（幸运地是罕见的）情况，我们需要一些替代的检验统计量来替代 t 检验和 ANOVA。

在这个配方中，我们将使用 Wilcoxon 秩和检验和 Kruskal-Wallis 检验来测试两个（或更多，在后一种情况下）总体之间的差异。

## 准备工作

对于这个配方，我们需要导入 pandas 包作为`pd`，SciPy 的`stats`模块，以及使用以下命令创建的默认随机数生成器实例：

```py
from numpy.random import default_rng
rng = default_rng(12345)
```

## 如何做...

按照以下步骤比较两个或更多个不服从正态分布的总体：

1.  首先，我们将生成一些样本数据用于分析：

```py
sample_A = rng.uniform(2.5, 4.5, size=22)
sample_B = rng.uniform(3.0, 4.4, size=25)
sample_C = rng.uniform(3.0, 4.4, size=30)
```

1.  接下来，我们设置在此分析中使用的显著性水平：

```py
significance = 0.05
```

1.  现在，我们使用`stats.kruskal`例程生成零假设的检验统计量和*p*值，即总体具有相同中位数值的零假设：

```py
statistic, p_value = stats.kruskal(sample_A, sample_B, sample_C)
print(f"Statistic: {statistic}, p value: {p_value}")
# Statistic: 5.09365664638392, p value: 0.07832970895845669
```

1.  我们将使用条件语句打印关于测试结果的声明：

```py
if p_value <= significance:
    print("Accept H0: all medians equal")
else:
    print("There are differences between population medians")
# There are differences between population medians
```

1.  现在，我们使用 Wilcoxon 秩和检验来获得每对样本之间比较的*p*值：

```py
_, p_A_B = stats.ranksums(sample_A, sample_B)
_, p_A_C = stats.ranksums(sample_A, sample_C)
_, p_B_C = stats.ranksums(sample_B, sample_C)
```

1.  接下来，我们使用条件语句打印出针对那些表明存在显著差异的比较的消息：

```py
if p_A_B > significance:
    print("Significant differences between A and B, p value", 
        p_A_B)
# Significant differences between A and B, p value
     0.08808151166219029

if p_A_C > significance:
    print("Significant differences between A and C, p value",
        p_A_C)
# Significant differences between A and C, p value 
     0.4257804790323789

if p_B_C > significance:
    print("Significant differences between B and C, p value",
        p_B_C) 
else:
    print("No significant differences between B and C, p value",
        p_B_C)
# No significant differences between B and C, p value
     0.037610047044153536

```

## 工作原理...

如果从数据抽样的总体不遵循可以用少量参数描述的分布，我们称数据为非参数数据。这通常意味着总体不是正态分布，但比这更广泛。在这个配方中，我们从均匀分布中抽样，但这仍然比通常需要非参数检验时更有结构化的例子。非参数检验可以和应该在我们不确定基础分布的任何情况下使用。这样做的代价是检验略微不够有力。

任何（真实）分析的第一步应该是绘制数据的直方图并通过视觉检查分布。如果你从一个正态分布的总体中抽取一个随机样本，你可能也期望样本是正态分布的（我们在本书中已经看到了几次）。如果你的样本显示出正态分布的典型钟形曲线，那么总体本身很可能也是正态分布的。你还可以使用**核密度估计**图来帮助确定分布。这在 pandas 绘图界面上可用，`kind="kde"`。如果你仍然不确定总体是否正态分布，你可以应用统计测试，比如 D'Agostino 的 K 平方检验或 Pearson 的卡方检验。这两个测试被合并成一个用于正态性检验的单一程序，称为 SciPy `stats`模块中的`normaltest`，还有其他几个正态性测试。

Wilcoxon 秩和检验——也称为 Mann-Whitney U 检验——是双样本 t 检验的非参数替代方法。与 t 检验不同，秩和检验不会比较样本均值，以量化两个总体是否具有不同分布。相反，它将样本数据组合并按大小排序。检验统计量是从具有最少元素的样本的秩的总和生成的。从这里开始，像往常一样，我们为零假设生成一个*p*值，即两个总体具有相同分布的假设。

Kruskal-Wallis 检验是一种一元 ANOVA 检验的非参数替代方法。与秩和检验一样，它使用样本数据的排名来生成检验统计量和零假设的*p*值，即所有总体具有相同中位数值的假设。与一元 ANOVA 一样，我们只能检测所有总体是否具有相同的中位数，而不能确定差异在哪里。为此，我们需要使用额外的测试。

在这个实验中，我们使用了 Kruskal-Wallis 检验来确定与我们三个样本对应的总体之间是否存在显著差异。我们发现了一个*p*值为`0.07`的差异，这离 5%的显著性并不远。然后我们使用秩和检验来确定总体之间的显著差异发生在哪里。在这里，我们发现样本 A 与样本 B 和 C 存在显著差异，而样本 B 和 C 之间没有显著差异。考虑到这些样本的生成方式，这并不奇怪。

不幸的是，由于我们在这个实验中使用了多个测试，我们对结论的整体信心并不像我们期望的那样高。我们进行了四次测试，置信度为 95%，这意味着我们对结论的整体信心仅约为 81%。这是因为错误在多次测试中累积，降低了整体信心。为了纠正这一点，我们需要调整每个测试的显著性阈值，使用 Bonferroni 校正（或类似方法）。

# 使用 Bokeh 创建交互式图表

测试统计和数值推理对于系统分析数据集是很好的。然而，它们并不能真正给我们一个数据集的整体图像，就像图表那样。数值是确定的，但在统计学中可能很难理解，而图表可以立即说明数据集之间的差异和趋势。因此，有大量的库用于以越来越有创意的方式绘制数据。一个特别有趣的用于生成数据图的包是 Bokeh，它允许我们通过利用 JavaScript 库在浏览器中创建交互式图。

在这个实验中，我们将看到如何使用 Bokeh 创建一个可以在浏览器中显示的交互式图。

## 准备工作

对于这个示例，我们需要将 pandas 包导入为`pd`，将 NumPy 包导入为`np`，使用以下代码构建默认随机数生成器的实例，并从 Bokeh 导入`plotting`模块，我们使用`bk`别名导入：

```py
from bokeh import plotting as bk
from numpy.random import default_rng
rng = default_rng(12345)
```

## 如何做...

这些步骤展示了如何使用 Bokeh 在浏览器中创建交互式绘图：

1.  我们首先需要创建一些样本数据来绘制：

```py
date_range = pd.date_range("2020-01-01", periods=50)
data = np.add.accumulate(rng.normal(0, 3, size=50))
series = pd.Series(data, index=date_range)
```

1.  接下来，我们使用`output_file`例程指定 HTML 代码的输出文件位置：

```py
bk.output_file("sample.html")
```

1.  现在，我们创建一个新的图，并设置标题和轴标签，并将*x*轴类型设置为`datetime`，以便我们的日期索引将被正确显示：

```py
fig = bk.figure(title="Time series data", 
                x_axis_label="date",
                x_axis_type="datetime",
                y_axis_label="value")
```

1.  我们将数据添加到图中作为一条线：

```py
fig.line(date_range, series)
```

1.  最后，我们使用`show`例程或`save`例程来保存或更新指定输出文件中的 HTML。我们在这里使用`show`来在浏览器中打开绘图：

```py
bk.show(fig)
```

Bokeh 绘图不是静态对象，应该通过浏览器进行交互。这里使用`matplotlib`重新创建了数据，以便进行比较：

![](img/dbe31bab-0892-4040-a72c-a2f9cd6c6900.png)图 6.3 - 使用 Matplotlib 创建的时间序列数据的绘图

## 它是如何工作的...

Bokeh 使用 JavaScript 库在浏览器中呈现绘图，使用 Python 后端提供的数据。这样做的好处是用户可以自行生成绘图。例如，我们可以放大以查看绘图中可能隐藏的细节，或者以自然的方式浏览数据。本示例只是展示了使用 Bokeh 可能性的一小部分。

`figure`例程创建一个代表绘图的对象，我们可以向其中添加元素，比如通过数据点的线条，就像我们向 matplotlib 的`Axes`对象添加绘图一样。在这个示例中，我们创建了一个简单的 HTML 文件，其中包含 JavaScript 代码来呈现数据。无论是保存还是调用`show`例程，这段 HTML 代码都会被转储到指定的文件中。在实践中，*p*值越小，我们对假设的总体均值正确性的信心就越大。

## 还有更多...

Bokeh 的功能远不止本文所描述的。Bokeh 绘图可以嵌入到文件中，比如 Jupyter 笔记本，这些文件也可以在浏览器中呈现，或者嵌入到现有的网站中。如果您使用的是 Jupyter 笔记本，您应该使用`output_notebook`例程，而不是`output_file`例程，将绘图直接打印到笔记本中。它有各种不同的绘图样式，支持在绘图之间共享数据（例如，可以在一个绘图中选择数据，并在其他绘图中突出显示），并支持流数据。

# 进一步阅读

统计学和统计理论的教科书有很多。以下书籍被用作本章的参考：

+   *Mendenhall, W., Beaver, R., and Beaver, B., (2006), Introduction To Probability And Statistics, 12th ed., (Belmont, Calif.: Thomson Brooks/Cole)*

pandas 文档([`pandas.pydata.org/docs/index.html`](https://pandas.pydata.org/docs/index.html))和以下 pandas 书籍是使用 pandas 的良好参考资料：

+   *McKinney, W.,*(*2017*),*Python for Data Analysis, 2nd ed.,*(*Sebastopol: O'Reilly Media, Inc,* *US*)

SciPy 文档([`docs.scipy.org/doc/scipy/reference/tutorial/stats.html`](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html))还包含了本章中多次使用的统计模块的详细信息。
