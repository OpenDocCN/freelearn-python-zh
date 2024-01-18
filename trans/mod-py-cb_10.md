# 第10章。统计编程和线性回归

在本章中，我们将研究以下内容：

+   使用内置的统计库

+   计数器中值的平均值

+   计算相关系数

+   计算回归参数

+   计算自相关

+   确认数据是随机的-零假设

+   定位异常值

+   一次分析多个变量

# 介绍

数据分析和统计处理是复杂、现代编程语言非常重要的应用。这个领域非常广泛。Python生态系统包括许多附加包，提供了复杂的数据探索、分析和决策功能。

我们将研究一些我们可以使用Python内置库和数据结构进行的基本统计计算。我们将研究相关性的问题以及如何创建回归模型。

我们还将讨论随机性和零假设的问题。确保数据集中确实存在可测量的统计效应是至关重要的。如果不小心的话，我们可能会浪费大量的计算周期来分析无关紧要的噪音。

我们还将研究一种常见的优化技术。它有助于快速产生结果。一个设计不良的算法应用于非常大的数据集可能是一种无效的时间浪费。

![](image/614271.jpg)

# 使用内置的统计库

大量的**探索性数据分析**（**EDA**）涉及到对数据的摘要。有几种可能有趣的摘要：

+   **中心趋势**：诸如均值、众数和中位数等值可以描述数据集的中心。

+   **极值**：最小值和最大值和一些数据的中心度量一样重要。

+   **方差**：方差和标准差用于描述数据的分散程度。大方差意味着数据分布广泛；小方差意味着数据紧密聚集在中心值周围。

如何在Python中获得基本的描述性统计信息？

## 准备就绪

我们将研究一些可以用于统计分析的简单数据。我们得到了一个原始数据文件，名为`anscombe.json`。它是一个JSON文档，其中包含四个（*x*，*y*）对的系列。

我们可以用以下方法读取这些数据：

```py
 **>>> from pathlib import Path 
>>> import json 
>>> from collections import OrderedDict 
>>> source_path = Path('code/anscombe.json') 
>>> data = json.loads(source_path.read_text(), object_pairs_hook=OrderedDict)** 

```

我们已经定义了数据文件的`Path`。然后我们可以使用`Path`对象来从这个文件中读取文本。`json.loads()`使用这个文本从JSON数据构建Python对象。

我们已经包含了一个`object_pairs_hook`，这样这个函数将使用`OrderedDict`类而不是默认的`dict`类来构建JSON。这将保留源文档中项目的原始顺序。

我们可以这样检查数据：

```py
 **>>> [item['series'] for item in data] 
['I', 'II', 'III', 'IV'] 
>>> [len(item['data']) for item in data] 
[11, 11, 11, 11]** 

```

整个JSON文档是一个具有`I`和`II`等键的子文档序列。每个子文档有两个字段-`series`和`data`。在`data`值内，有一个我们想要描述的观察值列表。每个观察值都有一对值。

数据看起来是这样的：

```py
    [ 
      { 
        "series": "I", 
        "data": [ 
          { 
            "x": 10.0, 
            "y": 8.04 
          }, 
          { 
            "x": 8.0, 
            "y": 6.95 
          }, 
          ... 
        ] 
      }, 
      ... 
    ] 

```

这是一个典型的JSON文档的字典列表结构。每个字典都有一个系列名称，带有`series`键，并且一个数据值序列，带有`data`键。`data`中的列表是一系列项目，每个项目都有一个`x`和一个`y`值。

要在这个数据结构中找到特定的系列，我们有几种选择：

+   一个`for...if...return`语句序列：

```py
     **>>> def get_series(data, series_name): 
          for s in data: 
              if s['series'] == series_name: 
                  return s** 

    ```

这个`for`语句检查数值序列中的每个系列。系列是一个带有系列名称的键为`'series'`的字典。`if`语句将系列名称与目标名称进行比较，并返回第一个匹配项。对于未知的系列名称，这将返回`None`。

+   我们可以这样访问数据：

```py
     **>>> series_1 = get_series(data, 'I') 
          >>> series_1['series'] 
          'I' 
          >>> len(series_1['data']) 
          11** 

    ```

+   我们可以使用一个过滤器来找到所有匹配项，然后选择第一个：

```py
     **>>> def get_series(data, series_name): 
          ...     name_match = lambda series: series['series'] == series_name 
          ...     series = list(filter(name_match, data))[0] 
          ...     return series** 

    ```

这个`filter（）`函数检查值序列中的每个系列。该系列是一个带有“series”键的字典，其中包含系列名称。`name_match` lambda对象将比较系列的名称键与目标名称，并返回所有匹配项。这用于构建一个`list`对象。如果每个键都是唯一的，第一个项目就是唯一的项目。这将为未知的系列名称引发`IndexError`异常。

现在我们可以这样访问数据：

```py
     **>>> series_2 = get_series(data, 'II') 
          >>> series_2['series'] 
          'II' 
          >>> len(series_2['data']) 
          11** 

    ```

+   我们可以使用生成器表达式，类似于过滤器，找到所有匹配项。我们从结果序列中选择第一个：

```py
     **>>> def get_series(data, series_name): 
          ...     series = list( 
          ...         s for s in data 
          ...            if s['series'] == series_name 
          ...         )[0] 
          ...     return series** 

    ```

这个生成器表达式检查值序列中的每个系列。该系列是一个带有“series”键的字典，其中包含系列名称。表达式`s['series'] == series_name`将比较系列的名称键与目标名称，并传递所有匹配项。这用于构建一个`list`对象，并返回列表中的第一个项目。这将为未知的系列名称引发`IndexError`异常。

现在我们可以这样访问数据：

```py
     **>>> series_3 = get_series(data, 'III') 
          >>> series_3['series'] 
          'III' 
          >>> len(series_3['data']) 
          11** 

    ```

+   在[第8章](text00088.html#page "第8章。功能和响应式编程特性")的*实现“存在”处理*配方中有一些这种处理的示例，*功能和响应式编程特性*。一旦我们从数据中选择了一个系列，我们还需要从系列中选择一个变量。这可以通过生成器函数或生成器表达式来完成：

```py
     **>>> def data_iter(series, variable_name): 
          ...     return (item[variable_name] for item in series['data'])** 

    ```

系列字典具有带有数据值序列的“data”键。每个数据值都是一个具有两个键“x”和“y”的字典。这个“data_iter（）”函数将从数据中的每个字典中选择其中一个变量。这个函数将生成一系列值，可以用于详细分析：

```py
 **>>> s_4 = get_series(data, 'IV') 
>>> s_4_x = list(data_iter(s_4, 'x')) 
>>> len(s_4_x) 
11** 

```

在这种情况下，我们选择了系列`IV`。从该系列中，我们选择了每个观察的`x`变量。结果列表的长度向我们展示了该系列中有11个观察。

## 如何做...

1.  要计算均值和中位数，使用`statistics`模块：

```py
     **>>> import statistics 
          >>> for series_name in 'I', 'II', 'III', 'IV': 
          ...     series = get_series(data, series_name) 
          ...     for variable_name in 'x', 'y': 
          ...         samples = list(data_iter(series, variable_name)) 
          ...         mean = statistics.mean(samples) 
          ...         median = statistics.median(samples) 
          ...         print(series_name, variable_name, round(mean,2), median) 
          I x 9.0 9.0 
          I y 7.5 7.58 
          II x 9.0 9.0 
          II y 7.5 8.14 
          III x 9.0 9.0 
          III y 7.5 7.11 
          IV x 9.0 8.0 
          IV y 7.5 7.04** 

    ```

这使用`get_series（）`和`data_iter（）`从给定系列的一个变量中选择样本值。`mean（）`和`median（）`函数很好地处理了这个任务。有几种可用的中位数计算变体。

1.  要计算`mode`，使用`collections`模块：

```py
     **>>> import collections 
          >>> for series_name in 'I', 'II', 'III', 'IV': 
          ...     series = get_series(data, series_name) 
          ...     for variable_name in 'x', 'y': 
          ...         samples = data_iter(series, variable_name) 
          ...         mode = collections.Counter(samples).most_common(1) 
          ...         print(series_name, variable_name, mode) 
          I x [(4.0, 1)] 
          I y [(8.81, 1)] 
          II x [(4.0, 1)] 
          II y [(8.74, 1)] 
          III x [(4.0, 1)] 
          III y [(8.84, 1)] 
          IV x [(8.0, 10)] 
          IV y [(7.91, 1)]** 

    ```

这使用`get_series（）`和`data_iter（）`从给定系列的一个变量中选择样本值。`Counter`对象非常优雅地完成了这项工作。实际上，我们从这个操作中得到了一个完整的频率直方图。`most_common（）`方法的结果显示了值和它出现的次数。

我们还可以使用`statistics`模块中的`mode（）`函数。该函数的优点是在没有明显模式时引发异常。这的缺点是没有提供任何额外的信息来帮助定位多模态数据。

1.  极值是用内置的`min（）`和`max（）`函数计算的：

```py
     **>>> for series_name in 'I', 'II', 'III', 'IV': 
          ...     series = get_series(data, series_name) 
          ...     for variable_name in 'x', 'y': 
          ...         samples = list(data_iter(series, variable_name)) 
          ...         least = min(samples) 
          ...         most = max(samples) 
          ...         print(series_name, variable_name, least, most) 
          I x 4.0 14.0 
          I y 4.26 10.84 
          II x 4.0 14.0 
          II y 3.1 9.26 
          III x 4.0 14.0 
          III y 5.39 12.74 
          IV x 8.0 19.0 
          IV y 5.25 12.5** 

    ```

这使用`get_series（）`和`data_iter（）`从给定系列的一个变量中选择样本值。内置的`max（）`和`min（）`函数提供了极值。

1.  要计算方差（和标准差），我们也可以使用`statistics`模块：

```py
     **>>> import statistics 
          >>> for series_name in 'I', 'II', 'III', 'IV': 
          ...     series = get_series(data, series_name) 
          ...     for variable_name in 'x', 'y': 
          ...         samples = list(data_iter(series, variable_name)) 
          ...         mean = statistics.mean(samples) 
          ...         variance = statistics.variance(samples, mean) 
          ...         stdev = statistics.stdev(samples, mean) 
          ...         print(series_name, variable_name, 
          ...            round(variance,2), round(stdev,2)) 
          I x 11.0 3.32 
          I y 4.13 2.03 
          II x 11.0 3.32 
          II y 4.13 2.03 
          III x 11.0 3.32 
          III y 4.12 2.03 
          IV x 11.0 3.32 
          IV y 4.12 2.03** 

    ```

这使用`get_series（）`和`data_iter（）`从给定系列的一个变量中选择样本值。统计模块提供了计算感兴趣的统计量的`variance（）`和`stdev（）`函数。

## 它是如何工作的...

这些函数通常是Python标准库的一等部分。我们已经在三个地方寻找了有用的函数：

+   `min（）`和`max（）`函数是内置的。

+   `collections`模块有`Counter`类，可以创建频率直方图。我们可以从中获取众数。

+   `statistics`模块有`mean()`、`median()`、`mode()`、`variance()`和`stdev()`，提供各种统计量。

请注意，`data_iter()`是一个生成器函数。我们只能使用这个生成器的结果一次。如果我们只想计算单个统计摘要值，那将非常有效。

当我们想要计算多个值时，我们需要将生成器的结果捕获到一个集合对象中。在这些示例中，我们使用`data_iter()`来构建一个`list`对象，以便我们可以多次处理它。

## 还有更多...

我们的原始数据结构`data`是一系列可变字典。每个字典有两个键——`series`和`data`。我们可以用统计摘要更新这个字典。生成的对象可以保存以供以后分析或显示。

这是这种处理的起点：

```py
    def set_mean(data): 
        for series in data: 
            for variable_name in 'x', 'y': 
                samples = data_iter(series, variable_name) 
                series['mean_'+variable_name] = statistics.mean(samples) 

```

对于每个数据系列，我们使用`data_iter()`函数提取单独的样本。我们对这些样本应用`mean()`函数。结果保存回`series`对象，使用由函数名称`mean`、`_`字符和`variable_name`组成的字符串键。

请注意，这个函数的大部分内容都是样板代码。整体结构需要重复用于中位数、众数、最小值、最大值等。将函数从`mean()`更改为其他内容时，可以看到这个样板代码中有两个变化的地方：

+   用于更新系列数据的键

+   对所选样本序列进行评估的函数

我们不需要提供函数的名称；我们可以从函数对象中提取名称，如下所示：

```py
 **>>> statistics.mean.__name__ 
'mean'** 

```

这意味着我们可以编写一个高阶函数，将一系列函数应用到一组样本中：

```py
    def set_summary(data, function): 
      for series in data: 
        for variable_name in 'x', 'y': 
          samples = data_iter(series, variable_name) 
          series[function.__name__+'_'+variable_name] = function(samples) 

```

我们用参数名`function`替换了特定的函数`mean()`，这个参数名可以绑定到任何Python函数。处理将应用给定的函数到`data_iter()`的结果。然后使用这个摘要来更新系列字典，使用函数的名称、`_`字符和`variable_name`。

这个更高级的`set_summary()`函数看起来像这样：

```py
    for function in statistics.mean, statistics.median, min, max: 
        set_summary(data, function) 

```

这将基于`mean()`、`median()`、`max()`和`min()`更新我们的文档。我们可以使用任何Python函数，因此除了之前显示的函数之外，还可以使用`sum()`等函数。

因为`statistics.mode()`对于没有单一众数值的情况会引发异常，所以这个函数可能需要一个`try:`块来捕获异常，并将一些有用的结果放入`series`对象中。也可能适当地允许异常传播，以通知协作函数数据是可疑的。

我们修改后的文档将如下所示：

```py
    [ 
      { 
        "series": "I", 
        "data": [ 
          { 
            "x": 10.0, 
            "y": 8.04 
          }, 
          { 
            "x": 8.0, 
            "y": 6.95 
          }, 
          ... 
        ], 
        "mean_x": 9.0, 
        "mean_y": 7.500909090909091, 
        "median_x": 9.0, 
        "median_y": 7.58, 
        "min_x": 4.0, 
        "min_y": 4.26, 
        "max_x": 14.0, 
        "max_y": 10.84 
      }, 
      ... 
    ] 

```

我们可以将其保存到文件中，并用于进一步分析。使用`pathlib`处理文件名，我们可以做如下操作：

```py
    target_path = source_path.parent / (source_path.stem+'_stats.json') 
    target_path.write_text(json.dumps(data, indent=2)) 

```

这将创建一个与源文件相邻的第二个文件。名称将与源文件具有相同的词干，但词干将扩展为字符串`_stats`和后缀`.json`。

# 计数器中值的平均值

`statistics`模块有许多有用的函数。这些函数是基于每个单独的数据样本可用于处理。然而，在某些情况下，数据已经被分组到箱中。我们可能有一个`collections.Counter`对象，而不是一个简单的列表。现在我们不是值，而是（值，频率）对。

如何对（值，频率）对进行统计处理？

## 准备就绪

均值的一般定义是所有值的总和除以值的数量。通常写成这样：

![准备就绪](Image00027.jpg)

我们已经将一些数据集*C*定义为一系列单独的值，*C* = {*c*[0], *c*[1], *c*[2], ... ,c[n]},等等。这个集合的平均值，μ[*C*]，是值的总和除以值的数量*n*。

有一个微小的变化有助于概括这个定义：

![准备就绪](Image00028.jpg)![准备就绪](Image00029.jpg)

*S*（*C*）的值是值的总和。 *n*（*C*）的值是使用每个值的代替的总和。 实际上，*S*（*C*）是*c*[*i*]¹的总和，*n*（*C*）是*c*[*i*]⁰的总和。 我们可以很容易地将这些实现为简单的Python生成器表达式。

我们可以在许多地方重用这些定义。 具体来说，我们现在可以这样定义均值，μ [*C*]：

μ [*C*] = *S*（*C*）/ *n*（*C*）

我们将使用这个一般想法对已经收集到箱中的数据进行统计计算。 当我们有一个`Counter`对象时，我们有值和频率。 数据结构可以描述如下：

*F* = { *c* [0] : *f* [0] , *c* [1] : *f* [1] , *c* [2] : *f* [2] , ... *c[m]* : *f[m]* }

值*c[i]*与频率*f[i]*配对。 这对执行类似的计算进行了两个小的更改：

![准备就绪](Image00032.jpg)![准备就绪](Image00033.jpg)

我们已经定义了![准备就绪](Image00030.jpg)使用频率和值的乘积。 类似地，我们已经定义了![准备就绪](Image00031.jpg)使用频率。 我们在每个名称上都包含了帽子^，以清楚地表明这些函数不适用于简单值列表； 这些函数适用于（值，频率）对的列表。

这些需要在Python中实现。 例如，我们将使用以下`Counter`对象：

```py
 **>>> from collections import Counter 
>>> raw_data = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8] 
>>> series_4_x = Counter(raw_data)** 

```

这些数据来自*使用内置统计库*配方。 `Counter`对象如下所示：

```py
 **>>> series_4_x 
Counter({8: 10, 19: 1})** 

```

这显示了样本集中的各个值以及每个不同值的频率。

## 如何做...

1.  定义`Counter`的总和：

```py
     **>>> def counter_sum(counter): 
          ...     return sum(f*c for c,f in counter.items())** 

    ```

我们可以这样使用：

```py
     **>>> counter_sum(series_4_x) 
          99** 

    ```

1.  定义`Counter`中值的总数：

```py
     **>>> def counter_len(counter):** 

     **...     return sum(f for c,f in counter.items())** 

    ```

我们可以这样使用：

```py
     **>>> counter_len(series_4_x) 
          11** 

    ```

1.  我们现在可以将这些组合起来计算已经放入箱中的数据的均值：

```py
     **>>> def counter_mean(counter): 
          ...    return counter_sum(counter)/counter_len(counter) 
          >>> counter_mean(series_4_x) 
          9.0** 

    ```

## 它是如何工作的...

`Counter`是一个字典。 此字典的键是实际计数的值。 字典中的值是每个项目的频率。 这意味着`items()`方法将生成可以被我们的计算使用的值和频率信息。

我们已经将每个定义![它是如何工作...](Image00030.jpg)和![它是如何工作...](Image00031.jpg)转换为生成器表达式。 因为Python被设计为紧密遵循数学形式主义，所以代码以相对直接的方式遵循数学。

## 还有更多...

要计算方差（和标准偏差），我们需要对这个主题进行两个更多的变化。 我们可以定义频率分布的总体均值，μ [*F*]：

![更多内容...](Image00034.jpg)

其中*c[i]*是`Counter`对象*F*的键，*f[i]*是`Counter`对象给定键的频率值。

方差，VAR [*F*]，可以以依赖于均值，μ [*F*]的方式定义。 公式如下：

![更多内容...](Image00035.jpg)

这计算了值*c*[*i*]与均值μ[*F*]之间的差异。 这是由该值出现的次数*f[i]*加权的。 这些加权差的总和除以计数，![更多内容...](Image00031.jpg)，减去一。

标准偏差，σ [*F*]，是方差的平方根：

σ [*F*] = √VAR [*F*]

这个标准偏差的版本在数学上非常稳定，因此更受青睐。 它需要对数据进行两次传递，但对于一些边缘情况，进行多次传递的成本要好于错误的结果。

计算的另一个变化不依赖于均值，μ [*F*]。 这不像以前的版本那样在数学上稳定。 这种变化分别计算值的平方和，值的总和以及值的计数：

![更多内容...](Image00036.jpg)![更多内容...](Image00037.jpg)

这需要额外的一次求和计算。我们需要计算值的平方和，![更多内容...](Image00038.jpg)：

```py
 **>>> def counter_sum_2(counter): 
...     return sum(f*c**2 for c,f in counter.items())** 

```

鉴于这三个求和函数，![更多内容...](Image00031.jpg) ，![更多内容...](Image00030.jpg) ，和![更多内容...](Image00039.jpg) ，我们可以定义分箱摘要的方差，*F*：

```py
 **>>> def counter_variance(counter): 
...    n = counter_len(counter) 
...    return (counter_sum_2(counter)-(counter_sum(counter)**2)/n)/(n-1)** 

```

`counter_variance()`函数非常接近数学定义。Python版本将1/( *n* - 1)项作为次要优化移动。

使用`counter_variance()`函数，我们可以计算标准差：

```py
 **>>> import math 
>>> def counter_stdev(counter): 
...    return math.sqrt(counter_variance(counter))** 

```

这使我们能够看到以下内容：

```py
 **>>> counter_variance(series_4_x) 
11.0 
>>> round(counter_stdev(series_4_x), 2) 
3.32** 

```

我们还可以利用`Counter`对象的`elements()`方法。虽然简单，但这将创建一个潜在的大型中间数据结构：

```py
 **>>> import statistics 
>>> statistics.variance(series_4_x.elements()) 
11.0** 

```

我们已经使用`Counter`对象的`elements()`方法创建了计数器中所有元素的扩展列表。我们可以计算这些元素的统计摘要。对于一个大的`Counter`，这可能会成为一个非常大的中间数据结构。

## 另请参阅

+   在[第6章](text00070.html#page "第6章. 类和对象的基础")的*设计具有大量处理的类*配方中，我们从略微不同的角度看待了这个问题。在那个配方中，我们的目标只是隐藏一个复杂的数据结构。

+   本章中的*一次性分析多个变量*配方将解决一些效率方面的考虑。在该配方中，我们将探讨通过数据元素的单次遍历来计算多个求和的方法。

# 计算相关系数

在*使用内置统计库*和*计数器中的值的平均值*配方中，我们探讨了总结数据的方法。这些配方展示了如何计算中心值，以及方差和极值。

另一个常见的统计摘要涉及两组数据之间的相关程度。这不是Python标准库直接支持的。

相关性的一个常用度量标准称为**皮尔逊相关系数**。*r* -值是-1到+1之间的数字，表示数据值之间相关的概率。

零值表示数据是随机的。*0.95*的值表示95%的值相关，5%的值不相关。*-.95*的值表示95%的值具有反向相关性：一个变量增加时，另一个变量减少。

我们如何确定两组数据是否相关？

## 准备工作

皮尔逊*r*的一个表达式是这样的：

![准备工作](Image00040.jpg)

这依赖于数据集各个部分的大量单独求和。每个∑ *z* 运算符都可以通过Python的`sum()`函数实现。

我们将使用*使用内置统计库*配方中的数据。我们可以用以下方法读取这些数据：

```py
 **>>> from pathlib import Path 
>>> import json 
>>> from collections import OrderedDict 
>>> source_path = Path('code/anscombe.json') 
>>> data = json.loads(source_path.read_text(), 
...     object_pairs_hook=OrderedDict)** 

```

我们已经定义了数据文件的`Path`。然后我们可以使用`Path`对象从该文件中读取文本。这个文本被`json.loads()`用来从JSON数据构建Python对象。

我们已经包含了一个`object_pairs_hook`，这样这个函数将使用`OrderedDict`类构建JSON，而不是默认的`dict`类。这将保留源文档中项目的原始顺序。

我们可以这样检查数据：

```py
 **>>> [item['series'] for item in data] 
['I', 'II', 'III', 'IV'] 
>>> [len(item['data']) for item in data] 
[11, 11, 11, 11]** 

```

整个JSON文档是一个具有`I`等键的子文档序列。每个子文档有两个字段—`series`和`data`。在`data`值中有一个我们想要描述的观察值列表。每个观察值都有一对值。

数据看起来是这样的：

```py
    [ 
      { 
        "series": "I", 
        "data": [ 
          { 
            "x": 10.0, 
            "y": 8.04 
          }, 
          { 
            "x": 8.0, 
            "y": 6.95 
          }, 
          ... 
        ] 
      }, 
      ... 
    ] 

```

这组数据有四个系列，每个系列都表示为字典列表结构。在每个系列中，各个项都是具有`x`和`y`键的字典。

## 如何做...

1.  识别所需的各种求和。对于这个表达式，我们看到以下内容：

+   ∑ *x[i] , y[i]*

+   ∑ *x[i]*

+   ∑ *y[i]*

+   ∑ *x[i]* ²

+   ∑ *y[i]* ²

+   ![How to do it...](Image00041.jpg)

计数*n*可以真正定义为源数据集中每个数据的总和。这也可以被认为是*x[i]* ^∘或*y[i]* ^∘。

1.  从`math`模块导入`sqrt()`函数：

```py
            from math import sqrt 

    ```

1.  定义一个包装计算的函数：

```py
            def correlation(data): 

    ```

1.  使用内置的`sum()`函数编写各种总和。这是在函数定义内缩进的。我们将使用`data`参数的值：给定系列的一系列值。输入数据必须有两个键，`x`和`y`：

```py
            sumxy = sum(i['x']*i['y'] for i in data) 
            sumx = sum(i['x'] for i in data) 
            sumy = sum(i['y'] for i in data) 
            sumx2 = sum(i['x']**2 for i in data) 
            sumy2 = sum(i['y']**2 for i in data) 
            n = sum(1 for i in data) 

    ```

1.  根据各种总和的最终计算*r*。确保缩进正确匹配。有关更多帮助，请参阅[第3章](text00039.html#page "第3章。函数定义")，*函数定义*：

```py
            r = ( 
                (n*sumxy - sumx*sumy) 
                / (sqrt(n*sumx2-sumx**2)*sqrt(n*sumy2-sumy**2)) 
                ) 
            return r 

    ```

我们现在可以使用这个来确定各个系列之间的相关程度：

```py
    for series in data: 
        r = correlation(series['data']) 
        print(series['series'], 'r=', round(r, 2)) 

```

输出如下所示：

```py
    I r= 0.82
    II r= 0.82
    III r= 0.82
    IV r= 0.82

```

所有四个系列的相关系数大致相同。这并不意味着这些系列彼此相关。这意味着在每个系列中，82%的*x*值可以预测*y*值。这几乎正好是每个系列中的11个值中的9个。

## 它是如何工作的...

总体公式看起来相当复杂。但是，它可以分解为许多单独的总和和结合这些总和的最终计算。每个总和操作都可以用Python非常简洁地表示。

传统上，数学表示法可能如下所示：

![How it works...](Image00042.jpg)

这在Python中以非常直接的方式进行翻译：

```py
    sum(item['x'] for item in data) 

```

最终的相关系数可以简化一些。当我们用稍微更Pythonic的*S*（*x*）替换更复杂的![How it works...](Image00042.jpg)时，我们可以更好地看到方程的整体形式：

![How it works...](Image00043.jpg)

虽然简单，但所示的实现并不是最佳的。它需要对数据进行六次单独的处理，以计算各种缩减。作为一种概念验证，这种实现效果很好。这种实现的优势在于证明了编程的可行性。它还可以作为创建单元测试和重构算法以优化处理的起点。

## 还有更多...

该算法虽然清晰，但效率低下。更有效的版本将一次处理数据。为此，我们将不得不编写一个明确的`for`语句，通过数据进行一次遍历。在`for`语句的主体内，计算各种总和。

优化的算法看起来像这样：

```py
    sumx = sumy = sumxy = sumx2 = sumy2 = n = 0 
    for item in data: 
        x, y = item['x'], item['y'] 
        n += 1 
        sumx += x 
        sumy += y 
        sumxy += x * y 
        sumx2 += x**2 
        sumy2 += y**2 

```

我们已经将许多结果初始化为零，然后从数据项的源`data`中累积值到这些结果中。由于这只使用了数据值一次，所以这将适用于任何可迭代的数据源。

从这些总和计算*r*的算法不会改变。

重要的是初始版本的算法和已经优化为一次性计算所有摘要的修订版本之间的并行结构。两个版本的明显对称性有助于验证两件事：

+   初始实现与相当复杂的公式匹配

+   优化后的实现与初始实现和复杂的公式匹配

这种对称性结合适当的测试用例，可以确保实现是正确的。

# 计算回归参数

一旦我们确定了两个变量之间存在某种关系，下一步就是确定一种估计因变量的方法，以便从自变量的值中估计。对于大多数现实世界的数据，会有许多小因素导致围绕中心趋势的随机变化。我们将估计一种最小化这些误差的关系。

在最简单的情况下，变量之间的关系是线性的。当我们绘制数据点时，它们会倾向于聚集在一条直线周围。在其他情况下，我们可以通过计算对数或将变量提高到幂来调整其中一个变量，从而创建一个线性模型。在更极端的情况下，需要多项式。

我们如何计算两个变量之间的线性回归参数？

## 准备工作

估计线的方程式是这样的：

![准备就绪](Image00044.jpg)

给定自变量*x*，依赖变量*y*的估计或预测值![准备就绪](Image00045.jpg)是通过α和β参数计算得到的。

目标是找到α和β的值，使得估计值![准备就绪](Image00045.jpg)和*y*的实际值之间的总体误差最小。这里是β的计算：

β = *r[xy]* (σ [*x*] /σ *[y]* )

其中*r[xy]*是相关系数。参见*计算相关系数*配方。σ [*x*]的定义是*x*的标准偏差。这个值可以直接通过`statistics`模块得到。

这里是α的计算：

α = μ [*y*] - βμ [*x*]

其中μ [*x*]是*x*的均值。这也可以直接通过`statistics`模块得到。

我们将使用*使用内置统计库*配方中的数据。我们可以用以下方法读取这些数据：

```py
 **>>> from pathlib import Path 
>>> import json 
>>> from collections import OrderedDict 
>>> source_path = Path('code/anscombe.json') 
>>> data = json.loads(source_path.read_text(), 
...     object_pairs_hook=OrderedDict)** 

```

我们已经定义了数据文件的`Path`。然后我们可以使用`Path`对象从这个文件中读取文本。这个文本被`json.loads()`用来从JSON数据构建一个Python对象。

我们已经包含了一个`object_pairs_hook`，这样这个函数将使用`OrderedDict`类来构建JSON，而不是默认的`dict`类。这将保留源文档中项目的原始顺序。

我们可以像下面这样检查数据：

```py
 **>>> [item['series'] for item in data] 
['I', 'II', 'III', 'IV'] 
>>> [len(item['data']) for item in data] 
[11, 11, 11, 11]** 

```

整个JSON文档是一个具有诸如`I`之类的键的子文档序列。每个子文档有两个字段：`series`和`data`。在`data`值内部有一个我们想要描述的观察值列表。每个观察值都有一对值。

数据看起来是这样的：

```py
    [ 
      { 
        "series": "I", 
        "data": [ 
          { 
            "x": 10.0, 
            "y": 8.04 
          }, 
          { 
            "x": 8.0, 
            "y": 6.95 
          }, 
          ... 
        ] 
      }, 
      ... 
    ] 

```

这组数据有四个系列，每个系列都表示为一个字典结构的列表。在每个系列中，各个项目都是一个带有`x`和`y`键的字典。

## 如何做...

1.  导入`correlation()`函数和`statistics`模块：

```py
        from ch10_r03 import correlation 
        import statistics 

```

1.  定义一个将产生回归模型的函数，`regression()`：

```py
            def regression(data): 

    ```

1.  计算所需的各种值：

```py
            m_x = statistics.mean(i['x'] for i in data) 
            m_y = statistics.mean(i['y'] for i in data) 
            s_x = statistics.stdev(i['x'] for i in data) 
            s_y = statistics.stdev(i['y'] for i in data) 
            r_xy = correlation(data) 

    ```

1.  计算β和α的值：

```py
            b = r_xy * s_y/s_x 
            a = m_y - b * m_x 
            return a, b 

    ```

我们可以使用这个`regression()`函数来计算回归参数，如下所示：

```py
    for series in data: 
        a, b = regression(series['data']) 
        print(series['series'], 'y=', round(a, 2), '+', round(b,2), '*x') 

```

输出显示了一个预测给定`x`值的期望`y`的公式。输出如下：

```py
    I y= 3.0 + 0.5 *x
    II y= 3.0 + 0.5 *x
    III y= 3.0 + 0.5 *x
    IV y= 3.0 + 0.5 *x

```

在所有情况下，方程式都是![如何做...](Image00046.jpg)。这个估计似乎是实际*y*值的一个相当好的预测器。

## 它是如何工作的...

α和β的两个目标公式并不复杂。β的公式分解为使用两个标准偏差的相关值。α的公式使用β值和两个均值。这些都是以前配方的一部分。相关性计算包含了实际的复杂性。

核心设计技术是使用尽可能多的现有特征构建新特征。这样可以使测试用例分布到基础算法上，从而广泛使用（和测试）基础算法。

*计算相关系数*的性能分析很重要，在这里也适用。这个过程对数据进行了五次单独的遍历，以获得相关性以及各种均值和标准偏差。

作为概念验证，这个实现证明了算法是有效的。它也作为创建单元测试的起点。有了一个有效的算法，对代码进行重构以优化处理是有意义的。

## 还有更多...

之前显示的算法虽然清晰，但效率低下。为了处理数据一次，我们将不得不编写一个明确的`for`语句，通过数据进行一次遍历。在`for`语句的主体内，我们需要计算各种和。我们还需要计算一些从总和中派生的值，包括平均值和标准差：

```py
    sumx = sumy = sumxy = sumx2 = sumy2 = n = 0 
    for item in data: 
        x, y = item['x'], item['y'] 
        n += 1 
        sumx += x 
        sumy += y 
        sumxy += x * y 
        sumx2 += x**2 
        sumy2 += y**2 
    m_x = sumx / n 
    m_y = sumy / n 
    s_x = sqrt((n*sumx2 - sumx**2)/(n*(n-1))) 
    s_y = sqrt((n*sumy2 - sumy**2)/(n*(n-1))) 
    r_xy = (n*sumxy - sumx*sumy) / (sqrt(n*sumx2-sumx**2)*sqrt(n*sumy2-sumy**2)) 
    b = r_xy * s_y/s_x 
    a = m_y - b * m_x 

```

我们已将一些结果初始化为零，然后从数据项源`data`中累积值到这些结果中。由于这只使用了数据值一次，因此这将适用于任何可迭代的数据源。

从这些总和中计算`r_xy`的计算与之前的示例没有变化。`α`或`β`值的计算也没有变化，`a`和`b`。由于这些最终结果与以前版本相同，我们有信心这种优化将计算出相同的答案，但只需对数据进行一次遍历。

# 计算自相关

在许多情况下，事件会以重复的周期发生。如果数据与自身相关，这被称为自相关。对于一些数据，间隔可能很明显，因为有一些可见的外部影响，比如季节或潮汐。对于一些数据，间隔可能很难辨别。

在*计算相关系数*配方中，我们看了一种测量两组数据之间相关性的方法。

如果我们怀疑我们有循环数据，我们能否利用以前的相关函数来计算自相关？

## 准备工作

自相关的核心概念是通过时间偏移T进行相关性的想法。这种测量有时被表达为*r[xx]*（T）：*x*和具有时间偏移T的*x*之间的相关性。

假设我们有一个方便的相关函数，*R*（*x*，*y*）。它比较两个序列，[*x*[0]，*x*[1]，*x*[2]，...]和[*y*[0]，*y*[1]，*y*[2]，...]，并返回两个序列之间的相关系数：

*r[xy]* = *R*（[*x*[0]，*x*[1]，*x*[2]，...]，[*y*[0]，*y*[1]，*y*[2]，...]）

我们可以通过使用索引值作为时间偏移来将其应用于自相关：

*r[xx]*（T）= *R*（[*x*[0]，*x*[1]，*x*[2]，...]，[*x*[0+T]，*x*[1+T]，*x*[2+T]，...]）

我们已经计算了相互偏移T的*x*值之间的相关性。如果T = 0，我们将每个项目与自身进行比较，相关性为*r[xx]*（0）= 1。

我们将使用一些我们怀疑其中有季节信号的数据。这是来自[http://www.esrl.noaa.gov/gmd/ccgg/trends/](http://www.esrl.noaa.gov/gmd/ccgg/trends/)的数据。我们可以访问[ftp://ftp.cmdl.noaa.gov/ccg/co2/trends/co2_mm_mlo.txt](ftp://ftp.cmdl.noaa.gov/ccg/co2/trends/co2_mm_mlo.txt)来下载原始数据文件。

文件有一些以`#`开头的前言行。这些必须从数据中过滤掉。我们将使用[第8章](text00088.html#page "第8章。功能和响应式编程特性")中的 *Picking a subset – three ways to filter* 配方，*Functional and Reactive Programming Features*，它将删除无用的行。

剩下的行有七列，值之间以空格分隔。我们将使用[第9章](text00099.html#page "第9章。输入/输出、物理格式和逻辑布局")中的 *Reading delimited files with the CSV module* 配方，*Input/Output, Physical Format, and Logical Layout*，来读取CSV数据。在这种情况下，CSV中的逗号将是一个空格字符。结果将有点尴尬，因此我们将使用[第9章](text00099.html#page "第9章。输入/输出、物理格式和逻辑布局")中的 *Upgrading CSV from Dictreader to namespace reader* 配方，*Input/Output, Physical Format, and Logical Layout*，创建一个更有用的命名空间，其中值已经正确转换。在该配方中，我们导入了`CSV`模块：

```py
    import csv 

```

以下是处理文件物理格式基本方面的两个函数。第一个是一个过滤器，用于拒绝注释行；或者，从另一个角度来看，传递非注释行：

```py
    def non_comment_iter(source): 
        for line in source: 
            if line[0] == '#': 
                continue 
            yield line 

```

`non_comment_iter()`函数将遍历给定的源并拒绝以`#`开头的行。所有其他行将原样传递。

`non_comment_iter()`函数可用于构建处理有效数据行的CSV读取器。读取器需要一些额外的配置来定义数据列和涉及的CSV方言的细节：

```py
    def raw_data_iter(source): 
        header = ['year', 'month', 'decimal_date', 'average', 
                  'interpolated', 'trend', 'days'] 
        rdr = csv.DictReader(source, 
            header, delimiter=' ', skipinitialspace=True) 
        return rdr 

```

`raw_data_iter()`函数定义了七个列标题。它还指定列分隔符是空格，并且可以跳过数据每列前面的额外空格。该函数的输入必须去除注释行，通常是通过使用`non_comment_iter()`等过滤函数。

该函数的结果是以字典形式的数据行，具有七个键。这些行看起来像这样：

```py
    [{'average': '315.71', 'days': '-1', 'year': '1958', 'trend': '314.62',
        'decimal_date': '1958.208', 'interpolated': '315.71', 'month': '3'},
     {'average': '317.45', 'days': '-1', 'year': '1958', 'trend': '315.29',
        'decimal_date': '1958.292', 'interpolated': '317.45', 'month': '4'},
    etc.

```

由于所有的值都是字符串，因此需要进行一次清洗和转换。这是一个可以在生成器表达式中使用的行清洗函数。这将构建一个`SimpleNamespace`对象，因此我们需要导入该定义：

```py
    from types import SimpleNamespace 
    def cleanse(row): 
        return SimpleNamespace( 
            year= int(row['year']), 
            month= int(row['month']), 
            decimal_date= float(row['decimal_date']), 
            average= float(row['average']), 
            interpolated= float(row['interpolated']), 
            trend= float(row['trend']), 
            days= int(row['days']) 
        ) 

```

该函数将通过将转换函数应用于字典中的值，将每个字典行转换为`SimpleNamespace`。大多数项目都是浮点数，因此使用`float()`函数。其中一些项目是整数，对于这些项目使用`int()`函数。

我们可以编写以下类型的生成器表达式，将此清洗函数应用于原始数据的每一行：

```py
    cleansed_data = (cleanse(row) for row in raw_data) 

```

这将对数据的每一行应用`cleanse()`函数。通常，预期是行来自`raw_data_iter()`。

对每一行应用`cleanse()`函数将创建如下所示的数据：

```py
    [namespace(average=315.71, days=-1, decimal_date=1958.208, 
        interpolated=315.71, month=3, trend=314.62, year=1958), 
     namespace(average=317.45, days=-1, decimal_date=1958.292, 
        interpolated=317.45, month=4, trend=315.29, year=1958), 
    etc. 

```

这些数据非常容易处理。可以通过简单的名称识别各个字段，并且数据值已转换为Python内部数据结构。

这些函数可以组合成一个堆栈，如下所示：

```py
    def get_data(source_file): 
        non_comment_data = non_comment_iter(source_file) 
        raw_data = raw_data_iter(non_comment_data) 
        cleansed_data = (cleanse(row) for row in raw_data) 
        return cleansed_data 

```

`get_data()`生成器函数是一组生成器函数和生成器表达式。它返回一个迭代器，该迭代器将产生源数据的单独行。`non_comment_iter()`函数将读取足够的行以便产生单个非注释行。`raw_data_iter()`函数将解析CSV的一行并产生一个包含单行数据的字典。

`cleansed_data`生成器表达式将对原始数据的每个字典应用`cleanse()`函数。单独的行是方便的`SimpleNamespace`数据结构，可以在其他地方使用。

该生成器将所有单独的步骤绑定到一个转换管道中。当需要更改步骤时，这将成为更改的焦点。我们可以在这里添加过滤器，或者替换解析或清洗函数。

使用`get_data()`函数的上下文将如下所示：

```py
    source_path = Path('co2_mm_mlo.txt') 
    with source_path.open() as source_file: 
        for row in get_data(source_file): 
            print(row.year, row.month, row.average) 

```

我们需要打开一个源文件。我们可以将文件提供给`get_data()`函数。该函数将以易于用于统计处理的形式发出每一行。

## 如何做...

1.  从`ch10_r03`模块导入`correlation()`函数：

```py
            from ch10_r03 import correlation 

    ```

1.  从源数据中获取相关的时间序列数据项：

```py
            co2_ppm = list(row.interpolated 
                for row in get_data(source_file)) 

    ```

在这种情况下，我们将使用插值数据。如果我们尝试使用平均数据，将会有报告间隙，这将迫使我们找到没有间隙的时期。插值数据有值来填补这些间隙。

我们已经从生成器表达式创建了一个`list`对象，因为我们将对其进行多个摘要操作。

1.  对于多个时间偏移T，计算相关性。我们将使用从`1`到`20`期的时间偏移。由于数据是每月收集的，我们怀疑T = 12将具有最高的相关性：

```py
            for tau in range(1,20): 
                data = [{'x':x, 'y':y} 
                    for x,y in zip(co2_ppm[:-tau], co2_ppm[tau:])] 
                r_tau_0 = correlation(data[:60]) 
                print(tau, r_tau_0) 

    ```

*计算相关系数*配方中的`correlation()`函数需要一个具有两个键的小字典：`x`和`y`。第一步是构建这些字典的数组。我们使用`zip()`函数来组合两个数据序列：

+   `co2_ppm[:-tau]`

+   `co2_ppm[tau:]`

`zip()`函数将从`data`的每个切片中组合值。第一个切片从开头开始。第二个从序列的`tau`位置开始。通常，第二个序列会更短，`zip()`函数在序列耗尽时停止处理。

我们使用`co2_ppm[:-tau]`作为`zip()`函数的一个参数值，以清楚地表明我们跳过了序列末尾的一些项目。我们跳过的项目数量与从第二个序列的开头省略的项目数量相同。

我们只取了前60个值来计算具有不同时间偏移值的自相关性。数据是按月提供的。我们可以看到非常强烈的年度相关性。我们已经突出显示了输出的这一行：

```py
    r_{xx}(τ= 1) =  0.862
    r_{xx}(τ= 2) =  0.558
    r_{xx}(τ= 3) =  0.215
    r_{xx}(τ= 4) = -0.057
    r_{xx}(τ= 5) = -0.235
    r_{xx}(τ= 6) = -0.319
    r_{xx}(τ= 7) = -0.305
    r_{xx}(τ= 8) = -0.157
    r_{xx}(τ= 9) =  0.141
    r_{xx}(τ=10) =  0.529
    r_{xx}(τ=11) =  0.857   
    **r_{xx}(τ=12) =  0.981** 

    r_{xx}(τ=13) =  0.847
    r_{xx}(τ=14) =  0.531
    r_{xx}(τ=15) =  0.179
    r_{xx}(τ=16) = -0.100
    r_{xx}(τ=17) = -0.279
    r_{xx}(τ=18) = -0.363
    r_{xx}(τ=19) = -0.349

```

当时间偏移为`12`时，*r[xx]*(12) = .981。几乎任何数据子集都可以获得类似引人注目的自相关性。这种高相关性证实了数据的年度周期。

整个数据集包含了近58年的将近700个样本。事实证明，季节变化信号在整个时间跨度上并不那么明显。这意味着有另一个更长的周期信号淹没了年度变化信号。

这种其他信号的存在表明正在发生更复杂的事情。这种效应的时间尺度长于五年。需要进一步分析。

## 它是如何工作的...

Python的一个优雅特性是数组切片概念。在[第4章](text00048.html#page "第4章。内置数据结构-列表、集合、字典")的*切片和切块列表*配方中，我们看了列表切片的基础知识。在进行自相关计算时，数组切片为我们提供了一个非常简单的工具，用于比较数据的两个子集。

算法的基本要素总结如下：

```py
    data = [{'x':x, 'y':y} 
        for x,y in zip(co2_ppm[:-tau], co2_ppm[tau:])] 

```

这些对是从`co2_ppm`序列的两个切片的`a zip()`构建的。这两个切片构建了用于创建临时对象`data`的预期(`x`,`y`)对。有了这个`data`对象，现有的`correlation()`函数计算了相关度量。

## 还有更多...

我们可以使用类似的数组切片技术在整个数据集中反复观察12个月的季节循环。在这个例子中，我们使用了这个：

```py
    r_tau_0 = correlation(data[:60]) 

```

前面的代码使用了可用的699个样本中的前60个。我们可以从各个地方开始切片，并使用不同大小的切片来确认周期在整个数据中都存在。

我们可以创建一个模型，展示12个月的数据是如何变化的。因为有一个重复的周期，正弦函数是最有可能的模型候选。我们将使用这个进行拟合：

![还有更多...](Image00047.jpg)

正弦函数本身的平均值为零，因此*K*因子是给定12个月周期的平均值。函数*f*(*x* - φ)将月数转换为在-2π ≤ *f*(*x* - φ) ≤ 2π范围内的适当值。例如，*f*(*x*) = 2π(( *x* -6)/12)可能是合适的。最后，缩放因子*A*将数据缩放以匹配给定月份的最小值和最大值。

### 长期模型

虽然有趣，这种分析并没有找到掩盖年度振荡的长期趋势。为了找到这种趋势，有必要将每个12个月的样本序列减少到一个单一的年度中心值。中位数或平均值对此都很有效。

我们可以使用以下生成器表达式创建一个月平均值序列：

```py
    from statistics import mean, median 
    monthly_mean = [ 
        {'x': x, 'y': mean(co2_ppm[x:x+12])}  
            for x in range(0,len(co2_ppm),12) 
    ] 

```

该生成器将构建一系列字典。每个字典都有回归函数使用的必需的`x`和`y`项。`x`值是一个简单的代表年份和月份的值：它是一个从零增长到696的数字。`y`值是12个月份值的平均值。

回归计算如下进行：

```py
    from ch10_r04 import regression 
    alpha, beta = regression(monthly_mean) 
    print('y=', alpha, '+x*', beta) 

```

这显示了一个明显的线，方程如下：

![长期模型](Image00048.jpg)

`x`值是与数据集中的第一个月（1958年3月）相偏移的月数。例如，1968年3月的`x`值为120。年均CO[2]浓度为*y*=323.1。该年的实际平均值为323.27。可以看出，这些值非常相似。

这个`相关`模型的*r*²值，显示了方程如何拟合数据，为0.98。这个上升的斜率是长期主导季节性波动的信号。

## 另请参阅

+   *计算相关系数*配方显示了计算一系列值之间相关性的核心函数

+   *计算回归参数*配方显示了确定详细回归参数的额外背景

# 确认数据是随机的-零假设

一个重要的统计问题被构建为关于数据集的零假设和备择假设。假设我们有两组数据，*S1*和*S2*。我们可以对数据形成两种假设：

+   **零假设**：任何差异都是次要的随机效应，没有显著差异。

+   备用：这些差异在统计上是显著的。一般来说，这种可能性小于5%。

我们如何评估数据，以查看它是否真正随机，还是存在一些有意义的变化？

## 准备工作

如果我们在统计学方面有很强的背景，我们可以利用统计理论来评估样本的标准差，并确定两个分布之间是否存在显著差异。如果我们在统计学方面薄弱，但在编程方面有很强的背景，我们可以进行一些编码，达到类似的结果而不需要理论。

我们可以通过各种方式比较数据集，以查看它们是否存在显著不同或差异是否是随机变化。在某些情况下，我们可能能够对现象进行详细的模拟。如果我们使用Python内置的随机数生成器，我们将得到与真正随机的现实世界事件基本相同的数据。我们可以将模拟与测量数据进行比较，以查看它们是否相同。

模拟技术只有在模拟相对完整时才有效。例如，赌场赌博中的离散事件很容易模拟。但是，网页交易中的某些离散事件，比如购物车中的商品，很难精确模拟。

在我们无法进行模拟的情况下，我们有许多可用的重采样技术。我们可以对数据进行洗牌，使用自助法，或者使用交叉验证。在这些情况下，我们将使用可用的数据来寻找随机效应。

我们将在*计算自相关*配方中比较数据的三个子集。这些是来自两个相邻年份和一个与其他两个年份相隔很远的第三年的数据值。每年有12个样本，我们可以轻松计算这些组的平均值：

```py
 **>>> from ch10_r05 import get_data 
>>> from pathlib import Path 
>>> source_path = Path('code/co2_mm_mlo.txt') 
>>> with source_path.open() as source_file: 
...     all_data = list(get_data(source_file)) 
>>> y1959 = [r.interpolated for r in all_data if r.year == 1959] 
>>> y1960 = [r.interpolated for r in all_data if r.year == 1960] 
>>> y2014 = [r.interpolated for r in all_data if r.year == 2014]** 

```

我们已经为三年的可用数据创建了三个子集。每个子集都是使用一个简单的筛选器创建的，该筛选器创建一个数值列表，其中年份与目标值匹配。我们可以按如下方式对这些子集进行统计：

```py
 **>>> from statistics import mean 
>>> round(mean(y1959), 2) 
315.97 
>>> round(mean(y1960), 2) 
316.91 
>>> round(mean(y2014), 2) 
398.61** 

```

这三个平均值是不同的。我们的假设是，`1959`和`1960`之间的差异只是普通的随机变化，没有显著性。然而，`1959`和`2014`之间的差异在统计上是显著的。

排列或洗牌技术的工作原理如下：

1.  对于汇总数据的每个排列：

1.  1959年数据和1960年数据之间的平均差异为*316.91-315.97=0.94*。我们可以称之为*T[obs]*，观察到的测试测量。

+   创建两个子集，*A*和*B*

+   计算平均值之间的差异，*T*

+   计算差异的数量，*T*，大于*T[obs]*和小于*T[obs]*的值

这两个计数告诉我们我们观察到的差异如何与所有可能的差异相比。对于大型数据集，可能存在大量的排列组合。在我们的情况下，我们知道24个样本中每次取12个的组合数由以下公式给出：

![准备就绪](Image00049.jpg)

我们可以计算*n*=24和*k*=12的值：

```py
 **>>> from ch03_r07 import fact_s 
>>> def binom(n, k): 
...     return fact_s(n)//(fact_s(k)*fact_s(n-k)) 
>>> binom(24, 12) 
2704156** 

```

有略多于2.7百万个排列。我们可以使用`itertools`模块中的函数来生成这些。`combinations()`函数将发出各种子集。处理需要超过5分钟（320秒）。

另一个计划是使用随机子集。使用270,156个随机样本大约需要35秒。使用仅10%的组合提供了足够准确的答案，以确定两个样本是否在统计上相似，并且零假设成立，或者两个样本是否不同。

## 如何做...

1.  我们将使用`random`和`statistics`模块。`shuffle()`函数是随机化样本的核心。我们还将使用`mean()`函数：

```py
            import random 
            from statistics import mean 

    ```

我们可以简单地计算样本之间观察到的差异以上和以下的值。相反，我们将创建一个`Counter`并在-0.001到+0.001的2,000个步骤中收集差异。这将提供一些信心，表明差异是正态分布的：

```py
            from collections import Counter 

    ```

1.  定义一个接受两组独立样本的函数。这些将被合并，并从集合中随机抽取子集：

```py
            def randomized(s1, s2, limit=270415): 

    ```

1.  计算平均值之间的观察到的差异，*T[obs]*：

```py
            T_obs = mean(s2)-mean(s1) 
            print( "T_obs = m_2-m_1 = {:.2f}-{:.2f} = {:.2f}".format( 
                mean(s2), mean(s1), T_obs) 
            ) 

    ```

1.  初始化一个`Counter`来收集详细信息：

```py
            counts = Counter() 

    ```

1.  创建样本的组合宇宙。我们可以连接这两个列表：

```py
            universe = s1+s2 

    ```

1.  使用`for`语句进行大量的重新采样；270,415个样本可能需要35秒。很容易扩展或收缩子集以平衡精度和计算速度的需求。大部分处理将嵌套在这个循环内：

```py
            for resample in range(limit): 

    ```

1.  洗牌数据：

```py
                random.shuffle(universe) 

    ```

1.  选择两个与原始数据大小匹配的子集：

```py
                a = universe[:len(s2)] 
                b = universe[len(s2):] 

    ```

由于Python列表索引的工作方式，我们可以确保两个列表完全分开宇宙中的值。由于第一个列表中不包括结束索引值`len(s2)`，这种切片清楚地分隔了所有项目。

1.  计算平均值之间的差异。在这种情况下，我们将通过`1000`进行缩放并转换为整数，以便我们可以累积频率分布：

```py
                delta = int(1000*(mean(a) - mean(b))) 
                counts[delta] += 1 

    ```

创建delta值的直方图的替代方法是计算大于*T[obs]*和小于*T[obs]*的值。使用完整的直方图提供了数据在统计上是正常的信心。

1.  在`for`循环之后，我们可以总结`counts`，显示有多少个差异大于观察到的差异，有多少个差异小于观察到的差异。如果任一值小于5%，这是一个统计学上显著的差异：

```py
            T = int(1000*T_obs) 
            below = sum(v for k,v in counts.items() if k < T) 
            above = sum(v for k,v in counts.items() if k >= T) 

            print( "below {:,} {:.1%}, above {:,} {:.1%}".format( 
                below, below/(below+above), 
                above, above/(below+above))) 

    ```

当我们对来自1959年和1960年的数据运行`randomized()`函数时，我们看到以下内容：

```py
    print("1959 v. 1960") 
    randomized(y1959, y1960) 

```

输出如下所示：

```py
    1959 v. 1960
    T_obs = m_2-m_1 = 316.91-315.97 = 0.93
    below 239,457 88.6%, above 30,958 11.4%

```

这表明11%的数据高于观察到的差异，88%的数据低于观察到的差异。这完全在正常统计噪音的范围内。

当我们对来自`1959`和`2014`的数据运行此操作时，我们看到以下输出：

```py
    1959 v. 2014
    T_obs = m_2-m_1 = 398.61-315.97 = 82.64
    below 270,414 100.0%, above 1 0.0%

```

涉及的数据只有270,415个样本中的一个示例高于平均值之间的观察到的差异，*T[obs]*。从1959年到2014年的变化在统计上是显著的，概率为3.7 x 10^(-6)。

## 工作原理...

计算所有270万个排列可以得到确切的答案。使用随机子集而不是计算所有可能的排列更快。Python随机数生成器非常出色，它确保随机子集将被公平分布。

我们使用了两种技术来计算数据的随机子集：

1.  用 `random.shuffle(u)` 对整个值域进行洗牌

1.  用类似 `a, b = u[x:], u[:x]` 的代码对值域进行分区

两个分区的均值是用 `statistics` 模块完成的。我们可以定义更有效的算法，通过数据的单次遍历来进行洗牌、分区和均值计算。这种更有效的算法将省略创建排列差异的完整直方图。

前面的算法将每个差异转换为-1000到+1000之间的值，使用如下：

```py
    delta = int(1000*(mean(a) - mean(b))) 

```

这使我们能够使用 `Counter` 计算频率分布。这将显示大多数差异实际上是零；这是对正态分布数据的预期。看到分布可以确保我们的随机数生成和洗牌算法中没有隐藏的偏差。

我们可以简单地计算上面和下面的值，而不是填充 `Counter` 。这种比较排列差异和观察差异 *T[obs]* 的最简单形式如下：

```py
    if mean(a) - mean(b) > T_obs: 
        above += 1 

```

这计算了大于观察差异的重采样差异的数量。从中，我们可以通过 `below = limit-above` 计算出低于观察值的数量。这将给我们一个简单的百分比值。

## 还有更多...

我们可以通过改变计算每个随机子集的均值的方式来进一步加快处理速度。

给定一个数字池 *P* ，我们创建两个不相交的子集 *A* 和 *B* ，使得：

*A* ∪ *B* = *P* ∧ *A* ∩ *B* = ∅

*A* 和 *B* 子集的并集覆盖了整个值域 *P* 。没有缺失值，因为 *A* 和 *B* 之间的交集是一个空集。

整体总和 *S[p]* 只需计算一次：

*S[P]* = ∑ *P*

我们只需要计算一个子集 *S[A]* 的总和：

*S[A] = ∑ A*

这意味着另一个子集的总和只是一个减法。我们不需要一个昂贵的过程来计算第二个总和。

集合的大小，*N[A]* 和 *N[B]* ，同样是恒定的。均值，μ [*A*] 和 μ [*B*] ，可以快速计算：

μ [*A*] = ( *S[A]* / *N[A]* )

μ [*B*] = ( *S[P]* - *S[A]* )/ *N[B]*

这导致了重采样循环的轻微变化：

```py
    a_size = len(s1) 
    b_size = len(s2) 
    s_u = sum(universe) 
    for resample in range(limit): 
        random.shuffle(universe) 
        a = universe[:len(s1)] 
        s_a = sum(a) 
        m_a = s_a/a_size 
        m_b = (s_u-s_a)/b_size 
        delta = int(1000*(m_a-m_b)) 
        counts[delta] += 1 

```

通过仅计算一个总和 `s_a` ，我们可以节省随机重采样过程的处理时间。我们不需要计算另一个子集的总和，因为我们可以将其计算为整个值域的总和之间的差异。然后我们可以避免使用 `mean()` 函数，并直接从总和和固定计数计算均值。

这种优化使得很容易迅速做出统计决策。使用重采样意味着我们不需要依赖于复杂的统计理论知识；我们可以重采样现有数据以表明给定样本符合零假设或超出预期，并需要提出一些替代假设。

## 另请参阅

+   这个过程可以应用于其他统计决策程序。这包括 *计算回归参数* 和 *计算自相关* 配方。

# 查找异常值

当我们有统计数据时，我们经常发现可以描述为异常值的数据点。异常值偏离其他样本，可能表明坏数据或新发现。异常值根据定义是罕见事件。

异常值可能是数据收集中的简单错误。它们可能代表软件错误，或者可能是测量设备未正确校准。也许日志条目无法读取是因为服务器崩溃或时间戳错误是因为用户错误输入了数据。

异常值也可能是有趣的，因为存在一些难以检测的其他信号。它可能是新颖的，或者罕见的，或者超出了我们设备的准确校准。在Web日志中，它可能暗示了应用程序的新用例，或者标志着新类型的黑客攻击的开始。

我们如何定位和标记潜在的异常值？

## 准备工作

定位异常值的一种简单方法是将值标准化以使它们成为Z分数。Z分数将测量值转换为测量值与均值之间的比率，以标准差为单位：

*Z[i]* = ( *x[i]* - μ [*x*] )/σ [*x*]

其中μ [*x*]是给定变量*x*的均值，σ [*x*]是该变量的标准差。我们可以使用`statistics`模块计算这些值。

然而，这可能有些误导，因为Z分数受涉及的样本数量限制。因此，*NIST工程和统计手册*，*1.3.5.17节*，建议使用以下规则来检测异常值：

准备工作

**MAD**（**中位数绝对偏差**）代替标准差。MAD是每个样本*x[i]*与总体中位数*x*之间的偏差的绝对值的中位数：

准备工作

使用缩放因子*0.6745*来缩放这些分数，以便可以将大于3.5的*M[i]*值识别为异常值。请注意，这与计算样本方差是平行的。方差测量使用均值，而这个测量使用中位数。值0.6745在文献中被广泛用作定位异常值的适当值。

我们将使用一些来自*使用内置统计库*配方的数据，其中包括一些相对平滑的数据集和一些具有严重异常值的数据集。数据位于一个JSON文档中，其中包含四个( *x* , *y* )对的系列。

我们可以使用以下方法读取这些数据：

```py
 **>>> from pathlib import Path 
>>> import json 
>>> from collections import OrderedDict 
>>> source_path = Path('code/anscombe.json') 
>>> data = json.loads(source_path.read_text(), 
...     object_pairs_hook=OrderedDict)** 

```

我们已经定义了数据文件的`Path`。然后，我们可以使用`Path`对象从该文件中读取文本。`json.loads()`使用这些文本从JSON数据构建Python对象。

我们已经包含了一个`object_pairs_hook`，以便该函数将使用`OrderedDict`类构建JSON，而不是默认的`dict`类。这将保留源文档中项目的原始顺序。

我们可以检查以下数据：

```py
 **>>> [item['series'] for item in data] 
['I', 'II', 'III', 'IV'] 
>>> [len(item['data']) for item in data] 
[11, 11, 11, 11]** 

```

整个JSON文档是具有诸如`I`和`II`等键的子文档序列。每个子文档有两个字段：`series`和`data`。`data`值是我们想要描述的观测值列表。每个观测值都是一对测量值。

## 如何做...

1.  导入`statistics`模块。我们将进行许多中位数计算。此外，我们可以使用`itertools`的一些功能，如`compress()`和`filterfalse()`。

```py
            import statistics 
            import itertools 

    ```

1.  定义`absdev()`映射。这将使用给定的中位数或计算样本的实际中位数。然后返回一个生成器，提供所有相对于中位数的绝对偏差：

```py
            def absdev(data, median=None): 
                if median is None: 
                    median = statistics.median(data) 
                return ( 
                    abs(x-median) for x in data 
                ) 

    ```

1.  定义`median_absdev()`缩减。这将定位绝对偏差值序列的中位数。这计算用于检测异常值的MAD值。这可以计算中位数，也可以给定已计算的中位数：

```py
            def median_absdev(data, median=None): 
                if median is None: 
                    median = statistics.median(data) 
                return statistics.median(absdev(data, median=median)) 

    ```

1.  定义修改后的Z分数映射，`z_mod()`。这将计算数据集的中位数，并使用它来计算MAD。然后使用偏差值来计算基于该偏差值的修改后的Z分数。返回的值是修改后的Z分数的迭代器。由于数据需要多次通过，输入不能是可迭代集合，因此必须是序列对象：

```py
            def z_mod(data): 
                median = statistics.median(data) 
                mad = median_absdev(data, median) 
                return ( 
                    0.6745*(x - median)/mad for x in data 
                ) 

    ```

在这个实现中，我们使用了一个常数`0.6745`。在某些情况下，我们可能希望将其作为参数。我们可以使用`def z_mod(data, threshold=0.6745)`来允许更改这个值。

有趣的是，MAD值为零的可能性。当大多数值不偏离中位数时，这种情况可能发生。当超过一半的点具有相同的值时，中位绝对偏差将为零。

1.  基于修改后的Z映射`z_mod()`定义异常值过滤器。任何值超过3.5都可以被标记为异常值。然后可以计算包括和不包括异常值的统计摘要。`itertools`模块有一个`compress()`函数，可以使用布尔选择器值的序列根据`z_mod()`计算的结果从原始数据序列中选择项目：

```py
            def pass_outliers(data): 
                return itertools.compress(data, (z >= 3.5 for z in z_mod(data))) 

            def reject_outliers(data): 
                return itertools.compress(data, (z < 3.5 for z in z_mod(data))) 

    ```

`pass_outliers()`函数仅传递异常值。`reject_outliers()`函数传递非异常值。通常，我们会显示两个结果——整个数据集和拒绝异常值的整个数据集。

大多数这些函数都多次引用输入数据参数，不能使用可迭代对象。这些函数必须给定一个`Sequence`对象。`list`或`tuple`是`Sequence`的例子。

我们可以使用`pass_outliers()`来定位异常值。这对于识别可疑的数据值很有用。我们可以使用`reject_outliers()`来提供已从考虑中移除异常值的数据。

## 工作原理...

转换堆栈可以总结如下：

1.  减少总体以计算总体中位数。

1.  将每个值映射到与总体中位数的绝对偏差。

1.  减少绝对偏差以创建中位绝对偏差MAD。

1.  将每个值映射到使用总体中位数和MAD的修改Z得分。

1.  根据修改后的Z得分过滤结果。

我们分别定义了这个堆栈中的每个转换函数。我们可以使用[第8章](text00088.html#page "第8章。功能和响应式编程特性")中的示例，*功能和响应式编程特性*，创建更小的函数，并使用内置的`map()`和`filter()`函数来实现这个过程。

我们不能轻松地使用内置的`reduce()`函数来定义中位数计算。为了计算中位数，我们必须使用递归中位数查找算法，将数据分成越来越小的子集，其中一个子集具有中位数值。

以下是我们如何将其应用于给定的样本数据：

```py
    for series_name in 'I', 'II', 'III', 'IV': 
        print(series_name) 
        series_data = [series['data'] 
            for series in data 
                if series['series'] == series_name][0] 

        for variable_name in 'x', 'y': 
            variable = [float(item[variable_name]) for item in series_data] 
            print(variable_name, variable, end=' ') 
            try: 
                print( "outliers", list(pass_outliers(variable))) 
            except ZeroDivisionError: 
                print( "Data Appears Linear") 
        print() 

```

我们遍历了源数据中的每个系列。`series_data`的计算从源数据中提取了一个系列。每个系列都有两个变量`x`和`y`。在样本集中，我们可以使用`pass_outliers()`函数来定位数据中的异常值。

`except`子句处理`ZeroDivisionError`异常。这个异常是由`z_mod()`函数对一组特别病态的数据引发的。以下是显示这些奇怪数据的输出行：

```py
    x [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0] Data Appears Linear

```

在这种情况下，至少一半的值是相同的。这个单一的多数值将被视为中位数。对于这个子集，与中位数的绝对偏差将为零。因此，MAD将为零。在这种情况下，异常值的概念是可疑的，因为数据似乎不反映普通的统计噪声。

这些数据不符合一般模型，必须对这个变量应用不同类型的分析。由于数据的特殊性，可能必须拒绝异常值的概念。

## 还有更多...

我们使用`itertools.compress()`来传递或拒绝异常值。我们还可以以类似的方式使用`filter()`和`itertools.filterfalse()`函数。我们将研究一些`compress()`的优化以及使用`filter()`代替`compress()`的方法。

我们使用了两个看起来相似的函数定义`pass_outliers`和`reject_outliers`。这种设计存在对关键程序逻辑的不必要重复，违反了DRY原则。以下是这两个函数：

```py
    def pass_outliers(data): 
        return itertools.compress(data, (z >= 3.5 for z in z_mod(data))) 

    def reject_outliers(data): 
        return itertools.compress(data, (z < 3.5 for z in z_mod(data))) 

```

`pass_outliers()`和`reject_outliers()`之间的区别很小，只是表达式的逻辑否定。一个版本中有`>=`，另一个版本中有`<`。这种代码差异并不总是容易验证。如果逻辑更复杂，执行逻辑否定是设计错误可能渗入代码的地方。

我们可以提取过滤规则的一个版本，创建类似以下内容：

```py
    outlier = lambda z: z >= 3.5 

```

然后我们可以修改`compress()`函数的两个用法，使逻辑否定明确：

```py
    def pass_outliers(data): 
        return itertools.compress(data, (outlier(z) for z in z_mod(data))) 

    def reject_outliers(data): 
        return itertools.compress(data, (not outlier(z) for z in z_mod(data))) 

```

将过滤规则公开为单独的lambda对象或函数定义有助于减少代码重复。否定更加明显。现在可以轻松比较这两个版本，以确保它们具有适当的语义。

如果我们想要使用`filter()`函数，我们必须对处理流水线进行根本性的转换。`filter()`高阶函数需要一个决策函数，为每个原始值创建一个真/假结果。处理这个将结合修改后的Z得分计算和决策阈值。决策函数必须计算这个：

![更多内容...](Image00052.jpg)

它必须计算这个，以确定每个*x[i]*值的异常值状态。这个决策函数需要两个额外的输入——总体中位数，![更多内容...](Image00053.jpg)，和MAD值。这使得过滤决策函数相当复杂。它会看起来像这样：

```py
    def outlier(mad, median_x, x): 
        return 0.6745*(x - median_x)/mad >= 3.5 

```

这个`outlier()`函数可以与`filter()`一起用于传递异常值。它可以与`itertools.filterfalse()`一起用于拒绝异常值并创建一个没有错误值的子集。

为了使用这个`outlier()`函数，我们需要创建一个类似这样的函数：

```py
    def pass_outliers2(data): 
        population_median = median(data) 
        mad = median_absdev(data, population_median) 
        outlier_partial = partial(outlier, mad, population_median) 
        return filter(outlier_partial, data) 

```

这计算了两个总体减少：`population_median`和`mad`。有了这两个值，我们可以创建一个偏函数`outlier_partial()`。这个函数将为前两个位置参数值`mad`和`population_median`绑定值。结果的偏函数只需要单独的数据值进行处理。

`outlier_partial()`和`filter()`处理等同于这个生成器表达式：

```py
    return ( 
        x for x in data if outlier(mad, population_median, x) 
    ) 

```

目前尚不清楚这个表达式是否比`itertools`模块中的`compress()`函数具有明显优势。但是，对于更熟悉`filter()`的程序员来说，它可能会更清晰一些。

## 另请参阅

+   有关异常值检测，请参见[http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm](http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)

# 一次分析多个变量

在许多情况下，我们会有多个变量的数据需要分析。数据可以可视化为填充网格，每一行包含特定的结果。每个结果行在列中有多个变量。

我们可以遵循列主序的模式，并独立处理每个变量（从数据列中）。这将导致多次访问每一行数据。或者，我们可以使用行主序的模式，并一次处理所有变量的每一行数据。

关注每个变量的优势在于，我们可以编写一个相对简单的处理堆栈。我们将有多个堆栈，每个变量一个，但每个堆栈可以重用`statistics`模块中的公共函数。

这种关注的缺点是，处理非常大数据集的每个变量需要从操作系统文件中读取原始数据。这个过程的这一部分可能是最耗时的。事实上，读取数据所需的时间通常主导了进行统计分析所需的时间。I/O成本如此之高，以至于专门的系统，如Hadoop，已被发明用来尝试加速对极大数据集的访问。

我们如何通过一组数据进行一次遍历并收集一些描述性统计信息？

## 准备工作

我们可能想要分析的变量将分为多个类别。例如，统计学家经常将变量分成以下类别：

+   **连续实值数据**：这些变量通常由浮点值测量，它们有一个明确定义的测量单位，并且它们可以取得由测量精度限制的值。

+   **离散或分类数据**：这些变量取自有限域中选择的值。在某些情况下，我们可以预先枚举域。在其他情况下，必须发现域的值。

+   **序数数据**：这提供了一个排名或顺序。通常，序数值是一个数字，但除此数字外，没有其他统计摘要适用于此数字，因为它实际上不是一个测量；它没有单位。

+   **计数数据**：这个变量是个别离散结果的摘要。通过计算一个实值均值，它可以被视为连续的。

变量可能彼此独立，也可能依赖于其他变量。在研究的初期阶段，可能不知道依赖关系。在后期阶段，软件的一个目标是发现这些依赖关系。之后，软件可以用来建模这些依赖关系。

由于数据的多样性，我们需要将每个变量视为一个独立的项目。我们不能将它们都视为简单的浮点值。适当地承认这些差异将导致一系列类定义的层次结构。每个子类将包含变量的独特特征。

我们有两种总体设计模式：

+   **急切**：我们可以尽早计算各种摘要。在某些情况下，我们不必为此积累太多数据。

+   **懒惰**：我们尽可能晚地计算摘要。这意味着我们将积累数据，并使用属性来计算摘要。

对于非常大的数据集，我们希望有一个混合解决方案。我们将急切地计算一些摘要，并且还使用属性从这些摘要中计算最终结果。

我们将使用*使用内置统计库*食谱中的一些数据，其中包括一系列相似数据系列中的两个变量。这些变量被命名为*x*和*y*，都是实值变量。*y*变量应该依赖于*x*变量，因此相关性和回归模型适用于这里。

我们可以用以下命令读取这些数据：

```py
 **>>> from pathlib import Path 
>>> import json 
>>> from collections import OrderedDict 
>>> source_path = Path('code/anscombe.json') 
>>> data = json.loads(source_path.read_text(), 
...     object_pairs_hook=OrderedDict)** 

```

我们已经定义了数据文件的路径。然后我们可以使用`Path`对象从该文件中读取文本。这些文本将被`json.loads()`使用，以从JSON数据构建Python对象。

我们已经包含了一个`object_pairs_hook`，以便该函数将使用`OrderedDict`类构建JSON，而不是默认的`dict`类。这将保留源文档中项目的原始顺序。

我们可以按以下方式检查数据：

```py
 **>>> [item['series'] for item in data] 
['I', 'II', 'III', 'IV'] 
>>> [len(item['data']) for item in data] 
[11, 11, 11, 11]** 

```

整个JSON文档是一个具有诸如`'I'`之类的键的子文档序列。每个子文档有两个字段："series"和"data"。在"data"数组中，有一个我们想要描述的观察值列表。每个观察值都有一对值。

## 如何做...

1.  定义一个类来处理变量的分析。这应该处理所有的转换和清洗。我们将使用混合处理方法：我们将在每个数据元素到达时更新总和和计数。直到请求这些属性时，我们才会计算最终的均值或标准差：

```py
            import math 
            class SimpleStats: 
                def __init__(self, name): 
                    self.name = name 
                    self.count = 0 
                    self.sum = 0 
                    self.sum_2 = 0 
                def cleanse(self, value): 
                    return float(value) 
                def add(self, value): 
                    value = self.cleanse(value) 
                    self.count += 1 
                    self.sum += value 
                    self.sum_2 += value*value 
                @property 
                def mean(self): 
                    return self.sum / self.count 
                @property 
                def stdev(self): 
                    return math.sqrt( 
                        (self.count*self.sum_2-self.sum**2)/(self.count*(self.count-1)) 
                        ) 

    ```

在这个例子中，我们已经为`count`、`sum`和平方和定义了摘要。我们可以扩展这个类以添加更多的计算。对于中位数或模式，我们将不得不积累个体值，并改变设计以完全懒惰。

1.  定义实例来处理输入列。我们将创建我们的`SimpleStats`类的两个实例。在这个示例中，我们选择了两个非常相似的变量，一个类就可以涵盖这两种情况：

```py
            x_stats = SimpleStats('x') 
            y_stats = SimpleStats('y') 

    ```

1.  定义实际列标题到统计计算对象的映射。在某些情况下，列可能不是通过名称标识的：我们可能使用列索引。在这种情况下，对象序列将与每行中的列序列匹配：

```py
            column_stats = { 
                'x': x_stats, 
                'y': y_stats 
            } 

    ```

1.  定义一个函数来处理所有行，使用每列的统计计算对象在每行内：

```py
            def analyze(series_data): 
                x_stats = SimpleStats('x') 
                y_stats = SimpleStats('y') 
                column_stats = { 
                    'x': x_stats, 
                    'y': y_stats 
                } 
                for item in series_data: 
                    for column_name in column_stats: 
                        column_stats[column_name].add(item[column_name]) 
                return column_stats 

    ```

外部`for`语句处理每一行数据。内部`for`语句处理每一行的每一列。处理显然是按行主要顺序进行的。

1.  显示来自各个对象的结果或摘要：

```py
            column_stats = analyze(series_data) 
            for column_key in column_stats: 
                print(' ', column_key, 
                      column_stats[column_key].mean, 
                      column_stats[column_key].stdev) 

    ```

我们可以将分析函数应用于一系列数据值。这将返回具有统计摘要的字典。

## 工作原理...

我们创建了一个处理特定类型列的清洁、过滤和统计处理的类。当面对各种类型的列时，我们将需要多个类定义。其思想是能够轻松创建相关类的层次结构。

我们为要分析的每个特定列创建了这个类的一个实例。在这个例子中，`SimpleStats`是为一个简单浮点值列设计的。其他设计可能适用于离散或有序数据。

该类的外部特性是`add()`方法。每个单独的数据值都提供给这个方法。`mean`和`stdev`属性计算摘要统计信息。

该类还定义了一个`cleanse()`方法，用于处理数据转换需求。这可以扩展到处理无效数据的可能性。它可能会过滤值，而不是引发异常。必须重写此方法以处理更复杂的数据转换。

我们创建了一组单独的统计处理对象。在这个例子中，集合中的两个项目都是`SimpleStats`的实例。在大多数情况下，将涉及多个类，并且统计处理对象的集合可能会相当复杂。

这些`SimpleStats`对象的集合应用于每行数据。`for`语句使用映射的键，这些键也是列名，将每列的数据与适当的统计处理对象相关联。

在某些情况下，统计摘要必须以惰性方式计算。例如，要发现异常值，我们需要所有数据。定位异常值的一种常见方法是计算中位数，计算与中位数的绝对偏差，然后计算这些绝对偏差的中位数。参见*定位异常值*配方。要计算模式，我们将所有数据值累积到`Counter`对象中。

## 还有更多...

在这种设计中，我们默认假设所有列都是完全独立的。在某些情况下，我们需要组合列来推导出额外的数据项。这将导致更复杂的类定义，可能包括对`SimpleStats`的其他实例的引用。确保按照依赖顺序处理列可能会变得相当复杂。

正如我们在[第8章](text00088.html#page "第8章。功能和响应式编程特性")的*使用堆叠的生成器表达式*配方中看到的，*功能和响应式编程特性*，我们可能会有一个涉及增强和计算派生值的多阶段处理。这进一步限制了列处理规则之间的顺序。处理这种情况的一种方法是为每个分析器提供与相关其他分析器的引用。我们可能会有以下相当复杂的一组类定义。

首先，我们将定义两个类来分别处理日期列和时间列。然后我们将结合这些类来创建基于两个输入列的时间戳列。

以下是处理日期列的类：

```py
    class DateStats: 
        def cleanse(self, value): 
            return datetime.datetime.strptime(date, '%Y-%m-%d').date() 
        def add(self, value): 
            self.current = self.cleanse(value) 

```

`DateStats`类只实现了`add()`方法。它清洗数据并保留当前值。我们可以为处理时间列定义类似的东西：

```py
    class TimeStats: 
        def cleanse(self, value): 
            return datetime.datetime.strptime(date, '%H:%M:%S').time() 
        def add(self, value): 
            self.current = self.cleanse(value) 

```

“TimeStats”类类似于“DateStats”；它只实现了“add()”方法。这两个类都专注于清洗源数据并保留当前值。

这是一个依赖于前两个类的类。这将使用“DateStats”和“TimeStats”对象的“current”属性来获取每个对象当前可用的值：

```py
    class DateTimeStats: 
        def __init__(self, date_column, time_column): 
            self.date_column = date_column 
            self.time_column = time_column 
        def add(self, value=None): 
            date = self.date_column.current 
            time = self.time_column.current 
            self.current = datetime.datetime.combine(date, time) 

```

“DateTimeStats”类结合了两个对象的结果。它需要一个“DateStats”类的实例和一个“TimeStats”类的实例。从这两个对象中，当前的清洗值作为“current”属性是可用的。

请注意，“value”参数在“DateTimeStats”实现的“add()”方法中未被使用。与接受“value”作为参数不同，值是从另外两个清洗对象中收集的。这要求在处理派生列之前，其他两列必须被处理。

为了确保值是可用的，需要对每一行进行一些额外的处理。基本的日期和时间处理映射到特定的列：

```py
    date_stats = DateStats() 
    time_stats = TimeStats() 
    column_stats = { 
        'date': date_stats, 
        'time': time_stats 
    } 

```

这个“column_stats”映射可以用来对每行数据应用两个基础数据清洗操作。然而，我们还有派生数据，必须在基础数据完成后计算。

我们可能会有这样的情况：

```py
    datetime_stats = DateTimeStats(date_stats, time_stats) 
    derived_stats = { 
        'datetime': datetime_stats 
    } 

```

我们建立了一个依赖于另外两个统计处理对象“date_stats”和“time_stats”的“DateTimeStats”实例。这个对象的“add()”方法将从另外两个对象中获取当前值。如果我们有其他派生列，我们可以将它们收集到这个映射中。

这个“derived_stats”映射可以用来应用统计处理操作，以创建和分析派生数据。整体处理循环现在有两个阶段：

```py
    for item in series_data: 
        for column_name in column_stats: 
            column_stats[column_name].add(item[column_name]) 
        for column_name in derived_stats: 
            derived_stats[column_name].add() 

```

我们已经为源数据中存在的列计算了统计数据。然后我们为派生列计算了统计数据。这个方法的一个令人愉快的特点是只使用了两个映射进行配置。我们可以通过更新“column_stats”和“derived_stats”映射来更改所使用的类。

### 使用map()

我们使用显式的“for”语句将每个统计对象应用于相应的列数据。我们也可以使用一个生成器表达式。我们甚至可以尝试使用“map()”函数。在[第8章](text00088.html#page "第8章. 函数式和响应式编程特性")的*组合map和reduce转换*一节中，可以了解到有关这种技术的一些额外背景。

另一个数据收集集合可能如下所示：

```py
    data_gathering = { 
        'x': lambda value: x_stats.add(value), 
        'y': lambda value: y_stats.add(value) 
    } 

```

我们提供了一个应用对象的“add()”方法到给定数据值的函数，而不是对象本身。

有了这个集合，我们可以使用一个生成器表达式：

```py
    [data_gathering[k](row[k]) for k in data_gathering)] 

```

这将对每个值“k”在行中可用的“data_gathering[k]”函数进行应用。

## 另请参阅

+   在[第6章](text00070.html#page "第6章. 类和对象的基础知识")的*类的设计与大量处理*和*使用惰性属性*一节中，还可以了解到一些适合这种整体方法的其他设计选择。
