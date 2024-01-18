# 统计数据收集和报告

在本章中，您将学习有关用于科学计算的统计学中使用的高级 Python 库。您将学习有关 Python 的 NumPY、Pandas、Matplotlib 和 Plotly 模块。您将学习有关数据可视化技术，以及如何绘制收集到的数据。

在本章中，我们将涵盖以下主题：

+   NumPY 模块

+   Pandas 模块

+   数据可视化

# NumPY 模块

NumPY 是一个提供数组高效操作的 Python 模块。NumPY 是 Python 科学计算的基本包。这个包通常用于 Python 数据分析。NumPY 数组是多个值的网格。

通过在终端中运行以下命令来安装 NumPY：

```py
$ pip3 install numpy
```

我们将使用`numpy`库对`numpy`数组进行操作。现在我们将看看如何创建`numpy`数组。为此，请创建一个名为`simple_array.py`的脚本，并在其中编写以下代码：

```py
import numpy as np my_list1 = [1,2,3,4] my_array1 = np.array(my_list1) print(my_list11, type(my_list1))
print(my_array1, type(my_array1))
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 simple_array.py
```

输出如下：

```py
[1, 2, 3, 4] <class 'list'>
[1 2 3 4] <class 'numpy.ndarray'>
```

在前面的例子中，我们导入了`numpy`库作为`np`来使用`numpy`的功能。然后我们创建了一个简单的列表，将其转换为数组，我们使用了**`np.array()`**函数**。**最后，我们打印了带有类型的`numpy`数组，以便更容易理解普通数组和`numpy`数组。

上一个例子是单维数组的例子。现在我们将看一个多维数组的例子。为此，我们必须创建另一个列表。让我们看另一个例子。创建一个名为`mult_dim_array.py`的脚本，并在其中编写以下内容：

```py
import numpy as np my_list1 = [1,2,3,4] my_list2 = [11,22,33,44] my_lists = [my_list1, my_list2]
my_array = np.array(my_lists)
print(my_lists, type(my_lists)) print(my_array, type(my_array))
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 mult_dim_array.py
```

输出如下：

```py
[[1, 2, 3, 4], [11, 22, 33, 44]] <class 'list'>
[[ 1 2 3 4]
 [11 22 33 44]] <class 'numpy.ndarray'>
```

在前面的例子中，我们导入了`numpy`模块。之后，我们创建了两个列表：`my_list1`和`my_list2`。然后我们创建了另一个列表的列表（`my_list1`和`my_list2`），并在列表（`my_lists`）上应用了`np.array()`函数，并将其存储在一个名为`my_array`的对象中。最后，我们打印了`numpy`数组。

现在，我们将看一下可以对数组进行的更多操作。我们将学习如何知道我们创建的数组`my_array`的大小和数据类型；也就是说，应用`shape()`函数我们将得到数组的`size`，应用`dtype()`函数我们将知道数组的`数据类型`。让我们看一个例子。创建一个名为`size_and_dtype.py`的脚本，并在其中编写以下内容：

```py
import numpy as np my_list1 = [1,2,3,4] my_list2 = [11,22,33,44] my_lists = [my_list1,my_list2] my_array = np.array(my_lists) print(my_array) size = my_array.shape print(size) data_type = my_array.dtype print(data_type)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 size_and_dtype.py
```

输出如下：

```py
[[ 1  2  3  4]
 [11 22 33 44]] (2, 4) int64
```

在前面的例子中，我们应用了`shape`函数`my_array.shape`来获取数组的大小。输出是`(2, 4)`。然后我们在数组上应用了`dtype`函数`my_array.dtype`，输出是`int64`**。**

现在，我们将看一些特殊情况数组的例子。

首先，我们将使用`np.zeros()`函数创建一个所有值为零的数组，如下所示：

```py
student@ubuntu:~$ python3 Python 3.6.7 (default, Oct 22 2018, 11:32:17) [GCC 8.2.0] on linux Type "help", "copyright", "credits" or "license" for more information. >>> import numpy as np >>> np.zeros(5) array([0., 0., 0., 0., 0.]) >>> 
```

在创建所有值为零的数组之后，我们将使用`numpy`的`np.ones()`函数创建所有值为 1 的数组，如下所示：

```py
>>> np.ones((5,5)) array([[1., 1., 1., 1., 1.],
 [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]) >>> 
```

`np.ones((5,5))`创建一个所有值为`1`的`5*5`数组。

现在，我们将使用`numpy`的`np.empty()`函数创建一个空数组，如下所示：

```py
>>> np.empty([2,2]) array([[6.86506982e-317,  0.00000000e+000],
 [6.89930557e-310,  2.49398949e-306]]) >>> 
```

`np.empty()`不会像`np.zeros()`函数一样将数组值设置为零。因此，它可能更快。此外，它要求用户在数组中手动输入所有值，因此应谨慎使用。

现在，让我们看看如何使用`np.eye()`函数创建一个对角线值为`1`的单位矩阵，如下所示：

```py
>>> np.eye(5) array([[1., 0., 0., 0., 0.],
 [0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]]) >>> 
```

现在，我们将看一下`range`函数，它用于使用`numpy`的`np.arange()`函数创建数组，如下所示：

```py
>>> np.arange(10) array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) >>> 
```

`np.arange(10)`函数创建了范围为`0-9`的数组。我们定义了范围值`10`，因此数组索引值从`0`开始。

# 使用数组和标量

在这一部分，我们将看一下使用`numpy`进行数组的各种算术运算。首先，我们将创建一个多维数组，如下所示：

```py
student@ubuntu:~$ python3 Python 3.6.7 (default, Oct 22 2018, 11:32:17) [GCC 8.2.0] on linux Type "help", "copyright", "credits" or "license" for more information. >>> import numpy as np >>> from __future__ import division >>> arr = np.array([[4,5,6],[7,8,9]]) >>> arr array([[4, 5, 6],
 [7, 8, 9]]) >>> 
```

在这里，我们导入了`numpy`模块来使用`numpy`的功能，然后我们导入了`__future__`模块，它将处理浮点数。之后，我们创建了一个二维数组`arr`，对其进行各种操作。

现在，让我们看一下对数组的一些算术运算。首先，我们将学习数组的乘法，如下所示：

```py
>>> arr*arr array([[16, 25, 36],
 [49, 64, 81]]) >>> 
```

在上面的乘法操作中，我们将`arr`数组乘以两次以得到一个乘法数组。您也可以将两个不同的数组相乘。

现在，我们将看一下对数组进行减法操作，如下所示：

```py
>>> arr-arr array([[0, 0, 0],
 [0, 0, 0]]) >>> 
```

如前面的例子所示，我们只需使用`**-**`运算符来对两个数组进行减法。在减法操作之后，我们得到了结果数组，如前面的代码所示。

现在我们将看一下对标量进行数组的算术运算。让我们看一些操作：

```py
>>> 1 / arr array([[0.25             ,  0.2        ,   0.16666667],
 [0.14285714 ,   0.125     ,  0.11111111]]) >>> 
```

在上面的例子中，我们将`1`除以我们的数组并得到了输出。请记住，我们导入了`__future__`模块，它实际上对这样的操作非常有用，可以处理数组中的浮点值。

现在我们将看一下`numpy`数组的指数运算，如下所示：

```py
>>> arr ** 3 array([[ 64, 125, 216],
 [343, 512, 729]]) >>> 
```

在上面的例子中，我们对数组取了立方，并得到了每个值的立方作为输出。

# 数组索引

使用数组作为索引来对数组进行索引。使用索引数组，将返回原始数组的副本。`numpy`数组可以使用任何其他序列或使用任何其他数组进行索引，但不包括元组。数组中的最后一个元素可以通过`-1`进行索引，倒数第二个元素可以通过`-2`进行索引，依此类推。

因此，要对数组进行索引操作，首先我们创建一个新的`numpy`数组，为此我们将使用`range()`函数来创建数组，如下所示：

```py
student@ubuntu:~$ python3 Python 3.6.7 (default, Oct 22 2018, 11:32:17) [GCC 8.2.0] on linux Type "help", "copyright", "credits" or "license" for more information. >>> import numpy as np >>> arr = np.arange(0,16) >>> arr array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]) >>> 
```

在上面的例子中，我们创建了范围为`16`（即`0-15`）的数组`arr`。

现在，我们将对数组`arr`执行不同的索引操作。首先，让我们获取数组中特定索引处的值：

```py
>>> arr[7] 7 >>> 
```

在上面的例子中，我们通过其索引值访问了数组，并在将索引号传递给数组`arr`后，数组返回了值`7`，这是我们传递的特定索引号。

在获取特定索引处的值之后，我们将获取一定范围内的值。让我们看下面的例子：

```py
>>> arr[2:10] array([2, 3, 4, 5, 6, 7, 8, 9]) >>> arr[2:10:2] array([2, 4, 6, 8])>>>
```

在上面的例子中，首先我们访问了数组并得到了范围为（`2-10`）的值。结果显示为`array([2, 3, 4, 5, 6, 7, 8, 9])`。在第二个术语中，`arr[2:10:2]`，实际上是指定在范围`2-10`内以两步的间隔访问数组。这种索引的语法是`arr[_start_value_:_stop_value_:_steps_]`。因此，第二个术语的输出是`array([2, 4, 6, 8])`。

我们还可以从索引值开始获取数组中的值直到末尾，如下例所示：

```py
>>> arr[5:] array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]) >>> 
```

正如我们在上面的例子中看到的，我们从第 5 个索引值开始访问数组中的值直到末尾。结果，我们得到的输出是`array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])`。

现在我们将看一下`numpy`数组的切片。在切片中，我们实际上是取原始数组的一部分并将其存储在指定的数组名称中。让我们看一个例子：

```py
>>> arr_slice = arr[0:8] >>> arr_slice array([0, 1, 2, 3, 4, 5, 6, 7]) >>> 
```

在上面的例子中，我们对原始数组进行了切片。结果，我们得到了一个包含值`0,1,2,…..,7`的数组切片。我们还可以给数组切片赋予更新后的值。让我们看一个例子：

```py
>>> arr_slice[:] = 29 >>> arr_slice array([29, 29, 29, 29, 29, 29, 29, 29]) >>> 
```

在前面的例子中，我们将数组切片中的所有值设置为`29`。但在为数组切片分配值时，重要的是分配给切片的值也将分配给数组的原始集合。

让我们看看给数组的切片赋值后的结果，以及对我们原始数组的影响：

```py
>>> arr array([29, 29, 29, 29, 29, 29, 29, 29,  8,  9, 10, 11, 12, 13, 14, 15]) >>>
```

现在，我们将看另一个操作；即，复制数组。对数组进行切片和复制的区别在于，当我们对数组进行切片时，所做的更改将应用于原始数组。当我们获得数组的副本时，它会给出原始数组的显式副本。因此，对数组的副本应用的更改不会影响原始数组。所以让我们看一个复制数组的例子：

```py
>>> cpying_arr = arr.copy() >>> cpying_arr array([29, 29, 29, 29, 29, 29, 29, 29,  8,  9, 10, 11, 12, 13, 14, 15]) >>> 
```

在前面的例子中，我们只是复制了原始数组。为此，我们使用了`array_name.copy()`函数，输出是原始数组的副本。

# 对二维数组进行索引

二维数组是一个数组的数组。在这种情况下，数据元素的位置通常是指两个索引而不是一个，并且它表示具有行和列数据的表。现在我们将对这种类型的数组进行索引。

所以，让我们来看一个二维数组的例子：

```py
>>> td_array = np.array(([5,6,7],[8,9,10],[11,12,13])) >>> td_array array([[  5,   6,    7],
 [  8,   9,  10], [11, 12,  13]]) >>> 
```

在前面的例子中，我们创建了一个名为`td_array`的二维数组。创建数组后，我们打印了`td_array`。现在我们还将通过索引获取`td_array`中的值。让我们看一个通过索引访问值的例子：

```py
>>> td_array[1] array([ 8,  9, 10]) >>>
```

在前面的例子中，我们访问了数组的第一个索引值，并得到了输出。在这种类型的索引中，当我们访问值时，我们得到整个数组。除了获取整个数组，我们还可以访问特定的值。让我们来看一个例子：

```py
>>> td_array[1,0] 8 >>> 
```

在前面的例子中，我们通过传递两个值来访问`td_array`的行和列。如输出所示，我们得到了值`8`。

我们也可以以不同的方式设置二维数组。首先，将我们的二维数组长度增加。让我们将长度设置为`10`。因此，为此，我们创建一个所有元素都是零的示例数组，然后我们将在其中放入值。让我们看一个例子：

```py
>>> td_array = np.zeros((10,10)) >>> td_array array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]) >>> for i in range(10):
 ...     td_array[i] = i ... >>> td_array array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.], [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.], [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.], [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.], [6., 6., 6., 6., 6., 6., 6., 6., 6., 6.], [7., 7., 7., 7., 7., 7., 7., 7., 7., 7.], [8., 8., 8., 8., 8., 8., 8., 8., 8., 8.], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]]) >>>
```

在前面的例子中，我们创建了一个长度为`10`乘以`10`的二维数组。

现在让我们在其中进行一些花式索引，如下例所示：

```py
>>> td_array[[1,3,5,7]] array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
 [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.], [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.], [7., 7., 7., 7., 7., 7., 7., 7., 7., 7.]]) >>> 
```

在前面的例子中，我们获取了特定的索引值。因此，在结果中，我们得到了输出。

# 通用数组函数

通用函数对`numpy`数组中的所有元素执行操作。现在，我们将看一个例子，对数组执行多个通用函数。首先，我们将对数组进行平方根处理。创建一个名为`sqrt_array.py`的脚本，并在其中写入以下内容：

```py
import numpy as np array = np.arange(16) print("The Array is : ",array) Square_root = np.sqrt(array) print("Square root of given array is : ", Square_root)
```

运行脚本，你会得到以下输出：

```py
student@ubuntu:~/work$ python3 sqrt_array.py
```

输出如下：

```py
The Array is : [ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15] Square root of given array is : [0\. 1\. 1.41421356 1.73205081 2\. 2.23606798
 2.44948974 2.64575131 2.82842712 3\. 3.16227766 3.31662479 3.46410162 3.60555128 3.74165739 3.87298335]
```

在前面的例子中，我们使用`numpy`的`range`函数创建了一个简单的数组。然后我们对生成的数组应用了`sqrt()`函数，以获得数组的平方根。在获取数组的平方根后，我们将对数组应用另一个通用函数，即指数`exp()`函数。让我们看一个例子。创建一个名为`expo_array.py`的脚本，并在其中写入以下内容：

```py
import numpy as np array = np.arange(16) print("The Array is : ",array) exp = np.exp(array) print("exponential of given array is : ", exp)
```

运行脚本，你会得到以下输出：

```py
student@ubuntu:~/work$ python3 expo_array.py
```

输出如下：

```py
The Array is :  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15] exponential of given array is :  [1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01
 5.45981500e+01 1.48413159e+02 4.03428793e+02 1.09663316e+03 2.98095799e+03 8.10308393e+03 2.20264658e+04 5.98741417e+04 1.62754791e+05 4.42413392e+05 1.20260428e+06 3.26901737e+06]
```

在前面的例子中，我们使用`numpy`的`range`函数创建了一个简单的数组。然后我们对生成的数组应用了`exp()`函数，以获得数组的指数。

# Pandas 模块

在本节中，我们将学习有关 pandas 模块的知识。pandas 模块提供了快速灵活的数据结构，专为处理结构化和时间序列数据而设计。pandas 模块用于数据分析。pandas 模块是建立在 NumPY 和 Matplotlib 等包之上的，并为我们提供了大部分分析和可视化工作的场所。要使用此模块的功能，您必须首先导入它。

首先，通过运行以下命令安装我们示例中需要的以下软件包：

```py
$ pip3 install pandas $ pip3 install matplotlib
```

在这里，我们将看一些使用 pandas 模块的例子。我们将学习两种数据结构：系列和数据框。我们还将看到如何使用 pandas 从`csv`文件中读取数据。

# 系列

pandas 系列是一维数组。它可以容纳任何数据类型。标签被称为索引。现在，我们将看一个不声明索引的系列和声明索引的系列的例子。首先，我们将看一个不声明索引的系列的例子。为此，请创建一个名为`series_without_index.py`的脚本，并在其中写入以下内容：

```py
import pandas as pd import numpy as np s_data = pd.Series([10, 20, 30, 40], name = 'numbers') print(s_data)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 series_without_index.py
```

输出如下：

```py
0 10 1 20 2 30 3 40 Name: numbers, dtype: int64
```

在上面的例子中，我们学习了不声明索引的系列。首先，我们导入了两个模块：pandas 和`numpy`。接下来，我们创建了将存储系列数据的`s_data`对象。在该系列中，我们创建了一个列表，而不是声明索引，我们提供了 name 属性，该属性将为列表提供一个名称，然后我们打印了数据。在输出中，左列是数据的索引。即使我们从未提供索引，pandas 也会隐式地给出。索引将始终从`0`开始。在列的下方是我们系列的名称和值的数据类型。

现在，我们将看一个声明索引的系列的例子。在这里，我们还将执行索引和切片操作。为此，请创建一个名为`series_with_index.py`的脚本，并在其中写入以下内容：

```py
import pandas as pd import numpy as np s_data = pd.Series([10, 20, 30, 40], index = ['a', 'b', 'c', 'd'], name = 'numbers') print(s_data) print() print("The data at index 2 is: ", s_data[2]) print("The data from range 1 to 3 are:\n", s_data[1:3])
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 series_with_index.py a    10 b    20 c    30 d    40 Name: numbers, dtype: int64 

The data at index 2 is:  30 The data from range 1 to 3 are:
 b    20 c    30 Name: numbers, dtype: int64
```

在上面的例子中，我们为数据在`index`属性中提供了索引值。在输出中，左列是我们提供的索引值。

# 数据框

在本节中，我们将学习有关 pandas 数据框的知识。数据框是具有列并且可能是不同数据类型的二维标记数据结构。数据框类似于 SQL 表或电子表格。在使用 pandas 时，它们是最常见的对象。

现在，我们将看一个例子，从`csv`文件中读取数据到 DataFrame 中。为此，您必须在系统中有一个`csv`文件。如果您的系统中没有`csv`文件，请按以下方式创建一个名为`employee.csv`的文件：

```py
Id, Name, Department, Country 101, John, Finance, US 102, Mary, HR, Australia 103, Geeta, IT, India 104, Rahul, Marketing, India 105, Tom, Sales, Russia
```

现在，我们将把这个`csv`文件读入 DataFrame 中。为此，请创建一个名为`read_csv_dataframe.py`的脚本，并在其中写入以下内容：

```py
import pandas as pd file_name = 'employee.csv' df = pd.read_csv(file_name) print(df) print() print(df.head(3)) print() print(df.tail(1))
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 read_csv_dataframe.py Output:
 Id    Name  Department     Country 0  101    John     Finance          US 1  102    Mary          HR   Australia 2  103   Geeta          IT       India 3  104   Rahul   Marketing       India 4  105     Tom       Sales      Russia 

 Id    Name  Department     Country 0  101    John     Finance          US 1  102    Mary          HR   Australia 2  103   Geeta          IT       India
Id  Name  Department  Country 4  105   Tom       Sales   Russia
```

在上面的例子中，我们首先创建了一个名为`employee.csv`的`csv`文件。我们使用 pandas 模块创建数据框。目标是将`csv`文件读入 DataFrame 中。接下来，我们创建了一个`df`对象，并将`csv`文件的内容读入其中。接下来我们打印一个 DataFrame。在这里，我们使用`head()`和`tail()`方法来获取特定数量的数据行。我们指定了`head(3)`，这意味着我们打印了前三行数据。我们还指定了`tail(1)`，这意味着我们打印了最后一行数据。

# 数据可视化

数据可视化是描述理解数据重要性并以可视化方式放置数据的努力的术语。在本节中，我们将看一下以下数据可视化技术：

+   Matplotlib

+   Plotly

# Matplotlib

Matplotlib 是 Python 中的数据可视化库，它允许我们使用几行代码生成图表、直方图、功率谱、条形图、误差图、散点图等。Matplotlib 通常使事情变得更容易，最困难的事情也变得可能。

要在您的 Python 程序中使用`matplotlib`，首先我们必须安装`matplotlib`。在您的终端中运行以下命令来安装`matplotlib`：

```py
$ pip3 install matplotlib
```

现在，您还必须安装另一个包`tkinter`，用于图形表示。使用以下命令安装它：

```py
$ sudo apt install python3-tk
```

在上面的例子中，我们使用`plt.figure()`函数在不同的画布上绘制东西。之后，我们使用`plt.plot()`函数。这个函数有不同的参数，对于绘制图表很有用。在上面的例子中，我们使用了一些参数；即`x1`，`x2`，`y1`和`y2`。这些是用于绘制的相应轴点。

现在，我们将看一些`matplotlib`的例子。让我们从一个简单的例子开始。创建一个名为`simple_plot.py`的脚本，并在其中写入以下内容：

```py
import matplotlib.pyplot as plt import numpy as np x = np.linspace(0, 5, 10) y = x**2 plt.plot(x,y) plt.title("sample plot") plt.xlabel("x axis") plt.ylabel("y axis") plt.show()
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 simple_plot.py
```

输出如下：

![](img/8d8b8572-e237-441f-8089-399fd0396d31.jpg)

在上面的例子中，我们导入了两个模块，`matplotlib`和`numpy`，来可视化数据以及分别创建数组*x*和*y*。之后，我们将两个数组绘制为`plt.plot(x,y)`。然后我们使用`xlabel()`，`ylabel()`和`title()`函数向图表添加标题和标签，并使用`plt.show()`函数显示这个绘图。因为我们在 Python 脚本中使用 Matplotlib，不要忘记在最后一行添加`plt.show()`来显示您的绘图。

现在我们将创建两个数组来显示绘图中的两行曲线，并且我们将对这两条曲线应用样式。在下面的例子中，我们将使用`ggplot`样式来绘制图表。`ggplot`是一个用于声明性创建图形的系统，基于图形语法。要绘制`ghraph`，我们只需提供数据，然后告诉`ggplot`如何映射变量以及使用什么图形原语，它会处理细节。在大多数情况下，我们从`ggplot()`样式开始。

现在，创建一个名为`simple_plot2.py`的脚本，并在其中写入以下内容：

```py
import matplotlib.pyplot as plt from matplotlib import style style.use('ggplot') x1 = [0,5,10]
y1 = [12,16,6] x2 = [6,9,11] y2 = [6,16,8] plt.subplot(2,1,1) plt.plot(x1, y1, linewidth=3) plt.title("sample plot") plt.xlabel("x axis") plt.ylabel("y axis") plt.subplot(2,1,2) plt.plot(x2, y2, color = 'r', linewidth=3) plt.xlabel("x2 axis") plt.ylabel("y2 axis") plt.show()
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 simple_plot2.py
```

输出如下：

现在`matplotlib`已经安装在您的系统中，我们将看一些例子。在绘图时，有两个重要的组件：图和轴。图是充当绘制所有内容的窗口的容器。它可以有各种类型的独立图。轴是您可以绘制数据和与之相关的任何标签的区域。轴由一个`x`轴和一个`y`轴组成。

在上面的例子中，首先我们导入了所需的模块，然后我们使用`ggplot`样式来绘制图表。我们创建了两组数组；即`x1`，`y1`和`x2`，`y2`。然后我们使用 subplot 函数`plt.subplot()`，因为它允许我们在同一画布中绘制不同的东西。如果您想要在不同的画布上显示这两个图，您也可以使用`plt.figure()`函数而不是`plt.subplot()`。

输出如下：

```py
import matplotlib.pyplot as plt from matplotlib import style style.use('ggplot') x1 = [0,5,10] y1 = [12,16,6] x2 = [6,9,11] y2 = [6,16,8] plt.figure(1) plt.plot(x1, y1, color = 'g', linewidth=3) plt.title("sample plot") plt.xlabel("x axis") plt.ylabel("y axis") plt.savefig('my_sample_plot1.jpg') plt.figure(2) plt.plot(x2, y2, color = 'r', linewidth=3) plt.xlabel("x2 axis") plt.ylabel("y2 axis") plt.savefig('my_sample_plot2.jpg') plt.show()
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 simple_plot3.py
```

输出如下：

![](img/a92750b5-5c24-45f9-a017-f463bba4d645.jpg)![](img/8b8221ae-82a7-4746-a89c-5364f14e95d2.jpg)

现在，我们将看一下如何使用`plt.figure()`函数绘制数组并使用 Matplotlib 保存生成的图。您可以使用`savefig()`方法将它们保存为不同的格式，如`png`，`jpg`，`pdf`等。我们将把前面的图保存在一个名为`my_sample_plot.jpg`的文件中。现在，我们将看一个例子。为此，创建一个名为`simple_plot3.py`的脚本，并在其中写入以下内容：

然后，我们使用`color`参数为图形线条提供特定的颜色，并且在第三个参数中，我们使用`linewidth`，它决定了图形线条的宽度。之后，我们还使用了`savefig()`方法来以特定的图像格式保存我们的图。您可以在运行 Python 脚本的当前目录中检查它们（如果您没有指定路径）。

您可以通过直接访问该目录来打开这些图像，或者您也可以使用以下方法使用`matplotlib`来打开这些生成的图像。现在，我们将看一个打开保存的图的示例。为此，请创建一个名为`open_image.py`的脚本，并在其中写入以下内容：

```py
import matplotlib.pyplot as plt import matplotlib.image as mpimg plt.imshow(mpimg.imread('my_sample_plot1.jpg')) plt.show()
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 open_image.py
```

输出如下：

![](img/5f61279d-d859-4627-9969-958d6ba9cd1a.jpg)

在前面的例子中，我们使用了 Matplotlib 的`imshow()`函数来打开图的保存图像。

现在，我们将看一些不同类型的图。Matplotlib 允许我们创建不同类型的图来处理数组中的数据，如直方图、散点图、条形图等。使用不同类型的图取决于数据可视化的目的。让我们看一些这些图。

# 直方图

这种类型的图表帮助我们以一种无法仅仅使用均值或中位数来应付的方式来检查数值数据的分布。我们将使用`hist()`方法来创建一个简单的直方图。让我们看一个创建简单直方图的例子。为此，请创建一个名为`histogram_example.py`的脚本，并在其中写入以下内容：

```py
import matplotlib.pyplot as plt import numpy as np x = np.random.randn(500) plt.hist(x) plt.show()
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 histogram_example.py
```

输出如下：

![](img/eb702ea3-c0ab-469d-8aa9-07e1dd69fd78.jpg)

在前面的例子中，我们使用`numpy`创建了一组随机数。然后，我们使用`plt.hist()`方法绘制了这些数值数据。

# 散点图

这种类型的图表将数据显示为一组点。它提供了一种方便的方式来可视化数值值的关系。它还帮助我们理解多个变量之间的关系。我们将使用`scatter()`方法来绘制散点图中的数据。在散点图中，点的位置取决于其`x`和`y`轴的值；也就是说，数据集中的每个值都是水平或垂直维度中的一个位置。让我们看一个散点图的例子。创建一个名为`scatterplot_example.py`的脚本，并在其中写入以下内容：

```py
import matplotlib.pyplot as plt import numpy as np x = np.linspace(-2,2,100) y = np.random.randn(100) colors = np.random.rand(100) plt.scatter(x,y,c=colors) plt.show()
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 scatterplot_example.py
```

输出如下：

![](img/985d2b20-6987-46b5-a513-fcd962fd61de.jpg)

在前面的例子中，我们得到了`x`和`y`的值。然后，我们使用`plt.scatter()`方法来绘制这些值，以获得`x`和`y`值的散点图。

# 条形图

条形图是用矩形条表示数据的图表。您可以将它们垂直或水平绘制。创建一个名为`bar_chart.py`的脚本，并在其中写入以下内容：

```py
import matplotlib.pyplot as plt from matplotlib import style style.use('ggplot') x1 = [4,8,12] y1 = [12,16,6] x2 = [5,9,11] y2 = [6,16,8] plt.bar(x1,y1,color = 'g',linewidth=3) plt.bar(x2,y2,color = 'r',linewidth=3) plt.title("Bar plot") plt.xlabel("x axis") plt.ylabel("y axis") plt.show()
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 bar_chart.py
```

输出如下：

![](img/692d75ea-6b6f-4e8b-8290-2f3c9c654f2b.jpg)

在前面的例子中，我们有两组值：`x1`，`y1`和`x2`，`y2`。在获得数值数据后，我们使用`plt.bar()`方法来绘制当前数据的条形图。

有多种技术可用于绘制数据。其中，有几种使用`matplotlib`进行数据可视化的技术或方法，我们已经看到了。我们还可以使用另一种数据可视化工具`plotly`来执行这些操作。

# Plotly

Plotly 是 Python 中的一个交互式、开源的绘图库。它是一个图表库，提供了 30 多种图表类型，如科学图表、3D 图形、统计图表、金融图表等。

要在 Python 中使用`plotly`，首先我们必须在系统中安装它。要安装`plotly`，请在您的终端中运行以下命令：

```py
$ pip3 install plotly
```

我们可以在线和离线使用`plotly`。对于在线使用，你需要有一个`plotly`账户，之后你需要在 Python 中设置你的凭据：

```py
 plotly.tools.set_credentials_file(username='Username', api_key='APIkey')
```

要离线使用`plotly`，我们需要使用`plotly`函数：`plotly.offline.plot()`

在这一部分，我们将使用 plotly 离线。现在，我们将看一个简单的例子。为此，创建一个名为`sample_plotly.py`的脚本，并在其中写入以下内容：

```py
import plotly from plotly.graph_objs import Scatter, Layout plotly.offline.plot({
 "data": [Scatter(x=[1, 4, 3, 4], y=[4, 3, 2, 1])], "layout": Layout(title="plotly_sample_plot") })
```

将前面的脚本命名为`sample_plotly.py`运行。你将得到以下输出：

```py
student@ubuntu:~/work$ python3 sample_plotly.py
```

输出如下：

![](img/d639864a-5006-48d1-bbc5-9772566cc3b5.jpg)

在前面的例子中，我们导入了`plotly`模块，然后将`plotly`设置为离线使用。我们在其中放入了一些有用于绘制图表的参数。在例子中，我们使用了一些参数：`data`和`layout`。在`data`参数中，我们使用散点函数定义了`x`和`y`数组，这些数组具有要在`x`和`y`轴上绘制的值。然后我们使用`layout`参数，在其中我们定义了布局函数以为图表提供标题。前面程序的输出保存为 HTML 文件，并在默认浏览器中打开。这个 HTML 文件与你的脚本在同一个目录中。

现在让我们看一些不同类型的图表来可视化数据。所以，首先，我们将从散点图开始。

# 散点图

创建一个名为`scatter_plot_plotly.py`的脚本，并在其中写入以下内容：

```py
import plotly import plotly.graph_objs as go import numpy as np  x_axis = np.random.randn(100) y_axis = np.random.randn(100)  trace = go.Scatter(x=x_axis, y=y_axis, mode = 'markers') data_set = [trace] plotly.offline.plot(data_set, filename='scatter_plot.html')
```

运行脚本，你将得到以下输出：

```py
student@ubuntu:~/work$ python3 scatter_plot_plotly.py
```

输出如下：

![](img/728d2983-9f6e-4a22-8a99-1ce9a9e585bd.jpg)

在前面的例子中，我们导入了`plotly`，然后通过使用`numpy`创建了随机数据，并在脚本中导入了`numpy`模块。生成数据集后，我们创建了一个名为`trace`的对象，并将我们的数值数据插入其中以进行散点。最后，我们将`trace`对象中的数据放入`plotly.offline.plot()`函数中，以获得数据的散点图。与我们的第一个示例图一样，这个例子的输出也以 HTML 格式保存，并显示在默认的网络浏览器中。

# 线散点图

我们还可以创建一些更有信息量的图表，比如线散点图。让我们看一个例子。创建一个名为`line_scatter_plot.py`的脚本，并在其中写入以下内容：

```py
import plotly import plotly.graph_objs as go import numpy as np x_axis = np.linspace(0, 1, 50) y0_axis = np.random.randn(50)+5 y1_axis = np.random.randn(50) y2_axis = np.random.randn(50)-5 trace0 = go.Scatter(x = x_axis,y = y0_axis,mode = 'markers',name = 'markers') trace1 = go.Scatter(x = x_axis,y = y1_axis,mode = 'lines+markers',name = 'lines+markers') trace2 = go.Scatter(x = x_axis,y = y2_axis,mode = 'lines',name = 'lines') data_sets = [trace0, trace1, trace2] plotly.offline.plot(data_sets, filename='line_scatter_plot.html')
```

运行脚本，你将得到以下输出：

```py
student@ubuntu:~/work$ python3 line_scatter_plot.py
```

输出如下：

![](img/61d2fa81-8592-4c5a-95ad-d68670ca8126.jpg)

在前面的例子中，我们导入了`plotly`，以及`numpy`模块。然后我们为 x 轴生成了一些随机值，也为三个不同的 y 轴生成了随机值。之后，我们将这些数据放入创建的`trace`对象中，最后将该数据集放入 plotly 的离线函数中。然后我们得到了散点和线的格式的输出。这个例子的输出文件以`line_scatter_plot.html`的名称保存在你当前的目录中。

# 箱线图

箱线图通常是有信息量的，也很有帮助，特别是当你有太多要展示但数据很少的时候。让我们看一个例子。创建一个名为`plotly_box_plot.py`的脚本，并在其中写入以下内容：

```py
import random import plotly from numpy import * N = 50. c = ['hsl('+str(h)+',50%'+',50%)' for h in linspace(0, 360, N)] data_set = [{
 'y': 3.5*sin(pi * i/N) + i/N+(1.5+0.5*cos(pi*i/N))*random.rand(20), 'type':'box', 'marker':{'color': c[i]} } for i in range(int(N))] layout = {'xaxis': {'showgrid':False,'zeroline':False, 'tickangle':45,'showticklabels':False},
 'yaxis': {'zeroline':False,'gridcolor':'white'}, 'paper_bgcolor': 'rgb(233,233,233)', 'plot_bgcolor': 'rgb(233,233,233)', } plotly.offline.plot(data_set)
```

运行脚本，你将得到以下输出：

```py
student@ubuntu:~/work$ python3 plotly_box_plot.py
```

输出如下：

![](img/976d4c5a-f03e-4eb3-93d3-39456e5e1a92.jpg)

在前面的例子中，我们导入了`plotly`，以及`numpy`模块。然后我们声明 N 为箱线图中的总箱数，并通过固定颜色的饱和度和亮度以及围绕色调进行变化，生成了一个彩虹颜色的数组。每个箱子由一个包含数据、类型和颜色的字典表示。我们使用列表推导来描述 N 个不同颜色的箱子，每个箱子都有不同的随机生成的数据。之后，我们格式化输出的布局并通过离线的`plotly`函数绘制数据。

# 等高线图

轮廓图通常用作科学图，并在显示热图数据时经常使用。让我们看一个轮廓图的例子。创建一个名为`contour_plotly.py`的脚本，并在其中写入以下内容：

```py
from plotly import tools import plotly import plotly.graph_objs as go trace0 = go.Contour(
 z=[[1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 7, 12, 13, 14, 15, 16], [3, 1, 6, 11, 12, 13, 16, 17], [4, 2, 7, 7, 11, 14, 17, 18], [5, 3, 8, 8, 13, 15, 18, 19], [7, 4, 10, 9, 16, 18, 20, 19], [9, 10, 5, 27, 23, 21, 21, 21]], line=dict(smoothing=0), ) trace1 = go.Contour(
 z=[[1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 7, 12, 13, 14, 15, 16], [3, 1, 6, 11, 12, 13, 16, 17], [4, 2, 7, 7, 11, 14, 17, 18], [5, 3, 8, 8, 13, 15, 18, 19], [7, 4, 10, 9, 16, 18, 20, 19], [9, 10, 5, 27, 23, 21, 21, 21]], line=dict(smoothing=0.95), ) data = tools.make_subplots(rows=1, cols=2,
 subplot_titles=('Smoothing_not_applied', 'smoothing_applied')) data.append_trace(trace0, 1, 1) data.append_trace(trace1, 1, 2) plotly.offline.plot(data)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 contour_plotly.py This is the format of your plot grid: [ (1,1) x1,y1 ]  [ (1,2) x2,y2 ]
```

输出如下：

![](img/5ddc9f4d-947e-426e-ad4a-a6fa45c5a213.jpg)

在前面的例子中，我们取了一个数据集，并对其应用了`contour()`函数。然后我们将轮廓数据附加到`data_set`中，并最终对数据应用了`plotly`函数以获得输出。这些是 plotly 中用于以可视方式绘制数据的一些技术。

# 总结

在本章中，我们学习了 NumPY 和 Pandas 模块，以及数据可视化技术。在 NumPY 模块部分，我们学习了数组的索引和切片以及通用数组函数。在 pandas 模块部分，我们学习了 Series 和 DataFrames。我们还学习了如何将`csv`文件读入 DataFrame。在数据可视化中，我们学习了 Python 中用于数据可视化的库：`matplotlib`和`plotly`。

在下一章中，您将学习有关 MySQL 和 SQLite 数据库管理的知识。

# 问题

1.  什么是 NumPy 数组？

1.  以下代码片段的输出是什么？

```py
import numpy as np
# input array
in_arr1 = np.array([[ 1, 2, 3], [ -1, -2, -3]] )
print ("1st Input array : \n", in_arr1) 
in_arr2 = np.array([[ 4, 5, 6], [ -4, -5, -6]] )
print ("2nd Input array : \n", in_arr2) 
# Stacking the two arrays horizontally
out_arr = np.hstack((in_arr1, in_arr2))
print ("Output stacked array :\n ", out_arr)
```

1.  如何比`np.sum`更快地对小数组求和？

1.  如何从 Pandas DataFrame 中删除索引、行或列？

1.  如何将 Pandas DataFrame 写入文件？

1.  pandas 中的 NaN 是什么？

1.  如何从 pandas DataFrame 中删除重复项？

1.  如何更改使用 Matplotlib 绘制的图形的大小？

1.  Python 中绘制图形的可用替代方法是什么？

# 进一步阅读

+   10 分钟到 pandas 文档：[`pandas.pydata.org/pandas-docs/stable/`](https://pandas.pydata.org/pandas-docs/stable/)

+   NumPy 教程：[`docs.scipy.org/doc/numpy/user/quickstart.html`](https://docs.scipy.org/doc/numpy/user/quickstart.html)

+   使用 plotly 进行图形绘制：[`plot.ly/d3-js-for-python-and-pandas-charts/`](https://plot.ly/d3-js-for-python-and-pandas-charts/)
