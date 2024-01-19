# 选择算法

与在无序项目列表中查找元素相关的一组有趣的算法是选择算法。通过这样做，我们将回答与选择一组数字的中位数和选择列表中第 i 个最小或最大元素等问题有关的问题。

在本章中，我们将涵盖以下主题：

+   通过排序进行选择

+   随机选择

+   确定性选择

# 通过排序进行选择

列表中的项目可能会经历统计查询，如查找平均值、中位数和众数值。查找平均值和众数值不需要对列表进行排序。但是，要在数字列表中找到中位数，必须首先对列表进行排序。查找中位数需要找到有序列表中间位置的元素。但是，如果我们想要找到列表中的最后一个最小的项目或列表中的第一个最小的项目呢？

要找到无序项目列表中的第 i 个最小数字，重要的是要获得该项目出现的位置的索引。但是因为元素尚未排序，很难知道列表中索引为 0 的元素是否真的是最小的数字。

处理无序列表时要做的一个实际和明显的事情是首先对列表进行排序。一旦列表排序完成，就可以确保列表中的第零个元素将包含列表中的第一个最小元素。同样，列表中的最后一个元素将包含列表中的最后一个最小元素。

假设也许在执行搜索之前无法负担排序的奢侈。是否可能在不必首先对列表进行排序的情况下找到第 i 个最小的元素？

# 随机选择

在上一章中，我们研究了快速排序算法。快速排序算法允许我们对无序项目列表进行排序，但在排序算法运行时有一种保留元素索引的方式。一般来说，快速排序算法执行以下操作：

1.  选择一个枢轴。

1.  围绕枢轴对未排序的列表进行分区。

1.  使用*步骤 1*和*步骤 2*递归地对分区列表的两半进行排序。

一个有趣且重要的事实是，在每个分区步骤之后，枢轴的索引在列表变得排序后也不会改变。正是这个属性使我们能够在一个不太完全排序的列表中工作，以获得第 i 个最小的数字。因为随机选择是基于快速排序算法的，它通常被称为快速选择。

# 快速选择

快速选择算法用于获取无序项目列表中的第 i 个最小元素，即数字。我们将算法的主要方法声明如下：

```py
    def quick_select(array_list, left, right, k): 

        split = partition(array_list, left, right) 

        if split == k: 
            return array_list[split] 
        elif split < k: 
            return quick_select(array_list, split + 1, right, k) 
        else: 
            return quick_select(array_list, left, split-1, k) 
```

`quick_select`函数的参数是列表中第一个元素的索引和最后一个元素的索引。第三个参数`k`指定了第 i 个元素。允许大于或等于零（0）的值，这样当`k`为 0 时，我们知道要在列表中搜索第一个最小的项目。其他人喜欢处理`k`参数，使其直接映射到用户正在搜索的索引，以便第一个最小的数字映射到排序列表的 0 索引。这都是个人偏好的问题。

对`partition`函数的方法调用，`split = partition(array_list, left, right),`返回`split`索引。`split`数组的这个索引是无序列表中的位置，其中`right`到`split-1`之间的所有元素都小于`split`数组中包含的元素，而`split+1`到`left`之间的所有元素都大于`split`数组中包含的元素。

当`partition`函数返回`split`值时，我们将其与`k`进行比较，以找出`split`是否对应于第 k 个项目。

如果`split`小于`k`，那么意味着第 k 个最小的项目应该存在或者在`split+1`和`right`之间被找到：

![](img/d6955997-f9c9-42dc-bbb7-520ecb2d38c8.jpg)

在前面的例子中，一个想象中的无序列表在索引 5 处发生了分割，而我们正在寻找第二小的数字。由于 5<2 为`false`，因此进行递归调用以搜索列表的另一半：`quick_select(array_list, left, split-1, k)`。

如果`split`索引小于`k`，那么我们将调用`quick_select`：

![](img/dec66f50-c35c-4fdd-9b7c-37b5820141b0.jpg)

# 分区步骤

分区步骤与快速排序算法中的步骤完全相同。有几点值得注意：

```py
    def partition(unsorted_array, first_index, last_index): 
        if first_index == last_index: 
            return first_index 

        pivot = unsorted_array[first_index] 
        pivot_index = first_index 
        index_of_last_element = last_index 

        less_than_pivot_index = index_of_last_element 
        greater_than_pivot_index = first_index + 1 

        while True: 

            while unsorted_array[greater_than_pivot_index] < pivot and  
                  greater_than_pivot_index < last_index: 
                  greater_than_pivot_index += 1 
            while unsorted_array[less_than_pivot_index] > pivot and 
                  less_than_pivot_index >= first_index: 
                  less_than_pivot_index -= 1 

            if greater_than_pivot_index < less_than_pivot_index: 
                temp = unsorted_array[greater_than_pivot_index] 
                unsorted_array[greater_than_pivot_index] = 
                    unsorted_array[less_than_pivot_index] 
                unsorted_array[less_than_pivot_index] = temp 
            else: 
                break 

        unsorted_array[pivot_index] =  
            unsorted_array[less_than_pivot_index] 
        unsorted_array[less_than_pivot_index] = pivot 

        return less_than_pivot_index 
```

在函数定义的开头插入了一个 if 语句，以处理`first_index`等于`last_index`的情况。在这种情况下，这意味着我们的子列表中只有一个元素。因此，我们只需返回函数参数中的任何一个，即`first_index`。

总是选择第一个元素作为枢轴。这种选择使第一个元素成为枢轴是一个随机决定。通常不会产生良好的分割和随后的良好分区。然而，最终将找到第 i 个元素，即使枢轴是随机选择的。

`partition`函数返回由`less_than_pivot_index`指向的枢轴索引，正如我们在前一章中看到的。

从这一点开始，您需要用铅笔和纸跟随程序执行，以更好地了解如何使用分割变量来确定要搜索第 i 小项的列表的部分。

# 确定性选择

随机选择算法的最坏情况性能为**O**(*n²*)。可以改进随机选择算法的一部分以获得**O**(*n*)的最坏情况性能。这种算法称为**确定性选择**。

确定性算法的一般方法如下：

1.  选择一个枢轴：

1.  将无序项目列表分成每组五个元素。

1.  对所有组进行排序并找到中位数。

1.  递归重复*步骤 1*和*步骤 2*，以获得列表的真实中位数。

1.  使用真实中位数来分区无序项目列表。

1.  递归进入可能包含第 i 小元素的分区列表的部分。

# 枢轴选择

在随机选择算法中，我们选择第一个元素作为枢轴。我们将用一系列步骤替换该步骤，以便获得真实或近似中位数。这将改善关于枢轴的列表的分区：

```py
    def partition(unsorted_array, first_index, last_index): 

        if first_index == last_index: 
            return first_index 
        else: 
            nearest_median =     
            median_of_medians(unsorted_array[first_index:last_index]) 

        index_of_nearest_median = 
            get_index_of_nearest_median(unsorted_array, first_index, 
                                        last_index, nearest_median) 

        swap(unsorted_array, first_index, index_of_nearest_median) 

        pivot = unsorted_array[first_index] 
        pivot_index = first_index 
        index_of_last_element = last_index 

        less_than_pivot_index = index_of_last_element 
        greater_than_pivot_index = first_index + 1 
```

现在让我们来研究分区函数的代码。`nearest_median`变量存储给定列表的真实或近似中位数：

```py
    def partition(unsorted_array, first_index, last_index): 

        if first_index == last_index: 
            return first_index 
        else: 
            nearest_median =   
            median_of_medians(unsorted_array[first_index:last_index]) 
        .... 
```

如果`unsorted_array`参数只有一个元素，则`first_index`和`last_index`将相等。因此无论如何都会返回`first_index`。

然而，如果列表大小大于 1，我们将使用数组的部分调用`median_of_medians`函数，由`first_index`和`last_index`标记。返回值再次存储在`nearest_median`中。

# 中位数的中位数

`median_of_medians`函数负责找到任何给定项目列表的近似中位数。该函数使用递归返回真实中位数：

```py
def median_of_medians(elems): 

    sublists = [elems[j:j+5] for j in range(0, len(elems), 5)] 

    medians = [] 
    for sublist in sublists: 
        medians.append(sorted(sublist)[len(sublist)/2]) 

    if len(medians) <= 5: 
        return sorted(medians)[len(medians)/2] 
    else: 
        return median_of_medians(medians) 
```

该函数首先将列表`elems`分成每组五个元素。这意味着如果`elems`包含 100 个项目，则语句`sublists = [elems[j:j+5] for j in range(0, len(elems), 5)]`将创建 20 个组，每个组包含五个或更少的元素：

```py
    medians = [] 
        for sublist in sublists: 
            medians.append(sorted(sublist)[len(sublist)/2]) 
```

创建一个空数组并将其分配给`medians`，它存储分配给`sublists`的每个五个元素数组中的中位数。

for 循环遍历`sublists`中的列表列表。对每个子列表进行排序，找到中位数，并将其存储在`medians`列表中。

`medians.append(sorted(sublist)[len(sublist)/2])`语句将对列表进行排序，并获取存储在其中间索引的元素。这成为五个元素列表的中位数。由于列表的大小较小，使用现有的排序函数不会影响算法的性能。

我们从一开始就明白，我们不会对列表进行排序以找到第 i 小的元素，那么为什么要使用 Python 的排序方法呢？嗯，由于我们只对五个或更少的非常小的列表进行排序，因此该操作对算法的整体性能的影响被认为是可以忽略的。

此后，如果列表现在包含五个或更少的元素，我们将对`medians`列表进行排序，并返回位于其中间索引的元素：

```py
    if len(medians) <= 5: 
            return sorted(medians)[len(medians)/2] 
```

另一方面，如果列表的大小大于五，我们将再次递归调用`median_of_medians`函数，并向其提供存储在`medians`中的中位数列表。

例如，以下数字列表：

*[2, 3, 5, 4, 1, 12, 11, 13, 16, 7, 8, 6, 10, 9, 17, 15, 19, 20, 18, 23, 21, 22, 25, 24, 14]*

我们可以使用代码语句`sublists = [elems[j:j+5] for j in range(0, len(elems), 5)]`将此列表分成每个五个元素一组，以获得以下列表：

*[[2, 3, 5, 4, 1], [12, 11, 13, 16, 7], [8, 6, 10, 9, 17], [15, 19, 20, 18, 23], [21, 22, 25, 24, 14]]*

对每个五个元素的列表进行排序并获取它们的中位数，得到以下列表：

*[3, 12, 9, 19, 22]*

由于列表的大小为五个元素，我们只返回排序列表的中位数，或者我们将再次调用`median_of_median`函数。

# 分区步骤

现在我们已经获得了近似中位数，`get_index_of_nearest_median`函数使用`first`和`last`参数指示的列表边界：

```py
    def get_index_of_nearest_median(array_list, first, second, median): 
        if first == second: 
            return first 
        else: 
            return first + array_list[first:second].index(median) 
```

如果列表中只有一个元素，我们再次只返回第一个索引。`arraylist[first:second]`返回一个索引为 0 到`list-1`大小的数组。当我们找到中位数的索引时，由于新的范围索引`[first:second]`代码返回，我们会丢失它所在的列表部分。因此，我们必须将`arraylist[first:second]`返回的任何索引添加到`first`中，以获得找到中位数的真实索引：

```py
    swap(unsorted_array, first_index, index_of_nearest_median) 
```

然后，我们使用交换函数将`unsorted_array`中的第一个元素与`index_of_nearest_median`进行交换。

这里显示了交换两个数组元素的实用函数：

```py
def swap(array_list, first, second): 
    temp = array_list[first] 
    array_list[first] = array_list[second] 
    array_list[second] = temp 
```

我们的近似中位数现在存储在未排序列表的`first_index`处。

分区函数将继续进行，就像快速选择算法的代码一样。分区步骤之后，数组看起来是这样的：

![](img/1a8dca81-e1c7-41c1-9eda-1f7f98110b05.jpg)

```py

 def deterministic_select(array_list, left, right, k): 

        split = partition(array_list, left, right) 

        if split == k: 
            return array_list[split] 
        elif split < k : 
            return deterministic_select(array_list, split + 1, right, k) 
        else: 
            return deterministic_select(array_list, left, split-1, k)
```

正如您已经观察到的那样，确定选择算法的主要功能看起来与其随机选择对应物完全相同。在初始`array_list`围绕近似中位数进行分区后，将与第 k 个元素进行比较。

如果`split`小于`k`，则会递归调用`deterministic_select(array_list, split + 1, right, k)`。这将在数组的一半中寻找第 k 个元素。否则，将调用`deterministic_select(array_list, left, split-1, k)`函数。

# 总结

本章已经探讨了如何在列表中找到第 i 小的元素的方法。已经探讨了简单地对列表进行排序以执行查找第 i 小元素的操作的平凡解决方案。

还有可能不一定要在确定第 i 小的元素之前对列表进行排序。随机选择算法允许我们修改快速排序算法以确定第 i 小的元素。

为了进一步改进随机选择算法，以便我们可以获得**O**(*n*)的时间复杂度，我们着手寻找中位数的中位数，以便在分区期间找到一个良好的分割点。

从下一章开始，我们将改变重点，深入探讨 Python 的面向对象编程概念。
