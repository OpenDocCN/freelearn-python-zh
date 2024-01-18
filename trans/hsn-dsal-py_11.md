# 选择算法

与在无序项目列表中查找元素相关的一组有趣的算法是选择算法。给定一个元素列表，选择算法用于从列表中找到第i个最小元素。在这样做的过程中，我们将回答与选择一组数字的中位数和在列表中选择第i个最小或最大元素有关的问题。

在本章中，我们将涵盖以下主题：

+   排序选择

+   随机选择

+   确定性选择

# 技术要求

本章中使用的所有源代码都在以下GitHub链接中提供：[https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter11](https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter11)。

# 排序选择

列表中的项目可能会接受统计调查，比如找到平均值、中位数和众数。找到平均值和众数并不需要列表被排序。然而，要在数字列表中找到中位数，列表必须首先被排序。找到中位数需要你找到有序列表中间位置的元素。此外，当我们想要找到列表中最后最小的项目或者第一个最小的项目时，可以使用选择算法。

要在无序项目列表中找到第i个最小数，获取该项目出现的索引是很重要的。由于列表的元素没有排序，很难知道列表中索引为0的元素是否真的是第一个最小数。

处理无序列表时一个实用且明显的做法是首先对列表进行排序。在列表排序后，你可以放心地认为索引为0的元素将持有列表中的第一个最小元素。同样，列表中的最后一个元素将持有列表中的最后一个最小元素。然而，在长列表中应用排序算法来获取列表中的最小值或最大值并不是一个好的解决方案，因为排序是一个非常昂贵的操作。

让我们讨论一下是否可能在不排序列表的情况下找到第i个最小元素。

# 随机选择

在前一章中，我们讨论了快速排序算法。快速排序算法允许我们对无序项目列表进行排序，但在排序算法运行时保留元素索引的方法。一般来说，快速排序算法执行以下操作：

1.  选择一个主元素

1.  围绕主元素对未排序的列表进行分区

1.  使用*步骤1*和*步骤2*递归地对分区列表的两半进行排序

一个有趣且重要的事实是，在每次分区步骤之后，主元素的索引不会改变，即使列表已经排序。这意味着在每次迭代后，所选的主元素值将被放置在列表中的正确位置。正是这个属性使我们能够在一个不太完全排序的列表中获得第i个最小数。因为随机选择是基于快速排序算法的，它通常被称为快速选择。

# 快速选择

快速选择算法用于获取无序项目列表中的第k个最小元素，并基于快速排序算法。在快速排序中，我们递归地对主元素的两个子列表进行排序。在快速排序中，每次迭代中，我们知道主元素值达到了正确的位置，两个子列表（左子列表和右子列表）的所有元素都被设置为无序。

然而，在快速选择算法中，我们递归地调用函数，专门针对具有第`k`小元素的子列表。在快速选择算法中，我们将枢轴点的索引与`k`值进行比较，以获取给定无序列表中的第`k`小元素。快速选择算法中将会有三种情况，它们如下：

1.  如果枢轴点的索引小于`k`，那么我们可以确定第`k`小的值将出现在枢轴点右侧的子列表中。因此，我们只需递归地调用快速选择函数来处理右子列表。

1.  如果枢轴点的索引大于`k`，那么很明显第`k`小的元素将出现在枢轴点左侧。因此，我们只需递归地在左子列表中寻找第`i`个元素。

1.  如果枢轴点的索引等于`k`，那么意味着我们已经找到了第`k`小的值，并将其返回。

让我们通过一个例子来理解快速选择算法的工作原理。假设有一个元素列表`{45, 23, 87, 12, 72, 4, 54, 32, 52}`，我们想要找出这个列表中第3个最小的元素——我们通过使用快速排序算法来实现这一点。

我们通过选择一个枢轴值，即45，来开始算法。在算法的第一次迭代之后，我们将枢轴值放置在列表中的正确位置，即索引4（索引从0开始）。现在，我们将枢轴值的索引（即4）与`k`的值（即第3个位置，或索引2）进行比较。由于这是在`k<枢轴`点（即2<4），我们只考虑左子列表，并递归调用函数。

现在，我们取左子列表并选择枢轴点（即**4**）。运行后，**4**被放置在其正确的位置（即0索引）。由于枢轴的索引小于`k`的值，我们考虑右子列表。同样，我们将**23**作为枢轴点，它也被放置在了正确的位置。现在，当我们比较枢轴点的索引和`k`的值时，它们是相等的，这意味着我们已经找到了第3个最小的元素，并将其返回。

这个过程也在下面的图表中显示：

![](Images/15c61235-c781-4d49-aff5-38a00a946718.png)

要实现快速选择算法，我们首先需要了解主要函数，其中有三种可能的情况。我们将算法的主要方法声明如下：

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

`quick_select`函数接受列表中第一个元素的索引以及最后一个元素的索引作为参数。第三个参数`k`指定了第`i`个元素。`k`的值应该始终是正数；只有大于或等于零的值才被允许，这样当`k`为0时，我们知道要在列表中搜索第一个最小的项。其他人喜欢处理`k`参数，使其直接映射到用户正在搜索的索引，这样第一个最小的数字就映射到排序列表的`0`索引。

对`partition`函数的方法调用`split = partition(array_list, left, right)`，返回`split`索引。`split`数组的这个索引是无序列表中的位置，`right`到`split-1`之间的所有元素都小于`split`数组中包含的元素，而`split+1`到`left`之间的所有元素都大于它。

当`partition`函数返回`split`值时，我们将其与`k`进行比较，以找出`split`是否对应于第`k`个项。

如果`split`小于`k`，那么意味着第`k`小的项应该存在或者被找到在`split+1`和`right`之间：

![](Images/b609a32d-9262-441a-9e70-12d26502cff8.png)

在上述示例中，一个想象中的未排序列表在索引**5**处发生了分割，而我们正在寻找第二小的数字。由于5<2得到`false`，因此进行递归调用以返回`quick_select(array_list, left, split-1, k)`，以便搜索列表的另一半。

如果`split`索引小于`k`，那么我们将调用`quick_select`，如下所示：

![](Images/46160450-f11f-4a78-9139-0ceb47a7e1e3.png)

# 理解分区步骤

分区步骤类似于快速排序算法中的步骤。有几点值得注意：

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

在函数定义的开头插入了一个`if`语句，以应对`first_index`等于`last_index`的情况。在这种情况下，这意味着我们的子列表中只有一个元素。因此，我们只需返回函数参数中的任何一个，即`first_index`。

第一个元素总是选择为枢轴。这种选择使第一个元素成为枢轴是一个随机决定。通常不会产生良好的分割，随后也不会产生良好的分区。然而，最终将找到第`i^(th)`个元素，即使枢轴是随机选择的。

`partition`函数返回由`less_than_pivot_index`指向的枢轴索引，正如我们在前一章中看到的。

# 确定性选择

随机选择算法的最坏情况性能是`O(n²)`。可以通过改进随机选择算法的元素部分来获得`O(n)`的最坏情况性能。我们可以通过使用一个算法，即**确定性选择**，获得`O(n)`的性能。

中位数中位数是一种算法，它为我们提供了近似中位数值，即接近给定未排序元素列表的实际中位数的值。这个近似中位数通常用作快速选择算法中选择列表中第`i^(th)`最小元素的枢轴点。这是因为中位数中位数算法在线性时间内找到了估计中位数，当这个估计中位数用作快速选择算法中的枢轴点时，最坏情况下的运行时间复杂度从`O(n²)`大幅提高到线性的`O(n)`。因此，中位数中位数算法帮助快速选择算法表现得更好，因为选择了一个好的枢轴值。

确定性算法选择第`i^(th)`最小元素的一般方法如下：

1.  选择一个枢轴：

1.  将未排序项目的列表分成每组五个元素。

1.  对所有组进行排序并找到中位数。

1.  递归执行*步骤1*和*2*，以获得列表的真实中位数。

1.  使用真实中位数来分区未排序项目的列表。

1.  递归到可能包含第`i^(th)`最小元素的分区列表部分。

让我们考虑一个包含15个元素的示例列表，以了解确定性方法确定列表中第三个最小元素的工作原理。首先，您需要将具有5个元素的列表分成两个，并对子列表进行排序。一旦我们对列表进行了排序，我们就找出子列表的中位数，也就是说，元素**23**、**52**和**34**是这三个子列表的中位数。我们准备了所有子列表中位数的列表，然后对中位数列表进行排序。接下来，我们确定这个列表的中位数，也就是中位数的中位数，即**34**。这个值是整个列表的估计中位数，并用于选择整个列表的分区/枢轴点。由于枢轴值的索引为7，大于`i^(th)`值，我们递归考虑左子列表。

算法的功能如下图所示：

![](Images/b894d148-b32e-4111-88f6-9d397cbc815b.png)

# 枢轴选择

为了有效地确定列表中第i个最小值的确定性算法，我们首先要实现枢轴选择方法。在随机选择算法中，我们以前选择第一个元素作为枢轴。我们将用一系列步骤替换该步骤，使我们能够获得近似中位数。这将改善关于枢轴的列表的分区：

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

现在让我们了解partition函数的代码。nearest_median变量存储给定列表的真实或近似中位数：

```py
    def partition(unsorted_array, first_index, last_index): 

        if first_index == last_index: 
            return first_index 
        else: 
            nearest_median =   
            median_of_medians(unsorted_array[first_index:last_index]) 
        .... 
```

如果unsorted_array参数只有一个元素，first_index和last_index将相等。因此，first_index会被返回。

然而，如果列表的大小大于1，我们将使用由first_index和last_index标记的数组部分调用median_of_medians函数。返回值再次存储在nearest_median中。

# 中位数中位数

median_of_medians函数负责找到任何给定项目列表的近似中位数。该函数使用递归返回真正的中位数：

```py
def median_of_medians(elems): 

    sublists = [elems[j:j+5] for j in range(0, len(elems), 5)] 

    medians = [] 
    for sublist in sublists: 
        medians.append(sorted(sublist)[len(sublist)//2]) 

    if len(medians) <= 5: 
        return sorted(medians)[len(medians)//2] 
    else: 
        return median_of_medians(medians) 
```

该函数首先将列表elems分成每组五个元素。这意味着如果elems包含100个项目，将会有20个组，由sublists = [elems[j:j+5] for j in range(0, len(elems), 5)]语句创建，每个组包含恰好五个元素或更少：

```py
    medians = [] 
        for sublist in sublists: 
            medians.append(sorted(sublist)[len(sublist)/2]) 
```

创建一个空数组并将其分配给medians，它存储分配给sublists的每个五个元素数组中的中位数。

for循环遍历sublists中的列表列表。每个子列表都被排序，找到中位数，并存储在medians列表中。

medians.append(sorted(sublist)[len(sublist)//2])语句将对列表进行排序并获得存储在其中间索引的元素。这成为五个元素列表的中位数。由于列表的大小很小，使用现有的排序函数不会影响算法的性能。

从一开始我们就明白，我们不会对列表进行排序以找到第i个最小的元素，那么为什么要使用Python的sorted方法呢？嗯，因为我们要对一个非常小的列表进行排序，只有五个元素或更少，所以这个操作对算法的整体性能的影响被认为是可以忽略的。

此后，如果列表现在包含五个或更少的元素，我们将对medians列表进行排序，并返回位于其中间索引的元素：

```py
    if len(medians) <= 5: 
            return sorted(medians)[len(medians)/2] 
```

另一方面，如果列表的大小大于五，我们将再次递归调用median_of_medians函数，向其提供存储在medians中的中位数列表。

例如，为了更好地理解中位数中位数算法的概念，我们可以看下面的数字列表：

*[2, 3, 5, 4, 1, 12, 11, 13, 16, 7, 8, 6, 10, 9, 17, 15, 19, 20, 18, 23, 21, 22, 25, 24, 14]*

我们可以将这个列表分成每组五个元素，使用代码语句sublists = [elems[j:j+5] for j in range(0, len(elems), 5]来获得以下列表：

*[[2, 3, 5, 4, 1], [12, 11, 13, 16, 7], [8, 6, 10, 9, 17], [15, 19, 20, 18, 23], [21, 22, 25, 24, 14]]*

对每个五个元素的列表进行排序并获得它们的中位数，得到以下列表：

*[3, 12, 9, 19, 22]*

由于列表有五个元素，我们只返回排序后列表的中位数；否则，我们将再次调用median_of_median函数。

中位数中位数算法也可以用于选择快速排序算法中的枢轴点，从而将快速排序算法的最坏情况性能从O(n²)显著提高到O(n log n)的复杂度。

# 分区步骤

现在我们已经获得了近似中位数，get_index_of_nearest_median函数使用first和last参数指示的列表边界：

```py
    def get_index_of_nearest_median(array_list, first, second, median): 
        if first == second: 
            return first 
        else: 
            return first + array_list[first:second].index(median) 
```

再次，如果列表中只有一个元素，我们只返回第一个索引。但是，`arraylist[first:second]`返回一个索引从`0`到`list-1`大小的数组。当我们找到中位数的索引时，由于`[first:second]`代码返回的新范围索引，我们失去了它所在的列表部分。因此，我们必须将`arraylist[first:second]`返回的任何索引添加到`first`以获得中位数的真实索引位置：

```py
    swap(unsorted_array, first_index, index_of_nearest_median) 
```

然后使用`swap`函数将`unsorted_array`中的第一个元素与`index_of_nearest_median`进行交换。

交换两个数组元素的`utility`函数如下所示：

```py
def swap(array_list, first, second): 
    temp = array_list[first] 
    array_list[first] = array_list[second] 
    array_list[second] = temp 
```

我们的近似中位数现在存储在未排序列表的`first_index`处。

分区函数继续进行，就像在快速选择算法的代码中一样。分区步骤之后，数组看起来是这样的：

![](Images/ae12fc42-27b4-4a09-b4ad-561e164c2163.png)

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

正如您已经观察到的那样，确定性选择算法的主要函数看起来与其随机选择对应函数完全相同。在对初始的`array_list`进行分区以获得近似中位数之后，会与第`k`个元素进行比较。

如果`split`小于`k`，那么会对`deterministic_select(array_list, split + 1, right, k)`进行递归调用。这将在数组的一半中寻找第`k`个元素。否则，会调用`deterministic_select(array_list, left, split-1, k)`函数。

# 总结

在本章中，我们讨论了回答如何在列表中找到第`i`个最小元素的各种方法。探讨了简单地对列表进行排序以执行找到第`i`个最小元素操作的平凡解决方案。

还有可能不一定在确定第`i`个最小元素之前对列表进行排序。随机选择算法允许我们修改快速排序算法以确定第`i`个最小元素。

为了进一步改进随机选择算法，以便获得`O(n)`的时间复杂度，我们着手寻找中位数中的中位数，以便在分区过程中找到一个良好的分割点。

在下一章中，我们将探讨字符串的世界。我们将学习如何高效地存储和操作大量文本。还将涵盖数据结构和常见的字符串操作。
