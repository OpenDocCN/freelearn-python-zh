# 排序

当收集到数据时，总会有必要对数据进行排序。排序操作对所有数据集都是常见的，无论是名称集合、电话号码还是简单的待办事项列表。

在本章中，我们将学习一些排序技术，包括以下内容：

+   冒泡排序

+   插入排序

+   选择排序

+   快速排序

+   堆排序

在我们对这些排序算法的处理中，我们将考虑它们的渐近行为。一些算法相对容易开发，但性能可能较差。其他一些稍微复杂的算法将表现出色。

排序后，对一组项目进行搜索操作变得更加容易。我们将从最简单的排序算法开始--冒泡排序算法。

# 排序算法

在本章中，我们将介绍一些排序算法，这些算法的实现难度各不相同。排序算法根据它们的内存使用、复杂性、递归性质、是否基于比较等等因素进行分类。

一些算法使用更多的 CPU 周期，因此具有较差的渐近值。其他算法在对一些值进行排序时会消耗更多的内存和其他计算资源。另一个考虑因素是排序算法如何适合递归或迭代表达。有些算法使用比较作为排序元素的基础。冒泡排序算法就是一个例子。非比较排序算法的例子包括桶排序和鸽巢排序。

# 冒泡排序

冒泡排序算法的思想非常简单。给定一个无序列表，我们比较列表中的相邻元素，每次只放入正确的大小顺序，只有两个元素。该算法依赖于一个交换过程。

取一个只有两个元素的列表：

![](img/f0a85ca0-df91-41ff-9f1e-0b3e3f9d27a9.jpg)

要对这个列表进行排序，只需将它们交换到正确的位置，**2** 占据索引 **0**，**5** 占据索引 **1**。为了有效地交换这些元素，我们需要一个临时存储区域：

![](img/0970d2ce-bc86-4644-8abd-a3ca29004526.jpg)

冒泡排序算法的实现从交换方法开始，如前面的图像所示。首先，元素**5**将被复制到临时位置`temp`。然后元素**2**将被移动到索引**0**。最后，**5**将从 temp 移动到索引**1**。最终，元素将被交换。列表现在将包含元素：`[2, 5]`。以下代码将交换`unordered_list[j]`的元素与`unordered_list[j+1]`的元素，如果它们不是按正确顺序排列的：

```py
    temp = unordered_list[j] 
    unordered_list[j] = unordered_list[j+1] 
    unordered_list[j+1] = temp 
```

现在我们已经能够交换一个两元素数组，使用相同的思想对整个列表进行排序应该很简单。

我们将在一个双重嵌套循环中运行这个交换操作。内部循环如下：

```py
    for j in range(iteration_number): 
        if unordered_list[j] > unordered_list[j+1]: 
            temp = unordered_list[j] 
            unordered_list[j] = unordered_list[j+1] 
            unordered_list[j+1] = temp 
```

在实现冒泡排序算法时，知道交换的次数是很重要的。要对诸如`[3, 2, 1]`的数字列表进行排序，我们需要最多交换两次元素。这等于列表长度减 1，`iteration_number = len(unordered_list)-1`。我们减去`1`是因为它恰好给出了最大迭代次数：

![](img/1b019547-0241-48e8-b5b4-00c2e924fc9c.jpg)

通过在精确两次迭代中交换相邻元素，最大的数字最终位于列表的最后位置。

if 语句确保如果两个相邻元素已经按正确顺序排列，则不会发生不必要的交换。内部的 for 循环只会在我们的列表中精确发生两次相邻元素的交换。

然而，你会意识到第一次运行 `for` 循环并没有完全排序我们的列表。这个交换操作必须发生多少次，才能使整个列表排序好呢？如果我们重复整个交换相邻元素的过程多次，列表就会排序好。外部循环用于实现这一点。列表中元素的交换会导致以下动态变化：

![](img/39090fa9-d4b6-4852-857f-3b50ecb814e8.jpg)

我们意识到最多需要四次比较才能使我们的列表排序好。因此，内部和外部循环都必须运行 `len(unordered_list)-1` 次，才能使所有元素都排序好：

```py
iteration_number = len(unordered_list)-1 
    for i in range(iteration_number): 
        for j in range(iteration_number): 
            if unordered_list[j] > unordered_list[j+1]: 
                temp = unordered_list[j] 
                unordered_list[j] = unordered_list[j+1] 
                unordered_list[j+1] = temp
```

即使列表包含许多元素，也可以使用相同的原则。冒泡排序也有很多变体，可以最小化迭代和比较的次数。

冒泡排序是一种高度低效的排序算法，时间复杂度为 `O(n2)`，最佳情况为 `O(n)`。通常情况下，不应该使用冒泡排序算法来对大型列表进行排序。然而，在相对较小的列表上，它的性能还是相当不错的。

有一种冒泡排序算法的变体，如果在内部循环中没有比较，我们就会简单地退出整个排序过程。在内部循环中不需要交换元素的情况下，表明列表已经排序好了。在某种程度上，这可以帮助加快通常被认为是缓慢的算法。

# 插入排序

通过交换相邻元素来对一系列项目进行排序的想法也可以用于实现插入排序。在插入排序算法中，我们假设列表的某个部分已经排序好了，而另一部分仍然未排序。在这种假设下，我们遍历列表的未排序部分，一次选择一个元素。对于这个元素，我们遍历列表的排序部分，并按正确的顺序将其插入，以使列表的排序部分保持排序。这是很多语法。让我们通过一个例子来解释一下。

考虑以下数组：

![](img/49dda629-6835-48b4-828e-a815559f5aa2.jpg)

该算法首先使用 `for` 循环在索引 **1** 和 **4** 之间运行。我们从索引 **1** 开始，因为我们假设索引 **0** 的子数组已经按顺序排序好了：

![](img/53faaf1a-49f5-4089-a3e7-9814324f0587.jpg)

在循环执行开始时，我们有以下情况：

```py
    for index in range(1, len(unsorted_list)): 
        search_index = index 
        insert_value = unsorted_list[index] 
```

在每次运行 `for` 循环时，`unsorted_list[index]` 处的元素被存储在 `insert_value` 变量中。稍后，当我们找到列表排序部分的适当位置时，`insert_value` 将被存储在该索引或位置上：

```py
    for index in range(1, len(unsorted_list)): 
        search_index = index 
        insert_value = unsorted_list[index] 

        while search_index > 0 and unsorted_list[search_index-1] >     
              insert_value : 
            unsorted_list[search_index] = unsorted_list[search_index-1] 
            search_index -= 1 

        unsorted_list[search_index] = insert_value 
```

`search_index` 用于向 `while` 循环提供信息--确切地指出在列表的排序部分中需要插入的下一个元素的位置。

`while` 循环向后遍历列表，受两个条件的控制：首先，如果 `search_index > 0`，那么意味着列表的排序部分还有更多的元素；其次，`while` 循环运行时，`unsorted_list[search_index-1]` 必须大于 `insert_value`。`unsorted_list[search_index-1]` 数组将执行以下操作之一：

+   在第一次执行 `while` 循环之前，指向 `unsorted_list[search_index]` 之前的一个元素

+   在第一次运行 `while` 循环后，指向 `unsorted_list[search_index-1]` 之前的一个元素

在我们的列表示例中，`while` 循环将被执行，因为 `5 > 1`。在 `while` 循环的主体中，`unsorted_list[search_index-1]` 处的元素被存储在 `unsorted_list[search_index]` 处。`search_index -= 1` 使列表遍历向后移动，直到它的值为 `0`。

我们的列表现在是这样的：

![](img/b17904e5-ddf3-4b03-a469-27bb91a2c855.jpg)

`while`循环退出后，`search_index`的最后已知位置（在这种情况下为`0`）现在帮助我们知道在哪里插入`insert_value`：

![](img/c172bc52-6f58-4dd7-bf0d-a6cb59b078f0.jpg)

在`for`循环的第二次迭代中，`search_index`将具有值**2**，这是数组中第三个元素的索引。此时，我们从左向右（朝向索引**0**）开始比较。**100**将与**5**进行比较，但由于**100**大于**5**，`while`循环将不会执行。**100**将被自己替换，因为`search_index`变量从未被减少。因此，`unsorted_list[search_index] = insert_value`将不会产生任何效果。

当`search_index`指向索引**3**时，我们将**2**与**100**进行比较，并将**100**移动到**2**所存储的位置。然后我们将**2**与**5**进行比较，并将**5**移动到最初存储**100**的位置。此时，`while`循环将中断，**2**将存储在索引**1**中。数组将部分排序，值为`[1, 2, 5, 100, 10]`。

前面的步骤将再次发生一次，以便对列表进行排序。

插入排序算法被认为是稳定的，因为它不会改变具有相等键的元素的相对顺序。它也只需要的内存不多于列表消耗的内存，因为它是原地交换。

它的最坏情况值为**O**(n²)，最佳情况为**O**(n)。

# 选择排序

另一个流行的排序算法是选择排序。这种排序算法简单易懂，但效率低下，其最坏和最佳渐近值为**O**(*n²*)。它首先找到数组中最小的元素，并将其与数据交换，例如，数组索引[**0**]处的数据。然后再次执行相同的操作；然而，在找到第一个最小元素后，列表剩余部分中的最小元素将与索引[**1**]处的数据交换。

为了更好地解释算法的工作原理，让我们对一组数字进行排序：

![](img/8c80e026-2a8e-420c-932f-a5558325cd3b.jpg)

从索引**0**开始，我们搜索列表中在索引**1**和最后一个元素的索引之间存在的最小项。找到这个元素后，它将与索引**0**处找到的数据交换。我们只需重复此过程，直到列表变得有序。

在列表中搜索最小项是一个递增的过程：

![](img/f0024d27-cc67-4fd2-b74d-79829b6bc126.jpg)

对元素**2**和**5**进行比较，选择**2**作为较小的元素。这两个元素被交换。

交换操作后，数组如下所示：

![](img/fad6b44b-d9b6-4eb9-9de1-f1c042ab1273.jpg)

仍然在索引**0**处，我们将**2**与**65**进行比较：

![](img/f598f296-8582-48c3-bba4-3620c0aa7a77.jpg)

由于**65**大于**2**，所以这两个元素不会交换。然后在索引**0**处的元素**2**和索引**3**处的元素**10**之间进行了进一步的比较。不会发生交换。当我们到达列表中的最后一个元素时，最小的元素将占据索引**0**。

一个新的比较集将开始，但这一次是从索引**1**开始。我们重复整个比较过程，将存储在那里的元素与索引**2**到最后一个索引之间的所有元素进行比较。

第二次迭代的第一步将如下所示：

![](img/d3575fa3-2dc3-4adb-a3f0-648d31b7e1d8.jpg)

以下是选择排序算法的实现。函数的参数是我们想要按升序排列的未排序项目列表的大小：

```py
    def selection_sort(unsorted_list): 

        size_of_list = len(unsorted_list) 

        for i in range(size_of_list): 
            for j in range(i+1, size_of_list): 

                if unsorted_list[j] < unsorted_list[i]: 
                    temp = unsorted_list[i] 
                    unsorted_list[i] = unsorted_list[j] 
                    unsorted_list[j] = temp 
```

算法从使用外部`for`循环开始遍历列表`size_of_list`，多次。因为我们将`size_of_list`传递给`range`方法，它将产生一个从**0**到`size_of_list-1`的序列。这是一个微妙的注释。

内部循环负责遍历列表，并在遇到小于`unsorted_list[i]`指向的元素时进行必要的交换。注意，内部循环从`i+1`开始，直到`size_of_list-1`。内部循环开始在`i+1`之间搜索最小的元素，但使用`j`索引：

![](img/8a52cc8e-82c2-4959-81ad-713c2ff0e31a.jpg)

上图显示了算法搜索下一个最小项的方向。

# 快速排序

快速排序算法属于分治算法类，其中我们将问题分解为更简单的小块来解决。在这种情况下，未排序的数组被分解成部分排序的子数组，直到列表中的所有元素都处于正确的位置，此时我们的未排序列表将变为已排序。

# 列表分区

在我们将列表分成更小的块之前，我们必须对其进行分区。这是快速排序算法的核心。要对数组进行分区，我们必须首先选择一个枢轴。数组中的所有元素将与此枢轴进行比较。在分区过程结束时，小于枢轴的所有元素将位于枢轴的左侧，而大于枢轴的所有元素将位于数组中枢轴的右侧。

# 枢轴选择

为了简单起见，我们将任何数组中的第一个元素作为枢轴。这种枢轴选择会降低性能，特别是在对已排序列表进行排序时。随机选择数组中间或最后一个元素作为枢轴也不会改善情况。在下一章中，我们将采用更好的方法来选择枢轴，以帮助我们找到列表中的最小元素。

# 实施

在深入代码之前，让我们通过使用快速排序算法对列表进行排序的步骤。首先要理解分区步骤非常重要，因此我们将首先解决该操作。

考虑以下整数列表。我们将使用以下分区函数对此列表进行分区：

![](img/129d9de4-f95a-41ad-ae21-bb3f3cd31416.jpg)

```py

    def partition(unsorted_array, first_index, last_index): 

        pivot = unsorted_array[first_index] 
        pivot_index = first_index 
        index_of_last_element = last_index 

        less_than_pivot_index = index_of_last_element 
        greater_than_pivot_index = first_index + 1 
        ... 
```

分区函数接收我们需要分区的数组作为参数：其第一个元素的索引和最后一个元素的索引。

枢轴的值存储在`pivot`变量中，而其索引存储在`pivot_index`中。我们没有使用`unsorted_array[0]`，因为当调用未排序数组参数时，索引`0`不一定指向该数组中的第一个元素。枢轴的下一个元素的索引，`first_index + 1`，标记了我们开始在数组中寻找大于`pivot`的元素的位置，`greater_than_pivot_index = first_index + 1`。

`less_than_pivot_index = index_of_last_element`标记了列表中最后一个元素的位置，即我们开始搜索小于枢轴的元素的位置：

```py
    while True: 

        while unsorted_array[greater_than_pivot_index] < pivot and 
              greater_than_pivot_index < last_index: 
              greater_than_pivot_index += 1 

        while unsorted_array[less_than_pivot_index] > pivot and 
              less_than_pivot_index >= first_index: 
              less_than_pivot_index -= 1 
```

在执行主`while`循环之前，数组如下所示：

![](img/e135214b-12e9-49a2-8b7c-d9cde7b8deb6.jpg)

第一个内部`while`循环每次向右移动一个索引，直到落在索引**2**上，因为该索引处的值大于**43**。此时，第一个`while`循环中断并不再继续。在第一个`while`循环的条件测试中，只有当`while`循环的测试条件评估为`True`时，才会评估`greater_than_pivot_index += 1`。这使得对大于枢轴的元素的搜索向右侧的下一个元素进行。

第二个内部`while`循环每次向左移动一个索引，直到落在索引**5**上，其值**20**小于**43**：

![](img/6782b5b8-9018-4ce3-9403-d6d9d4772ce9.jpg)

此时，内部`while`循环都无法继续执行：

```py
    if greater_than_pivot_index < less_than_pivot_index: 
        temp = unsorted_array[greater_than_pivot_index] 
            unsorted_array[greater_than_pivot_index] =    
                unsorted_array[less_than_pivot_index] 
            unsorted_array[less_than_pivot_index] = temp 
    else: 
        break
```

由于`greater_than_pivot_index < less_than_pivot_index`，if 语句的主体交换了这些索引处的元素。else 条件在任何时候`greater_than_pivot_index`变得大于`less_than_pivot_index`时打破无限循环。在这种情况下，这意味着`greater_than_pivot_index`和`less_than_pivot_index`已经交叉。

我们的数组现在是这样的：

![](img/f4888a16-82c6-4fdf-8293-a3dfb72e8e24.jpg)

当`less_than_pivot_index`等于`3`且`greater_than_pivot_index`等于`4`时，执行 break 语句。

一旦我们退出`while`循环，我们就会交换`unsorted_array[less_than_pivot_index]`的元素和`less_than_pivot_index`的元素，后者作为枢轴的索引返回：

```py
    unsorted_array[pivot_index]=unsorted_array[less_than_pivot_index] 
    unsorted_array[less_than_pivot_index]=pivot 
    return less_than_pivot_index 
```

下面的图片显示了代码在分区过程的最后一步中如何交换 4 和 43：

![](img/3dbea1c3-5526-45a0-89c1-c3952d09044c.jpg)

回顾一下，第一次调用快速排序函数时，它是围绕索引**0**的元素进行分区的。在分区函数返回后，我们得到数组`[4, 3, 20, 43, 89, 77]`。

正如你所看到的，元素**43**右边的所有元素都更大，而左边的元素更小。分区完成了。

使用分割点 43 和索引 3，我们将递归地对两个子数组`[4, 30, 20]`和`[89, 77]`进行排序，使用刚刚经历的相同过程。

主`quick sort`函数的主体如下：

```py
    def quick_sort(unsorted_array, first, last): 
        if last - first <= 0: 
            return 
    else: 
        partition_point = partition(unsorted_array, first, last) 
        quick_sort(unsorted_array, first, partition_point-1) 
        quick_sort(unsorted_array, partition_point+1, last) 
```

`quick sort`函数是一个非常简单的方法，不超过 6 行代码。繁重的工作由`partition`函数完成。当调用`partition`方法时，它返回分区点。这是`unsorted_array`中的一个点，左边的所有元素都小于枢轴，右边的所有元素都大于它。

当我们在分区进度之后立即打印`unsorted_array`的状态时，我们清楚地看到了分区是如何进行的：

```py
Output:
[43, 3, 20, 89, 4, 77]
[4, 3, 20, 43, 89, 77]
[3, 4, 20, 43, 89, 77]
[3, 4, 20, 43, 77, 89]
[3, 4, 20, 43, 77, 89]
```

退一步，让我们在第一次分区发生后对第一个子数组进行排序。`[4, 3, 20]`子数组的分区将在`greater_than_pivot_index`在索引`2`和`less_than_pivot_index`在索引`1`时停止。在那一点上，两个标记被认为已经交叉。因为`greater_than_pivot_index`大于`less_than_pivot_index`，`while`循环的进一步执行将停止。枢轴 4 将与`3`交换，而索引`1`将作为分区点返回。

快速排序算法的最坏情况复杂度为**O**(*n²*)，但在对大量数据进行排序时效率很高。

# 堆排序

在第十一章中，*图和其他算法*，我们实现了（二叉）堆数据结构。我们的实现始终确保在元素被移除或添加到堆后，使用 sink 和 float 辅助方法来维护堆顺序属性。

堆数据结构可以用来实现称为堆排序的排序算法。回顾一下，让我们创建一个简单的堆，其中包含以下项目：

```py
    h = Heap() 
    unsorted_list = [4, 8, 7, 2, 9, 10, 5, 1, 3, 6] 
    for i in unsorted_list: 
        h.insert(i) 
    print("Unsorted list: {}".format(unsorted_list)) 
```

堆`h`被创建，并且`unsorted_list`中的元素被插入。在每次调用`insert`方法后，堆顺序属性都会通过随后调用`float`方法得到恢复。循环终止后，我们的堆顶部将是元素`4`。

我们堆中的元素数量是`10`。如果我们在堆对象`h`上调用`pop`方法 10 次并存储实际弹出的元素，我们最终得到一个排序好的列表。每次`pop`操作后，堆都会重新调整以保持堆顺序属性。

`heap_sort`方法如下：

```py
    class Heap: 
        ... 
        def heap_sort(self): 
            sorted_list = [] 
            for node in range(self.size): 
                n = self.pop() 
                sorted_list.append(n) 

            return sorted_list 
```

`for`循环简单地调用`pop`方法`self.size`次。循环终止后，`sorted_list`将包含一个排序好的项目列表。

`insert`方法被调用*n*次。与`float`方法一起，`insert`操作的最坏情况运行时间为**O**(*n log n*)，`pop`方法也是如此。因此，这种排序算法的最坏情况运行时间为**O**(*n log n*)。

# 总结

在本章中，我们探讨了许多排序算法。快速排序比其他排序算法表现要好得多。在讨论的所有算法中，快速排序保留了它所排序的列表的索引。在下一章中，我们将利用这一特性来探讨选择算法。
