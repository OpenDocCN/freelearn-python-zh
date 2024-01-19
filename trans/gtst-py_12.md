# 搜索

在前面章节中开发的数据结构中，对所有这些数据结构执行的一个关键操作是搜索。在本章中，我们将探讨可以用来在项目集合中查找元素的不同策略。

另一个利用搜索的重要操作是排序。在没有某种搜索操作的情况下，几乎不可能进行排序。搜索的“搜索方式”也很重要，因为它影响了排序算法的执行速度。

搜索算法分为两种广义类型。一种类型假定要对其应用搜索操作的项目列表已经排序，而另一种类型则没有。

搜索操作的性能受到即将搜索的项目是否已经排序的影响，我们将在后续主题中看到。

# 线性搜索

让我们把讨论重点放在线性搜索上，这是在典型的 Python 列表上执行的。

![](img/7d36496b-cfbd-4da0-8a58-8d7731f4d253.jpg)

前面的列表中的元素可以通过列表索引访问。为了在列表中找到一个元素，我们使用线性搜索技术。这种技术通过使用索引从列表的开头移动到末尾来遍历元素列表。检查每个元素，如果它与搜索项不匹配，则检查下一个元素。通过从一个元素跳到下一个元素，列表被顺序遍历。

在处理本章和其他章节中的部分时，我们使用包含整数的列表来增强我们的理解，因为整数易于比较。

# 无序线性搜索

包含元素**60**、**1**、**88**、**10**和**100**的列表是无序列表的一个示例。列表中的项目没有按大小顺序排列。要在这样的列表上执行搜索操作，首先从第一个项目开始，将其与搜索项目进行比较。如果没有匹配，则检查列表中的下一个元素。这将继续进行，直到我们到达列表中的最后一个元素或找到匹配为止。

```py
    def search(unordered_list, term): 
       unordered_list_size = len(unordered_list) 
        for i in range(unordered_list_size): 
            if term == unordered_list[i]: 
                return i 

        return None 
```

`search`函数的参数是包含我们数据的列表和我们要查找的项目，称为**搜索项**。

数组的大小被获取，并决定`for`循环执行的次数。

```py
        if term == unordered_list[i]: 
            ... 
```

在`for`循环的每次迭代中，我们测试搜索项是否等于索引指向的项目。如果为真，则无需继续搜索。我们返回发生匹配的位置。

如果循环运行到列表的末尾，没有找到匹配项，则返回`None`表示列表中没有这样的项目。

在无序项目列表中，没有关于如何插入元素的指导规则。这影响了搜索的方式。缺乏顺序意味着我们不能依赖任何规则来执行搜索。因此，我们必须逐个访问列表中的项目。如下图所示，对于术语**66**的搜索是从第一个元素开始的，然后移动到列表中的下一个元素。因此**60**与**66**进行比较，如果不相等，我们将**66**与**1**、**88**等进行比较，直到在列表中找到搜索项。

![](img/9bdc438d-d8b8-4a41-9ce1-f1a8aff57e33.jpg)

无序线性搜索的最坏情况运行时间为`O(n)`。在找到搜索项之前，可能需要访问所有元素。如果搜索项位于列表的最后位置，就会出现这种情况。

# 有序线性搜索

在列表的元素已经排序的情况下，我们的搜索算法可以得到改进。假设元素已按升序排序，搜索操作可以利用列表的有序性使搜索更有效。

算法简化为以下步骤：

1.  顺序移动列表。

1.  如果搜索项大于循环中当前检查的对象或项目，则退出并返回`None`。

在迭代列表的过程中，如果搜索项大于当前项目，则没有必要继续搜索。

![](img/19f35cd9-f5ea-41cf-96af-cae60034e6fe.jpg)

当搜索操作开始并且第一个元素与(**5**)进行比较时，没有匹配。但是因为列表中还有更多元素，搜索操作继续检查下一个元素。继续进行的更有力的原因是，我们知道搜索项可能与大于**2**的任何元素匹配。

经过第 4 次比较，我们得出结论，搜索项不能在**6**所在的位置之上找到。换句话说，如果当前项目大于搜索项，那么就意味着没有必要进一步搜索列表。

```py
    def search(ordered_list, term): 
        ordered_list_size = len(ordered_list) 
        for i in range(ordered_list_size): 
            if term == ordered_list[i]: 
                return i 
            elif ordered_list[i] > term: 
                return None 

        return None 
```

`if`语句现在适用于此检查。`elif`部分测试`ordered_list[i] > term`的条件。如果比较结果为`True`，则该方法返回`None`。

方法中的最后一行返回`None`，因为循环可能会遍历列表，但仍然找不到与搜索项匹配的任何元素。

有序线性搜索的最坏情况时间复杂度为`O(n)`。一般来说，这种搜索被认为是低效的，特别是在处理大型数据集时。

# 二进制搜索

二进制搜索是一种搜索策略，通过不断减少要搜索的数据量，从而提高搜索项被找到的速度，用于在列表中查找元素。

要使用二进制搜索算法，要操作的列表必须已经排序。

*二进制*这个术语有很多含义，它帮助我们正确理解算法。

在每次尝试在列表中查找项目时，必须做出二进制决策。一个关键的决定是猜测列表的哪一部分可能包含我们正在寻找的项目。搜索项是否在列表的前半部分还是后半部分，也就是说，如果我们总是将列表视为由两部分组成？

如果我们不是从列表的一个单元移动到另一个单元，而是采用一个经过教育的猜测策略，我们很可能会更快地找到项目的位置。

举个例子，假设我们想要找到一本 1000 页书的中间页。我们已经知道每本书的页码是从 1 开始顺序编号的。因此可以推断，第 500 页应该正好在书的中间，而不是从第 1 页、第 2 页翻到第 500 页。假设我们现在决定寻找第 250 页。我们仍然可以使用我们的策略轻松找到这一页。我们猜想第 500 页将书分成两半。第 250 页将位于书的左侧。不需要担心我们是否能在第 500 页和第 1000 页之间找到第 250 页，因为它永远不会在那里找到。因此，使用第 500 页作为参考，我们可以打开大约在第 1 页和第 500 页之间的一半页面。这让我们更接近找到第 250 页。

以下是对有序项目列表进行二进制搜索的算法：

```py
def binary_search(ordered_list, term): 

    size_of_list = len(ordered_list) - 1 

    index_of_first_element = 0 
    index_of_last_element = size_of_list 

    while index_of_first_element <= index_of_last_element: 
        mid_point = (index_of_first_element + index_of_last_element)/2 

        if ordered_list[mid_point] == term: 
            return mid_point 

        if term > ordered_list[mid_point]: 
            index_of_first_element = mid_point + 1 
        else: 
            index_of_last_element = mid_point - 1 

    if index_of_first_element > index_of_last_element: 
        return None 
```

假设我们要找到列表中项目**10**的位置如下：

![](img/b70fb7b0-21da-4886-82f4-436f7390ad50.jpg)

该算法使用`while`循环来迭代地调整列表中用于查找搜索项的限制。只要起始索引`index_of_first_element`和`index_of_last_element`索引之间的差异为正，`while`循环就会运行。

算法首先通过将第一个元素(**0**)的索引与最后一个元素(**4**)的索引相加，然后除以**2**找到列表的中间索引`mid_point`。

```py
mid_point = (index_of_first_element + index_of_last_element)/2 
```

在这种情况下，**10**并不在列表中间位置或索引上被找到。如果我们搜索的是**120**，我们将不得不将`index_of_first_element`调整为`mid_point +1`。但是因为**10**位于列表的另一侧，我们将`index_of_last_element`调整为`mid_point-1`：

![](img/14511500-0b46-4755-855d-80fd775aad4f.jpg)

现在我们的`index_of_first_element`和`index_of_last_element`的新索引分别为**0**和**1**，我们计算中点`(0 + 1)/2`，得到`0`。新的中点是**0**，我们找到中间项并与搜索项进行比较，`ordered_list[0]`得到值**10**。哇！我们找到了搜索项。

通过将`index_of_first_element`和`index_of_last_element`的索引重新调整，将列表大小减半，这一过程会持续到`index_of_first_element`小于`index_of_last_element`为止。当这种情况不成立时，很可能我们要搜索的项不在列表中。

这里的实现是迭代的。我们也可以通过应用移动标记搜索列表开头和结尾的相同原则，开发算法的递归变体。

```py
def binary_search(ordered_list, first_element_index, last_element_index, term): 

    if (last_element_index < first_element_index): 
        return None 
    else: 
        mid_point = first_element_index + ((last_element_index - first_element_index) / 2) 

        if ordered_list[mid_point] > term: 
            return binary_search(ordered_list, first_element_index, mid_point-1,term) 
        elif ordered_list[mid_point] < term: 
            return binary_search(ordered_list, mid_point+1, last_element_index, term) 
        else: 
            return mid_point 
```

对二分查找算法的这种递归实现的调用及其输出如下：

```py
    store = [2, 4, 5, 12, 43, 54, 60, 77]
    print(binary_search(store, 0, 7, 2))   

Output:
>> 0
```

递归二分查找和迭代二分查找之间唯一的区别是函数定义，以及计算`mid_point`的方式。在`((last_element_index - first_element_index) / 2)`操作之后，`mid_point`的计算必须将其结果加到`first_element_index`上。这样我们就定义了要尝试搜索的列表部分。

二分查找算法的最坏时间复杂度为`O(log n)`。每次迭代将列表减半，遵循元素数量的 log n 进展。

不言而喻，`log x`假定是指以 2 为底的对数。

# 插值搜索

二分查找算法的另一个变体可能更接近于模拟人类在任何项目列表上执行搜索的方式。它仍然基于尝试对排序的项目列表进行良好猜测，以便找到搜索项目的可能位置。

例如，检查以下项目列表：

![](img/a934a442-62ab-4a5b-bbf4-8bab3c197e09.jpg)

要找到**120**，我们知道要查看列表的右侧部分。我们对二分查找的初始处理通常会首先检查中间元素，以确定是否与搜索项匹配。

更人性化的做法是选择一个中间元素，不仅要将数组分成两半，还要尽可能接近搜索项。中间位置是根据以下规则计算的：

```py
mid_point = (index_of_first_element + index_of_last_element)/2 
```

我们将用一个更好的公式替换这个公式，这个公式将使我们更接近搜索项。`mid_point`将接收`nearest_mid`函数的返回值。

```py
def nearest_mid(input_list, lower_bound_index, upper_bound_index, search_value): 
    return lower_bound_index + (( upper_bound_index -lower_bound_index)/ (input_list[upper_bound_index] -input_list[lower_bound_index])) * (search_value -input_list[lower_bound_index]) 
```

`nearest_mid`函数的参数是要执行搜索的列表。`lower_bound_index`和`upper_bound_index`参数表示我们希望在其中找到搜索项的列表范围。`search_value`表示正在搜索的值。

这些值用于以下公式：

```py
lower_bound_index + (( upper_bound_index - lower_bound_index)/ (input_list[upper_bound_index] - input_list[lower_bound_index])) * (search_value - input_list[lower_bound_index]) 
```

给定我们的搜索列表，**44**，**60**，**75**，**100**，**120**，**230**和**250**，`nearest_mid`将使用以下值进行计算：

```py
lower_bound_index = 0
upper_bound_index = 6
input_list[upper_bound_index] = 250
input_list[lower_bound_index] = 44
search_value = 230
```

现在可以看到，`mid_point`将接收值**5**，这是我们搜索项位置的索引。二分查找将选择**100**作为中点，这将需要再次运行算法。

以下是典型二分查找与插值查找的更直观的区别。对于典型的二分查找，找到中点的方式如下：

![](img/698b4f5c-a425-4e73-a34e-be589ffb03ca.jpg)

可以看到，中点实际上大致站在前面列表的中间位置。这是通过列表 2 的除法得出的结果。

另一方面，插值搜索会这样移动：

![](img/a6b7460b-0e68-44c8-bf82-0feb97a17fae.jpg)

在插值搜索中，我们的中点更倾向于左边或右边。这是由于在除法时使用的乘数的影响。从前面的图片可以看出，我们的中点已经偏向右边。

插值算法的其余部分与二分搜索的方式相同，只是中间位置的计算方式不同。

```py
def interpolation_search(ordered_list, term): 

    size_of_list = len(ordered_list) - 1 

    index_of_first_element = 0 
    index_of_last_element = size_of_list 

    while index_of_first_element <= index_of_last_element: 
        mid_point = nearest_mid(ordered_list, index_of_first_element, index_of_last_element, term) 

        if mid_point > index_of_last_element or mid_point < index_of_first_element: 
            return None 

        if ordered_list[mid_point] == term: 
            return mid_point 

        if term > ordered_list[mid_point]: 
            index_of_first_element = mid_point + 1 
        else: 
            index_of_last_element = mid_point - 1 

    if index_of_first_element > index_of_last_element: 
        return None 
```

`nearest_mid`函数使用了乘法操作。这可能产生大于`upper_bound_index`或小于`lower_bound_index`的值。当发生这种情况时，意味着搜索项`term`不在列表中。因此返回`None`表示这一点。

那么当`ordered_list[mid_point]`不等于搜索项时会发生什么呢？好吧，我们现在必须重新调整`index_of_first_element`和`index_of_last_element`，使算法专注于可能包含搜索项的数组部分。这就像我们在二分搜索中所做的一样。

```py
if term > ordered_list[mid_point]: 
index_of_first_element = mid_point + 1 
```

如果搜索项大于`ordered_list[mid_point]`处存储的值，那么我们只需要调整`index_of_first_element`变量指向索引`mid_point + 1`。

下面的图片展示了调整的过程。`index_of_first_element`被调整并指向`mid_point+1`的索引。

![](img/21370208-8cfb-43ed-8074-00e953845d80.jpg)这张图片只是说明了中点的调整。在插值中，中点很少将列表均分为两半。

另一方面，如果搜索项小于`ordered_list[mid_point]`处存储的值，那么我们只需要调整`index_of_last_element`变量指向索引`mid_point - 1`。这个逻辑被捕捉在 if 语句的 else 部分中`index_of_last_element = mid_point - 1`。

![](img/fedd86bd-8262-4578-97f9-25e3baf39778.jpg)

这张图片展示了重新计算`index_of_last_element`对中点位置的影响。

让我们使用一个更实际的例子来理解二分搜索和插值算法的内部工作原理。

假设列表中有以下元素：

```py
[ 2, 4, 5, 12, 43, 54, 60, 77] 
```

索引 0 存储了 2，索引 7 找到了值 77。现在，假设我们想在列表中找到元素 2。这两种不同的算法会如何处理？

如果我们将这个列表传递给插值`search`函数，`nearest_mid`函数将返回一个等于`0`的值。仅仅通过一次比较，我们就可以找到搜索项。

另一方面，二分搜索算法需要三次比较才能找到搜索项，如下图所示：

![](img/672dbf08-20a0-4dbd-ba05-04fa1160a0d1.png)

第一个计算出的`mid_point`是`3`。第二个`mid_point`是`1`，最后一个找到搜索项的`mid_point`是`0`。

# 选择搜索算法

二分搜索和插值搜索操作的性能比有序和无序线性搜索函数都要好。由于在列表中顺序探测元素以找到搜索项，有序和无序线性搜索的时间复杂度为`O(n)`。当列表很大时，这会导致性能非常差。

另一方面，二分搜索操作在尝试搜索时会将列表切成两半。在每次迭代中，我们比线性策略更快地接近搜索项。时间复杂度为`O(log n)`。尽管使用二分搜索可以获得速度上的提升，但它不能用于未排序的项目列表，也不建议用于小型列表。

能够找到包含搜索项的列表部分在很大程度上决定了搜索算法的性能。在插值搜索算法中，计算中间值以获得更高概率的搜索项。插值搜索的时间复杂度为`O(log(log n))`，这比其变体二分搜索更快。

# 摘要

在本章中，我们考察了两种搜索算法。讨论了线性搜索和二分搜索算法的实现以及它们的比较。本节还讨论了二分搜索变体——插值搜索。在接下来的章节中，知道使用哪种搜索操作将是相关的。

在下一章中，我们将利用所学知识对项目列表执行排序操作。
