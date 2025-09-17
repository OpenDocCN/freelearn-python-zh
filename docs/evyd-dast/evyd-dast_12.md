# 第十二章. 排序：从混乱中带来秩序

能够为特定应用构建正确的数据结构或集合类只是战斗的一半。除非你的问题域中的数据集非常小，否则你的数据集合将受益于一点组织。通过特定的值或值集对列表或集合中的元素进行组织被称为**排序**。

对数据进行排序并非绝对必要，但这样做可以使搜索或查找操作更加高效。同样，当你需要合并多个数据集时，在合并之前对各种数据集进行排序可以大大提高合并操作的效率。

如果你的数据是一组数值，那么排序可能只是按升序或降序排列它。然而，如果你的数据由复杂对象组成，你可以通过特定的值对集合进行排序。在这种情况下，数据排序所依据的字段或属性被称为**键**。例如，如果你有一个汽车对象的集合，并且你想按制造商（如福特、雪佛兰或道奇）对其进行排序，那么制造商就是键。然而，如果你想要按多个键排序，比如制造商和型号，那么制造商成为**主键**，而型号成为**次键**。这种模式的进一步扩展将导致**三级键**、**四级键**等等。

排序算法形态各异，大小不一，其中许多特别适合特定的数据结构。尽管对已知或甚至只是流行的排序算法进行全面考察超出了本书的范围，但在本章中，我们将重点关注那些相对常见或非常适合我们已考察的一些数据结构的算法。在每种情况下，我们将回顾我们一直在查看的四种语言中的示例，并讨论复杂度成本。在本章中，我们将涵盖以下内容：

+   选择排序

+   插入排序

+   冒泡排序

+   快速排序

+   归并排序

+   桶排序

+   计数排序

# 选择排序

选择排序可以被描述为原地比较。这个算法将一个集合或对象列表分为两部分。第一部分是已经排序的对象子集，范围从*0*到*i*，其中*i*是下一个要排序的对象。第二部分是尚未排序的对象子集，范围从*i*到*n*，其中*n*是集合的长度。

选择排序算法通过在集合中找到最小或最大值，并通过与当前索引的对象交换来将其放置在未排序子数组的开头。例如，考虑按升序对集合进行排序。一开始，已排序的子数组将包含 0 个成员，而未排序的子数组将包含集合中的所有成员。选择排序算法将在未排序的子数组中找到最小的成员，并将其放置在未排序子数组的开头。

到目前为止，已排序的子数组包含一个成员，而未排序的子数组包含原始集合中所有剩余的成员。这个过程将重复进行，直到未排序子数组中的所有成员都被放置在已排序子数组中。

给定以下值集合：

*S = {50, 25, 73, 21, 3}*

我们的算法将在*S[0...4]*中找到最小的值，在这个例子中是*3*，并将其放置在*S[0...4]:*的开头。

*S = {3, 25, 73, 21, 50}*

这个过程会重复进行*S[1...4]*，返回值为 21：

*S = {3, 21, 73, 25, 50}*

在*S[2...4]*的下一个评估返回值为 25：

*S = {3, 21, 25, 73, 50}*

最后，函数再次对*S[3...4]*进行重复，返回最小值 50：

*S = {3, 21, 25, 50, 73}*

没有必要检查集合中的最后一个对象，因为它，按照必然性，已经是剩余的最大值。然而，这只能算是一点点安慰，因为选择排序算法仍然有**O**(*n²*)的复杂度成本。此外，这个最坏情况的复杂度分数并不能完全说明这个特定情况。选择排序始终是**O**(*n²*)的复杂度，即使在最佳情况下也是如此。因此，选择排序可能是你可能会遇到的慢速且效率最低的排序算法。

### 备注

本章中的每个代码示例都将检查算法，以这些方法最基本的形式，这些方法与其父类分离。此外，在每个情况下，要排序的对象集合将在类级别定义，在示例代码之外。同样，后续的对象实例化和这些集合的填充也将定义在示例代码之外。要查看完整的类示例，请使用伴随此文本的代码示例。

**C#**

```py
    public void SelectionSort(int[] values) 
    { 
        if (values.Length <= 1) 
        return; 

        int j, minIndex; 
        for (int i = 0; i < values.Length - 1; i++) 
        { 
            minIndex = i; 
            for (j = i + 1; j < values.Length; j++) 
            { 
                if (values[j] < values[minIndex]) 
                { 
                    minIndex = j; 
                } 
            } 
            Swap(ref values[minIndex], ref values[i]); 
        } 
    } 

    void Swap(ref int x, ref int y) 
    { 
        int t = x; 
        x = y; 
        y = t; 
    } 

```

我们对`SelectionSort`方法的每个实现都是从确认`values`数组至少有两个成员开始的。如果没有，该方法将返回，因为没有足够的成员进行排序。否则，我们创建两个嵌套循环。外层`for`循环每次移动未排序数组的边界一个索引，而内层`for`循环用于在未排序边界内找到最小值。一旦我们得到最小值，方法就会将`i`处的成员与当前最小值的成员进行交换。由于 C#默认不支持通过引用传递原始数据类型，我们必须在`swap(ref int x, ref int y)`方法签名以及调用的参数上显式调用`ref`关键字。尽管创建一个单独的`swap`方法来执行此操作可能看起来更麻烦，但交换功能是几种流行排序算法的共同点，将此代码放在单独的方法中可以在以后节省一些按键操作。

### 提示

**嵌套循环**

记住，嵌套循环会自动使算法的复杂度呈指数级增加。任何包含`for`循环的算法都有复杂度成本**O**(n)，但一旦在第一个`for`循环内嵌套另一个`for`循环，复杂度成本就增加到**O**(n²)。在第二个循环内嵌套另一个`for`循环会使成本增加到**O**(n³)，依此类推。

还要注意，在任何实现中嵌套`for`循环都会成为观察者注意的红旗，你应该总是准备好为这种设计进行辩护。只有在你绝对必须的时候才嵌套`for`循环。

**Java**

```py
    public void selectionSort(int[] values) 
    { 
        if (values.length <= 1) 
            return; 

        int j, minIndex; 
        for (int i = 0; i < values.length - 1; i++) 
        { 
            minIndex = i; 
            for (j = i + 1; j < values.length; j++) 
            { 
                if (values[j] < values[minIndex]) 
                { 
                    minIndex = j; 
                } 
            } 

            int temp = values[minIndex]; 
            values[minIndex] = values[i]; 
            values[i] = temp; 
        } 
    } 

```

Java 实现的设计几乎与 C#实现相同，除了数组`length`函数的名称。然而，Java 根本不支持通过引用传递原始数据类型。尽管可以通过将原始数据类型传递给可变包装类的实例来模拟这种行为，但大多数开发者都认为这是一个坏主意。相反，我们的 Java 实现直接在`for`循环内执行交换。

**Objective-C**

```py
    -(void)selectionSort:(NSMutableArray<NSNumber*>*)values 
    { 
        if ([values count] <= 1) 
            return; 

        NSInteger j, minIndex; 
        for (int i = 0; i < [values count] - 1; i++) 
        { 
            minIndex = i; 
            for (j = i + 1; j < [values count]; j++) 
            { 
                if ([values[j] intValue] < [values[minIndex] intValue]) 
                { 
                    minIndex = j; 
                } 
            } 

            NSNumber *temp = (NSNumber*)values[minIndex]; 
            values[minIndex] = values[i]; 
            values[i] = temp; 
        } 
    } 

```

由于`NSArray`只能存储对象，我们需要将我们的值转换为`NSNumber`，当我们评估成员时，需要显式检查`intValue`对象。像 Java 一样，我们选择不创建一个单独的交换方法，而是通过引用传递值。否则，实现方式。

**Swift**

```py
    open func selectionSort( values: inout [Int]) 
    { 
        if (values.count <= 1) 
        { 
            return 
        } 

        var minIndex: Int 
        for i in 0..<values.count 
        { 
            minIndex = i 
            for j in i+1..<values.count 
            { 
                if (values[j] < values[minIndex]) 
                { 
                    minIndex = j 
                } 
            } 

            swap(x: &values[minIndex], y: &values[i]) 
        } 
    } 

    open func swap( x: inout Int, y: inout Int) 
    { 
        let t: Int = x 
        x = y 
        y = t 
    } 

```

Swift 不允许使用 C 样式的`for`循环，因此我们的方法必须使用 Swift 3.0 的等效方法。此外，由于 Swift 将数组视为`struct`实现而不是类实现，因此`values`参数不能简单地通过引用传递。因此，我们的 Swift 实现包括在`values`参数上使用`inout`修饰符。否则，功能与其前辈基本相同。此规则也适用于我们的`swap(x: inout Int, y: inout Int)`方法，该方法用于在排序过程中交换值。

# 插入排序

**插入排序**是一个非常简单的算法，它查看集合中的一个对象，并将其键与它之前的键进行比较。您可以将这个过程想象成我们中多少人按顺序排列一副扑克牌，逐个从左到右按升序移除和插入卡片。

例如，考虑按升序对集合进行排序的情况。插入排序算法将检查索引*i*处的对象，并确定其键值是否低于或优先于索引*i - 1*处的对象。如果是这样，索引*i*处的对象将被移除并插入到*i - 1*处。此时，函数将重复并继续以这种方式循环，直到*i - 1*处的对象键值不低于*i*处的对象键值。

给定以下值集：

*S = {50, 25, 73, 21, 3}*

我们将开始检查列表的*i = 1*。我们这样做是因为在*i = 0*处，*i - 1*是一个不存在的值，需要特殊处理。

由于 25 小于 50，因此它被移除并重新插入到*i = 0*的位置。由于我们处于索引 0，因此没有东西可以检查 25 左侧的，所以这次迭代完成：

*S = {25, 50, 73, 21, 3}*

接下来我们检查*i = 2*。由于 73 不小于 50，因此这个值不需要移动。由于我们已经将*i = 2*左侧的所有东西都排序好了，所以这次迭代立即完成。在*i = 3*处，值 21 小于 73，因此它被移除并重新插入到*i = 2*。再次检查，21 小于 50，所以值 21 被移除并重新插入到索引 1。最后，21 小于 25，所以值 21 被移除并重新插入到*i = 0*。由于我们现在处于索引 0，因此没有东西可以检查 21 左侧的，所以这次迭代完成：

*S = {21, 25, 50, 73, 3}*

最后，我们到达列表的*i = 4*，即列表的末尾。由于 3 小于 21，因此值 3 被移除并重新插入到*i = 3*。接下来，3 小于 73，所以值 3 被移除并重新插入到*i = 2*。在*i = 2*处，3 小于 50，所以值 3 被移除并重新插入到*i = 1*。在*i = 1*处，3 小于 25，所以值 3 被移除并重新插入到*i = 0*。由于我们现在处于索引 0，因此没有东西可以检查 3 左侧的，所以这次迭代和我们的排序函数都完成了：

*S = {3, 21, 25, 50, 73}*

如您所见，这个算法简单，但对于较大的对象或值列表来说可能成本较高。插入排序在最坏情况和平均情况下的复杂度都是**O(n²**)。然而，与选择排序不同，当对先前已排序的列表进行排序时，插入排序的效率有所提高。因此，它具有最佳复杂度**O(n**)，这使得该算法比选择排序略好一些。

**C#**

```py
    public void InsertionSort(int[] values) 
    { 
      if (values.Length <= 1) 
        return; 

      int j, value; 
      for (int i = 1; i < values.Length; i++) 
      { 
        value = values[i]; 
        j = i - 1; 

        while (j >= 0 && values[j] > value) 
        { 
          values[j + 1] = values[j]; 
          j = j - 1; 
        } 
        values[j + 1] = value; 
      }   
    } 

```

我们对`InsertionSort`方法的每个实现都首先确认`values`数组至少有两个成员。如果没有，则方法返回，因为没有足够的成员进行排序。否则，声明两个名为`j`和`value`的整数变量。接下来创建一个`for`循环，遍历集合的成员。索引`i`用于跟踪最后排序成员的位置。在这个`for`循环中，`value`被分配给最后排序的成员，而`j`用于跟踪当前迭代中未排序成员的位置。在我们的`while`循环中，`value`被分配给索引`j`处的成员，而`j`用于跟踪当前迭代中未排序成员的位置。我们的`while`循环将继续，直到`j`等于`0`且索引`j`处的值大于索引`i`处的值。在`while`循环的每次迭代中，我们将位置`j`处的成员与位置`j + 1`处的成员交换，然后循环将`j`的值减 1，以便在集合中回溯。最后一步是将存储在`value`中的成员设置在位置`j + 1`。

**Java**

```py
    public void insertionSort(int[] values) 
    { 
        if (values.length <= 1) 
            return; 

        int j, value; 
        for (int i = 1; i < values.length; i++) 
        { 
            value = values[i]; 
            j = i - 1; 

            while (j >= 0 && values[j] > value) 
            { 
                values[j + 1] = values[j]; 
                j = j - 1; 
            } 
            values[j + 1] = value; 
        }   
    } 

```

Java 实现的设计几乎与 C#实现相同，只是数组`length`函数的名称不同。

**Objective-C**

```py
    -(void)insertionSort:(NSMutableArray<NSNumber*>*)values 
    { 
        if ([values count] <= 1) 
            return; 

        NSInteger j, value; 
        for (int i = 1; i < [values count]; i++) 
        { 
            value = [values[i] intValue]; 
            j = i - 1; 

            while (j >= 0 && [values[j] intValue] > value) 
            { 
                values[j + 1] = values[j]; 
                j = j - 1; 
            } 
            values[j + 1] = [NSNumber numberWithInteger:value]; 
        } 
    } 

```

由于`NSArray`只能存储对象，我们需要将我们的值转换为`NSNumber`变量，并且在评估成员时需要显式检查`intValue`变量。否则，此实现与 C#或 Java 实现基本相同。

**Swift**

```py
    open func insertionSort( values: inout [Int]) 
    { 
        if (values.count <= 1) 
        { 
            return 
        } 

        var j, value: Int 
        for i in 1..<values.count 
        { 
            value = values[i]; 
            j = i - 1; 

            while (j >= 0 && values[j] > value) 
            { 
                values[j + 1] = values[j]; 
                j = j - 1; 
            } 
            values[j + 1] = value; 
        } 
    } 

```

Swift 不允许使用 C 风格的`for`循环，因此我们的方法必须使用 Swift 3.0 的等效方法。此外，由于 Swift 将数组视为结构体实现而不是类实现，因此`values`参数不能简单地通过引用传递。因此，我们的 Swift 实现包括在`values`参数上使用`inout`修饰符。否则，其功能与前辈基本相同。

# 冒泡排序

**冒泡排序**是另一种简单的算法，它通过遍历要排序的值或对象的列表，并比较相邻项或它们的键来确定它们是否处于错误的顺序。这个名字来源于无序项似乎会冒到列表顶部的样子。然而，一些开发者有时将其称为**下沉排序**，因为对象也可能看起来像是从列表中掉下来的。

总体来说，冒泡排序只是另一种低效的比较排序算法。然而，它确实具有其他比较排序算法所没有的一个显著优点，那就是：**内在地确定列表是否已排序**。冒泡排序通过不在之前迭代中已排序的对象上执行比较，并在集合被证明有序后停止，来实现这一点。

例如，考虑按升序对集合进行排序的情况。冒泡排序算法将检查索引 *i* 处的对象，并确定其键值是否低于或优先级低于索引 *i + 1* 处的对象，如果是这样，则交换这两个对象。

给定以下值集：

*S = {50, 25, 73, 21, 3}*

冒泡排序算法将比较*{i = 0, i = 1}*。由于 50 大于 25，所以这两个数被交换。接下来，方法比较*{i = 1, i = 2}*。在这种情况下，50 小于 73，所以没有变化。在*{i = 2, i = 3}*时，73 大于 21，所以它们被交换。最后，在*{i = 3, i = 4}*时，73 大于 3，所以它们也被交换。在我们的第一次迭代之后，我们的集合现在看起来是这样的：

*S = {25, 50, 21, 3, 73}*

让我们检查另一个迭代。在这个迭代中，我们的算法将首先比较*{i = 0, i = 1}*)，由于 25 小于 50，所以没有变化。接下来，我们检查*{i = 1, i = 2}*)。由于 50 大于 21，所以这两个数被交换。在*{i = 2, i = 3}*时，50 大于 3，所以这两个数被交换。由于在之前的迭代中*i = 4*已被排序，循环停止并重置到*i = 0*以进行下一次迭代。在第二次迭代之后，我们的集合看起来是这样的：

*S = {25, 21, 3, 50, 73}*

这表明通过集合的迭代包括*n - j*次比较，其中*n*是集合中项目数，*j*是当前迭代计数。因此，每次迭代后，冒泡排序都会变得*稍微*更有效率。此外，一旦集合被证明已排序，迭代就会完全停止。尽管冒泡排序的最坏情况和平均情况复杂度为**O**(*n²*)，但将排序限制为未排序的对象的能力为算法提供了**O**(*n*)的最佳情况复杂度，这使得这种方法略优于选择排序，但与插入排序大致相等。在列表已经排序的某些情况下，冒泡排序也比**快速排序**（我们稍后讨论）稍微有效率。然而，冒泡排序仍然是一个非常低效的算法，不适合除了小型对象集合之外的所有情况。

**C#**

```py
    public void BubbleSort(int[] values) 
    { 
      bool swapped; 
      for (int i = 0; i < values.Length - 1; i++) 
      { 
        swapped = false; 
        for (int j = values.Length - 1; j > i; j--) 
        { 
          if (values[j] < values[j - 1]) 
          { 
            Swap(ref values[j], ref values[j - 1]); 
            swapped = true; 
          } 
        } 

        if (swapped == false) 
          break; 
      } 
    } 

```

我们对`BubbleSort`方法的每个实现都是从声明一个名为`swapped`的布尔值开始的。这个值对于优化的冒泡排序方法至关重要，因为它用于跟踪当前迭代过程中是否有任何对象被交换。如果为`true`，则不能保证列表已排序，因此至少还需要进行一次迭代。如果为`false`，则没有对象被交换，这意味着列表已排序，算法可以立即停止。

接下来，我们创建一个`for`循环，遍历集合的成员。这个循环有效地跟踪我们的当前迭代。在这个循环内部，我们立即将`swapped`变量设置为`false`，然后创建另一个内部循环，它通过集合向后移动，对成对的对象进行比较。如果成对的两个对象被判定为顺序错误，`BubbleSort()`方法调用在选择排序讨论中检查的相同`swap()`方法，并将`swapped`改为`true`。否则，执行继续到`j`的下一个迭代。一旦内部循环完成，方法检查`swapped`变量以确定是否有对象被排序。如果为`false`，则执行继续到`i`的下一个迭代。否则，方法跳出外部循环，执行结束。

**Java**

```py
    public void bubbleSort(int[] values) 
    { 
        boolean swapped; 
        for (int i = 0; i < values.length - 1; i++) 
        { 
            swapped = false; 
            for (int j = values.length -1; j > i; j--) 
            { 
                if (values[j] < values[j - 1]) 
                { 
                    int temp = values[j]; 
                    values[j] = values[j - 1]; 
                    values[j - 1] = temp; 
                    swapped = true; 
                } 
            } 

            if (swapped == false) 
                break; 
        } 
    } 

```

Java 实现的设计几乎与 C#实现相同，只是数组`length`函数的名称不同。然而，Java 根本不支持通过引用传递原始数据类型。尽管可以通过将原始数据类型传递给可变包装类的实例来模拟这种行为，但大多数开发者都认为这是一个糟糕的想法。相反，我们的 Java 实现直接在`for`循环内部执行交换。

**Objective-C**

```py
    -(void)bubbleSortArray:(NSMutableArray<NSNumber*>*)values 
    { 
        bool swapped; 
        for (NSInteger i = 0; i < [values count] - 1; i++) 
        { 
            swapped = false; 
            for (NSInteger j = [values count] - 1; j > i; j--) 
            { 
                if (values[j] < values[j - 1]) 
                { 
                    NSInteger temp = [values[j] intValue]; 
                    values[j] = values[j - 1]; 
                    values[j - 1] = [NSNumber numberWithInteger:temp]; 
                    swapped = true; 
                } 
            } 

            if (swapped == false) 
                break; 
        } 
    } 

```

由于`NSArray`变量只能存储对象，我们需要将我们的值转换为`NSNumber`，在评估成员时需要显式检查`intValue`。与 Java 类似，我们选择不创建单独的交换方法，并通过引用传递值。否则，此实现与 C#或 Java 实现基本相同。

**Swift**

```py
    open func bubbleSort( values: inout [Int]) 
    { 
        var swapped: Bool     
        for i in 0..<values.count - 1 
        { 
            swapped = false 
            for j in ((i + 1)..<values.count).reversed() 
            { 
                if (values[j] < values[j - 1]) 
                { 
                    swap(x: &values[j], y: &values[j - 1]) 
                    swapped = true 
                }  
            } 

            if (swapped == false) 
            { 
                break 
            } 
        } 
    } 

```

Swift 不允许使用 C 风格的`for`循环，因此我们的方法必须使用 Swift 3.0 的等效方法。此外，由于 Swift 将数组视为结构体实现而不是类实现，`values`参数不能简单地通过引用传递。因此，我们的 Swift 实现包括在`values`参数上的`inout`修饰符。否则，其功能与前辈基本相同。这个规则也适用于我们的`swap(x: inout Int, y: inout Int)`方法，该方法在排序过程中用于交换值。

# 快速排序

快速排序是被称为**分而治之**算法集合中的一员。分而治之算法通过递归地将一组对象分解成两个或更多的子集，直到每个子集足够简单，可以直接解决。在快速排序的情况下，算法选择一个称为**基准点**的元素，然后通过将其前的所有较小元素和其后的所有较大元素进行排序。在基准点前后移动元素是快速排序算法的主要组成部分，被称为**分区**。分区在越来越小的子集上递归重复，直到每个子集包含 0 或 1 个元素，此时集合是有序的。

在保持快速排序改进性能方面，选择正确的枢轴点至关重要。例如，选择列表中的最小或最大元素将导致**O**(*n²*)复杂度。尽管没有万无一失的方法来选择最佳枢轴，但你的设计可以采取以下四种基本方法：

+   总是选择集合中的**第一个**对象。

+   总是选择集合中的**中值**对象。

+   总是选择集合中的**最后一个**对象。

+   从集合中随机选择一个对象。

在以下示例中，我们将采取第三种方法，选择集合中的最后一个对象作为枢轴。

尽管快速排序算法的最坏情况复杂度与其他我们迄今为止检查的排序一样，为**O**(*n²*)，但它具有改进的平均和最佳情况复杂度**O**(*n* log(*n*))，这使得它平均而言比选择排序、插入排序和冒泡排序方法更好。

**C#**

```py
    public void QuickSort(int[] values, int low, int high) 
    { 
      if (low < high) 
      { 
        int index = Partition(values, low, high); 

        QuickSort(values, low, index -1); 
        QuickSort(values, index +1, high); 
      } 
    } 

    int Partition(int[] values, int low, int high) 
    { 
      int pivot = values[high]; 
      int i = (low - 1); 
      for (int j = low; j <= high -1; j++) 
      { 
        if (values[j] <= pivot) 
        { 
          i++; 

          Swap(ref values[i], ref values[j]); 
        } 
      } 

    i++; 
      Swap(ref values[i], ref values[high]); 
      return i; 
    } 

```

我们对`QuickSort`方法的每个实现都是从检查低索引是否小于高索引开始的。如果为`false`，子集为空或有单个项目，因此根据定义它是有序的，方法返回。如果为`true`，方法首先通过调用`Partition(int[] values, int low, int high)`方法确定子集下一次划分的`index`。接下来，基于`index`定义的上下子集上递归调用`QuickSort(int[] values, int low, int high)`方法。

这个算法真正的魔力发生在`Partition(int[] values, int low, int high)`方法中。在这里，定义了一个用于枢轴的`index`变量，在我们的例子中是集合中的最后一个对象。接下来，`i`被定义为`low`索引`-1`。然后，我们的算法从`low`到`high -1`遍历列表。在循环中，如果`i`处的值小于或等于枢轴，我们就增加`i`，这样我们就有了集合中第一个未排序对象的索引，然后我们将其与`j`处的对象交换，`j`处的对象小于枢轴。

一旦循环完成，我们就将`i`增加一次，因为`i + 1`是集合中第一个大于枢轴的对象，而`i + 1`之前的所有对象都小于枢轴。我们的方法交换`i`处的值和索引`high`处的枢轴对象，这样枢轴也被正确排序。最后，方法返回`i`，这是`QuickSort(int[] values, int low, int high)`方法的下一个断点索引。

**Java**

```py
    public void quickSort(int[] values, int low, int high) 
    { 
        if (low < high) 
        { 
            int index = partition(values, low, high); 

            quickSort(values, low, index - 1); 
            quickSort(values, index + 1, high); 
        } 
    } 

    int partition(int[] values, int low, int high) 
    { 
        int pivot = values[high]; 
        int i = (low - 1); 
        for (int j = low; j <= high - 1; j++) 
        { 
            if (values[j] <= pivot) 
            { 
                i++; 

                int temp = values[i]; 
                values[i] = values[j]; 
                values[j] = temp; 
            } 
        } 

        i++; 
        int temp = values[i]; 
        values[i] = values[high]; 
        values[high] = temp; 

        return i;  
    } 

```

Java 实现的设计几乎与 C#实现相同，只是数组`length`函数的名称不同。然而，Java 根本不支持通过引用传递原始数据。尽管可以通过将原始数据传递给可变包装类的实例来模拟这种行为，但大多数开发者都认为这是一个坏主意。相反，我们的 Java 实现直接在`for`循环内和该方法本身执行交换操作。

**Objective-C**

```py
    -(void)quickSortArray:(NSMutableArray<NSNumber*>*)values forLowIndex:(NSInteger)low andHighIndex:(NSInteger)high 
    { 
        if (low < high) 
        { 
            NSInteger index = [self partitionArray:values forLowIndex:low andHighIndex:high]; 
            [self quickSortArray:values forLowIndex:low andHighIndex:index - 1]; 
            [self quickSortArray:values forLowIndex:index + 1 andHighIndex:high]; 
        } 
    } 

    -(NSInteger)partitionArray:(NSMutableArray<NSNumber*>*)values forLowIndex:(NSInteger)low andHighIndex:(NSInteger)high 
    { 
        NSInteger pivot = [values[high] intValue]; 
        NSInteger i = (low - 1); 
        for (NSInteger j = low; j <= high - 1; j++) 
        { 
            if ([values[j] intValue] <= pivot) 
            { 
                i++; 

                NSInteger temp = [values[i] intValue]; 
                values[i] = values[j]; 
                values[j] = [NSNumber numberWithInteger:temp]; 
            } 
        } 

        i++; 
        NSInteger temp = [values[i] intValue]; 
        values[i] = values[high]; 
        values[high] = [NSNumber numberWithInteger:temp]; 

        return i; 
    } 

```

由于 `NSArray` 变量只能存储对象，我们需要将我们的值转换为 `NSNumber`，在评估成员时需要显式检查 `intValue`。像 Java 一样，我们选择不创建单独的交换方法并通过引用传递值。否则，这个实现与 C# 或 Java 实现基本相同。

**Swift**

```py
    open func quickSort( values: inout [Int], low: Int, high: Int) 
    { 
        if (low < high) 
        { 
            let index: Int = partition( values: &values, low: low, high: high) 

            quickSort( values: &values, low: low, high: index - 1) 
            quickSort( values: &values, low: index + 1, high: high) 
        } 
    } 

    func partition( values: inout [Int], low: Int, high: Int) -> Int 
    { 
        let pivot: Int = values[high] 
        var i: Int = (low - 1) 
        var j: Int = low 

        while j <= (high - 1) 
        { 
            if (values[j] <= pivot) 
            { 
                i += 1 
                swap(x: &values[i], y: &values[j]) 
            } 

            j += 1 
        } 

        i += 1 
        swap(x: &values[i], y: &values[high]) 

        return i; 
    } 

```

Swift 不允许使用 C 风格的 `for` 循环，因此我们的 Swift 3.0 版本的 `mergeSort:` 方法在这方面有些受限。因此，我们将使用 `while` 循环来替换 `for` 循环。这样，我们定义 `j` 为 `low` 索引值，并在 `while` 循环的每次迭代中显式地增加 `j`。另外，由于 Swift 将数组视为结构体实现而不是类实现，`values` 参数不能简单地通过引用传递。因此，我们的 Swift 实现包括在 `values` 参数上使用 `inout` 装饰器。否则，其功能与前辈们基本相同。这个规则也适用于我们的 `swap(x: inout Int, y: inout Int)` 方法，该方法用于在排序过程中交换值。

# 归并排序

**归并排序**是分治算法的另一种流行版本。它是一个非常高效、通用的排序算法。算法的命名来源于它将集合分成两半，递归地对每个半集合进行排序，然后将两个排序后的半集合合并在一起。集合的每个半部分都会反复分成一半，直到只剩下一个对象，此时根据定义进行排序。在合并每个排序后的半部分时，算法会比较对象以确定每个子集的放置位置。

就分治算法而言，归并排序是最有效的算法之一。该算法的最坏、平均和最佳情况复杂度为 **O**(*n* log(*n*))，即使在最坏情况下也优于快速排序。

**C#**

```py
    public void MergeSort(int[] values, int left, int right) 
    { 
      if (left == right) 
        return; 

      if (left < right) 
      { 
        int middle = (left + right) / 2; 

        MergeSort(values, left, middle); 
        MergeSort(values, middle + 1, right); 

        int[] temp = new int[values.Length]; 
        for (int n = left; n <= right; n++) 
        { 
          temp[n] = values[n]; 
        } 

        int index1 = left;  
        int index2 = middle + 1; 
        for (int n = left; n <= right; n++) 
        { 
          if (index1 == middle + 1) 
          { 
            values[n] = temp[index2++]; 
          } 
          else if (index2 > right) 
          { 
            values[n] = temp[index1++]; 
          } 
          else if (temp[index1] < temp[index2]) 
          { 
            values[n] = temp[index1++]; 
          } 
          else 
          { 
            values[n] = temp[index2++]; 
          } 
        } 
      }     
    } 

```

在我们 `MergeSort` 方法的每个实现中，`left` 和 `right` 参数定义了整体 `values` 数组中集合的开始和结束位置。当方法最初被调用时，`left` 参数应该是 0，而 `right` 参数应该是 `values` 集合中最后一个对象的索引。

该方法首先检查 `left` 索引是否等于 `right` 索引。如果是 `true`，子集为空或只有一个项目，因此根据定义是有序的，方法返回。否则，方法检查 `left` 索引是否小于 `right` 索引。如果是 `false`，方法返回，因为该子集已经是有序的。

如果为`true`，方法执行将真正开始。首先，方法确定当前子集的中点，因为这将被用来将子集分成两个新的 halves。声明并定义`middle`变量，通过将`left`和`right`相加然后除以 2。接下来，通过传递值数组和使用`left`、`right`和`middle`作为指南，递归地调用每个 halves 的`MergeSort(int[] values, int left, int right)`方法。随后，方法创建一个名为`temp`的新数组，其大小与`values`相同，并仅填充与当前子集相关的索引。一旦`temp`数组被填充，方法创建两个名为`index1`和`index2`的`int`变量，它们代表当前子集内两个 halves 的起始点。

最后，我们到达`for`循环，它从开始到结束（`left`到`right`）遍历子集并对找到的值进行排序。每个`if`语句中的逻辑是显而易见的，但了解这些特定比较背后的推理是有帮助的：

+   第一次比较仅在左子集耗尽值时为`true`，此时将`values[n]`数组设置为`temp[index2]`的值。随后，使用后增量运算符，`index2`变量增加 1，将指针在右子集内向右移动一个索引。

+   第二次比较仅在右子集耗尽值时为`true`，此时将`values[n]`数组设置为`temp[index1]`的值。随后，使用后增量运算符，`index1`变量增加 1，将指针在左子集内向右移动一个索引。

+   第三次也是最后一次比较仅在左右子集都有尚未排序的值时才会评估。当`temp[index1]`数组中的值小于`temp[index2]`数组中的值时，此比较为`true`，此时将`values[n]`数组设置为`temp[index1]`。同样，随后，使用后增量运算符，`index1`变量增加 1，将指针在左子集内向右移动一个索引。

+   最后，当所有其他逻辑选项都无效时，默认行为假定`temp[index1]`数组中的值大于`temp[index2]`数组中的值，因此 else 块将`values[n]`数组中的值设置为`temp[index2]`。随后，使用后增量运算符，`index2`变量增加 1，将指针在右子集内向右移动一个索引。

**Java**

```py
    public void mergeSort(int[] values, int left, int right) 
    { 
        if (left == right) 
            return; 

        if (left < right) 
        { 
            int middle = (left + right) / 2; 

            mergeSort(values, left, middle); 
            mergeSort(values, middle + 1, right); 

            int[] temp = new int[values.length]; 
            for (int n = left; n <= right; n++) 
            { 
                temp[n] = values[n]; 
            } 

            int index1 = left; 
            int index2 = middle + 1; 
            for (int n = left; n <= right; n++) 
            { 
                if (index1 == middle + 1) 
                { 
                    values[n] = temp[index2++]; 
                } 
                else if (index2 > right) 
                { 
                    values[n] = temp[index1++]; 
                } 
                else if (temp[index1] < temp[index2]) 
                { 
                    values[n] = temp[index1++]; 
                } 
                else 
                { 
                    values[n] = temp[index2++]; 
                } 
            } 
        } 
    } 

```

Java 实现的设计几乎与 C#实现相同，只是数组`length`函数的名称不同。

**Objective-C**

```py
    -(void)mergeSort:(NSMutableArray*)values withLeftIndex:(NSInteger)left andRightIndex:(NSInteger)right 
    { 
        if (left == right) 
            return; 

        if (left < right) 
        { 
            NSInteger middle = (left + right) / 2; 

            [self mergeSort:values withLeftIndex:left andRightIndex:middle]; 
            [self mergeSort:values withLeftIndex:middle + 1 andRightIndex:right]; 
            NSMutableArray *temp = [NSMutableArray arrayWithArray:values]; 
            NSInteger index1 = left; 
            NSInteger index2 = middle + 1; 
            for (NSInteger n = left; n <= right; n++) 
            { 
                if (index1 == middle + 1) 
                { 
                    values[n] = temp[index2++]; 
                } 
                else if (index2 > right) 
                { 
                    values[n] = temp[index1++]; 
                } 
                else if (temp[index1] < temp[index2]) 
                { 
                    values[n] = temp[index1++]; 
                } 
                else 
                { 
                    values[n] = temp[index2++]; 
                } 
            } 
        } 
    } 

```

`mergeSort:withLeftIndex:andRightIndex:`的 Objective-C 实现与 C#和 Java 实现基本相同。

**Swift**

```py
    open func mergeSort( values: inout [Int], left: Int, right: Int) 
    { 
        if (values.count <= 1) 
        { 
            return 
        } 

        if (left == right) 
        { 
            return 
        } 

        if (left < right) 
        { 
            let middle: Int = (left + right) / 2 

            mergeSort(values: &values, left: left, right: middle) 
            mergeSort(values: &values, left: middle + 1, right: right) 

            var temp = values 

            var index1: Int = left 
            var index2: Int = middle + 1 
            for n in left...right 
            { 
                if (index1 == middle + 1) 
                { 
                    values[n] = temp[index2] 
                    index2 += 1 
                } 
                else if (index2 > right) 
                { 
                    values[n] = temp[index1] 
                    index1 += 1 
                } 
                else if (temp[index1] < temp[index2]) 
                { 
                    values[n] = temp[index1] 
                    index1 += 1 
                } 
                else 
                { 
                    values[n] = temp[index2] 
                    index2 += 1 
                } 
            } 
        } 
    } 

```

Swift 不允许使用 C 风格的`for`循环，因此我们的方法与 Swift 3.0 的等效方法在此情况下有些受限。由于 Swift 将数组视为结构体实现而不是类实现，`values`参数不能简单地通过引用传递。这对于这个归并排序实现来说并不一定是问题，因为每当方法递归调用时，整个`values`数组都会作为参数传递。然而，为了使该方法与其他在此讨论的算法更一致，并避免需要声明返回类型，此实现仍然在`values`参数上包含了`inout`修饰符。否则，其功能与前辈们基本相同。

# 桶排序

**桶排序**，也称为**箱排序**，是一种分布排序算法。分布排序是那些将原始值散布到任何中间结构中的算法，然后对这些中间结构进行排序、收集和合并到最终输出结构中的算法。需要注意的是，尽管桶排序被认为是分布排序，但大多数实现通常利用比较排序来对桶的内容进行排序。该算法通过在整个数组数组（称为**桶**）中分配值来排序值。元素根据其值和分配给每个桶的值范围进行分配。例如，如果一个桶接受从 5 到 10 的值范围，原始集合包括 3、5、7、9 和 11，那么值 5、7 和 9 将放入这个假设的桶中。

一旦所有值都分配到各自的桶中，然后通过递归调用桶排序算法再次对桶本身进行排序。最终，每个桶都被排序，然后排序结果被连接成一个完整的排序集合。

由于元素分配到桶的方式，桶排序可以比其他排序算法快得多，通常每个桶使用一个数组，其中值表示索引。尽管该算法仍然具有**O**(*n²*)的最坏情况复杂度，但平均和最佳情况复杂度仅为*O(n + k)*，其中*n*是原始数组中的元素数量，*k*是用于排序集合的总桶数。

**C#**

```py
    public void BucketSort(int[] values, int maxVal) 
    { 
      int[] bucket = new int[maxVal + 1]; 
      int num = values.Length; 
      int bucketNum = bucket.Length; 

      for (int i = 0; i < bucketNum; i++) 
      { 
        bucket[i] = 0; 
      } 

      for (int i = 0; i < num; i++) 
      { 
        bucket[values[i]]++; 
      } 

      int pos = 0; 
      for (int i = 0; i < bucketNum; i++) 
      { 
        for (int j = 0; j < bucket[i]; j++) 
        { 
          values[pos++] = i; 
        } 
      } 
    } 

```

我们对`BucketSort`方法的每个实现都是从根据`values`数组中的元素总数创建空桶开始的。接下来，使用`for`循环将基础值`0`填充到桶中。这立即被第二个`for`循环所跟随，该循环将元素从`values`分配到各个桶中。最后，使用嵌套`for`循环对桶中的元素以及`values`数组本身进行排序。

**Java**

```py
    public void BucketSort(int[] values, int maxVal) 
    { 
        int[] bucket = new int[maxVal + 1]; 
        int num = values.length; 
        int bucketNum = bucket.length; 

        for (int i = 0; i < bucketNum; i++) 
        { 
            bucket[i] = 0; 
        } 

        for (int i = 0; i < num; i++) 
        { 
            bucket[values[i]]++; 
        } 

        int pos = 0; 
        for (int i = 0; i < bucketNum; i++) 
        { 
            for (int j = 0; j < bucket[i]; j++) 
            { 
                values[pos++] = i; 
            } 
        } 
    } 

```

Java 实现的设计几乎与 C#实现相同，只是数组的`length`函数名称不同。

**Objective-C**

```py
    -(void)bucketSortArray:(NSMutableArray<NSNumber*>*)values withMaxValue:(NSInteger)maxValue 
    { 
        NSMutableArray<NSNumber*>*bucket = [NSMutableArray array]; 
        NSInteger num = [values count]; 
        NSInteger bucketNum = maxValue + 1; 

        for (int i = 0; i < bucketNum; i++) 
        { 
            [bucket insertObject:[NSNumber numberWithInteger:0] atIndex:i]; 
        } 

        for (int i = 0; i < num; i++) 
        { 
            NSInteger value=[bucket[[values[i] intValue]] intValue]+ 1; 
            bucket[[values[i] intValue]] = [NSNumber numberWithInteger:value]; 
        } 

        int pos = 0; 

        for (int i = 0; i < bucketNum; i++) 
        { 
            for (int j = 0; j < [bucket[i] intValue]; j++) 
            { 
                values[pos++] = [NSNumber numberWithInteger:i]; 
            } 
        } 
    } 

```

由于`NSArray`数组只能存储对象，我们需要将我们的值转换为`NSNumber`数组，并且在评估成员时需要显式检查`intValue`变量。否则，这种实现与 C#或 Java 实现的基本上是相同的。

**Swift**

```py
    open func bucketSort( values: inout [Int], maxVal: Int) 
    { 
        var bucket = [Int]() 
        let num: Int = values.count 
        let bucketNum: Int = bucket.count 

        for i in 0..<bucketNum 
        { 
            bucket[i] = 0 
        } 

        for i in 0..<num 
        { 
            bucket[values[i]] += 1 
        } 

        var pos: Int = 0 
        for i in 0..<bucketNum 
        { 
            for _ in 0..<bucket[i] 
            { 
                values[pos] = i 
                pos += 1 
            } 
        } 
    } 

```

Swift 不允许使用 C 风格的`for`循环，因此我们的方法必须使用 Swift 3.0 的等效方法。否则，其功能与其前辈基本相同。

# 摘要

在本章中，我们讨论了你在日常经验中可能会遇到的几种常见排序算法。我们首先介绍了几种比较排序，包括选择排序、插入排序和冒泡排序。我们指出，选择排序可能是你在现实生活中可能遇到的最不高效的排序算法，但这并不意味着它是完全学术性的。插入排序在某种程度上改进了选择排序，冒泡排序算法也是如此。接下来，我们考察了两种分而治之的排序算法，包括快速排序和归并排序。这两种方法都比比较排序更高效。最后，我们探索了一种常见且高效的分布排序，称为计数排序。计数排序是我们考察过的最有效率的算法，但它并不一定适合所有情况。
