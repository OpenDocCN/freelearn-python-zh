# 第十三章。搜索：找到你需要的东西

对你的集合进行排序可能会很昂贵，但通常这代表在创建集合后的单次成本。然而，在应用程序运行周期中，这种前期的时间和精力投入可以显著提高性能。即使添加新对象，当它被添加到已排序的集合中时，这个过程也要便宜得多。

当需要搜索你的集合以查找特定元素或值时，真正的性能提升才会到来。在本章中，我们将探讨排序集合如何根据你选择的搜索算法大大提高搜索时间。我们不会讨论你可以选择的所有搜索算法，但我们将检查三种最常见的算法：

+   线性搜索（顺序搜索）

+   二分搜索

+   跳转搜索

# 线性搜索

搜索，也称为**顺序**搜索，简单地说就是通过某种比较函数遍历一个集合，以定位匹配的元素或值。大多数线性搜索返回一个表示集合中匹配对象的索引的值，或者当对象未找到时，返回一些不可能的索引值，例如`-1`。这个搜索的替代版本可以返回对象本身，或者当对象未找到时返回`null`。

这是最简单的搜索模式，它具有**O**(*n*)的复杂度成本。这种复杂度在集合是无序的还是已经排序的情况下都是一致的。在非常小的集合中，线性搜索是完全可接受的，许多开发者每天都在使用它们。然而，当处理非常大的集合时，找到这种顺序搜索方法的替代方案通常是有益的。这尤其适用于处理非常复杂的对象列表，例如空间几何，其中搜索或分析可能是非常耗处理器的操作。

### 注意

本章中的每个代码示例都将通过操作中最基本的方法形式来检查搜索算法，这些方法与它们的父类分离。此外，在每个情况下，将要排序的对象集合将在类级别上定义，在下面展示的示例代码之外。同样，后续的对象实例化和这些集合的填充也将定义在示例代码之外。要查看完整的类示例，请使用伴随此文本的代码示例。

**C#**

线性搜索算法的第一个示例是在`LinearSearchIndex(int[] values, int key)`方法中。正如你所看到的，这个方法非常简单，几乎是自我解释的。这个实现有两个主要特点值得提及。首先，该方法接受`values`数组（值）和搜索`key`。其次，该方法返回任何匹配元素的索引`i`，或者如果搜索键未找到，则简单地返回`-1`。

```py
    public int LinearSearchIndex(int[] values, int key) 
    { 
        for (int i = 0; i < values.Length - 1; i++) 
        { 
            if (values[i] == key) 
            { 
                return i; 
            } 
        } 

        return -1; 
    } 

```

线性搜索的第二个例子几乎与第一个相同。然而，在`LinearSearchCustomer(Customer[] customers, int custId)`方法中，我们不是在搜索一个值，而是在搜索一个代表调用者想要检索的客户的键。请注意，现在的比较是在`Customer`对象的`customerId`字段上进行的；如果找到匹配项，则返回`customers[i]`处的`Customer`。如果没有找到匹配项，该方法返回`null`：

```py
    public Customer LinearSearchCustomer(Customer[] customers, int custId) 
    { 
        for (int i = 0; i < customers.Length - 1; i++) 
        { 
            if (customers[i].customerId == custId) 
            { 
                return customers[i]; 
            } 
        } 

        return null; 
    } 

```

**Java**

每个方法的 Java 实现的设计几乎与 C#实现相同，只是数组的`length`函数的名称不同。

```py
    public int linearSearchIndex(int[] values, int key) 
    { 
        for (int i = 0; i < values.length - 1; i++) 
        { 
            if (values[i] == key) 
            { 
                return i; 
            } 
        } 

        return -1; 
    } 

    public Customer linearSearchCustomer(Customer[] customers, int custId) 
    { 
        for (int i = 0; i < customers.length - 1; i++) 
        { 
            if (customers[i].customerId == custId) 
            { 
                return customers[i]; 
            } 
        } 

        return null; 
    } 

```

**Objective-C**

由于`NSArray`只能存储对象，我们需要将我们的值转换为`NSNumber`，在评估成员时，我们需要显式检查`intValue`。否则，这些实现与 C#或 Java 实现基本相同：

```py
    -(NSInteger)linearSearchArray:(NSMutableArray<NSNumber*>*)values byKey:(NSInteger) key 
    { 
        for (int i = 0; i < [values count] - 1; i++) 
        { 
            if ([values[i] intValue] == key) 
            { 
                return i; 
            } 
        } 

        return -1; 
    } 

    -(EDSCustomer*)linearSearchCustomers:(NSMutableArray<NSNumber*>*)customers byCustId:(NSInteger)custId 
    { 
        for (EDSCustomer *c in customers) 
        { 
            if (c.customerId == custId) 
            { 
                return c; 
            } 
        } 
        return nil; 
    } 

```

**Swift**

Swift 不允许 C 样式的`for`循环，因此我们的方法必须使用 Swift 3.0 的等效方法。此外，Swift 不允许方法返回`nil`，除非返回类型明确声明为可选，因此`linearSearchCustomer(customers: [Customer], custId: Int)`方法的返回类型为`Customer?`。否则，其功能与其前辈基本相同：

```py
    open func linearSearhIndex( values: [Int], key: Int) -> Int 
    { 
        for i in 0..<values.count 
        { 
            if (values[i] == key) 
            { 
                return i 
            } 
        } 

        return -1 
    } 

    open func linearSearchCustomer( customers: [Customer], custId: Int) -> Customer? 
    { 
        for i in 0..<customers.count 
        { 
            if (customers[i].custId == custId) 
            { 
                return customers[i] 
            } 
        } 
        return nil 
    } 

```

# 二分搜索

当处理未排序的集合时，顺序搜索可能是最合理的方法。然而，当与排序集合一起工作时，有更好的方法来找到与搜索键匹配的方法。一个替代方案是二分搜索。二分搜索通常实现为一个递归函数，其工作原理是反复将集合分成两半，并搜索越来越小的集合块，直到找到匹配项或搜索耗尽剩余选项并返回空。

例如，给定以下有序值集合：

*S = {8, 19, 23, 50, 75, 103, 121, 143, 201}*

使用线性搜索查找值`143`将具有**O**(8)的复杂度成本，因为`143`在我们的集合中位于索引 7（位置 8）。然而，二分搜索可以利用集合的排序特性来提高这种复杂度成本。

我们知道集合由九个元素组成，因此二分搜索将首先检查索引 5 的中值元素，并将其与键值`143`进行比较。由于*i[5] = 75*，这小于`143`，因此集合被分割，可能的匹配项的范围仅包括上半部分，留下：

*S = {103, 121, 143, 201}*

对于四个元素，中值元素是位置二的元素。位置*i[2] = 121*，这小于`143`，因此集合被分割，可能的匹配项的范围仅包括上半部分，留下：

*S = {143, 201}*

使用两个元素时，中值元素是位置一的元素。由于*i[1] = 143*，我们找到了匹配项，可以返回该值。这种搜索只花费了**O**(3)的时间，几乎提高了 67%的线性搜索时间。尽管个别结果可能会有所不同，但当集合已排序时，二分搜索模式始终比线性搜索更有效。这是在应用程序开始使用它们提供的数据之前花时间对集合进行排序的强有力的理由：

**C#**

`BinarySort(int[] values, int left, int right, int key)`首先检查`right`索引是否大于或等于`left`索引。如果不是，则指定范围内的没有元素，分析已经用尽，因此方法返回`-1`。我们稍后将检查原因。否则，方法执行继续，因为定义的范围内至少有一个对象。

接下来，方法检查`middle`索引处的值是否与我们的`key`匹配。如果是`true`，则返回`middle`索引。否则，方法检查`middle`索引处的值是否大于`key`值。如果是`true`，则以选择当前元素范围下半部分的边界递归调用`BinarySort(int[] values, int left, int right, int key)`。否则，`middle`索引处的值小于`key`，因此以选择当前元素范围上半部分的边界递归调用`BinarySort(int[] values, int left, int right, int key)`：

```py
    public int BinarySearch(int[] values, int left, int right, int key) 
    { 
        if (right >= left) 
        { 
            int middle = left + (right - left) / 2; 

            if (values[middle] == key) 
            { 
                return middle; 
            } 
            else if (values[middle] > key) 
            { 
                return BinarySearch(values, left, middle - 1, key); 
            } 

            return BinarySearch(values, middle + 1, right, key); 
        } 

        return -1; 
    } 

```

**Java**

除了`binarySearch(int[] values, int left, int right, int key)`这个名称外，Java 实现的设计与 C#实现相同：

```py
    public int binarySearch(int[] values, int left, int right, int key) 
    { 
        if (right >= left) 
        { 
            int mid = left + (right - left) / 2; 

            if (values[mid] == key) 
            { 
                return mid; 
            } 
            else if (values[mid] > key) 
            { 
                return binarySearch(values, left, mid - 1, key); 
            } 

            return binarySearch(values, mid + 1, right, key); 
        } 

        return -1; 
    } 

```

**Objective-C**

由于`NSArray`只能存储对象，我们需要将我们的值转换为`NSNumber`，并且在评估成员时，我们需要显式检查`intValue`。否则，这些实现与 C#或 Java 实现基本相同：

```py
    -(NSInteger)binarySearchArray:(NSMutableArray<NSNumber*>*)values withLeftIndex:(NSInteger)left 
rightIndex:(NSInteger)right
andKey:(NSInteger)key 
    { 
        if (right >= left) 
        { 
            NSInteger mid = left + (right - left) / 2; 

            if ([values[mid] intValue] == key) 
            { 
                return mid; 
            } 
            else if ([values[mid] intValue] > key) 
            { 
                return [self binarySearchArray:values withLeftIndex:left rightIndex:mid - 1 andKey:key]; 
            } 

            return [self binarySearchArray:values withLeftIndex:mid + 1 rightIndex:right andKey:key]; 
        } 

        return -1; 
    }  

```

**Swift**

基本上，Swift 实现与其前辈相同：

```py
    open func binarySearch( values: [Int], left: Int, right: Int, key: Int) -> Int 
    { 
        if (right >= left) 
        { 
            let mid: Int = left + (right - left) / 2 

            if (values[mid] == key) 
            { 
                return mid 
            } 
            else if (values[mid] > key) 
            { 
                return binarySearch(values: values, left: left, right: mid - 1, key: key) 
            } 

            return binarySearch(values: values, left: mid + 1, right: right, key: key) 
        } 

        return -1 
    }  

```

# 跳跃搜索

另一种可以改善排序数组性能的搜索算法是**跳跃搜索**。跳跃搜索在某种程度上与线性搜索和二分搜索算法相似，因为它从集合的第一个块开始，从左到右搜索元素块，并且在每次跳跃时，算法将搜索键值与当前步骤的元素值进行比较。如果算法确定键可能存在于当前元素子集中，下一步（无意中开玩笑）就是检查当前子集中的每个元素，以确定它是否小于键。

一旦找到一个不小于键的元素，该元素就会与键进行比较。如果元素等于键，则返回；否则，它大于键，这意味着键不存在于集合中。

跳跃长度 *m* 不是一个任意值，而是基于集合长度通过公式 *m =* √*n* 计算得出的，其中 *n* 是集合中元素的总数。跳跃搜索首先检查第一个块或子集的最后一个对象的值。

例如，让我们在以下有序值集合中搜索值 *143*：

*S = {8, 19, 23, 50, 75, 103, 121, 143, 201}*

由于我们的集合包含九个元素，*m = √* *n* 给我们一个值为 3 的结果。由于 *i[2] = 23*，并且这个值小于 *143*，算法跳到下一个块。接下来，*i[4] = 103*，这也小于 *143*，所以这个子集被排除。最后，*i[8] = 201*。由于 *201* 大于 *143*，键可能存在于第三个子集中：

*S[3] = {121, 143, 201}*

接下来，算法检查这个子集中的每个元素，以确定它是否小于 *143*。并且 *i[6] = 121*，所以算法继续检查。另外，*i[7] = 143*，这并不小于 *143*，所以执行继续到最后一步。由于 *i[7] = 143*，我们找到了与我们的键匹配的值，并且可以返回 *i* 的值。这次搜索的成本是 **O**(5)，这比线性搜索产生的 **O**(7) 略好，但比我们找到的 **O**(3) 成本略差。然而，对于更大的数据集，当集合已排序时，跳跃搜索在大多数情况下比线性搜索和二分搜索更有效。

再次强调，对集合进行排序确实在时间和性能上代表了一些前期成本，但你的应用程序运行周期中的回报远远超过了这些努力。

**C#**

我们对 `BubbleSort` 方法的每个实现都从声明三个 `int` 变量开始，以跟踪集合的大小、步长和先前评估的索引。随后，一个 `while` 循环使用 `prev` 和 `step` 值来定义和搜索集合的子集，以确定 `key` 可能存在的范围。如果没有找到可接受的子集，该方法返回 `-1`，表示 `key` 不能存在于这个集合中。否则，`prev` 和 `step` 的值标识了 `key` 可能存在的子集。

下一个 `while` 循环检查子集中每个元素，以确定它是否小于 `key`。如果没有找到可接受的元素，该方法返回 `-1`，表示 `key` 不能存在于这个集合中。否则，`prev` 的值标识了 `key` 在子集中可能的最佳匹配。

最后，将 `prev` 位置的元素与 `key` 进行比较。如果两个值匹配，则返回 `prev`。否则，我们到达执行结束，返回 `-1`：

```py
    public int JumpSearch(int[] values, int key) 
    { 
        int n = values.Length; 
        int step = (int)Math.Sqrt(n); 
        int prev = 0; 

        while (values[Math.Min(step, n) - 1] < key) 
        { 
            prev = step; 
            step += (int)Math.Floor(Math.Sqrt(n)); 
            if (prev >= n) 
            { 
                return -1; 
            } 
        } 

        while (values[prev] < key) 
        { 
            prev++; 
            if (prev == Math.Min(step, n)) 
            { 
                return -1; 
            } 
        } 

        if (values[prev] == key) 
        { 
            return prev; 
        } 

        return -1; 
    } 

```

**Java**

每个方法的 Java 实现在设计上几乎与 C# 实现相同，只是数组 `length` 函数的名称不同。

```py
    public int jumpSearch(int[] values, int key) 
    { 
        int n = values.length; 
        int step = (int)Math.sqrt(n); 
        int prev = 0; 

        while (values[Math.min(step, n) - 1] < key) 
        { 
            prev = step; 
            step += (int)Math.floor(Math.sqrt(n)); 
            if (prev >= n) 
            { 
                return -1; 
            } 
        } 

        while (values[prev] < key) 
        { 
            prev++; 
            if (prev == Math.min(step, n)) 
            { 
                return -1; 
            } 
        } 

        if (values[prev] == key) 
        { 
            return prev; 
        } 

        return -1; 
    } 

```

**Objective-C**

由于 `NSArray` 只能存储对象，我们需要将我们的值转换为 `NSNumber`，当我们评估成员时，需要显式检查 `intValue`。否则，这个实现本质上与 C# 或 Java 实现相同。

```py
    -(NSInteger)jumpSearchArray:(NSMutableArray<NSNumber*>*)values forKey: (NSInteger)key 
    { 
        NSInteger n = [values count]; 
        NSInteger step = sqrt(n); 

        NSInteger prev = 0; 
        while ([values[(int)fmin(step, n)-1] intValue] < key) 
        { 
            prev = step; 
            step += floor(sqrt(n)); 
            if (prev >= n) 
            { 
                return -1; 
            } 
        } 

        while ([values[prev] intValue] < key) 
        { 
            prev++; 
            if (prev == fmin(step, n)) 
            { 
                return -1; 
            } 
        } 

        if ([values[prev] intValue] == key) 
        { 
            return prev; 
        } 

        return -1; 
    } 

```

**Swift**

除了从 `sqrt()` 和 `floor()` 方法返回的值需要额外的类型转换外，其功能本质上与前辈相同：

```py
    open func jumpSearch( values: [Int], key: Int) -> Int 
    { 
        let n: Int = values.count 
        var step: Int = Int(sqrt(Double(n))) 

        var prev: Int = 0 

        while values[min(step, n) - 1] < key 
        { 
            prev = step 
            step = step + Int(floor(sqrt(Double(n)))) 
            if (prev >= n) 
            { 
                return -1 
            } 
        } 

        while (values[prev] < key) 
        { 
            prev = prev + 1 
            if (prev == min(step, n)) 
            { 
                return -1 
            } 
        } 

        if (values[prev] == key) 
        { 
            return prev 
        } 

        return -1 
    } 

```

# 摘要

在本章中，我们探讨了几个搜索算法。首先，我们研究了线性搜索，或顺序搜索。线性搜索几乎不能算是一个算法，因为你的代码只是简单地从左到右遍历集合中的元素，直到找到匹配项。当处理非常小的集合或未排序的集合时，这种方法是有用的，如果其他原因的话，那就是从开发角度来看易于实现。然而，当处理大型排序数据集时，有更好的替代方案。

接下来，我们研究了二分搜索算法。二分搜索算法本质上是通过将集合划分为更小的子集来分而治之，直到找到匹配项或可能的匹配项列表耗尽。与线性搜索的 **O**(*n*) 复杂度成本相比，二分搜索模式具有显著改进的 **O**(*log(n)*) 复杂度成本。然而，在运行二分搜索之前，确保集合被正确排序是绝对必要的，否则结果将毫无意义。

最后，我们研究了跳跃搜索。跳跃搜索通过顺序检查集合的子集来实现，每个子集的长度为 √*n*，其中 *n* 是集合中元素的总数。尽管实现起来稍微复杂一些，并且最坏情况下的复杂度为 **O**(*n*)，但跳跃搜索的平均成本复杂度显著提高，为 **O**(√*n*)，其中 *n* 是集合中元素的总数。
