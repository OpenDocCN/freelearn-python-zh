# 第十章。堆：有序树

**堆**是树数据结构的一个特殊类别，它们根据树节点的值或与每个节点关联的键进行排序。这种排序在最小堆中是升序的，意味着根节点的值或优先级小于其子节点，或者在最大堆中是降序的，意味着根节点的值或优先级大于其子节点。请注意，堆数据结构不应与计算机系统的堆内存混淆，后者通常用于系统动态分配的内存。

在本章中，我们将涵盖以下主题：

+   定义堆数据结构

+   数组实现

+   创建堆

+   常见操作

# 堆实现

与树类似，堆通常使用链表或链节点，或数组来实现。由于我们在第九章（第九章

二叉堆是一种树结构，其中树的所有层级都被完全填充，除了最后一层或最深层。在最后一层的情况下，节点从左到右填充，直到该层填满。如图所示，在基于数组的实现中，每个父节点有两个子节点，它们位于 *2i + 1* 和 *2i + 2*，其中 i 是父节点的索引，集合的第一个节点位于索引 0。

### 注意

不同的实现跳过了数组的 0 索引，以简化给定索引查找子节点和父节点的算术。在这个设计中，任何给定索引 i 的子节点位于 *2i* 和 2i + 1。

# 堆操作

并非所有堆数据结构的实现都公开相同的操作方法。然而，更常见的操作应该可用，或者根据开发者的需要提供。

+   **Insert**：Insert 操作向堆中添加一个新节点。此操作必须重新排序堆，以确保新添加的节点保持堆属性。此操作的操作成本为 **O**(*log n*)。

+   **FindMax**：FindMax 操作与最大堆同义，并返回集合中的最大值或最高优先级对象。在基于数组的实现中，这通常是索引 0 或索引 1 的对象，具体取决于设计。这相当于栈或队列中的 *peek* 操作，当使用堆实现优先队列时非常重要。此操作的操作成本为 **O**(*1*)。

+   **FindMin**: FindMin 操作与最小堆相关，并返回集合中最小值或最低优先级的对象。在基于数组的实现中，这通常是索引为 0 或 1 的对象，具体取决于设计。此操作的操作成本为**O(1**)。

+   **ExtractMax**: ExtractMax 操作与最大堆相关，既返回集合中最大值或最高优先级的对象，又将其从集合中移除。这相当于在栈或队列结构中的*pop*操作。与 FindMax 操作类似，这通常是索引为 0 或 1 的对象，具体取决于设计。此操作还将重新排序堆以维护堆属性。此操作的操作成本为**O(n**)。

+   **ExtractMin**: ExtractMin 操作与最小堆相关，既返回集合中最小值或最低优先级的对象，又将其从集合中移除。与 FindMin 操作类似，这通常是索引为 0 或 1 的对象，具体取决于设计。此操作还将重新排序堆以维护堆属性。此操作的操作成本为**O(n**)。

+   **DeleteMax**: 删除最大值操作与最大堆相关，简单地说就是从集合中移除最大值或最高优先级的对象。与 FindMax 操作类似，这通常是索引为 0 或 1 的对象，具体取决于设计。此操作还将重新排序堆以维护堆属性。此操作的操作成本为**O(n**)。

+   **DeleteMin**: 删除最小值操作与最小堆相关，简单地说就是从集合中移除最小值或最低优先级的对象。与 FindMin 操作类似，这通常是索引为 0 或 1 的对象，具体取决于设计。此操作还将重新排序堆以维护堆属性。此操作的操作成本为**O(n**)。

+   **Count**: Count 操作返回堆中的节点总数。此操作的操作成本为**O(1**)。

+   **Children**: 子节点操作将返回给定节点或节点索引的两个子节点。由于必须执行两次计算以收集子节点，因此此操作的操作成本为**O(2**)。

+   **Parent**: 父节点操作将返回任何给定节点或节点索引的父节点。此操作的操作成本为**O(1**)。

### 注意

这一系列操作可能会让你想起第九章中讨论的树数据结构，即“非线性结构：树”。重要的是要注意，尽管二叉堆与二叉搜索树非常相似，但两者不应混淆。像二叉搜索树一样，堆数据结构组织集合中的每个节点。堆根据节点或环境的某些任意属性对节点进行排序，而每个节点的值并不一定是有序的。另一方面，在二叉搜索树中，节点的值本身是有序的。

# 实例化堆

由于堆是一种树形数据结构，我们在讨论的语言中不会找到原生的具体实现并不奇怪。然而，堆数据结构实际上非常简单实现。因此，我们将构建自己的堆结构，具体来说是一个最小堆。

# 最小堆结构

在我们开始之前，我们需要详细说明我们的堆结构将具有的一些特性。首先，我们将使用数组来实现堆，第一个节点将占据这个数组的`0`索引。这个决定很重要，因为它影响我们计算每个节点父节点和子节点的公式。接下来，我们需要一个对象来表示我们堆中的节点。由于这将是我们的演示中一个非常简单的对象，我们将直接在堆实现中定义它的类。

由于这是一个最小堆，我们只需要实现`min`操作。因此，我们的实现必须公开`FindMin`（查看）、`ExtractMin`（弹出）和`DeleteMin`方法。堆的`Insert`、`Count`、`Children`和`Parent`操作将分别作为单独的方法实现。

我们的最小堆实现还需要两个辅助方法来重新排序集合，每当添加或删除节点时。我们将这些方法命名为`OrderHeap`和`SwapNodes`，它们的功能应该是自解释的。

### 注意

注意，最大堆的实现几乎与最小堆相同，只是在几个操作中交换了变量。我们将在实现中详细说明这些差异。

**C#**

C#提供了足够的功能，让我们用很少的代码创建一个通用的堆数据结构。首先，我们需要构建一个简单的类来表示堆的节点：

```py
    public class HeapNode 
    { 
        public int Data; 
    } 

```

这个类非常简单，只包含一个`public`属性来存储我们的整数数据。由于这个类的内容在以下每种语言示例中都是一致的，我们在这里只检查它。

接下来，我们可以实现我们的堆函数。下面是一个`MinHeap`类在 C#中的具体实现示例：

```py
    List<HeapNode> elements; 
    public int Count 
    { 
        get 
        { 
            return elements.Count; 
        } 
    } 

    public MinHeap() 
    { 
        elements = new List<HeapNode>(); 
    } 

```

我们的`MinHeap`类包含两个公共字段。第一个是一个名为`elements`的`List<HeapNode>`，它代表我们的堆集合。第二个是一个`Count`字段，它将返回集合中的元素总数。最后，我们的构造函数简单地初始化`elements`集合。

```py
    public void Insert(HeapNode item) 
    { 
        elements.Add(item); 
        OrderHeap(); 
    } 

```

`Insert(HeapNode item)`方法接受一个新的`HeapNode`对象并将其添加到集合中。一旦对象被添加，方法就调用`OrderHeap()`以确保新对象被放置在正确的位置以保持堆属性。

```py
    public void Delete(HeapNode item) 
    { 
        int i = elements.IndexOf(item); 
        int last = elements.Count - 1; 

        elements[i] = elements[last]; 
        elements.RemoveAt(last); 
        OrderHeap(); 
    } 

```

`Delete(HeapNode item)`方法接受一个要从中移除的`HeapNode`对象。该方法首先找到要移除的项的索引，然后获取集合中最后一个对象的索引。接下来，方法通过用堆中的最后一个节点的引用覆盖其位置来删除匹配的节点，然后移除最后一个节点。最后，调用`OrderHeap()`方法以确保最终的集合满足堆属性。

```py
    public HeapNode ExtractMin() 
    { 
        if (elements.Count > 0) 
        { 
            HeapNode item = elements[0]; 
            Delete(item); 
            return item; 
        } 

        return null; 
    } 

```

`ExtractMin()`方法首先确认`elements`集合至少有一个元素。如果没有，则方法返回`null`。否则，方法创建一个新的`HeapNode`实例，称为`item`，并将其设置为集合中的根对象，即最小的对象或具有最低优先级的对象。接下来，方法调用`Delete(item)`从集合中删除节点。最后，由于`ExtractMin`函数必须返回一个对象，方法将`item`返回给调用者。

```py
    public HeapNode FindMin() 
    { 
        if (elements.Count > 0) 
        { 
            return elements[0]; 
        } 

        return null; 
    } 

```

`FindMin()`方法与`ExtractMin()`方法非常相似，不同之处在于它不会从集合中移除返回的最小值。该方法首先确认元素集合至少有一个元素。如果没有，则方法返回`null`。否则，方法返回集合中的根对象，即最小的对象或具有最低优先级的对象。

```py
    private void OrderHeap() 
    { 
        for (int i = elements.Count - 1; i > 0; i--) 
        { 
            int parentPosition = (i - 1) / 2; 

            if (elements[parentPosition].Data > elements[i].Data) 
            { 
                SwapElements(parentPosition, i); 
            } 
        } 
    }   

    private void SwapElements(int firstIndex, int secondIndex) 
    { 
        HeapNode tmp = elements[firstIndex]; 
        elements[firstIndex] = elements[secondIndex]; 
        elements[secondIndex] = tmp; 
    } 

```

私有的`OrderHeap()`方法是`MinHeap`类的核心。这是负责维护集合堆属性的方法。该方法首先根据元素集合的长度建立了一个`for`循环，并从集合的末尾开始向前迭代。

### 注意

由于我们知道任何索引为 i 的对象的两个子节点位于索引*2i + 1*和*2i + 2*，我们同样知道任何索引为 i 的对象的父节点位于*(i - 1) / 2*。这个公式之所以有效，是因为结果值被定义为整数，这意味着任何浮点值都会被截断，只保留整数值。这个算法通过`OrderHeap()`方法中的`int parentPosition = (i - 1) / 2;`代码实现，确保堆数据结构保持其二进制性质。

使用最小堆属性公式，`for` 循环首先确定当前节点的父索引。接下来，将当前节点 `Data` 字段的值与父节点的值进行比较；如果父节点更大，则方法调用 `SwapElements(parentPosition, i)`。一旦评估了每个节点，方法就完成了，整个集合的堆属性是一致的。

### 注意

注意，通过交换 `if` 语句的两个操作数，或者简单地更改比较器从 `>` 到 `<`，或者，我们的集合实际上会从最小堆变为最大堆。利用这一知识，确实可以非常简单地创建一个堆集合，该集合可以在 *运行时* 被定义为最小堆或最大堆。

`SwapElements(int firstIndex, int secondIndex)` 方法的功能是显而易见的。给定索引处的每个节点都被交换，以强制执行堆属性。

```py
    public List<HeapNode> GetChildren(int parentIndex) 
    { 
        if (parentIndex >= 0) 
        { 
            List<HeapNode> children = new List<HeapNode>(); 
            int childIndexOne = (2 * parentIndex) + 1; 
            int childIndexTwo = (2 * parentIndex) + 2; 
            children.Add(elements[childIndexOne]); 
            children.Add(elements[childIndexTwo]); 

            return children; 
        } 

        return null; 
    } 

```

使用相同的规则，即任何对象的索引为 i 的两个子节点位于索引 *2i + 1* 和 *2i + 2*，`GetChildren(int parentIndex)` 方法会收集并返回给定父索引的两个子节点。该方法首先确认 `parentIndex` 不小于 `0`，否则返回 `null`。如果 `parentIndex` 有效，该方法会创建一个新的 `List<Heapnode>` 并使用计算出的子索引填充它，然后返回 `children` 集合。

```py
    public HeapNode GetParent(int childIndex) 
    { 
        if (childIndex > 0 && elements.Count > childIndex) 
        { 
            int parentIndex = (childIndex - 1) / 2; 
            return elements[parentIndex]; 
        } 

        return null; 
    } 

```

最后，`GetParent(int childIndex)` 方法与 `GetChildren` 方法的工作原理相同。如果给定的 `childIndex` 大于 `0`，则节点有一个父节点。该方法确认我们不是在搜索根节点，并确认索引不在集合的界限之外。如果任一检查失败，该方法返回 `null`。否则，该方法确定节点的父索引，然后返回在该索引处找到的节点。

**Java**

Java 还提供了构建我们 `MinHeap` 类的健壮实现所需的基本工具，而无需编写太多代码。以下是该类在 Java 中的可能外观：

```py
    List<HeapNode> elements; 

    public int size() 
    { 
        return elements.size(); 
    } 

    public MinHeap() 
    { 
        elements = new ArrayList<HeapNode>();  
    } 

```

我们的 `MinHeap` 类包括一个名为 `elements` 的公共字段，其类型为抽象类型 `List<HeapNode>`，代表我们的堆集合。该类还包括一个名为 `size()` 的方法，它将返回集合中的元素总数。最后，我们的构造函数只是将 `elements` 集合初始化为 `ArrayList<HeapNode>`：

```py
    public void insert(HeapNode item) 
    { 
        elements.add(item); 
        orderHeap(); 
    } 

```

`insert(HeapNode item)` 方法接受一个新的 `HeapNode` 对象并将其添加到集合中。一旦对象被添加，该方法调用 `orderHeap()` 确保新对象被放置在正确的位置以保持堆属性。

```py
    public void delete(HeapNode item) 
    { 
        int i = elements.indexOf(item); 
        int last = elements.size() - 1; 

        elements.set(i, elements.get(last)); 
        elements.remove(last); 
        orderHeap(); 
    } 

```

`delete(HeapNode item)` 方法接受一个要从中移除的 `HeapNode` 项目。该方法首先找到要删除的项目索引，然后获取集合中最后一个对象的索引。接下来，通过将堆中最后一个节点的引用覆盖到该位置来删除匹配的节点，然后删除最后一个节点。最后，调用 `orderHeap()` 确保最终的集合满足堆属性。

```py
    public HeapNode extractMin() 
    { 
        if (elements.size() > 0) 
        { 
            HeapNode item = elements.get(0); 
            delete(item); 
            return item; 
        } 

        return null; 
    } 

```

`extractMin()` 方法首先确认 `elements` 集合至少有一个元素。如果没有，该方法返回 `null`。否则，该方法创建一个名为 `item` 的新 `HeapNode` 实例，并将其设置为集合中的根对象，即最小的对象或具有最低优先级的对象。接下来，该方法调用 `delete(item)` 从集合中删除该节点。最后，由于 `ExtractMin` 函数必须返回一个对象，因此该方法将 `item` 返回给调用者。

```py
    public HeapNode findMin() 
    { 
        if (elements.size() > 0) 
        { 
            return elements.get(0); 
        } 

        return null; 
    } 

```

`findMin()` 方法与 `extractMin()` 方法非常相似，不同之处在于它不会从集合中删除返回的最小值。该方法首先确认 `elements` 集合至少有一个元素。如果没有，该方法返回 `null`。否则，该方法通过调用 `elements.get(0)` 返回集合中的根对象。

```py
    private void orderHeap() 
    { 
        for (int i = elements.size() - 1; i > 0; i--) 
        { 
            int parentPosition = (i - 1) / 2; 

            if (elements.get(parentPosition).Data > elements.get(i).Data) 
            { 
                swapElements(parentPosition, i); 
            } 
        } 
    } 

    private void swapElements(int firstIndex, int secondIndex) 
    { 
        HeapNode tmp = elements.get(firstIndex); 
        elements.set(firstIndex, elements.get(secondIndex)); 
        elements.set(secondIndex, tmp); 
    } 

```

私有的 `orderHeap()` 方法负责维护集合的堆属性。该方法首先根据 `elements` 集合的长度建立 `for` 循环，并从集合的末尾开始迭代到开头。

使用最小堆属性公式，`for` 循环首先确定当前节点的父索引。接下来，将当前节点的 `Data` 字段值与父节点的值进行比较，如果父节点更大，则调用 `swapElements(parentPosition, i)`。一旦评估了每个节点，方法就完成了，并且整个集合的堆属性是一致的。

`swapElements(int firstIndex, int secondIndex)` 方法的功能是显而易见的。给定索引处的每个节点都被交换，以强制执行堆属性。

```py
    public List<HeapNode> getChildren(int parentIndex) 
    { 
        if (parentIndex >= 0) 
        { 
            ArrayList<HeapNode> children = new ArrayList<HeapNode>(); 
            int childIndexOne = (2 * parentIndex) + 1; 
            int childIndexTwo = (2 * parentIndex) + 2; 
            children.add(elements.get(childIndexOne)); 
            children.add(elements.get(childIndexTwo)); 

            return children; 
        } 

        return null; 
    } 

```

使用相同的规则，即任何对象的两个子节点位于索引 *2i + 1* 和 *2i + 2*，`getChildren(int parentIndex)` 方法收集并返回给定父索引的两个子节点。该方法首先确认 `parentIndex` 不小于 0，否则返回 `null`。如果 `parentIndex` 有效，该方法创建一个新的 `ArrayList<Heapnode>` 并使用计算出的子索引填充它，然后再返回 `children` 集合。

```py
    public HeapNode getParent(int childIndex) 
    { 
        if (childIndex > 0 && elements.size() > childIndex) 
        { 
            int parentIndex = (childIndex - 1) / 2; 
            return elements.get(parentIndex); 
        } 

        return null; 
    } 

```

最后，`getParent(int childIndex)` 与 `getChildren` 的工作原理相同。如果给定的 `childIndex` 大于 0，则节点有一个父节点。该方法确认我们不是在搜索根节点，并确认索引不在集合的界限之外。如果任一检查失败，则方法返回 `null`。否则，该方法确定节点的父索引，然后返回在该索引处找到的节点。

**Objective-C**

使用 `NSMutableArray` 作为核心结构，Objective-C 也可以轻松实现最小堆数据结构。以下是 `EDSMinHeap` 类在 Objective-C 中的可能外观：

```py
    @interface EDSMinHeap() 
    { 
        NSMutableArray<EDSHeapNode*> *_elements; 
    } 

    @implementation EDSMinHeap 

    -(instancetype)initMinHeap{ 

        if (self = [super init]) 
        { 
            _elements = [NSMutableArray array]; 
        } 

        return self; 
    } 

```

使用类簇 `NSMutableArray`，我们为我们的类创建一个名为 `_elements` 的 ivar。初始化器实例化此数组，为我们构建 `EDSMinHeap` 类提供了底层数据结构。

```py
    -(NSInteger)getCount 
    { 
        return [_elements count]; 
    } 

```

我们的 `EDSMinHeap` 类包括一个名为 `Count` 的公共属性，`getCount()` 访问器返回 `_elements` 数组的 `count` 属性。

```py
    -(void)insert:(EDSHeapNode*)item 
    { 
        [_elements addObject:item]; 
        [self orderHeap]; 
    } 

```

`insert:` 方法接受一个新的 `EDSHeapNode` 对象并将其添加到数组中。一旦对象被添加，该方法就调用 `orderHeap` 确保新对象被放置在正确的位置以维护堆属性：

```py
    -(void)delete:(EDSHeapNode*)item 
    { 
        long i = [_elements indexOfObject:item]; 

        _elements[i] = [_elements lastObject]; 
        [_elements removeLastObject]; 
        [self orderHeap]; 
    } 

```

`delete:` 方法接受一个要从中移除的 `EDSHeapNode` 对象。该方法首先使用 `indexOfObject:` 找到要移除的对象的索引，然后通过用堆中的 `lastObject` 的引用覆盖其位置来删除匹配的节点。接下来，使用 `removeLastObject` 移除最后一个节点。最后，调用 `orderHeap:` 确保最终的集合满足堆属性。

```py
    -(EDSHeapNode*)extractMin 
    { 
        if ([_elements count] > 0) 
        { 
            EDSHeapNode *item = _elements[0]; 
            [self delete:item]; 
            return item; 
        } 

        return nil; 
    } 

```

`extractMin` 方法首先确认 `_elements` 集合至少有一个元素。如果没有，该方法返回 `nil`。否则，该方法创建一个名为 `item` 的新 `EDSHeapNode` 实例，并将其设置为集合中的根对象，即最小的对象或具有最低优先级的对象。接下来，该方法调用 `delete:` 从集合中删除该节点。最后，由于 *ExtractMin* 函数必须返回一个对象，因此该方法将 `item` 返回给调用者。

```py
    -(EDSHeapNode*)findMin 
    { 
        if ([_elements count] > 0) 
        { 
            return _elements[0]; 
        } 

        return nil; 
    } 

```

`findMin` 方法与 `extractMin` 方法非常相似，不同之处在于它不会从集合中移除返回的最小值。该方法首先确认元素集合至少有一个元素。如果没有，该方法返回 `nil`。否则，该方法返回集合中的第一个对象，即根节点。

```py
    -(void)orderHeap 
    { 
        for (long i = [_elements count] - 1; i > 0; i--) 
        { 
            long parentPosition = (i - 1) / 2; 

            if (_elements[parentPosition].data > _elements[i].data) 
            { 
                [self swapElement:parentPosition withElement:i]; 
            } 
        } 
    } 

    -(void)swapElement:(long)firstIndex withElement:(long)secondIndex 
    { 
        EDSHeapNode *tmp = _elements[firstIndex]; 
        _elements[firstIndex] = _elements[secondIndex]; 
        _elements[secondIndex] = tmp; 
    } 

```

私有的 `orderHeap` 方法负责维护集合的堆属性。该方法首先根据元素集合的长度建立 `for` 循环，并从后向前遍历集合。

使用最小堆属性公式，`for`循环首先识别当前节点的父索引。接下来，将当前节点`data`属性的值与父节点的值进行比较，如果父节点更大，则方法调用`swapElement:withElement:`。一旦每个节点都被评估，方法就完成了，整个集合的堆属性是一致的。

`swapElement:withElement:`方法的功能是显而易见的。给定索引处的每个节点都被交换，以强制执行堆属性。

```py
    -(NSArray<EDSHeapNode*>*)childrenOfParentIndex:(NSInteger)parentIndex 
    { 
        if (parentIndex >= 0) 
        { 
            NSMutableArray *children = [NSMutableArray array]; 
            long childIndexOne = (2 * parentIndex) + 1; 
            long childIndexTwo = (2 * parentIndex) + 2; 
            [children addObject:_elements[childIndexOne]]; 
            [children addObject:_elements[childIndexTwo]]; 

            return children; 
        } 
        return nil; 
    } 

```

使用规定任何对象在索引 i 处的两个子节点位于索引*2i + 1*和*2i + 2*的规则，`childrenOfParentIndex:`方法收集并返回给定父索引的两个子节点。该方法首先确认`parentIndex`不小于 0，否则返回`nil`。如果`parentIndex`有效，则方法创建一个新的`NSMutableArray`，并使用计算出的子索引中的节点填充它，然后返回`children`集合。

```py
    -(EDSHeapNode*)parentOfChildIndex:(NSInteger)childIndex 
    { 
        if (childIndex > 0 && [_elements count] > childIndex) 
        { 
            long parentIndex = (childIndex - 1) / 2; 
            return _elements[parentIndex]; 
        } 

        return nil; 
    } 

```

最后，`parentOfChildIndex:`与`childrenOfParentIndex:`的工作原理相同。如果给定的`childIndex`大于 0，则节点有一个父节点。该方法确认我们不是在搜索根节点，并且也确认索引不在集合的界限之外。如果任一检查失败，则方法返回`nil`。否则，该方法确定节点的父索引，然后返回在该索引处找到的节点。

**Swift**

我们的 Swift `MinHeap`类在结构和功能上与 C#和 Java 实现相似。以下是一个 Swift 中`MinHeap`类的示例：

```py
    public var _elements: Array = [HeapNode]() 
    public init () {} 

    public func getCount() -> Int 
    { 
        return _elements.count 
    } 

```

使用`Array`类，我们为我们的类创建一个名为`_elements`的私有属性。由于我们的属性是声明和实例化同时进行的，并且没有其他需要实例化的自定义代码，我们可以排除显式的公共初始化器并依赖于默认初始化器。我们的类还提供了一个名为`getCount()`的公共方法，它返回`_elements`数组的大小。

```py
    public func insert(item: HeapNode) 
    { 
        _elements.append(item) 
        orderHeap() 
    } 

```

`insert(HeapNode item)`方法接受一个新的`HeapNode`对象并将其添加到集合中。一旦对象被添加，方法就调用`orderHeap()`以确保新对象被放置在正确的位置以保持堆属性。

```py
    public func delete(item: HeapNode) 
    { 
        if let index = _elements.index(of: item) 
        { 
            _elements[index] = _elements.last! 
            _elements.removeLast() 
            orderHeap() 
        } 
    } 

```

`delete(HeapNode item)`方法接受一个要从集合中删除的`HeapNode`项。该方法首先找到要删除的项的`index`，然后通过用堆中的`last`对象的引用覆盖其位置来删除匹配的节点。最后，调用`orderHeap()`方法以确保最终的集合满足堆属性。

```py
    public func extractMin() -> HeapNode? 
    { 
        if (_elements.count > 0) 
        { 
            let item = _elements[0] 
            delete(item: item) 
            return item 
        } 

        return nil 
    } 

```

`extractMin()` 方法首先确认 `elements` 集合至少有一个元素。如果没有，则方法返回 `nil`。否则，该方法创建一个名为 `item` 的新变量，并将其设置为集合中的根对象，即最小的 `HeapNode` 或具有最低优先级的 `HeapNode`。接下来，该方法调用 `delete(item: Heapnode)` 从集合中删除节点。最后，该方法将 `item` 返回给调用者。

```py
    public func findMin() -> HeapNode? 
    { 
        if (_elements.count > 0) 
        { 
            return _elements[0] 
        } 

        return nil 
    } 

```

`findMin()` 方法与 `extractMin()` 方法非常相似，不同之处在于它不会从集合中移除返回的最小值。该方法首先确认元素集合至少有一个元素。如果没有，则方法返回 `nil`。否则，该方法返回 `_elements[0]`，这是集合中的根对象。

```py
    public func orderHeap() 
    { 
        for i in (0..<(_elements.count) - 1).reversed() 
        { 
            let parentPosition = (i - 1) / 2 

            if (_elements[parentPosition].data! > _elements[i].data!) 
            { 
                swapElements(first: parentPosition, second: i) 
            } 
        } 
    } 

    public func swapElements(first: Int, second: Int) 
    { 
        let tmp = _elements[first] 
        _elements[first] = _elements[second] 
        _elements[second] = tmp 
    } 

```

私有 `orderHeap()` 方法负责维护集合的堆属性。该方法首先根据元素集合的长度建立 `for` 循环，并从末尾开始迭代集合。

使用最小堆属性公式，`for` 循环首先确定当前节点的父索引。然后，将当前节点 `data` 字段的值与父节点的值进行比较，如果父节点更大，则方法调用 `swapElements(first: Int, second: Int)`。一旦评估了每个节点，方法完成，整个集合的堆属性保持一致。

`swapElements(int firstIndex, int secondIndex)` 方法的功能是显而易见的。给定索引处的每个节点都被交换，以强制执行堆属性：

```py
    public func getChildren(parentIndex: Int) -> [HeapNode]? 
    { 
        if (parentIndex >= 0) 
        { 
            var children: Array = [HeapNode]() 

            let childIndexOne = (2 * parentIndex) + 1; 
            let childIndexTwo = (2 * parentIndex) + 2; 
            children.append(_elements[childIndexOne]) 
            children.append(_elements[childIndexTwo]) 

            return children; 
        } 

        return nil; 
    } 

```

使用相同的规则，即任何对象的索引 `i` 的两个子节点位于索引 *2i + 1* 和 *2i + 2*，`getChildren(parentIndex: Int)` 方法收集并返回给定父索引的两个子节点。该方法首先确认 `parentIndex` 不小于 0，否则返回 `nil`。如果 `parentIndex` 有效，该方法创建一个新的 `Array`，包含 `HeapNode` 对象，并使用计算出的子索引填充它，然后返回 `children` 集合：

```py
    public func getParent(childIndex: Int) -> HeapNode? 
    { 
        if (childIndex > 0 && _elements.count > childIndex) 
        { 
            let parentIndex = (childIndex - 1) / 2; 
            return _elements[parentIndex]; 
        } 

        return nil; 
    } 

```

最后，`getParent(childIndex: Int)` 方法与 `getChildren` 方法的工作原理相同。如果给定的 `childIndex` 大于 `0`，则节点有一个父节点。方法确认我们不是在寻找根节点，并确认索引不在集合的界限之外。如果任一检查失败，则方法返回 `nil`。否则，方法确定节点的父索引，然后返回在该索引处找到的节点。

# 常见应用

堆数据结构实际上相当常见，尽管你可能并不总是意识到你正在处理一个。以下是堆数据结构最常见的应用之一：

+   **选择算法**：选择算法用于确定集合中的第 k 个最小或最大元素，或者集合的中值对象。在通常的集合中，这种操作的成本为 O(n)。然而，在一个使用数组实现的有序堆中，找到第 k 个元素是一个**O**(1)操作，因为我们只需简单地检查数组中的 k 索引即可找到该元素。

+   **优先队列**：优先队列是一种类似于标准队列的抽象数据结构，除了节点包含一个额外的值，表示该对象相对于集合中其他对象的优先级。由于堆数据结构的自然排序，优先队列通常使用堆来实现。

# 摘要

在本章中，我们学习了堆数据结构。我们考察了与堆一起工作时最常用的操作及其复杂度成本。随后，我们从头开始创建了自己的简单最小堆数据结构类，并讨论了如何使用最小堆属性公式来计算任何给定节点索引的父节点或子节点。最后，我们考察了堆数据结构最常见的应用。
