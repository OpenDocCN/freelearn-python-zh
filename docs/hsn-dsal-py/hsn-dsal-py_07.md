# 第七章：哈希和符号表

我们之前已经看过**数组**和**列表**，其中项目按顺序存储并通过索引号访问。索引号对计算机来说很有效。它们是整数，因此快速且易于操作。但是，它们并不总是对我们很有效。例如，如果我们有一个地址簿条目，比如在索引号 56 处，那个数字并没有告诉我们太多。没有任何东西将特定联系人与数字 56 联系起来。使用索引值从列表中检索条目是困难的。

在本章中，我们将研究一种更适合这种问题的数据结构：字典。字典使用关键字而不是索引号，并以`（键，值）`对的形式存储数据。因此，如果该联系人被称为*James*，我们可能会使用关键字*James*来定位联系人。也就是说，我们不会通过调用*contacts [56]*来访问联系人，而是使用*contacts* `james`。

字典是一种广泛使用的数据结构，通常使用哈希表构建。顾名思义，哈希表依赖于一种称为**哈希**的概念。哈希表数据结构以`键/值`对的方式存储数据，其中键是通过应用哈希函数获得的。它以非常高效的方式存储数据，因此检索速度非常快。我们将在本章讨论所有相关问题。

我们将在本章涵盖以下主题：

+   哈希

+   哈希表

+   不同的元素功能

# 技术要求

除了需要在系统上安装 Python 之外，没有其他技术要求。这是本章讨论的源代码的 GitHub 链接：[`github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter07`](https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter07)。

# 哈希

哈希是一个概念，当我们将任意大小的数据提供给函数时，我们会得到一个简化的小值。这个函数称为**哈希函数**。哈希使用一个哈希函数将给定的数据映射到另一个数据范围，以便新的数据范围可以用作哈希表中的索引。更具体地说，我们将使用哈希将字符串转换为整数。在本章的讨论中，我们使用字符串转换为整数，但它可以是任何其他可以转换为整数的数据类型。让我们看一个例子来更好地理解这个概念。我们想要对表达式`hello world`进行哈希，也就是说，我们想要得到一个数值，我们可以说*代表*该字符串。

我们可以使用`ord（）`函数获得任何字符的唯一序数值。例如，`ord（'f'）`函数给出 102。此外，要获得整个字符串的哈希值，我们只需对字符串中每个字符的序数进行求和。请参阅以下代码片段：

```py
>>> sum(map(ord, 'hello world'))
1116
```

对于整个`hello world`字符串获得的数值`1116`称为**字符串的哈希**。请参考以下图表，以查看导致哈希值`1116`的字符串中每个字符的序数值：

![](img/12710178-03f0-4efd-8f50-b84231ea4f63.png)

前面的方法用于获得给定字符串的哈希值，并且似乎运行良好。但是，请注意，我们可以更改字符串中字符的顺序，我们仍然会得到相同的哈希值；请参阅以下代码片段，我们对`world hello`字符串获得相同的哈希值：

```py
>>> sum(map(ord, 'world hello'))
1116
```

同样，对于`gello xorld`字符串，哈希值将是相同的，因为该字符串的字符的序数值之和将是相同的，因为`g`的序数值比`h`小 1，`x`的序数值比`w`大 1。请参阅以下代码片段：

```py
>>> sum(map(ord, 'gello xorld'))
1116
```

看一下下面的图表，我们可以观察到该字符串的哈希值再次为`1116`：

![](img/623ed462-1cfe-4b45-b619-692367501b52.png)

# 完美哈希函数

**完美哈希函数**是指我们为给定字符串（它可以是任何数据类型，这里我们现在限制讨论为字符串）得到唯一的哈希值。实际上，大多数哈希函数都是不完美的，并且会发生冲突。这意味着哈希函数给一个以上的字符串返回相同的哈希值；这是不希望的，因为完美哈希函数应该为一个字符串返回唯一的哈希值。通常，哈希函数需要非常快速，因此通常不可能创建一个为每个字符串返回唯一哈希值的函数。因此，我们接受这一事实，并且知道我们可能会遇到一些冲突，也就是说，两个或更多个字符串可能具有相同的哈希值。因此，我们尝试找到一种解决冲突的策略，而不是试图找到一个完美的哈希函数。

为了避免前面示例中的冲突，我们可以例如添加一个乘数，使得每个字符的序数值乘以一个随着字符串进展而不断增加的值。接下来，通过添加每个字符的乘以序数值来获得字符串的哈希值。为了更好地理解这个概念，请参考以下图表：

![](img/a488b9e9-4cb8-4039-8c67-4954d1527d26.png)

在上图中，每个字符的序数值逐渐乘以一个数字。请注意，最后一行是值的乘积结果；第二行是每个字符的序数值；第三行显示乘数值；第四行通过将第二行和第三行的值相乘得到值，因此 `104 x 1` 等于 `104`。最后，我们将所有这些乘积值相加，得到 `hello world` 字符串的哈希值，即 `6736`。

这个概念的实现如下函数所示：

```py
    def myhash(s): 
        mult = 1 
        hv = 0 
        for ch in s: 
            hv += mult * ord(ch) 
            mult += 1 
        return hv 
```

我们可以在下面显示的字符串上测试这个函数：

```py
for item in ('hello world', 'world hello', 'gello xorld'): 
        print("{}: {}".format(item, myhash(item))) 
```

运行此程序，我们得到以下输出：

```py
% python hashtest.py

hello world: 6736
world hello: 6616
gello xorld: 6742
```

我们可以看到，这一次对这三个字符串得到了不同的哈希值。但是，这并不是一个完美的哈希。让我们尝试字符串 `ad` 和 `ga`：

```py
% python hashtest.py

ad: 297
ga: 297
```

我们仍然得到两个不同字符串相同的哈希值。因此，我们需要制定一种解决这种冲突的策略。我们很快将看到这一点，但首先，我们将学习哈希表的实现。

# 哈希表

**哈希表**是一种数据结构，其中元素是通过关键字而不是索引号访问的，不同于**列表**和**数组**。在这种数据结构中，数据项以类似于字典的键/值对的形式存储。哈希表使用哈希函数来找到应该存储和检索元素的索引位置。这使我们能够快速查找，因为我们使用与键的哈希值对应的索引号。

哈希表数据结构中的每个位置通常称为**槽**或**桶**，可以存储一个元素。因此，形式为 `(key, value)` 的每个数据项将存储在哈希表中由数据的哈希值决定的位置上。例如，哈希函数将输入字符串名称映射到哈希值；`hello world` 字符串被映射到哈希值 92，找到哈希表中的一个槽位置。考虑以下图表：

![](img/b737f679-af81-47f6-ada2-c7beb23d6de8.png)

为了实现哈希表，我们首先创建一个类来保存哈希表项。这些项需要有一个键和一个值，因为我们的哈希表是一个 `{key-value}` 存储：

```py
    class HashItem: 
        def __init__(self, key, value): 
            self.key = key 
            self.value = value 
```

这为我们提供了一种非常简单的存储项的方法。接下来，我们开始研究哈希表类本身。像往常一样，我们从构造函数开始：

```py
    class HashTable: 
        def __init__(self): 
            self.size = 256 
            self.slots = [None for i in range(self.size)] 
            self.count = 0 
```

哈希表使用标准的 Python 列表来存储其元素。让我们将哈希表的大小设置为 256 个元素。稍后，我们将研究如何在开始填充哈希表时扩展哈希表的策略。我们现在将在代码中初始化一个包含 256 个元素的列表。这些是要存储元素的位置——插槽或桶。因此，我们有 256 个插槽来存储哈希表中的元素。最后，我们添加一个计数器，用于记录实际哈希表元素的数量：

![](img/c5ea5ffe-6a97-4132-837e-830e98b472c1.png)

重要的是要注意表的大小和计数之间的区别。表的大小是指表中插槽的总数（已使用或未使用）。表的计数是指填充的插槽的数量，也就是已添加到表中的实际（键-值）对的数量。

现在，我们需要决定将我们的哈希函数添加到表中。我们可以使用相同的哈希函数，它返回字符串中每个字符的序数值的总和，稍作修改。由于我们的哈希表有 256 个插槽，这意味着我们需要一个返回 1 到 256 范围内的值的哈希函数（表的大小）。一个很好的方法是返回哈希值除以表的大小的余数，因为余数肯定是 0 到 255 之间的整数值。

哈希函数只是用于类内部的，所以我们在名称前面加下划线（`_`）来表示这一点。这是 Python 中用来表示某些东西是内部使用的正常约定。这是`hash`函数的实现：

```py
    def _hash(self, key): 
        mult = 1 
        hv = 0 
        for ch in key: 
            hv += mult * ord(ch) 
            mult += 1 
        return hv % self.size 
```

目前，我们假设键是字符串。我们将讨论如何稍后使用非字符串键。现在，`_hash()`函数将为字符串生成哈希值。

# 在哈希表中存储元素

要将元素存储在哈希表中，我们使用`put()`函数将它们添加到表中，并使用`get()`函数检索它们。首先，我们将看一下`put()`函数的实现。我们首先将键和值嵌入`HashItem`类中，然后计算键的哈希值。

这是`put`函数的实现，用于将元素存储在哈希表中：

```py
    def put(self, key, value): 
        item = HashItem(key, value) 
        h = self._hash(key) 
```

一旦我们知道键的哈希值，它将被用来找到元素应该存储在哈希表中的位置。因此，我们需要找到一个空插槽。我们从与键的哈希值对应的插槽开始。如果该插槽为空，我们就在那里插入我们的项。

但是，如果插槽不为空，并且项的键与当前键不同，那么我们就会发生冲突。这意味着我们有一个项的哈希值与表中先前存储的某个项相同。这就是我们需要想出一种处理冲突的方法的地方。

例如，在下面的图表中，**hello world**键字符串已经存储在表中，当一个新的字符串`world hello`得到相同的哈希值`92`时，就会发生冲突。看一下下面的图表：

![](img/6fc3a488-d225-45a8-aaec-63afa11aa0df.png)

解决这种冲突的一种方法是从冲突的位置找到另一个空插槽；这种冲突解决过程称为**开放寻址**。我们可以通过线性地查找下一个可用插槽来解决这个问题，方法是在发生冲突的前一个哈希值上加`1`。我们可以通过将键字符串中每个字符的序数值的总和加`1`来解决这个冲突，然后再除以哈希表的大小来获得哈希值。这种系统化的访问每个插槽的方式是解决冲突的线性方式，称为**线性探测**。

让我们考虑一个例子，如下图所示，以更好地理解我们如何解决这个冲突。密钥字符串`eggs`的哈希值是 51。现在，由于我们已经使用了这个位置来存储数据，所以发生了冲突。因此，我们在哈希值中添加 1，这是由字符串的每个字符的序数值的总和计算出来的，以解决冲突。因此，我们获得了这个密钥字符串的新哈希值来存储数据——位置 52。请参见以下图表和代码片段以进行此实现：

![](img/343bdb1c-ac38-4fc1-a624-10dfbab473ec.png)

现在，考虑以下代码：

```py
    while self.slots[h] is not None: 
        if self.slots[h].key is key: 
            break 
        h = (h + 1) % self.size 
```

上述代码是用来检查槽是否为空，然后使用描述的方法获取新的哈希值。如果槽为空（这意味着槽以前包含`None`），则我们将计数增加一。最后，我们将项目插入到所需位置的列表中：

```py
    if self.slots[h] is None: 
        self.count += 1 
    self.slots[h] = item  
```

# 从哈希表中检索元素

要从哈希表中检索元素，将返回与密钥对应的存储值。在这里，我们将讨论检索方法的实现——`get()`方法。此方法将返回与给定密钥对应的表中存储的值。

首先，我们计算要检索的密钥的哈希值对应的值。一旦我们有了密钥的哈希值，我们就在哈希表的哈希值位置查找。如果密钥项与该位置处存储的密钥值匹配，则检索相应的`value`。如果不匹配，那么我们将 1 添加到字符串中所有字符的序数值的总和，类似于我们在存储数据时所做的操作，然后查看新获得的哈希值。我们继续查找，直到找到我们的密钥元素或者检查了哈希表中的所有槽。

考虑一个例子来理解以下图表中的概念，分为四步：

1.  我们计算给定密钥字符串`"egg"`的哈希值，结果为 51。然后，我们将此密钥与位置 51 处存储的密钥值进行比较，但不匹配。

1.  由于密钥不匹配，我们计算一个新的哈希值。

1.  我们查找新创建的哈希值位置 52 处的密钥；我们将密钥字符串与存储的密钥值进行比较，这里匹配，如下图所示。

1.  在哈希表中返回与此密钥值对应的存储值。请参见以下图表：

![](img/a5db02a4-57a6-4528-ab70-45ebc3299844.png)

为了实现这个检索方法，即`get()`方法，我们首先计算密钥的哈希值。接下来，我们在表中查找计算出的哈希值。如果匹配，则返回相应的存储值。否则，我们继续查看描述的计算出的新哈希值位置。以下是`get()`方法的实现：

```py
def get(self, key): 
    h = self._hash(key)    # computer hash for the given key 
    while self.slots[h] is not None:
        if self.slots[h].key is key: 
            return self.slots[h].value 
        h = (h+ 1) % self.size 
    return None        
```

最后，如果在表中找不到密钥，则返回`None`。另一个很好的选择可能是在表中不存在密钥的情况下引发异常。

# 测试哈希表

为了测试我们的哈希表，我们创建`HashTable`并将一些元素存储在其中，然后尝试检索它们。我们还将尝试`get()`一个不存在的密钥。我们还使用了两个字符串`ad`和`ga`，它们发生了冲突，并且由我们的哈希函数返回了相同的哈希值。为了正确评估哈希表的工作，我们也会处理这个冲突，只是为了看到冲突是如何正确解决的。请参见以下示例代码：

```py
ht = HashTable() 
    ht.put("good", "eggs") 
    ht.put("better", "ham") 
    ht.put("best", "spam") 
    ht.put("ad", "do not") 
    ht.put("ga", "collide") 

    for key in ("good", "better", "best", "worst", "ad", "ga"): 
        v = ht.get(key) 
        print(v) 
```

运行上述代码返回以下结果：

```py
% python hashtable.py

eggs
ham
spam
None
do not
collide 
```

如您所见，查找`worst`密钥返回`None`，因为密钥不存在。`ad`和`ga`密钥也返回它们对应的值，显示它们之间的冲突得到了正确处理。

# 使用[]与哈希表

使用`put()`和`get()`方法看起来并不方便。然而，我们更希望能够将我们的哈希表视为列表，因为这样会更容易使用。例如，我们希望能够使用`ht["good"]`而不是`ht.get("good")`来从表中检索元素。

这可以很容易地通过特殊方法`__setitem__()`和`__getitem__()`来完成。请参阅以下代码：

```py
    def __setitem__(self, key, value): 
        self.put(key, value) 

    def __getitem__(self, key): 
        return self.get(key) 
```

现在，我们的测试代码将会是这样的：

```py
    ht = HashTable() 
    ht["good"] = "eggs" 
    ht["better"] = "ham" 
    ht["best"] = "spam" 
    ht["ad"] = "do not" 
    ht["ga"] = "collide" 

    for key in ("good", "better", "best", "worst", "ad", "ga"): 
        v = ht[key] 
        print(v) 

    print("The number of elements is: {}".format(ht.count)) 
```

请注意，我们还打印了已存储在哈希表中的元素数量，使用`count`变量。

# 非字符串键

在实时应用中，通常我们需要使用字符串作为键。然而，如果有必要，您可以使用任何其他 Python 类型。如果您创建自己的类并希望将其用作键，您需要重写该类的特殊`__hash__()`函数，以便获得可靠的哈希值。

请注意，您仍然需要计算哈希值的模运算(`%`)和哈希表的大小以获取插槽。这个计算应该在哈希表中进行，而不是在键类中，因为表知道自己的大小（键类不应该知道它所属的表的任何信息）。

# 扩大哈希表

在我们的示例中，我们将哈希表的大小固定为 256。很明显，当我们向哈希表添加元素时，我们将开始填满空插槽，而在某个时刻，所有插槽都将被填满，哈希表将变满。为了避免这种情况，我们可以在表开始变满时扩大表的大小。

为了扩大哈希表的大小，我们比较表中的大小和计数。`size`是插槽的总数，`count`表示包含元素的插槽的数量。因此，如果`count`等于`size`，这意味着我们已经填满了表。哈希表的负载因子通常用于扩展表的大小；这给了我们一个关于表中有多少可用插槽被使用的指示。哈希表的负载因子通过将表中**已使用**的插槽数量除以表中的**总**插槽数量来计算。它的定义如下：

![](img/ee3e4e63-871f-4f89-9bfe-83df85d4fe6a.png)

当负载因子接近 1 时，这意味着表即将被填满，我们需要扩大表的大小。最好在表几乎填满之前扩大表的大小，因为当表填满时，从表中检索元素会变慢。负载因子为 0.75 可能是一个不错的值，用来扩大表的大小。

下一个问题是我们应该将表的大小增加多少。一种策略是简单地将表的大小加倍。

# 开放寻址

我们在示例中使用的冲突解决机制是线性探测，这是一种开放寻址策略的例子。线性探测很简单，因为我们使用了固定数量的插槽。还有其他开放寻址策略，它们都共享一个思想，即存在一个插槽数组。当我们想要插入一个键时，我们会检查插槽是否已经有项目。如果有，我们会寻找下一个可用的插槽。

如果我们有一个包含 256 个插槽的哈希表，那么 256 就是哈希表中元素的最大数量。此外，随着负载因子的增加，查找新元素的插入点将需要更长的时间。

由于这些限制，我们可能更喜欢使用不同的策略来解决冲突，比如链接法。

# 链接法

链接是处理哈希表中冲突问题的另一种方法。它通过允许哈希表中的每个插槽存储在冲突位置的多个项目的引用来解决这个问题。因此，在冲突的索引处，我们可以在哈希表中存储多个项目。观察以下图表——字符串**hello world**和**world hello**发生冲突。在链接的情况下，这两个项目都被允许存储在哈希值为**92**的位置上，使用一个**列表**。以下是用于显示使用链接解决冲突的示例图表：

![](img/46886635-c11e-45e1-bc94-291c680dc461.png)

在链接中，哈希表中的插槽被初始化为空列表：

![](img/3518632e-5693-470a-87b0-0e6428fe8e49.png)

当插入一个元素时，它将被追加到与该元素的哈希值对应的列表中。也就是说，如果您有两个具有哈希值`1075`的元素，这两个元素都将被添加到哈希表的`1075%256=51`插槽中存在的列表中：

![](img/d4084896-ca8c-4213-8e75-455df499452e.png)

前面的图表显示了具有哈希值`51`的条目列表。

然后通过链接避免冲突，允许多个元素具有相同的哈希值。因此，哈希表中可以存储的元素数量没有限制，而在线性探测的情况下，我们必须固定表的大小，当表填满时需要后续增长，这取决于负载因子。此外，哈希表可以容纳比可用插槽数量更多的值，因为每个插槽都包含一个可以增长的列表。

然而，在链接中存在一个问题——当列表在特定的哈希值位置增长时，它变得低效。由于特定插槽有许多项目，搜索它们可能会变得非常缓慢，因为我们必须通过列表进行线性搜索，直到找到具有我们想要的键的元素。这可能会减慢检索速度，这是不好的，因为哈希表的目的是高效的。以下图表演示了通过列表项进行线性搜索，直到找到匹配项：

![](img/c8def2e4-1a3e-4b25-ba26-fbad0abbafbb.png)

因此，当哈希表中的特定位置具有许多条目时，检索项目的速度会变慢。可以通过在使用列表的位置上使用另一个数据结构来解决这个问题，该数据结构可以执行快速搜索和检索。使用**二叉搜索树**（**BSTs**）是一个不错的选择，因为它提供了快速检索，正如我们在前一章中讨论的那样。

我们可以简单地在每个插槽中放置一个（最初为空的）BST，如下图所示：

![](img/f33540e3-5074-4d55-9b5b-2ce497987b10.png)

在前面的图表中，`51`插槽包含一个 BST，我们使用它来存储和检索数据项。但我们仍然可能会遇到一个潜在的问题——根据将项目添加到 BST 的顺序，我们可能会得到一个与列表一样低效的搜索树。也就是说，树中的每个节点都只有一个子节点。为了避免这种情况，我们需要确保我们的 BST 是自平衡的。

# 符号表

符号表由编译器和解释器使用，用于跟踪已声明的符号并保留有关它们的信息。符号表通常使用哈希表构建，因为从表中高效地检索符号很重要。

让我们看一个例子。假设我们有以下 Python 代码：

```py
    name = "Joe" 
    age = 27 
```

在这里，我们有两个符号，`name`和`age`。它们属于一个命名空间，可以是`__main__`，但如果您将其放在那里，它也可以是模块的名称。每个符号都有一个`value`；例如，`name`符号的值是`Joe`，`age`符号的值是`27`。符号表允许编译器或解释器查找这些值。因此，`name`和`age`符号成为哈希表中的键。与它们关联的所有其他信息成为符号表条目的`value`。

不仅变量是符号，函数和类也被视为符号，并且它们也将被添加到符号表中，以便在需要访问它们时，可以从符号表中访问。例如，`greet()`函数和两个变量存储在以下图表中的符号表中：

![](img/58c18771-70c0-454c-91fc-514e05854547.png)

在 Python 中，每个加载的模块都有自己的符号表。符号表以该模块的名称命名。这样，模块就充当了命名空间。只要它们存在于不同的符号表中，我们可以拥有相同名称的多个符号，并且可以通过适当的符号表访问它们。请参见以下示例，显示程序中的多个符号表： 

![](img/161c04e4-7336-4910-8a73-47ec4d89feba.png)

# 总结

在本章中，我们研究了哈希表。我们研究了如何编写一个哈希函数将字符串数据转换为整数数据。然后，我们研究了如何使用哈希键快速高效地查找与键对应的值。

另外，我们还研究了哈希表实现中由于哈希值冲突而产生的困难。这导致我们研究了冲突解决策略，因此我们讨论了两种重要的冲突解决方法，即线性探测和链表法。

在本章的最后一节中，我们研究了符号表，它们通常是使用哈希表构建的。符号表允许编译器或解释器查找已定义的符号（如变量、函数或类）并检索有关它们的所有信息。

在下一章中，我们将讨论图和其他算法。