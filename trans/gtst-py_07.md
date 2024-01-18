# 列表和指针结构

你已经在Python中看到了列表。它们方便而强大。通常，每当你需要在列表中存储东西时，你使用Python的内置列表实现。然而，在本章中，我们更感兴趣的是理解列表的工作原理。因此，我们将研究列表的内部。正如你将注意到的，有不同类型的列表。

Python的列表实现旨在强大并包含几种不同的用例。我们将对列表的定义更加严格。节点的概念对列表非常重要。我们将在本章讨论它们，但这个概念将以不同的形式在本书的其余部分中再次出现。

本章的重点将是以下内容：

+   了解Python中的指针

+   处理节点的概念

+   实现单向、双向和循环链表

在本章中，我们将处理相当多的指针。因此，提醒自己这些是有用的。首先，想象一下你有一所房子想要出售。由于时间不够，你联系了一个中介来寻找感兴趣的买家。所以你拿起你的房子，把它带到中介那里，中介会把房子带给任何可能想要购买它的人。你觉得荒谬吗？现在想象一下你有一些Python函数，用于处理图像。所以你在函数之间传递高分辨率图像数据。

当然，你不会把你的房子随身携带。你会把房子的地址写在一张废纸上，交给中介。房子还在原地，但包含房子方向的纸条在传递。你甚至可能在几张纸上写下来。每张纸都足够小，可以放在钱包里，但它们都指向同一所房子。

事实证明，在Python领域并没有太大的不同。那些大型图像文件仍然在内存中的一个地方。你所做的是创建变量，保存这些图像在内存中的位置。这些变量很小，可以在不同的函数之间轻松传递。

这就是指针的巨大好处：它们允许你用简单的内存地址指向潜在的大内存段。

指针存在于计算机的硬件中，被称为间接寻址。

在Python中，你不直接操作指针，不像其他一些语言，比如C或Pascal。这导致一些人认为Python中不使用指针。这是大错特错。考虑一下在Python交互式shell中的这个赋值：

```py
    >>> s = set()
```

我们通常会说`s`是set类型的变量。也就是说，`s`是一个集合。然而，这并不严格正确。变量`s`实际上是一个引用（一个“安全”的指针）指向一个集合。集合构造函数在内存中创建一个集合，并返回该集合开始的内存位置。这就是存储在`s`中的内容。

Python将这种复杂性隐藏起来。我们可以安全地假设`s`是一个集合，并且一切都运行正常。

# 数组

数组是数据的顺序列表。顺序意味着每个元素都存储在前一个元素的后面。如果你的数组非常大，而且内存不足，可能找不到足够大的存储空间来容纳整个数组。这将导致问题。

当然，硬币的另一面是数组非常快。由于每个元素都紧随前一个元素在内存中，不需要在不同的内存位置之间跳转。在选择列表和数组在你自己的实际应用程序中时，这可能是一个非常重要的考虑因素。

# 指针结构

与数组相反，指针结构是可以在内存中分散的项目列表。这是因为每个项目包含一个或多个链接到结构中其他项目的链接。这些链接的类型取决于我们拥有的结构类型。如果我们处理的是链表，那么我们将有链接到结构中下一个（可能是上一个）项目的链接。在树的情况下，我们有父子链接以及兄弟链接。在基于瓦片的游戏中，游戏地图由六边形构建，每个节点将链接到最多六个相邻的地图单元。

指针结构有几个好处。首先，它们不需要顺序存储空间。其次，它们可以从小开始，随着向结构中添加更多节点而任意增长。

然而，这是有代价的。如果你有一个整数列表，每个节点将占据一个整数的空间，以及额外的整数用于存储指向下一个节点的指针。

# 节点

在列表（以及其他几种数据结构）的核心是节点的概念。在我们进一步之前，让我们考虑一下这个想法。

首先，我们将创建一些字符串：

```py
>>> a = "eggs"
>>> b = "ham"
>>> c = "spam"
```

现在你有三个变量，每个变量都有一个唯一的名称、类型和值。我们没有的是一种方法来说明变量之间的关系。节点允许我们这样做。节点是数据的容器，以及一个或多个指向其他节点的链接。链接是一个指针。

一个简单类型的节点只有一个指向下一个节点的链接。

当然，根据我们对指针的了解，我们意识到这并不完全正确。字符串并没有真正存储在节点中，而是指向实际字符串的指针：

![](assets/f0120953-dde0-440a-b6e4-c435d9aaa949.jpg)

因此，这个简单节点的存储需求是两个内存地址。节点的数据属性是指向字符串`eggs`和`ham`的指针。

# 查找终点

我们创建了三个节点：一个包含**eggs**，一个**ham**，另一个**spam**。**eggs**节点指向**ham**节点，**ham**节点又指向**spam**节点。但**spam**节点指向什么？由于这是列表中的最后一个元素，我们需要确保它的下一个成员有一个清晰的值。

如果我们使最后一个元素指向空，则我们使这一事实清楚。在Python中，我们将使用特殊值`None`来表示空：

![](assets/51d9ca4b-c0d8-4d9b-96fc-ff09f98627c3.jpg)

最后一个节点的下一个指针指向None。因此它是节点链中的最后一个节点。

# 节点

这是我们迄今为止讨论的一个简单节点实现：

```py
    class Node: 
        def __init__(self, data=None): 
            self.data = data 
            self.next = None 
```

不要将节点的概念与Node.js混淆，Node.js是一种使用JavaScript实现的服务器端技术。

`next`指针被初始化为`None`，这意味着除非你改变`next`的值，否则节点将成为一个终点。这是一个好主意，这样我们就不会忘记正确终止列表。

你可以根据需要向`node`类添加其他内容。只需记住节点和数据之间的区别。如果你的节点将包含客户数据，那么创建一个`Customer`类并将所有数据放在那里。

你可能想要实现`__str__`方法，这样当节点对象传递给print时，它调用包含对象的`__str__`方法：

```py
    def __str__(self): 
        return str(data) 
```

# 其他节点类型

我们假设节点具有指向下一个节点的指针。这可能是最简单的节点类型。然而，根据我们的要求，我们可以创建许多其他类型的节点。

有时我们想从A到B，但同时也想从B到A。在这种情况下，我们除了下一个指针外还添加了一个前一个指针：

![](assets/124fe45b-0d55-4db6-b881-2ee91e2ed608.jpg)

从图中可以看出，我们让最后一个节点和第一个节点都指向“None”，表示我们已经到达它们作为列表端点的边界。第一个节点的前指针指向None，因为它没有前任，就像最后一个项目的后指针指向“None”一样，因为它没有后继节点。

您可能还在为基于瓦片的游戏创建瓦片。在这种情况下，您可能使用北、南、东和西代替前一个和后一个。指针的类型更多，但原理是相同的。地图末尾的瓦片将指向“None”：

![](assets/117da9a1-faa1-4324-a90b-72c8d7041c7f.jpg)

您可以根据需要扩展到任何程度。如果您需要能够向西北、东北、东南和西南移动，您只需将这些指针添加到您的“node”类中。

# 单链表

单链表是一个只有两个连续节点之间的指针的列表。它只能以单个方向遍历，也就是说，您可以从列表中的第一个节点移动到最后一个节点，但不能从最后一个节点移动到第一个节点。

实际上，我们可以使用之前创建的“node”类来实现一个非常简单的单链表：

```py
    >>> n1 = Node('eggs')
    >>> n2 = Node('ham')
    >>> n3 = Node('spam')
```

接下来，我们将节点链接在一起，使它们形成一个*链*：

```py
    >>> n1.next = n2
    >>> n2.next = n3
```

要遍历列表，您可以执行以下操作。我们首先将变量“current”设置为列表中的第一个项目：

```py
    current = n1
    while current:
        print(current.data)
        current = current.next 
```

在循环中，我们打印当前元素，然后将当前设置为指向列表中的下一个元素。我们一直这样做，直到我们到达列表的末尾。

但是，这种简单的列表实现存在几个问题：

+   程序员需要太多的手动工作

+   这太容易出错了（这是第一个问题的结果）

+   列表的内部工作方式对程序员暴露得太多

我们将在以下部分解决所有这些问题。

# 单链表类

列表显然是一个与节点不同的概念。因此，我们首先创建一个非常简单的类来保存我们的列表。我们将从一个持有对列表中第一个节点的引用的构造函数开始。由于此列表最初为空，因此我们将首先将此引用设置为“None”：

```py
    class SinglyLinkedList:
         def __init__(self):
             self.tail = None 
```

# 附加操作

我们需要执行的第一个操作是向列表附加项目。这个操作有时被称为插入操作。在这里，我们有机会隐藏“Node”类。我们的“list”类的用户实际上不应该与Node对象进行交互。这些纯粹是内部使用。

第一次尝试“append()”方法可能如下所示：

```py
    class SinglyLinkedList:
         # ...

         def append(self, data):
             # Encapsulate the data in a Node
             node = Node(data)

             if self.tail == None:
                 self.tail = node
             else:
                 current = self.tail
                 while current.next:
                     current = current.next
                 current.next = node 
```

我们将数据封装在一个节点中，因此它现在具有下一个指针属性。从这里开始，我们检查列表中是否存在任何现有节点（即“self.tail”指向一个节点）。如果没有，我们将新节点设置为列表的第一个节点；否则，通过遍历列表找到插入点，将最后一个节点的下一个指针更新为新节点。

我们可以附加一些项目：

```py
>>> words = SinglyLinkedList()
 >>> words.append('egg')
 >>> words.append('ham')
 >>> words.append('spam')
```

列表遍历将更多或更少地像以前一样工作。您将从列表本身获取列表的第一个元素：

```py
>>> current = words.tail
>>> while current:
        print(current.data) 
        current = current.next
```

# 更快的附加操作

在上一节中，附加方法存在一个大问题：它必须遍历整个列表以找到插入点。当列表中只有几个项目时，这可能不是问题，但等到您需要添加成千上万个项目时再等等。每次附加都会比上一次慢一点。一个**O**(n)证明了我们当前的“append”方法实际上会有多慢。

为了解决这个问题，我们将存储的不仅是列表中第一个节点的引用，还有最后一个节点的引用。这样，我们可以快速地在列表的末尾附加一个新节点。附加操作的最坏情况运行时间现在从**O**(n)减少到**O**(1)。我们所要做的就是确保前一个最后一个节点指向即将附加到列表中的新节点。以下是我们更新后的代码：

```py
    class SinglyLinkedList:
         def __init__(self): 
             # ...
             self.tail = None

         def append(self, data):
            node = Node(data)
            if self.head:
                self.head.next = node
                self.head = node
            else:
                self.tail = node
                self.head = node 
```

注意正在使用的约定。我们附加新节点的位置是通过`self.head`。`self.tail`变量指向列表中的第一个节点。

# 获取列表的大小

我们希望通过计算节点数来获取列表的大小。我们可以通过遍历整个列表并在遍历过程中增加一个计数器来实现这一点：

```py
    def size(self):
         count = 0
         current = self.tail
         while current:
             count += 1
             current = current.next
         return count 
```

这样做是可以的，但列表遍历可能是一个昂贵的操作，我们应该尽量避免。因此，我们将选择另一种重写方法。我们在`SinglyLinkedList`类中添加一个size成员，在构造函数中将其初始化为0。然后我们在`append`方法中将size增加一：

```py
class SinglyLinkedList:
     def __init__(self):
         # ...
         self.size = 0

     def append(self, data):
         # ...
         self.size += 1 
```

因为我们现在只读取节点对象的size属性，而不使用循环来计算列表中的节点数，所以我们可以将最坏情况的运行时间从**O**(n)减少到**O**(1)。

# 改进列表遍历

如果您注意到我们如何遍历我们的列表。那里我们仍然暴露给`node`类的地方。我们需要使用`node.data`来获取节点的内容和`node.next`来获取下一个节点。但我们之前提到客户端代码不应该需要与Node对象进行交互。我们可以通过创建一个返回生成器的方法来实现这一点。它看起来如下：

```py
    def iter(self):
        current = self.tail
        while current:
            val = current.data
            current = current.next
            yield val  
```

现在列表遍历变得简单得多，看起来也好得多。我们可以完全忽略列表之外有一个叫做Node的东西：

```py
    for word in words.iter():
        print(word) 
```

请注意，由于`iter()`方法产生节点的数据成员，我们的客户端代码根本不需要担心这一点。

# 删除节点

列表上的另一个常见操作是删除节点。这可能看起来很简单，但我们首先必须决定如何选择要删除的节点。是按索引号还是按节点包含的数据？在这里，我们将选择按节点包含的数据删除节点。

以下是从列表中删除节点时考虑的一个特殊情况的图示：

![](assets/681a0c61-f4fc-4b58-bd9d-ff3ade3c9d61.jpg)

当我们想要删除两个其他节点之间的节点时，我们所要做的就是将前一个节点直接指向其下一个节点的后继节点。也就是说，我们只需像前面的图像中那样将要删除的节点从链中切断。

以下是`delete()`方法的实现可能是这样的：

```py
    def delete(self, data):
        current = self.tail
        prev = self.tail
        while current:
            if current.data == data:
                if current == self.tail:
                    self.tail = current.next
                else:
                    prev.next = current.next
                self.size -= 1
                return
            prev = current
            current = current.next 
```

删除节点应该需要**O**(n)的时间。

# 列表搜索

我们可能还需要一种方法来检查列表是否包含某个项目。由于我们之前编写的`iter()`方法，这种方法实现起来相当容易。循环的每一次通过都将当前数据与正在搜索的数据进行比较。如果找到匹配项，则返回`True`，否则返回`False`：

```py
def search(self, data):
     for node in self.iter():
         if data == node:
             return True
     return False  
```

# 清空列表

我们可能希望快速清空列表。幸运的是，这非常简单。我们只需将指针`head`和`tail`设置为`None`即可：

```py
def clear(self): 
       """ Clear the entire list. """ 
       self.tail = None 
       self.head = None 
```

一举两得，我们将列表的`tail`和`head`指针上的所有节点都变成了孤立的。这会导致中间所有的节点都变成了孤立的。

# 双向链表

现在我们对单向链表有了扎实的基础，知道了可以对其执行的操作类型，我们现在将把注意力转向更高一级的双向链表主题。

双向链表在某种程度上类似于单向链表，因为我们利用了将节点串联在一起的相同基本思想。在单向链表中，每个连续节点之间存在一个链接。双向链表中的节点有两个指针：指向下一个节点和指向前一个节点的指针：

![](assets/48886177-4080-49b0-bc7b-0a8a995fc324.jpg)

单向链表中的节点只能确定与其关联的下一个节点。但是被引用的节点或下一个节点无法知道是谁在引用它。方向的流动是**单向的**。

在双向链表中，我们为每个节点添加了不仅引用下一个节点而且引用前一个节点的能力。

让我们检查一下两个连续节点之间存在的连接性质，以便更好地理解：

![](assets/03bff824-7bea-410e-acb8-a1d607bca4db.jpg)

由于存在指向下一个和前一个节点的两个指针，双向链表具有某些能力。

双向链表可以在任何方向遍历。根据正在执行的操作，双向链表中的节点可以在必要时轻松地引用其前一个节点，而无需指定变量来跟踪该节点。因为单向链表只能在一个方向上遍历，有时可能意味着移动到列表的开始或开头，以便影响列表中隐藏的某些更改。

由于立即可以访问下一个和前一个节点，删除操作要容易得多，后面在本章中会看到。

# 双向链表节点

创建一个类来捕获双向链表节点的Python代码，在其初始化方法中包括`prev`、`next`和`data`实例变量。当新创建一个节点时，所有这些变量默认为`None`：

```py
    class Node(object): 
        def __init__(self, data=None, next=None, prev=None): 
           self.data = data 
           self.next = next 
           self.prev = prev 
```

`prev`变量保存对前一个节点的引用，而`next`变量继续保存对下一个节点的引用。

# 双向链表

仍然很重要的是创建一个类，以捕获我们的函数将要操作的数据：

```py
    class DoublyLinkedList(object):
       def __init__(self):
           self.head = None
           self.tail = None
           self.count = 0
```

为了增强`size`方法，我们还将`count`实例变量设置为0。当我们开始向列表中插入节点时，`head`和`tail`将指向列表的头部和尾部。

我们采用了一个新的约定，其中`self.head`指向列表的起始节点，而`self.tail`指向列表中最新添加的节点。这与我们在单向链表中使用的约定相反。关于头部和尾部节点指针的命名没有固定的规则。

双向链表还需要提供返回列表大小、插入列表和从列表中删除节点的函数。我们将检查一些执行此操作的代码。让我们从`append`操作开始。

# 追加操作

在`append`操作期间，重要的是检查`head`是否为`None`。如果是`None`，则意味着列表为空，并且应该将`head`设置为指向刚创建的节点。通过头部，列表的尾部也指向新节点。在这一系列步骤结束时，`head`和`tail`现在将指向同一个节点：

```py
    def append(self, data): 
        """ Append an item to the list. """ 

           new_node = Node(data, None, None) 
           if self.head is None: 
               self.head = new_node 
               self.tail = self.head 
           else: 
               new_node.prev = self.tail 
               self.tail.next = new_node 
               self.tail = new_node 

               self.count += 1 
```

以下图表说明了在向空列表添加新节点时，双向链表的头部和尾部指针。

![](assets/4726485a-16d3-40e4-ae32-b63afef3b054.jpg)

算法的`else`部分仅在列表不为空时执行。新节点的前一个变量设置为列表的尾部：

```py
    new_node.prev = self.tail 
```

尾部的下一个指针（或变量）设置为新节点：

```py
    self.tail.next = new_node 
```

最后，我们更新尾部指针指向新节点：

```py
    self.tail = new_node 
```

由于`append`操作将节点数增加了一个，我们将计数器增加了一个：

```py
    self.count += 1 
```

`append`操作的视觉表示如下：

![](assets/e7446f9f-dddb-4220-99f4-182bc7fe4416.jpg)

# 删除操作

与单向链表不同，我们需要在遍历整个列表的时候跟踪先前遇到的节点，双向链表避免了这一步。这是通过使用前一个指针实现的。

从双向链表中删除节点的算法在完成节点删除之前，为基本上四种情况提供了支持。这些是：

+   当根本找不到搜索项时

+   当搜索项在列表的开头找到时

+   当搜索项在列表的尾部找到时

+   当搜索项在列表的中间找到时

当其`data`实例变量与传递给用于搜索节点的方法的数据匹配时，将识别要移除的节点。如果找到匹配的节点并随后删除，则将变量`node_deleted`设置为`True`。任何其他结果都会导致`node_deleted`被设置为`False`：

```py
    def delete(self, data): 
        current = self.head 
        node_deleted = False 
        ...    
```

在`delete`方法中，`current`变量被设置为列表的头部（即指向列表的`self.head`）。然后使用一组`if...else`语句搜索列表的各个部分，以找到具有指定数据的节点。

首先搜索`head`节点。由于`current`指向`head`，如果`current`为None，则假定列表没有节点，甚至无法开始搜索要删除的节点：

```py
    if current is None: 
        node_deleted = False     
```

然而，如果`current`（现在指向头部）包含正在搜索的数据，那么`self.head`被设置为指向`current`的下一个节点。由于现在头部后面没有节点了，`self.head.prev`被设置为`None`：

```py
    elif current.data == data: 
        self.head = current.next 
        self.head.prev = None 
        node_deleted = True 
```

如果要删除的节点位于列表的尾部，将采用类似的策略。这是第三个语句，搜索要删除的节点可能位于列表末尾的可能性：

```py
    elif self.tail.data == data: 
        self.tail = self.tail.prev 
        self.tail.next = None 
        node_deleted = True 
```

最后，查找并删除节点的算法循环遍历节点列表。如果找到匹配的节点，`current`的前一个节点将连接到`current`的下一个节点。在这一步之后，`current`的下一个节点将连接到`current`的前一个节点：

```py
else
    while current: 
        if current.data == data: 
            current.prev.next = current.next 
            current.next.prev = current.prev 
            node_deleted = True 
        current = current.next 
```

然后在评估所有`if-else`语句之后检查`node_delete`变量。如果任何`if-else`语句更改了这个变量，那么意味着从列表中删除了一个节点。因此，计数变量减1：

```py
    if node_deleted: 
        self.count -= 1 
```

作为删除列表中嵌入的节点的示例，假设存在三个节点A、B和C。要删除列表中间的节点B，我们将使A指向C作为它的下一个节点，同时使C指向A作为它的前一个节点：

![](assets/95e53e49-1f8e-4c1c-82ae-950c6bd20d8b.jpg)

在这样的操作之后，我们得到以下列表：

![](assets/290796b6-c426-4da7-81d2-0b4fd7fc3a65.jpg)

# 列表搜索

搜索算法类似于单向链表中`search`方法的算法。我们调用内部方法`iter()`返回所有节点中的数据。当我们循环遍历数据时，每个数据都与传入`contain`方法的数据进行匹配。如果匹配，则返回`True`，否则返回`False`以表示未找到匹配项：

```py
    def contain(self, data): 
        for node_data in self.iter(): 
            if data == node_data: 
                return True 
            return False 
```

我们的双向链表对于`append`操作具有**O**(1)，对于`delete`操作具有**O**(n)。

# 循环列表

循环列表是链表的一种特殊情况。它是一个端点连接的列表。也就是说，列表中的最后一个节点指向第一个节点。循环列表可以基于单向链表和双向链表。对于双向循环链表，第一个节点还需要指向最后一个节点。

在这里，我们将看一个单向循环链表的实现。一旦你掌握了基本概念，实现双向循环链表就应该很简单了。

我们可以重用我们在单链表部分创建的`node`类。事实上，我们也可以重用`SinglyLinkedList`类的大部分部分。因此，我们将专注于循环列表实现与普通单链表不同的方法。

# 附加元素

当我们向循环列表附加一个元素时，我们需要确保新节点指向尾节点。这在以下代码中得到了证明。与单链表实现相比，多了一行额外的代码：

```py
     def append(self, data): 
           node = Node(data) 
           if self.head: 
               self.head.next = node 
               self.head = node 
           else: 
               self.head = node 
               self.tail = node 
           self.head.next = self.tail 
           self.size += 1 
```

# 删除元素

我们可能认为我们可以遵循与附加相同的原则，并确保头部指向尾部。这将给我们以下实现：

```py
   def delete(self, data): 
       current = self.tail 
       prev = self.tail 
       while current: 
           if current.data == data: 
               if current == self.tail: 
                   self.tail = current.next 
                   self.head.next = self.tail 
               else: 
                   prev.next = current.next 
               self.size -= 1 
               return 
           prev = current 
           current = current.next 
```

与以前一样，只有一行需要更改。只有在删除尾节点时，我们需要确保头节点被更新为指向新的尾节点。

然而，这段代码存在一个严重的问题。在循环列表的情况下，我们不能循环直到当前变为`None`，因为那永远不会发生。如果您删除一个现有节点，您不会看到这一点，但是尝试删除一个不存在的节点，您将陷入无限循环。

因此，我们需要找到一种不同的方法来控制`while`循环。我们不能检查当前是否已经到达头部，因为那样它就永远不会检查最后一个节点。但是我们可以使用`prev`，因为它落后于当前一个节点。然而，有一个特殊情况。在第一个循环迭代中，`current`和`prev`将指向同一个节点，即尾节点。我们希望确保循环在这里运行，因为我们需要考虑只有一个节点的情况。更新后的`delete`方法现在如下所示：

```py
def delete(self, data): 
        current = self.tail 
        prev = self.tail 
        while prev == current or prev != self.head: 
            if current.data == data: 
                if current == self.tail: 
                    self.tail = current.next 
                    self.head.next = self.tail 
                else: 
                    prev.next = current.next 
                self.size -= 1 
                return 
            prev = current 
            current = current.next 
```

# 遍历循环列表

您不需要修改`iter()`方法。它对于我们的循环列表可以完美地工作。但是在遍历循环列表时，您需要设置一个退出条件，否则您的程序将陷入循环。以下是一种方法，可以使用计数器变量来实现：

```py
    words = CircularList() 
    words.append('eggs') 
    words.append('ham') 
    words.append('spam') 

    counter = 0 
    for word in words.iter(): 
       print(word) 
       counter += 1 
       if counter > 1000: 
           break 
```

一旦我们打印出1,000个元素，我们就跳出循环。

# 总结

在本章中，我们已经研究了链表。我们研究了构成列表的概念，如节点和指向其他节点的指针。我们实现了在这些类型的列表上发生的主要操作，并看到了它们的最坏情况运行时间是如何比较的。

在下一章中，我们将看两种通常使用列表实现的其他数据结构：栈和队列。
