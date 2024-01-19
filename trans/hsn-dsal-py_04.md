# 列表和指针结构

我们已经在 Python 中讨论了**列表**，它们方便而强大。通常情况下，我们使用 Python 内置的列表实现来存储任何数据。然而，在本章中，我们将了解列表的工作原理，并将研究列表的内部。

Python 的列表实现非常强大，可以包含多种不同的用例。节点的概念在列表中非常重要。我们将在本章讨论它们，并在整本书中引用它们。因此，我们建议读者仔细学习本章的内容。

本章的重点将是以下内容：

+   理解 Python 中的指针

+   理解节点的概念和实现

+   实现单向、双向和循环链表。

# 技术要求

根据本章讨论的概念执行程序将有助于更好地理解它们。我们已经提供了本章中所有程序和概念的源代码。我们还在 GitHub 上提供了完整的源代码文件，链接如下：[`github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter04`](https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter04)。

我们假设您已经在系统上安装了 Python。

# 从一个例子开始

让我们先提醒一下指针的概念，因为我们将在本章中处理它们。首先，想象一下你有一所房子想要卖掉。由于时间不够，你联系了一个中介来寻找感兴趣的买家。所以，你拿起你的房子，把它带到中介那里，中介会把房子带给任何可能想要买它的人。你觉得这很荒谬？现在想象一下你有一些处理图像的 Python 函数。所以，你在这些函数之间传递高分辨率图像数据。

当然，你不会带着你的房子四处走动。你要做的是把房子的地址写在一张废纸上，递给中介。房子还在原地，但包含房子地址的纸条在传递。你甚至可以在几张纸上写下来。每张纸都足够小，可以放在你的钱包里，但它们都指向同一所房子。

事实证明，在 Python 领域情况并没有太大不同。那些大型图像文件仍然在内存中的一个地方。

你要做的是创建变量，保存这些图像在内存中的位置。这些变量很小，可以在不同的函数之间轻松传递。

这就是指针的好处——它们允许你用一个简单的内存地址指向一个潜在的大内存段。

你的计算机硬件中支持指针，这被称为间接寻址。

在 Python 中，你不会直接操作指针，不像其他一些语言，比如 C 或 Pascal。这导致一些人认为 Python 中不使用指针。这是大错特错。考虑一下在 Python 交互式 shell 中的这个赋值：

```py
>>> s = set()
```

通常我们会说`s`是**集合**类型的变量。也就是说，`s`是一个集合。然而，这并不严格正确；变量`s`实际上是一个引用（一个*安全*指针）指向一个集合。集合构造函数在内存中创建一个集合，并返回该集合开始的内存位置。这就是存储在`s`中的内容。Python 隐藏了这种复杂性。我们可以安全地假设`s`是一个集合，一切都运行正常。

# 数组

数组是一系列数据的顺序列表。顺序意味着每个元素都存储在前一个元素的后面。如果你的数组非常大，而且内存不足，可能无法找到足够大的存储空间来容纳整个数组。这将导致问题。

当然，硬币的另一面是数组非常快速。由于每个元素在内存中紧随前一个元素，因此无需在不同的内存位置之间跳转。在选择在你自己的现实世界应用程序中列表和数组之间时，这可能是一个非常重要的考虑因素。

我们已经在第二章中讨论了数组，*Python 数据类型和结构*。我们看了数组数据类型，并讨论了可以对其执行的各种操作。

# 指针结构

与数组相反，指针结构是可以在内存中分散的项目列表。这是因为每个项目都包含一个或多个指向结构中其他项目的链接。这些链接的类型取决于我们拥有的结构类型。如果我们处理的是链表，那么我们将有指向结构中下一个（可能是上一个）项目的链接。在树的情况下，我们有父子链接以及兄弟链接。

指针结构有几个好处。首先，它们不需要顺序存储空间。其次，它们可以从小开始，随着向结构添加更多节点而任意增长。然而，指针的这种灵活性是有代价的。我们需要额外的空间来存储地址。例如，如果你有一个整数列表，每个节点都将占用空间来存储一个整数，以及额外的整数来存储指向下一个节点的指针。

# 节点

在列表（以及其他几种数据结构）的核心是节点的概念。在我们进一步讨论之前，让我们考虑一下这个想法。

首先，让我们考虑一个例子。我们将创建一些字符串：

```py
>>> a = "eggs"
>>> b = "ham"
>>> c = "spam"
```

现在你有了三个变量，每个变量都有一个唯一的名称、类型和值。目前，没有办法显示这些变量之间的关系。节点允许我们展示这些变量之间的关系。节点是数据的容器，以及一个或多个指向其他节点的链接。链接就是指针。

一种简单类型的节点只有一个指向下一个节点的链接。正如我们所知道的指针，字符串实际上并没有存储在节点中，而是有一个指向实际字符串的指针。考虑下面的图表中的例子，其中有两个节点。第一个节点有一个指向存储在内存中的字符串（**eggs**）的指针，另一个指针存储着另一个节点的地址：

![](img/0b94a5a7-7532-473e-91a9-06287979287c.png)

因此，这个简单节点的存储需求是两个内存地址。节点的数据属性是指向字符串**eggs**和**ham**的指针。

# 查找端点

我们已经创建了三个节点——一个包含**eggs**，一个**ham**，另一个**spam**。**eggs**节点指向**ham**节点，**ham**节点又指向**spam**节点。但是**spam**节点指向什么呢？由于这是列表中的最后一个元素，我们需要确保它的下一个成员有一个清晰的值。

如果我们使最后一个元素指向空，那么我们就清楚地表明了这一事实。在 Python 中，我们将使用特殊值**None**来表示空。考虑下面的图表。节点**B**是列表中的最后一个元素，因此它指向**None**：

![](img/938a3e9b-a0e1-4b54-bfbf-149bcfe6f360.png)

最后一个节点的下一个指针指向**None**。因此，它是节点链中的最后一个节点。

# 节点类

这是我们迄今为止讨论的一个简单节点实现：

```py
class Node:

    def __init__ (self, data=None):
        self.data = data 
        self.next = None
```

**Next**指针初始化为`None`，这意味着除非你改变**Next**的值，否则节点将成为一个端点。这是一个很好的主意，这样我们就不会忘记正确终止列表。

你可以根据需要向节点类添加其他内容。只要记住节点和数据之间的区别。如果你的节点将包含客户数据，那么创建一个`Customer`类，并把所有数据放在那里。

您可能想要做的一件事是实现`_str_`方法，以便在将节点对象传递给打印时调用所包含对象的`_str_`方法：

```py
def _str_ (self):
   return str(data)
```

# 其他节点类型

正如我们已经讨论过的，一个节点具有指向下一个节点的指针来链接数据项，但它可能是最简单的节点类型。此外，根据我们的需求，我们可以创建许多其他类型的节点。

有时我们想从节点**A**到节点**B**，但同时我们可能需要从节点**B**到节点**A**。在这种情况下，我们除了**Next**指针之外还添加了**Previous**指针：

![](img/70762d9f-ef9c-4282-bbc4-e7d6dca8689b.png)

从上图可以看出，我们除了数据和**Next**指针之外，还创建了**Previous**指针。还需要注意的是，**B**的**Next**指针是**None**，而节点**A**的**Previous**指针也是**None**，这表示我们已经到达了列表的边界。第一个节点**A**的前指针指向**None**，因为它没有前驱，就像最后一个项目**B**的**Next**指针指向**None**一样，因为它没有后继节点。

# 引入列表

列表是一个重要且流行的数据结构。列表有三种类型——单链表、双链表和循环链表。我们将在本章更详细地讨论这些数据结构。我们还将在接下来的小节中讨论各种重要操作，如`append`操作、`delete`操作以及可以在这些列表上执行的`traversing`和`searching`操作。

# 单链表

单链表是一种只有两个连续节点之间的指针的列表。它只能以单个方向遍历；也就是说，您可以从列表中的第一个节点到最后一个节点，但不能从最后一个节点移动到第一个节点。

实际上，我们可以使用之前创建的节点类来实现一个非常简单的单链表。例如，我们创建三个存储三个字符串的节点`n1`、`n2`和`n3`：

```py
>>> n1 = Node('eggs')
>>> n2 = Node('ham')
>>> n3 = Node('spam')
```

接下来，我们将节点链接在一起，形成一个链：

```py
>>> n1.next = n2
>>> n2.next = n3
```

要遍历列表，您可以像下面这样做。我们首先将`current`变量设置为列表中的第一个项目，然后通过循环遍历整个列表，如下面的代码所示：

```py
current = n1  
while current:
     print(current.data)
     current = current.next
```

在循环中，我们打印出当前元素，然后将`current`设置为指向列表中的下一个元素。我们一直这样做，直到我们到达列表的末尾。

然而，这种简单的列表实现存在几个问题：

+   程序员需要做太多的手动工作

+   这太容易出错了（这是第一点的结果）

+   列表的内部工作过于暴露给程序员

我们将在接下来的章节中解决所有这些问题。

# 单链表类

列表是一个与节点不同的概念。我们首先创建一个非常简单的类来保存我们的列表。我们从一个构造函数开始，它保存对列表中第一个节点的引用（在下面的代码中是`tail`）。由于这个列表最初是空的，我们将首先将这个引用设置为`None`：

```py
class SinglyLinkedList:
    def __init__ (self):
        self.tail = None
```

# 追加操作

我们需要执行的第一个操作是向列表追加项目。这个操作有时被称为插入操作。在这里，我们有机会隐藏`Node`类。我们的列表类的用户实际上不应该与`Node`对象交互。这些纯粹是内部使用的。

第一次尝试`append()`方法可能如下所示：

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

我们封装数据在一个节点中，以便它具有下一个指针属性。从这里开始，我们检查列表中是否存在任何现有节点（即`self.tail`是否指向一个`Node`）。如果是`None`，我们将新节点设置为列表的第一个节点；否则，我们通过遍历列表找到插入点，将最后一个节点的下一个指针更新为新节点。

考虑以下示例代码以追加三个节点：

```py
>>> words = SinglyLinkedList()
>>> words.append('egg')
>>> words.append('ham')
>>> words.append('spam')
```

列表遍历将按照我们之前讨论的方式进行。您将从列表本身获取列表的第一个元素，然后通过`next`指针遍历列表：

```py
>>> current = words.tail
>>> while current:
        print(current.data)
        current = current.next
```

# 更快的追加操作

在前一节中，追加方法存在一个大问题：它必须遍历整个列表以找到插入点。当列表中只有一些项目时，这可能不是问题，但当列表很长时，这将是一个大问题，因为我们需要每次遍历整个列表来添加一个项目。每次追加都会比上一次略慢。追加操作的当前实现速度降低了`O(n)`，这在长列表的情况下是不可取的。

为了解决这个问题，我们不仅存储了对列表中第一个节点的引用，还存储了对最后一个节点的引用。这样，我们可以快速地在列表的末尾追加一个新节点。追加操作的最坏情况运行时间现在从`O(n)`降低到了`O(1)`。我们所要做的就是确保前一个最后一个节点指向即将追加到列表中的新节点。以下是我们更新后的代码：

```py
class SinglyLinkedList:
    def init (self):
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

请注意正在使用的约定。我们追加新节点的位置是通过`self.head`。`self.tail`变量指向列表中的第一个节点。

# 获取列表的大小

我们希望能够通过计算节点的数量来获取列表的大小。我们可以通过遍历整个列表并在遍历过程中增加一个计数器来实现这一点：

```py
def size(self):
 count = 0
 current = self.tail
 while current:
     count += 1
     current = current.next 
 return count
```

这很好用。但是，列表遍历可能是一个昂贵的操作，我们应该尽量避免。因此，我们将选择另一种重写方法。我们在`SinglyLinkedList`类中添加一个 size 成员，在构造函数中将其初始化为`0`。然后我们在追加方法中将 size 增加一：

```py
class SinglyLinkedList:
    def init (self):
        # ...
```

```py

        self.size = 0

    def append(self, data):
        # ...
        self.size += 1
```

因为我们现在只是读取节点对象的 size 属性，而不是使用循环来计算列表中节点的数量，所以我们将最坏情况的运行时间从`O(n)`降低到了`O(1)`。

# 改进列表遍历

如果您注意到，在列表遍历的早期，我们向客户/用户公开了节点类。但是，希望客户端节点不要与节点对象进行交互。我们需要使用`node.data`来获取节点的内容，使用`node.next`来获取下一个节点。我们可以通过创建一个返回生成器的方法来访问数据。如下所示：

```py
def iter(self):
    current = self.tail 
    while current:
        val = current.data 
        current = current.next 
        yield val
```

现在，列表遍历变得简单得多，看起来也好得多。我们可以完全忽略列表之外有一个叫做节点的东西：

```py
for word in words.iter():
    print(word)
```

注意，由于`iter()`方法产生节点的数据成员，我们的客户端代码根本不需要担心这一点。

# 删除节点

您将在列表上执行的另一个常见操作是删除节点。这可能看起来很简单，但我们首先必须决定如何选择要删除的节点。它是由索引号还是由节点包含的数据来确定的？在这里，我们将选择根据节点包含的数据来删除节点。

以下是在从列表中删除节点时考虑的特殊情况的图示：

![](img/9a448606-c04c-4256-babb-acae72ff10b6.png)

当我们想要删除两个节点之间的一个节点时，我们所要做的就是使前一个节点指向其下一个要删除的节点的后继节点。也就是说，我们只需将要删除的节点从链表中切断，并直接指向下一个节点，如前面的图所示。

`delete()`方法的实现可能如下所示：

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
            self.count -= 1
            return
        prev = current
        current = current.next
```

删除节点的`delete`操作的时间复杂度为`O(n)`。

# 列表搜索

我们可能还需要一种方法来检查列表是否包含某个项目。由于我们之前编写的`iter()`方法，这种方法非常容易实现。循环的每次通过将当前数据与正在搜索的数据进行比较。如果找到匹配项，则返回`True`，否则返回`False`：

```py
def search(self, data):
    for node in self.iter():
        if data == node:
            return True 
    return False
```

# 清除列表

我们可能需要快速清除列表；有一种非常简单的方法可以做到。我们可以通过简单地将指针头和尾清除为`None`来清除列表：

```py
def clear(self):
    """ Clear the entire list. """
    self.tail = None
    self.head = None
```

# 双向链表

我们已经讨论了单链表以及可以在其上执行的重要操作。现在，我们将在本节中专注于双向链表的主题。

双向链表与单链表非常相似，因为我们使用了将字符串节点串在一起的相同基本概念，就像在单链表中所做的那样。单链表和双链表之间唯一的区别在于，在单链表中，每个连续节点之间只有一个链接，而在双链表中，我们有两个指针——一个指向下一个节点，一个指向前一个节点。请参考以下*节点*的图表；有一个指向下一个节点和前一个节点的指针，它们设置为`None`，因为没有节点连接到这个节点。考虑以下图表：

![](img/9f1ca24a-6fc0-4c9e-9668-ca3fa2df2ee0.png)

单链表中的节点只能确定与其关联的下一个节点。然而，没有办法或链接可以从这个引用节点返回。流动的方向只有一种。

在双向链表中，我们解决了这个问题，并且不仅可以引用下一个节点，还可以引用前一个节点。考虑以下示例图表，以了解两个连续节点之间链接的性质。这里，节点**A**引用节点**B**；此外，还有一个链接返回到节点**A**：

![](img/be75f67d-d6ff-420d-b5e0-1cd2ca679be2.png)

由于存在指向下一个和前一个节点的两个指针，双向链表具有某些功能。

双向链表可以在任何方向进行遍历。在双向链表中，可以很容易地引用节点的前一个节点，而无需使用变量来跟踪该节点。然而，在单链表中，可能难以返回到列表的开始或开头，以便在列表的开头进行一些更改，而在双向链表的情况下现在非常容易。

# 双向链表节点

创建双向链表节点的 Python 代码包括其初始化方法、`prev`指针、`next`指针和`data`实例变量。当新建一个节点时，所有这些变量默认为`None`：

```py
class Node(object):
    def __init__ (self, data=None, next=None, prev=None):
       self.data = data 
       self.next = next 
       self.prev = prev
```

`prev`变量引用前一个节点，而`next`变量保留对下一个节点的引用，`data`变量存储数据。

# 双向链表类

双向链表类捕获了我们的函数将要操作的数据。对于`size`方法，我们将计数实例变量设置为`0`；它可以用来跟踪链表中的项目数量。当我们开始向列表中插入节点时，`head`和`tail`将指向列表的头部和尾部。考虑以下用于创建类的 Python 代码：

```py
class DoublyLinkedList(object):
    def init (self): 
        self.head = None
        self.tail = None
        self.count = 0
```

我们采用了一个新的约定，其中`self.head`指向列表的起始节点，而`self.tail`指向添加到列表中的最新节点。这与我们在单链表中使用的约定相反。关于头部和尾部节点指针的命名没有固定的规则。

双链表还需要返回列表大小、向列表中插入项目以及从列表中删除节点的功能。我们将在以下子部分中讨论并提供关于双链表的重要功能和代码。让我们从附加操作开始。

# 附加操作

`append`操作用于在列表的末尾添加元素。重要的是要检查列表的`head`是否为`None`。如果是`None`，则表示列表为空，否则列表有一些节点，并且将向列表添加一个新节点。如果要向空列表添加新节点，则应将`head`指向新创建的节点，并且列表的尾部也应通过`head`指向该新创建的节点。经过这一系列步骤，头部和尾部现在将指向同一个节点。以下图示了当向空列表添加新节点时，双链表的`head`和`tail`指针：

![](img/39e8e53b-5a79-4ba8-912d-6fe36f8871d5.png)

以下代码用于将项目附加到双链表：

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

上述程序的`If`部分用于将节点添加到空节点；如果列表不为空，则将执行上述程序的`else`部分。如果要将新节点添加到列表中，则新节点的前一个变量应设置为列表的尾部：

```py
new_node.prev = self.tail
```

尾部的下一个指针（或变量）必须设置为新节点：

```py
self.tail.next = new_node
```

最后，我们更新尾部指针以指向新节点：

```py
self.tail = new_node
```

由于附加操作将节点数增加一，因此我们将计数器增加一：

```py
self.count += 1
```

以下图示了向现有列表附加操作的可视表示：

![](img/65d93f4e-efe7-408f-99c5-e7817f1262f5.png)

# 删除操作

与单链表相比，双链表中的删除操作更容易。

与单链表不同，我们需要在遍历整个列表的整个长度时始终跟踪先前遇到的节点，双链表避免了整个步骤。这是通过使用前一个指针实现的。

在双链表中，`delete`操作可能会遇到以下四种情况：

+   未找到要删除的搜索项在列表中

+   要删除的搜索项位于列表的开头

+   要删除的搜索项位于列表的末尾

+   要删除的搜索项位于列表的中间

要删除的节点是通过将数据实例变量与传递给方法的数据进行匹配来识别的。如果数据与节点的数据变量匹配，则将删除该匹配的节点。以下是从双链表中删除节点的完整代码。我们将逐步讨论此代码的每个部分：

```py
def delete(self, data):
    """ Delete a node from the list. """ 
    current = self.head 
    node_deleted = False 
    if current is None:       #Item to be deleted is not found in the list
        node_deleted = False 

    elif current.data == data:   #Item to be deleted is found at starting of list
        self.head = current.next  
        self.head.prev = None 
        node_deleted = True 

    elif self.tail.data == data:   #Item to be deleted is found at the end of list.
        self.tail = self.tail.prev  
        self.tail.next = None 
        node_deleted = True 
    else: 
        while current:          #search item to be deleted, and delete that node
            if current.data == data: 
                current.prev.next = current.next  
                current.next.prev = current.prev 
                node_deleted = True 
            current = current.next 

    if node_deleted: 
        self.count -= 1
```

最初，我们创建一个`node_deleted`变量来表示列表中被删除的节点，并将其初始化为`False`。如果找到匹配的节点并随后删除，则将`node_deleted`变量设置为`True`。在删除方法中，`current`变量最初设置为列表的`head`（即指向列表的`self.head`）。请参阅以下代码片段：

```py
def delete(self, data): 
    current = self.head 
    node_deleted = False
    ...
```

接下来，我们使用一组`if...else`语句来搜索列表的各个部分，找出具有指定数据的节点，该节点将被删除。

首先，我们在`head`节点处搜索要删除的数据，如果在`head`节点处匹配数据，则将删除该节点。由于`current`指向`head`，如果`current`为`None`，则表示列表为空，没有节点可以找到要删除的节点。以下是其代码片段：

```py
if current is None:
  node_deleted = False
```

但是，如果`current`（现在指向头部）包含正在搜索的数据，这意味着我们在`head`节点找到了要删除的数据，那么`self.head`被标记为指向`current`节点。由于现在`head`后面没有节点了，`self.head.prev`被设置为`None`。考虑以下代码片段：

```py
elif current.data == data: 
    self.head = current.next 
    self.head.prev = None
    node_deleted = True
```

同样，如果要删除的节点位于列表的“尾部”，我们通过将其前一个节点指向`None`来删除最后一个节点。这是双向链表中“删除”操作的第三种可能情况，搜索要删除的节点可能在列表末尾找到。`self.tail`被设置为指向`self.tail.prev`，`self.tail.next`被设置为`None`，因为后面没有节点了。考虑以下代码片段：

```py
elif self.tail.data == data:
   self.tail = self.tail.prev 
   self.tail.next = None
   node_deleted = True
```

最后，我们通过循环整个节点列表来搜索要删除的节点。如果要删除的数据与节点匹配，则删除该节点。要删除节点，我们使用代码`current.prev.next = current.next`使`current`节点的前一个节点指向当前节点的下一个节点。在那之后，我们使用`current.next.prev = current.prev`使`current`节点的下一个节点指向`current`节点的前一个节点。考虑以下代码片段：

```py
else
    while current:
       if current.data == data:
             current.prev.next = current.next 
             current.next.prev = current.prev 
             node_deleted = True
       current = current.next
```

为了更好地理解双向链表中的删除操作的概念，请考虑以下示例图。在下图中，有三个节点，**A**，**B**和**C**。要删除列表中间的节点**B**，我们实质上会使**A**指向**C**作为其下一个节点，同时使**C**指向**A**作为其前一个节点：

![](img/53dc9efd-2d80-4221-a5d9-deb25a9d0999.png)

进行此操作后，我们得到以下列表：

![](img/9f1a8569-6ab4-4de8-b252-429145333607.png)

最后，检查`node_delete`变量以确定是否实际删除了节点。如果删除了任何节点，则将计数变量减少`1`，这可以跟踪列表中节点的总数。以下代码片段减少了删除任何节点时的计数变量`1`：

```py
if node_deleted:
  self.count -= 1
```

# 列表搜索

在双向链表中搜索项目与在单向链表中的方式类似。我们使用`iter()`方法来检查所有节点中的数据。当我们遍历列表中的所有数据时，每个节点都与`contain`方法中传递的数据进行匹配。如果我们在列表中找到项目，则返回`True`，表示找到了该项目，否则返回`False`，这意味着在列表中未找到该项目。其 Python 代码如下：

```py
def contain(self, data):
    for node_data in self.iter():
       if data == node_data:
       return True 
    return False
```

双向链表中的追加操作具有运行时间复杂度`O(1)`，删除操作具有复杂度`O(n)`。

# 循环列表

循环链表是链表的特殊情况。在循环链表中，端点彼此相连。这意味着列表中的最后一个节点指向第一个节点。换句话说，我们可以说在循环链表中，所有节点都指向下一个节点（在双向链表的情况下还指向前一个节点），没有结束节点，因此没有节点将指向`Null`。循环列表可以基于单向链表和双向链表。在双向循环链表的情况下，第一个节点指向最后一个节点，最后一个节点指向第一个节点。考虑以下基于单向链表的循环链表的图示，其中最后一个节点**C**再次连接到第一个节点**A**，从而形成循环列表：

![](img/5226b232-2fca-4ddf-a27f-680cc8883b9a.png)

下图显示了基于双向链表的循环链表概念，其中最后一个节点**C**通过`next`指针再次连接到第一个节点**A**。节点**A**也通过`previous`指针连接到节点**C**，从而形成一个循环列表：

![](img/3dd10ee2-a107-4e9c-b869-0cd26cfa9414.png)

在这里，我们将看一个单链表循环列表的实现。一旦我们理解了基本概念，实现双链表循环列表应该是直截了当的。

我们可以重用我们在子节中创建的节点类——单链表。事实上，我们也可以重用大部分`SinglyLinkedList`类的部分。因此，我们将专注于循环列表实现与普通单链表不同的方法。

# 追加元素

要在单链表循环列表中追加一个元素，我们只需包含一个新功能，使新添加或追加的节点指向`tail`节点。这在以下代码中得到了演示。与单链表实现相比，多了一行额外的代码，如粗体所示：

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

# 在循环列表中删除元素

要删除循环列表中的一个节点，看起来我们可以类似于在追加操作中所做的方式来做。只需确保`head`指向`tail`。在删除操作中只有一行需要更改。只有当我们删除`tail`节点时，我们需要确保`head`节点被更新为指向新的尾节点。这将给我们以下实现（粗体字代码行是单链表中删除操作实现的一个补充）：

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

然而，这段代码存在一个严重的问题。在循环列表的情况下，我们不能循环直到`current`变成`None`，因为在循环链表的情况下，当前节点永远不会指向`None`。如果删除一个现有节点，你不会看到这一点，但是尝试删除一个不存在的节点，你将陷入无限循环。

因此，我们需要找到一种不同的方法来控制`while`循环。我们不能检查`current`是否已经到达`head`，因为那样它永远不会检查最后一个节点。但我们可以使用`prev`，因为它比`current`落后一个节点。然而，有一个特殊情况。在第一个循环迭代中，`current`和`prev`将指向相同的节点，即尾节点。我们希望确保循环在这里运行，因为我们需要考虑单节点列表。更新后的删除方法现在如下所示：

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

遍历循环链表非常方便，因为我们不需要寻找起始点。我们可以从任何地方开始，只需要在再次到达相同节点时小心停止遍历。我们可以使用我们在本章开头讨论过的`iter()`方法。它应该适用于我们的循环列表；唯一的区别是在遍历循环列表时，我们必须提及一个退出条件，否则程序将陷入循环并无限运行。我们可以通过使用一个计数变量来创建一个退出条件。考虑以下示例代码：

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

一旦我们打印出 1,000 个元素，我们就会跳出循环。

# 总结

在本章中，我们研究了链表。我们学习了列表的基本概念，如节点和指向其他节点的指针。我们实现了这些类型列表中发生的主要操作，并看到了最坏情况的运行时间是如何比较的。

在下一章中，我们将看两种通常使用列表实现的其他数据结构——栈和队列。
