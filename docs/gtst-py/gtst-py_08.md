# 第八章：堆栈和队列

在本章中，我们将在上一章中学到的技能的基础上构建，以创建特殊的列表实现。我们仍然坚持线性结构。在接下来的章节中，我们将介绍更复杂的数据结构。

在本章中，我们将研究以下内容：

+   实现堆栈和队列

+   堆栈和队列的一些应用

# 堆栈

堆栈是一种经常被比作一堆盘子的数据结构。如果你刚刚洗了一个盘子，你把它放在堆叠的顶部。当你需要一个盘子时，你从堆叠的顶部取出它。因此，最后添加到堆叠的盘子将首先从堆叠中移除。因此，堆栈是**后进先出**（**LIFO**）结构：

![](img/f9b6f83e-4f80-4394-8403-1be5aec87197.jpg)

上图描述了一堆盘子的堆栈。只有将一个盘子放在堆叠的顶部才可能添加一个盘子。从盘子堆中移除一个盘子意味着移除堆顶上的盘子。

堆栈上执行的两个主要操作是`push`和`pop`。当元素添加到堆栈顶部时，它被推送到堆栈上。当元素从堆栈顶部取出时，它被弹出堆栈。有时使用的另一个操作是`peek`，它可以查看堆栈上的元素而不将其弹出。

堆栈用于许多事情。堆栈的一个非常常见的用途是在函数调用期间跟踪返回地址。让我们想象一下我们有以下小程序：

```py
def b(): 
    print('b') 

def a(): 
    b() 

a() 
print("done") 
```

当程序执行到对`a()`的调用时，首先将以下指令的地址推送到堆栈上，然后跳转到`a`。在`a`内部，调用`b()`，但在此之前，返回地址被推送到堆栈上。一旦在`b()`中，函数完成后，返回地址就会从堆栈中弹出，这将带我们回到`a()`。当`a`完成时，返回地址将从堆栈中弹出，这将带我们回到`print`语句。

实际上，堆栈也用于在函数之间传递数据。假设你的代码中的某处有以下函数调用：

```py
   somefunc(14, 'eggs', 'ham', 'spam') 
```

将发生的是`14, 'eggs', 'ham'`和`'spam'`将依次被推送到堆栈上：

![](img/ada829bd-7a36-417d-8b9c-0c959bd9e8ed.jpg)

当代码跳转到函数时，`a, b, c, d`的值将从堆栈中弹出。首先将`spam`元素弹出并分配给`d`，然后将`"ham"`分配给`c`，依此类推：

```py
    def somefunc(a, b, c, d): 
        print("function executed")
```

# 堆栈实现

现在让我们来学习 Python 中堆栈的实现。我们首先创建一个`node`类，就像我们在上一章中使用列表一样：

```py
class Node: 
    def __init__(self, data=None): 
        self.data = data 
        self.next = None 
```

现在这对你来说应该很熟悉：一个节点保存数据和列表中下一个项目的引用。我们将实现一个堆栈而不是列表，但节点链接在一起的原则仍然适用。

现在让我们来看一下`stack`类。它开始类似于单链表。我们需要知道堆栈顶部的节点。我们还想跟踪堆栈中节点的数量。因此，我们将向我们的类添加这些字段：

```py
class Stack: 
    def __init__(self): 
        self.top = None 
        self.size = 0 
```

# 推送操作

`push`操作用于将元素添加到堆栈的顶部。以下是一个实现：

```py
   def push(self, data): 
       node = Node(data) 
       if self.top: 
           node.next = self.top 
           self.top = node                 
       else: 
           self.top = node 
       self.size += 1 
```

在下图中，在创建新节点后没有现有节点。因此`self.top`将指向这个新节点。`if`语句的`else`部分保证了这一点：

![](img/8c66894e-0d5c-43ff-af8d-bc571afa8205.jpg)

在我们有一个现有的堆栈的情况下，我们移动`self.top`，使其指向新创建的节点。新创建的节点必须有其**next**指针，指向堆栈上原来的顶部节点：

![](img/6eb72349-1b93-4d2c-ae5e-775a76109b02.jpg)

# 弹出操作

现在我们需要一个`pop`方法来从堆栈中移除顶部元素。在这样做的同时，我们需要返回顶部元素。如果没有更多元素，我们将使堆栈返回`None`：

```py
    def pop(self): 
        if self.top: 
            data = self.top.data 
            self.size -= 1  
            if self.top.next: 
                self.top = self.top.next 
            else: 
                self.top = None 
            return data 
        else: 
            return None 
```

这里需要注意的是内部的`if`语句。如果顶部节点的**next**属性指向另一个节点，那么我们必须将堆栈的顶部指向该节点：

![](img/e832ddc2-57ec-4252-ab91-d1031c910468.jpg)

当堆栈中只有一个节点时，`pop`操作将按以下方式进行：

![](img/068806f4-31b6-4dd2-8e06-1fe00b7a30a3.jpg)

移除这样的节点会导致`self.top`指向`None`：

![](img/42040b37-cc5f-4fb9-9e0b-f9789f7200aa.jpg)

# Peek

正如我们之前所说，我们也可以添加一个`peek`方法。这将只返回堆栈的顶部而不将其从堆栈中移除，使我们能够查看堆栈的顶部元素而不改变堆栈本身。这个操作非常简单。如果有一个顶部元素，返回它的数据，否则返回`None`（以便`peek`的行为与`pop`的行为相匹配）：

```py
    def peek(self): 
        if self.top 
            return self.top.data 
        else: 
            return None 
```

# 括号匹配应用程序

现在让我们看一个例子，说明我们如何使用我们的堆栈实现。我们将编写一个小函数，用于验证包含括号（（，[或{）的语句是否平衡，也就是说，闭合括号的数量是否与开放括号的数量匹配。它还将确保一个括号对确实包含在另一个括号中：

```py
    def check_brackets(statement): 
        stack = Stack() 
        for ch in statement: 
            if ch in ('{', '[', '('): 
                stack.push(ch) 
            if ch in ('}', ']', ')'): 
                last = stack.pop() 
            if last is '{' and ch is '}': 
                continue 
            elif last is '[' and ch is ']': 
                continue 
            elif last is '(' and ch is ')': 
                continue 
            else: 
                return False 
    if stack.size > 0: 
        return False 
    else: 
        return True 
```

我们的函数解析传递给它的语句中的每个字符。如果它得到一个开放括号，它将其推送到堆栈上。如果它得到一个闭合括号，它将堆栈的顶部元素弹出并比较两个括号，以确保它们的类型匹配：（应该匹配），[应该匹配]，{应该匹配}。如果它们不匹配，我们返回`False`，否则我们继续解析。

一旦我们到达语句的末尾，我们需要进行最后一次检查。如果堆栈为空，那么一切正常，我们可以返回`True`。但是如果堆栈不为空，那么我们有一些没有匹配的闭合括号，我们将返回`False`。我们可以用以下小代码测试括号匹配器：

```py
sl = ( 
   "{(foo)(bar)}hellois)a)test", 
   "{(foo)(bar)}hellois)atest", 
   "{(foo)(bar)}hellois)a)test))" 
) 
for s in sl: 
   m = check_brackets(s) 
   print("{}: {}".format(s, m)) 
```

只有三个语句中的第一个应该匹配。当我们运行代码时，我们得到以下输出：

![](img/348b931f-31df-4d67-a398-4e9d96f6db4f.png)

`True`，`False`，`False`。代码有效。总之，堆栈数据结构的`push`和`pop`操作吸引了**O**(*1*)。堆栈数据结构非常简单，但在现实世界中用于实现整个范围的功能。浏览器上的后退和前进按钮是由堆栈实现的。为了能够在文字处理器中具有撤销和重做功能，也使用了堆栈。

# 队列

另一种特殊类型的列表是队列数据结构。这种数据结构与你在现实生活中习惯的常规队列没有什么不同。如果你曾经在机场排队或者在邻里商店等待你最喜欢的汉堡，那么你应该知道队列是如何工作的。

队列也是一个非常基本和重要的概念，因为许多其他数据结构都是基于它们构建的。

队列的工作方式是，通常第一个加入队列的人会首先得到服务，一切条件相同。首先进入，先出的首字母缩写**FIFO**最好地解释了这一点。当人们站在队列中等待轮到他们接受服务时，服务只在队列的前面提供。人们离开队列的唯一时机是在他们被服务时，这只发生在队列的最前面。严格定义来说，人们加入队列的前面是不合法的，因为那里正在为人们提供服务：

![](img/76e5d4fd-9702-49c6-ba06-ae510f3137f4.jpg)

要加入队列，参与者必须首先移动到队列中最后一个人的后面。队列的长度并不重要。这是队列接受新参与者的唯一合法或允许的方式。

我们作为人，所形成的队列并不遵循严格的规则。可能有人已经在队列中决定退出，甚至有其他人替代他们。我们的目的不是模拟真实队列中发生的所有动态。抽象出队列是什么以及它的行为方式使我们能够解决大量的挑战，特别是在计算方面。

我们将提供各种队列的实现，但所有实现都将围绕 FIFO 的相同思想。我们将称添加元素到队列的操作为 enqueue。要从队列中删除元素，我们将创建一个`dequeue`操作。每次入队一个元素时，队列的长度或大小增加一个。相反，出队项目会减少队列中的元素数量。

为了演示这两个操作，以下表格显示了从队列中添加和移除元素的效果：

| **队列操作** | **大小** | **内容** | **操作结果** |
| --- | --- | --- | --- |
| `Queue()` | 0 | `[]` | 创建队列对象 |
| `Enqueue` "Mark" | 1 | `['mark']` | Mark 添加到队列中 |
| `Enqueue` "John" | 2 | `['mark','john']` | John 添加到队列中 |
| `Size()` | 2 | `['mark','john']` | 返回队列中的项目数 |
| `Dequeue()` | 1 | `['mark']` | John 被出队并返回 |
| `Dequeue()` | 0 | `[]` | Mark 被出队并返回 |

# 基于列表的队列

为了将到目前为止讨论的有关队列的一切内容转化为代码，让我们继续使用 Python 的`list`类实现一个非常简单的队列。这有助于我们快速开发并了解队列。必须在队列上执行的操作封装在`ListQueue`类中：

```py
class ListQueue: 
    def __init__(self): 
        self.items = [] 
        self.size = 0 
```

在初始化方法`__init__`中，`items`实例变量设置为`[]`，这意味着创建时队列为空。队列的大小也设置为`zero`。更有趣的方法是`enqueue`和`dequeue`方法。

# 入队操作

`enqueue`操作或方法使用`list`类的`insert`方法在列表的前面插入项目（或数据）：

```py
    def enqueue(self, data): 
        self.items.insert(0, data) 
        self.size += 1 
```

请注意我们如何将插入到队列末尾的操作实现。索引 0 是任何列表或数组中的第一个位置。但是，在我们使用 Python 列表实现队列时，数组索引 0 是新数据元素插入队列的唯一位置。`insert`操作将列表中现有的数据元素向上移动一个位置，然后将新数据插入到索引 0 处创建的空间中。以下图形可视化了这个过程：

![](img/acda141e-486b-4998-acf8-d12153b3b79e.jpg)

为了使我们的队列反映新元素的添加，大小增加了一个：

```py
self.size += 1 
```

我们可以使用 Python 的`shift`方法在列表上实现“在 0 处插入”的另一种方法。归根结底，实现是练习的总体目标。

# 出队操作

`dequeue`操作用于从队列中移除项目。参考队列主题的介绍，此操作捕获了我们为首次加入队列并等待时间最长的客户提供服务的地方：

```py
    def dequeue(self):
        data = self.items.pop()
        self.size -= 1
        return data
```

Python 的`list`类有一个名为`pop()`的方法。`pop`方法执行以下操作：

1.  从列表中删除最后一个项目。

1.  将从列表中删除的项目返回给调用它的用户或代码。

列表中的最后一个项目被弹出并保存在`data`变量中。在方法的最后一行，返回数据。

考虑下图中的隧道作为我们的队列。执行`dequeue`操作时，从队列前面移除数据`1`的节点：

![](img/6aa30ff3-231a-4368-b56d-b03352d05ef9.jpg)

队列中的结果元素如下所示：

![](img/fcfb8cff-59f5-476f-bdd6-05a3c74ee882.jpg)对于`enqueue`操作，我们能说些什么呢？它在多个方面都非常低效。该方法首先必须将所有元素向后移动一个空间。想象一下，当列表中有 100 万个元素需要在每次向队列添加新元素时进行移动。这通常会使大型列表的 enqueue 过程非常缓慢。

# 基于堆栈的队列

使用两个堆栈的另一种队列实现方式。再次，Python 的`list`类将被用来模拟一个堆栈：

```py
class Queue: 
    def __init__(self): 
        self.inbound_stack = [] 
        self.outbound_stack = [] 
```

前述的`queue`类在初始化时将两个实例变量设置为空列表。这些堆栈将帮助我们实现队列。在这种情况下，堆栈只是允许我们在它们上面调用`push`和`pop`方法的 Python 列表。

`inbound_stack` 仅用于存储添加到队列中的元素。在此堆栈上不能执行其他操作。

# 入队操作

`enqueue`方法是向队列添加元素的方法：

```py
def enqueue(self, data): 
    self.inbound_stack.append(data) 
```

该方法是一个简单的方法，只接收客户端想要追加到队列中的`data`。然后将此数据传递给`queue`类中的`inbound_stack`的`append`方法。此外，`append`方法用于模拟`push`操作，将元素推送到堆栈顶部。

要将数据`enqueue`到`inbound_stack`，以下代码可以胜任：

```py
queue = Queue() 
queue.enqueue(5) 
queue.enqueue(6) 
queue.enqueue(7) 
print(queue.inbound_stack) 
```

队列中`inbound_stack`的命令行输出如下：

```py
[5, 6, 7]
```

# 出队操作

`dequeue`操作比其`enqueue`对应操作更复杂一些。添加到我们的队列中的新元素最终会出现在`inbound_stack`中。我们不是从`inbound_stack`中删除元素，而是将注意力转向`outbound_stack`。正如我们所说，只能通过`outbound_stack`从我们的队列中删除元素：

```py
    if not self.outbound_stack: 
        while self.inbound_stack: 
            self.outbound_stack.append(self.inbound_stack.pop()) 
    return self.outbound_stack.pop() 
```

`if`语句首先检查`outbound_stack`是否为空。如果不为空，我们继续通过执行以下操作来移除队列前端的元素：

```py
return self.outbound_stack.pop() 
```

如果`outbound_stack`为空，那么在弹出队列的前端元素之前，`inbound_stack`中的所有元素都将移动到`outbound_stack`中：

```py
while self.inbound_stack: 
    self.outbound_stack.append(self.inbound_stack.pop()) 
```

只要`inbound_stack`中有元素，`while`循环将继续执行。

语句`self.inbound_stack.pop()`将删除最新添加到`inbound_stack`中的元素，并立即将弹出的数据传递给`self.outbound_stack.append()`方法调用。

最初，我们的`inbound_stack`填充了元素**5**，**6**和**7**：

![](img/0c9f8597-491f-473c-b03a-931f030741ea.jpg)

执行`while`循环的主体后，`outbound_stack`如下所示：

![](img/b9f482bd-7bd5-4c63-9587-bb17ff0eecd6.jpg)![](img/a8cd8bb4-4c0e-47a5-95c1-0e2ced3edb1b.png)

`dequeue`方法中的最后一行将返回`5`，作为对`outbound_stack`上的`pop`操作的结果：

```py
return self.outbound_stack.pop() 
```

这将使`outbound_stack`只剩下两个元素：

![](img/18c49954-4199-454a-bec0-f7619ad6113c.jpg)

下次调用`dequeue`操作时，`while`循环将不会被执行，因为`outbound_stack`中没有元素，这使得外部的`if`语句失败。

在这种情况下，立即调用`pop`操作，以便只返回队列中等待时间最长的元素。

使用此队列实现的典型代码运行如下：

```py
queue = Queue() 
queue.enqueue(5) 
queue.enqueue(6) 
queue.enqueue(7) 
print(queue.inbound_stack) 
queue.dequeue() 
print(queue.inbound_stack) 
print(queue.outbound_stack) 
queue.dequeue() 
print(queue.outbound_stack) 
```

前述代码的输出如下：

```py
 [5, 6, 7] 
 [] 
 [7, 6] 
 [7] 
```

代码示例向队列添加元素，并打印队列中的元素。调用`dequeue`方法后，再次打印队列时观察到元素数量的变化。

使用两个堆栈实现队列是面试中经常提出的一个问题。

# 基于节点的队列

使用 Python 列表来实现队列是一个很好的起点，可以让我们感受队列的工作原理。我们完全可以利用指针结构的知识来实现自己的队列数据结构。

可以使用双向链表实现队列，对该数据结构的`插入`和`删除`操作的时间复杂度为**O**(*1*)。

`node`类的定义与我们在双向链表中定义的`Node`相同，如果双向链表能够实现 FIFO 类型的数据访问，那么它可以被视为队列，其中添加到列表中的第一个元素是第一个被移除的。

# 队列类

`queue`类与双向链表`list`类非常相似：

```py
class Queue: 
def __init__(self): 
        self.head = None 
        self.tail = None 
        self.count = 0 
```

在创建`queue`类的实例时，`self.head`和`self.tail`指针被设置为`None`。为了保持`Queue`中节点数量的计数，这里也维护了`count`实例变量，并将其设置为`0`。

# 入队操作

元素通过`enqueue`方法添加到`Queue`对象中。在这种情况下，元素是节点：

```py
    def enqueue(self, data): 
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

`enqueue`方法的代码与双向链表的`append`操作中已经解释过的代码相同。它从传递给它的数据创建一个节点，并将其附加到队列的尾部，或者如果队列为空，则将`self.head`和`self.tail`都指向新创建的节点。队列中元素的总数增加了一行`self.count += 1`。

# 出队操作

使我们的双向链表作为队列的另一个操作是`dequeue`方法。这个方法是用来移除队列前面的节点。

要移除由`self.head`指向的第一个元素，使用`if`语句：

```py
def dequeue(self): 
current = self.head 
        if self.count == 1: 
            self.count -= 1 
            self.head = None 
            self.tail = None 
        elif self.count > 1: 
            self.head = self.head.next 
            self.head.prev = None 
            self.count -= 1 
```

`current`通过指向`self.head`来初始化。如果`self.count`为 1，则意味着列表中只有一个节点，也就是队列中只有一个节点。因此，要移除相关联的节点（由`self.head`指向），需要将`self.head`和`self.tail`变量设置为`None`。

另一方面，如果队列有许多节点，那么头指针将被移动以指向`self.head`的下一个节点。

在运行`if`语句之后，该方法返回被`head`指向的节点。`self.count`在`if`语句执行路径流程中的任何一种方式中都会减少一。

有了这些方法，我们成功地实现了一个队列，大量借鉴了双向链表的思想。

还要记住，将我们的双向链表转换为队列的唯一方法是两种方法，即`enqueue`和`dequeue`。

# 队列的应用

队列在计算机领域中用于实现各种功能。例如，网络上的每台计算机都不提供自己的打印机，可以通过排队来共享一个打印机。当打印机准备好打印时，它将选择队列中的一个项目（通常称为作业）进行打印。

操作系统还将进程排队以供 CPU 执行。让我们创建一个应用程序，利用队列来创建一个简单的媒体播放器。

# 媒体播放器队列

大多数音乐播放器软件允许用户将歌曲添加到播放列表中。点击播放按钮后，主播放列表中的所有歌曲都会依次播放。歌曲的顺序播放可以使用队列来实现，因为排队的第一首歌曲是首先播放的。这符合 FIFO 首字母缩写。我们将实现自己的播放列表队列，以 FIFO 方式播放歌曲。

基本上，我们的媒体播放器队列只允许添加曲目以及播放队列中的所有曲目。在一个完整的音乐播放器中，线程将被用来改进与队列的交互方式，同时音乐播放器继续用于选择下一首要播放、暂停或停止的歌曲。

`track`类将模拟音乐曲目：

```py
from random import randint 
class Track: 

    def __init__(self, title=None): 
        self.title = title 
        self.length = randint(5, 10) 
```

每个音轨都包含对歌曲标题的引用，以及歌曲的长度。长度是在 5 到 10 之间的随机数。随机模块提供了`randint`方法，使我们能够生成随机数。该类表示包含音乐的任何 MP3 音轨或文件。音轨的随机长度用于模拟播放歌曲或音轨所需的秒数。

要创建几个音轨并打印出它们的长度，我们需要做以下操作：

```py
track1 = Track("white whistle") 
track2 = Track("butter butter") 
print(track1.length) 
print(track2.length) 
```

上述代码的输出如下：

```py
    6
 7
```

由于为两个音轨生成的随机长度可能不同，因此您的输出可能会有所不同。

现在，让我们创建我们的队列。使用继承，我们只需从`queue`类继承：

```py
import time 
class MediaPlayerQueue(Queue): 

    def __init__(self): 
        super(MediaPlayerQueue, self).__init__() 
```

通过调用`super`来正确初始化队列。该类本质上是一个队列，其中包含队列中的多个音轨对象。要将音轨添加到队列中，需要创建一个`add_track`方法：

```py
    def add_track(self, track): 
        self.enqueue(track) 
```

该方法将`track`对象传递给队列`super`类的`enqueue`方法。这将实际上使用`track`对象（作为节点的数据）创建一个`Node`，并将尾部（如果队列不为空）或头部和尾部（如果队列为空）指向这个新节点。

假设队列中的音轨是按照先进先出的顺序播放的，那么`play`函数必须循环遍历队列中的元素：

```py
def play(self): 
        while self.count > 0: 
            current_track_node = self.dequeue() 
            print("Now playing {}".format(current_track_node.data.title)) 
            time.sleep(current_track_node.data.length) 
```

`self.count`用于计算音轨何时被添加到我们的队列以及何时被出队。如果队列不为空，对`dequeue`方法的调用将返回队列前面的节点（其中包含`track`对象）。然后，`print`语句通过节点的`data`属性访问音轨的标题。为了进一步模拟播放音轨，`time.sleep()`方法将暂停程序执行，直到音轨的秒数已经过去：

```py
time.sleep(current_track_node.data.length) 
```

媒体播放器队列由节点组成。当音轨被添加到队列时，该音轨会隐藏在一个新创建的节点中，并与节点的数据属性相关联。这就解释了为什么我们通过对`dequeue`的调用返回的节点的数据属性来访问节点的`track`对象：

![](img/46c48a15-546f-44d7-b5d2-0a916532f052.jpg)

您可以看到，`node`对象不仅仅存储任何数据，而是在这种情况下存储音轨。

让我们来试试我们的音乐播放器：

```py
track1 = Track("white whistle") 
track2 = Track("butter butter") 
track3 = Track("Oh black star") 
track4 = Track("Watch that chicken") 
track5 = Track("Don't go") 
```

我们使用随机单词创建了五个音轨对象的标题：

```py
print(track1.length) 
print(track2.length) 
>> 8 >> 9
```

由于随机长度的原因，输出应该与您在您的机器上获得的结果不同。

接下来，创建`MediaPlayerQueue`类的一个实例：

```py
media_player = MediaPlayerQueue() 
```

音轨将被添加，并且`play`函数的输出应该按照我们排队的顺序打印出正在播放的音轨：

```py
media_player.add_track(track1) 
media_player.add_track(track2) 
media_player.add_track(track3) 
media_player.add_track(track4) 
media_player.add_track(track5) 
media_player.play() 
```

上述代码的输出如下：

```py
    >>Now playing white whistle
 >>Now playing butter butter
 >>Now playing Oh black star
 >>Now playing Watch that chicken
 >>Now playing Don't go
```

在程序执行时，可以看到音轨是按照它们排队的顺序播放的。在播放音轨时，系统还会暂停与音轨长度相等的秒数。

# 摘要

在本章中，我们利用了将节点链接在一起来创建其他数据结构的知识，即栈和队列。我们已经看到了这些数据结构如何紧密地模仿现实世界中的栈和队列。具体的实现，以及它们不同的类型，都已经展示出来。我们随后将栈和队列的概念应用于编写现实生活中的程序。

我们将在下一章中讨论树。将讨论树的主要操作，以及在哪些领域应用数据结构。
