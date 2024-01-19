# 栈和队列

在本章中，我们将在上一章学到的技能基础上创建特殊的列表实现。我们仍然坚持使用线性结构。在接下来的章节中，我们将深入了解更复杂的数据结构的细节。

在本章中，我们将了解栈和队列的概念。我们还将使用各种方法在 Python 中实现这些数据结构，如`lists`和`node`。

在本章中，我们将涵盖以下内容：

+   使用各种方法实现栈和队列

+   栈和队列的一些真实应用示例

# 技术要求

你应该有一台安装了 Python 的计算机系统。本章讨论的概念的所有程序都在书中提供，也可以在以下链接的 GitHub 存储库中找到：[`github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter05`](https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter05)。

# 栈

栈是一种存储数据的数据结构，类似于厨房里的一堆盘子。你可以把一个盘子放在栈的顶部，当你需要一个盘子时，你从栈的顶部拿走它。最后添加到栈上的盘子将首先从栈中取出。同样，栈数据结构允许我们从一端存储和读取数据，最后添加的元素首先被取出。因此，栈是一种**后进先出**（**LIFO**）结构：

![](img/7db7f285-a46b-405d-a964-5a7367bb4533.png)

前面的图表描述了一堆盘子。只有将一个盘子放在堆的顶部才有可能添加一个盘子。从盘子堆中移除一个盘子意味着移除堆顶上的盘子。

栈上执行的两个主要操作是`push`和`pop`。当元素被添加到栈顶时，它被推送到栈上。当要从栈顶取出元素时，它被弹出栈。有时使用的另一个操作是`peek`，它可以查看栈顶的元素而不将其弹出。

栈用于许多事情。栈的一个非常常见的用途是在函数调用期间跟踪返回地址。假设我们有以下程序：

```py
def b(): 
    print('b') 

def a(): 
    b() 

a() 
print("done")
```

当程序执行到`a()`的调用时，发生以下情况：

1.  首先将当前指令的地址推送到栈上，然后跳转到`a`的定义

1.  在函数`a()`内部，调用函数`b()`

1.  函数`b()`的返回地址被推送到栈上

1.  一旦`b()`函数和函数执行完毕，返回地址将从栈中弹出，这将带我们回到函数`a()`。

1.  当函数`a`中的所有指令完成时，返回地址再次从栈中弹出，这将带我们回到`main`函数和`print`语句

栈也用于在函数之间传递数据。考虑以下示例。假设你的代码中有以下函数调用：

```py
   somefunc(14, 'eggs', 'ham', 'spam') 
```

内部发生的是，函数传递的值`14, 'eggs', 'ham'`和`'spam'`将依次被推送到栈上，如下图所示：

![](img/90a07abc-4ef2-4623-ab9b-73bfc8b6c7a2.png)

当代码调用`jump`到函数定义时，`a, b, c, d`的值将从栈中弹出。首先弹出`spam`元素并赋值给`d`，然后将`ham`赋值给`c`，依此类推：

```py
    def somefunc(a, b, c, d): 
        print("function executed")
```

# 栈实现

栈可以使用节点在 Python 中实现。我们首先创建一个`node`类，就像在上一章中使用列表一样：

```py
class Node: 
    def __init__(self, data=None): 
        self.data = data 
        self.next = None 
```

正如我们讨论的，一个节点包含数据和列表中下一个项目的引用。在这里，我们将实现一个栈而不是列表；然而，节点的相同原则在这里也适用——节点通过引用链接在一起。

现在让我们来看一下`stack`类。它的开始方式与单链表类似。我们需要两样东西来实现使用节点的栈：

1.  首先，我们需要知道位于栈顶的节点，以便我们能够通过这个节点应用`push`和`pop`操作。

1.  我们还希望跟踪栈中节点的数量，因此我们向栈类添加一个`size`变量。考虑以下代码片段用于栈类：

```py
class Stack: 
    def __init__(self): 
        self.top = None 
        self.size = 0 
```

# 推送操作

`push`操作是栈上的一个重要操作；它用于在栈顶添加一个元素。我们在 Python 中实现推送功能以了解它是如何工作的。首先，我们检查栈是否已经有一些项目或者它是空的，当我们希望在栈中添加一个新节点时。

如果栈已经有一些元素，那么我们需要做两件事：

1.  新节点必须使其下一个指针指向先前位于顶部的节点。

1.  我们通过将`self.top`指向新添加的节点，将这个新节点放在栈的顶部。请参阅以下图表中的两条指令：

![](img/e27ba744-6175-41ae-afc1-0ee0b18031cb.png)

如果现有栈为空，并且要添加的新节点是第一个元素，我们需要将此节点作为元素的顶部节点。因此，`self.top`将指向这个新节点。请参阅以下图表：

![](img/3c415c87-cfa5-41e5-87b9-35283f6d50fd.png)

以下是`stack`中`push`操作的完整实现：

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

# 弹出操作

现在，我们需要栈的另一个重要功能，那就是`pop`操作。它读取栈的顶部元素并将其从栈中移除。`pop`操作返回栈的顶部元素，并且如果栈为空则返回`None`。

要在栈上实现`pop`操作：

1.  首先，检查栈是否为空。在空栈上不允许`pop`操作。

1.  如果栈不为空，可以检查顶部节点是否具有其`next`属性指向其他节点。这意味着栈中有元素，并且顶部节点指向栈中的下一个节点。要应用`pop`操作，我们必须更改顶部指针。下一个节点应该在顶部。我们通过将`self.top`指向`self.top.next`来实现这一点。请参阅以下图表以了解这一点：

![](img/45afd13e-ada4-4424-b49e-5e78b9e2c4c1.png)

1.  当栈中只有一个节点时，在弹出操作后栈将为空。我们必须将顶部指针更改为`None`。见下图：

![](img/ee07d980-b726-4efe-bbc7-ac9a7bb29cb3.png)

1.  移除这样一个节点会导致`self.top`指向`None`：

![](img/c6831f6e-a390-4a57-b4cc-9abd84efb305.png)

1.  如果栈不为空，如果栈的顶部节点具有其`next`属性指向其他节点，则可以将栈的大小减少`1`。以下是 Python 中`stack`的`pop`操作的完整代码：

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

# 查看操作

还有另一个可以应用在栈上的重要操作——`peek`方法。这个方法返回栈顶的元素，而不从栈中删除它。`peek`和`pop`之间唯一的区别是，`peek`方法只返回顶部元素；然而，在`pop`方法的情况下，顶部元素被返回并且也从栈中删除。

弹出操作允许我们查看顶部元素而不改变栈。这个操作非常简单。如果有顶部元素，则返回其数据；否则，返回`None`（因此，`peek`的行为与`pop`相匹配）：

```py
    def peek(self): 
        if self.top 
            return self.top.data 
        else: 
            return None 
```

# 括号匹配应用

现在让我们看一个示例应用程序，展示我们如何使用我们的堆栈实现。我们将编写一个小函数，用于验证包含括号（`(`，`[`或`{`）的语句是否平衡，即关闭括号的数量是否与开放括号的数量匹配。它还将确保一个括号对确实包含在另一个括号中：

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

我们的函数解析传递给它的语句中的每个字符。如果它得到一个开放括号，它将其推送到堆栈上。如果它得到一个关闭括号，它将堆栈的顶部元素弹出并比较两个括号，以确保它们的类型匹配，`(`应该匹配`)`，`[`应该匹配`]`，`{`应该匹配`}`。如果它们不匹配，我们返回`False`；否则，我们继续解析。

一旦我们到达语句的末尾，我们需要进行最后一次检查。如果堆栈为空，那么很好，我们可以返回`True`。但是如果堆栈不为空，那么我们有一个没有匹配的关闭括号，我们将返回`False`。我们可以使用以下代码测试括号匹配器：

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

![](img/17bdec19-ac8d-4124-bb4e-16767635d5e8.png)

上述代码的输出是`True`，`False`和`False`。

总之，堆栈数据结构的`push`和`pop`操作吸引了*O(1)*的复杂性。堆栈数据结构很简单；然而，它被用于实现许多真实世界应用中的功能。浏览器中的后退和前进按钮是使用堆栈实现的。堆栈也用于实现文字处理器中的撤销和重做功能。

# 队列

另一种特殊的列表类型是队列数据结构。队列数据结构非常类似于你在现实生活中习惯的常规队列。如果你曾经在机场排队或在邻里商店排队等待你最喜欢的汉堡，那么你应该知道队列是如何工作的。

队列是非常基础和重要的概念，因为许多其他数据结构都是建立在它们之上的。

队列的工作方式如下。通常，第一个加入队列的人会首先被服务，每个人都将按照加入队列的顺序被服务。首先进入，先出的首字母缩写 FIFO 最好地解释了队列的概念。**FIFO**代表**先进先出**。当人们站在队列中等待轮到他们被服务时，服务只在队列的前端提供。人们只有在被服务时才会离开队列，这只会发生在队列的最前面。请参见以下图表，其中人们站在队列中，最前面的人将首先被服务：

![](img/e5fb97b2-07a9-4729-b679-8b1f1727db3f.png)

要加入队列，参与者必须站在队列中的最后一个人后面。这是队列接受新成员的唯一合法方式。队列的长度并不重要。

我们将提供各种队列的实现，但这将围绕 FIFO 的相同概念。首先添加的项目将首先被读取。我们将称添加元素到队列的操作为`enqueue`。当我们从队列中删除一个元素时，我们将称之为`dequeue`操作。每当一个元素被入队时，队列的长度或大小增加 1。相反，出队的项目会减少队列中的元素数量 1。

为了演示这两个操作，以下表格显示了从队列中添加和删除元素的效果：

| **队列操作** | **大小** | **内容** | **操作结果** |
| --- | --- | --- | --- |
| `Queue()` | 0 | `[]` | 创建了一个空的队列对象。 |
| `Enqueue` Packt  | 1 | `['Packt']` |  队列中添加了一个 *Packt* 项目。 |
| `Enqueue` 发布  | 2 | `['发布', 'Packt']` | 队列中添加了一个 *发布* 项目。 |
| `Size()` | 2 | `['Publishing', 'Packt']` | 返回队列中的项目数，在此示例中为 2。 |
| `Dequeue()` | 1 | `['Publishing']` | *Packt*项目被出队并返回。（这个项目是第一个添加的，所以它被第一个移除。） |
| `Dequeue()` | 0 | `[]` | *Publishing*项目被出队并返回。（这是最后添加的项目，所以最后返回。） |

# 基于列表的队列

队列可以使用各种方法实现，例如`list`、`stack`和`node`。我们将逐一讨论使用所有这些方法实现队列的方法。让我们从使用 Python 的`list`类实现队列开始。这有助于我们快速了解队列。必须在队列上执行的操作封装在`ListQueue`类中：

```py
class ListQueue: 
    def __init__(self): 
        self.items = [] 
        self.size = 0 
```

在初始化方法`__init__`中，`items`实例变量设置为`[]`，这意味着创建时队列为空。队列的大小也设置为`零`。`enqueue`和`dequeue`是队列中重要的方法，我们将在下一小节中讨论它们。

# 入队操作

`enqueue`操作将项目添加到队列中。它使用`list`类的`insert`方法在列表的前面插入项目（或数据）。请参阅以下代码以实现`enqueue`方法：

```py
  def enqueue(self, data): 
    self.items.insert(0, data)   # Always insert items at index 0
    self.size += 1               # increment the size of the queue by 1
```

重要的是要注意我们如何使用列表实现队列中的插入。概念是我们在列表的索引`0`处添加项目；这是数组或列表中的第一个位置。要理解在列表的索引`0`处添加项目时队列的工作原理的概念，请考虑以下图表。我们从一个空列表开始。最初，我们在索引`0`处添加一个项目`1`。接下来，我们在索引`0`处添加一个项目`2`；它将先前添加的项目移动到下一个索引。

接下来，当我们再次在索引`0`处向列表中添加一个新项目`3`时，已添加到列表中的所有项目都会被移动，如下图所示。同样，当我们在索引`0`处添加项目`4`时，列表中的所有项目都会被移动：

![](img/3f3d2269-8c89-4dcd-88b6-986e1ff83407.png)

因此，在我们使用 Python 列表实现队列时，数组索引`0`是唯一可以向队列中插入新数据元素的位置。`insert`操作将列表中现有的数据元素向上移动一个位置，然后将新数据插入到索引`0`处创建的空间中。

为了使我们的队列反映新元素的添加，大小增加了`1`：

```py
self.size += 1 
```

我们可以使用 Python 列表的`shift`方法作为在`0`处实现插入的另一种方法。

# 出队操作

`dequeue`操作用于从队列中删除项目。该方法返回队列中的顶部项目并将其从队列中删除。以下是`dequeue`方法的实现：

```py
  def dequeue(self):
    data = self.items.pop()    # delete the topmost item from the queue
    self.size -= 1             # decrement the size of the queue by 1
     return data
```

Python 的`list`类有一个名为`pop()`的方法。`pop`方法执行以下操作：

1.  从列表中删除最后一个项目

1.  将从列表中删除的项目返回给用户或调用它的代码

列表中的最后一个项目被弹出并保存在`data`变量中。在方法的最后一行，返回数据。

考虑以下图表作为我们的队列实现，其中添加了三个元素—`1`、`2`和`3`。执行`dequeue`操作时，数据为`1`的节点从队列的前面移除，因为它是最先添加的：

![](img/26a4406e-d047-41ca-a5ac-802bf0f38a3a.png)

队列中的结果元素如下所示：

![](img/57d7bcfd-6bd3-40eb-9fac-9d062009f56f.png)

由于一个原因，`enqueue`操作非常低效。该方法必须首先将所有元素向前移动一个空间。想象一下，列表中有 100 万个元素需要在每次向队列添加新元素时进行移动。这将使大型列表的入队过程非常缓慢。

# 基于堆栈的队列

队列也可以使用两个栈来实现。我们最初设置了两个实例变量来在初始化时创建一个空队列。这些是帮助我们实现队列的栈。在这种情况下，栈只是允许我们在其上调用`push`和`pop`方法的 Python 列表，最终允许我们获得`enqueue`和`dequeue`操作的功能。以下是`Queue`类：

```py
class Queue: 
    def __init__(self): 
        self.inbound_stack = [] 
        self.outbound_stack = [] 
```

`inbound_stack`只用于存储添加到队列中的元素。不能对此堆栈执行其他操作。

# 入队操作

`enqueue`方法用于向队列中添加项目。这个方法非常简单，只接收要附加到队列的`data`。然后将此数据传递给`queue`类中`inbound_stack`的`append`方法。此外，`append`方法用于模拟`push`操作，将元素推送到栈的顶部。以下代码是使用 Python 中的栈实现`enqueue`的方法：

```py
def enqueue(self, data): 
    self.inbound_stack.append(data) 
```

要将数据`enqueue`到`inbound_stack`，以下代码可以完成任务：

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

`dequeue`操作用于按添加的项目顺序从队列中删除元素。添加到我们的队列中的新元素最终会出现在`inbound_stack`中。我们不是从`inbound_stack`中删除元素，而是将注意力转移到另一个栈，即`outbound_stack`。我们只能通过`outbound_stack`删除队列中的元素。

为了理解`outbound_stack`如何用于从队列中删除项目，让我们考虑以下示例。

最初，我们的`inbound_stack`填充了元素**5**、**6**和**7**，如下图所示：

![](img/445b36d4-1c69-4ec0-bba4-7588a994ea88.png)

我们首先检查`outbound_stack`是否为空。由于开始时它是空的，我们使用`pop`操作将`inbound_stack`的所有元素移动到`outbound_stack`。现在`inbound_stack`变为空，而`outbound_stack`保留元素。我们在下图中展示了这一点，以便更清楚地理解：

![](img/9e911f2b-2f63-4ff0-b4ad-ec6f2d1e1860.png)

现在，如果`outbound_stack`不为空，我们继续使用`pop`操作从队列中删除项目。在前面的图中，当我们对`outbound_stack`应用`pop`操作时，我们得到了元素`5`，这是正确的，因为它是第一个添加的元素，应该是从队列中弹出的第一个元素。这样`outbound_stack`就只剩下两个元素了：

![](img/414ded21-9d68-426a-84e5-eed9190d9f37.png)

以下是队列的`dequeue`方法的实现：

```py
def dequeue(self):  
    if not self.outbound_stack: 
        while self.inbound_stack: 
            self.outbound_stack.append(self.inbound_stack.pop()) 
    return self.outbound_stack.pop() 
```

`if`语句首先检查`outbound_stack`是否为空。如果不为空，我们继续使用`pop`方法删除队列前端的元素，如下所示：

```py
return self.outbound_stack.pop() 
```

如果`outbound_stack`为空，那么在弹出队列的前端元素之前，`inbound_stack`中的所有元素都将移动到`outbound_stack`中：

```py
while self.inbound_stack: 
    self.outbound_stack.append(self.inbound_stack.pop()) 
```

`while`循环将在`inbound_stack`中有元素的情况下继续执行。

`self.inbound_stack.pop()`语句将删除添加到`inbound_stack`的最新元素，并立即将弹出的数据传递给`self.outbound_stack.append()`方法调用。

让我们考虑一个示例代码，以理解队列上的操作。我们首先使用队列实现向队列中添加三个项目，即`5`、`6`和`7`。接下来，我们应用`dequeue`操作从队列中删除项目。以下是代码：

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

上述代码的输出如下：

```py
 [5, 6, 7] 
 [] 
 [7, 6] 
 [7] 
```

前面的代码片段首先向队列添加元素，并打印出队列中的元素。接下来调用`dequeue`方法，然后再次打印队列时观察到元素数量的变化。

使用两个栈实现队列非常重要，关于这个问题在面试中经常被提出。

# 基于节点的队列

使用 Python 列表来实现队列是一个很好的开始，可以让我们了解队列的工作原理。我们也可以通过使用指针结构来实现自己的队列数据结构。

可以使用双向链表实现队列，并且在这个数据结构上进行`插入`和`删除`操作，时间复杂度为`*O(1)*`。

`node`类的定义与我们在讨论双向链表时定义的`Node`相同。如果双向链表能够实现 FIFO 类型的数据访问，那么它可以被视为队列，其中添加到列表中的第一个元素是要被移除的第一个元素。

# 队列类

`queue`类与双向链表`list`类和`Node`类非常相似，用于在双向链表中添加节点：

```py
class Node(object):
    def __init__(self, data=None, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev

class Queue: 
    def __init__(self): 
        self.head = None 
        self.tail = None 
        self.count = 0 
```

在创建`queue`类实例时，`self.head`和`self.tail`指针最初设置为`None`。为了保持`Queue`中节点数量的计数，这里还维护了`count`实例变量，最初设置为`0`。

# 入队操作

通过`enqueue`方法向`Queue`对象添加元素。元素或数据通过节点添加。`enqueue`方法的代码与我们在第四章中讨论的双向链表的`append`操作非常相似，*列表和指针结构*。

入队操作从传递给它的数据创建一个节点，并将其附加到队列的`tail`，如果队列为空，则将`self.head`和`self.tail`都指向新创建的节点。队列中元素的总数增加了一行`self.count += 1`。如果队列不为空，则新节点的`previous`变量设置为列表的`tail`，并且尾部的下一个指针（或变量）设置为新节点。最后，我们更新尾指针指向新节点。代码如下所示：

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

# 出队操作

使我们的双向链表作为队列的另一个操作是`dequeue`方法。这个方法移除队列前面的节点。为了移除`self.head`指向的第一个元素，使用了一个`if`语句：

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

`current`被初始化为指向`self.head`。如果`self.count`为`1`，那么意味着列表中只有一个节点，也就是队列。因此，要移除相关的节点（由`self.head`指向），`self.head`和`self.tail`变量被设置为`None`。

如果队列有多个节点，那么头指针会移动到`self.head`之后的下一个节点。

在执行`if`语句之后，该方法返回被`head`指向的节点。此外，在这两种情况下，即初始计数为`1`和大于`1`时，变量`self.count`都会减少`1`。

有了这些方法，我们已经实现了一个队列，很大程度上借鉴了双向链表的思想。

还要记住，将我们的双向链表转换成队列的唯一方法是`enqueue`和`dequeue`方法。

# 队列的应用

队列可以在许多实际的计算机应用程序中用于实现各种功能。例如，可以通过排队打印机要打印的内容，而不是为网络上的每台计算机提供自己的打印机。当打印机准备好打印时，它将选择队列中的一个项目（通常称为作业）进行打印。它将按照不同计算机给出的命令的顺序打印出来。

操作系统也会对要由 CPU 执行的进程进行排队。让我们创建一个应用程序，利用队列来创建一个简单的媒体播放器。

# 媒体播放器队列

大多数音乐播放器软件允许用户将歌曲添加到播放列表中。点击播放按钮后，主播放列表中的所有歌曲都会依次播放。使用队列可以实现歌曲的顺序播放，因为排队的第一首歌曲是要播放的第一首歌曲。这符合 FIFO 首字母缩写。我们将实现自己的播放列表队列以按 FIFO 方式播放歌曲。

我们的媒体播放器队列只允许添加曲目以及播放队列中的所有曲目。在一个完整的音乐播放器中，线程将被用于改进与队列的交互方式，同时音乐播放器继续用于选择下一首要播放、暂停或停止的歌曲。

`track`类将模拟音乐曲目：

```py
from random import randint 
class Track: 
    def __init__(self, title=None): 
        self.title = title 
        self.length = randint(5, 10) 
```

每个曲目都保存了歌曲的标题的引用，以及歌曲的长度。歌曲的长度是在`5`和`10`之间的随机数。Python 中的随机模块提供了`randint`函数，使我们能够生成随机数。该类表示包含音乐的任何 MP3 曲目或文件。曲目的随机长度用于模拟播放歌曲或曲目所需的秒数。

要创建几个曲目并打印出它们的长度，我们需要做以下操作：

```py
track1 = Track("white whistle") 
track2 = Track("butter butter") 
print(track1.length) 
print(track2.length) 
```

前面代码的输出如下：

```py
6
7
```

根据生成的两个曲目的随机长度，您的输出可能会有所不同。

现在，让我们创建我们的队列。使用继承，我们只需从`queue`类继承：

```py
import time 
class MediaPlayerQueue(Queue): 

    def __init__(self): 
        super(MediaPlayerQueue, self).__init__() 
```

通过调用`super`来适当初始化队列。这个类本质上是一个队列，它在队列中保存了一些曲目对象。要将曲目添加到队列，需要创建一个`add_track`方法：

```py
    def add_track(self, track): 
        self.enqueue(track) 
```

该方法将`track`对象传递给队列`super`类的`enqueue`方法。这将实际上使用`track`对象（作为节点的数据）创建一个`Node`，并将尾部（如果队列不为空）或头部和尾部（如果队列为空）指向这个新节点。

假设队列中的曲目是按照添加的第一首曲目到最后一首曲目的顺序依次播放（FIFO），那么`play`函数必须循环遍历队列中的元素：

```py
def play(self): 
        while self.count > 0: 
            current_track_node = self.dequeue() 
            print("Now playing {}".format(current_track_node.data.title)) 
            time.sleep(current_track_node.data.length) 
```

`self.count`用于计算何时向我们的队列添加了曲目以及何时曲目已被出队。如果队列不为空，对`dequeue`方法的调用将返回队列前端的节点（其中包含`track`对象）。然后，`print`语句通过节点的`data`属性访问曲目的标题。为了进一步模拟播放曲目，`time.sleep()`方法会暂停程序执行，直到曲目的秒数已经过去：

```py
time.sleep(current_track_node.data.length)
```

媒体播放器队列由节点组成。当一首曲目被添加到队列时，该曲目会隐藏在一个新创建的节点中，并与节点的数据属性相关联。这就解释了为什么我们通过对`dequeue`的调用返回的节点的数据属性来访问节点的`track`对象：

![](img/a61216ca-9e9e-4cc4-af3e-d1923711a688.png)

您可以看到，我们的`node`对象不仅仅存储任何数据，而是在这种情况下存储曲目。

让我们来试试我们的音乐播放器：

```py
track1 = Track("white whistle") 
track2 = Track("butter butter") 
track3 = Track("Oh black star") 
track4 = Track("Watch that chicken") 
track5 = Track("Don't go") 
```

我们使用随机单词创建了五个曲目对象作为标题：

```py
print(track1.length) 
print(track2.length) 
>> 8 >> 9
```

由于随机长度，输出应该与您在您的机器上得到的不同。

接下来，创建`MediaPlayerQueue`类的一个实例：

```py
media_player = MediaPlayerQueue()
```

曲目将被添加，`play`函数的输出应该按照我们排队的顺序打印出正在播放的曲目：

```py
media_player.add_track(track1) 
media_player.add_track(track2) 
media_player.add_track(track3) 
media_player.add_track(track4) 
media_player.add_track(track5) 
media_player.play() 
```

前面代码的输出如下：

```py
    >>Now playing white whistle
 >>Now playing butter butter
 >>Now playing Oh black star
 >>Now playing Watch that chicken
 >>Now playing Don't go
```

在程序执行时，可以看到曲目按照它们排队的顺序播放。在播放曲目时，系统还会暂停与曲目长度相等的秒数。

# 摘要

在这一章中，我们利用了我们对链接节点的知识来创建其他数据结构，即“栈”和“队列”。我们已经看到了这些数据结构如何紧密地模仿现实世界中的栈和队列。我们探讨了具体的实现，以及它们不同的类型。我们随后将应用栈和队列的概念来编写现实生活中的程序。

在下一章中，我们将考虑树。将讨论树的主要操作，以及适用它们数据结构的不同领域。
