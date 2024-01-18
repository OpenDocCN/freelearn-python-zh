# 树

树是一种分层的数据结构。当我们处理列表、队列和栈时，项目是相互跟随的。但在树中，项目之间存在着*父子*关系。

为了形象化树的外观，想象一棵树从地面长出。现在把这个形象从你的脑海中移除。树通常是向下绘制的，所以你最好想象树的根结构向下生长。

在每棵树的顶部是所谓的*根节点*。这是树中所有其他节点的祖先。

树被用于许多事情，比如解析表达式和搜索。某些文档类型，如XML和HTML，也可以以树形式表示。在本章中，我们将看一些树的用途。

在本章中，我们将涵盖以下领域：

+   树的术语和定义

+   二叉树和二叉搜索树

+   树的遍历

# 术语

让我们考虑一些与树相关的术语。

为了理解树，我们首先需要理解它们所依赖的基本思想。下图包含了一个典型的树，由字母A到M的字符节点组成。

![](assets/ecea614b-c914-4a6d-a097-aa3b3f11da67.png)

以下是与树相关的术语列表：

+   **节点**：每个圈起来的字母代表一个节点。节点是任何包含数据的结构。

+   **根节点**：根节点是所有其他节点都来自的唯一节点。一个没有明显根节点的树不能被认为是一棵树。我们树中的根节点是节点A。

+   **子树**：树的子树是一棵树，其节点是另一棵树的后代。节点F、K和L形成了原始树的子树，包括所有节点。

+   **度**：给定节点的子树数。只有一个节点的树的度为0。这个单个树节点也被所有标准视为一棵树。节点A的度为2。

+   **叶节点**：这是一个度为0的节点。节点J、E、K、L、H、M和I都是叶节点。

+   **边**：两个节点之间的连接。有时边可以将一个节点连接到自身，使边看起来像一个循环。

+   **父节点**：树中具有其他连接节点的节点是这些节点的父节点。节点B是节点D、E和F的父节点。

+   **子节点**：这是一个连接到其父节点的节点。节点B和C是节点A的子节点和根节点。

+   **兄弟节点**：所有具有相同父节点的节点都是兄弟节点。这使得节点B和C成为兄弟节点。

+   **级别**：节点的级别是从根节点到节点的连接数。根节点位于级别0。节点B和C位于级别1。

+   **树的高度**：这是树中的级别数。我们的树的高度为4。

+   **深度**：节点的深度是从树的根到该节点的边数。节点H的深度为2。

我们将从考虑树中的节点并抽象一个类开始对树的处理。

# 树节点

就像我们遇到的其他数据结构一样，如列表和栈，树是由节点构建而成的。但构成树的节点需要包含我们之前提到的关于父子关系的数据。

现在让我们看看如何在Python中构建一个二叉树`node`类：

```py
    class Node: 
        def __init__(self, data): 
            self.data = data 
            self.right_child = None 
            self.left_child = None 
```

就像我们以前的实现一样，一个节点是一个包含数据并持有对其他节点的引用的容器。作为二叉树节点，这些引用是指左右子节点。

为了测试这个类，我们首先创建了一些节点：

```py
    n1 = Node("root node")  
    n2 = Node("left child node") 
    n3 = Node("right child node") 
    n4 = Node("left grandchild node") 
```

接下来，我们将节点连接到彼此。我们让`n1`成为根节点，`n2`和`n3`成为它的子节点。最后，我们将`n4`作为`n2`的左子节点连接，这样当我们遍历左子树时，我们会得到一些迭代：

```py
    n1.left_child = n2 
    n1.right_child = n3 
    n2.left_child = n4 
```

一旦我们设置好了树的结构，我们就准备好遍历它了。如前所述，我们将遍历左子树。我们打印出节点并向下移动树到下一个左节点。我们一直这样做，直到我们到达左子树的末尾：

```py
    current = n1 
    while current: 
        print(current.data) 
        current = current.left_child 
```

正如你可能已经注意到的，这需要客户端代码中相当多的工作，因为你必须手动构建树结构。

# 二叉树

二叉树是每个节点最多有两个子节点的树。二叉树非常常见，我们将使用它们来构建Python中的BST实现。

以下图是一个以5为根节点的二叉树的示例：

![](assets/24dfae79-0a28-407a-af32-db8491f991a7.jpg)

每个子节点都被标识为其父节点的右子节点或左子节点。由于父节点本身也是一个节点，即使节点不存在，每个节点也会保存对右子节点和左子节点的引用。

常规二叉树没有关于如何排列树中元素的规则。它只满足每个节点最多有两个子节点的条件。

# 二叉搜索树

**二叉搜索树**（BST）是一种特殊类型的二叉树。也就是说，它在结构上是一棵二叉树。在功能上，它是一棵以一种能够高效搜索树的方式存储其节点的树。

BST有一种结构。对于具有值的给定节点，左子树中的所有节点都小于或等于该节点的值。此外，该节点的右子树中的所有节点都大于父节点的值。例如，考虑以下树：

![](assets/c7916505-4ac0-48c2-af08-e7759443935c.png)

这是BST的一个示例。测试我们的树是否具有BST的属性，你会意识到根节点左子树中的所有节点的值都小于5。同样，右子树中的所有节点的值都大于5。这个属性适用于BST中的所有节点，没有例外：

![](assets/1d79ef04-0fb1-4ad6-ac1b-adcfbb3ab621.png)

尽管前面的图看起来与之前的图相似，但它并不符合BST的条件。节点7大于根节点5；然而，它位于根节点的左侧。节点4位于其父节点7的右子树中，这是不正确的。

# 二叉搜索树实现

让我们开始实现BST。我们希望树能够保存对其自己根节点的引用：

```py
    class Tree: 
        def __init__(self): 
            self.root_node = None 
```

这就是维护树状态所需的全部内容。让我们在下一节中检查树上的主要操作。

# 二叉搜索树操作

基本上有两个操作对于使用BST是必要的。这些是“插入”和“删除”操作。这些操作必须遵循一个规则，即它们必须保持给BST赋予结构的原则。

在我们处理节点的插入和删除之前，让我们讨论一些同样重要的操作，这些操作将帮助我们更好地理解“插入”和“删除”操作。

# 查找最小和最大节点

BST的结构使得查找具有最大和最小值的节点非常容易。

要找到具有最小值的节点，我们从树的根开始遍历，并在到达子树时每次访问左节点。我们做相反的操作来找到树中具有最大值的节点：

![](assets/c229180d-2172-4729-bdfb-c90772604353.png)

我们从节点6到3到1向下移动，以找到具有最小值的节点。同样，我们向下移动6、8到节点10，这是具有最大值的节点。

查找最小和最大节点的相同方法也适用于子树。具有根节点8的子树中的最小节点是7。该子树中具有最大值的节点是10。

返回最小节点的方法如下：

```py
    def find_min(self): 
        current = self.root_node 
        while current.left_child: 
            current = current.left_child 

        return current 
```

`while`循环继续获取左节点并访问它，直到最后一个左节点指向`None`。这是一个非常简单的方法。返回最大节点的方法相反，其中`current.left_child`现在变为`current.right_child`。

在BST中查找最小值或最大值需要**O**(*h*)，其中*h*是树的高度。

# 插入节点

BST的操作之一是需要将数据插入为节点。在我们的第一个实现中，我们必须自己插入节点，但在这里，我们将让树负责存储其数据。

为了使搜索成为可能，节点必须以特定的方式存储。对于每个给定的节点，其左子节点将保存小于其自身值的数据，如前所述。该节点的右子节点将保存大于其父节点的数据。

我们将通过使用数据5来创建一个新的整数BST。为此，我们将创建一个数据属性设置为5的节点。

现在，要添加值为3的第二个节点，3与根节点5进行比较：

![](assets/d212c46e-6046-4151-bb46-cf1ed26e4051.jpg)

由于5大于3，它将放在节点5的左子树中。我们的BST将如下所示：

![](assets/0e1ead82-8fb4-4ae6-a8de-d0b1578f7bf9.jpg)

树满足BST规则，左子树中的所有节点都小于其父节点。

要向树中添加值为7的另一个节点，我们从值为5的根节点开始比较：

![](assets/b0225066-e12d-48d9-8d42-cd8da6f62f91.jpg)

由于7大于5，值为7的节点位于此根节点的右侧。

当我们想要添加一个等于现有节点的节点时会发生什么？我们将简单地将其添加为左节点，并在整个结构中保持此规则。

如果一个节点已经有一个子节点在新节点应该放置的位置，那么我们必须沿着树向下移动并将其附加。

让我们添加另一个值为1的节点。从树的根开始，我们比较1和5：

![](assets/13de8d9e-97ba-4834-9756-cf31e34a95eb.jpg)

比较表明1小于5，因此我们将注意力转向5的左节点，即值为3的节点：

![](assets/3621290e-d166-4e10-9c5f-824992152daa.png)

我们将1与3进行比较，由于1小于3，我们向下移动到节点3的下一级并向左移动。但那里没有节点。因此，我们创建一个值为1的节点，并将其与节点3的左指针关联，以获得以下结构：

![](assets/e9c6c705-4546-4627-9866-f3c30baf1fae.jpg)

到目前为止，我们只处理包含整数或数字的节点。对于数字，大于和小于的概念是清晰定义的。字符串将按字母顺序比较，因此在那里也没有太大的问题。但是，如果您想在BST中存储自定义数据类型，您必须确保您的类支持排序。

现在让我们创建一个函数，使我们能够将数据作为节点添加到BST中。我们从函数声明开始：

```py
    def insert(self, data): 
```

到现在为止，你已经习惯了我们将数据封装在节点中的事实。这样，我们将`node`类隐藏在客户端代码中，客户端代码只需要处理树：

```py
        node = Node(data) 
```

首先检查是否有根节点。如果没有，新节点将成为根节点（我们不能没有根节点的树）：

```py
        if self.root_node is None: 
            self.root_node = node 
        else: 
```

当我们沿着树走时，我们需要跟踪我们正在处理的当前节点以及其父节点。变量`current`始终用于此目的：

```py
        current = self.root_node 
        parent = None 
        while True: 
            parent = current 
```

在这里，我们必须进行比较。如果新节点中保存的数据小于当前节点中保存的数据，则我们检查当前节点是否有左子节点。如果没有，这就是我们插入新节点的地方。否则，我们继续遍历：

```py
        if node.data < current.data: 
            current = current.left_child 
            if current is None: 
                parent.left_child = node 
                return 
```

现在我们处理大于或等于的情况。如果当前节点没有右子节点，则新节点将插入为右子节点。否则，我们继续向下移动并继续寻找插入点：

```py
        else: 
            current = current.right_child 
            if current is None: 
                parent.right_child = node 
                return 
```

在BST中插入一个节点需要**O**(*h*)，其中*h*是树的高度。

# 删除节点

BST上的另一个重要操作是节点的`删除`或`移除`。在此过程中，我们需要考虑三种情况。我们要删除的节点可能有以下情况：

+   没有子节点

+   一个子节点

+   两个子节点

第一种情况是最容易处理的。如果要删除的节点没有子节点，我们只需将其与其父节点分离：

![](assets/226a56a7-c9ba-47f9-b04b-e7cd9aff5349.png)

因为节点A没有子节点，所以我们只需将其与其父节点节点Z分离。

另一方面，当我们想要删除的节点有一个子节点时，该节点的父节点将指向该特定节点的子节点：

![](assets/b0c13358-279e-485a-8d1f-1ea43fe7e18e.png)

为了删除只有一个子节点节点5的节点6，我们将节点9的左指针指向节点5。父节点和子节点之间的关系必须得到保留。这就是为什么我们需要注意子节点如何连接到其父节点（即要删除的节点）。存储要删除节点的子节点。然后我们将要删除节点的父节点连接到该子节点。

当我们想要删除的节点有两个子节点时，会出现一个更复杂的情况：

![](assets/42141387-d400-4f4b-8ea8-774c2df6d9f0.png)

我们不能简单地用节点6或13替换节点9。我们需要找到节点9的下一个最大后代。这是节点12。要到达节点12，我们移动到节点9的右节点。然后向左移动以找到最左节点。节点12被称为节点9的中序后继。第二步类似于查找子树中的最大节点。

我们用节点9的值替换节点9的值，并删除节点12。删除节点12后，我们得到了一个更简单的节点删除形式，这已经在之前进行过处理。节点12没有子节点，因此我们相应地应用删除没有子节点的节点的规则。

我们的`node`类没有父引用。因此，我们需要使用一个辅助方法来搜索并返回具有其父节点的节点。该方法类似于`search`方法：

```py
    def get_node_with_parent(self, data): 
        parent = None 
        current = self.root_node 
        if current is None: 
            return (parent, None) 
        while True: 
            if current.data == data: 
                return (parent, current) 
            elif current.data > data: 
                parent = current 
                current = current.left_child 
            else: 
                parent = current 
                current = current.right_child 

        return (parent, current) 
```

唯一的区别是，在我们更新循环内的当前变量之前，我们使用`parent = current`存储其父级。执行实际删除节点的方法始于这个搜索：

```py
    def remove(self, data): 
        parent, node = self.get_node_with_parent(data) 

        if parent is None and node is None: 
            return False 

        # Get children count 
        children_count = 0 

        if node.left_child and node.right_child: 
            children_count = 2 
        elif (node.left_child is None) and (node.right_child is None): 
            children_count = 0 
        else: 
            children_count = 1 
```

我们将父节点和找到的节点传递给`parent`和`node`，代码为`parent, node = self.get_node_with_parent(data)`。了解要删除的节点有多少子节点是有帮助的。这就是`if`语句的目的。

之后，我们需要开始处理节点可以被删除的各种条件。`if`语句的第一部分处理节点没有子节点的情况：

```py
        if children_count == 0: 
            if parent: 
                if parent.right_child is node: 
                    parent.right_child = None 
                else: 
                    parent.left_child = None 
            else: 
                self.root_node = None 
```

`if parent:` 用于处理只有一个节点的BST的情况。

在要删除的节点只有一个子节点的情况下，`if`语句的`elif`部分执行以下操作：

```py
        elif children_count == 1: 
            next_node = None 
            if node.left_child: 
                next_node = node.left_child 
            else: 
                next_node = node.right_child 

            if parent: 
                if parent.left_child is node: 
                    parent.left_child = next_node 
                else: 
                    parent.right_child = next_node 
            else: 
                self.root_node = next_node 
```

`next_node`用于跟踪节点指向的单个节点的位置。然后我们将`parent.left_child`或`parent.right_child`连接到`next_node`。

最后，我们处理了要删除的节点有两个子节点的情况：

```py
        ... 
        else: 
            parent_of_leftmost_node = node 
            leftmost_node = node.right_child 
            while leftmost_node.left_child: 
                parent_of_leftmost_node = leftmost_node 
                leftmost_node = leftmost_node.left_child 

            node.data = leftmost_node.data 
```

在查找中序后继时，我们使用`leftmost_node = node.right_child`移动到右节点。只要存在左节点，`leftmost_node.left_child`将计算为`True`，`while`循环将运行。当我们到达最左节点时，它要么是叶节点（意味着它没有子节点），要么有一个右子节点。

我们使用`node.data = leftmost_node.data`更新即将被移除的节点的值：

```py
    if parent_of_leftmost_node.left_child == leftmost_node: 
       parent_of_leftmost_node.left_child = leftmost_node.right_child 
    else: 
       parent_of_leftmost_node.right_child = leftmost_node.right_child 
```

前面的陈述使我们能够正确地将最左节点的父节点与任何子节点正确连接。请注意等号右侧保持不变。这是因为中序后继只能有一个右子节点作为其唯一子节点。

`remove`操作的时间复杂度为**O**(*h*),其中*h*是树的高度。

# 搜索树

由于`insert`方法以特定方式组织数据，我们将遵循相同的过程来查找数据。在这个实现中，如果找到了数据，我们将简单地返回数据，如果没有找到数据，则返回`None`：

```py
    def search(self, data): 
```

我们需要从最顶部开始搜索，也就是从根节点开始：

```py
        current = self.root_node 
        while True: 
```

我们可能已经经过了一个叶节点，这种情况下数据不存在于树中，我们将返回`None`给客户端代码：

```py
            if current is None: 
                return None 
```

我们也可能已经找到了数据，这种情况下我们会返回它：

```py
            elif current.data is data: 
                return data 
```

根据BST中数据存储的规则，如果我们正在搜索的数据小于当前节点的数据，我们需要向树的左侧移动：

```py
            elif current.data > data: 
                current = current.left_child 
```

现在我们只剩下一个选择：我们正在寻找的数据大于当前节点中保存的数据，这意味着我们需要向树的右侧移动：

```py
            else: 
                current = current.right_child 
```

最后，我们可以编写一些客户端代码来测试BST的工作原理。我们创建一棵树，并在1到10之间插入一些数字。然后我们搜索该范围内的所有数字。存在于树中的数字将被打印出来：

```py
    tree = Tree() 
    tree.insert(5) 
    tree.insert(2) 
    tree.insert(7) 
    tree.insert(9) 
    tree.insert(1) 

    for i in range(1, 10): 
        found = tree.search(i) 
        print("{}: {}".format(i, found)) 
```

# 树的遍历

访问树中的所有节点可以通过深度优先或广度优先完成。这种遍历方式不仅适用于二叉搜索树，而是适用于树的一般情况。

# 深度优先遍历

在这种遍历方式中，我们会在向上继续遍历之前，沿着一个分支（或边）到达其极限。我们将使用递归方法进行遍历。深度优先遍历有三种形式，即`中序`、`前序`和`后序`。

# 中序遍历和中缀表示法

我们大多数人可能习惯用这种方式表示算术表达式，因为这是我们通常在学校里学到的方式。操作符被插入（中缀）在操作数之间，如`3 + 4`。必要时，可以使用括号来构建更复杂的表达式：`(4 + 5) * (5 - 3)`。

在这种遍历方式中，您将访问左子树、父节点，最后是右子树。

返回树中节点的中序列表的递归函数如下：

```py
    def inorder(self, root_node): 
        current = root_node 
        if current is None: 
            return 
        self.inorder(current.left_child) 
        print(current.data) 
        self.inorder(current.right_child) 
```

我们通过打印节点并使用`current.left_child`和`current.right_child`进行两次递归调用来访问节点。

# 前序遍历和前缀表示法

前缀表示法通常被称为波兰表示法。在这里，操作符在其操作数之前，如`+ 3 4`。由于没有优先级的歧义，因此不需要括号：`* + 4 5 - 5 3`。

要以前序方式遍历树，您将按照节点、左子树和右子树节点的顺序访问。

前缀表示法是LISP程序员所熟知的。

用于此遍历的递归函数如下：

```py
    def preorder(self, root_node): 
        current = root_node 
        if current is None: 
            return 
        print(current.data) 
        self.preorder(current.left_child) 
        self.preorder(current.right_child) 
```

注意递归调用的顺序。

# 后序遍历和后缀表示法。

后缀或**逆波兰表示法**（**RPN**）将操作符放在其操作数之后，如`3 4 +`。与波兰表示法一样，操作符的优先级永远不会引起混淆，因此不需要括号：`4 5 + 5 3 - *`。

在这种遍历方式中，您将访问左子树、右子树，最后是根节点。

`后序遍历`方法如下：

```py
    def postorder(self, root_node): 
        current = root_node 
        if current is None: 
            return 
        self.postorder(current.left_child) 
        self.postorder(current.right_child) 

        print(current.data)
```

# 广度优先遍历

这种遍历方式从树的根开始，并从树的一个级别访问节点到另一个级别：

![](assets/798e2ee2-65c7-4c0a-bfea-165a880ed447.png)

第1级的节点是节点4。我们通过打印其值来访问此节点。接下来，我们移动到第2级并访问该级别上的节点，即节点2和8。在最后一级，第3级，我们访问节点1、3、5和10。

这种遍历的完整输出是4、2、8、1、3、5和10。

这种遍历模式是通过使用队列数据结构实现的。从根节点开始，我们将其推入队列。队列前端的节点被访问（出队），然后打印并存储以备后用。左节点被添加到队列中，然后是右节点。由于队列不为空，我们重复这个过程。

算法的干运行将根节点4入队，出队并访问节点。节点2和8被入队，因为它们分别是左节点和右节点。节点2被出队以进行访问。它的左节点和右节点，即1和3，被入队。此时，队列前端的节点是8。我们出队并访问节点8，之后我们入队其左节点和右节点。因此，这个过程一直持续，直到队列为空。

算法如下：

```py
    from collections import deque 
    class Tree: 
        def breadth_first_traversal(self): 
            list_of_nodes = [] 
            traversal_queue = deque([self.root_node]) 
```

我们将根节点入队，并在`list_of_nodes`列表中保留一个访问过的节点列表。`dequeue`类用于维护队列：

```py
        while len(traversal_queue) > 0: 
            node = traversal_queue.popleft() 
            list_of_nodes.append(node.data) 
```

```py
            if node.left_child: 
                traversal_queue.append(node.left_child) 

            if node.right_child: 
                traversal_queue.append(node.right_child) 
        return list_of_nodes 
```

如果`traversal_queue`中的元素数量大于零，则执行循环体。队列前端的节点被弹出并附加到`list_of_nodes`列表。第一个`if`语句将`node`的左子节点入队，如果存在左节点。第二个`if`语句对右子节点执行相同的操作。

`list_of_nodes`在最后一个语句中返回。

# 二叉搜索树的好处

我们现在简要地看一下，为什么使用BST比使用列表进行搜索更好。假设我们有以下数据集：5、3、7、1、4、6和9。使用列表，最坏的情况需要在找到搜索项之前搜索整个包含七个元素的列表：

![](assets/720691f5-6681-473f-a80c-1f1cdba24822.jpg)

搜索`9`需要六次跳跃。

使用树，最坏的情况是三次比较：

![](assets/ace39d6e-f99f-432c-96fa-258b74bd400c.jpg)

搜索`9`需要两步。

然而请注意，如果你按照1、2、3、5、6、7、9的顺序将元素插入树中，那么这棵树将不会比列表更有效。我们需要首先平衡树：

![](assets/260a95e2-481f-4441-b0dc-991d1aafed8b.jpg)

因此，重要的不仅是使用BST，而且选择自平衡树有助于改进`search`操作。

# 表达式树

树结构也用于解析算术和布尔表达式。例如，`3 + 4`的表达式树如下所示：

![](assets/d3bff613-f1df-495f-8f23-3b101b9b8633.jpg)

对于稍微复杂的表达式`(4 + 5) * (5-3)`，我们将得到以下结果：

![](assets/bf51f42d-d166-483b-93ff-0c22004c45da.jpg)

# 解析逆波兰表达式

现在我们将为后缀表示法中的表达式构建一棵树。然后我们将计算结果。我们将使用一个简单的树实现。为了保持简单，因为我们将通过合并较小的树来增长树，我们只需要一个树节点实现：

```py
    class TreeNode: 
        def __init__(self, data=None): 
            self.data = data 
            self.right = None 
            self.left = None 
```

为了构建树，我们将寻求栈的帮助。很快你就会明白为什么。但目前，让我们创建一个算术表达式并设置我们的栈：

```py
        expr = "4 5 + 5 3 - *".split() 
        stack = Stack() 
```

由于Python是一种试图具有合理默认值的语言，它的`split()`方法默认情况下会在空格上拆分。（如果你仔细想想，这很可能也是你期望的。）结果将是`expr`是一个包含值4、5、+、5、3、-和*的列表。

expr列表的每个元素都可能是操作符或操作数。如果我们得到一个操作数，那么我们将其嵌入到一个树节点中并将其推入堆栈。另一方面，如果我们得到一个操作符，那么我们将操作符嵌入到一个树节点中，并将其两个操作数弹出到节点的左右子节点中。在这里，我们必须小心确保第一个弹出的操作数进入右子节点，否则我们将在减法和除法中出现问题。

以下是构建树的代码：

```py
    for term in expr: 
        if term in "+-*/": 
            node = TreeNode(term) 
            node.right = stack.pop() 
            node.left = stack.pop() 
        else: 
            node = TreeNode(int(term)) 
        stack.push(node) 
```

请注意，在操作数的情况下，我们执行了从字符串到整数的转换。如果需要支持浮点数操作数，可以使用`float()`。

在这个操作结束时，我们应该在堆栈中只有一个元素，它包含了完整的树。

现在我们可能想要评估表达式。我们构建了以下小函数来帮助我们：

```py
    def calc(node): 
        if node.data is "+": 
            return calc(node.left) + calc(node.right) 
        elif node.data is "-": 
            return calc(node.left) - calc(node.right) 
        elif node.data is "*": 
            return calc(node.left) * calc(node.right) 
        elif node.data is "/": 
            return calc(node.left) / calc(node.right) 
        else: 
            return node.data 
```

这个函数非常简单。我们传入一个节点。如果节点包含一个操作数，那么我们就简单地返回该值。然而，如果我们得到一个操作符，那么我们就对节点的两个子节点执行操作符代表的操作。然而，由于一个或多个子节点也可能包含操作符或操作数，我们在两个子节点上递归调用`calc()`函数（要记住每个节点的所有子节点也都是节点）。

现在我们只需要从堆栈中弹出根节点并将其传递给`calc()`函数，我们就应该得到计算的结果：

```py
    root = stack.pop() 
    result = calc(root) 
    print(result) 
```

运行这个程序应该得到结果18，这是`(4 + 5) * (5 - 3)`的结果。

# 平衡树

之前我们提到，如果节点按顺序插入树中，那么树的行为就更像是一个列表，也就是说，每个节点恰好有一个子节点。我们通常希望尽量减少树的高度，填满树中的每一行。这个过程称为平衡树。

有许多类型的自平衡树，例如红黑树、AA树和替罪羊树。这些树在修改树的每个操作（如插入或删除）期间平衡树。

还有一些外部算法可以平衡树。这样做的好处是你不需要在每次操作时都平衡树，而是可以在需要时才进行平衡。

# 堆

在这一点上，我们简要介绍堆数据结构。堆是树的一种特殊形式，其中节点以特定的方式排序。堆分为最大堆和最小堆。在最大堆中，每个父节点必须始终大于或等于其子节点。因此，根节点必须是树中最大的值。最小堆则相反。每个父节点必须小于或等于其两个子节点。因此，根节点保存最小的值。

堆用于许多不同的事情。首先，它们用于实现优先队列。还有一种非常高效的排序算法，称为堆排序，使用了堆。我们将在后续章节中深入研究这些内容。

# 总结

在本章中，我们看了树结构和它们的一些示例用途。我们特别研究了二叉树，这是树的一个子类型，其中每个节点最多有两个子节点。

我们看到了二叉树如何作为可搜索的数据结构与BST一起使用。我们发现，在大多数情况下，在BST中查找数据比在链表中更快，尽管如果数据按顺序插入，情况就不同了，除非树是平衡的。

广度优先和深度优先搜索遍历模式也使用队列递归实现了。

我们还看了二叉树如何用来表示算术或布尔表达式。我们构建了一个表达式树来表示算术表达式。我们展示了如何使用栈来解析以逆波兰表示法编写的表达式，构建表达式树，最后遍历它以获得算术表达式的结果。

最后，我们提到了堆，这是树结构的一种特殊形式。在本章中，我们试图至少奠定堆的理论基础，以便在接下来的章节中为不同的目的实现堆。
