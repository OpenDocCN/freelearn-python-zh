# 第七章。集合：无重复

在计算机科学的范畴内，**集合**通常被用作一个简单的对象集合，其中不包含重复项。然而，在更广泛的数学领域，集合是一个抽象的数据结构，可以描述为按无特定顺序存储的不同对象或值的集合。为了讨论的目的，我们将选择将集合视为数学有限集合的计算机实现。

当处理可以应用集合论数学概念的问题时，集合数据结构提供了一组强大的工具，用于组合和检查类似对象集合之间的关系。然而，即使在集合论和数学之外，集合数据结构也提供了在日常生活中可能有用的功能。例如，由于集合自然地消除了重复项，任何需要维护或编辑唯一元素集合的应用程序都将从存储对象在集合数据结构中受益。同样，如果你需要从现有集合中消除重复项，大多数集合数据结构的实现都将允许你从一个数组集合中创建一个新的集合；在这样做的时候，你将自动过滤掉重复项。总的来说，集合是一个相对简单的数据结构，在分析数据集合时提供了巨大的功能性和力量。

在本章中，我们将涵盖以下内容：

+   集合数据结构的定义

+   集合论

+   初始化集合

+   常见集合运算

+   重新审视登录到服务的用户问题

+   案例研究 - 音乐播放列表

+   基于哈希表的集合

+   基于树的集合

+   基于数组的集合

# 集合论

集合的概念相对简单，但在实践中，由于其数学起源，具体的实现可能有些难以理解。因此，为了完全欣赏集合数据结构，有必要检查构建集合数据结构的基础——**集合论**的一些特性和函数。集合论是研究对象集合或*集合*的数学分支。尽管集合论是数学中的一个主要研究领域，有许多相互关联的子领域，但我们实际上只需要检查五个用于结合和关联集合的函数，以理解集合数据结构：

+   **并集**：并集是结合和关联集合的基本方法之一。一系列 *n* 个集合的并集是仅包含那些在这些集合中出现的不同元素的集合。这意味着，如果你将集合 *A* 和 *B* 结合起来，结果集合将只包含来自集合 *A* 和 *B* 的唯一元素。如果一个元素同时存在于 *A* 和 *B* 中，它将只在我们结果集中出现一次。我们使用符号 *A* ∪ *B* 来表示集合 *A* 和 *B* 的并集。以下维恩图表示了两个集合的并集：

![集合论](img/00008.jpeg)

+   **交集**：交集是组合和关联集合的第二种基本方法。一个由 *n* 个集合组成的交集是存在于每个被评估集合中的元素集合。因此，如果我们检查集合 *A* 和 *B* 的交集，我们的结果集合将只包括存在于 *A* 和 *B* 中的那些元素。任何只属于 *A* 或 *B* 的元素将被丢弃。我们使用符号 *A* ∩ *B* 来表示集合 *A* 与集合 *B* 的交集。以下维恩图表示了两个集合的交集：

![集合论](img/00009.jpeg)

+   **差集**：差集操作是交集操作的相反操作。一个由 *n* 个集合组成的差集是每个被评估集合中唯一的元素集合。如果我们检查集合 *A* 和 *B* 的差集，我们的结果集合将只包括存在于 *A* 或 *B* 中的那些元素。任何属于 *A* 和 *B* 交集的元素将被丢弃。我们使用符号 *A* Δ *B* 来表示集合 *A* 和 *B* 之间的差集。以下维恩图表示了两个集合的差集：

![集合论](img/00010.jpeg)

+   **补集**：*A* 在 *B* 中的补集，或称为**相对补集**，是存在于 *B* 但不存在于 *A* 中的元素集合。如果我们检查集合 *A* 和 *B* 的补集，只有那些只属于 *B* 的元素将包含在我们的结果集合中。任何只属于 *A* 或是 *A* 和 *B* 交集的元素将被丢弃。我们使用符号 *B\**A* 来表示集合 *A* 相对于集合 *B* 的相对补集。以下维恩图表示了两个集合的补集：

![集合论](img/00011.jpeg)

+   **子集**：子集是组合和关联集合的最终基本方法。子集操作确定集合 *A* 是否是集合 *B* 的子集，或者换句话说，集合 *B* 是否是集合 *A* 的**超集**。一个集合是另一个集合的子集的关系称为**包含**，或者当一个集合是另一个集合的超集时，称为**包含**。在下一个图中，我们可以说 *A* 是 *B* 的子集，或者 *B* 是 *A* 的超集。我们使用符号 *A* ⊂ *B* 来表示集合 *A* 是集合 *B* 的包含，或者 *B ⊃* *A* 来表示集合 *B* 是集合 *A* 的包含。

![集合论](img/00012.jpeg)

# 初始化集合

集合在开发中并不十分常见，但我们正在检查的每种语言都支持以某种具体形式实现的数据结构。以下是一些初始化集合、向集合中添加一些值（包括一个重复值）以及在每一步后将集合的计数打印到控制台的示例。

**C#**

C#通过`HashSet<T>`类提供了集合数据结构的具体实现。由于此类是泛型的，调用者可以定义用于元素的类型。例如，以下示例初始化了一个新集合，其中元素将是`string`类型：

```py
    HashSet<string, int> mySet = new HashSet<string>(); 
    mySet.Add("green");  
    Console.WriteLine("{0}", mySet.Count); 
    mySet.Add("yellow");  
    Console.WriteLine("{0}", mySet.Count); 
    mySet.Add("red");  
    Console.WriteLine("{0}", mySet.Count); 
    mySet.Add("red");  
    Console.WriteLine("{0}", mySet.Count); 
    mySet.Add("blue");  
    Console.WriteLine("{0}", mySet.Count); 

    /* Output:  
    1 
    2 
    3 
    3 since "red" already exists in the collection 
    4 
    */ 

```

**Java**

Java 提供了一个`HashSet<E>`类以及其他实现`Set<E>`接口的类。在本章中，我们将仅查看`HashSet<E>`类的示例：

```py
    HashSet<String> mySet = new HashSet< >(); 
    mySet.add("green");  
    System.out.println(mySet.size()); 
    mySet.add("yellow");  
    System.out.println(mySet.size()); 
    mySet.add("red");  
    System.out.println(mySet.size()); 
    mySet.add("red");  
    System.out.println(mySet.size()); 
    mySet.add("blue");  
    System.out.println(mySet.size()); 

    /* Output:  
    1 
    2 
    3 
    3 since "red" already exists in the collection 
    4 
    */ 

```

**Objective-C**

Objective-C 提供了不可变和可变的集合类，`NSSet`和`NSMutableSet`。在本章中，我们只将详细检查可变版本：

```py
    NSMutableSet *mySet = [NSMutableSet set]; 
    [mySet addObject:@"green"]; 
    NSLog(@"%li", (long)[mySet count]);  
    [mySet addObject:@"yellow"]; 
    NSLog(@"%li", (long)[mySet count]); 
    [mySet addObject:@"red"]; 
    NSLog(@"%li", (long)[mySet count]); 
    [mySet addObject:@"red"]; 
    NSLog(@"%li", (long)[mySet count]); 
    [mySet addObject:@"blue"]; 
    NSLog(@"%li", (long)[mySet count]);  

    /* Output:  
    1 
    2 
    3 
    3 since "red" already exists in the collection 
    4 
    */ 

```

**Swift**

Swift 中的集合使用`Set`类创建。当使用`var`初始化为**变量**时，Swift 集合是可变的，但它们也可以通过使用`let`初始化为**常量**来创建不可变。在本章中，我们只将详细检查可变版本：

```py
    let mySet: Set<String> = Set<String>() 
    mySet.insert(@"green")  
    print(mySet.count)  
    mySet.insert(@"yellow")  
    print(mySet.count) 
    mySet.insert(@"red")  
    print(mySet.count) 
    mySet.insert(@"red")  
    print(mySet.count) 
    mySet.insert(@"blue")  
    print(mySet.count)  

    /* Output:  
    1 
    2 
    3 
    3 since "red" already exists in the collection 
    4 
    */ 

```

## 集合操作

并非所有集合数据结构的具体实现都公开相同的操作方法。然而，更常见的操作应该是可用的，或者可以通过开发者的需要来提供。在检查这些操作时，请注意语言与之前讨论的集合理论操作语言的相似性。您会发现大多数集合数据结构的功能将紧密地反映集合理论的一般功能：

+   **添加**: 添加操作，有时称为插入，如果该对象尚未存在于集合中，则将其引入集合。这种防止重复对象被添加到集合中的功能是使用集合而不是许多其他数据结构的核心优势之一。大多数集合数据结构的实现将返回一个布尔值，表示元素是否可以添加到集合中。添加操作具有**O**(*n*)的成本。

+   **删除**: 删除或删除操作允许调用者从集合中删除一个值或对象，如果它存在。大多数集合数据结构的实现返回一个布尔值，表示删除操作是否成功。删除操作具有**O**(*n*)的成本。

+   **容量**: 容量操作返回集合可以存储的最大值数。这不是开发者在讨论的四种语言中自然看到的操作，因为这些语言中找到的每个可变集合都可以根据需要动态调整大小。然而，一些实现确实允许将集合的大小限制为其定义的一部分。容量具有**O**(1)的操作成本。

+   **并集**: 并集操作返回一个包含两个或多个集合中唯一元素的新集合。因此，此操作的最坏情况成本为**O**(*n*+*m*)，其中*n*是第一个集合的大小，*m*是第二个集合的大小。

+   **intersection**: 交集操作仅返回两个或多个集合之间共享的元素。这意味着，如果您向该方法提供两个集合，您将仅获得存在于两个集合中的那些元素。交集的成本为 **O**(*n***m*)，其中 *n* 是第一个集合的大小，*m* 是第二个集合的大小。有趣的是，如果您尝试对三个或更多集合进行交集操作，成本将变为 *(n-1) ** **O**(*L*)，其中 *n* 是参与操作的集合数量，*L* 是系列中最大集合的大小。显然，这个成本相当高，并且同时使用此操作在多个集合上可能会迅速失控。

+   **difference**: 差集操作是交集操作的相反，仅返回每个集合中独特的元素。此操作的成本为 **O**(*m*)，其中 *m* 是两个被评估集合中较短的那个的长度。

+   **subset**: 子集操作返回一个布尔值，确定集合 *A* 是否是集合 *B* 的子集。对于集合 *A* 被认为是集合 *B* 的子集，集合 *A* 中的每个元素也必须包含在集合 *B* 中。如果集合 *A* 的只有部分元素包含在集合 *B* 中，那么集合 *A* 和 *B* 有交集，但 *A* 不是 *B* 的子集。此操作的操作成本为 **O**(*m*)，其中 *m* 是集合 *A* 的长度。

+   **count**: `count` 操作，或称为大小，表示特定集合的 **基数**，这实际上是集合论中表达集合中元素数量的方式。计数通常是集合上的一个简单属性，因此具有 **O**(*1*) 的成本。

+   **isEmpty**: `isEmpty` 操作返回一个布尔值，表示集合是否包含任何元素。一些实现提供了相应的 `isFull` 操作，但仅限于那些可以将集合容量限制为特定值的实例。`isEmpty` 和 `isFull` 都有 **O**(*1*) 的成本。

# 示例：回顾登录服务的用户

让我们再次回顾 第二章 中的用户登录服务问题，*数组：基础集合*，并检查如果我们选择集合而不是数组或列表作为底层数据结构，代码将如何改变。

**C#**

在这个例子中，我们将`List<User>`对象替换为了`HashSet<User>`对象。我们的代码大部分没有改变，但请注意，我们排除了`CanAddUser(User)`方法。原本，这个方法通过确保集合有空间容纳另一个对象，然后确保要添加的对象尚未包含在集合中来验证已认证用户的操作。集合数据结构消除了进行第二步的需要，因为它本质上防止了重复对象的添加。由于我们的类现在只需要进行容量检查验证，我们可以将这个检查与`UserAuthenticated(User)`功能一起内联处理。作为额外的奖励，我们现在可以轻松地报告用户是否成功添加，因为`HashSet<T>.Add(T)`在成功时返回`true`，当对象已存在于集合中时返回`false`：

```py
    public class LoggedInUserSet 
    { 
        HashSet<User> _users; 

        public LoggedInUserSet() 
        { 
            _users = new HashSet<User>(); 
        } 

        public bool UserAuthenticated(User user) 
        { 
            if (_users.Count < 30) 
            { 
                return _users.Add(user); 
            } 
            return false; 
        } 

        public void UserLoggedOut(User user) 
        { 
            _users.Remove(user); 
        } 
    } 

```

**Java**

我们 Java 示例中的更改几乎与我们的 C#示例相同。同样，我们用`HashSet<User>`对象替换了`List<User>`对象。我们的代码大部分没有改变，除了排除了`canAddUser(User)`方法。在 Java 中，`HashSet<E>`类实现了`Set<E>`接口，并基于集合数据结构，消除了在添加对象之前检查对象是否存在于集合中的需要。由于我们的类现在只需要进行容量检查验证，我们可以将这个检查与`userAuthenticated(User)`功能一起内联处理。同样，我们现在可以轻松地报告用户是否成功添加，因为`HashSet<E>.add(E)`在成功时返回`true`，当对象已存在于集合中时返回`false`：

```py
    HashSet<User> _users; 

    public LoggedInUserSet() 
    { 
        _users = new HashSet<User>(); 
    } 

    public boolean userAuthenticated(User user) 
    { 
        if (_users.size() < 30) 
        { 
            return _users.add(user); 
        } 
        return false; 
    } 

    public void userLoggedOut(User user) 
    { 
        _users.remove(user); 
    } 

```

**Objective-C**

我们 Objective-C 示例的更改产生了一些有趣的结果。尽管我们用`NSMutableArray`集合替换了`NSMutableSet`集合，但大部分代码保持不变，包括我们不会返回一个表示`addObject:`操作成功或失败的`BOOL`值。这是因为`addObject:`不返回任何值；如果我们将其包含在`userAuthenticated:`中，我们不得不在调用集合上的`addObject:`之前调用`containsObject:`方法。由于这个练习的整个目的是使用集合来消除在添加新对象之前检查重复的需要，重新引入这个功能将违背初衷，并可能使我们处于比简单地坚持使用数组或列表更昂贵的位置。

这并不是说没有有效的应用可以从集合以及关于`addObject:`操作成功或失败的报告中获得好处；这只是说这种情况并不适用：

```py
    @interface EDSLoggedInUserSet() 
    { 
        NSMutableSet *_users; 
    } 
    @end 

    @implementation EDSLoggedInUserSet 
    -(instancetype)init 
    { 
        if (self = [super init]) 
        { 
            _users = [NSMutableSet set]; 
        } 
        return self; 
    } 

    -(void)userAuthenticated:(EDSUser *)user 
    { 
        if ([_users count] < 30) 
        { 
            [_users addObject:user]; 
        } 
    } 

    -(void)userLoggedOut:(EDSUser *)user 
    { 
        [_users removeObject:user]; 
    } 

```

**Swift**

我们 Swift 示例的结果几乎与我们的 Objective-C 示例完全相同。再次强调，我们正在用集合替换数组，但在 Swift 中集合的工作方式与在 Objective-C 中类似。因此，我们的最终代码更加简洁，但并不立即提供与我们的 C#和 Java 实现相同的功能：

```py
    var _users: Set<User> = Set<User>() 

    public func userAuthenticated(user: User) 
    { 
        if (_users.count < 30) 
        { 
            _users.insert(user) 
        } 
    } 

    public func userLoggedOut(user: User) 
    { 
        if let index = _users.indexOf(user) 
        { 
            _users.removeAtIndex(index) 
        } 
    } 

```

## 我们需要一份合同

如果你仔细观察解决已登录用户业务问题的三个方案中的每一个，你可能会注意到它们都共享一些公共方法。在我们的数组实现、列表实现和集合实现中，我们有两个名为 `UserAuthenticated()` 和 `UserLoggedOut()` 的公共方法，或者根据语言的不同，这些名称可能会有所变化。如果我们只是选择最适合我们需求的一个实现并继续前进，这不会成为问题。然而，如果我们有合理的理由保留这些类中的每一个，以便在特定的环境条件下高效工作，那会怎样呢？

实际上，看到多个类共享相同的公共方法，但在底层有独特实现的代码非常普遍。如果我们简单地创建三个（或更多）完全独立的独立实现，我们的应用程序将产生一个 *代码异味*。这是因为，每当我们想要使用特定的实现时，我们都需要通过名称来调用它，这需要我们对哪些类和实现可用有一定的预先了解。此外，尽管我们的代码可能运行得很好，但它将是脆弱的、不可扩展的，并且长期维护起来会更加困难。

一个更好的解决方案将涉及定义一个每个类都实现的合同。在 C#或 Java 中，我们会定义一个接口，而在 Objective-C 和 Swift 中，我们会定义一个协议。这两种模式之间的区别主要在于语义，因为它们都将为我们的调用者提供方法名称、方法期望的内容以及方法将返回的内容。重要的是，通过这样做，我们极大地简化并加强了功能实现和调用类结构的实现。

# 案例研究：音乐播放列表

**业务问题**：一个音乐流媒体服务希望为用户提供更好的流媒体体验。目前，用户播放列表只是一个简单的歌曲集合，被倒入一个没有提供过滤或排序集合方式的桶中。内容管理团队已经听到了用户的投诉，并已将构建更好的播放列表的任务分配给了工程团队。

这个新的播放列表工具将会有几个关键要求。更基本的要求包括能够从列表中添加和删除歌曲，能够区分空列表和包含元素的列表，以及能够报告列表中元素的总数。对于那些对付费高级服务不感兴趣的客户，列表将限制为 100 首歌曲，因此我们的播放列表工具还必须具备设置容量和轻松识别容量已满的能力。

此外，许多高级用户在他们的播放列表中拥有数千首歌曲，以及针对从骑自行车到洗衣服等一切事物的多个主题播放列表。对于这些用户，播放列表工具必须包括一些高级分析和编辑功能。首先，必须有一种简单的方法来轻松合并播放列表，并且由于我们不希望同时存在于两个播放列表中的歌曲出现两次，这种合并必须防止重复。接下来，播放列表应该能够轻松识别两个列表之间重复的歌曲，以及识别特定列表中独特歌曲。最后，一些用户可能希望了解有关他们播放列表集合的更多信息，例如是否有一个播放列表作为另一个播放列表的一部分存在。基于这些要求，开发者决定使用集合来表示播放列表将是最有效的方法，因此核心类的功能将基于该数据结构。

**C#**

C# 提供了泛型集合 `HashSet<T>`。这个类提供了我们在具体的集合实现中期望看到的所有基本操作，并增加了泛型类型转换的额外好处：

```py
    HashSet<Song> _songs; 
    public Int16 capacity { get; private set; } 
    public bool premiumUser { get; private set; } 
    public bool isEmpty  
    { 
        get 
        { 
            return _songs.Count == 0; 
        } 
    } 

    public bool isFull 
    { 
        get 
        { 
            if (this.premiumUser) 
            { 
                return false; 
            } 
            else 
            { 
                return _songs.Count == this.capacity; 
            } 
        } 
    } 

    public PlaylistSet(bool premiumUser, Int16 capacity) 
    { 
        _songs = new HashSet<Song>(); 
        this.premiumUser = premiumUser; 
        this.capacity = capacity;  
    } 

```

使用 `HashSet<T>` 接口，我们为我们的类创建了一个名为 `_songs` 的私有字段。我们的构造函数实例化了这个字段，为我们提供了构建 `PlaylistSet` 类的基础数据结构。我们还创建了四个公共字段：`capacity`、`premiumUser`、`isEmpty` 和 `isFull`。`capacity` 字段存储非高级用户可以在他们的播放列表中存储的最大歌曲数量，而 `premiumUser` 表示这个列表是否属于高级账户。`isEmpty` 和 `isFull` 字段允许我们的类轻松实现同名操作。`isEmpty` 字段简单地返回集合的计数是否为 `0`。`isFull` 字段首先检查这个列表是否属于高级账户。如果是 `true`，则集合永远不会满，因为我们允许高级用户在他们的播放列表中存储无限数量的歌曲。如果这个列表不属于高级账户，我们的获取器确保 `_songs` 的当前计数没有超过容量，并返回这个比较：

```py
    public bool AddSong(Song song) 
    { 
        if (!this.isFull) 
        { 
            return _songs.Add(song); 
        } 
        return false; 
    } 

```

`AddSong(Song song)` 方法为我们这个类提供了 *添加* 功能。该方法首先确认集合没有满。如果是这样，该方法返回 `false`，因为我们不能向列表中添加更多歌曲。否则，该方法返回 `HashSet<T>.Add(T)` 的结果，如果 `song` 被添加，则返回 `true`，这意味着歌曲不在列表中。

```py
    public bool RemoveSong(Song song) 
    { 
        return _songs.Remove(song); 
    } 

```

`RemoveSong(Song song)` 方法为我们这个类提供了 *移除* 功能。这个方法简单地返回 `HashSet<T>.Remove(T)` 的结果，如果歌曲存在于列表中，则返回 `true`；否则，返回 `false`：

```py
    public void MergeWithPlaylist(HashSet<Song> playlist) 
    { 
        _songs.UnionWith(playlist); 
    } 

```

`MergeWithPlaylist(HashSet<Song> playlist)` 方法为我们这个类提供了 *并集* 功能。幸运的是，`HashSet<T>` 通过 `Union(HashSet<T>)` 方法公开了并集功能，所以我们的方法只是简单地调用它。在这种情况下，`Union()` 将合并 `playlist` 参数和我们的现有 `_songs` 列表：

```py
    public HashSet<Song> FindSharedSongsInPlaylist(HashSet<Song> playlist) 
    { 
        HashSet<Song> songsCopy = new HashSet<Song>(_songs); 
        songsCopy.IntersectWith(playlist); 
        return songsCopy; 
    } 

```

接下来，`FindSharedSongsInPlaylist(HashSet<Song> playlist)` 方法为我们这个类提供了 *交集* 功能。同样，`HashSet<T>` 方便地提供了 `IntersectWith(HashSet<T>)` 方法，我们这个方法正是利用了它。请注意，这个方法不会修改我们的列表，而是返回我们的列表和 `playlist` 参数的实际交集。我们这样做是因为仅仅消除列表中独特歌曲并不是很有用。这个方法将用于整体应用程序中的其他功能的信息目的。

由于我们不是修改现有的列表，而是只返回关于交集的信息，我们的方法首先使用重载的 `HashSet<T>` 对象复制 `_songs` 集合。然后，我们的方法修改复制的列表，并返回交集操作的结果：

```py
    public HashSet<Song> FindUniqueSongs(HashSet<Song> playlist) 
    { 
        HashSet<Song> songsCopy = new HashSet<Song>(_songs); 
        songsCopy.ExceptWith(playlist); 
        return songsCopy; 
    } 

```

`FindUniqueSongs(HashSet<Song> playlist)` 方法为我们这个类提供了 *差集* 功能，并且其工作方法与上一个方法非常相似。同样，这个方法不会修改我们现有的集合，而是返回对复制的集合和 `playlist` 参数的 `ExceptWith()` 操作的结果：

```py
    public bool IsSubset(HashSet<Song> playlist) 
    { 
        return _songs.IsSubsetOf(playlist); 
    } 

    public bool IsSuperset(HashSet<Song> playlist) 
    { 
        return _songs.IsSupersetOf(playlist); 
    } 

```

`IsSubset(HashSet<Song> playlist)` 和 `IsSuperset(HashSet<Song> playlist)` 方法提供了它们名字所暗示的功能。这些方法分别利用 `HashSet<T>.IsSubSetOf(HashSet<T>)` 和 `HashSet<T>.IsSuperSetOf(HashSet<T>)` 方法，并返回一个表示这些比较结果的布尔值：

```py
    public int TotalSongs() 
    { 
        return _songs.Count; 
    } 

```

最后，`TotalSongs()` 方法返回 `_songs` 集合中找到的元素数量，为我们这个集合提供了 *计数* 功能。

**Java**

Java 提供了实现 `Set<E>` 接口的泛型集合 `HashSet<E>`。这个类提供了我们在具体的集合实现中期望看到的所有基本操作，并增加了泛型类型转换的额外好处：

```py

    private HashSet<Song> _songs; 
    public int capacity; 
    public boolean premiumUser; 
    public boolean isEmpty() 
    { 
        return _songs.size() == 0; 
    } 

    public boolean isFull() 
    { 
        if (this.premiumUser) 
        { 
            return false; 
        } 
        else { 
            return _songs.size() == this.capacity; 
        } 
    } 

    public PlaylistSet(boolean premiumUser, int capacity) 
    { 
        _songs = new HashSet<>(); 
        this.premiumUser = premiumUser; 
        this.capacity = capacity; 
    } 

```

使用 `HashSet<E>`，我们为我们的类创建了一个名为 `_songs` 的私有字段。我们的构造函数实例化这个字段，为我们提供了构建 `PlaylistSet` 类的基础数据结构。我们还创建了两个公共字段和两个公共访问器：`capacity`、`premiumUser`、`isEmpty()` 和 `isFull()`。`capacity` 字段存储非高级用户可以在他们的播放列表中存储的最大歌曲数量，而 `premiumUser` 表示这个列表是否属于高级账户。`isEmpty()` 和 `isFull()` 访问器允许我们的类轻松实现同名操作。这两个访问器的工作方式与它们的 C# 字段对应物完全相同。`isEmpty()` 方法简单地返回集合的计数是否为 `0`。`isFull()` 方法首先检查这个列表是否属于高级账户。

如果是 `true`，则集合永远不会满，因为我们允许高级用户在他们的播放列表中存储无限数量的歌曲。如果这个列表不属于高级账户，我们的获取器确保 `_songs` 的当前计数没有超过 `capacity`，并返回这个比较结果：

```py
    public boolean addSong(Song song) 
    { 
        if (!this.isFull()) 
        { 
            return _songs.add(song); 
        } 
        return false; 
    } 

```

`addSong(Song song)` 方法为我们这个类提供了 *添加* 功能。这个方法首先确认集合没有满。如果是这样，方法返回 `false`，因为我们不能向列表中添加更多歌曲。否则，方法返回 `HashSet<E>.add(E)` 的结果，如果歌曲被添加，并且只有在歌曲尚未存在于这个播放列表中时才会返回 `true`：

```py
    public boolean removeSong(Song song) 
    { 
        return _songs.remove(song); 
    } 

```

`removeSong(Song song)` 方法为我们这个类提供了 *删除* 功能。这个方法简单地返回 `HashSet<E>.remove(E)` 的结果，如果歌曲存在于集合中，则返回 `true`；否则，返回 `false`。

```py
    public void mergeWithPlaylist(HashSet<Song> playlist) 
    { 
        _songs.addAll(playlist); 
    } 

```

`mergeWithPlaylist(HashSet<Song> playlist)` 方法为我们这个类提供了 *并集* 功能，这也是我们的类开始真正与之前的 C# 示例不同的地方。`HashSet<E>` 提供了我们需要的 *并集* 功能，但只能通过调用 `HashSet<E>.addAll(HashSet<E>)` 方法来实现。这个方法接受一个 `Song` 对象的集合作为参数，并尝试将每个对象添加到我们的 `_songs` 集合中。如果被添加的 `Song` 元素已经存在于 `_songs` 集合中，该元素将被丢弃，只留下来自两个列表或两个集合的唯一的 `Song` 对象：

```py
    public HashSet<Song> findSharedSongsInPlaylist(HashSet<Song> playlist) 
    { 
        HashSet<Song> songsCopy = new HashSet<>(_songs); 
        songsCopy.retainAll(playlist); 
        return songsCopy; 
    } 

```

接下来，`findSharedSongsInplaylist(HashSet<Song> playlist)` 方法为我们这个类提供了 *交集* 功能。同样，`HashSet<E>` 提供了交集功能，但不是直接提供。我们的方法使用 `HashSet<E>.retainAll(HashSet<E>)` 方法，该方法保留 `_songs` 集合中所有也存在于 `playlist` 参数中的元素，或者两个集合的交集。正如我们的 C# 示例一样，我们并没有在原地修改 `_songs` 集合，而是返回 `_songs` 的一个副本和 `playlist` 参数之间的交集：

```py
    public HashSet<Song> findUniqueSongs(HashSet<Song> playlist) 
    { 
        HashSet<Song> songsCopy = new HashSet<>(_songs); 
        songsCopy.removeAll(playlist); 
        return songsCopy; 
    } 

```

`findUniqueSongs(HashSet<Song> playlist)` 方法为我们提供的类提供 *差异* 功能。再次，`HashSet<E>` 揭示了差异功能，但通过 `removeAll(HashSet<E>)` 方法。`removeAll()` 方法移除所有在播放列表参数或两个集合之间的差异中包含的 `_songs` 元素。同样，此方法不会修改我们现有的集合，而是返回 `_songs` 复制和 `playlist` 参数上的 `removeAll()` 方法或差异操作的结果：

```py
    public boolean isSubset(HashSet<Song> playlist) 
    { 
        return _songs.containsAll(playlist); 
    } 

    public boolean isSuperset(HashSet<Song> playlist) 
    { 
        return playlist.containsAll(_songs); 
    } 

```

`isSubset(HashSet<Song> playlist)` 和 `isSuperset(HashSet<Song> playlist)` 方法提供了同名功能。这两个方法都利用了 `HashSet<E>.containsAll(HashSet<E>)` 方法，并返回一个布尔值，表示这些比较的结果。我们的方法只是交换源集合和参数以获得所需的比较，因为 `HashSet<E>` 没有为每个函数提供特定的比较器：

```py
    public int totalSongs() 
    { 
        return _songs.size(); 
    } 

```

最后，`totalSongs()` 方法使用集合的 `size()` 方法返回 `_songs` 集合中找到的元素数量，为我们提供的集合提供 *计数* 功能。

**Objective-C**

Objective-C 提供了 `NSSet` 和 `NSMutableSet` 类簇作为集合数据结构的具体实现。这些类簇提供了我们在集合数据结构中预期看到的大部分功能，并且缺少的显式函数非常简单易实现，这使得 Objective-C 的实现相当直接：

```py
    @interface EDSPlaylistSet() 
    { 
        NSMutableSet<EDSSong*>* _songs; 
        NSInteger _capacity; 
        BOOL _premiumUser; 
        BOOL _isEmpty; 
        BOOL _isFull; 
    } 
    @end 

    @implementation EDSPlaylistSet 

    -(instancetype)playlistSetWithPremiumUser:(BOOL)isPremiumUser andCapacity:(NSInteger)capacity 
    { 
        if (self == [super init]) 
        { 
            _songs = [NSMutableSet set]; 
            _premiumUser = isPremiumUser; 
            _capacity = capacity; 
        } 
        return self;  
    } 

    -(BOOL)isEmpty 
    { 
        return [_songs count] == 0; 
    } 

    -(BOOL)isFull 
    { 
        if (_premiumUser) 
        { 
            return NO; 
        } 
        else 
        { 
            return [_songs count] == _capacity; 
        } 
    } 

```

使用 `NSMutableSet`，我们为我们的类创建了一个名为 `_songs` 的私有 ivar。我们的初始化器实例化了这个字段，为我们提供了构建 `EDSPlaylistSet` 类的基础数据结构。我们还在头文件中创建了四个公共属性：`capacity`、`premiumUser`、`isEmpty` 和 `isFull`，这些属性由同名的私有 ivar 支持。`capacity` 属性存储非高级用户可以在他们的播放列表中存储的最大歌曲数量，而 `premiumUser` 表示此列表是否属于高级账户。`isEmpty` 和 `isFull` 属性允许我们的类轻松实现同名操作。`isEmpty` 属性简单地返回集合的计数是否为 `0`，而 `isFull` 属性首先检查此列表是否属于高级账户。如果是 `true`，则集合永远不会满，因为我们允许高级用户在他们的播放列表中存储无限数量的歌曲。如果此列表不属于高级账户，我们的方法确保 `_songs` 的当前计数没有超过容量，并返回该比较的结果：

```py
    -(BOOL)addSong:(EDSSong*)song 
    { 
        if (!_isFull && ![_songs containsObject:song]) 
        { 
            [_songs addObject:song]; 
            return YES; 
        } 
        return NO; 
    } 

```

`addSong:` 方法为我们类提供 *添加* 功能。此方法首先确认集合未满，然后确认对象实际上包含在 `_songs` 集合中。如果集合未通过这两个测试，则方法返回 `NO`，因为我们不能向列表添加更多歌曲或歌曲已存在于集合中。否则，方法调用 `addObject:` 并返回 `YES`：

```py
    -(BOOL)removeSong:(EDSSong*)song 
    { 
        if ([_songs containsObject:song]) 
        { 
            [_songs removeObject:song]; 
            return YES; 
        } 
        else 
        { 
            return NO; 
        } 
    } 

```

`removeSong:` 方法为我们类提供 *移除* 功能。此方法确认歌曲存在于集合中，然后使用 `removeObject:` 移除歌曲，并最终返回 `YES`。如果歌曲不在集合中，则方法返回 `NO`：

```py
    -(void)mergeWithPlaylist:(NSMutableSet<EDSSong*>*)playlist 
    { 
        [_songs unionSet:playlist]; 
    } 

```

`mergeWithPlaylist:` 方法为我们类提供 *并集* 功能。幸运的是，`NSSet` 通过 `unionSet:` 方法公开了并集功能，因此我们的方法只需简单地调用它。在这种情况下，`unionSet:` 将将 `playlist` 参数与我们的现有 `_songs` 列表合并：

```py
    -(NSMutableSet<EDSSong*>*)findSharedSongsInPlaylist: (NSMutableSet<EDSSong*>*)playlist 
    { 
        NSMutableSet *songsCopy = [NSMutableSet setWithSet:_songs]; 
        [songsCopy intersectSet:playlist]; 
        return songsCopy; 
    } 

```

接下来，`findSharedSongsInplaylist:` 方法为我们类提供 *交集* 功能。同样，`NSSet` 通过 `intersectSet:` 方法公开了交集功能。正如我们的 C# 示例一样，我们不是在原地修改 `_songs` 集合，而是在 `_songs` 复制和 `playlist` 参数之间返回交集：

```py
    -(NSMutableSet<EDSSong*>*)findUniqueSongs:(NSMutableSet<EDSSong*>*)playlist 
    { 
        NSMutableSet *songsCopy = [NSMutableSet setWithSet:_songs]; 
        [songsCopy minusSet:playlist]; 
        return songsCopy; 
    } 

```

`findUniqueSongs:` 方法为我们类提供 *差集* 功能。再次，`NSSet` 通过 `minusSet:` 方法公开了差集功能。同样，此方法不会修改我们现有的集合，而是返回 `_songs` 复制和 `playlist` 参数上的 `minusSet:` 或差集操作的结果：

```py
    -(BOOL)isSubset:(NSMutableSet<EDSSong*>*)playlist 
    { 
        return [_songs isSubsetOfSet:playlist];  
    } 

    -(BOOL)isSuperset:(NSMutableSet<EDSSong*>*)playlist 
    { 
        return; 
    } 

```

`isSubset:` 和 `isSuperset:` 方法通过其名称提供功能。这些方法以与我们的 Java 示例使用 `Set<E>` 接口的 `containsAll(HashSet<E>)` 方法类似的方式，利用 `NSSet` 上的 `isSubsetOfSet:` 方法：

```py
    -(NSInteger)totalSongs 
    { 
        return [_songs count]; 
    } 

```

最后，`totalSongs` 方法返回 `_songs` 集合中找到的元素数量，为我们集合提供 *计数* 功能。

**Swift**

Swift 提供了 `Set` 类作为集合数据结构的具体实现。此类提供了我们在集合数据结构中预期看到的所有功能，甚至比其 Objective-C 对应物还要多，这使得 Swift 实现非常简洁：

```py
    var _songs: Set<Song> = Set<Song>() 

    public private(set) var _capacity: Int 
    public private(set) var _premiumUser: Bool 
    public private(set) var _isEmpty: Bool 
    public private(set) var _isFull: Bool 

    public init (capacity: Int, premiumUser: Bool) 
    { 
        _capacity = capacity 
        _premiumUser = premiumUser 
        _isEmpty = true 
        _isFull = false 
    } 

    public func premiumUser() -> Bool 
    { 
        return _premiumUser 
    } 

    public func isEmpty() -> Bool 
    { 
        return _songs.count == 0 
    } 

    public func isFull() -> Bool 
    { 
        if (_premiumUser) 
        { 
            return false 
        } 
        else 
        { 
            return _songs.count == _capacity 
        } 
    } 

```

使用 `Set`，我们为我们的类创建了一个私有实例变量 `_songs`，并在其声明时直接初始化，这为我们构建 `PlaylistSet` 类提供了底层的数据结构。我们还创建了四个公共字段：`_capacity`、`_premiumUser`、`_isEmpty` 和 `_isFull`，以及后三个字段的公共访问器。`capacity` 字段存储非高级用户可以在他们的播放列表中存储的最大歌曲数量，而 `premiumUser` 表示此列表是否属于高级账户。`isEmpty` 和 `isFull` 字段允许我们的类轻松实现同名操作。`isEmpty()` 字段简单地返回集合的计数是否为 `0`。`isFull()` 字段首先检查此列表是否属于高级账户。如果是 `true`，则集合永远不会满，因为我们允许高级用户在他们的播放列表中存储无限数量的歌曲。如果此列表不属于高级账户，我们的获取器将确保 `_songs` 的当前计数没有超过 `capacity`，并返回这个比较结果：

```py
    public func addSong(song: Song) -> Bool 
    { 
        if (!_isFull && !_songs.contains(song)) 
        { 
            _songs.insert(song) 
            return true 
        } 
        return false 
    } 

```

`addSong(song: Song)` 方法为我们类提供了 **add** 功能。此方法首先确认集合不为满，然后确认对象实际上包含在 `_songs` 集合中。如果集合未通过这两个测试，则方法返回 `false`，因为我们不能向列表中添加更多歌曲或歌曲已存在于集合中。否则，方法调用 `insert()` 并返回 `true`：

```py
    public func removeSong(song: Song) -> Bool 
    { 
        if (_songs.contains(song)) 
        { 
            _songs.remove(song) 
            return true 
        } 
        else 
        { 
            return false 
        } 
    } 

```

`removeSong(song: Song)` 方法为我们类提供了 **remove** 功能。此方法确认歌曲存在于集合中，然后使用 `remove()` 删除歌曲，并最终返回 `true`。如果歌曲不存在于集合中，则方法返回 `false`：

```py
    public func mergeWithPlaylist(playlist: Set<Song>) 
    { 
        _songs.unionInPlace(playlist) 
    } 

```

`mergeWithPlaylist(playlist: Set<Song>)` 方法为我们类提供了 **union** 功能。幸运的是，`Set` 通过 `unionInPlace()` 方法暴露了并集功能，因此我们的方法只需调用它。在这种情况下，`unionInPlace()` 将将 `playlist` 参数与我们的现有 `_songs` 列表合并：

```py
    public func findSharedSongsInPlaylist(playlist: Set<Song>) -> Set<Song> 
    { 
        return _songs.intersect(playlist) 
    } 

```

接下来，`findSharedSongsInplaylist(playlist: Set<Song>)` 方法为我们类提供了 **intersection** 功能。`Set` 类通过 `intersect()` 方法暴露了交集功能。`intersect()` 方法不会修改 `_songs`，但只返回 `_songs` 和 `playlist` 参数之间的交集结果，因此我们只需返回这个方法调用的结果：

```py
    public func findUniqueSongs(playlist: Set<Song>) -> Set<Song> 
    { 
        return _songs.subtract(playlist) 
    } 

```

`findUniqueSongs(playlist: Set<Song>)` 方法为我们类提供了 **difference** 功能。再次强调，`Set` 通过 `subtract()` 方法暴露了差集功能。`subtract()` 方法不会修改 `_songs`，但只返回 `_songs` 和 `playlist` 参数之间的差集结果，因此我们只需返回这个方法调用的结果：

```py
    public func isSubset(playlist: Set<Song>) -> Bool 
    { 
        return _songs.isSubsetOf(playlist) 
    } 

    public func isSuperset(playlist: Set<Song>) -> Bool 
    { 
        return _songs.isSupersetOf(playlist) 
    } 

```

`isSubset(playlist: Set<Song>)` 和 `isSuperset(playlist: Set<Song>)` 方法通过其名称提供功能。这些方法分别利用 `isSubSetOf()` 和 `isSuperSetOf()` 方法，并返回一个表示这些比较结果的布尔值：

```py
    public func totalSongs() -> Int 
    { 
        return _songs.count; 
    } 

```

最后，`totalSongs()` 方法返回在 `_songs` 集合中找到的元素数量，为我们集合提供 *计数* 功能。

# 高级主题

现在我们已经考察了集合在常见应用中的使用方式，我们应该花些时间来考察它们在底层是如何实现的。大多数集合有三种类型：基于哈希表的集合、基于树的集合和基于数组的集合。

## 基于哈希表的集合

基于哈希表的集合通常用于无序数据集合。因此，对于非专业应用，你将遇到的绝大多数集合将是基于哈希表的。基于哈希表的集合与字典具有相似的操作成本。例如，搜索、插入和删除操作的操作成本都是 **O**(*n*)。

## 基于树的集合

基于树的集合通常基于二叉搜索树，但有时也可以基于其他结构。由于它们的设计，二叉搜索树允许在平均情况下非常高效的搜索功能，因为每个被检查的节点都可以允许从剩余的搜索模式中丢弃树的分支。尽管搜索二叉搜索树的最坏情况下的操作成本是 **O**(*n*)，但在实践中这很少需要。

## 基于数组的集合

数组可以用来实现集合的子集，这使得在正确组织的基于数组的集合中进行并集、交集和差集操作变得更加高效。

# 摘要

在本章中，你学习了集合数据结构的基本定义。为了充分欣赏该结构的功能，我们简要地考察了集合数据结构所基于的集合论的基本原则。随后，我们探讨了集合的常见操作以及它们与集合论函数的关系。然后，我们研究了如何在文本中研究的四种语言中实现集合。接下来，我们再次审视了登录到服务的用户问题，看看我们能否使用集合数据结构而不是数组或列表来改进其实施。在此之后，我们考察了一个案例研究，其中集合将是有益的。最后，我们研究了集合的不同实现方式，包括基于哈希表的集合、基于树的集合和基于数组的集合。
