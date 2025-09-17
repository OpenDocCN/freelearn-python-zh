# 第四章：栈：后进先出集合

**栈**是一种抽象数据结构，它作为一个基于**后进先出**（**LIFO**）原则插入和删除对象的集合。因此，最清楚地定义栈结构的是**push**操作，它向集合中添加对象，以及**pop**操作，它从集合中移除对象。其他常见操作包括 peek、clear、count、empty 和 full，所有这些将在本章后面的高级主题部分进行探讨。

栈可以是基于数组或基于链表的。同样，类似于链表，栈可以是排序的或未排序的。考虑到链表的结构，基于链表的栈在排序操作上比基于数组的栈更有效率。

栈数据结构非常适合任何需要仅从列表尾部添加和移除对象的应用程序。一个很好的例子是沿着指定的路径或一系列操作进行回溯。如果应用程序允许在集合的任何位置添加或移除数据，那么与我们已经考察过的数据结构相比，链表将是一个更好的选择。

在本章中，我们将介绍以下内容：

+   栈数据结构的定义

+   初始化栈

+   案例研究：运动规划算法

+   栈实现

+   常见栈操作

+   基于数组的栈

+   基于列表的栈

+   搜索

# 初始化栈

每种语言都为栈数据结构提供了不同级别的支持。以下是一些初始化集合、向集合中添加对象以及从集合中移除顶部对象的示例。

**C#**

C# 通过 `Stack<T>` 泛型类提供了栈数据结构的具体实现。

```py
    Stack<MyObject> aStack = new Stack<MyObject>(); 
    aStack.Push(anObject); 
    aStack.Pop(); 

```

**Java**

Java 通过 `Stack<T>` 泛型类提供了栈数据结构的具体实现。

```py
    Stack<MyObject> aStack = new Stack<MyObject>(); 
    aStack.push(anObject); 
    aStack.pop(); 

```

**Objective-C**

Objective-C 没有提供栈数据结构的具体实现，但可以通过类簇 `NSMutableArray` 轻易地创建。请注意，这将创建一个基于数组的栈实现，这通常比基于链表的实现效率低。

```py
    NSMutableArray<MyObject *> *aStack = [NSMutableArray array]; 
    [aStack addObject:anObject]; 
    [aStack removeLastObject]; 

```

## UINavigationController

说不提供栈数据结构并不完全准确。任何 Objective-C 的 iOS 编程都会立即让开发者通过使用`UINavigationController`类接触到栈数据结构的实现。

`UINavigationController` 类管理导航堆栈，这是一个基于视图控制器数组的堆栈。该类公开了几个对应于基本堆栈操作的方法。这些包括 `pushViewController:animated:` (*push*), `popViewControllerAnimated:` (*pop*), `popToRootViewControllerAnimated:` (*clear*...sort of), 和 `topViewController:` (*peek*)。导航堆栈永远不会是 *empty*，除非它是一个 *nil* 对象，并且只有当你的应用添加了如此多的视图控制器以至于设备耗尽系统资源时，它才能被认为是 *full*。

由于这是一个基于数组的实现，你可以通过简单地检查集合本身的 `count` 来获取堆栈的 *count*。然而，这不是你可以用于应用中任何目的的集合类。如果你需要一个适用于更一般情况的堆栈，你需要自己构建一个。

**Swift**

与 Objective-C 一样，Swift 没有提供堆栈数据结构的具体实现，但 Array 类确实公开了一些类似堆栈的操作。以下示例演示了 `popLast()` 方法，它移除并返回数组中的最后一个对象：

```py
    var aStack: Array [MyObject](); 
    aStack.append(anObject) 
    aStack.popLast() 

```

## 堆栈操作

并非所有堆栈数据结构的实现都公开相同的操作方法。然而，更常见的操作应该可用或根据开发者的需要提供。这些操作中的每一个，无论是基于数组实现还是基于链表实现，都有 **O**(*1*) 的操作成本。

+   **push**：push 操作通过向集合追加（如果它是基于数组的）或向集合添加新节点（如果它是基于链表的）来将新对象添加到堆栈中。

+   **pop**：pop 操作是 push 的反操作。在大多数实现中，pop 操作既移除也返回堆栈顶部的对象给调用者。

+   **peek**：peek 操作返回堆栈顶部的对象给调用者，但不从集合中移除该对象。

+   **clear**：clear 操作从堆栈中移除所有对象，有效地将集合重置为空状态。

+   **count**：count 操作，有时也称为大小或长度，返回集合中对象的总数。

+   **empty**：empty 操作通常返回一个布尔值，表示集合是否有任何对象。

+   **full**：full 操作通常返回一个布尔值，表示集合是否已满或是否还有空间添加更多对象。

# 案例研究：运动规划算法

**商业问题**：一位工业工程师编程一个机器人制造设备，以在部件的顺序接收器中插入螺栓，然后在每个螺栓上安装并拧紧螺母。机器人携带每个操作不同的工具，并且可以在命令下自动在它们之间切换。然而，在工具之间切换的过程增加了整体工作流程的相当多的时间，尤其是在工具在每一个螺栓上反复切换时。这已被确定为效率低下的一个来源，工程师希望提高该过程的速度，减少完成每个单元所需的总时间。

为了消除反复切换工具引入的延迟，工程师决定编程机器人先安装所有的螺栓，然后再切换工具，返回并安装所有的螺母。为了进一步提高性能，他不想让机器人重置到它原来的起始位置，而是希望它在安装螺母的同时重走自己的步骤。通过在安装螺母之前移除重置，他的工作流程消除了在部件上跨越的两个额外的遍历。为了实现他的目标，工程师需要存储在插入螺栓时移动机器人跨越部件的命令，然后以相反的顺序播放它们。

由于数据和应用的本质，表示命令的类将需要几个基本的功能。首先，它需要一个机制来添加和删除命令作为正常操作的一部分，以及在工作流程遇到错误时能够重置系统。在重置的情况下，该类必须能够报告当前等待执行命令的数量，以便计算库存损失。最后，该类应该能够轻松报告当命令列表达到容量或所有命令都已完成时。

**C#**

正如我们在先前的实现示例中所看到的，C#通过`Stack<T>`类方便地公开了一个堆栈数据结构。以下是一个简单的 C#实现的示例：

```py
    public Stack<Command> _commandStack { get; private set; } 
    int _capacity; 
    public CommandStack(int commandCapacity) 
    { 
        this._commandStack = new Stack<Command>(commandCapacity); 
        this._capacity = commandCapacity; 
    } 

```

我们声明了两个字段。第一个是`_commandStack`，它代表我们的堆栈数据结构，也是这个类的核心。该字段是公开可见的，但只能由我们类中的方法修改。第二个字段是`_capacity`。该字段维护我们的调用者定义的集合中命令的最大数量。最后，构造函数初始化`_commandStack`并将`commandCapacity`分配给`_capacity`。

```py
    public bool IsFull() 
    { 
        return this._commandStack.Count >= this._capacity; 
    }  

    public bool IsEmpty() 
    { 
        return this._commandStack.Count == 0; 
    } 

```

我们的首要任务是验证我们的集合。第一个验证方法`IsFull()`检查我们的栈是否已达到其容量。由于我们的业务规则规定，机器人必须在进入新部件之前回溯所有命令，因此我们将始终跟踪添加到我们的集合中的命令数量。如果由于任何原因我们发现我们已超过预定义的`_commandStack`容量，那么在之前的回溯操作中肯定出了问题，必须解决。因此，我们检查`_commandStack.Count`是否大于或等于`_capacity`并返回该值。`IsEmpty()`是下一个验证方法。在尝试通过*查看*集合读取我们的栈的任何操作之前，必须调用此方法。这两个操作的成本都是**O**(*1*)。

```py
    public bool PerformCommand(Command command) 
    { 
        if (!this.IsFull()) 
        { 
            this._commandStack.Push(command); 
            return true; 
        } 
        return false; 
    } 

```

`PerformCommand(Command)`方法提供了我们类的*推送*功能。它接受一个类型为`Command`的单个参数，然后检查`_commandStack`是否已满。如果已满，`PerformCommand()`方法返回`false`。否则，我们通过调用`Stack<T>.Push()`方法将`command`添加到我们的集合中。然后方法返回`true`给调用者。此操作的成本是**O**(*1*)。

```py
    public bool PerformCommands(List<Command> commands) 
    { 
        bool inserted = true; 
        foreach (Command c in commands) 
        { 
            inserted = this.PerformCommand(c); 
        } 
        return inserted;
    } 

```

如果调用者有一个可以连续执行的命令脚本，我们的类包括`PerformCommands(List<Command>)`类。

`PerformCommands()`方法接受一个命令列表，并通过调用`PerformCommand()`按顺序将它们插入到我们的集合中。此操作的成本是**O**(*n*)，其中*n*是`commands`中的元素数量。

```py
    public Command UndoCommand() 
    { 
        return this._commandStack.Pop(); 
    } 

```

`UndoCommand()`方法提供了我们类的*弹出*功能。它不接受任何参数，但通过调用`Stack<T>.Pop()`从我们的栈中弹出最后一个`Command`。`Pop()`方法从我们的`_commandStack`集合中移除最后一个`Command`并返回它。如果`_commandStack`为空，`Pop()`返回一个`null`对象。这种行为实际上对我们有利，至少在这个代码块的作用域内是这样。由于`UndoCommand()`方法被设计为返回一个`Command`实例，如果`_commandStack`为空，我们无论如何被迫返回`null`。因此，在调用`Pop()`之前首先检查`IsEmpty()`将是浪费时间。此操作的成本是**O**(*1*)。

```py
    public void Reset() 
    { 
        this._commandStack.Clear(); 
    } 

    public int TotalCommands() 
    { 
        return this._commandStack.Count; 
    } 

```

我们`CommandStack`类的最后两种方法，`Reset()`和`TotalCommands()`，分别提供了*清晰*的功能和*计数*的功能。

**Java**

如前所述的实现示例所示，Java 也通过 `Stack<E>` 类公开了一个栈数据结构，它是 `Vector<E>` 的扩展，包括五个方法，允许它作为一个类操作。然而，`Stack<E>` 的 Java 文档建议您使用 `Deque<E>` 而不是 `Stack<E>`。然而，由于我们将在 第五章 中评估 `Queue<E>` 和 `Deque<E>`，即 *队列：FIFO 集合*，因此我们将在此处使用 `Stack<E>` 类。以下是一个简单的 Java 实现示例：

```py
    private Stack<Command> _commandStack; 
    public Stack<Command> GetCommandStack() 
    { 
        return this._commandStack;
    } 

    int _capacity; 

    public CommandStack(int commandCapacity) 
    { 
        this._commandStack = new Stack<Command>(); 
        this._capacity = commandCapacity; 
    } 

```

我们类声明了三个字段。第一个是 `_commandStack`，它代表我们的栈数据结构，也是这个类的核心。该字段是私有的，但我们还声明了一个公开可见的获取器 `GetCommandStack()`。这是必要的，因为只有我们类中的方法应该能够修改这个集合。第二个字段是 `_capacity`。该字段维护我们的调用者定义的集合中的最大命令数。最后，构造函数初始化 `_commandStack` 并将 `commandCapacity` 赋值给 `_capacity`。

```py
    public boolean isFull() 
    { 
        return this._commandStack.size() >= this._capacity; 
    } 

    public boolean isEmpty() 
    { 
        return this._commandStack.empty(); 
    } 

```

再次，我们需要在开始时对我们的集合进行一些验证。第一个验证方法是 `isFull()`，它检查我们的栈是否已达到其容量。由于我们的业务规则规定，机器人必须在其命令全部回溯之后才能继续到新的部件，我们将跟踪添加到我们的集合中的命令数量。如果由于任何原因我们发现我们已超过 `_commandStack` 的预定义容量，那么在之前的回溯操作中肯定出了问题，必须解决。因此，我们检查 `_commandStack.size()` 是否大于或等于 `_capacity` 并返回该值。`isEmpty()` 是下一个验证方法。此方法必须在尝试通过 *peek* 集合读取我们的栈的任何操作之前调用。这两个操作的成本都是 **O**(*1*)。

```py
    public boolean performCommand(Command command) 
    { 
        if (!this.IsFull()) 
        { 
            this._commandStack.push(command); 
            return true; 
        } 
        return false; 
    } 

```

`performCommand(Command)` 方法提供了我们类中的 *push* 功能。它接受一个类型为 `Command` 的单个参数，然后检查 `_commandStack` 是否已满。如果已满，`performCommand()` 返回 `false`。否则，我们通过调用 `Stack<t>.push()` 方法将 `command` 添加到我们的集合中。然后该方法向调用者返回 `true`。此操作的成本为 **O**(*1*)。

```py
    public boolean performCommands(List<Command> commands) 
    { 
        boolean inserted = true; 
        for (Command c : commands) 
        { 
            inserted = this.performCommand(c); 
        } 
        return inserted; 
    } 

```

如果调用者有一个可以连续执行的命令脚本，那么我们的类还包括 `performCommands(List<Command>)` 方法。

`performCommands()` 方法接受一个命令列表，并通过调用 `performCommand()` 依次将它们插入到我们的集合中。此操作的成本为 **O**(*n*)，其中 *n* 是 `commands` 中元素的数量。

```py
    public Command undoCommand() 
    { 
        return this._commandStack.pop(); 
    } 

```

`undoCommand()` 方法提供了我们类中的 *弹出* 功能。它不接受任何参数，通过调用 `Stack<E>.pop()` 弹出我们栈中的最后一个 `Command`。`pop()` 方法从我们 `_commandStack` 集合中移除最后一个 `Command` 并返回它。如果 `_commandStack` 为空，`pop()` 返回一个 `null` 对象。与 C# 示例一样，这种行为在这个代码块的作用域内对我们有利。由于 `undoCommand()` 方法被设计为返回 `Command` 的一个实例，如果 `_commandStack` 为空，我们无论如何都会被迫返回 `null`。因此，在调用 `pop()` 之前先检查 `isEmpty()` 是一种浪费时间的操作。这个操作的成本是 **O**(*1*)。

```py
    public void reset() 
    { 
        this._commandStack.removeAllElements(); 
    } 

    public int totalCommands() 
    { 
        return this._commandStack.size(); 
    } 

```

我们 `CommandStack` 类的最后两个方法，`Reset()` 和 `TotalCommands()`，分别提供了 *清除* 和 *计数* 功能。

**Objective-C**

如我们之前所见（并且很可能在文本结束前还会再次见到），Objective-C 并没有暴露出显式的具体实现栈数据结构，而是提供了 `NSMutableArray` 类簇来达到这个目的。有些人可能会认为这是 Objective-C 的一个弱点，指出由于没有提供开发者可能需要的每一个可想象的操作的方法，这很不方便。另一方面，也有人可能会认为 Objective-C 在其简洁性方面要强大得多，为开发者提供了一个简化的 API 和构建所需任何数据结构的基本组件。我将把这个问题的结论留给你自己得出。同时，这里有一个 Objective-C 中简单实现的例子：

```py
    @interface EDSCommandStack() 
    { 
        NSMutableArray<EDSCommand*> *_commandStack; 
        NSInteger _capacity; 
    } 

    -(instancetype)initWithCommandCapacity:(NSInteger)commandCapacity 
    { 
        if (self = [super init]) 
        { 
            _commandStack = [NSMutableArray array]; 
            _capacity = capacity; 
        } 
        return self; 
    } 

```

我们类声明了两个 **ivar** 属性。第一个是 `_commandStack`，它代表我们的栈数据结构以及这个类的核心。这个属性是私有的，但我们还声明了一个公开可见的访问器 `commandStack`。这是必要的，因为只有我们类中的方法应该能够修改这个集合。第二个属性是 `_capacity`。这个属性维护了我们调用者定义的集合中命令的最大数量。最后，构造函数初始化 `_commandStack` 并将 `commandCapacity` 赋值给 `_capacity`。

```py
    -(BOOL)isFull 
    { 
        return [_commandStack count] >= _capacity; 
    } 

    -(BOOL)isEmpty 
    {  
        return [_commandStack count] == 0; 
    } 

```

同样，我们需要在开始时对我们的集合进行一些验证。第一个验证方法 `isFull:` 检查我们的栈是否达到了其容量。由于我们的业务规则指出，机器人必须在其所有命令回溯之后才能继续到新的部件，我们将跟踪被添加到我们集合中的命令数量。如果由于任何原因我们发现我们已超过了 `_commandStack` 的预定义容量，那么在之前的回溯操作中肯定出了问题，必须得到解决。因此，我们检查 `[_commandStack count]` 是否大于或等于 `_capacity` 并返回该值。`isEmpty:` 是下一个验证方法。这两个操作的成本都是 **O**(*1*)。

### 注意

由于 Objective-C 对传递 `nil` 对象相当宽容，你可能甚至不会考虑 `isEmpty:` 是一个验证方法，而更像是它自己的属性。然而，考虑一下，如果这个方法被声明为一个属性，我们除了在实现文件中包含这个方法之外，还需要将其声明为 `readonly`。否则，Objective-C 会为我们动态生成 ivar `_isEmpty`，调用者可以直接修改这个值。为了简单和清晰起见，在这种情况下，仅仅声明这个值为一个方法会更好。

```py
    -(BOOL)performCommand:(EDSCommand*)command 
    { 
        if (![self isFull]) 
        { 
            [_commandStack addObject:command]; 
            return YES; 
        } 
        return NO; 
    } 

```

`performCommand:` 方法提供了我们类中的 *推送* 功能。它接受一个类型为 `Command` 的单个参数，然后检查 `_commandStack` 是否已满。如果已满，`performCmmand:` 返回 `NO`。否则，我们通过调用 `addObject:` 方法将 `command` 添加到我们的集合中。然后该方法向调用者返回 `YES`。这个操作的成本是 **O**(*1*)。

```py
    -(BOOL)performCommands:(NSArray<EDSCommand*> *)commands 
    { 
        bool inserted = true; 
        for (EDSCommand *c in commands) { 
            inserted =  [self performCommand:c]; 
        } 
        return inserted; 
    } 

```

如果调用者有一个可以连续执行的命令脚本，我们的类包括 `performCommands:` 类。`performCommands:` 接受一个 `EDSCommand` 对象的数组，并通过调用 `performCommand:` 将它们按顺序插入我们的集合中。这个操作的成本是 **O**(n)，其中 *n* 是 `commands` 中元素的数量。

```py
    -(EDSCommand*)undoCommand 
    { 
        EDSCommand *c = [_commandStack lastObject]; 
        [_commandStack removeLastObject]; 
        return c; 
    } 

```

`undoCommand:` 方法提供了我们类中的 *弹出* 功能。由于 Objective-C 没有提供堆栈结构的具体实现，我们的类在这里需要有些创新。这个方法通过调用 `lastObject` 从堆栈中获取顶部对象，然后通过调用 `removeLastObject` 从集合中移除命令。最后，它将 `Command` 对象 `c` 返回给调用者。这一系列调用有效地模拟了在 C# 和 Java 的具体堆栈实现中找到的 *弹出* 功能。尽管这个方法需要跳过一些障碍来完成工作，但我们始终在处理数组中的最后一个对象，因此这个操作仍然具有 **O**(*1*) 的成本。

```py
    -(void)reset 
    { 
        [_commandStack removeAllObjects]; 
    } 

    -(NSInteger)totalCommands 
    { 
        return [_commandStack count]; 
    } 

```

再次强调，我们的 `CommandStack` 类的最后两个方法，`reset()` 和 `totalCommands()`，分别提供了 *清除* 功能和 *计数* 功能。遵循 Objective-C 规则！

**Swift**

与 Objective-C 一样，Swift 并不直接暴露堆栈数据结构的具体实现，但我们可以使用可变的、通用的 `Array` 类来达到这个目的。以下是一个 Swift 中简单实现的例子：

```py
    public fileprivate(set) var _commandStack: Array = [Command]()
    public fileprivate(set) var _capacity: Int;

    public init (commandCapacity: Int) 
    { 
        _capacity = commandCapacity; 
    } 

```

我们的类声明了两个属性。第一个是 `_commandStack`，它代表我们的堆栈数据结构，并且是这个类的核心。这个属性是公开可见的，但只能由我们类中的方法修改。第二个属性是 `_capacity`。这个字段维护了我们调用者定义的集合中命令的最大数量。最后，构造函数初始化 `_commandStack` 并将 `commandCapacity` 赋值给 `_capacity`。

```py
    public func IsFull() -> Bool 
    { 
        return _commandStack.count >= _capacity 
    } 

    public func IsEmpty() -> Bool 
    { 
        return _commandStack.count == 0; 
    } 

```

与其他语言的示例一样，我们包含了两个验证方法，分别称为 `IsFull()` 和 `IsEmpty()`。`IsFull()` 方法检查我们的栈是否达到了其容量。由于我们的业务规则规定，机器人必须在其命令全部回溯之后才能继续到新的部件，我们将跟踪添加到我们集合中的命令数量。如果由于任何原因我们发现我们已超过 `_commandStack` 的预定义容量，那么之前的回溯操作就出了问题，必须解决。因此，我们检查 `_commandStack.count` 是否大于或等于 `_capacity` 并返回该值。在尝试从我们的栈中读取操作之前，必须调用 `IsEmpty()`。这两个操作的成本都是 **O**(*1*)。

```py
    public func PerformCommand(_command: Command) -> Bool 
    { 
        if (!IsFull()) 
        { 
            _commandStack.append(command) 
            return true; 
        } 
        return false; 
    } 

```

`PerformCommand(Command)` 方法为我们类提供了 *push* 功能。它接受一个类型为 `Command` 的单个参数，然后检查 `_commandStack` 是否已满。如果已满，`PerformCmmand()` 方法返回 `false`。否则，我们通过调用 `Array.append()` 方法将 `command` 添加到我们的集合中。然后方法返回 `true` 给调用者。这个操作的成本是 **O**(*1*)。

```py
    public func PerformCommands(_commands: [Command]) -> Bool 
    { 
        var inserted: Bool = true; 
        for c in commands 
        { 
            inserted = PerformCommand(c); 
        } 
        return inserted; 
    } 

```

如果调用者有一个可以连续执行的命令脚本，我们的类包括 `PerformCommands(List<Command>)` 类。`PerformCommands()` 接受一个命令列表，并通过调用 `PerformCommand()` 方法将这些命令按顺序插入到我们的集合中。这个操作的成本是 **O**(*n*)，其中 *n* 是 `commands` 中元素的数量。

```py
    public func UndoCommand() -> Command 
    { 
        return _commandStack.popLast()! 
    } 

```

`UndoCommand()` 方法为我们类提供了 *pop* 功能。它不接受任何参数，但通过调用 `Array.popLast()!` 并使用强制解包操作符来访问 `return` 内部的 *wrapped* 值，从而弹出我们栈中的最后一个 `Command`（假设对象不是 `nil`）。`popLast()` 方法从我们的 `_commandStack` 集合中移除最顶部的 `Command` 并返回它。如果 `_commandStack` 为空，`popLast()` 返回 `nil`。正如在 Java 和 Objective-C 中所见，这种行为在我们的代码块范围内对我们有利。由于 `UndoCommand()` 方法被设计为返回 `Command` 的一个实例，如果 `_commandStack` 为空，我们无论如何都会被迫返回 `nil`。因此，在调用 `popLast()` 之前首先检查 `IsEmpty()` 是一种浪费时间的行为。这个操作的成本是 **O**(*1*)。

```py
    public func Reset() 
    { 
        _commandStack.removeAll() 
    } 

    public func TotalCommands() -> Int 
    { 
        return _commandStack.count; 
    } 

```

我们 `CommandStack` 类的最后一个方法对，`Reset()` 和 `TotalCommands()`，分别提供了 *clear* 和 *count* 功能。

### 注意

**空合并运算符**，或称为其他语言中的**空合并运算符**，是更冗长的三元运算符和显式的`if...else`语句的简写。例如，C#和 Swift 将`??`指定为这个运算符。Swift 更进一步，包括`!`，或解包运算符，用于返回值是可选的或可能为 nil 的情况。Swift 中的`??`运算符在解包**可选**类型时定义默认值是必要的。

# 高级主题 - 栈实现

现在我们已经看到了栈在常见实践中的应用，让我们来考察你可能会遇到的不同类型的栈实现。最常见的两种实现是基于数组的栈和基于链表的栈。我们将在下面考察每一种。

## 基于数组的栈

基于数组的栈使用可变数组来表示集合。在这个实现中，数组的 0 位置代表栈的*底部*。因此，`array[0]`是第一个推入栈中的对象，也是最后一个弹出栈的对象。基于数组的结构对于排序栈来说并不实用，因为任何对结构的重新组织都会比基于列表的栈需要显著更多的操作成本。汉诺塔问题是一个典型的基于数组的排序示例，其操作成本为**O**(*2^n*)，其中*n*是起始塔上的盘子数量。汉诺塔问题将在第十二章中更详细地考察，*排序：从混乱中带来秩序*。

## 基于链表的栈

基于链表的栈使用一个指向栈中*底部*对象的指针，以及随着每个新对象从列表中的最后一个对象链接而来，后续的指针。从栈顶弹出对象只是简单地从集合中移除最后一个对象。对于需要排序数据的应用，链表栈要高效得多。

# 摘要

在本章中，我们学习了栈数据结构的基本定义，包括如何在所讨论的四种语言中初始化结构的具体实现。接下来，我们讨论了与栈数据结构相关联的最常见操作及其操作成本。我们通过一个案例研究来考察使用栈跟踪传递给机器人制造设备的命令。这些例子展示了 C#和 Java 如何提供栈的具体实现，而 Objective-C 和 Swift 则没有。最后，我们考察了两种最常见的栈类型，基于数组和基于链表的，并展示了基于数组的栈不适合用于排序栈。
