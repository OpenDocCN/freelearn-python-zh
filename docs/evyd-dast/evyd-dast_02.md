# 第二章。数组：基础集合

很常见，我们的应用程序在运行时需要在内存中存储多个用户数据或对象。一个解决方案是在我们的各种类中定义多个字段（属性）来存储我们所需的数据点。不幸的是，即使在处理最简单的流程时，这种方法也会很快变得无效。我们可能需要处理太多的字段，或者我们根本无法在编译时预测我们项目的所有动态需求。

解决这个问题的方法之一是使用数组。数组是简单的数据集合，由于许多其他数据结构都是建立在它们之上，因此它们是你日常编程经验中遇到的最常见的数据结构之一。

数组是包含特定类型固定数量项的容器。在 C 语言及其后裔语言中，数组的大小在创建数组时确定，并且从那时起长度保持固定。数组中的每个项称为**元素**，每个元素都可以通过其索引号访问。一般来说，数组是可以通过运行时确定的索引选择的数据项集合。

在本章中，我们将涵盖以下主题：

+   定义

+   可变数组与不可变数组

+   数组的示例应用

+   线性搜索

+   基本数组

+   对象数组

+   混合数组

+   多维数组

+   锯齿形数组

### 注意

注意，大多数语言中的数组使用所谓的**零基索引**，这意味着数组中的第一个项目索引为 0，第二个为 1，依此类推。

**索引错误**发生在源代码试图访问一个比实际想要访问的项目索引远一点的给定索引时。这种错误对于新手和经验丰富的程序员都是常见的，并且往往会导致**索引超出范围**或**索引超出界限**的运行时错误。

![数组：基础集合](img/00002.jpeg)

### 小贴士

**编译时间和运行时间**

在编译型编程语言（与解释型语言相对）中，编译时间和运行时间之间的区别仅仅是应用程序编译和运行之间的区别。在编译过程中，开发者编写的高级源代码被输入到另一个程序（通常称为编译器，奇怪的是）。编译器检查源代码是否有正确的语法，确认类型约束得到执行，优化代码，然后生成目标架构可以利用的低级语言的可执行文件。如果一个程序成功编译，我们知道源代码是良好形成的，生成的可执行文件可以启动。请注意，开发者有时会使用术语*编译时间*来包括编写源代码的实际过程，尽管这在语义上是不正确的。

在运行时，编译后的代码在执行环境中运行，但它仍然可能遇到错误。例如，尝试除以零、取消引用空内存指针、内存不足或尝试访问不存在的资源都可能导致你的应用程序崩溃，如果你的源代码没有优雅地处理这些场景。

# 可变数组与不可变数组

通常，基于 C 语言的编程语言共享许多相同的根本特性。例如，在 C 语言中，一旦创建了一个普通数组，其大小就不能改变。由于我们在这里检查的四种语言都是基于 C 语言的，因此我们将要处理的数组也将具有固定长度。然而，尽管数组的大小不能改变，但在数组创建后，结构的内容可以改变。

那么，数组是可变的还是不可变的？在可变性的术语中，我们说*普通 C 数组是* *不可变的*，因为一旦创建结构本身就不能改变。因此，通常不建议将普通 C 数组用于除静态数据集之外的其他用途。这是因为，每当数据集发生变化时，你的程序需要将修改后的数据复制到一个新的数组对象中，并丢弃旧的一个，这两个操作都是昂贵的操作。

在高级语言中，你将要处理的数组对象大多数不是普通的 C 数组，而是为开发者的便利而创建的包装类。数组包装类封装了底层数据结构的复杂性，以便于处理幕后重负载的方法和暴露数据集特性的属性。

### 小贴士

无论何时，一种语言为特定类型或数据结构提供了一个包装类，你应该利用它。这些比编写自己的实现更方便，并且通常更可靠。

## 案例研究：登录到网络服务的用户

**业务问题**：一位开发者创建了一个应用程序，用于将移动用户登录到特定的网络服务。由于服务器硬件的限制，该网络服务在任何给定时间只能允许 30 个连接的用户。因此，开发者需要一种方法来跟踪和限制连接到服务的移动设备用户数量。为了避免允许重复用户登录并超载服务，简单的连接计数是不够的，因为开发者将无法区分每个连接的所有者。维护一个表示已登录用户的对象数组被选为解决方案的核心组件。

**C#**

```py
    using System; 
    //... 
    User[] _users; 
    public LoggedInUserArray () 
    { 
        User[] users = new User[0]; 
       _users = users; 
    } 

```

在前面的例子中，有几个重要的部分我们需要注意。首先，我们将`User`实例存储在一个名为`_users`的私有类字段中。接下来，构造函数正在实例化一个新的`User`对象数组。最后，我们将数组实例化为长度为 0 的集合，并将其分配给我们的私有后端字段。这是因为我们的数组还没有分配任何用户，我们不希望通过尝试跟踪空值来进一步复杂化此代码。在现实世界的例子中，你可能会选择在一行中实例化和分配私有后端字段：

```py
    _users = new User[0]; 

```

前一个例子更为冗长，因此更易于阅读。然而，使用更简洁的例子则占用的空间更少。两种方法都可以行得通。接下来，我们将探讨一种方法，允许我们将`User`对象添加到数组中：

```py
    bool CanAddUser(User user) 
    { 
        bool containsUser = false; 
        foreach (User u in _users) 
        { 
            if (user == u) 
            { 
                containsUser = true; 
                break; 
            } 
        } 

        if (containsUser) 
        { 
            return false; 
        } else { 
            if (_users.Length >= 30) 
            { 
                return false; 
            } else { 
                return true; 
            } 
        } 
    } 

```

在这里，我们引入了一个私有方法来进行某种形式的**验证**。此方法的目的在于确定在此时刻将用户添加到数组中是否是一个有效的操作。首先，我们声明了一个名为`containsUser`的`bool`变量。我们将使用此标志来指示数组是否已经包含正在传递的`User`对象。接下来，我们执行了一个`for`循环来检查数组中的每个对象与传递的`User`对象是否匹配。如果我们找到一个匹配项，我们将`containsUser`标志设置为`true`并退出`for`循环以节省处理器时间。如果`containsUser`为`true`，我们知道找到了用户对象，添加另一个副本将违反我们指定的业务规则。因此，该方法返回`false`。如果用户不存在于数组中，则执行继续。

然后，我们通过评估其`Length`属性来检查数组是否已经包含 30 个或更多项。如果是`true`，则返回`false`，因为根据我们的业务规则，数组已满，添加更多将构成违规。否则，返回`true`，程序执行可以继续：

```py
    public void UserAuthenticated(User user) 
    { 
        if (this.CanAddUser(user)) 
        { 
            Array.Resize(ref _users, _users.Length + 1); 
            _users[_users.Length - 1] = user; 
            Console.WriteLine("Length after adding user {0}: {1}", user.Id, _users.Length); 
        } 
    } 

```

此方法在用户经过身份验证后调用，这是我们想要将用户添加到用户名单中的唯一时间。在这个方法中，我们通过调用`CanAddUser()`方法来验证添加用户操作。如果`CanAddUser()`方法返回`true`，则方法执行继续。首先，我们使用`Array`包装类的`Resize()`方法将数组扩展一个元素，为新添加的对象腾出空间。接下来，我们将新的`User`对象赋值到调整大小后的数组中的最后一个位置。最后，我们通过将用户 ID 和新的`_users`数组长度记录到控制台来进行一些简单的维护：

```py
    public void UserLoggedOut(User user) 
    { 
        int index = Array.IndexOf(_users, user); 
        if (index > -1) 
        { 
            User[] newUsers = new User[_users.Length - 1]; 
            for (int i = 0, j = 0; i < newUsers.Length - 1; i++, j++) 
            { 
                if (i == index) 
                { 
                    j++; 
                } 
                newUsers[i] = _users[j]; 
            }  
            _users = newUsers; 
        } 
        else 
        { 
            Console.WriteLine("User {0} not found.", user.Id); 
        } 
        Console.WriteLine("Length after logging out user {0}: {1}", user.Id, _users.Length); 
    }  

```

当一个先前认证的用户从网络服务中注销时，会调用此方法。它使用数组包装类的`IndexOf()`方法来确定传递的`User`对象是否存在于数组中。由于`IndexOf()`在找不到匹配对象时返回`-1`，此方法确认`i`的值等于`-1`。如果`index`的值等于`-1`，我们执行一些维护工作，例如在控制台中显示此用户 ID 当前未登录的消息。否则，我们开始从数组中删除对象的进程。

首先，我们必须创建一个比旧数组少一个元素的临时数组。接下来，我们从 0 循环到新数组的长度，其中`i`标记新数组中的位置，`j`标记旧数组中的位置。如果`i`等于我们想要删除的项目位置，我们就增加`j`以跳过旧数组中的该元素。最后，我们将从旧数组中正确位置的用户分配到新数组中。一旦我们遍历完数组，我们就将新列表分配给`_users`属性。之后，我们通过在控制台记录已删除的用户 ID 和`_users`数组的新长度来执行一些简单的维护工作。

**Java**

```py
    User[] _users; 

    public LoggedInUserArray() 
    { 
        User[] users = new User[0]; 
        _users = users; 
    } 

```

在前面的示例中，有几个重要的部分我们需要注意。首先，我们将`User`实例存储在一个名为`_users`的私有类字段中。其次，构造函数正在实例化一个新的`User`对象数组。最后，我们将数组实例化为长度为 0 的集合，并将其分配给我们的私有后端字段。这是因为我们的数组还没有分配任何用户，我们不希望通过尝试跟踪空值来进一步复杂化此代码。在现实世界的示例中，你可能会选择在一行中实例化和分配私有后端字段：

```py
    _users = new User[0]; 

```

前面的示例更冗长，因此更易读。然而，使用更简洁的示例会占用更少的空间。与 C#一样，两种方法都可以工作：

```py
    boolean CanAddUser(User user) 
    { 
        boolean containsUser = false; 
        for (User u : _users) 
        { 
            if (user.equals(u)) 
            { 
                containsUser = true; 
                break; 
            } 
        } 

        if (containsUser) 
        { 
            return false; 
        } else { 
            if (_users.length >= 30) 
            { 
                return false; 
            } else { 
                return true; 
            } 
        } 
    } 

```

在这里，我们引入一个私有方法来进行某种验证。这个方法的目的在于确定在这个时候将用户添加到数组中是否是一个有效的操作。首先，我们声明了一个名为`containsUser`的`boolean`变量。我们将使用这个标志来表示数组是否已经包含正在传递的`User`对象。接下来，我们执行一个`for`循环来检查数组中的每个对象与传递的`User`对象是否匹配。如果我们找到一个匹配项，我们将`containsUser`标志设置为`true`并退出`for`循环以节省处理器时间。如果`containsUser`为`true`，我们知道找到了用户对象，添加另一个副本将违反我们指定的业务规则。因此，该方法返回`false`。如果用户不存在于数组中，执行将继续。

接下来，我们通过评估其 `Length` 属性来检查数组是否已经包含 30 个或更多项目。如果是 `true`，我们返回 `false`，因为根据我们的业务规则，数组已满，添加更多将是一种违规行为。否则，我们返回 `true`，程序执行可以继续：

```py
    public void UserAuthenticated(User user) 
    { 
        if (this.CanAddUser(user)) 
        { 
            _users = Arrays.copyOf(_users, _users.length + 1); 
            _users[_users.length - 1] = user; 
            System.out.println("Length after adding user " + user.GetId() + ": " + _users.length); 
        } 
    } 

```

此方法在用户认证后调用，这是我们想要将用户添加到用户名单的唯一时间。在此方法中，我们通过调用 `CanAddUser()` 方法验证了添加用户操作。如果 `CanAddUser()` 返回 `true`，则方法执行继续。首先，我们使用 `Arrays` 包装类的 `copyOf()` 方法创建一个新数组的新副本，为我们的新添加腾出空间。接下来，我们将新的 `User` 对象分配给调整大小后的数组中的最后一个位置。最后，我们通过将用户 ID 和 `_users` 数组的新长度记录到控制台来进行一些简单的维护工作：

```py
    public void UserLoggedOut(User user) 
    { 
        int index = -1; 
        int k = 0; 
        for (User u : _users) 
        { 
            if (user == u) 
            { 
                index = k; 
                break; 
            } 
            k++; 
        } 

        if (index == -1) 
        { 
            System.out.println("User " + user.GetId() + " not found."); 
        } 
        else 
        { 
            User[] newUsers = new User[_users.length - 1]; 
            for (int i = 0, j = 0; i < newUsers.length - 1; i++, j++) 
            { 
                if (i == index) 
                { 
                    j++; 
                } 
                newUsers[i] = _users[j]; 
            } 

            _users = newUsers; 
        } 

        System.out.println("Length after logging out user " + user.GetId() + ": " + _users.length); 
    } 

```

当之前认证的用户从网络服务中注销时，会调用此方法。首先，它会遍历 `_users` 数组以定位与传入的 `User` 对象匹配的对象。我们将索引值初始化为 `-1`，这样，如果找不到匹配的对象，此值不会改变。接下来，此方法确认 `index` 的值是否等于 `-1`。如果是 `true`，我们通过在控制台记录此用户 ID 当前未登录来进行一些维护工作。否则，我们开始从 `_users` 数组中删除对象的过程。

首先，我们必须创建一个比旧数组少一个元素的临时数组。然后，我们从 0 遍历到新数组的长度，`i` 标记新数组中的位置，`j` 标记旧数组中的位置。如果 `i` 等于要删除的项目位置，我们增加 `j` 以跳过旧数组中的该元素。最后，我们将从旧数组中正确位置的用户分配到新数组中。一旦我们完成循环，我们将新列表分配给 `_users` 属性。之后，我们通过将删除的用户 ID 和 `_users` 数组的新长度记录到控制台来进行一些简单的维护工作。

**Objective-C**

在 Objective-C 中与原始 C 数组一起工作与在 C# 或 Java 中相当不同，主要是因为 Objective-C 不提供直接与原始类型一起工作的方法。然而，Objective-C 提供了 `NSArray` 包装类，我们将在下面的代码示例中使用它：

```py
    @interface EDSLoggedInUserArray() 
    { 
        NSArray *_users; 
    } 

    -(instancetype)init 
    { 
        if (self = [super init]) 
        { 
            _users = [NSArray array]; 
        } 
        return self; 
    } 

```

首先，我们的 Objective-C 类接口为我们的数组定义了一个 **ivar** 属性。接下来，我们的初始化器使用 `[NSArray array]` 便利初始化器实例化 `_users` 对象：

```py
    -(BOOL)canAddUser:(EDSUser *)user 
    { 
        BOOL containsUser = [_users containsObject:user]; 

        if (containsUser) 
        { 
            return false; 
        } 
        else 
        { 
            if ([_users count] >= 30) 
            { 
                return false; 
            } 
            else 
            { 
                return true; 
            } 
        } 
    } 

```

`canAddUser:`方法也作为我们 Objective-C 示例中的内部验证。此方法的目的是在当前时间将用户添加到数组中是否是一个有效的操作。由于我们正在使用`NSArray`，我们可以访问`containsUser:`方法，该方法可以立即确定传入的`User`对象是否存在于`_users`数组中。然而，不要被这段代码的简单性所迷惑，因为在`NSArray`的底层，`containsUser:`方法看起来像这样：

```py
    BOOL containsUser = NO; 
    for (EDSUser *u in _users) { 
        if (user.userId == u.userId) 
        { 
            containsUser = YES; 
            break; 
        } 
    } 

```

如果这段代码看起来很熟悉，那是因为它在功能上几乎与我们的之前的 C#和 Java 示例相同。`containsObject:`方法是为了我们的方便而存在的，并且在我们背后执行繁重的工作。再次强调，如果找到用户对象，添加另一个副本将违反我们指定的业务规则，并且方法返回`false`。如果用户不存在，则执行继续。

接下来，我们通过评估其`count`属性来检查数组是否已经包含 30 个或更多项。如果是，则返回`false`，因为根据我们的业务规则，数组已满，添加更多将违反规则。否则，返回`true`，程序执行可以继续：

```py
    -(void)userAuthenticated:(EDSUser *)user 
    { 
        if ([self canAddUser:user]) 
        { 
            _users = [_users arrayByAddingObject:user]; 
            NSLog(@"Length after adding user %lu: %lu", user.userId, [_users count]); 
        } 
    } 

```

这种方法是在用户认证成功后调用的，这是我们唯一想要将用户添加到用户角色列表中的时候。在这个方法中，我们通过调用`canAddUser:`来验证添加用户操作。如果`canAddUser:`返回`true`，则方法执行继续。我们使用`NSArray`类的`arrayByAddingObject:`方法创建一个包含我们新的`User`对象的新数组副本。最后，我们通过将用户 id 和`_users`数组的新长度记录到控制台来进行一些简单的维护操作：

```py
-(void)userLoggedOut:(EDSUser *)user 
{ 
    NSUInteger index = [_users indexOfObject:user]; 
    if (index == NSNotFound) 
    { 
        NSLog(@"User %lu not found.", user.userId); 
    } 
    else 
    { 
        NSArray *newUsers = [NSArray array]; 
        for (EDSUser *u in _users) 
        { 
            if (user != u) 
            { 
                newUsers = [newUsers arrayByAddingObject:u]; 
            } 
        } 

        _users = newUsers; 
    } 

    NSLog(@"Length after logging out user %lu: %lu", user.userId, [_users count]); 
} 

```

当之前认证过的用户从网络服务中注销时，将调用此方法。首先，它使用`NSArray indexOfObject:`数组来获取与已传入的`User`对象匹配的任何对象的索引。如果找不到对象，则方法返回`NSNotFound`，这相当于`NSIntegerMax`。

此方法接下来确认`index`的值是否等于`NSNotFound`。如果是，我们通过将此用户 id 当前未登录的控制台记录到控制台来进行一些维护操作。否则，我们开始从`_users`数组中删除对象的过程。

不幸的是，`NSArray` 不提供从底层不可变数组中删除对象的方法，因此我们需要有点创意。首先，我们创建一个名为 `newUsers` 的临时数组对象来保存我们想要保留的所有 `User` 对象。然后，我们遍历 `_users` 数组，检查每个对象是否与我们要删除的 `User` 匹配。如果没有匹配项，我们以与将新用户添加到 `_users` 时相同的方式将其添加到 `newUsers` 数组中。如果 `User` 对象匹配，我们简单地跳过它，从而从最终对象数组中删除它。正如你所想象的那样，这个程序非常耗时，如果可能的话，应尽量避免这种模式。一旦循环完成，我们将新数组赋值给 `_users` 属性。最后，我们通过将删除的用户 ID 和 `_users` 数组的新计数记录到控制台来进行一些简单的维护工作。

**Swift**

在 Swift 中与原始 C 数组一起工作与在 C# 或 Java 中做得很相似，因为它提供了 `Array` 类，我们将在下面的代码示例中使用它：

```py
var _users: Array = [EDSUser]() 

```

我们只需要一个类属性来支持我们的用户数组。Swift 数组与 C# 和 Java 一样具有类型依赖性，我们必须在声明数组属性时声明类型。注意 Swift 初始化数组的方式，它是通过在类型名称或对象类名称周围使用订阅操作符，而不是将其附加到名称上：

```py
    func canAddUser(user: EDSUser) -> Bool 
    { 
        if (_users.contains(user)) 
        { 
            return false; 
        } 
        else 
        { 
            if (_users.count >= 30) 
            { 
                return false; 
            } 
            else 
            { 
                return true; 
            } 
        } 
    } 

```

`canAddUser:` 方法也用作内部验证。此方法的目的在于确定在此时刻将用户添加到数组中是否是一个有效的操作。首先，我们使用 `Array.contains()` 方法来确定我们想要添加的用户是否已经存在于数组中。如果找到用户对象，添加另一个副本将违反我们指定的业务规则，并且方法返回 `false`。如果用户不存在，则继续执行。

接下来，我们使用 `_users` 数组的 `count` 属性来检查数组内的对象总数是否不大于或等于 30。如果为 `true`，则返回 `false`，因为根据我们的业务规则，数组已满，添加更多将违反规则。否则，返回 `true`，程序执行可以继续：

```py
    public func userAuthenticated(user: EDSUser) 
    { 
        if (self.canAddUser(user)) 
        { 
            _users.append(user) 
        }  
        print("Length after adding user \(user._userId): \ (_users.count)"); 
    } 

```

再次强调，此方法是在用户经过认证后调用的，这是我们想要将用户添加到用户名单的唯一时间。在这个方法中，我们通过调用 `canAddUser()` 方法来验证添加用户操作。如果 `canAddUser()` 返回 `true`，则方法执行继续，我们使用 `Array.append()` 方法将用户添加到数组中。最后，我们通过将用户 ID 和 `_users` 数组的新长度记录到控制台来进行一些简单的维护工作：

```py
    public func userLoggedOut(user: EDSUser) 
    { 
        if let index = _users.indexOf(user) 
        { 
            _users.removeAtIndex(index) 
        }  
        print("Length after logging out user \(user._userId): \(_users.count)") 
    } 

```

最后，为了在注销时删除用户，我们首先需要确定该对象是否存在于数组中，并获取其在数组中的索引。Swift 允许我们同时声明`index`变量，执行此检查，并将值赋给`index`。如果此检查返回`true`，我们调用`Array.removeAtIndex()`从数组中移除`user`对象。最后，我们通过记录被删除的用户 ID 和`_users`数组的新计数到控制台来进行一些简单的维护工作。

### 小贴士

**关注点分离**

当你检查前面的例子时，你可能会想知道当我们完成对它们的使用后，所有那些`User`对象会发生什么。如果是这样，那是个很好的发现！如果你仔细观察，你会看到在这个例子中我们没有实例化或修改任何一个`User`对象——只有包含对象的数组被修改了。这是有意为之的。

在面向对象编程中，**关注点分离**的概念规定，计算机程序应该被分解成尽可能少重叠的操作特性。例如，一个名为`LoggedInUserArray`的类，作为底层数组结构的包装器，应该只操作其数组的操作，对数组中的对象影响很小。在这种情况下，传入的`User`类对象的内部工作和细节不是`LoggedInUserArray`类的关注点。

一旦每个`User`从数组中移除，该对象就会继续其愉快的旅程。如果应用程序没有保留对`User`对象的任何其他引用，那么某种形式的**垃圾回收**最终会将其从内存中清除。无论如何，`LoggedInUserArray`类不负责垃圾回收，并且对这些细节保持中立。

# 高级主题

现在我们已经看到了数组在常见实践中的应用，让我们来探讨一些与数组相关的高级主题：搜索模式和数组中可以存储的基本对象类型的变体。

## 线性搜索

在学习数据结构时，不可避免地要讨论**搜索**和**排序**这两个主题。如果没有在数据结构中进行搜索的能力，数据对我们来说将几乎毫无用处。如果没有对数据集进行排序以便在特定应用中使用的能力，数据的管理将变得极其繁琐。

我们执行特定数据结构的搜索或排序所遵循的步骤或过程称为**算法**。算法在计算机科学中的性能或复杂度是通过使用**大 O 表示法**来衡量的，它来源于函数 *f(n) =* **O** *(g(n))*，读作*f of n equals big oh of g of n*。用最简单的术语来说，**大 O** 是我们用来描述算法运行最长时间的最坏情况的术语。例如，如果我们知道在数组中搜索的对象的索引，那么只需要一次比较就可以定位和检索该对象。因此，最坏情况需要一次比较，搜索的成本是 **O**(*1*)。

虽然我们将在稍后更详细地研究搜索和排序，但到目前为止，我们将研究**线性搜索**，或顺序搜索，这是搜索集合中最简单且效率最低的模式。迭代意味着重复执行一个过程。在线性搜索中，我们按顺序遍历对象集合，直到找到与我们的搜索模式匹配的项。对于包含 *n* 个项目的集合，最佳搜索情况是目标值等于集合中的第一个项目，这意味着只需要一次比较。在最坏的情况下，目标值根本不在集合中，这意味着需要 *n* 次比较。这意味着线性搜索的成本是 **O**(*n*)。如果你回顾代码示例，你会在几个地方看到 **O**(*n*) 搜索：

**C#**

这里是我们 C# 代码中的线性搜索算法，但已重新格式化以使用 `for` 循环，这更好地说明了 **O**(*n*) 成本的概念：

```py
    for (int i = 0; i < _users.Count; i++) 
    { 
        if (_users[i] == u) 
        { 
            containsUser = true; 
            break; 
        }  
    } 

```

**Java**

这里是我们 Java 代码中的线性搜索算法，但已重新格式化以使用 `for` 循环，这更好地说明了 **O**(*n*) 成本的概念：

```py
    for (int i = 0; I < _users.size(); i++) 
    { 
        if (_users[i].equals(u)) 
        { 
            containsUser = true; 
            break; 
        } 
    } 

```

**Objective-C**

这里是我们 Objective-C 代码中的线性搜索算法，但已重新格式化以使用 `for` 循环，这更好地说明了 **O**(*n*) 成本的概念：

```py
    for (int i = 1; i < [_users count]; i++)  
    { 
        if (((User*)[_users objectAtIndex:i]).userId == u.userId) 
        { 
            containsUser = YES; 
            break; 
        } 
    } 

```

**Swift**

我们的 Swift 代码中没有包含线性搜索的示例，但一个示例可能看起来像这样：

```py
    for i in 1..<_users.count 
    { 
        //Perform comparison 
    } 

```

## 原始数组

原始数组是仅包含原始类型的数组。在 C#、Java 和 Swift 中，你通过在原始类型上声明一个数组来声明一个原始数组。作为弱类型语言，Objective-C 不支持显式类型化的数组，因此也不支持显式原始数组。

**C#**

```py
    int[] array = new int[10]; 

```

**Java**

```py
    int[] array = new int[10]; 

```

**Objective-C**

```py
    NSArray *array = [NSArray array]; 

```

**Swift**

```py
    var array: Array = [UInt]() 

```

## 对象数组

对象数组是仅包含特定对象实例的数组。在 C#、Java 和 Swift 中，你通过在类上声明一个数组来声明一个对象数组。作为弱类型语言，Objective-C 不支持显式类型化的数组，因此也不支持显式对象数组。

**C#**

```py
    Vehicle[] cars = new Vehicle[10]; 

```

**Java**

```py
    Vehicle[] cars = new Vehicle[10]; 

```

**Objective-C**

```py
    NSArray *array = [NSArray array]; 

```

**Swift**

```py
    var vehicle: Array = [Vehicle]() 

```

## 混合数组

当与数组一起工作时，您使用一种数据类型声明数组，并且数组中的所有元素都必须匹配该数据类型。通常，这种约束是合适的，因为元素通常彼此紧密相关或共享相似的性质值。在其他时候，数组中的元素并不紧密相关或没有相似的性质值。在这些情况下，您可能希望能够在同一数组中混合匹配类型。C#和 Java 都提供了类似的机制来实现这一点——将数组声明为根类对象类型。由于 Objective-C 语言是弱类型，其数组默认就是混合的。Swift 提供了`AnyObject`类型来声明混合数组。

**C#**

```py
    Object[] data = new Object[10]; 

```

**Java**

```py
    Object[] data = new Object[10]; 

```

**Objective-C**

```py
    NSArray *data = [NSArray array]; 

```

**Swift**

```py
    var data: Array = [AnyObject]() 

```

与混合数组一起工作可能看起来很方便，但请注意，作为开发人员，您将类型检查的责任从编译器移走。对于像 Objective-C 这样的弱类型语言的开发人员来说，这不会是一个重大的调整，但经验丰富的强类型语言开发人员需要对此问题非常关注。

## 多维数组

多维数组是一个包含一个或多个额外数组的数组。我们正在使用的四种语言都可以支持*1...n*维度的多维数组。然而，请注意，超过三个级别的多维数组管理起来会变得极其困难。

有时，将多维数组概念化为与它们的维度相关的内容会有所帮助。例如，一个二维数组可能有行和列，或者*x*和*y*值。同样，一个三维数组可能有*x*、*y*和*z*值。让我们看看每种语言中二维和三维数组的示例。

**C#**

C#中的多维数组使用`[,]`语法创建，其中每个逗号代表数组中的一个额外维度。相应的`new`初始化器必须提供正确数量的尺寸参数以匹配定义，否则代码将无法编译：

```py
    //Initialize 
    int[,] twoDArray = new int[5, 5]; 
    int[, ,] threeDArray = new int[5, 6, 7]; 

    //Set values 
    twoDArray[2,5] = 90; 
    threeDArray[0, 0, 4] = 18; 

    //Get values 
    int x2y5 = twoDArray[2,5]; 
    int x0y0z4 = threeDArray[0,0,4]; 

```

**Java**

在 Java 中创建多维数组的语法简单涉及将`[]`配对连接起来，其中每一对代表数组中的一个维度。相应的`new`初始化器必须提供正确数量的括号大小参数以匹配定义，否则代码将无法编译：

```py
    //Initialize 
    int[][] twoDArray = new int[5][5]; 
    int[][][] threeDArray = new int[5][6][7]; 

    //Set values 
    twoDArray[2][5] = 90; 
    threeDArray[0][0][4] = 18; 

    //Get values 
    int x2y5 = twoDArray[2][5]; 
    int x0y0z4 = threeDArray[0][0][4]; 

```

**Objective-C**

Objective-C 不直接支持`NSArray`类中的多维数组。如果您的代码需要多维数组，您将需要使用`NSMutableArray`或一个普通的 C 数组，这两者都不在本章的范围内。

**Swift**

Swift 中的多维数组一开始看起来可能有些令人困惑，但您需要意识到您正在创建数组的数组。定义语法是`[[Int]]`，初始化语法是`[[1, 2], [3, 4]]`，其中初始化时使用的值可以是指定类型的任何值：

```py
    //Initialize 
    var twoDArray: [[Int]] = [[1, 2], [3, 4]] 
    var threeDArray: [[[Int]]] = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]] 

    //Set values 
    twoDArray[0][1] = 90; 
    threeDArray[0][0][2] = 18; 

    //Get values 
    var x0y1: Int = twoDArray[0][1]; 
    var x0y0z2: Int = threeDArray[0][0][2]; 

```

## 锯齿状数组

当多维数组包含不同大小的数组时，会创建交错数组。在极少数情况下，这种设计是必要的，但请注意，交错数组可能非常复杂且难以管理。C#、Java 和 Swift 支持交错数组。Objective-C 不支持使用`NSArray`的多维数组，因此也不支持使用它来创建交错数组。与多维数组类似，Objective-C 可以使用`NSMutableArray`或纯 C 数组来支持交错数组。

# 摘要

在本章中，你学习了数组结构的基本定义，数组在内存中的样子，以及我们讨论的四种语言如何实现纯 C 数组结构。接下来，我们讨论了可变数组和不可变数组之间的区别。通过示例，我们探讨了四种语言如何实现数组和数组功能。在本章的剩余部分，我们研究了线性搜索算法，并介绍了大**O**符号，包括如何将此符号应用于数组，并举例说明简单的迭代。我们讨论了原始数组、对象数组和混合数组之间的区别。最后，我们研究了多维数组及其对应项，交错数组。

作为最后的注意事项，了解何时使用数组是很重要的。数组非常适合存储少量恒定数据或变化极小甚至不变化的数据。如果你发现自己经常在操作数组中的数据，或者经常添加和删除对象，那么你可能需要考虑使用其他数据结构，例如列表，我们将在下一章中讨论。
