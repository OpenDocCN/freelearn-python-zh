# 第六章。字典：键值集合

**字典**是一种抽象数据结构，可以被描述为一个键的集合及其相关值的集合，其中每个键在集合中只出现一次。这种键和值之间的关联关系是为什么字典有时被称为**关联数组**。字典也被称为**映射**，或者更具体地说，对于基于**哈希表**的字典称为**哈希映射**，对于基于**搜索树**的字典称为**树映射**。与字典相关联的四个最常见函数是**添加**、**更新**、**获取**和**删除**。其他常见操作包括**包含**、**计数**、**重新分配**和**设置**。这些操作将在本章后面详细探讨。

字典的映射或关联性质使得插入、搜索和更新操作非常高效。通过在创建、编辑或获取值时指定键，大多数在设计良好的字典中的操作都具有最小的**O(1**)成本。也许正因为这种效率，字典是你日常开发经验中最常见的几种数据结构之一。

你可能会想知道，为什么描述为键值对的集合应该被称为字典。这个名字是类比于物理字典，其中每个单词（键）都有一个相关的定义（值）。如果这仍然有点抽象，可以考虑一个代客泊车服务。当你把车停在活动地点时，你从车里出来，有人在你离开前给你一张小票，然后开车离开。这张小票代表你的车，只代表你的车。没有其他带有相同标识符的小票与你现在持有的相同。因此，唯一能够取回你的车的方式就是向代客泊车服务出示这张特定的小票。一旦你这样做，有人就会带着你的车过来，你给他们小费，然后你开车离开。

这个过程是字典数据结构的一个具体例子。每张小票代表一个**键**，而每辆车代表某种类型的**值**。每个键都是唯一的，并且唯一地标识一个特定的值。当你的代码调用一个值时，代客泊车服务就是使用键来定位和返回你正在寻找的值的**集合**。给你的开发机器小费是可选的。

在本章中，我们将涵盖以下主题：

+   字典数据结构的定义

+   初始化字典

+   哈希表

+   常见的字典操作

+   案例研究 - 游艺厅票券总额

+   基于哈希表的字典

+   基于搜索树的字典

# 初始化字典

字典如此普遍，难怪我们正在检查的每种语言都通过具体的实现来支持它们。以下是一些初始化字典、向集合中添加几个键值对，然后从集合中删除这些对之一的示例。

**C#**

C# 通过 `Dictionary<TKey, TValue>` 类提供了字典数据结构的具体实现。由于这个类是泛型的，调用者可以定义用于键和值的类型。以下是一个示例：

```py
    Dictionary<string, int> dict = new Dictionary<string, int>(); 

```

这个示例初始化了一个新的字典，其中键将是 `string` 类型，值将是 `int` 类型：

```py
    dict.Add("green", 1); 
    dict.Add("yellow", 2);  
    dict.Add("red", 3); 
    dict.Add("blue", 4); 
    dict.Remove("blue"); 
    Console.WriteLine("{0}", dict["red"]); 

    // Output: 3 

```

**Java**

Java 提供了一个 `Dictionary<K, V>` 类，但最近已经弃用，转而使用实现了 `Map<K, V>` 接口的任何类。在这里，我们将查看 `HashMap<K, V>` 类的一个示例。这个类扩展了 `AbstractMap<K, V>` 并实现了 `Map<K, V>` 接口：

```py
    HashMap<String, String> dict = new HashMap<String, String>(); 
    dict.put("green", "1"); 
    dict.put("yellow", "2");  
    dict.put("red", "3"); 
    dict.put("blue", "4"); 
    dict.remove("blue"); 
    System.out.println(dict.get("red")); 

    // Output: 3 

```

这个类被称为 `HashMap`，因为它是一个具体的、基于哈希表的映射实现。值得注意的是，Java 不允许在 `HashMap` 类中使用原始数据类型作为键或值，因此在我们前面的例子中，我们用 String 类型替换了我们的值。

### 提示

**哈希表**

由于 Java 的一个字典实现被称为 **哈希表**，这似乎是介绍 **哈希表** 的好时机，有时也称为哈希表。哈希表使用 **哈希函数** 将数据映射到数组中的索引位置。技术上讲，哈希函数是任何可以将随机大小的数据绘制到静态大小的数据的函数。

在设计良好的哈希表中，搜索、插入和删除函数的成本为 **O**(1)，因为复杂性不依赖于集合中元素的数量。在许多情况下，与数组、列表或其他查找数据结构相比，哈希表要高效得多。这就是它们经常被用来构建字典的原因。这也是它们通常用于数据库索引、缓存以及作为 **集合** 数据结构基础的原因。我们将在第七章 Chapter 7 中更详细地讨论集合。*集合：无重复*。

事实上，哈希表是一种数据结构，尽管它们最常用于创建关联数组。那么，我们为什么不对哈希表数据结构进行更深入的探讨呢？在大多数语言中，对于类似的应用，字典比哈希表更受欢迎。这是因为字典是 **泛型类型化的**，而哈希表依赖于语言的根对象类型来内部分配值，例如 C# 的 `object` 类型。虽然哈希表允许几乎任何对象用作键或值，但泛型类型的字典将限制调用者只能将声明的类型对象作为元素的键或值。这种方法既类型安全又更高效，因为值不需要在每次更新或检索值时进行 **装箱** 和 **拆箱**（类型转换）。

话虽如此，不要犯下这样一个错误：认为字典仅仅是另一种名称的哈希表。确实，哈希表大致对应于`Dictionary<object, object>`的一些变体，但它是一个不同的类，具有不同的功能和方法。

**Objective-C**

Objective-C 提供了不可变和可变的字典类，分别是`NSDictionary`和`NSMutableDictionary`。由于我们将在后面的示例中使用可变字典，所以我们在这里只考察`NSDictionary`。`NSDictionary`可以使用`@{K : V, K : V}`语法初始化一个字面量数组，其中包含*1.*..*n*个键/值对。还有两种常见的初始化方法。第一种是`dictionaryWithObjectsAndKeys:`，它接受一个以`nil`结尾的对象/键对数组。第二种是`dictionaryWithObjects:forKeys:`，它接受一个对象数组和第二个键数组。与 Java 的`HashMap`类似，Objective-C 的`NSDictionary`和`NSMutableDictionary`类簇不允许使用标量数据作为键或值：

```py
    NSDictionary *dict = [NSDictionary dictionaryWithObjectsAndKeys: 
    [NSNumber numberWithInt:1], @"green",   
    [NSNumber numberWithInt:2], @"yellow",  
    [NSNumber numberwithInt:3], @"red", nil]; 

    NSArray *colors = @[@"green", @"yellow", @"red"]; 
    NSArray *positions = @[[NSNumber numberWithInt:1],  
                           [NSNumber numberWithInt:2],  
                           [NSNumber numberWithInt:3]]; 

    dict = [NSDictionary dictionaryWithObjects:positions forKeys:colors]; 
    NSLog(@"%li", (long)[(NSNumber*)[_points valueForKey:@"red"] integerValue]); 

    // Output: 3 

```

你可能会注意到`dictionaryWithObjects:forKeys:`方法更冗长，这使得它稍微易于阅读。然而，你必须格外小心，确保你的键和值正确地映射到对方。

**Swift**

Swift 中的字典是通过`Dictionary`类创建的。当使用`var`初始化为变量时，Swift 字典是可变的，但也可以通过使用`let`初始化为常量来创建不可变字典。字典中使用的键可以是整数或字符串。`Dictionary`类还可以接受任何类型的值，包括在其他语言中通常被认为是原生的类型，因为这些在 Swift 中实际上是命名类型，并且使用结构体在 Swift 标准库中定义。在任一情况下，都必须在初始化集合时声明你的键和值类型，并且之后不能更改。由于我们将在后面使用一个变量或可变字典，所以我们在这里初始化一个常量不可变集合：

```py
    let dict:[String: Int] = ["green":1, "yellow":2, "red":3]  
    print(dict[red]) 

    // Output: 3 

```

### 注意

我们将在第八章中更详细地考察**结构体**，*结构体：复杂类型*。

## 字典操作

并非所有字典数据结构的具体实现都公开了相同的操作方法。然而，更常见的操作应该是可用的，或者根据开发者的需要提供。以下是一些操作：

+   **add**：add 操作，有时称为插入，向集合中引入一个新的键/值对。add 操作具有**O(1**)的成本。

+   **get**：get 操作，有时称为**查找**，返回与给定键关联的值。如果找不到给定键的值，某些字典将引发一个*异常*。通过指定键，get 操作具有**O(1**)的成本。

+   **更新**：更新操作允许调用者修改集合中已经存在的值。并非所有字典实现都提供定义明确的更新方法，而是通过*引用*支持更新值。这意味着一旦使用获取操作从字典中取出对象，就可以直接修改它。通过指定键，更新操作的成本为**O**(*1*)。

+   **移除**：移除或*删除*操作将根据有效的键从集合中删除键/值对。大多数字典将优雅地忽略指定的不存在键。通过指定键，移除操作的成本为**O**(1)。

+   **包含**：包含操作返回一个布尔值，标识给定的键是否可以在集合中找到。包含操作必须遍历字典中的键集合以搜索匹配项。因此，此操作的最坏情况成本为**O**(*n*)。

+   **计数**：计数，有时被称为*大小*，可以是集合的一个方法，也可以是集合的一个属性，它返回字典中键/值元素的数量。计数通常是集合上的一个简单属性，因此，其成本为**O**(*1*)。

+   **重新分配**：重新分配操作允许将新值分配给现有键。在许多实现中，此操作不如更新操作常见，因为更新操作充当重新分配操作。通过指定键，重新分配操作的成本为**O**(*1*)。

+   **集合**：集合操作有时被视为添加和重新分配操作的单一替代方案。如果键不存在，则集合将插入一个新的键/值对，或者它将重新分配指定键的值。不需要在同一实现中支持集合、添加和重新分配操作。与添加和更新一样，集合操作的成本为**O**(*1*)。

# 案例研究：游艺厅票券总额

**业务问题**：一位游艺厅经理希望通过消除游戏中的实体票券来降低成本。票券成本很高且浪费，因为一旦顾客兑换后，它们就必须被丢弃或回收。她决定引入一个电子积分系统，允许顾客通过积分而不是票券来赚取积分，并将积分数字化存储。一旦她安装了支持转换的硬件，她需要一个移动应用程序，允许她和她的顾客高效地跟踪他们的当前积分总额。

这个应用程序有几个关键要求。首先，它应该仅根据客户在登记时提供的姓名存储客户数据。其次，它必须保持所有获得、损失和兑换的积分的累计总数。第三，它必须能够显示任何给定时间点的客户的积分和游艺厅中的客户总数。最后，它应该允许删除单个客户记录或一次性删除所有客户记录。基于这些要求，开发者决定使用字典来跟踪所有客户积分将是最有效的方法，因此核心类的功能将基于这种数据结构。

**C#**

C#提供了泛型集合`Dictionary<TKey, TValue>`。这个类提供了我们预期在具体字典实现中看到的所有基本操作，并且增加了泛型类型转换的优势：

```py
    Dictionary<string, int> _points; 
    public PointsDictionary() 
    { 
        _points = new Dictionary<string, int>(); 
    } 

```

使用`Dictionary<TKey, TValue>`，我们为我们的类创建了一个名为`_points`的私有字段。我们的构造函数实例化这个字段，为我们提供了构建`PointsDictionary`类的基础数据结构：

```py
    //Update - private 
    private int UpdateCustomerPoints(string customerName, int points)  
    { 
        if (this.CustomerExists(customerName)) 
        { 
            _points[customerName] = _points[customerName] += points; 
            return _points[customerName]; 
        } 
        return 0; 
    } 

```

`UpdateCustomerPoints(string customerName, int points)`方法为我们这个类提供了核心的*更新*功能。该方法首先确认键是否存在于我们的集合中。如果键不存在，该方法立即返回`0`。否则，我们使用下标符号来同时获取键并更新键的值。再次使用下标符号，我们最终将更新后的值返回给调用者。

我们将这个方法设为私有，选择创建几个更适合我们业务需求的额外更新方法。稍后讨论的这些公共方法将向调用者公开更新功能：

```py
    //Add 
    public void RegisterCustomer(string customerName) 
    { 
        this.RegisterCustomer(customerName, 0); 
    } 

    public void RegisterCustomer(string customerName, int previousBalance) 
    { 
        _points.Add(customerName, previousBalance); 
    } 

```

两个`RegisterCustomer()`方法为我们这个类提供了*添加*功能。在两种情况下，我们都需要一个客户名称作为键。如果返回的客户在登记时带有之前的余额，我们希望承认这一点，因此我们的类会重载该方法。最终，重载的方法调用`Dictionary<TKey, TValue>.Add(T)`来将新记录插入到集合中：

```py
    //Get 
    public int GetCustomerPoints(string customerName) 
    { 
        int points; 
        _points.TryGetValue(customerName, out points); 

        return points; 
    } 

```

我们的*获取*功能是通过`GetCustomerPoints(string customerName)`方法引入的。在这个方法中，我们使用`TryGetValue()`来安全地确认`customerName`键是否存在，并同时获取其值。如果键不存在，应用程序会优雅地处理这个问题，并且不会为`points`分配任何值。然后，该方法返回`points`中当前设置的任何值：

```py
    //Update - public 
    public int AddCustomerPoints(string customerName, int points) 
    { 
        return this.UpdateCustomerPoints(customerName, points); 
    } 

    public int RemoveCustomerPoints(string customerName, int points) 
    { 
        return this.UpdateCustomerPoints(customerName, -points); 
    } 

    public int RedeemCustomerPoints(string customerName, int points) 
    { 
        //Perform any accounting actions 
        return this.UpdateCustomerPoints(customerName, -points); 
    } 

```

接下来，我们来看公共更新方法，`AddCustomerPoints(string customerName, int points)`、`RemoveCustomerPoints(string customerName, int points)`和`RedeemCustomerPoints(string customerName, int points)`。每个这些方法都调用私有的`UpdateCustomerPoints(string customerName, int points)`方法，但在调用之前，后两种情况会先对`points`取反：

```py
    //Remove 
    public int CustomerCheckout(string customerName) 
    { 
        int points = this.GetCustomerPoints(customerName); 
        _points.Remove(customerName); 
        return points;  
    } 

```

`CustomerCheckout(string customerName)` 方法引入了集合的 *remove* 功能。该方法首先记录客户键的最终值，然后调用 `Dictionary<TKey, TValue>.Remove(T)` 从集合中删除客户的键。最后，它将客户的最后积分值返回给调用者：

```py
    //Contains 
    public bool CustomerExists(string customerName) 
    { 
        return _points.ContainsKey(customerName); 
    } 

```

`Dictionary<TKey, TValue>` 接口提供了一个方便的 `ContainsKey()` 方法，该方法被 `CustomerExists(string customerName)` 方法用来引入我们类的 *contains* 功能：

```py
    //Count 
    public int CustomersOnPremises() 
    { 
        return _points.Count; 
    } 

```

使用 `Dictionary<TKey, TValue>` 类的 `Count` 字段，`CustomersOnPremises()` 提供了 *count* 功能：

```py
    public void ClosingTime() 
    { 
        //Perform any accounting actions 
        _points.Clear(); 
    } 

```

最后，根据我们的业务需求，我们需要一种方法来从集合中移除所有对象。`ClosingTime()` 方法使用 `Dictionary<TKey, TValue>.Clear()` 方法来完成这个任务。

**Java**

如前所述，Java 提供了一个 `Dictionary` 类，但它已被弃用，转而使用实现 `Map<K, V>` 接口的任何类。`HashMap<K, V>` 实现了该接口，并基于哈希表提供字典。与先前的 C# 示例一样，`HashMap<K, V>` 类公开了我们预期在字典的具体实现中看到的所有基本操作：

```py
    HashMap<String, Integer> _points; 
    public PointsDictionary() 
    { 
        _points = new HashMap<>(); 
    } 

```

`HashMap<K, V>` 的实例成为我们 Java `PointsDictionary` 类的核心。再次强调，我们命名私有字段为 `_points`，而我们的构造函数实例化了集合。你可能注意到，当我们实例化 `_points` 集合时，我们没有显式声明类型。在 Java 中，当我们已经在声明时定义了键和值类型时，实例化时不需要显式声明类型。如果你真的想声明类型，这将在编译器中生成警告：

```py
    private Integer UpdateCustomerPoints(String customerName, int points) 
    { 
        if (this.CustomerExists(customerName)) 
        { 
            _points.put(customerName, _points.get(customerName) + points); 
            return _points.get(customerName); 
        }  
        return 0; 
    } 

```

`UpdateCustomerPoints(string customerName, int points)` 方法为我们这个类提供了核心的 *update* 功能。该方法首先确认键是否存在于我们的集合中。如果键不存在，该方法立即返回 `0`。否则，我们使用 `put()` 和 `get()` 来更新键的值。再次使用 `get()`，我们最终将更新后的值返回给调用者：

```py
    //Add 
    public void RegisterCustomer(String customerName) 
    { 
        this.RegisterCustomer(customerName, 0); 
    } 

    public void RegisterCustomer(String customerName, int previousBalance) 
    { 
        _points.put(customerName, previousBalance); 
    } 

```

两个 `RegisterCustomer()` 方法为我们这个类提供了 *add* 功能。在两种情况下，我们都需要一个客户名称作为键。如果一个返回的客户有之前的余额，我们希望承认这一点，以便我们的类重载该方法。最终，重载的方法调用 `HashMap<K, V>.put(E)` 将新记录插入到集合中：

```py
    //Get 
    public Integer GetCustomerPoints(String customerName) 
    { 
        return _points.get(customerName) == null ? 0 : _points.get(customerName); 
    } 

```

我们的 *get* 功能是通过 `GetCustomerPoints(string customerName)` 方法引入的。在这个方法中，我们使用 `get()` 方法，并检查返回值是否不为 null，以安全地确认 `customerName` 键是否存在。使用三元运算符，如果不存在则返回 `0`，如果存在则返回值：

```py
    //Update 
    public Integer AddCustomerPoints(String customerName, int points) 
    { 
        return this.UpdateCustomerPoints(customerName, points); 
    } 

    public Integer RemoveCustomerPoints(String customerName, int points) 
    { 
        return this.UpdateCustomerPoints(customerName, -points); 
    } 

    public Integer RedeemCustomerPoints(String customerName, int points) 
    { 
        //Perform any accounting actions 
        return this.UpdateCustomerPoints(customerName, -points); 
    } 

```

接下来，我们来看公共更新方法，`AddCustomerPoints(String customerName, int points)`、`RemoveCustomerPoints(String customerName, int points)` 和 `RedeemCustomerPoints(String customerName, int points)`。这些方法中的每一个都会调用私有的 `UpdateCustomerPoints(String customerName, int points)` 方法，但在后两种情况下，它首先会取反 `points`：

```py
    //Remove 
    public Integer CustomerCheckout(String customerName) 
    { 
        Integer points = this.GetCustomerPoints(customerName); 
        _points.remove(customerName); 
        return points; 
    } 

```

`CustomerCheckout(String customerName)` 方法引入了集合的 *remove* 功能。该方法首先记录客户键的最终值，然后调用 `HashMap<K, V>.remove(E)` 从集合中删除客户的键。最后，它将客户的最后积分值返回给调用者：

```py
    //Contains 
    public boolean CustomerExists(String customerName) 
    { 
        return _points.containsKey(customerName); 
    } 

```

`HashMap<K, V>` 方法提供了一个方便的 `containsKey()` 方法，`CustomerExists(String customerName)` 方法使用它来引入我们类的 *contains* 功能：

```py
    //Count 
    public int CustomersOnPremises() 
    { 
        return _points.size(); 
    } 

```

使用 `HashMap<K, V>` 类的 `size()` 字段，`CustomersOnPremises()` 提供了 *count* 功能：

```py
    //Clear 
    public void ClosingTime() 
    { 
        //Perform accounting actions 
        _points.clear(); 
    } 

```

最后，根据我们的业务需求，我们需要一种方法来从集合中移除所有对象。`ClosingTime()` 方法使用 `HashMap<K, V>.clear()` 方法来完成这项任务。

**Objective-C**

对于我们的 Objective-C 示例，我们将使用 `NSMutableDictionary` 类簇来表示我们的集合。`NSMutableDictionary` 类簇并没有暴露我们预期在字典的具体实现中看到的所有基本操作，但那些不直接可用的操作非常简单就可以复制。重要的是要注意，Objective-C 不允许将标量值添加到 `NSDictionary` 或 `NSMutableDictionary` 集合的实例中。因此，由于我们试图存储整数作为值，我们必须在将它们添加到集合之前，将每个 `NSInteger` 标量放入 `NSNumber` 包装器中。不幸的是，这给我们的实现增加了一些开销，因为所有这些值都必须在插入或从集合中检索时装箱和拆箱：

```py
    @interface EDSPointsDictionary() 
    { 
        NSMutableDictionary<NSString*, NSNumber*> *_points; 
    } 

    @implementation EDSPointsDictionary 

    -(instancetype)init 
    { 
        if (self = [super init]) 
        { 
            _points = [NSMutableDictionary dictionary]; 
        } 

        return self; 
    } 

```

使用类簇 `NSMutableDictionary`，我们为我们的类创建了一个名为 `_points` 的实例变量。我们的初始化器实例化了这个字典，为我们提供了构建 `PointsDictionary` 类的基础数据结构：

```py
    -(NSInteger)updatePoints:(NSInteger)points 
        forCustomer:(NSString*)customerName 
    {  
        if ([self customerExists:customerName]) 
        { 
            NSInteger exPoints = [[_points objectForKey:customerName] integerValue]; 
            exPoints += points; 

            [_points setValue:[NSNumber numberWithInteger:exPoints] forKey:customerName]; 
            return [[_points objectForKey:customerName] integerValue]; 
        } 
        return 0; 
    } 

```

`updatePoints:forCustomer:` 方法为我们类提供了核心的 *update* 功能。该方法首先通过调用我们的 `customerExists:` 方法来确认键是否存在于我们的集合中。如果键不存在，该方法立即返回 `0`。否则，该方法使用 `objectForKey:` 来获取存储的 `NSNumber` 对象。从这个对象中，我们立即通过调用对象的 `integerValue` 来提取 `NSInteger` 值。接下来，值在字典中使用 `setValue:forKey:` 调整并更新。再次使用 `objectForKey:`，我们最终将更新的值返回给调用者：

```py
    //Add 
    -(void)registerCustomer:(NSString*)customerName 
    { 
        [self registerCustomer:customerName withPreviousBalance:0]; 
    } 

    -(void)registerCustomer:(NSString*)customerName 
        withPreviousBalance:(NSInteger)previousBalance 
    { 
        NSNumber *points = [NSNumber numberWithInteger:previousBalance]; 
        [_points setObject:points forKey:customerName]; 
    } 

```

`registerCustomer:` 方法为我们的类提供了 *添加* 功能。在两种情况下，我们都需要一个客户名称作为键。如果一个回头客在结账时带有之前的余额，我们希望确认这一点，以便我们的类在 `registerCustomer:withPreviousBalance:` 中重载该方法。最终，重载的方法调用 `setObject:forKey:` 将新的键/值对插入字典中：

```py
    //Get 
    -(NSInteger)getCustomerPoints:(NSString*)customerName 
    { 
        NSNumber *rawsPoints = [_points objectForKey:customerName]; 
        return rawsPoints ? [rawsPoints integerValue] : 0; 
    } 

```

我们的 *获取* 功能是通过 `getCustomerPoints:` 方法引入的。在这个方法中，我们使用 `objectForKey:` 获取传递的键的 `NSNumber` 对象，并将其分配给 `rawPoints`。接下来，该方法检查 `rawPoints` 是否不为 `nil`，如果可用，则返回 `rawPoints` 的 `integerValue`，否则返回 0：

```py
    //Update 
    -(NSInteger)addPoints:(NSInteger)points 
        toCustomer:(NSString*)customerName 
    { 
        return [self updatePoints:points forCustomer:customerName]; 
    } 

    -(NSInteger)removePoints:(NSInteger)points 
        fromCustomer:(NSString*)customerName 
    { 
        return [self updatePoints:-points forCustomer:customerName]; 
    } 

    -(NSInteger)redeemPoints:(NSInteger)points 
        forCustomer:(NSString*)customerName 
    { 
        //Perform any accounting actions 
        return [self updatePoints:-points forCustomer:customerName]; 
    } 

```

接下来，我们来看公共更新方法，`addPoints:toCustomer:`, `removePoints:fromCustomer:` 和 `redeemPoints:forCustomer:`。这些方法中的每一个都调用私有的 `updatePoints:forCustomer:` 方法，但在后者两种情况下，它首先对 `points` 取反：

```py
    -(NSInteger)customerCheckout:(NSString*)customerName 
    { 
        NSInteger points = [[_points objectForKey:customerName] integerValue]; 
        [_points removeObjectForKey:customerName]; 
        return points; 
    } 

```

`customerCheckout:` 方法引入了集合的 *删除* 功能。该方法首先记录客户键的最终值，然后调用 `removeObjectForKey:` 从集合中删除客户的键。最后，它将客户的最后积分值返回给调用者：

```py
    //Contains 
    -(bool)customerExists:(NSString*)customerName 
    { 
        return [_points objectForKey:customerName]; 
    } 

```

`NSMutableDictionary` 类簇不提供一种机制来确定集合中是否存在键。一个简单的解决方案是直接调用 `objectForKey:`；如果返回的值是 `nil`，则表示键不存在，`nil` 评估为 `NO`。基于这个原则，因此我们的 `customerExists:` 方法简单地返回 `objectForKey:`，允许返回值被评估为 `BOOL`：

```py
    //Count 
    -(NSInteger)customersOnPremises 
    { 
        return [_points count]; 
    } 

```

使用 `NSDictionary` 类的 `count` 属性，`customersOnPremises` 提供了 *计数* 功能：

```py
    //Clear 
    -(void)closingTime 
    { 
        [_points removeAllObjects]; 
    } 

```

最后，根据我们的业务需求，我们需要一种方法来从集合中删除所有对象。`closingTime` 方法使用 `removeAllObjects` 方法来完成这项任务。

**Swift**

Swift 提供的 `Dictionary` 类，与 Objective-C 的 `NSMutableDictionary` 类一样，并不暴露我们在字典数据结构的具体实现中期望看到的所有操作。同样，这些缺失的功能很容易复制。值得注意的是 Swift 字典的值类型与其 Objective-C 对应类型之间的区别。由于 Swift 中的原始类型被包装在 `structs` 中，我们可以毫无问题地将 `Int` 对象添加到我们的集合中：

```py
    var _points = Dictionary<String, Int>() 

```

使用 `Dictionary` 类，我们为我们的类创建了一个私有属性，称为 `_points`。由于我们的属性是声明和实例化同时进行的，且没有其他自定义代码需要实例化，我们可以排除显式的公共初始化器，并依赖于默认初始化器：

```py
    public func updatePointsForCustomer(points: Int, customerName: String) -> Int 
    { 
        if customerExists(customerName) 
        { 
            _points[customerName] = _points[customerName]! + points 
            return _points[customerName]! 
        } 
        return 0 
    } 

```

`updatePointsForCustomer()` 方法为我们类的核心 *更新* 功能提供支持。该方法首先通过调用我们的 `customerExists()` 方法来确认键是否存在于我们的集合中。如果键不存在，该方法立即返回 `0`。否则，该方法使用下标符号来获取存储的值。接下来，该值在字典中进行调整和更新，同样使用下标符号。最后，我们将更新后的值返回给调用者：

```py
    //Add 
    public func registerCustomer(customerName: String) 
    { 
        registerCustomerWithPreviousBalance(customerName, previousBalance: 0) 
    } 

    public func registerCustomerWithPreviousBalance(customerName: String, previousBalance: Int) 
    { 
        _points[customerName] = previousBalance; 
    } 

```

`registerCustomer()` 方法为我们类提供了 *添加* 功能。在两种情况下，我们都需要一个客户名称作为键。如果返回的客户带有之前的余额登记入住，我们希望承认这一点，以便我们的类在 `registerCustomerWithPreviousBalance()` 中重载该方法。最终，重载的方法使用下标符号将新的键/值对插入到字典中：

```py
    //Get 
    public func getCustomerPoints(customerName: String) -> Int 
    { 
        let rawsPoints = _points[customerName] 
        return rawsPoints != nil ? rawsPoints! : 0; 
    } 

```

我们的 *获取* 功能是通过 `getCustomerPoints()` 方法引入的。在这个方法中，我们使用下标符号来获取键的值，但在返回值之前，我们确认返回值不是 `nil`。如果值不是 `nil`，我们的方法返回该值；否则，它返回 `0`：

```py
    //Update 
    public func addPointsToCustomer(points: Int, customerName: String) -> Int 
    { 
        return updatePointsForCustomer(points, customerName: customerName) 
    } 

    public func removePointsFromCustomer(points: Int, customerName: String) -> Int 
    { 
        return updatePointsForCustomer(-points, customerName: customerName) 
    } 

    public func redeemPointsForCustomer(points: Int, customerName: String) -> Int 
    { 
        //Perform any accounting actions 
        return updatePointsForCustomer(-points, customerName: customerName) 
    } 

```

接下来，我们来看公共更新方法，`addPointsToCustomer()`、`removePointsFromCustomer()` 和 `redeemPointsForCustomer()`。这些方法中的每一个都调用私有的 `updatePointsForCustomer()` 方法，但在调用之前，它对后两种情况下的 `points` 进行取反：

```py
    public func customerCheckout(customerName: String) -> Int 
    { 
        let points = _points[customerName] 
        _points.removeValueForKey(customerName) 
        return points!; 
    } 

```

`customerCheckout()` 方法引入了集合的 *移除* 功能。该方法首先记录客户键的最终值，然后调用 `removeObjectForKey:` 从集合中删除客户的键。最后，它将客户的最后积分值返回给调用者：

```py
    //Contains 
    public func customerExists(customerName: String) -> Bool 
    { 
        return _points[customerName] != nil 
    } 

```

与 `NSMutableDictionary` 类似，`Dictionary` 不提供一种机制来确定集合中是否存在键。幸运的是，我们的 Objective-C 中的解决方案在 Swift 中同样适用。我们的方法使用下标符号，如果返回的值是 `nil`，则键不存在，`nil` 评估为 `false`。因此，基于这个原则，我们的 `customerExists()` 方法简单地返回 `_points[cusrtomerName]`，允许返回值被评估为 `Bool`：

```py
    //Count 
    public func customersOnPremises() -> Int 
    { 
        return _points.count 
    } 

```

通过 `Dictionary` 类的 `count` 属性，`customersOnPremises()` 提供了 *计数* 功能：

```py
    //Clear 
    public func closingTime() 
    { 
        _points.removeAll() 
    } 

```

最后，根据我们的业务需求，我们需要一种方法来从集合中移除所有对象。`closingTime()` 方法使用 `Dictionary.removeAll()` 方法来完成这项任务。

# 高级主题

现在我们已经考察了字典在常见应用中的使用方式，我们应该花些时间来探讨字典在底层是如何实现的。大多数字典分为两种不同的类型：基于哈希表和基于搜索树。尽管这两种方法的机制相似，并且它们通常共享许多相同的方法和功能，但每种类型的内部工作原理和理想应用却非常不同。

## 基于哈希表的字典

字典最常见的一种实现方式是基于哈希表的关联数组。当正确实现时，哈希表方法非常高效，允许进行**O**(1)复杂度的搜索、插入和删除操作。在我们考察的每种语言中，基本的字典类默认都是基于哈希表的。基于哈希表的字典的一般概念是，指定键的映射存储在数组的索引中，该索引是通过将哈希函数应用于键获得的。调用者随后检查数组中相同索引处的指定键，并使用存储在该处的绑定来检索元素的值。

基于哈希表的字典有一个缺点，即哈希函数有可能产生**冲突**，或者有时会尝试将两个键映射到相同的索引。因此，基于哈希表的实现必须有一种机制来解决这个问题。存在许多**冲突解决策略**，但这些细节超出了本文的范围。

## 基于搜索树的字典

字典较少见的实现方式是基于搜索树的关联数组。基于搜索树的字典非常适合按某些标准或值的属性对键和值进行排序，并且可以构建以更高效地处理自定义键或值类型。基于搜索树的实现的一个优点是增加了超出先前描述的基本函数的操作，例如找到与指定键相似的映射的能力。然而，这些优点是有代价的，因为基于搜索树的实现的基本操作成本更高，而集合本身对可以处理的数据类型有更严格的限制。有关基于搜索树的字典的排序操作将在第十二章*排序：从混乱中带来秩序*中更详细地讨论。

# 概述

在本章中，你学习了字典或关联数组的基
