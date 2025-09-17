# 第八章。结构体：复杂类型

**结构体**是一组数据变量或值的集合，这些变量或值被组织在单个内存块下，而数据结构通常是某种以某种方式相互关联的对象集合。因此，结构体，也称为结构，更多的是一种复杂的数据类型，而不是数据结构。这个定义听起来很简单，但在这个情况下，外表是欺骗性的。结构体的主题是复杂的，我们正在检查的每种语言在支持结构体方面都有其独特的特点，如果它们支持的话。

在本章中，我们将涵盖以下内容：

+   结构数据结构的定义

+   创建结构体

+   结构体的常见应用

+   每种语言中结构体的示例

+   枚举

# 基础知识

由于语言之间的支持不同，我们将在本章采取不同的方法。我们不会将结构体作为一个整体来检查，然后再检查一个案例研究，而是将每种语言的结构体和案例研究同时进行检查。这将给我们机会在适当的环境中检查每种语言中结构体的细微差别。

## C#

在 C#中，结构体被定义为封装相关字段的小组值的值类型，这听起来与底层 C 语言实现非常相似。然而，C#结构体实际上与 C 中的结构体有很大不同，它们更类似于那种语言的常规类。例如，C#结构体可以有方法、字段、属性、常量、索引器、运算符方法、嵌套类型和事件，以及定义的构造函数（但不包括默认构造函数，它是自动定义的）。结构体还可以实现一个或多个接口，所有这些都使得 C#版本比 C 更加灵活。

然而，将结构体视为轻量级类是错误的。C#结构体不支持继承，这意味着它们不能从类或其他结构体继承，也不能用作其他结构或类的基类。结构体成员不能声明为抽象的、受保护的或虚拟的。与类不同，结构体可以在不使用`new`关键字的情况下实例化，尽管这样做会阻止结果对象在所有字段被分配之前被使用。最后，也许最重要的是，结构体是值类型，而类是引用类型。

这个最后一点不能过分强调，因为它代表了选择结构体而不是类的主要优势。结构体是值的集合，因此不存储诸如数组之类的对象的引用。因此，当你将结构体传递给方法时，它是按值传递而不是按引用传递。此外，根据 MSDN 文档，作为值类型，结构体不需要分配堆内存，因此不携带类在内存和处理需求方面的开销。

### 注意

这意味着什么？为什么这有益？当你使用 new 运算符创建一个新的类时，返回的对象将在堆上分配。另一方面，当你实例化一个结构体时，它直接在堆栈上创建，这带来了性能提升，因为堆栈提供的内存访问速度比堆快得多。只要你不过度使用堆栈并导致堆栈溢出，有策略地使用结构体可以极大地提高你应用程序的性能。

现在你可能自己在想，*如果结构体这么棒，我们为什么还要有类呢？* 首先，C#中结构体的应用非常有限。根据微软的说法，你应该只在类型的实例很小且生命周期短暂，或者它们通常嵌入在其他对象中时，才考虑使用结构体而不是类。此外，除非结构体至少满足以下三个标准之一，否则不应定义结构体：

+   结构体将逻辑上表示一个类似于整数、双精度浮点数等原始类型的单个值

+   结构体的每个实例都将小于 16 字节

+   结构体中的数据一旦实例化后将是不可变的

+   结构体不需要反复装箱和拆箱

这些要求相当严格！当你考虑你可以实际用结构体做什么时，前景会稍微变得糟糕一些。这里有一个提示--不多。让我们比较一下结构体和类的能力：

+   您可以设置和访问单个组件--类也可以这样做。

+   您可以将结构体传递给函数--是的，您也可以用类这样做。

+   您可以使用赋值运算符（`=`）将一个结构体的内容赋值给另一个结构体--这里没有特别之处。

+   您可以从函数中返回一个结构体，这实际上会创建结构体的一个副本，因此现在堆栈上有两个。类？检查。然而，在这方面类更优越，因为当一个函数返回类的实例时，对象是通过引用传递的，因此不需要创建额外的副本。

+   结构体*不能*使用等号运算符（`==`）进行相等性测试，因为结构体可能包含其他数据。然而，类可以使用等号运算符进行比较。事实上，如果您想在结构体中实现相同的功能，您必须逐个字段比较，这是很繁琐的。

如果有人要对这场对决进行评分，我认为结果可能看起来像*结构体：4，类：5（也许 6）*。所以很明显，在功能和便利性方面，类更加灵活，这就是为什么以 C 为基础的高级语言通常提供机制来实现这些更复杂对象的原因。

这并不意味着结构体没有其价值。尽管它们的实用性局限于非常狭窄的场景，但在某些时候，结构体是完成这项工作的正确工具。

### 在 C#中创建结构体

在 C#中创建结构体是一个相当简单的过程。我们只有两个要求：使用`using System`和用`struct`关键字声明我们的对象。以下是一个示例：

```py
    using System; 

    public struct MyStruct 
    { 
        private int xval; 
        public int X 
        { 
            get  
            { 
                return xval; 
            } 
            set  
            { 
                if (value < 100) 
                    xval = value; 
            } 
        } 

        public void WriteXToConsole() 
        { 
            Console.WriteLine("The x value is: {0}", xval); 
        } 
    } 

    //Usage 
    MyStruct ms1 = new MyStruct(); 
    MyStruct ms2 = MyStruct(); 

    ms.X = 9; 
    ms.WriteXToConsole(); 

    //Output 
    //The x value is: 9 

```

如前例所示，我们的结构体使用私有后置字段、公共访问器和名为`WriteXToConsole()`的一个实例方法声明，这些都是 C#中结构体的完全合法特性。注意`MyStruct`的两个实例。第一个使用`new`关键字实例化，而第二个没有。再次强调，这两个操作在 C#中都是完全有效的，尽管后者要求你在以任何方式使用对象之前必须填充所有成员属性。如果你将定义中的`struct`关键字改为`class`，第二个初始化器将无法编译。

接下来，我们将从第三章的例子中进行分析，*列表：线性集合*。在该章节的案例研究中，我们构建了一个存储`Waypoint`对象列表的数据结构。以下是 C#中`Waypoint`类的样子：

```py
    public class Waypoint 
    { 
        public readonly Int32 lat; 
        public readonly Int32 lon; 
        public Boolean active { get; private set; } 

        public Waypoint(Int32 latitude, Int32 longitude) 
        { 
            this.lat = latitude; 
            this.lon = longitude; 
            this.active = true; 
        } 

        public void DeactivateWaypoint() 
        { 
            this.active = false; 
        } 

        public void ReactivateWaypoint() 
        { 
            this.active = true; 
        } 
    }
```

如你所见，这个类非常简单。简单到让人质疑这样一个简单的值集合是否值得分配给类所提供的开销和资源，尤其是当你考虑到我们的列表可能包含数百个这样的`Waypoint`对象时。我们能否通过将类转换为结构体来提高性能，而无需进行重大的重构以支持这种更改？首先，我们需要确定这样做是否推荐，甚至是否可行，我们可以通过检查我们的结构体准则规则来做出这个决定。

### 规则 1：结构体将逻辑上表示单个值

在这种情况下，我们的类有三个字段，即`lat`、`lon`和`active`。三个显然不是单个，但根据规则，结构体必须逻辑上表示单个值，因此我们将类转换为结构体的计划仍然是有效的。这是因为`Waypoint`对象代表二维空间中的单个位置，而我们至少需要两个值来表示二维坐标，所以这里没有违反规则。此外，活动属性表示航点的状态，因此这也符合特征性可接受。在你对这种解释提出异议之前，让我指出，即使是微软也对此规则不太严格。例如，`System.Drawing.Rectangle`被定义为结构体，该类型存储表示矩形大小和位置的四个整数。大小和位置是单个对象的两个属性，这被认为是可接受的，所以我相信`Waypoint`在这里是合适的。

### 规则 2：结构体的每个实例必须小于 16 字节

我们的`Waypoint`类很容易符合这一安全规则。参考第一章，*数据类型：基础结构*，`Int32`结构体长度为 4 字节，布尔原始类型长度仅为 1 字节。这意味着单个`Waypoint`实例的总重量仅为九字节，我们还有七个字节的空间。

### 规则 3：数据必须是不可变的

结构体应该理想上是不可变的，这与它们作为值类型的状态有关。如前所述，每当传递一个值类型时，你最终得到的是该值的副本，而不是原始值的引用。这意味着当你更改结构体内的值时，你只是在更改该结构体，而不会影响到堆栈中可能存在的其他任何结构体。

这个要求可能对我们来说是一个问题，而且不是一个小问题。在我们的应用程序中，我们选择在对象本身上存储`Waypoint`值的活跃状态，而这个字段肯定不是不可变的。我们可以以某种方式将属性移出`Waypoint`类，但这样做需要比如果我们简单地让它保持原样进行更多的重构。由于我们目前想要避免重大的重构，我们将保持该字段不变，并将此规则视为对我们计划的打击。我们唯一的补救办法是检查我们代码中对`Waypoint`对象的使用，以确保我们永远不会创建一个`Waypoint`实例以这种方式传递，以至于我们失去了对正确实例的关注。从技术上来说，只要`Waypoint`通过下一个要求，我们仍然在业务中。

### 规则 4：结构体不需要重复装箱

由于`Waypoint`对象一旦实例化后就是直接使用的，因此每个实例很少，如果不是从未，会被装箱或解箱。因此，我们的类通过了这个测试，并符合转换为结构体的条件。

### 转换

接下来的问题是，*能否将 Waypoint 类转换为结构体？* 在我们的类中有三个需要注意的点可能需要解决。首先，我们有一个可变的`active`字段需要处理。在其当前形式下，这个字段并不太像结构体，因为它实际上应该是不可变的。由于在这个阶段我们真的没有其他办法，我们不得不以另一种方式处理它。主要来说，这意味着我们需要非常严格地监控我们对`Waypoint`对象的使用，以确保当我们认为我们在处理原始结构体时，我们实际上并没有在处理结构体的副本。尽管这可能会变得繁琐，但这并不不合理。我们的下一个关注点是定义的构造函数，但由于这不是没有参数或默认构造函数，所以这里一切正常，我们可以继续前进。最后，我们的类有两个名为`DeactivateWaypoint()`和`ReactivateWaypoint()`的公共方法。由于 C#也允许在结构体中使用公共方法，这两个方法在这里也是可以的。事实上，我们真正需要做的，将这个类转换为结构体，就是将`class`关键字改为`struct`关键字！以下是我们的结果代码：

```py
    public struct Waypoint 
    { 
        public readonly Int32 lat; 
        public readonly Int32 lon; 
        public Boolean active { get; private set; } 

        public Waypoint(Int32 latitude, Int32 longitude) 
        { 
            this.lat = latitude; 
            this.lon = longitude; 
            this.active = true; 
        } 

        public void DeactivateWaypoint() 
        { 
            this.active = false; 
        } 

        public void ReactivateWaypoint() 
        { 
            this.active = true; 
        } 
    }; 

```

最后，我们需要知道这个更改是否会在整体上代表我们应用的任何改进。没有对应用在运行时的广泛测试和分析，我们无法确定地说，但可能性很大，这个修改将对我们越野骑行应用的整体性能产生积极影响，而不会引入任何进一步的重构需求。

## Java

这将是一个简短的讨论，因为 Java 不支持结构体。显然，Java 的作者们决定，当这种语言最终从 C 编程的泥潭中爬出来时，它不会带着这些非面向对象的结构体四处奔波。因此，在 Java 中我们唯一的办法是创建一个具有公共属性的类来模拟结构体的行为，但没有任何性能上的优势。

## Objective-C

Objective-C 不支持直接使用结构体；然而，你可以在代码中实现和使用简单的 C 结构体。C 结构体与它们的 C#对应物类似，因为它们允许你将几个原始值组合成一个更复杂的值类型。但是，C 结构体不允许添加方法或初始化器，也不允许 C#结构体所享有的任何其他酷炫的面向对象编程特性。此外，C 结构体不能包含从`NSObject`继承的对象，因为这些是类而不是值类型。

话虽如此，在 Objective-C 应用程序中，结构体实际上是非常常见的。结构体最常见的一个应用是在**枚举**或**枚举类型**的定义中。枚举是一系列表示整数值的常量列表，其目的是在代码中创建更高层次的抽象，这样开发者就可以专注于值的含义，而不必担心它们在后台的实现方式。我们将在本章后面更详细地探讨枚举。

### 在 Objective-C 中创建结构体

Objective-C 中结构体的另一个常见来源可以在**Core Graphics 框架**中找到，该框架包含四个有用的结构体。我们将详细研究这些结构体，以展示如何在 Objective-C 中定义结构体：

+   `CGPoint`：这个结构体包含一个简单的二维坐标系，由两个`CGFloat`值组成。下面是`CGPoint`结构体的定义：

```py
        struct CGPoint { 
            CGFloat x; 
            CGFloat y; 
        }; 
        typedef struct CGPoint CGPoint; 

```

+   `CGSize`：这个结构体只是一个宽度和高度的容器，由两个`CGFloat`值组成。下面是`CGSize`结构体的定义：

```py
        struct CGSize { 
            CGFloat width; 
            CGFloat height; 
        }; 
        typedef struct CGSize CGSize; 

```

+   `CGRect`：这是一个定义矩形位置和大小的结构体，由一个`CGPoint`值和一个`CGSize`值组成。下面是`CGRect`结构体的定义：

```py
        struct CGRect { 
            CGPoint origin; 
            CGSize size; 
        }; 
        typedef struct CGRect CGRect; 

```

+   `CGVector`：这是一个仅包含二维向量的结构体，由两个`CGFloat`值组成。下面是`CGVector`结构体的定义：

```py
        struct CGVector { 
            CGFloat dx; 
            CGFloat dy; 
        }; 
        typedef struct CGVector CGVector; 

```

### 注意

你应该注意在每个结构体定义之后跟随的`typedef`和`struct`关键字。这一行是为了我们程序员的方便而包含的。无论何时我们需要调用这些结构体，如果结构体没有用`typedef`关键字装饰，我们都需要在调用结构体之前始终加上`struct`关键字，如下所示：

`struct CGRect rect;`

显然，这会很快变得令人厌烦。通过将`typedef`应用于结构体名称，我们允许调用者简单地使用结构体名称，而不需要`struct`关键字，如下所示：

`  struct CGRect rect;`

这使得我们的代码更容易编写，但同时也使得代码在长期来看更加简洁和易于阅读。

现在，我们将查看第三章中的`EDSWaypoint`类，并确定我们是否可以将该类转换为 C 结构体。以下是原始代码：

```py
    @interface EDSWaypoint() 
    { 
        NSInteger _lat; 
        NSInteger _lon; 
        BOOL _active; 
    } 

    @end 

    @implementation EDSWaypoint 

    -(instancetype)initWithLatitude:(NSInteger)latitude andLongitude:(NSInteger)longitude 
    { 
        if (self = [super init]) 
        { 
            _lat = latitude; 
            _lon = longitude; 
            _active = YES; 
        } 

        return self; 
    } 

    -(BOOL)active 
    { 
        return _active; 
    } 

    -(void)reactivateWaypoint 
    { 
        _active = YES; 
    } 

    -(void)deactivateWaypoint 
    { 
        _active = NO; 
    } 

    @end 

```

立刻在接口中，我们就看到了将这个类转换为结构体的一些问题。`_lat` 和 `_lon` ivars 都是 `NSInteger` 类，这意味着它们在结构体中使用是无效的，它们必须被移除或改为值类型。那么 `initWithLatitude:andLongitude:` 初始化器呢？不行，你也不能在 C 结构体中定义初始化器。所以，现在我们需要处理 `reactivateWaypoint` 和 `deactivateWaypoint` 方法。当然，这些简单的属性和方法肯定可以通过接受进入结构体的考验？不，它们不能。这里的一切都需要被移除。

因此，唯一剩下的问题就是我们应该如何处理 `_active` 值和相关的 `-(BOOL)active` 属性。实际上，`BOOL` 类型在结构体中使用是完全可以接受的，所以我们可以实际上保留这个属性。然而，`_active` 在 `EDSWaypoint` 结构体中确实代表了一个可变属性，这是不被提倡的，对吧？虽然不被提倡，但在 C 中结构体并不是不可变的。以下是一个使用 Core Graphics 结构体 `CGPoint` 的例子：

```py
    CGPoint p = CGPointMake(9.0, 5.2); 
    p.x = 9.8; 
    p.y = 5.5; 

```

如果你将此代码复制到你的应用程序中，编译器不会发出错误或警告，因为 `CGPoint` 不是不可变的，属性也不是只读的。因此，我们可以在最终的 struct 定义中保留 `_active` 值。不幸的是，对于 `-(BOOL)active` 属性来说，情况并非如此？像这样的属性访问器在 C 结构体中是禁止的，所以这个属性需要被移除，这代表了对我们应用程序处理 `Waypoint` 对象活动状态方式的重大改变。因此，如果我们想将这个类转换为结构体，我们将得到以下内容：

```py
    struct EDSWaypoint { 
        int lat; 
        int lon; 
        BOOL active; 
    }; 
    typedef struct EDSWaypoint EDSWaypoint; 

```

严格来说，`typedef` 声明不是必需的，但我们必须重构整个 `EDSWaypointList` 类以支持这些更改已经足够糟糕了。我们不应该再让我们的开发者每次想要访问这些类型之一时都要多输入八个额外的字符。

## Swift

就像在其他语言中一样，Swift 中的结构体是值类型，它们封装了一组相关的属性。与 C# 中的结构体类似，Swift 结构体比 C 结构体更像是一个常规类，并且与类共享以下所有能力：

+   能够定义属性以存储值

+   能够包含定义扩展功能的方法

+   能够定义下标以使用下标符号访问值

+   能够定义自定义初始化器

+   Swift 结构体可以被扩展以提供超出其初始化状态的额外功能

+   最后，Swift 结构体可以被定义为符合提供常规功能的协议

然而，请注意，Swift 的结构体不支持继承，这意味着它们不能从类或其他结构体继承，也不能作为其他结构体或类的基类。此外，它们不支持类型转换，以使编译器能够在运行时检查和解释实例的类型。这些结构体不能像类那样显式地被销毁以释放其资源，结构体也不支持自动引用计数进行内存管理。最后两点与 Swift 中的结构体（与其他语言一样）是值类型而不是类或引用类型的事实相关。

关于 Swift 的这一点需要再次强调。结构体是值的集合，因此它们不会像数组或字典等其他集合那样存储对象的引用。因此，当你将结构体作为参数传递给或从方法中返回时，它是按值传递而不是按引用传递。

那么，在 Swift 中何时应该选择使用结构体而不是类呢？Apple 的文档提供了一些一般性规则，以帮助你做出决定。你应该在以下情况下使用结构体：

+   你的对象的主要目的是收集一些简单的数据值

+   你预计你创建的对象在分配或发送该对象的实例时将被复制而不是被引用

+   你对象中的任何属性都是值类型，而不是类，你也期望它们的值会被复制而不是被引用

+   你的对象没有必要从现有对象或类型继承属性或行为

你会注意到，这个列表并不像 C# 中的相同列表那样严格，但它确实代表了一种很好的常识方法，用于决定使用结构体带来的价值是否超过了对象中有限的功能。

### 在 Swift 中创建结构体

如果你使用 Swift 超过五分钟，那么你很可能已经使用过一些内置的结构体，例如 `Int`、`String`、`Array`、`Dictionary` 以及 Swift 框架中定义的许多其他结构体。以下是一个使用 Swift 定义你自己的结构体的快速演示：

```py
    Public struct MyColor { 
        var red = 0 
        var green = 0 
        var blue = 0 
        var alpha = 0.0 
    } 

```

以下示例定义了一个名为 `MyColor` 的新结构体，它描述了一个基于 RGBA 的颜色定义。这个结构体有四个属性，分别称为 `red`、`green`、`blue` 和 `alpha`。尽管这些属性都已被定义为可变变量使用 `var`，但 Swift 中的存储属性也可以使用 `let` 定义为不可变。我们结构体中的前三个属性通过将其默认值设置为 `0` 被推断为 `Int` 类型，而剩余的属性通过将其默认值设置为 `0.0` 被推断为 `Double` 类型。由于我们尚未为我们的方法定义任何自定义初始化器，我们可以如下初始化这个对象的实例：

```py
    var color = MyColor()  
    color.red = 139 
    color.green = 0 
    color.blue = 139 
    color.alpha = .5 

```

上述代码初始化了我们的结构体，并将值设置为类似于 50%透明度的深洋红色。这个演示是好的，但初始化对于许多开发者的口味来说有点冗长。如果我们想在一行中创建一个新对象怎么办？在这种情况下，我们需要修改我们的结构体以包含一个自定义初始化器，如下所示：

```py
    public struct MyColor { 
        var red = 0 
        var green = 0 
        var blue = 0 
        var alpha = 0.0 

        public init(R: Int, G: Int, B: Int, A: Double) 
        { 
            red = R 
            green = G 
            blue = B 
            alpha = A 
        } 
    } 

var color = MyColor(R: 139, G:0, B:139, A:0.5) 

```

利用 Swift 允许结构体定义自定义初始化器的优势，我们创建了一个接受 RGBA 值并将它们分配给对象属性的 `init` 方法，极大地简化了对象创建。

现在，我们将查看 第三章 中的 `Waypoint` 类，并确定我们是否可以将该类转换为结构体。以下是原始代码：

```py
    public class Waypoint : Equatable 
    { 
        var lat: Int 
        var long: Int 
        public private(set) var active: Bool 

        public init(latitude: Int, longitude: Int) { 
            lat = latitude 
            long = longitude 
            active = true 
        } 

        public func DeactivateWaypoint() 
        { 
            active = false; 
        } 

        public func ReactivateWaypoint() 
        { 
            active = true; 
        } 
    } 

    public func == (lhs: Waypoint, rhs: Waypoint) -> Bool { 
        return (lhs.lat == rhs.lat && lhs.long == rhs.long) 
    } 

```

现在这是一个有趣的类对象。我们首先解决房间里的大象：`Equatable` 接口和名为 `==` 的公共函数被声明在类结构**外部**。我们的类必须实现 `Equatable` 接口，因为 `WaypointList` 中的几个方法需要比较两个 `Waypoint` 对象的相等性。没有这个接口和相关的 `==` 方法实现，这是不可能的，我们的代码也无法编译。幸运的是，Swift 结构体可以实施接口，如 `Equatable`，所以这根本不是问题，我们可以继续前进。

我们已经讨论并演示了 Swift 结构体可以定义自定义初始化器，所以我们的公共 `init` 方法就很好。`Waypoint` 类还有两个名为 `DeactivateWaypoint()` 和 `ActivateWaypoint()` 的方法。由于结构体旨在不可变，我们需要对类进行最后的更改，以将 `mutating` 关键字添加到每个方法中，以表示每个方法修改或突变实例中的一个或多个值。以下是我们的 `Waypoint` 类的最终版本：

```py
    public struct Waypoint : Equatable 
    { 
        var lat: Int 
        var long: Int 
        public private(set) var active: Bool 

        public init(latitude: Int, longitude: Int) { 
            lat = latitude 
            long = longitude 
            active = true 
        } 

        public mutating func DeactivateWaypoint() 
        { 
            active = false; 
        } 

        public mutating func ReactivateWaypoint() 
        { 
            active = true; 
        } 
    } 

    public func == (lhs: Waypoint, rhs: Waypoint) -> Bool { 
        return (lhs.lat == rhs.lat && lhs.long == rhs.long) 
    } 

```

### 注意

将 `mutating` 关键字添加到我们的实例方法中，将允许我们将 `Waypoint` 重新定义为结构体，但它也会给我们的实现引入一个新的限制。考虑以下示例：

`let point = Waypoint(latitude: 5, longitude: 10)`

`point.DeactivateWaypoint()`

这段代码将无法编译，并出现错误 `不可变的类型 'Waypoint' 只包含名为 DeactivateWaypoint 的 mutating 成员`。等等。现在怎么办？通过包含 `mutating` 关键字，我们也是明确地声明这个结构体是可变的类型。声明这个类型为不可变是可以的，除非你尝试调用其中一个 mutating 方法，这时代码将无法编译。在此之前，我们可以根据需要将 `Waypoint` 的任何实例声明为可变的 `var` 或不可变的 `let`，但现在，如果我们打算使用 `mutating` 方法，我们只能将这个对象声明为可变的实例。

# 枚举

如前所述，枚举增加了应用程序的抽象级别，并允许开发者关注值的含义，而不是担心值在内存中的存储方式。这是因为`enum`类型允许你用有意义的或易于记忆的名称标记特定的整数数值。

## 案例研究：地铁线路

**商业问题**：你与一个负责编写应用程序以跟踪地铁通勤列车团队的工程师合作。其中一个关键的业务需求是能够轻松识别列车当前位于哪个车站或正在前往哪个车站。每个车站都有一个独特的名称，但数据库通过其 ID 值（如 1100、1200、1300 等）跟踪车站。而不是通过名称跟踪车站，因为名称既繁琐又容易随时间变化，你的应用程序将利用车站 ID。然而，最初用名称而不是 ID 标记车站的原因是为了使通勤者更容易识别它们。这也适用于程序员，他们在编写代码时很难记住几十个甚至几百个车站的 ID。

你决定利用枚举数据结构来满足你的应用程序和开发者的需求。你的枚举将提供易于记忆的车站名称与它们关联的车站 ID 之间的映射，因此你的应用程序可以根据车站名称利用 ID，而你的程序员将使用名称。

**C#**

为了避免在大型车站多个列车线路重叠时产生混淆，我们不想简单地创建一个包含整个地铁线路所有车站的枚举。相反，我们将根据地铁的每条线路创建枚举。以下是一个定义为 C#枚举的 Silver Line：

```py
    public enum SilverLine 
    { 
        Wiehle_Reston_East = 1000, 
        Spring_Hill = 1100, 
        Greensboro = 1200, 
        Tysons_Corner = 1300, 
        McClean = 1400, 
        East_Falls_Church = 2000, 
        Ballston_MU = 2100, 
        Virginia_Sq_GMU = 2200, 
        Clarendon = 2300, 
        Courthouse = 2400, 
        Rosslyn = 3000, 
        Foggy_Bottom_GWU = 3100, 
        Farragut_West = 3200, 
        McPherson_Sq = 3300, 
        Metro_Center = 4000, 
        Federal_Triangle = 4100, 
        Smithsonian = 4200, 
        LEnfant_Plaza = 5000, 
        Federal_Center_SW = 5100, 
        Capital_South = 5200, 
        Eastern_Market = 5300, 
        Potomac_Ave = 5400, 
        Stadium_Armory = 6000, 
        Benning_Road = 6100, 
        Capital_Heights = 6200, 
        Addison_Road = 6300, 
        Morgan_Blvd = 6400, 
        Largo_Town_Center = 6500 
    } 

```

现在，无论我们想在何处使用`SilverLine`枚举的值，我们只需声明一个同名的值类型并分配一个值，如下所示：

```py
    SilverLine nextStop = SilverLine.Federal_Triangle; 
    nextStop = SilverLine.Smithsonian; 

```

在我们刚才看到的示例中，我们的代码初始化一个`SilverLine`值，以显示 Silver Line 的下一站为车站`4100`，使用`SilverLine.Federal_Triangle`。一旦车门在站台关闭，我们需要更新这个值以显示我们的列车正在前往车站`4200`，因此我们将值更新为`SilverLine.Smithsonian`。

**Java**

尽管 Java 不允许我们显式地定义结构体，但我们可以定义枚举。然而，定义可能不会像你预期的那样：

```py
    public enum SilverLine 
    { 
        WIEHLE_RESTON_EAST, 
        SPRING_HILL, 
        GREENSBORO, 
        TYSONS_CORNER, 
        MCCLEAN, 
        EAST_FALLS_CHURCH, 
        BALLSTON_MU, 
        VIRGINIA_SQ_GMU, 
        CLARENDON, 
        COURTHOUSE, 
        ROSSLYN, 
        FOGGY_BOTTOM_GWU, 
        FARRAGUT_WEST, 
        MCPHERSON_SQ, 
        METRO_CENTER, 
        FEDERAL_TRIANGLE, 
        SMITHSONIAN, 
        LENFANT_PLAZA, 
        FEDERAL_CENTER_SW, 
        CAPITAL_SOUTH, 
        EASTERN_MARKET, 
        POTOMAC_AVE, 
        STADIUM_ARMORY, 
        BENNING_ROAD, 
        CAPITAL_HEIGHTS, 
        ADDISON_ROAD, 
        MORGAN_BLVD, 
        LARGO_TOWN_CENTER 
    } 

```

你可能会注意到我们没有明确地为这些条目中的每一个分配整数值。这是因为 Java 不允许我们这样做。记住，Java 不支持结构体，所以在这个语言中的枚举实际上不是原语，而是它们自己类型的对象。因此，它们不遵循其他语言中枚举的规则，有些人认为 Java 枚举因此更加健壮。

对于我们计划使用此结构的情况，这个限制将是一个小障碍，因为我们不能直接将站点名称映射到它们关联的 ID 值。这里的一个选择是添加一个`public static`方法，它将操作`this`的字符串值，并使用该值在幕后将字符串映射到整数值。这可能是一个相当冗长的解决方案，但当你考虑到这是可能的这一事实时，它为解决整体业务问题开辟了一个全新的解决方案世界。

**Objective-C**

就像 Objective-C 不支持结构体一样，它也不直接支持枚举。幸运的是，在这种情况下，我们也可以使用底层的 C 语言枚举。下面是如何做的：

```py
    typedef enum NSUInteger
    {
        Wiehle_Reston_East = 1000,
        Spring_Hill = 1100,
        Greensboro = 1200,
        Tysons_Corner = 1300,
        McClean = 1400,
        East_Falls_Church = 2000,
        Ballston_MU = 2100,
        Virginia_Sq_GMU = 2200,
        Clarendon = 2300,
        Courthouse = 2400,
        Rosslyn = 3000,
        Foggy_Bottom_GWU = 3100,
        Farragut_West = 3200,
        McPherson_Sq = 3300,
        Metro_Center = 4000,
        Federal_Triangle = 4100,
        Smithsonian = 4200,
        LEnfant_Plaza = 5000,
        Federal_Center_SW = 5100,
        Capital_South = 5200,
        Eastern_Market = 5300,
        Potomac_Ave = 5400,
        Stadium_Armory = 6000,
        Benning_Road = 6100,
        Capital_Heights = 6200,
        Addison_Road = 6300,
        Morgan_Blvd = 6400,
        Largo_Town_Center = 6500
    } SilverLine;

```

首先，请注意，我们已经将`typedef`关键字集成到这个定义中，这意味着我们不需要在我们的代码中单独一行添加`SilverLine`枚举对象的声明。还要注意`enum`关键字，它是 C 中声明枚举所必需的。请注意，我们明确声明这个枚举是`NSUInteger`类型的值。我们在这里使用`NSUInteger`是因为我们不希望支持有符号值，但如果我们这样做，我们同样可以轻松地选择`NSInteger`来达到这个目的。最后，请注意，`enum`变量的实际名称在定义之后。

否则，我们的枚举定义与其他大多数基于 C 的语言的枚举定义相似，只是有几个注意事项。首先，如果你打算在当前文件的作用域之外使用枚举，则必须在头文件（`*.h`）中声明枚举。在任何情况下，枚举也必须在`@interface`或`@implementation`标签之外声明，否则你的代码将无法编译。最后，你的枚举名称必须在工作区内的所有其他对象中是唯一的。

**Swift**

Swift 中的结构体与 C#的结构体比 Objective-C 的结构体有更多的共同之处，这得益于它们的广泛灵活性。在我们的示例中，我们不会添加任何额外的方法或`init`函数，但如果我们需要，我们可以这样做：

```py
    public enum SilverLine : Int 
    { 
        case Wiehle_Reston_East = 1000 
        case Spring_Hill = 1100 
        case Greensboro = 1200 
        case Tysons_Corner = 1300 
        case McClean = 1400 
        case East_Falls_Church = 2000 
        case Ballston_MU = 2100 
        case Virginia_Sq_GMU = 2200 
        case Clarendon = 2300 
        case Courthouse = 2400 
        case Rosslyn = 3000 
        case Foggy_Bottom_GWU = 3100 
        case Farragut_West = 3200 
        case McPherson_Sq = 3300 
        case Metro_Center = 4000 
        case Federal_Triangle = 4100 
        case Smithsonian = 4200 
        case LEnfant_Plaza = 5000 
        case Federal_Center_SW = 5100 
        case Capital_South = 5200 
        case Eastern_Market = 5300 
        case Potomac_Ave = 5400 
        case Stadium_Armory = 6000 
        case Benning_Road = 6100 
        case Capital_Heights = 6200 
        case Addison_Road = 6300 
        case Morgan_Blvd = 6400 
        case Largo_Town_Center = 6500 
    } 

```

注意我们的定义中包含了`Int`声明。在大多数情况下，这并不是严格必要的，除非我们打算像我们在这里所做的那样明确地为条目设置值。这可以让编译器提前知道预期的类型，以便进行类型检查。如果我们选择省略显式值，我们也可以选择省略`Int`声明。

# 摘要

在本章中，你学习了结构体数据结构的基本定义，以及如何在适用语言中创建结构体。我们还考察了一些结构体的常见应用，包括非常常见的枚举数据类型。最后，我们查看了一些之前的代码示例，以检查我们是否可以使用结构体对象而不是自定义类来改进它们。
