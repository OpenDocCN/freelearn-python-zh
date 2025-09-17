# 第一章 数据类型：基础结构

将数据类型称为“基础结构”可能听起来有点名不副实，但当你考虑到开发者使用数据类型来构建他们的类和集合时，情况并非如此。因此，在我们检查适当的数据结构之前，快速回顾数据类型是个好主意，因为这些都是接下来内容的基石。本章旨在从 10,000 英尺的高度回顾最常见和最重要的基本数据类型。如果你已经对这些基本概念有很强的理解，那么你可以自由地浏览本章，甚至根据需要完全跳过它。

在本章中，我们将涵盖以下主题：

+   数值数据类型

+   类型转换、窄化、和宽化

+   32 位和 64 位架构关注点

+   布尔数据类型

+   逻辑操作

+   运算顺序

+   嵌套操作

+   短路操作

+   字符串数据类型

+   字符串的可变性

# 数值数据类型

对以下四种语言（C#、Java、Objective-C 和 Swift）中所有数值数据类型的详细描述，可以很容易地涵盖一本自己的书。在这里，我们将仅回顾每种语言中最常见的数值类型标识符。评估这些类型的最简单方法是基于数据的基本大小，使用每种语言的示例作为讨论的框架。

### 小贴士

**比较苹果与苹果！**

当你为多个移动平台开发应用程序时，你应该意识到你使用的语言可能共享一个数据类型标识符或关键字，但在底层，这些标识符可能并不等价。同样，一种语言中的相同数据类型在另一种语言中可能有不同的标识符。例如，考察 16 位无符号整数的情况，有时被称为`unsigned short`。嗯，在 Objective-C 中它被称为`unsigned short`。在 C#中，我们谈论的是`ushort`，而 Swift 则称之为`UInt16`。另一方面，Java 为 16 位无符号整数提供的唯一选择是`char`，尽管这个对象通常不会用于数值。这些数据类型中的每一个都代表一个 16 位无符号整数；它们只是使用了不同的名称。这看起来可能是一个小问题，但如果你使用每个平台的本地语言为多个设备开发应用程序，为了保持一致性，你需要了解这些差异。否则，你可能会引入平台特定的错误，这些错误非常难以检测和诊断。

## 整数类型

整数数据类型定义为表示整数，可以是**有符号的**（负数、零或正数）或**无符号的**（零或正数）。每种语言都使用自己的标识符和关键字来表示整数类型，因此最容易从内存长度的角度来考虑。就我们的目的而言，我们只将讨论表示 8 位、16 位、32 位和 64 位内存对象的整数类型。 

8 位数据类型，或更常见地称为 **bytes**，是我们将要考察的最小数据类型。如果你已经复习了二进制数学，你会知道一个 8 位的内存块可以表示 2⁸，或 256 个值。有符号字节可以在 -128 到 127，或 -(2⁷) 到 (2⁷) - 1 的范围内变化。无符号字节可以在 0 到 255，或 0 到 (2⁸) -1 的范围内变化。

16 位数据类型通常被称为 **short**，尽管这并不总是如此。这些类型可以表示 2¹⁶ 个值。有符号短整型可以在 -(2¹⁵) 到 (2¹⁵) - 1 的范围内变化。无符号短整型可以在 0 到 (2¹⁶) - 1 的范围内变化。

32 位数据类型最常见的是整数，尽管有时也被称为 **long**。整型可以表示 2³² 个值。有符号整数可以在 -2³¹ 到 2³¹ - 1 的范围内变化。无符号整数可以在 0 到 (2³²) - 1 的范围内变化。

最后，64 位数据类型最常见的是 long，尽管 Objective-C 将其识别为 **long** **long**。长整型可以表示 2⁶⁴ 个值。有符号长整型可以在 -(2⁶³) 到 (2⁶³) - 1 的范围内变化。无符号长整型可以在 0 到 (2⁶³) - 1 的范围内变化。

### 注意

注意，这些值恰好在我们将要使用的四种语言中是一致的，但某些语言可能会引入轻微的变化。熟悉一种语言的数字标识符的细节总是一个好主意。这尤其重要，如果你预期将处理涉及标识符极端值的情况。

**C#**

C# 将整数类型称为 **整型**。该语言提供了两种创建 8 位类型的机制，`byte` 和 `sbyte`。这两个容器可以存储多达 256 个值，无符号字节的范围从 0 到 255。有符号字节支持负值，因此范围从 -128 到 127：

```py
    // C# 
    sbyte minSbyte = -128; 
    byte maxByte = 255; 
    Console.WriteLine("minSbyte: {0}", minSbyte); 
    Console.WriteLine("maxByte: {0}", maxByte); 

    /* 
      Output 
      minSbyte: -128 
      maxByte: 255 
    */ 

```

有趣的是，C# 对于较长的位标识符会反转其模式。它不是像 `sbyte` 一样在有符号标识符前加 `s`，而是将无符号标识符前加 `u`。因此，对于 16 位、32 位和 64 位标识符，我们有 `short`、`ushort`；`int`、`uint`；`long` 和 `ulong` 分别：

```py
    short minShort = -32768; 
    ushort maxUShort = 65535; 
    Console.WriteLine("minShort: {0}", minShort); 
    Console.WriteLine("maxUShort: {0}", maxUShort); 

    int minInt = -2147483648; 
    uint maxUint = 4294967295; 
    Console.WriteLine("minInt: {0}", minInt); 
    Console.WriteLine("maxUint: {0}", maxUint); 

    long minLong = -9223372036854775808; 
    ulong maxUlong = 18446744073709551615;  
    Console.WriteLine("minLong: {0}", minLong); 
    Console.WriteLine("maxUlong: {0}", maxUlong); 

    /* 
      Output 
      minShort: -32768 
      maxUShort: 65535 
      minInt: -2147483648 
      maxUint: 4294967295 
      minLong: -9223372036854775808 
      maxUlong: 18446744073709551615 
    */ 

```

**Java**

Java 将整数类型作为其原始数据类型的一部分。Java 语言只为 8 位存储提供了一个构造，也称为 `byte`。它是一个有符号数据类型，因此它将表示从 -127 到 128 的值。Java 还提供了一个名为 `Byte` 的包装类，它包装原始值并提供对可解析字符串或文本的额外构造支持，这些字符串或文本可以转换为数值，例如文本 42。这种模式在 16 位、32 位和 64 位数据类型中重复：

```py
    //Java 
    byte myByte = -128; 
    byte bigByte = 127; 

    Byte minByte = new Byte(myByte); 
    Byte maxByte = new Byte("128"); 
    System.out.println(minByte);  
    System.out.println(bigByte); 
    System.out.println(maxByte); 

    /* 
      Output 
      -128 
      127 
      127 
    */ 

```

Java 与 C#共享所有整数数据类型的标识符，这意味着它也提供了`byte`、`short`、`int`和`long`标识符，用于 8 位、16 位、32 位和 64 位类型。Java 中的模式有一个例外是`char`标识符，它用于无符号 16 位数据类型。然而，需要注意的是，`char`数据类型通常仅用于 ASCII 字符赋值，而不是实际整数值：

```py
    //Short class 
    Short minShort = new Short(myShort); 
    Short maxShort = new Short("32767"); 
    System.out.println(minShort);  
    System.out.println(bigShort); 
    System.out.println(maxShort); 

    int myInt = -2147483648; 
    int bigInt = 2147483647; 

    //Integer class 
    Integer minInt = new Integer(myInt); 
    Integer maxInt = new Integer("2147483647"); 
    System.out.println(minInt);  
    System.out.println(bigInt); 
    System.out.println(maxInt); 

    long myLong = -9223372036854775808L; 
    long bigLong = 9223372036854775807L; 

    //Long class 
    Long minLong = new Long(myLong); 
    Long maxLong = new Long("9223372036854775807"); 
    System.out.println(minLong);  
    System.out.println(bigLong); 
    System.out.println(maxLong); 

    /* 
      Output 
      -32768 
      32767 
      32767 
      -2147483648 
      2147483647 
      2147483647 
      -9223372036854775808 
      9223372036854775807 
      9223372036854775807 
   */ 

```

在前面的代码中，请注意`int`类型和`Integer`类。与其他原始包装类不同，`Integer`与其支持的标识符名称不同。

此外，请注意`long`类型及其指定的值。在每种情况下，值都有后缀`L`。这是 Java 中`long`字面量的要求，因为编译器将所有数字字面量解释为 32 位整数。如果你想明确指定你的字面量大于 32 位，你必须附加后缀`L`。然而，当将字符串值传递给`Long`类构造函数时，这不是一个要求：

```py
    Long maxLong = new Long("9223372036854775807"); 

```

**Objective-C**

对于 8 位数据，Objective-C 提供了带符号和无符号格式的`char`数据类型。与其他语言一样，带符号的数据类型范围从-127 到 128，而无符号数据类型的范围从 0 到 255。开发者还有选择使用 Objective-C 的固定宽度对应类型`int8_t`和`uint8_t`。这种模式在 16 位、32 位和 64 位数据类型中重复。最后，Objective-C 还提供了`NSNumber`类作为每个整数类型的面向对象包装类：

### 注意

`char`或其他整数数据类型标识符与其固定宽度对应类型之间的区别是一个重要的区分。除了总是精确为 1 字节的`char`之外，Objective-C 中的其他每个整数数据类型的大小将根据实现和底层架构而变化。这是因为 Objective-C 基于 C，C 是为与各种底层架构以最高效率工作而设计的。虽然可以在运行时确定整数类型的确切长度，但在编译时，你只能确定`short <= int <= long <= long long`。

这就是**固定宽度整数**派上用场的地方。如果你需要更严格的字节数量控制，`(u)int<n>_t`数据类型允许你表示长度精确为 8 位、16 位、32 位或 64 位的整数。

```py
    //Objective-C 
    char number = -127; 
    unsigned char uNumber = 255; 
    NSLog(@"Signed char number: %hhd", number); 
    NSLog(@"Unsigned char uNumber: %hhu", uNumber); 

    //fixed width 
    int8_t fixedNumber = -127; 
    uint8_t fixedUNumber = 255; 
    NSLog(@"fixedNumber8: %hhd", fixedNumber8); 
    NSLog(@"fixedUNumber8: %hhu", fixedUNumber8); 

    NSNumber *charNumber = [NSNumber numberWithChar:number]; 
    NSLog(@"Char charNumber: %@", [charNumber stringValue]); 

    /*  
      Output 
      Signed char number: -127 
      Unsigned char uNumber: 255 
      fixedNumber8: -127 
      fixedUNumber8: 255 
      Char charNumber: -127 
    */ 

```

在前面的示例中，你可以看到，当在代码中使用`char`数据类型时，你必须指定`unsigned`标识符，例如`unsigned char`。然而，`signed`是默认的，并且可以省略，这意味着`char`类型等同于`signed char`。这种模式适用于 Objective-C 中每个整数数据类型。

Objective-C 中的更大整数类型包括 `short` 用于 16 位，`int` 用于 32 位，以及 `long long` 用于 64 位。每个这些类型都有一个遵循 `(u)int<n>_t` 模式的固定宽度对应类型。`NSNumber` 类中也为每种类型提供了支持方法：

```py
    //Larger Objective-C types 
    short aShort = -32768; 
    unsigned short anUnsignedShort = 65535; 
    NSLog(@"Signed short aShort: %hd", aShort); 
    NSLog(@"Unsigned short anUnsignedShort: %hu", anUnsignedShort); 

    int16_t fixedNumber16 = -32768; 
    uint16_t fixedUNumber16 = 65535; 
    NSLog(@"fixedNumber16: %hd", fixedNumber16); 
    NSLog(@"fixedUNumber16: %hu", fixedUNumber16); 

    NSNumber *shortNumber = [NSNumber numberWithShort:aShort]; 
    NSLog(@"Short shortNumber: %@", [shortNumber stringValue]); 

    int anInt = -2147483648; 
    unsigned int anUnsignedInt = 4294967295; 
    NSLog(@"Signed Int anInt: %d", anInt); 
    NSLog(@"Unsigned Int anUnsignedInt: %u", anUnsignedInt); 

    int32_t fixedNumber32 = -2147483648; 
    uint32_t fixedUNumber32 = 4294967295; 
    NSLog(@"fixedNumber32: %d", fixedNumber32); 
    NSLog(@"fixedUNumber32: %u", fixedUNumber32); 

    NSNumber *intNumber = [NSNumber numberWithInt:anInt]; 
    NSLog(@"Int intNumber: %@", [intNumber stringValue]); 

    long long aLongLong = -9223372036854775808; 
    unsigned long long anUnsignedLongLong = 18446744073709551615; 
    NSLog(@"Signed long long aLongLong: %lld", aLongLong); 
    NSLog(@"Unsigned long long anUnsignedLongLong: %llu", anUnsignedLongLong); 

    int64_t fixedNumber64 = -9223372036854775808; 
    uint64_t fixedUNumber64 = 18446744073709551615; 
    NSLog(@"fixedNumber64: %lld", fixedNumber64); 
    NSLog(@"fixedUNumber64: %llu", fixedUNumber64); 

    NSNumber *longlongNumber = [NSNumber numberWithLongLong:aLongLong]; 
    NSLog(@"Long long longlongNumber: %@", [longlongNumber stringValue]); 

    /*  
      Output 
      Signed short aShort: -32768 
      Unsigned short anUnsignedShort: 65535 
      fixedNumber16: -32768 
      fixedUNumber16: 65535 
      Short shortNumber: -32768 
      Signed Int anInt: -2147483648 
      Unsigned Int anUnsignedInt: 4294967295 
      fixedNumber32: -2147483648 
      fixedUNumber32: 4294967295 
      Int intNumber: -2147483648 
      Signed long long aLongLong: -9223372036854775808 
      Unsigned long long anUnsignedLongLong: 18446744073709551615 
      fixedNumber64: -9223372036854775808 
      fixedUNumber64: 18446744073709551615 
      Long long longlongNumber: -9223372036854775808 
    */ 

```

**Swift**

Swift 语言与其他语言类似，它为有符号和无符号整数提供了单独的标识符，例如 `Int8` 和 `UInt8`。这种模式适用于 Swift 中的每个整数数据类型，使其在记住哪个标识符适用于哪种类型方面可能是最简单的语言：

```py
    //Swift 
    var int8 : Int8 = -127 
    var uint8 : UInt8 = 255 
    print("int8: \(int8)") 
    print("uint8: \(uint8)") 

    /*  
      Output 
      int8: -127  
      uint8: 255 
    */ 

```

在前面的例子中，我已明确使用 `:Int8` 和 `: UInt8` 标识符来声明数据类型以演示显式声明。在 Swift 中，也可以省略这些标识符，并允许 Swift 在运行时动态推断类型：

```py
    //Larger Swift types 
    var int16 : Int16 = -32768 
    var uint16 : UInt16 = 65535 
    print("int16: \(int16)") 
    print("uint16: \(uint16)") 

    var int32 : Int32 = -2147483648 
    var uint32 : UInt32 = 4294967295 
    print("int32: \(int32)") 
    print("uint32: \(uint32)") 

    var int64 : Int64 = -9223372036854775808 
    var uint64 : UInt64 = 18446744073709551615 
    print("int64: \(int64)") 
    print("uint64: \(uint64)") 

    /*  
      Output 
      int16: -32768 
      uint16: 65535 
      int32: -2147483648 
      uint32: 4294967295 
      int64: -9223372036854775808 
      uint64: 18446744073709551615 
    */ 

```

我为什么需要了解这些？

你可能会问，我为什么需要了解这些数据类型的细节？难道我不能只声明一个 `int` 对象或类似的标识符，然后继续编写有趣的代码吗？现代计算机甚至移动设备提供了几乎无限的资源，所以这并不是什么大问题，对吧？

嗯，并不完全是这样。确实，在你的日常编程经验中的许多情况下，任何整数类型都适用。例如，在某个给定的一天，通过西弗吉尼亚州州立机动车辆管理局（**DMV**）办公室发行的牌照列表进行循环，可能会得到几十到几百个结果。你可以使用 `short` 或 `long long` 来控制 `for` 循环的迭代次数。无论如何，循环对你的系统性能的影响都非常小。

然而，如果你处理的数据集中每个离散的结果都可以适应 16 位类型，但你选择了一个 32 位标识符仅仅因为你习惯了这样做？你刚刚将管理该集合所需的内存量翻倍了。对于 100 或甚至 10 万个结果来说，这个决定可能无关紧要。然而，当你开始处理非常大的数据集时，有数十万甚至数百万个离散结果时，这样的设计决策可能会对系统性能产生巨大影响。

## 单精度浮点数

**单精度浮点数**，或更常见地称为 **floats**，是 32 位浮点容器，可以存储比整数类型具有更高精度的值，通常为六到七位有效数字。许多语言使用 `float` 关键字或标识符来表示单精度浮点值，我们讨论的四种语言也是如此。

你应该意识到浮点数会受到舍入误差的影响，因为它们不能精确地表示十进制数。浮点类型的算术是一个相当复杂的话题，其细节对于任何给定日子的大多数开发者来说并不相关。然而，熟悉每种语言中底层科学以及实现的细节仍然是一个好的实践。

### 注意

由于我绝不是该领域的专家，这次讨论只会触及这些类型背后的科学表面，我们甚至不会开始涉及算术。然而，在这个领域确实有真正的专家，我强烈建议你查看本章末尾的 *附加资源* 部分中列出的他们的一些作品。

**C#**

在 C# 中，`float` 关键字标识 32 位浮点数。C# 的 `float` 数据类型具有约 -3.4 × 10³⁸ 到 +3.4 × 10³⁸ 的范围和 6 位有效数字的精度：

```py
    //C# 
    float piFloat = 3.14159265358979323846264338327f; 
    Console.WriteLine("piFloat: {0}", piFloat); 

    /*  
      Output 
      piFloat: 3.141593 
    */ 

```

当你检查前面的代码时，你会注意到 `float` 值赋值带有 `f` 后缀。这是因为，与其他基于 C 的语言一样，C# 默认将赋值右侧的实数文字视为 **double**（稍后讨论）。如果你在赋值中省略 `f` 或 `F` 后缀，你将收到编译错误，因为你正在尝试将双精度值赋给单精度类型。

此外，请注意最后一位的舍入误差。我们用 30 位有效数字表示的 π 填充了 `piFloat` 对象。然而，`float` 只能保留 6 位有效数字，因此软件将之后的数字四舍五入。当 π 计算到 6 位有效数字时，我们得到 3.141592，但由于这个限制，我们的 `float` 值现在是 3.141593。

**Java**

与 C# 一样，Java 使用 **float** 标识符表示浮点数。在 Java 中，`float` 的近似范围为 -3.4 × 10³⁸ 到 +3.4 × 10³⁸，并且具有 6 或 7 位有效数字的精度：

```py
    //Java 
    float piFloat = 3.141592653589793238462643383279f; 
    System.out.println(piFloat);  

    /*  
      Output 
      3.1415927 
    */ 

```

当你检查前面的代码时，你会注意到浮点数值赋值带有 `f` 后缀。这是因为，与其他基于 C 的语言一样，Java 默认将赋值右侧的实数文字视为 `double`。如果你在赋值中省略 `f` 或 `F` 后缀，你将收到编译错误，因为你正在尝试将双精度值赋给单精度类型。

**Objective-C**

Objective-C 使用 `float` 标识符表示浮点数。在 Objective-C 中，`float` 的近似范围为 -3.4 × 10³⁸ 到 +3.4 × 10³⁸，并且具有 6 位有效数字的精度：

```py
    //Objective-C 
    float piFloat = 3.14159265358979323846264338327f; 
    NSLog(@"piFloat: %f", piFloat); 

    NSNumber *floatNumber = [NSNumber numberWithFloat:piFloat]; 
    NSLog(@"floatNumber: %@", [floatNumber stringValue]); 

    /*  
      Output 
      piFloat: 3.141593 
      floatNumber: 3.141593 
    */ 

```

当你检查前面的代码时，你会注意到 float 值赋值有 `f` 后缀。这是因为，像其他基于 C 的语言一样，Swift 默认将赋值右侧的实数字面量视为 double。如果你在赋值时省略 `f` 或 `F` 后缀，你将收到编译错误，因为你正在尝试将双精度值赋给单精度类型。

此外，请注意最后一位的舍入误差。我们用 pi 以 30 位有效数字的形式填充了 `piFloat` 对象，但 float 只能保留六位有效数字，因此软件将之后的数字都四舍五入。当 pi 以六位有效数字计算时，我们得到 3.141592，但我们的 float 值现在变成了 3.141593，这是由于这种限制。

**Swift**

Swift 使用 `float` 标识符表示浮点数。在 Swift 中，`float` 的近似范围为 -3.4 × 10³⁸ 到 +3.4 × 10³⁸，并且具有六位有效数字的精度：

```py
    //Swift 
    var floatValue : Float = 3.141592653589793238462643383279 
    print("floatValue: \(floatValue)") 

    /* 
      Output 
      floatValue: 3.141593 
    */ 

```

当你检查前面的代码时，你会注意到 float 值赋值有 `f` 后缀。这是因为，像其他基于 C 的语言一样，Swift 默认将赋值右侧的实数字面量视为 double。如果你在赋值时省略 `f` 或 `F` 后缀，你将收到编译错误，因为你正在尝试将双精度值赋给单精度类型。

此外，请注意最后一位的舍入误差。我们用 pi 以 30 位有效数字的形式填充了 `floatValue` 对象，但 float 只能保留六位有效数字，因此软件将之后的数字都四舍五入。当 pi 以六位有效数字计算时，我们得到 3.141592，但我们的 float 值现在变成了 3.141593，这是由于这种限制。

## 双精度浮点

**双精度浮点数**，或更常见地称为 **doubles**，是 64 位浮点值，允许存储比整数类型具有更高的精度，通常为 15 位有效数字。许多语言使用 double 标识符表示双精度浮点值，我们讨论的四种语言也是如此。

### 注意

在大多数情况下，选择`float`而不是`double`通常不会有什么影响，除非内存空间是一个考虑因素，在这种情况下，你将尽可能选择`float`。许多人认为在大多数情况下`float`比`double`性能更好，一般来说，这是正确的。然而，还有其他情况下`double`会比`float`性能更好。现实是每种类型的效率都会根据具体案例而变化，这些标准太多，无法在本讨论的上下文中详细说明。因此，如果你的特定应用程序确实需要达到顶峰效率，你应该仔细研究需求和环境因素，并决定最适合你情况的选择。否则，只需使用任何能完成工作的容器，然后继续前进。

**C#**

在 C#中，`double`关键字标识 64 位浮点值。C#的`double`具有大约的范围为±5.0 × 10^(−324)到±1.7 × 10³⁰⁸，并且精度为 14 或 15 位有效数字：

```py
    //C# 
    double piDouble = 3.14159265358979323846264338327; 
    double wholeDouble = 3d; 
    Console.WriteLine("piDouble: {0}", piDouble); 
    Console.WriteLine("wholeDouble: {0}", wholeDouble); 

    /*  
      Output 
      piDouble: 3.14159265358979 
      wholeDouble: 3 
    */ 

```

当你检查前面的代码时，你会注意到`wholeDouble`值赋值有`d`后缀。这是因为，像其他基于 C 的语言一样，C#默认将赋值右侧的实数字面量视为整数。如果你在赋值时省略`d`或`D`后缀，你将收到编译错误，因为你试图将一个整数值赋给双精度浮点类型。

此外，请注意最后一位的舍入误差。我们使用π到 30 位有效数字来填充`piDouble`对象，但`double`只能保留 14 位有效数字，因此软件将之后的数字四舍五入。当π计算到 15 位有效数字时，我们得到 3.141592653589793，但由于这个限制，我们的`float`值现在是 3.14159265358979。

**Java**

在 Java 中，`double`关键字标识 64 位浮点值。Java 的`double`具有大约的范围为±4.9 × 10^(−324)到±1.8 × 10³⁰⁸和 15 或 16 位有效数字的精度：

```py
    double piDouble = 3.141592653589793238462643383279; 
    System.out.println(piDouble); 

    /*  
      Output 
      3.141592653589793 
    */ 

```

当你检查前面的代码时，请注意最后一位的舍入误差。我们使用π到 30 位有效数字来填充`piDouble`对象，但`double`只能保留 15 位有效数字，因此软件将之后的数字四舍五入。当π计算到 15 位有效数字时，我们得到 3.1415926535897932，但由于这个限制，我们的`float`值现在是 3.141592653589793。

**Objective-C**

Objective-C 也使用`double`标识符表示 64 位浮点值。Objective-C 的`double`具有大约的范围为 2.3E^(-308)到 1.7E³⁰⁸和 15 位有效数字的精度。Objective-C 通过提供称为**long double**的更精确的`double`版本，将精度提升了一步。`long double`标识符用于 80 位存储容器，其范围为 3.4E^(-4932)到 1.1E⁴⁹³²和 19 位有效数字的精度：

```py
    //Objective-C 
    double piDouble = 3.14159265358979323846264338327; 
    NSLog(@"piDouble: %.15f", piDouble); 

    NSNumber *doubleNumber = [NSNumber numberWithDouble:piDouble]; 
    NSLog(@"doubleNumber: %@", [doubleNumber stringValue]); 

    /* 
      Output 
      piDouble: 3.141592653589793 
      doubleNumber: 3.141592653589793 
    */ 

```

在我们前面的示例中，请注意最后一位的舍入误差。我们使用 pi 的 30 位有效数字填充了 `piDouble` 对象，但 double 只能保留 15 位有效数字，因此软件将之后的数字四舍五入。当 pi 计算到 15 位有效数字时，我们得到 3.1415926535897932，但由于这个限制，我们的 float 值现在是 3.141592653589793。

**Swift**

Swift 使用 `double` 标识符表示 64 位浮点值。在 Swift 中，double 的近似范围是 2.3E^(-308) 到 1.7E³⁰⁸，精度为 15 位有效数字。请注意，根据 Apple 对 Swift 的文档，当 `float` 或 `double` 类型都适用时，推荐使用 double：

```py
    //Swift 
    var doubleValue : Double = 3.141592653589793238462643383279 
    print("doubleValue: \(doubleValue)") 

    /* 
      Output 
      doubleValue: 3.14159265358979 
    */ 

```

在我们前面的示例中，请注意最后一位的舍入误差。我们使用 pi 的 30 位有效数字填充了 `doubleValue` 对象，但 double 只能保留 15 位有效数字，因此软件将之后的数字四舍五入。当 pi 计算到 15 位有效数字时，我们得到 3.141592653589793，但由于这个限制，我们的 `float` 值现在是 3.141592653589793。

## 货币

由于浮点算术固有的不精确性，这是基于它们基于二进制算术的事实，浮点数和 double 无法准确表示我们用于货币的十进制倍数。将货币表示为 `float` 或 `double` 最初可能看起来是个好主意，因为软件会四舍五入你的算术中的微小误差。然而，当你开始在这些不精确的结果上执行更多和更复杂的算术运算时，你的精度误差将开始累积，并导致严重的不准确性和难以追踪的漏洞。这使得 float 和 double 数据类型在需要完美精度（10 的倍数）的货币处理中不足。幸运的是，我们讨论的每种语言都提供了一种处理货币以及需要高精度十进制值和计算的其它算术问题的机制。

**C#**

C# 使用 `decimal` 关键字来表示精确的浮点值。在 C# 中，`decimal` 的范围是 ±1.0 x 10^(-28) 到 ±7.9 x 10²⁸，精度为 28 或 29 位有效数字：

```py
    var decimalValue = NSDecimalNumber.init(string:"3.141592653589793238462643383279") 
    print("decimalValue \(decimalValue)") 

    /* 
      Output 
      piDecimal: 3.1415926535897932384626433833 
    */ 

```

在前面的示例中，请注意我们使用 pi 的 30 位有效数字填充了 `decimalValue` 对象，但框架将其四舍五入到 28 位有效数字。

**Java**

Java 以 `BigDecimal` 类的形式提供了一个面向对象的解决方案来解决货币问题：

```py
    BigDecimal piDecimal = new BigDecimal("3.141592653589793238462643383279"); 
    System.out.println(piDecimal); 

    /* 
      Output 
      3.141592653589793238462643383279 
    */ 

```

在前面的示例中，我们使用一个接受字符串表示的十进制值作为参数的构造函数初始化 `BigDecimal` 类。当程序运行时，输出证明 `BigDecimal` 类没有丢失任何我们预期的精度，返回了 30 位有效数字的 pi。

**Objective-C**

Objective-C 也以 `NSDecimalNumber` 类的形式提供了一个面向对象的解决方案来解决货币问题：

```py
    //Objective-C 
    NSDecimalNumber *piDecimalNumber = [[NSDecimalNumber alloc] initWithDouble:3.14159265358979323846264338327]; 
    NSLog(@"piDecimalNumber: %@", [piDecimalNumber stringValue]); 

    /* 
      Output 
      piDecimalNumber: 3.141592653589793792 
    */ 

```

**Swift**

Swift 还提供了一个面向对象的解决方案来解决货币问题，并且它与 Objective-C 中使用的同一个类相同，即 `NSDecimalNumber` 类。Swift 版本初始化略有不同，但与 Objective-C 的对应版本具有相同的功能：

```py
    var decimalValue = NSDecimalNumber.init(string:"3.141592653589793238462643383279") 
    print("decimalValue \(decimalValue)") 

    /* 
      Output 
      decimalValue 3.141592653589793238462643383279 
    */ 

```

注意，在 Objective-C 和 Swift 的示例中，精度都保留到 30 位有效数字，这证明了 `NSDecimalNumber` 类在处理货币和其他十进制值方面是优越的。

### 小贴士

在充分披露的精神下，使用这些自定义类型有一个简单且可以说是更优雅的替代方案。你可以直接使用 `int` 或 `long` 进行货币计算，并按分而不是按美元计数：

//C# long total = 316;

//$3.16

## 类型转换

在计算机科学领域，**类型转换**或**类型转换**意味着将一个对象或数据类型的实例转换为另一个。例如，假设你调用了一个返回整数值的方法，但你需要使用该值在另一个需要将 `long` 值作为输入参数的方法中。由于整数值根据定义存在于允许的 `long` 值范围内，因此 `int` 值可以被重新定义为 `long`。

这种转换可以通过隐式转换（有时称为**强制转换**）或显式转换（通常称为**类型转换**）来完成。要完全理解类型转换，我们还需要了解**静态**和**动态**语言之间的区别。

### 静态类型语言与动态类型语言

静态类型语言将在编译时执行其**类型检查**。这意味着，当你尝试构建你的解决方案时，编译器将验证并强制执行应用于应用程序中类型的每个约束。如果它们没有被强制执行，你将收到错误，并且应用程序将无法构建。C#、Java 和 Swift 都是静态类型语言。

动态类型语言，另一方面，在运行时进行大多数或所有的类型检查。这意味着应用程序可能构建得很好，但如果开发者没有在编写代码时小心谨慎，那么在应用程序实际运行时可能会遇到问题。Objective-C 是一种动态类型语言，因为它使用静态类型对象和动态类型对象的混合。本章前面讨论的用于数值的普通 C 对象都是静态类型对象的例子，而 Objective-C 类 `NSNumber` 和 `NSDecimalNumber` 都是动态类型对象的例子。以下是一个 Objective-C 代码示例：

```py
    double myDouble = @"chicken"; 
    NSNumber *myNumber = @"salad"; 

```

编译器将在第一行抛出错误，指出 `初始化 'double' 时使用了一个不兼容类型的表达式 'NSString *'`。这是因为 `double` 是一个普通的 C 对象，它是静态类型的。编译器在我们甚至开始构建之前就知道如何处理这个静态类型的对象，所以你的构建将失败。

然而，编译器只会在第二行抛出警告，指出`初始化 'NSNumber *' 的指针类型不兼容，表达式类型为 'NSString *'`。这是因为`NSNumber`是 Objective-C 类，它是动态类型的。编译器足够智能，能够捕捉到您的错误，但它将允许构建成功（除非您已在构建设置中指示编译器将警告视为错误）。

### 小贴士

虽然在前面的示例中，运行时即将发生的崩溃是明显的，但有些情况下，即使有警告，您的应用程序也能正常工作。然而，无论您使用的是哪种编程语言，始终在继续编写新代码之前一致地清理代码警告都是一个好主意。这有助于保持代码整洁，并避免任何难以诊断的运行时错误。

在那些不适宜立即处理警告的罕见情况下，您应该清楚地记录代码并解释警告的来源，以便其他开发者能够理解您的推理。作为最后的手段，您可以利用宏或预处理器（预编译器）指令，这些指令可以逐行抑制警告。

### 隐式和显式转换

**隐式转换**在您的源代码中不需要任何特殊的语法。这使得隐式转换变得相对方便。以下是一个 C#中的代码示例：

```py
    int a = 10; 
    double b = a++; 

```

在这种情况下，由于`a`可以被定义为`int`和`double`两种类型，因此转换为`double`类型是完全可接受的，因为我们已经手动定义了这两种类型。然而，由于隐式转换不一定手动定义它们的类型，编译器无法始终确定哪些约束适用于转换，因此无法在编译时检查这些约束。这使得隐式转换也具有一定的危险性。以下是一个同样在 C#中的代码示例：

```py
    double x = "54"; 

```

这是一种隐式转换，因为你没有告诉编译器如何处理字符串值。在这种情况下，当尝试构建应用程序时，转换将失败，编译器将抛出错误，指出`无法隐式转换类型 'string' 到 'double'`。现在，考虑这个示例的显式转换版本：

```py
    double x = double.Parse("42"); 
    Console.WriteLine("40 + 2 = {0}", x); 

    /* 
      Output 
      40 + 2 = 42 
    */ 

```

这种转换是显式的，因此是类型安全的，假设字符串值是*可解析的*。

### 扩展和收缩

在两种类型之间进行转换时，一个重要的考虑因素是变化的结果是否在目标数据类型的范围内。如果您的源数据类型支持的字节比目标数据类型多，则这种转换被认为是**收缩转换**。

窄化转换要么是无法证明始终成功的转换，要么是已知可能丢失信息的转换。例如，从浮点数到整数的转换将导致信息丢失（在这种情况下是精度），因为结果将被四舍五入到最接近的整数。在大多数静态类型语言中，窄化转换不能隐式执行。以下是一个例子，借鉴了本章前面提到的 C# 单精度和双精度示例：

```py
    //C# 
    piFloat = piDouble; 

```

在这个例子中，编译器将抛出一个错误，指出“无法隐式转换类型 'double' 到 'float'”。并且存在显式转换（你是否遗漏了一个转换？）。编译器将其视为窄化转换，并将精度损失视为错误。错误消息本身很有帮助，并建议显式转换作为我们问题的潜在解决方案：

```py
    //C# 
    piFloat = (float)piDouble;   

```

我们现在已经明确地将双精度值 `piDouble` 转换为 `float` 类型，编译器不再关心精度损失的问题。

如果您的源数据类型支持的字节少于您的目标数据类型，则该转换被认为是**宽化转换**。宽化转换将保留源对象的值，但可能会以某种方式更改其表示。大多数静态类型语言将允许隐式宽化转换。让我们再次借鉴我们之前的 C# 示例：

```py
    //C# 
    piDouble = piFloat; 

```

在这个例子中，编译器对隐式转换完全满意，应用将构建。让我们进一步扩展这个例子：

```py
    //C# 
    piDouble = (double)piFloat; 

```

这种显式转换提高了可读性，但以任何方式都没有改变语句的本质。编译器也认为这种格式完全可接受，尽管它可能有些冗长。除了提高可读性之外，在宽化转换时显式转换对您的应用程序没有任何增加。因此，如果您想在宽化转换时使用显式转换，这是一个个人偏好的问题。

# 布尔数据类型

布尔数据类型旨在表示二进制值，通常用 `1` 和 `0`、`true` 和 `false` 或甚至 `YES` 和 `NO` 表示。布尔类型用于表示基于布尔代数的真值逻辑。这仅仅是一种说法，即布尔值用于条件语句，如 `if` 或 `while`，以评估逻辑或有条件地重复执行。

等于操作包括任何比较两个实体值值的操作。等价操作符包括：

+   `==` 表示等于

+   `!=` 表示不等于

关系操作包括任何测试两个实体之间关系的操作。关系操作符包括：

+   `>` 表示大于

+   `>=` 表示大于或等于

+   `<` 表示小于

+   `<=` 表示小于或等于

逻辑运算包括程序中评估和操作布尔值的任何操作。主要有三个逻辑运算符，即`AND`、`OR`和`NOT`。另一个稍微不太常用的运算符是**异或**，或称为 XOR 运算符。所有布尔函数和语句都可以使用这四个基本运算符构建。

AND 运算符是最为严格的比较运算符。给定两个布尔变量 A 和 B，AND 运算符只有在 A 和 B 都为`true`时才会返回`true`。布尔变量通常使用称为**真值表**的工具进行可视化。以下为 AND 运算符的真值表：

| **A** | **B** | **A ^ B** |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

此表展示了 AND 运算符。在评估条件语句时，0 被视为`false`，而任何其他值都视为`true`。只有当 A 和 B 的值都为`true`时，A 与 B 的运算结果才为`true`。

OR 运算符是包含运算符。给定两个布尔变量 A 和 B，OR 运算符在 A 或 B 为`true`时返回`true`，包括 A 和 B 都为`true`的情况。以下为 OR 运算符的真值表：

| **A** | **B** | **A v B** |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

接下来，NOT A 运算符在 A 为`false`时为`true`，在 A 为`true`时为`false`。以下为 NOT 运算符的真值表：

| **A** | **!A** |
| --- | --- |
| 0 | 1 |
| 1 | 0 |

最后，XOR 运算符在 A 或 B 为`true`但不同时为`true`时为`true`。另一种说法是，XOR 在 A 和 B 不同时为`true`。在许多情况下，以这种方式评估表达式非常有用，因此大多数计算机架构都包括它。以下为 XOR 运算符的真值表：

| **A** | **B** | **A XOR B** |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

## 运算符优先级

就像算术一样，比较和布尔运算也有**运算符优先级**。这意味着架构将赋予一个运算符比另一个运算符更高的优先级。一般来说，所有语言的布尔运算顺序如下：

+   括号

+   关系运算符

+   等式运算符

+   位运算符（未讨论）

+   NOT

+   AND

+   OR

+   XOR

+   三元运算符

+   赋值运算符

在处理布尔值时，理解运算符优先级非常重要，因为错误地理解架构将如何评估复杂的逻辑运算会在代码中引入你无法解决的错误。如有疑问，请记住，就像算术中的括号一样，优先级最高，括号内的内容将首先被评估。

## 短路

如你所知，AND 运算符仅在两个操作数都为`true`时返回`true`，而 OR 运算符只要有一个操作数为`true`就会返回`true`。这些特性有时使得仅通过评估其中一个操作数就能确定表达式的结果成为可能。当你的应用程序在确定表达式的整体结果后立即停止评估时，这被称为**短路**。你可能会在代码中使用短路的三种主要原因。

首先，短路可以通过限制代码必须执行的操作数量来提高应用程序的性能。其次，当后续的操作可能基于先前操作数的值生成错误时，短路可以在达到更高风险的操作数之前停止执行。最后，短路可以通过消除嵌套逻辑语句的需要来提高代码的可读性和复杂性。

**C#**

C#使用`bool`关键字作为`System.Boolean`的别名，并存储`true`和`false`值：

```py
    //C# 
    bool a = true; 
    bool b = false; 
    bool c = a; 

    Console.WriteLine("a: {0}", a); 
    Console.WriteLine("b: {0}", b); 
    Console.WriteLine("c: {0}", c); 
    Console.WriteLine("a AND b: {0}", a && b); 
    Console.WriteLine("a OR b: {0}", a || b); 
    Console.WriteLine("NOT a: {0}", !a); 
    Console.WriteLine("NOT b: {0}", !b); 
    Console.WriteLine("a XOR b: {0}", a ^ b); 
    Console.WriteLine("(c OR b) AND a: {0}", (c || b) && a); 

    /* 
      Output 
      a: True 
      b: False 
      c: True 
      a AND b: False 
      a OR b: True 
      NOT a: False 
      NOT b: True 
      a XOR b: True 
      (c OR b) AND a: True 
    */ 

```

**Java**

Java 使用`boolean`关键字表示原始布尔数据类型。Java 还提供了一个`Boolean`包装类来表示相同的原始类型：

```py
    //Java 
    boolean a = true; 
    boolean b = false; 
    boolean c = a; 

    System.out.println("a: " + a); 
    System.out.println("b: " + b); 
    System.out.println("c: " + c); 
    System.out.println("a AND b: " + (a && b)); 
    System.out.println("a OR b: " + (a || b)); 
    System.out.println("NOT a: " + !a); 
    System.out.println("NOT b: " + !b); 
    System.out.println("a XOR b: " + (a ^ b)); 
    System.out.println("(c OR b) AND a: " + ((c || b) && a)); 

    /* 
      Output 
      a: true 
      b: false 
      c: true 
      a AND b: false 
      a OR b: true 
      NOT a: false 
      NOT b: true 
      a XOR b: true 
     (c OR b) AND a: true 
    */ 

```

**Objective-C**

Objective-C 使用`BOOL`标识符来表示布尔值：

```py
    //Objective-C 
    BOOL a = YES; 
    BOOL b = NO; 
    BOOL c = a; 

    NSLog(@"a: %hhd", a); 
    NSLog(@"b: %hhd", b); 
    NSLog(@"c: %hhd", c); 
    NSLog(@"a AND b: %d", a && b); 
    NSLog(@"a OR b: %d", a || b); 
    NSLog(@"NOT a: %d", !a); 
    NSLog(@"NOT b: %d", !b); 
    NSLog(@"a XOR b: %d", a ^ b); 
    NSLog(@"(c OR b) AND a: %d", (c || b) && a); 

    /* 
      Output 
      a: 1 
      b: 0 
      c: 1 
      a AND b: 0 
      a OR b: 1 
      NOT a: 0 
      NOT b: 1 
      a XOR b: 1 
      (c OR b) AND a: 1 
    */ 

```

### 注意

事实上，布尔数据类型给了 Objective-C 另一个证明它比其对手更复杂的机会。该语言没有提供一个标识符或类来表示逻辑值，而是提供了五个。为了简单起见（并且因为我的编辑器不会给我额外的页面），我们在这篇文章中只使用`BOOL`。如果你想了解更多，我鼓励你查看本章末尾的*附加资源*部分。

**Swift**

Swift 使用`Bool`关键字表示原始布尔数据类型：

```py
    //Swift 
    var a : Bool = true 
    var b : Bool = false 
    var c = a 

    print("a: \(a)") 
    print("b: \(b)") 
    print("c: \(c)") 
    print("a AND b: \(a && b)") 
    print("a OR b: \(a || b)") 
    print("NOT a: \(!a)") 
    print("NOT b: \(!b)") 
    print("a XOR b: \(a != b)") 
    print("(c OR b) AND a: \((c || b) && a)") 

    /* 
      Output 
      a: true 
      b: false 
      c: true 
      a AND b: false 
      a OR b: true 
      NOT a: false 
      NOT b: true 
      a XOR b: true 
      (c OR b) AND a: true 
    */ 

```

在前面的例子中，布尔对象`c`并未显式声明为`Bool`，但它被隐式地指定为`Bool`。用 Swift 的话说，在这种情况下数据类型已经被*推断*。此外，请注意 Swift 不提供特定的 XOR 运算符，所以如果你需要这种比较，你应该使用`(a != b)`模式。

### 小贴士

Objective-C nil 值

在 Objective-C 中，值`nil`也评估为`false`。尽管其他语言必须小心处理 NULL 对象，但 Objective-C 在尝试在 nil 对象上执行操作时不会崩溃。从我个人的经验来看，这可能会让在学习 Objective-C 之前先学习了 C#或 Java 的开发者感到有些困惑，因为他们期望未处理的 NULL 对象会导致他们的应用崩溃。然而，Objective-C 开发者通常利用这种行为来获得优势。很多时候，仅仅检查一个对象是否为`nil`在逻辑上就能确认操作是否成功，从而节省了你编写繁琐的逻辑比较。

# 字符串

字符串并不是精确的数据类型，尽管作为开发者，我们经常将它们当作这样。实际上，字符串只是值是文本的对象；在底层，字符串包含一个只读的 `char` 对象的顺序集合。字符串对象的这种只读性质使得字符串 **不可变**，这意味着一旦在内存中创建，对象就不能被更改。

重要的是要理解，更改任何不可变对象，不仅仅是字符串，意味着你的程序实际上是在内存中创建一个新的对象并丢弃旧的一个。这比简单地更改内存地址中的值要复杂得多，需要更多的处理。将两个字符串合并在一起称为 **连接**，这是一个成本更高的过程，因为你是在创建新对象之前先丢弃了两个对象。如果你发现你经常编辑字符串值，或者经常将字符串连接在一起，请注意，你的程序可能没有它本可以做到的那样高效。

在 C#、Java 和 Objective-C 中，字符串是严格不可变的。有趣的是，Swift 文档将字符串称为可变的。然而，行为与 Java 类似，即当字符串被修改时，它会在赋值给另一个对象时被复制。因此，尽管文档说不同，但在 Swift 中字符串实际上也是不可变的。

**C#**

C# 使用字符串关键字来声明字符串类型：

```py
    //C# 
    string one = "One String"; 
    Console.WriteLine("One: {0}", one); 

    String two = "Two String"; 
    Console.WriteLine("Two: {0}", two); 

    String red = "Red String"; 
    Console.WriteLine("Red: {0}", red); 

    String blue = "Blue String"; 
    Console.WriteLine("Blue: {0}", blue); 

    String purple = red + blue; 
    Console.WriteLine("Concatenation: {0}", purple); 

    purple = "Purple String"; 
    Console.WriteLine("Whoops! Mutation: {0}", purple); 

```

**Java**

Java 使用系统类 `String` 来声明字符串类型：

```py
    //Java 
    String one = "One String"; 
    System.out.println("One: " + one); 

    String two = "Two String"; 
    System.out.println("Two: " + two); 

    String red = "Red String"; 
    System.out.println("Red: " + red); 

    String blue = "Blue String"; 
    System.out.println("Blue: " + blue); 

    String purple = red + blue; 
    System.out.println("Concatenation: " + purple); 

    purple = "Purple String"; 
    System.out.println("Whoops! Mutation: " + purple); 

```

**Objective-C**

Objective-C 提供了 `NSString` 类来创建字符串对象：

```py
    //Objective-C 
    NSString *one = @"One String"; 
    NSLog(@"One: %@", one); 

    NSString *two = @"Two String"; 
    NSLog(@"Two: %@", two); 

    NSString *red = @"Red String"; 
    NSLog(@"Red: %@", red); 

    NSString *blue = @"Blue String"; 
    NSLog(@"Blue: %@", blue); 

    NSString *purple = [[NSArray arrayWithObjects:red, blue, nil] componentsJoinedByString:@""]; 
    NSLog(@"Concatenation: %@", purple); 

    purple = @"Purple String"; 
    NSLog(@"Whoops! Mutation: %@", purple); 

```

当你查看 Objective-C 的示例时，你可能会想知道为什么我们需要为创建紫色对象编写那么多额外的代码。这段代码是必要的，因为 Objective-C 并没有提供像我们使用的其他三种语言那样的字符串连接快捷机制。因此，在这种情况下，我选择将两个字符串放入一个数组中，然后调用 `NSArray` 方法 `componentsJoinedByString:`。我也可以选择使用 `NSMutableString` 类，它提供了一个用于连接字符串的方法。然而，由于我们讨论的语言中并没有涉及可变字符串类，所以我选择了不使用这种方法。

**Swift**

Swift 提供了 `String` 类来创建字符串对象：

```py
    //Swift 
    var one : String = "One String" 
    print("One: \(one)") 

    var two : String = "Two String" 
    print("Two: \(two)") 

    var red : String = "Red String" 
    print("Red: \(red)") 

    var blue : String = "Blue String" 
    print("Blue: \(blue)") 

    var purple : String = red + blue 
    print("Concatenation: \(purple)") 

    purple = "Purple String"; 
    print("Whoops! Mutation: \(purple)") 

    /* 
      Output from each string example: 
      One: One String 
      Two: Two String 
      Red: Red String 
      Blue: Blue String 
      Concatenation: Red StringBlue String 
      Whoops! Mutation: Purple String 
    */ 

```

# 摘要

在本章中，你学习了在四种最常见的移动开发语言中，程序员可用的基本数据类型。数值和浮点数据类型的特性和操作既取决于底层架构，也取决于语言的规范。你还学习了如何将对象从一个类型转换为另一个类型，以及转换的类型是如何根据源数据和目标数据类型的大小定义为宽转换或窄转换的。接下来，我们讨论了布尔类型及其在比较器中如何影响程序流程和执行。在这里，我们讨论了运算符的优先级顺序和嵌套操作。你还学习了如何使用短路来提高代码的性能。最后，我们考察了`String`数据类型以及与可变对象一起工作的含义。
