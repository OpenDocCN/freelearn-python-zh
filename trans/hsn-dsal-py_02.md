# Python数据类型和结构

在本章中，我们将更详细地研究Python数据类型。我们已经介绍了两种数据类型，字符串和列表，`str()`和`list()`。然而，这些数据类型是不够的，我们经常需要更专门的数据对象来表示/存储我们的数据。 Python有各种其他标准数据类型，用于存储和管理数据，我们将在本章中讨论。除了内置类型之外，还有几个内部模块，允许我们解决处理数据结构时的常见问题。首先，我们将回顾一些适用于所有数据类型的操作和表达式，并将讨论更多与Python数据类型相关的内容。

本章的目标如下：

+   了解Python 3.7支持的各种重要内置数据类型

+   探索各种高性能替代品的其他附加集合，以替代内置数据类型

# 技术要求

本章中使用的所有代码都在以下GitHub链接中提供：[https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter02](https://github.com/PacktPublishing/Hands-On-Data-Structures-and-Algorithms-with-Python-Second-Edition/tree/master/Chapter02)。

# 内置数据类型

Python数据类型可以分为三类：数字、序列和映射。还有一个表示`Null`或值的缺失的`None`对象。不应忘记其他对象，如类、文件和异常也可以被正确地视为*类型*；但是，它们在这里不会被考虑。

Python中的每个值都有一个数据类型。与许多编程语言不同，在Python中，您不需要显式声明变量的类型。Python在内部跟踪对象类型。

Python内置数据类型概述如下表所示：

| **类别** | **名称** | **描述** |
| --- | --- | --- |
| None | `None` | 它是一个空对象。 |
| 数字 | `int` | 这是一种整数数据类型。 |
|  | `float` | 这种数据类型可以存储浮点数。 |
|  | `complex` | 它存储复数。 |
|  | `bool` | 它是布尔类型，返回`True`或`False`。 |
| 序列 | `str` | 用于存储一串字符。 |
|  | `liXst` | 它可以存储任意对象的列表。 |
|  | `Tuple` | 它可以存储一组任意项目。 |
|  | `range` | 用于创建一系列整数。 |
| 映射 | `dict` | 它是一种以*键/值*对存储数据的字典数据类型。 |
|  | `set` | 它是一个可变的无序唯一项集合。 |
|  | `frozenset` | 它是一个不可变的集合。 |

# None类型

`None`类型是不可变的。它用作`None`来表示值的缺失；它类似于许多编程语言中的`null`，如C和C++。当实际上没有要返回的内容时，对象返回`None`。当`False`布尔表达式时，也会返回`None`。`None`经常用作函数参数的默认值，以检测函数调用是否传递了值。

# 数字类型

数字类型包括整数（`int`），即无限范围的整数，浮点数（`float`），复数（`complex`），由两个浮点数表示，以及布尔值（`bool`）在Python中。 Python提供了允许标准算术运算符（`+`，`-`，`*`和`/`）对它们进行操作的`int`数据类型，类似于其他编程语言。布尔数据类型有两个可能的值，`True`和`False`。这些值分别映射为`1`和`0`。让我们考虑一个例子：

```py
>>> a=4; b=5   # Operator (=) assigns the value to variable
>>>print(a, "is of type", type(a))
4 is of type 
<class 'int'>
>>> 9/5  
1.8
>>>c= b/a  *# division returns a floating point number* *>>>* print(c, "is of type", type(c))
1.25 is of type <class 'float'>
>>> c   # No need to explicitly declare the datatype
1.25
```

变量`a`和`b`是`int`类型，`c`是浮点类型。除法运算符（`/`）始终返回`float`类型；但是，如果希望在除法后获得`int`类型，可以使用地板除法运算符（`//`），它会丢弃任何小数部分，并返回小于或等于`x`的最大整数值。考虑以下例子：

```py
>>> a=4; b=5   
>>>d= b//a
*>>>* print(d, "is of type", type(d))1 is of type <class 'int'>
>>>7/5  # true division
1.4
>>> -7//5  # floor division operator
-2
```

建议读者谨慎使用除法运算符，因为其功能根据Python版本而异。在Python 2中，除法运算符仅返回`integer`，而不是`float`。

指数运算符（`**`）可用于获取数字的幂（例如，`x ** y`），模数运算符（`%`）返回除法的余数（例如，`a% b`返回`a/b`的余数）：

```py
>>> a=7; b=5 
>>> e= b**a  # The operator (**)calculates power 
>>>e
78125
>>>a%b
2
```

复数由两个浮点数表示。它们使用`j`运算符分配，以表示复数的虚部。我们可以通过`f.real`和`f.imag`访问实部和虚部，如下面的代码片段所示。复数通常用于科学计算。Python支持复数的加法，减法，乘法，幂，共轭等，如下所示：

```py
>>> f=3+5j
>>>print(f, "is of type", type(f))(3+5j) is of type <class 'complex'>
>>> f.real
3.0
>>> f.imag
5.0
>>> f*2   # multiplication
(6+10j)
>>> f+3  # addition
(6+5j)
>>> f -1  # subtraction
(2+5j)  
```

在Python中，布尔类型使用真值表示，即`True`和`False`；这类似于`0`和`1`。Python中有一个`bool`类，返回`True`或`False`。布尔值可以与逻辑运算符（如`and`，`or`和`not`）结合使用：

```py
>>>bool(2)
True
>>>bool(-2)
True
>>>bool(0)
False
```

布尔运算返回`True`或`False`。布尔运算按优先级排序，因此如果表达式中出现多个布尔运算，则优先级最高的运算将首先发生。以下表格按优先级降序列出了三个布尔运算符：

| **运算符** | **示例** |
| --- | --- |
| `not x` | 如果`x`为`True`，则返回`False`，如果`x`为`False`，则返回`True`。 |
| `x and y` | 如果`x`和`y`都为`True`，则返回`True`；否则返回`False`。 |
| `x or` `y` | 如果`x`或`y`中有一个为`True`，则返回`True`；否则返回`False`。 |

Python在评估布尔表达式时非常高效，因为它只在需要时评估运算符。例如，如果在表达式`x or y`中`x`为`True`，则无需评估`y`，因为表达式无论如何都是`True`，这就是为什么在Python中不会评估`y`。类似地，在表达式`x and y`中，如果`x`为`False`，解释器将简单地评估`x`并返回`False`，而不会评估`y`。

比较运算符（`<`，`<=`，`>`，`>=`，`==`和`!=`）适用于数字，列表和其他集合对象，并在条件成立时返回`True`。对于集合对象，比较运算符比较元素的数量，等价运算符（`==`）在每个集合对象在结构上等价且每个元素的值相同时返回`True`。让我们看一个例子：

```py
>>>See_boolean = (4 * 3 > 10) and (6 + 5 >= 11)
>>>print(See_boolean)
True
>>>if (See_boolean):
...    print("Boolean expression returned True")
   else:
...  print("Boolean expression returned False")
...

Boolean expression returned True
```

# 表示错误

应该注意的是，浮点数的本机双精度表示会导致一些意外的结果。例如，考虑以下情况：

```py
>>> 1-0.9
0.09999999999999998
>>> 1-0.9==.1
False
```

这是因为大多数十进制小数无法准确表示为二进制小数，这是大多数底层硬件表示浮点数的方式。对于可能存在此问题的算法或应用程序，Python提供了一个decimal模块。该模块允许精确表示十进制数，并便于更好地控制属性，如舍入行为，有效数字的数量和精度。它定义了两个对象，一个表示十进制数的`Decimal`类型，另一个表示各种计算参数的`Context`类型，如精度，舍入和错误处理。其用法示例如下：

```py
>>> import decimal
>>> x=decimal.Decimal(3.14)
>>> y=decimal.Decimal(2.74)
>>> x*y
Decimal('8.603600000000001010036498883')
>>> decimal.getcontext().prec=4
>>> x*y
Decimal('8.604')
```

在这里，我们创建了一个全局上下文，并将精度设置为`4`。`Decimal`对象可以被视为`int`或`float`一样对待。它们可以进行相同的数学运算，并且可以用作字典键，放置在集合中等等。此外，`Decimal`对象还有几种数学运算的方法，如自然指数`x.exp()`，自然对数`x.ln()`和以10为底的对数`x.log10()`。

Python还有一个`fractions`模块，实现了有理数类型。以下示例展示了创建分数的几种方法：

```py
>>> import fractions
>>> fractions.Fraction(3,4)
Fraction(3, 4)
>>> fractions.Fraction(0.5)
Fraction(1, 2)
>>> fractions.Fraction("0.25") 
Fraction(1, 4)
```

在这里还值得一提的是NumPy扩展。它具有数学对象的类型，如数组、向量和矩阵，以及线性代数、傅里叶变换、特征向量、逻辑操作等功能。

# 成员资格、身份和逻辑操作

成员资格运算符（`in`和`not in`）用于测试序列中的变量，如列表或字符串，并执行您所期望的操作；如果在`y`中找到了`x`变量，则`x in y`返回`True`。`is`运算符比较对象标识。例如，以下代码片段展示了对比等价性和对象标识：

```py
>>> x=[1,2,3]
>>> y=[1,2,3]
>>> x==y  # test equivalence 
True
>>> x is y   # test object identity
False
>>> x=y   # assignment
>>> x is y
True
```

# 序列

序列是由非负整数索引的对象的有序集合。序列包括`string`、`list`、`tuple`和`range`对象。列表和元组是任意对象的序列，而字符串是字符的序列。然而，`string`、`tuple`和`range`对象是不可变的，而`list`对象是可变的。所有序列类型都有许多共同的操作。请注意，对于不可变类型，任何操作都只会返回一个值，而不会实际更改该值。

对于所有序列，索引和切片操作适用于前一章节中描述的方式。`string`和`list`数据类型在[第1章](2818f56c-fbcf-422f-83dc-16cbdbd8b5bf.xhtml)中有详细讨论，*Python对象、类型和表达式*。在这里，我们介绍了一些对所有序列类型（`string`、`list`、`tuple`和`range`对象）都通用的重要方法和操作。

所有序列都有以下方法：

| **方法** | **描述** |
| --- | --- |
| `len(s)` | 返回`s`中元素的数量。 |
| `min(s,[,default=obj, key=func])` | 返回`s`中的最小值（对于字符串来说是按字母顺序）。 |
| `max(s,[,default=obj, key=func])` | 返回`s`中的最大值（对于字符串来说是按字母顺序）。 |
| `sum(s,[,start=0])` | 返回元素的和（如果`s`不是数字，则返回`TypeError`）。 |
| `all(s)` | 如果`s`中所有元素都为`True`（即不为`0`、`False`或`Null`），则返回`True`。 |
| `any(s)` | 检查`s`中是否有任何项为`True`。 |

此外，所有序列都支持以下操作：

| **操作** | **描述** |
| --- | --- |
| `s+r` | 连接两个相同类型的序列。 |
| `s*n` | 创建`n`个`s`的副本，其中`n`是整数。 |
| `v1,v2...,vn=s` | 从`s`中解包`n`个变量到`v1`、`v2`等。 |
| `s[i]` | 索引返回`s`的第`i`个元素。 |
| `s[i:j:stride]` | 切片返回`i`和`j`之间的元素，可选的步长。 |
| `x in s` | 如果`s`中存在`x`元素，则返回`True`。 |
| `x not in s` | 如果`s`中不存在`x`元素，则返回`True`。 |

让我们考虑一个示例代码片段，实现了对`list`数据类型的一些前述操作：

```py
>>>list() # an empty list   
>>>list1 = [1,2,3, 4]
>>>list1.append(1)  # append value 1 at the end of the list
>>>list1
[1, 2, 3, 4, 1]
>>>list2 = list1 *2    
[1, 2, 3, 4, 1, 1, 2, 3, 4, 1]
>>> min(list1)
1
>>> max(list1)
4
>>>list1.insert(0,2)  # insert an value 2 at index 0
>>> list1
[2, 1, 2, 3, 4, 1]
>>>list1.reverse()
>>> list1
[1, 4, 3, 2, 1, 2]
>>>list2=[11,12]
>>>list1.extend(list2)
>>> list1
[1, 4, 3, 2, 1, 2, 11, 12]
>>>sum(list1)
36
>>> len(list1)
8
>>> list1.sort()
>>> list1
[1, 1, 2, 2, 3, 4, 11, 12]
>>>list1.remove(12)   #remove value 12 form the list
>>> list1
[1, 1, 2, 2, 3, 4, 11]
```

# 了解元组

元组是任意对象的不可变序列。元组是一个逗号分隔的值序列；然而，通常的做法是将它们括在括号中。当我们想要在一行中设置多个变量，或者允许函数返回不同对象的多个值时，元组非常有用。元组是一种有序的项目序列，类似于`list`数据类型。唯一的区别是元组是不可变的；因此，一旦创建，它们就不能被修改，不像`list`。元组由大于零的整数索引。元组是**可散列**的，这意味着我们可以对它们的列表进行排序，并且它们可以用作字典的键。

我们还可以使用内置函数`tuple()`创建一个元组。如果没有参数，这将创建一个空元组。如果`tuple()`的参数是一个序列，那么这将创建一个由该序列元素组成的元组。在创建只有一个元素的元组时，重要的是要记住使用尾随逗号——没有尾随逗号，这将被解释为一个字符串。元组的一个重要用途是通过在赋值的左侧放置一个元组来一次性分配多个变量。

考虑一个例子：

```py
>>> t= tuple()   # create an empty tuple
>>> type(t)
<class 'tuple'>
>>> t=('a',)  # create a tuple with 1 element
>>> t
('a',)
>>> print('type is ',type(t))
type is  <class 'tuple'>
>>> tpl=('a','b','c')
>>> tpl('a', 'b', 'c')
>>> tuple('sequence')
('s', 'e', 'q', 'u', 'e', 'n', 'c', 'e')
>>> x,y,z= tpl   #multiple assignment 
>>> x
'a'
>>> y
'b'
>>> z
'c'
>>> 'a' in tpl  # Membership can be tested
True
>>> 'z' in tpl
False
```

大多数运算符，如切片和索引运算符，都像列表一样工作。然而，由于元组是不可变的，尝试修改元组的元素会导致`TypeError`。我们可以像比较其他序列一样比较元组，使用`==`、`>`和`<`运算符。考虑一个示例代码片段：

```py
>>> tupl = 1, 2,3,4,5  # braces are optional
>>>print("tuple value at index 1 is ", tupl[1])
tuple value at index 1 is  2
>>> print("tuple[1:3] is ", tupl[1:3])
tuple[1:3] is (2, 3)
>>>tupl2 = (11, 12,13)
>>>tupl3= tupl + tupl2   # tuple concatenation
>>> tupl3
(1, 2, 3, 4, 5, 11, 12, 13)
>>> tupl*2      # repetition for tuples
(1, 2, 3, 4, 5, 1, 2, 3, 4, 5)
>>> 5 in tupl    # membership test
True
>>> tupl[-1]     # negative indexing
5
>>> len(tupl)   # length function for tuple
5
>>> max(tupl)
5
>>> min(tupl)
1
>>> tupl[1] = 5 # modification in tuple is not allowed.
Traceback (most recent call last):  
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>>print (tupl== tupl2)
False
>>>print (tupl>tupl2)
False
```

让我们考虑另一个例子来更好地理解元组。例如，我们可以使用多个赋值来交换元组中的值：

```py
>>> l = ['one','two']
>>> x,y = l
('one', 'two')
>>> x,y = y,x
>>> x,y
('two', 'one')
```

# 从字典开始

在Python中，`字典`数据类型是最受欢迎和有用的数据类型之一。字典以键和值对的映射方式存储数据。字典主要是对象的集合；它们由数字、字符串或任何其他不可变对象索引。字典中的键应该是唯一的；然而，字典中的值可以被更改。Python字典是唯一的内置映射类型；它们可以被看作是从一组键到一组值的映射。它们使用`{key:value}`的语法创建。例如，以下代码可以用来创建一个将单词映射到数字的字典，使用不同的方法：

```py
>>>a= {'Monday':1,'Tuesday':2,'Wednesday':3} #creates a dictionary 
>>>b =dict({'Monday':1 , 'Tuesday': 2, 'Wednesday': 3})
>>> b
{'Monday': 1, 'Tuesday': 2, 'Wednesday': 3}
>>> c= dict(zip(['Monday','Tuesday','Wednesday'], [1,2,3]))
>>> c={'Monday': 1, 'Tuesday': 2, 'Wednesday': 3}
>>> d= dict([('Monday',1), ('Tuesday',2), ('Wednesday',3)])
>>>d
{'Monday': 1, 'Tuesday': 2, 'Wednesday': 3}
```

我们可以添加键和值。我们还可以更新多个值，并使用`in`运算符测试值的成员资格或出现情况，如下面的代码示例所示：

```py
>>>d['Thursday']=4     #add an item
>>>d.update({'Friday':5,'Saturday':6})  #add multiple items
>>>d
{'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
>>>'Wednesday' in d  # membership test (only in keys)
True
>>>5 in d       # membership do not check in values
False

```

如果列表很长，使用`in`运算符在列表中查找元素会花费太多时间。在列表中查找元素所需的运行时间随着列表大小的增加而线性增加。而字典中的`in`运算符使用哈希函数，这使得字典非常高效，因为查找元素所花费的时间与字典的大小无关。

注意当我们打印字典的`{key: value}`对时，它并没有按特定顺序进行。这不是问题，因为我们使用指定的键来查找每个字典值，而不是一个有序的整数序列，就像对字符串和列表一样：

```py
>>> dict(zip('packt', range(5)))
{'p': 0, 'a': 1, 'c': 2, 'k': 3, 't': 4}
>>> a = dict(zip('packt', range(5)))
>>> len(a)   # length of dictionary a
5
>>> a['c']  # to check the value of a key
2
>>> a.pop('a')  
1
>>> a{'p': 0, 'c': 2, 'k': 3, 't': 4}
>>> b= a.copy()   # make a copy of the dictionary
>>> b
{'p': 0, 'c': 2, 'k': 3, 't': 4}
>>> a.keys()
dict_keys(['p', 'c', 'k', 't'])
>>> a.values()
dict_values([0, 2, 3, 4])
>>> a.items()
dict_items([('p', 0), ('c', 2), ('k', 3), ('t', 4)])
>>> a.update({'a':1})   # add an item in the dictionary
>>> a{'p': 0, 'c': 2, 'k': 3, 't': 4, 'a': 1}
>>> a.update(a=22)  # update the value of key 'a'
>>> a{'p': 0, 'c': 2, 'k': 3, 't': 4, 'a': 22}

```

以下表格包含了所有字典方法及其描述：

| **方法** | **描述** |
| --- | --- |
| `len(d)` | 返回字典`d`中的项目总数。 |
| `d.clear()` | 从字典`d`中删除所有项目。 |
| `d.copy()` | 返回字典`d`的浅拷贝。 |
| `d.fromkeys(s[,value])` | 返回一个新字典，其键来自序列`s`，值设置为`value`。 |
| `d.get(k[,v])` | 如果找到，则返回`d[k]`；否则返回`v`（如果未给出`v`，则返回`None`）。 |
| `d.items()` | 返回字典`d`的所有`键:值`对。 |
| `d.keys()` | 返回字典`d`中定义的所有键。 |
| `d.pop(k[,default])` | 返回`d[k]`并从`d`中删除它。 |
| `d.popitem()` | 从字典`d`中删除一个随机的`键:值`对，并将其作为元组返回。 |
| `d.setdefault(k[,v])` | 返回`d[k]`。如果找不到，它返回`v`并将`d[k]`设置为`v`。 |
| `d.update(b)` | 将`b`字典中的所有对象添加到`d`字典中。 |
| `d.values()` | 返回字典`d`中的所有值。 |

# Python

应该注意，当将`in`运算符应用于字典时，其工作方式与应用于列表时略有不同。当我们在列表上使用`in`运算符时，查找元素所需的时间与列表的大小之间的关系被认为是线性的。也就是说，随着列表的大小变大，找到元素所需的时间最多是线性增长的。算法运行所需的时间与其输入大小之间的关系通常被称为其时间复杂度。我们将在接下来的章节中更多地讨论这个重要的主题。

与`list`对象相反，当`in`运算符应用于字典时，它使用哈希算法，这会导致每次查找时间的增加几乎独立于字典的大小。这使得字典作为处理大量索引数据的一种方式非常有用。我们将在[第4章](234b9cb7-47a2-4910-8039-d7fed6c4af81.xhtml)和[第14章](1f1d6528-c080-4c90-abab-ab41d55d721e.xhtml)中更多地讨论这个重要主题，即哈希的增长率。

# 对字典进行排序

如果我们想对字典的键或值进行简单的排序，我们可以这样做：

```py
>>> d = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6} 
>>> sorted(list(d)) 
['five', 'four', 'one', 'six', 'three', 'two']  
>>> sorted(list(d.values())) 
[1, 2, 3, 4, 5, 6] 

```

请注意，前面代码中的第一行按字母顺序对键进行排序，第二行按整数值的顺序对值进行排序。

`sorted()`方法有两个感兴趣的可选参数：`key`和`reverse`。`key`参数与字典键无关，而是一种传递函数给排序算法以确定排序顺序的方法。例如，在下面的代码中，我们使用`__getitem__`特殊方法根据字典的值对字典键进行排序：

![](Images/f060180f-67e3-4a18-a2cf-de92fa784c2c.png)

基本上，前面的代码对`d`中的每个键使用相应的值进行排序。我们也可以根据字典键的排序顺序对值进行排序。然而，由于字典没有一种方法可以通过其值返回一个键，就像列表的`list.index`方法一样，使用可选的`key`参数来做到这一点有点棘手。另一种方法是使用列表推导式，就像下面的例子演示的那样：

![](Images/8d05572b-0a3c-4320-9d9c-b6cf38b5243e.png)

`sorted()`方法还有一个可选的`reverse`参数，毫不奇怪，它确实做到了它所说的—反转排序列表的顺序，就像下面的例子一样：

![](Images/bf6f08ee-d55a-4c47-8424-f493cbe2fc19.png)

现在，假设我们有以下字典，其中英语单词作为键，法语单词作为值。我们的任务是将字符串值放在正确的数字顺序中：

```py
d2={'one':'uno','two':'deux','three':'trois','four':'quatre','five':'cinq','six':'six'}
```

当然，当我们打印这个字典时，它可能不会按正确的顺序打印。因为所有的键和值都是字符串，我们没有数字顺序的上下文。为了将这些项目放在正确的顺序中，我们需要使用我们创建的第一个字典，将单词映射到数字作为对英语到法语字典进行排序的一种方式：

![](Images/06a4e849-5cde-45c4-893c-d2182152f278.png)

请注意，我们正在使用第一个字典`d`的值来对第二个字典`d2`的键进行排序。由于我们两个字典中的键是相同的，我们可以使用列表推导式来对法语到英语字典的值进行排序：

![](Images/25cc0171-9321-4d8f-8b78-aafc7196f837.png)

当然，我们可以定义自己的自定义方法，然后将其用作排序方法的关键参数。例如，在这里，我们定义一个简单地返回字符串的最后一个字母的函数：

```py
def corder(string): 
    return (string[len(string)-1])
```

然后，我们可以将其用作排序函数的关键，按其最后一个字母对每个元素进行排序：

![](Images/95c948bf-cd5b-4254-add5-e98835006e00.png)

# 文本分析的字典

字典的常见用途是计算序列中相似项的出现次数；一个典型的例子是计算文本中单词的出现次数。以下代码创建了一个字典，其中文本中的每个单词都用作键，出现次数作为其值。这使用了一个非常常见的嵌套循环习语。在这里，我们使用它来遍历文件中的行的外部循环和字典的键的内部循环：

```py
def wordcount(fname):  
   try: 
        fhand=open(fname) 
   except:
        print('File can not be opened') 
        exit() 

   count=dict() 
   for line in fhand: 
        words=line.split() 
        for word in words: 
            if word not in count: 
                count[word]=1  
            else: 
                count[word]+=1 
   return(count)

```

这将返回一个字典，其中每个唯一单词在文本文件中都有一个元素。一个常见的任务是将这些项目过滤成我们感兴趣的子集。您需要在运行代码的同一目录中保存一个文本文件。在这里，我们使用了`alice.txt`，这是《爱丽丝梦游仙境》的一个简短摘录。要获得相同的结果，您可以从[davejulian.net/bo5630](http://davejulian.net/bo5630)下载`alice.txt`，或者使用您自己的文本文件。在下面的代码中，我们创建了另一个字典`filtered`，其中包含来自`count`的子集：

```py
count=wordcount('alice.txt') 
filtered={key:value for key, value in count.items() if value <20 and value>16 }
```

当我们打印过滤字典时，我们得到以下结果：

```py
{'once': 18, 'eyes': 18, 'There': 19, 'this,': 17, 'before': 19, 'take': 18, 'tried': 18, 'even': 17, 'things': 19, 'sort': 17, 'her,': 18, '`And': 17, 'sat': 17, '`But': 19, "it,'": 18, 'cried': 18, '`Oh,': 19, 'and,': 19, "`I'm": 19, 'voice': 17, 'being': 19, 'till': 19, 'Mouse': 17, '`but': 19, 'Queen,': 17}
```

请注意使用**字典推导**来构建过滤字典。字典推导的工作方式与我们在[第1章](2818f56c-fbcf-422f-83dc-16cbdbd8b5bf.xhtml)中看到的列表推导相同，即*Python对象、类型和表达式*。

# 集合

集合是无序的唯一项集合。集合本身是可变的——我们可以向其中添加和删除项目；但是，项目本身必须是不可变的。集合的一个重要区别是它们不能包含重复的项目。集合通常用于执行诸如交集、并集、差集和补集等数学运算。

与序列类型不同，集合类型不提供任何索引或切片操作。Python中有两种类型的集合对象，可变的`set`对象和不可变的`frozenset`对象。使用花括号内的逗号分隔的值创建集合。顺便说一句，我们不能使用`a={}`创建一个空集，因为这将创建一个字典。要创建一个空集，我们要么写`a=set()`，要么写`a=frozenset()`。

集合的方法和操作描述在下表中：

| **方法** | **描述** |
| --- | --- |
| `len(a)` | 提供了`a`集合中元素的总数。 |
| `a.copy()` | 提供了`a`集合的另一个副本。 |
| `a.difference(t)` | 提供了`a`集合中存在但不在`t`中的元素的集合。 |
| `a.intersection(t)` | 提供了两个集合`a`和`t`中都存在的元素的集合。 |
| `a.isdisjoint(t)` | 如果两个集合`a`和`t`中没有共同的元素，则返回`True`。 |
| `a.issubset(t)` | 如果`a`集合的所有元素也在`t`集合中，则返回`True`。 |
| `a.issuperset(t)` | 如果`t`集合的所有元素也在`a`集合中，则返回`True`。 |
| `a.symmetric_difference(t)` | 返回一个既在`a`集合中又在`t`集合中的元素的集合，但不在两者中都存在。 |
| `a.union(t)` | 返回一个既在`a`集合中又在`t`集合中的元素的集合。 |

在上表中，参数`t`可以是任何支持迭代的Python对象，所有方法都适用于`set`和`frozenset`对象。重要的是要意识到这些方法的操作符版本要求它们的参数是集合，而方法本身可以接受任何可迭代类型。例如，对于任何集合`s`，`s-[1,2,3]`将生成不支持的操作数类型。使用等效的`s.difference([1,2,3])`将返回一个结果。

可变的`set`对象具有其他方法，如下表所述：

| **方法** | **描述** |
| --- | --- |
| `s.add(item)` | 将项目添加到`s`；如果项目已经添加，则不会发生任何事情。 |
| `s.clear()` | 从集合`s`中删除所有元素。 |
| `s.difference_update(t)` | 从`s`集合中删除那些也在其他集合`t`中的元素。 |
| `s.discard(item)` | 从集合`s`中删除项目。 |
| `s.intersection_update(t)` | 从集合`s`中删除不在集合`s`和`t`的交集中的项目。 |
| `s.pop()` | 从集合`s`中返回一个任意项目，并从`s`集合中删除它。 |
| `s.remove(item)` | 从`s`集合中删除项目。 |
| `s.symetric_difference_update(t)` | 从集合`s`中删除不在集合`s`和`t`的对称差集中的所有元素。 |
| `s.update(t)` | 将可迭代对象`t`中的所有项目附加到`s`集合。 |

在这里，考虑一个简单的示例，显示了添加、删除、丢弃和清除操作：

```py
>>> s1 = set()
>>> s1.add(1)
>>> s1.add(2)
>>> s1.add(3)
>>> s1.add(4)
>>> s1
{1, 2, 3, 4}
>>> s1.remove(4)
>>> s1
{1, 2, 3}
>>> s1.discard(3)
>>> s1
{1, 2}
>>>s1.clear()
>>>s1
set()
```

以下示例演示了一些简单的集合操作及其结果：

![](Images/1a0c26a1-0555-49b8-8608-248609446dc5.png)

请注意，`set`对象不在乎其成员不全是相同类型，只要它们都是不可变的。如果您尝试在集合中使用可变对象，例如列表或字典，您将收到一个不可哈希类型错误。可哈希类型都有一个哈希值，在实例的整个生命周期中不会改变。所有内置的不可变类型都是可哈希的。所有内置的可变类型都不可哈希，因此不能用作集合的元素或字典的键。

还要注意在前面的代码中，当我们打印出`s1`和`s2`的并集时，只有一个值为`'ab'`的元素。这是集合的一个自然属性，它们不包括重复项。

除了这些内置方法之外，我们还可以对集合执行许多其他操作。例如，要测试集合的成员资格，请使用以下方法：

![](Images/5c5e5c1a-63b6-4006-afac-81716a723380.png)

我们可以使用以下方法循环遍历集合中的元素：

![](Images/2399773c-55a5-49a9-9a07-415c66c31853.png)

# 不可变集合

Python有一个名为`frozenset`的不可变集合类型。它的工作方式几乎与`set`完全相同，除了不允许更改值的方法或操作，例如`add()`或`clear()`方法。这种不可变性有几种有用之处。

例如，由于普通集合是可变的，因此不可哈希，它们不能用作其他集合的成员。另一方面，`frozenset`是不可变的，因此可以用作集合的成员：

![](Images/a230c9f0-720b-45ec-b2d3-42635c4e0682.png)

此外，`frozenset`的不可变属性意味着我们可以将其用作字典的键，如下例所示：

![](Images/b92757f5-6d64-4355-866e-0cdb00e71f53.png)

# 数据结构和算法的模块

除了内置类型之外，还有几个Python模块可以用来扩展内置类型和函数。在许多情况下，这些Python模块可能提供效率和编程优势，使我们能够简化我们的代码。

到目前为止，我们已经查看了字符串、列表、集合和字典的内置数据类型，以及十进制和分数模块。它们通常被术语**抽象数据类型**（**ADT**）描述。 ADT可以被认为是可以在数据上执行的操作集的数学规范。它们由其行为而不是其实现来定义。除了我们已经查看的ADT之外，还有几个Python库提供了对内置数据类型的扩展。这将在下一节中讨论。

# 集合

`collections`模块提供了更专门的、高性能的替代品，用于内置数据类型，以及一个实用函数来创建命名元组。以下表列出了`collections`模块的数据类型和操作及其描述：

| **数据类型或操作** | **描述** |
| --- | --- |
| `namedtuple()` | 创建具有命名字段的元组子类。 |
| `deque` | 具有快速追加和弹出的列表。 |
| `ChainMap` | 类似字典的类，用于创建多个映射的单个视图。 |
| `Counter` | 用于计算可散列对象的字典子类。 |
| `OrderedDict` | 记住条目顺序的字典子类。 |
| `defaultdict` | 调用函数以提供缺失值的字典子类。 |
| `UserDict UserList UserString` | 这三种数据类型只是它们基础基类的简单包装器。它们的使用在很大程度上已被能够直接对其各自的基类进行子类化所取代。可以用来作为属性访问基础对象。 |

# 双端队列

双端队列，通常发音为*decks*，是类似列表的对象，支持线程安全、内存高效的追加。双端队列是可变的，并支持列表的一些操作，如索引。双端队列可以通过索引分配，例如，`dq[1] = z`；但是，我们不能直接切片双端队列。例如，`dq[1:2]`会导致`TypeError`（我们将看一种从双端队列返回切片作为列表的方法）。

双端队列比列表的主要优势在于，在双端队列的开头插入项目要比在列表的开头插入项目快得多，尽管在双端队列的末尾插入项目的速度比列表上的等效操作略慢一些。双端队列是线程安全的，并且可以使用`pickle`模块进行序列化。

一个有用的思考双端队列的方式是填充和消耗项目。双端队列中的项目通常是从两端顺序填充和消耗的：

![](Images/bdd5dc25-b4ee-4f13-80dd-c2e7b82634c1.png)

我们可以使用`pop()`和`popleft()`方法来消耗双端队列中的项目，如下例所示：

![](Images/13e07836-1988-41fc-ba5a-aa3b008691b0.png)

我们还可以使用`rotate(n)`方法将所有项目向右移动和旋转`n`步，对于`n`整数的正值或`n`步的负值向左移动，使用正整数作为参数，如下例所示：

![](Images/bb9cb7b6-1956-438a-b247-b56375078185.png)

请注意，我们可以使用`rotate`和`pop`方法来删除选定的元素。还值得知道的是，返回双端队列切片的简单方法，可以按以下方式完成：

![](Images/811e6dec-b990-403d-a4a2-02d5214f0f38.png)

`itertools.islice()`方法的工作方式与列表上的切片相同，只是它不是以列表作为参数，而是以可迭代对象作为参数，并返回所选值，按起始和停止索引，作为列表。

双端队列的一个有用特性是它们支持一个`maxlen`可选参数，用于限制双端队列的大小。这使得它非常适合一种称为**循环缓冲区**的数据结构。这是一种固定大小的结构，实际上是端对端连接的，它们通常用于缓冲数据流。以下是一个基本示例：

```py
dq2=deque([],maxlen=3) 
for i in range(6):
    dq2.append(i) 
    print(dq2)
```

这将打印出以下内容：

![](Images/2d67615b-9051-493c-9727-6a2f6244f6f1.png)

在这个例子中，我们从右侧填充并从左侧消耗。请注意，一旦缓冲区已满，最旧的值将首先被消耗，然后从右侧替换值。在[第4章](234b9cb7-47a2-4910-8039-d7fed6c4af81.xhtml)中，当实现循环列表时，我们将再次看循环缓冲区。

# ChainMap对象

`collections.chainmap`类是在Python 3.2中添加的，它提供了一种将多个字典或其他映射链接在一起，以便它们可以被视为一个对象的方法。此外，还有一个`maps`属性，一个`new_child()`方法和一个`parents`属性。`ChainMap`对象的基础映射存储在列表中，并且可以使用`maps[i]`属性来检索第`i`个字典。请注意，尽管字典本身是无序的，`ChainMap`对象是有序的字典列表。

`ChainMap`在使用包含相关数据的多个字典的应用程序中非常有用。消费应用程序期望按优先级获取数据，如果两个字典中的相同键出现在基础列表的开头，则该键将优先考虑。`ChainMap`通常用于模拟嵌套上下文，例如当我们有多个覆盖配置设置时。以下示例演示了`ChainMap`的可能用例：

```py
>>> import collections
>>> dict1= {'a':1, 'b':2, 'c':3}
>>> dict2 = {'d':4, 'e':5}
>>> chainmap = collections.ChainMap(dict1, dict2)  # linking two dictionaries
>>> chainmap
ChainMap({'a': 1, 'b': 2, 'c': 3}, {'d': 4, 'e': 5})
>>> chainmap.maps
[{'a': 1, 'b': 2, 'c': 3}, {'d': 4, 'e': 5}]
>>> chainmap.values
<bound method Mapping.values of ChainMap({'a': 1, 'b': 2, 'c': 3}, {'d': 4, 'e': 5})
>>>> chainmap['b']   #accessing values 
2
>>> chainmap['e']
5
```

使用`ChainMap`对象而不仅仅是字典的优势在于我们保留了先前设置的值。添加子上下文会覆盖相同键的值，但不会从数据结构中删除它。当我们需要保留更改记录以便可以轻松回滚到先前的设置时，这可能很有用。

我们可以通过为`map()`方法提供适当的索引来检索和更改任何字典中的任何值。此索引表示`ChainMap`中的一个字典。此外，我们可以使用`parents()`方法检索父设置，即默认设置：

```py
>>> from collections import ChainMap
>>> defaults= {'theme':'Default','language':'eng','showIndex':True, 'showFooter':True}
>>> cm= ChainMap(defaults)   #creates a chainMap with defaults configuration
>>> cm.maps[{'theme': 'Default', 'language': 'eng', 'showIndex': True, 'showFooter': True}]
>>> cm.values()
ValuesView(ChainMap({'theme': 'Default', 'language': 'eng', 'showIndex': True, 'showFooter': True}))
>>> cm2= cm.new_child({'theme':'bluesky'}) # create a new chainMap with a child that overrides the parent.
>>> cm2['theme']  #returns the overridden theme'bluesky'
>>> cm2.pop('theme')  # removes the child theme value
'bluesky' 
>>> cm2['theme']
'Default'
>>> cm2.maps[{}, {'theme': 'Default', 'language': 'eng', 'showIndex': True, 'showFooter': True}]
>>> cm2.parents
ChainMap({'theme': 'Default', 'language': 'eng', 'showIndex': True, 'showFooter': True})
```

# 计数器对象

`Counter`是字典的一个子类，其中每个字典键都是可散列对象，关联的值是该对象的整数计数。有三种初始化计数器的方法。我们可以将任何序列对象、`key:value`对的字典或格式为`(object=value,...)`的元组传递给它，如下例所示：

```py
>>> from collections import Counter
>>> Counter('anysequence')
Counter({'e': 3, 'n': 2, 'a': 1, 'y': 1, 's': 1, 'q': 1, 'u': 1, 'c': 1})
>>> c1 = Counter('anysequence')
>>> c2= Counter({'a':1, 'c': 1, 'e':3})
>>> c3= Counter(a=1, c= 1, e=3)
>>> c1
Counter({'e': 3, 'n': 2, 'a': 1, 'y': 1, 's': 1, 'q': 1, 'u': 1, 'c': 1})
>>> c2
Counter({'e': 3, 'a': 1, 'c': 1})
>>> c3
Counter({'e': 3, 'a': 1, 'c': 1})
```

我们还可以创建一个空的计数器对象，并通过将其`update`方法传递给一个可迭代对象或字典来填充它。请注意，`update`方法添加计数，而不是用新值替换它们。填充计数器后，我们可以以与字典相同的方式访问存储的值，如下例所示：

```py
>>> from collections import Counter
>>> ct = Counter()  # creates an empty counter object
>>> ct
Counter()
>>> ct.update('abca') # populates the object
>>> ct
Counter({'a': 2, 'b': 1, 'c': 1})
>>> ct.update({'a':3}) # update the count of 'a'
>>> ct
Counter({'a': 5, 'b': 1, 'c': 1})
>>> for item in ct:
 ...  print('%s: %d' % (item, ct[item]))
 ...
a: 5
b: 1
c: 1
```

计数器对象和字典之间最显着的区别是计数器对象对于缺失的项返回零计数，而不是引发键错误。我们可以使用其`elements()`方法从`Counter`对象创建迭代器。这将返回一个迭代器，其中不包括小于一的计数，并且顺序不被保证。在下面的代码中，我们执行一些更新，从`Counter`元素创建一个迭代器，并使用`sorted()`按字母顺序对键进行排序：

```py
>>> ct
Counter({'a': 5, 'b': 1, 'c': 1})
>>> ct['x']
0
>>> ct.update({'a':-3, 'b':-2, 'e':2})
>>> ct
Counter({'a': 2, 'e': 2, 'c': 1, 'b': -1})
>>>sorted(ct.elements())
['a', 'a', 'c', 'e', 'e']
```

另外两个值得一提的`Counter`方法是`most_common()`和`subtract()`。最常见的方法接受一个正整数参数，确定要返回的最常见元素的数量。元素作为(key,value)元组的列表返回。

减法方法的工作方式与更新相同，只是它不是添加值，而是减去它们，如下例所示：

```py
>>> ct.most_common()
[('a', 2), ('e', 2), ('c', 1), ('b', -1)]
>>> ct.subtract({'e':2})
>>> ct
Counter({'a': 2, 'c': 1, 'e': 0, 'b': -1})
```

# 有序字典

有序字典的重要之处在于它们记住插入顺序，因此当我们对它们进行迭代时，它们会按照插入顺序返回值。这与普通字典相反，普通字典的顺序是任意的。当我们测试两个字典是否相等时，这种相等性仅基于它们的键和值；但是，对于`OrderedDict`，插入顺序也被视为两个具有相同键和值的`OrderedDict`对象之间的相等性测试，但是插入顺序不同将返回`False`：

```py
>>> import collections
>>> od1=  collections.OrderedDict()
>>> od1['one'] = 1
>>> od1['two'] = 2
>>> od2 =  collections.OrderedDict()
>>> od2['two'] = 2
>>> od2['one'] = 1
>>> od1==od2
False
```

类似地，当我们使用`update`从列表添加值时，`OrderedDict`将保留与列表相同的顺序。这是在迭代值时返回的顺序，如下例所示：

```py
>>> kvs = [('three',3), ('four',4), ('five',5)]
>>> od1.update(kvs)
>>> od1
OrderedDict([('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5)])
>>> for k, v in od1.items(): print(k, v)
```

```py
...
one 1
two 2
three 3
four 4
five 5
```

`OrderedDict`经常与sorted方法一起使用，以创建一个排序的字典。在下面的示例中，我们使用Lambda函数对值进行排序，并且在这里我们使用数值表达式对整数值进行排序：

```py
>>> od3 = collections.OrderedDict(sorted(od1.items(), key= lambda t : (4*t[1])- t[1]**2))
>>>od3
OrderedDict([('five', 5), ('four', 4), ('one', 1), ('three', 3), ('two', 2)])
>>> od3.values() 
odict_values([5, 4, 1, 3, 2])
```

# defaultdict

`defaultdict`对象是`dict`的子类，因此它们共享方法和操作。它作为初始化字典的便捷方式。使用`dict`时，当尝试访问尚未在字典中的键时，Python会抛出`KeyError`。`defaultdict`覆盖了一个方法，`missing(key)`，并创建了一个新的实例变量，`default_factory`。使用`defaultdict`，而不是抛出错误，它将运行作为`default_factory`参数提供的函数，该函数将生成一个值。`defaultdict`的一个简单用法是将`default_factory`设置为`int`，并用它快速计算字典中项目的计数，如下例所示：

```py
>>> from collections import defaultdict
>>> dd = defaultdict(int)
>>> words = str.split('red blue green red yellow blue red green green red')
>>> for word in words: dd[word] +=1
...
>>> dd
defaultdict(<class 'int'>, {'red': 4, 'blue': 2, 'green': 3, 'yellow': 1})

```

您会注意到，如果我们尝试使用普通字典来做这件事，当我们尝试添加第一个键时，我们会得到一个键错误。我们提供给`defaultdict`的`int`实际上是`int()`函数，它只是返回零。

当然，我们可以创建一个函数来确定字典的值。例如，以下函数在提供的参数是主要颜色（即`red`，`green`或`blue`）时返回`True`，否则返回`False`：

```py
def isprimary(c):
     if (c=='red') or (c=='blue') or (c=='green'): 
         return True 
     else: 
         return False
```

# 了解命名元组

`namedtuple`方法返回一个类似元组的对象，其字段可以通过命名索引以及普通元组的整数索引进行访问。这允许在某种程度上自我记录和更易读的代码。在需要轻松跟踪每个元组代表的内容的应用程序中，这可能特别有用。此外，`namedtuple`从元组继承方法，并且与元组向后兼容。

字段名称作为逗号和/或空格分隔的值传递给`namedtuple`方法。它们也可以作为字符串序列传递。字段名称是单个字符串，可以是任何合法的Python标识符，不能以数字或下划线开头。一个典型的例子如下所示：

```py
>>> from collections import namedtuple
>>> space = namedtuple('space', 'x y z')
>>> s1= space(x=2.0, y=4.0, z=10) # we can also use space(2.0,4.0, 10)
>>> s1
space(x=2.0, y=4.0, z=10)
>>> s1.x * s1.y * s1.z   # calculate the volume
80.0
```

除了继承的元组方法之外，命名元组还定义了三种自己的方法，`_make()`，`asdict()`和`_replace`。这些方法以下划线开头，以防止与字段名称可能发生冲突。`_make()`方法将可迭代对象作为参数，并将其转换为命名元组对象，如下例所示：

```py
>>> sl = [4,5,6]
>>> space._make(sl)
space(x=4, y=5, z=6)
>>> s1._1
4
```

`_asdict`方法返回一个`OrderedDict`对象，其中字段名称映射到索引键，值映射到字典值。`_replace`方法返回元组的新实例，替换指定的值。此外，`_fields`返回列出字段名称的字符串元组。`_fields_defaults`方法提供将字段名称映射到默认值的字典。考虑以下示例代码片段：

```py
>>> s1._asdict()
OrderedDict([('x', 3), ('_1', 4), ('z', 5)])
>>> s1._replace(x=7, z=9)
space2(x=7, _1=4, z=9)
>>> space._fields
('x', 'y', 'z')
>>> space._fields_defaults
{}
```

# 数组

`array`模块定义了一种类似于列表数据类型的数据类型数组，除了它们的内容必须是由机器架构或底层C实现确定的单一类型的约束。

数组的类型是在创建时确定的，并且由以下类型代码之一表示：

| **代码** | **C类型** | **Python类型** | **最小字节数** |
| --- | --- | --- | --- |
| 'b' | `signedchar` | int | 1 |
| 'B' | `unsignedchar` | int | 1 |
| 'u' | `Py_UNICODE` | Unicodecharacter | 2 |
| 'h' | `signedshort` | int | 2 |
| 'H' | `unsignedshort` | int | 2 |
| 'i' | `signedint` | int | 2 |
| 'I' | `unsignedint` | int | 2 |
| 'l' | `signedlong` | int | 4 |
| 'L' | `unsignedlong` | int | 8 |
| 'q' | `signedlonglong` | int | 8 |
| 'Q' | `unsignedlonlong` | int | 8 |
| 'f' | `float` | float | 4 |
| 'd' | `double` | float | 8 |

数组对象支持属性和方法：

| **属性或方法** | **描述** |
| --- | --- |
| `a.itemsize` | 一个数组项的大小（以字节为单位）。 |
| `a.append(x)` | 在`a`数组的末尾添加一个`x`元素。 |
| `a.buffer_info()` | 返回一个元组，包含用于存储数组的缓冲区的当前内存位置和长度。 |
| `a.byteswap()` | 交换`a`数组中每个项目的字节顺序。 |
| `a.count(x)` | 返回`a`数组中`x`的出现次数。 |
| `a.extend(b)` | 在`a`数组的末尾添加可迭代对象`b`的所有元素。 |
| `a.frombytes(s)` | 从字符串`s`中附加元素，其中字符串是机器值的数组。 |
| `a.fromfile(f,n)` | 从文件中读取`n`个机器值，并将它们附加到数组的末尾。 |
| `a.fromlist(l)` | 将`l`列表中的所有元素附加到数组。 |
| `a.fromunicode(s)` | 用Unicode字符串`s`扩展`u`类型的数组。 |
| `index(x)` | 返回`x`元素的第一个（最小）索引。 |
| `a.insert(i,x)` | 在数组的`i`索引位置插入值为`x`的项目。 |
| `a.pop([i])` | 返回索引`i`处的项目，并从数组中删除它。 |
| `a.remove(x)` | 从数组中删除第一个出现的`x`项。 |
| `a.reverse()` | 颠倒`a`数组中项目的顺序。 |
| `a.tofile(f)` | 将所有元素写入`f`文件对象。 |
| `a.tolist()` | 将数组转换为列表。 |
| `a.tounicode()` | 将`u`类型的数组转换为Unicode字符串。 |

数组对象支持所有正常的序列操作，如索引、切片、连接和乘法。

与列表相比，使用数组是存储相同类型数据的更有效的方法。在下面的例子中，我们创建了一个整数数组，其中包含从`0`到一百万减去`1`的数字，以及一个相同的列表。在整数数组中存储一百万个整数，大约需要相当于等效列表的90%的内存：

```py
>>> import array
>>> ba = array.array('i', range(10**6))
>>> bl = list(range(10**6))
>>> import sys
>>> 100*sys.getsizeof(ba)/sys.getsizeof(bl)
90.92989871246161
```

因为我们对节省空间感兴趣，也就是说，我们处理大型数据集和有限的内存大小，通常我们对数组进行原地操作，只有在需要时才创建副本。通常，enumerate用于对每个元素执行操作。在下面的片段中，我们执行简单的操作，为数组中的每个项目添加一。

值得注意的是，当对创建列表的数组执行操作时，例如列表推导，使用数组的内存效率优势将被抵消。当我们需要创建一个新的数据对象时，一个解决方案是使用生成器表达式来执行操作。

使用这个模块创建的数组不适合需要矢量操作的矩阵工作。在下一章中，我们将构建自己的抽象数据类型来处理这些操作。对于数值工作来说，NumPy扩展也很重要，可以在[www.numpy.org](http://www.numpy.org/)上找到。

# 总结

在最后两章中，我们介绍了Python的语言特性和数据类型。我们研究了内置数据类型和一些内部Python模块，尤其是`collections`模块。还有其他几个与本书主题相关的Python模块，但与其单独检查它们，不如在开始使用它们时，它们的使用和功能应该变得不言自明。还有一些外部库，例如SciPy。

在下一章中，我们将介绍算法设计的基本理论和技术。 
