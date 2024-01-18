# Python 中的数据类型和面向对象编程

在本章中，我们将讨论 Python 中的数据类型和**面向对象编程**（**OOP**）。我们将讨论 Python 中的列表、字典、元组和集合等数据类型。我们还将讨论 OOP，它的必要性以及如何在树莓派基于项目中编写面向对象的代码（例如，使用 OOP 来控制家用电器）。我们将讨论在树莓派 Zero 项目中使用 OOP。

# 列表

在 Python 中，列表是一种数据类型（其文档在此处可用，[`docs.python.org/3.4/tutorial/datastructures.html#`](https://docs.python.org/3.4/tutorial/datastructures.html#)），可用于按顺序存储元素。

本章讨论的主题如果不在实践中使用很难理解。任何使用此符号表示的示例：`>>>`都可以使用 Python 解释器进行测试。

列表可以包含字符串、对象（在本章中详细讨论）或数字等。例如，以下是列表的示例：

```py
    >>> sequence = [1, 2, 3, 4, 5, 6]
 >>> example_list = ['apple', 'orange', 1.0, 2.0, 3]
```

在前面的一系列示例中，`sequence`列表包含介于`1`和`6`之间的数字，而`example_list`列表包含字符串、整数和浮点数的组合。列表用方括号（`[]`）表示。项目可以用逗号分隔添加到列表中：

```py
    >>> type(sequence)
 <class 'list'>
```

由于列表是有序元素的序列，可以通过使用`for`循环遍历列表元素来获取列表的元素，如下所示：

```py
for item in sequence: 
    print("The number is ", item)
```

输出如下：

```py
 The number is  1
 The number is  2
 The number is  3
 The number is  4
 The number is  5
 The number is  6
```

由于 Python 的循环可以遍历一系列元素，它会获取每个元素并将其赋值给`item`。然后将该项打印到控制台上。

# 可以在列表上执行的操作

在 Python 中，可以使用`dir()`方法检索数据类型的属性。例如，可以检索`sequence`列表的可用属性如下：

```py
    >>> dir(sequence)
 ['__add__', '__class__', '__contains__', '__delattr__',
    '__delitem__', '__dir__', '__doc__', '__eq__',
    '__format__', '__ge__', '__getattribute__', '__getitem__',
    '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', 
    '__iter__', '__le__', '__len__', '__lt__', '__mul__',
    '__ne__', '__new__', '__reduce__', '__reduce_ex__',
    '__repr__', '__reversed__', '__rmul__', '__setattr__', 
    '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 
    'append', 'clear', 'copy', 'count', 'extend', 'index',
    'insert', 'pop', 'remove', 'reverse', 'sort']
```

这些属性使得可以在列表上执行不同的操作。让我们详细讨论每个属性。

# 向列表添加元素：

可以使用`append()`方法添加元素：

```py
    >>> sequence.append(7)
 >>> sequence
 [1, 2, 3, 4, 5, 6, 7]
```

# 从列表中删除元素：

`remove()`方法找到元素的第一个实例（传递一个参数）并将其从列表中删除。让我们考虑以下示例：

+   **示例 1**：

```py
       >>> sequence = [1, 1, 2, 3, 4, 7, 5, 6, 7]
 >>> sequence.remove(7)
 >>> sequence
 [1, 1, 2, 3, 4, 5, 6, 7]
```

+   **示例 2**：

```py
       >>> sequence.remove(1)
 >>> sequence
 [1, 2, 3, 4, 5, 6, 7]
```

+   **示例 3**：

```py
       >>> sequence.remove(1)
 >>> sequence
 [2, 3, 4, 5, 6, 7]
```

# 检索元素的索引

`index()`方法返回列表中元素的位置：

```py
    >>> index_list = [1, 2, 3, 4, 5, 6, 7]
 >>> index_list.index(5)
 4
```

在这个例子中，该方法返回元素`5`的索引。由于 Python 使用从 0 开始的索引，因此元素`5`的索引为`4`：

```py
    random_list = [2, 2, 4, 5, 5, 5, 6, 7, 7, 8]
 >>> random_list.index(5)
 3
```

在这个例子中，该方法返回元素的第一个实例的位置。元素`5`位于第三个位置。

# 从列表中弹出一个元素

`pop()`方法允许从指定位置删除一个元素并返回它：

```py
    >>> index_list = [1, 2, 3, 4, 5, 6, 7]
 >>> index_list.pop(3)
 4
 >>> index_list
 [1, 2, 3, 5, 6, 7]
```

在这个例子中，`index_list`列表包含介于`1`和`7`之间的数字。通过传递索引位置`(3)`作为参数弹出第三个元素时，数字`4`从列表中移除并返回。

如果没有为索引位置提供参数，则弹出并返回最后一个元素：

```py
    >>> index_list.pop()
 7
 >>> index_list
 [1, 2, 3, 5, 6]
```

在这个例子中，最后一个元素`(7)`被弹出并返回。

# 计算元素的实例数量：

`count()`方法返回元素在列表中出现的次数。例如，该元素在列表`random_list`中出现两次。

```py
 >>> random_list = [2, 9, 8, 4, 3, 2, 1, 7] >>> random_list.count(2) 2
```

# 在特定位置插入元素：

`insert()`方法允许在列表中的特定位置添加一个元素。例如，让我们考虑以下示例：

```py
    >>> day_of_week = ['Monday', 'Tuesday', 'Thursday',
    'Friday', 'Saturday']
```

在列表中，`Wednesday`缺失。它需要被放置在`Tuesday`和`Thursday`之间的位置 2（Python 使用**零基索引**，即元素的位置/索引从 0、1、2 等开始计数）。可以使用 insert 添加如下：

```py
    >>> day_of_week.insert(2, 'Wednesday')
 >>> day_of_week
 ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
    'Friday', 'Saturday']
```

# 读者的挑战

在前面的列表中，缺少 `Sunday`。使用列表的 `insert` 属性将其插入到正确的位置。

# 扩展列表

可以使用 `extend()` 方法将两个列表合并。`day_of_week` 和 `sequence` 列表可以合并如下：

```py
    >>> day_of_week.extend(sequence)
 >>> day_of_week
 ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
    'Saturday', 1, 2, 3, 4, 5, 6]
```

列表也可以组合如下：

```py
    >>> [1, 2, 3] + [4, 5, 6]
 [1, 2, 3, 4, 5, 6]
```

还可以将一个列表作为另一个列表的元素添加：

```py
    sequence.insert(6, [1, 2, 3])
 >>> sequence
 [1, 2, 3, 4, 5, 6, [1, 2, 3]]
```

# 清除列表的元素

可以使用 `clear()` 方法删除列表的所有元素：

```py
    >>> sequence.clear()
 >>> sequence
 []
```

# 对列表的元素进行排序

列表的元素可以使用 `sort()` 方法进行排序：

```py
    random_list = [8, 7, 5, 2, 2, 5, 7, 5, 6, 4]
 >>> random_list.sort()
 >>> random_list
 [2, 2, 4, 5, 5, 5, 6, 7, 7, 8]
```

当列表由一组字符串组成时，它们按照字母顺序排序：

```py
    >>> day_of_week = ['Monday', 'Tuesday', 'Thursday',
    'Friday', 'Saturday']
 >>> day_of_week.sort()
 >>> day_of_week
 ['Friday', 'Monday', 'Saturday', 'Thursday', 'Tuesday']
```

# 颠倒列表中的元素顺序

`reverse()` 方法使列表元素的顺序颠倒：

```py
    >>> random_list = [8, 7, 5, 2, 2, 5, 7, 5, 6, 4]
 >>> random_list.reverse()
 >>> random_list
 [4, 6, 5, 7, 5, 2, 2, 5, 7, 8]
```

# 创建列表的副本

`copy()` 方法可以创建列表的副本：

```py
    >>> copy_list = random_list.copy()
 >>> copy_list
 [4, 6, 5, 7, 5, 2, 2, 5, 7, 8]
```

# 访问列表元素

可以通过指定 `list_name[i]` 的索引位置来访问列表的元素。例如，可以按照以下方式访问 `random_list` 列表的第零个元素：

```py
 >>> random_list = [4, 6, 5, 7, 5, 2, 2, 5, 7, 8] 
 >>> random_list[0]4>>> random_list[3]7
```

# 访问列表中的一组元素

可以访问指定索引之间的元素。例如，可以检索索引为 2 和 4 之间的所有元素：

```py
    >>> random_list[2:5]
 [5, 7, 5]
```

可以按照以下方式访问列表的前六个元素：

```py
    >>> random_list[:6]
 [4, 6, 5, 7, 5, 2]
```

可以按照以下方式以相反的顺序打印列表的元素：

```py
    >>> random_list[::-1]
 [8, 7, 5, 2, 2, 5, 7, 5, 6, 4]
```

可以按照以下方式获取列表中的每个第二个元素：

```py
    >>> random_list[::2]
 [4, 5, 5, 2, 7]
```

还可以跳过前两个元素后获取第二个元素之后的每个第二个元素：

```py
    >>> random_list[2::2]
 [5, 5, 2, 7]
```

# 列表成员

可以使用 `in` 关键字检查一个值是否是列表的成员。例如：

```py
 >>> random_list = [2, 1, 0, 8, 3, 1, 10, 9, 5, 4]
```

在这个列表中，我们可以检查数字 `6` 是否是成员：

```py
    >>> 6 in random_list
 False
 >>> 4 in random_list
 True
```

# 让我们构建一个简单的游戏！

这个练习由两部分组成。在第一部分中，我们将回顾构建一个包含在 `0` 和 `10` 之间的十个随机数的列表。第二部分是给读者的一个挑战。执行以下步骤：

1.  第一步是创建一个空列表。让我们创建一个名为 `random_list` 的空列表。可以按照以下方式创建一个空列表：

```py
       random_list = []
```

1.  我们将使用 Python 的 `random` 模块 ([`docs.python.org/3/library/random.html`](https://docs.python.org/3/library/random.html)) 生成随机数。为了生成在 `0` 和 `10` 之间的随机数，我们将使用 `random` 模块的 `randint()` 方法。

```py
       random_number = random.randint(0,10)
```

1.  让我们将生成的数字附加到列表中。使用 `for` 循环重复此操作 `10` 次：

```py
       for index in range(0,10):
             random_number = random.randint(0, 10)
             random_list.append(random_number)
       print("The items in random_list are ")
       print(random_list)
```

1.  生成的列表看起来像这样：

```py
       The items in random_list are
 [2, 1, 0, 8, 3, 1, 10, 9, 5, 4]
```

我们讨论了生成一个随机数列表。下一步是接受用户输入，我们要求用户猜一个在 `0` 和 `10` 之间的数字。如果数字是列表的成员，则打印消息 `你的猜测是正确的`，否则打印消息 `对不起！你的猜测是错误的`。我们将第二部分留给读者作为挑战。使用本章提供的 `list_generator.py` 代码示例开始。

# 字典

字典 ([`docs.python.org/3.4/tutorial/datastructures.html#dictionaries`](https://docs.python.org/3.4/tutorial/datastructures.html#dictionaries)) 是一个无序的键值对集合的数据类型。字典中的每个键都有一个相关的值。字典的一个示例是：

```py
 >>> my_dict = {1: "Hello", 2: "World"} >>> my_dict   
 {1: 'Hello', 2: 'World'}
```

通过使用大括号 `{}` 创建字典。在创建时，新成员以以下格式添加到字典中：`key: value`（如前面的示例所示）。在前面的示例中，`1` 和 `2` 是键，而 `'Hello'` 和 `'World'` 是相关的值。添加到字典的每个值都需要有一个相关的键。

字典的元素没有顺序，即不能按照添加的顺序检索元素。可以通过遍历键来检索字典的值。让我们考虑以下示例：

```py
 >>> my_dict = {1: "Hello", 2: "World", 3: "I", 4: "am",
    5: "excited", 6: "to", 7: "learn", 8: "Python" }
```

有几种方法可以打印字典的键或值：

```py
 >>> for key in my_dict: ... 
 print(my_dict[value]) 
 ... Hello World I 
 am excited to learn Python
```

在前面的示例中，我们遍历字典的键并使用键`my_dict[key]`检索值。还可以使用字典中可用的`values()`方法检索值：

```py
 >>> for value in my_dict.values(): ... 

print(value) ... Hello World I am excited to learn Python
```

字典的键可以是整数、字符串或元组。字典的键需要是唯一的且不可变的，即创建后无法修改。无法创建键的重复项。如果向现有键添加新值，则字典中将存储最新值。让我们考虑以下示例：

+   可以按以下方式向字典添加新的键/值对：

```py
 >>> my_dict[9] = 'test' >>> my_dict {1: 'Hello', 2: 'World', 3: 'I', 4: 'am', 5: 'excited',
       6: 'to', 7: 'learn', 8: 'Python', 9: 'test'}
```

+   让我们尝试创建键`9`的重复项：

```py
 >>> my_dict[9] = 'programming' >>> my_dict {1: 'Hello', 2: 'World', 3: 'I', 4: 'am', 5: 'excited',
       6: 'to', 7: 'learn', 8: 'Python', 9: 'programming'}
```

+   如前面的示例所示，当我们尝试创建重复项时，现有键的值会被修改。

+   可以将多个值与一个键关联。例如，作为列表或字典：

```py
 >>> my_dict = {1: "Hello", 2: "World", 3: "I", 4: "am",
      "values": [1, 2, 3,4, 5], "test": {"1": 1, "2": 2} } 
```

字典在解析 CSV 文件并将每一行与唯一键关联的场景中非常有用。字典也用于编码和解码 JSON 数据

# 元组

元组（发音为*two-ple*或*tuh-ple*）是一种不可变的数据类型，按顺序排列并用逗号分隔。可以按以下方式创建元组：

```py
 >>> my_tuple = 1, 2, 3, 4, 5
 >>> my_tuple (1, 2, 3, 4, 5)
```

由于元组是不可变的，因此无法修改给定索引处的值：

```py
    >>> my_tuple[1] = 3
 Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 TypeError: 'tuple' object does not support item assignment
```

元组可以由数字、字符串或列表组成。由于列表是可变的，如果列表是元组的成员，则可以修改。例如：

```py
    >>> my_tuple = 1, 2, 3, 4, [1, 2, 4, 5]
 >>> my_tuple[4][2] = 3
 >>> my_tuple
 (1, 2, 3, 4, [1, 2, 3, 5])
```

元组在值无法修改的情况下特别有用。元组还用于从函数返回值。让我们考虑以下示例：

```py
 >>> for value in my_dict.items(): ... 

 print(value) 
 ...
 (1, 'Hello') (2, 'World') (3, 'I') (4, 'am') ('test', {'1': 1, '2': 2}) ('values', [1, 2, 3, 4, 5])
```

在前面的示例中，`items()`方法返回一个元组列表。

# 集合

集合（[`docs.python.org/3/tutorial/datastructures.html#sets`](https://docs.python.org/3/tutorial/datastructures.html#sets)）是一个无序的不可变元素的集合，不包含重复条目。可以按以下方式创建集合：

```py
 >>> my_set = set([1, 2, 3, 4, 5]) >>> my_set {1, 2, 3, 4, 5}
```

现在，让我们向这个集合添加一个重复的列表：

```py
 >>> my_set.update([1, 2, 3, 4, 5]) >>> my_set {1, 2, 3, 4, 5}
```

集合可以避免重复条目并保存唯一条目。可以将单个元素添加到集合中，如下所示：

```py
 >>> my_set = set([1, 2, 3, 4, 5]) >>> my_set.add(6)
 >>> my_set
 {1, 2, 3, 4, 5, 6}
```

集合用于测试元素在不同集合中的成员资格。有与成员资格测试相关的不同方法。我们建议使用集合的文档来了解每种方法（运行`help(my_set)`以查找成员资格测试的不同方法）。

# Python 中的面向对象编程

面向对象编程有助于简化代码并简化应用程序开发。在重用代码方面尤其有用。面向对象的代码使您能够重用使用通信接口的传感器的代码。例如，所有配有 UART 端口的传感器可以使用面向对象的代码进行分组。

面向对象编程的一个例子是**GPIO Zero 库**（[`www.raspberrypi.org/blog/gpio-zero-a-friendly-python-api-for-physical-computing/`](https://www.raspberrypi.org/blog/gpio-zero-a-friendly-python-api-for-physical-computing/)），在之前的章节中使用过。实际上，在 Python 中一切都是对象。

面向对象的代码在与其他人合作项目时特别有帮助。例如，您可以使用 Python 中的面向对象的代码实现传感器驱动程序并记录其用法。这使其他开发人员能够开发应用程序，而无需关注传感器接口背后的细节。面向对象编程为应用程序提供了模块化，简化了应用程序开发。我们将在本章中回顾一个示例，演示面向对象编程的优势。在本章中，我们将利用面向对象编程为我们的项目带来模块化。

让我们开始吧！

# 重新审视学生 ID 卡示例

让我们重新访问第十章中的身份证示例，*算术运算、循环和闪烁灯*（`input_test.py`）。我们讨论了编写一个简单的程序，用于捕获和打印属于一个学生的信息。学生的联系信息可以按以下方式检索和存储：

```py
name = input("What is your name? ") 
address = input("What is your address? ") 
age = input("How old are you? ")
```

现在，考虑一个情景，需要保存和在程序执行期间的任何时刻检索 10 个学生的信息。我们需要为用于保存学生信息的变量想出一个命名规范。如果我们使用 30 个不同的变量来存储每个学生的信息，那将会是一团糟。这就是面向对象编程可以真正帮助的地方。

让我们使用面向对象编程来重新编写这个例子，以简化问题。面向对象编程的第一步是声明对象的结构。这是通过定义一个类来完成的。类确定了对象的功能。让我们编写一个 Python 类，定义学生对象的结构。

# 类

由于我们将保存学生信息，所以类将被称为`Student`。类是使用`class`关键字定义的，如下所示：

```py
class Student(object):
```

因此，定义了一个名为`Student`的类。每当创建一个新对象时，Python 会在内部调用`__init__()`方法。

这个方法是在类内定义的：

```py
class Student(object): 
    """A Python class to store student information""" 

    def __init__(self, name, address, age): 
        self.name = name 
        self.address = address 
        self.age = age
```

在这个例子中，`__init__`方法的参数包括`name`、`age`和`address`。这些参数被称为**属性**。这些属性使得可以创建一个属于`Student`类的唯一对象。因此，在这个例子中，在创建`Student`类的实例时，需要`name`、`age`和`address`这些属性作为参数。

让我们创建一个属于`Student`类的对象（也称为实例）：

```py
student1 = Student("John Doe", "123 Main Street, Newark, CA", "29")
```

在这个例子中，我们创建了一个属于`Student`类的对象，称为`student1`，其中`John Doe`（姓名）、`29`（年龄）和`123 Main Street, Newark, CA`（地址）是创建对象所需的属性。当我们创建一个属于`Student`类的对象时，通过传递必要的参数（在`Student`类的`__init__()`方法中声明的），`__init__()`方法会自动调用以初始化对象。初始化后，与`student1`相关的信息将存储在对象`student1`下。

现在，属于`student1`的信息可以按以下方式检索：

```py
print(student1.name) 
print(student1.age) 
print(student1.address)
```

现在，让我们创建另一个名为`student2`的对象：

```py
student2 = Student("Jane Doe", "123 Main Street, San Jose, CA", "27")
```

我们创建了两个对象，分别称为`student1`和`student2`。每个对象的属性都可以通过`student1.name`、`student2.name`等方式访问。在没有面向对象编程的情况下，我们将不得不创建变量，如`student1_name`、`student1_age`、`student1_address`、`student2_name`、`student2_age`和`student2_address`等。因此，面向对象编程使得代码模块化。

# 向类添加方法

让我们为我们的`Student`类添加一些方法，以帮助检索学生的信息：

```py
class Student(object): 
    """A Python class to store student information""" 

    def __init__(self, name, age, address): 
        self.name = name 
        self.address = address 
        self.age = age 

    def return_name(self): 
        """return student name""" 
        return self.name 

    def return_age(self): 
        """return student age""" 
        return self.age 

    def return_address(self): 
        """return student address""" 
        return self.address
```

在这个例子中，我们添加了三个方法，分别是`return_name()`、`return_age()`和`return_address()`，它们分别返回属性`name`、`age`和`address`。类的这些方法被称为**可调用属性**。让我们回顾一个快速的例子，我们在其中使用这些可调用属性来打印对象的信息。

```py
student1 = Student("John Doe", "29", "123 Main Street, Newark, CA") 
print(student1.return_name()) 
print(student1.return_age()) 
print(student1.return_address())
```

到目前为止，我们讨论了检索有关学生的信息的方法。让我们在我们的类中包含一个方法，使得学生的信息可以更新。现在，让我们在类中添加另一个方法，使学生可以更新地址：

```py
def update_address(self, address): 
    """update student address""" 
    self.address = address 
    return self.address
```

让我们比较更新地址之前和之后的`student1`对象的地址：

```py
print(student1.address()) 
print(student1.update_address("234 Main Street, Newark, CA"))
```

这将在屏幕上打印以下输出：

```py
    123 Main Street, Newark, CA
 234 Main Street, Newark, CA
```

因此，我们已经编写了我们的第一个面向对象的代码，演示了模块化代码的能力。前面的代码示例可与本章一起下载，名称为`student_info.py`。

# Python 中的文档字符串

在面向对象的示例中，您可能已经注意到了一个用三个双引号括起来的句子：

```py
    """A Python class to store student information"""
```

这被称为**文档字符串**。文档字符串用于记录有关类或方法的信息。文档字符串在尝试存储与方法或类的使用相关的信息时特别有帮助（稍后将在本章中演示）。文档字符串还用于在文件开头存储与应用程序或代码示例相关的多行注释。Python 解释器会忽略文档字符串，它们旨在为其他程序员提供有关类的文档。

同样，Python 解释器会忽略以`#`符号开头的任何单行注释。单行注释通常用于对一块代码做特定的注释。包括结构良好的注释可以使您的代码易读。

例如，以下代码片段通知读者，生成并存储在变量`rand_num`中的随机数在`0`和`9`之间：

```py
# generate a random number between 0 and 9 
rand_num = random.randrange(0,10)
```

相反，提供没有上下文的注释将会让审阅您的代码的人感到困惑：

```py
# Todo: Fix this later
```

当您以后重新访问代码时，很可能您可能无法回忆起需要修复什么。

# self

在我们的面向对象的示例中，每个方法的第一个参数都有一个名为`self`的参数。`self`指的是正在使用的类的实例，`self`关键字用作与类的实例交互的方法中的第一个参数。在前面的示例中，`self`指的是对象`student1`。它相当于初始化对象并访问它如下：

```py
Student(student1, "John Doe", "29", "123 Main Street, Newark, CA") 
Student.return_address(student1)
```

在这种情况下，`self`关键字简化了我们访问对象属性的方式。现在，让我们回顾一些涉及树莓派的 OOP 的例子。

# 扬声器控制器

让我们编写一个 Python 类（下载的`tone_player.py`），它会播放一个音乐音调，指示您的树莓派已完成启动。对于本节，您将需要一个 USB 声卡和一个连接到树莓派的 USB 集线器的扬声器。

让我们称我们的类为`TonePlayer`。这个类应该能够控制扬声器音量，并在创建对象时播放任何传递的文件：

```py
class TonePlayer(object): 
    """A Python class to play boot-up complete tone""" 

    def __init__(self, file_name): 
        self.file_name = file_name
```

在这种情况下，必须传递给`TonePlayer`类要播放的文件的参数。例如：

```py
       tone_player = TonePlayer("/home/pi/tone.wav")
```

我们还需要能够设置要播放音调的音量级别。让我们添加一个执行相同操作的方法：

```py
def set_volume(self, value): 
    """set tone sound volume""" 
    subprocess.Popen(["amixer", "set", "'PCM'", str(value)], 
    shell=False)
```

在`set_volume`方法中，我们使用 Python 的`subprocess`模块来运行调整声音驱动器音量的 Linux 系统命令。

这个类最重要的方法是`play`命令。当调用`play`方法时，我们需要使用 Linux 的`play`命令播放音调声音：

```py
def play(self):
    """play the wav file"""
    subprocess.Popen(["aplay", self.file_name], shell=False)
```

把它全部放在一起：

```py
import subprocess 

class TonePlayer(object): 
    """A Python class to play boot-up complete tone""" 

    def __init__(self, file_name): 
        self.file_name = file_name 

    def set_volume(self, value): 
        """set tone sound volume""" 
        subprocess.Popen(["amixer", "set", "'PCM'", str(value)],
        shell=False) 

    def play(self): 
        """play the wav file""" 
        subprocess.Popen(["aplay", self.file_name], shell=False) 

if __name__ == "__main__": 
    tone_player = TonePlayer("/home/pi/tone.wav") 
    tone_player.set_volume(75) 
    tone_player.play()
```

将`TonePlayer`类保存到您的树莓派（保存为名为`tone_player.py`的文件），并使用来自*freesound*（[`www.freesound.org/people/zippi1/sounds/18872/`](https://www.freesound.org/people/zippi1/sounds/18872/)）等来源的音调声音文件。将其保存到您选择的位置并尝试运行代码。它应该以所需的音量播放音调声音！

现在，编辑`/etc/rc.local`并在文件末尾添加以下行（在`exit 0`行之前）：

```py
python3 /home/pi/toneplayer.py
```

这应该在 Pi 启动时播放一个音调！

# 灯光控制守护程序

让我们回顾另一个例子，在这个例子中，我们使用 OOP 实现了一个简单的守护程序，它在一天中的指定时间打开/关闭灯光。为了能够在预定时间执行任务，我们将使用`schedule`库（[`github.com/dbader/schedule`](https://github.com/dbader/schedule)）。可以按照以下方式安装它：

```py
    sudo pip3 install schedule
```

让我们称我们的类为`LightScheduler`。它应该能够接受开启和关闭灯光的开始和结束时间。它还应该提供覆盖功能，让用户根据需要开启/关闭灯光。假设灯光是使用**PowerSwitch Tail II**（[`www.powerswitchtail.com/Pages/default.aspx`](http://www.powerswitchtail.com/Pages/default.aspx)）来控制的。它的接口如下：

![](img/4616788e-12ba-409b-8fcc-499916c7a9bb.png)树莓派 Zero 与 PowerSwitch Tail II 的接口

以下是创建的`LightSchedular`类：

```py
class LightScheduler(object): 
    """A Python class to turn on/off lights""" 

    def __init__(self, start_time, stop_time): 
        self.start_time = start_time 
        self.stop_time = stop_time 
        # lamp is connected to GPIO pin2.
        self.lights = OutputDevice(2)
```

每当创建`LightScheduler`的实例时，GPIO 引脚被初始化以控制 PowerSwitch Tail II。现在，让我们添加开启/关闭灯光的方法：

```py
def init_schedule(self): 
        # set the schedule 
        schedule.every().day.at(self.start_time).do(self.on) 
        schedule.every().day.at(self.stop_time).do(self.off) 

    def on(self): 
        """turn on lights""" 
        self.lights.on() 

    def off(self): 
        """turn off lights""" 
        self.lights.off()
```

在`init_schedule()`方法中，传递的开始和结束时间被用来初始化`schedule`，以便在指定的时间开启/关闭灯光。

把它放在一起，我们有：

```py
import schedule 
import time 
from gpiozero import OutputDevice 

class LightScheduler(object): 
    """A Python class to turn on/off lights""" 

    def __init__(self, start_time, stop_time): 
        self.start_time = start_time 
        self.stop_time = stop_time 
        # lamp is connected to GPIO pin2.
        self.lights = OutputDevice(2) 

    def init_schedule(self): 
        # set the schedule 
        schedule.every().day.at(self.start_time).do(self.on) 
        schedule.every().day.at(self.stop_time).do(self.off) 

    def on(self): 
        """turn on lights""" 
        self.lights.on() 

    def off(self): 
        """turn off lights""" 
        self.lights.off() 

if __name__ == "__main__": 
    lamp = LightScheduler("18:30", "9:30") 
    lamp.on() 
    time.sleep(50) 
    lamp.off() 
    lamp.init_schedule() 
    while True:
        schedule.run_pending() 
        time.sleep(1)
```

在上面的例子中，灯光被安排在下午 6:30 开启，并在上午 9:30 关闭。一旦工作被安排，程序就会进入一个无限循环，等待任务执行。这个例子可以作为守护进程运行，通过在启动时执行文件（在`/etc/rc.local`中添加一行`light_scheduler.py`）。安排完工作后，它将继续作为后台守护进程运行。

这只是面向初学者的 OOP 及其应用的基本介绍。请参考本书网站以获取更多关于 OOP 的例子。

# 总结

在本章中，我们讨论了列表和 OOP 的优势。我们使用树莓派作为例子的中心，讨论了 OOP 的例子。由于本书主要面向初学者，我们决定在讨论例子时坚持 OOP 的基础知识。书中还有一些超出范围的高级方面。我们让读者通过本书网站上提供的其他例子来学习高级概念。
