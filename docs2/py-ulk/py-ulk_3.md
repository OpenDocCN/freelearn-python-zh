# 第三章 函数和实用工具

在了解对象之间是如何相互链接之后，让我们来看看在语言中执行代码的手段——函数。我们将讨论如何使用各种组合来定义和调用函数。然后，我们将介绍一些在日常生活中编程中非常有用的实用工具。我们将涵盖以下主题：

+   定义函数

+   装饰可调用对象

+   实用工具

# 定义函数

**关键点 1：如何定义函数。**

函数用于将一组指令和执行特定任务的逻辑组合在一起。因此，我们应该让函数执行一个特定的任务，并选择一个能给我们提示该任务的名称。如果一个函数很重要并且执行复杂操作，我们应该始终为此函数添加文档字符串，这样我们以后就可以轻松地访问和修改此函数。

在定义函数时，我们可以定义以下内容：

1.  位置参数（简单按照位置传递对象），如下所示：

    ```py
    >>> def foo(a,b):
    ...   print(a,b)
    ... 
    >>> foo(1,2)
    1 2
    ```

1.  默认参数（如果没有传递值，则使用默认值），如下所示：

    ```py
    >>> def foo(a,b=3):
    ...    print(a,b)
    ... 
    >>> foo(3)  
    3 3
    >>> foo(3,4)
    3 4
    ```

1.  关键字参数（必须以位置或关键字参数的形式传递），如下所示：

    ```py
    >>> def  foo(a,*,b):
    ...   print(a,b)
    ... 
    >>> foo(2,3)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: foo() takes 1 positional argument but 2 were given
    >>> foo(1,b=4)
    1 4
    ```

1.  参数列表，如下所示：

    ```py
    >>> def foo(a,*pa):
    ...   print(a,pa)
    ... 
    >>> foo(1)
    1 ()
    >>> foo(1,2)
    1 (2,)
    >>> foo(1,2,3)
    1 (2, 3)
    ```

1.  关键字参数字典，如下所示：

    ```py
    >>> def foo(a,**kw):
    ...   print(a,kw)
    ... 
    >>> foo(2)     
    2 {}
    >>> foo(2,b=4)
    2 {'b': 4}
    >>> foo(2,b=4,v=5)
    2 {'b': 4, 'v': 5}
    ```

    当函数被调用时，这是参数传递的方式：

1.  所有传递的位置参数都被消耗。

1.  如果函数接受一个参数列表，并且在第一步之后还有更多的位置参数传递，那么其余的参数将收集在一个参数列表中：

    ```py
    >>> def foo1(a,*args):
    ...   print(a,args)
    ... 
    >>> def foo2(a,):
    ...   print(a)
    ... 
    >>> foo1(1,2,3,4)
    1 (2, 3, 4)
    >>> foo2(1,2,3,4)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: foo2() takes 1 positional argument but 4 were given
    ```

1.  如果传递的位置参数少于定义的位置参数，则使用传递的关键字参数作为位置参数的值。如果没有找到位置参数的关键字参数，我们将得到一个错误：

    ```py
    >>> def foo(a,b,c):
    ...   print(a,b,c)
    ... 
    >>> foo(1,c=3,b=2)
    1 2 3
    >>> foo(1,b=2)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: foo() missing 1 required positional argument: 'c'
    ```

1.  传递的关键字变量仅用于关键字参数：

    ```py
    >>> def foo(a,b,*,c):           
    ...   print(a,b,c)
    ... 
    >>> foo(1,2,3)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: foo() takes 2 positional arguments but 3 were given
    >>> foo(1,2,c=3)
    1 2 3
    >>> foo(c=3,b=2,a=1)
    1 2 3
    ```

1.  如果还有更多的关键字参数，并且被调用的函数接受一个关键字参数列表，那么其余的关键字参数将作为关键字参数列表传递。如果函数没有接受关键字参数列表，我们将得到一个错误：

    ```py
    >>> def foo(a,b,*args,c,**kwargs):
    ...   print(a,b,args,c,kwargs)
    ... 
    >>> foo(1,2,3,4,5,c=6,d=7,e=8)        
    1 2 (3, 4, 5) 6 {'d': 7, 'e': 8}
    ```

    这里有一个示例函数，它使用了前面所有组合：

    ```py
    >>> def foo(a,b,c=2,*pa,d,e=5,**ka):
    ...   print(a,b,c,d,e,pa,ka)
    ... 
    >>> foo(1,2,d=4)
    1 2 2 4 5 () {}
    >>> foo(1,2,3,4,5,d=6,e=7,g=10,h=11)
    1 2 3 6 7 (4, 5) {'h': 11, 'g': 10}
    ```

# 装饰可调用对象

**关键点 2：改变可调用对象的行为。**

装饰器是可调用对象，它们用其他对象替换原始的可调用对象。在这种情况下，因为我们用另一个对象替换了一个可调用对象，所以我们主要希望被替换的对象仍然是可调用的。

语言提供了易于实现的语法，但首先，让我们看看我们如何手动完成这个任务：

```py
>>> def wrap(func):
...     def newfunc(*args):
...         print("newfunc",args)
...     return newfunc
...
>>> def realfunc(*args):
...     print("real func",args)
...
>>>
>>> realfunc = wrap(realfunc)
>>>
>>> realfunc(1,2,4)
('newfunc', (1, 2, 4))
```

使用装饰器语法，变得很容易。从前面的代码片段中获取 wrap 和`newfunc`的定义，我们得到这个：

```py
>>> @wrap
... def realfunc(args):
...     print("real func",args)
...
>>> realfunc(1,2,4)
('newfunc', (1, 2, 4))
```

要在装饰器函数中存储某种状态，比如说使装饰器更有用，并适用于更广泛的代码库，我们可以使用闭包或类实例作为装饰器。在第二章中，我们了解到闭包可以用来存储状态；让我们看看我们如何利用它们在装饰器中存储信息。在这个片段中，`deco`函数是替换添加函数的新函数。这个函数的闭包中有一个前缀变量。这个变量可以在装饰器创建时注入：

```py
>>> def closure_deco(prefix):
...     def deco(func):
...         return lambda x:x+prefix
...     return deco
... 
>>> @closure_deco(2)
... def add(a):
...     return a+1
... 
>>> add(2)
4
>>> add(3)
5
>>> @closure_deco(3)
... def add(a):
...     return a+1
... 
>>> add(2)
5
>>> add(3)
6
```

我们也可以用类来做同样的事情。在这里，我们在类的实例上保存状态：

```py
>>> class Deco:
...     def __init__(self,addval):
...         self.addval = addval
...     def __call__(self, func):
...         return lambda x:x+self.addval
... 
>>> @Deco(2)
... def add(a):
...     return a+1
... 
>>> add(1)
3
>>> add(2)
4
>>> @Deco(3)
... def add(a):
...     return a+1
... 
>>> add(1)
4
>>> add(2)
5
```

由于装饰器作用于任何可调用对象，它同样适用于方法和类定义，但在这样做的时候，我们应该考虑被装饰的方法隐式传递的不同参数。让我们先考虑一个简单的被装饰方法如下：

```py
>>> class K:
...     def do(*args):
...         print("imethod",args)
...
>>> k = K()
>>> k.do(1,2,3)
('imethod', (<__main__.K instance at 0x7f12ea070bd8>, 1, 2, 3))
>>>
>>> # using a decorator on methods give similar results
...
>>> class K:
...     @wrap
...     def do(*args):
...         print("imethod",args)
...
>>> k = K()
>>> k.do(1,2,3)
('newfunc', (<__main__.K instance at 0x7f12ea070b48>, 1, 2, 3))
```

由于被替换的函数成为类本身的方法，这工作得很好。对于静态方法和类方法来说，这就不成立了。它们使用描述符来调用方法，因此，它们的行性行为与装饰器不匹配，返回的函数表现得像一个简单的方法。我们可以通过首先检查被覆盖的函数是否是描述符，如果是，则调用它的`__get__`方法来解决这个问题：

```py
>>> class K:
...     @wrap
...     @staticmethod
...     def do(*args):
...         print("imethod",args)
...     @wrap
...     @classmethod
...     def do2(*args):
...         print("imethod",args)
...
>>> k = K()
>>> k.do(1,2,3)
('newfunc', (<__main__.K instance at 0x7f12ea070cb0>, 1, 2, 3))
>>> k.do2(1,2,3)
('newfunc', (<__main__.K instance at 0x7f12ea070cb0>, 1, 2, 3))
```

我们也可以通过在任意其他装饰器之上使用静态方法和类方法装饰器来轻松实现这一点。这使得通过属性查找找到的实际方法看起来像描述符，并且对于`staticmethod`和`classmethod`，正常执行发生。

这工作得很好，如下所示：

```py
>>> class K:
...     @staticmethod
...     @wrap
...     def do(*args):
...         print("imethod",args)
...     @classmethod
...     @wrap
...     def do2(*args):
...         print("imethod",args)
...
>>> k = K()
>>> k.do(1,2,3)
('newfunc', (1, 2, 3))
>>> k.do2(1,2,3)
('newfunc', (<class __main__.K at 0x7f12ea05e1f0>, 1, 2, 3))
```

我们可以使用装饰器来处理类，因为类本质上是一种可调用对象。因此，我们可以使用装饰器来改变实例创建过程，以便当我们调用类时，我们得到一个实例。类对象将被传递到装饰器，然后装饰器可以用另一个可调用对象或类来替换它。在这里，`cdeco`装饰器正在传递一个新的类来替换`cls`：

```py
>>> def cdeco(cls):
...     print("cdecorator working")
...     class NCls:
...         def do(*args):
...             print("Ncls do",args)
...     return NCls
...
>>> @cdeco
... class Cls:
...     def do(*args):
...         print("Cls do",args)
...
cdecorator working
>>> b = Cls()
>>> c = Cls()
>>> c.do(1,2,3)
('Ncls do', (<__main__.NCls instance at 0x7f12ea070cf8>, 1, 2, 3))
```

通常，我们使用它来更改属性，并为类定义添加新属性。我们也可以用它来将类注册到某个注册表中，等等。在下面的代码片段中，我们检查类是否有 do 方法。如果我们找到一个，我们就用`newfunc`来替换它：

```py
>>> def cdeco(cls):
...     if hasattr(cls,'do'):
...         cls.do = wrap(cls.do)
...     return cls
...
>>> @cdeco
... class Cls:
...     def do(*args):
...         print("Cls do",args)
...
>>> c = Cls()
>>> c.do(1,2,3)
('newfunc', (<__main__.Cls instance at 0x7f12ea070cb0>, 1, 2, 3))
```

# 实用工具

**关键 3：通过推导式轻松迭代。**

我们有各种语法和实用工具来有效地迭代迭代器。推导式在迭代器上工作，并提供另一个迭代器作为结果。它们是用原生 C 实现的，因此，它们比循环更快。

我们有列表、字典和集合推导式，分别产生列表、字典和集合作为结果。此外，迭代器避免了在循环中声明额外的变量：

```py
>>> ll = [ i+1 for i in range(10)]
>>> print(type(ll),ll)
<class 'list'> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> ld = { i:'val'+str(i) for i in range(10) }
>>> print(type(ld),ld)
<class 'dict'> {0: 'val0', 1: 'val1', 2: 'val2', 3: 'val3', 4: 'val4', 5: 'val5', 6: 'val6', 7: 'val7', 8: 'val8', 9: 'val9'}
>>> ls = {i for i in range(10)}
>>> print(type(ls),ls)
<class 'set'> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

生成器表达式创建生成器，可以用来为迭代产生生成器。要实现生成器，我们使用它来创建`set`、`dict`或`list`：

```py
>>> list(( i for i in range(10)))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> dict(( (i,'val'+str(i)) for i in range(10)))
{0: 'val0', 1: 'val1', 2: 'val2', 3: 'val3', 4: 'val4', 5: 'val5', 6: 'val6', 7: 'val7', 8: 'val8', 9: 'val9'}
>>> set(( i for i in range(10)))
{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

生成器对象不会一次性计算可迭代对象的所有值，而是在被循环请求时逐个计算。这节省了内存，我们可能对使用整个可迭代对象不感兴趣。生成器不是可以到处使用的银弹。它们并不总是导致性能提升。这取决于消费者和生成一个序列的成本：

```py
>>> def func(val):
...     for i in (j for j in range(1000)):
...         k = i + 5
... 
>>> def func_iter(val):
...     for i in [ j for j in range(1000)]:
...         k = i + 5
... 
>>> timeit.timeit(stmt="func(1000)", globals={'func':func_iter},number=10000)
0.6765081569974427
>>> timeit.timeit(stmt="func(1000)", globals={'func':func},number=10000)
0.838760247999744
```

**关键 4：一些有用的工具。**

`itertools` 工具是一个很好的模块，包含许多对迭代有帮助的函数。以下是我最喜欢的几个：

+   **itertools.chain(*iterable)**：这从一个可迭代对象的列表返回一个单一的迭代器。首先，第一个可迭代对象的所有元素都被耗尽，然后是第二个，依此类推，直到所有可迭代对象都被耗尽：

    ```py
    >>> list(itertools.chain(range(3),range(2),range(4)))
    [0, 1, 2, 0, 1, 0, 1, 2, 3]
    >>> 
    ```

+   **itertools.cycle**：这会创建一个迭代器的副本，并无限期地重复播放结果：

    ```py
    >>> cc = cycle(range(4))
    >>> cc.__next__()
    0
    >>> cc.__next__()
    1
    >>> cc.__next__()
    2
    >>> cc.__next__()
    3
    >>> cc.__next__()
    0
    >>> cc.__next__()
    1
    >>> cc.__next__()
    ```

+   **itertools.tee(iterable,number)**：这从一个单一的迭代器返回 `n` 个独立的迭代器：

    ```py
    >>> i,j = tee(range(10),2)
    >>> i
    <itertools._tee object at 0x7ff38e2b2ec8>
    >>> i.__next__()
    0
    >>> i.__next__()
    1
    >>> i.__next__()
    2
    >>> j.__next__()
    0
    ```

+   **functools.lru_cache**：这个装饰器使用记忆功能。它保存映射到参数的结果。因此，它对于加速具有相似参数且结果不依赖于时间或状态的函数非常有用：

    ```py
    In [7]: @lru_cache(maxsize=None)
    def fib(n):
        if n<2:
            return n
        return fib(n-1) + fib(n-2)
       ...: 

    In [8]: %timeit fib(30)
    10000000 loops, best of 3: 105 ns per loop

    In [9]:                         
    def fib(n):
        if n<2:
            return n
        return fib(n-1) + fib(n-2)
       ...: 

    In [10]: %timeit fib(30)
    1 loops, best of 3: 360 ms per loop
    ```

+   **functools.wraps**：我们刚刚看到了如何创建装饰器以及如何包装函数。装饰器返回的函数保留了其名称和属性，如文档字符串，这对用户或开发者来说可能没有帮助。我们可以使用这个装饰器将返回的函数与装饰的函数匹配。以下代码片段展示了它的用法：

    ```py
    >>> def deco(func):
    ...     @wraps(func) # this will update wrapper to match func
    ...     def wrapper(*args, **kwargs):
    ...         """i am imposter"""
    ...         print("wrapper")
    ...         return func(*args, **kwargs)
    ...     return wrapper
    ... 
    >>> @deco
    ... def realfunc(*args,**kwargs):
    ...     """i am real function """
    ...     print("realfunc",args,kwargs)
    ... 
    >>> realfunc(1,2)
    wrapper
    realfunc (1, 2) {}
    >>> print(realfunc.__name__, realfunc.__doc__)
    realfunc i am real function 
    ```

+   **Lambda 函数**：这些函数是简单的匿名函数。Lambda 函数不能有语句或注解。它们在创建 GUI 编程中的闭包和回调时非常有用：

    ```py
    >>> def log(prefix):
    ...     return lambda x:'%s : %s'%(prefix,x)
    ... 
    >>> err = log("error")
    >>> warn = log("warn")
    >>> 
    >>> print(err("an error occurred"))
    error : an error occurred
    >>> print(warn("some thing is not right"))
    warn : some thing is not right
    ```

    有时，lambda 函数使代码更容易理解。

    以下是一个使用迭代技术和 lambda 函数创建菱形图案的小程序：

    ```py
    >>> import itertools
    >>> af = lambda x:[i for i in itertools.chain(range(1,x+1),range(x-1,0,-1))]
    >>> output = '\n'.join(['%s%s'%('  '*(5-i),' '.join([str(j) for j in af(i)])) for i in af(5)])
    >>> print(output)
            1
          1 2 1
        1 2 3 2 1
      1 2 3 4 3 2 1
    1 2 3 4 5 4 3 2 1
      1 2 3 4 3 2 1
        1 2 3 2 1
          1 2 1
            1
    ```

# 摘要

在本章中，我们介绍了如何定义函数并将参数传递给它们。然后，我们详细讨论了装饰器；装饰器在框架中非常受欢迎。在结尾部分，我们收集了 Python 中可用的各种工具，这使得我们的编码工作变得稍微容易一些。

在下一章中，我们将讨论算法和数据结构。
