# 第二章. 命名空间和类

在上一章中，我们介绍了对象的工作原理。在本章中，我们将探讨对象如何通过引用提供给代码，特别是命名空间的工作原理、模块是什么以及它们是如何导入的。我们还将涵盖与类相关的话题，例如语言协议、MRO 和抽象类。我们将讨论以下主题：

+   命名空间

+   导入和模块

+   类的多重继承，MRO，super

+   协议

+   抽象类

# 对象引用的工作原理 - 命名空间

**关键 1：对象之间的相互关系。**

范围是名称在代码块中的可见性。命名空间是从名称到对象的映射。命名空间对于保持本地化和避免名称冲突非常重要。每个模块都有一个全局命名空间。模块在其`__dict__`属性中存储从变量名到对象的映射，这是一个普通的 Python 字典，并包含有关重新加载、包信息等信息。

每个模块的全局命名空间都有一个对内置模块的隐式引用；因此，内置模块中的对象总是可用的。我们还可以在主脚本中导入其他模块。当我们使用`import module name`语法时，在当前模块的全局命名空间中创建了一个模块名到模块对象的映射。对于像`import modname as modrename`这样的导入语句，创建了一个新名称到模块对象的映射。

当程序开始时，我们始终处于`__main__`模块的全局命名空间中，因为它是导入所有其他模块的模块。当我们从一个模块导入一个变量时，只在全局命名空间中为该变量创建一个条目，指向引用的对象。现在有趣的是，如果这个变量引用了一个函数对象，并且如果这个函数使用了一个全局变量，那么这个变量将在定义该函数的模块的全局命名空间中搜索，而不是在我们导入该函数的模块中。这是可能的，因为函数有`__globals__`属性，它指向其`__dict__`模块，或者简而言之，其模块命名空间。

所有已加载和引用的模块都缓存在`sys.modules`中。所有导入的模块都是指向`sys.modules`中对象的名称。让我们这样定义一个名为`new.py`的新模块：

```py
k = 10 
def foo():
    print(k)
```

通过在交互会话中导入此模块，我们可以看到全局命名空间是如何工作的。当此模块被重新加载时，其命名空间字典被更新，而不是重新创建。因此，如果你将任何新内容从模块外部附加到它，它将存活下来：

```py
>>> import importlib
>>> import new
>>> from new import foo
>>> import sys
>>> foo()
10
>>> new.foo()
10
>>> foo.__globals__ is sys.modules['new'].__dict__ # dictionary used by namespace and function attribute __globals__ is indeed same
True
>>> foo.__globals__['k'] = 20  # changing global namespace dictionary
>>> new.do   #attribute is not defined in the module
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'new' has no attribute 'do'
>>> foo.__globals__['do'] = 22 #we are attaching attribute to module from outside the module
>>> new.do
22
>>> foo()  # we get updated value for global variable
20
>>> new.foo()
20
>>> importlib.reload(new) #reload repopulates old modules dictionary
<module 'new' from '/tmp/new.py'>
>>> new.do #it didn't got updated as it was populated from outside.
22
>>> new.foo() #variables updated by execution of code in module are updated
10
>>>
```

如果我们在运行时使用定义在不同模块中的函数来组合一个类，例如使用元类或类装饰器，这可能会带来惊喜，因为每个函数可能使用不同的全局命名空间。

局部变量简单且按预期工作。每个函数调用都会获得自己的变量副本。非局部变量使得在当前代码块中可以访问外部作用域（非全局命名空间）中定义的变量。在下面的代码示例中，我们可以看到如何在嵌套函数中引用变量。

代码块能够引用在封装作用域中定义的变量。因此，如果一个变量不是在函数中定义的，而是在封装函数中定义的，我们就能获取它的值。如果我们在外部作用域中引用了一个变量，然后在代码块中为这个变量赋值，它将使解释器在寻找正确的变量时感到困惑，并且我们会从当前局部作用域中获取值。如果我们为变量赋值，它默认为局部变量。我们可以使用非局部关键字指定我们想要使用封装变量：

```py
>>> #variable in enclosing scope can be referenced any level deep
... 
>>> def f1():
...     v1 = "ohm"
...     def f2():
...         print("f2",v1)
...         def f3():
...             print("f3",v1)
...         f3()
...     f2()
... 
>>> f1()
f2 ohm
f3 ohm
>>> 
>>> #variable can be made non-local (variable in outer scopes) skipping one level of enclosing scope
... 
>>> def f1():
...     v1 = "ohm"
...     def f2():
...         print("f2",v1)
...         def f3():
...             nonlocal v1
...             v1 = "mho"
...             print("f3",v1)
...         f3()
...         print("f2",v1)
...     f2()
...     print("f1",v1)
... 
>>> f1()
f2 ohm
f3 mho
f2 mho
f1 mho
>>> 
>>> 
>>> #global can be specified at any level of enclosed function
... 
>>> v2 = "joule"
>>> 
>>> def f1():
...     def f2():
...         def f3():
...             global v2
...             v2 = "mho"
...             print("f3",v2)
...         f3()
...         print("f2",v2)
...     f2()
...     print("f1",v2)
... 
>>> f1()
f3 mho
f2 mho
f1 mho
```

由于在局部命名空间中搜索变量时无需进行字典查找，因此在函数中具有少量变量的函数中查找变量比在全局命名空间中搜索要快。类似地，如果我们将函数局部命名空间中引用的对象拉入函数块中，我们将会得到一点速度提升：

```py
In [6]: def fun():
   ...:     localsum = sum
   ...:     return localsum(localsum((a,a+1)) for a in range(1000))
   ...: 

In [8]: def fun2():
   ...:     return sum(sum((a,a+1)) for a in range(1000))
   ...: 

In [9]: %timeit fun2()
1000 loops, best of 3: 1.07 ms per loop

In [11]: %timeit fun()
1000 loops, best of 3: 983 µs per loop
```

# 带状态的函数 – 闭包

**关键 2：创建廉价的记忆状态函数。**

闭包是一个可以访问已执行完毕的封装作用域中变量的函数。这意味着引用的对象会一直保持活跃状态，直到函数在内存中。这种设置的主要用途是轻松保留一些状态，或者创建依赖于初始设置的专用函数：

```py
>>> def getformatter(start,end):
...     def formatter(istr):
...         print("%s%s%s"%(start,istr,end))
...     return formatter
... 
>>> formatter1 = getformatter("<",">")
>>> formatter2 = getformatter("[","]")
>>> 
>>> formatter1("hello")
<hello>
>>> formatter2("hello")
[hello]
>>> formatter1.__closure__[0].cell_contents
'>'
>>> formatter1.__closure__[1].cell_contents
'<'
```

我们可以通过创建一个类并使用实例对象来保存状态来实现同样的功能。闭包的优点在于变量存储在`__closure__`元组中，因此它们可以快速访问。与类相比，创建闭包所需的代码更少：

```py
>>> def formatter(st,en):
...     def fmt(inp):
...             return "%s%s%s"%(st,inp,en)
...     return fmt
... 
>>> fmt1 = formatter("<",">")
>>> fmt1("hello")
'<hello>'
>>> timeit.timeit(stmt="fmt1('hello')",
... number=1000000,globals={'fmt1':fmt1})
0.3326794120075647
>>> class Formatter:
...     def __init__(self,st,en):
...             self.st = st
...             self.en = en
...     def __call__(self, inp):
...             return "%s%s%s"%(self.st,inp,self.en)
... 
>>> fmt2 = Formatter("<",">")
>>> fmt2("hello")
'<hello>'
>>> timeit.timeit(stmt="fmt2('hello')",
... number=1000000,globals={'fmt2':fmt2})
0.5502702980011236
```

标准库中有一个这样的函数可用，名为`partial`，它利用闭包创建一个新函数，该函数始终使用一些预定义的参数调用：

```py
>>> import functools
>>> 
>>> def foo(*args,**kwargs):
...     print("foo with",args,kwargs)    
... 
>>> pfoo = functools.partial(foo,10,20,v1=23)
>>> 
>>> foo(1,2,3,array=1)
foo with (1, 2, 3) {'array': 1}
>>> pfoo()
foo with (10, 20) {'v1': 23}
>>> pfoo(30,40,array=12)
foo with (10, 20, 30, 40) {'v1': 23, 'array': 12}
```

# 理解导入和模块

**关键 3：为模块创建自定义加载器。**

导入语句获取当前模块命名空间中其他模块对象的引用。它包括搜索模块、执行代码以创建模块对象、更新缓存（`sys.modules`）、更新模块命名空间，以及创建对新导入的模块的引用。

内置的`__import__`函数搜索并执行模块以创建模块对象。`importlib`库有其实现，并且它还提供了一个可定制的接口给导入机制。各种类相互作用以完成任务。`__import__`函数应该返回一个模块对象。例如，在以下示例中，我们创建了一个模块查找器，它检查在构造时作为参数给出的任何路径中的模块。在这里，应该在给定路径处有一个名为`names.py`的空文件。我们已经加载了该模块，然后将其模块对象插入到`sys.modules`中，并添加了一个函数到该模块的全局命名空间：

```py
import os
import sys

class Spec:
    def __init__(self,name,loader,file='None',path=None,
                 cached=None,parent=None,has_location=False):
        self.name = name
        self.loader = loader
        self.origin = file
        self.submodule_search_locations = path
        self.cached = cached
        self.has_location = has_location

class Finder:
    def __init__(self, path):
        self.path = path

    def find_spec(self,name,path,target):
        print("find spec name:%s path:%s target:%s"%(name,path,target))
        return Spec(name,self,path)

    def load_module(self, fullname):
        print("loading module",fullname)
        if fullname+'.py' in os.listdir(self.path):
            import builtins
            mod = type(os)
            modobject = mod(fullname)
            modobject.__builtins__ = builtins
            def foo():
                print("hii i am foo")
            modobject.__dict__['too'] = foo
            sys.modules[fullname] = modobject
            modobject.__spec__ = 'asdfasfsadfsd'
            modobject.__name__ = fullname
            modobject.__file__ = 'aruns file'
            return modobject

sys.meta_path.append(Finder(r'/tmp'))
import notes
notes.too()

Output:
find spec name:notes path:None target:None
loading module notes
hii i am foo
```

## 自定义导入

如果模块有一个`__all__`属性，那么只有在这个属性中指定的可迭代名称将从模块导入`*`。假设我们创建了一个名为`mymod.py`的模块，如下所示：

```py
__all__ = ('hulk','k')

k = 10
def hulk():
    print("i am hulk")

def spidey():
    print("i am spidey")
```

我们无法从`mymod`导入`spidey`，因为它不包括在`__all__`中：

```py
>>> from mymod import *
>>> 
>>> hulk()
i am hulk
>>> k
10
>>> spidey()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'spidey' is not defined
```

# 类继承

我们已经讨论了实例和类的创建方式。我们也讨论了如何在类中访问属性。让我们深入了解这是如何适用于多个基类的。当类型搜索实例的属性存在时，如果类型从多个类继承，它们都会被搜索。有一个定义好的模式（**方法解析顺序**（**MRO**））。这个顺序在多重继承和菱形继承的情况下确定方法时起着重要作用。

## 方法解析顺序

**关键点 4：理解 MRO。**

方法以预定义的方式搜索类的基类。这个序列或顺序被称为方法解析顺序。在 Python 3 中，当在类中找不到属性时，它会在该类的所有基类中搜索。如果属性仍然没有找到，就会搜索基类的基类。这个过程会一直进行，直到我们耗尽所有基类。这类似于如果我们必须提问，我们首先会去找我们的父母，然后是叔叔和阿姨（同一级别的基类）。如果我们仍然得不到答案，我们会去找祖父母。以下代码片段显示了这一序列：

```py
>>> class GrandParent:
...     def do(self,):
...         print("Grandparent do called")
...
>>> class Father(GrandParent):
...     def do(self,):
...         print("Father do called")
...
>>> class Mother(GrandParent):
...     def do(self,):
...         print("Mother do called")
...
>>> class Child(Father, Mother):
...     def do(self,):
...         print("Child do called")
...
>>> c = Child() # calls method in Class
>>> c.do()
Child do called
>>> del Child.do # if method is not defined it is searched in bases
>>> c.do()  #Father's method
Father do called
>>> c.__class__.__bases__ =  (c.__class__.__bases__[1],c.__class__.__bases__[0]) #we swap bases order
>>> c.do() #Mothers's method
Mother do called
>>> del Mother.do
>>> c.do() #Fathers' method
Father do called
>>> del Father.do
>>> c.do()
Grandparent do called
```

## 超类的超能力

**关键点 6：在没有超类定义的情况下获取超类的方法。**

我们通常创建子类来专门化方法或添加新的功能。我们可能需要添加一些功能，这 80%与基类中的功能相同。然后，在子类的新方法中添加额外的功能并调用基类的方法将是自然的。要调用超类的方法，我们可以使用其类名来访问该方法，或者像这样使用 super：

```py
>>> class GrandParent:
...     def do(self,):
...         print("Grandparent do called")
...
>>> class Father(GrandParent):
...     def do(self,):
...         print("Father do called")
...
>>> class Mother(GrandParent):
...     def do(self,):
...         print("Mother do called")
...
>>> class Child(Father, Mother):
...     def do(self,):
...         print("Child do called")
...
>>> c = Child()
>>> c.do()
Child do called
>>> class Child(Father, Mother):
...     def do(self,):
...         print("Child do called")
...         super().do()
...
>>> c = Child()
>>> c.do()
Child do called
Father do called
>>> print("Father and child super calling")
Father and child super calling
>>> class Father(GrandParent):
...     def do(self,):
...         print("Father do called")
...         super().do()
...
>>> class Child(Father, Mother):
...     def do(self,):
...         print("Child do called")
...         super().do()
...
>>> c = Child()
>>> c.do()
Child do called
Father do called
Mother do called
>>> print("Father and Mother super calling")
Father and Mother super calling
>>> class Mother(GrandParent):
...     def do(self,):
...         print("Mother do called")
...         super().do()
...
>>> class Father(GrandParent):
...     def do(self,):
...         print("Father do called")
...         super().do()
...
>>> class Child(Father, Mother):
...     def do(self,):
...         print("Child do called")
...         super().do()
...
>>> c = Child()
>>> c.do()
Child do called
Father do called
Mother do called
Grandparent do called
>>> print(Child.__mro__)
(<class '__main__.Child'>, <class '__main__.Father'>, <class '__main__.Mother'>, <class '__main__.GrandParent'>, <class 'object'>)
```

# 在类中使用语言协议

所有提供特定功能的对象都有一些便于该行为的方法，例如，你可以创建一个类型为 worker 的对象，并期望它具有`submit_work(function, kwargs)`和`_completed()`方法。现在，我们可以期望所有具有这些方法的对象都可以在任何应用程序部分用作工作者。同样，Python 语言定义了一些方法，这些方法用于向对象添加特定的功能。如果一个对象拥有这些方法，它就具有那种功能。

我们将讨论两个非常重要的协议：迭代协议和上下文协议。

## 迭代协议

对于迭代协议，对象必须具有`__iter__`方法。如果一个对象具有它，我们就可以在任何使用迭代器对象的地方使用该对象。当我们在一个`for`循环中使用迭代器对象或将它传递给内置的`iter`函数时，我们就是在调用它的`__iter__`方法。此方法返回另一个或相同的对象，该对象负责在迭代过程中维护索引，并且从`__iter__`返回的对象必须有一个`__next__`方法，该方法提供序列中的下一个值，并在序列结束时引发`StopIteration`异常。在以下代码片段中，`BooksIterState`对象帮助保留用于迭代的索引。如果书籍的`__iter__`方法返回自身，那么在从两个循环访问对象时维护状态索引将会很困难：

```py
>>> class BooksIterState:
...     def __init__(self, books):
...             self.books = books
...             self.index = 0
...     def __next__(self,):
...             if self.index >= len(self.books._data):
...                     raise StopIteration
...             else:
...                     tmp = self.books._data[self.index]
...                     self.index += 1
...                     return tmp
... 
>>> class Books:
...     def __init__(self, data):
...             self._data = data
...     def __iter__(self,):
...             return BooksIterState(self)
... 
>>> ii = iter(Books(["don quixote","lord of the flies","great expectations"]))
>>> next(ii)
'don quixote'
>>> for i in Books(["don quixote","lord of the flies","great expectations"]):
...     print(i)
... 
don quixote
lord of the flies
great expectations
>>> next(ii)
'lord of the flies'
>>> next(ii)
'great expectations'
>>> next(ii)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 7, in __next__
StopIteration
>>> 
```

## 上下文管理器协议

提供执行上下文的对象类似于 try finally 语句。如果一个对象具有`__enter__`和`__exit__`方法，那么这个对象可以用作 try finally 语句的替代。最常见的用途是释放锁和资源，或者刷新和关闭文件。在以下示例中，我们创建一个`Ctx`类作为上下文管理器：

```py
>>> class Ctx:
...     def __enter__(*args):
...         print("entering")
...         return "do some work"
...     def __exit__(self, exception_type,
...                  exception_value,
...                  exception_traceback):
...         print("exit")
...         if exception_type is not None:
...             print("error",exception_type)
...         return True
... 
>>> with Ctx() as k:
...     print(k)
...     raise KeyError
... 
entering
do some work
exit
error <class 'KeyError'>
```

我们还可以使用`contextlib`模块的`contextmanager`装饰器轻松创建类似于以下代码所示的上下文管理器：

```py
>>> import contextlib
>>> @contextlib.contextmanager
... def ctx():
...     try:
...         print("start")
...         yield "so some work"
...     except KeyError:
...         print("error")
...     print("done")
... 
>>> with ctx() as k:
...     print(k)
...     raise KeyError
... 
start
so some work
error
done
```

有其他一些方法也应该了解，例如`__str__`、`__add__`、`__getitem__`等等，这些方法定义了对象的各种功能。在语言参考的`datamodel.html`中有一个它们的列表。你应该至少阅读一次，以了解可用的方法。以下是链接：[`docs.python.org/3/reference/datamodel.html#special-method-names`](https://docs.python.org/3/reference/datamodel.html#special-method-names)。

# 使用抽象类

**关键 6：为一致性创建接口。**

抽象类可以通过标准 `abc` 库包获得。它们在定义接口和通用功能方面非常有用。这些抽象类可以部分实现接口，并通过将方法定义为抽象的，使得其余的 API 对子类来说是强制性的。此外，通过简单地注册，可以将类转换为抽象类的子类。这些类对于使一组类符合单一接口非常有用。以下是使用它们的示例。在这里，工作类定义了一个接口，包含两个方法：do 和 `is_busy`，每种类型的工作者都必须实现。`ApiWorker` 是这个接口的实现：

```py
>>> from abc import ABCMeta, abstractmethod
>>> class Worker(metaclass=ABCMeta):
...     @abstractmethod
...     def do(self, func, args, kwargs):
...         """ work on function """
...     @abstractmethod
...     def is_busy(self,):
...         """ tell if busy """
...
>>> class ApiWorker(Worker):
...     def __init__(self,):
...         self._busy = False
...     def do(self, func, args=[], kwargs={}):
...         self._busy = True
...         res = func(*args, **kwargs)
...         self._busy = False
...         return res
...     def is_busy(self,):
...         return self._busy
...
>>> apiworker = ApiWorker()
>>> print(apiworker.do(lambda x: x + 1, (1,)))
2
>>> print(apiworker.is_busy())
False
```

# 摘要

现在，我们已经了解了如何操作命名空间，以及如何创建自定义模块加载类。我们可以使用多重继承来创建混合类，其中每个混合类都为子类提供新的功能。上下文管理器和迭代器协议是非常有用的结构，可以创建干净的代码。我们创建了抽象类，可以帮助我们为类设置 API 合同。

在下一章中，我们将介绍从标准 Python 安装中可用的函数和实用工具。
