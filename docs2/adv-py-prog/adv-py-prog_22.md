# 第十九章：*第十九章*：适配器模式

在前面的章节中，我们介绍了创建型模式，这些是**面向对象编程**（**OOP**）模式，帮助我们处理对象创建过程。我们接下来要介绍的下一个模式类别是**结构型设计模式**。

结构型设计模式提出了一种组合对象以创建新功能的方法。我们将要介绍的第一个模式是**适配器**模式。

在本章中，我们将讨论以下主题：

+   理解适配器模式

+   现实世界示例

+   用例

+   实现

到本章结束时，你将知道如何使用这种设计模式来创建接口，这些接口可以帮助那些否则无法通信的应用层。

# 技术要求

本章的代码文件可以通过以下链接访问：[`github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter19`](https://github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter19).

# 理解适配器模式

适配器模式是一种结构型设计模式，它帮助我们使两个不兼容的接口兼容。*这究竟意味着什么？*如果我们有一个旧的组件，我们想在新的系统中使用它，或者我们想在旧系统中使用的新组件，这两个组件很少能够通信而不需要代码更改。但是改变代码并不总是可能的，要么因为我们没有访问权限，要么因为这是不切实际的。在这种情况下，我们可以编写一个额外的层，它会对两个接口之间的通信所需的所有修改进行必要的修改。这个层被称为*适配器*。

为了进一步理解这种设计模式，让我们考虑一些现实生活中的例子。

# 现实世界示例

当你从大多数欧洲国家前往英国或美国，或者相反方向旅行时，你需要使用一个插头适配器来给你的笔记本电脑充电。另一种适配器是用于将某些设备连接到你的电脑：USB 适配器。

在软件类别中，Zope 应用程序服务器（[`www.zope.org`](http://www.zope.org)）因其**Zope 组件架构**（**ZCA**）而闻名，它有助于实现接口和适配器，这些接口被几个大型 Python 网络项目所使用。由前 Zope 开发者构建的*Pyramid*是一个 Python 网络框架，它从 Zope 中吸取了好的想法，为开发 Web 应用提供了一种更模块化的方法。Pyramid 使用适配器使得现有的对象能够符合特定的 API，而无需对其进行修改。Zope 生态系统中的另一个项目，*Plone CMS*，在底层使用适配器。

# 用例

通常，两个不兼容的接口中有一个是外来的，或者是旧的/遗留的。如果接口是外来的，这意味着我们没有访问源代码。如果是旧的，通常重构它是不可行的。

使用适配器在实现后使事物工作是一种好的方法，因为它不需要访问外部接口的源代码。如果我们必须重用一些旧代码，这通常也是一个实用的解决方案。

在此基础上，让我们开始用 Python 实现一个动手应用。

# 实现

让我们看看一个相对简单的应用来阐述适配的概念。考虑一个俱乐部的活动示例。它主要需要通过雇佣有才华的艺术家来组织表演和活动，以娱乐其客户。

在核心上，我们有一个`Club`类，它代表俱乐部，聘请的艺术家在某个晚上进行表演。`organize_performance()`方法是俱乐部可以执行的主要动作。代码如下：

```py
class Club: 
    def __init__(self, name): 
        self.name = name 

    def __str__(self): 
        return f'the club {self.name}' 

    def organize_event(self): 
        return 'hires an artist to perform for the people' 
```

大多数时候，我们的俱乐部会雇佣 DJ 进行表演，但我们的应用解决了组织由音乐家或音乐乐队、舞者、单口或单场表演等多种表演的需求。

通过我们的研究尝试重用现有代码，我们发现了一个开源贡献的库，它为我们带来了两个有趣的类：`Musician`和`Dancer`。在`Musician`类中，主要动作是通过`play()`方法执行的。在`Dancer`类中，是通过`dance()`方法执行的。

在我们的例子中，为了表明这两个类是外部的，我们将它们放在一个单独的模块中。`Musician`类的代码如下：

```py
class Musician:
 def __init__(self, name):
 self.name = name

 def __str__(self):
 return f'the musician {self.name}'

  def play(self):
 return 'plays music'
```

然后，`Dancer`类的定义如下：

```py
class Dancer:
     def __init__(self, name):
         self.name = name

     def __str__(self):
         return f'the dancer {self.name}'

     def dance(self):
         return 'does a dance performance'
```

客户端代码使用这些类，只知道如何调用`organize_performance()`方法（在`Club`类上）；它对`play()`或`dance()`（在外部库的相应类上）一无所知。

*我们如何在不改变* `Musician` *和* `Dancer` *类的情况下使代码工作？*

适配器来拯救！我们创建了一个通用的`Adapter`类，它允许我们将具有不同接口的多个对象适配到一个统一的接口。`__init__()`方法的`obj`参数是我们想要适配的对象，而`adapted_methods`是一个包含键/值对的字典，这些键/值对匹配客户端调用的方法和应该调用的方法。

`Adapter`类的代码如下：

```py
class Adapter:
     def __init__(self, obj, adapted_methods):
         self.obj = obj
         self.__dict__.update(adapted_methods)

     def __str__(self):
         return str(self.obj)
```

当处理不同类的实例时，我们有两种情况：

+   属于`Club`类的兼容对象不需要适配。我们可以将其视为原样。

+   需要先使用`Adapter`类来适配不兼容的对象。

结果是，客户端代码可以继续在所有对象上使用已知的`organize_performance()`方法，而无需意识到所使用类之间的任何接口差异。考虑以下代码：

```py
def main():
    objects = [Club('Jazz Cafe'), Musician('Roy Ayers'), \
      Dancer('Shane Sparks')]

    for obj in objects:
        if hasattr(obj, 'play') or hasattr(obj, 'dance'):
            if hasattr(obj, 'play'):
                adapted_methods = \
                  dict(organize_event=obj.play)
            elif hasattr(obj, 'dance'):            
                adapted_methods = \
                  dict(organize_event=obj.dance)

            # referencing the adapted object here
            obj = Adapter(obj, adapted_methods)

        print(f'{obj} {obj.organize_event()}') 
```

让我们回顾一下我们适配器模式实现的完整代码：

1.  我们定义了`Musician`和`Dancer`类（在`external.py`中）。

1.  然后，我们需要从外部模块（在`adapter.py`中）导入这些类：

    ```py
    from external import Musician, Dance
    ```

1.  我们随后定义`Adapter`类（在`adapter.py`中）。

1.  我们添加了`main()`函数，如前所述，以及通常的调用它的技巧（在`adapter.py`中）。

这里是执行`python adapter.py`命令时的输出，就像往常一样：

```py
the club Jazz Cafe hires an artist to perform for the 
people
the musician Roy Ayers plays music
the dancer Shane Sparks does a dance performance
```

如您所见，我们成功使`Musician`和`Dancer`类与客户端期望的接口兼容，而无需更改它们的源代码。

# 摘要

本章介绍了适配器设计模式。适配器使得在实现之后的事物能够工作。Pyramid Web 框架、Plone CMS 以及其他基于 Zope 或相关框架使用适配器模式来实现接口兼容性。在*实现*部分，我们看到了如何使用适配器模式来实现接口一致性，而无需修改不兼容模型的源代码。这是通过一个通用的`Adapter`类来完成的，它为我们做了这项工作。

总体来说，我们可以使用适配器模式使两个（或更多）不兼容的接口兼容，这在软件工程中有许多用途。

在下一章中，我们将介绍装饰器模式。
