# 第十八章：*第十八章*：其他创建型模式

在上一章中，我们介绍了一种第三种创建型模式，即建造者模式，它提供了一种创建复杂对象各个部分的好方法。除了工厂方法、抽象工厂和之前提到的建造者模式之外，还有其他创建型模式值得讨论，例如**原型**模式和**单例**模式。

在本章中，我们将讨论以下主题：

+   实现原型模式

+   实现单例模式

这些主题将完成我们对创建型模式的讨论，并帮助覆盖那些我们之前看到的设计模式不适用的情况。到本章结束时，我们将对创建型模式及其每个模式的使用案例有一个整体的理解。

# 技术要求

本章的代码文件可以通过以下链接访问：[`github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter18`](https://github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter18)。

# 实现原型模式

当你需要使用**克隆**技术根据现有对象创建对象时，原型模式非常有用。正如你可能猜到的，这个想法是使用该对象完整结构的副本来生成新对象。我们将看到，这在 Python 中几乎是自然而然的，因为我们有一个**复制**功能，这极大地帮助了使用这种技术。在创建对象副本的一般情况下，发生的情况是你创建了一个指向同一对象的新的引用，这被称为**浅复制**。但是，如果你需要复制对象，即原型模式的情况，你需要进行**深复制**。

有时候，我们需要创建一个对象的精确副本。例如，假设你想为销售团队推广的产品创建一个存储、分享和编辑演示和营销内容的软件应用。想想流行的直销或网络营销模式，这是一种在家中进行的活动，个人与公司合作，在其社交网络中使用促销工具（如宣传册、PowerPoint 演示文稿、视频等）来分销产品。

假设有一个用户，*鲍勃*，在一个网络营销组织中领导着一支分销团队。他们每天使用一个演示视频向潜在客户介绍产品。在某个时候，鲍勃让他的朋友，爱丽丝，加入他，她也使用了同一个视频（一个管理原则是遵循系统，或者说，*复制已经有效的方法*）。但爱丽丝很快发现了一些可以加入她团队并帮助她业务增长的人选，如果视频是法语，或者至少有字幕的话。*他们该怎么办？*原始的演示视频不能用于可能出现的不同定制需求。

为了帮助每个人，系统可以允许具有特定等级或信任水平的分销商，例如 *Bob*，在原始演示视频被支持公司的合规团队在公开使用前验证后创建独立副本。每个副本都称为 **克隆**；它是在特定时间点的原始对象的精确副本。

因此，Bob 在合规团队的验证下，复制了演示视频以应对新的需求，并将其交给 Alice。她可以在此基础上添加法语字幕进行修改。

通过克隆，Bob 和 Alice 可以拥有自己的视频副本，因此他们各自的更改不会影响对方的材料版本。在另一种情况下，即默认情况下实际发生的情况，每个人都会持有对同一个（引用）对象的引用；Bob 的更改会影响 Alice，反之亦然。

原型设计模式帮助我们创建对象克隆。在其最简单的版本中，这个模式只是一个接受一个对象作为输入参数并返回其克隆的 `clone()` 函数。在 Python 中，这可以通过 `copy` 模块中的 `deepcopy()` 函数来实现。

我们的讨论结构将与前几章相同。首先，我们将简要讨论现实生活中的应用和用例，然后在一个 Python 示例中实现动手操作。

## 现实世界例子

一个著名的非技术例子是苏格兰研究人员克隆乳腺细胞创造出的绵羊多莉。

许多 Python 应用程序都使用了原型模式 (`j.mp/pythonprot`)，但它很少被称为 *原型*，因为克隆对象是语言的一个内置功能。

## 用例

当我们有一个需要保持不变且需要创建其精确副本的现有对象时，原型模式非常有用，允许对副本的某些部分进行更改。

还有一个经常需要复制从数据库中填充并引用其他基于数据库的对象的对象的需求。克隆这样一个复杂对象成本很高（对数据库进行多次查询），因此原型是一个解决这个问题的便捷方式。

## 实现

现在，一些组织，甚至规模较小的组织，通过其基础设施/DevOps 团队、托管提供商或云服务提供商处理许多网站和应用。

当你必须管理多个网站时，有一个点会变得难以管理一切。你需要快速访问信息，例如涉及的 IP 地址、域名及其到期日期，以及 DNS 参数的详细信息。因此，你需要一种库存工具。

让我们想象一下这些团队如何处理这种类型的数据以进行日常活动，并简要讨论一个帮助整合和维护数据的软件实现（除了 Excel 电子表格之外）：

1.  首先，我们需要导入 Python 的标准`copy`模块，如下所示：

    ```py
    import copy
    ```

1.  在这个系统的核心，我们将有一个`Website`类来存储所有有用的信息，例如我们管理的网站的名称、域名、描述和作者。

在类的`__init__()`方法中，只有一些参数是固定的，`name`、`domain`、`description`和`author`，它们对应于我们之前列出的信息。但我们还希望有灵活性，客户端代码可以使用`kwargs`变量长度集合（一个 Python 字典）以`name=value`的形式传递更多参数。

注意，有一个 Python 惯用法，可以使用内置的`setattr()`函数在对象`obj`上设置任意属性名`attr`和值`val`：`setattr(obj, attr, val)`。

因此，我们在这个类的可选属性中使用这种技术，在初始化方法的末尾，如下所示：

```py
for key in kwargs:
    setattr(self, key, kwargs[key])
```

因此，我们的`Website`类定义如下：

```py
class Website: 
    def __init__(self, name, domain, description, \
      author, **kwargs): 
        '''Examples of optional attributes (kwargs): 
           category, creation_date, technologies, \
             keywords.
        ''' 
        self.name = name 
        self.domain = domain 
        self.description = description
        self.author = author

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __str__(self): 
        summary = [f'Website "{self.name}"\n',] 

        infos = vars(self).items()
        ordered_infos = sorted(infos)
        for attr, val in ordered_infos:
            if attr == 'name':
                continue
            summary.append(f'{attr}: {val}\n')

        return ''.join(summary)
```

1.  接下来，`Prototype`类实现了原型设计模式。

`Prototype`类的核心是`clone()`方法，它负责使用`copy.deepcopy()`函数克隆对象。由于克隆意味着我们允许为可选属性设置值，请注意我们在这里如何使用`setattr()`技术以及`attrs`字典。

此外，为了方便起见，`Prototype`类包含`register()`和`unregister()`方法，这些方法可以用来在字典中跟踪克隆的对象：

```py
class Prototype: 
    def __init__(self): 
        self.objects = dict() 

    def register(self, identifier, obj): 
        self.objects[identifier] = obj 

    def unregister(self, identifier): 
        del self.objects[identifier] 

    def clone(self, identifier, **attrs): 
        found = self.objects.get(identifier) 
        if not found: 
            raise ValueError(f'Incorrect object \
              identifier:{identifier}') 
        obj = copy.deepcopy(found) 
        for key in attrs:
            setattr(obj, key, attrs[key])
        return obj
```

1.  在`main()`函数中，如下面的代码所示，我们可以克隆一个`Website`实例，`site1`，以获取第二个对象，`site2`。基本上，我们实例化`Prototype`类，并使用其`.clone()`方法。这正是下面代码所展示的：

    ```py
    def main(): 
        keywords = ('python', 'data', 'apis', \
          'automation')
        site1 = Website('ContentGardening', 
                domain='contentgardening.com', 
                description='Automation and data-driven \
                  apps', 
                author='Kamon Ayeva',
                category='Blog',
                keywords=keywords)

        prototype = Prototype() 
        identifier = 'ka-cg-1' 
        prototype.register(identifier, site1)

        site2 = prototype.clone(identifier, 
                name='ContentGardeningPlayground',
                domain='play.contentgardening.com', 
                description='Experimentation for \
                  techniques featured on the blog', 
                category='Membership site',
                creation_date='2018-08-01')    
    ```

1.  为了结束该函数，我们可以使用`id()`函数，该函数返回对象的内存地址，以比较两个对象的地址，如下所示。当我们使用深拷贝克隆对象时，克隆的内存地址必须与原始对象的内存地址不同：

    ```py
        for site in (site1, site2): 
            print(site)
        print(f'ID site1 : {id(site1)} != ID site2 : \
          {id(site2)}')
    ```

你可以在`prototype.py`文件中找到程序的完整代码。以下是我们在代码中执行的操作的摘要：

1.  我们首先导入`copy`模块。

1.  我们定义`Website`类，包括其初始化方法`(__init__())`和字符串表示方法`(__str__()`），如前面所示。

1.  我们定义我们的`Prototype`类，如前面所示。

1.  然后，我们有`main()`函数，其中我们执行以下操作：

    +   我们定义所需的`keywords`列表。

    +   我们创建`Website`类的实例，称为`site1`（我们在这里使用`keywords`列表）。

    +   我们创建`Prototype`对象，并使用其`register()`方法将`site1`及其标识符注册（这有助于我们在字典中跟踪克隆的对象）。

    +   我们克隆`site1`对象以获取`site2`。

    +   我们显示结果（两个`Website`对象）

以下是在我的机器上执行`python prototype.py`命令时的示例输出：

```py
Website "ContentGardening"
author: Kamon Ayeva
category: Blog
description: Automation and data-driven apps
domain: contentgardening.com
keywords: ('python', 'data', 'apis', 'automation')
Website "ContentGardeningPlayground"
author: Kamon Ayeva
category: Membership site
creation_date: 2018-08-01
description: Experimentation for techniques featured on the 
blog
domain: play.contentgardening.com
keywords: ('python', 'data', 'apis', 'automation')
ID site1 : 140263689073376 != ID site2 : 140263689058816
```

的确，`Prototype`按预期工作。我们可以看到原始`Website`对象及其克隆的信息。查看`id()`函数的输出，我们可以看到两个地址是不同的。

使用这个程序，我们结束了关于原型模式的讨论。在下一节中，我们将介绍单例模式。

# 实现单例模式

单例模式提供了一种实现类的方法，只能创建一个对象，因此得名单例。正如您将在我们对这个模式的探索中理解的那样，或者在进行您自己的研究时，关于这个模式的讨论始终存在，有些人甚至认为它是一个**反模式**。

此外，有趣的是，当我们需要创建一个且仅创建一个对象时，它是有用的，例如，为了存储和维护程序的全局状态。在 Python 中，可以使用一些特殊的内置功能来实现这个模式。

单例模式将类的实例化限制为*一个*对象，这在需要单个对象协调系统动作时非常有用。

基本思想是为程序的需求创建特定类的一个实例。为了确保这一点，我们需要防止类被实例化多次的机制，同时也防止克隆。

首先，让我们讨论一些单例模式在现实生活中的例子。

## 现实世界例子

我们可以将一艘船或一艘船的船长视为单例模式的一个现实生活例子。在船上，他是负责人。他负责重要的决策，由于这个责任，许多请求都指向他。

在软件中，Plone CMS 在其核心中实现了一个单例。实际上，在 Plone 站点的根目录下有几个单例对象可用，称为`singleton`类，您不能在站点上下文中创建该`singleton`类的另一个实例。

## 用例

如前所述，单例模式的一个用例是创建一个维护程序全局状态的单一对象。其他可能的用例如下：

+   控制对共享资源的并发访问；例如，管理数据库连接的对象类

+   一种在意义上跨越整个应用程序或由不同用户访问的服务或资源，并完成其工作；例如，日志系统或实用程序的核心类

## 实现

让我们实现一个程序，从网页中获取内容，灵感来源于 Michael Ford 的教程([`docs.python.org/3/howto/urllib2.html`](https://docs.python.org/3/howto/urllib2.html))。我们只取了简单部分，因为重点是说明我们的模式，而不是构建一个特殊的网络爬虫工具。

我们将使用`urllib`模块通过 URL 连接到网页；程序的核心将是`URLFetcher`类，它通过`fetch()`方法负责执行工作。

我们希望能够跟踪被跟踪的网页列表，因此使用了单例模式。我们需要一个单独的对象来维护这个全局状态：

1.  首先，我们的原始版本，受教程启发但进行了修改，以帮助我们跟踪已获取的 URL 列表，如下所示：

    ```py
    import urllib.parse
    import urllib.request
    class URLFetcher:
        def __init__(self):
            self.urls = []

        def fetch(self, url):
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.code == 200:
                    the_page = response.read()
                    print(the_page)

                    urls = self.urls
                    urls.append(url)
                    self.urls = urls
    ```

1.  作为练习，添加一个通常的`if __name__ == '__main__'`块，其中包含几行代码来调用`URLFetcher`实例的`.fetch()`方法。

但是，我们的类实现了单例模式吗？这里有一个线索。要创建一个单例，我们需要确保只能创建一个实例。因此，为了确定我们的类是否实现了单例模式，我们可以使用一个技巧，即使用`is`运算符比较两个实例。

你可能已经猜到了第二个练习。将以下代码放入你的`if __name__ == '__main__'`块中，而不是你之前所用的代码：

```py
f1 = URLFetcher()
f2 = URLFetcher()
print(f1 is f2)
```

作为替代，使用这个简洁但仍然优雅的形式：

```py
print(URLFetcher() is URLFetcher())
```

通过这个更改，当执行程序时，你应该得到`False`作为输出。

1.  好吧！这意味着第一次尝试还没有给我们一个单例。记住，我们想要管理一个全局状态，使用一个，并且只有一个类的实例来运行程序。当前版本的类还没有实现单例。

在检查了网络上的文献和论坛后，你会发现有几种技术，每种技术都有其优缺点，其中一些可能已经过时。

由于现在许多人使用 Python 3，我们将选择推荐的**元类**技术。我们首先为单例实现一个元类。这个类如下实现了单例模式：

```py
class SingletonType(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super \
              (SingletonType,cls).__call__(*args, \
                 **kwargs)
        return cls._instances[cls]
```

1.  现在，我们将重写我们的`URLFetcher`类以使用该元类。我们还添加了一个`dump_url_registry()`方法，这对于获取当前跟踪的 URL 列表非常有用：

    ```py
    class URLFetcher(metaclass=SingletonType):
        def fetch(self, url):
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.code == 200:
                    the_page = response.read()
                    print(the_page)

                    urls = self.urls
                    urls.append(url)
                    self.urls = urls
        def dump_url_registry(self):
            return ', '.join(self.urls)
    if __name__ == '__main__':
        print(URLFetcher() is URLFetcher())
    ```

这次，通过执行程序，你得到`True`。

1.  让我们完成程序，使用一个`main()`函数来完成我们想要的功能，如下所示调用：

    ```py
    def main():
        MY_URLS = ['http://google.com', 
                   'http://python.org',
                   'https://www.python.org/error',
                   ]
        print(URLFetcher() is URLFetcher())
        fetcher = URLFetcher()
        for url in MY_URLS:
            try:
                fetcher.fetch(url)
            except Exception as e:
                print(e)

        print('-------')
        done_urls = fetcher.dump_url_registry()
        print(f'Done URLs: {done_urls}')
    ```

你将在`singleton.py`文件中找到程序的完整代码。以下是我们所做的工作的摘要：

1.  我们从所需的模块导入开始（`urllib.parse`和`urllib.request`）。

1.  如前所述，我们定义了`SingletonType`类，它有一个特殊的`__call__()`方法。

1.  如前所述，我们定义了`URLFetcher`类，该类实现了网页的获取器，通过`urls`属性进行初始化。如前所述，我们添加了它的`fetch()`和`dump_url_registry()`方法。

1.  然后，我们添加我们的`main()`函数。

1.  最后，我们添加 Python 中用于调用`main`函数的传统代码片段。

执行`python singleton.py`命令时的输出如下：

```py
[output truncated]
</script>\n    <script>window.jQuery || 
document.write(\'<script src="/static/js/libs/jquery-
1.8.2.min.js"><\\/script>\')</script>\n    <script 
src="//ajax.googleapis.com/ajax/libs/jqueryui/1.12.1/jquery
-ui.min.js"></script>\n    <script>window.jQuery || 
document.write(\'<script src="/static/js/libs/jquery-ui-
1.12.1.min.js"><\\/script>\')</script>\n\n    <script 
src="img/masonry.pkgd.min.js"></script>\n    
<script src="/static/js/libs/html-
includes.js"></script>\n\n    <script 
type="text/javascript" src="/static/js/main-
min.dd72c1659644.js" charset="utf-8"></script>\n    \n\n    
<!--[if lte IE 7]>\n    <script type="text/javascript" 
src="img/IE8-min.8af6e26c7a3b.js" 
charset="utf-8"></script>\n    \n    \n    <![endif]--
>\n\n    
<!--[if lte IE 8]>\n    <script type="text/javascript" 
src="/static/js/plugins/getComputedStyle-
min.d41d8cd98f00.js" charset="utf-8"></script>\n    \n    
\n    <![endif]-->\n\n    \n\n    \n    
\n\n</body>\n</html>\n'
HTTP Error 404: Not Found
-------
Done URLs: http://google.com, http://python.org
```

我们可以看到我们得到了预期的结果：程序能够连接到的页面内容和操作成功的 URL 列表。

我们看到 URL `https://www.python.org/error` 并不在`fetcher.dump_url_registry()`返回的列表中；确实，这是一个错误的 URL，对`urllib`的请求得到了`404`响应代码。

注意

前一个 URL 的链接不应该工作；这正是重点所在。

# 摘要

在本章中，我们看到了如何使用两种其他创建型设计模式：原型和单例。

原型用于创建对象的精确副本。正如我们在讨论的实现示例中所见，在 Python 中使用原型是自然的，并且基于内置功能，因此这不是什么需要提及的事情。单例模式可以通过使`singleton`类使用元类来实现，其类型已经预先定义了该元类。按照要求，元类的`__call__()`方法包含确保只能创建该类的一个实例的代码。

总体而言，这两种设计模式帮助我们实现了其他创建型模式不支持的使用案例；实际上，我们已经扩展了我们的设计模式工具箱，以覆盖更多的使用案例。

下一章将介绍适配器模式，这是一种结构型设计模式，可以用来使两个不兼容的软件接口兼容。

# 问题

1.  使用原型模式的高层次好处是什么？

1.  原型模式在数据库管理特定案例中有什么用途？

1.  使用单例模式的高层次好处是什么？

1.  在并发的情况下，应该在什么情况下使用单例模式？

# 进一步阅读

《*设计模式*》由 Gamma Enrich，Helm Richard，Johnson Ralph 和 Vlissides John 所著，可在[`www.amazon.com/Design-Patterns-Object-Oriented-Addison-Wesley-Professional-ebook/dp/B000SEIBB8`](https://www.amazon.com/Design-Patterns-Object-Oriented-Addison-Wesley-Professional-ebook/dp/B000SEIBB8)找到。
