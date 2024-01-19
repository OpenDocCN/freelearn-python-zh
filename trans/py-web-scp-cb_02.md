# 数据获取和提取

在本章中，我们将涵盖：

+   如何使用 BeautifulSoup 解析网站和导航 DOM

+   使用 Beautiful Soup 的查找方法搜索 DOM

+   使用 XPath 和 lxml 查询 DOM

+   使用 XPath 和 CSS 选择器查询数据

+   使用 Scrapy 选择器

+   以 Unicode / UTF-8 格式加载数据

# 介绍

有效抓取的关键方面是理解内容和数据如何存储在 Web 服务器上，识别要检索的数据，并理解工具如何支持此提取。在本章中，我们将讨论网站结构和 DOM，介绍使用 lxml、XPath 和 CSS 解析和查询网站的技术。我们还将看看如何处理其他语言和不同编码类型（如 Unicode）开发的网站。

最终，理解如何在 HTML 文档中查找和提取数据归结为理解 HTML 页面的结构，它在 DOM 中的表示，查询 DOM 以查找特定元素的过程，以及如何根据数据的表示方式指定要检索的元素。

# 如何使用 BeautifulSoup 解析网站和导航 DOM

当浏览器显示网页时，它会在一种称为**文档对象模型**（**DOM**）的表示中构建页面内容的模型。DOM 是页面整个内容的分层表示，以及结构信息、样式信息、脚本和其他内容的链接。

理解这种结构对于能够有效地从网页上抓取数据至关重要。我们将看一个示例网页，它的 DOM，并且检查如何使用 Beautiful Soup 导航 DOM。

# 准备就绪

我们将使用示例代码的`www`文件夹中包含的一个小型网站。要跟着做，请从`www`文件夹内启动一个 Web 服务器。可以使用 Python 3 来完成这个操作：

```py
www $ python3 -m http.server 8080
Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/) ...
```

可以通过右键单击页面并选择检查来检查 Chrome 中的网页 DOM。这将打开 Chrome 开发者工具。在浏览器中打开`http://localhost:8080/planets.html`。在 Chrome 中，您可以右键单击并选择“检查”以打开开发者工具（其他浏览器也有类似的工具）。

![](img/414227f7-dd30-4c7e-8bab-7fc02e136fcd.png)在页面上选择检查

这将打开开发者工具和检查器。DOM 可以在元素选项卡中检查。

以下显示了表中第一行的选择：

![](img/f3dd4285-7e9b-4b96-a3c5-3f31e318b983.png)检查第一行

每一行行星都在一个`<tr>`元素内。这个元素及其相邻元素有几个特征，我们将检查它们，因为它们被设计为模拟常见的网页。

首先，这个元素有三个属性：`id`，`planet`和`name`。属性在抓取中通常很重要，因为它们通常用于识别和定位嵌入在 HTML 中的数据。

其次，`<tr>`元素有子元素，在这种情况下是五个`<td>`元素。我们经常需要查看特定元素的子元素，以找到所需的实际数据。

这个元素还有一个父元素`<tbody>`。还有兄弟元素，以及一组`<tr>`子元素。从任何行星，我们可以向上到父元素并找到其他行星。正如我们将看到的，我们可以使用各种工具中的各种构造，比如 Beautiful Soup 中的**find**函数系列，以及`XPath`查询，轻松地导航这些关系。

# 如何做...

这个配方以及本章中的大多数其他配方都将以 iPython 的交互方式呈现。但是每个配方的代码都可以在脚本文件中找到。这个配方的代码在`02/01_parsing_html_wtih_bs.py`中。您可以输入以下内容，或者从脚本文件中复制粘贴。

现在让我们通过 Beautiful Soup 解析 HTML。我们首先通过以下代码将此页面加载到`BeautifulSoup`对象中，该代码创建一个 BeautifulSoup 对象，使用 requests.get 加载页面内容，并将其加载到名为 soup 的变量中。

```py
In [1]: import requests
   ...: from bs4 import BeautifulSoup
   ...: html = requests.get("http://localhost:8080/planets.html").text
   ...: soup = BeautifulSoup(html, "lxml")
   ...:
```

通过将其转换为字符串，可以检索`soup`对象中的 HTML（大多数 BeautifulSoup 对象都具有此特性）。以下显示了文档中 HTML 的前 1000 个字符：

```py
In [2]: str(soup)[:1000]
Out[2]: '<html>\n<head>\n</head>\n<body>\n<div id="planets">\n<h1>Planetary data</h1>\n<div id="content">Here are some interesting facts about the planets in our solar system</div>\n<p></p>\n<table border="1" id="planetsTable">\n<tr id="planetHeader">\n<th>\n</th>\n<th>\r\n Name\r\n </th>\n<th>\r\n Mass (10²⁴kg)\r\n </th>\n<th>\r\n Diameter (km)\r\n </th>\n<th>\r\n How it got its Name\r\n </th>\n<th>\r\n More Info\r\n </th>\n</tr>\n<tr class="planet" id="planet1" name="Mercury">\n<td>\n<img src="img/mercury-150x150.png"/>\n</td>\n<td>\r\n Mercury\r\n </td>\n<td>\r\n 0.330\r\n </td>\n<td>\r\n 4879\r\n </td>\n<td>Named Mercurius by the Romans because it appears to move so swiftly.</td>\n<td>\n<a href="https://en.wikipedia.org/wiki/Mercury_(planet)">Wikipedia</a>\n</td>\n</tr>\n<tr class="p'
```

我们可以使用`soup`的属性来导航 DOM 中的元素。`soup`代表整个文档，我们可以通过链接标签名称来深入文档。以下导航到包含数据的`<table>`：

```py
In [3]: str(soup.html.body.div.table)[:200]
Out[3]: '<table border="1" id="planetsTable">\n<tr id="planetHeader">\n<th>\n</th>\n<th>\r\n Name\r\n </th>\n<th>\r\n Mass (10²⁴kg)\r\n </th>\n<th>\r\n '
```

以下是获取表格的第一个子`<tr>`：

```py
In [6]: soup.html.body.div.table.tr
Out[6]: <tr id="planetHeader">
<th>
</th>
<th>
                    Name
                </th>
<th>
                    Mass (10²⁴kg)
                </th>
<th>
                    Diameter (km)
                </th>
<th>
                    How it got its Name
                </th>
<th>
                    More Info
                </th>
</tr>
```

请注意，此类表示法仅检索该类型的第一个子节点。要找到更多，需要迭代所有子节点，我们将在下一步中进行，或者使用查找方法（下一个示例）。

每个节点都有子节点和后代。后代是给定节点下面的所有节点（甚至比直接子节点更深层次的节点），而子节点是第一级后代。以下是获取表格的子节点，实际上是一个`list_iterator`对象：

```py
In [4]: soup.html.body.div.table.children
Out[4]: <list_iterator at 0x10eb11cc0>
```

我们可以使用`for`循环或 Python 生成器来检查迭代器中的每个子元素。以下使用生成器来获取所有子节点，并将它们的 HTML 组成的前几个字符作为列表返回：

```py
In [5]: [str(c)[:45] for c in soup.html.body.div.table.children]
Out[5]:
['\n',
 '<tr id="planetHeader">\n<th>\n</th>\n<th>\r\n ',
 '\n',
 '<tr class="planet" id="planet1" name="Mercury',
 '\n',
 '<tr class="planet" id="planet2" name="Venus">',
 '\n',
 '<tr class="planet" id="planet3" name="Earth">',
 '\n',
 '<tr class="planet" id="planet4" name="Mars">\n',
 '\n',
 '<tr class="planet" id="planet5" name="Jupiter',
 '\n',
 '<tr class="planet" id="planet6" name="Saturn"',
 '\n',
 '<tr class="planet" id="planet7" name="Uranus"',
 '\n',
 '<tr class="planet" id="planet8" name="Neptune',
 '\n',
 '<tr class="planet" id="planet9" name="Pluto">',
 '\n']
```

最后，节点的父节点可以使用`.parent`属性找到：

```py
In [7]: str(soup.html.body.div.table.tr.parent)[:200]
Out[7]: '<table border="1" id="planetsTable">\n<tr id="planetHeader">\n<th>\n</th>\n<th>\r\n Name\r\n </th>\n<th>\r\n Mass (10²⁴kg)\r\n </th>\n<th>\r\n '
```

# 它是如何工作的

Beautiful Soup 将页面的 HTML 转换为其自己的内部表示。这个模型与浏览器创建的 DOM 具有相同的表示。但是 Beautiful Soup 还提供了许多强大的功能，用于导航 DOM 中的元素，例如我们在使用标签名称作为属性时所看到的。当我们知道 HTML 中的标签名称的固定路径时，这些功能非常适合查找东西。

# 还有更多...

这种导航 DOM 的方式相对不灵活，并且高度依赖于结构。可能随着网页由其创建者更新，结构会随时间改变。页面甚至可能看起来相同，但具有完全不同的结构，从而破坏您的抓取代码。

那么我们该如何处理呢？正如我们将看到的，有几种搜索元素的方法比定义显式路径要好得多。一般来说，我们可以使用 XPath 和 Beautiful Soup 的查找方法来做到这一点。我们将在本章后面的示例中检查这两种方法。

# 使用 Beautiful Soup 的查找方法搜索 DOM

我们可以使用 Beautiful Soup 的查找方法对 DOM 进行简单搜索。这些方法为我们提供了一个更灵活和强大的构造，用于查找不依赖于这些元素的层次结构的元素。在本示例中，我们将检查这些函数的几种常见用法，以定位 DOM 中的各种元素。

# 准备工作

如果您想将以下内容剪切并粘贴到 ipython 中，您可以在`02/02_bs4_find.py`中找到示例。

# 如何做...

我们将从一个新的 iPython 会话开始，并首先加载行星页面：

```py
In [1]: import requests
 ...: from bs4 import BeautifulSoup
 ...: html = requests.get("http://localhost:8080/planets.html").text
 ...: soup = BeautifulSoup(html, "lxml")
 ...:
```

在上一个示例中，为了访问表格中的所有`<tr>`，我们使用了链式属性语法来获取表格，然后需要获取子节点并对其进行迭代。这会有一个问题，因为子节点可能是除了`<tr>`之外的其他元素。获取`<tr>`子元素的更优选方法是使用`findAll`。

让我们首先找到`<table>`：

```py
In [4]: table = soup.find("table")
   ...: str(table)[:100]
   ...:
Out[4]: '<table border="1" id="planetsTable">\n<tr id="planetHeader">\n<th>\n</th>\n<th>\r\n Nam'
```

这告诉 soup 对象在文档中查找第一个`<table>`元素。从这个元素中，我们可以使用`findAll`找到所有属于该表格的`<tr>`元素的后代：

```py
In [8]: [str(tr)[:50] for tr in table.findAll("tr")]
Out[8]:
['<tr id="planetHeader">\n<th>\n</th>\n<th>\r\n ',
 '<tr class="planet" id="planet1" name="Mercury">\n<t',
 '<tr class="planet" id="planet2" name="Venus">\n<td>',
 '<tr class="planet" id="planet3" name="Earth">\n<td>',
 '<tr class="planet" id="planet4" name="Mars">\n<td>\n',
 '<tr class="planet" id="planet5" name="Jupiter">\n<t',
 '<tr class="planet" id="planet6" name="Saturn">\n<td',
 '<tr class="planet" id="planet7" name="Uranus">\n<td',
 '<tr class="planet" id="planet8" name="Neptune">\n<t',
 '<tr class="planet" id="planet9" name="Pluto">\n<td>']
```

请注意这些是后代而不是直接的子代。将查询更改为`"td"`以查看区别。没有直接的子代是`<td>`，但每行都有多个`<td>`元素。总共会找到 54 个`<td>`元素。

如果我们只想要包含行星数据的行，这里有一个小问题。表头也被包括在内。我们可以通过利用目标行的`id`属性来解决这个问题。以下代码找到了`id`值为`"planet3"`的行。

```py
In [14]: table.find("tr", {"id": "planet3"})
    ...:
Out[14]:
<tr class="planet" id="planet3" name="Earth">
<td>
<img src="img/earth-150x150.png"/>
</td>
<td>
                    Earth
                </td>
<td>
                    5.97
                </td>
<td>
                    12756
                </td>
<td>
                    The name Earth comes from the Indo-European base 'er,'which produced the Germanic noun 'ertho,' and ultimately German 'erde,'
                    Dutch 'aarde,' Scandinavian 'jord,' and English 'earth.' Related forms include Greek 'eraze,' meaning
                    'on the ground,' and Welsh 'erw,' meaning 'a piece of land.'
                </td>
<td>
<a href="https://en.wikipedia.org/wiki/Earth">Wikipedia</a>
</td>
</tr>
```

太棒了！我们利用了这个页面使用这个属性来表示具有实际数据的表行。

现在让我们再进一步，收集每个行星的质量，并将名称和质量放入字典中：

```py
In [18]: items = dict()
    ...: planet_rows = table.findAll("tr", {"class": "planet"})
    ...: for i in planet_rows:
    ...: tds = i.findAll("td")
    ...: items[tds[1].text.strip()] = tds[2].text.strip()
    ...:

In [19]: items
Out[19]:
{'Earth': '5.97',
 'Jupiter': '1898',
 'Mars': '0.642',
 'Mercury': '0.330',
 'Neptune': '102',
 'Pluto': '0.0146',
 'Saturn': '568',
 'Uranus': '86.8',
 'Venus': '4.87'}
```

就像这样，我们已经从页面中嵌入的内容中制作了一个很好的数据结构。

# 使用 XPath 和 lxml 查询 DOM

XPath 是一种用于从 XML 文档中选择节点的查询语言，对于进行网页抓取的任何人来说，它是必须学习的查询语言。XPath 相对于其他基于模型的工具，为其用户提供了许多好处：

+   可以轻松地浏览 DOM 树

+   比 CSS 选择器和正则表达式等其他选择器更复杂和强大

+   它有一个很棒的（200+）内置函数集，并且可以通过自定义函数进行扩展

+   它得到了解析库和抓取平台的广泛支持

XPath 包含七种数据模型（我们之前已经看到了其中一些）：

+   根节点（顶级父节点）

+   元素节点（`<a>`..`</a>`）

+   属性节点（`href="example.html"`）

+   文本节点（`"this is a text"`）

+   注释节点（`<!-- a comment -->`）

+   命名空间节点

+   处理指令节点

XPath 表达式可以返回不同的数据类型：

+   字符串

+   布尔值

+   数字

+   节点集（可能是最常见的情况）

（XPath）**轴**定义了相对于当前节点的节点集。XPath 中定义了总共 13 个轴，以便轻松搜索不同的节点部分，从当前上下文节点或根节点。

**lxml**是一个 Python 包装器，位于 libxml2 XML 解析库之上，后者是用 C 编写的。C 中的实现有助于使其比 Beautiful Soup 更快，但在某些计算机上安装起来也更困难。最新的安装说明可在以下网址找到：[`lxml.de/installation.html`](http://lxml.de/installation.html)。

lxml 支持 XPath，这使得管理复杂的 XML 和 HTML 文档变得相当容易。我们将研究使用 lxml 和 XPath 一起的几种技术，以及如何使用 lxml 和 XPath 来导航 DOM 并访问数据。

# 准备工作

这些片段的代码在`02/03_lxml_and_xpath.py`中，如果你想节省一些输入。我们将首先从`lxml`中导入`html`，以及`requests`，然后加载页面。

```py
In [1]: from lxml import html
   ...: import requests
   ...: page_html = requests.get("http://localhost:8080/planets.html").text
```

到这一点，lxml 应该已经作为其他安装的依赖项安装了。如果出现错误，请使用`pip install lxml`进行安装。

# 如何做...

我们要做的第一件事是将 HTML 加载到 lxml 的“etree”中。这是 lxml 对 DOM 的表示。

```py
in [2]: tree = html.fromstring(page_html)
```

`tree`变量现在是 DOM 的 lxml 表示，它对 HTML 内容进行了建模。现在让我们来看看如何使用它和 XPath 从文档中选择各种元素。

我们的第一个 XPath 示例将是查找所有在`<table>`元素下的`<tr>`元素。

```py
In [3]: [tr for tr in tree.xpath("/html/body/div/table/tr")]
Out[3]:
[<Element tr at 0x10cfd1408>,
 <Element tr at 0x10cfd12c8>,
 <Element tr at 0x10cfd1728>,
 <Element tr at 0x10cfd16d8>,
 <Element tr at 0x10cfd1458>,
 <Element tr at 0x10cfd1868>,
 <Element tr at 0x10cfd1318>,
 <Element tr at 0x10cfd14a8>,
 <Element tr at 0x10cfd10e8>,
 <Element tr at 0x10cfd1778>,
 <Element tr at 0x10cfd1638>]
```

这个 XPath 从文档的根部通过标签名称进行导航，直到`<tr>`元素。这个例子看起来类似于 Beautiful Soup 中的属性表示法，但最终它更加具有表现力。请注意结果中的一个区别。所有的`<tr>`元素都被返回了，而不仅仅是第一个。事实上，如果每个级别的标签都有多个项目可用，那么这个路径的搜索将在所有这些`<div>`上执行。

实际结果是一个`lxml`元素对象。以下使用`etree.tostring()`获取与元素相关的 HTML（尽管它们已经应用了编码）：

```py
In [4]: from lxml import etree
   ...: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div/table/tr")]
Out[4]:
[b'<tr id="planetHeader">
\n <th>&#',
 b'<tr id="planet1" class="planet" name="Mercury">&#1',
 b'<tr id="planet2" class="planet" name="Venus">
',
 b'<tr id="planet3" class="planet" name="Earth">
',
 b'<tr id="planet4" class="planet" name="Mars">
\n',
 b'<tr id="planet5" class="planet" name="Jupiter">&#1',
 b'<tr id="planet6" class="planet" name="Saturn">&#13',
 b'<tr id="planet7" class="planet" name="Uranus">&#13',
 b'<tr id="planet8" class="planet" name="Neptune">&#1',
 b'<tr id="planet9" class="planet" name="Pluto">
',
 b'<tr id="footerRow">
\n <td>
']
```

现在让我们看看如何使用 XPath 来选择只有行星的`<tr>`元素。

```py
In [5]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div/table/tr[@class='planet']")]
Out[5]:
[b'<tr id="planet1" class="planet" name="Mercury">&#1',
 b'<tr id="planet2" class="planet" name="Venus">
',
 b'<tr id="planet3" class="planet" name="Earth">
',
 b'<tr id="planet4" class="planet" name="Mars">
\n',
 b'<tr id="planet5" class="planet" name="Jupiter">&#1',
 b'<tr id="planet6" class="planet" name="Saturn">&#13',
 b'<tr id="planet7" class="planet" name="Uranus">&#13',
 b'<tr id="planet8" class="planet" name="Neptune">&#1',
 b'<tr id="planet9" class="planet" name="Pluto">
']
```

在标签旁边使用`[]`表示我们要根据当前元素的某些条件进行选择。`@`表示我们要检查标签的属性，在这种情况下，我们要选择属性等于"planet"的标签。

还有另一个要指出的是查询中有 11 个`<tr>`行。如前所述，XPath 在每个级别上对所有找到的节点进行导航。这个文档中有两个表，都是不同`<div>`的子元素，都是`<body>`元素的子元素。具有`id="planetHeader"`的行来自我们想要的目标表，另一个具有`id="footerRow"`的行来自第二个表。

以前我们通过选择`class="row"`的`<tr>`来解决了这个问题，但还有其他值得简要提及的方法。首先，我们还可以使用`[]`来指定 XPath 的每个部分中的特定元素，就像它们是数组一样。看下面的例子：

```py
In [6]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div[1]/table/tr")]
Out[6]:
[b'<tr id="planetHeader">
\n <th>&#',
 b'<tr id="planet1" class="planet" name="Mercury">&#1',
 b'<tr id="planet2" class="planet" name="Venus">
',
 b'<tr id="planet3" class="planet" name="Earth">
',
 b'<tr id="planet4" class="planet" name="Mars">
\n',
 b'<tr id="planet5" class="planet" name="Jupiter">&#1',
 b'<tr id="planet6" class="planet" name="Saturn">&#13',
 b'<tr id="planet7" class="planet" name="Uranus">&#13',
 b'<tr id="planet8" class="planet" name="Neptune">&#1',
 b'<tr id="planet9" class="planet" name="Pluto">
']
```

XPath 中的数组从 1 开始而不是 0（一个常见的错误来源）。这选择了第一个`<div>`。更改为`[2]`选择了第二个`<div>`，因此只选择了第二个`<table>`。

```py
In [7]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div[2]/table/tr")]
Out[7]: [b'<tr id="footerRow">
\n <td>
']
```

这个文档中的第一个`<div>`也有一个 id 属性：

```py
  <div id="planets">  
```

这可以用来选择这个`<div>`：

```py
In [8]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div[@id='planets']/table/tr")]
Out[8]:
[b'<tr id="planetHeader">
\n <th>&#',
 b'<tr id="planet1" class="planet" name="Mercury">&#1',
 b'<tr id="planet2" class="planet" name="Venus">
',
 b'<tr id="planet3" class="planet" name="Earth">
',
 b'<tr id="planet4" class="planet" name="Mars">
\n',
 b'<tr id="planet5" class="planet" name="Jupiter">&#1',
 b'<tr id="planet6" class="planet" name="Saturn">&#13',
 b'<tr id="planet7" class="planet" name="Uranus">&#13',
 b'<tr id="planet8" class="planet" name="Neptune">&#1',
 b'<tr id="planet9" class="planet" name="Pluto">
']
```

之前我们根据 class 属性的值选择了行星行。我们也可以排除行：

```py
In [9]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div[@id='planets']/table/tr[@id!='planetHeader']")]
Out[9]:
[b'<tr id="planet1" class="planet" name="Mercury">&#1',
 b'<tr id="planet2" class="planet" name="Venus">
',
 b'<tr id="planet3" class="planet" name="Earth">
',
 b'<tr id="planet4" class="planet" name="Mars">
\n',
 b'<tr id="planet5" class="planet" name="Jupiter">&#1',
 b'<tr id="planet6" class="planet" name="Saturn">&#13',
 b'<tr id="planet7" class="planet" name="Uranus">&#13',
 b'<tr id="planet8" class="planet" name="Neptune">&#1',
 b'<tr id="planet9" class="planet" name="Pluto">
']
```

假设行星行没有属性（也没有标题行），那么我们可以通过位置来做到这一点，跳过第一行：

```py
In [10]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div[@id='planets']/table/tr[position() > 1]")]
Out[10]:
[b'<tr id="planet1" class="planet" name="Mercury">&#1',
 b'<tr id="planet2" class="planet" name="Venus">
',
 b'<tr id="planet3" class="planet" name="Earth">
',
 b'<tr id="planet4" class="planet" name="Mars">
\n',
 b'<tr id="planet5" class="planet" name="Jupiter">&#1',
 b'<tr id="planet6" class="planet" name="Saturn">&#13',
 b'<tr id="planet7" class="planet" name="Uranus">&#13',
 b'<tr id="planet8" class="planet" name="Neptune">&#1',
 b'<tr id="planet9" class="planet" name="Pluto">
']
```

可以使用`parent::*`来导航到节点的父级：

```py
In [11]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div/table/tr/parent::*")]
Out[11]:
[b'<table id="planetsTable" border="1">
\n ',
 b'<table id="footerTable">
\n <tr id="']
```

这返回了两个父级，因为这个 XPath 返回了两个表的行，所以找到了所有这些行的父级。`*`是一个通配符，代表任何名称的任何父级标签。在这种情况下，这两个父级都是表，但通常结果可以是任意数量的 HTML 元素类型。下面的结果相同，但如果两个父级是不同的 HTML 标签，那么它只会返回`<table>`元素。

```py
In [12]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div/table/tr/parent::table")]
Out[12]:
[b'<table id="planetsTable" border="1">
\n ',
 b'<table id="footerTable">
\n <tr id="']
```

还可以通过位置或属性指定特定的父级。以下选择具有`id="footerTable"`的父级：

```py
In [13]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div/table/tr/parent::table[@id='footerTable']")]
Out[13]: [b'<table id="footerTable">
\n <tr id="']
```

父级的快捷方式是`..`（`.`也表示当前节点）：

```py
In [14]: [etree.tostring(tr)[:50] for tr in tree.xpath("/html/body/div/table/tr/..")]
Out[14]:
[b'<table id="planetsTable" border="1">
\n ',
 b'<table id="footerTable">
\n <tr id="']
```

最后一个示例找到了地球的质量：

```py
In [15]: mass = tree.xpath("/html/body/div[1]/table/tr[@name='Earth']/td[3]/text()[1]")[0].strip()
    ...: mass
Out[15]: '5.97'
```

这个 XPath 的尾部`/td[3]/text()[1]`选择了行中的第三个`<td>`元素，然后选择了该元素的文本（这是元素中所有文本的数组），并选择了其中的第一个质量。

# 它是如何工作的

XPath 是**XSLT**（可扩展样式表语言转换）标准的一部分，提供了在 XML 文档中选择节点的能力。HTML 是 XML 的一种变体，因此 XPath 可以在 HTML 文档上工作（尽管 HTML 可能格式不正确，在这种情况下会破坏 XPath 解析）。

XPath 本身旨在模拟 XML 节点、属性和属性的结构。该语法提供了查找与表达式匹配的 XML 中的项目的方法。这可以包括匹配或逻辑比较 XML 文档中任何节点、属性、值或文本的任何部分。

XPath 表达式可以组合成非常复杂的路径在文档中。还可以根据相对位置导航文档，这在根据相对位置而不是 DOM 中的绝对位置找到数据时非常有帮助。

理解 XPath 对于知道如何解析 HTML 和执行网页抓取是至关重要的。正如我们将看到的，它是许多高级库的基础，并为其提供了实现，比如 lxml。

# 还有更多...

XPath 实际上是处理 XML 和 HTML 文档的一个了不起的工具。它在功能上非常丰富，我们仅仅触及了它在演示 HTML 文档中常见的一些示例的表面。

要了解更多，请访问以下链接：

+   [`www.w3schools.com/xml/xml_xpath.asp`](https://www.w3schools.com/xml/xml_xpath.asp)

+   [`www.w3.org/TR/xpath/`](https://www.w3.org/TR/xpath/)

# 使用 XPath 和 CSS 选择器查询数据

CSS 选择器是用于选择元素的模式，通常用于定义应该应用样式的元素。它们也可以与 lxml 一起用于选择 DOM 中的节点。CSS 选择器通常被广泛使用，因为它们比 XPath 更紧凑，并且通常在代码中更可重用。以下是可能使用的常见选择器的示例：

| **您要寻找的内容** | **示例** |
| --- | --- |
| 所有标签 | `*` |
| 特定标签（即`tr`） | `.planet` |
| 类名（即`"planet"`） | `tr.planet` |
| 具有`ID "planet3"`的标签 | `tr#planet3` |
| 表的子`tr` | `table tr` |
| 表的后代`tr` | `table tr` |
| 带有属性的标签（即带有`id="planet4"`的`tr`） | `a[id=Mars]` |

# 准备工作

让我们开始使用与上一个示例中使用的相同的启动代码来检查 CSS 选择器。这些代码片段也在`02/04_css_selectors.py`中。

```py
In [1]: from lxml import html
   ...: import requests
   ...: page_html = requests.get("http://localhost:8080/planets.html").text
   ...: tree = html.fromstring(page_html)
   ...:
```

# 如何做...

现在让我们开始使用 XPath 和 CSS 选择器。以下选择所有具有等于`"planet"`的类的`<tr>`元素：

```py
In [2]: [(v, v.xpath("@name")) for v in tree.cssselect('tr.planet')]
Out[2]:
[(<Element tr at 0x10d3a2278>, ['Mercury']),
 (<Element tr at 0x10c16ed18>, ['Venus']),
 (<Element tr at 0x10e445688>, ['Earth']),
 (<Element tr at 0x10e477228>, ['Mars']),
 (<Element tr at 0x10e477408>, ['Jupiter']),
 (<Element tr at 0x10e477458>, ['Saturn']),
 (<Element tr at 0x10e4774a8>, ['Uranus']),
 (<Element tr at 0x10e4774f8>, ['Neptune']),
 (<Element tr at 0x10e477548>, ['Pluto'])]
```

可以通过多种方式找到地球的数据。以下是基于`id`获取行的方法：

```py
In [3]: tr = tree.cssselect("tr#planet3")
   ...: tr[0], tr[0].xpath("./td[2]/text()")[0].strip()
   ...:
Out[3]: (<Element tr at 0x10e445688>, 'Earth')
```

以下示例使用具有特定值的属性：

```py
In [4]: tr = tree.cssselect("tr[name='Pluto']")
   ...: tr[0], tr[0].xpath("td[2]/text()")[0].strip()
   ...:
Out[5]: (<Element tr at 0x10e477548>, 'Pluto')
```

请注意，与 XPath 不同，不需要使用`@`符号来指定属性。

# 工作原理

lxml 将您提供的 CSS 选择器转换为 XPath，然后针对底层文档执行该 XPath 表达式。实质上，lxml 中的 CSS 选择器提供了一种简写 XPath 的方法，使得查找符合某些模式的节点比使用 XPath 更简单。

# 还有更多...

由于 CSS 选择器在底层使用 XPath，因此与直接使用 XPath 相比，使用它会增加一些开销。然而，这种差异几乎不成问题，因此在某些情况下，更容易只使用 cssselect。

可以在以下位置找到 CSS 选择器的完整描述：[`www.w3.org/TR/2011/REC-css3-selectors-20110929/`](https://www.w3.org/TR/2011/REC-css3-selectors-20110929/)

# 使用 Scrapy 选择器

Scrapy 是一个用于从网站提取数据的 Python 网络爬虫框架。它提供了许多强大的功能，用于浏览整个网站，例如跟踪链接的能力。它提供的一个功能是使用 DOM 在文档中查找数据，并且现在，相当熟悉的 XPath。

在这个示例中，我们将加载 StackOverflow 上当前问题的列表，然后使用 scrapy 选择器解析它。使用该选择器，我们将提取每个问题的文本。

# 准备工作

此示例的代码位于`02/05_scrapy_selectors.py`中。

# 如何做...

我们首先从`scrapy`中导入`Selector`，还有`requests`，以便我们可以检索页面：

```py
In [1]: from scrapy.selector import Selector
   ...: import requests
   ...:
```

接下来加载页面。在此示例中，我们将检索 StackOverflow 上最近的问题并提取它们的标题。我们可以使用以下查询来实现：

```py
In [2]: response = requests.get("http://stackoverflow.com/questions")
```

现在创建一个`Selector`并将其传递给响应对象：

```py
In [3]: selector = Selector(response)
   ...: selector
   ...:
Out[3]: <Selector xpath=None data='<html>\r\n\r\n <head>\r\n\r\n <title>N'>
```

检查此页面的内容，我们可以看到问题的 HTML 具有以下结构：

![](img/d72e8df6-61f1-4395-a003-009279e30ddb.png)StackOverflow 问题的 HTML

使用选择器，我们可以使用 XPath 找到这些：

```py
In [4]: summaries = selector.xpath('//div[@class="summary"]/h3')
   ...: summaries[0:5]
   ...:
Out[4]:
[<Selector xpath='//div[@class="summary"]/h3' data='<h3><a href="/questions/48353091/how-to-'>,
 <Selector xpath='//div[@class="summary"]/h3' data='<h3><a href="/questions/48353090/move-fi'>,
 <Selector xpath='//div[@class="summary"]/h3' data='<h3><a href="/questions/48353089/java-la'>,
 <Selector xpath='//div[@class="summary"]/h3' data='<h3><a href="/questions/48353086/how-do-'>,
 <Selector xpath='//div[@class="summary"]/h3' data='<h3><a href="/questions/48353085/running'>]
```

现在我们进一步深入每个问题的标题。

```py
In [5]: [x.extract() for x in summaries.xpath('a[@class="question-hyperlink"]/text()')][:10]
Out[5]:
['How to convert stdout binary file to a data URL?',
 'Move first letter from sentence to the end',
 'Java launch program and interact with it programmatically',
 'How do I build vala from scratch',
 'Running Sql Script',
 'Mysql - Auto create, update, delete table 2 from table 1',
 'how to map meeting data corresponding calendar time in java',
 'Range of L*a* b* in Matlab',
 'set maximum and minimum number input box in js,html',
 'I created generic array and tried to store the value but it is showing ArrayStoreException']
```

# 工作原理

在底层，Scrapy 构建其选择器基于 lxml。它提供了一个较小且略微简单的 API，性能与 lxml 相似。

# 还有更多...

要了解有关 Scrapy 选择器的更多信息，请参见：[`doc.scrapy.org/en/latest/topics/selectors.html`](https://doc.scrapy.org/en/latest/topics/selectors.html)。

# 以 unicode / UTF-8 加载数据

文档的编码告诉应用程序如何将文档中的字符表示为文件中的字节。基本上，编码指定每个字符有多少位。在标准 ASCII 文档中，所有字符都是 8 位。HTML 文件通常以每个字符 8 位编码，但随着互联网的全球化，情况并非总是如此。许多 HTML 文档以 16 位字符编码，或者使用 8 位和 16 位字符的组合。

一种特别常见的 HTML 文档编码形式被称为 UTF-8。这是我们将要研究的编码形式。

# 准备工作

我们将从位于`http://localhost:8080/unicode.html`的本地 Web 服务器中读取名为`unicode.html`的文件。该文件采用 UTF-8 编码，并包含编码空间不同部分的几组字符。例如，页面在浏览器中如下所示：

浏览器中的页面

使用支持 UTF-8 的编辑器，我们可以看到西里尔字母在编辑器中是如何呈现的：

编辑器中的 HTML

示例的代码位于`02/06_unicode.py`中。

# 如何做...

我们将研究如何使用`urlopen`和`requests`来处理 UTF-8 中的 HTML。这两个库处理方式不同，让我们来看看。让我们开始导入`urllib`，加载页面并检查一些内容。

```py
In [8]: from urllib.request import urlopen
   ...: page = urlopen("http://localhost:8080/unicode.html")
   ...: content = page.read()
   ...: content[840:1280]
   ...:
Out[8]: b'><strong>Cyrillic</strong> &nbsp; U+0400 \xe2\x80\x93 U+04FF &nbsp; (1024\xe2\x80\x931279)</p>\n <table class="unicode">\n <tbody>\n <tr valign="top">\n <td width="50">&nbsp;</td>\n <td class="b" width="50">\xd0\x89</td>\n <td class="b" width="50">\xd0\xa9</td>\n <td class="b" width="50">\xd1\x89</td>\n <td class="b" width="50">\xd3\x83</td>\n </tr>\n </tbody>\n </table>\n\n '
```

请注意，西里尔字母是以多字节代码的形式读入的，使用\符号，例如`\xd0\x89`。

为了纠正这一点，我们可以使用 Python 的`str`语句将内容转换为 UTF-8 格式：

```py
In [9]: str(content, "utf-8")[837:1270]
Out[9]: '<strong>Cyrillic</strong> &nbsp; U+0400 – U+04FF &nbsp; (1024–1279)</p>\n <table class="unicode">\n <tbody>\n <tr valign="top">\n <td width="50">&nbsp;</td>\n <td class="b" width="50">Љ</td>\n <td class="b" width="50">Щ</td>\n <td class="b" width="50">щ</td>\n <td class="b" width="50">Ӄ</td>\n </tr>\n </tbody>\n </table>\n\n '
```

请注意，输出现在已经正确编码了字符。

我们可以通过使用`requests`来排除这一额外步骤。

```py
In [9]: import requests
   ...: response = requests.get("http://localhost:8080/unicode.html").text
   ...: response.text[837:1270]
   ...:
'<strong>Cyrillic</strong> &nbsp; U+0400 – U+04FF &nbsp; (1024–1279)</p>\n <table class="unicode">\n <tbody>\n <tr valign="top">\n <td width="50">&nbsp;</td>\n <td class="b" width="50">Љ</td>\n <td class="b" width="50">Щ</td>\n <td class="b" width="50">щ</td>\n <td class="b" width="50">Ӄ</td>\n </tr>\n </tbody>\n </table>\n\n '
```

# 它是如何工作的

在使用`urlopen`时，通过使用 str 语句并指定应将内容转换为 UTF-8 来明确执行了转换。对于`requests`，该库能够通过在文档中看到以下标记来确定 HTML 中的内容是以 UTF-8 格式编码的：

```py
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
```

# 还有更多...

互联网上有许多关于 Unicode 和 UTF-8 编码技术的资源。也许最好的是以下维基百科文章，其中有一个很好的摘要和描述编码技术的表格：[`en.wikipedia.org/wiki/UTF-8`](https://en.wikipedia.org/wiki/UTF-8)
