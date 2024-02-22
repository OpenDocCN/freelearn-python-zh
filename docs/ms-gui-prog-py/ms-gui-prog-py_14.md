# 第十一章：使用 QTextDocument 创建富文本

无论是在文字处理器中起草商业备忘录、写博客文章还是生成报告，世界上大部分的计算都涉及文档的创建。这些应用程序大多需要能够生成不仅仅是普通的字母数字字符串，还需要生成富文本。富文本（与纯文本相对）意味着包括字体、颜色、列表、表格和图像等样式和格式特性的文本。

在本章中，我们将学习 PyQt 如何允许我们通过以下主题处理富文本：

+   使用标记创建富文本

+   使用`QTextDocument`操纵富文本

+   打印富文本

# 技术要求

对于本章，您将需要自第一章以来一直在使用的基本 Python 和 Qt 设置。您可能希望参考可以在[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter11`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter11)找到的示例代码。

查看以下视频以查看代码的实际效果：[`bit.ly/2M5P4Cq`](http://bit.ly/2M5P4Cq)

# 使用标记创建富文本

每个支持富文本的应用程序都必须有一些格式来表示内存中的文本，并在将其保存到文件时。有些格式使用自定义二进制代码，例如旧版本 Microsoft Word 使用的`.doc`和`.rtf`文件。在其他情况下，使用纯文本**标记语言**。在标记语言中，称为**标签**的特殊字符串指示富文本特性的放置。Qt 采用标记方法，并使用**超文本标记语言**（**HTML**）第 4 版的子集表示富文本。

Qt 中的富文本标记由`QTextDocument`对象呈现，因此它只能用于使用`QTextDocument`存储其内容的小部件。这包括`QLabel`、`QTextEdit`和`QTextBrowser`小部件。在本节中，我们将创建一个演示脚本，以探索这种标记语言的语法和功能。

鉴于 Web 开发的普及和普遍性，您可能已经对 HTML 有所了解；如果您不了解，下一节将作为一个快速介绍。

# HTML 基础

HTML 文档由文本内容和标签组成，以指示非纯文本特性。标签只是用尖括号括起来的单词，如下所示：

```py
<sometag>This is some content</sometag>
```

注意前面示例中的`</sometag>`代码。这被称为**闭合标签**，它与开放标签类似，但标签名称前面有一个斜杠（`/`）。通常只有用于包围（或有能力包围）文本内容的标签才使用闭合标签。

考虑以下示例：

```py
Text can be <b>bold<b> <br>
Text can be <em>emphasized</em> <br>
Text can be <u>underlined</u> <hr>
```

`b`、`em`和`u`标签需要闭合标签，因为它们包围内容的一部分并指示外观的变化。`br`和`hr`标签（*换行*和*水平线*，分别）只是指示包含在文档中的非文本项，因此它们没有闭合标签。

如果您想看看这些示例中的任何一个是什么样子，您可以将它们复制到一个文本文件中，然后在您的 Web 浏览器中打开它们。还可以查看示例代码中的`html_examples.html`文件。

有时，通过嵌套标签创建复杂结构，例如以下列表：

```py
<ol>
  <li> Item one</li>
  <li> Item two</li>
  <li> Item three</li>
</ol>
```

在这里，`ol`标签开始一个有序列表（使用顺序数字或字母的列表，而不是项目符号字符）。列表中的每个项目由`li`（列表项）标签表示。请注意，当嵌套标签使用闭合标签时，标签必须按正确顺序关闭，如下所示：

```py
<b><i>This is right</i></b>
<b><i>This is wrong!</b></i>
```

前面的错误示例不起作用，因为内部标签（`<i>`）在外部标签（`<b>`）之后关闭。

HTML 标签可以有属性，这些属性是用于配置标签的键值对，如下例所示：

```py
<img src="my_image.png" width="100px" height="20px">
```

前面的标签是一个用于显示图像的`img`（图像）标签。 其属性是`src`（指示图像文件路径），`width`（指示显示图像的宽度）和`height`（指示显示的高度）。

HTML 属性是以空格分隔的，所以不要在它们之间放逗号。 值可以用单引号或双引号引用，或者如果它们不包含空格或其他令人困惑的字符（例如闭合尖括号）则不引用； 但通常最好用双引号引用它们。 在 Qt HTML 中，大小通常以`px`（像素）或`％`（百分比）指定，尽管在现代 Web HTML 中，通常使用其他单位。

# 样式表语法

现代 HTML 使用**层叠样式表**（**CSS**）进行样式设置。 在第六章中，*为 Qt 应用程序设置样式*，我们讨论了 QSS 时学习了 CSS。 回顾一下，CSS 允许您对标签的外观进行声明，如下所示：

```py
b {
    color: red;
    font-size: 16pt;
}
```

前面的 CSS 指令将使粗体标签内的所有内容（在`<b>`和`</b>`之间）以红色 16 点字体显示。

某些标签也可以有修饰符，例如：

```py
a:hovered {
   color: green;
   font-size: 16pt;
}
```

前面的 CSS 适用于`<a>`（锚点）标签内容，但仅当鼠标指针悬停在锚点上时。 这样的修饰符也称为**伪类**。

# 语义标签与装饰标签

一些 HTML 标签描述了内容应该如何显示。 我们称这些为**装饰**标签。 例如，`<i>`标签表示文本应以斜体字打印。 但请注意，斜体字在现代印刷中有许多用途-强调一个词，表示已出版作品的标题，或表示短语来自外语。 为了区分这些用途，HTML 还有*语义*标签。 例如，`<em>`表示强调，并且在大多数情况下会导致斜体文本。 但与`<i>`标签不同，它还指示文本应该以何种方式斜体。 HTML 的旧版本通常侧重于装饰标签，而较新版本则越来越注重语义标签。

Qt 的富文本 HTML 支持一些语义标签，但它们只是等效的装饰标签。

现代 HTML 和 CSS 在网页上使用的内容远不止我们在这里描述的，但我们所涵盖的内容足以理解 Qt 小部件使用的有限子集。 如果您想了解更多，请查看本章末尾的*进一步阅读*部分中的资源。

# 结构和标题标签

为了尝试丰富的文本标记，我们将为我们的下一个大型游戏*Fight Fighter 2*编写广告，并在 QTextBrowser 中查看它。 首先，从第四章中获取应用程序模板，*使用 QMainWindow 构建应用程序*，并将其命名为`qt_richtext_demo.py`。

在`MainWindow.__init__（）`中，像这样添加一个`QTextBrowser`对象作为主窗口小部件：

```py
        main = qtw.QTextBrowser()
        self.setCentralWidget(main)
        with open('fight_fighter2.html', 'r') as fh:
            main.insertHtml(fh.read())
```

`QTextBrowser`基于`QTextEdit`，但是只读并预先配置为导航超文本链接。 创建文本浏览器后，我们打开`fight_fighter2.html`文件，并使用`insertHtml（）`方法将其内容插入浏览器。 现在，我们可以编辑`fight_fighter2.html`并查看它在 PyQt 中的呈现方式。

在编辑器中打开`fight_fighter2.html`并从以下代码开始：

```py
<qt>
  <body>
    <h1>Fight Fighter 2</h1>
    <hr>
```

HTML 文档是按层次结构构建的，最外层的标签通常是`<html>`。 但是，当将 HTML 传递给基于`QTextDocument`的小部件时，我们还可以使用`<qt>`作为最外层的标签，这是一个好主意，因为它提醒我们正在编写 Qt 支持的 HTML 子集，而不是实际的 HTML。

在其中，我们有一个`<body>`标签。 这个标签也是可选的，但它将使未来的样式更容易。

接下来，我们在`<h1>`标签内有一个标题。这里的*H*代表标题，标签`<h1>`到`<h6>`表示从最外层到最内层的部分标题。这个标签将以更大更粗的字体呈现，表明它是部分的标题。

在标题之后，我们有一个`<hr>`标签来添加水平线。默认情况下，`<hr>`会产生一个单像素厚的黑线，但可以使用样式表进行自定义。

让我们添加以下常规文本内容：

```py
    <p>Everything you love about fight-fighter, but better!</p>
```

`<p>`标签，或段落标签，表示一块文本。在段落标签中不严格需要包含文本内容，但要理解 HTML 默认不会保留换行。如果你想要通过换行来分隔不同的段落，你需要将它们放在段落标签中。（你也可以插入`<br>`标签，但是段落标签被认为是更语义化的更干净的方法。）

接下来，添加第一个子标题，如下所示：

```py
    <h2>About</h2>
```

在`<h1>`下的任何子部分应该是`<h2>`；在`<h2>`内的任何子部分应该是`<h3>`，依此类推。标题标签是语义标签的例子，表示文档层次结构的级别。

永远不要根据它们产生的外观来选择标题级别——例如，不要在`<h1>`下使用`<h4>`，只是因为你想要更小的标题文本。使用它们语义化，并使用样式来调整外观（参见*字体、颜色、图片和样式*部分了解更多信息）。

# 排版标签

Qt 富文本支持许多标签来改变文本的基本外观，如下所示：

```py
  <p>Fight fighter 2 is the <i>amazing</i> sequel to <u>Fight Fighter</u>, an <s>intense</s> ultra-intense multiplayer action game from <b>FightSoft Software, LLC</b>.</p>
```

在这个例子中，我们使用了以下标签：

| 标签 | 结果 |
| --- | --- |
| `<i>` | *斜体* |
| `<b>` | **粗体** |
| `<u>` | 下划线 |
| `<s>` | 删除线 |

这些是装饰性标签，它们每个都会改变标签内文本的外观。除了这些标签，还支持一些用于文本大小和位置的较少使用的标签，包括以下内容：

```py
    <p>Fight Fighter 2's new Ultra-Action<sup>TM</sup> technology delivers low-latency combat like never before.   Best of all, at only $1.99<sub>USD</sub>, you <big>Huge Action</big> for a <small>tiny</small> price.</p>
```

在前面的例子中，我们可以看到`<sup>`和`<sub>`标签，分别提供上标和下标文本，以及`<big>`和`<small>`标签，分别提供稍微更大或更小的字体。

# 超链接

超链接也可以使用`<a>`（锚点）标签添加到 Qt 富文本中，如下所示：

```py
    <p>Download it today from
    <a href='http://www.example.com'>Example.com</a>!</p>
```

超链接的确切行为取决于显示超链接的部件和部件的设置。

`QTextBrowser`默认会尝试在部件内导航到超链接；但请记住，这些链接只有在它们是资源 URL 或本地文件路径时才会起作用。`QTextBrowser`缺乏网络堆栈，不能用于浏览互联网。

然而，它可以配置为在外部浏览器中打开 URL；在 Python 脚本中，添加以下代码到`MainWindow.__init__()`：

```py
      main.setOpenExternalLinks(True)
```

这利用`QDesktopServices.openUrl()`来在桌面的默认浏览器中打开锚点的`href`值。每当你想要在文档中支持外部超链接时，你应该配置这个设置。

外部超链接也可以在`QLabel`部件上进行配置，但不能在`QTextEdit`部件内进行配置。

文档也可以使用超链接来在文档内部导航，如下所示：

```py
    <p><a href='#Features'>Read about the features</a></p>

    <br><br><br><br><br><br>

    <a name='Features'></a>
    <h2>Features</h2>
    <p>Fight Fighter 2 is so amazing in so many ways:</p>
```

在这里，我们添加了一个指向`#Features`（带有井号）的锚点，然后是一些换行来模拟更多的内容。当用户点击链接时，它将滚动浏览器部件到具有`name`属性（而不是`href`）为`Features`的锚点标签（不带井号）。

这个功能对于提供可导航的目录表格非常有用。

# 列表和表格

列表和表格非常有用，可以以用户能够快速解析的方式呈现有序信息。

列表的一个例子如下：

```py
    <ul type=square>
      <li>More players at once!  Have up to 72 players.</li>
      <li>More teams!  Play with up to 16 teams!</li>
      <li>Easier installation!  Simply:<ol>
        <li>Copy the executable to your system.</li>
        <li>Run it!</li>
      </ol></li>
      <li>Sound and music! &gt;16 Million colors on some systems!</li>
    </ul>
```

Qt 富文本中的列表可以是有序或无序的。在上面的例子中，我们有一个无序列表（`<ul>`）。可选的`type`属性允许您指定应使用什么样的项目符号。在这种情况下，我们选择了`square`；无序列表的其他选项包括`circle`和`disc`。

使用`<li>`（列表项）标签指定列表中的每个项目。我们还可以在列表项内部嵌套一个列表，以创建一个子列表。在这种情况下，我们添加了一个有序列表，它将使用顺序号来指示新项目。有序列表还接受`type`属性；有效值为`a`（小写字母）、`A`（大写字母）或`1`（顺序号）。

在最后一个项目中的`&gt;`是 HTML 实体的一个例子。这些是特殊代码，用于显示 HTML 特殊字符，如尖括号，或非 ASCII 字符，如版权符号。实体以一个和号开始，以一个冒号结束，并包含一个指示要显示的字符的字符串。在这种情况下，`gt`代表*greater than*。可以在[`dev.w3.org/html5/html-author/charref`](https://dev.w3.org/html5/html-author/charref)找到官方实体列表，尽管并非所有实体都受`QTextDocument`支持。

创建 HTML 表格稍微复杂，因为它需要多层嵌套。表标签的层次结构如下：

+   表格本身由`<table>`标签定义

+   表的标题部分由`<thead>`标签定义

+   表的每一行（标题或数据）由`<tr>`（表行）标签定义

+   在每一行中，表格单元格由`<th>`（表头）标签或`<td>`（表数据）标签定义

让我们用以下代码开始一个表格：

```py
    <table border=2>
      <thead>
        <tr bgcolor='grey'>
        <th>System</th><th>Graphics</th><th>Sound</th></tr>
      </thead>
```

在上面的例子中，我们从开头的`<table>`标签开始。`border`属性指定了表格边框的宽度（以像素为单位）；在这种情况下，我们希望有一个两像素的边框。请记住，这个边框围绕每个单元格，不会合并（也就是说，不会与相邻单元格的边框合并），因此实际上，每个单元格之间将有一个四像素的边框。表格边框可以有不同的样式；默认情况下使用*ridge*样式，因此这个边框将被着色，看起来略微立体。

在`<thead>`部分，有一行表格，填满了表头单元格。通过设置行的`bgcolor`属性，我们可以将所有表头单元格的背景颜色更改为灰色。

现在，让我们用以下代码添加一些数据行：

```py
      <tr><td>Windows</td><td>DirectX 3D</td><td>24 bit PCM</td></tr>
      <tr><td>FreeDOS</td><td>256 color</td><td>8 bit Adlib PCM</td></tr>
      <tr><td>Commodore 64</td><td>256 color</td><td>SID audio</td></tr>
      <tr><td>TRS80</td>
        <td rowspan=2>Monochrome</td>
        <td rowspan=2>Beeps</td>
      </tr>
      <tr><td>Timex Sinclair</td></tr>
      <tr>
        <td>BBC Micro</td>
        <td colspan=2 bgcolor='red'>No support</td>
      </tr>
    </table>
```

在上面的例子中，行包含了用于实际表格数据的`<td>`单元格。请注意，我们可以在单个单元格上使用`rowspan`和`colspan`属性，使它们占用额外的行和列，并且`bgcolor`属性也可以应用于单个单元格。

可以将数据行包装在`<tbody>`标签中，以使其与`<thead>`部分区分开，但这实际上在 Qt 富文本 HTML 中没有任何有用的影响。

# 字体、颜色、图像和样式

可以使用`<font>`标签设置富文本字体，如下所示：

```py
    <h2>Special!</h2>

    <p>
      <font face='Impact' size=32 color='green'>Buy Now!</font>
      and receive <tt>20%</tt> off the regular price plus a
      <font face=Impact size=16 color='red'>Free sticker!</font>
    </p>
```

`<font>`对于那些学习了更现代 HTML 的人可能会感到陌生，因为它在 HTML 5 中已被弃用。但正如您所看到的，它可以用来设置标签中的文本的`face`、`size`和`color`属性。

`<tt>`（打字机类型）标签是使用等宽字体的简写，对于呈现内联代码、键盘快捷键和终端输出非常有用。

如果您更喜欢使用更现代的 CSS 样式字体配置，可以通过在块级标签（如`<div>`）上设置`style`属性来实现：

```py
    <div style='font-size: 16pt; font-weight: bold; color: navy;
                background-color: orange; padding: 20px;
                text-align: center;'>
                Don't miss this exciting offer!
    </div>
```

在`style`属性中，您可以设置任何支持的 CSS 值，以应用于该块。

# 文档范围的样式

Qt 富文本文档*不*支持 HTML `<style>`标签或`<link>`标签来设置文档范围的样式表。相反，您可以使用`QTextDocument`对象的`setDefaultStyleSheet()`方法来设置一个 CSS 样式表，该样式表将应用于所有查看的文档。

回到`MainWindow.__init__()`，添加以下内容：

```py
        main.document().setDefaultStyleSheet(
            'body {color: #333; font-size: 14px;} '
            'h2 {background: #CCF; color: #443;} '
            'h1 {background: #001133; color: white;} '
        )
```

但是，请注意，这必须在 HTML 插入小部件之前添加。`defaultStyleSheet`方法仅适用于新插入的 HTML。

还要注意，外观的某些方面不是文档的属性，而是小部件的属性。特别是，文档的背景颜色不能通过修改`body`的样式来设置。

相反，设置小部件的样式表，如下所示：

```py
        main.setStyleSheet('background-color: #EEF;')
```

请记住，小部件的样式表使用 QSS，而文档的样式表使用 CSS。区别是微小的，但在某些情况下可能会起作用。

# 图片

可以使用`<img>`标签插入图像，如下所示：

```py
    <div>
      <img src=logo.png width=400 height=100 />
    </div>
```

`src`属性应该是 Qt 支持的图像文件的文件或资源路径（有关图像格式支持的更多信息，请参见第六章，*Qt 应用程序的样式*）。`width`和`height`属性可用于强制指定特定大小。

# Qt 富文本和 Web HTML 之间的区别

如果您有网页设计或开发经验，您无疑已经注意到 Qt 的富文本标记与现代网页浏览器中使用的 HTML 之间的几个区别。在创建富文本时，重要的是要记住这些区别，所以让我们来看一下主要的区别。

首先，Qt 富文本基于 HTML 4 和 CSS 2.1；正如您所见，它包括一些已弃用的标签，如`<font>`，并排除了许多更现代的标签，如`<section>`或`<figure>`。

此外，Qt 富文本基于这些规范的一个子集，因此它不支持许多标签。例如，没有输入或表单相关的标签，如`<select>`或`<textarea>`。

`QTextDocument`在语法错误和大小写方面也比大多数网页浏览器渲染器更严格。例如，当设置默认样式表时，标签名称的大小写需要与文档中使用的大小写匹配，否则样式将不会应用。此外，未使用块级标签（如`<p>`、`<div>`等）包围内容可能会导致不可预测的结果。

简而言之，最好不要将 Qt 富文本标记视为真正的 HTML，而是将其视为一种类似但独立的标记语言。如果您对特定标记或样式指令是否受支持有任何疑问，请参阅[`doc.qt.io/qt-5/richtext-html-subset.html`](https://doc.qt.io/qt-5/richtext-html-subset.html)上的支持参考。

# 使用 QTextDocument 操作富文本

除了允许我们在标记中指定富文本外，Qt 还为我们提供了一个 API 来编程创建和操作富文本。这个 API 称为**Qt Scribe Framework**，它是围绕`QTextDocument`和`QTextCursor`类构建的。

演示如何使用`QTextDocument`和`QTextCursor`类创建文档，我们将构建一个简单的发票生成器应用程序。我们的应用程序将从小部件表单中获取数据，并使用它来编程生成富文本文档。

# 创建发票应用程序 GUI

获取我们的 PyQt 应用程序模板的最新副本，并将其命名为`invoice_maker.py`。我们将通过创建 GUI 元素开始我们的应用程序，然后开发实际构建文档的方法。

从一个数据输入表单类开始您的脚本，如下所示：

```py
class InvoiceForm(qtw.QWidget):

    submitted = qtc.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setLayout(qtw.QFormLayout())
        self.inputs = dict()
        self.inputs['Customer Name'] = qtw.QLineEdit()
        self.inputs['Customer Address'] = qtw.QPlainTextEdit()
        self.inputs['Invoice Date'] = qtw.QDateEdit(
            date=qtc.QDate.currentDate(), calendarPopup=True)
        self.inputs['Days until Due'] = qtw.QSpinBox(
            minimum=0, maximum=60, value=30)
        for label, widget in self.inputs.items():
            self.layout().addRow(label, widget)
```

与我们创建的大多数表单一样，这个类基于`QWidget`，并通过定义一个`submitted`信号来携带表单值的字典来开始。在这里，我们还向`QFormLayout`添加了各种输入，以输入基本的发票数据，如客户名称、客户地址和发票日期。

接下来，我们将添加`QTableWidget`以输入发票的行项目，如下所示：

```py
        self.line_items = qtw.QTableWidget(
            rowCount=10, columnCount=3)
        self.line_items.setHorizontalHeaderLabels(
            ['Job', 'Rate', 'Hours'])
        self.line_items.horizontalHeader().setSectionResizeMode(
            qtw.QHeaderView.Stretch)
        self.layout().addRow(self.line_items)
        for row in range(self.line_items.rowCount()):
            for col in range(self.line_items.columnCount()):
                if col > 0:
                    w = qtw.QSpinBox(minimum=0)
                    self.line_items.setCellWidget(row, col, w)
```

该表格小部件的每一行都包含任务的描述、工作的费率和工作的小时数。因为最后两列中的值是数字，所以我们使用表格小部件的`setCellWidget()`方法来用`QSpinBox`小部件替换这些单元格中的默认`QLineEdit`小部件。

最后，我们将使用以下代码添加一个`submit`按钮：

```py
        submit = qtw.QPushButton('Create Invoice', clicked=self.on_submit)
        self.layout().addRow(submit)
```

`submit`按钮调用一个`on_submit()`方法，开始如下：

```py
   def on_submit(self):
        data = {
            'c_name': self.inputs['Customer Name'].text(),
            'c_addr': self.inputs['Customer Address'].toPlainText(),
            'i_date': self.inputs['Invoice Date'].date().toString(),
            'i_due': self.inputs['Invoice Date'].date().addDays(
                self.inputs['Days until Due'].value()).toString(),
            'i_terms': '{} days'.format(
                self.inputs['Days until Due'].value())
        }
```

该方法只是简单地提取输入表单中输入的值，进行一些计算，并使用`submitted`信号发射生成的数据`dict`。在这里，我们首先通过使用每个小部件的适当方法将表单的每个输入小部件的值放入 Python 字典中。

接下来，我们需要检索行项目的数据，如下所示：

```py
       data['line_items'] = list()
        for row in range(self.line_items.rowCount()):
            if not self.line_items.item(row, 0):
                continue
            job = self.line_items.item(row, 0).text()
            rate = self.line_items.cellWidget(row, 1).value()
            hours = self.line_items.cellWidget(row, 2).value()
            total = rate * hours
            row_data = [job, rate, hours, total]
            if any(row_data):
                data['line_items'].append(row_data)
```

对于表格小部件中具有描述的每一行，我们将检索所有数据，通过将费率和工时相乘来计算总成本，并将所有数据附加到我们的`data`字典中的列表中。

最后，我们将计算一个总成本，并使用以下代码将其附加到：

```py
        data['total_due'] = sum(x[3] for x in data['line_items'])
        self.submitted.emit(data)
```

在每一行的成本总和之后，我们将其添加到数据字典中，并使用数据发射我们的`submitted`信号。

这就是我们的`form`类，所以让我们在`MainWindow`中设置主应用程序布局。在`MainWindow.__init__()`中，添加以下代码：

```py
        main = qtw.QWidget()
        main.setLayout(qtw.QHBoxLayout())
        self.setCentralWidget(main)

        form = InvoiceForm()
        main.layout().addWidget(form)

        self.preview = InvoiceView()
        main.layout().addWidget(self.preview)

        form.submitted.connect(self.preview.build_invoice)
```

主小部件被赋予一个水平布局，以包含格式化发票的表单和视图小部件。然后，我们将表单的`submitted`信号连接到视图对象上将创建的`build_invoice()`方法。

这是应用程序的主要 GUI 和逻辑；现在我们只需要创建我们的`InvoiceView`类。

# 构建 InvoiceView

`InvoiceView`类是所有繁重工作发生的地方；我们将其基于只读的`QTextEdit`小部件，并且它将包含一个`build_invoice()`方法，当使用数据字典调用时，将使用 Qt Scribe 框架构建格式化的发票文档。

让我们从构造函数开始，如下例所示：

```py
class InvoiceView(qtw.QTextEdit):

    dpi = 72
    doc_width = 8.5 * dpi
    doc_height = 11 * dpi

    def __init__(self):
        super().__init__(readOnly=True)
        self.setFixedSize(qtc.QSize(self.doc_width, self.doc_height))
```

首先，我们为文档的宽度和高度定义了类变量。我们选择这些值是为了给我们一个标准的美国信件大小文档的纵横比，适合于普通计算机显示器的合理尺寸。在构造函数中，我们使用计算出的值来设置小部件的固定大小。这是我们在构造函数中需要做的所有事情，所以现在是时候开始真正的工作了——构建一个文档。

让我们从`build_invoice()`开始，如下所示：

```py
    def build_invoice(self, data):
        document = qtg.QTextDocument()
        self.setDocument(document)
        document.setPageSize(qtc.QSizeF(self.doc_width, self.doc_height))
```

正如您在前面的示例中所看到的，该方法首先创建一个新的`QTextDocument`对象，并将其分配给视图的`document`属性。然后，使用在类定义中计算的文档尺寸设置`pageSize`属性。请注意，我们基于 QTextEdit 的视图已经有一个我们可以检索的`document`对象，但我们正在创建一个新的对象，以便该方法每次调用时都会以空文档开始。

使用`QTextDocument`编辑文档可能会感觉有点不同于我们创建 GUI 表单的方式，通常我们会创建对象，然后配置并将它们放置在布局中。

相反，`QTextDocument`的工作流更像是一个文字处理器：

+   有一个`cursor`始终指向文档中的某个位置

+   有一个活动文本样式、段落样式或另一个块级样式，其设置将应用于输入的任何内容

+   要添加内容，用户首先要定位光标，配置样式，最后创建内容

因此，显然，第一步是获取光标的引用；使用以下代码来实现：

```py
        cursor = qtg.QTextCursor(document)
```

`QTextCursor`对象是我们用来插入内容的工具，并且它有许多方法可以将不同类型的元素插入文档中。

例如，在这一点上，我们可以开始插入文本内容，如下所示：

```py
        cursor.insertText("Invoice, woohoo!")
```

然而，在我们开始向文档中写入内容之前，我们应该构建一个基本的文档框架来进行工作。为了做到这一点，我们需要了解`QTextDocument`对象的结构。

# QTextDocument 结构

就像 HTML 文档一样，`QTextDocument`对象是一个分层结构。它由**框架**、**块**和**片段**组成，定义如下：

+   框架由`QTextFrame`对象表示，是文档的矩形区域，可以包含任何类型的内容，包括其他框架。在我们的层次结构顶部是**根框架**，它包含了文档的所有内容。

+   一个块，由`QTextBlock`对象表示，是由换行符包围的文本区域，例如段落或列表项。

+   片段，由`QTextFragment`对象表示，是块内的连续文本区域，共享相同的文本格式。例如，如果您有一个句子中包含一个粗体字，那么代表三个文本片段：粗体字之前的句子，粗体字，和粗体字之后的句子。

+   其他项目，如表格、列表和图像，都是从这些前面的类中派生出来的。

我们将通过在根框架下插入一组子框架来组织我们的文档，以便我们可以轻松地导航到我们想要处理的文档部分。我们的文档将有以下四个框架：

+   **标志框架**将包含公司标志和联系信息

+   **客户地址框架**将保存客户姓名和地址

+   **条款框架**将保存发票条款和条件的列表

+   **行项目框架**将保存行项目和总计的表格

让我们创建一些文本框架来概述我们文档的结构。我们将首先保存对根框架的引用，以便在创建子框架后可以轻松返回到它，如下所示：

```py
        root = document.rootFrame()
```

既然我们有了这个，我们可以通过调用以下命令在任何时候为根框架的末尾检索光标位置：

```py
        cursor.setPosition(root.lastPosition())
```

光标的`setPosition()`方法将我们的光标放在任何给定位置，根框架的`lastPosition()`方法检索根框架末尾的位置。

现在，让我们定义第一个子框架，如下所示：

```py
        logo_frame_fmt = qtg.QTextFrameFormat()
        logo_frame_fmt.setBorder(2)
        logo_frame_fmt.setPadding(10)
        logo_frame = cursor.insertFrame(logo_frame_fmt)
```

框架必须使用定义其格式的`QTextFrameFormat`对象创建，因此在我们写框架之前，我们必须定义我们的格式。不幸的是，框架格式的属性不能使用关键字参数设置，因此我们必须使用 setter 方法进行配置。在这个例子中，我们设置了框架周围的两像素边框，以及十像素的填充。

一旦格式对象被创建，我们调用光标的`insertFrame()`方法来使用我们配置的格式创建一个新框架。

`insertFrame()`返回创建的`QTextFrame`对象，并且将我们文档的光标定位在新框架内。由于我们还没有准备好向这个框架添加内容，并且我们不想在其中创建下一个框架，所以我们需要使用以下代码返回到根框架之前创建下一个框架：

```py
        cursor.setPosition(root.lastPosition())
        cust_addr_frame_fmt = qtg.QTextFrameFormat()
        cust_addr_frame_fmt.setWidth(self.doc_width * .3)
        cust_addr_frame_fmt.setPosition(qtg.QTextFrameFormat.FloatRight)
        cust_addr_frame = cursor.insertFrame(cust_addr_frame_fmt)
```

在上面的例子中，我们使用框架格式来将此框架的宽度设置为文档宽度的三分之一，并使其浮动到右侧。*浮动*文档框架意味着它将被推到文档的一侧，其他内容将围绕它流动。

现在，我们将添加术语框架，如下所示：

```py
        cursor.setPosition(root.lastPosition())
        terms_frame_fmt = qtg.QTextFrameFormat()
        terms_frame_fmt.setWidth(self.doc_width * .5)
        terms_frame_fmt.setPosition(qtg.QTextFrameFormat.FloatLeft)
        terms_frame = cursor.insertFrame(terms_frame_fmt)
```

这一次，我们将使框架的宽度为文档宽度的一半，并将其浮动到左侧。

理论上，这两个框架应该相邻。实际上，由于`QTextDocument`类渲染中的一个怪癖，第二个框架的顶部将在第一个框架的顶部下面一行。这对我们的演示来说没问题，但如果您需要实际的列，请改用表格。

最后，让我们添加一个框架来保存我们的行项目表格，如下所示：

```py
        cursor.setPosition(root.lastPosition())
        line_items_frame_fmt = qtg.QTextFrameFormat()
        line_items_frame_fmt.setMargin(25)
        line_items_frame = cursor.insertFrame(line_items_frame_fmt)
```

再次，我们将光标移回到根框架并插入一个新框架。这次，格式将在框架上添加 25 像素的边距。

请注意，如果我们不想对`QTextFrameFormat`对象进行任何特殊配置，我们就不必这样做，但是*必须*为每个框架创建一个对象，并且*必须*在创建新框架之前对它们进行任何配置。请注意，如果您有许多具有相同配置的框架，也可以重用框架格式。

# 字符格式

就像框架必须使用框架格式创建一样，文本内容必须使用**字符格式**创建，该格式定义了文本的字体和对齐等属性。在我们开始向框架添加内容之前，我们应该定义一些常见的字符格式，以便在文档的不同部分使用。

这是使用`QTextCharFormat`类完成的，如下所示：

```py
        std_format = qtg.QTextCharFormat()

        logo_format = qtg.QTextCharFormat()
        logo_format.setFont(
            qtg.QFont('Impact', 24, qtg.QFont.DemiBold))
        logo_format.setUnderlineStyle(
            qtg.QTextCharFormat.SingleUnderline)
        logo_format.setVerticalAlignment(
            qtg.QTextCharFormat.AlignMiddle)

        label_format = qtg.QTextCharFormat()
        label_format.setFont(qtg.QFont('Sans', 12, qtg.QFont.Bold))
```

在前面的示例中，我们创建了以下三种格式：

+   `std_format`，将用于常规文本。我们不会改变默认设置。

+   `logo_format`，将用于我们的公司标志。我们正在自定义其字体并添加下划线，以及设置其垂直对齐。

+   `label_format`，将用于标签；它们将使用 12 号字体并加粗。

请注意，`QTextCharFormat`允许您直接使用 setter 方法进行许多字体配置，或者甚至可以配置一个`QFont`对象分配给格式。我们将在文档的其余部分添加文本内容时使用这三种格式。

# 添加基本内容

现在，让我们使用以下命令向我们的`logo_frame`添加一些基本内容：

```py
        cursor.setPosition(logo_frame.firstPosition())
```

就像我们调用根框架的`lastPosition`方法来获取其末尾的位置一样，我们可以调用标志框架的`firstPosition()`方法来获取框架开头的位置。一旦在那里，我们可以插入内容，比如标志图像，如下所示：

```py
        cursor.insertImage('nc_logo.png')
```

图片可以像这样插入——通过将图像的路径作为字符串传递。然而，这种方法在配置方面提供的内容很少，所以让我们尝试一种稍微复杂的方法：

```py
        logo_image_fmt = qtg.QTextImageFormat()
        logo_image_fmt.setName('nc_logo.png')
        logo_image_fmt.setHeight(48)
        cursor.insertImage(logo_image_fmt, qtg.QTextFrameFormat.FloatLeft)
```

通过使用`QTextImageFormat`对象，我们可以首先配置图像的各个方面，如其高度和宽度，然后将其添加到枚举常量指定其定位策略。在这种情况下，`FloatLeft`将导致图像与框架的左侧对齐，并且随后的文本将围绕它。

现在，让我们在块中写入以下文本：

```py
        cursor.insertText('   ')
        cursor.insertText('Ninja Coders, LLC', logo_format)
        cursor.insertBlock()
        cursor.insertText('123 N Wizard St, Yonkers, NY 10701', std_format)
```

使用我们的`logo_format`，我们已经编写了一个包含公司名称的文本片段，然后插入了一个新块，这样我们就可以在另一行上添加包含地址的另一个片段。请注意，传递字符格式是可选的；如果我们不这样做，片段将以当前活动格式插入，就像在文字处理器中一样。

处理完我们的标志后，现在让我们来处理客户地址块，如下所示：

```py
        cursor.setPosition(cust_addr_frame.lastPosition())
```

文本块可以像框架和字符一样具有格式。让我们使用以下代码创建一个文本块格式，用于我们的客户地址：

```py
        address_format = qtg.QTextBlockFormat()
        address_format.setAlignment(qtc.Qt.AlignRight)
        address_format.setRightMargin(25)
        address_format.setLineHeight(
            150, qtg.QTextBlockFormat.ProportionalHeight)
```

文本块格式允许您更改文本段落中更改的设置：边距、行高、缩进和对齐。在这里，我们将文本对齐设置为右对齐，右边距为 25 像素，行高为 1.5 行。在`QTextDocument`中有多种指定高度的方法，`setLineHeight()`的第二个参数决定了传入值的解释方式。在这种情况下，我们使用`ProportionalHeight`模式，它将传入的值解释为行高的百分比。

我们可以将我们的块格式对象传递给任何`insertBlock`调用，如下所示：

```py
        cursor.insertBlock(address_format)
        cursor.insertText('Customer:', label_format)
        cursor.insertBlock(address_format)
        cursor.insertText(data['c_name'], std_format)
        cursor.insertBlock(address_format)
        cursor.insertText(data['c_addr'])
```

每次插入一个块，就像开始一个新段落一样。我们的多行地址字符串将被插入为一个段落，但请注意，它仍将被间隔为 1.5 行。

# 插入列表

我们的发票条款将以无序项目列表的形式呈现。有序和无序列表可以使用光标的`insertList()`方法插入到`QTextDocument`中，如下所示：

```py
        cursor.setPosition(terms_frame.lastPosition())
        cursor.insertText('Terms:', label_format)
        cursor.insertList(qtg.QTextListFormat.ListDisc)
```

`insertList()`的参数可以是`QTextListFormat`对象，也可以是`QTextListFormat.Style`枚举中的常量。在这种情况下，我们使用了后者，指定我们希望使用圆盘样式的项目列表。

列表格式的其他选项包括`ListCircle`和`ListSquare`用于无序列表，以及`ListDecimal`、`ListLowerAlpha`、`ListUpperAlpha`、`ListUpperRoman`和`ListLowerRoman`用于有序列表。

现在，我们将定义要插入到我们的列表中的一些项目，如下所示：

```py
        term_items = (
            f'<b>Invoice dated:</b> {data["i_date"]}',
            f'<b>Invoice terms:</b> {data["i_terms"]}',
            f'<b>Invoice due:</b> {data["i_due"]}',
        )
```

请注意，在上面的示例中，我们使用的是标记，而不是原始字符串。在使用`QTextCursor`创建文档时，仍然可以使用标记；但是，您需要通过调用`insertHtml()`而不是`insertText()`来告诉光标它正在插入 HTML 而不是纯文本，如下例所示：

```py
        for i, item in enumerate(term_items):
            if i > 0:
                cursor.insertBlock()
            cursor.insertHtml(item)
```

在调用`insertList()`之后，我们的光标位于第一个列表项内，因此现在我们需要调用`insertBlock()`来到达后续项目（对于第一个项目，我们不需要这样做，因为我们已经处于项目符号中，因此需要进行`if i > 0`检查）。

与`insertText()`不同，`insertHtml()`不接受字符格式对象。您必须依靠您的标记来确定格式。

# 插入表格

我们要在发票中插入的最后一件事是包含我们的行项目的表格。`QTextTable`是`QTextFrame`的子类，就像框架一样，我们需要在创建表格本身之前为其创建格式对象。

我们需要的类是`QTextTableFormat`类：

```py
        table_format = qtg.QTextTableFormat()
        table_format.setHeaderRowCount(1)
        table_format.setWidth(
            qtg.QTextLength(qtg.QTextLength.PercentageLength, 100))
```

在这里，我们配置了`headerRowCount`属性，该属性表示第一行是标题行，并且应在每页顶部重复。这相当于在标记中将第一行放在`<thead>`标记中。

我们还设置了宽度，但是我们没有使用像素值，而是使用了`QTextLength`对象。这个类的命名有些令人困惑，因为它不是特指文本的长度，而是指您可能在`QTextDocument`中需要的任何通用长度。`QTextLength`对象可以是百分比、固定或可变类型；在这种情况下，我们指定了值为`100`或 100%的`PercentageLength`。

现在，让我们使用以下代码插入我们的表格：

```py
        headings = ('Job', 'Rate', 'Hours', 'Cost')
        num_rows = len(data['line_items']) + 1
        num_cols = len(headings)

        cursor.setPosition(line_items_frame.lastPosition())
        table = cursor.insertTable(num_rows, num_cols, table_format)
```

在将表格插入`QTextDocument`时，我们不仅需要定义格式，还需要指定行数和列数。为此，我们创建了标题的元组，然后通过计算行项目列表的长度（为标题行添加 1），以及标题元组的长度来计算行数和列数。

然后，我们需要将光标定位在行项目框中并插入我们的表格。就像其他插入方法一样，`insertTable()`将我们的光标定位在插入的项目内部，即第一行的第一列。

现在，我们可以使用以下代码插入我们的标题行：

```py
        for heading in headings:
            cursor.insertText(heading, label_format)
            cursor.movePosition(qtg.QTextCursor.NextCell)
```

到目前为止，我们一直通过将确切位置传递给`setPosition()`来定位光标。`QTextCursor`对象还具有`movePosition()`方法，该方法可以接受`QTextCursor.MoveOperation`枚举中的常量。该枚举定义了表示约两打不同光标移动的常量，例如`StartOfLine`、`PreviousBlock`和`NextWord`。在这种情况下，`NextCell`移动将我们带到表格中的下一个单元格。

我们可以使用相同的方法来插入我们的数据，如下所示：

```py
        for row in data['line_items']:
            for col, value in enumerate(row):
                text = f'${value}' if col in (1, 3) else f'{value}'
                cursor.insertText(text, std_format)
                cursor.movePosition(qtg.QTextCursor.NextCell)
```

在这种情况下，我们正在迭代数据列表中每一行的每一列，并使用`insertText()`将数据添加到单元格中。如果列号为`1`或`3`，即货币值，我们需要在显示中添加货币符号。

我们还需要添加一行来保存发票的总计。要在表格中添加额外的行，我们可以使用以下`QTextTable.appendRows()`方法：

```py
        table.appendRows(1)
```

为了将光标定位到新行中的特定单元格中，我们可以使用表对象的`cellAt()`方法来检索一个`QTableCell`对象，然后使用该对象的`lastCursorPosition()`方法，该方法返回一个位于单元格末尾的新光标，如下所示：

```py
        cursor = table.cellAt(num_rows, 0).lastCursorPosition()
        cursor.insertText('Total', label_format)
        cursor = table.cellAt(num_rows, 3).lastCursorPosition()
        cursor.insertText(f"${data['total_due']}", label_format)
```

这是我们需要写入发票文档的最后一部分内容，所以让我们继续测试一下。

# 完成和测试

现在，如果您运行您的应用程序，填写字段，然后点击创建发票，您应该会看到类似以下截图的内容：

![](img/79fad3bf-5ad0-4208-a822-82277ebe1785.png)

看起来不错！当然，如果我们无法打印或导出发票，那么这张发票对我们就没有什么用处。因此，在下一节中，我们将看看如何处理文档的打印。

# 打印富文本

没有什么能像被要求实现打印机支持那样让程序员心生恐惧。将原始的数字位转化为纸上的墨迹在现实生活中是混乱的，在软件世界中也可能一样混乱。幸运的是，Qt 提供了`QtPrintSupport`模块，这是一个跨平台的打印系统，可以轻松地将`QTextDocument`转换为硬拷贝格式，无论我们使用的是哪个操作系统。

# 更新发票应用程序以支持打印

在我们将文档的尺寸硬编码为 8.5×11 时，美国以外的读者几乎肯定会感到沮丧，但不要担心——我们将进行一些更改，以便根据用户选择的文档尺寸来设置尺寸。

在`InvoiceView`类中，创建以下新方法`set_page_size()`，以设置页面大小：

```py
    def set_page_size(self, qrect):
        self.doc_width = qrect.width()
        self.doc_height = qrect.height()
        self.setFixedSize(qtc.QSize(self.doc_width, self.doc_height))
        self.document().setPageSize(
            qtc.QSizeF(self.doc_width, self.doc_height))
```

该方法将接收一个`QRect`对象，从中提取宽度和高度值以更新文档的设置、小部件的固定大小和文档的页面大小。

在`MainWindow.__init__()`中，添加一个工具栏来控制打印，并设置以下操作：

```py
        print_tb = self.addToolBar('Printing')
        print_tb.addAction('Configure Printer', self.printer_config)
        print_tb.addAction('Print Preview', self.print_preview)
        print_tb.addAction('Print dialog', self.print_dialog)
        print_tb.addAction('Export PDF', self.export_pdf)
```

当我们设置每个打印过程的各个方面时，我们将实现这些回调。

# 配置打印机

打印始于一个`QtPrintSupport.QPrinter`对象，它代表内存中的打印文档。在 PyQt 中打印的基本工作流程如下：

1.  创建一个`QPrinter`对象

1.  使用其方法或打印机配置对话框配置`QPrinter`对象

1.  将`QTextDocument`打印到`QPrinter`对象

1.  将`QPrinter`对象传递给操作系统的打印对话框，用户可以使用物理打印机进行打印

在`MainWindow.__init__()`中，让我们创建我们的`QPrinter`对象，如下所示：

```py
        self.printer = qtps.QPrinter()
        self.printer.setOrientation(qtps.QPrinter.Portrait)
        self.printer.setPageSize(qtg.QPageSize(qtg.QPageSize.Letter))
```

打印机创建后，我们可以配置许多属性；在这里，我们只是设置了方向和页面大小（再次设置为美国信纸默认值，但可以随意更改为您喜欢的纸张大小）。

您可以通过`QPrinter`方法配置打印机设置对话框中的任何内容，但理想情况下，我们宁愿让用户做出这些决定。因此，让我们实现以下`printer_config()`方法：

```py
    def printer_config(self):
        dialog = qtps.QPageSetupDialog(self.printer, self)
        dialog.exec()
```

`QPageSetupDialog`对象是一个`QDialog`子类，显示了`QPrinter`对象可用的所有选项。我们将我们的`QPrinter`对象传递给它，这将导致对话框中所做的任何更改应用于该打印机对象。在 Windows 和 macOS 上，Qt 将默认使用操作系统提供的打印对话框；在其他平台上，将使用一个特定于 Qt 的对话框。

现在用户可以配置纸张大小，我们需要允许`InvoiceView`在每次更改后重置页面大小。因此，让我们在`MainWindow`中添加以下方法：

```py
    def _update_preview_size(self):
        self.preview.set_page_size(
            self.printer.pageRect(qtps.QPrinter.Point))
```

`QPrinter.pageRect()`方法提取了一个`QRect`对象，定义了配置的页面大小。由于我们的`InvoiceView.set_page_size()`方法接受一个`QRect`，我们只需要将这个对象传递给它。

请注意，我们已经将一个常量传递给`pageRect()`，表示我们希望以**点**为单位获取大小。点是英寸的 1/72，因此我们的小部件大小将是物理页面尺寸的 72 倍英寸。如果您想要自己计算以缩放小部件大小，您可以请求以各种单位（包括毫米、Picas、英寸等）获取页面矩形。

不幸的是，`QPrinter`对象不是`QObject`的后代，因此我们无法使用信号来确定其参数何时更改。

现在，在`printer_config()`的末尾添加对`self._update_preview_size()`的调用，这样每当用户配置页面时都会被调用。您会发现，如果您在打印机配置对话框中更改纸张的大小，您的预览小部件将相应地调整大小。

# 打印一页

在我们实际打印文档之前，我们必须首先将`QTextDocument`打印到`QPrinter`对象中。这是通过将打印机对象传递给文档的`print()`方法来完成的。

我们将创建以下方法来为我们执行这些操作：

```py
    def _print_document(self):
        self.preview.document().print(self.printer)
```

请注意，这实际上并不会导致您的打印设备开始在页面上放墨水-它只是将文档加载到`QPrinter`对象中。

要实际将其打印到纸张上，需要打印对话框；因此，在`MainView`中添加以下方法：

```py
    def print_dialog(self):
        self._print_document()
        dialog = qtps.QPrintDialog(self.printer, self)
        dialog.exec()
        self._update_preview_size()
```

在这个方法中，我们首先调用我们的内部方法将文档加载到`QPrinter`对象中，然后将对象传递给`QPrintDialog`对象，通过调用其`exec()`方法来执行。这将显示打印对话框，用户可以使用它将文档发送到物理打印机。

如果您不需要打印对话框来阻止程序执行，您可以调用其`open()`方法。在前面的示例中，我们正在阻止，以便在对话框关闭后执行操作。

对话框关闭后，我们调用`_update_preview_size()`来获取新的纸张大小并更新我们的小部件和文档。理论上，我们可以将对话框的`accepted`信号连接到该方法，但实际上，可能会出现一些竞争条件导致失败。

# 打印预览

没有人喜欢浪费纸张打印不正确的东西，所以我们应该添加一个`print_preview`函数。`QPrintPreviewDialog`就是为此目的而存在的，并且与其他打印对话框非常相似，如下所示：

```py
    def print_preview(self):
        dialog = qtps.QPrintPreviewDialog(self.printer, self)
        dialog.paintRequested.connect(self._print_document)
        dialog.exec()
        self._update_preview_size()
```

再次，我们只需要将打印机对象传递给对话框的构造函数并调用`exec()`。我们还需要将对话框的`paintRequested`信号连接到一个插槽，该插槽将更新`QPrinter`中的文档，以便对话框可以确保预览是最新的。在这里，我们将其连接到我们的`_print_document()`方法，该方法正是所需的。

# 导出为 PDF

在这个无纸化的数字时代，PDF 文件已经取代了许多用途的硬拷贝，因此，添加一个简单的导出到 PDF 功能总是一件好事。`QPrinter`可以轻松为我们做到这一点。

在`MainView`中添加以下`export_pdf()`方法：

```py
    def export_pdf(self):
        filename, _ = qtw.QFileDialog.getSaveFileName(
            self, "Save to PDF", qtc.QDir.homePath(), "PDF Files (*.pdf)")
        if filename:
            self.printer.setOutputFileName(filename)
            self.printer.setOutputFormat(qtps.QPrinter.PdfFormat)
            self._print_document()
```

在这里，我们将首先要求用户提供文件名。如果他们提供了文件名，我们将使用该文件名配置我们的`QPrinter`对象，将输出格式设置为`PdfFormat`，然后打印文档。在写入文件时，`QTextDocument.print()`将负责写入数据并为我们保存文件，因此我们在这里不需要做其他事情。

这涵盖了发票程序的所有打印需求！花些时间测试这个功能，看看它如何与您的打印机配合使用。

# 总结

在本章中，您掌握了在 PyQt5 中处理富文本文档的方法。您学会了如何使用 Qt 的 HTML 子集在`QLabel`、`QTextEdit`和`QTextBrowser`小部件中添加富文本格式。您通过使用`QTextCursor`接口编程方式构建了`QTextDocument`。最后，您学会了如何使用 Qt 的打印支持模块将`QTextDocument`对象带入现实世界。

在第十二章中，*使用 QPainter 创建 2D 图形*，你将学习一些二维图形的高级概念。你将学会如何使用`QPainter`对象来创建图形，构建自定义小部件，并创建动画。

# 问题

尝试使用这些问题来测试你对本章的了解：

1.  以下 HTML 显示的不如你希望的那样。找出尽可能多的错误：

```py
<table>
<thead background=#EFE><th>Job</th><th>Status</th></thead>
<tr><td>Backup</td>
<font text-color='green'>Success!</font></td></tr>
<tr><td>Cleanup<td><font text-style='bold'>Fail!</font></td></tr>
</table>
```

1.  以下 Qt HTML 代码有什么问题？

```py
<p>There is nothing <i>wrong</i> with your television <b>set</p></b>
<table><row><data>french fries</data>
<data>$1.99</data></row></table>
<font family='Tahoma' color='#235499'>Can you feel the <strikethrough>love</strikethrough>code tonight?</font>
<label>Username</label><input type='text' name='username'></input>
<img source='://mypix.png'>My picture</img>
```

1.  这段代码应该实现一个目录。为什么它不能正确工作？

```py
   <ul>
     <li><a href='Section1'>Section 1</a></li>
     <li><a href='Section2'>Section 2</a></li>
   </ul>
   <div id=Section1>
     <p>This is section 1</p>
   </div>
   <div id=Section2>
     <p>This is section 2</p>
   </div>
```

1.  使用`QTextCursor`，在文档的右侧添加一个侧边栏。解释一下你会如何做到这一点。

1.  你正在尝试使用`QTextCursor`创建一个文档。它应该有一个顶部和底部的框架；在顶部框架中应该有一个标题，在底部框架中应该有一个无序列表。请纠正以下代码，使其实现这一点：

```py
   document = qtg.QTextDocument()
   cursor = qtg.QTextCursor(document)
   top_frame = cursor.insertFrame(qtg.QTextFrameFormat())
   bottom_frame = cursor.insertFrame(qtg.QTextFrameFormat())

   cursor.insertText('This is the title')
   cursor.movePosition(qtg.QTextCursor.NextBlock)
   cursor.insertList(qtg.QTextListFormat())
   for item in ('thing 1', 'thing 2', 'thing 3'):
       cursor.insertText(item)
```

1.  你正在创建自己的`QPrinter`子类以在页面大小更改时添加一个信号。以下代码会起作用吗？

```py
   class MyPrinter(qtps.QPrinter):

       page_size_changed = qtc.pyqtSignal(qtg.QPageSize)

       def setPageSize(self, size):
           super().setPageSize(size)
           self.page_size_changed.emit(size)
```

1.  `QtPrintSupport`包含一个名为`QPrinterInfo`的类。使用这个类，在你的系统上打印出所有打印机的名称、制造商、型号和默认页面大小的列表。

# 进一步阅读

有关更多信息，请参考以下链接：

+   Qt 对 Scribe 框架的概述可以在[`doc.qt.io/qt-5/richtext.html`](https://doc.qt.io/qt-5/richtext.html)找到

+   可以使用`QAbstractTextDocumentLayout`和`QTextLine`类来定义高级文档布局；关于如何使用这些类的信息可以在[`doc.qt.io/qt-5/richtext-layouts.html`](https://doc.qt.io/qt-5/richtext-layouts.html)找到

+   Qt 的打印系统概述可以在[`doc.qt.io/qt-5/qtprintsupport-index.html`](https://doc.qt.io/qt-5/qtprintsupport-index.html)找到
