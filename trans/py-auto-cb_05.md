# 生成精彩的报告

在本章中，我们将涵盖以下配方：

+   在纯文本中创建简单报告

+   使用模板生成报告

+   在 Markdown 中格式化文本

+   编写基本的 Word 文档

+   为 Word 文档设置样式

+   在 Word 文档中生成结构

+   向 Word 文档添加图片

+   编写简单的 PDF 文档

+   构建 PDF

+   聚合 PDF 报告

+   给 PDF 加水印和加密

# 介绍

在本章中，我们将看到如何编写文档并执行基本操作，如处理不同格式的模板，如纯文本和 Markdown。我们将花费大部分时间处理常见且有用的格式，如 Word 和 PDF。

# 在纯文本中创建简单报告

最简单的报告是生成一些文本并将其存储在文件中。

# 准备工作

对于这个配方，我们将以文本格式生成简要报告。要存储的数据将在一个字典中。

# 如何操作...

1.  导入`datetime`：

```py
>>> from datetime import datetime
```

1.  使用文本格式创建报告模板：

```py
>>> TEMPLATE = '''
Movies report
-------------

Date: {date}
Movies seen in the last 30 days: {num_movies}
Total minutes: {total_minutes}
'''
```

1.  创建一个包含要存储的值的字典。请注意，这是将在报告中呈现的数据：

```py
>>> data = {
    'date': datetime.utcnow(),
 'num_movies': 3,
    'total_minutes': 376,
}
```

1.  撰写报告，将数据添加到模板中：

```py
>>> report = TEMPLATE.format(**data)
```

1.  创建一个带有当前日期的新文件，并存储报告：

```py
>>> FILENAME_TMPL = "{date}_report.txt"
>>> filename = FILENAME_TMPL.format(date=data['date'].strftime('%Y-%m-%d'))
>>> filename
2018-06-26_report.txt
>>> with open(filename, 'w') as file:
...     file.write(report)
```

1.  检查新创建的报告：

```py
$ cat 2018-06-26_report.txt

Movies report
-------------

Date: 2018-06-26 23:40:08.737671
Movies seen in the last 30 days: 3
Total minutes: 376
```

# 工作原理...

*如何操作...*部分的第 2 步和第 3 步设置了一个简单的模板，并添加了包含报告中所有数据的字典。然后，在第 4 步，这两者被合并成一个特定的报告。

在第 4 步中，将字典与模板结合。请注意，字典中的键对应模板中的参数。诀窍是在`format`调用中使用双星号来解压字典，将每个键作为参数传递给`format()`。

在第 5 步中，生成的报告（一个字符串）存储在一个新创建的文件中，使用`with`上下文管理器。`open()`函数根据打开模式`w`创建一个新文件，并在块期间保持打开状态，该块将数据写入文件。退出块时，文件将被正确关闭。

打开模式确定如何打开文件，无论是读取还是写入，以及文件是文本还是二进制。`w`模式打开文件以进行写入，如果文件已存在，则覆盖它。小心不要错误删除现有文件！

第 6 步检查文件是否已使用正确的数据创建。

# 还有更多...

文件名使用今天的日期创建，以最小化覆盖值的可能性。日期的格式从年份开始，以天结束，已选择文件可以按正确顺序自然排序。

即使出现异常，`with`上下文管理器也会关闭文件。如果出现异常，它将引发`IOError`异常。

在写作中一些常见的异常可能是权限问题，硬盘已满，或路径问题（例如，尝试在不存在的目录中写入）。

请注意，文件可能在关闭或显式刷新之前未完全提交到磁盘。一般来说，处理文件时这不是问题，但如果尝试打开一个文件两次（一次用于读取，一次用于写入），则需要牢记这一点。

# 另请参阅

+   *使用模板生成报告*配方

+   *在 Markdown 中格式化文本*配方

+   *聚合 PDF 报告*配方

# 使用模板生成报告

HTML 是一种非常灵活的格式，可用于呈现丰富的报告。虽然可以将 HTML 模板视为纯文本创建，但也有工具可以让您更好地处理结构化文本。这也将模板与代码分离，将数据的生成与数据的表示分开。

# 准备工作

此配方中使用的工具 Jinja2 读取包含模板的文件，并将上下文应用于它。上下文包含要显示的数据。

我们应该从安装模块开始：

```py
$ echo "jinja2==2.20" >> requirements.txt
$ pip install -r requirements.txt
```

Jinja2 使用自己的语法，这是 HTML 和 Python 的混合体。它旨在 HTML 文档，因此可以轻松执行操作，例如正确转义特殊字符。

在 GitHub 存储库中，我们已经包含了一个名为`jinja_template.html`的模板文件。

# 如何做...

1.  导入 Jinja2 `Template`和`datetime`：

```py
>>> from jinja2 import Template
>>> from datetime import datetime
```

1.  从文件中读取模板到内存中：

```py
>>> with open('jinja_template.html') as file:
...     template = Template(file.read())
```

1.  创建一个包含要显示数据的上下文：

```py
>>> context = {
    'date': datetime.now(),
    'movies': ['Casablanca', 'The Sound of Music', 'Vertigo'],
    'total_minutes': 404,
}
```

1.  渲染模板并写入一个新文件`report.html`，结果如下：

```py
>>> with open('report.html', 'w') as file:
...    file.write(template.render(context))
```

1.  在浏览器中打开`report.html`文件：

![](img/47421547-e6ed-41d2-8573-cf9ffebbc7d2.png)

# 它是如何工作的...

*如何做...*部分中的步骤 2 和 4 非常简单：它们读取模板并保存生成的报告。

如步骤 3 和 4 所示，主要任务是创建一个包含要显示信息的上下文字典。然后模板呈现该信息，如步骤 5 所示。让我们来看看`jinja_template.html`：

```py
<!DOCTYPE html>
<html lang="en">
<head>
    <title> Movies Report</title>
</head>
<body>
    <h1>Movies Report</h1>
    <p>Date {{date}}</p>
    <p>Movies seen in the last 30 days: {{movies|length}}</p>
    <ol>
        {% for movie in movies %}
        <li>{{movie}}</li>
        {% endfor %}
    </ol>
    <p>Total minutes: {{total_minutes}} </p>
</body>
</html>
```

大部分是替换上下文值，如`{{total_minutes}}`在花括号之间定义。

注意标签`{% for ... %} / {% endfor %}`，它定义了一个循环。这允许基于 Python 的赋值生成多行或元素。

可以对变量应用过滤器进行修改。在这种情况下，将`length`过滤器应用于`movies`列表，以使用管道符号获得大小，如`{{movies|length}}`所示。

# 还有更多...

除了`{% for %}`标签之外，还有一个`{% if %}`标签，允许它有条件地显示：

```py
{% if movies|length > 5 %}
  Wow, so many movies this month!
{% else %}
  Regular number of movies
{% endif %}
```

已经定义了许多过滤器（在此处查看完整列表：[`jinja.pocoo.org/docs/2.10/templates/#list-of-builtin-filters`](http://jinja.pocoo.org/docs/2.10/templates/#list-of-builtin-filters)）。但也可以定义自定义过滤器。

请注意，您可以使用过滤器向模板添加大量处理和逻辑。虽然少量是可以的，但请尝试限制模板中的逻辑量。大部分用于显示数据的计算应该在之前完成，使上下文非常简单，并简化模板，从而允许进行更改。

处理 HTML 文件时，最好自动转义变量。这意味着具有特殊含义的字符，例如`<`字符，将被替换为等效的 HTML 代码，以便在 HTML 页面上正确显示。为此，使用`autoescape`参数创建模板。在这里检查差异：

```py
>>> Template('{{variable}}', autoescape=False).render({'variable': '<'})
'<'
>>> Template('{{variable}}', autoescape=True).render({'variable': '<'})
'<'
```

可以对每个变量应用转义，使用`e`过滤器（表示*转义*），并使用`safe`过滤器取消应用（表示*可以安全地渲染*）。

Jinja2 模板是可扩展的，这意味着可以创建一个`base_template.html`，然后扩展它，更改一些元素。还可以包含其他文件，对不同部分进行分区和分离。有关更多详细信息，请参阅完整文档。

Jinja2 非常强大，可以让我们创建复杂的 HTML 模板，还可以在其他格式（如 LaTeX 或 JavaScript）中使用，尽管这需要配置。我鼓励您阅读整个文档，并查看其所有功能！

完整的 Jinja2 文档可以在这里找到：[`jinja.pocoo.org/docs/2.10/.`](http://jinja.pocoo.org/docs/2.10/)

# 另请参阅

+   *在纯文本中创建简单报告*配方

+   *在 Markdown 中格式化文本*配方

# 在 Markdown 中格式化文本

**Markdown**是一种非常流行的标记语言，用于创建可以转换为样式化 HTML 的原始文本。这是一种良好的方式，可以以原始文本格式对文档进行结构化，同时能够在 HTML 中正确地对其进行样式设置。

在这个配方中，我们将看到如何使用 Python 将 Markdown 文档转换为样式化的 HTML。

# 准备工作

我们应该首先安装`mistune`模块，它将 Markdown 文档编译为 HTML：

```py
$ echo "mistune==0.8.3" >> requirements.txt
$ pip install -r requirements.txt
```

在 GitHub 存储库中，有一个名为`markdown_template.md`的模板文件，其中包含要生成的报告的模板。

# 如何做到这一点...

1.  导入`mistune`和`datetime`：

```py
>>> import mistune
```

1.  从文件中读取模板：

```py
>>> with open('markdown_template.md') as file:
...     template = file.read()
```

1.  设置要包含在报告中的数据的上下文：

```py
context = {
    'date': datetime.now(),
    'pmovies': ['Casablanca', 'The Sound of Music', 'Vertigo'],
    'total_minutes': 404,
}
```

1.  由于电影需要显示为项目符号，我们将列表转换为适当的 Markdown 项目符号列表。同时，我们存储了电影的数量：

```py
>>> context['num_movies'] = len(context['pmovies'])
>>> context['movies'] = '\n'.join('* {}'.format(movie) for movie in context['pmovies'])
```

1.  渲染模板并将生成的 Markdown 编译为 HTML：

```py
>>> md_report = template.format(**context)
>>> report = mistune.markdown(md_report)
```

1.  最后，将生成的报告存储在`report.html`文件中：

```py
>>> with open('report.html', 'w') as file:
...    file.write(report)
```

1.  在浏览器中打开`report.html`文件以检查结果：

![](img/3d49c70a-b883-4122-a5ac-36db8f95bfd7.png)

# 它是如何工作的...

*如何做...*部分的第 2 步和第 3 步准备模板和要显示的数据。在第 4 步中，产生了额外的信息——电影的数量，这是从`movies`元素派生出来的。然后，将`movies`元素从 Python 列表转换为有效的 Markdown 元素。注意新行和初始的`*`，它将被呈现为一个项目符号：

```py
>>> '\n'.join('* {}'.format(movie) for movie in context['pmovies'])
'* Casablanca\n* The Sound of Music\n* Vertigo'
```

在第 5 步中，模板以 Markdown 格式生成。这种原始形式非常易读，这是 Markdown 的优点：

```py
Movies Report
=======

Date: 2018-06-29 20:47:18.930655

Movies seen in the last 30 days: 3

* Casablanca
* The Sound of Music
* Vertigo

Total minutes: 404
```

然后，使用`mistune`，报告被转换为 HTML 并在第 6 步中存储在文件中。

# 还有更多...

学习 Markdown 非常有用，因为它被许多常见的网页支持，可以作为一种启用文本输入并能够呈现为样式化格式的方式。一些例子是 GitHub，Stack Overflow 和大多数博客平台。

实际上，Markdown 不止一种。这是因为官方定义有限或模糊，并且没有兴趣澄清或标准化它。这导致了几种略有不同的实现，如 GitHub Flavoured Markdown，MultiMarkdown 和 CommonMark。

Markdown 中的文本非常易读，但如果您需要交互式地查看它的外观，可以使用 Dillinger 在线编辑器在[`dillinger.io/`](https://dillinger.io/)上使用。

`Mistune`的完整文档在这里可用：[`mistune.readthedocs.io/en/latest/.`](http://mistune.readthedocs.io/en/latest/)

完整的 Markdown 语法可以在[`daringfireball.net/projects/markdown/syntax`](https://daringfireball.net/projects/markdown/syntax)找到，并且有一个包含最常用元素的好的速查表在[`beegit.com/markdown-cheat-sheet.`](https://beegit.com/markdown-cheat-sheet)上。

# 另请参阅

+   *在疼痛文本中创建简单报告*食谱

+   *使用报告模板*食谱

# 撰写基本 Word 文档

Microsoft Office 是最常见的软件之一，尤其是 MS Word 几乎成为了文档的事实标准。使用自动化脚本可以生成`docx`文档，这将有助于以一种易于阅读的格式分发报告。

在这个食谱中，我们将学习如何生成一个完整的 Word 文档。

# 准备工作

我们将使用`python-docx`模块处理 Word 文档：

```py
>>> echo "python-docx==0.8.6" >> requirements.txt
>>> pip install -r requirements.txt
```

# 如何做到这一点...

1.  导入`python-docx`和`datetime`：

```py
>>> import docx
>>> from datetime import datetime
```

1.  定义要存储在报告中的数据的`context`：

```py
context = {
    'date': datetime.now(),
    'movies': ['Casablanca', 'The Sound of Music', 'Vertigo'],
    'total_minutes': 404,
}
```

1.  创建一个新的`docx`文档，并包括一个标题，`电影报告`：

```py
>>> document = docx.Document()
>>> document.add_heading('Movies Report', 0)
```

1.  添加一个描述日期的段落，并在其中使用斜体显示日期：

```py
>>> paragraph = document.add_paragraph('Date: ')
>>> paragraph.add_run(str(context['date'])).italic = True
```

1.  添加有关已观看电影数量的信息到不同的段落中：

```py
>>> paragraph = document.add_paragraph('Movies see in the last 30 days: ')
>>> paragraph.add_run(str(len(context['movies']))).italic = True
```

1.  将每部电影添加为一个项目符号：

```py
>>> for movie in context['movies']:
...     document.add_paragraph(movie, style='List Bullet')
```

1.  添加总分钟数并将文件保存如下：

```py
>>> paragraph = document.add_paragraph('Total minutes: ')
>>> paragraph.add_run(str(context['total_minutes'])).italic = True
>>> document.save('word-report.docx')
```

1.  打开`word-report.docx`文件进行检查：

![](img/c0e40215-1fb8-45e6-82a6-dadd4fbe344d.png)

# 它是如何工作的...

Word 文档的基础是它被分成段落，每个段落又被分成运行。运行是一个段落的一部分，它共享相同的样式。

*如何做...*部分的第 1 步和第 2 步是导入和定义要存储在报告中的数据的准备工作。

在第 3 步中，创建了文档并添加了一个具有适当标题的标题。这会自动为文本设置样式。

处理段落是在第 4 步中介绍的。基于引入的文本创建了一个新段落，默认样式，但可以添加新的运行来更改它。在这里，我们添加了第一个带有文本“日期：”的运行，然后添加了另一个带有特定时间并标记为*斜体*的运行。

在第 5 步和第 6 步中，我们看到了有关电影的信息。第一部分以与第 4 步类似的方式存储了电影的数量。之后，电影逐个添加到报告中，并设置为项目符号的样式。

最后，第 7 步以与第 4 步类似的方式存储了所有电影的总运行时间，并将文档存储在文件中。

# 还有更多...

如果需要在文档中引入额外的行以进行格式设置，请添加空段落。

由于 MS Word 格式的工作方式，很难确定将有多少页。您可能需要对大小进行一些测试，特别是如果您正在动态生成文本。

即使生成了`docx`文件，也不需要安装 MS Office。还有其他应用程序可以打开和处理这些文件，包括免费的替代品，如 LibreOffice。

整个`python-docx`文档可以在这里找到：[`python-docx.readthedocs.io/en/latest/.`](https://python-docx.readthedocs.io/en/latest/)

# 另请参阅

+   *为 Word 文档设置样式*的方法

+   *在 Word 文档中生成结构*的方法

# 为 Word 文档设置样式

Word 文档可能非常简单，但我们也可以添加样式以帮助正确理解显示的数据。Word 具有一组预定义的样式，可用于变化文档并突出显示其中的重要部分。

# 准备工作

我们将使用`python-docx`模块处理 Word 文档：

```py
>>> echo "python-docx==0.8.6" >> requirements.txt
>>> pip install -r requirements.txt
```

# 如何操作...

1.  导入`python-docx`模块：

```py
>>> import docx
```

1.  创建一个新文档：

```py
>>> document = docx.Document()
```

1.  添加一个突出显示某些单词的段落，*斜体*，**粗体**和下划线：

```py
>>> p = document.add_paragraph('This shows different kinds of emphasis: ')
>>> p.add_run('bold').bold = True
>>> p.add_run(', ')
<docx.text.run.Run object at ...>
>>> p.add_run('italics').italic = True
>>> p.add_run(' and ')
<docx.text.run.Run object at ...>
>>> p.add_run('underline').underline = True
>>> p.add_run('.')
<docx.text.run.Run object at ...>
```

1.  创建一些段落，使用默认样式进行样式设置，如`List Bullet`、`List Number`或`Quote`：

```py
>>> document.add_paragraph('a few', style='List Bullet')
<docx.text.paragraph.Paragraph object at ...>
>>> document.add_paragraph('bullet', style='List Bullet')
<docx.text.paragraph.Paragraph object at ...>
>>> document.add_paragraph('points', style='List Bullet')
<docx.text.paragraph.Paragraph object at ...>
>>>
>>> document.add_paragraph('Or numbered', style='List Number')
<docx.text.paragraph.Paragraph object at ...>
>>> document.add_paragraph('that will', style='List Number')
<docx.text.paragraph.Paragraph object at ...>
>>> document.add_paragraph('that keep', style='List Number')
<docx.text.paragraph.Paragraph object at ...>
>>> document.add_paragraph('count', style='List Number')
<docx.text.paragraph.Paragraph object at ...>
>>> 
>>> document.add_paragraph('And finish with a quote', style='Quote')
<docx.text.paragraph.Paragraph object at 0x10d2336d8>
```

1.  创建一个不同字体和大小的段落。我们将使用`Arial`字体和`25`号字体大小。段落将右对齐：

```py
>>> from docx.shared import Pt
>>> from docx.enum.text import WD_ALIGN_PARAGRAPH
>>> p = document.add_paragraph('This paragraph will have a manual styling and right alignment')
>>> p.runs[0].font.name = 'Arial'
>>> p.runs[0].font.size = Pt(25)
>>> p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
```

1.  保存文档：

```py
>>> document.save('word-report-style.docx')
```

1.  打开`word-report-style.docx`文档以验证其内容：

![](img/eaaee1e8-82db-4952-bab9-1173dac10bdb.png)

# 它是如何工作的...

在第 1 步创建文档后，*如何操作...*部分的第 2 步添加了一个具有多个运行的段落。在 Word 中，一个段落可以包含多个运行，这些运行是可以具有不同样式的部分。一般来说，任何与单词相关的格式更改都将应用于运行，而影响段落的更改将应用于段落。

默认情况下，每个运行都使用`Normal`样式创建。任何`.bold`、`.italic`或`.underline`的属性都可以更改为`True`，以设置运行是否应以适当的样式或组合显示。值为`False`将停用它，而`None`值将保留为默认值。

请注意，此协议中的正确单词是*italic*，而不是*italics*。将属性设置为 italics 不会产生任何效果，但也不会显示错误。

第 4 步显示了如何应用一些默认样式以显示项目符号、编号列表和引用。还有更多样式，可以在文档的此页面中进行检查：[`python-docx.readthedocs.io/en/latest/user/styles-understanding.html?highlight=List%20Bullet#paragraph-styles-in-default-template`](https://python-docx.readthedocs.io/en/latest/user/styles-understanding.html?highlight=List%20Bullet#paragraph-styles-in-default-template)。尝试找出哪些样式最适合您的文档。

运行的`.font`属性显示在第 5 步中。这允许您手动设置特定的字体和大小。请注意，需要使用适当的`Pt`（点）对象来指定大小。

段落的对齐是在`paragraph`对象中设置的，并使用常量来定义它是左对齐、右对齐、居中还是两端对齐。所有对齐选项都可以在这里找到：[`python-docx.readthedocs.io/en/latest/api/enum/WdAlignParagraph.html.`](https://python-docx.readthedocs.io/en/latest/api/enum/WdAlignParagraph.html)

最后，第 7 步保存文件，使其存储在文件系统中。

# 还有更多...

`font`属性也可以用来设置文本的更多属性，比如小型大写字母、阴影、浮雕或删除线。所有可能性的范围都在这里显示：[`python-docx.readthedocs.io/en/latest/api/text.html#docx.text.run.Font.`](https://python-docx.readthedocs.io/en/latest/api/text.html#docx.text.run.Font)

另一个可用的选项是更改文本的颜色。注意，运行可以是先前生成的运行之一：

```py
>>> from docx.shared import RGBColor
>>> DARK_BLUE = RGBColor.from_string('1b3866')
>>> run.font.color.rbg = DARK_BLUE
```

颜色可以用字符串的常规十六进制格式描述。尝试定义要使用的所有颜色，以确保它们都是一致的，并且在报告中最多使用三种颜色，以免过多。

您可以使用在线颜色选择器，比如这个：[`www.w3schools.com/colors/colors_picker.asp`](https://www.w3schools.com/colors/colors_picker.asp)。记住不要在开头使用#。如果需要生成调色板，最好使用工具，比如[`coolors.co/`](https://coolors.co/)来生成好的组合。

整个`python-docx`文档在这里可用：[`python-docx.readthedocs.io/en/latest/.`](https://python-docx.readthedocs.io/en/latest/)

# 另请参阅

+   *编写基本的 Word 文档*配方

+   *在 Word 文档中生成结构*配方

# 在 Word 文档中生成结构

为了创建适当的专业报告，它们需要有适当的结构。MS Word 文档没有“页面”的概念，因为它是按段落工作的，但我们可以引入分页和部分来正确地划分文档。

在本配方中，我们将看到如何创建结构化的 Word 文档。

# 准备工作

我们将使用`python-docx`模块来处理 Word 文档：

```py
>>> echo "python-docx==0.8.6" >> requirements.txt
>>> pip install -r requirements.txt
```

# 如何做...

1.  导入`python-docx`模块：

```py
>>> import docx
```

1.  创建一个新文档：

```py
>>> document = docx.Document()
```

1.  创建一个有换行的段落：

```py
>>> p = document.add_paragraph('This is the start of the paragraph')
>>> run = p.add_run()
>>> run.add_break(docx.text.run.WD_BREAK.LINE)
>>> p.add_run('And now this in a different line')
>>> p.add_run(". Even if it's on the same paragraph.")
```

1.  创建一个分页并写一个段落：

```py
>>> document.add_page_break()
>>> document.add_paragraph('This appears in a new page')
```

1.  创建一个新的部分，将位于横向页面上：

```py
>>> section = document.add_section( docx.enum.section.WD_SECTION.NEW_PAGE)
>>> section.orientation = docx.enum.section.WD_ORIENT.LANDSCAPE
>>> section.page_height, section.page_width = section.page_width, section.page_height
>>> document.add_paragraph('This is part of a new landscape section')
```

1.  创建另一个部分，恢复为纵向方向：

```py
>>> section = document.add_section( docx.enum.section.WD_SECTION.NEW_PAGE)
>>> section.orientation = docx.enum.section.WD_ORIENT.PORTRAIT
>>> section.page_height, section.page_width = section.page_width, section.page_height
>>> document.add_paragraph('In this section, recover the portrait orientation')
```

1.  保存文档：

```py
>>> document.save('word-report-structure.docx')
```

1.  检查结果，打开文档并检查生成的部分：

![](img/1f465fba-8e0c-4ddf-be88-cf9313d3907d.png)

检查新页面：

![](img/ef161196-263d-4e4e-9345-b69e423633c5.png)

检查横向部分：

![](img/b1dcaa09-74bf-4b6d-9f0d-389d64df1542.png)

然后，返回到纵向方向：

![](img/7cc9f9f9-18ac-4b18-a61d-decef04d16df.png)

# 它是如何工作的...

在*如何做...*部分的第 2 步中创建文档后，我们为第一部分添加了一个段落。请注意，文档以一个部分开始。段落在段落中间引入了一个换行。

段落中的换行和新段落之间有一点差异，尽管对于大多数用途来说它们是相似的。尝试对它们进行实验。

第 3 步引入了分页符，但未更改部分。

第 4 步在新页面上创建一个新的部分。第 5 步还将页面方向更改为横向。在第 6 步，引入了一个新的部分，并且方向恢复为纵向。

请注意，当更改方向时，我们还需要交换宽度和高度。每个新部分都继承自上一个部分的属性，因此这种交换也需要在第 6 步中发生。

最后，在第 6 步保存文档。

# 还有更多...

一个部分规定了页面构成，包括页面的方向和大小。可以使用长度选项（如`Inches`或`Cm`）来更改页面的大小：

```py
>>> from docx.shared import Inches, Cm 
>>> section.page_height = Inches(10)
>>> section.page_width = Cm(20)
```

页面边距也可以用同样的方式定义：

```py
>>> section.left_margin = Inches(1.5) >>> section.right_margin = Cm(2.81) >>> section.top_margin = Inches(1) >>> section.bottom_margin = Cm(2.54)
```

还可以强制节在下一页开始，而不仅仅是在下一页开始，这在双面打印时看起来更好：

```py
>>> document.add_section( docx.enum.section.WD_SECTION.ODD_PAGE)
```

整个`python-docx`文档在这里可用：[`python-docx.readthedocs.io/en/latest/.`](https://python-docx.readthedocs.io/en/latest/)

# 另请参阅

+   *编写基本 Word 文档*配方

+   *对 Word 文档进行样式设置*配方

# 向 Word 文档添加图片

Word 文档能够添加图像以显示图表或任何其他类型的额外信息。能够添加图像是创建丰富报告的好方法。

在这个配方中，我们将看到如何在 Word 文档中包含现有文件。

# 准备工作

我们将使用`python-docx`模块来处理 Word 文档：

```py
$ echo "python-docx==0.8.6" >> requirements.txt
$ pip install -r requirements.txt
```

我们需要准备一个要包含在文档中的图像。我们将使用 GitHub 上的文件[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter04/images/photo-dublin-a1.jpg`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter04/images/photo-dublin-a1.jpg)，显示了都柏林的景色。您可以通过命令行下载它，就像这样：

```py
$ wget https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter04/images/photo-dublin-a1.jpg
```

# 如何做...

1.  导入`python-docx`模块：

```py
>>> import docx
```

1.  创建一个新文档：

```py
>>> document = docx.Document()
```

1.  创建一个带有一些文本的段落：

```py
>>> document.add_paragraph('This is a document that includes a picture taken in Dublin')
```

1.  添加图像：

```py
>>> image = document.add_picture('photo-dublin-a1.jpg')
```

1.  适当地缩放图像以适合页面（*14 x 10*）：

```py
>>> from docx.shared import Cm
>>> image.width = Cm(14)
>>> image.height = Cm(10)
```

1.  图像已添加到新段落。将其居中并添加描述性文本：

```py
>>> paragraph = document.paragraphs[-1]
>>> from docx.enum.text import WD_ALIGN_PARAGRAPH
>>> paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
>>> paragraph.add_run().add_break()
>>> paragraph.add_run('A picture of Dublin')
```

1.  添加一个带有额外文本的新段落，并保存文档：

```py
>>> document.add_paragraph('Keep adding text after the image')
<docx.text.paragraph.Paragraph object at XXX>
>>> document.save('report.docx')
```

1.  检查结果：

![](img/b2f48705-8e71-4bf7-b683-1c79f1420510.png)

# 它是如何工作的...

前几个步骤（*如何做...*部分的第 1 步到第 3 步）创建文档并添加一些文本。

第 4 步从文件中添加图像，第 5 步将其调整为可管理的大小。默认情况下，图像太大了。

调整图像大小时请注意图像的比例。请注意，您还可以使用其他度量单位，如`Inch`，也在`shared`中定义。

插入图像也会创建一个新段落，因此可以对段落进行样式设置，以使图像对齐或添加更多文本，例如参考或描述。通过`document.paragraph`属性在第 6 步获得段落。最后一个段落被获得并适当地样式化，使其居中。添加了一个新行和一个带有描述性文本的`run`。

第 7 步在图像后添加额外文本并保存文档。

# 还有更多...

图像的大小可以更改，但是如前所述，如果更改了图像的比例，需要计算图像的比例。如果通过近似值进行调整，调整大小可能不会完美，就像*如何做...*部分的第 5 步一样。

请注意，图像的比例不是完美的 10:14。它应该是 10:13.33。对于图像来说，这可能足够好，但对于更敏感于比例变化的数据，如图表，可能需要额外的注意。

为了获得适当的比例，将高度除以宽度，然后进行适当的缩放：

```py
>>> image = document.add_picture('photo-dublin-a1.jpg')
>>> image.height / image.width
0.75
>>> RELATION = image.height / image.width
>>> image.width = Cm(12)
>>> image.height = Cm(12 * RELATION)
```

如果需要将值转换为特定大小，可以使用`cm`、`inches`、`mm`或`pt`属性：

```py
>>> image.width.cm
12.0
>>> image.width.mm
120.0
>>> image.width.inches
4.724409448818897
>>> image.width.pt
340.15748031496065
```

整个`python-docx`文档在这里可用：[`python-docx.readthedocs.io/en/latest/.`](https://python-docx.readthedocs.io/en/latest/)

# 另请参阅

+   *编写基本 Word 文档*配方

+   *对 Word 文档进行样式设置*配方

+   *在 Word 文档中生成结构*配方

# 编写简单的 PDF 文档

PDF 文件是共享报告的常用方式。PDF 文档的主要特点是它们确切地定义了文档的外观，并且在生成后是只读的，这使得它们非常容易共享。

在这个配方中，我们将看到如何使用 Python 编写一个简单的 PDF 报告。

# 准备工作

我们将使用`fpdf`模块来创建 PDF 文档：

```py
>>> echo "fpdf==1.7.2" >> requirements.txt
>>> pip install -r requirements.txt
```

# 如何做...

1.  导入`fpdf`模块：

```py
>>> import fpdf
```

1.  创建文档：

```py
>>> document = fpdf.FPDF()
```

1.  为标题定义字体和颜色，并添加第一页：

```py
>>> document.set_font('Times', 'B', 14)
>>> document.set_text_color(19, 83, 173)
>>> document.add_page()
```

1.  写文档的标题：

```py
>>> document.cell(0, 5, 'PDF test document')
>>> document.ln()
```

1.  写一个长段落：

```py
>>> document.set_font('Times', '', 12)
>>> document.set_text_color(0)
>>> document.multi_cell(0, 5, 'This is an example of a long paragraph. ' * 10)
[]
>>> document.ln()
```

1.  写另一个长段落：

```py
>>> document.multi_cell(0, 5, 'Another long paragraph. Lorem ipsum dolor sit amet, consectetur adipiscing elit.' * 20) 
```

1.  保存文档：

```py
>>> document.output('report.pdf')
```

1.  检查`report.pdf`文档：

![](img/9eb543d1-256a-4293-ae39-8b7cbacaa201.png)

# 它是如何工作的...

`fpdf`模块创建 PDF 文档并允许我们在其中写入。

由于 PDF 的特殊性，最好的思考方式是想象一个光标在文档中写字并移动到下一个位置，类似于打字机。

首先要做的操作是指定要使用的字体和大小，然后添加第一页。这是在步骤 3 中完成的。第一个字体是粗体（第二个参数为`'B'`），比文档的其余部分大，用作标题。颜色也使用`.set_text_color`设置为 RGB 组件。

文本也可以使用`I`斜体和`U`下划线。您可以将它们组合，因此`BI`将产生粗体和斜体的文本。

`.cell`调用创建具有指定文本的文本框。前面的几个参数是宽度和高度。宽度`0`使用整个空间直到右边距。高度`5`（mm）适用于大小`12`字体。对`.ln`的调用引入了一个新行。

要写多行段落，我们使用`.multi_cell`方法。它的参数与`.cell`相同。在步骤 5 和 6 中写入两个段落。请注意在报告的标题和正文之间的字体变化。`.set_text_color`使用单个参数调用以设置灰度颜色。在这种情况下，它是黑色。

对于长文本使用`.cell`会超出边距并超出页面。仅用于适合单行的文本。您可以使用`.get_string_width`找到字符串的大小。

在步骤 7 中将文档保存到磁盘。

# 还有更多...

如果`multi_cell`操作占据页面上的所有可用空间，则页面将自动添加。调用`.add_page`将移动到新页面。

您可以使用任何默认字体（`Courier`、`Helvetica`和`Times`），或使用`.add_font`添加额外的字体。查看更多详细信息，请参阅文档：[`pyfpdf.readthedocs.io/en/latest/reference/add_font/index.html.`](http://pyfpdf.readthedocs.io/en/latest/reference/add_font/index.html)

字体`Symbol`和`ZapfDingbats`也可用，但用于符号。如果您需要一些额外的符号，这可能很有用，但在使用之前进行测试。其余默认字体应包括您对衬线、无衬线和等宽情况的需求。在 PDF 中，使用的字体将嵌入文档中，因此它们将正确显示。

保持整个文档中的高度一致，至少在相同大小的文本之间。定义一个您满意的常数，并在整个文本中使用它：

```py
>>> BODY_TEXT_HEIGHT = 5
>>> document.multi_cell(0, BODY_TEXT_HEIGHT, text)
```

默认情况下，文本将被调整对齐，但可以更改。使用`J`（调整对齐）、`C`（居中）、`R`（右对齐）或`L`（左对齐）的对齐参数。例如，这将产生左对齐的文本：

```py
>>> document.multi_cell(0, BODY_TEXT_HEIGHT, text, align='L')
```

完整的 FPDF 文档可以在这里找到：[`pyfpdf.readthedocs.io/en/latest/index.html.`](http://pyfpdf.readthedocs.io/en/latest/index.html)

# 另请参阅

+   *构建 PDF*

+   *汇总 PDF 报告*

+   *给 PDF 加水印和加密*

# 构建 PDF

在创建 PDF 时，某些元素可以自动生成，以使您的元素看起来更好并具有更好的结构。在本教程中，我们将看到如何添加页眉和页脚，以及如何创建到其他元素的链接。

# 准备工作

我们将使用`fpdf`模块创建 PDF 文档：

```py
>>> echo "fpdf==1.7.2" >> requirements.txt
>>> pip install -r requirements.txt
```

# 操作步骤...

1.  `structuring_pdf.py`脚本在 GitHub 上可用：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter05/structuring_pdf.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter05/structuring_pdf.py)。最相关的部分显示如下：

```py
import fpdf
from random import randint

class StructuredPDF(fpdf.FPDF):
    LINE_HEIGHT = 5

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        page_number = 'Page {number}/{{nb}}'.format(number=self.page_no())
        self.cell(0, self.LINE_HEIGHT, page_number, 0, 0, 'R')

    def chapter(self, title, paragraphs):
        self.add_page()
        link = self.title_text(title)
        page = self.page_no()
        for paragraph in paragraphs:
            self.multi_cell(0, self.LINE_HEIGHT, paragraph)
            self.ln()

        return link, page

    def title_text(self, title):
        self.set_font('Times', 'B', 15)
        self.cell(0, self.LINE_HEIGHT, title)
        self.set_font('Times', '', 12)
        self.line(10, 17, 110, 17)
        link = self.add_link()
        self.set_link(link)
        self.ln()
        self.ln()

        return link

    def get_full_line(self, head, tail, fill):
        ...
```

```py
    def toc(self, links):
        self.add_page()
        self.title_text('Table of contents')
        self.set_font('Times', 'I', 12)

        for title, page, link in links:
            line = self.get_full_line(title, page, '.')
            self.cell(0, self.LINE_HEIGHT, line, link=link)
            self.ln()

LOREM_IPSUM = ...

def main():
    document = StructuredPDF()
    document.alias_nb_pages()
    links = []
    num_chapters = randint(5, 40)
    for index in range(1, num_chapters):
        chapter_title = 'Chapter {}'.format(index)
        num_paragraphs = randint(10, 15)
        link, page = document.chapter(chapter_title,
                                      [LOREM_IPSUM] * num_paragraphs)
        links.append((chapter_title, page, link))

    document.toc(links)
    document.output('report.pdf')
```

1.  运行脚本，它将生成`report.pdf`文件，其中包含一些章节和目录。请注意，它会生成一些随机性，因此每次运行时具体数字会有所变化。

```py
$ python3 structuring_pdf.py
```

1.  检查结果。这是一个示例：

![](img/357ba66a-37c8-4d2a-b42b-6d0776f922b4.png)

在结尾处检查目录：

![](img/90d02b08-e982-4280-ae69-f447e5c78a80.png)

# 它是如何工作的...

让我们来看看脚本的每个元素。

`StructuredPDF`定义了一个从`FPDF`继承的类。这对于覆盖`footer`方法很有用，它在创建页面时每次创建一个页脚。它还有助于简化`main`中的代码。

`main`函数创建文档。它启动文档，并添加每个章节，收集它们的链接信息。最后，它调用`toc`方法使用链接信息生成目录。

要存储的文本是通过乘以 LOREM_IPSUM 文本生成的，这是一个占位符。

`chapter`方法首先打印标题部分，然后添加每个定义的段落。它收集章节开始的页码和`title_text`方法返回的链接以返回它们。

`title_text`方法以更大、更粗的文本编写文本。然后，它添加一行来分隔标题和章节的正文。它生成并设置一个指向以下行中当前页面的`link`对象：

```py
 link = self.add_link()
 self.set_link(link)
```

此链接将用于目录，以添加指向本章的可点击元素。

`footer`方法会自动向每个页面添加页脚。它设置一个较小的字体，并添加当前页面的文本（通过`page_no`获得），并使用`{nb}`，它将被替换为总页数。

在`main`中调用`alias_nb_pages`确保在生成文档时替换`{nb}`。

最后，在`toc`方法中生成目录。它写入标题，并添加所有已收集的引用链接作为链接、页码和章节名称，这是所有所需的信息。

# 还有更多...

注意使用`randint`为文档添加一些随机性。这个调用在 Python 的标准库中可用，返回一个在定义的最大值和最小值之间的数字。两者都包括在内。

`get_full_line`方法为目录生成适当大小的行。它需要一个开始（章节的名称）和结束（页码），并添加填充字符（点）的数量，直到行具有适当的宽度（120 毫米）。

为了计算文本的大小，脚本调用`get_string_width`，它考虑了字体和大小。

链接对象可用于指向特定页面，而不是当前页面，并且也不是页面的开头；使用`set_link(link, y=place, page=num_page)`。在[`pyfpdf.readthedocs.io/en/latest/reference/set_link/index.html`](http://pyfpdf.readthedocs.io/en/latest/reference/set_link/index.html)上查看文档。

调整一些元素可能需要一定程度的试错，例如，调整线的位置。稍微长一点或短一点的线可能是品味的问题。不要害怕尝试和检查，直到产生期望的效果。

完整的 FPDF 文档可以在这里找到：[`pyfpdf.readthedocs.io/en/latest/index.html.`](http://pyfpdf.readthedocs.io/en/latest/index.html)

# 另请参阅

+   *编写简单的 PDF 文档*食谱

+   *聚合 PDF 报告*食谱

+   *给 PDF 加水印和加密*食谱

# 聚合 PDF 报告

在这个食谱中，我们将看到如何将两个 PDF 合并成一个。这将允许我们将报告合并成一个更大的报告。

# 准备工作

我们将使用`PyPDF2`模块。`Pillow`和`pdf2image`也是脚本使用的依赖项：

```py
$ echo "PyPDF2==1.26.0" >> requirements.txt
$ echo "pdf2image==0.1.14" >> requirements.txt
$ echo "Pillow==5.1.0" >> requirements.txt
$ pip install -r requirements.txt
```

为了使`pdf2image`正常工作，需要安装`pdftoppm`，因此请在此处查看如何在不同平台上安装它的说明：[`github.com/Belval/pdf2image#first-you-need-pdftoppm.`](https://github.com/Belval/pdf2image#first-you-need-pdftoppm)

我们需要两个 PDF 文件来合并它们。对于这个示例，我们将使用两个 PDF 文件：一个是`structuring_pdf.py`脚本生成的`report.pdf`文件，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter05/structuring_pdf.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter05/structuring_pdf.py)，另一个是经过水印处理后的(`report2.pdf`)，命令如下：

```py
$ python watermarking_pdf.py report.pdf -u automate_user -o report2.pdf
```

使用加水印脚本`watermarking_pdf.py`，在 GitHub 上可用，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter05/watermarking_pdf.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter05/watermarking_pdf.py)。

# 如何操作...

1.  导入`PyPDF2`并创建输出 PDF：

```py
>>> import PyPDF2
>>> output_pdf = PyPDF2.PdfFileWriter()
```

1.  读取第一个文件并创建一个阅读器：

```py
>>> file1 = open('report.pdf', 'rb')
>>> pdf1 = PyPDF2.PdfFileReader(file1)
```

1.  将所有页面附加到输出 PDF：

```py
>>> output_pdf.appendPagesFromReader(pdf1)
```

1.  打开第二个文件，创建一个阅读器，并将页面附加到输出 PDF：

```py
>>> file2 = open('report2.pdf', 'rb')
>>> pdf2 = PyPDF2.PdfFileReader(file2)
>>> output_pdf.appendPagesFromReader(pdf2)
```

1.  创建输出文件并保存：

```py
>>> with open('result.pdf', 'wb') as out_file:
...     output_pdf.write(out_file)
```

1.  关闭打开的文件：

```py
>>> file1.close()
>>> file2.close()
```

1.  检查输出文件，并确认它包含两个 PDF 页面。

# 工作原理...

`PyPDF2`允许我们为每个输入文件创建一个阅读器，并将其所有页面添加到新创建的 PDF 写入器中。请注意，文件以二进制模式(`rb`)打开。

输入文件需要保持打开状态，直到保存结果。这是由于页面复制的方式。如果文件是打开的，则生成的文件可以存储为空文件。

PDF 写入器最终保存到一个新文件中。请注意，文件需要以二进制模式(`wb`)打开以进行写入。

# 还有更多...

`.appendPagesFromReader`非常方便，可以添加所有页面，但也可以使用`.addPage`逐个添加页面。例如，要添加第三页，代码如下：

```py
>>> page = pdf1.getPage(3)
>>> output_pdf.addPage(page)
```

`PyPDF2`的完整文档在这里：[`pythonhosted.org/PyPDF2/.`](https://pythonhosted.org/PyPDF2/)

# 另请参阅

+   *编写简单的 PDF 文档*示例

+   *结构化 PDF*示例

+   *加水印和加密 PDF*示例

# 加水印和加密 PDF

PDF 文件有一些有趣的安全措施，限制了文档的分发。我们可以加密内容，使其必须知道密码才能阅读。我们还将看到如何添加水印，以清楚地标记文档为不适合公开分发，并且如果泄漏，可以知道其来源。

# 准备工作

我们将使用`pdf2image`模块将 PDF 文档转换为 PIL 图像。`Pillow`是先决条件。我们还将使用`PyPDF2`：

```py
$ echo "pdf2image==0.1.14" >> requirements.txt
$ echo "Pillow==5.1.0" >> requirements.txt
$ echo "PyPDF2==1.26.0" >> requirements.txt
$ pip install -r requirements.txt
```

为了使`pdf2image`正常工作，需要安装`pdftoppm`，因此请在此处查看如何在不同平台上安装它的说明：[`github.com/Belval/pdf2image#first-you-need-pdftoppm.`](https://github.com/Belval/pdf2image#first-you-need-pdftoppm)

我们还需要一个 PDF 文件来加水印和加密。我们将使用 GitHub 上的`structuring_pdf.py`脚本生成的`report.pdf`文件，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/chapter5/structuring_pdf.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/chapter5/structuring_pdf.py)。

# 如何操作...

1.  `watermarking_pdf.py`脚本在 GitHub 上可用，网址为[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter05/watermarking_pdf.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter05/watermarking_pdf.py)。这里显示了最相关的部分：

```py
def encrypt(out_pdf, password):
    output_pdf = PyPDF2.PdfFileWriter()

    in_file = open(out_pdf, "rb")
    input_pdf = PyPDF2.PdfFileReader(in_file)
    output_pdf.appendPagesFromReader(input_pdf)
    output_pdf.encrypt(password)

    # Intermediate file
    with open(INTERMEDIATE_ENCRYPT_FILE, "wb") as out_file:
        output_pdf.write(out_file)

    in_file.close()

    # Rename the intermediate file
    os.rename(INTERMEDIATE_ENCRYPT_FILE, out_pdf)

def create_watermark(watermarked_by):
    mask = Image.new('L', WATERMARK_SIZE, 0)
    draw = ImageDraw.Draw(mask)
    font = ImageFont.load_default()
    text = 'WATERMARKED BY {}\n{}'.format(watermarked_by, datetime.now())
    draw.multiline_text((0, 100), text, 55, font=font)

    watermark = Image.new('RGB', WATERMARK_SIZE)
    watermark.putalpha(mask)
    watermark = watermark.resize((1950, 1950))
    watermark = watermark.rotate(45)
    # Crop to only the watermark
    bbox = watermark.getbbox()
    watermark = watermark.crop(bbox)

    return watermark

def apply_watermark(watermark, in_pdf, out_pdf):
    # Transform from PDF to images
    images = convert_from_path(in_pdf)
    ...
    # Paste the watermark in each page
    for image in images:
        image.paste(watermark, position, watermark)

    # Save the resulting PDF
    images[0].save(out_pdf, save_all=True, append_images=images[1:])
```

1.  使用以下命令给 PDF 文件加水印：

```py
$ python watermarking_pdf.py report.pdf -u automate_user -o out.pdf
Creating a watermark
Watermarking the document
$
```

1.  检查文档是否添加了`automate_user`水印和时间戳到`out.pdf`的所有页面：

![](img/62797425-1b6e-470f-b087-ec21197a66b1.png)

1.  使用以下命令加水印和加密。请注意，加密可能需要一些时间：

```py
$ python watermarking_pdf.py report.pdf -u automate_user -o out.pdf -p secretpassword
Creating a watermark
Watermarking the document
Encrypting the document
$
```

1.  打开生成的`out.pdf`文件，并检查是否需要输入`secretpassword`密码。时间戳也将是新的。

# 工作原理...

`watermarking_pdf.py`脚本首先使用`argparse`从命令行获取参数，然后将其传递给调用其他三个函数的`main`函数，`create_watermark`，`apply_watermark`和（如果使用密码）`encrypt`。

`create_watermark`生成带有水印的图像。它使用 Pillow 的`Image`类创建灰色图像（模式`L`）并绘制文本。然后，将此图像应用为新图像上的 Alpha 通道，使图像半透明，因此它将显示水印文本。

Alpha 通道使白色（颜色 0）完全透明，黑色（颜色 255）完全不透明。在这种情况下，背景是白色，文本的颜色是 55，使其半透明。

然后将图像旋转 45 度并裁剪以减少可能出现的透明背景。这将使图像居中并允许更好的定位。

在下一步中，`apply_watermark`使用`pdf2image`模块将 PDF 转换为 PIL`Images`序列。它计算应用水印的位置，然后粘贴水印。

图像需要通过其左上角定位。这位于文档的一半，减去水印的一半，高度和宽度都是如此。请注意，脚本假定文档的所有页面都是相等的。

最后，结果保存为 PDF；请注意`save_all`参数，它允许我们保存多页 PDF。

如果传递了密码，则调用`encrypt`函数。它使用`PdfFileReader`打开输出 PDF，并使用`PdfFileWriter`创建一个新的中间 PDF。将输出 PDF 的所有页面添加到新 PDF 中，对 PDF 进行加密，然后使用`os.rename`将中间 PDF 重命名为输出 PDF。

# 还有更多...

作为水印的一部分，请注意页面是从文本转换为图像的。这增加了额外的保护，因为文本不会直接可提取，因为它存储为图像。在保护文件时，这是一个好主意，因为它将阻止直接复制/粘贴。

这不是一个巨大的安全措施，因为文本可能可以通过 OCR 工具提取。但是，它可以防止对文本的轻松提取。

PIL 的默认字体可能有点粗糙。如果有`TrueType`或`OpenType`文件可用，可以通过调用以下内容添加并使用另一种字体：

```py
font = ImageFont.truetype('my_font.ttf', SIZE)
```

请注意，这可能需要安装`FreeType`库，通常作为`libfreetype`软件包的一部分提供。更多文档可在[`www.freetype.org/`](https://www.freetype.org/)找到。根据字体和大小，您可能需要调整大小。

完整的`pdf2image`文档可以在[`github.com/Belval/pdf2image`](https://github.com/Belval/pdf2image)找到，`PyPDF2`的完整文档在[`pythonhosted.org/PyPDF2/`](https://pythonhosted.org/PyPDF2/)，`Pillow`的完整文档可以在[`pillow.readthedocs.io/en/5.2.x/.`](https://pillow.readthedocs.io/en/5.2.x/)找到。

# 另请参阅

+   *编写简单的 PDF 文档*配方

+   *构建 PDF*配方

+   *聚合 PDF 报告*配方
