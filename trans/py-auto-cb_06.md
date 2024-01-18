# 与电子表格一起玩

在本章中，我们将涵盖以下食谱：

+   编写CSV电子表格

+   更新CSV电子表格

+   读取Excel电子表格

+   更新Excel电子表格

+   在Excel电子表格中创建新工作表

+   在Excel中创建图表

+   在Excel中处理格式

+   在LibreOffice中读写

+   在LibreOffice中创建宏

# 介绍

电子表格是计算机世界中最通用和无处不在的工具之一。它们直观的表格和单元格的方法被几乎每个使用计算机作为日常操作的人所使用。甚至有一个笑话说整个复杂的业务都是在一个电子表格中管理和描述的。它们是一种非常强大的工具。

这使得自动从电子表格中读取和写入变得非常强大。在本章中，我们将看到如何处理电子表格，主要是在最常见的格式Excel中。最后一个食谱将涵盖一个免费的替代方案，Libre Office，特别是如何在其中使用Python作为脚本语言。

# 编写CSV电子表格

CSV文件是简单的电子表格，易于共享。它们基本上是一个文本文件，其中包含用逗号分隔的表格数据（因此称为逗号分隔值），以简单的表格格式。CSV文件可以使用Python的标准库创建，并且可以被大多数电子表格软件读取。

# 准备工作

对于这个食谱，只需要Python的标准库。一切都已经准备就绪！

# 如何做到这一点...

1.  导入`csv`模块：

```py
>>> import csv
```

1.  定义标题以及数据的存储方式：

```py
>>> HEADER = ('Admissions', 'Name', 'Year')
>>> DATA = [
... (225.7, 'Gone With the Wind', 1939),
... (194.4, 'Star Wars', 1977),
... (161.0, 'ET: The Extra-Terrestrial', 1982)
... ]
```

1.  将数据写入CSV文件：

```py
>>> with open('movies.csv', 'w',  newline='') as csvfile:
...     movies = csv.writer(csvfile)
...     movies.writerow(HEADER)
...     for row in DATA:
...         movies.writerow(row)
```

1.  在电子表格中检查生成的CSV文件。在下面的屏幕截图中，使用LibreOffice软件显示文件：

![](assets/7608d599-692e-4267-93fd-e569ef57e858.png)

# 工作原理...

在*如何做*部分的步骤1和2中进行准备工作后，步骤3是执行工作的部分。

它以写（`w`）模式打开一个名为`movies.csv`的新文件。然后在`csvfile`中创建一个原始文件对象。所有这些都发生在`with`块中，因此在结束时关闭文件。

注意`newline=''`参数。这是为了让`writer`直接存储换行，并避免兼容性问题。

写入器使用`.writerow`逐行写入元素。第一个是`HEADER`，然后是每行数据。

# 还有更多...

所呈现的代码将数据存储在默认方言中。方言定义了每行数据之间的分隔符（逗号或其他字符），如何转义，换行等。如果需要调整方言，可以在`writer`调用中定义这些参数。请参见以下链接，了解可以定义的所有参数列表：

[https://docs.python.org/3/library/csv.html#dialects-and-formatting-parameters](https://docs.python.org/3/library/csv.html#dialects-and-formatting-parameters)。

CSV文件在简单时更好。如果要存储的数据很复杂，也许最好的选择不是CSV文件。但是在处理表格数据时，CSV文件非常有用。它们几乎可以被所有程序理解，甚至在低级别处理它们也很容易。

完整的`csv`模块文档可以在这里找到：

[https://docs.python.org/3/library/csv.html](https://docs.python.org/3/library/csv.html)。

# 另请参阅

+   *在*[第4章](e8536572-46e4-41ec-87b8-7f775fd61e63.xhtml)中的*读取和搜索本地文件*中的*读取CSV文件*食谱

+   *更新CSV文件*食谱

# 更新CSV文件

鉴于CSV文件是简单的文本文件，更新其内容的最佳解决方案是读取它们，将它们更改为内部Python对象，然后以相同的格式写入结果。在这个食谱中，我们将看到如何做到这一点。

# 准备工作

在这个配方中，我们将使用GitHub上的`movies.csv`文件。它包含以下数据：

| **招生** | **姓名** | **年份** |
| --- | --- | --- |
| 225.7 | 乱世佳人 | 1939年 |
| 194.4 | 星球大战 | 1968年 |
| 161.0 | 外星人 | 1982年 |

注意`星球大战`的年份是错误的（应为1977年）。我们将在配方中更改它。

# 如何做...

1.  导入`csv`模块并定义文件名：

```py
>>> import csv
>>> FILENAME = 'movies.csv'
```

1.  使用`DictReader`读取文件的内容，并将其转换为有序行的列表：

```py
>>> with open(FILENAME, newline='') as file:
...     data = [row for row in csv.DictReader(file)]
```

1.  检查获取的数据。将1968年的正确值更改为1977年：

```py
>>> data
[OrderedDict([('Admissions', '225.7'), ('Name', 'Gone With the Wind'), ('Year', '1939')]), OrderedDict([('Admissions', '194.4'), ('Name', 'Star Wars'), ('Year', '1968')]), OrderedDict([('Admissions', '161.0'), ('Name', 'ET: The Extra-Terrestrial'), ('Year', '1982')])]
>>> data[1]['Year']
'1968'
>>> data[1]['Year'] = '1977'
```

1.  再次打开文件，并存储值：

```py
>>> HEADER = data[0].keys()
>>> with open(FILENAME, 'w', newline='') as file:
...     writer = csv.DictWriter(file, fieldnames=HEADER)
...     writer.writeheader()
...     writer.writerows(data)
```

1.  在电子表格软件中检查结果。结果与*编写CSV电子表格*配方中的第4步中显示的结果类似。

# 工作原理...

在*如何做...*部分的第2步中导入`csv`模块后，我们从文件中提取所有数据。文件在`with`块中打开。`DictReader`方便地将其转换为字典列表，其中键是标题值。

然后可以操纵和更改方便格式化的数据。我们在第3步中将数据更改为适当的值。

在这个配方中，我们直接更改值，但在更一般的情况下可能需要搜索。

第4步将覆盖文件，并使用`DictWriter`存储数据。`DictWriter`要求我们通过`fieldnames`在列上定义字段。为了获得它，我们检索一行的键并将它们存储在`HEADER`中。

文件再次以`w`模式打开以覆盖它。`DictWriter`首先使用`.writeheader`存储标题，然后使用单个调用`.writerows`存储所有行。

也可以通过调用`.writerow`逐个添加行

关闭`with`块后，文件将被存储并可以进行检查。

# 还有更多...

CSV文件的方言通常是已知的，但也可能不是这种情况。在这种情况下，`Sniffer`类可以帮助。它分析文件的样本（或整个文件）并返回一个`dialect`对象，以允许以正确的方式进行读取：

```py
>>> with open(FILENAME, newline='') as file:
...    dialect = csv.Sniffer().sniff(file.read())
```

然后可以在打开文件时将方言传递给`DictReader`类。需要两次打开文件进行读取。

记得在`DictWriter`类上也使用方言以相同的格式保存文件。

`csv`模块的完整文档可以在这里找到：

[https://docs.python.org/3.6/library/csv.html](https://docs.python.org/3.6/library/csv.html)。

# 另请参阅

+   在[第4章](e8536572-46e4-41ec-87b8-7f775fd61e63.xhtml)的*读取CSV文件*配方中

+   *编写CSV电子表格*配方

# 读取Excel电子表格

MS Office可以说是最常见的办公套件软件，使其格式几乎成为标准。在电子表格方面，Excel可能是最常用的格式，也是最容易交换的格式。

在这个配方中，我们将看到如何使用`openpyxl`模块从Python中以编程方式获取Excel电子表格中的信息。

# 准备工作

我们将使用`openpyxl`模块。我们应该安装该模块，并将其添加到我们的`requirements.txt`文件中，如下所示：

```py
$ echo "openpyxl==2.5.4" >> requirements.txt
$ pip install -r requirements.txt
```

在GitHub存储库中，有一个名为`movies.xlsx`的Excel电子表格，其中包含前十部电影的出席信息。文件可以在此处找到：

[https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/movies.xlsx](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/movies.xlsx)。

信息来源是这个网页：

[http://www.mrob.com/pub/film-video/topadj.html](http://www.mrob.com/pub/film-video/topadj.html)。

# 如何做...

1.  导入`openpyxl`模块：

```py
>>> import openpyxl
```

1.  将文件加载到内存中：

```py
>>> xlsfile = openpyxl.load_workbook('movies.xlsx')
```

1.  列出所有工作表并获取第一个工作表，这是唯一包含数据的工作表：

```py
>>> xlsfile.sheetnames
['Sheet1']
>>> sheet = xlsfile['Sheet1']
```

1.  获取单元格`B4`和`D4`的值（入场和E.T.的导演）：

```py
>>> sheet['B4'].value
161
>>> sheet['D4'].value
'Steven Spielberg'
```

1.  获取行和列的大小。超出该范围的任何单元格将返回`None`作为值：

```py
>>> sheet.max_row
11
>>> sheet.max_column
4
>>> sheet['A12'].value
>>> sheet['E1'].value
```

# 它是如何工作的...

在第1步中导入模块后，*如何做…*部分的第2步将文件加载到`Workbook`对象的内存中。每个工作簿可以包含一个或多个包含单元格的工作表。

要确定可用的工作表，在第3步中，我们获取所有工作表（在此示例中只有一个），然后像字典一样访问工作表，以检索`Worksheet`对象。

然后，`Worksheet`可以通过它们的名称直接访问所有单元格，例如`A4`或`C3`。它们中的每一个都将返回一个`Cell`对象。`.value`属性存储单元格中的值。

在本章的其余配方中，我们将看到`Cell`对象的更多属性。继续阅读！

可以使用`max_columns`和`max_rows`获取存储数据的区域。这允许我们在数据的限制范围内进行搜索。

Excel将列定义为字母（A、B、C等），行定义为数字（1、2、3等）。记住始终先设置列，然后设置行（`D1`，而不是`1D`），否则将引发错误。

可以访问区域外的单元格，但不会返回数据。它们可以用于写入新信息。

# 还有更多...

也可以使用`sheet.cell(column, row)`检索单元格。这两个元素都从1开始。

从工作表中迭代数据区域内的所有单元格，例如：

```py
>>> for row in sheet:
...     for cell in row:
...         # Do stuff with cell
```

这将返回一个包含所有单元格的列表的列表，逐行：A1、A2、A3... B1、B2、B3等。

您可以通过`sheet.columns`迭代来检索单元格的列：A1、B1、C1等，A2、B2、C2等。

在检索单元格时，可以使用`.coordinate`、`.row`和`.column`找到它们的位置：

```py
>>> cell.coordinate
'D4'
>>> cell.column
'D'
>>> cell.row
4
```

完整的`openpyxl`文档可以在此处找到：

[https://openpyxl.readthedocs.io/en/stable/index.html](https://openpyxl.readthedocs.io/en/stable/index.html)。

# 另请参阅

+   *更新Excel电子表格*配方

+   *在Excel电子表格中创建新工作表*配方

+   *在Excel中创建图表*配方

+   *在Excel中处理格式*配方

# 更新Excel电子表格

在这个配方中，我们将看到如何更新现有的Excel电子表格。这将包括更改单元格中的原始值，还将设置在打开电子表格时将被评估的公式。我们还将看到如何向单元格添加注释。

# 准备就绪

我们将使用模块`openpyxl`。我们应该安装该模块，并将其添加到我们的`requirements.txt`文件中，如下所示：

```py
$ echo "openpyxl==2.5.4" >> requirements.txt
$ pip install -r requirements.txt
```

在GitHub存储库中，有一个名为`movies.xlsx`的Excel电子表格，其中包含前十部电影的观众人数信息。

文件可以在此处找到：

[https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/movies.xlsx](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/movies.xlsx)[.](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/chapter6/movies.xlsx)

# 如何做…

1.  导入模块`openpyxl`和`Comment`类：

```py
>>> import openpyxl
>>> from openpyxl.comments import Comment
```

1.  将文件加载到内存中并获取工作表：

```py
>>> xlsfile = openpyxl.load_workbook('movies.xlsx')
>>> sheet = xlsfile['Sheet1']
```

1.  获取单元格`D4`的值（E.T.的导演）：

```py
>>> sheet['D4'].value
'Steven Spielberg'
```

1.  将值更改为`Spielberg`：

```py
>>> sheet['D4'].value = 'Spielberg'
```

1.  向该单元格添加注释：

```py
>>> sheet['D4'].comment = Comment('Changed text automatically', 'User')
```

1.  添加一个新元素，获取`Admission`列中所有值的总和：

```py
>>> sheet['B12'] = '=SUM(B2:B11)'
```

1.  将电子表格保存到`movies_comment.xlsx`文件中：

```py
>>> xlsfile.save('movies_comment.xlsx')
```

1.  检查包含注释和在`A12`中计算`B`列总和的结果文件：

![](assets/fc401f36-2ee6-48e2-ba41-d2999eac7c60.png)

# 它是如何工作的...

在*如何做…*部分，第1步中的导入和第2步中的读取电子表格，我们在第3步中选择要更改的单元格。

在第4步中进行值的更新。在单元格中添加注释，覆盖`.coment`属性并添加新的`Comment`。请注意，还需要添加进行注释的用户。

值也可以包括公式的描述。在第6步，我们向单元格`B12`添加一个新的公式。在第8步打开文件时，该值将被计算并显示。

公式的值不会在Python对象中计算。这意味着公式可能包含错误，或者通过错误显示意外结果。请务必仔细检查公式是否正确。

最后，在第9步，通过调用文件的`.save`方法将电子表格保存到磁盘。

生成的文件名可以与输入文件相同，以覆盖该文件。

可以通过外部访问文件来检查注释和值。

# 还有更多...

您可以将数据存储在多个值中，并且它将被转换为Excel的适当类型。例如，存储`datetime`将以适当的日期格式存储。对于`float`或其他数字格式也是如此。

如果需要推断类型，可以在加载文件时使用`guess_type`参数来启用此功能，例如：

```py
>>> xlsfile = openpyxl.load_workbook('movies.xlsx', guess_types=True)
>>> xlsfile['Sheet1']['A1'].value = '37%'
>>> xlsfile['Sheet1']['A1'].value
0.37
>>> xlsfile['Sheet1']['A1'].value = '2.75'
>>> xlsfile['Sheet1']['A1'].value
2.75
```

向自动生成的单元格添加注释可以帮助审查结果文件，清楚地说明它们是如何生成的。

虽然可以添加公式来自动生成Excel文件，但调试结果可能会很棘手。在生成结果时，通常最好在Python中进行计算并将结果存储为原始数据。

完整的`openpyxl`文档可以在这里找到：

[https://openpyxl.readthedocs.io/en/stable/index.html](https://openpyxl.readthedocs.io/en/stable/index.html)。

# 另请参阅

+   *读取Excel电子表格*教程

+   *在Excel电子表格上创建新工作表*教程

+   *在Excel中创建图表*教程

+   *在Excel中处理格式*教程

# 在Excel电子表格上创建新工作表

在这个教程中，我们将演示如何从头开始创建一个新的Excel电子表格，并添加和处理多个工作表。

# 准备工作

我们将使用`openpyxl`模块。我们应该安装该模块，并将其添加到我们的`requirements.txt`文件中，如下所示：

```py
$ echo "openpyxl==2.5.4" >> requirements.txt
$ pip install -r requirements.txt
```

我们将在新文件中存储有关参与人数最多的电影的信息。数据从这里提取：

[http://www.mrob.com/pub/film-video/topadj.html](http://www.mrob.com/pub/film-video/topadj.html)。

# 如何做...

1.  导入`openpyxl`模块：

```py
>>> import openpyxl
```

1.  创建一个新的Excel文件。它创建了一个名为`Sheet`的默认工作表：

```py
>>> xlsfile = openpyxl.Workbook()
>>> xlsfile.sheetnames
['Sheet']
>>> sheet = xlsfile['Sheet']
```

1.  从源中向该工作表添加有关参与者人数的数据。为简单起见，只添加了前三个：

```py
>>> data = [
...    (225.7, 'Gone With the Wind', 'Victor Fleming'),
...    (194.4, 'Star Wars', 'George Lucas'),
...    (161.0, 'ET: The Extraterrestrial', 'Steven Spielberg'),
... ]
>>> for row, (admissions, name, director) in enumerate(data, 1):
...     sheet['A{}'.format(row)].value = admissions
...     sheet['B{}'.format(row)].value = name
```

1.  创建一个新的工作表：

```py
>>> sheet = xlsfile.create_sheet("Directors")
>>> sheet
<Worksheet "Directors">
>>> xlsfile.sheetnames
['Sheet', 'Directors']
```

1.  为每部电影添加导演的名称：

```py
>>> for row, (admissions, name, director) in enumerate(data, 1):
...    sheet['A{}'.format(row)].value = director
...    sheet['B{}'.format(row)].value = name
```

1.  将文件保存为`movie_sheets.xlsx`：

```py
>>> xlsfile.save('movie_sheets.xlsx')
```

1.  打开`movie_sheets.xlsx`文件，检查它是否有两个工作表，并且包含正确的信息，如下截图所示：

![](assets/ccd6f5d1-d03e-486c-a7ac-81df1c52b165.png)

# 它是如何工作的...

在*如何做…*部分，在第1步导入模块后，在第2步创建一个新的电子表格。这是一个只包含默认工作表的新电子表格。

要存储的数据在第3步中定义。请注意，它包含将放在两个工作表中的信息（两个工作表中都有名称，第一个工作表中有入场人数，第二个工作表中有导演的名称）。在这一步中，填充了第一个工作表。

请注意值是如何存储的。正确的单元格定义为列`A`或`B`和正确的行（行从1开始）。`enumerate`函数返回一个元组，第一个元素是索引，第二个元素是枚举参数（迭代器）。

之后，在第4步创建了新的工作表，使用名称`Directors`。`.create_sheet`返回新的工作表。

在第5步中存储了`Directors`工作表中的信息，并在第6步保存了文件。

# 还有更多...

可以通过`.title`属性更改现有工作表的名称：

```py
>>> sheet = xlsfile['Sheet']
>>> sheet.title = 'Admissions'
>>> xlsfile.sheetnames
['Admissions', 'Directors']
```

要小心，因为无法访问`xlsfile['Sheet']`工作表。那个名称不存在！

活动工作表，文件打开时将显示的工作表，可以通过`.active`属性获得，并且可以使用`._active_sheet_index`进行更改。索引从第一个工作表开始为`0`：

```py
>> xlsfile.active
<Worksheet "Admissions">
>>> xlsfile._active_sheet_index
0
>>> xlsfile._active_sheet_index = 1
>>> xlsfile.active
<Worksheet "Directors">
```

工作表也可以使用`.copy_worksheet`进行复制。请注意，某些数据，例如图表，不会被复制。大多数重复的信息将是单元格数据：

```py
new_copied_sheet = xlsfile.copy_worksheet(source_sheet)
```

完整的`openpyxl`文档可以在这里找到：

[https://openpyxl.readthedocs.io/en/stable/index.html](https://openpyxl.readthedocs.io/en/stable/index.html)。

# 另请参阅

+   读取Excel电子表格的方法

+   更新Excel电子表格并添加注释的方法

+   在Excel中创建图表

+   在Excel中使用格式的方法

# 在Excel中创建图表

电子表格包括许多处理数据的工具，包括以丰富多彩的图表呈现数据。让我们看看如何以编程方式将图表附加到Excel电子表格。

# 准备工作

我们将使用`openpyxl`模块。我们应该安装该模块，将其添加到我们的`requirements.txt`文件中，如下所示：

```py
$ echo "openpyxl==2.5.4" >> requirements.txt
$ pip install -r requirements.txt
```

我们将在新文件中存储有关观众人数最多的电影的信息。数据从这里提取：

[http://www.mrob.com/pub/film-video/topadj.html](http://www.mrob.com/pub/film-video/topadj.html)。

# 如何做...

1.  导入`openpyxl`模块并创建一个新的Excel文件：

```py
>>> import openpyxl
>>> from openpyxl.chart import BarChart, Reference
>>> xlsfile = openpyxl.Workbook()
```

1.  从源中在该工作表中添加有关观众人数的数据。为简单起见，只添加前三个：

```py
>>> data = [
...     ('Name', 'Admissions'),
...     ('Gone With the Wind', 225.7),
...     ('Star Wars', 194.4),
...     ('ET: The Extraterrestrial', 161.0),
... ]
>>> sheet = xlsfile['Sheet']
>>> for row in data:
... sheet.append(row)
```

1.  创建一个`BarChart`对象并填充基本信息：

```py
>>> chart = BarChart()
>>> chart.title = "Admissions per movie"
>>> chart.y_axis.title = 'Millions'
```

1.  创建对`data`的引用，并将`data`附加到图表：

```py
>>> data = Reference(sheet, min_row=2, max_row=4, min_col=1, max_col=2)
>>> chart.add_data(data, from_rows=True, titles_from_data=True)
```

1.  将图表添加到工作表并保存文件：

```py
>>> sheet.add_chart(chart, "A6")
>>> xlsfile.save('movie_chart.xlsx')
```

1.  在电子表格中检查生成的图表，如下截图所示：

![](assets/f68753d2-22dd-4b8c-8ed2-aa57da72b49c.png)

# 工作原理...

在*如何做...*部分，在步骤1和2中准备数据后，数据已准备在范围`A1:B4`中。请注意，`A1`和`B1`都包含不应在图表中使用的标题。

在步骤3中，我们设置了新图表并包括基本数据，如标题和*Y*轴的单位。

标题更改为`Millions`；虽然更正确的方式应该是`Admissions(millions)`，但这将与图表的完整标题重复。

步骤4通过`Reference`对象创建一个引用框，从第2行第1列到第4行第2列，这是我们的数据所在的区域，不包括标题。使用`.add_data`将数据添加到图表中。`from_rows`使每一行成为不同的数据系列。`titles_from_data`使第一列被视为系列的名称。

在步骤5中，将图表添加到单元格`A6`并保存到磁盘中。

# 还有更多...

可以创建各种不同的图表，包括柱状图、折线图、面积图（填充线和轴之间的区域的折线图）、饼图或散点图（其中一个值相对于另一个值绘制的XY图）。每种类型的图表都有一个等效的类，例如`PieChart`或`LineChart`。

同时，每个都可以具有不同的类型。例如，`BarChart`的默认类型是列，将柱形图垂直打印，但也可以选择不同的类型将其垂直打印：

```py
>>> chart.type = 'bar'
```

检查`openpyxl`文档以查看所有可用的组合。

可以使用`set_categories`来明确设置数据的*x*轴标签，而不是从数据中提取。例如，将步骤4与以下代码进行比较：

```py
data = Reference(sheet, min_row=2, max_row=4, min_col=2, max_col=2)
labels = Reference(sheet, min_row=2, max_row=4, min_col=1, max_col=1)
chart.add_data(data, from_rows=False, titles_from_data=False)
chart.set_categories(labels)
```

可以使用描述区域的文本标签来代替`Reference`对象的范围：

```py
chart.add_data('Sheet!B2:B4', from_rows=False, titles_from_data=False)
chart.set_categories('Sheet!A2:A4')
```

如果数据范围需要以编程方式创建，这种描述方式可能更难处理。

正确地在Excel中定义图表有时可能很困难。Excel从特定范围提取数据的方式可能令人困惑。记住要留出时间进行试验和错误，并处理差异。例如，在第4步中，我们定义了三个数据点的三个系列，而在前面的代码中，我们定义了一个具有三个数据点的单个系列。这些差异大多是微妙的。最后，最重要的是最终图表的外观。尝试不同的图表类型并了解差异。

完整的`openpyxl`文档可以在这里找到：

[https://openpyxl.readthedocs.io/en/stable/index.html](https://openpyxl.readthedocs.io/en/stable/index.html)。

# 另请参阅

+   *读取Excel电子表格*食谱

+   *更新Excel电子表格并添加注释*食谱

+   *在Excel电子表格上创建新工作表*食谱

+   *在Excel中处理格式*食谱

# 在Excel中处理格式

在电子表格中呈现信息不仅仅是将其组织到单元格中或以图表形式显示，还涉及更改格式以突出显示有关它的重要要点。在这个食谱中，我们将看到如何操纵单元格的格式以增强数据并以最佳方式呈现它。

# 准备工作

我们将使用`openpyxl`模块。我们应该安装该模块，并将其添加到我们的`requirements.txt`文件中，如下所示：

```py
$ echo "openpyxl==2.5.4" >> requirements.txt
$ pip install -r requirements.txt
```

我们将在新文件中存储有关出席人数最多的电影的信息。数据从这里提取：

[http://www.mrob.com/pub/film-video/topadj.html](http://www.mrob.com/pub/film-video/topadj.html)。

# 如何做...

1.  导入`openpyxl`模块并创建一个新的Excel文件：

```py
>>> import openpyxl
>>> from openpyxl.styles import Font, PatternFill, Border, Side
>>> xlsfile = openpyxl.Workbook()
```

1.  从来源中在此工作表中添加有关出席人数的数据。为简单起见，只添加前四个：

```py
>>> data = [
...    ('Name', 'Admissions'),
...    ('Gone With the Wind', 225.7),
...    ('Star Wars', 194.4),
...    ('ET: The Extraterrestrial', 161.0),
...    ('The Sound of Music', 156.4),
]
>>> sheet = xlsfile['Sheet']
>>> for row in data:
...    sheet.append(row)
```

1.  定义要用于样式化电子表格的颜色：

```py
>>> BLUE = "0033CC"
>>> LIGHT_BLUE = 'E6ECFF'
>>> WHITE = "FFFFFF"
```

1.  在蓝色背景和白色字体中定义标题：

```py
>>> header_font = Font(name='Tahoma', size=14, color=WHITE)
>>> header_fill = PatternFill("solid", fgColor=BLUE)
>>> for row in sheet['A1:B1']:
...     for cell in row:
...         cell.font = header_font
...         cell.fill = header_fill
```

1.  在标题后为列定义一个替代模式和每行一个边框：

```py
>>> white_side = Side(border_style='thin', color=WHITE)
>>> blue_side = Side(border_style='thin', color=BLUE)
>>> alternate_fill = PatternFill("solid", fgColor=LIGHT_BLUE)
>>> border = Border(bottom=blue_side, left=white_side, right=white_side)
>>> for row_index, row in enumerate(sheet['A2:B5']):
...     for cell in row:
...         cell.border = border
...         if row_index % 2:
...             cell.fill = alternate_fill
```

1.  将文件保存为`movies_format.xlsx`：

```py
>>> xlsfile.save('movies_format.xlsx')
```

1.  检查生成的文件：

![](assets/ef4c4635-405a-440e-ac36-4ad3ba1fe4c4.png)

# 它是如何工作的...

在*如何做...*部分，第1步中我们导入`openpyxl`模块并创建一个新的Excel文件。在第2步中，我们向第一个工作表添加数据。第3步也是一个准备步骤，用于定义要使用的颜色。颜色以十六进制格式定义，这在网页设计世界中很常见。

要找到颜色的定义，有很多在线颜色选择器，甚至嵌入在操作系统中。像[https://coolors.co/](https://coolors.co/)这样的工具可以帮助定义要使用的调色板。

在第4步中，我们准备格式以定义标题。标题将具有不同的字体（Tahoma）、更大的大小（14pt），并且将以蓝色背景上的白色显示。为此，我们准备了一个具有字体、大小和前景颜色的`Font`对象，以及具有背景颜色的`PatternFill`。

在创建`header_font`和`header_fill`后的循环将字体和填充应用到适当的单元格。

请注意，迭代范围始终返回行，然后是单元格，即使只涉及一行。

在第5步中，为行添加边框和交替背景。边框定义为蓝色顶部和底部，白色左侧和右侧。填充的创建方式与第4步类似，但是颜色是浅蓝色。背景只应用于偶数行。

请注意，单元格的顶部边框是上面一个单元格的底部，反之亦然。这意味着可能在循环中覆盖边框。

文件最终在第6步中保存。

# 还有更多...

要定义字体，还有其他可用的选项，如粗体、斜体、删除线或下划线。定义字体并重新分配它，如果需要更改任何元素。记得检查字体是否可用。

还有各种创建填充的方法。`PatternFill`接受几种模式，但最有用的是`solid`。`GradientFill`也可以用于应用双色渐变。

最好限制自己使用`PatternFill`进行实体填充。您可以调整颜色以最好地表示您想要的内容。记得包括`style='solid'`，否则颜色可能不会出现。

也可以定义条件格式，但最好尝试在Python中定义条件，然后应用适当的格式。

可以正确设置数字格式，例如：

```py
cell.style = 'Percent'
```

这将显示值`0.37`为`37%`。

完整的`openpyxl`文档可以在这里找到：

[https://openpyxl.readthedocs.io/en/stable/index.html](https://openpyxl.readthedocs.io/en/stable/index.html)。

# 另请参见

+   *读取Excel电子表格*配方

+   *更新Excel电子表格并添加注释*配方

+   *在Excel电子表格中创建新工作表*配方

+   *在Excel中创建图表*配方

# 在LibreOffice中创建宏

LibreOffice是一个免费的办公套件，是MS Office和其他办公套件的替代品。它包括一个文本编辑器和一个名为`Calc`的电子表格程序。Calc可以理解常规的Excel格式，并且也可以通过其UNO API在内部进行完全脚本化。UNO接口允许以编程方式访问套件，并且可以用不同的语言（如Java）进行访问。

其中一种可用的语言是Python，这使得在套件格式中生成非常复杂的应用程序非常容易，因为这样可以使用完整的Python标准库。

使用完整的Python标准库可以访问诸如加密、打开外部文件（包括ZIP文件）或连接到远程数据库等元素。此外，利用Python语法，避免使用LibreOffice BASIC。

在本配方中，我们将看到如何将外部Python文件作为宏添加到电子表格中，从而改变其内容。

# 准备工作

需要安装LibreOffice。它可以在[https://www.libreoffice.org/](https://www.libreoffice.org/)上找到。

下载并安装后，需要配置以允许执行宏：

1.  转到设置|安全以查找宏安全详细信息：

![](assets/bbdc16ad-dbf6-4455-8129-73b9d7d77d1d.png)

1.  打开宏安全并选择中等以允许执行我们的宏。这将在允许运行宏之前显示警告：

![](assets/99fc81d8-fb85-40bf-94ab-62dc031a733f.png)

要将宏插入文件中，我们将使用一个名为`include_macro.py`的脚本，该脚本可在[https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/include_macro.py](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/include_macro.py)上找到。带有宏的脚本也可以在此处作为`libreoffice_script.py`找到：

[https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/libreoffice_script.py](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/libreoffice_script.py)。

要将脚本放入的文件名为`movies.ods`的文件也可以在此处找到：[https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/movies.ods](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter06/movies.ods)。它以`.ods`格式（LibreOffice格式）包含了10部入场人数最高的电影的表格。数据是从这里提取的：

[http://www.mrob.com/pub/film-video/topadj.html](http://www.mrob.com/pub/film-video/topadj.html)。

# 如何做...

1.  使用`include_macro.py`脚本将`libreoffice_script.py`附加到文件`movies.ods`的宏文件中：

```py
$ python include_macro.py -h
usage: It inserts the macro file "script" into the file "spreadsheet" in .ods format. The resulting file is located in the macro_file directory, that will be created
 [-h] spreadsheet script

positional arguments:
 spreadsheet File to insert the script
 script Script to insert in the file

optional arguments:
 -h, --help show this help message and exit

$ python include_macro.py movies.ods libreoffice_script.py
```

1.  在LibreOffice中打开生成的文件`macro_file/movies.ods`。请注意，它会显示一个警告以启用宏（单击启用）。转到工具|宏|运行宏：

![](assets/a6d8b67c-0e48-4fff-bc98-1db332741dfa.png)

1.  在`movies.ods` | `libreoffice_script`宏下选择`ObtainAggregated`并单击运行。它计算聚合入场人数并将其存储在单元格`B12`中。它在`A15`中添加了一个`Total`标签：

![](assets/02d9a18b-278d-47dd-90b7-8c855e61077f.png)

1.  重复步骤2和3以再次运行。现在它运行所有的聚合，但是将`B12`相加，并在`B13`中得到结果：

![](assets/b861820b-7343-4563-8695-a6d9dc85a16d.png)

# 工作原理...

步骤1中的主要工作在`include_macro.py`脚本中完成。它将文件复制到`macro_file`子目录中，以避免修改输入。

在内部，`.ods`文件是一个具有特定结构的ZIP文件。脚本利用ZIP文件Python模块，将脚本添加到内部的适当子目录中。它还修改`manifest.xml`文件，以便LibreOffice知道文件中有一个脚本。

在步骤3中执行的宏在`libreoffice_script.py`中定义，并包含一个函数：

```py
def ObtainAggregated(*args):
    """Prints the Python version into the current document"""
    # get the doc from the scripting context
    # which is made available to all scripts
    desktop = XSCRIPTCONTEXT.getDesktop()
    model = desktop.getCurrentComponent()
    # get the first sheet
    sheet = model.Sheets.getByIndex(0)

    # Find the admissions column
    MAX_ELEMENT = 20
    for column in range(0, MAX_ELEMENT):
        cell = sheet.getCellByPosition(column, 0)
        if 'Admissions' in cell.String:
            break
    else:
        raise Exception('Admissions not found')

    accumulator = 0.0
    for row in range(1, MAX_ELEMENT):
        cell = sheet.getCellByPosition(column, row)
        value = cell.getValue()
        if value:
            accumulator += cell.getValue()
        else:
            break

    cell = sheet.getCellByPosition(column, row)
    cell.setValue(accumulator)

    cell = sheet.getCellRangeByName("A15")
    cell.String = 'Total'
    return None
```

变量`XSCRIPTCONTEXT`会自动创建并允许获取当前组件，然后获取第一个`Sheet`。之后，通过`.getCellByPosition`迭代表找到`Admissions`列，并通过`.String`属性获取字符串值。使用相同的方法，聚合列中的所有值，通过`.getValue`提取它们的数值。

当循环遍历列直到找到空单元格时，第二次执行时，它将聚合`B12`中的值，这是上一次执行中的聚合值。这是故意为了显示宏可以多次执行，产生不同的结果。

还可以通过`.getCellRangeByName`按其字符串位置引用单元格，将`Total`存储在单元格`A15`中。

# 还有更多...

Python解释器嵌入到LibreOffice中，这意味着如果LibreOffice发生变化，特定版本也会发生变化。在撰写本书时的最新版本的LibreOffice（6.0.5）中，包含的版本是Python 3.5.1。

UNO接口非常完整，可以访问许多高级元素。不幸的是，文档不是很好，获取起来可能会很复杂和耗时。文档是用Java或C++定义的，LibreOffice BASIC或其他语言中有示例，但Python的示例很少。完整的文档可以在这里找到：[https://api.libreoffice.org/](https://api.libreoffice.org/)，参考在这里：

[https://api.libreoffice.org/docs/idl/ref/index.html](https://api.libreoffice.org/docs/idl/ref/index.html)。

例如，可以创建复杂的图表，甚至是要求用户提供并处理响应的交互式对话框。在论坛和旧答案中有很多信息。基本代码大多数时候也可以适应Python。

LibreOffice是以前的项目OpenOffice的一个分支。UNO已经可用，这意味着在搜索互联网时会找到一些涉及OpenOffice的引用。

请记住，LibreOffice能够读取和写入Excel文件。一些功能可能不是100%兼容；例如，可能会出现格式问题。

出于同样的原因，完全可以使用本章其他食谱中描述的工具生成Excel格式的文件，并在LibreOffice中打开。这可能是一个不错的方法，因为`openpyxl`的文档更好。

调试有时也可能会很棘手。记住确保在用新代码重新打开文件之前，文件已完全关闭。

UNO还能够与LibreOffice套件的其他部分一起工作，比如创建文档。

# 另请参阅

+   *编写CSV电子表格*食谱

+   *更新Excel电子表格并添加注释和公式*食谱
