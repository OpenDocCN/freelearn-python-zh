# 处理各种文件

在这一章中，您将学习如何处理各种类型的文件，如PDF文件、Excel文件、CSV文件和`txt`文件。Python有用于在这些文件上执行操作的模块。您将学习如何使用Python打开、编辑和获取这些文件中的数据。

在本章中，将涵盖以下主题：

+   处理PDF文件

+   处理Excel文件

+   处理CSV文件

+   处理`txt`文件

# 处理PDF文件

在本节中，我们将学习如何使用Python模块处理PDF文件。PDF是一种广泛使用的文档格式，PDF文件的扩展名为`.pdf`。Python有一个名为`PyPDF2`的模块，对`pdf`文件进行各种操作非常有用。它是一个第三方模块，是作为PDF工具包构建的Python库。

我们必须首先安装这个模块。要安装`PyPDF2`，请在终端中运行以下命令：

```py
pip3 install PyPDF2
```

现在，我们将看一些操作来处理PDF文件，比如读取PDF、获取页面数、提取文本和旋转PDF页面。

# 阅读PDF文档并获取页面数

在本节中，我们将使用`PyPDF2`模块读取PDF文件。此外，我们将获取该PDF的页面数。该模块有一个名为`PdfFileReader()`的函数，可以帮助读取PDF文件。确保您的系统中有一个PDF文件。现在，我在我的系统中有`test.pdf`文件，所以我将在本节中使用这个文件。在`test.pdf`的位置输入您的PDF文件名。创建一个名为`read_pdf.py`的脚本，并在其中编写以下内容：

```py
import PyPDF2 with open('test.pdf', 'rb') as pdf:
 read_pdf= PyPDF2.PdfFileReader(pdf)
    print("Number of pages in pdf : ", read_pdf.numPages)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 read_pdf.py
```

以下是输出：

```py
Number of pages in pdf :  20
```

在前面的示例中，我们使用了`PyPDF2`模块。接下来，我们创建了一个`pdf`文件对象。`PdfFileReader()`将读取创建的对象。读取PDF文件后，我们将使用`numPages`属性获取该`pdf`文件的页面数。在这种情况下，有`20`页。

# 提取文本

要提取`pdf`文件的页面，`PyPDF2`模块有`extractText()`方法。创建一个名为`extract_text.py`的脚本，并在其中编写以下内容：

```py
import PyPDF2 with open('test.pdf', 'rb') as pdf:
 read_pdf = PyPDF2.PdfFileReader(pdf) pdf_page = read_pdf.getPage(1) pdf_content = pdf_page.extractText() print(pdf_content) 
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 extract_text.py
```

以下是输出：

```py
3Pythoncommands 9 3.1Comments........................................ .9 3.2Numbersandotherdatatypes........................ ......9 3.2.1The type function................................9 3.2.2Strings....................................... 10 3.2.3Listsandtuples................................ ..10 3.2.4The range function................................11 3.2.5Booleanvalues................................. .11 3.3Expressions..................................... ...11 3.4Operators.......................................
```

在前面的示例中，我们创建了一个文件阅读器对象。`pdf`阅读器对象有一个名为`getPage()`的函数，它以页面编号（从第0页开始）作为参数，并返回页面对象。接下来，我们使用`extractText()`方法，它将从我们在`getPage()`中提到的页面编号中提取文本。页面索引从`0`开始。

# 旋转PDF页面

在本节中，我们将看到如何旋转PDF页面。为此，我们将使用`PDF`对象的`rotate.Clockwise()`方法。创建一个名为`rotate_pdf.py`的脚本，并在其中编写以下内容：

```py
import PyPDF2

with open('test.pdf', 'rb') as pdf:
 rd_pdf = PyPDF2.PdfFileReader(pdf)
 wr_pdf = PyPDF2.PdfFileWriter()
 for pg_num in range(rd_pdf.numPages):
 pdf_page = rd_pdf.getPage(pg_num)
 pdf_page.rotateClockwise(90)
 wr_pdf.addPage(pdf_page)

 with open('rotated.pdf', 'wb') as pdf_out:
 wr_pdf.write(pdf_out)

print("pdf successfully rotated")
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 rotate_pdf.py
```

以下是输出：

```py
pdf successfully rotated
```

在前面的示例中，为了旋转`pdf`，我们首先创建了原始`pdf`文件的`pdf`文件阅读器对象。然后旋转的页面将被写入一个新的`pdf`文件。因此，为了写入新的`pdf`，我们使用`PyPDF2`模块的`PdfFileWriter()`函数。新的`pdf`文件将以名称`rotated.pdf`保存。现在，我们将使用`rotateClockwise()`方法旋转`pdf`文件的页面。然后，使用`addPage()`方法将页面添加到旋转后的`pdf`中。现在，我们必须将这些`pdf`页面写入新的`pdf`文件。因此，首先我们必须打开新的文件对象（`pdf_out`），并使用`pdf`写入对象的`write()`方法将`pdf`页面写入其中。在所有这些之后，我们将关闭原始（`test.pdf`）文件对象和新的（`pdf_out`）文件对象。

# 处理Excel文件

在本节中，我们将处理具有`.xlsx`扩展名的Excel文件。这个文件扩展名是用于Microsoft Excel使用的一种开放的XML电子表格文件格式。

Python有不同的模块：`xlrd`，pandas和`openpyxl`用于处理Excel文件。在本节中，我们将学习如何使用这三个模块处理Excel文件。

首先，我们将看一个使用`xlrd`模块的例子。`xlrd`模块用于读取、写入和修改Excel电子表格以及执行大量工作。

# 使用xlrd模块

首先，我们必须安装`xlrd`模块。在终端中运行以下命令以安装`xlrd`模块：

```py
   pip3 install xlrd
```

注意：确保您的系统中有一个Excel文件。我在我的系统中有`sample.xlsx`。所以我将在本节中始终使用该文件。

我们将学习如何读取Excel文件以及如何从Excel文件中提取行和列。

# 读取Excel文件

在本节中，我们将学习如何读取Excel文件。我们将使用`xlrd`模块。创建一个名为`read_excel.py`的脚本，并在其中写入以下内容：

```py
import xlrd excel_file = (r"/home/student/sample.xlsx") book_obj = xlrd.open_workbook(excel_file) excel_sheet = book_obj.sheet_by_index(0) result = excel_sheet.cell_value(0, 1)
print(result)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 read_excel.py
```

以下是输出：

```py
First Name
```

在前面的例子中，我们导入了`xlrd`模块来读取Excel文件。我们还提到了Excel文件的位置。然后，我们创建了一个文件对象，然后我们提到了索引值，以便从该索引开始阅读。最后，我们打印了结果。

# 提取列名

在本节中，我们正在从Excel表中提取列名。创建一个名为`extract_column_names.py`的脚本，并在其中写入以下内容：

```py
import xlrd excel_file = ("/home/student/work/sample.xlsx") book_obj = xlrd.open_workbook(excel_file) excel_sheet = book_obj.sheet_by_index(0) excel_sheet.cell_value(0, 0) for i in range(excel_sheet.ncols):
 print(excel_sheet.cell_value(0, i))
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 extract_column_names.py
```

以下是输出：

```py
Id First Name Last Name Gender Age Country
```

在前面的例子中，我们正在从Excel表中提取列名。我们使用`ncols`属性获取了列名。

# 使用pandas

在使用Pandas读取Excel文件之前，我们首先必须安装`pandas`模块。我们可以使用以下命令安装`pandas`：

```py
 pip3 install pandas
```

注意：确保您的系统中有一个Excel文件。我在我的系统中有`sample.xlsx`。所以我将在本节中始终使用该文件。

现在，我们将看一些使用`pandas`的例子。

# 读取Excel文件

在本节中，我们将使用`pandas`模块读取Excel文件。现在，让我们看一个读取Excel文件的例子。

创建一个名为`rd_excel_pandas.py`的脚本，并在其中写入以下内容：

```py
import pandas as pd 
excel_file = 'sample.xlsx'
df = pd.read_excel(excel_file)
print(df.head())
```

运行上述脚本，您将获得以下输出：

```py
student@ubuntu:~/test$ python3 rd_excel_pandas.py
```

以下是输出：

```py
 OrderDate     Region  ...   Unit Cost     Total
0  2014-01-09   Central  ...    125.00      250.00
1   6/17/15     Central    ...  125.00      625.00
2  2015-10-09   Central    ...    1.29        9.03
3  11/17/15     Central   ...     4.99       54.89
4  10/31/15     Central   ...     1.29       18.06
```

在前面的例子中，我们正在使用`pandas`模块读取Excel文件。首先，我们导入了`pandas`模块。然后，我们创建了一个名为`excel_file`的字符串，用于保存要打开的文件的名称，我们希望使用pandas进行操作。随后，我们创建了一个`df数据框`对象。在这个例子中，我们使用了pandas的`read_excel`方法来从Excel文件中读取数据。读取从索引零开始。最后，我们打印了`pandas`数据框。

# 在Excel文件中读取特定列

当我们使用pandas模块使用`read_excel`方法读取Excel文件时，我们还可以读取该文件中的特定列。要读取特定列，我们需要在`read_excel`方法中使用`usecols`参数。

现在，让我们看一个示例，读取Excel文件中的特定列。创建一个名为`rd_excel_pandas1.py`的脚本，并在其中写入以下内容：

```py
import pandas as pd

excel_file = 'sample.xlsx'
cols = [1, 2, 3]
df = pd.read_excel(excel_file , sheet_names='sheet1', usecols=cols)

print(df.head())
```

运行上述脚本，您将获得以下输出：

```py
student@ubuntu:~/test$ python3 rd_excel_pandas1.py
```

以下是输出：

```py
 Region      Rep    Item
0  Central    Smith    Desk
1  Central   Kivell    Desk
2  Central     Gill  Pencil
3  Central  Jardine  Binder
4  Central  Andrews  Pencil
```

在前面的例子中，首先我们导入了pandas模块。然后，我们创建了一个名为`excel_file`的字符串来保存文件名。然后我们定义了`cols`变量，并将列的索引值放在其中。因此，当我们使用`read_excel`方法时，在该方法内部，我们还提供了`usecols`参数，通过该参数可以通过之前在`cols`变量中定义的索引获取特定列。因此，在运行脚本后，我们只从Excel文件中获取特定列。

我们还可以使用pandas模块对Excel文件执行各种操作，例如读取具有缺失数据的Excel文件，跳过特定行以及读取多个Excel工作表。

# 使用openpyxl

`openpyxl`是一个用于读写`xlsx`，`xlsm`，`xltx`和`xltm`文件的Python库。首先，我们必须安装`openpyxl`。运行以下命令：

```py
 pip3 install openpyxl
```

现在，我们将看一些使用`openpyxl`的示例。

# 创建新的Excel文件

在本节中，我们将学习使用`openpyxl`创建新的Excel文件。创建一个名为`create_excel.py`的脚本，并在其中写入以下内容：

```py
from openpyxl import Workbook book_obj = Workbook() excel_sheet = book_obj.active excel_sheet['A1'] = 'Name' excel_sheet['A2'] = 'student' excel_sheet['B1'] = 'age' excel_sheet['B2'] = '24' book_obj.save("test.xlsx") print("Excel created successfully")
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 create_excel.py
```

以下是输出：

```py
Excel created successfully
```

现在，检查您当前的工作目录，您会发现`test.xlsx`已成功创建。在前面的示例中，我们将数据写入了四个单元格。然后，从`openpyxl`模块中导入`Workbook`类。工作簿是文档的所有其他部分的容器。接下来，我们将引用对象设置为活动工作表，并在单元格`A1`，`A2`和`B1`，`B2`中写入数值。最后，我们使用`save()`方法将内容写入`test.xlsx`文件。

# 追加数值

在本节中，我们将在Excel中追加数值。为此，我们将使用`append()`方法。我们可以在当前工作表的底部添加一组数值。创建一个名为`append_values.py`的脚本，并在其中写入以下内容：

```py
from openpyxl import Workbookbook_obj = Workbook() excel_sheet = book_obj.active rows = (
 (11, 12, 13), (21, 22, 23), (31, 32, 33), (41, 42, 43) ) for values in rows: excel_sheet.append(values) print() print("values are successfully appended") book_obj.save('test.xlsx')wb.save('append_values.xlsx')
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 append_values.py
```

以下是输出：

```py
values are successfully appended
```

在前面的示例中，我们在`append_values.xlsx`文件的工作表中追加了三列数据。我们存储的数据是元组的元组，并且为了追加这些数据，我们逐行通过容器并使用`append()`方法插入它。

# 读取多个单元格

在本节中，我们将读取多个单元格。我们将使用`openpyxl`模块。创建一个名为`read_multiple.py`的脚本，并在其中写入以下内容：

```py
import openpyxl book_obj = openpyxl.load_workbook('sample.xlsx') excel_sheet = book_obj.active cells = excel_sheet['A1': 'C6'] for c1, c2, c3 in cells:
 print("{0:6} {1:6} {2:6}".format(c1.value, c2.value, c3.value))
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 read_multiple.py
```

以下是输出：

```py
Id     First Name Last Name
 101 John   Smith 102 Mary   Williams 103 Rakesh Sharma 104 Amit   Roy105 Sandra Ace 
```

在前面的示例中，我们使用`range`操作读取了三列数据。然后，我们从单元格`A1 – C6`中读取数据。

同样地，我们可以使用`openpyxl`模块在Excel文件上执行许多操作，比如合并和拆分单元格。

# 处理CSV文件

**CSV**格式代表**逗号分隔值**。逗号用于分隔记录中的字段。这些通常用于导入和导出电子表格和数据库的格式。

CSV文件是使用特定类型的结构来排列表格数据的纯文本文件。Python具有内置的`csv`模块，允许Python解析这些类型的文件。`csv`模块主要用于处理从电子表格以及数据库以文本文件格式导出的数据，包括字段和记录。

`csv`模块具有所有必需的内置函数，如下所示：

+   `csv.reader`：此函数用于返回一个`reader`对象，该对象迭代CSV文件的行

+   `csv.writer`：此函数用于返回一个`writer`对象，该对象将数据写入CSV文件

+   `csv.register_dialect`：此函数用于注册CSV方言

+   `csv.unregister_dialect`：此函数用于取消注册CSV方言

+   `csv.get_dialect`：此函数用于返回具有给定名称的方言

+   `csv.list_dialects`：此函数用于返回所有已注册的方言

+   `csv.field_size_limit`：此函数用于返回解析器允许的当前最大字段大小

在本节中，我们将只看`csv.reader`和`csv.writer`。

# 读取CSV文件

Python具有内置模块`csv`，我们将在此处使用它来处理CSV文件。我们将使用`csv.reader`模块来读取CSV文件。创建一个名为`csv_read.py`的脚本，并在其中写入以下内容：

```py
import csv csv_file = open('test.csv', 'r') with csv_file:
 read_csv = csv.reader(csv_file) for row in read_csv: print(row)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 csv_read.py
```

以下是输出：

```py
['Region', 'Country', 'Item Type', 'Sales Channel', 'Order Priority', 'Order Date', 'Order ID', 'Ship Date', 'Units Sold'] ['Sub-Saharan Africa', 'Senegal', 'Cereal', 'Online', 'H', '4/18/2014', '616607081', '5/30/2014', '6593'] ['Asia', 'Kyrgyzstan', 'Vegetables', 'Online', 'H', '6/24/2011', '814711606', '7/12/2011', '124'] ['Sub-Saharan Africa', 'Cape Verde', 'Clothes', 'Offline', 'H', '8/2/2014', '939825713', '8/19/2014', '4168'] ['Asia', 'Bangladesh', 'Clothes', 'Online', 'L', '1/13/2017', '187310731', '3/1/2017', '8263'] ['Central America and the Caribbean', 'Honduras', 'Household', 'Offline', 'H', '2/8/2017', '522840487', '2/13/2017', '8974'] ['Asia', 'Mongolia', 'Personal Care', 'Offline', 'C', '2/19/2014', '832401311', '2/23/2014', '4901'] ['Europe', 'Bulgaria', 'Clothes', 'Online', 'M', '4/23/2012', '972292029', '6/3/2012', '1673'] ['Asia', 'Sri Lanka', 'Cosmetics', 'Offline', 'M', '11/19/2016', '419123971', '12/18/2016', '6952'] ['Sub-Saharan Africa', 'Cameroon', 'Beverages', 'Offline', 'C', '4/1/2015', '519820964', '4/18/2015', '5430'] ['Asia', 'Turkmenistan', 'Household', 'Offline', 'L', '12/30/2010', '441619336', '1/20/2011', '3830']
```

在上述程序中，我们将`test.csv`文件作为`csv_file`打开。然后，我们使用`csv.reader()`函数将数据提取到`reader`对象中，我们可以迭代以获取数据的每一行。现在，我们将看一下第二个函数`csv.Writer()`

# 写入CSV文件

要在`csv`文件中写入数据，我们使用`csv.writer`模块。在本节中，我们将一些数据存储到Python列表中，然后将该数据放入`csv`文件中。创建一个名为`csv_write.py`的脚本，并在其中写入以下内容：

```py
import csv write_csv = [['Name', 'Sport'], ['Andres Iniesta', 'Football'], ['AB de Villiers', 'Cricket'], ['Virat Kohli', 'Cricket'], ['Lionel Messi', 'Football']] with open('csv_write.csv', 'w') as csvFile:
 writer = csv.writer(csvFile) writer.writerows(write_csv) print(write_csv)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 csv_write.py
```

以下是输出：

```py
[['Name', 'Sport'], ['Andres Iniesta', 'Football'], ['AB de Villiers', 'Cricket'], ['Virat Kohli', 'Cricket'], ['Lionel Messi', 'Football']]
```

在上述程序中，我们创建了一个名为`write_csv`的列表，其中包含`Name`和`Sport`。然后，在创建列表后，我们打开了新创建的`csv_write.csv`文件，并使用`csvWriter()`函数将`write_csv`列表插入其中。

# 处理txt文件

纯文本文件用于存储仅表示字符或字符串的数据，并且不考虑任何结构化元数据。在Python中，无需导入任何外部库来读写文本文件。Python提供了一个内置函数来创建、打开、关闭、写入和读取文本文件。为了执行操作，有不同的访问模式来管理在打开文件中可能的操作类型。

Python中的访问模式如下：

+   **仅读取模式（`'r'`）**：此模式打开文本文件以供读取。如果文件不存在，它会引发I/O错误。我们也可以称此模式为文件将打开的默认模式。

+   **读写模式（`'r+'`）**：此模式打开文本文件以供读取和写入，并在文件不存在时引发I/O错误。

+   **仅写入模式（`'w'`）**：此模式将打开文本文件以供写入。如果文件不存在，则创建文件，并且对于现有文件，数据将被覆盖。

+   **写入和读取模式（`'w+'`）**：此模式将打开文本文件以供读取和写入。对于现有文件，数据将被覆盖。

+   **仅追加模式（`'a'`）**：此模式将打开文本文件以供写入。如果文件不存在，则创建文件，并且数据将被插入到现有数据的末尾。

+   **追加和读取模式（`'a+'`）**：此模式将打开文本文件以供读取和写入。如果文件不存在，则会创建文件，并且数据将被插入到现有数据的末尾。

# `open()`函数

此函数用于打开文件，不需要导入任何外部模块。

语法如下：

```py
 Name_of_file_object = open("Name of file","Access_Mode")
```

对于前面的语法，文件必须在我们的Python程序所在的相同目录中。如果文件不在同一目录中，那么在打开文件时我们还必须定义文件路径。这种情况的语法如下所示：

```py
Name_of_file_object = open("/home/……/Name of file","Access_Mode")
```

# 文件打开

打开文件的`open`函数为`"test.txt"`。

文件与`追加`模式相同的目录中：

```py
text_file = open("test.txt","a")
```

如果文件不在相同的目录中，我们必须在`追加`模式中定义路径：

```py
text_file = open("/home/…../test.txt","a")
```

# close()函数

此函数用于关闭文件，释放文件获取的内存。当文件不再需要或将以不同的文件模式打开时使用此函数。

语法如下：

```py
 Name_of_file_object.close()
```

以下代码语法可用于简单地打开和关闭文件：

```py
#Opening and closing a file test.txt:
text_file = open("test.txt","a") text_file.close()
```

# 写入文本文件

通过使用Python，您可以创建一个文本文件（`test.txt`）。通过使用代码，写入文本文件很容易。要打开一个文件进行写入，我们将第二个参数设置为访问模式中的`"w"`。要将数据写入`test.txt`文件，我们使用`file handle`对象的`write()`方法。创建一个名为`text_write.py`的脚本，并在其中写入以下内容：

```py
text_file = open("test.txt", "w") text_file.write("Monday\nTuesday\nWednesday\nThursday\nFriday\nSaturday\n") text_file.close()
```

运行上述脚本，您将获得以下输出：

![](assets/6620eefb-81eb-459b-b9b1-8c43968a5850.jpg)

现在，检查您的当前工作目录。您会发现一个我们创建的`test.txt`文件。现在，检查文件的内容。您会发现我们在`write()`函数中写入的日期将保存在`test.txt`中。

在上述程序中，我们声明了`text_file`变量来打开名为`test.txt`的文件。`open`函数接受两个参数：第一个是我们要打开的文件，第二个是表示我们要在文件上执行的权限或操作的访问模式。在我们的程序中，我们在第二个参数中使用了`"w"`字母，表示写入。然后，我们使用`text_file.close()`来关闭存储的`test.txt`文件的实例。

# 读取文本文件

读取文件和写入文件一样容易。要打开一个文件进行读取，我们将第二个参数即访问模式设置为`"r"`，而不是`"w"`。要从该文件中读取数据，我们使用`文件句柄`对象的`read()`方法。创建一个名为`text_read.py`的脚本，并在其中写入以下内容：

```py
text_file = open("test.txt", "r") data = text_file.read() print(data) text_file.close()
```

以下是输出：

```py
student@ubuntu:~$ python3 text_read.py Monday Tuesday Wednesday Thursday Friday Saturday
```

在上述程序中，我们声明了`text_file`变量来打开名为`test.txt`的文件。`open`函数接受两个参数：第一个是我们要打开的文件，第二个是表示我们要在文件上执行的权限或操作的访问模式。在我们的程序中，我们在第二个参数中使用了`"r"`字母，表示读取操作。然后，我们使用`text_file.close()`来关闭存储的`test.txt`文件的实例。运行Python程序后，我们可以在终端中轻松看到文本文件中的内容。

# 总结

在本章中，我们学习了各种文件。我们学习了PDF、Excel、CSV和文本文件。我们使用Python模块对这些类型的文件执行了一些操作。

在下一章中，我们将学习Python中的基本网络和互联网模块。

# 问题

1.  `readline()`和`readlines()`之间有什么区别？

1.  `open()`和`with open()`之间有什么区别？

1.  `r c:\\Downloads`的意义是什么？

1.  生成器对象是什么？

1.  `pass`的用途是什么？

1.  什么是lambda表达式？

# 进一步阅读

+   XLRD：[https://xlrd.readthedocs.io/en/latest/api.html](https://xlrd.readthedocs.io/en/latest/api.html)

+   `openoyxl`：[http://www.python-excel.org/](http://www.python-excel.org/)

+   关于生成器概念：[https://wiki.python.org/moin/Generators](https://wiki.python.org/moin/Generators)
