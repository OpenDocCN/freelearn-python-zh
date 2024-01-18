# 使用Treeview导航记录

您收到了应用程序中的另一个功能请求。现在，您的用户可以打开任意文件，他们希望能够查看这些文件中的内容，并使用他们已经习惯的数据输入表单来更正旧记录，而不必切换到电子表格。简而言之，现在终于是时候在我们的应用程序中实现读取和更新功能了。

在本章中，我们将涵盖以下主题：

+   修改我们的CSV模型以实现读取和更新功能

+   发现ttk`Treeview`小部件，并使用它构建记录列表

+   在我们的数据记录表单中实现记录加载和更新

+   重新设计菜单和应用程序，考虑到读取和更新

# 在模型中实现读取和更新

到目前为止，我们的整个设计都是围绕着一个只能向文件追加数据的表单；添加读取和更新功能是一个根本性的改变，几乎会触及应用程序的每个部分。这可能看起来是一项艰巨的任务，但通过逐个组件地进行，我们会发现这些变化并不那么令人难以承受。

我们应该做的第一件事是更新我们的文档，从`Requirements`部分开始：

```py
The program must:

* Provide a UI for reading, updating, and appending data to the CSV file
* ...
```

当然，还要更新后面不需要的部分：

```py
The program does not need to:

* Allow deletion of data.
```

现在，只需让代码与文档匹配即可。

# 将读取和更新添加到我们的模型中

打开`models.py`并考虑`CSVModel`类中缺少的内容：

+   我们需要一种方法，可以检索文件中的所有记录，以便我们可以显示它们。我们将其称为`get_all_records()`。

+   我们需要一种方法来按行号从文件中获取单个记录。我们可以称之为`get_record()`。

+   我们需要以一种既能追加新记录又能更新现有记录的方式保存记录。我们可以更新我们的`save_record()`方法来适应这一点。

# 实现`get_all_records()`

开始一个名为`get_all_records()`的新方法：

```py
    def get_all_records(self):
        if not os.path.exists(self.filename):
            return []
```

我们所做的第一件事是检查模型的文件是否已经存在。请记住，当我们的应用程序启动时，它会生成一个默认文件名，指向一个可能尚不存在的文件，因此`get_all_records()`将需要优雅地处理这种情况。在这种情况下返回一个空列表是有道理的，因为如果文件不存在，就没有数据。

如果文件存在，让我们以只读模式打开它并获取所有记录：

```py
        with open(self.filename, 'r') as fh:
            csvreader = csv.DictReader(fh)
            records = list(csvreader)
```

虽然不是非常高效，但在我们的情况下，将整个文件加载到内存中并将其转换为列表是可以接受的，因为我们知道我们的最大文件应该限制在仅401行：20个图形乘以5个实验室加上标题行。然而，这段代码有点太信任了。我们至少应该进行一些合理性检查，以确保用户实际上已经打开了包含正确字段的CSV文件，而不是其他任意文件。

让我们检查文件是否具有正确的字段结构：

```py
        csvreader = csv.DictReader(fh)
        missing_fields = (set(self.fields.keys()) -    
                          set(csvreader.fieldnames))
        if len(missing_fields) > 0:
            raise Exception(
                "File is missing fields: {}"
                .format(', '.join(missing_fields))
            )
        else:
            records = list(csvreader)
```

在这里，我们首先通过将我们的`fields`字典`keys`的列表和CSV文件的`fieldnames`转换为Python`set`对象来找到任何缺失的字段。我们可以从`keys`中减去`fieldnames`集合，并确定文件中缺少的字段。如果有任何字段缺失，我们将引发异常；否则，我们将CSV数据转换为`list`。

Python的`set`对象非常有用，可以比较`list`、`tuple`和其他序列对象的内容。它们提供了一种简单的方法来获取诸如差异（`x`中的项目不在`y`中）或交集（`x`和`y`中的项目）之类的信息，或者允许您比较不考虑顺序的序列。

在我们可以返回`records`列表之前，我们需要纠正一个问题；CSV文件中的所有数据都存储为文本，并由Python作为字符串读取。这大多数情况下不是问题，因为Tkinter会负责根据需要将字符串转换为`float`或`int`，但是`bool`值在CSV文件中存储为字符串`True`和`False`，直接将这些值强制转换回`bool`是行不通的。`False`是一个非空字符串，在Python中所有非空字符串都会被视为`True`。

为了解决这个问题，让我们首先定义一个应被解释为`True`的字符串列表：

```py
        trues = ('true', 'yes', '1')
```

不在此列表中的任何值都将被视为`False`。我们将进行不区分大小写的比较，因此我们的列表中只有小写值。

接下来，我们使用列表推导式创建一个包含`boolean`字段的字段列表，如下所示：

```py
        bool_fields = [
            key for key, meta
            in self.fields.items()
            if meta['type'] == FT.boolean]
```

我们知道`Equipment Fault`是我们唯一的布尔字段，因此从技术上讲，我们可以在这里硬编码它，但是最好设计您的模型，以便对模式的任何更改都将自动适当地处理逻辑部分。

现在，让我们通过添加以下代码来检查每行中的布尔字段：

```py
        for record in records:
            for key in bool_fields:
                record[key] = record[key].lower() in trues
```

对于每条记录，我们遍历我们的布尔字段列表，并根据我们的真值字符串列表检查其值，相应地设置该项的值。

修复布尔值后，我们可以将我们的`records`列表返回如下：

```py
        return records
```

# 实现`get_record()`

我们的`get_record()`方法需要接受行号并返回包含该行数据的单个字典。

如果我们利用我们的`get_all_records()`方法，这就非常简单了，如下所示：

```py
    def get_record(self, rownum):
        return self.get_all_records()[rownum]
```

由于我们的文件很小，拉取所有记录的开销很小，我们可以简单地这样做，然后取消引用我们需要的记录。

请记住，可能会传递不存在于我们记录列表中的`rownum`；在这种情况下，我们会得到`IndexError`；我们的调用代码将需要捕获此错误并适当处理。

# 将更新添加到`save_record()`

将我们的`save_record()`方法转换为可以更新记录的方法，我们首先需要做的是提供传入要更新的行号的能力。默认值将是`None`，表示数据是应追加的新行。

新的方法签名如下：

```py
    def save_record(self, data, rownum=None):
        """Save a dict of data to the CSV file"""
```

我们现有的逻辑不需要更改，但只有在`rownum`为`None`时才应运行。

因此，在该方法中要做的第一件事是检查`rownum`：

```py
        if rownum is not None:
            # This is an update, new code here
        else:
            # Old code goes here, indented one more level
```

对于相对较小的文件，更新单行的最简单方法是将整个文件加载到列表中，更改列表中的行，然后将整个列表写回到一个干净的文件中。

在`if`块下，我们将添加以下代码：

```py
            records = self.get_all_records()
            records[rownum] = data
            with open(self.filename, 'w') as fh:
                csvwriter = csv.DictWriter(fh,
                    fieldnames=self.fields.keys())
                csvwriter.writeheader()
                csvwriter.writerows(records)
```

再次利用我们的`get_all_records()`方法将CSV文件的内容提取到列表中。然后，我们用提供的`data`字典替换请求行中的字典。最后，我们以写模式（`w`）打开文件，这将清除其内容并用我们写入文件的内容替换它，并将标题和所有记录写回文件。

我们采取的方法使得两个用户同时在保存CSV文件中工作是不安全的。创建允许多个用户编辑单个文件的软件是非常困难的，许多程序选择使用锁文件或其他保护机制来防止这种情况。

这个方法已经完成了，这就是我们需要在模型中进行的所有更改，以实现更新和查看。现在，是时候向我们的GUI添加必要的功能了。

# 实现记录列表视图

记录列表视图将允许我们的用户浏览文件的内容，并打开记录进行查看或编辑。我们的用户习惯于在电子表格中看到这些数据，以表格格式呈现，因此设计我们的视图以类似的方式是有意义的。由于我们的视图主要存在于查找和选择单个记录，我们不需要显示所有信息；只需要足够让用户区分一个记录和另一个记录。

快速分析表明我们需要CSV行号、`Date`、`Time`、`Lab`和`Plot`。 

对于构建具有可选择行的类似表格的视图，Tkinter为我们提供了ttk `Treeview`小部件。为了构建我们的记录列表视图，我们需要了解`Treeview`。

# ttk Treeview

`Treeview`是一个ttk小部件，设计用于以分层结构显示数据的列。

也许这种数据的最好例子是文件系统树：

+   每一行可以代表一个文件或目录

+   每个目录可以包含额外的文件或目录

+   每一行都可以有额外的数据属性，比如权限、大小或所有权信息

为了探索`Treeview`的工作原理，我们将借助`pathlib`创建一个简单的文件浏览器。

在之前的章节中，我们使用`os.path`来处理文件路径。`pathlib`是Python 3标准库的一个新添加，它提供了更面向对象的路径处理方法。

打开一个名为`treeview_demo.py`的新文件，并从这个模板开始：

```py
import tkinter as tk
from tkinter import ttk
from pathlib import Path

root = tk.Tk()
# Code will go here

root.mainloop()
```

我们将首先获取当前工作目录下所有文件路径的列表。`Path`有一个名为`glob`的方法，将给我们提供这样的列表，如下所示：

```py
paths = Path('.').glob('**/*')
```

`glob()`会对文件系统树扩展通配符字符，比如`*`和`?`。这个名称可以追溯到一个非常早期的Unix命令，尽管现在相同的通配符语法在大多数现代操作系统中都被使用。

`Path('.')` 创建一个引用当前工作目录的路径对象，`**/*` 是一个特殊的通配符语法，递归地抓取路径下的所有对象。结果是一个包含当前目录下每个目录和文件的`Path`对象列表。

完成后，我们可以通过执行以下代码来创建和配置我们的`Treeview`小部件：

```py
tv = ttk.Treeview(root, columns=['size', 'modified'], 
                  selectmode='None')
```

与任何Tkinter小部件一样，`Treeview`的第一个参数是它的`parent`小部件。`Treeview`小部件中的每一列都被赋予一个标识字符串；默认情况下，总是有一个名为`"#0"`的列。这一列代表树中每个项目的基本标识信息，比如名称或ID号。要添加更多列，我们使用`columns`参数来指定它们。这个列表包含任意数量的字符串，用于标识随后的列。

最后，我们设置`selectmode`，确定用户如何在树中选择项目。

以下表格显示了`selectmode`的选项：

| **Value** | **Behavior** |
| --- | --- |
| `selectmode` | 可以进行选择 |
| `none`（作为字符串，而不是`None`对象） | 不能进行选择 |
| `browse` | 用户只能选择一个项目 |
| `extended` | 用户可以选择多个项目 |

在这种情况下，我们正在阻止选择，所以将其设置为`none`。

为了展示我们如何使用列名，我们将为列设置一些标题：

```py
tv.heading('#0', text='Name')
tv.heading('size', text='Size', anchor='center')
tv.heading('modified', text='Modified', anchor='e')
```

`Treeview` heading方法用于操作列`heading`小部件；它接受列名，然后是要分配给列`heading`小部件的任意数量的属性。

这些属性可以包括：

+   `text`：标题显示的文本。默认情况下为空。

+   `anchor`：文本的对齐方式；可以是八个基本方向之一或`center`，指定为字符串或Tkinter常量。

+   `command`：单击标题时要运行的命令。这可能用于按该列对行进行排序，或选择该列中的所有值，例如。

+   `image`：要在标题中显示的图像。

最后，我们将列打包到`root`小部件中，并扩展它以填充小部件： 

```py
tv.pack(expand=True, fill='both')
```

除了配置标题之外，我们还可以使用`Treeview.column`方法配置列本身的一些属性。

例如，我们可以添加以下代码：

```py
tv.column('#0', stretch=True)
tv.column('size', width=200)
```

在此示例中，我们已将第一列中的`stretch`设置为`True`，这将导致它扩展以填充可用空间；我们还将`size`列上的`width`值设置为`200`像素。

可以设置的列参数包括：

+   `stretch`：是否将此列扩展以填充可用空间。

+   `width`：列的宽度，以像素为单位。

+   `minwidth`：列可以调整的最小宽度，以像素为单位。

+   `anchor`：列中文本的对齐方式。可以是八个基本方向或中心，指定为字符串或Tkinter常量。

树视图配置完成后，现在需要填充数据。使用`insert`方法逐行填充`Treeview`的数据。

`insert`方法如下所示：

```py
mytreeview.insert(parent, 'end', iid='item1',
          text='My Item 1', values=['12', '42'])
```

第一个参数指定插入行的`parent`项目。这不是`parent`小部件，而是层次结构中插入行所属的`parent`行。该值是一个字符串，指的是`parent`项目的`iid`。对于顶级项目，该值应为空字符串。

下一个参数指定应将项目插入的位置。它可以是数字索引或`end`，将项目放在列表末尾。

之后，我们可以指定关键字参数，包括：

+   `text`：这是要显示在第一列中的值。

+   `values`：这是剩余列的值列表。

+   `image`：这是要显示在列最左侧的图像对象。

+   `iid`：项目ID字符串。如果不指定，将自动分配。

+   `open`：行在开始时是否打开（显示子项）。

+   `tags`：标签字符串列表。

要将我们的路径插入`Treeview`，让我们按如下方式遍历我们的`paths`列表：

```py
for path in paths:
    meta = path.stat()
    parent = str(path.parent)
    if parent == '.':
        parent = ''
```

在调用`insert`之前，我们需要从路径对象中提取和准备一些数据。`path.stat()`将给我们一个包含各种文件信息的对象。`path.parent`提供了包含路径；但是，我们需要将`root`路径的名称（当前为单个点）更改为一个空字符串，这是`Treeview`表示`root`节点的方式。

现在，我们按如下方式添加`insert`调用：

```py
    tv.insert(parent, 'end', iid=str(path),
        text=str(path.name), values=[meta.st_size, meta.st_mtime])
```

通过使用路径字符串作为项目ID，我们可以将其指定为其子对象的父级。我们仅使用对象的`name`（不包含路径）作为我们的显示值，然后使用`st_size`和`st_mtime`来填充大小和修改时间列。

运行此脚本，您应该会看到一个简单的文件树浏览器，类似于这样：

![](assets/1ceeffaf-50a0-4c9e-97d3-525c1026367f.png)

`Treeview`小部件默认不提供任何排序功能，但我们可以相当容易地添加它。

首先，让我们通过添加以下代码创建一个排序函数：

```py
def sort(tv, col):
    itemlist = list(tv.get_children(''))
    itemlist.sort(key=lambda x: tv.set(x, col))
    for index, iid in enumerate(itemlist):
        tv.move(iid, tv.parent(iid), index)
```

在上述代码片段中，`sort`函数接受一个`Treeview`小部件和我们将对其进行排序的列的ID。它首先使用`Treeview`的`get_children()`方法获取所有`iid`值的列表。接下来，它使用`col`的值作为键对各种`iid`值进行排序；令人困惑的是，`Treeview`的`set()`方法用于检索列的值（没有`get()`方法）。最后，我们遍历列表，并使用`move()`方法将每个项目移动到其父级下的新索引（使用`parent()`方法检索）。

为了使我们的列可排序，使用`command`参数将此函数作为回调添加到标题中，如下所示：

```py
tv.heading('#0', text='Name', command=lambda: sort(tv, '#0'))
tv.heading('size', text='Size', anchor='center',
           command=lambda: sort(tv, 'size'))
tv.heading('modified', text='Modified', anchor='e',
           command=lambda: sort(tv, 'modified'))
```

# 使用`Treeview`实现我们的记录列表

现在我们了解了如何使用`Treeview`小部件，让我们开始构建我们的记录列表小部件。

我们将首先通过子类化`tkinter.Frame`来开始，就像我们在记录表单中所做的那样。

```py
class RecordList(tk.Frame):
    """Display for CSV file contents"""
```

为了节省一些重复的代码，我们将在类常量中定义我们的列属性和默认值。这也使得更容易调整它们以满足我们的需求。

使用以下属性开始你的类：

```py
    column_defs = {
        '#0': {'label': 'Row', 'anchor': tk.W},
        'Date': {'label': 'Date', 'width': 150, 'stretch': True},
        'Time': {'label': 'Time'},
        'Lab': {'label': 'Lab', 'width': 40},
        'Plot': {'label': 'Plot', 'width': 80}
        }
    default_width = 100
    default_minwidth = 10
    default_anchor = tk.CENTER
```

请记住，我们将显示`Date`，`Time`，`Lab`和`Plot`。对于第一个默认列，我们将显示CSV行号。我们还为一些列设置了`width`和`anchor`值，并配置了`Date`字段以进行拉伸。我们将在`__init__()`中配置`Treeview`小部件时使用这些值。

让我们从以下方式开始定义我们的`__init__()`：

```py
    def __init__(self, parent, callbacks, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.callbacks = callbacks
```

与其他视图一样，我们将从`Application`对象接受回调方法的字典，并将其保存为实例属性。

# 配置Treeview小部件

现在，通过执行以下代码片段来创建我们的`Treeview`小部件：

```py
        self.treeview = ttk.Treeview(self,
            columns=list(self.column_defs.keys())[1:],
            selectmode='browse')
```

请注意，我们正在从我们的`columns`列表中排除`＃0`列；它不应在这里指定，因为它会自动创建。我们还选择了`browse`选择模式，这样用户就可以选择CSV文件的单独行。

让我们继续将我们的`Treeview`小部件添加到`RecordList`并使其填充小部件：

```py
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.treeview.grid(row=0, column=0, sticky='NSEW')
```

现在，通过迭代`column_defs`字典来配置`Treeview`的列和标题：

```py
        for name, definition in self.column_defs.items():
```

对于每组项目，让我们按如下方式提取我们需要的配置值：

```py
            label = definition.get('label', '')
            anchor = definition.get('anchor', self.default_anchor)
            minwidth = definition.get(
                'minwidth', self.default_minwidth)
            width = definition.get('width', self.default_width)
            stretch = definition.get('stretch', False)
```

最后，我们将使用这些值来配置标题和列：

```py
            self.treeview.heading(name, text=label, anchor=anchor)
            self.treeview.column(name, anchor=anchor,
                minwidth=minwidth, width=width, stretch=stretch)
```

# 添加滚动条

ttk的`Treeview`默认没有滚动条；它*可以*使用键盘或鼠标滚轮控件进行滚动，但用户合理地期望在可滚动区域上有滚动条，以帮助他们可视化列表的大小和当前位置。

幸运的是，ttk为我们提供了一个可以连接到我们的`Treeview`小部件的`Scrollbar`对象：

```py
        self.scrollbar = ttk.Scrollbar(self,
            orient=tk.VERTICAL, command=self.treeview.yview)
```

在这里，`Scrollbar`接受以下两个重要参数：

+   `orient`：此参数确定是水平滚动还是垂直滚动

+   `command`：此参数为滚动条移动事件提供回调

在这种情况下，我们将回调设置为树视图的`yview`方法，该方法用于使`Treeview`上下滚动。另一个选项是`xview`，它将用于水平滚动。

我们还需要将我们的`Treeview`连接回滚动条：

```py
        self.treeview.configure(yscrollcommand=self.scrollbar.set)
```

如果我们不这样做，我们的`Scrollbar`将不知道我们已经滚动了多远或列表有多长，并且无法适当地设置滚动条小部件的大小或位置。

配置了我们的`Scrollbar`后，我们需要将其放置在小部件上——通常是在要滚动的小部件的右侧。

我们可以使用我们的`grid`布局管理器来实现这一点：

```py
        self.scrollbar.grid(row=0, column=1, sticky='NSW')
```

请注意，我们将`sticky`设置为north、south和west。north和south确保滚动条拉伸到小部件的整个高度，west确保它紧贴着`Treeview`小部件的左侧。

# 填充Treeview

现在我们有了`Treeview`小部件，我们将创建一个`populate()`方法来填充它的数据：

```py
    def populate(self, rows):
        """Clear the treeview & write the supplied data rows to it."""
```

`rows`参数将接受`dict`数据类型的列表，例如从`model`返回的类型。其想法是控制器将从模型中获取一个列表，然后将其传递给此方法。

在重新填充`Treeview`之前，我们需要清空它：

```py
        for row in self.treeview.get_children():
            self.treeview.delete(row)
```

`Treeview`的`get_children()`方法返回每行的`iid`列表。我们正在迭代此列表，将每个`iid`传递给`Treeview.delete()`方法，正如您所期望的那样，删除该行。

清除`Treeview`后，我们可以遍历`rows`列表并填充表格：

```py
        valuekeys = list(self.column_defs.keys())[1:]
        for rownum, rowdata in enumerate(rows):
            values = [rowdata[key] for key in valuekeys]
            self.treeview.insert('', 'end', iid=str(rownum),
                                 text=str(rownum), values=values)
```

我们在这里要做的第一件事是创建一个我们实际想要从每一行获取的所有键的列表；这只是从`self.column_defs`减去`＃0`列的键列表。

接下来，我们使用 `enumerate()` 函数迭代行以生成行号。对于每一行，我们将使用列表推导创建正确顺序的值列表，然后使用 `insert()` 方法将列表插入到 `Treeview` 小部件的末尾。请注意，我们只是将行号（转换为字符串）用作行的 `iid` 和 `text`。

在这个函数中我们需要做的最后一件事是进行一些小的可用性调整。为了使我们的 `Treeview` 对键盘友好，我们需要将焦点放在第一项上，这样键盘用户就可以立即开始使用箭头键进行导航。

在 `Treeview` 小部件中实际上需要三个方法调用，如下所示：

```py
        if len(rows) > 0:
            self.treeview.focus_set()
            self.treeview.selection_set(0)
            self.treeview.focus('0')
```

首先，`focus_set` 将焦点移动到 `Treeview`。接下来，`selection_set(0)` 选择列表中的第一条记录。最后，`focus('0')` 将焦点放在 `iid` 为 `0` 的行上。当然，我们只在有任何行的情况下才这样做。

# 响应记录选择

这个小部件的目的是让用户选择和打开记录；因此，我们需要一种方法来做到这一点。最好能够从双击或键盘选择等事件触发这一点。

`Treeview` 小部件有三个特殊事件，我们可以使用它们来触发回调，如下表所示：

| **事件字符串** | **触发时** |
| --- | --- |
| `<<TreeviewSelect>>` | 选择行，例如通过鼠标点击 |
| `<<TreeviewOpen>>` | 通过双击或选择并按 *Enter* 打开行 |
| `<<TreeviewClose>>` | 关闭打开的行 |

`<<TreeviewOpen>>` 听起来像我们想要的事件；即使我们没有使用分层列表，用户仍然在概念上打开记录，并且触发动作（双击）似乎很直观。我们将将此事件绑定到一个方法，该方法将打开所选记录。

将此代码添加到 `__init__()` 的末尾：

```py
        self.treeview.bind('<<TreeviewOpen>>', self.on_open_record)
```

`on_open_record()` 方法非常简单；将此代码添加到类中：

```py
    def on_open_record(self, *args):
        selected_id = self.treeview.selection()[0]
        self.callbacks['on_open_record'](selected_id)
```

只需从 `Treeview` 中检索所选 ID，然后使用控制器中的 `callbacks` 字典提供的函数调用所选 ID。这将由控制器来做一些适当的事情。

`RecordList` 类现在已经完成，但是我们的其他视图类需要注意。

# 修改记录表单以进行读取和更新

只要我们在编辑视图，我们就需要查看我们的 `DataRecordForm` 视图，并调整它以使其能够更新记录。

花点时间考虑一下我们需要进行的以下更改：

+   表单将需要一种方式来加载控制器提供的记录。

+   表单将需要跟踪它正在编辑的记录，或者是否是新记录。

+   我们的用户需要一些视觉指示来指示正在编辑的记录。

+   我们的保存按钮当前在应用程序中。它在表单之外没有任何意义，因此它可能应该是表单的一部分。

+   这意味着我们的表单将需要一个在单击保存按钮时调用的回调。我们需要像我们的其他视图一样为它提供一个 `callbacks` 字典。

# 更新 `__init__()`

让我们从我们的 `__init__()` 方法开始逐步进行这些工作：

```py
    def __init__(self, parent, fields, 
                 settings, callbacks, *args, **kwargs):
        self.callbacks = callbacks
```

我们正在添加一个新的参数 `callbacks`，并将其存储为实例属性。这将为控制器提供一种方法来提供视图调用的方法。

接下来，我们的 `__init__()` 方法应该设置一个变量来存储当前记录：

```py
        self.current_record = None
```

我们将使用 `None` 来指示没有加载记录，表单正在用于创建新记录。否则，这个值将是一个引用 CSV 数据中行的整数。

我们可以在这里使用一个 Tkinter 变量，但在这种情况下没有真正的优势，而且我们将无法使用 `None` 作为值。

在表单顶部，在第一个表单字段之前，让我们添加一个标签，用于跟踪我们正在编辑的记录：

```py
        self.record_label = ttk.Label()
        self.record_label.grid(row=0, column=0)
```

我们将其放在第`0`行，第`0`列，但第一个`LabelFrame`也在那个位置。您需要逐个检查每个`LabelFrame`，并在其对`grid`的调用中递增`row`值。

我们将确保每当记录加载到表单中时，此标签都会得到更新。

在小部件的最后，`Notes`字段之后，让我们添加我们的新保存按钮如下：

```py
        self.savebutton = ttk.Button(self,
            text="Save", command=self.callbacks["on_save"])
        self.savebutton.grid(sticky="e", row=5, padx=10)
```

当点击按钮时，按钮将调用`callbacks`字典中的`on_save()`方法。在`Application`中创建`DataRecordForm`时，我们需要确保提供这个方法。

# 添加`load_record()`方法

在我们的视图中添加的最后一件事是加载新记录的方法。这个方法需要使用控制器中给定的行号和数据字典设置我们的表单。

让我们将其命名为`load_record()`如下：

```py
    def load_record(self, rownum, data=None):
```

我们应该首先从提供的`rownum`设置表单的`current_record`值：

```py
        self.current_record = rownum
```

回想一下，`rownum`可能是`None`，表示这是一个新记录。

让我们通过执行以下代码来检查：

```py
        if rownum is None:
            self.reset()
            self.record_label.config(text='New Record')
```

如果我们要插入新记录，我们只需重置表单，然后将标签设置为指示这是新记录。

请注意，这里的`if`条件专门检查`rownum`是否为`None`；我们不能只检查`rownum`的真值，因为`0`是一个有效的用于更新的`rownum`！

如果我们有一个有效的`rownum`，我们需要让它表现得不同：

```py
        else:
            self.record_label.config(text='Record #{}'.format(rownum))
            for key, widget in self.inputs.items():
                self.inputs[key].set(data.get(key, ''))
                try:
                    widget.input.trigger_focusout_validation()
                except AttributeError:
                    pass
```

在这个块中，我们首先使用正在编辑的行号适当地设置标签。

然后，我们循环遍历`inputs`字典的键和小部件，并从`data`字典中提取匹配的值。我们还尝试在每个小部件的输入上调用`trigger_focusout_validation()`方法，因为CSV文件可能包含无效数据。如果输入没有这样的方法（也就是说，如果我们使用的是常规的Tkinter小部件而不是我们的自定义小部件之一，比如`Checkbutton`），我们就什么也不做。

# 更新应用程序的其余部分

在我们对表单进行更改生效之前，我们需要更新应用程序的其余部分以实现新功能。我们的主菜单需要一些导航项，以便用户可以在记录列表和表单之间切换，并且需要在`Application`中创建或更新控制器方法，以整合我们的新模型和视图功能。

# 主菜单更改

由于我们已经在`views.py`文件中，让我们首先通过一些命令来在我们的主菜单视图中切换记录列表和记录表单。我们将在我们的菜单中添加一个`Go`菜单，其中包含两个选项，允许在记录列表和空白记录表单之间切换。

在`Options`和`Help`菜单之间添加以下行：

```py
        go_menu = tk.Menu(self, tearoff=False)
        go_menu.add_command(label="Record List",
                         command=callbacks['show_recordlist'])
        go_menu.add_command(label="New Record",
                         command=callbacks['new_record'])
        self.add_cascade(label='Go', menu=go_menu)
```

与以前一样，我们将这些菜单命令绑定到`callbacks`字典中的函数，我们需要在`Application`类中添加这些函数。

# 在应用程序中连接各部分

让我们快速盘点一下我们需要在`Application`类中进行的以下更改：

+   我们需要添加一个`RecordList`视图的实例

+   我们需要更新我们对`CSVModel`的使用，以便可以从中访问数据

+   我们需要实现或重构视图使用的几个回调方法

# 添加`RecordList`视图

我们将在`__init__()`中创建`RecordList`对象，就在`DataRecordForm`之后，通过执行以下代码片段：

```py
        self.recordlist = v.RecordList(self, self.callbacks)
        self.recordlist.grid(row=1, padx=10, sticky='NSEW')
```

请注意，当我们调用`grid()`时，我们将`RecordList`视图添加到已经包含`DataRecordForm`的网格单元中。**这是有意的**。当我们这样做时，Tkinter会将第二个小部件堆叠在第一个小部件上，就像将一张纸放在另一张纸上一样；我们将在稍后添加代码来控制哪个视图可见，通过将其中一个提升到堆栈的顶部。请注意，我们还将小部件粘贴到单元格的所有边缘。如果没有这段代码，一个小部件的一部分可能会在另一个小部件的后面可见。

类似地，我们需要更新记录表单的`grid`调用如下：

```py
        self.recordform.grid(row=1, padx=10, sticky='NSEW')
```

# 移动模型

目前，我们的数据模型对象仅在`on_save()`方法中创建，并且每次用户保存时都会重新创建。我们将要编写的其他一些回调函数也需要访问模型，因此我们将在`Application`类启动或选择新文件时创建一个可以由所有方法共享的单个数据模型实例。让我们看看以下步骤：

1.  首先，在创建`default_filename`后编辑`Application.__init__()`方法：

```py
        self.filename = tk.StringVar(value=default_filename)
        self.data_model = m.CSVModel(filename=self.filename.get())
```

1.  接下来，每当文件名更改时，`on_file_select()`方法需要重新创建`data_model`对象。

1.  将`on_file_select()`的结尾更改为以下代码：

```py
        if filename:
            self.filename.set(filename)
            self.data_model = m.CSVModel(filename=self.filename.get())
```

现在，`self.data_model`将始终指向当前数据模型，我们的所有方法都可以使用它来保存或读取数据。

# 填充记录列表

`Treeview`小部件已添加到我们的应用程序中，但我们需要一种方法来用数据填充它。

我们将通过执行以下代码创建一个名为`populate_recordlist()`的方法：

```py
    def populate_recordlist(self):
```

逻辑很简单：只需从模型中获取所有行并将它们发送到记录列表的`populate()`方法。

我们可以简单地写成这样：

```py
        rows = self.data_model.get_all_records()
        self.recordlist.populate(rows)
```

但要记住，如果文件出现问题，`get_all_records()`将引发一个`Exception`；我们需要捕获该异常并让用户知道出了问题。

使用以下代码更新代码：

```py
        try:
            rows = self.data_model.get_all_records()
        except Exception as e:
            messagebox.showerror(title='Error',
                message='Problem reading file',
                detail=str(e))
        else:
            self.recordlist.populate(rows)
```

在这种情况下，如果我们从`get_all_records()`获得异常，我们将显示一个显示`Exception`文本的错误对话框。

`RecordList`视图应在创建新模型时重新填充；目前，这在`Application.__init__()`和`Application.on_file_select()`中发生。

在创建记录列表后立即更新`__init__()`：

```py
        self.recordlist = v.RecordList(self, self.callbacks)
        self.recordlist.grid(row=1, padx=10, sticky='NSEW')
        self.populate_recordlist()
```

在`if filename:`块的最后，更新`on_file_select()`如下：

```py
        if filename:
            self.filename.set(filename)
            self.data_model = m.CSVModel(filename=self.filename.get())
            self.populate_recordlist()
```

# 添加新的回调函数

检查我们的视图代码，以下回调函数需要添加到我们的`callbacks`字典中：

+   `show_recordlist()`：当用户点击菜单中的记录列表选项时调用此函数，它应该导致记录列表可见

+   `new_record()`：当用户点击菜单中的新记录时调用此函数，它应该显示一个重置的`DataRecordForm`

+   `on_open_record()`：当打开记录列表项时调用此函数，它应该显示填充有记录ID和数据的`DataRecordForm`

+   `on_save()`：当点击保存按钮（现在是`DataRecordForm`的一部分）时调用此函数，它应该导致记录表单中的数据被更新或插入模型。

我们将从`show_recordlist()`开始：

```py
    def show_recordlist(self):
        """Show the recordform"""
        self.recordlist.tkraise()
```

记住，当我们布置主应用程序时，我们将`recordlist`叠放在数据输入表单上，以便一个遮挡另一个。`tkraise()`方法可以在任何Tkinter小部件上调用，将其提升到小部件堆栈的顶部。在这里调用它将使我们的`RecordList`小部件升至顶部并遮挡数据输入表单。

不要忘记将以下内容添加到`callbacks`字典中：

```py
        self.callbacks = {
             'show_recordlist': self.show_recordlist,
             ...
```

`new_record()`和`on_open_record()`回调都会导致`recordform`被显示；一个在没有行号的情况下调用，另一个在有行号的情况下调用。我们可以在一个方法中轻松地回答这两个问题。

让我们称这个方法为`open_record()`：

```py
    def open_record(self, rownum=None):
```

记住我们的`DataRecordForm.load_record()`方法需要一个行号和一个`data`字典，如果行号是`None`，它会重置表单以进行新记录。所以，我们只需要设置行号和记录，然后将它们传递给`load_record()`方法。

首先，我们将处理`rownum`为`None`的情况：

```py
        if rownum is None:
            record = None
```

没有行号，就没有记录。很简单。

现在，如果有行号，我们需要尝试从模型中获取该行并将其用于`record`：

```py
        else:
            rownum = int(rownum)
            record = self.data_model.get_record(rownum)
```

请注意，Tkinter可能会将`rownum`作为字符串传递，因为`Treeview`的`iid`值是字符串。我们将进行安全转换为`int`，因为这是我们的模型所期望的。

记住，如果在读取文件时出现问题，模型会抛出`Exception`，所以我们应该捕获这个异常。

将`get_record()`的调用放在`try`块中：

```py
        try:
            record = self.data_model.get_record(rownum)
        except Exception as e:
            messagebox.showerror(title='Error',
                message='Problem reading file',
                detail=str(e))
            return
```

在出现`Exception`的情况下，我们将显示一个错误对话框，并在不改变任何内容的情况下从函数中返回。

有了正确设置的`rownum`和`record`，现在我们可以将它们传递给`DataRecordForm`：

```py
        self.recordform.load_record(rownum, record)
```

最后，我们需要提升`form`小部件，使其位于记录列表的顶部：

```py
        self.recordform.tkraise()
```

现在，我们可以更新我们的`callbacks`字典，将这些键指向新的方法：

```py
        self.callbacks = {
            'new_record': self.open_record,
            'on_open_record': self.open_record,
            ...
```

你可以说我们不应该在这里有相同的方法，而只是让我们的视图拉取相同的键；然而，让视图在语义上引用回调是有意义的——也就是说，根据它们打算实现的目标，而不是它是如何实现的——然后让控制器确定哪段代码最符合这个语义需求。如果在某个时候，我们需要将这些分成两个方法，我们只需要在`Application`中做这个操作。

我们已经有了一个`on_save()`方法，所以将其添加到我们的回调中就足够简单了：

```py
        self.callbacks = {
            ...
            'on_save': self.on_save
        }
```

然而，我们当前的`on_save()`方法只处理插入新记录。我们需要修复这个问题。

首先，我们可以删除获取文件名和创建模型的两行，因为我们可以直接使用`Application`对象的`data_model`属性。

现在，用以下内容替换下面的两行：

```py
        data = self.recordform.get()
        rownum = self.recordform.current_record
        try:
            self.data_model.save_record(data, rownum)
```

我们只需要从`DataRecordForm`中获取数据和当前记录，然后将它们传递给模型的`save_record()`方法。记住，如果我们发送`None`的`rownum`，模型将插入一个新记录；否则，它将更新该行号的记录。

因为`save_record()`可能会抛出几种不同的异常，所以它在这里是在一个`try`块下面。

首先，如果我们尝试更新一个不存在的行号，我们会得到`IndexError`，所以让我们捕获它：

```py
        except IndexError as e:
            messagebox.showerror(title='Error',
                message='Invalid row specified', detail=str(e))
            self.status.set('Tried to update invalid row')
```

在出现问题的情况下，我们将显示一个错误对话框并更新状态文本。

`save_record()`方法也可能会抛出一个通用的`Exception`，因为它调用了模型的`get_all_records()`方法。

我们也会捕获这个异常，并显示一个适当的错误：

```py
        except Exception as e:
            messagebox.showerror(title='Error',
                message='Problem saving record', detail=str(e))
            self.status.set('Problem saving record')
```

这个方法中剩下的代码只有在没有抛出异常时才应该运行，所以将它移动到一个`else`块下面：

```py
    else:
        self.records_saved += 1
        self.status.set(
            "{} records saved this session".format(self.records_saved)
        )
        self.recordform.reset()
```

由于插入或更新记录通常会导致记录列表的更改，所以在成功保存文件后，我们还应该重新填充记录列表。

在`if`块下面添加以下行：

```py
            self.populate_recordlist()
```

最后，我们只想在插入新文件时重置记录表单；如果不是，我们应该什么都不做。

将对`recordform.reset()`的调用放在一个`if`块下面：

```py
            if self.recordform.current_record is None:
                self.recordform.reset()
```

# 清理

在退出`application.py`之前，确保删除保存按钮的代码，因为我们已经将该UI部分移动到`DataRecordForm`中。

在`__init__()`中查找这些行并删除它们：

```py
        self.savebutton = ttk.Button(self, text="Save",
                                     command=self.on_save)
        self.savebutton.grid(sticky="e", row=2, padx=10)
```

你还可以将`statusbar`的位置上移一行：

```py
        self.statusbar.grid(sticky="we", row=2, padx=10)
```

# 测试我们的程序

此时，您应该能够运行应用程序并加载一个示例CSV文件，如下面的截图所示：

![](assets/3b8b74a3-4a1c-4b7d-b269-4110be95eabf.png)

确保尝试打开记录，编辑和保存它，以及插入新记录和打开不同的文件。

你还应该测试以下错误条件：

+   尝试打开一个不是CSV文件的文件，或者一个带有不正确字段的CSV文件。会发生什么？

+   打开一个有效的CSV文件，选择一个记录进行编辑，然后在点击保存之前，选择一个不同的或空文件。会发生什么？

+   打开两个程序的副本，并将它们指向保存的CSV文件。尝试在程序之间交替编辑或更新操作。注意发生了什么。

# 摘要

我们已经将我们的程序从仅追加的形式改变为能够从现有文件加载、查看和更新数据的应用程序。您学会了如何制作读写模型，如何使用ttk `Treeview`，以及如何修改现有的视图和控制器来读取和更新记录。

在我们的下一章中，我们将学习如何修改应用程序的外观和感觉。我们将学习如何使用小部件属性、样式和主题，以及如何使用位图图形。
