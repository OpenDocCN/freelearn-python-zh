# 使用模型-视图类创建数据接口

绝大多数应用软件都是用来查看和操作组织好的数据。即使在不是显式*数据库应用程序*的应用程序中，通常也需要以较小的规模与数据集进行交互，比如用选项填充组合框或显示一系列设置。如果没有某种组织范式，GUI 和一组数据之间的交互很快就会变成一团乱麻的代码噩梦。**模型-视图**模式就是这样一种范式。

在本章中，我们将学习如何使用 Qt 的模型-视图小部件以及如何在应用程序中优雅地处理数据。我们将涵盖以下主题：

+   理解模型-视图设计

+   PyQt 中的模型和视图

+   构建一个**逗号分隔值**（**CSV**）编辑器

# 技术要求

本章具有与前几章相同的技术要求。您可能还希望从[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter05`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter05)获取示例代码。

您还需要一个或两个 CSV 文件来使用我们的 CSV 编辑器。这些可以在任何电子表格程序中制作，并且应该以列标题作为第一行创建。

查看以下视频，看看代码是如何运行的：[`bit.ly/2M66bnv`](http://bit.ly/2M66bnv)

# 理解模型-视图设计

模型-视图是一种实现**关注点分离**的软件应用设计范式。它基于古老的**模型-视图-控制器**（**MVC**）模式，但不同之处在于控制器和视图被合并成一个组件。

在模型-视图设计中，**模型**是保存应用程序数据并包含检索、存储和操作数据逻辑的组件。**视图**组件向用户呈现数据，并提供输入和操作数据的界面。通过将应用程序的这些组件分离，我们将它们的相互依赖性降到最低，使它们更容易重用或重构。

让我们通过一个简单的例子来说明这个过程。从第四章的应用程序模板开始，*使用 QMainWindow 构建应用程序*，让我们构建一个简单的文本文件编辑器：

```py
    # This code goes in MainWindow.__init__()
    form = qtw.QWidget()
    self.setCentralWidget(form)
    form.setLayout(qtw.QVBoxLayout())
    self.filename = qtw.QLineEdit()
    self.filecontent = qtw.QTextEdit()
    self.savebutton = qtw.QPushButton(
      'Save',
      clicked=self.save
    )

    form.layout().addWidget(self.filename)
    form.layout().addWidget(self.filecontent)
    form.layout().addWidget(self.savebutton)
```

这是一个简单的表单，包括一个用于文件名的行编辑，一个用于内容的文本编辑和一个调用`save()`方法的保存按钮。

让我们创建以下`save()`方法：

```py
  def save(self):
    filename = self.filename.text()
    error = ''
    if not filename:
      error = 'Filename empty'
    elif path.exists(filename):
      error = f'Will not overwrite {filename}'
    else:
      try:
        with open(filename, 'w') as fh:
          fh.write(self.filecontent.toPlainText())
      except Exception as e:
        error = f'Cannot write file: {e}'
    if error:
      qtw.QMessageBox.critical(None, 'Error', error)
```

这种方法检查是否在行编辑中输入了文件名，确保文件名不存在（这样你就不会在测试这段代码时覆盖重要文件！），然后尝试保存它。如果出现任何错误，该方法将显示一个`QMessageBox`实例来报告错误。

这个应用程序可以工作，但缺乏清晰的模型和视图分离。将文件写入磁盘的同一个方法也显示错误框并调用输入小部件方法。如果我们要扩展这个应用程序到任何程度，`save()`方法很快就会变成一个混合了数据处理逻辑和呈现逻辑的迷宫。

让我们用单独的`Model`和`View`类重写这个应用程序。

从应用程序模板的干净副本开始，让我们创建我们的`Model`类：

```py
class Model(qtc.QObject):

  error = qtc.pyqtSignal(str)

  def save(self, filename, content):
    print("save_called")
    error = ''
    if not filename:
      error = 'Filename empty'
    elif path.exists(filename):
      error = f'Will not overwrite {filename}'
    else:
      try:
        with open(filename, 'w') as fh:
          fh.write(content)
      except Exception as e:
        error = f'Cannot write file: {e}'
    if error:
      self.error.emit(error)
```

我们通过子类化`QObject`来构建我们的模型。模型不应参与显示 GUI，因此不需要基于`QWidget`类。然而，由于模型将使用信号和槽进行通信，我们使用`QObject`作为基类。模型实现了我们在前面示例中的`save()`方法，但有两个变化：

+   首先，它期望用户数据作为参数传入，不知道这些数据来自哪些小部件

+   其次，当遇到错误时，它仅仅发出一个 Qt 信号，而不采取任何特定于 GUI 的操作

接下来，让我们创建我们的`View`类：

```py
class View(qtw.QWidget):

  submitted = qtc.pyqtSignal(str, str)

  def __init__(self):
    super().__init__()
    self.setLayout(qtw.QVBoxLayout())
    self.filename = qtw.QLineEdit()
    self.filecontent = qtw.QTextEdit()
    self.savebutton = qtw.QPushButton(
      'Save',
      clicked=self.submit
    )
    self.layout().addWidget(self.filename)
    self.layout().addWidget(self.filecontent)
    self.layout().addWidget(self.savebutton)

  def submit(self):
    filename = self.filename.text()
    filecontent = self.filecontent.toPlainText()
    self.submitted.emit(filename, filecontent)

  def show_error(self, error):
    qtw.QMessageBox.critical(None, 'Error', error)
```

这个类包含与之前相同的字段和字段布局定义。然而，这一次，我们的保存按钮不再调用`save()`，而是连接到一个`submit()`回调，该回调收集表单数据并使用信号发射它。我们还添加了一个`show_error()`方法来显示错误。

在我们的`MainWindow.__init__()`方法中，我们将模型和视图结合在一起：

```py
    self.view = View()
    self.setCentralWidget(self.view)

    self.model = Model()

    self.view.submitted.connect(self.model.save)
    self.model.error.connect(self.view.show_error)
```

在这里，我们创建`View`类的一个实例和`Model`类，并连接它们的信号和插槽。

在这一点上，我们的代码的模型视图版本的工作方式与我们的原始版本完全相同，但涉及更多的代码。你可能会问，这有什么意义？如果这个应用程序注定永远不会超出它现在的状态，那可能没有意义。然而，应用程序往往会在功能上扩展，并且通常其他应用程序需要重用相同的代码。考虑以下情况：

+   你想提供另一种编辑形式，也许是基于控制台的，或者具有更多的编辑功能

+   你想提供将内容保存到数据库而不是文本文件的选项

+   你正在创建另一个也将文本内容保存到文件的应用程序

在这些情况下，使用模型视图模式意味着我们不必从头开始。例如，在第一种情况下，我们不需要重写任何保存文件的代码；我们只需要创建用户界面代码，发射相同的`submitted`信号。随着你的代码扩展和你的应用程序变得更加复杂，这种关注点的分离将帮助你保持秩序。

# PyQt 中的模型和视图

模型视图模式不仅在设计大型应用程序时有用，而且在包含数据的小部件上也同样有用。从第四章中复制应用程序模板，*使用 QMainWindow 构建应用程序*，让我们看一个模型视图在小部件级别上是如何工作的简单示例。

在`MainWindow`类中，创建一个项目列表，并将它们添加到`QListWidget`和`QComboBox`对象中：

```py
    data = [
      'Hamburger', 'Cheeseburger',
      'Chicken Nuggets', 'Hot Dog', 'Fish Sandwich'
    ]
    # The list widget
    listwidget = qtw.QListWidget()
    listwidget.addItems(data)
    # The combobox
    combobox = qtw.QComboBox()
    combobox.addItems(data)
    self.layout().addWidget(listwidget)
    self.layout().addWidget(combobox)
```

因为这两个小部件都是用相同的列表初始化的，所以它们都包含相同的项目。现在，让我们使列表小部件的项目可编辑：

```py
    for i in range(listwidget.count()):
      item = listwidget.item(i)
      item.setFlags(item.flags() | qtc.Qt.ItemIsEditable)
```

通过迭代列表小部件中的项目，并在每个项目上设置`Qt.ItemIsEditable`标志，小部件变得可编辑，我们可以改变项目的文本。运行应用程序，尝试编辑列表小部件中的项目。即使你改变了列表小部件中的项目，组合框中的项目仍然保持不变。每个小部件都有自己的内部列表模型，它存储了最初传入的项目的副本。在一个列表的副本中改变项目对另一个副本没有影响。

我们如何保持这两个列表同步？我们可以连接一些信号和插槽，或者添加类方法来做到这一点，但 Qt 提供了更好的方法。

`QListWidget`实际上是另外两个 Qt 类的组合：`QListView`和`QStringListModel`。正如名称所示，这些都是模型视图类。我们可以直接使用这些类来构建我们自己的带有离散模型和视图的列表小部件：

```py
    model = qtc.QStringListModel(data)
    listview = qtw.QListView()
    listview.setModel(model)
```

我们简单地创建我们的模型类，用我们的字符串列表初始化它，然后创建视图类。最后，我们使用视图的`setModel()`方法连接两者。

`QComboBox`没有类似的模型视图类，但它仍然在内部是一个模型视图小部件，并且具有使用外部模型的能力。

因此，我们可以使用`setModel()`将我们的`QStringListModel`传递给它：

```py
    model_combobox = qtw.QComboBox()
    model_combobox.setModel(model)
```

将这些小部件添加到布局中，然后再次运行程序。这一次，你会发现对`QListView`的编辑立即在组合框中可用，因为你所做的更改被写入了`QStringModel`对象，这两个小部件都会查询项目数据。

`QTableWidget`和`QTreeWidget`也有类似的视图类：`QTableView`和`QTreeView`。然而，没有现成的模型类可以与这些视图一起使用。相反，我们必须通过分别继承`QAbstractTableModel`和`QAbstractTreeModel`来创建自己的自定义模型类。

在下一节中，我们将通过构建自己的 CSV 编辑器来介绍如何创建和使用自定义模型类。

# 构建 CSV 编辑器

逗号分隔值（CSV）是一种存储表格数据的纯文本格式。任何电子表格程序都可以导出为 CSV，或者您可以在文本编辑器中手动创建。我们的程序将被设计成可以打开任意的 CSV 文件并在`QTableView`中显示数据。通常在 CSV 的第一行用于保存列标题，因此我们的应用程序将假定这一点并使该行不可变。

# 创建表格模型

在开发数据驱动的模型-视图应用程序时，模型通常是最好的起点，因为这里是最复杂的代码。一旦我们把这个后端放在适当的位置，实现前端就相当简单了。

在这种情况下，我们需要设计一个可以读取和写入 CSV 数据的模型。从第四章的应用程序模板中复制应用程序模板，*使用* *QMainWindow*，并在顶部添加 Python `csv`库的导入。

现在，让我们通过继承`QAbstractTableModel`来开始构建我们的模型：

```py
class CsvTableModel(qtc.QAbstractTableModel):
  """The model for a CSV table."""

  def __init__(self, csv_file):
    super().__init__()
    self.filename = csv_file
    with open(self.filename) as fh:
      csvreader = csv.reader(fh)
      self._headers = next(csvreader)
      self._data = list(csvreader)
```

我们的模型将以 CSV 文件的名称作为参数，并立即打开文件并将其读入内存（对于大文件来说不是一个很好的策略，但这只是一个示例程序）。我们将假定第一行是标题行，并在将其余行放入模型的`_data`属性之前使用`next()`函数检索它。

# 实现读取功能

为了创建我们的模型的实例以在视图中显示数据，我们需要实现三种方法：

+   `rowCount()`，必须返回表中的总行数

+   `columnCount()`，必须返回表中的总列数

+   `data()`用于从模型请求数据

在这种情况下，`rowCount()`和`columnCount()`都很容易：

```py
  def rowCount(self, parent):
    return len(self._data)

  def columnCount(self, parent):
    return len(self._headers)
```

行数只是`_data`属性的长度，列数可以通过获取`_headers`属性的长度来获得。这两个函数都需要一个`parent`参数，但在这种情况下，它没有被使用，因为它是指父节点，只有在分层数据中才适用。

最后一个必需的方法是`data()`，需要更多解释；`data()`看起来像这样：

```py
  def data(self, index, role):
    if role == qtc.Qt.DisplayRole:
      return self._data[index.row()][index.column()]
```

`data()`的目的是根据`index`和`role`参数返回表格中单个单元格的数据。现在，`index`是`QModelIndex`类的一个实例，它描述了列表、表格或树结构中单个节点的位置。每个`QModelIndex`包含以下属性：

+   `row`号

+   `column`号

+   `parent`模型索引

在我们这种表格模型的情况下，我们对`row`和`column`属性感兴趣，它们指示我们想要的数据单元的表行和列。如果我们处理分层数据，我们还需要`parent`属性，它将是父节点的索引。如果这是一个列表，我们只关心`row`。

`role`是`QtCore.Qt.ItemDataRole`枚举中的一个常量。当视图从模型请求数据时，它传递一个`role`值，以便模型可以返回适合请求上下文的数据或元数据。例如，如果视图使用`EditRole`角色进行请求，模型应返回适合编辑的数据。如果视图使用`DecorationRole`角色进行请求，模型应返回适合单元格的图标。

如果没有特定角色的数据需要返回，`data()`应该返回空。

在这种情况下，我们只对`DisplayRole`角色感兴趣。要实际返回数据，我们需要获取索引的行和列，然后使用它来从我们的 CSV 数据中提取适当的行和列。

在这一点上，我们有一个最小功能的只读 CSV 模型，但我们可以添加更多内容。

# 添加标题和排序

能够返回数据只是模型功能的一部分。模型还需要能够提供其他信息，例如列标题的名称或排序数据的适当方法。

要在我们的模型中实现标题数据，我们需要创建一个`headerData()`方法：

```py
  def headerData(self, section, orientation, role):

    if (
      orientation == qtc.Qt.Horizontal and
      role == qtc.Qt.DisplayRole
    ):
      return self._headers[section]
    else:
      return super().headerData(section, orientation, role)
```

`headerData()`根据三个信息——**section**、**orientation**和**role**返回单个标题的数据。

标题可以是垂直的或水平的，由方向参数确定，该参数指定为`QtCore.Qt.Horizontal`或`QtCore.Qt.Vertical`常量。

该部分是一个整数，指示列号（对于水平标题）或行号（对于垂直标题）。

如`data()`方法中的角色参数一样，指示需要返回数据的上下文。

在我们的情况下，我们只对`DisplayRole`角色显示水平标题。与`data()`方法不同，父类方法具有一些默认逻辑和返回值，因此在任何其他情况下，我们希望返回`super().headerData()`的结果。

如果我们想要对数据进行排序，我们需要实现一个`sort()`方法，它看起来像这样：

```py
  def sort(self, column, order):
    self.layoutAboutToBeChanged.emit() # needs to be emitted before a sort
    self._data.sort(key=lambda x: x[column])
    if order == qtc.Qt.DescendingOrder:
      self._data.reverse()
    self.layoutChanged.emit() # needs to be emitted after a sort
```

`sort()`接受一个`column`号和`order`，它可以是`QtCore.Qt.DescendingOrder`或`QtCore.Qt.AscendingOrder`，该方法的目的是相应地对数据进行排序。在这种情况下，我们使用 Python 的`list.sort()`方法来就地对数据进行排序，使用`column`参数来确定每行的哪一列将被返回进行排序。如果请求降序排序，我们将使用`reverse()`来相应地改变排序顺序。

`sort()`还必须发出两个信号：

+   在内部进行任何排序之前，必须发出`layoutAboutToBeChanged`信号。

+   在排序完成后，必须发出`layoutChanged`信号。

这两个信号被视图用来适当地重绘自己，因此重要的是要记得发出它们。

# 实现写入功能

我们的模型目前是只读的，但因为我们正在实现 CSV 编辑器，我们需要实现写入数据。首先，我们需要重写一些方法以启用对现有数据行的编辑：`flags()`和`setData()`。

`flags()`接受一个`QModelIndex`值，并为给定索引处的项目返回一组`QtCore.Qt.ItemFlag`常量。这些标志用于指示项目是否可以被选择、拖放、检查，或者——对我们来说最有趣的是——编辑。

我们的方法如下：

```py
  def flags(self, index):
    return super().flags(index) | qtc.Qt.ItemIsEditable
```

在这里，我们将`ItemIsEditable`标志添加到父类`flags()`方法返回的标志列表中，指示该项目是可编辑的。如果我们想要实现逻辑，在某些条件下只使某些单元格可编辑，我们可以在这个方法中实现。

例如，如果我们有一个存储在`self.readonly_indexes`中的只读索引列表，我们可以编写以下方法：

```py
  def flags(self, index):
    if index not in self.readonly_indexes:
      return super().flags(index) | qtc.Qt.ItemIsEditable
    else:
      return super().flags(index)
```

然而，对于我们的应用程序，我们希望每个单元格都是可编辑的。

现在模型中的所有项目都标记为可编辑，我们需要告诉我们的模型如何实际编辑它们。这在`setData()`方法中定义：

```py
  def setData(self, index, value, role):
    if index.isValid() and role == qtc.Qt.EditRole:
      self._data[index.row()][index.column()] = value
      self.dataChanged.emit(index, index, [role])
      return True
    else:
      return False
```

`setData()`方法接受要设置的项目的索引、要设置的值和项目角色。此方法必须承担设置数据的任务，然后返回一个布尔值，指示数据是否成功更改。只有在索引有效且角色为`EditRole`时，我们才希望这样做。

如果数据发生变化，`setData()`也必须发出`dataChanged`信号。每当项目或一组项目与任何角色相关的更新时，都会发出此信号，因此携带了三个信息：被更改的最左上角的索引，被更改的最右下角的索引，以及每个索引的角色列表。在我们的情况下，我们只改变一个单元格，所以我们可以传递我们的索引作为单元格范围的两端，以及一个包含单个角色的列表。

`data()`方法还有一个小改变，虽然不是必需的，但会让用户更容易操作。回去编辑该方法如下：

```py
  def data(self, index, role):
    if role in (qtc.Qt.DisplayRole, qtc.Qt.EditRole):
      return self._data[index.row()][index.column()]
```

当选择表格单元格进行编辑时，将使用`EditRole`角色调用`data()`。在这个改变之前，当使用该角色调用`data()`时，`data()`会返回`None`，结果，单元格中的数据将在选择单元格时消失。通过返回`EditRole`的数据，用户将可以访问现有数据进行编辑。

我们现在已经实现了对现有单元格的编辑，但为了使我们的模型完全可编辑，我们需要实现插入和删除行。我们可以通过重写另外两个方法来实现这一点：`insertRows()`和`removeRows()`。

`insertRows()`方法如下：

```py
  def insertRows(self, position, rows, parent):
    self.beginInsertRows(
      parent or qtc.QModelIndex(),
      position,
      position + rows - 1
    )
    for i in range(rows):
      default_row = [''] * len(self._headers)
      self._data.insert(position, default_row)
    self.endInsertRows()
```

该方法接受插入开始的*位置*，要插入的*行数*以及父节点索引（与分层数据一起使用）。

在该方法内部，我们必须在调用`beginInsertRows()`和`endInsertRows()`之间放置我们的逻辑。`beginInsertRows()`方法准备了底层对象进行修改，并需要三个参数：

+   父节点的`ModelIndex`对象，对于表格数据来说是一个空的`QModelIndex`

+   行插入将开始的位置

+   行插入将结束的位置

我们可以根据传入方法的起始位置和行数来计算所有这些。一旦我们处理了这个问题，我们就可以生成一些行（以空字符串列表的形式，长度与我们的标题列表相同），并将它们插入到`self._data`中的适当索引位置。

在插入行后，我们调用`endInsertRows()`，它不带任何参数。

`removeRows()`方法非常相似：

```py
  def removeRows(self, position, rows, parent):
    self.beginRemoveRows(
      parent or qtc.QModelIndex(),
      position,
      position + rows - 1
    )
    for i in range(rows):
      del(self._data[position])
    self.endRemoveRows()
```

再次，我们需要在编辑数据之前调用`beginRemoveRows()`，在编辑后调用`endRemoveRows()`，就像我们对插入一样。如果我们想允许编辑列结构，我们可以重写`insertColumns()`和`removeColumns()`方法，它们的工作方式与行方法基本相同。现在，我们只会坚持行编辑。

到目前为止，我们的模型是完全可编辑的，但我们将添加一个方法，以便将数据刷新到磁盘，如下所示：

```py
  def save_data(self):
    with open(self.filename, 'w', encoding='utf-8') as fh:
      writer = csv.writer(fh)
      writer.writerow(self._headers)
      writer.writerows(self._data)
```

这个方法只是打开我们的文件，并使用 Python 的`csv`库写入标题和所有数据行。

# 在视图中使用模型

现在我们的模型已经准备好使用了，让我们充实应用程序的其余部分，以演示如何使用它。

首先，我们需要创建一个`QTableView`小部件，并将其添加到我们的`MainWindow`中：

```py
    # in MainWindow.__init__()
    self.tableview = qtw.QTableView()
    self.tableview.setSortingEnabled(True)
    self.setCentralWidget(self.tableview)
```

如您所见，我们不需要做太多工作来使`QTableView`小部件与模型一起工作。因为我们在模型中实现了`sort()`，我们将启用排序，但除此之外，它不需要太多配置。

当然，要查看任何数据，我们需要将模型分配给视图；为了创建一个模型，我们需要一个文件。让我们创建一个回调来获取一个：

```py
  def select_file(self):
    filename, _ = qtw.QFileDialog.getOpenFileName(
      self,
      'Select a CSV file to open…',
      qtc.QDir.homePath(),
      'CSV Files (*.csv) ;; All Files (*)'
    )
    if filename:
      self.model = CsvTableModel(filename)
      self.tableview.setModel(self.model)
```

我们的方法使用`QFileDialog`类来询问用户要打开的 CSV 文件。如果选择了一个文件，它将使用 CSV 文件来创建我们模型类的一个实例。然后使用`setModel()`访问方法将模型类分配给视图。

回到`MainWindow.__init__()`，让我们为应用程序创建一个主菜单，并添加一个“打开”操作：

```py
    menu = self.menuBar()
    file_menu = menu.addMenu('File')
    file_menu.addAction('Open', self.select_file)
```

如果您现在运行脚本，您应该能够通过转到“文件|打开”并选择有效的 CSV 文件来打开文件。您应该能够查看甚至编辑数据，并且如果单击标题单元格，数据应该按列排序。

接下来，让我们添加用户界面组件，以便保存我们的文件。首先，创建一个调用`MainWindow`方法`save_file()`的菜单项：

```py
    file_menu.addAction('Save', self.save_file)
```

现在，让我们创建我们的`save_file()`方法来实际保存文件：

```py
  def save_file(self):
    if self.model:
      self.model.save_data()
```

要保存文件，我们实际上只需要调用模型的`save_data()`方法。但是，我们不能直接将菜单项连接到该方法，因为在实际加载文件之前模型不存在。这个包装方法允许我们创建一个没有模型的菜单选项。

我们想要连接的最后一个功能是能够插入和删除行。在电子表格中，能够在所选行的上方或下方插入行通常是有用的。因此，让我们在`MainWindow`中创建回调来实现这一点：

```py
  def insert_above(self):
    selected = self.tableview.selectedIndexes()
    row = selected[0].row() if selected else 0
    self.model.insertRows(row, 1, None)

  def insert_below(self):
    selected = self.tableview.selectedIndexes()
    row = selected[-1].row() if selected else self.model.rowCount(None)
    self.model.insertRows(row + 1, 1, None)
```

在这两种方法中，我们通过调用表视图的`selectedIndexes()`方法来获取所选单元格的列表。这些列表从左上角的单元格到右下角的单元格排序。因此，对于插入上方，我们检索列表中第一个索引的行（如果列表为空，则为 0）。对于插入下方，我们检索列表中最后一个索引的行（如果列表为空，则为表中的最后一个索引）。最后，在这两种方法中，我们使用模型的`insertRows()`方法将一行插入到适当的位置。

删除行类似，如下所示：

```py
  def remove_rows(self):
    selected = self.tableview.selectedIndexes()
    if selected:
      self.model.removeRows(selected[0].row(), len(selected), None)
```

这次我们只在有活动选择时才采取行动，并使用模型的`removeRows()`方法来删除第一个选定的行。

为了使这些回调对用户可用，让我们在`MainWindow`中添加一个“编辑”菜单：

```py
    edit_menu = menu.addMenu('Edit')
    edit_menu.addAction('Insert Above', self.insert_above)
    edit_menu.addAction('Insert Below', self.insert_below)
    edit_menu.addAction('Remove Row(s)', self.remove_rows)
```

此时，请尝试加载 CSV 文件。您应该能够在表中插入和删除行，编辑字段并保存结果。恭喜，您已经创建了一个 CSV 编辑器！

# 总结

在本章中，您学习了模型视图编程。您学习了如何在常规小部件中使用模型，以及如何在 Qt 中使用特殊的模型视图类。您创建了一个自定义表模型，并通过利用模型视图类的功能快速构建了一个 CSV 编辑器。

我们将学习更高级的模型视图概念，包括委托和数据映射在第九章中，*使用 QtSQL 探索 SQL*。

在下一章中，您将学习如何为您的 PyQt 应用程序设置样式。我们将使用图像、动态图标、花哨的字体和颜色来装扮我们的单调表单，并学习控制 Qt GUI 整体外观和感觉的多种方法。

# 问题

尝试这些问题来测试您从本章中学到的知识：

1.  假设我们有一个设计良好的模型视图应用程序，以下代码是模型还是视图的一部分？

```py
  def save_as(self):
    filename, _ = qtw.QFileDialog(self)
    self.data.save_file(filename)
```

1.  您能否至少说出模型不应该做的两件事和视图不应该做的两件事？

1.  `QAbstractTableModel`和`QAbstractTreeModel`都在名称中有*Abstract*。在这种情况下，*Abstract*在这里是什么意思？在 C++中，它的意思是否与 Python 中的意思不同？

1.  哪种模型类型——列表、表格或树——最适合以下数据集：

+   用户最近的文件

+   Windows 注册表

+   Linux `syslog`记录

+   博客文章

+   个人称谓（例如，先生，夫人或博士）

+   分布式版本控制历史

1.  为什么以下代码失败了？

```py
  class DataModel(QAbstractTreeModel):
    def rowCount(self, node):
      if node > 2:
        return 1
      else:
        return len(self._data[node])
```

1.  当插入列时，您的表模型工作不正常。您的`insertColumns()`方法有什么问题？

```py
    def insertColumns(self, col, count, parent):
      for row in self._data:
        for i in range(count):
          row.insert(col, '')
```

1.  当悬停时，您希望您的视图显示项目数据作为工具提示。您将如何实现这一点？

# 进一步阅读

您可能希望查看以下资源：

+   有关模型视图编程的 Qt 文档在[`doc.qt.io/qt-5/model-view-programming.html`](https://doc.qt.io/qt-5/model-view-programming.html)

+   马丁·福勒在[`martinfowler.com/eaaDev/uiArchs.html`](https://martinfowler.com/eaaDev/uiArchs.html)上介绍了**模型-视图-控制器**（**MVC**）及相关模式的概述。
