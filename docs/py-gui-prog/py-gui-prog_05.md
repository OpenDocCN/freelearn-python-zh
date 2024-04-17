# 第五章：规划我们应用程序的扩展

这个应用程序真的很受欢迎！经过一些初步测试和定位，数据录入人员现在已经使用您的新表单几个星期了。错误和数据输入时间的减少是显著的，人们对这个程序可能解决的其他问题充满了兴奋的讨论。即使主管也加入了头脑风暴，你强烈怀疑你很快就会被要求添加一些新功能。然而，有一个问题；这个应用程序已经是几百行的脚本了，你担心随着它的增长，它的可管理性。你需要花一些时间来组织你的代码库，为未来的扩展做准备。

在本章中，我们将学习以下主题：

+   如何使用**模型-视图-控制器**模式来分离应用程序的关注点

+   如何将代码组织成 Python 包

+   为您的包结构创建基本文件和目录

+   如何使用 Git 版本控制系统跟踪您的更改

# 分离关注点

适当的建筑设计对于任何需要扩展的项目都是至关重要的。任何人都可以支撑起一些支柱，建造一个花园棚屋，但是建造一座房子或摩天大楼需要仔细的规划和工程。软件也是一样的；简单的脚本可以通过一些快捷方式，比如全局变量或直接操作类属性来解决，但随着程序的增长，我们的代码需要以一种限制我们需要在任何给定时刻理解的复杂度的方式来隔离和封装不同的功能。

我们称之为**关注点的分离**，通过使用描述不同应用程序组件及其交互方式的架构模式来实现。

# MVC 模式

这些模式中最持久的可能是 MVC 模式，它是在 20 世纪 70 年代引入的。尽管这种模式多年来已经发展并衍生出各种变体，但基本的要点仍然是：将数据、数据的呈现和应用程序逻辑保持在独立的组件中。

让我们更深入地了解这些组件，并在我们的应用程序的上下文中理解它们。

# 什么是模型？

MVC 中的**模型**代表数据。这包括数据的存储，以及数据可以被查询或操作的各种方式。理想情况下，模型不关心或受到数据如何呈现或授予什么 UI 控件的影响，而是提供一个高级接口，只在最小程度上关注其他组件的内部工作。理论上，如果您决定完全更改程序的 UI（比如，从 Tkinter 应用程序到 Web 应用程序），模型应该完全不受影响。

模型中包含的功能或信息的一些示例包括以下内容：

+   准备并将程序数据写入持久介质（数据文件、数据库等）

+   从文件或数据库中检索数据并将其转换为程序有用的格式

+   一组数据中字段的权威列表，以及它们的数据类型和限制

+   根据定义的数据类型和限制验证数据

+   对存储的数据进行计算

我们的应用程序目前没有模型类；数据布局是在表单类中定义的，到目前为止，`Application.on_save()`方法是唯一关心数据持久性的代码。我们需要将这个逻辑拆分成一个单独的对象，该对象将定义数据布局并处理所有 CSV 操作。

# 什么是视图？

**视图**是向用户呈现数据和控件的接口。应用程序可能有许多视图，通常是在相同的数据上。视图不直接与模型交互，并且理想情况下只包含足够的逻辑来呈现 UI 并将用户操作传递回控制器。

在视图中找到的一些代码示例包括以下内容：

+   GUI 布局和小部件定义

+   表单自动化，例如字段的自动完成，小部件的动态切换，或错误对话框的显示

+   原始数据的格式化呈现

我们的`DataRecordForm`类是我们的主视图：它包含了我们应用程序用户界面的大部分代码。它还当前定义了我们数据记录的结构。这个逻辑可以留在视图中，因为视图确实需要一种在将数据临时传递给模型之前存储数据的方式，但从现在开始它不会再定义我们的数据记录。

随着我们继续前进，我们将向我们的应用程序添加更多视图。

# 什么是控制器？

**控制器**是应用程序的大中央车站。它处理用户的请求，并负责在视图和模型之间路由数据。MVC 的大多数变体都会改变控制器的角色（有时甚至是名称），但重要的是它充当视图和模型之间的中介。我们的控制器对象将需要保存应用程序使用的视图和模型的引用，并负责管理它们之间的交互。

在控制器中找到的代码示例包括以下内容：

+   应用程序的启动和关闭逻辑

+   用户界面事件的回调

+   模型和视图实例的创建

我们的`Application`对象目前充当着应用程序的控制器，尽管它也包含一些视图和模型逻辑。随着应用程序的发展，我们将把更多的展示逻辑移到视图中，将更多的数据逻辑移到模型中，留下的主要是连接代码在我们的`Application`对象中。

# 为什么要复杂化我们的设计？

最初，以这种方式拆分应用程序似乎会增加很多不必要的开销。我们将不得不在不同对象之间传输数据，并最终编写更多的代码来完成完全相同的事情。为什么我们要这样做呢？

简而言之，我们这样做是为了使扩展可管理。随着应用程序的增长，复杂性也会增加。将我们的组件相互隔离限制了任何一个组件需要管理的复杂性的数量；例如，当我们重新构造表单视图的布局时，我们不应该担心模型将如何在输出文件中结构化数据。程序的这两个方面应该彼此独立。

这也有助于我们在放置某些类型的逻辑时保持一致。例如，拥有一个独立的模型对象有助于我们避免在 UI 代码中散布临时数据查询或文件访问尝试。

最重要的是，如果没有一些指导性的架构策略，我们的程序很可能会变成一团无法解开的逻辑混乱。即使不遵循严格的 MVC 设计定义，始终遵循松散的 MVC 模式也会在应用程序变得更加复杂时节省很多麻烦。

# 构建我们的应用程序目录结构

将程序逻辑上分解为单独的关注点有助于我们管理每个组件的逻辑复杂性，将代码物理上分解为多个文件有助于我们保持每个文件的复杂性可管理。这也加强了组件之间的隔离；例如，您不能共享全局变量，如果您的模型文件导入了`tkinter`，那么您就知道您做错了什么。

# 基本目录结构

Python 应用程序目录布局没有官方标准，但有一些常见的约定可以帮助我们保持整洁，并且以后更容易打包我们的软件。让我们按照以下方式设置我们的目录结构：

1.  首先，创建一个名为`ABQ_Data_Entry`的目录。这是我们应用程序的**根目录**，所以每当我们提到**应用程序根目录**时，就是它。

1.  在应用程序根目录下，创建另一个名为`abq_data_entry`的目录。注意它是小写的。这将是一个 Python 包，其中将包含应用程序的所有代码；它应该始终被赋予一个相当独特的名称，以免与现有的 Python 包混淆。通常情况下，应用程序根目录和主模块之间不会有不同的大小写，但这也不会有任何问题；我们在这里这样做是为了避免混淆。

Python 模块的命名应始终使用全部小写的名称和下划线。这个约定在 PEP 8 中有详细说明，PEP 8 是 Python 的官方风格指南。有关 PEP 8 的更多信息，请参见[`www.python.org/dev/peps/pep-0008`](https://www.python.org/dev/peps/pep-0008)。

1.  接下来，在应用程序根目录下创建一个名为`docs`的文件夹。这个文件夹将用于存放关于应用程序的文档文件。

1.  最后，在应用程序根目录中创建两个空文件：`README.rst`和`abq_data_entry.py`。你的目录结构应该如下所示：

![](img/830b3415-2492-4ad3-a86e-1e17b65c7a9b.png)

# abq_data_entry.py 文件

就像以前一样，`abq_data_entry.py`是执行程序的主文件。不过，与以前不同的是，它不会包含大部分的程序。实际上，这个文件应该尽可能地简化。

打开文件并输入以下代码：

```py
from abq_data_entry.application import Application

app = Application()
app.mainloop()
```

保存并关闭文件。这个文件的唯一目的是导入我们的`Application`类，创建一个实例，并运行它。其余的工作将在`abq_data_entry`包内进行。我们还没有创建这个包，所以这个文件暂时无法运行；在我们处理文档之前，让我们先处理一下文档。

# README.rst 文件

自上世纪 70 年代以来，程序一直包含一个名为`README`的简短文本文件，其中包含程序文档的简要摘要。对于小型程序，它可能是唯一的文档；对于大型程序，它通常包含用户或管理员的基本预先飞行指令。

`README`文件没有规定的内容集，但作为基本指南，考虑以下部分：

+   **描述**：程序及其功能的简要描述。我们可以重用规格说明中的描述，或类似的描述。这可能还包含主要功能的简要列表。

+   **作者信息**：作者的姓名和版权日期。如果你计划分享你的软件，这一点尤为重要，但即使对于公司内部的软件，让未来的维护者知道谁创建了软件以及何时创建也是有用的。

+   **要求**：软件和硬件要求的列表，如果有的话。

+   **安装**：安装软件、先决条件、依赖项和基本设置的说明。

+   **配置**：如何配置应用程序以及有哪些选项可用。这通常针对命令行或配置文件选项，而不是在程序中交互设置的选项。

+   **用法**：启动应用程序的描述，命令行参数和用户需要了解的其他注意事项。

+   **一般注意事项**：用户应该知道的注意事项或关键信息。

+   **错误**：应用程序中已知的错误或限制的列表。

并不是所有这些部分都适用于每个程序；例如，ABQ 数据输入目前没有任何配置选项，所以没有理由有一个配置部分。根据情况，你可能会添加其他部分；例如，公开分发的软件可能会有一个常见问题解答部分，或者开源软件可能会有一个包含如何提交补丁的贡献部分。

`README`文件以纯 ASCII 或 Unicode 文本编写，可以是自由格式的，也可以使用标记语言。由于我们正在进行一个 Python 项目，我们将使用 reStructuredText，这是 Python 文档的官方标记语言（这就是为什么我们的文件使用`rst`文件扩展名）。

# ReStructuredText

reStructuredText 标记语言是 Python `docutils`项目的一部分，完整的参考资料可以在 Docutils 网站找到：[`docutils.sourceforge.net`](http://docutils.sourceforge.net)。`docutils`项目还提供了将 RST 转换为 PDF、ODT、HTML 和 LaTeX 等格式的实用程序。

基础知识可以很快掌握，所以让我们来看看它们：

+   段落是通过在文本块之间留下一个空行来创建的。

+   标题通过用非字母数字符号下划线单行文本来创建。确切的符号并不重要；你首先使用的符号将被视为文档其余部分的一级标题，你其次使用的符号将被视为二级标题，依此类推。按照惯例，`=`通常用于一级，`-`用于二级，`~`用于三级，`+`用于四级。

+   标题和副标题的创建方式与标题相似，只是在上下都有一行符号。

+   项目列表是通过在行首加上`*`、`-`或`+`和一个空格来创建的。切换符号将创建子列表，多行点由将后续行缩进到文本从第一个项目符号开始的位置来创建。

+   编号列表的创建方式与项目列表相似，但使用数字（不需要正确排序）或`#`符号作为项目符号。

+   代码示例可以通过用双反引号字符括起来来指定内联(`` ` ``)，或者在一个代码块中，用双冒号结束一个引入行，并缩进代码块。
+   表格可以通过用 `=` 符号包围文本列，并用空格分隔表示列断点，或者通过使用 `|`、`-` 和 `+` 构建 ASCII 表格来创建。在纯文本编辑器中创建表格可能会很繁琐，但一些编程工具有插件可以生成 RST 表格。

我们已经在第二章中使用了 RST，*用 Tkinter 设计 GUI 应用程序*，来创建我们的程序规范；在那里，您看到了标题、头部、项目符号和表格的使用。让我们逐步创建我们的 `README.rst` 文件：

1.  打开文件并以以下方式开始标题和描述：

```py
============================
 ABQ Data Entry Application
============================

Description
===========

This program provides a data entry form for ABQ Agrilabs laboratory data.

Features
--------

* Provides a validated entry form to ensure correct data
* Stores data to ABQ-format CSV files
* Auto-fills form fields whenever possible

```

1.  接下来，我们将通过添加以下代码来列出作者：

```py
Authors
=======

Alan D Moore, 2018

```

当然要添加自己。最终，其他人可能会在您的应用程序上工作；他们应该在这里加上他们的名字以及他们工作的日期。现在，添加以下要求：

```py

Requirements
============

* Python 3
* Tkinter

```

目前，我们只需要 Python 3 和 Tkinter，但随着我们的应用程序的增长，我们可能会扩展这个列表。我们的应用程序实际上不需要被安装，并且没有配置选项，所以现在我们可以跳过这些部分。相反，我们将跳到 `使用方法` 如下：

```py

Usage
=====

To start the application, run::

  python3 ABQ_Data_Entry/abq_data_entry.py

```

除了这个命令之外，关于运行程序没有太多需要了解的东西；没有命令行开关或参数。我们不知道任何错误，所以我们将在末尾留下一些一般的说明，如下所示：

```py
General Notes
=============

The CSV file will be saved to your current directory in the format "abq_data_record_CURRENTDATE.csv", where CURRENTDATE is today's date in ISO format.

This program only appends to the CSV file.  You should have a spreadsheet program installed in case you need to edit or check the file.


```

现在告诉用户文件将被保存在哪里以及它将被命名为什么，因为这是硬编码到程序中的。此外，我们应该提到用户应该有某种电子表格，因为程序无法编辑或查看数据。这就完成了 `README.rst` 文件。保存它，然后我们继续到 `docs` 文件夹。

# 填充文档文件夹

`docs` 文件夹是用于存放文档的地方。这可以是任何类型的文档：用户手册、程序规范、API 参考、图表等等。

现在，您可以复制我们在前几章中编写的程序规范、您的界面模型和技术人员使用的表单的副本。

在某个时候，您可能需要编写一个用户手册，但是现在程序足够简单，不需要它。

# 制作一个 Python 包

创建自己的 Python 包其实非常简单。一个 Python 包由以下三个部分组成：

+   一个目录

+   那个目录中的一个或多个 Python 文件

+   目录中的一个名为 `__init__.py` 的文件

一旦完成这一步，您可以整体或部分地导入您的包，就像导入标准库包一样，只要您的脚本与包目录在同一个父目录中。

注意，模块中的 `__init__.py` 有点类似于类中的 `self.__init__()`。其中的代码将在包被导入时运行。Python 社区一般不鼓励在这个文件中放置太多代码，而且由于实际上不需要任何代码，我们将保持此文件为空。

让我们开始构建我们应用程序的包。在`abq_data_entry`下创建以下六个空文件：

+   `__init__.py`

+   `widgets.py`

+   `views.py`

+   `models.py`

+   `application.py`

+   `constants.py`

这些 Python 文件中的每一个都被称为一个**模块**。模块只是一个包目录中的 Python 文件。您的目录结构现在应该是这样的：

![](img/06efc903-784c-426e-be9b-ddeb66de7849.png)

此时，您已经有了一个工作的包，尽管里面没有实际的代码。要测试这个，请打开一个终端/命令行窗口，切换到您的`ABQ_Data_Entry`目录，并启动一个 Python shell。

现在，输入以下命令：

```py

from abq_data_entry import application

```

这应该可以正常工作。当然，它什么也不做，但我们接下来会解决这个问题。

不要将此处的“包”一词与实际的可分发的 Python 包混淆，比如使用`pip`下载的那些。

# 将我们的应用程序拆分成多个文件

现在我们的目录结构已经就绪，我们需要开始解剖我们的应用程序脚本，并将其分割成我们的模块文件。我们还需要创建我们的模型类。打开您从第四章*减少用户错误：验证和自动化*中的`abq_data_entry.py`文件，让我们开始吧！

# 创建模型模块

当您的应用程序完全关注数据时，最好从模型开始。记住，模型的工作是管理我们应用程序数据的存储、检索和处理，通常是关于其持久存储格式的（在本例中是 CSV）。为了实现这一点，我们的模型应该包含关于我们数据的所有知识。

目前，我们的应用程序没有类似模型的东西；关于应用程序数据的知识散布在表单字段中，而`Application`对象只是在请求保存操作时获取表单包含的任何数据，并直接将其塞入 CSV 文件中。由于我们还没有检索或更新信息，所以我们的应用程序对 CSV 文件中的内容一无所知。

为了将我们的应用程序转移到 MVC 架构，我们需要创建一个模型类，它既管理数据存储和检索，又代表我们数据的权威来源。换句话说，我们必须在这里编码我们数据字典中包含的知识。我们真的不知道我们将如何使用这些知识，但它们应该在这里。

我们可以以几种方式存储这些数据，例如创建一个自定义字段类或一个`namedtuple`对象，但现在我们将保持简单，只使用一个字典，将字段名称映射到字段元数据。

字段元数据将同样被存储为关于字段的属性字典，其中将包括：

+   字段是否必填

+   字段中存储的数据类型

+   可能值的列表（如果适用）

+   值的最小、最大和增量（如果适用）

要为每个字段存储数据类型，让我们定义一些数据类型。打开`constants.py`文件并添加以下代码：

```py

class FieldTypes:
    string = 1
    string_list = 2
    iso_date_string = 3
    long_string = 4
    decimal = 5
    integer = 6
    boolean = 7

```

我们创建了一个名为`FieldTypes`的类，它简单地存储一些命名的整数值，这些值将描述我们将要存储的不同类型的数据。我们可以在这里只使用 Python 类型，但是区分一些可能是相同 Python 类型的数据类型是有用的（例如`long`、`short`和`date`字符串）。请注意，这里的整数值基本上是无意义的；它们只需要彼此不同。

Python 3 有一个`Enum`类，我们可以在这里使用它，但在这种情况下它添加的功能非常少。如果您正在创建大量常量，比如我们的`FieldTypes`类，并且需要额外的功能，可以研究一下这个类。

现在打开`models.py`，我们将导入`FieldTypes`并创建我们的模型类和字段定义如下：

```py

import csv
import os
from .constants import FieldTypes as FT

class CSVModel:
    """CSV file storage"""
    fields = {
        "Date": {'req': True, 'type': FT.iso_date_string},
        "Time": {'req': True, 'type': FT.string_list,
                 'values': ['8:00', '12:00', '16:00', '20:00']},
        "Technician": {'req': True, 'type':  FT.string},
        "Lab": {'req': True, 'type': FT.string_list,
                'values': ['A', 'B', 'C', 'D', 'E']},
        "Plot": {'req': True, 'type': FT.string_list,
                 'values': [str(x) for x in range(1, 21)]},
        "Seed sample":  {'req': True, 'type': FT.string},
        "Humidity": {'req': True, 'type': FT.decimal,
                     'min': 0.5, 'max': 52.0, 'inc': .01},
        "Light": {'req': True, 'type': FT.decimal,
                  'min': 0, 'max': 100.0, 'inc': .01},
        "Temperature": {'req': True, 'type': FT.decimal,
                        'min': 4, 'max': 40, 'inc': .01},
        "Equipment Fault": {'req': False, 'type': FT.boolean},
        "Plants": {'req': True, 'type': FT.integer,
                   'min': 0, 'max': 20},
        "Blossoms": {'req': True, 'type': FT.integer,
                     'min': 0, 'max': 1000},
        "Fruit": {'req': True, 'type': FT.integer,
                  'min': 0, 'max': 1000},
        "Min Height": {'req': True, 'type': FT.decimal,
                       'min': 0, 'max': 1000, 'inc': .01},
        "Max Height": {'req': True, 'type': FT.decimal,
                       'min': 0, 'max': 1000, 'inc': .01},
        "Median Height": {'req': True, 'type': FT.decimal,
                          'min': 0, 'max': 1000, 'inc': .01},
        "Notes": {'req': False, 'type': FT.long_string}
    }
```

注意我们导入`FieldTypes`的方式：`from .constants import FieldTypes`。点号在`constants`前面使其成为**相对导入**。相对导入可在 Python 包内部用于定位同一包中的其他模块。在这种情况下，我们位于`models`模块中，需要访问`abq_data_entry`包内的`constants`模块。单个点号表示我们当前的父模块（`abq_data_entry`），因此`.constants`表示`abq_data_entry`包的`constants`模块。

相对导入还可以区分我们的自定义模块与`PYTHONPATH`中的模块。因此，我们不必担心任何第三方或标准库包与我们的模块名称冲突。

除了字段属性之外，我们还在这里记录字段的顺序。在 Python 3.6 及更高版本中，字典会保留它们定义的顺序；如果您使用的是较旧版本的 Python 3，则需要使用`collections`标准库模块中的`OrderedDict`类来保留字段顺序。

现在我们有了一个了解哪些字段需要存储的类，我们需要将保存逻辑从应用程序类迁移到模型中。

我们当前脚本中的代码如下：

```py

datestring = datetime.today().strftime("%Y-%m-%d")
filename = "abq_data_record_{}.csv".format(datestring)
newfile = not os.path.exists(filename)

data = self.recordform.get()

with open(filename, 'a') as fh:
    csvwriter = csv.DictWriter(fh, fieldnames=data.keys())
    if newfile:
        csvwriter.writeheader()
    csvwriter.writerow(data)
```

让我们通过这段代码确定什么属于模型，什么属于控制器（即`Application`类）：

+   前两行定义了我们要使用的文件名。这可以放在模型中，但是提前思考，似乎用户可能希望能够打开任意文件或手动定义文件名。这意味着应用程序需要能够告诉模型要使用哪个文件名，因此最好将确定名称的逻辑留在控制器中。

+   `newfile`行确定文件是否存在。作为数据存储介质的实现细节，这显然是模型的问题，而不是应用程序的问题。

+   `data = self.recordform.get()`从表单中提取数据。由于我们的模型不知道表单的存在，这需要留在控制器中。

+   最后一块打开文件，创建一个`csv.DictWriter`对象，并追加数据。这明显是模型的关注点。

现在，让我们开始将代码移入`CSVModel`类：

1.  要开始这个过程，让我们为`CSVModel`创建一个允许我们传入文件名的构造函数：

```py
    def __init__(self, filename):
        self.filename = filename
```

构造函数非常简单；它只接受一个`filename`参数并将其存储为一个属性。现在，我们将迁移保存逻辑如下：

```py

    def save_record(self, data):
        """Save a dict of data to the CSV file"""

        newfile = not os.path.exists(self.filename)

        with open(self.filename, 'a') as fh:
            csvwriter = csv.DictWriter(fh, 
                fieldnames=self.fields.keys())
            if newfile:
                csvwriter.writeheader()
            csvwriter.writerow(data)
```

这本质上是我们选择从`Application.on_save()`中复制的逻辑，但有一个区别；在对`csv.DictWriter()`的调用中，`fieldnames` 参数由模型的`fields`列表而不是`data`字典的键定义。这允许我们的模型管理 CSV 文件本身的格式，并不依赖于表单提供的内容。

1.  在我们完成之前，我们需要处理我们的模块导入。`save_record()`方法使用`os`和`csv`库，所以我们需要导入它们。将此添加到文件顶部如下：

```py

import csv
import os

```

模型就位后，让我们开始处理我们的视图组件。

# 移动小部件

虽然我们可以将所有与 UI 相关的代码放在一个`views`文件中，但我们有很多小部件类，实际上应该将它们放在自己的文件中，以限制`views`文件的复杂性。

因此，我们将所有小部件类的代码移动到`widgets.py`文件中。小部件包括实现可重用 GUI 组件的所有类，包括`LabelInput`等复合小部件。随着我们开发更多的这些，我们将把它们添加到这个文件中。

打开`widgets.py`并复制`ValidatedMixin`、`DateInput`、`RequiredEntry`、`ValidatedCombobox`、`ValidatedSpinbox`和`LabelInput`的所有代码。这些是我们的小部件。

`widgets.py` 文件需要导入被复制代码使用的任何模块依赖项。我们需要查看我们的代码，并找出我们使用的库并将它们导入。显然，我们需要`tkinter`和`ttk`，所以在顶部添加它们如下：

```py
import tkinter as tk
from tkinter import ttk
```

我们的`DateInput` 类使用`datetime`库中的`datetime`类，因此也要导入它，如下所示：

```py

from datetime import datetime

```

最后，我们的`ValidatedSpinbox` 类使用`decimal`库中的`Decimal`类和`InvalidOperation`异常，如下所示：

```py

from decimal import Decimal, InvalidOperation

```

这是现在我们在`widgets.py`中需要的全部，但是当我们重构我们的视图逻辑时，我们会再次访问这个文件。

# 移动视图

接下来，我们需要创建`views.py`文件。视图是较大的 GUI 组件，如我们的`DataRecordForm`类。目前它是我们唯一的视图，但我们将在后面的章节中创建更多的视图，并将它们添加到这里。

打开`views.py`文件，复制`DataRecordForm`类，然后返回顶部处理模块导入。同样，我们需要`tkinter`和`ttk`，我们的文件保存逻辑依赖于`datetime`以获得文件名。

将它们添加到文件顶部如下：

```py

import tkinter as tk
from tkinter import ttk
from datetime import datetime

```

不过，我们还没有完成；我们实际的小部件还没有，我们需要导入它们。由于我们将在文件之间进行大量对象导入，让我们暂停一下，考虑一下处理这些导入的最佳方法。

我们可以导入对象的三种方式：

+   使用通配符导入从`widgets.py`中导入所有类

+   使用`from ... import ...`格式明确地从`widgets.py`中导入所有所需的类

+   导入`widgets`并将我们的小部件保留在它们自己的命名空间中

让我们考虑一下这些方法的相对优点：

+   第一个选项是迄今为止最简单的，但随着应用程序的扩展，它可能会给我们带来麻烦。通配符导入将会导入模块内在全局范围内定义的每个名称。这不仅包括我们定义的类，还包括任何导入的模块、别名和定义的变量或函数。随着应用程序在复杂性上的扩展，这可能会导致意想不到的后果和微妙的错误。

+   第二个选项更清晰，但意味着我们将需要维护导入列表，因为我们添加新类并在不同文件中使用它们，这导致了一个长而丑陋的导入部分，难以让人理解。

+   第三种选项是目前为止最好的，因为它将所有名称保留在命名空间内，并保持代码优雅简单。唯一的缺点是我们需要更新我们的代码，以便所有对小部件类的引用都包含模块名称。为了避免这变得笨拙，让我们将`widgets`模块别名为一个简短的名字，比如`w`。

将以下代码添加到你的导入中：

```py

from . import widgets as w

```

现在，我们只需要遍历代码，并在所有`LabelInput`、`RequiredEntry`、`DateEntry`、`ValidatedCombobox`和`ValidatedSpinbox`的实例之前添加`w.`。这应该很容易在 IDLE 或任何其他文本编辑器中使用一系列搜索和替换操作来完成。

例如，表单的`line 1`如下所示：

```py

# line 1
self.inputs['Date'] = w.LabelInput(
    recordinfo, "Date",
    input_class=w.DateEntry,
    input_var=tk.StringVar()
)
self.inputs['Date'].grid(row=0, column=0)
self.inputs['Time'] = w.LabelInput(
    recordinfo, "Time",
    input_class=w.ValidatedCombobox,
    input_var=tk.StringVar(),
    input_args={"values": ["8:00", "12:00", "16:00", "20:00"]}
)
self.inputs['Time'].grid(row=0, column=1)
self.inputs['Technician'] = w.LabelInput(
    recordinfo, "Technician",
    input_class=w.RequiredEntry,
    input_var=tk.StringVar()
)
self.inputs['Technician'].grid(row=0, column=2)
```

在你到处更改之前，让我们停下来，花一点时间重构这段代码中的一些冗余。

# 在我们的视图逻辑中消除冗余

查看视图逻辑中的字段定义：它们包含了很多与我们的模型中的信息相同的信息。最小值、最大值、增量和可能值在这里和我们的模型代码中都有定义。甚至输入小部件的类型直接与存储的数据类型相关。理想情况下，这应该只在一个地方定义，而且那个地方应该是模型。如果我们因为某种原因需要更新模型，我们的表单将不同步。

我们需要做的是将字段规范从我们的模型传递到视图类，并让小部件的详细信息从该规范中定义。

由于我们的小部件实例是在`LabelInput`类内部定义的，我们将增强该类的功能，以自动从我们模型的字段规范格式中计算出`input`类和参数。打开`widgets.py`文件，并像在`model.py`中一样导入`FieldTypes`类。

现在，找到`LabelInput`类，并在`__init__()`方法之前添加以下代码：

```py

    field_types = {
        FT.string: (RequiredEntry, tk.StringVar),
        FT.string_list: (ValidatedCombobox, tk.StringVar),
        FT.iso_date_string: (DateEntry, tk.StringVar),
        FT.long_string: (tk.Text, lambda: None),
        FT.decimal: (ValidatedSpinbox, tk.DoubleVar),
        FT.integer: (ValidatedSpinbox, tk.IntVar),
        FT.boolean: (ttk.Checkbutton, tk.BooleanVar)
    }

```

这段代码充当了将我们模型的字段类型转换为适合字段类型的小部件类型和变量类型的关键。

现在，我们需要更新`__init__()`，接受一个`field_spec`参数，并在给定时使用它来定义输入小部件，如下所示：

```py

    def __init__(self, parent, label='', input_class=None,
         input_var=None, input_args=None, label_args=None,
         field_spec=None, **kwargs):
        super().__init__(parent, **kwargs)
        input_args = input_args or {}
        label_args = label_args or {}
        if field_spec:
            field_type = field_spec.get('type', FT.string)
            input_class = input_class or 
            self.field_types.get(field_type)[0]
            var_type = self.field_types.get(field_type)[1]
            self.variable = input_var if input_var else var_type()
            # min, max, increment
            if 'min' in field_spec and 'from_' not in input_args:
                input_args['from_'] = field_spec.get('min')
            if 'max' in field_spec and 'to' not in input_args:
                input_args['to'] = field_spec.get('max')
            if 'inc' in field_spec and 'increment' not in input_args:
                input_args['increment'] = field_spec.get('inc')
            # values
            if 'values' in field_spec and 'values' not in input_args:
                input_args['values'] = field_spec.get('values')
        else:
            self.variable = input_var
        if input_class in (ttk.Checkbutton, ttk.Button, ttk.Radiobutton):
            input_args["text"] = label
            input_args["variable"] = self.variable
        else:
            self.label = ttk.Label(self, text=label, **label_args)
            self.label.grid(row=0, column=0, sticky=(tk.W + tk.E))
            input_args["textvariable"] = self.variable
        # ... Remainder of __init__() is the same
```

让我们逐步解析这些更改：

1.  首先，我们将`field_spec`添加为一个关键字参数，并将`None`作为默认值。我们可能会在没有字段规范的情况下使用这个类，所以我们保持这个参数是可选的。

1.  如果给出了`field_spec`，我们将执行以下操作：

    +   我们将获取`type`值，并将其与我们类的字段键一起使用以获取`input_class`。如果我们想要覆盖这个值，显式传递的`input_class`将覆盖检测到的值。

    +   我们将以相同的方式确定适当的变量类型。再次，如果显式传递了`input_var`，我们将优先使用它，否则我们将使用从字段类型确定的那个。我们将以任何方式创建一个实例，并将其存储在`self.variable`中。

    +   对于`min`、`max`、`inc`和`values`，如果字段规范中存在键，并且相应的`from_`、`to`、`increment`或`values`参数没有显式传递进来，我们将使用`field_spec`值设置`input_args`变量。

1.  如果没有传入`field_spec`，我们需要将`self.variable`从`input_var`参数中赋值。

1.  现在我们使用`self.variable`而不是`input_var`来分配输入的变量，因为这些值可能不再是相同的，而`self.variable`将包含正确的引用。

现在，我们可以更新我们的视图代码以利用这种新的能力。我们的`DataRecordForm`类将需要访问模型的`fields`字典，然后可以使用它将字段规范发送到`LabelInput`类。

回到`views.py`文件，在方法签名中编辑，以便我们可以传入字段规范的字典：

```py

def __init__(self, parent, fields, *args, **kwargs):

```

有了对`fields`字典的访问权限，我们只需从中获取字段规范，并将其传递到`LabelInput`类中，而不是指定输入类、输入变量和输入参数。

现在，第一行看起来是这样的：

```py

        self.inputs['Date'] = w.LabelInput(
            recordinfo, "Date",
            field_spec=fields['Date'])
        self.inputs['Date'].grid(row=0, column=0)
        self.inputs['Time'] = w.LabelInput(
            recordinfo, "Time",
            field_spec=fields['Time'])
        self.inputs['Time'].grid(row=0, column=1)
        self.inputs['Technician'] = w.LabelInput(
            recordinfo, "Technician",
            field_spec=fields['Technician'])
        self.inputs['Technician'].grid(row=0, column=2)
```

继续以相同的方式更新其余的小部件，用`field_spec`替换`input_class`、`input_var`和`input_args`。请注意，当您到达高度字段时，您将需要保留定义`min_var`、`max_var`和`focus_update_var`的`input_args`部分。

例如，以下是`Min Height`输入的定义：

```py

        self.inputs['Min Height'] = w.LabelInput(
            plantinfo, "Min Height (cm)",
            field_spec=fields['Min Height'],
            input_args={"max_var": max_height_var,
                        "focus_update_var": min_height_var})
```

就这样。现在，我们对字段规范的任何更改都可以仅在模型中进行，并且表单将简单地执行正确的操作。

# 创建应用程序文件

最后，让我们按照以下步骤创建我们的控制器类`Application`：

1.  打开`application.py`文件，并将脚本中的`Application`类定义复制进去。

1.  首先，我们要修复的是我们的导入项。在文件顶部添加以下代码：

```py

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from . import views as v
from . import models as m
```

当然，我们需要`tkinter`和`ttk`，以及`datetime`来定义我们的文件名。虽然我们只需要从`views`和`models`中各自选择一个类，但我们还是要将它们保留在各自的命名空间中。随着应用程序的扩展，我们可能会有更多的视图，可能还会有更多的模型。

1.  我们需要更新在新命名空间中`__init__()`中对`DataRecordForm`的调用，并确保我们传递所需的字段规范字典，如下所示：

```py

self.recordform = v.DataRecordForm(self, m.CSVModel.fields)

```

1.  最后，我们需要更新`Application.on_save()`以使用模型，如下所示：

```py

    def on_save(self):
        """Handles save button clicks"""

        errors = self.recordform.get_errors()
        if errors:
            self.status.set(
                "Cannot save, error in fields: {}"
                .format(', '.join(errors.keys())))
            return False

        # For now, we save to a hardcoded filename 
        with a datestring.
        datestring = datetime.today().strftime("%Y-%m-%d")
        filename = "abq_data_record_{}.csv".format(datestring)
        model = m.CSVModel(filename)
        data = self.recordform.get()
        model.save_record(data)
        self.records_saved += 1
        self.status.set(
            "{} records saved this session".
            format(self.records_saved)
        )
        self.recordform.reset()
```

正如您所看到的，使用我们的模型非常简单；我们只需通过传递文件名创建了一个`CSVModel`类，然后将表单的数据传递给`save_record()`。

# 运行应用程序

应用程序现在完全迁移到了新的数据格式。要测试它，请导航到应用程序根文件夹`ABQ_Data_Entry`，然后执行以下命令：

```py

python3 abq_data_entry.py

```

它应该看起来和行为就像第四章中的单个脚本*通过验证和自动化减少用户错误*一样，并且在下面的截图中运行无错误：

![](img/4151fc4d-d11b-4bf1-a5a3-df5ab3971dca.png)

成功！

# 使用版本控制软件

我们的代码结构良好，可以扩展，但是还有一个非常关键的问题我们应该解决：**版本控制**。您可能已经熟悉了**版本控制系统**（**VCS**），有时也称为**修订控制**或**源代码管理**，但如果不了解，它是处理大型和不断变化的代码库的不可或缺的工具。

在开发应用程序时，我们有时会认为自己知道需要更改什么，但事实证明我们错了。有时我们不完全知道如何编写某些代码，需要多次尝试才能找到正确的方法。有时我们需要恢复到很久之前更改过的代码。有时我们有多个人在同一段代码上工作，需要将他们的更改合并在一起。版本控制系统就是为了解决这些问题以及更多其他问题而创建的。

有数十种不同的版本控制系统，但它们大多数本质上都是相同的：

+   您有一个可用于进行更改的代码副本

+   您定期选择要提交回主副本的更改

+   您可以随时查看代码的旧版本，然后恢复到主副本

+   您可以创建代码分支来尝试不同的方法、新功能或大型重构

+   您随后可以将这些分支合并回主副本

VCS 提供了一个安全网，让您可以自由更改代码，而无需担心您会彻底毁坏它：返回到已知的工作状态只需几个快速的命令即可。它还帮助我们记录代码的更改，并在机会出现时与他人合作。

有数十种 VC 系统可供选择，但迄今为止，远远最流行的是**Git**。

# 使用 Git 的超快速指南

Git 是由 Linus Torvalds 创建的，用于 Linux 内核项目的版本控制软件，并且已经发展成为世界上最流行的 VC 软件。它被源代码共享网站如 GitHub、Bitbucket、SourceForge 和 GitLab 使用。Git 非常强大，掌握它可能需要几个月或几年；幸运的是，基础知识可以在几分钟内掌握。

首先，您需要安装 Git；访问[`git-scm.com/downloads`](https://git-scm.com/downloads)获取有关如何在 macOS、Windows、Linux 或其他 Unix 操作系统上安装 Git 的说明。

# 初始化和配置 Git 仓库

安装完 Git 后，我们需要通过以下步骤初始化和配置我们的项目目录为一个 Git 仓库：

1.  在应用程序的根目录（`ABQ_Data_Entry`）中运行以下命令：

```py

git init

```

此命令在我们项目根目录下创建一个名为`.git`的隐藏目录，并使用构成仓库的基本文件对其进行初始化。`.git`目录将包含关于我们保存的修订的所有数据和元数据。

1.  在我们添加任何文件到仓库之前，我们需要告诉 Git 忽略某些类型的文件。例如，Python 在执行文件时会创建字节码（`.pyc`）文件，我们不希望将这些文件保存为我们代码的一部分。为此，请在您的项目根目录中创建一个名为`.gitignore`的文件，并在其中放入以下行：

```py

*.pyc
__pycache__/

```

# 添加和提交代码

现在我们的仓库已经初始化，我们可以使用以下命令向我们的 Git 仓库添加文件和目录：

```py

git add abq_data_entry
git add abq_data_entry.py
git add docs
git add README.rst

```

此时，我们的文件已经准备就绪，但尚未提交到仓库。您可以随时输入`git status`来检查仓库及其中的文件的状态。

你应该得到以下输出：

```py

On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)

    new file:   README.rst
    new file:   abq_data_entry.py
    new file:   abq_data_entry/__init__.py
    new file:   abq_data_entry/application.py
    new file:   abq_data_entry/models.py
    new file:   abq_data_entry/views.py
    new file:   abq_data_entry/widgets.py
    new file:   docs/Application_layout.png
    new file:   docs/abq_data_entry_spec.rst
    new file:   docs/lab-tech-paper-form.png

Untracked files:
  (use "git add <file>..." to include in what will be committed)

    .gitignore
```

这向您展示了`abq_data_entry`和`docs`下的所有文件以及您直接指定的文件都已经准备好提交到仓库中。

让我们继续提交更改，如下所示：

```py

git commit -m "Initial commit"

```

这里的`-m`标志传入了一个提交消息，该消息将与提交一起存储。每次向仓库提交代码时，您都需要编写一条消息。您应该尽可能使这些消息有意义，详细说明您所做的更改以及背后的原因。

# 查看和使用我们的提交

要查看仓库的历史记录，请运行以下`git log`命令：

```py

alanm@alanm-laptop:~/ABQ_Data_Entry$ git log
commit df48707422875ff545dc30f4395f82ad2d25f103 (HEAD -> master)
Author: Alan Moore <alan@example.com>
Date:   Thu Dec 21 18:12:17 2017 -0600

    Initial commit


```

正如您所看到的，我们上次提交的`作者`、`日期`和`提交`消息都显示出来。如果我们有更多的提交，它们也会在这里列出，从最新到最旧。您在输出的第一行中看到的长十六进制值是**提交哈希**，这是一个唯一的值，用于标识提交。这个值可以用来在其他操作中引用提交。

例如，我们可以使用它将我们的存储库重置到过去的状态，如下所示：

1.  删除`README.rst`文件，并验证它已完全消失。

1.  现在，输入命令`git reset --hard df48707`，将`df48707`替换为您提交哈希的前七个字符。

1.  再次检查您的文件清单：`README.rst`文件已经回来了。

这里发生的是我们改变了我们的存储库，然后告诉 Git 将存储库的状态硬重置到我们的第一个提交。如果您不想重置您的存储库，您也可以暂时检出一个旧的提交，或者使用特定的提交作为基础创建一个分支。正如您已经看到的，这为我们提供了一个强大的实验安全网；无论您如何调整代码，任何提交都只是一个命令的距离！

Git 有许多更多的功能超出了本书的范围。如果您想了解更多信息，Git 项目在[`git-scm.com/book`](https://git-scm.com/book)提供了免费的在线手册，您可以在那里了解分支和设置远程存储库等高级功能。目前，重要的是在进行更改时提交更改，以便保持您的安全网并记录更改的历史。

# 总结

在本章中，您学会了为您的简单脚本做一些严肃的扩展准备。您学会了如何将应用程序的职责领域划分为单独的组件，以及如何将代码分割成单独的模块。您学会了如何使用 reStructuredText 记录代码并使用版本控制跟踪所有更改。

在下一章中，我们将通过实现一些新功能来测试我们的新项目布局。您将学习如何使用 Tkinter 的应用程序菜单小部件，如何实现文件打开和保存，以及如何使用消息弹出窗口来警告用户或确认操作。